"""
╔══════════════════════════════════════════════════════════════╗
║         EXTRACTEUR GEMINI — Bank Statement Extractor         ║
║         Deux modes disponibles :                             ║
║           1. HYBRIDE  : pdfplumber texte + Gemini Text       ║
║           2. VISION   : PDF → Images → Gemini Vision         ║
║         Version : 3.1.0                                      ║
╚══════════════════════════════════════════════════════════════╝
"""

import google.generativeai as genai
import pdfplumber
import pandas as pd
import json
import re
import io
import time
import hashlib
import traceback
from PIL import Image

# ── Import optionnel pdf2image ──
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("⚠️  pdf2image non disponible — mode vision désactivé")


# ══════════════════════════════════════════════════════════════
# LOGGER DE DÉBOGAGE
# ══════════════════════════════════════════════════════════════

class DebugLogger:
    """
    Logger structuré pour tracer chaque étape de l'extraction.
    Stocke les logs en mémoire pour affichage dans Streamlit.
    """

    LEVELS = {
        "INFO":    "ℹ️",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR":   "❌",
        "DEBUG":   "🔍",
        "STEP":    "▶️",
        "DATA":    "📊",
        "API":     "🤖",
        "TOKEN":   "🔤",
    }

    def __init__(self, verbose: bool = True):
        self.verbose  = verbose
        self.logs     = []          # Historique complet
        self.errors   = []          # Erreurs uniquement
        self.warnings = []          # Avertissements uniquement
        self._step    = 0           # Compteur d'étapes

    # ── Méthodes de log ──────────────────────────────────────

    def info(self, msg: str, detail: str = ""):
        self._log("INFO", msg, detail)

    def success(self, msg: str, detail: str = ""):
        self._log("SUCCESS", msg, detail)

    def warning(self, msg: str, detail: str = ""):
        self._log("WARNING", msg, detail)
        self.warnings.append(msg)

    def error(self, msg: str, detail: str = "", exc: Exception = None):
        full_detail = detail
        if exc:
            full_detail += f"\n{type(exc).__name__}: {str(exc)}"
            full_detail += f"\n{traceback.format_exc()}"
        self._log("ERROR", msg, full_detail)
        self.errors.append({"msg": msg, "detail": full_detail})

    def debug(self, msg: str, detail: str = ""):
        self._log("DEBUG", msg, detail)

    def step(self, msg: str):
        self._step += 1
        self._log("STEP", f"[Étape {self._step}] {msg}", "")

    def data(self, label: str, value):
        """Log une valeur de données (tronquée si trop longue)."""
        val_str = str(value)
        if len(val_str) > 500:
            val_str = val_str[:500] + "... [tronqué]"
        self._log("DATA", label, val_str)

    def api(self, msg: str, detail: str = ""):
        self._log("API", msg, detail)

    def token(self, msg: str, detail: str = ""):
        self._log("TOKEN", msg, detail)

    def separator(self, label: str = ""):
        """Séparateur visuel dans les logs."""
        sep = "─" * 50
        entry = {
            "level":     "INFO",
            "icon":      "─",
            "message":   f"{sep} {label} {sep}" if label else sep,
            "detail":    "",
            "timestamp": time.strftime("%H:%M:%S"),
        }
        self.logs.append(entry)
        if self.verbose:
            print(f"\n{'─'*20} {label} {'─'*20}")

    # ── Méthode interne ──────────────────────────────────────

    def _log(self, level: str, msg: str, detail: str = ""):
        icon = self.LEVELS.get(level, "•")
        ts   = time.strftime("%H:%M:%S")

        entry = {
            "level":     level,
            "icon":      icon,
            "message":   msg,
            "detail":    detail,
            "timestamp": ts,
        }
        self.logs.append(entry)

        # Afficher dans la console Python
        if self.verbose:
            print(f"[{ts}] {icon} {msg}")
            if detail:
                for line in detail.split("\n")[:5]:
                    print(f"       {line}")

    # ── Résumé ───────────────────────────────────────────────

    def get_summary(self) -> dict:
        return {
            "total_logs":    len(self.logs),
            "errors":        len(self.errors),
            "warnings":      len(self.warnings),
            "steps":         self._step,
            "has_errors":    len(self.errors) > 0,
            "error_details": self.errors,
        }

    def get_logs_as_text(self) -> str:
        """Retourne tous les logs en texte brut."""
        lines = []
        for log in self.logs:
            line = (
                f"[{log['timestamp']}] "
                f"{log['icon']} {log['message']}"
            )
            if log.get("detail"):
                for dl in log["detail"].split("\n")[:3]:
                    if dl.strip():
                        line += f"\n    └─ {dl.strip()}"
            lines.append(line)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# EXTRACTEUR GEMINI PRINCIPAL
# ══════════════════════════════════════════════════════════════

class GeminiExtractor:
    """
    Extracteur de relevés bancaires via Google Gemini.

    Modes disponibles :
    ┌─────────────────────────────────────────────────────┐
    │ MODE HYBRIDE (mode="hybrid")                        │
    │   PDF → pdfplumber → Texte brut → Gemini Text      │
    │   ✅ ~85% moins de tokens   ✅ Rapide               │
    │   ⚠️  Nécessite PDF texte natif lisible             │
    ├─────────────────────────────────────────────────────┤
    │ MODE VISION (mode="vision")                         │
    │   PDF → Images HD → Gemini Vision                   │
    │   ✅ Fonctionne sur tout PDF (natif + scanné)       │
    │   ✅ Lit comme un humain                            │
    │   ⚠️  Plus de tokens (images envoyées)              │
    └─────────────────────────────────────────────────────┘
    """

    # ── Prompt optimisé pour texte brut (mode Hybride) ──────
    PROMPT_TEXT = """Tu es un expert en extraction de données bancaires.
On te donne le texte brut extrait d'une page de relevé bancaire.

COLONNES DU TABLEAU (dans l'ordre d'apparition dans le PDF) :
Date | Batch/Ref | Libellé | D.Valeur | Débit | Crédit | Solde

RÈGLES STRICTES :
1. Extraire TOUTES les lignes de transaction visibles
2. IGNORER ces lignes :
   - En-têtes de colonnes (Date, Batch/Ref, Libellé, D.Valeur, Débit, Crédit, Solde)
   - Ligne TOTAUX ou TOTAL
   - Informations banque (Financial House, Branch ID, Account ID, Periode Du, Print Date, Page Num, Printed By, Working Date, BR.Net)
3. INCLURE : "Solde d'ouverture" et "Solde de clôture"
4. Montants : supprimer TOUS les espaces → "24 553 342" devient 24553342
5. Si colonne Débit vide pour cette ligne → null
6. Si colonne Crédit vide pour cette ligne → null
7. Les frais (SMS Pack, TVA, Historique client) → toujours Débit, jamais Crédit
8. VERST.//, RET.//, Interets Crediteurs → souvent Crédit

FORMAT DE RÉPONSE — JSON uniquement, sans markdown, sans explication :
{"transactions":[{"date":"JJ/MM/AAAA","reference":"XXX/","libelle":"texte complet","date_valeur":"JJ/MM/AAAA","debit":null,"credit":64000,"solde":24617342}]}"""

    # ── Prompt optimisé pour images (mode Vision) ────────────
    PROMPT_VISION = """Tu es un expert en extraction de données bancaires.
On te fournit une IMAGE d'une page de relevé bancaire.

COLONNES DU TABLEAU (dans l'ordre de gauche à droite) :
Date | Batch/Ref | Libellé | D.Valeur | Débit | Crédit | Solde

RÈGLES STRICTES :
1. Lire VISUELLEMENT chaque ligne du tableau
2. Extraire TOUTES les lignes de transactions
3. IGNORER : en-têtes de colonnes, ligne TOTAUX, informations banque en haut/bas de page
4. INCLURE : "Solde d'ouverture" et "Solde de clôture"
5. Montants : supprimer TOUS les espaces → "24 553 342" devient 24553342
6. Débit : montant dans la colonne DÉBIT → si vide = null
7. Crédit : montant dans la colonne CRÉDIT → si vide = null
8. Ne jamais confondre Débit et Crédit : regarder la COLONNE, pas le montant
9. Les frais (SMS Pack, TVA, Historique client) → Débit
10. VERST.//, RET.//, Interets Crediteurs → vérifier visuellement la colonne

FORMAT DE RÉPONSE — JSON uniquement, sans markdown :
{"transactions":[{"date":"JJ/MM/AAAA","reference":"XXX/","libelle":"texte complet","date_valeur":"JJ/MM/AAAA","debit":null,"credit":64000,"solde":24617342}]}"""

    def __init__(
        self,
        api_key:           str,
        mode:              str  = "vision",
        progress_callback       = None,
        verbose_debug:     bool = True,
    ):
        """
        Args:
            api_key:           Clé API Google Gemini
            mode:              "hybrid" ou "vision"
            progress_callback: fonction(step:int, message:str)
            verbose_debug:     Afficher les logs dans la console
        """
        self.api_key   = api_key
        self.mode      = mode
        self.progress_callback = progress_callback
        self._cache    = {}

        # Initialiser le logger
        self.logger = DebugLogger(verbose=verbose_debug)

        # Configurer Gemini
        self.logger.step("Configuration de l'API Gemini")
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config=genai.GenerationConfig(
                    temperature=0,
                    top_p=1,
                    top_k=1,
                    max_output_tokens=8192,
                ),
                safety_settings=[
                    {
                        "category":  "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category":  "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category":  (
                            "HARM_CATEGORY_SEXUALLY_EXPLICIT"
                        ),
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category":  (
                            "HARM_CATEGORY_DANGEROUS_CONTENT"
                        ),
                        "threshold": "BLOCK_NONE",
                    },
                ],
            )
            self.logger.success(
                "Gemini configuré",
                f"Modèle : gemini-1.5-flash | Mode : {mode}"
            )
        except Exception as e:
            self.logger.error(
                "Échec de la configuration Gemini",
                exc=e
            )
            raise

    # ──────────────────────────────────────────────────────────
    # MISE À JOUR DE LA PROGRESSION
    # ──────────────────────────────────────────────────────────

    def _update_progress(self, step: int, message: str):
        if self.progress_callback:
            self.progress_callback(step, message)

    # ══════════════════════════════════════════════════════════
    # POINT D'ENTRÉE PRINCIPAL
    # ══════════════════════════════════════════════════════════

    def extract(self, pdf_bytes: bytes) -> pd.DataFrame:
        """
        Lance l'extraction selon le mode configuré.

        Returns:
            DataFrame avec colonnes :
            Date, Référence, Libellé, Date_Valeur,
            Débit, Crédit, Solde
        """
        self.logger.separator("DÉBUT EXTRACTION")
        self.logger.info(
            f"Mode sélectionné : {self.mode.upper()}",
            f"Taille PDF : {len(pdf_bytes):,} bytes "
            f"({len(pdf_bytes)/1024:.1f} KB)"
        )

        if self.mode == "hybrid":
            return self._extract_hybrid(pdf_bytes)
        elif self.mode == "vision":
            return self._extract_vision(pdf_bytes)
        else:
            self.logger.error(
                f"Mode inconnu : '{self.mode}'",
                "Modes valides : 'hybrid' ou 'vision'"
            )
            return self._empty_df()

    # ══════════════════════════════════════════════════════════
    # MODE 1 : HYBRIDE (pdfplumber + Gemini Text)
    # ══════════════════════════════════════════════════════════

    def _extract_hybrid(self, pdf_bytes: bytes) -> pd.DataFrame:
        """
        Pipeline hybride :
        PDF → pdfplumber → Texte brut → Gemini Text → DataFrame
        """
        self.logger.separator("MODE HYBRIDE")
        self._update_progress(3, "📄 Lecture du PDF...")

        # ── Étape 1 : Extraction texte ────────────────────────
        self.logger.step("Extraction du texte via pdfplumber")
        try:
            pages = self._extract_text_from_pdf(pdf_bytes)
        except Exception as e:
            self.logger.error(
                "Échec de l'extraction texte pdfplumber",
                exc=e
            )
            self.logger.warning(
                "Basculement automatique vers le mode Vision"
            )
            return self._extract_vision(pdf_bytes)

        # ── Diagnostic texte extrait ──────────────────────────
        self.logger.separator("DIAGNOSTIC TEXTE EXTRAIT")
        total_pages = len(pages)

        if total_pages == 0:
            self.logger.error(
                "Aucune page extraite par pdfplumber",
                "Le PDF est peut-être vide, corrompu, "
                "ou entièrement constitué d'images scannées."
            )
            self.logger.warning(
                "Basculement automatique vers le mode Vision"
            )
            return self._extract_vision(pdf_bytes)

        # Analyser la qualité du texte extrait
        total_chars  = sum(p["char_count"] for p in pages)
        avg_chars    = total_chars / total_pages
        empty_pages  = [
            p for p in pages if p["char_count"] < 50
        ]

        self.logger.data(
            "Pages extraites",
            f"{total_pages} pages | "
            f"{total_chars:,} caractères total | "
            f"{avg_chars:.0f} chars/page en moyenne"
        )

        if empty_pages:
            self.logger.warning(
                f"{len(empty_pages)} page(s) avec peu de texte "
                f"(< 50 chars)",
                f"Pages concernées : "
                f"{[p['page_num'] for p in empty_pages]}"
            )

        if avg_chars < 100:
            self.logger.error(
                "Texte insuffisant extrait par pdfplumber",
                f"Moyenne : {avg_chars:.0f} chars/page\n"
                "Cause probable : PDF scanné sans couche texte\n"
                "Solution : Utiliser le mode Vision"
            )
            self.logger.warning(
                "Basculement automatique vers le mode Vision"
            )
            return self._extract_vision(pdf_bytes)

        # Afficher un aperçu du texte de la première page
        if pages:
            preview = pages[0]["text"][:300].replace("\n", " | ")
            self.logger.debug(
                f"Aperçu texte page 1",
                f"'{preview}...'"
            )

        self.logger.success(
            "Texte extrait avec succès",
            f"{total_pages} pages, "
            f"{total_chars:,} caractères"
        )
        self._update_progress(
            35,
            f"✅ {total_pages} page(s) lues — "
            f"envoi à Gemini..."
        )

        # ── Étape 2 : Gemini Text ─────────────────────────────
        self.logger.separator("APPELS API GEMINI")
        all_transactions = []

        for page_data in pages:
            page_num   = page_data["page_num"]
            text       = page_data["text"]
            char_count = page_data["char_count"]
            tokens_est = char_count // 4

            progress = 35 + int(
                (page_num / total_pages) * 50
            )
            self._update_progress(
                progress,
                f"🤖 Gemini analyse page "
                f"{page_num}/{total_pages} "
                f"(~{tokens_est} tokens)..."
            )

            self.logger.api(
                f"Page {page_num}/{total_pages}",
                f"{char_count} chars | ~{tokens_est} tokens"
            )

            # Vérifier que le texte est non-vide
            if not text or char_count < 30:
                self.logger.warning(
                    f"Page {page_num} ignorée",
                    f"Texte trop court ({char_count} chars)"
                )
                continue

            transactions = self._call_gemini_text(
                text,
                page_info=f"{page_num}/{total_pages}"
            )

            self.logger.data(
                f"Page {page_num} → résultat",
                f"{len(transactions)} transaction(s) extraite(s)"
            )

            all_transactions.extend(transactions)

            if page_num < total_pages:
                time.sleep(0.5)

        # ── Étape 3 : Résultat final ──────────────────────────
        self.logger.separator("RÉSULTAT FINAL")
        self.logger.data(
            "Total transactions extraites",
            f"{len(all_transactions)} transaction(s)"
        )

        if not all_transactions:
            self.logger.error(
                "Aucune transaction extraite (mode Hybride)",
                "Causes possibles :\n"
                "1. Le texte extrait n'est pas structuré en tableau\n"
                "2. Gemini n'a pas reconnu le format\n"
                "3. Le PDF ne contient pas de texte lisible\n"
                "→ Essayez le mode Vision"
            )
            return self._empty_df()

        self._update_progress(
            88, "🔧 Construction du tableau..."
        )
        df = self._build_dataframe(all_transactions)
        self.logger.success(
            "Extraction hybride terminée",
            f"{len(df)} lignes dans le DataFrame final"
        )
        return df

    # ── Extraction texte pdfplumber ───────────────────────────

    def _extract_text_from_pdf(
        self, pdf_bytes: bytes
    ) -> list:
        """Extrait le texte de chaque page via pdfplumber."""
        pages = []

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            total = len(pdf.pages)
            self.logger.info(
                f"PDF ouvert",
                f"{total} pages détectées"
            )

            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                progress = 5 + int((i / total) * 25)
                self._update_progress(
                    progress,
                    f"📄 Extraction texte page "
                    f"{page_num}/{total}..."
                )

                # Méthode 1 : extract_text avec layout
                raw_text = None
                try:
                    raw_text = page.extract_text(
                        x_tolerance=3,
                        y_tolerance=3,
                        layout=True,
                        x_density=7.25,
                        y_density=13,
                    )
                    self.logger.debug(
                        f"Page {page_num} — extract_text(layout)",
                        f"{len(raw_text or '')} chars extraits"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Page {page_num} — extract_text échoué",
                        str(e)
                    )

                # Méthode 2 : par mots si résultat insuffisant
                if not raw_text or len(raw_text.strip()) < 50:
                    self.logger.debug(
                        f"Page {page_num} — "
                        f"fallback extract_words",
                        f"Texte initial : "
                        f"'{(raw_text or '').strip()[:80]}'"
                    )
                    raw_text = self._words_to_text(page)
                    self.logger.debug(
                        f"Page {page_num} — extract_words",
                        f"{len(raw_text)} chars extraits"
                    )

                # Méthode 3 : extract_table si tout échoue
                if not raw_text or len(raw_text.strip()) < 50:
                    self.logger.debug(
                        f"Page {page_num} — "
                        f"fallback extract_tables"
                    )
                    raw_text = self._tables_to_text(page)
                    self.logger.debug(
                        f"Page {page_num} — extract_tables",
                        f"{len(raw_text)} chars extraits"
                    )

                # Nettoyer et stocker
                cleaned = self._clean_text(raw_text or "")
                char_count = len(cleaned)

                self.logger.debug(
                    f"Page {page_num} — après nettoyage",
                    f"{char_count} chars | "
                    f"Aperçu : "
                    f"'{cleaned[:150].replace(chr(10), ' | ')}'"
                )

                pages.append({
                    "page_num":    page_num,
                    "text":        cleaned,
                    "char_count":  char_count,
                    "total_pages": total,
                })

        return pages

    def _words_to_text(self, page) -> str:
        """Reconstruit le texte depuis les mots par position Y."""
        try:
            words = page.extract_words(
                x_tolerance=4,
                y_tolerance=4,
                keep_blank_chars=False,
            )
            if not words:
                return ""

            lines = {}
            for w in words:
                y = round(float(w["top"]) / 4) * 4
                lines.setdefault(y, []).append(w)

            result = []
            for y in sorted(lines.keys()):
                line_words = sorted(
                    lines[y], key=lambda w: w["x0"]
                )
                result.append(
                    " ".join(w["text"] for w in line_words)
                )
            return "\n".join(result)
        except Exception as e:
            self.logger.warning(
                "Erreur _words_to_text", str(e)
            )
            return ""

    def _tables_to_text(self, page) -> str:
        """Convertit les tableaux pdfplumber en texte."""
        try:
            tables = page.extract_tables()
            if not tables:
                return ""
            lines = []
            for table in tables:
                for row in table:
                    if row:
                        cells = [
                            str(c).strip()
                            for c in row
                            if c and str(c).strip()
                        ]
                        if cells:
                            lines.append(" | ".join(cells))
            return "\n".join(lines)
        except Exception as e:
            self.logger.warning(
                "Erreur _tables_to_text", str(e)
            )
            return ""

    def _clean_text(self, text: str) -> str:
        """Nettoie le texte pour réduire les tokens."""
        skip = [
            r"financial house s\.?a",
            r"fh sa\s*[-–]\s*ndjiemoum",
            r"historique compte client",
            r"branch id\s*:",
            r"account id\s*:",
            r"print date\s*:",
            r"page num\s*:",
            r"printed by\s*:",
            r"working date\s*:",
            r"br\.net ver",
            r"^\s*(ibantio|bdjoko|gtakeu|ideynou|mmodjo)\s*$",
        ]
        lines      = text.split("\n")
        result     = []
        prev_empty = False

        for line in lines:
            stripped = line.strip()
            lower    = stripped.lower()

            # Ignorer les lignes de métadonnées
            if any(re.search(p, lower) for p in skip):
                continue

            # Limiter les lignes vides consécutives
            if not stripped:
                if not prev_empty:
                    result.append("")
                prev_empty = True
            else:
                result.append(stripped)
                prev_empty = False

        return "\n".join(result).strip()

    # ══════════════════════════════════════════════════════════
    # MODE 2 : VISION (PDF → Images → Gemini Vision)
    # ══════════════════════════════════════════════════════════

    def _extract_vision(self, pdf_bytes: bytes) -> pd.DataFrame:
        """
        Pipeline vision :
        PDF → Images HD → Gemini Vision → DataFrame
        """
        self.logger.separator("MODE VISION")
        self._update_progress(3, "🖼️ Conversion PDF en images...")

        # ── Vérifier pdf2image ────────────────────────────────
        if not PDF2IMAGE_AVAILABLE:
            self.logger.error(
                "pdf2image non installé",
                "Installez : pip install pdf2image\n"
                "Et sur le système : sudo apt install "
                "poppler-utils"
            )
            return self._empty_df()

        # ── Convertir PDF → Images ────────────────────────────
        self.logger.step("Conversion PDF → Images")
        try:
            images = convert_from_bytes(
                pdf_bytes,
                dpi=200,
                fmt="PNG",
                thread_count=2,
            )
            self.logger.success(
                f"{len(images)} image(s) générée(s)",
                "DPI : 200 | Format : PNG"
            )
        except Exception as e:
            self.logger.error(
                "Échec conversion PDF → Images",
                "Vérifiez que poppler-utils est installé :\n"
                "  Linux : sudo apt install poppler-utils\n"
                "  Mac   : brew install poppler\n"
                "  Windows : télécharger poppler binaries",
                exc=e
            )
            return self._empty_df()

        total_pages      = len(images)
        all_transactions = []

        self.logger.separator("APPELS API GEMINI VISION")

        for page_num, image in enumerate(images, 1):
            progress = 10 + int(
                (page_num / total_pages) * 75
            )
            self._update_progress(
                progress,
                f"🤖 Gemini Vision — page "
                f"{page_num}/{total_pages}..."
            )

            self.logger.api(
                f"Page {page_num}/{total_pages}",
                f"Taille image : "
                f"{image.width}x{image.height}px"
            )

            transactions = self._call_gemini_vision(
                image, page_num, total_pages
            )

            self.logger.data(
                f"Page {page_num} → résultat",
                f"{len(transactions)} transaction(s)"
            )
            all_transactions.extend(transactions)

            if page_num < total_pages:
                time.sleep(1)

        # ── Résultat final ────────────────────────────────────
        self.logger.separator("RÉSULTAT FINAL")
        self.logger.data(
            "Total transactions",
            f"{len(all_transactions)}"
        )

        if not all_transactions:
            self.logger.error(
                "Aucune transaction extraite (mode Vision)",
                "Causes possibles :\n"
                "1. Image trop floue / résolution insuffisante\n"
                "2. Tableau non reconnu par Gemini\n"
                "3. Format de relevé non standard\n"
                "4. Erreur API Gemini"
            )
            return self._empty_df()

        self._update_progress(90, "🔧 Construction tableau...")
        df = self._build_dataframe(all_transactions)
        self.logger.success(
            "Extraction vision terminée",
            f"{len(df)} lignes dans le DataFrame final"
        )
        return df

    # ── Appel Gemini Vision ───────────────────────────────────

    def _call_gemini_vision(
        self,
        image:      Image.Image,
        page_num:   int,
        total:      int,
    ) -> list:
        """
        Envoie une image à Gemini Vision et retourne
        les transactions extraites.
        """
        # Optimiser l'image
        optimized = self._optimize_image(image)

        self.logger.debug(
            f"Image optimisée",
            f"{optimized.width}x{optimized.height}px"
        )

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                self.logger.api(
                    f"Envoi image p.{page_num} "
                    f"(tentative {attempt}/{max_retries})"
                )

                response = self.model.generate_content(
                    [self.PROMPT_VISION, optimized],
                    request_options={"timeout": 60},
                )

                raw = response.text
                self.logger.debug(
                    f"Réponse Gemini Vision p.{page_num}",
                    f"Longueur : {len(raw)} chars\n"
                    f"Aperçu : '{raw[:200]}'"
                )

                transactions = self._parse_response(
                    raw, page_num
                )
                return transactions

            except Exception as e:
                err = str(e)
                self.logger.warning(
                    f"Erreur page {page_num} "
                    f"(tentative {attempt})",
                    err[:200]
                )

                # Rate limit → attendre
                if any(
                    c in err
                    for c in ["429", "quota",
                               "RESOURCE_EXHAUSTED"]
                ):
                    wait = attempt * 15
                    self.logger.warning(
                        f"Rate limit — attente {wait}s"
                    )
                    self._update_progress(
                        -1,
                        f"⏳ Limite API — attente {wait}s..."
                    )
                    time.sleep(wait)
                    continue

                if attempt == max_retries:
                    self.logger.error(
                        f"Échec définitif page {page_num}",
                        exc=e
                    )
                    return []
                time.sleep(3)

        return []

    def _optimize_image(
        self, image: Image.Image
    ) -> Image.Image:
        """Optimise l'image pour Gemini (taille + mode)."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        max_w = 1800
        if image.width > max_w:
            ratio  = max_w / image.width
            new_h  = int(image.height * ratio)
            image  = image.resize(
                (max_w, new_h), Image.LANCZOS
            )
        return image

    # ══════════════════════════════════════════════════════════
    # APPEL GEMINI TEXT (mode Hybride)
    # ══════════════════════════════════════════════════════════

    def _call_gemini_text(
        self, text: str, page_info: str = ""
    ) -> list:
        """
        Envoie du texte brut à Gemini et retourne
        les transactions extraites.
        Gère le cache, les retries et le rate limiting.
        """
        # Cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._cache:
            self.logger.debug(
                f"Cache hit pour page {page_info}"
            )
            return self._cache[cache_key]

        # Chunking si texte trop long
        if len(text) > 8000:
            self.logger.warning(
                f"Texte trop long ({len(text)} chars) "
                f"— découpage en chunks",
                f"Page : {page_info}"
            )
            return self._call_gemini_text_chunked(
                text, page_info
            )

        user_prompt = (
            f"Voici le texte brut de la page {page_info} "
            f"du relevé bancaire :\n\n"
            f"---\n{text}\n---\n\n"
            f"Extrais toutes les transactions "
            f"et retourne uniquement le JSON."
        )

        self.logger.token(
            f"Prompt envoyé",
            f"System : {len(self.PROMPT_TEXT)} chars | "
            f"User : {len(user_prompt)} chars | "
            f"Total estimé : ~{(len(self.PROMPT_TEXT) + len(user_prompt))//4} tokens"
        )

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                self.logger.api(
                    f"Appel Gemini Text — page {page_info} "
                    f"(tentative {attempt}/{max_retries})"
                )

                response = self.model.generate_content(
                    [self.PROMPT_TEXT, user_prompt],
                    request_options={"timeout": 30},
                )

                raw = response.text
                self.logger.debug(
                    f"Réponse brute Gemini",
                    f"Longueur : {len(raw)} chars\n"
                    f"Début : '{raw[:300]}'"
                )

                transactions = self._parse_response(
                    raw, page_info
                )

                # Stocker en cache
                self._cache[cache_key] = transactions
                return transactions

            except Exception as e:
                err = str(e)
                self.logger.warning(
                    f"Erreur Gemini Text — page {page_info} "
                    f"(tentative {attempt})",
                    err[:300]
                )

                if any(
                    c in err
                    for c in ["429", "quota",
                               "RESOURCE_EXHAUSTED"]
                ):
                    wait = attempt * 15
                    self.logger.warning(
                        f"Rate limit — attente {wait}s"
                    )
                    self._update_progress(
                        -1,
                        f"⏳ Limite API — attente {wait}s..."
                    )
                    time.sleep(wait)
                    continue

                if "context" in err.lower() or "413" in err:
                    self.logger.warning(
                        "Contexte trop long — chunking"
                    )
                    return self._call_gemini_text_chunked(
                        text, page_info
                    )

                if attempt == max_retries:
                    self.logger.error(
                        f"Échec définitif page {page_info}",
                        exc=e
                    )
                    return []
                time.sleep(3)

        return []

    def _call_gemini_text_chunked(
        self, text: str, page_info: str
    ) -> list:
        """Découpe le texte en chunks et traite chacun."""
        chunks       = self._split_chunks(text, max_chars=5000)
        all_trans    = []

        self.logger.info(
            f"Chunking : {len(chunks)} chunk(s) "
            f"pour page {page_info}"
        )

        for i, chunk in enumerate(chunks, 1):
            self.logger.debug(
                f"Chunk {i}/{len(chunks)}",
                f"{len(chunk)} chars"
            )
            trans = self._call_gemini_text(
                chunk,
                page_info=f"{page_info} chunk {i}/{len(chunks)}"
            )
            all_trans.extend(trans)
            if i < len(chunks):
                time.sleep(0.5)

        return all_trans

    def _split_chunks(
        self, text: str, max_chars: int = 5000
    ) -> list:
        """Découpe intelligemment sur les lignes vides."""
        if len(text) <= max_chars:
            return [text]

        chunks  = []
        current = []
        curr_l  = 0

        for line in text.split("\n"):
            ll = len(line) + 1
            if curr_l + ll > max_chars and current:
                chunks.append("\n".join(current))
                current = [line]
                curr_l  = ll
            else:
                current.append(line)
                curr_l += ll

        if current:
            chunks.append("\n".join(current))

        return chunks

    # ══════════════════════════════════════════════════════════
    # PARSING DE LA RÉPONSE GEMINI
    # ══════════════════════════════════════════════════════════

    def _parse_response(
        self, response_text: str, page_info: str = ""
    ) -> list:
        """
        Parse la réponse JSON de Gemini.
        Robuste et détaillé en cas d'erreur.
        """
        if not response_text:
            self.logger.error(
                f"Réponse vide de Gemini — page {page_info}"
            )
            return []

        text = response_text.strip()

        # Nettoyer les balises markdown
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*",     "", text)
        text = text.strip()

        self.logger.debug(
            f"Texte après nettoyage markdown",
            f"Longueur : {len(text)} chars\n"
            f"Début : '{text[:200]}'"
        )

        # Extraire le JSON
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            self.logger.error(
                f"Pas de JSON valide dans la réponse "
                f"— page {page_info}",
                f"Contenu reçu :\n'{text[:500]}'"
            )
            return []

        json_str = json_match.group(0)
        self.logger.debug(
            "JSON extrait",
            f"Longueur : {len(json_str)} chars\n"
            f"Début : '{json_str[:200]}'"
        )

        # Parser le JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.warning(
                f"JSON malformé — tentative de réparation",
                f"Erreur : {e}"
            )
            json_str = self._repair_json(json_str)
            try:
                data = json.loads(json_str)
                self.logger.success("JSON réparé avec succès")
            except Exception as e2:
                self.logger.error(
                    f"JSON irréparable — page {page_info}",
                    f"Erreur : {e2}\n"
                    f"JSON brut : '{json_str[:300]}'",
                    exc=e2
                )
                return []

        transactions = data.get("transactions", [])

        if not isinstance(transactions, list):
            self.logger.error(
                "La clé 'transactions' n'est pas une liste",
                f"Type reçu : {type(transactions)}\n"
                f"Valeur : {str(transactions)[:200]}"
            )
            return []

        self.logger.debug(
            f"Transactions brutes parsées",
            f"{len(transactions)} éléments"
        )

        # Normaliser chaque transaction
        validated  = []
        skipped    = 0
        for i, t in enumerate(transactions):
            norm = self._normalize(t)
            if norm:
                validated.append(norm)
            else:
                skipped += 1
                self.logger.debug(
                    f"Transaction {i+1} ignorée",
                    f"Données : {str(t)[:150]}"
                )

        if skipped > 0:
            self.logger.warning(
                f"{skipped} transaction(s) ignorée(s) "
                f"(données invalides)"
            )

        self.logger.success(
            f"Parsing terminé — page {page_info}",
            f"{len(validated)} transactions valides "
            f"/ {len(transactions)} brutes"
        )
        return validated

    def _repair_json(self, s: str) -> str:
        """Répare les JSON légèrement malformés."""
        s = re.sub(r",\s*}", "}", s)
        s = re.sub(r",\s*]", "]", s)
        s = s.replace("None",  "null")
        s = s.replace("True",  "true")
        s = s.replace("False", "false")
        s = s.replace("'",     '"')
        return s

    def _normalize(self, t: dict) -> dict:
        """Normalise et valide une transaction."""
        if not isinstance(t, dict):
            return None

        libelle = str(t.get("libelle", "") or "").strip()
        if not libelle:
            return None

        return {
            "date":        self._fmt_date(t.get("date", "")),
            "reference":   str(
                t.get("reference", "") or ""
            ).strip(),
            "libelle":     libelle,
            "date_valeur": self._fmt_date(
                t.get("date_valeur", "")
            ),
            "debit":       self._fmt_amount(t.get("debit")),
            "credit":      self._fmt_amount(t.get("credit")),
            "solde":       self._fmt_amount(t.get("solde")),
        }

    def _fmt_date(self, val) -> str:
        if not val:
            return ""
        s = str(val).strip()
        if re.match(r"\d{2}/\d{2}/\d{4}", s):
            return s[:10]
        return s

    def _fmt_amount(self, val) -> float:
        if val is None or str(val).lower() in ("null", ""):
            return None
        try:
            s = re.sub(r"[^\d.]", "", str(val).replace(" ", ""))
            return float(s) if s else None
        except (ValueError, TypeError):
            return None

    # ══════════════════════════════════════════════════════════
    # CONSTRUCTION DU DATAFRAME
    # ══════════════════════════════════════════════════════════

    def _build_dataframe(
        self, transactions: list
    ) -> pd.DataFrame:
        """Construit le DataFrame final."""
        rows = [{
            "Date":        t.get("date", ""),
            "Référence":   t.get("reference", ""),
            "Libellé":     t.get("libelle", ""),
            "Date_Valeur": t.get("date_valeur", ""),
            "Débit":       t.get("debit"),
            "Crédit":      t.get("credit"),
            "Solde":       t.get("solde"),
        } for t in transactions]

        df = pd.DataFrame(rows)

        # Supprimer les vrais doublons
        mask_solde = df["Libellé"].str.contains(
            r"solde\s+d|solde\s+de\s+cl",
            case=False, na=False, regex=True,
        )
        df_normal = df[~mask_solde].drop_duplicates(
            subset=["Date", "Référence", "Libellé"],
            keep="first",
        )
        df_soldes = df[mask_solde]

        return pd.concat(
            [df_soldes, df_normal], ignore_index=True
        )

    # ══════════════════════════════════════════════════════════
    # MÉTHODES UTILITAIRES PUBLIQUES
    # ══════════════════════════════════════════════════════════

    def _empty_df(self) -> pd.DataFrame:
        return pd.DataFrame(columns=[
            "Date", "Référence", "Libellé",
            "Date_Valeur", "Débit", "Crédit", "Solde",
        ])

    def get_debug_logs(self) -> str:
        """Retourne tous les logs de débogage en texte."""
        return self.logger.get_logs_as_text()

    def get_debug_summary(self) -> dict:
        """Retourne un résumé du débogage."""
        return self.logger.get_summary()

    def get_debug_entries(self) -> list:
        """Retourne les entrées de log brutes."""
        return self.logger.logs

    # ── Méthode publique pour estimation tokens ───────────────

    def _extract_text_from_pdf_public(
        self, pdf_bytes: bytes
    ) -> list:
        """
        Alias public pour l'estimation de tokens
        (appelé depuis app.py avant confirmation).
        """
        return self._extract_text_from_pdf(pdf_bytes)
