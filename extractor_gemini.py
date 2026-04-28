"""
╔══════════════════════════════════════════════════════════════╗
║         EXTRACTEUR GEMINI — Bank Statement Extractor         ║
║         Modes : Vision | Hybride                             ║
║         Version : 3.2.0 — Correction extraction Vision       ║
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


# ══════════════════════════════════════════════════════════════
# LOGGER DE DÉBOGAGE
# ══════════════════════════════════════════════════════════════

class DebugLogger:
    """Logger structuré pour tracer chaque étape."""

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
        self.logs     = []
        self.errors   = []
        self.warnings = []
        self._step    = 0

    def info(self, msg, detail=""):
        self._log("INFO", msg, detail)

    def success(self, msg, detail=""):
        self._log("SUCCESS", msg, detail)

    def warning(self, msg, detail=""):
        self._log("WARNING", msg, detail)
        self.warnings.append(msg)

    def error(self, msg, detail="", exc=None):
        full = detail
        if exc:
            full += f"\n{type(exc).__name__}: {str(exc)}"
            full += f"\n{traceback.format_exc()}"
        self._log("ERROR", msg, full)
        self.errors.append({"msg": msg, "detail": full})

    def debug(self, msg, detail=""):
        self._log("DEBUG", msg, detail)

    def step(self, msg):
        self._step += 1
        self._log("STEP", f"[Étape {self._step}] {msg}", "")

    def data(self, label, value):
        val_str = str(value)
        if len(val_str) > 800:
            val_str = val_str[:800] + "... [tronqué]"
        self._log("DATA", label, val_str)

    def api(self, msg, detail=""):
        self._log("API", msg, detail)

    def token(self, msg, detail=""):
        self._log("TOKEN", msg, detail)

    def separator(self, label=""):
        sep = "─" * 40
        self._log(
            "INFO",
            f"{sep} {label} {sep}" if label else sep,
            ""
        )
        if self.verbose:
            print(f"\n{'─'*20} {label} {'─'*20}")

    def _log(self, level, msg, detail=""):
        icon = self.LEVELS.get(level, "•")
        ts   = time.strftime("%H:%M:%S")
        entry = {
            "level":     level,
            "icon":      icon,
            "message":   str(msg),
            "detail":    str(detail),
            "timestamp": ts,
        }
        self.logs.append(entry)
        if self.verbose:
            print(f"[{ts}] {icon} {msg}")
            if detail:
                for line in str(detail).split("\n")[:5]:
                    if line.strip():
                        print(f"       {line}")

    def get_summary(self):
        return {
            "total_logs":    len(self.logs),
            "errors":        len(self.errors),
            "warnings":      len(self.warnings),
            "steps":         self._step,
            "has_errors":    len(self.errors) > 0,
            "error_details": self.errors,
        }

    def get_logs_as_text(self):
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
# EXTRACTEUR GEMINI
# ══════════════════════════════════════════════════════════════

class GeminiExtractor:
    """
    Extracteur de relevés bancaires via Google Gemini.
    Mode Vision  : PDF → Images → Gemini Vision (multimodal)
    Mode Hybride : PDF → Texte pdfplumber → Gemini Text
    """

    # ── Prompt Vision ────────────────────────────────────────
    PROMPT_VISION = """Tu es un expert comptable spécialisé en extraction de données bancaires.
Tu reçois l'IMAGE d'une page de relevé bancaire camerounais.

OBJECTIF : Extraire TOUTES les lignes du tableau de transactions visible.

STRUCTURE DU TABLEAU (colonnes de gauche à droite) :
Date | Batch/Ref | Libellé | D.Valeur | Débit | Crédit | Solde

INSTRUCTIONS :
1. Lis chaque ligne du tableau de haut en bas
2. Pour chaque ligne de transaction, extrais les 7 champs
3. IGNORE : ligne d'en-tête, ligne TOTAUX, texte hors tableau
4. INCLUS : "Solde d'ouverture" et "Solde de clôture"
5. Montants : SUPPRIME les espaces → "24 553 342" → 24553342
6. Colonne DÉBIT vide → mettre null (pas 0)
7. Colonne CRÉDIT vide → mettre null (pas 0)
8. Ne confonds JAMAIS Débit et Crédit : regarde la colonne exacte
9. Les frais (SMS Pack, TVA, Historique) → DÉBIT, pas crédit
10. VERST.//, RET.//, Interets Crediteurs → vérifie la colonne

FORMAT DE RÉPONSE :
Retourne UNIQUEMENT le JSON suivant, sans texte avant ni après :

{"transactions":[{"date":"JJ/MM/AAAA","reference":"XXX/","libelle":"description","date_valeur":"JJ/MM/AAAA","debit":null,"credit":64000,"solde":24617342}]}

Si aucune transaction trouvée : {"transactions":[]}"""

    # ── Prompt Texte (Hybride) ───────────────────────────────
    PROMPT_TEXT = """Tu es un expert comptable spécialisé en extraction de données bancaires.
Tu reçois le TEXTE BRUT d'une page de relevé bancaire camerounais.

OBJECTIF : Extraire TOUTES les lignes du tableau de transactions.

STRUCTURE DU TABLEAU :
Date | Batch/Ref | Libellé | D.Valeur | Débit | Crédit | Solde

INSTRUCTIONS :
1. Identifie et extrais chaque ligne de transaction
2. IGNORE : en-têtes de colonnes, ligne TOTAUX, métadonnées banque
3. INCLUS : "Solde d'ouverture" et "Solde de clôture"
4. Montants : SUPPRIME les espaces → "24 553 342" → 24553342
5. Colonne DÉBIT vide → null
6. Colonne CRÉDIT vide → null
7. Les frais (SMS Pack, TVA, Historique) → DÉBIT
8. VERST.//, RET.//, Interets Crediteurs → souvent CRÉDIT

FORMAT DE RÉPONSE :
Retourne UNIQUEMENT le JSON, sans texte avant ni après :

{"transactions":[{"date":"JJ/MM/AAAA","reference":"XXX/","libelle":"description","date_valeur":"JJ/MM/AAAA","debit":null,"credit":64000,"solde":24617342}]}

Si aucune transaction : {"transactions":[]}"""

    def __init__(
        self,
        api_key:           str,
        mode:              str  = "vision",
        progress_callback       = None,
        verbose_debug:     bool = True,
    ):
        self.api_key           = api_key
        self.mode              = mode
        self.progress_callback = progress_callback
        self._cache            = {}
        self.logger            = DebugLogger(verbose=verbose_debug)

        # ── Configurer Gemini ─────────────────────────────────
        self.logger.step("Configuration API Gemini")
        try:
            genai.configure(api_key=api_key)

            self.model = genai.GenerativeModel(
                model_name="gemini-2.5-flash-lite",
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
                        "category":  "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category":  "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE",
                    },
                ],
            )
            self.logger.success(
                "Gemini configuré avec succès",
                f"Modèle : gemini-2.5-flash-lite | Mode : {mode}"
            )
        except Exception as e:
            self.logger.error(
                "Échec configuration Gemini", exc=e
            )
            raise

    # ──────────────────────────────────────────────────────────
    def _update_progress(self, step: int, msg: str):
        if self.progress_callback:
            self.progress_callback(step, msg)

    # ══════════════════════════════════════════════════════════
    # POINT D'ENTRÉE
    # ══════════════════════════════════════════════════════════

    def extract(self, pdf_bytes: bytes) -> pd.DataFrame:
        self.logger.separator("DÉBUT EXTRACTION")
        self.logger.info(
            f"Mode : {self.mode.upper()}",
            f"Taille PDF : {len(pdf_bytes):,} bytes "
            f"({len(pdf_bytes)/1024:.1f} KB)"
        )
        if self.mode == "hybrid":
            return self._extract_hybrid(pdf_bytes)
        return self._extract_vision(pdf_bytes)

    # ══════════════════════════════════════════════════════════
    # MODE VISION
    # ══════════════════════════════════════════════════════════

    def _extract_vision(self, pdf_bytes: bytes) -> pd.DataFrame:
        self.logger.separator("MODE VISION")
        self._update_progress(3, "🖼️ Conversion PDF en images...")

        # ── Vérifier pdf2image ────────────────────────────────
        if not PDF2IMAGE_AVAILABLE:
            self.logger.error(
                "pdf2image non installé",
                "pip install pdf2image\n"
                "sudo apt install poppler-utils"
            )
            return self._empty_df()

        # ── Convertir PDF → Images ────────────────────────────
        self.logger.step("Conversion PDF → Images haute résolution")
        try:
            images = convert_from_bytes(
                pdf_bytes,
                dpi=250,
                fmt="PNG",
                thread_count=1,
            )
            self.logger.success(
                f"{len(images)} image(s) générée(s)",
                f"DPI=250 | Format=PNG"
            )
        except Exception as e:
            self.logger.error(
                "Échec conversion PDF → Images",
                "Vérifiez poppler-utils :\n"
                "  Linux : sudo apt install poppler-utils\n"
                "  Mac   : brew install poppler",
                exc=e
            )
            return self._empty_df()

        total            = len(images)
        all_transactions = []

        self.logger.separator("APPELS API GEMINI VISION")

        for idx, image in enumerate(images):
            page_num = idx + 1
            progress = 10 + int((page_num / total) * 75)
            self._update_progress(
                progress,
                f"🤖 Gemini Vision — page {page_num}/{total}..."
            )

            self.logger.api(
                f"Traitement page {page_num}/{total}",
                f"Résolution : {image.width}x{image.height}px"
            )

            transactions = self._call_vision_single_page(
                image, page_num, total
            )

            self.logger.data(
                f"Résultat page {page_num}",
                f"{len(transactions)} transaction(s) extraite(s)"
            )

            # Afficher les 3 premières pour vérification
            if transactions:
                for i, t in enumerate(transactions[:3]):
                    self.logger.debug(
                        f"  Transaction {i+1}",
                        json.dumps(t, ensure_ascii=False)
                    )

            all_transactions.extend(transactions)

            # Pause entre pages pour éviter le rate limiting
            if page_num < total:
                self.logger.debug(
                    f"Pause 1.5s avant page {page_num+1}"
                )
                time.sleep(1.5)

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
                "1. Gemini n'a pas retourné de JSON valide\n"
                "2. Le format du relevé n'est pas reconnu\n"
                "3. Les images sont trop floues\n"
                "4. Erreur de parsing JSON\n"
                "→ Consultez les logs détaillés ci-dessus"
            )
            return self._empty_df()

        self._update_progress(90, "🔧 Construction du tableau...")
        df = self._build_dataframe(all_transactions)
        self.logger.success(
            "Extraction Vision terminée",
            f"{len(df)} lignes dans le DataFrame final"
        )
        return df

    # ── Appel Gemini pour UNE page (Vision) ──────────────────

    def _call_vision_single_page(
        self,
        image:    Image.Image,
        page_num: int,
        total:    int,
    ) -> list:
        """
        Envoie une image à Gemini et retourne les transactions.
        Gère les retries, le rate limiting et le debug.
        """
        # Optimiser l'image
        optimized = self._optimize_image(image)
        self.logger.debug(
            f"Image optimisée p.{page_num}",
            f"{optimized.width}x{optimized.height}px"
        )

        max_retries = 3
        last_error  = None

        for attempt in range(1, max_retries + 1):
            try:
                self.logger.api(
                    f"Envoi image à Gemini "
                    f"(page {page_num}, tentative {attempt}/{max_retries})"
                )

                # ── Appel API ─────────────────────────────────
                response = self.model.generate_content(
                    contents=[self.PROMPT_VISION, optimized],
                    request_options={"timeout": 120},
                )

                # ── Vérifier la réponse ───────────────────────
                if not response:
                    self.logger.warning(
                        f"Réponse None de Gemini "
                        f"(page {page_num})"
                    )
                    continue

                # ── Extraire le texte ─────────────────────────
                raw_text = None
                try:
                    raw_text = response.text
                except Exception as e:
                    self.logger.warning(
                        f"Impossible de lire response.text",
                        str(e)
                    )
                    # Essayer via candidates
                    try:
                        raw_text = (
                            response.candidates[0]
                            .content.parts[0].text
                        )
                    except Exception as e2:
                        self.logger.error(
                            "Impossible d'extraire le texte "
                            "de la réponse Gemini",
                            str(e2)
                        )
                        continue

                if not raw_text:
                    self.logger.warning(
                        f"Texte vide dans la réponse Gemini "
                        f"(page {page_num})"
                    )

                    # Log le finish_reason si disponible
                    try:
                        fr = (
                            response.candidates[0]
                            .finish_reason
                        )
                        self.logger.warning(
                            f"Finish reason : {fr}"
                        )
                    except Exception:
                        pass
                    continue

                self.logger.debug(
                    f"Réponse brute Gemini (page {page_num})",
                    f"Longueur : {len(raw_text)} chars\n"
                    f"Aperçu : '{raw_text[:400]}'"
                )

                # ── Parser le JSON ────────────────────────────
                transactions = self._parse_response(
                    raw_text, f"page {page_num}"
                )
                return transactions

            except Exception as e:
                last_error = e
                err_str    = str(e)
                self.logger.warning(
                    f"Erreur Gemini Vision "
                    f"(page {page_num}, tentative {attempt})",
                    err_str[:300]
                )

                # Rate limiting
                if any(
                    c in err_str
                    for c in [
                        "429", "quota",
                        "RESOURCE_EXHAUSTED",
                        "rate"
                    ]
                ):
                    wait = attempt * 20
                    self.logger.warning(
                        f"Rate limit détecté → attente {wait}s"
                    )
                    self._update_progress(
                        -1,
                        f"⏳ Limite API — attente {wait}s..."
                    )
                    time.sleep(wait)
                    continue

                # Timeout
                if "timeout" in err_str.lower():
                    self.logger.warning(
                        f"Timeout → retry dans 5s"
                    )
                    time.sleep(5)
                    continue

                if attempt == max_retries:
                    self.logger.error(
                        f"Échec définitif page {page_num}",
                        exc=last_error
                    )
                time.sleep(2)

        return []

    def _optimize_image(
        self, image: Image.Image
    ) -> Image.Image:
        """Optimise l'image pour Gemini Vision."""
        # Convertir en RGB
        if image.mode != "RGB":
            self.logger.debug(
                f"Conversion mode {image.mode} → RGB"
            )
            image = image.convert("RGB")

        # Redimensionner si nécessaire
        max_width = 2000
        if image.width > max_width:
            ratio  = max_width / image.width
            new_h  = int(image.height * ratio)
            image  = image.resize(
                (max_width, new_h), Image.LANCZOS
            )
            self.logger.debug(
                f"Image redimensionnée → {max_width}x{new_h}px"
            )

        return image

    # ══════════════════════════════════════════════════════════
    # MODE HYBRIDE
    # ══════════════════════════════════════════════════════════

    def _extract_hybrid(
        self, pdf_bytes: bytes
    ) -> pd.DataFrame:
        """Pipeline hybride : texte pdfplumber + Gemini Text."""
        self.logger.separator("MODE HYBRIDE")
        self._update_progress(3, "📄 Lecture du PDF...")

        # ── Extraire le texte ─────────────────────────────────
        self.logger.step(
            "Extraction texte via pdfplumber"
        )
        try:
            pages = self._extract_text_from_pdf(pdf_bytes)
        except Exception as e:
            self.logger.error(
                "Échec extraction texte pdfplumber",
                exc=e
            )
            self.logger.warning(
                "Basculement automatique vers mode Vision"
            )
            return self._extract_vision(pdf_bytes)

        # ── Diagnostic qualité texte ──────────────────────────
        total_pages = len(pages)
        if total_pages == 0:
            self.logger.error(
                "Aucune page extraite par pdfplumber",
                "PDF vide, corrompu ou 100% scanné"
            )
            return self._extract_vision(pdf_bytes)

        total_chars = sum(p["char_count"] for p in pages)
        avg_chars   = total_chars / total_pages
        empty_pages = [
            p for p in pages if p["char_count"] < 50
        ]

        self.logger.data(
            "Texte extrait",
            f"{total_pages} pages | "
            f"{total_chars:,} chars | "
            f"{avg_chars:.0f} chars/page"
        )

        if empty_pages:
            self.logger.warning(
                f"{len(empty_pages)} page(s) avec peu de texte",
                str([p["page_num"] for p in empty_pages])
            )

        if avg_chars < 100:
            self.logger.error(
                "Texte insuffisant",
                f"Moyenne {avg_chars:.0f} chars/page\n"
                "→ PDF probablement scanné\n"
                "→ Basculement vers mode Vision"
            )
            return self._extract_vision(pdf_bytes)

        self.logger.success(
            "Texte extrait avec succès",
            f"{total_pages} pages, {total_chars:,} chars"
        )

        self._update_progress(
            35,
            f"✅ {total_pages} page(s) lues — envoi à Gemini..."
        )

        # ── Appels Gemini Text ────────────────────────────────
        self.logger.separator("APPELS API GEMINI TEXT")
        all_transactions = []

        for page_data in pages:
            page_num   = page_data["page_num"]
            text       = page_data["text"]
            char_count = page_data["char_count"]

            progress = 35 + int(
                (page_num / total_pages) * 50
            )
            self._update_progress(
                progress,
                f"🤖 Gemini analyse page "
                f"{page_num}/{total_pages} "
                f"(~{char_count//4} tokens)..."
            )

            if not text or char_count < 30:
                self.logger.warning(
                    f"Page {page_num} ignorée",
                    f"Texte trop court ({char_count} chars)"
                )
                continue

            transactions = self._call_gemini_text(
                text, f"{page_num}/{total_pages}"
            )
            self.logger.data(
                f"Résultat page {page_num}",
                f"{len(transactions)} transaction(s)"
            )
            all_transactions.extend(transactions)

            if page_num < total_pages:
                time.sleep(0.5)

        # ── Résultat ──────────────────────────────────────────
        self.logger.separator("RÉSULTAT FINAL")
        if not all_transactions:
            self.logger.error(
                "Aucune transaction extraite (mode Hybride)",
                "→ Essayez le mode Vision"
            )
            return self._empty_df()

        self._update_progress(88, "🔧 Construction tableau...")
        df = self._build_dataframe(all_transactions)
        self.logger.success(
            "Extraction Hybride terminée",
            f"{len(df)} lignes"
        )
        return df

    # ── Extraction texte pdfplumber ───────────────────────────

    def _extract_text_from_pdf(
        self, pdf_bytes: bytes
    ) -> list:
        """Extrait le texte de chaque page."""
        pages = []

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            total = len(pdf.pages)
            self.logger.info(
                f"PDF ouvert : {total} pages"
            )

            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                progress = 5 + int((i / total) * 25)
                self._update_progress(
                    progress,
                    f"📄 Extraction texte page "
                    f"{page_num}/{total}..."
                )

                # Méthode 1 : extract_text
                raw_text = None
                try:
                    raw_text = page.extract_text(
                        x_tolerance=3,
                        y_tolerance=3,
                        layout=True,
                    )
                except Exception as e:
                    self.logger.warning(
                        f"extract_text échoué p.{page_num}",
                        str(e)
                    )

                # Méthode 2 : extract_words
                if not raw_text or len(raw_text.strip()) < 50:
                    raw_text = self._words_to_text(page)

                # Méthode 3 : extract_tables
                if not raw_text or len(raw_text.strip()) < 50:
                    raw_text = self._tables_to_text(page)

                cleaned    = self._clean_text(raw_text or "")
                char_count = len(cleaned)

                self.logger.debug(
                    f"Page {page_num}",
                    f"{char_count} chars | "
                    f"'{cleaned[:120].replace(chr(10), ' | ')}'"
                )

                pages.append({
                    "page_num":    page_num,
                    "text":        cleaned,
                    "char_count":  char_count,
                    "total_pages": total,
                })

        return pages

    def _words_to_text(self, page) -> str:
        """Reconstruit le texte depuis les mots."""
        try:
            words = page.extract_words(
                x_tolerance=4, y_tolerance=4
            )
            if not words:
                return ""
            lines = {}
            for w in words:
                y = round(float(w["top"]) / 4) * 4
                lines.setdefault(y, []).append(w)
            result = []
            for y in sorted(lines):
                wds = sorted(lines[y], key=lambda w: w["x0"])
                result.append(
                    " ".join(w["text"] for w in wds)
                )
            return "\n".join(result)
        except Exception as e:
            self.logger.warning(
                "_words_to_text échoué", str(e)
            )
            return ""

    def _tables_to_text(self, page) -> str:
        """Convertit les tableaux en texte."""
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
                "_tables_to_text échoué", str(e)
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
        lines, result, prev_empty = text.split("\n"), [], False
        for line in lines:
            s = line.strip()
            if any(re.search(p, s.lower()) for p in skip):
                continue
            if not s:
                if not prev_empty:
                    result.append("")
                prev_empty = True
            else:
                result.append(s)
                prev_empty = False
        return "\n".join(result).strip()

    # ══════════════════════════════════════════════════════════
    # APPEL GEMINI TEXT (Hybride)
    # ══════════════════════════════════════════════════════════

    def _call_gemini_text(
        self, text: str, page_info: str = ""
    ) -> list:
        """Envoie du texte à Gemini et retourne les transactions."""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._cache:
            self.logger.debug(f"Cache hit — page {page_info}")
            return self._cache[cache_key]

        if len(text) > 8000:
            return self._call_gemini_text_chunked(
                text, page_info
            )

        user_prompt = (
            f"Voici le texte de la page {page_info} :\n\n"
            f"---\n{text}\n---\n\n"
            f"Retourne uniquement le JSON."
        )

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                self.logger.api(
                    f"Appel Gemini Text page {page_info} "
                    f"(tentative {attempt}/{max_retries})"
                )
                response = self.model.generate_content(
                    contents=[self.PROMPT_TEXT, user_prompt],
                    request_options={"timeout": 30},
                )

                raw = None
                try:
                    raw = response.text
                except Exception:
                    try:
                        raw = (
                            response.candidates[0]
                            .content.parts[0].text
                        )
                    except Exception:
                        pass

                if not raw:
                    self.logger.warning(
                        f"Réponse vide (text, page {page_info})"
                    )
                    continue

                self.logger.debug(
                    f"Réponse brute (page {page_info})",
                    f"'{raw[:300]}'"
                )

                transactions = self._parse_response(
                    raw, page_info
                )
                self._cache[cache_key] = transactions
                return transactions

            except Exception as e:
                err = str(e)
                self.logger.warning(
                    f"Erreur Gemini Text page {page_info} "
                    f"(tentative {attempt})",
                    err[:200]
                )
                if any(
                    c in err
                    for c in ["429", "quota", "RESOURCE_EXHAUSTED"]
                ):
                    wait = attempt * 15
                    time.sleep(wait)
                    continue
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
        """Traite un texte long par chunks."""
        chunks = self._split_chunks(text)
        result = []
        self.logger.info(
            f"Chunking : {len(chunks)} chunks "
            f"pour page {page_info}"
        )
        for i, chunk in enumerate(chunks, 1):
            trans = self._call_gemini_text(
                chunk, f"{page_info} chunk {i}/{len(chunks)}"
            )
            result.extend(trans)
            if i < len(chunks):
                time.sleep(0.5)
        return result

    def _split_chunks(
        self, text: str, max_chars: int = 5000
    ) -> list:
        """Découpe le texte en chunks sur les lignes vides."""
        if len(text) <= max_chars:
            return [text]
        chunks, current, curr_l = [], [], 0
        for line in text.split("\n"):
            ll = len(line) + 1
            if curr_l + ll > max_chars and current:
                chunks.append("\n".join(current))
                current, curr_l = [line], ll
            else:
                current.append(line)
                curr_l += ll
        if current:
            chunks.append("\n".join(current))
        return chunks

    # ══════════════════════════════════════════════════════════
    # PARSING JSON — Méthode centrale et robuste
    # ══════════════════════════════════════════════════════════

    def _parse_response(
        self, raw: str, page_info: str = ""
    ) -> list:
        """
        Parse la réponse Gemini de manière robuste.
        Gère tous les cas : JSON propre, JSON dans du texte,
        JSON malformé, réponse en langage naturel.
        """
        if not raw or not raw.strip():
            self.logger.error(
                f"Réponse vide — page {page_info}"
            )
            return []

        text = raw.strip()
        self.logger.debug(
            f"Parsing réponse (page {page_info})",
            f"Longueur totale : {len(text)} chars\n"
            f"Début : '{text[:200]}'\n"
            f"Fin   : '{text[-100:]}'"
        )

        # ── Étape 1 : Nettoyer les balises markdown ───────────
        text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```\s*",     "", text)
        text = text.strip()

        # ── Étape 2 : Essai de parse direct ──────────────────
        if text.startswith("{"):
            result = self._try_parse_json(text, page_info)
            if result is not None:
                return result

        # ── Étape 3 : Extraire le JSON avec regex ─────────────
        # Chercher {"transactions": [...]}
        pattern = r'\{[^{}]*"transactions"\s*:\s*\[.*?\]\s*\}'
        match   = re.search(pattern, text, re.DOTALL)
        if match:
            self.logger.debug(
                "JSON trouvé via regex (pattern transactions)"
            )
            result = self._try_parse_json(
                match.group(0), page_info
            )
            if result is not None:
                return result

        # ── Étape 4 : Extraire n'importe quel objet JSON ──────
        # (le plus grand bloc {} trouvé)
        all_matches = list(
            re.finditer(r"\{", text)
        )
        best_json   = None
        best_count  = 0

        for m in all_matches:
            start   = m.start()
            sub     = text[start:]
            # Trouver la fermeture correspondante
            depth   = 0
            end_pos = None
            for i, ch in enumerate(sub):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end_pos = i + 1
                        break
            if end_pos:
                candidate = sub[:end_pos]
                if len(candidate) > best_count:
                    best_count = len(candidate)
                    best_json  = candidate

        if best_json:
            self.logger.debug(
                "JSON extrait par analyse des accolades",
                f"Longueur : {len(best_json)} chars"
            )
            result = self._try_parse_json(
                best_json, page_info
            )
            if result is not None:
                return result

        # ── Étape 5 : Réparer et réessayer ────────────────────
        if best_json:
            repaired = self._repair_json(best_json)
            self.logger.debug(
                "Tentative réparation JSON",
                f"'{repaired[:200]}'"
            )
            result = self._try_parse_json(
                repaired, page_info, is_repair=True
            )
            if result is not None:
                return result

        # ── Étape 6 : Parser ligne par ligne ──────────────────
        # En dernier recours : essayer de construire les
        # transactions depuis le texte brut
        self.logger.warning(
            f"JSON invalide — tentative parsing ligne par ligne",
            f"Contenu complet :\n'{text[:600]}'"
        )
        transactions = self._parse_fallback(text)
        if transactions:
            self.logger.warning(
                f"Parsing fallback : {len(transactions)} "
                f"transactions récupérées"
            )
            return transactions

        self.logger.error(
            f"Impossible de parser la réponse "
            f"(page {page_info})",
            f"Contenu reçu :\n'{text[:500]}'"
        )
        return []

    def _try_parse_json(
        self,
        json_str:  str,
        page_info: str  = "",
        is_repair: bool = False,
    ):
        """
        Tente de parser un JSON.
        Retourne la liste de transactions ou None si échec.
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            if not is_repair:
                self.logger.debug(
                    f"JSON parse error",
                    f"Erreur : {e}\n"
                    f"Position : char {e.pos}\n"
                    f"Extrait : '{json_str[max(0,e.pos-30):e.pos+30]}'"
                )
            return None

        # Vérifier la structure
        if not isinstance(data, dict):
            self.logger.warning(
                f"JSON n'est pas un objet dict",
                f"Type : {type(data)}"
            )
            return None

        transactions = data.get("transactions")

        if transactions is None:
            # Peut-être que la clé est différente
            # Chercher une liste dans les valeurs
            for key, val in data.items():
                if isinstance(val, list):
                    self.logger.debug(
                        f"Clé 'transactions' non trouvée, "
                        f"utilisation de '{key}'"
                    )
                    transactions = val
                    break

        if transactions is None:
            self.logger.warning(
                "Clé 'transactions' introuvable dans le JSON",
                f"Clés disponibles : {list(data.keys())}"
            )
            return None

        if not isinstance(transactions, list):
            self.logger.warning(
                f"'transactions' n'est pas une liste",
                f"Type : {type(transactions)}"
            )
            return None

        if len(transactions) == 0:
            self.logger.info(
                f"JSON valide mais 0 transaction "
                f"(page {page_info})"
            )
            return []

        self.logger.success(
            f"JSON parsé avec succès",
            f"{len(transactions)} transaction(s) brutes"
        )

        # Normaliser
        validated = []
        for i, t in enumerate(transactions):
            norm = self._normalize(t)
            if norm:
                validated.append(norm)
            else:
                self.logger.debug(
                    f"Transaction {i+1} invalide/ignorée",
                    f"{str(t)[:150]}"
                )

        self.logger.data(
            "Transactions validées",
            f"{len(validated)} / {len(transactions)}"
        )
        return validated

    def _parse_fallback(self, text: str) -> list:
        """
        Dernier recours : essaye de construire des transactions
        depuis un texte non-JSON (réponse en langage naturel).
        """
        transactions = []
        lines        = text.split("\n")

        date_pattern = re.compile(
            r"(\d{2}/\d{2}/\d{4})"
        )
        amount_pattern = re.compile(
            r"\b(\d{3,}(?:\s\d{3})*)\b"
        )

        for line in lines:
            line = line.strip()
            if not line:
                continue

            date_match = date_pattern.search(line)
            if not date_match:
                continue

            amounts = [
                int(m.replace(" ", ""))
                for m in amount_pattern.findall(line)
                if int(m.replace(" ", "")) > 100
            ]

            if not amounts:
                continue

            # Extraire le libellé (texte entre date et montants)
            libelle = re.sub(
                r"\d{2}/\d{2}/\d{4}|\d{3,}(?:\s\d{3})*|\d+/",
                " ", line
            ).strip()
            libelle = re.sub(r"\s+", " ", libelle).strip()

            if not libelle:
                continue

            t = {
                "date":        date_match.group(1),
                "reference":   "",
                "libelle":     libelle,
                "date_valeur": "",
                "debit":       None,
                "credit":      float(amounts[-2])
                               if len(amounts) >= 2
                               else None,
                "solde":       float(amounts[-1])
                               if amounts else None,
            }
            transactions.append(t)

        return transactions

    def _repair_json(self, s: str) -> str:
        """Tente de réparer un JSON malformé."""
        s = re.sub(r",\s*}", "}", s)
        s = re.sub(r",\s*]", "]", s)
        s = s.replace("None",  "null")
        s = s.replace("True",  "true")
        s = s.replace("False", "false")
        # Remplacer les guillemets simples si pas de guillemets
        # doubles
        if '"' not in s and "'" in s:
            s = s.replace("'", '"')
        return s

    # ══════════════════════════════════════════════════════════
    # NORMALISATION
    # ══════════════════════════════════════════════════════════

    def _normalize(self, t: dict) -> dict:
        """Valide et normalise une transaction."""
        if not isinstance(t, dict):
            return None

        libelle = str(t.get("libelle", "") or "").strip()
        if not libelle or libelle.lower() in (
            "none", "null", ""
        ):
            return None

        return {
            "date":        self._fmt_date(t.get("date")),
            "reference":   str(
                t.get("reference", "") or ""
            ).strip(),
            "libelle":     libelle,
            "date_valeur": self._fmt_date(
                t.get("date_valeur")
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
        if val is None or str(val).lower() in ("null", "", "none"):
            return None
        try:
            s = re.sub(
                r"[^\d.]",
                "",
                str(val).replace(" ", "")
            )
            return float(s) if s else None
        except (ValueError, TypeError):
            return None

    # ══════════════════════════════════════════════════════════
    # CONSTRUCTION DU DATAFRAME
    # ══════════════════════════════════════════════════════════

    def _build_dataframe(
        self, transactions: list
    ) -> pd.DataFrame:
        """Construit et déduplique le DataFrame final."""
        if not transactions:
            return self._empty_df()

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

        # Séparer soldes et transactions normales
        mask_solde = df["Libellé"].str.contains(
            r"solde\s+d|solde\s+de\s+cl",
            case=False, na=False, regex=True,
        )
        df_soldes  = df[mask_solde]
        df_normal  = df[~mask_solde].drop_duplicates(
            subset=["Date", "Référence", "Libellé"],
            keep="first",
        )

        df_final = pd.concat(
            [df_soldes, df_normal], ignore_index=True
        )

        self.logger.data(
            "DataFrame construit",
            f"{len(df_final)} lignes "
            f"({len(df_soldes)} soldes + "
            f"{len(df_normal)} transactions)"
        )
        return df_final

    # ══════════════════════════════════════════════════════════
    # MÉTHODES PUBLIQUES
    # ══════════════════════════════════════════════════════════

    def _empty_df(self) -> pd.DataFrame:
        return pd.DataFrame(columns=[
            "Date", "Référence", "Libellé",
            "Date_Valeur", "Débit", "Crédit", "Solde",
        ])

    def get_debug_logs(self) -> str:
        return self.logger.get_logs_as_text()

    def get_debug_summary(self) -> dict:
        return self.logger.get_summary()

    def get_debug_entries(self) -> list:
        return self.logger.logs

    def _extract_text_from_pdf_public(
        self, pdf_bytes: bytes
    ) -> list:
        """Alias public pour l'estimation de tokens."""
        return self._extract_text_from_pdf(pdf_bytes)
