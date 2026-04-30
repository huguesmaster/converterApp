"""
╔══════════════════════════════════════════════════════════════╗
║         EXTRACTEUR GEMINI — Bank Statement Extractor         ║
║         Modes : Vision | Hybride — Par Banque                ║
║         Version : 4.0.0 — Refactorisé avec configs banque    ║
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
from typing import List, Optional

# Import optionnel pdf2image
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

from bank_configs import get_bank_config, BankConfig
from cleaner import DataCleaner  # Assurez-vous que ce module existe toujours


class DebugLogger:
    # (Je garde votre classe DebugLogger inchangée pour minimiser les risques)
    # ... Copiez-collez ici votre classe DebugLogger existante complète ...
    # (Pour gagner de l'espace, je l'omets dans cette réponse, mais elle doit rester identique)

    # Note : Copiez votre classe DebugLogger originale ici


class GeminiExtractor:
    def __init__(
        self,
        api_key: str,
        mode: str = "vision",
        banque_nom: str = "Autre banque",
        progress_callback=None,
        verbose_debug: bool = True,
    ):
        self.api_key = api_key
        self.mode = mode
        self.banque_nom = banque_nom
        self.config: BankConfig = get_bank_config(banque_nom)
        self.progress_callback = progress_callback
        self.logger = DebugLogger(verbose=verbose_debug)
        self._cache = {}

        self._configure_gemini()

    def _configure_gemini(self):
        self.logger.step("Configuration API Gemini")
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name="gemini-2.5-flash-lite",   # ou gemini-1.5-flash selon votre quota
                generation_config=genai.GenerationConfig(
                    temperature=0,
                    top_p=1,
                    top_k=1,
                    max_output_tokens=8192,
                ),
                safety_settings=[{"category": cat, "threshold": "BLOCK_NONE"} 
                               for cat in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                                          "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]],
            )
            self.logger.success("Gemini configuré", f"Modèle: gemini-2.5-flash-lite | Banque: {self.banque_nom} | Mode: {self.mode}")
        except Exception as e:
            self.logger.error("Échec configuration Gemini", exc=e)
            raise

    def _update_progress(self, step: int, msg: str):
        if self.progress_callback:
            self.progress_callback(step, msg)

    # ====================== PROMPT DYNAMIQUE PAR BANQUE ======================
    def _build_prompt(self, is_vision: bool = True) -> str:
        c = self.config
        prompt = f"""Tu es un expert comptable spécialisé dans les relevés bancaires camerounais de **{c.nom}**.

**Colonnes typiques (de gauche à droite)** :
- Date opération → {", ".join(c.col_date)}
- Référence → {", ".join(c.col_ref)}
- Libellé / Description (peut être sur plusieurs lignes) → {", ".join(c.col_libelle)}
- Date Valeur → {", ".join(c.col_date_valeur)}
- Débit → {", ".join(c.col_debit)}
- Crédit → {", ".join(c.col_credit)}
- Solde → {", ".join(c.col_solde)}

**Instructions spécifiques à {c.nom}** :
{c.specific_instructions}

**Règles strictes** :
1. Fusionne intelligemment les libellés multi-lignes (lignes suivantes sans montant ni date).
2. Ne jamais inverser Débit et Crédit — respecte strictement la position des colonnes.
3. Montants : supprime tous les espaces et convertis en nombre entier (ex: "1 234 567" → 1234567). Utilise `null` pour les cellules vides.
4. Inclu les lignes de **Solde d'ouverture** et **Solde de clôture**.
5. Retourne **uniquement** du JSON valide, sans aucun texte supplémentaire :

{{
  "transactions": [
    {{
      "date": "JJ/MM/AAAA",
      "reference": "string",
      "libelle": "description complète fusionnée",
      "date_valeur": "JJ/MM/AAAA",
      "debit": null ou nombre,
      "credit": null ou nombre,
      "solde": nombre
    }}
  ]
}}
"""

        if not is_vision:
            prompt += "\nLe texte fourni peut contenir du bruit (en-têtes, pieds de page, numéros de page). Ignore-les et concentre-toi sur le tableau des transactions."
        return prompt

    # ====================== EXTRACTION ======================
    def extract(self, pdf_bytes: bytes) -> pd.DataFrame:
        self.logger.separator(f"DÉBUT EXTRACTION — {self.banque_nom} — Mode {self.mode.upper()}")
        if self.mode == "hybrid":
            return self._extract_hybrid(pdf_bytes)
        return self._extract_vision(pdf_bytes)

    # --------------------- MODE VISION ---------------------
    def _extract_vision(self, pdf_bytes: bytes) -> pd.DataFrame:
        if not PDF2IMAGE_AVAILABLE:
            self.logger.error("pdf2image non disponible")
            return self._empty_df()

        self._update_progress(5, "🖼️ Conversion PDF → Images...")
        try:
            images = convert_from_bytes(pdf_bytes, dpi=250, fmt="PNG")
            self.logger.success(f"{len(images)} page(s) convertie(s)")
        except Exception as e:
            self.logger.error("Échec conversion PDF→Images", exc=e)
            return self._empty_df()

        prompt = self._build_prompt(is_vision=True)
        all_transactions = []

        for idx, image in enumerate(images):
            page_num = idx + 1
            progress = 10 + int((page_num / len(images)) * 75)
            self._update_progress(progress, f"🤖 Vision — Page {page_num}/{len(images)}")

            transactions = self._call_vision_single_page(image, page_num, prompt)
            all_transactions.extend(transactions)

            if page_num < len(images):
                time.sleep(1.2)

        return self._build_and_validate_dataframe(all_transactions)

    def _call_vision_single_page(self, image: Image.Image, page_num: int, prompt: str) -> list:
        # Optimisation image + retry logic (votre code existant amélioré)
        # ... (gardez votre logique _optimize_image et retry existante)
        # Remplacez simplement le prompt statique par le prompt dynamique passé en paramètre
        # Je vous laisse adapter cette méthode avec votre code de retry robuste

        # Exemple simplifié :
        optimized = self._optimize_image(image)
        try:
            response = self.model.generate_content([prompt, optimized])
            raw_text = response.text if hasattr(response, 'text') else ""
            return self._parse_response(raw_text, f"page {page_num}")
        except Exception as e:
            self.logger.error(f"Erreur Vision page {page_num}", exc=e)
            return []

    # --------------------- MODE HYBRIDE ---------------------
    def _extract_hybrid(self, pdf_bytes: bytes) -> pd.DataFrame:
        # Votre logique existante de _extract_text_from_pdf + appel Gemini Text
        # Remplacez simplement le PROMPT_TEXT par self._build_prompt(is_vision=False)
        # ... (adaptez votre code existant)

        # Pour l'instant, je recommande de réutiliser votre _extract_hybrid et injecter le prompt dynamique.
        # Si vous voulez la version complète, dites-le-moi.

        self.logger.warning("Mode Hybride non encore fully refactorisé dans cette version. Utilisez Vision en priorité.")
        return self._extract_vision(pdf_bytes)  # fallback temporaire

    # ====================== PARSING & VALIDATION ======================
    def _parse_response(self, raw: str, context: str = "") -> list:
        # Votre méthode _parse_response robuste existante (regex, repair, fallback)
        # Elle reste très bonne — gardez-la telle quelle
        # ... (copiez votre implémentation existante ici)
        pass  # Remplacez par votre code existant

    def _build_and_validate_dataframe(self, transactions: List[dict]) -> pd.DataFrame:
        if not transactions:
            return self._empty_df()

        df = self._build_dataframe(transactions)   # votre méthode existante

        # Validation du solde final
        try:
            if not df.empty and "Solde" in df.columns:
                df["Solde"] = pd.to_numeric(df["Solde"], errors='coerce')
                last_solde_pdf = df["Solde"].iloc[-1]

                # Calcul cumulatif approximatif
                df["delta"] = df["Crédit"].fillna(0) - df["Débit"].fillna(0)
                calculated_final = df["delta"].cumsum().iloc[-1] + (df["Solde"].iloc[0] if pd.notna(df["Solde"].iloc[0]) else 0)

                if abs(calculated_final - last_solde_pdf) > 10:   # seuil tolérance
                    self.logger.warning(
                        "Discrepancy solde final détecté",
                        f"PDF: {last_solde_pdf:,.0f} | Calculé: {calculated_final:,.0f}"
                    )
        except Exception as e:
            self.logger.warning("Validation solde échouée", str(e))

        return df

    def _build_dataframe(self, transactions: list) -> pd.DataFrame:
        # Votre logique existante de construction + déduplication + séparation soldes
        # Gardez-la ou améliorez-la légèrement
        rows = [{**t} for t in transactions]
        df = pd.DataFrame(rows)
        # ... votre post-processing existant
        return df

    def _empty_df(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["Date", "Référence", "Libellé", "Date_Valeur", "Débit", "Crédit", "Solde"])

    # Méthodes publiques (inchangées)
    def get_debug_logs(self): 
        return self.logger.get_logs_as_text()

    def get_debug_summary(self):
        return self.logger.get_summary()

    def get_debug_entries(self):
        return self.logger.logs

    def _extract_text_from_pdf_public(self, pdf_bytes: bytes):
        return self._extract_text_from_pdf(pdf_bytes)  # si vous avez cette méthode
