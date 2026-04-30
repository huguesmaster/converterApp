"""
╔══════════════════════════════════════════════════════════════╗
║         EXTRACTEUR GEMINI — Bank Statement Extractor         ║
║         Modes : Vision | Hybride — Configuration par Banque  ║
║         Version : 4.1 — Intégration complète BankConfig      ║
╚══════════════════════════════════════════════════════════════╝
"""

import google.generativeai as genai
import pdfplumber
import pandas as pd
import json
import re
import io
import time
import traceback
from PIL import Image
from typing import List, Dict, Optional

# Import optionnel pdf2image
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

from bank_configs import get_bank_config, BankConfig
# from cleaner import DataCleaner   # Sera importé dans app.py


class DebugLogger:
    """Logger structuré pour le débogage détaillé."""
    LEVELS = {
        "INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌",
        "DEBUG": "🔍", "STEP": "▶️", "DATA": "📊", "API": "🤖", "TOKEN": "🔤",
    }

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.logs = []
        self.errors = []
        self.warnings = []
        self._step = 0

    def _log(self, level: str, msg: str, detail: str = ""):
        icon = self.LEVELS.get(level, "•")
        ts = time.strftime("%H:%M:%S")
        entry = {
            "level": level,
            "icon": icon,
            "message": str(msg),
            "detail": str(detail),
            "timestamp": ts,
        }
        self.logs.append(entry)
        if self.verbose:
            print(f"[{ts}] {icon} {msg}")
            if detail:
                for line in str(detail).split("\n")[:5]:
                    if line.strip():
                        print(f"       {line}")

    def info(self, msg, detail=""): self._log("INFO", msg, detail)
    def success(self, msg, detail=""): self._log("SUCCESS", msg, detail)
    def warning(self, msg, detail=""):
        self._log("WARNING", msg, detail)
        self.warnings.append(msg)
    def error(self, msg, detail="", exc=None):
        if exc:
            detail += f"\n{type(exc).__name__}: {str(exc)}\n{traceback.format_exc()}"
        self._log("ERROR", msg, detail)
        self.errors.append({"msg": msg, "detail": detail})
    def debug(self, msg, detail=""): self._log("DEBUG", msg, detail)
    def step(self, msg):
        self._step += 1
        self._log("STEP", f"[Étape {self._step}] {msg}")
    def data(self, label, value):
        val_str = str(value)[:800] + ("..." if len(str(value)) > 800 else "")
        self._log("DATA", label, val_str)
    def api(self, msg, detail=""): self._log("API", msg, detail)

    def get_summary(self) -> dict:
        return {
            "total_logs": len(self.logs),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "steps": self._step,
            "has_errors": len(self.errors) > 0,
            "error_details": self.errors,
        }

    def get_logs_as_text(self) -> str:
        lines = []
        for log in self.logs:
            line = f"[{log['timestamp']}] {log['icon']} {log['message']}"
            if log.get("detail"):
                for dl in log["detail"].split("\n")[:4]:
                    if dl.strip():
                        line += f"\n    └─ {dl.strip()}"
            lines.append(line)
        return "\n".join(lines)


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
        self._cache: Dict[str, list] = {}

        self._configure_gemini()

    def _configure_gemini(self):
        self.logger.step("Configuration API Gemini")
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name="gemini-2.5-flash-lite",
                generation_config=genai.GenerationConfig(
                    temperature=0,
                    top_p=1,
                    top_k=1,
                    max_output_tokens=8192,
                ),
                safety_settings=[
                    {"category": cat, "threshold": "BLOCK_NONE"}
                    for cat in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                                "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
                ],
            )
            self.logger.success(
                "Gemini configuré avec succès",
                f"Banque: {self.banque_nom} | Mode: {self.mode}"
            )
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

**Colonnes attendues (gauche à droite)** :
- Date : {", ".join(c.col_date)}
- Référence : {", ".join(c.col_ref)}
- Libellé/Désignation (peut s'étendre sur plusieurs lignes) : {", ".join(c.col_libelle)}
- Date Valeur : {", ".join(c.col_date_valeur)}
- Débit : {", ".join(c.col_debit)}
- Crédit : {", ".join(c.col_credit)}
- Solde : {", ".join(c.col_solde)}

**Instructions spécifiques pour {c.nom}** :
{c.specific_instructions}

**Règles OBLIGATOIRES** :
1. **Fusion des libellés** : Combine les lignes consécutives qui n'ont ni date ni montant dans le champ "libelle".
2. Respecte **strictement** les colonnes Débit et Crédit. Ne jamais les inverser.
3. Montants : supprime tous les espaces et convertis en nombre (ex: "24 553 342" → 24553342). Utilise `null` pour les cellules vides.
4. Inclu les lignes de **Solde d'ouverture** et **Solde de clôture**.
5. Retourne **uniquement** le JSON suivant, sans aucun texte supplémentaire :

```json
{{
  "transactions": [
    {{
      "date": "JJ/MM/AAAA",
      "reference": "string ou vide",
      "libelle": "description complète fusionnée",
      "date_valeur": "JJ/MM/AAAA",
      "debit": null ou nombre,
      "credit": null ou nombre,
      "solde": nombre
    }}
  ]
}}
