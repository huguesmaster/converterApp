"""
╔══════════════════════════════════════════════════════════════╗
║         EXTRACTEUR GEMINI — Bank Statement Extractor         ║
║         Modes : Vision | Hybride — Configuration par Banque  ║
║         Version : 4.0.1 — Robustesse & Prompt Dynamique      ║
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
from typing import List, Dict, Optional, Any

# Import optionnel pdf2image
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

from bank_configs import get_bank_config, BankConfig


# ══════════════════════════════════════════════════════════════
# DEBUG LOGGER
# ══════════════════════════════════════════════════════════════
class DebugLogger:
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
        entry = {"level": level, "icon": icon, "message": str(msg), "detail": str(detail), "timestamp": ts}
        self.logs.append(entry)
        if self.verbose:
            print(f"[{ts}] {icon} {msg}")
            if detail:
                for line in str(detail).split("\n")[:6]:
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
        lines = [f"[{log['timestamp']}] {log['icon']} {log['message']}" +
                 (f"\n    └─ {dl.strip()}" for dl in log.get("detail", "").split("\n")[:4] if dl.strip())
                 for log in self.logs]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# GEMINI EXTRACTOR - VERSION ROBUSTE
# ══════════════════════════════════════════════════════════════
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
                    temperature=0, top_p=1, top_k=1, max_output_tokens=8192,
                ),
                safety_settings=[{"category": cat, "threshold": "BLOCK_NONE"}
                                 for cat in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                                             "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]],
            )
            self.logger.success("Gemini configuré", f"Banque: {self.banque_nom} | Mode: {self.mode}")
        except Exception as e:
            self.logger.error("Échec configuration Gemini", exc=e)
            raise

    def _update_progress(self, step: int, msg: str):
        if self.progress_callback:
            self.progress_callback(step, msg)

    # ====================== PROMPT DYNAMIQUE ======================
    def _build_prompt(self, is_vision: bool = True) -> str:
        c = self.config
        prompt = f"""Tu es un expert comptable spécialisé dans les relevés bancaires camerounais de **{c.nom}**.

**Colonnes attendues** (gauche à droite) :
- Date : {", ".join(c.col_date)}
- Référence : {", ".join(c.col_ref)}
- Libellé (peut être multi-lignes) : {", ".join(c.col_libelle)}
- Date Valeur : {", ".join(c.col_date_valeur)}
- Débit : {", ".join(c.col_debit)}
- Crédit : {", ".join(c.col_credit)}
- Solde : {", ".join(c.col_solde)}

**Instructions spécifiques pour {c.nom}**:
{c.specific_instructions}

**Règles STRICTES**:
1. Fusionne les libellés sur plusieurs lignes consécutives quand il n'y a pas de montant.
2. Respecte strictement la colonne Débit et Crédit — ne jamais les inverser.
3. Montants : supprime tous les espaces → "24 553 342" devient 24553342. Utilise `null` pour les cellules vides.
4. Inclu les lignes "Solde d'ouverture" et "Solde de clôture".
5. Retourne **uniquement** ce JSON valide :

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
            prompt += "\n\nLe texte fourni est extrait via pdfplumber et peut contenir du bruit (en-têtes, numéros de page, pieds de page). Ignore tout ce qui n'est pas dans le tableau des transactions."
        return prompt

    # ====================== POINT D'ENTRÉE ======================
    def extract(self, pdf_bytes: bytes) -> pd.DataFrame:
        self.logger.separator(f"DÉBUT EXTRACTION — {self.banque_nom.upper()} — Mode {self.mode.upper()}")
        self.logger.info("Taille du PDF", f"{len(pdf_bytes):,} bytes")

        if self.mode == "hybrid":
            return self._extract_hybrid(pdf_bytes)
        return self._extract_vision(pdf_bytes)

    # ====================== MODE VISION ======================
    def _extract_vision(self, pdf_bytes: bytes) -> pd.DataFrame:
        if not PDF2IMAGE_AVAILABLE:
            self.logger.error("pdf2image non installé. pip install pdf2image + poppler-utils")
            return self._empty_df()

        self._update_progress(5, "🖼️ Conversion PDF en images...")
        try:
            images = convert_from_bytes(pdf_bytes, dpi=250, fmt="PNG", thread_count=1)
            self.logger.success(f"{len(images)} page(s) convertie(s)")
        except Exception as e:
            self.logger.error("Échec conversion PDF → Images", exc=e)
            return self._empty_df()

        prompt = self._build_prompt(is_vision=True)
        all_transactions: List[Dict] = []

        for idx, image in enumerate(images, 1):
            progress = 10 + int((idx / len(images)) * 75)
            self._update_progress(progress, f"🤖 Gemini Vision — Page {idx}/{len(images)}")

            transactions = self._call_vision_single_page(image, idx, prompt)
            all_transactions.extend(transactions)

            if idx < len(images):
                time.sleep(1.3)

        return self._build_and_validate_dataframe(all_transactions)

    def _call_vision_single_page(self, image: Image.Image, page_num: int, prompt: str) -> List[Dict]:
        optimized = self._optimize_image(image)
        max_retries = 3

        for attempt in range(1, max_retries + 1):
            try:
                self.logger.api(f"Appel Gemini Vision - Page {page_num} (tentative {attempt})")
                response = self.model.generate_content(
                    [prompt, optimized],
                    request_options={"timeout": 120}
                )

                raw_text = ""
                if hasattr(response, "text") and response.text:
                    raw_text = response.text
                else:
                    try:
                        raw_text = response.candidates[0].content.parts[0].text
                    except:
                        pass

                if raw_text:
                    self.logger.debug(f"Réponse brute page {page_num}", raw_text[:600])
                    return self._parse_response(raw_text, f"page {page_num}")

            except Exception as e:
                err_str = str(e)
                self.logger.warning(f"Erreur page {page_num} (tentative {attempt})", err_str[:250])

                if any(x in err_str for x in ["429", "RESOURCE_EXHAUSTED", "quota"]):
                    wait = attempt * 25
                    self.logger.warning(f"Rate limit détecté → pause {wait}s")
                    time.sleep(wait)
                    continue

                if attempt == max_retries:
                    self.logger.error(f"Échec définitif sur la page {page_num}", exc=e)
                time.sleep(2)

        return []

    def _optimize_image(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        if image.width > 2000:
            ratio = 2000 / image.width
            new_h = int(image.height * ratio)
            image = image.resize((2000, new_h), Image.LANCZOS)
        return image

    # ====================== MODE HYBRIDE ======================
    def _extract_hybrid(self, pdf_bytes: bytes) -> pd.DataFrame:
        self.logger.warning("Mode Hybride non pleinement implémenté → basculement vers Vision")
        return self._extract_vision(pdf_bytes)

    # ====================== PARSING ROBUSTE ======================
    def _parse_response(self, raw: str, context: str = "") -> List[Dict]:
        if not raw or not raw.strip():
            self.logger.warning("Réponse vide", context)
            return []

        text = raw.strip()
        # Nettoyage des balises markdown
        text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```", "", text).strip()

        # Tentative 1 : Parse JSON direct
        try:
            data = json.loads(text)
            transactions = data.get("transactions") or []
            if isinstance(transactions, list):
                normalized = [self._normalize(t) for t in transactions if self._normalize(t)]
                self.logger.success(f"JSON parsé avec succès ({len(normalized)} transactions)", context)
                return normalized
        except json.JSONDecodeError as e:
            self.logger.debug("JSON direct échoué", str(e))

        # Tentative 2 : Extraction regex d'un bloc JSON
        json_match = re.search(r'\{.*"transactions"\s*:\s*\[.*?\]\s*\}', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                transactions = data.get("transactions") or []
                return [self._normalize(t) for t in transactions if self._normalize(t)]
            except:
                pass

        self.logger.warning(f"Échec du parsing JSON pour {context}")
        return []

    def _normalize(self, t: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(t, dict):
            return None

        libelle = str(t.get("libelle", "")).strip()
        if not libelle or libelle.lower() in ("none", "null", ""):
            return None

        return {
            "date": self._fmt_date(t.get("date")),
            "reference": str(t.get("reference", "")).strip(),
            "libelle": libelle,
            "date_valeur": self._fmt_date(t.get("date_valeur")),
            "debit": self._fmt_amount(t.get("debit")),
            "credit": self._fmt_amount(t.get("credit")),
            "solde": self._fmt_amount(t.get("solde")),
        }

    def _fmt_date(self, val) -> str:
        if not val:
            return ""
        s = str(val).strip()
        return s[:10] if re.match(r"\d{2}/\d{2}/\d{4}", s) else s

    def _fmt_amount(self, val) -> Optional[float]:
        if val is None or str(val).lower() in ("null", "none", "", "0"):
            return None
        try:
            cleaned = re.sub(r"[^\d.]", "", str(val).replace(" ", ""))
            return float(cleaned) if cleaned else None
        except:
            return None

    # ====================== CONSTRUCTION & VALIDATION ======================
    def _build_and_validate_dataframe(self, transactions: List[Dict]) -> pd.DataFrame:
        if not transactions:
            self.logger.error("Aucune transaction extraite")
            return self._empty_df()

        df = pd.DataFrame([{
            "Date": t["date"],
            "Référence": t["reference"],
            "Libellé": t["libelle"],
            "Date_Valeur": t["date_valeur"],
            "Débit": t["debit"],
            "Crédit": t["credit"],
            "Solde": t["solde"],
        } for t in transactions])

        self.logger.success(f"DataFrame final construit : {len(df)} lignes")

        # Validation du solde final
        try:
            if not df.empty and "Solde" in df.columns:
                last_solde = pd.to_numeric(df["Solde"].iloc[-1], errors="coerce")
                df["delta"] = df["Crédit"].fillna(0) - df["Débit"].fillna(0)
                calculated_final = df["delta"].cumsum().iloc[-1]

                if pd.notna(last_solde) and abs(calculated_final - last_solde) > 100:
                    self.logger.warning(
                        "Discrepancy solde final détecté",
                        f"PDF: {last_solde:,.0f} | Calculé: {calculated_final:,.0f}"
                    )
        except Exception as e:
            self.logger.warning("Validation solde échouée", str(e))

        return df

    def _empty_df(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["Date", "Référence", "Libellé", "Date_Valeur", "Débit", "Crédit", "Solde"])

    # Méthodes publiques
    def get_debug_logs(self) -> str:
        return self.logger.get_logs_as_text()

    def get_debug_summary(self) -> dict:
        return self.logger.get_summary()

    def get_debug_entries(self) -> list:
        return self.logger.logs

    def _extract_text_from_pdf_public(self, pdf_bytes: bytes) -> list:
        """Pour estimation tokens"""
        return []
