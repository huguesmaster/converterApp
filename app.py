"""
Application OCR pour relevés bancaires camerounais
Banques supportées : UNICS, ADVANS, MUPECI, FINANCIAL HOUSE, CEPAC
"""

import streamlit as st
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import re
import io
import os
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURATION TESSERACT (Linux / Streamlit Cloud)
# ─────────────────────────────────────────────
TESSERACT_PATH = "/usr/bin/tesseract"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ─────────────────────────────────────────────
# REGEX CENTRALE – Dates ultra-flexibles
# Formats couverts :
#   02/Jan/2025  |  02/01/2025  |  02-01-2025
#   02 Jan 2025  |  02.01.2025  |  2/1/2025
# ─────────────────────────────────────────────
MONTH_MAP = {
    "jan": "01", "feb": "02", "fev": "02", "fév": "02",
    "mar": "03", "mars": "03", "apr": "04", "avr": "04",
    "may": "05", "mai": "05", "jun": "06", "juin": "06",
    "jul": "07", "juil": "07", "aug": "08", "aoû": "08", "aou": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12", "déc": "12",
}

DATE_PATTERN = re.compile(
    r"""
    \b
    (\d{1,2})           # jour
    [\/\-\.\s]          # séparateur
    ([A-Za-zÀ-ÿ]{3}|\d{1,2})   # mois (texte 3 lettres ou chiffres)
    [\/\-\.\s]?         # séparateur optionnel
    (\d{2,4})           # année
    \b
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Regex montants : gère (2,500.00) → négatif, 1 234,50, 1,234.50
AMOUNT_PATTERN = re.compile(
    r"""
    (\([\d\s,\.]+\))    # montant entre parenthèses ex: (2,500.00)
    |
    (-?\d[\d\s]*[,\.]\d{2,3}(?:[,\.]\d{2})?)  # montant standard
    """,
    re.VERBOSE,
)


# ─────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────

def normalize_date(day: str, month: str, year: str) -> str:
    """Normalise une date extraite en format DD/MM/YYYY."""
    month = month.strip().lower()
    if month in MONTH_MAP:
        month_num = MONTH_MAP[month]
    elif month.isdigit():
        month_num = month.zfill(2)
    else:
        return None

    year = year.strip()
    if len(year) == 2:
        year = "20" + year

    try:
        dt = datetime(int(year), int(month_num), int(day.strip()))
        return dt.strftime("%d/%m/%Y")
    except ValueError:
        return None


def extract_date(text: str):
    """Retourne (date_normalisée, match_object) ou (None, None)."""
    for m in DATE_PATTERN.finditer(text):
        day, month, year = m.group(1), m.group(2), m.group(3)
        normalized = normalize_date(day, month, year)
        if normalized:
            return normalized, m
    return None, None


def parse_amount(raw: str) -> float | None:
    """
    Convertit un montant textuel en float.
    (2,500.00) → -2500.0
    1 234,50   → 1234.50
    """
    if not raw:
        return None
    raw = raw.strip()

    negative = False
    if raw.startswith("(") and raw.endswith(")"):
        negative = True
        raw = raw[1:-1]

    # Supprime espaces et détecte le séparateur décimal
    raw = re.sub(r"\s", "", raw)

    # Format anglo-saxon : 1,234.56 → supprimer virgules
    if re.search(r"\d,\d{3}[,\.]", raw) or raw.count(",") > 1:
        raw = raw.replace(",", "")
    # Format français : 1234,56 → remplacer virgule par point
    elif "," in raw and "." not in raw:
        raw = raw.replace(",", ".")
    # Format mixte : 1.234,56
    elif "." in raw and "," in raw and raw.index(".") < raw.index(","):
        raw = raw.replace(".", "").replace(",", ".")

    try:
        value = float(raw)
        return -abs(value) if negative else value
    except ValueError:
        return None


def find_amounts_in_text(text: str) -> list[float]:
    """Extrait tous les montants d'une chaîne, retourne une liste de floats."""
    amounts = []
    for m in AMOUNT_PATTERN.finditer(text):
        raw = m.group(0)
        val = parse_amount(raw)
        if val is not None:
            amounts.append(val)
    return amounts


# ─────────────────────────────────────────────
# PRÉTRAITEMENT IMAGE
# ─────────────────────────────────────────────

def preprocess_image(pil_image: Image.Image) -> Image.Image:
    """
    Pipeline de prétraitement pour améliorer l'OCR :
    1. Conversion en niveaux de gris
    2. Augmentation du contraste
    3. Légère accentuation des bords
    4. Binarisation adaptive via seuillage
    """
    # 1. Niveaux de gris
    img = pil_image.convert("L")

    # 2. Contraste
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)

    # 3. Netteté
    img = img.filter(ImageFilter.SHARPEN)

    # 4. Upscale si la résolution est faible (aide Tesseract)
    w, h = img.size
    if w < 1500:
        scale = 1500 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    return img


# ─────────────────────────────────────────────
# EXTRACTION OCR
# ─────────────────────────────────────────────

def ocr_pdf(pdf_bytes: bytes, lang: str = "fra+eng") -> list[str]:
    """
    Convertit chaque page du PDF en image, applique le prétraitement,
    puis extrait le texte via Tesseract.
    Retourne une liste de chaînes (une par page).
    """
    pages_text = []
    try:
        images = convert_from_bytes(pdf_bytes, dpi=300)
    except Exception as e:
        st.error(f"Erreur conversion PDF→Image : {e}")
        return []

    progress = st.progress(0, text="OCR en cours…")
    for i, img in enumerate(images):
        processed = preprocess_image(img)
        config = "--oem 3 --psm 6"
        text = pytesseract.image_to_string(processed, lang=lang, config=config)
        pages_text.append(text)
        progress.progress((i + 1) / len(images), text=f"Page {i+1}/{len(images)} traitée")

    progress.empty()
    return pages_text


# ─────────────────────────────────────────────
# PARSEUR – Machine à états
# ─────────────────────────────────────────────

def parse_transactions(pages_text: list[str], bank: str) -> list[dict]:
    """
    Algorithme principal : machine à états avec ancrage par date.

    État A – Attente : on cherche une ligne avec une date valide.
    État B – Transaction ouverte : on fusionne les lignes orphelines
             (sans date) dans le libellé de la transaction courante.

    Gère :
      - Libellés multi-lignes (3-4 lignes)
      - Montants entre parenthèses (MUPECI)
      - Dates avec abréviations anglaises (UNICS)
      - Séparateurs de milliers en virgules (ADVANS)
    """
    transactions = []
    current = None  # transaction en cours de construction

    def flush(tx):
        """Finalise et enregistre la transaction courante."""
        if tx is None:
            return
        tx["libelle"] = " ".join(tx["libelle_parts"]).strip()
        tx.pop("libelle_parts", None)

        # Si montant non encore trouvé, tenter dans le libellé complet
        if tx.get("debit") is None and tx.get("credit") is None:
            amounts = find_amounts_in_text(tx["libelle"])
            if amounts:
                if amounts[0] < 0:
                    tx["debit"] = amounts[0]
                else:
                    tx["credit"] = amounts[0] if len(amounts) >= 1 else None
                    tx["debit"] = amounts[1] if len(amounts) >= 2 else None

        transactions.append(tx)

    for page_text in pages_text:
        lines = page_text.splitlines()

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # ── Détection de date ──
            date_str, date_match = extract_date(line_stripped)

            if date_str:
                # Nouvelle transaction → on flush la précédente
                flush(current)

                # Texte de la ligne sans la date
                remainder = line_stripped[date_match.end():].strip() if date_match else line_stripped

                # Extraction montants sur la même ligne
                amounts = find_amounts_in_text(remainder)
                debit = None
                credit = None

                # Nettoyage du libellé : retirer les montants trouvés
                libelle_clean = AMOUNT_PATTERN.sub("", remainder).strip()

                if bank == "MUPECI":
                    # MUPECI : (x,xxx.xx) = débit, sinon crédit
                    for a in amounts:
                        if a < 0:
                            debit = a
                        else:
                            credit = a
                else:
                    # Logique générique : 1er montant=débit, 2ème=crédit
                    # (à ajuster si la banque met crédit en premier)
                    if len(amounts) >= 1:
                        debit = amounts[0] if amounts[0] < 0 else None
                        credit = amounts[0] if amounts[0] >= 0 else None
                    if len(amounts) >= 2:
                        credit = amounts[1]

                current = {
                    "date": date_str,
                    "libelle_parts": [libelle_clean] if libelle_clean else [],
                    "debit": debit,
                    "credit": credit,
                    "solde": None,
                    "banque": bank,
                }

            else:
                # ── Ligne orpheline ──
                if current is not None:
                    # On ignore les lignes qui semblent être des entêtes
                    if re.match(r"^(date|libelle|montant|debit|credit|solde|balance|ref)", line_stripped, re.I):
                        continue

                    # Tentative d'extraction de montant orphelin
                    amounts = find_amounts_in_text(line_stripped)
                    if amounts and current.get("debit") is None and current.get("credit") is None:
                        for a in amounts:
                            if a < 0:
                                current["debit"] = a
                            else:
                                current["credit"] = a
                        # On n'ajoute pas ce montant au libellé
                    else:
                        # Texte pur → fusion au libellé
                        clean = AMOUNT_PATTERN.sub("", line_stripped).strip()
                        if clean:
                            current["libelle_parts"].append(clean)

    # Flush dernière transaction
    flush(current)
    return transactions


# ─────────────────────────────────────────────
# EXPORT EXCEL
# ─────────────────────────────────────────────

def to_excel(df: pd.DataFrame) -> bytes:
    """Génère un fichier Excel en mémoire."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Transactions")
        ws = writer.sheets["Transactions"]
        # Largeurs automatiques
        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 60)
    return buffer.getvalue()


# ─────────────────────────────────────────────
# INTERFACE STREAMLIT
# ─────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="OCR Relevés Bancaires – Cameroun",
        page_icon="🏦",
        layout="wide",
    )

    st.title("🏦 OCR Relevés Bancaires Cameroun")
    st.caption("UNICS · ADVANS · MUPECI · FINANCIAL HOUSE · CEPAC")

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Paramètres")
        bank = st.selectbox(
            "Banque",
            ["UNICS", "ADVANS", "MUPECI", "FINANCIAL HOUSE", "CEPAC"],
        )
        lang = st.selectbox(
            "Langue OCR",
            ["fra+eng", "fra", "eng"],
            help="fra+eng recommandé pour les relevés mixtes",
        )
        diagnostic_mode = st.toggle(
            "🔬 Mode Diagnostic",
            value=False,
            help="Affiche le texte brut extrait par l'OCR pour déboguer",
        )
        st.divider()
        st.markdown("**Format de dates attendu**")
        st.code("02/Jan/2025\n02/01/2025\n02-01-2025\n02 Jan 2025")
        st.markdown("**Montants négatifs (MUPECI)**")
        st.code("(2,500.00) → -2500.0")

    # ── Zone d'upload ──
    uploaded_files = st.file_uploader(
        "📂 Déposez vos relevés PDF",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("⬆️ Uploadez un ou plusieurs fichiers PDF pour commencer.")
        return

    if st.button("🚀 Lancer l'extraction OCR", type="primary", use_container_width=True):
        all_transactions = []
        raw_texts_debug = {}

        for uploaded_file in uploaded_files:
            st.subheader(f"📄 {uploaded_file.name}")
            pdf_bytes = uploaded_file.read()

            with st.spinner(f"Extraction OCR de {uploaded_file.name}…"):
                pages_text = ocr_pdf(pdf_bytes, lang=lang)

            if not pages_text:
                st.error("❌ Impossible de lire le PDF.")
                continue

            full_text = "\n".join(pages_text)
            raw_texts_debug[uploaded_file.name] = full_text

            # ── Mode Diagnostic ──
            if diagnostic_mode:
                with st.expander(f"🔬 Texte brut OCR – {uploaded_file.name}", expanded=False):
                    st.text_area(
                        "Texte extrait (brut)",
                        full_text,
                        height=400,
                        key=f"debug_{uploaded_file.name}",
                    )

            # ── Parsing ──
            transactions = parse_transactions(pages_text, bank)

            if not transactions:
                st.warning(
                    f"⚠️ Aucune transaction détectée dans **{uploaded_file.name}**.\n\n"
                    "Activez le **Mode Diagnostic** pour inspecter le texte brut OCR "
                    "et vérifier que les dates sont bien reconnues."
                )
            else:
                st.success(f"✅ {len(transactions)} transaction(s) détectée(s)")
                all_transactions.extend(transactions)

        # ── Résultats globaux ──
        if all_transactions:
            df = pd.DataFrame(all_transactions, columns=["date", "libelle", "debit", "credit", "solde", "banque"])
            df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
            df.sort_values("date", inplace=True)
            df["date"] = df["date"].dt.strftime("%d/%m/%Y")

            st.divider()
            st.subheader(f"📊 {len(df)} transactions extraites au total")

            # Métriques rapides
            col1, col2, col3 = st.columns(3)
            total_debit = df["debit"].sum()
            total_credit = df["credit"].sum()
            col1.metric("Total Débits", f"{total_debit:,.0f} FCFA")
            col2.metric("Total Crédits", f"{total_credit:,.0f} FCFA")
            col3.metric("Balance nette", f"{total_credit + total_debit:,.0f} FCFA")

            st.dataframe(df, use_container_width=True, hide_index=True)

            # ── Export Excel ──
            excel_bytes = to_excel(df)
            st.download_button(
                label="⬇️ Télécharger le fichier Excel",
                data=excel_bytes,
                file_name=f"releves_extraits_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True,
            )
        elif uploaded_files:
            st.error(
                "❌ Aucune transaction n'a pu être extraite de l'ensemble des fichiers.\n\n"
                "**Conseils :**\n"
                "- Activez le Mode Diagnostic pour voir le texte brut\n"
                "- Vérifiez que le PDF est un scan lisible (pas trop incliné)\n"
                "- Essayez de changer la langue OCR\n"
                "- Assurez-vous que Tesseract est bien installé (`packages.txt`)"
            )


if __name__ == "__main__":
    main()
