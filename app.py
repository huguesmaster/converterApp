import streamlit as st
import pdfplumber
import pandas as pd
import re
import io
from pdf2image import convert_from_bytes
import pytesseract

# Configuration Tesseract
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

st.set_page_config(page_title="Extracteur Bancaire Pro", layout="wide")

st.title("🏦 Extracteur de Relevés (Version Ultra-Robuste)")

def clean_amount(val):
    if val is None or str(val).strip() == "": return 0.0
    s = str(val).strip()
    is_negative = '(' in s or '-' in s
    # Garde uniquement les chiffres, les points et les virgules
    s = re.sub(r'[^\d.,]', '', s)
    if ',' in s and '.' in s: s = s.replace(',', '')
    elif ',' in s: s = s.replace(',', '.')
    try:
        num = float(s)
        return -num if is_negative else num
    except:
        return 0.0

def process_pdf(file_bytes):
    all_data = []
    images = convert_from_bytes(file_bytes)
    progress_bar = st.progress(0)
    
    # On stocke un échantillon du texte pour le diagnostic
    debug_text = ""

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            # 1. OCR amélioré (Noir et blanc + contraste)
            img = images[i].convert('L')
            text_ocr = pytesseract.image_to_string(img, lang='fra')
            if i == 0: debug_text = text_ocr[:1000] # Garde le début pour debug

            # Identification de la banque
            bank = "Inconnue"
            for b in ["UNICS", "ADVANS", "MUPECI", "FINANCIAL", "CEPAC"]:
                if b in text_ocr.upper():
                    bank = b
                    break

            # 2. Extraction avec tolérance maximale
            table = page.extract_table({
                "vertical_strategy": "text",
                "horizontal_strategy": "lines",
                "snap_y_tolerance": 6,
                "intersection_x_tolerance": 15
            })
            
            if table:
                current_row = None
                for row in table:
                    row = [str(c).replace('\n', ' ').strip() if c else "" for c in row]
                    if not row or len(row) < 3: continue

                    # REGEX ULTRA-SOUPLE : Cherche n'importe quoi qui ressemble à une date
                    # (Ex: 01/01/2025, 01-Jan-2025, 01/Jan/25)
                    date_found = re.search(r'\d{1,2}[./-][\w\d]{2,3}[./-]\d{2,4}', row[0])
                    
                    if date_found:
                        if current_row: all_data.append(current_row)
                        current_row = {
                            "Banque": bank,
                            "Date": row[0],
                            "Libellé": row[1] if len(row) > 1 else "",
                            "Débit": clean_amount(row[-3]) if len(row) > 3 else 0,
                            "Crédit": clean_amount(row[-2]) if len(row) > 2 else 0,
                            "Solde": clean_amount(row[-1]) if len(row) > 1 else 0
                        }
                    elif current_row and row[1]:
                        # Si la ligne ne contient pas de date, c'est la suite du libellé précédent
                        current_row["Libellé"] += " " + row[1]
                
                if current_row: all_data.append(current_row)

            progress_bar.progress((i + 1) / len(images))
            
    return pd.DataFrame(all_data), debug_text

uploaded_file = st.file_uploader("Téléchargez votre scan PDF", type="pdf")

if uploaded_file is not None:
    content = uploaded_file.read()
    df, debug_info = process_pdf(content)
    
    if not df.empty:
        st.success(f"Extraction réussie : {len(df)} transactions.")
        st.dataframe(df, use_container_width=True)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        st.download_button("📥 Télécharger l'Excel", output.getvalue(), "releve.xlsx")
    else:
        st.error("Aucune transaction détectée.")
        with st.expander("🔍 Voir le diagnostic technique"):
            st.write("L'IA a lu ce texte sur la première page :")
            st.code(debug_info)
            st.write("Si vous ne voyez pas de dates ou de montants dans le texte ci-dessus, le scan est trop sombre.")
