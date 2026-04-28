import streamlit as st
import pdfplumber
import pandas as pd
import re
import io
from pdf2image import convert_from_bytes
import pytesseract

# Configuration du chemin Tesseract pour Streamlit Cloud
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

st.set_page_config(page_title="OCR Bancaire Cameroun", layout="wide")

st.title("🏦 Extracteur Intelligent de Relevés (Scans PDF)")
st.info("Cette version fusionne les lignes de texte et nettoie les montants automatiquement.")

def clean_amount(val):
    """Nettoie les montants complexes : (25.00), 1,450.00, etc."""
    if val is None or str(val).strip() == "": return 0.0
    s = str(val).strip()
    # Gestion des parenthèses pour les débits (ex: MUPECI)
    is_negative = False
    if '(' in s and ')' in s:
        is_negative = True
    # Nettoyage des caractères non numériques sauf le point et la virgule
    s = re.sub(r'[^\d.,]', '', s)
    # Remplacement de la virgule par rien (milliers) et s'assurer que le point est le décimal
    if ',' in s and '.' in s: s = s.replace(',', '')
    elif ',' in s: s = s.replace(',', '.')
    
    try:
        num = float(s)
        return -num if is_negative else num
    except:
        return 0.0

def process_pdf(file_bytes):
    all_data = []
    # Conversion en images pour l'OCR (Identification de la banque)
    images = convert_from_bytes(file_bytes)
    
    progress_bar = st.progress(0)
    
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            # 1. OCR pour identifier la banque sur la page actuelle
            # On réduit la zone d'OCR au haut de la page pour aller plus vite
            img_page = images[i].convert('L')
            text_ocr = pytesseract.image_to_string(img_page, lang='fra')
            
            bank = "Inconnue"
            if "UNICS" in text_ocr.upper(): bank = "UNICS"
            elif "ADVANS" in text_ocr.upper(): bank = "ADVANS"
            elif "MUPECI" in text_ocr.upper(): bank = "MUPECI"

            # 2. Extraction du tableau avec réglages de précision
            # On utilise "lattice" pour les tableaux avec lignes, "stream" pour les autres
            table = page.extract_table({
                "vertical_strategy": "text", 
                "horizontal_strategy": "lines",
                "snap_y_tolerance": 5, # Tolérance pour les lignes un peu tordues (scans)
            })
            
            if table:
                current_row = None
                for row in table:
                    # Nettoyage des cellules
                    row = [str(c).strip() if c else "" for c in row]
                    
                    # Détection d'une nouvelle ligne de transaction (cherche une date)
                    # Format : 02/Jan/2025 ou 02/01/2025
                    has_date = re.search(r'\d{1,2}[/]\w{2,3}[/]\d{2,4}|\d{1,2}[/]\d{1,2}[/]\d{2,4}', row[0])
                    
                    if has_date:
                        # Si on avait une ligne en cours, on l'ajoute avant d'en créer une nouvelle
                        if current_row: all_data.append(current_row)
                        
                        current_row = {
                            "Banque": bank,
                            "Date": row[0].replace('\n', ' '),
                            "Libellé": row[1].replace('\n', ' '),
                            "Débit": clean_amount(row[-3]),
                            "Crédit": clean_amount(row[-2]),
                            "Solde": clean_amount(row[-1])
                        }
                    elif current_row and row[1]:
                        # Si pas de date mais du texte en colonne libellé, on fusionne (multi-lignes)
                        current_row["Libellé"] += " " + row[1].replace('\n', ' ')
                
                # Ajouter la dernière ligne de la page
                if current_row: all_data.append(current_row)

            progress_bar.progress((i + 1) / len(pdf.pages))
            
    return pd.DataFrame(all_data)

uploaded_file = st.file_uploader("Glissez votre scan PDF ici", type="pdf")

if uploaded_file is not None:
    file_content = uploaded_file.read()
    df = process_pdf(file_content)
    
    if not df.empty:
        st.success(f"Extraction réussie : {len(df)} transactions trouvées.")
        # Nettoyage final du libellé (doubles espaces)
        df['Libellé'] = df['Libellé'].str.replace(r'\s+', ' ', regex=True)
        
        st.dataframe(df, use_container_width=True)
        
        # Export Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        
        st.download_button(
            label="📥 Télécharger en Excel",
            data=output.getvalue(),
            file_name="releve_banque_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.error("Aucune transaction détectée.")
        st.warning("Conseil : Assurez-vous que le scan est bien droit et lisible.")
