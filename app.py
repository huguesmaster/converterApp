import streamlit as st
import pdfplumber
import pandas as pd
import re
import io
from pdf2image import convert_from_bytes
import pytesseract

# Configuration de la page
st.set_page_config(page_title="Extracteur de Relevés Cameroun", layout="wide")

st.title("🏦 Convertisseur de Relevés Bancaires (Scans PDF)")
st.write("Déposez vos scans PDF pour obtenir un fichier Excel propre.")

def clean_amount(val):
    if val is None or val == "": return 0.0
    # Nettoyage des parenthèses (MUPECI), virgules et espaces
    s = str(val).replace('(', '-').replace(')', '').replace(',', '').replace(' ', '').strip()
    try:
        return float(s)
    except:
        return 0.0

def process_pdf(file_bytes):
    all_data = []
    # Conversion PDF en images pour l'OCR si nécessaire
    images = convert_from_bytes(file_bytes)
    
    progress_bar = st.progress(0)
    total_pages = len(images)

    for i, image in enumerate(images):
        # OCR sur la page
        text = pytesseract.image_to_string(image, lang='fra')
        
        # Identification simplifiée de la banque
        bank = "Inconnue"
        if "UNICS" in text.upper(): bank = "UNICS"
        elif "ADVANS" in text.upper(): bank = "ADVANS"
        elif "MUPECI" in text.upper(): bank = "MUPECI"
        
        # Ici, on utilise pdfplumber pour la structure des tableaux
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            page = pdf.pages[i]
            table = page.extract_table()
            if table:
                for row in table:
                    # Filtrage : On cherche une date au début de la ligne
                    if row[0] and re.search(r'\d{2}/\d{2}/\d{4}|\d{2}/\w{3}/\d{4}', str(row[0])):
                        all_data.append({
                            "Banque": bank,
                            "Date": row[0].replace('\n', ' '),
                            "Libellé": row[1].replace('\n', ' ') if len(row) > 1 else "",
                            "Débit": clean_amount(row[-3]) if len(row) > 3 else 0,
                            "Crédit": clean_amount(row[-2]) if len(row) > 2 else 0,
                            "Solde": clean_amount(row[-1]) if len(row) > 1 else 0
                        })
        
        progress_bar.progress((i + 1) / total_pages)
    
    return pd.DataFrame(all_data)

uploaded_file = st.file_uploader("Choisir un fichier PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner('Analyse et OCR en cours...'):
        df = process_pdf(uploaded_file.read())
        
        if not df.empty:
            st.success("Extraction terminée !")
            st.dataframe(df)
            
            # Export Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            
            st.download_button(
                label="📥 Télécharger le fichier Excel",
                data=output.getvalue(),
                file_name="releve_converti.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.error("Aucune transaction détectée. Vérifiez la qualité du scan.")