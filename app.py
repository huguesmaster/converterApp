"""
╔══════════════════════════════════════════════════════════════╗
║         BANK STATEMENT EXTRACTOR — Streamlit                 ║
║         Version : 4.1 — Intégration complète                 ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import time
from datetime import datetime

from extractor import BankStatementExtractor
from extractor_gemini import GeminiExtractor
from cleaner import DataCleaner
from exporter import BankStatementExporter
from token_counter import TokenCounter
from bank_configs import get_bank_config

def pdfplumber_open(pdf_bytes: bytes):
    import pdfplumber
    return pdfplumber.open(io.BytesIO(pdf_bytes))


st.set_page_config(page_title="Bank Statement Extractor", page_icon="🏦", layout="wide")

# CSS (simplifié)
st.markdown("""
<style>
.main-header { background: linear-gradient(135deg, #1B3A5C, #2E75B6); padding: 1.8rem; border-radius: 16px; color: white; }
.success-box { background: #EAFAF1; border-left: 4px solid #27AE60; padding: 1rem; border-radius: 0 10px 10px 0; }
.error-box   { background: #FDEDEC; border-left: 4px solid #E74C3C; padding: 1rem; border-radius: 0 10px 10px 0; }
</style>
""", unsafe_allow_html=True)

# Constantes
BANQUES_CAMEROUN = [
    {"nom": "Financial House S.A", "code": "FH", "emoji": "🏛️"},
    {"nom": "BGFI Bank", "code": "BGFI", "emoji": "🔷"},
    {"nom": "UNICS", "code": "UNICS", "emoji": "🏦"},
    {"nom": "CEPAC", "code": "CEPAC", "emoji": "🏦"},
    {"nom": "ADVANS", "code": "ADVANS", "emoji": "🏦"},
    {"nom": "MUPECI", "code": "MUPECI", "emoji": "🏦"},
    {"nom": "Autre banque", "code": "AUTRE", "emoji": "🏦"},
]

METHODES = {
    "vision": "🔭 Gemini Vision (Recommandé)",
    "hybrid": "⚡ Gemini Hybride",
    "pdfplumber": "📄 pdfplumber (Gratuit)"
}

# Session State
if "extraction_done" not in st.session_state:
    st.session_state.extraction_done = False
    st.session_state.show_confirm = False
    st.session_state.banque_selectionnee = "Financial House S.A"
    st.session_state.pdf_bytes_cache = None
    st.session_state.debug_logs = ""
    st.session_state.debug_entries = []

def get_gemini_key():
    return st.session_state.get("gemini_key_input", "")

# Sidebar
with st.sidebar:
    st.title("🏦 Bank Extractor")
    uploaded_file = st.file_uploader("Chargez votre relevé PDF", type=["pdf"])

    banque_sel = st.selectbox("Banque", [b["nom"] for b in BANQUES_CAMEROUN])
    st.session_state.banque_selectionnee = banque_sel
    config = get_bank_config(banque_sel)

    with st.expander("Configuration banque"):
        st.write(f"**{config.emoji} {config.nom}**")

    method = st.radio("Méthode d'extraction", options=list(METHODES.keys()), 
                     format_func=lambda x: METHODES[x])

    if method in ("vision", "hybrid"):
        gemini_key = st.text_input("Clé API Gemini", type="password", placeholder="AIzaSy...")
        st.session_state["gemini_key_input"] = gemini_key

    if st.button("🚀 Lancer l'extraction", type="primary"):
        if uploaded_file:
            st.session_state.pdf_bytes_cache = uploaded_file.read()
            st.session_state.show_confirm = True
            st.rerun()

# Extraction
if st.session_state.show_confirm and uploaded_file:
    banque = st.session_state.banque_selectionnee
    method = method  # from sidebar

    if st.button("✅ Confirmer l'extraction"):
        progress_bar = st.progress(0)
        status = st.empty()

        def _progress(step, msg):
            progress_bar.progress(min(step/100, 1.0))
            status.info(msg)

        extractor = None
        try:
            if method in ("vision", "hybrid"):
                _progress(10, f"Initialisation Gemini pour {banque}...")
                extractor = GeminiExtractor(
                    api_key=get_gemini_key(),
                    mode=method,
                    banque_nom=banque,
                    progress_callback=_progress,
                    verbose_debug=True
                )
                df_raw = extractor.extract(st.session_state.pdf_bytes_cache)
            else:
                extractor = BankStatementExtractor(progress_callback=_progress)
                df_raw = extractor.extract(st.session_state.pdf_bytes_cache)

            # Nettoyage
            _progress(90, "Nettoyage des données...")
            cleaner = DataCleaner()
            df_clean = cleaner.clean(df_raw, banque_nom=banque)
            stats = cleaner.get_statistics(df_clean)

            st.session_state.df_clean = df_clean
            st.session_state.stats = stats
            st.session_state.extraction_done = True
            st.session_state.show_confirm = False

            if method in ("vision", "hybrid") and extractor:
                st.session_state.debug_logs = extractor.get_debug_logs()
                st.session_state.debug_entries = extractor.get_debug_entries()

            st.success("✅ Extraction terminée avec succès !")
            st.rerun()

        except Exception as e:
            st.error(f"Erreur : {str(e)}")
            if extractor and hasattr(extractor, 'get_debug_logs'):
                with st.expander("Logs de debug"):
                    st.text(extractor.get_debug_logs())
            st.session_state.show_confirm = False

# Résultats
if st.session_state.extraction_done and st.session_state.df_clean is not None:
    st.success(f"Extraction réussie : {len(st.session_state.df_clean)} lignes")
    st.dataframe(st.session_state.df_clean, use_container_width=True)

    if st.button("Nouvelle extraction"):
        for key in list(st.session_state.keys()):
            if key not in ["gemini_key_input"]:
                del st.session_state[key]
        st.rerun()
