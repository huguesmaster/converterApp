"""
╔══════════════════════════════════════════════════════════════╗
║         BANK STATEMENT EXTRACTOR — Streamlit                 ║
║         Compatible toutes banques du Cameroun                ║
║         Version : 4.2 — Interface Raffinée                   ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from extractor_gemini import GeminiExtractor
from cleaner import DataCleaner
from exporter import BankStatementExporter
from token_counter import TokenCounter
from bank_configs import get_bank_config

# ====================== CONFIGURATION ======================
st.set_page_config(
    page_title="Bank Statement Extractor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CSS RAFFINÉ ======================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1B3A5C 0%, #2E75B6 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(27, 58, 92, 0.3);
    }
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-top: 5px solid;
    }
    .success-box { 
        background: #d4edda; 
        border-left: 5px solid #28a745; 
        padding: 1rem; 
        border-radius: 8px; 
    }
    .error-box { 
        background: #f8d7da; 
        border-left: 5px solid #dc3545; 
        padding: 1rem; 
        border-radius: 8px; 
    }
    .bank-config {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E75B6;
    }
</style>
""", unsafe_allow_html=True)

# ====================== SESSION STATE ======================
if "extraction_done" not in st.session_state:
    st.session_state.update({
        "extraction_done": False,
        "show_confirm": False,
        "df_clean": None,
        "stats": None,
        "account_info": None,
        "file_name": "",
        "debug_logs": "",
        "debug_entries": [],
        "banque_selectionnee": "Financial House S.A",
        "pdf_bytes_cache": None,
        "rows_per_page": 100,
    })

# ====================== SIDEBAR ======================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0;">
        <h2>🏦 Bank Extractor</h2>
        <p style="color:#666; font-size:0.9rem;">Toutes banques du Cameroun</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    uploaded_file = st.file_uploader("📄 Relevé bancaire (PDF)", type=["pdf"])

    if uploaded_file:
        st.success(f"✅ {uploaded_file.name}")

    st.divider()

    # Banque
    banque_sel = st.selectbox(
        "🏦 Banque émettrice",
        options=["Financial House S.A", "BGFI Bank", "UNICS", "CEPAC", "ADVANS", 
                 "MUPECI", "SCB Cameroun", "BICEC", "UBA Cameroun", "Autre banque"]
    )
    st.session_state.banque_selectionnee = banque_sel
    config = get_bank_config(banque_sel)

    with st.expander("📋 Configuration banque", expanded=False):
        st.markdown(f"**{config.emoji} {config.nom}**")
        st.caption(f"Références : {', '.join(config.col_ref[:4])}")

    # Méthode
    st.divider()
    method = st.radio(
        "🤖 Méthode d'extraction",
        options=["vision", "hybrid", "pdfplumber"],
        format_func=lambda x: {
            "vision": "🔭 Gemini Vision (Recommandé)",
            "hybrid": "⚡ Gemini Hybride",
            "pdfplumber": "📄 pdfplumber (Gratuit)"
        }[x]
    )

    # Clé API
    if method in ("vision", "hybrid"):
        st.divider()
        gemini_key = st.text_input("🔑 Clé API Gemini", type="password", placeholder="AIzaSy...")
        if gemini_key:
            st.success("Clé API configurée")
        else:
            st.warning("Clé API requise pour les modes Gemini")

    st.divider()

    if st.button("🔄 Nouvelle extraction", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key not in ["gemini_key_input"]:
                del st.session_state[key]
        st.rerun()


# ====================== HEADER ======================
st.markdown("""
<div class="main-header">
    <h1>🏦 Bank Statement Extractor</h1>
    <p style="opacity:0.9; margin-top:0.5rem;">Extraction intelligente de relevés bancaires camerounais</p>
</div>
""", unsafe_allow_html=True)


# ====================== LOGIQUE PRINCIPALE ======================
if uploaded_file and not st.session_state.extraction_done and not st.session_state.show_confirm:
    st.info("Cliquez sur **Lancer l'extraction** dans la barre latérale pour commencer.")

if st.session_state.show_confirm and uploaded_file:
    banque = st.session_state.banque_selectionnee
    if st.button(f"✅ Confirmer et extraire — {banque}", type="primary", use_container_width=True):
        with st.spinner("Extraction en cours..."):
            try:
                pdf_bytes = uploaded_file.read()
                st.session_state.pdf_bytes_cache = pdf_bytes

                if method in ("vision", "hybrid"):
                    extractor = GeminiExtractor(
                        api_key=gemini_key,
                        mode=method,
                        banque_nom=banque,
                        verbose_debug=True
                    )
                    df_raw = extractor.extract(pdf_bytes)
                else:
                    # Placeholder pour pdfplumber
                    st.warning("Mode pdfplumber non encore implémenté dans cette version.")
                    st.stop()

                # Nettoyage
                cleaner = DataCleaner()
                df_clean = cleaner.clean(df_raw, banque_nom=banque)
                stats = cleaner.get_statistics(df_clean)

                # Sauvegarde
                st.session_state.df_clean = df_clean
                st.session_state.stats = stats
                st.session_state.account_info = {
                    "banque": banque,
                    "banque_emoji": get_bank_config(banque).emoji,
                    "extraction_date": datetime.now().strftime("%d/%m/%Y %H:%M")
                }
                st.session_state.extraction_done = True
                st.session_state.show_confirm = False

                if method in ("vision", "hybrid"):
                    st.session_state.debug_logs = extractor.get_debug_logs()

                st.success("✅ Extraction terminée avec succès !")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Erreur lors de l'extraction : {str(e)}")
                if 'extractor' in locals():
                    with st.expander("🔍 Logs détaillés"):
                        st.text(extractor.get_debug_logs())


# ====================== RÉSULTATS ======================
if st.session_state.extraction_done and st.session_state.df_clean is not None:
    df = st.session_state.df_clean
    stats = st.session_state.stats
    info = st.session_state.account_info

    st.markdown(f"""
    <div class="success-box">
        ✅ <b>{len(df)}</b> lignes extraites avec succès — 
        {info.get('banque_emoji', '')} <b>{info.get('banque', '')}</b>
    </div>
    """, unsafe_allow_html=True)

    # Statistiques
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Crédit", f"{stats.get('total_credit', 0):,.0f} FCFA")
    with col2:
        st.metric("Total Débit", f"{stats.get('total_debit', 0):,.0f} FCFA")
    with col3:
        st.metric("Flux Net", f"{stats.get('net', 0):,.0f} FCFA")
    with col4:
        st.metric("Transactions", stats.get('total_transactions', 0))

    # Tableau principal
    st.subheader("📋 Transactions extraites")
    st.dataframe(
        df.style.format({
            "Débit": "{:,.0f}".format,
            "Crédit": "{:,.0f}".format,
            "Solde": "{:,.0f}".format
        }),
        use_container_width=True,
        height=600
    )

    # Export
    exporter = BankStatementExporter()
    col_a, col_b = st.columns(2)
    with col_a:
        excel_bytes = exporter.to_excel(df, stats, info)
        st.download_button(
            label="📥 Télécharger en Excel",
            data=excel_bytes,
            file_name=f"releve_{banque.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    with col_b:
        if st.button("🔄 Nouvelle extraction"):
            for key in list(st.session_state.keys()):
                if key not in ["gemini_key_input"]:
                    del st.session_state[key]
            st.rerun()

    # Debug logs
    with st.expander("🔍 Voir les logs de débogage"):
        st.text(st.session_state.get("debug_logs", "Aucun log disponible"))
