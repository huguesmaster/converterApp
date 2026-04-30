"""
╔══════════════════════════════════════════════════════════════╗
║         BANK STATEMENT EXTRACTOR — Streamlit                 ║
║         Compatible : Toutes banques du Cameroun              ║
║         Version : 4.3 — Clé API + Interface Raffinée         ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from extractor_gemini import GeminiExtractor
from cleaner import DataCleaner
from exporter import BankStatementExporter
from bank_configs import get_bank_config

# ====================== CONFIGURATION ======================
st.set_page_config(
    page_title="Bank Statement Extractor — Cameroun",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CSS ======================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1B3A5C 0%, #2E75B6 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(27,58,92,0.3);
    }
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-top: 5px solid;
    }
    .success-box {
        background: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .error-box {
        background: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .bank-config {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E75B6;
    }
</style>
""", unsafe_allow_html=True)


# ====================== HELPER CLÉ GEMINI ======================
def get_gemini_key() -> str:
    """Récupère la clé API Gemini avec priorité : Secrets → Saisie manuelle"""
    # Priorité 1 : Secrets Streamlit (recommandé)
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key and str(key).strip():
            return str(key).strip()
    except Exception:
        pass

    # Priorité 2 : Saisie manuelle par l'utilisateur
    return st.session_state.get("gemini_key_input", "").strip()


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
        <p style="color:#666; font-size:0.9rem;">Extraction de relevés bancaires</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Upload PDF
    uploaded_file = st.file_uploader("📄 Chargez votre relevé PDF", type=["pdf"])
    if uploaded_file:
        st.success(f"✅ {uploaded_file.name} chargé")

    st.divider()

    # Banque
    banque_sel = st.selectbox(
        "🏦 Banque émettrice",
        options=[
            "Financial House S.A", "BGFI Bank", "UNICS", "CEPAC", 
            "ADVANS", "MUPECI", "SCB Cameroun", "BICEC", "UBA Cameroun", 
            "Autre banque"
        ]
    )
    st.session_state.banque_selectionnee = banque_sel
    config = get_bank_config(banque_sel)

    with st.expander("📋 Configuration banque", expanded=False):
        st.markdown(f"**{config.emoji} {config.nom}**")
        st.caption(f"Références : {', '.join(config.col_ref[:4])}")

    st.divider()

    # Méthode d'extraction
    method = st.radio(
        "🤖 Méthode d'extraction",
        options=["vision", "hybrid", "pdfplumber"],
        format_func=lambda x: {
            "vision": "🔭 Gemini Vision (Recommandé)",
            "hybrid": "⚡ Gemini Hybride",
            "pdfplumber": "📄 pdfplumber (Gratuit)"
        }[x]
    )

    # Clé API Gemini
    if method in ("vision", "hybrid"):
        st.divider()
        st.markdown("#### 🔑 Clé API Gemini")

        current_key = get_gemini_key()

        if current_key:
            st.success("✅ Clé API Gemini détectée et active")
            st.caption("Utilisée depuis les Secrets Streamlit")
        else:
            st.warning("⚠️ Clé API non configurée")
            key_input = st.text_input(
                "Saisissez votre clé API Gemini",
                type="password",
                placeholder="AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                help="Obtenez une clé gratuite sur https://aistudio.google.com/app/apikey"
            )
            st.session_state["gemini_key_input"] = key_input

            st.markdown(
                "[🔗 Obtenir une clé API Gemini](https://aistudio.google.com/app/apikey)", 
                unsafe_allow_html=True
            )

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
    <p style="opacity:0.9; margin: 0.5rem 0 0;">Extraction intelligente de relevés bancaires camerounais</p>
</div>
""", unsafe_allow_html=True)


# ====================== LOGIQUE D'EXTRACTION ======================
if uploaded_file and not st.session_state.extraction_done:
    if st.button(f"🚀 Lancer l'extraction — {st.session_state.banque_selectionnee}", 
                 type="primary", use_container_width=True):
        st.session_state.pdf_bytes_cache = uploaded_file.read()
        st.session_state.show_confirm = True
        st.rerun()


if st.session_state.show_confirm and uploaded_file:
    banque = st.session_state.banque_selectionnee
    method = method  # from sidebar

    if st.button("✅ Confirmer et lancer l'extraction", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def _progress(step: int, msg: str):
            progress_bar.progress(min(step / 100, 1.0))
            status_text.info(msg)

        extractor = None
        try:
            _progress(5, f"Initialisation pour {banque}...")

            if method in ("vision", "hybrid"):
                api_key = get_gemini_key()
                if not api_key:
                    st.error("❌ Clé API Gemini requise pour ce mode.")
                    st.stop()

                extractor = GeminiExtractor(
                    api_key=api_key,
                    mode=method,
                    banque_nom=banque,
                    progress_callback=_progress,
                    verbose_debug=True
                )
                df_raw = extractor.extract(st.session_state.pdf_bytes_cache)
            else:
                st.warning("Mode pdfplumber non encore implémenté.")
                st.stop()

            # Nettoyage
            _progress(85, "Nettoyage et structuration des données...")
            cleaner = DataCleaner()
            df_clean = cleaner.clean(df_raw, banque_nom=banque)
            stats = cleaner.get_statistics(df_clean)

            # Sauvegarde
            st.session_state.df_clean = df_clean
            st.session_state.stats = stats
            st.session_state.account_info = {
                "banque": banque,
                "banque_emoji": get_bank_config(banque).emoji,
                "extraction_date": datetime.now().strftime("%d/%m/%Y à %H:%M")
            }
            st.session_state.extraction_done = True
            st.session_state.show_confirm = False

            if extractor:
                st.session_state.debug_logs = extractor.get_debug_logs()

            progress_bar.progress(1.0)
            status_text.success("✅ Extraction terminée avec succès !")
            st.rerun()

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Erreur lors de l'extraction : {str(e)}")
            
            if extractor and hasattr(extractor, 'get_debug_logs'):
                with st.expander("🔍 Logs de débogage", expanded=True):
                    st.text(extractor.get_debug_logs())


# ====================== AFFICHAGE DES RÉSULTATS ======================
if st.session_state.extraction_done and st.session_state.df_clean is not None:
    df = st.session_state.df_clean
    stats = st.session_state.stats or {}
    info = st.session_state.account_info or {}

    st.markdown(f"""
    <div class="success-box">
        ✅ <b>{len(df)}</b> lignes extraites avec succès — 
        {info.get('banque_emoji', '')} <b>{info.get('banque', '')}</b><br>
        📅 {info.get('extraction_date', '')}
    </div>
    """, unsafe_allow_html=True)

    # Statistiques
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Crédits", f"{stats.get('total_credit', 0):,.0f} FCFA")
    with c2:
        st.metric("Total Débits", f"{stats.get('total_debit', 0):,.0f} FCFA")
    with c3:
        st.metric("Flux Net", f"{stats.get('net', 0):,.0f} FCFA")
    with c4:
        st.metric("Nombre de transactions", stats.get('total_transactions', 0))

    # Tableau
    st.subheader("📋 Transactions extraites")
    st.dataframe(
        df,
        use_container_width=True,
        height=550,
        hide_index=True
    )

    # Export
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Télécharger en Excel"):
            exporter = BankStatementExporter()
            excel_bytes = exporter.to_excel(df, stats, info)
            st.download_button(
                label="Télécharger Excel",
                data=excel_bytes,
                file_name=f"releve_{info.get('banque', 'banque').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    with col2:
        if st.button("🔄 Nouvelle extraction", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ["gemini_key_input"]:
                    del st.session_state[key]
            st.rerun()

    # Logs
    with st.expander("🔍 Logs de débogage"):
        st.text(st.session_state.get("debug_logs", "Aucun log disponible"))

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; font-size:0.85rem;'>"
    "Bank Statement Extractor v4.3 — Powered by Gemini AI</p>",
    unsafe_allow_html=True
)
