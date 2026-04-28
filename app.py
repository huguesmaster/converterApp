"""
╔══════════════════════════════════════════════════════╗
║        BANK STATEMENT EXTRACTOR — Streamlit          ║
║        Compatible : Financial House S.A              ║
╚══════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime

from extractor import BankStatementExtractor
from cleaner import DataCleaner
from exporter import BankStatementExporter

# ──────────────────────────────────────────────────────
# CONFIGURATION PAGE
# ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Statement Extractor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────
# CSS PERSONNALISÉ
# ──────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Général ── */
[data-testid="stAppViewContainer"] {
    background: #F0F4F8;
}

/* ── Header custom ── */
.main-header {
    background: linear-gradient(135deg, #1F4E79, #2E75B6);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    color: white;
    box-shadow: 0 4px 20px rgba(31,78,121,0.3);
}

.main-header h1 {
    font-size: 2rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}

.main-header p {
    margin: 0.3rem 0 0;
    opacity: 0.85;
    font-size: 1rem;
}

/* ── Cartes stats ── */
.stat-card {
    background: white;
    border-radius: 12px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border-top: 4px solid;
    height: 100%;
}

.stat-card.credit  { border-color: #27AE60; }
.stat-card.debit   { border-color: #E74C3C; }
.stat-card.net     { border-color: #F39C12; }
.stat-card.count   { border-color: #1F4E79; }

.stat-label {
    font-size: 0.78rem;
    color: #7F8C8D;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
}

.stat-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #2C3E50;
    line-height: 1.2;
    margin-top: 6px;
}

.stat-icon {
    font-size: 1.5rem;
    margin-bottom: 6px;
}

/* ── Upload zone ── */
.upload-hint {
    background: white;
    border: 2px dashed #AED6F1;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    color: #5D8AA8;
    margin-bottom: 1rem;
}

/* ── Tableau ── */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Boutons ── */
.stDownloadButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: white;
    border-right: 1px solid #E0E0E0;
}

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}

.badge-success { background:#D5F5E3; color:#1E8449; }
.badge-info    { background:#D6EAF8; color:#1A5276; }
.badge-warning { background:#FDEBD0; color:#784212; }

/* ── Alert custom ── */
.info-box {
    background: #EBF5FB;
    border-left: 4px solid #2E75B6;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.9rem;
}

/* ── Section title ── */
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1F4E79;
    margin: 1.5rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 1.5rem;
    color: #95A5A6;
    font-size: 0.82rem;
    margin-top: 3rem;
    border-top: 1px solid #E0E0E0;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────
def init_state():
    defaults = {
        'df_raw': None,
        'df_clean': None,
        'stats': None,
        'account_info': None,
        'file_name': '',
        'extraction_done': False,
        'edit_mode': False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()


# ──────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────
def format_amount(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ''
    try:
        return f"{float(val):,.0f}".replace(',', ' ')
    except Exception:
        return str(val)


def style_dataframe(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Applique un style conditionnel au DataFrame Streamlit."""

    def color_row(row):
        libelle = str(row.get('Libellé', '')).lower()
        if 'solde' in libelle:
            return ['background-color: #D6EAF8; font-weight: bold'] * len(row)

        styles = [''] * len(row)
        cols = list(row.index)

        if 'Débit' in cols:
            idx = cols.index('Débit')
            val = row.get('Débit')
            if val and not (isinstance(val, float) and np.isnan(val)):
                styles[idx] = 'background-color: #FADBD8; color: #C0392B; font-weight:600'

        if 'Crédit' in cols:
            idx = cols.index('Crédit')
            val = row.get('Crédit')
            if val and not (isinstance(val, float) and np.isnan(val)):
                styles[idx] = 'background-color: #D5F5E3; color: #1E8449; font-weight:600'

        return styles

    return (
        df.style
        .apply(color_row, axis=1)
        .format({
            'Débit':  lambda x: format_amount(x) if pd.notna(x) else '',
            'Crédit': lambda x: format_amount(x) if pd.notna(x) else '',
            'Solde':  lambda x: format_amount(x) if pd.notna(x) else '',
        })
    )


# ──────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    st.divider()

    st.markdown("### 📂 Chargement")
    uploaded_file = st.file_uploader(
        "Relevé bancaire PDF",
        type=["pdf"],
        help="Glissez votre PDF Financial House S.A (max 50 MB)"
    )

    if uploaded_file:
        st.markdown(f"""
        <div class="badge badge-success">✅ {uploaded_file.name}</div>
        <br><small style="color:#7F8C8D">
        Taille : {uploaded_file.size / 1024:.1f} KB
        </small>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🏦 Informations compte")
    account_name = st.text_input(
        "Titulaire",
        value="DJOKO KEMTCHOUANG AARON BALDERIC",
        placeholder="Nom du titulaire"
    )
    account_id = st.text_input(
        "N° Compte",
        value="0303041073731001",
        placeholder="Numéro de compte"
    )

    st.divider()

    st.markdown("### 🔧 Options")
    remove_duplicates = st.checkbox(
        "Supprimer les doublons", value=True
    )
    sort_by_date = st.checkbox(
        "Trier par date", value=True
    )
    rows_per_page = st.select_slider(
        "Lignes affichées",
        options=[25, 50, 100, 200, 500],
        value=100
    )

    st.divider()
    st.markdown("""
    <div style="font-size:0.78rem; color:#95A5A6; text-align:center">
    🏦 Bank Statement Extractor<br>
    Compatible Financial House S.A<br>
    v1.0.0
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────
# HEADER PRINCIPAL
# ──────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏦 Bank Statement Extractor</h1>
    <p>Transformez vos relevés PDF en données structurées Excel/CSV en quelques secondes</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────
# ZONE D'EXTRACTION
# ──────────────────────────────────────────────────────
if not st.session_state.extraction_done:

    if uploaded_file is None:
        st.markdown("""
        <div class="upload-hint">
            <div style="font-size:3rem">📄</div>
            <h3 style="color:#1F4E79;margin:0.5rem 0">
                Importez votre relevé bancaire PDF
            </h3>
            <p>Utilisez le panneau latéral gauche pour charger votre fichier</p>
        </div>

        <div class="info-box">
            <b>📌 Formats supportés :</b><br>
            ✅ Financial House S.A — PDF texte natif<br>
            ✅ PDF scannés (avec OCR automatique)<br>
            ✅ Relevés multi-pages<br>
            ✅ Toutes périodes (Jan–Déc 2025)
        </div>
        """, unsafe_allow_html=True)

        # Guide rapide
        with st.expander("📖 Guide d'utilisation rapide"):
            st.markdown("""
            1. **Chargez votre PDF** via le panneau latéral
            2. **Cliquez sur "Extraire"** pour lancer l'analyse
            3. **Vérifiez les données** dans le tableau interactif
            4. **Exportez** en Excel ou CSV
            """)

    else:
        # Bouton d'extraction
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style="text-align:center;padding:1rem;background:white;
                        border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.07);
                        margin-bottom:1rem">
                <div style="font-size:2rem">📄</div>
                <b>{uploaded_file.name}</b><br>
                <small style="color:#7F8C8D">
                    {uploaded_file.size/1024:.1f} KB — Prêt pour extraction
                </small>
            </div>
            """, unsafe_allow_html=True)

            extract_btn = st.button(
                "🚀 Extraire les données",
                use_container_width=True,
                type="primary"
            )

        if extract_btn:
            pdf_bytes = uploaded_file.read()

            # Barre de progression
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(step: int, message: str):
                progress_bar.progress(step / 100)
                status_text.markdown(
                    f"<div style='text-align:center;color:#1F4E79'>"
                    f"{message}</div>",
                    unsafe_allow_html=True
                )

            # ── Extraction ──
            extractor = BankStatementExtractor(
                progress_callback=update_progress
            )

            with st.spinner(""):
                df_raw = extractor.extract(pdf_bytes)

            # ── Nettoyage ──
            update_progress(92, "🧹 Nettoyage des données...")
            cleaner = DataCleaner()
            df_clean = cleaner.clean(df_raw)

            # Stats
            stats = cleaner.get_statistics(df_clean)

            # Infos compte
            account_info = {
                'account_name': account_name,
                'account_id': account_id,
                'period': (
                    f"{stats.get('periode_debut','')} → "
                    f"{stats.get('periode_fin','')}"
                ),
                'extraction_date': datetime.now().strftime(
                    "%d/%m/%Y %H:%M"
                ),
            }

            # Sauvegarder en session
            st.session_state.df_raw = df_raw
            st.session_state.df_clean = df_clean
            st.session_state.stats = stats
            st.session_state.account_info = account_info
            st.session_state.file_name = uploaded_file.name
            st.session_state.extraction_done = True

            progress_bar.progress(100)
            status_text.empty()
            st.rerun()


# ──────────────────────────────────────────────────────
# AFFICHAGE DES RÉSULTATS
# ──────────────────────────────────────────────────────
if st.session_state.extraction_done:
    df = st.session_state.df_clean
    stats = st.session_state.stats
    account_info = st.session_state.account_info

    if df is None or df.empty:
        st.error(
            "❌ Aucune donnée n'a pu être extraite. "
            "Vérifiez le format du PDF."
        )
        if st.button("🔄 Réessayer"):
            st.session_state.extraction_done = False
            st.rerun()

    else:
        # ── Bannière succès ──
        st.success(
            f"✅ **{len(df)} lignes extraites** depuis "
            f"`{st.session_state.file_name}`"
        )

        # ──────────────────────────────────────
        # CARTES STATISTIQUES
        # ──────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)

        cards = [
            (col1, "credit",  "💚", "Total Crédits",
             stats.get('total_credit', 0)),
            (col2, "debit",   "❤️",  "Total Débits",
             stats.get('total_debit', 0)),
            (col3, "net",     "💛", "Flux Net",
             stats.get('net', 0)),
            (col4, "count",   "💙", "Transactions",
             stats.get('total_transactions', 0)),
        ]

        for col, cls, icon, label, value in cards:
            with col:
                if cls == 'count':
                    val_str = f"{int(value):,}".replace(',', ' ')
                else:
                    val_str = format_amount(value)

                st.markdown(f"""
                <div class="stat-card {cls}">
                    <div class="stat-icon">{icon}</div>
                    <div class="stat-label">{label}</div>
                    <div class="stat-value">{val_str}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ──────────────────────────────────────
        # INFORMATIONS PÉRIODE
        # ──────────────────────────────────────
        if stats.get('periode_debut') or stats.get('solde_ouverture'):
            with st.expander("📅 Détails de la période", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("📅 Début", stats.get('periode_debut', 'N/A'))
                c2.metric("📅 Fin",   stats.get('periode_fin', 'N/A'))
                c3.metric(
                    "🔓 Solde ouverture",
                    format_amount(stats.get('solde_ouverture')) or 'N/A'
                )
                c4.metric(
                    "🔒 Solde clôture",
                    format_amount(stats.get('solde_cloture')) or 'N/A'
                )

        # ──────────────────────────────────────
        # BARRE D'OUTILS
        # ──────────────────────────────────────
        st.markdown(
            '<div class="section-title">📊 Données extraites</div>',
            unsafe_allow_html=True
        )

        tool_col1, tool_col2, tool_col3, tool_col4 = st.columns([3, 1, 1, 1])

        with tool_col1:
            search = st.text_input(
                "🔍 Rechercher",
                placeholder="Filtrer par libellé, référence, montant...",
                label_visibility="collapsed"
            )

        with tool_col2:
            # Export Excel
            exporter = BankStatementExporter()
            excel_bytes = exporter.to_excel(df, stats, account_info)
            st.download_button(
                label="📥 Excel",
                data=excel_bytes,
                file_name=f"releve_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        with tool_col3:
            # Export CSV
            csv_bytes = exporter.to_csv(df)
            st.download_button(
                label="📄 CSV",
                data=csv_bytes,
                file_name=f"releve_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with tool_col4:
            if st.button("🔄 Nouveau", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        # ──────────────────────────────────────
        # FILTRAGE
        # ──────────────────────────────────────
        df_display = df.copy()

        if search:
            mask = df_display.apply(
                lambda row: row.astype(str).str.contains(
                    search, case=False, na=False
                ).any(),
                axis=1
            )
            df_display = df_display[mask]
            st.caption(
                f"🔍 {len(df_display)} résultat(s) pour «{search}»"
            )

        # ──────────────────────────────────────
        # TABLEAU PRINCIPAL
        # ──────────────────────────────────────

        # Pagination
        total_rows = len(df_display)
        total_pages = max(1, -(-total_rows // rows_per_page))

        if total_pages > 1:
            page_col1, page_col2, page_col3 = st.columns([2, 3, 2])
            with page_col2:
                current_page = st.number_input(
                    f"Page (sur {total_pages})",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    step=1
                )
        else:
            current_page = 1

        start_idx = (current_page - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, total_rows)
        df_page = df_display.iloc[start_idx:end_idx]

        # Affichage avec style
        st.dataframe(
            style_dataframe(df_page),
            use_container_width=True,
            height=min(600, (len(df_page) + 1) * 38 + 3),
            hide_index=True,
        )

        st.caption(
            f"📋 Affichage {start_idx+1}–{end_idx} "
            f"sur {total_rows} lignes"
        )

        # ──────────────────────────────────────
        # ÉDITEUR (optionnel)
        # ──────────────────────────────────────
        with st.expander("✏️ Modifier les données manuellement"):
            st.info(
                "💡 Modifiez directement dans le tableau ci-dessous. "
                "Cliquez sur une cellule pour l'éditer."
            )
            edited_df = st.data_editor(
                st.session_state.df_clean,
                use_container_width=True,
                num_rows="dynamic",
                height=400,
            )

            col_save, col_cancel = st.columns(2)
            with col_save:
                if st.button("💾 Sauvegarder les modifications",
                             use_container_width=True, type="primary"):
                    st.session_state.df_clean = edited_df
                    cleaner = DataCleaner()
                    st.session_state.stats = cleaner.get_statistics(edited_df)
                    st.success("✅ Données mises à jour !")
                    st.rerun()
            with col_cancel:
                if st.button("❌ Annuler", use_container_width=True):
                    st.rerun()

        # ──────────────────────────────────────
        # GRAPHIQUE SIMPLE
        # ──────────────────────────────────────
        with st.expander("📈 Visualisation du solde"):
            df_chart = df[
                df['Date'].notna() &
                (df['Date'] != '') &
                df['Solde'].notna()
            ].copy()

            if not df_chart.empty:
                df_chart['_date'] = pd.to_datetime(
                    df_chart['Date'],
                    format='%d/%m/%Y',
                    errors='coerce'
                )
                df_chart = df_chart.dropna(subset=['_date', 'Solde'])
                df_chart = df_chart.sort_values('_date')

                if not df_chart.empty:
                    chart_data = df_chart.set_index('_date')['Solde']
                    st.line_chart(
                        chart_data,
                        use_container_width=True,
                        height=300,
                        color="#1F4E79"
                    )
                    st.caption(
                        "📈 Évolution du solde dans le temps"
                    )
            else:
                st.info("Pas assez de données pour le graphique.")


# ──────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🏦 Bank Statement Extractor — Compatible Financial House S.A<br>
    Développé avec Python & Streamlit
</div>
""", unsafe_allow_html=True)
