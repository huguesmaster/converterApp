"""
╔══════════════════════════════════════════════════════════════╗
║         BANK STATEMENT EXTRACTOR — Streamlit                 ║
║         Compatible : Toutes banques du Cameroun              ║
║         Modes : Gemini Vision | Gemini Hybride | pdfplumber  ║
║         Version : 3.1.0                                      ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import time
from datetime import datetime

# ── Imports locaux ──
from extractor import BankStatementExtractor
from extractor_gemini import GeminiExtractor
from cleaner import DataCleaner
from exporter import BankStatementExporter
from token_counter import TokenCounter
# ── Helper pour compter les pages sans charger toutes ──
def pdfplumber_open(pdf_bytes: bytes):
    """Ouvre un PDF depuis des bytes."""
    import pdfplumber
    return pdfplumber.open(io.BytesIO(pdf_bytes))
# ══════════════════════════════════════════════════════════════
# CONFIGURATION PAGE
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Bank Statement Extractor — Cameroun",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# CSS GLOBAL
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] {
    background: #F0F4F8;
}
[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E8ECF0;
}

/* ── Header ── */
.main-header {
    background: linear-gradient(135deg, #1B3A5C 0%, #2E75B6 100%);
    padding: 1.8rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 6px 24px rgba(27,58,92,0.25);
}
.main-header h1 {
    font-size: 1.9rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.3px;
}
.main-header .subtitle {
    margin: 0.4rem 0 0;
    opacity: 0.88;
    font-size: 0.95rem;
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}

/* ── Badges ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
}
.badge-gemini {
    background: linear-gradient(
        135deg,#4285F4,#EA4335,#FBBC04,#34A853
    );
    color: white;
}
.badge-success { background:#D5F5E3; color:#1E8449; }
.badge-warning { background:#FDEBD0; color:#784212; }
.badge-info    { background:#D6EAF8; color:#1A5276; }
.badge-danger  { background:#FADBD8; color:#922B21; }

/* ── Cartes stats ── */
.stat-card {
    background: white;
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
    box-shadow: 0 2px 14px rgba(0,0,0,0.06);
    border-top: 4px solid;
    height: 100%;
    transition: transform 0.2s;
}
.stat-card:hover { transform: translateY(-2px); }
.stat-card.credit  { border-color: #27AE60; }
.stat-card.debit   { border-color: #E74C3C; }
.stat-card.net-pos { border-color: #2980B9; }
.stat-card.net-neg { border-color: #E74C3C; }
.stat-card.count   { border-color: #8E44AD; }
.stat-card.solde   { border-color: #1B3A5C; }
.stat-label {
    font-size: 0.72rem;
    color: #95A5A6;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 700;
    margin-bottom: 6px;
}
.stat-icon  { font-size: 1.4rem; margin-bottom: 4px; }
.stat-value {
    font-size: 1.35rem;
    font-weight: 800;
    color: #2C3E50;
    line-height: 1.2;
}
.stat-sub {
    font-size: 0.72rem;
    color: #BDC3C7;
    margin-top: 3px;
}

/* ── Boîtes info ── */
.info-box {
    background: #EBF5FB;
    border-left: 4px solid #2E75B6;
    border-radius: 0 10px 10px 0;
    padding: 0.85rem 1.1rem;
    margin: 0.6rem 0;
    font-size: 0.87rem;
    line-height: 1.65;
}
.success-box {
    background: #EAFAF1;
    border-left: 4px solid #27AE60;
    border-radius: 0 10px 10px 0;
    padding: 0.85rem 1.1rem;
    margin: 0.6rem 0;
    font-size: 0.87rem;
    line-height: 1.65;
}
.warning-box {
    background: #FEF9E7;
    border-left: 4px solid #F39C12;
    border-radius: 0 10px 10px 0;
    padding: 0.85rem 1.1rem;
    margin: 0.6rem 0;
    font-size: 0.87rem;
    line-height: 1.65;
}
.error-box {
    background: #FDEDEC;
    border-left: 4px solid #E74C3C;
    border-radius: 0 10px 10px 0;
    padding: 0.85rem 1.1rem;
    margin: 0.6rem 0;
    font-size: 0.87rem;
    line-height: 1.65;
}

/* ── Upload zone ── */
.upload-zone {
    border: 2.5px dashed #AED6F1;
    border-radius: 14px;
    padding: 2.5rem;
    text-align: center;
    background: white;
    margin: 1rem 0;
}

/* ── Banques grid ── */
.bank-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.45rem;
    margin: 0.5rem 0;
}
.bank-chip {
    background: #F8F9FA;
    border: 1px solid #E0E0E0;
    border-radius: 8px;
    padding: 5px 7px;
    font-size: 0.73rem;
    text-align: center;
    color: #2C3E50;
    font-weight: 500;
}

/* ── Token estimate ── */
.token-estimate-card {
    background: white;
    border-radius: 14px;
    padding: 1.4rem;
    box-shadow: 0 2px 14px rgba(0,0,0,0.07);
    margin: 1rem 0;
    border: 1px solid #E8ECF0;
}
.token-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.8rem;
    margin: 1rem 0;
    text-align: center;
}
.token-metric {
    background: #F8F9FA;
    border-radius: 10px;
    padding: 0.8rem 0.5rem;
}
.token-metric-value {
    font-size: 1.25rem;
    font-weight: 800;
    color: #1B3A5C;
}
.token-metric-label {
    font-size: 0.7rem;
    color: #95A5A6;
    margin-top: 3px;
}

/* ── Log entries ── */
.log-entry {
    border-radius: 6px;
    padding: 5px 10px;
    margin: 2px 0;
    font-size: 0.82rem;
    font-family: 'Segoe UI', sans-serif;
}
.log-detail {
    font-size: 0.76rem;
    color: #5D6D7E;
    margin-top: 3px;
    padding-left: 1rem;
    border-left: 2px solid #BDC3C7;
    font-family: monospace;
    white-space: pre-wrap;
    word-break: break-all;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 1.5rem;
    color: #BDC3C7;
    font-size: 0.8rem;
    margin-top: 2rem;
    border-top: 1px solid #ECF0F1;
}

/* ── Responsive ── */
@media (max-width: 768px) {
    .token-grid { grid-template-columns: repeat(2,1fr); }
    .bank-grid  { grid-template-columns: repeat(2,1fr); }
    .main-header h1 { font-size: 1.5rem; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════

BANQUES_CAMEROUN = [
    {"nom": "Financial House S.A", "code": "FH",    "emoji": "🏛️"},
    {"nom": "Afriland First Bank",  "code": "AFB",   "emoji": "🌍"},
    {"nom": "SCB Cameroun",         "code": "SCB",   "emoji": "🏦"},
    {"nom": "BICEC",                "code": "BICEC", "emoji": "🏦"},
    {"nom": "UBA Cameroun",         "code": "UBA",   "emoji": "🌐"},
    {"nom": "Ecobank Cameroun",     "code": "ECO",   "emoji": "🌿"},
    {"nom": "SGBC",                 "code": "SGBC",  "emoji": "🔴"},
    {"nom": "BGFI Bank",            "code": "BGFI",  "emoji": "🔷"},
    {"nom": "CCA Bank",             "code": "CCA",   "emoji": "🏦"},
    {"nom": "Atlantic Bank",        "code": "ATL",   "emoji": "🌊"},
    {"nom": "Autre banque",         "code": "AUTRE", "emoji": "🏦"},
]

METHODES = {
    "vision":     "🔭 Gemini Vision — Image (Recommandé)",
    "hybrid":     "⚡ Gemini Hybride — Texte + IA",
    "pdfplumber": "📄 pdfplumber — Sans IA (Gratuit)",
}

METHOD_DESC = {
    "vision": """
    <div class="info-box">
        ✅ Lit le PDF comme un humain<br>
        ✅ Débit <b>ET</b> Crédit extraits fidèlement<br>
        ✅ Fonctionne sur PDF scanné et natif<br>
        ✅ Compatible toutes banques camerounaises<br>
        ⚠️ Consomme plus de tokens (images envoyées)
    </div>""",
    "hybrid": """
    <div class="info-box">
        ✅ ~85% moins de tokens vs Vision<br>
        ✅ Très rapide<br>
        ✅ Debug détaillé intégré<br>
        ⚠️ Nécessite un PDF texte natif lisible<br>
        ⚠️ Peut basculer en Vision si PDF scanné
    </div>""",
    "pdfplumber": """
    <div class="info-box">
        ✅ Entièrement gratuit, sans clé API<br>
        ✅ Rapide et fonctionne hors-ligne<br>
        ❌ Colonne Débit souvent vide<br>
        ❌ PDF scanné non supporté<br>
        ❌ Précision variable selon banque
    </div>""",
}


# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "df_clean":           None,
        "stats":              None,
        "account_info":       None,
        "file_name":          "",
        "extraction_done":    False,
        "extraction_method":  "vision",
        "token_counter":      None,
        "pages_text":         None,
        "show_confirm":       False,
        "estimate":           None,
        "banque_selectionnee": "Financial House S.A",
        "pdf_bytes_cache":    None,
        "debug_logs":         "",
        "debug_summary":      {},
        "debug_entries":      [],
        "account_name":       "",
        "account_id":         "",
        "rows_per_page":      100,
        "show_chart":         True,
        "show_tokens":        True,
        "show_debug":         True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state.token_counter is None:
        st.session_state.token_counter = TokenCounter()

init_state()


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def get_gemini_key() -> str:
    """Récupère la clé API Gemini."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return st.session_state.get("gemini_key_input", "")


def format_amount(val) -> str:
    """Formate un nombre avec séparateurs de milliers."""
    if val is None or (
        isinstance(val, float) and np.isnan(val)
    ):
        return ""
    try:
        return f"{float(val):,.0f}".replace(",", " ")
    except (ValueError, TypeError):
        return str(val)


def get_banque_info(nom: str) -> dict:
    """Retourne les infos d'une banque par son nom."""
    return next(
        (b for b in BANQUES_CAMEROUN if b["nom"] == nom),
        BANQUES_CAMEROUN[-1],
    )


def reset_app():
    """Remet l'application à l'état initial."""
    preserve = {
        "token_counter",
        "gemini_key_input",
        "account_name",
        "account_id",
        "rows_per_page",
        "show_chart",
        "show_tokens",
        "show_debug",
    }
    for k in list(st.session_state.keys()):
        if k not in preserve:
            del st.session_state[k]
    init_state()


def style_dataframe(df: pd.DataFrame):
    """Applique un style conditionnel au DataFrame."""

    def _row_style(row):
        libelle = str(row.get("Libellé", "")).lower()
        if re.search(r"solde", libelle):
            return [
                "background-color:#D6EAF8;"
                "font-weight:bold;color:#1A5276"
            ] * len(row)

        styles = [""] * len(row)
        cols   = list(row.index)

        if "Débit" in cols:
            idx = cols.index("Débit")
            val = row.get("Débit")
            if pd.notna(val) and val:
                styles[idx] = (
                    "background-color:#FADBD8;"
                    "color:#C0392B;font-weight:600"
                )

        if "Crédit" in cols:
            idx = cols.index("Crédit")
            val = row.get("Crédit")
            if pd.notna(val) and val:
                styles[idx] = (
                    "background-color:#D5F5E3;"
                    "color:#1E8449;font-weight:600"
                )
        return styles

    fmt = {
        "Débit":  lambda x: format_amount(x) if pd.notna(x) else "",
        "Crédit": lambda x: format_amount(x) if pd.notna(x) else "",
        "Solde":  lambda x: format_amount(x) if pd.notna(x) else "",
    }
    return df.style.apply(_row_style, axis=1).format(fmt)


def render_stat_card(
    col, css_class: str, icon: str,
    label: str, value: str, sub: str = ""
):
    """Rend une carte statistique HTML."""
    with col:
        sub_html = (
            f'<div class="stat-sub">{sub}</div>'
            if sub else ""
        )
        st.markdown(f"""
        <div class="stat-card {css_class}">
            <div class="stat-icon">{icon}</div>
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            {sub_html}
        </div>
        """, unsafe_allow_html=True)


def update_progress(step: int, message: str):
    """Met à jour la barre de progression globale."""
    if "progress_bar" in st.session_state:
        if step > 0:
            st.session_state.progress_bar.progress(
                min(step / 100, 1.0)
            )
    if "status_placeholder" in st.session_state:
        st.session_state.status_placeholder.markdown(
            f"""<div style='text-align:center;color:#1B3A5C;
            padding:0.5rem;font-size:0.95rem'>{message}</div>""",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Logo ──────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:0.5rem 0 1rem">
        <div style="font-size:2.2rem">🏦</div>
        <div style="font-weight:800;font-size:1rem;color:#1B3A5C">
            Bank Extractor
        </div>
        <div style="font-size:0.72rem;color:#95A5A6">
            Toutes banques du Cameroun
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── 1. Upload PDF ──────────────────────────────────────────
    st.markdown("#### 📂 Relevé bancaire")
    uploaded_file = st.file_uploader(
        "Glissez votre PDF ici",
        type=["pdf"],
        help="Formats : PDF natif ou scanné — max 50 MB",
        label_visibility="collapsed",
    )
    if uploaded_file:
        size_kb = uploaded_file.size / 1024
        st.markdown(f"""
        <div class="success-box">
            📄 <b>{uploaded_file.name}</b><br>
            <span style="color:#7F8C8D">
                {size_kb:.1f} KB
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── 2. Banque ──────────────────────────────────────────────
    st.markdown("#### 🏦 Banque émettrice")
    banque_noms = [b["nom"] for b in BANQUES_CAMEROUN]
    banque_sel  = st.selectbox(
        "Banque",
        options=banque_noms,
        index=0,
        label_visibility="collapsed",
    )
    st.session_state.banque_selectionnee = banque_sel
    banque_info = get_banque_info(banque_sel)
    st.caption(
        f"{banque_info['emoji']} {banque_sel} sélectionnée"
    )

    st.divider()

    # ── 3. Méthode d'extraction ────────────────────────────────
    st.markdown("#### 🤖 Méthode d'extraction")
    method_key = st.radio(
        "Méthode",
        options=list(METHODES.keys()),
        format_func=lambda x: METHODES[x],
        index=0,
        label_visibility="collapsed",
    )
    st.session_state.extraction_method = method_key
    st.markdown(
        METHOD_DESC.get(method_key, ""),
        unsafe_allow_html=True,
    )

    # ── 4. Clé API Gemini ──────────────────────────────────────
    if method_key in ("vision", "hybrid"):
        st.divider()
        st.markdown("#### 🔑 Clé API Gemini")

        existing_key = get_gemini_key()
        if existing_key:
            st.markdown("""
            <div class="success-box">
                ✅ Clé API configurée et active
            </div>
            """, unsafe_allow_html=True)
        else:
            key_input = st.text_input(
                "Clé API",
                type="password",
                placeholder="AIzaSy...",
                label_visibility="collapsed",
                help=(
                    "Clé gratuite sur : "
                    "https://aistudio.google.com/app/apikey"
                ),
            )
            st.session_state["gemini_key_input"] = key_input
            if not key_input:
                st.markdown("""
                <div class="warning-box">
                    🔗
                    <a href="https://aistudio.google.com/app/apikey"
                    target="_blank">
                        Obtenir une clé gratuite
                    </a>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    # ── 5. Informations compte ─────────────────────────────────
    with st.expander(
        "👤 Informations du titulaire", expanded=False
    ):
        st.session_state.account_name = st.text_input(
            "Nom du titulaire",
            value=st.session_state.account_name,
            placeholder="Nom complet",
        )
        st.session_state.account_id = st.text_input(
            "Numéro de compte",
            value=st.session_state.account_id,
            placeholder="Ex: 030304107...",
        )

    # ── 6. Options d'affichage ─────────────────────────────────
    with st.expander("⚙️ Options d'affichage", expanded=False):
        st.session_state.rows_per_page = st.select_slider(
            "Lignes par page",
            options=[25, 50, 100, 200, 500],
            value=st.session_state.rows_per_page,
        )
        st.session_state.show_chart = st.checkbox(
            "📈 Graphique du solde",
            value=st.session_state.show_chart,
        )
        st.session_state.show_tokens = st.checkbox(
            "📊 Utilisation des tokens",
            value=st.session_state.show_tokens,
        )
        st.session_state.show_debug = st.checkbox(
            "🔍 Logs de débogage",
            value=st.session_state.show_debug,
        )

    st.divider()

    # ── 7. Bouton reset ────────────────────────────────────────
    if st.button(
        "🔄 Nouvelle extraction",
        use_container_width=True,
    ):
        reset_app()
        st.rerun()

    st.markdown("""
    <div style="text-align:center;margin-top:1rem;
                font-size:0.72rem;color:#BDC3C7">
        v3.1.0 — Cameroun Edition
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HEADER PRINCIPAL
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🏦 Bank Statement Extractor</h1>
    <div class="subtitle">
        <span>
            Extraction intelligente de relevés bancaires PDF
        </span>
        <span class="badge badge-gemini">✨ Gemini AI</span>
        <span class="badge badge-info">
            🇨🇲 Toutes banques Cameroun
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE D'ACCUEIL — Aucun fichier chargé
# ══════════════════════════════════════════════════════════════
if (
    not st.session_state.extraction_done
    and not st.session_state.show_confirm
    and uploaded_file is None
):
    # Banques supportées
    st.markdown("### 🇨🇲 Banques supportées")
    bank_html = '<div class="bank-grid">'
    for b in BANQUES_CAMEROUN:
        bank_html += (
            f'<div class="bank-chip">'
            f'{b["emoji"]} {b["nom"]}</div>'
        )
    bank_html += "</div>"
    st.markdown(bank_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Comparaison méthodes
    col_l, col_m, col_r = st.columns(3)

    with col_l:
        st.markdown("""
        <div class="info-box">
            <b>🔭 Gemini Vision (Recommandé)</b><br><br>
            ✅ Lit le PDF comme un humain<br>
            ✅ Débit <b>ET</b> Crédit corrects<br>
            ✅ PDF scanné + natif<br>
            ✅ Toutes les banques<br>
            ✅ Libellés multi-lignes fusionnés<br>
            🔑 Clé API Gemini (gratuite)
        </div>
        """, unsafe_allow_html=True)

    with col_m:
        st.markdown("""
        <div class="info-box">
            <b>⚡ Gemini Hybride</b><br><br>
            ✅ 85% moins de tokens<br>
            ✅ Très rapide<br>
            ✅ Debug détaillé<br>
            ⚠️ PDF texte natif requis<br>
            ⚠️ Auto-bascule vers Vision<br>
            🔑 Clé API Gemini (gratuite)
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class="info-box">
            <b>📄 pdfplumber (Gratuit)</b><br><br>
            ✅ Aucune clé API<br>
            ✅ Rapide, hors-ligne<br>
            ❌ Colonne Débit vide<br>
            ❌ PDF scanné non supporté<br>
            ❌ Précision variable<br>
            &nbsp;
        </div>
        """, unsafe_allow_html=True)

    # Guide rapide
    with st.expander(
        "📖 Guide d'utilisation", expanded=False
    ):
        st.markdown("""
        **1.** Chargez votre PDF dans le panneau latéral

        **2.** Sélectionnez votre banque

        **3.** Choisissez la méthode :
        - **Vision** → meilleure précision (recommandé)
        - **Hybride** → plus économique en tokens
        - **pdfplumber** → gratuit mais moins précis

        **4.** Entrez votre clé API Gemini si besoin
        → [Obtenir une clé gratuite](https://aistudio.google.com/app/apikey)

        **5.** Cliquez **Lancer l'extraction**

        **6.** Vérifiez et exportez en **Excel** ou **CSV**
        """)

    # Zone upload visuelle
    st.markdown("""
    <div class="upload-zone">
        <div style="font-size:3.5rem">📄</div>
        <h3 style="color:#1B3A5C;margin:0.5rem 0">
            Chargez votre relevé PDF dans la barre latérale
        </h3>
        <p style="color:#95A5A6;margin:0">
            PDF natif ou scanné — toutes banques — max 50 MB
        </p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PRÊT À EXTRAIRE — Fichier chargé, pas encore confirmé
# ══════════════════════════════════════════════════════════════
if (
    not st.session_state.extraction_done
    and not st.session_state.show_confirm
    and uploaded_file is not None
):
    method    = st.session_state.extraction_method
    api_key   = get_gemini_key()
    banque    = st.session_state.banque_selectionnee
    b_info    = get_banque_info(banque)

    # Carte récapitulative
    st.markdown(f"""
    <div style="background:white;border-radius:14px;
                padding:1.5rem 2rem;
                box-shadow:0 2px 14px rgba(0,0,0,0.07);
                max-width:600px;margin:1.5rem auto;
                text-align:center">
        <div style="font-size:2.8rem">📄</div>
        <h3 style="color:#1B3A5C;margin:0.5rem 0">
            {uploaded_file.name}
        </h3>
        <div style="color:#7F8C8D;font-size:0.88rem;
                    margin-bottom:1rem">
            {uploaded_file.size / 1024:.1f} KB
            &nbsp;·&nbsp;
            {b_info['emoji']} {banque}
            &nbsp;·&nbsp;
            {METHODES[method]}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Alerte clé API manquante
    if method in ("vision", "hybrid") and not api_key:
        st.markdown("""
        <div class="error-box">
            ❌ <b>Clé API Gemini manquante !</b><br>
            Saisissez votre clé dans le panneau latéral
            gauche, ou passez en mode <b>pdfplumber</b>.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Bouton lancer
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        launch_btn = st.button(
            f"🚀 Lancer l'extraction — {banque}",
            use_container_width=True,
            type="primary",
        )

    if launch_btn:
        pdf_bytes = uploaded_file.read()
        st.session_state.pdf_bytes_cache = pdf_bytes

        # ── Mode Gemini : pré-calcul de l'estimation ──
        if method in ("vision", "hybrid"):

            if method == "hybrid":
                # Extraire le texte pour estimer les tokens
                with st.spinner(
                    "📄 Lecture du PDF "
                    "(calcul de l'estimation)..."
                ):
                    try:
                        temp_ext = GeminiExtractor(
                            api_key=api_key,
                            mode="hybrid",
                            verbose_debug=False,
                        )
                        pages = (
                            temp_ext
                            ._extract_text_from_pdf_public(
                                pdf_bytes
                            )
                        )
                        st.session_state.pages_text = pages
                        counter  = st.session_state.token_counter
                        estimate = (
                            counter.estimate_before_extraction(
                                pages
                            )
                        )
                        st.session_state.estimate = estimate
                    except Exception as e:
                        st.warning(
                            f"⚠️ Estimation impossible : {e} "
                            f"— extraction directe..."
                        )
                        st.session_state.estimate = None

            elif method == "vision":
                # Estimation basée sur le nombre de pages
                with st.spinner(
                    "📄 Lecture du PDF "
                    "(comptage des pages)..."
                ):
                    try:
                        with pdfplumber_open(pdf_bytes) as pdf:
                            n_pages = len(pdf.pages)
                    except Exception:
                        n_pages = 1

                    # ~2000 tokens/page en mode vision
                    in_tok  = n_pages * 2000
                    out_tok = n_pages * 800
                    cost    = (
                        in_tok / 1_000_000 * 0.075
                        + out_tok / 1_000_000 * 0.30
                    )
                    st.session_state.estimate = {
                        "pages":         n_pages,
                        "input_tokens":  in_tok,
                        "output_tokens": out_tok,
                        "total_tokens":  in_tok + out_tok,
                        "cost_usd":      round(cost, 4),
                        "cost_fcfa":     round(cost * 620, 1),
                        "within_free_tier": in_tok < 400_000,
                        "savings_pct":   0,
                    }

            st.session_state.show_confirm = True
            st.rerun()

        # ── Mode pdfplumber : pas d'estimation nécessaire ──
        else:
            st.session_state.estimate     = None
            st.session_state.show_confirm = True
            st.rerun()


# ══════════════════════════════════════════════════════════════
# CONFIRMATION AVANT EXTRACTION
# ══════════════════════════════════════════════════════════════
if (
    not st.session_state.extraction_done
    and st.session_state.show_confirm
    and uploaded_file is not None
):
    method    = st.session_state.extraction_method
    estimate  = st.session_state.estimate
    pdf_bytes = st.session_state.pdf_bytes_cache
    counter   = st.session_state.token_counter
    api_key   = get_gemini_key()
    banque    = st.session_state.banque_selectionnee
    b_info    = get_banque_info(banque)

    # ── Carte estimation tokens ────────────────────────────────
    if method in ("vision", "hybrid") and estimate:
        tier_ok    = estimate.get("within_free_tier", True)
        tier_color = "#27AE60" if tier_ok else "#E74C3C"
        tier_label = (
            "✅ Dans le quota gratuit Gemini"
            if tier_ok
            else "⚠️ Peut dépasser le quota gratuit"
        )
        savings = estimate.get("savings_pct", 0)
        mode_label = (
            "Hybride (texte)"
            if method == "hybrid"
            else "Vision (images)"
        )

        st.markdown(f"""
        <div class="token-estimate-card">
            <h4 style="margin:0 0 0.4rem;color:#1B3A5C">
                📊 Estimation avant extraction
                — Mode {mode_label}
            </h4>
            <p style="color:#7F8C8D;font-size:0.84rem;
                      margin:0 0 0.8rem">
                {"Mode Hybride : texte envoyé (pas d'images) → économies significatives"
                 if method == "hybrid"
                 else "Mode Vision : images envoyées à Gemini → précision maximale"}
            </p>
            <div class="token-grid">
                <div class="token-metric">
                    <div class="token-metric-value">
                        {estimate.get('pages', 0)}
                    </div>
                    <div class="token-metric-label">Pages</div>
                </div>
                <div class="token-metric">
                    <div class="token-metric-value">
                        ~{estimate.get('input_tokens', 0):,}
                    </div>
                    <div class="token-metric-label">
                        Tokens input
                    </div>
                </div>
                <div class="token-metric">
                    <div class="token-metric-value">
                        ~{estimate.get('output_tokens', 0):,}
                    </div>
                    <div class="token-metric-label">
                        Tokens output
                    </div>
                </div>
                <div class="token-metric">
                    <div class="token-metric-value"
                         style="color:#27AE60">
                        ~${estimate.get('cost_usd', 0):.4f}
                    </div>
                    <div class="token-metric-label">
                        Coût USD estimé
                    </div>
                </div>
            </div>
            <div style="background:{tier_color}18;
                        border:1px solid {tier_color}55;
                        border-radius:8px;padding:0.6rem 1rem;
                        color:{tier_color};font-weight:600;
                        font-size:0.87rem;text-align:center">
                {tier_label}
                &nbsp;·&nbsp;
                ~{estimate.get('cost_fcfa', 0):.0f} FCFA
                {"&nbsp;·&nbsp; Économie : ~" + str(savings) + "% vs mode image"
                 if savings > 0 else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif method == "pdfplumber":
        st.markdown("""
        <div class="success-box">
            📄 <b>Mode pdfplumber</b> — Extraction gratuite,
            aucun token consommé.
        </div>
        """, unsafe_allow_html=True)

    # ── Boutons Confirmer / Annuler ────────────────────────────
    st.markdown("---")
    col_ok, col_cancel = st.columns(2)
    with col_ok:
        confirm_btn = st.button(
            "✅ Confirmer et lancer l'extraction",
            use_container_width=True,
            type="primary",
        )
    with col_cancel:
        cancel_btn = st.button(
            "❌ Annuler",
            use_container_width=True,
        )

    if cancel_btn:
        st.session_state.show_confirm = False
        st.session_state.estimate     = None
        st.rerun()

    if confirm_btn:

        # ── Barre de progression ───────────────────────────────
        prog_bar     = st.progress(0)
        status_ph    = st.empty()

        st.session_state.progress_bar       = prog_bar
        st.session_state.status_placeholder = status_ph

        def _progress(step: int, msg: str):
            if step > 0:
                prog_bar.progress(min(step / 100, 1.0))
            status_ph.markdown(
                f"""<div style='text-align:center;
                color:#1B3A5C;padding:0.5rem;
                font-size:0.95rem'>{msg}</div>""",
                unsafe_allow_html=True,
            )

        # ── Lancer l'extraction ────────────────────────────────
        df_raw = None
        try:
            if method == "vision":
                _progress(5, "🔭 Initialisation Gemini Vision...")
                extractor = GeminiExtractor(
                    api_key=api_key,
                    mode="vision",
                    progress_callback=_progress,
                    verbose_debug=True,
                )
                df_raw = extractor.extract(pdf_bytes)

            elif method == "hybrid":
                _progress(5, "⚡ Initialisation Gemini Hybride...")
                extractor = GeminiExtractor(
                    api_key=api_key,
                    mode="hybrid",
                    progress_callback=_progress,
                    verbose_debug=True,
                )
                df_raw = extractor.extract(pdf_bytes)

            else:
                _progress(5, "📄 Extraction pdfplumber...")
                extractor = BankStatementExtractor(
                    progress_callback=_progress,
                )
                df_raw = extractor.extract(pdf_bytes)
                # Pas de logs de debug pour pdfplumber

            # ── Sauvegarder les logs de debug ─────────────────
            if method in ("vision", "hybrid"):
                st.session_state.debug_logs    = (
                    extractor.get_debug_logs()
                )
                st.session_state.debug_summary = (
                    extractor.get_debug_summary()
                )
                st.session_state.debug_entries = (
                    extractor.get_debug_entries()
                )

                # Enregistrer la consommation tokens
                pages   = st.session_state.pages_text or []
                in_tok  = (
                    estimate.get("input_tokens", 0)
                    if estimate else 0
                )
                out_tok = (
                    estimate.get("output_tokens", 0)
                    if estimate else 0
                )
                counter.record_extraction(
                    input_tokens  = in_tok,
                    output_tokens = out_tok,
                    pages         = (
                        estimate.get("pages", 1)
                        if estimate else 1
                    ),
                    file_name     = (
                        uploaded_file.name
                        if uploaded_file else ""
                    ),
                )

        except Exception as e:
            prog_bar.empty()
            status_ph.empty()
            st.markdown(f"""
            <div class="error-box">
                ❌ <b>Erreur lors de l'extraction :</b><br>
                {str(e)}
            </div>
            """, unsafe_allow_html=True)

            # Afficher les logs de debug si disponibles
            if method in ("vision", "hybrid") and \
               "extractor" in dir():
                with st.expander(
                    "🔍 Logs de debug (erreur)", expanded=True
                ):
                    st.text(extractor.get_debug_logs())

            st.session_state.show_confirm = False
            st.stop()

        # ── Nettoyage et finalisation ──────────────────────────
        _progress(92, "🧹 Nettoyage et structuration...")

        cleaner  = DataCleaner()
        df_clean = cleaner.clean(df_raw)
        stats    = cleaner.get_statistics(df_clean)

        account_info = {
            "account_name": st.session_state.account_name,
            "account_id":   st.session_state.account_id,
            "banque":       banque,
            "banque_emoji": b_info["emoji"],
            "period": (
                f"{stats.get('periode_debut', '')} → "
                f"{stats.get('periode_fin', '')}"
            ),
            "extraction_date": datetime.now().strftime(
                "%d/%m/%Y à %H:%M"
            ),
            "method": METHODES.get(method, method),
        }

        # ── Sauvegarder en session ─────────────────────────────
        st.session_state.df_clean         = df_clean
        st.session_state.stats            = stats
        st.session_state.account_info     = account_info
        st.session_state.file_name        = (
            uploaded_file.name if uploaded_file else ""
        )
        st.session_state.extraction_done  = True
        st.session_state.show_confirm     = False

        prog_bar.progress(1.0)
        _progress(100, "✅ Extraction terminée avec succès !")
        time.sleep(0.6)
        st.rerun()


# ══════════════════════════════════════════════════════════════
# RÉSULTATS
# ══════════════════════════════════════════════════════════════
if st.session_state.extraction_done:

    df    = st.session_state.df_clean
    stats = st.session_state.stats
    info  = st.session_state.account_info or {}

    rows_per_page = st.session_state.rows_per_page
    show_chart    = st.session_state.show_chart
    show_tokens   = st.session_state.show_tokens
    show_debug    = st.session_state.show_debug

    # ── DataFrame vide ─────────────────────────────────────────
    if df is None or df.empty:
        st.markdown("""
        <div class="error-box">
            ❌ <b>Aucune donnée extraite.</b><br>
            Vérifiez que votre PDF contient bien un tableau
            de transactions, ou essayez une autre méthode
            d'extraction.
        </div>
        """, unsafe_allow_html=True)
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if st.button(
                "🔄 Réessayer", type="primary",
                use_container_width=True
            ):
                reset_app()
                st.rerun()
        with col_r2:
            # Afficher les logs même en cas d'échec
            if st.session_state.debug_entries:
                if st.button(
                    "🔍 Voir les logs",
                    use_container_width=True
                ):
                    st.session_state.show_debug = True
        st.stop()

    # ── Bannière succès ────────────────────────────────────────
    st.markdown(f"""
    <div class="success-box">
        ✅ <b>{len(df)} lignes extraites</b>
        depuis <code>{st.session_state.file_name}</code>
        &nbsp;·&nbsp;
        {info.get('banque_emoji', '🏦')}
        <b>{info.get('banque', '')}</b>
        &nbsp;·&nbsp;
        {info.get('method', '')}
        &nbsp;·&nbsp;
        {info.get('extraction_date', '')}
    </div>
    """, unsafe_allow_html=True)

    # ── Cartes statistiques ────────────────────────────────────
    net_val = stats.get("net", 0)
    net_cls = "net-pos" if net_val >= 0 else "net-neg"
    net_ico = "📈" if net_val >= 0 else "📉"

    c1, c2, c3, c4, c5 = st.columns(5)

    render_stat_card(
        c1, "credit", "💚",
        "Total Crédits",
        format_amount(stats.get("total_credit", 0)),
        "Entrées"
    )
    render_stat_card(
        c2, "debit", "❤️",
        "Total Débits",
        format_amount(stats.get("total_debit", 0)),
        "Sorties"
    )
    render_stat_card(
        c3, net_cls, net_ico,
        "Flux Net",
        format_amount(net_val),
        "Crédit − Débit"
    )
    render_stat_card(
        c4, "count", "🔢",
        "Transactions",
        f"{stats.get('total_transactions', 0):,}".replace(",", " "),
        "Lignes"
    )
    render_stat_card(
        c5, "solde", "🏦",
        "Solde Final",
        format_amount(stats.get("solde_cloture")),
        info.get("banque", "")
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Détails du relevé ──────────────────────────────────────
    with st.expander(
        "📅 Informations du relevé", expanded=False
    ):
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("🏦 Banque",   info.get("banque", "N/A"))
        d2.metric("📅 Début",    stats.get("periode_debut", "N/A"))
        d3.metric("📅 Fin",      stats.get("periode_fin", "N/A"))
        d4.metric(
            "🔓 Solde ouverture",
            format_amount(stats.get("solde_ouverture")) or "N/A"
        )
        d5.metric(
            "🔒 Solde clôture",
            format_amount(stats.get("solde_cloture")) or "N/A"
        )
        if info.get("account_name"):
            st.caption(
                f"👤 {info['account_name']} — "
                f"N° {info.get('account_id', 'N/A')}"
            )

    # ── Instancier l'exporteur ─────────────────────────────────
    exporter = BankStatementExporter()

    # ── Barre d'outils ─────────────────────────────────────────
    st.markdown("### 📊 Données extraites")
    tb1, tb2, tb3, tb4, tb5 = st.columns([3, 1.3, 1.3, 1, 1])

    with tb1:
        search = st.text_input(
            "Recherche",
            placeholder=(
                "🔍 Filtrer par libellé, référence, "
                "montant..."
            ),
            label_visibility="collapsed",
        )

    with tb2:
        excel_bytes = exporter.to_excel(df, stats, info)
        fname_excel = (
            f"releve_"
            f"{info.get('banque', 'banque').replace(' ', '_')}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        )
        st.download_button(
            label="📥 Excel",
            data=excel_bytes,
            file_name=fname_excel,
            mime=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            ),
            use_container_width=True,
        )

    with tb3:
        csv_bytes = exporter.to_csv(df)
        fname_csv = (
            f"releve_"
            f"{info.get('banque', 'banque').replace(' ', '_')}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        )
        st.download_button(
            label="📄 CSV",
            data=csv_bytes,
            file_name=fname_csv,
            mime="text/csv",
            use_container_width=True,
        )

    with tb4:
        st.markdown(f"""
        <div style="text-align:center;padding:0.5rem;
                    background:#EBF5FB;border-radius:8px;
                    font-size:0.85rem;font-weight:700;
                    color:#1B3A5C;margin-top:2px">
            {len(df)} lignes
        </div>
        """, unsafe_allow_html=True)

    with tb5:
        if st.button(
            "🔄 Nouveau",
            use_container_width=True,
            help="Extraire un nouveau relevé",
        ):
            reset_app()
            st.rerun()

    # ── Filtrage ───────────────────────────────────────────────
    df_display = df.copy()
    if search:
        mask = df_display.apply(
            lambda row: row.astype(str)
            .str.contains(search, case=False, na=False)
            .any(),
            axis=1,
        )
        df_display = df_display[mask]
        st.caption(
            f"🔍 {len(df_display)} résultat(s) "
            f"pour « {search} »"
        )

    # ── Pagination ─────────────────────────────────────────────
    total       = len(df_display)
    total_pages = max(1, -(-total // rows_per_page))

    if total_pages > 1:
        pg1, pg2, pg3 = st.columns([2, 3, 2])
        with pg2:
            current_page = st.number_input(
                f"Page (1 à {total_pages})",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1,
            )
    else:
        current_page = 1

    start   = (current_page - 1) * rows_per_page
    end     = min(start + rows_per_page, total)
    df_page = df_display.iloc[start:end]

    # ── Tableau principal ──────────────────────────────────────
    st.dataframe(
        style_dataframe(df_page),
        use_container_width=True,
        height=min(650, max(300, (len(df_page) + 1) * 38 + 3)),
        hide_index=True,
    )
    st.caption(
        f"📋 Lignes {start + 1} à {end} sur {total} au total"
    )

    # ══════════════════════════════════════════════════════════
    # ÉDITEUR MANUEL
    # ══════════════════════════════════════════════════════════
    with st.expander(
        "✏️ Corriger les données manuellement",
        expanded=False
    ):
        st.markdown("""
        <div class="info-box">
            💡 <b>Mode édition</b> : cliquez sur une cellule
            pour la modifier. Utilisez les boutons
            <b>+</b> et <b>🗑️</b> pour ajouter / supprimer
            des lignes.
        </div>
        """, unsafe_allow_html=True)

        edited_df = st.data_editor(
            st.session_state.df_clean,
            use_container_width=True,
            num_rows="dynamic",
            height=420,
            column_config={
                "Date": st.column_config.TextColumn(
                    "Date",
                    help="Format : JJ/MM/AAAA",
                    max_chars=10,
                ),
                "Référence": st.column_config.TextColumn(
                    "Référence",
                    max_chars=20,
                ),
                "Libellé": st.column_config.TextColumn(
                    "Libellé",
                    max_chars=200,
                ),
                "Date_Valeur": st.column_config.TextColumn(
                    "Date Valeur",
                    help="Format : JJ/MM/AAAA",
                    max_chars=10,
                ),
                "Débit": st.column_config.NumberColumn(
                    "Débit",
                    format="%.0f",
                    min_value=0,
                ),
                "Crédit": st.column_config.NumberColumn(
                    "Crédit",
                    format="%.0f",
                    min_value=0,
                ),
                "Solde": st.column_config.NumberColumn(
                    "Solde",
                    format="%.0f",
                ),
            },
        )

        ec1, ec2, ec3 = st.columns([2, 2, 1])
        with ec1:
            if st.button(
                "💾 Sauvegarder les modifications",
                use_container_width=True,
                type="primary",
            ):
                st.session_state.df_clean = edited_df
                cleaner = DataCleaner()
                st.session_state.stats = (
                    cleaner.get_statistics(edited_df)
                )
                st.success("✅ Modifications sauvegardées !")
                st.rerun()

        with ec2:
            edited_excel = exporter.to_excel(
                edited_df, stats, info
            )
            st.download_button(
                "📥 Télécharger (version éditée)",
                data=edited_excel,
                file_name=(
                    f"releve_edite_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M')}"
                    f".xlsx"
                ),
                mime=(
                    "application/vnd.openxmlformats-"
                    "officedocument.spreadsheetml.sheet"
                ),
                use_container_width=True,
            )

        with ec3:
            if st.button("↩️ Annuler", use_container_width=True):
                st.rerun()

    # ══════════════════════════════════════════════════════════
    # GRAPHIQUE ÉVOLUTION DU SOLDE
    # ══════════════════════════════════════════════════════════
    if show_chart:
        with st.expander(
            "📈 Évolution du solde dans le temps",
            expanded=False
        ):
            df_chart = df[
                df["Date"].notna()
                & (df["Date"] != "")
                & df["Solde"].notna()
            ].copy()

            if not df_chart.empty:
                df_chart["_dt"] = pd.to_datetime(
                    df_chart["Date"],
                    format="%d/%m/%Y",
                    errors="coerce",
                )
                df_chart = (
                    df_chart
                    .dropna(subset=["_dt", "Solde"])
                    .sort_values("_dt")
                )

                if len(df_chart) >= 2:
                    st.line_chart(
                        df_chart.set_index("_dt")["Solde"],
                        use_container_width=True,
                        height=320,
                        color="#1B3A5C",
                    )
                    g1, g2, g3 = st.columns(3)
                    g1.metric(
                        "📈 Solde maximum",
                        format_amount(df_chart["Solde"].max()),
                    )
                    g2.metric(
                        "📉 Solde minimum",
                        format_amount(df_chart["Solde"].min()),
                    )
                    g3.metric(
                        "📊 Solde moyen",
                        format_amount(df_chart["Solde"].mean()),
                    )
                else:
                    st.info(
                        "Pas assez de points de données "
                        "pour afficher le graphique."
                    )
            else:
                st.info(
                    "Aucune date disponible "
                    "pour générer le graphique."
                )

    # ══════════════════════════════════════════════════════════
    # DASHBOARD TOKENS GEMINI
    # ══════════════════════════════════════════════════════════
    if show_tokens:
        with st.expander(
            "📊 Utilisation des tokens Gemini",
            expanded=False
        ):
            counter = st.session_state.token_counter

            if counter:
                sess = counter.get_session_stats()

                if sess["total_requests"] > 0:
                    # Métriques
                    tm1, tm2, tm3, tm4 = st.columns(4)
                    tm1.metric(
                        "📄 Fichiers traités",
                        sess["total_requests"],
                    )
                    tm2.metric(
                        "🔤 Tokens consommés",
                        f"{sess['total_input_tokens']:,}",
                    )
                    tm3.metric(
                        "💰 Coût total USD",
                        f"${sess['total_cost_usd']:.4f}",
                    )
                    tm4.metric(
                        "💵 Coût total FCFA",
                        f"~{sess['total_cost_fcfa']:.0f}",
                    )

                    # Barre de progression quota
                    used_pct = min(
                        sess["total_input_tokens"]
                        / 500_000 * 100,
                        100,
                    )
                    bar_clr = (
                        "#27AE60" if used_pct < 70
                        else "#F39C12" if used_pct < 90
                        else "#E74C3C"
                    )
                    st.markdown(f"""
                    <div style="margin:1rem 0">
                        <div style="display:flex;
                            justify-content:space-between;
                            margin-bottom:4px;
                            font-size:0.85rem;font-weight:600">
                            <span>
                                Quota gratuit mensuel
                            </span>
                            <span style="color:#7F8C8D">
                                {sess['total_input_tokens']:,}
                                / 500 000 tokens
                            </span>
                        </div>
                        <div style="background:#E0E0E0;
                            border-radius:10px;height:10px;
                            overflow:hidden">
                            <div style="width:{used_pct:.1f}%;
                                height:100%;
                                background:{bar_clr};
                                border-radius:10px;
                                transition:width 0.5s ease">
                            </div>
                        </div>
                        <div style="text-align:right;
                            font-size:0.78rem;
                            color:{bar_clr};margin-top:3px;
                            font-weight:600">
                            {used_pct:.1f}% utilisé
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Historique sessions
                    if sess.get("sessions"):
                        st.markdown("**Historique :**")
                        df_sess = pd.DataFrame(
                            sess["sessions"]
                        )
                        df_sess.columns = [
                            "Heure", "Fichier", "Pages",
                            "Tokens input", "Tokens output",
                            "Coût USD",
                        ]
                        df_sess["Coût USD"] = df_sess[
                            "Coût USD"
                        ].apply(lambda x: f"${x:.4f}")
                        st.dataframe(
                            df_sess,
                            use_container_width=True,
                            hide_index=True,
                        )

                    # Reset
                    if st.button(
                        "🔄 Remettre les compteurs à zéro"
                    ):
                        counter.reset()
                        st.success("✅ Compteurs remis à zéro")
                        st.rerun()

                else:
                    st.info(
                        "Aucune extraction Gemini effectuée "
                        "durant cette session."
                    )

                # Conseils
                with st.expander(
                    "💡 Conseils pour économiser les tokens"
                ):
                    st.markdown("""
                    | Pages | Mode | Tokens | Coût USD |
                    |-------|------|--------|----------|
                    | 1 | Vision | ~2 800 | ~$0.0002 |
                    | 1 | Hybride | ~500 | ~$0.00004 |
                    | 12 | Vision | ~33 000 | ~$0.003 |
                    | 12 | Hybride | ~6 000 | ~$0.0005 |

                    **🆓 Quota gratuit Gemini Flash :**
                    15 req/min — 1 500 req/jour

                    **💡 Astuces :**
                    - Utilisez le mode **Hybride** pour économiser
                    - Traitez un relevé à la fois
                    - gemini-1.5-flash est 10× moins cher que pro
                    """)

    # ══════════════════════════════════════════════════════════
    # LOGS DE DÉBOGAGE
    # ══════════════════════════════════════════════════════════
    if show_debug:
        with st.expander(
            "🔍 Logs de débogage de l'extraction",
            expanded=False
        ):
            summary = st.session_state.get(
                "debug_summary", {}
            )
            entries = st.session_state.get(
                "debug_entries", []
            )

            if not entries:
                st.info(
                    "Aucun log disponible. "
                    "Les logs sont générés avec les modes "
                    "Gemini Vision et Hybride."
                )
            else:
                # Résumé
                ds1, ds2, ds3, ds4 = st.columns(4)
                ds1.metric(
                    "📋 Logs total",
                    summary.get("total_logs", 0)
                )
                ds2.metric(
                    "▶️ Étapes",
                    summary.get("steps", 0)
                )
                ds3.metric(
                    "⚠️ Warnings",
                    summary.get("warnings", 0)
                )
                ds4.metric(
                    "❌ Erreurs",
                    summary.get("errors", 0)
                )

                # Erreurs critiques
                if summary.get("has_errors"):
                    st.markdown("**❌ Erreurs détectées :**")
                    for err in summary.get(
                        "error_details", []
                    ):
                        st.error(
                            f"**{err['msg']}**\n\n"
                            f"{err.get('detail', '')[:400]}"
                        )

                st.markdown("---")

                # Filtres
                fl1, fl2 = st.columns(2)
                with fl1:
                    level_filter = st.multiselect(
                        "Filtrer par niveau",
                        options=[
                            "ALL", "ERROR", "WARNING",
                            "SUCCESS", "API", "DATA",
                            "TOKEN", "DEBUG", "INFO", "STEP",
                        ],
                        default=[
                            "ERROR", "WARNING",
                            "SUCCESS", "API", "DATA", "STEP",
                        ],
                    )
                with fl2:
                    search_log = st.text_input(
                        "Rechercher dans les logs",
                        placeholder="mot-clé...",
                    )

                # Appliquer les filtres
                filtered = entries
                if level_filter and "ALL" not in level_filter:
                    filtered = [
                        e for e in filtered
                        if e["level"] in level_filter
                    ]
                if search_log:
                    sl = search_log.lower()
                    filtered = [
                        e for e in filtered
                        if sl in e["message"].lower()
                        or sl in e.get("detail", "").lower()
                    ]

                st.caption(
                    f"📋 {len(filtered)} log(s) affiché(s) "
                    f"sur {len(entries)} total"
                )

                # Couleurs par niveau
                COLOR_MAP = {
                    "ERROR":   "#FDEDEC",
                    "WARNING": "#FEF9E7",
                    "SUCCESS": "#EAFAF1",
                    "API":     "#EBF5FB",
                    "DATA":    "#F4ECF7",
                    "TOKEN":   "#FEF5E7",
                    "STEP":    "#E8F4FD",
                    "DEBUG":   "#F8F9FA",
                    "INFO":    "#FFFFFF",
                }

                # Afficher les logs
                for entry in filtered:
                    bg = COLOR_MAP.get(
                        entry["level"], "#FFFFFF"
                    )
                    detail_html = ""
                    detail_txt  = entry.get("detail", "")
                    if detail_txt:
                        # Tronquer si trop long
                        if len(detail_txt) > 600:
                            detail_txt = (
                                detail_txt[:600]
                                + "... [tronqué]"
                            )
                        detail_html = (
                            f'<div class="log-detail">'
                            f'{detail_txt}</div>'
                        )

                    st.markdown(f"""
                    <div class="log-entry"
                         style="background:{bg}">
                        <span style="color:#7F8C8D;
                                     font-size:0.72rem">
                            [{entry['timestamp']}]
                        </span>
                        &nbsp;
                        <b>{entry['icon']}</b>
                        &nbsp;
                        {entry['message']}
                        {detail_html}
                    </div>
                    """, unsafe_allow_html=True)

                # Télécharger les logs
                logs_text = st.session_state.get(
                    "debug_logs", ""
                )
                if logs_text:
                    st.markdown("---")
                    st.download_button(
                        "📥 Télécharger les logs complets (.txt)",
                        data=logs_text,
                        file_name=(
                            f"debug_logs_"
                            f"{datetime.now().strftime('%Y%m%d_%H%M')}"
                            f".txt"
                        ),
                        mime="text/plain",
                        use_container_width=False,
                    )


# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    🏦 <b>Bank Statement Extractor</b> v3.1.0
    &nbsp;|&nbsp;
    🇨🇲 Compatible toutes banques du Cameroun
    &nbsp;|&nbsp;
    ✨ Powered by Google Gemini AI + pdfplumber<br>
    <span style="font-size:0.75rem;color:#D5D8DC">
        Financial House · Afriland First Bank · SCB ·
        BICEC · UBA · Ecobank · SGBC · BGFI ·
        CCA Bank · Atlantic Bank
    </span>
</div>
""", unsafe_allow_html=True)
