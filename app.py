"""
╔══════════════════════════════════════════════════════════════╗
║         BANK STATEMENT EXTRACTOR — Streamlit + Gemini        ║
║         Compatible : Toutes banques du Cameroun              ║
║         Version : 3.0.0                                      ║
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
        135deg, #4285F4, #EA4335, #FBBC04, #34A853
    );
    color: white;
}
.badge-success { background: #D5F5E3; color: #1E8449; }
.badge-warning { background: #FDEBD0; color: #784212; }
.badge-info    { background: #D6EAF8; color: #1A5276; }
.badge-danger  { background: #FADBD8; color: #922B21; }

/* ── Cartes stats ── */
.stat-card {
    background: white;
    border-radius: 14px;
    padding: 1.3rem 1rem;
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
    font-size: 1.4rem;
    font-weight: 800;
    color: #2C3E50;
    line-height: 1.2;
}
.stat-sub {
    font-size: 0.72rem;
    color: #BDC3C7;
    margin-top: 3px;
}

/* ── Boîtes d'info ── */
.info-box {
    background: #EBF5FB;
    border-left: 4px solid #2E75B6;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.1rem;
    margin: 0.6rem 0;
    font-size: 0.88rem;
    line-height: 1.6;
}
.success-box {
    background: #EAFAF1;
    border-left: 4px solid #27AE60;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.1rem;
    margin: 0.6rem 0;
    font-size: 0.88rem;
}
.warning-box {
    background: #FEF9E7;
    border-left: 4px solid #F39C12;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.1rem;
    margin: 0.6rem 0;
    font-size: 0.88rem;
}
.error-box {
    background: #FDEDEC;
    border-left: 4px solid #E74C3C;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.1rem;
    margin: 0.6rem 0;
    font-size: 0.88rem;
}

/* ── Upload zone ── */
.upload-zone {
    border: 2.5px dashed #AED6F1;
    border-radius: 14px;
    padding: 2.5rem;
    text-align: center;
    background: white;
    transition: all 0.3s;
    cursor: pointer;
}
.upload-zone:hover {
    border-color: #2E75B6;
    background: #F0F8FF;
}

/* ── Banques ── */
.bank-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.5rem;
    margin: 0.5rem 0;
}
.bank-chip {
    background: #F8F9FA;
    border: 1px solid #E0E0E0;
    border-radius: 8px;
    padding: 5px 8px;
    font-size: 0.75rem;
    text-align: center;
    color: #2C3E50;
    font-weight: 500;
}

/* ── Estimation tokens ── */
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
    font-size: 1.3rem;
    font-weight: 800;
    color: #1B3A5C;
}
.token-metric-label {
    font-size: 0.7rem;
    color: #95A5A6;
    margin-top: 3px;
}

/* ── Progress bar custom ── */
.progress-container {
    background: white;
    border-radius: 14px;
    padding: 2rem;
    box-shadow: 0 2px 14px rgba(0,0,0,0.07);
    text-align: center;
    margin: 2rem auto;
    max-width: 600px;
}

/* ── Tableau ── */
.stDataFrame {
    border-radius: 12px !important;
}

/* ── Toolbar ── */
.toolbar-container {
    background: white;
    border-radius: 12px 12px 0 0;
    padding: 1rem 1.2rem;
    border-bottom: 1px solid #E8ECF0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.8rem;
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
    .token-grid  { grid-template-columns: repeat(2,1fr); }
    .bank-grid   { grid-template-columns: repeat(2,1fr); }
    .main-header h1 { font-size: 1.5rem; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════

# Banques camerounaises supportées
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
    "gemini":      "🤖 Gemini AI — Hybride (Recommandé)",
    "pdfplumber":  "📄 pdfplumber — Sans IA (Gratuit)",
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
        "extraction_method":  "gemini",
        "token_counter":      None,
        "pages_text":         None,
        "show_confirm":       False,
        "estimate":           None,
        "banque_selectionnee": "Financial House S.A",
        "pdf_bytes_cache":    None,
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
    """Récupère la clé Gemini (secrets Streamlit ou saisie)."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return st.session_state.get("gemini_key_input", "")


def format_amount(val) -> str:
    """Formate un nombre avec séparateurs de milliers."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    try:
        n = float(val)
        return f"{n:,.0f}".replace(",", " ")
    except (ValueError, TypeError):
        return str(val)


def reset_app():
    """Remet l'application à l'état initial."""
    keys_to_keep = {"token_counter", "gemini_key_input"}
    for k in list(st.session_state.keys()):
        if k not in keys_to_keep:
            del st.session_state[k]
    init_state()


def style_dataframe(df: pd.DataFrame):
    """
    Applique un style conditionnel au DataFrame :
    - Vert pour les crédits
    - Rouge pour les débits
    - Bleu pour les lignes de solde
    """
    def _row_style(row):
        libelle = str(row.get("Libellé", "")).lower()

        # Ligne de solde → fond bleu clair
        if re.search(r"solde", libelle):
            return [
                "background-color:#D6EAF8;"
                "font-weight:bold;color:#1A5276"
            ] * len(row)

        styles = [""] * len(row)
        cols   = list(row.index)

        # Débit → rouge clair
        if "Débit" in cols:
            idx = cols.index("Débit")
            val = row.get("Débit")
            if pd.notna(val) and val:
                styles[idx] = (
                    "background-color:#FADBD8;"
                    "color:#C0392B;font-weight:600"
                )

        # Crédit → vert clair
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
    """Rend une carte statistique."""
    with col:
        st.markdown(f"""
        <div class="stat-card {css_class}">
            <div class="stat-icon">{icon}</div>
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            {"" if not sub else
             f'<div class="stat-sub">{sub}</div>'}
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:

    # Logo / Titre sidebar
    st.markdown("""
    <div style="text-align:center;padding:0.5rem 0 1rem">
        <div style="font-size:2.2rem">🏦</div>
        <div style="font-weight:800;font-size:1rem;
                    color:#1B3A5C">Bank Extractor</div>
        <div style="font-size:0.72rem;color:#95A5A6">
            Toutes banques du Cameroun
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── 1. Upload PDF ──────────────────────────────
    st.markdown("#### 📂 Relevé bancaire")
    uploaded_file = st.file_uploader(
        "Glissez votre PDF ici",
        type=["pdf"],
        help="Formats supportés : PDF natif ou scanné — max 50 MB",
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

    # ── 2. Banque ──────────────────────────────────
    st.markdown("#### 🏦 Banque")
    banque_noms = [b["nom"] for b in BANQUES_CAMEROUN]
    banque_sel  = st.selectbox(
        "Sélectionner la banque",
        options=banque_noms,
        index=0,
        label_visibility="collapsed",
    )
    st.session_state.banque_selectionnee = banque_sel

    # Trouver l'emoji de la banque
    banque_info = next(
        (b for b in BANQUES_CAMEROUN if b["nom"] == banque_sel),
        BANQUES_CAMEROUN[-1]
    )
    st.caption(f"{banque_info['emoji']} {banque_sel} sélectionnée")

    st.divider()

    # ── 3. Méthode d'extraction ────────────────────
    st.markdown("#### 🤖 Méthode d'extraction")
    method_key = st.radio(
        "Méthode",
        options=list(METHODES.keys()),
        format_func=lambda x: METHODES[x],
        index=0,
        label_visibility="collapsed",
    )
    st.session_state.extraction_method = method_key

    # Afficher les avantages de chaque méthode
    if method_key == "gemini":
        st.markdown("""
        <div class="info-box">
            ✅ Précision maximale<br>
            ✅ Débit & Crédit exacts<br>
            ✅ Fonctionne sur tout PDF<br>
            ✅ ~85% moins de tokens<br>
            🔑 Clé API requise
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            ✅ Gratuit, sans clé API<br>
            ✅ Rapide<br>
            ⚠️ Colonne Débit souvent vide<br>
            ⚠️ PDF natif uniquement
        </div>
        """, unsafe_allow_html=True)

    # ── 4. Clé API Gemini ──────────────────────────
    if method_key == "gemini":
        st.divider()
        st.markdown("#### 🔑 Clé API Gemini")

        existing = get_gemini_key()
        if existing:
            st.markdown("""
            <div class="success-box">
                ✅ Clé API configurée
            </div>
            """, unsafe_allow_html=True)
        else:
            key_input = st.text_input(
                "Clé API",
                type="password",
                placeholder="AIzaSy...",
                label_visibility="collapsed",
                help=(
                    "Obtenez une clé gratuite sur : "
                    "https://aistudio.google.com/app/apikey"
                )
            )
            st.session_state["gemini_key_input"] = key_input
            if not key_input:
                st.markdown("""
                <div class="warning-box">
                    🔗 <a href="https://aistudio.google.com/app/apikey"
                    target="_blank">Obtenir une clé gratuite</a>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    # ── 5. Informations compte ─────────────────────
    with st.expander("👤 Informations compte", expanded=False):
        account_name = st.text_input(
            "Titulaire du compte",
            value="",
            placeholder="Nom complet du titulaire",
        )
        account_id = st.text_input(
            "Numéro de compte",
            value="",
            placeholder="Ex: 030304107...",
        )

    st.divider()

    # ── 6. Options d'affichage ─────────────────────
    with st.expander("⚙️ Options d'affichage", expanded=False):
        rows_per_page = st.select_slider(
            "Lignes par page",
            options=[25, 50, 100, 200, 500],
            value=100,
        )
        show_chart = st.checkbox(
            "Afficher le graphique de solde", value=True
        )
        show_tokens = st.checkbox(
            "Afficher l'utilisation des tokens", value=True
        )

    st.divider()

    # ── 7. Bouton reset ────────────────────────────
    if st.button(
        "🔄 Nouvelle extraction",
        use_container_width=True,
        help="Remet l'application à zéro"
    ):
        reset_app()
        st.rerun()

    # ── Version ────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;margin-top:1rem;
                font-size:0.72rem;color:#BDC3C7">
        v3.0.0 — Cameroun Edition
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HEADER PRINCIPAL
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
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
# SECTION : PAS DE FICHIER CHARGÉ
# ══════════════════════════════════════════════════════════════
if not st.session_state.extraction_done and uploaded_file is None:

    # Guide des banques supportées
    st.markdown("### 🇨🇲 Banques supportées")
    bank_html = '<div class="bank-grid">'
    for b in BANQUES_CAMEROUN:
        bank_html += f"""
        <div class="bank-chip">
            {b['emoji']} {b['nom']}
        </div>"""
    bank_html += "</div>"
    st.markdown(bank_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Deux colonnes : comparaison méthodes
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
        <div class="info-box">
            <b>🤖 Mode Gemini AI (Recommandé)</b><br><br>
            ✅ Lit le PDF comme un humain<br>
            ✅ Débit <b>ET</b> Crédit extraits correctement<br>
            ✅ Libellés multi-lignes fusionnés<br>
            ✅ Compatible PDF scannés & natifs<br>
            ✅ 85% de tokens économisés (mode hybride)<br>
            ✅ Fonctionne sur toutes les banques<br>
            🔑 Nécessite une clé API Gemini (gratuite)
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class="info-box">
            <b>📄 Mode pdfplumber (Basique)</b><br><br>
            ✅ Entièrement gratuit<br>
            ✅ Rapide (pas d'appel API)<br>
            ✅ Fonctionne hors-ligne<br>
            ⚠️ Colonne Débit souvent vide<br>
            ⚠️ Mauvais sur PDF scannés<br>
            ⚠️ Précision variable selon banque<br>
            ❌ Ne comprend pas le contexte
        </div>
        """, unsafe_allow_html=True)

    # Guide d'utilisation
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📖 Guide d'utilisation rapide", expanded=False):
        st.markdown("""
        #### Étapes pour extraire vos données :

        **1. Chargez votre PDF**
        → Utilisez le panneau latéral gauche pour importer
          votre relevé bancaire (format PDF)

        **2. Sélectionnez votre banque**
        → Choisissez votre banque dans la liste déroulante
          pour optimiser l'extraction

        **3. Choisissez la méthode d'extraction**
        → **Gemini AI** : meilleure précision (recommandé)
        → **pdfplumber** : rapide et gratuit

        **4. Configurez la clé API** (si Gemini sélectionné)
        → Obtenez une clé gratuite sur
          [Google AI Studio](https://aistudio.google.com/app/apikey)

        **5. Lancez l'extraction**
        → Vérifiez l'estimation de tokens
        → Confirmez et attendez le résultat

        **6. Exportez vos données**
        → Excel formaté ou CSV compatible
        """)

    st.markdown("""
    <div class="upload-zone">
        <div style="font-size:3.5rem">📄</div>
        <h3 style="color:#1B3A5C;margin:0.5rem 0">
            Chargez votre relevé PDF dans la barre latérale
        </h3>
        <p style="color:#95A5A6;margin:0">
            Accepte tous les formats PDF (natif ou scanné)
            jusqu'à 50 MB
        </p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SECTION : FICHIER CHARGÉ — PRÊT À EXTRAIRE
# ══════════════════════════════════════════════════════════════
if (
    not st.session_state.extraction_done
    and uploaded_file is not None
    and not st.session_state.show_confirm
):
    method    = st.session_state.extraction_method
    api_key   = get_gemini_key()
    banque    = st.session_state.banque_selectionnee

    # Carte récapitulative
    st.markdown(f"""
    <div style="background:white;border-radius:14px;
                padding:1.5rem 2rem;
                box-shadow:0 2px 14px rgba(0,0,0,0.07);
                max-width:580px;margin:1.5rem auto;
                text-align:center">
        <div style="font-size:2.8rem">📄</div>
        <h3 style="color:#1B3A5C;margin:0.5rem 0">
            {uploaded_file.name}
        </h3>
        <div style="color:#7F8C8D;font-size:0.88rem;
                    margin-bottom:1rem">
            {uploaded_file.size/1024:.1f} KB
            &nbsp;·&nbsp;
            {banque_info['emoji']} {banque}
            &nbsp;·&nbsp;
            {METHODES[method]}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Vérification clé API
    if method == "gemini" and not api_key:
        st.markdown("""
        <div class="error-box">
            ❌ <b>Clé API Gemini manquante !</b><br>
            Saisissez votre clé dans le panneau latéral,
            ou passez en mode pdfplumber.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Bouton principal
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

        # ── Mode Gemini : pré-extraction texte + estimation ──
        if method == "gemini":
            with st.spinner(
                "📄 Lecture du PDF (calcul de l'estimation)..."
            ):
                try:
                    temp_ext = GeminiExtractor(api_key=api_key)
                    pages    = temp_ext._extract_text_from_pdf(
                        pdf_bytes
                    )
                    st.session_state.pages_text = pages

                    counter  = st.session_state.token_counter
                    estimate = counter.estimate_before_extraction(pages)
                    st.session_state.estimate    = estimate
                    st.session_state.show_confirm = True
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Erreur lecture PDF : {e}")
                    st.stop()

        # ── Mode pdfplumber : extraction directe ──
        else:
            st.session_state.show_confirm = True
            st.session_state.estimate     = None
            st.rerun()


# ══════════════════════════════════════════════════════════════
# SECTION : CONFIRMATION AVANT EXTRACTION (Gemini uniquement)
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

    # ── Afficher l'estimation tokens (Gemini) ──
    if method == "gemini" and estimate:

        tier_ok    = estimate.get("within_free_tier", True)
        tier_color = "#27AE60" if tier_ok else "#E74C3C"
        tier_label = "✅ Dans le quota gratuit" \
                     if tier_ok \
                     else "⚠️ Peut dépasser le quota gratuit"

        st.markdown(f"""
        <div class="token-estimate-card">
            <h4 style="margin:0 0 0.5rem;color:#1B3A5C">
                📊 Estimation de consommation Gemini
            </h4>
            <p style="color:#7F8C8D;font-size:0.85rem;margin:0 0 1rem">
                Calculé sur le texte extrait du PDF
                (méthode hybride — sans envoi d'images)
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
                        Coût estimé (USD)
                    </div>
                </div>
            </div>
            <div style="background:{tier_color}18;
                        border:1px solid {tier_color}44;
                        border-radius:8px;padding:0.6rem 1rem;
                        color:{tier_color};font-weight:600;
                        font-size:0.88rem;text-align:center">
                {tier_label}
                &nbsp;·&nbsp;
                ~{estimate.get('cost_fcfa', 0):.0f} FCFA
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Comparaison avec mode image
        savings = estimate.get("savings_pct", 0)
        st.markdown(f"""
        <div class="success-box">
            💡 <b>Mode Hybride actif</b> : En utilisant le texte
            plutôt que les images, vous économisez environ
            <b>{savings}% de tokens</b> par rapport au mode
            vision classique.
        </div>
        """, unsafe_allow_html=True)

    # ── Boutons Confirmer / Annuler ──
    st.markdown("---")
    col_ok, col_cancel = st.columns(2)

    with col_ok:
        confirm_btn = st.button(
            "✅ Confirmer et extraire",
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

        # ── Barre de progression ──
        progress_bar       = st.progress(0)
        status_placeholder = st.empty()

        def update_progress(step: int, message: str):
            if step > 0:
                progress_bar.progress(min(step / 100, 1.0))
            status_placeholder.markdown(
                f"""<div style='text-align:center;
                    color:#1B3A5C;padding:0.5rem;
                    font-size:0.95rem'>{message}</div>""",
                unsafe_allow_html=True,
            )

        # ── Extraction ──
        try:
            if method == "gemini":
                api_key = get_gemini_key()
                update_progress(5, "🤖 Initialisation Gemini AI...")
                extractor = GeminiExtractor(
                    api_key=api_key,
                    progress_callback=update_progress,
                )
                df_raw = extractor.extract(pdf_bytes)

                # Enregistrer la consommation réelle
                pages  = st.session_state.pages_text or []
                in_tok = estimate.get("input_tokens", 0) \
                         if estimate else 0
                out_tok = estimate.get("output_tokens", 0) \
                          if estimate else 0
                counter.record_extraction(
                    input_tokens  = in_tok,
                    output_tokens = out_tok,
                    pages         = len(pages),
                    file_name     = uploaded_file.name,
                )

            else:
                update_progress(5, "📄 Extraction pdfplumber...")
                extractor = BankStatementExtractor(
                    progress_callback=update_progress,
                )
                df_raw = extractor.extract(pdf_bytes)

        except Exception as e:
            progress_bar.empty()
            status_placeholder.empty()
            st.error(f"❌ Erreur lors de l'extraction : {str(e)}")
            st.session_state.show_confirm = False
            st.stop()

        # ── Nettoyage ──
        update_progress(92, "🧹 Nettoyage et structuration...")
        cleaner  = DataCleaner()
        df_clean = cleaner.clean(df_raw)
        stats    = cleaner.get_statistics(df_clean)

        # ── Infos compte ──
        account_info = {
            "account_name": account_name
                            if "account_name" in dir()
                            else "",
            "account_id":   account_id
                            if "account_id" in dir()
                            else "",
            "banque":       st.session_state.banque_selectionnee,
            "period": (
                f"{stats.get('periode_debut', '')} → "
                f"{stats.get('periode_fin', '')}"
            ),
            "extraction_date": datetime.now().strftime(
                "%d/%m/%Y à %H:%M"
            ),
            "method": (
                "Gemini AI (Hybride)"
                if method == "gemini"
                else "pdfplumber"
            ),
        }

        # ── Sauvegarder en session ──
        st.session_state.df_clean        = df_clean
        st.session_state.stats           = stats
        st.session_state.account_info    = account_info
        st.session_state.file_name       = uploaded_file.name \
                                           if uploaded_file else ""
        st.session_state.extraction_done = True
        st.session_state.show_confirm    = False

        progress_bar.progress(1.0)
        update_progress(100, "✅ Extraction terminée !")
        time.sleep(0.5)
        st.rerun()


# ══════════════════════════════════════════════════════════════
# SECTION : RÉSULTATS
# ══════════════════════════════════════════════════════════════
if st.session_state.extraction_done:

    df    = st.session_state.df_clean
    stats = st.session_state.stats
    info  = st.session_state.account_info or {}

    # ── DataFrame vide ──
    if df is None or df.empty:
        st.markdown("""
        <div class="error-box">
            ❌ <b>Aucune donnée extraite.</b><br>
            Vérifiez que votre PDF contient bien un tableau
            de transactions lisible, ou essayez l'autre
            méthode d'extraction.
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔄 Réessayer", type="primary"):
            reset_app()
            st.rerun()
        st.stop()

    # ── Bannière succès ──
    method_label = info.get("method", "")
    banque_label = info.get("banque", "")
    st.markdown(f"""
    <div class="success-box">
        ✅ <b>{len(df)} lignes extraites</b> depuis
        <code>{st.session_state.file_name}</code>
        &nbsp;·&nbsp;
        {banque_info['emoji']} <b>{banque_label}</b>
        &nbsp;·&nbsp;
        <b>{method_label}</b>
        &nbsp;·&nbsp;
        {info.get('extraction_date', '')}
    </div>
    """, unsafe_allow_html=True)

    # ── Cartes statistiques ──
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
        "Crédit - Débit"
    )
    render_stat_card(
        c4, "count", "🔢",
        "Transactions",
        f"{stats.get('total_transactions', 0):,}".replace(",", " "),
        "Lignes extraites"
    )
    render_stat_card(
        c5, "credit" if (stats.get("solde_cloture") or 0) >= 0
            else "debit",
        "🏦",
        "Solde Final",
        format_amount(stats.get("solde_cloture")),
        info.get("banque", "")
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Détails période ──
    with st.expander(
        "📅 Informations du relevé", expanded=False
    ):
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric(
            "🏦 Banque",
            info.get("banque", "N/A")
        )
        d2.metric(
            "📅 Début",
            stats.get("periode_debut", "N/A")
        )
        d3.metric(
            "📅 Fin",
            stats.get("periode_fin", "N/A")
        )
        d4.metric(
            "🔓 Solde ouverture",
            format_amount(stats.get("solde_ouverture"))
            or "N/A"
        )
        d5.metric(
            "🔒 Solde clôture",
            format_amount(stats.get("solde_cloture"))
            or "N/A"
        )

    # ── Initialiser exporter ──
    exporter = BankStatementExporter()

    # ── Toolbar ──
    st.markdown("### 📊 Données extraites")
    tb1, tb2, tb3, tb4, tb5 = st.columns([3, 1.2, 1.2, 1, 1])

    with tb1:
        search = st.text_input(
            "Recherche",
            placeholder=(
                "🔍 Filtrer par libellé, référence, montant..."
            ),
            label_visibility="collapsed",
        )

    with tb2:
        excel_bytes = exporter.to_excel(df, stats, info)
        st.download_button(
            label="📥 Télécharger Excel",
            data=excel_bytes,
            file_name=(
                f"releve_{banque_label.replace(' ', '_')}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
            ),
            mime=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            ),
            use_container_width=True,
        )

    with tb3:
        csv_bytes = exporter.to_csv(df)
        st.download_button(
            label="📄 Télécharger CSV",
            data=csv_bytes,
            file_name=(
                f"releve_{banque_label.replace(' ', '_')}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

    with tb4:
        total_rows = len(df)
        st.markdown(f"""
        <div style="text-align:center;padding:0.5rem;
                    background:#F0F4F8;border-radius:8px;
                    font-size:0.85rem;font-weight:600;
                    color:#1B3A5C">
            {total_rows} lignes
        </div>
        """, unsafe_allow_html=True)

    with tb5:
        if st.button(
            "🔄 Nouveau",
            use_container_width=True,
            help="Extraire un nouveau relevé"
        ):
            reset_app()
            st.rerun()

    # ── Filtrage ──
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

    # ── Pagination ──
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

    # ── Tableau principal ──
    st.dataframe(
        style_dataframe(df_page),
        use_container_width=True,
        height=min(650, max(300, (len(df_page) + 1) * 38 + 3)),
        hide_index=True,
    )
    st.caption(
        f"📋 Affichage des lignes {start+1} à {end} "
        f"sur {total} au total"
    )

    # ── Éditeur manuel ──
    with st.expander(
        "✏️ Corriger les données manuellement", expanded=False
    ):
        st.markdown("""
        <div class="info-box">
            💡 <b>Mode édition</b> : Cliquez sur
            n'importe quelle cellule pour la modifier.
            Ajoutez ou supprimez des lignes avec les
            boutons + et 🗑️
        </div>
        """, unsafe_allow_html=True)

        edited_df = st.data_editor(
            st.session_state.df_clean,
            use_container_width=True,
            num_rows="dynamic",
            height=420,
            column_config={
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
                "Date": st.column_config.TextColumn(
                    "Date",
                    help="Format : JJ/MM/AAAA",
                    max_chars=10,
                ),
                "Date_Valeur": st.column_config.TextColumn(
                    "Date Valeur",
                    help="Format : JJ/MM/AAAA",
                    max_chars=10,
                ),
            },
        )

        sc1, sc2, sc3 = st.columns([2, 2, 1])
        with sc1:
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
                st.success("✅ Données mises à jour !")
                st.rerun()
        with sc2:
            # Télécharger les données éditées
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
        with sc3:
            if st.button(
                "↩️ Annuler", use_container_width=True
            ):
                st.rerun()

    # ── Graphique évolution du solde ──
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
                    df_chart.dropna(subset=["_dt", "Solde"])
                    .sort_values("_dt")
                )

                if len(df_chart) >= 2:
                    # Graphique principal
                    st.line_chart(
                        df_chart.set_index("_dt")["Solde"],
                        use_container_width=True,
                        height=320,
                        color="#1B3A5C",
                    )

                    # Stats du graphique
                    g1, g2, g3 = st.columns(3)
                    g1.metric(
                        "📈 Solde max",
                        format_amount(df_chart["Solde"].max()),
                    )
                    g2.metric(
                        "📉 Solde min",
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
                    "Aucune donnée de date disponible "
                    "pour le graphique."
                )

    # ── Dashboard tokens Gemini ──
    if show_tokens:
        with st.expander(
            "📊 Utilisation des tokens Gemini",
            expanded=False,
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
                        "🔤 Tokens (input)",
                        f"{sess['total_input_tokens']:,}",
                    )
                    tm3.metric(
                        "💰 Coût USD",
                        f"${sess['total_cost_usd']:.4f}",
                    )
                    tm4.metric(
                        "💵 Coût FCFA",
                        f"~{sess['total_cost_fcfa']:.0f}",
                    )

                    # Barre quota gratuit
                    used_pct = min(
                        sess["total_input_tokens"] / 500_000
                        * 100, 100
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
                                    font-size:0.85rem;
                                    font-weight:600">
                            <span>Quota gratuit mensuel</span>
                            <span style="color:#7F8C8D">
                                {sess['total_input_tokens']:,}
                                / 500 000 tokens
                            </span>
                        </div>
                        <div style="background:#E0E0E0;
                                    border-radius:10px;
                                    height:10px;overflow:hidden">
                            <div style="width:{used_pct:.1f}%;
                                        height:100%;
                                        background:{bar_clr};
                                        border-radius:10px">
                            </div>
                        </div>
                        <div style="text-align:right;
                                    font-size:0.78rem;
                                    color:{bar_clr};
                                    margin-top:3px;
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

                    # Reset compteurs
                    if st.button("🔄 Remettre à zéro"):
                        counter.reset()
                        st.success("✅ Compteurs remis à zéro")
                        st.rerun()

                else:
                    st.info(
                        "Aucune extraction Gemini effectuée "
                        "durant cette session."
                    )

                # Conseils économies
                with st.expander(
                    "💡 Conseils pour économiser les tokens"
                ):
                    st.markdown("""
                    | Pages | Tokens estimés | Coût USD |
                    |-------|---------------|----------|
                    | 1 page | ~400 | ~$0.00003 |
                    | 5 pages | ~2 000 | ~$0.00015 |
                    | 12 pages | ~5 000 | ~$0.00038 |
                    | 50 pages | ~20 000 | ~$0.0015 |

                    **🆓 Quota gratuit Gemini Flash :**
                    - 15 requêtes / minute
                    - 1 500 requêtes / jour
                    - Largement suffisant pour usage personnel

                    **💡 Astuces :**
                    - Le mode Hybride économise ~85% vs vision
                    - Traitez un relevé à la fois
                    - Gemini Flash est 10x moins cher que Pro
                    """)


# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    🏦 <b>Bank Statement Extractor</b> v3.0 &nbsp;|&nbsp;
    🇨🇲 Compatible toutes banques du Cameroun &nbsp;|&nbsp;
    ✨ Powered by Google Gemini AI + pdfplumber<br>
    <span style="font-size:0.75rem">
        Financial House · Afriland · SCB · BICEC · UBA ·
        Ecobank · SGBC · BGFI · CCA · Atlantic Bank
    </span>
</div>
""", unsafe_allow_html=True)
