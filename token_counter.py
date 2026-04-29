"""
Compteur et gestionnaire de tokens Gemini.
Affiche les métriques d'utilisation dans Streamlit.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import io

class TokenCounter:
    """
    Suit et affiche l'utilisation des tokens Gemini
    pour aider l'utilisateur à surveiller sa consommation.
    """

    # Tarifs Gemini 1.5 Flash (USD / 1M tokens)
    PRICE_INPUT_PER_M  = 0.075
    PRICE_OUTPUT_PER_M = 0.30

    def __init__(self):
        self._init_session()

    def _init_session(self):
        """Initialise les compteurs dans la session Streamlit."""
        if 'token_stats' not in st.session_state:
            st.session_state.token_stats = {
                'total_input_tokens':  0,
                'total_output_tokens': 0,
                'total_requests':      0,
                'total_pages':         0,
                'sessions': [],
                'last_reset': datetime.now().strftime("%d/%m/%Y %H:%M"),
            }

    def record_extraction(self, input_tokens: int, output_tokens: int, pages: int = 0):
        """Enregistre une nouvelle transaction et calcule son coût."""
        cost = (
            (input_tokens / 1_000_000) * self.PRICE_INPUT_PER_M +
            (output_tokens / 1_000_000) * self.PRICE_OUTPUT_PER_M
        )
        
        # Mise à jour des globaux
        st.session_state.token_stats['total_input_tokens'] += input_tokens
        st.session_state.token_stats['total_output_tokens'] += output_tokens
        st.session_state.token_stats['total_requests'] += 1
        st.session_state.token_stats['total_pages'] += pages
        
        # Ajout à l'historique de session
        new_session = {
            'Heure': datetime.now().strftime("%H:%M:%S"),
            'Pages': pages,
            'Tokens Input': input_tokens,
            'Tokens Output': output_tokens,
            'Coût USD': cost
        }
        st.session_state.token_stats['sessions'].insert(0, new_session)

    def render_dashboard(self):
        """Affiche le tableau de bord des statistiques dans Streamlit."""
        stats = st.session_state.token_stats
        
        st.subheader("📊 Suivi de consommation (Gemini 1.5 Flash)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Requêtes", stats['total_requests'])
        with col2:
            total_tokens = stats['total_input_tokens'] + stats['total_output_tokens']
            st.metric("Total Tokens", f"{total_tokens:,}")
        with col3:
            # Calcul du coût total
            total_cost = (
                (stats['total_input_tokens'] / 1_000_000) * self.PRICE_INPUT_PER_M +
                (stats['total_output_tokens'] / 1_000_000) * self.PRICE_OUTPUT_PER_M
            )
            st.metric("Coût estimé", f"${total_cost:.4f}")

        if stats['sessions']:
            with st.expander("Détail des dernières extractions"):
                df_sessions = pd.DataFrame(stats['sessions'])
                # Formatage pour l'affichage
                df_display = df_sessions.copy()
                df_display['Coût USD'] = df_display['Coût USD'].apply(lambda x: f"${x:.6f}")
                st.table(df_display)

        if st.button("🔄 Réinitialiser les compteurs"):
            st.session_state.token_stats = {
                'total_input_tokens': 0, 'total_output_tokens': 0,
                'total_requests': 0, 'total_pages': 0,
                'sessions': [], 'last_reset': datetime.now().strftime("%d/%m/%Y %H:%M")
            }
            st.rerun()

    def render_tips(self):
        """Affiche des conseils pour économiser les tokens."""
        with st.expander("💡 Conseils pour économiser les tokens"):
            st.markdown("""
            **🟢 Bonnes pratiques :**
            - Utilisez **pdfplumber** pour les PDF natifs (non-scannés).
            - Traitez les documents par **petits lots** (max 10 pages).
            - Préférez **Gemini 1.5 Flash** pour l'extraction de données massive.

            **📊 Estimation des coûts :**
            - 1 page scannée : ~500 à 1 000 tokens.
            - Coût pour 1 000 pages : ~0.07$ (Extrêmement compétitif).
            """)
