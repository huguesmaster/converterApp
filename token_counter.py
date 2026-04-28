"""
Compteur et gestionnaire de tokens Gemini.
Affiche les métriques d'utilisation dans Streamlit.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os


class TokenCounter:
    """
    Suit et affiche l'utilisation des tokens Gemini
    pour aider l'utilisateur à surveiller sa consommation.
    """

    # Tarifs Gemini 1.5 Flash (USD / 1M tokens)
    PRICE_INPUT_PER_M  = 0.075
    PRICE_OUTPUT_PER_M = 0.30

    # Limite gratuite Gemini Flash
    FREE_TIER_RPM  = 15    # Requêtes par minute
    FREE_TIER_TPM  = 1_000_000  # Tokens par minute
    FREE_TIER_RPD  = 1_500 # Requêtes par jour

    def __init__(self):
        self._init_session()

    def _init_session(self):
        """Initialise les compteurs en session Streamlit."""
        if 'token_stats' not in st.session_state:
            st.session_state.token_stats = {
                'total_input_tokens':  0,
                'total_output_tokens': 0,
                'total_requests':      0,
                'total_pages':         0,
                'sessions': [],
                'last_reset': datetime.now().strftime(
                    "%d/%m/%Y %H:%M"
                ),
            }

    def record_extraction(
        self,
        input_tokens:  int,
        output_tokens: int,
        pages:         int,
        file_name:     str = ""
    ):
        """
        Enregistre les tokens utilisés pour une extraction.
        """
        stats = st.session_state.token_stats
        stats['total_input_tokens']  += input_tokens
        stats['total_output_tokens'] += output_tokens
        stats['total_requests']      += 1
        stats['total_pages']         += pages

        stats['sessions'].append({
            'timestamp':     datetime.now().strftime("%H:%M:%S"),
            'file':          file_name,
            'pages':         pages,
            'input_tokens':  input_tokens,
            'output_tokens': output_tokens,
            'cost_usd':      self._calc_cost(
                input_tokens, output_tokens
            ),
        })

        # Garder seulement les 20 dernières sessions
        stats['sessions'] = stats['sessions'][-20:]

    def estimate_before_extraction(
        self, pages_text: list[dict]
    ) -> dict:
        """
        Estime les tokens AVANT d'appeler Gemini.
        Permet à l'utilisateur de valider avant de consommer.
        """
        total_chars = sum(p['char_count'] for p in pages_text)
        # ~4 chars = 1 token (estimation conservatrice)
        input_est  = total_chars // 4
        # Output : environ 60% de l'input pour du JSON
        output_est = int(input_est * 0.6)

        return {
            'pages':         len(pages_text),
            'input_tokens':  input_est,
            'output_tokens': output_est,
            'total_tokens':  input_est + output_est,
            'cost_usd':      self._calc_cost(input_est, output_est),
            'cost_fcfa':     self._usd_to_fcfa(
                self._calc_cost(input_est, output_est)
            ),
            'within_free_tier': input_est < 500_000,
        }

    def _calc_cost(
        self, input_tokens: int, output_tokens: int
    ) -> float:
        cost = (
            (input_tokens  / 1_000_000) * self.PRICE_INPUT_PER_M +
            (output_tokens / 1_000_000) * self.PRICE_OUTPUT_PER_M
        )
        return round(cost, 6)

    def _usd_to_fcfa(self, usd: float) -> float:
        """Conversion approximative USD → FCFA."""
        return round(usd * 620, 2)

    def get_session_stats(self) -> dict:
        """Retourne les stats de la session courante."""
        stats = st.session_state.token_stats
        total_in  = stats['total_input_tokens']
        total_out = stats['total_output_tokens']
        return {
            **stats,
            'total_cost_usd':  self._calc_cost(total_in, total_out),
            'total_cost_fcfa': self._usd_to_fcfa(
                self._calc_cost(total_in, total_out)
            ),
        }

    def reset(self):
        """Remet les compteurs à zéro."""
        if 'token_stats' in st.session_state:
            del st.session_state['token_stats']
        self._init_session()

    # ──────────────────────────────────────────────
    # AFFICHAGE STREAMLIT
    # ──────────────────────────────────────────────

    def render_estimate_card(self, estimate: dict):
        """
        Affiche la carte d'estimation AVANT extraction.
        Permet à l'utilisateur de confirmer ou annuler.
        """
        tier_color = "#27AE60" if estimate['within_free_tier'] \
                     else "#E74C3C"
        tier_label = "✅ Dans le quota gratuit" \
                     if estimate['within_free_tier'] \
                     else "⚠️ Hors quota gratuit"

        st.markdown(f"""
        <div style="background:white;border-radius:12px;
                    padding:1.2rem;box-shadow:0 2px 12px
                    rgba(0,0,0,0.08);border-left:4px solid
                    {tier_color};margin-bottom:1rem">
            <h4 style="margin:0 0 0.8rem;color:#1F4E79">
                📊 Estimation tokens Gemini
            </h4>
            <div style="display:grid;grid-template-columns:
                        repeat(3,1fr);gap:0.8rem;
                        text-align:center">
                <div>
                    <div style="font-size:1.4rem;font-weight:800;
                                color:#1F4E79">
                        {estimate['pages']}
                    </div>
                    <div style="font-size:0.75rem;color:#7F8C8D">
                        Pages
                    </div>
                </div>
                <div>
                    <div style="font-size:1.4rem;font-weight:800;
                                color:#2E75B6">
                        ~{estimate['total_tokens']:,}
                    </div>
                    <div style="font-size:0.75rem;color:#7F8C8D">
                        Tokens estimés
                    </div>
                </div>
                <div>
                    <div style="font-size:1.4rem;font-weight:800;
                                color:#27AE60">
                        ~${estimate['cost_usd']:.4f}
                    </div>
                    <div style="font-size:0.75rem;color:#7F8C8D">
                        Coût estimé USD
                    </div>
                </div>
            </div>
            <div style="margin-top:0.8rem;padding:0.5rem;
                        background:{tier_color}22;border-radius:6px;
                        text-align:center;font-weight:600;
                        color:{tier_color}">
                {tier_label}
            </div>
        </div>
        """, unsafe_allow_html=True)

    def render_usage_dashboard(self):
        """
        Affiche le tableau de bord d'utilisation complet.
        """
        stats = self.get_session_stats()

        st.markdown("### 📊 Utilisation Gemini — Session courante")

        # Métriques principales
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "📄 Fichiers traités",
            stats['total_requests']
        )
        m2.metric(
            "🔤 Tokens utilisés",
            f"{stats['total_input_tokens']:,}"
        )
        m3.metric(
            "💰 Coût USD",
            f"${stats['total_cost_usd']:.4f}"
        )
        m4.metric(
            "💵 Coût FCFA",
            f"{stats['total_cost_fcfa']:.0f}"
        )

        # Barre de progression quota gratuit
        free_tier_used_pct = min(
            (stats['total_input_tokens'] / 500_000) * 100, 100
        )
        bar_color = (
            "#27AE60" if free_tier_used_pct < 70
            else "#F39C12" if free_tier_used_pct < 90
            else "#E74C3C"
        )

        st.markdown(f"""
        <div style="margin:1rem 0">
            <div style="display:flex;justify-content:space-between;
                        margin-bottom:4px">
                <span style="font-size:0.85rem;font-weight:600">
                    Quota gratuit mensuel
                </span>
                <span style="font-size:0.85rem;color:#7F8C8D">
                    {stats['total_input_tokens']:,} /
                    500 000 tokens
                </span>
            </div>
            <div style="background:#E0E0E0;border-radius:10px;
                        height:10px;overflow:hidden">
                <div style="width:{free_tier_used_pct:.1f}%;
                            height:100%;background:{bar_color};
                            border-radius:10px;
                            transition:width 0.5s ease">
                </div>
            </div>
            <div style="text-align:right;font-size:0.78rem;
                        color:{bar_color};margin-top:3px;
                        font-weight:600">
                {free_tier_used_pct:.1f}% utilisé
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tableau des sessions
        if stats['sessions']:
            st.markdown("**Historique de la session :**")
            df_sessions = pd.DataFrame(stats['sessions'])
            df_sessions.columns = [
                'Heure', 'Fichier', 'Pages',
                'Tokens Input', 'Tokens Output', 'Coût USD'
            ]
            df_sessions['Coût USD'] = df_sessions[
                'Coût USD'
            ].apply(lambda x: f"${x:.4f}")
            st.dataframe(
                df_sessions,
                use_container_width=True,
                hide_index=True
            )

        # Bouton reset
        if st.button("🔄 Remettre les compteurs à zéro"):
            self.reset()
            st.success("✅ Compteurs remis à zéro")
            st.rerun()

    def render_tips(self):
        """Affiche des conseils pour économiser les tokens."""
        with st.expander("💡 Conseils pour économiser les tokens"):
            st.markdown("""
            **🟢 Bonnes pratiques :**
            - Utilisez **pdfplumber seul** si votre PDF est
              un texte natif bien structuré
            - Traitez un **PDF par mois** plutôt que
              tous les relevés en même temps
            - Le mode **Gemini Flash** est 10x moins cher
              que Gemini Pro

            **📊 Consommation typique :**
            | Pages | Tokens estimés | Coût USD |
            |-------|---------------|----------|
            | 1 | ~500 | $0.0001 |
            | 5 | ~2 500 | $0.0002 |
            | 12 | ~6 000 | $0.0005 |
            | 50 | ~25 000 | $0.002 |

            **🆓 Quota gratuit Gemini Flash :**
            - 15 requêtes / minute
            - 1 500 requêtes / jour
            - Largement suffisant pour usage personnel
            """)
