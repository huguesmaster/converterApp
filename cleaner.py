# cleaner.py
import pandas as pd
import numpy as np
import re
from typing import List

class DataCleaner:
    """
    Nettoyage avancé des relevés bancaires camerounais.
    Gère la fusion des libellés multi-lignes et les spécificités par banque.
    """

    def clean(self, df: pd.DataFrame, banque_nom: str = "Autre banque") -> pd.DataFrame:
        """Pipeline complet avec support banque-specific."""
        if df.empty:
            return df

        df = df.copy()
        df = self._clean_dates(df)
        df = self._clean_amounts(df)
        df = self._merge_multi_line_libelles(df)      # Nouvelle fonction clé
        df = self._clean_libelle(df)
        df = self._remove_duplicates(df)
        df = self._sort_by_date(df)
        df = self._post_process_by_bank(df, banque_nom)
        return df

    # ====================== DATES ======================
    def _clean_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ['Date', 'Date_Valeur', 'Valeur']:
            if col in df.columns:
                df[col] = df[col].apply(self._normalize_date)
        return df

    def _normalize_date(self, val) -> str:
        if not val or pd.isna(val):
            return ''
        s = str(val).strip()
        # Format JJ/MM/AAAA
        if re.match(r'\d{2}/\d{2}/\d{4}', s):
            return s[:10]
        # Autres formats possibles (ex: JJ-MMM-AAAA)
        return s

    # ====================== MONTANTS ======================
    def _clean_amounts(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ['Débit', 'Crédit', 'Solde', 'Debit', 'Credit', 'Balance']:
            if col in df.columns:
                # Nettoyage agressif des montants
                df[col] = df[col].astype(str).str.replace(r'[^\d.,-]', '', regex=True)
                df[col] = pd.to_numeric(df[col].str.replace(',', '.'), errors='coerce')
                df[col] = df[col].replace(0, np.nan)
        return df

    # ====================== FUSION LIBELLÉS MULTI-LIGNES ======================
    def _merge_multi_line_libelles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fusionne les lignes où le libellé continue sans date ni montant."""
        if 'Libellé' not in df.columns or df.empty:
            return df

        df = df.reset_index(drop=True)
        merged_libelles = []
        current_libelle = ""

        for i, row in df.iterrows():
            lib = str(row.get('Libellé', '')).strip()
            debit = pd.notna(row.get('Débit')) or pd.notna(row.get('Debit'))
            credit = pd.notna(row.get('Crédit')) or pd.notna(row.get('Credit'))
            has_date = bool(str(row.get('Date', '')).strip())

            if not has_date and not debit and not credit and lib and not lib.startswith(('SOLDE', 'Report', 'TOTAL')):
                # Ligne de continuation
                current_libelle += " " + lib if current_libelle else lib
            else:
                if current_libelle:
                    merged_libelles[-1] = current_libelle  # Mise à jour de la ligne précédente
                merged_libelles.append(lib)
                current_libelle = ""

        # Dernière ligne
        if current_libelle and merged_libelles:
            merged_libelles[-1] = current_libelle

        df['Libellé'] = merged_libelles
        return df

    def _clean_libelle(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'Libellé' in df.columns:
            df['Libellé'] = (
                df['Libellé']
                .astype(str)
                .str.replace(r'\s+', ' ', regex=True)
                .str.strip()
                .str.replace(r'^None$', '', regex=True)
                .str.replace(r'(?i)remettant\s*:?', '', regex=True)  # Nettoyage MUPECI/ADVANS
            )
        return df

    # ====================== DÉDUPLICATION & TRI ======================
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        subset = ['Date', 'Référence', 'Libellé']
        subset = [c for c in subset if c in df.columns]
        if subset:
            df = df.drop_duplicates(subset=subset, keep='first')
        return df

    def _sort_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'Date' not in df.columns:
            return df
        try:
            df['_date_sort'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
            df = df.sort_values('_date_sort', na_position='first')
            df = df.drop(columns=['_date_sort'])
        except:
            pass
        return df.reset_index(drop=True)

    # ====================== POST-PROCESSING PAR BANQUE ======================
    def _post_process_by_bank(self, df: pd.DataFrame, banque_nom: str) -> pd.DataFrame:
        """Règles spécifiques par banque."""
        if banque_nom == "UNICS":
            # Exemple : renommer colonnes si besoin
            if 'Particulars' in df.columns and 'Libellé' not in df.columns:
                df = df.rename(columns={'Particulars': 'Libellé'})
        elif banque_nom in ["CEPAC", "ADVANS"]:
            if 'Désignation' in df.columns:
                df = df.rename(columns={'Désignation': 'Libellé'})
            elif 'Libellé de l\'opération' in df.columns:
                df = df.rename(columns={'Libellé de l\'opération': 'Libellé'})
        elif banque_nom == "MUPECI":
            # Nettoyage spécifique des "Remettant :"
            if 'Libellé' in df.columns:
                df['Libellé'] = df['Libellé'].str.replace(r'Remett(?:ant)?\s*:?\s*', '', regex=True, flags=re.IGNORECASE)
        return df

    # ====================== STATISTIQUES ======================
    def get_statistics(self, df: pd.DataFrame) -> dict:
        stats = {
            'total_transactions': 0,
            'total_credit': 0.0,
            'total_debit': 0.0,
            'net': 0.0,
            'solde_ouverture': None,
            'solde_cloture': None,
            'periode_debut': '',
            'periode_fin': '',
        }

        if df.empty:
            return stats

        # Détection soldes (plus robuste)
        lib_lower = df['Libellé'].astype(str).str.lower()
        mask_ouv = lib_lower.str.contains('ouverture|opening|report solde antérieur|solde debut', na=False)
        mask_clo = lib_lower.str.contains('cl[ôo]ture|cloture|solde final|solde crediteur|total mouvements', na=False)

        normal_df = df[~(mask_ouv | mask_clo)]

        stats['total_transactions'] = len(normal_df)
        stats['total_credit'] = float(normal_df.get('Crédit', pd.Series(0)).sum(skipna=True) or 0)
        stats['total_debit']  = float(normal_df.get('Débit',  pd.Series(0)).sum(skipna=True) or 0)
        stats['net'] = stats['total_credit'] - stats['total_debit']

        # Soldes
        if mask_ouv.any():
            stats['solde_ouverture'] = float(df.loc[mask_ouv, 'Solde'].dropna().iloc[0]) if not df.loc[mask_ouv, 'Solde'].dropna().empty else None
        if mask_clo.any():
            stats['solde_cloture'] = float(df.loc[mask_clo, 'Solde'].dropna().iloc[-1]) if not df.loc[mask_clo, 'Solde'].dropna().empty else None

        # Période
        dates = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce').dropna()
        if not dates.empty:
            stats['periode_debut'] = dates.min().strftime('%d/%m/%Y')
            stats['periode_fin']   = dates.max().strftime('%d/%m/%Y')

        return stats
