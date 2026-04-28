import pandas as pd
import numpy as np
import re


class DataCleaner:
    """
    Nettoyage et enrichissement du DataFrame extrait.
    """

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline de nettoyage complet."""
        if df.empty:
            return df

        df = df.copy()
        df = self._clean_dates(df)
        df = self._clean_amounts(df)
        df = self._clean_libelle(df)
        df = self._remove_duplicates(df)
        df = self._sort_by_date(df)
        return df

    def _clean_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ['Date', 'Date_Valeur']:
            if col in df.columns:
                df[col] = df[col].apply(self._normalize_date)
        return df

    def _normalize_date(self, val) -> str:
        if not val or pd.isna(val):
            return ''
        s = str(val).strip()
        # Déjà au bon format
        if re.match(r'\d{2}/\d{2}/\d{4}', s):
            return s[:10]
        return s

    def _clean_amounts(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ['Débit', 'Crédit', 'Solde']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].replace(0, np.nan)
        return df

    def _clean_libelle(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'Libellé' in df.columns:
            df['Libellé'] = (
                df['Libellé']
                .astype(str)
                .str.replace(r'\s+', ' ', regex=True)
                .str.strip()
                .str.replace('None', '', regex=False)
            )
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Supprime les lignes identiques."""
        subset = ['Date', 'Référence', 'Libellé']
        subset = [c for c in subset if c in df.columns]
        if subset:
            df = df.drop_duplicates(subset=subset, keep='first')
        return df

    def _sort_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trie par date si possible."""
        if 'Date' not in df.columns:
            return df
        try:
            df['_date_sort'] = pd.to_datetime(
                df['Date'], format='%d/%m/%Y', errors='coerce'
            )
            # Garder les lignes sans date (solde ouverture/clôture) en place
            df = df.sort_values('_date_sort', na_position='first')
            df = df.drop(columns=['_date_sort'])
        except Exception:
            pass
        return df.reset_index(drop=True)

    def get_statistics(self, df: pd.DataFrame) -> dict:
        """Calcule les statistiques du relevé."""
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

        # Lignes de solde
        mask_ouv = df['Libellé'].str.contains(
            'ouverture', case=False, na=False
        )
        mask_clo = df['Libellé'].str.contains(
            r'cl[oô]ture', case=False, na=False
        )

        # Transactions normales
        mask_normal = ~(mask_ouv | mask_clo)
        normal_df = df[mask_normal]

        stats['total_transactions'] = len(normal_df)
        stats['total_credit'] = float(
            normal_df['Crédit'].sum(skipna=True) or 0
        )
        stats['total_debit'] = float(
            normal_df['Débit'].sum(skipna=True) or 0
        )
        stats['net'] = stats['total_credit'] - stats['total_debit']

        # Soldes
        if mask_ouv.any():
            val = df.loc[mask_ouv, 'Solde'].dropna()
            if not val.empty:
                stats['solde_ouverture'] = float(val.iloc[0])

        if mask_clo.any():
            val = df.loc[mask_clo, 'Solde'].dropna()
            if not val.empty:
                stats['solde_cloture'] = float(val.iloc[-1])

        # Période
        dates = pd.to_datetime(
            df['Date'], format='%d/%m/%Y', errors='coerce'
        ).dropna()
        if not dates.empty:
            stats['periode_debut'] = dates.min().strftime('%d/%m/%Y')
            stats['periode_fin'] = dates.max().strftime('%d/%m/%Y')

        return stats
