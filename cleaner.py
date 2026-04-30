"""
cleaner.py - Version 4.4
Approche prudente : minimiser la perte de lignes et maximiser la fidélité
"""

import pandas as pd
import numpy as np
import re
from typing import Optional

class DataCleaner:
    def clean(self, df: pd.DataFrame, banque_nom: str = "Autre banque") -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()

        df = self._clean_dates(df)
        df = self._clean_amounts(df)
        df = self._merge_multi_line_libelles_conservative(df)   # Version prudente
        df = self._clean_libelle(df)
        df = self._remove_duplicates_minimal(df)                # Très léger
        df = self._sort_by_date(df)
        df = self._post_process_by_bank(df, banque_nom)

        return df.reset_index(drop=True)

    # ====================== DATES ======================
    def _clean_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ['Date', 'Date_Valeur']:
            if col in df.columns:
                df[col] = df[col].apply(self._normalize_date)
        return df

    def _normalize_date(self, val) -> str:
        if not val or pd.isna(val):
            return ''
        s = str(val).strip()
        if re.match(r'\d{2}/\d{2}/\d{4}', s):
            return s[:10]
        return s

    # ====================== MONTANTS ======================
    def _clean_amounts(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ['Débit', 'Crédit', 'Solde']:
            if col in df.columns:
                df[col] = df[col].apply(self._parse_amount)
        return df

    def _parse_amount(self, val) -> Optional[float]:
        if val is None or pd.isna(val):
            return None
        s = str(val).strip()
        if s.lower() in ('null', 'none', '', '0'):
            return None
        try:
            s = re.sub(r'[^\d.,-]', '', s)
            s = s.replace(',', '.')
            if s.count('.') > 1:
                s = s.replace('.', '')
            return float(s) if s else None
        except:
            return None

    # ====================== FUSION LIBELLÉS (VERSION CONSERVATRICE) ======================
    def _merge_multi_line_libelles_conservative(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fusion très prudente : seulement si la ligne suivante est clairement une continuation"""
        if 'Libellé' not in df.columns or df.empty:
            return df

        df = df.reset_index(drop=True)
        result = []
        i = 0
        while i < len(df):
            current_lib = str(df.iloc[i].get('Libellé', '')).strip()
            current_row = df.iloc[i].copy()

            # Vérifier si les lignes suivantes sont des continuations (pas de date + pas de montant)
            j = i + 1
            while j < len(df):
                next_row = df.iloc[j]
                next_lib = str(next_row.get('Libellé', '')).strip()
                has_date = bool(str(next_row.get('Date', '')).strip())
                has_amount = (pd.notna(next_row.get('Débit')) and next_row.get('Débit') != 0) or \
                             (pd.notna(next_row.get('Crédit')) and next_row.get('Crédit') != 0)

                if not has_date and not has_amount and next_lib:
                    current_lib = f"{current_lib} {next_lib}".strip()
                    j += 1
                else:
                    break

            current_row['Libellé'] = current_lib
            result.append(current_row)
            i = j

        return pd.DataFrame(result)

    def _clean_libelle(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'Libellé' in df.columns:
            df['Libellé'] = (
                df['Libellé']
                .astype(str)
                .str.replace(r'\s+', ' ', regex=True)
                .str.strip()
                .str.replace(r'(?i)remettant\s*:?\s*', '', regex=True)
            )
        return df

    # ====================== DÉDUPLICATION MINIMALE ======================
    def _remove_duplicates_minimal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ne supprime que les lignes complètement identiques"""
        if df.empty:
            return df

        # On considère une ligne comme doublon seulement si TOUT est identique
        df = df.drop_duplicates(keep='first')
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

    def _post_process_by_bank(self, df: pd.DataFrame, banque_nom: str) -> pd.DataFrame:
        if banque_nom == "UNICS":
            if 'Particulars' in df.columns:
                df = df.rename(columns={'Particulars': 'Libellé'})
        elif banque_nom in ["CEPAC", "ADVANS"]:
            for old_col in ['Désignation', "Libellé de l'opération"]:
                if old_col in df.columns:
                    df = df.rename(columns={old_col: 'Libellé'})
                    break
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

        lib_lower = df.get('Libellé', pd.Series('')).astype(str).str.lower()

        mask_ouv = lib_lower.str.contains('ouverture|opening|report solde antérieur', na=False)
        mask_clo = lib_lower.str.contains('cl[ôo]ture|cloture|solde final|solde crediteur|total mouvements', na=False)

        normal_df = df[~(mask_ouv | mask_clo)]

        stats['total_transactions'] = len(normal_df)
        stats['total_credit'] = float(normal_df.get('Crédit', pd.Series(0)).sum(skipna=True) or 0)
        stats['total_debit'] = float(normal_df.get('Débit', pd.Series(0)).sum(skipna=True) or 0)
        stats['net'] = stats['total_credit'] - stats['total_debit']

        if mask_ouv.any():
            val = df.loc[mask_ouv, 'Solde'].dropna()
            if not val.empty:
                stats['solde_ouverture'] = float(val.iloc[0])

        if mask_clo.any():
            val = df.loc[mask_clo, 'Solde'].dropna()
            if not val.empty:
                stats['solde_cloture'] = float(val.iloc[-1])

        dates = pd.to_datetime(df.get('Date'), format='%d/%m/%Y', errors='coerce').dropna()
        if not dates.empty:
            stats['periode_debut'] = dates.min().strftime('%d/%m/%Y')
            stats['periode_fin'] = dates.max().strftime('%d/%m/%Y')

        return stats
