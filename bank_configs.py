"""
bank_configs.py - Configurations spécifiques par banque
Version 4.2 - Optimisé pour UNICS et autres banques camerounaises
"""

from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class BankConfig:
    nom: str
    code: str
    emoji: str

    col_date: List[str] = field(default_factory=lambda: ["Date", "Date Opération"])
    col_ref: List[str] = field(default_factory=lambda: ["Référence", "Ref", "Cheq#", "N° Pièce", "Batch/Ref"])
    col_libelle: List[str] = field(default_factory=lambda: ["Libellé", "Désignation", "Particulars", "Libellé de l'opération"])
    col_date_valeur: List[str] = field(default_factory=lambda: ["Date Valeur", "Valeur", "ValueDate"])
    col_debit: List[str] = field(default_factory=lambda: ["Débit", "Debit", "Debits"])
    col_credit: List[str] = field(default_factory=lambda: ["Crédit", "Credit", "Credits"])
    col_solde: List[str] = field(default_factory=lambda: ["Solde", "Balance"])

    solde_ouverture_patterns: List[str] = field(default_factory=lambda: [
        r"ouverture", r"opening balance", r"report solde", r"solde antérieur", r"solde debut"
    ])
    solde_cloture_patterns: List[str] = field(default_factory=lambda: [
        r"cl[ôo]ture", r"cloture", r"solde final", r"solde crediteur", r"total mouvements"
    ])

    specific_instructions: str = ""
    date_format_hint: str = "JJ/MM/AAAA"


BANK_CONFIGS: Dict[str, BankConfig] = {
    "Financial House S.A": BankConfig(
        nom="Financial House S.A",
        code="FH",
        emoji="🏛️",
        col_ref=["Batch/Ref", "Référence"],
        specific_instructions="""Financial House : Les libellés peuvent être sur plusieurs lignes. Fusionne-les correctement."""
    ),

    "BGFI Bank": BankConfig(
        nom="BGFI Bank",
        code="BGFI",
        emoji="🔷",
        col_ref=["N° Pièce", "Référence"],
        specific_instructions="""BGFI : Colonne 'N° Pièce' fréquente. Frais toujours en Débit."""
    ),

           "UNICS": BankConfig(
        nom="UNICS",
        code="UNICS",
        emoji="🏦",
        col_ref=["Cheq#", "Référence"],
        col_libelle=["Particulars"],
        specific_instructions="""UNICS : Tu dois extraire **ABSOLUMENT TOUTES** les lignes du tableau, sans exception.

Particulièrement important :
- La ligne du 03/03/2025 avec débit 308000 : "WDL chq no. 483387 fv NDEUMA GISCARD DESTAING 01032025"
- Toutes les lignes commençant par "WDL chq no."
- Toutes les lignes de "CASH CREDIT BY"

Ne saute aucune ligne contenant un montant en Débit ou Crédit. 
Retourne les montants sans virgule ni espace (ex: 308000 au lieu de 308,000). 
Sois extrêmement exhaustif sur toutes les  pages du document."""
    ),

    "CEPAC": BankConfig(
        nom="CEPAC",
        code="CEPAC",
        emoji="🏦",
        col_ref=["Référence", "VE N°", "CHQ N°"],
        col_libelle=["Désignation"],
        specific_instructions="""CEPAC : Libellé principal dans la colonne 'Désignation'. Références VE N° ou CHQ N°. Solde de clôture souvent 'SOLDE CRÉDITEUR'."""
    ),

    "ADVANS": BankConfig(
        nom="ADVANS",
        code="ADVANS",
        emoji="🏦",
        col_libelle=["Libellé de l'opération"],
        specific_instructions="""ADVANS : Libellés très détaillés. Fusion multi-lignes critique. Beaucoup de retraits chèques."""
    ),

    "MUPECI": BankConfig(
        nom="MUPECI",
        code="MUPECI",
        emoji="🏦",
        col_libelle=["Libelle et Référence"],
        specific_instructions="""MUPECI : Libellés longs avec 'Remettant :', 'VERST.', 'VAD', 'RETRAIT'. Nettoie les mentions 'Remettant :'."""
    ),

    "SCB Cameroun": BankConfig(
        nom="SCB Cameroun",
        code="SCB",
        emoji="🏦",
        specific_instructions="SCB : Structure généralement propre."
    ),

    "BICEC": BankConfig(
        nom="BICEC",
        code="BICEC",
        emoji="🏦",
    ),

    "UBA Cameroun": BankConfig(
        nom="UBA Cameroun",
        code="UBA",
        emoji="🌐",
    ),

    "Autre banque": BankConfig(
        nom="Autre banque",
        code="AUTRE",
        emoji="🏦",
        specific_instructions="Utilise la structure standard des relevés camerounais. Sois vigilant sur la fusion des libellés et la distinction Débit/Crédit."
    ),
}


def get_bank_config(nom_banque: str) -> BankConfig:
    """Retourne la config de la banque ou la config par défaut."""
    for key, cfg in BANK_CONFIGS.items():
        if nom_banque.lower() in key.lower() or cfg.nom.lower() in nom_banque.lower():
            return cfg
    return BANK_CONFIGS["Autre banque"]
