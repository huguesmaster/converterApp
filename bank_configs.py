"""
╔══════════════════════════════════════════════════════════════╗
║                BANK CONFIGURATIONS — Cameroun                ║
║         Configurations spécifiques par banque                ║
║         Version : 4.0.0                                      ║
╚══════════════════════════════════════════════════════════════╝
"""

from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class BankConfig:
    nom: str
    code: str
    emoji: str

    # Noms de colonnes possibles (variations fréquentes)
    col_date: List[str] = field(default_factory=lambda: ["Date", "Date d'op", "Date Opération"])
    col_ref: List[str] = field(default_factory=lambda: ["Référence", "Ref", "Cheq#", "N° Pièce", "Batch/Ref", "Réf"])
    col_libelle: List[str] = field(default_factory=lambda: ["Libellé", "Désignation", "Libellé de l'opération", "Particulars", "Libelle et Référence"])
    col_date_valeur: List[str] = field(default_factory=lambda: ["Date Valeur", "Valeur", "ValueDate", "V.D."])
    col_debit: List[str] = field(default_factory=lambda: ["Débit", "Debits", "Debit", "Débts"])
    col_credit: List[str] = field(default_factory=lambda: ["Crédit", "Credits", "Credit", "Crédits"])
    col_solde: List[str] = field(default_factory=lambda: ["Solde", "Balance", "Solde (XAF)"])

    # Patterns pour détecter les lignes de solde (ouverture / clôture)
    solde_ouverture_patterns: List[str] = field(default_factory=lambda: [
        r"ouverture", r"opening balance", r"report solde", r"solde antérieur", r"solde debut", r"report du"
    ])
    solde_cloture_patterns: List[str] = field(default_factory=lambda: [
        r"cl[ôo]ture", r"cloture", r"solde final", r"solde crediteur", r"solde clôture", r"total mouvements", r"solde au"
    ])

    # Instructions spécifiques pour le prompt Gemini (très important)
    specific_instructions: str = ""

    date_format_hint: str = "JJ/MM/AAAA"


# ══════════════════════════════════════════════════════════════
# CONFIGURATIONS PAR BANQUE
# ══════════════════════════════════════════════════════════════
BANK_CONFIGS: Dict[str, BankConfig] = {
    "Financial House S.A": BankConfig(
        nom="Financial House S.A",
        code="FH",
        emoji="🏛️",
        col_ref=["Batch/Ref", "Référence", "Cheq#"],
        specific_instructions="""Financial House utilise souvent des références de type Batch. Les libellés peuvent être sur 2 ou 3 lignes. Fusionne les descriptions contiguës sans montant. Attention à la colonne "Particulars" qui contient souvent le libellé."""
    ),

    "BGFI Bank": BankConfig(
        nom="BGFI Bank",
        code="BGFI",
        emoji="🔷",
        col_ref=["N° Pièce", "Référence", "Batch"],
        specific_instructions="""BGFI : colonne 'N° Pièce' très courante. Les frais bancaires (commissions, SMS, etc.) apparaissent toujours en Débit. Libellés parfois sur plusieurs lignes. Structure relativement classique."""
    ),

    "UNICS": BankConfig(
        nom="UNICS",
        code="UNICS",
        emoji="🏦",
        col_ref=["Cheq#", "Référence"],
        col_libelle=["Particulars"],
        specific_instructions="""UNICS : Utilise la colonne 'Cheq#' et 'Particulars' pour le libellé. Les libellés sont souvent sur plusieurs lignes. Le solde d'ouverture est indiqué comme "Opening Balance". Fusionne soigneusement les lignes de description."""
    ),

    "CEPAC": BankConfig(
        nom="CEPAC",
        code="CEPAC",
        emoji="🏦",
        col_ref=["Référence", "VE N°", "CHQ N°"],
        col_libelle=["Désignation"],
        specific_instructions="""CEPAC : La colonne principale du libellé est 'Désignation'. Références fréquentes : VE N°, CHQ N°. Le solde de clôture est souvent indiqué comme "SOLDE CRÉDITEUR" ou "TOTAL MOUVEMENTS". Libellés peuvent être longs."""
    ),

    "ADVANS": BankConfig(
        nom="ADVANS",
        code="ADVANS",
        emoji="🏦",
        col_libelle=["Libellé de l'opération"],
        specific_instructions="""ADVANS : Libellé très détaillé dans la colonne 'Libellé de l'opération'. Beaucoup de retraits chèques avec motif et nom du client. Fusion des libellés multi-lignes est critique ici. Solde souvent indiqué comme "Report du..." ou "Solde au..."."""
    ),

    "MUPECI": BankConfig(
        nom="MUPECI",
        code="MUPECI",
        emoji="🏦",
        col_libelle=["Libelle et Référence"],
        specific_instructions="""MUPECI : Libellés très longs incluant souvent "Remettant :", "VERST.", "VAD", "RETRAIT", "ChequeID". Nettoie les mentions "Remettant :". Beaucoup de virements et versements espèces. Fusion multi-lignes très importante."""
    ),

    "SCB Cameroun": BankConfig(
        nom="SCB Cameroun",
        code="SCB",
        emoji="🏦",
        specific_instructions="""SCB : Structure généralement propre et proche des standards internationaux. Colonnes bien alignées. Bonne détection des colonnes Débit/Crédit/Solde."""
    ),

    "BICEC": BankConfig(
        nom="BICEC",
        code="BICEC",
        emoji="🏦",
        specific_instructions="""BICEC : Libellés parfois longs. Attention aux frais et commissions qui apparaissent en Débit."""
    ),

    "UBA Cameroun": BankConfig(
        nom="UBA Cameroun",
        code="UBA",
        emoji="🌐",
        specific_instructions="""UBA : Format souvent clair avec bonnes séparations. Références internationales possibles."""
    ),

    "Ecobank Cameroun": BankConfig(
        nom="Ecobank Cameroun",
        code="ECO",
        emoji="🌿",
        specific_instructions="""Ecobank : Colonnes généralement bien structurées. Codes de transaction fréquents dans le libellé."""
    ),

    "SGBC": BankConfig(
        nom="SGBC",
        code="SGBC",
        emoji="🔴",
        specific_instructions="""SGBC (Société Générale) : Structure proche des standards bancaires internationaux."""
    ),

    "CCA Bank": BankConfig(
        nom="CCA Bank",
        code="CCA",
        emoji="🏦",
    ),

    "Atlantic Bank": BankConfig(
        nom="Atlantic Bank",
        code="ATL",
        emoji="🌊",
    ),

    "Autre banque": BankConfig(
        nom="Autre banque",
        code="AUTRE",
        emoji="🏦",
        specific_instructions="""Utilise la structure standard des relevés bancaires camerounais. Sois particulièrement vigilant sur la distinction Débit / Crédit, la fusion des libellés multi-lignes et la détection des soldes d'ouverture et de clôture."""
    ),
}


def get_bank_config(nom_banque: str) -> BankConfig:
    """Retourne la configuration correspondant à la banque ou la config par défaut."""
    config = BANK_CONFIGS.get(nom_banque)
    if config is None:
        # Recherche approximative par mot-clé
        for key, cfg in BANK_CONFIGS.items():
            if nom_banque.lower() in key.lower() or cfg.nom.lower() in nom_banque.lower():
                return cfg
        return BANK_CONFIGS["Autre banque"]
    return config


# Pour debug / affichage
def list_supported_banks() -> List[str]:
    return list(BANK_CONFIGS.keys())
