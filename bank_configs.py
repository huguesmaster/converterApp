# bank_configs.py
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class BankConfig:
    nom: str
    code: str
    emoji: str

    # Noms/variations de colonnes
    col_date: List[str] = field(default_factory=lambda: ["Date", "Date Opération"])
    col_ref: List[str] = field(default_factory=lambda: ["Référence", "Batch/Ref", "N° Pièce", "Cheq#", "Ref"])
    col_libelle: List[str] = field(default_factory=lambda: ["Libellé", "Description", "Libellé Opération"])
    col_date_valeur: List[str] = field(default_factory=lambda: ["D.Valeur", "Date Valeur", "Valeur"])
    col_debit: List[str] = field(default_factory=lambda: ["Débit", "Debit"])
    col_credit: List[str] = field(default_factory=lambda: ["Crédit", "Credit"])
    col_solde: List[str] = field(default_factory=lambda: ["Solde", "Balance"])

    # Patterns pour détecter les soldes
    solde_ouverture_patterns: List[str] = field(default_factory=lambda: [
        r"solde\s+d['’]ouverture", r"solde\s+initial", r"opening balance", r"solde debut"
    ])
    solde_cloture_patterns: List[str] = field(default_factory=lambda: [
        r"solde\s+de\s+cl[ôo]ture", r"solde\s+final", r"closing balance", r"solde cloture", r"solde fin"
    ])

    # Instructions spécifiques pour le prompt Gemini
    specific_instructions: str = ""

    date_format_hint: str = "JJ/MM/AAAA"


BANK_CONFIGS: Dict[str, BankConfig] = {
    "Financial House S.A": BankConfig(
        nom="Financial House S.A",
        code="FH",
        emoji="🏛️",
        col_ref=["Batch/Ref", "Référence", "Cheq#"],
        specific_instructions="""Financial House utilise souvent des références Batch. Les libellés sont fréquemment sur plusieurs lignes consécutives. Fusionne les lignes de description qui ne contiennent pas de montant."""
    ),
    "BGFI Bank": BankConfig(
        nom="BGFI Bank",
        code="BGFI",
        emoji="🔷",
        col_ref=["N° Pièce", "Référence", "Batch"],
        specific_instructions="""BGFI : colonne 'N° Pièce' très courante. Les frais (TVA, commissions, SMS Pack) apparaissent toujours en Débit. Attention aux libellés sur 3-4 lignes."""
    ),
    "SCB Cameroun": BankConfig(
        nom="SCB Cameroun",
        code="SCB",
        emoji="🏦",
        specific_instructions="SCB a généralement une bonne structure tabulaire. Priorise la détection stricte des colonnes Débit / Crédit / Solde."
    ),
    "BICEC": BankConfig(
        nom="BICEC",
        code="BICEC",
        emoji="🏦",
        specific_instructions="BICEC : libellés parfois longs et multi-lignes. Vérifier les soldes d'ouverture et de clôture avec précision."
    ),
    "UBA Cameroun": BankConfig(
        nom="UBA Cameroun",
        code="UBA",
        emoji="🌐",
        specific_instructions="UBA : format souvent clair. Attention aux références internationales et virements."
    ),
    "Ecobank Cameroun": BankConfig(
        nom="Ecobank Cameroun",
        code="ECO",
        emoji="🌿",
        specific_instructions="Ecobank : colonnes généralement bien alignées. Libellés peuvent inclure des codes de transaction."
    ),
    "SGBC": BankConfig(
        nom="SGBC",
        code="SGBC",
        emoji="🔴",
        specific_instructions="SGBC (Société Générale) : structure proche des standards internationaux."
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
        specific_instructions="Utilise la structure standard des relevés bancaires camerounais. Sois particulièrement vigilant sur la distinction Débit / Crédit et la fusion des libellés multi-lignes."
    ),
}


def get_bank_config(nom_banque: str) -> BankConfig:
    """Retourne la configuration de la banque ou la config par défaut."""
    return BANK_CONFIGS.get(nom_banque, BANK_CONFIGS["Autre banque"])
