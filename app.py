"""
╔════════════════════════════════════════════════════════════════════╗
║          BANK STATEMENT OCR EXTRACTOR - STREAMLIT APP              ║
║  Extraction de relevés bancaires camerounais (UNICS, ADVANS, etc)  ║
╚════════════════════════════════════════════════════════════════════╝

Auteur: IA Assistant
Version: 1.0
Date: 2025
Supports: PDF scannés, images PNG/JPG
"""

import streamlit as st
import pandas as pd
import numpy as np
import pytesseract
import cv2
from PIL import Image
import pdf2image
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import tempfile
import os
import io
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum

# ═══════════════════════════════════════════════════════════════════════════
# ▶ CONFIGURATION LOGGING
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# ▶ CONFIGURATION TESSERACT POUR CLOUD LINUX
# ═══════════════════════════════════════════════════════════════════════════

try:
    # Tentative 1 : Linux standard (Streamlit Cloud)
    pytesseract.pytesseract.pytesseract_cmd = '/usr/bin/tesseract'
except Exception as e:
    logger.warning(f"⚠️ Tesseract chemin par défaut. Erreur : {e}")

# Fallback pour configurations locales
if not os.path.exists(pytesseract.pytesseract.pytesseract_cmd or ''):
    for alt_path in ['/usr/bin/tesseract', '/opt/tesseract/bin/tesseract', 'tesseract']:
        try:
            pytesseract.pytesseract.pytesseract_cmd = alt_path
            break
        except:
            continue

# ═══════════════════════════════════════════════════════════════════════════
# ▶ CLASSES ET ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class BankType(Enum):
    """Types de banques supportées"""
    UNICS = "UNICS"
    ADVANS = "ADVANS"
    MUPECI = "MUPECI"
    FINANCIAL_HOUSE = "FINANCIAL HOUSE"
    CEPAC = "CEPAC"
    UNKNOWN = "UNKNOWN"


@dataclass
class Transaction:
    """Modèle de transaction bancaire"""
    trx_date: Optional[str] = None
    cheque_particulars: str = ""
    value_date: Optional[str] = None
    debit: Optional[float] = None
    credit: Optional[float] = None
    balance: Optional[float] = None
    raw_lines: List[str] = None
    confidence: float = 0.0
    
    def __post_init__(self):
        if self.raw_lines is None:
            self.raw_lines = []
    
    def to_dict(self) -> Dict:
        """Convertir en dictionnaire"""
        return {
            'Trx. Date': self.trx_date or '',
            'Cheq# / Particulars': self.cheque_particulars,
            'Value Date': self.value_date or '',
            'Debit': self.debit or '',
            'Credit': self.credit or '',
            'Balance': self.balance or '',
        }


# ═══════════════════════════════════════════════════════════════════════════
# ▶ REGEX PATTERNS ULTRA-FLEXIBLES
# ═══════════════════════════════════════════════════════════════════════════

class PatternLibrary:
    """Bibliothèque de regex pour extraction"""
    
    # Dates : support UNICS (02/Jan/2025) et autres formats
    DATE_PATTERNS = [
        r'\d{1,2}/[A-Za-z]{3}/\d{4}',              # 02/Jan/2025
        r'\d{2}/\d{2}/\d{4}',                      # 02/01/2025
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}',
    ]
    
    # Montants avec support parenthèses et virgules (ADVANS)
    AMOUNT_PATTERNS = [
        r'[\d,]+(?:\.\d{2})?',                     # 1,234,567.89
        r'\([\d,]+(?:\.\d{2})?\)',                 # (1,234.56)
        r'[\d\.]+(?:,\d{2})?',                     # Alternatives
    ]
    
    # Mois anglais
    MONTH_MAP = {
        'jan': '01', 'january': '01',
        'feb': '02', 'february': '02',
        'mar': '03', 'march': '03',
        'apr': '04', 'april': '04',
        'may': '05',
        'jun': '06', 'june': '06',
        'jul': '07', 'july': '07',
        'aug': '08', 'august': '08',
        'sep': '09', 'september': '09',
        'oct': '10', 'october': '10',
        'nov': '11', 'november': '11',
        'dec': '12', 'december': '12',
    }

# ═══════════════════════════════════════════════════════════════════════════
# ▶ PREPROCESSEUR D'IMAGE (OCR)
# ═══════════════════════════════════════════════════════════════════════════

class ImagePreprocessor:
    """Pipeline de prétraitement d'image pour OCR robuste"""
    
    @staticmethod
    def load_image(file_path: str) -> np.ndarray:
        """Charger image"""
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"❌ Impossible de charger l'image : {file_path}")
        return img
    
    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convertir en échelle de gris"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def apply_threshold(image: np.ndarray, method='binary') -> np.ndarray:
        """
        Appliquer seuillage adaptatif
        
        method: 'binary', 'adaptive', 'otsu'
        """
        if method == 'otsu':
            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'adaptive':
            thresh = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        else:  # binary
            _, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
        return thresh
    
    @staticmethod
    def denoise(image: np.ndarray) -> np.ndarray:
        """Réduire le bruit"""
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    @staticmethod
    def deskew(image: np.ndarray) -> np.ndarray:
        """Corriger l'inclinaison"""
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = 90 + angle
        
        (h, w) = image.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated
    
    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """Améliorer le contraste"""
        lab = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2GRAY)
    
    @classmethod
    def full_pipeline(cls, image_path: str, enhance: bool = True) -> np.ndarray:
        """Pipeline complet de prétraitement"""
        img = cls.load_image(image_path)
        gray = cls.to_grayscale(img)
        
        if enhance:
            gray = cls.enhance_contrast(gray)
        
        denoised = cls.denoise(gray)
        deskewed = cls.deskew(denoised)
        thresholded = cls.apply_threshold(deskewed, 'otsu')
        
        return thresholded


# ═══════════════════════════════════════════════════════════════════════════
# ▶ EXTRACTEUR OCR AVEC DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════

class OCRExtractor:
    """Extraction de texte brut via Tesseract"""
    
    @staticmethod
    def extract_text(image_path: str, lang: str = 'fra+eng') -> str:
        """
        Extraire texte via Tesseract
        
        lang: 'fra+eng' pour français + anglais (UNICS a du "chq no.")
        """
        try:
            config = '--oem 3 --psm 6'  # PSM 6 = colonnes/blocs
            text = pytesseract.image_to_string(
                Image.open(image_path),
                lang=lang,
                config=config
            )
            return text
        except Exception as e:
            logger.error(f"❌ Erreur OCR : {e}")
            raise
    
    @staticmethod
    def extract_with_confidence(image_path: str) -> Tuple[str, float]:
        """Extraire texte + confiance moyenne"""
        try:
            img = Image.open(image_path)
            data = pytesseract.image_to_data(img, lang='fra+eng', output_type='dict')
            
            confidences = [int(c) for c in data['conf'] if int(c) > 0]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            text = pytesseract.image_to_string(img, lang='fra+eng', config='--oem 3 --psm 6')
            return text, avg_confidence
        except Exception as e:
            logger.error(f"❌ Erreur confiance : {e}")
            return "", 0.0


# ═══════════════════════════════════════════════════════════════════════════
# ▶ PARSER DE TRANSACTIONS (LOGIQUE MÉTIER)
# ═══════════════════════════════════════════════════════════════════════════

class TransactionParser:
    """Parser avec mémoire pour fusionner lignes orphelines"""
    
    def __init__(self):
        self.transactions: List[Transaction] = []
        self.current_transaction: Optional[Transaction] = None
        self.line_buffer: List[str] = []
    
    def parse_text(self, raw_text: str) -> List[Transaction]:
        """
        Parser ultra-flexible avec mémoire
        
        Algorithme :
        1. Split par newline
        2. Pour chaque ligne :
           - Si date valide → nouvelle transaction
           - Sinon → ajouter à transaction courante
        3. Extraire montants et balance
        """
        lines = raw_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 5:
                continue
            
            # ┌─ DÉTECTION DE DATE (trigger nouvelle transaction)
            date_match = self._find_date(line)
            
            if date_match:
                # Sauvegarder transaction précédente
                if self.current_transaction and self.current_transaction.cheque_particulars:
                    self._finalize_transaction()
                
                # Créer nouvelle transaction
                self.current_transaction = Transaction(
                    trx_date=date_match,
                    raw_lines=[line]
                )
                
                # Extraire particulars de la même ligne (après la date)
                rest = line[len(date_match):].strip()
                if rest:
                    self.current_transaction.cheque_particulars = rest
            
            else:
                # ┌─ CONTINUATION DE TRANSACTION COURANTE
                if self.current_transaction is None:
                    # Orpheline (commence sans date)
                    self.current_transaction = Transaction(raw_lines=[line])
                
                # Ajouter à particulars (fusion multi-ligne)
                if self.current_transaction.cheque_particulars:
                    self.current_transaction.cheque_particulars += " " + line
                else:
                    self.current_transaction.cheque_particulars = line
                
                self.current_transaction.raw_lines.append(line)
        
        # Finaliser dernière transaction
        if self.current_transaction and self.current_transaction.cheque_particulars:
            self._finalize_transaction()
        
        return self.transactions
    
    def _find_date(self, line: str) -> Optional[str]:
        """
        Trouver et normaliser date dans ligne
        
        Support :
        - 02/Jan/2025
        - 02/01/2025
        """
        # Pattern 1 : DD/MMM/YYYY
        match1 = re.search(r'(\d{1,2})/([A-Za-z]{3})/(\d{4})', line)
        if match1:
            day, month, year = match1.groups()
            month_num = PatternLibrary.MONTH_MAP.get(month.lower(), '')
            if month_num:
                return f"{int(day):02d}/{month_num}/{year}"
        
        # Pattern 2 : DD/MM/YYYY
        match2 = re.search(r'(\d{2})/(\d{2})/(\d{4})', line)
        if match2:
            return match2.group(0)
        
        return None
    
    def _finalize_transaction(self):
        """Extraire montants et solde de transaction courante"""
        if not self.current_transaction:
            return
        
        full_text = '\n'.join(self.current_transaction.raw_lines)
        
        # Extraire montants
        self._extract_amounts(full_text)
        
        # Calculer confiance
        self.current_transaction.confidence = self._calculate_confidence()
        
        self.transactions.append(self.current_transaction)
    
    def _extract_amounts(self, text: str):
        """Extraire Debit, Credit, Balance"""
        # Chercher "Debit" suivi d'un montant
        debit_match = re.search(
            r'[Dd]ebit[:\s]+[\(]?([\d,]+(?:\.\d{2})?)',
            text
        )
        if debit_match:
            try:
                amount_str = debit_match.group(1).replace(',', '')
                self.current_transaction.debit = float(amount_str)
            except:
                pass
        
        # Chercher "Credit"
        credit_match = re.search(
            r'[Cc]redit[:\s]+[\(]?([\d,]+(?:\.\d{2})?)',
            text
        )
        if credit_match:
            try:
                amount_str = credit_match.group(1).replace(',', '')
                self.current_transaction.credit = float(amount_str)
            except:
                pass
        
        # Chercher "Balance" ou solde (dernier montant important)
        balance_match = re.search(
            r'[Bb]alance[:\s]+([\d,]+(?:\.\d{2})?)',
            text
        )
        if balance_match:
            try:
                amount_str = balance_match.group(1).replace(',', '')
                self.current_transaction.balance = float(amount_str)
            except:
                pass
    
    def _calculate_confidence(self) -> float:
        """Calculer confiance de la transaction"""
        score = 1.0
        
        if not self.current_transaction.trx_date:
            score -= 0.3
        if not self.current_transaction.cheque_particulars:
            score -= 0.3
        if not (self.current_transaction.debit or self.current_transaction.credit):
            score -= 0.2
        
        return max(0.0, score)


# ═══════════════════════════════════════════════════════════════════════════
# ▶ DÉTECTEUR DE TYPE DE BANQUE
# ═══════════════════════════════════════════════════════════════════════════

class BankDetector:
    """Détecter type de banque à partir du texte"""
    
    SIGNATURES = {
        BankType.UNICS: [
            'UNICS', 'Category II Microfinance', 'Bafoussam Branch',
            'chq no.', 'WDL chq'
        ],
        BankType.ADVANS: [
            'ADVANS', 'Advans', 'Cameroon', 'CBNC'
        ],
        BankType.MUPECI: [
            'MUPECI', 'Union', 'mupeci'
        ],
        BankType.FINANCIAL_HOUSE: [
            'FINANCIAL HOUSE', 'Financial House', 'FH'
        ],
        BankType.CEPAC: [
            'CEPAC', 'Cepac', 'cepac'
        ]
    }
    
    @classmethod
    def detect(cls, text: str) -> BankType:
        """Détecter banque à partir du texte"""
        text_upper = text.upper()
        
        for bank_type, signatures in cls.SIGNATURES.items():
            for sig in signatures:
                if sig.upper() in text_upper:
                    return bank_type
        
        return BankType.UNKNOWN


# ═══════════════════════════════════════════════════════════════════════════
# ▶ PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

class BankStatementPipeline:
    """Pipeline complet : PDF → OCR → Parse → Excel"""
    
    def __init__(self, diagnostic_mode: bool = False):
        self.diagnostic_mode = diagnostic_mode
        self.diagnostics = {
            'input_file': '',
            'num_pages': 0,
            'bank_type': BankType.UNKNOWN,
            'raw_text': '',
            'ocr_confidence': 0.0,
            'num_transactions': 0,
            'errors': []
        }
    
    def process_pdf(self, pdf_path: str) -> Tuple[List[Transaction], Dict]:
        """Traiter PDF complet"""
        try:
            self.diagnostics['input_file'] = pdf_path
            
            # Extraire images du PDF
            images = pdf2image.convert_from_path(pdf_path)
            self.diagnostics['num_pages'] = len(images)
            
            all_text = ""
            total_confidence = 0.0
            
            st.info(f"📄 {len(images)} page(s) détectée(s). Traitement OCR en cours...")
            progress_bar = st.progress(0)
            
            for idx, image in enumerate(images):
                # Sauvegarder image temporaire
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    image.save(tmp.name)
                    
                    # Prétraitement
                    preprocessed = ImagePreprocessor.full_pipeline(tmp.name, enhance=True)
                    cv2.imwrite(tmp.name, preprocessed)
                    
                    # OCR avec confiance
                    text, confidence = OCRExtractor.extract_with_confidence(tmp.name)
                    all_text += "\n" + text
                    total_confidence += confidence
                    
                    os.unlink(tmp.name)
                
                progress_bar.progress((idx + 1) / len(images))
            
            self.diagnostics['raw_text'] = all_text
            self.diagnostics['ocr_confidence'] = total_confidence / len(images) if images else 0
            
            # Détecter banque
            bank_type = BankDetector.detect(all_text)
            self.diagnostics['bank_type'] = bank_type
            
            # Parser
            parser = TransactionParser()
            transactions = parser.parse_text(all_text)
            self.diagnostics['num_transactions'] = len(transactions)
            
            return transactions, self.diagnostics
        
        except Exception as e:
            error_msg = f"❌ Erreur pipeline : {str(e)}"
            self.diagnostics['errors'].append(error_msg)
            logger.error(error_msg)
            raise
    
    def export_to_excel(self, transactions: List[Transaction], output_path: str):
        """Exporter transactions en Excel"""
        df = pd.DataFrame([t.to_dict() for t in transactions])
        
        # Formatage Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Transactions', index=False)
            
            # Formatage colonnes
            ws = writer.sheets['Transactions']
            for idx, col in enumerate(df.columns):
                ws.column_dimensions[chr(65 + idx)].width = 20
        
        logger.info(f"✅ Excel créé : {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# ▶ INTERFACE STREAMLIT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="🏦 Bank Statement OCR",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # ┌─ HEADER
    st.markdown("""
    <h1 style='text-align: center; color: #1f77b4;'>
    🏦 BANK STATEMENT OCR EXTRACTOR
    </h1>
    <p style='text-align: center; color: #666;'>
    Extraction automatisée de relevés bancaires camerounais
    </p>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # ┌─ SIDEBAR
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        diagnostic_mode = st.checkbox(
            "🔍 Mode Diagnostic",
            value=False,
            help="Affiche le texte brut OCR pour déboguer"
        )
        
        ocr_lang = st.selectbox(
            "🌐 Langue OCR",
            ["fra+eng", "eng", "fra"],
            help="fra+eng recommandé pour UNICS"
        )
        
        enhance_images = st.checkbox(
            "🖼️ Améliorer images",
            value=True,
            help="Appliquer CLAHE pour meilleur contraste"
        )
        
        st.divider()
        st.markdown("### 📋 Banques Supportées")
        for bank in BankType:
            if bank != BankType.UNKNOWN:
                st.markdown(f"- ✅ {bank.value}")
    
    # ┌─ MAIN CONTENT
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📤 Charger Relevé Bancaire")
        uploaded_file = st.file_uploader(
            "Sélectionnez un PDF ou image",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            help="Scans acceptés (OCR automatique)"
        )
    
    with col2:
        st.subheader("✨ Infos Fichier")
        if uploaded_file:
            st.metric("Taille", f"{uploaded_file.size / 1024:.1f} KB")
            st.metric("Type", uploaded_file.type)
    
    st.divider()
    
    if uploaded_file:
        # Sauvegarder fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        
        try:
            # ┌─ TRAITEMENT
            st.info("🔄 Traitement en cours...")
            
            pipeline = BankStatementPipeline(diagnostic_mode=diagnostic_mode)
            transactions, diagnostics = pipeline.process_pdf(tmp_path)
            
            # ┌─ AFFICHAGE DIAGNOSTIC
            if diagnostic_mode:
                st.warning("🔍 MODE DIAGNOSTIC ACTIF")
                
                with st.expander("📝 Texte Brut OCR", expanded=False):
                    st.code(diagnostics['raw_text'][:2000] + "...", language="text")
                
                with st.expander("📊 Métadonnées", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Pages", diagnostics['num_pages'])
                    with col2:
                        st.metric("Banque Détectée", diagnostics['bank_type'].value)
                    with col3:
                        st.metric("Confiance OCR", f"{diagnostics['ocr_confidence']:.1f}%")
            
            # ┌─ RÉSULTATS
            st.success(f"✅ {len(transactions)} transaction(s) extraite(s)")
            
            # Afficher tableau
            if transactions:
                df = pd.DataFrame([t.to_dict() for t in transactions])
                
                st.subheader("📊 Aperçu Transactions")
                st.dataframe(df, use_container_width=True, height=400)
                
                # Stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Débits", f"{df['Debit'].sum():,.0f}")
                with col2:
                    st.metric("Crédits", f"{df['Credit'].sum():,.0f}")
                with col3:
                    st.metric("Solde Final", f"{df['Balance'].iloc[-1]:,.0f}" if len(df) > 0 else "N/A")
                with col4:
                    st.metric("Confiance Moy", f"{np.mean([t.confidence for t in transactions]):.1%}")
                
                # ┌─ TÉLÉCHARGEMENT
                st.divider()
                st.subheader("⬇️ Télécharger Résultats")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Excel
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name='Transactions', index=False)
                    
                    st.download_button(
                        label="📊 Télécharger Excel",
                        data=excel_buffer.getvalue(),
                        file_name=f"bank_statement_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.ms-excel"
                    )
                
                with col2:
                    # CSV
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    
                    st.download_button(
                        label="📄 Télécharger CSV",
                        data=csv_buffer.getvalue(),
                        file_name=f"bank_statement_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            else:
                st.warning("⚠️ Aucune transaction détectée. Vérifiez la qualité du scan.")
        
        finally:
            # Nettoyage
            os.unlink(tmp_path)
    
    else:
        # ┌─ PAGE VIDE
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            ### 👆 Commencez par charger un relevé bancaire
            
            **Formats acceptés :**
            - 📄 PDF (scannés)
            - 🖼️ PNG, JPG
            
            **Banques supportées :**
            - UNICS (Microfinance)
            - ADVANS
            - MUPECI
            - FINANCIAL HOUSE
            - CEPAC
            
            **Fonctionnalités :**
            ✅ OCR intelligent avec prétraitement
            ✅ Fusion auto des libellés multi-lignes
            ✅ Détection automatique du type de banque
            ✅ Export Excel/CSV
            ✅ Mode diagnostic pour déboguer
            """)


if __name__ == "__main__":
    main()
