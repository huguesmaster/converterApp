"""
Extracteur Hybride : pdfplumber (texte) + Gemini (structuration)
Stratégie d'optimisation des tokens :
  1. pdfplumber extrait le texte brut de chaque page
  2. Gemini reçoit du TEXTE (pas des images)
  3. Gemini structure les données en JSON propre
  → ~85% de tokens économisés vs vision pure
"""

import google.generativeai as genai
import pdfplumber
import pandas as pd
import json
import re
import io
import time
import hashlib


class GeminiExtractor:
    """
    Extracteur hybride pdfplumber + Gemini Text.

    Flux :
    PDF bytes
      └─► pdfplumber  →  texte brut par page
            └─► Gemini Text  →  JSON structuré
                  └─► DataFrame pandas
    """

    # ── Prompt optimisé pour texte brut ──────────────
    # Compact = moins de tokens dans le prompt système
    SYSTEM_PROMPT = """Tu es un extracteur de relevés bancaires.
On te donne le texte brut d'une page de relevé Financial House S.A.

COLONNES DU TABLEAU (dans l'ordre d'apparition) :
Date | Batch/Ref | Libellé | D.Valeur | Débit | Crédit | Solde

RÈGLES :
- Extraire TOUTES les lignes de transaction
- Ignorer : en-têtes (Date/Batch/Ref/Libellé...), totaux (TOTAUX), 
  infos banque (Branch ID, Account ID, Printed By, Working Date, Page Num)
- Inclure : "Solde d'ouverture" et "Solde de clôture"
- Montants : supprimer les espaces → "24 553 342" devient 24553342
- Si Débit vide → null, si Crédit vide → null
- Les frais (SMS Pack, TVA, Historique) sont toujours des DÉBITS

RETOURNER UNIQUEMENT ce JSON sans markdown :
{"transactions":[{"date":"JJ/MM/AAAA","reference":"XXX/","libelle":"...","date_valeur":"JJ/MM/AAAA","debit":null_ou_nombre,"credit":null_ou_nombre,"solde":nombre}]}"""

    # Nombre max de tokens estimés par page
    # (pour décider si on chunke ou pas)
    MAX_TOKENS_PER_PAGE = 3000

    def __init__(self, api_key: str, progress_callback=None):
        self.api_key = api_key
        self.progress_callback = progress_callback
        self._cache = {}  # Cache pour éviter de retraiter

        genai.configure(api_key=api_key)

        # Modèle TEXT uniquement (beaucoup moins cher)
        self.model = genai.GenerativeModel(
            model_name="gemini-3.1-flash-lite-preview",
            generation_config=genai.GenerationConfig(
                temperature=0,
                top_p=1,
                top_k=1,
                max_output_tokens=4096,
            ),
            # Paramètres de sécurité permissifs pour données financières
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT",
                 "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH",
                 "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                 "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                 "threshold": "BLOCK_NONE"},
            ]
        )

    def _update_progress(self, step: int, message: str):
        if self.progress_callback:
            self.progress_callback(step, message)

    # ──────────────────────────────────────────────────
    # ÉTAPE 1 : EXTRACTION TEXTE VIA pdfplumber
    # ──────────────────────────────────────────────────

    def _extract_text_from_pdf(
        self, pdf_bytes: bytes
    ) -> list[dict]:
        """
        Extrait le texte brut de chaque page avec pdfplumber.
        Retourne une liste de dicts {page_num, text, char_count}.

        C'est cette étape qui remplace l'envoi d'images à Gemini.
        """
        pages_text = []

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            total = len(pdf.pages)

            for i, page in enumerate(pdf.pages):
                progress = 5 + int((i / total) * 30)
                self._update_progress(
                    progress,
                    f"📄 Extraction texte page {i+1}/{total}..."
                )

                # Méthode 1 : extraction directe du texte
                raw_text = page.extract_text(
                    x_tolerance=3,
                    y_tolerance=3,
                    layout=True,          # Préserve la mise en page
                    x_density=7.25,
                    y_density=13,
                )

                # Méthode 2 : si texte vide, essayer par mots
                if not raw_text or len(raw_text.strip()) < 50:
                    raw_text = self._extract_text_by_words(page)

                if raw_text:
                    # Nettoyer le texte avant envoi à Gemini
                    cleaned = self._preprocess_text(raw_text)
                    pages_text.append({
                        'page_num':   i + 1,
                        'text':       cleaned,
                        'char_count': len(cleaned),
                        'total_pages': total,
                    })

        return pages_text

    def _extract_text_by_words(self, page) -> str:
        """
        Extraction par mots avec reconstruction des lignes.
        Utilisé quand extract_text() donne un résultat vide.
        """
        words = page.extract_words(
            x_tolerance=4,
            y_tolerance=4,
            keep_blank_chars=False,
        )
        if not words:
            return ''

        # Grouper par ligne (position Y)
        lines = {}
        for w in words:
            y = round(float(w['top']) / 4) * 4
            lines.setdefault(y, []).append(w)

        # Reconstruire les lignes dans l'ordre
        result_lines = []
        for y in sorted(lines.keys()):
            line_words = sorted(lines[y], key=lambda w: w['x0'])
            line = ' '.join(w['text'] for w in line_words)
            result_lines.append(line)

        return '\n'.join(result_lines)

    def _preprocess_text(self, text: str) -> str:
        """
        Nettoie le texte avant envoi à Gemini pour réduire les tokens.

        Actions :
        - Supprime les lignes vides multiples
        - Supprime les lignes inutiles (logo, adresse...)
        - Normalise les espaces
        """
        lines = text.split('\n')
        cleaned_lines = []

        # Patterns à supprimer du texte
        skip_patterns = [
            r'^\s*$',                          # Lignes vides
            r'financial house s\.?a',          # Nom banque
            r'fh sa\s*-\s*ndjiemoum',          # Agence
            r'historique compte client',       # Titre
            r'branch id\s*:',                  # Info compte
            r'account id\s*:',
            r'print date\s*:',
            r'page num\s*:',
            r'printed by\s*:',
            r'working date\s*:',
            r'br\.net ver',
            r'^\s*ibantio\s*$',
            r'^\s*bdjoko\s*$',
            r'^\s*gtakeu\s*$',
            r'^\s*ideynou\s*$',
            r'^\s*mmodjo\s*$',
        ]

        prev_was_empty = False
        for line in lines:
            line_stripped = line.strip()
            line_lower = line_stripped.lower()

            # Vérifier si la ligne doit être ignorée
            should_skip = False
            for pattern in skip_patterns:
                if re.search(pattern, line_lower):
                    should_skip = True
                    break

            if should_skip:
                continue

            # Éviter les lignes vides consécutives
            if not line_stripped:
                if not prev_was_empty:
                    cleaned_lines.append('')
                prev_was_empty = True
            else:
                cleaned_lines.append(line_stripped)
                prev_was_empty = False

        return '\n'.join(cleaned_lines).strip()

    # ──────────────────────────────────────────────────
    # ÉTAPE 2 : COMPTAGE ET GESTION DES TOKENS
    # ──────────────────────────────────────────────────

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimation rapide du nombre de tokens.
        Règle approximative : 1 token ≈ 4 caractères en français.
        """
        return len(text) // 4

    def _should_chunk(self, text: str) -> bool:
        """
        Détermine si le texte doit être découpé en chunks
        pour éviter de dépasser les limites de tokens.
        """
        estimated = self._estimate_tokens(text)
        return estimated > self.MAX_TOKENS_PER_PAGE

    def _split_into_chunks(
        self, text: str, max_chars: int = 8000
    ) -> list[str]:
        """
        Découpe un texte long en chunks logiques.
        Découpe sur les lignes vides pour ne pas couper
        au milieu d'une transaction.
        """
        if len(text) <= max_chars:
            return [text]

        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_len = 0

        for line in lines:
            line_len = len(line) + 1  # +1 pour le \n

            if current_len + line_len > max_chars and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_len = line_len
            else:
                current_chunk.append(line)
                current_len += line_len

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

    def _get_cache_key(self, text: str) -> str:
        """Génère une clé de cache MD5 pour un texte."""
        return hashlib.md5(text.encode()).hexdigest()

    # ──────────────────────────────────────────────────
    # ÉTAPE 3 : APPEL GEMINI OPTIMISÉ
    # ──────────────────────────────────────────────────

    def _call_gemini(
        self, text: str, page_info: str = ""
    ) -> list[dict]:
        """
        Appelle Gemini avec du TEXTE (pas d'image).
        Gère le cache, les retries et le rate limiting.

        Returns:
            Liste de transactions extraites
        """
        # Vérifier le cache
        cache_key = self._get_cache_key(text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Si le texte est trop long → découper
        if self._should_chunk(text):
            return self._call_gemini_chunked(text, page_info)

        # Construire le prompt utilisateur
        # (le texte brut de la page)
        user_prompt = f"""Voici le texte brut de la page {page_info} :

---
{text}
---

Extrais toutes les transactions et retourne le JSON."""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    [self.SYSTEM_PROMPT, user_prompt],
                    request_options={"timeout": 30}
                )

                transactions = self._parse_response(response.text)

                # Mettre en cache
                self._cache[cache_key] = transactions
                return transactions

            except Exception as e:
                error_str = str(e)

                # Rate limiting → attendre
                if any(code in error_str
                       for code in ['429', 'quota', 'RESOURCE_EXHAUSTED']):
                    wait = (attempt + 1) * 15
                    self._update_progress(
                        -1,
                        f"⏳ Limite API Gemini — attente {wait}s..."
                    )
                    time.sleep(wait)
                    continue

                # Erreur de contexte trop long
                if 'context' in error_str.lower() or '413' in error_str:
                    # Réessayer avec chunking
                    return self._call_gemini_chunked(text, page_info)

                print(f"Erreur Gemini ({attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return []
                time.sleep(3)

        return []

    def _call_gemini_chunked(
        self, text: str, page_info: str = ""
    ) -> list[dict]:
        """
        Traite un texte long par chunks successifs.
        Chaque chunk est envoyé séparément à Gemini.
        """
        chunks = self._split_into_chunks(text, max_chars=6000)
        all_transactions = []

        for i, chunk in enumerate(chunks):
            self._update_progress(
                -1,
                f"🔄 Chunk {i+1}/{len(chunks)} de {page_info}..."
            )

            chunk_info = f"{page_info} (partie {i+1}/{len(chunks)})"
            transactions = self._call_gemini(chunk, chunk_info)
            all_transactions.extend(transactions)

            # Pause entre chunks
            if i < len(chunks) - 1:
                time.sleep(1)

        return all_transactions

    # ──────────────────────────────────────────────────
    # ÉTAPE 4 : PARSING DE LA RÉPONSE GEMINI
    # ──────────────────────────────────────────────────

    def _parse_response(self, response_text: str) -> list[dict]:
        """
        Parse la réponse JSON de Gemini.
        Robuste aux formats variés.
        """
        if not response_text:
            return []

        text = response_text.strip()

        # Supprimer les balises markdown si présentes
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()

        # Extraire le JSON
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not json_match:
            return []

        json_str = json_match.group(0)

        # Tenter de parser
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            json_str = self._repair_json(json_str)
            try:
                data = json.loads(json_str)
            except Exception:
                return []

        transactions = data.get("transactions", [])
        if not isinstance(transactions, list):
            return []

        return [
            t for t in
            (self._normalize(t) for t in transactions)
            if t is not None
        ]

    def _repair_json(self, s: str) -> str:
        """Répare les JSON légèrement malformés."""
        s = re.sub(r',\s*}', '}', s)
        s = re.sub(r',\s*]', ']', s)
        s = s.replace('None', 'null')
        s = s.replace("'", '"')
        return s

    def _normalize(self, t: dict) -> dict:
        """Normalise et valide une transaction."""
        if not isinstance(t, dict):
            return None

        libelle = str(t.get('libelle', '') or '').strip()
        if not libelle:
            return None

        return {
            'date':        self._fmt_date(t.get('date', '')),
            'reference':   str(t.get('reference', '') or '').strip(),
            'libelle':     libelle,
            'date_valeur': self._fmt_date(t.get('date_valeur', '')),
            'debit':       self._fmt_amount(t.get('debit')),
            'credit':      self._fmt_amount(t.get('credit')),
            'solde':       self._fmt_amount(t.get('solde')),
        }

    def _fmt_date(self, val) -> str:
        """Normalise une date."""
        if not val:
            return ''
        s = str(val).strip()
        if re.match(r'\d{2}/\d{2}/\d{4}', s):
            return s[:10]
        return s

    def _fmt_amount(self, val) -> float:
        """Convertit un montant en float ou None."""
        if val is None or val == '' or str(val).lower() == 'null':
            return None
        try:
            s = re.sub(r'[^\d.]', '', str(val).replace(' ', ''))
            return float(s) if s else None
        except (ValueError, TypeError):
            return None

    # ──────────────────────────────────────────────────
    # POINT D'ENTRÉE PRINCIPAL
    # ──────────────────────────────────────────────────

    def extract(self, pdf_bytes: bytes) -> pd.DataFrame:
        """
        Pipeline complet d'extraction hybride.

        Étapes :
        1. pdfplumber  → texte brut par page
        2. Gemini Text → JSON structuré par page
        3. Consolidation → DataFrame final

        Returns:
            DataFrame : Date, Référence, Libellé,
                        Date_Valeur, Débit, Crédit, Solde
        """
        self._update_progress(3, "📄 Lecture du PDF...")

        # ── Étape 1 : Extraction texte ──
        try:
            pages = self._extract_text_from_pdf(pdf_bytes)
        except Exception as e:
            self._update_progress(10, f"❌ Erreur lecture PDF: {e}")
            return self._empty_df()

        if not pages:
            self._update_progress(10, "❌ PDF illisible ou vide")
            return self._empty_df()

        total_pages = len(pages)
        self._update_progress(
            35,
            f"✅ {total_pages} page(s) extraite(s) — "
            f"envoi à Gemini..."
        )

        # ── Étape 2 : Structuration via Gemini ──
        all_transactions = []

        for page_data in pages:
            page_num  = page_data['page_num']
            text      = page_data['text']
            char_count = page_data['char_count']
            tokens_est = self._estimate_tokens(text)

            progress = 35 + int(
                (page_num / total_pages) * 50
            )
            self._update_progress(
                progress,
                f"🤖 Gemini structure la page "
                f"{page_num}/{total_pages} "
                f"(~{tokens_est} tokens)..."
            )

            # Appel Gemini avec le texte
            transactions = self._call_gemini(
                text,
                page_info=f"{page_num}/{total_pages}"
            )
            all_transactions.extend(transactions)

            # Pause minimale entre pages
            # (évite le rate limiting)
            if page_num < total_pages:
                time.sleep(0.5)

        self._update_progress(88, "🔧 Construction du tableau...")

        # ── Étape 3 : DataFrame ──
        if not all_transactions:
            self._update_progress(90, "⚠️ Aucune transaction extraite")
            return self._empty_df()

        df = self._build_dataframe(all_transactions)
        self._update_progress(95, "✅ Extraction terminée !")
        return df

    def _build_dataframe(self, transactions: list) -> pd.DataFrame:
        """Construit le DataFrame final depuis la liste."""
        rows = [{
            'Date':        t.get('date', ''),
            'Référence':   t.get('reference', ''),
            'Libellé':     t.get('libelle', ''),
            'Date_Valeur': t.get('date_valeur', ''),
            'Débit':       t.get('debit'),
            'Crédit':      t.get('credit'),
            'Solde':       t.get('solde'),
        } for t in transactions]

        df = pd.DataFrame(rows)

        # Supprimer les vrais doublons
        mask_solde = df['Libellé'].str.contains(
            r'solde\s+d|solde\s+de\s+cl',
            case=False, na=False, regex=True
        )
        df_normal = df[~mask_solde].drop_duplicates(
            subset=['Date', 'Référence', 'Libellé'],
            keep='first'
        )
        df_soldes = df[mask_solde]

        df_final = pd.concat(
            [df_soldes, df_normal], ignore_index=True
        )

        return df_final.reset_index(drop=True)

    def _empty_df(self) -> pd.DataFrame:
        return pd.DataFrame(columns=[
            'Date', 'Référence', 'Libellé',
            'Date_Valeur', 'Débit', 'Crédit', 'Solde'
        ])

    # ──────────────────────────────────────────────────
    # MÉTRIQUES D'UTILISATION
    # ──────────────────────────────────────────────────

    def get_usage_stats(self, pages: list[dict]) -> dict:
        """
        Calcule les métriques d'utilisation des tokens
        pour affichage dans l'interface.
        """
        total_chars  = sum(p['char_count'] for p in pages)
        total_tokens = self._estimate_tokens(total_chars * ' ')
        total_pages  = len(pages)

        # Coût estimé Gemini Flash
        # (prix : ~$0.075 / 1M tokens input)
        cost_usd = (total_tokens / 1_000_000) * 0.075

        # Comparaison avec le mode image
        image_tokens_estimate = total_pages * 2000
        image_cost_estimate = (
            image_tokens_estimate / 1_000_000
        ) * 0.075

        return {
            'total_pages':          total_pages,
            'total_chars':          total_chars,
            'tokens_used':          total_tokens,
            'tokens_saved':         image_tokens_estimate - total_tokens,
            'savings_pct': round(
                (1 - total_tokens / max(image_tokens_estimate, 1))
                * 100
            ),
            'cost_usd':             round(cost_usd, 6),
            'cost_image_usd':       round(image_cost_estimate, 6),
        }
