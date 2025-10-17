import os
import json
from flask import Flask, request, render_template, jsonify
from markupsafe import Markup
from markdown import markdown
from dotenv import load_dotenv
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import re
import unicodedata
import difflib
import random

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
DEFAULT_GOOGLE_MODEL = os.getenv('GOOGLE_MODEL', 'models/gemini-flash-latest')
ALLOW_LLM_FALLBACK = os.getenv('ALLOW_LLM_FALLBACK', 'true').lower() in ('1','true','yes')

if not api_key:
    raise RuntimeError('GOOGLE_API_KEY gerekli')

# Load CSV and build vector store (same as app.py)
df = pd.read_csv('financial_terms_final.csv')
docs = []

def normalize_tr(text: str) -> str:
    t = (text or '').lower().strip()
    replacements = {
        'ş':'s','ğ':'g','ç':'c','ö':'o','ü':'u','ı':'i','İ':'i','Ş':'s','Ğ':'g','Ç':'c','Ö':'o','Ü':'u'
    }
    t = ''.join(replacements.get(ch, ch) for ch in t)
    t = ''.join(ch for ch in unicodedata.normalize('NFKD', t) if not unicodedata.combining(ch))
    t = re.sub(r"[^a-z0-9\s/().%-]", ' ', t)
    t = re.sub(r"\s+", ' ', t).strip()
    return t

# Build term maps for fast lookup
TERM_MAP = {}           # exact original term -> desc
NORM_TERM_MAP = {}      # normalized term -> desc

for _, row in df.iterrows():
    term = str(row.get('Terim','')).strip()
    desc = ''
    for key in ('Açıklama','Tanım','Tanim','Description','Desc'):
        if key in row and pd.notna(row.get(key)):
            desc = str(row.get(key,'')).strip()
            if desc:
                break
    content = f"{term}: {desc}" if desc else term
    docs.append(Document(page_content=content, metadata={'term': term}))
    if term:
        TERM_MAP[term] = desc
        # index full term
        NORM_TERM_MAP[normalize_tr(term)] = desc
        # index base term without parentheses, e.g., "Halka Arz (IPO)" -> "Halka Arz"
        base = term.split('(')[0].strip()
        if base:
            NORM_TERM_MAP[normalize_tr(base)] = desc
        # index aliases inside parentheses: e.g., "IPO"
        m = re.search(r"\((.*?)\)", term)
        if m:
            aliases = re.split(r"[,/|;]", m.group(1))
            for al in aliases:
                al = al.strip()
                if al:
                    NORM_TERM_MAP[normalize_tr(al)] = desc

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(docs)
embeddings = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004', google_api_key=api_key)

# Try to load existing FAISS index to avoid re-embedding on every run
faiss_dir = 'faiss_index_financial_terms'
if os.path.exists(faiss_dir):
    try:
        vector_store = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        vector_store = FAISS.from_documents(texts, embeddings)
        vector_store.save_local(faiss_dir)
else:
    vector_store = FAISS.from_documents(texts, embeddings)
    try:
        vector_store.save_local(faiss_dir)
    except Exception:
        pass

# Slightly higher k to capture related/nearby concepts
retriever = vector_store.as_retriever(search_kwargs={'k':12})

llm = ChatGoogleGenerativeAI(model=DEFAULT_GOOGLE_MODEL, temperature=0.3, google_api_key=api_key, convert_system_message_to_human=True)
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, return_source_documents=True,
    chain_type='stuff',
    combine_docs_chain_kwargs={
        'prompt': PromptTemplate(
            input_variables=['context','question'],
            template=(
                "Aşağıda kaynak belgeler (context) veriliyor. Kullanıcının sorusunu sadece bu belgelerdeki bilgiye dayanarak cevapla.\n"
                "- Önce kısa ve doğrudan bir cevap ver.\n"
                "- Cevabı Türkçe yaz.\n"
                "- Cevapta hangi belgeden bilgi kullandıysan kaynak olarak belgenin 'term' alanını köşeli parantez içinde belirt.\n\n"
                "Context:\n{context}\n\nSoru: {question}\n\nCevap (kısa, kaynaklı):"
            )
        )
    }
)

app = Flask(__name__)

# Expose a server-side markdown renderer to Jinja templates
def render_markdown(text: str) -> Markup:
    try:
        html = markdown(text or '', extensions=['fenced_code', 'codehilite'])
    except Exception:
        # fallback: escape and wrap
        html = (text or '')
    return Markup(html)

app.jinja_env.globals['renderMarkdown'] = render_markdown

# Load small talk responses file if present
SMALL_TALK = {}
try:
    with open('small_talk_responses.json', 'r', encoding='utf-8') as f:
        SMALL_TALK = json.load(f)
except Exception:
    SMALL_TALK = {}

@app.route('/')
def index():
    # Provide default context so template can render even without prior conversations
    return render_template('index.html', conversations=[], conversation_history=[], current_session=None)

@app.route('/api/chat', methods=['POST'])
@app.route('/send_message', methods=['POST'])
def chat_api():
    data = request.json or {}
    # Accept both 'question' (used by /api/chat) and 'message' (used by inline template JS /send_message)
    question = (data.get('question') or data.get('message') or '')
    chat_history = data.get('chat_history', [])

    # small talk handling using SMALL_TALK json
    def is_small_talk(q):
        qn = normalize_tr((q or '').strip())
        # Normalize Turkish chars to catch variants (tesekkurler ~ teşekkürler)
        greet = r"^\s*(merhaba|merhabalar|selam|selamun aleykum|aleykum selam|hey+|hi+|hello+|yo+|whats up|sup)"
        howru = r"\b(nasilsin|naber|ne haber|napiyorsun|nasil gidiyor|keyifler nasil|iyi misin|how are you|how r u|whats up|hows it going|how you doing)\b"
        thanks = r"\b(tesekkur(ler| ederim| ederiz)?|sag ?ol(un)?|eyvallah|cok tesekkur|thanks?|thank you|thx|ty|appreciate( it)?)\b"
        bye = r"\b(gule gule|hosca kal(in)?|gorusuruz|gorusmek uzere|kendine iyi bak|esen kal|bay ?bay|bye( bye)?|see you( soon)?|take care|farewell|later|cya)\b"
        for p in (greet, howru, thanks, bye):
            if re.search(p, qn):
                return True
        return False

    def canned_small_talk_response(q):
        q = (q or '').strip().lower()
        lang_map = SMALL_TALK.get('tr', {})
        # greetings
        if re.search(r"^\s*(merhaba(?:lar)?|selam(?:lar)?|selamün[ '‘’`]?aleyküm|aleyküm[ '‘’`]?selam|hey+(?: there)?|hi+(?: there)?|hello+(?: there)?|yo+|what'?s up|sup)", q, re.IGNORECASE):
            return random.choice(lang_map.get('greetings', [])) if lang_map.get('greetings') else lang_map.get('fallback')
        if re.search(r"\b(nasıls[ıi]n?|nasilsin|nab[ée]r|naber|ne haber|napıyo?sun|napıyorsun|nasıl gidiyor|keyifler nasıl|iyimi?s[ıi]n?|how are (you|u)|how r u|hru|sup|what'?s up|how'?s it going|how you doing|how’ve you been|how r ya)\b", q, re.IGNORECASE):
            return random.choice(lang_map.get('how_are_you', [])) if lang_map.get('how_are_you') else lang_map.get('fallback')
        if re.search(r"\b(teşekkür(?:ler| ederim| ederiz)?|sağ ?ol(?:un)?|minnettar(?:ım)?|çok ?teşekkür|eyvallah|thanks?|thank you|thx|ty|appreciate(?: it)?)\b", q, re.IGNORECASE):
            return random.choice(lang_map.get('thanks', [])) if lang_map.get('thanks') else lang_map.get('fallback')
        if re.search(r"\b(güle güle|hoşça kal(?:ın)?|görüş(ürüz|mek üzere)?|kendine iyi bak|esen kal|bay ?bay|bye(?: bye)?|see you(?: soon)?|take care|farewell|later|cya|görüşürüz)\b", q, re.IGNORECASE):
            return random.choice(lang_map.get('goodbye', [])) if lang_map.get('goodbye') else lang_map.get('fallback')
        # default
        suffix = lang_map.get('guidance_suffix', '')
        return (lang_map.get('fallback') or '') + suffix

    if not question:
        return jsonify({'error':'empty question'}), 400

    if is_small_talk(question):
        return jsonify({'response': canned_small_talk_response(question), 'conversations': []})

    # 1) Deterministic CSV lookup (handles "X nedir?" type queries robustly)
    def extract_term_from_question(q: str) -> str | None:
        qn = normalize_tr(q)
        # patterns like "likidite nedir", "likidite ne demek", "likiditenin tanimi", "... aciklamasi"
        patterns = [
            r"^(.*?)\s+(nedir|ne demek|ne anlama gelir|ne anlamina gelir|ne ifade eder|ne)\b",
            r"^(.*?)\s+(tanimi|tanım|aciklama|açiklama|aciklamasi|açıklaması)\b",
            r"^\s*([a-z0-9 /().%-]+)\s*$"  # bare term
        ]
        for p in patterns:
            m = re.search(p, qn)
            if m:
                cand = (m.group(1) or '').strip()
                cand = re.sub(r"\b(nedir|ne demek|tanimi|tanım|aciklama|açiklama|aciklamasi|açıklaması)\b", '', cand).strip()
                if cand:
                    return cand
        return None

    def lookup_term(q: str) -> str | None:
        # exact normalized match only; avoid over-catching substrings
        cand = extract_term_from_question(q)
        if cand:
            candn = normalize_tr(cand)
            if candn in NORM_TERM_MAP:
                return NORM_TERM_MAP[candn]
            # fuzzy match on normalized keys to tolerate small typos (e.g., enflasyn -> enflasyon)
            keys = list(NORM_TERM_MAP.keys())
            close = difflib.get_close_matches(candn, keys, n=1, cutoff=0.8)
            if close:
                return NORM_TERM_MAP.get(close[0])
            # fallback: try matching against base forms of known terms
            for orig_term, d in TERM_MAP.items():
                base = orig_term.split('(')[0].strip()
                if base and normalize_tr(base) == candn:
                    return d
        return None

    # simple Turkish finance synonyms to help retrieval/LLM when term missing in CSV
    SYNONYMS = {
        'halka arz': ['ipo', 'initial public offering', 'ilk halka arz'],
        'hisse senedi': ['equity', 'stock', 'share'],
        'tahvil': ['bond', 'debt security'],
        'temettü': ['dividend'],
        'kaldıraç': ['leverage'],
        'faiz oranı': ['interest rate'],
        'bist': ['borsa istanbul', 'istanbul stock exchange'],
    }

    direct_def = lookup_term(question)
    if direct_def:
        return jsonify({'response': direct_def, 'conversations': []})

    # 2) Semantic nearest-term fallback from FAISS (works even without exact name)
    def definition_from_content(content: str) -> str | None:
        parts = (content or '').split(':', 1)
        if len(parts) == 2 and parts[1].strip():
            return parts[1].strip()
        return None

    def semantic_nearest_definition(q: str) -> str | None:
        qn = normalize_tr(q)
        # simple token similarity to guard against wrong hops
        def token_similarity(a: str, b: str) -> float:
            ta = set(t for t in normalize_tr(a).split() if len(t) >= 2)
            tb = set(t for t in normalize_tr(b).split() if len(t) >= 2)
            if not ta or not tb:
                return 0.0
            inter = len(ta & tb)
            union = len(ta | tb)
            return inter / union if union else 0.0

        # augment with synonyms to guide similarity
        extra = []
        for key, vals in SYNONYMS.items():
            if key in qn:
                extra.extend(vals)
        augmented = q if not extra else f"{q} (" + ', '.join(extra) + ")"
        try:
            hits = vector_store.similarity_search_with_score(augmented, k=1)
        except Exception:
            try:
                hits = vector_store.similarity_search_with_score(q, k=1)
            except Exception:
                hits = []
        if hits:
            doc, _score = hits[0]
            d = definition_from_content(getattr(doc, 'page_content', '') or '')
            # Accept only if the retrieved term is textually close to the asked term
            cand = extract_term_from_question(q) or q
            doc_term = str(getattr(doc, 'metadata', {}).get('term', '')).strip()
            base = doc_term.split('(')[0].strip()
            aliases = []
            m = re.search(r"\((.*?)\)", doc_term)
            if m:
                aliases = [al.strip() for al in re.split(r"[,/|;]", m.group(1)) if al.strip()]
            sims = [
                token_similarity(cand, doc_term),
                token_similarity(cand, base),
                *[token_similarity(cand, al) for al in aliases]
            ]
            if d and max(sims or [0.0]) >= 0.4:
                return d
        return None

    nearest_def = semantic_nearest_definition(question)
    if nearest_def:
        return jsonify({'response': nearest_def, 'conversations': []})

    # If no exact match, try retrieval with synonyms-augmented query
    try:
        qn = normalize_tr(question)
        extra = []
        for key, vals in SYNONYMS.items():
            if key in qn:
                extra.extend(vals)
        augmented_question = question if not extra else f"{question} (" + ', '.join(extra) + ")"
        retrieved = retriever.get_relevant_documents(augmented_question)
    except Exception:
        retrieved = []

    result = None
    source_documents = []

    def extract_definition_from_doc(doc, question):
        term = str(doc.metadata.get('term','')).strip()
        if not term:
            return None
        if term.lower() in question.lower():
            content = doc.page_content or ''
            parts = content.split(':',1)
            if len(parts)==2 and parts[1].strip():
                return parts[1].strip()
        return None

    for d in retrieved:
        defn = extract_definition_from_doc(d, question)
        if defn:
            answer = defn
            source_documents = [d]
            result = {'question': question, 'chat_history': chat_history, 'answer': answer, 'source_documents': source_documents}
            break

    if result is None:
        try:
            result = qa_chain({'question': question, 'chat_history': chat_history})
            source_documents = result.get('source_documents', [])
        except Exception as e:
            # As a last resort, provide a graceful fallback response
            fallback_msg = (
                "Bu konu kapsamım dışında kalıyor olabilir ya da veri setimde yer almıyor. "
                "Finans terimleriyle ilgili sorular sorabilirsiniz (örn: 'Likidite nedir?', 'Bileşik faiz nasıl hesaplanır?')."
            )
            return jsonify({'response': fallback_msg, 'conversations': []})

    # fallback LLM if needed (kept simple)
    answer = (result or {}).get('answer')
    if not answer or not str(answer).strip():
        answer = (
            "Bu konuda kesin bir bilgi bulamadım. Veri setimde bulunan finans terimlerine odaklanabilirim. "
            "Örnek: 'Likidite nedir?', 'F/K oranı neyi ifade eder?', 'Repo nedir?'."
        )
    # Return shape compatible with frontend template JS (expects 'response' and optionally 'conversations')
    return jsonify({'response': answer, 'conversations': []})


@app.route('/new_chat', methods=['POST'])
def new_chat():
    # Minimal endpoint to satisfy frontend - real implementation would create session etc.
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)
