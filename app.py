import streamlit as st
import pandas as pd
import numpy as np
import faiss
import ast
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv
import os
import re
import html
from sklearn.metrics.pairwise import cosine_similarity
import streamlit.components.v1 as components

# ===========================
# Load ENV & Konfigurasi
# ===========================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

gen_model = genai.GenerativeModel("gemini-2.0-flash")
embed_model_id = "models/embedding-001"

st.set_page_config(
    page_title="MahirCegahJudi",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# Load CSS
# ===========================
def load_local_css(path):
    with open(path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_local_css("assets/styles.css")



with open("assets/sidebar_toggle.js") as f:
    js_code = f.read()
components.html(f"<script>{js_code}</script>", height=0)

# ===========================
# Load Data & Index
# ===========================
@st.cache_resource
def load_data_and_index():
    df = pd.read_csv("data/berita_judi.csv")
    df["embedding"] = df["embedding"].apply(ast.literal_eval)
    index = faiss.read_index("data/faiss.index")
    return df, index

df, index = load_data_and_index()

# ===========================
# Embedding
# ===========================
def get_gemini_embedding(text):
    response = genai.embed_content(
        model=embed_model_id,
        content=text,
        task_type="retrieval_query"
    )
    return np.array(response["embedding"], dtype='float32')

# ===========================
# Search FAISS
# ===========================
def search_faiss(query, top_k=5):
    query_vector = get_gemini_embedding(query).reshape(1, -1).astype("float32")
    _, I = index.search(query_vector, top_k * 5)  # Ambil banyak dulu untuk variasi

    candidates = df.iloc[I[0]].drop_duplicates(subset=["title", "link"]).copy()
    embeddings = np.vstack(candidates["embedding"].values)

    # Diversifikasi hasil: hindari yang mirip satu sama lain
    selected = []
    selected_idx = []
    sim_matrix = cosine_similarity(embeddings)

    for i in range(len(candidates)):
        if len(selected_idx) >= top_k:
            break
        # Hindari menambahkan dokumen yang terlalu mirip dengan yang sudah dipilih
        if all(sim_matrix[i][j] < 0.92 for j in selected_idx):
            selected_idx.append(i)

    # Fallback: jika belum cukup hasil
    while len(selected_idx) < top_k and len(selected_idx) < len(candidates):
        if len(set(selected_idx)) < len(candidates):
            selected_idx.append(len(selected_idx))

    return candidates.iloc[selected_idx]
# ===========================
# Sanitize Gemini Output
# ===========================
def sanitize_html(text):
    text = html.unescape(text)
    return re.sub(r"<[^>]+>", "", text).strip()

# ===========================
# Generate Ringkasan
# ===========================
def generate_answer_summary(question, df_results):
    combined_answers = "\n".join([
        f"- {row['answer_1']}\n- {row['answer_2']}" for _, row in df_results.iterrows()
    ])
    prompt = f"""
Saya memiliki pertanyaan:
\"{question}\"

Berikut beberapa jawaban dari sumber terpercaya:
{combined_answers}

Buatlah jawaban yang ringkas dan informatif berdasarkan informasi di atas. Awali dengan: ✅ Jawaban Ringkasan:
"""
    response = gen_model.generate_content(prompt)
    return sanitize_html(response.text)

# ===========================
# Session State
# ===========================
if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = []

if "current_thread_index" not in st.session_state:
    st.session_state.current_thread_index = None

if "new_thread_pending" not in st.session_state:
    st.session_state.new_thread_pending = False

# ===========================
# Sidebar
# ===========================
with st.sidebar:
    st.image("assets/chatbot.png", width=70)
    st.title("📁 Riwayat Topik")

    if st.session_state.chat_threads:
        titles = [t["title"] for t in st.session_state.chat_threads]

        # Cek apakah sudah ada current_thread_index
        if st.session_state.current_thread_index is not None:
           selected_index = len(titles) - 1  # Pilih thread terbaru secara default
        else:
             selected_index = st.session_state.current_thread_index
        
        selected = st.radio("Pilih topik:", list(range(len(titles))),
                        format_func=lambda i: titles[i],
                        index=selected_index)

        st.session_state.current_thread_index = selected

        

    if st.button("➕ Mulai Topik Baru"):
        st.session_state.current_thread_index = None
        st.session_state.new_thread_pending = True
        st.session_state.selected_thread_index = None  # ⬅️ reset pilihan radio

st.markdown(
    '''
    <meta name="dicoding:email" content="nugrahadaffa568@gmail.com">
    ''',
    unsafe_allow_html=True
)


# ===========================
# Header
# ===========================


# Header tanpa HTML
st.title("🛡️ MahirCegahJudi")
st.caption("💬 Asisten edukasi dan pencegahan judi online berbasis AI.")
st.markdown("Silakan ajukan pertanyaan seputar bahaya atau dampak judi online:")

# ===========================
# Tampilkan Chat
# ===========================
if st.session_state.current_thread_index is not None:
    thread = st.session_state.chat_threads[st.session_state.current_thread_index]
    for msg in thread["messages"]:
        # Bubble Pertanyaan
        st.markdown(f"""
        <div class="chat-message user-msg">
            <div class="chat-bubble chat-question"> {msg["question"]}</div>
        </div>
        """, unsafe_allow_html=True)

        # Bubble Jawaban
        st.markdown(f"""
        <div class="chat-message assistant-msg">
            <div class="chat-bubble chat-answer">
                {msg["answer"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Referensi (Markdown - bisa dicopy)
        st.markdown("🔗 **Sumber Referensi:**")
        for ref in msg.get("sources", []):
            st.markdown(f"- **{ref['title']}**\n  {ref['link']}\n  _{ref['sumber']}_\n")

# ===========================
# Form Input
# ===========================
with st.form("ask_form", clear_on_submit=True):
    query = st.text_input("✍️ Pertanyaan Anda:", placeholder="Contoh: Apa dampak judi online bagi pelajar?")
    submitted = st.form_submit_button("Kirim")

if submitted and query:
    with st.spinner("🔍 Mencari jawaban..."):
        results = search_faiss(query)
        final_answer = generate_answer_summary(query, results)
        sources = results[["title", "link", "sumber"]].to_dict(orient="records")

    new_msg = {
        "question": query,
        "answer": final_answer,
        "sources": sources
    }

    # ⬇️ Tambahkan ke thread saat ini ATAU buat baru
    if st.session_state.new_thread_pending or st.session_state.current_thread_index is None:
        st.session_state.chat_threads.append({
            "title": query,
            "messages": [new_msg]
        })
        st.session_state.current_thread_index = len(st.session_state.chat_threads) - 1
        st.session_state.new_thread_pending = False
    else:
        st.session_state.chat_threads[st.session_state.current_thread_index]["messages"].append(new_msg)

    st.rerun()

