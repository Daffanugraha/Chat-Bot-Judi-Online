<div align="center">

<img src="assets/chatbot.png" width="100" alt="MahirCegahJudi Logo"/>

# 🛡️ MahirCegahJudi

**Chatbot Edukasi & Pencegahan Judi Online Berbasis AI**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12.4-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/LLM-Gemini_2.0_Flash-4285F4?logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/Embedding-Gemini_embedding--001-34A853?logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/VectorDB-FAISS-00B4D8?logo=meta&logoColor=white" />
  <img src="https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/Styling-CSS3-1572B6?logo=css3&logoColor=white" />
  <img src="https://img.shields.io/badge/Script-JavaScript-F7DF1E?logo=javascript&logoColor=black" />
  <img src="https://img.shields.io/badge/License-MIT-22c55e?logo=opensourceinitiative&logoColor=white" />
</p>

> *"Satu pertanyaan bisa menyelamatkan seribu keputusan buruk."*

[🚀 Demo](#-cara-menjalankan) • [📖 Dokumentasi](#-cara-kerja-sistem) • [🤝 Kontribusi](#-kontribusi)

</div>

---

## 📌 Tentang Proyek

**MahirCegahJudi** adalah chatbot berbasis **RAG (Retrieval-Augmented Generation)** yang dirancang untuk memberikan edukasi dan informasi seputar **bahaya judi online** kepada masyarakat Indonesia. Chatbot ini menggabungkan kekuatan pencarian semantik **FAISS** dengan kecerdasan generatif **Google Gemini** untuk menghasilkan jawaban yang akurat, informatif, dan berbasis sumber terpercaya.

### Mengapa Proyek Ini Penting?

Judi online merupakan salah satu masalah sosial yang kian marak di Indonesia, menyasar berbagai kalangan mulai dari pelajar hingga pekerja dewasa. MahirCegahJudi hadir sebagai solusi edukatif — memberikan informasi berbasis data nyata, bukan opini.

---

## ✨ Fitur Utama

| Fitur | Deskripsi |
|---|---|
| 🔍 **Pencarian Semantik** | Query diubah menjadi vektor lalu dicari kesamaannya di database menggunakan FAISS |
| 🤖 **Generasi Jawaban AI** | Jawaban dirangkum secara cerdas oleh Gemini 2.0 Flash |
| 📰 **Berbasis Berita Nyata** | Data diambil dari artikel berita terpercaya seputar judi online |
| 🔀 **Diversifikasi Hasil** | Menghindari jawaban redundan dengan cosine similarity filtering |
| 💬 **Multi-Thread Chat** | Mendukung percakapan berlapis dengan riwayat topik di sidebar |
| 🔗 **Sumber Referensi** | Setiap jawaban dilengkapi tautan ke artikel sumber aslinya |
| 🎨 **UI Kustom** | Tampilan modern dengan CSS kustom dan animasi |

---

## 🏗️ Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PIPELINE RAG                                 │
│                                                                     │
│  USER QUERY                                                         │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────────────┐                                            │
│  │  Gemini Embedding   │  ← models/embedding-001                   │
│  │  (retrieval_query)  │                                            │
│  └──────────┬──────────┘                                            │
│             │  float32 vector                                        │
│             ▼                                                        │
│  ┌─────────────────────┐     ┌──────────────────────┐              │
│  │   FAISS Index       │────▶│  Top-K Candidates    │              │
│  │  (Vector Search)    │     │  (top_k × 5 buffer)  │              │
│  └─────────────────────┘     └──────────┬───────────┘              │
│                                          │                           │
│                                          ▼                           │
│                              ┌──────────────────────┐              │
│                              │  Cosine Similarity   │              │
│                              │  Diversification     │              │
│                              │  (threshold: 0.92)   │              │
│                              └──────────┬───────────┘              │
│                                          │  top_k diverse results   │
│                                          ▼                           │
│                              ┌──────────────────────┐              │
│                              │   Gemini 2.0 Flash   │              │
│                              │  Answer Generation   │              │
│                              └──────────┬───────────┘              │
│                                          │                           │
│                                          ▼                           │
│                              ✅ Jawaban Ringkas + Sumber Referensi  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Struktur Direktori

```
Chat-Bot-Judi-Online/
│
├── 📁 assets/
│   ├── chatbot.png          # Logo/ikon sidebar
│   ├── styles.css           # Kustom styling UI
│   └── sidebar_toggle.js    # Script JS untuk sidebar
│
├── 📁 data/
│   ├── berita_judi.csv      # Dataset berita + embedding + jawaban
│   └── faiss.index          # Index vektor FAISS (pre-built)
│
├── 📁 .devcontainer/        # Konfigurasi GitHub Codespaces
│
├── app.py                   # Aplikasi utama Streamlit
├── requirements.txt         # Dependensi Python
├── packages.txt             # Dependensi sistem
├── .gitignore
└── README.md
```

---

## 🗄️ Spesifikasi Dataset

Dataset utama tersimpan di `data/berita_judi.csv` dengan struktur kolom berikut:

| Kolom | Tipe | Deskripsi |
|---|---|---|
| `title` | `string` | Judul artikel berita |
| `link` | `string` | URL sumber artikel |
| `sumber` | `string` | Nama media/penerbit |
| `answer_1` | `string` | Ringkasan / jawaban pertama dari konten artikel |
| `answer_2` | `string` | Ringkasan / jawaban alternatif dari konten artikel |
| `embedding` | `list[float]` | Vektor embedding (768 dimensi, format string Python literal) |

### 📊 Spesifikasi Teknis Dataset

- **Format embedding**: String Python list yang diparse dengan `ast.literal_eval()`
- **Model embedding**: `models/embedding-001` (Google Gemini)
- **Dimensi vektor**: 768 dimensi
- **Index**: FAISS `IndexFlatL2` tersimpan di `data/faiss.index`
- **Deduplikasi**: Berdasarkan kombinasi `title + link`

---

## 🔄 Alur Pengumpulan & Pemrosesan Data

```
FASE 1 — PENGUMPULAN DATA
─────────────────────────
  Web Scraping / Pengumpulan Manual
        │
        ▼
  Artikel berita seputar judi online
  (judul, link, sumber, konten)

FASE 2 — PEMBUATAN JAWABAN (Q&A Pairs)
────────────────────────────────────────
  Konten artikel
        │
        ▼
  Ekstraksi poin-poin penting
        │
        ▼
  answer_1 → Ringkasan utama
  answer_2 → Perspektif / sudut pandang lain

FASE 3 — EMBEDDING GENERATION
───────────────────────────────
  answer_1 + answer_2 (atau konten penuh)
        │
        ▼
  Google Gemini Embedding API
  (models/embedding-001, task_type="retrieval_document")
        │
        ▼
  Vektor float32[768] per dokumen

FASE 4 — FAISS INDEX BUILDING
───────────────────────────────
  Semua vektor dikumpulkan → numpy array
        │
        ▼
  faiss.IndexFlatL2 (atau IndexIVFFlat untuk skala besar)
        │
        ▼
  faiss.write_index() → data/faiss.index

FASE 5 — PENYIMPANAN
──────────────────────
  Embedding disimpan sebagai string di CSV
  Index FAISS disimpan sebagai file biner .index
```

---

## ⚙️ Cara Kerja Sistem (Detail Teknis)

### 1. 🔐 Inisialisasi & Konfigurasi

```python
# Memuat API key dari .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Model Gemini
gen_model   = genai.GenerativeModel("gemini-2.0-flash")   # Untuk generasi
embed_model = "models/embedding-001"                        # Untuk embedding
```

### 2. 📦 Loading Data & Index (dengan Caching)

```python
@st.cache_resource
def load_data_and_index():
    df = pd.read_csv("data/berita_judi.csv")
    df["embedding"] = df["embedding"].apply(ast.literal_eval)
    index = faiss.read_index("data/faiss.index")
    return df, index
```

> `@st.cache_resource` memastikan data hanya dimuat **sekali** meskipun pengguna mengirim banyak pertanyaan — menghemat memori dan waktu.

### 3. 🔍 FAISS Search + Diversifikasi

```python
def search_faiss(query, top_k=5):
    # Embed query
    query_vector = get_gemini_embedding(query).reshape(1, -1)
    
    # Ambil top_k × 5 kandidat (buffer untuk diversifikasi)
    _, I = index.search(query_vector, top_k * 5)
    candidates = df.iloc[I[0]].drop_duplicates(subset=["title", "link"])
    
    # Hitung cosine similarity antar kandidat
    embeddings  = np.vstack(candidates["embedding"].values)
    sim_matrix  = cosine_similarity(embeddings)
    
    # Greedy diversification — hanya ambil dokumen yang "cukup berbeda"
    selected_idx = []
    for i in range(len(candidates)):
        if len(selected_idx) >= top_k:
            break
        if all(sim_matrix[i][j] < 0.92 for j in selected_idx):
            selected_idx.append(i)
    
    return candidates.iloc[selected_idx]
```

**Kenapa threshold 0.92?** Dokumen dengan kemiripan ≥ 0.92 dianggap memiliki informasi yang hampir identik, sehingga cukup satu saja yang diambil untuk menghindari jawaban redundan.

### 4. 🤖 Generasi Jawaban dengan Gemini

```python
def generate_answer_summary(question, df_results):
    combined_answers = "\n".join([
        f"- {row['answer_1']}\n- {row['answer_2']}"
        for _, row in df_results.iterrows()
    ])

    prompt = f"""
    Saya memiliki pertanyaan: "{question}"
    Berikut beberapa jawaban dari sumber terpercaya:
    {combined_answers}
    Buatlah jawaban yang ringkas dan informatif ...
    """
    response = gen_model.generate_content(prompt)
    return sanitize_html(response.text)
```

Pola ini adalah **RAG (Retrieval-Augmented Generation)**: model AI tidak menjawab dari "memorinya" sendiri, melainkan berdasarkan konteks dokumen nyata yang diambil dari database.

---

## 🧱 Tech Stack

| Komponen | Teknologi | Versi |
|---|---|---|
| **Frontend / UI** | Streamlit | ≥ 1.25 |
| **AI Model — Generasi** | Google Gemini 2.0 Flash | Latest |
| **AI Model — Embedding** | Google Gemini `embedding-001` | Latest |
| **Vector Search** | FAISS (Facebook AI) | CPU |
| **Data Processing** | Pandas, NumPy | ≥ 2.1 / ≥ 1.24 |
| **Similarity Filtering** | Scikit-learn | ≥ 1.2 |
| **Styling** | CSS Kustom + JavaScript | — |
| **Env Management** | python-dotenv | Latest |
| **Bahasa** | Python | ≥ 3.10 |

---

## 🚀 Cara Menjalankan

### Prasyarat

- Python 3.10+
- Google Gemini API Key → [Dapatkan di sini](https://aistudio.google.com/app/apikey)

### Langkah Instalasi

```bash
# 1. Clone repository
git clone https://github.com/Daffanugraha/Chat-Bot-Judi-Online.git
cd Chat-Bot-Judi-Online

# 2. Install dependensi
pip install -r requirements.txt

# 3. Buat file .env dan tambahkan API key
echo "GOOGLE_API_KEY=your_api_key_here" > .env

# 4. Jalankan aplikasi
streamlit run app.py
```

Buka browser dan akses **http://localhost:8501** 🎉

---

## 🖥️ Tampilan Antarmuka

```
┌────────────────────────────────────────────────────┐
│  📁 Riwayat Topik       │  🛡️ MahirCegahJudi       │
│  ─────────────────      │  ──────────────────────  │
│  • Dampak judi online   │                           │
│  • Cara melaporkan judi │  [Pertanyaan pengguna]    │
│  • Hukum judi online    │                           │
│                         │  ✅ Jawaban Ringkasan:    │
│  [➕ Mulai Topik Baru]   │  ...jawaban AI...         │
│                         │                           │
│                         │  🔗 Sumber Referensi:    │
│                         │  - Judul Artikel 1        │
│                         │  - Judul Artikel 2        │
│                         │                           │
│                         │  [✍️ Pertanyaan Anda...]  │
│                         │              [Kirim]      │
└────────────────────────────────────────────────────┘
```

---

## 📦 Dependensi Lengkap

```txt
faiss-cpu
numpy>=1.24
pandas>=2.1
protobuf>=3.20
python-dotenv
scikit-learn>=1.2
streamlit>=1.25
google-generativeai
```

---

## 🔧 Konfigurasi Lanjutan

### Mengganti Jumlah Hasil Pencarian

Di fungsi `search_faiss()`, ubah parameter `top_k`:

```python
results = search_faiss(query, top_k=5)  # default: 5 artikel
```

### Mengganti Threshold Diversifikasi

```python
# Lebih ketat (lebih beragam, tapi mungkin kurang relevan)
if all(sim_matrix[i][j] < 0.85 for j in selected_idx):

# Lebih longgar (lebih relevan, tapi bisa redundan)
if all(sim_matrix[i][j] < 0.95 for j in selected_idx):
```

### Menambah Data Baru

1. Tambahkan baris baru ke `data/berita_judi.csv` dengan kolom yang sesuai
2. Generate embedding menggunakan Gemini `embedding-001`
3. Rebuild FAISS index:

```python
import faiss, numpy as np, pandas as pd, ast

df = pd.read_csv("data/berita_judi.csv")
df["embedding"] = df["embedding"].apply(ast.literal_eval)
vectors = np.vstack(df["embedding"].values).astype("float32")

index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)
faiss.write_index(index, "data/faiss.index")
```

---

## 🤝 Kontribusi

Kontribusi selalu disambut! Berikut cara berkontribusi:

1. **Fork** repository ini
2. Buat branch baru: `git checkout -b fitur/nama-fitur`
3. Commit perubahan: `git commit -m "feat: tambah fitur XYZ"`
4. Push ke branch: `git push origin fitur/nama-fitur`
5. Buat **Pull Request**

### Ide Kontribusi

- 📊 Menambah lebih banyak data berita
- 🌐 Integrasi dengan live news scraper
- 🔎 Implementasi re-ranking (BM25 + semantic hybrid)
- 🧪 Unit testing untuk fungsi embedding & search
- 🌍 Dukungan multi-bahasa

---

## ⚠️ Disclaimer

Proyek ini dibuat **murni untuk tujuan edukasi**. Seluruh informasi yang disajikan chatbot bersumber dari artikel berita publik dan dimaksudkan untuk meningkatkan kesadaran masyarakat tentang bahaya judi online, bukan untuk tujuan komersial maupun promosi.

---

## 👤 Author

**Daffa Nugraha**

- GitHub: [@Daffanugraha](https://github.com/Daffanugraha)
- Email: nugrahadaffa568@gmail.com

---

</div>
