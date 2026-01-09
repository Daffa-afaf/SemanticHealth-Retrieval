# 🩺 SemanticHealth Retrieval

<div align="center">

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.32.0-FF4B4B.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Sistem Information Retrieval untuk Konsultasi Kesehatan Indonesia**  
*Membandingkan performa TF-IDF, BM25, dan Semantic Search*

</div>

---

## 📋 Deskripsi Project

**SemanticHealth Retrieval** adalah sistem pencarian dokumen kesehatan berbasis AI yang membandingkan tiga metode Information Retrieval:

- **TF-IDF** (Term Frequency-Inverse Document Frequency) - Pendekatan statistik klasik
- **BM25** (Best Matching 25) - Probabilistic ranking function
- **Semantic Search** - Deep learning dengan sentence transformers

Project ini menggunakan dataset **Indo Online Health Consultation Multilabel** dengan **360,513 dokumen** pertanyaan-jawaban kesehatan dalam Bahasa Indonesia.

### 🎯 Tujuan Utama

1. Mengimplementasikan dan membandingkan 3 metode IR pada domain kesehatan
2. Menganalisis kelebihan dan kekurangan masing-masing metode
3. Menyediakan aplikasi interaktif untuk eksplorasi dan perbandingan hasil
4. Mengukur performa dengan metrik evaluasi standar (Precision, Recall, NDCG, MRR)

---

## 🚀 Fitur Utama

✨ **3 Model IR Terintegrasi**
- TF-IDF dengan preprocessing Sastrawi
- BM25 dengan tuning parameter optimal
- Semantic Search menggunakan multilingual sentence transformers

📊 **Mode Perbandingan Side-by-Side**
- Bandingkan hasil dari ketiga model secara langsung
- Visualisasi score distribution
- Analisis overlap dokumen

🔍 **Pencarian Real-time**
- Query dalam bahasa Indonesia natural
- Top-K hasil dengan ranking score
- Metadata lengkap (topic, year)

📈 **Evaluasi Komprehensif**
- Precision@K, Recall@K
- NDCG (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
- Visualisasi perbandingan metrik

---

## 🏗️ Arsitektur System

```
SemanticHealth-Retrieval/
├── app.py                          # Streamlit web application
├── generate_semantic_artifacts.py  # Script untuk generate embeddings
├── src/
│   └── preprocessing.py            # Text preprocessing utilities
├── notebook/
│   ├── 01_data_exploration.ipynb   # Eksplorasi dataset
│   ├── 02_preprocessing.ipynb      # Preprocessing & cleaning
│   ├── 03_tfidf_model.ipynb        # Implementasi TF-IDF
│   ├── 04_bm25_model.ipynb         # Implementasi BM25
│   ├── 05_semantic_search.ipynb    # Semantic search dengan transformers
│   └── 06_evaluation.ipynb         # Evaluasi & perbandingan model
├── artifacts/
│   ├── tfidf/                      # TF-IDF artifacts (vectorizer, matrix)
│   ├── bm25/                       # BM25 artifacts (model, stopwords)
│   └── semantic/                   # Semantic artifacts (embeddings, config)
└── data/
    ├── raw/                        # Dataset original
    ├── processed/                  # Dataset terproses
    └── visualisasi/                # Output visualisasi
```

---

## 📊 Dataset

**Source**: Indo Online Health Consultation - Multilabel

### Karakteristik Dataset

| Atribut | Detail |
|---------|--------|
| **Total Dokumen** | 360,513 |
| **Bahasa** | Indonesia |
| **Domain** | Kesehatan & Konsultasi Medis |
| **Format** | CSV (title, question, answer, topic, year) |
| **Periode** | 2016-2023 |
| **Topics** | 15+ kategori (penyakit, gejala, pengobatan, dll) |

### Sample Data

```
Title: Demam tinggi pada anak
Question: Anak saya demam 39°C sejak kemarin, apa yang harus dilakukan?
Answer: Untuk menangani demam tinggi pada anak, berikan kompres hangat...
Topic: pediatri
Year: 2023
```

---

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.9+** - Programming language
- **Streamlit** - Web application framework
- **scikit-learn** - TF-IDF implementation
- **rank-bm25** - BM25 implementation
- **sentence-transformers** - Semantic search with BERT

### NLP & ML Libraries
- **Sastrawi** - Indonesian stemming & stopwords
- **PyTorch** - Deep learning backend
- **NumPy & Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Visualization

### Model
- **paraphrase-multilingual-MiniLM-L12-v2** - Multilingual sentence transformer (support Indonesian)

---

## 🔧 Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/Daffa-afaf/SemanticHealth-Retrieval.git
cd SemanticHealth-Retrieval
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Letakkan file dataset di `data/raw/Indo-Online Health Consultation-Multilabel-Raw.csv`

### 4. Generate Artifacts

Jalankan notebook secara berurutan:

```bash
# 1. Preprocessing
jupyter notebook notebook/02_preprocessing.ipynb

# 2. Generate TF-IDF artifacts
jupyter notebook notebook/03_tfidf_model.ipynb

# 3. Generate BM25 artifacts
jupyter notebook notebook/04_bm25_model.ipynb

# 4. Generate Semantic artifacts
python generate_semantic_artifacts.py
```

### 5. Run Application

```bash
streamlit run app.py
```

Aplikasi akan terbuka di `http://localhost:8501`

---

## 📖 Dokumentasi

### Preprocessing Pipeline

1. **Text Cleaning**
   - Lowercase conversion
   - Remove special characters & numbers
   - Remove extra whitespaces

2. **Tokenization**
   - Word-level tokenization

3. **Stopword Removal**
   - Indonesian stopwords (Sastrawi)
   - Custom medical stopwords

4. **Stemming**
   - Sastrawi Indonesian stemmer
   - Reduce inflected words to root form

### Model Implementation

#### TF-IDF
- **Vectorizer**: scikit-learn TfidfVectorizer
- **Parameters**: max_features=10000, ngram_range=(1,2)
- **Similarity**: Cosine similarity

#### BM25
- **Implementation**: rank-bm25 library
- **Parameters**: k1=1.5, b=0.75 (tuned)
- **Tokenization**: Preprocessed tokens

#### Semantic Search
- **Model**: paraphrase-multilingual-MiniLM-L12-v2
- **Embedding Dim**: 384
- **Similarity**: Cosine similarity on embeddings

---

## 📊 Hasil Evaluasi

### Analisis Kelebihan & Kekurangan

#### ✅ Semantic Search
- **Kelebihan**: Memahami konteks semantik, baik untuk paraphrase
- **Kekurangan**: Lebih lambat, membutuhkan GPU untuk inference cepat

#### ✅ TF-IDF
- **Kelebihan**: Cepat, ringan, mudah interpretasi
- **Kekurangan**: Tidak memahami semantik, keyword-based

#### ✅ BM25
- **Kelebihan**: Balance antara speed dan accuracy
- **Kekurangan**: Masih keyword-based, butuh tuning parameter

---

## 🚢 Deployment ke Streamlit Cloud

### 1. Push ke GitHub

```bash
git init
git add .
git commit -m "Initial commit: SemanticHealth Retrieval System"
git branch -M main
git remote add origin https://github.com/Daffa-afaf/SemanticHealth-Retrieval.git
git push -u origin main
```

### 2. Deploy di Streamlit Cloud

1. Buka [share.streamlit.io](https://share.streamlit.io)
2. Login dengan GitHub
3. Click "New app"
4. Pilih repository: `Daffa-afaf/SemanticHealth-Retrieval`
5. Main file path: `app.py`
6. Click "Deploy"

**Note**: Karena artifacts cukup besar, pertimbangkan untuk:
- Gunakan Git LFS untuk file besar
- Atau generate artifacts on-the-fly saat deployment

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📝 License

Distributed under the MIT License.

---

## 👨‍💻 Author

**Daffa Afaf**

- GitHub: [@Daffa-afaf](https://github.com/Daffa-afaf)
- Project Link: [SemanticHealth-Retrieval](https://github.com/Daffa-afaf/SemanticHealth-Retrieval)

---

## 🙏 Acknowledgments

- Dataset Indo Online Health Consultation
- Sastrawi Indonesian NLP library
- Sentence Transformers by UKPLab
- Streamlit framework

---

## 📚 References

1. Robertson, S. E., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond.
2. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
3. Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval.

---

<div align="center">

⭐ **Star this repo if you find it useful!** ⭐

Made with ❤️ for Indonesian Healthcare Information Retrieval

</div>
