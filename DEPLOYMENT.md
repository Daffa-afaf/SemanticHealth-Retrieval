# Deployment Guide untuk Streamlit Cloud

## 📋 Prerequisites

1. Akun GitHub (sudah ada)
2. Akun Streamlit Cloud (gratis di share.streamlit.io)
3. Repository sudah di-push ke GitHub

---

## 🚀 Langkah Deployment

### 1. Push Repository ke GitHub ✅

Repository sudah di-push ke: https://github.com/Daffa-afaf/SemanticHealth-Retrieval

### 2. Login ke Streamlit Cloud

1. Buka [share.streamlit.io](https://share.streamlit.io)
2. Klik "Sign in with GitHub"
3. Authorize Streamlit Cloud untuk akses repository

### 3. Deploy App

1. Klik tombol "New app"
2. Isi form deployment:
   - **Repository**: `Daffa-afaf/SemanticHealth-Retrieval`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: `semantichealth-retrieval` (atau custom)

3. Klik "Deploy!"

### 4. Tunggu Build Process

Streamlit akan:
- Install dependencies dari `requirements.txt`
- Build environment
- Start aplikasi

Proses ini biasanya memakan waktu 5-10 menit.

---

## ⚠️ Catatan Penting

### File Besar (Artifacts)

Karena artifacts (embeddings, pkl files) **tidak di-commit** ke GitHub (file terlalu besar), ada 2 opsi:

#### **Opsi 1: Generate on-the-fly (Recommended)**

Tambahkan script startup di app yang generate artifacts jika belum ada.

Edit `app.py`, tambahkan di awal:

```python
def check_and_generate_artifacts():
    """Generate artifacts if not exists"""
    if not (ARTIFACTS_DIR / "tfidf").exists():
        st.warning("Generating TF-IDF artifacts...")
        # Run notebook 03_tfidf_model.ipynb
        
    if not (ARTIFACTS_DIR / "bm25").exists():
        st.warning("Generating BM25 artifacts...")
        # Run notebook 04_bm25_model.ipynb
```

#### **Opsi 2: Upload ke Cloud Storage**

Upload artifacts ke:
- Google Drive
- Dropbox
- AWS S3
- GitHub Releases

Kemudian download saat deployment.

#### **Opsi 3: Git LFS (Large File Storage)**

```bash
git lfs install
git lfs track "*.pkl"
git lfs track "*.npy"
git add .gitattributes
git commit -m "Add Git LFS"
git push
```

---

## 🔧 Troubleshooting

### Error: Module not found

Pastikan `requirements.txt` lengkap:

```bash
pip freeze > requirements.txt
```

### Error: Memory Limit

Streamlit Cloud free tier = 1GB RAM. Jika over:
- Reduce dataset size
- Use smaller model
- Upgrade to paid tier

### Error: Build timeout

Build timeout = 10 menit. Jika over:
- Reduce dependencies
- Pre-build artifacts
- Contact Streamlit support

---

## 📊 Post-Deployment

### 1. Test Aplikasi

Buka URL deployment Anda, test semua fitur:
- ✅ Search dengan TF-IDF
- ✅ Search dengan BM25
- ✅ Search dengan Semantic
- ✅ Comparison mode

### 2. Monitor Logs

Di dashboard Streamlit Cloud:
- Lihat logs real-time
- Monitor error
- Track usage

### 3. Update App

Setiap kali push ke GitHub `main` branch, app otomatis re-deploy.

```bash
git add .
git commit -m "Update feature X"
git push
```

---

## 🎉 URL Final

Setelah deployment, aplikasi Anda akan tersedia di:

```
https://semantichealth-retrieval-[username].streamlit.app
```

atau custom domain jika setup.

---

## 📝 Tips Optimization

1. **Cache Data Loading**
   ```python
   @st.cache_data
   def load_dataset():
       return pd.read_csv('data.csv')
   ```

2. **Cache Model Loading**
   ```python
   @st.cache_resource
   def load_model():
       return SentenceTransformer('model-name')
   ```

3. **Lazy Loading**
   Load artifacts hanya saat dibutuhkan, bukan di awal.

4. **Compress Data**
   Gunakan compressed format (parquet, feather) instead of CSV.

---

## 🆘 Support

Jika ada masalah:
1. Check [Streamlit Community Forum](https://discuss.streamlit.io)
2. Check [Streamlit Documentation](https://docs.streamlit.io)
3. Create GitHub Issue di repository
