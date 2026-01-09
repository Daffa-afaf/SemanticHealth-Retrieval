"""
Script to generate semantic search artifacts using sentence-transformers.
This creates embeddings for all documents in the corpus.

Usage:
    python generate_semantic_artifacts.py

Requirements:
    pip install sentence-transformers
"""
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# Check if sentence-transformers is installed
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("sentence-transformers not installed!")
    print("Install with: pip install sentence-transformers")
    exit(1)

# Paths
ARTIFACTS_DIR = Path("./artifacts")
SEMANTIC_DIR = ARTIFACTS_DIR / "semantic"
SEMANTIC_DIR.mkdir(parents=True, exist_ok=True)

# You can change this to any multilingual model
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

print(f"Loading semantic model: {MODEL_NAME}")
print("   This may take a few minutes on first run...")
model = SentenceTransformer(MODEL_NAME)

# Load corpus from one of the existing artifacts
print("\nLoading corpus data...")
tfidf_meta_path = ARTIFACTS_DIR / "tfidf" / "corpus_meta.pkl"
bm25_meta_path = ARTIFACTS_DIR / "bm25" / "corpus_meta.pkl"

if tfidf_meta_path.exists():
    df_meta = pd.read_pickle(tfidf_meta_path)
    print(f"   Loaded {len(df_meta)} documents from TF-IDF artifacts")
elif bm25_meta_path.exists():
    df_meta = pd.read_pickle(bm25_meta_path)
    print(f"   Loaded {len(df_meta)} documents from BM25 artifacts")
else:
    print("No corpus metadata found in artifacts/tfidf or artifacts/bm25!")
    print("   Please run TF-IDF or BM25 notebook first.")
    exit(1)

# Prepare texts for embedding
print("\nPreparing texts for embedding...")
# Combine title and answer for richer semantic representation
texts = []
for _, row in df_meta.iterrows():
    title = str(row.get('title', ''))
    answer = str(row.get('answer', ''))
    combined = f"{title} {answer}"
    texts.append(combined)

print(f"   {len(texts)} texts prepared")

# Generate embeddings
print("\nGenerating embeddings (this may take a while)...")
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    batch_size=32,
    convert_to_numpy=True
)

print(f"   Embeddings shape: {embeddings.shape}")

# Save artifacts
print("\nSaving semantic artifacts...")

# Save embeddings
np.save(SEMANTIC_DIR / "embeddings.npy", embeddings)
print(f"   Saved embeddings to {SEMANTIC_DIR / 'embeddings.npy'}")

# Save corpus metadata
df_meta.to_pickle(SEMANTIC_DIR / "corpus_meta.pkl")
print(f"   Saved corpus metadata to {SEMANTIC_DIR / 'corpus_meta.pkl'}")

# Optionally save the model locally (for offline use)
save_model = input("\nSave model locally for offline use? (y/n): ").strip().lower()
if save_model == 'y':
    model_path = SEMANTIC_DIR / "model"
    model.save(str(model_path))
    print(f"   Saved model to {model_path}")
else:
    print("   Model will be downloaded from Hugging Face on app startup")

# Create a summary file
summary = {
    "model_name": MODEL_NAME,
    "n_docs": len(df_meta),
    "embedding_dim": embeddings.shape[1],
    "corpus_source": "tfidf" if tfidf_meta_path.exists() else "bm25"
}

with open(SEMANTIC_DIR / "semantic_config.json", "w") as f:
    import json
    json.dump(summary, f, indent=2)

print(f"   Saved config to {SEMANTIC_DIR / 'semantic_config.json'}")

print("\nSemantic search artifacts generated successfully!")
print(f"\nArtifacts saved in: {SEMANTIC_DIR}")
print("\nYou can now run the Streamlit app with semantic search enabled!")
print("   Run: streamlit run app.py")
