"""Shared text preprocessing utilities.

Used across TF-IDF, BM25, and semantic search so query/doc processing stays consistent.
Documents are assumed sudah di-stem pada pipeline preprocessing sebelumnya; jangan re-stem dokumen lagi di sini.
"""

import re
from typing import List, Tuple

import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


def init_sastrawi() -> Tuple[object, List[str]]:
	"""Create stemmer and stopword list once."""

	stemmer = StemmerFactory().create_stemmer()
	stopwords = StopWordRemoverFactory().get_stop_words()
	return stemmer, stopwords


def clean_text_advanced(text: str) -> str:
	"""Aggressive cleaning; no re-stemming; remove digits/punct/urls/etc."""

	if pd.isna(text) or text == "":
		return ""

	text = str(text).lower()
	text = re.sub(r"http\S+|www\.\S+", "", text)
	text = re.sub(r"\S+@\S+", "", text)
	text = re.sub(r"\d+px", "", text)
	text = re.sub(r"padding|margin|font|vertical|align", "", text)
	text = re.sub(r"\b\w*\d+\w*\b", " ", text)
	text = re.sub(r"\b\d+\b", "", text)
	text = re.sub(r"\d{1,2}:\d{2}", "", text)
	text = re.sub(r"wib|wit|wita", "", text, flags=re.IGNORECASE)
	text = re.sub(r"[^\w\s]", " ", text)
	text = re.sub(r"\s+", " ", text).strip()
	return text


def preprocess_query(query: str, stemmer=None, stopwords: List[str] = None) -> str:
	"""Preprocess user query to align with pre-stemmed documents."""

	if pd.isna(query) or query == "":
		return ""

	# lazy init if not provided
	if stemmer is None or stopwords is None:
		stemmer, stopwords = init_sastrawi()

	text = str(query).lower()
	text = re.sub(r"http\S+|www\.\S+", "", text)
	text = re.sub(r"\S+@\S+", "", text)
	text = re.sub(r"@\w+|#\w+", "", text)
	text = re.sub(r"\d+px", "", text)
	text = re.sub(r"padding|margin|font|vertical|align", "", text)
	text = re.sub(r"\b\w*\d+\w*\b", " ", text)
	text = re.sub(r"\b\d+\b", "", text)
	text = re.sub(r"\d{1,2}:\d{2}", "", text)
	text = re.sub(r"wib|wit|wita", "", text, flags=re.IGNORECASE)
	text = re.sub(r"[^\w\s]", " ", text)
	text = re.sub(r"\s+", " ", text).strip()

	tokens = text.split()
	tokens = [t for t in tokens if len(t) > 1 and not any(c.isdigit() for c in t)]
	tokens = [t for t in tokens if t not in stopwords]
	tokens = [stemmer.stem(t) for t in tokens]
	return " ".join(tokens)


__all__ = ["init_sastrawi", "clean_text_advanced", "preprocess_query"]
