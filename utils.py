import re
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

def remove_symbol(text):
    """
    Hapus simbol dan karakter non-alfabetik dari teks.
    """
    return re.sub(r'[^\w\s]', '', str(text))

def preprocess_text(text, remove_func):
    """
    Proses teks: hapus simbol, ubah ke huruf kecil, hapus spasi ekstra.
    """
    return remove_func(text.strip().lower())


def load_data(path='./data/data_resto_with_jarak.csv', remove_func=remove_symbol):
    """
    Load dan bersihkan data tempat makan.
    """
    df = pd.read_csv(path)
    df['combined'] = (df['categories'] + ' ' + df['title']).apply(remove_func).str.strip().str.lower()
    return df

def load_tfidf_vectorizer(path='./model/tfidf.pkl'):
    return joblib.load(path)



def get_top_k_from_keyword(query, df, vectorizer, tfidf_matrix, k=5, remove_symbol_func=remove_symbol):
    """
    Hitung kemiripan antara query dan semua item dengan TF-IDF + cosine similarity.
    """
    query_clean = preprocess_text(query, remove_symbol_func)
    query_vec = vectorizer.transform([query_clean])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_k_index = similarities.argsort()[::-1][:k]
    top_k_scores = similarities[top_k_index]
    results = df.iloc[top_k_index].copy()
    results['similarity'] = top_k_scores
    return results
