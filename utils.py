import pandas as pd
import joblib

def load_data(path='./data/data_resto_with_jarak.csv'):
    return pd.read_csv(path)

def load_tfidf_vectorizer(path='./model/tfidf.pkl'):
    return joblib.load(path)

def load_cosine_matrix(path='./model/cosinesimilarity.pkl'):
    return joblib.load(path)

def preprocess_text(text, remove_symbol_func):
    return remove_symbol_func(text.strip().lower())

def get_top_k_from_cosine(query_combined_str, cosine_df, k=5):
    if query_combined_str not in cosine_df.columns:
        return [], []

    similarities = cosine_df[query_combined_str].drop(labels=[query_combined_str], errors='ignore')
    top_k = similarities.nlargest(k)
    return top_k.index.tolist(), top_k.values

def filter_and_sort_recommendations(df, top_k_index, top_k_scores, base_column=None, remove_symbol_func=None):

    df = df.copy()

    # Preprocess kolom 'combined' agar cocok dengan cosine_df index
    df['categories_cleaned'] = df['categories'].apply(remove_symbol_func)
    df['title_cleaned'] = df['title'].apply(remove_symbol_func)
    df['combined'] = (df['categories_cleaned'] + ' ' + df['title_cleaned']).str.strip().str.lower()

    # Filter hanya data yang muncul di top_k index
    df = df[df['combined'].isin(top_k_index)]

    # Tambahkan skor similarity dari top_k_scores
    df['similarity'] = df['combined'].map(dict(zip(top_k_index, top_k_scores)))

    sort_cols = []
    ascending = []

    if base_column:
        jarak_col = f"Jarak_Kampus{base_column.upper()}"
        if jarak_col in df.columns:
            # Bulatkan jarak ke 2 angka di belakang koma
            df[jarak_col] = df[jarak_col].round(2)
            sort_cols.append(jarak_col)
            ascending.append(True)

    # Tambah sorting berdasarkan totalScore dan similarity
    sort_cols += ['totalScore', 'similarity']
    ascending += [False, False]

    return df.sort_values(by=sort_cols, ascending=ascending)
