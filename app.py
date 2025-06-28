import streamlit as st
import pandas as pd
import joblib
import re

from utils import (
    load_data,
    load_tfidf_vectorizer,
    load_cosine_matrix,
    preprocess_text,
    get_top_k_from_cosine,
    filter_and_sort_recommendations,
)

# ---------------------
# Fungsi remove symbol
# ---------------------
def remove_symbol(text):
    return re.sub(r'[^\w\s]', '', text)

# ---------------------
# Load semua file
# ---------------------
@st.cache_resource
def load_all():
    df = load_data('./data/data_resto_with_jarak.csv')
    tfidf = load_tfidf_vectorizer('./model/tfidf.pkl')
    cosine_df = load_cosine_matrix('./model/cosinesimilarity.pkl')
    cosine_df.columns = cosine_df.columns.str.strip().str.lower()
    cosine_df.index = cosine_df.index.str.strip().str.lower()

    return df, tfidf, cosine_df

df, tfidf, cosine_df = load_all()

# ---------------------
# UI
# ---------------------
st.title("🍜 Sistem Rekomendasi Tempat Makan")

query = st.text_input("Masukkan kategori makanan (contoh: bakso, sate, penyetan)", value="bakso")
base_choice = st.selectbox("Pilih kampus terdekat sebagai acuan jarak", options=["", "A", "B", "C"])
k = st.slider("Jumlah rekomendasi", min_value=3, max_value=20, value=5)

if st.button("Cari Rekomendasi"):
    with st.spinner("🔍 Mencari rekomendasi..."):
        query_cleaned = preprocess_text(query, remove_symbol)

        matched_cols = [col for col in cosine_df.columns if query_cleaned in col]
        if not matched_cols:
                st.warning(f"❗ Tidak ditemukan hasil yang cocok untuk '{query}'. Coba kata lain.")
        else:
            selected_col = matched_cols[0]
            top_k_index, top_k_scores = get_top_k_from_cosine(selected_col, cosine_df, k)
            if not top_k_index:
                st.warning("❗ Tidak ditemukan rekomendasi.")
            else:
                rekomendasi_df = filter_and_sort_recommendations(
                    df, top_k_index, top_k_scores, base_choice, remove_symbol_func=remove_symbol
                )

                st.success(f"✅ Menampilkan {len(rekomendasi_df)} hasil teratas:")
                for _, row in rekomendasi_df.iterrows():
                    jarak_kampus = row.get(f'Jarak_Kampus{base_choice.upper()}', 'N/A') if base_choice else '-'
                    st.markdown(f"""
                    ---  
                    ### **{row['title']}**
                    📍 *{row['address']}*  
                    ⭐ **Rating:** `{row['totalScore']}`  
                    🧭 **Jarak dari Kampus {base_choice.upper() if base_choice else '-' }:** `{round(jarak_kampus,2)} km`   
                    """)


