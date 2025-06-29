import streamlit as st
from utils import (
    load_data, load_tfidf_vectorizer,
    get_top_k_from_keyword, remove_symbol
)
import joblib

# Load data dan model
df = load_data()
vectorizer = load_tfidf_vectorizer()
tfidf_matrix = vectorizer.transform(df['combined'])


st.title("🍽️ Rekomendasi Tempat Makan Sekitar UNAIR")
st.subheader("Solusi saat bingung mau makan dimana yang dekat kampus")
query = st.text_input("Cari makanan berdasarkan kata kunci:")
base_choice = st.selectbox("Pilih Kampus", ['A', 'B', 'C'])
k = st.slider("Jumlah Rekomendasi", 1, 20, 5)

if st.button("🔍 Cari Rekomendasi"):
    with st.spinner("Mencari..."):
        rekomendasi_df = get_top_k_from_keyword(
            query, df, vectorizer, tfidf_matrix, k=k, remove_symbol_func=remove_symbol
        )

        if base_choice:
            jarak_col = f"Jarak_Kampus{base_choice.upper()}"
            if jarak_col in rekomendasi_df.columns:
                rekomendasi_df[jarak_col] = rekomendasi_df[jarak_col].round(2)
                rekomendasi_df = rekomendasi_df.sort_values(
                    by=[jarak_col, 'totalScore', 'similarity'],
                    ascending=[True, False, False]
                )

        if rekomendasi_df.empty:
            st.warning("Tidak ada hasil. Coba kata kunci lain.")
        else:
            for _, row in rekomendasi_df.iterrows():
                st.markdown(f"""
                ---
                ### 🍴 {row['title']}

                📍 *{row['address']}*  

                ⭐ **Rating:** {row['totalScore']}

                🧭 **Jarak dari Kampus {base_choice.upper()}:** {row.get(jarak_col, '-')} km

                🍽️ **Kategori:** {row['categories']}
                
                🔍 **Kemiripan:** {round(row['similarity'], 3)}
                """)
