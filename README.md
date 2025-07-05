# Food Recommendation System Around Universitas Airlangga 🍽️📍  
A personalized food place recommender using content‑based and location‑based filtering

This project provides an intelligent recommendation system that suggests food places around Universitas Airlangga (UNAIR) based on user preferences and campus location. Built with `Streamlit`, this app helps users discover relevant food options near Campus A, B, or C.

---

## 🎯 Features

- 🔎 Recommend food places based on user-input keywords (e.g. "bakso", "mie", "nasi goreng")
- 📍 Filter recommendations based on campus proximity (Campus A, B, or C)
- 📊 Set number of results (1 to 20 recommendations)
- 📌 View full name, address, and location of each place
- ✅ Built with content-based filtering and location-based distance calculations

---

## 🗂 Repository Structure

```
├── data/
│ └── data_googlemap_UNAIR_fix.csv # Dataset of 680+ food places near UNAIR
├── cosine/
│ └── cosine_matrix.csv # Precomputed cosine similarity matrix
├── functions/
│ └── food_function.py # Functions for preprocessing, filtering, etc.
├── food_app.py # Streamlit app entry point
├── README.md # You are here :)
```

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/chiizzu/RekomendasiMakananSekitarUNAIR.git
   cd RekomendasiMakananSekitarUNAIR
2. **Install dependencies**
   ```
   pip install -r requirements.txt
3. **Run the App**
   ```
   streamlit run app.py

---

## 📌 Technologies Used

- **Python** — Core scripting & data handling

- **pandas** — Data manipulation

- **scikit-learn** — Cosine similarity and Feature Engineering

- **Streamlit** — Web app interface

## 🧠 Methodology
**Content-based filtering** compares keywords to food place names.

**Location-based filtering** computes distances from selected UNAIR campus coordinates.

Results are ranked based on **proximity, rating, and similarity**.

---
## 🙌 Acknowledgements
- Inspired by the daily struggles of UNAIR students deciding where to eat 😉

- Built with love, pandas, and a lot of "lapar jam 3 sore"
