import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("pet_disease_dataset_300.csv")
    # Ensure proper column names
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# ----------------------------
# TF-IDF Vectorization
# ----------------------------
@st.cache_resource
def build_tfidf_model(symptoms):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(symptoms)
    return vectorizer, vectors

vectorizer, vectors = build_tfidf_model(df["symptoms"].astype(str))

# ----------------------------
# Function: Find Most Similar Disease
# ----------------------------
def predict_disease(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, vectors).flatten()
    idx = similarity.argmax()
    return df.iloc[idx], similarity[idx]

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(
    page_title="🐾 Pet Disease Predictor",
    page_icon="🐶",
    layout="wide"
)

st.title("🐾 Pet Disease & Home Treatment Predictor")
st.markdown(
    """
    Enter your pet's symptoms below, and our system will suggest the **most likely disease**  
    along with **home treatment advice**.  
    """
)

# Input box
user_input = st.text_area("✍️ Describe your pet's symptoms:", height=150)

if st.button("🔍 Predict Disease"):
    if user_input.strip():
        result, score = predict_disease(user_input)

        st.success(f"**Predicted Disease:** {result['disease']}")
        st.info(f"**Home Treatment Advice:** {result['treatment']}")

        st.write("---")
        st.caption(f"🔎 Similarity Score: {score:.2f}")
    else:
        st.warning("⚠️ Please enter some symptoms to get a prediction.")

# ----------------------------
# Extra Section
# ----------------------------
with st.expander("📂 View Dataset Sample"):
    st.dataframe(df.head(10))
