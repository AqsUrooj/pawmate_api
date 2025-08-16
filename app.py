from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------
# Load and prepare data
# -----------------
df = pd.read_csv("pet_disease_dataset_300.csv")
df['Symptom (User Input)'] = df['Symptom (User Input)'].str.lower().str.strip()
df['Animal'] = df['Animal'].str.lower().str.strip()

vectorizer = TfidfVectorizer()
vectorizer.fit(df['Symptom (User Input)'])

# -----------------
# Flask app setup
# -----------------
app = Flask(__name__)

def get_diagnosis(user_input, animal_type=None):
    # Preprocess
    user_input_clean = user_input.lower().strip()
    
    # Filter by animal
    if animal_type:
        filtered_df = df[df['Animal'] == animal_type.lower()]
    else:
        filtered_df = df.copy()
    
    if filtered_df.empty:
        return None
    
    # Vectorize
    filtered_vectors = vectorizer.transform(filtered_df['Symptom (User Input)'])
    user_vector = vectorizer.transform([user_input_clean])
    
    # Similarity
    similarities = cosine_similarity(user_vector, filtered_vectors).flatten()
    best_idx = similarities.argmax()
    best_row = filtered_df.iloc[best_idx]
    
    return {
        "animal": best_row['Animal'].title(),
        "disease": best_row['Disease'],
        "treatment": best_row['Home Treatment (User-Friendly)'],
        "confidence": round(float(similarities[best_idx]), 2)
    }

@app.route("/diagnose", methods=["POST"])
def diagnose():
    data = request.json
    user_input = data.get("symptoms", "")
    animal_type = data.get("animal", None)
    
    result = get_diagnosis(user_input, animal_type)
    if not result:
        return jsonify({"error": "No matching data"}), 404
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
