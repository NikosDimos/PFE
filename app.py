from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

app = Flask(__name__)

# Charger le modèle et le tokenizer fine-tuné
model_path = "./legal_bert_model"  # Chemin vers le modèle fine-tuné
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Dictionnaire pour mapper les classes aux messages explicites
class_mapping = {
    0: "Le document est une assignation et il y a un vice de procédure.",
    1: "Le document est une assignation sans vice de procédure.",
    2: "Le document est une notification et il y a un vice de procédure.",
    3: "Le document est une notification sans vice de procédure."
}

# Fonction pour faire une prédiction sur un texte
def predict(text):
    print("[INFO] Tokenization du texte...")
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

    print("[INFO] Prédiction en cours...")
    with torch.no_grad():
        outputs = model(**inputs)

    # Calculer les probabilités pour chaque classe
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).item()

    # Utiliser le dictionnaire pour retourner un message explicite
    result_message = class_mapping.get(prediction, "Le type de document est inconnu.")

    # Obtenir la probabilité pour la classe prédite
    confidence = probs[0][prediction].item()

    print("[INFO] Prédiction terminée.")
    return result_message, confidence, probs[0].tolist()

# Route pour afficher la page HTML
@app.route('/')
def index():
    return render_template('index.html')

# Route pour recevoir le texte extrait et effectuer la prédiction
@app.route('/predict', methods=['POST'])
def predict_text():
    print("[INFO] Requête reçue pour une prédiction.")
    data = request.get_json()

    if not data or 'text' not in data:
        print("[ERROR] Aucun texte reçu.")
        return jsonify({"error": "Aucun texte reçu."}), 400

    text = data['text']

    if not text.strip():
        print("[ERROR] Texte vide fourni.")
        return jsonify({"error": "Le texte fourni est vide."}), 400

    print("[INFO] Texte reçu. Longueur du texte :", len(text))

    # Faire la prédiction sur le texte fourni
    result_message, confidence, all_confidences = predict(text)

    print("[INFO] Résultat de la prédiction :", result_message)
    print("[INFO] Score de confiance :", confidence)

    # Retourner le message explicite, la probabilité, les autres probabilités et un texte explicatif
    return jsonify({
        "result_message": result_message,
        "confidence": confidence,  # Probabilité pour la classe prédite
        "all_confidences": all_confidences,  # Probabilités pour toutes les classes
        "confidence_explanation": "Le score de confiance représente la probabilité que le modèle attribue à la classe prédite. Plus ce score est élevé, plus le modèle est certain de sa prédiction."
    })

if __name__ == "__main__":
    print("[INFO] Démarrage de l'application Flask...")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
