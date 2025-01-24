from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import fitz  # PyMuPDF pour extraire le texte du PDF

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

# Fonction pour extraire le texte d'un PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Fonction pour faire une prédiction sur un texte
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    # Calculer les probabilités pour chaque classe
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).item()

    # Utiliser le dictionnaire pour retourner un message explicite
    result_message = class_mapping.get(prediction, "Le type de document est inconnu.")

    # Obtenir la probabilité pour la classe prédite
    confidence = probs[0][prediction].item()  # La probabilité de la classe prédite

    return result_message, confidence, probs[0].tolist()  # Retourner également toutes les probabilités

# Route pour afficher la page HTML
@app.route('/')
def index():
    return render_template('index.html')

# Route pour recevoir le fichier PDF et effectuer la prédiction
@app.route('/predict', methods=['POST'])
def predict_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier téléchargé."}), 400

    file = request.files['file']
    if file and file.filename.endswith('.pdf'):
        # Sauvegarder temporairement le fichier
        pdf_path = './uploaded_file.pdf'
        file.save(pdf_path)

        # Extraire le texte du PDF
        text = extract_text_from_pdf(pdf_path)

        # Faire la prédiction sur le texte extrait
        result_message, confidence, all_confidences = predict(text)

        # Retourner le message explicite, la probabilité, les autres probabilités et un texte explicatif
        return jsonify({
            "result_message": result_message,
            "confidence": confidence,  # Probabilité pour la classe prédite
            "all_confidences": all_confidences,  # Probabilités pour toutes les classes
            "confidence_explanation": "Le score de confiance représente la probabilité que le modèle attribue à la classe prédite. Plus ce score est élevé, plus le modèle est certain de sa prédiction."
        })

    return jsonify({"error": "Veuillez télécharger un fichier PDF valide."}), 400


if __name__ == '__main__':
    app.run(debug=True)