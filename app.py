from flask import jsonify, request, Flask 
from inference_sdk import InferenceHTTPClient
import openai
from PIL import Image
import numpy as np
import cv2
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Initialisation du client Roboflow
CLIENT = InferenceHTTPClient(
    api_url=os.getenv("ROBOFLOW_API_URL", "https://detect.roboflow.com"),
    api_key=os.getenv("ROBOFLOW_API_KEY", "z9gW5fliiH9edLVUeMtP")
)

my_model = {
    "feuille de cacao": "anthracanose_cocoa-0kqjz/1",
    "feuille de manioc": "cassavadisease/1",
    "tomates": "tomato_2/1",
    "feuille de riz": "rice-diseases-qzjka/3",
    "boeufs": "lumpyskin/1",
    "animaux": "animal_pathologies/1",
    "porcs": "pig-skin-disease-single-label-u3gzh/1"
}

# Initialisation de l'API ChatGPT
openai.api_key = os.getenv('OPENAI_API_KEY')

@app.route('/prediction', methods=['POST'])
def model():
    if 'image' not in request.files:
        return jsonify({'error': 'ajouter une image'}), 400

    # Récupère le paramètre du modèle dans le formulaire
    model_name = request.form.get('model_name')
    if not model_name or model_name not in my_model:
        return jsonify({'error': 'Modèle non spécifié ou non reconnu'}), 400

    file = request.files['image']
    try:
        # Lecture et décodage de l'image
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Erreur lors du décodage de l\'image'}), 400

        # Utilisation du modèle choisi
        prediction = CLIENT.infer(image, model_id=my_model[model_name])

        # Appel à l'API ChatGPT pour obtenir un conseil
        prompt = f"Le modèle que vous me demandez : {prediction}. Pouvez-vous me donner un conseil sur ce modèle, son origine et s'il a déjà touché des animaux en Côte d'Ivoire ? En répondant ne parle pas du terme de model mais répond directement en donnant des conseils et en suivat les autres descriptions"
        chat_completion = openai.ChatCompletion.create(
            messages=[{"role": "user", "content": prompt}],
            model='gpt-4o-mini'
        )
        response = chat_completion.choices[0].message['content']

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({
        "prediction": prediction,
        "conseil": response
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000), debug=True)
