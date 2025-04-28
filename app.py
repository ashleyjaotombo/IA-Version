from flask import Flask, request, render_template
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

# Initialisation de Flask
app = Flask(__name__)

# Configuration des dossiers
UPLOAD_FOLDER = 'uploads'
TEST_FOLDER = 'Test'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = (224, 224)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Vérifie l'extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Prétraitement image
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        print(f"[ERREUR] Fichier introuvable : {image_path}")
        return None
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERREUR] Impossible de lire l'image : {image_path}")
        return None
    img = cv2.resize(img, IMG_SIZE)
    return img.astype(np.float32)

# Comparaison via MSE
def compare_images(img1_path, test_folder):
    img1 = preprocess_image(img1_path)
    if img1 is None:
        return None, float('inf')

    best_score = float('inf')
    best_match = None

    for filename in os.listdir(test_folder):
        if not filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
            continue

        img2_path = os.path.join(test_folder, filename)
        img2 = preprocess_image(img2_path)
        if img2 is None:
            continue

        score = np.mean((img1 - img2) ** 2)
        print(f"[MSE] {filename} → {score:.2f}")

        if score < best_score:
            best_score = score
            best_match = filename

    if best_match:
        print(f"[MATCH] Correspondance trouvée : {best_match} (score : {best_score:.2f})")
    else:
        print("[INFO] Aucune correspondance.")

    return best_match, best_score

# Route principale
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error="Aucune image reçue.")

        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"[UPLOAD] Image enregistrée : {filepath}")

            match, score = compare_images(filepath, TEST_FOLDER)
            result = f"Image la plus proche : {match} (MSE : {score:.2f})" if match else "Aucune correspondance trouvée."
            return render_template('index.html', result=result)

        return render_template('index.html', error="Format non supporté.")
    return render_template('index.html')

# Lancement
if __name__ == '__main__':
    app.run(debug=True, port=5001)
