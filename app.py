import os
import json
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# ── Model path ───────────────────────────────────────────────────────────
MODEL_PATH = 'model/brain_tumor_model.h5'

# ── Load class names ────────────────────────────────────────────────────
with open('model/class_names.json') as f:
    CLASS_NAMES = json.load(f)

# ── Pretty display names ─────────────────────────────────────────────────
DISPLAY_NAMES = {
    'glioma_tumor'     : 'Glioma Tumor',
    'meningioma_tumor' : 'Meningioma Tumor',
    'normal'           : 'No Tumor (Normal)',
    'pituitary_tumor'  : 'Pituitary Tumor',
}

DESCRIPTIONS = {
    'glioma_tumor'     : 'Gliomas are tumors that arise from glial cells in the brain or spine. They account for about 33% of all brain tumors.',
    'meningioma_tumor' : 'Meningiomas arise from the meninges, the membranes surrounding the brain and spinal cord. Most are benign and slow-growing.',
    'normal'           : 'No signs of tumor detected in this MRI scan. The brain tissue appears normal.',
    'pituitary_tumor'  : 'Pituitary tumors form in the pituitary gland at the base of the brain. Most are benign (adenomas).',
}

COLORS = {
    'glioma_tumor'     : '#e74c3c',
    'meningioma_tumor' : '#e67e22',
    'normal'           : '#2ecc71',
    'pituitary_tumor'  : '#3498db',
}

# ── Load model (bundled in repo) ─────────────────────────────────────────
print('Loading model...')
model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
print('Model loaded successfully!')

# ── Allowed file types ───────────────────────────────────────────────────
ALLOWED = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED

# ── Preprocess image ─────────────────────────────────────────────────────
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

# ── Routes ────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use JPG or PNG.'}), 400

    filename  = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(save_path)

    # Predict
    arr   = preprocess_image(save_path)
    preds = model.predict(arr, verbose=0)[0]

    # Build results
    top_idx   = int(np.argmax(preds))
    top_class = CLASS_NAMES[top_idx]
    top_conf  = float(preds[top_idx]) * 100

    all_preds = [
        {
            'label'      : CLASS_NAMES[i],
            'display'    : DISPLAY_NAMES[CLASS_NAMES[i]],
            'confidence' : round(float(preds[i]) * 100, 2),
            'color'      : COLORS[CLASS_NAMES[i]],
        }
        for i in range(len(CLASS_NAMES))
    ]
    all_preds.sort(key=lambda x: x['confidence'], reverse=True)

    return jsonify({
        'prediction'  : DISPLAY_NAMES[top_class],
        'confidence'  : round(top_conf, 2),
        'description' : DESCRIPTIONS[top_class],
        'color'       : COLORS[top_class],
        'all_preds'   : all_preds,
        'image_url'   : '/' + save_path,
        'is_normal'   : top_class == 'normal',
    })

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
