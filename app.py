import os
import numpy as np
import cv2
from keras.models import load_model
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('BrainTumor10EpochsCategorical.h5')
print("Model Loaded Successfully")

def get_className(classNo):
    return "No Brain Tumor" if classNo == 0 else "Brain Tumor Detected"

def getResult(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64))
    image = image.astype('float32') / 255.0

    input_img = np.expand_dims(image, axis=0)

    prediction = model.predict(input_img)
    result = np.argmax(prediction, axis=1)[0]

    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result = getResult(filepath)
    result_text = get_className(result)

    return jsonify({"result": result_text})

if __name__ == "__main__":
    app.run(debug=True)
