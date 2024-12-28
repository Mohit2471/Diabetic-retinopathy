from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO

app = Flask(__name__)

# Load the pre-trained model
model = load_model('DR_noDR.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        file = request.files['file']
        
        # Use BytesIO to handle the file in-memory
        img_bytes = BytesIO(file.read())
        
        # Load the image from BytesIO
        img = image.load_img(img_bytes, target_size=(224, 224))

        # Continue with the rest of the code as before...
        # Convert the image to a numpy array, make predictions, etc.
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image

        # Make a prediction
        prediction = model.predict(img_array)

        # Determine the predicted class (0 or 1)
        predicted_label = "Diabetic Retina" if prediction[0][0] > 0.5 else "No Issues Detected"

        return render_template('index.html', prediction=predicted_label)

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
