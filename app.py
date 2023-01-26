from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
app = Flask(__name__)

# Load your Tensorflow model
model = tf.keras.models.load_model("model.h5")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    image_file = request.files['image']
    # Open the image using pillow
    img = Image.open(image_file)
    # Perform any processing you want on the image using pillow
    img = img.resize((28, 28))
    img_array = np.array(img)
    # Make predictions using the Tensorflow model
    predictions = model.predict(img_array)
    # Return the predictions as a json
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
