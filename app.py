from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import io
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
# Load your Tensorflow model
model = tf.keras.models.load_model("model.h5")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    image_file = request.files['image']
    image_bytes = io.BytesIO(image_file.read())
    # Open the image using pillow
    img = Image.open(image_file)
    # Perform any processing you want on the image using pillow
    img = img.resize((28, 28))
    img.save("temp.png")
    # Make predictions using the Tensorflow model and add same preprocessing as I did for training
    test_image = tf.keras.preprocessing.image.load_img('temp.png', color_mode="grayscale", target_size=(28, 28))
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0) # add an extra dimension for batch size

    predictions = model.predict(test_image)
    predicted_classes = np.argmax(predictions, axis=1)
    predictions_percentage = predictions*100
    predictions_percentage = np.round(predictions_percentage, 2)

    class_names = ['bee', 'bowtie', 'butterfly', 'cat', 'diamond', 'eye', 'mushroom', 'octopus', 'popsicle', 'snowman']

    predicted_class = class_names[predicted_classes[0]]
    confidence = predictions_percentage[0][predicted_classes[0]]

    print("Model is predicting class " + predicted_class + " with " + str(confidence) + "% confidence.")
    # Return the predictions as a json
    return jsonify({'predictions': predictions.tolist(), 'predicted_class': predicted_class, 'confidence': str(confidence)})



if __name__ == '__main__':
    app.run(debug=True)
    