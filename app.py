from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Load your Tensorflow model
model = tf.keras.models.load_model("oldmodel.h5")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    image_file = request.files['image']
    # Open the image using pillow
    img = Image.open(image_file)
    # Perform any processing you want on the image using pillow

    # Get the bounding box coordinates
    left, upper, right, lower = img.getbbox()

    dimension = max(abs(left-right), abs(upper-lower))
    cropped_img = img.crop((left, upper, left+dimension, upper+dimension))

    cropped_img.show()
    cropped_img.save('ya.png')
    width, height = cropped_img.size

    bg = Image.new('RGB', (width, height), (255, 255, 255))
    bg.paste(cropped_img, (0, 0), cropped_img)

    # save the cropped image
    im_cropped = bg.resize((28, 28))
    im_cropped.save("temp.png")
    
    # Make predictions using the Tensorflow model and add same preprocessing as I did for training
    test_image = tf.keras.preprocessing.image.load_img('temp.png', color_mode="grayscale", target_size=(28, 28))
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0) # add an extra dimension for batch size

    predictions = model.predict(test_image)

    class_names = ['bee', 'bowtie', 'butterfly', 'cat', 'diamond', 'eye', 'mushroom', 'octopus', 'popsicle', 'snowman']
    predictions_percentage = predictions * 100
    predictions_percentage = np.round(predictions_percentage, 2)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_class = class_names[predicted_classes[0]]
    confidence = predictions_percentage[0][predicted_classes[0]]
    all_predictions = []
    for i in range(len(predictions[0])):
        all_predictions.append({"class_name": class_names[i], "confidence": str(predictions_percentage[0][i]) + "%"})
    print("Model is predicting class " + predicted_class + " with " + str(confidence) + "% confidence.")
    # Return the predictions as a json
    return jsonify({'predictions': all_predictions, 'predicted_class': predicted_class, 'confidence': str(confidence)})



if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=3000)