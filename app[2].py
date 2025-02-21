from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Set the template folder path explicitly (if necessary)
app = Flask(__name__, template_folder='templates')

# Load the pre-trained model (ensure the path to signature_model.h5 is correct)
model = tf.keras.models.load_model("signature_model.h5")


def predict_signature(image):
    """
    Function to predict whether the signature is genuine or forged.
    """
    image = image.resize((128, 128)).convert('L')  # Convert image to grayscale
    img_array = np.array(image) / 255.0  # Normalize the image
    img_array = img_array.reshape(1, 128, 128, 1)  # Reshape for the model input
    prediction = model.predict(img_array)  # Predict the class
    return "Genuine" if np.argmax(prediction) == 0 else "Forged"  # Return the predicted result


@app.route('/')
def index():
    """
    Route to render the index page with a welcome message.
    """
    return render_template('index.html')  # Make sure index.html is in the 'templates' folder


@app.route('/predict', methods=['POST'])
def predict():
    """
    Route to handle the file upload and prediction for two images.
    """
    # Check if both file inputs exist in the form data
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'Both files must be uploaded'}), 400  # Return error if either file is missing

    file1 = request.files['file1']
    file2 = request.files['file2']

    # Validate file types
    for file in [file1, file2]:
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG are allowed.'}), 400

    try:
        # Load and process both images
        image1 = Image.open(io.BytesIO(file1.read()))
        image2 = Image.open(io.BytesIO(file2.read()))

        # Predict results
        result1 = predict_signature(image1)
        result2 = predict_signature(image2)

        # Return predictions as JSON response
        return jsonify({
            'predictions': [
                {'filename': file1.filename, 'prediction': result1},
                {'filename': file2.filename, 'prediction': result2}
            ]
        })

    except Exception as e:
        # Handle any image processing or prediction errors
        return jsonify({'error': f'Error processing the images: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app in debug mode
