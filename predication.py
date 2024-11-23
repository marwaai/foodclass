import tensorflow as tf
import requests
import os
from io import BytesIO
from model import create_model, corrector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def download_image(url):
    # Download the image from the URL
    try:
        response = requests.get(url)
        if response.status_code == 200:
            image = BytesIO(response.content)
            return image
        else:
            print("Error downloading image, status code:", response.status_code)
            return None
    except Exception as e:
        print(f"Failed to download image: {e}")
        return None

def make_prediction(image_path_or_url):
    print(f"BASE_DIR: {BASE_DIR}")
    
    wieghtpath = os.path.join(BASE_DIR, 't')
    print(f"Weights path: {wieghtpath}")

    # Initialize lists to store models
    modlee = []
    correct = []

    # Check if the input is a URL or local file
    if image_path_or_url.startswith('http'):
        print(f"Downloading image from URL: {image_path_or_url}")
        img = download_image(image_path_or_url)
        if img is None:
            return "Error: Unable to download image"

        # Load the image using TensorFlow from the BytesIO object
        img = tf.image.decode_jpeg(img.getvalue(), channels=3)  # Decode the JPEG image
        img = tf.image.resize(img, [25, 25])  # Resize the image
        img = tf.expand_dims(img, 0)  # Add batch dimension
    else:
        print(f"Loading image from local file: {image_path_or_url}")
        img = tf.keras.preprocessing.image.load_img(image_path_or_url, target_size=(25, 25))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.expand_dims(img, 0)  # Make batch of size 1

    # Normalize the image if needed (i.e., scaling the pixel values)
    img = img / 255.0

    # Debugging: Check image preprocessing
    print(f"Image Array Shape: {img.shape}")
    print(f"Image Array Sample Data: {img[0][0][:5]}")  # Print some sample pixel data for inspection

    # Load models
    for i in range(85):
        modlee.append(create_model())  # Create model instances
    for i in range(2):
        correct.append(corrector())  # Create corrector models

    # Load the weights for the models
    count = 0
    for i in os.listdir(wieghtpath):
        if i != "correctm.weights.h5" and i != "correctm2.weights.h5":
            modlee[count].load_weights(os.path.join(wieghtpath, i))
            count += 1
        else:
            if i == "correctm.weights.h5":
                correct[0].load_weights(os.path.join(wieghtpath, 'correctm.weights.h5'))
            elif i == "correctm2.weights.h5":
                correct[1].load_weights(os.path.join(wieghtpath, 'correctm2.weights.h5'))

    # Debugging: Ensure models have weights loaded
    print(f"Loaded {len(modlee)} models and {len(correct)} corrector models.")

    # Predict with all models in modlee
    total_predictions = 0
    for model in modlee:
        model_pred = model.predict(img)
        print(f"Model Prediction (Sample): {model_pred[:5]}")  # Debugging: Print the prediction sample from each model
        total_predictions += model_pred

    # Correct the predictions with the 'correct' models
    corrected_predictions = correct[0].predict(total_predictions)
    print(f"Corrected Predictions (Sample): {corrected_predictions[:5]}")  # Debugging

    final_predictions = correct[1].predict(corrected_predictions + total_predictions)
    print(f"Final Predictions (Sample): {final_predictions[:5]}")  # Debugging

    # Return the class with the highest probability
    predicted_class = final_predictions.argmax()  # Get the class with the highest probability
    print(f"Predicted Class: {predicted_class}")  # Final debug output

    return predicted_class
