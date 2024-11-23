from django.shortcuts import render
from django.http import HttpResponse
from cloudinary.uploader import upload
from predication import make_prediction, BASE_DIR
import json
import os

def my_view(request):
    label = []
    prediction = 0

    # Load the label.json file
    with open(os.path.join(BASE_DIR, "label.json"), "r") as json_file:
        label = json.load(json_file)

    if request.method == 'POST' and 'food_image' in request.FILES:
        uploaded_file = request.FILES['food_image']

        # Upload the image to Cloudinary
        try:
            cloudinary_response = upload(uploaded_file)
            file_url = cloudinary_response['url']  # Get the URL of the uploaded image from Cloudinary
        except Exception as e:
            print(f"Error uploading to Cloudinary: {e}")
            return render(request, "home.html", {'prediction': "Error during upload"})

        # Perform prediction using the Cloudinary image URL
        try:
            prediction = make_prediction(file_url)  # Pass the Cloudinary URL to the prediction function
        except Exception as e:
            print(f"Error during prediction: {e}")
            return render(request, "home.html", {'prediction': "Error during prediction"})

        # Render the results based on the prediction
        try:
            food_info = label[prediction]
            return render(request, "home.html", {'food': food_info["food"], 'cal': food_info["calories_per_100g"], 'image_url': file_url})
        except IndexError:
            print(f"Prediction index out of range. Check the 'label.json' file.")
            return render(request, "home.html", {'prediction': "Invalid prediction index"})

    else:
        return render(request, "home.html", {'prediction': "No image uploaded"})
