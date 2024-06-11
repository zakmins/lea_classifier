from django.http import JsonResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from keras.models import load_model # type: ignore
from keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.xception import preprocess_input # type: ignore
import numpy as np
from .model_loader import model  # Import the preloaded model

from django.core.files.uploadedfile import InMemoryUploadedFile
from PIL import Image

# Create your views here.

def classify_image(request):
    if request.method == 'POST' and 'image' in request.FILES:
        uploaded_file = request.FILES['image']
        
        if isinstance(uploaded_file, InMemoryUploadedFile):
            img = Image.open(uploaded_file)
            
            # Convert the image to RGB mode if it's not already
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize the image to the required dimensions
            img = img.resize((299, 299))
            
            # Convert the image to a NumPy array
            img_array = np.array(img)
            
            # Expand dimensions to match the input shape expected by the model
            img_array = np.expand_dims(img_array, axis=0)
            
            # Preprocess the array
            img_array = preprocess_input(img_array)

            # Make prediction
            predicted_class = (model.predict(img_array) > 0.5).astype('int32')

            # Return JSON response
            return JsonResponse({'predicted_class': int(predicted_class)})
    return render(request, 'main.html')