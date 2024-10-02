import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from django.http import HttpResponse
from django.shortcuts import render, redirect
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from PIL import Image
import google.generativeai as genai


class_names = [
    'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy',
    'Blueberry_healthy', 'Cherry(including_sour)_Powdery_mildew', 'Cherry(including_sour)_healthy',
    'Corn(maize)_Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)_Common_rust', 
    'Corn_(maize)_Northern_Leaf_Blight', 'Corn(maize)_healthy', 'Grape_Black_rot', 
    'Grape_Esca(Black_Measles)', 'Grape_Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape_healthy',
    'Orange_Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach_healthy',
    'Pepper,_bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato_Early_blight', 
    'Potato_Late_blight', 'Potato_healthy', 'Raspberry_healthy', 'Soybean_healthy',
    'Squash_Powdery_mildew', 'Strawberry_Leaf_scorch', 'Strawberry_healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites Two-spotted_spider_mite', 'Tomato_Target_Spot',
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato__healthy'
]


plant_names = {
    'Apple__Apple_scab': 'Apple',
    'Apple_Black_rot': 'Apple',
    'Apple_Cedar_apple_rust': 'Apple',
    'Apple_healthy': 'Apple',
    'Blueberry_healthy': 'Blueberry',
    'Cherry(including_sour)_Powdery_mildew': 'Cherry',
    'Cherry(including_sour)_healthy': 'Cherry',
    'Corn(maize)_Cercospora_leaf_spot Gray_leaf_spot': 'Corn',
    'Corn(maize)_Common_rust': 'Corn',
    'Corn_(maize)_Northern_Leaf_Blight': 'Corn',
    'Corn(maize)_healthy': 'Corn',
    'Grape_Black_rot': 'Grape',
    'Grape_Esca(Black_Measles)': 'Grape',
    'Grape_Leaf_blight(Isariopsis_Leaf_Spot)': 'Grape',
    'Grape_healthy': 'Grape',
    'Orange_Haunglongbing(Citrus_greening)': 'Orange',
    'Peach__Bacterial_spot': 'Peach',
    'Peach_healthy': 'Peach',
    'Pepper,_bell_Bacterial_spot': 'Pepper (bell)',
    'Pepper,_bell_healthy': 'Pepper (bell)',
    'Potato_Early_blight': 'Potato',
    'Potato_Late_blight': 'Potato',
    'Potato_healthy': 'Potato',
    'Raspberry_healthy': 'Raspberry',
    'Soybean_healthy': 'Soybean',
    'Squash_Powdery_mildew': 'Squash',
    'Strawberry_Leaf_scorch': 'Strawberry',
    'Strawberry_healthy': 'Strawberry',
    'Tomato_Bacterial_spot': 'Tomato',
    'Tomato_Early_blight': 'Tomato',
    'Tomato_Late_blight': 'Tomato',
    'Tomato_Leaf_Mold': 'Tomato',
    'Tomato_Septoria_leaf_spot': 'Tomato',
    'Tomato_Spider_mites Two-spotted_spider_mite': 'Tomato',
    'Tomato_Target_Spot': 'Tomato',
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus': 'Tomato',
    'Tomato_Tomato_mosaic_virus': 'Tomato',
    'Tomato__healthy': 'Tomato'
}

genai.configure(api_key="AIzaSyALLmlsEYf8GlssEM9YuR0KfWs5V5nre5M")

def fetch_from_gemini(user_input):
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 250,
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    chat_session = model.start_chat(history=[{"role": "user", "parts": [user_input]}])
    response = chat_session.send_message(user_input)
    
    return response.text


model_path = 'plant_disease_model (1).h5' 
model = load_model(model_path)

def preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  
    return img_array


def predict_disease(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_names[predicted_class_index]
    predicted_plant_name = plant_names[predicted_class]
    return predicted_class, predicted_plant_name

import os
import tempfile
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.conf import settings


@csrf_exempt  
def generate_report(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image_upload')

        if not image_file:
            return JsonResponse({'status': 'error', 'message': 'No image uploaded.'}, status=400)

        
        with tempfile.TemporaryDirectory() as temp_dir:
            img_path = os.path.join(temp_dir, image_file.name)
            
            
            with open(img_path, 'wb+') as destination:
                for chunk in image_file.chunks():
                    destination.write(chunk)

            try:
               
                predicted_disease, predicted_plant_name = predict_disease(img_path)

                
                precautions_prompt = f"Provide precautions in 3 points and required fertilizers in 3 points for {predicted_disease}"
                precautions_text = fetch_from_gemini(precautions_prompt)

                
                pdf_content = (
                    f"*Predicted Disease:* {predicted_disease}\n"
                    f"*Predicted Plant:* {predicted_plant_name}\n\n"
                    f"*Precautions:*\n{precautions_text}"
                )

                
                pdf_buffer = generate_pdf(pdf_content, img_path)

                #
                response = HttpResponse(pdf_buffer, content_type='application/pdf')
                response['Content-Disposition'] = 'attachment; filename="disease_report.pdf"'
                return response

            except Exception as e:
            
                return JsonResponse({'status': 'error', 'message': f'Error generating report: {str(e)}'}, status=500)

    else:
        
        return render(request, 'generate_report.html')



def generate_pdf(text_content, img_path):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.setFont("Helvetica", 12)

    width, height = A4
    margin = 50
    line_height = 20

    y_position = height - margin

    clean_text_content = text_content.replace('#', '').replace('*', '').strip()
    lines = clean_text_content.split('\n')

    for line in lines:
        wrapped_lines = simpleSplit(line.strip(), "Helvetica", 12, width - 2 * margin)
        for wrapped_line in wrapped_lines:
            pdf.drawString(margin, y_position, wrapped_line)
            y_position -= line_height

            if y_position < margin:
                pdf.showPage()
                y_position = height - margin
                pdf.setFont("Helvetica", 12)

   
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        pdf.drawImage(img_path, margin, y_position - 200, width - 2 * margin, 200)

    pdf.showPage()
    pdf.save()
    buffer.seek(0)

    return buffer


def capture_image(request):
    if request.method == 'POST' and 'image' in request.FILES:
        image_file = request.FILES['image']
        image_instance = ImageModel(image=image_file)
        image_instance.save()
        return redirect('admin:index')
    return render(request, 'capture_image.html')

def home(request):
    return render(request, 'home.html')


def get_wordpress_posts(request):
    url = "https://sushmitha.socialmm.in/wp-json/wp/v2/posts"
    
    try:
        response = requests.get(url)
        response.raise_for_status()

        if response.headers.get('content-type') == 'application/json; charset=UTF-8':
            posts = response.json()
            return render(request, 'posts.html', {'posts': posts})
        else:
            return HttpResponse(f"Invalid content type: {response.headers.get('content-type')}", status=500)

    except requests.exceptions.RequestException as e:
        return HttpResponse(f"Request failed: {str(e)}", status=500)
    except ValueError as e:
        return HttpResponse(f"Invalid JSON response: {str(e)}", status=500)



from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class MyDataView(APIView):
    def get(self, request, format=None):
        data = {"key": "value"}  
        return Response(data, status=status.HTTP_200_OK)