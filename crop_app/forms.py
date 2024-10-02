from django import forms
from .models import ImageModel  # Assuming you have a model to store images

class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = ImageModel  # Replace with your actual image model
        fields = ['image']  # Replace 'image' with the appropriate field
