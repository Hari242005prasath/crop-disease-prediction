from django.db import models

class ImageModel(models.Model):
    image = models.ImageField(upload_to='captured_images/')

    def __str__(self):
        return f"Image {self.id}"
