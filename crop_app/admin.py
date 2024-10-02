from django.contrib import admin
from django.utils.safestring import mark_safe
from .models import ImageModel  # Replace with your actual model name

class ImageModelAdmin(admin.ModelAdmin):
    list_display = ('image_tag',)  # Adjust based on your model fields

    def image_tag(self, obj):
        return mark_safe(f'<img src="{obj.image.url}" width="150" height="150"/>')

    image_tag.short_description = 'Image Preview'

admin.site.register(ImageModel, ImageModelAdmin)
