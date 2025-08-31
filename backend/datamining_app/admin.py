from django.contrib import admin
from .models import Dataset, PreprocessingResult, ClassificationResult

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ['name', 'file_type', 'rows_count', 'columns_count', 'uploaded_at']
    list_filter = ['file_type', 'uploaded_at']
    search_fields = ['name', 'description']
    readonly_fields = ['uploaded_at', 'file_size', 'rows_count', 'columns_count']

@admin.register(PreprocessingResult)
class PreprocessingResultAdmin(admin.ModelAdmin):
    list_display = ['dataset', 'operation_type', 'created_at']
    list_filter = ['operation_type', 'created_at']
    search_fields = ['dataset__name']

@admin.register(ClassificationResult)
class ClassificationResultAdmin(admin.ModelAdmin):
    list_display = ['dataset', 'algorithm', 'accuracy', 'created_at']
    list_filter = ['algorithm', 'created_at']
    search_fields = ['dataset__name']
