from rest_framework import serializers
from .models import Dataset, PreprocessingResult, ClassificationResult

class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = ['id', 'name', 'description', 'file', 'uploaded_at', 'file_size', 'rows_count', 'columns_count', 'file_type']
        read_only_fields = ['uploaded_at', 'file_size', 'rows_count', 'columns_count']

class PreprocessingResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = PreprocessingResult
        fields = ['id', 'dataset', 'operation_type', 'parameters', 'result_data', 'created_at']

class ClassificationResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = ClassificationResult
        fields = ['id', 'dataset', 'algorithm', 'parameters', 'accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix', 'feature_importance', 'created_at']
