from django.db import models
from django.contrib.auth.models import User
import os

class Dataset(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    file_size = models.BigIntegerField(default=0)
    rows_count = models.IntegerField(default=0)
    columns_count = models.IntegerField(default=0)
    file_type = models.CharField(max_length=10, default='csv')
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return self.name
    
    def delete(self, *args, **kwargs):
        # Delete the file when the model instance is deleted
        if self.file:
            if os.path.isfile(self.file.path):
                os.remove(self.file.path)
        super().delete(*args, **kwargs)

class PreprocessingResult(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='preprocessing_results')
    operation_type = models.CharField(max_length=100)  # e.g., 'statistical_description', 'normalization'
    parameters = models.JSONField(default=dict)  # Store operation parameters
    result_data = models.JSONField(default=dict)  # Store the result
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.dataset.name} - {self.operation_type}"

class ClassificationResult(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='classification_results')
    algorithm = models.CharField(max_length=100)  # e.g., 'decision_tree', 'knn', 'naive_bayes'
    parameters = models.JSONField(default=dict)  # Store algorithm parameters
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    confusion_matrix = models.JSONField(default=dict)
    feature_importance = models.JSONField(default=dict, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.dataset.name} - {self.algorithm}"
