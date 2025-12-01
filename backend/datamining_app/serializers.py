from rest_framework import serializers
from .models import Dataset, PreprocessingResult, ClassificationResult, ClusteringResult, AssociationRuleResult, WebMiningResult

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

class ClusteringResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = ClusteringResult
        fields = ['id', 'dataset', 'algorithm', 'parameters', 'n_clusters', 'silhouette_score', 'davies_bouldin_score', 'cluster_labels', 'cluster_centers', 'inertia', 'result_data', 'created_at']

class AssociationRuleResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = AssociationRuleResult
        fields = ['id', 'dataset', 'min_support', 'min_confidence', 'frequent_itemsets', 'rules', 'result_data', 'created_at']

class WebMiningResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = WebMiningResult
        fields = ['id', 'dataset', 'algorithm', 'parameters', 'scores', 'result_data', 'created_at']
