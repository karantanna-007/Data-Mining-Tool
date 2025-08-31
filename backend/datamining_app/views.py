from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from django.http import JsonResponse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from scipy.stats import chi2_contingency
import json
import os

from .models import Dataset, PreprocessingResult, ClassificationResult
from .serializers import DatasetSerializer, PreprocessingResultSerializer, ClassificationResultSerializer

class DatasetViewSet(viewsets.ModelViewSet):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer
    parser_classes = (MultiPartParser, FormParser, JSONParser)
    
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            # Save the dataset
            dataset = serializer.save()
            
            # Read the file and get basic info
            try:
                if dataset.file.name.endswith('.csv'):
                    df = pd.read_csv(dataset.file.path)
                    dataset.rows_count = len(df)
                    dataset.columns_count = len(df.columns)
                    dataset.file_type = 'csv'
                elif dataset.file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(dataset.file.path)
                    dataset.rows_count = len(df)
                    dataset.columns_count = len(df.columns)
                    dataset.file_type = 'excel'
                
                dataset.file_size = os.path.getsize(dataset.file.path)
                dataset.save()
                
            except Exception as e:
                return Response({'error': f'Error reading file: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
            
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=True, methods=['get'])
    def preview(self, request, pk=None):
        """Get a preview of the dataset (first 10 rows)"""
        dataset = self.get_object()
        try:
            if dataset.file_type == 'csv':
                df = pd.read_csv(dataset.file.path)
            else:
                df = pd.read_excel(dataset.file.path)
            
            # Get first 10 rows and column info
            preview_data = {
                'columns': list(df.columns),
                'data': df.head(10).to_dict('records'),
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'data_types': df.dtypes.astype(str).to_dict()
            }
            
            return Response(preview_data)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=True, methods=['post'])
    def preprocess(self, request, pk=None):
        """Perform preprocessing operations on the dataset"""
        dataset = self.get_object()
        operation = request.data.get('operation')
        parameters = request.data.get('parameters', {})
        
        try:
            if dataset.file_type == 'csv':
                df = pd.read_csv(dataset.file.path)
            else:
                df = pd.read_excel(dataset.file.path)
            
            result_data = {}
            
            if operation == 'statistical_description':
                result_data = {
                    'description': df.describe().to_dict(),
                    'info': {
                        'shape': df.shape,
                        'columns': list(df.columns),
                        'dtypes': df.dtypes.astype(str).to_dict(),
                        'null_counts': df.isnull().sum().to_dict()
                    }
                }
            
            elif operation == 'correlation':
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    correlation_matrix = numeric_df.corr().to_dict()
                    covariance_matrix = numeric_df.cov().to_dict()
                    result_data = {
                        'correlation': correlation_matrix,
                        'covariance': covariance_matrix
                    }
                else:
                    result_data = {'error': 'No numeric columns found for correlation analysis'}
            
            elif operation == 'normalization':
                method = parameters.get('method', 'minmax')
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                
                if method == 'minmax':
                    scaler = MinMaxScaler()
                    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
                elif method == 'zscore':
                    scaler = StandardScaler()
                    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
                elif method == 'decimal':
                    for col in numeric_columns:
                        max_val = df[col].abs().max()
                        decimal_places = len(str(int(max_val)))
                        df[col] = df[col] / (10 ** decimal_places)
                
                result_data = {
                    'normalized_data': df.head(10).to_dict('records'),
                    'method': method,
                    'columns_normalized': list(numeric_columns)
                }
            
            elif operation == 'chi_square':
                categorical_columns = df.select_dtypes(include=['object']).columns
                if len(categorical_columns) >= 2:
                    col1, col2 = categorical_columns[0], categorical_columns[1]
                    if 'column1' in parameters:
                        col1 = parameters['column1']
                    if 'column2' in parameters:
                        col2 = parameters['column2']
                    
                    contingency_table = pd.crosstab(df[col1], df[col2])
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                    
                    result_data = {
                        'chi2_statistic': chi2,
                        'p_value': p_value,
                        'degrees_of_freedom': dof,
                        'contingency_table': contingency_table.to_dict(),
                        'columns_tested': [col1, col2]
                    }
                else:
                    result_data = {'error': 'Need at least 2 categorical columns for chi-square test'}
            
            # Save preprocessing result
            preprocessing_result = PreprocessingResult.objects.create(
                dataset=dataset,
                operation_type=operation,
                parameters=parameters,
                result_data=result_data
            )
            
            return Response({
                'id': preprocessing_result.id,
                'operation': operation,
                'result': result_data
            })
            
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=True, methods=['post'])
    def classify(self, request, pk=None):
        """Perform classification on the dataset"""
        dataset = self.get_object()
        algorithm = request.data.get('algorithm')
        parameters = request.data.get('parameters', {})
        target_column = request.data.get('target_column')
        
        try:
            if dataset.file_type == 'csv':
                df = pd.read_csv(dataset.file.path)
            else:
                df = pd.read_excel(dataset.file.path)
            
            if target_column not in df.columns:
                return Response({'error': 'Target column not found'}, status=status.HTTP_400_BAD_REQUEST)
            
            # Prepare data
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Handle categorical variables
            le = LabelEncoder()
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = le.fit_transform(X[col].astype(str))
            
            if y.dtype == 'object':
                y = le.fit_transform(y.astype(str))
            
            # Split data
            test_size = parameters.get('test_size', 0.2)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Initialize classifier
            if algorithm == 'decision_tree':
                criterion = parameters.get('criterion', 'entropy')
                clf = DecisionTreeClassifier(criterion=criterion, random_state=42)
            elif algorithm == 'knn':
                n_neighbors = parameters.get('n_neighbors', 5)
                clf = KNeighborsClassifier(n_neighbors=n_neighbors)
            elif algorithm == 'naive_bayes':
                clf = GaussianNB()
            elif algorithm == 'logistic_regression':
                clf = LogisticRegression(random_state=42, max_iter=1000)
            elif algorithm == 'neural_network':
                hidden_layer_sizes = parameters.get('hidden_layer_sizes', (100,))
                clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=42, max_iter=1000)
            else:
                return Response({'error': 'Unsupported algorithm'}, status=status.HTTP_400_BAD_REQUEST)
            
            # Train and predict
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            cm = confusion_matrix(y_test, y_pred).tolist()
            
            # Feature importance (if available)
            feature_importance = {}
            if hasattr(clf, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, clf.feature_importances_.tolist()))
            
            result_data = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'feature_importance': feature_importance,
                'algorithm': algorithm,
                'parameters': parameters
            }
            
            # Save classification result
            classification_result = ClassificationResult.objects.create(
                dataset=dataset,
                algorithm=algorithm,
                parameters=parameters,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                confusion_matrix=cm,
                feature_importance=feature_importance
            )
            
            return Response({
                'id': classification_result.id,
                'result': result_data
            })
            
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

class PreprocessingResultViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = PreprocessingResult.objects.all()
    serializer_class = PreprocessingResultSerializer

class ClassificationResultViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = ClassificationResult.objects.all()
    serializer_class = ClassificationResultSerializer
