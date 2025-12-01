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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from scipy.stats import chi2_contingency
from scipy.spatial.distance import cdist
import json
import os
import networkx as nx
from itertools import combinations

from .models import Dataset, PreprocessingResult, ClassificationResult, ClusteringResult, AssociationRuleResult, WebMiningResult
from .serializers import DatasetSerializer, PreprocessingResultSerializer, ClassificationResultSerializer, ClusteringResultSerializer, AssociationRuleResultSerializer, WebMiningResultSerializer

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
    
    @action(detail=True, methods=['post'])
    def cluster(self, request, pk=None):
        """Perform clustering on the dataset"""
        dataset = self.get_object()
        algorithm = request.data.get('algorithm')
        parameters = request.data.get('parameters', {})
        
        try:
            if dataset.file_type == 'csv':
                df = pd.read_csv(dataset.file.path)
            else:
                df = pd.read_excel(dataset.file.path)
            
            # Select only numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                return Response({'error': 'No numeric columns found for clustering'}, status=status.HTTP_400_BAD_REQUEST)
            
            n_clusters = parameters.get('n_clusters', 3)
            
            if algorithm == 'kmeans':
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(numeric_df)
                
                silhouette = silhouette_score(numeric_df, cluster_labels)
                davies_bouldin = davies_bouldin_score(numeric_df, cluster_labels)
                
                result_data = {
                    'cluster_labels': cluster_labels.tolist(),
                    'cluster_centers': kmeans.cluster_centers_.tolist(),
                    'inertia': float(kmeans.inertia_),
                    'silhouette_score': float(silhouette),
                    'davies_bouldin_score': float(davies_bouldin),
                    'n_clusters': n_clusters,
                    'algorithm': 'kmeans'
                }
                
                
                clustering_result = ClusteringResult.objects.create(
                    dataset=dataset,
                    algorithm='kmeans',
                    parameters=parameters,
                    n_clusters=n_clusters,
                    silhouette_score=silhouette,
                    davies_bouldin_score=davies_bouldin,
                    cluster_labels=cluster_labels.tolist(),
                    cluster_centers=kmeans.cluster_centers_.tolist(),
                    inertia=float(kmeans.inertia_),
                    result_data=result_data
                )
                
            elif algorithm == 'kmedoid':
                from sklearn.metrics.pairwise import pairwise_distances
                
                # Initialize medoids randomly
                np.random.seed(42)
                medoid_indices = np.random.choice(len(numeric_df), n_clusters, replace=False)
                
                # Calculate distance matrix
                distances = pairwise_distances(numeric_df)
                
                # K-Medoid iterations
                for iteration in range(100):
                    # Assign points to nearest medoid
                    cluster_labels = np.argmin(distances[medoid_indices].T, axis=1)
                    
                    # Update medoids
                    new_medoids = []
                    for i in range(n_clusters):
                        cluster_points = np.where(cluster_labels == i)[0]
                        if len(cluster_points) > 0:
                            # Find point with minimum average distance to other points in cluster
                            avg_distances = distances[cluster_points][:, cluster_points].sum(axis=1)
                            new_medoid = cluster_points[np.argmin(avg_distances)]
                            new_medoids.append(new_medoid)
                        else:
                            new_medoids.append(medoid_indices[i])
                    
                    if np.array_equal(medoid_indices, new_medoids):
                        break
                    medoid_indices = np.array(new_medoids)
                
                silhouette = silhouette_score(numeric_df, cluster_labels)
                davies_bouldin = davies_bouldin_score(numeric_df, cluster_labels)
                
                medoid_points = numeric_df.iloc[medoid_indices].values.tolist()
                
                result_data = {
                    'cluster_labels': cluster_labels.tolist(),
                    'medoid_indices': medoid_indices.tolist(),
                    'medoid_points': medoid_points,
                    'silhouette_score': float(silhouette),
                    'davies_bouldin_score': float(davies_bouldin),
                    'n_clusters': n_clusters,
                    'algorithm': 'kmedoid'
                }
                
                clustering_result = ClusteringResult.objects.create(
                    dataset=dataset,
                    algorithm='kmedoid',
                    parameters=parameters,
                    n_clusters=n_clusters,
                    silhouette_score=silhouette,
                    davies_bouldin_score=davies_bouldin,
                    cluster_labels=cluster_labels.tolist(),
                    result_data=result_data
                )
            
            else:
                return Response({'error': 'Unsupported clustering algorithm'}, status=status.HTTP_400_BAD_REQUEST)
            
            return Response({
                'id': clustering_result.id,
                'result': result_data
            })
            
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=True, methods=['post'])
    def association_rules(self, request, pk=None):
        """Perform association rule mining on the dataset"""
        dataset = self.get_object()
        parameters = request.data.get('parameters', {})
        
        try:
            if dataset.file_type == 'csv':
                df = pd.read_csv(dataset.file.path)
            else:
                df = pd.read_excel(dataset.file.path)
            
            min_support = parameters.get('min_support', 0.1)
            min_confidence = parameters.get('min_confidence', 0.5)
            
            # Convert data to binary format (one-hot encoding for categorical columns)
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            if len(categorical_cols) == 0:
                return Response({'error': 'No categorical columns found for association rules'}, status=status.HTTP_400_BAD_REQUEST)
            
            # Create binary matrix
            binary_df = pd.get_dummies(df[categorical_cols], prefix=categorical_cols)
            
            # Find frequent itemsets using simple apriori
            frequent_itemsets = []
            rules = []
            
            # Generate 1-itemsets
            itemset_support = {}
            for col in binary_df.columns:
                support = (binary_df[col] == 1).sum() / len(binary_df)
                if support >= min_support:
                    itemset_support[frozenset([col])] = support
                    frequent_itemsets.append({
                        'itemset': [col],
                        'support': float(support)
                    })
            
            # Generate k-itemsets
            current_itemsets = list(itemset_support.keys())
            k = 2
            while current_itemsets and k <= 3:  # Limit to 3-itemsets for performance
                new_itemsets = []
                for i in range(len(current_itemsets)):
                    for j in range(i + 1, len(current_itemsets)):
                        union = current_itemsets[i] | current_itemsets[j]
                        if len(union) == k:
                            # Calculate support
                            mask = binary_df[list(union)].all(axis=1)
                            support = mask.sum() / len(binary_df)
                            if support >= min_support:
                                itemset_support[union] = support
                                new_itemsets.append(union)
                                frequent_itemsets.append({
                                    'itemset': list(union),
                                    'support': float(support)
                                })
                
                current_itemsets = new_itemsets
                k += 1
            
            # Generate rules
            for itemset, support in itemset_support.items():
                if len(itemset) >= 2:
                    for item in itemset:
                        antecedent = frozenset([item])
                        consequent = itemset - antecedent
                        
                        if antecedent in itemset_support:
                            confidence = support / itemset_support[antecedent]
                            if confidence >= min_confidence:
                                lift = confidence / itemset_support.get(consequent, 0.01)
                                rules.append({
                                    'antecedent': list(antecedent),
                                    'consequent': list(consequent),
                                    'support': float(support),
                                    'confidence': float(confidence),
                                    'lift': float(lift)
                                })
            
            result_data = {
                'frequent_itemsets': frequent_itemsets,
                'rules': rules,
                'min_support': min_support,
                'min_confidence': min_confidence,
                'n_itemsets': len(frequent_itemsets),
                'n_rules': len(rules)
            }
            
            association_result = AssociationRuleResult.objects.create(
                dataset=dataset,
                min_support=min_support,
                min_confidence=min_confidence,
                frequent_itemsets=frequent_itemsets,
                rules=rules,
                result_data=result_data
            )
            
            return Response({
                'id': association_result.id,
                'result': result_data
            })
            
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=True, methods=['post'])
    def webmining(self, request, pk=None):
        """Perform web mining algorithms (PageRank, HITS) on the dataset"""
        dataset = self.get_object()
        algorithm = request.data.get('algorithm')
        parameters = request.data.get('parameters', {})
        
        try:
            if dataset.file_type == 'csv':
                df = pd.read_csv(dataset.file.path)
            else:
                df = pd.read_excel(dataset.file.path)
            
            # Expect columns: 'source' and 'target' for graph edges
            if 'source' not in df.columns or 'target' not in df.columns:
                return Response({'error': 'Dataset must have "source" and "target" columns for web mining'}, status=status.HTTP_400_BAD_REQUEST)
            
            # Create directed graph
            G = nx.DiGraph()
            for _, row in df.iterrows():
                G.add_edge(str(row['source']), str(row['target']))
            
            result_data = {}
            
            if algorithm == 'pagerank':
                alpha = parameters.get('alpha', 0.85)
                max_iter = parameters.get('max_iter', 100)
                
                pagerank_scores = nx.pagerank(G, alpha=alpha, max_iter=max_iter)
                
                # Sort by score
                sorted_scores = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
                
                result_data = {
                    'algorithm': 'pagerank',
                    'scores': dict(sorted_scores[:20]),  # Top 20 nodes
                    'all_scores': pagerank_scores,
                    'alpha': alpha,
                    'max_iter': max_iter,
                    'n_nodes': G.number_of_nodes(),
                    'n_edges': G.number_of_edges()
                }
                
            elif algorithm == 'hits':
                max_iter = parameters.get('max_iter', 100)
                
                hubs, authorities = nx.hits(G, max_iter=max_iter)
                
                # Sort by score
                sorted_hubs = sorted(hubs.items(), key=lambda x: x[1], reverse=True)
                sorted_authorities = sorted(authorities.items(), key=lambda x: x[1], reverse=True)
                
                result_data = {
                    'algorithm': 'hits',
                    'hubs': dict(sorted_hubs[:20]),  # Top 20 hubs
                    'authorities': dict(sorted_authorities[:20]),  # Top 20 authorities
                    'all_hubs': hubs,
                    'all_authorities': authorities,
                    'max_iter': max_iter,
                    'n_nodes': G.number_of_nodes(),
                    'n_edges': G.number_of_edges()
                }
            
            else:
                return Response({'error': 'Unsupported web mining algorithm'}, status=status.HTTP_400_BAD_REQUEST)
            
            webmining_result = WebMiningResult.objects.create(
                dataset=dataset,
                algorithm=algorithm,
                parameters=parameters,
                scores=result_data.get('scores') or result_data.get('hubs') or {},
                result_data=result_data
            )
            
            return Response({
                'id': webmining_result.id,
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

class ClusteringResultViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = ClusteringResult.objects.all()
    serializer_class = ClusteringResultSerializer

class AssociationRuleResultViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = AssociationRuleResult.objects.all()
    serializer_class = AssociationRuleResultSerializer

class WebMiningResultViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = WebMiningResult.objects.all()
    serializer_class = WebMiningResultSerializer
