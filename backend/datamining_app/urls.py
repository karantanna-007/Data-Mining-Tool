from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'datasets', views.DatasetViewSet)
router.register(r'preprocessing-results', views.PreprocessingResultViewSet)
router.register(r'classification-results', views.ClassificationResultViewSet)
router.register(r'clustering-results', views.ClusteringResultViewSet)
router.register(r'association-results', views.AssociationRuleResultViewSet)
router.register(r'webmining-results', views.WebMiningResultViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
