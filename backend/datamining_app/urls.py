from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'datasets', views.DatasetViewSet)
router.register(r'preprocessing-results', views.PreprocessingResultViewSet)
router.register(r'classification-results', views.ClassificationResultViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
