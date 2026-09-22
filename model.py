import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def create_model(num_classes=6):
    """
    Create a ResNet50 model for six-class
    canine/feline thoracic radiograph classification.
    """

    # Load ImageNet-pretrained ResNet50
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    # Replace the original 1000-class classifier
    # with a 6-class classifier
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model