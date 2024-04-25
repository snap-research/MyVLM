import torch
import torch.nn.functional as F
from open_clip import create_model_from_pretrained
from torch import nn


class ClassificationWrapper(nn.Module):

    def __init__(self, model_name: str, num_classes: int = 2, device: torch.device = torch.device('cuda')):
        super().__init__()
        self.model_name = model_name
        model, self.preprocess = create_model_from_pretrained(model_name, precision='fp16')
        self.model = CLIPLinearClassifier(model=model, num_labels=num_classes, hidden_size=1024)
        self.model.to(device)
        self._print_trainable_parameters()

    def _print_trainable_parameters(self):
        print("Trainable parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CLIPLinearClassifier(nn.Module):

    def __init__(self, model: nn.Module, num_labels: int = 2, hidden_size: int = 1024):
        super().__init__()
        self.num_labels = num_labels
        self.base = model
        self.classifier = nn.Linear(hidden_size, num_labels)
        for name, param in self.base.named_parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base.encode_image(x)
        features = F.normalize(features, dim=-1)
        logits = self.classifier(features)
        return logits
