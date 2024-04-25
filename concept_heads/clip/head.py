import pathlib
import torch
import torch.nn.functional as F
from PIL import Image
from open_clip import create_model_from_pretrained
from pathlib import Path
from torch import Tensor
from tqdm import tqdm
from typing import List, Dict, Optional, Union, Tuple

from concept_heads.clip.concept_head_training.model import CLIPLinearClassifier
from concept_heads.concept_head import ConceptHead

MODEL_NAME = "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384"


class CLIPConceptHead(ConceptHead):

    def __init__(self,
                 checkpoint: Optional[Union[Path, List[Path]]] = None,
                 concept_idx: int = 0,
                 device: torch.device = torch.device('cuda')):
        self.device = device
        self.model, self.preprocess = self._load_base_model(device=self.device)
        self.value_idx = concept_idx
        self.checkpoint = checkpoint
        if self.checkpoint is not None:
            self.models_list = self._load_linear_heads(checkpoint)

    def extract_signal(self, image_paths: List[Path]) -> Dict[Path, Tensor]:
        path_to_embed = {}
        for path in tqdm(image_paths):
            image = Image.open(path).convert("RGB")
            with torch.no_grad():
                if self.checkpoint is None:
                    path_to_embed[path] = self._extract_base_model_embeddings(image)
                else:
                    path_to_embed[path] = self._extract_classifiers_probabilities(image)
        return path_to_embed

    def _load_base_model(self, device: torch.device = torch.device('cuda')) -> Tuple[torch.nn.Module, torch.nn.Module]:
        model, preprocess = create_model_from_pretrained(MODEL_NAME, precision='fp16')
        model.to(device)
        model.eval()
        return model, preprocess

    def _load_linear_heads(self, checkpoint: Union[Path, List[Path]]) -> List[CLIPLinearClassifier]:
        """
        This technically saves the base model multiple times, but it's the easiest way to load the models.
        If runtime is important, this can be optimized.
        """
        if type(checkpoint) in [pathlib.PosixPath, str]:
            checkpoint = [checkpoint]
        models_list = []
        for ckpt in checkpoint:
            ckpt = torch.load(ckpt, map_location=self.device)
            model = CLIPLinearClassifier(model=self.model).cuda()
            model.load_state_dict(ckpt, strict=False)
            model.eval()
            models_list.append(model)
        return models_list

    def _extract_base_model_embeddings(self, image: Image.Image) -> Tensor:
        image = self.preprocess(image)
        with torch.cuda.amp.autocast():
            features = self.model.encode_image(image.unsqueeze(0).to(self.device))
            features = F.normalize(features, dim=-1)
        return features

    def _extract_classifiers_probabilities(self, image: Image.Image) -> Dict[int, Tensor]:
        image = self.preprocess(image)
        image_probas = {}
        for model_idx, model in enumerate(self.models_list):
            with torch.cuda.amp.autocast():
                outputs = model.forward(image.unsqueeze(0).to(self.device))
                probas = torch.softmax(outputs, dim=1)
            if len(self.models_list) == 1:
                # If we only have a single concept head, then the key should be the index of the target concept
                return {self.value_idx: probas}
            else:
                # Otherwise, we'll want to store the probabilities for each concept head
                image_probas[model_idx] = probas
        return image_probas
