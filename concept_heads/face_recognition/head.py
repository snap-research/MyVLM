import numpy as np
import torch
from PIL import Image
from insightface.app import FaceAnalysis
from pathlib import Path
from torch import Tensor
from tqdm import tqdm
from typing import List, Dict

from concept_heads.concept_head import ConceptHead


class FaceConceptHead(ConceptHead):

    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def extract_signal(self, image_paths: List[Path]) -> Dict[Path, Tensor]:
        output = {}
        for path in tqdm(image_paths):
            image = Image.open(path).convert("RGB")
            faces = self.app.get(np.array(image))
            if len(faces) == 0:
                output[path] = None
            else:
                embeddings = torch.stack([torch.from_numpy(f.normed_embedding) for f in faces])
                output[path] = embeddings
        return output
