from typing import Dict, Any, List

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

from vlms.vlm_wrapper import Processor


class MiniGPTv2Dataset(Dataset):
    """
    Dataset for MiniGPT_v2 personalization. For simplicity, this currently only supports captioning, but may be
    extended to support other tasks in the future by expanding the instruction_pool defined in the __init__ function
    and adding additional answers as inputs.
    """

    def __init__(self,
                 inputs: Dict[str, Any],
                 target_labels: List[str],
                 processor: Processor = None,
                 transforms: torchvision.transforms.Compose = None,
                 concept_name: str = None,
                 device: str = 'cuda',
                 torch_dtype: torch.dtype = torch.bfloat16):
        self.inputs = inputs
        self.images = inputs['images']
        self.target_labels = target_labels
        self.processor = processor
        self.transforms = transforms
        self.concept_name = concept_name
        self.torch_dtype = torch_dtype
        self.device = device
        # This can be extended to add more instructions during concept_embedding_training. For now, we have the captioning instruction
        self.instruction_pool = [
            f'[caption] A short image caption of {concept_name}:',
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image = self.inputs['images'][idx]
        answer = np.random.choice(self.target_labels[idx], size=1, replace=False)[0]
        instruction = np.random.choice(self.instruction_pool, size=1, replace=False)[0]
        instruction = f'<Img><ImageHere></Img> {instruction}'
        if self.transforms is not None:
            image = self.transforms(image)
        if self.processor is not None:
            image = self.processor.image_processor(image).to(self.device, self.torch_dtype)
        sample = {
            "image": image,
            "answer": answer,
            "instruction_input": instruction,
            "concept_signals": self.inputs['concept_signals'][idx],
        }
        return sample
