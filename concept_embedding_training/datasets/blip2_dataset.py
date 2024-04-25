from typing import Dict, Any, List

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from transformers import Blip2Processor


class BLIP2Dataset(Dataset):
    """
    Dataset for BLIP personalization. For simplicity, this currently only supports captioning, but may be
    extended to support other tasks in the future by changing the _get_target function.
    """
    def __init__(self,
                 inputs: Dict[str, Any],
                 target_labels: List[str],
                 processor: Blip2Processor = None,
                 transforms: torchvision.transforms.Compose = None,
                 device: str = 'cuda',
                 torch_dtype: torch.dtype = torch.bfloat16):
        self.inputs = {k: v.to('cpu') if type(v) == torch.Tensor else v for k, v in inputs.items()}
        self.images = inputs['images']
        self.target_labels = target_labels
        self.processor = processor
        self.transforms = transforms
        self.device = device
        self.torch_dtype = torch_dtype

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        batch = {k: v[idx] for k, v in self.inputs.items()}
        batch['label_text'] = self._get_target(idx)
        image = self.images[idx]
        if self.transforms is not None:
            image = self.transforms(image)
        if self.processor is not None:
            inputs = self.processor(images=image, text='', return_tensors="pt")
            image = inputs.data['pixel_values'][0]
        batch['pixel_values'] = image
        return batch

    def _get_target(self, idx: int) -> str:
        if type(self.target_labels[idx]) == str:
            self.target_labels[idx] = [self.target_labels[idx]]
        sampled_target = np.random.choice(self.target_labels[idx], size=1, replace=False)[0]
        return sampled_target
