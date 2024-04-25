import torch
from abc import abstractmethod, ABC
from pathlib import Path

from torch import nn
from transformers import AutoImageProcessor, AutoTokenizer
from typing import NamedTuple, Dict, Union, List, Optional


class Processor(NamedTuple):
    tokenizer: AutoTokenizer
    image_processor: AutoImageProcessor


class VLMWrapper(nn.Module):

    def __init__(self, device: str = 'cuda', torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.device = device
        self.torch_dtype = torch_dtype
        self.model, self.processor = self.set_model()

    @abstractmethod
    def set_model(self):
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, image_path: Path, prompt: str) -> Dict:
        raise NotImplementedError

    def prepare_concept_signals(self, concept_signals: torch.Tensor):
        if type(concept_signals) == torch.Tensor:
            concept_signals = concept_signals.unsqueeze(0).to(self.device)
        elif concept_signals is not None:
            concept_signals = [concept_signals]
        return concept_signals

    @abstractmethod
    def generate(self, inputs: Dict, concept_signals: Optional[torch.Tensor] = None) -> Union[str, List[str]]:
        raise NotImplementedError
