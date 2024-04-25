from abc import ABC, abstractmethod
from pathlib import Path
from torch import Tensor
from typing import List, Dict


class ConceptHead(ABC):

    @abstractmethod
    def extract_signal(self, image_paths: List[Path]) -> Dict[Path, Tensor]:
        """
        Defines the signal used to determine if the concept is present in a given image or not.
        For faces, this is the face embedding, while for objects this is the logits obtained from our linear classifier.
        """
        raise NotImplementedError
