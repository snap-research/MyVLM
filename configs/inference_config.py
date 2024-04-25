import pathlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import torch

from myvlm.common import ConceptType, VLMType, PersonalizationTask, VLM_TO_PROMPTS, VALID_IMAGE_EXTENSIONS


@dataclass
class InferenceConfig:
    # Name of the concept we wish to run inference on
    concept_name: str
    # The identifier that was used for concept_embedding_training the concept
    concept_identifier: str
    # Which type of concept is this? Person or object?
    concept_type: ConceptType
    # Which VLM you are running inference on
    vlm_type: VLMType
    # Which personalization task you want to run inference on
    personalization_task: PersonalizationTask
    # List of image paths we wish to run inference on. If a path is given, we iterate over this directory
    image_paths: Union[Path, List[str]]
    # Where are the concept embedding checkpoints saved to? This should contain a directory for each concept.
    checkpoint_path: Path = Path('./outputs')
    # Where to save the results to
    output_path: Path = Path('./outputs')
    # Where are the linear heads for the object concepts saved? This should contain a directory for each concept.
    concept_head_path: Path = Path('./object_concept_heads')
    # Which step of the concept head to use if working with objects
    classifier_step: int = 500
    # Defines which seed to use for the concept head and for the concept embeddings
    seed: int = 42
    # Which iterations to run inference on. If None, will run on all iterations that were saved
    iterations: Optional[List[int]] = None
    # List of prompts to use for inference. If None, we will use the default set of prompts defined in `common/common.py`
    prompts: Optional[List[str]] = None
    # Device to train on
    device: str = 'cuda'
    # Torch dtype
    torch_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        if not self.concept_head_path.name.startswith('seed_'):
            self.concept_head_path = self.concept_head_path / self.concept_name / f'seed_{self.seed}'

        if not self.checkpoint_path.name.startswith('seed_'):
            self.checkpoint_path = self.checkpoint_path / self.concept_name / f'seed_{self.seed}'

        self._verify_concept_embeddings_exist()
        if self.concept_type == ConceptType.OBJECT:
            self._verify_concept_heads_exist()

        self.inference_output_path = self.output_path / self.concept_name / 'inference_outputs' / f'seed_{self.seed}'
        self.inference_output_path.mkdir(parents=True, exist_ok=True)

        if type(self.image_paths) == pathlib.PosixPath and self.image_paths.is_dir():
            self.image_paths = [str(p) for p in self.image_paths.glob('*') if p.suffix in VALID_IMAGE_EXTENSIONS]

        # Set the threshold value for recognizing the concept
        self.threshold = 0.5 if self.concept_type == ConceptType.OBJECT else 0.675

        # Get the prompts. If None is given, then we use the default list for each VLM and task
        if self.prompts is None:
            self.prompts = VLM_TO_PROMPTS[self.vlm_type].get(self.personalization_task, None)
            if self.prompts is None:
                raise ValueError(f"Prompts for task {self.personalization_task} are not defined for {self.vlm_type}!")

    def _verify_concept_heads_exist(self):
        if not self.concept_head_path.exists() and self.concept_type == ConceptType.OBJECT:
            raise ValueError(f"Concept head path {self.concept_head_path} does not exist!")

    def _verify_concept_embeddings_exist(self):
        if not self.checkpoint_path.exists():
            raise ValueError(f"Concept checkpoint path {self.checkpoint_path} does not exist!")
