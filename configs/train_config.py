from dataclasses import dataclass
from pathlib import Path

import torch

from myvlm.common import ConceptType, VLMType, PersonalizationTask


@dataclass
class EmbeddingTrainingConfig:
    """
    This config defines various parameters for training the concept embedding.
    Please see example_configs/concept_embedding_training_captioning.yaml for an example of how to use this config.
    """
    # Name of the concept to train the embedding for
    concept_name: str
    # The concept identifier that we want to use for the concept (e.g., sks)
    concept_identifier: str
    # Type of the concept we wish to train
    concept_type: ConceptType
    # Which underlying VLM we want to use
    vlm_type: VLMType
    # Which personalization task you want to optimize for
    personalization_task: PersonalizationTask
    # Path to save the results to. This will be saved under the concept_name folder
    output_root: Path = Path('./outputs')
    # Path to the dataset to use for training the concept embedding. This should contain a directory for each concept.
    data_root: Path = Path('./data')
    # Where are the linear heads for the object concepts saved? This should contain a directory for each concept.
    concept_head_path: Path = Path('./object_concept_heads')
    # Which checkpoint from the classifier training, should we take if working with objects?
    classifier_step: int = 500
    # Threshold value to determine if the concept is present in the image. For objects: 0.5, For people: 0.675
    threshold: float = 0.5
    # Number of optimization steps for learning the concept embedding. See paper for more details.
    optimization_steps: int = 100
    # Learning rate for optimizing the embedding
    learning_rate: float = 1.0
    # Batch size to use for concept_embedding_training the concept embedding. In the paper, we use 4 by default.
    batch_size: int = 4
    # Regularization lambda value for the attention regularization loss
    # For BLIP-2 Captioning: 0.04
    # For LLaVA Captioning: 0.0075
    # For LLaVA VQA: 10.0
    # For MiniGPT-v2: 1.0
    reg_lambda: float = 0.04
    # Interval for saving the concept embeddings during concept_embedding_training
    save_interval: int = 25
    # Interval for generating intermediate personalized outputs during concept_embedding_training
    val_interval: int = 25
    # Seed
    seed: int = 42
    # Device to train on
    device: str = 'cuda'
    # Torch dtype to use
    torch_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        self.concept_data_path = self.data_root / self.concept_name
        assert self.concept_data_path.exists(), \
            f"Data path {self.concept_data_path} does not exist!"
        self.concept_head_path = self.concept_head_path / self.concept_name / f'seed_{self.seed}'
        if self.concept_type == ConceptType.OBJECT:
            assert self.concept_head_path.exists(), \
                f"Concept head path {self.concept_head_path} does not exist!"
        self.output_path = self.output_root / self.concept_name / f'seed_{self.seed}'
        self.output_path.mkdir(exist_ok=True, parents=True)

