from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConceptHeadTrainingConfig:
    # Name of the concept we want to train the linear head for
    concept_name: str
    # Output directory for the trained model and logs
    output_dir: Path = Path("./concept_head_training")
    # Path to the positive samples
    positive_samples_path: Path = Path("./data/positives")
    # Path to the negative samples
    negative_samples_path: Path = Path("./data/negatives")
    # The base feature extractor we want to extract features for
    model_name: str = "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384"
    # Number of positive samples to use for concept_embedding_training
    n_positive_samples: int = 4
    # Maximum number of concept_embedding_training steps
    max_steps: int = 500
    # Initial learning rate
    learning_rate: float = 1e-3
    # Final learning rate
    target_lr: float = 5e-6
    # Batch size for concept_embedding_training
    batch_size: int = 16
    # Number of workers for data loading
    num_workers: int = 6
    # Seed for reproducibility
    seed: int = 42

    def __post_init__(self):
        self.output_dir = self.output_dir / self.concept_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.positive_samples_path = self.positive_samples_path / self.concept_name
        self.negative_samples_path = self.negative_samples_path / self.concept_name
        # Make sure the data paths exist and are not empty
        assert self.positive_samples_path.exists() and len(list(self.positive_samples_path.glob("*"))) > 0
        assert self.negative_samples_path.exists() and len(list(self.negative_samples_path.glob("*"))) > 0