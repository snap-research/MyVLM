import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

from concept_heads.concept_head import ConceptHead
from configs.train_config import EmbeddingTrainingConfig
from myvlm.common import ConceptType


@dataclass
class Data:
    paths: List[Path]
    images: List[Image.Image]
    concept_signals: List[Union[torch.Tensor, Dict]]
    targets: List[str]


@dataclass
class EmbeddingData:
    pass


def load_data(concept_head: ConceptHead, cfg: EmbeddingTrainingConfig) -> Dict[str, Data]:
    """
    Loads the train and validation data for the concept embedding training. This includes:
    1. The image paths
    2. The images
    3. The signals obtained from the concept head indicating the presence of the concept in the images
    4. The targets (e.g., captions) for the images
    """
    # Load the augmented set of captions if they exist - this will contain multiple target captions per image
    if (cfg.concept_data_path / 'captions_augmented.json').exists():
        captions_path = cfg.concept_data_path / 'captions_augmented.json'
    else:
        captions_path = cfg.concept_data_path / 'captions.json'

    with open(captions_path, 'r') as f:
        image_path_to_captions = json.load(f)

    image_paths = [cfg.concept_data_path / p for p in image_path_to_captions.keys()]

    # Insert the concept identifier into the captions by replacing the concept_name with concept_identifier
    target_captions = []
    for captions in list(image_path_to_captions.values()):
        name_to_replace = cfg.concept_name if cfg.concept_type == ConceptType.OBJECT else cfg.concept_name.title()
        target_captions.append([caption.replace(name_to_replace, cfg.concept_identifier) for caption in captions])

    # Randomly sample batch_size images for concept_embedding_training
    train_paths = np.random.choice(image_paths, size=min(len(image_paths), cfg.batch_size), replace=False).tolist()
    train_targets = [target_captions[image_paths.index(path)] for path in train_paths]
    train_images = [Image.open(path) for path in train_paths]

    # Get the remaining images for validation
    val_paths = list(set(image_paths) - set(train_paths))
    val_targets = [target_captions[image_paths.index(path)] for path in val_paths]
    val_images = [Image.open(path) for path in val_paths]

    # Extract the concept head signals for all the images
    train_signals = concept_head.extract_signal(train_paths)

    # If we're working with people, we need to filter out embeddings of other people that may appear in  train images
    train_signals = [train_signals[path] for path in train_paths]

    if cfg.concept_type == ConceptType.PERSON:
        train_signals = filter_out_other_people_embeddings(train_signals, cfg)

    val_signals = concept_head.extract_signal(val_paths)
    val_signals = [val_signals[path] for path in val_paths]

    data_dict = {
        'train': Data(paths=train_paths, images=train_images, concept_signals=train_signals, targets=train_targets),
        'val': Data(paths=val_paths, images=val_images, concept_signals=val_signals, targets=val_targets)
    }
    return data_dict


def filter_out_other_people_embeddings(face_embeds: List[torch.Tensor],
                                       cfg: EmbeddingTrainingConfig) -> List[torch.Tensor]:
    """
    For each image, if multiple faces are detected, we filter out the embeddings that are likely to belong to other
    individuals. This is done as follows.
    1. We first identify the images that contain only a single individual. We assume that this is our target.
       Note: we assume that the target subject appears alone at least once.
    2. For the remaining images, we filter out the embeddings that are not close to the embedding of the target subject.
    """
    # Initialize an empty list for the final filtered output
    filtered_embeds = []
    # Single person embeddings concatenated for comparison
    single_person_embeds_cat = torch.cat([e for e in face_embeds if e is not None and e.shape[0] == 1])
    # Iterate through the original face_embeds list
    for embed in face_embeds:
        if embed is None:
            filtered_embeds.append(None)
        elif embed.shape[0] == 1:
            # This is a single embedding, add it directly to the final list
            filtered_embeds.append(embed)
        else:
            # This is a multi-person embedding, process it
            is_person_close = []
            for m in embed:
                # Check if this person is close to at least one single person embedding
                similarity = F.cosine_similarity(m.unsqueeze(0), single_person_embeds_cat, dim=1)
                is_close = torch.any(similarity > (1 - cfg.threshold)).item()
                is_person_close.append(is_close)

            is_person_close_tensor = torch.tensor(is_person_close, dtype=torch.bool)

            # Error handling based on your logic
            if torch.sum(is_person_close_tensor) == 0:
                raise ValueError("The target subject was not identified in an image!")
            if torch.sum(is_person_close_tensor) > 1:
                raise ValueError("Multiple people were identified as the target subject!")

            # Filter the multi-person embedding and add to the final list
            filtered_multi_embed = embed[is_person_close_tensor]
            filtered_embeds.append(filtered_multi_embed)

    return filtered_embeds


def load_additional_vqa_data(cfg: EmbeddingTrainingConfig) -> Dict[str, List]:
    """
    Load the additional VQA data that will be used for training the concept embedding for personalized VQA.
    These were created using the script generate_augmented_vqa_data.py.
    Note that we need to replace the concept name with the concept identifier in the questions and answers.
    """
    with open(cfg.concept_data_path / 'additional_llava_vqa_data.json', 'r') as f:
        additional_vqa_data = json.load(f)
    image_paths = list(additional_vqa_data.keys())
    questions_and_answers = []
    for image_samples in list(additional_vqa_data.values()):
        reformated_samples = []
        for q, a in image_samples:
            reformated_samples.append((q.replace(cfg.concept_name, cfg.concept_identifier),
                                       a.replace(cfg.concept_name, cfg.concept_identifier)))
        questions_and_answers.append(reformated_samples)

    return {'image_paths': image_paths, 'questions_and_answers': questions_and_answers}


def get_image_transforms():
    """ Get the image transforms that will be applied during the optimization of the concept embeddings. """
    image_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ColorJitter(brightness=0.6),
        torchvision.transforms.RandomRotation(degrees=45),
    ])
    return image_transforms


def cosine_distance(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    # Cosine distance
    if len(key.shape) < 2:
        key = key.view(1, -1)
    return 1 - F.cosine_similarity(key.float(), query.float(), dim=1)
