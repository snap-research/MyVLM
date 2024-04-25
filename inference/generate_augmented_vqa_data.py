import json
from dataclasses import dataclass
from pathlib import Path

import pyrallis
import torch
from tqdm import tqdm

from myvlm.common import VALID_IMAGE_EXTENSIONS, ConceptType
from vlms.llava_wrapper import LLaVAWrapper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUESTIONS_FOR_OBJECTS = [
    'What color is {}?',
    'Where is {} in the image?',
    'Where is {} positioned in the image?',
    'Does {} appear to be the main subject of the image?',
    'What objects is {} interacting with in the image?',
    'How would you describe the texture of {} in the image?',
    'What types of materials is {} be made of?',
    'Is {} large or small in the image?',
    'Is {} close to the camera or far away?',
]
QUESTIONS_FOR_PEOPLE = [
    'What is {} wearing in the image?',
    'What color shirt is {} wearing?',
    'What is {} doing in the image?',
    'Where is {} in the image?',
    'Can you describe what {} is wearing?',
    'From left to right, where is {} positioned in the image?',
    'What kind of hair does {} have?',
    'What is the expression on {} face?',
    'Is there anything unique about {}\'s appearance?'
]


@dataclass
class GenerationConfig:
    # The name of the concept to augment
    concept_name: str
    # The identifier that was used for concept_embedding_training the concept
    concept_identifier: str
    # The general class of the concept.
    # For example, if the concept is the rainbow cat statue, the class is 'the cat statue'.
    # For people, you can use "the man" or "the woman", for example.
    # This is used to guide the VLM to answer the question on the target subject.
    concept_class: str
    # Path to the images of the concept. We will save the additional questions and answers in this directory.
    # Type of the concept - either people or objects. We use this to determine which questions to ask
    concept_type: ConceptType
    # Directory containing the images we want to run inference on
    images_root: Path
    # Torch dtype
    torch_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        self.image_paths = [str(p) for p in self.images_root.glob('*') if p.suffix.lower() in VALID_IMAGE_EXTENSIONS]
        self.questions = QUESTIONS_FOR_PEOPLE if self.concept_type == ConceptType.PERSON else QUESTIONS_FOR_OBJECTS


@pyrallis.wrap()
def generate_vqa_questions_for_llava(cfg: GenerationConfig):
    """
    This is used to generate additional questions and answers for concept_embedding_training LLaVA for personalized VQA. For simplicity,
    we support only LLaVA, but this can be extended to other VLMs by using their corresponding wrappers.
    """
    vlm_wrapper = LLaVAWrapper(device=DEVICE, torch_dtype=cfg.torch_dtype)

    if (cfg.images_root / f'additional_llava_vqa_data.json').exists():
        with open(cfg.images_root / f'additional_llava_vqa_data.json', 'r') as f:
            path_to_questions_answers = json.load(f)
    else:
        path_to_questions_answers = {}

    for image_path in tqdm(cfg.image_paths):
        if image_path in path_to_questions_answers:
            continue

        path_to_questions_answers[image_path] = []
        for question in cfg.questions:
            # When asking questions, use the general class to help the VLM model answer the question
            question = question.format(cfg.concept_class)
            inputs = vlm_wrapper.preprocess(image_path, prompt=question)
            answer = vlm_wrapper.generate(inputs, concept_signals=None)[0].lower()
            # Save teh question and answer using the concept's identifier to be used during concept_embedding_training
            path_to_questions_answers[image_path].append([
                question.replace(cfg.concept_class, cfg.concept_identifier),
                answer.replace(cfg.concept_class, cfg.concept_identifier)
            ])

    with open(cfg.images_root / f'additional_llava_vqa_data.json', 'w') as f:
        json.dump(path_to_questions_answers, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    generate_vqa_questions_for_llava()
