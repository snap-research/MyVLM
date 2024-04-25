import os
import random
from enum import Enum, auto

import numpy as np
import torch


class ConceptType(Enum):
    OBJECT = auto()
    PERSON = auto()


class VLMType(str, Enum):
    BLIP2 = 'blip-2'
    LLAVA = 'llava'
    MINIGPT_V2 = 'minigpt-v2'


class PersonalizationTask(str, Enum):
    CAPTIONING = 'captioning'
    VQA = 'vqa'
    REC = 'rec'


class MyVLMLayerMode(Enum):
    TRAIN = auto()
    INFERENCE = auto()


VLM_TO_LAYER = {
    VLMType.BLIP2: "model.vision_model.encoder.layers[38].mlp.fc2",
    VLMType.LLAVA: "model.model.mm_projector.linear2",
    VLMType.MINIGPT_V2: "model.llama_proj"
}

VLM_TO_EMBEDDING_DIM = {
    VLMType.BLIP2: 1408,
    VLMType.LLAVA: 4096,
    VLMType.MINIGPT_V2: 4096
}

VLM_TO_PROMPTS = {
    VLMType.BLIP2: {
        PersonalizationTask.CAPTIONING: [''],
    },
    VLMType.LLAVA: {
        PersonalizationTask.CAPTIONING: ['Please caption this image of {concept}.'],
        PersonalizationTask.VQA: [
            'Where is {concept} in the image?',
            'Where is {concept} positioned in the image?',
            'What is {concept} doing in the image?',
        ],
    },
    VLMType.MINIGPT_V2: {
        PersonalizationTask.CAPTIONING: [
            '[caption] A short image caption of {concept}:',
            '[refer] {concept} in the image'
        ],
    }
}

VALID_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

CLIP_MODEL_NAME = "DFN5B-CLIP-ViT-H-14-384"
MINIGPT_V2_CKPT_PATH = "/path/to/minigptv2_checkpoint.pth"
HF_TOKEN_FOR_LLAMA = 'IF WORKING WITH MINIGPT_V2, ENTER YOUR HF TOKEN FOR DOWNLOAIDNG LLAMA WEIGHTS'


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
