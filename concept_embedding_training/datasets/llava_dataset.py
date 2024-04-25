import copy
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

from vlms.llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from vlms.llava.conversation import conv_templates
from vlms.llava.mm_utils import tokenizer_image_token
from vlms.vlm_wrapper import Processor


class LLaVADataset(Dataset):
    """
    Dataset for LLaVA personalization. This currently supports both captioning and vqa tasks. For VQA,
    additional VQA data should be provided, which includes the image paths and the prompts and answers for each image.
    The concept name should also be provided for VQA tasks since this will be used in the prompts.
    """

    def __init__(self,
                 inputs: Dict[str, Any],
                 target_labels: List[str],
                 processor: Processor = None,
                 transforms: torchvision.transforms.Compose = None,
                 additional_vqa_data: Optional[Dict] = None,
                 concept_identifier: str = None,
                 device: str = 'cuda',
                 torch_dtype: torch.dtype = torch.bfloat16):
        self.inputs = inputs
        self.images = inputs['images']
        self.target_labels = target_labels
        self.processor = processor
        self.transforms = transforms
        self.concept_name = concept_identifier
        self.additional_vqa_data = additional_vqa_data
        self.torch_dtype = torch_dtype
        self.device = device
        if self.additional_vqa_data is not None:
            assert self.concept_name is not None, "Concept name must be provided if additional VQA data is provided."

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        batch = {k: v[idx] for k, v in self.inputs.items()}
        # Redefine the input ids and labels after performing augmentations on the inputs and targets
        input_ids, targets, attention_mask = self._get_target(idx, batch)
        batch['input_ids'] = input_ids.squeeze(0)
        batch['labels'] = targets.squeeze(0)
        batch['attention_mask'] = attention_mask.squeeze(0)
        # Perform augmentations on the input image and preprocess according to LLaVA
        image = self.images[idx]
        if self.transforms is not None:
            image = self.transforms(image)
        if self.processor is not None:
            inputs = self.processor.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
            inputs = inputs.to(self.device, self.torch_dtype)
            batch['images'] = inputs[0]
        return batch

    def _get_target(self, idx: int, batch: Dict[str, Any]) -> str:
        if type(self.target_labels[idx]) == str:
            self.target_labels[idx] = [self.target_labels[idx]]

        if self.additional_vqa_data is not None and len(self.additional_vqa_data['image_paths']) > 0:
            q_idx = self.additional_vqa_data['image_paths'].index(str(batch['image_paths']))
            additional_questions = self.additional_vqa_data['questions_and_answers'][q_idx]
        else:
            additional_questions = []

        # Add the standard captioning language instruction
        additional_questions += [(f'Please caption this image of {self.concept_name}.', self.target_labels[idx][0])]

        # Sample a random instruction and corresponding target
        sampled_idx = np.random.choice(len(additional_questions), size=1, replace=False)[0]
        prompt = additional_questions[sampled_idx][0]
        if prompt == f'Please caption this image of {self.concept_name}.':
            # If we got the captioning instruction, randomly select one of the possible captions
            sampled_target = np.random.choice(self.target_labels[idx], size=1, replace=False)[0]
        else:
            # Otherwise, we have a single pre-defined answer to the sampled query
            sampled_target = additional_questions[sampled_idx][1]

        # Now we need to encode the instruction and prepare the new targets
        input_ids, targets, attention_mask = self._encode_instruction_and_target(prompt, sampled_target)

        return input_ids, targets, attention_mask

    def _encode_instruction_and_target(self, prompt: str, sampled_target: str):
        inp = DEFAULT_IMAGE_TOKEN + '\n' + f'{prompt}'
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt() + f" {sampled_target}"
        input_ids = tokenizer_image_token(
            prompt,
            self.processor.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0).to(self.device)

        sep = conv.sep + conv.roles[1] + ": "
        parts = prompt.split(sep)
        parts[0] += sep
        instruction_len = len(tokenizer_image_token(parts[0], self.processor.tokenizer)) - 2

        target_ids = tokenizer_image_token(
            sampled_target,
            self.processor.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0).to(self.device)

        # For the target, need to set everything up to the target equal to -100
        targets = copy.deepcopy(input_ids)
        targets[:, :1] = -100
        for i in range(target_ids.shape[0]):
            targets[i, 1:1 + instruction_len] = -100

        # Get the new attention mask
        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)

        return input_ids, targets, attention_mask
