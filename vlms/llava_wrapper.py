import copy
from pathlib import Path
from typing import Dict, Union, List, Tuple, NamedTuple, Optional, Any

import torch
from PIL import Image

from vlms.llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from vlms.llava.conversation import conv_templates, SeparatorStyle, Conversation
from vlms.llava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from vlms.llava.model.builder import load_pretrained_model
from vlms.vlm_wrapper import VLMWrapper, Processor


class LLaVAInput(NamedTuple):
    image_tensor: torch.Tensor
    input_ids: torch.Tensor
    targets: Optional[torch.Tensor]
    attention_mask: torch.Tensor
    stopping_criteria: KeywordsStoppingCriteria


class LLaVAWrapper(VLMWrapper):

    def __init__(self, device: str = 'cuda', torch_dtype: torch.dtype = torch.bfloat16):
        self.model_path = 'liuhaotian/llava-v1.6-vicuna-7b'
        self.temperature = 0.2
        self.top_p = 0.7
        super().__init__(device, torch_dtype)

    def set_model(self):
        model_name = get_model_name_from_path(self.model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(self.model_path,
                                                                     model_base=None,
                                                                     model_name=model_name,
                                                                     load_8bit=False,
                                                                     load_4bit=True,
                                                                     device=self.device)
        processor = Processor(tokenizer=tokenizer, image_processor=image_processor)
        model.prepare_mm_projector_for_grace()
        try:  # Seems like we need to wrap this in a try-except sometimes. Not really sure why, but it seems to work.
            model = model.to(self.device)
        except:
            pass
        return model, processor

    def prepare_inputs(self, image: Image.Image, prompt: str, target: str) -> LLaVAInput:
        image_tensor = self._prepare_image_tensor(image)
        input_ids, prompt, conv = self._prepare_input_ids(prompt, target)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.processor.tokenizer, input_ids)
        targets = self._prepare_targets(prompt, target, input_ids, conv)
        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)
        return LLaVAInput(image_tensor, input_ids, targets, attention_mask, stopping_criteria)

    def preprocess(self, image_path: Path, prompt: str, target: str = '') -> Dict[str, Any]:
        temp_inputs = self.prepare_inputs(image=Image.open(image_path), prompt=prompt, target=target)
        generation_inputs = self._prepare_inputs_for_generation(input_ids=temp_inputs.input_ids,
                                                                targets=temp_inputs.targets,
                                                                image_tensor=temp_inputs.image_tensor,
                                                                stopping_criteria=temp_inputs.stopping_criteria)
        return generation_inputs

    def generate(self, inputs: Dict[str, Any], concept_signals: Optional[torch.Tensor] = None) -> Union[str, List[str]]:
        if concept_signals is not None:
            concept_signals = self.prepare_concept_signals(concept_signals=concept_signals)
        with torch.cuda.amp.autocast():
            output = self.model.generate(inputs=inputs['input_ids'],
                                         images=inputs['image_tensor'],
                                         concept_signals=concept_signals,
                                         do_sample=True if self.temperature > 0 else False,
                                         temperature=self.temperature,
                                         top_p=self.top_p,
                                         stopping_criteria=[inputs['stopping_criteria']],
                                         max_new_tokens=512,
                                         output_attentions=False,
                                         return_dict_in_generate=True)
        output = self.processor.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        return output

    def _prepare_image_tensor(self, image: Image.Image) -> torch.Tensor:
        """ Preprocesses the input image using the vision encoder's preprocessor. """
        image_tensor = self.processor.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
        image_tensor = image_tensor.to(self.device, self.torch_dtype)
        return image_tensor

    def _prepare_input_ids(self, prompt: str, target: str) -> Tuple[torch.Tensor, str, Conversation]:
        """ Encode the language instruction using the tokenizer. """
        inp = DEFAULT_IMAGE_TOKEN + '\n' + f'{prompt}'
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt() + f" {target}"
        input_ids = tokenizer_image_token(
            prompt,
            self.processor.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0).to(self.device)
        return input_ids, prompt, conv

    def _prepare_targets(self, prompt: str, target: str, input_ids: torch.Tensor, conv: Conversation) -> torch.Tensor:
        """ Prepare the target ids for optimization. """
        sep = conv.sep + conv.roles[1] + ": "
        parts = prompt.split(sep)
        parts[0] += sep
        instruction_len = len(tokenizer_image_token(parts[0], self.processor.tokenizer)) - 2
        target_ids = tokenizer_image_token(
            target,
            self.processor.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0).to(self.device)
        # For the target, need to set everything up to the target equal to -100
        targets = copy.deepcopy(input_ids)
        targets[:, :1] = -100
        for i in range(target_ids.shape[0]):
            targets[i, 1:1 + instruction_len] = -100
        return targets

    def _prepare_inputs_for_generation(self,
                                       input_ids: torch.Tensor,
                                       targets: torch.Tensor,
                                       image_tensor: torch.Tensor,
                                       stopping_criteria) -> Dict[str, Any]:
        """ Prepares the input tensors for the generate function by removing the target from the input_ids. """
        target_len = [len(t[t != -100]) for t in targets]
        inputs_no_target = [input_id[:-t_len] for input_id, t_len in zip(input_ids, target_len)]
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            inputs_no_target,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id
        )
        generation_inputs = dict(
            image_tensor=image_tensor,
            input_ids=input_ids_padded,
            targets=None,
            attention_mask=input_ids_padded.ne(self.processor.tokenizer.pad_token_id),
            stopping_criteria=stopping_criteria
        )
        return generation_inputs
