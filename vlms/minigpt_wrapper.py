import re
from argparse import Namespace
from pathlib import Path
from typing import Dict, Optional, Union, List

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

from vlms.minigpt4.common.config import Config as MiniGPTConfig
from vlms.minigpt4.common.registry import registry
from vlms.minigpt4.conversation.conversation import CONV_VISION_minigptv2
from vlms.vlm_wrapper import Processor, VLMWrapper


class MiniGPTWrapper(VLMWrapper):

    def __init__(self, device: str = 'cuda', torch_dtype: torch.dtype = torch.bfloat16):
        self.cfg_path = 'vlms/minigpt4/configs/minigptv2_eval.yaml'
        super().__init__(device, torch_dtype)

    def set_model(self):
        args = Namespace()
        args.cfg_path = self.cfg_path
        args.gpu_id = 0
        args.options = None
        minigpt_cfg = MiniGPTConfig(args)
        device = 'cuda:{}'.format(args.gpu_id)
        model_config = minigpt_cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to(device)
        vis_processor_cfg = minigpt_cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        processor = Processor(tokenizer=model.llama_tokenizer, image_processor=vis_processor)
        return model, processor

    def prepare_inputs(self, image: Image.Image, prompt: str, target: str) -> Dict:
        image_tensor = self.processor.image_processor(image).to(self.device, self.torch_dtype)
        instruction = f"<Img><ImageHere></Img> {prompt}"
        input_dict = {
            "original_image": image,
            "image": image_tensor,
            "answer": target,
            "instruction_input": instruction,
        }
        return input_dict

    def preprocess(self, image_path: Path, prompt: str) -> Dict:
        input_dict = {
            'image': self._prepare_image_tensor(Image.open(image_path)),
            'instruction_input': self._prepare_language_instruction(prompt)
        }
        return input_dict

    def generate(self, inputs: Dict, concept_signals: Optional[torch.Tensor] = None) -> Union[str, List[str]]:
        if concept_signals is not None:
            inputs['concept_signals'] = self.prepare_concept_signals(concept_signals=concept_signals)
        output = self.model.generate(images=inputs['image'].unsqueeze(0),
                                     texts=[inputs['instruction_input']],
                                     concept_signals=inputs.get('concept_signals', None),
                                     max_new_tokens=128,
                                     do_sample=False)
        return output

    def _prepare_image_tensor(self, image: Image.Image) -> torch.Tensor:
        image_tensor = self.processor.image_processor(image).to(self.device, self.torch_dtype)
        return image_tensor

    def _prepare_language_instruction(self, prompt: str) -> str:
        conv_temp = CONV_VISION_minigptv2.copy()
        conv_temp.system = ""
        conv_temp.append_message(conv_temp.roles[0], "<Img><ImageHere></Img>")
        conv_temp.append_message(conv_temp.roles[0], prompt)
        conv_temp.append_message(conv_temp.roles[1], None)
        prompt = conv_temp.get_prompt()
        return prompt

    def draw_and_save_referring_localization_result(self,
                                                    image_tensor: torch.tensor,
                                                    result_str: str,
                                                    output_path: Path):
        image_tensor = image_tensor.cpu()
        reverse_norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
        reverse_norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
        image_tensor = image_tensor * reverse_norm_std + reverse_norm_mean
        pil_img = torchvision.transforms.ToPILImage()(image_tensor)
        bounding_box_size = 100
        image_height, image_width = 448, 448
        integers = [int(i) for i in re.findall(r'-?\d+', result_str)]
        if len(integers) == 4:
            x0, y0, x1, y1 = int(integers[0]), int(integers[1]), int(integers[2]), int(integers[3])
            left = x0 / bounding_box_size * image_width
            bottom = y0 / bounding_box_size * image_height
            right = x1 / bounding_box_size * image_width
            top = y1 / bounding_box_size * image_height
            color = (0, 255, 0)
            new_image = cv2.rectangle(np.array(pil_img), (int(left), int(bottom)), (int(right), int(top)), color, 2)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            Image.fromarray(new_image).save(output_path)
