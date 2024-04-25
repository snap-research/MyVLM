import torch
from PIL import Image
from pathlib import Path
from transformers import Blip2Processor
from typing import Dict, Union, List, Optional

from vlms.blip2.modeling_blip_2 import Blip2ForConditionalGeneration
from vlms.vlm_wrapper import VLMWrapper


class BLIP2Wrapper(VLMWrapper):

    def __init__(self, device: str = 'cuda', torch_dtype: torch.dtype = torch.bfloat16):
        self.model_path = "Salesforce/blip2-flan-t5-xl"
        super().__init__(device, torch_dtype)

    def set_model(self):
        processor = Blip2Processor.from_pretrained(self.model_path)
        model = Blip2ForConditionalGeneration.from_pretrained(self.model_path,
                                                              torch_dtype=self.torch_dtype,
                                                              device_map="auto")
        try:  # Seems like we need to wrap this in a try-except sometimes. Not really sure why, but it seems to work.
            model = model.to(self.device, self.torch_dtype)
        except:
            pass
        return model, processor

    def preprocess(self, image_path: Path, prompt: str) -> Dict:
        inputs = self.processor(
            images=[Image.open(image_path)],
            text=[prompt],
            return_tensors="pt"
        ).to(self.device, self.torch_dtype)
        return inputs

    def generate(self, inputs: Dict, concept_signals: Optional[torch.Tensor] = None) -> Union[str, List[str]]:
        if concept_signals is not None:
            inputs['concept_signals'] = self.prepare_concept_signals(concept_signals=concept_signals)
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return output
