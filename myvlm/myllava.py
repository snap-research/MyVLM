from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from tqdm import tqdm

from concept_embedding_training.datasets.llava_dataset import LLaVADataset
from myvlm.myvlm import MyVLM
from vlms.llava.conversation import conv_templates, SeparatorStyle
from vlms.llava.mm_utils import KeywordsStoppingCriteria
from vlms.vlm_wrapper import Processor


class MyLLaVA(MyVLM):

    def validate(self, val_dataloader, temperature: float = 0.2, top_p: float = 0.7):
        conv = conv_templates["llava_v1"].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        outputs = []
        for val_batch in tqdm(val_dataloader):
            stopping_criteria = KeywordsStoppingCriteria(keywords,
                                                         self.vlm.processor.tokenizer,
                                                         val_batch['input_ids'])
            target_len = [len(t[t != -100]) for t in val_batch['labels']]
            input_ids = val_batch['input_ids']
            masks = val_batch['attention_mask']
            max_len = max(target_len)
            input_ids = torch.stack([in_id[:-max_len] for in_id in input_ids])
            attention_masks = torch.stack([mask[:-max_len] for mask in masks])
            generated_ids = self.vlm.model.generate(inputs=input_ids,
                                                    images=val_batch['images'],
                                                    concept_signals=val_batch['concept_signals'],
                                                    attention_mask=attention_masks,
                                                    stopping_criteria=[stopping_criteria],
                                                    do_sample=True,
                                                    temperature=temperature,
                                                    top_p=top_p,
                                                    max_new_tokens=512)
            output = self.vlm.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            output = [out.strip() for out in output]
            outputs.append(output[0])
        print('')
        for output in outputs:
            print(f"Personalized Output: {output}")
        print('-' * 100)

    def _compute_regularization_loss(self, outputs) -> torch.Tensor:
        """ Apply L2 regularization to encourage sparsity over the self attention of the learned concept embedding. """
        reg_losses = []
        for probas in outputs.attentions:
            for sample_idx in range(self.cfg.batch_size):
                reg_loss = self.cfg.reg_lambda * \
                           torch.mean(probas[sample_idx, :, outputs.concept_token_idxs[sample_idx], :] ** 2)
                reg_losses.append(reg_loss)
        reg_loss = sum(reg_losses)
        return reg_loss

    def _init_datasets(self,
                       inputs: Dict[str, Any],
                       target_labels: List[List[str]],
                       processor: Processor,
                       image_transforms: Compose,
                       additional_vqa_data: Optional[Dict] = None) -> Tuple[Dataset, Dataset]:
        """ Define the train and validation datasets. Note that this is the same set of images for both. """
        train_dataset = LLaVADataset(
            inputs.copy(),
            target_labels=target_labels,
            processor=processor,
            transforms=image_transforms,
            additional_vqa_data=additional_vqa_data,
            concept_identifier=self.cfg.concept_identifier,
        )
        val_dataset = LLaVADataset(
            inputs.copy(),
            target_labels=target_labels,
            processor=processor,
            transforms=None,
            additional_vqa_data=additional_vqa_data,
            concept_identifier=self.cfg.concept_identifier,
        )
        return train_dataset, val_dataset

    def _collate_func(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ Need to join the list of samples into a single dictionary to be passed to the VLM. """
        joined_batch = {}

        joined_batch['input_ids'] = torch.nn.utils.rnn.pad_sequence(
            [b['input_ids'] for b in batch],
            batch_first=True,
            padding_value=self.vlm.processor.tokenizer.pad_token_id
        )
        joined_batch['attention_mask'] = joined_batch['input_ids'].ne(self.vlm.processor.tokenizer.pad_token_id)
        joined_batch['images'] = torch.stack([b['images'] for b in batch])
        joined_batch['labels'] = torch.nn.utils.rnn.pad_sequence(
            [b['labels'] for b in batch],
            batch_first=True,
            padding_value=-100
        )
        joined_batch['concept_signals'] = [b['concept_signals'] for b in batch]
        if type(joined_batch['concept_signals'][0]) == torch.Tensor:
            joined_batch['concept_signals'] = torch.stack(joined_batch['concept_signals']).to(device=self.device,
                                                                                              dtype=self.vlm.torch_dtype)

        return joined_batch
