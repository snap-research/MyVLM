import torch
from torchvision.transforms import Compose
from tqdm import tqdm
from transformers import Blip2Processor
from typing import List, Dict, Any, Optional

from concept_embedding_training.datasets.blip2_dataset import BLIP2Dataset
from myvlm.myvlm import MyVLM


class MyBLIP2(MyVLM):

    def validate(self, val_dataloader):
        outputs = []
        for val_batch in tqdm(val_dataloader):
            generated_ids = self.vlm.model.generate(**val_batch, max_new_tokens=512)
            output = self.vlm.processor.batch_decode(generated_ids, skip_special_tokens=True)
            output = [out.strip() for out in output]
            outputs.extend(output)
        print('')
        for output in outputs:
            print(f"Personalized Output: {output}")
        print('-' * 100)

    def _compute_regularization_loss(self, outputs) -> torch.Tensor:
        """ Apply L2 regularization to encourage sparsity over the original 32 learnable queries. """
        if outputs.qformer_outputs.cross_attentions is None:
            return torch.tensor(0.0).to(self.device)
        reg_losses = []
        for attention_probs in outputs.qformer_outputs.cross_attentions:
            reg_loss = self.cfg.reg_lambda * torch.mean(attention_probs[:, :, :32, 257:] ** 2)
            reg_losses.append(reg_loss)
        reg_loss = sum(reg_losses)
        return reg_loss

    def _init_datasets(self,
                       inputs: Dict[str, Any],
                       target_labels: List[List[str]],
                       processor: Blip2Processor,
                       image_transforms: Compose,
                       additional_vqa_data: Optional[Dict] = None):
        """ Define the train and validation datasets. Note that this is the same set of images for both. """
        train_dataset = BLIP2Dataset(
            inputs.copy(),
            target_labels=target_labels,
            processor=processor,
            transforms=image_transforms,
        )
        val_dataset = BLIP2Dataset(
            inputs.copy(),
            target_labels=target_labels,
            processor=processor,
            transforms=None,
        )
        return train_dataset, val_dataset

    def _collate_func(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ Need to join the list of samples into a single dictionary to be passed to the VLM. """
        joined_batch = {
            'concept_signals': [b['concept_signals'] for b in batch],
            'pixel_values': torch.stack([b['pixel_values'] for b in batch], dim=0).to(device=self.device,
                                                                                      dtype=self.vlm.torch_dtype),
        }

        if type(batch[0]['concept_signals']) == torch.Tensor:
            joined_batch['concept_signals'] = torch.stack(joined_batch['concept_signals']).to(device=self.device,
                                                                                              dtype=self.vlm.torch_dtype)

        target_encoding = self.vlm.processor.tokenizer([b['label_text'] for b in batch],
                                                       padding="longest",
                                                       return_tensors="pt")
        labels = target_encoding.input_ids
        labels[labels == self.vlm.processor.tokenizer.pad_token_id] = -100
        joined_batch['labels'] = labels.to(self.device)

        input_ids = self.vlm.processor(text=[''] * len(batch), return_tensors="pt")['input_ids']
        joined_batch['input_ids'] = input_ids.to(self.device)

        return joined_batch
