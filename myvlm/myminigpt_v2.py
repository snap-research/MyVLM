from typing import List, Dict, Any, Optional

import torch
from torchvision.transforms import Compose
from tqdm import tqdm

from concept_embedding_training.datasets.minigpt_v2_dataset import MiniGPTv2Dataset
from myvlm.myvlm import MyVLM
from vlms.minigpt4.conversation.conversation import CONV_VISION_minigptv2
from vlms.vlm_wrapper import Processor


class MyMiniGPT_v2(MyVLM):

    def validate(self, val_dataloader):
        outputs = []
        for val_batch in tqdm(val_dataloader):
            conv_temp = CONV_VISION_minigptv2.copy()
            conv_temp.system = ""
            conv_temp.append_message(conv_temp.roles[0], "<Img><ImageHere></Img>")
            conv_temp.append_message(conv_temp.roles[0], val_batch['samples']['instruction_input'][0].split('</Img>')[1])
            conv_temp.append_message(conv_temp.roles[1], None)
            prompt = conv_temp.get_prompt()
            output = self.vlm.model.generate(images=val_batch['samples']['image'],
                                             texts=[prompt],
                                             max_new_tokens=128,
                                             concept_signals=val_batch['concept_signals'],
                                             do_sample=False)[0].strip()
            outputs.append(output)
        print('')
        for output in outputs:
            print(f"Personalized Output: {output}")
        print('-' * 100)

    def _compute_regularization_loss(self, outputs) -> torch.Tensor:
        reg_losses = []
        for probas in outputs.attentions:
            # Note that this assumes that all language instructions are identical, which is the case here
            # If we change this, then we'll need to change this logic to work per sample
            concept_idx = outputs.concept_embed_idx
            reg_loss = self.cfg.reg_lambda * torch.mean(probas[:, :, concept_idx, :] ** 2)
            reg_losses.append(reg_loss)
        reg_loss = sum(reg_losses)
        return reg_loss

    def _init_datasets(self,
                       inputs: Dict[str, Any],
                       target_labels: List[List[str]],
                       processor: Processor,
                       image_transforms: Compose,
                       additional_vqa_data: Optional[Dict] = None):
        """ Define the train and validation datasets. Note that this is the same set of images for both. """
        train_dataset = MiniGPTv2Dataset(
            inputs.copy(),
            target_labels=target_labels,
            processor=processor,
            transforms=image_transforms,
            concept_name=self.cfg.concept_identifier,
        )
        val_dataset = MiniGPTv2Dataset(
            inputs.copy(),
            target_labels=target_labels,
            processor=processor,
            transforms=None,
            concept_name=self.cfg.concept_identifier,
        )
        return train_dataset, val_dataset

    def _collate_func(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ Need to join the list of samples into a single dictionary to be passed to the VLM. """

        # Join the samples into a single dictionary just like the None collate function
        joined_batch = {
            'image': torch.stack([sample['image'] for sample in batch]),
            'instruction_input': [sample['instruction_input'] for sample in batch],
            'answer': [sample['answer'] for sample in batch],
        }

        concept_signals = [sample['concept_signals'] for sample in batch]
        if type(concept_signals[0]) == torch.Tensor:
            concept_signals = torch.stack(concept_signals).to(self.cfg.device)

        return {'samples': joined_batch, 'concept_signals': concept_signals}

    def _reformat_embeds(self, embeds: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """ This is needed because the collate function merges them into a Dict and we need a list :). """
        new_embeds = []
        for concept_idx in embeds.keys():
            for sample_idx in range(embeds[concept_idx].shape[0]):
                new_embeds.append({concept_idx: embeds[concept_idx][sample_idx]})
        embeds = new_embeds
        return embeds
