from typing import Tuple, List, Dict, Any, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose
from tqdm import tqdm
from transformers import Blip2Processor

from configs.train_config import EmbeddingTrainingConfig
from myvlm.common import MyVLMLayerMode, VLM_TO_EMBEDDING_DIM
from myvlm.myvlm_layer import MyVLMLayer
from myvlm.utils import brackets_to_periods, parent_module
from vlms.vlm_wrapper import VLMWrapper, Processor


class MyVLM(torch.nn.Module):

    def __init__(self,
                 vlm: VLMWrapper,
                 layer: str,
                 concept_name: str,
                 cfg: EmbeddingTrainingConfig,
                 device: str = 'cuda'):
        super().__init__()
        self.vlm = vlm
        self.layer = layer
        self.concept_name = concept_name
        self.cfg = cfg
        self.device = device

        for n, p in self.vlm.model.named_parameters():
            p.requires_grad = False

        edit_module = parent_module(self.vlm, brackets_to_periods(self.layer))
        layer_name = self.layer.rsplit(".", 1)[-1]
        original_layer = getattr(edit_module, layer_name)
        setattr(edit_module, layer_name,
                MyVLMLayer(layer=original_layer,
                           embedding_dim=VLM_TO_EMBEDDING_DIM[cfg.vlm_type],
                           threshold=cfg.threshold,
                           torch_dtype=cfg.torch_dtype,
                           device=self.device))

    def generate(self, *args, **kwargs):
        return self.vlm.generate(*args, **kwargs)

    def train_embedding(self, inputs, target_labels, image_transforms=None, additional_vqa_data=None):

        self.inputs = inputs
        setattr(eval(f"self.vlm.{self.layer}"), "training", True)
        setattr(eval(f"self.vlm.{self.layer}"), "mode", MyVLMLayerMode.TRAIN)

        pbar = tqdm(range(self.cfg.optimization_steps))
        train_dataset, val_dataset = self._init_datasets(inputs=inputs,
                                                         target_labels=target_labels,
                                                         processor=self.vlm.processor,
                                                         image_transforms=image_transforms,
                                                         additional_vqa_data=additional_vqa_data)
        train_dataloader, val_dataloader = self._init_dataloaders(train_dataset, val_dataset, self.cfg)

        optimizer, scheduler = None, None
        checkpoints_dict = {}
        for i in pbar:
            setattr(eval(f"self.vlm.{self.layer}"), "iter", i)

            for batch_idx, batch in enumerate(train_dataloader):

                batch['output_attentions'] = True
                outputs = self.vlm.model(**batch)

                if optimizer is None:
                    # After the first pass, we have initialized the embedding so we can create the optimizer
                    optimizer = torch.optim.AdamW(self.vlm.model.parameters(),
                                                  lr=self.cfg.learning_rate,
                                                  weight_decay=1e-4)
                    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, self.cfg.learning_rate)

                loss = outputs.loss

                reg_loss = 0.
                if self.cfg.reg_lambda > 0:
                    reg_loss = self._compute_regularization_loss(outputs)
                    loss += reg_loss

                loss.backward()

                parameters = eval(f"self.vlm.{self.layer}").values
                torch.nn.utils.clip_grad_norm_(parameters, 0.05, norm_type=2)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                pbar.set_description(f"Loss: {loss:0.3f} | Reg Loss: {reg_loss:0.3f}")

                if self._should_validate(i, batch_idx):
                    self.validate(val_dataloader)

                if self._should_save_checkpoint(i, batch_idx):
                    checkpoints_dict[i] = {
                        "keys": eval(f"self.vlm.{self.layer}").keys.clone().detach().requires_grad_(False),
                        "values": eval(f"self.vlm.{self.layer}").values.clone().detach().requires_grad_(False),
                    }
                    checkpoints_dict[i] = {k: v.cpu() for k, v in checkpoints_dict[i].items()}

        print("*" * 80)
        print("Finished concept_embedding_training concept embedding!")
        print("*" * 80)
        setattr(eval(f"self.vlm.{self.layer}"), "mode", MyVLMLayerMode.INFERENCE)
        return checkpoints_dict

    def validate(self, val_dataloader):
        raise NotImplementedError

    def _compute_regularization_loss(self, outputs) -> torch.Tensor:
        """ Apply attention-based regularization to encourage sparsity over the concept embeddings. """
        raise NotImplementedError

    def _init_datasets(self,
                       inputs: Dict[str, Any],
                       target_labels: List[List[str]],
                       processor: Union[Blip2Processor, Processor],
                       image_transforms: Compose,
                       additional_vqa_data: Optional[Dict] = None) -> Tuple[Dataset, Dataset]:
        raise NotImplementedError

    def _init_dataloaders(self,
                          train_dataset: Dataset,
                          val_dataset: Dataset,
                          cfg: EmbeddingTrainingConfig) -> Tuple[DataLoader, DataLoader]:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self._collate_func
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,  # Best to keep this as 1 for easiest compatibility across all VLMs
            shuffle=False,
            num_workers=0,
            collate_fn=self._collate_func
        )
        return train_dataloader, val_dataloader

    def _collate_func(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError

    def _should_validate(self, i: int, batch_idx: int) -> bool:
        return i == self.cfg.optimization_steps - 1 or (i % self.cfg.val_interval == 0 and i > 0 and batch_idx == 0)

    def _should_save_checkpoint(self, i: int, batch_idx: int) -> bool:
        return i == self.cfg.optimization_steps - 1 or (i % self.cfg.save_interval == 0 and i > 0 and batch_idx == 0)
