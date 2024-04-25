from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from tqdm import tqdm

from myvlm.common import seed_everything
from concept_heads.clip.concept_head_training import datasets
from concept_heads.clip.concept_head_training.config import ConceptHeadTrainingConfig
from concept_heads.clip.concept_head_training.datasets import ClassificationDataset
from concept_heads.clip.concept_head_training.model import ClassificationWrapper


class Coach:
    """
    Coach for concept_embedding_training linear classifiers on top of CLIP features.
    """

    def __init__(self, cfg: ConceptHeadTrainingConfig):
        self.cfg = cfg
        seed_everything(self.cfg.seed)
        self.device = torch.device('cuda')
        self.classifier = self._init_model()
        self.train_dataloader, self.val_dataloader = self._init_datasets()
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        self.criterion = self._init_criterion()

    def train(self):
        """ Simple concept_embedding_training loop for concept_embedding_training our linear classifiers. """
        self.classifier.train()
        self.global_step = 0
        while self.global_step < self.cfg.max_steps:
            for step, batch in enumerate(self.train_dataloader):
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)

                with torch.cuda.amp.autocast():
                    outputs = self.classifier.forward(images)

                loss = torch.mean(self.criterion(outputs, labels))
                loss.backward()

                parameters = [p for p in self.classifier.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(parameters, 0.1, norm_type=2)

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                if self.global_step % 10 == 0:
                    print(f"Step: {self.global_step} | Loss: {loss.item()}")
                    with open(self.cfg.output_dir / 'log.txt', 'a') as f:
                        f.write(f"Step: {self.global_step} | Loss: {loss.item()}\n")

                if self.global_step % 50 == 0:
                    self.validate()

                if self.global_step % 100 == 0:
                    self._save_checkpoint()

                if self.global_step > self.cfg.max_steps:
                    print("Finished concept_embedding_training!")
                    self._save_checkpoint()
                    break

    def validate(self):
        """
        Run validation on the validation images. We'll also store some useful information regarding the probabilities
        obtained for the positive and negative samples.
        """
        print("Running validation...")
        self.classifier.eval()
        positive_correct, positive_total = 0, 0
        negative_correct, negative_total = 0, 0
        positive_probabilities, negative_probabilities = [], []
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader):
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)

                with torch.cuda.amp.autocast():
                    outputs = self.classifier.forward(images)

                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probabilities, 1)

                positive_correct += (predicted[labels == 1] == labels[labels == 1]).sum().item()
                positive_total += labels[labels == 1].size(0)
                negative_correct += (predicted[labels == 0] == labels[labels == 0]).sum().item()
                negative_total += labels[labels == 0].size(0)
                positive_probabilities.extend(probabilities[labels == 1, 1].cpu().numpy())
                negative_probabilities.extend(probabilities[labels == 0, 1].cpu().numpy())

        print(f"Test | Step: {self.global_step} | Positive Accuracy: {100 * positive_correct / positive_total}")
        print(f"Test | Step: {self.global_step} | Negative Accuracy: {100 * negative_correct / negative_total}")

        with open(self.cfg.output_dir / 'log.txt', 'a') as f:
            f.write(f"Test | Step: {self.global_step} | Positive Accuracy: {100 * positive_correct / positive_total}\n")
            f.write(f"Test | Step: {self.global_step} | Negative Accuracy: {100 * negative_correct / negative_total}\n")
            f.write(f"Positive probabilities: {positive_probabilities}\n")
            f.write(f"Average negative positive probabilities: {np.mean(negative_probabilities)}\n")
            f.write(f"Max negative positive probabilities: {np.max(negative_probabilities)}\n")
        self.classifier.train()

    def _init_model(self) -> ClassificationWrapper:
        return ClassificationWrapper(model_name=self.cfg.model_name, num_classes=2, device=self.device)

    def _init_datasets(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        paths = datasets.load_and_split_image_paths(positive_image_root=self.cfg.positive_samples_path,
                                                    negative_image_root=self.cfg.negative_samples_path,
                                                    n_positive_samples=self.cfg.n_positive_samples)
        # Let's save the validation paths for later
        with open(self.cfg.output_dir / 'val_paths.txt', 'w') as f:
            for path in paths.val_positive_paths + paths.val_negative_paths:
                f.write(f"{path}\n")

        train_dataset = ClassificationDataset(
            positive_paths=paths.train_positive_paths,
            negative_paths=paths.train_negative_paths,
            processor=self.classifier.preprocess,
            transforms=datasets.get_train_image_transforms()
        )
        val_dataset = ClassificationDataset(
            positive_paths=paths.val_positive_paths,
            negative_paths=paths.val_negative_paths,
            processor=self.classifier.preprocess,
            transforms=None
        )
        # Use weighted sampling for concept_embedding_training
        weights = [1 / len(paths.train_positive_paths)] * len(paths.train_positive_paths) + \
                  [1 / len(paths.train_negative_paths)] * len(paths.train_negative_paths)
        sampler = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=len(weights))
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=sampler,
            shuffle=sampler is None,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            drop_last=True,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            drop_last=False,
        )
        return train_dataloader, val_dataloader

    def _init_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.classifier.parameters(), lr=self.cfg.learning_rate, weight_decay=0.01)

    def _init_scheduler(self) -> torch.optim.lr_scheduler:
        return lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                              T_max=self.cfg.max_steps,
                                              eta_min=self.cfg.target_lr)

    def _init_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss(reduction='none')

    def _save_checkpoint(self):
        """ Save only the trained parameters of the classifier (i.e., the linear head). """
        ckpt = {}
        model_params = {k: v for k, v in self.classifier.named_parameters()}
        for name, param in self.classifier.state_dict().items():
            if model_params[name].requires_grad:
                ckpt[name.replace("model.", "")] = param
        torch.save(ckpt, self.cfg.output_dir / f"{self.cfg.model_name.split('/')[1]}-"
                                               f"{self.cfg.concept_name}-step-{self.global_step}.pt")
