from typing import Dict

import torch

from configs.inference_config import InferenceConfig
from myvlm.common import VLM_TO_LAYER, MyVLMLayerMode
from vlms.vlm_wrapper import VLMWrapper


def load_concept_embeddings(seed: int, cfg: InferenceConfig) -> Dict[int, Dict]:
    """ Load the dictionary saved during the concept_embedding_training of the concept embeddings. """
    ckpt_path = cfg.concept_checkpoint_path / f'checkpoints_{cfg.concept_name}_seed_{seed}.pt'
    iteration_to_concept_embeddings = torch.load(ckpt_path, map_location='cpu')
    return iteration_to_concept_embeddings


def set_concept_embeddings(vlm_wrapper: VLMWrapper,
                           concept_embeddings: Dict[str, torch.Tensor],
                           iteration: int,
                           cfg: InferenceConfig):
    """ Adds the concept embeddings to the VLM model. """
    n_keys = concept_embeddings['keys'].shape[0]
    layer = VLM_TO_LAYER[cfg.vlm_type]
    setattr(eval(f'vlm_wrapper.{layer}'), 'iter', iteration)
    setattr(eval(f"vlm_wrapper.{layer}"), "mode", MyVLMLayerMode.INFERENCE)
    setattr(eval(f'vlm_wrapper.{layer}'), 'key_idx_to_value_idx', {k: 0 for k in range(n_keys)})
    setattr(eval(f'vlm_wrapper.{layer}'), 'keys', concept_embeddings['keys'].cuda())
    setattr(eval(f'vlm_wrapper.{layer}'), 'values', torch.nn.Parameter(concept_embeddings['values'].cuda()))
