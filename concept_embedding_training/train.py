import json
from typing import Dict, List

import pyrallis
import torch
from PIL import Image

from concept_embedding_training import data_utils
from concept_embedding_training.data_utils import Data, EmbeddingData, load_data
from concept_heads.clip.head import CLIPConceptHead
from concept_heads.face_recognition.head import FaceConceptHead
from configs.inference_config import InferenceConfig
from configs.train_config import EmbeddingTrainingConfig
from inference.run_myvlm_inference import run_inference
from myvlm import myblip2, myllava, myminigpt_v2
from myvlm.common import ConceptType, VLM_TO_LAYER, seed_everything, VLM_TO_PROMPTS, PersonalizationTask, VLMType, \
    CLIP_MODEL_NAME
from myvlm.myvlm import MyVLM
from vlms.blip2_wrapper import BLIP2Wrapper
from vlms.llava_wrapper import LLaVAWrapper
from vlms.minigpt_wrapper import MiniGPTWrapper

VLM_TYPE_TO_WRAPPER = {
    VLMType.BLIP2: BLIP2Wrapper,
    VLMType.LLAVA: LLaVAWrapper,
    VLMType.MINIGPT_V2: MiniGPTWrapper
}
VLM_TYPE_TO_MYVLM = {
    VLMType.BLIP2: myblip2.MyBLIP2,
    VLMType.LLAVA: myllava.MyLLaVA,
    VLMType.MINIGPT_V2: myminigpt_v2.MyMiniGPT_v2
}


@pyrallis.wrap()
def main(cfg: EmbeddingTrainingConfig):
    # Save the config to the output path for future reference
    seed_everything(cfg.seed)
    with (cfg.output_path / f'config_{cfg.vlm_type}_{cfg.personalization_task}.yaml').open('w') as f:
        pyrallis.dump(cfg, f)
    print('\n' + pyrallis.dump(cfg))

    # Load the VLM wrapper
    vlm_wrapper = VLM_TYPE_TO_WRAPPER[cfg.vlm_type](device=cfg.device, torch_dtype=cfg.torch_dtype)

    # Load the concept head that was previously trained
    if cfg.concept_type == ConceptType.OBJECT:
        head_path = cfg.concept_head_path / f'{CLIP_MODEL_NAME}-{cfg.concept_name}-step-{cfg.classifier_step}.pt'
        concept_head = CLIPConceptHead(head_path)
    else:
        concept_head = FaceConceptHead()

    # Load the data
    data_dict = load_data(concept_head, cfg)

    # If we're doing VQA with LLaVA, load the additional vqa data that we pre-generated with generate_augmented_vqa_data
    additional_vqa_data = None
    if cfg.personalization_task == PersonalizationTask.VQA and cfg.vlm_type == VLMType.LLAVA:
        if (cfg.concept_data_path / 'additional_llava_vqa_data.json').exists():
            additional_vqa_data = data_utils.load_additional_vqa_data(cfg=cfg)

    # Get the MyVLM model and give it the VLM we want to personalize
    myvlm = VLM_TYPE_TO_MYVLM[cfg.vlm_type](vlm=vlm_wrapper,
                                            layer=VLM_TO_LAYER[cfg.vlm_type],
                                            concept_name=cfg.concept_name,
                                            cfg=cfg)

    concept_embedding_checkpoints = train_concept_embedding(myvlm=myvlm,
                                                            data_dict=data_dict,
                                                            cfg=cfg,
                                                            additional_vqa_data=additional_vqa_data)
    torch.save(concept_embedding_checkpoints, cfg.output_path /
               f'concept_embeddings_{cfg.vlm_type}_{cfg.personalization_task}.pt')

    # Run inference on the validation samples after concept_embedding_training the concept embedding
    inference_config = InferenceConfig(
        concept_name=cfg.concept_name,
        concept_identifier=cfg.concept_identifier,
        concept_type=cfg.concept_type,
        vlm_type=cfg.vlm_type,
        personalization_task=PersonalizationTask.CAPTIONING,
        image_paths=data_dict['val'].paths,
        checkpoint_path=cfg.output_path,
        concept_head_path=cfg.concept_head_path,
        prompts=VLM_TO_PROMPTS[cfg.vlm_type][cfg.personalization_task],
        device=cfg.device,
        torch_dtype=cfg.torch_dtype
    )
    outputs = run_inference(myvlm,
                            concept_signals=data_dict['val'].concept_signals,
                            iteration_to_concept_data=concept_embedding_checkpoints,
                            iterations=list(concept_embedding_checkpoints.keys()),
                            cfg=inference_config)
    with open(cfg.output_path / f'inference_outputs_{cfg.vlm_type}_{cfg.personalization_task}.json', 'w') as f:
        json.dump(outputs, f, indent=4)


def train_concept_embedding(myvlm: MyVLM,
                            data_dict: Dict[str, Data],
                            cfg: EmbeddingTrainingConfig,
                            additional_vqa_data: Dict[str, List] = None) -> Dict[str, EmbeddingData]:
    # Let's get the original VLM answer over the concept_embedding_training images
    train_paths = data_dict['train'].paths
    train_prompt = VLM_TO_PROMPTS[cfg.vlm_type][PersonalizationTask.CAPTIONING][0].format(concept=cfg.concept_identifier)
    for path in train_paths:
        input_dict = myvlm.vlm.preprocess(image_path=path, prompt=train_prompt)
        output = myvlm.vlm.generate(inputs=input_dict, concept_signals=None)[0]
        print(f"Image: {path.stem} | Original VLM Answer: {output}")

    inputs = {
        'image_paths': train_paths,
        'images': [Image.open(path).convert("RGB") for path in train_paths],
        'concept_signals': data_dict['train'].concept_signals
    }
    if type(inputs['concept_signals']) == torch.Tensor:
        inputs['concept_signals'] = torch.stack(inputs['concept_signals']).to(cfg.device)

    # Train the concept embedding
    concept_embedding_checkpoints = myvlm.train_embedding(
        inputs=inputs,
        target_labels=data_dict['train'].targets,
        image_transforms=data_utils.get_image_transforms(),
        additional_vqa_data=additional_vqa_data
    )

    # Run inference on the train samples after optimization
    print("-" * 100)
    for idx, path in enumerate(train_paths):
        input_dict = myvlm.vlm.preprocess(image_path=path, prompt=train_prompt)
        signal = data_dict['train'].concept_signals[idx]
        output = myvlm.vlm.generate(inputs=input_dict, concept_signals=signal)[0]
        print(f"Image: {path.stem} | Personalized Answer: {output}")
        print("-" * 100)

    return concept_embedding_checkpoints


if __name__ == '__main__':
    main()
