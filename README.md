# MyVLM: Personalizing VLMs for User-Specific Queries 

> Recent large-scale vision-language models (VLMs) have demonstrated remarkable capabilities in understanding and generating textual descriptions for visual content. However, these models lack an understanding of user-specific concepts. In this work, we take a first step toward the personalization of VLMs, enabling them to learn and reason over user-provided concepts. For example, we explore whether these models can learn to recognize you in an image and communicate what you are doing, tailoring the model to reflect your personal experiences and relationships. To effectively recognize a variety of user-specific concepts, we augment the VLM with external concept heads that function as toggles for the model, enabling the VLM the identify the presence of specific target concepts in a given image. Having recognized the concept, we learn a new concept embedding in the intermediate feature space of the VLM. This embedding is tasked with guiding the language model to naturally integrate the target concept in its generated response. We apply our technique to BLIP-2 and LLaVA for personalized image captioning and further show its applicability for personalized visual question-answering. Our experiments demonstrate our ability to generalize to unseen images of learned concepts while preserving the model behavior on unrelated inputs.

<a href=""><img src="https://img.shields.io/badge/arXiv-b31b1b.svg"></a>
<a href="https://snap-research.github.io/MyVLM/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a> 
<a href="https://www.youtube.com/watch?v=8Fy17kK_aZY&ab_channel=YuvalAlaluf"><img src="https://img.shields.io/static/v1?label=&message=Video&color=darkgreen" height=20.5></a> 

<p align="center">
<img src="docs/teaser.jpg" width="800px"/>  
<br>
Given a set of images depicting user-specific concepts such as &lt;you&gt;, &lt;your-dog&gt;, and &lt;your-friend&gt; (left), we teach a pretrained vision-language model (VLM) to understand and reason over these concepts. First, we enable the model to generate personalized captions incorporating the concept into its output text (middle). We further allow the user to ask subject-specific questions about these concepts, querying the model with questions such as "What are &lt;you&gt; doing?" or "What is my &lt;your-friend&gt; wearing?" (right).
</p>

# Code and Data Coming Soon!

## Citation
If you use this code for your research, please cite the following work:
```
@misc{

}
```
