## Breaking the Modality Barrier: Universal Embedding Learning with Multimodal LLMs

Tiancheng Gu♥*, Kaicheng Yang♠*, Ziyong Feng♠, Xingjun Wang♣, Yanzhao Zhang♣, Dingkun Long♣, Yingda Chen♣, Weidong Cai♥, Jiankang Deng♦† ♥The University of Sydney ♠DeepGlint ♣Tongyi Lab, Alibaba Group♦Imperial College London tigu8498@uni.sydney.edu.au, kaichengyang@deepglint.com

Project Page Code

[Figure 1]

Hard Negative Enhanced Instruction Tuning

Textual Discriminative Knowledge Distillation

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

# arXiv:2504.17432v4[cs.CV]8Dec2025

###### MLLM

UniME

(a) The Proposed UniME Framework

Retrieval

###### Instruction

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Given an image, find a similar everyday image with the described changes: Reduce the number of puppies to one puppy with stuffed animals.

Query

Classification

Instruction

[Figure 10]

[Figure 11]

Bear

Represent the given image for classification.

[Figure 12]

###### VQA

Instruction

[Figure 13]

[Figure 14]

UniME

Romen

Represent the given image with the following question: What is in the bowl?

[Figure 15]

###### Visual Grounding

Instruction

[Figure 16]

[Figure 17]

[Figure 18]

Search

EVA-CLIP(8B) E5-V(LLaVA-1.6-7B) VLM2Vec(LLaVA-1.6-7B)

Crop the image to to isolate the object labeled as "traffic light"

UniME(Phi3.5-V-4.2B)(Ours) UniME(LLaVA-1.6-7B)(Ours)

(c) Performance Comparision

(b) UniME can Handle Diverse Downstream Tasks

Figure 1: The UniME framework incorporates textual discriminative knowledge distillation and hard negative enhanced instruction tuning stages to learn discriminative representations for diverse downstream tasks. Our framework achieves state-of-the-art performance on both the MMEB benchmark and multiple retrieval tasks.

### Abstract

downstream tasks. In the first stage, we perform textual discriminative knowledge distillation from a powerful LLM-based teacher model to enhance the embedding capability of the MLLM’s language component. In the second stage, we introduce hard negative enhanced instruction tuning to further advance discriminative representation learning. Specifically, we initially mitigate false negative contamination and then sample multiple hard negatives per instance within each batch, forcing the model to focus on challenging samples. This approach not only improves discriminative power but also enhances instruction-following ability in downstream tasks. We conduct extensive experiments on the MMEB benchmark and multiple retrieval tasks, including short&long caption retrieval and compositional retrieval. Results demonstrate that UniME achieves consistent performance improvement across all tasks, exhibiting superior discriminative and compositional capabilities.

The Contrastive Language-Image Pre-training (CLIP) framework has become a widely used approach for multimodal representation learning, particularly in image-text retrieval and clustering. However, its efficacy is constrained by three key limitations: (1) text token truncation, (2) isolated image-text encoding, and (3) deficient compositionality due to bag-of-words behavior. While recent Multimodal Large Language Models (MLLMs) have demonstrated significant advances in generalized vision-language understanding, their potential for learning transferable multimodal representations remains underexplored. In this work, we present UniME (Universal Multimodal Embedding), a novel two-stage framework that leverages MLLMs to learn discriminative representations for diverse

* Equal Contribution † Corresponding Author.

This work is licensed under a Creative Commons Attribution 4.0 International License. MM ’25, Dublin, Ireland

© 2025 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2035-2/2025/10 https://doi.org/10.1145/3746027.3754845

### Keywords

Vision-Language Model, Multi-Modal, Compositional Understanding

### 1 Introduction

Modern AI applications increasingly rely on multimodal embeddings to process diverse data types, powering essential tasks like image-text retrieval [3, 51], Retrieval Augmented Generation (RAG) [10, 25], and Visual Question Answering (VQA) [9, 13, 16]. As a seminal model, CLIP [44] demonstrates notable text-image retrieval performance via cross-modal contrastive supervision using large-scale web-collected image-text pairs. However, despite its widespread use, CLIP presents notable limitations. Firstly, it restricts text token length to 77, hindering its ability to process detailed descriptions and limiting its utility in cross-modal retrieval tasks that require extensive contextual information [6, 21, 66]. Moreover, CLIP employs a dual-encoder architecture that processes images and text separately, which compromises its effectiveness in complex tasks such as instruction-following multimodal retrieval [24, 36, 58]. Additionally, CLIP exhibits limited advanced language understanding, struggles with compositionality, and tends to display bag-of-words behavior [54, 64].

The success of Large Language Models (LLMs) [2, 17, 52, 53, 61] has motivated researchers to adapt LLMs to understand multimodal inputs. Multimodal Large Language Models (MLLMs) as a key component in the construction of general-purpose AI assistants have demonstrated remarkable progress [33, 35]. For example, Qwen2-VL [55] innovates beyond fixed-resolution visual processing, achieving robust performance across diverse image resolutions and aspect ratios. Similarly, LLaVA-OneVision [29] introduces a unified modeling approach that enables effective task transfer across scenarios while maintaining architectural simplicity. While these MLLMs show impressive vision-language reasoning capabilities, these MLLMs are inherently constrained by their autoregressive next-token prediction objective, which limits their effectiveness in learning multimodal representations compared to contrastive methods such as CLIP [23, 24].

Recent advances in LLM-based models have demonstrated substantial progress on the MTEB benchmark [39]. Inspired by these developments [4, 28], researchers are now actively investigating MLLMs for unified multimodal representation learning. E5-V [23] proposes an unimodal contrastive learning approach that trains the language component of MLLM on sentence pairs to bridge cross-modal representation disparities. However, this method encounters two primary constraints: (1) constraints arising from the limited scale and diversity of training data [41]; (2) inherent challenges caused by the MLLM’s causal attention mechanism, which fundamentally restricts its ability to learn complex contextual representations [38, 59, 60]. These factors collectively constrain the model’s full embedding potential. VLM2Vec [24] introduces the Massive Multimodal Embedding Benchmark (MMEB), comprising 36 datasets across 4 meta-tasks, and develops a contrastive framework that converts state-of-the-art vision-language models into embedding models through MMEB training. Nevertheless, the existence of falsenegative samples in the batch significantly complicates the discrimination of hard negative pairs when using the standard InfoNCE loss.

To overcome these challenges, we present UniME (Universal Multimodal Embedding), a novel two-stage framework that empowers multimodal large language models (as shown in Figure 1)

to learn universal representations for diverse downstream visionlanguage tasks. In the first textual discriminative knowledge distillation stage, we leverage a powerful LLM-based teacher model to enhance the embedding capabilities of MLLM’s language component. In the second stage of hard negative enhanced instruction tuning, we first eliminate false negative contamination, then implement a hard negative sampling strategy that selects multiple challenging negatives per instance within each batch. This approach forces the model to focus on challenging negative samples, thereby learning more discriminative multimodal representations while also improving instruction-following ability in downstream tasks. We evaluate our approach comprehensively on the MMEB benchmark and multiple retrieval tasks, including both short&long caption retrieval and compositional retrieval. Experimental results demonstrate that UniME achieves significant performance improvement across all tasks, exhibiting both robust discriminative power and superior compositional understanding. The main contributions of this paper are summarized as follows:

- • We present UniME (Universal Multimodal Embedding), a novel two-stage framework that empowers Multimodal Large Language Models (MLLMs) to learn universal representations for diverse downstream tasks.
- • We propose textual discriminative knowledge distillation, leveraging a powerful LLM-based teacher model to enhance the embedding capability of the MLLM’s language component.
- • We introduce hard negative enhanced instruction tuning to further advance discriminative representation learning through false negative filtering and hard negative sampling.
- • We conduct extensive experiments on the MMEB benchmark and multiple retrieval tasks, including both short&long caption retrieval and compositional retrieval. Results show that UniME demonstrates robust discriminative and compositional capabilities, achieving notable performance improvements across all evaluated tasks.

### 2 Related work 2.1 Multimodal Large Language Models

Multimodal Large Language Models (MLLMs) extend LLMs to process and integrate cross-modal information [5, 22, 56, 56]. A seminal development in this field is LLaVA [35], which encodes images into token representations using vision encoders like CLIP [18, 20, 44, 49, 62, 63] and projects them into the LLM’s token space. Following this breakthrough, numerous MLLM variants [31, 42, 67, 69] have demonstrated remarkable performance in multimodal comprehension and reasoning tasks. CogVLM [57] incorporates a trainable visual expert module within the attention and feed-forward network layers of the language model, yielding substantial performance enhancements across 17 canonical cross-modal benchmarks. LLaVA-1.6 [34] introduces the “AnyRes” technique to process variable high-resolution images, significantly improving fine-grained visual understanding. Phi3.5-Vision [1], a 4.2-billion parameter model evolved from phi-3.5-mini, exhibits strong reasoning capabilities for both single and multi-image text prompts. While these advancements, the autoregressive next-token prediction objective of MLLM inherently constrains its capacity to learn efficient multimodal representations.

[Figure 19]

[Figure 20]

Unimodal Embedding

[Figure 21]

Freeze Lernable

Text Tokens Instruction Tokens Text Embedding Tokens = +

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Multimodal Embedding

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

OFFLINE

LLM BackBone

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

LoRA

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

Vision Encoder

Instruction

Text

Image Multi-modal

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

(a) Training (b) Inference

- Figure 2: The framework of the Textual Discriminative Knowledge Distillation stage. We leverage the state-of-the-art LLM-based embedding model to enhance the discriminative capabilities of the MLLM’s language component.

### 2.2 LLMs for Representation Learning

presents a contrastive training framework that can handle any combination of images and text, as well as high-resolution images and long text inputs. Despite these improvements, current methods still face challenges in effectively discriminating hard negative samples during retrieval.

As large language models increasingly exhibit remarkable proficiency in natural language processing, recent research has pivoted towards harnessing decoder-only architectures for effective representation learning [4, 12, 28, 37, 47]. Previous work has adapted the prompt-based representation method for autoregressive models, enabling Large Language Models (LLMs) to perform in-context learning and scale to various model sizes. LLM2Vec [4] transforming pre-trained decoder-only LLMs into versatile text encoders by incorporating three principal advancements: bidirectional attention mechanisms, masked next-token prediction, and unsupervised contrastive alignment. Concurrently, NV-Embed [28] introduces a latent attention layer and eliminates the causal attention mask during contrastive training, substantially enhancing the efficiency of embeddings generated from decoder-only LLMs. While these approaches show promising embedding performance, their exclusive focus on text-only inputs fails to meet the growing demands of multimodal applications.

### 3 Method

This section first establishes the preliminary definitions, including task formulation and feature extraction (Section 3.1). Then we start to introduce our proposed novel two-stage framework UniME. We elaborate on the first textual discriminative knowledge distillation stage in Section 3.2. After that, we present the second hard negative enhanced instruction tuning stage in Section 3.3. Finally, we introduce the training recipe in Section 3.4.

### 3.1 Preliminary

- 3.1.1 Task Definition. To address the limitations of dual-tower encoder structures in acquiring unified multimodal representations [44], we employ Multimodal Large Language Models (MLLMs) with robust multimodal understanding to learn universal multimodal embeddings. Specifically, we feed both the query and candidate data into the MLLM using customized prompts to extract their respective embeddings. We then calculate the similarity (Θ) between the query (𝒒) and candidates (𝒄), followed by ranking and selecting the most relevant pairs (𝑷). This procedure is formalized as follows:

𝑷 = Rank(Θ(𝜙(𝒒),𝜙(𝒄))), (1) where 𝜙 denotes the MLLM employed to extract respective embeddings. Notably, both the query and candidate may be either unimodal (text or image only) or multimodal (interleaved imagetext).

- 3.1.2 Feature Extraction by MLLM.. Unlike the dual-tower structure of CLIP, MLLM incorporates three essential components: a vision tower, a projection layer, and an LLM backbone. This unified structure supports flexible processing of both unimodal (image

### 2.3 Multimodal Representation Learning

CLIP [44] demonstrates superior image-text retrieval capabilities through large-scale cross-modal contrastive learning, but it suffers from three inherent constraints: (1) 77-token text truncation limits fine-grained semantic alignment [6, 21, 66]; (2) Disjoint dualencoder architecture impedes cross-modal fusion for instructionsensitive tasks [24, 36, 58]; (3) Primitive language modeling induces bag-of-words representations [54, 64]. Recent research addresses these limitations through two complementary approaches. Firstly, MagicLens [68] employs lightweight dual encoders to enable relation-aware image retrieval guided by textual instructions. Secondly, MLLM-based approaches emerge for robust representation learning. E5-V [23] proposes a single-modality training approach that significantly outperforms traditional multimodal training on image-text pairs while reducing training costs. VLM2Vec [24]

[Figure 68]

[Figure 69]

Freeze

[Figure 70]

[Figure 71]

[Figure 72]

Instruction Given an image, find a similar everyday image with the described changes: Dog play on the ground.

[Figure 73]

Instruction

Learnable

Find a day-to-day image that looks similar to the provided image.

Query Embedding Positive Embedding

[Figure 74]

False Negative Contamination

Hard Negative Sampling

[Figure 75]

[Figure 76]

Positive

Positive

Hard Negative Embedding

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Hard Negative

Hard Negative

[Figure 86]

[Figure 87]

[Figure 88]

LoRA

[Figure 89]

[Figure 90]

play? Shovel or spoon ?

Query

Candidates

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

False Negative

False Negative

[Figure 96]

Instruction

[Figure 97]

[Figure 98]

[Figure 99]

Given an image, find a similar everyday image with the image.......

- Figure 3: The framework of the Hard Negative Enhanced Instruction Tuning stage. We further improve the discriminative capabilities of the MLLM through false negative filtering and hard negative sampling.

or text) and multimodal (interleaved image-text) inputs, enabling diverse task execution within a single framework. In this work, we present a novel two-stage framework that enables MLLMs to learn universal multimodal embedding for diverse downstream tasks. In the first textual discrimination knowledge distillation stage, we follow the previous work [23] and employ the prompt: “<Text> Summary above sentences in one word: \n” to guide the LLM to compress textual information into a single embedding at the last token. In the second hard negative enhanced instruction tuning stage, we utilize task-specific prompts from VLM2Vec [24] to adapt UniME fit diverse downstream tasks such as: "<Image> Represent the given image with the following question: <Text>" for Visual Question Answering (VQA) tasks and "<Image> Find a caption for the news in the given photo." for retrieval tasks.

embeddings from the teacher model and student model:

∑︁𝑛

exp 𝑒𝑠⊤𝑖𝑒𝑠𝑖/𝜏 𝑛

exp 𝑒𝑡⊤𝑖𝑒𝑡𝑖/𝜏 𝑛

KL

, (2)

L𝐾𝐿 =

exp 𝑒𝑠⊤𝑗𝑒𝑠𝑖/𝜏

exp 𝑒𝑡⊤𝑗𝑒𝑡𝑖/𝜏

𝑖=1

𝑗=1

𝑗=1

where𝜏 is the temperature hyper-parameter used to soften distribution representation. By distilling the relationships between different samples within a batch, our method demonstrates enhanced efficiency compared to direct contrastive learning under identical data and training conditions and achieves significant performance improvements in downstream tasks.

- 3.2.2 Inference. During the training phase, our approach exclusively utilizes text-only inputs and optimizes solely the language model component within the multimodal language model architecture, while maintaining other parameters frozen. For inference, we restore the original vision encoder and projection layer to enable multimodal processing. For unimodal inputs (text or image), we use modality-specific standardized prompts. For interleaved image-text inputs, we process each modality independently with its corresponding prompt and aggregate the embeddings through element-wise summation to produce the final multimodal representation.
- 3.3 Hard Negative Enhanced Instruction Tuning

### 3.2 Textual Discriminative Knowledge Distillation

- 3.2.1 Training. Inspired by previous research [23], we enhance the LLM backbone of the MLLM to improve its overall embedding capabilities. The autoregressive decoder architecture of LLMs, constrained by a causal masking mechanism, inherently restricts their discriminative capacity and poses challenges in effectively distinguishing between diverse items. To address this limitation, we introduce discriminative textual knowledge distillation (as shown in Figure 2), transferring knowledge from the state-of-the-art LLMbased embedding model NV-Embed V2 [28], which employs multiple diverse datasets and removes the causal attention mask during contrastive training. Specifically, we first decouple the LLM component from the MLLM architecture and process text-only inputs using the embedding prompt: “<Text> Summary above sentences in one word: \n”. Then we obtain the normalized student text embeddings 𝑒𝑠 ∈ R𝑛×𝑑 and teacher text embeddings (which is extracted offline) 𝑒𝑡 ∈ R𝑛×𝑑 from the hidden state of the final token, where 𝑛 is the batch size, 𝑑 is the dimension of the embeddings. Subsequently, we implement discriminative distribution alignment by minimizing the Kullback-Leibler (KL) divergence [48] between the

After textual discriminative knowledge distillation, UniME develops preliminary discriminative capabilities but exhibits limited visual sensitivity. This insensitivity results in deviations in imagetext alignment and limits the discriminative performance, despite the MLLM’s extensive pretraining on large-scale datasets. Moreover, the generic instruction prompts used in the first stage hinder UniME’s effectiveness in complex retrieval tasks. To address these limitations, we introduce an additional hard negative enhanced instruction tuning stage (as shown in Figure 3), which aims to:

- Table 1: Results on the MMEB benchmark. IND represents the in-distribution dataset, and OOD represents the out-of-distribution dataset. The reported scores are the average Precision@1 over the corresponding datasets. The best results are marked in bold. †: UniME with textual discrimination distillation only. ‡: UniME with both textual discrimination distillation and hard negative enhanced instruction tuning.

Per Meta-Task Score Average Score

Models #Parameters

Classification VQA Retrieval Grounding IND OOD Overall # of Datasets → 10 10 12 4 20 16 36

###### Zero-shot on MMEB

CLIP(ViT-L) [24] 0.4B 42.8 9.1 53.0 51.8 37.1 38.7 39.2 OpenCLIP(ViT-L) [44] 0.4B 41.5 6.9 44.6 53.5 32.8 36.0 36.6 Magiclens(ViT-L) [68] 0.4B 38.8 8.3 35.4 26.0 31.0 23.7 27.1 SigLIP(So/14) [65] 0.9B 40.3 8.4 31.6 59.5 32.3 38.0 35.0 BLIP2(ViT-L) [30] 1.2B 27.0 4.2 33.9 47.0 25.3 25.1 28.0 CLIP(ViT-BigG/14) [8] 2.5B 52.3 14.0 50.5 60.3 38.9 45.8 44.3 EVA-CLIP [50] 8B 56.0 10.4 49.2 58.9 38.1 45.6 43.7 E5-V(Phi3.5-V) [23] 4.2B 39.1 9.6 38.0 57.6 33.1 31.9 36.1 E5-V(LLaVA-1.6) [23] 7B 39.7 10.8 39.4 60.2 34.2 33.4 37.5 UniME†(Phi3.5-V) 4.2B 42.5(+3.4) 18.3(+8.7) 40.5(+2.5) 59.9(+2.3) 36.0(+2.9) 38.3(+6.4) 40.3(+4.2) UniME†(LLaVA-1.6) 7B 43.0(+3.3) 17.7(+6.9) 42.5(+3.1) 63.2(+3.0) 37.6(+3.4) 38.6(+5.2) 41.6(+4.1)

###### Fine-tuning on MMEB

CLIP(ViT-L) [24] 0.4B 55.2 19.7 53.2 62.2 47.6 42.8 47.6 VLM2Vec(Phi3.5-V) [24] 4.2B 54.8 54.9 62.3 79.5 66.5 52.0 62.9 VLM2Vec(LLaVA-1.6) [24] 7B 56.8 50.4 63.3 82.6 64.9 53.9 63.3 UniME‡(Phi3.5-V) 4.2B 54.8(+0.0) 55.9(+1.0) 64.5(+2.2) 81.8(+2.3) 68.2(+1.7) 52.7(+0.7) 64.2(+1.3) UniME‡(LLaVA-1.6) 7B 60.6(+3.8) 52.9(+2.5) 67.9(+4.6) 85.1(+2.5) 68.4(+3.5) 57.9(+4.0) 66.6(+3.3) UniME‡(LLaVA-OneVision) 7B 66.8 66.6 70.6 90.9 74.6 65.8 70.7

(1) further enhance discriminative capabilities, (2) improve crossmodal alignment, and (3) strengthen instruction-following ability for downstream tasks.

- 3.3.1 False Negative Contamination. The presence of false negatives in training batches hinders effective hard negative differentiation under the standard InfoNCE loss. As evidenced in Table 8, false negatives frequently appear as candidate samples. To mitigate this, we introduce a filtering mechanism based on the similarity threshold between the query and positive samples, defined as:

𝛼 = 𝑐𝑜𝑠(𝑒𝑞,𝑒𝑐+) + 𝛽, where 𝛽 is a hyper-parameter used to control the threshold margin. During training, we exclude all negative samples whose similarity to the query exceeds 𝛼, effectively eliminating false negatives while preserving challenging hard negatives.

- 3.3.2 Hard Negative Sampling. Hard negative samples, distinct in label from the positive sample but closely embedded, are poised to offer the greatest utility through the provision of substantial gradient information in the context of contrastive learning. In contrast, easy negatives yield negligible gradients and contribute minimally to the learning process. Drawing inspiration from prior research [26, 27, 46], we propose a hard negative sampling strategy to optimize both training efficiency and discriminative performance. The textual knowledge distillation stage equips UniME with preliminary discriminative ability to autonomously identify hard negatives for each query. Building on this capability, we sample 𝑘 corresponding hard negative 𝑒𝑐− within each training batch as follows:

𝑒𝑐− = Rank𝑘(𝑐𝑜𝑠(𝑒𝑞,𝑒𝑐)), where 𝑒𝑐 ∉ 𝑒𝑐+,𝑒𝑐∗ , (3)

where 𝑒𝑐∗ and 𝑒𝑐+ denote filtered false negative candidates and positive candidates respectively, 𝑒𝑞 is the query embedding, and 𝑒𝑐 represents all candidate embeddings. The function 𝑐𝑜𝑠(·) computes pairwise similarity scores, and Rank𝑘 selects the top-𝑘 highestscoring candidates as hard negatives. To maintain batch consistency when fewer than𝑘 hard negatives are obtained (which occurs rarely), we duplicate existing hard negatives to preserve the fixed size 𝑘.

3.3.3 Training Objective. After obtaining the embedding of the queries (𝑒𝑞), positive candidates (𝑒𝑐+) and hard negative candidates (𝑒𝑐−), we utilize the Noise Contrastive Estimation (InfoNCE) loss [40] over the in-batch sampled hard negatives as follows:

𝑒𝑥𝑝(𝑐𝑜𝑠(𝑒𝑞,𝑒𝑐+)/𝜏) 𝑒𝑥𝑝(𝑐𝑜𝑠(𝑒𝑞,𝑒𝑐+)/𝜏) + 𝑘𝑖=1 𝑒𝑥𝑝(𝑐𝑜𝑠(𝑒𝑞,𝑒𝑐𝑖−)/𝜏)

, (4)

L = −𝑙𝑜𝑔

where 𝑘 denotes the set of all hard negatives, and 𝜏 is a temperature hyper-parameter.

### 3.4 Training Recipe

- Stage1: Textual Discriminative Knowledge Distillation. We employ QLoRA (Quantized Low-Rank Adaptation) [11] for parameter-efficient fine-tuning of the large language model component. This stage exclusively utilizes text-only inputs and introduces minimal trainable parameters, typically less than 5% of the total. The complete training procedures for Phi3.5-V and LLaVA-1.6 require approximately 1 hour and 2 hours, respectively.
- Stage2: Hard Negative Enhanced Instruction Tuning. To overcome GPU memory limitations in large-batch MLLM training, we

- Table 2: Results of zero-shot text-image retrieval on short caption datasets (Flickr30K and MS-COCO), long caption datasets (ShareGPT4V and Urban1K) and compositional benchmark (SugarCrepe). The reported scores are the average Recall@1 over the corresponding datasets. The best results are marked in bold. †: UniME with textual discrimination distillation only. ‡: UniME with both textual discrimination distillation and hard negative enhanced instruction tuning.

Short Caption Retrieval Long Caption Retrieval Compositional Retrieval Flickr30K COCO ShareGPT4V Urban1K SugarCrepe

Models #Parameters

𝑞𝑡 → 𝑐𝑖 𝑞𝑖 → 𝑐𝑡 𝑞𝑡 → 𝑐𝑖 𝑞𝑖 → 𝑐𝑡 𝑞𝑡 → 𝑐𝑖 𝑞𝑖 → 𝑐𝑡 𝑞𝑡 → 𝑐𝑖 𝑞𝑖 → 𝑐𝑡 Replace Swap Add

OpenCLIP(ViT-L) [44] 0.4B 67.3 87.2 37.0 58.1 81.8 84.0 47.0 47.0 79.5 62.7 74.9 CLIP(ViT-BigG/14) [8] 2.5B 79.5 92.9 51.3 67.3 90.1 93.6 77.8 80.7 86.5 68.9 88.4 EVA-CLIP [50] 8B 80.3 94.5 52.0 70.1 93.1 91.2 80.4 77.8 85.9 70.3 86.7 E5-V(Phi3.5-V) [23] 4.2B 72.2 79.6 44.7 53.4 86.0 88.5 83.8 83.6 88.2 66.6 75.3 E5-V(LLaVA-1.6) [23] 7B 77.3 85.7 49.1 57.6 85.1 82.1 88.9 83.2 86.3 68.7 66.9 UniME†(Phi3.5-V) 4.2B 72.0(-0.2) 80.6(+1.0) 44.9(+0.2) 57.2(+0.8) 86.8(+3.8) 92.3(+1.3) 85.1(+2.3) 86.9(+3.3) 90.2(+2.0) 67.6(+1.0) 91.2(+15.9) UniME†(LLaVA-1.6) 7B 77.2(-0.1) 84.6(-1.1) 51.0(+1.9) 56.4(-1.2) 89.8(+4.7) 86.9(+4.8) 91.3(+2.4) 82.4(-0.8) 89.5(+3.2) 64.8(-3.9) 94.2(+27.3) VLM2Vec(Phi3.5-V) [24] 4.2B 68.7 83.0 43.7 59.8 90.1 92.0 87.9 86.8 86.2 66.7 84.2 VLM2Vec(LLaVA-1.6) [24] 7B 76.0 90.6 46.8 66.6 85.8 90.7 84.7 90.8 85.8 66.3 86.5 UniME‡ (Phi3.5-V) 4.2B 77.0(+11.3) 88.2(+5.2) 49.8(+6.1) 66.8(+7.0) 92.1(+2.0) 96.4(+4.4) 92.7(+4.8) 95.1(+8.3) 90.1(+3.9) 70.9(+4.2) 93.3(+9.1) UniME‡ (LLaVA-1.6) 7B 81.9(+5.9) 93.4(+2.8) 53.7(+6.1) 70.1(+3.5) 93.9(+8.1) 97.2(+6.5) 95.2(+10.5) 95.9(+5.1) 89.0(+3.2) 71.5(+5.2) 94.4(+7.9) UniME‡ (LLaVA-OV) 7B 83.3 94.4 54.8 74.0 93.9 89.3 94.3 95.5 80.5 65.5 82.2

employ a dual strategy: (1) Following VLM2Vec [24], we implement GradCache [14], a gradient caching technique that decouples backpropagation between contrastive loss computation and encoder updates; (2) we employ QLoRA [11] for parameter-efficient finetuning of all parameters within the MLLM. This combined approach facilitates effective training while ensuring manageable memory consumption.

- 4 Experiments

- 4.1 Implementation

We evaluate our proposed method through extensive experiments on three different multimodal large language models: Phi3.5Vision [1], and LLaVA-1.6 [34]. Our implementation leverages PyTorch with DeepSpeed [45] ZeRO stage-2 optimization to enhance training efficiency and facilitate community adoption. The training of UniME is conducted on 8×NVIDIA A100 (80GB) GPUs to accommodate the substantial computational demands. In the textual discriminative knowledge distillation stage, we implement gradient accumulation with a batch size of 768, a learning rate of 5e-4, and a LoRA rank of 32. We employ NV-Embed V2 [28] (one of the stateof-the-art embedding model on the MTEB benchmark [39]) as our teacher model. The training completes within two epochs. In the hard negative enhanced instruction turning stage, we use the lowresolution (336×336) image inputs. We increase the accumulated batch size to 1024, reduce the learning rate to 1e-4 for Phi3.5-V and 2e-5 for LLaVA-1.6, and decrease the LoRA rank to 16. We sample 𝑘 = 8 hard negatives within a batch with a similarity threshold of (𝛽=0.1). Each model undergoes 1,000 training steps during this stage.

### 4.2 Datasets and Metrics

- 4.2.1 Training. Following E5-V [23], we utilize the Natural Language Inference (NLI) dataset [15] which contains around 273k sentence pairs for the first textual discriminative knowledge distillation stage. For the hard negative enhanced instruction tuning stage, similar to the VLM2Vec [24], we employ 20 in-distribution datasets from the MMEB benchmark, which cover four core multimodal tasks: classification, visual question answering, multimodal

retrieval, and visual grounding. This comprehensive training corpus, incorporating both unimodal and multimodal input data, totals 662k carefully curated training pairs, ensuring robust model adaptation across diverse multimodal tasks.

- 4.2.2 Evaluation. In this study, we evaluate UniME across both in-distribution (20 test sets) and out-of-distribution (16 test sets) benchmarks from MMEB [24] to assess its multimodal embedding capabilities across diverse retrieval tasks. Following standard evaluation protocols [24, 36], we report Precision, which quantifies the proportion of correct matches among the top-ranked candidates for each dataset. To further examine the unimodal embedding performance of UniME, we conduct experiments on multiple crossmodal retrieval tasks, including short-caption image-text retrieval (Flickr30K [43] and COCO2014 [32]), long-caption image-text retrieval (ShareGPT4V [7] and Urban1K [66]), and compositional retrieval (SugarCrepe [19]). Consistent with the MMEB benchmark, Precision serves as the primary evaluation metric across all datasets. 4.3 Main Results.
- 4.3.1 Multi-Modal Retrieval. In Table 1, we present the performance of our proposed UniME against existing baseline models. Under identical training data and configuration settings, our proposed UniME consistently achieves significant performance improvements over E5-V across different foundation models. Specifically, UniME exhibits an average performance improvement of 4.2% over E5-V using the Phi3.5-V model. With LLaVA-1.6 as the base model, UniME further achieves an average enhancement of 4.1%. These significant performance improvements are primarily due to our proposed textual discrimination knowledge distillation method, which more efficiently enhances the discriminative capability of embeddings from a powerful teacher model. As depicted in Figure 4, we randomly select 50 samples from COCO and visualize the cross-modal cosine similarity matrix. The diagonal clarity of the UniME matrix is significantly enhanced compared to that of E5-V, indicating that UniME learns representations with superior distinctiveness. After the hard negative enhanced instruction tuning stage, beneficial from the hard negatives, the discriminative capability of the embeddings from UniME is further improved.

Training Loss Grad Norm

E5V UniME

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Random Sample Negatives Easy Negatives Hard Negatives

##### Figure 4: The discrimination comparison between E5-V and UniME†. † represents the UniME model only training on the first textual discrimination knowledge distillation stage.

Figure 5: The comparison of training loss and pre-clip gradient norms for hard negatives, easy negatives, and random sample negatives.

Table 3: Ablation study of different training stages. We report the mean scores on the MMEB benchmark, short and long cross-modal retrieval, as well as compositional cross-modal retrieval.

Compared with VLM2Vec [24], our model achieves 1.3% and 10.3% performance improvement when using Phi3.5-V and LLaVA-1.6 as the base model.

Stage1 Stage2 MMEB RShort RLong RCompos ✘ ✘ 25.3 44.2 62.9 63.1

- 4.3.2 Short&Long Caption Cross-Modal Retrieval. We evaluate our approach on zero-shot cross-modal retrieval tasks. Firstly, we conduct experiments on the short-caption datasets Flickr30K and MSCOCO. After the textual discrimination knowledge distillation stage, UniME achieves comparable retrieval performance to E5V. Subsequent hard negative enhanced instruction tuning further boosts UniME’s performance, yielding a significant improvement of 5.2%–11.3% over VLM2Vec. For long-caption retrieval tasks on the ShareGPT4V and Urban1K datasets, UniME demonstrates superior performance across all metrics. Specifically, following the textual discriminative knowledge distillation stage, UniME exhibits a performance improvement of 1.3%-3.8% based on the Phi3.5-V model. Subsequent enhancement through hard negative enhanced instruction tuning leads to additional gains, with UniME outperforming VLM2Vec by 2.0%-8.3%. It is noteworthy that, compared to EVA-CLIP(8B), UniME achieves performance improvements of 14.8% and 18.1% in long-caption retrieval on the Urban1K dataset. This significant enhancement primarily stems from the limitation of EVA-CLIP(8B), which is constrained by a 77-token length restriction, thereby inhibiting its ability to fully convey the complete semantic information of long captions.
- 4.3.3 Compositional Cross-Modal Retrieval. We evaluate the capacity of our UniME model to discriminate hard negative samples using the compositional benchmark SugarCrepe. As shown in Table 2, UniME consistently delivers superior performance across all evaluated metrics. Following the textual discriminative knowledge distillation phase, the Phi3.5-V-based UniME outperforms E5-V by 2.0% in relation replacement, 1.0% in object swapping, and 15.9% in attribute addition tasks. After the second hard negative enhanced instruction tuning stage, beneficial from the hard negative, the compositional understanding capabilities of UniME are further enhanced and it achieves 3.9%, 4.2%, and 9.1% performance improvement compared with VLM2Vec. Additionally, UniME exhibits improvements of 4.2%, 0.6%, and 6.6% compared to EVA-CLIP(8B), underscoring its robust capability to discriminate against hard negative samples.

- ✔ ✘ 40.3 63.7 87.8 83.0

✘ ✔ 63.8 61.5 84.2 77.1

- ✔ ✔ 64.2 70.4 94.1 84.8

5 Analysis

- 5.1 Analysis of the Hard Negative

In Figure 5, we visualize the training loss and pre-clip gradient norms for three negative types: easy negatives (least similar in batch), random negatives (randomly sampled in batch), and hard negatives (most similar in batch after removing positives and false negatives). Since easy negatives are easily distinguishable, the model struggles to enhance its discriminative power through learning from such data, consequently leading to a rapid convergence of the training loss to nearly zero. Using random negatives, the training loss converges more slowly compared to easy negatives, but it eventually nears zero. In contrast, hard negatives pose considerable challenges, sustaining elevated training losses. Correspondingly, gradient norms for easy negatives are minimal, whereas those for hard negatives are substantially higher, differing by orders of magnitude.

- 5.2 Ablation on Training Stages

We conduct an ablation study based on Phi3.5-V to evaluate different training stages. As shown in Table 3, the initial embeddings from Phi3.5-V exhibit weak discriminative properties, leading to suboptimal task performance. After the initial stage of textual discriminative knowledge distillation, the model registers performance improvements of 15%, 19.5%, 24.9%, and 19.9% on the MMEB benchmark, short and long caption cross-modal retrieval, and compositional retrieval tasks, respectively. Focusing solely on the second stage, which involves hard negative enhanced instruction tuning, results in performance gains of 38.5%, 17.3%, 21.3%, and 14.0% in the same tasks. Notably, the enhancement in MMEB benchmark performance after the second stage markedly exceeds that of the first, primarily due to improved model capabilities in following complex

Base Model

Base Model

[Figure 104]

[Figure 105]

Pastoral (p=90.0%) Farm(p=2.4%) Peaceful(p=1.5%)

Playground(p=49.8%) Recreational(p=10.2%)

Park(p=8.9%) Outdoor(p=8.5%) Community(p=6.4%)

Rural(p=1.3%) Ambience(p=0.8%)

UniME

UniME

[Figure 106]

[Figure 107]

Farm(p=92.5%) Cow(p=4.8%) Waterfront(p=0.7%)

Playground(p=80.3%) Park(p=8.7%) Trees(p=6.3%)

Pasture(p=0.5%) House(p=0.4%)

Buildings(p=3.1%) Pool(p=2.4%)

UniME

[Figure 108]

UniME

[Figure 109]

Playground(p=42.9%) Swimming Pool(p=40.1%) Trees(p=12.7%) House(p=2.1%) Play(p=1.9%)

House(p=64.9%) Farmhouse(p=21.8%) Cow(p=9.1%)

Water(p=1.6%) Grass(p=0.9%)

- Figure 6: Visualization of the top-k next predicted tokens before and after different training stages based on Phi3.5-V. †: UniME with textual discrimination distillation only. ‡: UniME with both textual discrimination distillation and hard negative enhanced instruction tuning.

Table 4: Ablation study of the false negative filtering threshold 𝛽. FalseNeg(%): proportion of samples which filtered false negatives.

##### Table 5: Ablation study of the number𝑘 of the hard negatives. HardNeg(%): proportion of hard negative samples within a batch.

Average Score IND OOD Overall

Average Score IND OOD Overall

Model Top-𝑘 HardNeg(%)

Model 𝛽 FalseNeg(%)

4 0.4% 67.8 52.4 63.8 8 0.8% 68.2 52.7 64.2

-0.1 81.7% 61.0 43.4 55.3

- 0.0 53.2% 66.1 49.0 61.1
- 0.1 22.9% 68.2 52.7 64.2
- 0.2 18.2% 68.2 51.9 63.7
- 0.3 13.1% 68.3 52.1 63.9

Phi3.5-V

16 1.6% 68.1 51.6 63.5 32 3.2% 68.4 51.8 63.6 64 6.4% 68.1 51.1 63.2

Phi3.5-V

training stages in Figure 6. We observe that before training, the predicted tokens are more abstract, such as “Pastoral” and “Peaceful”. After the textual discriminative knowledge distillation, the tokens shift towards more concrete semantics, including “cow”, “waterfront”, and “house”, though the probability distribution remains largely concentrated on "Farm". After the second stage of hard negative enhanced instruction tuning, the probability distribution becomes more evenly spread across multiple tokens that align with the image’s semantics, thereby allowing the embeddings to more accurately express the semantic content of the image with enhanced discriminative capability.

instructions. By integrating both training stages, our UniME model achieves optimal performance across all evaluated downstream tasks.

### 5.3 Ablation on the Threshold 𝛽

The threshold 𝛽 directly influences the percentage of the filtered false negatives. We conduct an ablation study on 20% of the entire dataset to identify the optimal value of 𝛽. As illustrated in Table 8, setting 𝛽 to -0.1 results in filtering false negatives for 81.7% of samples. However, the inclusion of some hard negative samples in the filtered set results in poor model performance. Increasing 𝛽 from -0.1 to 0.1 reduces the proportion of samples with filtered false negatives from 81.7% to 22.9%, thereby significantly improving performance. Further increasing 𝛽 to 0.3 filters false negatives in only 13.1% of samples, resulting in a slight performance decline due to persistent false negatives.

### 6 Conclusion

In this paper, we introduce UniME (Universal Multimodal Embedding), a novel two-stage framework that enables large multimodal language models with the capacity to learn discriminative representations applicable to a variety of downstream tasks. In the first textual discriminative knowledge distillation stage, we leverage a powerful LLM-based teacher model to enhance the embedding capability of the MLLM’s language component. In the second hard negative enhanced instruction tuning stage, we initially mitigate false negative contamination and then sample multiple hard negatives per instance within each batch, forcing the model to focus on challenging samples. This approach not only improves discriminative power but also enhances instruction-following ability in downstream tasks. We conduct extensive experiments on the MMEB benchmark and multiple retrieval tasks, including short&long caption retrieval and compositional retrieval. Results demonstrate that UniME achieves consistent performance improvement across all tasks, exhibiting superior discriminative and compositional capabilities. We hope that our work provides insights into multimodal representation learning.

### 5.4 Ablation on the Number of Hard Negatives

The value of 𝑘 directly determines the number of hard samples sampled for each instance in a batch. In Table 5, we illustrates the impact of (𝑘) on the performance of UniME based on Phi-3.5V. When we set 𝑘=8, compared to 𝑘=4, the higher diversity of hard negative samples leads to superior performance. Further increasing the value of 𝑘 introduces easy negatives, causing the model to lose focus on learning hard negatives, gradually reducing performance.

### 5.5 Visualization of the Output Distribution

To further explore the semantic representations captured by UniME embeddings, we utilize the prompt "<Image> Summary above image in one word: \n" and visualize the prediction probability of the top-k next predicted tokens before and after our proposed different

### References

- [1] Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, et al. 2024. Phi-3 technical report: A highly capable language model locally on your phone. arXiv:2404.14219 (2024).
- [2] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. 2023. Qwen technical report. arXiv:2309.16609 (2023).
- [3] Alberto Baldrati, Lorenzo Agnolucci, Marco Bertini, and Alberto Del Bimbo.

2023. Zero-shot composed image retrieval with textual inversion. In ICCV. 15338– 15347.

- [4] Parishad BehnamGhader, Vaibhav Adlakha, Marius Mosbach, Dzmitry Bahdanau, Nicolas Chapados, and Siva Reddy. 2024. Llm2vec: Large language models are secretly powerful text encoders. COLM (2024).
- [5] Yi Bin, Wenhao Shi, Yujuan Ding, Zhiqiang Hu, Zheng Wang, Yang Yang, SeeKiong Ng, and Heng Tao Shen. 2024. Gallerygpt: Analyzing paintings with large multimodal models. In ACMMM. 7734–7743.
- [6] Anjia Cao, Xing Wei, and Zhiheng Ma. 2024. FLAME: Frozen Large Language Models Enable Data-Efficient Language-Image Pre-training. arXiv:2411.11927

(2024).

- [7] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. 2024. Sharegpt4v: Improving large multi-modal models with better captions. In ECCV.
- [8] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. 2022. Reproducible scaling laws for contrastive language-image learning. arXiv:2212.07143 (2022).
- [9] Sanghyuk Chun, Seong Joon Oh, Rafael Sampaio De Rezende, Yannis Kalantidis, and Diane Larlus. 2021. Probabilistic embeddings for cross-modal retrieval. In CVPR.
- [10] Xin Cong, Bowen Yu, Mengcheng Fang, Tingwen Liu, Haiyang Yu, Zhongkai Hu, Fei Huang, Yongbin Li, and Bin Wang. 2023. Universal Information Extraction with Meta-Pretrained Self-Retrieval. In ACL.
- [11] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. Qlora: Efficient finetuning of quantized llms. NeurIPS (2023).
- [12] Aniket Didolkar, Andrii Zadaianchuk, Rabiul Awal, Maximilian Seitzer, Efstratios Gavves, and Aishwarya Agrawal. 2025. CTRL-O: Language-Controllable ObjectCentric Visual Representation Learning. In CVPR.
- [13] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and Pierre Colombo. 2024. Colpali: Efficient document retrieval with vision language models. In ICLR.
- [14] Luyu Gao, Yunyi Zhang, Jiawei Han, and Jamie Callan. 2021. Scaling deep contrastive learning batch size under memory limited setup. arXiv:2101.06983

(2021).

- [15] Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. Simcse: Simple contrastive learning of sentence embeddings. arXiv:2104.08821 (2021).
- [16] François Gardères, Maryam Ziaeefard, Baptiste Abeloos, and Freddy Lecue. 2020. Conceptbert: Concept-aware representation for visual question answering. In EMNLP.
- [17] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. 2024. The llama 3 herd of models. arXiv:2407.21783 (2024).
- [18] Tiancheng Gu, Kaicheng Yang, Xiang An, Ziyong Feng, Dongnan Liu, Weidong Cai, and Jiankang Deng. 2024. Rwkv-clip: A robustvision-language representation learner. In EMNLP.
- [19] Cheng-Yu Hsieh, Jieyu Zhang, Zixian Ma, Aniruddha Kembhavi, and Ranjay Krishna. 2023. Sugarcrepe: Fixing hackable benchmarks for vision-language compositionality. NeurIPS (2023).
- [20] Xiaoxing Hu, Kaicheng Yang, Jun Wang, Haoran Xu, Ziyong Feng, and Yupei Wang. 2025. Decoupled Global-Local Alignment for Improving Compositional Understanding. arXiv preprint arXiv:2504.16801 (2025).
- [21] Weiquan Huang, Aoqi Wu, Yifan Yang, Xufang Luo, Yuqing Yang, Liang Hu, Qi Dai, Xiyang Dai, Dongdong Chen, Chong Luo, et al. 2024. Llm2clip: Powerful language model unlock richer visual representation. arXiv:2411.04997 (2024).
- [22] Chaoya Jiang, Hongrui Jia, Mengfan Dong, Wei Ye, Haiyang Xu, Ming Yan, Ji Zhang, and Shikun Zhang. 2024. Hal-eval: A universal and fine-grained hallucination evaluation framework for large vision language models. In ACMMM. 525–534.
- [23] Ting Jiang, Minghui Song, Zihan Zhang, Haizhen Huang, Weiwei Deng, Feng Sun, Qi Zhang, Deqing Wang, and Fuzhen Zhuang. 2024. E5-v: Universal embeddings with multimodal large language models. arXiv:2407.12580 (2024).
- [24] Ziyan Jiang, Rui Meng, Xinyi Yang, Semih Yavuz, Yingbo Zhou, and Wenhu Chen. 2025. Vlm2vec: Training vision-language models for massive multimodal embedding tasks. ICLR (2025).
- [25] Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active retrieval augmented generation. In EMNLP. 7969–7992.

- [26] Yannis Kalantidis, Mert Bulent Sariyildiz, Noe Pion, Philippe Weinzaepfel, and Diane Larlus. 2020. Hard negative mixing for contrastive learning. NeurIPS

(2020).

- [27] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-Domain Question Answering.. In EMNLP.
- [28] Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. 2024. Nv-embed: Improved techniques for training llms as generalist embedding models. ICLR (2024).
- [29] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. 2024. Llava-onevision: Easy visual task transfer. arXiv:2408.03326 (2024).
- [30] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML.
- [31] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. 2024. Vila: On pre-training for visual language models. In CVPR.
- [32] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In ECCV.
- [33] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024. Improved baselines with visual instruction tuning. In CVPR.
- [34] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. 2024. LLaVA-NeXT: Improved reasoning, OCR, and world knowledge. https://llava-vl.github.io/blog/2024-01-30-llava-next/
- [35] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. NeurIPS (2023).
- [36] Yikun Liu, Pingan Chen, Jiayin Cai, Xiaolong Jiang, Yao Hu, Jiangchao Yao, Yanfeng Wang, and Weidi Xie. 2024. LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant. CVPR (2024).
- [37] Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and Jimmy Lin. 2024. Finetuning llama for multi-stage text retrieval. In SIGIR.
- [38] Hieu Man, Nghia Ngo, Franck Dernoncourt, and Thien Nguyen. 2024. Ullme: A unified framework for large language model embeddings with generationaugmented learning. In EMNLP.
- [39] Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers. 2022. MTEB: Massive Text Embedding Benchmark. arXiv:2210.07316 (2022).
- [40] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018. Representation learning with contrastive predictive coding. arXiv:1807.03748 (2018).
- [41] Yassine Ouali, Adrian Bulat, Alexandros Xenos, Anestis Zaganidis, Ioannis Maniadis Metaxas, Brais Martinez, and Georgios Tzimiropoulos. 2024. Discriminative Fine-tuning of LVLMs. arXiv:2412.04378 (2024).
- [42] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. 2023. Kosmos-2: Grounding multimodal large language models to the world. arXiv:2306.14824 (2023).
- [43] Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier,andSvetlana Lazebnik. 2015. Flickr30k entities:Collectingregion-to-phrase correspondences for richer image-to-sentence models. In ICCV.
- [44] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In ICML.
- [45] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. 2020. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In SIGKDD.
- [46] Joshua David Robinson, Ching-Yao Chuang, Suvrit Sra, and Stefanie Jegelka. 2020. Contrastive Learning with Hard Negative Samples. In ICLR.
- [47] Jungkyoo Shin, Bumsoo Kim, and Eunwoo Kim. 2025. Generative Modeling of Class Probability for Multi-Modal Representation Learning. In CVPR.
- [48] Jonathon Shlens. 2014. Notes on kullback-leibler divergence and likelihood. arXiv:1404.2000 (2014).
- [49] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. 2023. Eva-clip: Improved training techniques for clip at scale. arXiv:2303.15389 (2023).
- [50] Quan Sun, Jinsheng Wang, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, and Xinlong Wang. 2023. EVA-CLIP-18B: Scaling CLIP to 18 Billion Parameters. arXiv:2402.04252 (2023).
- [51] Yuanmin Tang, Jing Yu, Keke Gai, Jiamin Zhuang, Gang Xiong, Gaopeng Gou, and Qi Wu. 2025. Missing Target-Relevant Information Prediction with World Model for Accurate Zero-Shot Composed Image Retrieval. In CVPR.
- [52] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv:2302.13971 (2023).
- [53] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv:2307.09288 (2023).
- [54] Michael Tschannen, Manoj Kumar, Andreas Steiner, Xiaohua Zhai, Neil Houlsby, and Lucas Beyer. 2023. Image captioners are scalable vision learners too. NeurIPS

- (2023).

- [55] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. 2024. Qwen2-vl: Enhancing visionlanguage model’s perception of the world at any resolution. arXiv:2409.12191

(2024).

- [56] Puyi Wang, Wei Sun, Zicheng Zhang, Jun Jia, Yanwei Jiang, Zhichao Zhang, Xiongkuo Min, and Guangtao Zhai. 2024. Large multi-modality model assisted ai-generated image quality assessment. In ACMMM. 7803–7812.
- [57] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. 2023. CogVLM: Visual Expert for Pretrained Language Models. arXiv:2311.03079 [cs.CV]
- [58] Cong Wei, Yang Chen, Haonan Chen, Hexiang Hu, Ge Zhang, Jie Fu, Alan Ritter, and Wenhu Chen. 2024. Uniir: Training and benchmarking universal multimodal information retrievers. In ECCV. Springer, 387–404.
- [59] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. 2024. Show-o: One single transformer to unify multimodal understanding and generation. arXiv:2408.12528 (2024).
- [60] Yin Xie, Kaicheng Yang, Ninghua Yang, Weimo Deng, Xiangzi Dai, Tiancheng Gu, Yumeng Wang, Xiang An, Yongle Zhao, Ziyong Feng, et al. 2024. Croc: Pretraining Large Multimodal Models with Cross-Modal Comprehension. arXiv:2410.14332

(2024).

- [61] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. 2024. Qwen2. 5

technical report. arXiv:2412.15115 (2024).

- [62] Kaicheng Yang, Jiankang Deng, Xiang An, Jiawei Li, Ziyong Feng, Jia Guo, Jing Yang, and Tongliang Liu. 2023. Alip: Adaptive language-image pre-training with synthetic caption. In ICCV. 2922–2931.
- [63] Kaicheng Yang, Tiancheng Gu, Xiang An, Haiqiang Jiang, Xiangzi Dai, Ziyong Feng, Weidong Cai, and Jiankang Deng. 2025. Clip-cid: Efficient clip distillation via cluster-instance discrimination. In AAAI, Vol. 39. 21974–21982.
- [64] Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. 2022. When and why vision-language models behave like bags-of-words, and what to do about it? arXiv:2210.01936 (2022).
- [65] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. 2023. Sigmoid loss for language image pre-training. In ICCV.
- [66] Beichen Zhang, Pan Zhang, Xiaoyi Dong, Yuhang Zang, and Jiaqi Wang. 2024. Long-clip: Unlocking the long-text capability of clip. In ECCV.
- [67] Hao Zhang, Hongyang Li, Feng Li, Tianhe Ren, Xueyan Zou, Shilong Liu, Shijia Huang, Jianfeng Gao, Leizhang, Chunyuan Li, et al. 2024. Llava-grounding: Grounded visual chat with large multimodal models. In ECCV.
- [68] Kai Zhang, Yi Luan, Hexiang Hu, Kenton Lee, Siyuan Qiao, Wenhu Chen, Yu Su, and Ming-Wei Chang. 2024. Magiclens: Self-supervised image retrieval with open-ended instructions. arXiv:2403.19651 (2024).
- [69] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. 2023. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv:2304.10592 (2023).

This supplementary material provides additional details on our experimental settings, including training configurations and evaluation benchmarks as described in Sec.A. Additionally, it presents expanded results in Sec. B, including an ablation study on the LoRA rank and detailed findings on the MMEB benchmark. Finally, Sec. C provides supplementary visualizations that depict the output distribution and examples of negative data.

Table 6: Training hyperparameters and computational requirements for UniME(Phi3.5-V) and UniME(LLaVA-1.6).

Hyperparameter UniME(Phi3.5-V) UniME(LLaVA-1.6)

- Stage 1:Textual Discriminative Knowledge Distillation

Training samples 273K Batch size 768 Learning rate 5×10−4 LoRA rank 32 Epochs 2 DeepSpeed stage 2 GPU configuration 8×A100 Precision FP16 Training time 1 hour 2 hours

- Stage 2: Hard Negative Enhanced Instruction Tuning

Training samples 662K Batch size 1024 Learning rate 1×10−4 2×10−5 LoRA rank 16 Training steps 1000 Optimizer AdamW GPU configuration 8×A100 Precision BF16 Training time 26 hours 37 hours

### A Detailed Experiment Settings

- A.1 Training Details We provide the training configurations of UniME in Table 6.

- Stage1: Textual Discriminative Knowledge Distillation. We use 8×A100 GPUs (80GB each) to train UniME with text-only data. The text embeddings of the teacher model are extracted offline, and the model is trained by utilizing QLoRA [11], training durations are significantly reduced. Specifically, training completes in one hour for the Phi3.5-V [1] backbone and two hours for the LLaVA-1.6 [34] backbone.
- Stage2: Hard Negative Enhanced Instruction Tuning. Upon integrating the backbone with stage 1 LoRA outputs, we further advance UniME on image-text pairs derived from the MMEB training set. Under identical hardware conditions, this phase requires

26 hours for Phi-3.5V and 37 hours for LLaVA-1.6.

- A.2 Retrieval Task Evaluation Benchmarks

We evaluate UniME on diverse retrieval benchmarks, including short-caption, long-caption, and compositional image-text tasks (as shown in Table 7). For each benchmark, we adopt its standard

evaluation protocol. In retrieval tasks, we primarily report Recall@1 as the evaluation metric.

- Table 7: Summary of the evaluation benchmarks. # Queries represents the number of test queries, and # Candidates denotes the number of test candidates per query.

Benchmark Zero-shot #Queries #Candidates

Flickr30K [43] ✔ 1K 5K COCO [32] ✔ 5K 25K ShareGPT4V [7] ✔ 1K 1K Urban1K [66] ✔ 1K 1K SugarCrepe [19] ✔ 7.5K 2

B External Results

- Table 8: Performance comparison of different LoRA ranks under two training stages (IND: In-Domain, OOD: Out-OfDomain)

Average Score IND OOD Overall

Stage LoRA

- Stage1

4 36.0 37.9 40.0 8 35.8 37.8 39.9

16 35.9 38.0 40.0 32 36.0 38.3 40.3 64 35.6 37.8 39.9

- Stage2

4 67.4 51.0 62.6 8 68.2 51.0 63.8

16 68.2 52.7 64.2 32 67.6 51.8 63.2 64 67.0 51.5 62.7

### B.1 Ablation on the LoRA Rank

Due to computational limitations and the proven efficacy of LoRA [11], we utilize QLoRA to fine-tune the backbone of UniME. As detailed in Table 8, we assess the impact of various LoRA ranks on the MMEB benchmarks [24] utilizing the Phi-3.5V backbone. Our findings indicate that a LoRA rank of 32 delivers optimal performance in Stage 1, whereas a rank of 16 is most effective in Stage 2.

### B.2 Specific results on the MMEB

We present comparative results ofeight models on the MMEB benchmark in Table 9. The performance metrics for CLIP, SigLIP, BLIP-2, and MagicLens are sourced directly from VLM2Vec [24]. Conversely, results for EVA-CLIP(8B), E5-V, VLM2Vec, and UniME are obtained through reproduction in our experiments. For E5-V, VLM2Vec, and UniME, we only report the metrics for their best-performing variants, all of which employ the LLaVA-1.6 [34] backbone.

Base Model

Base Model

[Figure 110]

[Figure 111]

Pastoral (p=90.0%) Farm(p=2.4%) Peaceful(p=1.5%)

Playground(p=49.8%) Recreational(p=10.2%)

Park(p=8.9%) Outdoor(p=8.5%) Community(p=6.4%)

Rural(p=1.3%) Ambience(p=0.8%)

UniME

UniME

[Figure 112]

[Figure 113]

Farm(p=92.5%) Cow(p=4.8%) Waterfront(p=0.7%)

Playground(p=80.3%) Park(p=8.7%) Trees(p=6.3%)

Pasture(p=0.5%) House(p=0.4%)

Buildings(p=3.1%) Pool(p=2.4%)

UniME

[Figure 114]

UniME

[Figure 115]

Playground(p=42.9%) Swimming Pool(p=40.1%) Trees(p=12.7%) House(p=2.1%) Play(p=1.9%)

House(p=64.9%) Farmhouse(p=21.8%) Cow(p=9.1%)

Water(p=1.6%) Grass(p=0.9%)

MM ’25, October 27–31, 2025, Dublin, Ireland Tiancheng Gu, Kaicheng Yang et al.

Base Model

Base Model

[Figure 116]

[Figure 117]

Market(p=83.0%) Pumpkin(p=8.7%)

Tourism(p=86.0%) Rome(p=1.7%) Panorama(p=1.6%)

Halloween(p=4.0%) Autumn(p=3.1%) Festive(p=1.4%)

Arch(p=1.1%) Historical(p=1.0%)

UniME

[Figure 118]

UniME

[Figure 119]

Haystacks(p=75.8%) Pumpkin(p=18.9%)

Rome(p=72.7%) Roman(p=10.3%) Tourists(p=6.0%)

Marketplace(p=3.3%) Boy(p=3.1%) Blue(p=2.4%)

Crowd(p=5.6%) Visitors(p=3.0%)

UniME

[Figure 120]

[Figure 121]

UniME

Hay(p=57.6%) Boy(p=22.1%) Pumpkin(p=10.7%)

Rome(p=60.3%) Trees(p=14.4%) Crowd(p=13.9%)

People(p=2.2%) Visitors(p=1.8%)

Blue(p=5.1%) Son(p=1.9%)

##### Figure 7: Visualization of the top-k next predicted tokens before and after different training stages based on Phi3.5-V. †: UniME with textual discrimination distillation only. ‡: UniME with both textual discrimination distillation and hard negative enhanced instruction tuning.

C Further Analysis

[Figure 122]

- C.1 Visualization of the Output Distribution

In Figure 7, we further present additional examples that compare the top-k prediction probabilities of subsequent tokens, illustrating the evolution of model behavior across various training stages of our UniME.

- C.2 Schematic illustration of Negative Data

[Figure 123]

[Figure 124]

Instruction

Instruction Find a day-to-day image that looks similar to the provided image.

Identify the scene shown in the image

Positive

Positive

Glacier

[Figure 125]

[Figure 126]

Cheetah

Figure 8 provides additional examples of negative samples encountered during training. Our observations indicate that false negatives are more prevalent in textual data than in visual data, attributable primarily to the presence of synonyms.

Hard Negative

Hard Negative

[Figure 127]

[Figure 128]

Same appearance?

Cloudy is Cloudy Scene?

Serval

False Negative

False Negative

[Figure 129]

[Figure 130]

Mountain

Leopard

Figure 8: Schematic illustration of Negative data.

Table 9: The detailed results of the baselines and our UniMEon MMEB, which includes 20 in-distribution datasets and 16 out-of-distribution datasets. The out-of-distribution datasets are highlighted with a yellow background in the table. We only introduce the best version of UniME, E5-V and VLM2Vec in the table, which uses LLaVA-1.6 as the backbone.

CLIP SigLIP BLIP2 MagicLens EVA-CLIP E5-V VLM2Vec UniME Classification (10 tasks)

ImageNet-1K 55.8 45.4 10.3 48.0 75.0 40.5 66.5 71.3 N24News 34.7 13.9 36.0 33.7 33.8 31.5 76.4 79.5 HatefulMemes 51.1 47.2 49.6 49.0 49.3 49.3 60.9 64.6 VOC2007 50.7 64.3 52.1 51.6 44.3 76.7 84.0 90.4 SUN397 43.4 39.6 34.5 57.0 62.7 52.3 73.2 75.9 Place365 28.5 20.0 21.5 31.5 38.7 32.0 42.1 45.6 ImageNet-A 25.5 42.6 3.2 8.0 54.8 18.2 39.9 45.5 ImageNet-R 75.6 75.0 39.7 70.9 95.4 56.7 74.6 78.4 ObjectNet 43.4 40.3 20.6 31.6 67.8 34.2 34.3 36.4 Country-211 19.2 14.2 2.5 6.2 38.7 5.9 16.1 18.7 All Classification 42.8 40.3 27.0 38.8 56.0 39.7 56.8 60.6

#### VQA (10 tasks)

OK-VQA 7.5 2.4 8.7 12.7 9.9 15.1 66.5 68.3 A-OKVQA 3.8 1.5 3.2 2.9 2.8 4.7 54.9 58.7 DocVQA 4.0 4.2 2.6 3.0 7.4 9.1 64.4 67.6 InfographicsVQA 4.6 2.7 2.0 5.9 6.0 8.7 34.8 37.0 ChartQA 1.4 3.0 0.5 0.9 1.5 4.2 33.1 33.4 Visual7W 4.0 1.2 1.3 2.5 2.2 4.5 49.8 51.7 ScienceQA 9.4 7.9 6.8 5.2 14.1 9.6 37.3 40.5 VizWiz 8.2 2.3 4.0 1.7 4.3 8.6 39.9 42.7 GQA 41.3 57.5 9.7 43.5 44.7 34.1 57.3 63.6 TextVQA 7.0 1.0 3.3 4.6 10.8 9.5 65.7 65.2 All VQA 9.1 8.4 4.2 8.3 10.4 10.8 50.4 52.9

#### Retrieval (12 tasks)

VisDial 30.7 21.5 18.0 24.8 20.4 57.6 75.3 79.7 CIRR 12.6 15.1 9.8 39.1 36.0 41.0 51.3 52.2 VisualNews_t2i 78.9 51.0 48.1 50.7 82.4 43.9 70.7 74.8 VisualNews_i2t 79.6 52.4 13.5 21.1 88.2 46.8 75.2 78.8 MSCOCO_t2i 59.5 58.3 53.7 54.1 65.3 68.6 69.9 74.9 MSCOCO_i2t 57.7 55.0 20.3 40.0 67.2 54.8 67.7 73.8 NIGHTS 60.4 62.9 56.5 58.1 0.2 0.1 63.3 66.2 WebQA 67.5 58.1 55.4 43.0 70.9 33.7 83.6 89.8 FashionIQ 11.4 20.1 9.3 11.2 16.1 11.2 15.2 16.5 Wiki-SS-NQ 55.0 55.1 28.7 18.7 46.7 61.0 63.4 66.6 OVEN 41.1 56.0 39.5 1.6 1.8 0.5 49.6 55.7 EDIS 81.0 23.6 54.4 62.6 95.6 53.8 73.7 86.2 All Retrieval 53.0 31.6 33.9 35.4 49.2 39.4 63.3 67.9

#### Visual Grounding (4 tasks)

MSCOCO 33.8 46.4 28.9 22.1 35.8 41.7 77.0 76.5 RefCOCO 56.9 70.8 47.4 22.8 59.9 62.2 85.9 89.3 RefCOCO-matching 61.3 50.8 59.5 35.6 70.0 74.9 83.8 90.6 Visual7W-pointing 55.1 70.1 52.0 23.4 70.2 61.8 83.6 84.1 All Visual Grounding 51.8 59.5 47.0 26.0 58.9 60.2 82.6 85.1

#### Final Score (36 tasks)

All 39.2 35.0 28.0 27.1 43.7 37.5 63.3 66.6 All IND 37.1 32.3 25.3 31.0 38.1 34.2 64.9 68.4 All OOD 38.7 38.0 25.1 23.7 45.6 33.4 53.9 57.9

