# arXiv:2502.09838v3[cs.CV]21Feb2025

[Figure 1]

## HealthGPT: A Medical Large Vision-Language Model for Unifying Comprehension and Generation via Heterogeneous Knowledge Adaptation

Tianwei Lin1, Wenqiao Zhang1, Sijing Li1, Yuqian Yuan1, Binhe Yu2, Haoyuan Li3, Wanggui He3, Hao Jiang3, Mengze Li4, Xiaohui Song1, Siliang Tang1, Jun Xiao1, Hui Lin1, Yueting Zhuang1, Beng Chin Ooi5 1Zhejiang University, 2University of Electronic Science and Technology of China, 3Alibaba, 4The Hong Kong University of Science and Technology, 5National University of Singapore

Project Page Code

[Figure 2]

[Figure 3]

#### 7 Medical Multi-Modal Comprehension Tasks

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

- 1. X-Ray

Comprehension

[Figure 9]

[Figure 10]

- 2. CT

Comprehension

- 3. MRI

- 4. Microsopy Comprehension What is the purpose of the different membrane treatments used in this study?

[Figure 11]

[Figure 12]

The purpose of the different membrane treatments used…

- 5. OCT Comprehension What is the purpose of comparing the OCT structure image and OCTA image with H&E histology?

Comp. Perf.

List all anatomical locations showing pulmonary edema, hazy opacity, or mediastinal displacement.

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Left hilar structures, left lung, right hilar structures, right lung.

[Figure 19]

[Figure 20]

[Figure 21]

Which abdominal organ shows any indication of a lesion or abnormality in the CT image?

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

To confirm the histological position of the obtained OCT brain images.

[Figure 26]

No abdominal organs show any clear indications of lesions or abnormalities.

[Figure 27]

[Figure 28]

[Figure 29]

7. Ultrasound Comprehension

###### 6. Fundus Comprehension

What type of imaging technique is used in this image? The image is a sagittal grayscale ultrasonographic…

What specific findings or pathological changes can be observed in this fundus image?

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Could you explain what this mass in the MRI means for my health? Is it very serious?

Comprehension

[Figure 37]

[Figure 38]

[Figure 39]

The fundus image appears normal with no noticeable signs of pathology…

[Figure 40]

[Figure 41]

Certainly, the MRI shows a defined mass in your left nasal cavity. There is no sign of the ...

[Figure 42]

5. Report-to-CXR

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Here is the chest X-ray image for you.

[Figure 47]

The X-ray shows no pleural effusion or pneumothorax.

[Figure 48]

[Figure 49]

[Figure 50]

- 1. CT2MRI Generation

[Figure 51]

I need a version of this CT representation in MRI.

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

The image has been transformed into MRI.

- 2. MRI2CT Generation

Gen. Performance

[Figure 58]

[Figure 59]

[Figure 60]

Gen. Perf.

[Figure 61]

[Figure 62]

4. Super Resolution

3. Image Reconstruction

[Figure 63]

Could you improve the quality of this MRI image?

Reconstruct the following medical images.

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

Transform the MRI display into a CT image.

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Here is the CT version of the MRI image.

[Figure 82]

Here is the reconstructed medical image you need.

Here is the image with improved resolution.

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

5 Medical Multi-Modal Generation Tasks

Figure 1: HealthGPT enables medical multi-modal comprehension and generation, outperforming both state-of-the-art unified visual models and medical-specific models across various tasks. This highlights its superior capability in tackling complex tasks in healthcare applications. Comp.Perf. and Gen.Perf. denote the results of comprehension and generation.

###### Abstract

We present HealthGPT, a powerful Medical Large VisionLanguage Model (Med-LVLM) that integrates medical visual comprehension and generation capabilities within a unified autoregressive paradigm. Our bootstrapping philosophy is to progressively adapt heterogeneous comprehension and generation knowledge to pre-trained large language models (LLMs). This is achieved through a novel heterogeneous low-rank adaptation (H-LoRA) technique, which is complemented by a tailored hierarchical visual perception approach and a three-stage learning strategy. To effectively learn the HealthGPT, we devise a comprehensive medical domain-specific comprehension and generation dataset

called VL-Health. Experimental results demonstrate exceptional performance and scalability of HealthGPT in medical visual unified tasks. Our project can be accessed at https://github.com/DCDmllm/HealthGPT.

### 1 Introduction

Large Vision-Language Models (LVLMs) (Liu et al. 2023; OpenAI 2023; Liu et al. 2024c; Chen et al. 2024b) have demonstrated outstanding open-world visual comprehension and reasoning abilities through language-based interactive dialogue over the past years, simultaneously opening up new opportunities for applications in specialized domains.

Specifically, recent studies (Li et al. 2024a; Tu et al. 2024) have utilized pre-trained large language models (LLMs) and visual instruction data to build interactive diagnostic tools and treatment planning systems, revealing the immense potential of LVLMs in medical scenarios. However, these studies primarily concentrate on visual comprehension tasks that produce text-based outputs, such as medical visual question answering (Li et al. 2024a) or report generation (Nath et al. 2024), and deficient the “drawing” capability needed for medical visual generation. In practice, integrating visual comprehension and generation can significantly enhance the multifunctionality of medical LVLMs.

Recent studies have increasingly focused on developing unified LVLMs capable of comprehending and generating content across diverse visual modalities. Earlier approaches predominantly utilized continuous visual tokens fed into LLMs, using the LLMs themselves as conditional generators for external generative models (Ge et al. 2024; Wu et al.

- 2023; Dong et al. 2023). More recent research has explored the use of discrete visual tokens for image representation and generation within a fully autoregressive framework (Team
- 2024; Wang et al. 2024a; Xie et al. 2024). These methods not only enhance controllability but also demonstrate early success in open-world, any-to-any tasks, highlighting the preliminary potential of a unified autoregressive learning paradigm in multi-modal tasks.

While unified LVLMs have achieved initial success in general scenarios, such a unified framework remains underexplored in the medical domain. Adapting the aforementioned general unified model paradigm to the medical domain presents two major challenges: (i) High-scale and -quality Data Limitations. Open-world models necessitate extensive pre-training on billions or even more diverse, multi-modal data samples for comprehension and generation tasks (Lu et al. 2024; Team 2024). However, the accessible medical data significantly lacks in scale and quality compared to natural multi-modal datasets. Its specialized and domain-specific characteristics make it challenging to develop a unified medical model from scratch. (ii) Conflicts between Comprehension and Generation. Comprehension tasks often strip away visual details to focus on abstraction, while generation tasks require detailed preservation, making tokens sensitive to all visual alterations. As shown in Figure 2, which features experiments conducted on medical images, the performance in comprehension (or generation) tasks steadily decreases as the proportion of generation (or comprehension) data increases, and vice versa. This highlights a dilemma in autoregressive multi-modal training, stemming from the need to maintain consistency between pre- and post-LVLMs. While some methods have explored mutual enhancement between comprehension and generation (Pan et al. 2024; Tong et al. 2024), improvements still exhibit diminishing returns, with performance degradation remaining a significant issue.

###### (a) (b)

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Figure 2: With a fixed amount of comprehension (generation) data, increasing the proportion of the other type leads to significant performance degradation.

To tackle the aforementioned challenges, we propose HealthGPT (see Figure 1) , which progressively adapts a pre-trained LLM as an unified medical multi-modal model with a small amount of visual instruction data. We devise innovative Parameter-Efficient Fine-Tuning (PEFT) approach (Ding et al. 2023), called Heterogeneous Low-Rank Adaptation (H-LoRA), which decouples the learning process of LVLMs for comprehension and generation tasks. Inspired by the plug-and-play nature of LoRA (Hu et al. 2021), H-LoRA enables the model to store heterogeneous comprehension and generation knowledge in independent “plugins”, thus avoiding joint optimization issues caused by conflicts between comprehension and generation tasks. In addition, we also consider the variety of sub-tasks among comprehension or generation tasks. Qualitative research highlights the limitations of a single LoRA in handling multidimensional task scenarios, mainly due to catastrophic forgetting and interference (Liu et al. 2024d; Lin et al. 2024). To address this, we draw on the concept of Mixture of Experts (MoE) (Masoudnia and Ebrahimpour 2014) and introduce LoRA experts. The aim is to dynamically transfer task-shared knowledge to adapt to downstream tasks. Unlike MoELoRA (Luo et al. 2024a), H-LoRA employs reversible matrix block multiplication to combine LoRA experts, significantly reducing the overhead of multiple matrix multiplications. Notably, when using four experts, it requires only 67% of the MoELoRA training time.

To effectively leverage H-LoRA in HealthGPT, we further introduce a Hierarchical Visual Perception (HVP) and devise a corresponding Three-stage Learning Strategy (TLS). HVP: we separate visual details learning from Vision transformer (ViT) for comprehension and generation. As is widely recognized, the ViT encodes visual concepts with increasing abstraction, generally, becoming finer as we progress over levels (Vig 2019). Thus, we maintain the visual features of the anterior and posterior layers to accommodate the differing requirements for visual granularity in comprehension and generation tasks while preventing po-

tential task interference. TLS: In the first and second stages, given the heterogeneity between comprehension and generation tasks, we first train H-LoRA plugins for HealthGPT to incorporate both medical comprehension and generation knowledge, thus endowing the LLMs with capabilities for vision-language alignment and vision-to-vision reconstruction. Additionally, through minimal mixed-task training, we built fusion embedding layers and output heads that merge text and visual tokens, establishing a unified LVLM foundation for visual instruction fine-tuning. In the third stage, by only training the H-LoRA plugins, HealthGPT is able to rapidly adapt to a wide range of downstream medical tasks, covering various types of medical comprehension and generation tasks.

To effectively implement our approach, we have curated a dataset for training unified medical LVLMs, called VL-Health, including seven comprehension tasks and five generation tasks (Figure 1). Through quantitative analysis and validation on multi-modal tasks, the results demonstrate that HealthGPT is capable of unifying medical multimodal abilities in data-constrained scenarios, achieving performance comparable to or better than existing state-of-theart (SOTA) models across multiple metrics. Overall, the main contributions of this paper are summarized as follows:

- • Unified Med-LVLM. We introduce HealthGPT, which, to the best of our knowledge, is the first unified framework for multi-modal comprehension and generation in complex medical scenarios.
- • Effective Learning Paradigm. We present H-LoRA, an optimized multi-LoRA PEFT architecture based on taskgated decoupling, is designed to effectively mitigate data conflict issues.
- • Holistic Training Dataset. We curated VL-Health, a comprehensive dataset designed for both comprehension and generation tasks.
- • Superior Downstream Improvements: Extensive experiments are conducted and the results confirm HealthGPT’s effectiveness in medical vision-language comprehension and generation.

### 2 Related Work

Medical Vision Large Language Models. Recently, medical vision large language models (Med-VLLMs) have made significant progress, demonstrating excellent performance in understanding medical images and responding to human queries based on these images (Zhou et al. 2023; Tian et al.

- 2023). XrayGPT (Thawkar et al. 2023) combines a medical visual encoder (MedClip) (Wang et al. 2022) with a fine-tuned LLM , using a simple linear transformation layer to achieve alignment between visual and textual information, significantly enhancing the understanding of medical images. On this basis, LLaVA-Med (Li et al. 2024b) further enhances visual-text alignment in medical contexts by

selecting high-quality image-text pairs from PubMed papers and synthesized VQA datasets. BiomedGPT (Luo et al. 2024b) employs a BERT-style encoder and GPT-style decoder architecture, pre-trained on interdisciplinary datasets. Compared to commercial models like Med-PaLM (Singhal et al. 2023), BiomedGPT significantly reduces model size while maintaining superior performance. However, issues of language adaptability and dataset specificity still remain. To address these, HuatuoGPT-Vision (Chen et al. 2024a) introduces the PubMedVision dataset, which contains 1.3 million high-quality medical samples, significantly improving the model’s adaptability across diverse medical applications. However, current Med-VLLMs mainly focus on medical comprehension and lack the capability for the medical vision-language generation.

Unified Visual Comprehension and Generation Models. Recent research has increasingly concentrated on creating unified LVLMs that are adept at understanding and producing content across various visual modalities. NExTGPT (Wu et al. 2023) achieves perception and generation for arbitrary combinations of multi-modal inputs and outputs by aligning LLMs. Similarly, SEED (Ge et al. 2023), SEEDX (Ge et al. 2024), and DreamLLM (Dong et al. 2023) employ learnable queries and leverage next-token prediction to generate visual tokens, providing conditional inputs to external generation modules. Unlike these methods, which function as external conditioners, Unified-IO (Lu et al. 2022), Unified-IO 2 (Lu et al. 2024), and Chameleon (Team 2024) internalize multi-modal generation tasks within a unified Transformer architecture by extending multi-modal vocabularies, enabling direct generation based on next-token prediction. Building on this concept, Lumina-mGPT (Liu et al. 2024a) and ANOLE (Chern et al. 2024) further enhance the generation capabilities of unified models using high-quality data, particularly improving the quality and flexibility of image generation.

### 3 Preliminaries

Large Vision-Language Models. The input to a LVLM typically consists of an image ximg and a discrete text sequence xtxt. The visual encoder Eimg converts the input image ximg into a sequence of visual tokens V = [vi]N

i=1, while the text sequence xtxt is mapped into a sequence of text tokens T = [ti]N

v

i=1 using an embedding function Etxt. The LLM MLLM(·|θ) models the joint probability of the token sequence U = {V,T }, which is expressed as:

t

Nr

Pθ(ri|{U,r<i}), (1)

Pθ(R|U) =

i=1

where R = [ri]N

i=1 is the text response sequence. The LVLM iteratively generates the next token ri based on r<i. The optimization objective is to minimize the cross-entropy loss of the response R. It is worth noting that most LVLMs adopt

r

[Figure 95]

Figure 3: The HealthGPT architecture integrates hierarchical visual perception and H-LoRA, employing a task-specific hard router to select visual features and H-LoRA plugins, ultimately generating outputs with an autoregressive manner.

a design paradigm based on ViT, alignment adapters, and pre-trained LLMs(Liu et al. 2023, 2024b), enabling quick adaptation to downstream tasks.

VQGAN. VQGAN (Esser, Rombach, and Ommer 2021) employs latent space compression and indexing mechanisms to effectively learn a complete discrete representation of images. VQGAN first maps the input image ximg to a latent representation z = E(x) through a encoder E. Then, the latent representation is quantized using a codebook Z = {zk}Kk=1, generating a discrete index sequence I = [im]Nm=1, where im ∈ Z represents the quantized code index:

I = Quantize(z|Z) = arg min

∥z − zk∥2. (2)

zk∈Z

In our approach, the discrete index sequence I serves as a supervisory signal for the generation task, enabling the model to predict the index sequence Iˆ from input conditions such as text or other modality signals. Finally, the predicted index sequence Iˆ is upsampled by the VQGAN decoder G, generating the high-quality image xˆimg = G(Iˆ).

Low Rank Adaptation. LoRA(Hu et al. 2021) effectively captures the characteristics of downstream tasks by introducing low-rank adapters. The core idea is to decompose the bypass weight matrix ∆W ∈ Rd

in×dout into two lowrank matrices {A ∈ Rd

in×r,B ∈ Rr×d

out

}, where r ≪ min{din,dout}, significantly reducing learnable parameters. The output with the LoRA adapter for the input x is then given by:

h = xW0 + αx∆W/r = xW0 + αxAB/r, (3) where matrix A is initialized with a Gaussian distribution, while the matrix B is initialized as a zero matrix. The scaling factor α/r controls the impact of ∆W on the model.

### 4 HealthGPT

##### 4.1 Unified Autoregressive Generation.

HealthGPT (Figure 3) utilizes a discrete token representation that covers both text and visual outputs, unifying visual comprehension and generation as an autoregressive task. For comprehension, Mllm receives the input joint sequence U and outputs a series of text token R = [r1,r2,...,rN

], where ri ∈ Vtxt, and Vtxt represents the LLM’s vocabulary:

r

Nr

Pθ(ri | U,r<i). (4)

Pθ(R | U) =

i=1

For generation, Mllm first receives a special start token ⟨START IMG⟩, then generates a series of tokens corresponding to the VQGAN indices I = [i1,i2,...,iN

], where ij ∈ Vvq, and Vvq represents the index range of VQGAN. Upon completion of generation, the LLM outputs an end token ⟨END IMG⟩:

i

Ni

Pθ(ij | U,i<j). (5)

Pθ(I | U) =

j=1

Finally, the generated index sequence I is fed into the decoder G, which reconstructs the target image xˆimg = G(I).

##### 4.2 Hierarchical Visual Perception

Given the differences in visual perception between comprehension and generation tasks—where the former focuses on abstract semantics and the latter emphasizes complete semantics—we employ ViT to compress the image into discrete visual tokens at multiple hierarchical levels. Specifically, the image is converted into a series of features {f1,f2,...,fL} as it passes through L ViT blocks.

To address the needs of various tasks, the hidden states are divided into two types: (i) Concrete-grained features FCon = {f1,f2,...,fk},k < L, derived from the shallower layers of ViT, containing sufficient global features, suitable for generation tasks; (ii) Abstract-grained features FAbs = {fk+1,fk+2,...,fL}, derived from the deeper layers of ViT, which contain abstract semantic information closer to the text space, suitable for comprehension tasks.

The task type T (comprehension or generation) determines which set of features is selected as the input for the downstream large language model:

FTimg = FCon, if T = generation task FAbs, if T = comprehension task

(6)

We integrate the image features FTimg and text features T into a joint sequence through simple concatenation, which is then

fed into the LLM Mllm for autoregressive generation.

##### 4.3 Heterogeneous Knowledge Adaptation

We devise H-LoRA, which stores heterogeneous knowledge from comprehension and generation tasks in separate modules and dynamically routes to extract task-relevant knowledge from these modules. At the task level, for each task type T, we dynamically assign a dedicated H-LoRA submodule θT, which is expressed as:

R = MLLM(U|θ,θT), θT = {AT,BT,RTouter}. (7)

At the feature level for a single task, H-LoRA integrates the idea of Mixture of Experts (MoE) (Masoudnia and Ebrahimpour 2014) and designs an efficient matrix merging and routing weight allocation mechanism, thus avoiding the significant computational delay introduced by matrix splitting in existing MoELoRA (Luo et al. 2024a). Specifically, we first merge the low-rank matrices (rank = r) of k LoRA experts into a unified matrix:

Amerged,Bmerged = Concat({Ai}k1),Concat({Bi}k1), (8) where Amerged ∈ Rd

in×rk and Bmerged ∈ Rrk×d

out

. The k-dimension routing layer generates expert weights W ∈ Rtoken num×k based on the input hidden state x, and these are expanded to Rtoken num×rk as follows:

Wexpanded = αkW/r ⊗ 1r, (9)

where ⊗ denotes the replication operation. The overall output of H-LoRA is computed as:

OH-LoRA = (xAmerged ⊙ Wexpanded)Bmerged, (10)

where ⊙ represents element-wise multiplication. Finally, the output of H-LoRA is added to the frozen pre-trained weights to produce the final output:

O = xW0 + OH-LoRA. (11)

(a) (b)

###### （K）

[Figure 96]

900 800 700 600 500 400 300 200 100

765K 783K

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

0

Comp. Gen.

Figure 4: Data statistics of VL-Health.

##### 4.4 Training Pipeline

1st Stage: Multi-modal Alignment. In the first stage, we design separate visual adapters and H-LoRA submodules for medical unified tasks. For the medical comprehension task, we train abstract-grained visual adapters using high-quality image-text pairs to align visual embeddings with textual embeddings, thereby enabling the model to accurately describe medical visual content. During this process, the pretrained LLM and its corresponding H-LoRA submodules remain frozen. In contrast, the medical generation task requires training concrete-grained adapters and H-LoRA submodules while keeping the LLM frozen. Meanwhile, we extend the textual vocabulary to include multimodal tokens, enabling the support of additional VQGAN vector quantization indices. The model trains on image-VQ pairs, endowing the pre-trained LLM with the capability for image reconstruction. This design ensures pixel-level consistency of pre- and post-LVLM. The processes establish the initial alignment between the LLM’s outputs and the visual inputs. 2nd Stage: Heterogeneous H-LoRA Plugin Adaptation. The submodules of H-LoRA share the word embedding layer and output head but may encounter issues such as bias and scale inconsistencies during training across different tasks. To ensure that the multiple H-LoRA plugins seamlessly interface with the LLMs and form a unified base, we fine-tune the word embedding layer and output head using a small amount of mixed data to maintain consistency in the model weights. Specifically, during this stage, all HLoRA submodules for different tasks are kept frozen, with only the word embedding layer and output head being optimized. Through this stage, the model accumulates foundational knowledge for unified tasks by adapting H-LoRA plugins.

3rd Stage: Visual Instruction Fine-Tuning. In the third stage, we introduce additional task-specific data to further optimize the model and enhance its adaptability to downstream tasks such as medical visual comprehension (e.g., medical QA, medical dialogues, and report generation) or generation tasks (e.g., super-resolution, denoising, and

Table 1: Comparison of HealthGPT with other LVLMs and unified multi-modal models on medical visual comprehension tasks. Bold and underlined text indicates the best performance and second-best performance, respectively.

VQA-RAD ↑ SLAKE ↑ PathVQA ↑ Type Model # Params

Medical LVLM close all close all close all

MMMU -Med

↑ OMVQA↑ Avg. ↑

| | | | |
|---|---|---|---|
| | | | |
|Comp. Only|Med-Flamingo 8.3B ✓ LLaVA-Med 7B ✓ HuatuoGPT-Vision 7B ✓ BLIP-2 6.7B ✗ LLaVA-v1.5 7B ✗ InstructBLIP 7B ✗ Yi-VL 6B ✗ InternVL2 8B ✗ Llama-3.2 11B ✗|58.6 43.0 47.0 25.5 61.9 31.3 28.7 34.9<br><br>60.2 48.1 58.4 44.8 62.3 35.7 30.0 41.3 66.9 53.0 59.8 49.1 52.9 32.0 42.0 50.0 43.4 36.8 41.6 35.3 48.5 28.8 27.3 26.9<br><br>51.8 42.8 37.1 37.7 53.5 31.4 32.7 44.7<br><br>61.0 44.8 66.8 43.3 56.0 32.3 25.3 29.0<br><br>52.6 42.1 52.4 38.4 54.9 30.9 38.0 50.2 64.9 49.0 66.6 50.1 60.0 31.9 43.3 54.5 68.9 45.5 72.4 52.1 62.8 33.6 39.3 63.2<br><br><br><br><br>|41.4 47.6 50.7 36.1 41.5 44.8 44.9 52.5 54.7|
|Comp. & Gen.|Show-o 1.3B ✗ Unified-IO 2 7B ✗ Janus 1.3B ✗ HealthGPT-M3 3.8B ✓ HealthGPT-L14 14B ✓<br><br>|50.6 33.9 31.5 17.9 52.9 28.2 22.7 45.7 46.2 32.6 35.9 21.9 52.5 27.0 25.3 33.0 70.9 52.8 34.7 26.9 51.9 27.9 30.0 26.8 73.7 55.9 74.6 56.4 78.7 39.7 43.3 68.5 77.7 58.3 76.4 64.5 85.9 44.4 49.2 74.4<br><br>|42.6 33.8 33.5 61.3 66.4<br><br>|

Table 2: The experimental results for the four modality conversion tasks.

CT to MRI (Brain) CT to MRI (Pelvis) MRI to CT (Brain) MRI to CT (Pelvis) Model

SSIM ↑ PSNR ↑ MSE ↓ SSIM ↑ PSNR ↑ MSE ↓ SSIM ↑ PSNR ↑ MSE ↓ SSIM ↑ PSNR ↑ MSE ↓

pix2pix 71.09 32.65 36.85 59.17 31.02 51.91 78.79 33.85 28.33 72.31 32.98 36.19 CycleGAN 54.76 32.23 40.56 54.54 30.77 55.00 63.75 31.02 52.78 50.54 29.89 67.78 BBDM 71.69 32.91 34.44 57.37 31.37 48.06 86.40 34.12 26.61 79.26 33.15 33.60 Vmanba 69.54 32.67 36.42 63.01 31.47 46.99 79.63 34.12 26.49 77.45 33.53 31.85 DiffMa 71.47 32.74 35.77 62.56 31.43 47.38 79.00 34.13 26.45 78.53 33.68 30.51 HealthGPT-M3 79.38 33.03 33.48 71.81 31.83 43.45 85.06 34.40 25.49 84.23 34.29 27.99 HealthGPT-L14 79.73 33.10 32.96 71.92 31.87 43.09 85.31 34.29 26.20 84.96 34.14 28.13

modality conversion). Notably, by this stage, the word embedding layer and output head have been fine-tuned, only the H-LoRA modules and adapter modules need to be trained. This strategy significantly improves the model’s adaptability and flexibility across different tasks.

### 5 Experiments

##### 5.1 Data and Experimental Setup

Data Details. We curate VL-Health dataset (see Figure 4). For medical visual comprehension, we leverage multiple medical-specific datasets, including PubMedVision (Chen et al. 2024a), LLaVA-Med (Li et al. 2024b), PathVQA (He et al. 2020), MIMIC-CXR-VQA (Bae et al.

- 2024), SLAKE (Liu et al. 2021), and VQA-RAD (Lau et al. 2018). Additionally, we incorporate high-quality openworld data from LLaVA-1.5 (Liu et al. 2024b) to preserve the model’s general knowledge and instruction-following capabilities. For generation tasks, we construct a reconstruction dataset based on LLaVA-558k (Liu et al. 2024b), and also explore two key tasks in personalized medical image enhancement—super-resolution and modality conversion—using the IXI (Davies et al. 2014) and Syn-

thRAD2023 (Thummerer et al. 2023) datasets. Detailed data selection and instruction templates are in the Appendix.

Model Details. We select CLIP-L/14 (Radford et al. 2021) as the visual encoder and used the hidden states of its second and penultimate layers as concrete-grained and abstract-grained features for model’s dynamic hierarchical visual perception. Drawing on the successful experiences of LLaVA, we employ a MLP to align the multi-modal feature embeddings. We choose the parameter-efficient phi-3mini (Abdin et al. 2024) and phi-4 (Abdin et al. 2024) as the base model. For visual comprehension and generation tasks, we set the rank of H-LoRA to 16 and 64, with four experts. Additionally, we use the f8-8192 version of VQGAN as the image indexing and upsampling module.

##### 5.2 Main Experiments

Comprehension. We compare HealthGPT with several existing models, including medical-specific LVLMs (e.g., Med-Flamingo (Moor et al. 2023), LLaVA-Med (Li et al. 2024b), HuatuoGPT-Vision (Chen et al. 2024a)) as well as recent open-world LVLMs (e.g., BLIP-2 (Li et al. 2023b), LLaVA-v1.5 (Liu et al. 2024b), InstructBLIP (Dai et al. 2023), Yi-VL (Young et al. 2024), InternVL2 (Chen

Table 3: Comparison results of super-resolution task.

Model SSIM ↑ PSNR ↑ MSE ↓ LPIPS ↓

| | |
|---|---|
| | |
|SRGAN DASR Real-ESRGAN LIIF BSRGAN HealthGPT-M3 HealthGPT-L14<br><br>|71.34 32.01 41.27 24.50 71.57 32.34 38.25 19.17 67.30 31.87 42.57 20.64 73.27 32.13 40.14 22.93 69.97 31.97 41.52 28.72 78.19 32.76 34.47 12.02 77.94 32.71 35.19 12.43<br><br>|

[Figure 103]

[Figure 104]

Figure 5: Performance comparison of LoRA, MoELoRA, and H-LoRA under different rank settings.

et al. 2024b), Llama-3.2 (Dubey et al. 2024)). Additionally, we test several SOTA unified visual comprehension and generation models, including Show-o (Xie et al. 2024), Unified-IO 2 (Lu et al. 2024), and Janus (Wu et al. 2024). The experimental results are shown in Table 1, with the following key observations: (i) SOTA Results Compared with LVLMs: In medical visual comprehension tasks, HealthGPT demonstrates superior performance, significantly outperforming both medical-specific models (e.g., HuatuoGPT-Vision) and general-purpose models (e.g., Llama-3.2). (ii) Surpassing Current Unified LVLMs: Despite being trained on billions of data points, unified models still exhibit poor generalization performance in medical visual comprehension. For instance, Unified-IO 2 scored only 33.8. In contrast, HealthGPT-M3, with only 3.8B parameters, scored 61.3 on the medical multi-modal unified task, significantly outperforming existing unified models in medical downstream scenarios. (iii) Stable Improvement with Large Base Model: Our method demonstrates excellent scalability, with HealthGPT-L14 achieving a score of 66.4 in the larger model configuration. This result significantly outperforms all other models, highlighting the effectiveness of scaling up the base model for enhanced performance in medical tasks.

Generation. We study three key tasks in medical imaging. (i) Modality Conversion: In this task, we focus on the conversion between CT and MRI modalities for the brain and pelvic regions, designing four specific sub-tasks. All comparative models (Pix2Pix (Isola et al. 2017), CycleGAN (Zhu et al. 2017), BBDM (Li et al. 2023a),

###### （a） （b）

[Figure 105]

[Figure 106]

(%)

[Figure 107]

[Figure 108]

(%)

[Figure 109]

Figure 6: The loss visualization (a) and performance comparison (b) with respect to different visual perceptions.

Vmamba (Liu et al. 2024e), and DiffMa (Wang et al. 2024b)) trained a separate model for each sub-task, while HealthGPT unify all tasks into a single training process. The experimental results, shown in Table 11, demonstrate that our approach outperforms other methods across multiple evaluation metrics. For instance, in the CT2MRI-Brain task, HealthGPT-M3 achieves an SSIM of 79.38, significantly surpassing traditional methods like Pix2Pix (71.09) and the recent DiffMa (71.47). (ii) Super-Resolution: We conduct 4× super-resolution experiments on the IXI dataset, with the results presented in Table 3. Notably, most existing methods fail to fully leverage the prior knowledge of key structures in medical images, resulting in significant shortcomings in detail recovery. In contrast, our method significantly mitigates this issue. Specifically, HealthGPT-M3 excels in key metrics such as SSIM, PSNR, and ISE, achieving scores of 78.19, 32.76, and 34.47, respectively. Additionally, HealthGPT-M3 achieves the lowest score of 12.34, further validating its exceptional performance in human visual perception. (iii) Reconstruction: We compare HealthGPT-M3 with unified models with reconstruction capabilities, such as Unified-IO 2 and SEED-X. The results show that our approach performs better controllability for visual reconstruction. We also train HealthGPT-L14 with a similar number of trainable parameters to the M3 version. Hence, the similar performance between the two models meets our expectations. Details are in the Appendix.

##### 5.3 In-Depth Study

Effect of Heterogeneous Low-Rank Adaptation. H-LoRA provides an optimized multi-LoRA architecture for multitask learning. We conduct extensive validation of this structure, with results presented in Table 4, comparing the performance of LoRA, MoELoRA, and H-LoRA in medical unified comprehension and generation tasks. In the majority of comprehension tasks and all generation tasks, H-LoRA demonstrates superior performance, particularly in the OmniMedVQA benchmark, where it improved from 64.90 to 68.50. Notably, despite some applications of MoELoRA in certain scenarios, it do not show advantages in this task and

Table 4: We present the performance and speed differences of LoRA, MoELoRA (n=4), and H-LoRA (n=4) on medical visual comprehension and generation tasks.

Comp. Gen. Model VQA-RAD SLAKE PathVQA

Training Time

MMMU -Med

OMVQA RECOM MTRANS SR

close all close all close all

+LoRA 71.3 57.2 70.0 53.4 76.4 38.6 41.30 65.10 62.67 59.99 65.88 1.00× +MoELoRA 72.5 57.2 66.4 52.4 73.2 36.0 39.30 64.90 67.31 59.76 65.91 1.49× +H-LoRA 73.7 55.9 74.6 56.4 78.7 39.7 43.30 68.50 67.69 60.30 66.14 1.00×

HealthGPT w/

Table 5: Comparison between the H-LoRA-based Three-Stage Learning Strategy and the mixed-training approach.

Comp. Gen. Training Strategy VQA-RAD SLAKE PathVQA CT MRI close all close all close all

MMMU -Med

OMVQA

Brain Pelvis Brain Pelvis Mixed-Training 56.6 37.9 45.0 32.9 65.7 33.6 44.0 48.9 65.64 62.75 56.61 50.77

HealthGPT w/

3-stage-Training 72.5 55.2 77.9 59.6 79.7 49.0 42.7 68.5 70.84 72.99 65.26 61.33

[Figure 110]

Figure 7: Case study of report-to-CXR under different instructions. (a) shows a normal CXR image for comparison. (b) and (c) illustrate generated cases with varying severity and affected regions. The graffiti areas indicate abnormal conditions.

had a training time approximately 50% longer than LoRA. Figure 5 illustrates the performance of the three PEFT methods in medical visual comprehension and generation tasks across different ranks, with H-LoRA consistently outperforming the other methods in all scenarios, demonstrating significant advantages in handling diverse tasks.

Different Learning Strategy. We propose a three-stage learning strategy for H-LoRA that decouples comprehension and generation tasks. Unlike methods that train both tasks simultaneously, our approach reduces performance degradation from task conflicts (see Table 5). In the medical visual comprehension task, mixed training causes catastrophic forgetting and degrades visual reconstruction, whereas our strategy effectively uses the medical embedding knowledge in pre-trained LLMs to mitigate these conflicts. Meanwhile, we examine how fusing heterogeneous H-LoRA plugins in the second training stage results in minimal performance degradation. Detailed results are in the Appendix.

Hierarchical Visual Perception Analysis. We conduct an ablation analysis on visual perceptual inputs for comprehension and generation tasks. Figure 6 shows that comprehension tasks converge more efficiently with abstract-grained

inputs, while generation tasks perform better with concretegrained inputs. This highlights the importance of the hierarchical visual perception we propose, suggesting that tailoring visual inputs for specific tasks at different hierarchies can significantly improve efficiency.

Report-to-CXR Task. We further explore the medical image generation task without reference images, using a small amount of MIMIC-CXR data (Johnson et al. 2019) for instruction fine-tuning. Figure 7 annotates images with varying injury degrees and locations, comparing them to healthy CXR images. We observe that HealthGPT effectively generates CXR images based on the instructions, showcasing its potential in healthcare education and auxiliary diagnosis.

### 6 Conclusion

In this paper, we introduce HealthGPT, a Med-LVLM that unifies medical vision-language comprehension and generation through a novel heterogeneous knowledge adaptation approach. Experimental results demonstrate that HealthGPT achieves significant performance improvements across multiple medical comprehension and generation tasks, showcasing its potential for healthcare applica-

tions.

### References

Abdin, M.; Aneja, J.; Behl, H.; Bubeck, S.; Eldan, R.; Gunasekar, S.; Harrison, M.; Hewett, R. J.; Javaheripi, M.; Kauffmann, P.; et al. 2024. Phi-4 technical report. arXiv preprint arXiv:2412.08905.

Bae, S.; Kyung, D.; Ryu, J.; Cho, E.; Lee, G.; Kweon, S.; Oh, J.; JI, L.; Chang, E.; Kim, T.; et al. 2024. MIMIC-ExtMIMIC-CXR-VQA: A Complex, Diverse, And Large-Scale Visual Question Answering Dataset for Chest X-ray Images. Chen, J.; Gui, C.; Ouyang, R.; Gao, A.; Chen, S.; Chen, G. H.; Wang, X.; Zhang, R.; Cai, Z.; Ji, K.; et al. 2024a. Huatuogpt-vision, towards injecting medical visual knowledge into multimodal llms at scale. arXiv preprint

- arXiv:2406.19280. Chen, Z.; Wang, W.; Tian, H.; Ye, S.; Gao, Z.; Cui, E.; Tong,

- W.; Hu, K.; Luo, J.; Ma, Z.; et al. 2024b. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821.

Chern, E.; Su, J.; Ma, Y.; and Liu, P. 2024. Anole: An open, autoregressive, native large multimodal models for interleaved image-text generation. arXiv preprint

- arXiv:2407.06135.

Dai, W.; Li, J.; Li, D.; Tiong, A. M. H.; Zhao, J.; Wang, W.; Li, B.; Fung, P.; and Hoi, S. 2023. InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning. arXiv:2305.06500.

Davies, R. L.; Royston, P. A.; Leung, M. S.; Haider, M. E. A. M. J.; Barkhof, S. G. A. L.; and B., P. E. T. M. 2014. The IXI Dataset. Accessed: 2025-01-30.

Ding, N.; Qin, Y.; Yang, G.; Wei, F.; Yang, Z.; Su, Y.; Hu, S.; Chen, Y.; Chan, C.-M.; Chen, W.; et al. 2023. Parameter-efficient fine-tuning of large-scale pre-trained language models. Nature Machine Intelligence, 5(3): 220– 235.

Dong, R.; Han, C.; Peng, Y.; Qi, Z.; Ge, Z.; Yang, J.; Zhao, L.; Sun, J.; Zhou, H.; Wei, H.; et al. 2023. Dreamllm: Synergistic multimodal comprehension and creation. arXiv preprint arXiv:2309.11499.

Dubey, A.; Jauhri, A.; Pandey, A.; Kadian, A.; Al-Dahle, A.; Letman, A.; Mathur, A.; Schelten, A.; Yang, A.; Fan, A.; et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Esser, P.; Rombach, R.; and Ommer, B. 2021. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 12873–12883.

Ge, Y.; Ge, Y.; Zeng, Z.; Wang, X.; and Shan, Y. 2023. Planting a seed of vision in large language model. arXiv preprint arXiv:2307.08041.

Ge, Y.; Zhao, S.; Zhu, J.; Ge, Y.; Yi, K.; Song, L.; Li, C.; Ding, X.; and Shan, Y. 2024. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396.

He, X.; Zhang, Y.; Mou, L.; Xing, E.; and Xie, P. 2020. Pathvqa: 30000+ questions for medical visual question answering. arXiv preprint arXiv:2003.10286.

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; and Chen, W. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.

Hu, Y.; Li, T.; Lu, Q.; Shao, W.; He, J.; Qiao, Y.; and Luo, P. 2024. Omnimedvqa: A new large-scale comprehensive evaluation benchmark for medical lvlm. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 22170–22183.

Isola, P.; Zhu, J.-Y.; Zhou, T.; and Efros, A. A. 2017. Imageto-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, 1125–1134.

Johnson, A. E.; Pollard, T. J.; Greenbaum, N. R.; Lungren, M. P.; Deng, C.-y.; Peng, Y.; Lu, Z.; Mark, R. G.; Berkowitz, S. J.; and Horng, S. 2019. MIMIC-CXR-JPG, a large publicly available database of labeled chest radiographs. arXiv preprint arXiv:1901.07042.

Lau, J. J.; Gayen, S.; Ben Abacha, A.; and DemnerFushman, D. 2018. A dataset of clinically generated visual questions and answers about radiology images. Scientific data, 5(1): 1–10.

- Li, B.; Xue, K.; Liu, B.; and Lai, Y.-K. 2023a. Bbdm: Imageto-image translation with brownian bridge diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern Recognition, 1952–1961.
- Li, C.; Wong, C.; Zhang, S.; Usuyama, N.; Liu, H.; Yang,

- J.; Naumann, T.; Poon, H.; and Gao, J. 2024a. Llavamed: Training a large language-and-vision assistant for biomedicine in one day. Advances in Neural Information Processing Systems, 36. Li, C.; Wong, C.; Zhang, S.; Usuyama, N.; Liu, H.; Yang,
- J.; Naumann, T.; Poon, H.; and Gao, J. 2024b. Llavamed: Training a large language-and-vision assistant for biomedicine in one day. Advances in Neural Information Processing Systems, 36.

Li, J.; Li, D.; Savarese, S.; and Hoi, S. 2023b. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, 19730–19742. PMLR.

Lin, T.; Liu, J.; Zhang, W.; Li, Z.; Dai, Y.; Li, H.; Yu, Z.; He, W.; Li, J.; Jiang, H.; et al. 2024. Teamlora: Boosting lowrank adaptation with expert collaboration and competition. arXiv preprint arXiv:2408.09856.

Liu, B.; Zhan, L.-M.; Xu, L.; Ma, L.; Yang, Y.; and Wu,

- X.-M. 2021. Slake: A semantically-labeled knowledgeenhanced dataset for medical visual question answering. In 2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI), 1650–1654. IEEE.

Liu, D.; Zhao, S.; Zhuo, L.; Lin, W.; Qiao, Y.; Li, H.; and Gao, P. 2024a. Lumina-mgpt: Illuminate flexible photorealistic text-to-image generation with multimodal generative pretraining. arXiv preprint arXiv:2408.02657.

Liu, H.; Li, C.; Li, Y.; and Lee, Y. J. 2024b. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 26296–26306.

Liu, H.; Li, C.; Li, Y.; Li, B.; Zhang, Y.; Shen, S.; and Lee,

- Y. J. 2024c. LLaVA-NeXT: Improved reasoning, OCR, and world knowledge. https://llava-vl.github.io/blog/2024-0130-llava-next/.

Liu, H.; Li, C.; Wu, Q.; and Lee, Y. J. 2023. Visual Instruction Tuning. In NeurIPS.

Liu, Q.; Wu, X.; Zhao, X.; Zhu, Y.; Xu, D.; Tian, F.; and Zheng, Y. 2024d. When moe meets llms: Parameter efficient fine-tuning for multi-task medical applications. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, 1104– 1114.

Liu, Y.; Tian, Y.; Zhao, Y.; Yu, H.; Xie, L.; Wang, Y.; Ye, Q.; and Liu, Y. 2024e. VMamba: Visual State Space Model. arXiv preprint arXiv:2401.10166.

Lu, J.; Clark, C.; Lee, S.; Zhang, Z.; Khosla, S.; Marten, R.; Hoiem, D.; and Kembhavi, A. 2024. Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision Language Audio and Action. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 26439– 26455.

Lu, J.; Clark, C.; Zellers, R.; Mottaghi, R.; and Kembhavi, A. 2022. Unified-io: A unified model for vision, language, and multi-modal tasks. In The Eleventh International Conference on Learning Representations.

Luo, T.; Lei, J.; Lei, F.; Liu, W.; He, S.; Zhao, J.; and Liu, K. 2024a. Moelora: Contrastive learning guided mixture of experts on parameter-efficient fine-tuning for large language models. arXiv preprint arXiv:2402.12851.

Luo, Y.; Zhang, J.; Fan, S.; Yang, K.; Hong, M.; Wu, Y.; Qiao, M.; and Nie, Z. 2024b. Biomedgpt: An open multimodal large language model for biomedicine. IEEE Journal of Biomedical and Health Informatics.

Masoudnia, S.; and Ebrahimpour, R. 2014. Mixture of experts: a literature survey. Artificial Intelligence Review, 42: 275–293.

Moor, M.; Huang, Q.; Wu, S.; Yasunaga, M.; Dalmia, Y.; Leskovec, J.; Zakka, C.; Reis, E. P.; and Rajpurkar, P. 2023.

Med-flamingo: a multimodal medical few-shot learner. In Machine Learning for Health (ML4H), 353–367. PMLR.

Nath, V.; Li, W.; Yang, D.; Myronenko, A.; Zheng, M.; Lu, Y.; Liu, Z.; Yin, H.; Law, Y. M.; Tang, Y.; et al. 2024. Vilam3: Enhancing vision-language models with medical expert knowledge. arXiv preprint arXiv:2411.12915.

OpenAI. 2023. GPT-4V(ision) System Card. https://cdn. openai.com/papers/GPTV System Card.pdf.

Pan, K.; Tang, S.; Li, J.; Fan, Z.; Chow, W.; Yan, S.; Chua, T.-S.; Zhuang, Y.; and Zhang, H. 2024. AutoEncoding Morph-Tokens for Multimodal LLM. arXiv preprint arXiv:2405.01926.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, 8748–8763. PMLR.

Singhal, K.; Azizi, S.; Tu, T.; Mahdavi, S. S.; Wei, J.; Chung, H. W.; Scales, N.; Tanwani, A.; Cole-Lewis, H.; Pfohl, S.; et al. 2023. Large language models encode clinical knowledge. Nature, 620(7972): 172–180.

Team, C. 2024. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818.

Thawkar, O.; Shaker, A.; Mullappilly, S. S.; Cholakkal, H.; Anwer, R. M.; Khan, S.; Laaksonen, J.; and Khan, F. S. 2023. Xraygpt: Chest radiographs summarization using medical vision-language models. arXiv preprint arXiv:2306.07971.

Thummerer, A.; van der Bijl, E.; Galapon Jr, A.; Verhoeff, J. J.; Langendijk, J. A.; Both, S.; van den Berg, C. N. A.; and Maspero, M. 2023. SynthRAD2023 Grand Challenge dataset: Generating synthetic CT for radiotherapy. Medical physics, 50(7): 4664–4674.

Tian, D.; Jiang, S.; Zhang, L.; Lu, X.; and Xu, Y. 2023. The role of large language models in medical image processing: a narrative review. Quantitative Imaging in Medicine and Surgery, 14(1): 1108.

Tong, S.; Fan, D.; Zhu, J.; Xiong, Y.; Chen, X.; Sinha, K.; Rabbat, M.; LeCun, Y.; Xie, S.; and Liu, Z. 2024. MetaMorph: Multimodal Understanding and Generation via Instruction Tuning. arXiv preprint arXiv:2412.14164.

Tu, T.; Azizi, S.; Driess, D.; Schaekermann, M.; Amin, M.; Chang, P.-C.; Carroll, A.; Lau, C.; Tanno, R.; Ktena, I.; et al. 2024. Towards generalist biomedical AI. NEJM AI, 1(3): AIoa2300138.

Vig, J. 2019. A multiscale visualization of attention in the transformer model. arXiv preprint arXiv:1906.05714.

Wang, X.; Zhang, X.; Luo, Z.; Sun, Q.; Cui, Y.; Wang, J.; Zhang, F.; Wang, Y.; Li, Z.; Yu, Q.; et al. 2024a. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869.

Wang, Z.; Wu, Z.; Agarwal, D.; and Sun, J. 2022. Medclip: Contrastive learning from unpaired medical images and text. arXiv preprint arXiv:2210.10163.

Wang, Z.; Zhang, L.; Wang, L.; and Zhang, Z. 2024b. Soft Masked Mamba Diffusion Model for CT to MRI Conversion. arXiv preprint arXiv:2406.15910.

Wu, C.; Chen, X.; Wu, Z.; Ma, Y.; Liu, X.; Pan, Z.; Liu, W.; Xie, Z.; Yu, X.; Ruan, C.; and Luo, P. 2024. Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation. arXiv:2410.13848.

Wu, S.; Fei, H.; Qu, L.; Ji, W.; and Chua, T.-S. 2023. Next-gpt: Any-to-any multimodal llm. arXiv preprint arXiv:2309.05519.

Xie, J.; Mao, W.; Bai, Z.; Zhang, D. J.; Wang, W.; Lin, K. Q.; Gu, Y.; Chen, Z.; Yang, Z.; and Shou, M. Z. 2024. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528.

Young, A.; Chen, B.; Li, C.; Huang, C.; Zhang, G.; Zhang, G.; Li, H.; Zhu, J.; Chen, J.; Chang, J.; et al. 2024. Yi: Open foundation models by 01. ai. arXiv preprint arXiv:2403.04652.

Zhou, H.; Liu, F.; Gu, B.; Zou, X.; Huang, J.; Wu, J.; Li, Y.; Chen, S. S.; Zhou, P.; Liu, J.; et al. 2023. A survey of large language models in medicine: Progress, application, and challenge. arXiv preprint arXiv:2311.05112.

Zhu, J.-Y.; Park, T.; Isola, P.; and Efros, A. A. 2017. Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE international conference on computer vision, 2223–2232.

### Appendix

This is the Appendix for “HealthGPT: A Medical Large Vision-Language Model for Unifying Comprehension and Generation via Heterogeneous Knowledge Adaptation”. This Appendix is organized as follows:

- • Section A presents the experimental implementation details, the training process of HealthGPT, and the specifics of VL-Health.
- • Section B systematically provides an analysis of Heterogeneous Low-Rank Adaptation.
- • Section C shows supplementary experimental results to validate the effectiveness of HealthGPT. A Implementation Details

##### A.1 Model Details

We employ CLIP-L/14 (Radford et al. 2021) as the visual feature extractor, extracting both shallow and deep features to serve as visual tokens. The model uses alignment adapters, implemented with two-layer MLPs, to align shallow features, representing concrete visual granularity, and deep features, representing abstract visual granularity. These visual tokens are concatenated with text tokens and input into the large language models (LLMs).

HealthGPT offers two versions: HealthGPT-M3 and HealthGPT-L14, which are based on Phi-3-mini (Abdin et al. 2024) and Phi-4 (Abdin et al. 2024) as the pre-trained LLMs, respectively. In addition, we expand the LLM vocabulary with 8192 VQ indices derived from VQGAN-f8-8192 (Esser, Rombach, and Ommer 2021), serving as multi-modal tokens to further augment the model’s capacity for understanding both visual and textual input. Figure 6 shows the details.

Table 6: Overview of the Components of HealthGPT.

Model ViT Adapter MLP-dims Model dims LLM Params Vocab Size H-LoRA Rank HealthGPT-M3 CLIP-L/14 2-layer MLP 1024 3072 Phi-3-mini 3.8B 40206 16(Comp.), 64(Gen.) HealthGPT-L14 CLIP-L/14 2-layer MLP 1024 5120 Phi-4 14B 108547 8(Comp.), 32(Gen.)

##### A.2 Training Details

In this study, we propose a three-stage learning strategy that is compatible with our innovative heterogeneous low-rank adaptation (H-LoRA). We provide a detailed hyperparameter configuration for the model’s three-stage training process. The specific hyperparameter settings used are listed in Table 7. These hyperparameters are crucial for ensuring the model’s learning efficacy and final performance.

Table 7: Overview of Hyperparameter Configurations.

HealthGPT-M3 HealthGPT-L14 Hyperparameter Stage-1 Stage-2 Stage-3 Stage-1 Stage-2 Stage-3

Comp. Gen. Comp. Gen. Comp. Gen. Comp. Gen. Comp. Gen. Comp. Gen. Optimizer AdamW AdamW AdamW AdamW AdamW AdamW Adapter LR 1e-3 2e-5 2e-5 2e-5 1e-3 2e-5 2e-5 2e-5 Learning Rate / 2e-4 2e-4 2e-4 / 1e-4 2e-4 2e-4 Global Batch Size 256 64 32 128 64 256 64 32 128 64 Weight Decay 0 0 0 0 0 0 Dropout Rate 0 0.05 0.05 0.05 0 0.05 0.05 0.05 LR Scheduler Warm Up Constant Warm Up Warm Up Constant Warm Up Max Sequence Length 2048 2048 2048 2048 2048 2048

It is worth noting that we sometimes observe instances of loss spikes during the training of medical visual comprehension and generation tasks. Through repeated validation, we discovered that larger model parameters and learning rates tend to lead to this issue, which is the reason for the slight differences in hyperparameters between HealthGPT-M3 and HealthGPT-L14.

（a） （b）

[Figure 111]

[Figure 112]

| |
|---|

Figure 8: VL-Health dataset collection distribution.

##### A.3 VL-Health

The construction of the VL-Health dataset involves two key steps: (i) data collection, (ii) data processing, as detailed below: Data Collection: During the collection phase, we carefully considered the diversity of medical images and the complexity of the tasks, selecting appropriate subsets for comprehension and generation tasks. For comprehension tasks, we selected datasets such as VQA-RAD (Lau et al. 2018), SLAKE (Liu et al. 2021), PathVQA (He et al. 2020), and MIMIC-CXR-VQA (Bae et al. 2024), which cover various medical imaging modalities like radiology and pathology, and include professional annotations to assist the model in learning tasks such as lesion detection and disease diagnosis. Additionally, large-scale multi-modal datasets like LLaVA-Med (Li et al. 2024b) and PubMedVision (Chen et al. 2024a) were included to provide broader medical knowledge support and facilitate the training of complex reasoning tasks. For generation tasks, we focused on four mainstream task categories: super-resolution image generation, modality conversion, text-to-image generation, and image reconstruction. The IXI (Davies et al. 2014) dataset, containing a large number of healthy brain MRI images, is suitable for training superresolution models; the MIMIC-CHEST-XRAY (Bae et al. 2024) dataset, with X-ray images and their corresponding textual reports, is appropriate for text-to-image generation tasks; the SynthRAD2023 (Thummerer et al. 2023) dataset provides a large number of paired CT and MRI images, supporting modality conversion model training; for image reconstruction tasks, we rewrote and adjusted the LLaVA-558k (Liu et al. 2024b) dataset.

Data Processing: After data collection, we performed filtering and processing of the raw data. For VisualQA tasks, we standardized the data entries into two forms: open-ended questions and single-choice questions, enabling flexible training and evaluation. Additionally, considering that multi-image data has a minimal impact on performance but introduces extra padding and training time, we excluded multi-image data. For the scanned image data in generation tasks, we applied slicing extraction, image registration, data augmentation, and normalization to treat 2D images as visual inputs for model training or used VQGAN-generated indices to supervise the generation tasks.

Data Statistics This section provides detailed statistical information about the VL-Health dataset to offer a more comprehensive understanding.

Data Overview: To ensure a balanced development of the model’s comprehension and generation capabilities, in addition to the LLaVA-558k and PubMedVision-PT datasets used for alignment, the VL-Health dataset ultimately selected 765,802 additional visual question-answering (VQA) training samples (to endow the model with visual comprehension and instructionfollowing capabilities) and 783,045 generation training samples (to provide the model with reconstruction and visual generation instruction-following abilities). This contributes to the transfer of knowledge between comprehension and generation tasks, enhancing the model’s overall performance. For medical image comprehension tasks, images were selected from VQA-RAD (approximately 450 images), SLAKE (approximately 630 images), PathVQA (approximately 2,600 images), MIMIC-CXR-VQA (approximately 52,000 images), LLaVA-Med (approximately 61,000 images), and PubMedVision (approximately 500,000 images). Multiple question-answer pairs were retained for each image to enhance the model’s understanding and generalization of the image content. Table 8 shows the data distribution of VL-Health for three-stage learning strategy, where mixed-47k is based on the sampling of all data in stage-1.

Diversity and Quality Assessment: VL-Health covers 11 modalities, including CT, MRI, X-ray, microscopy, OCT, ultrasound, and fundus photography, which aids the model in learning features from various modalities. The dataset also encompasses a wide range of diseases, from common to rare, and from localized lesions to systemic diseases, including pulmonary diseases, skeletal abnormalities, brain lesions, tumors, cardiovascular diseases, and cellular abnormalities. This provides comprehensive training support to the model, enabling it to learn the characteristics and diagnosis of various diseases.

Table 8: Data distribution of VL-Health in three-stage learning strategy.

|Medical Task|Stage-1|Stage-2|
|---|---|---|
| | | |

Comp. LLaVA-558k, PubMedVision-PT

Mixed-47k

Gen. LLaVA-558k Medical Task Stage-3

Comp. LLaVA Med, MIMIC CXR VQA, PubMedVision-FT, LLaVA-665k, PathVQA, SLAKE, VQA-RAD Gen. IXI, SynthRAD2023, MIMIC-CHEST-XRAY

Data Format. All data samples are converted into a unified instruction-response format for training and evaluation. Specifically, the VL-Health dataset consists of the following components:

- • Task Type: Specifies the granularity of visual features output by the visual encoder and selects the corresponding HLoRA submodule. For generation tasks, the response also includes multi-modal tokens corresponding to VQ indices.
- • Task Instruction: Guides the model to interpret the image and generate a response, covering various aspects of the image and specifying the output format.
- • Response: The textual output generated based on the task instruction and input image, ensuring it meets the question and formatting requirements.
- • Input Image: Provides the visual signal for the model to process.
- • Target Image Index: In generation tasks, this is added as a multi-modal token to the response for autoregressive generation.

### B Analysis of Heterogeneous Low-Rank Adaptation

We propose H-LoRA, which utilizes hard routing selection to allocate plugins for knowledge learning and representation across tasks, thereby preventing conflicts arising from heterogeneous knowledge. Furthermore, within each task, we optimized based on MoELoRA, enhancing performance while reducing computational overhead. The pseudocode is detailed Algorithm 1.

Algorithm 1: H-LoRA Algorithm

Input: concrete-grained visual features FCon, abstract-grained visual features FAbs, comprehension-based H-LoRA modules ({AComp.i }ki=1,RComp.outer ), generation-based H-LoRA modules ({AGen.i }ki=1,RGen.outer), task type T (comprehension or generation), number of LoRA experts k, origin linear layer weights W0, text features T , hidden state h

Output: final output O // Select task-specific image features if T = generation task then

Fimg ← FCon

else if T = comprehension task then

Fimg ← FAbs end if U ← concat(Fimg,T ) // Concatenate image features and text features {Ai}ki=1,{Bi}ki=1,Router ← {ATi }ki=1,{BiT}ki=1,RTouter // Assign task-specific H-LoRA submodule // Merge LoRA experts’ matrices

Amerged ← concat({Ai}ki=1) Bmerged ← concat({Bi}ki=1) W ← R(h) // Generate routing weights based on input hidden state x

Wexpanded ← α × W/r ⊗ 1r // Expand routing weights to match merged matrices OH-LoRA ← (x · Amerged ⊙ Wexpanded) · Bmerged // Compute H-LoRA output using element-wise multiplication O ← x · W0 + OH-LoRA // Add H-LoRA output to pre-trained weights to get final output Return O

We further analyzed the computational overhead differences between MoELoRA and H-LoRA. Assuming that both methods use the same number of LoRA experts k, we can compare their time complexity from the perspective of the operational steps involved.

Computational Overhead of MoELoRA. In MoELoRA, the operations involving the expert matrix mainly include the following steps: (i) Expert Multiplication: MoELoRA requires 2k multiplications with the LoRA experts. (ii) Router Multiplication: One multiplication with the Router is required. (iii) Router Output Expansion: MoELoRA needs to perform k

expansion operations on the Router’s output weights to generate the appropriate shapes that match the dimensions of the input and LoRA experts while iterating through the experts. (iv) Dot Product: For each expanded Router weight, a dot product with the intermediate state of the expert is required, resulting in k multiplications. (v) Addition: Finally, k addition operations are required to accumulate the results from each LoRA expert into the final output. Assuming the time complexity of each operation is the same, the additional time complexity introduced when equipping a fully connected layer with MoELoRA is: O(2k+1+k+k+k) = O(5k+1). Thus, MoELoRA introduces an additional time overhead of O(5k+1) during computation. H-LoRA. In contrast to MoELoRA, H-LoRA reduces the computational overhead by concatenating the LoRA expert matrices. Specifically: (i) Expert Multiplication: H-LoRA merges all LoRA experts by directly creating a larger A and B matrix, instead of performing independent operations for each expert. This process can be implemented through matrix initialization without additional concatenation operations. Therefore, only 2 multiplications with the LoRA experts are required. (ii) Router Multiplication: H-LoRA still requires one multiplication with the Router. (iii) Router Output Expansion: H-LoRA only requires one expansion operation on the Router’s output weights. (iv) Dot Product: H-LoRA only requires one dot product between the Router’s output and the expert’s intermediate state. (v) Addition: Finally, H-LoRA only requires one addition operation to accumulate the LoRA expert results into the intermediate state. Therefore, the additional time complexity introduced by H-LoRA is: O(2 + 1 + 1 + 1 + 1) = O(6).

Comparing the two, we see that MoELoRA introduces a linear increase in additional time complexity with respect to the number of experts k, resulting in a complexity of O(5k + 1), while H-LoRA’s additional time complexity is fixed at O(6), independent of k. We observe that when k is small, the time complexity differences between MoELoRA and H-LoRA are negligible. However, as k increases, MoELoRA’s computational overhead grows linearly, while H-LoRA’s remains constant. This makes H-LoRA significantly more computationally efficient than MoELoRA, particularly in large-scale tasks. We will further demonstrate the significant advantage of H-LoRA in training time in subsequent experiments, validating its efficiency in practical applications.

### C Supplemental Experimental Results

In this section, we include additional experiments to demonstrate the superiority of HealthGPT and articulate our design philosophy.

##### C.1 Results: OmniMedVQA Benchmark

OmniMedVQA (Hu et al. 2024) is a novel, large-scale medical visual question answering (VQA) benchmark designed to encompass various modalities and anatomical regions by collecting diverse images from multiple medical datasets. Our experimental results are presented in Table 9.

Table 9: Performance comparison of OmniMedVQA Benchmark.

OmniMedVQA ↑ Type Model # Params

Medical LVLM CT X-ray FDM MiS OCT MRI USS Avg.

|Comp. Only<br><br>|Med-Flamingo 8.3B ✓ LLaVA-Med 7B ✓ HuatuoGPT-Vision 7B ✓ BLIP-2 6.7B ✗ LLaVA-v1.5 7B ✗ InstructBLIP 7B ✗ Yi-VL 6B ✗ InternVL2 8B ✗ Llama-3.2 11B ✗|30.1 33.9 25.5 37.0 60.0 27.6 30.4 28.4 32.8 42.7 31.6 55.3 45.0 53.6 35.3 41.5 51.4 62.3 59.3 40.4 60.1 26.6 29.1 22.3 36.9 29.1 22.7 21.4 28.0 55.7 35.5 42.1 49.2 52.9 49.7 20.1 22.2 34.1 30.6 38.6 31.9 25.5 51.2 47.1 27.7 62.6 67.6 55.0 40.3 40.2 57.9 53.2 64.0 59.1 58.1 49.1 37.6 55.2 71.4 82.1 62.5 65.2 68.6|34.9 41.3 50.0 26.9 44.7 29.0 50.2 54.5 63.2|
|---|---|---|---|
|Comp. & Gen.|Show-o 1.3B ✗ Unified-IO 2 7B ✗ Janus 1.3B ✗ HealthGPT-M3 3.8B ✓ HealthGPT-L14 14B ✓<br><br>|29.0 50.4 30.9 22.0 30.8 34.2 33.8 10.8 37.7 12.3 25.3 32.6 30.9 37.7 24.9 54.8 35.9 62.7 54.2 50.7 36.8 35.3 81.9 54.6 88.2 89.3 78.5 51.4 39.0 86.6 64.1 88.6 99.7 80.9 62.2<br><br>|33.0 26.8 45.7 68.5 74.4<br><br>|

Through our analysis, we make the following observations: (i) HealthGPT-M3 outperforms other models in 4 out of 7 sub-tasks, achieving an average score that exceeds cutting-edge medical Large Vision-Language Models (LVLMs) as well as

general LVLMs; (ii) the unified model demonstrates relatively weak performance on OmniMedVQA; however, our approach effectively mitigates performance degradation caused by generation tasks, serving as a unified model; (iii) HealthGPT-L14 excels across all sub-tasks, achieving optimal or near-optimal results with an average score of 74.4, significantly surpassing other models.

##### C.2 Stability Analysis of Number of Experts

We investigated the impact of the number of LoRA experts on model performance within a multi-LoRA architecture, conducting extensive experiments on MoELoRA and H-LoRA with varying numbers of experts. The experimental results are presented in Table 10. As the number of experts increases, the training time for MoELoRA is significantly prolonged. When n = 8, the training time for MoELoRA is twice that of LoRA, whereas H-LoRA incurs no additional training delay and performs better. It is estimated that at n = 32, the training time for MoELoRA could reach eight times that of LoRA, preventing it from completing training and inference. This result aligns with the analysis in Appendix B, indicating that H-LoRA not only avoids introducing additional training delays compared to LoRA but also outperforms MoELoRA.

Table 10: We explored the performance of MoELoRA and H-LoRA with different numbers of LoRA experts. At n = 32, MoELoRA was unable to complete training.

n=2 n=4 n=8 n=32 Model

| |Comp. Gen. Time<br><br>|Comp. Gen. Time<br><br>|Comp. Gen. Time<br><br>|Comp. Gen. Time<br><br>|
|---|---|---|---|---|
| | | | | |
|+MoELoRA HealthGPT w/<br><br>+H-LoRA|50.3 62.98 1.22×<br><br>51.5 63.48 0.99×<br><br><br>|50.0 64.33 1.49× 52.8 64.71 1.00×<br><br>|50.8 63.71 2.09× 53.6 64.98 0.99×<br><br>|/ / 5.81× 53.5 64.74 1.01×<br><br>|

##### C.3 Impact of Heterogeneous Knowledge Fusion on Performance

Traditional unified models often utilize mixed training methods, which may result in performance degradation due to variations in task modes. To address this, we propose a three-phase learning strategy to support H-LoRA, effectively mitigating inter-task conflicts. Specifically, the second phase (Heterogeneous H-LoRA Plugin Adaptation) integrates LLMs with different H-LoRA plugins into a new unified foundation by mixing the training of the embedding layers and output heads for two tasks. Figure 9 illustrates the impact of this phase on the performance of medical comprehension and generation tasks. We observe that the second phase effectively unifies the model with minimal impact on overall performance, significantly alleviating the conflict issues arising from mixed training in medical scenarios.

##### C.4 Human Evaluation.

We further conduct human evaluation on the VQA-RAD, SLAKE, and PathVQA benchmarks, which contain 1,000 open-ended questions. Specifically, we recruit 5 clinicians to rank the randomly shuffled responses from HealthGPT-L14, LLaVA-Med, HuatuoGPTVision, Llama-3.2, InternVL-2 and Show-o. During the evaluation, questions were randomly selected, and the model-generated responses were anonymized and ranked. The results, as shown in Figure 10, indicate that HealthGPT was frequently selected as the best answer. This suggests that HealthGPT has further application potential in medical care scenarios.

##### C.5 Reconstruction Performance

Currently, unified models that align visual features based on reconstruction tasks include pre-LVLMs, post-LVLMs, as well as UnifiedIO 2 (Lu et al. 2024) and SEED-X (Ge et al. 2024). To investigate the controllability of visual generation in rigorous settings such as medical contexts, we evaluated the performance of these models in medical image reconstruction in Table 11. Experimental results demonstrate that HealthGPT exhibits the most stable reconstruction performance with a small amount of data.

67.7 67.0

65.7 65.4

[Figure 113]

[Figure 114]

Figure 9: Performance changes before and after the stage-2.

##### C.6 Case Study

Figures 11 and 12 illustrate examples of modality transformation and super-resolution reconstruction. In Figure 11, the results generated by our method in the CT (MRI) to MRI (CT) transformation task are highly close to the ground truth, effectively guiding the model in the transformation across different regions. For the MRI super-resolution reconstruction task, Figure 12 demonstrates the accuracy of our method in restoring scan image details, accurately reconstructing the essential details of the image.

Table 11: The experimental results for the four reconstruction tasks.

CT(Brain) CT(Pelvis) MRI (Brain) MRI(Pelvis) Model

SSIM ↑ PSNR ↑ MSE ↓ SSIM ↑ PSNR ↑ MSE ↓ SSIM ↑ PSNR ↑ MSE ↓ SSIM ↑ PSNR ↑ MSE ↓

SEED-X 20.18 27.66 112.11 21.53 28.02 102.87 4.90 27.62 112.86 6.31 27.89 106.21 Unified-IO 2 83.93 36.09 17.95 85.36 35.10 25.46 87.50 34.25 25.47 86.31 33.53 29.80 HealthGPT-M3 91.73 36.42 15.46 94.26 37.30 12.53 88.76 33.97 27.05 84.40 33.11 32.62

(a)

(b)

[Figure 115]

5.62

HealthGPT LLaVA-Med HuatuoGPT-Vision Llama-3.2 InternVL-2 Show-o

[Figure 116]

9.16

34.08

13.33

Human Evaluation

21.94

15.87

Figure 10: (a) Proportion of model responses selected as the best in human evaluation. (b) Human Evaluation Dataset.

[Figure 117]

###### Figure 11: Case of modality transfer.

[Figure 118]

###### Figure 12: Case of MRI image super-resolution.

