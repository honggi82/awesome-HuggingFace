# arXiv:2510.18795v3[cs.CV]7Jan2026

## ProCLIP: Progressive Vision-Language Alignment via LLM-based Embedder

##### Xiaoxing Hu1,2*, Kaicheng Yang3*, Ziyang Gong1, Qi Ming4, Zonhao Guo5, Yu Tian 6, Xiang An3, Ziyong Feng3, Xue Yang1

1Shanghai Jiao Tong University, 2Beijing Institute of Technology, 3DeepGlint 4Beijing University of Technology, 5Tsinghua University

∗Equal contribution Corresponding author

##### Abstract

Contrastive Language-Image Pre-training (CLIP) is constrained by its 77-token limit, lack of multilingual support, and coarse-grained semantic understanding. While replacing CLIP’s text encoder with a Large Language Model (LLM) offers a potential solution, direct contrastive alignment often disrupts the established cross-modal representation due to the lack of alignment priors between the two encoders. In this paper, we present ProCLIP, a curriculum learning framework designed to bridge LLMs and CLIP’s visual space, thereby unlocking CLIP’s potential for long-text, multilingual, and fine-grained understanding. The framework consists of two stages: ❶ Representation Inheritance, which distills CLIP’s original text-space knowledge into the LLM to establish an initial VL prior, and ❷ Contrastive Tuning, which refines the image-text alignment using self-distillation regularization to prevent catastrophic forgetting. To ensure semantic consistency, we introduce instance semantic alignment and embedding structure alignment losses throughout both stages. Extensive experiments demonstrate that ProCLIP improves zero-shot classification by 6.8%–13.5% and exhibits superior performance in long-text, multilingual, and fine-grained cross-modal retrieval tasks. The Code is available at https:

//github.com/VisionXLab/ProCLIP

##### 1 Introduction

CLIP (Radford et al., 2021) establishes robust joint vision-language representations via large-scale contrastive learning, serving as a cornerstone for tasks like retrieval (Yang et al., 2023), generation (Wang et al., 2022), and detection (Wu et al., 2023b). Despite its success, CLIP is constrained by its 77token English-only supervisory signal (Zhang et al., 2024; Tschannen et al., 2025). Furthermore, the lack of fine-grained textual supervision hampers

[Figure 1]

Figure 1: Centered Kernel Alignment (CKA) analysis of visual representations. ProCLIP outperforms the LLM2CLIP in retaining pretrained knowledge, providing a robust foundation for alignment.

its ability to capture nuanced semantic details (Hu et al., 2025).

To overcome CLIP’s limitations, LongCLIP (Zhang et al., 2024) interpolates positional embeddings for longer inputs, yet struggles with fine-grained and multilingual understanding. Meanwhile, FLAME (Cao et al., 2025) and LLM2CLIP (Huang et al., 2024) replace CLIP’s text encoder with LLM-based embedders (BehnamGhader et al., 2024; Lee et al., 2024) to leverage their rich linguistic priors. However, these methods typically perform "from-scratch alignment," directly forcing the image encoder to align with the LLM via contrastive learning. By neglecting CLIP’s original pretrained knowledge, this approach risks overfitting and undermines generalization, especially under data-scarce conditions. This motivates a critical question: How can we systematically leverage CLIP’s pretrained knowledge to achieve efficient crossmodal alignment with LLMs while preserving generalization?

We answer this question by proposing ProCLIP, a progressive alignment framework that reconceptualizes pretrained weights as foundational priors to bridge LLMs and CLIP’s visual space. ProCLIP

adopts a two-stage curriculum:

❶ Representation Inheritance: We distill knowledge from the original CLIP text encoder into a lightweight adapter for the LLM. This anchors the frozen LLM to CLIP’s textual space, establishing a robust initial prior.

❷ Contrastive Tuning: Building on this foundation, we refine the joint representation space via contrastive tuning. A self-distillation constraint is applied to the image encoder to safeguard intrinsic vision-language knowledge and mitigate catastrophic forgetting.

We evaluated the preservation of pre-trained knowledge within the vision encoder following alignment, using Centered Kernel Alignment (CKA). As illustrated in Fig. 1, ProCLIP demonstrates superior retention of pre-trained features. This suggests that its carefully designed framework effectively facilitates cross-modal alignment without compromising generalization. ProCLIP offers several advantages:1) Simplicity: An intuitive framework using pretrained weights as a semantic bridge. 2) Superiority: SOTA performance under identical data constraints. 3) Versatility: Extends CLIP to long-text, multilingual, and fine-grained scenarios with limited data. 4) Potential: A robust paradigm for modular upgrades in multi-encoder models.

Our main contributions are summarized as follows:

- • We identify a key limitation in current LLM-augmented CLIP models: "fromscratch alignment" neglects valuable pretrained priors, leading to compromised generalization.
- • We propose ProCLIP, a progressive framework that distills CLIP’s priors into an LLMbased embedder, followed by self-distilled contrastive tuning to refine alignment without catastrophic forgetting.
- • We demonstrate ProCLIP’s efficacy through extensive experiments. ProCLIP achieves 6.8%–13.5% gains in zero-shot classification and shows significant improvements across short/long-text, multilingual retrieval, and fine-grained understanding tasks.

##### 2 Related Work

Vision-Language Contrastive Learning. Visionlanguage contrastive learning, exemplified by CLIP (Radford et al., 2021), aims to align mul-

timodal representations in a shared semantic space via large-scale pretraining. As a foundational bridge between modalities, CLIP enables diverse open-vocabulary tasks, including image classification (Zhou et al., 2022b,a; Kim et al., 2024), semantic segmentation (Ding et al., 2022; Li et al., 2022; Ghiasi et al., 2022; Xu et al., 2022; Cho et al.,

- 2024; Lan et al., 2024), and object detection (Du et al., 2022; Kaul et al., 2023). Despite its versatility, CLIP is fundamentally constrained by its text encoder’s fixed input length and limited capacity, which hampers its performance in processing longform, multilingual, and fine-grained semantics. To address these bottlenecks, Long-CLIP (Zhang et al.,

2024) extends input lengths through positional embedding interpolation, while LoTLIP (Wu et al., 2024) aggregates diverse textual information using corner tokens to enhance long-text understanding. However, these methods remain tethered to the original CLIP text encoder’s architecture, preventing the integration of broader open-world knowledge and leaving multilingual support unresolved.

LLM-based Representation Learning. Large Language Models (LLMs) have demonstrated exceptional proficiency across diverse natural language processing tasks (Touvron et al., 2023; Achiam et al., 2023; Bai et al., 2023; Liu et al., 2024a). Recent research has pivoted toward repurposing decoder-only architectures for robust representation learning. Notably, LLM2Vec (BehnamGhader et al., 2024) transforms LLMs into versatile encoders via bidirectional attention and contrastive alignment, while Qwen3-Embedding (Zhang et al., 2025b) leverages massive-scale pretraining to achieve state-of-theart performance on the MTEB benchmark (Muennighoff et al., 2022). Inspired by these advancements, recent works (Huang et al., 2024; Cao et al.,

- 2025; Zhang et al., 2025a) replace CLIP’s original text encoder with LLM-based embedders to handle multilingual, long-form, and complex textual inputs. However, these methods often employ coarse alignment strategies that inadvertently degrade the model’s inherent generalization. Addressing this gap by developing more refined and efficient alignment techniques remains a critical and open challenge. Knowledge Distillation. Knowledge distillation (KD) (Hinton et al., 2015) facilitates knowledge transfer from a teacher to a student model, or within a single model via self-distillation (Zhang et al., 2019). In the vision-language domain, sev-

[Figure 2]

### Representation Inheritance

𝓛𝒊𝒏𝒔

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Adapter LLM embedder

𝓛𝒔𝒕𝒓𝒖𝒄𝒕

CLIP text encoder

[Figure 9]

### Contrastive Tuning

[Figure 10]

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

[Figure 11]

[Figure 12]

###### EMA update

[Figure 13]

[Figure 14]

[Figure 15]

𝓛𝒓𝒆𝒈

[Figure 16]

[Figure 17]

Adapter

[Figure 18]

###### CLIP vision encoder Self-Distillation

𝓛𝒊𝒏𝒇𝒐

LLM embedder

Figure 2: The training pipeline of our proposed ProCLIP. It consists of representation inheritance via crossarchitecture distillation and contrastive tuning integrated with self-distillation regularization.

eral KD techniques have been adapted for CLIP. TinyCLIP (Wu et al., 2023a) employs affinity mimicking to replicate cross-modal interactions, while CLIP-KD (Yang et al., 2024a) integrates featurebased and contrastive distillation to maximize teacher-student similarity. Furthermore, CLIPCID (Yang et al., 2024b) utilizes cluster-instance discrimination to transfer semantic priors. Unlike these methods that primarily focus on model compression or performance enhancement, our work introduces a self-distillation mechanism specifically designed to mitigate catastrophic forgetting and safeguard the model’s intrinsic generalization during the transition to LLM-based architectures.

from large-scale image–text pairs through contrastive learning, bridging both modalities in a shared embedding space. Given a batch of imagetext pairs {(Ii,Ti)}Bi=1, the image encoder EI and text encoder ET map them into the joint semantic space as {(vi,ti)}Bi=1. To optimize both encoders in a dual-tower architecture, a symmetric contrastive learning objective is imposed on the resulting representations:

B

exp(vi · t⊤i /τ)

LCLIP = −

log

+

B j=1 exp(vi · t⊤

j /τ) text-to-image

i=1

exp(ti · vi⊤/τ)

##### 3 Methodology

log

###### .

B j=1 exp(ti · v⊤

j /τ) image-to-text

In this section, we first introduce the preliminary (Sec. 3.1), including contrastive languageimage pre-training and improving CLIP with an LLM-based embedder. Then we present our proposed ProCLIP framework, which comprises two primary training stages: 1) Representation Inheritance via Cross-Architecture Distillation (Sec. 3.2). 2) Contrastive Tuning Integrated with Self-Distillation Regularization (Sec. 3.3).

(1)

However, the native CLIP text encoder is limited to sequences of up to 77 tokens. A common solution is to interpolate the position embeddings of the CLIP text encoder and fine-tune the model. Alternatively, one may replace the CLIP text encoder with an LLM-based embedder. The latter approach not only improves long-text understanding but also enhances multilingual understanding and fine-grained semantic comprehension, resulting in a more versatile vision-language dual-encoder. In this work, we investigate a more efficient alignment strategy that

###### 3.1 Preliminary

Contrastive Language-Image Pre-training. Contrastive Language-Image Pre-training (CLIP) (Radford et al., 2021) learns to align images and text

leverages an LLM-based embedder to enhancing CLIP’s comprehensive capabilities.

Improving CLIP with LLM-based Embedder. LLM2CLIP (Huang et al., 2024) first introduces an LLM-based embedder into CLIP, demonstrating enhanced long-text understanding. Given an LLMbased encoder GT, it encodes texts {Ti}Ni=1 offline into embeddings {t′i}Ni=1. This process is typically performed in an offline manner. During contrastive fine-tuning, a MLP-based adapter is used to map {t′}Ni=1 into the CLIP embedding space for dimensional alignment. The mapped text features and the image features from the CLIP image encoder are then optimized via the contrastive loss in Eq. 1, resulting in a newly aligned representation space. However, applying contrastive learning directly to fine-tuning data to optimize the adapter and vision encoder hinders the convergence of the new dual-tower architecture to an optimal parameter space. This arises because the text representations from the LLM-based embedder and adapter lack prior alignment with the vision encoder. Moreover, unconstrained fine-tuning may also cause excessive drift from the original pre-trained representation, while limited fine-tuning data (e.g., 3M samples) cannot compensate for the knowledge acquired during large-scale pre-training (e.g., 400M samples). To overcome these challenges, we propose a progressive alignment pipeline that improves multimodal alignment while preserving pre-trained knowledge.

- 3.2 Stage I: Representation Inheritance via Cross-Architecture Distillation.

Given a pre-trained image and text encoder of the CLIP model {EI,ET} and a pre-trained LLMbased embedder GT, our goal is to replace the CLIP text encoder ET with the LLM-based embedder GT to enhance comprehensive abilities. Consistent with prior works (Huang et al., 2024; Cao et al., 2025; Zhang et al., 2025a), we initially extract embeddings from textual captions offline using GT: t′ = {GT(Ti) ∈ Rd}Ni=1, where d represents the dimension of the LLM-based embedder.

The embedding space of the LLM-based embedder exhibits no prior alignment with the CLIP image-text representation space. To bridge this gap, we adopt a cross-architecture distillation strategy that transfers knowledge from the CLIP text embedding space to the LLM embedding space. Specifically, given a batch of texts {Ti}Bi=1, we first utilize a single-layer MLP to unify the dimensions

of LLM embeddings and CLIP text embeddings. To facilitate fine-grained semantic alignment, we propose an instance semantic alignment loss, denoted as Lins. Let t∗i = Adapter(t′i) and ei = E(Ti) represent the projected LLM and original CLIP embeddings, respectively. This loss function leverages text-only data to distill knowledge from CLIP’s text encoder into the LLM-based embedder, defined as follows:

B

∥t∗i − ei∥2. (2)

Lins =

i=1

Since Lins only focuses on instance-level alignment without capturing the global embedding structure, we propose the embedding structure alignment loss Lstruct. This loss measures inter-sample distances within a batch in both the CLIP text encoder and LLM-based embedder spaces, and aligns the two globally by minimizing their pairwise distance discrepancy. The structural loss is:

Lstruct =

B

∥d(t∗i,t∗j) − d(ei,ej)∥2, (3)

i<j

where d(·,·) denotes the Euclidean distance. The overall loss is the first stage is defined as:Ldis = Lins + Lstruct.

3.3 Stage II: Contrastive Tuning Integrated with Self-Distillation Regularization.

After the above phase, the Adapter(GT) has already been preliminarily adapted to CLIP’s visionlanguage embedding space, making subsequent fine-tuning with vision-language contrastive learning significantly easier. We utilize the InfoNCE loss (Radford et al., 2021) to better align the image embedding vi and the projected LLM embedding t∗i, which can be formulated as:

B

Linfo = −

i=1

exp(vi · t∗⊤i /τ)

log

+

B j=1 exp(vi · t∗⊤

j /τ)

exp(t∗i · vi⊤/τ)

,

log

B j=1 exp(t∗

i · v⊤

j /τ)

(4)

where τ is a learnable temperature parameter. Beyond standard contrastive learning, we impose a self-distillation constraint on the CLIP image encoder to mitigate excessive forgetting of pre-trained knowledge during adaptation—essential for preserving generalization. On the image encoder side,

Flickr30k COCO ShareGPT4V Urban-1k DOCCI DCI Avg. I2T T2I I2T T2I I2T T2I I2T T2I I2T T2I I2T T2I I2T T2I

Method #Data

Model Architecture: CLIP ViT-B/32 CLIP 400M 80.3 59.8 51.5 30.6 77.3 66.0 60.9 46.8 58.1 53.4 43.1 40.3 61.8 49.5 LLM2CLIP 3M 83.5 70.1 55.6 41.1 94.2 93.4 78.2 84.2 76.2 77.1 62.2 64.4 75.0 71.1 ProCLIP 3M 86.0 73.5 57.8 43.5 94.4 92.6 80.8 85.3 78.1 79.5 65.7 68.3 77.1(+2.1) 73.8(+2.7) LLM2CLIP 15M 86.2 72.2 58.5 43.2 95.3 94.2 80.6 85.3 79.2 80.7 64.3 67.6 77.4 73.9 ProCLIP 15M 86.6 72.6 59.0 43.5 94.5 93.9 82.2 85.3 78.4 80.6 67.1 69.2 78.0(+0.6) 74.2(+0.3) LLM2CLIP 30M 87.8 72.4 61.1 44.3 96.7 95.9 86.6 88.8 82.9 82.9 67.9 69.5 80.5 75.7 ProCLIP 30M 90.2 74.6 62.4 45.9 96.8 95.9 88.5 89.9 82.9 84.1 70.6 71.9 81.9(+1.4) 77.0(+1.3)

Model Architecture: CLIP ViT-B/16 CLIP 400M 82.7 63.4 53.7 33.3 76.1 68.9 67.5 53.5 66.8 57.0 45.4 43.0 65.4 45.6 LLM2CLIP 3M 88.0 75.3 60.5 44.8 94.4 94.4 80.6 86.0 81.7 82.2 67.2 69.1 78.7 75.3 ProCLIP 3M 89.4 77.6 61.7 46.8 94.3 93.3 82.9 88.1 81.0 82.5 67.3 72.0 79.4(+0.7) 76.7(+1.4) LLM2CLIP 15M 88.9 76.6 62.4 46.5 95.0 95.2 84.5 88.4 83.8 85.1 69.3 72.4 80.7 77.3 ProCLIP 15M 90.8 77.9 63.2 47.8 94.2 94.9 85.8 89.6 82.5 84.6 70.2 74.0 81.2(+0.5) 78.0(+0.7) LLM2CLIP 30M 90.2 78.1 65.4 48.5 96.8 96.4 89.7 91.3 86.2 86.8 73.1 74.8 83.6 79.3 ProCLIP 30M 92.7 79.1 67.1 49.7 96.0 96.4 90.0 93.4 85.1 87.3 73.6 76.9 84.2(+0.6) 80.5(+1.2)

Model Architecture: CLIP ViT-L/14 CLIP 400M 86.6 64.6 57.2 36.4 78.0 68.7 68.4 56.0 65.8 63.1 45.4 43.9 66.9 55.5 LLM2CLIP 3M 92.4 80.1 65.5 49.7 95.2 95.6 83.6 89.0 85.1 85.9 70.0 74.4 82.0 79.1 ProCLIP 3M 92.8 81.1 66.4 51.9 95.1 94.8 86.9 92.3 85.9 86.9 71.2 76.1 83.0(+1.0) 80.5(+1.4) LLM2CLIP 15M 91.3 80.6 67.0 50.6 96.3 95.3 86.4 90.5 86.4 88.5 71.7 75.3 83.2 80.1 ProCLIP 15M 93.4 81.4 67.6 52.5 96.1 95.4 88.3 92.6 86.2 88.4 74.4 76.8 84.3(+1.3) 81.2(+1.1) LLM2CLIP 30M 93.1 81.0 68.2 52.0 97.5 97.7 92.7 93.9 88.2 89.6 74.9 78.3 85.8 82.1 ProCLIP 30M 94.5 81.6 69.3 53.2 96.8 97.0 93.0 94.4 87.5 89.8 75.9 79.5 86.2(+0.4) 82.6(+0.5)

Table 1: Cross-modal retrieval performance Recall@1 on multiple datasets.

Image to Text Retrieval

0.8

LLM2CLIP ProCLIP

| |
|---|

0.6

Recall@1

0.4

0.2

0.0

quz mi te sw fil hi bn el cs ar fa fi he tr en sv no hr th pl ko pt nl hu id vi zh es ro uk da ja it ru fr de

Text to Image Retrieval

0.8

LLM2CLIP ProCLIP

| |
|---|

0.6

Recall@1

0.4

0.2

0.0

quz mi te sw fil hi bn el cs ar fa fi he tr en sv no hr th pl ko pt nl hu id vi zh es ro uk da ja it ru fr de

Figure 3: Per-language image-text retrieval performance on the XM3600 benchmark.

we apply a regularization loss that is symmetric to the one used in the first stage(Eq. 2, Eq. 3):

Lreg =

B

B

Lins(vi,vi∗)+

i=1

i<j

Lstruct(vi,vj∗) (5)

where vi∗ = EI∗(Ii),vj∗ = EI∗(Ij) denote the vision embeddings derived from the EMA-updated image

encoder. The EMA update procedure is defined as follows:

EI∗ = αEI∗ + (1 − α)EI, (6) where α controls the update rate of the teacher model parameters. The overall loss function of the contrastive tuning stage is defined as Ltune = Linfo + λregLreg, where λreg is a loss weight.

##### 4 Experiments

4.1 Experimental Setup

Datasets and Benchmarks. For the alignment dataset, we use CC3M (Changpinyo et al., 2021), CC12M (Changpinyo et al., 2021), and YFCC15M (Thomee et al., 2016),combined the high-quality captions from DreamLIP (Zheng et al., 2024). We conduct experiments with data scales of 3M (CC3M), 15M (CC3M + CC12M), and 30M (CC3M + CC12M + YFCC15M) to explore the effects of data scaling. For the benchmark, we perform zero-shot classification on 11 different classification datasets, robustness evaluations on

###### 5 ImageNet variants, retrieval evaluations on 6

Caltech101

CIFAR100

ImageNet

CIFAR10

Food101

SUN397

Aircraft

Flowers

DTD

Cars

Pets

Method #Data

Avg.

Model Architecture: CLIP ViT-B/32 CLIP 400M 83.1 88.7 63.5 61.5 57.6 18.8 42.8 84.6 89.4 66.0 61.9 65.2 LLM2CLIP 3M 49.6 89.2 61.5 60.3 11.5 8.6 47.8 38.0 79.0 22.6 41.0 46.3 ProCLIP 3M 64.5 90.7 65.8 65.0 21.2 11.6 52.0 51.7 83.3 30.8 47.9 53.1(+6.8) LLM2CLIP 15M 57.2 88.3 61.4 61.3 19.6 8.4 50.6 42.3 80.7 23.5 43.3 48.8 ProCLIP 15M 74.9 90.0 66.5 65.1 39.6 13.9 53.7 68.5 86.7 35.5 53.3 58.9(+10.1) LLM2CLIP 30M 58.5 88.3 61.0 61.2 20.6 8.4 50.3 37.6 81.7 26.0 45.1 49.0 ProCLIP 30M 74.4 88.8 66.9 65.9 38.0 16.2 53.0 64.5 86.8 40.4 54.0 59.0(+10.0)

Model Architecture: CLIP ViT-B/16 CLIP 400M 87.9 89.7 66.8 63.1 63.7 22.8 45.0 87.0 90.4 67.6 67.1 68.3 LLM2CLIP 3M 56.9 92.6 64.4 62.2 15.4 11.7 50.9 46.5 82.9 23.6 45.8 50.3 ProCLIP 3M 73.1 92.5 68.9 67.9 32.3 13.5 54.1 59.8 87.0 35.8 54.8 58.2(+7.9) LLM2CLIP 15M 63.2 90.8 64.5 62.9 27.3 9.9 52.8 50.3 83.2 23.7 46.5 52.3 ProCLIP 15M 80.3 90.8 69.7 67.4 44.3 16.5 56.7 75.8 88.4 40.8 58.6 62.7(+10.4) LLM2CLIP 30M 64.4 90.2 64.6 63.7 27.0 11.2 55.0 45.9 84.0 27.1 49.7 53.0 ProCLIP 30M 81.0 89.3 68.3 68.2 48.5 17.9 57.3 70.2 88.8 44.8 59.2 63.0(+10.0)

Model Architecture: CLIP ViT-L/14 CLIP 400M 92.6 94.9 77.0 66.8 76.5 30.7 54.4 93.2 93.9 78.1 74.5 75.7 LLM2CLIP 3M 64.8 95.4 72.9 66.4 18.8 10.4 54.8 47.3 88.3 26.8 52.8 54.4 ProCLIP 3M 83.4 96.6 78.3 72.4 45.1 16.2 59.6 65.9 92.3 41.8 62.5 64.9 (+10.5) LLM2CLIP 15M 70.1 95.2 72.3 66.4 32.4 9.5 58.0 54.3 88.3 26.6 54.0 57.0 ProCLIP 15M 87.1 95.4 77.6 72.3 59.8 21.1 62.1 77.0 92.4 48.8 66.0 69.3 (+12.3) LLM2CLIP 30M 71.2 94.0 70.5 67.0 32.1 11.3 57.8 54.7 89.3 28.8 56.4 57.5 ProCLIP 30M 88.9 94.1 77.7 72.5 61.1 25.2 62.8 81.5 92.9 57.2 67.8 71.0 (+13.5)

Table 2: Zero-shot classification performance on 11 datasets. The best results are marked in bold.

Robustness IN-V2 IN-A IN-O IN-R IN-S Model Architecture: CLIP ViT-L/14

Method #Data

CLIP 400M 69.8 70.8 32.2 87.8 59.6 LLM2CLIP 3M 49.0 46.6 32.4 75.0 44.8 ProCLIP 3M 58.3 63.3 31.6 84.0 52.3 LLM2CLIP 15M 50.8 50.1 33.8 78.2 46.3 ProCLIP 15M 62.1 66.4 34.2 86.4 55.3 LLM2CLIP 30M 52.7 52.7 34.0 78.6 47.3 ProCLIP 30M 63.4 68.0 34.1 86.8 55.7

Table 3: Robustness performance. The best results are marked in bold.

datasets, multilingual cross-modal retrieval evaluation on XM3600 (Thapliyal et al., 2022), and fine-grained understanding evaluation on MMVPVLM (Tong et al., 2024) and SugarCrepe (Hsieh et al., 2023). Regarding the model, we mainly employ three OpenAI pre-trained CLIP models, ViTB/32, ViT-B/16, and ViT-L/14, to investigate the effects of model scaling. Additionally, we conduct experiments with pretrained EVA02-CLIP (Fang et al., 2023) ViT-L/14 and SigLIP2 (Tschannen et al., 2025) SO/14@224 to assess the impacts of different model architectures. For the LLM-based

95

| |89.9 90.0<br><br>92.5<br><br>CLIP 90.7<br><br>LLM2CLIP| | | |
|---|---|---|---|---|
| |ProCLIP| | | |
| |79.4| | | |
| |74.8<br><br>75.5| | | |
| |73.7| | | |
| | | | | |
| |61.0| | | |
| | | | | |
| | | | | |
| | | | | |

90

85

80

AverageScore

75

70

65

60

55

50 add replace swap

Figure 4: Compositional performance on SugarCrepe.

embedder, we primarily use LLaMA3-8B-CC consistent with LLM2CLIP (Huang et al., 2024).

Implementation Details. For the representation inheritance phase, we train for four epochs, followed by another four epochs for contrastive tuning. During training, we employ AdamW (Loshchilov, 2019) as the optimizer, with a learning rate of 1 × 10−5 and a weight decay of 0.2. The parameters β1 and β2 are set to 0.9 and 0.98, respectively. In the first stage, the training batch size is set to 1024, while in the second stage it is increased to

4096. The loss weight λreg is set at 0.0004. Other training details can be found in the appendix.

###### 4.2 Main Results

Cross-Modal Retrieval. Tab. 1 shows that ProCLIP consistently outperforms LLM2CLIP across all datasets and model scales. In short-text scenarios (e.g., Flickr30k and COCO), ProCLIP achieves substantial gains; notably, with ViT-L/14 (30M), it reaches 95.0% I2T R@1 on Flickr30k, outperforming LLM2CLIP by 2%. This advantage extends to long-text benchmarks (DOCCI, DCI, Urban-1k), where ProCLIP (ViT-B/16, 30M) attains 73.6% I2T and 76.9% T2I R@1 on DCI. Across all training scales (3M to 30M), ProCLIP yields stable improvements, particularly in T2I retrieval. These results validate ProCLIP’s versatility in handling both concise and complex textual scenarios.

Multilingual cross-modal Retrieval. Leveraging the LLM-based embedder, ProCLIP exhibits robust multilingual capabilities. As illustrated in Fig. 3, ProCLIP consistently outperforms LLM2CLIP on the XM3600 benchmark (Thapliyal et al., 2022) across various languages. This superiority is primarily attributed to our progressive alignment strategy, which more effectively bridges the semantic gap between the CLIP image encoder and the multilingual LLM space while preserving pretrained visual priors.

Zero-Shot Classification. Tab. 2 summarizes zeroshot classification performance across 11 downstream tasks. We observe that LLM2CLIP severely compromises CLIP’s inherent generalization, with average performance dropping by 16.2% (ViTB/32), 15.3% (ViT-B/16), and 18.2% (ViT-L/14) even when trained on 30M samples. In contrast, ProCLIPconsistently achieves substantial improvements over LLM2CLIP across all configurations, yielding a 10.0%–13.5% average gain on the 30M dataset. This superiority stems from the wellcrafted alignment curriculum of ProCLIP, which leverages pretrained knowledge to achieve more effective and robust cross-modal alignment.

Robustness. Tab. 3 evaluates the robustness of ProCLIP across various data scales and model architectures, showing consistent average gains of 5.9%– 9.3%. Notably, on challenging out-of-distribution (OOD) benchmarks such as ImageNet-A and ImageNet-R, ProCLIP outperforms LLM2CLIP by over 10 percentage points. This substantial margin highlights ProCLIP’s superior capacity to handle distribution shifts and complex perturbations.

Fine-Grained Understanding. Fig. 4 demonstrates the significant potential of ProCLIP in compositional understanding tasks, surpassing the CLIP baseline by an average margin of 11.3%– 17.7%. These improvements demonstrate that the LLM-based embedder enhances fine-grained semantic discrimination, and the consistent superiority of our method underscores the effectiveness of the progressive alignment strategy. Fig. 5 presents the fine-grained vision-language understanding performance on the MMVP benchmark (Tong et al., 2024) using CLIP ViT-L/14. LLM2CLIP improves over CLIP by 2.9%, 5.9%, and 4.4% at 3M, 15M, and 30M data scales, respectively. Our ProCLIP model further advances these results, achieving gains of 3.0%, 2.2%, and 10.4% on the corresponding data scales.

Comparison with Other Methods. To further prove the effectiveness of ProCLIP, we provide a comprehensive comparison of all recent LLM embedder-based CLIP models, including FLAME, ShareLock, LIFT, SAIL, LiT, and our baseline LLM2CLIP. As shown in Tab. 4, under the same or lower training costs, ProCLIP consistently achieves superior performance across various model sizes. Benefiting from representation inheritance and self-distillation regularization, ProCLIP not only achieves significant performance improvements in In-1k classification but also enhances general retrieval capabilities on COCO and Flickr30k.

###### 4.3 Ablation Study

Ablation of Different LLM-based Embedder. Fig. 6 compares several LLM embedders (Qwen3, GME, NV-Embedv2, and Llama3-CC) paired with ViT-L/14. Among them, Llama3-CC achieves the best overall results in classification and retrieval. Notably, while retrieval performance remains relatively consistent across embedders, ImageNet classification accuracy varies significantly. This suggests that the alignment discrepancy between each LLM’s feature space and the original CLIP space differs, leading to varying degrees of generalization degradation after training.

Ablation of Different Base Model. Tab. 6 shows ProCLIP excels across architectures, surpassing LLM2CLIP in classification and retrieval. The remaining classification gap relative to vanilla CLIP arises from the latter’s overwhelming data scale, a dominant factor in zero-shot tasks.

Ablation of λreg. Tab. 7 presents the sensitivity analysis of the regularization weight for λreg. The

[Figure 19]

COCO Flickr30k I2T T2I I2T T2I

Method ViT Init LLM Embedder #Data IN-1k

FLAME random Mistral-Nemo 3M 36.0 43.3 28.6 67.3 53.6 ShareLock DINOv2 B/14 Llama3 3M 52.1 - - - LIFT random NV-Embedv2 512M 43.6 34.6 36.0 69.1 72.9 LiT CLIP B/16 Llama3-CC 3M 51.0 56.2 41.9 85.2 71.9 LLM2CLIP CLIP B/16 Llama3-CC 3M 45.8 60.5 44.8 88.0 75.3 ProCLIP CLIP B/16 Llama3-CC 3M 54.8 61.7 46.8 89.4 77.6

𝚫

𝚫

𝚫

𝚫

𝚫

SAIL DINOv2 L/14 NV-Embedv2 3M 54.0 45.4 32.9 - LiT CLIP L/14 Llama3-CC 3M 60.1 59.4 44.6 88.0 74.7 LLM2CLIP CLIP L/14 Llama3-CC 3M 52.8 65.5 49.7 92.4 80.1 ProCLIP CLIP L/14 NV-Embedv2 3M 61.4 64.8 51.7 91.9 81.4 ProCLIP CLIP L/14 Llama3-CC 3M 62.5 66.4 51.9 92.8 81.1

𝚫

Table 4: Comparison with other methods across different model scales and LLM embedders.

Figure 5: MMVP performance.

Stage 1 Stage 2

85

Qwen3-Embedding GME

Method

IN-1k I2T Avg. T2I Avg. Lins Lstruct Linfo Lreg

| |
|---|

80

NV-Embedv2

Llama-CC

CLIP 74.5 66.9 55.5 LLM2CLIP 52.8 82.0 79.1

75

Performance

70

65

✓ 58.9 69.3 59.4 ✓ ✓ 59.5 70.3 61.2 ✓ ✓ ✓ 59.2 82.9 80.2 ✓ ✓ ✓ ✓ 62.5 83.0 80.5

60

ProCLIP

55

50

I2T Average T2I Average ImageNet

Table 5: Ablation on different components.

Figure 6: Ablation on different LLM-based embedders.

Base Model Method IN-1k I2T Avg. T2I Avg. CLIP ViT-L

λreg IN-1k I2T Avg. T2I Avg. 0 59.2 82.9 80.2

LLM2CLIP 52.8 82.0 79.1 ProCLIP 62.5 83.0 80.5

2e-4 60.3 82.8 80.3 4e-4 62.5 83.0 80.5 6e-4 62.6 82.6 79.8 1e-3 62.5 81.8 79.5

LLM2CLIP 56.4 83.6 80.9

EVA02-CLIP ViT-L

- ProCLIP 66.8 84.2 82.2

SigLIP2 SO/14@224

LLM2CLIP 63.8 84.6 81.9

- ProCLIP 67.7 86.3 83.9

Table 7: Ablation of Lreg

Table 6: Ablation study on different base model architectures. All methods use 3M training samples.

accuracy from 59.2% to 62.5%. Ultimately, structured self-distillation yields the best results by stabilizing the image representation space and preserving pretrained knowledge, effectively balancing fine-tuning gains with generalization.

results indicate that 4e-4 offers an optimal balance and is thus selected as our final configuration. Furthermore, the empirical evidence suggests that excessively strong regularization hampers alignment, leading to suboptimal results.

##### 5 Conclusion

Ablation of Different Components. Tab. 5 details the ablation study of our components. First, instance semantic distillation achieves 58.9% ImageNet-1k zero-shot accuracy using text only, confirming the successful transfer of CLIP’s textual priors to the adapter. Adding structural alignment loss further boosts both classification and retrieval by enabling the LLM to capture CLIP’s global geometric structure beyond point-wise semantics. Subsequent image-text contrastive tuning significantly enhances retrieval but compromises classification due to image encoder overfitting. This is mitigated by self-distillation, which recovers classification

This paper presents ProCLIP, a progressive visionlanguage alignment framework for integrating CLIP image encoders with LLM-based embedders. Inspired by curriculum learning, ProCLIP employs a two-stage strategy: first, it anchors the LLM-based embedder to CLIP’s textual space via knowledge distillation to inherit pretrained semantic priors; second, it performs cross-modal contrastive tuning while utilizing self-distillation to prevent overfitting. To maintain feature-space consistency, we introduce a complementary distillation scheme incorporating instance semantic and structural alignment losses. Extensive evaluations

across various scales and architectures validate the efficacy and generality of our approach.

##### 6 Limitation

Training Efficiency. While the first stage of our progressive alignment framework incurs only modest additional computational overhead, the second stage requires unfreezing the vision encoder and performing online self-distillation, which significantly increases training cost. Compared to the baseline, our ProCLIP incurs approximately 35% additional computational overhead. However, our training remains within a manageable computational budget—for instance, training a CLIP ViT-L model with ProCLIP on 3M data only 1.5 hours on 8× H100 GPUs.

##### References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, and 1 others. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, and 1 others. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.

Parishad BehnamGhader, Vaibhav Adlakha, Marius Mosbach, Dzmitry Bahdanau, Nicolas Chapados, and Siva Reddy. 2024. Llm2vec: Large language models are secretly powerful text encoders. arXiv preprint arXiv:2404.05961.

Anjia Cao, Xing Wei, and Zhiheng Ma. 2025. Flame: Frozen large language models enable data-efficient language-image pre-training. In CVPR, pages 4080– 4090.

Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. 2021. Conceptual 12m: Pushing webscale image-text pre-training to recognize long-tail visual concepts. In CVPR, pages 3558–3568.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. 2024. Sharegpt4v: Improving large multi-modal models with better captions. In ECCV, pages 370– 387. Springer.

Seokju Cho, Heeseong Shin, Sunghwan Hong, Anurag Arnab, Paul Hongsuck Seo, and Seungryong Kim. 2024. Cat-seg: Cost aggregation for open-vocabulary semantic segmentation. In CVPR, pages 4113–4123.

Zheng Ding, Jieke Wang, and Zhuowen Tu. 2022. Open-vocabulary universal image segmentation with maskclip. arXiv preprint arXiv:2208.08984.

Yu Du, Fangyun Wei, Zihe Zhang, Miaojing Shi, Yue Gao, and Guoqi Li. 2022. Learning to prompt for open-vocabulary object detection with visionlanguage model. In CVPR, pages 14084–14093.

Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. 2023. Eva: Exploring the limits of masked visual representation learning at scale. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19358– 19369.

Golnaz Ghiasi, Xiuye Gu, Yin Cui, and Tsung-Yi Lin. 2022. Scaling open-vocabulary image segmentation with image-level labels. In ECCV, pages 540–557. Springer.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. 2017. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913.

Tiancheng Gu, Kaicheng Yang, Xiang An, Ziyong Feng, Dongnan Liu, Weidong Cai, and Jiankang Deng. 2024. Rwkv-clip: a robust vision-language representation learner. arXiv preprint arXiv:2406.06973.

Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. 2018. Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3608–3617.

Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, and 1 others. 2021a. The many faces of robustness: A critical analysis of out-of-distribution generalization. In ICCV, pages 8340–8349.

Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. 2021b. Natural adversarial examples. In CVPR, pages 15262–15271.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.

Cheng-Yu Hsieh, Jieyu Zhang, Zixian Ma, Aniruddha Kembhavi, and Ranjay Krishna. 2023. Sugarcrepe: Fixing hackable benchmarks for vision-language compositionality. Advances in neural information processing systems, 36:31096–31116.

Xiaoxing Hu, Kaicheng Yang, Jun Wang, Haoran Xu, Ziyong Feng, and Yupei Wang. 2025. Decoupled global-local alignment for improving compositional understanding. arXiv preprint arXiv:2504.16801.

Weiquan Huang, Aoqi Wu, Yifan Yang, Xufang Luo, Yuqing Yang, Liang Hu, Qi Dai, Chunyu

Wang, Xiyang Dai, Dongdong Chen, and 1 others. 2024. Llm2clip: Powerful language model unlocks richer visual representation. arXiv preprint arXiv:2411.04997.

Drew A Hudson and Christopher D Manning. 2019. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709.

Prannay Kaul, Weidi Xie, and Andrew Zisserman. 2023. Multi-modal classifiers for open-vocabulary object detection. In ICML, pages 15946–15969. PMLR.

Gahyeon Kim, Sohee Kim, and Seokju Lee. 2024. Aapl: Adding attributes to prompt learning for visionlanguage models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pages 1572–1582.

Mengcheng Lan, Chaofeng Chen, Yiping Ke, Xinjiang Wang, Litong Feng, and Wayne Zhang. 2024. Clearclip: Decomposing clip representations for dense vision-language inference. In ECCV, pages 143–160. Springer.

Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. 2024. Nv-embed: Improved techniques for training llms as generalist embedding models. arXiv preprint arXiv:2405.17428.

Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. 2023a. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125.

Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen Koltun, and René Ranftl. 2022. Languagedriven semantic segmentation. arXiv preprint arXiv:2201.03546.

Xianhang Li, Yanqing Liu, Haoqin Tu, Hongru Zhu, and Cihang Xie. 2025. Openvision: A fully-open, costeffective family of advanced vision encoders for multimodal learning. arXiv preprint arXiv:2505.04601.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. 2023b. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In ECCV, pages 740– 755. Springer.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, and 1 others. 2024a. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437.

Yanqing Liu, Xianhang Li, Letian Zhang, Zirui Wang, Zeyu Zheng, Yuyin Zhou, and Cihang Xie. 2025. Openvision 2: A family of generative pretrained visual encoders for multimodal learning. arXiv preprint arXiv:2509.01644.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, and 1 others. 2024b. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer.

I Loshchilov. 2019. Decoupled weight decay regularization. In ICLR.

Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers. 2022. Mteb: Massive text embedding benchmark. arXiv preprint arXiv:2210.07316.

Yasumasa Onoe, Sunayana Rane, Zachary Berger, Yonatan Bitton, Jaemin Cho, Roopal Garg, Alexander Ku, Zarana Parekh, Jordi Pont-Tuset, Garrett Tanzer, and 1 others. 2024. Docci: Descriptions of connected and contrasting images. In ECCV, pages 291–309. Springer.

Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. 2015. Flickr30k entities: Collecting region-to-phrase correspondences for richer imageto-sentence models. In ICCV, pages 2641–2649.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, and 1 others. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR.

Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. 2019. Do imagenet classifiers generalize to imagenet? In ICML, pages 5389–5400. PMLR.

Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. 2019. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326.

Ashish V Thapliyal, Jordi Pont-Tuset, Xi Chen, and Radu Soricut. 2022. Crossmodal-3600: A massively multilingual multimodal evaluation dataset. arXiv preprint arXiv:2205.12522.

Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. 2016. Yfcc100m: The new data in multimedia research. Communications of the ACM, 59(2):64–73.

Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. 2024. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In CVPR, pages 9568–9578.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, and 1 others. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, and 1 others. 2025. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786.

Jack Urbanek, Florian Bordes, Pietro Astolfi, Mary Williamson, Vasu Sharma, and Adriana RomeroSoriano. 2024. A picture is worth more than 77 text tokens: Evaluating clip-style models on dense captions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26700–26709.

Haohan Wang, Songwei Ge, Zachary Lipton, and Eric P Xing. 2019. Learning robust global representations by penalizing local predictive power. NIPS, 32.

Zihao Wang, Wei Liu, Qian He, Xinglong Wu, and Zili Yi. 2022. Clip-gen: Language-free training of a text-to-image generator with clip. arXiv preprint arXiv:2203.00386.

Kan Wu, Houwen Peng, Zhenghong Zhou, Bin Xiao, Mengchen Liu, Lu Yuan, Hong Xuan, Michael Valenzuela, Xi Stephen Chen, Xinggang Wang, and 1 others. 2023a. Tinyclip: Clip distillation via affinity mimicking and weight inheritance. In ICCV, pages 21970–21980.

Wei Wu, Kecheng Zheng, Shuailei Ma, Fan Lu, Yuxin Guo, Yifei Zhang, Wei Chen, Qingpei Guo, Yujun Shen, and Zheng-Jun Zha. 2024. Lotlip: Improving language-image pre-training for long text understanding. NIPS, 37:64996–65019.

Xiaoshi Wu, Feng Zhu, Rui Zhao, and Hongsheng Li. 2023b. Cora: Adapting clip for open-vocabulary detection with region prompting and anchor prematching. In CVPR, pages 7031–7040.

Mengde Xu, Zheng Zhang, Fangyun Wei, Yutong Lin, Yue Cao, Han Hu, and Xiang Bai. 2022. A simple baseline for open-vocabulary semantic segmentation with pre-trained vision-language model. In ECCV, pages 736–753. Springer.

Chuanguang Yang, Zhulin An, Libo Huang, Junyu Bi, Xinqiang Yu, Han Yang, Boyu Diao, and Yongjun Xu. 2024a. Clip-kd: An empirical study of clip model distillation. In CVPR, pages 15952–15962.

Kaicheng Yang, Jiankang Deng, Xiang An, Jiawei Li, Ziyong Feng, Jia Guo, Jing Yang, and Tongliang Liu. 2023. Alip: Adaptive language-image pre-training with synthetic caption. In ICCV, pages 2922–2931.

Kaicheng Yang, Tiancheng Gu, Xiang An, Haiqiang Jiang, Xiangzi Dai, Ziyong Feng, Weidong Cai, and Jiankang Deng. 2024b. Clip-cid: Efficient clip distillation via cluster-instance discrimination. AAAI.

Beichen Zhang, Pan Zhang, Xiaoyi Dong, Yuhang Zang, and Jiaqi Wang. 2024. Long-clip: Unlocking the long-text capability of clip. In ECCV, pages 310– 325. Springer.

Le Zhang, Qian Yang, and Aishwarya Agrawal. 2025a. Assessing and learning alignment of unimodal vision and language models. In CVPR, pages 14604–14614.

Linfeng Zhang, Jiebo Song, Anni Gao, Jingwei Chen, Chenglong Bao, and Kaisheng Ma. 2019. Be your own teacher: Improve the performance of convolutional neural networks via self distillation. In CVPR, pages 3713–3722.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, and 1 others. 2025b. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176.

Kecheng Zheng, Yifei Zhang, Wei Wu, Fan Lu, Shuailei Ma, Xin Jin, Wei Chen, and Yujun Shen. 2024. Dreamlip: Language-image pre-training with long captions. In European Conference on Computer Vision, pages 73–90. Springer.

Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. 2022a. Conditional prompt learning for vision-language models. In CVPR, pages 16816– 16825.

Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. 2022b. Learning to prompt for visionlanguage models. IJCV, 130(9):2337–2348.

##### A Appendix

- A.1 Training Details

Details of the hyperparameter configurations used for two-stage training of ProCLIP are presented in Tab. 8. Under the default setting, our MLP layers are consistent with the baseline LLM2CLIP, both consisting of four linear layers.

Hyperparameters of stage1

Batch size 1024 (8 × 128) Optimizer AdamW

Weight decay 0.05 Adam β (0.9,0.98) Adam ϵ 1e-6

Learning rate 1e-5 Learning rate schedule cosine decay

Epochs 4 Training GPUs 8×H100

Hyperparameters of stage2

Batch size 4096 (8 × 512) Optimizer AdamW

Weight decay 0.05 Adam β (0.9,0.98) Adam ϵ 1e-6

Learning rate 1e-5 Learning rate schedule cosine decay

Ema α 0.999

λreg 0.0004 Epochs 4 Training GPUs 8×H100

- Table 8: Detailed hyperparameters for training ProCLIP.

A.2 Details of Benchmarks.

Zero-Shot Classification & Linear Probe. Following the previous works (Yang et al., 2023; Gu et al., 2024), we evaluate the zero-shot classification and linear probe performance of the models on 11 datasets. The detailed information about these datasets and the prompt used in zero-shot classification are presented in Tab. 9 and Tab. 16.

Dataset Classes Train size Test size Evaluation metric Food101 102 75,750 25,250 accuracy CIFAR10 10 50,000 10,000 accuracy CIFAR100 100 50,000 10,000 accuracy SUN397 397 19,850 19,850 accuracy Cars 196 8,144 8,041 accuracy Aircraft 100 6,667 3,333 mean per class DTD 47 3,760 1,880 accuracy Pets 37 3,680 3,669 mean per class Caltech101 101 3,000 5,677 mean-per-class Flowers 102 2,040 6,149 mean per class ImageNet 1000 1,281,167 50,000 accuracy

- Table 9: List of zero-shot datasets with the data distribution and evaluation metrics. Robustness. We evaluated the robustness of our model on five out-of-distribution datasets, including ImageNet-v2 (Recht et al., 2019), ImageNet-A (Hendrycks et al., 2021b), ImageNet-O (Hendrycks et al., 2021b), ImageNet-R (Hendrycks et al., 2021a), and ImageNet-Sketch (Wang et al., 2019). Cross-Modal Retrieval. Following the previous works (Huang et al., 2024; Cao et al., 2025), we evaluate the cross-modal retrieval performance of the models on 6 datasets: Flickr30k (Plummer et al., 2015), COCO (Lin et al., 2014), ShareGPT4V (Chen et al., 2024), Urban-1k (Zhang et al., 2024), DOCCI (Onoe et al., 2024), and DCI (Urbanek et al., 2024). The details information about these dataset are present on Tab. 10.

Dataset Test Images Evaluation Protocol Text type

MSCOCO 5,000 Image-to-Text & Text-to-Image short Flickr30k 1,000 Image-to-Text & Text-to-Image short ShareGPT4V 1000 Image-to-Text & Text-to-Image long Urban-1k 1000 Image-to-Text & Text-to-Image long DOCCI 5000 Image-to-Text & Text-to-Image long DCI 7805 Image-to-Text & Text-to-Image long

Table 10: Zero-shot image-text retrieval evaluation settings.

Multilingual Retrieval. We evaluated the multilingual capabilities of our model on XM3600 (Thapliyal et al., 2022). XM3600 contains 3,600 images covering a total of 36 languages, including Arabic (ar), Bengali(bn), Chinese-Simplified (zh), Croatian (hr), Czech (cs), Danish (da), Dutch (nl), English (en),Farsi (fa), Filipino (fil), Finnish (fi), French (fr), German (de), Greek (el), Hebrew (he), Hindi (hi), Hungarian (hu), Indonesian (id), Italian (it), Japanese (ja), Korean (ko),Maori(mi), Norwegian (no), Persian (fa), Polish (pl), Portuguese (pt), Romanian (ro), Russian (ru), Spanish (es), Swedish (sv), Swahili(sw), Thai (th), Turkish (tr), Telugu (te), Ukrainian (uk), and Vietnamese (vi).

Fine-Grained Understanding. We evaluated the fine-grained understanding capability of the VLM on MMVP-VLM (Tong et al., 2024) and SugarCrepe. MMVP-VLM consists of 150 samples in total, testing 9 patterns:

- • ☼ Orientation and Direction: Questions about the direction something is facing or moving, such as the direction the dog or duck is facing, or the orientation of the school bus.
- • Presence of Specific Features: Questions that focus on the existence or non-existence of certain elements or features in the image.
- • State and Condition: Questions that pertain to the state or condition of an object, such as whether a flag is blowing in the wind or if the ground is wet.
- • Quantity and Count: Questions about the number of objects or features present in the image.
- • Positional and Relational Context: This aspect refers to the model’s ability to understand the position and relationship of objects or elements within an image in relation to each other and their surroundings.
- • Color and Appearance: Questions regarding the color of certain objects or elements.

###### REPLACE SWAP ADD

###### Object Attribute Relation Object Attribute Object Attribute # negative captions 1652 788 1406 246 666 2062 692

Table 11: Number of hard negative captions of all types in SugarCrepe.

Flickr30k COCO ShareGPT4V Urban-1k DOCCI DCI Avg. I2T T2I I2T T2I I2T T2I I2T T2I I2T T2I I2T T2I I2T T2I

Method

CLIP 86.6 64.6 57.2 36.4 78.0 68.7 68.4 56.0 65.8 63.1 45.4 43.9 66.9 55.5 Long-CLIP 89.7 75.7 62.5 46.4 93.3 92.1 82.4 86.2 66.4 78.6 57.2 64.0 75.2 73.8 ProCLIP 92.1 81.0 65.8 51.1 94.7 93.6 85.0 91.0 84.8 86.6 68.7 74.7 81.9 79.7

Table 12: Compared to Long-CLIP on multiple cross-modal retrieval datasets.

- • Structural and Physical Characteristics: This category involves the model’s ability to identify and analyze the physical attributes and structural features of objects in an image.
- • Text: Questions related to text or symbols present in the image.
- • Viewpoint and Perspective: Questions concerning the perspective from which the photo was taken.

SugarCrepe contains three types of negative samples—Add, Replace, and Swap—designed to probe whether vision-language models (VLMs) can recognize text that differs only in subtle structural changes yet conveys a completely different overall meaning in image-text matching tasks.

MLLM benchmarks. We further integrate the fine-tuned vision encoder into LLaVA and evaluate its performance on several MLLM downstream benchmarks, including SEED-Bench (Li

- et al., 2023a), GQA (Hudson and Manning, 2019), VizWiz (Gurari et al., 2018), PoPE (Li et al., 2023b), TextVQA (Singh et al., 2019), MMBench (Liu et al., 2024b), and VQAv2 (Goyal et al., 2017).

A.3 More results. Compared to Long-CLIP. Long-CLIP (Zhang

- et al., 2024) is a classic method for extending the long-text capabilities of CLIP. Here, we compare ProCLIP with Long-CLIP. We restricted the data volume for ProCLIP to 1M, consistent with the Long-CLIP experiments. The results are shown in Table 12. Our method outperforms the Long-CLIP paradigm in both short-text and long-text retrieval.

Furthermore, we extend CLIP’s capabilities to be both multilingual and fine-grained understanding.

Linear Probe. We conduct linear probe evaluations of the model on 11 datasets. As shown in Tab.9, our method consistently achieves superior performance. This advantage stems from our progressive alignment framework, which stabilizes training through two-stage regularization that prevents overfitting in the vision encoder while preserving generalization capability.

Further Analysis of Data Scale and Model Scale. We further analysis the effects of data scale and model scale, as shown in Tab. 14. For data scale, we observe that model performance improves with increasing data size. For example, when trained on 3M samples, ProCLIP achieves a zero-shot IN-1k accuracy of 62.5, which rises to 67.8 when the dataset size increases to 30M. Under the same data scale, ProCLIP consistently outperforms LLM2CLIP. Notably, when we randomly sample 1M images from CC3M for training, ProCLIP achieves comparable or even superior zero-shot retrieval performance compared with LLM2CLIP, reaching 61.8 on zero-shot IN-1k. This highlights the data efficiency of ProCLIP. For model scale, we further expand the linear layers by three times, using 12 layers in total, which leads to additional performance gains. This suggests that ProCLIP can continue to benefit from simple parameter scaling.

MLLM Performance. As shown in Tab. 13, when integrating the fine-tuned vision encoder into the MLLM, we observe performance improvements over CLIP on most benchmarks. This can be mainly attributed to the alignment with high-quality data, which enhances the semantic representation capability of the vision encoder. ProCLIP and

Method SEED-Bench (image) GQA VizWiz PoPE TextVQA MMBench VQAv2 CLIP 65.3 62.0 44.0 85.7 54.2 65.5 77.4

LLM2CLIP 66.4 61.7 44.6 86.3 55.0 65.5 77.9 ProCLIP 66.4 62.3 44.7 85.9 54.5 65.0 78.1

Table 13: MLLM(7B) performance under 2242 image resolution.

Method Data Adapter IN-1k I2T Avg. T2I Avg. CLIP 400M - 74.5 66.9 55.5

ProCLIP 1M 4×linear 61.8 81.9 79.7 LLM2CLIP 3M 4×linear 52.8 82.0 79.1 ProCLIP 3M 4×linear 62.5 83.0 80.5 LLM2CLIP 15M 4×layers 54.0 83.2 80.1 ProCLIP 15M 4×layers 66.0 84.3 81.2

LLM2CLIP 30M 4×linear 56.4 85.8 82.1 ProCLIP 30M 4×linear 67.8 86.2 82.6 ProCLIP 30M 12×linear 71.5 86.8 82.8

Table 14: Comparison of data and model scales under ViT-L architecture.

LLM2CLIP achieve relatively comparable performance, indicating that ProCLIP does not exhibit a significant advantage within the MLLM benchmarks. We attribute this to the fact that our method, compared with the baseline, does not place additional emphasis on the downstream MLLM benchmarks. A further discussion on this issue can be found in Openvision (Li et al., 2025) and OpenVision2 (Liu et al., 2025).

###### A.4 Future Works.

Training Efficiency. ProCLIP is subject to the limitation of increased computational overhead. We consider the following directions to potentially reduce computational overhead:

- • Adopting a PEFT-based approach to fine-tune the vision encoder in the second stage
- • Fine-tuning only part of the vision encoder parameters in the second stage, such as the last few Transformer blocks
- • Replacing online distillation with offline distillation, which would substantially reduce the additional computational cost introduced in the second stage

Fine-grained Visual Alignment. Future work explores the integration of a dedicated local alignment loss to explicitly strengthen the model’s fine-grained visual-textual alignment capabilities, thereby enhancing performance on tasks that require detailed spatial understanding.

More Model Architecture. Our approach replaces the original CLIP text encoder with an LLM-based embedder to enhance multiple capabilities. From another perspective, we consider whether the vision encoder in the dual-tower architecture can also be replaced to address limitations in visual representation. For instance, the CLIP image encoder is known to lack locality; substituting the image encoder may mitigate this limitation. We leave a thorough investigation of this direction for future work.

###### A.5 Visualization.

As illustrated in Fig. 7, ProCLIP effectively extends long-text processing capabilities, enabling precise image-text alignment even within highly detailed descriptions.

Caltech101

CIFAR100

ImageNet

CIFAR10

Food101

SUN397

Aircraft

Flowers

DTD

Cars

Pets

Avg. Model Architecture: CLIP ViT-B/32

Method Data

CLIP 400M 88.6 95.1 80.1 73.4 80.8 44.9 76.3 89.3 92.7 94.7 74.3 80.9 LLM2CLIP 3M 87.9 95.7 83.1 74.1 78.0 44.9 77.7 90.4 92.4 94.6 74.2 81.2 ProCLIP 3M 88.4 95.9 83.1 74.3 79.5 44.1 78.2 90.3 92.6 95.0 74.4 81.4 LLM2CLIP 15M 87.7 95.7 82.7 74.0 77.5 44.2 78.3 90.2 92.5 94.4 74.2 81.0 ProCLIP 15M 88.7 95.9 82.8 74.8 80.8 44.9 78.1 90.2 92.8 95.1 74.4 81.7 LLM2CLIP 30M 87.6 95.9 83.0 74.1 76.3 43.5 77.6 90.1 92.8 93.8 74.3 80.8 ProCLIP 30M 88.2 96.0 83.1 75.1 79.0 43.8 77.8 89.8 92.6 94.9 74.5 81.4

Model Architecture: CLIP ViT-B/16

CLIP 400M 92.7 96.0 82.5 75.7 85.9 52.8 78.9 93.1 93.9 96.4 79.6 84.4 LLM2CLIP 3M 91.6 97.0 84.5 76.0 82.1 50.1 80.3 92.3 93.6 95.7 79.6 83.9 ProCLIP 3M 92.8 96.8 84.6 76.4 85.6 52.0 80.6 93.3 94.2 97.0 79.7 84.8 LLM2CLIP 15M 91.9 97.0 84.9 75.6 83.7 50.7 80.4 92.9 93.8 96.6 79.6 84.3 ProCLIP 15M 92.6 96.7 84.3 76.6 85.6 51.4 80.8 93.6 94.3 96.7 79.8 84.8 LLM2CLIP 30M 91.3 96.6 84.8 75.3 80.6 48.2 80.3 92.5 93.4 95.0 79.7 83.4 ProCLIP 30M 92.3 96.6 85.7 77.0 84.7 50.1 81.2 93.1 94.0 96.7 79.5 84.6

Model Architecture: CLIP ViT-L/14

CLIP 400M 95.3 89.1 87.2 79.4 90.7 63.0 81.8 95.3 96.9 98.8 82.9 88.1 LLM2CLIP 3M 94.5 98.6 89.2 79.6 86.7 57.7 83.4 94.1 96.4 97.1 82.5 87.2

- ProCLIP 3M 95.3 98.5 88.8 80.3 90.3 61.0 83.6 95.2 96.9 98.7 81.9 88.2 LLM2CLIP 15M 94.4 98.5 88.8 78.5 86.0 55.0 82.7 93.9 95.9 97.1 82.6 86.7 ProCLIP 15M 95.2 98.4 88.6 79.7 90.5 61.4 83.3 95.3 96.8 98.7 83.0 86.7 LLM2CLIP 30M 94.1 98.2 88.4 78.7 84.6 54.8 82.4 93.7 95.8 96.5 82.2 86.3 ProCLIP 30M 95.1 98.4 89.0 80.3 90.0 60.0 83.9 95.2 96.8 98.5 82.7 88.2

Model Architecture: EVA02-CLIP ViT-L/14

EVA02-CLIP 2B 95.6 99.5 94.2 80.4 94.2 69.5 85.0 94.8 97.6 99.4 84.1 90.4 LLM2CLIP 3M 94.1 99.5 93.3 79.4 85.0 54.3 84.0 93.2 97.3 96.9 84.1 87.4

- ProCLIP 3M 95.3 99.5 94.0 81.0 93.9 65.7 85.9 95.4 97.8 99.3 84.5 90.2 Table 15: Linear Probe performance on 11 datasets.

CIFAR 10 & CIFAR 100 a photo of a {label}. a blurry photo of a {label}. a black and white photo of a {label}. a low contrast photo of a {label}. a high contrast photo of a {label}. a bad photo of a {label}. a good photo of a {label}. a photo of a small {label}. a photo of a big {label}. a photo of the {label}. a blurry photo of the {label}. a black and white photo of the {label}. a low contrast photo of the {label}. a high contrast photo of the {label}. a bad photo of the {label}. a good photo of the {label}. a photo of the small {label}. a photo of the big {label}.

Food101 a photo of {label}, a type of food.

Caltech101 a photo of a {label}. a painting of a {label}. a plastic {label}. a sculpture of a {label}. a sketch of a {label}. a tattoo of a {label}. a toy {label}. a rendition of a {label}. a embroidered {label}. a cartoon {label}. a {label} in a video game. a plushie {label}. an origami {label}. art of a {label}. graffiti of a {label}. a drawing of a {label}. a doodle of a {label}. a photo of the {label}. a painting of the {label}. the plastic {label}. a sculpture of the {label}. a sketch of the {label}. a tattoo of the {label}. the toy {label}. a rendition of the {label}. the embroidered {label}. the cartoon {label}. the {label} in a video game. the plushie {label}. the origami {label}. art of the {label}. graffiti of the {label}. a drawing of the {label}. a doodle of the {label}.

Stanford Cars a photo of a {label}. a photo of the {label}. a photo of my {label}. i love my {label}! a photo of my dirty {label}. a photo of my clean {label}. a photo of my new {label}. a photo of my old {label}.

DTD a photo of a {label} texture. a photo of a {label} pattern. a photo of a {label} thing. a photo of a {label} object. a photo of the {label} texture. a photo of the {label} pattern. a photo of the {label} thing. a photo of the {label} object.

FGVC Aircraft a photo of a {label}, a type of aircraft. a photo of the {label}, a type of aircraft.

Flowers102 a photo of a {label}, a type of flower.

Pets a photo of a {label}, a type of pet.

SUN39 a photo of a {label}. a photo of the {label}.

ImageNet a bad photo of a {label}. a photo of many {label}. a sculpture of a {label}. a photo of the hard to see {label}. a low resolution photo of the {label}. a rendering of a {label}. graffiti of a {label}. a bad photo of the {label}. a cropped photo of the {label}. a tattoo of a {label}. the embroidered {label}. a photo of a hard to see {label}. a bright photo of a {label}. a photo of a clean {label}. a photo of a dirty {label}. a dark photo of the {label}. a drawing of a {label}. a photo of my {label}. the plastic {label}. a photo of the cool {label}. a close-up photo of a {label}. a black and white photo of the {label}. a painting of the {label}. a painting of a {label}. a pixelated photo of the {label}. a sculpture of the {label}. a bright photo of the {label}. a cropped photo of a {label}. a plastic {label}. a photo of the dirty {label}. a jpeg corrupted photo of a {label}. a blurry photo of the {label}. a photo of the {label}. a good photo of the {label}. a rendering of the {label}. a {label} in a video game. a photo of one {label}. a doodle of a {label}. a close-up photo of the {label}. a photo of a {label}. the origami {label}. the {label} in a video game. a sketch of a {label}. a doodle of the {label}. an origami {label}. a low resolution photo of a {label}. the toy {label}. a rendition of the {label}. a photo of the clean {label}. a photo of a large {label}. a rendition of a {label}. a photo of a nice {label}. a photo of a weird {label}. a blurry photo of a {label}. a cartoon {label}. art of a {label}. a sketch of the {label}. a embroidered {label}. a pixelated photo of a {label}. itap of the {label}. a jpeg corrupted photo of the {label}. a good photo of a {label}. a plushie {label}. a photo of the nice {label}. a photo of the small {label}. a photo of the weird {label}. the cartoon {label}. art of the {label}. a drawing of the {label}. a photo of the large {label}. a black and white photo of a {label}. the plushie {label}. a dark photo of a {label}. itap of a {label}. graffiti of the {label}. a toy {label}. itap of my {label}. a photo of a cool {label}. a photo of a small {label}. a tattoo of the {label}.

Table 16: Full list of prompts to evaluate the performance of zero-shot classification on 11 visual recognition datasets.

In the image, there are two main objects placed on a wooden surface against a beige wall. The first object is a white owl-shaped candle holder. The owl, made of a white material, is sitting on the left side of the image. It has a small candle inside it, which is lit, casting a warm glow. The second object is a goldcolored clock. The clock is standing upright the image. It has a white face with black numbers and hands. The clock and the owl are positioned in such a way that they seem to be in conversation with each other, creating an interesting visual dynamic.

[Figure 20]

[Figure 21]

ProCLIP

[Figure 22]

[Figure 23]

CLIP

In the tranquil expanse of a grassy field, a zebra, adorned with a mesmerizing pattern of black and white stripes, is captured in the act of grazing. The zebra's head is lowered towards the earth, its mouth busy nibbling on the lush green grass beneath it. The field, a vibrant canvas of green, is speckled with patches of white flowers that add a touch of contrast to the scene. The zebra's mane, standing upright, adds a dynamic element to the otherwise serene landscape. The image is a beautiful representation of nature's simplicity and elegance.

[Figure 24]

[Figure 25]

[Figure 26]

ProCLIP

[Figure 27]

CLIP

[Figure 28]

In the heart of a verdant field, a spectacle unfolds. Two majestic horses, one as white as a cloud and the other as black as night, are caught in a moment of pure athleticism. They are rearing up on their hind legs, their front legs reaching for the sky in a display of power and grace. The riders, clad in protective helmets, hold on tightly, their bodies leaning forward in sync with the horses’ movements. The field is a vibrant green, a stark contrast to the horses and riders. In the distance, trees stand tall and proud, their leaves rustling gently in the breeze.

[Figure 29]

#### ProCLIP

[Figure 30]

[Figure 31]

#### CLIP

The image captures a vibrant and appetizing bento box, a traditional Japanese meal served in a compartmentalized box. The box is divided into four distinct sections, each filled with a variety of food items. In the top left section, there‘s a slice of bread. Adjacent to it, in the top right section, are chunks of pineapple, their bright yellow color adding a tropical touch to the meal. The bottom left section is filled with a serving of broccoli, its green color standing out against the yellow of the box. Next to it, in the bottom right section, is a serving of meatballs, their brown color hinting at a savory flavor.

[Figure 32]

[Figure 33]

ProCLIP

[Figure 34]

[Figure 35]

CLIP

Figure 7: The visualization of ProCLIP vs CLIP in long-text cross-modal retrieval. From top to bottom: Case 1, Case 2, Case 3, and Case 4.

