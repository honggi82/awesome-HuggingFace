# arXiv:2406.09412v1[cs.CV]13Jun2024

## Explore the Limits of Omni-modal Pretraining at Scale

##### Yiyuan Zhang1,4∗ Handong Li2,3∗ Jing Liu2,3† Xiangyu Yue1

1 Multimedia Lab, The Chinese University of Hong Kong

- 2 School of Artificial Intelligence, University of Chinese Academy of Sciences
- 3 Institute of Automation, Chinese Academy of Science 4 Shanghai AI Laboratory

#### Abstract

We propose to build omni-modal intelligence, which is capable of understanding any modality and learning universal representations. In specific, we propose a scalable pretraining paradigm, named Multimodal Context (MiCo), which can scale up the numbers of modalities and amount of data, together with the model parameters, in the pretraining process. With MiCo, the pretrained models show significant emergent abilities in multimodal learning, which are evaluated on the following tasks: i) single-modality perception benchmarks of 10 different modalities, ii) 25 cross-modality understanding tasks of retrieval, question-answering, captioning, and iii) 18 multimodal large language model benchmarks. Our models establish 37 new records for state-of-the-art performance. We hope that our research could contribute to the development of omni-modal intelligence. Code and Models

#### 1 Introduction

Working

In the development of artificial intelligence, scalable pre-training has emerged as a promising pathway towards general intelligence [1–5]. Additionally, pre-training has been established as an effective

[Figure 1]

[Figure 2]

[Figure 3]

Temporal Video Paired Normal Maps

Paired Depth

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

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

Visual Content Caption

Training Omni-modal Intelligence

“Change a place, Will my life be better? when leaving here, I know nothing When the hustle and bustle of the city covers up the trembling of the heart, I always fantasize about a completely different life… Still far away In the scenery that I could not reach when I was a child, I started to hear ”

[Figure 55]

[Figure 56]

Learning Universal Representations

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

Audio Paired Caption

[Figure 62]

Video Paired Audio

“A man is speaking in a foreign language while music plays in the background. ”, “The music is playing” , “a male voice is being recorded.” .

[Figure 63]

Figure 1: Omni-modal Pretraining. We propose collecting large-scale omni-modal paired data, including text, image, video, depth, and normal maps, to learn universal representations.

approach for learning more general and transferable representations across various modalities. For

∗Equal Contribution †Corresponding Author

Preprint. Under review.

example, CLIP [4] constructs million-scale text-image pairs for cross-modal contrastive learning, making it one of the most impactful foundation models in the community [6, 7]. Researchers have further extended the capabilities of CLIP [4] to more data modalities, e.g. audio [8], point clouds [9], and more comprehensive tasks, e.g. reasoning about images/ videos with large language models (LLMs) [10, 11]. The main contributions of CLIP [4] are two-fold: collecting web-scale text-image data and proposing a scalable vision-language pretraining paradigm. As more modalities e.g. audio, video, and 3D content, are getting widely used in this multimodal era [12, 11, 13, 14, 6, 7], such developments present additional challenges, including multimodal misalignment, misinterpretation, and bias amplification, in achieving coherent multimodal understanding with LLMs.

Working

In this paper, we aim to enhance the comprehensive abilities of CLIP in visual understanding and further bolster its multimodal capacities across audio, video, 3D content, and more, as illustrated in Figure 1. This is significantly challenging. Therefore, we shift our focus from training a general multimodal model to understanding how the human brain performs coherent multimodal cognition. As outlined in Richard Mayer’s Cognitive Theory of Multimedia Learning [15], our brain processes multimedia signals through two distinct channels—auditory and visual—in sensory memory, as depicted in Figure 2. The sensory memory integrates these signals with prior knowledge through words, transforming new multimedia information into long-term memory. Notably, 1) multimedia signals in the brain share channels, and 2) words function as the reasoning interface in our brain.

Omni-Encoder (ViT)

Interface

Alignment

Sensory Memory

Generative Reasoning

[Figure 64]

Words

[Figure 65]

LLM

Ears

“Interface” Modality

[Figure 66]

[Figure 67]

[Figure 68]

ViT

[Figure 69]

[Figure 70]

Prior Knowledge

Photos

Learned Representations

Eyes

“Knowledge” Modality

(a) Dual-Channel Multimodal Cognition Theory

(b) Brain-Inspired Omni-modal Learning Architecture

- Figure 2: Multimedia Cognition Process in Brain Inspires our Design. We split diverse modalities into two types and employ individual neural networks to learn representations from each type respectively.

Inspired by these insights, we categorize diverse modalities into two types: “knowledge modality” and “interface modality”. Knowledge modalities, primarily derived from raw sensors, contribute knowledge in diverse formats. For example, images and depth maps offer visual knowledge, while audio and video provide auditory and spatiotemporal knowledge. The language modality, developed by humans, is inherently more abstract and naturally functions as the interface modality, facilitating learning, reasoning, and the coordination of knowledge. To this end, we design an omni-modal learning architecture, illustrated in Figure 2 (b), with two distinct branches: one for knowledge modalities and one for the interface modality, i.e. natural language. The knowledge and interface modalities are aligned through a novel generative reasoning method, as detailed in § 3.3.

In addition to the architecture design, the next challenge is how to further enhance the benefits of integrating multiple data modalities. The key to learning token sequences is the context relationship [21], which assigns a unique vector to each input position in a sequence. This approach improves sequence modeling by capturing the sequential relationship among tokens. Moreover, since different modalities (e.g., text, image, audio) offer complementary information, integrating these sources fosters a more comprehensive understanding of the data. Modeling token sequences from different modalities under the same context can help the model understand modality characteristics and joint semantics.

Therefore, we propose a Multimodal Context (MiCo) framework. We first map different modalities into a joint embedding space by sharing backbone networks. Then we build contextual relationships by sharing the same position embeddings and utilizing additional context embeddings to enhance coherent multimodal understanding, as shown in Figure 1. Subsequently, we employ omnimodal contrastive learning, omnimodal feature matching, and omnimodal caption generation processes for pretraining (detailed in § 3.4). Moreover, MiCo can incorporate existing text-image, text-audio, and text-video datasets for joint multimodal context learning (§ 3.3), which leads to better omni-modal learning capacity, further modality extensibility, and easier scalability of multimodal data. Meanwhile, we are the first to explore multimodal scaling laws in pretraining modalities, model parameters, and data scales (detailed in Figure 6).

(a) Masked Modelling with Autoencoders (b) Contrastive Learning (c) Multimodal Context (Ours)

SimCLR, MoCo, etc.

[Figure 71]

[Figure 72]

“When leaving here, I know nothing. When the hustle and bustle of the city covers up the trembling of the heart”

[Figure 73]

Visual Caption

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

| | | | |[Figure 79]| |
|---|---|---|---|---|---|
|[Figure 80]| | | | | |
| | | | | | |
| | |[Figure 81]<br><br>| | | |

| | |
|---|---|
| | |
| | |
| | |

[Figure 82]

[Figure 83]

Audio Caption

[Figure 84]

CLIP

- 1. Feature Contrastive learning
- 2. Token Masked Generation

“A trio of roses in soft hues of pink and coral”

BERT, MAE, AudioMAE, etc.

Transferable, Modality Tuples

Omni-modality, Transferable, General- purpose

General-purpose, Single-modality

- Figure 3: Evolution of Pretraining Paradigms. Masked modeling [16–18] has shown great success in singlemodality general-purpose understanding. Contrastive learning [19, 4, 20] distinguishes transferable features with modality tuples. We aim to achieve general-purpose omni-modal understanding and learn transferable, universal representations.

- As shown in Figure 3, we compare MiCo with existing pretraining approaches. With omnimodal contrastive learning, omnimodal feature matching, and omnimodal caption generation processes, MiCo successfully integrates the advantages of both masked modeling and contrastive learning. In other words, MiCo represents the next-generation evolution of masked modeling [16–18] and contrastive learning methods [19, 4, 20] for the multimodal era, offering significant benefits in omnimodal learning, strong transferability, and general-purpose representations. To thoroughly evaluate the effectiveness of MiCo, we conduct extensive experiments on universal single-modality perception benchmarks, cross-modal retrieval, captioning, and question-answer (QA) benchmarks, as well as zero-shot QA benchmarks for multimodal large language models. MiCo achieves impressive results across these benchmarks, establishing more than 37 new state-of-the-art (SOTA) performances and showing remarkable improvements of over 20% on some benchmarks. These results compellingly illustrate that MiCo is a promising next-generation pretraining paradigm for the multimodal era. 2 Related Work

Vision-Language Pretraining. MCAN [22] first aligns vision and language features by stacking deep cross-attention blocks. Then more works [23–26] scale their models and improve the visionlanguage fusion process to build better alignment. VL-BERT [27] introduced the Masked Language Model (MLM) paradigm, focusing on generic tasks across both vision and language modalities. Then Oscar [28] proposed to enrich the representation of object semantics by integrating visual and textual content. Subsequent frameworks have further refined and extended these capabilities. Notably, VinVL [29], SimVLM [24], VLMO [23], ALBEF [30], and Florence [31] have explored and demonstrated the advantages of joint representations that ensure semantic consistency across the visual and natural language. Additionally, the versatility of multimodal models extends into specialized applications such as few-shot learning [32], and sequence-to-sequence [25, 33]. BEiTv3 [26] treats images as a “foreign language”, employing a cross-modal mask-and-reconstruction process with partially shared parameters.

More-Modality Pretraining. MMV [34] pioneered multimodal pretraining using text, video, and audio pairs. They proposed multimodal contrastive learning for alignment. Then VATT [35] further developed pretraining multiple modalities with transformers. After CLIP [4], more works [36, 14, 8, 9, 37, 38] propose to adapt pretrained CLIP models to more modalities including point cloud, depth, audio, video, etc. Another direction is to exploit multimodal complementary benefits and construct better and more modality pairs for pertaining foundation models such as VAST [39] and VALOR [40], which improve the abilities for multimodal understanding.

Despite significant advancements in multimodal learning, several key challenges impede the development of comprehensive omni-modal intelligence: 1) Focus on Vision-Language Modalities: Current methods [26, 23, 28, 24] predominantly cater to vision and language tasks. The inflexibility of these works limits the extension with more modalities such as video, depth, normals, and audio.

- 2) Architectural Constraints: The development of architectures capable of handling a broader array of modalities is still in its nascent stages. Crafting scalable and efficient multimodal learning architectures presents a significant challenge. 3) Data Availability: There is a notable scarcity of publicly accessible multimodal datasets that include paired data (such as video, depth, audio, and

captions). 4) Utilizing Multimodal Benefits: Although leveraging the synergistic benefits of multiple modalities is crucial for achieving omni-modal intelligence [41], understanding and optimizing the interaction between highly disparate modalities remains a complex and largely unexplored area.

#### 3 Multimodal Context

##### 3.1 Large-Scale Data Collection

We use the HD-VILA [42] dataset, which contains 371.5K hours of 720p (1280 × 720) videos. We remove video clips that are shorter than 5s or longer than 30s. Then, we collect a dataset containing 1.7M paired video clips (∼510M frames), audio, and subtitles {(xV ,xVT ,xA)}. Then we enrich the dataset by adding captions to video frames (images), and audio with pretrained captioners [39], getting (xI,xIT) and (xA,xAT ). Finally, we use pretrained monocular depth estimation models [43, 44]3 to generate depth and normal maps, getting (xI,xD,xN). Thus, we collect million-scale multimodal paired data {(xI,xD,xN,xIT),(xA,xAT ),(xV ,xVT )}, where xT,xI,xA,xV ,and xD denote the modality-specific samples of text captions, image, audio, video clips, depth, and normal maps. We split our dataset into several subsets including 1M, 10M, 110M, and 334M multimodal data pairs, and we provide detailed illustrations in Appendix C.

##### 3.2 Architecture Design for Omni-modal Learning

We first investigate several variants of encoder architectures with four data modalities. With our collected data, we pretrain architectures for 300K steps by the same contrastive [4] and maskedgeneration loss functions [26] (details in Appendix B). We take the captioning and retrieval tasks on image, audio, and video modalities as the main evaluation benchmark for designing architectures.

Architectural Designs. We construct the vanilla architecture from CLIP [4]. A text encoder of Transformer [21] takes text inputs and outputs text embeddings zT, and an image encoder of Vision Transformer [45] takes image input xI ∈ R3×H×W and outputs image embeddings zI, respectively.

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

ViT

LLM (LLama2) LLM

Text Encoder (BERT)

ViT ViT ViT BERT

(iv) ViT + LLM

(i) Modality-Specific

(ii) Text Encoder Only

(iii) LLM Decoder Only

Figure 4: Options of Architecture Design for Omni-Modal Pretraining.

- As shown in Figure 4, we propose 4 architectures for omni-modal learning: i) Modality-specific encoders for each modality, employing individual transformers to extract multimodal embeddings, then fuse them as BEiT-3 [26]. ii) BERT (text encoder) as a unified multimodal encoder to extract multimodal embeddings and generates texts. iii) LLM (text decoder) as a unified multimodal encoder and text generator. iv) A ViT as a unified multimodal encoder besides text, and an LLM deals with text embeddings and generation.

Empirical Discovery. Referring to Table 1, we conclude several guidelines for designing architectures in omni-modal learning: 1) Pure language models are difficult to retrieval tasks. Both (ii) and (iii) deliver a significant performance drop in retrieval tasks. 2) No more than 2 Encoders. Comparing (i) with (ii) & (iii), we observe that additional encoders are beneficial for retrieval tasks; however, comparison between (i) and (iv) suggests that discrepancies among multiple encoders can also hinder multimodal alignment. 3) Language is an individual branch for alignment. Comparing (ii) & (iii), with (iv), improvements are significant in both retrieval and captioning.

3Geowizard [43] delivers significantly better annotations obviously, while DPT [44] predicts much faster (about 34.7˜ × faster). We use the Geowizard to annotate the high-quality data about 2M.

- Table 1: Architecture Design of Omni-Modal Learning Paradigm. The default backbone and LLM are ViT-g and Llama-2-7B [46]. We pretrain models for 300k steps, then evaluate performances on the MSRVTT, VATEX, AudioCaps, ClothoV2, COCO, and Flicker datasets for caption (CIDEr) and retrieval tasks (R@1).

Video Audio Image MSRVTT VATEX AudioCaps ClothoV2 COCO Flickr CIDEr (%) R@1 (%) R@1 (%) CIDEr (%) R@1 (%) R@1 (%)

Architecture

- (i) Modality-Specific 74.3 73.5 42.3 22.3 65.2 88.4
- (ii) Text Encoder (BERT) 77.0 53.2 23.1 43.9 46.7 51.6
- (iii) LLM (LLama-2-7B) 75.2 60.3 14.7 43.6 60.8 81.3
- (iv) ViT + LLM 77.9 79.5 49.7 47.2 67.5 90.5

##### 3.3 Multimodal Context Construction

Preliminary. The context is proposed to assign a unique vector to each token in a sequence [21], which reinforces potential relevance between positions. Different modalities (e.g., text, image, audio) provide complementary information. Learning multimodal context leads to a more holistic and nuanced understanding of data. It can also leverage the strengths of each modality and guide the model to understand the interactions between different types of information. Therefore, we seek to construct the context relationship across diverse modalities and extend the learning capacity to omni-modalies. We provide the overview of mico pretraining paradigm in Figure 5.

Single Dataset with Multimodal Paired Data. As mentioned in § 3.1, we build a dataset with multimodal paired data {(xI,xD,xN,xIT),(xA,xAT ),(xV ,xVT )}, then we employ the omni-modal encoder f(·;θ) to extract features zI,zA,zV ,zD, and zN, then use text encoder to extract text features zT. Therefore, we construct the context by a top-down design: 1) For the whole multimodal embeddings, they share the same position embeddings EPos to build a modality-fused context relationship across diverse modalities. 2) Then, for each specific context, they’re labeled by modality embeddings including EMI ,EMA,EMV ,EMD,EMN, etc to indicate modality types. 3) Within the same modality context, we employ the context embeddings ECI to construct uni-modal context relationships. Thus, the construction of the multimodal context can be formulated as:

zI = [zI1,zI2,··· ,zL

I ] + ECI, for each modality, z = [zI + EMI ,zA + EMA,zV + EMV ,zD + EMD,zN + EMN] + EPos,

I

(1)

where ECI is up to the sample length of a specific modality. Meanwhile, the text features of specific captions can be easily concatenated, where their position embeddings EPos′ are also shared:

zT = [zTI ,zTA,zTV ] + EPos′ . (2)

Please note that, in the context construction, we use vanilla position embedding [21] to build these contexts in all of these learnable embeddings EPos,EPos′ ,EM, and EC instead of Rotary Position Embedding (RoPE) [47], which is similar to EVA-CLIP-8B [48] and EVA-CLIP-18B.

Multiple Datasets Combination of Cross-Modal Datasets. Besides multimodal paired data, our proposed paradigm can also leverage existing web-scale text-image, text-audio, and text-video datasets to jointly pretraining models towards omni-modal universal representations. Given datasets

DI = {(xjI,xjT)}N

j=1,DA = {(xjA,xjT)}N

j=1, and DV = {(xjV ,xjT)}N

j=1, each pair of data possess local and simple context, for example, a pair of text-image data (xI,xT) corresponds to a simple context (zI + EPos,EPos′ ), which may limit the learned representations of models. We propose to build the multimodal context by cross-dataset joint sampling with sampling context embedding ESam:

I

A

V

(xI,xIT) = Sample(DI), (xA,xAT ) = Sample(DA), (xV ,xVT ) = Sample(DV ),

zI = f(xI;θ) + ESamT−I, zTI = f′(xT;θ′) + ESamT−I, for each modality, z = [zI + EMI ,zA + EMA,zV + EMV ] + EPos, zT = [zTI ,zTA,zTV ] + EPos′ .

(3)

In this way, we successfully combine existing multiple cross-modal datasets towards learning omnimodal universal representations by building more universal and complicated multimodal contexts (Equation 3) for pretraining models, therefore,

mico can outperform existing pretraining methods by better generalization learning ability, modality extensibility, and easier for scaling data.

Working

Omni-Modal Contrastive Learning Omni-Modal Feature Matching Omni-Modal Caption Generation

|V1| |
|---|---|

| |VN|
|---|---|

|V1|VN|D1|DN|A1|AN|
|---|---|---|---|---|---|

Masked

|T1|
|---|
|T2|
|T3|
|T4|
|…|
|TN|

|D1| |
|---|---|

| |
|---|

|DN|
|---|

Concatenate

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

“Will you choose to travel and update your mind?”

Logits

Causal Inference

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

|A1| | | | |AN|
|---|---|---|---|---|---|

“A man is speaking with music playing in the background”

…

… … …

…

…

*

*

*

*

*

*

Omni-Encoder (ViT) Text Encoder / LLM

“Change a place, Will my life be better? when leaving here I know nothing”

“A man is speaking in a foreign language while music plays in the background. ”,

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

Temporal Video Paired Depth Video Paired Audio Paired Normal

Audio Paired Caption

Visual Content Caption

- Figure 5: Overview of Multimodal Context Pretraining Paradigm. We use a shared ViT for multimodal feature extraction, and another branch is to employ a text encoder. We concatenate these multimodal sequences as multimodal contexts and perform contrastive learning and masked modeling.

##### 3.4 Pretraining Objectives

Omni-modal Contrastive Learning. The omni-modality representations are denoted as z. Subsequently, z and zT are projected into the same space using MLPs. The omni-modal contrastive learning is formulated by the dot product of z and zT. We use vz and vT to denote projected vectors:

NB

NB

exp(τ· < viz,viT >) NB j=1 exp(τ· < viz,vjT >))

exp(τ· < viz,viT >)) NB j=1 exp(τ· < vjz,viT >))

- 1

- 2

- 1

- 2

, (4)

LCon = −

−

log

log

i=1

i=1

where < ·,· >, NB and τ denote the dot product, batch size, and a learnable parameter.

Omni-modal Feature Matching Process is designed to improve the semantic alignment between multimodal (knowledge modalities) and textual features. We employ an MLP layer to perform binary predictions pv of (z,zT). Following a hard negative mining strategy [30], we assigns y = 1 if features are matched, and y = 0 otherwise.

LMatch = E(viz,viT)∼(Z,T ) [y log pv + (1 − y)log (1 − pv)] (5)

Omni-modal Caption Generation Process. We employ conditional causal masked (60%) language modeling for generative omni-modal reasoning. In specific, a single-directional causal attention mask is used to avoid information leakage, and the masked tokens are reconstructed using a prediction layer of BERT [18]. We use cm and c<m to denote masked tokens and former tokens, respectively.

LGen = −E(viT,viT)∼(V,T ) log P (cm | c<m,vz) (6)

#### 4 Experiment

##### 4.1 Experimental Setup

We evaluate our model on three different benchmarks: 1) Single-modality Understanding § 4.2 (following previous practices [4, 36, 14] in fine-tuning & zero-shot setting in classification and forecasting tasks), 2) Cross-modality Understanding § 4.3 (following BEiT-3 [26], VAST [39] in finetuning and dataset splits for Caption, QA, and retrieval tasks), and 3) Multimodal Understanding with

Large Language Models § 4.4 (following LLava [49], VideoChat [50], OneLLM [12] in multimodal zero-shot QA). Detailed experimental settings including datasets introduction, splits, and evaluation metrics can be found in our Appendix C.

Implementation Details. We implement our pretraining paradigm with ViT backbone scaling from ViT-B (8 NVIDIA Tesla A100 GPUs) to ViT-g (64 NVIDIA Tesla A100 GPUs). It takes about 2∼6 days for pretraining. We pretrain models 200k steps for ViT-B, 300k steps for ViT-L and ViT-g. The initial learning rate is set to 1e-4, and a linear decay schedule is used. The batch size on each GPU is set to 1,024. More implementation details can be found in the Appendix B.

- Table 2: State-of-the-art Abilities of MiCo for Omni-modal Perception. We conduct experiments on the single modality evaluation following the same practice of previous Sotas. We report the Accuracy (%) of MMLU [51], IN-1K [52], K700 [53], NYU-D [54], Ego4D [55], Indian Pines, and Fraud datasets, R@1 (%) for MSR-VTT [56] and SYSU [57], mAP for AS-2M [58], F1-score for Fraud, and Mean Absulte Error↓ for PCQM4M and Global Weather Forecasting [59] benchmarks.

|Methods (Backbone)|[Figure 124]<br><br>[Figure 125]<br><br>[Figure 126]<br><br>[Figure 127]<br><br>[Figure 128]<br><br>[Figure 129]<br><br>[Figure 130]<br><br>[Figure 131]<br><br>[Figure 132]<br><br>[Figure 133]<br><br>[Figure 134]<br><br>Text Image Video Depth Audio Thermal IMU Graph Time-Series Hyperspectral Tabular<br><br>MMLU IN-1K K700/MSR-VTT NYU-D AS-2M SYSU Ego4D PCQM4M Global Weather IP Fraud<br><br>|
|---|---|
|ImageBind (ViT-H) [14] Meta-Trans (ViT-L) [36] Absolute SOTA|43.6 80.2 42.9/36.8 54.0 43.4 72.6 25.0 0.815↓ 8.439↓ 83.6 0.847 37.3 88.1 33.2/31.5 41.5 38.9 71.3 73.9 0.886↓ 7.892↓ 78.1 0.809 90.0 [60] 91.0 [33] 92.1/62.8 [38] 76.7 [61] 48.6 [62] 77.9 [63] 52.5 [64] 0.123 [65] 7.602↓ [13] 98.0 [66] 0.860 [67]<br><br>|
|MiCo (ViT-g) [Ours]<br><br>|68.9 89.8 91.6/64.3 84.6 50.5 80.3 77.2 0.742↓ 7.834↓ 98.5 0.913|

- 4.2 Evaluation on Single-modality Understanding

Exceptional Omni-modal Perception Abilities. As shown in Table 2, MiCo achieves stateof-the-art performances on a range of benchmarks across 10 modalities. For text understanding (MMLU), MiCo attains the accuracy of 68.9%, outperforming both ImageBind [14] (43.6%) and Meta-Transformer [36] (37.3%). In image recognition (IN-1K), MiCo delivers Top-1 Acc. of 89.8%. On K700 and MSR-VTT, MiCo achieves 91.6% for Acc. and R@1 of 64.3%, outperforming existing retrieval methods. Regrading 3D singe-view tasks (NYU-D), MiCo outperforms the absolute SOTA [61] by +7.9%. On AS-2M, MiCo achieves the mAP of 50.5%, which is better than BEATS-

- 3 [62] by +1.9%. MiCo also excels in thermal sensing (SYSU) and IMU tasks (Ego4D), MiCo achieves an accuracy of 80.3% and 77.2%, respectively. These results highlight MiCo’s comprehensive and outstanding performances, establishing it as a powerful model for omni-modal perception.

- Table 3: Powerful Cross-Modal Abilities. We evaluate MiCo on the mainstream cross-modal tasks including 11 retrieval tasks (COCO [68], Flickr [69], ClothoV1 [70], ClothoV2 [70], AudioCaps [71], MSRVTT [72], YouCook2 [73], VALOR-32K [40], VATEX [74], DEDeMo [75], and ANET [76]), 7 caption tasks (COCO, ClothoV1, ClothoV2, AudioCaps, MSRVTT, YouCook2, VALOR-32K), and 6 question-answer (QA) tasks (TGIF [77], MSVD [78], VQAv2 [79], MSRVTT, MUSIC [80], and ANET) with the metrics of R@1, CIDEr, and Acc. Impressively, MiCo archives 20 new SoTA performances.

Text-to-Image Retrieval Image Caption Visual QA COCO Flickr Flickr(ZS) COCO TGIF MSVD VQAv2

[Figure 135]

Image

SOTA 68.3 [81] 90.3 [26] 89.7 [81] 154.9* [82] 78.7 [40] 60.2 [83] 84.3 [84] MiCo 68.1 91.1 ↑ 0.8 90.1 ↑ 0.4 152.4 78.9 ↑ 0.2 60.4 ↑ 0.2 80.5

Text-to-Audio Retrieval Audio Caption ClothoV1 ClothoV2 AudioCaps ClothoV1 ClothoV2 AudioCaps

[Figure 136]

Audio

SOTA 17.5 [40] 21.5 [85] 42.2 [85] 42.3 [40] 48.8 [85] 78.7 [85] MiCo 21.2 ↑ 3.7 23.3 ↑ 1.8 41.0 49.6 ↑ 7.3 50.8 ↑ 2.0 66.2

Text-to-Video-Audio Retrieval MSRVTT YouCook2 VALOR-32K VATEX DiDeMo ANET

[Figure 137]

[Figure 138]

Video-Audio

SOTA 54.4 [40] 31.3 [86] 73.2 [40] 76.9 [40] 57.6 [40] 63.4 [40] MiCo 64.3 ↑ 9.9 51.3 ↑ 20.0 78.7 ↑ 5.5 81.3 ↑ 4.4 63.6 ↑ 6.0 68.5 ↑ 5.1

Video-Audio Caption Video-Audio QA MSRVTT YouCook2 VALOR-32K MSRVTT MUSIC ANET

[Figure 139]

[Figure 140]

Video-Audio

SOTA 74.0 [40] 190.0 [87] 61.5 [40] 49.2 [40] 78.9 [40] 48.6 [40] MiCo 79.3 ↑ 5.3 197.8 ↑ 7.8 62.8 ↑ 1.3 50.4 ↑ 1.2 79.7 ↑ 0.8 51.0 ↑ 2.4

- Table 4: Evaluation on LLM Benchmarks. The MLLM evaluation involves 6 VQA tasks (GQA [88], VQAv2 [89], OKVQA [90], TextVQA (TVQA) [91], ScienceQA (SQA) [92] and Vizwiz [93]), 2 image captioning tasks (Nocaps [94] and Flickr30K [69]), and 4 multimodal benchmarks (MME [95], MM Bench (MMB) [96], MMVet [97] and SEED [98]). The LLMs are Chinchilla [99], Vicuna [100], Qwen [101], LLaMA [102] and LLaMA2 [46]. The evaluation metrics for VQA and captioning tasks are accuracy and CIDEr, respectively. The results in bold and underline are the best and second-best results, respectively.

Method LLM

Visual Question Answering Image Caption MM Benchmark GQA VQAv2 OKVQA TVQA SQA Vizwiz NoCaps Flickr MME MMB MMVet SEED

Vision Specialist LLM Flamingo-9B [32] Chinchilla-7B - 51.8 44.7 30.1 - 28.8 - 61.5 - - - Flamingo-80B [32] Chinchilla-70B - 56.3 50.6 31.8 - 31.6 - 67.2 - - - BLIP-2 [81] Vicuna-7B - - - 40.1 53.8 - 107.5 74.9 - - - BLIP-2 [81] Vicuna-13B 41.0 41.0 - 42.5 61 19.6 103.9 71.6 1293.8 - 22.4 InstructBLIP [103] Vicuna-7B 49.2 - - 50.1 60.5 34.5 123.1 82.4 - 36 26.2 InstructBLIP [103] Vicuna-13B 49.5 - - 50.7 63.1 34.3 121.9 82.8 1212.8 - 25.6 IDEFICS-9B [104] LLaMA-7B 38.4 50.9 38.4 25.9 - 35.5 - 27.3 - 48.2 - IDEFICS-80B [104] LLaMA-65B 45.2 60.0 45.2 30.9 - 36.0 - 53.7 - 54.5 - LLaMA-Ad.v2 [105] LLaMA-7B 43.9 - 55.9 43.8 54.2 - 42.7 30.5 972.7 38.9 31.4 32.7 Qwen-VL [106] Qwen-7B 57.5 78.2 56.6 61.5 68.2 38.9 120.2 81.0 1487.5 60.6 - 58.2 LLaVA-v1.5 [49] Vicuna-7B 62.0 78.5 - 58.2 66.8 50.0 - - 1510.7 64.3 30.5 58.6

Multimodal Generalist LLM ImageBind-LLM [107] LLaMA-7B 41.1 - - 24.0 51.4 - 29.6 23.5 775.7 - - ChatBridge-13B [108] Vicuna-13B 41.8 - 45.2 - - - 115.7 82.5 - - - AnyMAL-13B [109] LLaMA2-13B - 59.6 33.1 24.7 52.7 24.4 - - - - - AnyMAL-70B [109] LLaMA2-70B - 64.2 42.6 32.9 70.8 33.8 - - - - - OneLLM-7B [CVPR’24] LLaMA2-7B 59.5 71.6 58.9 34.0 63.4 45.9 115.9 78.6 1392.0 60.0 29.1 61.2 MiCo-Chat-7B Vicuna-7B 61.5 78.1 56.6 53.4 71.3 49.1 111.8 76.3 1485.7 65.2 31.4 67.7

- Table 5: Zero-Shot Audio & Video generative benchmark with LLMs. We evaluate models by audio captioning on Clotho Caption [70], audio QA on Clotho AQA [110] and VideoChatGPT scoring benchmark [111] using the same Vicuna-7B.

Clotho Caption Clotho AQA CIDEr SPIDEr Acc.

Cor. Det. Con. Tem. Cons.

Method 0-shot

Method

FeatureCut [112] ✗ 43.6 27.9 Wavcaps [85] ✗ 48.8 31.0 MWAFM [113] ✗ - - 22.2 Pengi [114] ✗ - 27.1 64.5

VideoLLaMA [115] 1.96 2.18 2.16 1.82 1.79 VideoChat [116] 2.23 2.50 2.53 1.94 2.24 Video-ChatGPT [111] 2.40 2.52 2.62 1.98 2.37 BT-Adapter [117] 2.68 2.69 3.27 2.34 2.46 LLaMa-VID [118] 2.96 3.00 3.53 2.46 2.51

ChatBridge-13B [108] ✓ 26.2 - OneLLM-7B ✓ 29.1 19.5 57.9 MiCo-Chat-7B [Ours] ✓ 33.3 21.9 63.9

MiCo-Chat-7B [Ours] 3.00 3.01 3.61 2.49 2.71

##### 4.3 Evaluation on Cross-Modal Understanding

Table 3 illustrates the powerful performances of MiCo on 25 cross-modal benchmarks, achieving more than 20 new SOTA performances. For text-to-image retrieval, MiCo achieves outstanding results with R@1 of 68.1% on COCO, and 91.1% on Flickr, outperforming previous SOTA methods. For VQA, MiCo demonstrates robust performance with accuracy scores of 78.9% on TGIF, 60.4% on MSVD, and 80.5% on VQA v2, highlighting its strong visual comprehension and reasoning abilities. In text-to-audio retrieval, MiCo achieves outstanding performances of 21.2% on ClothoV1, 23.3% on ClothoV2, and 41.0% on AudioCaps, while in audio captioning, it achieves 49.6% on ClothoV1, and 50.8% on ClothoV2, all outperforming previous best results. For text-to-video retrieval, MiCo sets new SOTA performances with metrics of 64.3% R@1 on MSRVTT and 81.3% on VATEX, and in video-audio caption, it achieves impressive performances of 79.3% on MSRVTT, 197.8% on YouCook2, and 62.8% on VALOR-32K. Finally, in video-audio QA, MiCo also delivers superior performances of 50.4% on MSRVTT, 79.9% on MUSIC, and 51.0 on ANET. These results collectively highlight MiCo’s exceptional and versatile capabilities in cross-modal comprehension and reasoning tasks, establishing it as a promising direction in this field.

##### 4.4 Evaluation on Multimodal Understanding with Large Language Models

MiCo highlights its Omni-modal Zero-shot Comprehension and Reasoning Abilities. Beyond traditional caption, retrieval, and QA tasks, we also evaluate the abilities of MiCo aligned with LLMs for zero-shot multimodal QA. We use ChatBridge [108] as our baseline and Vicuna-7B as the large language model for each modality. As shown in Table 4, 5, and 6, MiCo-Chat-7B shows outstanding performances across both Vision LLMs and Multimodal LLMs. It directly delivers outstanding performances on the SQA (71.3%), MMB (65.2%), MMVet (31.4%), and SEED (67.7%) benchmarks while another 4 competitive performances. Besides, MiCo-Chat-7B also delivers significantly impressive performances on both zero-shot caption and QA tasks on audio and video modalities, where MiCo-Chat-7B achieves 6 new SOTA performances including Clotho Caption,

###### Table 6: Zero-shot Video QA with LLMs. In comparison with leading methods, we report results with 1 token for each frame, where Res. indicates image resolution.

MSVD-QA MSRVTT-QA ActivityNet-QA Acc Score Acc Score Acc Score

Method LLM Res.

FrozenBiLM [119] DeBERTa-V2 224 32.2 – 16.8 – 24.7 – VideoLLaMA [120] Vicuna-7B 224 51.6 2.5 29.6 1.8 12.4 1.1 LLaMA-Adapter [105] LLaMA-7B 224 54.9 3.1 43.8 2.7 34.2 2.7 VideoChat [116] Vicuna-7B 224 56.3 2.8 45.0 2.5 26.5 2.2 Video-ChatGPT [111] Vicuna-7B 224 64.9 3.3 49.3 2.8 35.2 2.7 LLaMA-VID [118] Vicuna-7B 224 69.7 3.7 57.7 3.2 47.4 3.3 VideoChat2 [50] [CVPR’24] Vicuna-7B 224 70.0 3.9 54.1 3.3 49.1 3.3

MiCo-Chat-7B Vicuna-7B 224 73.7 4.1 60.1 3.6 50.1 3.3

###### Table 7: Ablation Study on pretraining modalities, data scale, pretraining process, and parameters. Our default setting is to pretrain a base model for 30k steps with 10M data using all objective functions and evaluate it on the MSRVTT, VATEX, DIDEMO, MSVD, AudioCaps, ClothoV2, COCO, and Flicker datasets for retrieval tasks.

[Figure 141]

[Figure 142]

[Figure 143]

Video Audio Image Average MSRVTT(VA) VATEX(VA) DIDEMO(VA) MSVD(V) AudioCaps(A) ClothoV2(A) COCO(I) Flickr(I)

Model Factors

###### Pretraining Modalities

- (a) I 39.7 57.3 38.4 39.7 10.2 4.4 50.2 75.7 39.4
- (b) I+3D 42.0 58.5 38.1 40.1 10.8 4.2 51.2 76.9 40.2
- (c) I+A 37.6 56.2 30.8 36.2 22.0 14.5 46.8 71.0 39.4
- (d) I+V 41.7 60.9 39.2 42.6 12.2 5.1 51.3 77.0 41.3
- (e) I+V+A 42.2 61.1 40.1 41.2 23.4 15.4 48.7 74.2 43.2
- (f) I+V+A+3D 45.7 ↑ 6.0 64.0 ↑ 6.7 42.7 ↑ 4.3 42.8 ↑ 3.1 24.6 ↑ 14.4 15.9 ↑ 11.5 49.9 77.1 ↑ 1.4 45.3 ↑ 5.9 Data Scale

- (h) 1M 44.2 63.2 40.1 40.7 21.9 11.2 48.2 77.5 43.4
- (i) 10M 45.7 64.0 42.7 42.8 24.6 15.9 49.9 77.1 45.3
- (j) 110M 48.5 65.7 41.7 43.0 26.3 17.1 49.6 78.1 46.3
- (k) 334M 49.1 ↑ 4.9 66.3 ↑ 3.1 43.2 ↑ 3.1 44.1 ↑ 3.4 27.0 ↑ 5.1 17.5 ↑ 6.3 51.5 ↑ 3.3 80.9 ↑ 3.4 47.5 ↑ 4.1 Pretraining Process

- (l) LCon 40.1 57.4 39.1 41.4 23.1 14.4 47.4 73.7 42.1
- (m) LCon + LMatch 43.9 61.4 38.0 41.6 23.6 15.5 48.8 74.3 43.4
- (n) LCon + LMatch + LGen 45.7 ↑ 5.6 64.0 ↑ 6.6 42.7 ↑ 3.6 42.8 ↑ 1.4 24.6 ↑ 1.5 15.9 ↑ 1.5 49.9 ↑ 2.5 77.1 ↑ 3.4 45.3 ↑ 3.2 Model Scale

- (o) Base-86M 45.7 64.0 42.7 42.8 24.6 15.9 49.9 77.1 45.3
- (p) Large-331M 58.2 72.0 57.2 52.8 31.6 18.7 60.8 87.5 54.9
- (q) Giant-1.3B 62.5 ↑ 16.8 79.9 ↑ 15.9 61.1 ↑ 18.4 56.0 ↑ 13.2 37.4 ↑ 12.8 20.8 ↑ 4.9 67.1 ↑ 17.2 90.7 ↑ 13.6 59.4 ↑ 14.1

AQA, MSVD-QA, MSRVTT-QA, ActivityNet-QA. These results are important proof that the MiCo pretraining paradigm shows a promising direction in developing large omni-modal models.

14

17.5 ViT-L ViT-g

| |1M<br><br>10M<br><br>100M 300M<br><br>| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

15.0

Image

Contrastive Con+Match Con+Match+Gen

Image+Video

12

15.0

3.0

12.5

Image+Video+Audio

Image+Video+Audio+Depth

12.5

10

10.0

Loss

2.5

10.0

7.5

8

7.5

5.0

2.0

6

2.5

5.0

1.5

4

0k 100k 200k 300k

0k 100k 200k 300k

0k 100k 200k 300k

0k 100k 200k 300k

Step

Step

Step

Step

- Figure 6: Multimodal Scaling Law. Training loss curves for MiCo under scaling factors (modality, data, parameters, process) settings.

##### 4.5 Ablation Study: Multimodal Scaling Laws

Scaling Modalities. From (a) to (f), we gradually scale up input modalities. In Figure 6, all modalities (I+V+A+3D) achieves the highest scores, highlighting the importance and effectiveness of MiCo for diverse multimodal inputs.

Scaling Multimodal Data. From (h) to (k) in Table 7, we investigate the impact of the omni-modal data scale from 1M to 334M. It proves that the MiCo has great potential for further scaling.

Pretraining Objectives. From (l) to (n), we analyze the impact of each pretraining objective. The combination of contrastive, matching, and generative losses (LCon + LMatch + LGen) yields the best performance, demonstrating the value of multiple complementary objectives.

Scaling Parameters. From (o) to (q), we assess the effect of model size. Larger models, particularly the Giant-1.3B, show superior performance, confirming that increasing model parameters with MiCo enhances learning and generalization abilities across diverse modalities.

#### 5 Conclusion and Limitation

In this paper, we propose a novel framework, termed MiCo, to train foundation models with enhanced visual perception abilities and omni-modal capacities. With experiments on a reasonably large scale of both model and data, we conclude that the key to omni-modal learning is to simulate the multimedia cognition process of the human brain. In MiCo, we use image, depth, and normal maps to simulate the fundamental visual perception ability, distance spatial awareness, and geometry awareness of human visual cognition. In addition, captions, audio, and video provide prior knowledge, auditory perception, and spatial-temporal awareness. In future work, we plan to enhance our joint pretraining by incorporating additional modalities, including optical flow, IMU data, and event files, etc. We believe MiCo is an important attempt to simulate the multimedia cognition of human brains, and we expect it could inspire future works to develop more powerful omni-modal foundation models.

#### References

- [1] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019. 1
- [2] OpenAI. Gpt-4 technical report. ArXiv, abs/2303.08774, 2023.
- [3] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877– 1901, 2020.
- [4] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763. PMLR, 2021. 2, 3, 4, 6
- [5] Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint arXiv:2303.12712, 2023. 1
- [6] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2
- [7] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 2
- [8] Andrey Guzhov, Federico Raue, Jörn Hees, and Andreas Dengel. Audioclip: Extending clip to image, text and audio. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 976–980. IEEE, 2022. 2, 3
- [9] Le Xue, Mingfei Gao, Chen Xing, Roberto Martín-Martín, Jiajun Wu, Caiming Xiong, Ran Xu, Juan Carlos Niebles, and Silvio Savarese. Ulip: Learning a unified representation of language, images, and point clouds for 3d understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1179–1189, 2023. 2, 3
- [10] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024. 2
- [11] Kunchang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 2
- [12] Jiaming Han, Kaixiong Gong, Yiyuan Zhang, Jiaqi Wang, Kaipeng Zhang, Dahua Lin, Yu Qiao, Peng Gao, and Xiangyu Yue. Onellm: One framework to align all modalities with language. arXiv preprint arXiv:2312.03700, 2023. 2, 7

- [13] Xiaohan Ding, Yiyuan Zhang, Yixiao Ge, Sijie Zhao, Lin Song, Xiangyu Yue, and Ying Shan. Unireplknet: A universal perception large-kernel convnet for audio, video, point cloud, time-series and image recognition. arXiv preprint arXiv:2311.15599, 2023. 2, 7, 22
- [14] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind: One embedding space to bind them all. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15180–15190, 2023. 2, 3, 6, 7
- [15] Richard E Mayer. Multimedia learning. In Psychology of learning and motivation, volume 41, pages 85–139. Elsevier, 2002. 2
- [16] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16000–16009, 2022. 3
- [17] Po-Yao Huang, Hu Xu, Juncheng Li, Alexei Baevski, Michael Auli, Wojciech Galuba, Florian Metze, and Christoph Feichtenhofer. Masked autoencoders that listen. Advances in Neural Information Processing Systems, 35:28708–28720, 2022.
- [18] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In NAACL-HLT, 2019. 3, 6
- [19] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9729–9738, 2020. 3
- [20] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. arXiv preprint arXiv:2002.05709, 2020. 3
- [21] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 2, 4, 5
- [22] Zhou Yu, Jun Yu, Yuhao Cui, Dacheng Tao, and Qi Tian. Deep modular co-attention networks for visual question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6281–6290, 2019. 3
- [23] Wenhui Wang, Hangbo Bao, Li Dong, and Furu Wei. Vlmo: Unified vision-language pretraining with mixture-of-modality-experts. arXiv preprint arXiv:2111.02358, 2021. 3
- [24] Zirui Wang, Jiahui Yu, Adams Wei Yu, Zihang Dai, Yulia Tsvetkov, and Yuan Cao. Simvlm: Simple visual language model pretraining with weak supervision. arXiv preprint arXiv:2108.10904, 2021. 3
- [25] Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. arXiv preprint arXiv:2202.03052, 2022. 3
- [26] Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, et al. Image as a foreign language: Beit pretraining for all vision and vision-language tasks. arXiv preprint arXiv:2208.10442,

2022. 3, 4, 6, 7

- [27] Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, and Jifeng Dai. Vl-bert: Pre-training of generic visual-linguistic representations. arXiv preprint arXiv:1908.08530,

- 2019. 3

[28] Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, et al. Oscar: Object-semantics aligned pre-training for vision-language tasks. In European Conference on Computer Vision, pages 121–137. Springer,

- 2020. 3

- [29] Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and Jianfeng Gao. Vinvl: Revisiting visual representations in vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5579–5588, 2021. 3
- [30] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. Advances in neural information processing systems, 34:9694–9705,

2021. 3, 6

- [31] Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, et al. Florence: A new foundation model for computer vision. arXiv preprint arXiv:2111.11432, 2021. 3
- [32] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. NeurIPS, 35:23716–23736, 2022. 3, 8
- [33] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. arXiv preprint arXiv:2205.01917, 2022. 3, 7
- [34] Jean-Baptiste Alayrac, Adria Recasens, Rosalia Schneider, Relja Arandjelovi´c, Jason Ramapuram, Jeffrey De Fauw, Lucas Smaira, Sander Dieleman, and Andrew Zisserman. Selfsupervised multimodal versatile networks. Advances in Neural Information Processing Systems, 33:25–37, 2020. 3
- [35] Hassan Akbari, Liangzhe Yuan, Rui Qian, Wei-Hong Chuang, Shih-Fu Chang, Yin Cui, and Boqing Gong. Vatt: Transformers for multimodal self-supervised learning from raw video, audio and text. Advances in Neural Information Processing Systems, 34:24206–24221, 2021. 3
- [36] Yiyuan Zhang, Kaixiong Gong, Kaipeng Zhang, Hongsheng Li, Yu Qiao, Wanli Ouyang, and Xiangyu Yue. Meta-transformer: A unified framework for multimodal learning. arXiv preprint arXiv:2307.10802, 2023. 3, 6, 7
- [37] Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and Christoph Feichtenhofer. Videoclip: Contrastive pre-training for zero-shot video-text understanding. arXiv preprint arXiv:2109.14084, 2021. 3
- [38] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Jilan Xu, Zun Wang, et al. Internvideo2: Scaling video foundation models for multimodal video understanding. arXiv preprint arXiv:2403.15377, 2024. 3, 7
- [39] Sihan Chen, Handong Li, Qunbo Wang, Zijia Zhao, Mingzhen Sun, Xinxin Zhu, and Jing Liu. Vast: A vision-audio-subtitle-text omni-modality foundation model and dataset. arXiv preprint arXiv:2305.18500, 2023. 3, 4, 6
- [40] Sihan Chen, Xingjian He, Longteng Guo, Xinxin Zhu, Weining Wang, Jinhui Tang, and Jing Liu. Valor: Vision-audio-language omni-perception pretraining model and dataset. arXiv preprint arXiv:2304.08345, 2023. 3, 7, 24
- [41] Nanyi Fei, Zhiwu Lu, Yizhao Gao, Guoxing Yang, Yuqi Huo, Jingyuan Wen, Haoyu Lu, Ruihua Song, Xin Gao, Tao Xiang, et al. Towards artificial general intelligence via a multimodal foundation model. Nature Communications, 13(1):3094, 2022. 4
- [42] Hongwei Xue, Tiankai Hang, Yanhong Zeng, Yuchong Sun, Bei Liu, Huan Yang, Jianlong Fu, and Baining Guo. Advancing high-resolution video-language representation with large-scale video transcriptions. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 4
- [43] Xiao Fu, Wei Yin, Mu Hu, Kaixuan Wang, Yuexin Ma, Ping Tan, Shaojie Shen, Dahua Lin, and Xiaoxiao Long. Geowizard: Unleashing the diffusion priors for 3d geometry estimation from a single image. arXiv preprint arXiv:2403.12013, 2024. 4

- [44] Ainaz Eftekhar, Alexander Sax, Jitendra Malik, and Amir Zamir. Omnidata: A scalable pipeline for making multi-task mid-level vision datasets from 3d scans. In ICCV, pages 10786–10796, 2021. 4
- [45] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR, 2021. 4
- [46] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 5, 8
- [47] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. 5
- [48] Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva: Exploring the limits of masked visual representation learning at scale. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19358–19369, 2023. 5
- [49] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2023. 7, 8
- [50] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. arXiv preprint arXiv:2311.17005, 2023. 7, 9
- [51] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020. 7
- [52] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, pages 248–255. Ieee, 2009. 7, 22
- [53] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, et al. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950, 2017. 7, 24
- [54] Pushmeet Kohli Nathan Silberman, Derek Hoiem and Rob Fergus. Indoor segmentation and support inference from rgbd images. In ECCV, 2012. 7
- [55] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In CVPR, pages 18995–19012, 2022. 7
- [56] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In CVPR, pages 5288–5296, 2016. 7
- [57] Ancong Wu, Wei-Shi Zheng, Hong-Xing Yu, Shaogang Gong, and Jianhuang Lai. Rgb-infrared cross-modality person re-identification. In ICCV, pages 5380–5389, 2017. 7
- [58] Jort F Gemmeke, Daniel PW Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R Channing Moore, Manoj Plakal, and Marvin Ritter. Audio set: An ontology and human-labeled dataset for audio events. In 2017 IEEE international conference on acoustics, speech and signal processing (ICASSP), pages 776–780. IEEE, 2017. 7, 24
- [59] Haixu Wu, Hang Zhou, Mingsheng Long, and Jianmin Wang. Interpretable weather forecasting for worldwide stations with a unified deep model. Nature Machine Intelligence, 5(6):602–611,

2023. 7, 23

- [60] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 7

- [61] Rohit Girdhar, Mannat Singh, Nikhila Ravi, Laurens van der Maaten, Armand Joulin, and Ishan Misra. Omnivore: A single model for many visual modalities. In CVPR, pages 16102–16112,

2022. 7

- [62] Sanyuan Chen, Yu Wu, Chengyi Wang, Shujie Liu, Daniel Tompkins, Zhuo Chen, and Furu Wei. Beats: Audio pre-training with acoustic tokenizers. arXiv preprint arXiv:2212.09058,

2022. 7

- [63] Yiyuan Zhang, Sanyuan Zhao, Yuhao Kang, and Jianbing Shen. Modality synergy complement learning with cascaded aggregation for visible-infrared person re-identification. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XIV, pages 462–479. Springer, 2022. 7
- [64] Evangelos Kazakos, Arsha Nagrani, Andrew Zisserman, and Dima Damen. Slow-fast auditory streams for audio recognition. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 855–859. IEEE, 2021. 7
- [65] Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, and Tie-Yan Liu. Do transformers really perform badly for graph representation? Advances in neural information processing systems, 34:28877–28888, 2021. 7
- [66] Neetu Sigger, Tuan Thanh Nguyen, Gianluca Tozzi, Quoc-Tuan Vien, and Sinh Van Nguyen. Diffspectralnet: Unveiling the potential of diffusion models for hyperspectral image classification. arXiv preprint arXiv:2312.12441, 2023. 7
- [67] Inkit Padhi, Yair Schiff, Igor Melnyk, Mattia Rigotti, Youssef Mroueh, Pierre Dognin, Jerret Ross, Ravi Nair, and Erik Altman. Tabular transformers for modeling multivariate time series. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 3565–3569. IEEE, 2021. 7
- [68] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 7, 25
- [69] Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In ICCV, pages 2641–2649, 2015. 7, 8, 25
- [70] Konstantinos Drossos, Samuel Lipping, and Tuomas Virtanen. Clotho: An audio captioning dataset. In ICASSP, pages 736–740. IEEE, 2020. 7, 8, 23, 24
- [71] Chris Dongjoo Kim, Byeongchang Kim, Hyunmin Lee, and Gunhee Kim. Audiocaps: Generating captions for audios in the wild. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 119–132, 2019. 7, 23, 24
- [72] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In CVPR, pages 5288–5296, 2016. 7, 23, 24
- [73] Luowei Zhou, Chenliang Xu, and Jason Corso. Towards automatic learning of procedures from web instructional videos. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018. 7, 24
- [74] Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-Fang Wang, and William Yang Wang. Vatex: A large-scale, high-quality multilingual dataset for video-and-language research. In ICCV, pages 4581–4591, 2019. 7, 24
- [75] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In Proceedings of the IEEE international conference on computer vision, pages 5803–5812, 2017. 7, 24

- [76] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 9127–9134,

2019. 7, 25

- [77] Yuncheng Li, Yale Song, Liangliang Cao, Joel Tetreault, Larry Goldberg, Alejandro Jaimes, and Jiebo Luo. Tgif: A new dataset and benchmark on animated gif description. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4641–4650, 2016. 7
- [78] Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and motion. In Proceedings of the 25th ACM international conference on Multimedia, pages 1645–1653, 2017. 7, 24
- [79] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913, 2017. 7, 25
- [80] Guangyao Li, Yake Wei, Yapeng Tian, Chenliang Xu, Ji-Rong Wen, and Di Hu. Learning to answer questions in dynamic audio-visual scenarios. In CVPR, pages 19108–19118, 2022. 7, 24
- [81] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023. 7, 8
- [82] Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In International Conference on Machine Learning, pages 23318–23340. PMLR, 2022. 7
- [83] Weicheng Kuo, AJ Piergiovanni, Dahun Kim, Xiyang Luo, Ben Caine, Wei Li, Abhijit Ogale, Luowei Zhou, Andrew Dai, Zhifeng Chen, et al. Mammut: A simple architecture for joint learning for multimodal tasks. arXiv preprint arXiv:2303.16839, 2023. 7
- [84] Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, et al. Pali: A jointly-scaled multilingual language-image model. arXiv preprint arXiv:2209.06794, 2022. 7
- [85] Xinhao Mei, Chutong Meng, Haohe Liu, Qiuqiang Kong, Tom Ko, Chengqi Zhao, Mark D Plumbley, Yuexian Zou, and Wenwu Wang. Wavcaps: A chatgpt-assisted weakly-labelled audio captioning dataset for audio-language multimodal research. arXiv preprint arXiv:2303.17395,

2023. 7, 8

- [86] Linjie Li, Jie Lei, Zhe Gan, Licheng Yu, Yen-Chun Chen, Rohit Pillai, Yu Cheng, Luowei Zhou, Xin Eric Wang, William Yang Wang, et al. Value: A multi-task benchmark for videoand-language understanding evaluation. arXiv preprint arXiv:2106.04632, 2021. 7
- [87] Dohwan Ko, Joonmyung Choi, Hyeong Kyu Choi, Kyoung-Woon On, Byungseok Roh, and Hyunwoo J Kim. Meltr: Meta loss transformer for learning to fine-tune video foundation models. arXiv preprint arXiv:2303.13009, 2023. 7
- [88] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, 2019. 8
- [89] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In CVPR, pages 6904–6913, 2017. 8
- [90] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In CVPR, 2019. 8

- [91] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In CVPR, pages 8317–8326,

2019. 8

- [92] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. NeurIPS, 2022. 8
- [93] Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In CVPR, pages 3608–3617, 2018. 8
- [94] Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra, Devi Parikh, Stefan Lee, and Peter Anderson. nocaps: novel object captioning at scale. In ICCV, 2019. 8
- [95] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023. 8
- [96] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023. 8
- [97] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023. 8
- [98] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seedbench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023. 8
- [99] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022. 8
- [100] Vicuna. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. https: //vicuna.lmsys.org/, 2023. 8
- [101] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 8
- [102] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 8
- [103] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning, 2023. 8
- [104] Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M Rush, Douwe Kiela, et al. Obelisc: An open web-scale filtered dataset of interleaved image-text documents. arXiv preprint arXiv:2306.16527, 2023. 8
- [105] Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, et al. Llama-adapter v2: Parameter-efficient visual instruction model. arXiv preprint arXiv:2304.15010, 2023. 8, 9
- [106] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023. 8

- [107] Jiaming Han, Renrui Zhang, Wenqi Shao, Peng Gao, Peng Xu, Han Xiao, Kaipeng Zhang, Chris Liu, Song Wen, Ziyu Guo, et al. Imagebind-llm: Multi-modality instruction tuning. arXiv preprint arXiv:2309.03905, 2023. 8
- [108] Zijia Zhao, Longteng Guo, Tongtian Yue, Sihan Chen, Shuai Shao, Xinxin Zhu, Zehuan Yuan, and Jing Liu. Chatbridge: Bridging modalities with large language model as a language catalyst. arXiv preprint arXiv:2305.16103, 2023. 8
- [109] Seungwhan Moon, Andrea Madotto, Zhaojiang Lin, Tushar Nagarajan, Matt Smith, Shashank Jain, Chun-Fu Yeh, Prakash Murugesan, Peyman Heidari, Yue Liu, et al. Anymal: An efficient and scalable any-modality augmented language model. arXiv preprint arXiv:2309.16058,

2023. 8

- [110] Samuel Lipping, Parthasaarathy Sudarsanam, Konstantinos Drossos, and Tuomas Virtanen. Clotho-aqa: A crowdsourced dataset for audio question answering. In 2022 30th European Signal Processing Conference (EUSIPCO), pages 1140–1144. IEEE, 2022. 8
- [111] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023. 8, 9
- [112] Zhongjie Ye, Yuqing Wang, Helin Wang, Dongchao Yang, and Yuexian Zou. Featurecut: An adaptive data augmentation for automated audio captioning. In 2022 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), pages 313–318. IEEE, 2022. 8
- [113] Guangyao Li, Yixin Xu, and Di Hu. Multi-scale attention for audio question answering. arXiv preprint arXiv:2305.17993, 2023. 8
- [114] Soham Deshmukh, Benjamin Elizalde, Rita Singh, and Huaming Wang. Pengi: An audio language model for audio tasks. arXiv preprint arXiv:2305.11834, 2023. 8
- [115] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. 8
- [116] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 8, 9
- [117] Ruyang Liu, Chen Li, Yixiao Ge, Ying Shan, Thomas H Li, and Ge Li. One for all: Video conversation is feasible without video instruction tuning. arXiv preprint arXiv:2309.15785,

2023. 8

- [118] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. arXiv preprint arXiv:2311.17043, 2023. 8, 9
- [119] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Zero-shot video question answering via frozen bidirectional language models. NeurIPS, 35:124–141, 2022. 9
- [120] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. 9
- [121] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In Proceedings of the IEEE/CVF international conference on computer vision, pages 568–578, 2021. 22
- [122] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021.
- [123] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. arXiv preprint arXiv:2201.03545, 2022. 22

- [124] Tawsifur Rahman, Amith Khandakar, Muhammad Abdul Kadir, Khandaker Rejaul Islam, Khandakar F Islam, Rashid Mazhar, Tahir Hamid, Mohammad Tariqul Islam, Saad Kashem, Zaid Bin Mahbub, et al. Reliable tuberculosis detection using chest x-ray with deep learning, segmentation and visualization. Ieee Access, 8:191586–191601, 2020. 23
- [125] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Densecaptioning events in videos. In Proceedings of the IEEE international conference on computer vision, pages 706–715, 2017. 24
- [126] Anna Rohrbach, Atousa Torabi, Marcus Rohrbach, Niket Tandon, Christopher Pal, Hugo Larochelle, Aaron Courville, and Bernt Schiele. Movie description. International Journal of Computer Vision, 123:94–120, 2017. 24

### Appendix

- A Summary The appendix is organized as follows:

- • § B detailed training and evaluation settings of our models including hyper-parameters regarding models and optimizers.
- • § C presents a comprehensive introduction on the datasets we use for evaluation and their corresponding metrics.

- B Training Configuration

##### B.1 Pretraining Settings

We detail the specific pretraining configurations of MiCo, focusing on the multi-dataset joint training corpora, the dataset mix ratios for each corpus, and the learning objectives for each corpus. To improve data quality, we employed a trained vision captioner to generate new captions for the CC4M datasets, replacing the original captions. Although MiCo has only been trained for 300,000 steps, it has already demonstrated outstanding performance on various downstream tasks. We anticipate that further increasing the number of training steps will significantly enhance the model’s capabilities.

The pretraining of MiCo involves a combination of different datasets, each contributing uniquely to the model’s learning process. The model, with a parameter size of 1.0 billion and a sample size of 334 million, utilizes a diverse training corpus to achieve its results.

- 1. VAST-27M: This dataset contributes 324 million samples to the training process. With a batch size of 2048, the model undergoes 160,000 steps, completing one epoch.
- 2. VALOR-1M: In this dataset, 1 million samples are used with a batch size of 1024. The training spans 70,000 steps, which equates to approximately 71.7 epochs.
- 3. WavCaps, CC4M, and WebVid-2.5M: These datasets are combined, contributing 9 million samples in total. The batch size for this combined dataset is 1024, and the model is trained over 70,000 steps, resulting in 8.0 epochs.

The careful selection and combination of these datasets, along with the application of new, highquality captions for the CC4M datasets, enhance the training efficiency and the quality of the learned representations.

##### B.2 Fine-tuning Settings

We detail the downstream task finetuning settings, specifying the learning rate, batch size, epoch, training objectives, and resolution. The configurations also include the number of sampled video frames or audio clips used in training and testing phases. Here are the comprehensive settings:

Retrieval Tasks (RET)

- • Image-Text Modality

- – MSCOCO: Learning rate of 1e-5, batch size of 256, 5 epochs, with the objective for retrieval, and a resolution of 384.
- – Flickr: Learning rate of 1e-5, batch size of 256, 5 epochs, with the objective for retrieval, and a resolution of 384.

- • Audio-Text Modality (A-T)

- – ClothoV1/V2: Learning rate of 2e-5, batch size of 64, 10 epochs, with the objective for retrieval, using 3 audio clips during both training and testing.
- – AudioCaps: Learning rate of 2e-5, batch size of 64, 10 epochs, with the objective for retrieval, using 1 audio clip during both training and testing.

##### • Multi-modal (MM)

- – MSRVTT: Learning rate of 2e-5, batch size of 64, 3.6 epochs, with the objective for retrieval, using 8 video frames during training and 16 during testing, with a resolution of 224.
- – YouCook2: Learning rate of 3e-5, batch size of 64, 30 epochs, with the objective for retrieval, using 8 video frames during training and 16 during testing, with a resolution of 224.
- – VALOR-32K: Learning rate of 2e-5, batch size of 64, 10 epochs, with the objective for retrieval, using 8 video frames during both training and testing, with a resolution of 224.
- – VATEX: Learning rate of 2e-5, batch size of 64, 2.5 epochs, with the objective for retrieval, using 8 video frames during training and 16 during testing, with a resolution of 224.
- – DiDeMo: Learning rate of 2e-5, batch size of 64, 40 epochs, with the objective for retrieval, using 8 video frames during training and 32 during testing, and 2 audio clips during both training and testing, with a resolution of 224.
- – ANET: Learning rate of 2e-5, batch size of 64, 20 epochs, with the objective for retrieval, using 8 video frames during training and 32 during testing, and 2 audio clips during both training and testing, with a resolution of 224.

Captioning Tasks (CAP)

- • Image-Text Modality

- – MSCOCO: Learning rate of 1e-5, batch size of 64, 5 epochs, with the objective for caption, and a resolution of 480.
- – MSCOCO(SCST): Learning rate of 2.5e-6, batch size of 64, 2.5 epochs, with the objective for caption, and a resolution of 480.

- • Audio-Text Modality (A-T)

- – ClothoV1/V2: Learning rate of 2e-5, batch size of 64, 10 epochs, with the objective for caption, using 3 audio clips during both training and testing.
- – AudioCaps: Learning rate of 2e-5, batch size of 64, 10 epochs, with the objective for caption, using 1 audio clip during both training and testing.

- • Multi-modal (MM)

- – MSRVTT: Learning rate of 2e-5, batch size of 128, 10 epochs, with the objective for caption, using 8 video frames during both training and testing, with a resolution of 224.
- – YouCook2: Learning rate of 3e-5, batch size of 64, 30 epochs, with the objective for caption, using 8 video frames during training and 16 during testing, with a resolution of 224.
- – VALOR-32K: Learning rate of 1e-5, batch size of 64, 10 epochs, with the objective for caption, using 8 video frames during training and 12 during testing, with a resolution of 224.

Question Answering Tasks (QA)

- • Visual-Text Modality (Vis)

- – MSVD-QA: Learning rate of 1e-5, batch size of 64, 10 epochs, with the objective for QA, using 8 video frames during training and 14 during testing, with a resolution of 224.
- – TGIF-FrameQA: Learning rate of 2e-5, batch size of 64, 10 epochs, with the objective for QA, using 4 video frames during both training and testing, with a resolution of 224.
- – VQAv2: Learning rate of 2e-5, batch size of 128, 20 epochs, with the objective for QA, and a resolution of 384.

- • Multi-modal (MM)

– MSRVTT-QA: Learning rate of 2e-5, batch size of 64, 4.5 epochs, with the objective for QA, using 8 video frames and 1 audio clip during both training and testing, with a resolution of 224.

Table 8: Detailed training configurations of MiCo for multimodal learning. Apart from the configurations shown in the table, for image tasks, we use random left-right flipping, random resized crop, color jitter of 0.4, Auto-augment, and no repeated augmentation for every model.

|settings<br><br>|Image ViT-L ViT-g<br><br>|Audio ViT-L ViT-g<br><br>|Video ViT-L ViT-g<br><br>|Depth & Normal Map ViT-L ViT-g|
|---|---|---|---|---|
| | | | | |
|Input Shape batch size optimizer LR LR schedule weight decay warmup epochs epochs<br><br>|224 224 4096 512 AdamW AdamW 4×10−3 5×10−5 cosine cosine<br><br>0.05 1×10−8<br><br>5 0 90 30|224 224<br><br>4096 512 AdamW AdamW 4×10−3 5×10−5 cosine cosine<br><br>0.05 1×10−8<br><br>5 0 90 30<br><br>|224 224 4096 512 AdamW AdamW 4×10−3 5×10−5 cosine cosine<br><br>0.05 1×10−8<br><br>5 0 90 20|224 224<br><br>4096 512 AdamW AdamW 4×10−3 5×10−5 cosine cosine<br><br>0.05 1×10−8<br><br>5 0 90 20<br><br>|
|mixup alpha cutmix alpha erasing prob.<br><br>|0.8 0.0<br>1.0 0.0 0.25 0.25<br>|0.8 0.0<br><br>1.0 0.0<br><br><br>0.25 0.25<br><br>|0.8 0.0<br><br>1.0 0.0 0.25 0.25<br><br><br>|0.8 0.0<br>1.0 0.0 0.25 0.25<br>|
|dropout rate<br><br>|0.1 0.2|0.1 0.2<br><br>|0.1 0.3|0.2 0.3|

- Algorithm 1 Multimodal Context Pretraining Algorithm, PyTorch-like

|def train(video_pixels=None, image_pixels=None, depth_pixels=None, audio_spectrograms=<br><br>None): # Get Mixed Data modal_inputs = [video_pixels, image_pixels, depth_pixels, audio_spectrograms] modal_captions = [video_captions, image_captions, depth_captions, audio_captions]<br><br># Extract Features modal_feats = [self.encoder(modal) for modal in modal_inputs if modal is not None] multimodal_feats = torch.cat(modal_feats) concatenated_captions = ’’.join(modal_captions) text_feats = self.text_encoder(concatenated_captions)<br><br># Losses contra_loss = Contrasive_Loss(multimodal_feats, text_feats) matching_loss = Matching_Loss(modal_captions, multimodal_feats) gen_loss = Generation_Loss(modal_captions.mask(0.6), multimodal_feats)<br><br># Total Loss loss = contra_loss + matching_loss + gen_loss<br><br>return loss|
|---|

- – MUSIC-AVQA: Learning rate of 2e-5, batch size of 64, 20 epochs, with the objective for QA, using 8 video frames and 2 audio clips during both training and testing, with a resolution of 224.
- – ANET-QA: Learning rate of 2e-5, batch size of 64, 10 epochs, with the objective for QA, using 8 video frames during training and 16 during testing, and 2 audio clips during both training and testing, with a resolution of 224.

These settings have been optimized to balance efficiency and performance, even though most hyperparameters are not precisely tuned.

For evaluation purposes, we employ different strategies tailored to specific tasks:

- 1. Retrieval Tasks: All candidates are initially ranked using Omni-modal Contrastive Loss. Following this, the Top-50 candidates undergo a reranking process through the Omni-modal Matching Process.
- 2. Captioning Tasks: Beam search with a beam size of 3 is utilized to generate captions, ensuring a comprehensive exploration of possible outputs.
- 3. Question Answering (QA) Tasks: These are treated as open-ended generative problems. Questions are used as prefixes, and answers are generated without any constraints, allowing for flexible and contextually appropriate responses.

- Algorithm 2 Dataset Split Algorithm

|import pandas as pd from sklearn.model_selection import train_test_split<br><br># Assume ‘data‘ is a DataFrame containing the full dataset with columns [’category’, ’<br><br>vision_caption’, ’audio_caption’, ’depth’, ’normal’, ’subtitle’] # Adding an ’index’ column to keep track of the original indices data[’index’] = data.index<br><br># Define the sizes of each subset subset_sizes = [1e6, 1e7, 1.1e7, 3.34e7]<br><br># Function to create stratified samples def create_subset(data, size):<br><br>subset, _ = train_test_split(data, train_size=size, stratify=data[’category’],<br><br>random_state=42) return subset<br><br># Creating subsets subset_1M = create_subset(data, 1e6) subset_10M = create_subset(data, 1e7) subset_110M = create_subset(data, 1.1e7) subset_334M = create_subset(data, 3.34e7)<br><br># Reset index for each subset subset_1M.reset_index(drop=True, inplace=True) subset_10M.reset_index(drop=True, inplace=True) subset_110M.reset_index(drop=True, inplace=True) subset_334M.reset_index(drop=True, inplace=True)<br><br>|
|---|

For comparisons with state-of-the-art (SOTA) models and ablation studies, we use the following evaluation metrics: 1) Retrieval Tasks: Recall@1. 2) Captioning Tasks: CIDEr. 3) QA Tasks: Accuracy (Acc) These metrics provide a comprehensive assessment of the model’s performance across different types of tasks.

#### C Datasets and Metrics

Dataset Split To split the mix of datasets into subsets of 1M, 10M, 110M, and 334M video clips while preserving its diversity and quality, we employed a proportional stratified sampling method. Initially, the dataset, which spans over 15 categories (including music, gaming, education, entertainment, and animals) and includes vision, audio, depth, normal maps, and text modalities, was organized and labeled. Stratified random sampling was then used to ensure each subset accurately reflected the distribution of categories and modalities present in the full dataset. This method involved selecting samples proportionally from each category to maintain representative distributions. The vision and audio captions were also kept proportional in length and quantity, ensuring that each subset retained the comprehensive characteristics of the original dataset.

##### C.1 Single-modality Evaluation Details

Text. The MMLU (Massive Multitask Language Understanding) benchmark is designed to evaluate the multitask accuracy of language models across 57 diverse tasks, including subjects like mathematics, history, and biology. It assesses models’ abilities to generalize and apply knowledge in various domains, providing a comprehensive measure of text understanding and reasoning skills.

Image. We conduct experiments on ImageNet-1K [52], a dataset comprising approximately 1.3 million images across 1,000 categories. In line with common practices [121–123, 13], base-scale models are trained for 300 epochs. Large-scale models undergo pre-training on ImageNet-22K, which includes 14.2 million images, for 90 epochs, followed by fine-tuning on ImageNet-1K for an additional 20 epochs.

Thermal and Hyperspectral data understanding. We conduct experiments on infrared image recognition using the RegDB dataset, X-ray scan analysis with the Chest X-Ray dataset [124], and hyperspectral data recognition using the Indian Pine dataset4.

Depth. The NYU Depth Dataset (NYU-D) comprises RGB and depth image pairs captured from indoor scenes. It includes 1,449 densely labeled pairs for training and testing, along with over 400,000 unlabeled frames.

Audio. For audio recognition, Audioset-2M dataset comprises over 2 million human-labeled 10second audio clips drawn from YouTube videos. It covers a wide range of 527 sound event classes, providing a comprehensive resource for training and evaluating audio event detection and classification models.

Video. The Kinetics-700 dataset contains 700,000 video clips covering 700 human action classes, used for action recognition tasks. The MSR-VTT dataset includes 10,000 video clips paired with multiple textual descriptions, supporting video captioning, retrieval, and content understanding research.

Time-series. Global Weather Forecasting [59] includes global, regional, and Olympics data from NCEI and CMA, comprising hourly weather measurements from thousands of stations. Evaluation involved splitting data into training, validation, and test sets (7:1:2) using MSE and MAE metrics.

Graph. PCQM4M-LSC dataset is a large-scale collection of 4.4 million organic molecules, each with up to 23 heavy atoms and associated quantum-mechanical properties. Aimed at predicting molecular properties through machine learning, this dataset is highly relevant for applications in drug discovery and material science.

Tabular. The fraud dataset comprises transaction records, including features like transaction amount, location, time, and user information. It is designed for machine learning models to detect fraudulent activities. This dataset is crucial for developing and testing algorithms to enhance security in financial systems and reduce economic losses due to fraud.

IMU. The Ego4D dataset includes inertial measurement unit (IMU) data captured from wearable devices, providing detailed motion and orientation information. This dataset supports research in human activity recognition, augmented reality, and robotics, offering comprehensive insights into human movements and interactions with the environment.

##### C.2 Cross-modality Evaluation Details

We evaluated MiCo across several well-known downstream datasets, including MSRVTT, VATEX, YouCook2, VALOR-32K, MSVD, DiDeMo, ActivityNet Caption, TGIF, MUSIC-AVQA, Clotho, AudioCaps, MSCOCO, Flickr30K, and VQAv2. The specific train/validation/test splits for these benchmarks are detailed below:

#### Retrieval Tasks

##### Audio-Text Modality (A-T)

- • ClothoV1 [70]: This dataset includes 2,893 audio clips for training and 1,045 for validation. The corresponding captions number 14,465 for training and 5,225 for validation.
- • ClothoV2 [70]: Contains 3,839 audio clips for training and 1,045 for validation, with 19,195 captions for training and 5,225 for validation.
- • AudioCaps [71]: Comprises 49,291 audio clips for training, 428 for validation, and 816 for testing, along with 49,291 captions for training, 2,140 for validation, and 4,080 for testing.

##### Video-Text Modality (V-T)

• MSRVTT [72]: Comprises 10K video clips and 200K captions, spanning diverse topics such as human activities, sports, and natural landscapes. We evaluate text-to-video retrieval,

4https://github.com/danfenghong/IEEE_TGRS_SpectralFormer/blob/main/data/IndianPine. mat

video captioning, and video QA using this dataset. Contains 9,000 videos for training and 1,000 for testing, with 180,000 captions for training and 1,000 for testing.

- • YouCook2 [73]: Comprises 14K video clips extracted from 2K instructional cooking videos on YouTube. Each video features multiple actions performed by chefs, along with corresponding textual descriptions and temporal annotations. Includes 10,337 videos for training and 3,492 for validation, with matching captions.
- • VALOR-32K [40]: An audiovisual video-language benchmark containing 32K 10-second video clips sourced from AudioSet [58]. Each clip includes annotations with captions that describe both the visual and audio content. Consists of 25,000 videos for training, 3,500 for validation, and 3,500 for testing, each with corresponding captions.
- • DiDeMo [75]: Comprises 10K long-form videos sourced from Flickr, with each video annotated with four short sentences in temporal order. For this benchmark, we concatenate these short sentences and evaluate ’paragraph-to-video’ retrieval, using the official split. Features 8,394 videos for training, 1,065 for validation, and 1,003 for testing, along with their captions.
- • ActivityNet (ANET) [125]: Includes 20K long-form videos (average length of 180 seconds) from YouTube, accompanied by 100K captions. We evaluate text-to-video retrieval and video QA on this dataset. Comprises 10,009 videos for training and 4,917 for testing, with corresponding captions.
- • LSMDC [126]: Contains 101,046 videos for training, 7,408 for validation, and 1,000 for testing, with corresponding captions.

#### Captioning Tasks

##### Audio-Text Modality (A-T)

- • ClothoV1 [70]: This dataset includes 2,893 audio clips for training and 1,045 for validation. The corresponding captions number 14,465 for training and 5,225 for validation.
- • ClothoV2 [70]: Contains 3,839 audio clips for training and 1,045 for validation, with 19,195 captions for training and 5,225 for validation.
- • AudioCaps [71]: Comprises 49,838 audio clips for training, 495 for validation, and 975 for testing, along with 49,438 captions for training, 2,475 for validation, and 4,875 for testing.

##### Video-Text Modality (V-T)

- • MSRVTT [72]: Contains 6,513 videos for training, 497 for validation, and 2,990 for testing, with 130,260 captions for training, 9,940 for validation, and 59,800 for testing.
- • YouCook2 [73]: Includes 10,337 videos for training and 3,492 for validation, with matching captions.
- • VALOR-32K [40]: Consists of 25,000 videos for training, 3,500 for validation, and 3,500 for testing, each with corresponding captions.
- • VATEX [74]: Consists of 41,250 video clips sourced from the Kinetics-600 dataset [53], accompanied by 825,000 sentence-level descriptions. Contains 25,991 videos for training, 3,000 for validation, and 6,000 for testing, with 259,910 captions for training, 30,000 for validation, and 60,000 for testing.

#### Question Answering (QA) Tasks

##### Video-Text Modality (V-T)

- • MSRVTT-QA [78]: Contains 6,513 videos for training, 497 for validation, and 2,990 for testing, with 158,581 QA pairs for training, 12,278 for validation, and 72,821 for testing.
- • MUSIC-AVQA [80]: An audiovisual video QA benchmark containing over 45K Q-A pairs, covering 33 different question templates across various modalities and question types. Includes 9,277 videos for training, 3,815 for validation, and 6,399 for testing, with 32,087 QA pairs for training, 4,595 for validation, and 9,185 for testing.

- • ANET-QA [76]: Comprises 3,200 videos for training, 1,800 for validation, and 800 for testing, with 32,000 QA pairs for training, 18,000 for validation, and 8,000 for testing.

#### Image-Based Tasks

- • MSCOCO [68]: Comprises 123K images, each paired with 5 annotated captions. We evaluate text-to-image retrieval and image captioning on this dataset.
- • Flickr30K [69]: Contains 31K images, each paired with five descriptive captions. This dataset is widely used for evaluating image captioning and text-to-image retrieval tasks.

#### Visual Question Answering

• VQAv2 [79]: A large-scale Visual Question Answering dataset comprising over 265K images and 1.1M questions, designed to improve the balance of answer types per question. This dataset is used to evaluate models’ abilities to understand and reason about visual content by providing accurate answers to questions based on the images.

