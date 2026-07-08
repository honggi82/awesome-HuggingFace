## mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration

Qinghao Ye∗ Haiyang Xu∗ Jiabo Ye∗ Ming Yan† Anwen Hu Haowei Liu Qi Qian Ji Zhang Fei Huang Jingren Zhou Alibaba Group

{yeqinghao.yqh, shuofeng.xhy, yejiabo.yjb, ym119608}@alibaba-inc.com Code & Demo & Models: https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2

# arXiv:2311.04257v2[cs.CL]9Nov2023

[Figure 1]

### Abstract

Multi-modal Large Language Models (MLLMs) have demonstrated impressive instruction abilities across various open-ended tasks. However, previous methods primarily focus on enhancing multi-modal capabilities. In this work, we introduce a versatile multi-modal large language model, mPLUG-Owl2, which effectively leverages modality collaborationtoimproveperformancein bothtextand multi-modal tasks. mPLUG-Owl2 utilizes a modularized network design, with the language decoder acting as a universal interface for managing different modalities. Specifically, mPLUG-Owl2 incorporates shared functional modules to facilitate modalitycollaborationandintroducesamodality-adaptivemodule that preserves modality-specific features. Extensive experiments reveal that mPLUG-Owl2 is capable of generalizing both text tasks and multi-modal tasks and achieving state-ofthe-art performances with a single generic model. Notably, mPLUG-Owl2 is the first MLLM model that demonstrates the modality collaboration phenomenon in both pure-text and multi-modal scenarios, setting a pioneering path in the development of future multi-modal foundation models.

[Figure 2]

mPLUG-Owl2

BLIP-2, MiniGPT-4, LLaVA, etc.

(Modality-Adaptive Language Decoder)

(Vanilla Language Decoder)

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

1 1 1 0 0 0 0 0 0

Feed Forward Network

Feed Forward Network Modality-Adaptive Module

Attention Module

[Figure 7]

[Figure 8]

Modality Collaboration

Modality Interference

### 1. Introduction

(a) (b)

- • Text Instruction Understanding
- • Visual Concept Understanding

- • Text Instruction Understanding
- • Visual Concept Understanding

LargeLanguageModels(LLMs)suchasGPT-3[6], LLaMA [57, 58], and GPT-4 [46] have garnered significant attention due to their exceptional generalization abilities in text understanding and generation. To facilitate the visionlanguage applications, GPT-4V1 [45] has recently demonstrated impressive multi-modal capabilities in diverse tasks, e.g., description , question answering, etc., sparking interest among researchers in the potential convergence of the vision-language field. This has led to the emergence of a

Figure 1. An overall performance comparison between mPLUGOwl2 and existing MLLMs and difference between existing MLLMs and our proposed model. (a) Previous approaches utilize a standard language decoder (i.e., LLM) to manage different types of instructions, leading to modality interference and performance degradation. (b) We introduce mPLUG-Owl2, which uses a modality-adaptive language decoder to handle different modalities within distinct modules while sharing some parameters for modality collaboration. This approach mitigates the issue of modality interference.

∗Equal contribution †Corresponding author 1https://openai.com/research/gpt-4v-system-card

group of Multi-modal Large Language Models (MLLMs) [5, 15, 31, 38, 65, 66, 68, 75], which aim to enhance LLMs with the ability to understand and handle visual problems.

Previous studies [27, 63] in multi-modal learning suggest that different modalities can effectively collaborate, thereby enhancing the performance of both text and multi-modal tasks simultaneously. However, MLLMs is a unified model that supports different modalities and tasks without finetuning for specific tasks. Recent works utilize cross-modal alignment modules (e.g., Q-former [15, 31, 75] and linear layer [10, 38]) to map visual features from the vision encoder intothefrozenLLMstocarryoutmulti-modaltasksbyleveraging preserved language capabilities. This strategy, unfortunately, restricts the potential of modality collaboration. As a result, some researchers [38, 68] opt to fine-tune LLMs during multi-modal instruction tuning. While fine-tuning significantly improves multi-modal tasks, it risks weakening text task performance [16]. As illustrated in Figure 1, the challenge of modality collaboration in MLLMs is from applying asingle moduleto balance the gain of modality collaboration and modality interference, where modalities may interfere with each other on a large number of instruction datasets across multiple modalities.

To mitigate this challenge, we present a new generalpurpose multi-modal foundation model, mPLUG-Owl2, in thiswork. Ourmodelfeaturesamodularizednetworkdesign that takes both modality collaboration and modality interference into account, using the language decoder as a universal interface for managing multi-modal signals. Specifically, mPLUG-Owl2 incorporates certain shared functional modules to promote modality collaboration and introduces a modality-adaptive module that serves as a pivot across different modalities. Therefore, vision and language modalities are projected into a shared semantic space for crossmodality interaction, while the proposed module helps preserve modality-specific features. With our novel architecture, modalities with varying information densities are shielded from modality interference due to the modalityadaptive module and can collaborate effectively in capturing shared information. Furthermore, we introduce an innovative two-stage training paradigm that consists of visionlanguage pre-training and joint vision-language instruction tuning. This paradigm trains the vision encoder across two stages, enabling it to capture both low-level and high-level semantic visual information more effectively.

Extensive experiments illustrate the effectiveness and generalization abilities of mPLUG-Owl2, which achieves state-of-the-art performance on 8 classic vision-language benchmarks using a single generic model. Furthermore, it either first or second in performance on 5 recent zeroshot multi-modal benchmarks, underscoring its adaptability and proficiency in multi-modal instruction comprehension and generation. In addition to its cutting-edge performance

in multi-modal tasks, mPLUG-Owl2 also achieves state-ofthe-art results on multiple pure-text benchmarks. Moreover, we provide in-depth analysis to demonstrate and validate the impact of modality collaboration through our proposed modality-adaptive module, especially in enhancing text tasks, including understanding, knowledge, and reasoning. Finally, comprehensive ablation studies validate the effectiveness of the proposed MLLM training paradigm, which can help inspire the development of future multimodal foundation models.

### 2. Related Work

Multi-Modal Large Language Foundation Models. The successful application of Large Language Models (LLMs) has paved the way for developing several approaches aiming to augment the perceptual capacities of LLMs with additional modalities, all within a unified model. There are three primary methods for constructing multi-modal large language foundational models, each showing promise for robust zero-shot generalization capabilities in the visionlanguage domain. For instance, Flamingo [2] is a forerunner in this area, using a frozen vision encoder and a large language model equipped with gated cross-attention for crossmodality alignment. In contrast, PaLM-E [16] integrates extracted visual features directly through linear layers into the pre-trained PaLM [12] model, which boasts 520 billion parameters, thereby leading to robust performance across numerous real-world applications. This approach has been broadly adopted by models such as LLaVA [38], Shikra [10], etc. One significant limitation of this method, however, is the creation of lengthy visual sequences. To address this, BLIP-2 [31], drawing inspiration from DETR [8], developed a Q-former to reduce the sequence length of visual features efficiently. This design has been mirrored by Kosmos-1 [23], mPLUG-Owl [68], and MiniGPT-4 [75]. Nevertheless, it should be noted that these methods directly align the visual features with the LLMs, treating vision and language signals as equivalent, thereby overlooking the unique granularities between vision and language modalities. To alleviate this problem, we introduce modality-adaptive module. Our proposed model leads to superior performance in both zeroshot and fine-tuning evaluation settings in terms of both image and video.

Instruction Tuning with MLLMs. Instruction tuning optimizes pre-trained large language models to comprehend and adhere to natural instructions, thereby enhancing their ability to generalize unseen tasks in a zero-shot manner. Researchers often employ models such as ChatGPT and GPT-4 [46] to generate diverse and expansive instruction datasets, including those like Alpaca [56], ShareGPT [1], and WizardLM [61]. As multi-modal large language models emerge, research communities are beginning to create high-quality, diverse multi-modal datasets. For instance, MiniGPT-4 [75] utilizes GPT-3.5 to rephrase captions generated by pre-

trained models. Concurrently, LLaVA [38], SVIT [72], and LRV-Instruction [36] take advantage of image annotations, such as bounding boxes of objects, image captions, and region descriptions, to prompt GPT-4 to generate instructions and responses using self-instruction methods. Models such as mPLUG-Owl [68] and LLaVA-1.5 [37] further advance this area by undergoing joint training with language-only and vision-and-language instruction data, thereby mitigating the risk of catastrophic forgetting of language knowledge. Rather than merely preventing this phenomenon of catastrophic forgetting, mPLUG-Owl2, with the help of the modality-adaptive module, can gain from the collaborative efforts of modalities by being jointly trained with languageonly and multi-modal instruction data, thus enhancing both multi-modal and language-only performance.

### 3. Methodology

- 3.1. Overview

Figure 2 (a) sketches the overview of the mPLUG-Owl2. Specifically, our model comprises a vision encoder, a visual abstractor, a text embedding layer, and a language decoder. Notably, the standard implementation of the text embedding layer and language decoder involves the use of a large language model, such as GPT [6] or LLaMA [57]. We first briefly introduce our model’s architecture in Section

- 3.2. Furthermore, we handle different types of modalities by introducing the modality-adaptive module in Section
- 3.3. Lastly, we introduce the training paradigm for training mPLUG-Owl2 with modality collaboration in Section 3.4.

- 3.2. Model Architecture

As depicted in Figure 2, our model, referred to as mPLUGOwl2, is composed of three main components: a fundamental vision encoder [48], a visual abstractor, and a language decoder. Specifically, we utilize ViT-L/14 as the vision encoder and LLaMA-2-7B [58] as the language decoder. The vision encoder processes an input image with an H × W resolution and produces a sequence of 14H × W14 tokens. These visual token features are then combined with text token embeddings and fed into the language decoder that serves as a universal interface that converts various vision-language tasks into text-generation tasks. However, with the increase in image resolution, the encoded visual token sequences can exponentially lengthen. Additionally, the presence of abundant redundancy in the images (e.g., background, similar patches) leads to computational waste and introduces considerable noise. To address this, we propose a visual abstractor equipped with a fixed set of learnable queries to extract higher semantic features from images. Specifically, we feed the extracted visual token sequence I = [I1,I2,··· ,IP] ∈ RP×d and a fixed number of K learnable queries Q ∈ RK×d into the visual abstractor. Here, P = 14H × W14 represents the number of visual patches,

and D is the hidden dimension. The visual abstractor consists of a series of visual abstractor layers. In the i-th layer of the visual abstractor, the compressed visual representations Vi+1 are computed as follows:

Ci = Attn(Vi,[I;Vi],[I;Vi]), (1) Vi+1 = SwiGLU(CiW1)W2. (2)

Here, Attn(·,·,·) represents the self-attention operation, while W1 ∈ Rd×d

′×d are learnable parameters. The function SwiGLU(···) refers to the SwiGLU activation function [51]. We designate V0 = Q to initiate the process. Moreover, to augment the fine-grained perception ability, we integrate sinusoidal positional embeddings with the image feature I and Vi, thereby preserving positional information, which has been proven essential in [8]. Hence, the computation required by the language decoder decreases from O((P +L)2) to O((K +L)2), significantly reducing computational load when P ≫ K, particularly in scenarios involving multiple images and when the text length L is relatively short. Once the compressed visual feature is obtained, it is concatenated with text token embeddings and then processed by the language decoder to generate the prediction.

′

and W2 ∈ Rd

#### 3.3. Modality-Adaptive Module

Prior approaches [15, 38, 68, 75] typically attempt to align visual features with language features by projecting image features into the language semantic space. However, this strategy can cause a mismatch in granularity , where image features often contain fruitful semantic information compared to the discrete semantic information within text embedding features. Those methods disregard the unique characteristics of visual and textual information, thus potentially limiting the model’s performance. To this end, we propose a new approach, namely, the Modality-Adaptive Module (MAM), which decouples vision-language representations by projecting visual features and language features into a shared semantic space while preserving the distinctive properties of each modality.

Formally, given a vision-language sequence X ∈ R(L

V +LT)×d and modality indicators M ∈ {0,1}(Lv+LT), we first define modality separated operation ϕ as:

ϕ(X,M,m) = X ⊙ {M=m}, (3)

where m ∈ {0,1} is the type of modalities (i.e., vision or language). Given the previous layer’s output vectors Hl−1,l ∈ [1,L], where L is the number of language decoder layers, we first normalized different modalities into the same magnitude as follows:

H˜l−1 = LNV (ϕ(Hl−1, M, 0)) + LNT(ϕ(Hl−1, M, 1)), (4)

Cross-Modality Features

This meme is funny because it is an image of a sleeping dog with the caption "MONDAY. JUST...MONDAY." The humor comes from the juxtaposition of the dog's peaceful, relaxed state, emphasizing the reluctance or disdain for Monday.

[Figure 9]

Language-Decoder

Norm0 Norm1

 

| | |
|---|---|
| | |

[Figure 10]

[Figure 11]

Visual Abstractor

Text

Language Decoder

Linear

[Figure 12]

Feed Forward Network (FFN)

[Figure 13]

MatMul

Visual Encoder

Image

Softmax

Stage-2: Joint Instruction Tuning

Modality-Adaptive Module

MatMul & Scale

[Figure 14]

[Figure 15]

Language-Decoder

x L

!!" !!!

!"" !"! !#

   

Visual Abstractor Text Embedding

[Figure 16]

[Figure 17]

Visual Abstractor

Text

Norm0 Norm1

ExplainTextWhyInputthis meme is funny.

…

Vision Encoder

 

[Figure 18]

[Figure 19]

[Figure 20]

Modality Indicator

[Figure 21]

Visual Encoder

Image

…

[Figure 22]

[Figure 23]

[Figure 24]

Learnable Image Queries

Text

Multi-Modal Inputs

Stage-1: Pre-training

(a) (b) (c)

Modality-Adaptive Module

- Figure 2. Illustration of the proposed mPLUG-Owl2 and its training paradigm. (a) An overview of mPLUG-Owl2, which consists of a vision encoder, visual abstractor, text embedding layer, and a language decoder. (b) Details of the proposed modality-adaptive module, which takes multi-modal inputs and employs different parameters to project various modalities into a shared semantic space for relational learning while preserving modality-specific features, thereby enabling modality collaboration. (c) The training paradigm of mPLUG-Owl2 involves first pre-training the visual-related modules, including the vision encoder and visual abstractor. Simultaneously, newly added parameters in the language decoder are also learned during the pre-training stage. During the instruction tuning stage, both language instructions and multi-modal instructions are used to jointly train the entire model.

where LNV and LNT are layer normalization [4] for visual features and language features respectively. Then, we reformulate the self-attention operation by leveraging separated linear projection layers for key projection matrix and value projection matrix while preserving query projection matrix shared as follows:

characteristics while achieving modality collaboration via the proposed modality-adaptive module.

#### 3.4. Training Paradigm

As depicted in Figure 2 (c), we employ a two-stage approach in training mPLUG-Owl2, comprising pre-training and visual instruction tuning similar to [38, 68], which aims to alignthepre-trainedvisionencoderandlanguagemodelduring the pre-training phase, and then fine-tune the language model with language modeling loss during the instruction tuning phase. However, we find that simply freezing a pretrained vision encoder and training a vision-language projector to align visual data with language models can limit their capacity to interpret complex visual information, such as scene text and visual knowledge. To address the issue, we make the vision encoder trainable throughout both the pre-training and instruction tuning stages. This strategy allows the model to capture both low-level and high-level semantic visual information more effectively. Specifically, for the pre-training stage, we enable the vision encoder, visual abstractor, and a part of the modality-adaptive module to be trainable, while keeping the pre-trained language model frozen. Meanwhile, prior research in multi-modal learning [63] has indicated that significant enhancements can be achieved through the collaborative learning of unimodal and multi-modal sources. Based on this, we adopt a joint training approach by tuning the whole model during the instruction tuning stage, incorporating both text and

HlQ = H˜l−1WlQ, (5) HlK = ϕ(H˜l−1,M,0)WK

l , (6) HlV = ϕ(H˜l−1,M,0)WV

l + ϕ(H˜l−1,M,1)WK

0

1

l , (7) Cl = Softmax

l + ϕ(H˜l−1,M,1)WV

0

1

HlQHlK⊤

HlV , (8)

√

d

where WlQ,WK

l ∈ Rd×d are the learnable projection matrices, and Cl ∈ R(L

l ,WK

l ,WV

l ,WV

0

1

0

1

V +LT)×d is the context features of l-th layer. In this manner, we can calculate the similarities between these two modalities within a shared semantic space, while also preserving the unique characteristics of each modality through different value projection layers. Moreover, by decoupling the key and value projection matrix, we can avoid interference between the two modalities, particularly in relation to granularity mismatch. In a similar vein, we also aim to model these characteristics by using different layer normalization layers. Finally, in order to promote modality collaboration within the same feature space, we maintain a shared FFN for both modalities. As a consequence, the model is able to preserve modality

Image Caption General VQA General VQA (Zero-shot) Model Type Method #Params COCO

Flickr30K

VQAv2 OKVQA GQA VizWizQA TextVQA SciQA (IMG)

(Zero-Shot)

BLIP-2 [31] 8.2B - 74.9 65.0 45.9 41.0 19.6 42.5 61.0 InstructBLIP [15] 8.2B 102.2 82.4 - - 49.2 34.5 50.1† 60.5

Unified-IOXL [41] 2.9B 122.3 - 77.9 54.0 - 57.4‡ - PaLM-E-12B [16] 12B 135.0 - 76.2 55.5 - - - Shikra [10] 7.2B 117.5 73.9 77.4 47.2 - - - -

Generalists

LLaVA-1.5 [37] 7.2B - - 78.5 - 62.0 50.0 46.1/58.2† 66.8 Qwen-VL-Chat [5] 9.6B 131.9 81.0 78.2 56.6 57.5 38.9 61.5‡ 68.2 mPLUG-Owl2 8.2B 137.3 85.1 79.4 57.7 56.1 54.5 54.3/58.2† 68.7

GIT [59] 0.7B 114.8 49.6 78.6 - - 68.0 59.8 GIT2 [59] 5.1B 145.0 50.7 81.7 - - 71.0 59.8 PaLI-17B [11] 17B 149.1 - 84.3 64.5 - 71.6 58.8 -

Specialists

- Table 1. Performance comparison on image caption and visual question answering. For image caption, CIDEr is reported for evaluation, and accuracy is reported for VQA. Note that specialists are fine-tuned on each individual dataset. † denotes OCR inputs are utilized. ‡ indicates the model has trained on the dataset. We gray out those specialists’ methods which are individually fine-tuned on the dataset as well as those fine-tuned results of generalists.

|Method|Vision Encoder|Language Model<br><br>|MME|MMBench<br><br>|MM-Vet|SEED-Bench<br><br>|Q-Bench|
|---|---|---|---|---|---|---|---|
|BLIP-2 [31] MiniGPT-4 [75] LLaVA [38] mPLUG-Owl [68] InstructBLIP [15] LLaMA-Adapter-v2 [19] Otter [30] Qwen-VL-Chat [5] LLaVA-1.5 [37]<br><br>|ViT-g (1.3B) ViT-g (1.3B) ViT-L (0.3B) ViT-L (0.3B) ViT-g (1.3B) ViT-L (0.3B) ViT-L (0.3B) ViT-G (1.9B) ViT-L (0.3B)|Vicuna (7B) Vicuna (7B) Vicuna (7B) LLaMA (7B) Vicuna (7B) LLaMA (7B) LLaMA (7B) Qwen (7B) Vicuna (7B)<br><br>|1293.84 581.67 502.82 967.34 1212.82 1328.40 1292.26 1487.58 1510.70<br><br>|23.0 36.2 46.6 36.0 39.5 48.3 60.6 64.3<br><br>|22.4 22.1 28.1 26.2 31.4 24.6 30.5|46.4 42.8 33.5 34.0 53.4 32.7 32.9 58.2 58.6<br><br>|54.7 58.9 55.8 58.1 47.2 61.6 60.7<br><br>|
|mPLUG-Owl2<br><br>|ViT-L (0.3B)|LLaMA (7B)<br><br>|1450.19<br><br>|64.5<br><br>|36.2|57.8<br><br>|62.9|

- Table 2. Zero-shot multi-modal evaluation on multi-modal benchmarks including MME [17], MMBench [39], MM-Vet [70], SEEDBench [29], and Q-Bench [60]. The overall scores are reported for evaluation. For MMBench and Q-Bench, we report test results.

multi-modal instructions. This methodology enhances the model’s comprehension of visual concepts embedded within the text by the multi-modal instructions. Concurrently, the text instruction data augments the model’s understanding of intricate natural instructions, thereby ensuring the preservation of its linguistic capabilities.

### 4. Experiments

#### 4.1. Implementation

Data sets mPLUG-Owl2 is first pre-trained on image-text pairs and fine-tunes on mono-modal and multi-modal instructiondata. Forpre-trainingdata, werandomlypickabout 400 million image-text pairs from five public datasets: Conceptual Captions (CC3M/CC12M) [9], COCO [35], Laionen [49], COYO [7], DataComp [18]. For instruction data, we collect 5 types of datasets including 1) image captioning (i.e., TextCaps [53], COCO [35]); 2) image question answering (i.e., VQAv2 [21], OKVQA [43], OCR-VQA [44], GQA [24], and A-OKVQA [50]); 3) region-aware QA (i.e., RefCOCO [69], VisualGenome [26]); 4) multi-modal instruct data (i.e., LLaVA-instruct-150K [38]); 5) text-only instruct

data (i.e., ShareGPT-80K [1], SlimOrca [34]). Details can be found in the Appendix.

Training Settings We pre-train the model for 42,500 iterations with a batch size 8,192 for about 348 million imagetext pairs. Since we adopt the language modeling loss, the large batch size can be easily achieved by the gradient accumulation technique. mPLUG-Owl2 adopts ViT-L [48] with patch size 14 × 14 and pre-trained at resolution 224 × 224. We use the same data augmentation in BLIP2 [31], including random resized cropping, and horizontal flipping with a probability of 0.5. The number of layers in the visual abstractor is set to 6 and it is randomly initialized. The number of learnable queries is set to 64. For the language model, LLaMA-2 [58] is employed for handling multi-modal features with 7B parameters, and the parameters of modality-adaptive modules are initialized from the language model. We use the AdamW [40] optimizer with β1 = 0.9, β2 = 0.98 and ϵ =1e-6 for optimization. The cosine learning rate decay scheduler with a peak learning rate of 1e-4 and with warmup steps 1k. For the learning rate of the vision encoder, we employ layer-wise learning rate decay with a factor of 0.9 to retain the low-level visual

representation. For the instruction tuning stage, we train the whole model for 1 epoch with a learning rate of 2e-5 and batch size 256. Besides, we increase the resolution from 224 × 224 to 448 × 448. The layer-wise learning rate decay is also employed which is crucial for retaining good visual representation in our experiments.

#### 4.2. Main Results

Image Caption and Visual Question Answering. We assess mPLUG-Owl2 using a wide range of academic benchmarks for evaluating vision-language models. Our evaluation includes eight popular benchmarks, as summarized in Table 1. As the results show, our mPLUG-Owl2 surpasses previous generalist models in both captioning and question answering tasks. Specifically, mPLUG-Owl2 achieves stateof-the-art performance on the Flickr30K datasets, even compared with models with more powerful backbones (e.g., Qwen-VL-Chat [5] and InstructBLIP [15]). Moreover, mPLUG-Owl2 exhibits distinct advantages in visual question answering, especially in OCR-free scenarios, where mPLUG-Owl2 achieves 54.3% accuracy on the TextVQA dataset in a zero-shot manner, demonstrating the benefits of our training strategy. Also worth noting is that mPLUGOwl2showsstrongzero-shotperformanceontheScienceQA (Image Set) and VizWizQA datasets.

MLLM-oriented Multi-modal Benchmarks. Given the robust zero-shot capabilities of Multi-Modal Language Models (MLLMs), traditional evaluation metrics often fall short in providing a detailed ability assessment. This problem is further exacerbated by their inability to match the given answer accurately, leading to significant robustness issues. To address these challenges, research communities have introduced a series of benchmarks including MME [17], MMBench [39], MM-Vet [70], SEED-Bench [29], and Q-Bench [60]. These benchmarks systematically structure and evaluate complex multi-modal tasks. We applied our model, in a zero-shot manner, to five recently popular multimodal benchmarks. For a fair comparison, we select models with similar language model sizes, particularly those from the LLaMA family, and detail their differences in the vision encoder. The results of our evaluation are listed in Table 2. In the table, mPLUG-Owl2 achieves higher zero-shot performance in terms of MMBench, MM-Vet, and Q-Bench. Conversely, the performance on MME is lower because of the limited number of test samples in MME, which could potentially lead to sensitive fluctuations in performance. Particularly, it exhibits significant improvement on Q-Bench, a benchmark for examining the low-level visual perception of MLLMs. This improvement occurs when applying a smaller visual backbone (i.e., ViT-L), leading to enhanced low-level visual perception. This demonstrates the effectiveness of

- our training strategy for training visual backbone.

|Method<br><br>|MMLU|BBH<br><br>|AGIEval<br><br>|ARC-c<br><br>|ARC-e|
|---|---|---|---|---|---|
|LLaMA-2 [58] WizardLM [61] LLaMA-2-Chat [58] Vicuna-v1.5 [73]|46.8 38.1 46.2 51.1<br><br>|38.2 34.7 35.6 41.2<br><br>|21.8 23.2 28.5 21.2|40.3 47.5 54.9 56.6<br><br>|56.1 59.6 71.6 72.8|
|mPLUG-Owl2<br><br>|53.4|45.0|32.7<br><br>|65.8|79.9|

- Table 3. Performance on pure-text benchmarks of mPLUGOwl2 compared to LLaMA-2 (7B) family variants. We adopt 5-shot for MMLU and 0-shot for BBH, AGIEval, and ARC as [14].

Natural Language Understanding and Generation. Current MLLMs often outperform in various multi-modal downstream tasks by leveraging the power of large language models. Nevertheless, the intrinsic capabilities of these models often play a significant role in determining the performance of MLLMs, an aspect that has often been overlooked in prior multi-modal language model studies. Accordingly, we have also assessed the performance of our model in the context of natural language understanding and generation. We perform the evaluation on MMLU [22], BBH [55], AGIEval [74] and ARC [13]. The results are illustrated in Table 3. As observed in the table, mPLUGOwl2 excels in examination and reasoning, showing a significant improvement on MMLU and BBH by 2.3% and 3.8% respectively. This indicates that mPLUG-Owl2 not only performs well on multi-modal tasks but also achieves better performance compared to the other instruction-tuned LLMs, showing the promising way for developing strong MLLMs.

Method

MSRVTT-QA MSVD-QA TGIF-QA

Accuracy Score Accuracy Score Accuracy Score Exacting Match

|Flamingo-80B [2] FrozenBiLM [64] BLIP-2 [31] HiTeA [67] InstructBLIP [15]|17.4 16.8 -<br><br>9.2 -<br><br>21.7 -<br><br>22.1 -<br><br><br>|35.6 32.2 18.3 37.4 41.8 -|- 41.0 -<br><br>- -<br><br>- -<br><br>- -<br>|
|---|---|---|---|
|mPLUG-Owl2|23.6 -|42.4 -<br><br>|61.6 -|

GPT-Assisted

|Video Chat [32] LLaMA-Adapter [19] Video-LLaMA [71] Video-ChatGPT [42]<br><br>|45.0 2.5 43.8 2.7 29.6 1.8 49.3 2.8|56.3 2.8 54.9 3.1 51.6 2.5 64.9 3.3<br><br>|34.4 2.3<br><br>- -<br>- -<br><br><br>51.4 3.0|
|---|---|---|---|
|mPLUG-Owl2|46.7 2.9|65.4 3.5<br><br>|67.1 3.7|

- Table 4. Zero-shot evaluation on video question answering. Accuracy and relevance score are reported.

Zero-Shot Video Question Answering. Given that videos can be viewed as a sequence of images, we conducted a comprehensive quantitative evaluation using several commonly employed video question-answering datasets, including MSRVTT-QA [62], MSVD-QA [62], and TGIF-QA [25]. These datasets aided in the zero-shot evaluation of the model’s ability to understand video content, with the

83.9

82.3

79.9

mPLUG-Owl2 w/o MAM w/ MAM

78.0

80

71.7

71.2

| |
|---|

65.8

62.4

| |
|---|

60

53.4

51.6

Scores

45.0

43.8

43.2

40

36.1

32.7

30.6

27.5

26.8

20

0

MMLU(Exam)AGIEval(Exam)ARC-c(Exam)ARC-e(Exam)BoolQ(Know)TrivialQA(Know)Xsum(Understanding)HellaSwag(Reason)BBH(Reason)

Text Datasets

- Figure 3. Performance of text benchmarks across various capabilities under modality collaboration.

results summarized in Table 4. We employed two types of evaluations: 1) Exact matching, which is commonly used in previous video question-answering evaluations; and 2) GPT-assisted evaluation [42] that assesses the model’s capabilities by measuring the accuracy of the model’s generated predictions and providing a relative score on a scale of 1-5. We observe that our model achieves superior results on all three video datasets under a zero-shot setting. Furthermore, in terms of relevancy, our model generates more accurate answers than other video MLLMs, thereby demonstrating its superiority and excellent generalization capabilities.

- 4.3. Discussion

Modality Collaboration for Text Performance. To demonstrate how modality collaboration enhances not only the multi-modal performance but also the text capability of MLLMs, we evaluate the performance of text benchmarks in terms of various abilities including examination, knowledge, understanding, and reasoning. As observed in Figure 3, both examination and knowledge capabilities of MLLMs have significantly improved thanks to the benefits of modality collaboration facilitated by the modality-adaptive module. This improvement arises because multi-modal data allows the model to utilize visual information to understand concepts that cannot be described through language. Similarly, the model can generate richer and more substantial responses due to a more concrete understanding of these concepts. Additionally, multi-modal data enhances the reasoning ability of the model because images contain rich information (such as relationships and spatial aspects). The model learns from these aspects and associates them with the text, thereby indirectly enhancing the reasoning ability of the text.

Impact of Joint Vision-Language Instruction Tuning. Table 5 presents the results of instruction tuning with vari-

- ous types of data as well as whether using modality-adaptive module. These results show that even without multimodal instruction data, the model’s performance on multimodal benchmarks is respectable due to the effective visionlanguage alignment achieved during pre-training. However,

|MAM|Text Inst.|MM Inst.<br><br>|VQAv2 Q-Bench MMLU BBH|
|---|---|---|---|
| |✓<br><br>✓<br><br>|✓ ✓<br><br>|58.2 54.4 51.8 43.6 76.3 61.3 45.4 25.7 76.2 60.3 51.6 43.2|

✓ ✓ 60.5 55.6 51.8 44.0 ✓ ✓ 76.5 60.2 46.1 30.6 ✓ ✓ ✓ 76.8 62.2 52.8 45.0

Table 5. Performance comparison among different types of instruction data and structures.

|Unfreeze<br><br>|Layer-wise lr.|VQAv2 TextVQA MMBench Q-Bench|
|---|---|---|
|✓ ✓<br><br>|✓|74.8 39.8 63.8 60.7 76.2 (+1.4) 40.3 (+0.5) 62.7 (-1.1) 61.6 (+0.9) 76.8 (+2.0) 42.5 (+2.7) 64.5 (+0.7) 62.2 (+1.5)<br><br>|

- Table 6. Influence of learning strategies for visual encoder.

|# Learnable Queries<br><br>|VQAv2 TextVQA MMBench Q-Bench|
|---|---|
|8 16 32 64 128<br><br>|58.3 18.6 47.6 52.4 66.2 28.5 52.9 54.9 72.4 36.3 60.2 57.8 76.8 42.5 64.5 62.2 76.7 44.4 63.6 61.6|

- Table 7. Performance in terms of number of learnable queries.

when solely using multi-modal instruction data, we observe an increase in performance on multi-modal datasets, while performance on text tasks decreases by about 5.7%. This phenomenon can be counterbalanced by the joint visionlanguage tuning proposed, as shown in the table’s third row, where the multi-modal performance begins to slightly decrease due to modality interference. To counter this drawback, we apply our proposed modality-adaptive module to the model. Results show that the performance on both multimodal and text benchmarks improves, with a minimum increase of 0.6% on the VQAv2 dataset and 1.6% on MMLU.

Impact of Trainable Vision Encoder. Table 6 delivers the performance of the training vision encoder during instruction tuning with modality collaboration. It can be observed that enabling the vision encoder to be trainable improves performance on VQAv2 and Q-Bench by at least 1.4% and 0.9%, respectively, suggesting the benefits of modality collaboration. Conversely, it results in a 1.1% performance drop in MM-Bench, indicating a degree of forgetting and damage to the general visual representation due to the limited diversity of instruction data. To mitigate this challenge, we apply layer-wise learning rate decay with an exponential decay factor of 0.9, which preserves the representation of lower layers while modifying higher semantic representations. By applying the layer-wise learning rate decay, we can notice that performance on TextVQA has increased further with 2.2%, showing the effectiveness of our training strategy.

Impact of Number of Learnable Queries. To investigate the effect of the number of learnable queries Q, we conduct experiments using different numbers of queries in the visual

abstractor, as shown in Table 7. It can be observed that the model consistently exhibits improvement as the number of learnable queries increases until it reaches a saturation point, suggesting that 64 may be the optimal number for representing an image. Notably, there is a significant performance boost observed when the number is increased from 8 to 64, e.g., the performance of VQAv2 is increased 18.5%. These findings suggest that a higher number of learnable queries can capture image information more comprehensively, thereby enhancing the model’s image comprehension capabilities.

Layer #0

Visual Tokens Textual Tokens Visual Tokens Textual Tokens

[Figure 25]

[Figure 26]

w/ Modality-Adaptive Module w/o Modality-Adaptive Module

Layer #15

Visual Tokens Textual Tokens Visual Tokens Textual Tokens

[Figure 27]

[Figure 28]

w/ Modality-Adaptive Module w/o Modality-Adaptive Module

Layer #31

Visual Tokens Textual Tokens Visual Tokens Textual Tokens

[Figure 29]

[Figure 30]

|Resolution<br><br>|VQAv2 TextVQA MMBench MM-Vet Q-Bench|
|---|---|
|224 × 224 336 × 336 448 × 448|76.8 42.5 64.5 34.0 62.2<br><br>78.5 (+1.7) 49.8 (+7.3) 65.2 (+0.7) 34.6 (+0.6) 62.4 (+0.2)<br><br>79.4 (+2.6) 54.3 (+11.8) 65.4 (+0.9) 36.2 (+2.2) 62.6 (+0.4)<br>|

w/ Modality-Adaptive Module w/o Modality-Adaptive Module

- Figure 4. Visualization of the attention maps with and without the Modality-Adaptive Module. We demonstrate the attention maps for the 0-th, 15-th, and 31-st layers, where the range of visual tokens is indicated by orange and the range of text tokens is indicated by blue.

When using the Modality-Adaptive Module, it can be observed that the model explicitly pays more attention to the textual content in the earlier stages and focuses more on the visual content in the later stages. The Modality-Adaptive Modulepreventsvisualandtextualtokensfrombeingtreated as the same and encourages collaboration between different modalities.

Impact of Modality-Adaptive Module in UnrelatedModality Scenarios. We present a question: "What are the seven colors of the rainbow?" along with a randomly selected image. In this example, the image input acts as a disturbance to the model. We aim to investigate the impact of our module on data that contains unrelated modalities. The responses and attention maps of the model are shown in

- Figure 5. Our proposed model, mPLUG-Owl2, which incorporates the Modality-Adaptive Module, accurately identifies all seven colors. During the generation process, it can be observed that the model primarily focuses on the textual input. On the other hand, when the Modality-Adaptive Module is not utilized, mPLUG-Owl2 only identifies six colors. The model’s ability to comprehend text instructions is disrupted, and it is also evident that it places more emphasis on the image during generation. Thanks to the Modality-Adaptive Module, mPLUG-Owl2 is better able to capture modalityspecific features when modeling multimodal inputs. This enhances the adaptability of modality collaboration, resulting in reduced disturbance when the text and image are unrelated.

Table 8. Influence of different input image resolutions.

Impact of Image Resolution. Image resolution plays a crucial role in vision-language tasks, as a higher resolution can reduce image blur and improve understanding of finegrained details. To explore the impact of image resolution on performance across different benchmarks, we adjust the image resolution from 224×224 to 448×448 and the results are listed in Table 8. As observed in the table, using a higher resolution proves advantageous for multi-modal tasks, particularly in the question answering scenario. Specifically, the performance of VQAv2 has increased from 76.8 to 79.4, representing a 2.6% boost. Simultaneously, there is an 11.8 point lift in the TextVQA benchmark when enlarging the resolution from 224 × 224 to 448 × 448. This suggests that OCR-related tasks benefit significantly from increasing the resolution.

#### 4.4. Qualitative Analysis

Impact of Modality-Adaptive Module in Multi-Modal Scenario. We investigate the impact of the ModalityAdaptive Module in multi-modal scenarios by visualizing the attention maps of mPLUG-Owl2 with and without this module using image caption input, as shown in Figure 4. Each attention map illustrates the attention scores of generated tokens on the input sequence during the generation process.

It can be observed that regardless of whether the Modality-Adaptive Module is incorporated or not, the model focuses more on the textual tokens in the earlier layers while paying more attention to the visual tokens in the later layers. This suggests that the modeling of visual and textual information plays different roles in the collaboration of multimodal language models (MLLMs). An intuitive explanation is that MLLMs initially use syntactic information to comprehend instructions and then identify relevant visual content tokens by considering the textual input.

### 5. Conclusion

In this paper, we present mPLUG-Owl2, a highly capable generalist model by leveraging modality collaboration for

Neural Information Processing Systems, 35:23716–23736,

mPLUG-Owl2 w/ Modality-Adaptive Module

[Figure 31]

2022. 2, 6

User: <image> What are the seven colors of the rainbow? Response: orange, yellow, green, blue, indigo, violet and white.

- [3] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, et al. Openflamingo: An opensource framework for training large autoregressive visionlanguage models. arXiv preprint arXiv:2308.01390, 2023. 16
- [4] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450,

2016. 4

- [5] Jinze Bai, Shuai Bai, Shusheng Yang, Shĳie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. ArXiv, abs/2308.12966, 2023. 2, 5, 6, 16
- [6] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, PranavShyam, GirishSastry, AmandaAskell, SandhiniAgarwal, Ariel Herbert-Voss, Gretchen Krueger, T. J. Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeff Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are fewshot learners. ArXiv, abs/2005.14165, 2020. 1, 3
- [7] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo700m: Image-text pair dataset. https://github.com/ kakaobrain/coyo-dataset, 2022. 5
- [8] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020. 2, 3
- [9] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3558–3568, 2021. 5
- [10] Ke Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. ArXiv, abs/2306.15195, 2023. 2, 5, 15, 16
- [11] Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, et al. Pali: A jointlyscaled multilingual language-image model. arXiv preprint arXiv:2209.06794, 2022. 5
- [12] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam M. Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Benton C. Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghe-

|Visual Tokens Textual Tokens<br><br>Average Attention Map<br><br>[Figure 32]|
|---|

mPLUG-Owl2 w/o Modality-Adaptive Module

[Figure 33]

User: <image> What are the seven colors of the rainbow? Response: Orange, Yellow, Green, Blue, Purple, and White.

Visual Tokens Textual Tokens

[Figure 34]

Average Attention Map

- Figure 5. Visualization of the attention maps with and without the Modality-Adaptive Module. We demonstrate the average of attention maps across each layer, where the range of visual tokens is indicated by orange and the range of text tokens is indicated by blue.

enhancing performance across both text and multi-modal tasks. The inclusion of shared functional modules and a modality-adaptive module in mPLUG-Owl2 strengthens the model’s ability to harmonize modality collaboration and preserve modality-specific characteristics. The extensive experimental evaluations highlight mPLUG-Owl2’s proficiency in generalizing across various tasks, thereby achieving state-of-the-art performances with a singular, generalized model. Most notably, mPLUG-Owl2 stands as the first MLLM model to exhibit the phenomena of modality collaboration in both pure-text and multi-modal contexts. This not only enhances the model’s vision-language understanding but also improves its language capabilities in terms of understanding, knowledge, and reasoning. This represents a significant contribution to the field and opens up exciting opportunities for the future development of multi-modal foundation models.

### References

- [1] Sharegpt. http://sharegpt.com, 2023. 2, 5, 17
- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in

- mawat, Sunipa Dev, Henryk Michalewski, Xavier García, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Díaz, Orhan Firat, Michele Catasta, Jason Wei, Kathleen S. Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. Palm: Scaling language modeling with pathways. J. Mach. Learn. Res., 24:240:1–240:113, 2022. 2
- [13] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018. 6
- [14] OpenCompass Contributors. Opencompass: A universal evaluation platform for foundation models. https: //github.com/open-compass/opencompass, 2023. 6
- [15] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Albert Li, Pascale Fung, and Steven C. H. Hoi. Instructblip: Towards general-purposevision-languagemodelswithinstructiontuning. ArXiv, abs/2305.06500, 2023. 2, 3, 5, 6, 13, 14, 15, 16
- [16] Danny Driess, F. Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Ho Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, and Peter R. Florence. Palm-e: An embodied multimodal language model. In International Conference on Machine Learning, 2023. 2, 5
- [17] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023. 5, 6
- [18] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. arXiv preprint arXiv:2304.14108, 2023. 5
- [19] Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shĳie Geng, Aojun Zhou, W. Zhang, Pan Lu, Conghui He, Xiangyu Yue, Hongsheng Li, and Yu Jiao Qiao. Llama-adapter v2: Parameter-efficient visual instruction model. ArXiv, abs/2304.15010, 2023. 5, 6, 15, 16
- [20] Tao Gong, Chengqi Lyu, Shilong Zhang, Yudong Wang, Miao Zheng, Qian Zhao, Kuikun Liu, Wenwei Zhang, Ping Luo, and Kai Chen. Multimodal-gpt: A vision and language model for dialogue with humans. arXiv preprint arXiv:2305.04790, 2023. 15
- [21] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the V in VQA matter: Elevating the role of image understanding in Visual Question

- Answering. In Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 5, 13, 17
- [22] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuringmassivemultitasklanguageunderstanding. arXivpreprint arXiv:2009.03300, 2020. 6
- [23] ShaohanHuang, LiDong, WenhuiWang, YaruHao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Qiang Liu, Kriti Aggarwal, Zewen Chi, Johan Bjorck, Vishrav Chaudhary, Subhojit Som, Xia Song, and Furu Wei. Language is not all you need: Aligning perception with language models. ArXiv, abs/2302.14045, 2023. 2
- [24] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019. 5, 13, 17
- [25] Yunseok Jang, Yale Song, Youngjae Yu, Youngjin Kim, and Gunhee Kim. Tgif-qa: Toward spatio-temporal reasoning in visual question answering. In Proceedings of the IEEE conferenceoncomputervisionandpatternrecognition, pages 2758–2766, 2017. 6
- [26] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123:32–73, 2017. 5, 15, 17
- [27] Gukyeong Kwon, Zhaowei Cai, Avinash Ravichandran, Erhan Bas, Rahul Bhotika, and Stefano Soatto. Masked vision and language modeling for multi-modal representation learning. arXiv preprint arXiv:2208.02131, 2022. 2
- [28] Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, and Victor Sanh. Obelics: An open webscale filtered dataset of interleaved image-text documents,

2023. 13, 16

- [29] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023. 5, 6
- [30] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning. ArXiv, abs/2305.03726,

2023. 5, 15, 16

- [31] Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. Blip-2: Bootstrappinglanguage-image pre-training with frozen image encoders and large language models. ArXiv, abs/2301.12597, 2023. 2, 5, 6, 16
- [32] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 6
- [33] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji rong Wen. Evaluating object hallucination in large vision-language models. ArXiv, abs/2305.10355, 2023. 13

- [34] Wing Lian, Guan Wang, Bleys Goodson, Eugene Pentland, Austin Cook, Chanvichet Vong, and "Teknium". Slimorca: An open dataset of gpt-4 augmented flan reasoning traces, with verification, 2023. 5, 17
- [35] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 5, 13, 17
- [36] Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lĳuan Wang. Aligning large multi-modal model with robust instruction tuning. arXiv preprint arXiv:2306.14565, 2023. 3
- [37] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. ArXiv, abs/2310.03744, 2023. 3, 5, 14, 16
- [38] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. ArXiv, abs/2304.08485, 2023. 2, 3, 4, 5, 13, 14, 15, 16, 17
- [39] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023. 5, 6, 13, 15
- [40] Ilya Loshchilov and Frank Hutter. Fixing weight decay regularization in adam. 2018. 5
- [41] Jiasen Lu, Christopher Clark, Rowan Zellers, Roozbeh Mottaghi, and Aniruddha Kembhavi. Unified-io: A unified model for vision, language, and multi-modal tasks. ArXiv, abs/2206.08916, 2022. 5
- [42] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and FahadShahbazKhan. Video-chatgpt: Towardsdetailedvideo understanding via large vision and language models. ArXiv, abs/2306.05424, 2023. 6, 7
- [43] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE/cvf conference on computer vision and pattern recognition, pages 3195–3204, 2019. 5, 13, 17
- [44] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In 2019 international conference on document analysis and recognition (ICDAR), pages 947–

952. IEEE, 2019. 5, 13, 17

- [45] OpenAI. Gpt-4v(ision) system card. 2023. 1
- [46] OpenAI. Gpt-4 technical report. ArXiv, abs/2303.08774,

2023. 1, 2, 16

- [47] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. ArXiv, abs/2306.14824, 2023. 13, 16
- [48] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 3, 5, 17

- [49] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 5
- [50] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In European Conference on Computer Vision, pages 146–

162. Springer, 2022. 5, 13, 17

- [51] Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020. 3
- [52] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatronlm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019. 15
- [53] Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh. Textcaps: a dataset for image captioning with reading comprehension. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23– 28, 2020, Proceedings, Part II 16, pages 742–758. Springer,

2020. 5, 13, 17

- [54] ZhiqingSun, ShengShen, ShengcaoCao, HaotianLiu, Chunyuan Li, Yikang Shen, Chuang Gan, Liangyan Gui, YuXiongWang, YimingYang, KurtKeutzer, andTrevorDarrell. Aligning large multimodal models with factually augmented rlhf. ArXiv, abs/2309.14525, 2023. 13, 15, 16
- [55] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, et al. Challenging big-benchtasksandwhetherchain-of-thoughtcansolvethem. arXiv preprint arXiv:2210.09261, 2022. 6
- [56] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford_ alpaca, 2023. 2
- [57] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. ArXiv, abs/2302.13971, 2023. 1, 3
- [58] Hugo Touvron, Louis Martin, Kevin R. Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Daniel M. Bikel, Lukas Blecher, Cristian Cantón Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony S. Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel M. Kloumann, A. V. Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta,

- Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, R. Subramanian, Xia Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zhengxu Yan, IliyanZarov, YuchenZhang, AngelaFan, MelanieKambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. ArXiv, abs/2307.09288, 2023. 1, 3, 5, 6, 17
- [59] Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li, Kevin Lin, Zhe Gan, Zicheng Liu, Ce Liu, and Lĳuan Wang. Git: A generative image-to-text transformer for vision and language. arXiv preprint arXiv:2205.14100, 2022. 5
- [60] Haoning Wu, Zicheng Zhang, Erli Zhang, Chaofeng Chen, Liang Liao, Annan Wang, Chunyi Li, Wenxiu Sun, Qiong Yan, Guangtao Zhai, and Weisi Lin. Q-bench: A benchmark for general-purpose foundation models on low-level vision. arXiv preprint arXiv:2309.14181, 2023. 5, 6, 13, 16
- [61] Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. Wizardlm: Empowering large language models to follow complex instructions. ArXiv, abs/2304.12244, 2023. 2, 6
- [62] Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and motion. In Proceedings of the 25th ACM international conference on Multimedia, pages 1645–1653, 2017. 6
- [63] Haiyang Xu, Qinghao Ye, Ming Yan, Yaya Shi, Jiabo Ye, Yuanhong Xu, Chenliang Li, Bin Bi, Qi Qian, Wei Wang, etal. mplug-2: Amodularizedmulti-modalfoundationmodel across text, image and video. In International Conference on Machine Learning, 2023. 2, 4
- [64] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Zero-shot video question answering via frozen bidirectional language models. In NeurIPS, 2022. 6
- [65] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, YuhaoDan, ChenlinZhao, GuohaiXu, ChenliangLi, Junfeng Tian, Qian Qi, Ji Zhang, and Fei Huang. mplug-docowl: Modularized multimodal large language model for document understanding. CoRR, abs/2307.02499, 2023. 2
- [66] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Guohai Xu, Chenliang Li, Junfeng Tian, Qi Qian, Ji Zhang, et al. Ureader: Universal ocr-free visually-situated language understanding with multimodal large language model. In The 2023 Conference on Empirical Methods in Natural Language Processing, 2023. 2
- [67] Qinghao Ye, Guohai Xu, Ming Yan, Haiyang Xu, Qi Qian, Ji Zhang, and Fei Huang. Hitea: Hierarchical temporalaware video-language pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15405–15416, 2023. 6
- [68] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178, 2023. 2, 3, 4, 5, 15, 16
- [69] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In Computer Vision–ECCV 2016: 14th European

- Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14, pages 69–85. Springer, 2016. 5, 15, 17
- [70] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lĳuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023. 5, 6, 16
- [71] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. ArXiv, abs/2306.02858, 2023. 6
- [72] Bo Zhao, Boya Wu, and Tiejun Huang. Svit: Scaling up visual instruction tuning. arXiv preprint arXiv:2307.04087,

2023. 3

- [73] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric. P Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena, 2023. 6
- [74] Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models. arXiv preprint arXiv:2304.06364, 2023. 6
- [75] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. ArXiv, abs/2304.10592, 2023. 2, 3, 5, 14, 15, 16

### A. Additional Experimental Results

In this section, we provide more experimental results for the completeness of our proposed method.

#### A.1. Hallucination Evaluation

Overall

Other

Attribute

4.0

4.0

4.0

3.0

3.0

3.0

2.0

2.0

2.0

1.0

Holistic

Adversarial

4.0 3.0 2.0 1.01.0

1.0

1.0 2.0 3.0 4.0 1.0

N/A

1.0 2.0

1.0 2.0

1.0 2.0 3.0 4.0

2.0

3.0 4.0

3.0

4.0

3.0 4.0

Environment

Comparison

Relation Counting

LLaVA7B

LLaVA-RLHF7B

IDEFICS80B

Kosmos-2 IDEFICS9B

InstructBLIP7B

mPLUG-Owl2

LLaVA-SFT7+B

- Figure 6. Detailed performance of various models across the eight categories in MMHal-Bench [54], where "Overall" represents the average performance across all categories.

We measure the hallucination of our model on image description using MMHal-Bench [54] and compare the results with other recent vision-language models, including Kosmos-2 [47], IDEFICS [28], InstructBLIP [15], LLaVA [38], andLLaVA-RLHF[54]. Following[54], weuseGPT-4 to evaluate the overall score and hallucination rate of different MLLMs. As depicted in Figure 6, we find that our mPLUG-Owl2 tends to generate the response with reduced hallucination compared to other methods, especially surpassing IDEFICS [28] with 80 billion parameters, showing the superiority of our methods. Besides, we can notice that our model excels at attribute and counting because the visual abstractor can effectively identify the main parts of the image, which reduces the hallucination.

WealsostudythehallucinationofrecentpopularMLLMs and present the results in Figure 7. In the first example, the query asks the models to recognize the pattern on the wall. However, the pattern is not clearly visible in the image, causing other models to mistakenly perceive it as a solid color. Our model, on the other hand, accurately notices the white pattern on the wall and correctly answers the question. In the second example, there are only a few trees in the image. However, InstructBLIP incorrectly considers that there are no trees in the image. LLaVA and LLaVA-1.5, on the other hand, hallucinate and consider the tree in the image to be dense. MiniGPT-4 gives the correct answer, but with minimal explanation. Our mPLUG-Owl2, however,

answers the question correctly and provides a more detailed explanation.

#### A.2. POPE Evaluation

We also conduct the hallucination evaluation using POPE [33], the results are shown in Table 9. As we can observe in the table, we can find mPLUG-Owl2 achieves higher F1 scores on the popular and adversarial split, showing the robustness of our model in terms of object hallucination compared to other MLLMs.

#### A.3. Detailed Evaluation Results on MMBench

MMBench [39] is a meticulously designed benchmark that comprehensively assesses the diverse skills of visionlanguage models. The results from the test set for various MLLMs are presented in Table 10.

#### A.4. Detailed Evaluation Results on MM-Vet

WeprovidethedetailedresultsofMM-VetinTable11. Itcan be observed that by training the visual encoder of mPLUGOwl2, it exhibits stronger OCR capability compared to the model with the same backbone (i.e., LLaVA, Otter). Besides, mPLUG-Owl2 surpasses models with stronger language decoders such as LLaVA-13B which equips LLM with 13 billion parameters.

#### A.5. Detailed Evaluation Results on Q-Bench

For evaluating the low-level visual perception abilities, we have included the results of Q-Bench [60] on the test set. By training the visual encoder, the ability of mPLUG-Owl2 in terms of low-level perception has been improved significantly, as it outperforms the model with a stronger visual encoder (i.e., ViT-L (0.3B) v.s. ViT-G (1.9B)), showing the effectiveness of our training paradigm.

#### A.6. Detailed Evaluation Results on MMHal-Bench

We include Table 13 for the full evaluation results on MMHal-Bench [54].

### B. Implementation

#### B.1. Data Mixture

In this section, we detail our final training data mixture used during the instruction tuning stage in Table 14. Specifically, we process the VQAv2 [21] data by selecting the answer with the highest confidence and combining question-answer pairs that share the same image. This combining strategy is also applied to GQA [24], OKVQA [43], and OCRVQA [44] datasets. Additionally, for multiple-choice questions in A-OKVQA [50], we augment the dataset by switching the order of options to enhance robustness in terms of multiple choices. For caption datasets like COCO [35] and TextCaps [53], we randomly select one caption from the ground truth

[Figure 35]

Input Image:

Is the wall solid color in this image?

Query

[Figure 36]

InstructBLIP LLaVA

Yes, the wall in the image is solid color.

[Figure 37]

!

Yes, the wall in the image is solid color, which is tan or beige.

[Figure 38]

!

LLaVA-1.5

Yes, the wall in the image is solid color, which is pink.

MiniGPT-4

The wall in this image appears to be solid color.

[Figure 39]

mPLUG-Owl2

No, the wall in this image is not solid color. It has a pink and white pattern.

- Example #1
- Example #2

[Figure 40]

Input Image:

Query

Is the tree dense in this image?

[Figure 41]

InstructBLIP LLaVA

No, there is no tree in the image. Yes, the tree in the image is dense, with a large number of branches and leaves. Yes, the tree in this image is dense. The tree is not dense in this image. No, the tree in the image is not dense. It appears to be a small tree in the background.

[Figure 42]

!

[Figure 43]

!

LLaVA-1.5 MiniGPT-4 mPLUG-Owl2

[Figure 44]

Figure 7. Example cases compared with InstructBLIP [15], LLAVA [38], LLAVA-1.5 [37], MiniGPT-4 [75] and our mPLUG-Owl2.

- Table 9. Object hallucination benchmark using POPE evaluation pipeline . "Yes" signifies the likelihood of the model producing a positive response.

|Datasets<br><br>|Metrics|
|---|---|
|Random<br><br>|Accuracy (↑) Precision (↑) Recall (↑) F1-Score (↑) Yes (→ 50%)|
|Popular<br><br>|Accuracy (↑) Precision (↑) Recall (↑) F1-Score (↑) Yes (→ 50%)|
|Adversarial<br><br>|Accuracy (↑) Precision (↑) Recall (↑) F1-Score (↑) Yes (→ 50%)|

mPLUG-Owl2 Shikra [10] InstructBLIP [15] MiniGPT-4 [75] LLaVA [38] MM-GPT [20] mPLUG-Owl [68]

- 88.28 86.90 88.57 79.67 50.37 50.10 53.97

94.34 94.40 84.09 78.24 50.19 50.05 52.07 82.20 79.27 95.13 82.20 99.13 100.00 99.60 87.85 86.19 89.27 80.17 66.64 66.71 68.39

- 44.91 43.26 56.57 52.53 98.77 99.90 95.63

86.20 83.97 82.77 69.73 49.87 50.00 50.90 89.46 87.55 76.27 65.86 49.93 50.00 50.46 82.06 79.20 95.13 81.93 99.27 100.00 99.40 85.60 83.16 84.66 73.02 66.44 66.67 66.94

- 45.86 45.23 62.37 62.20 99.40 100.00 98.57

- 84.12 83.10 72.10 65.17 49.70 50.00 50.67

- 85.54 85.60 65.13 61.19 49.85 50.00 50.34

- 82.13 79.60 95.13 82.93 99.07 100.00 99.33

- 83.80 82.49 77.32 70.42 66.32 66.67 66.82

48.00 46.50 73.03 67.77 99.37 100.00 98.67

Method Language Model Vision Model Overall LR AR RR FP-S FP-C CP

MMGPT [20] LLaMA-7B CLIP ViT-L/14 16.0 1.1 23.8 20.7 18.3 5.2 18.3 MiniGPT-4 [75] Vicuna-7B EVA-G 12.0 13.6 32.9 8.9 28.8 11.2 28.3 InstructBLIP [15] Vicuna-7B EVA-G 33.9 21.6 47.4 22.5 33.0 24.4 41.1 LLaMA-Adapter-v2 [19] LLaMA-7B CLIP ViT-L/14 38.9 7.4 45.3 19.2 45.0 32.0 54.0 LLaVA [54] Vicuna-7B CLIP ViT-L/14 36.2 15.9 53.6 28.6 41.8 20.0 40.4 G2PT [39] Vicuna-7B ViT-G 39.8 14.8 46.7 31.5 41.8 34.4 49.8 Otter-I [30] LLaMA-7B CLIP ViT-L/14 48.3 22.2 63.3 39.4 46.8 36.4 60.6 mPLUG-Owl† [68] LLaMA-7B CLIP ViT-L/14 62.3 37.5 75.4 56.8 67.3 52.4 67.2 Shikra [10] Vicuna-7B CLIP ViT-L/14 60.2 33.5 69.6 53.1 61.8 50.4 71.7

mPLUG-Owl2 LLaMA2-7B CLIP ViT-L/14 65.4 29.2 69.7 61.7 67.0 60.0 79.5

- Table 10. CircularEval multi-choice accuracy results on MMBench [39] dev set. We adopt the following abbreviations: LR for Logical Reasoning; AR for Attribute Reasoning; RR for Relation Reasoning; FP-C for Fine-grained Perception (Cross Instance); FP-S for Finegrained Perception (Single Instance); CP for Coarse Perception. Baseline results are taken from [39]. † denotes the model is carefully optimized for MMBench.

for each image. Concurrently, some regional-VQA [26, 69] datasets are also used to improve regional abilities.

#### B.2. Training Hyper-parameters

We report the detailed training hyper-parameter settings of mPLUG-Owl2 in Table 15. Specifically, we leverage the model parallelism with Megatron [52] distributed training framework to ensure a larger resolution training while maintaining efficiency.

### C. Summary of the Evaluation Benchmarks

We provide a detailed summary of the used evaluation benchmarks and corresponding metrics in Table 16.

### D. Broader Impact

mPLUG-Owl2 employs off-the-shelf LLM and web-sourced data. Consequently, it inherits some of the weaknesses of the original LLM and web-crawled data, such as generating

uncensored text or producing biased outputs. We address these shortcomings by enhancing the model’s grounding on the visual and instructional input and executing joint vision-language instruction tuning on a diverse range of high-quality datasets. However, we advise against deploying mPLUG-Owl2 models for any downstream applications without prior evaluation of safety and fairness specific to the respective application.

Model Rec OCR Know Gen Spat Math Total Transformers Agent (GPT-4) [46] 18.2 3.9 2.2 3.2 12.4 4.0 13.4±0.5 MiniGPT-4-7B [75] 27.4 15.0 12.8 13.9 20.3 7.7 22.1±0.1 BLIP-2-12B [31] 27.5 11.1 11.8 7.0 16.2 5.8 22.4±0.2 LLaVA-7B [38] 28.0 17.1 16.3 18.9 21.2 11.5 23.8±0.6 MiniGPT-4-13B [75] 29.9 16.1 20.4 22.1 22.2 3.8 24.4±0.4 Otter-9B [30] 27.3 17.8 14.2 13.8 24.4 3.8 24.7±0.3 OpenFlamingo-9B [3] 28.7 16.7 16.4 13.1 21.0 7.7 24.8±0.2 InstructBLIP-13B [15] 30.8 16.0 9.8 9.0 21.1 10.5 25.6±0.3 InstructBLIP-7B [15] 32.4 14.6 16.5 18.2 18.6 7.7 26.2±0.2 LLaVA-7B (LLaMA-2) [38] 32.9 20.1 19.0 20.1 25.7 5.2 28.1±0.4 LLaMA-Adapter v2-7B [19] 38.5 20.3 31.4 33.4 22.9 3.8 31.4±0.1 LLaVA-13B (V1.3) [38] 38.1 22.3 25.2 25.8 31.3 11.2 32.5±0.1 LLaVA-13B (LLaMA-2) [38] 39.2 22.7 26.5 29.3 29.6 7.7 32.9±0.1 mPLUG-Owl2 41.3 27.4 27.5 27.9 30.3 7.7 36.2±0.1

- Table 11. Evaluation results on various MLLMs regarding each core VL capability on MM-Vet [70]. Rec stands for recognition; Know indicates knowledge; Gen is generation; Spat means spatial. All the numbers are presented in % and the full score is 100%.

|Method<br><br>|Yes-or-no What How Distortion Others In-context Distortion In-context Others Overall|
|---|---|
|IDEFICS [28] InstructBLIP [15] Kosmos-2 [47] LLaMA-Adapter-v2 [19] LLaVA-1.5 [37] LLaVA [38] MiniGPT-4 [75] mPLUG-Owl [68] Otter [30] Qwen-VL [5] Shikra [10]<br><br>|0.6004 0.4642 0.4671 0.4038 0.5990 0.4726 0.6477 0.5151 0.7099 0.5141 0.4300 0.4500 0.6301 0.5719 0.6439 0.5585 0.6058 0.3124 0.3539 0.3865 0.4654 0.4349 0.4735 0.4334 0.6661 0.5466 0.5165 0.5615 0.6181 0.5925 0.5455 0.5806 0.6460 0.5922 0.5576 0.4798 0.6730 0.5890 0.7376 0.6007 0.5712 0.5488 0.5185 0.4558 0.5800 0.5719 0.6477 0.5472 0.6077 0.5033 0.4300 0.4558 0.5251 0.5342 0.6098 0.5177 0.7245 0.5488 0.4753 0.4962 0.6301 0.6267 0.6667 0.5893 0.5766 0.3970 0.4259 0.4212 0.4893 0.4760 0.5417 0.4722 0.6533 0.6074 0.5844 0.5413 0.6635 0.5822 0.7300 0.6167 0.6909 0.4793 0.4671 0.4731 0.6086 0.5308 0.6477 0.5532|
|mPLUG-Owl2<br><br>|0.7318 0.5531 0.5864 0.5374 0.7136 0.5788 0.7338 0.6294|

Table 12. Detailed evaluation results for different MLLMs on the test set of Q-Bench [60].

|Method<br><br>|Overall Hallucination Score ↑ Rate ↓|Score in Each Question Type ↑ Attribute Adversarial Comparison Counting Relation Environment Holistic Other<br><br>|
|---|---|---|
|Kosmos-2 [47] IDEFICS9B [28] IDEFICS80B [28] InstructBLIP7B [15] InstructBLIP13B [15] LLaVA7B [38] LLaVA-RLHF7B [54]|1.69 0.68<br><br>1.89 0.64<br><br>2.05 0.61<br><br><br>2.10 0.58<br><br><br>2.14 0.58<br><br>1.55 0.76<br><br>2.05 0.68<br><br><br>|2.00 0.25 1.42 1.67 1.67 2.67 2.50 1.33<br><br>1.58 0.75 2.75 1.83 1.83 2.50 2.17 1.67<br>2.33 1.25 2.00 2.50 1.50 3.33 2.33 1.17<br>3.42 2.08 1.33 1.92 2.17 3.67 1.17 1.08 2.75 1.75 1.25 2.08 2.50 4.08 1.50 1.17<br><br><br>1.33 0.00 1.83 1.17 2.00 2.58 1.67 1.83<br>2.92 1.83 2.42 1.92 2.25 2.25 1.75 1.08<br>|
|mPLUG-Owl2<br><br>|2.17 0.56<br><br>|3.67 2.25 2.17 2.75 1.25 2.08 1.50 1.75|

Table 13. Detailed evaluation results for different MLMMs on MMHal-Bench.

|Data Type|Data Name Size|
|---|---|
|Text|ShareGPT [1] 40K<br><br>SlimOrca [34] 518K<br><br>|
|Dialogue<br><br>|LLaVA [38] 158K|
|Caption<br><br>|COCO [35] 82K TextCaps [53] 22K|
|VQA<br><br>|VQAv2 [21] 83K GQA [24] 72K OKVQA [43] 9K OCRVQA [44] 80K A-OKVQA [50] 50K|
|Regional-VQA<br><br>|RefCOCO [69] 30K VisualGenome [26] 86K|

Total 1.23M Table 14. Instruction-following Data Mixture of mPLUG-Owl2.

|Configuration<br><br>|Pre-training Instruction Tuning|
|---|---|
|ViT init. LLM init. Visual Abstractor init. Image resolution ViT sequence length LLM sequence length Learnable query numbers Optimizer Optimizer hyperparameter Peak learning rate Minimum learning rate ViT learning rate decay ViT Drop path rate Learning rate schedule Weight decay Gradient clip Training steps Warm-up steps Global batch size Gradient Acc. Numerical precision Optimizer sharding Activation checkpointing Model parallelism Pipeline parallelism<br><br>|CLIP-L/14 [48] Pre-train stage LLaMA-2 [58] LLaMA-2 [58]<br><br>Random Pre-train stage 224 × 224 448 × 448 256 1024 256 2048 64 64<br><br>AdamW β1 = 0.9, β2 = 0.98, ϵ = 1e−6<br><br>1e−4 2e−5 1e−6 1e−7<br><br>0 0 Cosine<br><br>0.05 0 1.0<br><br>42,500 4,800 1,000 250 8,192 256<br><br>16 bfloat16 ✓ ✓<br><br>1 2 1|

Table 15. Training hyper-parameters of mPLUG-Owl2.

Task Dataset Description Split Metric Image Caption

COCO Captioning of natural images karpathy-test CIDEr (↑) Flickr30K Captioning of natural images karpathy-test CIDEr (↑)

VQAv2 VQA on natural images test-dev VQA Score (↑) OKVQA VQA on natural images requiring outside knowledge val VQA Score (↑) GQA VQA on scene understanding and reasoning test-balanced EM (↑)

General VQA

VizWizQA VQA on photos taken by people who are blind test-dev VQA Score (↑) TextVQA VQA on natural images containing text val VQA Score (↑) SciQA-Img Multi-choice VQA on a diverse set of science topics test Accuracy (↑)

MSRVTT-QA Video Question Answering test Accuracy (↑) / Relevance Score (↑) MSVD-QA Video Question Answering test Accuracy (↑) / Relevance Score (↑) TGIF-QA GIF Question Answering test Accuracy (↑) / Relevance Score (↑)

VideoQA

MMLU A benchmark designed to measure knowledge acquirement dev Accuracy (↑) BBH A suite of 23 challenging BIG-Bench tasks test Accuracy (↑) AGIEval A human-centric benchmark specifically designed to evaluate the general abilities of foundation model test Accuracy (↑) ARC-c A multiple-choice question-answering dataset, containing questions from science exams from grade 3 to grade 9. test Accuracy (↑) ARC-e A multiple-choice question-answering dataset, containing questions from science exams from grade 3 to grade 9. test Accuracy (↑)

Text Benchmark

MME Open-ended VL Benchmark by yes/no questions Perception Accuracy (↑) MMBench Open-ended VL Benchmark by Multi-choice VQA with Circular Evaluation test Accuracy (↑) MM-Vet Open-ended VL Benchmark with Various Abilities test GPT-4 Score (↑)

Instruction Following

SEED-Bench Open-ended VL Benchmark by Multi-choice VQA Image & Video Accuracy (↑) Q-Bench Open-ended Low-level Vision Benchmark by Multi-choice VQA test Accuracy (↑)

POPE Object existence by yes/no questions random/popular/adversarial Accuracy / Precision / Recall / F1 (↑)

Hallucination

MMHal-Bench Open-ended hallucination benchmarks test GPT-4 Score (↑)

###### Table 16. Summary of the evaluation benchmarks of mPLUG-Owl2. EM stands for exacting matching.

