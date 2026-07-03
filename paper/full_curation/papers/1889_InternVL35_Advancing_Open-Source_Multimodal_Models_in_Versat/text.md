# arXiv:2508.18265v2[cs.CV]27Aug2025

## InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency

Weiyun Wang∗, Zhangwei Gao∗, Lixin Gu∗, Hengjun Pu∗, Long Cui∗, Xingguang Wei∗, Zhaoyang Liu∗, Linglin Jing∗, Shenglong Ye∗, Jie Shao∗, Zhaokai Wang∗, Zhe Chen∗, Hongjie Zhang, Ganlin Yang, Haomin Wang, Qi Wei, Jinhui Yin, Wenhao Li, Erfei Cui, Guanzhou Chen, Zichen Ding, Changyao Tian, Zhenyu Wu, Jingjing Xie, Zehao Li, Bowen Yang, Yuchen Duan, Xuehui Wang, Zhi Hou, Haoran Hao, Tianyi Zhang, Songze Li, Xiangyu Zhao, Haodong Duan, Nianchen Deng, Bin Fu, Yinan He, Yi Wang, Conghui He, Botian Shi, Junjun He, Yingtong Xiong, Han Lv, Lijun Wu, Wenqi Shao, Kaipeng Zhang, Huipeng Deng, Biqing Qi, Jiaye Ge, Qipeng Guo, Wenwei Zhang, Songyang Zhang, Maosong Cao, Junyao Lin, Kexian Tang, Jianfei Gao, Haian Huang, Yuzhe Gu, Chengqi Lyu, Huanze Tang, Rui Wang, Haijun Lv, Wanli Ouyang, Limin Wang, Min Dou, Xizhou Zhu, Tong Lu, Dahua Lin, Jifeng Dai, Weijie Su, Bowen Zhou , Kai Chen , Yu Qiao , Wenhai Wang †, Gen Luo †

InternVL Team, Shanghai AI Laboratory

Code: https://github.com/OpenGVLab/InternVL Model: https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B

### Abstract

We introduce InternVL3.5, a new family of open-source multimodal models that significantly advances versatility, reasoning capability, and inference efficiency along the InternVL series. A key innovation is the Cascade Reinforcement Learning (Cascade RL) framework, which enhances reasoning through a two-stage process: offline RL for stable convergence and online RL for refined alignment. This coarse-to-fine training strategy leads to substantial improvements on downstream reasoning tasks, e.g., MMMU and MathVista. To optimize efficiency, we propose a Visual Resolution Router (ViR) that dynamically adjusts the resolution of visual tokens without compromising performance. Coupled with ViR, our Decoupled VisionLanguage Deployment (DvD) strategy separates the vision encoder and language model across different GPUs, effectively balancing computational load. These contributions collectively enable InternVL3.5 to achieve up to a +16.0% gain in overall reasoning performance and a 4.05× inference speedup compared to its predecessor, i.e., InternVL3. In addition, InternVL3.5 supports novel capabilities such as GUI interaction and embodied agency. Notably, our largest model, i.e., InternVL3.5-241B-A28B, attains state-of-the-art results among opensource MLLMs across general multimodal, reasoning, text, and agentic tasks—narrowing the performance gap with leading commercial models like GPT-5. All models and code are publicly released.

### 1 Introduction

The recent trend of Multimodal Large Language Models (MLLMs) [46, 126, 128] has gone beyond simple multimodal understanding and gradually focused on more general, complex, and realistic tasks such as textrelated tasks [12, 44, 49, 84, 106, 177], reasoning tasks [43, 76, 134, 161, 170] and agentic tasks [38, 48, 50, 101, 116, 156, 160, 161, 174]. In these aspects, commercial models have created huge gaps with current open-source models, as shown in Table 2. Thereby, recent open-source efforts [46, 90, 126, 137] aim to explore advanced reinforcement learning (RL) methods to mitigate the gap and pursue higher multimodal intelligence. However, despite much effort in RL algorithms [23, 24, 90, 109, 133, 152, 171, 183] and verifiers [82, 136, 143], a stable, effective, and scalable reinforcement learning framework for MLLMs still remains an open problem in the

* equal contribution; corresponding authors; † technical leaders.

76.5

75.1

[Figure 1]

72.6

[Figure 2]

70.5 70.4

[Figure 3]

[Figure 4]

67.6 66.9

[Figure 5]

65.9 65.3 65.2 64.4 63.6

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

62.2

[Figure 11]

60.8

[Figure 12]

59.4 58.8

[Figure 13]

[Figure 14]

56.9 56.5

[Figure 15]

[Figure 16]

55.0

[Figure 17]

53.8

[Figure 18]

[Figure 19]

[Figure 20]

50.9 50.7

[Figure 21]

[Figure 22]

47.3

46.4

44.9

[Figure 23]

[Figure 24]

42.7

[Figure 25]

[Figure 26]

InternVL3.5-241B-A28BGemini-2.5ProGPT-5GLM-4.5V-106B-A12BInternVL3.5-38BInternVL3.5-30B-A3BInternVL3.5-14BStep-3-321B-A38BClaude3.7SonnetInternVL3.5-8BInternVL3.5-20B-A4BMiMo-VL-RL-7BGLM-4.1V-9BInternVL3.5-4BQwen2.5-VL-72BKimi-VL-2506-16B-A3BGPT-4oKeye-VL-8BQwen2.5-VL-32BInternVL3.5-2BGemma-3-27BQwen2.5-VL-7BMiniCPM-V-4-4BGemma-3-12BQwen2.5-VL-3BInternVL3.5-1B

- Figure 1: Comparison between InternVL3.5 and leading MLLMs in general capabilities. Hatched bars represent closed-source commercial models. We report average scores on a set of multimodal general, reasoning, text, and agentic benchmarks: MMBench v1.1 (en) [71], MMStar [11], BLINK [36], HallusionBench [41], AI2D [55], OCRBench [72], MMVet [168], MME-RealWorld (en) [178], MVBench [63], VideoMME [35], MMMU [170], MathVista [76], MathVision [134], MathVerse [175], DynaMath [189], WeMath [100], LogicVista [153], MATH500 [45], AIME24 [84], AIME25 [85], GPQA [106], MMLU-Pro [146], GAOKAO [177], IFEval [185], SGP-Bench [102], VSI-Bench [161], ERQA [121], SpaCE-10 [38], and OmniSpatial [50].

community. Furthermore, the growth of multimodal capabilities, e.g., long visual context and high-resolution understanding [5, 81, 140, 147, 180, 187], often comes with ever increasing computational costs, which have become a crucial bottleneck of real-world applications.

In this work, we introduce InternVL3.5, an advanced family of InternVL series [13, 14, 15, 37, 79, 80, 187] with stronger capabilities in versatility, reasoning, and efficiency. Compared to InternVL3 [187], InternVL3.5 achieves superior performance through our proposed Cascade RL framework, which enhances reasoning capabilities in an efficient, scalable, and stable manner. Cascade RL consists of two complementary substages: an offline RL stage [142], which efficiently achieves satisfactory performance, and an online RL stage [183], which carefully refines the output distribution and further push the performance upper bound of the model. The offline stage serves as an effective warm-up, ensuring high-quality rollouts for the subsequent online stage, thereby enabling the progressive improvement of MLLM reasoning abilities. In practice, Cascade RL demonstrates promising salability and stability, with a clear gain seen from InternVL3.5-1B to InternVL3.5-241B (Figure 5).

In addition, we equip InternVL3.5 with a much faster inference speed than its predecessor through two novel methods, namely Visual Resolution Router (ViR) and Decoupled Vision-Language Deployment (DvD). In particular, ViR aims to dynamically select the best trade-off resolution of visual tokens, thus reducing the inference costs with a negligible performance sacrifice. In practice, ViR can be efficiently integrated into InternVL3.5 with a light training stage namely Visual Consistency Learning (ViCO). Furthermore, DvD aims to deploy ViTs and LLMs on separate GPUs to maximize computational parallelism and hardware utilization. These two methods can be seamlessly combined to realize the hardware-friendly implementation for InternVL3.5.

We conduct extensive experiments on public benchmarks to compare InternVL3.5 with existing MLLMs. As

- shown in Figure 1, the InternVL3.5 series consistently maintains a leading position among open source MLLMs in terms of overall score. Compared to the latest commercial model, i.e., GPT-5 [98], InternVL3.5-241B-A28B narrows the gap to 3.9%. In addition, our detailed ablation study demonstrates that InternVL3.5 achieves up to +16.0% improvement in overall reasoning performance and 4.05× speed-up in inference efficiency compared to its predecessor (i.e., InternVL3 [187]). For example, InternVL3.5-8B and InternVL3.5-241B-A28B achieve scores of 73.4 and 77.7, respectively, on the MMMU benchmark [170], showing their strong reasoning capabilities among existing open source MLLMs. In terms of versatility, InternVL3.5 remains competitive against both open-source and closed-source MLLMs in text tasks, GUI tasks, embodied tasks, SVG-based understanding and generation, etc. For example, InternVL3.5-30B-A3B and InternVL3.5-241B-A28B surpass the latest open-source MLLM (Step-3 [129]) by +2.0 and +8.4 in text tasks, respectively.

[Figure 27]

[Figure 28]

[Figure 29]

448×448 Tiles

###### Thumbnail

MLP Projector

[Figure 30]

Qwen3 / GPT-OSS

[Figure 31]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 32]|
|---|

[Figure 33]

[Figure 34]

Pixel Shuffle

[Figure 35]

Vision-Language Connector

Text Tokenizer

InternVL3.5

Input Image (800×1300)

Matching 2:3 (896×1344)

[Figure 36]

InternViT-300M / InternViT-6B Chat Message

[Figure 37]

[Figure 38]

MLP Projector

[Figure 39]

[Figure 40]

[Figure 41]

Single-Image

[Figure 42]

Video

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Pre-defined Aspect Ratios

High Resolution

Low Resolution

|1:1|
|---|

|1:2|
|---|

|1:3|
|---|

[Figure 47]

Top-1

Multi-Image

|1:4|
|---|

|1:5|
|---|

|1:6|
|---|

Visual Resolution Router

|2:3|
|---|

|3:2|
|---|

|...|
|---|

Dynamic High Resolution

InternVL3.5-Flash

(a) Data Preprocessing (c) Connector Architecture

(b) Overall Model Architecture

- Figure 2: Overall architecture. InternVL3.5 adopts the “ViT–MLP–LLM” paradigm as in previous versions. Building upon InternVL3.5, we further introduce InternVL3.5-Flash, which is extended with an additional visual

resolution router (ViR) to dynamically select the appropriate compression rate (e.g., 14 or 161 ) for each image patch. Unlike Dynamic High Resolution which only splits image patches from the perspective of image width

and height, our proposed ViR further introduces adaptivity from the perspective of semantic content.

In summary, our contributions include three folds:

- (1) We release InternVL3.5, the latest family of the InternVL series with advanced reasoning abilities, powerful versatility, and promising efficiency. InternVL3.5 comprises various model scales (from 1B to 241B) with both dense and mixture-of-experts (MoE) models. All of our models and codes are publicly released.
- (2) We propose three innovative designs in InternVL3.5, including cascade reinforcement learning (Cascade RL), visual resolution router (ViR) and decoupled vision-language deployment (DvD). These technologies significantly improve the capabilities and efficiency of InternVL3.5, providing practical tips to the community.
- (3) We conduct extensive experiments and demonstrate that InternVL3.5 exhibits leading performance among open-source MLLMs [46, 126, 138, 154, 164]. Compared to the latest commercial model, i.e., GPT-5 [98], InternVL3.5 even achieves slightly better results on general multimodal capabilities. We believe our approach and open source will further advance the community.

### 2 InternVL3.5

Compared to its predecessors, the InternVL3.5 series achieves superior performance and faster inference. In Section 2.1, we introduce the model architectures of InternVL3.5 and InternVL3.5-Flash. For InternVL3.5-Flash, we further incorporate a Visual Resolution Router (ViR) module that dynamically selects the minimal resolution of visual tokens, achieving better inference efficiency. Section 2.2 and Section 2.3 describe the pre-training and post-training procedures of InternVL3.5, respectively. The details of our proposed Cascade Reinforcement Learning (Cascade RL) and Visual Consistency Learning (ViCO) methods are elaborated in Section 2.3. In

- Section 2.4, we present the test-time scaling approach used to further improve model performance. Finally, in
- Section 2.5, we describe the training and inference infrastructure supporting InternVL3.5, including implementation details of the Decoupled Vision-Language Deployment (DvD) framework. The overall architecture is

- shown in Figure 2, and the training recipes are illustrated in Figure 3.

#### 2.1 Model Architecture

InternVL3.5. We follow the “ViT–MLP–LLM” paradigm adopted in previous versions of InternVL [13, 14, 37, 142, 187]. As shown in Table 1, we initialize the language model using the Qwen3 series [159] and GPT-OSS [99], and the vision encoder using InternViT-300M and InternViT-6B [15]. The Dynamic High Resolution strategy introduced in InternVL1.5 [14] is also retained in our design.

InternVL3.5-Flash. Compared to InternVL3.5, InternVL3.5-Flash further integrates the Visual Resolution Router (ViR), thus yielding a series of efficient variants suitable for resource-constrained scenarios. Specifically, in InternVL3.5, each image patch is initially represented as 1024 visual tokens for the vision encoder, which are then compressed into 256 tokens via a pixel shuffle module before being passed to the Large Language Model (LLM). In InternVL3.5-Flash, as shown in Figure 2, an additional pixel shuffle module with a higher

#Param Vision Language Total Dense Models

Model Vision Encoder Language Model

- InternVL3.5-1B InternViT-300M Qwen3-0.6B 0.3B 0.8B 1.1B
- InternVL3.5-2B InternViT-300M Qwen3-1.7B 0.3B 2.0B 2.3B InternVL3.5-4B InternViT-300M Qwen3-4B 0.3B 4.4B 4.7B InternVL3.5-8B InternViT-300M Qwen3-8B 0.3B 8.2B 8.5B InternVL3.5-14B InternViT-300M Qwen3-14B 0.3B 14.8B 15.1B InternVL3.5-38B InternViT-6B Qwen3-32B 5.5B 32.8B 38.4B

MoE Models

InternVL3.5-20B-A4B InternViT-300M GPT-OSS-20B 0.3B 20.9B 21.2B (A4B) InternVL3.5-30B-A3B InternViT-300M Qwen3-30B-A3B 0.3B 30.5B 30.8B (A3B) InternVL3.5-241B-A28B InternViT-6B Qwen3-235B-A22B 5.5B 235.1B 240.7B (A28B)

- Table 1: Pre-trained models used in the InternVL3.5 series. “A” denotes the number of activated parameters.

compression rate is included, enabling compression of visual tokens down to 64 tokens. For each patch, the patch router determines the appropriate compression rate by assessing its semantic richness, and routes it to the corresponding pixel shuffle module accordingly. Benefiting from this patch-aware compression mechanism, InternVL3.5-Flash is able to reduce the number of visual tokens by 50% while maintaining nearly 100% of the performance of InternVL3.5, as shown in Section 3.15.

#### 2.2 Pre-Training

Training Objective. During the pre-training stage, we update all model parameters jointly using the combination of large-scale text and multimodal corpora. Specifically, given an arbitrary training sample consisting of a multimodal token sequence x = (x1,x2,...,xL), the next token prediction (NTP) loss [103] is calculated on each text token as follows:

Li = −log pθ (xi | x1,...,xi−1), (1)

where xi is the predicted token and prefix tokens in {x1,x2,...,xi−1} can be either text tokens or image tokens. In particular, for conversation samples, only response tokens are included for the loss calculation. Additionally, to mitigate bias toward either longer or shorter responses during training, we adopt the square averaging [13] to reweight the NTP loss as follows:

wi j wj · Li, wi =

′

L

i =

1 N0.5

, (2)

where N denotes the number of tokens in the training sample on which the loss needs to be calculated. The random JPEG compression [13] is also included to enhance the model’s real-world performance.

Data. The pre-training corpora can be classified into two categories: (1) Multimodal data: this subset of data is mainly sourced from the training corpora of InternVL3 [187], covering a diverse range of domains such as image captioning, general question answering, mathematics, scientific disciplines, charts, optical character recognition (OCR), knowledge grounding, document understanding, multi-turn dialogue, and medical data. (2) Text-only data: this part of data is constructed based on the training corpora of InternLM series [9, 124] and is further augmented with open-source datasets [6, 73, 75, 94]. The pre-training corpora contains approximately 116M samples, corresponding to about 250B tokens. The ratio between text-only and multimodal data is approximately 1 : 2.5. The maximum sequence length is set to 32K tokens to adapt long-context understanding and reasoning.

#### 2.3 Post-Training

After the pre-training stage, we adopt a three-stage post-training strategy comprising: (1) Supervised Fine-Tuning (SFT), which maintains the same training objective as pre-training but leverages higher-quality conversation data to further enhance the model’s capabilities. (2) Cascade Reinforcement Learning (Cascade RL), which combines the benefits of offline and online RL methods to facilitate the reasoning capabilities. (3) Visual Consistency Learning (ViCO), which aims to integrate visual resolution router (ViR) into InternVL3.5 to construct InternVL3.5-Flash, by minimizing the output divergence of different visual compression rates.

Cascade RL ViCO

Native Pre-training

Supervised Fine-tuning

MPO GSPO

Consistency Training Router Training

~250B tokens ~130B tokens

270K samples ~30B tokens

[Figure 48]

[Figure 49]

###### InternVL3.5 InternVL3.5-Flash

- Figure 3: Training recipes of InternVL3.5. InternVL3.5 consists of three training stages: (1) native pre-training for vision-language alignment, (2) supervised fine-tuning for adaptation to downstream tasks, (3) Cascade RL for improvement on reasoning capabilities. InternVL3.5-Flash is an efficient version of InternVL3.5, which further integrates a visual resolution router (ViR) through consistency training and router training.

Supervised Fine-Tuning. During the SFT phase, we adopt the same objective as in the pre-training stage and use the square averaging strategy [13] to calculate the final loss. In this stage, the context window is set to 32K tokens to adapt long-context information. Compared to InternVL3, the SFT stage of InternVL3.5 contains more high-quality and diverse training data derived from three sources: (1) Instruction-following data from InternVL3, which are reused to preserve broad coverage of vision–language tasks. (2) Multimodal reasoning data in the “Thinking” mode, which are included to instill long-thinking capabilities in the model. To construct such data, we leverage a large-scale reasoning model to sample rollouts with detailed reasoning processes. In addition to validating whether answers are factually correct, we implement strict filtering measures for the reasoning processes themselves: this includes evaluating how clear the thinking is, weeding out redundancy, and ensuring that formatting remains consistent. The questions in these datasets cover various expert domains, such as mathematics and scientific disciplines, thereby strengthening performance on different reasoning tasks. (3) Capability-expansion datasets, which endow InternVL3.5 with new skills, including GUI-based interaction, embodied interaction, and scalable vector graphics (SVG) understanding and generation.

Cascade Reinforcement Learning. Compared to Pre-training and Supervised Fine-tuning (SFT), the core advantage of RL lies in its ability to introduce negative samples, which prune low-quality regions in the model’s output space and thereby enhance the overall response quality. As a derivative of the PPO algorithm [108], DPO [104] enables training based on existing rollouts, which we also regard as a form of offline RL. Offline RL algorithms [104, 142] often offer a higher training efficiency, but their performance ceiling is generally lower compared to online RL methods. In contrast, despite the effectiveness of online RL algorithms [108, 109, 167, 184], they are often computationally expensive and time-consuming. In this work, we propose Cascade RL, which aims to combine the benefits of offline RL and online RL to progressively facilitate the post-training of MLLMs in an efficient manner. Specifically, we first fine-tune the model using an offline RL algorithm [142] as an efficient warm-up stage to reach satisfied results, which can guarantee high-quality rollouts for the latter stage. Subsequently, we employ an online RL algorithm [183] to further refine the output distribution based on rollouts generated by the model itself. Compared to the single offline or online RL stage, our cascaded RL achieves significant performance improvements at a fraction of the GPU time cost.

During the offline RL stage, we employ mixed preference optimization (MPO) [142] to fine-tune the model. Specifically, the training objective of MPO is a combination of preference loss Lp, quality loss Lq, and generation loss Lg, which can be formulated as follows:

LMPO = wpLp + wqLq + wgLg, (3)

where w∗ represents the weight assigned to each loss component. The DPO loss [104], BCO loss [53], and LM loss [8] serve as the preference loss, quality loss, and generation loss, respectively.

During the online RL stage, we employ GSPO [183], without reference model constraints, as our online RL algorithm, which we find more effective in training both dense and mixture-of-experts (MoE) models. Similar to GRPO [109], the advantage is defined as the normalized reward across responses sampled from the same query:

r (x,yi) − mean {r (x,yi)}Gi=1 std {r (x,yi)}Gi=1

, (4)

Ai =

where yi is the i-th response generated for the query x, G is the total number of generated responses to the query, and r (x,yi) denotes the reward for this response. The training objective of GSPO is given by:

G

1 G

min si(θ) Ai,clip(si(θ),1 − ε,1 + ε) Ai , (5)

LGSPO(θ) = Ex∼D,{y

i}Gi=1∼πθ old (·|x)

i=1

where the importance sampling ratio is defined as the geometric mean of the per-token ratios:

 , (6)

  1

|yi|

1 |yi|

πθ (yi,t | x,yi,<t) πθ

πθ (yi | x) πθ

log

= exp

si(θ) =

(yi | x)

|yi|

(yi,t | x,yi,<t)

old

old

t=1

where πθ (yi | x,yi,<t) and πθ (yi,t | x,yi,<t) denote the generation probability of response yi and token yi,t under the policy model with parameters θ, respectively.

Compared to directly training the model with a single RL paradigm, Cascade RL offers the following advantages:

- (1) Better training stability: In the offline RL stage, the rollout collection and parameter updates are decoupled, effectively mitigating issues such as reward hacking. During the online RL stage, we empirically observe that stronger models exhibit more stable and robust training dynamics. As a result, the performance gains achieved in the MPO stage further enhance the stability of the GSPO stage and reduce sensitivity to the algorithm.
- (2) Improved training efficiency: In the MPO stage, rollouts can be shared across different models, amortizing the sampling cost typically incurred during online RL. (3) Higher performance ceiling: Moreover, as shown in Section 3.15, models fine-tuned with MPO take fewer training steps to achieve higher performance in the subsequent online RL phase, further reducing training overhead.

Visual Consistency Learning. We further include ViCO as an additional training stage to integrate the visual resolution router (ViR) into InternVL3.5, thereby reducing the inference cost of InternVL3.5. The obtained efficient version of InternVL3.5 are termed as InternVL3.5-Flash. In particular, ViCO comprises two stages:

- (1) Consistency training: In this stage, the entire model is trained to minimize the divergence between response distributions conditioned on visual tokens with different compression rates. In practice, we introduce an extra reference model, which is frozen and initialized with InternVL3.5. Given a sample, each image patch is represented as either 256 or 64 tokens, and the training objective is defined as follows:

LViCO = Eξ∼R

1 N

N

i=1

KL πθ

ref

(yi | y<i,I) πθ

policy

(yi | y<i,Iξ) , (7)

where KL denotes the KL divergence and ξ denotes the compression rate, which is uniformly sampled from {14, 161 }. The image Iξ is represented as 256 tokens when ξ = 14 and 64 tokens when ξ = 161 . We note that the reference model always performs inference with ξ = 14.

- (2) Router training: This stage aims to train the ViR to select an appropriate trade-off resolution for different inputs. ViR is formulated as a binary classifier and trained using standard cross-entropy loss. To construct the route targets, we first compute the KL divergence between the model outputs conditioned on uncompressed visual tokens (i.e., 256 tokens per patch) and those conditioned on compressed visual tokens (i.e., 64 tokens per patch). During this stage, the main MLLM (ViT, MLP and LLM) is kept frozen, and only the ViR is trained. Specifically, we first compute the loss ratio for each patch:

LViCO yi | I 1

, (8)

16

ri =

LViCO yi | I1

4

which quantifies the relative increase in loss caused by compressing the visual tokens. Based on this ratio, the binary ground-truth label for the patch router is defined as:

- 0, ri < τ (compression has negligible impact)
- 1, ri ≥ τ (compression has significant impact),

yirouter =

(9)

where yirouter = 0 and yirouter = 1 indicate that the compression rate ξ is set to 161 and 14, respectively. During training, we store the historical ri values of a sliding window, and τ is a dynamical threshold computed from the k-th percentile of historical ri values. In practice, the target distribution is balanced. During the consistency training stage, all patches of the same image are represented with a random compression rate, in order to ensure that the model retains its capability when no compression is applied. As shown in Section 3.15, InternVL3.5-Flash reduces 50% of the visual tokens while maintaining nearly 100% of the original performance.

Data. For the supervised fine-tuning (SFT) stage, the datasets comprise approximately 56 million samples, which corresponds to around 130 billion tokens. The proportion of text-only data to multimodal data is roughly 1:3.5. For the cascade reinforcement learning stage, we use MMPR-v1.2 [142] as the training data for offline RL, which contains about 200K sample pairs. Based on MMPR-v1.2, we compute the accuracy of each query using the provided rollouts and select those whose model accuracy falls between 0.2 and 0.8 for online RL. We further extend the dataset with recent multimodal datasets [70, 83, 90, 135, 163] to enhance diversity. The resulting dataset, termed MMPR-Tiny, consists of approximately 70K queries. We directly reuse the rollouts from MMPR-v1.2 for both offline RL and data filtering in online RL, thereby reducing the cost of sampling additional rollouts.

For the ViCO stage, we primarily leverage datasets identical to the SFT stage during consistency training, ensuring that the model retains its original performance. During router training, we use a subset of the SFT data, primarily composed of OCR and VQA examples, which are rich in visual information and sometimes require high-resolution understanding. This enables the resolution router to learn how to dynamically decide whether each image patch can be compressed based on the visual information.

#### 2.4 Test-Time Scaling

Test-time scaling (TTS) has been empirically demonstrated as an effective approach to enhance the reasoning capabilities of LLMs and MLLMs, particularly for complex tasks that require multi-step inference [65, 82, 113, 143, 179]. In this work, we implement a comprehensive test-time scaling approach that simultaneously improves reasoning depth (i.e., deep thinking) and breadth (i.e., parallel thinking). We note that unless otherwise specified, the experimental results reported in Section 3 are obtained without applying TTS. Thus far, we have only applied TTS to reasoning benchmarks, since we found that the model already exhibits strong perception and understanding capabilities, and initiating TTS yields no significant improvement.

Deep Thinking. By activating the Thinking mode, we guide the model to deliberately engage in step-by-step reasoning (i.e., decomposing complex problems into logical steps and validating intermediate conclusions) prior to generating the final answer. This approach systematically improves the logical structure of solutions for complex problems, particularly those requiring multi-step inference, and enhances reasoning depth.

Parallel Thinking. Following InternVL3, for reasoning tasks, we adopt the Best-of-N (BoN) strategy by employing VisualPRM-v1.1 [143] as the critic model to select the optimal response from multiple reasoning candidates. This approach improves the reasoning breadth.

#### 2.5 Infrastructure

Training Framework. The model training is conducted mainly based on the XTuner framework [21], which incorporates a series of optimization strategies tailored for LLM and MoE training. These include fully shared data parallelism (FSDP) [182] to partition model parameters across GPUs, data packing [13] to reduce padding tokens while balancing the token computation load across ranks for improved training efficiency, FP8 training based on DeepGEMM [68] and liger-kernel’s fused cross-entropy operator [47] to accelerate the training process, FlashAttention-3 [26, 27] to support packed inputs and speed up attention computation, and the TMA-Adaptive FP8 Grouped GEMM kernel [118] to optimize the training of MoE models. For the online stage, we use verl [111] as our codebase. For InternVL3.5-20B-A4B, we implement an accelerated version of window attention with sink in GPT-OSS-20B through Triton [130].

Decoupled Vision-Language Deployment. In multimodal inference, the vision encoder and language model have distinct computational characteristics. The vision encoder that transforms images into semantic features is highly parallelizable and does not rely on long-term history state. In contrast, the language model adopts the inference in an autoregressive manner, which requires previous states to compute the next one. This sequential property makes the language part more sensitive to memory bandwidth and latency. When MLLMs are deployed online at scale, the vision and language models often block each other, thus incurring additional inference cost. This effect becomes more pronounced with larger vision models or higher-resolution images.

As shown in Figure 4, we propose Decoupled Vision-Language Deployment (DvD) to address this issue by separating vision and language processing, with a particular focus on optimizing the prefilling stage. The vision subsystem batches and processes images to produce compact feature embeddings, which are then transmitted to the language subsystem for fusion with the text context prior to decoding. This separation alleviates blocking and brings multimodal prefilling performance closer to that of pure language models. In our system implementation, the ViT and MLP (and ViR for InternVL3.5-Flash) are deployed on the vision server, while the language server

Input Message

Mixed

LLM Response

ViT & MLP

Computational pipeline (time)

Shared Server for Vision and Language

(a) Vanilla Deployment

ViT & MLP

Image

Vision

Vision Server Language Server

BF16

Input Message

Communication

Text

LLM

Response

Language

Computational pipeline (time)

(b) Decoupled Vision-Language Deployment

- Figure 4: Overview of Decoupled Vision-Language Deployment. DvD decouples the vision and language models and deploys them on separate servers. The right side shows a time-consumption trace of the pipeline. (a) In the original deployment, the ViT, MLP, and LLM are executed sequentially. Given their substantial differences in size and computation patterns, this serial design significantly slows down inference. (b) With DvD, the inference of the ViT and the LLM is performed in parallel and asynchronously. Thus, ViT’s computations can overlap with the LLM’s prefilling and decoding, reducing resource conflicts and improving inference speed.

executes only the LLM. The communication is unidirectional, transmitting BF16 visual features over TCP, with RDMA optionally employed to achieve a higher transmission speed. Vision processing, feature transmission, and language processing are organized into an asynchronous three-stage pipeline, enabling overlapped execution and minimizing pipeline stalls.

DvD increases GPU utilization and processing efficiency on the vision side, while enabling the language server to focus exclusively on the LLM’s prefilling and decoding without being blocked by vision computation. This design leads to improved throughput and responsiveness. Moreover, the architecture supports independent hardware cost optimization for the vision and language modules, and facilitates the seamless integration of new modules without requiring modifications to the language server deployment.

### 3 Experiments

Here, we first compare the overall performance of InternVL3.5 with recent leading multimodal large language models (MLLMs) in Section 3.1. Subsequently, we evaluate our models in various domains, including multimodal reasoning and mathematics (Section 3.2), optical character recognition (OCR), chart and document understanding (Section 3.3), multi-image understanding (Section 3.4), real-world comprehension (Section 3.5), comprehensive multimodal evaluation (Section 3.6), multimodal hallucination evaluation (Section 3.7), visual grounding (Section 3.8), multimodal multilingual understanding (Section 3.9), video understanding (Section 3.10), GUI (Section 3.11), embodied (Section 3.12), and SVG (Section 3.13) tasks, most of which were tested using VLMEvalKit [31]. Additionally, the evaluation of the language capabilities of InternVL3.5 is

- presented in Section 3.14. Finally, we ablate newly proposed designs in InternVL3.5, including the Cascade Reinforcement Learning, Visual Resolution Router, and Decoupled Vision-Language Deployment (Section 3.15).

#### 3.1 Overall Comparison with Other Advanced MLLMs

- Table 2 presents a comprehensive evaluation of InternVL3.5’s performance across 35 benchmarks categorized into four key multimodal task types: (1) General Tasks: MMStar [11], MMVet [168], MMBench V1.1 (en) [71], MTVQA [119], AI2D [55], OCRBench [72], WildVision [78], MME-RealWorld (en) [178], HallusionBench [41], MVBench [63], VideoMME [35], MLVU [186], (2) Reasoning Tasks: MMMU [170], MathVista [76], MathVision [76], MathVerse [175], DynaMath [189], WeMath [100], OlympiadBench [43], LogicVista [153]; (3) Text-Centric Tasks: MATH500 [65], AIME24 [84], AIME25 [85], GPQA [106],

###### Task Benchmark InternVL3.5 InternVL3.5 GLM-4.1V Kimi-VL-2506 Qwen2.5-VL GLM-4.5V Step-3 GPT-5 Size – 30B-A3B 241B-A28B 9B 16B-A3B 72B 106B-A12B 321B-A38B –

MMStar 72.0 77.9 72.9 70.4 70.8 75.3 69.0* 75.7* MMVet 85.5 81.2 66.4* 78.1 76.2 75.2* 79.4* 77.6* MMBench V1.1 (en) 84.8 87.4 85.8 84.4 88.4 88.2 81.1* 88.6* MTVQA 33.7 39.3 25.5* 27.2* 31.7 30.5* 30.6* 33.1* AI2D (w/ mask) 86.8 87.3 87.9 81.9† 88.7 88.1 83.7* 89.5* OCRBench 88.0 90.7 84.2 86.9 88.5 87.2 83.7* 80.7* WildVision 75.8 82.8 74.0* 64.8* 78.6* 79.0* 89.4* 77.4* MME-RealWorld (en) 64.8 65.1 61.7* 54.5* 63.2* 61.7* 54.0* 68.0* HallusionBench 53.8 57.3 63.2 59.8† 55.2 65.4 64.2 65.2* MVBench 72.1 76.5 68.4 59.7† 70.4 73.0 64.2* 74.0* VideoMME (w/o sub) 68.7 72.9 68.2 67.8 73.3 74.6 63.6* 81.8* MLVU 73.0 78.2 71.5* 74.2 74.6 75.3* 62.2* 77.3* LongVideoBench 63.8 67.1 65.7* 64.5 60.7 68.8* 57.7* 72.6* Overall 71.0 74.1 68.9 67.2 70.8 72.5 67.9 74.0

General

MMMU (val) 75.6 77.7 68.0 64.0 68.2‡ 75.4 74.2 84.2 MathVista 80.9 82.7 80.7 80.1 74.2‡ 84.6 79.2* 81.9* MathVision 55.7 63.9 54.4 54.4† 39.3‡ 65.6 64.8 72.0* MathVerse (vision-only) 60.4 68.5 68.4 54.6† 47.3‡ 72.1 62.7* 81.2* DynaMath 36.5 46.5 42.5 28.1† 35.9‡ 53.9 50.1 60.9* WeMath 48.4 62.3 63.8 42.0† 49.1‡ 68.8 59.8* 71.1* OlympiadBench 62.9 68.7 56.3* 47.4† 37.8* 64.0* 66.8* 73.2* LogicVista 55.7 66.7 60.4 51.4† 55.7‡ 62.4 60.2* 70.0* Overall 59.5 67.1 61.8 52.8 50.9 68.4 64.7 74.3

Reasoning

MATH500 96.6 98.0 81.8* 91.8* 82.8* 94.2* 85.6* 97.8*

- AIME24 79.4 84.7 36.2* 54.0* 15.0* 80.1* 86.6* 90.0*

- AIME25 62.7 75.6 32.0* 39.1* 13.3* 72.8* 82.9 94.6 GPQA 68.2 73.2 50.3* 42.3* 52.0* 56.6* 73.0 85.7 MMLU-Pro 75.3 81.3 57.1* 68.5* 51.1* 69.7* 58.6* 85.6* C-Eval 83.2 90.9 72.3* 64.4* 88.2* 89.1* 84.7* 88.2* GAOKAO 91.9 94.5 78.4* 72.6* 92.9* 93.1* 70.2* 94.1* IFEval 74.3 83.7 71.5* 65.8* 83.9* 82.4* 73.4* 94.6* Overall 78.9 85.3 60.0 62.3 59.9 79.8 76.9 91.3

Text

SGP-Bench 69.4 70.7 57.1* 44.9* 57.1* 66.1* 56.5* 77.5* ScreenSpot 86.6 89.8 – – 87.1 – – – ScreenSpot-v2 87.3 92.9 – 91.4 – – – – OSWorld-G 42.4 53.2 – 52.5 – – – – VSI-Bench 63.7 69.5 39.2* 37.4* 36.1* 41.4* 34.2* 37.5* ERQA 41.5 46.8 45.8† 36.0† 44.8† 46.5† 44.5† 65.7* SpaCE-10 45.5 55.0 43.4* 39.2* 37.9* 51.6* 42.6* 43.8* OmniSpatial 48.1 51.9 47.7† 37.3† 47.9† 51.0† 47.0† 59.6* Overall 60.6 66.2 – – – – – –

Agentic

#### Table 2: The overall comparison of InternVL3.5 series and existing open-source and closed-source MLLMs.

*: reproduced through VLMEvalkit [31]. †: reported by GLM-4.5V [46]. ‡: reported by OpenCompass [20].

MMLU-Pro [61], GAOKAO [177], IFEval [185]; (4) Agentic Tasks: SGP-Bench [102], ScreenSpot [16], ScreenSpot-v2 [150], OSWorld-G [156], VSI-Bench [161], ERQA [121], SpaCE-10 [38].

We report results of our flagship models (InternVL3.5-30B-A3B and InternVL3.5-241B-A28B) and frontier open-source MLLMs (GLM-4.1V [46], Kimi-VL-A3B-2506 [125], GLM-4.5V [46], Qwen2.5-VL-72B [5] and Step-3 [129]). We also include a state-of-the-art closed-source MLLM (GPT-5 [98]) for comparison.

These results highlight InternVL3.5’s strong capabilities across diverse tasks. For general multimodal tasks, InternVL3.5 demonstrates leading performance among open-source models on general multimodal understanding (e.g., MMVet), multilingual (e.g., MTVQA), OCR (e.g., OCRBench), real-world (e.g., MME-RealWorld), and video (e.g., LongVideoBench) benchmarks. It even achieves a similar overall score as GPT-5 [98] (74.1 vs. 74.0), the state-of-the-art closed-source MLLM. Nevertheless, some tasks like HallusionBench still pose challenges, indicating the need for further refinements.

We also observe particularly significant gains in complex multimodal reasoning, as evidenced by the scores of 77.7 on MMMU and 82.7 on MathVista, which surpass most open-source models and approaching top-tier commercial systems. These improvements are largely driven by enhanced training strategies, especially Cascade RL, and refined test-time scaling methodologies, enabling robust generalization in challenging domains such as mathematical reasoning (e.g., MathVerse) and multidisciplinary understanding (e.g., MTVQA).

For text-related tasks, InternVL3.5 outperforms most open-source models with significant margins. This is primarily due to our native pre-training strategy introduced in InternVL3 [187], where we include a large amount of text-only data in the training process. This strategy allows the model to simultaneously acquire linguistic and multimodal abilities in a more efficient and integrated manner, and preserve the language capabilities of the pre-trained LLM to avoid the catastrophic forgetting issue [80]. As a result, InternVL3.5 achieves high performance on both general (e.g., IFEval) and reasoning (e.g., AIME24 and MMLU-Pro) benchmarks and narrows the gap with GPT-5.

We also demonstrate the versatility of InternVL3.5 through agentic benchmarks. On SGP-Bench, an SVG understanding benchmark, InternVL3.5 achieves leading performance (69.4 and 70.7) that surpasses all opensource models. For GUI tasks, our models show strong abilities on GUI grounding (ScreenSpot) and online agentic (OSWorld-G) benchmarks. Evaluation on embodied tasks (VSI-Bench, ERQA, SpaCE-10, OmniSpatial) validates InternVL3.5’s spatial reasoning skills and the competence to understand complex and dynamic environments, highlighting its potential in embodied agents, robotic navigation, and interactive scene perception.

#### 3.2 Multimodal Reasoning and Mathematics

To comprehensively evaluate the multimodal reasoning and mathematical capabilities of InternVL3.5, we conduct extensive experiments across a series of benchmarks, including MMMU [170] for multidisciplinary reasoning, MathVista [76], MathVision [134], and MathVerse [175] for mathematical reasoning, as well as DynaMath [189], WeMath [100], and LogicVista [153] for complementary logical reasoning assessment.

- As shown in Table 3, InternVL3.5 achieves state-of-the-art performance across all evaluated benchmarks among open-source models. Compared with the previous generation InternVL3, InternVL3.5 improves reasoning performance by more than 10 points across all model sizes relative to their counterparts of the comparable scale. Furthermore, InternVL3.5-241B-A28B consistently outperforms all open-source counterparts, obtaining an overall average score of 66.9. It is followed by InternVL3.5-38B, which secures the second highest score with an average of 66.0. At mid-scale, the performance of InternVL3.5 is particularly impressive, with InternVL3.530B-A3B and InternVL3.5-14B attaining scores of 75.6 and 73.3 on MMMU respectively, both outperforming the larger InternVL3-78B (72.2). At lightweight scales, InternVL3.5 achieves substantial gains over open-source baselines. Compared to its predecessor InternVL3, it delivers marked improvements: the InternVL3.5-2B model has an average score of 50.7, significantly higher than that of InternVL3-2B with a score of 32.4; the InternVL3.5-4B model, with a score of 57.4, far exceeds the score of 33.5 of MiniCPM-V-4; and the InternVL3.5-8B model achieves a score of 60.3, substantially surpassing the score of 44.3 of InternVL3-8B. These significant improvements are mainly from our Cascade RL, showing its strong scalability for reasoning tasks. The ablation study about how different training stages influence the reasoning abilities of our models is

- presented in Section 3.15.

Furthermore, our experiments also demonstrate that Cascade RL can be seamlessly combined with parallel thinking and obtain further gains. For instance, with parallel thinking, the overall reasoning scores of InternVL3.5-4B, InternVL3.5-8B and InternVL3.5-241B-A28B are further improved by +2.6%, +2.1% and +1.8%, respectively, highlighting the effectiveness of test-time scaling for reasoning-related tasks.

|Model<br><br>|MMMU (val)<br><br>MathVista (mini)<br><br>MathVision<br><br>MathVerse (vision-only)<br><br>DynaMath (worst case)<br><br>WeMath LogicVista|Overall|
|---|---|---|
|InternVL3-1B [187] InternVL3.5-1B w/ Parallel Thinking [143]<br><br>|43.4 45.8 18.8 18.7 5.8 13.4 29.8<br><br>44.2 59.3 27.3 37.8 17.2 21.5 29.3 51.0 69.5 33.4 45.8 25.0 36.4 41.8<br><br><br>|25.1 33.8 43.3<br><br>|
|Ovis-2B [77] Qwen2.5-VL-3B [5] InternVL3-2B [187] InternVL3.5-2B w/ Parallel Thinking [143]<br><br>|45.6 64.1 17.7 29.4 10.0 9.9 34.7 51.2 61.2 21.9 31.2 13.2 22.9 40.3 48.6 57.0 21.7 25.3 14.6 22.4 36.9 59.0 71.8 42.8 53.4 31.5 48.5 47.7 64.8 74.3 49.0 54.2 33.3 52.6 49.4<br><br>|30.2 34.6 32.4 50.7 53.9<br><br>|
|Ovis-4B [77] MiniCPM-V-4-4B [164] InternVL2.5-4B [13] InternVL3.5-4B w/ Parallel Thinking [143]<br><br>|49.0 69.6 21.5 38.5 18.0 16.9 35.3 51.2 66.9 20.7 18.3 14.2 32.7 30.6 51.8 64.1 18.4 27.7 15.2 21.2 34.2 66.6 77.1 54.4 61.7 35.7 50.1 56.4 71.4 79.2 57.5 60.0 38.3 53.0 60.4<br><br>|35.5 33.5 33.2 57.4 60.0<br><br>|
|MiniCPM-o2.6 [164] Ovis-8B [77] Qwen2.5-VL-8B [5] MiMo-VL-RL-8B [154] Keye-VL-8B [126] GLM-4.1V-9B [46] InternVL3-8B [187] InternVL3.5-8B w/ Parallel Thinking [143]<br><br>|50.9 73.3 21.7 35.0 10.4 25.2 36.0 57.4 71.8 25.9 42.3 20.4 27.2 39.4 55.0 67.8 25.4 41.1 21.0 35.2 44.1 66.7 81.5 60.4 71.5 45.9 66.3 61.4 71.4 80.7 50.8 54.8 37.3 60.7 54.8 68.0 80.7 54.4 68.4 42.5 63.8 60.4 62.7 71.6 29.3 39.8 25.5 37.1 44.1 73.4 78.4 56.8 61.5 37.7 57.0 57.3 73.4 80.8 59.9 62.6 39.9 57.6 62.6<br><br>|36.1 40.6 41.4 64.8 58.6 62.6 44.3 60.3 62.4<br><br>|
|Gemma-3-12B [122] Ovis2-16B [77] InternVL3-14B [187] InternVL3.5-14B w/ Parallel Thinking [143]<br><br>|55.2 56.1 30.3 21.1 20.8 33.6 41.2 60.7 73.7 30.1 45.8 26.3 45.0 47.4 67.1 75.1 37.2 44.4 31.3 43.0 51.2<br><br>73.3 80.5 59.9 62.8 38.7 58.7 60.2<br><br>74.3 81.5 61.4 64.1 41.3 60.8 61.3<br><br><br>|36.9 47.0 49.9 62.0 63.5<br><br>|

|Kimi-VL-A3B-2506 [125] InternVL3.5-20B-A4B w/ Parallel Thinking [143] InternVL3.5-30B-A3B w/ Parallel Thinking [143]<br><br>|64.0 80.1 54.4 54.6 28.1 42.0 51.4 72.6 78.0 53.0 57.0 33.1 41.4 56.8<br><br>74.0 79.2 55.7 58.9 35.9 45.5 58.2<br><br>75.6 80.9 55.7 60.4 36.5 48.4 55.7 75.0 82.1 58.5 61.4 38.3 57.5 59.7<br><br><br>|53.5 56.0 58.2 59.0 61.8<br><br>|
|---|---|---|
|Gemma-3-27B [122] Ovis2-34B [77] Qwen2.5-VL-32B [5] Skywork-R1V3-38B [110] InternVL3-38B [187] InternVL3.5-38B w/ Parallel Thinking [143]<br><br>|64.9 59.8 39.8 34.0 28.5 37.9 47.3 66.7 76.1 31.9 50.1 27.5 51.9 49.9 70.2 74.8 38.1 57.6 35.1 46.5 52.6 76.0 77.1 52.6 59.6 35.1 56.5 59.7 70.1 75.1 34.2 48.2 35.3 48.6 58.4 76.9 81.9 63.7 67.6 41.7 64.8 65.3 76.7 83.8 65.6 66.5 44.9 67.8 64.7<br><br>|44.6 50.6 53.6 59.5 52.8 66.0 67.1<br><br>|
|GPT-5-nano-20250807 [98] GPT-5-20250807 [98] Claude-3.7-Sonnet [2] Gemini-2.0-Pro [29] Gemini-2.5-Pro [29] Doubao-1.5-Pro [42] GLM-4.5V [46] QvQ-72B-Preview [127] Qwen2.5-VL-72B [5] Step3-321B-A38B [129] InternVL3-78B [187] InternVL3.5-241B-A28B w/ Parallel Thinking [143]<br><br>|72.6 73.1 59.7 66.6 47.9 59.4 57.5 81.8 81.9 72.0 81.2 60.9 71.1 70.0 75.0 66.8 41.9 46.7 39.7 49.3 58.2<br><br>69.9 71.3 48.1 67.3 43.3 56.5 53.2<br><br>74.7 80.9 69.1 76.9 56.3 78.0 73.8<br><br>73.8 78.6 51.5 64.7 44.9 65.7 64.2<br><br>75.4 84.6 65.6 72.1 53.9 68.8 62.4<br><br><br>70.3 70.3 34.9 48.2 30.7 39.0 58.2 68.2 74.2 39.3 47.3 35.9 49.1 55.7<br><br><br>74.2 79.2 64.8 62.7 50.1 59.8 60.2 72.2 79.0 43.1 51.0 35.1 46.1 55.9<br><br><br>77.7 82.7 63.9 68.5 46.5 62.3 66.7<br><br>78.7 84.8 65.9 71.6 47.8 64.4 67.6<br><br><br>|62.4 74.1 53.9 58.5 72.8 63.3 69.0 50.2 52.8 64.4 54.6 66.9 68.7<br><br>|

- Table 3: Comparison of multimodal reasoning and mathematical performance. MMMU [170] is a multidisciplinary reasoning benchmark. MathVista [76], MathVision [134], MathVerse [175], DynaMath [189], and WeMath [100] are mathematics benchmarks. LogicVista [153] is a logical reasoning benchmark. Part of the results are collected from other papers [46, 126, 138, 154, 187] and the OpenCompass leaderboard [20]. The overall score is the average score of all benchmarks.

#### 3.3 OCR, Chart, and Document Understanding

To evaluate the comprehensive capabilities of the model across tasks related to text, document, and chart comprehension, we conduct an extensive assessment on nine benchmarks: AI2D [55], ChartQA [87], TextVQA [112], DocVQA [89], InfoVQA [88], OCRBench [72], SEED-2-Plus [59], CharXiv [148], and VCR [176].

- As shown in Table 4, InternVL3.5 achieves competitive results on these benchmarks, outperforming other open-source and closed-source models. At the lightweight scale, InternVL3.5 demonstrates significant potential. For instance, InternVL3.5-2B attains an overall average score of 76.7 across nine benchmarks, surpassing InternVL3-2B of similar size, which scores 74.7. Specifically, on DocVQA, InfoVQA, and SEED-2-Plus, InternVL3.5-2B achieves scores of 89.4, 70.8, and 68.0, respectively, compared to 88.3, 66.1, and 64.6 for InternVL3-2B, highlighting InternVL3.5’s strong vision-language understanding capabilities.

|Model<br><br>|AI2D (w / wo M)<br><br>ChartQA (test avg)<br><br>TextVQA (val)<br><br>DocVQA (test)<br><br>InfoVQA (test)<br><br>OCR Bench<br><br>SEED-2 Plus<br><br>CharXiv (RQ / DQ)<br><br>VCR-EN-Easy (EM / Jaccard)|Overall|
|---|---|---|
|InternVL3-1B [187] InternVL3.5-1B<br><br>|69.4 / 78.3 75.3 74.1 81.9 53.7 790 58.2 21.0 / 47.1 89.3 / 96.2 71.1 / 81.8 77.7 71.5 85.6 60.5 795 62.3 26.9 / 60.6 83.5 / 94.0<br><br>|68.6 71.3<br><br>|
|Qwen2-VL-2B [138] Aquila-VL-2B [40] Qwen2.5-VL-3B [5] InternVL3-2B [187] InternVL3.5-2B<br><br>|74.7 / 84.6 73.5 79.7 90.1 65.5 809 62.4 – 81.5 / –<br><br>75.0 / – 76.5 76.4 85.0 58.3 772 63.0 – 70.0 / –<br><br><br>81.6 / – 84.0 79.3 93.9 77.1 797 67.6 31.3 / 58.6 –<br><br>78.7 / 87.4 80.2 77.0 88.3 66.1 835 64.6 28.3 / 54.7 91.2 / 96.9<br><br>78.8 / 89.1 80.7 76.5 89.4 70.8 836 68.0 31.6 / 65.0 90.1 / 96.4<br><br><br>|– – – 74.7 76.7<br><br>|
|MiniCPM-V-4-4B [164] InternVL3.5-4B<br><br>|80.9 / 91.4 73.0 81.4 94.0 67.0 862 67.0 31.9 / 56.4 80.9 / 90.1 82.6 / 92.3 86.0 77.9 92.4 78.0 822 69.4 39.6 / 71.1 91.6 / 97.0<br><br>|75.0 80.0<br><br>|
|Ovis1.6-Gemma2-9B [77] MiniCPM-V2.6-8B [164] Molmo-7B-D [30] Qwen2-VL-7B [138] Qwen2.5-VL-7B [5] Keye-VL-8B [126] GLM-4.1V-9B [46]<br><br>InternVL3-8B [187]<br>InternVL3-9B [187] InternVL3.5-8B<br><br><br>|84.4 / – – – – – 830 – – –<br><br>82.1 / – 82.4 80.1 90.8 – 852 65.7 31.0 / 57.1 73.9 / 85.7<br><br>– / 93.2 84.1 81.7 92.2 72.6 694 – – –<br><br>83.0 / 92.1 83.0 84.3 94.5 76.5 866 69.0 – 89.7 / 93.8<br><br>83.9 / – 87.3 84.9 95.7 82.6 864 70.4 42.5 / 73.9 –<br><br>85.8 / 88.5 72.5 75.7 87.0 63.0 853 67.8 36.8 / 75.2 –<br><br>82.2 / 87.0 70.0 79.6 93.3 80.3 823 71.8 53.4 / 82.4 32.7 / 55.2 85.2 / 92.6 86.6 80.2 92.7 76.8 880 69.7 37.6 / 73.6 94.5 / 98.1<br><br>84.6 / 92.9 86.2 79.4 93.6 79.6 877 68.8 38.0 / 72.5 94.2 / 97.9<br><br><br>84.0 / 92.8 86.7 78.2 92.3 79.1 840 70.8 44.4 / 72.2 92.6 / 97.3<br><br><br><br><br>|– – – – – – 72.5 81.3 81.3 81.2<br><br>|
|Gemma3-12B [122] InternVL3-14B [187] InternVL3.5-14B<br><br>|84.2 / 87.2 75.7 67.7 87.1 64.9 702 65.5 24.9 / 65.4 – 86.0 / 93.7 87.3 80.5 94.1 83.6 875 70.3 43.1 / 82.2 94.8 / 98.2<br><br>85.1 / 93.3 86.5 77.8 93.4 78.3 836 70.7 47.9 / 76.6 93.4 / 97.7<br><br><br>|– 83.4 82.0<br><br>|
|Kimi-VL-A3B-2506 [125] InternVL3.5-20B-A4B InternVL3.5-30B-A3B<br><br>|81.9 / 91.2 73.7 77.7 93.5 74.5 869 70.8 46.8 / 71.5 81.1 / 90.6<br><br>85.9 / 93.5 86.6 78.5 92.9 78.1 870 69.3 38.2 / 78.5 93.7 / 97.8<br><br>86.8 / 94.5 87.4 80.5 94.2 81.4 880 70.6 48.0 / 81.8 94.9 / 98.2<br><br><br>|78.3 81.7 83.9<br><br>|
|Gemma3-27B [122] Qwen2.5-VL-32B [5] Cambrian-34B [131] VILA-1.5-40B [66] InternVL3-38B [187] InternVL3.5-38B<br><br>|84.5 / 88.2 78.0 65.1 86.6 65.1 717 68.1 25.7 / 63.8 –<br><br>84.0 / 92.9 64.9 78.9 94.8 83.4 850 72.1 46.1 / 84.9 88.9 / 93.0 79.5 / – 75.6 76.7 75.5 46.0 600 – 27.3 / 59.7 79.7 / 89.3 69.9 / – 67.2 73.6 – – 460 – 24.0 / 38.7 –<br><br>88.9 / 95.5 89.2 83.9 95.4 85.0 886 71.6 46.4 / 87.2 96.1 / 98.7 87.8 / 95.1 88.8 82.7 94.0 83.8 870 71.0 48.1 / 83.1 95.3 / 98.4<br><br>|– 80.7 – – 85.5 84.6<br><br>|
|GPT-4V [95] GPT-4o-20240513 [97] Claude-3-Opus [2] Claude-3.5-Sonnet [2] Gemini-1.5-Pro [105] GLM-4.5V [46] NVLM-D-72B [25] Molmo-72B [30] Qwen2-VL-72B [138] Qwen2.5-VL-72B [5] InternVL3-78B [187] InternVL3.5-241B-A28B<br><br>|78.2 / 89.4 78.5 78.0 88.4 75.1 645 53.8 37.1 / 79.9 52.0 / 65.4<br><br>84.6 / 94.2 85.7 77.4 92.8 79.2 736 72.0 47.1 / 84.5 91.6 / 96.4<br><br>70.6 / 88.1 80.8 67.5 89.3 55.6 694 44.2 30.2 / 71.6 62.0 / 77.7 81.2 / 94.7 90.8 74.1 95.2 74.3 788 71.7 60.2 / 84.3 63.9 / 74.7 79.1 / 94.4 87.2 78.8 93.1 81.0 754 – 43.3 / 72.0 62.7 / 77.7 88.1 / 93.7 86.6 72.0 94.5 84.1 872 74.0 59.5 / 88.2 36.5 / 44.9<br><br>85.2 / 94.2 86.0 82.1 92.6 – 853 – – –<br><br><br><br><br>– / 96.3 87.3 83.1 93.5 81.9 – – – –<br><br>88.1 / – 88.3 85.5 96.5 84.5 877 – – 91.3 / 94.6<br><br>88.7 / – 89.5 83.5 96.4 87.3 885 73.0 49.7 / 87.4 –<br><br>89.7 / 96.0 89.7 84.3 95.4 86.5 906 71.9 46.0 / 85.1 96.0 / 98.6 87.3 / 95.0 88.0 84.5 94.9 82.0 907 71.9 48.3 / 83.9 97.0 / 99.0<br><br><br>|70.0 81.6 67.3 78.7 – 75.8 – – – – 85.8 85.2<br><br>|

- Table 4: Comparison of OCR, chart, and document understanding performance. We compare OCR-related performance across 9 benchmarks: AI2D [55], ChartQA [87], TextVQA [112], DocVQA [89], InfoVQA [88], OCRBench [72], SEED-2-Plus [59], CharXiv [148], and VCR [176]. Part of results are collected from the OpenCompass leaderboard [20] and other papers [2, 30, 32, 148, 176]. When calculating Overall, the score of OCR Bench is normalized from 0-1000 to 0-100.

Moreover, as the model scale increases, InternVL3.5 continues to deliver enhanced performance in visionlanguage understanding. Across the same nine benchmarks, InternVL3.5-4B, InternVL3.5-20B-A4B, InternVL3.5-14B, InternVL3.5-30B-A3B, and InternVL3.5-38B achieve overall average scores of 80.0, 81.7, 82.0, 83.9, and 84.6, respectively, demonstrating consistent improvements with larger model sizes. On AI2D, InternVL3.5-2B, InternVL3.5-4B, InternVL3.5-14B and InternVL3.5-38B obtain scores of 78.8/89.1, 82.6/92.3,

- 85.1/93.3, and 87.8/95.1, respectively, indicating a consistent upward trend across larger models. On ChartQA, InternVL3.5-4B shows a notable improvement in chart understanding, increasing from 80.7 (InternVL3.5-2B) to 86.0, reflecting a significant gain in visual reasoning capability. Similarly, on CharXiv and VCR-EN-Easy, larger model variants also achieve measurable improvements in text and document understanding.

#### 3.4 Multi-Image Understanding

To assess InternVL3’s ability to understand and reason over multiple images —– a key aspect of multimodal interaction —– we conduct comprehensive evaluations on a suite of widely recognized benchmarks, including BLINK [36], Mantis-Eval [51], MMIU [91], MuirBench [132], MMT-Bench [166], and MIRB [181]. These benchmarks evaluate critical skills such as cross-image reasoning and context integration, which are essential for effective multimodal systems.

|Model Name<br><br>|BLINK (val)<br><br>Mantis Eval<br><br>MMIU<br><br>Muir Bench<br><br>MMT (val)<br><br>MIRB (avg)|Overall<br><br>|RealWorld QA<br><br>MME-RW (EN)<br><br>WildVision (win rate)<br><br>R-Bench (dis)<br><br>|Overall|
|---|---|---|---|---|
|InternVL3-1B [187] InternVL3.5-1B<br><br>|42.9 50.2 39.3 31.2 52.9 36.1 44.0 54.8 45.2 41.7 54.5 44.2<br><br>|42.1 47.4<br><br>|58.2 46.0 43.8 60.4 57.6 46.8 49.2 57.4<br><br>|52.1 50.6<br><br>|
|Qwen2-VL-2B [138] Qwen2.5-VL-3B [5] InternVL3-2B [187] InternVL3.5-2B<br><br>|44.4 – – – 55.1 – 47.6 – – 47.7 – –<br><br>50.3 65.9 43.0 38.8 59.5 42.9<br><br>51.3 58.5 44.9 44.0 58.5 45.9<br><br><br>|– – 50.1 50.5<br><br>|62.6 – – – 65.4 53.1 – – 64.3 53.8 48.8 67.5 62.0 49.7 66.6 62.4<br><br>|– – 58.6 58.5<br><br>|
|MiniCPM-V-4 [164] InternVL3.5-4B<br><br>|53.0 71.4 – 46.1 59.7 – 58.1 62.7 49.2 53.1 64.3 55.9<br><br>|– 57.2<br><br>|66.0 58.4 46.0 64.2 66.3 59.8 69.8 68.7<br><br>|58.7 65.1<br><br>|
|Qwen2-VL-7B [138] Qwen2.5-VL-7B [5] MiniCPM-V2.6 [164] Keye-VL-8B [126] GLM-4.1V-9B [46]<br><br>InternVL3-8B [187]<br><br>InternVL3-9B [187] InternVL3.5-8B<br><br><br>|53.2 – – – 64.0 – 56.4 – – 59.6 – – 53.0 69.0 – – 60.8 – 53.4 – – 50.4 65.4 – 65.9 – – 74.7 68.4 – 55.5 70.1 46.8 55.0 65.0 56.8<br><br>58.6 70.1 50.4 51.4 65.4 58.6<br><br>59.5 70.5 49.4 55.8 66.7 57.3<br><br><br>|– – – – – 58.2 59.1 59.9<br><br>|70.1 56.5 – 64.0<br><br>68.5 57.4 – – 65.0 – – –<br><br>65.9 48.8 66.6 65.7 70.6 61.7 74.0 69.1 70.8 62.0 69.8 74.1 70.5 61.3 63.8 70.3 67.5 62.8 76.6 69.7<br><br>|– – – 61.7 68.9 69.2 66.5 67.4<br><br>|
|Gemma3-12B [122] InternVL3-14B [187] InternVL3.5-14B<br><br>|52.6 – – 49.7 58.4 – 60.3 76.0 50.9 56.2 70.3 59.3 57.6 73.7 51.3 58.0 68.0 59.0<br><br>|– 62.2 61.3<br><br>|59.8 48.9 72.8 60.4 70.7 64.0 69.8 69.3 70.5 63.2 73.0 70.9<br><br>|60.5 68.5 69.4<br><br>|
|Kimi-VL-A3B-2506 [125] InternVL3.5-20B-A4B InternVL3.5-30B-A3B<br><br>|56.8 – – 64.6 63.9 –<br><br>59.0 74.2 50.5 58.5 66.6 54.4<br><br>60.4 72.8 55.1 53.1 66.6 59.0<br><br><br>|– 60.5 61.2<br><br>|72.4 54.5 64.8 69.3<br><br>71.2 60.0 69.6 71.5<br><br>72.3 64.8 75.8 70.7<br><br><br>|65.3 68.1 70.9<br><br>|
|Gemma3-27B [122] Qwen2.5-VL-32B [5] Cambrian-34B [131] InternVL3-38B [187] InternVL3.5-38B<br><br>|54.4 – – 55.2 61.8 – 61.2 – – 64.5 66.4 –<br><br>– – – – – – 64.0 77.9 57.4 63.8 71.8 62.3 60.9 77.4 58.9 63.7 71.8 71.9<br><br>|– – – 66.2 67.4<br><br>|65.1 51.6 79.8 61.6 71.2 60.3 85.2 69.5 67.8 44.1 – – 75.6 67.3 71.6 73.3 75.9 66.0 80.0 73.1<br><br>|64.5 71.6 – 72.0 73.8<br><br>|
|GPT-4V [95] GPT-4o-20240513 [97] Claude-3.5-Sonnet [2] Gemini-1.5-Pro [105] GLM-4.5V [46] Qwen2-VL-72B [138] Qwen2.5-VL-72B [5] InternVL3-78B [187] InternVL3.5-241B-A28B<br><br>|54.6 62.7 – 62.3 64.3 53.1 68.0 – 55.7 68.0 65.4 –<br><br>– – 53.4 – – –<br>– – 53.4 – 64.5 –<br><br><br>65.3 – – 75.3 70.9 –<br><br>– – – – 71.8 – 64.4 – – 70.7 – –<br><br>66.3 79.3 60.4 64.5 73.2 64.3 61.4 77.0 61.3 47.5 72.7 73.0<br><br><br>|– – – – – – – 68.0 65.5<br><br>|61.4 – 71.8 65.6 75.4 45.2 80.6 77.7 60.1 51.6 – – 67.5 38.2 – – 75.2 61.7 79.0 72.9<br><br>77.8 – – – 75.7 63.2 – –<br><br>78.0 65.4 73.6 77.4 75.2 65.1 82.8 75.4<br><br><br>|– 69.7 – – 72.2 – – 73.6 74.6<br><br>|

- Table 5: Comparison of multi-image and real-world understanding performance. Multi-image benchmarks include BLINK [36], Mantis-Eval [51], MMIU [91], MuirBench [132], MMT-Bench [166], and MIRB [181]. Real-world benchmarks include RealWorldQA [22], MME-RealWorld [178], WildVision [78], and R-Bench [60]. Part of the results are sourced from the benchmark papers and the OpenCompass leaderboard [20].

- As shown in Table 5, across various model scales, InternVL3.5 consistently outperforms other open-source and closed-source counterparts, including earlier versions such as InternVL3. For example, InternVL3.5-38B achieves an overall score of 67.4, which is higher than the 66.2 achieved by InternVL3-38B. On the lightweight scale, InternVL3.5-2B achieves an overall score of 50.5, and its performance on individual benchmarks is 51.3 on BLINK, 58.5 on Mantis, 44.9 on MMIU, 44.0 on Muir, 58.5 on MMT and 45.9 on MIRB. Furthermore, larger model sizes lead to significant improvements in multi-image understanding capabilities. When the model is scaled up to InternVL3.5-4B, the overall score increases to 57.2, with scores of 58.1 on BLINK, 62.7 on Mantis, 49.2 on MMIU, 53.1 on Muir, 64.3 on MMT, and 55.9 on MIRB. As the model size continues to grow, performance across all benchmarks improves consistently, with InternVL3.5-8B achieving an overall score of 59.9, InternVL3.5-14B reaching 61.3, and InternVL3.5-241B-A28B improving to 65.5.

#### 3.5 Real-World Comprehension

To evaluate the performance of InternVL3.5 on realistic and complex tasks, we provide experimental results on four real-world comprehension benchmarks: RealWorldQA [22], MME-RealWorld [178], WildVision [78], and R-Bench [60].

- As shown in Table 5, InternVL3.5 achieves comparable or superior performance compared to existing methods, e.g., Qwen2.5-VL, MiniCPM-V-4 and Keye-VL. For example, the smallest variant InternVL3.5-1B demonstrates promising performance with a RealWorldQA score of 57.6, an MME-RealWorld score of 46.8, a WildVision win rate of 49.2, and an R-Bench score of 57.4. Scaling up the model results in further improvements, as larger models provide more robust representations and stronger comprehension capabilities in real-world scenarios.
- At the higher end of the scale, the InternVL3.5-38B and InternVL3.5-241B-A28B models achieve top-tier results among the InternVL3.5 series. In particular, InternVL3.5-241B-A28B records an overall score of 74.6. Compared to competitive models, such as GPT-4o [95]—which scores 45.2 on MME-RealWorld and 80.6 on

|Model<br><br>|MME (sum)<br><br>MMB v1.1 (EN)<br><br>MMVet (turbo)<br><br>MMStar|Overall<br><br>|HallBench (avg)<br><br>CRPE (relation)<br><br>POPE (avg)<br><br>|Overall|
|---|---|---|---|---|
|InternVL3-1B [187] InternVL3.5-1B<br><br>|1934.4 69.9 59.5 51.5 1910.2 69.9 56.5 51.9<br><br>|62.5 61.6<br><br>|41.4 64.0 90.7 41.0 68.4 86.8<br><br>|65.4 65.4<br><br>|
|Qwen2-VL-2B [138] Qwen2.5-VL-3B [5] InternVL3-2B [187] InternVL3.5-2B<br><br>|1872.0 72.2 49.5 48.0<br><br>2157.0 – – –<br><br>2221.2 78.6 62.2 60.7 2123.3 76.6 71.7 62.7<br><br>|59.1 – 70.2 71.7<br><br>|41.7 – – 46.3 73.6 –<br><br>42.5 71.5 89.6 48.6 75.6 87.2<br><br><br>|– – 67.9 70.5<br><br>|
|MiniCPM-V-4-4B [164] InternVL3.5-4B<br><br>|2167.7 79.1 56.6 59.0 2272.3 80.3 76.6 65.0<br><br>|68.0 75.8<br><br>|46.9 74.3 82.4 44.8 75.0 88.9<br><br>|67.9 69.6<br><br>|
|Qwen2-VL-7B [138] Qwen2.5-VL-7B [5] MiniCPM-V2.6 [164] Keye-VL-8B [126] GLM-4.1V-9B [46]<br><br>InternVL3-8B [187]<br><br>InternVL3-9B [187] InternVL3.5-8B<br><br><br>|2326.8 80.7 62.0 60.7<br><br>2347.0 82.6 67.1 63.9<br>2348.4 78.0 60.0 57.5 2214.7 76.3 70.0 72.8 2445.8 85.8 66.4 72.9 2415.4 81.7 81.3 68.2 2372.8 81.7 76.2 66.3 2380.6 79.5 83.1 69.3<br><br><br>|71.6 74.4 69.8 74.5 78.1 79.4 77.2 79.2<br><br>|50.6 74.4 88.1 52.9 76.4 86.4<br><br>48.1 75.2 87.3 57.7 76.5 87.0 63.2 78.9 87.1<br>49.9 76.3 91.1<br><br><br>51.2 75.0 90.4 54.5 75.1 88.7<br><br><br>|71.0 71.9 70.2 73.7 76.4 72.4 72.2 72.8<br><br>|
|Gemma3-12B [122] InternVL3-14B [187] InternVL3.5-14B<br><br>|2044.7 71.8 64.9 56.1 2478.3 83.5 80.2 68.8 2398.4 81.5 81.7 70.4<br><br>|66.5 80.3 79.8<br><br>|47.2 69.1 85.2 55.1 77.3 90.2 54.0 76.6 87.7<br><br>|67.2 74.2 72.8<br><br>|
|Kimi-VL-A3B-2506 [125] InternVL3.5-20B-A4B InternVL3.5-30B-A3B<br><br>|2352.7 84.4 78.1 70.4 2318.1 85.2 80.3 70.4 2461.9 84.8 85.5 72.0<br><br>|79.2 79.7 82.6<br><br>|59.8 60.9 86.5<br><br>52.0 76.8 89.4<br><br>53.8 77.6 89.6<br><br><br>|69.1 72.7 73.7<br><br>|
|Gemma3-27B [122] Qwen2.5-VL-32B [5] Cambrian-34B [131] InternVL3-38B [187] InternVL3.5-38B<br><br>|1816.9 77.2 68.3 61.7 2402.9 85.2 69.6 67.8<br><br>– 78.3 53.2 54.2 2523.6 86.9 83.9 71.5 2492.4 87.3 82.2 75.3<br><br>|68.0 77.1 – 83.1 83.5<br><br>|47.9 71.64 85.8 53.7 77.0 86.8 41.6 – – 57.1 77.1 90.6 59.7 77.7 90.4<br><br>|68.4 72.5 – 74.9 75.9<br><br>|
|GPT-4V [95] GPT-4o-20240513 [97] Claude-3-Opus [2] Claude-3.5-Sonnet [2] Gemini-1.5-Pro [105] GLM-4.5V [46] Qwen2-VL-72B [138] Qwen2.5-VL-72B [5] InternVL3-78B [187] InternVL3.5-241B-A28B<br><br>|1926.6 80.0 67.5 56.0 – 83.1 69.1 64.7 1586.8 60.1 51.7 45.7<br><br>– 80.9 70.1 65.1<br><br>– 74.6 64.0 59.1<br><br><br>2423.8 88.2 75.2 75.3 2482.7 85.9 74.0 68.3 2448.0 88.4 76.2 70.8 2549.8 87.7 81.3 72.5 2525.9 87.4 81.2 77.9<br><br>|68.1 – 53.5 – – 81.3 79.2 80.7 83.1 84.2<br><br>|46.5 – – 55.0 76.6 86.9 37.8 – – 55.5 – – 45.6 – – 64.5 55.4 85.9<br><br>58.1 – – 55.2 79.2 –<br>59.1 79.2 90.3 57.3 78.0 90.7<br><br><br>|– 72.8 – – – 68.6 – – 76.2 75.3<br><br>|

- Table 6: Comparison of comprehensive multimodal understanding and hallucination performance. Comprehensive multimodal benchmarks include MME [34], MMBench [71], MMVet [168], and MMStar [11]. Hallucination-related benchmarks encompass HallusionBench [41], CRPE [144], and POPE [64]. Part of the results are sourced from the benchmark papers and the OpenCompass leaderboard [20]. When calculating Overall, the score of MME is normalized from 0-2800 to 0-100.

WildVision—the InternVL3.5 series exhibits competitive strengths. InternVL3.5-241B-A28B not only surpasses GPT-4o on RealWorldQA and closely matches its R-Bench performance but also considerably outperforms it on MME-RealWorld, indicating a robust overall performance on tasks demanding both perceptual precision and comprehensive understanding.

- 3.6 Comprehensive Multimodal Understanding In Table 6, we evaluate InternVL3.5 on a set of comprehensive multimodal understanding benchmarks, including

- MME [34], MMBench (English and Chinese) [71], MMBench v1.1 (English) [71], and MMVet [168] and MMStar [11].

We observe that InternVL3.5 outperforms existing methods like Keye-VL, Qwen2.5-VL and MiniCPM-V-4, especially on MMStar and MMVet. For instance, InternVL3.5-4B achieves an MMVet score of 76.6 and MMStar of 65.0, compared to 56.6 and 59.0 of MiniCPM-V-4. The improvements remain significant as model size grows, where InternVL3.5-241B-A28B finally achieves 87.4 on MMBench v1.1, 81.2 on MMVet, 77.9 on MMStar, and an overall score of 84.2.

|Model Name|RefCOCO val test-A test-B<br><br>|RefCOCO+ val test-A test-B<br><br>|RefCOCOg val test<br><br>|Overall|
|---|---|---|---|---|
|Grounding-DINO-L [69] UNINEXT-H [158] ONE-PEACE [139]<br><br>|90.6 93.2 88.2 92.6 94.3 91.5 92.6 94.2 89.3<br><br>|82.8 89.0 75.9 85.2 89.6 79.8 88.8 92.2 83.2|86.1 87.0<br><br>88.7 89.4<br><br>89.2 89.3<br><br><br>|86.6 88.9 89.8|
|InternVL3-1B [187] InternVL3.5-1B<br><br>|85.8 90.1 81.7 85.4 89.7 80.2<br><br>|76.6 84.1 69.2<br><br>77.7 85.5 69.5<br><br><br>|82.8 82.6 81.9 81.6<br><br>|81.6 81.4<br><br>|
|InternVL3-2B [187] InternVL3.5-2B<br><br>|89.8 92.6 86.4 88.7 91.6 84.8<br><br>|84.0 89.2 76.5 82.7 88.4 76.6<br><br>|87.6 87.2 85.6 85.5<br><br>|86.7 85.5<br><br>|
|Qwen2.5-VL-3B [5] InternVL3.5-4B<br><br>|89.1 91.7 84.0 92.5 94.3 88.2<br><br>|82.4 88.0 74.1 87.6 92.3 81.6<br><br>|85.2 85.7 89.6 89.3<br><br>|85.0 89.4<br><br>|
|Shikra-7B [10] CogVLM-Grounding [141] Qwen2-VL-7B [138] Qwen2.5-VL-7B [5] TextHawk2 [169]<br><br>InternVL3-8B [187]<br>InternVL3-9B [187] InternVL3.5-8B<br><br><br>|87.0 90.6 80.2 92.8 94.8 89.0<br><br>91.7 93.6 87.3<br><br>90.0 92.5 85.4<br><br>91.9 93.0 87.6<br><br>92.5 94.6 88.0<br><br><br>91.8 93.2 86.6 92.4 94.7 88.7<br><br><br>|81.6 87.4 72.1 88.7 92.9 83.4<br><br>85.8 90.5 79.5 84.2 89.1 76.9<br>86.2 90.0 80.4 88.2 92.5 81.8<br><br><br>86.4 91.0 79.9<br><br>87.9 92.4 82.4<br><br><br>|82.3 82.2 89.8 90.8<br><br>87.3 87.8<br><br>87.2 87.2<br><br>88.2 88.1<br><br>89.6 90.0<br><br><br>88.0 88.5<br><br>89.6 89.4<br><br><br>|82.9 90.3 87.9 86.6 88.2 89.6 88.2 89.7<br><br>|
|Ferret-v2-13B [173] InternVL3-14B [187] InternVL3.5-14B InternVL3.5-20B-A4B InternVL3.5-30B-A3B<br><br>|92.6 95.0 88.9 92.0 94.4 87.8<br><br>92.6 94.7 89.4 91.9 94.1 88.8<br><br>93.1 95.4 90.1<br><br><br>|87.4 92.1 81.4<br>87.4 92.1 81.5<br><br>88.3 92.7 82.5<br><br>87.6 92.0 82.7<br><br>89.6 93.2 84.4<br><br><br><br><br>|89.4 90.0<br><br>88.6 89.3<br><br>90.1 90.5<br><br>89.1 90.0<br><br>90.6 91.0<br><br><br><br><br>|89.6 89.1 90.1 89.5 90.9<br><br>|
|InternVL3-38B [187] InternVL3.5-38B<br><br>|93.2 95.1 90.2 90.3 91.8 89.0<br><br>|89.8 93.2 85.2 87.5 90.0 84.7<br><br>|91.4 91.5 89.7 89.9<br><br>|91.2 89.1<br><br>|
|Qwen2-VL-72B [138] Qwen2.5-VL-72B [5] InternVL3-78B [187] InternVL3.5-241B-A28B<br><br>|93.2 95.3 90.7<br><br>92.7 94.6 89.7<br><br>93.4 95.4 90.3<br><br>94.1 96.3 91.5<br><br><br>|90.1 93.8 85.6 88.9 92.2 83.7<br><br>90.1 93.8 85.3<br><br>91.6 94.6 86.9<br><br><br>|89.9 90.4 89.9 90.3<br><br>91.5 91.5<br><br>92.0 92.1<br><br><br>|91.1 90.3 91.4 92.4<br><br>|

- Table 7: Comparison of visual grounding performance. We evaluate InternVL3.5’s visual grounding capability on RefCOCO, RefCOCO+, and RefCOCOg datasets [54, 86]. Part of the results are collected from [138].

We note that InternVL3.5 does not achieve a notable improvement compared to InternVL3. This is partly because the model’s understanding performance has approached saturation, and also partly stems from our optimization of text and reasoning capabilities—which, while achieving improvements on relevant benchmarks, slightly impairs the performance of multimodal understanding.

#### 3.7 Multimodal Hallucination Evaluation

To evaluate the propensity for hallucination of InternVL3.5, we conduct experiments on three established benchmarks: HallusionBench [41], CRPE [144], and POPE [64]. The results are shown in Table 6. Compared with previous InternVL series, the new InternVL3.5 models provide consistent improvements in handling multimodal hallucination challenges across various model scales, e.g., +2.6 on 2B scale and +1.0 on 38B scale on the overall score. Despite these advancements, there are minor declines on some model scales such as 14B and 241B-A28B, indicating that further enhancement on data and training strategies is needed to achieve more consistent improvements on all model scales, which remains an important future direction to build a more trustworthy multimodal model.

##### 3.8 Visual Grounding For the visual grounding task, we evaluate InternVL3.5 on RefCOCO, RefCOCO+, and RefCOCOg datasets [54,

- 86]. As shown in Table 7, InternVL3.5 maintains the strong capabilities of InternVL3, which already achieves the upper bound of this tasks, i.e., approximately 90% average accuracy. However, the InternVL3.5 training scheme still provides additional gains on several model sizes. For example, InternVL3.5-14B achieves an overall score of 90.1 on the RefCOCO series, outperforming InternVL3-14B by +0.8%. In addition, InternVL3.5-241B-A28B builds a new state-of-the-art performance on RefCOCO, with an overall score of 92.4, further highlighting the potential of InternVL3.5 for real-world applications requiring precise multimodal understanding.

|Model<br><br>|MMMB en zh pt ar tr ru<br><br>|Multilingual MMBench en zh pt ar tr ru|MTVQA (avg)<br><br>|Overall|
|---|---|---|---|---|
|InternVL3-1B [187] InternVL3.5-1B<br><br>|79.4 70.1 62.3 58.0 47.6 61.9 77.0 73.1 67.2 59.0 53.5 66.3<br><br>|72.6 66.2 62.3 48.0 39.5 60.3 71.2 66.3 61.7 45.8 45.7 60.2<br><br>|22.2 22.9<br><br>|47.9 49.1<br><br>|
|Qwen2-VL-2B [138] Qwen2.5-VL-3B [5] InternVL3-2B [187] InternVL3.5-2B<br><br>|78.3 74.2 72.6 68.3 61.8 72.8 – – – – – – 81.9 78.3 75.4 68.6 62.9 74.6 80.2 77.7 75.9 68.5 69.1 76.3<br><br>|72.1 71.1 69.9 61.1 54.4 69.3 – – – – – – 81.3 77.8 75.9 66.4 59.5 70.7 78.4 75.9 73.7 63.7 62.0 71.4<br><br>|20.0 24.8 26.7 28.5<br><br>|52.6 – 57.4 58.0<br><br>|
|MiniCPM-V-4-4B [164] InternVL3.5-4B<br><br>|82.0 80.2 75.6 60.1 63.8 71.7 84.3 82.6 81.0 76.4 75.2 81.4<br><br>|80.8 80.4 71.6 51.9 59.1 67.7<br><br>81.5 81.1 76.7 71.0 72.4 75.7<br><br><br>|22.6 29.6<br><br>|54.5 62.1<br><br>|
|mPLUG-Owl2 [165] Qwen2-VL-7B [138] Qwen2.5-VL-7B [5] Keye-VL-8B [126] GLM-4.1V-9B [46]<br><br>InternVL3-8B [187]<br>InternVL3-9B [187] InternVL3.5-8B<br><br><br>|67.3 61.0 59.7 45.8 45.4 62.6<br><br>83.9 82.4 81.2 79.0 74.7 82.4<br><br>– – – – – – 66.8 83.0 74.1 73.8 72.0 76.8 82.6 83.6 79.4 80.4 80.4 82.9 85.1 83.1 82.5 81.6 76.2 83.4<br><br>84.8 83.7 80.6 69.9 68.5 80.8 84.9 83.0 81.4 79.6 77.4 82.1<br><br><br>|66.2 59.4 58.2 37.9 47.7 60.4<br><br>81.8 81.6 79.1 75.6 74.5 79.3<br><br>– – – – – – 53.9 87.8 57.4 67.2 67.14 68.7 83.0 86.0 79.8 78.8 78.5 82.0<br><br>85.5 85.6 83.2 79.2 75.9 82.6<br><br>86.5 85.2 79.1 64.3 68.3 79.1<br><br><br>82.5 80.7 79.0 75.9 74.8 77.6<br><br><br>|– 25.6 29.2 22.3 25.5 30.2 27.1 35.2<br><br>|– 61.6 – 54.6 62.8 64.7 60.7 65.0<br><br>|
|Gemma3-12B [122] InternVL3-14B [187] InternVL3.5-14B<br><br>|77.6 77.3 77.3 73.4 73.9 75.9 85.7 84.7 83.1 83.7 79.3 83.6 85.1 84.1 82.7 80.3 79.4 83.5<br><br>|72.1 72.5 68.3 53.8 60.5 60.4 86.7 85.8 83.2 81.1 80.7 83.8 84.0 83.7 80.0 77.8 77.0 77.0<br><br>|24.4 31.6 34.2<br><br>|55.0 66.2 65.5<br><br>|
|Kimi-VL-A3B-2506 [125] InternVL3.5-20B-A4B InternVL3.5-30B-A3B<br><br>|83.1 78.9 76.9 71.3 71.0 76.3<br><br>85.1 83.1 83.2 82.2 80.4 83.8<br><br>86.4 85.7 83.4 83.3 81.7 85.0<br><br><br>|82.3 79.9 76.9 62.8 66.7 73.8<br><br>85.3 84.2 82.1 78.9 79.5 82.5<br><br>86.1 86.3 83.1 82.3 81.5 83.5<br><br><br>|27.2 28.2 33.7<br><br>|59.1 64.4 67.3<br><br>|
|Gemma3-27B [122] Qwen2.5-VL-32B [5] InternVL3-38B [187] InternVL3.5-38B<br><br>|78.9 77.8 77.8 75.4 76.1 76.9<br><br>85.2 83.0 81.9 81.8 79.9 83.5<br><br>86.7 85.6 84.5 84.8 82.6 85.1 86.7 85.5 85.1 84.1 84.3 85.3<br><br><br>|79.0 77.2 74.5 71.8 74.0 74.1<br><br>88.1 85.9 80.2 81.6 79.7 84.5<br>89.0 89.3 87.1 84.6 84.3 87.4 87.4 86.9 84.2 82.0 83.4 85.6<br><br><br>|27.5 31.4 32.4 36.1<br><br>|59.9 65.7 68.1 68.7<br><br>|
|GPT-4V [95] GPT-4o [97] Gemini-1.0-Pro [120] GLM-4.5V [46] Qwen2-VL-72B [138] Qwen2.5-VL-72B [5] InternVL3-78B [187] InternVL3.5-241B-A28B<br><br>|75.0 74.2 71.5 73.5 69.0 73.1 – – – – – – 75.0 71.9 70.6 69.9 69.6 72.7 87.1 86.9 84.8 84.5 84.6 84.3<br><br>86.8 85.3 85.2 84.8 84.2 85.3<br><br>– – – – – –<br><br>87.2 86.6 85.5 86.5 84.6 86.1 87.6 86.4 85.3 84.2 85.1 86.0<br><br><br>|77.6 74.4 72.5 72.3 70.5 74.8 – – – – – – 73.6 72.1 70.3 61.1 69.8 70.5 89.1 89.3 86.9 83.7 84.0 87.2 86.9 87.2 85.8 83.5 84.4 85.3 – – – – – – 89.4 90.3 88.7 86.1 86.6 88.1 88.9 87.7 87.0 86.5 86.7 87.6<br><br>|22.0 27.8 – 30.5 30.9 31.7 32.5 39.3<br><br>|56.1 – – 67.5 67.2 – 68.9 70.8<br><br>|

#### Table 8: Comparison of multimodal multilingual performance. We evaluate multilingual capabilities across

- 3 benchmarks, including MMMB [114], Multilingual MMBench [114] and MTVQA [119] with six languages: English (en), Chinese (zh), Portuguese (pt), Arabic (ar), Turkish (tr), and Russian (ru).

#### 3.9 Multimodal Multilingual Understanding

InternVL3.5 exhibits strong multimodal multilingual understanding across a variety of benchmarks and languages. As summarized in Table 8, InternVL3.5 consistently achieves high scores on MMMB [114], Multilingual MMBench [114] and MTVQA [119], covering six languages including English, Chinese, Portuguese, Arabic, Turkish, and Russian. Compared to InternVL3, InternVL3.5 has significant improvements in its language capabilities, thus achieving better results on these multilingual benchmarks. For example, InternVL3.5-1B achieves up to 1.2% gains over InternVL3-1B. Compared to other leading multimodal models, such as Qwen2.5-VL and GPT-4V, InternVL3.5 also demonstrates notable improvements in both overall accuracy and language coverage, especially at larger scales. For example, InternVL3.5-241B-A28B outperforms GPT-4V by +14.7% on the overall score across all multilingual benchmarks. The results highlight InternVL3.5’s robust capability to handle complex multilingual and multimodal tasks, making it highly effective for global applications that require comprehensive cross-language understanding.

|Model Name|Video-MME (wo / w sub)<br><br>MVBench<br><br>MMBench-Video (val)<br><br>MLVU (M-Avg)<br><br>LongVideoBench (val total)<br><br>|Overall|
|---|---|---|
|InternVL3-1B [187] InternVL3.5-1B<br><br>|51.0 / 53.0 63.1 1.30 53.0 48.1<br><br>52.4 / 55.0 61.0 1.39 56.6 53.0<br><br><br>|51.9 54.1<br><br>|
|Qwen2-VL-2B [138] Qwen2.5-VL-3B [5] InternVL3-2B [187] InternVL3.5-2B<br><br>|55.6 / 60.4 63.2 – – – 61.5 / 67.6 67.0 1.63 68.2 43.3 58.9 / 61.4 70.4 1.42 64.2 55.4 58.4 / 61.9 65.9 1.56 64.4 57.4<br><br>|– 60.3 59.6 60.0<br><br>|
|MiniCPM-V-4-4B [164] InternVL3.5-4B<br><br>|61.2 / 65.8 58.7 – – – 65.4 / 68.6 71.2 1.59 70.4 60.8<br><br>|– 64.9<br><br>|
|VideoChat2-HD [62] LLaVA-OneVision-7B [58] MiniCPM-V-2.6 [164] Qwen2-VL-7B [138] Qwen2.5-VL-7B [5] Keye-VL-8B [126] GLM-4.1V-9B [126]<br><br>InternVL3-8B [187]<br><br>InternVL3-9B [187] InternVL3.5-8B<br><br><br>|45.3 / 55.7 62.3 1.22 47.9 – 58.2 / – 56.7 – – – 60.9 / 63.6 – 1.70 – 54.9 63.3 / 69.0 67.0 1.44 – 55.6<br><br>65.1 / 71.6 69.6 1.79 70.2 45.3<br><br>67.7 / – – – – 64.8<br>68.2 / 73.6 68.4 1.63 71.5 65.7<br><br><br>66.3 / 68.9 75.4 1.69 71.4 58.8 66.7 / 68.9 74.3 1.69 70.8 62.5 66.0 / 68.6 72.1 1.67 70.2 62.1<br><br><br>|– – – – 63.6 – 67.0 66.2 66.6 65.8<br><br>|
|InternVL3-14B [187] InternVL3.5-14B<br><br>|70.4 / 73.0 76.6 1.73 73.3 63.9 67.9 / 71.0 72.8 1.73 72.1 62.7<br><br>|69.1 67.4<br><br>|
|Kimi-VL-A3B-2506 [125] InternVL3.5-20B-A4B InternVL3.5-30B-A3B<br><br>|67.8 / 72.6 59.7 – 74.2 64.5 62.4 / 64.9 73.3 1.54 65.6 58.3<br><br>68.7 / 71.8 72.1 1.69 73.0 63.8<br><br><br>|– 62.6 67.6<br><br>|
|Oryx-1.5-32B [74] Qwen2.5-VL-32B [5] VILA-1.5-40B [66] InternVL3-38B [187] InternVL3.5-38B<br><br>|67.3 / 74.9 70.1 1.52 72.3 – 70.5 / 77.9 – 1.93 – – 60.1 / 61.1 – 1.61 56.7 –<br><br>72.7 / 75.0 76.9 1.81 77.8 67.3 70.9 / 74.2 75.0 1.90 77.0 65.7<br><br>|– – – 71.7 71.0<br><br>|
|GPT-4V/4T [1] GPT-4o-20240513 [95] GPT-4o-20240806 [97] Gemini-1.5-Pro [105] GLM-4.5V [58] VideoLLaMA2-72B [17] LLaVA-OneVision-72B [58] Qwen2-VL-72B [138] Qwen2.5-VL-72B [5] InternVL3-78B [187] InternVL3.5-241B-A28B<br><br>|59.9 / 63.3 43.7 1.53 49.2 59.1<br><br>71.9 / 77.2 – 1.63 64.6 66.7<br><br>– – 1.87 – – 75.0 / 81.3 – 1.30 – 64.0 74.6 / 80.7 73.0 2.05 75.3 68.8 61.4 / 63.1 62.0 – – – 66.2 / 69.5 59.4 – 66.4 61.3<br><br>71.2 / 77.8 73.6 1.70 – – 73.3 / 79.1 70.4 2.02 74.6 60.7<br>72.7 / 75.7 78.7 1.81 79.5 65.7<br><br><br>72.9 / 76.0 76.5 1.74 78.2 67.1<br><br><br>|54.4 – – – 73.5 – – – 70.9 72.1 71.4<br><br>|

- Table 9: Comparison of video understanding performance. We evaluate InternVL3.5’s video understanding capabilities across 5 benchmarks. For Video-MME [35], MMBench-Video [33], MLVU [186], and LongVideoBench [149], we test with four different settings: 16, 32, 48, and 64 frames, and report the maximum results. For MVBench [63], we conduct testing using 16 frames. When calculating Overall, the score of MMBench-Video is normalized from 0-3 to 0-100.

#### 3.10 Video Understanding

InternVL3.5 demonstrates remarkable video understanding capabilities across a comprehensive set of benchmarks. As presented in Table 9, InternVL3.5 consistently achieves competitive or leading scores on Video-

- MME [35], MVBench [63], MMBench-Video [33], MLVU [186], LongVideoBench [149]. Performance improvements are observed across almost all metrics for small-size models. In particular, InternVL3.5-1B outperforms InternVL3-1B by +2.2% of overall performance. For larger InternVL3.5 variants (such as 38B) also deliver comparable results with other state-of-the-art models of similar scale. Furthermore, InternVL3.5 exhibits robust generalization on challenging tasks involving long video sequences and complex reasoning, as reflected in its performance on LongVideoBench. In particular, InternVL3.5-1B achieves significant improvements on LongVideoBench, i.e., +4.9%. The model’s ability to process multi-frame inputs and handle diverse video scenarios underscores its versatility. These results highlight InternVL3.5’s substantial progress in video understanding, positioning it as a highly capable solution for advanced multimodal video analysis tasks.

|Model<br><br>|ScreenSpot ScreenSpot-v2 OSWorld-G<br><br>|WindowsAgentArena WebArena-Lite-v2|
|---|---|---|
|ShowUI-2B [67] UI-TARS-2B [101] JEDI-3B [155] OS-Atlas-4B [150] Qwen2.5-VL-3B [5] InternVL3.5-4B<br><br>|75.1 – –<br><br>82.3 84.7 –<br><br>– 80.9 27.3<br><br>70.1 71.9 –<br><br>– 80.9 27.3<br><br>83.6 85.1 33.9<br><br><br>|– –<br>– –<br>– –<br>– –<br>– –<br><br><br>9.7 7.8<br><br>|
|OS-Atlas-7B [150] UGround-V1-7B [39] Aguvis-7B [157] UI-TARS-7B [101] UI-TARS-1.5-7B [101] JEDI-7B [155] Qwen2.5-VL-7B [5] GLM-4.1V-9B-Thinking† [46] MiMo-VL-7B-RL [154] InternVL3-8B [187] InternVL3.5-8B InternVL3.5-14B<br><br>|82.5 84.1 47.5<br><br>86.3 – 36.4 81.8 – 38.7 89.5 91.6 47.5<br><br>– 89.7 64.2<br>– 91.7 54.1<br>– 88.8 31.4<br>– – –<br><br><br>87.2 90.5 50.7 79.5 81.4 – 87.9 86.2 36.4 87.5 88.6 44.7<br><br><br>|– –<br><br>– –<br><br>– –<br><br>– –<br><br><br>15.9 17.5 – –<br><br>3.4 –<br><br>– –<br><br>– –<br><br>– –<br><br><br>10.5 12.3 12.5 12.3<br><br>|
|GTA1-32B [162] Qwen2.5-VL-32B [5] Gemma-3-27B-IT† [122] InternVL3-38B [187] InternVL3.5-20B-A4B InternVL3.5-30B-A3B InternVL3.5-38B<br><br>|– 93.2 61.9<br><br>– 91.3 46.5<br><br>– – –<br><br><br>85.6 88.3 –<br><br>85.5 87.6 38.2<br><br>86.6 87.3 42.4 81.0 83.5 42.9<br><br><br>|– –<br>– –<br>– –<br>– –<br><br><br>11.0 9.5<br><br>11.0 10.4 14.5 7.1<br><br><br>|
|Operator‡ [96] Aguvis-72B [157] UI-TARS-72B [101] GPT-4o [97] Claude-3.5-Sonnet [2] Claude-3.7-Sonnet [4] Gemini-2.0-Flash [29] Gemini-2.5-Pro [28] Seed1.5-VL [42] Qwen2.5-VL-72B [5] InternVL3-78B [187] InternVL3.5-241B-A28B<br><br>|– 70.5 40.6 89.2 – – 88.4 90.3 57.1 18.1 – –<br><br>83.0 – –<br><br>– 87.6 –<br><br>84.0 – –<br><br><br>– – 45.2<br>– 95.2 62.9<br><br><br>87.1 – –<br>88.7 90.9 –<br><br>89.8 92.9 53.2<br><br><br>|– – 3.5 9.0<br><br>17.9 10.3<br><br>3.5 1.9 – – 6.4 1.9<br><br>– –<br><br>– –<br><br>– –<br><br><br>9.7 14.4 – –<br><br>18.0 11.7<br><br><br>|

- Table 10: Comparison of GUI grounding and online agentic evaluation results. To assess the GUI agent capabilities of InternVL3.5, we conducted evaluations on a diverse set of platforms. We evaluate GUI grounding capabilities across 3 benchmarks including ScreenSpot [16], ScreenSpot-v2 [150] and OSWorld-G [155]. For online agentic evaluation, our assessment covers Ubuntu, Windows, and Web utilizing the OSWorld [156], WindowsAgentArena [7], and WebArena-Lite-v2 [145]. Models with symbols † and ‡ are evaluated under 100 and 200 steps, respectively, while all other results were evaluated under 50 steps.

#### 3.11 GUI Agent Tasks

To validate the GUI agent capabilities of InternVL3.5, we conduct detailed experiments in Table 10. In particular, we evaluate InternVL3.5 on six GUI grounding and agent tasks, namely ScreenSpot [16], ScreenSpot-v2 [150], OSWorld-G [156], OSWorld [156], WindowsAgentArena [7], and WebArena-Lite-v2 [145]. In GUI grounding tasks, InternVL3.5 outperforms most open-source models and is close to the performance of closed-source models. For example, InternVL3.5-241B-A28B outperforms the specialized model, e.g., +2.6% against UITARS-72B [101] on ScreenSpot-v2. Compared to the most advanced commercial model, i.e., Seed1.5-VL [42], InternVL3.5-241B-A28B still maintains close performance on ScreenSpot-v2, i.e., 92.9 vs. 95.2. In GUI agent tasks, InternVL also demonstrates competitive results on different platforms. In particular, InternVL3.5-241BA28B achieves the best results against existing generalist MLLMs, e.g., +8.3% over Qwen2.5-VL-72B on WindowsAgentArena. In particular, GPT-4o only achieves a score of 3.5 on this challenging benchmark. On WebArena-Lite-v2, the performance score of GPT-4o further decreases to 1.9, but InternVL3.5-241B-A28B can still achieve a top-tier score of 11.7. These results confirm the great potential of InternVL3.5 as a fundamental model for GUI tasks.

###### Method VSI-Bench ERQA SpaCE-10 OmniSpatial Overall

- InternVL3-1B [187] 29.7 30.3 41.3 35.5 34.2

- InternVL3.5-1B 49.3 35.3 33.6 40.7 39.7 Qwen2.5-VL-3B [5] 27.9 38.0 34.8 40.3 35.3

InternVL3-2B [187] 31.5 31.5 44.2 38.0 36.3

- InternVL3.5-2B 53.8 37.3 34.6 42.3 42.0 MiniCPM-V-4-4B [164] 30.3 36.3 39.0 43.1 37.2 InternVL3.5-4B 54.9 38.5 35.5 45.8 43.7 Qwen2.5-VL-7B [5] 35.9 38.8 33.3 39.2 36.8 MiMo-VL-RL-8B [154] 36.4 37.8 36.1 46.5 39.2 Keye-VL-8B [126] 28.6 35.3 38.6 46.5 37.2 GLM-4.1V-9B [46] 39.2 45.8 43.4 47.7 44.0 InternVL3-8B [187] 42.1 35.3 40.0 41.6 39.7 InternVL3.5-8B 56.3 41.0 39.5 47.8 46.1 Gemma-3-12B [122] 21.9 36.1 41.5 43.7 35.8 InternVL3-14B [187] 48.9 39.5 47.3 45.9 45.4 InternVL3.5-14B 60.8 41.8 48.8 47.6 49.7 Kimi-VL-A3B-2506 [125] 37.4 36.0 39.2 37.3 37.5 InternVL3.5-20B-A4B 60.1 41.6 51.6 45.4 46.7 InternVL3.5-30B-A3B 63.7 41.5 49.7 48.1 50.8 Gemma-3-27B [122] 22.0 37.5 41.5 44.8 36.4 Qwen2.5-VL-32B [5] 34.7 40.7 32.6 47.4 38.8 InternVL3-38B [187] 48.9 42.8 53.1 48.5 48.3 InternVL3.5-38B 66.3 43.3 43.8 51.4 51.2

- GPT-4o-20241120 [97] 34.0 47.0 49.0 47.8 44.5
- GPT-5-20250807 [98] 37.5 65.7 43.8 59.6 51.7 Gemini-2.5-Pro [29] 47.8 48.3 52.7 55.2 51.0 Claude-3.7-Sonnet [2] 47.0 35.5 46.2 46.9 43.9 GLM-4.5V [46] 41.4 46.5 51.6 51.0 47.6 Qwen2.5-VL-72B [5] 36.1 44.8 37.9 47.9 41.7 Step3-321B-A38B [129] 34.2 44.5 42.6 47.0 42.1 InternVL3-78B [187] 48.4 45.9 52.5 49.3 49.0 InternVL3.5-241B-A28B 69.5 46.8 55.0 51.9 55.8

- Table 11: Comparison of embodied task performance. We compare InternVL3.5 and existing methods on VSI-Bench [161], ERQA [121], SpaCE-10 [38], and OmniSpatial [50]. For SpaCE-10 [38], we report the single-choice performance.

#### 3.12 Embodied Agent Tasks

In Table 11, we evaluate the embodied capabilities of InternVL3.5 on four benchmarks: VSI-Bench [161], ERQA [121], Space10 [38], and OmniSpatial [50]. From this table, we observe the strong embodied capabilities of InternVL3.5 against previous works. On VSI-Bench, the most popular benchmark for spatial reasoning, InternVL3.5-1B achieves an overall score of 49.3, outperforming its predecessor by +19.6%. We note that InternVL3.5-1B can already achieve the state-of-the-art performance on VSI-Bench against much larger models like Qwen2.5-VL-72B [5]. When the model size of InternVL3.5 increases, the performance on VSI-Bench consistently improves from 49.3 to 69.5, greatly validating the salability of InternVL3.5 on embodied tasks. In addition to VSI-Bench, InternVL3.5 also demonstrates promising results on ERQA, Space10 and OminiSpatial. Among them, InternVL3.5B-241B-A28B scores 46.8 on ERQA, which is close to the 48.3 score of the top closed-source model Gemini-2.5-Pro. In terms of overall performance, InternVL3.5B-241B-A28B also reaches the highest score against other models, confirming its strong capabilities in embodied tasks.

#### 3.13 SVG Tasks

Scalable vector graphics (SVG) is a common format for describing graphics on web pages, and its understanding is significant for web-based agent tasks. To evaluate this capability, We evaluate InternVL3.5 on SGP-Bench [102] (Table 12), where it delivers strong results across model scales and sets new open-source

|Model<br><br>|Semantics ↑ Count ↑ Color ↑ Shape ↑ Reasoning ↑|Overall ↑|
|---|---|---|
|Gemma-1.1-2B [123]<br><br>InternVL3.5-1B<br><br>InternVL3.5-2B InternVL3.5-4B<br><br><br>|32.1 33.3 25.0 35.6 28.7 25.5 22.7 24.9 24.4 30.4 25.2 20.1 45.4 33.4 26.5 41.2 55.2 81.9 62.4 39.4<br><br>|31.7 25.0 30.7 57.7<br><br>|
|InternLM2.5-7B [9] Keye-VL-8B [126] GLM-4.1V-9B [46] InternVL3-8B [187] InternVL3.5-8B<br><br>|27.3 31.7 59.8 51.5 28.2 41.4 47.5 71.4 54.9 40.6 41.6 55.6 79.1 61.5 40.0 33.7 46.5 69.8 59.1 36.1 39.7 54.0 82.3 53.4 41.7<br><br>|42.1 52.2 57.1 50.6 54.9<br><br>|
|Gemma-3-12B [122] DeepSeek-Coder-V2-16B [188] InternVL3-14B [187] InternVL3.5-14B<br><br>|24.8 30.8 47.2 25.7 22.8 30.9 37.9 63.7 54.8 26.8 38.2 52.9 74.4 54.1 41.7 44.3 55.3 77.8 63.7 45.1<br><br>|30.5 45.1 52.9 58.5<br><br>|
|Kimi-VL-A3B-2506 [125] InternVL3.5-20B-A4B InternVL3.5-30B-A3B<br><br>|31.1 41.5 67.0 47.4 32.4 51.2 60.6 89.8 74.7 55.5 51.8 66.8 91.8 75.7 53.8<br><br>|44.9 67.6 69.4<br><br>|
|Gemma-3-27B [122] Qwen2.5-VL-32B [5] InternVL3-38B [187] InternVL3.5-38B<br><br>|36.7 51.4 76.3 62.1 39.4 40.0 55.7 76.3 61.2 43.9 40.8 58.7 82.2 63.6 43.9 47.6 66.5 90.5 80.4 54.6<br><br>|54.7 56.5 59.1 69.5<br><br>|
|GPT-5 [98] GLM-4.5V [46] Qwen2.5-VL-72B [5] Step3-321B-A38B [129] InternVL3-78B [187] InternVL3.5-241B-A28B<br><br>|67.8 72.6 91.7 81.9 68.7 47.3 63.7 87.3 72.3 55.8<br><br>40.2 55.1 80.1 62.0 41.1 35.9 54.0 82.8 63.2 38.6<br>41.0 59.1 84.0 65.2 47.0 51.2 69.2 92.1 77.6 58.0<br><br><br>|77.5 66.1 57.1 56.5 60.3 70.7<br><br>|

#### Table 12: Comparison of SVG understanding performance on SGP-Bench [102].

state-of-the-art at larger capacities. At the small scale, InternVL3.5-4B already surpasses models such as Kimi-VL-A3B [125] and even the earlier InternVL3-14B [187]. In the mid-size range (8B, 14B), InternVL3.5 shows broad improvements, and the 14B variant surpasses larger counterparts such as Gemma-3-27B [122]. At the high end, InternVL3.5-30B-A3B and InternVL3.5-38B achieve nearly 70% accuracy on SGP-Bench, advancing the state-of-the-art among open models and outperforming competitors such as Step3-321B-A38B [129], Qwen2.5-VL-72B [5] and GLM-4.5V [46]. Finally, InternVL3.5-241B-A28B sets a new record for open-source models, achieving the best performance across all categories except when compared to GPT-5 [98].

In the SArena-Icon generation tasks (Text2SVG and Img2SVG), InternVL3.5 establishes new state-of-the-art performance among open models (Table 13). In Text2SVG, our models achieve markedly lower FID and FID-C scores than previous baselines, with the 38B variant reducing FID to 14.56. This performance even surpasses GPT-4o [97] (15.18), highlighting that InternVL3.5 is able to match or exceed the capabilities of much larger proprietary systems. In Img2SVG, the 30B and 38B variants deliver leading results on structural similarity metrics, outperforming open-source peers of comparable scale and closely matching the best proprietary systems. Furthermore, InternVL3.5-241B-A28B achieves an even stronger balance of fidelity and perceptual quality, with an FID of 11.27 and an FID-C of 4.43 in Text2SVG, together with competitive Img2SVG scores. These results highlight both the efficiency and scalability of InternVL3.5, showing that it consistently outperforms prior open models across all metrics, narrows the gap with proprietary leaders, and establishes itself as the most powerful open-source framework for SVG generation to date.

#### 3.14 Evaluation on Language Capability

To evaluate the language capabilities of InternVL3.5, we use benchmarks covering comprehensive assessments in general knowledge (MMLU [44], CMMLU [61], C-Eval [49], GAOKAO-Bench [177]), linguistic understanding (TriviaQA [52], NaturalQuestions [56], C3 [115], RACE [57]), reasoning (WinoGrande [107], HellaSwag [172], BigBench Hard [117]), mathematics (GSM8K-Test [18], MATH [45], AIME24 [84], AIME25 [85]), and

|Model<br><br>|Text2SVG FID ↓ FID-C ↓ CLIP ↑|Img2SVG DINO ↑ SSIM ↑ LPIPS ↓ PSNR ↑<br><br>|
|---|---|---|
|InternVL3.5-1B<br><br>InternVL3.5-2B InternVL3.5-4B<br><br><br>|22.50 12.16 72.43 20.98 11.26 72.71 17.06 7.54 74.35<br><br>|0.79 0.57 0.35 7.35 0.81 0.56 0.34 7.44 0.84 0.61 0.30 8.37<br><br>|
|Llama-3.1-8B [32] Qwen2.5-VL-7B [5] Keye-VL-8B [126] GLM-4.1V-9B [46] InternVL3-8B [187] InternVL3.5-8B<br><br>|19.43 11.25 71.86 24.78 15.45 71.38<br><br>21.96 14.39 71.17<br>22.68 10.45 73.20<br>23.06 14.30 71.45 17.36 7.13 75.01<br><br><br>|– – – – 0.78 0.51 0.38 6.53<br><br>0.80 0.53 0.37 6.94<br><br>0.82 0.54 0.35 7.33<br><br>0.81 0.56 0.36 7.22 0.85 0.62 0.29 8.74<br><br><br>|
|Llama-3.2-11B-Vision [32] Gemma-3-12B [122] InternVL3-14B [187] InternVL3.5-14B<br><br>|28.16 14.35 71.49 17.14 10.41 71.62 19.00 13.22 71.49 15.90 5.99 75.91<br><br>|0.76 0.47 0.39 5.91<br><br>0.82 0.58 0.35 7.63<br>0.83 0.56 0.36 7.34 0.86 0.61 0.31 8.46<br><br><br>|
|Kimi-VL-A3B [125] InternVL3.5-20B-A4B InternVL3.5-30B-A3B<br><br>|30.81 16.99 70.54 16.78 5.60 77.46 16.31 5.84 76.40<br><br>|0.80 0.56 0.36 7.18 0.91 0.71 0.20 12.75 0.88 0.65 0.27 9.64<br><br>|
|Gemma-3-27B [122] Qwen2.5-VL-32B [5] InternVL3-38B [187] InternVL3.5-38B<br><br>|15.15 9.30 73.28 20.04 10.39 73.23 18.01 11.04 73.08 14.56 5.22 76.49<br><br>|0.83 0.60 0.35 7.83<br>0.84 0.56 0.36 7.50 0.83 0.55 0.35 7.31 0.86 0.61 0.32 8.39<br><br><br>|
|Grok-3 [151] Llama-3.1-70B [32]<br><br>Llama-3.1-405B [32] DeepSeek-V3-671B-A37B [68] GPT-4o [97] GLM-4.5V [46] Claude-3.7-Sonnet [2] Claude-4-Sonnet [3] Gemini-2.5-Flash [19]<br><br>Llama-3.2-90B-Vision [32] Llama-4-Scout [93] Llama-4-Maverick [92] Step3-321B-A38B [129] Qwen2.5-VL-72B [5] InternVL3-78B [187] InternVL3.5-241B-A28B<br><br><br>|21.97 8.69 76.80<br><br>18.03 8.30 73.88 16.79 8.39 73.92 24.99 8.80 76.47<br><br>15.18 6.76 77.74<br>16.64 5.09 78.35<br><br>14.38 3.50 80.79<br>15.84 4.29 80.58<br>16.72 5.21 78.22<br><br>19.31 8.55 74.00<br><br>17.91 9.38 73.56<br><br><br>14.93 6.53 75.82<br><br>20.06 9.71 74.18<br><br>15.95 9.88 73.68<br><br><br>17.58 10.60 73.12 11.27 4.43 76.81<br><br><br><br><br>|– – – –<br><br>– – – –<br><br>– – – –<br><br>– – – –<br><br><br>0.87 0.62 0.32 8.44<br><br>0.87 0.63 0.32 8.67<br><br><br>0.91 0.65 0.29 9.26<br><br>0.92 0.67 0.28 9.86<br><br><br>0.88 0.59 0.32 8.32 0.76 0.44 0.38 5.78 0.84 0.58 0.35 7.74 0.86 0.60 0.33 8.03<br><br>0.83 0.56 0.34 7.52<br><br>0.84 0.58 0.35 7.83<br><br>0.85 0.58 0.34 7.80 0.88 0.64 0.29 9.19<br><br><br>|

#### Table 13: Comparison of SVG generation performance on SArena-Icon (Text2SVG and Img2SVG).

coding (HumanEval [12]) tasks. As shown in Table 14, InternVL3.5 achieves even better performance than its corresponding language models on most benchmarks. Specifically, InternVL3.5-1B outperforms Qwen3-0.6B on 15 of 16 text-related benchmarks, with an overall performance gain of +6.7. For larger models such as InternVL3.5-241B-A28B, the performance improvement is also obvious, i.e., +2.3 over Qwen3-235B-A22B. These improvements come not only from the high-quality text corpora we use during pre-training and SFT, but also from our Cascade RL, which significantly benefits text-based reasoning tasks. The improvement of InternVL3.5 in text capabilities has also greatly compensated for the shortcomings of open-source multimodal models in general capabilities.

#### 3.15 Ablation Study

Cascade Reinforcement Learning (Cascade RL). To validate the effectiveness of Cascade RL, we conduct an ablation study in Figure 5 and Table 15. We evaluate a baseline model InternVL3 as well as InternVL3.5

|Dataset Version<br><br>|Qwen3-0.6B<br><br>InternVL3.5-1B|Qwen3-1.7B<br><br>InternVL3.5-2B<br><br>|Qwen3-4B<br><br>InternVL3.5-4B|Qwen3-8B<br><br>InternVL3.5-8B<br><br>|Qwen3-14B<br><br>InternVL3.5-14B|Qwen3-30B-A3B<br><br>InternVL3.5-30B-A3B|Qwen3-32B<br><br>InternVL3.5-38B<br><br>|Qwen3-235B-A22B<br><br>InternVL3.5-241B-A28B|
|---|---|---|---|---|---|---|---|---|
|MMLU 4d595a CMMLU c13365 C-Eval 2daf24 GAOKAO 4c31db|52.8 49.1 43.4 46.6 42.6 49.0 40.4 48.4<br><br>|62.6 61.9 59.8 59.4 61.0 61.0 64.1 68.1|73.0 73.8<br><br>71.8 70.6<br><br>72.2 71.9<br><br><br>80.0 83.2|76.9 78.2<br><br>76.6 76.2<br><br>77.9 77.9<br><br><br>85.6 84.1<br><br>|81.1 81.5 81.6 79.8 81.0 80.3 90.0 87.2<br><br>|81.4 83.0 83.0 82.5<br>82.9 83.2 89.5 92.6<br>|83.6 84.6<br><br>84.5 84.4<br><br><br>83.3 85.0 93.2 93.4<br><br>|87.8 89.1 87.4 90.2 86.1 90.9 95.0 94.5<br><br>|
|TriviaQA 2121ce NaturalQuestions 3dcea1 C3 8c358f RACE-High 69ee4f|18.7 20.4 11.2 15.0 64.1 65.5 45.3 67.2<br><br>|30.6 31.7 21.4 23.5 54.2 79.5 65.6 78.8|39.9 40.2 29.3 29.0 79.8 89.5 82.7 86.4<br><br>|52.0 49.7 36.5 32.6 83.0 91.6 86.2 88.4<br><br>|60.9 55.3 42.2 36.7 84.3 94.6 87.2 90.7<br><br>|58.4 57.6 42.5 36.7 90.8 95.1 78.7 92.3|63.4 59.9 45.8 40.1 93.5 95.6 85.6 92.1<br><br>|73.7 74.8 54.8 50.4 96.4 97.8 90.5 94.2|
|WinoGrande b36770 HellaSwag e42710 BBH 5b92b0<br><br>|51.5 54.9 42.7 49.5 41.5 46.8|53.5 59.3<br><br>59.0 74.4<br><br>54.5 62.3<br><br><br>|64.8 69.1 78.1 88.5 72.6 78.6|70.8 75.2 84.6 91.5 78.4 79.8<br><br>|76.6 80.5 88.2 94.1 81.1 82.3<br><br>|73.0 86.5 89.0 96.3 81.5 83.8<br><br>|74.8 83.4 90.5 95.9 87.4 87.5|84.8 91.7 91.1 97.0 88.9 86.5<br><br>|
|GSM8K 1d7fe4 MATH 393424<br><br>AIME2024 –<br><br>AIME2025 – HumanEval 8e312c<br><br><br>|59.6 62.8 32.4 68.2 10.0 13.7 13.3 14.7 39.6 45.7|75.4 77.2 43.5 85.5 40.0 44.2 23.3 42.9 72.0 65.9<br><br>|87.8 92.0 54.1 94.4 66.7 72.8 50.0 57.6 82.3 87.8<br><br>|89.8 90.9 60.8 93.3 76.0 77.7 67.3 64.0 85.4 93.9|92.5 95.6 62.0 93.6 80.0 77.4 66.7 63.9 89.0 97.0<br><br>|91.8 91.4 59.0 93.6 83.3 79.4 70.0 62.7 89.0 96.3<br><br>|93.4 91.5 61.6 94.3 73.3 81.5 60.0 72.1 89.6 98.2<br><br>|94.4 91.6 71.8 94.5 86.7 84.1 83.3 75.6 91.5 98.2|
|Overall –<br><br>|38.1 44.8|52.5 61.0|67.8 74.1<br><br>|74.6 77.8<br><br>|77.8 80.7|77.8 82.0|79.0 83.7<br><br>|85.3 87.6|

InternVL3.5-241B-A28B

InternVL3.5-30B-A3B

Qwen3-235B-A22B

Qwen3-30B-A3B

InternVL3.5-14B

InternVL3.5-38B

InternVL3.5-1B

InternVL3.5-2B

InternVL3.5-4B

InternVL3.5-8B

Qwen3-0.6B

Qwen3-1.7B

Qwen3-14B

Qwen3-32B

Qwen3-4B

Qwen3-8B

- Table 14: Comparison of text-related performance across multiple benchmarks. Results were obtained using the OpenCompass toolkit [20]. We compare InternVL3.5 with Qwen3 models, whose corresponding pre-trained base models are employed as the initialization of the language component in InternVL3.5. Please note that the evaluation scores of the Qwen3 series may differ from those officially reported, as we have adopted the prompt versions provided in the table across all datasets for OpenCompass evaluation.

InternVL3 66.9

66.0

62.4

62.0

InternVL3.5-Instruct

61.4

60.4

60.3

59.0

57.7

57.4

InternVL3.5-MPO

56.6

56.3

56.1

54.9

53.6

InternVL3.5-CascadeRL

52.7

52.7

52.3

50.7

49.6

49.4

44.3

41.1

38.5

33.8

32.4

29.1

25.125.7

1B 2B 4B 8B 14B 30B-A3B 38B 241B-A28B

- Figure 5: Ablation on Cascade RL. We report average scores on the same set of multimodal reasoning and mathematical benchmarks as in Table 3. Full results are provided in Table 15.

after different training stages: InternVL3.5-Instruct (after SFT), InternVL3.5-MPO (after the first substage in Cascade RL), and InternVL3.5-CascadeRL (after both substages in Cascade RL). From it we can see that InternVL3.5-Instruct already outperforms InternVL3 by margins, e.g., + 9.3% on the 8B model. Even based on these strong SFT baselines, the performance of InternVL3.5 can be further improved after the MPO stage, providing up to +3.5% average gains on reasoning tasks. Compared to MPO-based models, our Cascade RL still provides orthogonal gains for all dense and MoE models. For example, InternVL3.5-2B obtains 12.2% average performance gains on reasoning tasks compared to the SFT model. Similar merits can also be observed on larger models, e.g., +6.5% on InternVL3.5-241B-A28B. These ablations confirm the effectiveness, stability, and scalability of our Cascade RL. Additionally, we present a comparison of the efficiency and effectiveness of our proposed Cascade RL against MPO and GSPO in Table 16. For MPO and Cascade RL, we report performance after training for one episode, whereas for GSPO, we report results after both one and two episodes. We observe that MPO achieves performance gains with only a small number of GPU hours, while GSPO yields more significant improvements but at the cost of substantially higher computational consumption. In contrast, Cascade RL attains even greater performance improvements while requiring only half the GPU hours of GSPO.

|Model<br><br>|MMMU (val)<br><br>MathVista (mini)<br><br>MathVision<br><br>MathVerse (vision-only)<br><br>DynaMath (worst case)<br><br>WeMath LogicVista|Overall<br><br>|
|---|---|---|
|InternVL3-1B InternVL3.5-1B-Instruct InternVL3.5-1B-MPO InternVL3.5-1B-CascadeRL|43.4 45.8 18.8 18.7 5.8 13.4 29.8<br><br>37.2 48.6 15.8 27.0 8.4 13.9 29.1 40.3 50.5 22.0 32.1 9.0 16.8 32.7<br><br>44.2 59.3 27.3 37.8 17.2 21.5 29.3<br>|25.1 25.7 29.1 33.8<br><br>|
|InternVL3-2B InternVL3.5-2B-Instruct InternVL3.5-2B-MPO InternVL3.5-2B-CascadeRL|48.6 57.0 21.7 25.3 14.6 22.4 36.9<br><br>53.0 60.8 27.0 39.6 19.8 28.1 41.2<br><br>54.3 62.6 34.2 46.4 21.0 28.1 40.9<br><br><br>59.0 71.8 42.8 53.4 31.5 48.5 47.7|32.4 38.5 41.1 50.7<br><br>|
|InternVL3.5-4B-Instruct InternVL3.5-4B-MPO InternVL3.5-4B-CascadeRL|64.3 71.4 40.5 50.0 30.7 35.6 53.5<br><br>65.4 71.7 48.0 54.9 30.7 39.8 55.9<br><br>66.6 77.1 54.4 61.7 35.7 50.1 56.4<br><br><br>|49.4 52.3 57.4|
|InternVL3-8B InternVL3.5-8B-Instruct InternVL3.5-8B-MPO InternVL3.5-8B-CascadeRL<br><br>|62.7 71.6 29.3 39.8 25.5 37.1 44.1 68.1 74.2 46.4 55.8 30.7 46.0 53.9 71.2 75.9 52.6 54.8 33.1 47.7 58.6 73.4 78.4 56.8 61.5 37.7 57.0 57.3|44.3 53.6 56.3 60.3<br><br>|
|InternVL3-14B InternVL3.5-14B-Instruct InternVL3.5-14B-MPO InternVL3.5-14B-CascadeRL<br><br>|67.1 75.1 37.2 44.4 31.3 43.0 51.2 71.8 73.4 48.7 55.5 31.9 45.7 57.5 73.3 74.0 53.0 57.5 32.3 45.2 60.9 73.3 80.5 59.9 62.8 38.7 58.7 60.2|49.9 54.9 56.6 62.0<br><br>|
|InternVL3.5-30B-A3B-Instruct InternVL3.5-30B-A3B-MPO InternVL3.5-30B-A3B-CascadeRL|72.3 73.3 45.1 50.4 31.9 39.7 56.4 71.7 75.3 50.7 58.5 32.9 43.7 59.7 75.6 80.9 55.7 60.4 36.5 48.4 55.7<br><br>|52.7 56.1 59.0|
|InternVL3-38B InternVL3.5-38B-Instruct InternVL3.5-38B-MPO InternVL3.5-38B-CascadeRL|70.1 75.1 34.2 48.2 35.3 48.6 58.4 73.9 75.9 58.2 59.0 29.7 47.5 60.0<br><br>76.9 80.5 56.3 59.4 36.9 55.6 64.2<br><br>76.9 81.9 63.7 67.6 41.7 64.8 65.3<br><br><br>|52.8 57.7 61.4 66.0|
|InternVL3-78B InternVL3.5-241B-A28B-Instruct InternVL3.5-241B-A28B-MPO InternVL3.5-241B-A28B-CascadeRL<br><br>|72.2 79.0 43.1 51.0 35.1 46.1 55.9 76.2 80.1 55.6 61.7 36.5 49.7 63.3<br><br>76.0 82.2 55.3 64.1 38.3 51.3 69.4<br>77.7 82.7 63.9 68.5 46.5 62.3 66.7<br>|54.6 60.4 62.4 66.9<br><br>|

#### Table 15: Comparison of multimodal reasoning performance after different training stages.

|Model|GPU Hours|MMMU Val<br><br>MathVista MINI<br><br>MathVision MINI<br><br>MathVerse Vision-Only<br><br>DynaMath (Worst)<br><br>WeMath LogicVista|Overall<br><br>|
|---|---|---|---|
|InternVL3.5-8B-Instruct<br><br>+MPO<br><br>+GSPO (1 episode)<br><br>+GSPO (2 episodes)<br><br><br>+CascadeRL (ours)<br><br>|– ∼0.3K Hours ∼5.5K Hours<br><br>∼11.0K Hours ∼5.8K Hours|68.1 74.2 46.4 55.8 30.7 46.0 53.9<br><br>71.2 75.9 52.6 54.8 33.1 47.7 58.6<br><br>73.8 77.9 51.6 58.8 35.1 48.9 54.8<br><br>72.0 78.1 51.6 58.5 35.7 54.1 57.0<br><br>73.4 78.4 56.8 61.5 37.7 57.0 57.3<br>|53.6 56.3 57.3 58.2 60.3<br><br>|

#### Table 16: Comparison of training efficiency and effectiveness of MPO, GSPO, and Cascade RL.

Visual Resolution Router (ViR). In Tables 17 and 18, we compare efficiency and performance of InternVL3.5 with and without ViR, and models equipped with ViR are called InternVL3.5-Flash. In Table 18 , we validate the efficiency improvement brought by ViR. From it we can see that the proposed DvD can already accelerate inference by up to 2.01×, based on which ViR still provides significant efficiency gains, e.g., 4.05× speedup. Note that the efficiency gains of ViR are also obvious for the large MoE model, i.e., InternVL3.5-241B-A28B, which is significant for real-world application.

In Table 17, we compare the performance of InternVL3.5 and InternVL3.5-Flash on several significant benchmarks. For these results, we can see that InternVL3.5-Flash can maintain the multimodal understanding and reasoning performance. In high-resolution tasks like DocVQA and InfoVQA, InternVL3.5-Flash can reach almost the 100% performance of InternVL3.5, e.g., 80.2 vs. 79.8 on 8B model size. Even when the model size improves to 241B, similar observations still hold. These results further confirm that ViR can greatly benefit the model performance without sacrificing performance.

Decoupled Vision-Language Deployment (DvD). In Table 18, we conduct detailed ablation on InternVL3.5 to show the benefit of DvD. From this table, the first observation is that DvD greatly accelerates the inference of both dense and MoE models, by up to 2.01 times and 1.97 times for InternVL3.5-241B-A28B and InternVL3.538B, respectively. In addition, the efficiency gains of DvD can benefit both the pre-filling and next-token

###### Model DocVQA ChartVQA InfoVQA TextVQA OCRBench AI2D MMStar MMMU Mathvista Overall

InternVL3.5-8B 92.3 86.7 76.2 78.2 83.2 84.0 69.3 73.4 78.4 80.2 InternVL3.5-8B-Flash 91.9 86.6 76.0 77.2 83.0 83.6 68.6 72.9 78.0 79.8

InternVL3.5-38B 94.0 88.8 81.0 82.7 87.0 87.8 75.3 76.9 81.9 83.9 InternVL3.5-38B-Flash 93.5 88.1 81.0 82.1 86.5 87.2 75.0 76.3 81.3 83.4

InternVL3.5-30B-MoE 94.2 87.4 77.8 80.5 88.0 86.8 72.0 75.6 80.9 82.6 InternVL3.5-30B-MoE-Flash 93.2 87.3 77.6 80.2 87.8 86.3 71.7 75.2 80.5 82.2

InternVL3.5-235B-MoE 94.9 88.0 82.0 84.5 90.7 86.9 77.9 77.7 82.7 85.0 InternVL3.5-235B-MoE-Flash 94.0 87.9 81.0 84.1 90.3 86.1 77.4 77.0 83.0 84.5

#### Table 17: Performance comparison of InternVL3.5 and InternVL3.5-Flash.

Request Throughput (requests / s) InternVL3.5-38B InternVL3.5-241B-A28B 448

Resolution Setting

Baseline 12.39 11.29

+ DvD 14.69 (1.19 ×) 14.05 (1.24 ×) + DvD + ViR 18.62 (1.50 ×) 20.84 (1.85 ×)

Baseline 2.71 2.54

896

+ DvD 5.06 (1.87 ×) 4.73 (1.86 ×) + DvD + ViR 10.97 (4.05 ×) 8.81 (3.47 ×)

Baseline 1.48 1.37

1344

+ DvD 2.92 (1.97 ×) 2.75 (2.01 ×) + DvD + ViR 5.14 (3.47 ×) 4.27 (3.12 ×)

Table 18: Ablation of Decoupled Vision-Language Deployment (DvD) and Visual Resolution Router (ViR) on inference efficiency. We send 16 requests per second to the deployed server. In all settings, the language models are deployed on 8 A100 GPUs.

prediction stages of InternVL3.5. Another finding is that as the input resolution increases, DVD efficiency also improves. For example, on InternVL3.5-38B, the speed-up of DvD can be improved from 1.19 to 1.97 as the resolution increases from 448 to 1344. This phenomenon can be attributed to the fact that larger input resolution or visual backbone leads to higher visual computational costs and further blocks the computation of the LLM. It is worth noting that the increased computational cost of high-resolution images arises because mainstream MLLMs typically adopt a dynamic high-resolution strategy, which increases the number of patches fed into the vision encoder and thereby raises the overall computation. In practical applications, beyond high-resolution images, many tasks also require multi-image and video understanding. In such scenarios, the number of patches processed by the vision encoder grows even further, leading to greater visual overhead. We believe that our proposed DvD can deliver even more significant performance gains in these scenarios.

### 4 Conclusion

In this work, we introduce InternVL3.5, the latest family of InternVL models that demonstrates stronger general performance and faster speed across a wide range of tasks. InternVL3.5 adopts a new reinforcement learning (RL) framework, namely Cascade RL, which combines the benefits of offline and online RL methods to boost reasoning capabilities. In addition, two techniques are further introduced to reduce the inference cost of InternVL3.5, namely Visual Resolution Router (ViR) and Decoupled Vision-Language Deployment (DvD). Benefiting from these innovations, InternVL3.5 achieves +16.0% improvement in overall reasoning performance and 4.05× speed-up in inference efficiency compared to InternVL3. Besides, InternVL3.5 has significant improvements in its versatility against InternVL3. Specifically, InternVL3.5-241B-A28B achieves the highest overall score across multimodal general, reasoning, text, and agency tasks among leading open-source MLLMs, significantly narrowing the performance gap with top-tier commercial models like GPT-5. We believe that our open source models and codes will push forward multimodal AI research and its real-world applications.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 17

- [2] Anthropic. The claude 3 model family: Opus, sonnet, haiku. https://www.anthropic.com, 2024. 11, 12, 13, 14, 18, 19, 21
- [3] Anthropic. Introducing claude 4: Claude sonnet 4 and claude opus 4, May 2025. 21
- [4] Sonnet Anthropic. Claude 3.7 sonnet system card. 2025. 18
- [5] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 2, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21
- [6] Loubna Ben Allal, Anton Lozhkov, Guilherme Penedo, Thomas Wolf, and Leandro von Werra. Smollm-corpus, July

2024. 4

- [7] Rogerio Bonatti, Dan Zhao, Francesco Bonacci, Dillon Dupont, Sara Abdali, Yinheng Li, Yadong Lu, Justin Wagle, Kazuhito Koishida, Arthur Bucker, et al. Windows agent arena: Evaluating multi-modal os agents at scale. arXiv preprint arXiv:2409.08264, 2024. 18
- [8] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 5
- [9] Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, et al. Internlm2 technical report. arXiv preprint arXiv:2403.17297, 2024. 4, 20
- [10] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023. 15
- [11] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330, 2024. 2, 8, 14
- [12] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021. 1, 21
- [13] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024. 2, 3, 4, 5, 7, 11
- [14] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024. 2, 3
- [15] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024. 2, 3
- [16] Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Yantao Li, Jianbing Zhang, and Zhiyong Wu. Seeclick: Harnessing gui grounding for advanced visual gui agents. arXiv preprint arXiv:2401.10935, 2024. 10, 18
- [17] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024. 17
- [18] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. 20
- [19] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 21

- [20] OpenCompass Contributors. Opencompass: A universal evaluation platform for foundation models. https: //github.com/open-compass/opencompass, 2023. 9, 11, 12, 13, 14, 22
- [21] XTuner Contributors. Xtuner: A toolkit for efficiently fine-tuning llm. https://github.com/InternLM/ xtuner, 2023. 7
- [22] X.AI Corp. Grok-1.5 vision preview: Connecting the digital and physical worlds with our first multimodal model. https://x.ai/blog/grok-1.5v, 2024. 13
- [23] Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, et al. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456, 2025. 1
- [24] Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, et al. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025. 1
- [25] Wenliang Dai, Nayeon Lee, Boxin Wang, Zhuolin Yang, Zihan Liu, Jon Barker, Tuomas Rintamaki, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Nvlm: Open frontier-class multimodal llms. arXiv preprint arXiv:2409.11402, 2024. 12
- [26] Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations (ICLR), 2024. 7
- [27] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems (NeurIPS), 2022. 7
- [28] DeepMind. Gemini 2.5 pro. https://deepmind.google/technologies/gemini/pro/, 2025. 18
- [29] Google Deepmind. Introducing gemini 2.0: our new ai model for the agentic era. https://blog.google/ technology/google-deepmind/google-gemini-ai-update-december-2024/, 2024. 11, 18, 19
- [30] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models. arXiv preprint arXiv:2409.17146, 2024. 12
- [31] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 11198–11201, 2024. 8, 9
- [32] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 12, 21

- [33] Xinyu Fang, Kangrui Mao, Haodong Duan, Xiangyu Zhao, Yining Li, Dahua Lin, and Kai Chen. Mmbench-video: A long-form multi-shot benchmark for holistic video understanding. arXiv preprint arXiv:2406.14515, 2024. 17
- [34] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023. 14
- [35] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024. 2, 8, 17
- [36] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. arXiv preprint arXiv:2404.12390, 2024. 2, 12, 13
- [37] Zhangwei Gao, Zhe Chen, Erfei Cui, Yiming Ren, Weiyun Wang, Jinguo Zhu, Hao Tian, Shenglong Ye, Junjun He, Xizhou Zhu, et al. Mini-internvl: A flexible-transfer pocket multimodal model with 5% parameters and 90% performance. arXiv preprint arXiv:2410.16261, 2024. 2, 3
- [38] Ziyang Gong, Wenhao Li, Oliver Ma, Songyuan Li, Jiayi Ji, Xue Yang, Gen Luo, Junchi Yan, and Rongrong Ji. Space-10: A comprehensive benchmark for multimodal large language models in compositional spatial intelligence. arXiv preprint arXiv:2506.07966, 2025. 1, 2, 10, 19
- [39] Boyu Gou, Ruohan Wang, Boyuan Zheng, Yanan Xie, Cheng Chang, Yiheng Shu, Huan Sun, and Yu Su. Navigating the digital world as humans do: Universal visual grounding for GUI agents. In The Thirteenth International Conference on Learning Representations, 2025. 18

- [40] Shuhao Gu, Jialing Zhang, Siyuan Zhou, Kevin Yu, Zhaohu Xing, Liangdong Wang, Zhou Cao, Jintao Jia, Zhuoyi Zhang, Yixuan Wang, et al. Infinity-mm: Scaling multimodal performance with large-scale and high-quality instruction data. arXiv preprint arXiv:2410.18558, 2024. 12
- [41] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: An advanced diagnostic suite for entangled language hallucination & visual illusion in large vision-language models. arXiv preprint arXiv:2310.14566, 2023. 2, 8, 14, 15
- [42] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025. 11, 18
- [43] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008, 2024. 1, 8
- [44] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In The International Conference on Learning Representations,

2020. 1, 20

- [45] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Joaquin Vanschoren and Sai-Kit Yeung, editors, Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual, 2021. 2, 20
- [46] Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. Glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv preprint arXiv:2507.01006, 2025. 1, 3, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20, 21
- [47] Pin-Lun Hsu, Yun Dai, Vignesh Kothapalli, Qingquan Song, Shao Tang, Siyu Zhu, Steven Shimizu, Shivam Sahni, Haowen Ning, Yanning Chen, and Zhipeng Wang. Liger-kernel: Efficient triton kernels for llm training. In Championing Open-source DEvelopment in ML Workshop @ ICML25, 2025. 7
- [48] Xueyu Hu, Tao Xiong, Biao Yi, Zishu Wei, Ruixuan Xiao, Yurun Chen, Jiasheng Ye, Meiling Tao, Xiangxin Zhou, Ziyu Zhao, et al. Os agents: A survey on mllm-based agents for general computing devices use. arXiv preprint arXiv:2508.04482, 2025. 1
- [49] Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Yao Fu, et al. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models. Advances in Neural Information Processing Systems, 36, 2024. 1, 20
- [50] Mengdi Jia, Zekun Qi, Shaochen Zhang, Wenyao Zhang, Xinqiang Yu, Jiawei He, He Wang, and Li Yi. Omnispatial: Towards comprehensive spatial reasoning benchmark for vision language models. arXiv preprint arXiv:2506.03135,

2025. 1, 2, 19

- [51] Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei, Max Ku, Qian Liu, and Wenhu Chen. Mantis: Interleaved multi-image instruction tuning. arXiv preprint arXiv:2405.01483, 2024. 12, 13
- [52] Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. arXiv preprint arXiv:1705.03551, 2017. 20
- [53] Seungjae Jung, Gunsoo Han, Daniel Wontae Nam, and Kyoung-Woon On. Binary classifier optimization for large language model alignment. arXiv preprint arXiv:2404.04656, 2024. 5
- [54] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing, pages 787–798, 2014. 15
- [55] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In European Conference on Computer Vision, pages 235–251, 2016. 2, 8, 11, 12
- [56] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:453–466, 2019. 20
- [57] Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. Race: Large-scale reading comprehension dataset from examinations. arXiv preprint arXiv:1704.04683, 2017. 20
- [58] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 17

- [59] Bohao Li, Yuying Ge, Yi Chen, Yixiao Ge, Ruimao Zhang, and Ying Shan. Seed-bench-2-plus: Benchmarking multimodal large language models with text-rich visual comprehension. arXiv preprint arXiv:2404.16790, 2024. 11, 12
- [60] Chunyi Li, Jianbo Zhang, Zicheng Zhang, Haoning Wu, Yuan Tian, Wei Sun, Guo Lu, Xiaohong Liu, Xiongkuo Min, Weisi Lin, et al. R-bench: Are your large multimodal model robust to real-world corruptions? arXiv preprint arXiv:2410.05474, 2024. 13
- [61] Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Timothy Baldwin. Cmmlu: Measuring massive multitask language understanding in chinese. arXiv preprint arXiv:2306.09212, 2023. 10, 20
- [62] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 17
- [63] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195–22206, 2024. 2, 8, 17
- [64] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In The Conference on Empirical Methods in Natural Language Processing, pages 292–305, 2023. 14, 15
- [65] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023. 7, 8
- [66] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26689–26699, 2024. 12, 17
- [67] Kevin Qinghong Lin, Linjie Li, Difei Gao, Zhengyuan Yang, Shiwei Wu, Zechen Bai, Weixian Lei, Lijuan Wang, and Mike Zheng Shou. Showui: One vision-language-action model for gui visual agent, 2024. 18
- [68] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024. 7, 21
- [69] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European Conference on Computer Vision, pages 38–55. Springer, 2025. 15
- [70] Wentao Liu, Qianjun Pan, Yi Zhang, Zhuo Liu, Ji Wu, Jie Zhou, Aimin Zhou, Qin Chen, Bo Jiang, and Liang He. Cmm-math: A chinese multimodal math dataset to evaluate and enhance the mathematics reasoning of large multimodal models. arXiv preprint arXiv:2409.02834, 2024. 7
- [71] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281,

2023. 2, 8, 14

- [72] Yuliang Liu, Zhang Li, Hongliang Li, Wenwen Yu, Mingxin Huang, Dezhi Peng, Mingyu Liu, Mingrui Chen, Chunyuan Li, Lianwen Jin, et al. On the hidden mystery of ocr in large multimodal models. arXiv preprint arXiv:2305.07895, 2023. 2, 8, 11, 12
- [73] Zihan Liu, Yang Chen, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Acemath: Advancing frontier math reasoning with post-training and reward modeling. arXiv preprint, 2024. 4
- [74] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Oryx mllm: On-demand spatialtemporal understanding at arbitrary resolution. arXiv preprint arXiv:2409.12961, 2024. 17
- [75] Dakuan Lu, Xiaoyu Tan, Rui Xu, Tianchu Yao, Chao Qu, Wei Chu, Yinghui Xu, and Yuan Qi. Scp-116k: A high-quality problem-solution dataset and a generalized pipeline for automated extraction in the higher education science domain, 2025. 4
- [76] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023. 1, 2, 8, 10, 11
- [77] Shiyin Lu, Yang Li, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, and Han-Jia Ye. Ovis: Structural embedding alignment for multimodal large language model. arXiv preprint arXiv:2405.20797, 2024. 11, 12

- [78] Yujie Lu, Dongfu Jiang, Wenhu Chen, William Yang Wang, Yejin Choi, and Bill Yuchen Lin. Wildvision: Evaluating vision-language models in the wild with human preferences. arXiv preprint arXiv:2406.11069, 2024. 8, 13
- [79] Gen Luo, Wenhan Dou, Wenhao Li, Zhaokai Wang, Xue Yang, Changyao Tian, Hao Li, Weiyun Wang, Wenhai Wang, Xizhou Zhu, Yu Qiao, and Jifeng Dai. Mono-internvl-1.5: Towards cheaper and faster monolithic multimodal large language models. arXiv preprint arXiv:2507.12566, 2025. 2
- [80] Gen Luo, Xue Yang, Wenhan Dou, Zhaokai Wang, Jifeng Dai, Yu Qiao, and Xizhou Zhu. Mono-internvl: Pushing the boundaries of monolithic multimodal large language models with endogenous visual pre-training. arXiv preprint arXiv:2410.08202, 2024. 2, 10
- [81] Gen Luo, Yiyi Zhou, Yuxin Zhang, Xiawu Zheng, Xiaoshuai Sun, and Rongrong Ji. Feast your eyes: Mixture-ofresolution adaptation for multimodal large language models. arXiv preprint arXiv:2403.03003, 2024. 2
- [82] Liangchen Luo, Yinxiao Liu, Rosanne Liu, Samrat Phatale, Harsh Lara, Yunxuan Li, Lei Shu, Yun Zhu, Lei Meng, Jiao Sun, et al. Improve mathematical reasoning in language models by automated process supervision. arXiv preprint arXiv:2406.06592, 2, 2024. 1, 7
- [83] Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Tianjun Zhang, Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl, 2025. Notion Blog. 7
- [84] MAA. American invitational mathematics examination - aime. https://maa.org/

- maa-invitational-competitions/, 2024. 1, 2, 8, 20

[85] MAA. American invitational mathematics examination - aime. https://maa.org/

- maa-invitational-competitions/, 2025. 2, 8, 20

- [86] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11–20, 2016. 15
- [87] Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Proceedings of the Annual Meeting of the Association for Computational Linguistics, pages 2263–2279, 2022. 11, 12
- [88] Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1697–1706, 2022. 11, 12
- [89] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 2200–2209, 2021. 11, 12
- [90] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, et al. Mm-eureka: Exploring visual aha moment with rule-based large-scale reinforcement learning. arXiv preprint arXiv:2503.07365, 2025. 1, 7
- [91] Fanqing Meng, Jin Wang, Chuanhao Li, Quanfeng Lu, Hao Tian, Jiaqi Liao, Xizhou Zhu, Jifeng Dai, Yu Qiao, Ping Luo, et al. Mmiu: Multimodal multi-image understanding for evaluating large vision-language models. arXiv preprint arXiv:2408.02718, 2024. 12, 13
- [92] Meta AI. Llama 4 maverick 17b-128e (model card). Hugging Face, 2025. Released Apr 5, 2025; MoE with 17B activated / 400B total; native multimodality; knowledge cutoff Aug 2024. 21
- [93] Meta AI. Llama 4 scout 17b-16e (model card). Hugging Face, 2025. Model release date: Apr 5, 2025; MoE 17B active / 109B total; 10M context. 21
- [94] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling. arXiv preprint arXiv:2501.19393,

2025. 4

- [95] OpenAI. Gpt-4v(ision) system card. https://cdn.openai.com/papers/GPTV_System_Card.pdf,

2023. 12, 13, 14, 16, 17

- [96] OpenAI. Computer-using agent: Introducing a universal interface for ai to interact with the digital world. 2025. 18
- [97] OpenAI. Gpt-4o system card. https://openai.com/index/gpt-4o-system-card/, 2025. 12, 13, 14, 16, 17, 18, 19, 20, 21

- [98] OpenAI. Gpt-5 system card. https://cdn.openai.com/pdf/ 8124a3ce-ab78-4f06-96eb-49ea29ffb52f/gpt5-system-card-aug7.pdf, 2025. 2, 3, 10, 11, 19, 20
- [99] OpenAI. Introducing gpt-oss. https://openai.com/index/introducing-gpt-oss/, 2025. 3
- [100] Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, Zhuoma GongQue, Shanglin Lei, Zhe Wei, Miaoxuan Zhang, et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning? arXiv preprint arXiv:2407.01284, 2024. 2, 8, 10, 11
- [101] Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. Ui-tars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326,

2025. 1, 18

- [102] Zeju Qiu, Weiyang Liu, Haiwen Feng, Zhen Liu, Tim Z Xiao, Katherine M Collins, Joshua B Tenenbaum, Adrian Weller, Michael J Black, and Bernhard Schölkopf. Can large language models understand symbolic graphics programs? arXiv preprint arXiv:2408.08313, 2024. 2, 10, 19, 20
- [103] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019. 4
- [104] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024. 5
- [105] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 12, 13, 14, 17
- [106] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024. 1, 2, 8
- [107] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 8732–8740, 2020. 20
- [108] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 5
- [109] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 1, 5
- [110] Wei Shen, Jiangbo Pei, Yi Peng, Xuchen Song, Yang Liu, Jian Peng, Haofeng Sun, Yunzhuo Hao, Peiyu Wang, and Yahui Zhou. Skywork-r1v3 technical report. arXiv preprint arXiv:2507.06167, 2025. 11
- [111] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024. 7
- [112] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8317–8326, 2019. 11, 12
- [113] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024. 7
- [114] Hai-Long Sun, Da-Wei Zhou, Yang Li, Shiyin Lu, Chao Yi, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, De-Chuan Zhan, et al. Parrot: Multilingual visual instruction tuning. arXiv preprint arXiv:2406.02539, 2024. 16
- [115] Kai Sun, Dian Yu, Dong Yu, and Claire Cardie. Investigating prior knowledge for challenging chinese machine reading comprehension. Transactions of the Association for Computational Linguistics, 8:141–155, 2020. 20
- [116] Qiushi Sun, Zhoumianze Liu, Chang Ma, Zichen Ding, Fangzhi Xu, Zhangyue Yin, Haiteng Zhao, Zhenyu Wu, Kanzhi Cheng, Zhaoyang Liu, et al. Scienceboard: Evaluating multimodal autonomous agents in realistic scientific workflows. arXiv preprint arXiv:2505.19897, 2025. 1
- [117] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, et al. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022. 20

- [118] Suzhongling, Rong Fu, Weihan Cao, Jianfei Gao, Minxi Jin, PeiZhilin, and Hui Wang. TMA-adaptive FP8 grouped GEMM: Eliminating padding requirements in low-precision training and inference on hopper. In ES-FoMo III: 3rd Workshop on Efficient Systems for Foundation Models, 2025. 7
- [119] Jingqun Tang, Qi Liu, Yongjie Ye, Jinghui Lu, Shu Wei, Chunhui Lin, Wanqing Li, Mohamad Fitri Faiz Bin Mahmood, Hao Feng, Zhen Zhao, et al. Mtvqa: Benchmarking multilingual text-centric visual question answering. arXiv preprint arXiv:2405.11985, 2024. 8, 16
- [120] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 16
- [121] Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, et al. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025. 2, 10, 19
- [122] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, et al. Gemma 3 technical report. arXiv preprint arXiv:2503.19786,

2025. 11, 12, 13, 14, 16, 18, 19, 20, 21

- [123] Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295, 2024. 20
- [124] InternLM Team. Internlm: A multilingual language model with progressively enhanced capabilities. https: //github.com/InternLM/InternLM, 2023. 4
- [125] Kimi Team, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, et al. Kimi-vl technical report. arXiv preprint arXiv:2504.07491, 2025. 10, 11, 12, 13, 14, 16, 17, 19, 20, 21
- [126] Kwai Keye Team, Biao Yang, Bin Wen, Changyi Liu, Chenglong Chu, Chengru Song, Chongling Rao, Chuan Yi, Da Li, Dunju Zang, et al. Kwai keye-vl technical report. arXiv preprint arXiv:2507.01949, 2025. 1, 3, 11, 12, 13, 14, 16, 17, 19, 20, 21
- [127] Qwen Team. Qvq: To see the world with wisdom, December 2024. 11
- [128] Qwen Team. Qwen2.5: A party of foundation models. https://qwenlm.github.io/blog/qwen2.5/, September 2024. 1
- [129] StepFun Team. Step3: Cost-effective multimodal intelligence. 2, 10, 11, 19, 20, 21
- [130] Philippe Tillet, Hsiang-Tsung Kung, and David Cox. Triton: an intermediate language and compiler for tiled neural network computations. In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, pages 10–19, 2019. 7
- [131] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860, 2024. 12, 13, 14
- [132] Fei Wang, Xingyu Fu, James Y Huang, Zekun Li, Qin Liu, Xiaogeng Liu, Mingyu Derek Ma, Nan Xu, Wenxuan Zhou, Kai Zhang, et al. Muirbench: A comprehensive benchmark for robust multi-image understanding. arXiv preprint arXiv:2406.09411, 2024. 12, 13
- [133] Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vl-rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint arXiv:2504.08837, 2025. 1
- [134] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. arXiv preprint arXiv:2402.14804, 2024. 1, 2, 10, 11
- [135] Peijie Wang, Zhong-Zhi Li, Fei Yin, Dekang Ran, and Cheng-Lin Liu. Mv-math: Evaluating multimodal math reasoning in multi-visual contexts. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19541–19551, 2025. 7
- [136] Peiyi Wang, Lei Li, Zhihong Shao, RX Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. arXiv preprint arXiv:2312.08935, 2023. 1
- [137] Peiyu Wang, Yichen Wei, Yi Peng, Xiaokun Wang, Weijie Qiu, Wei Shen, Tianyidan Xie, Jiangbo Pei, Jianhao Zhang, Yunzhuo Hao, et al. Skywork r1v2: Multimodal hybrid reinforcement learning for reasoning. arXiv preprint arXiv:2504.16656, 2025. 1

- [138] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 3, 11, 12, 13, 14, 15, 16, 17
- [139] Peng Wang, Shijie Wang, Junyang Lin, Shuai Bai, Xiaohuan Zhou, Jingren Zhou, Xinggang Wang, and Chang Zhou. One-peace: Exploring one general representation model toward unlimited modalities. arXiv:2305.11172, 2023. 15
- [140] Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Xiaotao Gu, Shiyu Huang, Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang. Lvbench: An extreme long video understanding benchmark, 2025. 2
- [141] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079, 2023. 15
- [142] Weiyun Wang, Zhe Chen, Wenhai Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Jinguo Zhu, Xizhou Zhu, Lewei Lu, Yu Qiao, and Jifeng Dai. Enhancing the reasoning ability of multimodal large language models via mixed preference optimization. arXiv preprint arXiv:2411.10442, 2024. 2, 3, 5, 7
- [143] Weiyun Wang, Zhangwei Gao, Lianjie Chen, Zhe Chen, Jinguo Zhu, Xiangyu Zhao, Yangzhou Liu, Yue Cao, Shenglong Ye, Xizhou Zhu, et al. Visualprm: An effective process reward model for multimodal reasoning. arXiv preprint arXiv:2503.10291, 2025. 1, 7, 11
- [144] Weiyun Wang, Yiming Ren, Haowen Luo, Tiantong Li, Chenxiang Yan, Zhe Chen, Wenhai Wang, Qingyun Li, Lewei Lu, Xizhou Zhu, et al. The all-seeing project v2: Towards general relation comprehension of the open world. arXiv preprint arXiv:2402.19474, 2024. 14, 15
- [145] Xuehui Wang, Zhenyu Wu, JingJing Xie, Zichen Ding, Bowen Yang, Zehao Li, Zhaoyang Liu, Qingyun Li, Xuan Dong, Zhe Chen, Weiyun Wang, Xiangyu Zhao, Jixuan Chen, Haodong Duan, Tianbao Xie, Shiqian Su, Chenyu Yang, Yue Yu, Yuan Huang, Yiqian Liu, Xiao Zhang, Xiangyu Yue, Weijie Su, Xizhou Zhu, Wei Shen, Jifeng Dai, and Wenhai Wang. Mmbench-gui: Hierarchical multi-platform evaluation framework for gui agents. arXiv preprint arXiv:2507.19478, 2025. 18
- [146] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. arXiv preprint arXiv:2406.01574, 2024. 2
- [147] Zhaokai Wang, Xizhou Zhu, Xue Yang, Gen Luo, Hao Li, Changyao Tian, Wenhan Dou, Junqi Ge, Lewei Lu, Yu Qiao, and Jifeng Dai. Parameter-inverted image pyramid networks for visual perception and multimodal understanding. IEEE transactions on pattern analysis and machine intelligence, 2025. 2
- [148] Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, et al. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. arXiv preprint arXiv:2406.18521, 2024. 11, 12
- [149] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. arXiv preprint arXiv:2407.15754, 2024. 17
- [150] Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, et al. Os-atlas: A foundation action model for generalist gui agents. arXiv preprint arXiv:2410.23218, 2024. 10, 18
- [151] xAI. Grok 3 beta — the age of reasoning agents. https://x.ai/news/grok-3, 2025. 21
- [152] Zhiheng Xi, Guanyu Li, Yutao Fan, Honglin Guo, Yufang Liu, Xiaoran Fan, Jiaqi Liu, Jingchao Ding, Wangmeng Zuo, Zhenfei Yin, et al. Bmmr: A large-scale bilingual multimodal multi-discipline reasoning dataset. arXiv preprint arXiv:2507.03483, 2025. 1
- [153] Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts. arXiv preprint arXiv:2407.04973, 2024. 2, 8, 10, 11
- [154] LLM-Core-Team Xiaomi. Mimo-vl technical report, 2025. 3, 11, 18, 19
- [155] Tianbao Xie, Jiaqi Deng, Xiaochuan Li, Junlin Yang, Haoyuan Wu, Jixuan Chen, Wenjing Hu, Xinyuan Wang, Yuhui Xu, Zekun Wang, Yiheng Xu, Junli Wang, Doyen Sahoo, Tao Yu, and Caiming Xiong. Scaling computer-use grounding via user interface decomposition and synthesis, 2025. 18
- [156] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh J Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. Advances in Neural Information Processing Systems, 37:52040–52094, 2024. 1, 10, 18

- [157] Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tianbao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, and Caiming Xiong. Aguvis: Unified pure vision agents for autonomous gui interaction. arXiv preprint arXiv:2412.04454, 2024. 18
- [158] B. Yan, Yi Jiang, Jiannan Wu, D. Wang, Ping Luo, Zehuan Yuan, and Huchuan Lu. Universal instance perception as object discovery and retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. 15
- [159] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 3
- [160] Chenyu Yang, Shiqian Su, Shi Liu, Xuan Dong, Yue Yu, Weijie Su, Xuehui Wang, Zhaoyang Liu, Jinguo Zhu, Hao Li, et al. Zerogui: Automating online gui learning at zero human cost. arXiv preprint arXiv:2505.23762, 2025. 1
- [161] Jihan Yang, Shusheng Yang, Anjali Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in Space: How Multimodal Large Language Models See, Remember and Recall Spaces. arXiv preprint arXiv:2412.14171, 2024. 1, 2, 10, 19
- [162] Yan Yang, Dongxu Li, Yutong Dai, Yuhao Yang, Ziyang Luo, Zirui Zhao, Zhiyuan Hu, Junzhe Huang, Amrita Saha, Zeyuan Chen, et al. Gta1: Gui test-time scaling agent. arXiv preprint arXiv:2507.05791, 2025. 18
- [163] Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615, 2025. 7
- [164] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024. 3, 11, 12, 13, 14, 16, 17, 19
- [165] Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Haowei Liu, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration. arXiv preprint arXiv:2311.04257, 2023. 16
- [166] Kaining Ying, Fanqing Meng, Jin Wang, Zhiqian Li, Han Lin, Yue Yang, Hao Zhang, Wenbo Zhang, Yuqi Lin, Shuo Liu, Jiayi Lei, Quanfeng Lu, Runjian Chen, Peng Xu, Renrui Zhang, Haozhe Zhang, Peng Gao, Yali Wang, Yu Qiao, Ping Luo, Kaipeng Zhang, and Wenqi Shao. Mmt-bench: A comprehensive multimodal benchmark for evaluating large vision-language models towards multitask agi. arXiv preprint arXiv:2404.16006, 2024. 12, 13
- [167] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476,

2025. 5

- [168] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023. 2, 8, 14
- [169] Ya-Qi Yu, Minghui Liao, Jiwen Zhang, and Jihao Wu. Texthawk2: A large vision-language model excels in bilingual ocr and grounding with 16x fewer tokens. arXiv preprint arXiv:2410.05261, 2024. 15
- [170] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. arXiv preprint arXiv:2311.16502, 2023. 1, 2, 8, 10, 11
- [171] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025. 1
- [172] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In Proceedings of the Annual Meeting of the Association for Computational Linguistics, pages 4791–4800, 2019. 20
- [173] Haotian Zhang, Haoxuan You, Philipp Dufter, Bowen Zhang, Chen Chen, Hong-You Chen, Tsu-Jui Fu, William Yang Wang, Shih-Fu Chang, Zhe Gan, et al. Ferret-v2: An improved baseline for referring and grounding with large language models. arXiv preprint arXiv:2404.07973, 2024. 15
- [174] Junlei Zhang, Zichen Ding, Chang Ma, Zijie Chen, Qiushi Sun, Zhenzhong Lan, and Junxian He. Breaking the data barrier – building gui agents through task generalization. arXiv preprint arXiv:2504.10127, 2025. 1

- [175] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? arXiv preprint arXiv:2403.14624, 2024. 2, 8, 10, 11
- [176] Tianyu Zhang, Suyuchen Wang, Lu Li, Ge Zhang, Perouz Taslakian, Sai Rajeswar, Jie Fu, Bang Liu, and Yoshua Bengio. Vcr: Visual caption restoration. arXiv preprint arXiv:2406.06462, 2024. 11, 12
- [177] Xiaotian Zhang, Chunyang Li, Yi Zong, Zhengyu Ying, Liang He, and Xipeng Qiu. Evaluating the performance of large language models on gaokao benchmark. arXiv preprint arXiv:2305.12474, 2023. 1, 2, 10, 20
- [178] Yi-Fan Zhang, Huanyu Zhang, Haochen Tian, Chaoyou Fu, Shuangqing Zhang, Junfei Wu, Feng Li, Kun Wang, Qingsong Wen, Zhang Zhang, et al. Mme-realworld: Could your multimodal llm challenge high-resolution real-world scenarios that are difficult for humans? arXiv preprint arXiv:2408.13257, 2024. 2, 8, 13
- [179] Yichi Zhang, Siyuan Zhang, Yao Huang, Zeyu Xia, Zhengwei Fang, Xiao Yang, Ranjie Duan, Dong Yan, Yinpeng Dong, and Jun Zhu. Stair: Improving safety alignment with introspective reasoning. arXiv preprint arXiv:2502.02384,

2025. 7

- [180] Yipeng Zhang, Yifan Liu, Zonghao Guo, Yidan Zhang, Xuesong Yang, Chi Chen, Jun Song, Bo Zheng, Yuan Yao, Zhiyuan Liu, et al. Llava-uhd v2: an mllm integrating high-resolution feature pyramid via hierarchical window transformer. arXiv preprint arXiv:2412.13871, 2024. 2
- [181] Bingchen Zhao, Yongshuo Zong, Letian Zhang, and Timothy Hospedales. Benchmarking multi-image understanding in vision and language models: Perception, knowledge, reasoning, and multi-hop reasoning. arXiv preprint arXiv:2406.12742, 2024. 12, 13
- [182] Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al. Pytorch fsdp: experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277, 2023. 7
- [183] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025. 1, 2, 5
- [184] Rui Zheng, Shihan Dou, Songyang Gao, Yuan Hua, Wei Shen, Binghai Wang, Yan Liu, Senjie Jin, Qin Liu, Yuhao Zhou, et al. Secrets of rlhf in large language models part i: Ppo. arXiv preprint arXiv:2307.04964, 2023. 5
- [185] Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023. 2, 10
- [186] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: A comprehensive benchmark for multi-task long video understanding. arXiv preprint arXiv:2406.04264, 2024. 8, 17
- [187] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21
- [188] Qihao Zhu, Daya Guo, Zhihong Shao, Dejian Yang, Peiyi Wang, Runxin Xu, Y Wu, Yukun Li, Huazuo Gao, Shirong Ma, et al. Deepseek-coder-v2: Breaking the barrier of closed-source models in code intelligence. arXiv preprint arXiv:2406.11931, 2024. 20
- [189] Chengke Zou, Xingang Guo, Rui Yang, Junyu Zhang, Bin Hu, and Huan Zhang. Dynamath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models. arXiv preprint arXiv:2411.00836,

2024. 2, 8, 10, 11

