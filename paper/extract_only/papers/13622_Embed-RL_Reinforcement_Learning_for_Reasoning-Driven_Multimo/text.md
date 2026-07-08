# arXiv:2602.13823v3[cs.CV]12Mar2026

## Embed-RL: Reinforcement Learning for Reasoning-Driven Multimodal Embeddings

Haonan Jiang1,2∗, Yuji Wang1,2∗, Yongjie Zhu2†, Xin Lu2, Wenyu Qin2, Meng Wang2, Pengfei Wan2, and Yansong Tang1‡

1Tsinghua Shenzhen International Graduate School, Tsinghua University 2Kling Team, Kuaishou Technology {jiang-hn24@mails, yuji-wan24@mails, tang.yansong@sz}.tsinghua.edu.cn {zhuyongjie, luxin09, qinwenyu, wangmeng46, wanpengfei}@kuaishou.com

Abstract. Leveraging Multimodal Large Language Models (MLLMs) has become pivotal for advancing Universal Multimodal Embeddings (UME) in addressing diverse cross-modal tasks. Recent studies demonstrate that incorporating generative Chain-of-Thought (CoT) reasoning can substantially enhance task-specific representations compared to discriminative methods. However, the generated reasoning CoTs of existing generative embedding methods are limited to the textual analysis of queries and are irrelevant to the retrieval of the targets. To address these limitations, we propose a reasoning-driven UME framework that integrates Embedder-Guided Reinforcement Learning (EG-RL) to optimize the Reasoner to produce evidential Traceability CoT (T-CoT). Our key contributions are threefold: (1) We design an EG-RL framework where the Embedder provides explicit supervision to the Reasoner, ensuring the generated CoT traces are aligned with embedding tasks. (2) We introduce T-CoT, which extracts critical multimodal cues to focus on retrieval-relevant elements and provides multimodal inputs for the Embedder. (3) With limited computational resources, our framework outperforms the pioneering embedding model on both MMEB-V2 and UVRB benchmarks. The integration of multimodal evidence in structured reasoning, paired with retrieval-oriented alignment, effectively strengthens cross-modal semantic consistency and boosts the model’s fine-grained matching capability as well as its generalization across complex scenarios. Our work demonstrates that targeted reasoning optimization can significantly improve multimodal embedding quality, providing a practical and efficient solution for reasoning-driven UME development. Project page.

Keywords: Multimodal Embedding · Generative Reasoning · Reinforcement Learning

#### 1 Introduction

Multimodal embedding, as a core supporting technology for cross-modal tasks, has been widely applied to numerous important directions such as image-text

∗: Equal Contribution. Work done during an internship at Kuaishou Technology. †: Project Leader. ‡: Corresponding Author.

retrieval, video moment localization, and visual document understanding [26,36]. Traditional multimodal embedding methods adopt dual-encoder architectures, such as CLIP [43], BLIP [34], and SigLIP [67]. These methods demonstrate weaker ability in bridging the gap between different modalities compared with Multimodal Large Language Models (MLLMs) [2, 3, 31, 32, 51]. Additionally, MLLMs benefit from their strong multimodal understanding and instructionfollowing capabilities, enabling them to adapt to diverse and complex task requirements. Therefore, an increasing body of literature [18,26,41,46,63] proves that MLLMs can be used to learn Universal Multimodal Embedding (UME) that captures general-purpose content similarity. Meanwhile, evaluation benchmarks such as the Multimodal Embedding Benchmark (MMEB) [26] and its upgraded version MMEB-V2 [41] have addressed this academic demand for UME research, covering 78 instruction-aware tasks across three modalities.

[Figure 1]

[Figure 2]

MLLM

MLLM

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Reasoner

Reasoner

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Autoregressive

GRPO

[Figure 11]

[Figure 12]

Visual

Text

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Text

Generated CoT

Visual

Visual

Text

Visual

Text

Evidential

[Figure 18]

Tokens

Tokens

Textual CoT

Tokens

Tokens

Tokens

Tokens

Tokens

Tokens

Traceability CoT

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Embedder

Guided

Reward

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

MLLM Embedder

MLLM Embedder

MLLM Embedder

MLLM Embedder

[Figure 33]

Last Discriminative Embedding Generative Embedding Decoupled Reasoner Embedding EG-RL Embedding

Last Last Last

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

(a) Comparison of different multimodal embedding frameworks

VLM2Vec-V2.0 DUME-7B

Query, Instruction,

[Figure 41]

###### Query T-CoT Before RL: Irrelevant to Retrieval

Similarity: 0.5664

and Images

UME-R1-7B Embed-RL-4B

<thinking>The provided image is a vintage Pepsi-Cola advertisement featuring a glass of cola, not a bottle.

VD-OOD VD-VisRAG

Given an image, find a similar

Since the current image does not contain an orange bottle, I cannot locate such an element within this

[Figure 42]

I-CLS

everyday image with the

image. {"bbox_2d": []} </thinking> <rethink>There is no orange soft drink bottle in the provided image, so no bounding box can be provided for a matching element.</rethink> <answer>No orange soft drink bottle in the provided image.</answer>

described changes: Change to real soft drink bottle in orange.

VDVidore-V2

I-QA

[Figure 43]

| |
|---|

Query

Query T-CoT After RL: Retrieval Targeted

Similarity: 0.5977

Image

VD-

I-RET

<thinking>A suitable replacement would be an image of a real orange-colored soft drink bottle, such as an

Vidore-V1

orange Fanta, in a similar advertising or product display context. {"bbox_2d": [120, 100, 320, 500]} </thinking>

[Figure 44]

<rethink>The bounding box corresponds to the glass of cola, so the replacement bottle must be visually

Target

V-MR

I-VG

identifiable as orange and part of a similar context. </rethink>

Image

<answer> Replace the glass of cola in the original image with a real orange soft drink bottle such as Fanta.

</answer>

V-RET

V-CLS

V-QA

(b) RL-Evolved Evidential Traceability CoT Reasoning (c) Performance Comparison

- Fig. 1: Multimodal embedding optimization via Embedder-Guided Reinforcement Learning (EG-RL). (a) Frameworks evolution. (b) Reasoning enhancement with RL-optimized evidential Traceability CoT (T-CoT). (c) Comparison of multi-task performance.

Currently, most MLLM-powered embedding methods are discriminative embedding models [29,39,71]. These models typically extract embedding features directly from the final hidden states of input tokens, failing to fully leverage MLLMs’ inherent generative capabilities and reasoning potential. Consequently, recent studies explore integrating generative reasoning into UME tasks, as shown in Figure 1. For instance, approaches like UME-R1 [30] unify discriminative and generative embeddings through textual Chain-of-Thoughts (CoTs) generated by the MLLM Embedder. However, simultaneously optimizing contrastive loss and next-token prediction objectives causes conflicting gradients, leading to suboptimal performance [8,30]. In contrast, the decoupled Reasoner-Embedder paradigm

proposed by TTE [13], which relies on massive computational resources and data distillation from large models, aims to alleviate this problem by decoupling the two processes, using pre-trained MLLMs to generate offline CoT reasoning to enhance embedding quality with only the Embedder trained. Nevertheless, the CoTs generated by the Reasoner of TTE are not specifically designed for embedding, as they are not trained together with the Embedder. This misalignment may introduce noise and even lead to hallucinations. Moreover, relying only on textual reasoning fails to fully leverage the MLLM Embedder’s potential to process multimodal signals, whose rich representations could significantly enhance retrieval performance. Such insufficiency of multimodal cues leads to notable embedding alignment bias, where critical visual-spatial cues and video-temporal signals are not fully captured in retrieval, resulting in less effective cross-modal matching and restricted generalization on complex real-world multimodal tasks.

To address the above issues, this paper proposes a reasoning-driven decoupled UME framework. This framework leverages the Embedder-Guided Reinforcement Learning (EG-RL) algorithm to optimize the CoTs generated by the Reasoner, using our novel process reward oriented to the alignment between query and target and verifiable outcome reward for retrieval. Firstly, we construct a dataset that initially trains the Embedder to generate high-quality embeddings conditioned on the sequence of preceding input and CoT tokens. The trained Embedder acts as a reward model and provides stable and reliable reward signals. Secondly, inspired by the region-aware paradigm [15,45,47,75] that makes the model focus on the region of interest (RoI), we propose the evidential Traceability CoT (TCoT) that explicitly guides the Embedder to focus on task-related information, effectively filter out redundant visual elements, and integrate modality-specific critical cues to adapt to long-text retrieval, coarse-grained semantic matching, and fine-grained alignment for robust performance across heterogeneous tasks. The main contributions of this paper are summarized as follows:

- 1. Embedder-Guided Reinforcement Learning. We propose a novel decoupled RL framework where the Embedder guides the Reasoner to optimize CoT trajectories for specific embedding tasks. This approach resolves key conflict between generative and embedding objectives, ensures the Reasoner’s output greatly improves retrieval quality, and addresses core challenge of adapting general CoTs to embedding tasks.
- 2. Evidential Traceability CoT for Embedding. We further extend CoT reasoning to complex multimodal scenarios, integrating explicit visual localization information, video keyframes, and text keywords into detailed inference trajectories. This design enables the model to focus on core retrieval-related information and effectively mitigate the negative impact of redundant multimodal and text data on overall embedding alignment performance.
- 3. Efficient Performance Improvement Across Multiple Benchmarks. Under computationally constrained settings, the framework proposed in this paper outperforms state-of-the-art generative embedding models on both MMEB-V2 [41] and video retrieval UVRB [20] benchmark datasets, and achieves exceptional performance across diverse combinatorial scenarios.

#### 2 Related Work

##### 2.1 Universal Multimodal Embedding

Constructing robust multimodal representations is a fundamental core challenge in multimodal learning. Pioneering models such as CLIP [43] and ALIGN [24] adopt a dual-encoder architecture and learn effective representations through contrastive learning on rich large-scale image-text paired data. However, they struggle to handle interleaved image-text inputs, and their text encoders lack sufficient capacity to understand truly complex textual content. To address this issue, researchers leverage Multimodal Large Language Models (MLLMs) to build embedding models [26,36,39,41,71,77], capitalizing on their strong multimodal comprehension capabilities to enhance overall learning performance.

Existing works focus on different aspects: VLM2Vec [26] transforms MLLMs into embedding models via contrastive learning and achieves outstanding performance on unconventional retrieval tasks such as visual question answering and localization; MM-Embed [36] explores using off-the-shelf MLLMs as zeroshot rerankers to optimize retrieval results; LamRA [39] unifies the multimodal retrieval paradigm through two-stage retrieval training and joint reranking. MegaPairs [77] and GME [72] address the modality imbalance problem with automated pipelines; LLaVE [29] and Unite [27] focus on hard negative sample mining.

Recent studies focus on instruction-aware representations: MMEB [26] and MMEB-V2 [41] construct a comprehensive evaluation benchmark covering 78 tasks. UME-R1 [30] first introduces reasoning mechanisms, yet simultaneous optimization of dual components via Reinforcement Learning (RL) leads to conflicts, and redundant Chain-of-Thought (CoT) trajectories dilute representations. TTE [13] adopts a decoupled yet computationally expensive architecture, and its Reasoner is misaligned with retrieval tasks, resulting in task-irrelevant outputs. In this paper, we propose a decoupled RL framework enabling separate optimization of dual components and generating retrieval-relevant reasoning trajectories through a dual reward mechanism, addressing the aforementioned challenges.

##### 2.2 Multimodal Reasoning with Reinforcement Learning

MLLMs [2,31,38,51] extend and enrich the capabilities of Large Language Models (LLMs) to the multimodal domain, achieving promising results across diverse tasks including visual question answering [3,17,38,50], visual grounding [12,28,33], and keyframe extraction [35, 52, 68, 79]. Early works [23, 53, 58, 60, 73] mostly completed reasoning tasks using standardized CoT prompts. Since DeepSeekR1 [19] proposed the Group Relative Policy Optimization‌ (GRPO) RL algorithm, numerous recent works have optimized advanced RL algorithms [64, 74] and enhanced the reasoning capabilities of MLLMs [6,14,44,56,59].

GRIT [15] interleaves bounding box coordinates with textual reasoning chains and designs an RL scheme based on the GRPO algorithm, enabling efficient training with dual robust rewards and no additional annotated data. GroundR1 [6] proposes an RL framework to achieve grounded visual reasoning without

extra annotations, guiding response generation through dual rewards to improve reasoning reliability and interpretability. BRPO [11] uses Intersection over Union (IoU)-based rewards to guide models to autonomously generate visual-text reflections, combined with a visual token mechanism to mitigate the problems of visual attention dilution and hallucinations. DeepEyes [75] adopts end-to-end RL to induce models to develop the ability of “thinking with images”, improving performance on various reasoning tasks. TreeVGR [47] proposes the TreeBench benchmark and the TreeVGR training paradigm, enhancing visual grounding reasoning capabilities by jointly supervising localization and reasoning via RL. Inspired by grounding reasoning, this paper further proposes evidential Traceability CoT (T-CoT), which constructs structured multimodal reasoning chains by extracting bounding boxes of images, keyframes of videos, and keywords of text. This method enables the model to focus on the core regions of retrieval tasks, thereby improving embedding quality.

#### 3 Methodology

##### 3.1 Preliminaries

We focus on the universal multimodal retrieval task, where given a query q (text, image, or interleaved text-image modalities) and a candidate set Ω = {cn}Nn=1, the goal is to retrieve the most relevant candidate from Ω.

To learn discriminative multimodal embeddings, we adopt contrastive learning with the InfoNCE loss [42], as shown in Figure 2(a). For a query qi, its positive target t+i , and its negative target set T − = {t−j }j̸=i, the InfoNCE loss optimizes the model to maximize the similarity between qi and t+i while minimizing the similarities to all t−j ∈ T −. The loss is defined as:

exp cos(hqi,ht+

)/τ

LInfoNCE = −N1 Ni=1 log

, (1)

i

)/τ + t−∈T− exp(cos(hqi,ht−)/τ)

exp cos(hqi,ht+

i

where hq

and ht are embeddings of qi and target t (extracted as the last-layer hidden states of the last token from a vision-language model), cos(·,·) denotes cosine similarity, and τ is the temperature hyperparameter.

i

##### 3.2 Data Construction

To support the training of reasoning-driven universal multimodal embeddings, we construct a high-quality multimodal dataset following a “sampling-annotationfiltering-splitting” pipeline, as illustrated in Figure 2(a). The dataset integrates diverse modalities (image, video, visual document) and ensures alignment between reasoning trajectories and retrieval objectives through strict quality control.

We first curate the initial data pool by adopting a stratified sampling strategy across three core sources, and referencing the data paradigm of VLM2Vec-V2 [41]:

- (a) Data Construction

- (b) Embedder-Guided Reinforcement Learning

Evidential Traceability CoT (T-CoT) Annotation for Query and Positive

Original Data T-CoT Filtering

Merged Dataset

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

<thinking> XxxX

[Figure 50]

[Figure 51]

20%

"jaggery”, "kitchen utensil"

Query T-CoT

Positive T-CoT

<rethink> xXx <answer> Xxx

[Figure 52]

Keywords

Text Image Video

Reinforcement Learning Data

Query

|[Figure 53]<br><br>| |
|---|
|
|---|

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

<thinking> XxxX <rethink> xXx

80%

Qwen3-VL

Qwen3 -VL

Key

[Figure 58]

<answer> Xxx

Contrastive Learning Data

Objects

Contrastive

Is Match?

Last Last

Loss

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

<thinking> XxxX <rethink> xXx <answer> Xxx

[Figure 69]

[Figure 70]

Embedder Embedder

Key Frames

Text Image Video

[Figure 71]

[Figure 72]

Positive T-CoT

Query T-CoT

Yes / No

[Figure 73]

Query

Positive

Positive

Outcome Reward

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

Positive Similarity G

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

Positive Query Negative

Positive

_

Positive T-CoT G

Similarity G

[Figure 85]

[Figure 86]

Embedder

Negative

weight X

[Figure 87]

[Figure 88]

###### Reward Function

[Figure 89]

Similarity G

[Figure 90]

MLLM Reasoner

[Figure 91]

Query MMCoT

Query MMCoT

Query T-CoT G

Process Reward

Format + Process +Outcome

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Query

[Figure 97]

[Figure 98]

Positive T-CoT G

Negative T-CoT G

[Figure 99]

T-CoT

Embedder

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Reference KL Loss Model

Discriminator

Negative T-CoT G

[Figure 104]

Negative Similarity G

[Figure 105]

Which best matches

Predict

[Figure 106]

[Figure 107]

[Figure 108]

Index

query T-CoT?

- Fig. 2: Overview of the proposed data synthesis and EG-RL framework. (a) Data Construction generates T-CoT annotations for query-positive pairs, filters and splits the dataset to enable contrastive and reinforcement learning, laying the groundwork for reasoning-aware embedding. (b) Embedder-Guided Reinforcement Learning finetunes the MLLM with a process-outcome reward function, encouraging T-CoT trajectories that yield more discriminative and beneficial generative embeddings.

(1) Image-centric tasks from MMEB-train [26], covering image classification, Question Answering, retrieval, and grounding; (2) Video-language instruction data from LLaVA-Hound [70], including video captioning, QA, and retrieval; (3) Visual document retrieval data from ViDoRe [16] and VisRAG [65].

Next, we perform evidential Traceability Chain-of-Thought (T-CoT) annotation for all query-positive pairs. Each T-CoT follows a structured threepart format: (1) <thinking> extracts modality-specific cues (text keywords via text_keywords, image spatial locations via bbox_2d (two-dimensional bounding box), video critical moments via key_frames); (2) <rethink> refines reasoning logic to focus on key retrieval-relevant aspects; (3) the final answer summarizes core retrieval-relevant information. We design task-specific prompts for annotation, ensuring T-CoT aligns with diverse multimodal retrieval scenarios.

Following annotation, we perform a strict CoT-guided relevance filtering to eliminate noisy samples. With a custom-designed judgment prompt, we assess whether the T-CoTs of queries and their positive samples are clearly irrelevant or contradictory to the task description. We retain only samples labeled “No”, meaning those that are relevant and not contradictory, for contrastive learning.

This filtering step effectively mitigates noise interference in contrastive learning. The initial dataset contains 2.22 million samples, and 1.83 million are preserved after filtering, yielding a retention rate of approximately 80%. Approximately 20% of the filtered-out samples are uniformly sampled and used in the reinforcement learning stage, as these hard examples are valuable for model exploration in reinforcement learning. In addition, we assign training weights to different datasets based on their task importance and data quality.

The constructed dataset exhibits three features: (1) modal diversity, covering three modalities of text, image, and video; (2) reasoning alignment, where TCoT explicitly integrates multimodal cues and retrieval-related logic, avoiding informational redundancy; (3) quality assurance, as rigorous filtering and weighted sampling ensure that the dataset is free of significant noise and balanced across tasks. This dataset lays a solid foundation for the two-stage training, enabling Embedder to learn reasoning-aware representations and the Reasoner to optimize the generation of retrieval-centric T-CoTs through reinforcement learning.

##### 3.3 Embedder-Guided Reinforcement Learning

To address the misalignment between generative reasoning and embedding objectives, we propose a decoupled reinforcement learning framework in which the pre-trained Embedder provides supervision to the Reasoner. This framework optimizes the generation of T-CoT to prioritize retrieval-relevant multimodal cues, leveraging a dual-guidance reward mechanism and Group Relative Policy Optimization (GRPO) [19]. Figure 2(b) illustrates the workflow of this stage.

EG-RL Framework Design. First, we fully train an Embedder using the InfoNCE loss to equip it with robust embedding capabilities. Our RL framework maintains strict separation between two components: the Reasoner, which is responsible for generating T-CoT, and the Embedder, which is frozen after contrastive training. This decoupling ensures three key benefits: (1) targeted optimization of reasoning without disrupting the Embedder’s learned discriminative capabilities; (2) stable reward signals from the frozen Embedder that consistently evaluate T-CoT quality based on embedding alignment; and (3) flexible integration of multi-source rewards to internalize both retrieval and reranking knowledge. The Reasoner takes multimodal queries as input and outputs structured T-CoT, which integrates three critical cues: text keywords, image bounding boxes, and video keyframes. Additionally, we recrop the content within the bounding boxes and keyframes based on T-CoT to achieve multimodal reasoning-aware embeddings. This structured reasoning is then concatenated with the original input to form the Embedder’s input, denoted as I:

I = [xtext, ximg, xvid, T-CoT(x), <emb>]. (2)

In this equation, <emb> is a special token whose hidden state is extracted as the final embedding, and the evaluation of this embedding by the Embedder directly guides the Reasoner’s policy update.

Reward Function with Process and Outcome Guidance. We design a three-component reward function to align T-CoT generation with embedding

quality, combining format compliance, outcome-level retrieval effectiveness, and process-level T-CoT alignment:

Format Reward (Rformat): This reward ensures T-CoT strictly follows the predefined template (<thinking> → <rethink> → <answer>) and includes all required multimodal cues. Reward 1 for full compliance, 0 otherwise, guaranteeing T-CoT output interpretability and compatibility with the Embedder module.

Embedder-Guided Outcome Reward (Routcome): This reward measures how T-CoT improves embedding alignment by jointly assessing the ranking accuracy of positive samples and the similarity margin between positive and hard negative samples. The margin is a softmax-weighted average of negative similarities scaled by a temperature parameter. For a query qi with positive target t+i and in-batch negatives {t−j }j̸=i, embeddings are eq

###### = πe(qi,oqi) and et

i

= πe(tj,otj), where oqi and otj are T-CoT outputs for query and target. The reward for oqi is defined as:

j

Routcome(oqi) = Acck(eq

###### ,t+i ) · sim(eq,et+

###### ) − Eτ sim(eq

i

i

i

) , (3)

###### ,et−

j

###### ,t+i ) denotes the top-k retrieval accuracy, which measures whether t+i is among the top-k ranked targets when sorted by cosine similarity to eq

where Acck(eq

i

;

i

sim(·,·) denotes the cosine similarity between normalized embeddings; and Eτ[·] stands for the softmax-weighted average of cosine similarities between eq

and embeddings of in-batch negative targets.

i

Additionally, we compute Routcome symmetrically for positive targets: taking t+i as the anchor, qi as its positive query, and other in-batch queries as negatives to calculate Routcome for oti. This symmetric computation enforces consistent embedding alignment in both query-to-target and target-to-query directions. This reward optimizes T-CoT with embedding learning as the core objective, enhancing its discriminative ability across samples.

T-CoT Process Reward (Rprocess): We employ an independent pretrained Vision-Language Model Discriminator D for listwise comparison to align TCoT outputs of queries and targets. Let qcot be the query’s T-CoT output and {cjcot}mj=1 the T-CoT outputs of m in-batch candidate targets, comprising positive samples from multiple rollouts of the query’s data pair and negative samples. After shuffling, the index set of ground-truth positives is denoted P. To mitigate position bias, we feed qcot and shuffled {cjcot}mj=1 to D as pairwise inputs. The reward quantifies alignment via D’s selection correctness, formally defined as:

1, if D qcot,{cjcot}mj=1 ∈ P, 0, otherwise,

(4)

Rprocess(oi) =

where oi denotes the T-CoT generation outcome of the i-th sample, and D(·,·) outputs the index of the candidate T-CoT most aligned with qcot in the shuffled candidate set. A reward of 1 indicates that D correctly selects a positive T-CoT from the ground-truth set, signifying well-aligned query-target T-CoT pairs; a reward of 0 means D fails to select any positive T-CoT, indicating misalignment between query and target T-CoT outputs. We compute Rprocess symmetrically in

the reverse direction, moving from positive targets to their corresponding queries, to ensure consistent embedding alignment across both directions.

This reward guides Reasoner to align query and target T-CoT outputs. Since T-CoT is the intermediate process for embedding generation, T-CoT alignment quantified by D selection correctness directly improves embedding quality.

The total reward is a weighted combination of these three components: Rtotal = αRformat + βRprocess + γRoutcome, (5)

where α,β,γ are non-negative weighting coefficients that balance the relative contributions of the three reward components in the total reward optimization.

Policy Optimization with GRPO. We adopt GRPO to optimize the Reasoner’s policy, and use group-based rewards to stabilize the training process. For each query-target pair q ∼ S, where S denotes the training sample set of query-target pairs. We sample G = 8 candidate T-CoT sequences {oi}Gi=1 according to the old policy πθ

. The optimization objective is defined as:

old

Lgrpo = E q∼S,

{oi}∼πθold

G

1 G

i=1

min(rθ(oi)Ai,clip(rθ(oi),1 − ϵ,1 + ϵ)Ai)

− βDKL(πθ∥πref) ,

(6)

where rθ(oi) = πθ(oi|q)/πθ

(oi|q) denotes importance ratio, ϵ is the clipping threshold of importance ratio, β is a hyperparameter weighting the KullbackLeibler divergence term, πref denotes the reference policy model before optimization, and Ai = (ri − µr)/σr represents advantage, where µr = mean({r1,...,rG}) and σr = std({r1,...,rG}) are the mean and std of group rewards.

old

#### 4 Experiments

##### 4.1 Implementation Details

We train Qwen3-VL-2B [2] and Qwen3-VL-4B [2] as Embedders with the DeepSpeed Zero2 optimization framework, and adopt a sub-batch strategy following VLM2Vec [26]. The models are trained for 2 epochs with a learning rate of 1e-4 and weight decay of 0.01; the batch size is computationally light: 512 for the 2B model and 256 for the 4B model, and we use Low-Rank Adaptation (LoRA) [21] for fine-tuning. For reinforcement learning, Qwen3-VL-8B [2] is trained as Reasoner via the GRPO algorithm [19] for 1 epoch, with a computationally light batch size of 256, a learning rate of 3e-6, and standard GRPO hyperparameters.

##### 4.2 Baselines and Datasets

We compare with representative multimodal embedding models with diverse architectures, modalities and scales. These baselines cover image, video and

visual document retrieval, ensuring thorough and fair evaluation. We evaluate against GME [72], ColPali [16], VLM2Vec [26], LamRA [39], CAFe [63], VLM2VecV2 [41], TTE [13],UME-R1 [30], InternVideo2 [55], Unite [18], and GVE [20].

For the training phase, we followed VLM2Vec-V2 [41] and constructed a comprehensive dataset from three key sources: video-language instruction data from LLaVA-Hound [70], visual document retrieval data from ViDoRe [16] and VisRAG [65], and image-based vision-language task data from MMEB-train [26]. Detailed training procedures, hyperparameters, and dataset construction are in the supplementary material. We evaluate on two comprehensive benchmarks:

MMEB-V2 (Massive Multimodal Embedding Benchmark) [41]: It is a comprehensive and robust benchmark consisting of 78 diverse tasks across three core visual modalities (image, video, and visual document). MMEB-V2 extends the original MMEB [26] by introducing five additional meta-tasks focused specifically on video and visual document understanding, bringing the total to nine metatasks. We adopted Hit@1 as the evaluation metric for image and video tasks, and Normalized Discounted Cumulative Gain (NDCG@5) [22] for visual documents.

UVRB (Universal Video Retrieval Benchmark) [20]: It is a suite of 16 datasets to identify capability gaps in video retrieval across tasks and domains. UVRB measures multi-dimensional generalization over textual, composite, and visual retrieval tasks, including across coarse-grained, fine-grained, and long-context scenarios. We report mean Average Precision (mAP) for all UVRB tasks.

Table 1: Comparison of performance between baselines and our method on MMEB-V2. CLS: classification, QA: question answering, RET: retrieval, GD: grounding, MRET: moment retrieval, VDR: ViDoRe, VR: VisRAG, OOD: out-of-domain. The highest and second-highest values are highlighted in bold and underline.

Image Video VisDoc

Model

All CLS QA RET GD Overall CLS QA RET MRET Overall VDRv1 VDRv2 VR OOD Overall

# of Datasets 10 10 12 4 36 5 5 5 3 18 10 4 6 4 24 78

Baseline Models ColPali-V1.3-3B [16] 40.3 11.5 48.1 40.3 34.9 26.7 37.8 21.6 25.5 28.2 83.6 52.0 81.1 43.1 71.0 44.4 GME-2B [72] 54.4 29.9 66.9 55.5 51.9 34.9 42.0 25.6 32.4 33.9 86.1 54.0 82.5 43.1 72.7 54.1 GME-7B [72] 57.7 34.7 71.2 59.3 56.0 37.4 50.4 28.4 38.2 38.6 89.4 55.6 85.0 44.4 75.2 57.8 LamRA-2-7B [39] 59.2 26.5 70.0 62.7 54.1 39.3 42.6 24.3 34.6 35.2 22.0 11.5 37.4 21.0 23.9 40.4 LamRA-2.5-7B [39] 51.7 34.1 66.9 56.7 52.4 32.9 42.6 23.2 37.6 33.7 56.3 33.3 58.2 40.1 50.2 47.4 VLM2Vec-2B [26] 58.7 49.3 65.0 72.9 59.7 33.4 30.5 20.6 33.0 29.0 49.8 13.5 51.8 33.5 41.6 47.0 VLM2Vec-7B [26] 62.7 56.9 69.4 82.2 65.5 39.1 30.0 29.0 40.6 34.0 56.9 9.4 59.1 38.1 46.4 52.3 VLM2Vec-V2-2B [41] 62.9 56.3 69.5 77.3 64.9 39.3 34.3 28.8 38.5 34.9 75.5 44.9 79.4 39.4 65.4 58.0 VLM2Vec-V2-7B [41] 65.7 61.5 70.0 85.2 68.1 45.9 33.9 27.6 39.3 36.4 78.8 52.6 82.7 42.1 69.3 61.2 CAFe-7B [63] 63.6 61.7 69.1 87.6 67.6 35.8 58.7 34.4 39.5 42.4 70.7 49.6 79.5 38.1 63.9 60.6 TTEs-2B [13] 67.9 66.6 70.2 84.1 70.1 47.3 49.1 34.4 33.2 32.1 77.5 53.2 83.2 41.1 68.8 63.1 UME-R1-2B [30] 64.8 62.8 67.6 77.2 66.6 44.3 51.2 32.9 39.7 42.2 72.4 46.2 79.2 37.2 63.9 60.1 UME-R1-7B [30] 67.1 69.2 71.9 84.9 71.3 48.6 60.7 38.2 39.3 47.5 75.7 50.5 83.7 37.6 67.1 64.5 Ours

Embed-RL-2B 62.8 67.9 68.6 90.4 69.2 57.0 55.9 45.1 49.4 52.1 79.9 52.0 84.6 65.7 74.1 66.8 Embed-RL-4B 63.7 70.5 71.3 91.4 70.1 57.6 58.4 45.1 49.5 53.0 80.2 53.4 84.9 67.1 74.7 68.1

##### 4.3 Main Results

Table 1 compares the comprehensive performance of our proposed Embed-RL models with various baseline approaches on the MMEBV2 benchmark. Our Embed-RL models consistently achieve superior performance directly compared to all baseline models. Specifically, Embed-RL-4B attains the best overall score of 68.1, outperforming the strong next top baseline UME-R1-7B by 3.6 points. Embed-RL-2B follows closely with an overall score of 66.8, also surpassing all baseline variants. Across different modalities, our models show clear and notable advantages. In the Image modality, Embed-RL-4B impressively achieves the best grounding (GD) performance of 91.4, with Embed-RL-2B ranking second at 90.4. For the Video modality, both Embed-RL-2B and Embed-RL-4B outperform all baselines in overall score, with Embed-RL-2B scoring 52.1 and Embed-RL-4B scoring 53.0, and they achieve a video retrieval (RET) score of 45.1. In the visual document modality, we observe significant improvements in out-of-domain (OOD) performance, where Embed-RL-4B reaches 67.1 and Embed-RL-2B 65.7, far exceeding prior baseline results. These results demonstrate the effectiveness of our proposed approach across diverse visual modalities and task types.

Table 2: Video retrieval performance on UVRB. Domain dimensions: Coarse-grained (CG), Finegrained (FG), Long-context (LC). The best and second-best results are marked in bold and underline.

Model CG FG LC InternVideo2-6B [55] 50.4 41.7 42.3 VLM2Vec-V2 [41] 49.8 50.2 76.2 GME-7B [72] 51.8 50.7 78.8 Unite-7B [27] 54.1 53.9 74.6 GVE-3B [20] 55.2 54.1 76.4 Embed-RL-2B 59.1 54.6 86.9 Embed-RL-4B 60.7 55.6 86.1

In the broader field of video retrieval, benefiting from the effectiveness of our T-CoT to accurately locate keywords or keyframes, our model exhibits significant advantages in Coarse-grained (CG), Fine-grained (FG), and Long-context (LC) retrieval tasks. As shown in Table 2, detailing video retrieval performance of different models on the UVRB dataset across the three domains, our Embed-RL models consistently excel: 4B tops CG at 60.7 and FG at 55.6, while 2B leads LC at 86.9, both outperforming all existing baselines by a clear margin.

Additionally, Figure 3 presents the visualization of our T-CoT on text, image and video; we crop bbox and keyframe to achieve multi-modal CoT input, and our T-CoT accurately locates retrieval needs to improve retrieval performance. More scores and visualizations are provided in the supplementary material.

##### 4.4 Ablation Study

To dissect the contribution of each component in our framework, we conduct ablation experiments on MMEB-V2, where Embed-RL-2B serves as the model.

Analysis of the reward model in the RL Stage.As shown in Table 3, removing the RL stage leads to an overall performance decline of 1.5 points, dropping the score from 66.8 to 65.3. This result verifies that RL fine-tuning is indispensable for embedding alignment. Omitting weighted negative sampling, a core module of our contrastive reward mechanism, brings a performance reduction of 0.3 points from 66.8 to 66.5. This outcome emphasizes the component’s key

|Text-Image → Text|
|---|

|Text → Video|
|---|

Query: The frames begins … various shades of purple flowers … A table with a matching floral piece …The visible text … "Angela Flower” … . (Total 200 words)

Query: Align the question with image for representation: What toy is held by more children?

|[Figure 109]<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

Target

Query Traceability CoT for Long Text

[Figure 110]

[Figure 111]

<thinking>The description outlines a visual sequence. {"text_keywords": ["purple flowers", "table with floral piece", "Angela Flower"]}</thinking> <rethink>The sequence involves hands arranging flowers, structural

[Figure 112]

[Figure 113]

Target: teddy bear

transformation, and final composition with brand text.</rethink> <answer>A video showing individuals arranging purple flowers on a curved metallic structure with visible text "Angela Flower."</answer>

Query Traceability CoT for Image

<thinking>The image shows three children on grass. The child on the left holds a small teddy bear, the child in the middle holds a small round object (not a toy), and

|Text-Video → Text|
|---|

|[Figure 114]|
|---|
|[Figure 115]|
|[Figure 116]|

- 1st
- 2nd

Query: What utensil is used to turn the red liquid into a foamy pink substance?

Target: An electric hand mixer is used to turn the red liquid into a foamy pink substance.

the child on the right holds a large teddy bear. Only two children are holding teddy

bears. {"bbox_2d": [[30, 370, 220, 550], [440, 220, 800, 480]]}</thinking> <rethink>The small teddy bear is held by one child, and the large teddy bear is held by another. Since only one child holds each, neither is held by more children than the other.</rethink> <answer>The teddy bear is held by more children.</answer>

[Figure 117]

Query Traceability CoT for Video

<thinking>The video shows a red liquid being transformed into a foamy pink substance. Frame 2 clearly depicts a mixer with beaters actively blending the liquid, creating the foam. This is the only frame showing the transformation process. {"key_frames": [2]}</thinking>

4th

|[Figure 118]| |
|---|---|
| | |

[Figure 119]

…

<rethink>Frame 2 is the only frame showing the utensil actively turning the red liquid into the foamy pink substance, which is a mixer with beaters.</rethink> <answer>A mixer with beaters is used to turn the red liquid into a foamy pink substance.</answer>

|[Figure 120]|
|---|

last

|[Figure 121]|
|---|

|[Figure 122]|
|---|

Cropped Image

[Figure 123]

Select Frame

for Embedder

for Embedder

- Fig. 3: Example visualization of our reasoning-driven embedding framework on multimodal retrieval tasks. The figure shows the evidential Traceability CoT reasoning process for video and visual document retrieval.

function in prioritizing hard negative examples to strengthen discriminative embedding learning.The process reward is formulated to reward logical reasoning steps and align query T-CoT with target T-CoT. It contributes 0.8 points to the overall performance, as its exclusion lowers the score from 66.8 to 66.0. This component shows the most notable influence on video tasks, where performance falls from 52.1 to 51.3. This trend reveals that video understanding strongly depends on step-by-step reasoning and further reflects the critical role of the process reward in T-CoT alignment.Additionally, the outcome reward is built to reward final predictions. It contributes 1.0 point to the overall performance, and its removal reduces the total score from 66.8 to 65.8. This reward ensures that the reasoning process remains consistent with the objective of the target task.

Impact of Reasoning Components on T-CoT. As shown in Table 4, removing the reasoning process while retaining the answer part leads to a performance decrease of 1.3 points from 66.8 to 65.5. Image grounding and video moment retrieval see notable drops, falling from 69.2 to 67.9 and 52.1 to 50.5, which highlights the importance of multimodal evidence tracking for fine-grained alignment tasks. Removing multimodal cues leads to a performance reduction of 1.0 points from 66.8 to 65.8, validating the necessity of extracting multimodal cues through bounding boxes and keyframes to enhance alignment between multimodal representations and retrieval objectives. Most critically, using only raw input without T-CoT causes a catastrophic decrease in overall performance of 6.6 points, from 66.8 to 60.2. The greatest impact appears on video tasks, where performance falls from 52.1 to 43.7. This dramatic decline demonstrates the necessity of high-quality evidential Traceability CoT for retrieval accuracy, as it

enables the model to decompose complex retrieval and understanding tasks into manageable steps and improve cross-modal embedding quality.

Table 3: Ablation Study on Reward Components in EG-RL stage.

Table 4: Ablation Study on Reasoning Components in T-CoT.

Model Image Video VisDoc All

Embed-RL-2B 69.2 52.1 74.1 66.8

w/o EG-RL 68.0 50.1 72.7 65.3 w/o weighted negative 68.9 51.7 73.9 66.5

w/o process reward 68.3 51.3 73.5 66.0 w/o outcome reward 68.1 51.2 73.1 65.8

Model Image Video VisDoc All

Embed-RL-2B 69.2 52.1 74.1 66.8 w/o reasoning 67.9 50.5 73.1 65.5

w/o multimodal cues 68.1 51.4 73.3 65.8 w/ raw input 60.4 43.7 72.4 60.2

##### Ablation Study on Model’s Discriminative Ability for Candidates.

We define the top-ranked candidates with the highest similarity (excluding positive samples) as highly similar candidate samples. On this basis, we study how optimizing the reasoner with EG-RL improves the model’s ability to distinguish between similar candidates. Specifically, we calculate the difference between the similarity of the query to the most similar candidate and that to the second-most similar candidate on each dataset, both before and after RL. This difference measures whether the model assigns a significantly higher similarity to positive samples than to other highly similar candidates. As shown in Figure 4, we observe that on different datasets across three modalities, the radar chart obtained after RL prominently encloses the one obtained before RL. This indicates that the computed similarity difference becomes larger after RL, widening the gap between the query’s similarity to the top-ranked candidate and the second-ranked candidate. It demonstrates that the model’s ability to discriminate between similar candidates is effectively enhanced. Meanwhile, the bar chart shows that the model achieves consistent overall improvement on three-modality datasets. This verifies that optimizing T-CoT with RL strengthens the model’s general ability to distinguish between different candidates.

Image VisDoc Video

Δs Before RL Δs After RL

ViDoRe tabfquad

RefCOCO

ActivityNetQA

Category Level

0.20

0.13

0.14

ViDoSeek page

VisRAG ChartQA

QV Highlight

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

0.10

ImageNet-R

MSCOCO

MVBench

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

0.0880.090

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

0.11

0.20

0.08

0.09

0.10

0.08

0.08

0.068

0.062

0.06

0.054

0.052

0.11

0.15

0.10

0.11

0.03

0.05

0.04

ChartQA

WebQA

VisRAG InfoVQA

VisRAG ArxivQA

UCF101

Smth SmthV2

0.02

0.12

0.11

0.05 0.13

0.03 0.06

Visual7W Pointing

Infographics VQA

VisRAG PlotQA

VisDoRe ArxivQA

Charades STA

Video MME

0.00

Image VisDoc Video

###### Fig. 4: Similarity difference ∆s = sim(query, top1) − sim(query, top2) before and after EG-RL. Here, sim(·, ·) denotes cosine similarity of normalized embeddings, top1 is the most similar positive candidate and top2 the second-most similar. This metric quantifies the model’s discriminative ability over similar candidates on multimodal datasets.

Ablation on traceable evidence count and retrieval metrics.We also systematically analyze the relationship between the number of traceable evidence pieces and core retrieval metrics across all datasets before and after reinforcement learning.For images and visual document data, we count the change in the number of bounding boxes.For video data, we similarly count the change in the number of keyframes.We observe that after reinforcement learning, the T-CoT generated by the Reasoner tends to produce more bounding boxes.For the video modality, the model tends to focus on fewer keyframes.The retrieval metrics yield consistent and significant improvements, with the curves after reinforcement learning lying entirely above those before.For the image modality, the model captures more visual evidence to boost reasoning accuracy and recall.For the video modality, it concentrates on critical frames and conducts precise keyframe extraction and temporal localization to identify key content.These changes are particularly pronounced on complex samples involving multi-object localization and multi-person relationship reasoning.

Video

VisDoc

###### Image

Count Before RL

0.80

14k 12k 10k

0.90

- 0
- 1k
- 2k
- 3k
- 4k
- 5k
- 6k

0.90

- 0

2k

4k

6k

8k

- 1k

0.75

KeyframesCount

0.80

0.80

0.70

CountBbox

CountBbox

Count After RL

NDCG@5

hit@1

hit@1

8k 6k 4k 2k

0.70

0.65

0.75

0.60

0.60

Metric Before RL

0.60

0.55

0.50

0.50

0.50

0

Metric After RL

N24News OVEN MSCOCO_t2i VisDial MSCOCO RefCOCO

ViDoRe biomedical

VisRAG PlotQA

VisRAG SlideVQA

MVBench

Moment EgoSchema HMDB51 UCF101 Seeker

- Fig. 5: Relationship between traceable evidence counts and retrieval metrics across datasets. Hit@1 is employed for Image and Video; NDCG@5 is used for VisDoc. Bounding box counts are shown for Image and VisDoc, while keyframe counts for Video.

#### 5 Conclusion

This work addresses key limitations of generative universal multimodal embedding (UME) methods: chain-of-thought (CoT) remains text-only, resulting in poor retrieval relevance, while joint optimization of generative and embedding objectives gives rise to gradient conflicts that impede cross-modal matching. We propose Embed-RL, a reasoning-driven UME model built upon Embedder-Guided RL (EG-RL), which serves as a decoupled reinforcement learning framework that integrates multimodal evidential Traceability CoT (T-CoT) and a retrieval-oriented dual-reward mechanism to enable precise reasoning-embedding alignment. Extensive experiments on the MMEB-V2 and UVRB benchmarks demonstrate that Embed-RL outperforms state-of-the-art counterparts within computational constraints, achieving significant improvements in cross-modal retrieval and out-of-domain generalization tasks. This work shows that targeted reasoning optimization can substantially enhance multimodal embeddings, providing an efficient solution for reasoning-driven UME and valuable insights into the integration of generative reasoning with multimodal representation learning.

Text-Image → Text

###### Query:

Represent the given image with the following question: What fruit is to the right of the empty crate?

[Figure 148]

Query T-CoT Before RL: <thinking>The image shows a market stall with multiple bunches of bananas and a crate

on the left. To the right of the empty crate, there is a pile of red apples. The apples

are clearly visible and located at coordinates [560, 480, 800, 580]. {\"bbox_2d\":

| |
|---|

[560, 480, 800, 580]}</thinking>\n<rethink>The empty crate is on the left side of the

image, and the red apples are positioned directly to its right, confirming the fruit in

question.</rethink>\n<answer>The fruit to the right of the empty crate is apples.</answer>

[Figure 149]

Ground to incorrect information.

[Figure 150]

Similarity: 0.3613

[Figure 151]

###### Query T-CoT After RL:

<thinking>The image shows a market stall with multiple bunches of bananas and a pile of

red apples. An empty wooden crate is visible on the lower left. To the right of this empty crate, there are more bananas displayed on a table. {\"bbox_2d\": [0, 760, 200,

998], \"bbox_2d\": [200, 720, 500, 900]}</thinking>\n<rethink>The empty crate is

located on the lower left, and the bananas to its right are clearly visible on the table,

confirming the fruit is bananas.</rethink>\n<answer>The fruit to the right of the

empty crate is bananas.</answer>

| | |
|---|---|
| | |

[Figure 152]

Ground to accurate information.

[Figure 153]

Similarity: 0.4883

###### Target:

The fruit is a banana. Target T-CoT:

<thinking>The task requires identifying key concepts in the positive text output. The

text \"The fruit is a banana.\" explicitly identifies a fruit and its name. Key concepts

include \"fruit\" and \"banana\" as the subject and object. {\"text_keywords\": [\"fruit\", \"banana\"]}</thinking>\n<rethink>The keywords confirm the output

identifies a specific type of fruit, aligning with the expected content of the

task.</rethink>\n<answer>The text output identifies banana as a fruit.</answer>

###### Fig. 6: Example 1 of T-CoT Before and After EG-RL.

Text → Image

###### Query:

What insights can be drawn about the effects of scaffold pore size on tissue

regeneration? Query T-CoT:

<thinking>The question asks about insights from scaffold pore size on tissue regeneration, implying a scientific context. Key visual concepts include “scaffold,” “pore size,” and “tissue regeneration.” These terms suggest an image showing a porous

biomaterial structure with cells or tissue growing through or around it.

{\"text_keywords\": [\"scaffold\", \"pore size\", \"tissue regeneration\", \"biomaterial\", \"cells\", \"porous structure\"]}</thinking>\n<rethink>The keywords

indicate we need an image depicting a scaffold with visible pores and evidence of

tissue or cell growth, illustrating the relationship between pore size and regeneration.</rethink>\n<answer>An image of a porous scaffold with visible pores and growing tissue or cells, illustrating the relationship between scaffold pore size and

tissue regeneration.</answer>

###### Target T-CoT Before RL:

[Figure 154]

| |
|---|

<thinking>The task requires analyzing a biomedical slide about \"Scaffolds on peripheral nerves.\" The image contains key text elements: a title, a finding about

conduction velocity, and a citation. The most important visual element is the title, as it

defines the slide’s subject. The bounding box should cover the title text. {\"bbox_2d\": [50, 50, 500, 120]}</thinking>\n<rethink>The title \"Scaffolds on

peripheral nerves\" is the central and most informative element, as it directly

corresponds to the slide’s topic. The other text provides supporting context but is secondary to the main heading.</rethink>\n<answer>The image displays a slide titled \"Scaffolds on peripheral nerves,\" indicating a biomedical topic on nerve regeneration

[Figure 155]

and scaffold interactions.</answer>

Similarity: 0.4961

[Figure 156]

Missed grounding.

Target T-CoT After RL: <thinking>The task is to analyze a biomedical slide about scaffolds on peripheral nerves.

[Figure 157]

| |
|---|

The key visual elements are the red title \"Scaffolds on peripheral nerves\" and the

| |
|---|

text block on the right stating \"Conduction velocity of regenerated nerve became normal at scaffold pore diameter about 5 μm\". These are the most important parts of the slide. I will extract bounding boxes for both text blocks. {\"bbox_2d\": [[40, 50, 490, 120], [620, 70, 880, 210]]}</thinking>\n<rethink>The bounding boxes correctly identify the two key textual elements: the title and the critical finding about nerve conduction velocity. These are the essential components for understanding the slide’s

message.</rethink>\n<answer>The image is a biomedical slide titled \"Scaffolds on

peripheral nerves\" with the key finding that \"Conduction velocity of regenerated nerve became normal at scaffold pore diameter about 5 μm\".</answer>

[Figure 158]

Similarity: 0.5234

[Figure 159]

Complete grounding.

Fig. 7: Example 2 of T-CoT Before and After EG-RL.

[Figure 160]

Text → Video

###### Query:

A man in an outside field is demonstrating the art of throwing a javelin.

Query T-CoT: <thinking>The description specifies a man in an outdoor field performing a javelin

[Figure 161]

throw. Key visual elements include a person (man), an outdoor setting (field), and the

action of throwing a javelin. No additional context like weather, clothing, or audience is

provided, so focus remains on core visual components. {\"text_keywords\": [\"man\",

\"outside field\", \"throwing\", \"javelin\"]}</thinking>\n<rethink>The task demands a

video capturing a javelin throw by a man in an open field — the core visual concepts are the athlete, the outdoor environment, and the action of throwing a

[Figure 162]

javelin.</rethink>\n<answer>A video showing a man in an outdoor field demonstrating

the act of throwing a javelin.</answer>

###### Target T-CoT Before RL:

<thinking>The video frames depict a tennis player in action on a red clay court, with

visible green grass and trees in the background. The player is shown in various stages

of a swing, including frames where they are mid-motion (Frame 2, 4, 6) and preparing

[Figure 163]

to hit the ball (Frame 1, 3, 5, 7, 8). These frames capture the dynamic movement and

setting. {\"key_frames\": [1, 3, 5, 7]}</thinking>\n<rethink>The key frames (1, 3, 5, 7) capture the player's motion across different phases of a tennis swing, set against the

court and background environment.</rethink>\n<answer>The video shows a tennis

player on a red clay court, captured mid-swing in various frames, with green grass and

trees visible in the background.</answer>

[Figure 164]

Similarity: 0.3164

[Figure 165]

[Figure 166]

Misguided and query-irrelevant.

[Figure 167]

###### Target T-CoT After RL :

<thinking>The frames show a person performing athletic movements on a red track

adjacent to a green field, with trees and buildings visible in the background. The person is seen in various poses, including swinging a long object (likely a javelin) and

running. Frame 3 clearly shows the athlete in mid-throw motion, and Frame 6 shows the

follow-through. These frames capture the key actions and setting. {\"key_frames\": [3,

[Figure 168]

6]}</thinking>\n<rethink>The key frames (3 and 6) capture the athlete in the critical

moments of the javelin throw motion and follow-through, along with the context of the

track and surrounding environment.</rethink>\n<answer>The video shows an athlete performing a javelin throw on a red track next to a green field, with trees and

buildings in the background. The motion is captured in mid-throw and follow-through

phases.</answer>

[Figure 169]

Similarity: 0.5742

[Figure 170]

[Figure 171]

Query-aligned and key-action focused.

###### Fig. 8: Example 3 of T-CoT Before and After EG-RL.

#### References

- 1. Anne Hendricks, L., Wang, O., Shechtman, E., Sivic, J., Darrell, T., Russell, B.: Localizing moments in video with natural language. In: Proceedings of the IEEE international conference on computer vision. pp. 5803–5812 (2017)
- 2. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li, Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, D., Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun, Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, T., Xu, Y., Xu, H., Xu, J., Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, Y., Zhu, K.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631 (2025)
- 3. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al.: Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923

(2025)

- 4. Bolya, D., Huang, P.Y., Sun, P., Cho, J.H., Madotto, A., Wei, C., Ma, T., Zhi, J., Rajasegaran, J., Rasheed, H., et al.: Perception encoder: The best visual embeddings are not at the output of the network. arXiv preprint arXiv:2504.13181 (2025)
- 5. Cai, Q., Liang, H., Dong, H., Qiang, M., An, R., Han, Z., Zhu, Z., Cui, B., Zhang, W.: Lovr: A benchmark for long video retrieval in multimodal contexts. arXiv preprint arXiv:2505.13928 (2025)
- 6. Cao, M., Zhao, H., Zhang, C., Chang, X., Reid, I., Liang, X.: Ground-r1: Incentivizing grounded visual reasoning via reinforcement learning. arXiv preprint arXiv:2505.20272 (2025)
- 7. Chai, W., Song, E., Du, Y., Meng, C., Madhavan, V., Bar-Tal, O., Hwang, J.N., Xie, S., Manning, C.D.: Auroracap: Efficient, performant video detailed captioning and a new benchmark. arXiv preprint arXiv:2410.03051 (2024)
- 8. Chen, H., Liu, H., Luo, Y., Wang, L., Yang, N., Wei, F., Dou, Z.: Moca: Modalityaware continual pre-training makes better bidirectional multimodal embeddings. arXiv preprint arXiv:2506.23115 (2025)
- 9. Chen, H., Wang, L., Yang, N., Zhu, Y., Zhao, Z., Wei, F., Dou, Z.: mme5: Improving multimodal multilingual embeddings via high-quality synthetic data. arXiv preprint arXiv:2502.08468 (2025)
- 10. Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G., Gordon, C., Schuhmann, C., Schmidt, L., Jitsev, J.: Reproducible scaling laws for contrastive language-image learning. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 2818–2829 (2023)
- 11. Chu, X., Chen, X., Wang, G., Tan, Z., Huang, K., Lv, W., Mo, T., Li, W.: Qwen look again: Guiding vision-language reasoning models to re-attention visual information. arXiv preprint arXiv:2505.23558 (2025)
- 12. Chung, J., Kim, J., Kim, S., Lee, J., Kim, M.S., Yu, Y.: Don’t look only once: Towards multimodal interactive reasoning with selective visual revisitation. arXiv preprint arXiv:2505.18842 (2025)
- 13. Cui, X., Cheng, J., Chen, H.y., Shukla, S.N., Awasthi, A., Pan, X., Ahuja, C., Mishra, S.K., Yang, Y., Xiao, J., et al.: Think then embed: Generative context improves multimodal embedding. arXiv preprint arXiv:2510.05014 (2025)
- 14. Duan, C., Fang, R., Wang, Y., Wang, K., Huang, L., Zeng, X., Li, H., Liu, X.: Gotr1: Unleashing reasoning capability of mllm for visual generation with reinforcement learning. arXiv preprint arXiv:2505.17022 (2025)

- 15. Fan, Y., He, X., Yang, D., Zheng, K., Kuo, C.C., Zheng, Y., Narayanaraju, S.J., Guan, X., Wang, X.E.: Grit: Teaching mllms to think with images. arXiv preprint arXiv:2505.15879 (2025)
- 16. Faysse, M., Sibille, H., Wu, T., Omrani, B., Viaud, G., Hudelot, C., Colombo, P.: Colpali: Efficient document retrieval with vision language models. arXiv preprint arXiv:2407.01449 (2024)
- 17. Geng, X., Xia, P., Zhang, Z., Wang, X., Wang, Q., Ding, R., Wang, C., Wu, J., Zhao, Y., Li, K., et al.: Webwatcher: Breaking new frontier of vision-language deep research agent. arXiv preprint arXiv:2508.05748 (2025)
- 18. Gu, T., Yang, K., Feng, Z., Wang, X., Zhang, Y., Long, D., Chen, Y., Cai, W., Deng, J.: Breaking the modality barrier: Universal embedding learning with multimodal llms. arXiv preprint arXiv:2504.17432 (2025)
- 19. Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al.: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948 (2025)
- 20. Guo, Z., Li, M., Zhang, Y., Long, D., Xie, P., Chu, X.: Towards universal video retrieval: Generalizing video embedding via synthesized multimodal pyramid curriculum. arXiv preprint arXiv:2510.27571 (2025)
- 21. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. ICLR 1(2), 3 (2022)
- 22. Järvelin, K., Kekäläinen, J.: Cumulated gain-based evaluation of ir techniques. ACM Transactions on Information Systems (TOIS) 20(4), 422–446 (2002)
- 23. Ji, D., Zhu, L., Gao, S., Xu, P., Lu, H., Ye, J., Zhao, F.: Tree-of-table: Unleashing the power of llms for enhanced large-scale table understanding. arXiv preprint arXiv:2411.08516 (2024)
- 24. Jia, C., Yang, Y., Xia, Y., Chen, Y.T., Parekh, Z., Pham, H., Le, Q., Sung, Y.H., Li, Z., Duerig, T.: Scaling up visual and vision-language representation learning with noisy text supervision. In: International conference on machine learning. pp. 4904–4916. PMLR (2021)
- 25. Jiang, T., Song, M., Zhang, Z., Huang, H., Deng, W., Sun, F., Zhang, Q., Wang, D., Zhuang, F.: E5-v: Universal embeddings with multimodal large language models. arXiv preprint arXiv:2407.12580 (2024)
- 26. Jiang, Z., Meng, R., Yang, X., Yavuz, S., Zhou, Y., Chen, W.: Vlm2vec: Training vision-language models for massive multimodal embedding tasks. arXiv preprint arXiv:2410.05160 (2024)
- 27. Kong, F., Zhang, J., Liu, Y., Zhang, H., Feng, S., Yang, X., Wang, D., Tian, Y., Zhang, F., Zhou, G., et al.: Modality curation: Building universal embeddings for advanced multimodal information retrieval. arXiv preprint arXiv:2505.19650 (2025)
- 28. Lai, X., Tian, Z., Chen, Y., Li, Y., Yuan, Y., Liu, S., Jia, J.: Lisa: Reasoning segmentation via large language model. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9579–9589 (2024)
- 29. Lan, Z., Niu, L., Meng, F., Zhou, J., Su, J.: Llave: Large language and vision embedding models with hardness-weighted contrastive learning. arXiv preprint arXiv:2503.04812 (2025)
- 30. Lan, Z., Niu, L., Meng, F., Zhou, J., Su, J.: Ume-r1: Exploring reasoning-driven generative multimodal embeddings. arXiv preprint arXiv:2511.00405 (2025)
- 31. Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Zhang, P., Li, Y., Liu, Z., et al.: Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326 (2024)

- 32. Li, F., Zhang, R., Zhang, H., Zhang, Y., Li, B., Li, W., Ma, Z., Li, C.: Llava-nextinterleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895 (2024)
- 33. Li, G., Xu, J., Zhao, Y., Peng, Y.: Dyfo: A training-free dynamic focus visual search for enhancing lmms in fine-grained visual understanding. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 9098–9108 (2025)
- 34. Li, J., Li, D., Savarese, S., Hoi, S.: Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In: International conference on machine learning. pp. 19730–19742. PMLR (2023)
- 35. Liao, Z., Xie, Q., Zhang, Y., Kong, Z., Lu, H., Yang, Z., Deng, Z.: Improved visual-spatial reasoning via r1-zero-like training. arXiv preprint arXiv:2504.00883

(2025)

- 36. Lin, S.C., Lee, C., Shoeybi, M., Lin, J., Catanzaro, B., Ping, W.: Mm-embed: Universal multimodal retrieval with multimodal llms. arXiv preprint arXiv:2411.02571

(2024)

- 37. Lin, Z., Cen, S., Jiang, D., Karhade, J., Wang, H., Mitra, C., Ling, T., Huang, Y., Liu, S., Chen, M., et al.: Towards understanding camera motions in any video. arXiv preprint arXiv:2504.15376 (2025)
- 38. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. Advances in neural information processing systems 36, 34892–34916 (2023)
- 39. Liu, Y., Zhang, Y., Cai, J., Jiang, X., Hu, Y., Yao, J., Wang, Y., Xie, W.: Lamra: Large multimodal model as your advanced retrieval assistant. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 4015–4025 (2025)
- 40. Luo, H., Ji, L., Zhong, M., Chen, Y., Lei, W., Duan, N., Li, T.: Clip4clip: An empirical study of clip for end to end video clip retrieval and captioning. Neurocomputing 508, 293–304 (2022)
- 41. Meng, R., Jiang, Z., Liu, Y., Su, M., Yang, X., Fu, Y., Qin, C., Chen, Z., Xu, R., Xiong, C., et al.: Vlm2vec-v2: Advancing multimodal embedding for videos, images, and visual documents. arXiv preprint arXiv:2507.04590 (2025)
- 42. Oord, A.v.d., Li, Y., Vinyals, O.: Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748 (2018)
- 43. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PmLR (2021)
- 44. Su, Z., Li, L., Song, M., Hao, Y., Yang, Z., Zhang, J., Chen, G., Gu, J., Li, J., Qu, X., et al.: Openthinkimg: Learning to think with images via visual tool reinforcement learning. arXiv preprint arXiv:2505.08617 (2025)
- 45. Su, Z., Xia, P., Guo, H., Liu, Z., Ma, Y., Qu, X., Liu, J., Li, Y., Zeng, K., Yang, Z., et al.: Thinking with images for multimodal reasoning: Foundations, methods, and future frontiers. arXiv preprint arXiv:2506.23918 (2025)
- 46. Thirukovalluru, R., Meng, R., Liu, Y., Su, M., Nie, P., Yavuz, S., Zhou, Y., Chen, W., Dhingra, B., et al.: Breaking the batch barrier (b3) of contrastive learning via smart batch mining. arXiv preprint arXiv:2505.11293 (2025)
- 47. Wang, H., Li, X., Huang, Z., Wang, A., Wang, J., Zhang, T., Zheng, J., Bai, S., Kang, Z., Feng, J., et al.: Traceable evidence enhanced visual grounded reasoning: Evaluation and methodology. arXiv preprint arXiv:2507.07999 (2025)
- 48. Wang, J., Wang, C., Huang, K., Huang, J., Jin, L.: Videoclip-xl: Advancing long description understanding for video clip models. arXiv preprint arXiv:2410.00741

(2024)

- 49. Wang, J., Yuan, L., Zhang, Y., Sun, H.: Tarsier: Recipes for training and evaluating large video description models. arXiv preprint arXiv:2407.00634 (2024)
- 50. Wang, P., Ling, H.: Svqa-r1: Reinforcing spatial reasoning in mllms via viewconsistent reward optimization. arXiv preprint arXiv:2506.01371 (2025)
- 51. Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., et al.: Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191 (2024)
- 52. Wang, Q., Liu, J., Liang, J., Jiang, Y., Zhang, Y., Chen, J., Zheng, Y., Wang, X., Wan, P., Yue, X., et al.: Vr-thinker: Boosting video reward models through thinking-with-image reasoning. arXiv preprint arXiv:2510.10518 (2025)
- 53. Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., Zhou, D.: Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171 (2022)
- 54. Wang, Y., He, Y., Li, Y., Li, K., Yu, J., Ma, X., Li, X., Chen, G., Chen, X., Wang, Y., et al.: Internvid: A large-scale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942 (2023)
- 55. Wang, Y., Li, K., Li, X., Yu, J., He, Y., Chen, G., Pei, B., Zheng, R., Wang, Z., Shi, Y., et al.: Internvideo2: Scaling foundation models for multimodal video understanding. In: European Conference on Computer Vision. pp. 396–416. Springer

(2024)

- 56. Wang, Y., Liu, W., Niu, J., Zhang, H., Tang, Y.: Vg-refiner: Towards tool-refined referring grounded reasoning via agentic reinforcement learning. arXiv preprint arXiv:2512.06373 (2025)
- 57. Wei, C., Chen, Y., Chen, H., Hu, H., Zhang, G., Fu, J., Ritter, A., Chen, W.: Uniir: Training and benchmarking universal multimodal information retrievers. In: European Conference on Computer Vision. pp. 387–404. Springer (2024)
- 58. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q.V., Zhou, D., et al.: Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems 35, 24824–24837 (2022)
- 59. Wu, M., Yang, J., Jiang, J., Li, M., Yan, K., Yu, H., Zhang, M., Zhai, C., Nahrstedt, K.: Vtool-r1: Vlms learn to think with images via reinforcement learning on multimodal tool use. arXiv preprint arXiv:2505.19255 (2025)
- 60. Xu, G., Jin, P., Wu, Z., Li, H., Song, Y., Sun, L., Yuan, L.: Llava-cot: Let vision language models reason step-by-step. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 2087–2098 (2025)
- 61. Xu, J., Mei, T., Yao, T., Rui, Y.: Msr-vtt: A large video description dataset for bridging video and language. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 5288–5296 (2016)
- 62. Xu, Y., Li, X., Yang, Y., Meng, D., Huang, R., Wang, L.: Carebench: A fine-grained benchmark for video captioning and retrieval (2025), https://arxiv.org/abs/ 2501.00513
- 63. Yu, H., Zhao, Z., Yan, S., Korycki, L., Wang, J., He, B., Liu, J., Zhang, L., Fan, X., Yu, H.: Cafe: Unifying representation and generation with contrastive-autoregressive finetuning. arXiv preprint arXiv:2503.19900 (2025)
- 64. Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, L., et al.: Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476 (2025)
- 65. Yu, S., Tang, C., Xu, B., Cui, J., Ran, J., Yan, Y., Liu, Z., Wang, S., Han, X., Liu, Z., et al.: Visrag: Vision-based retrieval-augmented generation on multi-modality documents. arXiv preprint arXiv:2410.10594 (2024)

- 66. Yuan, H., Ni, J., Liu, Z., Wang, Y., Zhou, J., Liang, Z., Zhao, B., Cao, Z., Dou, Z., Wen, J.R.: Momentseeker: A task-oriented benchmark for long-video moment retrieval. arXiv preprint arXiv:2502.12558 (2025)
- 67. Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L.: Sigmoid loss for language image pretraining. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 11975–11986 (2023)
- 68. Zhang, H., Gu, X., Li, J., Ma, C., Bai, S., Zhang, C., Zhang, B., Zhou, Z., He, D., Tang, Y.: Thinking with videos: Multimodal tool-augmented reinforcement learning for long video reasoning. arXiv preprint arXiv:2508.04416 (2025)
- 69. Zhang, K., Luan, Y., Hu, H., Lee, K., Qiao, S., Chen, W., Su, Y., Chang, M.W.: Magiclens: Self-supervised image retrieval with open-ended instructions. arXiv preprint arXiv:2403.19651 (2024)
- 70. Zhang, R., Gui, L., Sun, Z., Feng, Y., Xu, K., Zhang, Y., Fu, D., Li, C., Hauptmann, A.G., Bisk, Y., et al.: Direct preference optimization of video large multimodal models from language model reward. In: Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). pp. 694–717 (2025)
- 71. Zhang, X., Zhang, Y., Xie, W., Li, M., Dai, Z., Long, D., Xie, P., Zhang, M., Li, W., Zhang, M.: Gme: Improving universal multimodal retrieval by multimodal llms. arXiv preprint arXiv:2412.16855 (2024)
- 72. Zhang, X., Zhang, Y., Xie, W., Li, M., Dai, Z., Long, D., Xie, P., Zhang, M., Li, W., Zhang, M.: Bridging modalities: Improving universal multimodal retrieval by multimodal large language models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 9274–9285 (2025)
- 73. Zhang, X., Du, C., Pang, T., Liu, Q., Gao, W., Lin, M.: Chain of preference optimization: Improving chain-of-thought reasoning in llms. Advances in Neural Information Processing Systems 37, 333–356 (2024)
- 74. Zheng, C., Liu, S., Li, M., Chen, X.H., Yu, B., Gao, C., Dang, K., Liu, Y., Men, R., Yang, A., et al.: Group sequence policy optimization. arXiv preprint arXiv:2507.18071 (2025)
- 75. Zheng, Z., Yang, M., Hong, J., Zhao, C., Xu, G., Yang, L., Shen, C., Yu, X.: Deepeyes: Incentivizing" thinking with images" via reinforcement learning. arXiv preprint arXiv:2505.14362 (2025)
- 76. Zhou, J., Liu, Z., Liu, Z., Xiao, S., Wang, Y., Zhao, B., Zhang, C.J., Lian, D., Xiong, Y.: Megapairs: Massive data synthesis for universal multimodal retrieval. arXiv preprint arXiv:2412.14475 (2024)
- 77. Zhou, J., Xiong, Y., Liu, Z., Liu, Z., Xiao, S., Wang, Y., Zhao, B., Zhang, C.J., Lian, D.: Megapairs: Massive data synthesis for universal multimodal retrieval. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 19076–19095 (2025)
- 78. Zhu, B., Lin, B., Ning, M., Yan, Y., Cui, J., HongFa, W., Pang, Y., Jiang, W., Zhang, J., Li, Z., Zhang, C.W., Li, Z., Liu, W., Yuan, L.: Languagebind: Extending video-language pretraining to n-modality by language-based semantic alignment. In: The Twelfth International Conference on Learning Representations (2024)
- 79. Zhu, L., Chen, Q., Shen, X., Cun, X.: Vau-r1: Advancing video anomaly understanding via reinforcement fine-tuning. arXiv preprint arXiv:2505.23504 (2025)

## Embed-RL: Reinforcement Learning for Reasoning-Driven Multimodal Embeddings

### Supplementary Material

#### A Additional details

In this supplementary material, we elaborate on further insights, provide detailed derivations, and include additional qualitative results to enhance the comprehensiveness of our work.

#### B Training Details

##### B.1 Contrastive Learning

We trained the Qwen3-VL-2B-Instruct [2] and Qwen3-VL-4B-Instruct [2] models using the DeepSpeed Zero2 optimization framework. Key training hyperparameters were summarized as follows: The training process was conducted for 2 epochs with a batch size of 512 for Qwen3-VL-2B-Instruct and 256 for Qwen3VL-4B-Instruct. Following the sub-batch training scheme in VLM2Vec [26], we adopted a sub-batch training strategy that ensured samples in each sub-batch are drawn from the same dataset, where the sub-batch size was set to 256 for Qwen3-VL-2B-Instruct and 128 for Qwen3-VL-4B-Instruct. We set the initial learning rate to 1e-4, using a cosine learning rate scheduler with 10 warm-up steps and a weight decay of 0.01. We employed Low-Rank Adaptation (LoRA) [21] for fine-tuning: the rank r and scaling factor α were set to 64 and 128 for Qwen3VL-2B-Instruct, and 96 and 192 for Qwen3-VL-4B-Instruct, respectively. For comparison, UME-R1 [30] was trained with a batch size of 1024 and TTE [13] was trained with 8192, indicating that our training scale was significantly smaller than both methods. Notably, our experimental results could be further scaled up with more abundant computational resources.

##### B.2 Embedder-Guided Reinforcement Learning

In the reinforcement learning stage, we trained the Qwen3-VL-8B-Instruct [2] Reasoner with the GRPO [19]. We adopted in-batch negative contrastive rewards for optimization, with GRPO hyperparameters set as group size G = 8, clipping parameter ε = 0.2, and KL-divergence coefficient β = 0.01. We set batch size to 256, learning rate to 3e-6, and trained the model for one epoch. We also restricted each step to samples from the same dataset to avoid overly simple negative samples affecting optimization.

For the embedder-guided outcome reward Routcome, two core hyperparameters balance retrieval accuracy and similarity margin calculation: the top-k parameter

for retrieval accuracy Acck and the temperature parameter τ for softmax-weighted negative sampling.

,t+i ) to 8. Specifically, Acc8(eq

Top-k Retrieval Accuracy k: We set the parameter k in Acck(eq

i

,t+i ) checks whether all 8 T-CoT rollouts of a query-target pair rank in the top-8 by cosine similarity to eq

i

, which captures the embedding alignment consistency critical for stable GRPO reward optimization.

i

Temperature Parameter (τ): The temperature τ scales the softmax weights for hard negative sampling in the similarity margin calculation, which is defined as:

sim(eqi,et−

)

τ · sim(eq

j

j̸=i exp

,et−

)

i

j

. (7)

Eτ sim(eq

,et−

) =

sim(eqi,et−

i

) τ

j

j

j̸=i exp

We set τ = 0.5 for all experiments, a value chosen to emphasize hard negatives while avoiding overfitting to noisy negative samples.

For process reward, we employ an independent Qwen3-VL-8B-Instruct [2] as the pretrained VLM Discriminator D. It performs listwise comparison to align query T-CoT outputs with corresponding target outputs. The discriminator selects the candidate T-CoT that best matches the query. Selection correctness quantifies alignment quality and forms the process reward signal.

We set α = 0.05, β = 0.8, and γ = 0.2 for the total reward Rtotal to balance the contributions of format, process, and outcome rewards respectively.

##### B.3 Multimodel Vision Processing

For visual input processing, we set specific pixel constraints for images and videos to balance computational efficiency and feature fidelity:

For images: MIN_PIXELS = 128×32×32 and MAX_PIXELS = 768×32×32. For videos: VIDEO_MIN_PIXELS = 128×32×32, VIDEO_MAX_PIXELS

= 300 × 32 × 32, and VIDEO_TOTAL_PIXELS = 300 × 32 × 32 × 8.

Video frame sampling hyperparameters were fixed as FRAME_FACTOR = 2, FPS = 2.0, FPS_MIN_FRAMES = 8 and FPS_MAX_FRAMES = 8.

For multimodal cropping, we first convert the relative coordinates of bounding boxes, which are scaled to the range 0–1000, into the original image coordinates, then conduct cropping on the raw image. Keyframes corresponding to the sampled frames are re-extracted and concatenated between the </thinking>and <rethink>tokens.

#### C Detailed Dataset Construction

##### C.1 Data Sources and Initial Sampling Strategy

To enable effective multi-modal task training, we adopt the training data paradigm of VLM2Vec-V2 [41] and build a comprehensive dataset from three core sources: (1) video-language instruction data (LLaVA-Hound [70]), (2) visual document retrieval data (ViDoRe [16] and VisRAG [65]), and (3) image-based vision task data (MMEB-train [26]). We apply a stratified sampling strategy across data modalities to ensure balanced coverage:

- – Image-based datasets: Maximum 50,000 samples per dataset
- – Document-based datasets: Maximum 100,000 samples per dataset
- – Video-based datasets: Maximum 300,000 samples per dataset

Full sampling is used if the original dataset size is smaller than the above maximum. As shown in Table 5, we presented the exact number of samples selected for each dataset in our experiments.

Table 5: Statistics of Initial Sampling and CoT-guided Filtering.

Dataset Initial Samples Filtered Samples Retention Ratio Weight Modality

Image-based (MMEB-train) A-OKVQA 50,000 37,929 75.86% 0.26 Text-Image → Text

CIRR 50,000 35,085 70.17% 0.43 Text-Image → Text-Image

ChartQA 50,000 39,512 79.02% 0.35 Text-Image → Text DocVQA 50,000 47,401 94.80% 0.84 Text-Image → Text

HatefulMemes 25,500 16,572 64.99% 0.30 Text-Image → Text ImageNet-1K 50,000 44,409 88.82% 2.25 Text-Image → Text InfographicsVQA 50,000 40,746 81.49% 0.31 Text-Image → Text

MSCOCO 50,000 26,429 52.86% 3.78 Text-Image → Text-Image

MSCOCO-i2t 50,000 46,596 93.19% 2.58 Text-Image → Text MSCOCO-t2i 50,000 43,173 86.35% 2.32 Text → Text-Image

N24News 50,000 30,320 60.64% 1.65 Text-Image → Text NIGHTS 47,823 43,167 90.26% 0.23 Text-Image → Text-Image OK-VQA 27,027 19,900 73.63% 0.25 Text-Image → Text

SUN397 50,000 45,864 91.73% 0.22 Text-Image → Text VOC2007 23,532 20,454 86.92% 0.24 Text-Image → Text Visual7W 50,000 41,677 83.35% 1.68 Text-Image → Text

VisDial 50,000 34,652 69.30% 3.75 Text → Text-Image VisualNews-i2t 50,000 34,364 68.73% 2.91 Text-Image → Text VisualNews-t2i 50,000 28,684 57.37% 3.49 Text → Text-Image

WebQA 50,000 43,910 87.82% 0.23 Text → Text-Image Video-based (LLaVA-Hound) Caption Retrieval 300,000 283,721 94.57% 5.27 Video → Text

Video QA 300,000 273,906 91.30% 4.38 Video-Text → Text Video Retrieval 300,000 260,410 86.80% 5.76 Text → Video

Document-based ViDoRe 100,000 83,964 83.96% 5.0 Text-Image → Text VisRAG 100,000 60,266 60.27% 6.0 Text → Image

Image-based 1,123,882 865,074 76.97% - Image-centric Video-based 900,000 819,037 91.02% - Video-centric Document-based 200,000 144,230 72.12% - Document-centric

Total 2,223,882 1,828,341 82.21% - Multimodal

We exclude three classification datasets (HatefulMemes, N24News, VOC2007) from the first-stage contrastive learning. This stage only employs contrastive loss under a single-dataset sub-batch constraint. Their limited number of categories results in false negatives and noise with large sub-batches, which impairs representation learning. This explains the model’s weak performance on image classification, as it never encounters these samples during training.

##### C.2 CoT-guided Relevance Filtering

We generate chain-of-thought (CoT) annotations for queries and positive samples using Qwen3-VL-8B [2], and further conduct strict relevance filtering via a custom prompt with the same model to discard annotations that are irrelevant to or conflict with the query task. To mitigate noise in contrastive learning, only samples labeled "No" are retained. Table 5 lists the sample size, retention ratio, training weight, and modality of each dataset.

CoT Relevance and Conflict Judgment Prompt

Your task is to determine whether the content of ‘query_cot‘ and ‘pos_cot‘ are ∗∗obviously irrelevant∗∗

or ∗∗obviously conflicting∗∗ based on the given ‘qry‘ (task description). ∗∗Rules∗∗:

- 1. Only output a single word: "Yes" (if obviously irrelevant/conflicting) or "No" (if relevant and not conflicting)

- 2. "Obviously irrelevant": The content of query_cot and pos_cot have no logical connection to each other or to the qry task

- 3. "Obviously conflicting": The core conclusions/key information of query_cot and pos_cot are mutually contradictory

- 4. Only judge "obvious" cases − if the relevance is ambiguous, output "No"

##### C.3 RL Dataset Sampling

Using the high-quality CoT-filtered dataset, we apply equidistant sampling to construct the reinforcement learning (RL) training set, ensuring uniform distribution across challenging sub-datasets, as shown in Table 6.

- Table 6: Reinforcement Learning Dataset Construction and Sampling Strategy.

Sampled Dataset RL Samples Modality

A-OKVQA 1,000 Text-Image → Text llavahound video retrieval 2,000 Text → Video ViDoRe colpali train set 2,000 Text-Image → Text VisualNews-t2i 1,000 Text → Text-Image VisualNews-i2t 1,000 Text-Image → Text VisRAG-Ret-Train-In-domain-data 2,000 Text → Image CIRR 2,000 Text-Image → Text-Image ChartQA 1,000 Text-Image → Text OK-VQA 1,000 Text-Image → Text llavahound qa 2,000 Video-Text → Text llavahound caption retrieval 2,000 Video → Text Visual7W 1,000 Text-Image → Text N24News 1,000 Text-Image → Text

Total 19,000 Multimodal

- D Detailed Scores of MMEB-V2 We report the detailed metrics for our MMEB-V2 dataset [41], as shown in

- Table 7 and Table 8. The highest and second-highest values are highlighted in bold and underlined, respectively.

- Table 7: Detailed results of baselines and Embed-RL on full MMEB-v2 benchmark. Video and Visual Doc results are shown on the next table.

ColPali v1.3 [16] GME-7B [72] VLM2Vec-7B [26] VLM2Vec-V2-2B [41] CAFe-7B [63] UME-R1-2B [30] UME-R1-7B [30] Embed-RL-2B Embed-RL-4B

Avg - All (78 tasks) 44.4 57.8 52.3 58.0 60.6 60.1 64.5 66.8 68.1 Avg - Image (36 tasks, Hit@1) 34.9 56.0 65.5 64.9 67.6 66.6 71.3 69.2 71.2 Avg - Video (18 tasks, Hit@1) 28.2 38.4 33.7 34.6 42.4 42.2 47.5 52.1 53.0 Avg - Visdoc (24 tasks, NDCG@5) 71.0 75.2 46.4 65.4 63.9 63.9 67.1 74.1 74.7 I-CLS (10) 40.3 57.7 62.7 62.9 63.6 64.8 67.1 62.8 63.7 I-QA (10) 11.5 34.7 56.9 56.3 61.7 62.8 69.2 67.9 70.5 I-RET (12) 48.1 71.2 69.4 69.5 69.1 67.6 71.9 68.6 71.3 I-VG (4) 40.3 59.3 82.2 77.3 87.6 77.2 84.9 90.4 91.4 V-CLS (5) 26.7 37.4 39.1 39.3 35.8 44.3 48.6 57.0 57.6 V-QA (5) 37.8 50.4 30.0 34.3 58.7 51.0 60.7 55.9 58.4 V-RET (5) 21.6 28.4 29.0 28.8 34.4 32.9 38.2 45.1 45.1 V-MR (3) 25.5 37.0 38.9 36.8 39.5 39.7 39.3 49.4 49.5 VD-Vidore-V1 (10) 83.6 89.4 56.9 75.7 70.7 72.4 75.7 79.9 80.2 VD-Vidore-V2 (4) 52.0 55.6 9.4 45.1 49.6 46.2 50.5 52.0 53.4 VD-VisRAG (6) 81.1 85.0 59.1 79.6 79.5 79.2 83.7 84.6 84.9 VD-OOD (4) 43.1 44.4 38.1 39.6 38.1 37.2 37.6 65.7 67.1 ImageNet-1K 42.4 64.6 80.1 80.8 77.3 75.3 80.4 78.0 79.5 N24News 25.5 50.5 79.7 72.9 83.2 81.1 82.3 44.9 48.3 HatefulMemes 50.6 53.6 69.7 56.3 78.7 75.2 79.0 65.0 66.2 VOC2007 69.8 80.3 80.7 85.0 89.8 80.0 90.8 78.7 79.5 SUN397 56.1 69.5 77.4 71.0 79.9 79.4 80.3 75.4 79.2 Place365 27.5 39.1 37.4 35.9 45.0 42.6 46.8 43.9 43.1 ImageNet-A 14.9 41.2 58.1 47.4 55.2 50.4 53.9 59.2 58.1 ImageNet-R 64.6 83.9 73.9 89.3 88.0 88.7 90.1 88.5 88.2 ObjectNet 45.6 69.0 40.1 65.2 22.5 52.0 42.3 74.8 75.4 Country211 6.0 24.8 29.8 25.2 16.7 23.4 25.0 20.0 19.4 OK-VQA 9.4 33.2 56.8 51.5 67.3 62.4 71.7 61.4 67.3 A-OKVQA 6.6 21.0 47.3 43.6 63.8 51.1 58.7 54.7 59.3 DocVQA 11.3 41.4 89.7 90.1 79.2 92.2 93.8 92.4 94.3 InfographicsVQA 5.0 20.3 60.0 58.8 53.3 67.7 79.2 76.7 77.5 ChartQA 5.7 17.8 56.9 47.4 48.8 64.9 75.1 80.7 80.9 Visual7W 6.1 22.2 52.7 52.9 52.5 54.1 55.2 52.7 55.3 ScienceQA 16.3 28.0 38.5 38.2 65.4 42.7 53.7 57.3 61.6 VizWiz 27.6 39.0 39.9 43.3 43.8 46.8 51.6 54.5 56.2 GQA 8.3 76.9 55.1 64.9 65.7 67.3 69.3 64.9 68.5 TextVQA 18.8 46.8 71.6 72.2 76.8 78.6 83.5 83.8 84.3 VisDial 41.2 60.8 81.9 82.7 82.7 76.6 80.7 81.5 84.9 CIRR 8.2 54.9 51.1 57.5 60.4 53.7 55.3 47.6 61.2 VisualNews_t2i 50.1 79.7 80.5 74.5 69.5 71.7 76.8 71.9 73.7 VisualNews_i2t 47.6 83.6 81.2 78.2 79.4 74.2 82.0 73.6 73.9 MSCOCO_t2i 59.2 71.2 77.2 75.3 75.4 75.1 78.3 79.4 78.9 MSCOCO_i2t 49.9 57.7 73.9 71.4 73.1 68.9 71.4 75.3 76.3 NIGHTS 65.5 67.6 67.6 68.6 66.7 67.2 68.1 66.3 66.4 WebQA 53.8 91.4 88.3 90.6 89.3 90.0 90.9 89.3 90.5 FashionIQ 5.9 37.8 17.1 19.5 39.0 17.1 23.4 24.0 31.9 Wiki-SS-NQ 80.5 78.2 62.3 66.9 61.2 62.0 72.5 68.9 69.6 OVEN 50.0 75.1 66.5 64.3 60.8 66.9 71.4 61.4 60.7 EDIS 64.7 96.0 85.7 84.1 71.3 88.0 92.0 84.5 87.4 MSCOCO 36.7 31.4 75.7 67.1 84.7 69.5 72.7 92.9 93.6 RefCOCO 64.5 60.9 87.6 87.1 89.4 83.3 91.4 94.9 95.9 RefCOCO-Matching 3.9 78.4 84.6 85.8 83.0 84.4 91.1 85.8 88.0 Visual7W-Pointing 56.1 66.5 81.0 69.2 93.2 71.5 84.2 88.0 87.9

###### Table 8: Detailed results of baselines and Embed-RL on Video and Visual Doc of MMEB-v2 benchmark.

ColPali v1.3 [16] GME-7B [72] VLM2Vec-7B [26] VLM2Vec-V2-2B [41] CAFe-7B [63] UME-R1-2B [30] UME-R1-7B [30] Embed-RL-2B Embed-RL-4B

K700 23.4 39.7 35.5 38.0 40.1 35.8 42.8 55.8 56.8 SmthSmthV2 25.1 30.6 32.1 42.8 35.8 44.1 50.4 56.7 59.5 HMDB51 24.8 47.9 42.2 40.9 46.9 54.4 58.3 56.7 60.1 UCF101 49.4 54.7 61.8 60.0 39.6 67.2 70.0 79.3 78.5 Breakfast 10.9 14.3 23.8 14.8 16.6 20.1 21.5 36.7 33.0 MVBench 33.7 46.6 28.5 33.7 48.9 49.9 58.2 50.8 55.9 Video-MME 30.6 39.2 27.8 30.7 46.0 41.7 47.3 47.1 50.5 NExTQA 35.2 53.6 20.3 20.9 62.4 59.9 69.6 53.9 58.2 EgoSchema 38.4 46.8 21.8 34.0 60.0 45.4 52.4 53.0 52.8 ActivityNetQA 51.3 65.6 51.4 52.3 76.0 57.8 76.0 74.8 74.4 DiDeMo 22.8 26.4 29.3 30.4 37.8 32.4 40.0 45.3 46.8 MSR-VTT 17.6 31.8 34.5 28.3 36.5 34.3 38.9 45.7 46.2 MSVD 45.4 49.7 46.7 48.1 56.4 55.4 60.8 67.2 65.8 VATEX 16.7 24.9 25.5 26.5 32.0 29.9 32.6 43.6 43.4 YouCook2 5.3 9.1 9.0 10.6 9.5 12.7 18.5 23.5 23.3 QVHighlight 19.9 59.5 57.7 49.4 58.4 57.5 54.9 70.7 73.6 Charades-STA 29.0 14.0 19.8 20.2 18.7 20.4 21.9 26.4 25.0 MomentSeeker 27.6 37.4 39.3 40.8 41.4 41.2 41.1 50.9 49.9

ViDoRe_arxivqa 81.7 86.9 60.2 80.6 73.3 73.9 73.6 86.1 88.7 ViDoRe_docvqa 56.6 57.5 34.7 44.9 38.3 37.9 41.1 45.7 47.5 ViDoRe_infovqa 84.9 91.6 70.4 83.7 80.6 76.2 80.8 86.8 86.9 ViDoRe_tabfquad 86.9 94.6 78.2 89.2 80.7 86.1 90.2 94.5 94.7 ViDoRe_tatdqa 70.9 74.1 27.6 43.8 37.8 40.6 46.7 54.6 54.8 ViDoRe_shiftproject 75.1 96.8 38.6 60.8 52.0 66.8 65.0 70.7 69.0 ViDoRe_artificial_intelligence 95.7 99.6 67.7 88.5 86.0 85.9 89.5 94.0 91.6 ViDoRe_energy 94.7 95.3 60.4 86.5 84.8 83.3 85.7 86.7 88.1 ViDoRe_government_reports 93.6 98.8 61.8 85.0 85.0 82.6 89.8 89.0 90.7 ViDoRe_healthcare_industry 95.9 99.3 69.9 92.2 88.4 90.8 94.3 91.1 90.4 ViDoRe_esg_reports_human_labeled_v2 51.3 63.4 6.8 45.6 50.7 50.2 50.4 56.9 59.8 ViDoRe_biomedical_lectures_v2_multilingual 54.7 49.5 5.1 44.3 50.9 46.2 50.7 51.0 50.1 ViDoRe_economics_reports_v2_multilingual 49.0 54.2 13.9 43.0 54.3 45.7 57.8 53.0 53.9 ViDoRe_esg_reports_v2_multilingual 52.9 55.4 11.9 46.6 42.3 42.7 43.2 46.9 49.7 VisRAG_ArxivQA 80.9 87.4 52.6 76.9 74.0 74.3 80.5 84.9 86.9 VisRAG_ChartQA 72.3 86.1 57.7 83.7 82.7 86.0 85.0 88.3 88.5 VisRAG_MP-DocVQA 82.0 89.7 60.6 88.1 75.1 75.6 83.4 79.1 79.3 VisRAG_SlideVQA 85.1 92.6 54.7 84.1 87.6 87.1 91.5 92.3 92.6 VisRAG_InfoVQA 83.5 88.6 66.0 82.3 87.9 84.4 89.2 90.0 89.6 VisRAG_PlotQA 79.3 76.5 62.7 75.9 69.4 68.0 72.7 73.0 72.4 ViDoSeek-page 38.1 32.6 16.3 29.1 22.5 21.2 21.3 82.0 84.4 ViDoSeek-doc 87.5 90.3 69.4 79.0 73.8 75.9 75.3 82.6 82.4 MMLongBench-page 27.1 36.9 0.4 15.8 13.3 11.9 12.3 47.7 51.0 MMLongBench-doc 80.4 85.2 28.8 63.0 42.6 39.7 41.3 50.3 50.7

#### E Detailed Scores of MMEB-V1

We also report our performance on MMEB-V1 [26], including both in-domain and out-of-domain performance, as shown in Table 9.

- Table 9: Results on the MMEB-V1 benchmark (consisting of 36 image embedding tasks). IND and OOD denote the in-distribution and out-of-distribution datasets, respectively. The highest and second-highest values are highlighted in bold and underline.

Per Meta-Task Score Average Score

Model

Classification VQA Retrieval Grounding IND OOD Overall # of Datasets 10 10 12 4 20 16 36

Baseline Models

CLIP [43] 42.8 9.1 53.0 51.8 37.1 38.7 37.8 BLIP2 [34] 27.0 4.2 33.9 47.0 25.3 25.1 25.2 SigLIP [67] 40.3 8.4 31.6 59.5 32.3 38.0 34.8 OpenCLIP [10] 47.8 10.9 52.3 53.3 39.3 40.2 39.7 UniIR (BLIPFF) [57] 42.1 15.0 60.1 62.2 44.7 40.4 42.8 UniIR (CLIPSF) [57] 44.3 16.2 61.8 65.3 47.1 41.7 44.7 Magiclens [69] 38.8 8.3 35.4 26.0 31.0 23.7 27.8

MLLM-based Baseline Models

E5-V [25] 21.8 4.9 11.5 19.0 14.9 11.5 13.3 VLM2Vec-2B [26] 59.0 49.4 65.4 73.4 66.0 52.6 60.1 VLM2Vec-7B [26] 62.6 57.8 69.9 81.7 72.2 57.8 65.8 VLM2Vec-V2 [41] 62.9 56.3 69.5 77.3 68.8 59.9 64.9 MMRet-7B [77] 56.0 57.4 69.9 83.6 68.0 59.1 64.1 CAFe-V1-7B [63] 65.2 65.6 70.0 91.2 75.8 62.4 69.8 CAFe-V2-7B [63] 63.6 61.7 69.1 87.6 72.8 61.1 67.6 mmE5-11B [9] 67.6 62.8 70.9 89.7 72.3 66.7 69.8 LLaVE-2B [29] 62.1 60.2 65.2 84.9 69.4 59.8 65.2 LLaVE-7B [29] 65.7 65.4 70.9 91.9 75.0 64.4 70.3 UniME-4B [18] 54.8 55.9 64.5 81.8 68.2 52.7 64.2 UniME-7B [18] 66.8 66.6 70.6 90.9 74.6 65.8 70.7 UME-R1-2B [30] 64.8 62.8 67.6 77.2 71.5 60.4 66.6 UME-R1-7B [30] 67.1 69.2 71.9 84.9 76.1 65.1 71.3

Ours

Embed-RL-2B 62.8 67.9 68.6 90.4 71.9 65.9 69.2 Embed-RL-4B 63.7 70.5 71.3 91.4 74.3 67.3 71.2

#### F Prompt for synthesizing multimodal chain of thought

To enable precise guidance for visual reasoning and retrieval tasks, we design a hierarchically structured prompting system that instructs models to execute visual analysis tasks across text, image, and video modalities. This system consists of two scenario-specialized core modules, which impose constraints on reasoning logic, output formatting, and evidence anchoring, and adheres to a framework of two-round reasoning with fixed-format output to guarantee the consistency and accuracy of the generated results.

##### F.1 Basic Visual Reasoning Prompts

Such prompts guide the model to complete basic reasoning based on inputs, supporting various visual tasks, and are divided into 4 items according to inputs and objectives:

- – Text-to-Image Retrieval Reasoning Prompts: Focus on text-to-image retrieval, extract key visual concepts, anchor textual evidence to output a JSON keyword list, adapt to subtasks, and output results in a fixed format.
- – Image Reasoning Prompts: For image-based tasks, anchor visual evidence to output the 2D bounding box coordinates of key elements, locate core features, and complete reasoning and answer output in accordance with the process.
- – Video Reasoning Prompts: Adapt to video sequence tasks, output 1-based key frame indices based on frame visual evidence, identify core frames, and generate results following a fixed process.
- – Text-to-Video Retrieval Reasoning Prompts: For text-to-video retrieval, extract visual concepts containing temporal dynamics, output a JSON keyword list, and standardize result output according to subtasks. Text-to-Image Retrieval Visual Reasoning Prompt

You are a visual reasoning assistant specialized in text−to−image retrieval. Given a text description and a task, analyze the text content to determine the key visual concepts needed for retrieving matching images.

∗∗Rules:∗∗

- 1. Keep reasoning concise and grounded in textual evidence. Limit each step to 1−2 sentences.

- 2. Base your reasoning solely on the textual content and the task description.

- 3. Rephrase the final answer to preserve its exact meaning, changing only wording/phrasing if needed.

- 4. In your thinking process, you must extract and output key visual concepts from the text description. Use JSON format with key ’text_keywords’ to specify the important keywords as a list.

- 5. First, think between ‘<thinking>‘ and ‘</thinking>‘ while output necessary keywords from the text in JSON with key ’text_keywords’. Then, based on the thinking contents, rethink between ‘<rethink>‘

and ‘</rethink>‘. Finally, output the answer within ‘<answer>...</answer>‘.

- 6. Your thought process should adapt to the task type: ∗ For ∗∗caption−based retrieval∗∗ (e.g., find image from caption): Extract key visual elements,

objects, scenes, and relationships. ∗ For ∗∗news retrieval∗∗ (e.g., find news image from headline): Identify key people, locations, events, and contextual elements. ∗ For ∗∗dialogue−based retrieval∗∗ (e.g., find image from conversation): Summarize visual attributes, actions, and scene details. ∗ For ∗∗question−based retrieval∗∗ (e.g., find factual image): Identify the key concepts and

relationships. Now, process the following input: TASK/QUESTION: {question}

###### Image-based Visual Reasoning Prompt

You are a visual reasoning assistant. Given an image and a task description or question, analyze the image step−by−step to produce the required output. The task may involve image retrieval, classification, question answering, or object identification.

∗∗Rules:∗∗

- 1. Keep reasoning concise and grounded in visual evidence. Limit each step to 1−2 sentences.

- 2. Base your reasoning solely on the visual content of the image and the task description.

- 3. Rephrase the final answer to preserve its exact meaning, changing only wording/phrasing if needed.

- 4. In your thinking process, you must output coordinates for the key visual element(s) relevant to answering the question. Use JSON format with key ’bbox_2d’ to specify the bounding box as [x1, y1, x2, y2]. For multiple elements, use a list of bboxes: [[x1, y1, x2, y2], [x1, y1, x2, y2]].

- 5. First, think between ‘<thinking>‘ and ‘</thinking>‘ while output necessary coordinates needed to answer the question in JSON with key ’bbox_2d’. Then, based on the thinking contents and coordinates, rethink between ‘<rethink>‘ and ‘</rethink>‘. Finally, output the answer within ‘< answer>...</answer>‘.

- 6. Your thought process should adapt to the task: ∗ For ∗∗retrieval∗∗ (e.g., find similar image): Identify and locate key visual elements that define the

match. ∗ For ∗∗classification∗∗ (e.g., scene, object, domain): Locate distinguishing visual features that belong

to the class. ∗ For ∗∗question answering∗∗: Locate the visual or textual clues in the image that lead to the answer. ∗ For ∗∗object identification/segmentation∗∗: Provide the object’s location and boundaries.

Now, process the following input: IMAGE: {image} TASK/QUESTION: {question}

###### Video Sequence Visual Reasoning Prompt

You are a video reasoning assistant. Given a video sequence (multiple frames) and a task description, analyze the video content step−by−step to produce the required output. The task may involve video

captioning, video question answering, or video retrieval. ∗∗Rules:∗∗

- 1. Keep reasoning concise and grounded in visual evidence from the video frames. Limit each step to 1−2 sentences.

- 2. Base your reasoning solely on the visual content of the video frames and the task description.

- 3. Rephrase the final answer to preserve its exact meaning, changing only wording/phrasing if needed.

- 4. In your thinking process, you must identify and output key frames from the video sequence that are most relevant to answering the question. Use JSON format with key ’key_frames’ to specify the frame indices as a list (using 1−based indexing).

- 5. First, think between ‘<thinking>‘ and ‘</thinking>‘ while output necessary key frame indices in JSON with key ’key_frames’. Then, based on the thinking contents, rethink between ‘<rethink>‘ and ‘</

rethink>‘. Finally, output the answer within ‘<answer>...</answer>‘.

- 6. Your thought process should adapt to the task type: ∗ For ∗∗video captioning/description∗∗: Identify frames that show main events, transitions, or key

moments in the video sequence. ∗ For ∗∗video question answering∗∗: Locate frames that contain the visual evidence needed to answer the specific question. ∗ For ∗∗video retrieval∗∗: Identify frames that represent the core content or distinguishing features of

the video. Now, process the following input: VIDEO FRAMES: {video} TASK/QUESTION: {question}

###### Text-to-Video Retrieval Visual Reasoning Prompt

You are a visual reasoning assistant specialized in text−to−video retrieval. Given a video description and a task, analyze the text content to determine the key visual concepts needed for retrieving matching video clips or keyframes.

∗∗Rules:∗∗

- 1. Keep reasoning concise and grounded in textual evidence. Limit each step to 1−2 sentences.

- 2. Base your reasoning solely on the textual content and the task description.

- 3. Rephrase the final answer to preserve its exact meaning, changing only wording/phrasing if needed.

- 4. In your thinking process, you must extract and output key visual concepts from the text description. Use JSON format with key ’text_keywords’ to specify the important keywords as a list.

- 5. First, think between ‘<thinking>‘ and ‘</thinking>‘ while output necessary keywords from the text in JSON with key ’text_keywords’. Then, based on the thinking contents, rethink between ‘<rethink>‘

and ‘</rethink>‘. Finally, output the answer within ‘<answer>...</answer>‘.

- 6. Your thought process should adapt to the task type: ∗ For ∗∗video−based retrieval∗∗ (e.g., find video from description): Extract key visual elements,

objects, scenes, actions, temporal sequences, and relationships. ∗ For ∗∗scene−based retrieval∗∗ (e.g., find video from scene description): Identify key people, locations, events, contextual elements, and temporal progression. ∗ For ∗∗action−based retrieval∗∗ (e.g., find video from action sequence): Summarize visual attributes,

actions, scene details, and temporal dynamics. Now, process the following input: TASK/QUESTION: {question}

##### F.2 Positive Sample Verification Reasoning Prompts

These prompts analyze the rationale behind positive samples to support model training and verification, and are categorized into three types according to the positive sample modality:

- – Text Positive Sample Analysis Prompts: Anchor textual evidence to output a JSON-formatted keyword list, generate a text summary through two rounds of reasoning, and clarify the rationale for labeling the sample as positive.
- – Image Positive Sample Analysis Prompts: Infer image content in conjunction with the target task, output bounding box coordinates of key elements, generate an image summary through two rounds of reasoning, and clarify the core features and grounding basis.
- – Video Positive Sample Analysis Prompts: Anchor video evidence to output 1-indexed key frame indices, generate a structured video analysis through two rounds of reasoning, and clarify the core rationale for positive sample validity.

###### Positive Text Output Visual Reasoning Analysis Prompt

You are a visual reasoning assistant. Given a task description and a positive text output, analyze the text

to determine the key concepts that make it the correct output. ∗∗Rules:∗∗

- 1. Keep reasoning concise and grounded in the textual evidence. Limit each step to 1−2 sentences.

- 2. Use the task description to understand the context of the output.

- 3. In your thinking process, you must extract and output key concepts from the text. Use JSON format with key ’text_keywords’ to specify the important keywords as a list.

- 4. First, think between ‘<thinking>‘ and ‘</thinking>‘ while output necessary keywords from the text in JSON with key ’text_keywords’. Then, based on the thinking contents, rethink between ‘<rethink>‘

and ‘</rethink>‘. Finally, output a brief description of the text within ‘<answer>...</answer>‘. Now, process the following input: POSITIVE TEXT OUTPUT: {pos_text}

###### Positive Image Output Visual Reasoning Analysis Prompt

You are a visual reasoning assistant. Given a task description and a positive image output (with optional

text template), analyze the task to determine what the image should contain and locate key regions. ∗∗Rules:∗∗

- 1. Keep reasoning concise and grounded in the task description. Limit each step to 1−2 sentences.

- 2. Use the task description to infer what the target image should look like.

- 3. In your thinking process, you must output coordinates for the key visual element(s) relevant to answering the question. If no question is specified, output the most important element. Use JSON format with key ’bbox_2d’ to specify the bounding box as [x1, y1, x2, y2]. For multiple elements, use a list of bboxes: [[x1, y1, x2, y2], [x1, y1, x2, y2]].

- 4. First, think between ‘<thinking>‘ and ‘</thinking>‘ while output necessary coordinates in JSON with key ’bbox_2d’. Then, based on the thinking contents, rethink between ‘<rethink>‘ and ‘</rethink >‘. Finally, output a brief description of the image within ‘<answer>...</answer>‘.

Now, process the following input: POSITIVE TEXT OUTPUT: {pos_text} POSITIVE IMAGE OUTPUT: {pos_image_description}

###### Positive Video Output Visual Reasoning Analysis Prompt

You are a visual reasoning assistant for video. Given a task description and a positive video output, analyze

the output to determine the key moments or concepts that make it correct for the task. ∗∗Rules:∗∗

- 1. Keep reasoning concise and grounded in the provided video output evidence. Limit each step to 1−2 sentences.

- 2. Use the task description to understand the context of the output.

- 3. In your thinking process, you must identify and output the indices of the most relevant or representative frames from the described video sequence. Use JSON format with key ’key_frames’ to specify the

frame indices as a list (using 1−based indexing).

- 4. First, think between ‘<thinking>‘ and ‘</thinking>‘ while outputting necessary key frame indices in JSON with key ’key_frames’. Then, based on the thinking contents, rethink between ‘<rethink>‘ and ‘</rethink>‘. Finally, output a brief analysis of the video output within ‘<answer>...</answer >‘.

Now, process the following input: POSITIVE TEXT OUTPUT: {pos_text} POSITIVE VIDEO OUTPUT: {pos_video_output}

#### G Comprehensive Performance Characterization on Video Retrieval

To fully validate the cross-dimensional generalization capability of our proposed model in complex and diverse retrieval scenarios, we conduct a systematic performance evaluation on the Universal Video Retrieval Benchmark (UVRB) [20]. This benchmark consists of 16 datasets targeting distinct core capabilities, comprehensively covering multiple retrieval paradigms (textual, composed, and visual retrieval) and diverse semantic scenarios (coarse-grained, fine-grained, and longcontext retrieval). It thus enables accurate quantification of the model’s universal adaptation capacity across heterogeneous retrieval tasks. Experimental results demonstrate that our model outperforms counterparts with equivalent parameter scales, maintains a consistent performance advantage, and achieves the optimal comprehensive capability among models of the same parameter scale. Specifically, in core retrieval dimensions: (1) it attains state-of-the-art performance in coarse-grained semantic retrieval tasks and the second-best result in fine-grained semantic understanding scenarios; (2) it secures the optimal and second-best performances in spatial fine-grained perception (object/appearance recognition) and temporal fine-grained perception (motion/dynamics capture) subtasks, respectively. These results fully highlight the model’s robust capability in multi-dimensional semantic understanding and spatiotemporal feature extraction.

##### G.1 Video Retrieval Performance on UVRB Datasets

Baselines. Following the experimental settings of GVE [20], we evaluate 16 representative baselines spanning diverse architectures, parameter scales, and training data compositions. These baselines are categorized into two groups: (1) traditional CLIP-based embedding models, including CLIP4Clip [40], ViCLIP

- [54], VideoCLIP-XL [48], LanguageBind [78], and the InternVideo2 (1B/6B)
- [55]; (2) recent MLLM-based embedding models, including GVE-2B/7B [20], GME-2B/7B [72], Unite-2B/7B [27], VLM2Vec-V2 [41], BGE-VL [76], UniME-7B [18], and B3-7B [46].

Datasets. We adopt the Universal Video Retrieval Benchmark (UVRB) [20], which assesses model universality via 16 test datasets targeting distinct core abilities. UVRB covers diverse retrieval scenarios, with datasets categorized as: coarse-grained retrieval (MSRVTT [61], DiDeMo [1], CRB-G [62]); fine-grained retrieval (spatial: CRB-S [62]/VDC-O [7], temporal: CRB-T [62]/CMRB [37], partially relevant: DREAM-E [49]/LoVR-Theme2Clip [5]/PEV-K [4]); longcontext retrieval (LoVR-V [5], VDC-D [7]); and composed query retrieval (MSTI/MS-TV adapted from MomentSeeker [66], MSRVTT-I2V [61], LoVR-C2V [5]).

Metrics. Following GVE [20], we adopt Recall@1 (R@1) as the primary metric, which measures the accuracy of identifying the most relevant item. For challenging datasets with ambiguous queries (e.g., CMRB, LoVR-TH), we additionally report Recall@10 (R@10) to reflect performance on top-k retrieval. For

- Table 10: Performance of video retrieval on UVRB datasets: AVG values represent the average across 16 datasets, with the highest score in each column bolded and the second-highest underlined. Metrics include R@1 (Recall@1), R@10 (Recall@10) and P@1 (Precision@1). The highest and second-highest values are highlighted in bold and underline.

Model AVG MSRVTT DiDeMo CRB-G CRB-S VDC-O CRB-T CMRB DREAM-E

R@1 R@1 R@1 R@1 R@1 R@1 R@10 R@1

CLIP4Clip [40] 39.0 33.3 29.7 51.1 49.7 62.0 28.9 28.0 19.1 ViCLIP [54] 35.2 38.6 30.6 44.7 43.7 53.0 34.9 22.9 23.5 VideoCLIP-XL [48] 49.1 44.3 40.3 82.8 83.9 73.5 48.7 27.4 26.3 LanguageBind [78] 48.7 47.9 42.1 71.6 68.7 75.9 46.6 29.0 28.0 InternVideo2-1B [55] 40.4 44.9 40.4 58.6 56.8 64.4 47.0 35.5 24.2 InternVideo2-6B [55] 42.7 48.5 41.8 60.8 61.2 65.0 45.5 34.6 27.1 GME-2B [72] 48.8 39.0 30.3 69.0 71.8 71.5 40.0 29.8 24.0 Unite-2B [27] 48.0 36.7 29.8 69.9 72.3 72.7 40.9 28.4 22.3 VLM2Vec-V2 [41] 50.8 33.0 29.9 82.8 84.3 77.5 41.0 28.6 22.8 BGE-VL [76] 44.3 33.7 31.8 69.0 68.8 63.9 35.9 22.5 21.2 UniME-7B [18] 52.1 35.1 33.5 81.5 82.7 74.3 47.6 31.7 29.3 B3-7B [46] 51.1 28.2 35.0 81.5 82.5 76.8 41.5 31.2 21.6 GME-7B [72] 53.0 43.6 37.7 74.0 76.7 73.1 44.2 30.4 27.4 Unite-7B [27] 53.8 43.9 38.6 79.8 80.4 75.3 47.2 35.1 27.9 GVE-3B [20] 54.4 43.1 37.6 85.0 84.6 78.6 49.6 36.3 28.0 GVE-7B [20] 57.3 46.4 43.3 86.5 84.7 79.4 53.9 39.8 30.2

Embed-RL-2B 58.7 43.7 42.2 91.4 89.7 84.6 49.4 36.5 31.8 Embed-RL-4B 60.2 44.0 46.1 92.1 89.9 85.9 53.8 38.2 32.0

Model LoVR-TH PEV-K LoVR-V VDC-D MS-TI MS-TV MSRVTT-I2V LoVR-C2V

R@10 R@1 R@1 R@1 P@1 P@1 R@1 R@1

CLIP4Clip [40] 33.8 17.9 36.0 56.6 17.3 18.3 92.4 50.3 ViCLIP [54] 20.2 7.5 23.0 39.5 28.3 24.3 84.6 43.3 VideoCLIP-XL [48] 43.9 22.9 38.0 82.0 23.0 22.3 86.1 40.3 LanguageBind [78] 42.5 30.3 54.0 67.9 22.8 23.3 82.7 46.3 InternVideo2-1B [55] 29.8 2.6 28.0 48.5 26.5 23.0 79.4 36.8 InternVideo2-6B [55] 30.2 8.6 33.0 51.6 23.5 20.5 86.8 45.2 GME-2B [72] 44.6 35.4 53.0 83.9 35.0 34.0 82.7 36.6 Unite-2B [27] 44.5 35.5 57.0 79.2 25.0 23.3 86.3 44.5 VLM2Vec-V2 [41] 49.2 32.4 61.0 91.3 27.5 25.0 84.1 38.5 BGE-VL [76] 38.7 18.4 55.0 72.2 30.3 23.3 77.9 46.5 UniME-7B [18] 50.4 32.3 48.0 84.7 31.0 30.5 86.7 53.7 B3-7B [46] 46.2 38.7 59.0 85.3 27.5 26.5 88.4 47.1 GME-7B [72] 52.3 39.6 71.0 86.5 34.8 33.3 86.0 37.0 Unite-7B [27] 55.5 44.0 62.0 87.1 27.8 23.0 88.3 44.8 GVE-3B [20] 52.2 33.0 61.0 91.8 34.0 26.8 89.1 40.3 GVE-7B [20] 54.2 41.3 68.0 94.8 34.3 28.0 89.9 41.5

Embed-RL-2B 56.5 33.6 80.0 93.8 19.3 21.0 89.1 51.4 Embed-RL-4B 57.9 31.9 77.0 95.2 15.8 21.0 87.9 49.0

MS-TI and MS-TV (containing multiple positive candidates), we use Precision@1 (P@1) as the key metric.

Performance. Table 10 presents the video retrieval performance of diverse models on the Universal Video Retrieval Benchmark (UVRB) [20] datasets. Our Embed-RL-2B and Embed-RL-4B models outperform all baselines in the average score (AVG), with Embed-RL-4B achieving the highest AVG (0.602). Both models secure the first or second place on key datasets across different retrieval paradigms, including coarse-grained retrieval (DiDeMo [1], CRB-G [62]) and fine-grained/long-context retrieval (CRB-S [62], VDC-O/VDC-D [7]), which verifies their superior multi-dimensional video retrieval capability.

##### G.2 Capability Characterization of the UVRB Evaluation Metrics

The Universal Video Retrieval Benchmark (UVRB) adopts unweighted arithmetic means for all metric calculations to ensure fair comparison across heterogeneous datasets, and comprehensively evaluates video retrieval models from three orthogonal dimensions (Tasks, Domains, Sub-domains).

UVRB covers 16 datasets that are exhaustively partitioned into non-overlapping categories according to Tasks, Domains and Sub-domains, as detailed in Table 11. This partition is the foundation for quantifying distinct model capabilities in different retrieval scenarios.

- Table 11: Detailed Partition of Datasets in the Universal Video Retrieval Benchmark (UVRB) Across Tasks, Domains, and Sub-domains

Partition Content DTXT {MSRVTT, DiDeMo, CRB-G, CRB-S, VDC-O, CRB-T, CMRB,

DREAM-E, LoVR-TH, PEV-K, LoVR-V, VDC-D} DCMP {MS-TI, MS-TV} DVIS {MSRVTT-I2V, LoVR-C2V} DCG {MSRVTT, DiDeMo, CRB-G} DFG {CRB-S, VDC-O, CRB-T, CMRB, DREAM-E, LoVR-TH, PEV-K} DLC {LoVR-V, VDC-D} DS {CRB-S, VDC-O} DT {CRB-T, CMRB} DPR {DREAM-E, LoVR-TH, PEV-K}

Based on the above dataset partition, Table 12 defines the calculation rules and corresponding capability characterization for each evaluation dimension. All metrics are computed as unweighted arithmetic means of corresponding datasets (denoted as D), with results rounded to three decimal places. The three core dimensions are defined as follows:

- 1. Task dimension: Distinguishes retrieval paradigms by query formats (textual, composed, visual), reflecting cross-modal alignment ability for different query types;
- 2. Domain dimension: Assesses model performance across different levels of semantic granularity (coarse-grained, fine-grained, long-context), measuring generalization on retrieval tasks involving short/long context and high/lowlevel semantics;
- 3. Sub-domain dimension: Further decomposes fine-grained retrieval into three sub-tasks (spatial, temporal, partially relevant), pinpointing model strengths and weaknesses in fine-grained understanding.

The overall AVG score is the arithmetic mean of the three task columns (TXT, CMP, VIS) and three domain columns (CG, FG, LC). This score aggregates model performance across core retrieval paradigms, rather than taking raw averages over datasets.

- Table 12: Calculation Rules and Capability Characterization for Model Evaluation on UVRB (Table 13). All Metrics Are Computed as Unweighted Arithmetic Means of Corresponding Datasets (Denoted as D), with Results Rounded to Three Decimal Places.

Dimension Level Column Name Capability Characterization

Tasks TXT Text-to-video retrieval (12 datasets) Tasks CMP Composed retrieval (text+image/video, 2 datasets) Tasks VIS Visual-to-video retrieval (image/video, 2 datasets)

Domains CG Coarse-grained semantic retrieval (3 datasets) Domains FG Fine-grained semantic retrieval (7 datasets) Domains LC Long-context retrieval (long text/video, 2 datasets)

- Sub-domains S Spatial fine-grained (object/appearance, 2 datasets)
- Sub-domains T Temporal fine-grained (motion/dynamics, 2 datasets) Sub-domains PR Partially relevant retrieval (3 datasets) Overall AVG Aggregated task and domains performance (core retrieval paradigms)

Table 13 presents the video retrieval performance of mainstream models on UVRB, characterized by the above-defined abilities (Tasks, Domains, and Sub-domains). The AVG score is the arithmetic mean of performance across textual (TXT), composed (CMP), and visual (VIS) retrieval tasks, aggregating model performance across core retrieval paradigms (rather than raw averages over datasets).

- Table 13: Video Retrieval Performance of Models on UVRB by Ability Dimensions (Tasks, Domains, Sub-domains). The AVG score denotes the mean performance across textual (TXT), composed (CMP), and visual (VIS) retrieval tasks. Domains involve coarse-grained (CG), fine-grained (FG), and long-context (LC) retrieval, while subdomains include spatial (S), temporal (T), and partially relevant (PR) retrieval. The highest and second-highest values are highlighted in bold and underline.

Model Tasks Domains Sub-domains AVG TXT CMP VIS CG FG LC S T PR

CLIP4Clip [40] 41.6 40.1 17.8 71.4 38.0 36.0 46.3 55.9 28.5 23.6 ViCLIP [54] 37.5 33.6 26.3 64.0 38.0 31.5 31.3 48.4 28.9 17.1 VideoCLIP-XL [48] 51.0 55.0 22.7 63.2 55.8 49.3 60.0 78.7 38.1 31.0 LanguageBind [78] 50.8 54.3 23.1 64.5 53.9 47.9 61.0 72.3 37.8 33.6 InternVideo2-1B [55] 42.0 42.2 24.8 58.1 48.0 40.3 38.3 60.6 41.3 18.9 InternVideo2-6B [55] 44.5 44.8 22.0 66.0 50.4 41.7 42.3 63.1 40.0 22.0 GME-2B [72] 41.6 53.9 34.5 59.7 46.1 47.1 68.5 71.6 34.9 34.7 Unite-2B [27] 50.7 53.6 24.2 65.4 45.5 47.1 68.1 72.5 34.7 34.1 VLM2Vec-V2 [41] 53.8 58.7 26.3 61.3 49.8 50.2 76.2 80.9 34.8 34.8 BGE-VL [76] 48.0 49.7 26.8 62.2 44.8 40.6 63.6 66.4 29.2 26.1 UniME-7B [18] 54.2 56.1 30.8 70.2 50.0 51.8 66.4 78.5 39.6 37.3 B3-7B [46] 53.8 57.0 27.0 67.8 48.2 50.5 72.2 79.7 36.4 35.5 GME-7B [72] 56.2 60.4 34.1 61.5 51.8 50.7 78.8 74.9 37.3 39.8 Unite-7B [27] 55.9 60.9 25.4 66.6 54.1 53.9 74.6 77.9 41.2 42.5 GVE-3B [20] 57.1 61.9 30.4 64.7 55.2 54.1 76.4 81.6 43.0 37.7

Embed-RL-2B 58.7 61.1 20.1 70.3 59.1 54.6 86.9 87.2 43.0 40.6 Embed-RL-4B 58.5 62.0 18.4 70.3 60.7 55.6 86.1 87.9 46.0 40.6

#### H Training Trajectory Dynamics

This section details dynamic training trajectories of 2B and 4B Embed-RL models via core metrics.

##### H.1 Training Metrics

We track key metrics throughout the reinforcement learning phase, as shown in Figure 9. We observe that the entropy declines gradually and then plateaus, while the response length increases steadily with ongoing training. Meanwhile, the reward exhibits a fluctuating upward trend, which is attributed to the effects of in-batch reward sample sampling as the discrepancies between individual samples far outweigh the inherent growth of the reward itself.

Entropy

Response Length

Reward

240

0.32

0.40

210

ResponseLength

Entropy

Reward

180

0.24

0.36

150

0.16

120

0.32

0 100 200 300 Step

0 100 200 300 Step

0 100 200 300 Step

- Fig. 9: Key RL-phase metrics of Embedder-Guided RL (entropy, response length, reward).

Additionally, we track two core training metrics for 2B and 4B-scale EmbedRL models during contrastive learning: contrastive training loss and gradient norm, as presented in Figure 10. Based on our training experience, neither an excessively large nor an overly small converged loss is favorable.An overly large converged loss suggests that the model fails to correctly discriminate positive and negative samples, whereas an excessively small one indicates that in-batch negative samples are too easily distinguished, leaving the model unable to learn effective discriminative information.Proper adjustment of the sampling ratio and sub-batch size is thus required to ensure the model converges correctly.

###### Embed-RL-2B Training Loss

###### Embed-RL-2B Grad Norm

- 0

- 1

- 2

- 3

15

GradNormValue

10

LossValue

5

0

0 1000 2000 3000 Step

0 1000 2000 3000 Step

###### Embed-RL-4B Training Loss

Embed-RL-4B Grad Norm

1.5

40

30

GradNormValue

1.0

LossValue

20

0.5

10

0.0

0

0 2000 4000 6000 Step

0 2000 4000 6000 Step

- Fig. 10: Contrastive training loss and gradient norm of 2B and 4B scale Embed-RL models.

#### I Efficiency and Latency

The proposed evidential Traceability CoT (T-CoT) demonstrates prominent efficiency advantages and negligible latency overhead compared with traditional generative embedding methods. For any multimodal retrieval target, T-CoT only needs to be generated once offline, and the derived multimodal embedding vector can be directly stored in the retrieval database. This is different from generative embedding approaches that require on-the-fly reasoning chain and embedding generation for each query, which leads to repeated computational costs. Designed to be targeted and concise, T-CoT only extracts core retrievalrelated multimodal cues such as text keywords, image bounding boxes and video keyframes while abandoning redundant content. This ensures it does not significantly increase single embedding inference latency. Additionally, the stable semantic representation of T-CoT-based embeddings allows for long-term caching and reuse in subsequent tasks without frequent re-generation or updates. This further reduces inference latency and computational consumption in largescale scenarios and makes the framework more suitable for practical industrial deployment.

#### J Limitations

This work has several notable limitations. First, the weight coefficients of the multi-component reward function are empirically set for simplicity, lacking an adaptive optimization mechanism for diverse multimodal tasks, which may lead to suboptimal performance in specific scenarios. Second, the constructed dataset excludes partial classification tasks, resulting in relatively weak performance on image classification subtasks; we recommend designing additional loss for classification tasks to avoid false negatives while adapting to large-batch contrastive loss. Finally, we have not applied any hard negative sample mining or curriculum learning strategies, which are expected to further enhance the model’s discriminative capability and training stability if incorporated.

#### K Exploratory Perspectives

In numerous practical systems such as Multimodal Content Understanding, Recommendation Systems (RS), and Retrieval-Augmented Generation (RAG), high-quality Universal Multimodal Embeddings serve as the core foundation supporting downstream tasks. The quality of embeddings directly determines the performance ceiling of downstream tasks—whether it is the matching accuracy of recommendation systems, the answer correctness of RAG systems, or the semantic alignment performance of cross-modal tasks, all are closely related to it.

In existing solutions, to extract richer semantic information, multi-dimensional feature fusion is often performed through techniques such as semantic segmentation, image and video fine-grained understanding, and user profile analysis. However, this approach typically incurs additional computational and storage overhead, and the feature fusion module suffers from high design complexity.

To address this, we strive to deeply internalize key information capture capabilities, including core visual region localization, video keyframe extraction, and core semantic keyword mining, into the embedding large model itself. This exploration aims to simplify the model structure of feature fusion, reduce the computational and storage costs of semantic vectors, and simultaneously enable embedding results to naturally incorporate task-relevant core semantic information.

#### L Comparative Examples of T-CoT Before and After Embedder-Guided RL

Figures 11 to 8 present additional comparative examples of T-CoT before and after the application of Embedder-Guided Reinforcement Learning. These examples fully demonstrate that after RL optimization, the Reasoner achieves more accurate localization of key regions and exhibits improved embedding quality, which further validates the effectiveness of the proposed EG-RL approach in enhancing the performance of T-CoT.

Text-Image → Text

Query:

Represent the given image with the following question: Is the plastic bowl to the left

of the bread on the right?

###### Query T-CoT Before RL:

<thinking>The image shows a meal in bento boxes on a checkered tablecloth. The bread

[Figure 172]

[Figure 173]

| |
|---|

is not visible; instead, there are apple slices in a clear plastic container on the top left

and rice in a blue container on the bottom left. The \"bread\" mentioned in the

question is not present in the image. The plastic bowl (containing apple slices) is on the

left side of the image, but there is no bread to its right. {\"bbox_2d\": [[100, 0, 450,

250]]}</thinking>\n<rethink>The plastic bowl with apple slices is located on the left,

but there is no bread visible anywhere in the image, so the condition \"to the left of

the bread on the right\" cannot be evaluated as true or false.</rethink>\n<answer>No,

there is no bread visible in the image, so the plastic bowl is not to the left of the

bread on the right.</answer>

[Figure 174]

Similarity: 0.5742

[Figure 175]

Failed to ground.

###### Query T-CoT After RL:

<thinking>The image shows a meal in bento-style containers on a checkered tablecloth.

[Figure 176]

| | |
|---|---|
| | |
| | |

[Figure 177]

On the right side, there is a container with bread (specifically, sliced bread with

spinach) and vegetables. To the left of this container, there is a plastic bowl containing

sliced apples. The question asks if the plastic bowl is to the left of the bread on the

right. The plastic bowl is indeed located to the left of the bread container.

{\"bbox_2d\": [[105, 0, 437, 280], [437, 105, 870, 720]]}</thinking>\n<rethink>The plastic bowl with apples is positioned to the left of the container with bread and

vegetables, which is on the right side of the image. This confirms the spatial

relationship described in the question.</rethink>\n<answer>Yes, the plastic bowl is to

the left of the bread on the right.</answer>"

[Figure 178]

Similarity: 0.7070

[Figure 179]

Ground to accurate information.

###### Target:

Yes, the bowl is to the left of the bread. Target T-CoT:

<thinking>The task involves spatial reasoning about object placement. The output

confirms a relative position: \"bowl\" is to the \"left of the bread.\" Key concepts

include object names and directional relationship. {\"text_keywords\": [\"bowl\",

\"bread\", \"left\", \"Yes\"]}</thinking>\n<rethink>The keywords indicate a spatial

relationship between two objects, with a clear affirmative confirmation of their

relative positioning.</rethink>\n<answer>The text confirms that the bowl is positioned

to the left of the bread.</answer>

###### Fig. 11: Example 4 of T-CoT Before and After EG-RL.

Text-Image → Text

###### Query:

Represent the given image with the following question: What is needed to use this transportation?

[Figure 180]

###### Query T-CoT Before RL:

<thinking>The image shows a black car being lifted by a car lift, with a person in a blue

shirt and red cap operating the lift. The car is on a platform, and the lift mechanism is

| |
|---|

visible. The key visual element is the car lift, which requires electricity to operate.

{\"bbox_2d\": [550, 340, 997, 997]}</thinking>\n<rethink>The car lift in the image is a

mechanical device that requires power to function, so electricity is needed to use this

transportation-related equipment.</rethink>\n<answer>Electricity is needed to use this

transportation.</answer>

[Figure 181]

Similarity: 0.4141

[Figure 182]

Missed grounding.

###### Query T-CoT After RL:

<thinking>The image shows a black car parked on a ramp, with a man in a red cap and

blue shirt appearing to work on or inspect it. The car is positioned on a loading ramp,

[Figure 183]

suggesting it is being loaded or unloaded, likely for transport. To use this type of transportation (a car), one typically needs a driver and a vehicle in operational condition. The man’s activity implies he may be a mechanic or technician, indicating that maintenance or preparation is needed. The key visual elements are the car and the

man interacting with it. {\"bbox_2d\": [[540, 350, 997, 930], [700, 610, 950,

997]]}</thinking>\n<rethink>The car is the primary transportation vehicle, and the

man’s presence suggests that maintenance or preparation is required to use it. The

ramp indicates it may be part of a transport or loading process, reinforcing the need

[Figure 184]

Similarity: 0.6055

for operational readiness.</rethink>\n<answer>To use this transportation, a driver and

a properly maintained vehicle are needed.</answer>

[Figure 185]

Complete grounding.

###### Target:

driver Target T-CoT:

<thinking>The task requires identifying key concepts from the positive text output

\"driver\". The word \"driver\" is a noun and likely refers to a person who operates a

vehicle or a software component that controls hardware. Key concepts include the role

or function implied by the term. {\"text_keywords\": [\"driver\", \"person\", \"vehicle\", \"software\", \"hardware\"]}</thinking>\n<rethink>The term \"driver\" is

contextually ambiguous but commonly refers to either a human operator or a software

component; both interpretations are valid based on common usage.</rethink>

\n<answer>The text output \"driver\" refers to either a person who operates a vehicle

or a software component that interfaces with hardware.</answer>

Fig. 12: Example 5 of T-CoT Before and After EG-RL.

Text-Video → Text

[Figure 186]

###### Query:

Who did Chandler think was \"the other guy\" after Joey said he had a feeling Kathy

was into some other guy?

Query T-CoT Before RL: <thinking>The video frames show two men, Chandler and Joey, engaged in a

[Figure 187]

conversation. Their expressions and body language suggest a comedic exchange. The

question references a specific line from the show \“Friends\” where Chandler jokes that Joey is \“the other guy\” after Joey expresses concern about Kathy. The visual

content aligns with this scene, showing the characters interacting in a way consistent

with their dynamic. {\“key_frames\”: [1, 2, 3, 4, 5, 6, 7, 8]}</thinking>\n<rethink>The frames depict Chandler and Joey in a familiar setting, with their expressions and gestures indicating a humorous dialogue. This matches the context of the question, which references a well-known joke from the show.</rethink>\n<answer>

[Figure 188]

Chandler</answer>

[Figure 189]

Similarity: 0.6719

[Figure 190]

[Figure 191]

Excessive attention to non-key points.

###### Query T-CoT After RL:

[Figure 192]

<thinking>The video frames show two men, Joey and Chandler, engaged in a

conversation. In frames 1, 3, and 7, Chandler appears to be reacting with surprise or disbelief, which aligns with his character's comedic response to Joey's statement

about Kathy. The context of their interaction and Chandler's facial expressions

suggest he is reacting to Joey's comment about \"the other guy.\" {\"key_frames\": [1, 3, 7]}</thinking>\n<rethink>Chandler's surprised reaction in frames 1, 3, and 7

[Figure 193]

indicates he is responding to Joey's claim about Kathy being interested in someone

else, which is consistent with his character's typical behavior in this

scenario.</rethink>\n<answer>Chandler thought it was himself.</answer>

[Figure 194]

Similarity: 0.7773

[Figure 195]

[Figure 196]

Focus on key points.

Target:

[Figure 197]

Chandler thinks the \"other guy\" is himself.

###### Target T-CoT :

<thinking>The output states a belief held by Chandler regarding \"the other guy.\" The

key elements are Chandler’s perspective, the phrase \"the other guy,\" and the selfreferential conclusion that it is himself. {\"text_keywords\": [\"Chandler\", \"the other guy\", \"himself\", \"thinks\"]}</thinking>\n<rethink>The keywords indicate that the output is a statement of Chandler's belief that \"the other guy\" refers to

himself, implying self-awareness or confusion.</rethink>\n<answer>Chandler believes

that \"the other guy\" refers to himself.</answer>

###### Fig. 13: Example 6 of T-CoT Before and After EG-RL.

Text → Video

###### Query:

A young girl does gymnastics outside in the middle of grass in a sunlight.

###### Query T-CoT:

<thinking>The description specifies a young girl performing gymnastics outdoors on

[Figure 198]

grass under sunlight. Key visual elements include the subject (young girl), activity

(gymnastics), location (grass), and environmental condition (sunlight). These are critical

for video retrieval to match the correct visual content. {\"text_keywords\": [\"young girl\", \"gymnastics\", \"outside\", \"grass\", \"sunlight\"]}</thinking>\n<rethink>The

task requires identifying a video of a young girl doing gymnastics on grass in sunlight,

so the retrieval should prioritize these core visual concepts to ensure accurate

[Figure 199]

match.</rethink>\n<answer>A video of a young girl performing gymnastics on grass

outdoors under sunlight.</answer>

###### Target T-CoT Before RL:

<thinking>The video shows a young girl in a green top and pink skirt playing

energetically on a grassy lawn in front of buildings. Frame 1 shows her posing, Frame 2

[Figure 200]

shows her smiling and moving, Frame 3-5 capture her performing a cartwheel, and Frame 7 shows her running toward the camera. Frame 8 is a still photo of her hugging

another child. These frames represent the main actions and the conclusion.

{\”key_frames\“: [1, 2, 3, 5, 7, 8]}</thinking>\n<rethink>The key frames (1, 2, 3, 5, 7, 8)

capture the girl‘s initial pose, her joyful movement, the cartwheel action, her running

toward the camera, and the final embrace with another child, showing the full

[Figure 201]

sequence of play and interaction.</rethink>\n<answer>The video shows a young girl in a green top and pink skirt playing on a grassy lawn in front of buildings. She poses, runs,

performs a cartwheel, and then runs toward the camera before hugging another child

in a final photo.</answer>

[Figure 202]

[Figure 203]

Similarity: 0.4961

[Figure 204]

Overly detailed and off-topic.

[Figure 205]

###### Target T-CoT After RL :

<thinking>The video shows a young girl playing energetically on a grassy lawn in front of

buildings. Frame 1 establishes the setting and the girl’s initial pose. Frames 2-5

capture her dynamic movements, including a jump and a handstand. Frame 6 shows her

[Figure 206]

running toward the camera. Frame 7 shows her running closer while smiling. Frame 8

ends with her hugging another child. These frames represent the key actions and transitions. {\"key_frames\": [1, 3, 5, 6, 8]}</thinking>\n<rethink>The key frames (1, 3,

5, 6, 8) capture the sequence: initial pose, acrobatic moves, running toward camera, and

the final embrace, showing the progression of playful activity.</rethink>\n<answer>The

video shows a young girl in a turquoise top and pink skirt playing on a grassy lawn in

[Figure 207]

front of buildings. She performs a jump and a handstand, then runs toward the camera,

ending with her hugging another child.</answer>

[Figure 208]

Similarity: 0.5664

[Figure 209]

Concise and query-aligned.

###### Fig. 14: Example 7 of T-CoT Before and After EG-RL.

