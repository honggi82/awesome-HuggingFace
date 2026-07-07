# arXiv:2503.12797v3[cs.CV]2Apr2026

## KARL: Knowledge-Aware Reasoning and Reinforcement Learning for Knowledge-Intensive Visual Grounding

Xinyu Ma1∗, Ziyang Ding2∗, Zhicong Luo3, Chi Chen4†, Zonghao Guo4, Xuebo Liu1, Derek F. Wong1†, Zhen Zhao5, Xiaoyi Feng3, Maosong Sun4

1University of Macau, 2Shandong University, 3Northwestern Polytechnical University, 4Tsinghua University, 5Shanghai AI Laboratory

chenchithu@gmail.com, derekfw@um.edu.mo

### Abstract

Knowledge-Intensive Visual Grounding (KVG) requires models to localize objects using fine-grained, domain-specific entity names rather than generic referring expressions. Although Multimodal Large Language Models (MLLMs) possess rich entity knowledge and strong generic grounding capabilities, they often fail to effectively utilize such knowledge when grounding specialized concepts, revealing a knowledge– grounding gap between internal knowledge and grounding predictions. To address this challenge, we propose a knowledge-aware training paradigm for KVG. Our approach first constructs knowledge-guided reasoning data to encourage models to activate domain-relevant entity knowledge during grounding, and then introduces KARL, a Knowledge-Aware Reinforcement Learning framework that adaptively modulates reward signals according to the model’s estimated knowledge mastery of different entities. To facilitate systematic evaluation, we introduce KVG-Bench, a benchmark spanning 10 domains with 1.3K curated test cases covering 531 images and 882 entities. Extensive experiments show that our approach consistently outperforms a wide range of baseline models and achieves substantially stronger cross-domain generalization on unseen categories. The data, codes, and models are released at https://github.com/thunlp/KARL

### 1 Introduction

Human experts demonstrate fine-grained discrimination of visual concepts by flexibly leveraging domain knowledge to refine discriminative features, resulting in superior domain-specific visual perception compared to non-experts [36, 15, 11]. In contrast, recent Multimodal Large Language Models (MLLMs) exhibit rich fine-grained knowledge of diverse entities [14, 13] and strong generic visual grounding capabilities [40, 6, 1], yet often fail to operationalize such knowledge when perception must culminate in a grounding prediction. As illustrated in fig. 1a, for the same entity, a model may correctly identify the queried concept yet predict an erroneous bounding box when required to ground it in the image. This discrepancy suggests the presence of a knowledge–grounding gap, where internal entity knowledge and grounding ability do not reliably compose into accurate knowledge-intensive grounding.

To systematically investigate this issue, we introduce Knowledge-Intensive Visual Grounding (KVG), a visual grounding task that extends conventional visual grounding [16, 49] by requiring

∗Equal contribution †Corresponding author

Preprint.

Knowledge-Grounding Gap

[Figure 1]

56.94%

Q1 Clumber Spaniel

Inconsistency

SeenUnseenoverall

MLLM

- 56.69%

60.25%

61.76%

59.22%

- 57.19%

Q2

(488,306), (999,989)

67.3%

To identify the Clumber Spaniel in the image, we need to carefully analyze the visual features of each dog and compare them …

( Planning)

57.54%

Q1.What breed is the left dog?

1. Clumber Spaniel: Color: The Clumber Spaniel has a distinctive white face with… 2. American Water Spaniel: Color: The American Water Spaniel has a dark brown body with a lighter brown head. Size and Shape: It has a more slender build compared to the Clumber Spaniel …

Q2.Where is the Clumber Spaniel

( Visual Analysis)

57.72%

Knowledge-Guided Reasoning Grounding

70.04%

KARL

By comparing these features: The Clumber Spaniel has a distinctive white face … which matches the dog on the left., …

( Comparison)

57.11%

68.41%

Therefore, based on the visual features and comparisons, the dog on the left is identified as the Clumber Spaniel.

( Conclusion)

(0,19), (435,997)

50% 56.667% 63.333% 70%

Qwen3-VL-8B Qwen3-VL-8B (SFT) Qwen3-VL-8B (CoT) KARL

- Figure 1: (a) While the MLLM can correctly recognize the entity (Q1), it fails to ground it (Q2), revealing an inconsistency between knowledge and grounding. Our method integrates knowledgeguided reasoning to bridge this gap. (b) (b) KARL achieves substantially stronger grounding performance than the baseline model and zero-shot CoT prompting, showing that knowledge-guided reasoning for KVG cannot be effectively induced by simple prompting alone.

both fine-grained visual perception and domain-specific entity knowledge integration. As shown in fig. 1a, each query in KVG utilizes domain-specific terminology (“Clumber Spaniel”) rather than generic descriptions (“the left dog”), and images contain visually similar distractors that necessitate knowledge-guided differentiation. Thus, successful grounding depends not only on recognizing visual patterns, but also on aligning domain-level knowledge with spatial evidence selection. Unlike the mathematical and geometric tasks studied by previous reasoning MLLMs [44, 45], KVG emphasizes the integration of entity knowledge within visual perception processes.

By requiring the composition of entity knowledge with fine-grained visual discrimination, KVG poses substantial challenges for current MLLMs. For instance, although Qwen3-VL-8B [1] achieves high accuracy on the widely used RefCOCO benchmark [49], its performance substantially declines on KVG (fig. 1b), indicating that strong generic grounding does not readily generalize to knowledgeintensive settings. Moreover, even when equipped with explicit step-by-step reasoning prompting or trained with recent reasoning-guided visual grounding frameworks [2, 48], models fail to demonstrate consistent performance improvements on KVG. These observations suggest that the core difficulty lies not in insufficient reasoning depth, but in the ineffective alignment between fine-grained entity knowledge and grounding decisions.

To mitigate this limitation, we propose a knowledge-aware training paradigm tailored to knowledgeintensive visual grounding. Our approach begins with constructing knowledge-guided reasoning data that encourages models to explicitly activate domain-relevant entity knowledge prior to producing grounding predictions. By guiding models to articulate and connect entity-level knowledge with visual evidence, this stage aims to strengthen the alignment between knowledge utilization and grounding. Building on this foundation, we introduce KARL, a Knowledge-Aware Reinforcement Learning framework that dynamically modulates optimization signals at the entity level. Rather than applying uniform reward signals across all samples, KARL adjusts reinforcement strength according to the model’s estimated knowledge mastery of different entities, enabling differentiated optimization across heterogeneous entity types. This design is motivated by our observation that entities vary substantially in their prior knowledge availability within MLLMs, leading to uneven grounding behavior across entities. By explicitly coupling entity knowledge utilization with visual grounding optimization, our method aims to reduce the knowledge–grounding gap and enhance the model’s ability to leverage entity knowledge during grounding.

To facilitate systematic evaluation on KVG, we introduce KVG-Bench, a high-quality, expertlycurated benchmark encompassing 10 distinct domains with 1.3K manually curated test cases linked to 531 images and 882 entities. Our extensive experiments conducted across KVG and related tasks reveal two key findings: (1) Knowledge-guided reasoning training consistently improves KVG performance over direct fine-tuning and generic reasoning strategies, demonstrating the effectiveness of explicitly activating entity-level knowledge during grounding; (2) KARL achieves substantially stronger cross-domain generalization on unseen categories, particularly compared with standard

GRPO optimization, indicating that knowledge-aware reward scaling helps better generalize across entities with different knowledge levels.

In summary, our contributions are threefold:

- • We introduce Knowledge-Intensive Visual Grounding (KVG) task and KVG-Bench, a benchmark designed to evaluate models’ ability to leverage domain-specific entity knowledge for visual grounding. Our empirical observations suggest the presence of a knowledge–grounding gap in current MLLMs.
- • We propose a knowledge-guided reasoning training strategy that constructs CoT reasoning data to encourage models to explicitly activate and align entity-level knowledge with visual evidence during grounding, differing from recent reasoning-guided grounding approaches that primarily emphasize structured reasoning depth.
- • We present KARL, a Knowledge-Aware Reinforcement Learning framework that dynamically modulates optimization signals according to entity-level knowledge mastery rather than applying uniform reward schemes. This design promotes more balanced optimization across entities with heterogeneous knowledge levels and leads to improved generalization in knowledge-intensive grounding.

### 2 Related Work

- 2.1 Multimodal Large Language Models

Recent years have witnessed rapid advancements in MLLMs [31, 23, 22, 6, 46, 1], demonstrating strong performance in tasks such as visual grounding [5, 32, 47, 26, 20] and reasoning [35, 44, 10, 51]. Early MLLM-based grounding approaches such as Shikra [5], Groma [26], and LISA [20] adopt end-to-end architectures that align textual queries with image regions, enabling direct prediction of bounding boxes or segmentation masks. While effective, these methods primarily rely on learned visual-text alignment and typically perform grounding in a single-step manner without explicit reasoning.

Inspired by recent reinforcement learning approaches for enhancing reasoning in large language models [12], recent works such as UniVG-R1 [2], Perception-R1 [48], and Visual-RFT [24] adopt reinforcement learning to improve visual grounding performance by strengthening the model’s reasoning capabilities. However, these approaches mainly focus on strengthening generic reasoning capabilities for grounding, without explicitly encouraging the utilization of domain-specific knowledge during the reasoning process. In contrast, our work introduces knowledge-guided reasoning that encourages models to connect entity-level knowledge with visual evidence during grounding.

- 2.2 Visual Grounding Benchmarks

Visual Grounding aims to locate objects or regions within an image based on a textual query [16, 49, 33]. To enable more comprehensive evaluation of MLLMs’ grounding capabilities, recent studies have introduced several more challenging benchmarks [7, 8, 20] in addition to the widely used RefCOCO/+/g datasets [16, 28]. SK-VG [8] explores grounding with scene-level knowledge integration, while ReasonSeg [20] requires models to leverage world knowledge during object localization. In contrast, our proposed KVG task focuses on knowledge-intensive setting, requiring models to utilize fine-grained entity knowledge to distinguish visually similar entities while performing accurate grounding.

- 3 KVG-Bench

- 3.1 Task Definition

The task of knowledge-intensive visual grounding (KVG) aims to predict the bounding box B = fθ(XI,XT) of a target entity through the joint understanding of visual input XI and textual query XT. While sharing structural similarities with referring expression comprehension (REC), the KVG task significantly elevates the challenge beyond standard REC tasks. As exemplified in fig. 2a, the queries of KVG involve fine-grained entity specifications (e.g., “Boeing 747” and “White-lipped

###### Aircraft

Mollusca

300

CathedralMilan

Boeing 747

Airbus A380

| |
|---|

Out of Domain

Douglas DC-3

Big Ben

In Domain

[Figure 2]

[Figure 3]

Sukhoi Su-30

Gaura

225

Hibiscus

Petunia

AudiV8

Easternmole

150

Brushrabbit

BMWM2

75

Keydeer

Barisia

0

Air.

Car

Rep.

Bird

Food

Dog

Mol.

Mam.

Flwr.

Ldmk.

Okeniarosacea

Dicesnake

KVG-Bench

Nucellaostrina

Distribution of test cases

Cornsnake

Otalalactea

Tokaygecko Snowyowl

200

Thebapisana

Out of Domain

In Domain

IrishTerrier

150

Rockwren

Chihuahua

Incadove Woodduck

100

ShihTzu

Lima bean

Barn owl Cactus Wren

Diamond dove

Risotto

Cheesecake

50

e

ﬂ

Waf

0

Air.

Car

Rep.

Bird

Food

Dog

Mol.

Mam.

Flwr.

Ldmk.

Find and give the bounding box of Boeing 747

Find and give the bounding box of White-lipped snail

Aircraft Car Reptilia Bird Food

Distribution of entities

Dog Mollusca Mammal Flower Landmark

(a) Examples of KVG-Bench (b) Statistics of KVG-Bench

- Figure 2: (a) KVG-Bench images contain multiple subordinate-category entities (e.g., multiple Boeing models in the left image); (b) KVG-Bench exhibits high diversity across categories and entities.

snail”) rather than generic categories such as “aircraft” and “mollusk”. Moreover, each image typically contains multiple objects belonging to the same category as the target entity (e.g., several aircraft within a single image). This setup requires models to leverage fine-grained entity knowledge and perform careful visual comparison to identify the correct grounding target among visually similar entities.

#### 3.2 Benchmark Construction

KVG-Bench comprises 1,336 test instances spanning 10 categories with 882 distinct entities, as statistically visualized in fig. 2b. The construction includes two key parts: image collection and data annotation.

We designed a meticulous collection process to ensure the diversity and complexity of the images. First, we carefully selected 10 categories from the field of fine-grained visual recognition (FGVR) [41, 27, 19, 17, 29] that are suitable for visual grounding, excluding categories which are challenging for object localization such as “sports” and “scene”. Second, an entity list for each category was systematically developed through initial extraction of fine-grained labels from existing datasets, followed by comprehensive enrichment of entity names via ChatGPT-assisted expansion. We then retrieved web images using these entity names as search queries, enforcing strict quality criteria: each image must contain at least two entities from the same category with clear visual disparities.

The annotation process prioritized quality control. Five annotators independently annotated each image with bounding boxes and entity labels by cross-referencing contextual information (e.g. caption, webpage metadata) with authoritative sources (e.g., Wikipedia entries) to verify entity identities. To ensure consistency, all the annotations underwent independent re-evaluation by annotators who did not participate in the initial labeling, with conflicting cases cross-verified through multi-annotator reconciliation and persistently inconsistent instances eliminated to ensure annotation accuracy. By integrating comprehensive validation protocols and expert-aligned annotation workflows, KVG-Bench provides a challenging testbed for evaluating knowledge-intensive visual grounding in MLLMs.

To evaluate generalization, we divide the benchmark into seen and unseen splits based on whether the target entity’s category appears in the training data. The seen split contains 5 categories (Aircraft, Car, Reptilia, Bird, Food), while the unseen split includes 5 held-out categories (Dog, Mollusca, Mammal, Flower, Landmark). This design evaluates both in-domain grounding and out-of-domain generalization to novel fine-grained concepts.

#### 3.3 Human Evaluation

To assess the difficulty of KVG-Bench, we conducted human evaluations with 11 non-expert volunteers under two experimental settings: Closed-Book (no external resources) and Open-Book (allowing participants to consult Wikipedia once per instance to simulate expert-level knowledge integration). Participants were randomly assigned several categories, with each category being evaluated by at least five evaluators to mitigate knowledge bias. The evaluation results, as shown in table 1, reveal

###### Knowledge-Guided Grounding Data Construction

###### Knowledge-Aware Reinforcement Learning

Entity Knowledge Estimation Knowledge-Aware Optimization

Original Data Synthesized Data

MLLMs

Completions

Rewards

Advantages

[Figure 4]

What model is this aircraft? A. Airbus A330 B. Fokker 50 C. Saab 340 D. …

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

- r1

- r2

rn

- r3

- A1

- A2

An

- A3

- CoT1

- CoT2

- CoT3

[Figure 9]

Format Reward

CoT Rationales

Composite

Fokker 50

[Figure 10]

KA-IoU Reward

###### Stage-1 Model

Knowledge

[Figure 11]

IoU

Airbus A330

Visual Feature Analysis

Knowledge Mastery Level

reward scaling

… … …

Pos. Neg.

Comparison

Entity Knowledge Level

High

CoTn

Q: Where is the Fokker 50? A: (39, 78), (424, 305)

High Low

Fokker 50

Saab 340 …

Conclusion

Airbus A330

Pos. Neg.

Low

… …

- Figure 3: Overview of the proposed two-stage knowledge-aware training framework. (Left) Knowledge-guided grounding data construction and CoT-SFT training, which establish knowledgeguided reasoning for visual grounding. (Right) KARL, which estimates entity-level knowledge mastery and performs knowledge-aware reward scaling to balance positive and negative optimization signals under GRPO.

significant performance differences between settings. Notably, the Open-Book Setting demonstrated significant performance elevation (78.83% accuracy) compared to Closed-Book results (56.41%). This validates that KVG-Bench requires synergistic integration of expert-level knowledge and finegrained visual comparison, thereby highlighting the importance of integrating domain knowledge with visual grounding.

- 4 Method

While current MLLMs exhibit strong fine-grained entity knowledge and grounding capabilities, we observe a notable knowledge–grounding gap in the KVG task. In such scenarios, models may correctly identify fine-grained entities, yet fail to reliably ground the same entities in the image. This indicates that current training paradigms do not sufficiently align entity-level knowledge utilization with visual grounding optimization. To address this limitation, we propose a two-stage knowledge-aware training framework (fig. 3) that explicitly encourages effective knowledge utilization during grounding while maintaining training stability.

#### 4.1 Knowledge-Guided Grounding Data Construction

To bridge the knowledge–grounding gap, we construct training data that synthesize multi-entity grounding scenarios and augments them with knowledge-guided chain-of-thought (CoT) rationales. This design alleviates the scarcity of suitable grounding data while providing a reasoning-aware initialization that encourages models to leverage entity knowledge during grounding.

Multi-Entity Grounding Data Synthesis. Training images must satisfy two criteria: (1) fine-grained entity annotations with precise bounding boxes, and (2) multiple visually similar entities within the same image. To address the scarcity of such data, we leverage multiple FGVR datasets across five categories as the source of entity annotations and corresponding images. Since the original datasets lack bounding box annotations, we design a two-step annotation procedure. We first use

- Qwen2-VL-7B [39] to generate bounding boxes via entity-specific prompts, and then selectively refine predictions using SAM3 [4] to improve annotation quality. Manual inspection on a random subset of 200 samples indicates over 95% accuracy, where a prediction is considered correct if the generated box achieves an IoU greater than 0.5 with manually verified annotations. To further increase grounding difficulty and prevent shortcut learning, we synthesize composite images containing at least two entities from the same category using structured layouts, while preserving annotation consistency. This design encourages models to perform fine-grained discrimination rather than relying on coarse memorization. Importantly, the synthesized training data is constructed from external FGVR datasets and is fully disjoint from KVG-Bench, ensuring zero image overlap during training. Knowledge-Guided Reasoning Generation. The primary objective of the first-stage training is to enable the model to utilize its internal entity knowledge for visual grounding, thereby narrowing the knowledge–grounding gap observed in KVG. Rather than directly optimizing grounding accuracy, this stage encourages the model to incorporate entity knowledge into the reasoning process. To construct such reasoning data, we employ Qwen2-VL-72B [39] and Qwen3-VL-32B [1] as teacher models to generate knowledge-guided CoT rationales. The teachers are provided with the image,

the target entity label, and the ground-truth bounding box, together with a prompt that instructs the model to produce step-by-step reasoning before predicting the final grounding result. The prompt encourages the rationale to explicitly reference entity-related knowledge, including comparisons of fine-grained visual attributes, distinguishing characteristics among visually similar entities, and verification of these attributes against visual evidence in the image. Example prompts are briefly illustrated in the main text, while full implementation details are provided in the supplementary material.

These rationales illustrate how entity knowledge can be invoked and verified against visual evidence before producing the final grounding prediction. We then perform supervised fine-tuning using the constructed CoT rationales, which we refer to as CoT-SFT. The resulting stage-1 model demonstrates a preliminary ability to incorporate entity knowledge during reasoning, serving as a stable initialization for subsequent knowledge-aware reinforcement learning.

#### 4.2 Knowledge-Aware Reinforcement Learning

Following the first-stage training, we optimize the model through reinforcement learning to further enhance the model’s knowledge-intensive visual grounding capability, building upon its acquired knowledge-guided reasoning foundation. However, models typically exhibit heterogeneous knowledge mastery across different entities. Applying uniform reward signals to all samples may disproportionately reinforce entities that are already well mastered while providing insufficient learning pressure for under-mastered ones. Such imbalance can further widen entity-level performance disparities and lead to suboptimal optimization. Motivated by this observation, we propose Knowledge-Aware Reinforcement Learning (KARL), which modulates reward signals according to the model’s estimated entity-level knowledge mastery.

Entity Knowledge Estimation. Since grounding performance may depend on multiple factors beyond entity knowledge, we estimate knowledge mastery using a recognition task that isolates the model’s knowledge of fine-grained entities. Let E denote the set of entities in the training data. For each entity e ∈ E, we estimate the model’s knowledge mastery (i.e., its ability to correctly identify the entity) using single-entity images sampled from the source dataset. Specifically, we randomly sample

a set of images {x(je)}N

j=1 containing entity e and convert each image into a multiple-choice entity

e

recognition question. Given an image x(je), the model is required to select the correct fine-grained entity label among several distractors drawn from the same category. The knowledge mastery of

entity e is estimated using its average accuracy over sampled images. Based on this estimate, each entity is assigned to a discrete knowledge level via a mapping ke = ϕ(e), where ke ∈ {1,...,K} denotes the knowledge level of entity e. Implementation details are provided in the supplementary material.

Knowledge-Aware Optimization. To account for heterogeneous entity knowledge mastery, we modulate the reinforcement learning reward according to the knowledge level ke assigned to the entity e in each sample. Our reward design consists of a knowledge-aware Intersection over Union (IoU) reward together with a format reward. The KA-IoU reward evaluates the spatial alignment between the predicted bounding box B˜ and the ground-truth box B, while incorporating entity-level knowledge modulation. It is defined as

· IoU(B,B˜), if IoU(B,B˜) ≥ τ, s−k

s+k

(1)

RKA-IoU =

e

, otherwise,

e

where τ is a threshold for valid spatial alignment. The positive scaling factor s+k

and negative scaling factor s−k

e

are determined by the entity’s knowledge level. Specifically, entities with higher knowledge levels receive smaller positive scaling and stronger negative scaling (i.e., more negative rewards for incorrect predictions), while entities with lower knowledge levels receive larger positive scaling and weaker negative scaling. In addition to grounding accuracy, we include a lightweight format reward to enforce valid output structure. Specifically, regular expression-based pattern matching is used to ensure that reasoning traces are enclosed within “<think>” and “</think>” tags, and that the final grounded prediction is enclosed within “<answer>” and “</answer>” tags with a valid bounding box format. This auxiliary reward encourages structurally consistent outputs during optimization. The overall reward is defined as: R = RKA-IoU + Rformat.

e

We optimize the policy using Group Relative Policy Optimization (GRPO) [12]. For each grounding query q, a group of candidate outputs is sampled from the old policy, and policy updates are performed

- Table 1: KVG results of KARL and baseline models. KARL achieves the best overall performance among all models, with particularly strong improvements on unseen categories.

Seen Categories Unseen categories

Models

Avg. Air. Car Rep. Bird Food Avg. Dog Mol. Mam. Flwr. Ldmk. Avg.

Human Evaluation

Human 59.33 66.67 50.84 44.17 65.33 57.27 48.33 45.33 51.67 64.45 68.00 55.56 56.41 Human + search 81.33 85.56 68.00 74.17 86.67 78.03 78.89 74.00 74.17 84.44 86.67 79.63 78.83

Large-Scale MLLMs InternVL2-76B [6] 62.50 74.04 60.00 41.04 76.43 59.22 78.40 51.11 56.25 43.82 55.42 57.90 58.68

- Qwen2-VL-72B [39] 63.16 75.96 59.31 40.24 77.14 59.34 80.80 42.96 59.82 65.17 66.27 62.32 60.55

- Qwen3-VL-32B [1] 72.37 68.27 63.45 48.21 77.14 63.38 87.20 48.89 51.79 66.29 81.93 66.18 64.52 Specialist Grounding Models

YOLO-World [9] 41.45 28.85 8.28 14.74 30.71 23.36 50.40 2.22 24.11 1.12 3.61 17.83 21.11 G-DINO-1.6-Pro [37] 39.47 41.35 48.97 23.11 24.29 33.59 44.00 40.00 39.29 32.58 27.71 37.68 35.25 DINO-X [34] 43.42 49.04 42.76 28.29 41.43 38.89 62.40 35.56 48.21 31.46 49.40 45.77 41.69

###### 7B-Scale MLLMs

Shikra-7B [5] 20.39 25.96 15.17 16.33 28.57 20.33 51.20 19.26 25.00 16.85 22.89 27.94 23.43 CogVLM-G [40] 46.71 64.42 49.66 34.26 63.57 48.61 79.20 31.11 54.46 56.18 66.27 56.43 51.80 DeepSeek-VL2 [43] 51.32 60.57 53.10 29.08 63.57 47.98 62.40 35.56 50.89 44.94 39.76 47.06 47.60

- Qwen2-VL-7B [39] 48.03 74.04 51.30 33.07 65.71 50.38 76.00 33.33 54.46 57.30 59.04 55.33 52.40

- Qwen3-VL-8B [1] 61.84 68.27 55.17 40.24 75.00 56.94 73.60 44.44 42.86 65.17 66.27 57.54 57.19 Reasoning MLLMs

Perception-R1 [48] 23.03 32.69 14.48 12.35 37.14 21.84 41.60 11.85 26.79 16.85 28.92 25.18 23.20 Visual-RFT [24] 36.18 51.92 32.41 20.70 47.86 34.72 52.80 21.48 31.25 46.07 44.58 38.24 36.15 UniVG-R1 [2] 57.24 74.04 52.41 35.06 78.57 55.30 73.60 40.00 55.36 60.67 67.46 58.46 56.59

###### KARL 71.05 83.65 70.34 45.02 87.86 67.30 85.60 50.37 69.64 80.90 67.47 70.04 68.41

using normalized group-relative advantages with Kullback–Leibler (KL) regularization.

Data Filtering. To improve training efficiency, we filter data using the Stage-1 CoT-SFT model before Stage-2 reinforcement learning. For each training instance, we perform REC inference 4 times with stochastic sampling. Instances with 4/4 correct predictions are considered trivial and removed, while those with 0/4 correct predictions are regarded as overly difficult and also discarded. We retain only samples with mixed outcomes (1–3 correct predictions), resulting in approximately 2K instances used for Stage-2 reinforcement learning.

### 5 Experiment

#### 5.1 Implementation Details

Datasets. We conducted experiments based on several FGVR datasets including FGVC-Aircraft [27], Stanford-Cars [19], iNaturalist2017 [38], and Food101 [3]. For the iNaturalist2017 dataset, we use the categories of Reptilia and Aves as the training sources for constructing grounding data. Following the pipeline described in Sec. 4.1, we construct 25K training samples for Stage-1 and further select 2K instances for Stage-2 reinforcement learning.

Baseline Models. We build KARL upon Qwen3-VL-8B-Instruct [1] (hereafter Qwen3-VL-8B) due to its strong visual grounding capability and rich knowledge. For fair comparison, we evaluate several MLLMs with strong grounding ability across different model scales. For large-scale models, we used InternVL2-Llama3-76B [6], Qwen2VL-72B [39], and Qwen3VL-32B-Instruct [1] (hereafter

- Qwen3-VL-32B). For 7B-scale models, we used Shikra [5], CogVLM-Grounding [40], DeepSeekVL-2 [43], Qwen2-VL-7B [39] and Qwen3-VL-8B [1]. Additionally, we conducted comparisons with three specialist models: YOLO-World [9], Ground-ingDINO-1.6-Pro [37] and DINO-X [34]. Moreover, We also compare with recent reasoning-guided grounding MLLMs: Perception-R1 [48], Visual-RFT [24] and UniVG-R1 [2]. For Qwen3-VL-8B, we further evaluate several variants under different reasoning and training settings. Evaluation Settings. We evaluate all methods on KVG-Bench using accuracy as the evaluation metric, following standard practice in referring expression comprehension (REC) [47, 5, 39]. Specifically, given a predicted bounding box B˜ and ground-truth bounding box B, we compute their IoU, and a prediction is considered correct if IoU ≥ 0.5. For multi-category evaluation, the overall accuracy is computed as a weighted average according to the number of instances in each category. Training Details. All experiments are conducted on 8 NVIDIA A800 80GB GPUs. In Stage 1, we

- Table 2: Performance Comparison of KARL and Qwen3VL-8B on general multimodal benchmarks. KARL maintains performance comparable to the base model, indicating that general multimodal capabilities are preserved.

Models MMStar RealWorldQA AI2D CV-Bench

Qwen3VL-8B3 63.5 69.4 83.6 86.1 KARL 64.1 69.9 83.7 86.7

- Table 3: Ablation study on the effectiveness of the proposed two-stage knowledge-aware training framework. The results show that the two-stage design and the KARL optimization consistently improve both seen and unseen performance.

#### Stage Method Seen Unseen Overall

- 1

SFT 61.49 62.87 62.05 CoT-SFT 67.05 66.73 66.92

- 2

SFT 59.22 61.76 60.25 CoT-SFT 67.05 65.07 66.24 GRPO 67.29 66.36 66.92 KARL 67.30 70.04 68.41

optimize the model using Adam [18] with a learning rate of 1 × 10−6 and β1 = 0.9, β2 = 0.999. In Stage 2, we adopt AdamW [25] with a learning rate of 5×10−7 and the same momentum parameters. The accumulated batch size is set to 16 in stage 1 and 8 in stage 2. For GRPO training, the maximum completion length is set to 1500 tokens, and four samples are generated for each input query. In KARL, entities are grouped into five knowledge levels according to the estimated knowledge mastery. The positive reward scaling factors are set to 0.65, 0.85, 1.0, 1.2, and 1.4 (capped at 1.0), while the negative reward scaling factors are set to −0.6, −0.5, −0.4, −0.2, and 0.0.

#### 5.2 Main Results

Table 1 presents the performance of KARL and baseline models on KVG-Bench. Overall, KARL consistently outperforms all baselines across both seen and unseen categories, demonstrating the effectiveness of knowledge-aware reinforcement learning for knowledge-intensive visual grounding.

On seen categories, KARL reaches 67.30%, outperforming strong 7B baselines such as Qwen3-VL8B (56.94%) and even surpassing several large-scale MLLMs (e.g., 63.38% for Qwen3-VL-32B). This result suggests that explicitly encouraging models to utilize fine-grained entity knowledge during grounding can yield improvements beyond those obtained from model scaling alone. More importantly, on unseen categories, which evaluate cross-domain generalization, KARL achieves the best performance of 70.04%. It consistently surpasses both large-scale MLLMs and recent reasoning-guided grounding models such as UniVG-R1 (58.46%) and Visual-RFT (38.24%). The improvements are observed across diverse semantic domains, including Dog (85.60%), Flower (80.90%), and Mammal (69.64%), indicating that the model can effectively transfer entity-level knowledge to previously unseen categories. Notably, generic reasoning strategies do not consistently improve performance on KVG. For example, adding zero-shot CoT prompting to the base model yields performance comparable to the original model (see fig. 1b). Similarly, recent reasoning-guided grounding methods such as Visual-RFT and UniVG-R1, both built upon Qwen2-VL-7B, show only limited improvements over their base model and do not consistently outperform it on KVG-Bench, with performance in some cases even lower. These observations suggest that generic reasoning strategies are insufficient for knowledge-intensive grounding. In contrast, by explicitly encouraging models to activate entity-level knowledge during grounding and adapting reinforcement learning signals according to knowledge mastery, KARL achieves the best overall performance of 68.41%, establishing a new state of the art on KVG-Bench.

General Capability. To comprehensively assess the model’s general capabilities across diverse multimodal scenarios, we conducted evaluations on established multimodal benchmarks. As shown

3reproduced using lmms-eval [50, 21]

- Table 4: Ablation study on different reward scaling strategies. Removing either positive or negative reward scaling degrades performance, highlighting the importance of knowledge-aware reward signals.

#### Strategy Seen Unseen Overall

KARL 67.30 70.04 68.41 w/o Neg. 66.67 66.91 66.77 w/o Pos. 67.17 66.36 66.84 w/o Pos. & Neg. 67.29 66.36 66.92

in table 2, KARL maintains performance comparable to the base model, demonstrating preserved general capabilities without degradation.

#### 5.3 Ablation Study

To evaluate the effectiveness of the proposed two-stage knowledge-aware training framework (KARL) and the contributions of its individual components, we conducted a series of ablation experiments. The analysis focuses on the necessity of the two-stage framework and the impact of knowledge-aware reward shaping strategies.

Effectiveness of Training Strategies Across Stages. As shown in table 3, we compare different training strategies across the two stages. Replacing standard SFT with CoT-SFT in Stage 1 improves performance on seen categories from 61.49% to 67.05%, demonstrating that knowledge-guided reasoning supervision provides a stronger initialization for knowledge-intensive grounding.

In Stage 2, we further optimize the Stage-1 models using different training strategies on the same filtered training data. Specifically, Stage-2 SFT continues training from the Stage-1 SFT model, while all other methods (CoT-SFT, GRPO, and KARL) are initialized from the same Stage-1 CoT-SFT checkpoint. Comparing Stage-2 results, both GRPO and KARL further improve performance over CoT-SFT across seen and unseen categories, highlighting the benefit of reinforcement learning for refining grounding predictions. More importantly, compared with standard GRPO, KARL achieves comparable performance on seen categories (67.30% vs. 67.29%) while delivering a substantial improvement on unseen categories (70.04% vs. 66.36%). This result suggests that knowledge-aware reward scaling helps rebalance optimization across entities with different knowledge levels, leading to improved generalization on previously unseen categories.

Analysis of Reward Scaling Strategies. To further analyze the contributions of KARL, we examine the effects of different reward scaling components in table 4. Specifically, "w/o Neg." applies only positive reward scaling; "w/o Pos." applies only negative penalty scaling; "w/o Pos. & Neg. " is vanilla GRPO with fixed positive and negative rewards. Removing the negative scaling (w/o Neg.) reveals the critical role of penalizing incorrect answers: performance drops on both seen (67.30%→66.67%) and unseen (70.04%→66.91%) categories, with a more pronounced decline on unseen ones, indicating that knowledge-guided penalties help the model avoid spurious correlations and enhance generalization. In contrast, removing positive scaling (w/o Pos.) highlights the importance of reinforcing correct predictions. When positive scaling is removed, unseen accuracy decreases substantially (70.04%→66.36%), indicating that reinforcing correct predictions is important for improving generalization to unseen categories.

Finally, when comparing KARL with the vanilla GRPO baseline (w/o Pos. & Neg.), unseen performance suffers a large degradation (70.04%→66.36%), confirming that the combination of positive and negative knowledge scaling is essential for achieving strong generalization beyond memorization. Overall, these results validate the effectiveness of the proposed knowledge-aware training framework and reward scaling strategy in KARL, which improves grounding performance and generalization to unseen categories.

#### 5.4 Analysis

To gain deeper insights into our model’s capabilities, we conducted the following analysis, examining how knowledge-aware reward scaling affects performance across different knowledge conditions and how it influences the model’s reasoning behavior in practice:

High Knowledge Low Knowledge

75%

Δ = 0.24%

71.16% 71.40%

Δ = 2.10%

67.00% 64.90%

62.5%

60.93%

55.30%

50%

Qwen3VL-8B GRPO KARL

- Figure 4: Knowledge evaluation results. KARL shows larger gains on low-knowledge entities, demonstrating the effectiveness of knowledge-aware reward scaling on low knowledge.

( Visual Analysis & Comparison)

The key distinguishing feature is the tail shape: the Mississippi kite has a slightly forked tail, while the White-tailed kite has a deeply forked, almost swallow-like tail. Additionally, the coloration white head vs. grayish-brown head is a major differentiator. …

The Bird on the Left: - Wings: The bird on the left has broad,

slightly rounded wings with a noticeable dark trailing edge. The wingtips are not forked or deeply split. Tail: The tail is relatively short and appears to be slightly forked or notched, but not deeply so. - Body: The body is compact, with a grayish-brown upper body and lighter underparts. … The Bird on the right: - Tail:

The tail is deeply forked, almost like a swallow, which is a key distinguishing feature. - Body: The body is

slender and streamlined, with a white head and underparts, and dark gray or black upperparts. The contrast between the white head and dark back is very pronounced.

To identify the Mississippi kite in the image, we must carefully analyze the visual features

of both …

[Figure 12]

Find and give the bounding box of Mississippi kite

| |
|---|

| |
|---|

KARL:

|Qwen3-VL-8B: (619.0,225.0),(963.0,719.0)<br><br>[Figure 13]|
|---|

( Planning)

( Conlusion)

Final Answer: (77.0,348.0),(289.0,630.0)

- Figure 5: Case study comparing KARL and Qwen3-VL-8B. KARL correctly grounds the target entity by leveraging knowledge-guided reasoning, while the baseline model produces an incorrect prediction.

Knowledge-Level Analysis. To simplify the analysis, we divide entities into two knowledge levels based on the estimated knowledge mastery: the top three categories are treated as high knowledge, while the remaining two are treated as low knowledge. As shown in fig. 4, both GRPO and KARL bring substantial improvements over the base model across different knowledge levels, indicating that reinforcement learning generally improves grounding performance regardless of the knowledge level. However, on low-knowledge entities, KARL achieves a much larger gain (+2.10%) than GRPO, which is nearly nine times larger than the improvement observed in the high-knowledge setting. This result suggests that knowledge-aware reward scaling in KARL enables more effective learning under low knowledge conditions.

Qualitative Results. As illustrated in fig. 5, we conducted a comparative case analysis between KARL and Qwen3-VL-8B. In this case, KARL follows a structured reasoning pipeline—initiating with a planning step, then performing a detailed visual analysis and comparison of the entities, and finally drawing a conclusion based on the most discriminative features (e.g., slightly forked vs. deeply forked), thereby arriving at the accurate identification. The visual evidence demonstrates our model’s capability to generate accurate answers through a knowledge-aware reasoning process that systematically integrates domain specific knowledge with visual observations, in contrast to the baseline model’s tendency to produce incorrect responses from superficial pattern recognition.

- 6 Conclusion

In this work, we investigate knowledge-intensive visual grounding and identify a gap between entity knowledge and grounding performance in current MLLMs. To address this limitation, we propose a knowledge-aware training paradigm that combines knowledge-guided reasoning supervision with knowledge-aware reinforcement learning. The resulting framework enables models to better

incorporate entity-level knowledge into grounding decisions. Extensive experiments on KVG-Bench demonstrate that KARL consistently outperforms strong baseline models and achieves the best overall performance on the benchmark, while maintaining competitive performance on general multimodal tasks. Further analyses show that knowledge-guided reasoning provides a strong initialization for KVG, and that knowledge-aware reward scaling improves cross-domain generalization across entities with different knowledge levels. These results highlight the importance of explicitly integrating entity knowledge into the grounding process and suggest a promising direction for improving knowledgeintensive perception in MLLMs.

### References

- [1] Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631 (2025)
- [2] Bai, S., Li, M., Liu, Y., Tang, J., Zhang, H., Sun, L., Chu, X., Tang, Y.: Univg-r1: Reasoning guided universal visual grounding with reinforcement learning. arXiv preprint arXiv:2505.14231

(2025)

- [3] Bossard, L., Guillaumin, M., Van Gool, L.: Food-101–mining discriminative components with random forests. In: Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part VI 13. pp. 446–461. Springer (2014)
- [4] Carion, N., Gustafson, L., Hu, Y.T., Debnath, S., Hu, R., Suris, D., Ryali, C., Alwala, K.V., Khedr, H., Huang, A., Lei, J., Ma, T., Guo, B., Kalla, A., Marks, M., Greer, J., Wang, M., Sun, P., Rädle, R., Afouras, T., Mavroudi, E., Xu, K., Wu, T.H., Zhou, Y., Momeni, L., Hazra, R., Ding, S., Vaze, S., Porcher, F., Li, F., Li, S., Kamath, A., Cheng, H.K., Dollár, P., Ravi, N., Saenko, K., Zhang, P., Feichtenhofer, C.: Sam 3: Segment anything with concepts (2025)
- [5] Chen, K., Zhang, Z., Zeng, W., Zhang, R., Zhu, F., Zhao, R.: Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195 (2023)
- [6] Chen, Z., Wang, W., Cao, Y., Liu, Y., Gao, Z., Cui, E., Zhu, J., Ye, S., Tian, H., Liu, Z., et al.: Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271 (2024)
- [7] Chen, Z., Wang, P., Ma, L., Wong, K.Y.K., Wu, Q.: Cops-ref: A new dataset and task on compositional referring expression comprehension. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10086–10095 (2020)
- [8] Chen, Z., Zhang, R., Song, Y., Wan, X., Li, G.: Advancing visual grounding with scene knowledge: Benchmark and method. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 15039–15049 (2023)
- [9] Cheng, T., Song, L., Ge, Y., Liu, W., Wang, X., Shan, Y.: Yolo-world: Real-time openvocabulary object detection. In: Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR) (2024)
- [10] Deng, L., Liu, Y., Li, B., Luo, D., Wu, L., Zhang, C., Lyu, P., Zhang, Z., Zhang, G., Ding, E., et al.: R-cot: Reverse chain-of-thought problem generation for geometric reasoning in large multimodal models. arXiv preprint arXiv:2410.17885 (2024)
- [11] Goldstein, E.B.: Cognitive psychology: Connecting mind, research and everyday experience. Wadsworth Publishing (2007)
- [12] Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al.: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948 (2025)
- [13] He, H., Geng, Z., Peng, Y.: Fine-r1: Make multi-modal LLMs excel in fine-grained visual recognition by chain-of-thought reasoning. In: The Fourteenth International Conference on Learning Representations (2026)
- [14] He, H., Li, G., Geng, Z., Xu, J., Peng, Y.: Analyzing and boosting the power of fine-grained visual recognition for multi-modal large language models. In: The Thirteenth International Conference on Learning Representations (2025)
- [15] Hegdé, J.: Time course of visual perception: coarse-to-fine processing and beyond. Progress in neurobiology 84(4), 405–439 (2008)

- [16] Kazemzadeh, S., Ordonez, V., Matten, M., Berg, T.: Referitgame: Referring to objects in photographs of natural scenes. In: Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). pp. 787–798 (2014)
- [17] Khosla, A., Jayadevaprakash, N., Yao, B., Fei-Fei, L.: Novel dataset for fine-grained image categorization. In: First Workshop on Fine-Grained Visual Categorization, IEEE Conference on Computer Vision and Pattern Recognition. Colorado Springs, CO (June 2011)
- [18] Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. In: Bengio, Y., LeCun, Y. (eds.) 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings (2015)
- [19] Krause, J., Stark, M., Deng, J., Fei-Fei, L.: 3d object representations for fine-grained categorization. In: Proceedings of the IEEE international conference on computer vision workshops. pp. 554–561 (2013)
- [20] Lai, X., Tian, Z., Chen, Y., Li, Y., Yuan, Y., Liu, S., Jia, J.: Lisa: Reasoning segmentation via large language model. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9579–9589 (2024)
- [21] Li, B., Zhang, P., Zhang, K., Pu, F., Du, X., Dong, Y., Liu, H., Zhang, Y., Zhang, G., Li, C., Liu, Z.: Lmms-eval: Accelerating the development of large multimoal models (March 2024)
- [22] Liu, H., Li, C., Li, Y., Lee, Y.J.: Improved baselines with visual instruction tuning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 26296–26306 (June 2024)
- [23] Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. Advances in neural information processing systems 36, 34892–34916 (2023)
- [24] Liu, Z., Sun, Z., Zang, Y., Dong, X., Cao, Y., Duan, H., Lin, D., Wang, J.: Visual-rft: Visual reinforcement fine-tuning. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 2034–2044 (2025)
- [25] Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: International Conference on Learning Representations (2019)
- [26] Ma, C., Jiang, Y., Wu, J., Yuan, Z., Qi, X.: Groma: Localized visual tokenization for grounding multimodal large language models. In: European Conference on Computer Vision. pp. 417–435. Springer (2024)
- [27] Maji, S., Rahtu, E., Kannala, J., Blaschko, M., Vedaldi, A.: Fine-grained visual classification of aircraft. arXiv preprint arXiv:1306.5151 (2013)
- [28] Mao, J., Huang, J., Toshev, A., Camburu, O., Yuille, A.L., Murphy, K.: Generation and comprehension of unambiguous object descriptions. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 11–20 (2016)
- [29] Nilsback, M.E., Zisserman, A.: Automated flower classification over a large number of classes. In: 2008 Sixth Indian conference on computer vision, graphics & image processing. pp. 722–729. IEEE (2008)
- [30] OpenAI: GPT4 technical report. arXiv preprint arXiv:2303.08774 (2023)
- [31] OpenAI: Hello gpt-4 (2024)
- [32] Peng, Z., Wang, W., Dong, L., Hao, Y., Huang, S., Ma, S., Wei, F.: Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824 (2023)
- [33] Qiao, Y., Deng, C., Wu, Q.: Referring expression comprehension: A survey of methods and datasets. IEEE Transactions on Multimedia 23, 4426–4440 (2020)
- [34] Ren, T., Chen, Y., Jiang, Q., Zeng, Z., Xiong, Y., Liu, W., Ma, Z., Shen, J., Gao, Y., Jiang, X., Chen, X., Song, Z., Zhang, Y., Huang, H., Gao, H., Liu, S., Zhang, H., Li, F., Yu, K., Zhang, L.: Dino-x: A unified vision model for open-world object detection and understanding (2024)
- [35] Shi, W., Hu, Z., Bin, Y., Liu, J., Yang, Y., Ng, S.K., Bing, L., Lee, R.K.W.: Math-LLaVA: Bootstrapping mathematical reasoning for multimodal large language models. In: Al-Onaizan, Y., Bansal, M., Chen, Y.N. (eds.) Findings of the Association for Computational Linguistics: EMNLP 2024. pp. 4663–4680. Association for Computational Linguistics, Miami, Florida, USA (Nov 2024)

- [36] Tanaka, J.W., Taylor, M.: Object categories and expertise: Is the basic level in the eye of the beholder? Cognitive psychology 23(3), 457–482 (1991)
- [37] Tianhe, R., Hongjie, H., Xiaoke, J.: Grounding dino 1.6 (2024)
- [38] Van Horn, G., Mac Aodha, O., Song, Y., Cui, Y., Sun, C., Shepard, A., Adam, H., Perona, P., Belongie, S.: The inaturalist species classification and detection dataset. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 8769–8778 (2018)
- [39] Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., et al.: Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191 (2024)
- [40] Wang, W., Lv, Q., Yu, W., Hong, W., Qi, J., Wang, Y., Ji, J., Yang, Z., Zhao, L., XiXuan, S., Xu, J., Chen, K., Xu, B., Li, J., Dong, Y., Ding, M., Tang, J.: CogVLM: Visual expert for pretrained language models. In: The Thirty-eighth Annual Conference on Neural Information Processing Systems (2024)
- [41] Welinder, P., Branson, S., Mita, T., Wah, C., Schroff, F., Belongie, S., Perona, P.: Caltech-ucsd birds 200 (2010)
- [42] Weyand, T., Araujo, A., Cao, B., Sim, J.: Google Landmarks Dataset v2 - A Large-Scale Benchmark for Instance-Level Recognition and Retrieval. In: Proc. CVPR (2020)
- [43] Wu, Z., Chen, X., Pan, Z., Liu, X., Liu, W., Dai, D., Gao, H., Ma, Y., Wu, C., Wang, B., et al.: Deepseek-vl2: Mixture-of-experts vision-language models for advanced multimodal understanding. arXiv preprint arXiv:2412.10302 (2024)
- [44] Xu, G., Jin, P., Hao, L., Song, Y., Sun, L., Yuan, L.: Llava-o1: Let vision language models reason step-by-step. arXiv preprint arXiv:2411.10440 (2024)
- [45] Yao, H., Huang, J., Wu, W., Zhang, J., Wang, Y., Liu, S., Wang, Y., Song, Y., Feng, H., Shen, L., et al.: Mulberry: Empowering mllm with o1-like reasoning and reflection via collective monte carlo tree search. arXiv preprint arXiv:2412.18319 (2024)
- [46] Yao, Y., Yu, T., Zhang, A., Wang, C., Cui, J., Zhu, H., Cai, T., Li, H., Zhao, W., He, Z., et al.: Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800 (2024)
- [47] You, H., Zhang, H., Gan, Z., Du, X., Zhang, B., Wang, Z., Cao, L., Chang, S.F., Yang, Y.: Ferret: Refer and ground anything anywhere at any granularity. In: The Twelfth International Conference on Learning Representations (2024)
- [48] Yu, E., Lin, K., Zhao, L., jisheng yin, Wei, Y., Peng, Y., Wei, H., Sun, J., Han, C., Ge, Z., Zhang, X., Jiang, D., Wang, J., Tao, W.: Perception-r1: Pioneering perception policy with reinforcement learning. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems

(2025)

- [49] Yu, L., Poirson, P., Yang, S., Berg, A.C., Berg, T.L.: Modeling context in referring expressions. In: Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14. pp. 69–85. Springer (2016)
- [50] Zhang, K., Li, B., Zhang, P., Pu, F., Cahyono, J.A., Hu, K., Liu, S., Zhang, Y., Yang, J., Li, C., Liu, Z.: Lmms-eval: Reality check on the evaluation of large multimodal models (2024)
- [51] Zhang, R., Jiang, D., Zhang, Y., Lin, H., Guo, Z., Qiu, P., Zhou, A., Lu, P., Chang, K.W., Qiao, Y., et al.: Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In: European Conference on Computer Vision. pp. 169–186 (2024)

- A KVG-Bench This section details the data collection and annotation protocol for KVG-Bench.

- A.1 Data Collection

Category Selection Ten categories were strategically curated from established fine-grained visual recognition (FVGR) datasets including FGVC-Aircraft [27], Stanford-Cars [19], iNaturalist2017 [38], Food101 [3], Stanford-Dogs [17], Flower-102 [29], and Google-Landmarks-v2 [42]. Categories requiring ambiguous spatial localization (e.g., "sports" and "scenes") were systematically excluded.

Entity List Curation We built the list of detailed entity names through a step-by-step process. First, we started with existing names from aforementioned FGVR datasets to make sure they fit the right categories. Then, we used ChatGPT [30] to collect more entities in these categories by querying with category name and example entities. Finally, we checked all these entity names against Wikipedia to confirm their accuracy and avoid confusing or incorrect terms.

Web Image Retrieval The image collection process employed diversified search strategies, systematically generating query variations (e.g., “X versus Y”, “X compared to Y”, “differences between X and Y”) to retrieve visually discriminative instances across search engines. The image collection process focused on two core principles: diversity and challenge. To ensure diversity, images were collected to encompass a variety of distinct entities within each category, achieved through systematically varied search query combinations. This approach guaranteed a wide range of entity interactions and visual scenarios. For challenge, we specifically selected images containing multiple entities from the same category (requires fine-grained visual discrimination) or exhibited high visual similarity with subtle distinguishing features.

- A.2 Annotation

The annotation process prioritized quality control. Five annotators manually annotated each image with bounding boxes and entity labels by cross-referencing contextual information (e.g. caption, webpage metadata) with authoritative sources (e.g., Wikipedia entries) to verify entity identities. To ensure consistency, the annotations underwent independent re-evaluation by annotators who did not participate in the initial labeling, with conflicting cases cross-verified through multi-annotator reconciliation and persistently inconsistent instances eliminated to ensure annotation accuracy.

- B Method This section elaborates on the implementation details of the proposed method.

#### B.1 Knowledge-Guided Grounding Data Construction

Multi-Entity Grounding Data Synthesis Our data processing pipeline comprises three core stages: categorization, fine-grained annotation, and image composition.

Category Organization. We first organize images from external fine-grained visual recognition datasets into category-specific subsets according to their semantic domains. Specifically, we use FGVC-Aircraft [27], Stanford-Cars [19], and Food101 [3] as the data sources for the aircraft, car, and food categories, respectively. For iNaturalist2017 [38], which covers multiple biological groups, we further select the Reptilia and Aves subsets to construct data for the reptilia and bird categories. These five categories serve as the training sources for the grounding data used in our two-stage optimization pipeline.

Box Annotation. Since the original datasets do not provide bounding box annotations, we adopt a two-step annotation procedure. We first use Qwen2-VL-7B [39] to generate bounding boxes with entity-specific prompts. Given an image and its verified entity label E, we query the model with “Find and give the bounding box of {E}”, and use the returned coordinates as pseudo annotations. We then manually inspect the generated boxes on a random subset and identify categories with relatively lower annotation quality. Based on this inspection, we further apply SAM3 [4] to selectively refine Qwengenerated predictions for those categories, rather than uniformly processing all data. In practice,

CoT Generation Prompt

<|vision_start|>Image.jpg<|vision_end|> This image shows [entity1] ([bbox1]), [entity2] ([bbox2]), ···, and [entityn] ([bboxn]). The bounding box of [target entity] is [target bbox]. Give the reasoning process that would identify it based on the image and your knowledge Note that you MUST pay attention to the differences from other objects of the same type in this image and make a detailed comparison between them to find evidence that distinguishes this object from the others Note that you MUST first analyze the visual features that help you make a judgment, and then compare the objects Note that when an object is “[Unknown]”, you can still make a comparison based on its visual features without knowing its name

- Table 5: Comparison of SFT performance using CoT data generated by different large-scale visionlanguage models.

Models Air. Car Rep. Bird Food

- CoT-SFT (Qwen2VL-72B) 75.00 76.92 69.66 41.04 72.86

- CoT-SFT (Qwen3VL-32B) 72.37 75.96 61.38 44.62 78.57

this refinement is mainly used for the food category, where the initial predictions are comparatively less accurate. Manual inspection on a random subset of 200 samples indicates over 95% annotation accuracy, where a prediction is considered correct if its generated box achieves an IoU greater than 0.5 with manually verified annotations.

Multi-Entity Image Composition. Based on each category-specific dataset Dc, we synthesize multi-entity grounding examples by iteratively sampling instances without replacement. At each iteration, we randomly sample k images, where k is drawn from a predefined discrete distribution over [2,6]. To avoid trivial shortcuts, all sampled instances are required to correspond to different finegrained entities. The selected images are then composed into a single scene using one of four layout strategies: horizontal concatenation, vertical concatenation, grid arrangement, or random placement. During composition, we apply the corresponding geometric transformations to remap each original bounding box to its new location in the synthesized image, thereby preserving annotation consistency. The resulting composite dataset Dc′ contains multi-entity images, transformed bounding boxes, and entity labels, creating training examples in which correct grounding requires distinguishing among semantically related fine-grained entities rather than relying on coarse category cues. Finally, we construct 25K samples for Stage-1 training and further select 2K instances for Stage-2 reinforcement learning.

Knowledge-Guided Reasoning Generation To construct high-quality Chain-of-Thought (CoT) training data, we generate reasoning annotations by harnessing high-capacity open-source Multimodal Large Language Models (MLLMs). Specifically, the model takes the image, ground-truth entity annotations (names and bounding boxes), and a CoT generation instruction as input, producing detailed reasoning processes for training. fig. 6 illustrates a representative training instance.

To determine which model produces more effective CoT supervision, we further analyze the downstream SFT performance using CoT data generated by different models. As shown in table 5, CoT data produced by Qwen2-VL-72B-Instruct [39] leads to stronger SFT performance on the Aircraft, Car, and Reptilia categories, while Qwen3-VL-32B-Instruct [1] performs better on Bird and Food. This observation suggests that different models possess complementary strengths in capturing category-specific visual knowledge during reasoning generation. Based on this finding, we adopt a category-aware CoT data construction strategy. Specifically, we use Qwen2-VL-72B-Instruct to generate CoT annotations for Aircraft, Car, and Reptilia, and Qwen3-VL-32B-Instruct for Bird and Food, where each model demonstrates superior performance. This strategy enables us to leverage the complementary reasoning capabilities of different large-scale models, thereby improving the overall quality and diversity of the generated CoT training data.

|[Figure 14]<br><br>[Figure 15]<br><br>|
|---|

Qwen2-VL-72B CoT

( Planning)

To identify the Red crossbill in the image, we need to carefully analyze its visual features and compare them with the other birds present.

( Visual Analysis)

1. Red crossbill (179,154),(297,339): - Coloration: The bird has a distinctive reddish-brown coloration with darker wings. - Beak Shape: The beak appears to be slightly crossed at the tips, which is characteristic of crossbills……2. Tropical mockingbird (150,631),(441,887): Coloration: This bird has a grayish body with a white belly and black wings……3. Great-tailed grackle (632,68),(815,504): - Coloration: This bird is entirely black with a glossy sheen……4. Semipalmated plover (605,623),(812,910): - Coloration: This bird has a brown and white plumage with a distinct black ring around the neck……

( Comparison)

Comparison: - The Red crossbill has a unique reddish-brown coloration and a crossed beak……

( Conclusion)

Based on these comparisons, the bird in the top left corner of the image, with the bounding box coordinates (179,154),(297,339)

Find and give the bounding box of Red crossbill

Figure 6: Example of Chain-of-Thought data generated by Qwen2-VL-72B.

#### B.2 Knowledge-Aware Reinforcement Learning

Entity Knowledge Estimation To quantify the degree of visual knowledge possessed by the model, we construct a controlled visual recognition probe to estimate a knowledge mastery score for each entity. For every entity, we randomly sample up to 50 images containing the target entity and ask Qwen3-VL-8B-Instruct [1] to identify its category. Each question is formulated as a four-choice multiple-choice task, where the candidate options are semantically similar categories drawn from the same super-category. The model’s average accuracy (ACCavg) across the evaluated images is used as the knowledge mastery score of that entity.

We categorize entities into five knowledge levels based on ACCavg: completely correct (ACCavg ≥ 0.7), mostly correct (0.45 ≤ ACCavg < 0.7), partially correct (0.3 ≤ ACCavg < 0.45), mostly incorrect (0.1 ≤ ACCavg < 0.3), and completely incorrect (ACCavg < 0.1). These thresholds are designed with respect to the four-choice classification setting, where random guessing yields an expected accuracy of 25%. In particular, the partially correct interval (0.30–0.45) corresponds to performance slightly above the chance level, while the mostly incorrect and completely incorrect ranges capture entities for which the model performs at or below chance level. Conversely, the mostly correct and completely correct ranges represent entities where the model demonstrates increasingly reliable recognition ability.

To obtain an unbiased estimation of visual knowledge distribution, we evaluate all entities in the dataset. The resulting statistics reveals a highly skewed knowledge landscape, Specifically, 46.79% of entities are categorized as mostly incorrect and 19.27% as completely incorrect, meaning that 66.06% of entities cannot be reliably recognized by the model. An additional 21.36% fall into the partially correct category, indicating unstable knowledge representations. In contrast, only a small fraction of entities are classified as mostly correct (11.40%) or completely correct (1.18%), suggesting that strong visual knowledge is concentrated on a very limited subset of concepts. These results indicate that visual knowledge in current MLLMs is highly uneven and sparse, motivating the need for knowledge-aware training strategies to better align entity-level knowledge with visual grounding.

Knowledge-Aware Optimization Table 6 presents the parameter search for the rewardscaling strategy used in our knowledge-aware reinforcement learning stage. Specifically, Scaling V1 adopts positive weights: [0.6,0.8,1.0,1.2,1.4] with relatively stronger negative penalties: [−0.8,−0.6,−0.4,−0.2,0.0], while Scaling V2 uses slightly different positive weights: [0.65,0.85,1.0,1.15,1.25] combined with milder penalties: [−0.6,−0.5,−0.2,−0.1,0.0]. The final KARL scaling integrates these two schedules into a unified configuration, resulting in positive weights: [0.65,0.85,1.0,1.2,1.4] and negative penalties: [−0.6,−0.5,−0.4,−0.2,0.0].

Combining the low-knowledge design from V1 with the high-knowledge design from V2, the final KARL scaling achieves the highest average performance (68.41%) among all variants, demonstrating that integrating the two regimes effectively balances learning signals across entity types. This unified schedule is therefore selected as our reward-scaling strategy.

- Table 6: Reward scaling parameter search for KARL. The proposed KARL scaling strategy achieves the best overall average performance

Strategy Avg Scaling (KARL) 68.41 Scaling V1 67.73 Scaling V2 67.29

Table 7: Performance comparison on RefCOCO, RefCOCO+, and RefCOCOg benchmarks.

RefCOCO RefCOCO+ RefCOCOg

Model

Avg val testA testB val testA testB val test

Qwen3-VL-8b-Instruct4 91.12 93.74 87.52 86.78 90.57 81.06 88.74 88.73 88.70 KARL (mixed) 92.09 93.57 88.60 87.47 91.70 81.75 89.12 88.89 89.28

### C Experiment

This section provides additional details about the experiments and analysis, along with discussions on empirical findings.

#### C.1 Implementation Details

Baseline Models. When evaluating certain baseline models, some adjustments were made to accommodate minor differences in their output formats.For DeepSeek-VL2 [43], the first output bounding box was selected as the model’s predicted answer for downstream evaluation. For the evaluation of GroundingDINO-1.6-Pro [37] and DINO-X [34], we used their official APIs in deepdataspace . Images were converted to base64 format, with oversized images downsampled. Detection targets were fine-grained labels (e.g., “Buick Enclave”) instead of generic categories (e.g., “car”). API requests followed official configurations, and the highest-score bounding box was selected. YOLO-World [9] was evaluated locally under identical criteria. Regarding Shikra-7B [5], we have replicated the evaluation code based on the officially released model and code. However, the actual performance is significantly lower than that reported in the paper. Similar discrepancies have also been observed in replication attempts by other researchers.

Training Details. We use Adam optimizer [18] with learning rate as 1e − 6, β1 = 0.9 and β2 = 0.999 in the CoT-SFT stage and AdamW optimizer [25] with learning rate as 5e − 7, β1 = 0.9 and β2 = 0.999 in the GRPO stage. The accumulated batch size is set to 16 in stage 1 and 8 in stage2. The GRPO stage employs a maximum completion length of 1500 tokens with 4 samples per input.

#### C.2 Complementarity with Generic Grounding Data

We further examine whether our KVG-oriented training can complement, rather than replace, standard grounding supervision. To this end, we conduct a mixed-training experiment by incorporating a portion of RefCOCO training data into both Stage 1 and Stage 2. Specifically, we include 10,788 RefCOCO training samples in Stage 1 and 2,698 samples in Stage 2. For the RefCOCO samples, we use direct-answer supervision without CoT rationales in Stage 1, and disable knowledge-aware reward scaling in Stage 2, since these samples do not involve fine-grained entity-level knowledge.

As shown in Tab. 7, the mixed-training model achieves slightly better average performance on RefCOCO, RefCOCO+, and RefCOCOg than the base Qwen3-VL-8B-Instruct model. This result suggests that our KVG data and KARL-based optimization are complementary to existing generic grounding supervision: when trained jointly, they can enhance knowledge-intensive grounding ability without sacrificing competitive performance on standard grounding benchmarks. Notably, our goal here is not to replace generic grounding data with KVG data alone, but to show that the two forms of supervision can be combined effectively within a unified training pipeline.

4reproduced using lmms-eval [50]

Table 8: Joint correctness statistics of Qwen3-VL-8B-Instruct on the REC-style grounding task and the MCQ-based REG-style knowledge task over KVG-Bench. The relatively large proportion of cases that are correct on REG but incorrect on REC indicates a clear knowledge-grounding gap.

#### REC Correct REC Incorrect

REG Correct 46.11% 22.90% REG Incorrect 11.08% 19.91%

#### C.3 Quantitative Analysis of the Knowledge-Grounding Gap

In the main paper, we qualitatively discuss the knowledge-grounding gap, namely that MLLMs may possess the knowledge required to identify a target entity while still failing to ground it correctly. To further quantify this phenomenon, we compare model performance on two task forms derived from the same KVG-Bench instances: the original referring expression comprehension (REC) task for visual grounding, and a multiple-choice referring expression generation (REG) variant designed to probe entity-level knowledge.

The REG-style evaluation is constructed following a protocol similar to that used in the knowledge estimation analysis. Specifically, for each query, we convert the original grounding instance into a multiple-choice question in which the model is asked to identify the entity referred to by the query. We create three distractor options for each case. Whenever possible, we first select other entities appearing in the same image as distractors, since these candidates are visually co-present and therefore constitute more challenging alternatives. If fewer than three such entities are available, we randomly sample the remaining distractors from other entity labels within the same category, ensuring that each question always contains three negative options in total. This design yields an MCQ-based REG task that preserves the semantic scope of the original KVG query while removing the need for explicit localization.

We adopt this REG-style task as a proxy for knowledge because it mainly evaluates whether the model can correctly identify the target fine-grained entity among semantically related alternatives, without requiring box prediction. In contrast, REC additionally requires translating such knowledge into precise visual grounding. Therefore, comparing correctness across REG and REC on the same test cases provides a direct way to examine whether the model’s failure stems from lacking the relevant entity knowledge itself, or from failing to effectively use that knowledge during grounding.

As shown in table 8, only 46.11% of cases are solved correctly on both tasks, while 19.91% fail on both. More importantly, the proportion of cases where the model answers the knowledge task correctly but fails on grounding reaches 22.90%, which is substantially higher than the opposite case, where grounding is correct but knowledge prediction is wrong (11.08%). This asymmetry provides direct quantitative evidence for the knowledge-grounding gap: for a considerable portion of examples, the model already captures the relevant entity knowledge, yet fails to translate such knowledge into accurate visual localization. In other words, the bottleneck is often not the absence of knowledge itself, but the inability to effectively leverage that knowledge during grounding.

This result also suggests that improving KVG performance requires more than strengthening either visual grounding or factual knowledge in isolation. A key challenge is to better align knowledge usage with fine-grained grounding decisions, which motivates our emphasis on knowledge-aware reasoning and optimization.

