## Text-Only Data Synthesis for Vision Language Model Training

### Xiaomin Yu1, Wenjie Zhang1, Ziyue Qiao2, Chengwei Qin1,*, Hui Xiong1

#### 1 The Hong Kong University of Science and Technology (Guangzhou) 2 Great Bay University

GitHub: https://github.com/Yu-xm/Modality_Gap_Theory.git

Email: yuxm02@gmail.com * Corresponding Author.

Paired

### Abstract

Training vision-language models (VLMs) typically requires large-scale, high-quality imagetext pairs, but collecting or synthesizing such data is costly. In contrast, text data is abundant and inexpensive, prompting the question: can high-quality multimodal training data be synthesized purely from text? To address this problem, we propose a cross-integrated threestage multimodal data synthesis framework, termed Unicorn, which generates two datasets: Unicorn-1.2M and Unicorn-471K-Instruction. In Stage 1: Diverse Caption Data Synthesis, we construct 1.2M semantically diverse highquality captions by expanding sparse caption seeds using large language models (LLMs). In Stage 2: Instruction-Tuning Data Generation, we further process 471K captions into multiturn instruction-tuning tasks to support complex reasoning. Finally, in Stage 3: Modality Representation Transfer, these textual captions representations are transformed into visual representations, resulting in diverse synthetic image representations. This three-stage process enables us to construct Unicorn-1.2M for pretraining and Unicorn-471K-Instruction for instruction-tuning, without relying on real images. By eliminating the dependency on real images while maintaining data quality and diversity, our framework offers a cost-effective and scalable solution for VLM training.

[Figure 1]

# arXiv:2503.22655v2[cs.AI]27May2026

[Figure 2]

A close-up portrait of a brown bear, showcasing its striking... <omitted>

###### MLLM

[Figure 3]

A close-up portrait of a brown bear, showcasing its striking... <omitted>

- (a)
- (b)

A close-up portrait of a brown bear, showcasing its striking... <omitted>

Real Image Synthetic Image Captions

Synthetic Image Embeddings

Image-free

e

[Figure 4]

c

a

[Figure 5]

p

S

n

ntatio

Paired

Synthetic Image Embeddings

e

###### Mean Shift

s

n

pre

[Figure 6]

A serene beach scene at sunset, featuring a lone palm tree leaning... <omitted>

A serene beach scene at sunset, featuring a lone palm tree leaning... <omitted>

- d

R

- e

A serene beach scene at sunset, featuring a lone palm tree leaning... <omitted>

are

h

S

Synthetic Diverse Captions

Text Embeddings

[Figure 7]

109

Figure 1: Unlike traditional image-text data synthesis frameworks, Unicorn removes the dependency on real image data, offering a more efficient solution.

44

6.84

12

0.3

4

API / 1e-3$ Time / Day Storage / GB

image-text datasets. Constructing such datasets remains costly and difficult, as it requires not only massive amounts of multimodal data but also careful curation to ensure data quality, semantic diversity, and cross-modal alignment.

Existing approaches for acquiring image-text training data mainly rely on two strategies. Manual annotation (Lin et al., 2014; Plummer et al., 2015) provides high-quality captions but is limited by annotation cost, dataset scale, and semantic diversity. Large-scale web crawling (Sharma et al., 2018) improves scalability, but introduces noisy captions, inconsistent data quality, compliance risks, and potential copyright concerns. Recent works (Chen et al., 2024a; Li et al., 2024) augment real images with model-generated captions, such as fine-grained captions produced by powerful multimodal models. Although these methods improve data quality, they still require real images and expensive multimodal generation pipelines.

### 1 Introduction

Vision-Language Models (VLMs) (Liu et al., 2024a; He et al., 2024) have become a central paradigm for bridging visual and textual modalities, enabling models to understand, generate, and reason over multimodal inputs. Recent VLMs have demonstrated strong capabilities across visual question answering, image captioning, multimodal reasoning, and human-computer interaction. However, the success of high-performance VLMs largely depends on large-scale, high-quality, and diverse

In contrast, text data is abundant, inexpensive, and highly diverse. This raises a fundamental ques-

tion: Can text-only data provide effective multimodal supervision for VLM training without relying on real or synthetic images? More specifically, we ask whether visual supervision can be constructed by text directly in a shared representation space, as shown in Fig. 1. A key obstacle to this idea is the modality gap (Zhang et al., 2024): even in a shared representation space, text embeddings and image embeddings with similar semantics are not perfectly aligned, but are separated by systematic modality-dependent biases. Therefore, if textual representations can be transformed into visually compatible representations by mitigating this modality gap, then text-only data may serve as a scalable source of multimodal supervision (Tewel et al., 2022; Gu et al., 2023; Liu et al., 2024b).

To investigate this possibility, we first conduct a large-scale analysis of paired image-text representations. Our analysis reveals three empirical findings: ❶ the image-text modality gap contains a stable global component; ❷ this component is largely consistent across semantic regions; and ❸ it is transferable across unseen modality regions. These findings suggest that text-derived representations can be transformed into visually compatible representations through a shared transfer rule.

Motivated by these findings, we propose Unicorn, a scalable text-only multimodal data synthesis framework. Unicorn synthesizes data in the representation space. Specifically, Unicorn first expands sparse textual seeds into dense visual captions, then derives instruction-following data from these captions, and finally transfers their text embeddings into synthetic image representations in the shared embedding space. This pipeline constructs two datasets without using real images: Unicorn-1.2M for pretraining and Unicorn-471KInstruction for instruction tuning.

Using these datasets, we train Unicorn-8B, a VLM trained entirely from text-only synthetic multimodal supervision. Experimental results show that Unicorn-8B achieves competitive performance across multiple benchmarks compared with methods trained on conventional image-text pairs, demonstrating the feasibility of large-scale VLM training without real images.

Overall, our contributions are threefold. ❶ We reveal a stable and transferable modality gap structure. ❷ We propose Unicorn, a text-only data synthesis framework for VLM Training. ❸ We show that text-only synthetic data can effectively train a competitive VLM without real images.

### 2 Modality Gap Analysis

We first examine whether text representations can be made visually compatible in a shared representation space. Since text and image embeddings may remain systematically shifted even with similar semantics, directly treating text embeddings as image embeddings is suboptimal. We therefore analyze paired image-text representations to characterize the modality gap. Given 1M imagetext pairs {(xi,yi)}1i=1M, we use the image encoder fI(·) and text encoder fT(·) of LLM2CLIP to map images and texts into the same shared representation space, i.e., vi = fI(xi) and ui = fT(yi), where vi,ui ∈ Rd. All representations are ℓ2normalized before computation. For each imagetext pair, we define the pairwise modality gap as ∆i = vi − ui, and define the global modality gap as ∆¯ = N−1 i ∆i. The pairwise gap can then be decomposed as:

###### vi = ui + ∆¯ + ϵi, (1)

where ϵi denotes the residual alignment noise after removing the global modality bias.

Finding 1: Stable Global Bias. We first analyze the directional consistency between the pairwise gap ∆i and the global gap ∆¯. As shown in Fig. 2(a), the distribution of cos(∆i,∆)¯ is mostly concentrated in the positive region, with a peak around 0.35, and the average directional consistency is approximately 0.33. This indicates that most pairwise gaps are positively aligned with the global gap, suggesting the existence of a stable global modalitybias direction. We further compute the gap energy concentration ratio ρ = ∥∆¯∥22/(N−1 i ∥∆i∥22). In our experiments, ρ ≈ 0.11, indicating that the global gap explains approximately 11% of the pairwise gap energy. Thus, the global component is a stable and exploitable structural component. Additional analyses in the Appendix B show that the residual term ϵi = ∆i − ∆¯ remains substantial but is approximately zero-centered. These results suggest that image-text discrepancy is better modeled as a stable global bias plus residual noise.

Implication 1.

This finding suggests that text representations should not be directly used as image representations. Instead, a representation-space transfer step is required to compensate for the global modality bias.

###### (a)

###### (b)

###### (c)

0.95

| | | |
|---|---|---|
| | | |
| | | |
| | | |

0.36

- 0
- 1
- 2
- 3
- 4

0.90

Pairedcosine

cos(Δ,Δ)k

0.85

0.34

Density

No transfer

Source-gap transfer

0.80

0.32

0.75

0.30

0.70

−0.2 0.0 0.2 0.4 0.6

50 100 200 500

50 50/50

50 80/19

100 50/50

100 80/19

200 50/50

200 80/19

500 50/50

500 80/19

cos(Δi, Δ)̄

Number of clusters

K / source-target split

- Figure 2: Empirical principles of the modality gap in the shared representation space. (a) Pairwise modality gaps are directionally aligned with the global gap, indicating a stable global modality bias. (b) Local gaps estimated from unsupervised semantic clusters remain consistently aligned with the global gap across different clustering granularities. (c) A gap estimated from source clusters consistently improves paired cosine similarity on disjoint target clusters, demonstrating cross-region transferability.

- Finding 2: Cross-Semantic Consistency. We next examine whether the global gap is strongly tied to specific semantics. Since the 1M paired data do not contain domain annotations, we approximate semantic regions in an unsupervised manner. Specifically, we apply K-means clustering to text

representations ui and treat each cluster as a local semantic region, with K ∈ {50,100,200,500}. For the k-th semantic cluster Ck, we define its local gap as ∆¯k = |Ck|−1 i∈Ck(vi − ui), and compute its directional consistency with the global gap, namely cos(∆¯k,∆)¯ . As shown in Fig. 2(b), local gaps remain strongly aligned with the global gap across different clustering granularities. Most cluster-level similarities are concentrated in the range of approximately 0.8 to 0.9, and the distribution only decreases mildly as K increases. This trend is expected because larger K produces finer semantic clusters with fewer samples per cluster, making local gap estimates more sensitive to sample-level residuals. Nevertheless, even under the finest clustering setting, local gaps remain substantially and positively aligned with the global gap. These results indicate that the dominant modalitygap direction is not specific to individual semantic categories, but mainly reflects a shared modalitylevel bias across semantic regions.

Implication 2.

This finding suggests that a shared transfer rule can be applied to diverse textual data, rather than requiring a separate modality correction for every semantic category.

- Finding 3: Cross-Modality Transferability. Directional consistency alone does not guarantee

transferability. We therefore conduct a crosscluster gap transfer experiment. Based on the text-clustering results, we split clusters into source clusters and target clusters. We estimate the gap only using image-text pairs from source clusters,

i.e., cˆsrc = |Dsrc|−1 i∈Dsrc(vi − ui), and apply it to disjoint target clusters by constructing

vˆi = ui + cˆsrc. On target clusters, we compare cos(ui,vi) before transfer with cos(ui + cˆsrc,vi) after transfer. As shown in Fig. 2(c), under all K values and source/target split settings, the sourceestimated gap consistently improves paired cosine similarity on target clusters. Specifically, the paired cosine similarity before transfer is around 0.29, and increases to approximately 0.365 ∼ 0.370 after transfer, with an average improvement of about +0.075. This improvement remains stable across different clustering granularities and source/target split ratios. This result provides direct evidence that the modality gap contains a transferable component shared across semantic regions.

Implication 3.

This finding shows that the modality gap structure can generalize to unseen semantic regions. Therefore, for synthetic captions without paired images, we can still apply a shared transfer rule to convert text embeddings into visually compatible representations.

### 3 Data Synthesis Pipeline

Motivated by the empirical principles established in Sec. 2, we design Unicorn as a text-only data synthesis pipeline for constructing visually compatible multimodal supervision. The key idea is to first make textual captions sufficiently visual-

###### Data Synthesis Pipeline

Sample

###### Seed Dataset

1.2M 1.2M 471k

###### StageⅠ. Diverse Caption Data Synthesis

###### StageⅡ. Instruction-Tuning Data Synthesis

###### Question Answering

Open-Domain Caption Seed

Domain-Specific Knowledge Seed

A ferry, painted in a standard white with blue and red trim, is departing from the main port. The ferry's deck is visible, featuring rows of seating arranged in a linear pattern, with safety railings lining the edges. The upper deck houses a glass-enclosed control room from which the captain operates the vessel. The ferry's hull cuts through the calm water, creating a visible wake that spreads out in a V-shape behind the vessel. The port, from which the ferry is departing, is a bustling scene with other boats docked at various piers, and several people can be seen walking along the paved walkways. The sky above is clear, with a few scattered clouds, and the sun casts a gentle light over the scene, highlighting the ferry's white paint and the glinting water.

Multiple Choice

Complex Realsoning

Manually Annotated Caption Dataset

Diverse Caption

A ferry departs a bustling port, cutting through calm waters.

Add Details

Task Specifc

Domain

A ferry, painted in a standard white with blue and red trim, is departing from the main port. The ferry's deck is visible, featuring rows of seating arranged in a linear pattern, with safety railings lining the edges. The upper deck houses a glass-enclosed control room f r o m w h i c h t h e c a p t a i n operates the vessel. The ferry's hull... <omitted>

###### Question:

[Figure 8]

Manually Annotated Caption Dataset

[Figure 9]

###### Question:

Given the ferry's departure from the main port during what appears to be daytime, and considering the visible wake it creates, what can you infer about the water conditions and the ferry's speed?

.

###### Question:

[Figure 10]

Where is the control room located on the ferry?

[Figure 11]

[Figure 12]

\

How are the seats arranged on the ferry's deck?

LLMs

###### Options:

###### Web-Crawled Caption Dataset

LLMs

###### Answer:

[Figure 13]

[Figure 14]

[Figure 15]

- A. On the lower deck
- B. In the middle of deck
- C. On the upper deck
- D. Under the water Answer:C

The seats on the ferry's deck are arranged in a linear pattern, with rows of seating facing forward.

Web-Crawled Caption Dataset

###### Answer:

heavy snow falls over a snow lined river .

[Figure 16]

[Figure 17]

[Figure 18]

The calm water and the visible V-shaped wake suggest that the water...<omitted>

...

Long Context Captions

Unicorn-1.2M

Synthetic Dataset

Paired

Paired

"conversations": [ {

###### StageⅢ. Modality Representation Transfer

Synthetic Image Embeddings

"from": "human", "value": "<image>\nProvide a detailed description of the image."

Text Encoder

}, {

Text Embeddings Synthetic Image Embeddings

###### 1.2M

"from": "gpt", "value": "A ferry, painted in a standard white with blue...<omitted>"

###### Paired

} ]

Sample

###### Representation Space

Unicon-471k-Instruction

[Figure 19]

"conversations": [ {

、

"from": "human", "value": "<image>\nWhere is the control room located on the

Synthetic Image Embeddings

Text Representation Distribution Space

###### Text Embedding

ferry?\nA:On the lower deck\nB:In the middle of deck...<omitted>" }, {

Mean Shift

471k

"from": "gpt", "value": "A ferry, painted in a standard white...<omitted>"

Paired

}

Image Representation Distribution Space

Synthetic Image Embedding

... ]

- Figure 3: Unicorn’s text-only data synthesis pipeline, comprising three cross-integrated stages, (1) Diverse Caption Data Synthesis, (2) Instruction-Tuning Data Synthesis, and (3) Representation Transfer, ultimately yields two synthetic datasets: Unicorn-1.2M and Unicorn-471K-Instruction. Notably, unlike traditional approaches that rely on real images, Unicorn generates image representations paired with text, entirely free of real image data.

izable, then convert their dense visual semantics into instruction-following supervision, and finally transfer their text embeddings toward the visual representation space through modality-gap-guided representation transfer.

seeds include two types of image caption datasets: manually annotated captions and web-crawled captions. Manually annotated caption datasets provide basic information about entities, events, and contexts. To improve diversity and reduce redundancy, we retain only one caption per image. Web-crawled caption datasets provide large-scale and semantically diverse captions. Since web-crawled captions may contain noise, we filter and select captions that more closely resemble high-quality captions.

3.1 Seed Dataset

According to the findings in Sec. 2, the dominant modality gap remains largely consistent across semantic regions, suggesting that a unified representation transfer rule can be applied to diverse textual sources. Therefore, before synthesis, we construct a seed dataset that covers both open-domain visual scenarios and domain-specific concepts.

Domain-Specific Knowledge Seeds. To inject domain-specific knowledge, we collect 10K finegrained image labels from six fine-grained image classification datasets across 18 domains. For each dataset, we design specialized prompt templates. For example, we use templates such as “An image of a delicious <label>”, where <label> is dynamically replaced with the actual class name. Each label is paired with 10 randomly selected templates. This process generates 107K domain-specific knowledge seeds, enriching the seed dataset with fine-grained domain concepts.

To construct high-quality, diverse captions, we collect a seed dataset with two main objectives: ❶ to provide diverse textual visual scenarios for fine-grained alignment during VLM pretraining, and ❷ to incorporate domain-specific knowledge, enhancing the model’s understanding of specialized content. Importantly, we use only the textual captions from these datasets without involving any images. Specifically, given M caption datasets {Dm}Mm=1, where each dataset Dm contains image captions {si}Ni=1m, we filter and merge the original textual captions from these datasets, resulting in a seed dataset S = {s1,s2,...,sN} containing 1.2M high-quality textual samples, referred to as Unicorn-1.2M-Seed. This seed dataset consists of two categories: open-domain caption seeds and domain-specific knowledge seeds.

3.2 Stage 1. Diverse Caption Data

The first condition for text-only multimodal supervision is semantic visualizability: a textual description should contain sufficiently concrete visual information so that its corresponding visual equivalence set becomes less ambiguous. Therefore, given a sparse seed caption si, our goal is to generate a dense visual caption ci = Φ(si) that specifies objects, attributes, spatial relations, back-

Open-Domain Caption Seeds. Open-domain

Caption

Caption Caption

The filming location at night is a vibrant urban setting, characterized by a dense network of neon lights that cast a multicolored glow across the scene. The primary subject is a large, retro-style sign that reads \"Wonder Magazine\" in bold, cursive letters... <omitted>

The image depicts old multi-pane Gothic windows set into the exterior of a stone building. Each window consists of a large, arched frame divided into several smaller panes by thin, vertical and horizontal mullions. The panes are arranged in a symmetrical pattern... <omitted>

A person stands on the sidelines, observing as a football player in a white jersey and blue shorts stretches his right leg out in a futile attempt to intercept a deflected ball. The ball, having been deflected by another player, slowly trickles past the football player's outstretched foot... <omitted>

###### Instruction Instruction

Instruction

Question: What is the primary subject of the scene? Options:

Question: What architectural style is represented by the windows in the image?

Question: How does the deflection of the ball affect the outcome of the play?

Answer: The deflection of the ball by another player changes the trajectory and speed of the ball, making it difficult for the football player in the white jersey and blue shorts to intercept it...<omitted>

Answer: The windows in the image are representative of the Gothic architectural style.

- A: A group of pedestrians
- B: A large, retro-style sign
- C: A series of smaller advertisements
- D: A few parked vehicles Answer: B

(a) Multiple Choice (b) Question-Answering (c) Complex Reasoning

- Figure 4: Data formats for the three instruction-tuning tasks. Each dialogue’s answer is derived from the textual content of diverse caption.

ground, lighting, and scene context. Formally, let X(s) denote the set of possible images that are semantically consistent with a textual description s. Semantic densification aims to reduce the uncertainty of the corresponding visual representation distribution, i.e.,

it derives questions and answers directly from captions, ensuring that all supervision remains textonly and semantically consistent with ci.

As shown in Fig. 4, we design three instructiontuning tasks. ❶ Multiple Choice. This task generates multiple-choice questions based on a given caption ci. Each question contains four options, denoted as A, B, C, and D, with only one correct answer. ❷ Question Answering. This task generates open-ended questions and short answers based on the dense caption. ❸ Complex Reasoning. This task generates questions that require multi-step reasoning, hypothetical deduction, or commonsense inference grounded in the dense caption. For these tasks, we sample 171K, 199K, and 100K captions from the dense caption set C, respectively. For each selected caption ci, we insert it into a task-specific template and use Qwen2.5-72B-Instruct (Qwen et al., 2025) to generate an instruction-tuning sample tj. Since all generated questions and answers are derived from the original dense caption, the resulting instruction data remain consistent with the underlying imaginary visual scene. This design mitigates hallucination compared with pipelines that synthesize image-text pairs without explicit semantic control. Finally, we synthesize a total of 471K high-quality instruction-tuning samples, denoted as T = {t1,t2,...,tM}.

Varx∼X(ci)[fI(x)] < Varx∼X(si)[fI(x)]. (2)

Thus, the dense caption ci can be regarded as describing a more specific imaginary visual scene, even though no real image is used.

Given the Unicorn-1.2M-Seed dataset S = {s1,s2,...,sN}, we use Qwen2.5-72B-Instruct (Qwen et al., 2025) as the rewriting function Φ to enrich each seed caption si with additional visual details, producing the dense visual caption ci = Φ(si). The prompt is provided in the Appendix. This process produces a dense caption set C = {c1,c2,...,cN} containing 1.2M samples. The resulting dense captions serve three roles in Unicorn: ❶ they provide caption-level supervision for pretraining; ❷ they serve as semantic sources for instruction synthesis in Stage 2; and ❸ after text encoding, they are transformed into synthetic image representations in Stage 3.

##### 3.3 Stage 2. Instruction-Tuning Data

After semantic densification, each caption ci describes a concrete imaginary visual scene. However, the caption is insufficient for training instruction-following VLMs. Therefore, we further convert dense visual semantics into instructiontuning data. This stage does not introduce images;

##### 3.4 Stage 3. Representation Transfer

The third stage implements the representation compatibility principle established in Sec. 2. The empirical analysis shows that text embeddings and

###### Training

###### Inference

Real Image

[Figure 20]

Synthetic Image Embeddings Language Insruction

Language Insruction

[Figure 21]

[Figure 22]

[Figure 23]

Real Image Embeddings

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Projection

Projection

[Figure 28]

[Figure 29]

[Figure 30]

###### Image Encoder

Large Language Model

Large Language Model

[Figure 31]

[Figure 32]

Mean Shift

Real Image Embeddings

[Figure 33]

Language Response

Language Response

[Figure 34]

[Figure 35]

Representation Space

(a) Training Stage (b) Inference Stage

- Figure 5: Training and inference processes of Unicorn-8B. Training aligns synthetic image representations with LLM embeddings, while inference adjusts real image embeddings as input.

image embeddings are systematically shifted in the shared representation space, and that the dominant component of this shift is stable and transferable across semantic regions. Therefore, we synthesize visually compatible image representations by applying modality-gap-guided transfer in the shared embedding space.

images. Thus, Stage 3 converts text-only dense captions into representation-level visual supervision without generating pixels.

##### 3.5 Synthetic Dataset Construction

After the three stages, Unicorn produces three components: dense visual captions C, instructiontuning data T, and synthetic image representations Vˆ = {vˆi}Ni=1. These components are paired to construct two text-only multimodal datasets.

Given the dense caption set C = {c1,c2,...,cN}, we encode each caption ci using the text encoder of LLM2CLIP (Huang et al., 2026), obtaining a text representation ui ∈ Rd. The full set of dense-caption text representations is denoted as U = {u1,u2,...,uN}. We adopt LLM2CLIP because it can process long-form captions and complex semantic captions through its extended input window, making it suitable for encoding dense visual captions.

Unicorn-1.2M. For each dense caption ci, we pair it with its synthetic image representation vˆi, forming a pretraining dataset: DI = {(ˆvi,ci)}Ni=1. This dataset contains 1.2M synthetic multimodal pairs.

Unicorn-471K-Instruction. For each instructiontuning sample tj generated from a dense caption ca(j), we pair it with the corresponding synthetic image representation vˆa(j), forming an instructiontuning dataset DII = {(ˆva(j),tj)}Mj=1, where a(j) denotes the index of the dense caption from which instruction sample tj is generated. This dataset contains 471K synthetic instruction pairs.

According to Sec. 2, a calibrated text-to-visual transfer can be written as ui − µT + µV , where µT and µV denote the centers of text and image representation distributions, respectively. However, in the text-only setting, real images are unavailable, and µV cannot be estimated from the synthetic data. Therefore, we use an image-free approximation that removes the dominant text-side bias:

4 Unicorn-8B

Based on the two synthetic datasets, we train a VLM referred to as Unicorn-8B. This section presents the Unicorn-8B’s architecture and details of its training and inference processes. Fig. 5 shows the relevant details.

vˆi = Norm(ui − µU), (3)

where µU = |U|−1 i ui is the mean of the text representations, and Norm(·) denotes ℓ2 normalization. The resulting vˆi is used as the synthetic image representation paired with caption ci.

##### 4.1 Architecture

This operation should not be interpreted as reconstructing the real image representation of ci. Instead, it constructs a visually compatible surrogate in the shared representation space. As shown in Sec. 2, modality gap transfer improves text-image paired alignment across unseen semantic regions, suggesting that such a shared transfer rule can generalize to synthetic dense captions without paired

Unicorn-8B is composed of two components: the projector W and the backbone LLM fφ. For the projector W, we adopt a multilayer perceptron (MLP) architecture equipped with GELU activation functions, following the design in LLaVA (Liu et al., 2024a). This component serves as a modality adapter, effectively bridging the input representa-

- Table 1: Ablation study on modality representation transfer. We compare directly using text embeddings, applying C3 Align, and the proposed transfer in Unicorn.

Method Text-only

General Reasoning Hallucination

Avg. ↑ MME MMStar SQA RealWorldQA MMMU MMMU-P VisuLogic LogicVista CRPE POPE HallBench

w/o. Align ✓ 57.46 33.87 66.02 39.18 33.41 31.26 24.10 26.44 39.07 62.37 40.12 41.21 C3 Align ✓ 58.02 32.13 65.74 38.66 33.95 30.72 23.90 26.97 38.64 61.31 39.58 40.87 Unicorn ✓ 60.24 35.13 68.81 42.35 36.87 34.05 26.80 29.53 42.32 64.21 43.01 43.94

- Table 2: Scale-matched comparison between paired image-text pretraining and text-only pretraining. For each scale, Unicorn is matched to the pretraining data size of the corresponding image-text baseline, while all baselines use their original pretraining data. All methods use the same 417K SFT data.

General Reasoning Hallucination

Method Text-only Pretrain Scale

Avg. ↑ MME MMStar SQA RealWorldQA MMMU MMMU-P VisuLogic LogicVista CRPE POPE HallBench

LLaVA-1.5 ✗ 558K 54.86 33.27 64.91 38.73 33.16 29.88 22.90 25.41 37.66 60.73 38.84 40.03 Unicorn∗ ✓ 558K 56.71 34.20 65.12 39.83 34.23 30.27 24.30 26.81 39.22 62.14 40.38 41.20 Bunny ✗ 1.2M 58.37 34.73 67.28 41.08 34.62 32.41 25.30 27.86 40.51 63.12 41.26 42.41 Unicorn ✓ 1.2M 60.24 35.13 68.81 42.35 36.87 34.05 26.80 29.53 42.32 64.21 43.01 43.94

tions to the latent space of the LLM. We utilize LLaMA3-8B-Instruct as the backbone LLM.

##### 4.2 Training & Inference

Training. In the pretraining stage, synthetic image representations vˆi are aligned with pre-trained LLM representation Hq through the projection matrix W. During this phase, only W is updated, while the weights of the LLM fφ remain frozen. Subsequently, in the instruction-tuning stage, the Unicorn-471K-Instruction is employed to update both W and the LLM fφ weights, thereby enhancing the model’s instruction-following capability.

Inference. In the inference stage, real images from the test set are decoded by the image encoder to get real image representations. To mitigate the modality gap, the global mean of the test image representation set E[V ] is subtracted from each input test image representation vi ∈ Rh, resulting in a transformed representation v˜i = vi − E[V ].

### 5 Experiments

We evaluate Unicorn on three benchmark groups: general visual understanding, multimodal reasoning, and hallucination. Detailed experimental settings are provided in Appendix E.

Modality Representation Transfer. Table 1 compares alignment strategies for constructing synthetic image representations. Directly using text embeddings without alignment achieves an average score of 41.21, indicating that text embeddings pre-

serve useful semantics but remain biased toward the text-side distribution. C3 (Zhang et al., 2024) Align obtains a lower average score of 40.87, despite improving a few individual metrics. This suggests that adding stochastic or Gaussian-style perturbations to model residual uncertainty is insufficient: the residual term ϵi in the modality-gap model is not the dominant factor to correct before the global bias is removed, and isotropic noise may even disturb semantic consistency. In contrast, Unicorn explicitly compensates for the structured modality shift and achieves the best average score of 43.94, improving over w/o Align by +2.73. The gains are consistent across general, reasoning, and hallucination benchmarks, supporting the theoretical claim that text embeddings must be transferred toward the visual-compatible distribution rather than used directly or perturbed with generic noise.

Image-Text Training Baselines. Table 2 compares Unicorn with paired image-text pretraining baselines. To ensure a fair comparison, all methods are matched in data scale. Under the same training data scale setting, Unicorn∗ achieves an average score of 41.20, outperforming LLaVA-1.5 by +1.17. Unicorn further reaches 43.94, surpassing Bunny by +1.53. The gains are especially clear on reasoning-oriented benchmarks such as MMMU, MMMU-Pro, VisuLogic, and LogicVista. We attribute this improvement to the dense captions used in Unicorn: compared with conventional image-text captions, they provide more explicit de-

- Table 3: Ablation study on semantic densification. The variant w/o Semantic Densification directly uses raw caption seeds for pretraining, while Unicorn uses dense visual captions.

Method

General Reasoning Hallucination

Avg. ↑ MME MMStar SQA RealWorldQA MMMU MMMU-P VisuLogic LogicVista CRPE POPE HallBench

w/o. Dense 57.36 33.87 66.14 39.72 33.96 31.42 24.30 26.88 39.51 62.43 40.28 41.44 Unicorn 60.24 35.13 68.81 42.35 36.87 34.05 26.80 29.53 42.32 64.21 43.01 43.94

- Table 4: Ablation study on seed diversity. Open-domain uses open-domain captions, while Unicorn uses both open-domain captions and domain-specific knowledge captions.

General Reasoning Hallucination

Method

Avg. ↑ MME MMStar SQA RealWorldQA MMMU MMMU-P VisuLogic LogicVista CRPE POPE HallBench

Open-domain 58.86 34.27 67.56 40.91 34.95 32.18 25.10 27.42 40.13 63.48 41.08 42.36 Unicorn 60.24 35.13 68.81 42.35 36.87 34.05 26.80 29.53 42.32 64.21 43.01 43.94

scriptions of objects, attributes, spatial relations, and scene context, thereby offering richer semantic supervision for VLM training. These results indicate that, with semantic densification and modalitygap-guided transfer, text-only representation-level supervision can serve as an effective alternative to standard paired image-text supervision.

Ablation on Semantic Densification. Table 3 evaluates the role of semantic densification. Removing this stage and using raw caption seeds reduces the average score from 43.94 to 41.44. This drop supports the semantic visualizability assumption in Sec. 2: sparse captions correspond to a broad visual equivalence set X(s), making the induced visual representation distribution ambiguous. Dense captions reduce this uncertainty by specifying objects, attributes, relations, background, and scene context. The performance degradation is particularly evident on reasoning and hallucinationrelated benchmarks, indicating that semantic densification does not merely increase text length but provides more determinate visual semantics for constructing stable synthetic image representations.

Ablation on Seed Diversity. Table 4 studies the effect of seed diversity. Using only open-domain caption seeds yields an average score of 42.36, while the full Unicorn setting reaches 43.94. The improvement is larger on reasoning benchmarks, where domain-specific knowledge seeds provide fine-grained concepts that are less covered by opendomain captions. This result is consistent with the observation in Sec. 2 that the dominant modality gap is shared across semantic regions: a uni-

fied transfer rule can be applied to diverse textual sources, but the semantic coverage of those sources determines the richness of the supervision. Therefore, combining open-domain and domain-specific seeds improves both coverage and precision.

Cost Analysis. We provide cost analysis in Appendix D. Since Unicorn operates representation spaces, it avoids image collection, image storage, and image-conditioned annotation, making it practical for scalable VLM data construction.

Scale Analysis. Additional analyses on data scales are provided in Appendix E. The results show that Unicorn benefits from increased textonly supervision, indicating that the proposed synthesis pipeline can effectively scale with data size.

### 6 Conclusion

In this paper, we present Unicorn, a text-only data synthesis framework for VLM training. By analyzing the modality gap in shared representation space, we show that text embeddings can be transferred into visually compatible representations through a stable and transferable modality gap structure. Guided by this finding, Unicorn constructs VLM training data through a three-stage pipeline without using real images. Experiments show that Unicorn achieves competitive performance against paired image-text pretraining baselines. These results demonstrate that text-only data can be a scalable and effective source of multimodal supervision, providing an alternative to conventional imagedependent data construction. We hope Unicorn inspires future works on scalable synthetic data.

### Limitations

This work mainly focuses on VLM training in the image-text setting and has not yet been extended to broader multimodal inputs such as video, audio, or 3D data. Future work may explore whether similar text-only synthesis principles can be transferred to other modality forms, further broadening the applicability of the proposed framework.

### References

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. 2024a. Sharegpt4v: Improving large multi-modal models with better captions. In European Conference on Computer Vision, pages 370–387. Springer.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and 1 others. 2024b. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37:27056–27087.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and 1 others. 2025. Mme: A comprehensive evaluation benchmark for multimodal large language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Sophia Gu, Christopher Clark, and Aniruddha Kembhavi. 2023. I can’t believe there’s no images! learning visual tasks using only language supervision. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2672–2683.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, and 1 others. 2024. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14375–14385.

Muyang He, Yexin Liu, Boya Wu, Jianhao Yuan, Yueze Wang, Tiejun Huang, and Bo Zhao. 2024. Efficient multimodal learning from data-centric perspective. arXiv preprint arXiv:2402.11530.

Weiquan Huang, Aoqi Wu, Yifan Yang, Xufang Luo, Yuqing Yang, Usman Naseem, Chunyu Wang, Qi Dai, Xiyang Dai, Dongdong Chen, and 1 others. 2026. Llm2clip: Powerful language model unlocks richer cross-modality representation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 5131–5139.

Xiaotong Li, Fan Zhang, Haiwen Diao, Yueze Wang, Xinlong Wang, and Ling-Yu Duan. 2024.

Densefusion-1m: Merging vision experts for comprehensive multimodal perception. Advances in Neural Information Processing Systems, 37:18535–18556.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. 2023. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024a. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26296–26306.

Yang Liu, Xiaomin Yu, Gongyu Zhang, Zhen Zhu, Christos Bergeles, Prokar Dasgupta, Alejandro Granados, and Sebastien Ourselin. 2024b. Arcsin: Adaptive ranged cosine similarity injected noise for language-driven visual tasks. arXiv preprint arXiv:2402.17298.

Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, KaiWei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521.

Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. 2015. Flickr30k entities: Collecting region-to-phrase correspondences for richer imageto-sentence models. In Proceedings of the IEEE international conference on computer vision, pages 2641–2649.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, and 25 others. 2025. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. 2018. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565.

Yoad Tewel, Yoav Shalev, Idan Schwartz, and Lior Wolf. 2022. Zerocap: Zero-shot image-to-text generation for visual-semantic arithmetic. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 17918–17928.

Weiyun Wang, Yiming Ren, Haowen Luo, Tiantong Li, Chenxiang Yan, Zhe Chen, Wenhai Wang, Qingyun

Li, Lewei Lu, Xizhou Zhu, and 1 others. 2024. The all-seeing project v2: Towards general relation comprehension of the open world. In European Conference on Computer Vision, pages 471–490. Springer.

Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. 2024. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts. arXiv preprint arXiv:2407.04973.

Weiye Xu, Jiahao Wang, Weiyun Wang, Zhe Chen, Wengang Zhou, Aijun Yang, Lewei Lu, Houqiang Li, Xiaohua Wang, Xizhou Zhu, and 1 others. 2025. Visulogic: A benchmark for evaluating visual reasoning in multi-modal large language models. arXiv preprint arXiv:2504.15279.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, and 1 others. 2024. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556– 9567.

Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, and 1 others. 2025. Mmmupro: A more robust multi-discipline multimodal understanding benchmark. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15134– 15186.

Yuhui Zhang, Elaine Sui, and Serena Yeung. 2024. Connect, collapse, corrupt: Learning cross-modal tasks with uni-modal data. In International Conference on Learning Representations, volume 2024, pages 51711–51736.

### A Theoretical Justification

This section provides a theoretical justification for the modality-gap-guided representation transfer used in Unicorn. The analysis complements the empirical principles in Section 2 and formalizes why a shared gap-transfer rule can convert text representations into visually compatible representations in the shared embedding space.

##### A.1 Global Modality Bias as the Optimal Translation

Let ui = fT(yi) and vi = fI(xi) denote the text and image representations of a paired sample. We model their relationship as

vi = ui + c + ϵi, E[ϵi] = 0, (4)

where c denotes the modality-level bias and ϵi denotes the sample-level residual discrepancy. Under the squared-error criterion, the optimal global transfer vector is

c∗ = arg min

E ∥ui + c − vi∥22 . (5) Taking the derivative with respect to c gives

c

E[ui + c∗ − vi] = 0, (6) and therefore

c∗ = E[vi − ui]. (7)

Thus, the empirical global gap ∆¯ = N−1 i(vi − ui) is the finite-sample estimator of the optimal translation. This shows that the global modality gap is not an ad-hoc correction, but the optimal constant

transfer vector under the mean-squared alignment objective. The corresponding transferred representation is vˆi = ui + ∆¯. Its error can be written as

vˆi − vi = (∆¯ − c) − ϵi. (8)

When ∆¯ consistently estimates c, the remaining error is dominated by the residual term ϵi. Therefore, global transfer should be understood as improving visual compatibility rather than exactly reconstructing each image representation.

##### A.2 Consistency of the Empirical Gap Estimator

Assume that ∆i = vi − ui are independent samples with finite covariance Σ∆, and that E[∆i] = c. The empirical estimator is

N

1 N

∆¯ =

∆i. (9)

i=1

Then

1 N

E[∆]¯ = c, Cov(∆)¯ =

Σ∆. (10) Hence, the estimation error decreases as O(N−1/2). Equivalently,

E ∥∆¯ − c∥22 =

1 N

Tr(Σ∆). (11)

This explains why estimating the modality gap from a large paired representation set is stable. Although the residual term ϵi may remain large at the instance level, the global component can still be estimated reliably from sufficiently many samples.

- A.3 Why Cross-Region Transfer Improves Alignment We next analyze why a gap estimated from one semantic region can improve alignment in another region.

Let Rs and Rt denote source and target semantic regions. Suppose the local gap in each region can be written as

cs = c + rs, ct = c + rt, (12)

where c is the shared modality-level bias and rs,rt are region-specific residual biases. For a target sample, assume

vi = ui + ct + ϵi, E[ϵi] = 0. (13) Without transfer, the expected squared alignment error is

E0 = E ∥ui − vi∥22 = ∥ct∥22 + E ∥ϵi∥22 . (14)

If we estimate a transfer vector from the source region and apply it to the target region, the transferred error becomes

Es→t = E ∥ui + cs − vi∥22 = ∥cs − ct∥22 + E ∥ϵi∥22 . (15) Therefore, source-to-target transfer improves the expected squared error whenever

∥cs − ct∥22 < ∥ct∥22. (16) Equivalently,

2⟨cs,ct⟩ > ∥cs∥22. (17) This condition holds when the source and target gap directions are sufficiently aligned and the regionspecific residual biases are small relative to the shared global bias. This directly corresponds to the empirical observations in Section 2: local gaps remain positively aligned with the global gap, and a source-estimated gap consistently improves target-region cosine similarity.

For normalized representations, reducing squared distance is closely related to improving cosine similarity. If a and b are ℓ2-normalized, then

∥a − b∥22 = 2 − 2cos(a,b). (18)

Thus, a reduction in alignment error corresponds to an increase in paired cosine similarity. This explains why the cross-cluster transfer experiment can be interpreted as evidence for transferable visual compatibility.

- A.4 Semantic Densification Reduces Visual Ambiguity

The first stage of Unicorn expands a sparse caption si into a dense visual caption ci = Φ(si). This operation is motivated by the fact that sparse captions are compatible with many visually different images. Let X(s) denote the set of images semantically consistent with a textual description s, and let Z = fI(x) be the corresponding image representation. A dense caption can be viewed as a refinement of the sparse caption because it introduces additional visual constraints, such as objects, attributes, relations, background, and scene context.

Formally, let S denote a sparse description and C denote its densified description. Since C contains additional information conditioned on S, the law of total variance gives

Var(Z | S) = E[Var(Z | C) | S] + Var(E[Z | C] | S). (19) Since the second term is non-negative, we obtain

###### E[Var(Z | C) | S] ≤ Var(Z | S). (20)

This shows that conditioning on a denser visual description reduces the expected uncertainty of the corresponding visual representation. Therefore, semantic densification does not merely increase the length of captions; it reduces visual ambiguity and makes the text representation closer to a specific visual-compatible region in the shared embedding space.

###### (a)

###### (b)

0.10

| |Cumulative<br><br>Individual| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

Explainedvariance

6

0.08

Density

0.06

4

0.04

2

0.02

0

0.00

0.9 1.0 1.1 1.2 1.3 1.4

1 2 3 4 5 6 7 8 9 10

Principal component

‖εi‖2

###### (c)

(d)

0.0825

1.0

Mean

Cosineimprovement

0.0800

Median

Ratio > 0.8

0.9

0.0775

Value

| | |
|---|---|
| | |

0.0750

0.8

0.0725

0.7

0.0700

100 200 300 400 500

50 50/50

50 80/19

100 50/50

100 80/19

200 50/50

200 80/19

500 50/50

500 80/19

K

K / source-target split

Figure 6: Additional analysis of the modality gap. (a) Residual norm distribution after removing the global gap. (b) PCA explained variance of pairwise gap vectors. (c) Cluster-level statistics across different semantic granularities. (d) Cross-cluster transfer improvement on disjoint target clusters.

- A.5 Image-Free Approximation in Unicorn The ideal text-to-visual transfer based on paired statistics can be written as

vˆi = Norm(ui − µT + µV ), (21)

- where µT and µV are the centers of text and image representation distributions. However, in the text-only synthesis setting, real images are unavailable, and therefore µV cannot be estimated from the synthetic data. This makes the exact paired mean-shift transfer unidentifiable without additional assumptions.

Unicorn therefore adopts an image-free approximation:

vˆi = Norm(ui − µU), (22)

- where µU = |U|−1 i ui is the center of the dense-caption text representations. This operation removes the dominant text-side mean component and normalizes the resulting representation. It should not be interpreted as an unbiased estimator of the real image embedding. Instead, it constructs a visually compatible surrogate by reducing text-side distributional bias and placing the synthetic representation in a normalized shared embedding space.

Combined with the empirical transferability of the modality gap, this approximation provides a practical image-free mechanism for constructing representation-level visual supervision. The resulting synthetic image representations are therefore best understood as visual-compatible training signals, rather than exact substitutes for real image features.

- B Additional Analysis This section provides additional analyses for the modality-gap principles discussed in Sec. 2.

Residual structure after removing the global bias. In Sec. 2, we decompose the pairwise image-text gap as ∆i = ∆¯ + ϵi, where ∆¯ denotes the global modality bias and ϵi denotes the residual term. To further examine the residual structure, we analyze the norm distribution of ϵi = ∆i − ∆¯. As shown in Fig. 6(a), the residual norm is mainly concentrated between 1.05 and 1.20, indicating that substantial instance-level discrepancy remains after removing the global gap. Meanwhile, the residual mean norm is close to zero, suggesting that the residuals are approximately centered after global-bias removal. This supports our modeling choice: the modality gap should not be treated as a deterministic translation alone,

but as a combination of a stable global bias and sample-level residual variation. Therefore, the purpose of representation transfer is not to reconstruct the exact image embedding for each sample, but to construct a visually compatible approximation in the shared representation space.

Dimensionality of pairwise gap variations. We further perform PCA on the pairwise gap vectors ∆i. As shown in Fig. 6(b), the first principal component explains only about 2.2% of the variance, and the top ten components explain about 9.7% in total. This indicates that the residual variation of the modality gap is not concentrated in a single dominant direction. In other words, although the global gap ∆¯ provides a stable and useful modality-level correction, the full pairwise discrepancy contains richer sample-level variations. This observation is consistent with the residual analysis above and explains why a single global vector improves visual compatibility but cannot precisely recover real image representations.

Cluster-level stability across semantic granularities. In the main text, we show that local gaps estimated from unsupervised semantic clusters remain positively aligned with the global gap. Fig. 6(c) provides a complementary summary across different values of K. The mean and median similarities decrease mildly as K increases, while remaining consistently positive. This trend is expected because larger K produces finer clusters with fewer samples per cluster, making each local gap estimate noisier. The ratio of clusters with similarity larger than 0.8 also decreases as semantic regions become finer, indicating that semantic-specific residuals become more visible at fine granularities. Nevertheless, the local gaps remain substantially aligned with the global gap across all settings, supporting the conclusion that the dominant modality bias is shared across semantic regions rather than being fully determined by specific categories.

Robustness of cross-cluster transfer. Finally, we provide the distribution of cosine improvements in the cross-cluster transfer experiment. As shown in Fig. 6(d), the source-estimated gap consistently improves paired cosine similarity on disjoint target clusters across different clustering granularities and source/target split ratios. The improvements are concentrated around +0.075, showing that the gain is stable rather than caused by a particular split. Since the target clusters are not used to estimate the transfer vector, this result further supports the cross-region transferability of the modality gap. It also justifies applying a shared gap-transfer rule to synthetic dense captions without paired images.

### C Data Synthesis Prompts

This section provides the prompts used in the text-only data synthesis pipeline. The prompts are used for two purposes. First, the detailed caption prompt expands sparse caption seeds into dense visual descriptions, making the textual input more visually specific and less ambiguous. Second, the instructionsynthesis prompts convert dense visual captions into three types of instruction-following data: multiple choice, visual question answering, and complex reasoning. All prompts operate only on textual inputs and do not require real or synthetic images.

##### Detailed Caption

Please expand the given image caption to include observable, objective, and reasonable details, avoiding any literary rhetoric, subjective judgments, or emotional tones. Please focus especially on depicting the detailed features of the primary subject, ensuring a clear and specific caption of its appearance and structure, while making all details as comprehensive and concrete as possible. Ensure the caption remains natural and all content is presented in a single paragraph. The image caption is as follows: <seed>

##### Task 1: Multiple Choice

You MUST ONLY output valid JSON. Generate nothing but JSON! The JSON must have the structure:

{

"id": " ", "QA Pairs": [

{

"Question": " ", "Options": {

- "A": " ",

- "B": " ",
- "C": " ",
- "D": " "

}, "Answer": " "

},

... ]

}

Please generate exactly 6 to 8 vision question-answer pairs related to the image caption. Each question should have four options (A, B, C, D), with exactly one correct answer. The questions should be diverse in type and complexity, ensuring a range of reasoning skills (e.g., “when,” “why,” “what,” “how,” “does/do,” “is/are,” “where,” “who,” “which one,” etc.). Each question should require integrating and analyzing the overall information from the image caption, covering various aspects of the image caption content.

The image caption is as follows: <caption>

##### Task 2: Visual Question Answering

You MUST ONLY output valid JSON. Generate nothing but JSON! The JSON must have the structure:

{

"id": " ", "QA Pairs": [

{

"Question": " ", "Answer": " "

},

... ]

}

Please generate exactly 6 to 8 vision question-answer pairs based on the image caption. Incorporate a variety of question types—such as “what,” “why,” “how,” “does/do,” “is/are,” “where,” “who,” and “which,”—to explore different aspects and details of the image. Ensure responses incorporate diverse and authentic language, steering clear of verbatim phrases from the image caption. Address diverse elements of the image by examining chronological sequences, spatial relationships, causal links, and inferential insights to showcase a comprehensive understanding.

The image caption is as follows: <caption>

##### Task 3: Complex Reasoning

You MUST ONLY output valid JSON. Generate nothing but JSON! The JSON must have the structure:

{

"id": " ", "QA Pairs": [

{

"Question": " ", "Answer": " "

},

... ]

}

Design a collection of intricate reasoning challenges based on the image caption. Each problem should present a unique scenario requiring thorough analysis, such as multi-step deductions, hypothetical reasoning, complex problem-solving, or cause-and-effect evaluations. Provide comprehensive, step-by-step solutions for each problem, clearly outlining the logical progression and inference methods used. Ensure the challenges vary in format and complexity, promoting deep understanding and stimulating intellectual engagement.

The image caption is as follows: <caption>

### D Synthetic Data Quality Evaluation

In this section, we evaluate the quality and scalability of the synthetic data produced by Unicorn. Since Unicorn is designed as an image-free data synthesis pipeline, our analysis focuses on three aspects that are directly relevant to large-scale text-only multimodal supervision: cost efficiency, length distribution, and lexical diversity.

Cost Analysis. Unicorn avoids the collection, generation, and storage of real or synthetic images. As a result, the major cost of data construction comes from text-only LLM rewriting, instruction synthesis, and representation extraction. As shown in Table 5, constructing Unicorn-1.2M requires an API cost of

- Table 5: Construction cost of Unicorn-1.2M.

Dataset API/10−3$ Time / Day Storage / GB Unicorn-1.2M 0.30 12 4

- Table 6: Diversity statistics of Unicorn-1.2M.

Dataset Scale TTR ↑ Entropy ↑ Unicorn-1.2M 1,243,195 0.68 6.07

approximately $0.0003 per sample, 12 days of generation time, and 4 GB of storage. These numbers indicate that the image-free design substantially reduces the practical overhead of large-scale multimodal data construction. In particular, because Unicorn stores textual data and synthetic representations rather than image files, its storage requirement remains lightweight even at the million-sample scale. This makes the proposed pipeline feasible for scalable VLM pretraining without relying on expensive image acquisition or image-conditioned annotation.

Length Distribution. We further analyze the length distribution of Unicorn-1.2M to examine whether the generated dense captions provide sufficiently rich visual semantics. The caption lengths form a concentrated and smooth distribution rather than being dominated by very short or excessively long samples. The average length is around 129 words, with a minimum length of 51 words and a maximum length of 483 words. This suggests that the semantic densification stage effectively expands sparse caption seeds into detailed visual descriptions while maintaining a controlled length range. Therefore, Unicorn1.2M provides richer supervision than short caption templates, yet avoids overly verbose descriptions that may introduce irrelevant textual noise.

Diversity Analysis. To assess lexical diversity, we compute two standard statistics: Type-Token Ratio (TTR) and token-level entropy. As shown in Table 6, Unicorn-1.2M achieves a TTR of 0.68 and an entropy of 6.07. These results indicate that the generated captions contain a diverse vocabulary and are not limited to repetitive template-like expressions. This diversity mainly comes from two sources. First, the open-domain caption seeds cover a broad range of everyday visual scenarios. Second, the domain-specific knowledge seeds introduce fine-grained concepts from specialized domains. Together, these two sources allow Unicorn-1.2M to provide both broad semantic coverage and detailed visual concepts, which are important for constructing scalable text-only multimodal supervision.

### E More Experiments

##### E.1 Setting

Training Setting. We build Unicorn on top of Llama-3-8B-Instruct. The visual input is connected to the language model through a two-layer MLP projector with GELU activation. Instead of using image patch features during training, Unicorn takes the synthetic image representations generated by our text-only pipeline as pseudo-visual tokens, and maps them into the LLM feature space through the projector. Training is conducted in two stages. In the first stage, we perform modality-alignment pretraining on Unicorn-1.2M for one epoch, where only the projector is optimized, and the LLM backbone is frozen. The learning rate is set to 5 × 10−4. In the second stage, we initialize the projector from the first stage and perform full-parameter instruction tuning on Unicorn-471K-Instruction for one epoch, with the learning rate reduced to 1 × 10−5. All experiments are run on 8 NVIDIA A100 GPUs. The complete training process takes approximately 27 hours.

Evaluation Setting. We evaluate Unicorn on three groups of multimodal benchmarks. The general perception group includes MME (Fu et al., 2025), MMStar(Chen et al., 2024b), ScienceQA-Image(Lu et al., 2022), and RealWorldQA. The complex reasoning group includes MMMU(Yue et al., 2024), MMMUPro(Yue et al., 2025), VisuLogic(Xu et al., 2025), and LogicVista(Xiao et al., 2024). The hallucination evaluation group includes CRPE(Wang et al., 2024), POPE(Li et al., 2023), and HallusionBench(Guan

- Table 7: Effect of pretraining data scale. We vary the proportion of Unicorn-1.2M used for modality-alignment pretraining while keeping the full 417K instruction-tuning data fixed.

Pretrain Ratio

General Reasoning Hallucination

Avg. ↑ MME MMStar SQA RealWorldQA MMMU MMMU-P VisuLogic LogicVista CRPE POPE HallBench

25% 55.96 33.13 65.18 38.92 32.84 29.63 23.40 25.76 38.04 60.83 38.92 40.24 50% 58.34 34.27 67.42 40.91 35.03 32.01 25.10 27.81 40.62 62.94 41.18 42.33 75% 59.43 34.87 68.17 41.78 36.12 33.18 26.10 28.77 41.58 63.67 42.29 43.27 100% 60.24 35.13 68.81 42.35 36.87 34.05 26.80 29.53 42.32 64.21 43.01 43.94

- Table 8: Effect of instruction-tuning data scale. We vary the proportion of Unicorn-471K-Instruction used for instruction tuning while keeping the full 1.2M pretraining data fixed.

General Reasoning Hallucination

SFT Ratio

Avg. ↑ MME MMStar SQA RealWorldQA MMMU MMMU-P VisuLogic LogicVista CRPE POPE HallBench

30% 50.96 30.80 60.73 35.28 25.84 22.73 18.40 20.16 32.45 55.88 34.12 35.21 60% 56.42 33.40 65.92 39.15 31.76 29.21 23.60 25.89 38.07 61.53 39.48 40.40 100% 60.24 35.13 68.81 42.35 36.87 34.05 26.80 29.53 42.32 64.21 43.01 43.94

et al., 2024). For consistency across benchmarks, we report accuracy as the main evaluation metric.

##### E.2 Scale Analysis

Effect of Pretraining Scale. Table 7 shows that Unicorn benefits consistently from larger text-only pretraining data. The average score increases from 40.24 at 25% data to 42.33 at 50%, 43.27 at 75%, and 43.94 at 100%. The gains exhibit diminishing returns, with larger improvement from 25% to 50% and smaller improvement from 75% to 100%, which is consistent with standard scaling behavior in pretraining. Importantly, the improvement appears across all benchmark groups, including general understanding, reasoning, and hallucination evaluation. This suggests that representation-level visual supervision is scalable: once text is made visually specific and transferred into the visual-compatible space, increasing text-only synthetic data continues to improve VLM performance.

Effect of SFT Scale. Table 8 shows the effect of the instruction-tuning scale. Increasing the SFT ratio from 30% to 60% improves the average score from 35.21 to 40.40, and using the full SFT set further improves it to 43.94. Compared with the pretraining scale, the SFT scale has a larger impact on instruction-following and reasoning benchmarks, suggesting that sufficient instruction data is critical for adapting synthetic visual representations to downstream VQA-style interactions.

### F Synthetic Data Examples

This section provides representative examples from the two synthetic datasets constructed by Unicorn. We show three examples from Unicorn-1.2M for pretraining and three examples from Unicorn-471KInstruction for instruction tuning. Each pretraining sample consists of a dense visual caption and a captioning-style conversation, while each instruction-tuning sample contains multiple question-answer turns derived from the corresponding dense caption.

##### F.1 Unicorn-1.2M Pretraining Example 1

- ID: 910010 Caption. The person is standing next to a structure that is primarily composed of a metal frame, which is sleek and robust, with clean, straight lines and a polished finish that reflects ambient light. The frame supports an inset beveled glass top, which is rectangular in shape and has a slightly angled edge that creates a subtle, refined border. The glass is transparent and free of noticeable imperfections, allowing a clear view through its surface. The combination of the metal frame and beveled glass top gives the structure a modern and elegant appearance.

Conversation. Human: Provide a detailed and rich narrative explaining each component of the image. <image>

GPT: The person is standing next to a structure that is primarily composed of a metal frame, which is sleek and robust, with clean, straight lines and a polished finish that reflects ambient light. The frame supports an inset beveled glass top, which is rectangular in shape and has a slightly angled edge that creates a subtle, refined border. The glass is transparent and free of noticeable imperfections, allowing a clear view through its surface. The combination of the metal frame and beveled glass top gives the structure a modern and elegant appearance.

##### Pretraining Example 2

- ID: 910011 Caption. A mature gum tree, approximately one meter in diameter at the base and with a smooth, grayish-bark, has fallen across a residential property. The tree’s trunk and a significant portion of its branches have come to rest on both a person and the roof of a house. The person is partially obscured by the tree, with their lower body visible beneath the trunk, while the roof shows signs of damage from the impact, with shingles dislodged and a section of the structure visibly deformed. The surrounding area is cluttered with broken branches and leaves, and the ground is marked by the uprooted tree’s exposed roots. Conversation. Human: Provide a comprehensive and detailed description of the image, emphasizing narrative and visuals. <image> GPT: A mature gum tree, approximately one meter in diameter at the base and with a smooth, grayish bark, has fallen across a residential property. The tree’s trunk and a significant portion of its branches have come to rest on both a person and the roof of a house. The person is partially obscured by the tree, with their lower body visible beneath the trunk, while the roof shows signs of damage from the impact, with shingles dislodged and a section of the structure visibly deformed. The surrounding area is cluttered with broken branches and leaves, and the ground is marked by the uprooted tree’s exposed roots.

Pretraining Example 3

- ID: 910012 Caption. A blue and silver train, consisting of six carriages, is stationary on the platform at the railway station. The train’s exterior is sleek, with a series of large, rectangular windows spaced evenly along the length of each carriage. The doors are aligned with the platform’s edge, and a digital display above the doors shows the destination and next stop. The platform is well-lit by overhead lights, and yellow safety lines are painted on the ground to guide passengers. A few passengers are visible, standing near the train, some consulting their phones or waiting with luggage. The station’s architecture includes a modern, glass canopy that extends over the platform, providing shelter from the elements. Conversation. Human: <image> Offer a complete visual description of the photo with an emphasis on details.

GPT: A blue and silver train, consisting of six carriages, is stationary on the platform at the railway station. The train’s exterior is sleek, with a series of large, rectangular windows spaced evenly along the length of each carriage. The doors are aligned with the platform’s edge, and a digital display above the doors shows the destination and the next stop. The platform is well-lit by overhead lights, and yellow safety lines are painted on the ground to guide passengers. A few passengers are visible, standing near the train, some consulting their phones or waiting with luggage. The station’s architecture includes a modern, glass canopy that extends over the platform, providing shelter from the elements.

##### F.2 Unicorn-471M-Instruction Instruction-Tuning Example 1: Multiple Choice

ID: 337637

Caption. The beekeeper, dressed in a full protective suit with a white, mesh-covered helmet, reaches into the wooden hive box to extract a rectangular frame. The frame, approximately 18 inches long and 9 inches wide, is made of light-colored wood and is filled with hexagonal honeycomb cells. Each cell is meticulously constructed and some are filled with honey, while others contain developing bee larvae. The surface of the frame is covered with a dense layer of bees, their bodies varying in color from golden yellow to dark brown, with distinct black stripes. The bees are moving steadily, some clustering around the cells while others are engaged in various tasks. The beekeeper’s gloved hands are carefully handling the frame to avoid disturbing the bees excessively. The hive box, constructed of weathered, light brown wood, is set on a wooden stand to elevate it off the ground. The surrounding area is a grassy field with scattered wildflowers, providing a natural habitat for the bees.

Conversation. Human: What is the beekeeper wearing to protect himself?

- A: A light jacket and jeans
- B: A full protective suit with a white, mesh-covered helmet
- C: A raincoat and boots

- D: A beekeeper’s hat and gloves <image> GPT: B Human: What is the beekeeper doing with the frame?

- A: Painting it
- B: Extracting it from the hive box
- C: Cleaning it
- D: Installing it in a new hive GPT: B Human: What is the approximate size of the frame the beekeeper is handling?

- A: 12 inches long and 6 inches wide
- B: 18 inches long and 9 inches wide
- C: 24 inches long and 12 inches wide
- D: 15 inches long and 7 inches wide GPT: B Human: What are the hexagonal cells in the frame used for?

- A: Storing pollen and nectar
- B: Housing developing bee larvae and storing honey
- C: Providing structural support
- D: Protecting the bees from predators

- GPT: B Human: What is the color of the wood used to construct the hive box?

- A: Dark brown
- B: Black
- C: Weathered, light brown
- D: White

- GPT: C

##### Instruction-Tuning Example 2: Visual Question Answering

ID: 567214

Caption. The serial number label on the back of a rectangular electronic device is prominently displayed. The device itself has a flat, matte surface with a uniform dark gray color. Its rectangular shape measures approximately 10 inches in length and 6 inches in width. The back panel is smooth and unadorned except for the label, which is affixed in the center. The label is a standard white sticker with black text, measuring about 2 inches by 1 inch. The text on the label is clearly printed and includes a series of alphanumeric characters, specifically “SN1234567890ABCD.” The label also features a small barcode below the serial number, which spans about 1.5 inches in length. The edges of the device are slightly rounded, and there are no visible screws or other fasteners on the back panel.

Conversation. Human: What is prominently displayed on the back of the device? <image> GPT: The serial number label is prominently displayed on the back of the device. Human: Why is the label placed in the center of the back panel? GPT: The label is placed in the center of the back panel to ensure it is easily visible and accessible for identification and reference. Human: How does the device’s surface feel to the touch? GPT: The device’s surface feels flat and matte, with a uniform dark gray color. Human: Does the back panel have any visible screws or fasteners? GPT: No, the back panel is smooth and unadorned except for the label, and there are no visible screws or other fasteners. Human: Is the serial number label a standard white sticker? GPT: Yes, the serial number label is a standard white sticker with black text. Human: Where is the barcode located on the label? GPT: The barcode is located below the serial number on the label. Human: Who might use the serial number and barcode on the device? GPT: The serial number and barcode might be used by manufacturers, service centers, and customers for tracking, warranty claims, and product identification. Human: Which dimensions does the label have? GPT: The label measures about 2 inches by 1 inch.

##### Instruction-Tuning Example 3: Complex Reasoning

ID: 407133

Caption. A boy, approximately 8 years old, is running in shallow water at a beach. He is wearing blue and green swim trunks with a pattern of small waves. His skin is lightly tanned, and his hair, which is light brown, is damp and slightly tousled from the water. His arms are swinging at his sides in a natural running motion, and his legs are spread apart, with one foot just breaking the surface of the water and the other submerged, kicking up a small splash. The water around him is clear and calm, with a light ripple from his movement. The beach sand is visible beneath the water, and the horizon shows a clear blue sky with a few fluffy white clouds.

Conversation. Human: Based on the image caption, what can you infer about the weather conditions at the beach? <image>

GPT: The weather conditions at the beach can be inferred to be sunny and pleasant. The caption mentions a clear blue sky with a few fluffy white clouds, which indicates a sunny day with minimal cloud cover. This type of weather is typically associated with warm and comfortable temperatures, ideal for beach activities like swimming and running in the water.

Human: What evidence in the caption suggests that the boy has been in the water for some time?

GPT: The caption provides several pieces of evidence suggesting that the boy has been in the water for some time. His hair is described as damp and slightly tousled from the water, indicating that he has been in the water recently and possibly for a while. Additionally, his skin is lightly tanned, which could suggest he has been exposed to the sun for a period, likely while engaging in water activities.

Human: If the boy is running in shallow water, what can you deduce about the depth of the water and the safety of the beach?

GPT: The caption states that the boy is running in shallow water, which suggests that the water is not deep enough to pose a significant risk of drowning. The fact that he can run and his legs are spread apart, with one foot breaking the surface and the other submerged, indicates that the water is likely only a few inches to a foot deep. This shallow depth, combined with the clear and calm water and visible sand beneath, suggests that the beach is relatively safe for children to play in.

### G Qualitative Analysis

Fig. 7 presents qualitative examples of Unicorn on two types of visual question-answering tasks. The left part shows examples from ScienceQA, while the right part shows examples from our fine-grained iNaturalist-VQA test set.

ScienceQA. The ScienceQA examples cover different forms of visual understanding, including chemical structure recognition, object/category identification, commonsense reasoning, and map-based knowledge reasoning. For instance, Unicorn correctly identifies the molecular formula from a visual chemical structure, selects the mammal from candidate animals, infers why a child looks at the stars based on the astronaut context, and answers a geography question using the map cue. These cases suggest that the model is not merely relying on language priors, but can integrate visual evidence with textual options to produce the correct answer.

iNaturalist-VQA. To further evaluate fine-grained visual recognition, we construct iNaturalist-VQA, a fine-grained animal and plant VQA test set. Compared with general VQA examples, this setting is more challenging because the candidate categories often belong to visually similar biological groups, requiring the model to capture subtle visual cues such as body shape, texture, color patterns, and local structures. As shown in Fig. 7, Unicorn correctly distinguishes categories such as moths, lizards, birds, marine mammals, and plants. These results indicate that the synthetic supervision produced by Unicorn preserves useful fine-grained visual semantics and can support recognition beyond coarse object categories.

###### ScienceQA

###### iNaturallist-VQA

Questions: Select the chemical formula for this molecule.

Questions: Select the mammal below.

Questions: What is the image mainly about?

Questions: What is the image mainly about?

[Figure 36]

###### Options:

###### Options:

###### Options:

[Figure 37]

Options:

[Figure 38]

- (A) Bull Shark
- (B) Yak
- (C) Shoebill
- (D) Eastern Newt Prediction: (B) Ground Truth: (B)

- (A) Common Mime Swallowtail
- (B) Lunate Zale Moth
- (C) Red-headed Ash Borer
- (D) Evergreen Bagworm

- (A) West Gulf Rough-scaled Lizard
- (B) Glossy Snake
- (C) Turniptail Gecko
- (D) Tropical House Gecko

- (A) H3S2
- (B) He2S2
- (C) H2S
- (D) H2SN Prediction: (C) Ground Truth: (C)

[Figure 39]

###### Prediction: (B) Ground Truth: (B)

###### Prediction: (A) Ground Truth: (A)

Questions: Complete the statement. Ethane is ().

Questions: What is the image mainly about?

Questions: What is the image mainly about?

Questions: Select the chemical formula for this molecule.

[Figure 40]

[Figure 41]

Options:

Options:

[Figure 42]

###### Options:

[Figure 43]

- (A) Bridled Titmouse
- (B) Scarlet Minivet
- (C) Black-bellied Plover
- (D) Azure-crowned Hummingbird Prediction: (A) Ground Truth: (A)

- (A) Gray Petaltail
- (B) Handmaiden Moth
- (C) Spiny Shield Bug
- (D) Brimstone Moth

###### Options:

- (A) An elementary substance
- (B) A compound

- (A) Cougar
- (B) Impala Prediction: (B) Ground Truth: (B)

Prediction: (B) Ground Truth: (B)

###### Prediction: (B) Ground Truth: (B)

Questions: Why did young Mae look at the stars?

Questions: What is the capital of Idaho?

Questions: What is the image mainly about?

Questions: What is the image mainly about?

[Figure 44]

[Figure 45]

###### Options:

###### Options:

Options:

###### Options:

[Figure 46]

[Figure 47]

- (A) She wanted to watch the weather.
- (B) She wanted to travel to outer space. Prediction: (B) Ground Truth: (B)

- (A) Pierre
- (B) Nampa
- (C) Santa Fe
- (D) Boise Prediction: (D) Ground Truth: (D)

- (A) Cape Porcupine
- (B) Blue Monkey
- (C) Southern Fur Seal
- (D) Asian Jackal

- (A) Swamp Lousewort
- (B) Spring Starflower
- (C) Rice Cutgrass
- (D) Shell ginger Prediction: (B) Ground Truth: (B)

###### Prediction: (C) Ground Truth: (C)

###### Figure 7: Examples of Unicorn-8B’s correct responses on the ScienceQA and iNaturallist-VQA.

