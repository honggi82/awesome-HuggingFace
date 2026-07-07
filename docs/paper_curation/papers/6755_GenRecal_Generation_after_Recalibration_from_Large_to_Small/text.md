# arXiv:2506.15681v4[cs.CL]25Jun2026

## GenRecal: Generation after Recalibration from Large to Small Vision-Language Models

Byung-Kwan Lee† Ryo Hachiuma Yong Man Ro

NVIDIA NVIDIA KAIST byungkwanl@nvidia.com rhachiuma@nvidia.com ymro@kaist.ac.kr

Yu-Chiang Frank Wang Yueh-Hua Wu

NVIDIA NVIDIA frankwang@nvidia.com krisw@nvidia.com

Image and Text Prompt

MM-Vet

[Figure 1]

74 72 70 68 66 64 62

73.2

<image> A gear is one of a set of toothed wheels that work together to alter the relation between the speed of a driving mechanism

70.4

Teacher VLMs

InternVL2.5-78B (Qwen2.5 Tokenizer)

67.8

F E

|[…, 32, 14448, 374, 825, 315, 264, 738, 315, 25507, 291, 22696, 429, 975, 3786, 311, 11596, 279, 12687, 1948, 279, 4628, 315, 264, 9842, 16953, 13]|
|---|

65.9

64.2

Qwen2-VL-72B (Qwen2 Tokenizer)

C D

|[…, 32, 14448, 374, 825, 315, 264, 738, 315, 25507, 291, 22696, 429, 975, 3786, 311, 11596, 279, 12687, 1948, 279, 4628, 315, 264, 9842, 16953, 13]|
|---|

62.0

InternVL2-76B (Llama-3 Tokenizer)

60

A B C D E F

|[…, 32, 14787, 374, 832, 315, 264, 743, 315, 26588, 291, 23529, 430, 990,<br><br>3871, 311, 11857, 279, 12976, 1990, 279, 4732, 315, 264, 10043, 17383, 13]|
|---|

- A: Qwen2-VL-7B (Baseline)

- B: Qwen2-VL-7B (SFT)

- C: Qwen2-VL-72B → Qwen2-VL-7B (Trad. Distill.)

- D: Qwen2-VL-72B → Qwen2-VL-7B (GenRecal)

- E: InternVL2.5-78B → Qwen2-VL-7B (GenRecal)

- F: InternVL2.5-78B → InternVL2.5-8B (GenRecal)

Student VLMs

InternVL2.5-8B (InternLM2.5 Tokenizer)

|[…, 290, 14586, 505, 959, 446, 395, 871, 446, 25951, 422, 23000, 560, 1115, 3942, 442, 11750, 410, 12837, 2100, 410, 4790, 446, 395, 9978, 17090, 281]|
|---|

Qwen2-VL-7B (Qwen2 Tokenizer)

|[…, 32, 14448, 374, 825, 315, 264, 738, 315, 25507, 291, 22696, 429, 975, 3786, 311, 11596, 279, 12687, 1948, 279, 4628, 315, 264, 9842, 16953, 13]|
|---|

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

| |
|---|

| |
|---|

Trad. Distill. GenRecal

Trad. Distill. GenRecal

Fig. 1: (Left) Visualizing the token indices of a given image and text prompt and representing the possibility of distillation among various VLM pair combinations, comparing traditional distillation with our proposed distillation framework, GenRecal. Note that the parentheses mean each VLM’s LLM tokenizer, ‘...’ indicates the placement of image features, and the number of these features varies depending on the image embedding strategy. (Right) Comparing the performance of a challenging evaluation benchmark, MM-Vet [136], with [A] baseline, [B] SFT on the baseline, [C] traditional distillation and [D] GenRecal from same token types of large VLMs, and GenRecal with more powerful [E] large and [F] small VLMs.

Abstract. Recent advancements in vision-language models (VLMs) have

leveraged large language models (LLMs) to achieve performance on par with closed-source systems like GPT-4V. However, deploying these models in real-world scenarios, particularly on resource-constrained devices, remains challenging due to their substantial computational demands. This has spurred interest in distilling knowledge from large VLMs into smaller, more efficient counterparts. A key challenge arises here from the diversity of VLM architectures, which are built on different LLMs and employ varying token types—differing in vocabulary size, token splits,

† Project Lead

[Figure 6]

[Figure 7]

Changing Teacher VLMs for Distillation

MMMU

[Figure 8]

[Figure 9]

70

###### GenRecal

GPT-4o

MMB

[Figure 10]

[Figure 11]

InternVL2.5-78B → InternVL2.5-8B

Qwen2-VL

60

Accuracy(%)

InternVL2.5

MM-Vet

50

Molmo

MMMU

40

LLaVA-OneVision

NVLM-72B → InternVL2.5-8B InternVL2-76B → InternVL2.5-8B Qwen2-VL-72B → InternVL2.5-8B InternVL2.5-78B → InternVL2.5-8B

MMMU-Pro

30

0 20 40 60 80

1 2 4 8 72 78

7

…

Accuracy (%)

Model Size (B)

- Fig. 2: (Left) Comparison of the challenging benchmark performances, MMB [83], MM-Vet [136], MMMU [138], and MMMU-Pro [139] by changing large VLMs. The more powerful large VLMs we select, the greater the performance improvement we can achieve. (Right) Comparing the performance of the challenging benchmark: MMMU [138], with GenRecal and various VLMs across model sizes. Note that all the experiments in Fig. 1 and Fig. 2 are conducted on the equal training dataset.

and token index ordering. To address this challenge of limitation to a specific VLM type, we present Generation after Recalibration (GenRecal), a general-purpose distillation framework for VLMs. GenRecal incorporates a Recalibrator that aligns and adapts feature representations between heterogeneous VLMs, enabling effective knowledge transfer across different types of VLMs. Through extensive experiments on multiple challenging benchmarks, we demonstrate that GenRecal significantly improves baseline performances, eventually outperforming largescale open- and closed-source VLMs.

Keywords: Cross-Tokenizer · Model Distillation · Efficient AI

### 1 Introduction

Vision-language models (VLMs) have emerged as powerful tools for understanding and processing multimodal information, enabling tasks such as image captioning and visual question answering [75,113]. Recent advancements in VLMs [3, 74, 80] have leveraged large-scale language models (LLMs) to enhance reasoning and generation capabilities by incorporating extensive textual knowledge [98, 107]. In pursuit of superior performance, state-of-the-art VLMs [19,26,28,72,125] now integrate scaled-up LLMs with up to 72B parameters, achieving results comparable to proprietary models such as GPT-4V [93] and Claude-3.5 Sonnet [4].

However, the increasing scale of recent VLMs introduces substantial computational overhead, limiting their practicality in real-world scenarios—particularly for on-device deployment. To mitigate this challenge, recent work has explored distillation techniques [10,31,105] that transfer knowledge from large (teacher) VLMs to smaller (student) ones. Yet, existing distillation methods suffer from a fundamental limitation: they typically assume that teacher and student produce sequences of equal token length, enabling token-level distance metrics such as KL divergence. This assumption breaks down when the two models use different tokenizers or adopt different image-splitting mechanisms (see Appendix A), mak-

|[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>Small<br><br>Large<br><br>Recalibrator<br><br>[Figure 15]<br><br>VLM-body<br><br>𝑞𝑠 𝑎𝑠<br><br>VLM-body<br><br>𝑞𝑙 𝑎𝑙<br><br>|[Figure 16]<br><br>VLM-head|
|---|
<br><br>VLM-body<br><br>𝑞𝑙 𝑎𝑙<br><br>|[Figure 17]<br><br>VLM-head|
|---|
<br><br>Alignment<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]|
|---|

|[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>Small<br><br>Large<br><br>Recalibrator<br><br>[Figure 27]<br><br>VLM-body<br><br>𝑞𝑠 𝑎𝑠<br><br>VLM-body<br><br>𝑞𝑙 𝑎𝑙<br><br>|[Figure 28]<br><br>VLM-head|
|---|
<br><br>VLM-body<br><br>𝑞𝑙 𝑎𝑙<br><br>|[Figure 29]<br><br>VLM-head|
|---|
<br><br>Distillation<br><br>[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]|
|---|

###### Recalibrator

|Proj-post|
|---|

|Decoder Block|
|---|

Body

|Decoder Block|
|---|

|Proj-pre|
|---|

(a) Stage1: Alignment (b) Stage2: Distillation

(c) Recalibrator

- Fig. 3: Overview of the GenRecal architecture and its training stages. We denote the question and answer tokens from the small VLM as qs and as, and those from the large VLM as ql and al. For simplicity, the vision encoder and image-token projector are omitted from the illustration. Note that Recalibrator is used only during training as a bridge between heterogeneous VLMs. During inference, Recalibrator is removed, leaving the small VLM’s architecture unchanged and no extra computational cost.

ing the distillation process infeasible. Consequently, the range of viable teacherstudent VLM pairs becomes severely restricted (see Appendix B). In other words, current distillation approaches only operate when the teacher and student share identical vocabulary sizes, token splits, and token-index-ordering schemes (collectively referred to as token types), as illustrated on the left side of Fig. 1. These differences naturally lead to mismatched input or output token lengths across VLMs, preventing effective distillation. Note that even VLMs within the same family may employ different token types, further hindering compatibility.

To overcome this limitation, we propose Generation after Recalibration (GenRecal), a general-purpose distillation framework for token types-agnostic VLM distillation. At its core, GenRecal introduces a Recalibrator that aligns and adapts the feature representations of small VLMs with those of large VLMs. This design is inspired by prior studies demonstrating that word embeddings from different models can be linearly mapped into a shared embedding space [24,91,106]. Unlike approaches that align only word embeddings, Recalibrator performs joint visual–linguistic alignment on the hidden representations before the language head. By projecting small VLM features into large VLMs’ feature space as shared representation, Recalibrator effectively bridges the gap between the large and small VLMs, enabling general-purpose knowledge transfer.

To demonstrate the necessity of GenRecal, we first compare its performance with that of traditional distillation, as shown in experiment types C and D of Fig. 1. Although both experiments employ the same token types of VLMs—Qwen2VL-72B as the teacher and Qwen2-VL-7B as the student—GenRecal consistently outperforms the traditional distillation method implemented by LLaVAKD [10]. This result highlights that aligning feature representations between large and small VLMs is crucial for minimizing information loss during distillation, even when the models are homogeneous. Moreover, when we replace the large VLM from Qwen2-VL-72B with the more powerful InternVL2.5-78B, GenRecal achieves enhanced performance. Subsequently, replacing the small VLM with InternVL2.5-8B leads to an even stronger model, as summarized on the right

|[Figure 36]<br><br>Recalibrator<br><br>VLM-body<br><br>|𝑞𝑙|
|---|
<br><br>|𝑎𝑙|
|---|
<br><br>|[Figure 37]<br><br>VLM-head|
|---|
<br><br>VLM-body<br><br>|𝑞𝑙|
|---|
<br><br>|𝑎𝑙|
|---|
<br><br>|[Figure 38]<br><br>VLM-head|
|---|
<br><br>Alignment<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>|
|---|

###### Recalibrator

|Proj-post|
|---|

|Decoder Block|
|---|

Body

|Decoder Block|
|---|

|Proj-pre|
|---|

- Fig. 4: Overview of regularization simultaneously done with first training stagealignment. Note that its propagation rule is different with Fig. 3(c) where Recalibrator takes small VLM’s tokens for the question and large VLM’s tokens for the answer. In contrast, in this figure, only large VLM’s tokens (question and answer) are fed into Recalibrator. Basically, Rec-body has the hidden dimension of the small VLM, so we should change this hidden dimension of large VLM’s tokens to that of small VLM’s ones via Proj-pre when the large VLM’s tokens are fed into Recalibrator.

side of Fig. 1. These experiments highlight the significance of GenRecal because traditional methods cannot handle VLM pairs with different token types.

Our contributions can be summarized as follows:

- – Token Types-agnostic Recalibration: GenRecal employs the Recalibrator to align and adapt the feature representations of large and small VLMs, enabling general-purpose distillation across token types that differ in vocabulary size, token splits, and token-index ordering.
- – Broad Applicability: GenRecal is compatible with a wide range of VLM architectures across different model sizes, overcoming the challenge of selecting large VLMs that have different token types and demonstrating its practicality for real-world deployment in resource-constrained settings.

### 2 Related Work

Large-scale VLMs such as NVLM-72B [26], Qwen2-VL-72B [125], and InternVL2.578B [19] have recently approached the performance of GPT-4V [93] and Claude-

- 3.5 Sonnet [4] (see Appendix C for the evolution of VLMs). However, they impose a significant computational burden in real-world applications, such as on-device processing. Hence, it is necessary to develop small VLMs for deployment on lightweight devices, and this demand has led to two types of distillation approaches. The first constructs visual instruction tuning datasets [18, 33, 40, 72, 111,126,140,144] from large-scale VLMs and trains small VLMs on these curated datasets. The second performs feature distillation: LLaVA-MoD [105] conducts logit-based distillation on a mixture-of-experts (MoE) [100, 103] architecture. LLaVA-KD [10] also uses logit-based distillation but adopts a three-stage training process in which the trainable parameters differ at each stage to effectively warm up and distill the knowledge. Align-KD [31] distills both vision-encoder

- Algorithm 1 First stage Loss Functions (Alignment)

- 1: Input: (ql, al), (qs, as), gtl (label index)
- 2: [zql, zal] ←VLM-bodyl([ql, al])
- 3: [zqs, zas] ←VLM-bodys([qs, as])
- 4: [rqs, ral] ← Recalibrator([zqs, zal])
- 5: Lar ← CE(VLM-headl(ral), gtl)
- 6: Lkl ← DKL(VLM-headl(zal) | VLM-headl(ral))
- 7: Return: Lar + Lkl

features and decoder logits, while encouraging the student VLM’s first decoder layer’s attention map to resemble that of the large VLM. MoVE-KD [12] employs multiple vision encoders for the large VLM and applies a mixture of LoRA [41] to the small VLM, distilling visual information through visual attention maps. Besides, there are earlier distillation studies (see Appendix C), but they consider neither token-derived nor different-token-type distillation.

However, existing works are largely constrained to VLMs sharing identical token types, including vocabulary size, token segmentation, and token index ordering. When the large and small VLMs employ different tokenizers or image embedding strategies, distillation becomes problematic—KL divergence cannot be computed due to mismatched output token counts, and token indices may no longer correspond to semantically equivalent meaning. Overcoming this limitation would represent a significant advancement, broadening the applicability and robustness of distillation techniques for diverse VLM architectures.

### 3 GenRecal: Generation after Recalibration

#### 3.1 Model Architecture and Components

We briefly illustrate the overall model architecture and training procedure in Fig. 3. GenRecal consists of three main components: a large (teacher) VLM, a small (student) VLM, and a Recalibrator module. For the large VLMs, we select models with over 72B parameters to ensure competitive performance: NVLM-72B (Qwen2-72B) [26], Qwen2-VL-72B (Qwen2-72B) [125], InternVL276B (Llama3-70B) [20], and InternVL2.5-78B (Qwen2.5-72B) [19]. For the small VLMs, we adopt various model sizes, including Qwen2-VL-2B/7B (Qwen2-2B/7B) and InternVL2.5-1B/2B/4B/8B (InternLM2.5-1.8B/7B and Qwen2.5-0.5B/3B/72B). Note that the parentheses denote the LLMs used within each VLM. Throughout this section, we divide a VLM architecture into four key modules: the vision encoder, vision projector, VLM-body, and VLM-head (i.e. language head). Recalibrator is designed to acquire shared feature representations within a common latent space. It consists of two decoder blocks and two projection layers, as depicted in Fig. 3(c). We denote the decoder blocks as Rec-body, and the projection layers as Proj-pre and Proj-post. The decoder blocks, Rec-body, adopt the same structural design as the student VLM’s decoder, while both Proj-pre and Projpost are implemented as single linear layers.

##### Algorithm 2 First stage Regularization (Alignment)

- 1: Input: (ql, al), gtl (label index)
- 2: [zql, zal] ←VLM-bodyl([ql, al])
- 3: [rql, ral] ← Recalibrator([zql, zal])
- 4: Lar ← CE(VLM-headl(ral), gtl)
- 5: Lkl ← DKL(VLM-headl(zal) | VLM-headl(ral))
- 6: Return: Lar + Lkl

##### Algorithm 3 Second stage Loss Functions (Distillation)

- 1: Input: (ql, al), (qs, as), (gtl, gts) (label index)
- 2: [zql, zal] ←VLM-bodyl([ql, al])
- 3: [zqs, zas] ← VLM-bodys([qs, as])
- 4: [rqs, ral] ← Recalibrator([zqs, zal])
- 5: Lar ← CE(VLM-headl(ral), gtl)
- 6: Lar ← Lar + CE(VLM-heads(zas), gts)
- 7: Lkl ← DKL(VLM-headl(zal) | VLM-headl(ral))
- 8: Return: Lar + Lkl

#### 3.2 Design and Propagation of Recalibrator

To construct a shared representation space, we first extract and align hidden features from both the large and small VLMs. Since the hidden space of a large VLM typically encompasses a richer vocabulary and more expressive representations, we regard it as the shared latent space which the small VLM is projected into. However, direct alignment of the two VLMs’ tokens within this space is infeasible due to discrepancies in vocabulary size, token split, and token index ordering. To overcome this challenge, we draw inspiration from the principle of autoregressive modeling, wherein a model predicts answer tokens for the given question tokens.

Specifically, we feed the same question-answer pair into both the large and small VLMs, obtaining their VLM-body outputs as:

] = VLM-bodyl([ql,al]), (l : large) [zq

[zq

,za

l

l

] = VLM-bodys([qs,as]), (l : small)

,za

s

s

(1)

where q and a denote the question and answer tokens produced by each model’s vision encoder, vision projector, and word embedding [90]. We then extract the question features zq

from the large VLM, and concatenate them to form a joint sequence: [zq

from the small VLM and the answer features za

s

l

], which is then passed through Recalibrator. Because directly embedding both features into a shared space is infeasible due to different token types, we instead design an autoregressive loss that predicts the answer token index of the large VLM given the question token index of the small VLM. This loss encourages Recalibrator to effectively project the small VLM’s features into the large VLM’s latent space.

,za

s

l

However, the hidden dimensions of large and small VLMs often differ. To address this, in Recalibrator, we first apply Proj-pre to za

, aligning its dimen-

l

- Table 1: Evaluation of standard model size open-source VLMs and GenRecal on several challenging vision-language benchmarks: AI2D [45], ChartQA [88], MathVista [85], MMB [83], MMBCN [83], MM-Vet [136], MMMU [138], MMMU-Pro [139], BLINK [32], SEED-2-Plus [73], and RealWorldQA (RWQA). We use the notations of ‘XX-GenRecal (YY )’ where ‘XX’ and ‘YY ’ denote the student (small) and teacher (large) VLM.

VLMs AI2D ChartQA MathVista MMB MMBCN MM-Vet MMMU MMMU-Pro BLINK SEED-2-Plus RWQA

Cambrian-1-8B [118] 73.0 73.3 49.0 75.9 - 48.0 42.7 - 44.9 59.7 60.0 Cambrian-1-13B [118] 73.6 73.8 48.0 75.7 - 48.9 40.0 - 43.1 60.0 58.6 Eagle-8B [104] 76.1 80.1 52.7 75.9 - 33.3 43.8 - 22.4 57.5 63.8 Eagle-13B [104] 74.0 77.6 54.4 75.7 - 42.6 41.6 - 21.8 60.2 62.9 VILA1.5-8B [78] 58.8 - 37.3 75.3 69.9 43.2 38.6 - 39.5 45.2 43.4 VILA1.5-13B [78] 69.9 - 42.5 74.9 66.3 44.3 37.9 - 48.1 50.2 53.3 CogVLM2-19B [37] 73.4 81.0 38.6 80.5 - 60.4 44.3 - - 66.0 62.9 LLaVA-OneVision-7B [72] 81.4 80.0 63.2 80.8 - 57.5 48.8 24.1 53.0 65.4 69.9 InternVL2-8B [21] 83.8 83.3 58.3 81.7 81.2 54.2 49.3 29.0 50.9 67.3 64.2

- MiniCPM-V2.5-8B [132] 78.4 - 54.3 77.2 74.2 52.8 45.8 - - 61.4 63.5
- MiniCPM-V2.6-8B [132] 82.1 - 60.6 - - 60.0 49.8 27.2 55.2 65.7 65.0 Qwen2-VL-7B [125] 77.5 83.0 58.2 83.0 80.5 62.0 54.1 30.5 53.8 68.6 68.5 InternVL2.5-8B [19] 84.8 84.8 64.4 84.6 82.6 62.8 56.0 34.3 54.8 69.7 70.1

Qwen2-VL-7B-GenRecal (InternVL2.5-78B) 93.9 95.3 68.8 88.4 87.4 70.4 65.6 49.6 64.3 70.7 79.2 InternVL2.5-8B-GenRecal (InternVL2.5-78B) 93.0 93.6 74.9 89.5 88.2 73.2 68.1 48.8 65.3 72.3 81.4

VILA1.5-3B [78] 57.9 - 31.6 - - 38.8 34.2 - 39.7 41.4 53.2 Phi-3.5-Vision-4B [1] 77.8 81.8 43.9 76.0 66.1 43.2 43.0 19.7 58.3 62.2 53.6 InternVL2-4B [21] 78.9 81.5 58.6 78.6 73.9 51.0 34.3 32.7 46.1 63.9 60.7 InternVL2.5-4B [21] 81.4 84.0 60.5 81.1 79.3 60.6 52.3 32.7 50.8 66.9 64.3

InternVL2.5-4B-GenRecal (InternVL2.5-78B) 90.0 91.1 70.4 88.4 86.2 66.1 58.3 45.9 63.5 69.3 75.2 InternVL2-2B [21] 74.1 76.2 46.3 73.2 70.9 39.5 34.3 18.2 43.8 60.0 57.3 Qwen2-VL-2B [125] 60.2 73.5 43.0 74.9 73.5 49.5 41.1 21.2 45.2 61.2 62.6 Aquila-VL-2B [19] 75.0 76.5 59.0 - - 43.8 47.4 - 34.1 63.0 65.0 InternVL2.5-2B [19] 74.9 79.2 51.3 74.7 71.9 60.8 43.6 23.7 44.0 60.0 60.1 Qwen2-VL-2B-GenRecal (InternVL2.5-78B) 89.1 90.9 60.5 82.9 81.0 57.3 52.9 37.5 53.4 65.5 72.8 InternVL2.5-2B-GenRecal (InternVL2.5-78B) 91.8 90.7 62.5 84.1 80.4 61.2 59.2 36.6 55.5 67.3 73.0 LLaVA-OneVision-0.5B [72] 57.1 61.4 34.8 61.6 55.5 32.2 31.4 - 52.1 45.7 55.6 InternVL2-1B [21] 64.1 72.9 37.7 65.4 60.7 32.7 36.7 14.8 38.6 54.3 50.3 InternVL2.5-1B [19] 69.3 75.9 43.2 70.7 66.3 48.8 40.9 19.4 42.0 59.0 57.5 InternVL2.5-1B-GenRecal (InternVL2.5-78B) 82.9 89.4 56.5 80.1 74.4 54.4 45.6 28.7 48.5 65.1 70.8

sionality to that of zq

. The concatenated sequence [zq

)] is then propagated through Rec-body decoder blocks. The output of Rec-body is subsequently passed through Proj-post to restore the hidden dimensionality of the large VLM. This forward rule of Recalibrator is illustrated in Fig. 3(c).

,Proj-pre(za

s

s

l

Finally, the answer part of the resulting features [rq

] = Recalibrator([zq

,ra

,za

]) is fed into the VLM-head of the large VLM to compute the autoregressive loss:

s

s

l

l

Lar = CE(VLM-headl(ra

),gtl), (2)

l

where gtl denotes the answer token index from the large VLM. Additionally, we employ a KL divergence loss to further enhance the Recalibrator’s projection capability in the first training stage and to distill knowledge from the teacher in the second training stage. This process enables the large VLM to interpret the hidden features of the small VLM, thus establishing a bridge for general-purpose distillation across heterogeneous token types.

#### 3.3 Training Process

To achieve general-purpose distillation, we adopt a three-stage training process. In the first stage, only Recalibrator is trained while keeping all parameters of both the large and small VLMs frozen, as illustrated in Fig. 3(a). During this stage, we compute both the autoregressive loss and the KL divergence, described in Algorithm 1 where the purple components indicate the trainable parameters. This step primarily serves to align the feature representations between the large and small VLMs. For the first stage, we introduce regularization loss, as illustrated in Fig. 4, which is designed to prevent the feature representations of

- Table 2: (b) Comparison of GenRecal with large-scale open-source and closedsource VLMs on challenging benchmarks: MMB [83], MM-Vet [136], MM-Vet-v2 [137], MMMU [138], MMMU-Pro [139], MMStar [14], AI2D [45], ChartQA [88], SEED-2Plus [73], MathVista [85], BLINK [32], RealWorldQA (RWQA).

VLMs MMB MM-Vet MM-Vet-v2 MMMU MMMU-Pro MMStar AI2D ChartQA SEED-2-Plus MathVista BLINK RWQA

NVLM-72B [26] - 58.9 - 59.7 - 63.7 85.2 86.0 68.4 66.6 48.0 69.9 LLaVA-OneVision-72B [72] 85.8 60.6 - 56.8 31.0 65.8 85.6 83.7 - 67.5 55.4 71.9 Molmo-72B [28] - 61.1 - 54.1 - 63.3 83.4 87.3 - 58.6 - 73.7 Qwen2-VL-72B [125] 86.5 74.0 68.7 64.5 46.2 68.3 88.1 88.3 72.3 70.5 60.5 77.8 InternVL2-76B [21] 86.5 65.7 68.4 62.7 40.0 67.4 87.6 88.4 69.7 65.5 56.8 72.2 InternVL2.5-78B [72] 88.3 72.3 65.5 70.1 48.6 69.5 89.1 88.3 71.3 72.3 63.8 78.7 Claude-3.5-Sonnet [4] 82.6 70.1 71.8 68.3 51.5 65.1 81.2 90.8 71.7 67.7 60.1 60.1 Gemini-1.5-Pro [114] 73.9 64.0 66.9 62.2 46.9 59.1 79.1 87.2 70.8 63.9 59.1 67.5 GPT-4o (0513) [93] 83.4 69.1 71.0 69.1 51.9 64.7 84.6 85.7 72.0 63.8 68.0 75.4

InternVL2.5-8B-GenRecal (NVLM-72B) 88.8 63.9 59.3 60.3 34.7 65.1 88.5 84.3 69.3 67.5 55.6 72.5 InternVL2.5-8B-GenRecal (InternVL2-76B) 89.0 72.4 65.0 64.6 42.8 65.8 92.1 90.4 70.9 71.9 57.7 75.5 InternVL2.5-8B-GenRecal (Qwen2-VL-72B) 89.5 71.4 62.6 65.6 43.8 66.1 92.2 87.5 71.3 71.4 61.4 78.8 InternVL2.5-8B-GenRecal (InternVL2.5-78B) 89.5 73.2 67.2 68.1 48.8 67.8 93.0 93.6 72.3 74.9 65.3 81.4

Recalibrator from drifting too far from those of the large VLM. We use both Algorithm 1 and Algorithm 2 together in the first stage.

In the second stage, we keep using the same loss functions of the first stage and additionally incorporate the small VLM’s own autoregressive loss in Algorithm 3. As depicted in Fig. 3(b), this stage enables knowledge distillation from the large VLM to the small one by training the small VLM’s VLM-body. In the final stage, we remove both Recalibrator and the large VLM, and fine-tune the student VLM. Specifically, we train all parameters of the student VLM except for the vision encoder through supervised fine-tuning (SFT) to further enhance its instruction-following capability.

### 4 Experiments

#### 4.1 Implementation Detail

To ensure reproducibility, we outline three key technical aspects of GenRecal: (a) the structure of the Recalibrator, (b) the projection layers, and (c) training and evaluation details.

- (a) Structure Detail of Recalibrator. This is composed of two transformer decoder blocks (Rec-body) and two projectors (Proj-pre and Proj-post). The decoder blocks follow all configurations of the small VLM’s decoder block, such as the causal mask, hidden dimensions, number of heads, and FFN structure. We use transformer decoder blocks for Rec-body because they naturally process the concatenated student and teacher features in Fig. 3 as a sequence. Moreover, these features must preserve token ordering, so we introduce new positional embeddings (NPE). This design enables effective sequential modeling and representation alignment between heterogeneous VLMs. To this end, we employ an additional RoPE [110] to realign their positional embeddings, and their position IDs are reassigned accordingly. Lastly, we apply an additional layer norm [6] to the output features of the Recalibrator for stable adaptation.
- (b) Projection Layer Design. Two projection layers (Proj-pre and Proj-post) are linear layers introduced for hidden dimension matching. Similar to the role of vision projector that matches the hidden dimensions between a CLIP-like vision encoder and an LLM, Proj-pre matches the teacher features to the student

||68.1|
|---|
<br><br>78B<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|58.3|
|---|
<br><br>|53.7|
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|59.2|
|---|
<br><br>|49.6|
|---|
<br><br>|42.3|
|---|
<br><br>| |
|---|
<br><br>|45.6|
|---|
<br><br>|42.2|
|---|
<br><br>|41.5|
|---|
<br><br>|41.0|
|---|
<br><br>8B 4B 2B<br><br>8B<br><br>4B<br><br>2B<br><br>1B<br><br>Teacher VLM<br><br>StudentVLM|
|---|

StudentVLM

- Fig. 5: Distillation performance on MMMU [138] for various pairings os teacher and student VLM. Each cell indicates the resulting score when using the corresponding teacher (rows) and student (columns) model sizes.

hidden dimension, while Proj-post restores the dimension back to the teacher hidden dimension so that the large VLM-head can be utilized for distillation. This explanation is technically depicted in Fig. 3(c) and Fig. 4.

- (c) Details of Training and Evaluation. By using DeepSpeed engine with ZeRO-3 [99], we train and evaluate GenRecal on 128 NVIDIA A100 80GB GPUs. We use AdamW optimizer [84] and apply a linearly decayed learning rate from 1e-4 to 1e-5 at each training stage. The first and second stages use the entire 9M dataset (see Appendix E for dataset composition and analysis), taking approximately 2–4 hours and 8–11 hours, respectively, depending on model size. The last stage takes 4–6 hours on a 6M dataset obtained by removing general visual question answering samples [72]. For stable training, we handle large batch sizes via gradient accumulation with 16 steps. We use 4 (first and third stages) or 2 (second stage) samples per GPU, leading to a total batch size of 8192 (128×16×4) or 4096 (128×16×2). For evaluation, we remove the Recalibrator and the large VLM, and let the small VLM generate answers using the default generation hyperparameters.
- (d) Dataset Composition. Since the 9M corpus has a decisive impact on the results, we summarize it here. It covers general visual question answering, dense captioning, chart/diagram/document understanding, commonsense knowledge, science & math, and multi-dimensional reasoning, curated from public sources such as LLaVA-OneVision [72], Cambrian [118], Infinity-MM [33], and DenseFusion [76], among others. For analysis we group these into three domainsKnowledge, Science & Math, and Chart & Document—which align with benchmarks such as MMMU [138], MathVista [85], and ChartQA [88], respectively. A full source list and per-domain breakdown are provided in Appendix E.

#### 4.2 Validation and Analysis for GenRecal

Higher Size, Higher Performance. Tab. 1 demonstrates that GenRecal outperforms not only standard-size VLMs but also smaller ones, highlighting that GenRecal generalizes well across model sizes. This table also shows that choosing a higher-performing small VLM such as InternVL2.5-8B [19] substantially

[Figure 44]

###### Measurement & Legends

###### Question

|[Figure 45]<br><br>Recalibrator<br><br>VLM-body<br><br>𝑧𝑠<br><br>𝑟𝑠<br><br>𝑞𝑠|
|---|

|[Figure 46]<br><br>VLM-body<br><br>Recalibrator<br><br>𝑧𝑙<br><br>𝑟𝑙<br><br>𝑞𝑙|
|---|

[Figure 47]

𝑧𝑠

The graph below shows the long-term international migration, UK, 1999-2008. Summarize the information by selecting

𝑟𝑠

and reporting the main features, and

𝑧𝑙

make comparisons where relevant. You should write at least 150 words.

𝑟𝑙

Initial Training Period

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Student: 8B, Teacher: 78B Student: 4B, Teacher: 78B Student: 2B, Teacher: 78B Student: 1B, Teacher: 78B

Final Training Period

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Student: 8B, Teacher: 78B Student: 4B, Teacher: 78B Student: 2B, Teacher: 78B Student: 1B, Teacher: 78B

- Fig. 6: An overview of our training pipeline, illustrating both the question prompt and the measurement/legend annotations (top), followed by t-SNE visualizations (bottom) of teacher and student VLM pairings at the initial and final training stages. The question prompt (upper-left) shows the format of the question, while the measurement and legend box (upper-right) shows key model components to measure. Each scatter plot in the lower panels corresponds to a different combination of teacher and student VLM sizes, capturing how the learned representations evolve from early to later training.

improves the overall distillation results. Furthermore, for a fixed large VLM, selecting a smaller student VLM leads to lower distillation performance, implying that employing more capable small VLMs is crucial for achieving higher distillation performance.

Next, we examine the effect of changing large VLMs to determine whether the aforementioned property holds consistently across different large VLMs. As shown in Fig. 2 and Tab. 2, selecting more powerful large VLMs leads to greater performance improvements. To investigate this further, we use the InternVL2.5 series [19] and experiment with various combinations of large and small VLM sizes, as illustrated in Fig. 5. The results consistently indicate that choosing larger teacher and student VLMs leads to higher distillation performance.

Analysis of Recalibrator. We now focus on the Recalibrator to explore its role in general-purpose distillation. First, we show that the loss curves for training the Recalibrator converge stably across various combinations of large and small VLMs (see Appendix D). Since the Recalibrator losses (‘Recalib(qs,al)’ and ‘Recalib(ql,al)’) are minimized well relative to the perplexity of ‘SmallVLM(qs,as)’ and ‘LargeVLM(ql,al)’, we infer that the Recalibrator successfully aligns the feature representations of large and small VLMs, and that it generalizes well across teacher–student combinations. Furthermore, Fig. 6 visualizes multiple feature spaces to assess whether the large- and small-VLM features after the Recali-

- Table 3: Comparing the final distillation performances from the teacher model: InternVL2.5-78B [19] with and without the regularization term (denoted as “Reg”). Note that, the student models are each InternVL2.5-8B, 4B, 2B, and 1B.

Table 4: Comparing SFT, traditional distillations, and GenRecal where we use equal training dataset (Section 4.1(b)) and choose the teacher (Qwen2-VL72B [125]) that has the same token type of student.

VLMs Reg MMB MathVista MM-Vet MMMU InternVL2.5-8B - 84.6 64.4 62.8 56.0 w. GenRecal ✗ 88.2 69.8 63.5 58.9 w. GenRecal ✓ 89.5 74.9 73.2 68.1 InternVL2.5-4B - 81.1 60.5 60.6 52.3 w. GenRecal ✗ 82.9 61.7 61.1 53.6 w. GenRecal ✓ 88.4 70.4 66.1 58.3 InternVL2.5-2B - 74.7 51.3 60.8 43.6 w. GenRecal ✗ 75.8 53.2 60.9 45.4 w. GenRecal ✓ 84.1 62.5 61.2 59.2 InternVL2.5-1B - 70.7 43.2 48.8 40.9 w. GenRecal ✗ 72.6 45.9 49.8 41.1 w. GenRecal ✓ 80.1 56.5 54.4 45.6

VLMs MMB MathVista MM-Vet MMMU Qwen2-VL-7B 83.0 58.2 62.0 54.1 w. SFT 84.3 60.5 64.2 56.3 w. MiniLLM 84.4 61.3 65.1 57.4 w. DistiLLM 84.8 61.5 65.3 57.9 w. LLaVA-KD 85.0 61.8 65.9 58.2 w. GenRecal 87.8 69.5 67.8 64.2 Qwen2-VL-2B 74.9 43.0 49.5 41.1 w. SFT 76.3 45.8 51.8 44.3 w. MiniLLM 77.5 46.2 52.5 45.3 w. DistiLLM 77.9 46.6 53.0 46.1 w. LLaVA-KD 78.3 46.9 53.7 46.9 w. GenRecal 82.9 58.3 57.1 51.4

[Figure 58]

[Figure 59]

|[Figure 60]<br><br>Recalibrator<br><br>VLM-body<br><br>| | |
|---|---|
|𝑞𝑠(𝑖)| |
<br><br>| | |
|---|---|
|𝑎𝑠(𝑖)| |
<br><br>VLM-body<br><br>|𝑞𝑙(𝑗)|
|---|
<br><br>|𝑎𝑙(𝑗)|
|---|
<br><br>Recalibrator<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Averaging Averaging<br><br>Cos-Sim|
|---|

|()𝑖SampleIndexforSmallVLM|
|---|

()𝑖SampleIndexforSmallVLM

Cos-Sim

|Sample Index ( 𝑗 ) for Large VLM|
|---|

(a) Cosine Similarity (b) Without Regularization (c) With Regularization

- Fig. 7: The process of (a) computing cosine similarity, (b) the result without regularization or (c) with regularization. Because the number of the embedded tokens for small and large VLMs are naturally different due to vocabulary size, token split, and token ordering, therefore we do average the output tokens and compute cosine similarity.

brator are truly matched and can be regarded as a shared representation. The small-VLM features before the Recalibrator are initially close to those after it, but by the end of training they become matched to the large-VLM features after the Recalibrator. This indicates that the Recalibrator is effective at forming a shared feature representation for large and small VLMs.

Importance of Regularization. We conduct an ablation study that investigates how important the regularization term is for the final distillation performances. We first remove the regularization term and analyze the differences before and after its removal. To illustrate this in detail, we simply prepare 10 different question-answer pairs and input them through VLM-body of both small and large VLMs. Afterwards, we propagate their output features into Recalibrator, as described in Fig. 7(a). We then compute a 10 × 10 cosine similarity matrix between Recalibrator’s output features from the small and large VLMs. To ensure reliable cosine similarity values, we use all training dataset samples and compute the average of the cosine similarity values for all the samples.

With the regularization term included, as shown in Fig. 7(c), we observe that the diagonal values in the cosine similarity matrix are significantly higher than the off-diagonal values. This implies that the features of small and large VLMs are explicitly shared through Recalibrator, aligned with the results in

- Table 5: Importance of the autoregressive loss (Lar) for representation alignment. We compare the baseline VLMs, GenRecal trained without Lar (✗, using only Lkl), and GenRecal trained with Lar (✓), using InternVL2.5-78B [19] as the teacher VLM.

VLMs Lar AI2D ChartQA MathVista MMB MMBCN MM-Vet MMMU MMMU-Pro BLINK SEED-2-Plus RWQA Avg Qwen2-VL-7B [125] - 77.5 83.0 58.2 83.0 80.5 62.0 54.1 30.5 53.8 68.6 68.5 65.4 InternVL2.5-8B [19] - 84.8 84.8 64.4 84.6 82.6 62.8 56.0 34.3 54.8 69.7 70.1 68.1 Qwen2-VL-7B-GenRecal (InternVL2.5-78B) ✗ 66.2 67.5 47.8 63.4 55.9 49.6 45.8 30.5 52.1 65.3 60.9 55.0 InternVL2.5-8B-GenRecal (InternVL2.5-78B) ✗ 68.4 70.1 55.2 68.9 62.5 55.4 50.6 35.1 58.7 68.2 64.4 59.8 Qwen2-VL-7B-GenRecal (InternVL2.5-78B) ✓ 93.9 95.3 68.8 88.4 87.4 70.4 65.6 49.6 64.3 70.7 79.2 75.8 InternVL2.5-8B-GenRecal (InternVL2.5-78B) ✓ 93.0 93.6 74.9 89.5 88.2 73.2 68.1 48.8 65.3 72.3 81.4 77.1

Fig. 6 as well. In contrast, when the regularization term is removed, as depicted in Fig. 7(b), the diagonal values are comparable with the off-diagonal values. This indicates that Recalibrator fails to explicitly align large and small VLMs’ features, preventing feature matching and sharing. As a result, this leads to a significant drop in final distillation performances, as reported in Tab. 3. Insufficient alignment of large and student VLMs reduces efficient knowledge transfer during distillation. These findings highlight the crucial role of the regularization term in feature alignment and knowledge transfer for general-purpose distillation.

Autoregressive Loss for Representation Alignment. To align the student representations with the teacher’s latent space, GenRecal uses two objectives: KL divergence loss (Lkl) and autoregressive loss (Lar). The KL divergence loss performs standard distillation by matching the logit-level distribution of teacher and student. However, matching distributions alone does not guarantee precise alignment at the token level. The autoregressive loss provides explicit hardtarget supervision, allowing it to predict the exact ground-truth token sequence. In short, Lkl transfers knowledge at the distribution level, whereas Lar ensures exact alignment at the token level. As shown in Tab. 5, removing Lar consistently degrades performance across benchmarks, confirming that autoregressive supervision is essential for effective representation alignment.

#### 4.3 Ablation Studies: Training Computation and Configurations

In Tab. 6(a), we compare the model sizes and training FLOPs of the small VLM, large VLM, and Recalibrator to assess whether Recalibrator introduces additional computational overhead during training. The results show that Recalibrator’s FLOPs are substantially lower than those of both the large and small VLMs. Note that Recalibrator is removed at inference time, thus incurring no additional computational cost during inference. We further conduct a series of ablation studies to examine various configurations affecting the distillation performance of GenRecal. As shown in Tab. 6(b), we vary the decoder depth of Recalibrator from 1 to 20 and select two depths for layer number of Rec-body based on a trade-off between computational efficiency and performance. Additionally, Tab. 6(c) demonstrates the importance of employing a new positional embedding (NPE) to handle newly introduced position IDs and to integrate RoPE [110] from both large and small VLMs, which have different token types.

To analyze the effect of dataset scale, we control the dataset size from 1M to 9M samples, as shown in Tab. 6(d). The performance improvements between

- 5M and 9M samples are marginal, suggesting that a 5M dataset is a practical

- Table 6: FLOPs comparison and several ablation studies on various configurations influencing the performance of GenRecal. Note that the teacher VLM in (b), (c), (d),

- (e), and (f) is InternVL2.5-78B [19], and the student VLM in (b) is InternVL2.5-8B [19].

(a) Model Size and FLOPs

(b) Recalibrator Depth

(c) Effect of NPE

Depths MMB MathVista MM-Vet MMMU Avg 20 89.9 75.1 73.6 68.3 76.7 15 89.8 75.2 73.5 68.4 76.7 10 89.6 75.0 73.4 68.3 76.6

VLM NPE MMB MathVista MM-Vet MMMU Avg Qwen2-VL-7B-GenRecal ✗ 79.2 60.0 64.0 63.9 66.8 Qwen2-VL-7B-GenRecal ✓ 88.4 68.8 70.4 65.6 73.3 InternVL2.5-8B-GenRecal ✗ 85.2 65.1 67.3 64.4 70.5 InternVL2.5-8B-GenRecal ✓ 89.5 74.9 73.2 68.1 76.4

Small VLM Large VLM Recalibrator Size Qwen2-VL-7B (25.7 × 1012) Qwen2-VL-72B (260 × 1012) 524.8M (2.27 × 1012) InternVL2.5-8B (27.1 × 1012) InternVL2.5-78B (269 × 1012) 503.3M (2.2 × 1012) InternVL2.5-8B (27.1 × 1012) Qwen2-VL-72B (260 × 1012) 503.3M (2.2 × 1012) Qwen2-VL-7B (25.7 × 1012) InternVL2.5-78B (269 × 1012) 524.8M (2.27 × 1012)

5 89.4 74.8 73.3 68.2 76.4 2 89.5 74.9 73.2 68.1 76.4 1 88.0 73.0 72.0 66.5 74.9

(e) Fine-Tuning

(f) Cross-Tokenizer Methods

(d) Training Dataset Size

VLM Finetuned MMB MathVista MM-Vet MMMU Avg InternVL2.5-78B [19] ✗ 88.3 72.3 72.3 70.1 75.8 InternVL2.5-78B-SFT ✓ 89.4 75.2 74.2 71.7 77.6 InternVL2.5-8B [19] ✗ 84.6 64.4 62.8 56.0 67.0 InternVL2.5-8B-SFT ✓ 86.5 65.2 65.0 59.9 69.2 InternVL2.5-8B-GenRecal ✓ 89.5 74.9 73.2 68.1 76.4

VLM MathVista MMB MM-Vet MMMU Avg InternVL2.5-8B 64.4 84.6 62.8 56.0 67.0 InternVL2.5-8B-ULD [9] 66.5 86.2 64.1 60.8 69.4 InternVL2.5-8B-MOT [25] 68.4 87.5 65.0 62.3 70.8 InternVL2.5-8B-GenRecal 74.9 89.5 73.2 68.1 76.4

Dataset Size MMB MathVista MM-Vet MMMU Avg 9M 89.5 74.9 73.2 68.1 76.4 7M 89.4 74.7 73.0 67.8 76.2 5M 89.3 74.5 72.8 67.6 76.0 3M 88.9 73.6 71.9 66.6 75.3 1M 86.6 68.6 67.0 60.9 70.8 0M 84.6 64.4 62.8 56.0 67.0

(h) Different VLM Family

(g) Different LLMs

(i) Recent VLMs

VLM MM-Vet MathVista MMMU Avg Ovis1.6-Gemma-9B [86] 65.0 67.3 55.0 62.4 Ovis1.6-Gemma-9B-GenRecal (InternVL2.5-78B) 74.0 75.3 67.7 72.3 InternVL2.5-78B 72.3 72.3 70.1 71.6 InternVL2.5-8B [19] 62.8 64.4 56.0 61.1 InternVL2.5-8B-GenRecal (Ovis1.6-Gemma-27B) 68.5 69.9 61.2 66.5 Ovis1.6-Gemma-27B [86] 67.2 69.7 61.7 66.2

VLM MMB MathVista MM-Vet MMMU Avg Qwen3-VL-8B [7] 86.8 77.2 74.5 69.6 77.0 Qwen3-VL-8B (Qwen3-VL-32B) 89.9 80.1 77.6 72.7 80.1 Qwen3-VL-8B (InternVL3.5-38B) 89.7 80.3 77.4 72.5 80.0 InternVL3.5-8B [127] 86.5 78.4 83.1 73.4 80.4 InternVL3.5-8B (Qwen3-VL-32B) 89.6 81.5 86.0 76.3 83.4 InternVL3.5-8B (InternVL3.5-38B) 89.4 81.3 86.2 76.5 83.4 Qwen3-VL-32B [7] 90.6 83.8 79.4 76.0 82.5 InternVL3.5-38B [127] 90.3 81.9 82.2 76.9 82.8

LLMs MMLU GPQA MATH Avg Qwen2-7B [131] 70.8 38.4 48.6 52.6 Qwen2-7B-GenRecal (Llama-3-70B) 78.4 38.9 49.1 55.5 Llama-3-70B [30] 82.5 39.5 50.4 57.5 InternLM2.5-7B [11] 72.8 38.4 60.1 57.1 InternLM2.5-7B-GenRecal (Qwen2.5-72B) 83.7 43.6 61.8 63.0 Qwen2.5-72B [131] 86.1 45.9 62.1 64.7

choice for users with limited training resources. Furthermore, Tab. 6(e) shows that fine-tuning the large VLM on our dataset yields further gains, indicating that the small VLM trained under the same setting has the potential to match or even surpass the large VLM’s performance.

We also compare GenRecal with recent cross-tokenizer distillation methods in Tab. 6(f). These baselines perform distillation via cross-token matching in the logit space using the classical Wasserstein distance [9] or optimal transport [25], whereas GenRecal introduces the Recalibrator to align hidden features for semantic correspondence. GenRecal achieves a substantial improvement over these approaches, as discussed in the next section. Lastly, Tab. 6(g)–(i) demonstrate GenRecal’s strong compatibility across different LLMs, VLM families, and recently released VLMs. In all cases, GenRecal consistently delivers superior performance, validating its general applicability to diverse model architectures.

#### 4.4 Discussion

Representation Mismatch Beyond Tokenizer Mismatch. Even within a family the tokenizer can differ—InternVL2.5-8B uses InternLM2.5 [11] while InternVL2.5-78B uses Qwen2.5 [131], whereas Qwen2-VL-7B/72B [125] share the Qwen2 tokenizer (Fig. 1). More importantly, even with an identical tokenizer, a large teacher–student gap causes a representation mismatch: conventional distillation routes knowledge through the low-dimensional student head, creating a bottleneck. GenRecal instead distills through the high-capacity teacher head, with the Recalibrator warm-starting student features into the teacher space (Fig. 3). This is why it helps even under the same tokenizer: in Tab. 4 (Qwen2VL-72B teacher, 7B/2B students), GenRecal beats both SFT and traditional KD—MiniLLM [34] (reverse KL), DistiLLM [52] (skewed KL), and LLaVAKD [10] (3-stage KL)—by a clear margin, showing explicit feature alignment matters even with shared token types.

Cross-Token Matching. Cross-token matching takes two forms: (a) wordembedding matching, infeasible without retraining the decoder, and (b) semantic feature matching. For (b), ULD [9] and MOT [25] align logit-space probability mass via Wasserstein distance or optimal transport (Tab. 6(f)), but can-

###### Table 7: Comparison of computational cost (FLOPs and Training Time) and performance across cross-tokenizer distillation methods. We average performance across four benchmarks: MMB [83], MathVista [85], MM-Vet [136], and MMMU [138].

Method Extra FLOPs Training Stages Training Time Avg. Perform.

GenRecal Recalibrator 298.3×1012 3 3.1+10.4+5.3=18.8 hours 76.4 ULD [9] Vocab Matching 296.1×1012 1 11.1×2(Epochs)=22.2 hours 69.4 MOT [25] Optimal Transport 296.1×1012 1 12.5×2(Epochs)=25.0 hours 70.8

###### Table 8: Decomposing the source of GenRecal’s improvement on Qwen2-VL-7B with the same-token-type teacher Qwen2-VL-72B, so that traditional logit-KD baselines remain valid. All rows use the same 9M data. Avg. is over MMB [83], MathVista [85], MM-Vet [136], and MMMU [138].

Setup (all use the same 9M data) Avg. ∆ Baseline Qwen2-VL-7B (no training) 64.3 –

+ 9M SFT (Stage-3 recipe only, no distill, no Recalib.) 66.3 +2.0 + multi-stage SFT+logit KD (LLaVA-KD recipe, no Recalib.) 67.7 +3.4 + GenRecal (Recalibrator) 72.3 +8.0

###### Table 9: Stage-wise checkpoints of GenRecal with InternVL2.5-8B (student) and InternVL2.5-78B (teacher), compared against an SFT-only baseline on the same data. Stage 1 trains only the Recalibrator and leaves the student unchanged.

Checkpoint Trainable parameters Data MMB MathVista MM-Vet MMMU Avg Baseline — — 84.6 64.4 62.8 56.0 67.0

- After Stage 1 Recalibrator only (align) 9M 84.6 64.4 62.8 56.0 67.0
- After Stage 2 Recalibrator + Student 9M 88.2 72.5 70.9 65.7 74.3

- After Stage 3 (GenRecal full) Student only (SFT polish) 6M 89.5 74.9 73.2 68.1 76.4 SFT only Student only (SFT polish) 9M 86.5 65.2 65.0 59.9 69.2

not handle token-split indices and resort to zero-padding or truncation, losing information. GenRecal avoids logit-level matching altogether: the Recalibrator aligns features before the head so the teacher VLM-head reads student features directly, needing no token correspondence, probability sorting, padding, or transport cost. It adds only a small FLOPs fraction (Tab. 6(a)), so GenRecal’s total cost is comparable to ULD/MOT yet reaches higher accuracy (Tab. 7; same data, InternVL2.5-8B student and -78B teacher).

Why a Non-Linear Recalibrator, and Where the Novelty Lies. Prior shared-space results rely on linear word-embedding maps [24, 91, 106], which act on input embeddings, not on the answer-conditioned hidden states after the VLM-body. Aligning these post-body features while preserving token order exceeds a linear map, so we use shallow transformer decoder blocks. The novelty is thus not the module itself—depth beyond two layers saturates (Tab. 6(b))but the stable recipe that makes cross-tokenizer teacher-head distillation work.

Isolating the Recalibrator from the Recipe. To check whether the gains come from the Recalibrator or merely from the multi-stage recipe and final SFT, we disentangle them on Qwen2-VL-7B with the same-token-type teacher Qwen2-VL-72B (so logit-KD baselines stay valid), all on the same 9M data. The recipe alone (SFT+logit-KD, no Recalibrator) gives +3.4 Avg, while adding the Recalibrator adds a further +4.6 (Tab. 8). Nor is the gain from Stage-3

- Table 10: Effect of training-data scale on GenRecal with InternVL2.5-8B (student) and InternVL2.5-78B (teacher), compared against SFT-only at full scale. Stage 3 uses a random ∼70% subsample of the Stage 2 data.

Setting Align data Distill data Final SFT data MMB MathVista MM-Vet MMMU Avg Baseline (no train) — — — 84.6 64.4 62.8 56.0 67.0 SFT only — — 9M 86.5 65.2 65.0 59.9 69.2 GenRecal @ 1M 1M 1M 0.7M 86.6 68.6 67.0 60.9 70.8 GenRecal @ 3M 3M 3M 2M 88.9 73.6 71.9 66.6 75.3 GenRecal @ 5M 5M 5M 3M 89.3 74.5 72.8 67.6 76.0 GenRecal @ 9M (full) 9M 9M 6M 89.5 74.9 73.2 68.1 76.4

SFT: after Stage 2, before any final SFT, GenRecal already reaches 74.3 Avg vs. 69.2 for SFT-only on the same 9M (Tab. 9). Nor is it mere data scale: GenRecal@1M (∼11% data) already beats SFT@9M, and @3M nears the full result (Tab. 10). Further analyses are provided in the appendix: open-weight teacher dependence and the SFT fallback (Appendix F), component-level ablations (Appendix G), the regularization formulation (Appendix H), error-profile and calibration (Appendix I), and a compute/fairness comparison against baselines (Appendix J). Note that GenRecal operates under open-weight teacher access, where the teacher’s parameters and hidden features are available for training.

### 5 Conclusion

We present Generation after Recalibration (GenRecal), a general-purpose distillation framework that transfers knowledge from large to small vision-language models (VLMs) even when they adopt different token types. Existing distillation methods assume that the teacher and student share the same token types, while GenRecal removes this constraint through a Recalibrator that aligns the student’s hidden features into the teacher’s representation space. Since the Recalibrator is discarded after training, inference preserves the student’s original architecture and cost. Our analyses further indicate that these gains stem primarily from the recalibration mechanism rather than from the multi-stage recipe, and that feature alignment remains beneficial even when the teacher and student already share a tokenizer. For future work, we plan to extend the Recalibrator to intermediate layers to capture fine-grained, sequential knowledge, and to explore distillation from multiple VLM sources. We envision GenRecal as a pivotal and versatile framework for knowledge distillation.

### Acknowledgements

We thank our colleagues at NVIDIA, in particular Sharath Turuvekere Sreenivas, for valuable discussions on cross-tokenizer and cross-model knowledge distillation. We further thank Sepehr Sameni and Daniel Korzekwa for experimenting with this approach on Minitron and Cosmos families; subsequently, Sharath Turuvekere Sreenivas and Pavlo Molchanov developed the follow-up X-Token [109], extending it to multi-teacher cross-tokenizer distillation on the logit space.

### References

- 1. Abdin, M., Jacobs, S.A., Awan, A.A., Aneja, J., Awadallah, A., Awadalla, H., Bach, N., Bahree, A., Bakhtiari, A., Behl, H., et al.: Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219 (2024)
- 2. Agarwal, R., Vieillard, N., Zhou, Y., Stanczyk, P., Garea, S.R., Geist, M., Bachem, O.: On-policy distillation of language models: Learning from self-generated mistakes. In: The Twelfth International Conference on Learning Representations

(2024)

- 3. Alayrac, J.B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al.: Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems 35, 23716–23736 (2022)
- 4. Anthropic: The claude 3 model family: Opus, sonnet, haiku. https : / / www . anthropic . com (2024), https : / / www - cdn . anthropic . com / de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf
- 5. Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., et al.: Program synthesis with large language models. arXiv preprint arXiv:2108.07732 (2021)
- 6. Ba, J.L., Kiros, J.R., Hinton, G.E.: Layer normalization (2016), https://arxiv. org/abs/1607.06450
- 7. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631

(2025)

- 8. Ben-Baruch, E., Karklinsky, M., Biton, Y., Ben-Cohen, A., Lawen, H., Zamir, N.: It’s all in the head: Representation knowledge distillation through classifier sharing. arXiv preprint arXiv:2201.06945 (2022)
- 9. Boizard, N., Haddad, K.E., HUDELOT, C., Colombo, P.: Towards cross-tokenizer distillation: the universal logit distillation loss for LLMs. Transactions on Machine Learning Research (2025), https://openreview.net/forum?id=bwRxXiGO9A
- 10. Cai, Y., Zhang, J., He, H., He, X., Tong, A., Gan, Z., Wang, C., Bai, X.: Llavakd: A framework of distilling multimodal large language models. arXiv preprint arXiv:2410.16236 (2024)
- 11. Cai, Z., Cao, M., Chen, H., Chen, K., Chen, K., Chen, X., Chen, X., Chen, Z., Chen, Z., Chu, P., et al.: Internlm2 technical report. arXiv preprint arXiv:2403.17297 (2024)
- 12. Cao, J., Zhang, Y., Huang, T., Lu, M., Zhang, Q., An, R., Ma, N., Zhang, S.: Move-kd: Knowledge distillation for vlms with mixture of visual encoders. arXiv preprint arXiv:2501.01709 (2025)
- 13. Chen, J., Zhu, D., Shen, X., Li, X., Liu, Z., Zhang, P., Krishnamoorthi, R., Chandra, V., Xiong, Y., Elhoseiny, M.: Minigpt-v2: large language model as a unified interface for vision-language multi-task learning. arXiv preprint arXiv:2310.09478

(2023)

- 14. Chen, L., Li, J., Dong, X., Zhang, P., Zang, Y., Chen, Z., Duan, H., Wang, J., Qiao, Y., Lin, D., et al.: Are we on the right way for evaluating large visionlanguage models? arXiv preprint arXiv:2403.20330 (2024)
- 15. Chen, L., Li, J., Dong, X., Zhang, P., He, C., Wang, J., Zhao, F., Lin, D.: Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793 (2023)

- 16. Chen, P., Liu, S., Zhao, H., Jia, J.: Distilling knowledge via knowledge review. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5008–5017 (2021)
- 17. Chen, Q., Qin, L., Zhang, J., Chen, Z., Xu, X., Che, W.: M3cot: A novel benchmark for multi-domain multi-step multi-modal chain-of-thought. In: Proc. of ACL

(2024)

- 18. Chen, Y., Hu, H., Luan, Y., Sun, H., Changpinyo, S., Ritter, A., Chang, M.W.: Can pre-trained vision and language models answer visual information-seeking questions? arXiv preprint arXiv:2302.11713 (2023)
- 19. Chen, Z., Wang, W., Cao, Y., Liu, Y., Gao, Z., Cui, E., Zhu, J., Ye, S., Tian, H., Liu, Z., et al.: Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271

(2024)

- 20. Chen, Z., Wang, W., Tian, H., Ye, S., Gao, Z., Cui, E., Tong, W., Hu, K., Luo, J., Ma, Z., et al.: How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821 (2024)
- 21. Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Muyan, Z., Zhang, Q., Zhu, X., Lu, L., et al.: Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238 (2023)
- 22. Cho, S., Hachiuma, R., Badki, A., Su, H., Lee, B.K., Song, C.H., Liu, S., Radhakrishnan, S., Kim, S., Wang, Y.C.F., Chen, M.H.: Spatialclaw: Rethinking action interface for agentic spatial reasoning (2026), https://arxiv.org/abs/2606.13673
- 23. Chung, H.W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., Dehghani, M., Brahma, S., et al.: Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416 (2022)
- 24. Conneau, A., Lample, G., Ranzato, M., Denoyer, L., Jégou, H.: Word translation without parallel data. arXiv preprint arXiv:1710.04087 (2017)
- 25. Cui, X., Zhu, M., Qin, Y., Xie, L., Zhou, W., Li, H.: Multi-level optimal transport for universal cross-tokenizer knowledge distillation on language models. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 39, pp. 23724– 23732 (2025)
- 26. Dai, W., Lee, N., Wang, B., Yang, Z., Liu, Z., Barker, J., Rintamaki, T., Shoeybi, M., Catanzaro, B., Ping, W.: Nvlm: Open frontier-class multimodal llms. arXiv preprint (2024)
- 27. Dai, W., Li, J., Li, D., Tiong, A., Zhao, J., Wang, W., Li, B., Fung, P., Hoi, S.: InstructBLIP: Towards general-purpose vision-language models with instruction tuning. In: Thirty-seventh Conference on Neural Information Processing Systems

- (2023)

28. Deitke, M., Clark, C., Lee, S., Tripathi, R., Yang, Y., Park, J.S., Salehi, M., Muennighoff, N., Lo, K., Soldaini, L., et al.: Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models. arXiv preprint arXiv:2409.17146

- (2024)

- 29. Dong, X., Zhang, P., Zang, Y., Cao, Y., Wang, B., Ouyang, L., Zhang, S., Duan, H., Zhang, W., Li, Y., et al.: Internlm-xcomposer2-4khd: A pioneering large visionlanguage model handling resolutions from 336 pixels to 4k hd. arXiv preprint arXiv:2404.06512 (2024)
- 30. Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al.: The llama 3 herd of models. arXiv preprint arXiv:2407.21783 (2024)

- 31. Feng, Q., Li, W., Lin, T., Chen, X.: Align-kd: Distilling cross-modal alignment knowledge for mobile vision-language model. arXiv preprint arXiv:2412.01282

(2024)

- 32. Fu, X., Hu, Y., Li, B., Feng, Y., Wang, H., Lin, X., Roth, D., Smith, N.A., Ma, W.C., Krishna, R.: Blink: Multimodal large language models can see but not perceive. arXiv preprint arXiv:2404.12390 (2024)
- 33. Gu, S., Zhang, J., Zhou, S., Yu, K., Xing, Z., Wang, L., Cao, Z., Jia, J., Zhang, Z., Wang, Y., et al.: Infinity-mm: Scaling multimodal performance with large-scale and high-quality instruction data. arXiv preprint arXiv:2410.18558 (2024)
- 34. Gu, Y., Dong, L., Wei, F., Huang, M.: Minillm: Knowledge distillation of large language models. In: The Twelfth International Conference on Learning Representations (2024)
- 35. Heo, B., Kim, J., Yun, S., Park, H., Kwak, N., Choi, J.Y.: A comprehensive overhaul of feature distillation. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 1921–1930 (2019)
- 36. Hinton, G., Vinyals, O., Dean, J.: Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531 (2015)
- 37. Hong, W., Wang, W., Ding, M., Yu, W., Lv, Q., Wang, Y., Cheng, Y., Huang, S., Ji, J., Xue, Z., et al.: Cogvlm2: Visual language models for image and video understanding. arXiv preprint arXiv:2408.16500 (2024)
- 38. Hosu, V., Lin, H., Sziranyi, T., Saupe, D.: Koniq-10k: An ecologically valid database for deep learning of blind image quality assessment. IEEE Transactions on Image Processing 29, 4041–4056 (2020)
- 39. Hsieh, C.Y., Li, C.L., Yeh, C.K., Nakhost, H., Fujii, Y., Ratner, A., Krishna, R., Lee, C.Y., Pfister, T.: Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes. arXiv preprint arXiv:2305.02301 (2023)
- 40. Hu, A., Xu, H., Ye, J., Yan, M., Zhang, L., Zhang, B., Li, C., Zhang, J., Jin, Q., Huang, F., et al.: mplug-docowl 1.5: Unified structure learning for ocr-free document understanding. arXiv preprint arXiv:2403.12895 (2024)
- 41. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685 (2021)
- 42. Huang, Z., Chen, K., He, J., Bai, X., Karatzas, D., Lu, S., Jawahar, C.: Icdar2019 competition on scanned receipt ocr and information extraction. In: 2019 International Conference on Document Analysis and Recognition (ICDAR). pp. 1516–

1520. IEEE (2019)

- 43. Kang, M., Diao, S., Hachiuma, R., Hwang, S.J., Molchanov, P., Wang, Y.C.F., Lee, B.K.: Agent explorative policy optimization for multimodal agentic reasoning. arXiv preprint arXiv:2605.28774 (2026)
- 44. Kaplan, J., McCandlish, S., Henighan, T., Brown, T.B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., Amodei, D.: Scaling laws for neural language models. arXiv preprint arXiv:2001.08361 (2020)
- 45. Kembhavi, A., Salvato, M., Kolve, E., Seo, M., Hajishirzi, H., Farhadi, A.: A diagram is worth a dozen images. In: Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14. pp. 235–251. Springer (2016)
- 46. Kim, J., Kim, K., Kim, W., Lee, B.K., Park, C.: Why and when visual token pruning fails? a study on relevant visual information shift in mllms decoding. arXiv preprint arXiv:2604.12358 (2026)

- 47. Kim, J., Lee, B.K., Ro, Y.M.: Distilling robust and non-robust features in adversarial examples by information bottleneck. In: Ranzato, M., Beygelzimer, A., Dauphin, Y., Liang, P., Vaughan, J.W. (eds.) Advances in Neural Information Processing Systems. vol. 34, pp. 17148–17159. Curran Associates, Inc.

(2021), https://proceedings.neurips.cc/paper_files/paper/2021/file/ 8e5e15c4e6d09c8333a17843461041a9-Paper.pdf

- 48. Kim, J., Lee, B.K., Ro, Y.M.: Demystifying causal features on adversarial examples and causal inoculation for robust network by adversarial instrumental variable regression. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 12302–12312 (June 2023)
- 49. Kim, J., Lee, B.K., Ro, Y.M.: Causal unsupervised semantic segmentation. Pattern Recognition 171, 112173 (2026). https://doi.org/https://doi.org/ 10.1016/j.patcog.2025.112173, https://www.sciencedirect.com/science/ article/pii/S0031320325008349
- 50. Kim, Y., Kim, J., Lee, B.K., Shin, S., Ro, Y.M.: Mitigating dataset bias in image captioning through clip confounder-free captioning network. In: 2023 IEEE International Conference on Image Processing (ICIP). pp. 1720–1724 (2023). https://doi.org/10.1109/ICIP49359.2023.10222502
- 51. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment anything. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 4015–4026

(2023)

- 52. Ko, J., Kim, S., Chen, T., Yun, S.Y.: Distillm: Towards streamlined distillation for large language models. arXiv preprint arXiv:2402.03898 (2024)
- 53. Lee, B.K.: Training encoder-attention through fully-connected crfs for efficient end-to-end lane detection model (2020)
- 54. Lee, B.K.: Building high-performing, efficient-size vision language models: merge, modify, and distill (2025)
- 55. Lee, B.K., Chee, Y., Ro, Y.M.: Recursive think-answer process for llms and vlms. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings. pp. 9608–9621 (June 2026)
- 56. Lee, B.K., Chung, S., Kim, C.W., Park, B., Ro, Y.M.: Phantom of latent for large language and vision models. arXiv preprint arXiv:2409.14713 (2024)
- 57. Lee, B.K., Chung, S., Kim, C.W., Park, B., Ro, Y.M.: TroL: Traversal of layers for large language and vision models. In: Al-Onaizan, Y., Bansal, M., Chen, Y.N. (eds.) Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. pp. 11314–11342. Association for Computational Linguistics, Miami, Florida, USA (Nov 2024). https://doi.org/10.18653/v1/2024.emnlpmain.633, https://aclanthology.org/2024.emnlp-main.633/
- 58. Lee, B.K., Hachiuma, R., Ro, Y.M., Wang, F., Wu, Y.H.: Unified reinforcement and imitation learning for vision-language models. In: Belgrave, D., Zhang, C., Lin, H., Pascanu, R., Koniusz, P., Ghassemi, M., Chen, N. (eds.) Advances in Neural Information Processing Systems. vol. 38, pp. 156508–156534. Curran Associates, Inc. (2025), https://proceedings.neurips.cc/paper_files/paper/ 2025/file/e58497367bc8730f61a87d37800c0a06-Paper-Conference.pdf
- 59. Lee, B.K., Hachiuma, R., Wang, Y.C.F., Ro, Y.M., Wu, Y.H.: Vlsi: Verbalized layers-to-interactions from large to small vision language models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 29545–29557 (June 2025)

- 60. Lee, B.K., Kim, C.W., Park, B., Ro, Y.M.: Meteor: Mamba-based traversal of rationale for large language and vision models. In: Globerson, A., Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J., Zhang, C. (eds.) Advances in Neural Information Processing Systems. vol. 37, pp. 40278–

40315. Curran Associates, Inc. (2024). https://doi.org/10.52202/0790171274, https://proceedings.neurips.cc/paper_files/paper/2024/file/ 473a9a75edc46eff5ff224d53d5f7294-Paper-Conference.pdf

- 61. Lee, B.K., Kim, J., Ro, Y.M.: Masking adversarial damage: Finding adversarial saliency for robust and sparse network. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 15126–15136 (June 2022)
- 62. Lee, B.K., Kim, J., Ro, Y.M.: Mitigating adversarial vulnerability through causal parameter estimation by adversarial double machine learning. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 4499– 4509 (October 2023)
- 63. Lee, B.K., Lu, X., Diao, S., Kang, M., Muralidharan, S., Sapra, K., Tao, A., Molchanov, P., Choi, Y., Wang, Y.C.F., et al.: Zone of proximal policy optimization: Teacher in prompts, not gradients. arXiv preprint arXiv:2606.18216 (2026)
- 64. Lee, B.K., Park, B., Kim, C.W., Ro, Y.M.: CoLLaVO: Crayon large language and vision mOdel. In: Ku, L.W., Martins, A., Srikumar, V. (eds.) Findings of the Association for Computational Linguistics: ACL 2024. pp. 1121–1138. Association for Computational Linguistics, Bangkok, Thailand (Aug 2024). https://doi. org/10.18653/v1/2024.findings-acl.66, https://aclanthology.org/2024. findings-acl.66/
- 65. Lee, B.K., Park, B., Won Kim, C., Man Ro, Y.: Moai: Mixture of all intelligence for large language and vision models. In: Leonardis, A., Ricci, E., Roth, S., Russakovsky, O., Sattler, T., Varol, G. (eds.) Computer Vision – ECCV 2024. pp. 273–302. Springer Nature Switzerland, Cham (2025)
- 66. Lee, B.K., Wang, Y.C.F., Hachiuma, R.: Masking teacher and reinforcing student for distilling vision-language models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 10126–10141 (June 2026)
- 67. Lee, B.K., Yu, Y., Ro, Y.M.: Towards adversarial robustness of bayesian neural network through hierarchical variational inference (2021), https://openreview. net/forum?id=Cue2ZEBf12
- 68. Lee, Y.J., Kim, S., Lee, B.K., Moon, M., Hwang, Y., Kim, J.M., Neubig, G., Welleck, S., Choi, H.J.: Refinebench: Evaluating refinement capability of language models via checklists. In: The Fourteenth International Conference on Learning Representations (2026), https://openreview.net/forum?id=GYJFJz9Dy5
- 69. Lee, Y.J., Lee, B.K., Lee, D., Oh, K.J., Hwang, Y., Choi, H.J., et al.: Enhancing conversational agents with skill-of-mind-infused large language model
- 70. Lee, Y.J., Lee, B.K., Zhang, J., Hwang, Y., Ko, B., Kim, H.G., Yao, D., Rong, X., Joo, E., Han, S.H., Ko, B., Choi, H.J.: Multiverse: A multi-turn conversation benchmark for evaluating large vision and language models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 708–719 (October 2025)
- 71. Lee, Y.S., Sultan, M., El-Kurdi, Y., Naseem, T., Munawar, A., Florian, R., Roukos, S., Astudillo, R.F.: Ensemble-instruct: Instruction tuning data generation with a heterogeneous mixture of lms. In: Findings of the Association for Computational Linguistics: EMNLP 2023. pp. 12561–12571 (2023)

- 72. Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Li, Y., Liu, Z., Li, C.: Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326

(2024)

- 73. Li, B., Ge, Y., Chen, Y., Ge, Y., Zhang, R., Shan, Y.: Seed-bench-2-plus: Benchmarking multimodal large language models with text-rich visual comprehension. arXiv preprint arXiv:2404.16790 (2024)
- 74. Li, J., Li, D., Savarese, S., Hoi, S.: Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597 (2023)
- 75. Li, J., Li, D., Xiong, C., Hoi, S.: BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In: Chaudhuri, K., Jegelka, S., Song, L., Szepesvari, C., Niu, G., Sabato, S. (eds.) Proceedings of the 39th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 162, pp. 12888–12900. PMLR (17–23 Jul 2022)
- 76. Li, X., Zhang, F., Diao, H., Wang, Y., Wang, X., Duan, L.Y.: Densefusion-1m: Merging vision experts for comprehensive multimodal perception. arXiv preprint arXiv:2407.08303 (2024)
- 77. Li, Y., Zhang, Y., Wang, C., Zhong, Z., Chen, Y., Chu, R., Liu, S., Jia, J.: Minigemini: Mining the potential of multi-modality vision language models. arXiv preprint arXiv:2403.18814 (2024)
- 78. Lin, J., Yin, H., Ping, W., Lu, Y., Molchanov, P., Tao, A., Mao, H., Kautz, J., Shoeybi, M., Han, S.: Vila: On pre-training for visual language models. arXiv preprint arXiv:2312.07533 (2023)
- 79. Liu, F., Wang, X., Yao, W., Chen, J., Song, K., Cho, S., Yacoob, Y., Yu, D.: Mmc: Advancing multimodal chart understanding with large-scale instruction tuning. arXiv preprint arXiv:2311.10774 (2023)
- 80. Liu, H., Li, C., Li, Y., Lee, Y.J.: Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744 (2023)
- 81. Liu, H., Li, C., Li, Y., Li, B., Zhang, Y., Shen, S., Lee, Y.J.: Llava-next: Improved reasoning, ocr, and world knowledge (January 2024), https://llava-vl.github. io/blog/2024-01-30-llava-next/
- 82. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. In: Thirty-seventh Conference on Neural Information Processing Systems (2023)
- 83. Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al.: Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281 (2023)
- 84. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: International Conference on Learning Representations (2019), https://openreview. net/forum?id=Bkg6RiCqY7
- 85. Lu, P., Bansal, H., Xia, T., Liu, J., Li, C., Hajishirzi, H., Cheng, H., Chang, K.W., Galley, M., Gao, J.: Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255 (2023)
- 86. Lu, S., Li, Y., Chen, Q.G., Xu, Z., Luo, W., Zhang, K., Ye, H.J.: Ovis: Structural embedding alignment for multimodal large language model. arXiv:2405.20797

(2024)

- 87. Lu, Y., Jiang, D., Chen, W., Wang, W.Y., Choi, Y., Lin, B.Y.: Wildvision: Evaluating vision-language models in the wild with human preferences. arXiv preprint arXiv:2406.11069 (2024)
- 88. Masry, A., Long, D.X., Tan, J.Q., Joty, S., Hoque, E.: Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244 (2022)

- 89. McKinzie, B., Gan, Z., Fauconnier, J.P., Dodge, S., Zhang, B., Dufter, P., Shah, D., Du, X., Peng, F., Weers, F., et al.: Mm1: Methods, analysis & insights from multimodal llm pre-training. arXiv preprint arXiv:2403.09611 (2024)
- 90. Mikolov, T., Chen, K., Corrado, G., Dean, J.: Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781 (2013)
- 91. Mikolov, T., Le, Q.V., Sutskever, I.: Exploiting similarities among languages for machine translation. arXiv preprint arXiv:1309.4168 (2013)
- 92. Muralidharan, S., Sreenivas, S.T., Joshi, R., Chochowski, M., Patwary, M., Shoeybi, M., Catanzaro, B., Kautz, J., Molchanov, P.: Compact language models via pruning and knowledge distillation. arXiv preprint arXiv:2407.14679 (2024)
- 93. OpenAI: Gpt-4v(ision) system card (2023), https://openai.com/research/gpt4v-system-card, Last accessed on 2024-02-13
- 94. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)
- 95. Park, W., Kim, D., Lu, Y., Cho, M.: Relational knowledge distillation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 3967–3976 (2019)
- 96. Passalis, N., Tzelepi, M., Tefas, A.: Probabilistic knowledge transfer for lightweight deep representation learning. IEEE Transactions on Neural Networks and learning systems 32(5), 2030–2039 (2020)
- 97. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: Meila, M., Zhang, T. (eds.) Proceedings of the 38th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 139, pp. 8748–8763. PMLR (18–24 Jul 2021)
- 98. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., Liu, P.J.: Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv e-prints (2019)
- 99. Rajbhandari, S., Rasley, J., Ruwase, O., He, Y.: Zero: Memory optimizations toward training trillion parameter models. In: SC20: International Conference for High Performance Computing, Networking, Storage and Analysis. pp. 1–16. IEEE

- (2020)

100. Riquelme, C., Puigcerver, J., Mustafa, B., Neumann, M., Jenatton, R., Susano Pinto, A., Keysers, D., Houlsby, N.: Scaling vision with sparse mixture of experts. Advances in Neural Information Processing Systems 34, 8583–8595

- (2021)

- 101. Romero, A., Ballas, N., Kahou, S.E., Chassang, A., Gatta, C., Bengio, Y.: Fitnets: Hints for thin deep nets. arXiv preprint arXiv:1412.6550 (2014)
- 102. Sanh, V., Debut, L., Chaumond, J., Wolf, T.: Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108 (2019)
- 103. Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., Dean, J.: Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In: International Conference on Learning Representations (2017), https: //openreview.net/forum?id=B1ckMDqlg
- 104. Shi, M., Liu, F., Wang, S., Liao, S., Radhakrishnan, S., Huang, D.A., Yin, H., Sapra, K., Yacoob, Y., Shi, H., et al.: Eagle: Exploring the design space for multimodal llms with mixture of encoders. arXiv preprint arXiv:2408.15998 (2024)

- 105. Shu, F., Liao, Y., Zhuo, L., Xu, C., Zhang, G., Shi, H., Chen, L., Zhong, T., He, W., Fu, S., et al.: Llava-mod: Making llava tiny via moe knowledge distillation. arXiv preprint arXiv:2408.15881 (2024)
- 106. Smith, S.L., Turban, D.H., Hamblin, S., Hammerla, N.Y.: Offline bilingual word vectors, orthogonal transformations and the inverted softmax. arXiv preprint arXiv:1702.03859 (2017)
- 107. Soldaini, L., Kinney, R., Bhagia, A., Schwenk, D., Atkinson, D., Authur, R., Bogin, B., Chandu, K., Dumas, J., Elazar, Y., Hofmann, V., Jha, A., Kumar, S., Lucy, L., Lyu, X., Lambert, N., Magnusson, I., Morrison, J., Muennighoff, N., Naik, A., Nam, C., Peters, M., Ravichander, A., Richardson, K., Shen, Z., Strubell, E., Subramani, N., Tafjord, O., Walsh, E., Zettlemoyer, L., Smith, N., Hajishirzi, H., Beltagy, I., Groeneveld, D., Dodge, J., Lo, K.: Dolma: an open corpus of three trillion tokens for language model pretraining research. In: Ku, L.W., Martins, A., Srikumar, V. (eds.) Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 15725–15788. Association for Computational Linguistics, Bangkok, Thailand (Aug 2024). https://doi.org/10.18653/v1/2024.acl-long.840, https://aclanthology.org/2024.acl-long.840/
- 108. Son, W., Na, J., Choi, J., Hwang, W.: Densely guided knowledge distillation using multiple teacher assistants. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 9395–9404 (2021)
- 109. Sreenivas, S.T., Hanasoge, A.V., Yang, M., Taghibakhshi, A., Muralidharan, S., Aithal, A., Molchanov, P.: X-token: Projection-guided cross-tokenizer knowledge distillation. arXiv preprint arXiv:2605.21699 (2026)
- 110. Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., Liu, Y.: Roformer: Enhanced transformer with rotary position embedding. Neurocomputing 568, 127063 (2024)
- 111. Sujet AI, Allaa Boutaleb, H.R.: Sujet-finance-qa-vision-100k: A large-scale dataset for financial document vqa (2024), https://huggingface.co/datasets/sujetai/Sujet-Finance-QA-Vision-100k
- 112. Sun, S., Cheng, Y., Gan, Z., Liu, J.: Patient knowledge distillation for bert model compression. arXiv preprint arXiv:1908.09355 (2019)
- 113. Tan, H., Bansal, M.: Lxmert: Learning cross-modality encoder representations from transformers. arXiv preprint arXiv:1908.07490 (2019)
- 114. Team, G., Anil, R., Borgeaud, S., Wu, Y., Alayrac, J.B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A.M., Hauth, A., et al.: Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805 (2023)
- 115. Tian, Y., Han, Y., Chen, X., Wang, W., Chawla, N.V.: Beyond answers: Transferring reasoning capabilities to smaller llms using multi-teacher knowledge distillation. arXiv preprint arXiv:2402.04616 (2024)
- 116. Tian, Y., Krishnan, D., Isola, P.: Contrastive representation distillation. arXiv preprint arXiv:1910.10699 (2019)
- 117. Timiryasov, I., Tastet, J.L.: Baby llama: knowledge distillation from an ensemble of teachers trained on a small dataset with no performance penalty. arXiv preprint arXiv:2308.02019 (2023)
- 118. Tong, S., Brown, E., Wu, P., Woo, S., Middepogu, M., Akula, S.C., Yang, J., Yang, S., Iyer, A., Pan, X., et al.: Cambrian-1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860 (2024)
- 119. Tung, F., Mori, G.: Similarity-preserving knowledge distillation. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 1365–1374

(2019)

- 120. Turc, I., Chang, M.W., Lee, K., Toutanova, K.: Well-read students learn better: On the importance of pre-training compact models. arXiv preprint arXiv:1908.08962 (2019)
- 121. Van Horn, G., Mac Aodha, O., Song, Y., Cui, Y., Sun, C., Shepard, A., Adam, H., Perona, P., Belongie, S.: The inaturalist species classification and detection dataset. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 8769–8778 (2018)
- 122. Wan, F., Zhong, L., Yang, Z., Chen, R., Quan, X.: Fusechat: Knowledge fusion of chat models. arXiv preprint arXiv:2408.07990 (2024)
- 123. Wang, G.H., Ge, Y., Wu, J.: Distilling knowledge by mimicking features. IEEE Transactions on Pattern Analysis and Machine Intelligence 44(11), 8183–8195

(2021)

- 124. Wang, J., Chen, Y., Zheng, Z., Li, X., Cheng, M.M., Hou, Q.: Crosskd: Cross-head knowledge distillation for object detection. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 16520–16530 (2024)
- 125. Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., Fan, Y., Dang, K., Du, M., Ren, X., Men, R., Liu, D., Zhou, C., Zhou, J., Lin, J.: Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution (2024), https://arxiv.org/abs/2409.12191
- 126. Wang, W., Chen, Z., Wang, W., Cao, Y., Liu, Y., Gao, Z., Zhu, J., Zhu, X., Lu, L., Qiao, Y., et al.: Enhancing the reasoning ability of multimodal large language models via mixed preference optimization. arXiv preprint arXiv:2411.10442 (2024)
- 127. Wang, W., Gao, Z., Gu, L., Pu, H., Cui, L., Wei, X., Liu, Z., Jing, L., Ye, S., Shao, J., et al.: Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265 (2025)
- 128. Woo, S., Debnath, S., Hu, R., Chen, X., Liu, Z., Kweon, I.S., Xie, S.: Convnext v2: Co-designing and scaling convnets with masked autoencoders. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16133–16142 (2023)
- 129. Wu, T., Tao, C., Wang, J., Yang, R., Zhao, Z., Wong, N.: Rethinking kullbackleibler divergence in knowledge distillation for large language models. arXiv preprint arXiv:2404.02657 (2024)
- 130. Xu, K., Rui, L., Li, Y., Gu, L.: Feature normalized knowledge distillation for image classification. In: European conference on computer vision. pp. 664–680. Springer (2020)
- 131. Yang, A., Yang, B., Hui, B., Zheng, B., Yu, B., Zhou, C., Li, C., Li, C., Liu, D., Huang, F., et al.: Qwen2 technical report. arXiv preprint arXiv:2407.10671 (2024)
- 132. Yao, Y., Yu, T., Zhang, A., Wang, C., Cui, J., Zhu, H., Cai, T., Li, H., Zhao, W., He, Z., et al.: Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800 (2024)
- 133. Yim, J., Joo, D., Bae, J., Kim, J.: A gift from knowledge distillation: Fast optimization, network minimization and transfer learning. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 4133–4141

(2017)

- 134. Yu, S., Nam, D., Lee, B.K., Son, J.: Hide to see: Reasoning-prefix masking for visual-anchored thinking in vlm distillation. arXiv preprint arXiv:2605.11651

(2026)

- 135. Yu, T., Zhang, H., Yao, Y., Dang, Y., Chen, D., Lu, X., Cui, G., He, T., Liu, Z., Chua, T.S., et al.: Rlaif-v: Aligning mllms through open-source ai feedback for super gpt-4v trustworthiness. arXiv preprint arXiv:2405.17220 (2024)

- 136. Yu, W., Yang, Z., Li, L., Wang, J., Lin, K., Liu, Z., Wang, X., Wang, L.: Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490 (2023)
- 137. Yu, W., Yang, Z., Ren, L., Li, L., Wang, J., Lin, K., Lin, C.C., Liu, Z., Wang, L., Wang, X.: Mm-vet v2: A challenging benchmark to evaluate large multimodal models for integrated capabilities. arXiv preprint arXiv:2408.00765 (2024)
- 138. Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G., Stevens, S., Jiang, D., Ren, W., Sun, Y., et al.: Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. arXiv preprint arXiv:2311.16502 (2023)
- 139. Yue, X., Zheng, T., Ni, Y., Wang, Y., Zhang, K., Tong, S., Sun, Y., Yu, B., Zhang, G., Sun, H., et al.: Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813 (2024)
- 140. Yunxin Li, e.a.: Cognitive visual-language mapper: Advancing multimodal comprehension with enhanced visual knowledge alignment. Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (2024)
- 141. Zagoruyko, S., Komodakis, N.: Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. arXiv preprint arXiv:1612.03928 (2016)
- 142. Zhang, S., Zhang, X., Sun, Z., Chen, Y., Xu, J.: Dual-space knowledge distillation for large language models. arXiv preprint arXiv:2406.17328 (2024)
- 143. Zhang, Y., Zhang, R., Gu, J., Zhou, Y., Lipka, N., Yang, D., Sun, T.: Llavar: Enhanced visual instruction tuning for text-rich image understanding. arXiv preprint arXiv:2306.17107 (2023)
- 144. Zhang, Y.F., Wen, Q., Fu, C., Wang, X., Zhang, Z., Wang, L., Jin, R.: Beyond llava-hd: Diving into high-resolution large multimodal models. arXiv preprint arXiv:2406.08487 (2024)
- 145. Zhu, D., Chen, J., Shen, X., Li, X., Elhoseiny, M.: Minigpt-4: Enhancing visionlanguage understanding with advanced large language models. arXiv preprint arXiv:2304.10592 (2023)
- 146. Zong, Z., Ma, B., Shen, D., Song, G., Shao, H., Jiang, D., Li, H., Liu, Y.: Mova: Adapting mixture of vision experts to multimodal context. arXiv preprint arXiv:2404.13046 (2024)

### A Token Type Examples

|[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>A gear or gearwheel is a rotating machine part typically used to transmit rotational motion and/or torque by<br><br>means of a series of teeth that engage with compatible teeth of another gear or other part. The teeth can be integral saliences or cavities machined on the part, or separate pegs inserted into it. In the latter case, the gear is usually called a cogwheel. A cog may be one of those pegs or the whole gear. Two or more meshing gears are called a gear train.<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[290, 14586, 607, 14586, 38417, 505, 395, 41101, 5817, 1087, 11287, 1629, 442, 29583, 5902, 1813, 11531, 454, 5304, 40750, 684, 3527, 446, 395, 4169, 446, 17977, 560, 16714, 579, 18294, 17977, 446, 2595, 14586, 607, 1148, 1087, 281, 707, 17977, 777, 517, 25528, 4433, 12126, 607, 55506, 1507, 7983, 1735, 519, 410,<br><br>1087, 328, 607, 8784, 30913, 269, 21816, 1263, 563, 281, 890, 410, 15394, 1286, 328, 410, 14586, 505, 6144,<br><br>2758, 395, 60179, 38417, 281, 493, 60179, 1377, 517, 959, 446, 1995, 30913, 269, 607, 410, 4521, 14586, 281, 9173, 607, 937, 11447, 418, 52230, 657, 2758, 395, 14586, 5585, 281], Length: 103<br><br>[Figure 67]<br><br>[Figure 68]<br><br>[32, 14448, 476, 14448, 38590, 374, 264, 41396, 5662, 949, 11136, 1483, 311, 29282, 91271, 11379, 323, 5144, 41031, 553, 3363, 315, 264, 4013, 315, 17832, 429, 16579, 448, 18146, 17832, 315, 2441, 14448, 476,<br><br>1008, 949, 13, 576, 17832, 646, 387, 25098, 4274, 11975, 476, 56609, 1361, 7845, 1589, 389, 279, 949, 11,<br><br>476, 8651, 30687, 82, 21578, 1119, 432, 13, 758, 279, 15271, 1142, 11, 279, 14448, 374, 5990, 2598, 264, 61566, 38590, 13, 362, 61566, 1231, 387, 825, 315, 1846, 30687, 82, 476, 279, 4361, 14448, 13, 9043, 476, 803, 11294, 287, 53160, 525, 2598, 264, 14448, 5426, 13], Length: 102<br><br>[Figure 69]<br><br>[Figure 70]<br><br>[32, 14787, 477, 14787, 39690, 374, 264, 42496, 5780, 961, 11383, 1511, 311, 30382, 92371, 11633, 323, 5255, 42131, 555, 3445, 315, 264, 4101, 315, 18311, 430, 16988, 449, 18641, 18311, 315, 2500, 14787, 477, 1023, 961, 13, 578, 18311, 649, 387, 26154, 4371, 12242, 477, 57709, 1385, 8002, 1619, 389, 279, 961, 11,<br><br>477, 8821, 31787, 82, 22306, 1139, 433, 13, 763, 279, 15629, 1162, 11, 279, 14787, 374, 6118, 2663, 264, 62666, 39690, 13, 362, 62666, 1253, 387, 832, 315, 1884, 31787, 82, 477, 279, 4459, 14787, 13, 9220, 477, 810, 11546, 287, 54260, 527, 2663, 264, 14787, 5542, 13], Length: 102<br><br><br>InternVL2.5-2B, 8B (InternLM2.5 Tokenizer)<br><br>Qwen2-1B, 7B, 72B | InternVL2.5-1B, 4B, 78B (Qwen2.5 Tokenizer)<br><br>InternVL2-76B (Llama-3 Tokenizer)<br><br>Text<br><br>[Figure 71]<br><br>[Figure 72]<br><br>Image<br><br>|[Figure 73]<br><br>[Figure 74]|
|---|
<br><br>Image Split<br><br>| | | | |
|---|---|---|---|
| |[Figure 75]<br><br>[Figure 76]<br><br>| | |
| | | | |
<br><br>InternVL2, 2.5<br><br>Qwen2-VL<br><br>|
|---|

### B Possible Distillation Pair for VLM Combinations: Traditional Distillation versus GenRecal

|Qwen2-VL-72B<br><br>(Qwen2-72B)<br><br>InternVL2.5-78B<br><br>(Qwen2.5-72B)<br><br>InternVL2-76B<br><br>(Llama-3-70B)<br><br>InternVL2.5-8B (InternLM2.5-7B)<br><br>Qwen2-VL-7B (Qwen2-7B)<br><br>Qwen2-VL-2B (Qwen2-7B)<br><br>|X|
|---|
<br><br>InternVL2.5-4B (Qwen2.5-3B)<br><br>InternVL2.5-2B (InternLM2.5-1.8B)<br><br>InternVL2.5-1B (Qwen2.5-0.5B)<br><br>|X|
|---|
<br><br>|X|
|---|
<br><br>|X|
|---|
<br><br>|X|
|---|
<br><br>|X|
|---|
<br><br>|O|
|---|
<br><br>|O|
|---|
<br><br>|X|
|---|
<br><br>|X|
|---|
<br><br>|X|
|---|
<br><br>|X|
|---|
<br><br>|X|
|---|
<br><br>|X|
|---|
<br><br>|X|
|---|
<br><br>|O|
|---|
<br><br>|X|
|---|
<br><br>|O|
|---|
<br><br>TeacherVLM<br><br>Student VLM<br><br>VLM Name (LLM Name)|
|---|

###### TeacherVLM

(a) Traditional Distillation

|Qwen2-VL-72B (Qwen2-72B)<br><br>InternVL2.5-78B (Qwen2.5-72B)<br><br>InternVL2-76B (Llama-3-70B)<br><br>InternVL2.5-8B (InternLM2.5-7B)<br><br>Qwen2-VL-7B (Qwen2-7B)<br><br>Qwen2-VL-2B (Qwen2-7B)<br><br>|O|
|---|
<br><br>InternVL2.5-4B (Qwen2.5-3B)<br><br>InternVL2.5-2B (InternLM2.5-1.8B)<br><br>InternVL2.5-1B (Qwen2.5-0.5B)<br><br>|O|
|---|
<br><br>|O|
|---|
<br><br>|O|
|---|
<br><br>|O|
|---|
<br><br>|O|
|---|
<br><br>|O|
|---|
<br><br>|O|
|---|
<br><br>|O|
|---|
<br><br>|O|
|---|
<br><br>|O|
|---|
<br><br>|O|
|---|
<br><br>|O|
|---|
<br><br>|O|
|---|
<br><br>|O|
|---|
<br><br>|O|
|---|
<br><br>|O|
|---|
<br><br>|O|
|---|
<br><br>TeacherVLM<br><br>Student VLM<br><br>VLM Name (LLM Name)|
|---|

(b) GenRecal

- Fig. 8: We explore the range of distillation combinations between teacher and student VLMs using two approaches: (a) traditional distillation [10] and (b) our proposed model, GenRecal. Unlike traditional distillation—which supports only a limited set of pairings—GenRecal offers the flexibility to select any model for distillation, thereby enabling a more versatile and comprehensive distillation framework.

### C Extended Related Works

Evolution of Vision-Language Models. Since the emergence of visual instruction tuning [82], many variations of VLMs have emerged, such as LLaVA1.5 [80], InstructBLIP [27], ShareGPT-4V [15], and MiniGPT-v2 [13,145], typically starting at a standard model size of 7B. After that period, numerous VLMs [29, 77, 81, 89] have divided an image into multiple sub-regions to focus on the image’s details so that visual perception is enhanced. CoLLaVO [64] and MoAI [65] utilize computer vision models directly for visual capability, and Mini-Gemini [77], MoVA [146], and Eagle [104] employ multiple vision encoders such as CLIP [97], ConvNext [128], DINO-v2 [94], and SAM [51]. In parallel, Meteor [60] explores the efficient way of learning complex reasoning abilities, and TroL [57] and Phantom [56] investigate propagation modification for how we can embed vision-language knowledge as much as possible despite using the same architectures. More recently, many VLMs like Molmo-72B [28], LLaVAOneVision-72B [72], NVLM-72B [26], Qwen2-VL-72B [125] and InternVL2.578B [19] have employed large-scale language models such as Qwen2/Qwen2.572B [131]. Thanks to scaling laws [23, 44], they have closely reached the performances of GPT-4V [93] and Claude-3.5 Sonnet [4]. Therefore, developing VLMs with large-scale language models is getting to a standard these days. Beyond architectural design, recent efforts further explore training and evaluating such VLMs, including unified reinforcement and imitation learning [58], recursive think–answer reasoning [55], multimodal agentic reasoning [22,43], and multi-turn conversation benchmarking [70].

Knowledge Distillation. This provides an effective framework [36] for transferring the rich representations of large, high-performing models to smaller, efficient counterparts while preserving performance. Early studies primarily focused on aligning the output logits of a “teacher” model with those of a “student” [102,120]. Subsequently, the FitNet framework [101] extended this idea by supervising the student using intermediate feature representations rather than final outputs. In this setup, convolutional layers project the student’s features to resemble those of the teacher, and discrepancies are minimized using an L2 loss.

Following this feature-based paradigm, later methods explored more sophisticated ways of leveraging internal representations for effective knowledge transfer. For instance, [8, 16, 112, 124] demonstrated that incorporating multiple intermediate layers from the teacher significantly enhances distillation effectiveness. Probabilistic approaches such as PKT [96] reformulated the teacher’s knowledge as a probability distribution, aligning it with the student’s outputs via KL divergence. Similarly, RKD [95] captured inter-sample relationships, while CRD [116] combined contrastive learning with traditional distillation.

Multi-stage and relational strategies further broadened the field. AT [141] utilized attention maps from several teacher layers, and FSP [133] employed matrices derived from feature maps to guide the student. Later, SP [119] evaluated similarity among input samples, and OFD [35] introduced a margin-based distance metric derived from ReLU activations to capture essential information.

As the methodology matured, two main paradigms emerged. On-policy approaches [2,5,34,52] dynamically sample data during training, whereas off-policy methods [9,52,112,120,142] rely on pre-collected datasets. While early research mostly adopted single-teacher frameworks [2, 34, 92, 105], multi-teacher distillation has gained increasing attention despite challenges such as architectural differences, vocabulary mismatches, and task misalignment [71,117,122]. Representative examples include DKMF [123] and FNKD [130], which utilize teacher feature representations, while DGKD [108] aggregates knowledge from multiple teachers to boost student performance.

Recent work has also emphasized refining loss functions used in distillation. MiniLLM [34] and DistiLLM [52] propose modified KL divergences—such as reverse or skewed variants—to reduce overfitting, particularly for long-tail predictions. Similarly, [129] introduced a dynamic loss-balancing scheme that adaptively combines conventional and reverse KL divergences during training. Finally, to enhance reasoning capabilities, recent studies integrate chain-of-thought (CoT) supervision. Methods such as [39,115] leverage detailed reasoning traces from larger models, while TinyLLM [115] aggregates multi-teacher reasoning to enrich training signals and improve generalization.

Efficient AI for Vision-Language Models. In the vision-language setting, recent works distill or compress large VLMs into smaller, deployable ones: VLsI [59] transfers knowledge through verbalized layer-to-layer interactions, masking-teacher with student reinforcement [66] and reasoning-prefix masking [134] refine the teacher–student signal, visual-token pruning [46] removes redundant visual computation, and broader recipes merge, modify, and distill VLMs [54]. For language models, related efforts study prompt-based teaching [63], skill-of-mind conversational modeling [69], and refinement-capability evaluation [68]. The authors’ broader research additionally spans robustness [47,53,61,62,67] and causal representation learning [48–50].

### D Loss Graphs for Recalibrator

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

|[Figure 81]<br><br>[Figure 82]<br><br>InternVL2.5-78B<br><br>(Qwen2.5-72B)|
|---|

|[Figure 83]<br><br>[Figure 84]<br><br>InternVL2.5-78B<br><br>(Qwen2.5-72B)|
|---|

|[Figure 85]<br><br>[Figure 86]<br><br>InternVL2.5-78B<br><br>(Qwen2.5-72B)|
|---|

|[Figure 87]<br><br>[Figure 88]<br><br>InternVL2.5-78B<br><br>(Qwen2.5-72B)|
|---|

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

|[Figure 93]<br><br>[Figure 94]<br><br>InternVL2.5-8B<br><br>(InternLM2.5-7B)|
|---|

|[Figure 95]<br><br>[Figure 96]<br><br>InternVL2.5-4B<br><br>(Qwen2.5-3B)|
|---|

|[Figure 97]<br><br>[Figure 98]<br><br>InternVL2.5-2B<br><br>(InternLM2.5-1.8B)|
|---|

|[Figure 99]<br><br>[Figure 100]<br><br>InternVL2.5-1B<br><br>(Qwen2.5-0.5B)|
|---|

Cross-Entropy(Perplexity)

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

|[Figure 105]<br><br>[Figure 106]<br><br>InternVL2-76B (Llama-3-70B)|
|---|

|[Figure 107]<br><br>[Figure 108]<br><br>NVLM-72B (Qwen2-72B)|
|---|

|[Figure 109]<br><br>[Figure 110]<br><br>InternVL2.5-78B (Qwen2.5-72B)|
|---|

|[Figure 111]<br><br>[Figure 112]<br><br>Qwen2-VL-72B (Qwen2-72B)|
|---|

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

|[Figure 117]<br><br>[Figure 118]<br><br>InternVL2.5-8B (InternLM2.5-7B)|
|---|

|[Figure 119]<br><br>[Figure 120]<br><br>InternVL2.5-8B (InternLM2.5-7B)|
|---|

|[Figure 121]<br><br>[Figure 122]<br><br>Qwen2-VL-7B (Qwen2-7B)|
|---|

|[Figure 123]<br><br>[Figure 124]<br><br>Qwen2-VL-7B (Qwen2-7B)|
|---|

Training Iterations

[Figure 125]

Recalib(𝑞𝑠, 𝑎𝑙) Recalib(𝑞𝑙, 𝑎𝑙) SmallVLM (𝑞𝑠, 𝑎𝑠) LargeVLM (𝑞𝑙, 𝑎𝑙)

- Fig. 9: Illustrating the loss graphs of training Recalibrator, where we deal with various combinations of VLMs: NVLM [26], Qwen2-VL [125], InternVL2 [21], and InternVL2.5 [19]. Note that, the parenthesis in the figure means the name of LLM used in VLMs. ‘Recalib(qs, al)’ and ‘Recalib(ql, al)’ represent the cross entropy loss with Recalibrator logits and VLM-head of large VLM (see Section 3.2). ‘SmallVLM(qs, as)’ and ‘LargeVLM(ql, al)’ means original cross entropy loss for SFT without Recalibrator. They are not used in training Recalibrator. They just represent the averaged cross entropy (perplexity) during the whole training to compare them with ‘Recalib(qs, al)’ and ‘Recalib(ql, al)’.

### E Dataset Composition and Analysis

We gather a 9M visual instruction tuning dataset covering a wide range of visionlanguage capabilities, such as general visual question answering, dense image captioning, chart/diagram/document understanding, commonsense knowledge, science and math understanding, and multi-dimensional reasoning. Our dataset includes LLaVA-OneVision [72], MMC [79], DenseFusion [76], Cambrian [118], GPT-4V-filtered synthetic data of SA-1B [51] and Infinity-MM [33], FinanceQA [111], Wikipedia knowledge [140], InfoSeek [18], science & mathematical reasoning (SMR) [144], document-downstream/reasoning [40], WildVision [87], SROIE [42], RLAI-F [135], M3CoT [17], LLaVAR [143], KonIQ [38], iNaturalist2018 [121]. For further analysis, we particularly categorize the visual instruction tuning dataset that we use to build GenRecal into three domains: ‘Knowledge’, ‘Science & Math’, and ‘Chart & Document’. Based on these categories, we conduct dataset composition analysis by removing each of the three domains. This implies that MMMU [138] requires VLMs’ ability for the domain of ‘Knowledge’ more than the others, while MathVista [85] requires their capabilities for ‘Science & Math’.

[Figure 126]

[Figure 127]

[Figure 128]

MM-Vet MMMU MathVista

Accuracy(%)

Stage1 Stage2 Stage3 Stage1 Stage2 Stage3 Stage1 Stage2 Stage3

Fig. 10: Accuracy trends across different datasets: (Left) MMMU [138], and (Right) MathVista [85] over three training stages. The performance is shown for different category of visual instruction tuning dataset: “All” (without any exclusion), “w.o. Knowledge”, “w.o. Science & Math”, and “w.o. Chart & Document”. The results indicate the impact of these exclusions on accuracy progression, highlighting how the absence of specific knowledge domains affects. Note that, Note that each data point within the same stage represents 20% of the training progress, with five data points measured per stage.

- – Knowledge: GPT-4V filtered synthetic of SA-1B [51] and Infinity-MM [33], Wikipedia Knowledge [140], Wild-Vision, InfoSeek [18], WildVision [87], and iNaturalist2018 [121]
- – Science & Math: LLaVA-OneVision (subset) [72], Cambrian (subset) [118], Finance-QA [111], SMR [144], M3CoT [17], and KonIQ [38]
- – Chart & Document: LLaVA-OneVision (subset) [72], Cambrian (subset) [118], MMC [79], RLAI-F [135], document-downstream/reasoning [40], SROIE [42], and LLAVAR [143]

### F Open-Weight Teacher Dependence and SFT Fallback

GenRecal trains with the teacher’s hidden states and language head, so it assumes an open-weight teacher. This matches our central motivation—distilling heterogeneous open VLMs with different token types—rather than closed/API models. When only a closed/API teacher is available, the natural fallback is supervised fine-tuning (SFT) on teacher-generated responses. Tab. 11 compares this fallback with GenRecal on two students. Notably, a substantial portion of our 9M corpus already consists of closed/API-generated samples (e.g., InfinityMM [33], LLaVA-OneVision [72], Cambrian [118], and RLAI-F [135]; see Appendix E). Thus the “SFT only” row is effectively trained on closed/API-generated responses, yet GenRecal still outperforms it by +6.0 and +7.2 Avg. We therefore scope the general-purpose claim to open-weight teachers and state the teacheraccess assumption explicitly in the revision.

We additionally clarify our response-collection setup. Some 9M samples contain very short answers (e.g., “A” or “32”). To provide a richer distillation signal, we collect correct large-teacher responses with more detailed reasoning before training, and all experiments are based on this setup.

- Table 11: Open-weight teacher dependence and the SFT fallback for closed/API teachers. “SFT only” is trained on closed/API-generated responses drawn from the same 9M corpus, yet GenRecal still outperforms it by a large margin.

Qwen2-VL-7B student InternVL2.5-8B student

Method MMB MathVista MM-Vet MMMU Avg MMB MathVista MM-Vet MMMU Avg Baseline 83.0 58.2 62.0 54.1 64.3 84.6 64.4 62.8 56.0 67.0 w/ SFT only 84.3 60.5 64.2 56.3 66.3 86.5 65.2 65.0 59.9 69.2 w/ GenRecal 87.8 69.5 67.8 64.2 72.3 89.5 74.9 73.2 68.1 76.4 ∆ (GenRecal−SFT) +3.5 +9.0 +3.6 +7.9 +6.0 +3.0 +9.7 +8.2 +8.2 +7.2

### G Component-Level Ablation of GenRecal

We complement the stage-wise analysis (Tab. 9) with a component-level view in Tab. 12, where each row removes a single component of GenRecal and the corresponding source ablation in the main text is cited. Removing the autoregressive loss Lar is by far the most damaging, as it collapses feature alignment; removing the regularization term is the next most harmful. The new positional embedding (NPE) and a deeper Rec-body provide smaller but consistent improvements. This confirms that the Recalibrator and its training objectives—not the multi-stage SFT recipe—are responsible for the bulk of the gain.

- Table 12: Component-level ablation of GenRecal (InternVL2.5-8B student, InternVL2.5-78B teacher). Each row removes one component; the last column cites the source ablation in the main text. Avg. is over MMB / MathVista / MM-Vet / MMMU.

Variant MMB MathVista MM-Vet MMMU Avg Source Baseline (no teacher signal) 84.6 64.4 62.8 56.0 67.0 w/ SFT only (no Recalibrator) 86.5 65.2 65.0 59.9 69.2 Tab. 6(e)

GenRecal w/o Lar (KL only) 68.9 55.2 55.4 50.6 57.5 Tab. 5 GenRecal w/o regularization 88.2 69.8 63.5 58.9 70.1 Tab. 3

GenRecal w/o NPE 85.2 65.1 67.3 64.4 70.5 Tab. 6(c) GenRecal w/ depth-1 Rec-body (vs. 2) 88.0 73.0 72.0 66.5 74.9 Tab. 6(b) GenRecal (full) 89.5 74.9 73.2 68.1 76.4 —

### H Regularization Formulation Ablation

The regularization loss Lreg passes only the teacher’s own tokens [zq

] through the Recalibrator and constrains the result to remain readable by the teacher head (Algorithm 2). To motivate this specific formulation, Tab. 13 ablates alternative constraints on the same teacher input. A pointwise feature constraint (Feature MSE) that simply keeps the Recalibrator near identity helps only marginally, because matching features in ℓ2 does not guarantee they remain interpretable by the language head. Enforcing the constraint through the teacher head—either as a KL or a CE term—is substantially more effective, and combining both (our choice) is best. This explains why the teacher-only pass anchors the Recalibrator on the teacher manifold and prevents drift, complementing the cosine-diagonal analysis in Fig. 7.

,za

l

l

- Table 13: Ablation over the form of the regularization loss Lreg applied to the teacher

input [zql, zal] (InternVL2.5-8B student, InternVL2.5-78B teacher). Constraining the Recalibrator output through the teacher head is far more effective than a pointwise feature constraint. Avg. is over MMB / MathVista / MM-Vet / MMMU.

Lreg form on teacher input [zql, zal] MMB MathVista MM-Vet MMMU Avg None 88.2 69.8 63.5 58.9 70.1 Feature MSE: ∥Recalibrator([zql, zal]) − [zql, zal]∥2 88.5 71.2 65.3 61.8 71.7 KL only via VLM-headl 88.8 72.6 68.5 64.0 73.5 CE only via VLM-headl 89.1 73.7 69.8 65.6 74.6 Ours (Algorithm 2): CE + KL via VLM-headl 89.5 74.9 73.2 68.1 76.4

### I Error Profile and Calibration Analysis

Beyond aggregate benchmark scores, we analyze how GenRecal errs relative to the teacher, pooled over MathVista [85], MM-Vet [136], MMMU [138], and AI2D [45]. In Tab. 14, “Err.” is the number of wrong answers; the category columns (Perception/OCR, Spatial, Calculation, Reasoning, Hallucination) report the % distribution among errors; JSD is the Jensen–Shannon divergence between each model’s error distribution and the teacher’s; ECE is the expected calibration error; and WConf. is the average confidence on wrong answers. GenRecal is markedly closer to the teacher than SFT-only in both what it gets wrong (lower JSD) and how confident it is when wrong (lower ECE and WConf.), indicating that distillation transfers the teacher’s error structure and calibration, not merely its accuracy.

- Table 14: Error-profile and calibration analysis pooled over MathVista / MM-Vet / MMMU / AI2D. Category columns are % among wrong predictions. JSD / ECE / WConf.: lower is better; JSD is reported as 102× the raw value, and WConf. is the average confidence on wrong answers.

Method Err. Perc/OCR Spat. Calc. Reas. Hall. JSD×102 ↓ ECE↓ WConf.↓

SFT-only 520 28 14 22 23 13 0.53 0.21 0.71 GenRecal 360 34 16 19 19 12 0.02 0.13 0.58

Teacher 340 35 17 18 18 12 0 0.11 0.55

### J Compute and Baseline-Fairness Comparison

To address baseline fairness, we emphasize that all distillation baselines in Tab. 4 use the identical 9M corpus and the same optimization setup within each comparable stage; differences in the number of stages follow the original recipe of each method. Tab. 15 summarizes per-method compute (per-step model-pass FLOPs) and approximate wall-clock time on the Qwen2-VL-7B student with the Qwen2VL-72B teacher. Stage 1 of GenRecal is lightweight because only the ∼525M Recalibrator (Tab. 6(a)) receives gradients while the rest is a forward pass. As a result, despite its three-stage recipe and extra module, GenRecal’s total cost is comparable to—or lower than—the distillation baselines, while achieving substantially higher average performance.

- Table 15: Compute and baseline-fairness comparison on the Qwen2-VL-7B student with the Qwen2-VL-72B teacher. FLOPs are per-step model-pass FLOPs (same convention as Tab. 6(a)); wall-clock time is approximate. Avg. is over MMB / MathVista / MM-Vet / MMMU.

Method Recipe Data Avg FLOPs (×1012) Time SFT-only 1-stage 9M 66.3 25.7 ∼8h MiniLLM 2-stage 9M/9M 67.1 311.4 ∼21h DistiLLM 2-stage 9M/9M 67.4 287.0 ∼17h LLaVA-KD 3-stage 9M/9M/6M 67.7 285.7 ∼24h GenRecal (Ours) 3-stage 9M/9M/6M 72.3 288.0 ∼18h

