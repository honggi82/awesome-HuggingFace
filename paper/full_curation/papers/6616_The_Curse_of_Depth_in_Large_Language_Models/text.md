# arXiv:2502.05795v5[cs.LG]22Feb2026

## The Curse of Depth in Large Language Models

Wenfang Sun∗1, Xinyuan Song∗2, Pengxiang Li∗3, Lu Yin4, Yefeng Zheng1, Shiwei Liu†5 1Westlake University, China 2Emory University, USA 3Dalian University of Technology, China

- 4University of Surrey, UK
- 5University of Oxford, UK Abstract

In this paper, we introduce the Curse of Depth, a concept that highlights, explains, and addresses the recent observation in modern Large Language Models (LLMs) where nearly half of the layers are less effective than expected. We first confirm the wide existence of this phenomenon across the most popular families of LLMs such as Llama, Mistral, DeepSeek, and Qwen. Our analysis, theoretically and empirically, identifies that the underlying reason for the ineffectiveness of deep layers in LLMs is the widespread usage of Pre-Layer Normalization (Pre-LN). While Pre-LN stabilizes the training of Transformer LLMs, its output variance exponentially grows with the model depth, which undesirably causes the derivative of the deep Transformer blocks to be an identity matrix, and therefore barely contributes to the training. To resolve this training pitfall, we propose LayerNorm Scaling (LNS), which scales the variance of output of the layer normalization inversely by the square root of its depth.1 This simple modification mitigates the output variance explosion of deeper Transformer layers, improving their contribution. Across a wide range of model sizes (130M to 7B), our experiments show that LNS consistently outperforms previous normalization and scaling techniques in enhancing LLM pre-training performance. Moreover, this improvement seamlessly carries over to supervised fine-tuning. All these gains can be attributed to the fact that LayerNorm Scaling enables deeper layers to contribute more effectively during training. Our code is available at LayerNorm-Scaling.

###### Model Parameter Scaling

Pre-LN (Baseline) Ours

FFN

| |
|---|

3.6

FFN

| | |
|---|---|
|Scal|ing|
| | |

| |
|---|

3.4

Layer Norm

Layer Norm

| |
|---|

3.2

###### Loss

| |
|---|

3.0

66% Fewer Params

Attention

Attention

| | |
|---|---|
|Scal|ing|
| | |

| |
|---|

2.8

Layer Norm

Layer Norm

2.6

| |
|---|

2.4

107 108 108.5 109 7 × 109

#Parameters (log scale)

(a) Pre-LN

(b) LayerNorm Scaling

Figure 1: Left: Schematic diagrams of (a) Pre-LN and (b) LayerNorm Scaling. LayerNorm Scaling applies a scaling factor inversely proportional to the square root of the layer index ℓ, preventing excessive variance growth. Right: Language modeling loss of scaling up parameter count up to 7B. All models are trained for 20B tokens using OLMo (Groeneveld et al., 2024).

*Equal contribution. Accepted at NeurIPS 2025. †Correspondence to: Shiwei Liu, shiwei.liu@maths.ox.ac.uk. 1We found that combining LNS with Scaled Initialization (Groeneveld et al., 2024; Radford et al., 2019; Shoeybi et al., 2020)

diminishes the effectiveness of LNS. Therefore, we recommend removing the latter when applying LNS.

### Contents

- 1 Introduction 3
- 2 Empirical Evidence of the Curse of Depth 4

- 2.1 Open-weight Large-scale LLMs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 2.2 In-house Small-scale LLaMa-130M . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5

- 3 Analysis of the Curse of Depth 6

- 3.1 Pre-LN Transformers . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

4 LayerNorm Scaling (LNS) 8

- 4.1 Theoretical Analysis of LayerNorm Scaling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8

5 Experiments 10

- 5.1 LLM Pre-training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10

- 6 Ablation Study 14
- 7 Related Work 15
- 8 Conclusion 16

- 5.2 Supervised Fine-tuning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 5.3 Scaling Up Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11

- 5.3.1 OLMo . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 5.3.2 Qwen2.5 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12

- 5.4 LNS Effectively Scales Down Output Variance . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- 5.5 LNS Enhances the Effectiveness of Deep Layers . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- 5.6 LayerNorm Scaling in Vision Transformer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13

- A Proofs of the Theorems of curse of depth 20

- A.1 Proof of Lemma 3.2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20

- A.1.1 Variance of the Attention . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- A.1.2 Variance of the Feed-Forward Network . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- A.2 Proof of Theorem 3.3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- A.2.1 Proof of Lemma A.2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- A.2.2 Analysis of the Upper Bound . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23

- A.3 Proof of Lemma 4.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- A.4 Proof of Theorem 4.2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- A.5 Proof of theorem 4.3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27

- B Variance Growth in Pre-LN Training 29
- C Performance Drop of Layer Pruning in Vision–Language Models (Qwen 2.5-VL) 29
- D Limitations 29

### 1 Introduction

Recent studies reveal that the deeper layers (Transformer blocks) in modern LLMs tend to be less effective than the earlier ones (Gromov et al., 2024; Li et al., 2024b; Men et al., 2024; Yin et al., 2024). On the one hand, this interesting observation provides an effective indicator for LLM compression. For instance, we can compress deeper layers significantly more (Dumitru et al., 2024; Lu et al., 2024; Yin et al., 2024) to achieve high compression ratios. Even more aggressively, entire deep layers can be pruned completely without compromising performance (Muralidharan et al., 2024; Siddiqui et al., 2024).

On the other hand, having many layers ineffective is undesirable as modern LLMs are extremely resourceintensive to train, often requiring thousands of GPUs trained for multiple months, let alone the labor used for data curation and administration (Achiam et al., 2023; Touvron et al., 2023). Ideally, we want all layers in a model to be well-trained, with sufficient diversity in features from layer to layer, to maximize the utility of resources (Li et al., 2024b). The existence of ill-trained layers suggests that there must be something off with current LLM paradigms. Addressing such limitations is a pressing need for the community to avoid the waste of valuable resources, as new versions of LLMs are usually trained with their previous computing paradigm which results in ineffective layers.

To seek the immediate attention of the community, we re-introduce the concept of the Curse of Depth (CoD) to systematically present the phenomenon of ineffective deep layers in various LLM families, to identify the underlying reason behind it, and to rectify it by proposing LayerNorm Scaling. We first state the Curse of Depth below.

The Curse of Depth. The Curse of Depth refers to the observed phenomenon where deeper layers in modern LLMs contribute significantly less (but not nothing) to learning and representation compared to earlier layers. These deeper layers often exhibit remarkable robustness to pruning and perturbations, implying they fail to perform meaningful transformations. This behavior prevents these layers from effectively contributing to training and representation learning, resulting in resource inefficiency.

Empirical Evidence of CoD. To demonstrate that CoD is a common phenomenon across prominent LLM families, we perform layer pruning experiments on Qwen3, LLaMA2, and DeepSeek. Specifically, we prune one layer at a time, without any fine-tuning, and directly evaluate the resulting pruned models on the MMLU benchmark (Hendrycks et al., 2021), as shown in Figure 2. Key findings: (1) Most models, including the latest Qwen3, exhibit surprising resilience to the removal of deeper layers; (2) The number of layers that can be removed without causing significant performance drop increases with model size; (3) Representations in deeper layers are significantly more similar to each other than those in earlier layers.

Identifying the Root Cause of CoD. We theoretically and empirically identify the root cause of CoD as the use of Pre-Layer Normalization (Pre-LN) (Baevski and Auli, 2019; Dai et al., 2019), which normalizes layer inputs before applying the main computations, such as attention or feedforward operations, rather than after. Specifically, while stabilizing training, we observe that the output variance of Pre-LN accumulates significantly with layer depth as shown in Figure 4, causing the derivatives of deep Pre-LN layers to approach an identity matrix. This behavior prevents these layers from introducing meaningful transformations, leading to diminished representation learning.

##### Mitigating CoD through LayerNorm Scaling. We propose LayerNorm Scaling (LNS), which scales

the output of Layer Normalization by the square root of the depth √1l. LayerNorm Scaling effectively scales down the output variance across layers of Pre-LN. LNS consistently delivers better pre-training performance

than existing normalization and scaling techniques across various model sizes from 130M to 7B. Unlike previous LayerNorm variants (Li et al., 2024b; Liu et al., 2020), LayerNorm Scaling is simple to implement, requires no hyperparameter tuning, and introduces no additional parameters during training. Furthermore, we show that the model pre-trained with LayerNorm Scaling achieves better performance on downstream tasks in self-supervised fine-tuning, all thanks to the more diverse feature representations learned in deep layers.

(a) BERT-Large

(b) DeepSeek-7B

(c) Qwen3-8B

(d) LLaMa2-13B

0.0

0

0

0

−5

−0.2

−10

−5

(ℓ)SQuADΔP

−10

(ℓ)MMLUΔP

(ℓ)MMLUΔP

(ℓ)MMLUΔP

−0.4

−20

−10

−15

−0.6

−30

−20

−15

−0.8

−40

−25

−1.0

−20

Poriginal ΔP(ℓ)

Poriginal ΔP(ℓ)

Poriginal ΔP(ℓ)

Poriginal

−50

ΔP(ℓ)

−30

| |
|---|

| |
|---|

| |
|---|

0 5 10 15 20

0 5 10 15 20 25 30

0 5 10 15 20 25 30 35

0 5 10 15 20 25 30 35 40

Layer Index ℓ

Layer Index ℓ

Layer Index ℓ

Layer Index ℓ

(e) BERT-Large Angular Distance

(f) DeepSeek-7B Angular Distance

(g) Qwen3-8B Angular Distance

(h) LLaMa2-13B Angular Distance

1.0

1.0

1.0

1.0

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

| | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

thSubsequentnLayer

thSubsequentnLayer

thSubsequentnLayer

thSubsequentnLayer

16

0.8

20

0.8

20

0.8

20

0.8

12

0.6

15

0.6

15

0.6

15

0.6

8

0.4

10

0.4

10

0.4

10

0.4

4

0.2

5

0.2

5

0.2

5

0.2

0

0.0

0

0.0

0

0.0

0

0.0

0 4 8 12 16 20

0 5 10 15 20 25

0 6 12 18 24 30

0 6 12 18 24 30 36

Layer Index ℓ

Layer Index ℓ

Layer Index ℓ

Layer Index ℓ

- Figure 2: Results of open-weight large-scale LLMs. Top: Performance drop after removing a single layer without fine-tuning. Bottom: Angular distance from the initial layer ℓ (x-axis) and its subsequent nth layer (y-axis). The results demonstrate that in Pre-LN LLMs, deeper layers produce highly similar representations to their adjacent layers, and their removal results in minimal performance degradation. In contrast, Post-LN models show the opposite trend: deep layers contribute more substantially to model performance.

### 2 Empirical Evidence of the Curse of Depth

To empirically analyze the impact of layer normalization on the Curse of Depth in LLMs, we conduct a series of evaluations inspired by Li et al. (2024b), to compare Pre-LN and Post-LN models.

Methodology: We evaluate Pre-LN and Post-LN models by assessing the impact of layer pruning at different depths. Our hypothesis is that Pre-LN models exhibit diminishing effectiveness in deeper layers, whereas Post-LN models have less effective early layers.

#### 2.1 Open-weight Large-scale LLMs

Models: To verify this, we empirically quantify the contribution of individual layers to overall model performance across a diverse set of LLMs, including Qwen3 (Team, 2025), LLaMA2 (Touvron et al., 2023), DeepSeek (Bi et al., 2024), and BERT-Large (Devlin, 2019). These models were chosen to ensure architectural and application diversity. BERT-Large represents a Post-LN model, whereas the rest are Pre-LN-based. This selection enables a comprehensive evaluation of the effects of layer normalization across varying architectures and model scales.

Evaluation Metric: To empirically assess the impact of deeper layers in LLMs, we adopt two metrics, Performance Drop and Angular Distance, inspired by Gromov et al. (2024); Li et al. (2024b).

Performance Drop ∆P(ℓ) quantifies the importance of each layer by measuring the performance change after its removal. A smaller ∆P(ℓ) indicates that the pruned layer contributes less to the model’s overall performance. For BERT-Large, we evaluate using the SQuAD v1.1 dataset (Rajpurkar, 2016), while for other models, we use MMLU (Hendrycks et al., 2021), a standard benchmark for multi-task language understanding.

Angular Distance d(xℓ,xℓ+n) quantifies the directional change between the input representations at layer

ℓ and layer ℓ + n on a neutral pre-training dataset. Formally, given a token T, let xℓT and xℓT+n denote its input to layers ℓ and ℓ + n, respectively. The angular distance is defined as:

d(xℓ,xℓ+n) =

1 π

arccos

xℓT · xℓT+n ∥xℓT∥2∥xℓT+n∥2

, (1)

(a) Post-LN Angular Distance

(b) Pre-LN Angular Distance

0.4

[Figure 5]

[Figure 6]

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

357 SubsequentLayerthn

357 SubsequentLayerthn

0.4

0.3

0.3

0.2

0.2

0.1

0.1

0 3 6 9

0 3 6 9

Layer Index

Layer Index

(c) Post-LN Performance Drop

(d) Pre-LN Performance Drop

5

5

Poriginal

Poriginal

0

0

()ARCP

()ARCP

- -15
- -10
- -5

- -15
- -10
- -5

0 3 6 9

0 3 6 9

Layer Index

Layer Index

- Figure 3: Results of in-house small-scale LLaMa-130M. Angular Distance (a, b): Each column represents the angular distance from the initial layer ℓ (x-axis) and its subsequent nth layer (y-axis). The distance is scaled to the range [0, 1], where yellow indicates smaller distances and purple indicates larger distances. Performance Drop (c, d): ARC-e performance drop of removing each single layer from LLaMa-130M.

where ∥ · ∥2 denotes the L2-norm. To reduce variance, we report the average distance over 256K tokens sampled from the C4 dataset. Smaller values of d(xℓ,xℓ+n) indicate higher similarity between the two representations, suggesting limited transformation. Such layers can be considered redundant, as their removal minimally impacts the model’s internal representations. Ideally, each layer should introduce meaningful representational shifts to fully leverage the model’s capacity (Gromov et al., 2024; Yang et al., 2023).

Experimental Results: (1) Pruning deep layers in Pre-LN LLMs leads to negligible, and sometimes even positive, changes in performance, as shown in Figure 2-Top. Specifically, Figure 2 (b)–(d) reveals that a wide range of deeper layers—particularly beyond the 18th—can be pruned with minimal impact on performance. This indicates that deep layers in Pre-LN architectures contribute little to the model’s overall effectiveness. In contrast, Figure 2 (a) shows that pruning deep layers in BERT-Large (a Post-LN model) leads to a substantial drop in accuracy, while pruning early layers has a relatively minor effect. (2) Pre-LN models exhibit decreasing angular distance in deeper layers, indicating highly similar representations, as shown in Figure 2-Bottom. For instance, the angular distance in DeepSeek-7B falls below 0.2 after the 18th layer. Qwen3-8B demonstrates a higher similarity, with nearly half of its layers exhibiting distances below 0.2 from their preceding layers. In LLaMA2-13B, the angular distance approaches zero across the final one-third of the network. These similar representations align with the pruning results in Figures 2 (b)–(d), where pruning later layers has little effect, while pruning early layers significantly degrades performance.

#### 2.2 In-house Small-scale LLaMa-130M

To eliminate the influence of other confounding variables, we train two LLaMA-130M models from scratch that differ only in their Layer Normalization, thereby clearly distinguishing Post-LN from Pre-LN, following Li et al. (2024b). The results are illustrated in Figure 3.

In Post-LN models, early layers exhibit high similarity (low angular distance, Figure 3-a) and their removal causes minimal performance loss (Figure 3-c), while deeper layers become more distinct and critical. Conversely, Pre-LN LLaMa-130M demonstrates a gradual decrease in angular distance with depth, resulting in highly similar deep layers (Figure 3-b). Removing most layers after the first in Pre-LN causes negligible performance loss (Figure 3-d), indicating their limited contribution. These consistent findings, observed in

Layer 0 Layer 1 Layer 2 Layer 3 Layer 4 Layer 5 Layer 6 Layer 7 Layer 8 Layer 9 Layer 10 Layer 11

###### Pre-LN

###### Pre-LN + Scaled Initialization

LayerNorm Scaling

200

25

175

175

AverageVariance(Var)

AverageVariance(Var)

AverageVariance(Var)

150

150

20

125

125

15

100

100

75

75

10

50

50

5

25

25

0

0

0

0 2000 4000 6000 8000 10000

0 2000 4000 6000 8000 10000

0 2000 4000 6000 8000 10000

Update Step

Update Step

Update Step

- Figure 4: Layerwise output variance. This figure compares the output variance across various layers for different setups: (1) Pre-LN; (2) Pre-LN with Scaled Initialization (Radford et al., 2019; Shoeybi et al., 2020); and (3) LayerNorm Scaling. The experiments are conducted on the LLaM-130M model trained for 10,000 steps. The proposed LayerNorm Scaling effectively controls the variance across layers.

both open-weight and in-house LLMs, lead to the conclusion that the widespread use of Pre-LN is the root cause of the ineffectiveness of deep layers in LLMs.

### 3 Analysis of the Curse of Depth

Preliminaries. This paper primarily focuses on Pre-LN Transformer (Baevski and Auli, 2019; Dai et al., 2019). Let xℓ ∈ Rd be the input vector at the ℓ-th layer of Transformer, where d denotes the feature dimension of each layer. For simplicity, we assume all layers to have the same dimension d. The layer output y is calculated as follows:

y = xℓ+1 = x′ℓ + FFN(LN(x′ℓ)), (2)

x′ℓ = xℓ + Attn(LN(xℓ)), (3) where LN denotes the layer normalization function. In addition, the feed-forward network (FFN) and the multi-head self-attention (Attn) sub-layers are defined as follows:

FFN(x) = W2F(W1x), Attn(x) = WO(concat(head1(x),...,headh(x))),

(WQix)⊤(WKiX) √dhead

(WV iX)⊤,

headi(x) = softmax

(4)

where F is an activation function, concat concatenates input vectors, softmax applies the softmax function, and W1 ∈ Rd

head×d, and WO ∈ Rd×d are parameter matrices, and dFFN and dhead are the internal dimensions of FFN and multi-head self-attention sub-layers, respectively. X ∈ Rd×s, where s is the input sequence length.

ffn×d, W2 ∈ Rd×d

, WQi ∈ Rd

head×d, WKi ∈ Rd

head×d, WV i ∈ Rd

ffn

The derivatives of Pre-Ln Transformers are:

∂Pre-LN(x) ∂x

∂f(LN(x)) ∂LN(x)

∂LN(x) ∂x

= I +

, (5)

where f here represents either the multi-head attention function or the FFN function. If the term

∂x becomes too small, the Pre-LN layer ∂Pre-LN(∂x x) behaves like an identity map. Our main objective is to prevent identity map behavior for very deep Transformer networks. The first step in this process is to compute the variance σx2

∂f(LN(x)) ∂LN(x)

∂LN(x)

of vector xℓ.

ℓ

#### 3.1 Pre-LN Transformers

Assumption 3.1. Let xℓ and x′ℓ denote the input and intermediate vectors of the ℓ-th layer. Moreover, let Wℓ denote the model parameter matrix at the ℓ-th layer. We assume that, for all layers, xℓ, x′ℓ, and Wℓ follow normal and independent distributions with mean µ = 0.

denote the variances of x′ℓ and xℓ, respectively. These two variances exhibit the same overall growth trend, which is:

##### Lemma 3.2. Let σx2′

and σx2

ℓ

ℓ

σx2

ℓ

= σx2

Θ

1

ℓ−1

k=1

1 σx

1 +

k

, (6)

where the growth of σx2

ℓ

is sub-exponential, as shown by the following bounds:

Θ(L) ≤ σx2

L

≤ Θ(exp(L)). (7)

Here, the notation Θ means: if f(x) ∈ Θ g(x) , then there exist constants C1,C2 such that C1|g(x)| ≤ |f(x)| ≤ C2|g(x)| as x → ∞. The lower bound Θ(L) ≤ σx2

indicates that σx2

grows at least linearly, while the upper bound σx2

ℓ

ℓ

≤ Θ(exp(L)) implies that its growth does not exceed an exponential function of L. Based on Assumption 3.1 and the work of (Takase et al., 2023b), we obtain the following:

ℓ

- Theorem 3.3. For a Pre-LN Transformer with L layers, using Equations (2) and (3), the partial derivative ∂yL ∂x1 can be written as:

L−1

∂x′ℓ ∂xℓ

∂yℓ ∂x′

∂yL ∂x1

·

=

. (8)

ℓ

ℓ=1

The Euclidean norm of ∂y

∂x1 is given by:

L

L−1

∂yL ∂x1 2 ≤

l=1

1 σx

1 σx2

1 +

A +

B , (9)

ℓ

ℓ

where A and B are constants for the Transformer network. Then the upper bound for this norm is given as follows: when σx2

grows exponentially, (i.e., at its upper bound), we have:

ℓ

σx2

ℓ

∼ exp(ℓ),

∂yL ∂x1 2 ≤ M, (10)

where the gradient norm converges to a constant M. Conversely, when σx2

grows linearly (i.e., at its lower bound), we have

ℓ

∂yL ∂x1 2 ≤ Θ(L), (11)

σx2

∼ ℓ,

ℓ

which means that the gradient norm grows linearly in L. The detailed description of A and B, as well as the complete proof, are provided in Appendix A.2. From

- Theorem 3.3, we observe that when the variance grows exponentially, as the number of layers L → ∞,

the norm ∂y

is bounded above by a fixed constant M. This result implies that even an infinitely deep Transformer remains stable, and by the Weierstrass Theorem, the network is guaranteed to converge. Consequently, this implies that for very large L, deeper layers behave nearly as an identity map from xℓ to yℓ, thereby limiting the model’s expressivity and hindering its ability to learn meaningful transformations. This phenomenon is empirically illustrated in Figure 5, where we visualize the Jacobian of the pre-LN residual

L ∂x1

2

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

- Figure 5: Visualization of the Jacobian matrices of pre-LN residual blocks across different layers of a pre-trained LLaMA2-7B model. Each heatmap shows the token-averaged Jacobian at a specific layer. As depth increases, the Jacobians exhibit a pronounced diagonal dominance with vanishing off-diagonal entries, indicating that deep LayerNorm blocks increasingly approximate identity mappings.

blocks across depth in a pre-trained LLaMA2-7B model, revealing a clear collapse toward identity mappings in deeper layers. This outcome is undesirable, therefore, we would instead prefer the variance to increase more gradually—e.g., linearly—so that ∂y

exhibits linear growth. This observation highlights the necessity of appropriate variance control mechanisms, such as scaling strategies, to prevent excessive identity mappings and enhance network depth utilization.

L ∂x1

2

### 4 LayerNorm Scaling (LNS)

To mitigate the abovementioned issue, we propose LayerNorm Scaling, a simple yet effective normalization strategy. The core idea of LayerNorm Scaling is to control the exponential growth of output variance in Pre-LN by scaling the normalized outputs according to layer depth. Specifically, we apply a scaling factor inversely proportional to the square root of the layer index to scale down the output of LN layers, enhancing the contribution of deeper Transformer layers during training. LayerNorm Scaling is illustrated in Figure 1.

Formally, for a Transformer model with L layers, the output of Layer Normalization in each layer ℓ is

scaled by a factor of √1ℓ. Let h(ℓ) denote the input to Layer Normalization at layer ℓ. The modified output is computed as:

1 √

h(ℓ) = LayerNorm(h(ℓ)) ×

, (12)

ℓ

where ℓ ∈ {1,2,...,L}. This scaling prevents excessive variance growth with depth, addressing a key limitation of Pre-LN. Unlike Mix-LN, which stabilizes gradients in deeper layers but suffers from training instability caused by Post-LN (Nguyen and Salazar, 2019; Wang et al., 2024), LayerNorm Scaling preserves the stability advantages of Pre-LN while enhancing the contribution of deeper layers to representation learning. Applying LayerNorm Scaling leads to a notable reduction of layerwise output variance as shown in Figure 4, resulting in a lower training loss. Moreover, compared with previous LayerNorm variants (Li et al., 2024b; Liu et al., 2020), LayerNorm Scaling is hyperparameter-free, easy to implement, and does not introduce additional learnable parameters, making it computationally efficient and readily applicable to existing Transformer architectures.

#### 4.1 Theoretical Analysis of LayerNorm Scaling

- Lemma 4.1. After applying our scaling method, the variances of x′ℓ and xℓ, denoted as σx2′

and σx2

, respectively, exhibit the same growth trend, which is:

ℓ

ℓ

σx2

ℓ

= σx2

Θ

1

ℓ−1

k=1

1 √

1 +

kσx

k

, (13)

with the following growth rate bounds:

≤ Θ(L(2−ϵ)). (14) where ϵ is a small number with 1/2 ≤ ϵ < 1.

Θ(L) ≤ σx2

L

From Lemma 4.1, we can conclude that our scaling method effectively slows the growth of the variance upper bound, reducing it from exponential to polynomial growth. Specifically, it limits the upper bound to a quadratic rate instead of an exponential one. Based on Theorem 3.3, after scaling, we obtain the following:

- Theorem 4.2. For the scaled Pre-LN Transformers, the Euclidean norm of ∂y

∂x1 is given by:

L

L−1

∂yL ∂x1 2 ≤

ℓ=1

1 ℓσx

1 ℓ2σx2

B , (15)

1 +

A +

ℓ

ℓ

where A and B are dependent on the scaled neural network parameters. Then the upper bound for the norm is given as follows: when σx2

###### grows at ℓ(2−ϵ), (i.e., at its upper bound), we obtain:

ℓ

σx2

ℓ

∼ ℓ(2−ϵ),

∂yL ∂x1 2 ≤ ω(1), (16)

where ω denotes that if f(x) = ω(g(x)), then limx→∞ fg((xx)) = ∞. Meanwhile, when σx2

grows linearly (i.e., at its lower bound), we obtain:

ℓ

∂yL ∂x1 2 ≤ Θ(L). (17)

σx2

∼ ℓ,

ℓ

The detailed descriptions of A and B, and ϵ, along with the full proof, are provided in Appendices A.3 and A.4.

By comparing Theorem 3.3 (before scaling) with Theorem 4.2 (after scaling), we observe a substantial reduction in the upper bound of variance. Specifically, it decreases from exponential growth Θ(exp(L)) to at most quadratic growth Θ(L2). In fact, this growth is even slower than quadratic expansion, as it follows Θ(L(2−ϵ)) for some small ϵ > 0.

When we select a reasonable upper bound for this expansion, we find that ∂y

no longer possesses a strict upper bound. That is, as the depth increases, ∂y

L ∂x1

2

continues to grow gradually. Consequently, fewer layers act as identity mappings compared to the original Pre-LN where nearly all deep layers collapsed into identity transformations. Instead, the after-scaled network effectively utilizes more layers, even as the depth approaches infinity, leading to improved expressivity and trainability.

L ∂x1

2

In addition to making the deeper layers more effective, our variance-scaling approach can also reduce sudden spikes in the loss landscape during training. Based on (Takase et al., 2023b)’s work, We formalize this in the following theorem Theorem 4.3, which gives a rigorous upper bound on the gradient norm with respect to the attention parameters.

- Theorem 4.3. For the Pre-Ln transformers with weight W1 on its first layer’s query projection. Then the L-layer backpropagated gradient norm with respect to W1 satisfies the following upper bound:

L−1

∂yL ∂W1 2 ≤

ℓ=1

1 ℓ2σx2

1 ℓσx

A′ +

B′ , (18)

1 +

ℓ

ℓ

where A′ and B′ are dependent on the scaled neural network parameters defined in A.5.

From (18), we can easily get that if we do not want so many loss spikes, we need to let the ∂y

do not explode. Which in our assumption means that the variance of the deep layer should not be too small. Based on the above result (15), the good variance growth rate is sub linearly growth. which is:

L ∂W1

2

σx2

∼ ℓ, (19)

ℓ

which is actually the LayerNorm Scaling convergence rate. Therefore, the LayerNorm Scaling method can provide a moderate scaling of the variance, both to make the deeper layers effective and to prevent the initial layers from exploding.

The proof of Theorem 4.3 is in Section A.5. Then we can easily generalize to a more general situation for layer l. By carefully controlling the propagation of gradients through the attention blocks, we can observe that for every layer ℓ, the ∂y

has the same upper bound as in Result 18. The proof is omitted here. As a result, LayerNorm Scaling improves stability for every layer (especially the first layer) and avoids sharp gradient amplification, which would otherwise result in an unstable or inefficient optimization process.

L ∂Wℓ

2

### 5 Experiments

#### 5.1 LLM Pre-training

To evaluate the effectiveness of LayerNorm Scaling, we follow the experimental setup of Li et al. (2024b), using the identical model configurations and training conditions to compare LNS with widely used normalization techniques, including Post-LN (Nguyen and Salazar, 2019), DeepNorm (Wang et al., 2024), and Pre-LN (Dai et al., 2019). In line with Lialin et al. (2023) and Zhao et al. (2024), we conduct experiments using LLaMA-based architectures with model sizes of 130M, 250M, 350M, and 1B parameters.

Table 1: Perplexity (↓) comparison of various layer normalization methods.

LLaMA-130M LLaMA-250M LLaMA-350M LLaMA-1B Training Tokens 2.2B 3.9B 6.0B 8.9B Post-LN (Ba, 2016) 26.95 1409.79 1368.33 1390.75 DeepNorm (Wang et al., 2024) 27.17 22.77 1362.59 1409.08 Mix-LN (Li et al., 2024b) 26.07 21.39 1363.21 1414.78 Pre-LN (Baevski and Auli, 2019) 26.73 21.92 19.58 17.02 Pre-LN + LayerNorm Scaling 25.76 20.35 18.20 15.71

The architecture incorporates RMSNorm (Shazeer, 2020) and SwiGLU activations (Zhang and Sennrich, 2019), which are applied consistently across all model sizes and normalization methods. For optimization, we use the Adam optimizer (Kingma, 2015) and adopt size-specific learning rates: 1 × 10−3 for models up to 350M parameters, and 5 × 10−4 for the 1B parameter model. All models share the same architecture, hyperparameters, and training schedule, with the only difference being the choice of normalization method. Unlike Mix-LN (Li et al., 2024b), which introduces an additional hyperparameter α manually set to 0.25, LayerNorm Scaling requires no extra hyperparameters, making it simpler to implement. Table 1 shows that LayerNorm Scaling consistently outperforms other normalization methods across different model sizes. While DeepNorm performs comparably to Pre-LN on smaller models, it struggles with larger architectures like LLaMA-1B, showing signs of instability and divergence in loss values. Similarly, Mix-LN outperforms Pre-LN in smaller models but faces convergence issues with LLaMA-350M, indicating its sensitivity to architecture design and hyperparameter tuning due to the introduction of Post-LN. Notably, Mix-LN was originally evaluated on LLaMA-1B with 50K steps (Li et al., 2024b), while our setting extends training to 100K steps, where Mix-LN fails to converge, highlighting its instability in large-scale settings caused by the usage of Post-LN.

Table 2: Fine-tuning performance (↑) of LLaMA with various layer normalizations.

Method MMLU BoolQ ARC-e PIQA Hellaswag OBQA Winogrande Average LLaMA-250M

Post-LN (Ba, 2016) 22.95 37.83 26.94 52.72 26.17 11.60 49.56 32.54 DeepNorm (Wang et al., 2024) 23.60 37.86 36.62 61.10 25.69 15.00 49.57 35.63 Mix-LN (Li et al., 2024b) 26.53 56.12 41.68 66.34 30.16 18.00 50.56 41.34 Pre-LN (Baevski and Auli, 2019) 24.93 38.35 40.15 63.55 26.34 16.20 49.01 36.93

- Pre-LN + LayerNorm Scaling 27.08 58.17 45.24 67.38 32.81 18.80 52.49 43.14

LLaMA-1B

Post-LN (Ba, 2016) 22.95 37.82 25.08 49.51 25.04 13.80 49.57 31.96 DeepNorm (Wang et al., 2024) 23.35 37.83 27.06 52.94 26.19 11.80 49.49 32.67 Mix-LN (Li et al., 2024b) 23.19 37.83 25.08 49.51 25.04 11.80 49.57 31.72 Pre-LN (Baevski and Auli, 2019) 26.54 62.20 45.70 67.79 30.96 17.40 50.51 43.01

- Pre-LN + LayerNorm Scaling 28.69 61.80 48.85 67.92 33.94 18.60 54.30 44.87

In contrast, LayerNorm Scaling solves the Curse of Depth without compromising the training stability. LayerNorm Scaling achieves the lowest perplexity across all tested model sizes, showing stable performance improvements over existing methods. For instance, on LLaMA-130M and LLaMA-1B, LayerNorm Scaling reduces perplexity by 0.97 and 1.31, respectively, compared to Pre-LN. Notably, LayerNorm Scaling maintains stable training dynamics for LLaMA-1B, a model size where Mix-LN fails to converge. These findings demonstrate that LayerNorm Scaling provides a robust and computationally efficient normalization strategy, enhancing large-scale training of language models without additional implementation complexity.

#### 5.2 Supervised Fine-tuning

To verify whether the gains in pre-training can be translated to the stage of post-training, we perform SFT with the models obtained from Section 5.1 on the Commonsense170K dataset (Hu et al., 2023) across eight downstream tasks. We adopt the same fine-tuning configurations as used in Li et al. (2024b). The results, presented in Table 2, demonstrate that LayerNorm Scaling consistently surpasses other normalization techniques in all evaluated datasets. For the LLaMA-250M model, LayerNorm Scaling improves average performance by 1.80% and achieves a 3.56% gain on ARC-e compared to Mix-LN. Similar trends are observed with the LLaMA-1B model, where LayerNorm Scaling outperforms Pre-LN, Post-LN, Mix-LN, and DeepNorm on seven out of eight tasks, with an average gain of 1.86% over the best baseline. These results confirm that LayerNorm Scaling enhances generalization on diverse downstream tasks by improving the representation quality of deep layers.

#### 5.3 Scaling Up Training

##### 5.3.1 OLMo

Model Size Scaling. To further assess the scalability and robustness of LNS, we conduct additional experiments using the OLMo repository (Groeneveld et al., 2024), scaling training across model sizes of 60M, 150M, 300M, 1B, and 7B parameters. All models are trained on a fixed 20B-token budget to ensure comparability. These experiments are designed to evaluate whether the performance gains observed with LNS in smaller-scale settings extend to more challenging and state-of-the-art LLM training regimes. As shown in Figure 1, LNS consistently and substantially outperforms the standard Pre-LN baseline across all model sizes. Remarkably, for the 7B model, LNS reduces the final loss from 2.69 to 2.50. These results underscore the scalability of LNS and its effectiveness in large-scale pre-training scenarios.

Loss Curve. Figure 6 shows the training loss curves of 7B models trained with Pre-LN and LNS. While LayerNorm Scaling exhibits slightly slower convergence at the early stages of training, it consistently outperforms Pre-LN as training progresses, ultimately achieving a substantial loss gap. We attribute this to the uncontrolled accumulation of output variance in Pre-LN, which amplifies with depth and training steps,

Comparison of Training Loss Curves for OLMo-7B

Pre-LN

LayerNorm Scaling

3.4

3.2

Loss

3.0

2.8

2.6

5000 10000 15000 20000 25000 30000 35000

Step

Figure 6: Training loss of OLMo-7B with Pre-LN and LNS.

ultimately impairing the effective learning of deeper layers. In contrast, LNS mitigates this issue by scaling down the output variance in proportion to depth, thereby enabling more stable and effective training across all layers during training.

Beating OLMo’s Scaled Initialization. OLMo adopts the scaled initialization proposed in Zhang et al. (2019) and used by Mehta et al. (2024), which scales input projections by 1/√dmodel, and output projections by 1/√2 · dmodel · l at every layer. This method is designed to enhance training stability and to scale down variance at initialization. To evaluate the effectiveness of LNS, we compare it against this state-of-the-art initialization by training OLMo-1B on 20B tokens. As shown in Table 3, LNS achieves consistently lower training loss, indicating that it may offer a more effective alternative for large-scale LLM training.

Table 3: Comparison with OLMo’s Scaled Initialization.

Method Model # Tokens Training Loss Perplexity

OLMo’s Scaled Initialization OLMo-1B 20B 2.96 19.30 LayerNorm Scaling OLMo-1B 20B 2.85 17.28

##### 5.3.2 Qwen2.5

We further evaluate the generalizability of LNS by applying it to a state-of-the-art architecture, Qwen2.5-0.5B (Yang et al., 2024). We train the model for 6B tokens and compare LNS against the standard Pre-LN setup. Consistent with previous findings, Table 4 illustrates that LNS yields a notable reduction in perplexity—from 20.62 to 19.57—highlighting its effectiveness even on strong, modern architectures.

Table 4: Perplexity (PPL ↓) comparison under scaled-up pre-training. For LLaMA-1B and 7B, training is scheduled for 100B tokens but is terminated early to report results. Qwen-2.5 is trained with a fixed budget of 6B tokens.

Model # Params # Tokens Pre-LN (PPL) LNS (PPL) Qwen2.5-0.5B 0.5B 6B 20.62 19.57

The consistent benefits observed across increased model scales, larger training datasets, and diverse architectures suggest that LNS is a promising technique for enhancing the training of contemporary large language models, ensuring that deeper layers contribute more effectively to learning.

#### 5.4 LNS Effectively Scales Down Output Variance

As LNS is proposed to reduce output variance, we empirically validate this claim during the pre-training of LLMs. We compare the layerwise output variance of three configurations: (1) the standard Pre-LN (Ba, 2016),

- (2) Pre-LN with Scaled Initialization (Radford et al., 2019; Shoeybi et al., 2020), which scales the initialization of the feedforward layers’ weights W0 and W2 by √12L, where L is the total number of Transformer layers, and

- (3) Pre-LN with LNS. The average output variance across layers is shown in Figure 4. For both vanilla Pre-LN and Scaled Initialization, the output variance in shallow layers (blue) remains relatively stable throughout training, while variance in deeper layers (red) grows substantially after 2K iterations, reaching up to 175 in the final layer. Since Scaled Initialization only operates at initialization, it is insufficient to constrain output variance during training. In contrast, LNS consistently suppresses the growth of output variance in deeper layers, capping it at approximately 25.

#### 5.5 LNS Enhances the Effectiveness of Deep Layers

1.0

1.0

[Figure 12]

[Figure 13]

(b) LLaMA-130M with LayerNorm Scaling

(a) LLaMA-130M with Pre-LN

(a) Pre-LN Angular Distance

(b) LNS Angular Distance

0.0

0.0

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

0.8

0.8

7

7

2.5

2.5

SubsequentLayerhn

SubsequentLayerhn

5.0

5.0

()ARCP

()ARCP

0.6

0.6

5

5

7.5

7.5

3

3

10.0

10.0

0.4

0.4

Poriginal P( )

Poriginal P( )

12.5

12.5

| |
|---|

| |
|---|

0.2

0.2

15.0

15.0

0 2 4 6 8 10 Layer Index

0 2 4 6 8 10 Layer Index

0 3 6 9

0 3 6 9

Layer Index

Layer Index

0.0

0.0

- Figure 7: Left: Performance drop of layer pruning on LLaMA-130M. Right: The angular distance between representations of subsequent layers is shown. LayerNorm Scaling enables deep layers to make a meaningful contribution to the model.

Furthermore, to assess whether LNS enhances the effectiveness of deeper layers by promoting more diverse feature representations, we analyze the layerwise performance drop and the angular distance of LNS, as shown in Figure 7. Compared to Pre-LN, the performance degradation in LayerNorm Scaling is more uniformly distributed across layers, indicating a more balanced contribution from each layer. Notably, pruning the deeper layers of LNS results in a more significant accuracy drop, suggesting these layers play a more critical role in task performance. Additionally, features learned under LNS exhibit greater distinction: most layers show a substantial angular distance, exceeding 0.6, from their adjacent layers, indicating more diverse representations. In sharp contrast, the layerwise angular distance in Pre-LN remains significantly lower and progressively decreases with depth, suggesting reduced feature diversity.

#### 5.6 LayerNorm Scaling in Vision Transformer

To evaluate whether LNS also benefits architectures beyond language models, we conduct experiments on ViTS on ImageNet-1K. Since ViT-S includes LayerScale (Touvron et al., 2021) by default—which may interfere with the effect of LNS—we remove LayerScale from all evaluated variants to ensure a fair comparison. We then test different insertion positions of LNS. The top-1 accuracy results are summarized in Table 5. Whereas LNS in language models is typically most effective directly after normalization, in Vision Transformers, the best position is after the attention and MLP blocks. We next examine whether this performance gain correlates with better control of layer-wise variance.

Figure 8 plots the average output variance of each transformer block during training. Without LayerScale, variance in deeper layers grows rapidly—exceeding ∼ 3,000 by 30K update steps. Applying LNS after Attn/MLP controls this growth to below ∼150, confirming that LNS stabilizes the forward signal even in vision transformers.

Table 5: Top-1 accuracy (%) of ViT-S model with and without LNS.

##### Model Variant LNS Position Top-1 Accuracy

###### ViT (w/o LayerScale) – 67.91 ViT (w/o LayerScale) after LayerNorm 66.43 ViT (w/o LayerScale) after Attn/MLP 68.75

###### vit_w/o_LayerScale

###### vit_w/o_LayerScale_ w_LNS_after_mlp

140

3000

36

|[Figure 14]| |
|---|---|
| | |
| | |
| | |
| | |
| | |

120

2500

30

AverageVariance(Var)

AverageVariance(Var)

100

2000

24

LayerIndex

80

18

1500

60

12

1000

40

6

500

20

1

0

0

0 5000 10000 15000 20000 25000 30000

0 5000 10000 15000 20000 25000 30000

Update Step

Update Step

- Figure 8: Layer-wise output variance of ViT-S without LayerScale (left) and with LNS after Attn/MLP (right). LNS significantly reduces the variance growth compared to the baseline.

These preliminary findings indicate that the variance-control mechanism underlying LNS generalizes to vision transformers when the scaling is applied after Attn/MLP. We leave a more detailed theoretical understanding of this behavior to future work and community discussion.

### 6 Ablation Study

Comparing Against Other Scaling Methods. We first compare LNS with previous scaling approaches, including (1) Scaled Initialization (Radford et al., 2019; Shoeybi et al., 2020), which scales the initialization of W0 and W2 by the overall depth 1/

√

2L; (2) Depth-Scaled Initialization (Zhang et al., 2019) scales the initialization of weight matrices by the current depth 1/

√

2l; (3) SkipInit (De and Smith, 2020) introduces a learnable parameter after FFN/Att layers, initialized as 1/

√

L; (4) LayerScale (Touvron et al., 2021) applies per-channel weighting using a diagonal matrix, diag(λ1,...,λd), where each weight λi is initialized to a small value (e.g., λi = ϵ). Table 6 presents the results of LLaMA-130M and LLaMA-250M.

First, we observe that methods involving learnable parameters, such as LayerScale and SkipInit, consistently degrade performance in LLMs. Among initialization-based techniques, a larger scaling factor proves beneficial: Scaled Initialization yields lower perplexity compared to Depth-Scaled Initialization. Notably, LNS achieves the best overall performance, underscoring the advantage of applying scaling dynamically during training. Interestingly, combining LNS with Scaled Initialization results in worse performance than using LNS alone, highlighting the importance of removing conflicting initialization strategies prior to adopting LNS.

Comparison with Other Layer Normalization. In addition, we conducted comparisons using LLaMA-130M to evaluate LayerNorm Scaling against recently proposed normalization methods, including Admin (Liu et al., 2020), Sandwich-LN (Ding et al., 2021), Group-LN (Ma et al., 2024; Wu and He, 2018), and Mix-LN (Li et al., 2024b). Table 7 shows that Admin and Group-LN degrade performance. Sandwich-LN slightly outperforms Pre-LN. Both Mix-LN and LayerNorm Scaling improve over Pre-LN by good margins. However, Mix-LN fails to reduce perplexity under 26, falling short of LayerNorm Scaling and suffers from

Table 6: Comparing LNS against other scaling methods. Perplexity (↓) is reported.

LLaMA-130M LLaMA-250M

Training Tokens 2.2B 3.9B Pre-LN 26.73 21.92 + LayerScale (Touvron et al., 2021) 27.93 23.45 + SkipInit (De and Smith, 2020) 27.41 22.29 + Depth-Scaled Initialization (Zhang et al., 2019) 26.95 21.50 + Scaled Initialization (Shoeybi et al., 2020) 26.04 20.98 + LayerNorm Scaling 25.76 20.35 + LayerNorm Scaling + Scaled Initialization 25.80 20.79

instability in large-scale scenarios as shown in Table 1. Table 7: Comparison against other normalization methods on LLaMA-130M. All methods use the identical configurations. Perplexity (↓) is reported.

Pre-LN Admin Group-LN Sandwich-LN Mix-LN LayerNorm Scaling 26.73 27.91 28.01 26.51 26.07 25.76

Effect of Positions of LNS. The results in Table 8 show that inserting the scaling factor at different points can have a considerable influence on the model’s performance. Placing it after the residual connection (“After Residual”) leads to a perplexity of 1358.11, which indicates training divergence. In contrast, LNS incorporates the scaling factor after LN achieving the best perplexity of 25.76, surpassing both the baseline Pre-LN setting (26.73) and other placements. This suggests that modifying the LayerNorm to include the scaling factor can enhance training stability and final performance for this model configuration.

Table 8: Effects of Insertion Position of LayerNorm Scaling on LLaMA-130M

Pre-LN Before LN After Attn/FFN After Residual LNS Only After Attn LNS Only After FFN Ours (After LN) 26.73 26.97 26.53 1358.11 26.89 26.43 25.76

### 7 Related Work

Ineffectiveness of Deeper Layers in Transformers. The ineffectiveness of deep layers in LLMs has been previously reported. Yin et al. (2024) found that deeper layers of LLMs can tolerate significantly higher levels of pruning compared to shallower layers, achieving high sparsity. Similarly, Gromov et al. (2024) and Men et al. (2024) demonstrated that removing early layers causes a dramatic decline in model performance, whereas removing deep layers does not. Lad et al. (2024) showed that the middle and deep layers of GPT-2 and Pythia exhibit remarkable robustness to perturbations such as layer swapping and layer dropping. Recently, Li et al. (2024a) highlighted that early layers contain more outliers and are therefore more critical for fine-tuning. While these studies effectively highlight the limitations of deep layers in LLMs, they stop short of identifying the root cause of this issue or proposing viable solutions to address it.

Layer Normalization in Language Models. LN (Ba, 2016) was initially applied after the residual connection in the original Transformer (Vaswani, 2017), which is known as Post-LN. Later on, Pre-LN (Baevski and Auli, 2019; Dai et al., 2019; Nguyen and Salazar, 2019) dominated LLMs, due to its compelling performance and stability (Bi et al., 2024; Brown et al., 2020; Jiang et al., 2023; Touvron et al., 2023). Prior works have studied the effect of Pre-LN and Post-LN. Xiong et al. (2020) proves that Post-LN tends to have larger gradients near the output layer, which necessitates smaller learning rates to stabilize training, whereas

Pre-LN scales down gradients with the depth of the model, working better for deep Transformers. Wang et al. (2019) empirically confirmed that Pre-LN facilitates stacking more layers and Post-LN suffers from gradient vanishing. The idea of connecting multiple layers was proposed in previous works (Bapna et al., 2018; Dou et al., 2018; Wang et al., 2019). Admin introduces additional parameters to control residual dependencies, stabilizing Post-LN. DeepNorm (Wang et al., 2024) enables stacking 1000-layer Transformers by upscaling the residual connection before applying LN. Additionally, Ding et al. (2021) proposed Sandwich LayerNorm, normalizing both the input and output of each transformer sub-layer. Takase et al. (2023a) introduced B2T to bypass all LN except the final one in each layer. Li et al. (2024b) recently combines Post-LN and Pre-LN to enhance the middle layers. Zhu et al. (2025b) introduces Dynamic Tanh (DyT) as a normalization-free alternative in Transformers, delivering comparable performance. Zhuo et al. (2025) proposes HybridNorm, a hybrid normalization scheme combining QKV normalization with Post-Norm FFN to stabilize training in deep transformers. De and Smith (2020) also states that normalized residual blocks in deep networks are close to the identity function and proposes SkipInit to remove normalization by introducing a learnable scalar multiplier on the residual branch initialized to 1/

√

L. Our experiments suggest that SkipInit’s learnable parameter does not improve performance and sometimes harms training.

### 8 Conclusion

In this paper, we re-introduce the concept of the Curse of Depth in LLMs, highlighting an urgent yet often overlooked phenomenon: nearly half of the deep layers in modern LLMs are less effective than expected. We discover the root cause of this phenomenon is Pre-LN which is widely used in almost all modern LLMs. To tackle this issue, we introduce LayerNorm Scaling. By scaling the output variance inversely with the layer depth, LayerNorm Scaling ensures that all layers, including deeper ones, contribute meaningfully to training. Our experiments show that this simple modification improves performance, reduces resource usage, and stabilizes training across various model sizes. LayerNorm Scaling is easy to implement, hyperparameter-free, and provides a robust solution to enhance the efficiency and effectiveness of LLMs.

### References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Jimmy Lei Ba. Layer normalization. arXiv preprint arXiv:1607.06450, 2016. Alexei Baevski and Michael Auli. Adaptive input representations for neural language modeling. ICLR, 2019. Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie

Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. Ankur Bapna, Mia Xu Chen, Orhan Firat, Yuan Cao, and Yonghui Wu. Training deeper neural machine translation models with transparent attention. EMNLP, 2018.

Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, et al. Deepseek llm: Scaling open-source language models with longtermism. arXiv preprint arXiv:2401.02954, 2024.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. NeurIPS, 2020.

Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V Le, and Ruslan Salakhutdinov. Transformerxl: Attentive language models beyond a fixed-length context. ACL, 2019.

Soham De and Samuel L. Smith. Batch normalization biases residual blocks towards the identity function in deep networks, 2020. URL https://arxiv.org/abs/2002.10444.

Jacob Devlin. Bert: Pre-training of deep bidirectional transformers for language understanding. NAACL, 2019.

Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, et al. Cogview: Mastering text-to-image generation via transformers. NeurIPS, 34: 19822–19835, 2021.

Zi-Yi Dou, Zhaopeng Tu, Xing Wang, Shuming Shi, and Tong Zhang. Exploiting deep representations for neural machine translation. EMNLP, 2018.

Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. Glm: General language model pretraining with autoregressive blank infilling. arXiv preprint arXiv:2103.10360, 2021.

Razvan-Gabriel Dumitru, Vikas Yadav, Rishabh Maheshwary, Paul-Ioan Clotan, Sathwik Tejaswi Madhusudhan, and Mihai Surdeanu. Layer-wise quantization: A pragmatic and effective method for quantizing llms beyond integer bit-levels. arXiv preprint arXiv:2406.17415, 2024.

Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, et al. Olmo: Accelerating the science of language models. arXiv preprint arXiv:2402.00838, 2024.

Andrey Gromov, Kushal Tirumala, Hassan Shapourian, Paolo Glorioso, and Daniel A Roberts. The unreasonable ineffectiveness of the deeper layers. arXiv preprint arXiv:2403.17887, 2024.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. ICLR, 2021.

Zhiqiang Hu, Lei Wang, Yihuai Lan, Wanyu Xu, Ee-Peng Lim, Lidong Bing, Xing Xu, Soujanya Poria, and Roy Ka-Wei Lee. Llm-adapters: An adapter family for parameter-efficient fine-tuning of large language models. EMNLP, 2023.

Tianjin Huang, Haotian Hu, Zhenyu Zhang, Gaojie Jin, Xiang Li, Li Shen, Tianlong Chen, Lu Liu, Qingsong Wen, Zhangyang Wang, et al. Stable-spam: How to train in 4-bit more stably than 16-bit adam. arXiv preprint arXiv:2502.17055, 2025a.

Tianjin Huang, Ziquan Zhu, Gaojie Jin, Lu Liu, Zhangyang Wang, and Shiwei Liu. Spam: Spike-aware adam with momentum reset for stable llm training. arXiv preprint arXiv:2501.06842, 2025b.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

Diederik P Kingma. Adam: A method for stochastic optimization. ICLR, 2015. Vedang Lad, Wes Gurnee, and Max Tegmark. The remarkable robustness of llms: Stages of inference? arXiv

preprint arXiv:2406.19384, 2024. Michel Ledoux. The Concentration of Measure Phenomenon, volume 89 of Mathematical Surveys and Monographs. American Mathematical Society, Providence, RI, 2001. Pengxiang Li, Lu Yin, Xiaowei Gao, and Shiwei Liu. Owlore: Outlier-weighed layerwise sampled low-rank projection for memory-efficient llm fine-tuning. arXiv preprint arXiv:2405.18380, 2024a. Pengxiang Li, Lu Yin, and Shiwei Liu. Mix-ln: Unleashing the power of deeper layers by combining pre-ln and post-ln. arXiv preprint arXiv:2412.13795, 2024b.

Vladislav Lialin, Sherin Muckatira, Namrata Shivagunde, and Anna Rumshisky. Relora: High-rank training through low-rank updates. In ICLR, 2023.

Liyuan Liu, Xiaodong Liu, Jianfeng Gao, Weizhu Chen, and Jiawei Han. Understanding the difficulty of training transformers. EMNLP, 2020.

Haiquan Lu, Yefan Zhou, Shiwei Liu, Zhangyang Wang, Michael W Mahoney, and Yaoqing Yang. Alphapruning: Using heavy-tailed self regularization theory for improved layer-wise pruning of large language models. NeurIPS, 2024.

Xuezhe Ma, Xiaomeng Yang, Wenhan Xiong, Beidi Chen, Lili Yu, Hao Zhang, Jonathan May, Luke Zettlemoyer, Omer Levy, and Chunting Zhou. Megalodon: Efficient llm pretraining and inference with unlimited context length. NeurIPS, 2024.

Sachin Mehta, Mohammad Hossein Sekhavat, Qingqing Cao, Maxwell Horton, Yanzi Jin, Chenfan Sun, Iman Mirzadeh, Mahyar Najibi, Dmitry Belenko, Peter Zatloukal, et al. Openelm: An efficient language model family with open training and inference framework. arXiv preprint arXiv:2404.14619, 2024.

Xin Men, Mingyu Xu, Qingyu Zhang, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han, and Weipeng Chen. Shortgpt: Layers in large language models are more redundant than you expect. arXiv preprint arXiv:2403.03853, 2024.

Saurav Muralidharan, Sharath Turuvekere Sreenivas, Raviraj Bhuminand Joshi, Marcin Chochowski, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Jan Kautz, and Pavlo Molchanov. Compact language models via pruning and knowledge distillation. In NeurIPS, 2024.

Toan Q Nguyen and Julian Salazar. Transformers without tears: Improving the normalization of self-attention. IWSLT, 2019.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models

are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019. P Rajpurkar. Squad: 100,000+ questions for machine comprehension of text. EMNLP, 2016. Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020. Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. ICML, 2020. Shoaib Ahmed Siddiqui, Xin Dong, Greg Heinrich, Thomas Breuel, Jan Kautz, David Krueger, and Pavlo

Molchanov. A deeper look at depth pruning of llms. ICML, 2024. Sho Takase, Shun Kiyono, Sosuke Kobayashi, and Jun Suzuki. B2t connection: Serving stability and performance in deep transformers. ACL, 2023a. Sho Takase, Shun Kiyono, Sosuke Kobayashi, and Jun Suzuki. Spike no more: Stabilizing the pre-training of large language models. arXiv preprint arXiv:2312.16903, 2023b. Qwen Team. Qwen3: Think deeper, act faster, April 2025. URL https://qwenlm.github.io/blog/qwen3/. Accessed: 2025-05-11. Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles, Gabriel Synnaeve, and Herv´e J´egou. Going deeper with image transformers. In ICCV, pages 32–42, 2021.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozie`re, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

A Vaswani. Attention is all you need. NeurIPS, 2017. Roman Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science.

Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 2018. Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Dongdong Zhang, and Furu Wei. Deepnet: Scaling transformers to 1,000 layers. TPAMI, 2024. Qiang Wang, Bei Li, Tong Xiao, Jingbo Zhu, Changliang Li, Derek F Wong, and Lidia S Chao. Learning deep transformer models for machine translation. ACL, 2019. E. T. Whittaker and G. N. Watson. A Course of Modern Analysis. Cambridge Mathematical Library.

Cambridge University Press, 4 edition, 1996. Yuxin Wu and Kaiming He. Group normalization. In ECCV, pages 3–19, 2018. Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan,

Liwei Wang, and Tieyan Liu. On layer normalization in the transformer architecture. In ICML, pages 10524–10533. PMLR, 2020.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

Greg Yang, Dingli Yu, Chen Zhu, and Soufiane Hayou. Tensor programs vi: Feature learning in infinite-depth neural networks. arXiv preprint arXiv:2310.02244, 2023.

Lu Yin, You Wu, Zhenyu Zhang, Cheng-Yu Hsieh, Yaqing Wang, Yiling Jia, Mykola Pechenizkiy, Yi Liang, Zhangyang Wang, and Shiwei Liu. Outlier weighed layerwise sparsity (owl): A missing secret sauce for pruning llms to high sparsity. ICML, 2024.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024.

Biao Zhang and Rico Sennrich. Root mean square layer normalization. NeurIPS, 32, 2019. Biao Zhang, Ivan Titov, and Rico Sennrich. Improving deep transformer with depth-scaled initialization and

merged attention. arXiv preprint arXiv:1908.11365, 2019. Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, and Yuandong Tian. Galore: Memory-efficient llm training by gradient low-rank projection. ICML, 2024. Jiachen Zhu, Xinlei Chen, Kaiming He, Yann LeCun, and Zhuang Liu. Transformers without normalization. arXiv preprint arXiv:2503.10622, 2025a. Jiachen Zhu, Xinlei Chen, Kaiming He, Yann LeCun, and Zhuang Liu. Transformers without normalization, 2025b. URL https://arxiv.org/abs/2503.10622.

Zhijian Zhuo, Yutao Zeng, Ya Wang, Sijun Zhang, Jian Yang, Xiaoqing Li, Xun Zhou, and Jinwen Ma. Hybridnorm: Towards stable and efficient transformer training via hybrid normalization, 2025. URL https://arxiv.org/abs/2503.04598.

### A Proofs of the Theorems of curse of depth

- A.1 Proof of Lemma 3.2 Proof. Given Equation (2) from (Takase et al., 2023b), we have:

y = xℓ+1 = x′ℓ + FFN(LN(x′ℓ)), x′ℓ = xℓ + Attn(LN(xℓ)).

(20)

Based on our Assumption 3.1, let Var(Attn(LN(xℓ))) = σAttn2 . Then we can write: Var(x′ℓ) = Var(xℓ) + Var(Attn(LN(xℓ))) + Cov(Attn(LN(xℓ)),Var(xℓ))

= σx2

+ σAttn2 + ρ1 · σx

ℓ · σAttn,

ℓ

- where ρ1 is the correlation factor. Similarly, let Var(FFN(LN(x′ℓ))) = σFFN2 . Then we have:

σx2

ℓ+1

= σ(x′ℓ)2 + σFFN2 + ρ2 · σx′

ℓ

· σFFN, (22)

- where ρ2 is the correlation factor. Thus, the relationship between Var(xℓ+1) and Var(xℓ) becomes:

(21)

σx2

= σx2

+ σAttn2 + σFFN2 + ρ1 · σx

ℓ · σAttn + ρ2 · σx′

· σFFN. (23)

ℓ+1

ℓ

ℓ

- A.1.1 Variance of the Attention The scaled dot-product attention mechanism is defined as:

QKT √dk

V. (24) The softmax function outputs a probability distribution over the keys. Let the softmax output be

Attn(Q,K,V ) = softmax

T

A = softmax QK

√dk , where A is a matrix with each row summing to 1. The final attention output is obtained by multiplying the softmax output A with the value matrix V :

Attn(Q,K,V ) = AV. (25)

- Lemma A.1 ((Ledoux, 2001)). Let {Xi}Ni=1 be independent and identically distributed random variables with

Xi N

mean m and variance σ2 < ∞. Define the softmax weights pi = e

j=1 eXj , and let p = (p1,...,pN). Then, as N → ∞, with high probability, the softmax vector p concentrates around the uniform distribution on N elements. In particular,

N

2

1 n

E

pi −

lim

= 0, (26)

n→∞

i=1

which implies that the softmax output becomes asymptotically indistinguishable, in expectation, from the uniform distribution.

According to the above lemma, to simplify the analysis, we make the following additional assumptions: The softmax output A is approximately uniform, meaning each element of A is roughly 1/n, where n is the number of keys/values. Given this assumption, the variance of the attention is:

1 n

Var(Attn(Q,K,V )) ∼ Var(AV ) =

n

1 n · nσV2 · dhead = dheadσV2 = σW2 d. (27)

dheadVar(Vi) =

i=1

where W is the universal weight matrix defined as before.

##### A.1.2 Variance of the Feed-Forward Network

The feed-forward network (FFN) in transformers typically consists of two linear transformations with a ReLU activation in between. The FFN can be written as:

FFN(x) = W2 · ReLU(W1 · x + b1) + b2. (28)

where W1 and W2 are weight matrices, and b1 and b2 are bias vectors. Using the result obtained by Wang et al. (2024), we get:

= σW4 . (29) In conclusion:

σFFN2 ∼ σW2

· σW2

1

2

= σx2

+ σW2 + ρ2 · σx

σx2′

ℓ · σW

ℓ

ℓ

σW2 σx2

σW σx

= σx2

(1 +

)

+

ℓ

ℓ

ℓ

1 σx

= σx2

Θ(1 +

).

ℓ

ℓ

(30)

2 W

1 + σ

σx2ℓ + ρ2 · σ

σxℓ . into Equation (23) we get:

For simplicity, we set the numerator part to 1. Substitute σx′

= σx

W

ℓ

ℓ

σx2

= σx2

+ σW2 + σW4 d2 + ρ1 · σx

· σW2 d

ℓ · σW + ρ2 · σx′

ℓ+1

ℓ

ℓ

ρ22σW3 dσx

ρ2σW4 d2 2σx

= σx2

+ σW2 + σW4 d2 + ρ1 · σx

ℓ · σW2 d +

ℓ

ℓ · σW + ρ2 · σx

+

(31)

2

ℓ

ℓ

1 σx

= σx2

Θ(1 +

).

ℓ

ℓ

From the result we can generally infer that the variance accumulates layer by layer. The variance with regard to σx

:

1

ℓ−1

1 σx

σx2

= σx2

. (32)

Θ

1 +

1

ℓ

k

k=1

We can also obtain a similar result for σx2′

. We observe that for any σx2

ℓ

≥ 1, the sequence is increasing, meaning each term in the product is bounded. Consequently, the entire product is bounded above by:

k

ℓ−1

1 σx

σx2

≤ σx2

1 +

1

ℓ

1

k=1

Taking the natural logarithm of both sides:

= σx2

1

1 +

1 σx

1

ℓ−1

= expΘ(L). (33)

ℓ−1

ℓ−1

1 σx2

1 σx2

) = log σx2

log(σx2

1 +

=

log 1 +

1

ℓ

k

k

k=1

k=1

2

ℓ−1

1 σx2

- 1

- 2

1 σx2

+ log(σx2

≥

−

).

1

k

k

k=1

Exponentiating both sides to find the lower bound for σx2

, we obtain:

ℓ

σx2

ℓ

≥ σx2

1

exp

ℓ−1

k=1

1 σx2

- 1

- 2σx2

−

k

k

.

+ log(σx2

)

1

(34)

This provides a tighter lower bound for σx2

compared to the upper bound of Equation (33). Since we know the upper bound of variance grows exponentially, the lower bound must be sub-exponential. Therefore, for σx2

ℓ

= ℓ, we must have:

ℓ

σx2

ℓ

≥ σx2

1

exp

ℓ−1

k=1

- 1

- 2k

1 k −

√

= Θ(exp(

L)) ≥ Θ(L).

| |
|---|

Therefore, the increasing lower bound for σx2

must grow faster than a linear function. So, the increase of variance is sub-exponential. A large increase in such bound will lead to gradient spikes, which can connect to previous studies in Huang et al. (2025a,b).

ℓ

#### A.2 Proof of Theorem 3.3

In this proof, we will divide the argument into two parts: first, the calculation of the Lemma A.2, and second, the analysis of ∂y

∂x1.

ℓ

- Lemma A.2. For an L-layered Pre-LN Transformer, ∂y

∂x1 using Equations (2) and (3) is given by:

L

L−1

∂yL ∂x1

=

n=1

The upper bound for the norm of ∂y

∂x1 is:

L

∂yℓ ∂x′

ℓ

∂x′ℓ ∂xℓ

·

. (35)

L−1

σ2 σx′

∂yL ∂x1 2 ≤

√

1 +

d + √dFFN)2 × 1 + 2dh √s + 2 +

(

l=1

ℓ

σ2 σx

1 √s

σ2d dhead + 1 + dhead/d .

ℓ

(36)

Here, h denotes the number of heads, s is the sequence length, and d, dFFN, and dhead are the dimension of the embedding, FFN layer and multi-head attention layer, respectively. The standard deviation of WQ, WK, WV , and WFFN at layer ℓ is σ based on Assumption 3.1.

##### A.2.1 Proof of Lemma A.2

Proof. Our derivation follows results in (Takase et al., 2023b), specifically Equation (7), which provides an upper bound on the norm of ∂y

∂x1 as:

ℓ

∂yℓ ∂x1 2

=

L−1

∂x′ℓ ∂xℓ

∂yℓ ∂x′

ℓ

l=1

. (37)

2

Thus, we can estimate the upper bound of the gradient norm of ∂y

∂x1 by analyzing the spectral norms of the Jacobian matrices for the FFN layer and the self-attention layer, namely,

ℓ

FFN:

∂yℓ ∂x′

ℓ 2

Attention:

We now derive an upper bound of ∥∂y

∂x′ℓ ∥2 as follows:

ℓ

∂FFN(LN(x′ℓ)) ∂LN(x′

∂yℓ ∂x′

≤ 1 +

ℓ) 2

ℓ 2

∂x′ℓ ∂xℓ 2

. (38)

∂LN(x′ℓ) ∂x′

. (39)

ℓ 2

be the standard deviations of Wℓ1 and Wℓ2, respectively. From Assumption 3.1, the

Let σw1

and σw2

ℓ

ℓ

spectral norms of Wℓ1 and Wℓ2 are given by their standard deviations and dimensions (Vershynin, 2018), so wo have:

∥W1∥2 ∼ σ1 d + dFFN.

.

For simplicity, we assume that d, and dFFN are equal, thus, ∂FFN(LN(x′ℓ)) ∂LN(x′

√

= ∥Wℓ1Wℓ2∥2 ≤ σ1σ2(

d + dffn)2. (40)

ℓ) 2

Finally, we have the following bound:

∂yℓ ∂x′

σw1

σw2

√

ℓ

ℓ

≤ 1 +

d + √dFFN)2

σx′

(

ℓ 2

ℓ

σℓ2 σx′

√

= 1 +

. (41)

d + √dFFN)2

(

ℓ

′

Following a similar procedure for the FFN, we rewrite ∥∂x

∂x ∥2 in Equation (38) as:

∂x′ ∂x 2 ≤ 1 +

∂Attn(LN(x)) ∂LN(x) 2

∂LN(x) ∂x 2

. (42)

Let Z(·) = concat(head1(·),...,headh(·)) and JZ denote the Jacobian of the Z(·). We can now express the spectral norm of the Jacobian matrix of attntion as:

∂Attn(LN(xℓ)) ∂LN(xℓ) 2

∂Z(LN(xℓ)) ∂LN(xℓ) 2

= WℓOZ(LN(xℓ))

= ∥WℓOJℓZ∥2. (43)

From (Vershynin, 2018), we know that:

∥JℓZ∥2 ≤ h √s + 2 +

1 √s

σ3 d3dhead + σxℓ

√

d + dhead . (44)

Here h is the number of heads, s is the sequence length, and the standard deviation of WQ, WK, and WV is σ.

By combining the inequalities (41), (44) and (42), and assuming that all σ values are the same for simplicity. we obtain:

L−1

σ2 σx′

∂yL ∂x1 2 ≤

√

1 +

d + √dFFN)2 × 1 + 2dh √s + 2 +

(

l=1

ℓ

σ2 σx

1 √s

σ2d dhead + 1 + dhead/d .

ℓ

(45)

| |
|---|

##### A.2.2 Analysis of the Upper Bound

As discussed in (Takase et al., 2023b), σ should be sufficiently small, and the standard deviation, σx′

or σx

ℓ

ℓ

should satisfy the condition σ2 ≪ σx′

to maintain the lazy training scheme. Thus, we obtain the following bound for the product over ℓ from 1 to L:

ℓ

To find the bound for ∂y

with respect to ℓ, we simplify the given inequality by approximating σx

ℓ ∂x1

ℓ

2

and σx′

, and this layer does not significantly affect the overall performance of deep Transformer networks. Furthermore, based on Lemma 3.2, we assume that σx′

. Based on Equation (30), σx

is only one layer ahead of σx′

ℓ

ℓ

ℓ

= σx

.

ℓ

ℓ

###### Equation (A.2) can be expressed in a traditional product form (Whittaker and Watson, 1996) for σx

:

ℓ

where

L−1

∂yL ∂x1 2 ≤

l=1

1 σx

1 σx2

B , (46)

1 +

A +

ℓ

ℓ

σ2 (

+ 2dh √s + 2 +

1 √s

σ2 d dhead + 1 + dhead/d , (47) and

√

A =

d + √dFFN)2

B = 2dh √s + 2 +

1 √s

σ4d dhead, (48) where A and B are independent of σx

###### , and under our assumption, are treated as constants. From classical infinite series analysis, it is known that as σx

ℓ

grows at a faster rate, the upper bound of the product decreases. The proof is omitted here for brevity. For the upper bound on the convergence rate of σx2

ℓ

###### , we assume σx2

= exp(ℓ) without loss of generality. Under this condition, we can derive the following result:

ℓ

ℓ

Taking the natural logarithm of the product:

log

L−1

k=1

A ek

B e2k

1 +

+

L−1

A ek

log 1 +

=

k=1

B e2k

+

.

Using the Taylor series expansion for log(1 + x), and applying this to our sum, we get:

∞

A ek

log 1 +

+

k=1

B e2k

∞

=

k=1

A ek

B e2k −

- 1

- 2

+

A ek

B e2k

+

2

1 3

+

A ek

B e2k

+

3

− ··· .

By evaluating the sums for each order of terms, we find that the result is a constant. Carrying this out for each term, we obtain:

L−1

A ek

B e2k ∼

A e − 1

log

1 +

+

k=1

Thus, the product is approximately:

B e2 − 1 −

- 1

- 2

+

A2 e2 − 1

A · B e3 − 1

+ 2

B2 e4 − 1

+

.

A2 e2 − 1

B2 e4 − 1

A · B e3 − 1

A e − 1

∂yL ∂x1 2 ≤ exp

B e2 − 1 −

- 1

- 2

+

+ 2

+

= M, (49) where M is a constant.

For the lower bound on the convergence rate of σx2

###### , we assume σx2

###### = ℓ without loss of generality. Under this condition, we derive the following result. Taking the logarithm of the product, applying the Taylor series expansion for log(1 + x), and applying this to our sum:

ℓ

ℓ

∞

A k

log 1 +

k=1

B ek2

+

For the first-order terms:

∞

=

k=1

A k

- 1

- 2

B ek2 −

+

A k

B ek2

+

2

1 3

+

∞

k=1

A k

B ek2

+

∞

1 k

= A

k=1

∞

1 ek2

+ B

.

k=1

A k

B ek2

+

3

− ··· .

The series ∞k=1 k1 is the harmonic series, which diverges. However, we approximate it using the EulerMascheroni constant γ and the fact recognize that the harmonic series grows logarithmically:

∞

1 k ∼ log n + γ (for large n).

k=1

The other series such as ∞k=1 e1

2

k2 converge because ek

###### grows very rapidly. For higher-order terms, they converge to constant, involving the series ∞k=1 k12 converges to π

2

6 , so they contribute a constant. Exponentiating both sides, we get:

∞

k=1

B ek2 ∼ exp(A(log n + γ) + const).

A k

+

1 +

Thus, the growth rate of the upper bound for ∂y

is:

L ∂x1

2

∂yL ∂x1 2 ≤ Θ(L). (50)

- A.3 Proof of Lemma 4.1 Proof. After scaling, the equation becomes:

1 √

y = xℓ+1 = x′ℓ + FFN(

LN(x′ℓ)),

ℓ

1 √

x′ℓ = xℓ + Attn(

LN(xℓ)).

ℓ

Following the same analysis as before, we scale the Attention and FFN sub-layers, yielding:

(51)

1 nℓ · n · σV2 =

σAttn2 =

1 ℓ

σV2 =

σW2

σW2

σW2 ℓ

, σFFN2 ∼

ℓ ·

1

ℓ

σW4 ℓ2

2

=

. (52)

In conclusion:

σW √

1 √

σx2′

###### = σx2

###### + σW2 + ρ2 · σx

= σx2

ℓ ·

Θ(1 +

). (53) Similarly, we obtain:

ℓ

ℓ

ℓ

ℓσx

ℓ

ℓ

1 √

σx2

###### = σx2

). (54) From the result we can generally infer that the variance accumulates layer by layer. The variance with

Θ(1 +

ℓ+1

ℓ

ℓσx

ℓ

regard to σx

###### :

1

ℓ−1

1 √

###### = σx2

σx2

, (55)

Θ

1 +

1

ℓ

kσx

k=1

k

We can also obtain a similar result for σx2′

. Taking the natural logarithm of both sides:

ℓ

###### ) = log σx2

log(σx2

1

ℓ

ℓ−1

k=1

ℓ−1

≥

k=1

1 kσx2

k

1 +

- 1

- 2

−

1 kσx2

k

1 kσx2

k

ℓ−1

=

log 1 +

k=1

2

###### + log(σx2

).

1

1 kσx2

k

###### + log(σx2

)

1

(56)

To establish a lower bound for σx2

, we exponentiate both sides. Setting σx2

ℓ

ℓ

= ℓ, we must have:

ℓ−1

1 k −

- 1

- 2k

σx2

≥ σx2

= Θ(exp(log L)) ≥ Θ(L). (57)

exp

1

ℓ

k=1

Therefore, the increasing lower bound σx2

is greater than a linear function. Similarly, assuming σx2

ℓ

= ℓ(2−ϵ), we have:

ℓ

ℓ−1

ℓ−1

ℓϵ/2−1 − 1 ϵ/2 − 1 ≤ Θ(ℓ(2−ϵ)) ≤ Θ(ℓ2).

1 k2−ϵ/2 ∼ exp

1 k2−ϵ/2 ∼ exp

σx2

= σx2

1 +

1

ℓ

(58)

k=1

k=1

Here ϵ is a small constant with 1/2 ≤ ϵ < 1. Therefore, the increasing upper bound of σx2

is slower than the ℓ3 function, leading to:

ℓ

σx2

≤ Θ(L2)

ℓ

.

| |
|---|

#### A.4 Proof of Theorem 4.2

Proof. Similarly, after applying the scaling transformation, we derive an upper bound for ∥∂y

∂x′ℓ ∥2 as follows:

ℓ

∂FFN(LN(x′ℓ)) ∂LN(x′

∂LN(x′ℓ) ∂x′

∂yℓ ∂x′

1 √

≤ 1 +

ℓ) 2

ℓ 2

ℓ 2

ℓ 2

(59)

σℓ2 ℓσx′

√

= 1 +

.

d + √dFFN)2

(

ℓ

Similarly, rewriting Equation (38) after scaling, we have ∂x′ ∂x 2 ≤ 1 +

∂Attn(LN(x)) ∂LN(x) 2

1 √

ℓ 2

∂LN(x) ∂x 2

. (60)

By combining the bound (59), and inequality (60), and assuming all σ are equal for simplicity, we obtain:

L−1

σ2 ℓσx′

∂yL ∂x1 2 ≤

√

1 +

d + √dFFN)2 × 1 + 2dh √s + 2 +

(

(61)

l=1

ℓ

σ2 ℓσx

1 √s

σ2d dhead + 1 + dhead/d .

ℓ

. After scaling, it becomes:

Equation (61) is a traditional product form (Whittaker and Watson, 1996) for σx

ℓ

L−1

∂yL ∂x1 2 ≤

l=1

1 ℓ2σx2

1 ℓσx

A +

B , (62)

1 +

ℓ

ℓ

where A and B retain their forms from Equation (47) and Equation (48) and are treated as constants. Regarding the upper bound on the convergence rate of σx2

= ℓ(2−ϵ) without loss of generality. For large L, the product can be approximated using the properties of infinite products:

, we assume σx2

ℓ

ℓ

L−1

ℓ=1

A ℓ2−ϵ/2 +

B ℓ4−ϵ ∼ exp

1 +

L−1

ℓ=1

A ℓ2−ϵ/2 +

B ℓ4−ϵ . (63)

Then, by evaluating the sum in the exponent, we obtain:

L−1

ℓϵ/2−1 − 1 ϵ/2 − 1

A ℓ2−ϵ/2 +

B ℓ4−ϵ ∼ exp A ·

1 +

ℓ=1

Therefore, we establish the upper bound:

ℓϵ−3 − 1 ϵ − 3

+ B ·

. (64)

ℓϵ/2−1 − 1 ϵ/2 − 1

ℓϵ−3 − 1 ϵ − 3

∂yL ∂x1 2 ≤ Θ exp A ·

+ B ·

= ω(1), (65)

where ω(1) denotes a growth strictly greater than a constant as defined before.

| |
|---|

- A.5 Proof of theorem 4.3 Proof. We start with the Equation (3) and the chain-rule:

∂L ∂W1

∂L ∂yL

∂yL ∂x1

∂x1 ∂Attn(LN(x1))

∂Attn(LN(x1)) ∂Attn(x1)

∂Attn(x1) ∂W1

=

. (66)

where L is the loss function and ∂y∂L

L

∂yL ∂W1

∂yL ∂x1

=

only relates to the composition of loss function. So we only consider:

∂LN(x1) ∂Attn(LN(x1))

∂Attn(LN(x1)) ∂Attn(x1)

∂Attn(x1) ∂W1

∂x1 ∂LN(x1)

. (67)

∂yL ∂W1 2 ≤

∂yL ∂x1 2 ·

∂x1 ∂LN(x1) 2 ·

∂LN(x1) ∂Attn(LN(x1)) 2 ·

∂Attn(LN(x1)) ∂Attn(x1) 2 ·

∂Attn(x1) ∂W1 2

. (68)

We know that:

∂x1 ∂ LN(x1) 2

= σx

. (69)

1

The paper (Vershynin, 2018) tells us that: ∂ Attn(LN(x1)) ∂ LN(x1) 2

= ∥W1OJ1Z∥2. (70) Now We want to calculate by writing the multi-head attention (MHA) operation. Although various

formulations exist, one common definition of MHA is as follows. For a given input x we compute

Q = xWQ, K = xWK, V = xWV , (71) and then for each head (we assume head index i and dk the per-head dimension) headi = softmax QiK

⊤

√dki Vi. The outputs of all h heads are then concatenated and projected with WO: Attn(x) = head1,...,headh WO.

⊤

Q = xW1, and then through head = softmax QK

√dk V, so that (ignoring the outer projection WO) we have

(xW1)K⊤ √dk

Attn(x) ∼ softmax

V. (72) So we have:

∥x∥2 √dk

∂ Attn(x) ∂W1 2 ≤

, (73) That is, writing it out explicitly,

∥x∥2 √

∂yL ∂W1 2 ≤

∂yL ∂x1 2

1∥W1OJ1Z∥2

. (74) We know Equation (44):

σx

d

∥JℓZ∥2 ≤ h √s + 2 +

1 √s

σ3 d3dhead + σxℓ

√

d + dhead . (75)

Assume that x ∈ Rn is distributed as a multivariate normal with mean 0 and covariance σ12. Then ni=1 yi2 follows a χ2 distribution with n degrees of freedom. Thus,

√

E[∥x∥2] = σ12E χ2n = σ12

Γ n+12 Γ n2

2

. (76)

For large n, the chi-square distribution is concentrated around its mean (n) and one often approximates

∥x1∥2 ∼ σ12d. (77)

√

∥WO∥2 ≤ σ

d + hdhead . (78) Combine above together we have:

∂yL ∂W1 2 ≤

∂yL ∂x1 2

(σx

)2∥W1O∥2h √s + 2 +

1 √s

1

σ3 d3dhead + σx

1

√

d + dhead . (79)

This is the desired upper bound for ∂y

###### . Then we substitute the Equation (A.2) into the bound: and substituting the bounds for ∥W1O∥2 and

L ∂W1

2

∂yL ∂x1

into the original inequality yields

2

L−1

√

∂yL ∂W1 2 ≤

)2σ

Aℓ (σx

d + hdhead h

1

(80)

ℓ=1

√

√s + 2 +

1 √s

σ3 d3dhead + σx

×

d + dhead .

1

That is our final upper bound. For clarity, we summarize the answer:

L−1

σ2 σx′

1 + 2dh √s + 2 +

1 √s

∂yL ∂W1 2 ≤

√

1 +

d + √dFFN)2

(

ℓ=1

ℓ

σ2 σx

dhead d × (σx

σ2d dhead + 1 +

×

ℓ

√

)2σ

d + hdhead h ×

1

√

√s + 2 +

1 √s

σ3 d3dhead + σx

d + dhead .

1

So we find our that it is in the following form:

(81)

L−1

∂yL ∂x1 2 ≤

l=1

1 σx

1 σx2

A′ +

###### B′ , (82)

1 +

ℓ

ℓ

| |
|---|

### B Variance Growth in Pre-LN Training

To analyze the impact of Pre-LN on variance propagation, we track the variance of layer outputs across different depths during training.

Variance at 1000 Epochs Across Layers

Variance at 3000 Epochs Across Layers

Variance at 6000 Epochs Across Layers

- 1

- 2

- 3

- 4

- 5

175

80

150

60

125

Variance

Variance

Variance

100

40

75

50

20

25

0

0

0 2 4 6 8 10

0 2 4 6 8 10

0 2 4 6 8 10

Layer Index

Layer Index

Layer Index

- Figure 9: Variance growth across layers in LLaMA-130M with Pre-LN. Each subplot shows the variance at different training stages (1000, 3000, and 6000 epochs). In all cases, the variance follows an exponential growth pattern as depth increases, indicating that deeper layers experience uncontrolled variance amplification regardless of training progress.

Figure 9 illustrates the layer-wise variance in LLaMA-130M with Pre-LN at 1000, 3000, and 6000 epochs. Across all stages, variance remains low in shallow layers but grows exponentially in deeper layers, confirming that this issue persists throughout training rather than being a temporary effect. This highlights the necessity of stabilization techniques like LayerNorm Scaling to control variance and ensure effective deep-layer learning.

### C Performance Drop of Layer Pruning in Vision–Language Models (Qwen 2.5-VL)

To examine whether the Curse of Depth also manifests in vision–language models (VLMs), we perform layer–pruning experiments on Qwen 2.5-VL-7B (Bai et al., 2025). For both its vision encoder and language decoder, we prune one transformer layer at a time and directly evaluate the pruned model on the MMMU benchmark (Yue et al., 2024). Figure 10 presents the resulting performance drops.

We observe that the language branch clearly suffers from the Curse of Depth, whereas the vision branch remains uniformly important. This suggests that the phenomenon is more pronounced in autoregressive language components of VLMs and may not directly transfer to vision encoders. A detailed modality–specific theoretical account is left to future work and community discussion.

### D Limitations

While this work offers a comprehensive analysis of the Curse of Depth in LLMs and proposes LayerNorm Scaling as an effective remedy, several limitations remain:

Scope of Architectures. Our study primarily focuses on Transformer-based LLMs using Pre-LN. Although Pre-LN dominates modern architectures, our theoretical study does not cover models employing alternative normalization strategies (e.g., Post-LN only (Du et al., 2021), normalization-free architectures (Zhu et al., 2025a)) or emerging paradigms such as mixture-of-experts or structured sparsity-based models.

(a) Qwen2.5-VL-7B-V

(b) Qwen2.5-VL-7B-L

0

0

1

5

2

10

3

15

MMMUP

MMMUP

4

20

5

25

6

30

Poriginal P( )

Poriginal P( )

| |
|---|

| |
|---|

7

35

0 5 10 15 20 25 30

0 5 10 15 20 25

Layer Index

Layer Index

- Figure 10: Performance drop of layer pruning on Qwen 2.5-VL-7B. (a) Vision branch shows relatively uniform sensitivity across layers. (b) Language branch exhibits a clear Curse of Depth: deeper layers contribute much less than early ones.

Task Coverage. Most empirical evaluations, including pruning and angular distance analyses, were conducted using general-purpose benchmarks like MMLU. While these tasks reflect broad model capabilities, domain-specific or long-context reasoning tasks may reveal different dynamics in deep layer contributions, which we leave for future work.

Fine-grained Representation Quality. While LNS improves angular distance and performance sensitivity across layers, a deeper analysis of what types of information are represented or lost in deeper layers remains unexamined. For example, whether LNS helps preserve syntactic, semantic, or factual knowledge across depth is unclear.

