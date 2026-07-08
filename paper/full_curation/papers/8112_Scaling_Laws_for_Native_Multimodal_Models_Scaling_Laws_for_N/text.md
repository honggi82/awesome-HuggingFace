## Scaling Laws for Native Multimodal Models

Mustafa Shukor2 Enrico Fini1 Victor Guilherme Turrisi da Costa1 Matthieu Cord2

Joshua Susskind1 Alaaeldin El-Nouby1

1Apple 2Sorbonne University

# arXiv:2504.07951v4[cs.CV]9Aug2025

- 2
- 3
- 4

### Abstract

- 2
- 3
- 4

Early: L ∝ C−0.0492 Late: L ∝ C−0.0494

Early: L ∝ C−0.0492 Late: L ∝ C−0.0494

ValidationLoss

ValidationLoss

Building general-purpose models that can effectively perceive the world through multimodal signals has been a long-standing goal. Current approaches involve integrating separately pre-trained components, such as connecting vision encoders to LLMs and continuing multimodal training. While such approaches exhibit remarkable sample efficiency, it remains an open question whether such late-fusion architectures are inherently superior. In this work, we revisit the architectural design of native multimodal models (NMMs)-those trained from the ground up on all modalities—and conduct an extensive scaling laws study, spanning 457 trained models with different architectures and training mixtures. Our investigation reveals no inherent advantage to late-fusion architectures over early-fusion ones, which do not rely on image encoders or tokenizers. On the contrary, early-fusion exhibits stronger performance at lower parameter counts, is more efficient to train, and is easier to deploy. Motivated by the strong performance of the early-fusion architectures, we show that incorporating Mixture of Experts (MoEs) allows models to learn modality-specific weights, significantly benefiting performance.

MoE: L ∝ C−0.0474

MoE: L ∝ C−0.0474

1018 1020 1022 1024

FLOPs

·10−2

Early: ND ∝ C0.053 Late: ND ∝ C0.076

4

MoE: ND ∝ C−0.312

N/D

2

1018 1020 1022 1024

0

FLOPs

1018 1020 1022 1024

FLOPs

Figure 1. Scaling properties of Native Multimodal Models. Based on the scaling laws study in § 3.1, we observe: (1) early and late fusion models provide similar validation loss L when trained with the same compute budget C (FLOPs); (2) This performance is achieved via a different trade-off between parameters N and number of training tokens D, where early-fusion models requires fewer parameters. (3) Sparse early-fusion models achieve lower loss and require more training tokens for a given FLOP budget.

from fully leveraging cross-modality co-dependancies. An additional challenge is scaling such systems; each component (e.g., vision encoder, LLM) has its own set of hyperparameters, pre-training data mixtues, and scaling properties with respect to the amount of data and compute applied. A more flexible architecture might allow the model to dynamically allocate its capacity across modalities, simplifying scaling efforts.

### 1. Introduction

Multimodality provides a rich signal for perceiving and understanding the world. Advances in vision [23, 52, 55, 80] and language models [3, 19, 67] have enabled the development of powerful multimodal models that understand language, images, and audio. A common approach involves grafting separately pre-trained unimodal models, such as connecting a vision encoder to the input layer of an LLM [6, 9, 35, 43, 62, 64, 73, 78].

In this work, we focus on the scaling properties of native multimodal models trained from the ground up on multimodal data. We first investigate whether the commonly adopted late-fusion architectures hold an intrinsic advantage by comparing them to early-fusion models, which process raw multimodal inputs without relying on dedicated vision encoders. We conduct scaling experiments on early and late fusion architectures, deriving scaling laws to pre-

Although this seems like a convenient approach, it remains an open question whether such late-fusion strategies are inherently optimal for understanding multimodal signals. Moreover, with abundant multimodal data available, initializing from unimodal pre-training is potentially detrimental, as it may introduce biases that prevent the model

dict their performance and compute-optimal configurations. Our findings indicate that late fusion offers no inherent advantage when trained from scratch. Instead, early-fusion models are more efficient and are easier to scale. Furthermore, we observe that native multimodal models follow scaling laws similar to those of LLMs [26], albeit with slight variations in scaling coefficients across modalities and datasets. Our results suggest that model parameters and training tokens should be scaled roughly equally for optimal performance. Moreover, we find that different multimodal training mixtures exhibit similar overall trends, indicating that our findings are likely to generalize to a broader range of settings.

While our findings favor early fusion, multimodal data is inherently heterogeneous, suggesting that some degree of parameter specialization may still offer benefits. To investigate this, we explore leveraging Mixture of Experts (MoEs) [59], a technique that enables the model to dynamically allocate specialized parameters across modalities in a symmetric and parallel manner, in contrast to late-fusion models, which are asymmetric and process data sequentially. Training native multimodal models with MoEs results in significantly improved performance and therefore, faster convergence. Our scaling laws for MoEs suggest that scaling number of training tokens is more important than the number of active parameters. This unbalanced scaling is different from what is observed for dense models, due to the higher number of total parameters for sparse models. In addition, Our analysis reveals that experts tend to specialize in different modalities, with this specialization being particularly prominent in the early and last layers.

#### 1.1. Summary of our findings

Our findings can be summarized as follows:

Native Early and Late fusion perform on par: Early fusion models trained from scratch perform on par with their late-fusion counterparts, with a slight advantage to earlyfusion models for low compute budgets (Figure 3). Furthermore, our scaling laws study indicates that the computeoptimal models for early and late fusion perform similarly as the compute budget increases (Figure 1 Top).

NMMs scale similarly to LLMs: The scaling laws of native multimodal models follow similar laws as text-only LLMs with slightly varying scaling exponents depending on the target data type and training mixture (Table 2).

Late-fusion requires more parameters: Computeoptimal late-fusion models require a higher parameters-todata ratio when compared to early-fusion (Figure 1 bottom). Sparsity significantly benefits early-fusion NMMs: Sparse NMMs exhibit significant improvements compared to their dense counterparts at the same inference cost (Figure 10). Furthermore, they implicitly learn modalityspecific weights when trained with sparsity (Figure 12). In

Expression Definition

N Number of parameters in the multimodal decoder. For MoEs this

refers to the active parameters only. D Total number of multimodal tokens.

Nv Number of parameters in the vision-specific encoder. Only exists

in late-fusion architectures.

Dv Number of vision-only tokens. C Total number of FLOPs, estimated as C = 6ND for early-fusion

and C = 6(NvDv + ND) for late-fusion.

L Validation loss measured as the average over interleaved image-

text, image-caption, and text-only data mixtures.

Table 1. Definitions of the expressions used throughout the paper.

addition, compute-optimal models rely more on scaling the number of training tokens than the number of active parameters as the compute-budget grows (Figure 1 Bottom).

Modality-agnostic routing beats Modality-aware routing for Sparse NMMs: Training sparse mixture of experts with modality-agnostic routing consistently outperforms models with modality-aware routing (Figure 11).

### 2. Preliminaries

#### 2.1. Definitions

Native Multimodal Models (NMMs): Models that are trained from scratch on all modalities simultaneously without relying on pre-trained LLMs or vision encoders. Our focus is on the representative image and text modalities, where the model processes both text and images as input and generates text as output.

Early fusion: Enabling multimodal interaction from the beginning, using almost no modality-specific parameters (e.g., except a linear layer to patchify images). Using a single transformer model, this approach processes raw multimodal input—tokenized text and continuous image patches—with no image discretization. In this paper, we refer to the main transformer as the decoder.

Late fusion: Delaying the multimodal interaction to deeper layers, typically after separate unimodal components has processed that process each modality independently (e.g., a vision encoder connected to a decoder).

Modality-agnostic routing: In sparse mixture-of-experts, modality-agnostic routing refers to relying on a learned router module that is trained jointly with the model.

Modality-aware routing: Routing based on pre-defined rules such as routing based on the modality type (e.g., vision-tokens, token-tokens).

#### 2.2. Scaling Laws

We aim to understand the scaling properties of NMMs and how different architectural choices influence trade-offs. To this end, we analyze our models within the scaling laws framework proposed by Hoffmann et al. [26], Kaplan et al. [31]. We compute FLOPs based on the total number of parameters, using the approximation C = 6ND, as adopted in prior work [2, 26]. However, we modify this estimation to suit our setup: for late-fusion models, FLOPs is computed

FLOPs

FLOPs

L = E + NAα + DB

N ∝ Ca D ∝ Cb L ∝ Cc D ∝ Nd

β

|[Figure 1]|
|---|

|[Figure 2]|
|---|

1e+20 1e+22

1e+20 1e+22

L(N, D) N 0.2777 + D 0.34299

L(N, D) N 0.30051 + D 0.33491

Model Data E α β a b c d GPT3 [10] Text – – – – – -0.048 Chinchilla [26] Text 1.693 0.339 0.285 0.46 0.54 –

Validation Loss

Validation Loss

3.0

3.0

Text 2.222 0.3084 0.3375 0.5246 0.4774 -0.0420 0.9085 Image-Caption 1.569 0.3111 0.3386 0.5203 0.4785 -0.0610 0.9187

2.8

2.8

NMM (early-fusion)

Interleaved 1.966 0.2971 0.338 0.5315 0.4680 -0.0459 0.8791

2.6

2.6

AVG 1.904 0.301 0.335 0.5262 0.473 -0.0492 0.8987 NMM (late-fusion) AVG 1.891 0.2903 0.3383 0.6358 0.4619 -0.0494 0.6732 Sparse NMM (early-fusion) AVG 2.158 0.710 0.372 0.361 0.656 -0.047 1.797

2.4

2.4

0.3B 1B

0.3B 1B

200B

200B

500B

500B

3B

3B

Tokensseen

Tokensseen

Parameters

800B

Parameters

800B

Table 2. Scaling laws for native multimodal models. We report the scaling laws results for early and late fusion models. We fit the scaling laws for different target data types as well as their average loss (AVG).

- Figure 2. Scaling laws for early-fusion and late-fusion native multimodal models. Each point represents a model (300M to 3B parameters) trained on varying number of tokens (250M to 400B). We report the average cross-entropy loss on the validation sets of interleaved (Obelics), Image-caption (HQITP), and text-only data (DCLM).

quence packing for the image captioning dataset to reduce the amount of padded tokens. Similar to previous works [2, 5, 26], we evaluate performance on held-out subsets of interleaved (Obelics), Image-caption (HQITP), and text-only data (DCLM). Further implementation details are provided in Appendix A.

- as 6(NvDv + ND). We consider a setup where, given a compute budget C, our goal is to predict the model’s final performance, as well as determine the optimal number of parameters or number of training tokens. Consistent with prior studies on LLM scaling [26], we assume a power-law relationship between the final model loss and both model size (N) and training tokens (D):

### 3. Scaling native multimodal models

In this section, we present a scaling laws study of native multimodal models, examining various architectural choices § 3.1, exploring different data mixtures § 3.2, analyzing the practical trade-offs between late and early fusion NMMs, and comparing the performance of native pretraining and continual pre-training of NMMs § 3.3.

A Nα

B Dβ

. (1) Here, E represents the lowest achievable loss on the dataset, while NAα captures the effect of increasing the number of parameters, where a larger model leads to lower loss, with the rate of improvement governed by α. Similarly, DB

L = E +

+

Setup. We train models ranging from 0.3B to 4B active parameters, scaling the width while keeping the depth constant. For smaller training token budgets, we reduce the warm-up phase to 1K steps while maintaining 5K steps for larger budgets. Following H¨agele et al. [25], models are trained with a constant learning rate, followed by a cooldown phase using an inverse square root scheduler. The cool-down phase spans 20% of the total steps spent at the constant learning rate. To estimate the scaling coefficients in Eq 1, we apply the L-BFGS algorithm [51] and Huber loss [28] (with δ = 10−3), performing a grid search over initialization ranges.

accounts for the benefits of a higher number of tokens, with β determining the rate of improvement. Additionally, we assume a linear relationship between compute budget (FLOPs) and both N and D (C ∝ ND). This further leads to power-law relationships detailed in Appendix C.7.

β

#### 2.3. Experimental setup

Our models are based on the autoregressive transformer architecture [71] with SwiGLU FFNs [58] and QK-Norm [17] following Li et al. [39]. In early-fusion models, image patches are linearly projected to match the text token dimension, while late-fusion follows the CLIP architecture [55]. We adopt causal attention for text tokens and bidirectional attention for image tokens, we found this to work better. Training is conducted on a mixture of public and private multimodal datasets, including DCLM [39], Obelics [34], DFN [21], COYO [11], and a private collection of HighQuality Image-Text Pairs (HQITP). Images are resized to 224×224 resolution with a 14×14 patch size. We use a context length of 1k for the multimodal sequences. For training efficiency, we train our models with bfloat16, Fully Sharded Data Parallel (FSDP) [82], activation checkpointing, and gradient accumulation. We also use se-

#### 3.1. Scaling laws of NMMs

Scaling laws for early-fusion and late-fusion models. Figure 2 (left) presents the final loss averaged across interleaved, image-caption, and text datasets for early-fusion NMMs. The lowest-loss frontier follows a power law as a function of FLOPs. Fitting the power law yields the expression L ∝ C−0.049, indicating the rate of improvement with increasing compute. When analyzing the scaling laws per data type (e.g., image-caption, interleaved, text), we observe that the exponent varies (Table 2). For instance, the model achieves a higher rate of improvement for image-

Image-Caption

- 2.8
- 3

ValidationLoss

- 2.6

2.4

2.2

1020 1021 1022

FLOPs

Interleaved

- 2.8
- 3

2.6

1020 1021 1022

FLOPs

Text-only

3.4

3.2

- 2.8
- 3

1020 1021 1022

FLOPs

Late-289M Late-494M Late-1B Late-2.4B Early-275M Early-464M Early-932M Early-2.28B

- Figure 3. Early vs late fusion: scaling training FLOPs. We compare early and late fusion models when scaling both the number of model parameters and the number of training tokens. Overall, early fusion shows a slight advantage, especially at smaller model sizes, and the gap decreases when scaling the number of parameters N.

caption data (L ∝ C−0.061) when compared to interleaved documents (L ∝ C−0.046).

To model the loss as a function of the number of training tokens D and model parameters N, we fit the parametric function in Eq 1, obtaining scaling exponents α = 0.301 and β = 0.335. These describe the rates of improvement when scaling the number of model parameters and training tokens, respectively. Assuming a linear relationship between compute, N, and D (i.e., C ∝ ND), we derive the

- law relating model parameters to the compute budget (see Appendix C for details). Specifically, for a given compute budget C, we compute the corresponding model size N at

logarithmically spaced D values and determine Nopt, the parameter count that minimizes loss. Repeating this across different FLOPs values produces a dataset of (C,Nopt), to which we fit a power law predicting the compute-optimal model size as a function of compute: N∗ ∝ C0.526.

Similarly, we fit power laws to estimate the computeoptimal training dataset size as a function of compute and model size:

Dopt ∝ C0.473, Dopt ∝ N0.899. These relationships allow practitioners to determine the optimal model and dataset size given a fixed compute budget. When analyzing by data type, we find that interleaved data benefits more from larger models (a = 0.532) compared to image-caption data (a = 0.520), whereas the opposite trend holds for training tokens.

We conduct a similar study on late-fusion models in Figure 2 (right) and observe comparable scaling behaviors. In particular, the loss scaling exponent (c = −0.0494) is nearly identical to that of early fusion (c = −0.0492). This trend is evident in Figure 3, where early fusion outperforms late fusion at smaller model scales, while both architectures converge to similar performance at larger model sizes. We also observe similar trends when varying late-fusion con-

Relative Memory Usage

Relative Train time

|Late| | | |
|---|---|---|---|
| | | | |
| |Early| | |
| | | | |

0

GBperGPU

−5

−10

2 4 6

FLOPs (×1021)

|Late| | | |
|---|---|---|---|
| | | | |
| | | | |
| |Early| | |
| | | | |
| | | | |

0

Hours

−50

−100

−150

2 4 6

FLOPs (×1021)

Figure 4. Early vs late: pretraining efficiency. Early-fusion is faster to train and consumes less memory. Models are trained on 16 H100 GPUs for 160k steps (300B tokens).

figurations, such as using a smaller vision encoder with a larger text decoder Appendix B.

Scaling laws of NMMs vs LLMs. Upon comparing the scaling law coefficients of our NMMs to those reported for text-only LLMs (e.g., GPT-3, Chinchilla), we find them to be within similar ranges. In particular, for predicting the loss as a function of compute, GPT-3 [10] follows L ∝ C−0.048, while our models follow L ∝ C−0.049, suggesting that the performance of NMMs adheres to similar scaling laws as LLMs. Similarly, our estimates of the α and β parameters in Eq 1 (α = 0.301, β = 0.335) closely match those reported by Hoffmann et al. [26] (α = 0.339, β = 0.285). Likewise, our computed values of a = 0.526 and b = 0.473 align closely with a = 0.46 and b = 0.54 from [26], reinforcing the idea that, for native multimodal models, the number of training tokens and model parameters should be scaled proportionally. However, since the gap between a and b is smaller than in LLMs, this principle holds even more strongly for NMMs. Additionally, as a = 0.526 is greater than b = 0.473 in our case, the optimal model size for NMMs is larger than that of LLMs,

45-45-10

40-20-40

30-30-40

20-40-40

0492

0488

0486

0463

| | | |−0|.|
|---|---|---|---|---|
| |L|= 29.|002C| |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | |−0|.|
|---|---|---|---|---|
| |L|= 29.|574C| |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | |−0|.|
|---|---|---|---|---|
| |L|= 28.|590C| |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | |−0|.|
|---|---|---|---|---|
| |L|= 25.|623C| |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

- 3.5
- 4

- 3.5
- 4

- 3.5
- 4

- 3.5
- 4

ValidationLoss

- 2.5
- 3

- 2.5
- 3

- 2.5
- 3

- 2.5
- 3

1019 1020 1021 1022

1019 1020 1021 1022

1019 1020 1021 1022

1019 1020 1021 1022

FLOPs

FLOPs

FLOPs

FLOPs

0.275B 0.464B 0.932B 1.627B 2.280B 3.354B

Figure 5. Scaling laws with different training mixtures. Early-fusion models follow similar scaling trends when changing the pretraining mixtures. However, increasing the image captions leads to a higher scaling exponent norm (see Table 3).

C-I-T (%) I/T ratio E α β a b d c

- 1 45-45-10 1.19 1.906 0.301 0.335 0.527 0.474 0.901 -0.0492

- 2 40-20-40 0.65 1.965 0.328 0.348 0.518 0.486 0.937 -0.0486

- 3 30-30-40 0.59 1.847 0.253 0.338 0.572 0.428 0.748 -0.0463

- 4 20-40-40 0.49 1.836 0.259 0.354 0.582 0.423 0.726 -0.0488

Table 3. Scaling laws for different training mixtures. Earlyfusion models. C-I-T refer to image-caption, interleaved and text

while the optimal number of training tokens is lower, given a fixed compute budget.

Compute-optimal trade-offs for early vs. late fusion NMMs. While late- and early-fusion models reduce loss

- at similar rates with increasing FLOPs, we observe distinct trade-offs in their compute-optimal models. Specifically,

Nopt is larger for late-fusion models, whereas Dopt is larger for early-fusion models. This indicates that, given a fixed compute budget, late-fusion models require a higher number of parameters, while early-fusion models benefit more from a higher number of training tokens. This trend is also

reflected in the lower DNopt

∝ C0.053 for early fusion compared to DNopt

opt

∝ C0.076 for late fusion. As shown in Figure 1 (bottom), when scaling FLOPs, the number of parameters of early fusion models becomes significantly lower, which is crucial for reducing inference costs and, consequently, lowering serving costs after deployment.

opt

Early-fusion is more efficient to train. We compare the training efficiency of late- and early-fusion architectures. As shown in Figure 4, early-fusion models consume less memory and train faster under the same compute budget. This advantage becomes even more pronounced as compute increases, highlighting the superior training efficiency of early fusion while maintaining comparable performance to late fusion at scale. Notably, for the same FLOPs, latefusion models have a higher parameter count and higher effective depth (i.e., additional vision encoder layers alongside decoder layers) compared to early-fusion models.

Interleaved

Cross-entropy

Late

2.7

Early

2.65

2.6

40 60 80

% of Interleaved

Text-only

2.9

Late

Early

2.85

2.8

10 15 20 25 30

% of Text

Figure 7. Early vs late fusion: changing the training mixture. We vary the training mixtures and plot the final training loss. Early fusion models attain a favorable performance when increasing the proportion of interleaved documents and text-only data.

#### 3.2. Scaling laws for different data mixtures

We investigate how variations in the training mixture affect the scaling laws of native multimodal models. To this end, we study four different mixtures that reflect common community practices [34, 41, 46, 81], with Image CaptionInterleaved-Text ratios of 45-45-10 (our default setup),

30-30-40 , 40-20-40 , and 20-40-40 . For each mixture, we conduct a separate scaling study by training 76 different models, following our setup in § 3.1. Overall, Figure 5 shows that different mixtures follow similar scaling trends; however, the scaling coefficients vary depending on the mixture (Table 3). Interestingly, increasing the proportion of image-caption data (mixtures 1 and 2) leads to lower a and higher b, whereas increasing the ratio of interleaved and text data (mixtures 3 and 4) have the opposite effect. Notably, image-caption data contains more image tokens than text tokens; therefore, increasing its proportion results in more image tokens, while increasing interleaved and text data increases text token counts. This suggests that, when image tokens are prevalent, training for longer decreases the loss faster than increasing the model size. We also found that for a fixed model size, increasing text-only and interleaved data ratio is in favor of early-fusion Figure 7.

Image-Caption

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

2.5

ValidationLoss

2.7

2.4

2.6

2.3

2.5

200B 600B 1T 1.4T

Tokens seen

Interleaved

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

2.9

2.8

200B 600B 1T 1.4T

Tokens seen

Text-only

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

200B 600B 1T 1.4T

Tokens seen

Early Early + Init.

Figure 8. Early native vs initializing from LLMs: initializing from pre-trained models and scaling training tokens. We compare training with and without initializing from DCLM-1B.

#### 3.3. Native multimodal pre-training vs. continual training of LLMs

In this section, we compare training natively from scratch to continual training after initializing from a pre-trained LLM. We initialize the model from DCLM-1B [21] that is trained on more than 2T tokens. Figure 8 shows that native multimodal models can close the gap with initialized models when trained for longer. Specifically, on image captioning data, the model requires fewer than 100B multimodal tokens to reach comparable performance. However, on interleaved and text data, the model may need longer training—up to 1T tokens. Considering the cost of pre-training, these results suggest that training natively could be a more efficient approach for achieving the same performance on multimodal benchmarks.

### 4. Towards multimodal specialization

Previously, we demonstrated that early-fusion models achieve performance on par with late-fusion models under a fixed compute budget. However, multimodal data is inherently heterogeneous, and training a unified model to fit such diverse distributions may be suboptimal. Here, we argue for multimodal specialization within a unified architecture. Ideally, the model should implicitly adapt to each modality, for instance, by learning modality-specific weights or specialized experts. Mixture of Experts is a strong candidate for this approach, having demonstrated effectiveness in LLMs. In this section, we highlight the advantages of sparse earlyfusion models over their dense counterparts.

Setup. Our sparse models are based on the dropless-MoE implementation of Gale et al. [24], which eliminates token dropping during training caused by expert capacity constraints. We employ a top-k expert-choice routing mechanism, where each token selects its top-k experts among the E available experts. Specifically, we set k = 1 and E = 8, as we find this configuration to work effectively. Additionally, we incorporate an auxiliary load-balancing loss [59] with a weight of 0.01 to ensure a balanced expert utilization.

###### L = 26.287−0.047

- 3
- 4

ValidationLoss

0.275B 0.464B

- 0.932B

- 1.627B

- 2.280B

- 3.354B

1019 1020 1021 1022

FLOPs

Figure 9. Scaling laws for sparse early-fusion NMMs. We report the final validation loss averaged across interleaved, imagecaptions and text data.

Following Abnar et al. [2], we compute training FLOPs as 6ND, where N represents the number of active parameters.

#### 4.1. Sparse vs dense NMMs when scaling FLOPs

We compare sparse MoE models to their dense counterparts by training models with different numbers of active parameters and varying amounts of training tokens. Figure 10 shows that, under the same inference cost (or number of active parameters), MoEs significantly outperform dense models. Interestingly, this performance gap is more pronounced for smaller model sizes. This suggests that MoEs enable models to handle heterogeneous data more effectively and specialize in different modalities. However, as dense models become sufficiently large, the gap between the two architectures gradually closes.

#### 4.2. Scaling laws for sparse early-fusion models

We train different models (ranging from 300M to 3.4B active parameters) on varying amounts of tokens (ranging from 250M to 600B) and report the final loss in Figure 9. We fit a power law to the convex hull of the lowest loss as a function of compute (FLOPs). Interestingly, the exponent (−0.048) is close to that of dense NMMs (−0.049), indicating that both architectures scale similarly. However, the multiplicative constant is smaller for MoEs (27.086) compared to dense models (29.574), revealing lower loss. Additionally, MoEs require longer training to reach saturation compared to dense models (Appendix C for more details). We also predict the coefficients of Eq 1 by considering N as the number of active parameters. Table 2 shows significantly higher α compared to dense models. Interestingly, b is significantly higher than a, revealing that the training tokens should be scaled at a higher rate than the number of parameters when training sparse NMMs. We also experiment with a scaling law that takes into account the sparsity [2] and reached similar conclusions Appendix C.7.

#### 4.3. Modality-aware vs. Modality-agnostic routing

Another alternative to MoEs is modality-aware routing, where multimodal tokens are assigned to experts based on

Image-Caption

Interleaved

3

2.8

ValidationLoss

2.6

2.8

2.4

2.6

2.2

100B 200B 300B 400B

100B 200B 300B 400B

Tokens seen

Tokens seen

Dense-275M Dense-464M Dense-932M Dense-1.6B MoE-275M MoE-464M MoE-932M MoE-1.63B

Figure 10. MoE vs Dense: scaling training FLOPs. We compare MoE and dense early-fusion models when scaling both the amount of training tokens and model sizes. MoEs beat dense models when matching the number of active parameters.

their modalities, similar to previous works [7, 75]. We train models with distinct image and text experts in the form of FFNs, where image tokens are processed only by the image FFN and text tokens only by the text FFN. Compared to modality-aware routing, MoEs exhibit significantly better performance on both image-caption and interleaved data as presented in Figure 11.

#### 4.4. Emergence of expert specialization and sharing

We investigate multimodal specialization in MoE architectures. In Figure 13, we visualize the normalized number of text and image tokens assigned to each expert across layers. To quantify this specialization, we compute a specialization score, defined as the average, across all experts within a layer, of 1 − H(p), where H is the binary entropy of each expert’s text/image token distribution. We plot this specialization score in Figure 12. Higher specialization scores indicate a tendency for experts to focus on either text or image tokens, while lower scores indicate a shared behavior. These visualizations provide clear evidence of modalityspecific experts, particularly in the early layers. Furthermore, the specialization score decreases as the number of layers increases, before rising again in the last layers. This suggests that early and final layers exhibit higher modality specialization compared to mid-layers. This behavior is intuitive, as middle layers are expected to hold higherlevel features that may generalize across modalities, and consistent with findings in [61] that shows increasing alignment between modalities across layers. The emergence of both expert specialization and cross-modality sharing in our modality-agnostic MoE, suggests it may be a preferable approach compared to modality-aware sparsity. All data displayed here is from an early-fusion MoE model with 1B active parameters trained for 300B tokens.

Accuracy CIDEr AVG VQAv2 TextVQA OKVQA GQA VizWiz COCO TextCaps

Late-fusion 46.8 69.4 25.8 50.1 65.8 22.8 70.7 50.9 Early-fusion 47.6 69.3 28.1 52.1 65.4 23.2 72.0 53.8 Early-MoEs 48.2 69.8 30.0 52.1 65.4 23.6 69.6 55.7

Table 4. Supervised finetuning on the LLaVA mixture. All models are native at 1.5B scale and pre-trained on 300B tokens.

Image-Caption

Interleaved

2.8

- 2.8
- 3

ValidationLoss

2.6

2.4

2.6

2.2

100B 200B 300B 400B

100B 200B 300B 400B

Tokens seen

Tokens seen

Aware-275M Aware-464M Aware-932M Aware-1.63

Agnostic-275M Agnostic-464M Agnostic-932M Agnostic-1.63B

Figure 11. Modality-aware vs modality agnostic routing for sparse NMMs. We compare modality-agnostic routing with modality-aware routing when scaling both the amount of training tokens and model sizes.

### 5. Evaluation on downstream tasks with SFT

Following previous work on scaling laws, we primarily rely on validation losses. However, we generally find that this evaluation correlates well with performance on downstream tasks. To validate this, we conduct a multimodal instruction tuning stage (SFT) on the LLaVA mixture [43] and report accuracy and CIDEr scores across several VQA and captioning tasks. Table 4 confirms the ranking of different model configurations. Specifically, early fusion outperforms late fusion, and MoEs outperform dense models. However, since the models are relatively small (1.5B scale), trained from scratch, and fine-tuned on a small dataset, the overall scores are lower than the current state of the art. Further implementation details can be found in Appendix A.

### 6. Related work

Large multimodal models. A long-standing research goal has been to develop models capable of perceiving the world through multiple modalities, akin to human sensory experience. Recent progress in vision and language processing has shifted the research focus from smaller, taskspecific models toward large, generalist models that can handle diverse inputs [29, 67]. Crucially, pre-trained vision and language backbones often require surprisingly little adaptation to enable effective cross-modal communication [32, 47, 62, 68, 69]. Simply integrating a vision encoder with either an encoder-decoder architecture [45, 48, 63, 72]

I/Tspecialization

- 0.8
- 1

0.6

0.4

0.2

0

0 2 4 6 8 10 12 14 16 18 20 22

Layers

HQITP Obelics

Figure 12. MoE specialization score. Entropy-based image/text specialization score (as described in § 4.4) across layers for two data sources: HQITP and Obelics. HQITP has a more imbalanced image-to-text token distribution, resulting in generally higher specialization. Despite this difference, both data sources exhibit a similar trend: the specialization score decreases in the early layers before increasing again in the final layers.

or a decoder-only LLM has yielded highly capable multimodal systems [1, 6, 9, 13, 16, 35, 43, 49, 64, 73, 78, 83]. This late-fusion approach, where modalities are processed separately before being combined, is now well-understood, with established best practices for training effective models [34, 41, 46, 81]. In contrast, early-fusion models [8, 18, 66], which combine modalities at an earlier stage, remain relatively unexplored, with only a limited number of publicly released models [8, 18]. Unlike [18, 66], our models utilize only a single linear layer and rely exclusively on a nexttoken prediction loss. Furthermore, we train our models from scratch on all modalities without image tokenization.

Native Multimodal Models. We define native multimodal models as those trained from scratch on all modalities simultaneously [67] rather than adapting LLMs to accommodate additional modalities. Due to the high cost of training such models, they remain relatively underexplored, with most relying on late-fusion architectures [27, 79]. Some multimodal models trained from scratch [4, 66, 76] re-

- lax this constraint by utilizing pre-trained image tokenizers such as [20, 70] to convert images into discrete tokens, integrating them into the text vocabulary. This approach enables models to understand and generate text and images, facilitating a more seamless multimodal learning process.

Scaling laws. Scaling law studies aim to predict how model performance scales with training compute. Early works [26, 31] found that LLM performance follows a power-law relationship with compute, enabling the compute-optimal estimation of the number of model parameters and training tokens at scale for a given budget. Similar research has extended these findings to sparse Mixture of Experts (MoE) models, considering factors such as sparsity, number of experts, and routing granularity [15, 33, 74]. Scaling laws have also been observed across various domains, including image models [23], video models [56], protein LLMs [14], and imitation learning [54]. However, few stud-

Layer 0

Layer 16

Layer 23

100

100

100

| |Text<br><br>Image| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

| |Text<br><br>Image| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

| |Text<br><br>Image| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

%ofI/Ttokens

75

75

75

50

50

50

25

25

25

0

0

0

7 2 4 3 0 5 1 6 Experts

2 5 1 7 0 3 6 4 Experts

7 2 0 4 5 6 1 3 Experts

Figure 13. MoE specialization frequency. Percentage of text and image tokens routed to each expert on interleaved data from Obelics. Experts are ordered for better visualization. The first layer shows the highest amount of unimodal experts.

ies have investigated scaling laws for multimodal models. Notably, Aghajanyan et al. [5] examined multimodal models that tokenize modalities into discrete tokens and include multimodal generation. In contrast, we focus on studying early-fusion models that take raw multimodal inputs and are trained on interleaved multimodal data.

Mixture of experts (MoEs). MoEs [59] scale model capacity efficiently by sparsely activating parameters, enabling large models with reduced per-sample compute. While widely studied in LLMs [22, 30, 36, 37, 42, 65, 77, 84], MoEs remain underexplored in multimodal settings. Prior work has examined contrastive models [50], late-fusion LLMs [38, 40], and modality-specific experts [7, 12, 60]. We focus on analyzing MoEs in early-fusion multimodal models.

### 7. Limitations

Our study finds that scaling law coefficients are broadly consistent across training mixtures, though a broader exploration is needed to validate this observation. While validation loss scales predictably with compute, the extent to which this correlates with downstream performance remains unclear and warrants further investigation. The accuracy of scaling law predictions improves with higher FLOPs, but their extrapolation to extreme model sizes is still an open question (Appendix D for more details).

### 8. Conclusion

We explore various strategies for compute-optimal pretraining of native multimodal models. We found the NMMs follow similar scaling laws to those of LLMs. Contrary to common belief, we find no inherent advantage in adopting late-fusion architectures over early-fusion ones. While both architectures exhibit similar scaling properties, early-fusion models are more efficient to train and outperform latefusion models at lower compute budgets. Furthermore, we show that sparse architectures encourage modality-specific specialization, leading to performance improvements while maintaining the same inference cost.

### Acknowledgment

We thank Philipp Dufter, Samira Abnar, Xiujun Li, Zhe Gan, Alexander Toshev, Yinfei Yang, Dan Busbridge, and Jason Ramapuram for many fruitful discussions. We thank Denise Hui, and Samy Bengio for infra and compute support. Finally, we thank, Louis B´ethune, Pierre Ablin, Marco Cuturi, and the MLR team at Apple for their support throughout the project.

### References

- [1] Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219, 2024. 8
- [2] Samira Abnar, Harshay Shah, Dan Busbridge, Alaaeldin Mohamed Elnouby Ali, Josh Susskind, and Vimal Thilak. Parameters vs flops: Scaling laws for optimal sparsity for mixture-of-experts language models. arXiv preprint arXiv:2501.12370, 2025. 2, 3, 6, 18, 20
- [3] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 1

- [4] Armen Aghajanyan, Bernie Huang, Candace Ross, Vladimir Karpukhin, Hu Xu, Naman Goyal, Dmytro Okhonko, Mandar Joshi, Gargi Ghosh, Mike Lewis, et al. Cm3: A causal masked multimodal model of the internet. arXiv preprint arXiv:2201.07520, 2022. 8
- [5] Armen Aghajanyan, Lili Yu, Alexis Conneau, Wei-Ning Hsu, Karen Hambardzumyan, Susan Zhang, Stephen Roller, Naman Goyal, Omer Levy, and Luke Zettlemoyer. Scaling laws for generative mixed-modal language models. In International Conference on Machine Learning, pages 265–279. PMLR, 2023. 3, 8
- [6] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736,

2022. 1, 8

- [7] Hangbo Bao, Wenhui Wang, Li Dong, Qiang Liu, Owais Khan Mohammed, Kriti Aggarwal, Subhojit Som, and Furu Wei. Vlmo: Unified vision-language pretraining with mixture-of-modality-experts. arXiv preprint arXiv:2111.02358, 2021. 7, 8
- [8] Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, and Sa˘gnak Ta¸sırlar. Introducing our multimodal models, 2023. 8
- [9] Lucas Beyer, Andreas Steiner, Andr´e Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, et al. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726, 2024. 1, 8

- [10] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 3, 4
- [11] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo700m: Image-text pair dataset. https://github.com/ kakaobrain/coyo-dataset, 2022. 3, 13
- [12] Junyi Chen, Longteng Guo, Jia Sun, Shuai Shao, Zehuan Yuan, Liang Lin, and Dongyu Zhang. Eve: Efficient visionlanguage pre-training with masked prediction and modalityaware moe. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1110–1119, 2024. 8
- [13] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024. 8
- [14] Xingyi Cheng, Bo Chen, Pan Li, Jing Gong, Jie Tang, and Le Song. Training compute-optimal protein language models. bioRxiv, 2024. 8
- [15] Aidan Clark, Diego de Las Casas, Aurelia Guy, Arthur Mensch, Michela Paganini, Jordan Hoffmann, Bogdan Damoc, Blake Hechtman, Trevor Cai, Sebastian Borgeaud, et al. Unified scaling laws for routed language models. In International conference on machine learning, pages 4057–4086. PMLR, 2022. 8
- [16] Wenliang Dai, Nayeon Lee, Boxin Wang, Zhuolin Yang, Zihan Liu, Jon Barker, Tuomas Rintamaki, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Nvlm: Open frontier-class multimodal llms. arXiv preprint arXiv:2409.11402, 2024. 8
- [17] Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Peter Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, et al. Scaling vision transformers to 22 billion parameters. In International Conference on Machine Learning, pages 7480–7512. PMLR, 2023. 3
- [18] Haiwen Diao, Yufeng Cui, Xiaotong Li, Yueze Wang, Huchuan Lu, and Xinlong Wang. Unveiling encoder-free vision-language models. arXiv preprint arXiv:2406.11832,

2024. 8

- [19] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 1

- [20] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12873–12883, 2021. 8
- [21] Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander Toshev, and Vaishaal Shankar. Data filtering networks. arXiv preprint arXiv:2309.17425, 2023. 3, 6, 13

- [22] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1–39, 2022. 8
- [23] Enrico Fini, Mustafa Shukor, Xiujun Li, Philipp Dufter, Michal Klein, David Haldimann, Sai Aitharaju, Victor Guilherme Turrisi da Costa, Louis B´ethune, Zhe Gan, Alexander T Toshev, Marcin Eichner, Moin Nabi, Yinfei Yang, Joshua M. Susskind, and Alaaeldin El-Nouby. Multimodal autoregressive pre-training of large vision encoders, 2024. 1, 8
- [24] Trevor Gale, Deepak Narayanan, Cliff Young, and Matei Zaharia. Megablocks: Efficient sparse training with mixtureof-experts. Proceedings of Machine Learning and Systems, 5:288–304, 2023. 6
- [25] Alexander H¨agele, Elie Bakouch, Atli Kosson, Loubna Ben Allal, Leandro Von Werra, and Martin Jaggi. Scaling laws and compute-optimal training beyond fixed training durations. arXiv preprint arXiv:2405.18392, 2024. 3
- [26] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. In Proceedings of the 36th International Conference on Neural Information Processing Systems, pages 30016– 30030, 2022. 2, 3, 4, 8, 17
- [27] Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, et al. Language is not all you need: Aligning perception with language models. Advances in Neural Information Processing Systems, 36:72096–72109, 2023. 8
- [28] Peter J. Huber. Robust Estimation of a Location Parameter, pages 492–518. Springer New York, New York, NY, 1992. 3
- [29] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 7
- [30] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024. 8
- [31] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361,

2020. 2, 8, 15

- [32] Jing Yu Koh, Ruslan Salakhutdinov, and Daniel Fried. Grounding language models to images for multimodal inputs and outputs. In International Conference on Machine Learning, pages 17283–17300. PMLR, 2023. 7
- [33] Jakub Krajewski, Jan Ludziejewski, Kamil Adamczewski, Maciej Pi´oro, Michał Krutul, Szymon Antoniak, Kamil Ciebiera, Krystian Kr´ol, Tomasz Odrzyg´o´zd´z, Piotr Sankowski, et al. Scaling laws for fine-grained mixture of experts. arXiv preprint arXiv:2402.07871, 2024. 8, 18

- [34] Hugo Laurenc¸on, Lucile Saulnier, L´eo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander Rush, Douwe Kiela, et al. Obelics: An open web-scale filtered dataset of interleaved image-text documents. Advances in Neural Information Processing Systems, 36, 2024. 3, 5, 8, 13
- [35] Hugo Lauren¸con, L´eo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models? arXiv preprint arXiv:2405.02246, 2024. 1, 8
- [36] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. arXiv preprint arXiv:2006.16668, 2020. 8
- [37] Mike Lewis, Shruti Bhosale, Tim Dettmers, Naman Goyal, and Luke Zettlemoyer. Base layers: Simplifying training of large, sparse models. In International Conference on Machine Learning, pages 6265–6274. PMLR, 2021. 8
- [38] Dongxu Li, Yudong Liu, Haoning Wu, Yue Wang, Zhiqi Shen, Bowen Qu, Xinyao Niu, Guoyin Wang, Bei Chen, and Junnan Li. Aria: An open multimodal native mixture-ofexperts model. arXiv preprint arXiv:2410.05993, 2024. 8
- [39] Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Gadre, Hritik Bansal, Etash Guha, Sedrick Keh, Kushal Arora, et al. Datacomp-lm: In search of the next generation of training sets for language models. arXiv preprint arXiv:2406.11794, 2024. 3, 13, 15
- [40] Bin Lin, Zhenyu Tang, Yang Ye, Jiaxi Cui, Bin Zhu, Peng Jin, Junwu Zhang, Munan Ning, and Li Yuan. Moe-llava: Mixture of experts for large vision-language models. arXiv preprint arXiv:2401.15947, 2024. 8
- [41] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26689–26699, 2024. 5, 8
- [42] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024. 8
- [43] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024. 1, 7, 8
- [44] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 13
- [45] Jiasen Lu, Christopher Clark, Rowan Zellers, Roozbeh Mottaghi, and Aniruddha Kembhavi. Unified-io: A unified model for vision, language, and multi-modal tasks. In The Eleventh International Conference on Learning Representations, 2022. 7
- [46] Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, Sam Dodge, Bowen Zhang, Philipp Dufter, Dhruti Shah, Xianzhi Du, Futang Peng, Anton Belyi, et al. Mm1: methods, analysis and insights from multimodal llm pre-training. In European Conference on Computer Vision, pages 304–323. Springer, 2025. 5, 8, 13

- [47] Jack Merullo, Louis Castricato, Carsten Eickhoff, and Ellie Pavlick. Linearly mapping from image to text space. In The Eleventh International Conference on Learning Representations, 2023. 7
- [48] David Mizrahi, Roman Bachmann, Oguzhan Kar, Teresa Yeo, Mingfei Gao, Afshin Dehghan, and Amir Zamir. 4m: Massively multimodal masked modeling. Advances in Neural Information Processing Systems, 36:58363–58408, 2023. 7
- [49] Seungwhan Moon, Andrea Madotto, Zhaojiang Lin, Tushar Nagarajan, Matt Smith, Shashank Jain, Chun-Fu Yeh, Prakash Murugesan, Peyman Heidari, Yue Liu, et al. Anymal: An efficient and scalable any-modality augmented language model. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 1314–1332, 2024. 8
- [50] Basil Mustafa, Carlos Riquelme, Joan Puigcerver, Rodolphe Jenatton, and Neil Houlsby. Multimodal contrastive learning with limoe: the language-image mixture of experts. Advances in Neural Information Processing Systems, 35:9564– 9576, 2022. 8
- [51] Jorge Nocedal. Updating quasi newton matrices with limited storage. Mathematics of Computation, 35(151):951– 958, 1980. 3
- [52] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 1
- [53] Tim Pearce and Jinyeop Song. Reconciling kaplan and chinchilla scaling laws. arXiv preprint arXiv:2406.12907, 2024. 15
- [54] Tim Pearce, Tabish Rashid, Dave Bignell, Raluca Georgescu, Sam Devlin, and Katja Hofmann. Scaling laws for pre-training agents and world models. arXiv preprint arXiv:2411.04434, 2024. 8
- [55] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 1, 3, 15
- [56] Jathushan Rajasegaran, Ilija Radosavovic, Rahul Ravishankar, Yossi Gandelsman, Christoph Feichtenhofer, and Jitendra Malik. An empirical study of autoregressive pretraining from videos. arXiv preprint arXiv:2501.05453,

2025. 8

- [57] Kanchana Ranasinghe, Brandon McKinzie, Sachin Ravi, Yinfei Yang, Alexander Toshev, and Jonathon Shlens. Perceptual grouping in contrastive vision-language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5571–5584, 2023. 13
- [58] Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020. 3
- [59] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-

- of-experts layer. arXiv preprint arXiv:1701.06538, 2017. 2, 6, 8
- [60] Sheng Shen, Zhewei Yao, Chunyuan Li, Trevor Darrell, Kurt Keutzer, and Yuxiong He. Scaling vision-language models with sparse mixture of experts. In The 2023 Conference on Empirical Methods in Natural Language Processing, 2023. 8
- [61] Mustafa Shukor and Matthieu Cord. Implicit multimodal alignment: On the generalization of frozen llms to multimodal inputs. Advances in Neural Information Processing Systems, 37:130848–130886, 2024. 7
- [62] Mustafa Shukor, Corentin Dancette, and Matthieu Cord. epalm: Efficient perceptual augmentation of language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22056–22069, 2023. 1, 7
- [63] Mustafa Shukor, Corentin Dancette, Alexandre Rame, and Matthieu Cord. Unival: Unified model for image, video, audio and language tasks. Transactions on Machine Learning Research Journal, 2023. 7
- [64] Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Aractingi, Caroline Pascal, Martino Russi, Andres Marafioti, et al. Smolvla: A vision-language-action model for affordable and efficient robotics. arXiv preprint arXiv:2506.01844,

2025. 1, 8

- [65] Xingwu Sun, Yanfeng Chen, Yiqing Huang, Ruobing Xie, Jiaqi Zhu, Kai Zhang, Shuaipeng Li, Zhen Yang, Jonny Han, Xiaobo Shu, et al. Hunyuan-large: An open-source moe model with 52 billion activated parameters by tencent. arXiv preprint arXiv:2411.02265, 2024. 8
- [66] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. 8
- [67] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 1, 7, 8
- [68] Maria Tsimpoukelli, Jacob L Menick, Serkan Cabi, SM Eslami, Oriol Vinyals, and Felix Hill. Multimodal few-shot learning with frozen language models. Advances in Neural Information Processing Systems, 34:200–212, 2021. 7
- [69] Th´eophane Vallaeys, Mustafa Shukor, Matthieu Cord, and Jakob Verbeek. Improved baselines for data-efficient perceptual augmentation of llms. arXiv preprint arXiv:2403.13499,

2024. 7

- [70] Aaron van den Oord, Oriol Vinyals, and koray kavukcuoglu. Neural discrete representation learning. In Advances in Neural Information Processing Systems. Curran Associates, Inc.,

2017. 8

- [71] A Vaswani. Attention is all you need. Advances in Neural Information Processing Systems, 2017. 3
- [72] Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In International conference on machine learning, pages 23318–23340. PMLR, 2022. 7

- [73] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 1, 8
- [74] Siqi Wang, Zhengyu Chen, Bei Li, Keqing He, Min Zhang, and Jingang Wang. Scaling laws across model architectures: A comparative analysis of dense and MoE models in large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 5583–5595, Miami, Florida, USA, 2024. Association for Computational Linguistics. 8, 18
- [75] Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, et al. Image as a foreign language: Beit pretraining for all vision and visionlanguage tasks. arXiv preprint arXiv:2208.10442, 2022. 7
- [76] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 8
- [77] Tianwen Wei, Bo Zhu, Liang Zhao, Cheng Cheng, Biye Li, Weiwei L¨u, Peng Cheng, Jianhao Zhang, Xiaoyu Zhang, Liang Zeng, et al. Skywork-moe: A deep dive into training techniques for mixture-of-experts language models. arXiv preprint arXiv:2406.06563, 2024. 8
- [78] Le Xue, Manli Shu, Anas Awadalla, Jun Wang, An Yan, Senthil Purushwalkam, Honglu Zhou, Viraj Prabhu, Yutong Dai, Michael S Ryoo, et al. xgen-mm (blip-3): A family of open large multimodal models. arXiv preprint arXiv:2408.08872, 2024. 1, 8
- [79] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. Transactions on Machine Learning Research, 2022. 8
- [80] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023. 1
- [81] Haotian Zhang, Mingfei Gao, Zhe Gan, Philipp Dufter, Nina Wenzel, Forrest Huang, Dhruti Shah, Xianzhi Du, Bowen Zhang, Yanghao Li, et al. Mm1. 5: Methods, analysis & insights from multimodal llm fine-tuning. arXiv preprint arXiv:2409.20566, 2024. 5, 8
- [82] Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, ChienChin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al. Pytorch fsdp: experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277, 2023. 3
- [83] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. MiniGPT-4: Enhancing vision-language understanding with advanced large language models. In The Twelfth International Conference on Learning Representations, 2024. 8
- [84] Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, and William Fedus. Stmoe: Designing stable and transferable sparse expert models. arXiv preprint arXiv:2202.08906, 2022. 8

## Scaling Laws for Native Multimodal Models Supplementary Material

This supplementary material is organized as follows:

- • Appendix A: contains the implementation details and the hyperparameters used to train our models.
- • Appendix B: contains detailed comparison between early and late fusion models.
- • Appendix C: contains more details about scaling laws derivation, evaluation and additional results.
- • Appendix D: contains discussion about the paper limitations.
- • Appendix E: contains more results about MoEs and modality specialization.

### A. Experimental setup

In Table 6, we show the pre-training hyperparameters for different model configurations used to derive the scaling laws. The number of parameters ranges from 275M to 3.7B, with model width increasing accordingly, while the depth remains fixed at 24 layers. Learning rates vary by model size, decreasing as the model scales up. Based on empirical experiments and estimates similar to [46], we found these values to be effective in our setup. Training is optimized using a fully decoupled AdamW optimizer with momentum values β1 = 0.9, β2 = 0.95, and a weight decay of 1e−4. The batch size is set to 2k samples, which account for 2M tokens, given 1k context length. Gradient clipping is set to 1.0, with a maximum warmup duration of 5k iterations, adjusted for shorter training runs: 1k and 2.5k warmup steps for models trained between 1k–4k and 5k–15k steps, respectively. For MoEs, we found that longer warmup is significantly better, so we adopt a 2.5k warmup for all runs under 20k steps. We use a constant learning rate schedule with cooldown during the final 20% of training, gradually reducing to zero following an inverse square root schedule. For vision processing, image inputs are divided into (14,14) patches, with augmentations including Random Resized Crop (resizing images to 224px with a scale range of [0.4, 1.0]) and Random Horizontal Flip with a probability of 0.5. We train our models on mixture of interleaved, image captions and text only data Table 5. For late fusion models, we found that using smaller learning rate for the vision encoder significantly boost the performance Table 8, and when both the encoder and decoder are initialized (Appendix B.7) we found that freezing the vision encoder works best Table 7.

Data type dataset #samples sampling prob.

DFN [21] 2B 27% COYO [11] 600M 11.25% HQITP[57] 400M 6.75%

Image-Caption

Interleaved Obelics [34] 141M Docs 45% Text DCLM [39] 6.6T Toks 10%

- Table 5. Pre-training data mixture. Unless otherwise specified, the training mixture contains 45%, 45% and 10% of image captions, interleaved documents and text-only data.

Early-fusion

Params 275M 468M 932M 1.63B 2.28B 3.35B width 800 1088 1632 2208 2624 3232 depth 24 Learning rate 1.5e-3 1.5e-3 5e-4 4.2e-4 4e-4 3.5e-4

Late-fusion

Params 289M 494M 1B 1.75B 2.43B 3.7B vision encoder width 384 512 768 1024 1184 1536 vision encoder depth 24 width 768 1024 1536 2048 2464 3072 depth 24 Learning rate 1.5e-3 1.5e-3 5e-4 4.2e-4 3.8e-4 3.3e-4

Early-fusion MoEs

Active Params 275M 468M 932M 1.63B 2.28B 3.35B width 800 1088 1632 2208 2624 3232 depth 24 Learning rate 1.5e-3 1.5e-3 5e-4 4.2e-4 4e-4 3.5e-4

Training tokens 2.5B-600B Optimizer Fully decoupled AdamW [44] Optimizer Momentum β1 = 0.9,β2 = 0.95 Minimum Learning rate 0 Weight decay 1e-4 Batch size 2k Patch size (14, 14) Gradient clipping 1.0 MAximum Warmup iterations 5k Augmentations:

RandomResizedCrop size 224px scale [0.4, 1.0]

RandomHorizontalFlip p = 0.5

- Table 6. Pre-training hyperparameters We detail the hyperaparmeters used for pre-training different model configurations to derive scaling laws.

Vision encoder Interleaved Image-Caption Text AVG AVG (SFT) lr scaler (CE) (CE) (CE) (CE) (Acc) 1 2.521 2.15 2.867 2.513 43.49 0.1 2.502 2.066 2.862 2.477 52.27 0.01 2.502 2.066 2.859 2.476 53.76 0.001 2.513 2.066 2.857 2.479 – 0 (frozen) 2.504 2.061 2.856 2.474 54.14

- Table 7. Vision encoder scaler. Freezing the vision encoder works best when initializing late-fusion models with pre-trained models.

Image-Caption

Interleaved

2.8

2.6

ValidationLoss

2.4

2.6

2.2

100B 200B 300B 400B

100B 200B 300B 400B

Tokens seen

Tokens seen

Text-only

- 2.8
- 3

100B 200B 300B 400B

Tokens seen

Late-1B Late-2.4B Late-3.7B Early-932M Early-2.28B Early-3.3B

- Figure 14. Early vs late fusion: scaling training FLOPs. We compare early and late fusion models when scaling both the model size and the number of training tokens. The gap decreases mainly due to scaling models size.

40 60 80

2.6

2.65

2.7

% of Interleaved

Cross-entropy

Interleaved

Late

Early

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0 18 27 45 63 72

2.3

2.4

2.5

2.6

% of Interleaved

Image-CaptionCE

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0 18 27 45 63 72 90 % of Interleaved

- 2.9
- 3

TextCE

- Figure 15. Early vs late fusion: changing the training mixture. We vary the training mixtures and plot the final training loss. Early fusion models become better when increasing the proportion of interleaved documents. Early and late fusion has 1.63B and 1.75B parameters respectively.

Vision encoder Interleaved Image-Caption Text AVG AVG (SFT) lr scaler (CE) (CE) (CE) (CE) (Acc) 0.1 2.674 2.219 3.072 2.655 34.84 0.01 2.672 2.197 3.071 2.647 38.77 0.001 2.674 2.218 3.073 2.655 38.46

- Table 8. Vision encoder scaler. Reducing the learning rate for the vision encoder is better when training late-fusion models from scratch.

### B. Late vs early fusion

This section provides additional comparison between early and late fusion models.

#### B.1. Scaling FLOPs

Figure 14 compares early-fusion and late-fusion models when scaling FLOPs. Specifically, for each model size, we train multiple models using different amounts of training tokens. The performance gap between the two approaches mainly decreases due to increasing model sizes rather than increasing the number of training tokens. Despite the decreasing gap, across all the models that we train, earlyfusion consistently outperform late-fusion.

#### B.2. Changing the training data mixture

We analyze how the performance gap between early and late fusion models changes with variations in the training data mixture. As shown in Figure 16 and Figure 15, when fixing the model size, increasing the ratio of text and interleaved data favors early fusion. Interestingly, the gap remains largely unchanged for other data types. We also observe interference effects between different data types. Specifically, increasing the amount of interleaved data negatively impacts performance on image captions and vice versa. Additionally, increasing the proportion of text-only data slightly improves interleaved performance but increases loss on image captions. Overall, we find that text-only and interleaved data are correlated across different setups.

#### B.3. Scaling image resolution is in favor of earlyfusion

We examine how both architectures perform with varying image resolution. We fix the number of model parameters to 1.63B and 1.75B for early and late fusion respecively. All models are trained for 100K steps or 200B tokens. Since

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

2.66

InterleavedCE

2.66

2.65

2.65

10 20 30

% of Text

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

2.48

2.46

TextCE

2.44

2.42

10 20 30

% of Text

Text-only

2.9

Late

Early

2.85

2.8

10 15 20 25 30

% of Text

- Figure 16. Early vs late fusion: changing the amount of text-only data in the training mixture (isoFLOPs). We vary the ratio of text-only data and plot the final training loss. The gap increases with the text data ratio in favor of early fusion model. Early fusion has 1.63B parameters and late fusion 1.75B parameters.

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | |L|ate|-1B| | | |
| | | | | | | | | | | | |
| | | | | | |E|arl|y-9|32|M| |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

112px 168px 224px 280px 336px

2.5

2.52

2.54

2.56

Image resolution

ValidationLoss

Image-Caption

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | |L|ate|-1B| | | | | | |
| | | | | | | | | | | | |
| | | |E|arl|y-9|32|M| | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

112px 168px 224px 280px 336px

2.69

2.7

2.71

2.72

Image resolution

Interleaved

- Figure 17. Early vs late fusion: training with different image resolutions (isoFLOPs). For the same training FLOPs we vary the image resolution (and thus the number of image tokens) during training and report the final training loss. Increasing resolution, hurts the performance on text and interleaved documents, while helping image captioning. The gap stays almost the same on text and interleaved data while slightly increase on image captioning in favor of early fusion.

by normalization by FLOPs. When matching the text model size, early fusion performs better at higher ratios of interleaved data.

#### B.5. Different late-fusion configuration

We examine how this scaling changes with different latefusion configurations. Instead of scaling both the vision and text models equally, as done in the main paper, we fix the vision encoder size to 300M and scale only the text model. Figure 19 shows that late-fusion models lag behind at smaller model sizes, with the gap closing significantly as the text model scales. This suggests that allocating more parameters to shared components is more beneficial, further supporting the choice of early-fusion models.

#### B.6. Different context lengths

In the paper, we use a 1k context length following [31]. Also following, this paper, we ignore the context length effect, as the model dimension dominates the training compute estimate. Moreover, [53] empirically found that scaling coefficients are robust to context length. Nevertheless, Our initial experiments (Figure 20) indicate that scaling the context length did not significantly affect the comparison between late and early fusion.

the patch size remains constant, increasing the resolution results in a higher number of visual tokens. For all resolutions, we maintain the same number of text tokens. As

- shown in Figure 17, the early-fusion model consistently outperforms the late-fusion model across resolutions, particularly for multimodal data, with the performance gap widening at higher resolutions. Additionally, we observe that the loss on text and interleaved data increases as resolution increases.

- B.4. Early-fusion is consistently better when matching the late-fusion model size

In this section, we compare the late-fusion model with different configurations of early-fusion one. Specifically, we train early-fusion models that match the late-fusion model in total parameters (Params), text model size (Text), and FLOPs (FLOPs), assuming 45-45-10 training mixture. As

- shown in Figure 18, early fusion consistently outperforms late fusion when normalized by total parameters, followed

#### B.7. Initializing from LLM and CLIP

We study the case where both late and early fusion models are initialized from pre-trained models, specifically DCLM1B [39] and CLIP-ViT-L [55] for late fusion. Interestingly, Figure 21 shows that for text and interleaved multimodal documents, early fusion can match the performance of late fusion when trained for longer. However, closing the gap on image caption data remains more challenging. Notably, when considering the overall training cost, including that of pre-trained models, early fusion requires significantly longer training to compensate for the vision encoder’s pretraining cost.

Text CE

Interleaved CE

Paired CE

2.8

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

2.6

3

2.5

2.7

2.4

2.9

2.6

2.3

0 18 27 45 63 72 90

0 18 27 45 63 72

18 27 45 63 72

% of Interleaved

% of Interleaved

% of Interleaved

L E (Text) E (FLOPs) E (Params)

- Figure 18. Early vs late fusion: changing the training mixture and early-fusion configuration. We vary the training mixtures and plot the final training loss for different configuration of early fusion models. For the same number of total parameters early fusion consistently outperform late fusion.

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0.05 0.1 0.2 0.3 0.4

2.2

2.4

2.6

2.8

Tokens seen

Image-Caption CE

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0.05 0.1 0.2 0.3 0.4

2.6

- 2.8
- 3

Tokens seen

Interleaved CE

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0.05 0.1 0.2 0.3 0.4

- 2.8
- 3

3.2

Tokens seen

Text CE

Late-0.555B Late-1.14B Late-2.320B Late-3.33B

Early-0.464B Early-0.932B Early-1.627B Early-3.354B

- Figure 19. Early vs late fusion: scaling training FLOPs while fixing the vision encoder size. We compare early and late fusion models when scaling both the amount of training tokens and model sizes. For late fusion mdoels, we fix the vision encoder size (300M) and scale the text model (250M, 834M, 2B, 3B). The gap between early and late get tighter when scaling the text model.

0.1T 0.2T 0.3T

2.6

2.65

2.7

2.75

2.8

InterleavedCE

- Late-1k

- Late-2k

- Early-1k

- Early-2k

- Figure 20. Early vs late fusion with different context lengths.

Image-Caption CE

Interleaved CE

Text CE

2.8

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

2.6

2.4

2.78

2.55

2.76

2.3

2.74

2.2

2.5

100B 400B 1T

100B 400B 1T

100B 400B 1T

Tokens seen

Tokens seen

Tokens seen

Late-init Early-Init

Figure 21. Early vs late fusion when initializing the encoder and decoder. Early-fusion can match the performance of latefusion models when trained for longer. However, the gap is bigger on image-caption data.

### C. Scaling laws

#### C.1. Fitting L = F(N,D)

Following [26], we determine the parameters that minimize the following objective across all our runs i:

min

- a,b,e,α,β i

Huberδ (LSE (a − α log Ni, b − β log Di, e) − log Li) ,

(2)

We perform this optimization across various initialization ranges and select the parameters that achieve the lowest loss across all initializations. Specifically, our grid search spans {0,0.5,2.5} for α and β, {0,5,10,...,30} for a and

- b, and {−1,−0.5,1,0.5} for e. We use the L-BFGS algorithm with δ = 1e − 3.

- C.2. Fitting N ∝ Ca, D ∝ Cb, D ∝ Nd

While these equations have a closed-form solution [26] for early-fusion models that can be derived from Eq 1, this is not the case for late-fusion models without specifying either the vision encoder or text model size. To ensure a fair comparison, we derive these equations for both models, by performing linear regression in log space. We found that the regression is very close to the coefficient found with closed-form derivation Table 9. For instance, to derive N = KaCa, given a FLOP budget C and a set of linearly spaced tokens Di ranging from 10B to 600B, we compute the model size for each Di as Ni = 6CD for early fusion and Ni = 6CD + 0.483 ∗ Nv for late fusion (for the 45-45-10 mixture, Dv = 0.544D, thus C = 6D(0.544Nv + Nt)). We then apply Eq 1 to obtain the loss for each model size and select N that has the minimum loss. We repeat this for all FLOP values corresponding to our runs, resulting in a set of points (C,Nopt) that we use to regress a and Ka. We follow a similar procedure to find b and d. For late-fusion models, we regress a linear model to determine Nv given N. Notably, even though we maintain a fixed width ratio for late-fusion models, this approach is more accurate, as embedding layers prevent a strictly fixed ratio between text and vision model sizes. We present the regression results in

- Figure 22.

Model a b d n dn

Closed form 0.52649 0.47351 0.89938 1.11188 -0.05298 Regression 0.52391 0.47534 0.90052 1.10224 -0.04933

- Table 9. Scaling laws parameters for early-fusion. Doing regression to derive the scaling laws coefficients leads to very close results to using the closed-form solution.

#### C.3. Fitting L ∝ Cc

To determine the relationship between the final model loss and the compute budget C, we begin by interpolating the points corresponding to the same model size and compute

the convex hull that covers the minimum loss achieved by all runs for each FLOP. This results in a continuous mapping from the FLOPs to the lowest loss. We consider a range of FLOPs, excluding very small values (≤ 3e19), and construct a dataset of (C,L) for linearly spaced compute C. Using this data, we find the linear relationship between L and C in the log space and deduce the exponent c. We visualize the results in Figure 26.

N ∝ C0.526

N

1B

1020 1021 1022

C

10B

| |N|∝ C0.628| |
|---|---|---|---|
| | | | |
| | | | |

1B

100M

1020 1021 1022

C

D ∝ C0.462

| |D|∝ C0.473| |
|---|---|---|---|
| | | | |

100B

100B

D

1020 1021 1022

1020 1021 1022

C

C

101.8

D/N ∝ C−0.076

D/N ∝ C−0.053

C D/N

101.8

101.7

101.6

1020 1021 1022

1020 1021 1022

C

Figure 22. Regression results of the scaling laws coefficients. our estimation of the scaling coefficients is close to the closed form solution.

#### C.4. Scaling laws for different target data type

In Figure 27, we derive the scaling laws for different target data types. In general, we observe that the model learns image captioning faster than interleaved data, as indicated by the higher absolute value of the scaling exponent (e.g., 0.062 vs 0.046), despite using the same data ratio for captioning and interleaved data (45% each). Additionally, we find that the model learns more slowly on textonly data, likely due to the smaller amount of text-only data (10%). Across model configurations, we find that early fusion scales similarly to late fusion on image captioning but has a lower multiplicative constant (49.99 vs 47.97). For MoEs, the model learns faster but exhibits a higher multiplicative constant. On text and interleaved data, early and late fusion models scale similarly and achieve comparable

| | |0.275|B| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | |0.464|B| | | | | | | | |
| | | | | | | | | | | | |
| | |0.932|B| | | | | | | | |
| | | | | | | | | | | | |
| | |1.627 2.280|B B| | | | | | | | |
| | | | | | | | | | | | |
| | |3.354|B| | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

4

3.5

Predictedloss

3

2.5

2.5 3 3.5 4

Observed loss

| | |0.27|5B| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | |0.46|4B| | | | | | | | |
| | | | | | | | | | | | |
| | |0.93|2B| | | | | | | | |
| | | | | | | | | | | | |
| | |1.62 2.28|7B 0B| | | | | | | | |
| | | | | | | | | | | | |
| | |3.35|4B| | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

- 3.5
- 4

Predictedloss

- 2.5
- 3

2.5 3 3.5 4

Observed loss

- Figure 23. Observed vs predicted loss. We visualize the loss predicted by our scaling laws (Eq 1) and the actual loss achived by each run.

performance. However, MoEs demonstrate better overall performance while learning slightly more slowly.

#### C.5. Scaling laws for different training mixtures

We investigate how the scaling laws change when modifying the training mixtures. Specifically, we vary the ratio of image caption, interleaved, and text-only data and report the results in Figure 28. Overall, we observe similar scaling trends, with only minor changes in the scaling coefficients. Upon closer analysis, we find that increasing the ratio of a particular data type in the training mixture, leads to a corresponding increase in its scaling exponent. For instance, increasing the ratio of image captions from 30% to 40% raises the absolute value of the exponent from 0.056 to 0.061. However, for text-only data, we do not observe significant changes in the scaling coefficients when varying its proportion in the training mixture.

Parameter MSE R2 MAE (%)

Held-in 0.0029 0.9807 0.8608 Held-out 0.0004 0.9682 0.5530

- Table 10. Scaling laws prediction errors. We report the mean square error, R2 and mean absolute error for the loss prediction for held-in and held-out (8B model) data.

Model E α β a b d

Avg 1.80922 0.29842 0.33209 0.54302 0.48301 0.92375 Std 0.33811 0.10101 0.02892 0.08813 0.05787 0.23296

- Table 11. Scaling laws sensitivity. We report the mean and standard deviation after bootstrapping with 100 iterations.

#### C.6. Scaling laws evaluation

For each model size and number of training tokens, we compute the loss using the estimated functional form in Eq 1 and compare it to the actual loss observed in our runs. Figure 23, Figure 24, and Table 10 visualizes these comparisons, showing that our estimation is highly accurate, particularly for lower loss values and larger FLOPs. We also assess our scaling laws in an extrapolation setting, predicting performance beyond the model sizes used for fitting. Notably, our approach estimates the performance of an 8B model with reasonable accuracy.

Additionally, we conduct a sensitivity analysis using bootstrapping. Specifically, we sample P points with replacement (P being the total number of trained models) and re-estimate the scaling law coefficients. This process is repeated 100 times, and we report the mean and standard deviation of each coefficient. Table 11 shows that our estimation is more precise for β than for α, primarily due to the smaller number of model sizes relative to the number of different token counts used to derive the scaling laws.

#### C.7. Scaling laws for sparse NMMs.

Similar to dense models, we fit a parametric loss function (Eq 1) to predict the loss of sparse NMMs based on the number of parameters and training tokens, replacing the total parameter count with the number of active parameters. While incorporating sparsity is standard when deriving scaling laws for MoEs [2, 33, 74], we focus on deriving scaling laws specific to the sparsity level used in our MoE setup. This yields coefficients that are implicitly conditioned on the sparsity configuration.

We also experiment with a sparsity-aware formulation of the scaling law as proposed in [2], and observe consistent

275M 464M 932M

- 3.5
- 4

- 1.63B

- 2.28B

- 3.35B

Predictedloss

that techniques for extending context length could be beneficial.

275M 464M 932M

4

8.13B

Scaling laws for multimodal MoEs models. For MoEs, we consider only a single configuration (top-1 routing with 8 experts). We found this configuration to work reasonably well in our setup, and follow a standard MoEs implementation. However, the findings may vary when optimizing more the MoE architecture or exploring different load-balancing, routing strategies or different experts implementations.

- 1.63B

- 2.28B

- 3.35B

3

3.5

Predictedloss

8.13B

3

2.5

2.5

### E. Mixture of experts and modality-specific specialization

2.5 3 3.5 4

#### E.1. MoEs configuration

Observed loss

2.5 3 3.5 4

We experiment with different MoEs configuration by changing the number of experts and the top-k. We report a sample of these experiments in Table 13.

- Figure 24. Observed vs predicted loss. We visualize the loss predicted by our scaling laws Eq 1 and the actual loss achieved by each run. We can reliably predict the performance of models larger (8B params) than those used to fit the scaling laws.

Observed loss

#### E.2. MoEs specialization

trends (Table 12). In particular, the exponents associated with model size (N) are substantially larger than those for training tokens (β), reinforcing the importance of scaling model size in sparse architectures. Additionally, we observe that the terms governing the scaling of active parameters decompose into two components.

15.0

Text

Modalityspecialization

12.5

Image

10.0

7.5

### D. Discussion and Limitations

5.0

Scaling laws for multimodal data mixtures. Our scaling laws study spans different model configurations and training mixtures. While results suggest that the scaling law coefficients remain largely consistent across mixtures, a broader exploration of mixture variations is needed to validate this observation and establish a unified scaling law that accounts for this factor.

2.5

0 5 10 15 20 Layers

Figure 25. Modality-specific specialization. We visualize the experts specialization to text and image modalities. Models are evaluated on Obelics.

Scaling laws and performance on downstream tasks. Similar to previous scaling law studies, our analysis focuses on pretraining performance as measured by the validation loss. However, the extent to which these findings translate to downstream performance remains an open question and requires further investigation.

We investigate multimodal specialization in MoE architectures. We compute a specialization score as the average difference between the number of text/images tokens assigned to each expert and a uniform assignment (1/E). Additionally, we visualize the normalized number of text and image tokens assigned to each expert across layers. Figure 25 shows clear modality-specific experts, particularly in the early layers. Furthermore, the specialization score decreases as the number of layers increases but rises again in the very last layers. This suggests that early and final layers require more modality specialization compared to midlayers. Additionally, we observe several experts shared between text and image modalities, a phenomenon not present in hard-routed or predefined modality-specific experts.

Extrapolation to larger scales. The accuracy of scaling law predictions improves with increasing FLOPs Appendix C. Furthermore, we validate our laws when extrapolating to larger model sizes (Appendix C.6). However, whether these laws can be reliably extrapolated to extremely large model sizes remains an open question.

High resolution and early-fusion models. Training earlyfusion models with high-resolution inputs leads to a significant increase in vision tokens. While pooling techniques have been widely adopted for late-fusion models, alternative approaches may be necessary for early fusion. Given the similarity of early-fusion models to LLMs, it appears

vs L(N,D,S) = NAα + DB

L(N,D) = E + NAα + DB

###### + (1−CS)

+ (1−Sd)

δNγ + E

β

β

λ

Model E A B α β λ δ γ C d L(N,D) (Eq 1) 2.158 381773 4659 0.710 0.372 – – – – – L(N,D,S) [2] 1.0788 1 4660 0.5890 0.3720 0.2 0.2 0.70956 1.0788 381475

Table 12. Scaling laws for sparse native multimodal models.

Accuracy CIDEr

AVG VQAv2 TextVQA OKVQA GQA VizWiz COCO TextCaps 4-E-top-1 40.0552 64.068 14.284 41.948 61.46 18.516 62.201 34.08 8-E-top-1 41.6934 65.684 17.55 42.908 63.26 19.065 67.877 39.63 8-E-top-2 42.8546 66.466 19.162 45.344 63.94 19.361 65.988 41.649 8-E-top-2 finegrained 39.904 62.76 15.58 41.88 61.6 17.7 57.52 35.42

###### Table 13. SFT results with different MoEs configurations. .

- 2.5
- 3

3.5

4

- 4.5

L = 29.923C−0.0494

ValidationLoss

ValidationLoss

1019 1020 1021 1022

FLOPs

45-45-10

L = 29.574C−0.0492

- 3.5
- 4

ValidationLoss

- 2.5
- 3

1019 1020 1021 1022

FLOPs

L = 27.086C−0.048

- 3
- 4

1019 1020 1021 1022

FLOPs

0.289B 0.494B 1B 1.748B 2.430B 3.714B 0.275B 0.464B 0.932B 1.627B 2.280B 3.354B 0.275B 0.464B 0.932B 1.627B 2.280B 3.354B

- Figure 26. Scaling laws for native multimodal models. From left to right: late-fusion (dense), early-fusion (dense) and early-fusion MoEs. The scaling exponents are very close for all models. However, MoEs leads to overall lower loss (smaller multiplicative constant) and takes longer to saturate.

45-45-10

| | | | | |
|---|---|---|---|---|
| | |L = 49.9|9C−0.062| |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

4

Image-CaptionCE

3.5

3

2.5

1019 1020 1021 1022

FLOPs

| | | | | |
|---|---|---|---|---|
| | |L = 25.30|3−0.0460| |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

4

3.5

InterleavedCE

3

2.5

1019 1020 1021 1022

FLOPs

45-45-10

###### L = 22.642−0.042

4

TextCE

3.5

3

1019 1020 1021 1022

FLOPs

| | | | | |
|---|---|---|---|---|
| | |L = 47.9|7C−0.061| |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

- 3.5
- 4

Image-CaptionCE

- 2.5
- 3

1019 1020 1021 1022

FLOPs

4

| | | | | |
|---|---|---|---|---|
| |L|= 25.114|C−0.0458| |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

3.5

InterleavedCE

- 2.5
- 3

1019 1020 1021 1022

FLOPs

45-45-10

| | | | | |
|---|---|---|---|---|
| |L|= 22.709|C−0.042| |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

- 3.5
- 4

TextCE

3

1019 1020 1021 1022

FLOPs

L = 51.857C−0.064

- 2
- 3
- 4

Image-CaptionCE

InterleavedCE

1019 1020 1021 1022

FLOPs

L = 22.715C−0.044

- 3
- 4

1019 1020 1021 1022

FLOPs

- 2.5
- 3

- 3.5
- 4

- 4.5

| | | |0.040| |
|---|---|---|---|---|
| | |L = 20.0|36C−| |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

TextCE

1019 1020 1021 1022

FLOPs

0.289B 0.494B 1B 1.748B 2.430B 3.714B 0.275B 0.464B 0.932B 1.627B 2.280B 3.354B 0.275B 0.464B 0.932B 1.627B 2.280B 3.354B

- Figure 27. Scaling laws for native multimodal models. From top to bottom: late-fusion (dense), early-fusion (dense) and early-fusion MoEs. From left to right: cross-entropy on the validation set of image-caption, interleaved and text-only data.

| | | | | |
|---|---|---|---|---|
| | |L = 47.9|7C−0.061| |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

4

Image-CaptionCE

3.5

3

2.5

1019 1020 1021 1022

FLOPs

| | | |0.061| |
|---|---|---|---|---|
| |L|= 49.477|C−| |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

- 3.5
- 4

Image-CaptionCE

3

2.5

1019 1020 1021 1022

FLOPs

| | | |−0.056| |
|---|---|---|---|---|
| |L|= 39.518|C| |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

4

Image-CaptionCE

3.5

- 2.5
- 3

1019 1020 1021 1022

FLOPs

45-45-10

4

| | | | | |
|---|---|---|---|---|
| |L|= 25.114|C−0.0458| |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

3.5

InterleavedCE

3

2.5

1019 1020 1021 1022

FLOPs

40-20-40

- 3.5
- 4

| | | | | |
|---|---|---|---|---|
| |L|= 22.112|C−0.043| |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

InterleavedCE

3

2.5

1019 1020 1021 1022

FLOPs

30-30-40

- 3.5
- 4

| | | |0.043| |
|---|---|---|---|---|
| |L|= 22.11|1C−| |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

InterleavedCE

- 2.5
- 3

1019 1020 1021 1022

FLOPs

20-40-40

| | | | | |
|---|---|---|---|---|
| |L|= 22.709|C−0.042| |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

4

TextCE

3.5

3

1019 1020 1021 1022

FLOPs

| | | | | |
|---|---|---|---|---|
| |L|= 21.35|2C−0.042| |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

4

3.5

TextCE

3

2.5

1019 1020 1021 1022

FLOPs

| | | | | |
|---|---|---|---|---|
| |L|= 20.25|7C−0.041| |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

- 3.5
- 4

TextCE

- 2.5
- 3

1019 1020 1021 1022

FLOPs

| | | | | |
|---|---|---|---|---|
| |L =|46.216C|−0.0589| |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

- 3.5
- 4

Image-CaptionCE

- 2.5
- 3

1019 1020 1021 1022

FLOPs

| | | | | |
|---|---|---|---|---|
| |L|= 23.88|8C−0.045| |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

3.5

InterleavedCE

3

2.5

1019 1020 1021 1022

FLOPs

| | | | | |
|---|---|---|---|---|
| |L|= 22.150|C−0.0425| |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

- 3.5
- 4

TextCE

- 2.5
- 3

1019 1020 1021 1022

FLOPs

0.275B 0.464B 0.932B 1.627B 2.280B 3.354B

- Figure 28. Scaling laws for early-fusion native multimodal models. Our runs across different training mixtures (Image-captionInterleaved-Text) and FLOPs. We visulize the final validation loss on 3 data types: HQITP (left), Obelics (middle) and DCLM (right).

