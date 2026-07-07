## Progressive Residual Warmup for Language Model Pretraining

Tianhao Chen1 Xin Xu1 Lu Yin2 Hao Chen1 Yang Wang3 Shizhe Diao4 Can Yang1

# arXiv:2603.05369v1[cs.CL]5Mar2026

### Abstract

Transformer architectures serve as the backbone for most modern Large Language Models, therefore their pretraining stability and convergence speed are of central concern. Motivated by the logical dependency of sequentially stacked layers, we propose Progressive Residual Warmup (ProRes) for language model pretraining. ProRes implements an “early layer learns first” philosophy by multiplying each layer’s residual with a scalar that gradually warms up from 0 to 1, with deeper layers taking longer warmup steps. In this way, deeper layers wait for early layers to settle into a more stable regime before contributing to learning. We demonstrate the effectiveness of ProRes through pretraining experiments across various model scales, as well as normalization and initialization schemes. Comprehensive analysis shows that ProRes not only stabilizes pretraining but also introduces a unique optimization trajectory, leading to faster convergence, stronger generalization and better downstream performance. Our code is available at https://github.com/dandingsky/ProRes.

### 1. Introduction

The Transformer (Vaswani et al., 2017) has become one of the most dominant backbones for Large Language Models (LLMs), underpinning recent advances across natural language processing and beyond. Among the many components of the Transformer architecture, residual connection (He et al., 2016) and normalization (Ba et al., 2016) play a key role in its optimization, enabling scaling LLMs up to trillion parameters (Team et al., 2025).

Despite these advances, scaling Transformers poses unique optimization challenges. A large body of prior work has explored stabilizing Transformer training, including switch-

1The Hong Kong University of Science and Technology, Hong Kong, China 2University of Surrey, Guildford, UK 3The University of Hong Kong, Hong Kong, China 4NVIDIA, Santa Clara, US. Correspondence to: Can Yang <macyang@ust.hk>.

Preprint. March 6, 2026.

Pre-LN

15.5

Pre-LN (DS-Init)

DeepNorm

15.0

Pre-LN (Scaled Init)

Pre-LN (ProRes-equal)

14.5

LayerNorm Scaling

Sandwich-LN

Pre-LN (ProRes)

14.0

Perplexity

13.5

13.0

12.5

12.0

24 48 72 96 120 Number of Layers

Figure 1. Evaluation perplexity (↓) of different methods as model depth increases. See Section 4.4 for details.

ing from Post-LN to Pre-LN (Xiong et al., 2020), bounding model updates through depth- or width-aware initializations (Zhang et al., 2019b; Huang et al., 2020; Zhang et al., 2019a; Liu et al., 2020; Wang et al., 2024; Yang et al., 2022; Sun et al., 2025), and introducing additional flexibility into residual connections (Wang et al., 2019; Bachlechner et al., 2021; Xie et al., 2023; Zhu et al., 2024). While effective, these methods are largely not training-phase-aware, in the sense that their mechanisms are typically applied at initialization, and then leave the optimizer to figure out how to adapt throughout training. In practice, Transformer training exhibits distinct phases (Li et al., 2025). Early training, such as the warmup stage, is often characterized by large and chaotic model updates, whereas the stable training phase involves relatively small and gradual modifications. Moreover, convergence dynamics are heterogeneous across depth, with shallow1 layers tending to converge earlier than deeper layers (Erdogan et al., 2025). These inherent dynamics suggest that Transformer optimization may benefit from explicitly coordinating layerwise learning across different stages of training.

1We use “shallow” or “early”, and “deeper” or “later” interchangeably to describe layer order.

Motivated by these considerations, we revisit the role of residual connections in layerwise learning from a temporal perspective. Modern LLMs typically consist of sequentially stacked Transformer layers, where each layer updates its input representation2 through residual branches consisting of attention and feed-forward sublayers. While residual connections enable effective optimization at scale, they also allow all layers to modify representations simultaneously from initialization. In the absence of explicit coordination, deeper layers may begin contributing before upstream representations have stabilized, potentially leading to inefficient updates or conflicting learning signals. This observation raises a natural question: can residual contributions be scheduled to respect the staged nature of Transformer training? In particular, can earlier layers be prioritized during early training, while allowing deeper layers to fully engage once upstream representations have matured?

In this work, we propose Progressive Residual Warmup (ProRes), a simple and scalable approach for coordinating layerwise residual learning over the course of training. ProRes assigns each layer’s residual with a predefined scaling factor. This scalar is initialized to 0, and gradually increases to 1 as training progresses. In particular, deeper layers take longer residual warmup than shallow layers, delaying their contribution to representations until shallow layers enter a more stable regime. As a result, shallow layers are prioritized in early training, while deeper layers progressively engage later on. This design controls model updates in the early stages, reduces unnecessary interference across layers, and preserves the expressive capacity of deep layers in later phases of optimization.

We evaluate ProRes through extensive language model pretraining experiments across a range of model scales, initialization methods, and normalization schemes. As shown in Figure 1, ProRes consistently improves depth scaling performance, enabling deeper models to achieve better perplexity without compromising training stability. We further analyze the optimization dynamics introduced by ProRes, shedding light on how coordinating learning order across depth affects learning behavior and representation evolution.

Our main contributions can be summarized as follows:

- • We propose ProRes, a residual learning scheme that explicitly coordinates layerwise contributions while respecting the staged nature of Transformer convergence.
- • Extensive pretraining experiments from 71M to 7B parameters show that ProRes improves performance across a wide range of model scales, initialization methods, and normalization schemes.
- • We analyze the learning dynamics induced by ProRes,

2We use “representation”, “hidden state”, and “activation” interchangeably. Unless otherwise mentioned, they refer to the token representation at certain Transformer layers.

providing insight into how learning order across layers influences training stability and efficiency.

### 2. Related Work

Transformer initialization and normalization. The objective of better initialization or normalization schemes is to stabilize training or accelerate convergence, by controlling the magnitude of activations, weights, gradients and model updates, etc. The original Post-LN (Vaswani et al., 2017) stabilizes activations by performing normalization after the residual connection. In more recent works, Pre-LN and its variants are adopted because of their stability and efficiency at scale (Radford et al., 2019; Xiong et al., 2020; Henry et al., 2020; Ding et al., 2021). To enable effective depth scaling, several works proposed to scale layerwise weights according to the total number of layers, either uniformly (Radford et al., 2019; Shoeybi et al., 2019; Huang et al., 2020; Wang et al., 2024), or layer-specific (Zhang et al., 2019a; Gururangan et al., 2023). Yang et al. (2022) proposed an initialization scheme that enables hyperparameter transfer among various scales. OLMo et al. (2024) found that a simple normal initialization for all weights is more stable and transferable in their setting. A number of works investigated enhancing or replacing normalization with learnable gates or predefined scalings on the residual connection (Bachlechner et al., 2021; De & Smith, 2020; Touvron et al., 2021; Liu et al., 2024; Sun et al., 2025; Chen et al., 2025). Efforts have been made to combine the advantages of Pre-LN and Post-LN (Xie et al., 2023; Li et al., 2024; 2026). Several works explored enhancing representation capacity by expanding the width of residual stream (Wang et al., 2019; Zhu et al., 2024; Xie et al., 2025).

Progressive growing and freezing layers. A similar line of research focuses on progressive training of Transformers. Gong et al. (2019) proposed to train deeper BERT models by progressively stacking shallower ones, resulting in an overall shorter training time. Yang et al. (2020) further proposed only training the newly added layers on top for improved efficiency. Similarly, Erdogan et al. (2025) found that shallow layers converge earlier than deeper layers, and proposed a progressive freezing training method to reduce convergence wall time.

### 3. Method 3.1. Overview

We begin by describing the implementation of ProRes using Pre-LN Transformer as an example. The forward equation of a Pre-LN layer is:

##### xl+1 = xl + F(Norm(xl)), (1)

Table 1. Forward equations of Transformer variants, with (✔) and without (✘) ProRes.

Method ProRes Forward Equation Pre-LN

✘ xl+1 = xl + F(Norm(xl))

✔ xl+1 = xl + α(l,t) · F(Norm(xl)) Post-LN

✘ xl+1 = Norm(xl + F(xl))

✔ xl+1 = Norm(xl + α(l,t) · F(xl)) Sandwich-LN

✘ xl+1 = xl + Norm(F(Norm(xl)))

✔ xl+1 = xl + α(l,t) · Norm(F(Norm(xl))) DeepNorm

✘ xl+1 = Norm(α · xl + Fβ(xl))

✔ xl+1 = Norm(α · xl + α(l,t) · Fβ(xl)) LayerNorm Scaling

√

✘ xl+1 = xl + F(Norm(xl)/

l)

√

✔ xl+1 = xl + α(l,t) · F(Norm(xl)/

l)

where xl is the input representation for layer l, F denotes the attention or FFN module, and Norm denotes normalization layer. ProRes modifies the residual connection with a scalar:

xl+1 = xl + α(l,t) · F(Norm(xl)). (2)

Here α(l,t) ∈ R is a predefined scalar that depends on the layer index l and the training step t. An example schedule for this variable is:

t T × l

,1 , l = 1,...,L, (3)

α(l,t) = min

where T is the warmup length for the first layer and L is the total number of layers. In other words, α(l,t) warms up from 0 to 1 linearly during training, and the warmup length increases linearly with layer index. Under this schedule, residual branches are activated sequentially from shallow to deep layers. We study multiple alternative schedules in Section 4.3.

Applying ProRes to other Transformer variants follows the same principle, as long as they consist of stacked Transformer layers with residual connection. All variants studied in this paper are summarized in Table 1.

- 3.2. Motivation of ProRes ProRes is motivated by several principles.

Principle 1: Identity behavior at initialization. Deep residual networks benefit from behaving close to identity mapping at initialization, which helps control activation growth and ensures the network has well behaved gradients. Prior work has shown that normalization in residual architectures implicitly biases networks toward identity mappings at initialization (De & Smith, 2020). ProRes makes this bias explicit and exact by initializing the residual scaling parameters as α(l,t) = 0 when t = 0, making identity the exact starting point in the network’s optimization trajectory.

- Principle 2: Bounded model update with respect to depth and time. Stable optimization in deep Transformers requires controlling the overall magnitude of model updates so that training dynamics remain well behaved as model depth increases. Several prior works (Zhang et al., 2019b; Huang et al., 2020; Wang et al., 2024) established that preserving a bounded model update is critical for training deep Transformers. However, these bounds are primarily derived from model behavior at initialization, where updates are typically the most unstable, and may be unnecessarily conservative when applied uniformly throughout training. In practice, Transformer optimization progresses through distinct phases (Li et al., 2025), such as warmup, stable training, and decay (Hu et al., 2024; Kosson et al., 2024; Wen et al., 2024; Dremov et al., 2025), during which the scale and smoothness of model updates differ substantially. Once the model enters a stable training regime, update constraints designed for initialization may limit learning capacity. Moreover, empirical studies show that shallow layers tend to converge earlier than deeper layers during training (Erdogan et al., 2025), which calls for layer-specific treatment. ProRes extends the bounded update principle from initialization to the entire training trajectory in a layerwise and temporal manner. By progressively activating residual branches, ProRes allows shallow layers to stabilize before deeper layers begin to contribute. This encourages representation updates to remain bounded across both depth and training time, stabilizing the warmup phase without sacrificing learning capacity in the stable training regime.
- Principle 3: Respecting sequential learning and contribution order. In a Transformer with sequentially stacked layers, deeper layers’ inputs depend on shallow layers, while shallow layers’ gradients depend on deeper layers. Early in training, allowing deeper layers to introduce large residual modifications can inject noise into intermediate representations and skew gradient signals for shallow layers. ProRes enforces a sequential learning order by delaying the contribution of deeper residual branches, ensuring that deeper layers build upon stable representations rather than amplifying early-stage noise with randomly initialized deep residual branches.

### 4. Experiments

We evaluate ProRes across a range of pretraining settings to assess its effectiveness, scalability, and generality. First, we conduct pretraining experiments on models ranging from 130M to 1.3B parameters on 50B tokens, comparing and combining ProRes with several Transformer variants. We also scale up training to 7B parameters on Pre-LN in Appendix B. Second, we perform ablations on the residual warmup schedule α(l,t) to study the effect of learning and contribution order across layers. Third, we examine depth

scaling behavior by training models with depths ranging from 12 to 120 layers. Finally, in Appendix C, we evaluate ProRes on an alternative pretraining corpus to verify its robustness across data distributions.

#### 4.1. Experiment Setup

Model architecture. All models are decoder-only Transformers with Llama-based Attention and MLP (Grattafiori et al., 2024), which utilize RMSNorm (Zhang & Sennrich, 2019), SwiGLU activation (Shazeer, 2020) and Rotary Position Embedding (Su et al., 2024). The initialization follows (OLMo et al., 2024) where all layers’ weights are initialized from truncated normal distribution N(0,0.022). Depth-dependent initializations are scaled based on this default initialization, including DS-Init (Zhang et al., 2019a), Scaled Init (Shoeybi et al., 2019), and DeepNorm (Wang

- et al., 2024).

Baselines and ProRes schedule. We evaluate ProRes on diverse Transformer variants listed in Table 1, including Pre-LN (Xiong et al., 2020), Post-LN (Vaswani et al., 2017), Sandwich-LN (Ding et al., 2021), DeepNorm (Wang et al., 2024), and LayerNorm Scaling (LNS) (Sun et al., 2025). For Pre-LN, we additionally experiment with two initialization methods, namely DS-Init (as implemented by Gururangan et al. (2023)), and Scaled Init. For experiments with ProRes, we adopt the linear schedule in Equation (3) with T = 1000. We did not tune this schedule for the main experiment. The effect of schedule choice is studied separately in Section 4.3.

Pretraining data. We primarily train on the C4-en dataset (Raffel et al., 2020; Dodge et al., 2021) tokenized with the Llama Tokenizer (Grattafiori et al., 2024). Unless otherwise stated, all experiments train on a randomly sampled subset with 50B tokens and sequences packed to length of 1024. Evaluation perplexities are calculated on a randomly sampled subset of the C4-en test set with 10M tokens. The set of experiments that train on ClimbMix (Diao

- et al., 2025) use the same preprocessing protocol, and evaluation perplexities on ClimbMix are calculated on a held out 10M tokens subset.

Optimizer and learning rates. All experiments use the AdamW optimizer (Loshchilov & Hutter, 2017; Kingma, 2014) with β = (0.9,0.95), weight decay 0.1, ϵ = 10−8, and gradient clipping 1.0. A global batch size of 512 with 100k training steps is adopted for all runs. We use the Warmup-Stable-Decay learning rate scheduler (Hu et al., 2024) with 2000 warmup steps and a linear decay to zero in the final 10% steps. Post-LN and DeepNorm use a longer warmup of 10% training steps for better convergence. Learning rates are tuned separately for each configuration with details provided in Appendix A.

Table 2. Perplexity (↓) on C4-en test set across model scales. Cells with ProRes are shaded by improvement magnitude.

Method ProRes 130M 350M 1.3B Pre-LN

✘ 14.67 12.36 10.32 ✔ 14.30 11.74 9.86

✘ 14.62 12.24 10.32 ✔ 14.29 11.73 9.85

Pre-LN (DS-Init)

✘ 14.63 12.29 10.30 ✔ 14.28 11.72 9.84

Pre-LN (Scaled Init)

✘ 14.55 11.97 10.16 ✔ 14.50 11.78 9.94

Sandwich-LN

✘ 14.45 11.74 9.93 ✔ 14.22 11.62 9.89

LayerNorm Scaling

✘ 14.84 12.74 11.62 ✔ 14.72 11.92 10.53

Post-LN

✘ 14.57 12.38 10.32 ✔ 14.45 11.97 10.09

DeepNorm

Evaluation. Other than perplexity measured on test set of the pretraining corpus, we evaluate the pretrained models on general reasoning benchmarks using LM Evaluation Harness (Gao et al., 2024). We report performance on PIQA (Bisk et al., 2020), SIQA (Sap et al., 2019), HellaSwag (Zellers et al., 2019), WinoGrande (Sakaguchi et al., 2021), ARC-Easy and ARC-Challenge (Clark et al., 2018), OpenBookQA (Mihaylov et al., 2018), RACE (Lai et al., 2017), LAMBADA (Paperno et al., 2016), and MMLU (Hendrycks et al., 2020). Perplexity is additionally reported on WikiText (Merity et al., 2016) and LAMBADA.

#### 4.2. Main Results

For the main experiment, we study the effectiveness of ProRes via pretraining models at scales of 130M, 350M, and 1.3B, and across various Transformer architectures.

Perplexity on the pretraining corpus. Table 2 shows perplexity on the test split of the pretraining corpus. Across all configurations, applying ProRes consistently leads to a notable decrease in perplexity, demonstrating its effectiveness as a general residual modification. Among the evaluated architectures, Post-LN benefits the most from ProRes. We attribute this to the normalization in Post-LN naturally biasing contribution towards deeper layers (Xiong et al., 2020; Li et al., 2024), whereas ProRes corrects for this inherent tendency via sequentially activating shallow to deep residuals. These improvements are consistent across scales and more effective for larger models. In contrast, combining ProRes with LNS yields smaller gains at larger scales. We hypothesize that since LNS already down-weights deeper residuals by 1/

√

l, the additional scaling induced by ProRes further suppresses their contribution, delaying effective learning in

Table 3. Zero-shot accuracy (↑) on reasoning benchmarks for 1.3B models.

Method ProRes PIQA SIQA Hella Wino ARC-e ARC-c OBQA RACE Lambada MMLU Avg Pre-LN

✘ 72.85 41.86 52.45 55.41 53.54 26.45 33.40 31.29 39.30 27.69 43.42 ✔ 73.34 41.04 56.39 57.54 54.84 28.24 34.20 33.21 42.71 28.44 45.00

✘ 72.58 41.66 52.92 56.35 53.79 27.39 34.20 32.25 39.69 27.79 43.86 ✔ 73.45 40.74 56.02 57.77 56.52 27.13 35.60 32.73 42.34 27.80 45.01

Pre-LN (DS-Init)

✘ 72.47 39.97 53.65 54.54 52.86 26.11 32.00 32.15 37.78 27.30 42.88 ✔ 73.83 41.10 56.48 56.83 56.78 28.07 35.20 31.10 42.34 28.08 44.98

Pre-LN (Scaled Init)

✘ 72.69 40.94 53.96 57.22 55.98 27.39 33.60 31.77 38.79 27.70 44.00

Sandwich-LN

- ✔ 72.31 40.99 55.55 56.12 56.06 27.65 34.60 31.87 41.63 28.17 44.49

LayerNorm Scaling

✘ 73.34 41.30 55.68 56.35 55.77 27.82 33.20 33.59 38.93 28.19 44.42

- ✔ 73.99 41.25 55.89 55.88 55.72 27.99 35.80 33.01 39.08 27.77 44.64

✘ 69.97 39.71 47.00 53.43 50.42 24.74 31.40 30.14 33.24 26.24 40.63

Post-LN

- ✔ 72.74 40.94 51.81 54.62 53.58 27.56 34.40 31.77 37.82 27.56 43.28

DeepNorm

✘ 72.96 39.76 52.48 54.30 54.08 26.79 34.80 32.15 40.27 27.89 43.55

- ✔ 73.23 40.43 54.71 53.99 55.89 26.37 34.80 32.73 42.31 27.98 44.24

Average ∆Acc – +0.86 +0.18 +2.67 +0.73 +1.85 +0.90 +1.71 +0.44 +2.89 +0.43 +1.27

- Table 4. Perplexity (↓) on WikiText and Lambada for 1.3B models. Final row shows mean perplexity reduction (baseline minus ProRes).

Method ProRes WikiText Lambada Pre-LN

✘ 25.61 20.46 ✔ 23.67 15.05

✘ 25.77 19.83 ✔ 23.98 15.57

Pre-LN (DS-Init)

✘ 25.63 20.93

Pre-LN (Scaled Init)

- ✔ 23.62 15.42

Sandwich-LN

✘ 24.90 19.77

- ✔ 24.12 16.09

✘ 24.28 20.79 ✔ 24.19 18.79

LayerNorm Scaling

✘ 30.37 33.53 ✔ 26.74 22.91

Post-LN

✘ 25.60 19.72 ✔ 24.77 17.15

DeepNorm

Average ∆PPL – 1.58 4.86

deeper layers within the fixed 100k training steps.

Performance on reasoning benchmarks. Table 3 reports zero-shot accuracy on several reasoning benchmarks. Overall, ProRes provides consistent improvements across architectures, yielding an average gain of 1.27% over the corresponding baselines. The largest gains are observed on HellaSwag, ARC-Easy, OBQA, and LAMBADA. On LAMBADA, ProRes increases accuracy by 2.89%, indicating enhanced language modeling of long range dependency. The top three average scores are achieved by Pre-LN mod-

els with ProRes, and the top five highest averages are all ProRes variants, highlighting the consistent benefit across architectures.

Generalization to out-of-distribution corpora. Table 4 reports perplexity on WikiText and LAMBADA. While ProRes reduces perplexity moderately on the pretraining corpus (∆PPL ≈ 0.4), it yields substantially larger gains on these out-of-distribution datasets. Notably, the average perplexity reduction reaches 4.86 on LAMBADA, corresponding well with the observed accuracy improvement on this benchmark.

#### 4.3. Warmup Schedule Ablation

We now study the influence of different ProRes schedules α(l,t) on training dynamics and model performance, with the goal of empirically validating the design motivations discussed in Section 3.2. We conduct ablation on a diverse set of schedules listed in Table 5, and across 3 representative Transformer variants: Pre-LN, Post-LN, and Sandwich-LN. For all ablation runs, we pretrain 350M models with 24 layers (L = 24) on 6B tokens over 60k training steps, which aligns with the Chinchilla Scaling Law (Hoffmann et al., 2022). Perplexities on test set are reported in Table 6. We summarize the main observations regarding layer activation order, warmup length, and schedule design below.

Effect of layer contribution order. We observe that the order in which residual branches are activated across depth plays a significant role in optimization. Schedules that activate residuals progressively from shallow to deep layers (“linear”) consistently outperform schedules that activate all

- Table 5. ProRes warmup schedules. Here l ∈ {1, . . . , L} denotes the layer index, t the training step, and T the warmup length for the first layer. “Warmup” denotes the number of training steps required for all layers to complete residual warmup. See Appendix D for visualization.

Schedule Warmup α(l,t) linear T ·L min T t×l,1 linear-sqrt T ·L min T t×l,1

- 1/2

linear-square T ·L min T t×l,1

- 2

equal T min(t/T,1) reverse T ·L min T×(L t−l+1),1 stagewise-0 T ·L clip t−T×T(l−1),0,1 stagewise-L T ·L clip t−T×T(l−1),0,1 · (1−1/L)+1/L stagewise-√

√

√

l T ·L clip t−T×T(l−1),0,1 · (1−1/

l)+1/

l

fix-L – 1/L fix-√

√

L – 1/

L fix-√

√

l – 1/

l

layers simultaneously3 (“equal”) or prioritize deeper layers early in training (“reverse”). This trend holds across all three architectures, indicating that respecting the sequential dependency structure of stacked Transformer layers is beneficial for learning.

Warmup length and interaction with layer order. The “linear” schedule remains effective across a wide range of warmup lengths, provided the warmup is neither too short (< 1k steps) nor so long that it occupies a substantial fraction of training (> 48k steps). In contrast, the “equal” schedule is considerably more sensitive to warmup length and can lead to degraded performance or even divergence as warmup increases. We conjecture that the “equal” schedule primarily delays chaotic updates at initialization, but fails to respect the sequential dependency across layers; as a result, simultaneously scaling up all residuals may amplify representation and gradient noise. The “reverse” schedule, which activates deeper layers prior to shallow ones, becomes increasingly detrimental as warmup length increases. A longer warmup leads to a clearer separation of learning phases across layers, allowing deeper layers to dominate optimization early in training. Once this imbalance is established, it becomes difficult to correct by reactivating shallow layers later. This behavior mirrors a typical divergence pattern observed in Post-LN Transformers (Xiong et al., 2020; Wang et al., 2024), where the last few layers dominate model update, leaving shallow layers with vanishing gradients and limited signal to escape poor local minima.

3We note that BranchNorm (Liu et al., 2024) can be viewed as a special case of Post-LN with an “equal” ProRes schedule.

Table 6. Ablation of ProRes warmup schedules. Perplexity (↓) on C4 for 350M models trained with 6B tokens. “Warmup” denotes the number of training steps required for all layers to complete residual warmup. Blue indicates improvement while red indicates degrading performance relative to no-ProRes baseline. Perplexities higher than 1000 are marked as diverged. Additional baselines: DeepNorm (15.71) and LNS (14.63), trained under the same configuration without ProRes.

Schedule Warmup Pre-LN Post-LN Sandwich-LN

– – 15.21 16.83 14.76 linear 1k 14.60 15.98 14.63 linear 3k 14.49 15.65 14.56 linear 6k 14.44 15.33 14.51 linear 12k 14.41 15.15 14.45 linear 24k 14.43 15.12 14.46 linear 48k 14.57 15.18 14.44 linear-sqrt 24k 14.64 15.44 14.59 linear-square 24k 14.46 14.84 14.47 equal 1k 14.79 16.29 14.76 equal 12k 15.06 diverged 15.12 equal 24k 15.29 diverged 15.14 reverse 1k 14.98 diverged 14.96 reverse 12k 15.57 diverged 15.79 reverse 24k 15.65 diverged 16.78 stagewise-0 24k 19.51 diverged 18.31 stagewise-L 24k 15.35 15.09 14.81 stagewise-√

l 24k 14.80 15.48 14.71

fix-L – 15.77 20.18 15.47 fix-√

L – 15.48 17.04 14.99 fix-√

l – 14.98 16.21 14.82

Static vs. dynamic constraint on model update. In Principle 2 of Section 3.2, we hypothesized that techniques designed to bound model updates at initialization may be unnecessarily conservative once training enters a stable regime. We empirically verify this by observing that the “fix” schedules always perform worse than their “stagewise” counterparts (e.g. “fix-√

l” and “stagewise-√

l”), especially for Post-LN. The “fix” schedules constrain model update by scaling down the residual outputs according to model depth throughout the entire training process. “stagewise” variants apply an equivalent constraint at initialization, but progressively relax it by restoring residual magnitudes from shallow to deep layers during training. This dynamic treatment stabilizes the warmup phase comparably to “fix” schedules, without hurting learning potential in the stable training phase. Moreover, DeepNorm provides a more principled static approach for controlling model updates of Post-LN, yet still underperforms several ProRes schedules, further suggesting that static methods might fail to fully utilize the learning potential in the stable training phase.

Identifying the optimal ProRes schedule. The “linear” schedule is the most robust overall, achieving strong per-

formance across all three architectures and a wide range of warmup lengths, and is the best-performing evaluated schedule for Pre-LN and Sandwich-LN. For Post-LN, “linearsquare” and “stagewise-L” perform notably better than “linear”, highlighting the importance of a sufficiently gentle introduction of residuals during early training, which corresponds to Principle 1 in Section 3.2. Overall, the optimal schedule is architecture-dependent, with “linear” serving as a strong default choice.

#### 4.4. Depth Scaling Experiment

Loss Spike, Pre-LN

- 0.0

- 0.1

- 0.2

- 0.3

- 0.4

Loss Spike, Pre-LN (ProRes)

Grad Spike, Pre-LN

Grad Spike, Pre-LN (ProRes)

SpikeScore

24 48 72 96 120 Number of Layers

Figure 2. Spike score (↓) of loss and gradient norm as model depth increases.

We investigate how ProRes facilitates residual learning as model depth increases, by comparing its depth scaling behavior against several baseline methods. Results are summarized in Figure 1.

Setup. We start from a base configuration of a 71Mparameter Transformer with 12 layers and scale directly to 24, 48, 72, 96, and 120 layers, keeping all other dimensions fixed. For visual clarity in Figure 1, ProRes is applied only to Pre-LN, the best-performing combination according to Section 4.2, while studies across other normalization schemes are left to future work. Two ProRes schedules are evaluated: “linear” (Pre-LN (ProRes)) and “equal” (Pre-LN (ProRes-equal)). The same set of noProRes baselines from Section 4.1 is adopted, excluding Post-LN. All models are trained under the same setting as Section 4.1, with a learning rate of 2 × 10−3 for all depths except DeepNorm, which required additional search from {5×10−4,1×10−3,1.5×10−3,2×10−3} for optimal convergence. For Pre-LN (ProRes), we use the “linear” schedule with T = 1000 at depths up to 72, and T = 500 for 96 and 120 layers to ensure the warmup length does not exceed

the total training steps. No additional tuning is performed for the “linear” schedule. For Pre-LN (ProRes-equal), we found that T = 1000 leads to degraded performance at 120 layers, and instead used T = 12k for all depths.

Depth scaling performance. Figure 1 shows that Pre-LN (ProRes) consistently delivers the best performance among all methods, with the lead becoming more pronounced as depth increases. LNS matches ProRes at depths up to 48 layers, but falls short beyond 72 layers. We hypothesize that the static scaling of LNS, while controlling activation growth at initialization, overly downscales deeper residuals as model depth increases, limiting their contribution and learning potential. This is also observed in the scaling trend for Pre-LN (DS-Init), which initializes deeper layer weights to smaller values, leading to slower perplexity reduction beyond 72 layers. Sandwich-LN enforces a constant norm on residual outputs and provides better scaling potential than LNS at larger depths. Depth-dependent initializations improve the performance of Pre-LN mildly. DeepNorm surpasses Pre-LN from 72 layers onwards but requires careful tuning of learning rates for convergence.

Training stability. To understand how ProRes affects training stability, we measure loss and gradient spikes using the spike score defined by OLMo et al. (2024), which quantifies the percentage of data points that are at least seven standard deviations away from a rolling average of the previous 1000 steps. We compute spike scores over the stable training phase (10%–90% of training progress). As shown in Figure 2, applying ProRes maintains near-zero loss spikes as depth increases, indicating that ProRes not only enhances performance but also stabilizes training.

### 5. Analysis

In this section, we analyze the training dynamics that ProRes introduces. First, we show that applying ProRes naturally mitigates the exponential activation growth often observed in Pre-LN. Second, we study the evolution of layerwise representations during training, and show that ProRes enables smoother and more stable representation updates.

#### 5.1. Activation Growth in Pre-LN

Prior works have shown that Pre-LN Transformers often exhibit exponential activation growth across depth, rather than a desired linear growth (Li et al., 2024; Sun et al., 2025), which can limit the effective contribution of deeper layers (Gromov et al., 2024; Men et al., 2025).

We demonstrate that applying ProRes naturally resolves this issue. Figure 3 shows the layerwise activation norms during training for 1.3B models in Section 4.2. In early training (0–20k steps), activation norms of vanilla Pre-LN quickly

2500

[Figure 1]

[Figure 2]

2000

20

20

2000

ActivationNorm

ActivationNorm

16

16

1500

LayerIndex

LayerIndex

1500

12

12

1000

1000

8

8

500

500

4

4

0

0

0

0

0 20k 40k 60k 80k 100k Training Step

0 20k 40k 60k 80k 100k Training Step

(a) Pre-LN without ProRes

(b) Pre-LN with ProRes Figure 3. Layerwise activation norm of Pre-LN, with and without ProRes.

1.0

1.0

[Figure 3]

[Figure 4]

20

20

0.8

0.8

CosineSimilarity

CosineSimilarity

16

16

LayerIndex

LayerIndex

- 0.6

0.6

12

12

0.4

0.4

8

8

0.2

0.2

4

4

0.0

0.0

0

0

0 2000 4000 6000 8000 10000 Training Step

0 2000 4000 6000 8000 10000 Training Step

(b) Pre-LN with ProRes Figure 4. Cosine similarity of layerwise residuals between intermediate (first 10% steps) and final checkpoint.

(a) Pre-LN without ProRes

grows exponentially, while the ProRes variant facilitates a gentle and more linear growth. We attribute this behavior to differences in how residual updates are coordinated across depth. In vanilla Pre-LN, all layers update representations concurrently from initialization, even when upstream representations are still highly unstable. As a result, deeper layers may apply updates based on rapidly changing inputs, while shallow layers are simultaneously influenced by fluctuating gradient signals from deeper layers. By contrast, ProRes delays the contribution of deeper residual branches during early training, allowing shallow layers to first stabilize representations before deeper layers begin refining them. This staged interaction leads to more controlled activation growth across depth.

#### 5.2. Evolution of Layerwise Representations

To further examine how ProRes affects training dynamics, we analyze the evolution of layerwise residual outputs over the course of training. Specifically, we measure the cosine similarity between residual outputs at intermediate checkpoints and those of the final fully trained model. Figure 4 shows that both vanilla Pre-LN and ProRes exhibit earlier convergence in shallow layers. However, the representation evolution under ProRes is notably smoother across layers.

In contrast, vanilla Pre-LN shows frequent fluctuations in representation similarity throughout training, indicating less efficient evolution of residual outputs. These observations empirically support our hypothesis that explicitly coordinating residual contributions across depth reduces counterproductive updates and leads to more stable representation learning.

### 6. Conclusions

We introduced ProRes, a simple and scalable residual warmup scheme that explicitly coordinates layerwise learning over the course of Transformer training. By progressively activating residual contributions across depth, ProRes prioritizes shallow layers early while allowing deeper layers to engage once upstream representations stabilize. Extensive pretraining experiments show that ProRes consistently improves performance and depth scaling across model sizes, initialization methods, and normalization schemes. These results suggest that training-phase-aware residual scheduling is an effective and practical direction for improving Transformer optimization.

### Acknowledgements

This work was partially supported by an Area of Excellence project (AoE/E-601/24-N), a Theme-based Research Project (T32-615/24-R) and the Innovation and Technology Commission (ITCPD/17-9) from the Research Grants Council of the Hong Kong Special Administrative Region, China.

### Impact Statement

This paper presents a method for improving the optimization and depth scaling of Transformer-based language models. The primary goal of this work is to advance the understanding and design of scalable training techniques in machine learning. While improved training stability and efficiency may facilitate the development of larger and more capable models, the broader societal impacts are consistent with those of existing large language models. We do not identify any new ethical concerns or societal risks that are unique to the techniques proposed in this work.

### References

Ba, J. L., Kiros, J. R., and Hinton, G. E. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

Bachlechner, T., Majumder, B. P., Mao, H., Cottrell, G., and McAuley, J. Rezero is all you need: Fast convergence at large depth. In Uncertainty in Artificial Intelligence, pp. 1352–1361. PMLR, 2021.

Bisk, Y., Zellers, R., Gao, J., Choi, Y., et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 7432–7439, 2020.

Chen, T., Xu, X., Liu, Z., Li, P., Song, X., Jaiswal, A. K., Zhang, F., Hu, J., Wang, Y., Chen, H., et al. Gpas: Accelerating convergence of llm pretraining via gradient-preserving activation scaling. arXiv preprint arXiv:2506.22049, 2025.

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and Tafjord, O. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

De, S. and Smith, S. Batch normalization biases residual blocks towards the identity function in deep networks. Advances in Neural Information Processing Systems, 33: 19964–19975, 2020.

Diao, S., Yang, Y., Fu, Y., Dong, X., Su, D., Kliegl, M., Chen, Z., Belcak, P., Suhara, Y., Yin, H., et al. Climb: Clustering-based iterative data mixture bootstrapping for language model pre-training. arXiv preprint arXiv:2504.13161, 2025.

Ding, M., Yang, Z., Hong, W., Zheng, W., Zhou, C., Yin, D., Lin, J., Zou, X., Shao, Z., Yang, H., et al. Cogview: Mastering text-to-image generation via transformers. Advances in neural information processing systems, 34: 19822–19835, 2021.

Dodge, J., Sap, M., Marasovi´c, A., Agnew, W., Ilharco, G., Groeneveld, D., Mitchell, M., and Gardner, M. Documenting large webtext corpora: A case study on the colossal clean crawled corpus. arXiv preprint arXiv:2104.08758, 2021.

Dremov, A., H¨agele, A., Kosson, A., and Jaggi, M. Training dynamics of the cooldown stage in warmup-stable-decay learning rate scheduler. arXiv preprint arXiv:2508.01483, 2025.

Erdogan, G., Parthasarathy, N., Ionescu, C., Hudson, D. A., Lerchner, A., Zisserman, A., Sajjadi, M. S., and Carreira, J. Layerlock: Non-collapsing representation learning with progressive freezing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 19461– 19470, 2025.

Gao, L., Tow, J., Abbasi, B., Biderman, S., Black, S., DiPofi, A., Foster, C., Golding, L., Hsu, J., Le Noac’h, A., Li, H., McDonell, K., Muennighoff, N., Ociepa, C., Phang, J., Reynolds, L., Schoelkopf, H., Skowron, A., Sutawika, L., Tang, E., Thite, A., Wang, B., Wang, K., and Zou, A. The language model evaluation harness, 07 2024. URL https://zenodo.org/records/12608602.

Gong, L., He, D., Li, Z., Qin, T., Wang, L., and Liu, T. Efficient training of bert by progressively stacking. In International conference on machine learning, pp. 2337– 2346. PMLR, 2019.

Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Gromov, A., Tirumala, K., Shapourian, H., Glorioso, P., and Roberts, D. A. The unreasonable ineffectiveness of the deeper layers. arXiv preprint arXiv:2403.17887, 2024.

Gururangan, S., Wortsman, M., Gadre, S. Y., Dave, A., Kilian, M., Shi, W., Mercat, J., Smyrnis, G., Ilharco, G., Jordan, M., Heckel, R., Dimakis, A., Farhadi, A., Shankar, V., and Schmidt, L. open lm: a minimal but performative language modeling (lm) repository, 2023. URL https://github.com/ mlfoundations/open_lm/. GitHub repository.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

Henry, A., Dachapally, P. R., Pawar, S. S., and Chen, Y. Query-key normalization for transformers. In Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 4246–4253, 2020.

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., Casas, D. d. L., Hendricks, L. A., Welbl, J., Clark, A., et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

Hu, S., Tu, Y., Han, X., He, C., Cui, G., Long, X., Zheng, Z., Fang, Y., Huang, Y., Zhao, W., et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395, 2024.

Huang, X. S., Perez, F., Ba, J., and Volkovs, M. Improving transformer optimization through better initialization. In International Conference on Machine Learning, pp. 4475–

4483. PMLR, 2020. Kingma, D. P. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Kosson, A., Messmer, B., and Jaggi, M. Analyzing & reducing the need for learning rate warmup in gpt training. Advances in Neural Information Processing Systems, 37: 2914–2942, 2024.

Lai, G., Xie, Q., Liu, H., Yang, Y., and Hovy, E. Race: Large-scale reading comprehension dataset from examinations. arXiv preprint arXiv:1704.04683, 2017.

Li, M. Z., Agrawal, K. K., Ghosh, A., Teru, K. K., Santoro, A., Lajoie, G., and Richards, B. A. Tracing the representation geometry of language models from pretraining to post-training. arXiv preprint arXiv:2509.23024, 2025.

Li, P., Yin, L., and Liu, S. Mix-ln: Unleashing the power of deeper layers by combining pre-ln and post-ln. arXiv preprint arXiv:2412.13795, 2024.

Li, T., Han, D., Cao, Z., Huang, H., Zhou, M., Chen, M., Zhao, E., Jiang, X., Jiang, G., and Huang, G. Siamesenorm: Breaking the barrier to reconciling pre/postnorm. arXiv preprint arXiv:2602.08064, 2026.

Liu, L., Liu, X., Gao, J., Chen, W., and Han, J. Understanding the difficulty of training transformers. arXiv preprint arXiv:2004.08249, 2020.

Liu, Y., Zeng, X., Meng, F., and Zhou, J. Branchnorm: Robustly scaling extremely deep transformers. In Findings

of the Association for Computational Linguistics: ACL 2024, pp. 11675–11687, 2024.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Men, X., Xu, M., Zhang, Q., Yuan, Q., Wang, B., Lin, H., Lu, Y., Han, X., and Chen, W. Shortgpt: Layers in large language models are more redundant than you expect. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 20192–20204, 2025.

Merity, S., Xiong, C., Bradbury, J., and Socher, R. Pointer sentinel mixture models. arXiv preprint arXiv:1609.07843, 2016.

Mihaylov, T., Clark, P., Khot, T., and Sabharwal, A. Can a suit of armor conduct electricity? a new dataset for open book question answering. arXiv preprint arXiv:1809.02789, 2018.

OLMo, T., Walsh, P., Soldaini, L., Groeneveld, D., Lo, K., Arora, S., Bhagia, A., Gu, Y., Huang, S., Jordan, M., et al. 2 olmo 2 furious. arXiv preprint arXiv:2501.00656, 2024.

Paperno, D., Kruszewski, G., Lazaridou, A., Pham, N.-Q., Bernardi, R., Pezzelle, S., Baroni, M., Boleda, G., and Fern´andez, R. The lambada dataset: Word prediction requiring a broad discourse context. In Proceedings of the 54th annual meeting of the association for computational linguistics (volume 1: Long papers), pp. 1525– 1534, 2016.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21 (140):1–67, 2020.

Sakaguchi, K., Bras, R. L., Bhagavatula, C., and Choi, Y. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106, 2021.

Sap, M., Rashkin, H., Chen, D., LeBras, R., and Choi, Y. Socialiqa: Commonsense reasoning about social interactions. arXiv preprint arXiv:1904.09728, 2019.

Shazeer, N. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020.

Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., and Catanzaro, B. Megatron-lm: Training multibillion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019.

Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Sun, W., Song, X., Li, P., Yin, L., Zheng, Y., and Liu, S. The curse of depth in large language models. arXiv preprint arXiv:2502.05795, 2025.

Team, K., Bai, Y., Bao, Y., Chen, G., Chen, J., Chen, N., Chen, R., Chen, Y., Chen, Y., Chen, Y., et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.

Touvron, H., Cord, M., Sablayrolles, A., Synnaeve, G., and J´egou, H. Going deeper with image transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 32–42, 2021.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Wang, H., Ma, S., Dong, L., Huang, S., Zhang, D., and Wei, F. Deepnet: Scaling transformers to 1,000 layers. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(10):6761–6774, 2024.

Wang, Q., Li, B., Xiao, T., Zhu, J., Li, C., Wong, D. F., and Chao, L. S. Learning deep transformer models for machine translation. arXiv preprint arXiv:1906.01787, 2019.

Wen, K., Li, Z., Wang, J., Hall, D., Liang, P., and Ma, T. Understanding warmup-stable-decay learning rates: A river valley loss landscape perspective. arXiv preprint arXiv:2410.05192, 2024.

Xie, S., Zhang, H., Guo, J., Tan, X., Bian, J., Awadalla, H. H., Menezes, A., Qin, T., and Yan, R. Residual: Transformer with dual residual connections. arXiv preprint arXiv:2304.14802, 2023.

Xie, Z., Wei, Y., Cao, H., Zhao, C., Deng, C., Li, J., Dai, D., Gao, H., Chang, J., Yu, K., et al. mhc: Manifold-constrained hyper-connections. arXiv preprint arXiv:2512.24880, 2025.

Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L., and Liu, T. On layer normalization in the transformer architecture. In International conference on machine learning, pp. 10524–10533. PMLR, 2020.

Yang, C., Wang, S., Yang, C., Li, Y., He, R., and Zhang, J. Progressively stacking 2.0: A multi-stage layerwise training method for bert training speedup. arXiv preprint arXiv:2011.13635, 2020.

Yang, G., Hu, E. J., Babuschkin, I., Sidor, S., Liu, X., Farhi, D., Ryder, N., Pachocki, J., Chen, W., and Gao, J. Tensor programs v: Tuning large neural networks via zero-shot hyperparameter transfer. arXiv preprint arXiv:2203.03466, 2022.

Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.

Zhang, B. and Sennrich, R. Root mean square layer normalization. Advances in neural information processing systems, 32, 2019.

Zhang, B., Titov, I., and Sennrich, R. Improving deep transformer with depth-scaled initialization and merged attention. arXiv preprint arXiv:1908.11365, 2019a.

Zhang, H., Dauphin, Y. N., and Ma, T. Fixup initialization: Residual learning without normalization. arXiv preprint arXiv:1901.09321, 2019b.

Zhu, D., Huang, H., Huang, Z., Zeng, Y., Mao, Y., Wu, B., Min, Q., and Zhou, X. Hyper-connections. arXiv preprint arXiv:2409.19606, 2024.

Table 7. Perplexity (↓) on ClimbMix test split across model scales. Method ProRes 130M 350M 1.3B Pre-LN

✘ 11.11 9.43 8.38 ✔ 10.82 9.13 8.09

### A. Learning Rate Tuning

In this section, we detail the learning rate tuning for the main experiments in Section 4.1. We first note that the final performance of the WSD learning rate scheduler (Hu et al., 2024) is not quite sensitive to learning rate choices. A smaller learning rate tends to have a lower loss during the constant learning rate phase, but a larger learning rate would catch up after the decay stage, as long as it trains stably. A similar phenomenon is discussed in (OLMo et al., 2024). With this property in mind, our goal is to find a preferably larger learning rate such that training takes place smoothly, without notable loss spikes or gradient spikes. We only tune the learning rates for no-ProRes baselines, and directly apply their tuned learning rates to ProRes variants.

For the 1.3B models, we searched over learning rates {1 × 10−3,6 × 10−4,3 × 10−4}. We selected 6 × 10−4 for all baselines except Post-LN and DeepNorm, for which 3 × 10−4 yielded more stable training.

For the 350M models, we evaluated {5×10−4,1×10−3,1.5×10−3} and adopted 1×10−3 for all models except Post-LN, which performed best with 5 × 10−4.

For the 130M models, we considered {1 × 10−3,1.5 × 10−3,2 × 10−3}. The final choices were 1 × 10−3 for Post-LN, 1.5 × 10−3 for DeepNorm, and 2 × 10−3 for the remaining baselines.

### B. Pretrain Experiments on 7B parameters

0.02

0.02

AbsoluteLossGap

AbsoluteLossGap

0.00

0.00

0.02

0.02

Pre-LN

Pre-LN

Pre-LN (ProRes)

Pre-LN (ProRes)

Finish Residual Warmup LR Decay

Finish Residual Warmup LR Decay

0.04

0.04

0 20k 40k 60k 80k 100k Training Step

0 20k 40k 60k 80k 100k Training Step

Figure 5. Loss gap across training steps of 7B models. Left: training loss gap (0.99 EMA smoothed) between Pre-LN and its ProRes variant. Right: evaluation loss gap calculated every 1000 steps.

We conduct pretraining experiments on 7B parameter models following the settings in Section 4.1, using a learning rate of 3 × 10−4. Vanilla Pre-LN and Pre-LN with ProRes are evaluated. For the ProRes variant, we adopt the linear schedule in Equation (3) with T = 1000, which corresponds to a residual warmup length of 32,000 steps given 32 layers in total.

- Figure 5 shows the loss gaps during training. We note that ProRes has higher loss initially due to its restricted model updates when α(l,t) is small. As training proceeds, ProRes gradually unlocks learning potentials of each layer and achieves lower loss than the baseline. The loss gap keeps increasing steadily even after all residuals have finished warming up. Moreover, ProRes is able to maintain and even slightly increase its advantage after the learning rate decay stage.

### C. Pretrain Experiments on Alternative Corpus

To verify that ProRes generalizes well to pretraining corpus other than C4, we conduct a set of experiments on the ClimbMix dataset (Diao et al., 2025). We adopt the same preprocessing pipeline in Section 4.1. Evaluation perplexities on held out test set are reported in Table 7.

### D. Visualization of ProRes Schedules

- 0

- 1

l

- 0

- 1

(l,t)

l

0

L

- 0

- 1

0 4000 8000 12000

0 4000 8000 12000

0 4000 8000 12000

t

- Figure 6. Visualization of various schedules in Table 5. We use T = 1000, L = 12 for visualizing α(l, t). Darker lines indicate shallow layers while brighter lines indicate deeper layers.

### E. Use of Compute Resources

All experiments were conducted on a single node of 8×H800 GPUs. Training under the settings in Section 4.1 required approximately 260h for 7B, 100h for 3B, 50h for 1.3B, 20h for 350M, and 7h for 130M models.

