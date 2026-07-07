# arXiv:2601.04890v1[cs.LG]8Jan2026

[Figure 1]

2026-01-09

## Learnable Multipliers: Freeing the Scale of Language Model Matrix Layers

Maksim Velikanov*, Ilyas Chahed*, Jingwei Zuo, Dhia Eddine Rhaiem, Younes Belkada, Hakim Hacid

[Figure 2]

#### Falcon LLM Team

https://huggingface.co/tiiuae

###### Abstract

Applying weight decay (WD) to matrix layers is standard practice in large-language-model pretraining. Prior work suggests that stochastic gradient noise induces a Brownian-like expansion of the weight matrices W, whose growth is counteracted by WD, leading to a WD-noise equilibrium with a certain weight norm ∥W∥. In this work, we view the equilibrium norm as a harmful artifact of the training procedure, and address it by introducing learnable multipliers to learn the optimal scale. First, we attach a learnable scalar multiplier to W and confirm that the WD–noise equilibrium norm is suboptimal: the learned scale adapts to data and improves performance. We then argue that individual row and column norms are similarly constrained, and free their scale by introducing learnable per-row and per-column multipliers. Our method can be viewed as a learnable, more expressive generalization of µP multipliers. It outperforms a well-tuned µP baseline, reduces the computational overhead of multiplier tuning, and surfaces practical questions such as forward-pass symmetries and the width-scaling of the learned multipliers. Finally, we validate learnable multipliers with both Adam and Muon optimizers, where it shows improvement in downstream evaluations matching the improvement of the switching from Adam to Muon.

## 1. Introduction

Pretraining large-scale language models presents significant challenges for both the optimization algorithm and the choice of hyperparameters. The most widely used and reliable optimizer is Adam (Kingma & Ba, 2015), or, rather, its weight decay version AdamW (Loshchilov & Hutter, 2019) that incorporates decay directly in the parameter update: θt ← θt − ηλθt, where η is the learning rate (LR) and λ the weight decay hyperparameter. The WD term of AdamW is critical for both improving model performance and stabilizing training at large scale (Devlin et al., 2019; Brown et al., 2020). Recent alternatives to Adam, such as Muon (Jordan, 2024), similarly rely on the explicit WD term in its parameter update to maintain stability and performance (Liu et al., 2025).

The practical ubiquity and necessity of WD term motivates a deeper investigation into its effects on training dynamics that might reveal the reasons behind its success, or potential shortcomings. Prior work points to a wide range of WD effects on the training, from improving the bias-variance tradeoff (D' Angelo et al., 2024) to imposing certain structures in the model weights (Kobayashi et al., 2024). Central to this work is the stochastic gradient noise perspective on weight decay (Kosson et al., 2024; Zuo et al., 2025). From this viewpoint, the gradient noise induces a Brownianlike component to the optimizer updates that may cause uncontrollable growth of model weights if left unchecked. Weight decay counteracts this Brownian expansion, resulting in a noise-WD equilibrium. In the equilibrium, the norm of the model weights W scales predictably with the

∗Equal contribution.

learning rate η and WD λ, as can be both observed empirically and derived from toy models (Kosson et al., 2024; Zuo et al., 2025):

η λ

∥W(η,λ)∥ ∝ S(η,λ) =

. (1)

Our work builds on a straightforward interpretation of the equilibrium norm scaling (1): the weight norms are dictated by the optimization hyperparameters instead of being learned from the data. In other words, WD traps model weights in the noise-driven equilibrium, preventing them from learning the scale suitable for a given training data. To escape the noise-WD equilibrium, we propose to re-parametrize the model weights with Learnable Multipliers (LRM), introduced in sec. 2. For example, in the scalar case, W → sW with the learnable scale s ∈ R. We expect learnable multipliers to not experience the same noise-WD problem by looking at the modern LLM practices: WD is not applied to scalar and vector-like weights, e.g. RMSNorm weights, without stability or performance issues caused by removing WD from matrix layers.

Overview of the results. We first introduce scalar and vector learnable multipliers in section 2. We provide a general discussion of their placement within language model architectures, their connection to maximal update parametrization (µP), and the instances of learnable multipliers in prior work. We then investigate LRMs in the context of language model pretraining.

- • In section 3, we confirm that features learned by matrix layers alone are limited by noiseWD equilibrium, while adding learnable multipliers results in richer representations. First, we design a series of experiments to show that the matrix layers fail to adapt their scale when it is required for optimal loss minimization, while adding multipliers recovers this lost performance. Then, we demonstrate that multipliers allow for a more diverse scale distribution across residual blocks. Likewise, vector LRMs enable a more diverse scale distribution for internal features within each model block.
- • In section 4, we point to various aspects essential for stable and effective application of learnable multipliers to model pretraining. This includes handling the reparameterization symmetries of a given model architecture, exploring width µP scaling, and other details such as addressing interaction of multipliers with gradient clipping.
- • In section 5, we validate the performance in a longer end-to-end pretraining run. Learnable multipliers maintain an increasing performance gap over the baseline throughout the whole training, supporting the earlier conclusion that multipliers enable richer model representation. Finally, we explore the role of multiplier initialization using tuned values of µP multipliers from (Zuo et al., 2025). LRMs maintain the same level of performance regardless of whether the tuned values of forward and WD multipliers are used, while having tuned learning rate multipliers is still important for optimal performance.
- • Learnable multipliers act on weight matrices in an isolated manner, suggesting a native application to a wide range of architectures and optimizers. We illustrate this applicability by doing experiments on a hybrid attention-SSM architecture and applying LRMs to structurally distinct attention, SSM, and MLP residual blocks. To illustrate optimizer applicability, we perform some of the experiments for Adam and Muon, showing similar performance gains and behavior patterns in both cases.

Notations. We use relative mean square convention for the norms of matrices and vectors. For example, the norm of a matrix W ∈ Rn×m is computed as ∥W∥ = nm1 ni=1 mj=1 Wij2.

## 2. Learnable Multipliers

Consider a linear layer yi = j Wijxj, where xj ∈ Rdin and yi ∈ Rdout are the input and output features, Wij ∈ Rdin×dout is the feature map matrix, and we have used index notation for vectors and matrices. The weight reparametrization amounts to using another matrix Wij and possibly additional weights to be learned by the optimization algorithm instead of the effective weight matrix Wij. In this work, we reparametrize Wij with either scalar multiplier s or vector multipliers ri,cj.

To escape the noise-WD equilibrium value (1) of the feature map matrix Wij, it is sufficient to add

Scalar Multiplier : Wij = sWij, s ∈ R. (2) Here, the learnable matrix weight Wij is still subject to the noise-WD equilibrium with the norm ∥W∥ ∝ λη. The scalar multiplier s is supposed to learn freely so that the full matrix norm ∥W∥ = s∥W∥ optimally adapts to a given data distribution.

We make a step further and hypothesize that not only the norm of the whole matrix ∥W∥, but also the norms of its individual rows ∥Wi•∥ and columns ∥W•j∥ might also be stuck in the noise-WD equilibrium. Hence, we attach a learnable scale parameter to each row and column with

Vector Multipliers : Wij = riWijcj, ri ∈ Rdout, cj ∈ Rdin. (3)

As for the scalar case, ∥W∥ is expected to have the equilibrium value, while each component of the learnable row ri and column cj multipliers is supposed to learn the respective optimal scale.

It is instructive to relate the gradients of the reparametrized matrix Wij and the introduced multipliers s,ri,cj to the gradients of the effective matrix Gij = ∂W∂L

, where L is the training loss. Direct application of the chain rule gives

ij

∂L ∂Wij

= ricjGij,

∂L ∂ri

=

j

WijcjGij,

∂L ∂cj

=

i

riWijGij,

∂L ∂s

=

ij

WijGij. (4)

We see that row/column multipliers accumulate the gradients across the respective column/row of the gradient matrix Gij, while scalar multiplier accumulates the gradients across the whole matrix. This extra averaging reduces the gradient noise level in the multipliers and intuitively explains why they do not experience noise-driven Brownian expansion that needs to be countered by weight decay.

In fact, learnable multipliers are partially used in the nowadays standard Pre-LN architectures (Xiong et al., 2020) through RMSNorm learnable weights that can be viewed as column multipliers cj of the first linear layer in the block. While RMSNorm weights already provide scale adaptation to a part of the model, we argue that adding the multipliers to the remaining parts of the model yields further performance improvement.

Finally, we note a natural idea of using a logarithmic scale for the learnable multipliers, for example, s → es in the scalar case. Such reparameterization can be beneficial when different multipliers tend to have both large and small scales, making it problematic to learn with uniformly scaled optimizer updates given by the multipliers’ learning rate. However, in agreement with (Salimans & Kingma, 2016), we observed that log-scale parametrization gives at most a slight performance advantage while posing stability issues we discuss in more detail in sec. 4.1.

Model placement. While vector multipliers (3) are strictly more expressive than scalars (2), using both row ri and column cj multipliers for all the matrix layers is clearly redundant. We previously mentioned that the column multiplier cj of the first linear layer in a block is equivalent to the RMSNorm weights of that block: using both is one example of such redundancy. Another example of redundancy is using both row multipliers for MLP up projection and column multipliers for down projection. As we explain in sec.4.1, redundant multipliers give rise to a symmetry transformation in the model parameters that may lead to NaN values during training. We provide our recommended placement of multipliers for gated MLP, attention, and mamba2 (Dao & Gu,

- 2024) blocks in sec. C.

Implementation. For inference, learnable multipliers can be merged with their matrix Wij into the effective matrix Wij, and thus do not introduce any memory or latency overhead.

During training, however, there are two distinct implementation strategies with different impacts on the training throughput. A simpler approach is to explicitly use the reparametrized expression (2),(3) in the model forward pass, relying on the standard automatic differentiation and optimizer implementations to handle the update of the multipliers. In this case, the throughput is expected to drop at most by a couple of percent since the multiplier parameter count is tiny compared to the matrix layers. Yet, this drop can be further reduced by using effective matrices Wij in the model’s forward and backward pass, while manually handling the dynamics of multipliers and learnable matrix Wij on the optimizer level with the help of gradient relations (4).

Maximal update parametrization. (Yang & Hu, 2021; Yang et al., 2022, 2024b; Dey et al.,

- 2025) also use scalar reparameterization (2) but with non-learnable scale s equipped with scaling rules w.r.t. model dimensions for predictable model size scaling. To maximize the performance, many pretrained language models additionally tune the multipliers on a smaller scale, and then transfer tuned multipliers to the target model scale (Dey et al., 2023; Hu et al., 2024; Zuo et al., 2025).

Our scalar multipliers (2) can be viewed as learnable version of µP multipliers, significantly affecting established µP workflows such as hyperparameter transfer. On the one hand, learnable µP multipliers no longer allow to enforcing µP scaling with model dimensions, and thus require a separate analysis of the model size scaling behavior. On the other hand, learnable multipliers reduce the need to perform compute-intensive tuning of the multipliers. As an example, (Zuo et al., 2025) performed extensive tuning of 12 forward (weight reparametrization), 16 learning rate, and 7 weight decay multipliers. Our approach removes the need to tune both forward and weight decay multipliers that were responsible for the scale of model weights.

Learnable multipliers in the literature. Reparametrization of model weights with learnable multipliers was previously proposed in various deep learning contexts and for various reasons. Weight normalization (Salimans & Kingma, 2016) introduces multipliers as a replacement for batch normalization. Several works add learnable multipliers to residuals to enable stable training of very deep models (Bachlechner et al., 2021; De & Smith, 2020; Zhang et al., 2019; Huang et al., 2020; Nishida et al., 2024). For transformer models, multipliers are used within parameter-efficient finetuning methods to increase expressivity (Liu et al., 2022, 2024; Wang et al., 2024), or together with extra normalization layers to address gradients mismatch along depth. Our perspective of using the multipliers to address the noise-WD equilibrium explains the mechanism by which they improve the performance and guides towards their comprehensive placement throughout the model architecture.

Projector experiment: loss after LR decay

Logits norm

Projector matrix and multipliers norms

- 100
- 101
- 102

20.0

0.80

| |Vector Projector Norm (VPN) Scalar Projector Norm (SPN) Frozen Projector Norm (FPN)<br><br>|
|---|---|
| | |
| | |
| | |
| | |
| | |

| |VPN SPN FPN<br><br>|
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>VPN matrix<br><br>VPN multiplier<br><br>SPN matrix<br><br>SPN multiplier<br><br>FPN matrix Matrix ABM Multiplier ABM<br><br>|
|---|

17.5

0.79

15.0

0.78

12.5

10.0

0.77

10 1

7.5

0.76

5.0

10 2

2.5

0.75

10 3

0.0

8 2 8 1 80 81 82 Projector norm scale S( P, P)

8 2 8 1 80 81 82 Projector norm scale S( P, P)

8 2 8 1 80 81 82 Projector norm scale S( P, P)

VPN with strong weight decay: S( P, P) = 1/64

VPN with weak weight decay: S( P, P) = 64

MLP experiment: loss after LR decay

0.040

40

20.0

2.00

| |Adam<br><br>Adam + MLP learnable scalars<br><br>Muon<br><br>Muon + MLP learnable scalars|
|---|---|
| | |
| | |
| | |
| | |
| | |

Matrix

Matrix

0.78

Experiment

Experiment

0.035

35

17.5

1.75

Adam Brownian Motion (ABM)

Adam Brownian Motion

0.030

30

15.0

1.50

0.77

Multipliernorm

0.025

25

12.5

1.25

Matrixnorm

0.020

20

10.0

1.00

0.76

Multiplier

Multiplier

0.015

15

7.5

0.75

Experiment

Experiment

0.75

Adam Brownian Motion

Adam Brownian Motion

0.010

10

5.0

0.50

0.005

5

2.5

0.25

0.74

0.000

0

0.0

0.00

2 2 2 1 20 21 22 MLP norm scale S( MLP, MLP)

0 5 10 15 20 25 30 GigaTokens

0 5 10 15 20 25 30 GigaTokens

- Figure 1: Projector and MLP scale (1) sweep experiments described in section 3. Norm scales

S(ηP,λP) and S(ηMLP,λMLP) use relative values of learning rate and weight decay. (Top left): the final loss of three projector norm configurations. (Top middle and right): trajectories of projector ∥W∥ and multiplier ∥c∥ norms during training are compared between the experiment and Adam Brownian Motion simulation. (Bottom left and middle): logits and matrix/multipliers norms for the considered configuration. (Bottom right): the final loss of three MLP experiment configurations.

## 3. What is learned by the multipliers?

So far, the inability of matrix layers to learn data-dependent scales has been hypothesized but not experimentally validated. In this section, we support this hypothesis by providing different views on feature scales learned by the model.

We start by examining whether matrix layers can natively escape noise-WD equilibrium under significant loss optimization pressure. Under the scaling assumption (1), varying the norm scale S(η,λ) = λη forces the model weights out of typical and presumably optimal scale. We perform two such tests for the LM head layer and projections of MLP block, with the results depicted in figure 1. In these experiments, we vary S(η,λ) while keeping the effective learning rate ηeff ≡

√ηλ = const Zuo et al. (2025) to isolate the effect of norm change from overall learning speed.

Projector experiment. Consider a standard final projector layer (LM head) Wij that maps normalized backbone features x,mean{x2j} = 1 to probability logits yi = j Wijcjxj, where cj are learnable RMSNorm weights. We compare the following configurations differentiated by their multiplier type.

- 1. Frozen projector norm (FPN): cj → 1. In this configuration, the logits scale is determined only by the scale of W and its correlation with features x.
- 2. Scalar projector norm (SPN): cj → s with learnable scalar multiplier s.
- 3. Vector projector norm (VPN): a standard configuration with freely learnable cj.

First, we observe in Figure 1 (top left) that the FPN configuration, where the logits scale relies entirely on ∥W∥, suffers a clear performance drop at extreme values of the equilibrium projector norm S(ηP,λP) = ηP

λP . In contrast, both the SPN and VPN configurations maintain stable performance. This performance gap can be traced to the logit norms (top middle), which remain stable (and presumably optimal) for the multiplier-equipped configurations, but vary significantly for FPN. Finally, we find that the norm ∥W∥ for all the configurations (top right) indeed follows the noise-WD equilibrium scaling (1), while scalar and vector multipliers are not restricted and adjust their scale to compensate for the large/small projector norm. The only exception to this equilibrium scaling is the FPN configuration at small equilibrium scales: the optimization pressure to maintain a non-vanishing logit norm is strong enough to pull the weights away from equilibrium.

To clearly identify the extent to which projector and multipliers follow a noise-driven dynamics, we compare their norms with Adam Brownian Motion (ABM). To simulate pure noise-determined trajectories, we generate a sequence of zero mean i.i.d. gradients gt ∼ N(0,1) and use them in the AdamW optimizer update with the same λ and schedule of η as was used in the training. On figure 1 (bottom left and middle), we observe that ABM and experimental trajectories of ∥W∥ fully coincide for strong WD and are quite close for weak WD, while experimental multipliers trajectories show no resemblance with their ABM version. The same applies to final norm values on figure 1 (top right).

MLP experiment. Next, we design a similar experiment for residual MLP blocks, where we vary the norm scale S(ηMLP,λMLP) only for the 3 matrix layers of the gated MLP block. To test the ability of MLP matrix layers to adapt their scale we consider two configurations:

- 1. Frozen RMSnorm weights across all backbone layers to restrict the scale adaption ability of all the backbone blocks.
- 2. Same as above, but with scalar learnable multipliers added to MLP block to compensate for varied norm scale S(ηMLP,λMLP) of its matrix layers.

The loss behavior of these configurations is depicted on figure 1 (bottom right), where we observe that configuration without learnable multipliers suffers from loss degradation at larger norm scales S(ηMLP,λMLP) of MLP matrices. We attribute this degradation to output’s magnitude mismatch between MLP and attention/SSM blocks, and investigate it in more details in section B. Importantly, we perform MLP experiment for both Adam and Muon optimizers and observe essentially identical behavior between the two optimizers. This suggest that noise-WD equilibrium trap is not a specific property of Adam but a general phenomenon persisting across different parameter update rules.

- 3.1 Features scale diversity

Having confirmed the ability of multipliers to learn the scale that was fixed in matrix-only architecture configurations, we proceed with an investigation of new features that are learned when the model scales are freed by adding the multipliers.

Depth-wise scales. We add scalar multipliers to all matrix layers in the model and measure the norms of residual blocks output across the model’s depth. The results are depicted in figure 2.

For block outputs we observe an overall increasing trend towards later layers, suggesting that larger contribution of these layers to the residual could be beneficial for loss optimization but could not be fully learned by classical architecture without multipliers. Moreover, we observe

LRM-to-baseline block output ratios

Output scalar multiplers

Internal scalar multipliers

- 100
- 101

- 0
- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8

0.6

| |MLP Attn SSM<br><br>| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

|MLP Attn SSM<br><br>| | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

| |MLP gate<br><br>Attn key<br><br>SSM dt| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

0.5

0.4

0.3

0.2

0.1

10 1

0.0

0 5 10 15 20 25 30 35 layer index

0 5 10 15 20 25 30 35 layer index

0 5 10 15 20 25 30 35 layer index

- Figure 2: (Left): a ratio of residual block output norms of the configuration with scalar multipliers to the configuration without. (Middle): values of scalar multipliers at the end of each block. (Right): values of internal scalar multipliers at selected locations of each block.

10 1 100 101 Rows distribution

0.00

0.05

0.10

0.15

0.20

probability

Attention Input Projection

Standard Parametrization Vector Multipliers

| |
|---|

10 1 100 101 Rows distribution

0.0

0.1

0.2

0.3

0.4

0.5

MLP Gate Projection

Standard Parametrization Vector Multipliers

| |
|---|

10 1 100 101 Rows distribution

0.0

0.1

0.2

0.3

0.4

0.5

0.6

0.7

SSM Input Projection

Standard Parametrization Vector Multipliers

| |
|---|

- Figure 3: Distributions of row norms ∥Wi•∥ of attention/SSM input and MLP gate projections, which correspond to internal features of these blocks. We collect the norms across all the model layers while normalizing norm values of each layer by their mean to align the scale of different layers and focus on within-layer distribution.

that attention layers in the second half of the model have significantly different scales that are further amplified by multipliers values (left and middle subfigures). Additionally, dt projection that controls memorization and forgetting in the SSM block show significant variation across the layers, suggesting specializing of those layers in varying temporal scales.

Width-wise scales. To motivate vector multipliers (3) in section 2, we assumed that row ∥Wi•∥ and column ∥W•j∥ norms are also subject to noise-WD equilibrium. Now, we confirm this assumption on figure 3 by measuring the distribution of row (output feature) norms of the effective layer matrices W for configurations with and without learnable vector multipliers.

We select attention/SSM input and MLP gate projections, whose outputs go into essential nonlinear transformations of those layers, and hence very sensitive to the scale. Yet, the configuration without multipliers show a very narrow distribution implying low scale diversity of internal features in these layers. Adding learnable vector multipliers significantly broadens the norm distribution, suggesting that a larger diversity of internal feature scales in these blocks is beneficial for the model.

Q/K multipliers symmetry

3.5

| || |
|---|
<br><br>No WD on Q/K multipliers Light WD on Q/K multipliers| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

3.0

2.5

log(||/||)MqMk10

2.0

1.5

1.0

0.5

0.0

2 1 0 1 2 3

log10 |MqMk|

(a)

Residual-Output symmetry

No WD on multipliers

Light WD on Attention output multpliers

6

MLP

Attention

4

logRMS(out)

2

0

2

Bands show IQR

0 5000 10000 15000 20000 25000

Training Steps

(b)

- Figure 4: Effect of light WD on (a) Q/K multiplier symmetry and (b) residual output symmetry. Light WD (orange) suppresses the drift and unbounded norm growth observed in the baseline without WD (blue).

## 4. Aspects of multiplier training dynamics

### 4.1 Symmetries

Unlike weight matrices which require weight decay to bound their scale, learnable multipliers seem to not require any scale control due to their ability to freely learn the optimal scale. However, the multipliers are vulnerable to a different source of scale instability: architecture symmetries. These symmetries represent scaling transformation in the parameter space of the model that does not change final model’s output (Dinh et al., 2017). While harmless in exact arithmetic, this drift introduces significant instability under low-precision formats like bfloat16, as it increases quantization error and degrades gradient estimates (Micikevicius et al., 2017; D' Angelo et al., 2024; Le Scao et al., 2022). We highlight two symmetries common in language model architectures:

- • Multiplicative symmetry. When two learnable factors a and b appear only through their product ab, the reparameterization (a,b)  → (sa,s−1b) leaves the forward map unchanged for any s ̸= 0. As a result, ∥a∥ can grow while ∥b∥ shrinks. An example of that is the product of queries Q and keys K in the attention computation. Figure 4a illustrates this: for the baseline without intervention (blue), the product of the Q/K multipliers (x-axis) remains stable while their scale ratio (y-axis) drifts significantly.
- • Normalization symmetry. Assume the model uses a quantity c only through its normalized

version cˆ := c/∥c∥rms (e.g., the activation before the final head projection is rescaled to unit RMS: Only the relative magnitudes of the residuals producing that activation matters). Then for any s > 0, the rescaling c  → sc leaves cˆ, and hence the model output unchanged. Consequently, the residuals norms can grow without bound while the final output remains identical. This unbounded growth is demonstrated in Figure 4b, where the RMS of the residual outputs (blue lines) increases over training steps when no weight decay is applied.

In standard architecture without learnable multipliers, these symmetries are essentially fixed by equilibrium norms of the weight matrices. When we first added learnable multipliers, we have

Scaling recipe LR η WD λ Multiplier s ∥y∥ ∥∥∆yy∥∥ Standard through LR d−1 const const d12 d−12 Through LR & WD d−1 d const const const Through multiplier const const d−1 const const

- Table 1: Scaling of the output norm ∥y∥ of a linear layer y = sWx, x ∈ Rd,s ∈ R, and relative

norm ∥∆y∥

∥y∥ of the update ∆y = ∆Wx with width d. The standard LR-only scaling recipe is taken from Yang et al. (2022); Dey et al. (2023); Hu et al. (2024), its adjustment to include WD is given in Kosson et al. (2025), and an alternative version that scales multipliers was used in Zuo et al. (2025).

a drift along symmetry directions, leading to large activations, and, eventually, divergence or NaN values.

There are several possible approaches to address the symmetry-induced instabilities. One may manually rescale weights along the symmetry directions back to “normal” values or remove all the multipliers subject to the symmetry transformations, with specific configuration provided in section C. The former requires extensive architecture-specific engineering, and we have found the latter to surprisingly reduce final model performance. We converged to applying a small weight decay λlrm = 2 × 10−3 to multipliers as a simple yet effective solution to handle the symmetries.

### 4.2 Scaling with model width

When the size of a model is scaled, a natural requirement is to keep constant the magnitude of model activations for stable and performant forward pass, while efficient feature learning requires keeping constant the magnitude of activation updates (see, for example, desideratum 1 of Yang et al. (2024a)). While µP satisfies these requirements with certain scaling rules of forward multipliers and/or learning rate, both the presence of equilibrium (1) and the addition of learnable multipliers prompt revisiting of µP scaling rules. In this section, we briefly explore the above questions via an experiment where we (i) use scalar learnable multipliers (2) (ii) vary model width d (iii) keep learning rate η and weight decay λ fixed across widths d. The results are depicted on figure 5.

Equilibrium matrix norms. First, we observe on figure 5 (left) that the norms of matrix layers stay almost constant across considered model widths. This suggests that the equilibrium norm does not scale with model size, for example, via increased noise level at larger sizes. Such conclusion is also consistent with width-agnostic Adam Brownian Motion (ABM) model that accurately describes equilibrium norm in our projector experiment in figure 1. Note that we still see a slight growth of the norms at very small widths. One interpretation of this growth is that the noisy regime described by (1) requires a low enough ratio of signal (e.g. measured by # of tokens in the batch) to model capacity (e.g. measured by # of model parameters).

Implications for µP scaling rules. Let us now highlight the effect of equilibrium norm on LR-only µP scaling recipe typically used in documented LLM applications, and outlined in the top row of table 1. Consider a linear layer y = Wx, x ∈ Rd without learnable multipliers. Then, a width-independent equilibrium (1) breaks the optimal hyperparameter transfer of this scaling rule, leading to exploding activation ∥y∥ ∝ d21 and vanishing relative update strength ∥∆y∥

∥y∥ ∝ d−12. Indeed, assuming normalized input ∥x∥ = const, fixed alignment between x and W, and equilibrium

Matrix Weight Norms

Output Norms

Multiplier Norms

Projector

24

22

Attn key

- 0.075

SSM dt

- d 1

- d 2

22

20

0.070

20

2 2

0.065

2 2

2 4

0.060

Embedding

SSM in_proj

2 4

2 6

Projector

SSM out_proj

Embedding

SSM dt

0.055

Attn in_proj

MLP up_proj

Projector

SSM out

2 6

2 8

Attn out_proj

MLP gate_proj

Attn QK Attn out

MLP gate

MLP down_proj

MLP out

0.050

27 28 29 210 211 212 213

27 28 29 210 211 212 213

27 28 29 210 211 212 213

Model width

Model width

Model width

- Figure 5: The width scaling of the norms of linear layers matrices, various activations throughout the model, and scalar multipliers attached to the selected model activations. We use geometric average to aggregate the values across the layers. Other experimental details can be found in section 4, and we provide time evolution of selected output norms on figure 10.

norm ∥W∥ ∝ λη, we have ∥y∥ ∝ d∥W∥∥x∥ ∝ d × λη ∝ d12. Similarly, for the relative updates ∥∆y∥

= √ηλ ∝ d−12, where we have assumed a scale-free optimizer with ∥∆W∥ ∝ η, and fixed alignment between ∆W and x.

∥y∥ ∝ dd∥∥∆WW∥∥∥∥xx∥∥ = ∥∆W∥

∥W∥ ∝ √η

η/λ

The breakdown of LR-only recipe demonstrated above actually originates from omitting weight decay scaling rather than from a shortcoming of underlying µP foundations. To have a maximal feature learning infinite width limit, Yang & Littwin (2023) notes that the total weight decay coefficient ηλ needs to be fixed when scaling width d. This requirement already fixes the effective learning rate ηeff = √ηλ which governs relative magnitude of updates ∥∆y∥

∥y∥ ∝ ηeff. Then, constant magnitude of activations ∥y∥ can be achieved by either adding weight decay scaling λ ∝ d, or, by reparametrizing the layer as y = sWx and moving the scaling to the multiplier: η,λ = const, s ∝ d−1. The second approach naturally connects to the learnable multipliers and allows us to compare the learned and µP-scaled values of s, that we discuss below.

Learnable multipliers scaling. Now, we return to our experiment with scalar learnable multipliers to check if the right width scaling is learned automatically by the multipliers. As the relative scale of activation update ∥∆y∥

∥y∥ induced by matrix update ∆W is independent from multiplier s, it is sufficient to look only at the scale of activations ∥y∥.

Indeed, on figure 5 (middle) we observe stable activation norms of projector (model’s logits) and selected internal activations: product of the norms of attention keys and values which governing attention logits scale and SSM dt activation that governs forgetting/memorization of the SSM hidden state. While the norms of embeddings and residual blocks outputs jointly grow with width, we note that they are, in a sense, ambiguous due to the residual normalization symmetry. The MLP gate activation seem to fall somewhere in between: formally, it should have fixed scale due to SiLU(·) non-linearity applied to it, but as SiLU(·) asymptotically reduces to homogeneous ReLU(·), the MLP gate is subject to the same residual normalization symmetry. Overall, stable projector, SSM dt and attention QK norms confirms that LRMs adjust to width and learn the required scale.

On figure 10 (right) we look at the actual value of the learned multipliers and compare it to expected µP scaling: d−1 for projector and SSM dt, and d−2 for attention QK, where a factor of d comes from each of Q and K. We observe that all the three considered multipliers scale only a slightly slower than the predicted d−1 and d−2 trends. To locate the source of this mismatch, let us define

Loss

Grad Norm

Loss Difference

- 0
- 1
- 2
- 3
- 4
- 5
- 6
- 7

0.025

| |Exclude Multipliers<br><br>Include Multipliers| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| |Exclude Multipliers<br><br>Include Multipliers| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

- 5
- 6
- 7
- 8
- 9
- 10

0.020

0.015

LossDifference

GradNorm

0.010

Loss

0.005

0.000

0.005

0.010

0 100 200 300 400 500 600 Training Steps

0 100 200 300 400 500 600 Training Steps

0 5 k 10 k 15 k 20 k 25 k 30 k Training Steps

- Figure 6: Training dynamics when excluding (blue) vs. including (orange) multiplier gradients from the global clip norm: (Left) initial loss, (Middle) initial gradient norm, and (Right) the long-term loss difference.

the alignment α between W and x such that ∥y∥ = sdα∥W∥∥x∥, and α ∝ const(d) corresponds to the standard µP limit assumption. As the absence of width scaling of ∥W∥ and ∥y∥ was confirmed independently, and ∥x∥ = 1 by architecture design, we conclude that α ∝ (sd)−1 ∝ da,a < 0. This decay of alignment α between the input features and model weights contradicts the established µP regime and prompts further investigation beyond the scope of current work.

### 4.3 Gradient clipping

Clipping ℓ2-norm of the gradients is a standard approach aimed to improve training stability (Pascanu et al., 2013). We also followed this practice and applied and applied clipping at the value of 1. Contradictory to the intuition developed in this work, we have observed from none to negative impact of adding scalar learnable multiplier in our initial experiments. The unsatisfactory performance was revealed to be related gradient clipping as demonstrated on figure 6 (middle): the run with multipliers had large gradient norms in the initial stages of the training that were clipped, while the gradient norms excluding the contribution from multipliers were significantly below the clipping threshold. This forces an overly aggressive clipping factor that unnecessarily shrinks the updates for all parameters. Excluding multipliers from grad norm computation mitigated the issue and resulted in significantly better performance, as shown in figure 6 (right).

### 4.4 Learning the projector scale

The scale of the final output layer of a neural network has a strong impact on its feature learning ability: large scale leads to so called “lazy training” regime while smaller scales force backbone feature to update significantly to reach the required change in model outputs Chizat et al. (2019); Yang & Hu (2021). Attaching learnable multiplier to the projector (LM head) layer might induce transition between feature learning and lazy regimes, and, therefore, requires careful consideration.

Starting with a standard configuration with final RMSNorm weights acting as projector column multipliers cj, and we first added row multipliers ri that allow to directly adjust the scale of each individual logit. The addition of ri resulted in performance degradation that can be explained within lazy training paradigm: direct learning of individual logits creates a shortcut to quickly fit marginal token distribution without meaningfully updating internal model features. Next, we tried to also remove column multipliers cj while sweeping over fixed scalar multiplier s to further enhance feature learning strength Atanasov et al. (2025), but did not observe an expected improvement in performance and leave further investigation for future work.

Abaltion: P tuning configuration × optimizer × learnable multipliers

Loss change after adding learnable multipliers

0.750

0.001

| |P tuning configuration<br><br>NONE<br><br>LR<br><br>LRWD<br><br>FULL| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Adam Muon start of LR decay

0.000

0.745

0.001

Loss_LRMLoss_no_LRM

0.002

0.740

Finalloss

0.003

0.004

0.735

0.005

0.006

0.730

0.007

0.725

0.008

Adam Adam + LRM Muon Muon + LRM

0 20 40 60 80 100 GigaTokens

- Figure 7: (Left): Loss values after LR decay for 4 multiplier tuning configurations. Each configuration is trained both with standard parametrization and with learnable vector multipliers, marked with +LRM. (Right): loss difference between the runs with and without learnable multipliers.

## 5. Results

### 5.1 Multiplier tuning ablation

Following Zuo et al. (2025), µP model size scaling naturally suggest to tune (i) learning rate, (ii) weight decay and (iii) forward multipliers in order to improve model performance. Our approach automatically removes the need to tune forward multipliers as they become learnable. The need for WD multipliers as a mean to control the weight norms also becomes questionable. To jointly clarify the effects of tuning each hyperparameter type and learning vector multipliers (3), we consider the following 4 tuning configurations with progressive tuning cost

- 1. NONE: no tuned multipliers (i.e. all set to 1).
- 2. LR: only LR multipliers are tuned while forward and WD multipliers are set to 1.
- 3. LRWD: LR and WD multipliers are tuned while forward multipliers are set to 1.
- 4. FULL: All multipliers are tuned.

We perform this ablation for both Adam and Muon optimizers. The tuned values are borrowed from Zuo et al. (2025); in the case where forward multipliers are learnable, the tuned values are used as initialization of the multipliers. The final loss values are depicted in figure 7 (left). We observe a consistent tiering of results for both optimizers: (i) non-learnable and fully not tuned is clearly the worst; (ii) non-learnable configurations with at least learning rate tuning are close to learnable multipliers without any tuning and provide moderate performance boost; (iii) learnable multipliers with learning rate tuning are close to each other and provide the most performance boost. Crucially, this hierarchy holds for Muon as well, confirming that learnable multipliers provide a generalized benefit that complements the optimizer’s specific update rule.

Such grouping prompts several observations. First, learnable multipliers always improve performance of its parent configuration. Second, learnable multipliers do not require tuned initialization or WD multipliers, aligned with expectation that multipliers automatically converge to the optimal scale. Thirdly, tuning the LR multipliers appears essential and, by design, cannot be covered with learnable forward multipliers. Lastly, we note surprisingly absent effect of tuning forward an WD for non learnable configurations. We speculate that this lost performance boost may come from discrepancy between our training data mix and the mix used for tuning. This highlights a brittle and narrow nature of the tuning in contrast to robust data adaptation of learnable approach.

MATH lvl5

Optimizer Hellaswag ARC-C MMLU MMLU-PRO BBH GSM8K

Average Adam 48.91 38.70 44.18 18.26 9.70 48.35 7.52 30.80 Adam+LRM

###### 19.92

49.89

38.73

45.33

12.03

49.10

9.07

32.01

(+0.98)

(+0.03)

(+1.15)

(+1.66)

(+2.33)

(+0.75)

(+1.55)

(+1.21)

Muon 50.31 39.04 47.98 18.96 10.13 48.02 8.69 31.88 Muon+LRM

###### 50.56

###### 39.39

###### 13.52

###### 50.63

###### 9.49

###### 32.98

47.96

19.32

(+0.25)

(+0.35)

(-0.02)

(+0.36)

(+3.39)

(+2.61)

(+0.80)

(+1.10)

- Table 2: Performance comparison of Baseline vs. Learnable multipliers settings. Gains are in green, losses are in red, and the best score per benchmark is in bold. All values are percentages (%). The reported evaluation score are obtained by averaging over checkpoints obtained at an additional 40GT of training after the LR decay. The full evaluation trajectories are reported in figure 11.

In figure 7 (right) we depict the loss difference between the run with and without learnable multipliers. Learnable multipliers develop a loss gap that continues to slowly grow throughout the constant LR stage while slightly shrinking during LR decay. The latter shrinking may be explained from noise point of view: LR decay reduces the noise level in matrix layers, resulting in a less restrictive noise-WD equilibrium with an ability to partially learn the scale. The increase in the loss gap during training supports that the role of learnable multipliers is to increase feature scale diversity and thus arrive at asymptotically better performance, in contrast to simply “speeding up” the training.

### 5.2 Long training validation

As a final validation of our approach, we perform a longer, 200GT duration run for configuration with all tuned multipliers, identical to that of Falcon-H1-0.5B, and its version with learnable vector multipliers. This duration roughly corresponds to ×20 of Chinchilla compute-optimal duration, and, therefore, reasonably mimics real pretraining settings.

Table 2 details the downstream performance. The full evaluation trajectories for these runs are reported in Figure 11 in the Appendix. While Muon itself is a stronger baseline than Adam (31.88% vs. 30.80% average benchmark score), applying learnable multipliers provide equal boost other the respective baseline: 1.21% for Adam and 1.10% for Muon. This suggests that the “noise-WD equilibrium” trap is a general phenomenon affecting various optimizers, and learnable multipliers are a universal solution to escape it.

We observe consistent improvement from LRM across the capabilities board in table 2. Yet, one may notice a general trend: modest improvement for knowledge related benchmarks (ARC-C, MMLU), and a much more significant boost for reasoning related benchmarks (BBH, MATH lvl5, GSM8K). This suggest an uneven impact of learnable multipliers on different model capabilities, though a comprehensive investigation is required to confirm this hypothesis.

## 6. Conclusion and discussion

We have shown that the equilibrium between noise expansion and weight decay, experienced by matrix layers, significantly reduces the scale diversity of model internal representations. To address this problem, we have added learnable multipliers to suitable locations in the model architecture.

These multipliers adjust the fixed scale of matrix layers to the underlying training data and ensure richer representation scales. We validate learnable multipliers in a realistic setting of a long language model pre-training run and observe a sizable improvement. Thus, we conclude that reparameterizing matrix weights with learnable multipliers is a universal path to improving the pretraining performance without any sacrifice in the inference speed or memory cost.

Yet, many questions are left open. In this work, we heavily rely on clear distinction between matrix and scalar/vector weights: dynamics of matrices is noise-dominated and requires WD to counter Brownian expansion, while dynamics of scalar and vectors seem to be signal-dominated, removing the need of WD and ensuring the ability to freely learn the optimal scale. However, signal-to-noise ratio in the weight gradients forms a continuous spectrum, suggesting the existing of a certain criteria of when a parameter tensor acquires scale-adaptation ability. Hence, an interesting direction for future work is to mechanistically understand the difference between matrix and scalar/vector dynamics, find an empirically measurable indicator of the noise level, or build a minimal mathematical model exhibiting both training regimes.

Next set of question is related to developing a complete set of scaling rules, generalizing classical µP scaling to the presence of learnable multipliers. For example, should we scale LR and WD (which constraints symmetries) of multipliers with model size? Or, does application of learnable multipliers automatically ensures maximal feature learning in infinite-width limit without manual scaling rules required in classical µP Yang & Littwin (2023)?

It is practically relevant to further investigate the the relation between learnable multipliers and the difference in improvement it provides to different capabilities we have preliminary seen in table 2. A interesting hypothesis to explore is whether learned multipliers enhance only a certain types of circuits learned by the model Elhage et al. (2021). How training stability and performance improvement of LRMs scale with model size is another practical question.

Finally, we may rephrase the improvement of learnable multipliers over standard architecture in a more general way: standard training procedure has internal flaws preventing converging of the model to the global minimum of population loss for a given data distribution and model architecture, even at asymptotically long training; these flaws must be explicitly addressed to access loss values closer to global optimum. The unlearned matrix scale, corrected by learnable multipliers, is one example of such flaw and its correction. It is an open question whether there are other flaws such kind and whether they can be corrected. For example, are there other parts of parameter matrices apart from row and column norms that are not learned automatically?

## References

Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. GQA: Training generalized multi-query transformer models from multi-head checkpoints. In The 2023 Conference on Empirical Methods in Natural Language Processing, 2023. URL https://openreview.net/forum?id=hmOwOZWzYE.

Alexander Atanasov, Alexandru Meterez, James B Simon, and Cengiz Pehlevan. The optimization landscape of SGD across the feature learning strength. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id= iEfdvDTcZg.

Thomas Bachlechner, Bodhisattwa Prasad Majumder, Henry Mao, Gary Cottrell, and Julian McAuley. Rezero is all you need: fast convergence at large depth. In Cassio de Campos and Marloes H. Maathuis (eds.), Proceedings of the Thirty-Seventh Conference on Uncertainty in

Artificial Intelligence, volume 161 of Proceedings of Machine Learning Research, pp. 1352–1361. PMLR, 27–30 Jul 2021. URL https://proceedings.mlr.press/v161/bachlechner21a.html.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 1877–1901. Curran Associates, Inc., 2020. URL https://proceedings.neurips. cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf.

Lénaïc Chizat, Edouard Oyallon, and Francis Bach. On lazy training in differentiable programming. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper_files/paper/2019/file/ ae614c557843b1df326cb29c57225459-Paper.pdf.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. Palm: Scaling language modeling with pathways, 2022. URL https://arxiv.org/abs/2204.02311.

Tri Dao and Albert Gu. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality, 2024. URL https://arxiv.org/abs/2405.21060.

Soham De and Samuel L. Smith. Batch normalization biases residual blocks towards the identity function in deep networks, 2020. URL https://arxiv.org/abs/2002.10444.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In North American Chapter of the Association for Computational Linguistics, 2019. URL https://api.semanticscholar.org/ CorpusID:52967399.

Nolan Dey, Gurpreet Gosal, Zhiming, Chen, Hemant Khachane, William Marshall, Ribhu Pathria, Marvin Tom, and Joel Hestness. Cerebras-gpt: Open compute-optimal language models trained on the cerebras wafer-scale cluster, 2023. URL https://arxiv.org/abs/2304.03208.

Nolan Dey, Bin Claire Zhang, Lorenzo Noci, Mufan Li, Blake Bordelon, Shane Bergsma, Cengiz Pehlevan, Boris Hanin, and Joel Hestness. Don’t be lazy: Completep enables compute-efficient deep transformers, 2025. URL https://arxiv.org/abs/2505.01618.

Laurent Dinh, Razvan Pascanu, Samy Bengio, and Yoshua Bengio. Sharp minima can generalize for deep nets. In ICML, 2017.

Francesco D' Angelo, Maksym Andriushchenko, Aditya Varre, and Nicolas Flammarion. Why do we need weight decay in modern deep learning? In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 23191–23223. Curran Associates, Inc., 2024. URL https://proceedings.neurips.cc/paper_files/paper/2024/file/ 29496c942ed6e08ecc469f4521ebfff0-Paper-Conference.pdf.

Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. A mathematical framework for transformer circuits. Transformer Circuits Thread, 2021. https://transformer-circuits.pub/2021/framework/index.html.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre. Training compute-optimal large language models, 2022. URL https://arxiv.org/abs/2203.15556.

Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, Xinrong Zhang, Zheng Leng Thai, Kaihuo Zhang, Chongyi Wang, Yuan Yao, Chenyang Zhao, Jie Zhou, Jie Cai, Zhongwu Zhai, Ning Ding, Chao Jia, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong Sun. Minicpm: Unveiling the potential of small language models with scalable training strategies, 2024. URL https://arxiv.org/abs/2404. 06395.

Xiao Shi Huang, Felipe Perez, Jimmy Ba, and Maksims Volkovs. Improving transformer optimization through better initialization. In Hal Daumé III and Aarti Singh (eds.), Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 4475–4483. PMLR, 13–18 Jul 2020. URL https://proceedings.mlr. press/v119/huang20f.html.

Keller Jordan. Muon: An optimizer for hidden layers in neural networks, 2024. URL https: //kellerjordan.github.io/posts/muon/.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Yoshua Bengio and Yann LeCun (eds.), 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015. URL http: //arxiv.org/abs/1412.6980.

Seijin Kobayashi, Yassir Akram, and Johannes von Oswald. Weight decay induces low-rank attention layers. In The Thirty-eighth Annual Conference on Neural Information Processing Systems,

2024. URL https://openreview.net/forum?id=oDeqjIM9Sk.

Atli Kosson, Bettina Messmer, and Martin Jaggi. Rotational equilibrium: How weight decay balances learning across neural networks, 2024. URL https://arxiv.org/abs/2305.17212.

Atli Kosson, Jeremy Welborn, Yang Liu, Martin Jaggi, and Xi Chen. Weight decay may matter more than mup for learning rate transfer in practice, 2025. URL https://arxiv.org/abs/2510. 19093.

Teven Le Scao, Angela Fan, et al. Bloom: A 176b-parameter open-access multilingual language model. arXiv:2211.05100, 2022.

Haokun Liu, Derek Tam, Muqeeth Mohammed, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin Raffel. Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id= rBCvMG-JsPd.

Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, Yanru Chen, Huabin Zheng, Yibo Liu, Shaowei Liu, Bohong Yin, Weiran He, Han Zhu, Yuzhi Wang, Jianzhou Wang, Mengnan Dong, Zheng Zhang, Yongsheng Kang, Hao Zhang, Xinran Xu, Yutao Zhang, Yuxin Wu, Xinyu Zhou, and Zhilin Yang. Muon is scalable for llm training, 2025. URL https://arxiv.org/abs/2502.16982.

Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, KwangTing Cheng, and Min-Hung Chen. Dora: Weight-decomposed low-rank adaptation, 2024. URL https://arxiv.org/abs/2402.09353.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id= Bkg6RiCqY7.

Paulius Micikevicius, Sharan Narang, Jonah Alben, et al. Mixed precision training. arXiv:1710.03740, 2017.

Kosuke Nishida, Kyosuke Nishida, and Kuniko Saito. Initialization of large language models via reparameterization to mitigate loss spikes. In Conference on Empirical Methods in Natural Language Processing, 2024. URL https://api.semanticscholar.org/CorpusID:273186687.

Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio. On the difficulty of training recurrent neural networks. In Proceedings of the 30th International Conference on Machine Learning, pp. 1310–

1318. PMLR, 2013. Tim Salimans and Diederik P. Kingma. Weight normalization: A simple reparameterization to accelerate training of deep neural networks, 2016. URL https://arxiv.org/abs/1602.07868. Qiushi Wang, Yuchen Fan, Junwei Bao, Hongfei Jiang, and Yang Song. Bora: Bi-dimensional weight-decomposed low-rank adaptation, 2024. URL https://arxiv.org/abs/2412.06441.

Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tieyan Liu. On layer normalization in the transformer architecture. In Hal Daumé III and Aarti Singh (eds.), Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 10524–10533. PMLR, 13–18 Jul 2020. URL https://proceedings.mlr.press/v119/xiong20b.html.

Greg Yang and Edward J. Hu. Tensor programs iv: Feature learning in infinite-width neural networks. In Marina Meila and Tong Zhang (eds.), Proceedings of the 38th International Conference

on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 11727–11737. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/v139/yang21c.html.

Greg Yang and Etai Littwin. Tensor programs ivb: Adaptive optimization in the infinite-width limit, 2023. URL https://arxiv.org/abs/2308.01814.

Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao. Tensor programs v: Tuning large neural networks via zero-shot hyperparameter transfer, 2022. URL https://arxiv.org/abs/ 2203.03466.

Greg Yang, James B. Simon, and Jeremy Bernstein. A spectral condition for feature learning, 2024a. URL https://arxiv.org/abs/2310.17813.

Greg Yang, Dingli Yu, Chen Zhu, and Soufiane Hayou. Tensor programs VI: Feature learning in infinite depth neural networks. In The Twelfth International Conference on Learning Representations, 2024b. URL https://openreview.net/forum?id=17pVDnpwwl.

Hongyi Zhang, Yann N. Dauphin, and Tengyu Ma. Residual learning without normalization via better initialization. In International Conference on Learning Representations, 2019. URL https: //openreview.net/forum?id=H1gsz30cKX.

Jingwei Zuo, Maksim Velikanov, Ilyas Chahed, Younes Belkada, Dhia Eddine Rhayem, Guillaume Kunsch, Hakim Hacid, Hamza Yous, Brahim Farhat, Ibrahim Khadraoui, Mugariya Farooq, Giulia Campesan, Ruxandra Cojocaru, Yasser Djilali, Shi Hu, Iheb Chaabane, Puneesh Khanna, Mohamed El Amine Seddik, Ngoc Dung Huynh, Phuc Le Khac, Leen AlQadi, Billel Mokeddem, Mohamed Chami, Abdalgader Abubaker, Mikhail Lubinets, Kacper Piskorski, and Slim Frikha. Falcon-h1: A family of hybrid-head language models redefining efficiency and performance, 2025. URL https://arxiv.org/abs/2507.22448.

## A. Experiment settings

We perform all our experiments with Falcon-H1-0.5B architecture (Zuo et al., 2025). The main reason for this choice is the availability of 35 extensively tuned µP multipliers that serve as a strong baseline for our learnable multipliers. The 0.5B model scale provides a reasonable tradeoff between the model’s ability and the computational cost of running multiple experiments. Finally, we rely on the available training infrastructure for hybrid attention-SSM models to reduce infrastructure implementations and focus on the multiplier-related aspects.

The training duration for most of the experiments is fixed to 30 GT, comprised of 25GT of the constant learning rate stage and 5GT of exponential LR decay with ×8 LR reduction. As can be seen from figure 1, this duration is sufficient for the weight to stabilize in noise-WD equilibrium. Also, 30GT corresponds to ×3 of Chinchilla (Hoffmann et al., 2022) compute optimal duration for 0.5B model scale, ensuring that the model is adequately trained. For the final validation, we use 240GT, a ×24 of compute-optimal duration to test the behavior of the multipliers in a more realistic setting of longer training duration.

The other training hyperparameters also follow (Zuo et al., 2025). Specifically, we use a warmup duration of 0.1GT, batch size rampup with the square root LR scaling rule, and the global weight decay value of 0.1, which is further modified by the tuned weight decay multipliers as given by table 8 of (Zuo et al., 2025). For each configuration of multipliers, we perform a learning rate sweep on a log-scale grid with √2 step and use the optimal value for experiments in the paper.

Additional details for selected experiments. In all the experiments we use zloss Chowdhery et al. (2022) with coefficient 10−4 as it leads to better model performance. However, we disable the zloss for projector experiment discussed in section 3 and on figure 1 as the zloss directly affects the behavior of the model logits, convoluting the clear interpretation of the logits norm required to investigate of learnable multipliers only.

For width scaling experiment discussed in section 4.2, we used a smaller model with 12 layers to access a wider range of widths within reasonable compute budget. The RMSNorms in all blocks were frozen to ensure unit normalization ∥x∥ = 1 of input block features, enabling cleaner interpretation of the observed norms.

## B. MLP experiment

In section 3 and on figure 1 (bottom right) we observed that varying the scale MLP relative to all the other parts of the model leads to performance degradation while adding learnable multipliers to MLP removes this degradation. In this section, we illustrate that the degradation if indeed related to scale mismatch between mlp and other blocks In this section, we extend the discussion of MLP experiment presented in section 3 and on figure 1 by showing the behavior of internal activations and parameter norms as we vary the norm scale S(ηMLP,λMLP) = ηMLP

λMLP of the MLP matrix layers.

Let us first repeat and complement a brief description of the MLP experiment provided in section 3. First, we set RMSNorm weights cj → 1 in (5) for all the MLP, attention, and SSM blocks to restrict the scale adaptation ability of these blocks. Then, we vary the norm scale of MLP block matrices in the following way: for each of Wup,Wgate,Wdown (see (6)) we change learning rate η and λ while keeping ηλ = const. We denote ηMLP = η/η0 and λMLP = λ/λ0 the relative change

1For simplicity, we follow the implementation in our codebase and report the norm of the merged QKV matrix. Same applies to the merged WXZBCdt projection matrix of the SSM block

MLP

###### SSM

Attention

Embedding

- 20
- 21
- 22
- 23
- 24
- 25
- 26
- 27

- 20
- 21
- 22
- 23
- 24
- 25
- 26
- 27

- 20
- 21
- 22
- 23
- 24
- 25
- 26
- 27

- 20
- 21
- 22
- 23
- 24
- 25
- 26
- 27

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

Outputnorm

2 1

2 1

2 1

2 1

2 2

2 2

2 2

2 2

2 2 2 1 20 21 22

2 2 2 1 20 21 22

2 2 2 1 20 21 22

2 2 2 1 20 21 22

MLP

###### SSM

Attention

Embedding

20

20

20

20

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

2 1

2 1

2 1

2 1

Matrixweightnorm

2 2

2 2

2 2

2 2

2 3

2 3

2 3

2 3

2 4

2 4

2 4

2 4

2 5

2 5

2 5

2 5

2 2 2 1 20 21 22

2 2 2 1 20 21 22

2 2 2 1 20 21 22

2 2 2 1 20 21 22

MLP multipliers

Conv1d (SSM)

###### D (SSM)

Loss after LR decay

- 20
- 21
- 22
- 23
- 24
- 25
- 26

- 20
- 21
- 22
- 23
- 24
- 25
- 26

- 20

- 21

- 22

- 23

- 24

- 25

- 26

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

Adam

0.78

Adam + MLP learnable scalars

Multiplierweightnorm

Muon

Muon + MLP learnable scalars

0.77

Loss

0.76

2 1

2 1

2 1

0.75

2 2

2 2

2 2

2 3

2 3

2 3

0.74

2 4

2 4

2 4

2 2 2 1 20 21 22

2 2 2 1 20 21 22

2 2 2 1 20 21 22

2 2 2 1 20 21 22

MLP norm scale S( MLP, MLP)

MLP norm scale S( MLP, MLP)

MLP norm scale S( MLP, MLP)

MLP norm scale S( MLP, MLP)

- Figure 8: All the subplots, except bottom right, show behavior of different norms as the MLP norm

scale S(ηMLP,λMLP) is varied. The rows contain similar types of norms and share y-axis scale for easier comparison. The bottom right subplot duplicates the bottom right subplot of figure 1; we add it for convenience as an illustration of performance across the four considered configurations.

(Top row): The norms of the output of MLP, attention and SSM residual blocks, as well as the norm of the outputs of embedding layer. The norms are averaged across the tokens in a batch. Then, we apply geometric average to aggregate residual block norms across model layers.

(Middle row): The norms of matrix weights: Wup,Wgate,Wdown for MLP (see (6)), WQKV 1and Wout for attention (see (7)), WXZBCdt and Wout for SSM (see (8),(9),(10),(11),(12)), and the embeddings matrix. To aggregate the weight norms into a single metric for residual blocks, in addition to geometric averaging across the layers we further apply geometric averaging across the matrix types within the block, for example Wup,Wgate,Wdown for MLP.

(Bottom row): The norms of various non-matrix parameters that are free from the noise-WD equilibrium and can adjust their scale. For the three MLP multipliers we again apply geometric average; Conv1d acts as a row multiplier for WXBC in the SSM block (see (8),(9)); D parameter scales the skip connection in the recurrent SSM computation, such that larger values of D make SSM block computation closer to gated MLP.

of learning rate and weight decay, which are kept the same for all 3 MLP layers2. In figures 1,8,9, we use these relative values to focus on the change of the matrix norms3. The above fully describes our baseline configuration, and in the configuration with learnable MLP multipliers, we add three scalar multipliers sup,sgate,sdown to the three MLP matrix layers4. These scalar multipliers had a fixed learning rate η = 10−2 and no weight decay. Finally, we run two versions of the experiment with Muon and Adam optimizers, while keeping all the other settings identical.

With the experiment settings settled, we proceed to a discussion of the behavior of parameter and output norms as MLP matrix norm scaled S(ηMLP,λMLP) is varied. First, we see a clear picture on figures 8 and 9 for Adam and Muon configurations with learnable multipliers:

- 1. The norms of MLP matrices follow the equilibrium scale S(ηMLP,λMLP) = ηMLP

λMLP .

- 2. Yet, the outputs of all residual blocks stay constant regardless of the MLP matrices scale.
- 3. The constant level of residual blocks output is achieved thanks to MLP multiplier compensating the change in scale of MLP matrices (bottom left plot of figure 8).

This confirms the main thesis of the current work: the ability of learnable multipliers to freely adjust their scale in order to compensate a (presumably) suboptimal scale of its respective matrix layers, which is trapped in noise-WD equilibrium (1).

Next, we look at the configurations without learnable multipliers which turned out to be more nuanced. As we have restrained the scale adaptation ability of residual blocks by freezing the respective RMSNorm weights, pure equilibrium norm scaling (1) would create a imbalance between scales of MLP and attention/SSM blocks. This imbalance is expected to significantly degrade the model performance, which translates in a loss gap between configuration with and without learnable multipliers, seen on figure 8 (bottom right). However, this imbalance also creates a significant optimization pressure to balance back the scales of MLP and attention/SSM blocks, and the model manages to align the scales of the residual blocks as can be seen in the top row of figure 8. This scale adaption is achieved via accumulating several mechanism which we list below.

- • MLP. On fig. 8 (middle left) we see that the norms of MLP matrix layers slightly deviate noise-WD scaling (1). We interpret this deviation us a result of optimization pressure to reduce the scale gap between MLP and attention/SSM blocks. This introduces a strong enough gradient signal force that modifies the equilibrium (1) which was governed by gradient noise and WD forces only.
- • Attention. We observe on fig. 8 (top row, third column) the growth of the outputs of

attention blocks with S(ηMLP,λMLP). To estimate the norm of the attention block outputs, we may ignore the attention scores and approximate the norm as ∥WoutWV x∥ (see (7)). One way to increase this norm is to increase alignment between Wout and WV , and also between WV and x. Another way is for Wout or WV to escape noise-WD equilibrium, which is confirmed by the growth of the respective matrix norms on fig. 8 (middle row, third column). We expect that value matrix is WV is more prone to escaping noise-WD equilibrium in groupquery attention with large ratios of Q heads to KV heads: sharing of the KV heads results

- 2The current absolute values of η, λ, as well as the baseline absolute values η0, λ0 might be different between Wup, Wgate, and Wdown projections. This is the case for our experiment, where we use tuned µP multipliers from Zuo et al. (2025) with slightly different multipliers for the three MLP projections.
- 3Same conventions apply for the projector experiment in figure 1. 4Multiplicative symmetry makes one of the scalar multipliers sup and sdown redundant. We keep both of them

for simplicity, and, for the short training duration of MLP experiment, we did not observe any symmetry-induced training instabilities discussed in section 4.1.

into more gradient signal coming through the matrix, resulting in higher signal-to-noise ratio. We suspect all these three mechanisms to increase the attention output to take place. Yet, from a slower growth of the attention outputs compared to SSM outputs we may conclude that employing these mechanisms negatively impacts the quality of attention blocks. but slightly lags similar growth of SSM block output.

- • SSM. Surprisingly, the outputs of SSM blocks grow at the same rate as outputs of MLP block while the norms of SSM matrices stay constant, in agreement with their noise-WD equilibrium values (fig. 8 top and middle row, second column). After noticing this, we explored components of SSM block and identified two parts: conv1d (see (8),(9)) and SSM skip connection scale D that in fact play a role of learnable multipliers as they have vector-like shapes and hence not subject to noise-WD equilibrium. Indeed on fig. 8 (bottom row, second and third columns) we observe the growth of these parameters, explaining the growth of SSM outputs at fixed norm of the respective matrix layers.
- • Embedding. The norm of embedding matrix stays constant, following its fixed equilibrium value (fig. 8, middle right), while the embedding outputs slightly grows to catch up with residual blocks outputs. To explain this growth, we note that that embedding output norm is a average norm of token embedding vectors weighted with frequency of each token in the data. Then, most frequent tokens display vector-like behavior with high signal-to-noise ratio in the gradients which allows them to partially escape noise-WD equilibrium and adjust their norm. On the other hand, low frequency tokens have low signal-to-noise ratio and therefore get trapped in the noise-WD equilibrium.

Finally, let us comment on the differences and similarities of Adam and Muon optimizers in the observed behaviors. In almost all the considered aspects, two optimizers behave identically, suggesting that noise-WD equilibrium mechanism could be a general phenomenon applicable to wide range of optimizer update rules. The cases where we observe a moderate difference between Adam and Muon include the growth of attention, conv1d and D norms.

Remark on experiment design. As we have seen above, the configuration without learnable multipliers manage to adapt the scale of attention and SSM blocks. This creates a complex picture of behaviors to ensure the growth of attention/SSM outputs, failing our original intent in restricting scale adaptation ability of these blocks to produce a clean separation between configurations with and without learnable multipliers.

A few choices in the design of this MLP experiment may help to satisfy the original intention in the restricting the attention/SSM scale adaptability. For attention, we can switch from grouped query attention (GQA) to multi-head attention (MHA): this decreases the signal-to-noise ration in value matrices, making it harder for them to escape the equilibrium. For SSM, we can activate the internal RMSNorm5, as in (11), nullifying the effect of conv1d and D in the SSM output scale.

## C. Multipliers placement

In section 4.1 we highlighted multiplicative and normalization symmetries that cause training instabilities if left unchecked. In this section, we write down the forward pass of the architecture employed in our experiments, Falcon-H1, in order to show such placement of learnable multipliers

5The reason we did not have this RMSNorm in our experiment is absence of it in Falcon-H1-0.5B architecture used throughout the work.

MLP

###### SSM

Attention

- 100

- 101

- 102

- 103

- 100

- 101

- 102

- 103

- 100

- 101

- 102

- 103

Adam

10 1

10 1

10 1

0 5 10 15 20 25 30 35 Layer index

0 5 10 15 20 25 30 35 Layer index

0 5 10 15 20 25 30 35 Layer index

- 100

- 101

- 102

- 103

- 100

- 101

- 102

- 103

- 100
- 101
- 102
- 103

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

Adam+MLPlearnablescalars

10 1

10 1

10 1

0 5 10 15 20 25 30 35 Layer index

0 5 10 15 20 25 30 35 Layer index

0 5 10 15 20 25 30 35 Layer index

- 100

- 101

- 102

- 103

- 100
- 101
- 102
- 103

- 100

- 101

- 102

- 103

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

Muon

10 1

10 1

10 1

0 5 10 15 20 25 30 35 Layer index

0 5 10 15 20 25 30 35 Layer index

0 5 10 15 20 25 30 35 Layer index

- 100

- 101

- 102

- 103

- 100

- 101

- 102

- 103

- 100
- 101
- 102
- 103

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | |MLP nor S( MLP,<br><br>|m scale MLP)<br><br>0.25| |
| | | | | | | |0.5<br>1<br>2 4<br>| |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

Muon+MLPlearnablescalars

10 1

10 1

10 1

0 5 10 15 20 25 30 35 Layer index

0 5 10 15 20 25 30 35 Layer index

0 5 10 15 20 25 30 35 Layer index

- Figure 9: The behavior the norms of the MLP, attention and SSM residual blocks outputs across the model layers. This figure complements figure 8, where the norms were averaged across the layers. Overall, we observe strong layer-specific patterns, suggesting that the layer-averaged metrics in figure 8 adequately capture the behavior of the norms with respect to MLP scale S(ηMLP,λMLP).

that removes all the redundancy, and associated symmetries, without reducing scale adaption expressivity of the multipliers. While in this section we focus on the specific hybrid attention-SSM architecture we used, we believe that it illustrates general principles guiding the symmetry-aware multiplier placement and can be easily generalized to other architectures.

However, when experimenting with such symmetry-aware placement, we observed a slightly worse performance than for configuration with all multipliers. Additionally, we note that there is no reliable way to fix residual normalization symmetry. Hence, in all our experiments, we chose to use both row and column multipliers while fixing the symmetry with multipliers weight decay, as discussed in section 4.1. In spite of using WD-based symmetry handling, we view symmetry-aware placement of multipliers as useful for better understanding of their effect, and potentially useful in future scenarios of LRMs usage.

### C.1 Embedding and projector

For embedding, we use both column and vector multipliers. Theoretically, we could have removed all the multipliers from embedding to fix normalization residual symmetry. However, we observe that removing these multipliers results in a suboptimal performance. It could be hypothetically related to the dominant contribution of backbone blocks in the residual compared to the initial embeddings (see figure 8, top row), which results in block outputs ignoring the fixed embedding scale instead of adapting to it.

For the projector, we again note that the well-established RMSNorm just before the projector already contains column multipliers for the projector matrix (see also projector experiment described in sec. 3).

To summarize, we recommend adding both vector and column to the embedding layer while leaving the projector layer unchanged.

### C.2 Residual blocks

We use a nowadays standard pre-LN design of the model blocks which uses RMSNorm in the beginning of each block. Specifically, for a residual z ∈ Rd at the beginning of a block, the output x ∈ Rd6 of RMSNorm is given by

cjzj

RMSNorm(z)

=

, (5)

1 d k zk2

j

where cj are the learnable weights of the RMSNorm layer. As discussed in sections 2 and 3, cj act as column multipliers for the matrix layer that follows RMSNorm. In all the cases, we keep these “column multipliers”.

Gated MLP block. The output of the block yi is computed as

yi =

k

Wikdown SiLU

j′

Wkjgate′ xj′

j

Wkjupxj (6)

Then, a maximally expressive configuration of multipliers without symmetries would be

6We slightly abuse the notation x (or, equivalently, xj). In section 2 it denotes any internal activation just before application of a linear transformation by a matrix W. In section 3 it denotes final token features after applying RMSnorm normalization but before multiplying by the weights cj. And, finally, in this section we include the RMSNorm weights cj inside xj to lighten the notations and to focus on the new learnable multipliers introduced to the classical architectures that already use RMSNorm layer.

- • Row and column multipliers for Wdown.
- • No multipliers for Wup.
- • Row only multipliers for Wgate.

Let us again remind that we don’t include column multipliers for Wgate and Wup because of the previously assumed usage of RMSNorm weight. Note, however, that removing RMSNorm weight while adding a column multiplier for both Wgate and Wup would provide a non-redundant but more expressive configuration. At the moment, we have not tested this option (and similar options for the other blocks), leaving it for future work.

Attention block. The contribution yi,lh of a single attention head h to the block output at position l reads

yi,lh =

k

Wikout,h

l′

Softmaxl′

m j

WmjQ,hxj,l

j′

WmjK,h′ xj′,l′

j′′

WkjV,h′′xj′′,l′. (7)

Then, a maximally expressive configuration of multipliers without symmetries would be

- • Row and column multipliers for Wout.
- • No multipliers for WV and WK.
- • Row only multipliers for WQ.

Let us comment on this placement. Using row multipliers for both key and query matrices is redundant and was already discussed sec. 4.1. Hence, we are left with the choice to put the multipliers either on the key or the query. A more expressive choice is dictated by the structure of the Group Query Attention (GQA) (Ainslie et al., 2023): a single key head is shared with several query heads. Therefore, putting multiplier os queries allows the model to learn per-head attention scales instead of per-group scales we would get in the case of attaching multipliers to keys. A similar reasoning applies to the choice of Wout column multipliers vs. WV row multipliers: attaching multipliers to output projection allows the model to learn with per-head output scales, while value projection multipliers would only learn per-group scale.

SSM (Mamba2) block. Taking into account a more complicated structure of the Mamba2 block, we break its computation into parts, focus on a single head, and, for simplicity, omit the temporal index and most of the internal channel indices. Then, Mamba2 forward pass reads

X = SiLU conv1d WXx) , Z = SiLU WZx) , (8) B = SiLU conv1d WBx) , C = SiLU conv1d WCx) , (9) dt = Softplus Wdtx + bdt , (10) F = RMSNorm SSM X,B,C,dt ⊙ Z , (11) yi =

WikoutFk. (12)

j

Here SSM X,B,C,dt is the mamba2 sequence transformation (Dao & Gu, 2024), and conv1d(·) is casual per-channel convolution. Let us comment on this structure to arrive at our final multiplier

configuration. RMSNorm layer is typically used, but can be skipped in some cases, including Falcon-H1-0.5B architecture that we use for our experiments.

Let us comment on each part of the computation to determine where learnable multipliers are required. Casual conv1d can be viewed as a more expressive operation than row multipliers, as it adds short-range temporal mixing in addition to the per-channel rescaling. Therefore, we do not apply row multipliers to WX,WB,WC. Since the output of both Wdt and WZ goes into a nonlinearity, and the respective parts do not have native parameters able to learn the scale, we apply row multipliers to these matrices. Finally, output projection Wout does not have any symmetries associated with it, and thus requires both row and column multipliers unless the RMSNorm layer is present and already contains the column multiplier. Summarizing, we have

- • Row only multiplier for Wout. Column multiplier is added if the internal RMSNorm layer is skipped.
- • No multipliers for WX,WB,WC.
- • Row only multiplier for WZ and Wdt.

## D. Additional plots

Output Norms by Width

Embedding

Outputs (MLP / Attn)

Projector

100

W128 W256 W512

MLP output

10 1

- 100

- 101

Attention output

SignalNorm

10 1

W1024 W2048 W4096

10 2

10 2

10 3

10 1

100 101 102 103 104 Training Steps

100 101 102 103 104 Training Steps

100 101 102 103 104 Training Steps

- Figure 10: Evolution of signal norms during training across different model widths (W128 to W4096). (Center) MLP and Attention output norms show consistent norm growth in the later stages of training, which we attribute to drift along the direction of residual normalization symmetry. (Left) Embedding output norms also grow to match residual growth. (Right) Projector output norms also grow with training time but seem to plateau at the same level, presumably corresponding to the reasonable logits scale. The model width affect the projector output norm only in the intermediate stage of training, while for MLP and attention outputs norm are grow with different offset for different widths, reflecting arbitrary scale of residuals due to normalization symmetry.

Model Type

adam adam_lrm muon muon_lrm

Average

bbh

0.34

| | | |
|---|---|---|
| | | |

| | |
|---|---|
| | |

0.14

| | |
|---|---|
| | |

| | |
|---|---|

| | |
|---|---|
| | |

0.32

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

0.12

0.30

| | |
|---|---|

0.28

Average

| | |
|---|---|
| | |

0.10

| | |
|---|---|

bbh

| | |
|---|---|
| | |

0.26

0.08

| | | |
|---|---|---|
| | | |

| | | | |
|---|---|---|---|
| | | | |

0.24

| | |
|---|---|
| | |

| | |
|---|---|

| | |
|---|---|

| |
|---|
| |

| | | | | |
|---|---|---|---|---|
| | | | | |

| | |
|---|---|

| | |
|---|---|
| | |

| |
|---|
| |

| | |
|---|---|
| | |

| | | |
|---|---|---|
| | | |

| | |
|---|---|

0.06

| |
|---|
| |

| | |
|---|---|
| | |

| | | | |
|---|---|---|---|
| | | | |

| |
|---|
| |

| | |
|---|---|
| | |
| | |

| | | | |
|---|---|---|---|
| | | | |

| | | |
|---|---|---|
| | | |

0.22

0.04

0.20

100 120 140 160 180 200 220 240

100 120 140 160 180 200 220 240

GTok

GTok

hellaswag

arc_challenge

0.40

| | |
|---|---|

| | |
|---|---|

| | | | | |
|---|---|---|---|---|
| | | | | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

0.50

| | |
|---|---|

| | |
|---|---|
| | |

| | |
|---|---|

| | |
|---|---|
| | |

| | | | | |
|---|---|---|---|---|
| | | | | |

| | |
|---|---|
| | |

| | | |
|---|---|---|

| | | | |
|---|---|---|---|
| | | | |

0.38

0.48

arc_challenge

| | |
|---|---|
| | |

hellaswag

0.36

| | | | |
|---|---|---|---|
| | | | |

| | |
|---|---|

0.46

| | | |
|---|---|---|
| | | |

| | | | |
|---|---|---|---|
| | | | |

0.34

| | |
|---|---|

0.44

| |
|---|
| |
| |

| |
|---|
| |
| |

| | | |
|---|---|---|
| | | |

| |
|---|
| |
| |

| |
|---|
| |
| |

| | |
|---|---|

| | |
|---|---|
| | |

| |
|---|
| |
| |

0.32

| | |
|---|---|

| | |
|---|---|

| | |
|---|---|
| | |

0.42

| | | |
|---|---|---|
| | | |

| | | | |
|---|---|---|---|
| | | | |

| | |
|---|---|

| | |
|---|---|

| |
|---|
| |
| |

| |
|---|
| |
| |

0.30

100 120 140 160 180 200 220 240

100 120 140 160 180 200 220 240

GTok

GTok

mmlu

mmlu_pro

0.48

| | |
|---|---|
| | |

0.20

| | |
|---|---|

| | |
|---|---|
| | |

| | |
|---|---|

| | |
|---|---|
| | |

| | |
|---|---|

| | |
|---|---|
| | |

| | | | |
|---|---|---|---|
| | | | |

| | |
|---|---|
| | |

| | |
|---|---|

0.46

| | |
|---|---|

| | |
|---|---|
| | |

| | |
|---|---|

0.18

| | |
|---|---|
| | |

0.44

| | |
|---|---|
| | |

mmlu_pro

0.16

| | | | |
|---|---|---|---|
| | | | |

| | | | |
|---|---|---|---|
| | | | |

mmlu

0.42

0.14

| | |
|---|---|

| |
|---|
| |

0.40

| | | |
|---|---|---|
| | | |

| | |
|---|---|

| | | | |
|---|---|---|---|
| | | | |

| | |
|---|---|

| |
|---|
| |

| | | | |
|---|---|---|---|
| | | | |

| | |
|---|---|
| | |

| | | | |
|---|---|---|---|
| | | | |

| |
|---|
| |

| | | | |
|---|---|---|---|
| | | | |

0.12

| | |
|---|---|

| |
|---|
| |

0.38

| | |
|---|---|

| |
|---|
| |

| | |
|---|---|

0.36

0.10

0.34

100 120 140 160 180 200 220 240

100 120 140 160 180 200 220 240

GTok

GTok

gsm8k

math

| | | |
|---|---|---|
| | | |

| | |
|---|---|
| | |

0.5

0.10

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|

| |
|---|
| |

| | |
|---|---|
| | |

0.4

| |
|---|
| |

0.08

gsm8k

math

0.3

| |
|---|
| |

| |
|---|
| |

| | |
|---|---|

| |
|---|
| |
| |

0.06

| | |
|---|---|

| |
|---|
| |
| |

| | | | | |
|---|---|---|---|---|
| | | | | |

| | |
|---|---|

| | | | |
|---|---|---|---|
| | | | |

| | |
|---|---|
| | |

| | | | |
|---|---|---|---|
| | | | |

0.2

| |
|---|
| |

| | |
|---|---|

0.04

| |
|---|
| |

| | | | |
|---|---|---|---|
| | | | |

| | | | |
|---|---|---|---|
| | | | |

| | | | |
|---|---|---|---|
| | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |

| | | | |
|---|---|---|---|
| | | | |

| | | | |
|---|---|---|---|
| | | | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|

| |
|---|
| |

0.1

0.02

100 120 140 160 180 200 220 240

100 120 140 160 180 200 220 240

GTok

GTok

- Figure 11: The detailed evaluation curves for the long runs reported in table 2. The markers correspond to actual evaluation score at a given checkpoint while solid lines denote running window average score over 20 last checkpoints. We evaluated checkpoints every gigatoken to carefully average the benchmarks stochasticity, and started this frequent evaluation only from 100GT due to compute constraints. We perform ×32 exponential decay from 160GT to 200GT, which explains the growth of the scores in this time window is thanks to the learning rate decay. After the end of exponential decay, the model was trained for 40 more gigatokens with minimal learning rate to obtain enough evaluation points ensuring well averaged scores reported in table 2.

