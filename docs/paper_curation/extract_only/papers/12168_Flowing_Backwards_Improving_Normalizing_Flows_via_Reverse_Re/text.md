## Flowing Backwards: Improving Normalizing Flows via Reverse Representation Alignment

### Yang Chen1,2, Xiaowei Xu2, Shuai Wang1, Chenhui Zhu1,2, Ruxue Wen2, Xubin Li2, Tiezheng Ge2, Limin Wang1,3,

1State Key Laboratory for Novel Software Technology, Nanjing University 2Alibaba Group, 3Shanghai AI Lab

Corresponding author: lmwang@nju.edu.cn Code: https://github.com/MCG-NJU/FlowBack

| |
|---|

Noise Token Image Token Label

Learnable Label Trainable

# arXiv:2511.22345v2[cs.CV]4Dec2025

[Figure 1]

TARBlock

[Figure 2]

[Figure 3]

TARBlock

[Figure 4]

TARBlock

[Figure 5]

Frozen

[Figure 6]

TARBlock

[Figure 7]

[Figure 8]

TARBlock

TARBlock

[Figure 9]

NFLoss

NFLoss

TARBlock

[Figure 10]

[Figure 11]

TARBlock

TARBlock

[Figure 12]

TARBlock

[Figure 13]

[Figure 14]

TARBlock

TARBlock

[Figure 15]

TARBlock

[Figure 16]

[Figure 17]

TARBlock

TARBlock

[Figure 18]

| | |
|---|---|
| | |

| | |
|---|---|
| | |

b) Generation

c) Classification

a) Training

d) FID vs Validation Accuracy

Figure 1: Take TARFlow as a representative NF. (a) Training process maps images to a noise distribution. (b) The reverse pass generates images. (c) Optimizing a label token by NF loss to classify. (d) The FID-Accuracy plot demonstrates that our representation alignment improves both generation quality and classification performance.

### 1 Introduction

##### Abstract

Normalizing Flows (NFs) represent a distinct class of generative models, characterized by their exact mathematical invertibility (Rezende and Mohamed 2015; Dinh, Krueger, and Bengio 2014; Dinh, Sohl-Dickstein, and Bengio 2016). This property defines their rigid, dual-pathway architecture: a forward pass transforms data into a simple latent distribution for exact log-likelihood optimization, while a mathematically precise reverse pass generates data from that same latent space (Figure 1a,b). This structure implies the synergy of NFs between data generation and representation learning, where the two are truly two sides of the same coin.

Normalizing Flows (NFs) are a class of generative models distinguished by a mathematically invertible architecture, where the forward pass transforms data into a latent space for density estimation, and the reverse pass generates new samples from this space. This characteristic creates an intrinsic synergy between representation learning and data generation. However, the generative quality of standard NFs is limited by poor semantic representations from log-likelihood optimization. To remedy this, we propose a novel alignment strategy that creatively leverages the invertibility of NFs: instead of regularizing the forward pass, we align the intermediate features of the generative (reverse) pass with representations from a powerful vision foundation model, demonstrating superior effectiveness over naive alignment. We also introduce a novel training-free, test-time optimization algorithm for classification, which provides a more intrinsic evaluation of the NF’s embedded semantic knowledge. Comprehensive experiments demonstrate that our approach accelerates the training of NFs by over 3.3×, while simultaneously delivering significant improvements in both generative quality and classification accuracy. New state-of-the-art results for NFs are established on ImageNet 64×64 and 256×256.

This inherent synergy suggests a clear path toward enhancing the generative capabilities of NFs: by improving the quality of their learned representations. However, this potential remains largely underexploited. Standard NFs, optimized solely for log-likelihood on the forward pass, often fail to learn semantically meaningful features, which in turn limits their generative quality. The model’s rigid adherence to the likelihood objective prevents it from fully realizing the benefits of its own architectural duality.

This line of inquiry is particularly relevant given recent findings that actively improving a model’s representational quality can enhance its generative capabilities. For instance, a notable method, REPA (Yu et al. 2024), regularized the internal features of diffusion models against a strong, pretrained visual encoder. This ‘representation-first’ strategy yielded significant gains in both training efficiency and generation quality, demonstrating the effectiveness of leveraging high-quality external guidance.

Inspired by REPA, our work aims to adapt and extend it to the unique context of NFs. We ask: how can the invertible structure of NFs be leveraged to effectively synergize representation learning and generation? By utilizing the reverse generative pathway for alignment—a strategy uniquely enabled by NFs—we move beyond direct application to explore new alignment strategies within this model class. This forms the core motivation for our work.

In this paper, we first need a way to probe the intrinsic discriminative abilities of a given NF. Departing from the standard linear-probe protocol, we introduce a novel trainingfree, test-time optimization algorithm, visually depicted in Figure 1c. Instead of training an auxiliary classifier, our method directly leverages the model’s own loss landscape to perform classification, providing a more direct measure of its embedded semantic knowledge. Our initial evaluation using this framework confirms a critical weakness: standard NFs, despite their generative prowess, exhibit poor discriminative performance.

To address this deficiency, we propose reverse representation alignment, a novel strategy built upon two core components: representation alignment and the exploitation of the model’s reverse pass. We term the act of operating on the generative pathway ‘flowing backwards’. In this process, our method directly enforces semantic consistency by aligning the intermediate features during the true generative pass (zto-x), a concept uniquely enabled by the invertible architecture of NFs.

As conceptually illustrated in Figure 1d, our comprehensive experiments show that this reverse alignment strategy is remarkably effective. It not only yields substantial improvements in generative quality as measured by FID but also, when assessed with our test-time method, unlocks a dramatic increase in classification accuracy. In summary, our main contributions are:

- • We design and systematically evaluate several alignment strategies for NFs, culminating in our proposed reverse representation alignment (R-REPA) method that uniquely leverages the model’s invertibility.
- • We propose a novel training-free, test-time optimization method for NF-based classification (Figure 1c), serving as both a diagnostic tool and a more intrinsic evaluation metric.
- • We demonstrate empirically that our approach significantly enhances both the generative and discriminative performance of NFs (Figure 1d), establishing a new stateof-the-art for synergizing these two capabilities.

### 2 Related Work

Normalizing Flows. NFs are exact likelihood-based generative models (Dinh, Krueger, and Bengio 2014; Rezende and Mohamed 2015). While historically surpassed by diffusion models (Ma et al. 2024; Wang et al. 2024, 2025b,a) in sample quality, recent work has revitalized the field. TARFlow (Zhai et al. 2024) introduces a Transformerbased architecture that achieves state-of-the-art likelihoods and generates samples with quality comparable to diffusion models. JetFormer (Tschannen, Pinto, and Kolesnikov 2024) and FARMER (Zheng et al. 2025) leverage an NF as a core, jointly trained component within a unified autoregressive model for high-fidelity joint image-text generation, eliminating the need for pre-trained autoencoders. Our concurrent work STARFlow (Gu et al. 2025) successfully scales NFs in terms of both model capacity and task complexity. These works demonstrate the renewed potential of NFs when integrated with modern architectures.

Representation Alignment for Generation. A recent paradigm for accelerating generative model training is Representation Alignment, which leverages features from pretrained vision foundation models (VFMs) as guidance. The seminal REPA (Yu et al. 2024) introduced a loss to align the internal hidden states of a denoising network with VFM features, drastically improving convergence and final sample quality. This simple yet powerful principle has proven highly effective and was quickly extended. Subsequent work has used it to enable stable end-to-end training of the entire latent diffusion pipeline (Lee, Park, and Kim 2024), to improve the VAE’s latent space directly (Zheng et al. 2024), and to adapt the alignment strategy to other backbones like U-Nets (Tian et al. 2024b), demonstrating its broad utility.

### 3 Preliminary: Normalizing Flow and TARFlow

Normalizing Flows (NFs) (Rezende and Mohamed 2015; Dinh, Krueger, and Bengio 2014; Dinh, Sohl-Dickstein, and Bengio 2016) represent a class of generative models based on exact likelihood built on the formula for change of variables. Given a continuous data distribution pdata for inputs x ∈ RD, a Normalizing Flow learns a parametric, invertible transformation fθ : RD → RD. This function maps complex data x to a simple, tractable base distribution p0 (e.g., a standard Gaussian), often referred to as the noise or latent space. The model is trained by maximizing the loglikelihood (MLE) of the data:

Ex∼p

max

data

θ

∂fθ(x) ∂x

log p0(fθ(x)) + log det

, (1)

where the first term encourages mapping data points to highdensity regions of the prior p0, while the log-determinant of the Jacobian term penalizes excessive local volume shrinkage, ensuring the transformation remains bijective. A generative model is obtained by the inverse transformation fθ−1, with a sampling procedure x = fθ−1(z), where z ∼ p0(z).

A prominent and computationally efficient variant is the Autoregressive Flow (AF) (Kingma et al. 2016; Papamakar-

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

- Figure 2: Selected Samples on ImageNet 256 × 256 from L-TARFlow + R-REPA. We use classifier-free guidance equal to 2.0.

ios, Murray, and Pavlakou 2017). In an AF, the transformation fθ is directly defined by a pair of parametergenerating functions (µθ,σθ), which specify an elementwise affine map. Crucially, these functions are architecturally constrained to be autoregressive: the parameters for each dimension d are computed using only the preceding input dimensions x<d. The forward (encoding) and inverse (sampling) processes for each dimension d ∈ [1,D] are:

zd = (xd − µθ(x<d)) ⊙ σθ(x<d)−1, xd = µθ(x<d) + σθ(x<d) ⊙ zd.

(2)

This autoregressive structure ensures the Jacobian of fθ is triangular, greatly simplifying the log-determinant term in

Eq. 1 to a simple sum: − Dd=1 log σθ(x<d).

Recently, TARFlow (Zhai et al. 2024) was introduced as a high-performance NF architecture for image data, building upon the AF framework. TARFlow is constructed by stacking multiple Transformer AutoRegressive Blocks (TARBlocks), z = fθT ◦ ··· ◦ fθ1(x), where each block fθt processes its input using a different autoregressive ordering πt. By alternating these orderings, the stacked blocks can capture dependencies across all dimensions. The parameters µ and σ for each block are modeled using causal Transformer layers. Assuming the final output z = xT follows a standard Gaussian prior, the end-to-end training objective becomes:

T

D

- 1

- 2∥z∥22 −

log σθt(xtπ−t1

) , (3)

Ex∼p

data −

max

θ

<d

t=1

d=1

where xt = fθt(xt−1) and x0 = x. Additionally, TARFlow employs noise augmented training and score-base denoising

to improve the modeling capability.

### 4 Methodology

In this section, we present our proposed methods for enhancing and leveraging Normalizing Flows. We begin by introducing a novel, training-free classification algorithm that directly utilizes the learned probability density from class-conditional NFs. Subsequently, we describe a representation alignment algorithm specifically designed to improve the discriminative quality of the latent representations

Algorithm 1: Training-Free Classification via Single-Step Gradient.

Require: Image x, pre-trained model fθ, class embeddings E ∈ RK×D

emb. Ensure: Predicted class label ypred.

- 1: Initialize logits λ ← 0 ∈ RK.
- 2: Compute weighted class embedding:
- 3: p ← softmax(λ)
- 4: eeff ← pTE
- 5: Compute the log-likelihood score:
- 6: L(λ) ← log p(x | eeff;θ)
- 7: Compute the gradient with respect to the logits:
- 8: g ← ∇λL(λ)
- 9: Predict the class corresponding to the largest gradient component:
- 10: ypred ← argmaxk(g)k
- 11: return ypred

within NFs. Finally, to address the computational challenges of modeling high-dimensional data, we adapt the TARFlow architecture to operate within a compressed latent space, enabling efficient generation.

#### 4.1 Training-Free Classification with NFs

We introduce a novel, training-free classification algorithm that leverages the density estimation capabilities of a pretrained class-conditional TARFlow model. Instead of training a separate classifier, our method reframes classification as an inference-time optimization problem. The core idea is to find the class label y that maximizes the conditional loglikelihood log p(x|y;θ) for a given input image x, where θ are the frozen parameters of the generative model.

We achieve this by estimating the gradient of the loglikelihood with respect to a“soft” class conditioning. This is done by first defining a set of classification logits, λ ∈ RK, which control a weighted average of the model’s class embeddings E. The gradient of the log-likelihood is then computed with respect to these logits at their initialization point. The class with the largest gradient component is selected

stop-grad

|REPAlign|
|---|

TARBlock

TARBlock

TARBlock

AlignLoss

MLP

Attention

AlignLoss

TARBlock

TARBlock

TARBlock

Attention

AlignLoss

Attention

TARBlock

TARBlock

TARBlock

Pretrained Visual Encoder

Attention

TARBlock

TARBlock

TARBlock

Block

REPA for Block a) Forward REPA b) Detach REPA c) Reverse REPA

- Figure 3: An overview of our Representation Alignment (REPA) mechanism. Left: Intermediate features from a TARFlow block are projected by an MLP and aligned with features from a pre-trained visual encoder. Right: The three gradient backpropagation strategies explored: (a) Forward REPA (F-REPA), updating all preceding blocks; (b) Detach REPA (D-REPA), updating only the current block; and (c) Reverse REPA (R-REPA), which leverages the inverse (generative) computational graph to update all subsequent blocks. While we depict alignment at a single location for clarity, this mechanism can be applied concurrently across multiple layers.

- as the prediction, as it indicates the direction of greatest increase in likelihood. The entire process requires only a single forward and backward pass through the model and is detailed in Algorithm 1.

#### 4.2 Representation Alignment

While the MLE objective excels at density modeling, the learned intermediate features of a NF are not inherently optimized to be semantically meaningful. To address this, we introduce a feature alignment mechanism that injects highlevel semantic guidance from a powerful, pre-trained vision model into the TARFlow’s generative process.

As illustrated in Figure 3 (left), we use a pre-trained, frozen vision encoder Φ(·) to extract a semantic representation v = Φ(x) ∈ RP×D from an input image x, where P is the number of patches and D is the embedding dimension. The objective is to align TARFlow’s intermediate features with this target representation y.

Let h(t,l) be the feature map from layer l of block t within TARFlow. We project these features into the semantic space using a learnable head, Projϕ (a simple MLP). The alignment loss then maximizes the patch-wise similarity between the projected and target features:

P

[p]

1 P

L(alignt,l)(θ,ϕ) := −

sim v[p], Projϕ h(t,l)

,

p=1

(4) where p is the patch index and sim(·,·) is a similarity function, such as cosine similarity. This alignment can be flexibly applied to any set of layers A = {(t1,l1),...}.

Crucially, we explore three distinct strategies for backpropagating the gradient of this alignment loss, each manip-

ulating the computational graph to control how the parameters are updated by the alignment loss. These strategies are visualized in Figure 3 (right).

Forward Strategy. As the most direct approach, this strategy involves backpropagating the gradient of L(alignt,l) through the forward computational graph. As illustrated in Figure 3 (a), this updates both the projector ϕ and all parameters of the TARFlow layers preceding layer (t,l).

Detach Strategy. This strategy draws an analogy to diffusion models, treating each TARFlow block as a network operating at a specific timestep t. To isolate the alignment process to this single ‘timestep’, we detach the input to the block. Consequently, the gradient only updates the parameters within that block (i.e., θt) and the projector ϕ, preventing any influence on preceding blocks (Figure 3, b).

Reverse Strategy. This novel strategy fundamentally alters the update mechanism by leveraging the computational graph of the reverse (generative) process. Specifically, we first compute the latent variable z = fθ(x) via the forward pass and then detach it. A new computational graph is then constructed by executing the inverse flow fθ−1, starting from the detached latent zdetached. The alignment loss is computed within this inverse pass. Crucially, backpropagation from this loss occurs entirely on the generative graph, inherently confining gradient updates to the parameters of layers subsequent to the alignment layer (t,l) (relative to the original forward pass). Figure 3 (c) conceptualizes this, showing how the stop-gradient on z reroutes the gradient path exclusively through the generative pathway.

TARFlow linear probe +R-REPA linear probe

70

TARFlow ours +R-REPA ours

60

| |
|---|

Accuracy(%)

| |
|---|

50

| |
|---|

| |
|---|

40

| |
|---|

| |
|---|

30

| |
|---|

| |
|---|

20

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

| |
|---|

10

| |
|---|

| |
|---|

| |
|---|

(1,2) (1,8) (2,6) (3,4) (4,2) (4,8) (5,6) (6,4) (7,2) (7,8) (8,6)

Location (Block, Layer)

- Figure 4: Validation of our proposed classification metric against the standard linear probing protocol. The plot shows classification accuracy of our metric and standard linear probing.

Final Loss Formulation. The total training objective is a weighted sum of the NF loss and the averaged alignment losses from all chosen layers in the set A:

  1

 , (5)

L(alignt,l)(θ,ϕ)

Ltotal = LNF + λalign

|A|

(t,l)∈A

where λalign is a hyperparameter balancing the two terms. The gradient computation for Lalign follows one of the three strategies outlined above.

#### 4.3 Moving to Latent Space

To scale our method to high-resolution synthesis, we transition from modeling pixels directly to modeling the latent space of a pre-trained Variational Autoencoder (VAE). This established strategy allows us to offload the task of low-level perceptual compression. Our primary work thus focuses on the core challenge: applying the powerful density estimation and refinement techniques of TARFlow to the compact and semantically-rich latent codes.

The training process is adapted to this latent space. For a clean latent vector x obtained from the VAE encoder, our NF model, fθ, is trained on a noisy version x˜:

x˜ = x + ϵ, where ϵ ∼ N(0,σ2I). (6) The model thus learns to estimate the density pθ(˜x).

The generative process mirrors this by first sampling from this learned noisy distribution and then applying a denoising step. Specifically, a noisy latent sample x˜ is generated via the inverse transformation, x˜ = fθ−1(z), where z ∼ N(0,I) is a sample from the base distribution. This sample is subsequently refined using the score-based denoising procedure:

xˆ = x˜ + σ2∇x˜ log pθ(˜x). (7)

Finally, the refined latent vector xˆ is decoded into the final image, effectively scaling the precise likelihood modeling and powerful sample refinement of TARFlow to the highresolution domain.

### 5 Experiments

Dataset and Task. We conduct our class-conditional image generation and classification experiments on the ImageNet-1K dataset (Deng et al. 2009). Our models are trained exclusively on the training set and evaluated at two distinct resolutions: 64 × 64 and 256 × 256.

Evaluation Metrics. To assess the performance of our generative model, we employ a standard suite of metrics to measure sample fidelity and diversity: Fr´echet inception distance (FID) (Heusel et al. 2017), sFID (Nash et al. 2021), and Inception Score (IS) (Salimans et al. 2016). Following TARFlow, we sample 4096 images for evaluation. Besides, we report results at the optimal CFG scale for each model, determined via a grid search detailed in Figure 5.

To evaluate the discriminative ability, we measure the classification accuracy on the ImageNet-1K validation set. This is achieved using our proposed test-time optimization classification. Figure 4 confirms the validity of our approach. We compare our single-score evaluation (dashed lines) with the layer-specific results from standard linear probing (solid lines). While linear probing produces a wide range of outcomes depending on the layer, our method’s score consistently reflects the overall performance ceiling for both TARFlow and +R-REPA models. This demonstrates that our method is a valid and efficient alternative to the expensive process of layer-by-layer linear evaluation.

Implementation Details. For experiments at the 64 × 64 resolution, our model architecture strictly adheres to the design of TARFlow (Zhai et al. 2024). Specifically, the model is composed of 8 TARBlocks. Each block, in turn, contains 8 layers of causal attention. The channel dimension is set to 1024, and the model operates on non-overlapping image patches of size 4 × 4.

For the higher resolution of 256 × 256, we first leverage a pre-trained VAE-ft-EMA (Esser, Rombach, and Ommer 2021) to compress images into a lower-dimensional latent space. Our generative model then operates on this latent representation. The transformer architecture is enhanced with two key components: Rotary Position Embeddings (RoPE) (Su et al. 2024) and the SwiGLU activation function (Touvron et al. 2023a,b). To maintain consistent patch number with 64 × 64, we use a patch size of 2×2. All other hyperparameters, including the number of TARBlocks (8), layers per block (8), and channel dimension (1024), are kept consistent with the 64 × 64 configuration.

#### 5.1 Ablation Studies

In this section, we conduct a series of ablation studies to systematically determine our optimal model configuration. We first investigate the core design of the REPA mechanism (its backpropagation strategy, location, and depth), and then analyze key hyperparameters for training and sampling. Unless otherwise specified, all ablations are performed on ImageNet 64×64 at 400k training iterations.

Backpropagation Strategy. As shown in Table 1, the choice of backpropagation strategy proves to be critical. We find the Forward strategy consistently degrades sFID (e.g.,

50

Forward 1&2

TARFlow

15.0

Detach 1&2 Detach 7&8 Reverse 7&8

- 12

- 13

- 14

- 15

- 16

+ R-REPA

| |
|---|

FID-4096

FID-4096

###### FID-4096

25

| |
|---|

14.5

| |
|---|

| |
|---|

| |
|---|

15 12 11

14.0

| |
|---|

| |
|---|

3.3x faster

| |
|---|

| |
|---|

| |
|---|

| |
|---|

10 20 40 60 120 200 Training Epoch

1.5 2.0 2.5 3.0 3.5 4.0 4.5 CFG Scale

0.10 0.15 0.20 0.25 0.30  oise Standard Deviation

- Figure 5: Hyperparameters Ablations and Training Convergence. Left: CFG search results on ImageNet 64×64. The Reverse REPA strategy applied to later model blocks yields the best performance across various CFG scales. Center: Ablation of noise standard deviation on latent space. We identify an optimal noise standard deviation of 0.20. Right: Reverse REPA improves sample fidelity and accelerates training convergence by 3.3× on ImageNet 64×64.

Type Blocks Layers FID↓ sFID↓ IS↑ Acc.(%)↑ TARFlow (Zhai et al. 2024) 12.91 33.79 36.62 37.43

- Setup Group 1: Alignment applied to All blocks Forward All 2 12.25 37.97 40.85 46.97 Detach All 2 12.19 34.31 41.98 49.06 Reverse All 2 12.21 33.80 42.08 49.91

- Setup Group 2: Alignment applied to selected 2 blocks Forward 1 & 2 2 12.67 39.99 41.11 61.16 Detach 1 & 2 2 12.73 33.84 37.81 61.63 Detach 7 & 8 2 12.12 34.00 41.18 55.14 Reverse 7 & 8 2 11.93 33.78 40.90 55.21

- Setup Group 3: Ablation of alignment layers Reverse 7 & 8 2 11.93 33.78 40.90 55.21 Reverse 7 & 8 4 11.84 33.61 46.06 58.91 Reverse 7 & 8 6 11.71 33.68 44.31 57.35

Table 1: Ablation study of our proposed alignment method on ImageNet 64×64, evaluated at 400k training iterations. We vary the alignment Type and which blocks and layers to apply it to. Best results for each metric are in bold.

from 33.79 to 37.97). We hypothesize that this is because its unconstrained gradient flow creates a tension between the MLE objective and the alignment loss in the model’s early blocks. Forcing these foundational blocks, which are likely specialized for low-level spatial statistics, to also conform to high-level semantics proves detrimental to overall sample coherence. The ‘Detach’ strategy improves upon this by localizing updates. However, the ‘Reverse’ strategy is demonstrably superior, achieving a better FID (11.93 vs. 12.12) in direct comparison. By updating only the generative path (fθ−1), it effectively refines synthesis without disrupting the core density model.

Alignment Block and Layer. The choice of which blocks to align creates a trade-off between generative quality and

Model Res. Iter. FID-4096↓ Acc.(%)↑

TARFlow 64 1M 11.76 39.97 +R-REPA (Ours) 64 400K 11.71 57.76 +R-REPA (Ours) 64 600K 11.53 57.75 +R-REPA (Ours) 64 800K 11.48 57.65 +R-REPA (Ours) 64 1M 11.25 57.02

Latent-TARFlow 256 400K 13.82 45.97 Latent-TARFlow 256 1M 13.05 40.22

+R-REPA (Ours) 256 400K 13.26 57.85 +R-REPA (Ours) 256 1M 12.79 56.24

Table 2: Training progress comparison. Our method, +RREPA, consistently outperforms the vanilla TARFlow baselines across different checkpoints on both ImageNet 64×64 and 256×256 resolutions.

feature semantics. Aligning Blocks 7 & 8, which are the first to operate on the latent variable z during the generative process, yields the best FID (11.93). Guiding these initial synthesis steps helps establish a strong high-level image structure. Conversely, aligning Blocks 1 & 2 produces the best semantic accuracy (61.63%). These blocks are the first to process the input image x during encoding, so aligning them directly optimizes feature extraction but harms FID by constraining the delicate final synthesis steps.

Finally, within our best-performing setup (R-REPA, aligning 7 & 8 blocks), we found that guiding deeper layers leads to better fidelity. As shown in Group 3, the FID progressively improves from 11.93 to 11.71 as we move alignment from the 2nd to the 6th layer. This suggests that guiding more refined features closer to a block’s output provides a more potent refinement signal. While an intermediate (4th) layer achieves the best IS (46.06), the deepest (6th) layer is the clear choice for maximizing the final image quality.

Classifier-Free Guidance (CFG) Scale. We optimized the CFG scale, a critical hyperparameter, for each model

###### Model FID↓ sFID↓

Diffusion Models / Flow Matching EDM (Karras et al. 2022) 1.36 – iDDPM (Nichol and Dhariwal 2021) 2.92 3.79 ADM (Dhariwal and Nichol 2021) 2.09 4.29

Generative Adversarial Networks (GANs) IC-GAN (Casanova et al. 2021) 6.70 – BigGAN (Brock, Donahue, and Simonyan 2018) 4.06 3.96

Consistency Models (CMs)

CD (LPIPS) (Song et al. 2023) 4.70 – iCT-deep (Song and Dhariwal 2023) 3.25 –

Normalizing Flows

TARFlow (Zhai et al. 2024)† 4.21 5.34 + R-REPA (Ours) 3.69 4.34

Table 3: Image generation results on class-conditional ImageNet 64×64. We report FID and sFID with 50K samples. †Result obtained using their officially released codebase.

to ensure a fair comparison. As shown in Figure 5, our final configuration (‘Reverse 7&8’) achieves the lowest FID while also demonstrating robustness across a wide scale range, with an optimum near 3.1. All results are reported

- at their respective optimal scales.

Latent Noise Std. For the 256×256 latent-space model, the standard deviation (σ) of noise added to VAE latents (Eq. 6) is crucial. We ablated this value, with results in Figure 5. A clear performance peak exists at σ = 0.20, which we use for all latent-space experiments.

#### 5.2 Main Results

Equipped with the optimal R-REPA configuration from our ablations, we now evaluate its performance against baselines and state-of-the-art models on ImageNet at 64×64 and 256×256 resolutions.

Performance against Baselines. We first compare our method, R-REPA, directly against the vanilla TARFlow baselines. As shown in Table 2, our method provides substantial improvements in both sample quality (FID) and learned feature discriminability (Accuracy).

On ImageNet 64×64, our final model achieves an FID of 11.25 and an accuracy of 57.02%, significantly outperforming the baseline’s 11.76 FID and 39.97% accuracy. The faster convergence of discriminative accuracy over generative quality indicates that the model learns high-level semantics early in training before progressively refining finegrained details for synthesis. This efficiency is a crucial advantage of our approach and leads to accelerated training (3.3×), as illustrated in Figure 5. Quantitatively, our model at just 400k iterations already surpasses the fully-trained (1M iter.) baseline in both FID (11.71 vs. 11.76) and, most notably, accuracy (57.76% vs. 39.97%).

A similar leap is observed on the 256×256 latent-space task, where FID improves from 13.05 to 12.79 and accuracy

Model FID↓ sFID↓ IS↑

Diffusion Models

ADM (Dhariwal and Nichol 2021) 4.59 5.25 186.70 CDM (Ho et al. 2022) 4.88 – 158.71 LDM-4 (Rombach et al. 2022) 3.60 7.51 247.67 DiT (Peebles and Xie 2023) 2.27 4.60 278.24 SiT (Ma et al. 2024) 2.06 4.50 270.30

Autoregressive (discrete)

RQ-Trans. (Lee et al. 2022) 3.80 – 323.7 LlamaGen-3B (Sun et al. 2024) 2.18 – 263.33 VAR (Tian et al. 2024a) 1.73 – 350.2

Autoregressive (continuous)

MAR-AR (Li et al. 2024) 4.69 – 244.6 MAR (Li et al. 2024) 1.55 – 303.7 DART (Gu et al. 2024) 3.82 – 263.8

Normalizing Flow

Latent-TARFlow 5.15 6.78 243.49 +R-REPA (Ours) 4.95 6.89 234.99 +Patch Size 1 (Ours) 4.18 4.96 240.8

Table 4: Class-conditional generation on ImageNet 256×256. We report FID, sFID, and IS with 50K samples. Lower is better for ↓, higher is better for ↑.

jumps from 40.22% to 56.24%. This demonstrates that representation alignment not only enhances final performance but also provides a more efficient training signal, leading to superior models in significantly less time.

Generation on ImageNet 64×64. As shown in Table 3, our R-REPA strategy delivers state-of-the-art generative performance for Normalizing Flows on the class-conditional ImageNet 64×64 benchmark. Our method significantly improves upon the strong TARFlow baseline, reducing the FID from 4.21 to 3.69 and the sFID from 5.34 to 4.34. This performance not only surpasses established GANs like BigGAN (FID 4.06) but also brings flow-based models into closer competition with powerful diffusion models such as iDDPM (FID 2.92). Crucially, this top-tier result is achieved with just two sampling steps, highlighting the exceptional inference efficiency of our method compared to the multistep iterative process required by competing paradigms.

Generation on ImageNet 256×256. We further test the scalability of our approach on the challenging ImageNet 256×256 benchmark by operating in the latent space of a pre-trained VAE. As presented in Table 4, our method again demonstrates remarkable effectiveness. Our optimized configuration, which combines R-REPA with an architectural adjustment to a 1x1 patch size, achieves a highly competitive FID of 4.18 and sFID of 4.96—a substantial improvement over the baseline. Even the direct application of R-REPA provides a clear boost, reducing FID to 4.95.

Most importantly, these competitive high-resolution results are achieved while preserving the two-step sampling efficiency. This demonstrates that the benefits of our approach are not confined to smaller scales but are robust and

scalable. By delivering high-fidelity results with a minimal computational budget, our work establishes REPA-enhanced Normalizing Flows as a compelling and highly efficient paradigm for high-resolution image synthesis.

### 6 Conclusion

In this work, we introduce R-REPA, a novel training strategy that enhances the semantic awareness of NFs. It leverages their unique invertibility to enforce semantic consistency directly on the generative (z-to-x) pass, thereby unlocking the powerful synergy between representation learning and generation inherent in the architecture. The empirical results are compelling. R-REPA establishes a new stateof-the-art for NFs on ImageNet by delivering simultaneous gains in generative fidelity (FID) and classification accuracy over the strong TARFlow baseline. This accuracy gain is rigorously quantified by our novel training-free classification algorithm—a more intrinsic probe of the model’s learned semantics. Furthermore, our method demonstrates robust high-resolution scalability while also dramatically boosting training efficiency by over 3.3×. Ultimately, our work establishes a powerful new principle for advancing NFs: that fostering a virtuous cycle between semantic representation and the generative process is a direct and effective route to higher fidelity.

### Acknowledgements

This work is supported by the National Key R&D Program of China (No. 2022ZD0160900), the Natural Science Foundation of Jiangsu Province (No. BK20250009), the Collaborative Innovation Center of Novel Software Technology and Industrialization, Alibaba Group through Alibaba Innovative Research Program.

### Appendix

This appendix provides supplementary material to support the main paper. We begin in Section A with a comprehensive overview of the algorithmic details of our proposed RREPA method. This includes a review of the foundational TARFlow block computation, a detailed explanation of our accelerated implementation which is critical for efficient training, and an in-depth analysis of the gradient flow that highlights the fundamental differences between our reversepass alignment and alternative forward-pass strategies. In Section B, we outline our experimental setup, covering training, sampling, classification procedures, and the computational resources used. Section C presents additional ablation studies that further validate our design choices, including the selection of alignment layers and the alignment loss weight. Finally, in Section D, we offer additional visualizations to provide further qualitative insight into the performance of our model.

### A Algorithm Details for R-REPA

#### A.1 Review of the TARFlow Block Computation

We begin by briefly recapping the core computational process of a single TARFlow block, fθ, which forms the basis of our model. This block transforms an input tensor

Algorithm 2: Accelerated R-REPA Training Step

- 1: Input: Image tensor x, model parameters θ = {θt}Tt=1, alignment layer set A.
- 2: Hyperparameters: Alignment loss weight λalign.
- 3: Pre-trained Models: Frozen vision encoder Φ, learnable projector ϕ.

- # — 1. Forward Pass and Caching —

4: Initialize empty list for cached features: cached ← [ ] 5: x0 ← x 6: for t = 1 to T do 7: cached.append(stop gradient(xt−1)) 8: xt ← fθt(xt−1) ▷ Standard forward block 9: end for

10: z ← xT 11: LNF ← −log p(z) − Tt=1 log |detJft

θ

|

- # — 2. Accelerated Pseudo-Reverse Pass —

12: Initialize alignment loss: Lalign ← 0 13: Target features y ← Φ(x)

14: zT ← stop gradient(z) 15: for t = T down to 1 do 16: xˆt−1 ← cached[t − 1]

17: ▷ Compute pseudo-inverse using ztrev and xˆt−1 18: zt−1,{h(t,l)}l ← (fθt)−accel1 (zt,cond = xˆt−1) 19: for each layer l ∈ At do 20: Lalign ← Lalign − sim(Projϕ(h(revt,l)),y) 21: end for 22: end for

- # — 3. Total Loss and Parameter Update —

- 23: Ltotal ← LNF + λalignLalign
- 24: Update θ and ϕ using the gradient ∇θ,ϕLtotal

x ∈ RB×D×C to an output tensor z of the same shape, where B is the batch size, D is the sequence length (i.e., number of tokens), and C is the channel dimension.

As described in the main paper, the transformation is autoregressive with respect to the sequence dimension. For each token index d ∈ [1,D], the forward (encoding) and inverse (sampling) operations are defined as:

zd = (xd − µθ(x<d)) ⊙ σθ(x<d)−1 xd = µθ(x<d) + σθ(x<d) ⊙ zd

(8)

Here, xd and zd denote the tensors for the d-th token, and x<d represents all preceding tokens {x1,...,xd−1}.

A key computational property, which is central to our subsequent analysis, is that the forward pass (encoding) is parallelizable because all input tokens {xd} are known simultaneously. In contrast, the inverse pass (sampling) is inherently sequential, as generating xd requires prior computation of all x<d.

#### A.2 Accelerated R-REPA Implementation

The core concept of our Reverse Representation Alignment (R-REPA) is to apply an alignment loss during the generative (reverse) pass of the TARFlow model. However, a naive implementation of the reverse pass, x = fθ−1(z), is computationally prohibitive for training. Each inverse block

(fθt)−1 is autoregressive, meaning the computation for each token (xt−1)d depends on the previously generated tokens (xt−1)<d. This inherent sequentiality makes parallel computation on GPUs inefficient and creates a severe training bottleneck.

To overcome this, we introduce an accelerated implementation specifically for calculating the R-REPA loss during training. This method cleverly utilizes features cached during the standard forward pass to break the sequential dependency of the reverse pass.

The procedure is as follows. During the standard forward pass, which computes the sequence of transformations x0 → x1 → ··· → xT = z (where x0 = x), we cache the input to each block, xt−1. Crucially, we detach these cached tensors from the computational graph using a ‘stop gradient’ operation, denoted as xˆt−1 = stop gradient(xt−1).

Next, we construct a “pseudo-reverse” pass to compute the alignment loss. We start with the final latent variable, zT = stop gradient(z). Then, for each block t from T down to 1, we perform a modified inverse operation. Instead of conditioning on sequentially generated outputs, we use the corresponding pre-computed and cached tensor xˆt−1 to provide the autoregressive context. This allows the inverse transformation for all tokens to be computed in parallel:

ztd−1 = µtθ(xˆt<d−1) + σθt(xˆt<d−1) ⊙ ztd (9)

While the output of this operation, zt−1, is numerically identical to the cached xˆt−1 due to the block’s invertibility, this formulation creates a valid computational graph. The R-

REPA loss, L(alignt,l), is calculated using intermediate features h(revt,l) from this pseudo-reverse pass.

The gradient from Lalign backpropagates through the parameters of block t (θt) and, via the dependency on zt, continues to flow to all subsequent blocks (θt+1,...,θT). The ‘stop gradient’ on the conditioning tensor xˆt−1 prevents gradients from affecting any preceding blocks (θ1,...,θt−1). This achieves the exact update pattern required for R-REPA while transforming the reverse pass into a fully parallelizable computation. The entire process is detailed in Algorithm 2.

#### A.3 Computational Efficiency

Our accelerated implementation for R-REPA is highly effective in practice. Compared to a naive approach that faithfully executes the sequential autoregressive computation, our parallelized method yields a substantial speedup of approximately 50×. Furthermore, it reduces the peak memory footprint by nearly 50%, making the reverse-pass alignment computationally feasible for deep models.

We also provide a direct comparison of the training throughput for our final R-REPA strategy against the Forward and Detach variants. Table 5 details the performance under an identical experimental setup where two blocks are aligned (corresponding to Group 2 in Table 1 of the main paper, note that align block location does not influence speed of Detach-REPA).

As shown in Table 5, the Forward strategy is the fastest, as it only requires adding a loss to the existing forward com-

##### Strategy Aligned Blocks Speed (iter/s)

Forward-REPA 2 2.11 Detach-REPA 2 1.93 Reverse-REPA (Ours) 2 1.87

Table 5: Training throughput comparison of different REPA strategies. The accelerated R-REPA introduces only a minor overhead compared to other methods, making it highly practical. The results are tested on the same devices.

putational graph. Both the Detach and our accelerated Reverse strategies introduce a minor computational overhead due to graph manipulation (detaching inputs or constructing the pseudo-reverse pass), resulting in a slight reduction in throughput. However, this modest decrease in speed is a small price for the significant improvements in generative quality, as demonstrated in the main paper. This makes our accelerated approach a highly practical and effective training method.

#### A.4 Detailed Gradient Flow Analysis for Forward-REPA and R-REPA

To elucidate the fundamental difference between the Forward-REPA and our proposed Reverse-REPA (R-REPA) strategies, we analyze the gradient backpropagation pathways in a simplified two-block TARFlow model. This analysis reveals how each strategy directs the alignment supervision to different parts of the model.

Toy Model Setup. Consider a model composed of two sequential TARFlow blocks, fθ1 and fθ2, with parameters θ1 and θ2, respectively. The forward (encoding) process is x f

1

2

−→θ x1 f

−→θ z. We apply alignment at both blocks by extracting intermediate features h1 from fθ1 and h2 from fθ2. The corresponding alignment losses, L1align and L2align, are computed by comparing the projected features against a target y = Φ(x) via a projector Projϕ. The total alignment loss is Lalign = L1align + L2align.

Analysis of Forward-REPA. In the Forward strategy, the alignment losses are computed on the standard forward computational graph. The gradient flow is as follows:

- • Gradient from L2align: The gradient ∇L2align updates the projector parameters ϕ. Since h2 is a function of the parameters θ2 and the input x1, the gradient backpropagates

to update θ2. Critically, because x1 = fθ1(x), the gradient continues to flow through x1 to update the parameters of

the preceding block, θ1. Thus, ∇L2align affects {ϕ,θ2,θ1}.

- • Gradient from L1align: Similarly, ∇L1align updates ϕ and the parameters of the first block, θ1. The gradient path terminates here as it does not depend on any subsequent blocks. Thus, ∇L1align affects {ϕ,θ1}.

The core principle of Forward-REPA is that an alignment loss at a given layer updates the parameters of that layer and

Hyperparameter Value Batch size 256 Optimizer AdamW Betas (β1,β2) (0.9, 0.95) Learning Rate 1e-4 Weight Decay 1e-4 EMA Decay Rate 0.9999 λalign 0.1 sim(·,·) cos. sim

Table 6: Key hyperparameters for model training.

all preceding layers in the computational graph. This constitutes a form of causal correction, adjusting upstream computations to improve downstream feature representations.

Analysis of R-REPA. R-REPA fundamentally alters the gradient pathway by constructing a new computational graph based on the generative (inverse) process, as detailed in Algorithm 2.

- • Gradient from L2align: This loss is computed during the pseudo-reverse pass using features from

(fθ2)−accel1 (zdetached,cond = xˆ1). The gradient updates ϕ and θ2. However, the ‘stop gradient’ operation on the conditioning input xˆ1 explicitly severs the computational

graph, preventing the gradient from flowing back to θ1. Thus, ∇L2align affects only {ϕ,θ2}.

- • Gradient from L1align: This loss is computed using fea-

tures from (fθ1)−accel1 (x1rev,cond = xˆ). The gradient updates ϕ and θ1. Crucially, the input to this operation, x1rev, is the output of the previous reverse step, i.e., x1rev = (fθ2)−1(...). Therefore, a dependency on θ2 exists, and the gradient flows “upward” through the generative chain to also update θ2. Thus, ∇L1align affects {ϕ,θ1,θ2}.

The core principle of R-REPA is that an alignment loss at a given layer updates the parameters of that layer and all subsequent layers (relative to the original forward pass). This provides generative guidance, optimizing how higherlevel abstract representations (from later blocks) are transformed into lower-level, semantically-aligned features (by earlier blocks).

Summary of Differences. R-REPA’s reverse flow mechanism ensures that semantic guidance is injected in a manner that aligns with the model’s generative direction, optimizing from abstract latents toward concrete data, which we posit is key to its enhanced performance.

### B More Experimental Details

#### B.1 Training Setup

We train our models using the AdamW optimizer and maintain an exponential moving average (EMA) of the model weights for evaluation. Consistent with TARFlow (Zhai et al. 2024), our data augmentation pipeline includes a resize, center crop and a random horizontal flip. For the RREPA component, we follow the precedent set by prior work

Strategy Blocks Layer FID ↓ sFID ↓ Acc (%) ↑

Detach All 2nd 12.19 34.31 49.00 Detach All 4th 12.14 33.97 54.14 Detach All 6th 12.37 34.32 54.98

Table 7: Additional ablation study of the alignment layer for the Detach strategy, applied to all blocks. This analysis supplements Table 1 from the main paper, showing the limited impact of layer choice in this specific setting, and further demonstrating the advantages of Reverse strategy. All metrics are evaluated on ImageNet 64×64 at 400K iterations.

on representation alignment (Yu et al. 2024). Specifically, we employ a frozen DINOv2-B (Oquab et al. 2023) as the pre-trained visual encoder to provide the target representations. The learnable projector, which maps TARFlow’s internal features to the DINOv2 feature space, is a 3-layer MLP with SiLU activation functions and a hidden dimension of 1024. The key hyperparameters used for training are summarized in Table 6. These settings are consistent for imagenet 64×64 and 256×256. As for the VAE, we take the off-shelf VAE-ft-EMA with a downsample factor of 8 and channel 4 from huggingface (AI 2022).

#### B.2 Sampling Setup

For generating samples, our procedure strictly follows the methodology established in TARFlow (Zhai et al. 2024). To ensure deterministic and reproducible sampling for evaluation, which minimizes variance from stochasticity, we employ a fixed random seed. The initial noise tensor z ∼ N(0,I) is generated on the CPU before being transferred to the GPU for the inverse transformation process.

#### B.3 Classification Setup

Our training-free classification is performed as described in Algorithm 1 of the main paper. We investigated the sensitivity of the classification outcome to several implementation choices, including multi-step optimization instead of a single gradient step, various early stopping criteria, and different learning rates for the logits. We found the final classification prediction to be remarkably robust to these variations. This robustness stems from the core principle of the algorithm: to identify the class that corresponds to the direction of the steepest ascent in the conditional log-likelihood. A single gradient computation is sufficient to determine this direction effectively, making further optimization steps or complex heuristics largely redundant for the final prediction.

#### B.4 Computational Resources

All experiments were conducted with 8 NVIDIA GPUs. The ablation studies, which were run for 400,000 iterations, required approximately 2.5 days to complete for the ImageNet 64×64 models and 3.5 days for the 256×256 latent-space models. The full training for our final models, which extends for 2.5 times as many iterations (1 million total), took proportionally longer, requiring approximately 6-7 days for the 64×64 models and 8-9 days for the 256×256 models.

[Figure 31]

###### Figure 6: Additional samples on ImageNet 256 × 256 from L-TARFlow + R-REPA with PS=2, CFG=2.0.

Encoder FID ↓ sFID ↓ IS ↑ Acc ↑ CLIP-B 12.41 33.99 41.99 38.99 MAE-B 12.32 34.00 39.69 37.57 DINOv2-B 11.71 33.68 44.31 57.35 DINOv2-L 11.74 33.87 45.00 58.80

Table 8: Quantitative comparison using different encoders with R-REPA. The baseline FID (12.91) represents the model without R-REPA. Arrows indicate whether a higher (↑) or lower (↓) value is better.

### C Additional Experimental Results

In this section, we provide additional ablation studies that supplement the results presented in the main paper. These experiments further justify our hyperparameter choices and reinforce the core findings of our work.

#### C.1 Further Ablation on Alignment Layer for the Detach Strategy

To further investigate the design choices presented in Table 1 of the main paper, we conducted an additional ablation study on the alignment layer depth specifically for the Detach strategy. In this setup, we applied representation alignment to all TARBlocks and varied the internal layer used for alignment.

As shown in Table 7, varying the alignment layer from the 2nd to the 6th results in only minor fluctuations in generative quality, with FID scores remaining within a narrow range (12.14 to 12.37). While deeper layers (4th and 6th) show a slight improvement in classification accuracy, the overall impact on sample fidelity (FID, sFID) is not substantial. This suggests that within this specific Detach configuration, the model’s performance is not highly sensitive to the precise layer chosen for alignment. This observation validates our initial design choice in Table 1 of using the 2nd layer as a consistent setting for comparing different strategies, as it simplifies the experimental setup without significantly biasing the results for the Detach method.

Furthermore, it is important to highlight that even the best-performing Detach configuration from this ablation (FID 12.14, Acc 54.98%) is outperformed by our proposed Reverse strategy (FID 11.71, Acc 57.35% with layer 6th, as shown in Table 1 of the main paper). This comparison reinforces the superiority of leveraging the generative pathway for representation alignment.

#### C.2 Ablation on Encoder Architectures

To evaluate the robustness of our proposed R-REPA method, we conduct an ablation study using various pre-trained vision encoders. Following our final setting, R-REPA is applied to layer 6 of blocks 7 and 8. We replace the default encoder(DINOv2-B) with several popular models: CLIP-B, MAE-B, and DINOv2-L. The quantitative results are presented in Table 8.

The experiment highlights two critical points:

Model λalign FID ↓ Baseline 0 13.82 Latent-TARFlow + R-REPA 0.1 13.26 Latent-TARFlow + R-REPA 0.5 13.74

Table 9: Ablation study on the alignment loss weight λalign on ImageNet 256×256 with 400K iterations. The results jus-

tify our choice of λalign = 0.1 in the main experiments.

- 1. Robustness across encoders: R-REPA demonstrates strong performance and robustness across all tested encoder architectures. Each configuration significantly outperforms the baseline FID of 12.91, indicating that our method is a general-purpose enhancement not limited to a specific encoder.
- 2. Correlation between generation and discrimination: The results reveal a strong positive correlation between generation quality (FID) and the discriminative power of the features (Acc). Encoders that yield better accuracy, such as DINOv2, also achieve the best FID scores. This observation empirically supports our motivation of improving generation quality by enhancing the discriminative features of the underlying model.

#### C.3 Ablation on Alignment Loss Weight λalign

We analyzed the sensitivity of R-REPA to the alignment loss weight, λalign, on the ImageNet 256×256 benchmark. This hyperparameter, defined in Equation 5 of the main paper, balances the standard Normalizing Flow objective (LNF) with our proposed representation alignment loss (Lalign).

Table 9 presents the FID scores for λalign values of 0.1 and 0.5. The results clearly indicate that a smaller weight of 0.1 yields a better FID (13.26) compared to a larger weight of 0.5 (13.74). This suggests that while the alignment provides a crucial semantic signal, an overly strong alignment term can interfere with the primary density estimation task, slightly degrading the final sample fidelity. Based on this finding, we selected λalign = 0.1 as the optimal value for all main experiments, a choice that is now empirically validated by this ablation.

### D Additional Visualizations

In this section, we provide further visual evidence of our model’s generative capabilities at both high and low resolutions.

Figure 6 displays additional, randomly selected samples from our Latent-TARFlow + R-REPA model on the ImageNet 256×256 benchmark. To generate these images, we used a classifier-free guidance (CFG) scale of 2.0. The model itself operates on a 2×2 patch representation within the VAE latent space, demonstrating its capability to synthesize diverse and high-fidelity images.

In a similar vein, Figure 7 presents samples for the ImageNet 64×64 resolution. These were generated by our TARFlow + R-REPA model, which processes the input as a sequence of 4×4 image patches, employing a CFG scale

[Figure 32]

Figure 7: Samples on ImageNet 64 × 64 from TARFlow + R-REPA with PS=4, CFG=2.5.

of 2.5. The results confirm the model’s ability to produce outputs with consistent quality and semantic coherence.

### Ethical Statement

We acknowledge the ethical considerations associated with the development and application of powerful generative models. Our work, which focuses on enhancing Normalizing Flows for high-fidelity image generation, carries implications that merit careful consideration.

Potential for Misuse. Our research contributes to the advancement of high-fidelity image generation. Like all powerful generative technologies, the methods presented herein have a dual-use nature. While intended for positive applications such as creative content generation, data augmentation, and advancing fundamental machine learning research, we recognize the potential for misuse. This includes the creation of synthetic media for malicious purposes, such as generating misinformation or deceptive content. We advocate for the responsible development and deployment of generative models and encourage the broader research community to continue developing robust detection and mitigation strategies for such misuse.

Bias and Fairness. Our models are trained on the ImageNet-1K dataset and leverage features from a DINOv2 model, which was pre-trained on a large corpus of web data. It is well-documented that such large-scale datasets may contain societal biases, including demographic and stereotypical representations. Consequently, our model may inherit and potentially amplify these biases. The generated outputs may not be representative of all populations and could reflect the biases present in the training data. We urge users of our models and methods to be critically aware of these limitations and to conduct fairness assessments before deployment in any real-world application.

Data and Model Licensing. The datasets (ImageNet-1K) and pre-trained models (DINOv2, VAE-ft-EMA) used in this work are publicly available and were used in accordance with their respective licenses, which permit academic

research. Our work does not involve the use of personally identifiable or sensitive private data.

Computational Resources and Environmental Impact. The training of our models required significant computational resources, as detailed in Appendix B.4. We are mindful of the environmental impact associated with largescale machine learning experiments. A key contribution of our work is the demonstration of a more efficient training paradigm for Normalizing Flows, which accelerates convergence by over 3.3×. This improvement directly contributes to reducing the overall computational cost and associated energy consumption required to achieve state-of-the-art performance, representing a step towards more sustainable AI research.

Our primary goal is to advance the scientific understanding of generative models. We release our findings to the research community to foster further innovation and to encourage a transparent and critical discussion about the capabilities and societal implications of these technologies.

### References

AI, S. 2022. sd-vae-ft-ema. https://huggingface.co/ stabilityai/sd-vae-ft-ema. Accessed: 2024-05-23.

Brock, A.; Donahue, J.; and Simonyan, K. 2018. Large scale GAN training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096.

Casanova, A.; Careil, M.; Verbeek, J.; Drozdzal, M.; and Romero Soriano, A. 2021. Instance-conditioned gan. Advances in Neural Information Processing Systems, 34: 27517–27529.

Deng, J.; Dong, W.; Socher, R.; Li, L.-J.; Li, K.; and FeiFei, L. 2009. ImageNet: A Large-Scale Hierarchical Image Database. In CVPR.

Dhariwal, P.; and Nichol, A. 2021. Diffusion models beat gans on image synthesis. Advances in Neural Information Processing Systems, 34: 8780–8794.

Dinh, L.; Krueger, D.; and Bengio, Y. 2014. Nice: Nonlinear independent components estimation. arXiv preprint arXiv:1410.8516.

Dinh, L.; Sohl-Dickstein, J.; and Bengio, S. 2016. Density estimation using real nvp. arXiv preprint arXiv:1605.08803.

Esser, P.; Rombach, R.; and Ommer, B. 2021. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 12873–12883.

Gu, J.; Chen, T.; Berthelot, D.; Zheng, H.; Wang, Y.; Zhang, R.; Dinh, L.; Bautista, M. A.; Susskind, J.; and Zhai, S. 2025. STARFlow: Scaling Latent Normalizing Flows for High-resolution Image Synthesis. arXiv preprint arXiv:2506.06276.

Gu, J.; Wang, Y.; Zhang, Y.; Zhang, Q.; Zhang, D.; Jaitly, N.; Susskind, J.; and Zhai, S. 2024. Dart: Denoising autoregressive transformer for scalable text-to-image generation. arXiv preprint arXiv:2410.08159.

Heusel, M.; Ramsauer, H.; Unterthiner, T.; Nessler, B.; and Hochreiter, S. 2017. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30.

Ho, J.; Saharia, C.; Chan, W.; Fleet, D. J.; Norouzi, M.; and Salimans, T. 2022. Cascaded Diffusion Models for High Fidelity Image Generation. J. Mach. Learn. Res., 23: 47–1.

Karras, T.; Aittala, M.; Aila, T.; and Laine, S. 2022. Elucidating the design space of diffusion-based generative models. arXiv preprint arXiv:2206.00364.

Kingma, D. P.; Salimans, T.; Jozefowicz, R.; Chen, X.; Sutskever, I.; and Welling, M. 2016. Improved variational inference with inverse autoregressive flow. Advances in neural information processing systems, 29.

Lee, D.; Kim, C.; Kim, S.; Cho, M.; and Han, W.-S. 2022. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 11523–11532.

Lee, S.-H.; Park, S.; and Kim, G.-M. 2024. REPA-E: Endto-End Training of Latent-Diffusion Models via Representation Alignment. In arXiv preprint arXiv:2405.18373.

Li, T.; Tian, Y.; Li, H.; Deng, M.; and He, K. 2024. Autoregressive Image Generation without Vector Quantization. arXiv preprint arXiv:2406.11838.

Ma, N.; Goldstein, M.; Albergo, M. S.; Boffi, N. M.; Vanden-Eijnden, E.; and Xie, S. 2024. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. arXiv preprint arXiv:2401.08740.

Nash, C.; Menick, J.; Dieleman, S.; and Battaglia, P. W. 2021. Generating Images with Sparse Representations. In Proceedings of the 38th International Conference on Machine Learning (ICML).

Nichol, A. Q.; and Dhariwal, P. 2021. Improved denoising diffusion probabilistic models. In International Conference on Machine Learning, 8162–8171. PMLR.

Oquab, M.; Darcet, T.; Moutakanni, T.; Vo, H.; Szafraniec, M.; Khalidov, V.; Fernandez, P.; Haziza, D.; Massa, F.; ElNouby, A.; et al. 2023. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193.

Papamakarios, G.; Murray, I.; and Pavlakou, T. 2017. Masked Autoregressive Flow for Density Estimation. In Guyon, I.; von Luxburg, U.; Bengio, S.; Wallach, H. M.; Fergus, R.; Vishwanathan, S. V. N.; and Garnett, R., eds., Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, 2338– 2347.

Peebles, W.; and Xie, S. 2023. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 4195–4205.

Rezende, D.; and Mohamed, S. 2015. Variational Inference with Normalizing Flows. In Bach, F.; and Blei, D., eds., Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, 1530–1538. Lille, France: PMLR.

Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Ommer, B. 2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 10684– 10695.

Salimans, T.; Goodfellow, I.; Zaremba, W.; Cheung, V.; Radford, A.; and Chen, X. 2016. Improved techniques for training gans. Advances in neural information processing systems, 29.

Song, Y.; and Dhariwal, P. 2023. Improved techniques for training consistency models. arXiv preprint arXiv:2310.14189.

Song, Y.; Dhariwal, P.; Chen, M.; and Sutskever, I. 2023. Consistency Models. arXiv preprint arXiv:2303.01469.

Su, J.; Ahmed, M.; Lu, Y.; Pan, S.; Bo, W.; and Liu, Y. 2024. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568: 127063.

Sun, P.; Jiang, Y.; Chen, S.; Zhang, S.; Peng, B.; Luo, P.; and Yuan, Z. 2024. Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation. arXiv preprint arXiv:2406.06525.

Tian, K.; Jiang, Y.; Yuan, Z.; Peng, B.; and Wang, L. 2024a. Visual autoregressive modeling: Scalable image generation via next-scale prediction. arXiv preprint arXiv:2404.02905. Tian, Y.; Wang, X.; Li, S.; Wu, Z.; Ye, B.; Zheng, Y.; Liu, B.; Li, J.; and Zhou, J.-R. 2024b. U-REPA: A U-Net based representation alignment framework for accelerating diffusion model training. In arXiv preprint arXiv:2405.16642.

Touvron, H.; Lavril, T.; Izacard, G.; Martinet, X.; Lachaux, M.-A.; Lacroix, T.; Rozi`ere, B.; Goyal, N.; Hambro, E.; Azhar, F.; et al. 2023a. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Touvron, H.; Martin, L.; Stone, K.; Albert, P.; Almahairi, A.; Babaei, Y.; Bashlykov, N.; Batra, S.; Bhargava, P.; Bhosale, S.; et al. 2023b. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Tschannen, M.; Pinto, A. S.; and Kolesnikov, A. 2024. JetFormer: An autoregressive generative model of raw images and text. arXiv preprint arXiv:2411.19722.

Wang, S.; Gao, Z.; Zhu, C.; Huang, W.; and Wang, L. 2025a. PixNerd: Pixel Neural Field Diffusion. arXiv:2507.23268.

Wang, S.; Li, Z.; Song, T.; Li, X.; Ge, T.; Zheng, B.; and Wang, L. 2024. Exploring DCN-like architecture for fast image generation with arbitrary resolution. In The Thirtyeighth Annual Conference on Neural Information Processing Systems.

Wang, S.; Tian, Z.; Huang, W.; and Wang, L. 2025b. DDT: Decoupled Diffusion Transformer. arXiv:2504.05741.

Yu, S.; Kwak, S.; Jang, H.; Jeong, J.; Huang, J.; Shin, J.; and Xie, S. 2024. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940.

Zhai, S.; Zhang, R.; Nakkiran, P.; Berthelot, D.; Gu, J.; Zheng, H.; Chen, T.; Bautista, M. A.; Jaitly, N.; and Susskind, J. 2024. Normalizing flows are capable generative models. arXiv preprint arXiv:2412.06329.

Zheng, G.; Zhao, Q.; Yang, T.; Xiao, F.; Lin, Z.; Wu, J.; Deng, J.; Zhang, Y.; and Zhu, R. 2025. FARMER: Flow AutoRegressive Transformer over Pixels. arXiv:2510.23588.

Zheng, Y.; Tian, Y.; Li, S.; Wu, Z.; Liu, B.; Li, J.; Ye, B.; and Zhou, J.-R. 2024. LightningDiT: A Vision-FoundationModel-Aligned VAE for Fast and High-Quality Generation. In arXiv preprint arXiv:2405.15438.

