# arXiv:2407.11633v3[cs.CV]9Sep2024

## Scaling Diffusion Transformers to 16 Billion Parameters

Zhengcong Fei, Mingyuan Fan, Changqian Yu Debang Li, Junshi Huang∗

Kunlun Inc. Beijing, China

{feizhengcong}@gmail.com

### Abstract

In this paper, we present DiT-MoE, a sparse version of the diffusion Transformer, that is scalable and competitive with dense networks while exhibiting highly optimized inference. The DiT-MoE includes two simple designs: shared expert routing and expert-level balance loss, thereby capturing common knowledge and reducing redundancy among the different routed experts. When applied to conditional image generation, a deep analysis of experts specialization gains some interesting observations: (i) Expert selection shows preference with spatial position and denoising time step, while insensitive with different class-conditional information; (ii) As the MoE layers go deeper, the selection of experts gradually shifts from specific spacial position to dispersion and balance. (iii) Expert specialization tends to be more concentrated at the early time step and then gradually uniform after half. We attribute it to the diffusion process that first models the low-frequency spatial information and then high-frequency complex information. Based on the above guidance, a series of DiT-MoE experimentally achieves performance on par with dense networks yet requires much less computational load during inference. More encouragingly, we demonstrate the potential of DiT-MoE with synthesized image data, scaling diffusion model at a 16.5B parameter that attains a new SoTA FID-50K score of 1.80 in 512×512 resolution settings. The project page: https://github.com/feizc/DiT-MoE.

### 1 Introduction

Recently, diffusion models [42, 89, 90, 9] have emerged as powerful deep generative models in various domains, such as image [19, 44, 79], video [45, 62, 88, 43, 60], 3D object [58, 71, 72] and so on [95]. This advancement is attributed to diffusion models’ ability to learn denoising tasks over diverse noise distributions, effectively transforming random noise into a target data distribution through iterative denoising processes. In particular, Transformer-based structure shows that increasing network capacity with additional parameters generally boosts performance [10, 69, 30, 32]. For example, Stable Diffusion 3 [24] as the competitive diffusion models to date, some with over 8B parameters. However, training and serving such models is expensive [67]. This is partially because these deep networks are typically dense, i.e., every example is processed using every parameter, thereby, scale comes at a high computational cost.

Conditional computation [4, 3] is a promising scaling technique, which aims to enhance model capacity while maintaining relatively constant training and inference cost by applying only a subset of parameters to each example. In fields of NLP, sparse mixture of experts (MoE) are becoming increasingly popular [86, 14, 15] as a practical implementation that employs a routing mechanism to

∗Corresponding author

Preprint. Under review.

|[Figure 1]| |[Figure 2]| |[Figure 3]|[Figure 4]|
|---|---|---|---|---|---|
| | | | |[Figure 5]|[Figure 6]|
|[Figure 7]| |[Figure 8]|[Figure 9]|[Figure 10]| |
| | |[Figure 11]|[Figure 12]| | |
|[Figure 13]|[Figure 14]|[Figure 15]| |[Figure 16]| |
|[Figure 17]|[Figure 18]| | | | |

- Figure 1: DiT-MoE model achieve state-of-the-art image quality. We show selected samples generated from our class-conditional XL/2-8E2A (left) and G/2-16E2A (right) models trained on ImageNet at 512×512 and 256×256 resolution, respectively.

control computational costs [61]. Moreover, the applications of MoE architectures in Transformers have yielded successful attempts at scaling language models to a substantial size with remarkable performance [21, 25, 55, 100]. Conventional MoE architectures in Transformers typically substitute the Feed-Forward Network (FFN) with MoE layers, each consisting of multiple experts that are structurally identical to a standard FFN. We along with a similar sparse design and investigate its effectiveness in diffusion Transformers [69, 59].

In this work, we explore conditional computation tailored specifically for Diffusion Tranformers (DiT) [69] at scale. We propose DiT-MoE, a sparse variant of the DiT architecture for image generation. The DiT-MoE replaces a subset of the dense feedforward layers in DiT with sparse MoE layers, where each token of image patch is routed to a subset of experts, i.e., MLP layers. Moreover, our architecture involves two principal designs: shared part of experts to capture common knowledge and balance expert loss to reduce redundancy in different routed experts. We also provide a comprehensive analysis to demonstrate that these designs offer opportunities to train a parameter-efficient MoE diffusion model while some interesting phenomena about expert routing from different perspectives are observed.

Starting from a small-scale model, we validate the benefits of DiT-MoE architecture and present an effective recipe for the scale training of DiT-MoE. We then conduct an evaluation of class-based

Noise

Output Hidden

AdaLN

| | | |
|---|---|---|
| | | |

MoE

Linear and Reshape

Layer Norm

Shared Expert Expert

Expert Expert Expert

Expert

AdaLN

N

DiT-MoE Block

Multi-Head Self-Attention

Patchify Embed

Router

MLP

AdaLN

Timestep Label

Noised Latent

Input Hidden

Input Tokens Conditioning

- Figure 2: Overview of the DiT-MoE architecture. Generally, DiT-MoE is built upon the DiT and composed of MoE-inserted Transformer blocks. In between, we replace the MLP with a sparsely activated mixture of MLPs. The right subfigure demonstrates the details of MoE layer integration of the shared expert strategy.

image generation in the ImageNet benchmarks. Experiment results indicate that DiT-MoE matches the performance of state-of-the-art dense models, while requiring less time to inference. Alternatively, DiT-MoE-S can match the cost of DiT-B while achieving better performance. Leveraging with additional synthesis data, we subsequently scale up the model parameters to 16.5B while only activating 3.1B parameters, which attains a new state-of-the-art FID-50K score of 1.80 in 512×512 resolution. Our contributions can be summarized as follows:

- • MoE for diffusion transformers. We present DiT-MoE, a sparsely-activated diffusion Transformer model for image synthesis. In between, it incorporates simple and effective designs, including shared components of experts to capture common knowledge, and an auxiliary expert-level balance loss to minimize redundancy among routed experts.
- • Expert routing analysis. We have conducted statistics on the selection of experts in different scenarios and found interesting observations about expert selection preference with spatial position and denoising time step at different MoE layers, which can effectively guide future network design and interpretability.
- • Model parameters at scale. We introduce a series of DiT-MoE models and show that these models can be stably trained, and seamlessly used for efficient inference. More encouragingly, we further undertake a preliminary endeavor that DiT-MoE can performed and scale beyond 16B with well-selected synthesised data.
- • Performance and inference. We show that DiT-MoEs strongly outperform their dense counterparts on conditional image generation tasks at the ImageNet benchmark. At inference time, the DiT-MoE models can be flexible to match the performance of the largest dense model while using as little as half of the amount of computation. Finally, we publicly release the code and trained model checkpoint.

### 2 Methodology

We first briefly describe diffusion models and network conditional computation with MoEs. We then present how we apply this methodology to diffusion transformers, and explain our design choices for optimizing expert routing algorithms. Finally, we provide computation analysis with different parameter scaling settings.

#### 2.1 Preliminaries

Diffusion models. Diffusion models [42, 89] constitute a class of generative models that simulate a gradual noising and denoising process through a series of latent variables. They are characterized by a Markovian forward process and a learned reverse process. Specifically, the forward diffusion

process incrementally adds noise to an input image x0, transitioning it through a sequence of states x1,...,xT according to a predetermined variance schedule β1,...,βT. The reverse process, learned during training, aims to recover the original data from its noised version. The forward noising process is defined as:

q(xt|x0) = N(√αt,(1 − αt)I) = √αtx0 + (1 − αt)ϵ, (1) where αt+βt = 1 and ϵ ∼ N(0,I) is the Gaussian noise. Diffusion models are trained to estimate the reverse process, pθ(xt−1|xt), by approximating the variational lower bound of pθ(x0:T|xt)d(x0:T) as computed by [89]. In practice, this reverse process is generally conditioned on the timestep t and aims to either predict the noise ϵ or reconstruct the original image x0. Formally, a noise prediction network ϵθ(xt,t) is incorporated by minimizing a noise prediction objective, i.e., minθ Et,x

0,ϵ||ϵ −

ϵθ(xt,t)||22, where t is uniformly distributed between 1 and T. To learn conditional diffusion models, e.g., class-conditional [19] or text-to-image [76, 6] models, additional condition information is integrated into the noise prediction objective as:

0,c,ϵ||ϵ − ϵθ(xt,t,c)||22, (2) where c can be the condition index or its continuous embedding.

Et,x

min

θ

Conditional computation with MoEs. Conditional computation seeks to activate subsets of a neural network depending on the input [4, 3]. A mixture-of-experts model exemplifies this concept by assigning different model experts to various regions of the input space [48]. We follow the framework of [86], who present a mixture of experts layer in deep learning, comprising E experts, defined as:

E

g(x)iei(x), (3)

MoE(x) =

i=1

where x ∈ RD is the input to the layer, ei : RD → RD denotes the function computed by expert i, and g : RD → RE is the routing function that determines the input-conditioned weights for the

experts. Both ei and g are parameterized by neural networks. As originally defined, this structure remains a dense network. However, if g is sparse, i.e., restricted to assign only k ≪ E non-zero weights, then unused experts need not be computed. This approach enables super-linear scaling of the number of model parameters relative to the computational cost of inference and training.

#### 2.2 MoEs for Diffusion Transformers

Here we explore the application of sparsity to diffuion models within the context of the Diffusion Transformers (DiT) [69]. DiT has demonstrated superior scalability across various parameter settings, achieving enhanced generative performance compared to CNN-based U-Net architectures [80, 23] with higher training computation efficiency. Similar to vision transformers [20], DiT processes images as a sequence of patches. An input image is first divided into a grid of equal-sized patches. These are linearly projected to features identical to the model’s hidden dimension. After adding positional embeddings, the patch embeddings, i.e., image patch tokens, are processed by a sequence of Transformer blocks, which consists predominately of alternating self-attention and MLP layers. The standard MLPs consist of two layers and a GeLU [39] non-linearity:

MLP(x) = W2σgelu(W1x), (4) For DiT-MoE, we replace a subset of these with MoE layers, where each expert is an MLP; see

- Figure 2 for viewing. The experts share the same architecture and it follows a similar design pattern as [78, 15, 21].

On top of the generic MoE architecture, we introduce extra designs to exploit the potential of expert specialization. As illustrated in the right subfigure 2, our architecture incorporates two principal strategies: shared expert routing and expert load balance loss. Both of these strategies are designed to optimize the level of expert specialization and introduction as below:

Shared expert routing. Under conventional routing strategies, tokens assigned to different experts may require access to overlapping knowledge or information. Consequently, multiple experts may converge in acquiring this shared knowledge within their respective parameters, leading to parameter redundancy. Referring to [15, 75], we incorporate additional ns experts to serve as shared experts. That is, regardless of the original router module, each image patch token will be deterministically assigned to these shared experts.

- Table 1: Scaling law model size. The model sizes, detailed hyperparameters settings, and inference burden for MoE scaling experiments.

Total param. Activate param. #Blocks L Hidden dim. D #Head n Gflops

S/2-8E2A 199M 71M 12 384 6 15.43 S/2-16E2A 369M 71M 12 384 6 15.44 B/2-8E2A 795M 286M 12 768 12 61.68 L/2-8E2A 2.8B 1.0B 24 1024 16 219.26 XL/2-8E2A 4.1B 1.5B 28 1152 16 323.74 G/2-16E2A 16.5B 3.1B 40 1408 16 690.94

Expert-level balance loss. Directly learned routing strategies often encounter the issue of load imbalance, leading to significant performance defects [85]. To address this, we introduce an expertlevel balance loss, calculated as follows:

T

n

n KT

Lbalance = α

t=1

i=1

1 T

I(t,i)

1

P(t,i), (5)

t=1

where α is expert-level balance factor, T is the length of image patch sequence, I(t,i) denotes the indicator function that image token t selects expert i and P(t,i) is the probability distribution of token t for expert.

#### 2.3 Computation Analysis

In DiT-MoE, some of the MLPs are replaced by MoE layers, which helps increase the model capacity while keeping the activated number of parameters, and thus compute efficiency. Formally, the MoE modules are applied to MLPs every e layer. When using MoE, there are n possible experts per layer, with a router choosing the top K experts and shared ns experts at each image patch token. This design allows DiT-MoE to optimize various properties by adjusting n, K, and e. Specifically, increasing n enhances model capacity at the cost of higher memory usage, while increasing K raises the number of active parameters and computational requirements. Conversely, increasing e reduces model capacity but also decreases both computation and memory requirements, along with communication dependencies. Various configurations of DiS are delineated in Table 1. They cover a wide range of total model sizes and flop allocations, from 199M to 16.5B, thus affording comprehensive insights into the scalability performance. Aligned with [69], Gflop metric is evaluated in 256×256 size image generation with patch size p = 2, checked with both calflops and torchprofile python package. We set e = 1 by default. The model is named according to their configs and patch size p; for instance, DiT-MoE L/2-8E2A refers to the Large version config, p = 2, n = 8, and K = 2.

### 3 Experiments

In this section. we begin by outlining our experimental setups in Section 3.1. Next, we present the experimental results of different DiT-MoE design spaces in Section 3.2, and provide a detailed routing analysis. Then, we provide comparative results with diffusion models in Section 3.3. Finally, we explore further scaling model with synthesized data and show some impressive cases.

#### 3.1 Experimental Settings

Datasets. Following settings [69] for class-conditional image generation task, we utilize ImageNet [17] datatset at resolutions of 256×256 and 512×512, which comprises 1,281,167 training images across 1,000 different classes. The only data augmentation is horizontal flips. We train 500K, 1M, and 7M iterations at both resolutions, with a batch size of 1024, respectively. For the synthesis training data, we use open-source text-to-image models including SDXL [70] and SD3-Medium [24] to create approximately 5M different 512×512 images according to the given tag label. Specifically, we use the prompt template “[image class], in a natural and realistic style.” to create images with different seeds and filters with top CLIP similarity [73]. Finally, we construct a mixed training image set with a real-to-synthesis ratio of 1:5.

[Figure 19]

- Figure 3: Ablation experiments on ImageNet dataset at 256×256 resolution. We report FID metrics on 50K generated samples without CFG. (a) Incorporation of shared expert routing can accelerate the training as well as optimize generated results. (b) Number of experts and (c) model parameters scaling. As we expected, increasing the expert number and the model size can consistently improve the generation performance. However, directly changing the share experts number influences the results marginally.

[Figure 20]

[Figure 21]

[Figure 22]

- Figure 4: Training loss curves for in small version for different elements. (a) The effect of expert balance loss. We can see that 0.005 can achieve best performance. (b) The effect of experts number with same activate experts. We can see that increase expert number results consistently performance enhancement. (c) The influence of MoE layer replacement. If only half of MLP are replace with MoE layer, we can see that MoE in deep layer results in better performance, which is consisent with experted in visualization. Note that Figures are experimented in rectifide flow training.

Implementation details. We use the AdamW optimizer [50] without weight decay across all datasets, maintaining a constant learning rate of 1e-4. In line with [69], we utilize an exponential moving average of DiT-MoE weights over training with a decay of 0.9999. All results were reported using the EMA model. Our models are trained on Nvidia A100 GPU. When trained on ImageNet dataset at different resolutions, we adopt classifier-free guidance [41] following [79] and use an offthe-shelf pre-trained variational autoencoder (VAE) model [52] from Stable Diffusion [79] available in huggingface2. The VAE encoder has a downsampling factor of 8. We retrain diffusion hyperparameters from [69], using a tmax = 1000 linear variance schedule ranging from 1 × 10−4 to 2 × 10−2 and parameterization of the covariance. We set the share experts number ns to 2 and the expert-level balance factor α to 0.005 by default.

Evaluation metrics. We measure image generation performance with Fréchet Inception Distance (FID) [40], a widely adopted metric for assessing the quality of generated images. We follow convention when comparing against prior works and report FID-50K using 250 DDPM sampling steps [66] following the process of [19]. We additionally report Inception Score [83], sFID [63] and Precision/Recall [53] as secondary metrics.

#### 3.2 Model Design Analysis

In this section, we ablate on the ImageNet dataset with a resolution of 256×256, evaluate the FID score on 50K generated samples following [2, 30], and identify the optimal settings.

2https://huggingface.co/stabilityai/sd-vae-ft-ema

[Figure 23]

- Figure 5: Frequency for selected experts per image class. We show the 12 MoE layers of DiTMoE-S/2-8E2A. The x-axis corresponds to the 8 experts in a MoE layer. The y-axis is the 1000 ImageNet classes. For each pair (expert e, image class i), we show the average routing frequency for the patches corresponding to all generated images with class i that particular expert e. The darker the color, the higher the frequency of selection. The larger the layer number, the deeper the MoE layers.

Effect of shared expert routing. To assess the impact of the shared expert routing strategy, we conducted an ablation study by removing the shared expert while maintaining the same number of activated parameters as in the conventional expert routing approach, and trained the model from scratch. As illustrated in Figure 3 (a), the results indicate that incorporating an additional shared expert enhances performance across most steps compared to conventional expert routing. These findings support the hypothesis that the shared expert strategy facilitates better knowledge disentangling and contributes to improved MoE model performance.

Optimal share expert number. We then examine the optimal number of shared experts at scale. Using the small version of MoE-DiT, which comprises 16 total experts, we maintain the number of activated experts at 2 and experimented with incorporating 1, 2, and 4 shared experts. As depicted

- in Figure 3 (b), we can find that varying the ratio of shared experts to routed experts does not significantly affect performance. Considering the trade-off between memory usage and performance, we standardize the number of shared experts to 2 when scaling up DiT-MoE.

Effect of expert-level balance loss. As a import part of balance expert loading, we range from 0.5, 0.01 to 0.05, 0.005. We also dropout the expert balance loss, i.e., set to zero. As piloted in Figure

- 4 (a), as we expert, increase the expert-level balance loss leads to a performance first improve then decrease. The optimal factor is 0.005. Meantime, we leave how to dynamically adjust the factor according to current training situation in future work.

Influence of increasing expert number. We directly increase the expert number from 8 to 16, while keeping the number of activated parameters fixed at 2. As reported in Figure 3 (b), the adjustment leads to consistently improved generative performance, albeit with a significant increase in GPU memory consumption. On the other hand, the loss curve in Figure 4(b) also demonstrates that incorporation of MoE in text to image tasks can achieve competitive performance and helps to faster loss convergence.

Scaling model size. We also explore scaling properties of DiT-MoE by examining the effect of model depth, i.e., number of blocks, hidden dimension, and head number. Specifically, we train four variants of DiT-MoE model, spanning configurations from Small to XL, as detailed in Table 1, and denoted as (S, B, L, XL) for simple. As shown in Figure 3 (c), the performance improves as the depth increase from 12 to 28. Similarly, increasing the width from 384 to 1152 yields performance gains. Overall, across all configurations, impressive improvements in the FID metric are observed throughout all training stages by augmenting the depth and width of the model architecture.

[Figure 24]

- Figure 6: Frequency for selected experts per image patch position. We show the 12 MoE layers of DiT-MoE-S/2-8E2A. The x-axis corresponds to the 8 experts in a MoE layer. The y-axis are the 256 patches in ImageNet images with 322 × 322 = 256 sequence length of patch size 2, at 256×256 resolution with VAE compression 8. For each pair (expert e, image patch id i), we show the average routing frequency for all the patches with patch-id i that were assigned to that particular expert e.

[Figure 25]

- Figure 7: Frequency for selected experts per denoising time step. We show the 12 MoE layers of DiT-MoE-S/2-8E2A. The x-axis corresponds to the 8 experts in a layer. The y-axis are the 250 DDPM steps for sampling the synthesis image. For each pair (expert e, inference step i), we show the average routing frequency for all time step i that were assigned to that particular expert e.

#### 3.3 Expert Specialization Analysis

Although large-scale MoEs have led to strong performance [78, 15], it remains essential to explore the internal mechanisms of these complex models within the context of DiT. We posit that a thorough routing analysis can provide valuable insights for designing new algorithms. Specifically, we first sample 50 images for each image class with 250 DDPM steps, resulting in a total of 50K data points. We then calculate the frequency of expert selection from three perspectives: image class, spatial position, and denoising time step. The visualization heat maps are presented in Figures 5, 6, and 7, respectively. From our observations, several key insights emerge: (i) Generally, there is no obvious redundancy in the learned experts routing and each expert at a different MoE layer is routed sometimes. (ii) Expert selection shows a preference for spatial position and denoising step, but is less sensitive to class-conditional information, consistent with previous assumption [36]; (iii) As shown in Figure 5, no clear patterns or variations are evident in the expert routing mechanism for different class-conditional scenarios. (iv) As the MoE layers become deeper, expert selection transitions from specific positional preferences to a more dispersed and balanced distribution. For instance, in Figure 6, the heat map of MoE layer 0 the heat map for MoE layer 0 indicates a strong correlation between image patches and spatial clustering, whereas the heat map for MoE layer 9 shows a more uniform expert selection distribution. (v) As in Figure 7, during the early inference steps (e.g., steps less than 50), expert choices are more concentrated, while in later steps (e.g., steps greater than 100), the

- Table 2: Benchmarking class-conditional image generation on ImageNet 256×256 dataset. We can see that DiT-MoE-XL/2 achieves state-of-the-art FID metrics towards best competitors with less inference cost.

Class-Conditional ImageNet 256×256 Model FID↓ sFID↓ IS↑ Precision↑ Recall↑ GAN

BigGAN-deep [7] 6.95 7.36 171.4 0.87 0.28 StyleGAN-XL [84] 2.30 4.02 265.12 0.78 0.53 Diff. based on U-Net

ADM [19] 10.94 6.02 100.98 0.69 0.63 ADM-U 7.49 5.13 127.49 0.72 0.63 ADM-G 4.59 5.25 186.70 0.82 0.52 ADM-G, ADM-U 3.94 6.14 215.84 0.83 0.53 CDM [44] 4.88 - 158.71 - LDM-8 [79] 15.51 - 79.03 0.65 0.63 LDM-8-G 7.76 - 209.52 0.84 0.35 LDM-4 10.56 - 103.49 0.71 0.62 LDM-4-G 3.60 - 247.67 0.87 0.48 VDM++ [51] 2.12 - 267.70 - Diff. based on Transformer

U-ViT-H/2 [2] 2.29 5.68 263.88 0.82 0.57 DiT-XL/2 [69] 2.27 4.60 278.24 0.83 0.57 SiT-XL/2 [59] 2.06 4.50 270.27 0.82 0.59 Large-DiT-3B [34] 2.10 4.52 304.36 0.82 0.60 Large-DiT-7B [34] 2.28 4.35 316.20 0.83 0.58 LlamaGen-3B [91] 2.32 - 280.10 0.32 0.56 DiT-MoE-XL/2-8E2A 1.72 4.47 315.73 0.83 0.64

distribution becomes more uniform. In summary, these findings on expert routing can effectively inform future structural designs and enhance network interpretability.

Base on the above observation, we assume that the shallow layer in DiT is sparse, and more capacity is required in deep layer. Therefore, we try to replace the dense DiT with half of MoE layer. Specifically,

we replace with shallow half (0 ∼ N2 ) and deep half (N2 ∼ N) layers with MoE layers. The results

- in Figure 4(c) show that replacing MoE in deep layers achieve lower loss and better generative performance.

#### 3.4 Compare with State-of-the-arts

We present the evaluation results of conditional image generation for various metrics compared with dense competitors in Tables 2 and 3. On the class-conditional ImageNet 256×256 dataset, our DiTMoE-XL achieves an FID score of 1.72, surpassing all previous models with different architectures. Notably, DiT-MoE-XL, which activates only 1.5 billion parameters, significantly outperforms the Transformer-based competitors Large-DiT-3B, Large-DiT-7B, and LlamaGen-3B. This demonstrates the potential of MoE in diffusion models. On the class-conditional ImageNet 512×512 dataset, we observe similar advancements in nearly all evaluation metrics as expected.

#### 3.5 Scaling up DiT-MoE with Synthesis Data

Building on the previous expert routing analysis, finally, we test how well DiT-MoE can scale to a very large number of parameters, while continuing to improve performance. For this, we expand the size of the model into giant version, detailed hyper-parameter setting listed in Table 1, and use an extensive training dataset augmented with synthesis data. We train a 40-block DiT-MoE model, incorporating 32 total experts with 2 activate experts, resulting in a model with 16.5B parameters while keeping a prominent inference efficient. We successfully train DiT-MoE-G/2-16E2A, which is, as far as we are aware, the largest diffusion transformer model for class-condition image generation to date. It achieves an impressive state-of-the-art FID50K score of 1.80 at a 512×512 resolution at the ImageNet benchmark. Figure 1 showcases a selection of generated samples at different resolutions, demonstrating the high-quality image generation capacities of both DiT-MoE models.

- Table 3: Benchmarking class-conditional image generation on ImageNet 512×512 dataset. DiT-MoE demonstrates a promising performance compared with dense networks for diffusion.

Class-Conditional ImageNet 512×512 Model FID↓ sFID↓ IS↑ Precision↑ Recall↑ GAN

BigGAN-deep [7] 8.43 8.13 177.90 0.88 0.29 StyleGAN-XL [84] 2.41 4.06 267.75 0.77 0.52 Diff. based on U-Net

ADM [19] 23.24 10.19 58.06 0.73 0.60 ADM-U 9.96 5.62 121.78 0.75 0.64 ADM-G 7.72 6.57 172.71 0.87 0.42 ADM-G, ADM-U 3.85 5.86 221.72 0.84 0.53 VDM++ [51] 2.65 - 278.10 - -

Diff. based on Transformer

U-ViT-H/4 [2] 4.05 6.44 263.79 0.84 0.48 DiT-XL/2 [69] 3.04 5.02 240.82 0.84 0.54 Large-DiT-3B [34] 2.52 5.01 303.70 0.82 0.57 DiT-MoE-XL/2-8E2A 2.30 4.82 298.35 0.85 0.57

- 4 Related Works

Conditional computation. To increase the number of model parameters without a corresponding rise in computational costs, conditional computation [4, 12, 16] selectively activates only relevant parts of the model based on the input, similar to decision trees [57]. This dynamic adaptation of neural networks has been applicable to various deep learning tasks [5, 3, 18, 81, 37]. For instance, [94] propose dynamically combining a set of base convolution kernels based on input image features to enhance model capacity. Additionally, techniques in [93, 29, 26, 27] adjust the forward Transformer layers at the token level to expedite inference. For efficient deployment, [8, 96] dynamically alter the neural network architecture according to specified efficiency constraints, thereby optimizing the balance between efficiency and accuracy. In a similar vein, we employ the mixture-of-experts strategy, which utilizes a gating network to dynamically route inputs to various experts.

Mixture of experts. MoEs [48, 49, 11, 97] typically integrate the outputs of sub-models, or experts, through an input-dependent router, and have been successfully applied in diverse scenarios [47, 35, 92, 98]. In the field of NLP, [86] introduced top-k gating in LSTMs, incorporating auxiliary losses to maintain expert balance [38]. [55] extended to transformers, demonstrating substantial improvements in neural machine translation [87]. Recent advancements in large-scale language models [86, 55, 25] have enabled input tokens to select either all experts [22] or a sparse mixture, facilitating the scaling of language models to trillions of parameters [15]. [35] sped up pre-training with over one trillion parameters and one expert per input, outperforming dense baseline [74] with transfer and distillation benefits. [56] alternatively employed balanced routing via a linear assignment problem. In the domain of CV, [99, 68] combine CNN with MoE for robust image classification. [78, 82, 77] scale vision transformers with adaptive per-image computing, thereby reducing the computational burden by half compared to dense competitors.

MoEs for diffusion models. While previous studies predominantly utilize a single model to tackle denoising tasks across various timesteps [69, 32, 70, 31, 28, 46], several investigations have explored the deployment of multiple expert models, each specializing in a distinct range of timesteps [13]. PPAP [36] achieves this by training multiple classifiers on segmented timesteps, each employed for classifier guidance. e-DiffI [1] and ERNIE-ViLG [33] utilize a consistent set of denoisers across these experts, whereas MEME [54] advocates for distinct architectures tailored to each timestep segment. Additionally, [64] deals with routing problem of denoising tasks across timesteps and designs a routing strategies for learning these tasks in a single model. These methodologies enhance generative quality while maintaining comparable inference costs, albeit at the expense of increased memory requirements. They operate on the premise that the characteristics of denoising tasks vary across timesteps. We extend it by analyzing the expert routing mechanism and demonstrating that both temporal and spatial elements without class-conditional information influence different MoE

layers. The most similar to work is [65], which also explores experts routing, however, they focus on a form of multi-task learning for time step and not actually sparse, i.e., base version vs. dense version comes to 144M vs. 131M. In contrast, we delve into the time-space routing mechanism and modeling of >10B model size.

### 5 Conclusion

In this paper, we employ sparse conditional computation to train some of the largest diffusion transformer models, achieving efficient inference and substantial improvements in image generation tasks. Alongside DiT-MoE, we incorporate simple designs to facilitate the effective utilization of model sparsity in relation to inputs. We further provide a detailed analysis of the expert routing mechanism, demonstrating the characters of space-time preference for different MoE layers. This methodology aligns with recent analyses indicating that model sparsity is a highly promising strategy for reducing CO2 emissions associated with model training. Our work represents an initial exploration of large-scale conditional computation for diffusion models. Future extensions could involve training stable and faster, heterogeneous expert architectures and better knowledge distillation. We anticipate that the importance of sparse model scaling will continue to grow in multimodal generation.

### References

- [1] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, et al. ediff-i: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022.
- [2] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22669–22679, 2023.
- [3] Emmanuel Bengio, Pierre-Luc Bacon, Joelle Pineau, and Doina Precup. Conditional computation in neural networks for faster models. arXiv preprint arXiv:1511.06297, 2015.
- [4] Yoshua Bengio. Deep learning of representations: Looking forward. In International conference on statistical language and speech processing, pages 1–37. Springer, 2013.
- [5] Yoshua Bengio, Nicholas Léonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013.
- [6] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.
- [7] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096, 2018.
- [8] Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, and Song Han. Once-for-all: Train one network and specialize it for efficient deployment. arXiv preprint arXiv:1908.09791, 2019.
- [9] Hanqun Cao, Cheng Tan, Zhangyang Gao, Yilun Xu, Guangyong Chen, Pheng-Ann Heng, and Stan Z Li. A survey on generative diffusion models. IEEE Transactions on Knowledge and Data Engineering, 2024.
- [10] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-\sigma: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. arXiv preprint arXiv:2403.04692, 2024.
- [11] Ke Chen, Lei Xu, and Huisheng Chi. Improved learning algorithms for mixture of experts in multiclass classification. Neural networks, 12(9):1229–1252, 1999.
- [12] Kyunghyun Cho and Yoshua Bengio. Exponentially increasing the capacity-to-computation ratio for conditional computation in deep learning. arXiv preprint arXiv:1406.7362, 2014.

- [13] Florinel-Alin Croitoru, Vlad Hondru, Radu Tudor Ionescu, and Mubarak Shah. Diffusion models in vision: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023.
- [14] Damai Dai, Li Dong, Shuming Ma, Bo Zheng, Zhifang Sui, Baobao Chang, and Furu Wei. Stablemoe: Stable routing strategy for mixture of experts. arXiv preprint arXiv:2204.08396, 2022.
- [15] Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y Wu, et al. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models. arXiv preprint arXiv:2401.06066, 2024.
- [16] Andrew Davis and Itamar Arel. Low-rank approximations for conditional feedforward computation in deep neural networks. arXiv preprint arXiv:1312.4461, 2013.
- [17] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.
- [18] Ludovic Denoyer and Patrick Gallinari. Deep sequential neural network. arXiv preprint arXiv:1410.0510, 2014.
- [19] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.
- [20] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.
- [21] Nan Du, Yanping Huang, Andrew M Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, et al. Glam: Efficient scaling of language models with mixture-of-experts. In International Conference on Machine Learning, pages 5547–5569. PMLR, 2022.
- [22] David Eigen, Marc’Aurelio Ranzato, and Ilya Sutskever. Learning factored representations in a deep mixture of experts. arXiv preprint arXiv:1312.4314, 2013.
- [23] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021.
- [24] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.
- [25] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1–39, 2022.
- [26] Zheng-cong Fei. Fast image caption generation with position alignment. arXiv preprint arXiv:1912.06365, 2019.
- [27] Zhengcong Fei. Partially non-autoregressive image captioning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 1309–1316, 2021.
- [28] Zhengcong Fei, Mingyuan Fan, Li Zhu, and Junshi Huang. Progressive text-to-image generation. arXiv preprint arXiv:2210.02291, 2022.
- [29] Zhengcong Fei, Xu Yan, Shuhui Wang, and Qi Tian. Deecap: Dynamic early exiting for efficient image captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12216–12226, 2022.

- [30] Zhengcong Fei, Mingyuan Fan, Changqian Yu, and Junshi Huang. Scalable diffusion models with state space backbone. arXiv preprint arXiv:2402.05608, 2024.
- [31] Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, and Junshi Huang. Diffusion-rwkv: Scaling rwkv-like architectures for diffusion models. arXiv preprint arXiv:2404.04478, 2024.
- [32] Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, Youqiang Zhang, and Junshi Huang. Dimba: Transformer-mamba diffusion models. arXiv preprint arXiv:2406.01159, 2024.
- [33] Zhida Feng, Zhenyu Zhang, Xintong Yu, Yewei Fang, Lanxin Li, Xuyi Chen, Yuxiang Lu, Jiaxiang Liu, Weichong Yin, Shikun Feng, et al. Ernie-vilg 2.0: Improving text-to-image diffusion model with knowledge-enhanced mixture-of-denoising-experts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10135–10145, 2023.
- [34] Peng Gao, Le Zhuo, Ziyi Lin, Chris Liu, Junsong Chen, Ruoyi Du, Enze Xie, Xu Luo, Longtian Qiu, Yuhang Zhang, et al. Lumina-t2x: Transforming text into any modality, resolution, and duration via flow-based large diffusion transformers. arXiv preprint arXiv:2405.05945, 2024.
- [35] Dariu M Gavrila and Stefan Munder. Multi-cue pedestrian detection and tracking from a moving vehicle. International journal of computer vision, 73:41–59, 2007.
- [36] Hyojun Go, Yunsung Lee, Jin-Young Kim, Seunghyun Lee, Myeongho Jeong, Hyun Seung Lee, and Seungtaek Choi. Towards practical plug-and-play diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1962–1971, 2023.
- [37] Yizeng Han, Gao Huang, Shiji Song, Le Yang, Honghui Wang, and Yulin Wang. Dynamic neural networks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(11):7436–7456, 2021.
- [38] Jakob Vogdrup Hansen. Combining predictors: comparison of five meta machine learning methods. Information Sciences, 119(1-2):91–105, 1999.
- [39] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016.
- [40] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.
- [41] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.
- [42] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [43] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.
- [44] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. The Journal of Machine Learning Research, 23(1):2249–2281, 2022.
- [45] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022.
- [46] Vincent Tao Hu, Stefan Andreas Baumann, Ming Gui, Olga Grebenkova, Pingchuan Ma, Johannes S Fischer, and Björn Ommer. Zigma: A dit-style zigzag mamba diffusion model. arXiv preprint arXiv:2403.13802, 2024.

- [47] Yu Hen Hu, Surekha Palreddy, and Willis J Tompkins. A patient-adaptable ecg beat classifier using a mixture of experts approach. IEEE transactions on biomedical engineering, 44(9): 891–900, 1997.
- [48] Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. Adaptive mixtures of local experts. Neural computation, 3(1):79–87, 1991.
- [49] Michael I Jordan and Robert A Jacobs. Hierarchical mixtures of experts and the em algorithm. Neural computation, 6(2):181–214, 1994.
- [50] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
- [51] Diederik P Kingma and Ruiqi Gao. Understanding diffusion objectives as the elbo with simple data augmentation. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.
- [52] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.
- [53] Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. Advances in Neural Information Processing Systems, 32, 2019.
- [54] Yunsung Lee, JinYoung Kim, Hyojun Go, Myeongho Jeong, Shinhyeok Oh, and Seungtaek Choi. Multi-architecture multi-expert diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 13427–13436, 2024.
- [55] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. arXiv preprint arXiv:2006.16668, 2020.
- [56] Mike Lewis, Shruti Bhosale, Tim Dettmers, Naman Goyal, and Luke Zettlemoyer. Base layers: Simplifying training of large, sparse models. In International Conference on Machine Learning, pages 6265–6274. PMLR, 2021.
- [57] Wei-Yin Loh. Classification and regression trees. Wiley interdisciplinary reviews: data mining and knowledge discovery, 1(1):14–23, 2011.
- [58] Shitong Luo and Wei Hu. Diffusion probabilistic models for 3d point cloud generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2837–2845, 2021.
- [59] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. arXiv preprint arXiv:2401.08740, 2024.
- [60] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024.
- [61] Saeed Masoudnia and Reza Ebrahimpour. Mixture of experts: a literature survey. Artificial Intelligence Review, 42:275–293, 2014.
- [62] Kangfu Mei and Vishal Patel. Vidm: Video implicit diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 9117–9125, 2023.
- [63] Charlie Nash, Jacob Menick, Sander Dieleman, and Peter W Battaglia. Generating images with sparse representations. arXiv preprint arXiv:2103.03841, 2021.
- [64] Byeongjun Park, Sangmin Woo, Hyojun Go, Jin-Young Kim, and Changick Kim. Denoising task routing for diffusion models. arXiv preprint arXiv:2310.07138, 2023.

- [65] Byeongjun Park, Hyojun Go, Jin-Young Kim, Sangmin Woo, Seokil Ham, and Changick Kim. Switch diffusion transformer: Synergizing denoising tasks with sparse mixture-of-experts. arXiv preprint arXiv:2403.09176, 2024.
- [66] Gaurav Parmar, Richard Zhang, and Jun-Yan Zhu. On aliased resizing and surprising subtleties in gan evaluation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11410–11420, 2022.
- [67] David Patterson, Joseph Gonzalez, Quoc Le, Chen Liang, Lluis-Miquel Munguia, Daniel Rothchild, David So, Maud Texier, and Jeff Dean. Carbon emissions and large neural network training. arXiv preprint arXiv:2104.10350, 2021.
- [68] Svetlana Pavlitska, Christian Hubschneider, Lukas Struppek, and J Marius Zöllner. Sparselygated mixture-of-expert layers for cnn interpretability. In 2023 International Joint Conference on Neural Networks (IJCNN), pages 1–10. IEEE, 2023.
- [69] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.
- [70] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [71] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022.
- [72] Guocheng Qian, Jinjie Mai, Abdullah Hamdi, Jian Ren, Aliaksandr Siarohin, Bing Li, HsinYing Lee, Ivan Skorokhodov, Peter Wonka, Sergey Tulyakov, et al. Magic123: One image to high-quality 3d object generation using both 2d and 3d diffusion priors. arXiv preprint arXiv:2306.17843, 2023.
- [73] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
- [74] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.
- [75] Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, and Yuxiong He. Deepspeed-moe: Advancing mixture-ofexperts inference and training to power next-generation ai scale. In International conference on machine learning, pages 18332–18346. PMLR, 2022.
- [76] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.
- [77] Cedric Renggli, André Susano Pinto, Neil Houlsby, Basil Mustafa, Joan Puigcerver, and Carlos Riquelme. Learning to merge tokens in vision transformers. arXiv preprint arXiv:2202.12015, 2022.
- [78] Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, André Susano Pinto, Daniel Keysers, and Neil Houlsby. Scaling vision with sparse mixture of experts. Advances in Neural Information Processing Systems, 34:8583–8595, 2021.
- [79] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [80] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234–241. Springer, 2015.

- [81] Clemens Rosenbaum, Tim Klinger, and Matthew Riemer. Routing networks: Adaptive selection of non-linear functions for multi-task learning. arXiv preprint arXiv:1711.01239, 2017.
- [82] Carlos Riquelme Ruiz, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, André Susano Pinto, Daniel Keysers, and Neil Houlsby. Scaling vision with sparse mixture of experts. In Advances in Neural Information Processing Systems, 2021.
- [83] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016.
- [84] Axel Sauer, Katja Schwarz, and Andreas Geiger. Stylegan-xl: Scaling stylegan to large diverse datasets. In ACM SIGGRAPH 2022 conference proceedings, pages 1–10, 2022.
- [85] N Shazeer, A Mirhoseini, K Maziarz, A Davis, Q Le, G Hinton, and J Dean. The sparsely-gated mixture-of-experts layer. Outrageously large neural networks, 2017.
- [86] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.
- [87] Tianxiao Shen, Myle Ott, Michael Auli, and Marc’Aurelio Ranzato. Mixture models for diverse machine translation: Tricks of the trade. In International conference on machine learning, pages 5719–5728. PMLR, 2019.
- [88] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792, 2022.
- [89] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015.
- [90] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.
- [91] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.
- [92] Jun Tani and Stefano Nolfi. Learning to perceive the world as articulated: an approach for hierarchical learning in sensory-motor systems. Neural Networks, 12(7-8):1131–1141, 1999.
- [93] Ji Xin, Raphael Tang, Jaejun Lee, Yaoliang Yu, and Jimmy Lin. Deebert: Dynamic early exiting for accelerating bert inference. arXiv preprint arXiv:2004.12993, 2020.
- [94] Brandon Yang, Gabriel Bender, Quoc V Le, and Jiquan Ngiam. Condconv: Conditionally parameterized convolutions for efficient inference. Advances in neural information processing systems, 32, 2019.
- [95] Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, and Ming-Hsuan Yang. Diffusion models: A comprehensive survey of methods and applications. ACM Computing Surveys, 56(4):1–39, 2023.
- [96] Jiahui Yu, Linjie Yang, Ning Xu, Jianchao Yang, and Thomas Huang. Slimmable neural networks. arXiv preprint arXiv:1812.08928, 2018.
- [97] Seniha Esen Yuksel, Joseph N Wilson, and Paul D Gader. Twenty years of mixture of experts. IEEE transactions on neural networks and learning systems, 23(8):1177–1193, 2012.
- [98] Assaf Zeevi, Ron Meir, and Robert Adler. Time series prediction using mixtures of experts. Advances in neural information processing systems, 9, 1996.

- [99] Yihua Zhang, Ruisi Cai, Tianlong Chen, Guanhua Zhang, Huan Zhang, Pin-Yu Chen, Shiyu Chang, Zhangyang Wang, and Sijia Liu. Robust mixture-of-expert training for convolutional neural networks. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 90–101, 2023.
- [100] Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, and William Fedus. Designing effective sparse expert models. arXiv preprint arXiv:2202.08906, 2(3):17, 2022.

