## Guiding a Diffusion Transformer with the Internal Dynamics of Itself

# arXiv:2512.24176v2[cs.CV]31Mar2026

Xingyu Zhou1 Qifan Li1 Xiaobin Hu2 Hai Chen34 Shuhang Gu1*

1University of Electronic Science and Technology of China 2National University of Singapore 3Sun Yat-sen University 4North China Institute of Computer Systems Engineering

{xy.chous526, shuhanggu}@gmail.com

### Abstract

The diffusion model presents a powerful ability to capture the entire (conditional) data distribution. However, due to the lack of sufficient training and data to learn to cover lowprobability areas, the model will be penalized for failing to generate high-quality images corresponding to these areas. To achieve better generation quality, guidance strategies such as classifier free guidance (CFG) can guide the samples to the high-probability areas during the sampling stage. However, the standard CFG often leads to over-simplified or distorted samples. On the other hand, the alternative line of guiding diffusion model with its bad version is limited by carefully designed degradation strategies, extra training and additional sampling steps. In this paper, we proposed a simple yet effective strategy Internal Guidance (IG), which introduces an auxiliary supervision on the intermediate layer during training process and extrapolates the intermediate and deep layer’s outputs to obtain generative results during sampling process. This simple strategy yields significant improvements in both training efficiency and generation quality on various baselines. On ImageNet 256×256, SiTXL/2+IG achieves FID=5.31 and FID=1.75 at 80 and 800 epochs. More impressively, LightningDiT-XL/1+IG achieves FID=1.34 which achieves a large margin between all of these methods. Combined with CFG, LightningDiT-XL/1+IG achieves the current state-of-the-art FID of 1.19.

### 1. Introduction

Denoising Diffusion models have achieved notable success in high-dimensional data generation and rapidly applied to fields such as images, videos and 3D. Recently, the introduction of the Transformer architecture has further enhanced the scalability of the diffusion model in these fields [11, 52, 55], demonstrating its outstanding performance. As the scale of the diffusion model continues to increase, how to efficiently achieve excellent generation results has become a highly

*Corresponding author. Project page: Internal Guidance

focused research topic.

The training objective of the diffusion model is to cover the entire (conditional) data distribution. The model is severely penalized for failing to cover low-probability areas, but it lacks sufficient training and data to learn to generate high-quality images corresponding to these areas. Several studies introduce additional guidance strategies during the sampling stage to encourage the generative process to focus on high-probability areas of the data distribution. Among them, classifier-free guidance (CFG) [24] leads to a better alignment between the prompt and the result by avoiding the unconditional results. Nevertheless, an excessively high CFG coefficient may push the sampling trajectory beyond the expected range of the conditional distribution, resulting in over-simplified or distorted samples [44]. An alternative line of research attempts to guide a well-trained model using a bad version of itself [38] instead of relying on CFG. By avoiding the results of the corresponding degraded version of the generative model during the sampling stage, the generation quality is improved while avoiding overly simplified results. Although promising, these approaches still require carefully designed degradation strategies [26, 27], extra training [38] or additional sampling steps [5, 29], thereby limiting their potential for the large-scale application.

To address this, we propose a simple yet effective sampling guidance strategy, named Internal Guidance (IG), which utilizes the intermediate-layer outputs within the Diffusion Transformer to enhance generation quality. Specifically, we introduce an auxiliary supervisory signal at an intermediate layer during training process, enabling the model to produce a weaker generative output from mid-level representations. Building upon this, during the sampling stage, we leverages the relationship between intermediate and deep outputs during sampling to extrapolate. This internal guidance achieves an Autoguidance-like [38] effect that maintains diversity and improves the generation quality but without any additional sampling steps. Moreover, we show that IG can also combine with the guidance interval [44] or CFG to further enhance generative performance. Besides providing intermediate results for guidance, we observe that

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Figure 1. Visualization Results. We visualize our latent diffusion system with proposed IG together with LightningDiT-XL trained on ImageNet 256 × 256 resolution. With an IG scale of 1.4 and a CFG scale of 1.45, and further combining the guidance interval, we ultimately achieved the state-of-the-art FID = 1.19. More uncurated samples are provided in Supplementary Material.

the introduction of intermediate supervision alone can alleviate partial vanishing gradients. It’s convergence performance comparable to, or even better than, those achieved with self-supervised regularization applied to intermediate layers. Further leveraging the insight in IG, we discuss how modeling the discrepancy between intermediate and deep outputs within the loss function can inspire the design of new training acceleration strategies.

We conduct comprehensive experiments to evaluate the effect of our proposed internal guidance. Our IG brings significant performance improvements to both DiTs, SiTs and LightningDiT. Ultimately, we compared our generative performance with the current state-of-the-art models on SiT-XL/2 and LightningDiT-XL/1 with IG. For SiT-XL/2 training, we show the model achieves FID = 5.31 and FID

- = 1.75 on class-conditional ImageNet generation only using 80 and 800 training epochs (without CFG) which already exceeds the FID of the vanilla SiT-XL at 1400 epoch and REPA at 800 epochs that uses pre-trained representation model to align. Moreover, with classifier-free guidance and guidance interval, our scheme shows an improved FID at 1.46 at 800 epochs. And for LightningDiT-XL/1 model which uses selfsupervised representation-based tokenizer, it achieves FID
- = 2.42 using 60 epochs and further achieves FID = 1.34 at 680 epochs, which achieves a large margin between all of these methods. Combined with classifier-free guidance and guidance interval, our scheme achieves state-of-the-art results of FID = 1.19 at 680 epochs.

### 2. Related Work

We discuss with the most relevant literature here and provide a more discussion in Supplementary Material.

Sampling Guidance in Diffusion Models. The sampling guidance after the training has become the key to improving the quality of the visual generation. Classifier-free guidance (CFG) [24] has become the standard method which forms a guided score by extrapolating from an unconditional one. While highly effective, CFG will cause the problem of reduced generation diversity when its guiding coefficient becomes larger. Several studies [4, 14, 44, 60, 75] have reduced the downsides of CFG by making the guidance weight noise level-dependent. Another research direction separates the class guidance part from the CFG and only uses a weaker version of the current model for guidance. This approach improves image quality while maintaining diversity. While promising, these methods are limited by the complex design of degradation strategies [26, 27], extra training [38] or additional sampling steps [5, 29], which are not available for current large-scale image generators in practice. In this paper, our proposed internal guidance can be seamlessly integrated into the current advanced diffusion Transformer with almost no additional cost, demonstrating its potential for large-scale application.

Intermediate Representation Regularization for Diffusion Transformers. Many recent works have attempted to introduce intermediate representation regularization in diffu-

sion transformer training. MaskDiT [82] and SD-DiT [84] combine MAE’s [21] and IBOT’s [83] training strategy into the DiT’s training process. TREAD [41] designs a token routing strategy to speed up Diffusion Transformer’s training. More recently, self-supervised learning has been introduced into diffusion training to further enhance training efficiency. These methods can be broadly categorized into: (1) aligning intermediate network representations with pre-trained self-supervised models [46, 76, 79], and (2) incorporating self-supervised representation learning as a regularization mechanism during training [33, 72]. In this paper, we employ the simplest approach of introducing an auxiliary supervision loss to alleviate the gradient vanishing problem in deep networks. This design bears conceptual similarity to DeepFlow [64], but it mainly focuses on introducing second-order ODEs in the training of the diffusion Transformer. Interestingly, we observed that the convergence performance of this simple method is comparable to that achieved by incorporating self-supervised learning as a form of regularization.

### 3. Methodology

#### 3.1. Preliminaries

Our work is based on Flow-matching [51] and Diffusion [25], [52] and [1] have proposed a unified perspective to understand these two types of generative modeling methods. We first introduce the relevant preliminaries. The standard Flow-matching and Diffusion-based models both start from Gaussian noise ϵ ∼ N(0,I) and gradually transform it into data samples x∗ ∼ p(x) through stochastic processes. This process can be expressed as

xt = αtx∗ + σtϵ, (1)

where αt and σt are decreasing and increasing functions of time t ∈ [0,T]. The difference between the generative models based on Flow-matching and Diffusion lies in that the former typically interpolates noise and data within a finite time interval t ∈ [0,1], while the latter defines a forward stochastic differential equation (SDE), which converges to a Gaussian distribution N(0,I) as t → ∞. The denoising network Dθ estimates the original data by minimizing the denoising diffusion objective:

∗∼pxEϵ∼N(0,I)∥Dθ(xt,t) − x0∥2. (2)

Ex

As the network size of generative models continues to increase, how to efficiently obtain high-quality generation results has become the focus of attention. Next, we will introduce our framework from two aspects: the loss function and the corresponding sampling guidance after training completion. Furthermore, we will discuss some related properties of the proposed sampling guidance, as well as the design of new training acceleration methods based on this.

Training:

Output Layer Denoising Objective

Di

DenoisingObjective

[Figure 19]

[Figure 20]

DiT/SiTBlock

DiT/SiTBlock

DiT/SiTBlock

DiT/SiTBlock

DiT/SiTBlock

DiT/SiTBlock

OutputLayer

Df

Sampling:

Dw(x;c) = Di(x;c)+ w(Df (x;c) − Di(x;c))

Figure 2. The overall framework of our proposed Internal Guidance. We introduce an additional auxiliary supervision loss during training, and utilize the intermediate layer outputs during sampling process to guide the final outputs.

#### 3.2. Internal Guidance: Guiding a Diffusion Model with Its Own Internal Dynamics

Training a Diffusion Model with Intermediate Supervision. The difficulty of training deep networks has been explored in many aspects in various fields [19, 20, 31, 71]. Recently, in the field of generation, people have introduced self-supervised representation learning as a regularization method for training diffusion models [33, 72]. In this paper, we attempt the simplest solution, which is to apply an auxiliary supervision in the intermediate layers of the network. Specifically, we additionally define an output layer after the intermediate part of the network. Accordingly, the intermediate and final losses Linter and Lfinal are defined as

Linter = ∥Di(xt,t) − x0∥2, Lfinal = ∥Df(xt,t) − x0∥2,

(3)

where Di(·) and Df(·) denote the outputs of the intermediate and final layers, respectively. The overall framework is illustrated in Figure 2. For instance, for the training of a deep Diffusion Transformer, its final training loss becomes:

L = Lfinal + λLinter, (4)

where λ > 0 is a hyperparameter that controls the trade-off between the loss of the intermediate output and that of the final output. As shown in Table 2, We verified the effect of this loss on the SiT-B/2 model [52]. Although this method is very simple, it can surprisingly achieve results that are similar to or even better than those obtained by using the regularization method based on representation learning for complex designs [33, 72].

Sampling a Diffusion Model with Its Internal Outputs. The sampling stage is crucial for obtaining high-quality generated results. The standard classifier-free guidance (CFG) [38] sampling strategy can effectively improve the generation quality, but will cause an undesirable reduction in diversity. An alternative line of research attempts to maintain

[Figure 21]

[Figure 22]

posed IG is a plug and play approach that no longer requires specific training on particular bad versions for specific experimental setups or additional sampling consumption. And this sampling method can also enhance the quality of image generation while maintaining diversity. Building upon this, we conduct an extensive discussion on the properties of IG.

### 4. Discussion

In this section, we conduct a detailed analysis of the specific impacts of IG from two perspectives: (1) the compatibility between IG and CFG and (2) Guidance Interval in IG.

(a) No guidance

(b) Classifier-free guidance (CFG)

#### 4.1. The compatibility between IG and CFG

[Figure 23]

[Figure 24]

Although Internal Guidance (IG) can significantly improve the generation results without additional consumption during the sampling stage, its combination with Classifier-Free Guidance (CFG) can further improve the generation quality. To further elucidate the compatibility between our proposed IG and CFG, we conducted a series of visualization experiments to analyze at which levels these guidance strategies improve generative quality. We study a 2D toy example like [38] where a small-scale denoising network is trained to perform conditional diffusion in a synthetic dataset (Figure 3). The dataset is designed to exhibit low local dimensionality (i.e., highly anisotropic and narrow support) and a hierarchical emergence of local details upon noise removal. Both properties that can be expected from the actual manifold of realistic images [3, 57]. Details of the experimental setup are provided in the Supplementary Material.

(c) Internal guidance (IG) (d) IG+CFG

- Figure 3. A fractal-like 2D distribution with two classes indicated with gray and orange regions. Approximately 99% of the probability mass is inside the shown contours. (a) Conditional sampling using a small denoising diffusion model generates outliers due to its limited fitting capability. (b) Classifier-free guidance (w

= 2.5) eliminates the vast majority of outliers but reduces diversity by over-emphasizing the class. (c) Internal guidance (w = 2) can maintain diversity as Autoguidance [38] while allowing some outliers at the ends of the branches. (d) The combination of IG and CFG can significantly reduce outliers without reducing diversity.

As shown in Figure 3(a), sampling from a diffusion model without guidance produces many outliers outside the data distribution. When using CFG, this guidance pulls samples toward higher values of log [pcon(x | c)/puncon(x | c)] [38], where pcon and puncon are the conditional and unconditional distributions learned by the denoising diffusion model. As shown in Figure 3(b), increasing the CFG coefficient shifts samples away from the opposite class and reduces outliers, but also suppresses branches near other classes, which enhances inter-class separability while reducing diversity. In contrast, Autoguidance [38] provides class-independent gradients that point inward towards the data manifold.The trajectories of these samples generated by this well trained model generally follow the local orientation and branching of the data manifold, pushing the samples deeper into the “good side” concentrates them at the manifold.

diversity by guiding a well-trained model using a bad version of itself [38]. While promising, these approaches are still limited by carefully designed degradation strategies [26, 27], extra training [38] or additional sampling steps [5, 29]. By applying an auxiliary supervision to the intermediate layer of the denoising model in the training stage, we can obtain the output of the final layer and that of its intermediate layer during the sampling stage. This naturally amounts to creating a bad version corresponding to the current final layer’s output. In this way, we propose Internal Guidance (IG), the intermediate output Di can be used to guide the final output Df to shift away from its own poor distribution, thereby achieving better generation results. Similar to classifier free guidance (CFG), its guiding effect is achieved by extrapolating the two denoised results using a factor w

Furthermore, our proposed IG exhibits a performance similar to that of Autoguidance (Figure 3(c)). When the IG coefficient increases, these samples produce a narrower distribution than the ground truth, but in practice this does not appear to have an adverse effect on the images. However, this reduced intra-class sample distance may introduce new outliers when a large Autoguidance or IG coefficient is applied before the model is fully trained. In this case,

Dw(x;c) = Di(x;c) + w(Df(x;c) − Di(x;c)). (5) Unlike the original Autoguidance [38] strategy, our pro-

the stronger output fails to accurately point toward highprobability regions, rendering the guidance less effective. Because CFG provides class-conditional directionality, it complements IG by further suppressing such outliers. Combining the two achieves superior generative quality across training stages (Figure 3(d)). In Section 5.2, we present a detailed quantitative validation to demonstrate this effect.

#### 4.2. Guidance Interval in Internal Guidance

Meanwhile, the guidance interval [44] can also affect the performance of internal guidance. The original guidance interval encourages using large CFG coefficient for sampling within a continuous interval of noise levels, which is helpful for achieving better generation performance. For our proposed internal guidance, we also redefined the guidance function of Equation 5 by replacing ω with a piecewise constant function:

Dw(x;c) = Di(x;c) + w(σ)(Df(x;c) − Di(x;c)), where w(σ) =

w if σ ∈ (σlow,σhigh], 1 otherwise,

(6)

where, σ is the coefficient that controls the intensity of the noise. In the original guidance interval, applying CFG during periods of high noise range drastically reduces the variation in the results, basically leading them towards a handful of “template images” per class. Therefore, the guidance interval used in the CFG indicates that this guidance should only be used in the middle and low-noise ranges. For our proposed IG with the guidance interval, we observed the opposite phenomenon: the internal guidance does not need to be applied in the low-noise range, but applied in the high-noise and middle-noise ranges. In Section 5.2, we have provided a detailed presentation of the generated results after applying the guidance interval to the IG sampling.

### 5. Experiments 5.1. Setup

Implementation details. For experiments on DiT [55] and SiT [52], we strictly follow their original setup. For experiments on LightningDiT [78], we find using the recipe in LightningDiT leads to instability or abnormal training dynamics at early epochs and slow EMA model convergence at early epochs. To solve these issues, we instead use the Muon optimizer [34] instead of AdamW and change the EMA weight from 0.9999 to 0.9995. Other optimization hyperparameters are the same as vanilla LightningDiT. We use ImageNet-1K [9] for diffusion training, where each image is preprocessed to the resolution of 256×256 and follow ADM for other data preprocessing protocols. Each image is then encoded into a compressed vector z ∈ R32×32×4 using the Stable Diffusion VAE or z ∈ R16×16×32 using the VA-VAE.

- Table 1. Auxiliary supervision loss on different intermediate layers on ImageNet 256×256 with internal guidance (w = 1.5). ↓ and ↑ indicate whether lower or higher values are better.

Block Layers

Baseline +IG

FID↓ IS↑ FID↓ IS↑ SiT-B/2 [52] 33.02 43.71 – –

2 30.45 47.97 20.39 61.38 4 30.60 47.70 19.02 65.06 6 34.15 42.38 25.04 52.39 8 38.05 37.97 34.05 40.35

10 36.37 40.91 35.58 40.86 2 and 6 33.34 43.08 31.55 46.66

- Table 2. Convergence analysis of the SiT-B/2 model on ImageNet 256×256 without internal guidance and classifier-free guidance (CFG). ↓ and ↑ indicate whether lower or higher values are better, respectively. All models were trained for 80 epochs.

Methods Learning Paradigms FID↓ IS↑ SiT-B/2 Baseline [52] 33.02 43.71

SRA [33] Self-Supervised 29.10 50.20 Disperse Loss [72] Self-Supervised 31.45 47.05

Auxiliary Supervision Supervised 30.45 47.97

Detailed experimental details and hyperparameter settings, are provided in the Supplementary Material.

Evaluation protocol. We report Fr´echet inception distance (FID) [23], sFID [53], inception score [61], precision(Pre.) and recall(Rec.) [43] using 50000 samples. Specifically, we uniformly randomly sample the 1,000 class labels 50,000 times and generate images accordingly SD, SiT, REPA for a fair comparison. Sampling follows original DiT, SiT and LightningDiT that we use the SDE Euler-Maruyama sampler with 250 steps for the experiments on DiT and SiT, and use the ODE Heun sampler with 125 steps for the experiments on LightningDiT. Full evaluation protocol details are also provided in the Supplementary Material.

#### 5.2. Ablation Studies

The position of intermediate supervision. We begin by examining the effect of attaching the auxiliary supervision loss to different layers, which are shown in Table 1. We find that regularizing only the first few layers (e.g., 4 in SiT-B/2 [52] at 80 epochs) in training is effective. However, placing the auxiliary supervision loss in the latter half of the network or adding multiple auxiliary supervision losses in multiple intermediate layers cannot improve the convergence speed. We hypothesize that it interferes with the training of the deep layer’s output. Furthermore, we are surprised to find that the convergence effect with the auxiliary supervision loss is comparable to that of methods incorporate self-supervised

80

SiT-B/2+CFG

24

22

70

SiT-B/2+IG(1.5)+CFG SiT-B/2+IG(1.3)+CFG

20

18

60

16

FID-50K

14

50

12

DiT-B/2

FID-50K

10

40

6.88 7.30

8

SiT-B/2

6

6.50

30

|DiT-B/2+IG DiT-L/2+IG SiT-B/2+IG SiT-L/2+IG SiT-XL/2+IG<br><br>|
|---|

4

DiT-L/2 SiT-L/2

1 1.5 2 2.5

20

CFG Coefficient

SiT-XL/2

10

- Figure 4. The combination of IG and CFG can yield a higher FID value, which is superior to simply apply CFG. In particular, the FID value generated by the combination of a lower IG coefficient and CFG is higher than that of a higher coefficient.

0

20 70 120 170 220 270 320 370

Training Iteration (K)

Figure 5. Scalability of IG. The relative improvement of IG over the vanilla model (DiTs and SiTs) becomes increasingly significant as the model size grows.

Table 3. The IG coefficient and the corresponding guidance interval analysis on ImageNet 256×256. ↓ and ↑ indicate whether lower or higher values are better, respectively.

Table 4. Ablation study for λ. λ 0.25 0.5 0.75 1 FID↓ 30.38 30.60 31.58 31.24

Sampling ω Guidance Interval FID↓ IS↑ w/o IG 1.0 [0, 1) 30.60 47.70

IS↑ 47.86 47.70 46.23 46.24

- 1.3 [0, 1) 21.88 59.55

- 1.5 [0, 1) 19.02 65.06

- 1.7 [0, 1) 17.61 68.16

- 1.9 [0, 1) 17.38 69.12

- 2.1 [0, 1) 17.75 68.67

- 2.3 [0, 1) 18.53 66.77

- 2.3 [0.3, 1) 16.19 72.95

- 2.3 [0.3, 0.7) 18.23 66.21 2.3 [0, 0.7) 20.69 60.14

periments. In the Table 3, we can see that when no guidance interval is used, the optimal FID is approximately around the IG coefficient ω of 1.9. After increasing the IG coefficient, we find it actually has a positive effect on the generation quality when the low-noise range does not apply IG. This is somewhat contrary to the conclusion of guidance interval used in CFG [44]. In our rough search, the best generative results are achieved when the IG coefficient is 2.3 and the guidance interval range is [0.3,1).

w/ IG

representation learning regularization (Table 2). In largescale experiments, we apply the auxiliary supervision loss and internal guidance to the 8-th layer of the large-scale diffusion transformer model.

Scalability. We investigate the scalability of the auxiliary supervision loss with internal guidance by varying the model sizes of the diffusion transformers. In general, the effect of achieving efficient generation results of IG becomes more significant as the diffusion transformer model increases in size. We demonstrate this by plotting FID-50K of different SiT and DiT models with IG and without CFG in Figure: IG achieves the same FID level more quickly with larger models. To facilitate comparison, we set the IG coefficient to 2.3 and the guidance interval range is [0.3,1) in all the DiTs and SiTs models. And the IG is applied in the fourth layer of the B/2 scale model and in the eighth layer of other scales for the experiments.

Compatibility between IG and CFG. We also demonstrate that the proposed IG combined with CFG can further enhance the generation quality in the quantitative experiment. We apply IG on the fourth layer of SiT-B/2 to conduct the subsequent experiments. We conducted a rough search for the optimal combination rules of IG and CFG. As shown in Figure 4, after SiT-B/2 combines with our IG and then applies CFG, the best generated metric FID can be reduced from the original optimal 7.30 with only CFG to 6.50. Meanwhile, we find that when combining CFG, using a lower IG coefficient can yield better generation results, which is different from the optimal coefficient when only adding IG alone (Table 3 and Figure 7). And this process basically does not involve significant computational costs, which indicates that the IG can serve as an easily integrable component for modern generative models to enhance the generation quality.

Effect of λ. Finally, we examine the effect of the regularization coefficient λ by training SiT-B/2 models for 400K with different coefficients 0.25 to 1.0 and comparing the performance. As shown in Table 4, the performance becomes stable after λ <= 0.5.

#### 5.3. Comparsion with State-of-the-art Methods

Guidance interval in internal guidance. Next, we examine the role of guidance interval in internal guidance. We also apply IG on the fourth layer of SiT-B/2 to conduct ex-

We use several recent diffusion-based generation methods as baselines, each employing different inputs and network archi-

Table 5. Class-conditional performance on ImageNet 256×256. SiT + IG reaches an FID of 1.75 in and further achieves an FID of 1.46 with CFG. LightningDiT + IG reaches an FID of 1.34 without CFG, outperforming all prior methods by a large margin, and achieves an FID of 1.19 with CFG. For a fair comparison, all prior methods employ random sampling 50K samples across 1,000 class labels.

Generation@256 w/o CFG Generation@256 w/ CFG or Autoguidance

Method Epochs #Params

FID↓ sFID↓ IS↑ Prec.↑ Rec.↑ FID↓ sFID↓ IS↑ Prec.↑ Rec.↑ Pixel Diffusion

ADM [10] 400 554M 10.94 6.02 101.0 0.69 0.63 3.94 6.14 215.8 0.83 0.53 RIN [32] 480 410M 3.42 - 182.0 - - - - - - PixelFlow [6] 320 677M - - - - - 1.98 5.83 282.1 0.81 0.60 PixNerd [73] 160 700M - - - - - 2.15 4.55 297.0 0.79 0.59 SiD2 [28] 1280 - - - - - - 1.38 - - - -

###### Vanilla Latent Diffusion

DiT [55] 1400 675M 9.62 6.85 121.5 0.67 0.67 2.27 4.60 278.2 0.83 0.57 MaskDiT [82] 1600 675M 5.69 10.34 177.9 0.74 0.60 2.28 5.67 276.6 0.80 0.61 SiT [52] 1400 675M 8.61 6.32 131.7 0.68 0.67 2.06 4.50 270.3 0.82 0.59 TREAD [41] 740 675M - - - - - 1.69 4.73 292.7 0.81 0.63 MDTv2 [15] 1080 675M - - - - - 1.58 4.52 314.7 0.79 0.65

5.31 5.68 147.7 0.80 0.56 – – – – –

80

SiT+IG (ours)

678M

800 1.75 4.98 228.6 0.80 0.62 1.46 4.79 265.7 0.80 0.64 Latent Diffusion with Self-supervised Representation Model REPA [79]

80

7.90 5.06 122.6 0.70 0.65 - - - - -

675M

800 5.90 5.69 157.8 0.70 0.69 1.42 4.70 305.7 0.80 0.65 REPA-E [46]

80

3.46 4.17 159.8 0.77 0.63 1.67 4.12 266.3 0.80 0.63

675M

800 1.83 - 217.3 - - 1.26 4.11 314.9 0.79 0.66 REG [76]

3.40 - 184.1 - - 1.86 4.49 321.4 0.76 0.63

80

677M

480 2.20 4.67 219.1 0.77 0.66 1.40 4.24 296.9 0.77 0.66 LightningDiT [78]

64

5.14 4.22 130.2 0.76 0.62 2.11 4.16 252.3 0.81 0.58

675M

800 2.17 4.36 205.6 0.77 0.65 1.35 4.15 295.3 0.79 0.65 RAE (DiTDH) [81] 800 839M 1.60 5.40 242.7 0.79 0.65 1.28 4.72 262.9 0.78 0.67 LightningDiT+IG (ours)

2.42 3.81 173.7 0.79 0.62 – – – – – 680 1.34 3.94 229.3 0.78 0.65 1.19 4.11 269.0 0.79 0.66

60

678M

tectures. Specifically, we consider the following three types of approaches: (a) Pixel Diffusion: ADM [10], RIN [32], PixelFlow [6], PixNerd [73] and SiD2 [28], (2) Vanilla Latent Diffusion: DiT [55], MaskDiT [82], SiT [52], TREAD [41], MDTv2 [15], (c) Latent Diffusion with Self-supervised Representation Model: REPA [79], REPA-E [46] and REG [76], LightningDiT [78] and DiTDH [81]. We initialized our models on SiT and LightningDiT respectively to separately verify the effectiveness of the IG in the latent diffusion model with original VAE and aligned VAE that represents the latent space using the pretrained self-supervised representation model (i.e., DINOv2-B [54]).

We compare the FID values between SiT-XL/2 and LightningDiT-XL/1 with IG with other state-of-the-art methods without CFG. As shown in Table 5, IG shows consistent and significant improvement across all model variants under random sampling. We provide balanced sampling results in the Supplementary Material. In particular, on SiT-XL/2, using IG (1.8) leads to FID=5.31 at 80 epochs, which already exceeds the FID of the vanilla SiT-XL at 1400 epoch and REPA at 800 epochs which uses pre-trained representation model to align. Note that the performance continues to improve with longer training; for instance, with SiT-XL/2, FID becomes 1.75 at 800 epochs. On LightningDiT-XL/1,

using IG (1.7) leads to FID=2.42 at 60 epochs, which also already exceeds the FID of the vanilla LightningDiT-XL/1 at 64 epochs. For longer training, the FID becomes 1.34 at 680 epochs with LightningDiT-XL/1, which achieve a large margin between all of these methods.

Finally, we provide a quantitative comparison between SiT-XL/2 or LightningDiT-XL/1 with IG and other recent diffusion model methods using CFG. At 800 epochs, SiTXL/2 + IG (1.8) achieves FID of 1.46 with a CFG scale of w = 1.35 and an extra guidance interval. Meanwhile, LightningDiT/1 + IG (1.4) achieves state-of-the-art FID of 1.19 with a CFG scale of w = 1.45 and an extra guidance interval at 680 epochs. We provide visual results of LightningDiTXL/2 + IG in Figure 8. Moreover, we also provide more uncurated examples and experimental results on ImageNet 512×512 in the Supplementary Material, we show that IG also provides significant improvements in such setup.

### 6. Further Discussion on Training Acceleration

After conducting the above analysis, we naturally came up with a new question: Can the effect of this IG be replicated during the training process, so as to achieve the goal of training acceleration? We also demonstrated the denoising

###### 512 iterations 4096 iterations

###### 512 iterations

###### 512 iterations

SiT-B/2 SiT-B/2+REPA

90

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

80

SiT-B/2+Training Loss (ours)

SiT-B/2+IG (ours)

70

60

FID-50K

50

| |
|---|

| |
|---|

| |
|---|

40

30

20

10

0 100 200 300 400 Training Iteration (K)

(b) No guidance (long training)

(c) Internal guidance (ours)

(d) Loss of accelerated training (ours)

(a) No guidance (short training)

Figure 6. Internal guidance inspires new training acceleration methods. (a) Conditional sampling using a not well-trained denoising diffusion model generates a large number of outliers. (b) Outliers can gradually be eliminated with sufficient training. (c) Internal guidance can also eliminate outliers with not well-trained denoising diffusion model. (d) The loss function we proposed can accelerate convergence. In SiT-B/2 experiments, our proposed loss function demonstrates superior accelerated convergence performance compared to REPA [79].

###### SiT-XL/2 + IG + CFG

diffusion model training process through a 2D toy example. As shown in Figure 6(a) and (b), comprehensive training can significantly reduce many outliers near the branches compared to incomplete training. With our proposed IG, since its guiding direction points towards the overall direction of the stronger model, outliers near the branches can also be effectively reduced (Figure 6c). Based on the above observations, we incorporate the guiding effect of IG into the training process of the denoising diffusion model. Specifically, we integrate the gradient vectors obtained by calculating the differences between the final and intermediate outputs during the sampling process into the loss function, enabling the model to learn the guidance signal during training. Similar to [66], we rewrite the training objectives for the intermediate and final layers during the training process

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Intermediate objective: x0, Final objective: x0 + ω · sg(Df(xt) − Di(xt)).

(7)

Where ω > 0 is a hyperparameter that controls the strength of the guiding signal for the final layer’s training objective. In practice, we utilize the EMA model that updates along with the model to obtain the outputs at both the intermediate and final layers during the training process and set ω = 0.5 following [66]. As shown in Figure 6(d), the outliers of all branches have been significantly reduced after modifying the training objective. In the subsequent SiT-B/2 experiment, our designed training loss significantly outperformed the REPA [79] using pre-trained self-supervised representation models after 400K iterations, demonstrating performance comparable to that of IG. Due to the flexibility and tunability of IG in generating images during the reasoning process, we suggest continuing to use IG in actual applications.

Figure 7. The combination of IG and CFG can produce better visual quality. When only adding IG, better generation results can be obtained. Incorporating CFG can further reduce the inaccurate content in the generated results.

an auxiliary supervision loss in the deep Diffusion Transformer and utilize the output of its intermediate layers to guide the output results at the deep level during the sampling stage. This plug-and-play method has significantly improved the quality of image generation. We further analyze the properties of each part of our proposed IG, including its compatibility with CFG, its own guidance interval and discuss the design of new training acceleration methods based on its insight. Our IG can significantly improve the generation performance of various scale diffusion transformers with less computational resources. Ultimately, we get the state-of-the-art FID = 1.19 result for image generation when using LightningDiT-XL/1+IG with additional CFG. We hope our work would facilitate the research on more applicable sampling methods for visual generation.

### 7. Conclusion

In this paper, we have presented Internal Guidance (IG), a simple but effective guidance strategy for enhancing both generation quality and training efficiency. We introduce

Acknowledgement. This work was supported by National Natural Science Foundation of China (No. 62476051).

### References

- [1] Michael S Albergo, Nicholas M Boffi, and Eric VandenEijnden. Stochastic interpolants: A unifying framework for flows and diffusions. arXiv preprint arXiv:2303.08797, 2023. 3
- [2] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22669–22679, 2023. 6
- [3] Bradley CA Brown, Anthony L Caterini, Brendan Leigh Ross, Jesse C Cresswell, and Gabriel Loaiza-Ganem. Verifying the union of manifolds hypothesis for image data. arXiv preprint arXiv:2207.02862, 2022. 4
- [4] Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-toimage generation via masked generative transformers. arXiv preprint arXiv:2301.00704, 2023. 2, 6
- [5] Chubin Chen, Jiashu Zhu, Xiaokun Feng, Nisha Huang, Meiqi Wu, Fangyuan Mao, Jiahong Wu, Xiangxiang Chu, and Xiu Li. s2-guidance: Stochastic self guidance for training-free enhancement of diffusion models. arXiv preprint arXiv:2508.12880, 2025. 1, 2, 4, 6
- [6] Shoufa Chen, Chongjian Ge, Shilong Zhang, Peize Sun, and Ping Luo. Pixelflow: Pixel-space generative models with flow. arXiv preprint arXiv:2504.07963, 2025. 7
- [7] Hyungjin Chung, Jeongsol Kim, Geon Yeong Park, Hyelin Nam, and Jong Chul Ye. Cfg++: Manifold-constrained classifier free guidance for diffusion models. arXiv preprint arXiv:2406.08070, 2024. 6
- [8] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691, 2023. 2
- [9] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 5
- [10] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 7, 1, 6
- [11] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024. 1, 6
- [12] Weichen Fan, Amber Yijia Zheng, Raymond A Yeh, and Ziwei Liu. Cfg-zero*: Improved classifier-free guidance for flow matching models. arXiv preprint arXiv:2503.18886,

2025. 6

- [13] Johannes S Fischer, Ming Gui, Pingchuan Ma, Nick Stracke, Stefan A Baumann, and Bj¨orn Ommer. Boosting latent diffusion with flow matching. arXiv preprint arXiv:2312.07360,

2023. 6

- [14] Shanghua Gao, Pan Zhou, Ming-Ming Cheng, and Shuicheng Yan. Masked diffusion transformer is a strong image synthesizer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 23164–23173, 2023. 2, 6
- [15] Shanghua Gao, Pan Zhou, Ming-Ming Cheng, and Shuicheng Yan. Mdtv2: Masked diffusion transformer is a strong image synthesizer. arXiv preprint arXiv:2303.14389, 2023. 7, 6
- [16] Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. 6
- [17] Google. Nano banana:, 2025. 6
- [18] Ali Hatamizadeh, Jiaming Song, Guilin Liu, Jan Kautz, and Arash Vahdat. Diffit: Diffusion vision transformers for image generation. In European Conference on Computer Vision, pages 37–55. Springer, 2024. 6
- [19] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision, pages 1026–1034, 2015. 3
- [20] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 3
- [21] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000– 16009, 2022. 3
- [22] D Hendrycks. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016. 5
- [23] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 5, 1
- [24] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 1, 2, 6
- [25] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 3
- [26] Susung Hong. Smoothed energy guidance: Guiding diffusion models with reduced energy curvature of attention. Advances in Neural Information Processing Systems, 37:66743–66772,

2024. 1, 2, 4, 6

- [27] Susung Hong, Gyuseong Lee, Wooseok Jang, and Seungryong Kim. Improving sample quality of diffusion models using self-attention guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7462– 7471, 2023. 1, 2, 4, 6
- [28] Emiel Hoogeboom, Thomas Mensink, Jonathan Heek, Kay Lamerigts, Ruiqi Gao, and Tim Salimans. Simpler diffusion (sid2): 1.5 fid on imagenet512 with pixel-space diffusion. arXiv preprint arXiv:2410.19324, 2024. 7

- [29] Junha Hyung, Kinam Kim, Susung Hong, Min-Jung Kim, and Jaegul Choo. Spatiotemporal skip guidance for enhanced video diffusion sampling. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 11006– 11015, 2025. 1, 2, 4, 6
- [30] Aapo Hyv¨arinen and Peter Dayan. Estimation of nonnormalized statistical models by score matching. Journal of Machine Learning Research, 6(4), 2005. 5
- [31] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning, pages 448–456. pmlr, 2015. 3
- [32] Allan Jabri, David Fleet, and Ting Chen. Scalable adaptive computation for iterative generation. arXiv preprint arXiv:2212.11972, 2022. 7
- [33] Dengyang Jiang, Mengmeng Wang, Liuzhuozheng Li, Lei Zhang, Haoyu Wang, Wei Wei, Guang Dai, Yanning Zhang, and Jingdong Wang. No other representation component is needed: Diffusion transformers can provide representation guidance by themselves. arXiv preprint arXiv:2505.02831,

2025. 3, 5

- [34] Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024. 5, 2, 3
- [35] Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up gans for text-to-image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10124–10134, 2023. 6
- [36] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In Proc. NeurIPS, 2022. 5
- [37] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35:26565–26577, 2022. 5, 6
- [38] Tero Karras, Miika Aittala, Tuomas Kynk¨a¨anniemi, Jaakko Lehtinen, Timo Aila, and Samuli Laine. Guiding a diffusion model with a bad version of itself. Advances in Neural Information Processing Systems, 37:52996–53021, 2024. 1, 2, 3,

- 4, 5, 6

[39] Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. Analyzing and improving the training dynamics of diffusion models. In Proc. CVPR, 2024.

- 5, 6

- [40] Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. Analyzing and improving the training dynamics of diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24174–24184, 2024. 2, 5
- [41] Felix Krause, Timy Phan, Ming Gui, Stefan Andreas Baumann, Vincent Tao Hu, and Bj¨orn Ommer. Tread: Token routing for efficient architecture-agnostic diffusion training. arXiv preprint arXiv:2501.04765, 2025. 3, 7, 6
- [42] Nupur Kumari, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Ensembling off-the-shelf models for gan training. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10651–10662, 2022. 6

- [43] Tuomas Kynk¨a¨anniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. Advances in neural information processing systems, 32, 2019. 5, 1
- [44] Tuomas Kynk¨a¨anniemi, Miika Aittala, Tero Karras, Samuli Laine, Timo Aila, and Jaakko Lehtinen. Applying guidance in a limited interval improves sample and distribution quality in diffusion models. Advances in Neural Information Processing Systems, 37:122458–122483, 2024. 1, 2, 5, 6
- [45] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas M¨uller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space, 2025. 6
- [46] Xingjian Leng, Jaskirat Singh, Yunzhong Hou, Zhenchang Xing, Saining Xie, and Liang Zheng. Repa-e: Unlocking vae for end-to-end tuning with latent diffusion transformers. arXiv preprint arXiv:2504.10483, 2025. 3, 7, 1, 6
- [47] Tianhong Li, Dina Katabi, and Kaiming He. Return of unconditional generation: A self-supervised representation generation method. In Advances in Neural Information Processing Systems, pages 125441–125468. Curran Associates, Inc.,

2024. 6

- [48] Tianhong Li, Dina Katabi, and Kaiming He. Return of unconditional generation: A self-supervised representation generation method. Advances in Neural Information Processing Systems, 37:125441–125468, 2024. 6
- [49] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. Advances in Neural Information Processing Systems, 37:56424–56445, 2024. 1
- [50] Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori B Hashimoto, Luke Zettlemoyer, and Mike Lewis. Contrastive decoding: Open-ended text generation as optimization. In Proceedings of the 61st annual meeting of the association for computational linguistics (volume 1: Long papers), pages 12286–12312, 2023. 6
- [51] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 3, 6
- [52] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, pages 23–40. Springer, 2024. 1, 3, 5, 7
- [53] Charlie Nash, Jacob Menick, Sander Dieleman, and Peter W Battaglia. Generating images with sparse representations. arXiv preprint arXiv:2103.03841, 2021. 5, 1
- [54] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 7, 3
- [55] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF inter-

national conference on computer vision, pages 4195–4205,

2023. 1, 5, 7, 6

- [56] Pablo Pernias, Dominic Rampas, Mats L Richter, Christopher J Pal, and Marc Aubreville. W¨urstchen: An efficient architecture for large-scale text-to-image diffusion models. arXiv preprint arXiv:2306.00637, 2023. 6
- [57] Phillip Pope, Chen Zhu, Ahmed Abdelkader, Micah Goldblum, and Tom Goldstein. The intrinsic dimension of images and its impact on learning. arXiv preprint arXiv:2104.08894,

2021. 4

- [58] Sucheng Ren, Qihang Yu, Ju He, Xiaohui Shen, Alan Yuille, and Liang-Chieh Chen. Beyond next-token: Next-x prediction for autoregressive visual generation. arXiv preprint arXiv:2502.20388, 2025. 1
- [59] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2
- [60] Seyedmorteza Sadat, Jakob Buhmann, Derek Bradley, Otmar Hilliges, and Romann M Weber. Cads: Unleashing the diversity of diffusion models through condition-annealed sampling. arXiv preprint arXiv:2310.17347, 2023. 2, 6
- [61] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016. 5, 1
- [62] Axel Sauer, Kashyap Chitta, Jens M¨uller, and Andreas Geiger. Projected gans converge faster. Advances in Neural Information Processing Systems, 34:17480–17492, 2021. 6
- [63] Axel Sauer, Katja Schwarz, and Andreas Geiger. Styleganxl: Scaling stylegan to large diverse datasets. In ACM SIGGRAPH 2022 conference proceedings, pages 1–10, 2022. 6
- [64] Inkyu Shin, Chenglin Yang, and Liang-Chieh Chen. Deeply supervised flow-based generative models. arXiv preprint arXiv:2503.14494, 2025. 3
- [65] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2818–2826, 2016. 1
- [66] Zhicong Tang, Jianmin Bao, Dong Chen, and Baining Guo. Diffusion models without classifier-free guidance. arXiv preprint arXiv:2502.12154, 2025. 8
- [67] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839–84865, 2024. 1
- [68] Yuchuan Tian, Jing Han, Chengcheng Wang, Yuchen Liang, Chao Xu, and Hanting Chen. Dic: Rethinking conv3x3 designs in diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2469–2478,

2025. 6

- [69] Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Guy Wolf, and Yoshua Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport. arXiv preprint arXiv:2302.00482, 2023. 6

- [70] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 6
- [71] Liwei Wang, Chen-Yu Lee, Zhuowen Tu, and Svetlana Lazebnik. Training deeper convolutional networks with deep supervision. arXiv preprint arXiv:1505.02496, 2015. 3
- [72] Runqian Wang and Kaiming He. Diffuse and disperse: Image generation with representation regularization. arXiv preprint arXiv:2506.09027, 2025. 3, 5
- [73] Shuai Wang, Ziteng Gao, Chenhui Zhu, Weilin Huang, and Limin Wang. Pixnerd: Pixel neural field diffusion. arXiv preprint arXiv:2507.23268, 2025. 7
- [74] Shuai Wang, Zhi Tian, Weilin Huang, and Limin Wang. Ddt: Decoupled diffusion transformer. arXiv preprint arXiv:2504.05741, 2025. 1
- [75] Xi Wang, Nicolas Dufour, Nefeli Andreou, Marie-Paule Cani, Victoria Fern´andez Abrevaya, David Picard, and Vicky Kalogeiton. Analysis of classifier-free guidance weight schedulers. arXiv preprint arXiv:2404.13040, 2024. 2, 6
- [76] Ge Wu, Shen Zhang, Ruijing Shi, Shanghua Gao, Zhenyuan Chen, Lei Wang, Zhaowei Chen, Hongcheng Gao, Yao Tang, Jian Yang, et al. Representation entanglement for generation: Training diffusion transformers is much easier than you think. arXiv preprint arXiv:2507.01467, 2025. 3, 7, 6
- [77] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 6
- [78] Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15703–15712, 2025. 5, 7, 1, 2, 6
- [79] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940,

2024. 3, 7, 8, 1, 2, 6

- [80] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412, 2017. 6
- [81] Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690, 2025. 7, 1, 6
- [82] Hongkai Zheng, Weili Nie, Arash Vahdat, and Anima Anandkumar. Fast training of diffusion models with masked transformers. arXiv preprint arXiv:2306.09305, 2023. 3, 7
- [83] Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. ibot: Image bert pre-training with online tokenizer. arXiv preprint arXiv:2111.07832, 2021. 3
- [84] Rui Zhu, Yingwei Pan, Yehao Li, Ting Yao, Zhenglong Sun, Tao Mei, and Chang Wen Chen. Sd-dit: Unleashing the power of self-supervised discrimination in diffusion transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8435–8445, 2024. 3

## Guiding a Diffusion Transformer with the Internal Dynamics of Itself Supplementary Material

In this file, we provide more implementation and experimental details which are not included in the main text. In Section A, we provide the details of all the evaluation metrics and further compare the generation performance under uniform sampling. In Section B, we provide a detailed comparison of computational resource consumption during the sampling stage between the proposed IG and REPA. In Section C, we provide a comparison of the generate results under different sampling guidance methods. In Section D, we provide the experimental results of the proposed IG on the 512 × 512 ImageNet dataset. In Section E, we provide more hyperparameter settings and implementation details for Diffusion Transformer experiments of various scales. In Section F, We provide hyperparameter settings and implementation details for the 2D toy example. In Section G, We provide a more detailed discussion on the related work of our approach. In Section H, we provide more uncurated generation results on ImageNet 256×256 from the SiT-XL+IG with CFG.

- Table 6. Class-conditional performance on ImageNet 256×256 with balanced sampling. LightningDiT + IG reaches an FID of 1.24 without CFG, outperforming all prior state-of-the-art VQ-based and Diffusion-based methods by a large margin, and achieves an FID of 1.07 with CFG. For a fair comparison, all prior methods employ uniform balanced sampling 50K samples across 1,000 class labels.

w/o CFG w/ CFG or Autoguidance

Method Epochs #Params

FID↓ IS↑ Prec.↑ Rec.↑ FID↓ IS↑ Prec.↑ Rec.↑ Autoregressive

VAR [67] 350 2.0B 1.92 323.1 0.82 0.59 1.73 350.2 0.82 0.60 MAR [49] 800 943M 2.35 227.8 0.79 0.62 1.55 303.7 0.81 0.62 xAR [58] 800 1.1B - - - - 1.24 301.6 0.83 0.64

###### Latent Diffusion with Self-supervised Representation Model

REPA [79] 800 675M 5.78 158.3 0.70 0.68 1.29 306.3 0.79 0.64 DDT [74] 400 675M 6.27 154.7 0.68 0.69 1.26 310.6 0.79 0.65 REPA-E [46] 800 675M 1.70 217.3 0.77 0.66 1.15 304.0 0.79 0.66 LightningDiT [78] 800 675M 2.11 207.0 0.77 0.65 1.28 300.5 0.80 0.65 RAE (DiTDH) [81] 800 839M 1.51 242.9 0.79 0.63 1.13 262.6 0.78 0.67 LightningDiT+IG (ours) 680 678M 1.24 229.6 0.78 0.66 1.07 274.1 0.79 0.66

### A. Evaluation Metric

We strictly follow the setup and use the same reference batches of ADM [10] for evaluation, following their official implementation. We use NVIDIA A6000 pro GPUs or 4090Ti GPUs for evaluation and enable tf32 precision for faster generation. In what follows, we explain the main concept of metrics that we used for the evaluation.

- • FID [23] measures the feature distance between the distributions of real and generated images. It uses the Inception-v3 network [65] and computes distance based on an assumption that both feature distributions are multivariate gaussian distributions.
- • sFID [53] proposes to compute FID with intermediate spatial features of the Inception-v3 network to capture the generated images’ spatial distribution.
- • IS [61] also uses the Inception-v3 network but use logit for evaluation of the metric. Specifically, it measures a KL-divergence between the original label distribution and the distribution of logits after the softmax normalization.
- • Precision and recall [43] are based on their classic definitions: the fraction of realistic images and the fraction of training data manifold covered by generated data.

As stated in [81], using class-balanced sampling can lead to better results compared to random sampling because it more closely approximates the true label distribution of the training set. As the FID value approaches a lower range, these subtle differences begin to have a greater impact. The vast majority of VQ-based generation methods [49, 58, 67] have adopt the balanced sampling. To conduct a comprehensive and fair comparison with more baselines, we also adopt balanced sampling to conduct a comprehensive comparison of the current state-of-the-art VQ and Diffusion-based methods, as shown in Table 6.

### B. Computational Cost Comparison

We compare the computational efficiency of SiT-XL/2+REPA [79] and SiT-XL+IG under the same model scale in Table 7. IG introduces only a marginal increase in parameter count and FLOPs relative to SiT-XL, while maintaining neral identical latency. Despite the minimal computational overhead, IG yields substantial improvements in generation quality, achieving a relative reduction in FID, alongside an increase in IS. These results demonstrate that IG simultaneously improves generation quality and computational efficiency, highlighting its effectiveness as a general-purpose enhancement for generative models.

Table 7. Computational cost and performance comparison. This table compares REPA and IG on ImageNet 256×256, detailing model size, FLOPs, sampling steps, latency, and generation quality metrics. the proposed IG achieves substantially better sample quality with negligible increases in computational cost.

|Method<br><br>|#Params FLOPs↓ Latency (s)↓<br><br>|FID↓ IS↑|
|---|---|---|
|SiT-XL/2 + REPA<br><br>|675 114.46 6.18|5.90 157.8|
|SiT-XL/2 + IG<br><br>|678 (+0.44%)<br><br>114.47 (+0.01%)<br><br>6.19 (+0.16%)|1.75 (-70.34%)<br><br>228.6 (+44.87%)|

### C. Comparison of Guidance Methods

To further illustrate the advantages of our proposed IG, we compared it with the original CFG [24] and Autoguidance [38] methods. For CFG method, we use the vanilla LightningDiT-XL/1 model which is trained for 800 epochs and employed its official CFG coefficients. For the Autoguidance method, since it is difficult to determine the optimal bad version model, we initialized a 12-layer LightningDiT model and trained it for 10 epochs. We roughly searched for the optimal coefficients of Autoguidance and determined it to be 1.15. We also compare our IG with original Autoguidance on the vanilla EDM2-S [40] model on ImageNet 64 dataset with the same training setting, which is shown in Table 8. As shown in Table 9, our IG method outperforms both adding only CFG or only adding Autoguidance. Moreover, further combining CFG can achieve the current state-of-the-art in image generation, and has a significant improvement compared to other guidance methods.

Table 8. Comparision with original Autoguidance on the EDM2-S model on ImageNet 64 dataset.

Model EDM2-S EDM2-S+AutoGuidance EDM2-S+IG FID↓ 1.58 1.01 0.99

Table 9. Different guidance strategies comparison. This table compares various sampling guidance methods. We compare the results of the standard CFG, Autoguidance, as well as our proposed IG and the combination of IG and CFG. Our proposed IG is more flexible compared to Autoguidance and can significantly enhance the results when combined with CFG as a plug-and-play approach.

|Method<br><br>|Epochs<br><br>|FID↓ sFID↓ IS↑ Prec.↑ Rec.↑|
|---|---|---|
|LightningDiT-XL/1 + CFG [78] LightningDiT-XL/1 + Autoguidance [38] LightningDiT-XL/1 + IG (ours) LightningDiT-XL/1 + IG + CFG (ours)<br><br>|800 680 680 680|1.35 4.15 295.3 0.79 0.65 1.81 4.06 215.9 0.79 0.63 1.34 3.94 229.3 0.78 0.65 1.19 4.11 269.0 0.79 0.66<br><br>|

### D. 512×512 ImageNet

To further validate IG’s effectiveness, we conduct experiments at 512×512 resolution following REPA’s protocol [79] with Muon optimizer [34]. The RGB images are processed through the VAE [59] to yield 64×64×3 latents. As demonstrated in Table, IG surpasses the performance of REPA trained for 200 epochs and SiT-XL/2 trained for 600 epochs in terms of FID at only 60 epochs, demonstrating its superior effectiveness.

### E. Hyperparameter and More Implementation Details

More implementation details. We implement our models based on the original DiT, SiT and LightningDiT implementation. To speed up training and save GPU memory, we use mixed-precision (fp16) with gradient clipping and FlashAttention [8] operation for attention computation. We also pre-compute compressed latent vectors from raw pixels via stable diffusion VAE and VA-VAE and use these latent vectors. We do not apply any data augmentation as discussed in MaskDiT and REPA. For stable diffusion VAE, we use stabilityai/sd-vae-ft-ema for encoding images to latent vectors and decoding latent vectors to images. For the output head, we use a Linear layer to map the feature to the output size.

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Figure 8. Samples on ImageNet 512×512 from SiT-XL/2 + IG (1.4) using CFG with w = 1.8.

Table 10. Performance comparison on ImageNet 512×512.

Model Epochs FID↓ sFID↓ IS↑ Pre.↑ Rec.↑

Pixel diffusion VDM++ - 2.65 - 278.1 - ADM-G, ADM-U 400 2.85 5.86 221.7 0.84 0.53 Simple diffusion (U-Net) 800 4.28 - 171.0 - Simple diffusion (U-ViT, L) 800 4.53 - 205.3 - -

Latent diffusion, Transformer

MaskDiT 800 2.50 5.10 256.3 0.83 0.56 DiT-XL/2 600 3.04 5.02 240.8 0.84 0.54 SiT-XL/2 600 2.62 4.18 252.2 0.84 0.57 + REPA 80 2.44 4.21 247.3 0.84 0.56 + REPA 100 2.32 4.16 255.7 0.84 0.56 + REPA 200 2.08 4.19 274.6 0.83 0.58 + IG 60 1.78 3.92 286.3 0.81 0.62

The use of Muon In our large-scale experiment with the LightningDiT-XL/1 model, we adopt the Muon optimizer [34]. We find that in the environment of PyTorch, when using the pre-trained self-supervised representation model (such as DINOv2-B [54]) to align the encoder of the VAE, there will be unstable training issues in the early stage of training. After careful investigation and experimentation, we discover that replacing the AdamW optimizer with Muon could effectively alleviate such issue. Compared to AdamW, the Muon optimizer has a certain acceleration convergence effect in the early stage of model training, but we observe that the gap between them gradually narrowed as the training time increased, reaching similar final convergence results. In the later stage of training, there is a small probability of encountering training instability issues again.

Resuming the training with full precision can solve this problem.

Computing resources. We use NVIDIA A6000 pro 96 GB GPUs for training largest model (LightningDiT-XL/1, SiT-XL/2); and uses NVIDIA 4090 24GB GPUs for training smaller model (L, B). When sampling, we use either NVIDIA A6000 pro 96 GB GPUs or NVIDIA RTX 4090 24GB GPUs to obtain the samples for evaluation.

Table 11. Defualt hyperparameter setup. Unless other otherwise specified, we use these sets of hyperparameters for different models. In our ablation experiments, settings are also kept the same except those we point out in Table 11.

SiT-B SiT-L SiT-XL DiT-B DiT-L LightningDiT-XL

Architecture Input dim. 32×32×4 32×32×4 32×32×4 32×32×4 32×32×4 16×16×32 Patch size 2 2 2 2 2 2 Num. layers 12 24 28 12 24 28 Hidden dim. 768 1024 1152 768 1024 1152 Num. heads 12 16 16 12 16 16

###### Optimization

Batch size 256 256 256 256 256 1024 Optimizer AdamW AdamW AdamW AdamW AdamW Muon lr 0.0001 0.0001 0.0001 0.0001 0.0001 0.0002 (β1,β2) (0.9, 0.999) (0.9, 0.999) (0.9, 0.999) (0.9, 0.999) (0.9, 0.999) (0.9, 0.95)

EMA decay 0.9999 0.9999 0.9999 0.9999 0.9999 0.9995 Interpolants or Denoising

αt 1 − t 1 − t 1 − t - - 1 − t σt t t t - - t wt σt σt σt - - σt T - - - 1000 1000 Training objective v-prediction v-prediction v-prediction noise-prediction noise-prediction v-prediction Sampler Euler-Maruyama Euler-Maruyama Euler-Maruyama DDPM DDPM Heun Sampling steps 250 250 250 250 250 125 CFG Scale - - 1.35 (if used) - - 1.45 (if used)

IG Auxiliary supervision position 4 8 8 4 8 8 IG Scale 2.3 2.3 1.4 2.3 2.3 1.4

### F. Details of the 2D toy example

In this Section, we describe the construction of the 2D toy dataset used in the analysis of Section 4, as well as the associated model architecture, training setup, and sampling parameters.

Dataset. For each of the two classes c, we model the fractal-like data distribution as a mixture of Gaussians Mc = {ϕi},{µi},{Σi} , where ϕi, µi, and Σi represent the weight, mean, and 2×2 covariance matrix of each component i, respectively. This lets us calculate the ground truth scores and probability densities analytically and, consequently, to visualize them without making any additional assumptions. The probability density for a given class is given by

ϕi N(x;µi,Σi), where (8)

pdata(x | c) =

i∈Mc

- 1

- 2

1 (2π)2 det(Σ)

(x − µ)⊤Σ−1(x − µ) . (9)

exp −

N(x;µ,Σ) =

Applying heat diffusion to pdata(x|c), we obtain a sequence of increasingly smoothed densities p(x|c;σ) parameterized by noise level σ:

ϕi N(x;µi,Σ∗i,σ), where Σ∗i,σ = Σi + σ2I. (10) The score function of p(x|c;σ) is then given by

p(x | c;σ) =

i∈Mc

ϕi N(x;µi,Σ∗i,σ)(Σ∗i,σ)−1(µi − x) i∈Mc ϕi N(x;µi,Σ∗ i,σ)

∇x log p(x | c;σ) = i∈M

. (11)

c

We construct Mc to represent a thin tree-like structure by starting with one main “branch” and recursively subdividing it into smaller ones. Each branch is represented by 8 anisotropic Gaussian components and the subdivision is performed 6 times, decaying ϕ after each subdivision and slightly randomizing the lengths and orientations of the two resulting sub-branches. This yields 127×8 = 1016 components per class and 1016×2 = 2032 components in total. We define the coordinate system so that the mean and standard deviation of pdata, marginalized over c, are equal to 0 and σdata = 0.5 along each axis, respectively, matching the recommendations by Karras et al. [37].

Models. We implement the denoiser models D as simple multi-layer perceptrons, utilizing the magnitude-preserving design principles from EDM2 [40]. Following Autoguidance [38], we design the model interface so that for a given noisy sample, each model outputs a single scalar representing the logarithm of the corresponding unnormalized probability density, as opposed to directly outputting the denoised sample or the score vector. Concretely, let us denote the output of a given model by Gθ(x;σ,c). The corresponding normalized probability density is then given by

pθ(x | c;σ) = exp Gθ(x;σ,c) exp Gθ(x;σ,c) dx. (12)

By virtue of defining Gθ this way, we can derive the score vector, and by extension, the denoised sample, from Gθ through automatic differentiation:

∇x log pθ(x|c;σ) = ∇xGθ(x;σ,c) (13)

Dθ(x;σ,c) = x + σ2∇xGθ(x;σ,c). (14)

Besides Equation 13, according to the description of Autoguidance, trying out the alternative formulations where the model outputs the score vector or the denoised sample directly is qualitatively more or less identical.

To connect the above definition of Gθ to the raw network layers, we apply preconditioning using the same general principles as in EDM [36]. Denoting the function represented by the raw network layers as Fθ, we define Gθ as

- 1

- 2∥x∥22 −

gθ σn

Gθ(x;σ,c) = −

n

1 4

Fθ,i x∗;

i=1

log σ,c

2

##### x

, where x∗ =

σ2 + σdata2

(15)

and the sum is taken over the n output features of Fθ. We scale the output of Fθ by a learned scaling factor gθ that we initialize to zero. The goal of Equation 15 is to satisfy the following three requirements:

- • The input of Fθ should have zero mean and unit magnitude. This is achieved through the division by σ2 + σdata2 .

- • After initialization, Gθ should represent the best possible first-order approximation of the correct solution. This is achieved through the −21∥x∗∥22 term, as well as the fact that gθ = 0 after initialization.

- • After training, √gθ · Fθ should have approximately unit magnitude. This is achieved through the division by σn. In practice, we use an MLP with one input layer and four hidden layers, interspersed with SiLU [22] activation functions

and implemented using the magnitude-preserving primitives from EDM2 [39]. Additionally, we defined an output layer after the first hidden layer to produce a weaker version of the output. The input is a 4-dimensional vector x∗x;x∗y; 41 log σ;1 and the output of each hidden layer has n features, where n = 64.

Training. Given that we have the exact score function of the ground truth distribution readily available (Equation 10), we train the models using exact score matching [30] for simplicity and increased robustness. We thus define the loss function and the additional supervision loss as

(x;σ) − ∇x log p(x;σ)∥22 , (16)

train,x∼p(x;σ)σ2 ∥∇x log pθ

L(θi) = Eσ∼p

i

(x;σ) − ∇x log p(x;σ)∥22 , (17)

train,x∼p(x;σ)σ2 ∥∇x log pθ

L(θd) = Eσ∼p

d

where σ ∼ ptrain is realized as log(σ) ∼ N(Pmean,Pstd) [37] and θi and θd represent the intermediate layer and the deep layer network respectively. In particular, for the training acceleration method depicted in Figure 4, our loss function is

(x;σ) − ∇x log p(x;σ)∥22 , (18)

train,x∼p(x;σ)σ2 ∥∇x log pθ

L(θi) = Eσ∼p

i

(x;σ)))∥22 ,

train,x∼p(x;σ)σ2 ∥∇x log pθ

L(θd) = Eσ∼p

(x;σ) − (∇x log p(x;σ) + 0.5 ∗ (∇x log pθ

(x;σ) − ∇x log pθ

i

d

d

(19) The final training loss is L = L(θd) + 0.5 ∗ L(θi) The more commonly used denoising score matching do not show any significant differences in the model behavior or training dynamics.

We train these models for 4096 iterations using a batch size of 4096 samples. We set Pmean = −2.3 and Pstd = 1.5, and use αref/ max(t/tref,1) learning rate decay schedule with αref= 0.01 and tref= 512 iterations, along with a power function EMA profile [39] with σrel= 0.010. Overall, the setup is robust with respect to the hyperparameters; the phenomena illustrated in Figures 3 and 4 remain unchanged across a wide range of parameter choices.

Sampling. We use the standard EDM sampler [37] with N = 32 Heun steps (NFE = 63), σmin= 0.002, σmax= 5, and ρ = 7. We chose the values of N and σmax to be much higher than what is actually needed for this dataset in order to avoid potential discretization errors from affecting our conclusions. In Figure 3, we set w = 2.5 for CFG, w = 2 for autoguidance and w = 2 for internal guidance (IG), and w1 = 1 and w2 = 1.5 for IG + CFG. In Figure 4, we set w = 2 for internal guidance.

### G. More Discussion on Related Work

Denoising transformers. Many recent works have tried to use transformer backbones for diffusion or flow-based model training. First, several works like U-ViT [2], MDT [15], and DiffiT [18] how transformer-based backbones with skip connections can be an effective backbone for training diffusion models. Intriguingly, DiT and SiT show a pure transformer architecture can be a scalable architecture for training diffusion-based models. Based on these improvements, Stable diffusion 3 [11], FLUX 1 [45] and Nano Banana [17] show pure transformers can be scaled up for challenging text-to-image generation, and Sora, CogvideoX [77] and Wan [70] demonstrate their success in text-to-video generation. Our work analyzes and improves the training of Diffusion Transformer architecture based on a simple auxiliary supervision to the early layers.

Sampling guidance in diffusion models. Achieving better generation results over diffusion models remains challenging yet essential. Early approaches, such as classifier guidance [10], introduce control by incorporating classifier gradients into the sampling process. However, this method requires separately trained classifiers, making it less flexible and computationally demanding. To overcome these limitations, classifier free guidance (CFG) [24] is proposed, enabling guidance without the need for an external classifier. Instead, CFG trains conditional and unconditional models simultaneously and interpolates between their outputs during sampling. Subsequently, several studies [4, 7, 12, 14, 44, 60, 75] have reduced the downsides of CFG by making the guidance weight noise level-dependent. Another research direction separates the class guidance part from the CFG and only uses a weaker version of the current model for guidance [5, 26, 27, 29, 38], which also bears conceptual similarity to contrastive decoding [50] used in large language models to reduce the repetitiveness of generations. Our approach does not require any additional sampling steps. It can be applied to any deep diffusion Transformer network in a plug-and-play manner while achieving better generation results.

Efficient training of generative models. The issue of how to accelerate the convergence of generative models during the training process has also received significant attention. Previous studies have explored architectural optimization [41, 55, 68], improved flow-based theories [13, 51], optimized data [69, 80] to accelerate convergence. Meanwhile, there have been several approaches in generative adversarial network [16] that try to accelerate training with better convergence using pretrained visual encoder [35, 42, 62, 63]. Another line of work tries to exploit the pretrained visual encoders for improving diffusion model training from scratch [48, 56], usually by training two diffusion models where one model generates the pretrained representations and the other model generates the target data conditioned on the generated representation. More recently, representation learning has been introduced into generative model training to further enhance training efficiency [46, 47, 76, 78, 79, 81]. Our method not only alleviate the problem of gradient vanishing during training, but also can be directly applied in the sampling stage to achieve better generation results on models with less training consumption.

### H. More Qualitative Results

Below we show some uncurated generation results on ImageNet 256×256 from the SiT-XL+IG (1.8). We use classifier-free guidance with w = 2.5.

[Figure 53]

[Figure 54]

Class 024: great grey owl, great gray owl, Strix nebulosa

Class 022: bald eagle, American eagle, Haliaeetus leucocephalus

[Figure 55]

[Figure 56]

Class 089: sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita

Class 033: loggerhead, loggerhead turtle, Caretta caretta

[Figure 57]

[Figure 58]

Class 207: golden retriever

Class 250: Siberian husky

[Figure 59]

[Figure 60]

Class 264: Cardigan, Cardigan Welsh corgi Class 288: leopard, Panthera pardus

- Figure 9. Uncurated visualization results of LightningDiT-XL/1 + IG (1.8) use CFG with w = 2.5.

[Figure 61]

[Figure 62]

Class 388: giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca

Class 309: bee

[Figure 63]

[Figure 64]

Class 417: balloon Class 425: barn

[Figure 65]

[Figure 66]

Class 429: baseball Class 511: convertible

[Figure 67]

[Figure 68]

Class 661: Model T Class 780: schooner

- Figure 10. Uncurated visualization results of LightningDiT-XL/1 + IG (1.8) use CFG with w = 2.5.

[Figure 69]

[Figure 70]

Class 873: triumphal arch Class 928: ice cream, icecream

[Figure 71]

[Figure 72]

Class 930: French loaf Class 933: cheeseburger

[Figure 73]

[Figure 74]

Class 947: mushroom Class 971: bubble

[Figure 75]

[Figure 76]

Class 975: lakeside, lakeshore Class 980: volcano

- Figure 11. Uncurated visualization results of LightningDiT-XL/1 + IG (1.8) use CFG with w = 2.5.

