# arXiv:2511.23199v1[cs.CV]28Nov2025

## Vision Bridge Transformer at Scale

Zhenxiong Tan1 Zeqing Wang1 Xingyi Yang2,1 Songhua Liu3,1 Xinchao Wang1∗ 1National University of Singapore 2The Hong Kong Polytechnic University 3Shanghai Jiao Tong University

https://Yuanshi9815.github.io/ViBT homepage

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Source 3D style Manga style Source Change the scene Replace the item

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Add text "BRIDGE IT"

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Video Stylization Depth-to-Video

Figure 1. Results of vision bridge transformer on various vision translation tasks.

### Abstract

We introduce Vision Bridge Transformer (ViBT), a largescale instantiation of Brownian Bridge Models designed for conditional generation. Unlike traditional diffusion models that transform noise into data, Bridge Models directly model the trajectory between inputs and outputs, creating

∗Corresponding author. (xinchao@nus.edu.sg)

an efficient data-to-data translation paradigm. By scaling these models to 20B and 1.3B parameters, we demonstrate their effectiveness for image and video translation tasks. To support this scale, we adopt a Transformer architecture and propose a variance-stabilized velocity-matching objective for robust training. Together, these advances highlight the power of scaling Bridge Models for instruction-based image editing and complex video translation.

### 1. Introduction

Generative models have advanced remarkably, beginning with Generative Adversarial Networks (GANs) [16, 25] that enabled high-quality image synthesis via adversarial training. More recently, probability-path methods, especially diffusion models [18, 34, 47], have further elevated generative capabilities. Transformer-based architectures trained at scale have significantly enhanced the fidelity and diversity of synthesized images [8, 15, 42, 60] and videos [9, 51, 54]. Building upon this success, extending these models to conditional vision generation tasks has become a natural direction [49, 50, 64, 69]. Typically, these approaches inject visual conditions into the generative process by incorporating source images or videos as auxiliary inputs [64, 69].

Despite their success, the underlying noise-to-vision modeling paradigm widely adopted by these models [15, 15, 16] can be unnatural for conditional generation tasks. In this paradigm, models start from noise and gradually refine it toward the target [8, 29, 54, 60]. However, in many conditional scenarios (e.g., image editing, colorization, and frame interpolation), inputs already closely resemble the desired outputs, making this process unintuitive. Moreover, incorporating additional conditioning tokens introduces substantial computational overhead under transformer architectures [49, 50], especially in video settings [1, 22, 54].

In contrast, the vision-to-vision paradigm provides a more intuitive alternative by directly modeling the transformation between structured source and target domains [31, 58]. Unlike the noise-to-vision approach, it explicitly models the direct path from conditioning inputs to target outputs, naturally capturing the strong correlations inherent in the data. Previous works demonstrated the feasibility of vision-to-vision modeling using Bridge Models [31, 35, 72], which construct stochastic processes connecting source and target data distributions. Although Brownian Bridge based formulations [31] have shown promising results, prior work has largely been limited to small-scale architectures and relatively simple tasks [5, 56, 67, 71], leaving their potential for complex vision translation scenarios underexplored .

This work introduces Vision Bridge Transformer (ViBT), the first Brownian Bridge Model scaled to large-scale settings for complex vision translation tasks. ViBT leverages transformer architectures [42, 53] initialized from leading flow-matching models [54, 60], inheriting strong generative priors. By scaling ViBT to 20B and 1.3B parameters, we demonstrate the feasibility of applying Bridge Models to previously unexplored tasks within this framework.

However, scaling Bridge Models to such large-scale architectures necessitates a robust training objective. We observe that conventional displacement-style targets [31] disproportionately bias training toward early generation steps, while naive velocity-based objectives [5, 37] exhibit severe numerical instability, negatively affecting convergence and

performance. To address these issues, we propose variancestabilized velocity matching objective, which maintains numerical stability and equally emphasizes learning across all timesteps, facilitating efficient training at scale.

Extensive experiments demonstrate that ViBT effectively generalizes to a wide range of complex vision translation tasks, including instruction-based image editing, instruction-guided video stylization, depth-to-video synthesis, image coloring, and video frame interpolation, achieving results competitive with traditional conditional diffusion methods while being significantly more efficient. Comprehensive ablation studies further verify the effectiveness of the variance-stabilized velocity matching objective.

### 2. Related Works

Generative models Early generative models, such as Variational Autoencoders (VAEs) [44] and Generative Adversarial Networks (GANs) [16, 24], enabled high-quality synthesis through latent modeling and adversarial training. Diffusion models [18, 46] later introduced iterative denoising processes, significantly advancing image and video generation. Flow-matching models [15, 34] further reframed generation as learning deterministic or stochastic paths between distributions. More recently, transformer-based architectures trained at scale have further enhanced the fidelity and diversity of generative models [42, 54, 60].

Conditional generation Conditional diffusion models incorporate auxiliary signals such as text, images, poses, or depth maps through additional encoders, auxiliary branches, or cross-attention mechanisms. Representative methods include ControlNet [69], IP-Adapter [64], and T2IAdapters [41]. With the emergence of Diffusion Transformers (DiT) [42], recent approaches [29, 49, 60] incorporate conditions directly into transformer attention layers for stronger guidance. However, these methods introduce substantial computational overhead, especially in video tasks.

Bridge models Bridge Models [31, 35, 72] construct stochastic processes directly connecting source and target distributions, providing an alternative to noise-driven generation. Early approaches employed Schr¨odinger bridges [13], stochastic interpolants [2], and entropic optimal transport [12]. Recent diffusion-based variants, such as Denoising Diffusion Bridge Models (DDBM) [72] and Brownian Bridge methods [31], demonstrated promising results for conditional generation and image translation tasks.

Several recent works have highlighted the potential of Bridge Models in vision tasks, demonstrating improved structural and stylistic consistency in exemplar-guided image translation [30], enhanced temporal coherence in video synthesis [52], and increased efficiency during training and inference for basic image translation tasks [5, 21].

### 3. Preliminaries

Probability path modeling [13, 34, 37] defines a class of generative models that describe continuous-time processes transporting mass from a prior distribution p0 to a target distribution p1. Generally, these models are represented by a stochastic differential equation (SDE):

dXt = v(Xt,t)dt + σ(t)dWt, t ∈ [0,1], (1)

with boundary conditions X0 ∼ p0 and X1 ∼ p1, velocity field v : Rd×[0,1] → Rd, diffusion coefficient σ : [0,1] → R≥0, and standard Brownian motion Wt.

In practice, the velocity field is typically parameterized by a neural network vθ, trained via a velocity-matching objective [34, 37]:

0,x1), t, Xt ∥vθ(Xt,t) − ut(Xt |x0,x1)∥2 ,

L(θ) = E(x

(2) where ut(·|x0,x1) denotes the instantaneous velocity induced by a chosen teacher trajectory (deterministic or stochastic), and Xt is sampled accordingly. Throughout, we use uppercase Xt for stochastic states and lowercase xt for deterministic trajectories.

Rectified Flow Rectified Flow [15, 37] is a deterministic realization of probability path modeling obtained by setting

- σ(t) ≡ 0 in Eq. (1). It defines linear deterministic trajectories connecting endpoints x0 ∼ p0, typically sampled from a standard Gaussian distribution N(0,I), and x1 ∼ p1, drawn from the data distribution:

xt = (1 − t)x0 + tx1. (3)

Under this linear interpolation, the instantaneous velocity target simplifies to a constant vector:

ut ≡ x1 − x0. (4)

Brownian Bridge In contrast to the deterministic Rectified Flow, the standard Brownian Bridge [2, 31] incorporates stochasticity via a constant diffusion coefficient

- σ(t) ≡ 1. Given fixed endpoints (x0,x1), its conditional intermediate states follow a Gaussian distribution:

###### , t(1 − t)I

###### Xt |(x0,x1)∼ N (1 − t)x0 + tx1

###### .

linear interpolation

maximal variance at t=0.5

(5) Brownian Bridges are particularly suited to data-to-data transport tasks, such as denoising corrupted samples or translating data between structured image and video domains. Pairs of endpoints (x0,x1) are sampled from their respective source and target distributions. Under this stochastic formulation, the instantaneous velocity target used in velocity matching is expressed as:

x1 − Xt 1 − t

. (6)

ut(Xt |x0,x1) =

### 4. Methodology

Our method leverages a transformer-based architecture to model vision translation tasks in latent space. Given paired source and target data (images or videos), we encode them into latent representations x0 ∼ psource and x1 ∼ ptarget using a pre-trained VAE encoder [27], and apply the Brownian Bridge formulation to directly model the transformation from x0 to x1.

#### 4.1. Stabilized velocity matching

During training, given latent endpoint pairs (x0,x1) ∼ psource,target, we uniformly sample a time t ∈ [0,1] and Gaussian noise ϵ ∼ N(0,I). According to the conditional distribution of the Brownian bridge (5), the intermediate latent state xt is constructed as:

xt = (1 − t)x0 + tx1 + t(1 − t)ϵ. (7)

The velocity-based training target at this noisy state, derived from Eq. (6), is given by:

x1 − xt 1 − t

t 1 − t

ϵ. (8)

ut(xt|x1) =

= (x1 − x0) −

Accordingly, the training objective is given by the mean squared error between the predicted and target velocities:

0, x1 ∥vθ(xt,t) − ut(xt|x1)∥2 . (9) However, this objective faces critical issues as t → 1: the target velocity ut(xt|x1) diverges as O √1 1−t , causing instability, and the loss excessively focuses on these states, neglecting intermediate ones (Fig. 2).

Lvelocity = Et, ϵ, x

An alternative approach adopted in previous works [31] is to use a displacement-based training target defined as

dt(xt|x1) = x1 − xt. (10)

Accordingly, the displacement-based training objective is given by the mean squared error:

0, x1 ∥dθ(xt,t) − dt(xt|x1)∥2 .

Ldisplacement = Et, ϵ, x

(11) This displacement formulation naturally avoids numerical divergence, as it remains stable across all timesteps. However, its magnitude diminishes as t → 1 at the rate O(√1 − t), causing the training loss to be dominated by samples at smaller values of t.

Motivated by the above numerical instability and imbalanced loss across timesteps, we propose stabilized velocity matching, which introduces a normalization factor α(x0,x1,t) to balance loss contributions across timesteps. We rescale the original velocity target as:

ut(xt|x1) α(x0,x1,t)

. (12)

u˜t(xt|x1) =

Algorithm 1: Training Input: data pairs (x0,x1) ∼ psource,target, model vθ,

latent dimension D

- 1 repeat

- 2 Sample latent pair (x0,x1), interpolation time t ∼ U(0,1), and noise ϵ ∼ N(0,I);
- 3 Construct intermediate state xt = (1 − t)x0 + tx1 + t(1 − t)ϵ;

- 4 Compute velocity target ut = (x1 − xt)/(1 − t);
- 5 Compute normalization factor α2 = 1 + tD/[(1 − t)∥x1 − x0∥2];
- 6 Compute stabilized velocity loss

Lvelocity˜ = ∥v

θ(xt,t)−ut

α ∥2;

- 7 Update model parameters θ by gradient descent on Lvelocity˜ ;
- 8 until convergence;

##### Algorithm 2: Inference

Input: source-target latent pair (x0,x1), trained model vθ, latent dimension D, discretization steps N, discretization schedule 0 = t0 < t1 < ··· < tN = 1

- 1 Initialize x ← x0;
- 2 for k = 0,1,...,N − 1 do

- 3 Compute step size ∆t ← tk+1 − tk;
- 4 Compute scaling factor η ← ∆t1−t

k+1

1−tk ;

- 5 Sample noise ϵ ∼ N(0,I);
- 6 Update latent state: x ← x + ∆tvθ(x,tk) + η ϵ
- 7 end Output: Final state x approximating the target x1

Instantaneous target scale

Cumulative target contribution

- 4

- 0

- 1

| |Displacement<br><br>Velocity<br><br>Stabilized velocity| | |
|---|---|---|---|
| | | | |

()Ct

()St

- 0

0 t [0, 1] 1

0 t [0, 1] 1

###### Figure 2. Instantaneous and cumulative target contributions. S(t) = E∥τt∥2 with τt ∈ {dt, ut, u˜t}. C(t) =

t

0 S(s) ds 0.999

0 S(s)ds.

Specifically, we define α(x0,x1,t) based on the normalized root mean square magnitude of the velocity:

E ∥ut(xt|x1)∥2 ∥x1 − x0∥2

α(x0,x1,t)2 =

(13)

tD (1 − t)∥x1 − x0∥2

, (14)

= 1 +

where D denotes the latent dimensionality 1 . As shown in Fig. 2, this choice significantly reduces divergence and ensures balanced loss contributions throughout training.

The resulting stabilized velocity-matching objective is:

0, x1 v ˜θ(xt,t) − u˜t(xt|x1) 2 , (15)

Lvelocity˜ = Et, ϵ, x

where v˜θ(xt,t) = vθ(xt,t)/α(x0,x1,t) normalizes the network prediction for loss calculation only, while the network still directly predicts velocity. The complete training procedure is summarized in Algorithm 1.

1Derivations of factor α(·) are in the Supplementary Material C.

#### 4.2. Variance-corrected sampling

To sample from the trained Brownian Bridge model with stabilized velocity matching, we discretize the continuoustime SDE defined in Eq. (1) using the Euler-Maruyama discretization [38]. Given a schedule 0 = t0 < t1 < ··· < tN = 1, the sampling process starts from the source x0 and iteratively updates the latent state towards the target x1.

The standard Euler–Maruyama discretization yields:

xstandardk+1 = xk + ∆tk vθ(xk,tk) + ∆tk ϵk, (16)

where ∆tk = tk+1 − tk and {ϵk}Kk=0−1 are i.i.d. samples drawn from N(0,I). This scheme assumes a locally

constant variance structure, i.e., the stochastic term scales purely with √∆tk.

However, in the Brownian Bridge process, the variance should gradually shrink as the trajectory approaches the target x1, reflecting decreasing uncertainty near t = 1. Consequently, the noise magnitude in the naive discretization becomes overly large at late steps, leading to biased trajectories and degraded sample quality.

To correct this mismatch, a scaling factor can be applied to continuously modulate the variance across timesteps. In practice, the diffusion term is rescaled by the ratio 1−t

2, resulting in a variance-corrected stochastic update [2, 31]:

k+1 1−tk

1 − tk+1 1 − tk

xcorrectedk+1 = xk + ∆tk vθ(xk,tk) velocity toward target

+ ∆tk

ϵk

###### .

variance-corrected noise

(17) This correction ensures that the variance decays smoothly as t → 1, aligning the discrete sampling dynamics with the intrinsic structure of the Brownian Bridge. The complete inference procedure is summarized in Algorithm 2.

2Derivations of the scaling ratio are in the Supplementary Material C.

### 5. Experiments

We conduct extensive experiments to explore the effectiveness of scaling Brownian Bridge diffusion models across various complex vision conditional generation tasks. We begin with instruction-based image editing tasks in Section 5.1 to evaluate the model’s ability to perform finegrained and instruction-based content modification. Next, we extend our study to video stylization in Section 5.2, where input videos are transformed into target styles specified by textual instructions while preserving the original motion and structure. Finally, we examine video translation tasks focusing primarily on depth-to-video synthesis in Section 5.3. Additionally, detailed ablation studies in Section 5.4 validate the effectiveness of our proposed stabilized velocity-matching loss and explore key properties of the scaled Brownian Bridge diffusion process.

Training and inference details For image and video modalities, we respectively initialize our models from stateof-the-art pre-trained models: Qwen-Image-Editing [60] with 20B parameters for image-based tasks, and Wan

- 2.1 [54] with 1.3B parameters for video-based tasks. During training, the image model employs LoRA [19] with a rank of 128, while the video model undergoes full-parameter updates. We train our models using the Prodigy optimizer [39] with a learning rate of 1 and set save warmup=True. By default, we train each model for 20,000 iterations on

- 1 NVIDIA H100 GPU with a batch size of 1.

#### 5.1. Instruction-based image editing

We first evaluate our bridge models on the complex image editing task, which involves modifying specific content within an input image based on textual instructions while preserving other regions. In this task, the input image serves as the source domain psource, and the edited image represents the target domain ptarget. The brownian bridge model directly learns the transformation between the base image and the edited output.

Dataset We create a synthetic dataset for instructionbased image editing based on the Open Images Dataset [28]. Specifically, we first randomly sample 5,000 images and generate corresponding editing instructions using the vision-language model Qwen3-VL [61]. We then produce edited images based on these instructions using the Qwen-Image-Editing model [60]. Additionally, we enrich our dataset by incorporating stylized image data generated by OmniConsistency [48]. Finally, we filter the generated instruction-image pairs with Qwen3-VL to ensure high alignment between the instructions and the image edits. The detailed dataset construction process is described in the Supplementary Material, Section B.

[Figure 24]

[Figure 25]

[Figure 26]

“Change the background to a snowy tundra landscape.”

Source Image

Step1X-Edit FLUX Kontext

[Figure 27]

[Figure 28]

[Figure 29]

ChatGPT Ours

Qwen-Image-Edit

Figure 3. Qualitative comparison on image editing.

Model Add Adjust Extract Replace Remove Bg. Style Hybrid Action Avg. MagicBrush 2.84 1.58 1.51 1.97 1.58 1.75 2.38 1.62 1.22 1.83 Ins.Pix2Pix 2.45 1.83 1.44 2.01 1.50 1.44 3.55 1.20 1.46 1.88 AnyEdit 3.18 2.95 1.88 2.47 2.23 2.24 2.85 1.56 2.65 2.45 Step1X-Edit 3.88 3.14 1.76 3.40 2.41 3.16 4.63 2.64 2.52 3.06 UniWorld-V1 3.82 3.64 2.27 3.47 3.24 2.99 4.21 2.96 2.74 3.26 ViBT 4.20 3.70 2.31 3.86 2.91 3.92 4.85 2.72 3.52 3.55 FLUX Kontext 3.82 3.64 2.27 3.47 3.24 2.99 4.21 2.96 2.74 3.71 ViBT (s = 0.5)3 4.14 4.20 2.64 3.72 3.03 4.06 4.87 3.19 3.95 3.76 Qwen-Image-Edit 4.17 4.29 2.44 4.30 3.90 4.15 4.00 3.32 4.51 3.90

Table 1. Model ranking on ImgEdit-Bench based on average score.

Evaluation and baselines We adopt the ImgEditBench [65] as our evaluation benchmark, as it provides a comprehensive assessment across multiple editing dimensions, including instruction-following accuracy, editing quality, and preservation of image details. All evaluations presented in this section strictly follow the official protocols defined by ImgEdit-Bench. We compare our bridge model with representative diffusion-based methods, including InstructPix2Pix [4], Qwen-Image-Editing [60], Step1Xedit [36], FLUX.1 Kontext [29], and several other notable approaches [32, 66, 68].

Results and analysis Table 1 reports the quantitative results on ImgEdit-Bench4. ViBT performs on a similar level to current state-of-the-art methods across the different editing categories. In tasks such as object addition and style transfer, ViBT achieves notably stronger results, outperforming competing approaches by a clear margin. The qualitative results in Figure 3 and 4 show that ViBT produces clear instruction-following edits while keeping the original scene content, achieving visual quality comparable to leading diffusion-based methods.

3The noise scale is set to s = 0.5; details are given in Section 5.4. 4Results of baselines are reported from ImgEdit Bench [32, 33, 65].

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Source Image Move to the beach Change the color Add a silk Oil Painting Origami

Figure 4. Qualitative results of the image editing.

[Figure 36]

[Figure 37]

Source Video

[Figure 38]

Rave Ins.V2V

[Figure 39]

[Figure 40]

"Make this video drawing by Van Gogh"

TokenFlow

Ours

Figure 5. Comparison of stylized videos under the Van Gogh style.

CLIP Score ↑

Method NIQE ↓ TOPIQNR ↑ MUSIQ ↑ MANIQA ↑ CLIPIQA ↑

TokenFlow 4.767 0.376 55.186 0.267 0.378 0.683 Ins.V2V 4.268 0.467 60.621 0.306 0.440 0.827 RAVE 6.514 0.351 50.595 0.269 0.377 0.683 ViBT 4.328 0.503 64.045 0.348 0.486 0.782

Table 2. Quantitative results on the video stylization task.

#### 5.2. Video stylization

In the video domain, we first consider the instruction-based video stylization task, which aims to modify the visual style of an input video according to a given textual instruction while preserving its original content and motion dynamics.

Dataset and training We use the open-source Ditto-1M dataset [3] for training our bridge video model. Specifically, we randomly sample 10,000 video samples from the subset global style1 of Ditto-1M, which contains videos paired with style descriptions. We train bridge model on

- 4 NVIDIA H100 GPUs for 50,000 iterations in this task.

Evaluation and baselines For evaluation, we construct a benchmark comprising 100 videos generated by Wan

- 2.2 14B [54] using the first 100 prompts from MovieGen Bench [43], serving as inputs for the stylization task. These videos do not overlap with our training set. Each video is paired with a randomly sampled textual style instruction. We stylize videos consisting of 81 frames each, and

[Figure 41]

[Figure 42]

[Figure 43]

SourceVideoSculptureIllustrationArcher

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Figure 6. Qualitative comparisons of ViBT on the video stylization task with different styles.

uniformly sample 5 frames per video for quality assessment. We quantitatively evaluate these sampled frames using widely adopted image-quality metrics, including NIQE [40], TOPIQ-NR [7], MUSIQ [26], MANIQA [63], CLIPIQA [55], and CLIPScore [17]. These metrics comprehensively measure perceptual image quality, aesthetic appeal, and visual-semantic alignment with textual instructions. We compare our method against three diffusionbased video stylization methods, Instruct Video-to-Video (InsV2V) [11], RAVE [23], and TokenFlow [45].

Results and analysis Quantitative results in Table 2 show that ViBT outperforms the baselines in most metrics, demonstrating its effectiveness in generating high-quality stylized videos that align well with the given instructions. Qualitative comparisons in Figure 5 further illustrate that ViBT can apply the desired style to the input video while preserving the original motion and structure. Figure 6 futher demonstrates that ViBT can effectively stylize videos across various artistic styles while preserving original content and motion. More stylized video examples are available in the Supplementary Material, Section A.

Perceptual quality Ground truth similarity

Method Base Model

CLIP Score↑ VBench Score↑ NIQE↓ TOPIQNR↑ MUSIQ↑ MANIQA↑ CLIPIQA↑ SSIMc↑ PSNR↑ DISTS↓

ControlVideo SD 1.5 6.641 0.443 50.735 0.354 0.436 0.385 9.067 0.465 0.732 0.62 Control A Video SD 1.5 5.102 0.374 52.391 0.254 0.334 0.276 8.510 0.348 0.715 0.59 VideoComposer SD 2.1 6.750 0.305 43.691 0.276 0.237 0.329 9.656 0.457 0.722 0.59 Wan Fun Control Wan 2.1 5.346 0.477 59.086 0.335 0.459 0.427 10.899 0.281 0.776 0.69 ViBT Wan 2.1 4.896 0.477 59.625 0.331 0.477 0.429 11.403 0.230 0.781 0.71

Table 3. Quantitative comparison on the depth-to-video task.

###### Avg. Score

Subj. Cons.

Bkgd. Cons.

Aesth. Qual.

Img. Qual.

Obj. Class

Multi Objs.

Spatial Rel.

Temp. Style

Overall Cons.

Human Action

Temp. Flicker

Motion Smooth.

Dyn. Degree

Appear. Style

Method

Color

Scene

Control Video 0.899 0.94 0.54 0.52 0.57 0.26 0.706 0.46 0.29 0.20 0.24 0.80 0.991 0.990 0.11 0.229 0.55 Control A Video 0.791 0.88 0.48 0.59 0.59 0.25 0.799 0.44 0.43 0.21 0.24 0.83 0.982 0.976 0.72 0.235 0.59 Video Composer 0.873 0.92 0.44 0.48 0.67 0.23 0.854 0.32 0.29 0.22 0.24 0.91 0.963 0.949 0.88 0.222 0.59 Wan Fun 0.913 0.93 0.60 0.57 0.87 0.65 0.848 0.70 0.46 0.24 0.26 1.00 0.989 0.978 0.86 0.211 0.69 ViBT 0.907 0.93 0.63 0.63 0.91 0.71 0.835 0.74 0.54 0.25 0.27 1.00 0.990 0.976 0.82 0.221 0.71

Table 4. Quantitative comparison on the VBench attribute breakdown for the depth-to-video task.

#### 5.3. Video translation

To verify the versatility and generalization capability of bridge model, we further explore its application to video translation tasks. We primarily investigate depth-to-video synthesis, a fundamental yet challenging scenario.

Dataset and training To create the training dataset, we first generate 1,003 videos using Wan 2.2 14B with prompts sourced from the MovieGen Bench [43]. We then transform these synthesized videos into depth maps using the Depth Anything V2 [62] model, forming depth-video pairs for training. Detailed generation procedures are provided in Supplementary Material, Section B.

Evaluation and baselines We evaluate the brownian bridge model on the depth-to-video synthesis task, broadly adhering to the evaluation protocols outlined in VBench [20]. Specifically, we first generated 946 reference videos using Wan 2.2 14B based on the prompts provided by VBench, and subsequently converted these videos into corresponding depth maps. These depth maps were employed as conditioning inputs across all methods. Further details regarding the generation procedure are provided in the Supplementary Material, Section B.

For comprehensive assessment, we initially applied the quality metrics discussed in Section 5.2. We then augmented this analysis by introducing reference-based metrics including SSIM [59], PSNR [6], and DISTS [14], to quantitatively measure similarity between generated outputs and ground-truth videos. Additionally, we included the VBench Score [20] as an extra criterion to capture finer-grained and interpretable dimensions of video quality.

We compare ViBT against three representative diffusionbased controllable video generation models: Control-A-

[Figure 53]

[Figure 54]

[Figure 55]

"A vibrant orange bird ..."

Condition

VideoComposer Control A Video

[Figure 56]

[Figure 57]

[Figure 58]

ControlVideo

Wan Fun Control Ours

Figure 7. Qualitative comparison on the depth-to-video task.

Video [10], ControlVideo [70], and VideoComposer [57]. To provide a direct baseline for evaluating the effectiveness of our proposed method, we also incorporate Wan-Fun Control [1], a flow-matching-based method initialized from the same Wan 2.1 1.3B model as ViBT.

Results Table 3 presents quantitative comparisons on video frame quality, condition-following accuracy, textfollowing accuracy, and the overall VBench Score. Across all metrics, ViBT consistently outperforms the baselines, indicating strong video generation quality and reliable conditioning behavior. To further examine specific aspects, Table 4 reports fine-grained attribute evaluations under VBench, where ViBT achieves leading performance on most attributes. Figure 7 provides qualitative examples, showing that ViBT produces richer and more detailed visuals that align more closely with the depth conditions.

Additional experimental results, including qualitative evaluations on extra video translation tasks such as video interpolation and video colorization, can be found in the Supplementary Material, Section A.

###### Depth-to-Video Image Edit

Training Objective SSIM↑ PSNR↑ NIQE↓ DISTS↓ CLIP Score↑ VBench Score↑ Add Adjust Extract Replace Remove Bg. Style Compose Action Avg. Displacement 0.409 11.04 4.91 0.26 0.772 0.695 4.18 3.79 2.23 3.57 2.65 3.97 4.847 2.74 3.519 3.50 Velocity 0.428 10.81 5.45 0.27 0.773 0.698 4.09 3.89 2.19 3.34 2.13 3.90 4.897 2.62 3.149 3.36 Stabilized velocity 0.429 11.40 4.90 0.23 0.78 0.71 4.20 3.70 2.31 3.86 2.91 3.92 4.850 2.72 3.518 3.55

Table 5. Quantitative comparison of different training objectives.

Depth-to-Video Image Edit Noise Scale (s) SSIM↑ PSNR↑ NIQE↓ DISTS↓ CLIP Score↑ VBench Score↑ Add Adjust Extract Replace Remove Bg. Style Compose Action Avg.

- s = 0 0.347 9.808 5.432 0.3103 0.717 0.604 3.91 4.29 2.01 2.45 1.60 3.35 4.65 2.56 3.07 3.10 s = 0.1 0.331 9.206 5.413 0.3452 0.675 0.536 3.43 4.00 2.04 2.31 1.61 3.53 4.46 2.58 3.29 3.03 s = 0.5 0.398 10.227 5.185 0.2617 0.752 0.666 4.15 4.20 2.64 3.72 3.03 4.06 4.87 3.19 3.95 3.76

- s = 1 (default) 0.429 11.403 4.896 0.2304 0.781 0.709 4.20 3.70 2.31 3.86 2.91 3.92 4.85 2.72 3.52 3.55

- s = 2 0.396 11.305 4.499 0.2295 0.784 0.711 4.14 3.49 2.36 3.94 3.16 3.64 4.82 2.46 2.98 3.44 s = 4 0.394 10.146 5.912 0.3820 0.670 0.482 3.70 2.67 2.24 3.60 2.88 2.93 4.43 1.78 2.50 2.97

Table 6. Quantitative comparison across different noise scales (s).

[Figure 59]

[Figure 60]

"A lively brown-yellow dog ..."

| |Displacement<br><br>Velocity|
|---|---|
| |Stabilized velocity|
| | |
| | |
| | |

- 100
- 101

Loss(logscale)

Condition

Displacement

[Figure 61]

[Figure 62]

10 1

0 20K Training Step

Stabilized velocity

Velocity

(a) Training loss curves.

(b) Visualization results.

Figure 8. Comparison of different training objectives in depth-tovideo synthesis task.

#### 5.4. Ablation and analysis

Training objectives We compare three training objectives to validate our proposed stabilized velocity matching objective defined in Eq. (15), along with displacement matching Eq. (11) and velocity matching Eq. (2). Table 5 shows that stabilized velocity matching consistently achieves the best performance on both depth-to-video and image editing tasks. Specifically, it surpasses other objectives on all evaluated metrics for depth-to-video generation, and it also attains the highest average scores in diverse image editing scenarios. Moreover, Figure 8 highlights its superior training stability and improved visual quality compared to alternative objectives.

Noise scale Several previous works [5, 31] further extend the Brownian Bridge formulation by modifying the diffusion term in Eq. (1). Instead of fixing the diffusion coefficient as a constant σ(t) ≡ 1, they introduce a global noise scale parameter s such that σ(t) ≡ s, leading to the generalized SDE:

dXt = vθ(Xt,t)dt + sdWt. (18)

To investigate its impact, we conduct experiments across different values of s, summarizing results in Table 6. The corresponding training and inference modifications for this generalized formulation are detailed in the Supplementary Material, Section C. Our findings indicate that moderate noise scales (s = 1 or s = 2) achieve better performance for the depth-to-video task, with s = 2 showing strong overall scores. For image editing tasks, a smaller noise scale (s = 0.5) surprisingly achieves the highest average performance, notably outperforming the default s = 1 setting. However, excessively small (s < 0.5) or large (s > 2) noise scales significantly degrade quality on both tasks. These observations highlight that optimal noise scales differ across tasks, contrasting with previous work [5] advocating an extremely small noise scale (s = 0.005).

### 6. Conclusion

In this paper, we introduced the Visual Bridge Transformer, a large-scale instantiation of Brownian Bridge models, effectively scaling this paradigm to 20B parameters for conditional image and video generation. By proposing a stabilized velocity-matching objective, we addressed the numerical instability inherent in conventional training methods, significantly improving model stability and performance. Extensive experiments demonstrated that our framework consistently outperforms existing baselines across multiple challenging vision translation tasks, including instructionbased image editing and video translation tasks.

### 7. Limitations and future work

While our Visual Bridge Transformer demonstrates strong results, we observed that adjusting the noise scale s can further optimize performance across different vision tasks. Future work may explore adaptive or automated methods to select this parameter, potentially enhancing the versatility and effectiveness of Bridge Models.

## Vision Bridge Transformer at Scale Supplementary Material

### A. Additional experimental results

Efficiency comparison The Brownian Bridge formulation in ViBT enables more efficient training and inference by reducing reliance on auxiliary conditional branches or additional conditioning tokens. To quantitatively illustrate this potential advantage, we perform theoretical inference latency comparisons between ViBT and conventional conditional diffusion transformer (DiT) variants, which inject conditions by introducing extra tokens into attention layers. For image translation, ViBT is instantiated from QwenImage-Editing [60], while for video translation, ViBT is built upon Wan 2.1 1.3B [54]. Corresponding conditional DiT variants derived from these models serve as our baselines. We measure the inference latency for a single forward pass under a single NVIDIA H200 GPU, ensuring a clean architectural efficiency comparison independent from sampling schedules or runtime optimization.

Tables S1 and S2 detail the raw data for this comparison, including exact token counts and per-step latencies under various image resolutions and video settings. Figure S1 further visualizes the latency comparisons, clearly demonstrating that ViBT consistently reduces inference latency across all evaluated image and video translation scenarios compared to the conditional DiT baselines.

Additional video translation tasks Besides the depth-tovideo synthesis task presented in Section 5.3, we further evaluate ViBT on two additional video translation tasks: (1) video colorization and (2) video frame interpolation.

For video colorization, we directly apply ViBT to transform grayscale videos into colored videos. Figure S3 shows qualitative examples of video colorization results, highlighting ViBT’s strong generalization capability.

For video frame interpolation, we first construct a coarse video by repeating each original frame (except the first frame) k times in pixel space. ViBT is then applied to refine this coarse video, enhancing both visual quality and temporal coherence. Figure S2 illustrates this interpolation pipeline clearly. In our experiments, we set k = 4 to generate 4× interpolated frames between each original frame. This increases the frame rate of videos generated by Wan 2.1 from 15 FPS to 60 FPS, while maintaining high visual quality and temporal coherence. Qualitative results for this interpolation task are provided in Figure S4.

Notably, ViBT is capable of producing high-quality and temporally coherent results within only a few inference steps (e.g., 4 steps), demonstrating its efficiency.

Conditional DiT ViBT

Resolution

Tokens Latency (ms) Tokens Latency (ms) Speedup

1024 × 1024 8,192 437 4,096 192 2.28× 1328 × 1328 10,624 613 5,312 258 2.38×

Table S1. Inference efficiency comparison (image).

Conditional DiT ViBT

Resolution

Tokens Latency (ms) Tokens Latency (ms) Speedup

480P (5s) 65,520 1,510 32,760 459 3.29× 480P (10s) 127,920 5,407 63,960 1,444 3.74× 720P (5s) 151,200 7,437 75,600 1,958 3.80× 720P (10s) 295,200 28,577 147,600 7,097 4.03×

Table S2. Inference efficiency comparison (video).

Inference Latency Comparison

| |Image| | |Video<br><br>28.6| | | | |
|---|---|---|---|---|---|---|---|---|
| |Conditional DiT| | |Ours| | | | |
| | | | | | | | | |
| |0.44 0.19 0.61 0.26| | |1.51<br><br>5.41<br><br>7.44<br><br>0.46 1.44 1.96<br><br>7.10| | | | |
| | | | | | | | | |

30

ForwardLatency(s)

20

10

0

1024x1024 1328x1328 480P5s 480P10s 720P5s 720P10s

- Figure S1. Comparison between Conditional DiT and ViBT.

|1|
|---|

|2|
|---|

|3|
|---|

|1|2|2|2|2|3|3|3|3|
|---|---|---|---|---|---|---|---|---|

|1|2|3|4|5|6|7|8|9|
|---|---|---|---|---|---|---|---|---|

Source Frames

Result

Bridge Diffusion steps

Upsampled

- Figure S2. Illustration of video frame interpolation pipeline.

Ablation study on variance-corrected sampling To validate the effectiveness variance-corrected sampling strategy described in Eq. (17), we perform an ablation study by comparing it with the standard Euler-Maruyama discretization method without variance correction. Figure S5 provides qualitative results for this comparison on the image editing task. We observe that the naive discretization method (without variance correction) introduces noticeable artifacts, leading to degraded visual quality. In contrast, the variance-corrected sampling generates a cleaner and visually coherent image.

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

OutputSourceVideo

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

Figure S3. Qualitative results on video colorization task.

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

OutputSourceVideo

[Figure 82]

81 frams / 15 FPS

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

321 Frams / 60 FPS

Figure S4. Qualitative results on video frame interpolation task.

[Figure 91]

[Figure 92]

[Figure 93]

"Change to sunset view..."

Source image Naive discretization Variance-corredted

Figure S5. Ablation study on variance-corrected sampling.

Influence of inference steps and schedule We further investigate how the number of inference steps and the discretization schedule affect ViBT’s performance. As illustrated in Figure S6, increasing the inference steps consistently improves the generation quality. Moreover, the choice of timestep scheduler significantly influences the performance. Specifically, we adopt the shifting strategy introduced in Stable Diffusion 3 [15], which uses a shift coefficient γ to allocate more inference steps towards the earlier stages (t → 0) of the diffusion process. This shifted schedule is formulated as:

i γ N + (γ − 1)i

, (19)

ti =

where N denotes total steps and i the step index.

Figure S7 illustrates how increasing γ redistributes step density, placing more steps at earlier stages. Our experiments show that γ = 5 achieves significantly better visual quality than the linear schedule (γ = 1), especially with fewer inference steps (e.g., 4 or 8 steps).

[Figure 94]

"A serene landscape photograph in a soft, ethereal style, capturing the reflection of a pristine snow-capped mountain peak in a ..."

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Linear ( )

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

Shifted ( )

Figure S6. Ablation on inference steps and timestep schedule.

Timestep schedule ti

Step density

3

- 0
- 1

| |Linear ( = 1)<br><br>= 2 = 5 = 10<br><br>|
|---|---|
| | |

Relativedensity

ti

0

0 ti 1

0 Step index i N

Figure S7. Step density and timestep schedule for different γ.

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

Image Stylization

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Image Editing

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

Depth-to-Video

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

Video Stylization

Figure S8. Visualization of the intermediate stages in the ViBT bridge process.

Additional visualizations We provide supplementary qualitative results: Figure S8 visualizes intermediate generation states at different timesteps t in the ViBT bridge process, Figure S9 shows additional examples of image

stylization tasks, Figure S10 presents further results on instruction-based image editing, and Figure S11 provides extra visualizations of video stylization outputs.

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

Source Image

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

Vector style

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

Ink Style

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

Origami Style

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

Paper Cutting

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

Ghibli Style

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

Jojo Style

- Figure S9. Additional examples of image stylization generated by ViBT.

Source Image

Change the scene Change the color Replace Remove

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

Source Image Add different objects

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

- Figure S10. Additional qualitative results on instruction-based image editing.

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

Source Video

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

Different Styles

Figure S11. Additional results of video stylization tasks.

### B. Experimental details

Image editing dataset construction We construct our image editing dataset by first randomly sampling 5,000 images from the Open Images Dataset [28]. These images are cropped and resized into resolutions supported by the Qwen-Image-Editing model, specifically: 1328 × 1328

- (1:1), 1664 × 928 (16:9), 928 × 1664 (9:16), 1472 × 1104 (4:3), 1104×1472 (3:4), 1584×1056 (3:2), and 1056×1584
- (2:3). Subsequently, we generate corresponding editing instructions for these images using the vision-language model Qwen3-VL [61]. We then produce edited images based on these instructions using Qwen-Image-Editing [60]. To ensure high-quality alignment, we further score the generated instruction-image pairs using Qwen3-VL, filtering out pairs with low alignment scores. This filtered set constitutes Part 1 of our training data, comprising approximately 3,335 validated samples.

Additionally, we enrich the dataset by incorporating stylized images generated by OmniConsistency [48]. These images retain their original 1024 × 1024 resolution, with editing instructions uniformly formulated as “Convert the image to a [style] style image.” This augmentation forms Part 2 of our dataset, introducing further diversity with approximately 2,605 samples.

Depth-to-Video dataset construction To create the training dataset for depth-to-video synthesis, we first generate 1,003 videos using Wan 2.2 14B [54] with prompts sourced from the MovieGen Bench [43]. These videos are synthesized at a resolution of 832 × 480 with 81 frames each, using a classifier-free guidance (CFG) scale of 5 and 50 sampling steps.

We then transform these synthesized videos into depth maps using the Depth Anything V2 [62] model, forming depth-video pairs for training. It should be noted that the generated depth maps utilize the default inferno colormap format provided by Depth Anything V2, rather than grayscale images.

### C. Theoretical analysis and extensions

Normalization factor for stabilized velocity matching Conditioned on endpoints (x0,x1), the Brownian Bridge latent at time t can be written as

xt = (1 − t)x0 + tx1 + t(1 − t)ϵ, ϵ ∼ N(0,I).

(20) The velocity target is

x1 − xt 1 − t

. (21)

ut(xt | x1) =

Substituting xt gives

x1 − xt = (1 − t)(x1 − x0) − t(1 − t)ϵ, (22) ut(xt | x1) = (x1 − x0) −

t 1 − t

ϵ. (23)

We define the normalization factor via the (conditional) expected squared normlknjugytfrd5e4s3w2aq1 fc gvbhnjm

Eϵ ∥ut(xt | x1)∥2 ∥x1 − x0∥2

α(x0,x1,t)2 =

, (24)

where the expectation is taken over ϵ with (x0,x1) fixed. Using E[ϵ] = 0 and E[∥ϵ∥2] = D, we obtain

t 1 − t

Eϵ ∥ut(xt | x1)∥2 = x1 − x0 2 +

D, (25)

and hence

α(x0,x1,t)2 = ∥x1 − x0∥2 + 1−t tD ∥x1 − x0∥2

(26)

tD (1 − t)∥x1 − x0∥2

. (27)

= 1 +

This is the normalization factor used in the stabilized velocity loss in Eq. (15).

Depth-to-Video evaluation details For evaluation on the depth-to-video synthesis task, we generate 946 reference videos using Wan 2.2 14B based on the prompts provided by VBench [20]. These videos are also synthesized at a resolution of 832×480 with 81 frames each, using a CFG scale of 5 and 50 sampling steps. We then convert these videos into corresponding depth maps using the Depth Anything V2 [62] model, which are employed as conditioning inputs across all methods. The prompts used for generating the source videos are the extended versions [54]. However, for fair evaluation during testing, we use the original prompts provided by VBench.

Variance-corrected noise scaling Write the process as a deterministic interpolation plus a zero-mean Brownian Bridge:

Xt = (1 − t)x0 + tx1 + Bt, (28)

where {Bt}t∈[0,1] is a Brownian Bridge from 0 to 0. For 0 ≤ t1 ≤ t2 ≤ 1, its covariance satisfies

] = 0, (29) Var(Bt

E[Bt

2

) = t2(1 − t2)I, (30) Cov(Bt

2

) = t1(1 − t2)I. (31)

,Bt

1

2

Thus the conditional variance is

###### ) (32) − Cov(Bt

2 | Bt

Var(Bt

###### ) = Var(Bt

1

2

###### )−1 (33) · Cov(Bt

###### ,Bt

###### ) Var(Bt

2

1

1

) (34)

###### ,Bt

1

2

t21(1 − t2)2 t1(1 − t1)

I (35)

= t2(1 − t2)I −

(t2 − t1)(1 − t2) 1 − t1

I. (36)

=

Since the endpoints only affect the mean,

(t2 − t1)(1 − t2) 1 − t1

I. (37)

2 | Xt

Var(Xt

) =

1

For a discretization schedule

0 = t0 < t1 < ··· < tN = 1, (38) set t1 = tk, t2 = tk+1 and ∆tk = tk+1 − tk to obtain

1 − tk+1 1 − tk

I. (39)

k+1 | Xt

Var Xt

= ∆tk

k

Therefore an increment of the form

1 − tk+1 1 − tk

ϵk, ϵk ∼ N(0,I),

+ ∆tk

###### Xt

###### = Xt

k+1

k

(40) matches the Brownian Bridge conditional variance.

When discretizing

###### dXt = vθ(Xt,t)dt + dWt, (41)

the Euler–Maruyama update with variance correction becomes

xcorrectedk+1 = xk + ∆tk vθ(xk,tk) + ∆tk

1 − tk+1 1 − tk

ϵk, (42)

which is precisely the update in Eq. (17).

Training and inference with noise scale Under the generalized Brownian Bridge SDE in Eq. (18), we keep the network architecture and stabilized velocity objective unchanged, and only rescale the stochastic terms by the global noise scale s. Concretely, the intermediate state construction in training and the variance-corrected noise in inference are both multiplied by s, as summarized below.

- Algorithm S1: Training with noise scale s

Input: data pairs (x0,x1) ∼ psource,target, model vθ,

latent dimension D, noise scale s

- 1 repeat

- 2 Sample latent pair (x0,x1), interpolation time t ∼ U(0,1), and noise ϵ ∼ N(0,I);
- 3 Construct intermediate state xt = (1 − t)x0 + tx1 + s t(1 − t)ϵ;

- 4 Compute velocity target ut = (x1 − xt)/(1−t);
- 5 Compute normalization factor α2 = 1 + s2tD/[(1 − t)∥x1 − x0∥2];
- 6 Compute stabilized velocity loss

Lvelocity˜ = ∥v

θ(xt,t)−ut

α ∥2;

- 7 Update model parameters θ by gradient descent on Lvelocity˜ ;
- 8 until convergence;

- Algorithm S2: Inference with noise scale s

Input: source-target latent pair (x0,x1), trained model vθ, latent dimension D, discretization steps N, discretization schedule 0 = t0 < t1 < ··· < tN = 1, noise scale s

- 1 Initialize x ← x0;
- 2 for k = 0,1,...,N − 1 do

- 3 Compute step size ∆t ← tk+1 − tk;
- 4 Compute scaling factor η ← s ∆t1−t

k+1

1−tk ;

- 5 Sample noise ϵ ∼ N(0,I);
- 6 Update latent state: x ← x + ∆tvθ(x,tk) + η ϵ
- 7 end Output: Final state x approximating the target x1

### Acknowledgment

This project is supported by NUS IT’s Research Computing group under grant number NUSREC-HPC-00001. We thank Ruonan Yu and Sicheng Feng for helpful discussions.

### References

- [1] aigc-apps. VideoX-Fun: A more flexible framework that can generate videos at any resolution and create videos from images. https://github.com/aigc-apps/VideoXFun, 2025. VideoX-Fun is an open-source video generation pipeline supporting AI image and video generation and training baseline and LoRA models. 2, 7
- [2] Michael S Albergo, Nicholas M Boffi, and Eric VandenEijnden. Stochastic interpolants: A unifying framework for flows and diffusions. arXiv preprint arXiv:2303.08797,

2023. 2, 3, 4

- [3] Qingyan Bai, Qiuyu Wang, Hao Ouyang, Yue Yu, Hanlin Wang, Wen Wang, Ka Leong Cheng, Shuailei Ma, Yanhong Zeng, Zichen Liu, Yinghao Xu, Yujun Shen, and Qifeng Chen. Scaling instruction-based video editing with a highquality synthetic dataset. arXiv preprint arXiv:2510.15742,

2025. 6

- [4] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023. 5
- [5] Cl´ement Chadebec, Onur Tasar, Sanjeev Sreetharan, and Benjamin Aubin. Lbm: Latent bridge matching for fast image-to-image translation. arXiv preprint arXiv:2503.07535, 2025. 2, 8
- [6] Chaofeng Chen and Jiadi Mo. IQA-PyTorch: Pytorch toolbox for image quality assessment. [Online]. Available: https://github.com/chaofengc/IQAPyTorch, 2022. 7
- [7] Chaofeng Chen, Jiadi Mo, Jingwen Hou, Haoning Wu, Liang Liao, Wenxiu Sun, Qiong Yan, and Weisi Lin. Topiq: A top-down approach from semantics to distortions for image quality assessment. IEEE Transactions on Image Processing, 33:2404–2418, 2024. 6
- [8] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023. 2
- [9] Junsong Chen, Yuyang Zhao, Jincheng Yu, Ruihang Chu, Junyu Chen, Shuai Yang, Xianbang Wang, Yicheng Pan, Daquan Zhou, Huan Ling, et al. Sana-video: Efficient video generation with block linear diffusion transformer. arXiv preprint arXiv:2509.24695, 2025. 2
- [10] Weifeng Chen, Jie Wu, Pan Xie, Hefeng Wu, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. Control-a-video: Controllable text-to-video generation with diffusion models,

2023. 7

- [11] Jiaxin Cheng, Tianjun Xiao, and Tong He. Consistent videoto-video transfer using synthetic dataset. arXiv preprint arXiv:2311.00213, 2023. 6

- [12] Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. Advances in neural information processing systems, 26, 2013. 2
- [13] Valentin De Bortoli, James Thornton, Jeremy Heng, and Arnaud Doucet. Diffusion schr¨odinger bridge with applications to score-based generative modeling. Advances in neural information processing systems, 34:17695–17709, 2021. 2, 3
- [14] Keyan Ding, Kede Ma, Shiqi Wang, and Eero P Simoncelli. Image quality assessment: Unifying structure and texture similarity. IEEE transactions on pattern analysis and machine intelligence, 44(5):2567–2581, 2020. 7
- [15] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

2024. 2, 3, 10

- [16] Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. 2
- [17] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. In Proceedings of the 2021 conference on empirical methods in natural language processing, pages 7514–7528, 2021. 6
- [18] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2
- [19] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022. 5
- [20] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 7, 15
- [21] Haorui Ji, Tao Jun Lin, and Hongdong Li. Dpbridge: Latent diffusion bridge for dense prediction. ArXiv, abs/2412.20506, 2024. 2
- [22] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. arXiv preprint arXiv:2503.07598, 2025. 2
- [23] Ozgur Kara, Bariscan Kurtkaya, Hidir Yesiltepe, James M. Rehg, and Pinar Yanardag. Rave: Randomized noise shuffling for fast and consistent video editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 6
- [24] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019. 2
- [25] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In Proceedings of

- the IEEE/CVF conference on computer vision and pattern recognition, pages 8110–8119, 2020. 2
- [26] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5148–5157, 2021. 6
- [27] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 3
- [28] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. International journal of computer vision, 128(7):1956–1981, 2020. 5, 15
- [29] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742,

2025. 2, 5

- [30] Eungbean Lee, Somi Jeong, and Kwanghoon Sohn. Ebdm: exemplar-guided image translation with brownian-bridge diffusion models. In European Conference on Computer Vision, pages 306–323. Springer, 2024. 2
- [31] Bo Li, Kaitao Xue, Bin Liu, and Yu-Kun Lai. Bbdm: Imageto-image translation with brownian bridge diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern Recognition, 2023. 2, 3, 4, 8
- [32] Zongjian Li, Zheyuan Liu, Qihui Zhang, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Yang Ye, Wangbo Yu, Yuwei Niu, and Li Yuan. Uniworld-v2: Reinforce image editing with diffusion negative-aware finetuning and mllm implicit feedback. arXiv preprint arXiv:2510.16888, 2025. 5
- [33] Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. Uniworld: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025. 5
- [34] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 2, 3
- [35] Guan-Horng Liu, Arash Vahdat, De-An Huang, Evangelos A Theodorou, Weili Nie, and Anima Anandkumar. I 2 sb: Image-to-image schr\” odinger bridge. arXiv preprint arXiv:2302.05872, 2023. 2
- [36] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025. 5
- [37] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 2, 3
- [38] Gisiro Maruyama. Continuous markov processes and stochastic equations. Rendiconti del Circolo Matematico di Palermo, 4(1):48–90, 1955. 4

- [39] Konstantin Mishchenko and Aaron Defazio. Prodigy: An expeditiously adaptive parameter-free learner. In Forty-first International Conference on Machine Learning, 2024. 5
- [40] Anish Mittal, Rajiv Soundararajan, and Alan C Bovik. Making a “completely blind” image quality analyzer. IEEE Signal processing letters, 20(3):209–212, 2012. 6
- [41] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI conference on artificial intelligence, pages 4296–4304, 2024. 2
- [42] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

- 2023. 2

[43] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, ChihYao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720,

- 2024. 6, 7, 15

- [44] Yunchen Pu, Zhe Gan, Ricardo Henao, Xin Yuan, Chunyuan Li, Andrew Stevens, and Lawrence Carin. Variational autoencoder for deep learning of images, labels and captions. Advances in neural information processing systems, 29, 2016. 2
- [45] Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. arXiv preprint arXiv:2412.03069, 2024. 6
- [46] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2
- [47] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 2
- [48] Yiren Song, Cheng Liu, and Mike Zheng Shou. Omniconsistency: Learning style-agnostic consistency from paired stylization data. arXiv preprint arXiv:2505.18445, 2025. 5, 15
- [49] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14940–14950, 2025. 2
- [50] Zhenxiong Tan, Qiaochu Xue, Xingyi Yang, Songhua Liu, and Xinchao Wang. Ominicontrol2: Efficient conditioning for diffusion transformers. arXiv preprint arXiv:2503.08280,

2025. 2

- [51] Meituan LongCat Team, Xunliang Cai, Qilong Huang, Zhuoliang Kang, Hongyu Li, Shijun Liang, Liya Ma, Siyu Ren, Xiaoming Wei, Rixu Xie, et al. Longcat-video technical report. arXiv preprint arXiv:2510.22200, 2025. 2
- [52] Viacheslav Vasilev, Arseny Ivanov, Nikita Gushchin, Maria Kovaleva, and Alexander Korotin. Time-correlated video bridge matching. ArXiv, abs/2510.12453, 2025. 2

- [53] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 2
- [54] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2, 5, 6, 9, 15
- [55] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In Proceedings of the AAAI conference on artificial intelligence, pages 2555–2563, 2023. 6
- [56] Peiyong Wang, Bohan Xiao, Qisheng He, Carri Glide-Hurst, and Ming Dong. Score-based image-to-image brownian bridge. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 10765–10773, 2024. 2
- [57] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. Advances in Neural Information Processing Systems, 36:7594–7611, 2023. 7
- [58] Yuji Wang, Zehua Chen, Xiaoyu Chen, Yixiang Wei, Jun Zhu, and Jianfei Chen. Framebridge: Improving imageto-video generation with bridge models. arXiv preprint arXiv:2410.15371, 2024. 2
- [59] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 7
- [60] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025. 2, 5, 9, 15
- [61] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 5, 15
- [62] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. arXiv:2406.09414, 2024. 7, 15
- [63] Sidi Yang, Tianhe Wu, Shuwei Shi, Shanshan Lao, Yuan Gong, Mingdeng Cao, Jiahao Wang, and Yujiu Yang. Maniqa: Multi-dimension attention network for no-reference image quality assessment. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1191–1200, 2022. 6
- [64] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 2

- [65] Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025. 5
- [66] Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang

- Zhang, and Yueting Zhuang. Anyedit: Mastering unified high-quality image editing for any idea. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26125–26135, 2025. 5
- [67] Conghan Yue, Zhengwei Peng, Junlong Ma, Shiyan Du, Pengxu Wei, and Dongyu Zhang. Image restoration through generalized ornstein-uhlenbeck bridge. arXiv preprint arXiv:2312.10299, 2023. 2
- [68] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instructionguided image editing. Advances in Neural Information Processing Systems, 36:31428–31449, 2023. 5
- [69] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023. 2
- [70] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. Controlvideo: Training-free controllable text-to-video generation. arXiv preprint arXiv:2305.13077, 2023. 7
- [71] Kaiwen Zheng, Guande He, Jianfei Chen, Fan Bao, and Jun Zhu. Diffusion bridge implicit models. arXiv preprint arXiv:2405.15885, 2024. 2
- [72] Linqi Zhou, Aaron Lou, Samar Khanna, and Stefano Ermon. Denoising diffusion bridge models. ArXiv, abs/2309.16948,

2023. 2

