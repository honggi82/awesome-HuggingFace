[Figure 1]

## DICE: Discrete Inversion Enabling Controllable Editing for Masked Generative Models

# arXiv:2410.08207v3[cs.CV]12Nov2025

Xiaoxiao He1∗, Quan Dao1∗, Ligong Han1,2,3†, Song Wen1, Minhao Bai1, Di Liu1, Han Zhang4, Felix Juefei-Xu5, Chaowei Tan1, Bo Liu6, Martin Renqiang Min7, Kang Li1, Faez Ahmed8, Akash Srivastava2,3, Hongdong Li9, Junzhou Huang10, & Dimitris N. Metaxas1 1Rutgers University 2MIT-IBM Watson AI Lab 3 Red Hat AI Innovation 4Google DeepMind 5NYU 6Walmart Global Tech 7NEC Labs America 8Massachusetts Institute of Technology 9ANU 10UT Arlington ∗ Equal Contributions † Project Lead & Corresponding Author Project Website: [Link]

### Abstract

Recent advances in discrete diffusion models have demonstrated strong performance in image generation and masked language modeling, yet they remain limited in their capacity for controlled content editing. We propose DICE (Discrete Inversion for Controllable Editing), a novel framework that pioneers precise inversion capabilities for discrete diffusion models, including both masked generative and multinomial diffusion variants. Our key innovation lies in capturing noise sequences and masking patterns during reverse diffusion process, enabling both accurate reconstruction and flexible editing without relying on predefined masks or attention-based manipulations. Through comprehensive experiments across image and text modalities using models such as Paella, VQDiffusion, RoBERTa and LLaDA, we demonstrate that DICE successfully maintains high fidelity to the original data while significantly expanding editing capabilities. These results establish new possibilities for fine-grained content manipulation in discrete spaces.

### 1. Introduction

Continuous diffusion models operate in continuous spaces, leveraging stochastic differential equations (SDEs) or their deterministic counterparts, ordinary differential equations (ODEs), to model the forward and reverse diffusion processes [49, 51].

Advances such as flow matching [12, 29, 30] have enhanced their efficiency and flexibility. These models have been successfully applied in various domains, including image editing [4, 19, 21, 36, 38, 63], medical imaging [22], and solving inverse problems [11, 52]. In image editing, continuous diffusion models enable controlled manipulation of images while preserving consistency with the underlying

Input Image Inpainting w/ Mask Ours (w/o Mask)

[Figure 2]

[Figure 3]

[Figure 4]

| |
|---|

Black and white cat dog on floor

Figure 1. Illustration of the limitation of masked inpainting method. Inpainting with masked generation inadvertently modifies the orientation of the head, resulting in a less favourable result. With our discrete inversion method, we are able to edit the image while preserving other properties of the object being edited. This is achieved by injecting the information from the input image into the logit space. Dotted red box indicates the masked region.

data distribution. A key capability enabling this is inversion—the process of reversing the diffusion model to recover the original noise vector or latent representation that could have generated a given data sample. Two main inversion approaches exist: deterministic inversion using ODEs (e.g., DDIM Inversion [49]) and stochastic inversion by recording noise sequences (e.g., CycleDiffusion [59], DDPM Inversion [26]).

Discrete diffusion models are designed for inherently discrete data such as text or image tokens [15]. They adapt the diffusion framework to discrete spaces by defining appropriate transition kernels that corrupt and restore discrete data [3, 18, 25]. Prominent examples include multinomial diffusion [18, 25], D3PM [3], and masked generative models like MaskGIT [9], Muse [8]. Despite their success in generation tasks, discrete diffusion models face limitations in controlled content editing. For instance, masked generative models achieve image editing through masked inpainting,

where regions are masked and regenerated based on new conditions. However, this approach, as illustrated in Figure 1, lacks the ability to inject information from the masked area into the inpainting process, limiting fine-grained control over the editing outcome.

Moreover, existing ODE-based inversion techniques developed for continuous diffusion models are not directly applicable to discrete diffusion models due to inherent differences in data representation and diffusion processes. This gap hinders the ability to perform precise inversion and controlled editing in discrete spaces. Thus, we propose DICE (Discrete Inversion for Controllable Editing), the first inversion algorithm for discrete diffusion models to the best of our knowledge. Our method extends the stochastic inversion approach to discrete diffusion models, including both multinomial diffusion and masked generative models. The core idea is to record the noise sequence to recover a stochastic trajectory in the reverse diffusion process. Specifically, given an artificial trajectory where latent states have low correlation, we fit reverse sampling steps to this trajectory and save the residuals between targets and predictions. This process imprints the information of the original input data into the recorded residuals. During editing or inference, the residuals are added back, allowing us to inject and control the amount of information introduced into the inference process.

Our approach enables accurate reconstruction of the original input data and facilitates controlled editing without the need for predefined masks or attention map manipulation. It provides a flexible framework for fine-grained content manipulation in discrete spaces, overcoming the limitations of existing methods. We validate the effectiveness of DICE through extensive experiments on both image and text modalities. We evaluate our method on models such as VQ-Diffusion [18], Paella [43], and RoBERTa [31], demonstrating its versatility across different types of discrete generative models. Additionally, we introduce a novel text-editing dataset to further showcase our method’s capabilities and to facilitate future research in this area. Our contributions can be summarized as follows:

- • We introduce DICE, an inversion algorithm for discrete diffusion models, including multinomial diffusion and masked generative models. By recording and injecting noise sequences or masking patterns, DICE enables accurate reconstruction and controlled editing of discrete data without predefined masks or attention manipulation.
- • We validate the effectiveness of DICE through comprehensive experiments on both image and text modalities, demonstrating its versatility across different types of discrete generative models.
- • We show that our approach can transform a model primarily trained for understanding tasks, such as RoBERTa, into a competitive generative model for text generation and editing, illustrating the potential for extending dis-

crete diffusion models to new applications.

### 2. Related Work

Discrete diffusion. Generative modeling over discrete spaces using diffusion models has been extensively explored in recent years. Early developments include [48], Argmax Flows and Multinomial Diffusion [25], and D3PM [3], which model the forward process as a discrete-time, discrete-state Markov chain. These methods train a neural network to reverse this process via a variational objective. Subsequent works such as [14] and [18] leveraged VQ-GAN to tokenize images, enabling efficient non-autoregressive generation strategies such as MaskGIT [9], Muse [8], and MMVID [20] that iterate masking and prediction. Drawing inspiration from continuous diffusion models trained via score matching [50], recent methods introduce discrete analogues through ratio matching [32, 35], which learn unnormalized density ratios. Discrete flow matching has also been proposed in this direction [17]. In natural language processing, models such as BERT [13] and RoBERTa [31] have been interpreted as instances of discrete denoising [55]. More recently, there has been a surge of interest in developing diffusion-based language models [2, 32, 39, 45, 47, 60, 64], with successful systems demonstrating strong scalability and competitive performance with autoregressive LLMs.

Diffusion inversion. Diffusion inversion is the problem of taking an image and a text prompt that describes it and finding a noise latent that would generate the exact same image. More broadly, it refers to recovering the latent or noise vector that reproduces a given input under a diffusion model. Traditional approaches include deterministic inversion via neural ODEs [10], such as DDIM inversion [49] and flow matching [29, 30], which reverse learned forward trajectories. Stochastic methods based on SDEs [51], including CycleDiffusion [59] and DDPM Inversion [26], reconstruct the input by tracking noise along stochastic paths. To improve inversion quality, ReNoise [16, 40] applies fixed-point iterations, and GNRI [46] uses a Newton-Raphson scheme to solve the DDIM inversion equation. Null-text Inversion [38] improves reconstruction by optimizing null embeddings at test time, while Negative-Prompt Inversion [21, 37] introduces a closed-form approximation that reduces runtime without sacrificing fidelity. Our approach generalizes DDPM Inversion to discrete diffusion models, enabling effective inversion across both continuous and discrete domains.

Inversion-based image editing. A large body of diffusionbased image editing methods are grounded in DDIM inversion [49], which serves as the basis for reconstructing editable latents. These approaches often incorporate additional guidance mechanisms for controlled manipulation. For example, Prompt-to-Prompt [23] modifies crossattention maps, while Plug-and-Play [54], TF-ICON [33], and StyleAligned [24] expand this to self-attention layers.

𝑡 + 1 𝑡 𝑡 − 1

𝑡 𝑡 − 1 q-sample

[Figure 5]

[Figure 6]

[Figure 7]

𝑥  |  = 𝒟 𝑥 ,𝑐,𝑡

[Figure 8]

[Figure 9]

[Figure 10]

ForwardDDIM

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

𝒙 𝒙

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

𝒙 𝒙

𝒙 𝒙

ReverseDDIM

[Figure 22]

𝒙 |𝒄

q-sample

q-sample

q-sample

𝒄:

source prompt

predict 𝒙 latent 𝒛

posterior mean latent 𝒛

“Black and white cat on floor”

(c) Non-ODE reconstruction

- (e) Inversion for MGM
- (f) Editing for MGM

(a) ODE-based reconstruction

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

ForwardDDIM

𝒙 |𝒄′

𝒙 |𝒄′

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

𝒙 𝒙

𝒙

𝒙

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

𝑥  |  = 𝒟 𝑥 ,𝑐′,𝑡

ReverseDDIM

[Figure 41]

[Figure 42]

[Figure 43]

𝒙 |𝒄′

𝒙 |𝒄′

𝒙 |𝒄′

𝒄′:

editing prompt

𝑡 𝑡 − 1

predict 𝒙 free sampling w/ 𝒄′

posterior mean free sampling w/ 𝒄′

“Black and white cat dog on floor”

[Figure 44]

[Figure 45]

(b) ODE-based editing

(d) Non-ODE editing

- Figure 2. Here we demonstrate the two types of reconstruction and editing paradigms, namely ODE-based and Non-ODE based. (a,b) shows the ODE-based editing and reconstructions, while it provides accurate editing and reconstruction performances, it highly depends on the underlying ODE trajectory, which is not feasible in the discrete diffusion. However, the Non-ODE editing samples a trajectory by directly adding noise to x0 and record the difference between the predicted xt−1 and the sampled xt−1 as indicated in the red arrow (c,d). In this way, we are able to reconstruct/edit the image without the strong condition of having an underlying ODE. (e,f) illustrate inversion and editing process for masked generative modeling (MGM) as in Algorithm 1.

In contrast, DDPM inversion-based methods [26] provide user-friendly alternatives by avoiding complex attention map operations. They can be integrated with semantic guidance techniques such as SEGA [6], and this combination is exemplified by LEDITS [53], which enables real image editing through DDPM inversion with semantic control. Other editing methods such as InstructPix2Pix [7] rely on supervised fine-tuning over synthetic pairs and do not involve inversion, while Pix2PixZero [41] focuses on image-to-image translation using DDIM inversion with continuous diffusion.

information. Discrete diffusion model [3, 18, 25] proposed an alternative likelihood-based model for categorical data, and defines the forward process following:

q(xt|xt−1) = Cat(v(xt);π = Qtv(xt−1)). (1)

where Qt is the transition matrix between adjacent states following mask-and-replace strategy. The posterior distribution given x0 has a closed-form solution,

(Q⊤t v(xt)) ⊙ (Qt−1v(x0)) v(xt)⊤Qtv(x0)

. (2)

q (xt−1|xt,x0) =

### 3. Methods

where Qt = Qt ···Q1 is the cumulative transition matrix. The details of Qt and Qt are given in the supplementary materials. The inference process is as below:

##### 3.1. Preliminaries

Masked generative modeling. Masked generative modeling is widely used in representation learning for both natural language processing and computer vision. It works by masking parts of the input and training the model to reconstruct the missing data. In models like BERT [13] and RoBERTa [31], masked tokens ([MASK]) are predicted based on the surrounding context, excelling in text completion and embedding representation learning. For image generation, Paella [43] adapts this approach for text-conditional image generation by renoising tokens instead of masking. The inference process in masked generative models typically involves iterative renoise/remask and repredict steps.

K

πθ(xt,t) = pθ (xt−1|xt) =

q (xt−1|xt,x˜0)pθ (˜x0|xt),

x ˜0=1

(3) with pθ(˜x0|xt) is parameterized by a neural network. We gradually denoise from xT to x0 using 3. For numerical stability, the implementation uses log space instead of probability space. Masked generative models can be viewed as a special case of multinomial diffusion models with an additional absorbing state (or the [MASK] state). Its training objective can be viewed as a reweighted ELBO [5].

Multinomial Diffusion. Denoting x0 ∈ {1,...,K}D as a data point of dimension D. We use v(x(ti)) to denote the one hot column vector representation of the i-th entry of xt. To simplify notation, in the following we drop index i and any function that operates on vector xt is populated along its dimension. Diffusion model defines a markov chain q(x1:T|x0) = ΠTt=1q(xt|xt−1) that gradually adds noise to the data x0 for T times so that xT contains little to no

##### 3.2. Discrete Inversion for Controllable Editing

Non ODE-based inversion. ODE-based generative models, such as DDIM and flow matching, define an ODE trajectory. Due to the deterministic nature of ODEs, inversion can be achieved by solving the ODE using the Euler method in forward direction, ensuring reconstruction based on the inherent properties of the ODE. In contrast, another line of

- Algorithm 1 Discrete Inversion for Masked Generative Modeling Inversion:

- 1: y0 ← D(x0,c,t = 0)
- 2: Sample noise token map n
- 3: for t from 1 to τ do
- 4: mt ← GenerateMask(t) ▷ Sampling masks according to inference algorithm
- 5: xt ← x0 ⊙ (1 − mt) + n ⊙ mt
- 6: yˆ0|t ← Dθ(xt,c,t = t)
- 7: zt ← y0 − yˆ0|t ▷ Eq 4
- 8: end for

Editing/Sampling:

- 9: for t from τ to 1 do
- 10: yˆ0|t ← Dθ(xt,c′,t = t)
- 11: g ∼ Gumbel(0,I)
- 12: y˜0 ← yˆ0|t + λ1 · zt + λ2 · g
- 13: x˜0 ← arg maxy˜0
- 14: xt−1 ← x˜0 ⊙ (1 − mt−1) + n ⊙ mt−1
- 15: end for
- 16: Return x0.

- Algorithm 2 Discrete Inversion for Multinomial Diffusion Inversion:

- 1: for t from 1 to τ do
- 2: xt ∼ q(xt|x0) ▷ Independent q-sample using Eq 5
- 3: yt ← log(onehot(xt))
- 4: end for
- 5: for t from τ to 1 do
- 6: yˆt−1 ← log(πθ(xt,c,t)) ▷ Log posterior using Eq 3
- 7: zt ← yt−1 − yˆt−1 ▷ Eq 6
- 8: end for

Editing/Sampling:

- 9: for t from τ to 1 do
- 10: xˆ0 ← pθ(x0|xt = arg maxyt)
- 11: g ∼ Gumbel(0,I)
- 12: yt−1 ← log(q(xt−1|xt,xˆ0;c′)) + λ1 · zt + λ2 · g
- 13: end for
- 14: Return x0 = arg maxy0.

research focuses on SDE-based models, such as CycleDiffusion [59] and DDPM Inversion [26]. Broadly speaking, these approaches ensure reconstruction by recording the noises or residuals that are required to reproduce the stochastic trajectory. CycleDiffusion records the Gaussian noise zt during sampling from posterior p(xt−1|xt,x0 = x0) and injects information of the input signal by feeding the true x0. DDPM Inversion, on the other hand, incorporates information into zt by fitting the reverse process into an artificial stochastic trajectory obtained by independent q-sample. For both

CycleDiffusion and DDPM Inversion, the key idea is to utilize the Gaussian reparameterization trick, x = µ + σz ⇔ x ∼ N(x;µ,σ2), and keeping track of the “noise” that could have generated the sample from mean. For discrete diffusion models, we utilize the Gumbel-Max trick [27, 34], x = arg max(log(π) + g) ⇔ x ∼ Cat(x;π). Figure 2 provides an intuition of the proposed method.

Inverting masked generative models. In masked generative modeling, the stochastic trajectory xt is constructed according to the specific inference algorithm of the model in use. For example, in Paella [43], the masking is inclusive, meaning that as the time step t increases, the set of masked tokens grows. In contrast, the Unleashing Transformer [5] employs random masking at each step, where masks are generated independently using the q-sample function. Without loss of generality, we define a denoiser function Dθ (parameterized by θ). This denoiser outputs the logits of the predicted unmasked data given the noisy tokens xt. Unlike DDPM or multinomial diffusion, where xt−1 is not sampled from a posterior given xt, the inference of masked modeling takes a different approach. In masked modeling, xt is obtained from sampled xˆ0|t by re-noising. Since the categorical sampling occurs when drawing from the denoiser’s predicted logits, we accordingly define a corresponding latent sequence:

###### yˆ0|t =log(pθ(x0|xt)) = Dθ(xt,t)

zt :=y0 − yˆ0|t. (4) With our proposed latent space, accurate reconstruction is guaranteed. However, for editing tasks, this level of precision may not be ideal if the latent variable zt dominates the generation process. The detailed algorithm is given in Algorithm 1.

To provide more flexibility, we introduce the hyperparameters τ, λ1, and λ2, which allow for finer control over the editing process. Specifically, τ represents the starting (and largest) timestep at which the editing process begins, while λ1 controls the amount of information injected from the original input, and λ2 governs the introduction of random noise (Algorithm 1 line 12).

Noise injection. We discuss three strategies as follows:

Linear. This is a natural form inspired by the Gumbel-Max trick: thinking of λ1·z as a correction term, then log(π)+λ1· z is the corrected logit and λ2 is the inverse of temperature of the logit to control the sharpness of the resulting categorical distribution, as

arg max(log(π) + λ1 · z + λ2 · g)

1 λ2

(log(π) + λ1 · z) + g), λ2 > 0.

=arg max(

λ1 then controls how much correction we would like to introduce in the original logit.

Variance preserving. From another perspective, z is the artificial “Gumbel” noise that could have been sampled to

realize the target tokens. Then, if we treat z as Gumbel noise and want to perturb it with random Gumbel noise, addition does not result in a Gumbel distribution. One way is to approximate this sum with another Gumbel distribution. If G1 ∼ Gumbel(µ1,β1), G2 ∼ Gumbel(µ2,β2) and G = λ1G1 + λ2G2, then the moment matching Gumbel approximation for G is

Gumbel(µG,βG), with

βG = λ21β12 + λ22β22, µG = λ1µ1 + λ2µ2 + γ(λ1β1 + λ2β2 − βG),

where γ ≈ 0.58 is the Euler-Mascheroni constant. We consider the variance preserving form:

y˜ = log(π) + λ1 · z + λ2 · g, λ1 + λ2 = 1. Max. The third way is inspired by the property of Gumbel distribution [57], that if G1, G2 are iid random variables following Gumbel(µ,β) then max{G1,G2} − β log 2 follows the same distribution. We also consider the max function for noise injection:

y˜ = log(π) + max{λ1 · z,λ2 · g}.

We empirically find that linear strategy gives best results. The emperical studies can be find in Supplementary Materials Figure 8.

Inverting multinomial diffusion is more straightforward given its inference is similar to DDPM. We start by sampling a stochastic trajectory, {xt}, a sequence of independent q-sample’s from q(xt|x0) (we populate the following sampling operation along the dimension of xt),

xt = arg max(log(q(xt|x0)) + g), with (5) q(xt|x0) = Cat(xt;π = Qtv(x0)) and g ∼ Gumbel(0,I).

Note that here we use the Gumbel max trick [27], which is equivalent to sampling from categorical distribution q(xt|x0). Note that below the latent zt ∈ RD×K.

yt−1 =log(onehot(xt−1)), and yˆt−1 =log(πθ(xt,t)),

zt :=yt−1 − yˆt−1 (6)

In this reverse process, the latent space {xT,zT,zt−1,...,z1} together with the fixed discrete diffusion model πθ also uniquely define the same stochastic trajectory x0,x1,...,xT. The detailed algorithm is given in Algorithm 2.

Analysis. We provide a quantitative analysis of mutual information in diffusion models using a simple, closed-form DDPM example. While not directly analyzing the discrete case, this study offers insight into how noise and latent injection affect information flow, motivating our scheduling strategy for λ decay. Full details are provided in the supplementary materials.

### 4. Experiments

In this section, we demonstrate the effectiveness of our proposed inversion methods on both image and language diffusion models. Our experiments show that the methods can preserve identity in both vision and language tasks while successfully making the intended changes. The implementation details are in Supplementary Materials Section D.

##### 4.1. Image diffusion model

For the image diffusion model, we mainly investigate the use of absorbing state discrete model [3] including a masked generative model, Paella, and a multinomial diffusion model, VQ-Diffusion. We show the inversion reconstruction and image editing performance in both categories with DICE.

Dataset. The Prompt-based Image Editing Benchmark (PIEBench) by [28] is a recently introduced dataset designed to evaluate text-to-image (T2I) editing methods. The dataset assesses language-guided image editing in 9 different scenarios with 700 images. The benchmark’s detailed annotations and variety of editing tasks were instrumental in thoroughly assessing our method’s capabilities, ensuring a fair and consistent comparison with existing approaches.

Table 1. Inversion Reconstruction performance. The metric is calculated between the original and inverted images. Due to the encoding and decoding steps in the VQ-VAE/GAN process, some inaccuracies are introduced by the quantization. † The PSNR is Inf due to the reconstruction of our method yielding the same VQ-VAE/GAN latents. Base model is Paella [43].

Method PSNR ↑ LPIPS×103 ↓ MSE×104 ↓ SSIM×102 ↑

Inpainting 10.50 565.11 1002.09 30.13 Ours 30.91 39.81 11.07 90.22 Ours† Inf 0.07 0.01 99.99

###### 4.1.1 Inversion Reconstruction

In this section, we evaluate the accuracy of inversion without editing. This is achieved by first inverting the image and then using the recorded latent code to reconstruct the original image.

Evaluation Metrics. Here, we evaluate the image similarity by PSNR, LPIPS, MSE and SSIM of the original and the generated image under the same prompt with DICE.

Quantitative Analysis. The reconstruction performance of our method, as shown in Table 1, far surpasses the baseline Inpainting + Paella model across all metrics. In the case of masked inpainting, all image tokens are replaced with randomly sampled tokens, meaning the model lacks any prior information about the original image. As a result, the reconstructed image differs significantly from the one being inverted, leading to lower similarity scores. In contrast, our method demonstrates near-perfect reconstruction, as indicated by the metrics, and notably produces an identical image without the errors typically introduced by the

Input Image Paella VQ-Diffusion

Input Image Paella VQ-Diffusion

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

(a) two origami birds sitting on a branch (b) A cat dog sitting on a wooden chair

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

(c) a cat tiger sitting next to a mirror (d) white plate with fruits pizza on it

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

(e) drawing of tulip lion on the coffee (f) meat balls sushi on white plate

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

(g) a plate with steak salmon on it (h) white tiger cat on brown ground

- Figure 3. Visualization of editing results. Editing results for our method using Paella and VQ-Diffusion are presented, along with their corresponding prompts. The results demonstrate that our method can effectively modify the input image according to the target prompt while preserving the image structure. Editing with masked generative model (Paella [43]) is more stable and easier than with multinomial diffusion models (VQ-Diffusion [18]). VQ-VAE/GAN quantization process, as seen in the results marked with (†). This highlights the superior accuracy and consistency of our approach in generating high-fidelity reconstructions. Visual results can be viewed in Figure 9 of the Supplementary Materials.
- 4.1.2 Editing Performance

[54] to measure the structural similarity between the original and generated images. To evaluate how well the background is preserved outside the annotated editing mask, we use Peak Signal-to-Noise Ratio (PSNR), Learned Perceptual Image Patch Similarity (LPIPS) [62], Mean Squared Error (MSE), and Structural Similarity Index Measure (SSIM) [56]. We also assess the consistency between the edit prompt and the generated image using CLIP [42] Similarity Score [58], which is calculated over the whole image and specifically within the regions defined by the editing mask.

In this section, we discuss the editing performance of our proposed method. Since there is no discrete diffusion inversion exists, we compare our method with masked generation as indicated in the original paper. In addition to that, we also demonstrate the metric from continuous counterparts.

Results. In Table 2, we demonstrate the quantitative result of DICE using Paella and VQ-Diffusion compared to continuous diffusion model and also inpainting. Notably, our approach with the Paella model achieves the lowest structure distance 11.34, outperforming all other methods, including the continuous diffusion models. Additionally, while the DDPM Inversion with Stable Diffusion v1.4 shows the

Evaluation Metrics. To demonstrate the effectiveness of our proposed inversion method, we employ eight metrics covering three key aspects: structure distance, background preservation, and edit prompt-image consistency, as outlined in [28]. We utilize the structure distance metric proposed by

highest CLIP similarity scores for both whole and edited regions, our method maintains competitive CLIP similarity with Paella. Given the significant reduction in structure distance, our method offers a superior balance between structural preservation and semantic alignment in edits. Furthermore, when combined with VQ-Diffusion, our method continues to show strong performance. The results in Table

- 3 clearly demonstrate the superior background preservation capabilities of our method compared to DDIM+SD1.4. All four metrics underscore the structural consistency of our approach in preserving the unedited regions of the image. These results show the effectiveness of our method in maintaining background integrity during editing and provide evidence that information about the original image is instilled into the latent space of DICE.

In Figure 3, we show the editing results for both Paella and VQ-Diffusion using DICE. Both models successfully modify real images according to the target prompts. In all cases, our results exhibit both high fidelity to the input image and adherence to the target prompt. Additional visualization results can be viewed in Figure 9 and 10 in Supplementary Materials.

##### 4.2. Language Diffusion Model

In this section, we evaluate DICE on RoBERTa [31] and LLaDA [39], a text discrete diffusion model, to generate sentences with opposing sentiments while preserving structural similarities. We begin with two prompts, one with a positive sentiment and another with a negative sentiment. Each prompt contains two sentences: the first sentence indicates the sentiment type and sets the contextual background, and the second sentence is the target for inversion and generation. Initially, we invert the second sentence of the negative sentiment prompt using the entire prompt as context, which produces a noised token representation of that sentence. Next, we condition the model on the positive sentiment by concatenating the first sentence of the positive sentiment prompt with the noised token of the inverted negative sentence. This setup guides the model to generate a new second sentence that mirrors the structure of the original negative sentence but expresses a positive sentiment instead. Through this process, we assess the model’s capability to invert and generate text that aligns with a specified sentiment while retaining the original sentence’s structural elements.

Inversion Process. In our experiment, we focus on inverting the second sentence, indicated as red in the dataset, while keeping the first sentence intact (black), as it usually contains essential context. During the reverse process, we aim to reconstruct/edit the second sentence by recovering it from the noised tokens acquired in the inversion phase.

Dataset Generation. In order to evaluate the editing performance, we designed and proposed a new dataset for Sentiment Editing. The objective is to edit the sentiment of the

sentence while preserving the structure of the sentence and also sticking to the theme of the sentence. Here, we demonstrate two sets of sentences in our dataset. Please refer to supplementary materials for the details.

###### 4.2.1 Inversion Reconstruction

Similar to the image generation section, we first demonstrate the inversion and reconstruction capabilities of the proposed methods. This process involves inverting the sentences, followed by using the same prompt to generate the reconstructed version of the second sentence.

Evaluation Metric. For reconstruction, we use Hit Rate, which is defined as the proportion of cases where each method generates an identical sentence to the original. In addition, we compute the Semantic Textual Similarity (STS) score by measuring the cosine similarity between the sentence embeddings, using the model proposed by [44].

Quantitative Analysis. Table 5 compares DICE with Masked Generation using RoBERTa across two metrics: Accuracy and Semantic Textual Similarity. Our method significantly surpasses Masked Generation in both metrics, demonstrating that our zt latent space effectively captures the information of the sentence being inverted and facilitates its subsequent reconstruction.

###### 4.2.2 Sentence Editing

In this section, we evaluate the editing performance of the proposed inversion method on RoBERTa and LLaDA.

Evaluation Metric. We evaluate the sentence sentiment editing task based on two criteria: (1) structural preservation, which assesses whether the sentence structure is retained, and (2) sentiment correctness, which evaluates whether the sentiment of the edited sentence aligns with the sentiment of the original prompt. Both the structural preservation rate and sentiment correctness rate are calculated using ChatGPT4 [1] as a classifier. Qualitative samples are given in Table 7. Results. Table 6 presents a comparative analysis of two text editing methods that both employ RoBERTa and LLaDA, focusing on the effectiveness in terms of Structure Preservation and Sentiment Correctness. Our method significantly outperforms masked generation in both metrics. This difference highlights the superior capability of our inversion method to encode the original structure of the text in the latent space and the flexibility to adjust its sentiment more accurately. In Supplementary Materials, we demonstrate both the initial prompt and the edited result. Our approach retains the sentence structure of the negative prompt while modifying its sentiment to a more positive one.

- Table 2. Quantitative results on image editing performance. Comparison of our proposed method with the masked inpainting with Paella, as well as continuous diffusion model (Stable Diffusion v1.4) using DDIM inversion. “P2P” refers to Prompt-to-Prompt [23], and “Prompt” denotes editing performed solely through forward edit prompts. Entries marked with an asterisk (∗) are cited from [28]. †: For VQ-Diffusion, the images are down-sampled to 256 × 256. Please note that due to differences in base models and editing algorithms, the metrics across methods are not directly comparable. However, our method significantly outperforms both inpainting and strong baselines (e.g., Null-Text Inversion + SD1.4) in terms of structural preservation. As expected, inpainting achieves a high CLIP score since it directly generates image patches based on the target prompt.

Method Structure CLIP Similarity Inversion+Model Editing Distance×103 ↓ Whole ↑ Edited ↑

Continuous

DDIM+SD1.4 P2P 69.43∗ 25.01∗ 22.44∗ Null-Text + SD1.4 P2P 13.44∗ 24.75∗ 21.86∗ Negative-Prompt + SD1.4 P2P 16.17∗ 24.61∗ 21.87∗ DDPM-Inversion + SD1.4 Prompt 22.12 26.22 23.02

Discrete

Inpainting + Paella Prompt 91.10 25.36 23.42 Ours + Paella Prompt 11.34 23.79 21.23 Ours + VQ-Diffusion† Prompt 12.70 23.85 21.02

- Table 3. Background Preservation. Quantitative comparison of background preservation between our proposed method and DDIM+SD 1.4, achieved by masking the edited region and calculating image similarity with the unedited masked image. The inpainting is served as upper bound since only the masked region are edited and background are not modified.

Method Background Preservation Inversion+Model Editing PSNR ↑ LPIPS×103 ↓ MSE×104 ↓ SSIM×102 ↑ DDIM+SD1.4 P2P 17.87 208.80 219.88 71.14 Ours+Paella Prompt 27.29 52.90 43.76 89.79

- Table 4. Editing results of our method with RoBERTa. The sentence in red is the one being inverted, and the blue sentence represents the editing result.

Negative Prompt Our Edited Results

|Negative: Regarding the lecture. It was dull and confusing.|Positive: Regarding the lecture.<br><br>It was clear and surprising.<br><br>|
|---|---|
|Negative: Despite the initial problems. The project ended in failure.<br><br>|Positive: Despite the initial problems. New project still in progress.|
|Negative: Regarding the new app. It’s complicated and not useful.<br><br>|Positive: Regarding the new app. It’s On and It’s Epic.|

- 5. Conclusion

- Table 5. Text Inversion Reconstruction Performance. Evaluation of the text reconstruction performance by Masked Generation and DICE method using RoBERTa as the language model.

Editing Method Accuracy×102 ↑

Textual Similarity×102 ↑

Masked Generation 0.0 6.57 Ours 99.74 99.90

- Table 6. Text Editing Performance. Evaluation of the text editing performance between Masked Generation and DICE using ChatGPT as a classifier.

Structure Preservation×102 ↑

Sentiment Correctness×102 ↑

Editing Method

Masked Generation + RoBERTa 29.80 12.94 Masked Generation + LLaDA 22.88 21.18 Ours + RoBERTa 94.76 72.51 Ours + LLaDA 94.12 72.29

diffusion and masked generative models. By leveraging recorded noise sequences and masking patterns during the reverse diffusion process, DICE enables accurate reconstruction and flexible editing of discrete data without the need for predefined masks or cross-attention manipulation. Our experiments across multiple models and modalities demonstrate the effectiveness of DICE in preserving data fidelity while enhancing editing capabilities. Furthermore, we demonstrate the potential of DICE for converting RoBERTa, a model traditionally focused on data understanding, into a generative model for text generation and editing. We believe that DICE enhances the capabilities of discrete generative models, offering new opportunities for fine-grained content manipulation in discrete spaces.

In this paper, we introduced DICE, an inversion algorithm for discrete diffusion models, including multinomial

### References

- [1] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 7

- [2] M. Arriola, A. Gokaslan, J. T. Chiu, Z. Yang, Z. Qi, J. Han, S. S. Sahoo, and V. Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. arXiv preprint arXiv:2503.09573, 2025. 2
- [3] J. Austin, D. D. Johnson, J. Ho, D. Tarlow, and R. Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in Neural Information Processing Systems, 34:17981–17993, 2021. 1, 2, 3, 5
- [4] O. Avrahami, D. Lischinski, and O. Fried. Blended diffusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18208–18218, 2022. 1
- [5] S. Bond-Taylor, P. Hessey, H. Sasaki, T. P. Breckon, and C. G. Willcocks. Unleashing transformers: Parallel token prediction with discrete absorbing diffusion for fast highresolution image generation from vector-quantized codes. In European Conference on Computer Vision, pages 170–188. Springer, 2022. 3, 4
- [6] M. Brack, F. Friedrich, D. Hintersdorf, L. Struppek, P. Schramowski, and K. Kersting. Sega: Instructing text-toimage models using semantic guidance. Advances in Neural Information Processing Systems, 36:25365–25389, 2023. 3
- [7] T. Brooks, A. Holynski, and A. A. Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023. 3
- [8] H. Chang, H. Zhang, J. Barber, A. Maschinot, J. Lezama, L. Jiang, M.-H. Yang, K. Murphy, W. T. Freeman, M. Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. arXiv preprint arXiv:2301.00704,

2023. 1, 2

- [9] H. Chang, H. Zhang, L. Jiang, C. Liu, and W. T. Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11315–11325, 2022. 1, 2
- [10] R. T. Chen, Y. Rubanova, J. Bettencourt, and D. K. Duvenaud. Neural ordinary differential equations. Advances in neural information processing systems, 31, 2018. 2
- [11] H. Chung, J. Kim, M. T. Mccann, M. L. Klasky, and J. C. Ye. Diffusion posterior sampling for general noisy inverse problems. arXiv preprint arXiv:2209.14687, 2022. 1
- [12] Q. Dao, H. Phung, B. Nguyen, and A. Tran. Flow matching in latent space. arXiv preprint arXiv:2307.08698, 2023. 1

- [13] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. 2, 3
- [14] P. Esser, R. Rombach, A. Blattmann, and B. Ommer. Imagebart: Bidirectional context with multinomial diffusion for autoregressive image synthesis. Advances in neural information processing systems, 34:3518–3532, 2021. 2
- [15] P. Esser, R. Rombach, and B. Ommer. Taming transformers for high-resolution image synthesis. In CVPR, pages 12873– 12883, 2021. 1
- [16] D. Garibi, O. Patashnik, A. Voynov, H. Averbuch-Elor, and D. Cohen-Or. Renoise: Real image inversion through iterative noising. In European Conference on Computer Vision, pages 395–413. Springer, 2024. 2
- [17] I. Gat, T. Remez, N. Shaul, F. Kreuk, R. T. Chen, G. Synnaeve, Y. Adi, and Y. Lipman. Discrete flow matching. arXiv preprint arXiv:2407.15595, 2024. 2
- [18] S. Gu, D. Chen, J. Bao, F. Wen, B. Zhang, D. Chen, L. Yuan, and B. Guo. Vector quantized diffusion model for text-toimage synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10696– 10706, 2022. 1, 2, 3, 6
- [19] L. Han, Y. Li, H. Zhang, P. Milanfar, D. Metaxas, and F. Yang. Svdiff: Compact parameter space for diffusion fine-tuning. arXiv preprint arXiv:2303.11305, 2023. 1
- [20] L. Han, J. Ren, H.-Y. Lee, F. Barbieri, K. Olszewski, S. Minaee, D. Metaxas, and S. Tulyakov. Show me what and tell me how: Video synthesis via multimodal conditioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3615–3625, 2022. 2
- [21] L. Han, S. Wen, Q. Chen, Z. Zhang, K. Song, M. Ren, R. Gao, A. Stathopoulos, X. He, Y. Chen, et al. Proxedit: Improving tuning-free real image editing with proximal guidance. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 4291–4301, 2024. 1,

- 2

[22] X. He, C. Tan, L. Han, B. Liu, L. Axel, K. Li, and D. N. Metaxas. Dmcvr: Morphology-guided diffusion model for

- 3d cardiac volume reconstruction. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 132–142. Springer, 2023. 1

- [23] A. Hertz, R. Mokady, J. Tenenbaum, K. Aberman, Y. Pritch, and D. Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 2, 8
- [24] A. Hertz, A. Voynov, S. Fruchter, and D. Cohen-Or. Style aligned image generation via shared attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4775–4785, 2024. 2

- [25] E. Hoogeboom, D. Nielsen, P. Jaini, P. Forré, and M. Welling. Argmax flows and multinomial diffusion: Learning categorical distributions. Advances in Neural Information Processing Systems, 34:12454–12465, 2021. 1, 2, 3
- [26] I. Huberman-Spiegelglas, V. Kulikov, and T. Michaeli. An edit friendly ddpm noise space: Inversion and manipulations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12469–12478, 2024. 1, 2, 3, 4
- [27] E. Jang, S. Gu, and B. Poole. Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144,

2016. 4, 5

- [28] X. Ju, A. Zeng, Y. Bian, S. Liu, and Q. Xu. Direct inversion: Boosting diffusion-based editing with 3 lines of code. arXiv preprint arXiv:2310.01506, 2023. 5, 6, 8
- [29] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 1, 2
- [30] X. Liu, C. Gong, and Q. Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 1, 2
- [31] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019. 2, 3, 7
- [32] A. Lou, C. Meng, and S. Ermon. Discrete diffusion language modeling by estimating the ratios of the data distribution. arXiv preprint arXiv:2310.16834, 2023. 2
- [33] S. Lu, Y. Liu, and A. W.-K. Kong. Tf-icon: Diffusion-based training-free cross-domain image composition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2294–2305, 2023. 2
- [34] C. J. Maddison, D. Tarlow, and T. Minka. A* sampling. Advances in neural information processing systems, 27, 2014. 4
- [35] C. Meng, K. Choi, J. Song, and S. Ermon. Concrete score matching: Generalized score matching for discrete data. Advances in Neural Information Processing Systems, 35:34532– 34545, 2022. 2
- [36] C. Meng, Y. Song, J. Song, J. Wu, J.-Y. Zhu, and S. Ermon. Sdedit: Image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021. 1, 3, 6
- [37] D. Miyake, A. Iohara, Y. Saito, and T. Tanaka. Negativeprompt inversion: Fast image inversion for editing with textguided diffusion models. arXiv preprint arXiv:2305.16807,

2023. 2

- [38] R. Mokady, A. Hertz, K. Aberman, Y. Pritch, and D. CohenOr. Null-text inversion for editing real images using guided diffusion models. arXiv preprint arXiv:2211.09794, 2022. 1, 2
- [39] S. Nie, F. Zhu, Z. You, X. Zhang, J. Ou, J. Hu, J. Zhou, Y. Lin, J.-R. Wen, and C. Li. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025. 2, 7
- [40] Z. Pan, R. Gherardi, X. Xie, and S. Huang. Effective real image editing with accelerated iterative diffusion inversion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15912–15921, 2023. 2
- [41] G. Parmar, K. Kumar Singh, R. Zhang, Y. Li, J. Lu, and J.-Y. Zhu. Zero-shot image-to-image translation. In ACM SIGGRAPH 2023 conference proceedings, pages 1–11, 2023. 3
- [42] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 6
- [43] D. Rampas, P. Pernias, and M. Aubreville. A novel sampling scheme for text-and image-conditional image synthesis in quantized latent spaces. arXiv preprint arXiv:2211.07292,

2022. 2, 3, 4, 5, 6

- [44] N. Reimers. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084,

2019. 7

- [45] S. Sahoo, M. Arriola, Y. Schiff, A. Gokaslan, E. Marroquin, J. Chiu, A. Rush, and V. Kuleshov. Simple and effective masked diffusion language models. Advances in Neural Information Processing Systems, 37:130136–130184, 2024. 2
- [46] D. Samuel, B. Meiri, H. Maron, Y. Tewel, N. Darshan, S. Avidan, G. Chechik, and R. Ben-Ari. Lightning-fast image inversion and editing for text-to-image diffusion models. arXiv preprint arXiv:2312.12540, 2023. 2
- [47] J. Shi, K. Han, Z. Wang, A. Doucet, and M. Titsias. Simplified and generalized masked diffusion for discrete data. Advances in neural information processing systems, 37:103131–103167,

2024. 2

- [48] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, pages 2256–2265. PMLR, 2015. 2
- [49] J. Song, C. Meng, and S. Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021. 1, 2
- [50] Y. Song and S. Ermon. Generative modeling by estimating gradients of the data distribution. Advances in Neural Information Processing Systems, 32, 2019. 2

- [51] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 1, 2
- [52] A. Stathopoulos, L. Han, and D. Metaxas. Score-guided diffusion for 3d human recovery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 906–915, 2024. 1
- [53] L. Tsaban and A. Passos. Ledits: Real image editing with ddpm inversion and semantic guidance. arXiv preprint arXiv:2307.00522, 2023. 3
- [54] N. Tumanyan, M. Geyer, S. Bagon, and T. Dekel. Plugand-play diffusion features for text-driven image-to-image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1921–1930,

2023. 2, 6

- [55] A. Wang and K. Cho. Bert has a mouth, and it must speak: Bert as a markov random field language model. arXiv preprint arXiv:1902.04094, 2019. 2
- [56] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600– 612, 2004. 6
- [57] Wikipedia contributors. Gumbel distribution — Wikipedia, The Free Encyclopedia. https://en.wikipedia. org/wiki/Gumbel_distribution, 2024. [Online; accessed 8-October-2024]. 5, 2
- [58] C. Wu, L. Huang, Q. Zhang, B. Li, L. Ji, F. Yang, G. Sapiro, and N. Duan. Godiva: Generating open-domain videos from natural descriptions. arXiv preprint arXiv:2104.14806, 2021. 6
- [59] C. H. Wu and F. De la Torre. Unifying diffusion models’ latent space, with applications to cyclediffusion and guidance. arXiv preprint arXiv:2210.05559, 2022. 1, 2, 4
- [60] J. Ye, Z. Xie, L. Zheng, J. Gao, Z. Wu, X. Jiang, Z. Li, and L. Kong. Dream 7b, 2025. 2
- [61] L. Zhang, A. Rao, and M. Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 3, 6
- [62] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 6
- [63] Z. Zhang, L. Han, A. Ghosh, D. N. Metaxas, and J. Ren. Sine: Single image editing with text-to-image diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6027–6037, 2023. 1

[64] L. Zheng, J. Yuan, L. Yu, and L. Kong. A reparameterized discrete diffusion model for text generation. arXiv preprint arXiv:2302.05737, 2023. 2

### A.

Figure 4: This paper was accepted to CVPR 2025 but later desk-rejected post camera-ready, due to a withdrawal from ICLR made 14 days before reviewer assignment.

### B. Details on Multinomial Diffusion Models

Definition of Qt with mask-and-replace strategy. Following mask-and-replace strategy as:





αt + βt βt βt ··· 0 βt αt + βt βt ··· 0 βt βt αt + βt ··· 0

, (7)

Qt =

 

 

... . γt γt γt ··· 1

. . .

given αt ∈ [0,1],βt = (1 − αt − γt)/K and γt the probability of a token to be replaced with a [MASK] token. Cumulative transition matrix. The cumulative transition matrix Qt and q (xt|x0) can be computed via closed form:

Qtv (x0) = α¯tv (x0) + γ ¯t − β¯t v(K + 1) + β¯t1, (8)

where α¯t = ti=1 αi,γ¯t = 1 − ti=1 (1 − γi), and β¯t = (1 − α¯t − γ¯t)/(K + 1) can be calculated and stored in advance.

### C. Analysis on Mutual Information

Here we provide an analysis to quantify the amount of information encoded in latent. Since the inversion involves model forward function call which is difficult to analyze. We describe in the following a simple yet prototypical example of DDPM, where the posterior mean can be computed in closed-form thus allows us to compute the mutual information.

Remark C.1. Given a simple Gaussian DDPM with x0 ∼ N(0,I), latents {zt} are obtained with DDPM inversion [26], then the mutual information between zt and x0:

D 2

I(zt;x0) =

βt2αt−1 + 1 − αt−1 + αt(1 − αt) 1 − αt−1 + αt(1 − αt)

log(

).

(9)

The mutual information between zt and x0 is illustrated in Supplementary Materials. We observe that the amount of information encoded from x0 into zt decreases as t increases, motivating us to explore different scheduling strategies for λ’s.

Proof. We assumed that x0 satisfies standard Gaussian distribution N(0,ID). Since

xt = √αtxt−1 + √1 − αtϵt

where both xt−1 and ϵt are independent standard Gaussian random variables, xt is also standard Gaussian, and in each dimension

Cov(xt,xt−1) = √αt, which leads to

µˆt(xt) = E(xt−1|xt) = √αtxt. Therefore,

zt = x′t−1 − µˆt(xt)

= ( αt−1x0 + 1 − αt−1ϵ) −

√αt(√αtx0 + √1 − αtϵ′)

= βt · αt−1x0 + 1 − αt−1ϵ + αt(1 − αt)ϵ′. Let

E = 1 − αt−1ϵ + αt(1 − αt)ϵ′

which is a Gaussian error term independent to x0 with mean 0 and variance 1−αt−1+αt(1−αt). Thus we can calculate the mutual information

I(zt;x0) = H(zt) − H(zt|x0)

= H(zt) − H(E)

D 2

log(2πe(βt2αt−1 + 1 − αt−1 + αt(1 − αt)) −

=

D 2

log(2πe(1 − αt−1 + αt(1 − αt))

βt2αt−1 + 1 − αt−1 + αt(1 − αt) 1 − αt−1 + αt(1 − αt)

D 2

=

log(

).

| |
|---|

We also provide the relationship between the mutual information of zt,z0 and the timestep t in Figure 5.

### D. Implementation Details

For all reconstruction task, we employ a τ = 1.0 and λ1 = 1.0,λ2 = 0.0 with 32 sampling steps and 26 renoising steps.

The hyper-parameters for Paella editing experiment is CFG= 10.0, λ1 = 0.7, λ2 = 0.3 and τ = 0.9. The hyperparameters for VQ-Diffusion in editing is CFG= 5.0, λ1 = 0.2, λ2 = 0.8.

For sentiment editing task with RoBERTa, we utilize two sets of hyperparameter: τ = 0.7, λ1 = 0.2, λ2 = 0.8 and τ = 0.7, λ1 = 0.25, λ2 = 0.75.

All models are implemented in PyTorch 2.0 and inferenced on a single NVIDIA A100 40GB.

[Figure 70]

Figure 4. CVPR Situation

Plot of I(zt; x0) vs. Time Step t

1e 5

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

2.0

1.5

(;)Izx0t

1.0

0.5

0.0

0 200 400 600 800 1000

Time step t

Figure 5. Mutual information between zt and x0. Computed with a simple DDPM setting by assuming x0 ∼ N(0, I).

### E. Ablation Studies

##### E.1. Noise Injection Function

Addition. In the main text we have adopted the addition function as noise injection function,

y˜ = log(π) + λ1 · z + λ2 · g.

This is a natural form inspired by the Gumbel-Max trick: thinking of λ1 · z as a correction term, then log(π) + λ1 · z is the corrected logit and λ2 is the inverse of temperature of the logit to control the sharpness of the resulting categorical distribution, as

arg max(log(π) + λ1 · z + λ2 · g)

1 λ2

(log(π) + λ1 · z) + g), λ2 > 0.

=arg max(

λ1 then controls how much correction we would like to introduce in the original logit.

Variance preserving. From another perspective, z is the artificial “Gumbel” noise that could have been sampled to realize the target tokens. Then, if we treat z as Gumbel

noise and want to perturb it with random Gumbel noise, addition does not result in a Gumbel distribution. One way is to approximate this sum with another Gumbel distribution. If G1 ∼ Gumbel(µ1,β1), G2 ∼ Gumbel(µ2,β2) and G = λ1G1 + λ2G2, then the moment matching Gumbel approximation for G is

Gumbel(µG,βG), with

βG = λ21β12 + λ22β22, µG = λ1µ1 + λ2µ2 + γ(λ1β1 + λ2β2 − βG),

where γ ≈ 0.5772 is the Euler-Mascheroni constant. We consider the variance preserving form:

y˜ = log(π) + λ1 · z + λ2 · g, λ1 + λ2 = 1.

Max. The third way is inspired by the property of Gumbel distribution [57], that if G1, G2 are iid random variables following Gumbel(µ,β) then max{G1,G2} − β log 2 follows the same distribution. We also consider the max function for noise injection:

y˜ = log(π) + max{λ1 · z,λ2 · g}.

##### E.2. Hyperparameter Search

In this section, we analyze the impact of varying hyperparameters λ1,λ2,τ, and CFG scale on the quality of image generation and adherence to textual descriptions, quantified through Structure Distance and CLIP similarity. The hyperparameters play specific roles: λ controls the amount of noise introduced in each reverse step, τ governs the percentage of tokens replaced with random tokens during inversion, and Classifier-Free Guidance (CFG) scales the influence of the text prompt during image synthesis. To limit the search space and simplify the ablation, we choose λ1 = λ and λ2 = 1 − λ and vary the value of λ. Evaluation metrics are given in Figure 6.

Effect of λ1 and λ2: With a fixed CFG of 10.0, the graphs indicate that increasing λ results in a rise in Structure Distance, suggesting a decline in structural integrity of the images.

This increase in noise appears to allow for greater exploration of the generative space at the expense of some loss in image clarity.

Effect of τ: Higher τ values, particularly at 0.9, show a notable rise in Structure Distance as CLIP similarity increases. This implies that more token replacement can lead to images that align better with the text prompts but may suffer in maintaining structural fidelity, likely due to xT contains less information of the original image while λ injects additional noise during editing phase.

Effect of CFG Scale: Varying CFG at a fixed λ of 0.7 and τ of 0.9 reveals that higher CFG values substantially improve Structure Distance, but to an extent (CFG of 10). Beyond this point, further increases in CFG do not yield significant improvements in structural quality, indicating a diminishing return on higher guidance levels. This plateau suggests that while increasing CFG helps in aligning the generated images more closely with the text prompts initially, the benefits in structural integrity and clarity become less visible as CFG values exceed a certain threshold. This finding underscores the need for a balanced approach in setting CFG, where too much guidance may not necessarily lead to better outcomes in terms of image quality and fidelity to the textual description.

Effect of noise injection function: We also conducted evaluations using a variance-preserving noise injection function by setting λ1 =

√

λ and λ2 = √1 − λ. The results of these experiments are presented in Figure 7. As for the max function, we performed a manual inspection of the visual examples generated with this function. The quality of these examples was noticeably inferior, we therefore omit the corresponding evaluation curves from our analysis.

In conclusion, this ablation study demonstrates that increasing λ and τ can enhance adherence to text prompts through broader explorations in generative spaces, yet this benefit is offset by a decrease in the structural quality of the images. On the other hand, raising CFG values enhances the structural integrity of images to a certain threshold, after which the improvements plateau, indicating a ceiling to the effectiveness of higher CFG settings. This analysis offers empirical guidance for selecting hyperparameters, balancing the trade-offs between text alignment and image quality to optimize image synthesis outcomes.

### F. Additional Results on Image Editing

Reconstruction result with Paella. In Figure 9 we demonstrates the inversion reconstruction result with Paella using our proposed method.

Image editing with diversity. As shown in Figure 11, our method enables diverse image editing results through stochastic variation. The first three rows demonstrate the impact of varying both the inversion masks and the injected Gumbel noise, while the last two rows focus on variations

CFG=10.0

values 0.1 0.3 0.5 0.7

0.07

0.06

StructureDstance()

0.05

0.04

0.03

0.02

| |
|---|

0.01

| |
|---|

| |
|---|

0.00

| |
|---|

22.75 23.00 23.25 23.50 23.75 24.00 24.25 24.50

CLIP Similarity( )

=0.7

0.030

CFG

6.0 7.5 8.0

0.025

StructureDstance()

10.0 16.0

0.020

0.015

0.010

| |
|---|

0.005

| |
|---|

0.000

22.6 22.8 23.0 23.2 23.4 23.6 23.8 24.0 24.2

CLIP Similarity( )

CFG=10.0

values 0.3 0.5 0.7 0.9

0.07

| |
|---|

0.06

StructureDstance()

0.05

0.04

| |
|---|

0.03

0.02

| |
|---|

0.01

0.00

22.75 23.00 23.25 23.50 23.75 24.00 24.25 24.50

CLIP Similarity( )

=0.9

CFG

| |
|---|

0.07

6.0 7.5 8.0

| | |
|---|---|
| | |

| |
|---|

StructureDstance()

0.06

10.0 16.0

0.05

| |
|---|

0.04

0.03

0.02

0.01

23.2 23.4 23.6 23.8 24.0 24.2 24.4 24.6

CLIP Similarity( )

- Figure 6. The effect of hyperparameters λ1, λ2, τ, CFG on the Structure Distance (↓) and CLIP similarity (↑) with addition function as noise inject function. In our implementation, to limit the search space, we choose λ1 = λ and λ2 = 1−λ for simplicity.

22.5 23.0 23.5 24.0 24.5

CLIP Similarity( )

0.00

0.02

0.04

0.06

0.08

StructureDstance()

| |
|---|

| |
|---|

CFG=10.0

values 0.1 0.3 0.5 0.7 0.9 1.0

22.5 23.0 23.5 24.0 24.5

CLIP Similarity( )

0.00

0.02

0.04

0.06

0.08

StructureDstance()

| |
|---|

| |
|---|

| |
|---|

| |
|---|
| |

| |
|---|

| |
|---|

CFG=10.0

values 0.3 0.5 0.7 0.9

22.4 22.6 22.8 23.0 23.2 23.4 23.6 23.8 24.0

CLIP Similarity( )

0.000

0.005

0.010

0.015

0.020

0.025

0.030

StructureDstance()

| |
|---|

| |
|---|
| |

| |
|---|

=0.7

CFG

5.0 7.5

10.0 16.0

23.0 23.2 23.4 23.6 23.8 24.0 24.2

CLIP Similarity( )

0.00

0.02

0.04

0.06

0.08

0.10

StructureDstance()

| |
|---|

| | | |
|---|---|---|
| | | |

| |
|---|

| |
|---|

=0.9

CFG

5.0 7.5 10.0 16.0

| | |
|---|---|
| | |

- Figure 7. The effect of hyperparameters λ1, λ2 with variance

√

λ and λ2 = √1 − λ.

preserving scheme. We set λ1 =

produced by changing only the inversion masks.

Additional baselines. We compare with SDEdit [36] and ControlNet [61]1. Results are shown in Figure 12 and Table 8.

Noise injection functions. We compare various noise injection functions, including taking the maximum of Gumbel noise and the recorded noise, as well as the variancepreserving noise injection function.

Mask schedule functions. In Figure 13, we present four types of mask scheduling functions: (a, c) concave up and (b, d) concave down. Our results indicate that concave up mask scheduling functions perform better than their concave down counterparts. Quantitative results are shown in Table 9.

1We use the ControlNet-InPaint model based on Stable Diffusion v1.5: https : / / github . com / mikonvergence / ControlNetInpaint

Lambda = 0.1: Linear, Exponential, and Uniform

Type Linear

0.07

Exponential Uniform

0.06

StructureDstance()

| |
|---|

0.05

0.04

0.03

0.02

| | |
|---|---|

0.01

20.0 20.5 21.0 21.5 22.0 22.5 CLIP Similarity( )

Lambda = 0.5: Linear, Exponential, and Uniform

Type Linear

0.05

Exponential Uniform

0.04

StructureDstance()

| |
|---|

0.03

0.02

0.01

| | |
|---|---|
| | |

0.00

19.5 20.0 20.5 21.0 21.5 22.0 CLIP Similarity( )

Lambda = 0.3: Linear, Exponential, and Uniform

Type Linear

0.06

Exponential

StructureDstance()

Uniform

0.05

0.04

0.03

0.02

| | |
|---|---|

0.01

0.00

19.0 19.5 20.0 20.5 21.0 21.5 22.0 22.5 CLIP Similarity( )

Lambda = 0.7: Linear, Exponential, and Uniform

0.030

Type Linear

0.025

Exponential Uniform

StructureDstance()

| |
|---|

0.020

0.015

0.010

0.005

| | |
|---|---|
| | |

| | | |
|---|---|---|

| | |
|---|---|

0.000

19.5 20.0 20.5 21.0 21.5 22.0 CLIP Similarity( )

Figure 8. The effect of different λ schedule on the Structure Distance (↓) and CLIP similarity (↑). In our implementation, to limit the search space, we choose λ1 = λ and λ2 = 1 − λ for simplicity.

ChatGPT

- 1. Thanks to her efforts. The event was a huge success.

Despite her efforts. The event was a complete disaster.

- 2. ...

Comparison between inclusive and random masks. To understand the impact of randomness in the masking schedule, we illustrate masks that are inclusive compared to totally random. Inclusive mask is mask schedule that are increasingly growing, which is used in Paella, compared to randomly sampled masks.

### G. Details on Text Editing Experiments

Dataset generation. To generate the dataset, we utilize ChatGPT-4o with the following prompt:

User

Generate 200 pairs of sentences that contains the same meaning, but one with positive sentiment and one with negative sentiment. For both positive sentiment and negative sentiment, you need to write two sentences with the first part being a hint of the sentiment and the second part being the actual content. The first part for both sentences should be same. write in the format like: hint. positive. hint. negative. Make sure that there are two lines for each pairs. Also, the hint should provide enough context and both positive and negative sentiment should be related to the hint. Do not repeat the hint, also make sure that there is only two sentences in each of the line, one is the hint and the other is about the sentiment.

The sentences is then added with a prefix to indicates the sentiment of the context. Here we demonstrates a subset of our generated dataset:

###### Dataset

- 1. Positive Sentiment: Thanks to her efforts. The event was a huge success. Negative Sentiment: Despite her efforts. The event was a complete disaster.
- 2. Positive Sentiment: This book is definitely interesting. I can’t put it down; it’s full of surprises. Negative Sentiment: This book is definitely interesting. I can’t wait to finish it; it’s so predictable.
- 3. Positive Sentiment: Regarding the lecture. It was insightful and engaging. Negative Sentiment: Regarding the lecture. It was dull and confusing.
- 4. Positive Sentiment: Despite the initial problems. The project was a success. Negative Sentiment: Despite the initial problems. The project ended in failure.
- 5. Positive Sentiment: Regarding the new app. It’s userfriendly and very helpful. Negative Sentiment: Regarding the new app. It’s complicated and not useful.
- 6. Positive Sentiment: Reflecting on my environmental initiatives. Implementing changes has reduced my carbon footprint. Negative Sentiment: Reflecting on my environmental initiatives. It’s challenging to maintain, and progress is slow.
- 7. Positive Sentiment: The business proposal was wellreceived. The ideas were innovative, and the presentation was convincing. Negative Sentiment: The business proposal was rejected. The ideas were impractical, and the presentation was unconvincing.
- 8. Positive Sentiment: The training program was highly effective. It boosted skills and confidence, and everyone left motivated. Negative Sentiment: The training program was ineffective. It didn’t teach much, and most people left feeling unmotivated.
- 9. ...

User

Given three sentences, confirm that the second sentence is roughly the same sentence structure as the first sentence, then confirm that the second sentence has positive sentiment. Output only two numbers with each number indicating whether the corresponding criteria is satisfied. Use 1 for satisfied and 0 for not satisfied. The sentences are given below: The event was a complete disaster. This event was a fantastic comedy game.

ChatGPT 1 1

Comparison between masked inpainting and DICE. In Figure 10 we demonstrates the reconstruction and editing results with our DICE and Masked Inpainting.

### I. Limitations

While Discrete Inversion shows promise, we empirically find that editing with multinomial diffusion models may not work as robustly as with masked generative models. Furthermore, it may appear less effective in style transfer tasks, such as transforming an image of a cat into a silver cat statue. Interesting future directions include: (1) developing a more theoretical analysis of mutual information and convergence for continuous and discrete inversion algorithms, (2) extending Discrete Inversion to score distillation sampling, and (3) exploring the integration of Semantic Guidance within discrete settings.

### H. Additional Results on Sentiment Editing

Evaluation. Below, we demonstrate the prompt used for evaluating the editing results:

Negative Prompt Our Edited Results

|Negative Sentiment: This book is definitely interesting. I can’t wait to finish it; it’s so predictable. It’s cramped and lacks proper facilities.|Positive Sentiment: This book is definitely interesting. I can’t wait to see it; it sounds so beautiful.<br><br>It’s spacious and has great facilities.|
|---|---|
|Negative Sentiment: Despite her efforts. The event was a complete disaster.<br><br>|Positive Sentiment: Thanks to her efforts. This event was a fantastic comedy game.|
|Negative Sentiment: Regarding the lecture. It was dull and confusing.<br><br>|Positive Sentiment: Regarding the lecture. It was clear and surprising.|
|Negative Sentiment: Despite the initial problems. The project ended in failure.|Positive Sentiment: Despite the initial problems.<br><br>New project still in progress.<br><br>|
|Negative Sentiment: Regarding the new app. It’s complicated and not useful.<br><br>|Positive Sentiment: Regarding the new app. It’s On and It’s Epic.|
|Negative Sentiment: Reflecting on my environmental initiatives. It’s challenging to maintain, and progress is slow.<br><br>|Positive Sentiment: Reflecting on my environmental initiatives.<br><br>It’s easy to understand, and progress is undeniable.|

Table 7. Editing results of our method with RoBERTa. The sentences in black are the prompts used for inversion and editing in their respective column. The sentence in red is the one being inverted, and the blue sentence represents the editing result.

Method Structure CLIP Similarity Inversion+Model Editing Distance×103 ↓ Whole ↑ Edited ↑

- ControlNet-InPaint (scale=0.5) + SD1.5 Prompt 65.12 25.50 22.85
- ControlNet-InPaint (scale=1.0) + SD1.5 Prompt 60.87 24.35 21.40 SDEdit (t0 = 0.4) + Paella Prompt 30.52 23.14 20.72 SDEdit (t0 = 0.6) + Paella Prompt 38.62 23.22 20.86 Inpainting + Paella Prompt 91.10 25.36 23.42 Ours + Paella Prompt 11.34 23.79 21.23

Table 8. Additional baselines. We compare with SDEdit [36] and ControlNet [61].

Structure CLIP Similarity Mask Schedule Distance×103 ↓ Whole ↑ Edited ↑

- (a): 1 − cos(t · π/2) 7.54 23.48 20.96
- (b): cos((t − 1) · π/2) 25.39 23.56 21.24
- (c): 1 −

√1 − t 5.11 22.99 20.50

- (d): √t 26.35 23.59 21.36

- (e): t 11.34 23.79 21.23

√1 − t, (d): √t.

Table 9. Comparison with different masking schedule. (a): 1 − cos(t · π/2), (b): cos((t − 1) · π/2),(c): 1 −

Input Image Reconstruction Editing Input Image Reconstruction Editing

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

(a) two origami birds sitting on a branch (b) A cat dog sitting on a wooden chair

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

(c) a cat tiger sitting next to a mirror (d) white plate with fruits pizza on it

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

(e) drawing of tulip lion on the coffee (f) meat balls sushi on white plate

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

(g) a plate with steak salmon on it (h) white tiger cat on brown ground

###### Figure 9. Reconstruction and editing result with DICE+Paella.

Reconstruction Editing Input Image Masked Inpainting Discrete Inversion

Masked Inpainting Discrete Inversion

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

- (a) a round square cake with orange frosting on a wooden plate

- (b)
- (c)
- (d)

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

a cat dog sitting on a wooden chair

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

a colorful red bird standing on a branch

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

meat balls sushi on white plate

- Figure 10. Reconstruction and editing result with DICE and masked inpainting. Notice that for reconstruction, we use the red prompt, but for editing we use the green prompt.

Input Image

Different Samples from DICE

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

VaryinginMaskandGumbelNoiseVaryinginMask

A sketch sculpture of a cat

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

two origami birds sitting on a branch

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

white tiger cat on brown ground

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

a cat dog sitting on a wooden chair

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

a dog lion is laying down on a white background

- Figure 11. Image Editing with Diversity. Due to the stochastic nature of our method, we can generate diverse outputs. The first three rows illustrate variations in both inversion masks and injected Gumbel noise (λ1 = 0.7, λ2 = 0.3). The last two rows demonstrate variations using only inversion masks (λ1 = 1, λ2 = 0).

SDEdit 𝑡 = 0.4 ControlNet-InPaint (scale = 0.5)

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

SDEdit 𝑡 = 0.6 ControlNet-InPaint (scale = 1.0)

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

- Figure 12. Editing results with SDEdit and ControlNet. For SDEdit we show examples of t0 = 0.4, 0.6. For ControlNet we show examples of conditioning scale of 0.5 and 1.

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

(a): 1 − cos 𝑡 (b): cos 𝑡 − 1

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

(c): 1 − 1 − 𝑡 (d): 𝑡

[Figure 161]

[Figure 162]

#### (e): 𝑡

√1 − t, (d): √t.

Figure 13. Comparison with different masking schedule. (a): 1 − cos(t · π/2), (b): cos((t − 1) · π/2),(c): 1 −

[Figure 163]

[Figure 164]

- (a) Max
- (b) Variance Preserving

[Figure 165]

[Figure 166]

Inclusive Mask

[Figure 167]

[Figure 168]

- Figure 14. Comparison with different noise injection functions.

- (b) Sum

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

- (c) Max

[Figure 173]

[Figure 174]

(a) Original

- Figure 15. Inversion reconstruction comparison with different lambda schedule.

Random Mask

[Figure 175]

[Figure 176]

###### Figure 16. Comparison between inclusive and random masks.

- (a) Different Noise between Inversion and Inference
- (b) Different Noise in Each Sampling Renoise Step

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

(d) Same Noise in Each Renoise Step (Paella default)

- (c) Different Noise in Each Renoise Step for Both Inversion and Sampling

[Figure 183]

[Figure 184]

Figure 17. Comparison with different noise token schedule. Here we show visualization results of using different noise tokens in inversion and inference, using different noise tokens in each renoising step of the sampling process, using different noise tokens in each renoising step of both inversion and sampling process, and ours by using the same tokens in both inversion and inference.

