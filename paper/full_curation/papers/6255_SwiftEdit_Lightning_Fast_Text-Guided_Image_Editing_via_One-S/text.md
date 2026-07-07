arXiv:2412.04301v4[cs.CV]2Jun2025

# SwiftEdit: Lightning Fast Text-Guided Image Editing via One-Step Diffusion

Trong-Tung Nguyen, Quang Nguyen, Khoi Nguyen, Anh Tran, Cuong Pham† Qualcomm AI Research+

{tunnguy,quanghon,khoinguy,anhtra,pcuong}@qti.qualcomm.com

⚡SwiftEdit : Make Your Edit in 0.23 seconds ⚡

Edited Image Source Image Edited Image Source Image Edited Image

###### Source Image

|[Figure 1]|
|---|

|[Figure 2]|
|---|

|[Figure 3]|[Figure 4]|
|---|---|

|[Figure 5]|
|---|

|[Figure 6]|
|---|

basket of apples → basket of puppies in forest → at sea mouth closed → mouth opened

|[Figure 7]|[Figure 8]|
|---|---|

|[Figure 9]|
|---|

|[Figure 10]|[Figure 11]|
|---|---|

|[Figure 12]|
|---|

empty street → crowded street couples on beach → couples on beach dog holding flower → dog holding flower

hold hands

- Figure 1. SwiftEdit empowers instant, localized image editing using only text prompts, freeing users from the need to define masks. In just 0.23 seconds on a single A100 GPU, it unlocks a world of creative possibilities demonstrated across diverse editing scenarios.

## Abstract

Recent advances in text-guided image editing enable users to perform image edits through simple text inputs, leveraging the extensive priors of multi-step diffusion-based textto-image models. However, these methods often fall short of the speed demands required for real-world and on-device applications due to the costly multi-step inversion and sampling process involved. In response to this, we introduce SwiftEdit, a simple yet highly efficient editing tool that achieve instant text-guided image editing (in 0.23s). The advancement of SwiftEdit lies in its two novel contributions: a one-step inversion framework that enables one-step image reconstruction via inversion and a mask-guided editing technique with our proposed attention rescaling mechanism to perform localized image editing. Extensive experiments are provided to demonstrate the effectiveness and efficiency of SwiftEdit. In particular, SwiftEdit enables instant textguided image editing, which is extremely faster than pre-

+Qualcomm Vietnam Company Limited † also affiliated with Posts & Telecom. Inst. of Tech., Vietnam Contact email: nguyentrongtung11101999@gmail.com

vious multi-step methods (at least 50× times faster) while maintain a competitive performance in editing results. Our project is at https://swift-edit.github.io/.

## 1. Introduction

Recent text-to-image diffusion models [5, 24, 26, 27] have achieved remarkable results in generating high-quality images semantically aligned with given text prompts. To generate realistic images, most of them rely on multi-step sampling techniques, which reverse the diffusion process starting from random noise to realistic image. To overcome this time-consuming sampling process, some works focus on reducing the number of sampling steps to a few (4-8 steps) [29] or even one step [5, 20, 39, 40] via distillation techniques while not compromising results. These approaches not only accelerate image generation but also enable faster inference for downstream tasks, such as image editing.

For text-guided image editing, recent approaches [11, 13, 19] use an inversion process to determine the initial noise for a source image, allowing for (1) source image reconstruction and (2) content modification aligned with guided text while preserving other details. Starting from this inverted noise, additional techniques, such as attention ma-

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

PSNRScore

SwiftEdit

25

ReNoise-SDXL-Turbo

TurboEdit ICD-SD15 Null-text Inversion

20

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

MasaCtrl

- 23
- 24
- 25

CLIPScore

Pix2Pix-Zero

Plug-and-Play

DDIM

100 101 102 Execution Time (s) in log scale

- Figure 2. Comparing our one-step SwiftEdit with few-step and multi-step diffusion editing methods in terms of background preservation (PSNR), editing semantics (CLIP score), and runtime. Our method delivers lightning-fast text-guided editing while achieving competitive results.

nipulation and hijacking [3, 21, 35], are applied at each denoising step to inject edits gradually while preserving key background elements. This typical approach, however, is resource-intensive, requiring two lengthy multi-step processes: inversion and editing. To address this, recent works [6, 8, 33] use few-step diffusion models, like SD-Turbo [30], to reduce the sampling steps required for inversion and editing, incorporating additional guidance for disentangled editing via text prompts. However, these methods still struggle to achieve sufficiently fast text-guided image editing for on-device applications while maintaining performance competitive with multistep approaches.

In this work, we take a different approach by building on a one-step text-to-image model for image editing. We introduce SwiftEdit – the first one-step text-guided image editing tool – which achieves at least 50× faster execution than previous multi-step methods while maintaining competitive editing quality. Notably, both the inversion and editing in SwiftEdit are accomplished in a single step.

Inverting one-step diffusion models is challenging, as existing techniques like DDIM Inversion [31] and Null-text Inversion [19] are unsuitable for our one-step real-time editing goal. To achieve this, we design a novel one-step inversion framework inspired by encoder-based GAN Inversion methods [36, 37, 41]. Unlike GAN inversion, which requires domain-specific networks and retraining, our inversion framework generalizes to any input images. For this, we leverage SwiftBrushv2 [5], a recent one-step textto-image model known for speed, diversity, and quality, using it as both the one-step image generator and backbone for our one-step inversion network. We then train it with weights initialized from SwiftBrushv2 to handle any source inputs through a two-stage training strategy, combining supervision from both synthetic and real data.

Following the one-step inversion, we introduce an efficient mask-based editing technique. Our method can either accept an input editing mask or infer it directly from the

trained inversion network and guidance prompts. The mask is then used in our novel attention-rescaling technique to blend and control the edit strength while preserving background elements, enabling high-quality editing results.

To the best of our knowledge, our work is the first to explore diffusion-based one-step inversion using a one-step text-to-image generation model to instantly perform textguided image editing (in 0.23 seconds). While being significantly fast compared to other multi-step and few-step editing methods, our approach achieves a competitive editing result as shown in Fig. 2. In summary, our main contribution includes:

- • We propose a novel one-step inversion framework trained with a two-stage strategy. Once trained, our framework can invert any input images into an editable latent in a single step without further retraining or finetuning.
- • We show that our well-trained inversion framework can produce an editing mask guided by source and target text prompts within a single batchified forward pass.
- • We propose a novel attention-rescaling technique for mask-based editing, offering flexible control over editing strength while preserving key background information.

## 2. Related Work

### 2.1. Text-to-image Diffusion Models

Diffusion-based text-to-image models [24, 26, 27] typically rely on computationally expensive iterative denoising to generate realistic images from Gaussian noise. Recent advances [16, 18, 28, 32] alleviate this by distilling the knowledge from multi-step teacher models into a few-step student network. Notable works [5, 15, 16, 20, 32, 39, 40] show that this knowledge can be distilled even into a onestep student model. Specifically, Instaflow [15] uses rectified flow to train a one-step network, while DMD [40] applies distribution-matching objectives for knowledge transfer. DMDv2 [39] removes costly regression losses, enabling efficient few-step sampling. SwiftBrush [20] utilizes an image-free distillation method with text-to-3D generation objectives, and SwiftBrushv2 [5] integrates posttraining model merging and clamped CLIP loss, surpassing its teacher model to achieve state-of-the-art one-step textto-image performance. These one-step models provide rich prior information about text-image alignment and are extremely fast, making them ideal for our one-step text-based image editing approach.

### 2.2. Text-based Image Editing

Several approaches leverage the strong prior of imagetext relationships in text-to-image models to perform textguided multi-step image editing via an inverse-to-edit approach. First, they invert the source image into “informative” noise. Methods like DDIM Inversion [31] use

linear approximations of noise prediction, while Nulltext Inversion [19] enhances reconstruction quality through costly per-step optimization. Direct Inversion [11] bypasses these issues by disentangling source and target generation branches. Second, editing methods such as [3, 10, 21, 22, 35] manipulate attention maps to embed edits while preserving background content. However, their multi-step diffusion process remains too slow for practical applications.

To address this issue, several works [6, 8, 33] enable few-step image editing using fast generation models [29]. ICD [33] achieves accurate inversion in 3-4 steps with a consistency distillation framework, followed by text-guided editing. ReNoise [8] refines the sampling process with an iterative renoising technique at each step. TurboEdit [6] uses a shifted noise schedule to align inverted noise with the expected schedule in fast models like SDXL Turbo [29]. Though these methods reduce inference time, they fall short of instant text-based image editing needed for fast applications. Our one-step inversion and one-step localized editing approach dramatically boosts time efficiency while surpassing few-step methods in editing performance.

### 2.3. GAN Inversion

GAN inversion [2, 4, 14, 17, 23, 36, 41] maps a source image into the latent space of a pre-trained GAN, allowing the generator to recreate the image, which is valuable for tasks like image editing. Effective editing requires a latent space that can both reconstruct the image and support realistic edits through variations in the latent code. Approaches fall into three groups: encoder-based [23, 41, 42], optimizationbased [4, 14, 17], and hybrid [1, 2, 41]. Encoder-based methods learn a mapping from the image to the latent code for fast reconstruction. Optimization-based methods refine a code by iteratively optimizing it, while hybrid methods combine both, using an encoder’s output as initialization for further optimization. Inspired by encoder-based speed, we develop a one-step inversion network, but instead of GAN, we leverage a one-step text-to-image diffusion model. This allows us to achieve text-based image editing across diverse domains rather than being restricted to specific domain as in GAN-based methods.

## 3. Preliminaries

Multi-step diffusion model. Text-to-image diffusion model ϵϕ attempts to generate image xˆ given the target prompt embedding cy (extracted from the CLIP text encoder of a given text prompt y) through a T iterative denoising steps, starting from Gaussian noise, zT = ϵ ∼ N(0,I):

zt − σtϵϕ(zt,t,cy) αt

+ δtϵt, ϵt ∼ N(0,I), (1)

zt−1 =

where t is the timestep, and σt,αt,δt are three coefficients. The final latent z = z0 is then input to a VAE decoder D to

produce the image xˆ = D(z).

One-step diffusion model. The traditional diffusion model’s sampling process requires multiple steps, making it time-consuming. To address this, one-step text-toimage diffusion models like InstaFlow [15], DMD [40], DMD2 [39], SwiftBrush [20], and SwiftBrushv2 [5] have been developed, reducing the sampling steps to a single step. Specifically, one-step text-to-image diffusion model G aims to transform a noise input ϵ ∼ N(0,1), given a text prompt embedding cy, directly into an image latent zˆ, without iterative denoising steps, or zˆ = G(ϵ,cy). SwiftBrushv2 (SBv2) stands out in one-step image generation by quickly producing high-quality, diverse outputs, forming the basis of our approach. Building on its predecessor, SBv2 integrates key improvements: it uses SD-Turbo initialization for enhanced output quality, a clamped CLIP loss to strengthen visual-text alignment, and model fusion with post-enhancement techniques, all contributing to superior performance and visual fidelity.

Score Distillation Sampling (SDS) [25] is a popular objective function that utilizes the strong prior learned by 2D diffusion models to optimize a target data point z by calculating its gradient as follows:

∂z ∂θ

∇θLSDS ≜ Et,ϵ w(t)(ϵϕ(zt,t,cy) − ϵ)

, (2)

where z = g(θ) is rendered by a differentiable image generator g parameterized by θ, zt denotes a perturbed version of z with a random amount of noise ϵ, and w(t) is a scaling function corresponding to the timestep t. The objective of SDS loss is to provide an updated direction that would move z to a high-density region of the data manifold using the score function of the diffusion model ϵϕ(zt,t,cy). Notably, this gradient omits the Jacobian term of the diffusion backbone, removing the expensive computation when backpropagating through the entire diffusion model U-Net.

Image-Prompt via Decoupled Cross-Attention. IPAdapter [38] introduces an image-prompt condition x that can be seamlessly integrated into a pre-trained text-toimage generation model. It achieves this through a decoupled cross-attention mechanism, which separates the conditioning effects of text and image features. This is done by adding an extra cross-attention layer to each cross-attention layer in the original U-Net. Given image features cx (extracted from x by a CLIP image encoder), text features cy (from text prompt y using a CLIP text encoder), and query features Zl from the previous U-Net layer l − 1, the output hl of the decoupled cross-attention is computed as:

hl = Attn(Ql,Ky,Vy) + sx Attn(Ql,Kx,Vx), (3) where Attn(.) denotes the attention operation. Scaling factors sx is used to control the influence of cx on the gener-

###### Stage 1: Training with Synthetic Images

Reconstructed Image ̂x

[Figure 13]

IP-Adapter

Synthetic Image x

|[Figure 14]|
|---|

[Figure 15]

|IP-SBv2 GIP(.)<br><br>[Figure 16]|
|---|

|SBv2 G(.)| |
|---|---|
| | |

Inversion Net Fθ(.)

[Figure 17]

[Figure 18]

|[Figure 19]|
|---|

|[Figure 20]|
|---|

|[Figure 21]|
|---|

[Figure 22]

[Figure 23]

Inverted Noise

Text Encoder ‘ (‘“An orange cat sitting on a fence” )

###### VAE

VAE

[Figure 24]

[Figure 25]

Stage 2: Training with Real Images

Inverted Noise

|[Figure 26]|
|---|

[Figure 27]

|IP-SBv2 GIP(.)<br><br>[Figure 28]|
|---|

|[Figure 29]|
|---|

Inversion Net Fθ(.)

[Figure 30]

[Figure 31]

|[Figure 32]|
|---|

|[Figure 33]|
|---|

Reconstructed Image ̂x

Real Image x

[Figure 34]

IP-Adapter

- Figure 3. Proposed two-stage training for our one-step inversion framework. In stage 1, we warms up our inversion network on synthetic data generated by SwiftBrushv2. At stage 2, we shift our focus to real images, continue to train our inversion network to enable instantly image inversion for any input images without additional fine-tuning or retraining.

ated output. Ql = WQZl is the query matrix projected by the weight matrix WQ. The key and value matrices for text

features cy are Ky = WyKcy and Vy = WyV cy, respectively, while the projected key and value matrices for image

features cx are Kx = WxKcx and Vx = WxV cx. Notably, only the two weight matrices WxK and WxV are trainable, while the remaining weights remain frozen to preserve the original behavior of the pretrained diffusion model.

- 4. Proposed Method

main gap poses a challenge, as the original noise ϵ is unavailable, preventing us from computing regression objective and potentially causing ˆϵ to deviate from the desired distribution. In the following section, we discuss our inversion network and a two-stage training strategy designed to overcome these challenges effectively.

Our Inversion Network Fθ follows the architecture of the one-step diffusion model G and is initialized with G’s weights. However, we found this approach suboptimal: the inverted noise ˆϵ predicted by Fθ attempts to perfectly reconstruct the input image, leading to overfitting on specific patterns from the input. This tailoring makes the noise overly dependent on input features, which limits editing flexibility.

Our goal is to enable instant image editing with the onestep text-to-image model, SBv2. In Sec. 4.1, we develop a one-step inversion network that predicts inverted noise to reconstruct a source image when passed through SBv2. We introduce a two-stage training strategy for this inversion network, enabling single-step reconstruction of any input images without further retraining. An overview is shown in Fig. 3. During inference, as described in Sec. 4.2, we use self-guided editing mask to locate edited regions. Our attention-rescaling technique then utilizes the mask to achieve disentangled editing and control the editing strength while preserving the background.

To overcome this, we introduce an auxiliary, imageconditioned branch – similar to IP-Adapter [38] – within the one-step generator G, named GIP. This branch integrates image features encoded from the input image x along with text prompt y, aiding in reconstruction and reducing the need for Fθ to embed extensive visual details from the input image. This approach effectively alleviates the burden on ˆϵ, enhancing both reconstruction and editing capabilities. We compute the inverted noise ˆϵ along with the reconstructed image latent zˆ as follows:

### 4.1. Inversion Network and Two-stage Training

ˆϵ = Fθ(z,cy), zˆ = GIP(ˆϵ,cy,cx). (4)

Given an input image that may be synthetic (generated by a model like SBv2) or real, our first objective is to inverse and reconstruct it using SBv2 model. To achieve this, we develop a one-step inversion network Fθ to transform the image latent z into an inverted noise ˆϵ = Fθ(z,cy), and then feed back to SBv2 to compute the reconstructed latent zˆ = G(ˆϵ,cy) = G(Fθ(z,cy),cy). For synthetic images, training Fθ is straightforward, with pairs (ϵ,z), where ϵ is the noise used to generate z, allowing direct regression of ˆϵ to ϵ, and aligning the inverted noise with SBv2’s input noise distribution. However, for real images, the do-

Stage 1: Training with synthetic images. As mentioned above, this stage aims to pretrain the inversion network Fθ with synthetic training data sampled from a text-to-image diffusion network G, i.e., SBv2. In Fig. 3, we visualize the flow of stage 1 training in orange color. Pairs of training samples (ϵ,z) are created as follows:

ϵ ∼ N(0,1), z = G(ϵ,cy). (5) We combine the reconstruction loss Lstage1rec and regression

Visualization of Inverted Noise

encourages ˆϵ to capture source image patterns, aiding reconstruction but constraining future editing flexibility (see Fig. 4, column 2). To address this, we introduce a new regularization term Lstage2regu , inspired by Score Distillation Sampling (SDS) as defined in Eq. (2). The SDS gradient steers the optimized latent toward dense regions of the data manifold. Given that the real image latent z = E(x) already lies in a high-density region, we shift the optimization focus to the noise term ϵ, treating our inverted noise as an added noise to z. We then compute the loss gradient as follows:

Source Image

without with

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

ˆϵ = Fθ(z,cy), zt = αtz + σtˆϵ, ∇θLstage2regu ≜ Et,ϵˆ w(t)(ˆϵ − ϵϕ(zt,t,cy))

∂ˆϵ ∂θ

. (8)

- Figure 4. Comparison of inverted noise predicted by our inversion network when trained without and with stage 2 regularization loss.

Our regularization gradient has the opposite sign of Eq. (2) since it optimizes ˆϵ instead of z (derivation details in Appendix). After initializing from stage 1, ϵˆ resembles Gaussian noise N(0,1), making the noisy latent zt compatible with the multi-step teacher’s training data. This allows the teacher to accurately predict ϵϕ(zt,t,cy), and achieve ϵϕ(zt,t,cy) − ϵˆ ≈ 0. Thus, ˆϵ stays the same. Over time, the reconstruction loss nudges Fθ to generate an inverted noise, ϵˆ, tailored for reconstruction, diverging from N(0,1) and creating an unfamiliar zt. The resulting gradient prevents excessive drift from the original distribution, reinforcing stability from stage 1, as shown in third column of Fig. 4. Similar to stage 1, we combine both perceptual losses Lstage2perceptual and regularization loss Lstage2regu where we set λstage2 = 1. During training , we train only the inversion network, keeping the IP-Adapter branch and decoupled cross-attention layers frozen to retain the image prior features learned in stage 1. Flow of training stage 2 are visualized as teal color in Fig. 3.

loss Lstage1regr to train the inversion network Fθ and part of the IP-Adapter branch (including the linear mapping and crossattention layers for image conditions). The regression loss Lstage1regr encourages Fθ(.) to produce an inverted noise ˆϵ that closely follows SBv2’s input noise distribution by regressing ˆϵ to ϵ. This ensures that the inverted noise remains close to the multivariate normal distribution, which is crucial for effective editability as shown in prior work [19]. On the other hand, the reconstruction loss Lstage1rec enforces alignment between the reconstructed latent zˆ and the original source latent z, preserving input image details. In summary, the training objectives are as follows:

Lstage1rec = ||z − zˆ||22, Lstage1regr = ||ϵ − ˆϵ||22, (6) Lstage1 = Lstage1rec + λstage1.Lstage1regr , (7)

where we set λstage1 = 1 during training. After this stage, our inversion framework could reconstruct source input images generated by the SBv2 model. However, it fails to work with real images due to the domain gap which motivates us to continue training with stage 2.

### 4.2. Attention Rescaling for Mask-aware Editing (ARaM)

During inference, given a source image xsource, a source prompt ysource, and an editing prompt yedit, our target is to produce an edited image xedit following the editing prompt without modifying irrelevant background elements. After two-stage training, we obtain a well-trained inversion network Fθ to transform source image latent zsource = E(xsource) to inverted noise ˆϵ. Intuitively, we can use the one-step image generator, GIP(.), to regenerate the image but with an edit prompt embedding cedity as guided prompt instead. The edited image latent is computed via zedit = GIP(ˆϵ,cedity ,cx). As discussed in Sec. 4.1, the source image condition cx is crucial for reconstruction, with its influence modulated by sx as shown in Eq. (3). To illustrate this, we vary sx while generating the edited image xedit = D(zedit) in orange block of Fig. 5b. As shown, higher values of sx enforce fidelity to the source image, limiting editing flexi-

- Stage 2: Training with real images. We replace the reconstruction loss from stage 1 with a perceptual loss using the Deep Image Structure and Texture Similarity (DISTS)

metric [7]. This perceptual loss, Lstage2perceptual = DISTS(x,xˆ), compares xˆ = D(zˆ) (where zˆ = GIP(ˆϵ,cy,cx)) with the real input image x. DISTS is trained on real images, capturing perceptual details in structure and texture, making it a more robust visual similarity measure than the pixel-wise reconstruction loss used in stage 1.

Since the original noise ϵ, used to reconstruct z in SBv2, is unavailable at this stage, we cannot directly apply the regression objective from stage 1. Training stage 2 solely with Lstage2perceptual can cause the inverted noise ˆϵ to drift from the ideal noise distribution N(0,I), as the perceptual loss

in an edited image which both follow prompt edit semantics and achieve good background preservation compared to using the same sx. On the other hand, we introduce the additional sy to lessen/strengthen the edit prompt-alignment effect within the editing region M which could be used to control the editing strength as shown in Fig. 5c.

Source Prompt: An orange cat sitting on top of a fence.

[Figure 41]

[Figure 42]

|Normalized Diffrence| |
|---|---|
| | |

#### -

The editing mask M discussed above can either be provided by the user or generated automatically from our inversion network Fθ. To extract self-guided editing mask, we observe that a well-trained Fθ can discern spatial semantic differences in the inverted noise maps when conditioned on varying text prompts. As shown in Fig. 5a, we input the source image latent zsource to Fθ with two different text prompts: the source csourcey and the edit cedity . The difference noise map, ˆϵsource − ˆϵedit, is then computed and normalized, yielding the editing mask M, which effectively highlights the editing areas.

Source Image Editing Mask

Edit Prompt: A black cat sitting on top of a fence.

- (a) Self-guided editing mask extraction. Given source and editing prompts, our inversion network predicts two different noise maps, highlighting the editing regions M.

|[Figure 43]|✅ Edit Semantic<br><br>✅ BG|
|---|---|
|Better Preservation|Preservation|

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Better Edit Semantic

With Global Scale With ARaM

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

- (b) Effect of global scale and our edit-aware scale. Comparison of edited results between varying global image condition scale sx with our ARaM.

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

Control editing strength with mask-guided text-alignment scale

[Figure 60]

- (c) Effect of editing strength scale. Visualization of edited results when varying mask-based text-alignment scale sy.

## 5. Experiments

### 5.1. Experimental Setup

Dataset and evaluation metrics. We evaluate our editing performance on PieBench [11], a popular benchmark containing 700 samples across 10 diverse editing types. Each sample includes a source prompt, edit prompt, instruction prompt, source image, and a manually annotated editing mask. Using PieBench’s metrics, we assess both background preservation and editing semantics, aiming for a balance between them for high-quality edits. Background preservation is evaluated with PSNR and MSE scores on unedited regions of the source and edited images. Editing alignment is assessed using CLIP-Whole and CLIP-Edited scores, measuring prompt alignment with the full image and edited region, respectively.

- Figure 5. Illustration of Attention Rescaling for Mask-aware Editing (ARaM). We apply attention rescaling with our selfguided editing mask to achieve local image editing and enable editing strength control.

bility due to tight control by cx. Conversely, lower sx allows more flexible edits but reduces reconstruction quality. Based on this observation, we introduce Attention Rescaling for Mask-aware editing (ARaM) in GIP, guided by the editing mask M. The key idea is to amplify the influence of cx in non-edited regions for better preservation while reducing its effect within edited regions, providing greater editing flexibility. To implement this, we reformulate the computation in Eq. (3) within GIP by removing the global scale sx and introducing region-specific scales as follows:

Implementation details. Our inversion network is based on the architecture of SBv2, initialized with SBv2 weights for stage 1 training. In stage 2, we continue training from stage 1’s pretrained weights. For image encoding, we adopt the IP-Adapter [38] design, using a pretrained CLIP image encoder followed by a small projection network that maps the image embeddings to a sequence of features with length N = 4, matching the text feature dimensions of the diffusion model. Both stages use the Adam optimizer [12] with weight decay of 1e-4, a learning rate of 1e-5, and an exponential moving average (EMA) in every iteration. In stage 1, we train with a batch size of 4 for 100k iterations on synthetic samples generated by SBv2, paired with 40k captions from the JourneyDB dataset [34]. For stage 2, we train with a batch size of 1 and train over 180k iterations using 5k real images and their prompt descriptions from the CommonCanvas dataset [9]. All experiments are conducted on a single NVIDIA A100 40GB GPU.

hl = sy.M.Attn(Ql,Ky,Vy)

(9)

+ sedit.M.Attn(Ql,Kx,Vx)

+ snon-edit.(1 − M).Attn(Ql,Kx,Vx).

This disentangled cross-attention differs slightly from Eq. (3) in three scaling factors: sy, sedit, and snon-edit, apply on different image regions. Two scaling factors sedit, and snon-edit are used to separately control the influence of the image condition cx on the editing and non-editing regions. As shown in violet block of Fig. 5b, this effectively results

Background Preservation CLIP Semantics

Type Method

Runtime↓ PSNR↑ MSE×104↓ Whole ↑ Edited↑ (seconds)

DDIM + P2P 17.87 219.88 25.01 22.44 25.98 NT-Inv + P2P 27.03 35.86 24.75 21.86 134.06

DDIM + MasaCtrl 22.17 86.97 23.96 21.16 23.21 Direct Inversion + MasaCtrl 22.64 81.09 24.38 21.35 29.68

Multi-step (50 steps)

DDIM + P2P-Zero 20.44 144.12 22.80 20.54 35.57 Direct Inversion + P2P-Zero 21.53 127.32 23.31 21.05 35.34

DDIM + PnP 22.28 83.64 25.41 22.55 12.62 Direct Inversion + PnP 22.46 80.45 25.41 22.62 12.79

ReNoise (SDXL Turbo) 20.28 54.08 24.29 21.07 5.11 TurboEdit 22.43 9.48 25.49 21.82 1.32 ICD (SD 1.5) 26.93 3.32 22.42 19.07 1.62

Few-steps (4 steps)

SwiftEdit (Ours) 23.33 6.60 25.16 21.25 0.23 SwiftEdit (Ours with GT masks) 23.31 6.18 25.56 21.91 0.23

One-step

Table 1. Quantitative comparison of SwiftEdit against other editing methods with metrics employed from PieBench [11].

.

> 130s > 12s > 1.3s < 0.3s

Source Image NT + P2P

Pix2PixZero MasaCtrl

Plugand-Play ReNoise TurboEdit

DDIM + P2P

ICD (SD 1.5)

SwiftEdit (Ours)

[Figure 61]

two birds sitting on a branch → two origami birds sitting on branch

[Figure 62]

an orange van with surfboards on top → an orange van with ﬂowers on top

[Figure 63]

a cup of coﬀee with drawing of tulip … → a cup of coﬀee with drawing of lion …

[Figure 64]

a yellow bird with a red break → a crochet bird with a red break

[Figure 65]

white tiger on brown ground → white cat on brown ground

- Figure 6. Comparative edited results. The first column shows the source image, while source and edit prompts are noted under each row.

Comparison Methods. We perform an extensive comparison of SwiftEdit with representative multi-step and recently introduced few-step image editing methods. For multi-step methods, we choose Prompt-to-Prompt (P2P) [10], MasaCtrl [3], Pix2Pix-Zero (P2P-Zero) [22], and Plug-and-Play [35], combined with corresponding inversion methods such as DDIM [31], Null-text Inversion (NT-Inv) [19], and Direct

Inversion [11]. For few-step methods, we select Renoise [8], TurboEdit [6], and ICD [33].

### 5.2. Comparison with Prior Methods

Quantitative Results. In Tab. 1, we present the quantitative results comparing SwiftEdit to various multi-step and few-step image editing methods. Overall, SwiftEdit

Preference Rate User Study

Methods

Edit Semantic

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

SwiftEdit

BG Preserve

NT-Inverse

0 20 40 60 80 100 Percentage

TurboEdit

###### Figure 7. User Study.

###### Method PSNR↑ LPIPS×103↓ MSE×104↓ SSIM×102↑

- w/o stage 1 22.26 111.57 7.03 72.39
- w/o stage 2 17.95 305.23 17.46 55.97 w/o IP-Adapter 18.57 165.78 16.11 63.87 Full Setting (Ours) 24.35 89.69 4.59 76.34

Table 2. Impact of inversion framework design on real image reconstruction.

CLIP Semantics Whole (↑) Edited(↑)

Setting Lregrstage1 Lstageregu 2

- Setting 1 ✗ ✗ 22.91 19.07
- Setting 2 ✗ ✓ 22.98 19.01
- Setting 3 ✓ ✗ 24.19 20.55
- Setting 4 (Full) ✓ ✓ 25.16 21.25 Table 3. Effect of loss on editing semantics score.

demonstrates superior time efficiency due to our one-step inversion and editing process, while maintaining competitive editing performance. Compared to multi-step methods, SwiftEdit shows strong results in background preservation scores, surpassing most approaches. Although it achieves a slightly lower PSNR score than NT-Inv + P2P, it has a better MSE score and is approximately 500 times faster. In terms of CLIP Semantics, we also achieve competitive results in CLIP-Whole (second best) and CLIP-Edited. Compared with few-step methods, SwiftEdit performs as the secondbest in background preservation (with ICD being the best) and second-best in CLIP Semantics (with TurboEdit leading), while maintaining a speed advantage, being at least 5 times faster than these methods. Since SwiftEdit allows for user-defined editing masks, we also report results using the ground-truth editing masks from PieBench [11]. As shown in the last row of Tab. 1, results with the ground-truth masks show slight improvements, indicating that our self-guided editing masks are nearly as accurate as the ground truth.

Qualitative Results. In Fig. 6, we present visual comparisons of editing results generated by SwiftEdit and other methods. As illustrated, SwiftEdit successfully adheres to the given edit prompt while preserving essential background details. This balance demonstrates SwiftEdit’s strength over other multi-step methods, as it produces highquality edits while being significantly faster. When compared to few-step methods, SwiftEdit demonstrates a clear advantage in edit quality. Although ICD [33] scores high on background preservation (as shown in Tab. 1), it often fails to produce edits that align with the prompt. TurboEdit [6], while achieving a higher CLIP score than SwiftEdit,

generates lower-quality results that compromise key background elements, as seen in the first, second, and fifth rows of Fig. 6. This highlights SwiftEdit’s high-quality edits with prompt alignment and background preservation.

User Study. We conducted a user study with 140 participants to evaluate preferences for different editing results. Using 20 random edit prompts from PieBench [11], participants compared images edited by three methods: Null-text Inversion [19], TurboEdit [6], and our SwiftEdit. Participants selected the most appropriate edits based on background preservation and editing semantics. As shown in Fig. 7, SwiftEdit was the preferred choice, with 47.8% favoring it for editing semantics and 40% for background preservation, while also surpassing other methods in speed.

## 6. Ablation Study

Analysis of Inversion Framework Design. We conduct ablation studies to evaluate the impact of our inversion framework and two-stage training on image reconstruction. Our two-stage strategy is essential for the one-step inversion framework’s effectiveness. In Tab. 2, we show that omitting any stages degrades reconstruction quality. The IP-Adapter with decoupled cross-attention is critical; removing it leads to poor reconstruction, as seen in row 3.

Effect of loss on Editing Quality. As noted by [19], an editable noise should follow a normal distribution to ensure flexibility. We conduct ablation studies to assess the impact of our loss functions on noise editability. As shown in Tab. 3, omitting any loss component reduces editability, measured by CLIP Semantics, while using both yields the highest scores. This emphasizes the importance of each loss in maintaining noise distributions that enhance editability.

## 7. Conclusion and Discussion

Conclusion. In this work, we introduce SwiftEdit, a lightning-fast text-guided image editing tool capable of instant edits in 0.23 seconds. Extensive experiments demonstrate SwiftEdit’s ability to deliver high-quality results while significantly surpassing previous methods in speed, enabled by its one-step inversion and editing process. We hope SwiftEdit will facilitate interactive image editing.

Discussion. While SwiftEdit achieves instant-level image editing, challenges remain. Its performance still relies on the quality of the SBv2 generator, thus, biases in the training data can transfer to our inversion network. For future work, we want to improve the method by transitioning from instant-level to real-time editing capabilities. This enhancement would address current limitations and have a significant impact across various fields.

## References

- [1] David Bau, Jun-Yan Zhu, Jonas Wulff, William Peebles, Hendrik Strobelt, Bolei Zhou, and Antonio Torralba. Inverting layers of a large generator. In ICLR workshop, page 4,

2019. 3

- [2] David Bau, Jun-Yan Zhu, Jonas Wulff, William Peebles, Hendrik Strobelt, Bolei Zhou, and Antonio Torralba. Seeing what a gan cannot generate. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4502– 4511, 2019. 3
- [3] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 22560–22570,

2023. 2, 3, 7

- [4] Antonia Creswell and Anil Anthony Bharath. Inverting the generator of a generative adversarial network. IEEE transactions on neural networks and learning systems, 30(7):1967– 1974, 2018. 3
- [5] Trung Dao, Thuan Hoang Nguyen, Thanh Le, Duc Vu, Khoi Nguyen, Cuong Pham, and Anh Tran. Swiftbrush v2: Make your one-step diffusion model better than its teacher. In European Conference on Computer Vision, pages 176–192. Springer, 2025. 1, 2, 3
- [6] Gilad Deutch, Rinon Gal, Daniel Garibi, Or Patashnik, and Daniel Cohen-Or. Turboedit: Text-based image editing using few-step diffusion models. In SIGGRAPH Asia 2024 Conference Papers, New York, NY, USA, 2024. Association for Computing Machinery. 2, 3, 7, 8
- [7] Keyan Ding, Kede Ma, Shiqi Wang, and Eero P. Simoncelli. Image quality assessment: Unifying structure and texture similarity. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(5):2567–2581, 2022. 5
- [8] Daniel Garibi, Or Patashnik, Andrey Voynov, Hadar Averbuch-Elor, and Daniel Cohen-Or. Renoise: Real image inversion through iterative noising. In Computer Vision

– ECCV 2024, pages 395–413, Cham, 2025. Springer Nature Switzerland. 2, 3, 7

- [9] Aaron Gokaslan, A Feder Cooper, Jasmine Collins, Landan Seguin, Austin Jacobson, Mihir Patel, Jonathan Frankle, Cory Stephenson, and Volodymyr Kuleshov. Commoncanvas: An open diffusion model trained with creativecommons images. arXiv preprint arXiv:2310.16825, 2023.

- 6

[10] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-or. Prompt-to-prompt image editing with cross-attention control. In The Eleventh International Conference on Learning Representations, 2023. 3,

- 7

- [11] Xuan Ju, Ailing Zeng, Yuxuan Bian, Shaoteng Liu, and Qiang Xu. Pnp inversion: Boosting diffusion-based editing with 3 lines of code. International Conference on Learning Representations (ICLR), 2024. 1, 3, 6, 7, 8, 13
- [12] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In 3rd International Conference on

- Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015. 6
- [13] Senmao Li, Joost van de Weijer, Taihang Hu, Fahad Shahbaz Khan, Qibin Hou, Yaxing Wang, and Jian Yang. Stylediffusion: Prompt-embedding inversion for text-based editing. arXiv preprint arXiv:2303.15649, 2023. 1
- [14] Zachary C Lipton and Subarna Tripathi. Precise recovery of latent vectors from generative adversarial networks. arXiv preprint arXiv:1702.04782, 2017. 3
- [15] Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, and Qiang Liu. Instaflow: One step is enough for high-quality diffusion-based text-to-image generation. In International Conference on Learning Representations, 2024. 2, 3, 12
- [16] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023. 2
- [17] Fangchang Ma, Ulas Ayaz, and Sertac Karaman. Invertibility of convolutional generative networks from partial measurements. Advances in Neural Information Processing Systems, 31, 2018. 3
- [18] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14297–14306, 2023. 2
- [19] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6038–6047, 2023. 1, 2, 3, 5, 7, 8
- [20] Thuan Hoang Nguyen and Anh Tran. Swiftbrush: One-step text-to-image diffusion model with variational score distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1, 2, 3, 12
- [21] Trong-Tung Nguyen, Duc-Anh Nguyen, Anh Tran, and Cuong Pham. Flexedit: Flexible and controllable diffusion-based object-centric image editing. arXiv preprint arXiv:2403.18605, 2024. 2, 3
- [22] Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot image-toimage translation. New York, NY, USA, 2023. Association for Computing Machinery. 3, 7
- [23] Guim Perarnau, Joost van de Weijer, Bogdan Raducanu, and Jose M. Alvarez.´ Invertible Conditional GANs for image editing. In NIPS Workshop on Adversarial Training, 2016. 3
- [24] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In The Twelfth International Conference on Learning Representations, 2024. 1, 2
- [25] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In The Eleventh International Conference on Learning Representations, 2023. 3

- [26] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, 2022. 1, 2
- [27] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding. In Advances in Neural Information Processing Systems, pages 36479–36494. Curran Associates, Inc., 2022. 1, 2
- [28] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In International Conference on Learning Representations, 2022. 2
- [29] Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast highresolution image synthesis with latent adversarial diffusion distillation. In SIGGRAPH Asia 2024 Conference Papers, New York, NY, USA, 2024. Association for Computing Machinery. 1, 3
- [30] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In European Conference on Computer Vision, pages 87–103. Springer,

2025. 2

- [31] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 2, 7
- [32] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In Proceedings of the 40th International Conference on Machine Learning, pages 32211–32252. PMLR, 2023. 2
- [33] Nikita Starodubcev, Mikhail Khoroshikh, Artem Babenko, and Dmitry Baranchuk. Invertible consistency distillation for text-guided image editing in around 7 steps. arXiv preprint arXiv:2406.14539, 2024. 2, 3, 7, 8
- [34] Keqiang Sun, Junting Pan, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, et al. Journeydb: A benchmark for generative image understanding. Advances in Neural Information Processing Systems, 36, 2024. 6
- [35] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1921–1930, 2023. 2, 3, 7
- [36] Tengfei Wang, Yong Zhang, Yanbo Fan, Jue Wang, and Qifeng Chen. High-fidelity gan inversion for image attribute editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2, 3
- [37] Weihao Xia, Yulun Zhang, Yujiu Yang, Jing-Hao Xue, Bolei Zhou, and Ming-Hsuan Yang. Gan inversion: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(3):3121–3138, 2023. 2

- [38] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. 2023. 3, 4, 6
- [39] Tianwei Yin, Micha¨el Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. In NeurIPS, 2024. 1, 2, 3, 12
- [40] Tianwei Yin, Micha¨el Gharbi, Richard Zhang, Eli Shechtman, Fr´edo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In CVPR, 2024. 1, 2, 3
- [41] Jiapeng Zhu, Yujun Shen, Deli Zhao, and Bolei Zhou. Indomain gan inversion for real image editing. In Proceedings of European Conference on Computer Vision (ECCV), 2020. 2, 3
- [42] Jun-Yan Zhu, Philipp Kr¨ahenb¨uhl, Eli Shechtman, and Alexei A Efros. Generative visual manipulation on the natural image manifold. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part V 14, pages 597–613. Springer, 2016. 3

# SwiftEdit: Lightning Fast Text-Guided Image Editing via One-Step Diffusion Supplementary Material

In this supplementary material, we first provide a detailed derivation of the regularization loss used in Stage 2, as outlined in Sec. 8. Next, we present several additional ablation studies in Sec. 9. Finally, we include more quantitative and qualitative results in Sec. 10, and Sec. 11. Then we discuss societal impacts in Sec. 12.

## 8. Derivation of the Regularization Loss in Stage 2

We provide a detailed derivation of the gradient of our proposed regularization loss, as defined in Eq. (8) of the main paper. The regularization loss is formulated as follows:

Lstage2regu = Et,ϵˆ w(t)∥ϵϕ(zt,t,cy) − ˆϵ∥22 , (10)

where ϵϕ(.) is a teacher denoising UNet, here, we use SD 2.1 in our implementation.

The gradient of the loss w.r.t our inversion network’s parameters θ is computed as:

∇θLstage2regu ≜ Et,ϵˆ [w(t)(ϵϕ(zt,t,cy) − ˆϵ) (

∂ˆϵ ∂θ

∂ϵϕ(zt,t,cy) ∂θ −

)],

(11)

where we absorb all constants into w(t). Expanding the term ∂ϵϕ(zt,t,cy)

∂θ , we have: ∂ϵϕ(zt,t,cy) ∂θ

∂ϵϕ(zt,t,cy) ∂zt

∂zt ∂z

∂z ∂θ

. (12)

=

Since z (extracted from real images) and θ are independent, ∂∂θz = 0, thus, we can turn Eq. (11) into:

∇θLstage2regu ≜ Et,ϵˆ w(t)(ϵϕ(zt,t,cy) − ˆϵ)(−

∂ˆϵ ∂θ

= Et,ϵˆ w(t)(ˆϵ − ϵϕ(zt,t,cy))

∂ˆϵ ∂θ

) (13)

, (14)

which has the opposite sign of the SDS gradient w.r.t z loss as discussed in the main paper.

## 9. Additional Ablation Studies

Compatibility of multi-step inversion with one-step textto-image model. To showcase the strength of our one-step inversion framework, we test existing inversion techniques on one-step generators. Specifically, we evaluate multi-step methods like DDIM Inversion (DDIMInv) and direct inversion on SBv2. As shown in the first and second row of

Reconstructed Image Edited Image

Source Image

[Figure 66]

[Figure 67]

[Figure 68]

Src Prompt: “ ” —> Edit Prompt: “dog”

[Figure 69]

[Figure 70]

[Figure 71]

Src Prompt: “ ” —> Edit Prompt: “ orange car ”

[Figure 72]

[Figure 73]

[Figure 74]

Src Prompt: “ dog” —> Edit Prompt: “ horse ”

[Figure 75]

[Figure 76]

[Figure 77]

Src Prompt: “ pizza” —> Edit Prompt: “ fried chicken ”

[Figure 78]

[Figure 79]

[Figure 80]

Src Prompt: “ woman” —> Edit Prompt: “ woman in red lipstick, sunglasses, scarf, hat”

Figure 8. Edit images with flexible prompting. SwiftEdit achieves satisfactory reconstructed and edited results with flexible source and edit prompt input (denoted under each image).

Tab. 5, these methods yield lower performance and slower inference time, while SwiftEdit excels with superior results and high efficiency.

Combined with other one-step text-to-image models. As discussed in the main paper, our inversion framework is not limited to SBv2 and can be seamlessly integrated with

###### Model PSNR↑ CLIP-Whole↑ CLIP-Edited↑

Ours + InstaFlow† 24.88 24.03 20.47 Ours + DMD2† 26.08 23.35 19.84 Ours + SBv1‡ 25.09 23.64 19.96

Ours + SBv2‡ (SwiftEdit) 23.33 25.16 21.25

Table 4. Ablation studies on combining our technique with other one-step text-to-image generation models. † means that these models are based on SD 1.5 while ‡ means that these models are based on SD 2.1.

Ours + SBv2 (SwiftEdit)

Ours + DMDv2 (SD 1.5) Ours + SBv1

Source Image Ours + Instaflow

[Figure 81]

a colorful bird standing on a branch -> a red bird standing on a branch

[Figure 82]

a plate with steak on it->a plate with salmon on it

[Figure 83]

white tiger on brown ground -> white cat on brown ground

Figure 9. Qualitative results when combining our inversion framework with other one-step text-to-image generation models.

other one-step text-to-image generators. To demonstrate this, we conducted experiments replacing SBv2 with alternative models, including DMD2 [39], InstaFlow [15], and SBv1 [20]. For these experiments, the architecture and pretrained weights of each generator G were used to initialize our inversion network in Stage 1. Specifically, DMD2 was implemented using the SD 1.5 backbone, while InstaFlow uses SD 1.5. All training experiments for both stages were conducted on the same dataset, similar to the experiments presented in Tab. 1 of the main paper.

Figure 9 presents edited results obtained by integrating our inversion framework with different one-step image generators. As shown, these one-step models integrate well with our framework, enabling effective edits. Additionally, quantitative results are provided in Tab. 4. The results indicate that our inversion framework combined with SBv2 (SwiftEdit) achieves the best editing performance in terms of CLIP-Whole and CLIP-Edited scores, while DMD2 demonstrates superior background preservation.

Two-stage training rationale. We provide additional ablation study where we train our network in a single stage using a mixed dataset of synthetic and real images. In particular, we construct a mixed training dataset comprised of: 10,000 synthetic image samples (generated by SBv2 using COCOA

Effects on Background Preservation

Effects on Editing Semantic

21.8

s_non-edit = 0.2 s_non-edit = 0.4 s_non-edit = 0.6

- 20

- 21

- 22

- 23

- 24

21.6

- s_non-edit = 0.8

- s_non-edit = 1

21.4

Clip-Edited

21.2

PSNR

21.0

s_non-edit = 0.2 s_non-edit = 0.4 s_non-edit = 0.6

20.8

- s_non-edit = 0.8

- s_non-edit = 1

20.6

20.4

0.0 0.2 0.4 0.6 0.8 1.0 s_edit

0.0 0.2 0.4 0.6 0.8 1.0 s_edit

(a) Varying sedit scale at different levels of snon-edit with default sy = 2.

Effects on Background Preservation

Effects on Editing Semantic

21.8

s_non-edit = 0.2 s_non-edit = 0.4 s_non-edit = 0.6

- 20

- 21

- 22

- 23

- 24

21.6

21.4

- s_non-edit = 0.8

- s_non-edit = 1

21.2

Clip-Edited

PSNR

21.0

s_non-edit = 0.2 s_non-edit = 0.4 s_non-edit = 0.6

20.8

20.6

- s_non-edit = 0.8

- s_non-edit = 1

20.4

20.2

1 2 3 4 s_y

1 2 3 4 s_y

(b) Varying sy scale at different levels of snon-edit with default sedit = 0.

Figure 10. Effects on background preservation and editing semantics while varying sedit and sy at different levels of snon-edit.

prompts), and 10,000 real samples of COCOA dataset. The goal of this experiment is to understand the behavior and advantage of two-stage training compared to single stage training with mixed dataset. As shown in the third row of Tab. 5, the combined training stage resulted in lower performance across all metrics compared to our two-stage strategy. This highlights the effectiveness of our two-stage strategy.

Varying scales. To better understand the effect of varying scales used in Eq. (9) in the main paper, we present two comprehensive plots evaluating the performance of SwiftEdit on 100 random test samples from the PieBench benchmark. Particularly, the plots depict results for varying sedit ∈ {0,0.2,0.4,0.6,0.8,1} (see Fig. 10a) or sy ∈ {0.5,1,1.5,2,2.5,3,3.5,4} (see Fig. 10b) at different levels of snon-edit ∈ {0.2,0.4,0.6,0.8,1}. As shown in Fig. 10a, it is evident at different levels of snon-edit that lower sedit generally improves editing semantics (CLIPEdited scores) but slightly compromises background preservation (PSNR). Conversely, higher sy can enhance promptimage alignment (CLIP-Edited scores, Fig. 10b), but excessive values (sy > 2) may harm prompt-alignment result. In all of our experiments, we use default choice of scale parameters setting where we set sedit = 0, snon-edit = 1, and sy = 2.

## 10. More Quantitative Results

In Tab. 6, we provide full scores on PieBench of comparison results in Tab. 1, with additional scores related to background preservation such as Structure Distance (SDis),

###### Method SDis↓ PSNR↑ LPIPS↓ MSE↓ SSIM ↑ CLIP-W ↑ CLIP-E↑ Time (s)↓

DirectInv + SBv2 0.050 15.5 0.25 0.003 0.65 24.3 20.3 9.25 DDIMInv + SBv2 0.060 14.4 0.29 0.004 0.63 22.7 19.7 3.85 SwiftEdit (Mixed Training) 0.005 22.5 0.09 0.0008 0.79 23.5 19.3 0.23

###### SwiftEdit (Ours) 0.001 23.3 0.08 0.0006 0.81 25.2 21.3 0.23

Table 5. Comparison of SwiftEdit with other settings on PieBench.

Type Method SDis×103↓ PSNR↑ LPIPS×103↓ MSE×104↓ SSIM×102 ↑ CLIP-W ↑ CLIP-E↑ Time↓

DDIM + P2P 69.43 17.87 208.80 219.88 71.14 25.01 22.44 25.98 NT-Inv + P2P 13.44 27.03 60.67 35.86 84.11 24.75 21.86 134.06

DDIM + MasaCtrl 28.38 22.17 106.62 86.97 79.67 23.96 21.16 23.21 Direct Inversion + MasaCtrl 24.70 22.64 87.94 81.09 81.33 24.38 21.35 29.68

Multi-step (50 steps)

DDIM + P2P-Zero 61.68 20.44 172.22 144.12 74.67 22.80 20.54 35.57 Direct Inversion + P2P-Zero 49.22 21.53 138.98 127.32 77.05 23.31 21.05 35.34

DDIM + PnP 28.22 22.28 113.46 83.64 79.05 25.41 22.55 12.62 Direct Inversion + PnP 24.29 22.46 106.06 80.45 79.68 25.41 22.62 12.79

InstructPix2Pix 57.91 20.82 158.63 227.78 76.26 23.61 21.64 3.85 InstructDiffusion 75.44 20.28 155.66 349.66 75.53 23.26 21.34 7.68

ReNoise (SDXL Turbo) 78.44 20.28 189.77 54.08 70.90 24.30 21.07 5.10 TurboEdit 16.10 22.43 108.59 9.48 79.68 25.50 21.82 1.31 ICD (SD 1.5) 10.21 26.93 63.61 3.33 83.95 22.42 19.07 1.38

Few-steps (4 steps)

SwiftEdit (Ours) 13.21 23.33 91.04 6.58 81.05 21.16 21.25 0.23 SwiftEdit (Ours with GT masks) 13.25 23.31 93.88 6.19 81.36 25.56 21.91 0.23

One-step

Table 6. Quantitative comparison of SwiftEdit against other editing methods with metrics employed from PieBench [11].

.

Source Image SwiftEdit Mask Edited Result Source Image SwiftEdit Mask Edited Result

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

dog on chair → fox on chair young child → old woman

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

girl → girl in sunglasses goat and cat → horse and cat

- Figure 11. Visualization of our extracted mask along with edited results using guided text described under each image row.

## 11. More Qualitative Results

Self-guided Editing Mask. In Fig. 11, we show more editing examples along with self-guided editing masks extracted directly from our inversion network.

Flexible Prompting. As shown in Fig. 8, SwiftEdit consistently reconstructs images with high fidelity, even with minimal source prompt input. It operates effectively with just a single keyword (last three rows) or no prompt at all (first two rows). Notably, SwiftEdit performs complex edits with ease, as demonstrated in the last row of Fig. 8, by simply combining keywords in the edit prompt. These results highlight its capabilities as a lightning-fast and user-friendly editing tool.

LPIPS, and SSIM. We additionally compare with other training-based image editing methods such as InstructPix2Pix (InstructP2P), and InstructDiffusion (InstructDiff). Unlike these methods, which require multi-step sampling and paired training data, SwiftEdit trains on source images alone for one-step editing. As shown, SwiftEdit outperforms both in quality and speed, thanks to its efficient onestep inversion and editing framework.

Facial Identity and Expression Editing. In Fig. 12, given a simple source prompt “man” and a portrait image, SwiftEdit can achieve face identity and facial expression editing via a simple edit prompt by just combining expression word (denoted on each row) and identity word (denoted on each column).

Additional Results on PieBench. In Figs. 13 to 15, we provide extensive editing results compared with other methods

###### Edited Image

“beckham” “ronaldo” “tom cruise” “chris evans”

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

“ ”

###### Soure Image

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

“smiling”

“man”

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

“angry”

- Figure 12. Face identity and expression editing via simple prompts. Given a portrait input image, SwiftEdit can perform a variety of facial identities along with expression editing scenarios guided by simple text within just 0.23 seconds.

on the PieBench benchmark.

## 12. Societal Impacts

As an AI-powered visual generation tool, SwiftEdit delivers lightning-fast, high-quality, and customizable editing capabilities through simple prompt inputs, significantly enhancing the efficiency of various visual creation tasks. However, societal challenges may arise as such tools could be exploited for unethical purposes, including generating sensitive or harmful content to spread disinformation. Addressing these concerns are essential and several ongoing works have been conducted to detect and localize AI-manipulated images to mitigate potential misuse.

[Figure 109]

a monkey wearing colorful goggles and a colorful scarf -> a man wearing colorful goggles and a colorful scarf

[Figure 110]

a poster of a bus driving down a road with mountains … -> a poster of a bus road with mountains …

[Figure 111]

a woman in a coat holding a camera->a woman in a coat holding a phone

[Figure 112]

a ﬂuﬀy cat with yellow eyes sitting on a wooden ﬂoor->a ﬂuﬀy panda with yellow eyes sitting on a wooden ﬂoor

[Figure 113]

a digital art woman with curly hair standing …->a digital art woman with straight hair standing …

[Figure 114]

a black bird with a yellow beak and yellow feet->a green bird with a yellow beak and yellow feet

[Figure 115]

a stream in a lush green forest with rocks->a road in a lush green forest with rocks

[Figure 116]

a collie dog is sitting on a bed->a garﬁeld cat is sitting on a sofa

[Figure 117]

a vase with sunﬂowers and pears on a table->a vase with sunﬂowers and bananas on a table

Figure 13. Comparative results on the PieBench benchmark

[Figure 118]

a cat sitting on a wooden chair -> a dog sitting on a wooden chair

[Figure 119]

a colorful bird standing on a branch->a red bird standing on a branch

[Figure 120]

a beautiful young woman with clean background->a beautiful young woman with blue background

[Figure 121]

an orange cat sitting on top of a fence -> a black cat sitting on top of a fence

[Figure 122]

a church in the countryside with a fence and trees-> a church in the countryside with a fence and trees

[Figure 123]

a plate with steak on it->a plate with salmon on it

[Figure 124]

a golden retriever holding a ﬂower sitting on … -> a golden retriever holding a ﬂower sitting on …

[Figure 125]

a colorful car is parked on the street->a colorful motorcycle is parked on the street

[Figure 126]

a paraglider is ﬂying over a mountain with snow … -> a paraglider is ﬂying over a mountain with snow …

Figure 14. Comparative results on the PieBench benchmark

[Figure 127]

a tiger swimming in a pond of green algae -> a dog swimming in a pond of green algae

[Figure 128]

a small mushroom is sitting on top of a pine branch-> a small mushroom is sitting on top of a pine branch

[Figure 129]

a man sitting on a rock with trees in the background->a man sitting on a rock with a city in the background

[Figure 130]

a woman with gold makeup->a woman with blue makeup

[Figure 131]

painting of a shepherd dog sitting in a laundry room … >painting of a poodle dog sitting in a laundry room …

[Figure 132]

two boats are docked on the shore of a lake-> two boats are docked on the shore of a lake

[Figure 133]

a woman with a mask and ﬂowers in her hair->a woman with a mask and crown in her hair

[Figure 134]

woman with brown hair->woman with blue hair

[Figure 135]

a husky dog running on a path in the woods-> a husky dog running on a path in the woods

Figure 15. Comparative results on the PieBench benchmark

