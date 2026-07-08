## FreeMorph: Tuning-Free Generalized Image Morphing with Diffusion Model

Yukang Cao1∗ Chenyang Si2∗‡ Jinghao Wang3 Ziwei Liu1† 1S-Lab, Nanyang Technological University, 2Nanjing University 3The Chinese University of Hong Kong https://yukangcao.github.io/FreeMorph/

Generated transitions

Input source Input target

[Figure 1]

arXiv:2507.01953v1[cs.CV]2Jul2025

[Figure 2]

[Figure 3]

[Figure 4]

- Figure 1. Examples of image morphing obtained via FreeMorph. Given two input images, FreeMorph effectively generates smooth transitions between them within 30 seconds.

### Abstract

diffusion model. In this paper, we introduce FreeMorph to address these challenges by integrating two key innovations. 1) We first propose a guidance-aware spherical interpolation design that incorporates explicit guidance from the input images by modifying the self-attention modules, thereby addressing identity loss and ensuring directional transitions throughout the generated sequence. 2) We further introduce a step-oriented variation trend that blends self-attention modules derived from each input image to achieve controlled and consistent transitions that respect both inputs. Our extensive evaluations demonstrate that FreeMorph outperforms existing methods, being 10× ∼ 50× faster and establishing a new state-of-the-art for image morphing.

We present FreeMorph, the first tuning-free method for image morphing that accommodates inputs with different semantics or layouts. Unlike existing methods that rely on finetuning pre-trained diffusion models and are limited by time constraints and semantic/layout discrepancies, FreeMorph delivers high-fidelity image morphing without requiring perinstance training. Despite their efficiency and potential, tuning-free methods face challenges in maintaining highquality results due to the non-linear nature of the multi-step denoising process and biases inherited from the pre-trained

∗ Equal contributions, ‡ Project lead, † Corresponding author. Jinghao was a Master student at NTU during this work.

### 1. Introduction

Given two distinct input images, image morphing [25, 51] aims to gradually change attributes such as shape, texture, and overall layout to produce a series of intermediate images that transition smoothly from one to the other. This process is widely used in fields such as animation, film, and photo editing [1, 45, 46], offering an effective means of enhancing creative expression. Historically, image morphing relied on image warping [9, 37, 44] for aligning corresponding points and on color interpolation [2, 21] for blending. These methods, however, often fall short when handling complex textural and semantic transitions, making them less effective for images with intricate details. With advancements in deep learning, Generative Adversarial Networks (GANs) [4, 11, 17, 34] and Variational Autoencoders (VAEs) [20] have significantly improved image morphing by enabling latent code interpolation. Despite their capabilities, these approaches still face challenges with real-world images due to limited training data and information loss during GAN inversion. This underscores the need for methods that better preserve identity and offer greater generalization.

Recently, with the availability of large-scale text-image datasets, vision-language models (e.g., Chameleon [40]), diffusion models (e.g., Stable Diffusion [31, 32, 39]), and transformers (e.g., PixArt-α [6], FLUX [3]) have demonstrated impressive capabilities in generating high-quality images from text prompts. These advancements have paved the way for new generative image morphing techniques. Specifically, Wang and Golland [43] leverages the local linearity of CLIP-based text embeddings to create smooth transitions by interpolating latent image features. Building on this idea, IMPUS [47] introduces a multi-phase training framework that includes optimizing text embeddings and training Low-Rank Adaptation (LoRA) modules to better capture semantics. While this method yields more visually appealing results, it requires extensive training, typically around 30 minutes per case. DiffMorpher [49] proposes to directly interpolate latent noise and leverage Adaptive Instance Normalization (AdaIN) to improve performance. However, these methods still struggle to process images with diverse semantics and intricate layouts, limiting their practical effectiveness.

Given these issues, our objective is to achieve image morphing without requiring further tuning. Nonetheless, this goal introduces two key challenges: 1) Non-directional transitions and identity loss1. While converting input images into latent features using a pre-trained diffusion model and then applying spherical interpolation might seem straightforward, this approach often results in inconsistent transitions. This is due to the non-linear nature of the multistep denoising process. Additionally, this method inherits

1Non-directional transitions, akin to identity loss, result in generated images that deviate from the identity of the input images.

biases from the pre-trained model, which can lead to identity loss in the generated images. 2) Achieving consistent transitions2. A diffusion model does not inherently provide an effective "variation trend" to capture the gradual changes between images. Consequently, achieving smooth and gradual transitions in a tuning-free manner remains a significant challenge without additional adjustments.

In this paper, we present FreeMorph, a novel tuning-free method capable of instantly generating directional and realistic transitions between two images. Our method introduces two novel components: 1) Guidance-aware spherical interpolation: We first enhance the pre-trained diffusion model by incorporating explicit guidance from the input images through modifications to its self-attention modules. This is achieved through spherical interpolation, which produces intermediate features used in two key ways. First, we perform spherical feature aggregation to blend the key and value features of the self-attention modules, ensuring consistent transitions throughout the generated image sequence. Second, to address identity loss, we introduce a prior-driven self-attention mechanism that incorporates explicit guidance from the input images to preserve their unique identities. 2) Step-oriented variation trend: To achieve consistent transitions, we introduce a novel step-oriented variation trend. This method blends two self-attention modules, each derived from one of the input images, enabling a controlled and consistent transition that respects both inputs. To further improve the quality of the generated image sequences, we designed an improved reverse denoising and forward diffusion process that seamlessly integrates these innovative components into the original DDIM framework. As shown in Fig. 1 and Fig. 4, our approach adeptly handles diverse input types, whether they have similar or distinct semantics and layouts, producing smooth and realistic transitions.

To thoroughly assess FreeMorph and benchmark it against current methods, we also collect a new evaluation dataset that includes four distinct sets of image pairs, categorized by their semantic and layout similarity. Our extensive evaluations demonstrate that FreeMorph substantially outperforms existing approaches. FreeMorph produces highfidelity image sequences with smooth and coherent transformations in under 30 seconds, making it 50× faster than IMPUS [47] and 10× faster than DiffMorpher [49].

### 2. Related Work

Text-to-Image Generation. Recently, diffusion models [28, 30–32] have emerged as the de facto standard for text-to-image generation. These models employ a series of denoising steps (e.g., DDIM, DDPM) [15, 38] to transform Gaussian noise into images, effectively capturing and interpreting details from textual prompts. Trained on billions

2Inconsistent transitions are those with abrupt changes.

of text-image pairs [35], these models exhibit a remarkable ability to understand the distribution of real-world images, generating high-quality, diverse outputs while maintaining strong generalization capabilities. Our work harnesses the capabilities of diffusion models, particularly their ability to generate smooth transitions between two specified images [19, 29, 33], to address the image morphing task.

Image Morphing. Image morphing is a long-standing computer vision and graphics problem. Before the deep learning era, techniques such as mesh warping [9, 37, 44] and field morphing [2, 21] were the primary approaches in this domain. Early approaches [10, 26] utilize GANs [? ] to achieve this objective. However, they generally suffer from three main limitations: (1) the need for extensive training, (2) poor generalization to out-of-domain inputs, and (3) an inability to handle inputs with varying layouts and semantic structures. Recently, advancements in diffusion models have led to significant progress, as demonstrated by methods such as DiffMorpher [49], IMPUS [47], and the work of Wang and Golland [43]. These approaches focus on optimizing text embeddings for two images and fine-tuning pre-trained text-to-image diffusion models to achieve smooth interpolation. However, they often require extensive fine-tuning for each image pair and are limited to images with similar semantics and layouts. This can also hinder the generalizability of pre-trained diffusion models due to constraints imposed by LoRA modules in the U-Net architecture. In contrast, our method offers a tuning-free framework that requires no modifications to the original diffusion models, thereby preserving their inherent generalizability. Additionally, our approach significantly improves efficiency and can handle images with different layouts and semantics, addressing a key limitation of existing techniques.

Tuning-Free Text-Guided Image Editing. Recent image translation methods have emerged that edit either generated or real-world images using text in a training-free manner, without altering the internal computations of the U-Net. For instance, SDEdit [24] proposes a straightforward method that adds T time steps of Gaussian noise to an original image and then denoises it using guiding text. Conversely, EDICT [42] and FPI [23] focus on inverting a reference image back to the latent space and subsequently applying the inverted latent as a condition guided by text. Additionally, methods like P2P [13], PnP [41], and MasaCtrl [5] modify the attention mechanism within diffusion models to enhance alignment between the guiding text and the consistency of generated images with their originals. Drawing inspiration from these techniques, our method facilitates image morphing in a tuning-free manner. Notably, our approach also achieves comparable image editing performance by framing text-guided editing as a special case of morphing between a real and a generated image.

### 3. Methodology

Given two independent images, Ileft and Iright, as input, our objective is to generate a sequence of intermediate images

S = {Ij}Jj=1 that smoothly transforms from one to the other in a tuning-free manner. We set J = 5 for the experiments reported in this paper. As illustrated in Algorithm 1, our pipeline employs a pre-trained diffusion model as its foundation and integrates guidance from the input images into the multi-step denoising process. In the subsequent sections, we first introduce the preliminaries that underpin our method in Sec. 3.1. Next, we describe the FreeMorph framework in detail. This framework comprises three main components: 1) the guidance-aware spherical interpolation (Sec. 3.2), which includes our proposed spherical feature aggregation and prior-driven self-attention mechanism; 2) a step-oriented variation trend that enables controlled and consistent image morphing (Sec. 3.3); and 3) our improved forward diffusion and reverse denoising processes (Sec. 3.4).

#### 3.1. Preliminaries

Denoising Diffusion Implicit Model (DDIM). The Denoising Diffusion Implicit Model (DDIM) [38], trained on large-scale text-image datasets, is designed to reconstruct images from noisy inputs. After training, it establishes a deterministic mapping from an initial noise state xT to an image x0, a process we refer as reverse denoising steps:

√1 − α¯tϵθ(xt) √α¯t

xt−1 =√α¯t−1(

xt −

)

(1)

+ 1 − α¯t−1 − σt2ϵθ(xt, t) + σtϵt.

Conversely, by inverting the formula above, we can derive the forward diffusion process, which incrementally adds noise to an image to predict its noise state:

α¯t α¯t−1

xt−1+

xt =

(2)

√αt(

1 αt − 1 −

1 αt−1 − 1)ϵθ(xt−1, t − 1).

Latent Diffusion Model (LDM). Building upon DDIM, the Latent Diffusion Model (LDM) [31] is a refined variant of diffusion models that effectively balances image quality with denoising efficiency. Specifically, LDM utilizes a pre-trained variational auto-encoder (VAE) [20] to map images into a latent space and then trains the diffusion model within this space. Furthermore, LDM enhances the UNet architecture by incorporating self-attention modules, cross-attention layers, and residual blocks to integrate text prompts as conditional inputs during image generation. The attention mechanism in LDM’s UNet can be formulated as:

Q · KT √dk

ATT(Q,K,V ) = softmax(

) · V (3)

[Figure 5]

[Figure 6]

Spherical Interpolation

[Figure 7]

[Figure 8]

Replace Key, Value features

- Figure 2. Replacing the key and value feature in the attention mechanism. We can observe that good key and value features would lead to smooth transitions and identity preservation. where Q denotes the query features from spatial data, and K and V are key and value features derived from either spatial data (for self-attention) or text embeddings (for crossattention). The noise estimator in LDM is then extended to

ϵθ(xt,t,y), where y denotes the text embedding.

Our approach builds upon the Stable Diffusion model [39], a pre-trained LDM developed by StabilityAI, and utilizes a vision-language model (VLM), LLaVA [22], for generating captions for the input images.

- 3.2. Guidance-aware spherical interpolation

the right image Iright can largely enhance the smoothness and identity preservation of the image transitions, although some imperfections may remain (see Fig. 2). Motivated by this finding, and recognizing that the query features (Q) largely reflect the overall image layout, we propose first blending features from both the left and right images (Ileft, Iright) to provide explicit guidance for the multi-step denoising process. Specifically, in the denoising step t, we first feed the latent of the input images zt−left and zt−right to the pre-trained UNet ϵθ to obtain the key and value features. Following that, We then substitute the original K and V with those derived from the input images and compute their average to modify the attention mechanism:

[Figure 9]

Latent noise inverted from

[Figure 10]

[Figure 11]

( )

Existing image morphing methods [25, 47, 49] typically involve training Low-rank Adaptation (LoRA) modules for each input image to enhance semantic comprehension and achieve smooth transitions. However, this approach is often inefficient and time-consuming and struggles with images that differ in semantics or layout. In this paper, we propose a tuning-free image morphing approach built on the pre-trained Stable Diffusion model. By leveraging the capabilities of DDIM (as in Eq. 2) for image inversion and interpolation, one might consider converting the input images (Ileft, Iright) into latent features (z0−left, z0−right) and applying spherical interpolation may seem like a simple straightforward solution:

Noise distortion rate

- 1

- 2 · (ATT(Qt−j, Kt−left, Vt−left)

ATT(Qt−j, Kt−j, Vt−j) : =

+ ATT(Qt−j, Kt−right, Vt−right))

(5)

where Qt−j, Kt−j, Vt−j are obtained by inputting zt−j to the pre-trained UNet ϵθ. Note that zt−j, zt−left and zt−right are derived based on Eq. 3.

Prior-driven Self-attention Mechanism. While our feature blending technique significantly improves identity preservation in image morphing, we found that using this approach uniformly in both forward diffusion and reverse denoising stages can result in transitions where the image sequences change minimally and fail to accurately represent the input images (see Fig. 6). This outcome is anticipated because the latent noise will largely influence the reverse denoising process, as shown in Fig. 3. Consequently, applying our feature blending, depicted in Eq. 5, introduces ambiguity as the consistent and strong constraints from the input images cause each latent noise i to appear similar, thereby limiting the effectiveness of the transitions. To tackle this issue, we further propose a prior-driven self-attention mechanism that prioritizes the latent features from spherical interpolation to ensure smooth transitions within the latent noise, while emphasizing the input images to maintain identity preservation afterward. Specifically, during the reverse denoising stage, we use the approach described in Eq. 5, while for the forward diffusion steps, we employ a different attention mechanism

sin((1 − j) · ϕ) sinϕ ·z0−left +

sin(j · ϕ)

sinϕ ·z0−right, (4) where j ∈ [1,J] is the index of intermediate images, and ϕ = arccos( z

z0−j =

T 0−left·z0−right

||z0−left||·||z0−right). Recall that we set J = 5 in our paper. However, directly inverting these interpolated latent features z0−j to generate images often results in inconsistent transitions and identity loss (see Fig. 2). This issue arises because (1) the multi-step denoising process is highly non-linear, leading to discontinuous image sequences, and (2) there is no explicit guidance to control the denoising, causing the model to inherit biases from the pre-trained diffusion model.

Spherical Feature Aggregation. Drawing insights from previous image editing techniques [5, 13, 27, 36, 41], we observed that using the features z0−j as initialization and replacing the key and value features (K and V ) in the attention mechanism (as described in Eq. 3) with features from

Spherical Interpolation

Replace Key, Value features

[Figure 16]

Latent noise inverted from

[Figure 17]

[Figure 18]

# ( )

Noise distortion rate

- Figure 3. Effectiveness of the latent noise on the generated images. The pre-trained diffusion model is robust to the noise distortion within the latent space. Algorithm 1 FreeMorph Input: Ileft, Iright

#### 3.3. Step-oriented variation trend

After obtaining image sequences that are directional and accurately reflect the input identities, the next challenge is to achieve a consistent and gradual transition from the left image Ileft to the right image Iright. This problem stems from the lack of a "variation trend" that captures the changes from Ileft to Iright. To this end, we propose a step-oriented variation trend that gradually changes the influence between the input images (Ileft and Iright):

- 1: Caption the input images via pre-trained LLaVA → Textleft, Textright.
- 2: Obtain image features z0−left, z0−right, and text embedding yleft, yright via VAE and text encoder of pre-trained Stable Diffusion.
- 3: Applying spherical interpolation to obtain z0−j where j ∈ [1,J] as initialization.
- 4: Forward diffusion steps (from image to latent noise): for t = 1 to T do

if t < λ1 · T then

Apply the original attention mechanism.

else if t < λ2 · T then Apply the prior-driven self-attention mechanism as in Eq. 6.

else

Apply the step-oriented motion flow as in Eq. 7. end if

end for

- 5: High-frequency Gaussian noise injection.
- 6: Reverse denoising steps (from latent noise to image): for t = 1 to T do

if t < λ3 · T then

Apply the step-oriented motion flow as in Eq. 7. else if t < λ4 · T then

Apply the spherical feature aggregation as in Eq. 5. else

Apply the original attention mechanism. end if

end for

- 7: Add text-conditioned features.

ATT(Qt−j, Kt−j, Vt−j) : = (1 − αj) · ATT(Qt−j, Kt−left, Vt−left)

+ αj · ATT(Qt−j, Kt−right, Vt−right), (7)

where αj = j/(J +2−1), with J +2 representing the total number of images, which includes the J generated images and the 2 input images.

#### 3.4. Forward diffusion and reverse denoising process

High-frequency Gaussian Noise Injection. As discussed earlier, FreeMorph incorporates features from both the left and right images during the forward diffusion and reverse denoising stages. Nevertheless, we have observed that this can occasionally impose overly stringent constraints on the generation process. To mitigate this issue and allow for greater flexibility, we propose introducing Gaussian noise into the latent vector z in the high-frequency domain after the forward diffusion steps:

IFFT(FFT(z)), if m = 1 IFFT(FFT(g)), if m = 0

(8)

z :=

Output: J intermediate images gradually change from Ileft to Iright.

Here, IFFT(·) and FFT(·) denote the inverse fast Fourier transform and fast Fourier transform, respectively. g ∼ N(0,1) represents a randomly sampled noise vector, and m is a binary high-pass filter mask of the same size as z.

as follows by modifying the self-attention modules:

Overall process. To enhance the efficacy of our image morphing process, we have found that consistently applying either guidance-aware spherical interpolation (Sec. 3.2) or step-oriented variation trend (Sec. 3.3) across all denoising steps yields suboptimal results (see Sec. 4.3). To address this, we have developed a refined approach for both forward diffusion and reverse denoising processes. We provide an

J

1 J

ATT(Qt−j,Kt−j,Vt−j) :=

ATT(Qt−j,Kt−k,Vt−k) (6) Refer to Sec. 4.3 for detailed ablation studies on this design.

k=1

###### Table 1. Quantitative comparison with existing image morphing techniques.

MorphBench Morph4Data Overall

Method

LPIPSsum ↓ FIDmean ↓ PPLsum ↓ LPIPSsum ↓ FIDmean ↓ PPLsum ↓ LPIPSsum ↓ FIDmean ↓ PPLsum ↓

IMPUS [47] 130.52 152.43 3263.03 134.88 210.66 3199.90 265.40 174.76 6462.93 DiffMorpher [49] 90.57 157.18 2264.20 98.56 292.54 2394.05 189.13 209.10 4658.25 Spherical Interpolation 119.77 169.17 2994.35 103.74 245.22 2593.58 223.52 198.34 5587.93 Ours 84.91 141.32 2122.80 80.30 201.09 2007.52 162.99 152.88 4192.82

Input

Input source

Generated transitions Input source

Input target

Generated transitions

target

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

- Figure 4. More results produced by FreeMorph. Our method can achieve smooth and high-fidelity image transitions for input images with either similar or different semantics and layouts. overview algorithm of our proposed FreeMorph in Algorithm. 1. Specifically:

[Figure 27]

###### as follows: λ1 = 0.3, λ2 = 0.6, λ3 = 0.2, λ4 = 0.6.

Evaluation Datasets. DiffMorpher [49] introduced MorphBench, which includes 24 animation pairs and 66 image pairs, predominantly featuring images with similar semantics or layouts. To complement this dataset and mitigate potential biases, we introduce Morph4Data, a newly curated evaluation dataset comprising four categories: 1) Class-A, consisting of 25 image pairs with similar layouts but different semantics, sourced from Wang and Golland [43]; 2) Class-B, containing image pairs with both similar layouts and semantics, including 11 pairs of faces from CelebA-HQ [16] and 10 pairs of various car types; 3) Class-C, featuring 15 pairs of randomly sampled images from ImageNet-1K [8] with no semantic or layout similarity; 4) Class-D, comprising 15 pairs of dog and cat images randomly sampled from the internet.

- • Forward diffusion: We use the standard self-attention mechanism for the first λ1 · T steps. From λ1 · T to λ2 ·T, we apply the feature blending technique from Eq. 6. For the remaining steps, we implement the step-oriented variation trend.
- • Reverse denoising: We begin with the step-oriented variation trend for the first λ3 · T steps, followed by the feature blending method from Eq. 5 for steps between λ3 · T and λ4 · T. The process ends with the original self-attention mechanism for the final steps to produce images with higher fidelity.

[Figure 28]

Here, λ1, λ2, λ3, and λ4 are hyper-parameters and T = 50 is the total number of steps.

### 4. Experiments

#### 4.1. Quantitative Evaluations

We evaluate the performance of FreeMorph across various scenarios, comparing it with state-of-the-art image morphing techniques and conducting ablation studies to highlight the effectiveness of our proposed components.

Following IMPUS [47] and DiffMorpher [49], we conducted quantitative comparisons using the following metrics: 1) Frechet Inception Distance (FID) [14], which assesses the similarity between the distributions of input and generated images; 2) Perceptual Path Length (PPL) [18], where we calculate the sum of PPL loss between adjacent images; and 3) Learned Perceptual Image Patch Similarity (LPIPS) [50], which we also sum for adjacent images to evaluate the smoothness and coherence of the generated transitions. The results, detailed in Table 1, demonstrate the superior performance of our method across both datasets, showing enhanced fidelity, smoothness, and directness.

Implementation Details. We use version 2.1 of the publicly available Stable Diffusion model. Both the forward diffusion and reverse denoising processes employ a DDIM schedule with T = 50 steps. It takes under 30 seconds for our method to produce a morphing sequence using NVIDIA A100 GPU. Following the Stable Diffusion setup, we operate on an image resolution of 768 × 768. We set the classifier-free guidance (CFG) parameter to 7.5 to incorporate text-conditioned features. The hyperparameters are set

Input source

Input target

###### Input source Input target

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

DiffMorpherIMPUS Spherical

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Interpolation Ours

[Figure 39]

[Figure 40]

Generated transitions

Generated transitions

- Figure 5. Qualitative comparison with existing image morphing techniques. Unlike other methods that struggle or fail to generate smooth and high-fidelity results without identity loss, our approach consistently achieves high-quality transitions, yielding superior results.

###### Table 2. User studies.

PUS [47] exhibits identity loss and produces unsmooth transitions; For instance, in the second example of Fig. 5, IMPUS exhibits (i) identity loss, where the third generated image deviates from the original identity, and (ii) an abrupt transition between the third and fourth generated images. 2) Although Diffmorpher [49] achieves smoother transitions than IMPUS, its results often suffer from blurriness and lower overall quality (see the first example in Fig. 5); 3) We also evaluate a baseline approach, ‘Slerp’, which involves applying only spherical interpolation and the DDIM process. The visualizations show that this baseline approach struggles with (i) accurately interpreting the input images due to the absence of explicit guidance, (ii) suboptimal image quality, and (iii) abrupt transitions. In contrast, our method consistently delivers superior performance, characterized by smoother transitions and higher image quality. Additional comparisons are available in the Appendix.

IMPUS [47] DiffMorpher [49] Slerp Ours Preference 17.16% 14.89% 7.82% 60.13%

User studies To enhance our comparative analysis by including human preferences, we conducted user studies. We recruited 30 volunteers, including animators, AI experts, and gaming enthusiasts aged 20 to 35, to select their preferred results. Each participant was shown 50 random pairs of comparative results. The outcomes, presented in Table 2, demonstrate the subjective effectiveness of our proposed approach. Note that slerp denotes the method that only applies spherical interpolation.

#### 4.2. Qualitative Evaluations

Qualitative Results. In Fig. 1 and Fig. 4, we present a wide range of results produced by FreeMorph, which consistently demonstrate its ability to generate high-quality and smooth transitions. FreeMorph excels across diverse scenarios, accommodating images with different semantics and layouts, as well as those with similar characteristics. FreeMorph also effectively handles subtle variations, such as cakes with different colors and individuals with different expressions.

#### 4.3. Further Analysis

Analysis of Guidance-aware Spherical Interpolation. In Fig. 6, we present ablation studies to evaluate the effects of the proposed spherical feature aggregation (Eq. 5) and the prior-driven self-attention mechanism (Eq. 6). The results indicate that using either component alone produces suboptimal outcomes. Specifically, (i) spherical feature aggregation is crucial for achieving directional transitions in which the characteristics of Ileft gradually diminish, and (ii) the prior-driven self-attention mechanism is vital for preserving identity in the generated images. The combination of

Qualitative Comparisons. We provide qualitative comparisons with existing image morphing methods in Fig. 5. An effective image morphing outcome should exhibit gradual transitions from the source (left) image to the target (right) image while preserving the original identities. Based on this criterion, several observations can be made: 1) When handling images with varying semantics and layouts, IM-

###### Table 3. Quantitative comparison for ablation studies.

MorphBench Morph4Data Overall

Method

LPIPSsum ↓ FIDmean ↓ PPLsum ↓ LPIPSsum ↓ FIDmean ↓ PPLsum ↓ LPIPSsum ↓ FIDmean ↓ PPLsum ↓ w/ only Eq. 6 157.01 320.05 3425.19 141.12 411.80 3028.05 298.13 355.24 6453.24

- w/ only Eq. 5 99.69 155.51 2491.10 90.80 217.26 2270.05 190.49 179.20 4761.15

- w/ only Eq. 6 and Eq. 5 211.52 243.08 5288.10 139.55 290.11 3488.87 351.08 261.12 8776.96 w/o noise injection 99.49 154.53 2487.16 89.12 211.23 2228.03 188.61 176.28 4715.19

- w/o Eq. 5 87.41 155.46 2185.30 81.10 218.95 2027.58 168.52 179.82 4212.88

- w/o Eq. 6 120.01 148.54 3000.35 101.28 215.43 2572.06 221.30 174.19 5572.41 w/o step-oriented motion flow 118.50 154.71 2962.48 93.39 214.93 2334.68 211.89 177.80 5297.17

- Ours (Var-A) 153.40 184.54 3835.08 115.91 243.20 2897.63 269.31 207.04 6732.70

- Ours (Var-B) 93.54 158.44 2338.62 85.76 245.36 2144.08 179.31 191.78 4482.70 Ours 84.91 141.32 2122.80 80.30 201.09 2007.52 162.99 152.88 4192.82

[Figure 41]

Input

Input

Input

Input

source

target

source

target

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

A2A1Ours

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Generated transitions

Generated transitions

Figure 8. Analysis of high-frequency noise injection and steporiented motion flow. A1: w/o step-oriented motion flow; A2: w/o high-frequency noise injection

w/o high-frequency

AblationStudies

Noise injection

###### Figure 6. Analysis of guidance-aware spherical interpolation.

sess its impact. We observe that without this component, the model tends to produce abrupt changes rather than smooth transitions. Additionally, the final generated image exhibits high-contrast colors that differ from the target image Iright. In contrast, the step-oriented variation trend enables our method to achieve smoother transitions and produce a final image that is more closely aligned with the target image.

Input

Input

Input

Input

w/o step-oriented

Wo-aggressive

source

target

source

target

motion flow

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

w/o high-frequency noise injection

Ours

[Figure 56]

[Figure 57]

Ours

- (Var-A)

Ours

- (Var-B)

Ours

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

Analysis of High-frequency Noise Injection. We then disable high-frequency noise injection and present the corresponding ablation study in Fig. 8. The results indicate that incorporating the proposed high-frequency noise injection enhances the model’s flexibility and contributes to smoother transitions.

Ours

Generated transitions

Generated transitions

Figure 7. Analysis of reverse diffusion and forward denoising process.

normal Aggressive + gradual + normal

both components allows FreeMorph to produce smooth transitions while effectively maintaining identity. By comparing the last two rows in Fig. 6, we demonstrate the importance of our step-oriented variation trend and the specially designed reverse and forward processes.

AblationStudies

### 5. Conclusion

We have introduced FreeMorph, a novel tuning-free pipeline capable of generating smooth, high-quality transitions between two input images in under 30 seconds. Specifically, we propose incorporating explicit guidance from the input images by modifying the self-attention modules. This is achieved through two novel components: spherical feature aggregation and a prior-driven self-attention mechanism. Additionally, we introduce a step-oriented variation trend to ensure directional transitions consistent with both input images. We also designed an improved forward diffusion and reverse denoising process to integrate our proposed modules into the original DDIM framework. Extensive experiments demonstrate that FreeMorph delivers high-fidelity results across various scenarios, significantly outperforming existing image morphing techniques.

Wo-normal

Analysis of Reverse and Forward Process. In Fig. 7, we evaluate our method against two variants: (i) “Ours (Var-A),” which omits the original attention mechanism, and (ii) “Ours (Var-B),” which swaps the application steps of the guidanceaware spherical interpolation and the step-oriented variation trend in both the reverse and forward processes. A comparison of these variants with our final design reveals that (i) the original attention mechanism is crucial for achieving high-fidelity results, and (ii) the specific configuration of the reverse and forward processes in our final design yields optimal performance.

Analysis of Step-oriented Variation Trend. In Fig. 8, we first disable the proposed step-oriented variation trend to as-

### References

- [1] Alyaa Qusay Aloraibi. Image morphing techniques: A review. Technium, 9, 2023. 2
- [2] Thaddeus Beier and Shawn Neely. Feature-based image metamorphosis. In SIGGRAPH, pages 35–42. ACM, 1992. 2, 3
- [3] Forest Black. Flux.1. https://blackforestlabs.ai/announcingblack-forest-labs/, 2024. 2
- [4] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. In ICLR, 2019. 2
- [5] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22560–22570, 2023. 3, 4
- [6] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023. 2
- [7] Xinyuan Chen, Yaohui Wang, Lingjun Zhang, Shaobin Zhuang, Xin Ma, Jiashuo Yu, Yali Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Seine: Short-to-long video diffusion model for generative transition and prediction. In The Twelfth International Conference on Learning Representations, 2023. 13
- [8] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 6
- [9] Karl M Fant. A nonaliasing, real-time spatial transform technique. IEEE Computer Graphics and Applications, 6(1): 71–80, 1986. 2, 3
- [10] Noa Fish, Richard Zhang, Lilach Perry, Daniel Cohen-Or, Eli Shechtman, and Connelly Barnes. Image morphing with perceptual constraints and stn alignment. In Computer Graphics Forum, pages 303–313. Wiley Online Library, 2020. 3
- [11] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In NeurIPS,

2014. 2

- [12] Jiayi Guo, Xingqian Xu, Yifan Pu, Zanlin Ni, Chaofei Wang, Manushree Vasu, Shiji Song, Gao Huang, and Humphrey Shi. Smooth diffusion: Crafting smooth latent spaces in diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7548–7558,

2024. 12

- [13] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-or. Prompt-to-prompt image editing with cross-attention control. In ICLR, 2023. 3, 4
- [14] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium.

In NeurIPS, 2017. 6

- [15] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 2
- [16] Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of GANs for improved quality, stability, and variation. In ICLR. OpenReview.net, 2018. 6
- [17] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In CVPR, pages 4401–4410. Computer Vision Foundation / IEEE, 2019. 2
- [18] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. pages 8110–8119, 2020. 6
- [19] Valentin Khrulkov, Gleb V. Ryzhakov, Andrei Chertkov, and Ivan V. Oseledets. Understanding DDPM latent codes through optimal transport. In ICLR. OpenReview.net, 2023. 3
- [20] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 2, 3
- [21] Tong-Yee Lee, Young-Ching Lin, YN Sun, and Leeween Lin. Fast feature-based metamorphosis and operator design. In Computer Graphics Forum, pages 15–22. Wiley Online Library, 1998. 2, 3
- [22] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 4
- [23] Barak Meiri, Dvir Samuel, Nir Darshan, Gal Chechik, Shai Avidan, and Rami Ben-Ari. Fixed-point inversion for text-toimage diffusion models. arXiv preprint arXiv:2312.12540,

2023. 3

- [24] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. In ICLR. OpenReview.net, 2022. 3
- [25] Chigozie Nri. Differentiable morphing. https://github.com/volotat/DiffMorph, 2022. 2, 4
- [26] Sanghun Park, Kwanggyoon Seo, and Junyong Noh. Neural crossbreed: neural based image metamorphosis. ACM Transactions on Graphics (TOG), 39(6):1–15, 2020. 3
- [27] Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot image-to-image translation. In SIGGRAPH (Conference Paper Track), pages 11:1–11:11. ACM, 2023. 4
- [28] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2
- [29] He Qiyuan, Jinghao Wang, Ziwei Liu, and Angela Yao. Aid: Attention interpolation of text-to-image diffusion. Advances in Neural Information Processing Systems, 2024. 3, 12
- [30] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022. 2
- [31] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 10674–

10685. IEEE, 2022. 2, 3

- [32] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022. 2
- [33] Dvir Samuel, Rami Ben-Ari, Nir Darshan, Haggai Maron, and Gal Chechik. Norm-guided latent space exploration for text-to-image generation. Advances in Neural Information Processing Systems, 36, 2024. 3
- [34] Axel Sauer, Tero Karras, Samuli Laine, Andreas Geiger, and Timo Aila. Stylegan-t: Unlocking the power of gans for fast large-scale text-to-image synthesis. In International conference on machine learning, pages 30105–30118. PMLR, 2023. 2
- [35] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 3
- [36] Yujun Shi, Chuhui Xue, Jiachun Pan, Wenqing Zhang, Vincent YF Tan, and Song Bai. Dragdiffusion: Harnessing diffusion models for interactive point-based image editing. In CVPR, 2024. 4
- [37] Douglas B Smythe. A two-pass mesh warping algorithm for object transformation and image interpolation. Rapport technique, 1030:31, 1990. 2, 3
- [38] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021. 2, 3
- [39] Stability.AI. Stable diffusion. https://stability. ai/blog/stable-diffusion-public-release,

2022. 2, 4, 27

- [40] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

- 2

[41] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-toimage translation. In CVPR, pages 1921–1930. IEEE, 2023.

- 3, 4

- [42] Bram Wallace, Akash Gokul, and Nikhil Naik. Edict: Exact diffusion inversion via coupled transformations. In CVPR, pages 22532–22541, 2023. 3
- [43] Clinton J Wang and Polina Golland. Interpolating between images with diffusion models. arXiv preprint arXiv:2307.12560,

2023. 2, 3, 6, 14

- [44] George Wolberg. Digital image warping. IEEE computer society press Los Alamitos, CA, 1990. 2, 3
- [45] George Wolberg. Recent advances in image morphing. Proceedings of CG International’96, pages 64–71, 1996. 2
- [46] George Wolberg. Image morphing: a survey. The visual computer, 14(8-9):360–372, 1998. 2
- [47] Zhaoyuan Yang, Zhengyang Yu, Zhiwei Xu, Jaskirat Singh, Jing Zhang, Dylan Campbell, Peter Tu, and Richard Hartley. Impus: Image morphing with perceptually-uniform sampling using diffusion models. In The Twelfth International Conference on Learning Representations, 2023. 2, 3, 4, 6, 7

- [48] Yan Zeng, Guoqiang Wei, Jiani Zheng, Jiaxin Zou, Yang Wei, Yuchen Zhang, and Hang Li. Make pixels dance: Highdynamic video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8850–8860, 2024. 13
- [49] Kaiwen Zhang, Yifan Zhou, Xudong Xu, Bo Dai, and Xingang Pan. Diffmorpher: Unleashing the capability of diffusion models for image morphing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

2024. 2, 3, 4, 6, 7

- [50] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 6
- [51] Bhushan Zope and Soniya B Zope. A survey of morphing techniques. International Journal of Advanced Engineering, Management and Science, 3(2):239773, 2017. 2

### A. Further Analysis

#### A.1. Usage of the Fast Fourier Transform (FFT)

In our approach, we employ the fast Fourier transform (FFT) to inject high-frequency Gaussian noise, which enhances flexibility. An alternative and straightforward variation involves replacing the FFT with the discrete cosine transform (DCT). To investigate this, we conducted experiments using both FFT and DCT, presenting the results in Fig. 9. The findings indicate that DCT performs comparably to FFT.

Input source Input target

[Figure 62]

[Figure 63]

[Figure 64]

Ours w/ DCT

[Figure 65]

Ours

Generated transitions

Input source Input target

[Figure 66]

[Figure 67]

[Figure 68]

Ours w/ DCT

[Figure 69]

Ours

Generated transitions

###### Figure 9. Analysis of the usage of Fast Fourier Transform (FFT) over Discrete Cosine Transform (DCT).

### B. Qualitative Comparisons

#### B.1. Qualitative Comparisons with AID [29] and Smooth Diffusion [12]

In addition to the comparisons discussed in the main paper, we extend our evaluation to include AID [29] and Smooth Diffusion [12]. As illustrated in Fig. 10 and Fig. 11, the results demonstrate that both methods are limited to processing images with similar layouts and semantics, rendering them ineffective for inputs with different layouts or semantics. Beyond their qualitative shortcomings, it is worth noting that (1) AID relies on IP-Adapter for image morphing, which adversely affects training efficiency, and (2) Smooth Diffusion requires parameter tuning, making it slower and less efficient than our approach.

Input source Input target Input source Input target

[Figure 70]

[Figure 71]

AID

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Ours

Generated transitions Generated transitions

Figure 10. Qualitative comparisons with AID [29].

Input source Input target

[Figure 82]

AID

Smooth Diffusion

Ours

[Figure 83]

[Figure 84]

[Figure 85]

Ours

Generated transitions

Input source Input target

[Figure 86]

Smooth Diffusion

[Figure 87]

[Figure 88]

[Figure 89]

Ours

Generated transitions

###### Figure 11. Qualitative comparisons with Smooth Diffusion [12]

#### B.2. Comparison with video generative models

Given the rapid development of video generative techniques. Methods like PixelDance [48] and SEINE [7] have been designed to achieve image morphing. We hereby provide more comparisons with these video generative models to demonstrate our performance. Considering PixelDance hasn’t released code or an online demo, we ran FreeMorph on the examples from their webpages to perform qualitative comparisons (see Fig. 12 below). Surprisingly, our method performs similarly with PixelDance and outperforms SEINE in reducing ghost artifacts.

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

Ours

Ours

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

SEINE

PixelDance

Comparison with PixelDance

Comparison with SEINE

Figure 12. Comparisons with video generative models.

#### B.3. Comparison with GAN-based morphing methods

We further compare our method with the early GAN-based morphing method (Neural Crossbreed) to demonstrate the performance. The results, presented in Fig. 13, show superior image quality, identity preservation, and smoother transitions. Unlike GAN-based approaches, ours is training-free, is able to handle out-of-domain inputs, and remains robust to varying layouts and semantics. Additional evaluations and discussions will be included in the revised version.

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

Ours

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

Neural

Crossbreed

###### Figure 13. Comparison with GAN-based morphing methods.

#### B.4. Comparison with Wang and Golland [43]

We further compare with Wang and Golland [39] and present the results in Fig. 14. We can clearly observe that our method consistently show superior performance over it, both qualitatively and quantitatively.

[Figure 126]

Wang and Golland [39]

[Figure 127]

[Figure 128]

Ours

[Figure 129]

Figure 14. Comparison with Wang and Golland [43].

##### B.5. Experiments with different poses/actions We further present results for various poses and actions below (Fig. 15), using input images from the MorphBench dataset.

[Figure 130]

Figure 15. Qualitative results with different poses/actions.

#### B.6. Additional Qualitative Comparisons

We provide additional qualitative comparisons with three methods in Fig. 16–Fig. 23. These results reinforce the conclusions drawn in Sec. 4.2 of the main paper, offering further evidence of the superior performance of our FreeMorph method in achieving high-fidelity and smooth image morphing.

[Figure 131]

[Figure 132]

- Figure 16. More qualitative comparisons with existing techniques (Part I).

[Figure 133]

[Figure 134]

###### Figure 17. More qualitative comparisons with existing techniques (Part II).

[Figure 135]

[Figure 136]

###### Figure 18. More qualitative comparisons with existing techniques (Part III).

[Figure 137]

[Figure 138]

###### Figure 19. More qualitative comparisons with existing techniques (Part IV).

[Figure 139]

[Figure 140]

###### Figure 20. More qualitative comparisons with existing techniques (Part V).

[Figure 141]

[Figure 142]

###### Figure 21. More qualitative comparisons with existing techniques (Part VI).

[Figure 143]

[Figure 144]

###### Figure 22. More qualitative comparisons with existing techniques (Part VII).

[Figure 145]

[Figure 146]

###### Figure 23. More qualitative comparisons with existing techniques (Part VIII).

### C. More Qualitative Results

To provide a better understanding of the intermediate generated transitions, in addition to the animated videos, we also present generated images in Fig. 24–Fig. 27, which correspond to the animated videos in the HTML file.

[Figure 147]

[Figure 148]

- Figure 24. Images with different semantics and different layouts.

[Figure 149]

[Figure 150]

###### Figure 25. Images with similar semantics and similar layouts.

[Figure 151]

###### Figure 26. Images with different semantics and similar layouts.

[Figure 152]

###### Figure 27. Images with similar semantics and different layouts.

### D. Visualization of Morph4Data

We present a range of visualizations from our collected Morph4Data to enhance understanding of the dataset and the distinctions among its different classes.

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

- Class-a
- Class-b
- Class-c
- Class-d

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

Figure 28. Examples of 4 classes in Morph4Data.

### E. Applications

We highlight that our FreeMorph method can be adapted for image editing tasks. Specifically, this is accomplished by (1) using the same image as both the "input source" and "input target," and (2) employing different text prompts, where the first prompt describes the original image and subsequent prompts indicate the desired editing direction. An example is provided in Fig. 29. Notably, our method produces image editing results that align correctly with the text prompts, preserving the original identity while effectively generating smooth transitions throughout the editing process.

[Figure 193]

a photo of a deer standing in front of mountain

a photo of a mountain

###### Figure 29. Application of FreeMorph in image editing

edition

### F. Limitations and Failure Cases

While our method establishes a new state-of-the-art, we acknowledge that it has certain limitations. We illustrate several failure cases in Fig. 30. Specifically: 1) Although our model can achieve reasonable results when processing images with no semantic or layout similarity, the generated transitions may not be smooth, potentially leading to abrupt changes. 2) Our method inherits biases from Stable Diffusion [39], resulting in difficulties in accurately transitioning images that model human limbs.

Input source

Input target

Generated transitions

[Figure 194]

[Figure 195]

Figure 30. Failure cases.

### G. Societal Impact

Our research advances the image morphing task across a range of semantics and layouts, establishing a more versatile pipeline. However, there is a risk of misuse, such as brands creating misleading advertisements that distort consumer perceptions and create unrealistic product expectations. This practice not only undermines consumer trust but also raises significant ethical concerns about the authenticity of marketing. Additionally, the complexities of copyright and consent are amplified, as manipulated images blur the lines of ownership and accountability. Therefore, we advocate for strict legal compliance and usage restrictions to regulate the application of image morphing techniques and derivative models.

