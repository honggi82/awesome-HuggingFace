# arXiv:2411.14793v3[cs.CV]20Mar2025

## Style-Friendly SNR Sampler for Style-Driven Generation

Jooyoung Choi1,∗ Chaehun Shin1,∗ Yeongtak Oh1 Heeseung Kim1 Jungbeom Lee2 Sungroh Yoon1,3,† 1Data Science and AI Laboratory, ECE, Seoul National University 2Amazon 3AIIS, ASRI, INMC, ISRC, and Interdisciplinary Program in AI, Seoul National University

{jy choi,chaehuny,dualism9306,gmltmd789}@snu.ac.kr,jungbeol@amazon.com,sryoon@snu.ac.kr https://stylefriendly.github.io

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

Figure 1. Fine-tuning text-to-image diffusion models on the style-friendly noise levels enables learning novel styles from reference images and text prompts. We present ‘A kangaroo holding a beer, wearing ski goggles and passionately singing silly songs’ in various styles including watercolor painting, flat illustration, and 3d rendering styles. References are shown in the red insert box.

### Abstract

Recent text-to-image diffusion models generate high-quality images but struggle to learn new, personalized styles, which

† Correspondence to: Sungroh Yoon (sryoon@snu.ac.kr) ∗ Both authors contributed equally to this work

limits the creation of unique style templates. In style-driven generation, users typically supply reference images exemplifying the desired style, together with text prompts that specify desired stylistic attributes. Previous approaches popularly rely on fine-tuning, yet it often blindly utilizes objectives and noise level distributions from pre-training without adaptation. We discover that stylistic features predom-

inantly emerge at higher noise levels, leading current finetuning methods to exhibit suboptimal style alignment. We propose the Style-friendly SNR sampler, which aggressively shifts the signal-to-noise ratio (SNR) distribution toward higher noise levels during fine-tuning to focus on noise levels where stylistic features emerge. This enhances models’ ability to capture novel styles indicated by reference images and text prompts. We demonstrate improved generation of novel styles that cannot be adequately described solely with a text prompt, enabling the creation of new style templates for personalized content creation.

### 1. Introduction

Recently, large-scale text-to-image diffusion models [2, 9, 31, 40, 41, 44, 49] have achieved remarkable progress in visual content creation. In particular, open-weights such as Stable Diffusion series [9, 44] and FLUX [31] have been among the most notable for their photorealistic image quality and language understanding capabilities. Behind this strong performance lies the advancement of the diffusion framework that encompasses the principles of score-based models [57] and flow matching [35, 36], diffusion formulations [18, 35, 36, 57], loss weighting [5, 26], noise level scheduling [19, 24], and architectural improvements [9, 25, 39]. These advancements have predominantly focused on generating high-quality images with respect to object-centric benchmarks [12, 22] and metrics [16, 30].

Motivated by the success of text-to-image models, there is a growing need for style-driven generation [15, 32, 46, 53, 62], where the generated samples capture styles desired by individual users or artists. Here, “style” encompasses various elements such as color schemes, layouts, illumination, and brushwork [8, 11, 33, 53], all contributing to the unique nuances of an image, and is typically specified or clarified using text prompts. However, relying solely on prompt engineering has its limitations in reflecting unique styles, especially those not present in the pre-training data.

To achieve more effective style-driven generation, previous research has explored methods such as fine-tuning [32, 47, 53], image variation [62] and editing [15]. These approaches commonly use style reference images and their accompanying text prompts to guide the generation process. However, they blindly apply the same objective functions and noise level distributions used for pretraining—originally optimized for object-centric benchmarks [12, 22]—without acknowledging that style images emphasize different visual factors such as color schemes, layout, and brushstrokes, rather than simply depicting specific objects. Consequently, even with numerous styledriven generation studies [15, 32, 46, 53, 62], we observe prevalent failure cases when fine-tuning diffusion models on style references. Addressing these failures sometimes re-

Probability Distribution of logSNR

0.40

SD3 and Flux

= 6 and = 1

0.35

- Style friendly ( = 2)

- Style friendly ( = 3)

0.30

ProbabilityDensity

0.25

0.20

0.15

0.10

0.05

0.00

12.5 10.0 7.5 5.0 2.5 0.0 2.5 5.0 7.5 logSNR

Figure 2. Probability distribution of Log-SNR. We bias the distribution towards the shaded region where style features emerge.

quires two-stage training with heavy human feedback [53].

In this paper, we address these limitations by introducing the Style-friendly SNR sampler, a method that ensures capturing novel styles during fine-tuning. Our approach is motivated by two key observations: 1) diffusion models [9, 31] struggle to learn new styles, and 2) styles emerge at higher noise levels. These observations reveal that standard finetuning procedures often devote a large portion of computation toward lower noise levels, which are less relevant for learning novel styles. Building upon these observations, we propose reallocating fine-tuning computation on higher noise levels, where essential style attributes emerge.

Our method, focusing on the reallocation of noise level sampling during fine-tuning, enhances the fidelity of styledriven generation, capturing the novel styles indicated by text prompts while preserving the capability of depicting the desired content. Moreover, we unveil the key components that explain why these models excel at learning objectcentric concepts but struggle with styles, providing deeper insights into the diffusion process for style-driven generation. Ultimately, our approach facilitates the creation of style templates from reference images, which can be easily shared and utilized to practitioners for content creation, expanding the capabilities of diffusion models.

### 2. Training Diffusion Models

#### 2.1. Diffusion Process and SNR Formulation

Various diffusion models [18, 35, 36, 52, 57] are based on the forward process that progressively degrades data x0 into pure noise x1 as time t progresses from 0 to 1, following the unified formulation below:

##### xt = αtx0 + σtϵ, (1)

where αt and σt are predefined noise schedules, and ϵ ∼ N(0,I) represents standard Gaussian noise.

Recent state-of-the-art flow matching frameworks, such as Stable Diffusion 3 (SD3) [9] and FLUX [31], utilize the noise schedule from rectified flow [35, 36], where αt = 1−t and σt = t, with t varying continuously in the range [0,1]. This choice is effective due to straight diffusion trajectories.

Instead of parameterizing the diffusion process using the timestep t, Kingma et al. [26, 27] characterize the noise level using the log signal-to-noise ratio (log-SNR), which offers a more intuitive measure of the noise at each step:

αt2 σt2

. (2)

λt = log

In flow matching framework, the log-SNR is simplified as λt = 2log 1−t t using timestep t.

- 2.2. Diffusion Training and SNR Sampling

Recent flow matching frameworks predict the velocity field vθ(xt,t) by minimizing the following training objective:

LFM(x0) = Et∼p(t) ϵ − x0 − vθ(xt,t) 2 , (3)

where p(t) is the timestep sampling distribution, ϵ − x0 is the target velocity derived from the forward process in Eq. (1). The representative text-to-image models utilizing flow matching formulation, such as SD3 [9] and FLUX [31], introduce a SNR sampler for training. This

sampler samples the logit of t, defined as log 1− t t from N(µ,σ2), where the parameters µ and σ are chosen as 0 and 1 to optimize CLIP [42] and FID scores [16] on COCO2014 validation set [34].

In addition, they propose shifting timestep t to tnew by k for high resolution training:

tnew =

kt 1 + (k − 1)t

, (4)

which is equivalent to shifting λt by −2log k as follows:

λt

new

= 2log

1 − tnew tnew

= λt − 2log k, (5)

where k is defined as 3. Following the above formulation, resulting log-SNR sampling distribution p(λt) in training time is represented as N(−2log 3,22), as visualized by the yellow curve in Fig. 2. This curve demonstrates that pretraining SD3 and FLUX focus on particular noise levels.

- 3. Method

- 3.1. Observations

Diffusion Models Struggle to Capture Styles. We begin by examining the fine-tuning capabilities of current text-toimage diffusion models, which have primarily been studied for object-driven generation [47] from reference images. Fine-tuning is typically guided by accompanying text

|[Figure 31]<br><br>[Figure 32]| |
|---|---|
| | |

|[Figure 33]|[Figure 34]|
|---|---|
| | |

|[Figure 35]|[Figure 36]|
|---|---|
| | |

|[Figure 37]|[Figure 38]|
|---|---|
| | |

|[Figure 39]<br><br>[Figure 40]| |
|---|---|
| | |

|[Figure 41]|[Figure 42]|
|---|---|
| | |

(a) FLUX Object (b) FLUX Style (c) Our Style

Figure 3. Fine-tuning capability. (a) While FLUX succeeds in learning objects, (b) it struggles to capture styles, demonstrating that learning novel objects and styles requires distinct strategies. (c) We enable FLUX to learn styles. References are shown in the red insert box.

prompts that specify the desired objects or styles. As shown in Fig. 3a, these methods often excel at producing highquality object-driven images. However, we observe poorer results when they are applied to replicate distinctive color schemes, illumination, or brushwork—particularly when using the same SNR sampler from pre-training [9]. In Fig. 3b, when provided with a “glowing” style reference, the FLUX model only applies glowing effects to specific object details (such as the fur of a sloth), neglecting broader stylistic elements like the dark background and blue lighting of the original reference. Similarly, with a Van Gogh oil painting style reference, the model manages to replicate the blue color tone but fails to accurately reproduce distinctive brushstroke characteristics.

Styles Emerge at Higher Noise Levels. To better understand why diffusion models struggle to capture new styles, we investigate at which noise levels stylistic features emerge during generation using a pre-trained FLUX model [31]. Specifically, we switch from a prompt without style descriptions (yw/o style) to one including style descriptions (yw/ style) at different points in the denoising process. Omitting the style prompt in the initial 10% of generation steps significantly reduces style alignment (Fig. 4c),

- as quantitatively confirmed by the low CLIP similarity scores when initial denoising steps omit style information (Fig. 4g). Conversely, omitting the style prompt at later denoising steps minimally affects style alignment (Fig. 4d,e), which is also supported quantitatively in Fig. 4h. These results demonstrate that styles are predominantly determined
- at early denoising steps, corresponding to intervals with higher noise levels (low log-SNR λt values).

|𝑦 /  : a raccoon wearing an astronaut helmet|
|---|

|𝑦 / : a raccoon wearing an astronaut helmet in {style prompt}|
|---|

Denoising step Denoising step Denoising step

|100%|
|---|

|3%|97%|
|---|---|

|10%|90%|
|---|---|

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

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

(b) Style prompt in 𝜆 ∈ (−8.89,∞)

(c) Style prompt in 𝜆 ∈ (−6.52,∞)

(a) Style prompt everywhere

- (g) Omitting style prompt in earlier steps
- (h) Omitting style prompt in later steps

Denoising step Denoising step Denoising step

|[Figure 58]<br><br>10%|90%|
|---|---|

|3%|97%|
|---|---|

|100%|
|---|

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

(d) Style prompt in 𝜆 ∈ (−∞,−6.52)

(e) Style prompt in 𝜆 ∈ (−∞,−8.89)

(f) Style prompt nowhere

- Figure 4. Prompt switching during generation. λt indicates log-SNR. The bar graphs above each image represent the denoising steps, illustrating when each prompt is applied and at what point the prompt switch occurs. The style prompts are ‘minimalist flat round logo’, ‘sticker’, ‘detailed pen and ink drawing’, and ‘cartoon’. Styles emerge in the initial 10% of denoising steps; therefore, (c) and (f) fail to capture target styles. In contrast, omitting style prompts in later steps (d,e) still preserves styles well, similar to the fully styled baseline (a).

- (g) and (h) quantify these observations, showing the average CLIP similarity across 5 prompts and 5 styles when omitting (g) or including
- (h) the style prompt in earlier steps. Here, we use FLUX with 28 inference steps.

#### 3.2. Style-Friendly SNR Sampler

Our earlier observations show that styles primarily emerge during early denoising steps, characterized by higher noise levels (lower log-SNR values). However, existing finetuning methods use the SNR sampler from pre-training, optimized mainly for object-centric benchmarks [12, 22], as indicated by the yellow curve in Fig. 2. Consequently, standard fine-tuning procedures place insufficient emphasis on noise levels crucial for capturing styles, failing to achieve alignment with reference styles.

Building upon this motivation, we propose to fine-tune diffusion models by biasing the noise level distribution towards higher noise levels (lower log-SNR λt values) where stylistic features emerge. Specifically, we sample log-SNR from a normal distribution:

λt ∼ N(µlow,σlarge2 ), (6)

with a lowered mean µlow, thereby focusing the training on higher noise levels essential for style learning. In addition, we choose sufficiently large σlarge to cover the wide range of

log-SNR values critical for style learning (shaded region in Fig. 2). Finally, to map the sampled λt back to the timestep domain, we compute t = 1/(1 + exp(λt/2)).

While timestep shifting in [9] weakly biases the noise level distribution as in Eq. (5), our Style-friendly SNR sampler aggressively targets the high-noise regions critical for capturing style. Setting the mean to µlow = −6 and σlarge ≥ 2 targets the shaded region of Fig. 2, substantially improving style alignment across a variety of reference styles.

#### 3.3. Trainable Parameters of MM-DiT

We fine-tune both FLUX-dev [31] and SD3.5 [9, 59] by training LoRA [20] adapters on specific layers to capture new styles. Multi-Modal Diffusion Transformer (MMDiT) [9], the core architecture for both FLUX-dev and SD3.5 comprises dual-stream transformer blocks with separate parameters for text tokens and image tokens, which interact through joint attention mechanisms. To effectively learn the stylistic features encompassing both visual and linguistic characteristics, we train LoRA on the attention lay-

ers of both modalities. Additionally, FLUX includes singlestream blocks [6] that handle both modalities simultaneously with attention mechanisms and projection layers that skip this attention, to which we also apply LoRA. This targeted fine-tuning achieves high style-alignment without training the entire network, providing a parameter-efficient method for fine-tuning MM-DiT.

### 4. Experiments

We compare our method against both fine-tuning and nonfine-tuning approaches for style-driven generation.

- • Fine-tuning: SD3 sampler [9], Direct Consistency Optimization (DCO) [32], and StyleDrop [53] are selected as representative fine-tuning methods. The SD3 sampler utilizes a flow matching loss [35, 36] with timestep shifting in Eq. (5), while DCO is based on reinforcement learning [43] and employs a regularized loss. For StyleDrop, due to the lack of official open-source implementations, we adopt the unofficial implementation provided by [1].
- • Image variation: IP-Adapter [62] and RB-Modulation

- [46] generate image variations using CLIP [42] or CSD [54] image embeddings respectively. IP-Adapter reconstructs images to best reflect the information contained in the CLIP image embedding, whereas RBModulation applies gradient guidance at test time to maximize the similarity score evaluated by CSD.

- • Editing: Style-Aligned [15] is an editing-based method that shares and manually uplifts self-attention weights to enforce the reference style consistency.
- • Detailed prompt: We name this baseline GPT-4o Prompt, where we utilize GPT-4o [23] to generate detailed style descriptions from reference images and apply these in text-to-image generation without fine-tuning.

Following StyleDrop [53], we select 18 reference styles as

- our fine-tuning targets. For each style, we use 23 evaluation prompts, generating 2 images per prompt, which results in a total of 828 images per experiment. We fine-tune FLUX-dev [31] and SD3.5-8B [9, 59] with LoRA [20] at rank 32 (except where noted for rank ablation). Following [15, 53] we measure style alignment via human evaluation, DINO ViTS/16 [3], and CLIP ViT-B/32 [42] image similarity (CLIPI), and alignment to the text prompts via CLIP text-image similarity (CLIP-T). Additional implementation details are in the Appendix.

#### 4.1. Analysis of Style-Friendly SNR Sampler

In Fig. 5 and Fig. 6, we conduct experiments to analyze the impact of varying the parameters of our Style-friendly SNR sampler—specifically, the mean (µ) and the standard deviation (σ) of the log-SNR sampling distribution, as well as the LoRA rank.

Effect of Varying µ. We experiment with µ values ranging from 0 to −8 for both FLUX-dev and SD3.5-8B. As shown in Fig. 5, increasing µ progressively impairs the model’s ability to capture reference styles. This trend is quantitatively confirmed in Fig. 6a, where DINO similarity scores decrease with increasing µ (towards zero), clearly indicating poorer style alignment. Conversely, in Fig. 5, when µ is set to −6, the models begin to capture and reflect the reference styles effectively.

Effect of Varying σ. We also investigate the effect of varying the standard deviation σ of the log-SNR sampling distribution in Fig. 5 and Fig. 6b. When σ is less than 2, the model does not sufficiently cover the style-emerging region as shown by green curve of Fig. 2, leading to lower style alignment due to limited exposure to critical noise levels. In Fig. 5, σ ≥ 2 reflects the ‘glowing’ style. Also, in Fig. 6b, style similarity remains low for σ < 2 but improves when σ ≥ 2. For FLUX, DINO score is highest at σ = 3, whereas for SD3.5, σ = 2 yields the best results. Based on these findings, we adopt these hyperparameters for each respective model.

Effect of Varying Rank. In Fig. 6c, we examine the impact of model capacity by varying the LoRA rank. Notably, with low µ = −6, a rank of 4 achieves higher DINO similarity compared to the SD3 sampler at rank 32 (dotted lines). This demonstrates focusing on higher noise levels (lower λt) has a more pronounced effect on style learning than model capacity alone.

#### 4.2. Qualitative Results

We compare our Style-friendly SNR sampler against previous methods, including the SD3 sampler [9], Direct Consistency Optimization (DCO) [32], and IP-Adapter [61, 62], which use FLUX-dev as the backbone model; RBModulation [46], which uses Stable Cascade [40]; and Style-Aligned [15], which uses SDXL [41].

In Fig. 7, our Style-friendly SNR sampler accurately captures the styles of reference images, reflecting stylistic features including color schemes, layouts, illumination, and brushstrokes. In contrast, fine-tuning FLUX-dev with the standard SD3 sampler often fails to capture key stylistic components, such as layout (column 1), color scheme (columns 2-4), and illumination (column 5).

Fine-tuning FLUX-dev with DCO struggles to learn the reference styles due to strong regularization that prevents significant deviation from the pre-trained model. As seen in column 4 of the IP-Adapter results, woman in the reference appears, indicating content leakage. IP-Adapter with FLUX-dev and RB-Modulation rely on embeddings of CLIP [42] and CSD [54], which may not capture fine stylistic details, leading to less accurate style reproduction. Style-

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Reference 𝜇 = −6,𝜎 = 3 𝜇 = −6,𝜎 = 2 𝜇 = −6,𝜎 = 1 𝜇 = −4,𝜎 = 2 SD3 sampler

- Figure 5. Effect of varying µ and σ. Diffusion models start to capture the reference glowing style when µ is lower and σ is larger. The prompt is ‘a hybrid creature that is a mix of a waffle and a hippopotamus, in glowing style’. Samples are generated with the same seed.

-8.0 -6.0 -4.0 SD3 -2.0 0.0 Value

0.30

0.35

0.40

0.45

0.50

0.55

DINOImageSimilarity

Log-SNR Mean Ablation (DINO) FLUX

| |
|---|

SD3.5

(a) Varying µ.

0.4 1.0 2.0 3.0 4.0 Value

0.30

0.35

0.40

0.45

0.50

0.55

DINOSimilarity

Log-SNR Standard Deviation Ablation (DINO) FLUX

| |
|---|

SD3.5

(b) Varying σ.

4 8 32 Rank Value

0.30

0.35

0.40

0.45

0.50

0.55

DINOSimilarity

Rank Ablation (DINO) FLUX SD3.5

| |
|---|

FLUX+SD3 sampler

SD3.5+SD3 sampler

(c) Varying LoRA Rank.

- Figure 6. SNR sampler analysis. DINO similarities of varying SNR sampler parameters with FLUX and SD3.5-8B. Dotted lines in (c) indicate results of SD3 sampler [9]. Unless specified, we use µ = −6, σ = 2, and rank 32. CLIP scores are shown in the Appendix.

Aligned shares self-attention features within the diffusion model, which can cause artifacts such as destroyed structure (columns 1-3) when attention features conflict. While GPT4o prompt utilizes the detailed style description, it often fails to reflect key stylistic features such as color scheme, highlighting the necessity of image guidance for effective style-driven generation.

#### 4.3. Quantitative Results

We further conduct a user study to quantify human preferences using Amazon Mechanical Turk. Following previ-

- ous work [53], we compare our method to each method with two separate questionnaires. According to the reference style image and target text prompt, users are asked to select which of the two generated images is more similar to the style in the reference image and represents the target text prompt better. We obtain 450 answers from 150 participants for each comparison, and the results are presented in Tab. 1. Our method outperforms prior arts in both aspects (p < 0.05 in the Wilcoxon signed-rank test), consistent with the qualitative results and demonstrates the superiority of our method in learning stylistic elements. More details on

- our user study are provided in the Appendix. In Tab. 2, we evaluate our method and prior works us-

ing DINO [3] and CLIP image similarities (CLIP-I) to assess style alignment, and CLIP text-image similarity (CLIPT) for text alignment. Our method achieves the highest

Style Alignment Method Model win tie lose

Style-Aligned [15] SDXL 61.0 % 7.1% 31.9% RB-Mod [46] Cascade 55.6 % 12.6% 31.8% IP-Adapter [62] FLUX-dev 59.2 % 8.0% 32.8% DCO [32] FLUX-dev 56.0 % 10.2% 33.8% SD3 sampler [9] FLUX-dev 56.0 % 9.2% 34.8%

Text Alignment Method Model win tie lose

Style-Aligned [15] SDXL 60.7% 7.5% 31.8% RB-Mod [46] Cascade 54.3% 6.3% 39.4% IP-Adapter [62] FLUX-dev 56.0% 4.6% 39.4% DCO [32] FLUX-dev 53.2% 10.0% 36.8% SD3 sampler [9] FLUX-dev 56.5% 14.0% 29.5%

Table 1. Human evaluation. User preference results comparing style and text alignments between our method and the baselines.

DINO and CLIP-I scores, demonstrating its superior ability to capture styles from the reference images. Notably, our approach improves DCO, indicating that the key factor in style-driven fine-tuning is not whether the method relies on diffusion loss or reinforcement learning-based loss, but rather which noise levels are emphasized during training.

Flat cartoon illustration

Cartoon line drawing

Watercolor painting

3d rendering Glowing

|[Figure 80]|
|---|

|[Figure 81]|
|---|

|[Figure 82]|
|---|

|[Figure 83]|
|---|

|[Figure 84]|
|---|

ReferenceStyle-friendly

|[Figure 85]|[Figure 86]|
|---|---|
| |[Figure 87]|

|[Figure 88]|[Figure 89]|
|---|---|
| |[Figure 90]|

|[Figure 91]|[Figure 92]|
|---|---|
| |[Figure 93]|

|[Figure 94]|[Figure 95]|
|---|---|
| |[Figure 96]|

|[Figure 97]|[Figure 98]|
|---|---|
| |[Figure 99]|

|[Figure 100]|[Figure 101]|
|---|---|
| |[Figure 102]|

|[Figure 103]|[Figure 104]|
|---|---|
| |[Figure 105]|

|[Figure 106]|[Figure 107]|
|---|---|
| |[Figure 108]|

|[Figure 109]|[Figure 110]|
|---|---|
| |[Figure 111]|

|[Figure 112]|[Figure 113]|
|---|---|
| |[Figure 114]|

samplerDCOSD3

|[Figure 115]|[Figure 116]|
|---|---|
| |[Figure 117]|

|[Figure 118]|[Figure 119]|
|---|---|
| |[Figure 120]|

|[Figure 121]|[Figure 122]|
|---|---|
| |[Figure 123]|

|[Figure 124]|[Figure 125]|
|---|---|
| |[Figure 126]|

|[Figure 127]|[Figure 128]|
|---|---|
| |[Figure 129]|

|[Figure 130]|[Figure 131]|
|---|---|
| |[Figure 132]|

|[Figure 133]|[Figure 134]|
|---|---|
| |[Figure 135]|

|[Figure 136]|[Figure 137]|
|---|---|
| |[Figure 138]|

|[Figure 139]|[Figure 140]|
|---|---|
| |[Figure 141]|

|[Figure 142]|[Figure 143]|
|---|---|
| |[Figure 144]|

RB-ModulationStyle-AlignedIP-AdapterGPT-4oprompt

|[Figure 145]|[Figure 146]|
|---|---|
| |[Figure 147]|

|[Figure 148]|[Figure 149]|
|---|---|
| |[Figure 150]|

|[Figure 151]|[Figure 152]|
|---|---|
| |[Figure 153]|

|[Figure 154]|[Figure 155]|
|---|---|
| |[Figure 156]|

|[Figure 157]|[Figure 158]|
|---|---|
| |[Figure 159]|

|[Figure 160]|[Figure 161]|
|---|---|
| |[Figure 162]|

|[Figure 163]|[Figure 164]|
|---|---|
| |[Figure 165]|

|[Figure 166]|[Figure 167]|
|---|---|
| |[Figure 168]|

|[Figure 169]|[Figure 170]|
|---|---|
| |[Figure 171]|

|[Figure 172]|[Figure 173]|
|---|---|
| |[Figure 174]|

|[Figure 175]|[Figure 176]|
|---|---|
| |[Figure 177]|

|[Figure 178]|[Figure 179]|
|---|---|
| |[Figure 180]|

|[Figure 181]|[Figure 182]|
|---|---|
| |[Figure 183]|

|[Figure 184]|[Figure 185]|
|---|---|
| |[Figure 186]|

|[Figure 187]|[Figure 188]|
|---|---|
| |[Figure 189]|

- Figure 7. Qualitative comparison. We show ‘A fluffy baby sloth with a knitted hat trying to figure out a laptop’, ‘A christmas tree’, and ‘An espresso machine’ in various styles. Target style prompts are shown above. Our Style-friendly SNR sampler effectively captures the styles indicated by both reference images and their accompanying text prompts. Fine-tuning methods (SD3 sampler and DCO) often miss stylistic nuances due to insufficient focus on the relevant noise levels or overly strong regularization.

|Method<br><br>|Model|Metrics|
|---|---|---|
| | |DINO ↑ CLIP-I ↑ CLIP-T ↑|
|Style-Aligned [15] RB-Mod [46] IP-Adapter [62] GPT-4o Prompt<br><br>|SDXL Cascade FLUX FLUX|0.410 0.675 0.340 0.317 0.647 0.363 0.361 0.656 0.354 0.299 0.621 0.338<br><br>|
|StyleDrop [53]†<br><br>|MUSE|0.465 0.665 0.325<br><br>|
|SD3 sampler [9]<br><br>+Style-friendly DCO [32]<br><br>+Style-friendly<br><br>|SD3.5 SD3.5 SD3.5 SD3.5<br><br>|0.424 0.670 0.350 0.489 0.698 0.349 0.399 0.661 0.355 0.478 0.695 0.351<br><br>|
|SD3 sampler [9]<br><br>+Style-friendly DCO [32]<br><br>+Style-friendly<br><br>|FLUX FLUX FLUX FLUX<br><br>|0.373 0.645 0.350 0.478 0.691 0.343 0.373 0.643 0.353 0.488 0.698 0.341<br><br>|

Table 2. Quantitative comparison. Style alignment (DINO and CLIP-I) and text alignment (CLIP-T) with 18 styles from [53]. Our style-friendly exhibits superior style-alignment scores. Rows 1-3 show non-fine-tuning baselines. †: Results obtained using an unofficial implementation [1].

While our CLIP-T is slightly lower compared to some methods, we already showed superior text alignment in human evaluation (Tab. 1). This discrepancy arises because textual style descriptions alone can be inherently ambigu-

- ous and often correspond to common interpretations. Consequently, methods that accurately reflect unique styles may deviate from these typical interpretations, leading to lower CLIP-T scores despite higher alignment to the intended reference styles. Overall, our quantitative results confirm that our method accurately reflects both styles and texts.

- 4.4. Applications

While Dreambooth [47] paper demonstrates generating multi-panel comics by generating each panel with a finetuned diffusion model, we define the entire multi-panel comic itself as a unique style. Our method treats multiple panels as a single image during fine-tuning, enabling the generation of coherent multi-panel comics from only a single reference (see the first row of Fig. 8). By specifying a new subject in the target prompt, the model consistently places that subject across all comic-style panels. Beyond comics, our method also extends to typography, leveraging the spelling capabilities of recent models [31, 59]. As shown in the second row of Fig. 8, this flexibility allows users to effortlessly generate a broad range of customized textual elements in unique styles.

- 5. Related Works

- 5.1. Diffusion Models

Diffusion models generate data from noise, encapsulating approaches based on denoising score matching [24, 56, 57], maximum likelihood training [27], and rectified flow [35,

|[Figure 190]|
|---|

|[Figure 191]|
|---|

|[Figure 192]|
|---|

TypographyMulti-Panel

|[Figure 193]|
|---|

|[Figure 194]|
|---|

|[Figure 195]|
|---|

Style References Generated Samples

Figure 8. Multi-panel and typography. First row demonstrates generating multiple coherent panels as a single image. Second row shows customized typography with a unique style.

36]. One of the critical factors influencing the performance of diffusion models is the sampling distribution over noise levels during training, known as the importance sampling of noise levels. Studies focusing on noise schedule adjustment [19, 38] and weight adjustment [5, 24, 26] have succeeded in training high-quality diffusion models by carefully weighting different noise levels. Their effectiveness has been shown with object-centric metrics and benchmarks [12, 16, 22, 30].

#### 5.2. Style-Driven Generation

With advancements in text-to-image models, practitioners have increasingly sought to generate images featuring personal styles [32, 46, 53, 62]. Fine-tuning methods [32, 47, 53] have been particularly prominent in this area. StyleDrop [53], a study closely related to our work, utilizes a masked generative model [4] and involves human data selection through multi-stage training. Some works focus on learning multiple concepts simultaneously [21, 29] or merging several fine-tuned models [32, 51], while others analyze the diffusion model’s U-Net [45] layers to identify those most effective for learning styles [10]. However, several existing methods have not yet been validated or applied to recently released large-scale models [9, 31]. In contrast, our method provides an adaptable fine-tuning strategy, which we validate on up-to-date diffusion models.

As an alternative approach for style-driven generation, zero-shot approaches have been proposed [15, 46, 60, 62], but these methods still fall short in style alignment compared to fine-tuning and are often limited to specific domains [14, 48]. Furthermore, some methods [15, 46] require extra inference-time gradient guidance [46] or inversion [15], increasing inference cost. Due to these limitations, we focus on fine-tuning in our work, aiming to provide insights into the behavior of diffusion objectives to

make fine-tuning more accessible and effective. While finetuning is not the only option, our results show that it is a promising approach.

### 6. Conclusion

In this paper, we observed that stylistic features in diffusion models emerge predominantly at higher noise levels. To address the limitations of previous fine-tuning approaches in capturing new artistic styles, we proposed the Style-friendly SNR sampler, which biases the SNR distribution towards higher noise levels. We showed style-driven generation that reflects reference styles specified via text prompts. We hope this work will serve as a stepping stone toward using diffusion models as digital art previewers.

Acknowledgement This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) [No. 2022R1A3B1077720], Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) [NO.RS-2021-II211343, Artificial Intelligence Graduate School Program (Seoul National University)], and the BK21 FOUR program of the Education and Research Program for Future ICT Pioneers, Seoul National University in 2024.

### References

- [1] aim uofa. Styledrop-pytorch. https://github.com/ aim-uofa/StyleDrop-PyTorch, 2023. 5, 8
- [2] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, et al. ediff-i: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022. 2
- [3] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 5, 6
- [4] Huiwen Chang, Han Zhang, Jarred Barber, Aaron Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Patrick Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. In International Conference on Machine Learning, pages 4055–4075. PMLR, 2023. 8
- [5] Jooyoung Choi, Jungbeom Lee, Chaehun Shin, Sungwon Kim, Hyunwoo Kim, and Sungroh Yoon. Perception prioritized training of diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11472–11481, 2022. 2, 8
- [6] Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Peter Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, et al. Scaling vision transformers to 22 billion pa-

- rameters. In International Conference on Machine Learning, pages 7480–7512. PMLR, 2023. 5
- [7] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In Advances in neural information processing systems, pages 8780–8794, 2021. 13
- [8] Alexei A Efros and William T Freeman. Image quilting for texture synthesis and transfer. In Seminal Graphics Papers: Pushing the Boundaries, Volume 2, pages 571–576. 2023. 2
- [9] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024. 2, 3, 4, 5, 6, 8, 12, 13, 15, 16
- [10] Yarden Frenkel, Yael Vinker, Ariel Shamir, and Daniel Cohen-Or. Implicit style-content separation using b-lora. arXiv preprint arXiv:2403.14572, 2024. 8
- [11] Leon A Gatys, Alexander S Ecker, and Matthias Bethge. Image style transfer using convolutional neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2414–2423, 2016. 2
- [12] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating textto-image alignment. Advances in Neural Information Processing Systems, 36, 2024. 2, 4, 8
- [13] Nicholas Guttenberg. Diffusion with offset noise. https: //www.crosslabs.org/blog/diffusion-withoffset-noise, 2023. 13, 16
- [14] Zecheng He, Bo Sun, Felix Juefei-Xu, Haoyu Ma, Ankit Ramchandani, Vincent Cheung, Siddharth Shah, Anmol Kalia, Harihar Subramanyam, Alireza Zareian, et al. Imagine yourself: Tuning-free personalized image generation. arXiv preprint arXiv:2409.13346, 2024. 8
- [15] Amir Hertz, Andrey Voynov, Shlomi Fruchter, and Daniel Cohen-Or. Style aligned image generation via shared attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4775–4785,

2024. 2, 5, 6, 8, 13, 16

- [16] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 2, 3, 8
- [17] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 12
- [18] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2
- [19] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. In International Conference on Machine Learning, pages 13213–13232. PMLR, 2023. 2, 8
- [20] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. 4, 5, 12

- [21] Hexiang Hu, Kelvin CK Chan, Yu-Chuan Su, Wenhu Chen, Yandong Li, Kihyuk Sohn, Yang Zhao, Xue Ben, Boqing Gong, William Cohen, et al. Instruct-imagen: Image generation with multi-modal instruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4754–4763, 2024. 8
- [22] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. arXiv preprint arXiv: 2307.06350, 2023. 2, 4, 8
- [23] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 5, 14
- [24] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35:26565–26577, 2022. 2, 8
- [25] Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. Analyzing and improving the training dynamics of diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24174–24184, 2024. 2
- [26] Diederik Kingma and Ruiqi Gao. Understanding diffusion objectives as the elbo with simple data augmentation. Advances in Neural Information Processing Systems, 36, 2024. 2, 3, 8
- [27] Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. Advances in neural information processing systems, 34:21696–21707, 2021. 3, 8
- [28] Diederik P Kingma. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 12
- [29] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of textto-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1931–1941, 2023. 8
- [30] Tuomas Kynk¨a¨anniemi, Tero Karras, Miika Aittala, Timo Aila, and Jaakko Lehtinen. The role of imagenet classes in fr´echet inception distance. In The Eleventh International Conference on Learning Representations, 2023. 2, 8
- [31] Black Forest Labs. Flux.1-dev. https : / / huggingface . co / black - forest - labs / FLUX.1-dev, 2024. 2, 3, 4, 5, 8, 12, 16
- [32] Kyungmin Lee, Sangkyung Kwak, Kihyuk Sohn, and Jinwoo Shin. Direct consistency optimization for compositional textto-image personalization. arXiv preprint arXiv:2402.12004,

2024. 2, 5, 6, 8, 13, 16

- [33] Yijun Li, Chen Fang, Jimei Yang, Zhaowen Wang, Xin Lu, and Ming-Hsuan Yang. Universal style transfer via feature transforms. In Advances in neural information processing systems, pages 386–396, 2017. 2
- [34] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13, pages 740–755. Springer, 2014. 3

- [35] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. 2, 3, 5, 8
- [36] Xingchao Liu, Chengyue Gong, et al. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2023. 2, 3, 5, 8
- [37] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14297–14306, 2023. 12
- [38] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International conference on machine learning, pages 8162–8171. PMLR,

2021. 8

- [39] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205,

2023. 2

- [40] Pablo Pernias, Dominic Rampas, Mats L. Richter, Christopher J. Pal, and Marc Aubreville. Wuerstchen: An efficient architecture for large-scale text-to-image diffusion models,

- 2023. 2, 5, 13

[41] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In The Twelfth International Conference on Learning Representations,

- 2024. 2, 5

- [42] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 3, 5, 13
- [43] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024. 5, 13
- [44] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2
- [45] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234–241. Springer, 2015. 8
- [46] Litu Rout, Yujia Chen, Nataniel Ruiz, Abhishek Kumar, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Rb-modulation: Training-free personalization of diffusion models using stochastic optimal control. arXiv preprint arXiv:2405.17401, 2024. 2, 5, 6, 8, 13, 16

- [47] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500– 22510, 2023. 2, 3, 8
- [48] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman. Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6527–6536, 2024. 8
- [49] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 2
- [50] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In European Conference on Computer Vision, pages 87–103. Springer,

2025. 22

- [51] Viraj Shah, Nataniel Ruiz, Forrester Cole, Erika Lu, Svetlana Lazebnik, Yuanzhen Li, and Varun Jampani. Ziplora: Any subject in any style by effectively merging loras. In European Conference on Computer Vision, pages 422–438. Springer, 2025. 8
- [52] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015. 2
- [53] Kihyuk Sohn, Lu Jiang, Jarred Barber, Kimin Lee, Nataniel Ruiz, Dilip Krishnan, Huiwen Chang, Yuanzhen Li, Irfan Essa, Michael Rubinstein, et al. Styledrop: Text-to-image synthesis of any style. Advances in Neural Information Processing Systems, 36, 2024. 2, 5, 6, 8, 12, 13, 23
- [54] Gowthami Somepalli, Anubhav Gupta, Kamal Gupta, Shramay Palta, Micah Goldblum, Jonas Geiping, Abhinav Shrivastava, and Tom Goldstein. Measuring style similarity in diffusion models. arXiv preprint arXiv:2404.01292, 2024. 5, 13
- [55] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021. 13
- [56] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019. 8
- [57] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021. 2, 8
- [58] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In International Conference on Machine Learning, pages 32211–32252. PMLR, 2023. 22
- [59] stabilityai. stable-diffusion-3.5-large. https :

- / / huggingface . co / stabilityai / stable diffusion-3.5-large, 2024. 4, 5, 8, 12, 16
- [60] Haofan Wang, Matteo Spinelli, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen. Instantstyle: Free lunch towards style-preserving in text-to-image generation. arXiv preprint arXiv:2404.02733, 2024. 8
- [61] XLabs-AI. flux-ip-adapter. https://huggingface. co/XLabs-AI/flux-ip-adapter, 2024. 5
- [62] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 2, 5, 6, 8, 13, 16, 22

## Style-Friendly SNR Sampler for Style-Driven Generation Supplementary Material

### A. Experimental Details

#### A.1. Style Prompts

We conduct all quantitative evaluations using the 18 reference styles shown in the appendix of the StyleDrop paper [53]. The style prompts for these 18 styles can also be found in the StyleDrop appendix.

#### A.2. Evaluation Prompts

We present 23 evaluation prompts collected from StyleDrop paper [53] used for our quantitative and qualitative comparisons:

- • An Opera house in Sydney in {style prompt} style
- • A fluffy baby sloth with a knitted hat trying to figure out a laptop, close up in {style prompt} style
- • A Golden Gate bridge in {style prompt} style
- • The letter ‘G’ in {style prompt} style
- • A man riding a snowboard in {style prompt} style
- • A panda eating bamboo in {style prompt} style
- • A friendly robot in {style prompt} style
- • A baby penguin in {style prompt} style
- • A moose in {style prompt} style
- • A towel in {style prompt} style
- • An espresso machine in {style prompt} style
- • An avocado in {style prompt} style
- • A crown in {style prompt} style
- • A banana in {style prompt} style
- • A bench in {style prompt} style
- • A boat in {style prompt} style
- • A butterfly in {style prompt} style
- • An F1 race car in {style prompt} style
- • A Christmas tree in {style prompt} style
- • A cow in {style prompt} style
- • A hat in {style prompt} style
- • A piano in {style prompt} style
- • A wood cabin in {style prompt} style In Fig. S1, we present the detailed style descriptions gen-

erated by GPT-4o, which serve as text prompts for the GPT4o Prompt baseline. Specifically, for each of the 18 reference styles, we obtained comprehensive textual descriptions using GPT-4o. These detailed prompts were directly used for text-to-image generation.

#### A.3. User Study

In this section, we provide detailed information about the setup of our user study. Our user study aims to measure human preferences in two key objectives of style-driven image generation: style alignment and text alignment. To assess these preferences, we conduct pairwise comparisons be-

tween our method and each baseline for each objective. Participants are shown the reference image, target text prompt, and two generated images (one from each method) and are asked to choose the image that better satisfies the objective. We collect three responses from each of the 150 participants, resulting in a total of 450 responses for each comparison. The full instructions used in our questionnaires are as follows.

For style alignment objective,

- • Given a reference image and two machine-generated images, select which machine-generated output better matches the style of the reference image for each pair.
- • Please focus only on the style including color schemes, layouts, illumination, and brushstrokes.
- • If it’s difficult to determine a preference, please select “Cannot Determine / Both Equally”.

For text alignment objective,

- • Given a reference image and two machine-generated images, select which machine-generated output better matches the target text for each pair.
- • Please focus only on the text, without regard for the reference image.
- • If it’s difficult to determine a preference, please select “Cannot Determine / Both Equally”.

#### A.4. Implementation

To ensure reproducibility, we provide pseudo-code implementations of Style-friendly SNR samplers in Fig. S2 and the addition of LoRA [20] parameters to MM-DiT for training in Fig. S3.

Optimizer and Learning Rate. We use the Adam optimizer [28] at a learning rate of 10-4 for 300 steps. The batch size is set to 1, with gradient accumulation over 4 steps.

Guidance Scale and Inference Steps. During inference, we use a guidance scale [17, 37] of 7.0. The number of denoising steps is set to 28.

Model-Specific Details. FLUX-dev [31] is a guidancedistilled model [37] that takes guidance scale as input. For fine-tuning, we fix the guidance scale to 1.0 to match standard diffusion training, enable gradient checkpointing for memory efficiency, and use BF16 quantization for both finetuning and inference. SD3.5-8B [59] use FP16 precision.

Disabling Timestep Shifting. SD3 [9] uses a timestep shifting mechanism. However, we disable this shifting for our Style-friendly SNR sampler to isolate the effect of our proposed SNR sampling strategy.

Baseline Implementation. We use the Hugging Face Diffusers library (version 0.31.0) for consistent training and

inference across methods.

### B. Baselines

#### B.1. Direct Consistency Optimization

Direct Consistency Optimization (DCO) [32] is a finetuning method inspired by direct preference optimization [43] commonly used in large language models (LLMs). Instead of directly minimizing the diffusion loss, DCO aims to ensure that the diffusion loss of the fine-tuned model is lower than that of the pre-trained model on the reference data. The objective function is defined as:

LDCO(x0) = Et,ϵ − logσ(−βT

||vθ(xt,t) − v(xt,t)||2 − ||vϕ(xt,t) − v(xt,t)||2) , (7)

where v(xt,t) is target velocity field, vθ, is fine-tuning model, and vϕ is frozen pre-trained model.

In this objective, the parameter βT controls the strength of the preference towards the fine-tuned model over the pretrained model. DCO increases the relative likelihood of the fine-tuned model over the pre-trained model, penalizing less when the fine-tuned model’s loss is smaller. This helps preserve the text-to-image alignment of the pre-trained model.

However, DCO requires computations involving both the fine-tuned and pre-trained models, making it computationally more intensive than directly fine-tuning using the standard diffusion loss. In our experiments, we observed that using a large value of βT = 1000 resulted in slower convergence and suboptimal performance. Therefore, we set βT = 1 to achieve better results.

#### B.2. IP-Adapter

IP-Adapter [62] is designed to enable text-to-image models to generate identity-preserving images by training a compact adapter that encodes CLIP image embeddings [42]. This adapter introduces the CLIP image embedding as an additional input by concatenating its output with the text embeddings. The parameter-efficient nature of IP-Adapter allows for easy training and deployment across various textto-image models. However, a notable limitation is its restricted style alignment due to the expressive constraints of CLIP embeddings, which may result in generated images that do not fully capture detailed stylistic characteristics. IPAdapter allows adjusting the conditioning strength by scaling the embeddings with a factor between 0 and 1; we use a scale of 0.6 in all experiments. Using a scale of 1 can lead to content leakage beyond the style.

#### B.3. RB-Modulation

RB-Modulation [46] is a zero-shot approach using Stable Cascade [40], a model accepting both CLIP image embed-

dings and text embeddings as inputs. During the denoising process, RB-Modulation employs gradient guidance of a CSD [54], a model fine-tuned from CLIP to measure style similarity, resembling classifier guidance [7]. At each denoising step, CSD computes the similarity between the approximated x0 and the reference image, guiding the generation process to enhance this similarity. RB-Modulation also aggregates multiple attention features.

However, this approach relies on models that accept CLIP image embeddings, limiting model selection. Additionally, using gradient guidance of CSD increases inference costs, making the generation process more computationally intensive.

#### B.4. Style-Aligned

Style-Aligned [15] generates consistent sets of images with the same style by ensuring that features of each image attend to those of a reference image through shared key and value features in self-attention layers of image tokens. It first maps the reference image to noise using DDIM inversion [55] and shares self-attention features during denoising. The fidelity to the reference style can be controlled by amplifying the self-attention logits in the diffusion model. However, Style-Aligned is not directly applicable to MMDiT [9] architecture that lacks image-only self-attention layers. Moreover, artificially amplifying self-attention logits can lead to artifacts and lower-quality images due to conflicting attention features.

#### B.5. Offset Noise

Offset noise [13] is a method proposed to fine-tune diffusion models for generating monochromatic images. During the diffusion process, a constant offset noise—identical across all pixel positions—is added to the standard Gaussian noise, scaled by a small factor (e.g., 0.1). This introduces an explicit bias toward monotonic noise patterns, encouraging the model to learn and reproduce solid colors. While offset noise aids in learning monotonous patterns, it can hinder the model’s capacity to learn more complex styles.

Here, we additionally experiment with incorporating offset noise into our training process in Tab. S1. Offset noise with a scale of 0.1 improves the SD3 sampler’s results in DINO and CLIP-I scores, as many reference styles from the StyleDrop paper [53] have monochromatic backgrounds, favoring this trick. However, it still does not reach the performance of our Style-friendly SNR sampler. Moreover, when we combine our Style-friendly approach with a smaller scale of offset noise (0.01), we observe a slight improvement in the style alignment of FLUX-dev.

This quantitative evaluation is based on the monochromatic backgrounds prevalent in the StyleDrop [53] references. Our qualitative comparisons in Fig. S6 show that offset noise struggles with complex references, failing to cap-

[Figure 196]

[Figure 197]

Soft washes of color with delicate shading and detailed linework emphasize the texture and character of the building. The use of light and shadow enhances depth, creating a classic, timeless watercolor illustration.

A glowing, bioluminescent effect reminiscent of an X-ray scan, using bright cyan tones against a pitch-black background. The high contrast and fine details emphasize a surreal and otherworldly aesthetic.

[Figure 198]

[Figure 199]

A loose and fluid approach with vibrant color transitions, blending warm and cool tones in an organic manner. The soft edges and bleeding pigments create a dreamy, almost ethereal effect.

Clean, modern, and minimalistic design using soft pastel tones and geometric shapes. The absence of outlines and the smooth gradients give it a polished, professional feel.

[Figure 200]

Simple, playful strokes with bright, unmixed colors give a naive and spontaneous feel. The use of bold outlines and uneven brushwork adds to the charm of a carefree,

Bold, flat design with crisp, clean edges and a drop shadow effect, making it appear like a sticker or cutout. The smooth color transitions and simplified shapes add to

[Figure 201]

unrefined artistic approach.

the playful and modern aesthetic.

[Figure 202]

[Figure 203]

Thick, visible brushstrokes with swirling, dynamic

A smooth, polished, and cartoonish 3D model with

patterns create a sense of movement and depth. The use of

simplified facial features. The soft shading, realistic hair

bold, contrasting colors and a rich impasto texture gives the image a highly expressive and emotional atmosphere.

texture, and subtle lighting create a blend of realism and stylization.

[Figure 204]

[Figure 205]

Expressive and bold brushstrokes create a sense of movement and texture. The use of vibrant yet earthy tones, along with swirling background patterns, adds a dynamic and emotional depth to the composition.

A simple, naive, and hand-drawn aesthetic with visible crayon strokes and uneven coloring. The rough texture and imperfect lines add to the charm of a playful, unpolished artwork.

[Figure 206]

[Figure 207]

A simple and clean cartoon-like approach using bold outlines and flat colors. The exaggerated, symbolic composition conveys a strong message with minimal visual elements, making it both modern and impactful.

A highly reflective, gold-like surface with smooth, liquidlike textures. The dripping effect and polished finish create a futuristic and surrealistic feel.

[Figure 208]

[Figure 209]

A clean and modern aesthetic with soft color palettes and smooth gradients. The absence of outlines and the use of simple geometric shapes contribute to a polished and contemporary design.

A detailed and rustic carving with deep grooves and textured patterns. The organic, handcrafted appearance gives it an ancient, folklore-inspired look.

[Figure 210]

[Figure 211]

A monochrome composition created with flowing,

Bright, exaggerated, and glossy 3D modeling with a soft,

dynamic strokes. The expressive lines and rhythmic

cartoonish aesthetic. The smooth reflections and bold,

movement give it an energetic and almost dreamlike atmosphere, reminiscent of classic pen-and-ink studies.

saturated colors create a fun and whimsical atmosphere.

[Figure 212]

[Figure 213]

Smooth, flowing, and ethereal ribbons of vibrant color set

A high-gloss, semi-transparent glass effect combined with

against a dark background. The soft glow and gradient blending create a futuristic and dynamic visual effect.

a metallic base. The dramatic lighting and reflections enhance the sci-fi and high-tech aesthetic.

Figure S1. Detailed style description. These descriptions serve as the input prompts for our GPT-4o [23] prompt baseline, which generates images solely based on these textual style specifications.

- 1 # Inputs: mu, sigma, B, latent
- 2 # sample log-SNR
- 3 logsnr = torch.normal(mean=mu, std=sigma, size=(B,))
- 4 # compute timestep t
- 5 t = torch.nn.functional.sigmoid(-logsnr / 2).view(B, 1, 1, 1)
- 6 # sample noise
- 7 noise = torch.randn_like(latent)
- 8 # diffuse latent
- 9 noisy_latent = (1.0 - t) * latent + t * noise

Figure S2. PyTorch implementation of a Style-friendly SNR sampler.

ture intricate stylistic details. This indicates that while offset noise can help with simple, uniform styles, it is vulnerable to complex styles.

- 1 # Inputs: model_name, rank
- 2 # Configure LoRA for the specified model
- 3 if model_name == "FLUX":
- 4 target_modules = [
- 5 "to_k", "to_q", "to_v", "to_out.0",
- 6 "add_k_proj", "add_q_proj", "add_v_proj", "proj_mlp", "proj_out"
- 7 ]
- 8 elif model_name == "SD3":
- 9 target_modules = [
- 10 "to_k", "to_q", "to_v", "to_out.0",
- 11 "add_k_proj", "add_q_proj", "add_v_proj", "to_add_out"
- 12 ]
- 13 else:
- 14 raise ValueError(f"Unsupported model: {model_name}")

- 15
- 16 # LoRA configuration
- 17 transformer_lora_config = LoraConfig(
- 18 r=rank,
- 19 lora_alpha=rank,
- 20 init_lora_weights="gaussian",
- 21 target_modules=target_modules,
- 22 )
- 23
- 24 # Add adapter to the transformer
- 25 transformer.add_adapter(transformer_lora_config)

###### Figure S3. PyTorch implementation of LoRA integration.

Log-SNR Mean Ablation (CLIP) FLUX

Log-SNR Standard Deviation Ablation (CLIP) FLUX

Rank Ablation (CLIP) FLUX

0.72

0.72

0.72

SD3.5

SD3.5

FLUX+SD3 sampler

| |
|---|

| |
|---|

SD3.5

SD3.5+SD3 sampler

0.70

0.70

0.70

CLIPImageSimilarity

0.68

0.68

0.68

CLIPSimilarity

CLIPSimilarity

0.66

0.66

0.66

0.64

0.64

0.64

0.62

0.62

0.62

0.60

0.60

0.60

-8.0 -6.0 -4.0 SD3 -2.0 0.0 Value

0.4 1.0 2.0 3.0 4.0 Value

4 8 32 Rank Value

(a) Varying µ.

(b) Varying σ.

(c) Varying LoRA Rank.

- Figure S4. SNR sampler analysis. CLIP-I similarities with FLUX and SD3.5-8B. Dotted lines in (c) indicate the results of SD3 sampler [9].

### C. Additional Results

Effectiveness Compared to Increasing Model Capacity. To demonstrate that our method is more effective than increasing model capacity, we conduct an additional experiment where we fine-tune the model using the SD3 sampler with a higher LoRA rank of 128. As shown in Tab. S2, our Style-friendly SNR sampler with a rank of 32 achieves higher DINO and CLIP-I scores compared to the SD3 sampler with a rank of 128. This indicates that focusing on the critical noise levels where styles emerge has a more significant impact than increasing the number of trainable parameters.

#### C.1. Quantitative Results

CLIP Scores. In the main paper, we presented analyses of the mean µ, standard deviation σ, and LoRA rank using the DINO similarity score. In Fig. S4a, we provide the corresponding CLIP image similarity (CLIP-I) scores to further validate our findings. The CLIP-I scores exhibit a similar trend to the DINO scores, where decreasing µ enhances style alignment. Varying σ affects the CLIP-I scores consistently with the DINO results. Our Style-friendly SNR sampler with µ = −6 and a rank of 4 still outperforms the SD3 sampler with a rank of 32 (dotted lines).

Trainable Parameters. To validate the importance of fine-tuning both transformer blocks of MM-DiT [9], we

|Method|Model<br><br>|Metrics|
|---|---|---|
| | |DINO ↑ CLIP-I ↑ CLIP-T ↑|
|SD3 Sampler [9] w/ offset 0.1 Style-friendly<br><br>w/ offset 0.01|SD3.5 SD3.5 SD3.5 SD3.5<br><br>|0.424 0.670 0.350 0.452 0.678 0.353 0.489 0.698 0.349 0.476 0.697 0.350<br><br>|
|SD3 Sampler [9] w/ offset 0.1 Style-friendly<br><br>w/ offset 0.01|FLUX-dev FLUX-dev FLUX-dev FLUX-dev<br><br>|0.373 0.645 0.350 0.451 0.679 0.349 0.461 0.686 0.344 0.500 0.704 0.341<br><br>|

Table S1. Incorporating offset noise. Offset noise improves SD3 sampler but still does not reach the performance of our Stylefriendly SNR sampler; combining our Style-friendly approach with Offset Noise at a smaller scale (0.01) slightly enhances the style alignment of FLUX-dev. Here, we use σ = 2 for Stylefriendly.

|Method<br><br>|Model<br><br>|Metrics|
|---|---|---|
| | |DINO ↑ CLIP-I ↑ CLIP-T ↑|
|SD3 Sampler [9] w/ rank 128<br><br>Style-friendly<br><br>|FLUX-dev FLUX-dev FLUX-dev|0.373 0.645 0.350 0.426 0.668 0.345 0.461 0.686 0.344<br><br>|

- Table S2. Comparison to increasing LoRA rank.

Method DINO CLIP-I CLIP-T Style-friendly 0.489 0.698 0.349

w/o Text attn 0.462 0.693 0.349

- Table S3. Ablation study on trainable parameters.

conduct an ablation study on SD3.5-8B, comparing the results of training LoRA adapters on only the imagetransformer blocks versus training on both the image and text-transformer blocks. As shown in Tab. S3, fine-tuning both the image and text-transformer blocks leads to higher DINO and CLIP-I scores compared to fine-tuning only the image-transformer blocks, while the CLIP-T scores are identical. This indicates that including the text-transformer blocks in the fine-tuning process enhances the model’s ability to learn stylistic features without compromising text alignment. These results suggest that to effectively capture new styles, it is beneficial to fine-tune both the visual and linguistic components of MM-DiT.

#### C.2. Qualitative Results

SD3.5 Samples. We extend our qualitative comparison by evaluating our Style-friendly SNR sampler using the SD3.58B model [59], comparing it against previous fine-tuning methods, namely the SD3 sampler [9] and DCO [32]. As shown in Fig. S5, the results are consistent with the qualitative comparisons using FLUX-dev presented in the main paper.

Additional Comparison. We further demonstrate the effectiveness of a Style-friendly SNR sampler in learning complex style templates, such as multi-panel images. As shown in Fig. S6, our method captures the given multipanel style, generating images that closely resemble the reference. In contrast, previous fine-tuning approaches, SD3 sampler [9] and DCO [32], fail to learn the multi-panel concept, producing images without the panel structure. The offset noise [13] method attempts to reflect the style but still generates images with a single panel or fewer panels than the reference. Zero-shot approaches including IPAdapter [62], RB-Modulation [46], and Style-Aligned [15] also attempt to generate multi-panel images but often produce outputs with structures different from the reference, as shown in Fig. S7. This highlights the capability of our method to handle challenging styles that other approaches struggle with.

Object fine-tuning using Style-friendly SNR sampler. We further evaluate whether our proposed Stylefriendly SNR sampler—designed specifically for style learning—affects object-driven generation performance. As shown in Fig. S8, our Style-friendly SNR sampler and the original SNR sampler produce qualitatively similar results when fine-tuning on object references. Our approach successfully captures critical details, including shapes, colors, and prominent text or patterns (e.g., on the bowl and can). However, it occasionally omits subtle features, such as the small teeth of the monster toy. These findings support our hypothesis that distinct approaches are required for object-centric and style-driven fine-tuning; our Style-friendly sampler, while slightly suboptimal for objectcentric fine-tuning, excels in capturing nuanced style characteristics.

Additional Samples. We present additional samples using the FLUX-dev [31] to demonstrate the versatility of our method. Fig. S9 shows that even when fine-tuned on square reference images, our model can generate images with different aspect ratios while maintaining the reference style. For each prompt, we show results from two different random seeds to illustrate diversity across various aspect ratios. Fig. S10 provides additional typography samples in different aspect ratios, exhibiting our capability to produce stylized textual content.

### D. Limitations and Discussions

Style Prompt Design. As shown in Fig. S11, using a different style prompt during fine-tuning can lead to emphasizing different stylistic features, such as child-like elements or background architectures (second row) instead of watercolor painting elements (first row), which may not align

Wooden sculpture Line drawing Cartoon line drawing 3d rendering Glowing

|[Figure 214]|
|---|

|[Figure 215]|
|---|

|[Figure 216]|
|---|

|[Figure 217]|
|---|

|[Figure 218]|
|---|

ReferenceStyle-friendly

|[Figure 219]| |
|---|---|
|[Figure 220]|[Figure 221]|

|[Figure 222]| |
|---|---|
|[Figure 223]|[Figure 224]|

|[Figure 225]| |
|---|---|
|[Figure 226]|[Figure 227]|

|[Figure 228]| |
|---|---|
|[Figure 229]|[Figure 230]|

|[Figure 231]| |
|---|---|
|[Figure 232]|[Figure 233]|

|[Figure 234]| |
|---|---|
|[Figure 235]|[Figure 236]|

|[Figure 237]| |
|---|---|
|[Figure 238]|[Figure 239]|

|[Figure 240]| |
|---|---|
|[Figure 241]|[Figure 242]|

|[Figure 243]| |
|---|---|
|[Figure 244]|[Figure 245]|

|[Figure 246]| |
|---|---|
|[Figure 247]<br><br>[Figure 248]|[Figure 249]|

DCOsamplerSD3

|[Figure 250]| |
|---|---|
|[Figure 251]|[Figure 252]|

|[Figure 253]| |
|---|---|
|[Figure 254]|[Figure 255]|

|[Figure 256]| |
|---|---|
|[Figure 257]|[Figure 258]|

|[Figure 259]| |
|---|---|
|[Figure 260]|[Figure 261]|

|[Figure 262]| |
|---|---|
|[Figure 263]<br><br>[Figure 264]|[Figure 265]|

- Figure S5. Comparison of fine-tuning the SD3.5-8B. We show ‘A crown’, ‘A Golden Gate bridge’, and ‘An espresso machine’ in various styles. The results with SD3.5-8B are consistent with the qualitative comparison based on FLUX-dev presented in the main paper.

with the user’s focus. Users should be mindful that variations in the style prompt can lead to different results. Nevertheless, our approach demonstrates effective style learning for style prompts given by the users.

Computational Cost. While fine-tuning diffusion models remains the most promising approach for achieving style alignment, it involves significant computational costs. Finetuning for a new style typically requires around 300 fine-

|[Figure 266]|
|---|

Reference

A humorous sequence featuring a red cartoon character in a multi-panel comic style

|[Figure 267]|
|---|

|[Figure 268]|
|---|

|[Figure 269]|
|---|

|[Figure 270]|
|---|

SD3samplerStyle-friendlyOffsetnoiseDCO

|[Figure 271]|
|---|

|[Figure 272]|
|---|

|[Figure 273]|
|---|

|[Figure 274]|
|---|

|[Figure 275]|
|---|

|[Figure 276]|
|---|

|[Figure 277]|
|---|

|[Figure 278]|
|---|

|[Figure 279]|
|---|

|[Figure 280]|
|---|

|[Figure 281]|
|---|

|[Figure 282]|
|---|

- Figure S6. Additional qualitative comparison. Our Style-friendly approach successfully captures complex multi-panel styles, generating images that closely resemble the reference. The prompts used are “A fluffy baby sloth with a knitted hat trying to figure out a laptop, close up in {style prompt} style”, “A banana in {style prompt} style”, “A Christmas tree in {style prompt} style”, and “A bench in {style prompt} style”.

|[Figure 283]|
|---|

Reference

A humorous sequence featuring a red cartoon character in a multi-panel comic style

|[Figure 284]|
|---|

|[Figure 285]|
|---|

|[Figure 286]|
|---|

|[Figure 287]|
|---|

RB-ModulationStyle-AlignedIP-AdapterStyle-friendly

|[Figure 288]|
|---|

|[Figure 289]|
|---|

|[Figure 290]|
|---|

|[Figure 291]|
|---|

|[Figure 292]|
|---|

|[Figure 293]|
|---|

|[Figure 294]|
|---|

|[Figure 295]|
|---|

|[Figure 296]|
|---|

|[Figure 297]|
|---|

|[Figure 298]|
|---|

|[Figure 299]|
|---|

- Figure S7. Additional qualitative comparison. Our method effectively captures the multi-panel style, whereas zero-shot methods generate images with different structures or introduce artifacts.

Number ‘3’ clock ‘Bon Appetit’ bowl ‘Transatlantic IPA’ can Monster toy RC car Bear plushie

|[Figure 300]|
|---|

|[Figure 301]|
|---|

|[Figure 302]|
|---|

|[Figure 303]|
|---|

|[Figure 304]|
|---|

|[Figure 305]|
|---|

Reference

|[Figure 306]|
|---|

|[Figure 307]|
|---|

|[Figure 308]|
|---|

|[Figure 309]|
|---|

|[Figure 310]|
|---|

|[Figure 311]|
|---|

SD3samplerStyle-friendly

|[Figure 312]|
|---|

|[Figure 313]|
|---|

|[Figure 314]|
|---|

|[Figure 315]|
|---|

|[Figure 316]|
|---|

|[Figure 317]|
|---|

A {object name} on top of green grass with sunflowers around it

|[Figure 318]|
|---|

|[Figure 319]|
|---|

|[Figure 320]|
|---|

|[Figure 321]|
|---|

|[Figure 322]|
|---|

[Figure 323]

Style-friendlySD3sampler

|[Figure 324]|
|---|

|[Figure 325]|
|---|

|[Figure 326]|
|---|

|[Figure 327]|
|---|

|[Figure 328]|
|---|

[Figure 329]

A {object name} on top of a purple rug in a forest

|[Figure 330]|
|---|

|[Figure 331]|
|---|

|[Figure 332]|
|---|

|[Figure 333]|
|---|

|[Figure 334]|
|---|

[Figure 335]

Style-friendlySD3sampler

|[Figure 336]|
|---|

|[Figure 337]|
|---|

|[Figure 338]|
|---|

|[Figure 339]|
|---|

|[Figure 340]|
|---|

[Figure 341]

A purple{object name}

- Figure S8. Object fine-tuning comparison. We compare our Style-friendly SNR sampler and the standard sampler on object-driven finetuning. Both approaches generate similar overall results, though our Style-friendly sampler occasionally misses minor details, such as the small teeth of the monster toy. Nevertheless, the Style-friendly sampler reliably captures the object’s overall shape, color, and key details such as text and patterns on the bowl and can. The object names are written at the top of the reference images.

Kid crayon

Flowing smoke

Wooden

Watercolor painting 3d rendering Glowing

drawing

wave design

sculpture

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

GeneratedSamples(Prompt1)GeneratedSamples(Prompt2)Reference

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

- Figure S9. Additional samples. Each row shows images generated with the same random seed at a resolution of 1216×832, using the prompts “a cute city made of sushi in {style prompt} style” and “mischievous ferret with a playful grin squeezes itself into a large glass jar, in {style prompt} style”.

|[Figure 372]|
|---|

|[Figure 373]|
|---|

|[Figure 374]|
|---|

|[Figure 375]|
|---|

|[Figure 376]|
|---|

|[Figure 377]|
|---|

|[Figure 378]|
|---|

|[Figure 379]|
|---|

|[Figure 380]|
|---|

|[Figure 381]|
|---|

|[Figure 382]|
|---|

|[Figure 383]|
|---|

|[Figure 384]|
|---|

|[Figure 385]|
|---|

|[Figure 386]|
|---|

|[Figure 387]|
|---|

|[Figure 388]|
|---|

|[Figure 389]|
|---|

|[Figure 390]|
|---|

|[Figure 391]|
|---|

References

Generated Samples

- Figure S10. Typography. The first column shows reference images. The second and third columns display samples generated at a resolution of 832×1216, and the fourth column presents samples at 704×1408 resolution. The prompts used are “the words that says ‘{letters}’ are written in English, in {style prompt} style”, where ‘{letters}’ represents the words synthesized in the samples.

tuning steps, and due to the iterative nature of diffusion models, generating a single image during inference can take several seconds. We anticipate that future work will explore applying our Style-friendly SNR sampler during the training of zero-shot models [62] or integrating it with models that offer faster inference speeds, such as Consistency Models [58] or Adversarial Diffusion Distillation models [50]. These developments could reduce both training and inference times, making style-driven generation more accessible and efficient.

### E. Broader Impact

Our Style-friendly SNR sampler makes diffusion models successful in fine-tuning various style references. This advancement allows diffusion models to function effectively as digital art previewers, benefiting artists and non-expert users by simplifying the creative process. However, we note that it is important to be careful of copyright when using reference images for fine-tuning. Practitioners should ensure they have permissions to use reference images.

|[Figure 392]|
|---|

|[Figure 393]|
|---|

|[Figure 394]|
|---|

|[Figure 395]|
|---|

|[Figure 396]|
|---|

|[Figure 397]|
|---|

|[Figure 398]|
|---|

|[Figure 399]|
|---|

|[Figure 400]|
|---|

|[Figure 401]|
|---|

… in watercolor painting style … in watercolor painting style

|[Figure 402]|
|---|

|[Figure 403]|
|---|

|[Figure 404]|
|---|

|[Figure 405]|
|---|

|[Figure 406]|
|---|

|[Figure 407]|
|---|

|[Figure 408]|
|---|

|[Figure 409]|
|---|

… in child like simplicity effect

… in vintage architectural effect

- Figure S11. Effect of Style Prompt Design. The first row shows images generated using style prompts from the StyleDrop paper [53] during both fine-tuning and generation. The second row shows images generated using a different style prompt during both fine-tuning and generation. Each column is generated using the same random seed. This demonstrates how varying the style prompt can lead to different stylistic elements being emphasized in the generated images.

