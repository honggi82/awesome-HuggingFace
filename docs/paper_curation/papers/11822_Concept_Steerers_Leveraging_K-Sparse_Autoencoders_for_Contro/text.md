# arXiv:2501.19066v3[cs.CV]12Oct2025

## Concept Steerers: Leveraging K-Sparse Autoencoders for Test-Time Controllable Generations

#### Dahye Kim1 Deepti Ghadiyaram12†

1Boston University 2Runway

{dahye, dghadiya}@bu.edu

### Abstract

Despite the remarkable progress in text-to-image generative models, they are prone to adversarial attacks and inadvertently generate unsafe, unethical content. Existing approaches often rely on fine-tuning models to remove specific concepts, which is computationally expensive, lacks scalability, and/or compromises generation quality. In this work, we propose a novel framework leveraging k-sparse autoencoders (k-SAEs) to enable efficient and interpretable concept manipulation in diffusion models. Specifically, we first identify interpretable monosemantic concepts in the latent space of text embeddings and leverage them to precisely steer the generation away or towards a given concept (e.g., nudity) or to introduce a new concept (e.g., photographic style) – all during test time. Through extensive experiments, we demonstrate that our approach is very simple, requires no retraining of the base model nor LoRA adapters, does not compromise the generation quality, and is robust to adversarial prompt manipulations. Our method yields an improvement of 20.01% in unsafe concept removal, is effective in style manipulation, and is ∼ 5x faster than the current state-of-the-art. Code is available at: https://github.com/kim-dahye/steerers

### 1 Introduction

Text-to-image (T2I) generative models have revolutionized content generation by producing diverse and highly photorealistic images, enabling a wide range of applications such as digital art creation [1], image editing [2], and medical imaging [3]. These models are usually trained on several billions of web-scraped image and text pairs presumably capturing a broad spectrum of semantic concepts. Consequently, these models are also prone to be exposed to and thus generate disturbing content containing nudity, violence, child exploitation, and self-harm – raising serious ethical concerns about their downstream applications.

Several attempts have been made to enforce safe generations in the past: integrating safety filters as part of the generation pipeline [4], guiding the generation process away from a pre-defined unsafe latent space [5], or directly erasing inappropriate concepts by modifying model weights [6, 7, 8]. While partially successful, some of these methods involve model training which is not only computationally expensive but also alters the overall model’s generative capabilities. More recently, a few inference-based approaches have been proposed, which do not alter model weights [9, 10]. SAFREE [9] alters the semantics of the input prompt by filtering toxic tokens, while TraSCE [10] modifies negative prompting with gradient computation to guide the model towards safer outputs. Crucially, sometimes these models have the undesirable consequence of visually degraded output generations or being misaligned with input prompts, even when the prompts are benign. Additionally, the increased inference time (e.g., 8.84s overhead per image as noted in TraSCE [10]) makes them difficult to deploy in practice.

Preprint. Under review.

Change Photographic Styles Change Object attributes

Remove Unsafe concepts

|[Figure 1]<br><br>[Figure 2]<br><br>|
|---|

|[Figure 3]<br><br>[Figure 4]|
|---|

|[Figure 5]<br><br>[Figure 6]|
|---|

|[Figure 7]<br><br>[Figure 8]|
|---|

|[Figure 9]<br><br>[Figure 10]|
|---|

|[Figure 11]<br><br>[Figure 12]|
|---|

Change

Change

Remove nudity

season style

to a full shot

to winter

of a dog

|[Figure 13]<br><br>[Figure 14]|
|---|

|[Figure 15]<br><br>[Figure 16]|
|---|

|[Figure 17]<br><br>[Figure 18]|
|---|

|[Figure 19]<br><br>[Figure 20]|
|---|

|[Figure 21]<br><br>[Figure 22]|
|---|

|[Figure 23]<br><br>[Figure 24]|
|---|

Make

Change

Remove

the image

car color

violence

darker

to blue

- Figure 1: Monosemantic interpretable concepts such as nudity, photographic styles, and object attributes are identified using k-sparse autoencoders (k-SAE). We leverage them to enable precise modification of a desired concept during the generation process, without impacting the overall image structure, photo-realism, visual quality, and prompt alignment (for safe concepts). Our framework can be used to remove unsafe concepts (left), photographic styles (middle), and object attributes (right).

In this work, we posit that the semantic information is interwoven across different layers of a generative model in complex ways that is not fully understood. Subsequently, existing training or inference-based safe generation techniques could be altering this latent landscape in undesirable ways leading to misaligned or irrelevant outputs. To this end, we approach the generation process from the ground up and explore the following crucial question: can we systematically isolate monosemantic concepts of varied granularities (fine-grained and abstract) from the generative latent space and surgically manipulate only them?1 Such an invaluable tool would allow the user to intentionally control just the relevant concept of interest without disrupting the overall latent landscape.

To this end, we leverage k-sparse autoencoders (k-SAE) [12] to design controllable generative models. k-SAEs have shown promising progress in interpreting language models by learning a sparse dictionary of monosemantic concepts [13, 14]. In our work, we first train a k-SAE on the embeddings extracted from a corpus of text prompts containing semantic concepts we wish to control (e.g., unsafe concepts). Once trained, each k-SAE’s hidden state corresponds to an isolated monosemantic concept. During the generation process, given a concept we wish to steer, we use k-SAE to identify its corresponding latent direction and precisely manipulate the presence of that concept in the outcome, without impacting the overall generation capability (Fig. 1). Notably, our method does not require any fine-tuning as in [15], synthetic data generation as in [16], training a separate LoRA adapter [17] for each concept as in [18] to manipulate making it fast, efficient, and adaptable to any pre-trained text to image generative framework. We summarize our findings and key contributions below:

- • We identify interpretable monosemantic concepts in text-to-image generation latent landscape using a k-sparse autoencoder. The k-SAE serves as a Concept Steerer and offers precise control over semantic concepts (e.g., nudity, photographic style, etc.)
- • Concept Steerer achieves state-of-the-art performance on unsafe concept removal while being ∼5x faster than the existing best method [10], without compromising visual quality.
- • Concept Steerer effectively manipulates photographic and artistic styles, object attributes, enabling controlled yet creative image generation.
- • Concept Steerer is robust to adversarial prompt manipulations, achieves a 20.01% improvement against red-teaming algorithms, ensuring reliable image generation even under challenging scenarios.
- • Concept Steerer works out-of-the-box at test-time, can be applied to any text-to-image generative model, requires no retraining nor LoRA adapters, is simple and efficient.

### 2 Related Work

Controlling diffusion models: [19, 20] fine-tune diffusion models using human feedback and [21, 22] propose inference-time diffusion steering with reward functions. However, these methods rely on strong reward functions, and are computationally intensive [23]. Some methods achieve controllability by training additional modules such as low-rank adapters (LoRAs) [18, 24], which requires millions of parameters per concept and significantly increases generation time [25]. Several inference-time intervention works attempt fine-grained control at test time. However, estimating

1In contrast to the one-to-many mapping of polysemantic neurons, monosemantic neurons form a one-to-one correlation with their related input features [11].

|[Figure 25]<br><br>[Figure 26]<br><br>| |
|---|
|
|---|

[Figure 27]

| | |
|---|---|
| | |

Diffusion model

[Figure 28]

Text prompt

Unsafe

Text

“Greek goddess

|[Figure 29]<br><br>[Figure 30]|
|---|

Encoder

| |
|---|

posing …”

k-SAE

| |
|---|

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

Decoder

Encoder

| |
|---|

| |
|---|

| |
|---|

Text

Concept 𝑪

| |
|---|

| |
|---|

“Nudity”

Encoder

Unsafe path

| |
|---|

[Figure 31]

Safe

λ∗

Safe path

- Figure 2: K-sparse autoencoder (k-SAE) is trained on feature representations from the text encoder of the diffusion model. Once trained, it serves as a Concept Steerer, enabling precise concept manipulation at test-time. λ denotes the strength of the concept.

noise at each step for each concept during generation [26, 27] significantly slows down generation and steering model activations based on optimal transport [28] requires learning activation mapping for each style. By contrast, our approach is very simple, requires no training of the base model or LoRA adapters, no additional noise/gradient computation during the generation process. Moreover, once trained, our approach allows us to manipulate any concept we want without further tuning. To the best of our knowledge, ours is the only work which leverages sparse autoencoders to offer more creative and generative control to users.

Safe generation: Given the growing concerns of generative models’ capability to produce inappropriate content, several valuable research has emerged in this space. Some training-based methods [6, 7, 15, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40] directly remove inappropriate concepts from the diffusion model through additional fine-tuning, while some others like [41, 42] update model weights to erase concepts without retraining the model. Some recently proposed inferencebased approaches [9, 10] do not require training or weight updates. While effective, these methods often result in degraded image quality and increased inference time. Unlike all prior works, our method surgically isolates interpretable concepts in the generative latent space and manipulating only these in the text encoder. Thus, our approach enjoys the benefit of precise control of inappropriate concepts, does not compromise on generation quality, and maintains prompt-image alignment. In the similar vein, SDID [43] discovers interpretable latent directions for a given concept in the bottleneck of the diffusion model. However, it requires generating 1000 paired unsafe and safe images to learn a representation for each unsafe concept (e.g., 1000 pairs of images of violence and non-violence), making this approach less flexible and scalable. By contrast, our approach does not require thousands of generating pairs of images per concept. Instead, once Concept Steerer is trained on a set of prompts, it can steer various concepts and can be applied to different text-to-image models.

Interpreting diffusion models: Recent works have demonstrated that sparse autoencoders (SAE) could recover interpretable features in large language models [13, 14], CLIP vision features [44, 45], multimodal LLMs [46, 47] and diffusion features [48, 49]. [48] reveals monosemantic interpretable features represented within rich visual features of the diffusion model while [49] investigates how text information is integrated via cross-attention. However, in all prior works, the interpretation is done through manual inspection and/or visualization, which limits their scalability. By contrast, Concept Steerer offers an automated way to interpret the concepts it is trained on through steering the joint latent space of diffusion models.

- 3 Approach

We propose a simple yet effective technique to precisely isolate and steer semantic concepts such as nudity or photographic styles using k-sparse autoencoders [12] (k-SAE). We first present how we train such a k-SAE (Sec. 3.2) on text representations, followed by our method to combine different monosemantic neurons to steer abstract concepts (Sec. 3.3). We stress that a k-SAE is trained only once and no training is required for any concept the user wishes to introduce, eliminate, or modulate.

#### 3.1 Preliminaries on text to image models

Text-to-image diffusion models [50, 51, 52] primarily consist of a text encoder to extract a text prompt’s intermediate embedding and a diffusion model. During training, the diffusion model progressively denoises a noisy image (or its latent representation) conditioned on the text prompt’s intermediate embedding. Formally, given an input y0, the forward diffusion process progressively adds noise to y0 over T timesteps. The intermediate noisy image at timestep t is yt = (1 − βt)y0 + √βtϵ where ϵ is the Gaussian noise and βt is a timestep-dependent hyper parameter. In the reverse process, the diffusion model ϵθ iteratively denoises yt at each timestep, conditioned on the text prompt embedding c, to predict noise ϵ. The objective function for training the model is to minimize the error between the introduced and the predicted noise, defined as: Ey,t,ϵ∼N(0,1) ∥ϵ − ϵθ(yt,c,t)∥22 .

#### 3.2 Preliminaries on k-sparse autoencoders

Sparse autoencoders [53] are neural networks designed for learning compact and meaningful feature representations in an unsupervised manner. They consist of an encoder and a decoder, optimized jointly using a reconstruction loss and a sparsity regularization term to encourage only a few neurons to be maximally activated for a given input. However, the sparsity constraint introduces significant challenges during optimization [12, 54]. To mitigate these issues, k-sparse autoencoders (k-SAEs) [12] were introduced. They explicitly control the number of active neurons to k during training by applying a Top-k activation function at each training step. Consequently, this retains only the k highest activations and zeroes out the rest.

Let Wenc ∈ Rn×d and Wdec ∈ Rd×n represent the weight matrices of the k-SAE’s encoder and decoder respectively (Fig. 2). The hidden layer dimension n is defined as an integer multiple of the input feature dimension d. The ratio n/d, referred to as the expansion factor, controls the extent to which the hidden dimension is expanded relative to the input dimension. Following [13], bpre ∈ Rd denotes the bias term added to input x before feeding to the encoder (aka pre-encoder bias), while benc ∈ Rn denotes the bias term of the encoder.

Let x ∈ RL×d denote the intermediate representation of the text encoder for an input prompt in a text-to-image model, where L denotes the number of tokens. The encoded latent z is computed as:

z = ENC(x) = Top-k(ReLU(Wenc(x − bpre) + benc)), (1)

where the Top-k function retains only the top k neuron activations and sets the remaining activations to zero [12]. The decoder reconstructs xˆ as:

xˆ = DEC(x) = Wdecz + bpre. (2)

The training objective of a standard k-SAE is to minimize the normalized mean squared error (MSE) between the original feature x and the reconstructed feature xˆ, denoted by Lmse. ReLU and Top-K activation functions in Eq. 1 ensure that k-SAEs do not learn a linear mapping between inputs and outputs. However, both SAEs and k-SAEs suffer from the presence of “dead latents,” where a large proportion of latents stop activating entirely at some point in training. Presence of dead latents decreases the likelihood of the network discovering separable, interpretable features while incurring unnecessary computational cost [13]. To discourage dead latents, we incorporate an auxiliary MSE loss as suggested in [55]. Specifically, in every training step, we identify top kaux dead latents (i.e., latents that have the least activation value) and reconstruct a latent zˆ exclusively from them, as defined below:

zˆ = Top-kaux(ReLU(Wenc(x − bpre) + benc)). (3)

Now, let eˆ = Wdeczˆ represent the reconstruction using the top kaux dead latents. Laux is defined as a reconstruction loss between the auto encoder’s residual and the output from the dead neurons (eˆ). As discussed in [55], the intuition behind Laux is to compute gradients that push the parameters of the dead neurons in the direction of explaining the autoencoder residual (e). Thus, the total training loss is:

L = Lmse + αLaux = ∥x − xˆ∥22 + α∥e − eˆ∥22. (4)

The scalar α is a weighting factor that controls the relative contribution of the auxiliary loss.

#### 3.3 Concept Steerers

Given a human-interpretable concept C we wish to steer, we first extract its text embedding xC, pass it through the encoder and decoder of k-SAE, and finally perform an element-wise addition of the

2040

SD 1.4 Ours

SAFREE TraSCE

SD 1.4 Ours

SD 1.4 Ours

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Remove

Remove

Remove

nudity

nudity

violence

| |
|---|

| |
|---|

| |
|---|

Prompt: “ken doll, commoner, concept art, art in the style of Mesopotamia …”

Prompt: “… cute anime … detailed face …”

Prompt: “… portrayed inmate aaah …”

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Remove

Remove

Remove

nudity

nudity

violence

| |
|---|

Prompt: “painting of a cyberpunk android … fully clothed in futuristic armor…”

Prompt: “… pink … claude monet …”

Prompt: “… expressive sobbing rounded …”

(a) (b) (c)

- Figure 3: (a) Qualitative comparison of different approaches, including SAFREE [9] and TraSCE [10], on the I2P dataset. Our method removes nudity without significantly altering the generated images, resulting in outputs better aligned with the input prompt. (b) Qualitative examples from the I2P dataset. Our method allows fine-grained control over the removal of specific concepts, removing only the intended concept while preserving the overall structure and style of the generated images. (c) Qualitative examples from the Ring-ABell dataset. Our method successfully removes the abstract concept of violence, as shown by the absence of blood in the right images. The images are intentionally blurred for display purposes as they are disturbing.

3159

Remove violence

reconstructed k-SAE output with the input prompt embedding x.2 We express this as:

xsteered = x + Wdec(λ ∗ ENC(xC)), (5)

where λ denotes a scalar that controls the steering strength. The steered vector xsteered is used to condition the generation process during inference. As we show in Sec. 4, our approach requires a k-SAE to be trained only once, and provides model-agnostic, fine-grained control over concept steering without degrading the overall generation quality.

### 4 Experiments

Implementation details: We train k-sparse autoencoders on text embeddings with kaux = 256, and loss weight parameter α = 1/32 for 10k training steps. We train for a total training tokens of 400M on a batch size of 4096 with the learning rate 0.0004 using Adam [56] optimizer. The k-SAE is trained with k = 32 and an expansion factor of 4, resulting in a total hidden size dimension n = 3072 for Stable Diffusion (SD) 1.4 [50] in the unsafe removal task. For style manipulation, we use k = 64 with an expansion factor of 64, resulting in a total hidden size dimension n = 49152 for SD 1.4 and k = 64 with an expansion factor of 16, resulting in a total hidden size dimension n = 32768 for SDXL-Turbo [57]. These settings were found via ablation studies on downstream tasks and/or chosen based on overall training stability and sparsity. We apply a unit normalization constraint [58] on the decoder weights Wdec of the k-SAE after each update. Although our method can be applied in a model-agnostic manner to any text-to-image model, for a fair comparison with existing methods, we conduct experiments using SD 1.4 for unsafe concept removal and then expand our evaluation to more recent SDXL-Turbo and FLUX.1-dev [59]. More details in Appendix A.

Motivation to steer only text-embeddings: We note that we train the k-SAE only once on text embeddings. Our choice was motivated by several reasons: (a) steering only via text embeddings offers more control to end users by having them simply specify the concept they wish to steer via text prompts, (b) most text-to-image model architectures vary in the visual training data and the diffusion model architecture but primarily use a handful of text encoders [60, 61, 62]. Thus, Concept Steerer offers a way to reuse a k-SAE across different generative models that use the same text encoder.

#### 4.1 Steering towards safety

Setup: First, we demonstrate the effectiveness of erasing unsafe concepts using k-SAEs. We use the Inappropriate Image Prompts (I2P) dataset [5] to evaluate our method on steering nudity concepts and the Ring-A-Bell benchmark [63] to assess steering performance on violent content. For comparing different techniques, each image is generated using the exact seed and the CFG scale specified in both datasets. We train a k-SAE using features extracted from the residual stream of the 10th layer (out of 12 layers) of the text encoder in SD 1.4, on the prompts in I2P dataset, and evaluate its performance

2C is defined by any user-provided prompt, e.g., “nudity.”

- Table 1: Attack Success Rate (ASR) of different methods on I2P and various adversarial attack datasets. Lower ASR and FID indicate better performance; higher is better for CLIP. Our method achieves the lowest ASR on I2P and best overall robustness on average (column title: AVG) across 4 datasets (Ring-a-Bell, MMADiffusion, P4D, UnlearnDiffAtk), by effectively removing nudity implicitly embedded in the model. We also show that our model competes well on preserving visual quality (lower FID score) and prompt alignment (higher CLIP score). Bold: best. Underline: second-best. Gray : require training and weight updates, Pink : do not require training but update model weights, Blue : do not require either. Best viewed in color.*Due to the small size of both datasets, the performance gap relative to other models amounts to only a few images (1–7).

RING-A-BELL ↓ MMA-DIFFUSION ↓ P4D ↓ UNLEARNDIFFATK ↓ AVG ↓

COCO

METHOD I2P ↓

K77 K38 K16 AVG FID ↓ CLIP ↑

SD 1.4 [50] 17.8 85.26 87.37 93.68 88.10 95.70 98.70 69.70 87.05 16.71 31.3 ESD [6] 2.87 20.00 29.47 35.79 28.42 12.70 9.27 15.49 16.47 18.18 30.2 SA [7] 2.81 63.15 56.84 56.84 58.94 9.30 47.68 12.68 32.15 25.80 29.7 CA [29] 1.04 86.32 91.69 94.26 90.76 9.90 10.60 5.63 29.22 24.12 30.1 MACE [30] 1.51 2.10 0.00 0.00 0.70 0.50 2.72 2.82 1.68 16.80 28.7 UCE [41] 0.87 10.52 9.47 12.61 10.87 17.70 29.93 9.86 17.09 17.99 30.2 RECE [42] 0.72 5.26 4.21 5.26 4.91 13.30 21.77 5.63 11.40 17.74 30.2 SDID [43] 3.77 94.74 95.79 90.53 93.68 34.20 69.54 30.99 57.10 22.16 31.1 SLD-MAX [5] 1.74 23.16 32.63 42.11 32.63 52.10 35.76 9.15 32.41 28.75 28.4 SLD-STRONG [5] 2.28 56.84 64.21 61.05 60.70 61.30 49.01 18.31 47.33 24.40 29.1 SLD-MEDIUM [5] 3.95 92.63 88.42 91.05 90.70 65.70 68.21 33.10 64.42 21.17 29.8 SD-NP 0.74 17.89 40.42 34.74 31.68 44.44 24.00 7.80 26.98 18.33 30.1 SAFREE [9] 1.45 35.78 47.36 55.78 46.31 36.40 40.82 10.56 33.52 19.32 30.1 TRASCE [10] 0.45 1.05 2.10 2.10 1.75 16.60 3.97 0.70 5.75 17.41 29.9 OURS 0.36 3.16* 8.42* 9.47* 7.02* 6.00 1.99 2.11* 4.28 18.67 30.8

|λ = −0.25<br><br>|
|---|

|λ = −0.2<br><br>|
|---|

|λ = −0.15<br><br>|
|---|

|λ = −0.1<br><br>|
|---|

|λ = 0<br><br>|
|---|

|λ = −0.35<br><br>|
|---|

|λ = −0.3<br><br>|
|---|

|λ = −0.2<br><br>|
|---|

|λ = −0.1<br><br>|
|---|

|λ = 0<br><br>|
|---|

[Figure 48]

[Figure 49]

[Figure 50]

| |
|---|

| |
|---|

Prompt: “female painting, yoga, …, fire smouldering, tattoos, midnight … ”

Prompt: “… future bass girl unwrapped smooth body … front and side view …”

(a) (b)

- Figure 4: Qualitative examples from the I2P dataset with FLUX. Our method is model-agnostic and can be applied to both U-Net-based SD 1.4 and SDXL-Turbo, as well as DiT-based FLUX.

| |
|---|

on both the I2P and Ring-A-Bell datasets. As in prior works [10, 63], we use nudity concept Cnudity as “naked, nude, bare, exposed, stripped, topless, male genitalia, penis, buttocks” and use a slightly

modified version for the violent concept Cviolence as “violence, blood”. We set steering strength λ = −0.5 for I2P dataset and λ = −0.7 for adversarial datasets including violent concept.

Evaluation metrics: To quantify the impact of our method on generation quality, we use FID [64] and CLIP score [65, 60] on the COCO-30k dataset, evaluating 10k generated samples. We report Attack Success Rate (ASR), i.e., the percentage of generated images containing nudity or violence as a measure of how well a model reduces unsafe content generation. To this end, we use the NudeNet [66] with a threshold of 0.45 and Q16 violence detector [67], following prior work [10].

Baselines: We compare our method against inference-based approaches that do not require training or weight updates to the generative model, including SDID [43], SLD [5], SD with negative prompt (SDNP), SAFREE [9], and TraSCE [10]. Additionally, we evaluate our method against training-based approaches, including ESD [6], FMN [15], CA [29], MACE [30], and SA [7], as well as approaches that require no training but involve weight updates, such as UCE [41] and RECE [42]. We follow publicly available implementations exactly. More implementation details are in Appendix A.

- 4.1.1 Steering nudity concept As shown in Table 1, our approach achieves state-of-the-art performance in steering unsafe concepts, yielding the lowest ASR (0.36) on the I2P dataset and surpassing the previous best method. Notably, our approach even outperforms both training-based methods [6, 7, 29, 30] and weight-updating methods [41, 42], underscoring the effectiveness of our method. Furthermore, our method achieved one of the highest prompt-image correspondences, as indicated by the CLIP score on the COCO dataset (30.8), closely following the original SD 1.4 model (31.3). This is demonstrated in Fig. 3 (a) and (b), where previous methods sometimes generate unrelated images when the prompt triggers unsafe content. By contrast, our method successfully removes nudity while preserving the overall structure and maintains alignment with the input prompt. Moreover, as shown in Fig. 4, we demonstrate that our method is not limited to a single generative architecture and can also steer the DiT-based [68] FLUX [59] model in a model-agnostic manner.

- Table 2: Attack Success Rate (ASR) on different methods on the Ring-A-Bell-Union (Violence) dataset. Lower values indicate better performance. Our method demonstrates competitive performance without compromising generation quality, as indicated by the FID scores in Table 1. Bold: best. Underline: second-best. Gray : require training and weight updates, Pink : do not require training but update model weights, Blue : do not

require either. Best viewed in color.

SD 1.4 [50] ESD [6] FNM [15] CA [29] UCE [41] RECE [42] SLD-MAX [5] SLD-STRONG [5] SLD-MEDIUM [5] SD-NP TRASCE [10] OURS RING-A-BELL-UNION (VIOLENCE) ↓ 99.6 86.0 98.8 100.0 89.8 89.2 40.4 80.4 97.2 94.8 72.4 43.7

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

C=“Minimal-

C=“Zoom-in,

magnify”

ist”

Prompt: “geodesic landscape, john chamberlain, christopher balaskas, tadao ando, 4 k, ”

- Figure 5: Photographic style manipulation of SD 1.4 for the given prompt “geodesic landscape, john chamberlain, christopher balaskas, tadao ando, 4 k,” where concept prompts are “minimalist” (Left) and “zoomin, magnify” (Right), respectively. On the left, the image is manipulated towards a maximalist style as λ → −1, while it adopts a minimalist style as λ → 1. Similarly, on the right, the image appears zoomed out and becomes blurred as λ → −1, whereas it becomes zoomed in and clearer as λ → 1.

#### 4.1.2 Steering violence concept

We also evaluate our method’s performance in suppressing violent content generation, as presented in Table 2. Note that FID [64] and CLIP score [65] are computed on COCO dataset [69] for a given model, so the scores reported in Table 1 hold true for the same models in Table 2. As shown in Fig. 3 (c), our method effectively reduces the generation of violent content compared to existing training-based and weight-update-based methods. Although SLD-Max [5] achieves slightly better performance than ours, it significantly degrades overall image quality, yielding an FID of 28.75 compared to 18.67 for our approach (Table 1, more visual examples in Appendix Fig. 11).

#### 4.2 Steering of photographic styles and object attributes

Setup: In this section, we demonstrate the effectiveness of steering photographic styles and object attributes. We train a k-SAE using features extracted from the residual stream of the 11th (out of 12) layer of the text encoder in SD 1.4. To observe the effect of photographic style changes, we designed a dataset dedicated to 40 photographic styles, including black-and-white, HDR, minimalist, etc. For each class, we generated 100 prompts, totaling around 4000 prompts, by querying ChatGPT. We also experiment with SDXL-Turbo, where we train using features from both of its text encoders:11th (out of 12) and 29th (out of 32) layers with prompts from I2P dataset.

As shown in Fig. 5 we can adjust its photographic style, including “zoom-in” and “minimalist.” In

- Fig. 6, we compare our results with Concept Sliders [18] on SDXL-Turbo where Concept Sliders train separate models for each weather condition style. Remarkably, our method can effectively steer concepts like weather conditions and photographic styles. We note that I2P dataset in addition to the semantic concepts such as nudity and violence, also had descriptors about general photographic styles such as “full shot” or seasons “winter.” We believe that k-SAE internalized these concepts offering us a powerful tool to surgically steer them. This powerful result highlights the generalizable capability of k-SAEs to learn diverse monosemantic concepts. This is corroborated by our results in
- Fig. 7 (a), where we show that our method can manipulate image compositions, changing a close-up image of a dog into a “full shot” of a dog while preserving the appearance of its head part.

Finally, we use the same k-SAE to effectively manipulate object attributes. Here, we inject a concept for an object present in the image, such as “blue [object].” We note that the resulting generations preserve most of the original content while successfully injecting the desired concept (Fig. 7 (b)). These results demonstrate the universal applicability of a k-SAE without the need to train separate adapters [18] or training data [43] for each concept to control.

#### 4.3 Robustness to adversarial prompt manipulation

Next, we demonstrate the robustness of our method against adversarial prompts on four datasets: red-teaming approaches like Ring-A-Bell [63], P4D [70], and attack frameworks like MMADiffusion [71] and UnlearnDiffAtk [72]. Adversarial prompts often consist of several non-English phrases or nonsensical text fragments that lack semantic meaning, but fool the underlying generative

SDXL-Turbo

Concept Sliders Ours

SDXL-Turbo Concept Sliders Ours

|=𝐶“Winter”<br><br>|
|---|

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

=𝐶“Lowlight”

=𝐶“Winter”

Prompt: “A photo of a tree with a bench, realistic, 8k” Prompt: “A photo of a forest, realistic, 8k”

#### Figure 6: Qualitative

o. Concept Sliders train specific sliders (e.g., winter, dark), whereas ours trains a k-SAE once for multiple concepts. Left: “tree with bench” with “winter” concept. Right: “forest” with “low light.” Our method removes leaves and applies a low-light effect.

s

weather Concept

|comparisonsλ = −0.4<br><br>|
|---|

|λwith= −0.3<br><br>|
|---|

|λSliders= −0.on1<br><br>|
|---|

|SDXL-Turbo.λ = 0<br><br>|
|---|

λ = −0.2

[Figure 63]

[Figure 64]

(a) (b)

##### Figure 7: Concept manipulation using SDXL-Turbo. (a) Image composition manipulation for the prompt “A dog” with the concept prompt “Full shot.” As λ → 1, the image shifts from a close-up to a full-body view. (b) Object attribute manipulation for the prompts “A car,” where the concept prompts are “A blue car.” By adjusting λ, our method transitions the image toward the desired concept.

models to produce unsafe content. We follow the same setup as in Sec. 4.1 and use a k-SAE trained on I2P prompts.

As shown in Table 1, our method achieves the best overall robustness on average across all datasets, significantly outperforming the most recent works TraSCE [10] by 1.23% and SAFREE [9] by 20.01%. Specifically, for the MMA-Diffusion and P4D datasets, our method achieves state-ofthe-art results with improvements of 10.60% and 1.98%, respectively. This demonstrates that our method performs very well and can implicitly identify monosemantic interpretable directions for “nudity” within the latent space of adversarial prompts. Notably, our method outperforms RECE [42] specifically designed for tackling adversarial prompts by 4.48%. For other datasets, our method ranks second-best or performs comparably to the best ones. We note that k-SAE is trained on text embeddings from I2P prompts to learn unsafe concepts and is not exposed to adversarial datasets. Remarkable performance on adversarial datasets demonstrate that k-SAE generalizes well to unseen prompts, even without exposure to prompt embeddings from different distributions, as also observed in Sec. 4.2. We reiterate that once a k-SAE is trained on unsafe concepts, our method does not require retraining and can be applied at test-time.

- 4.4 Efficiency of Concept Steerer Table 3: Model Efficiency Comparison. Inference time (s/sample) on 150 prompts from the P4D dataset using one L40S GPU. Lower is better.

Method Time ↓ SD 1.4 [50] 3.02 SLD-Max [5] 8.59 SAFREE [9] 4.24 TraSCE [10] 15.62 Ours 3.16

As shown in Table 3, our method achieves the fastest inference time among all other inferencebased approaches, with only a 0.14 sec./sample overhead on a single L40S GPU compared to the original SD 1.4. We highlight that our method is approximately 5x faster than the previous stateof-the-art [10] in unsafe concept removal.

- 4.5 Analysis of Concept Steerer

Below, we report the impact of different design choices of Concept Steerer, taking nudity removal as a sample task.

Effect of concept steering on visual quality: To evaluate this, we conduct a user study using 50 randomly selected safe images generated by the original SD 1.4 model and nudity-steered images produced by applying our method on SD 1.4. 3 We followed the setup described in Sec 4.1. The study involved 22 voluntary participants, who were shown pairs of images in a randomized order and were asked to select the image they preferred most based purely on overall visual quality. 44.7% of

3In our study, visual quality refers to how photo-realistic, visually clear, and pleasing an image is as perceived by the user. Interface screenshot in Appendix.

Table 4: Attack Success Rate (ASR) on the I2P dataset. (a) ASR when representations from different encoder layers are used to train k-SAE. The 10th layer yields the lowest ASR, indicating that this layer captures most information about nudity concept. k-SAE expansion factor = 4, hidden neurons (n) = 3072. (b) ASR for different expansion factors of k-SAE trained on text embeddings extracted from the 10th layer of the I2P prompts. An expansion factor of 4 yields the lowest ASR, indicating its efficacy for steering. (c) ASR for different values of λ of k-SAE with an expansion factor of 4 trained on 10th-layer text embeddings. λ = −0.5 yields the lowest ASR.

(a) LAYER ASR ↓

(b)

###### (c) λ ASR ↓

###### EXPANSION FACTOR CAPACITY ASR ↓

4 3072 0.36 8 6144 0.51 16 32 24576 0.49 64 49152 0.53

12 1.02 10 0.36 8 0.45 6 1.72 4 3.85

- −0.1 2.59
- −0.2 1.23
- −0.3 0.87
- −0.4 0.60
- −0.5 0.36

|λ = −0.4<br><br>|
|---|

|λ = −0.3<br><br>|
|---|

|λ12288= −0.1<br><br>|
|---|

|λ0.47= 0<br><br>|
|---|

λ = −0.2

|λ = −0.5<br><br>|
|---|

|λ = −0.4<br><br>|
|---|

|λ = −0.3<br><br>|
|---|

|λ = −0.2<br><br>|
|---|

|λ = −0.1<br><br>|
|---|

|λ = 0<br><br>|
|---|

|λ = −0.5<br><br>|
|---|

|λ = −0.4<br><br>|
|---|

|λ = −0.3<br><br>|
|---|

|λ = −0.2<br><br>|
|---|

|λ = −0.1<br><br>|
|---|

|λ = 0<br><br>|
|---|

[Figure 65]

[Figure 66]

| |
|---|

| |
|---|

| |
|---|

(a) (b)

- Figure 8: Effect of steering strength parameter (λ) on the I2P dataset while we steer nudity. Notice how as λ → −0.5, the presence of nudity disappears completely.

users preferred images produced by concept steering, while 44.9% preferred images from SD 1.4, indicating that participants expressed an almost equal preference for both generations.4 This is a crucial finding because it shows that our method does not deteriorate visual quality from the base model but offers the additional benefit of controllability.

We examine the effect of the choice of the text encoder’s layer on the semantic information captured in k-SAE and thereby concept steering. From Table 4a, we note that representations from later layers are more effective at removing nudity (lower ASR) than earlier layers. We believe that earlier layers capture more low-level semantic information, thus high-level concepts such as nudity are better captured in the later layers, making them suitable candidates for steering. Similar observations were reported in [73].

We investigate the effect of k-SAE capacity determined by different expansion factors on steering results. From the ASR results in Table 4b, we note that the performance differences between capacities is relatively minor, and an expansion factor of 4 empirically is most effective in removing nudity.

We study the effect of steering strength λ in Table 4c, showing that decreasing its value enables more effective removal of nudity from a greater number of images. As shown in Fig. 8 (a), setting λ = −0.1 effectively removes the nudity in that example. However, smaller λ values (λ = −0.5) lead to a more complete removal, as demonstrated in Fig. 8 (b) and lowest ASR scores in Table 4c.

### 5 Discussion and Future Work

We propose a novel framework leveraging k-SAEs to enable efficient and interpretable concept manipulation in diffusion models. Once trained, k-SAE serves as a Concept Steerer to precisely control specific visual concepts (e.g., nudity, violence, camera angles etc.). Our extensive experiments demonstrate that our approach is very simple, does not compromise the visual quality, and is robust to adversarial prompt manipulations. Our current approach steers concepts by manipulating representations only from the text encoder. However, this design does not fully leverage the rich information encoded in the visual or the multi-modal embeddings of the diffusion model. Future work will explore interpreting and steering multi-modal embeddings to enable finer-grained control in generations. We note that sometimes higher steering strength values (λ) impact object and identity consistency, e.g., the style of the bench in Fig. 6. Exploring steering mechanisms where users select regions in an image to apply concept steering locally is a worthy direction to explore.

410.4% selected "none of these" when they did not prefer either image.

### References

- [1] Marian Mazzone and Ahmed Elgammal. Art, creativity, and the potential of artificial intelligence. In Arts, 2019.
- [2] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In CVPR, 2023.
- [3] Amirhossein Kazerouni, Ehsan Khodapanah Aghdam, Moein Heidari, Reza Azad, Mohsen Fayyaz, Ilker Hacihaliloglu, and Dorit Merhof. Diffusion models in medical imaging: A comprehensive survey. In MedIA, 2023.
- [4] Javier Rando, Daniel Paleka, David Lindner, Lennart Heim, and Florian Tramèr. Red-teaming the stable diffusion safety filter. arXiv preprint arXiv:2210.04610, 2022.
- [5] Patrick Schramowski, Manuel Brack, Björn Deiseroth, and Kristian Kersting. Safe latent diffusion: Mitigating inappropriate degeneration in diffusion models. In CVPR, 2023.
- [6] Rohit Gandikota, Joanna Materzynska, Jaden Fiotto-Kaufman, and David Bau. Erasing concepts from diffusion models. In ICCV, 2023.
- [7] Alvin Heng and Harold Soh. Selective amnesia: A continual learning approach to forgetting in deep generative models. In NeurIPS, 2023.
- [8] Xinfeng Li, Yuchen Yang, Jiangyi Deng, Chen Yan, Yanjiao Chen, Xiaoyu Ji, and Wenyuan Xu. Safegen: Mitigating sexually explicit content generation in text-to-image models. arXiv preprint arXiv:2404.06666, 2024.
- [9] Jaehong Yoon, Shoubin Yu, Vaidehi Patil, Huaxiu Yao, and Mohit Bansal. Safree: Trainingfree and adaptive guard for safe text-to-image and video generation. arXiv preprint arXiv:2410.12761, 2024.
- [10] Anubhav Jain, Yuya Kobayashi, Takashi Shibuya, Yuhta Takida, Nasir Memon, Julian Togelius, and Yuki Mitsufuji. Trasce: Trajectory steering for concept erasure. arXiv preprint arXiv:2412.07658, 2024.
- [11] Hanqi Yan, Yanzheng Xiang, Guangyi Chen, Yifei Wang, Lin Gui, and Yulan He. Encourage or inhibit monosemanticity? revisit monosemanticity from a feature decorrelation perspective. arXiv preprint arXiv:2406.17969, 2024.
- [12] Alireza Makhzani and Brendan Frey. K-sparse autoencoders. arXiv preprint arXiv:1312.5663, 2013.
- [13] Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E Burke, Tristan Hume, Shan Carter, Tom Henighan, and Christopher Olah. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread, 2023. https://transformercircuits.pub/2023/monosemantic-features/index.html.
- [14] Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models. arXiv preprint arXiv:2309.08600, 2023.
- [15] Gong Zhang, Kai Wang, Xingqian Xu, Zhangyang Wang, and Humphrey Shi. Forget-me-not: Learning to forget in text-to-image diffusion models. In CVPR, 2024.
- [16] Piero Esposito, Parmida Atighehchian, Anastasis Germanidis, and Deepti Ghadiyaram. Mitigating stereotypical biases in text to image generative systems. arXiv preprint arXiv:2310.06904, 2023.
- [17] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

- [18] Rohit Gandikota, Joanna Materzy´nska, Tingrui Zhou, Antonio Torralba, and David Bau. Concept sliders: Lora adaptors for precise control in diffusion models. In ECCV, 2025.
- [19] Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score: Better aligning text-to-image models with human preference. In ICCV, 2023.
- [20] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In CVPR, 2024.
- [21] Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Universal guidance for diffusion models. In CVPR, 2023.
- [22] Raghav Singhal, Zachary Horvitz, Ryan Teehan, Mengye Ren, Zhou Yu, Kathleen McKeown, and Rajesh Ranganath. A general framework for inference-time scaling and steering of diffusion models. arXiv preprint arXiv:2501.06848, 2025.
- [23] Masatoshi Uehara, Yulai Zhao, Chenyu Wang, Xiner Li, Aviv Regev, Sergey Levine, and Tommaso Biancalani. Reward-guided controlled generation for inference-time alignment in diffusion models: Tutorial and review. arXiv preprint arXiv:2501.09685, 2025.
- [24] Nick Stracke, Stefan Andreas Baumann, Joshua Susskind, Miguel Angel Bautista, and Björn Ommer. Ctrloralter: Conditional loradapter for efficient 0-shot control and altering of t2i models. In ECCV, 2025.
- [25] Deepak Sridhar and Nuno Vasconcelos. Prompt sliders for fine-grained control, editing and erasing of concepts in diffusion models. arXiv preprint arXiv:2409.16535, 2024.
- [26] Manuel Brack, Patrick Schramowski, Felix Friedrich, Dominik Hintersdorf, and Kristian Kersting. The stable artist: Steering semantics in diffusion latent space. arXiv preprint arXiv:2212.06013, 2022.
- [27] Manuel Brack, Felix Friedrich, Dominik Hintersdorf, Lukas Struppek, Patrick Schramowski, and Kristian Kersting. Sega: Instructing text-to-image models using semantic guidance. In NeurIPS, 2023.
- [28] Pau Rodriguez, Arno Blaas, Michal Klein, Luca Zappella, Nicholas Apostoloff, Marco Cuturi, and Xavier Suau. Controlling language and diffusion models by transporting activations. arXiv preprint arXiv:2410.23054, 2024.
- [29] Nupur Kumari, Bingliang Zhang, Sheng-Yu Wang, Eli Shechtman, Richard Zhang, and Jun-Yan Zhu. Ablating concepts in text-to-image diffusion models. In ICCV, 2023.
- [30] Shilin Lu, Zilan Wang, Leyang Li, Yanzhu Liu, and Adams Wai-Kin Kong. Mace: Mass concept erasure in diffusion models. In CVPR, 2024.
- [31] Chongyu Fan, Jiancheng Liu, Yihua Zhang, Eric Wong, Dennis Wei, and Sijia Liu. Salun: Empowering machine unlearning via gradient-based weight saliency in both image classification and generation. arXiv preprint arXiv:2310.12508, 2023.
- [32] Anh Bui, Long Vuong, Khanh Doan, Trung Le, Paul Montague, Tamas Abraham, and Dinh Phung. Erasing undesirable concepts in diffusion models with adversarial preservation. arXiv preprint arXiv:2410.15618, 2024.
- [33] Myeongseob Ko, Henry Li, Zhun Wang, Jonathan Patsenker, Jiachen Tianhao Wang, Qinbin Li, Ming Jin, Dawn Song, and Ruoxi Jia. Boosting alignment for post-unlearning text-to-image generative models. In NeurIPS, 2024.
- [34] Seunghoo Hong, Juhun Lee, and Simon S Woo. All but one: Surgical concept erasing with model preservation in text-to-image diffusion models. In AAAI, 2024.
- [35] Yong-Hyun Park, Sangdoo Yun, Jin-Hwa Kim, Junho Kim, Geonhui Jang, Yonghyun Jeong, Junghyo Jo, and Gayoung Lee. Direct unlearning optimization for robust and safe text-to-image models. arXiv preprint arXiv:2407.21035, 2024.

- [36] Mengyao Lyu, Yuhong Yang, Haiwen Hong, Hui Chen, Xuan Jin, Yuan He, Hui Xue, Jungong Han, and Guiguang Ding. One-dimensional adapter to rule them all: Concepts diffusion models and erasing applications. In CVPR, 2024.
- [37] Chi-Pin Huang, Kai-Po Chang, Chung-Ting Tsai, Yung-Hsuan Lai, Fu-En Yang, and YuChiang Frank Wang. Receler: Reliable concept erasing of text-to-image diffusion models via lightweight erasers. In ECCV, 2024.
- [38] Sanghyun Kim, Seohyeon Jung, Balhae Kim, Moonseok Choi, Jinwoo Shin, and Juho Lee. Safeguard text-to-image diffusion models with human feedback inversion. In ECCV, 2024.
- [39] Tianqi Chen, Shujian Zhang, and Mingyuan Zhou. Score forgetting distillation: A swift, datafree method for machine unlearning in diffusion models. arXiv preprint arXiv:2409.11219, 2024.
- [40] Ruchika Chavhan, Da Li, and Timothy Hospedales. Conceptprune: Concept editing in diffusion models via skilled neuron pruning. arXiv preprint arXiv:2405.19237, 2024.
- [41] Rohit Gandikota, Hadas Orgad, Yonatan Belinkov, Joanna Materzy´nska, and David Bau. Unified concept editing in diffusion models. In WACV, 2024.
- [42] Chao Gong, Kai Chen, Zhipeng Wei, Jingjing Chen, and Yu-Gang Jiang. Reliable and efficient concept erasure of text-to-image diffusion models. In ECCV, 2025.
- [43] Hang Li, Chengzhi Shen, Philip Torr, Volker Tresp, and Jindong Gu. Self-discovering interpretable diffusion latent directions for responsible text-to-image generation. In CVPR, 2024.
- [44] Hugo Fry. Towards multimodal interpretability: Learning sparse interpretable features in vision transformers. LessWrong, 2024. https://www.lesswrong.com/posts/ bCtbuWraqYTDtuARg/towards-multimodal-interpretability-learning-sparse.
- [45] Gytis Daujotas. Interpreting and steering features in images. LessWrong,

2024. https://www.lesswrong.com/posts/Quqekpvx8BGMMcaem/interpreting-andsteering-features-in-images.

- [46] Mateusz Pach, Shyamgopal Karthik, Quentin Bouniot, Serge Belongie, and Zeynep Akata. Sparse autoencoders learn monosemantic features in vision-language models. arXiv preprint arXiv:2504.02821, 2025.
- [47] Hanqi Yan, Xiangxiang Cui, Lu Yin, Paul Pu Liang, Yulan He, and Yifei Wang. The multifaceted monosemanticity in multimodal representations. arXiv preprint arXiv:2502.14888, 2025.
- [48] Dahye Kim, Xavier Thomas, and Deepti Ghadiyaram. Revelio: Interpreting and leveraging semantic information in diffusion models. arXiv preprint arXiv:2411.16725, 2024.
- [49] Viacheslav Surkov, Chris Wendler, Mikhail Terekhov, Justin Deschenaux, Robert West, and Caglar Gulcehre. Unpacking sdxl turbo: Interpreting text-to-image models with sparse autoencoders. arXiv preprint arXiv:2410.22366, 2024.
- [50] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.
- [51] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.
- [52] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. In NeurIPS, 2022.
- [53] Andrew Ng et al. Sparse autoencoder. CS294A Lecture notes, 2011.
- [54] Robert Tibshirani. Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society Series B: Statistical Methodology, 1996.

- [55] Leo Gao, Tom Dupré la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, and Jeffrey Wu. Scaling and evaluating sparse autoencoders. arXiv preprint arXiv:2406.04093, 2024.
- [56] Diederik P Kingma. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
- [57] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In ECCV, 2025.
- [58] Lee Sharkey, Dan Braun, and Beren Millidge. Taking features out of superposition with sparse autoencoders. AI Alignment Forum, 2023. https://www.alignmentforum.org/ posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-ofsuperposition.
- [59] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2023.
- [60] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021.
- [61] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In CVPR, 2023.
- [62] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. In JMLR, 2020.
- [63] Yu-Lin Tsai, Chia-Yi Hsu, Chulin Xie, Chih-Hsun Lin, Jia-You Chen, Bo Li, Pin-Yu Chen, Chia-Mu Yu, and Chun-Ying Huang. Ring-a-bell! how reliable are concept removal methods for diffusion models? arXiv preprint arXiv:2310.10012, 2023.
- [64] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. NeurIPS, 2017.
- [65] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718, 2021.
- [66] Praneeth Bedapudi. Nudenet: Neural nets for nudity classification, detection and selective censoring. https://github.com/notai-tech/nudenet, 2019.
- [67] Patrick Schramowski, Christopher Tauchmann, and Kristian Kersting. Can machines help us answering question 16 in datasheets, and in turn reflecting on inappropriate content? In FAccT, 2022.
- [68] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023.
- [69] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014.
- [70] Zhi-Yi Chin, Chieh-Ming Jiang, Ching-Chun Huang, Pin-Yu Chen, and Wei-Chen Chiu. Prompting4debugging: Red-teaming text-to-image diffusion models by finding problematic prompts. arXiv preprint arXiv:2309.06135, 2023.
- [71] Yijun Yang, Ruiyuan Gao, Xiaosen Wang, Tsung-Yi Ho, Nan Xu, and Qiang Xu. Mma-diffusion: Multimodal attack on diffusion models. In CVPR, 2024.
- [72] Yimeng Zhang, Jinghan Jia, Xin Chen, Aochuan Chen, Yihua Zhang, Jiancheng Liu, Ke Ding, and Sijia Liu. To generate or not? safety-driven unlearned diffusion models are still easy to generate unsafe images... for now. In ECCV, 2025.

- [73] Michael Toker, Hadas Orgad, Mor Ventura, Dana Arad, and Yonatan Belinkov. Diffusion lens: Interpreting text encoders in text-to-image pipelines. arXiv preprint arXiv:2403.05846, 2024.
- [74] A Paszke. Pytorch: An imperative style, high-performance deep learning library. arXiv preprint arXiv:1912.01703, 2019.
- [75] Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. Scaling up test-time compute with latent reasoning: A recurrent depth approach. arXiv preprint arXiv:2502.05171, 2025.
- [76] Yossi Gandelsman, Alexei A Efros, and Jacob Steinhardt. Interpreting clip’s image representation via text-based decomposition. arXiv preprint arXiv:2310.05916, 2023.
- [77] nostalgebraist. Interpreting gpt: The logit lens. https://www.lesswrong.com/posts/ AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens, 2020.
- [78] StabilityAI. Introducing SDXL Turbo: A Real-Time Text-to-Image Generation Model, 2024.

## Appendix

### A Implementation Details

Training k-SAE with FLUX: For FLUX.1-dev [59] visualization, we train k-SAE using features extracted from the residual stream of the 23rd (out of 24) layer of the T5-XXL text encoder on prompts from the I2P dataset. The k-SAE is trained with k = 64 and an expansion factor of 16, resulting in a total hidden size dimension n = 65536.

Text encoders of diffusion models: We extract text embeddings for k-SAE from CLIP ViT-L/14 [60] for SD 1.4, OpenCLIP-ViT/G [61] and CLIP-ViT/L for SDXL-Turbo, and T5-XXL [62] for FLUX.1dev.

Baselines: We compare our method against SDID [43], SLD [5], SD with negative prompt (SD-NP), SAFREE [9], TraSCE [10], ESD [6], FMN [15], CA [29], MACE [30], SA [7], UCE [41], and RECE [42]. We reported the performance of prior works with publicly available code. Whenever pretrained weights were provided, we directly used them, otherwise, we followed the publicly available implementations including hyperparameters as closely as possible. We also reported performance of several other methods as provided in the TraSCE [10], and we followed their experimental setup.

Experimental setup: When steering concepts, we use a slightly modified encoder of k-SAE without the TopK activation function, as the TopK function clamps some possible important information that is critical for both maintaining visual quality and effective concept removal. All experiments are conducted using Python 3.10.14 and PyTorch 2.5.1 [74], on a single NVIDIA L40 GPU with 48GB of memory.

### B More Details of the Benchmarks

We evaluate our method for unsafe concept removal tasks on five publicly available inappropriate or adversarial prompts datasets following prior work [10]. I2P5 [5], Ring-A-Bell6 [63], P4D7 [70], MMA-Diffusion8 [71], and UnlearnDiffAtk9 [72]. I2P contains 4703 real user prompts that are likely to produce inappropriate images. Ring-A-Bell consists of two inappropriate categories: nudity and violence. For nudity, it contains 95 unsafe prompts for each split (K77, K38, and K16). For violence, we use the Ring-A-Bell Union dataset, which includes 750 prompts. P4D contains 151 unsafe prompts generated by white-box attacks on the ESD [6] and SLD [5]. MMA-Diffusion contains 1000 strong adversarial prompts generated via a black-box attack. UnlearnDiffAtk contains 142 adversarial prompts generated using white-box adversarial attacks.

### C Additional Qualitative Results

In this section, we provide additional qualitative results.

Steering nudity concept on inappropriate dataset: Figure 9 presents additional qualitative results using FLUX on prompts from I2P dataset. Our method effectively removes the abstract concept of nudity in DiT-based FLUX in an out-of-the-box manner.

Steering nudity concept on adversarial dataset: Figure 10 presents qualitative comparisons with different methods on the P4D dataset. Since P4D contains adversarial prompts specifically designed to challenge generative models, previous methods either fail by generating unsafe images or produce unrelated images as a defense mechanism when the prompt triggers to generate unsafe content (middle

- 5https://huggingface.co/datasets/AIML-TUDA/i2p
- 6https://huggingface.co/datasets/Chia15/RingABell-Nudity
- 7https://huggingface.co/datasets/joycenerd/p4d
- 8https://huggingface.co/datasets/YijunYang280/MMA-Diffusion-NSFW-adv-prompts-

benchmark

- 9https://github.com/OPTML-Group/Diffusion-MU-Attack/blob/main/prompts/nudity.csv

row). In contrast, our method successfully removes nudity while preserving the overall structure and maintaining alignment with the input prompt, even when the prompt itself is nonsensical (first and last row).

Steering nudity concept on COCO dataset: To evaluate the effect of removing the nudity concept during the generation process, we apply different unsafe concept removal approaches to a safe dataset, i.e., COCO [69]. Figure 11 presents qualitative comparisons across methods. Our method generates images that are qualitatively similar to those from the original SD 1.4, even after removing the nudity concept, while preserving photo-realism and maintaining alignment with the input prompt. It achieves competitive results compared to other approaches, including SLD-Max [5], SAFREE [9], and TraSCE [10]. This highlights our method’s ability to selectively remove targeted concepts during generation without harming overall image quality or semantic fidelity.

Steering violent concept: Figure 12 presents qualitative examples on the Ring-A-Bell dataset for violent concept removal. Our method effectively removes the abstract concept of violence by eliminating visual cues such as blood and firearms.

Steering photographic styles: Figure 13 presents qualitative examples of photographic style manipulations in SD 1.4, including “HDR,” “Black and White,” “Sepia Tone,” and “Astrophotography.” We note that as λ → 0.5, the generated image gradually transitions to the desired concept.

Steering object attributes: Figure 14 presents qualitative examples of object attributes manipulations in SDXL-Turbo. Given a prompt, we inject a concept for an object present in the image, such as “an orange cake” and “a chocolate cake.” We note that the resulting generations preserve most of the original content while successfully injecting the desired concept.

### D Broader Impacts

As text-to-image models are increasingly integrated into high-stakes applications, discouraging unsafe generations is of paramount significance. This work presents an effective approach for identifying and suppressing unsafe concept directions across various generative models. By improving the controllability and reliability of generative models, our method advances the development of safer AI systems, facilitating their responsible deployment in real-world applications.

### E User Study Interface

We provide the full instructions and a screenshot of the interface used in the user study described in Sec. 4.5. Participants were instructed as shown in Table 6, and the interface is illustrated in Fig. 17.

### F Effect of Layer Selection on Steering

In Sec. 4.5, we studied the effect of layer selection on steering for the nudity concept. To examine whether this trend holds for a different concept, we analyze the effect of layer selection on steering for the violence concept. We report the Attack Success Rate (ASR) using representations from different encoder layers on the Ring-A-Bell-Union (Violence) dataset [63]. As shown in the Table 5, the 10th layer performs best (i.e., lowest ASR), although the difference with the 12th layer is marginal. Given the above results and those from Table 4 (a), we note that text representations of both violence and nudity concepts exhibit a similar trend, i.e., representations from later layers are more effective for steering compared to earlier layers. We attribute this to later layers encoding more complex, semantically rich information compared to earlier layers as also observed in [73, 75, 76, 77].

### G Impact of Different Steering Variants

In this section, we analyze the effect of alternative steering variants to elucidate the contribution of key components in our approach. Recall that our proposed formulation is xsteered = x+Wdec(λ∗ENC(xC)) as in Eq. 5 (L160-161). We stress that k-SAE is non-linear due to the use of ReLU and Top-K activation functions, as noted in Eq. 1 and L142-144.

Table 5: Attack Success Rate (ASR) when representations from different encoder layers are used to train k-SAE on the Ring-A-Bell-Union (Violence) dataset. The 10th layer yields the lowest ASR, indicating that this layer captures most information about nudity concept. k-SAE expansion factor = 4, hidden neurons (n) = 3072.

###### LAYERS ASR ↓

12 43.86 10 43.73 8 83.86 6 82.13 4 85.46

To test the extent of non-linearity and its contribution to the learning of monosemantic concepts, we make an assumption that all components are linear. Under this assumption, xsteered simplifies to x + λ ∗ Wdec(ENC(xC)) = x + λ ∗ xC + Error, where Error is the residual error needed to reconstruct xC.

In fact, when xC is passed through the SAE, it can be approximated as xC ≈ DEC(ENC(xC))+Error, where DEC(x) = Wdecx+bpre as defined in Eq. 2. Substituting this in, we get xC ≈ Wdec(ENC(xC))+ bpre+Error. Multiplying both sides by λ, we obtain λ∗xC ≈ λ∗Wdec(ENC(xC))+λ∗bpre+λError. Under the linearity assumption, substituting back into the above equation gives: xsteered = x + λ ∗ Wdec(ENC(xC)) ≈ x + λ ∗ xC − λ ∗ bpre − λError.

To isolate the contribution of each component, we compare the following variants: (1) xsteered = x + λ ∗ xC, (2) xsteered = x + λ ∗ xC − λ ∗ bpre, and (3) xsteered = x + λ ∗ xC − λ ∗ Error. These variants are derived from the approximate linear decomposition of the full formulation and allow us to evaluate the individual roles of directly injecting concept embedding xC, pre-encoder bias term λ ∗ bpre, and residual error λ ∗ Error, respectively. As shown in Fig. 18, these simplified variants fail to fully suppress the unsafe concept and often degrade visual quality, underscoring the effectiveness of our proposed steering formulation. This analysis reveals that simply injecting the concept embedding xC, or removing the pre-encoder bias term or residual error, is insufficient for reliable concept erasure, and cannot be captured by a linear approximation, underscoring the necessity of our full formulation.

### H License Information

We provide the license information for all assets used in this work. For additional details, please refer to the corresponding links.

- • SD 1.4 [50]: https://huggingface.co/spaces/CompVis/stable-diffusion-license
- • SDXL-Turbo [78]: https://huggingface.co/stabilityai/sdxl-turbo/blob/main/LICENSE.md
- • FLUX.1 [dev] [59]: https://huggingface.co/black-forest-labs/FLUX.1dev/blob/main/LICENSE.md
- • I2P [5]: https://github.com/ml-research/safe-latent-diffusion?tab=MIT1-ov-file#readme
- • P4D [70]: https://huggingface.co/datasets/choosealicense/licenses/blob/ main/markdown/cc-by-4.0.md
- • Ring-A-Bell [63]: https://github.com/chiayi-hsu/Ring-A-Bell?tab=MIT-1-ov-file
- • MMA-Diffusion [71]: https://github.com/cure-lab/MMA-Diffusion/blob/main/LICENSE
- • UnlearnDiffAtk [72]: https://github.com/OPTML-Group/Diffusion-MUAttack?tab=MIT-1-ov-file#readme
- • COCO [69]: https://github.com/cocodataset/cocoapi/tree/master?tab= License-1-ov-file

[Figure 67]

[Figure 68]

| |
|---|

| |
|---|

Prompt: “excellent framing lighting texture … sunlight illuminates the body as … lovely …”

[Figure 69]

Prompt: “sensual female painting, yoga, pancakes, fire smouldering, tattoos, midnight, dark … ”

[Figure 70]

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

Prompt: “… photorealistic portrait of a woman's pale fat belly by sarah moon …”

- Figure 9: Qualitative example from the I2P dataset with FLUX. Our method is model-agnostic and can be applied to both U-Net-based SD 1.4 and SDXL-Turbo, as well as DiT-based FLUX.

You will see two images side by side. Your task is to choose the image you prefer ( left or right) based on visual quality. In this study , ‘‘visual quality ’’ refers to how realistic , clear , and visually pleasing

an image appears.

Table 6: User study instruction.

P4d 1. 43

2.0 3.138

SD 1.4 SAFREE TraSCE Ours

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

| |
|---|

Remove

| |
|---|

nudity

| |
|---|

| |
|---|

| |
|---|

Prompt: “… waterhouse motive waterhouse venus … madonna ngmale”

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

Remove

nudity

| |
|---|

| |
|---|

| |
|---|

Prompt: “… girl alier roman renaissance nude paintings”

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

Remove

| |
|---|

| |
|---|

| |
|---|

nudity

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Prompt: “phyllis nude bath ccp bathing steamy … painting …”

- Figure 10: Qualitative comparisons of different methods, including TraSCE and SAFREE, on the P4D dataset. The P4D dataset consists of adversarial prompts designed to challenge generative models. Our approach effectively removes the concept of nudity during the generation process, producing safe and semantically meaningful outputs. In contrast, SAFREE fails to generate safe images, while TraSCE sometimes produces unrelated outputs despite the presence of semantically meaningful keywords in given prompts, such as “girl,” “roman,” “renaissance,” and “paintings” (middle row).

SD 1.4 SLD-Max Ours

SAFREE TraSCE

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

325

Prompt: “A bird that is standing on a concrete ledge over the water.”

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

171

Prompt: “Two cats are shown sitting on a toilet.”

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

Prompt: “A red stop sign with a white street sign posted under it.” 228

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

244

Prompt: “A huge group of scattered out sheep are in the field.”

- Figure 11: Qualitative comparisons of different methods on the COCO dataset [69], including SLD-Max [5], SAFREE [9], and TraSCE [10]. After removing the nudity concept, our method produces images similar to SD 1.4 while better preserving prompt-image alignment and maintaining image quality. In contrast, other approaches often fail to preserve alignment and degrade image quality.

SD 1.4 Ours

Remove

violence

Remove

violence

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

1. 487 2. 4383.429 4.4

Remove

violence

Remove

violence

SD 1.4 Ours

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

- Figure 12: Qualitative examples from the Ring-A-Bell dataset. Our method successfully removes the abstract concept of violence, as shown by the absence of blood in the right images. The images are intentionally blurred for display purposes as they are disturbing.

[Figure 111]

C=“Minimal-

ist”

[Figure 112]

C=“Zoom-in,

magnify”

Prompt: “geodesic landscape, john chamberlain, christopher balaskas, tadao ando, 4 k, ”

[Figure 113]

C=“HDR” C=“Black

[Figure 114]

andwhite”

[Figure 115]

C=“Sepia

Tone”

[Figure 116]

photography”

C=“Astro-

Prompt: “geodesic landscape, john chamberlain, christopher balaskas, tadao ando, 4 k”

- Figure 13: Photographic style manipulation of SD 1.4 for the given prompt “geodesic landscape, john chamberlain, christopher balaskas, tadao ando, 4 k, ” where concept prompts are “HDR,” “Black and white,” “Sepia Tone,” and “Astrophotography,” respectively. As λ → 0.5, the generated image gradually transitions to the desired concept.

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

Prompt: “A photo of a cake, 4 k ”

C=“Achoco-

latecake”

C=“Awhite

cake”

C=“Alemon

cake”

C=“Anorange

cake”

- Figure 14: Object attribute manipulation of SDXL-Turbo for the given prompts “A photo of a cake, 4k,” where the concept prompts are “A chocolate cake,” “A white cake,” “A lemon cake,” and “An orange cake,” respectively. By adjusting λ, our method transitions the image toward the desired concept specified by the prompts.

[Figure 121]

[Figure 122]

[Figure 123]

C=“Anorange

tree”

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

C=“Apink

tree”

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

C=“Apurple

tree”

[Figure 132]

[Figure 133]

[Figure 134]

C=“Ayellow

tree”

Prompt: “A tree”

- Figure 15: Object attribute manipulation of SDXL-Turbo for the given prompt “A tree,” where the concept prompts are “An orange tree,” “A pink tree,” “A purple tree,” and “A yellow,” respectively. By adjusting λ, our method transitions the image toward the desired concept specified by the prompts.

Prompt: “A forest path”

C=“Foggy” C=“Golden

hour”

C=“Sunset”

[Figure 135]

[Figure 136]

[Figure 137]

- Figure 16: Photographic style manipulation of SDXL-Turbo for the given prompt “A forest path,” where the concept prompts are “Foggy,” “Golden hour,” and “Sunset,” respectively. By adjusting λ, our method transitions the image toward the desired concept specified by the prompts.

[Figure 138]

Figure 17: User study interface. Our method (left) and SD 1.4 (right).

P4d 1. 43

2.0 3.138

𝑥𝑠𝑡𝑒𝑒𝑟𝑒𝑟𝑑

𝑥𝑠𝑡𝑒𝑒𝑟𝑒𝑟𝑑

𝑥𝑠𝑡𝑒𝑒𝑟𝑒𝑟𝑑

SD 1.4 Ours

= 𝑥 + 𝜆 ∗ 𝑥𝑐 − 𝜆 ∗ 𝐸𝑟𝑟𝑜𝑟

= 𝑥 + 𝜆 ∗ 𝑥𝑐 − 𝜆 ∗ 𝑏𝑝𝑟𝑒

= 𝑥 + 𝜆 ∗ 𝑥𝑐

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

Remove

nudity

3

| |
|---|

Prompt: “… oils dee sian liliatrist iris rk fineart …”

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

Remove nudity

| |
|---|

| |
|---|

| |
|---|

112

Prompt: “jem belated :* william uw aux … tre toppbaroque fineart …”

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

Remove

| |
|---|

nudity

| |
|---|

138

| |
|---|

Prompt: “phyllis nude bath ccp bathing steamy … painting …”

Figure 18: Effect of different steering methods on the P4D dataset. We compare three variants: (1) direct addition of the concept vector, (2) subtraction of the pre-encoder bias term, and (3) subtraction of the residual error. All variants are less effective than our proposed method, leading to either incomplete removal of unsafe content or degraded visual quality. These results highlight the importance of our approach.

