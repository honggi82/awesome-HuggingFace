## PALP: Prompt Aligned Personalization of Text-to-Image Models

Moab Arar*1,2, Andrey Voynov2, Amir Hertz2, Omri Avrahami*2,4, Shlomi Fruchter1, Yael Pritch 2, Daniel Cohen-Or*1,2, Ariel Shamir*2,3

1 Tel-Aviv University, 2 Google Research 3 Reichman University, 4 The Hebrew University of Jerusalem

# arXiv:2401.06105v1[cs.CV]11Jan2024

||[Figure 1]|
|---|
<br><br>[Figure 2]<br><br>[Figure 3]|
|---|

|[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>|
|---|

[Figure 7]

[Figure 8]

[Figure 9]

Figure 1. Prompt aligned personalization allow rich and complex scene generation, including all elements of a condition prompt (right).

### Abstract

existing baselines and state-of-the-art techniques.

### 1. Introduction

Text-to-image models have shown exceptional abilities to generate a diversity of images in various settings (place, time, style, and appearances), such as ”a sketch of Paris on a rainy day” or ”a Manga drawing of a teddy bear at night” [36, 41]. Recently, personalization methods even allow one to include specific subjects (objects, animals, or people) into the generated images [15, 39]. In practice, however, such models are difficult to control and may require significant prompt engineering and re-sampling to create the specific image one has in mind. It is even more acute with personalized models, where it is challenging to include the personal item or character in the image and simultaneously fulfill the textual prompt describing the content and style. This work proposes a method for better personalization and prompt alignment, especially suited for complex prompts.

Content creators often aim to create personalized images using personal subjects that go beyond the capabilities of conventional text-to-image models. Additionally, they may want the resulting image to encompass a specific location, style, ambiance, and more. Existing personalization methods may compromise personalization ability or the alignment to complex textual prompts. This trade-off can impede the fulfillment of user prompts and subject fidelity. We propose a new approach focusing on personalization methods for a single prompt to address this issue. We term our approach prompt-aligned personalization. While this may seem restrictive, our method excels in improving text alignment, enabling the creation of images with complex and intricate prompts, which may pose a challenge for current techniques. In particular, our method keeps the personalized model aligned with a target prompt using an additional score distillation sampling term. We demonstrate the versatility of our method in multi- and single-shot settings and further show that it can compose multiple subjects or use inspiration from reference images, such as artworks. We compare our approach quantitatively and qualitatively with

A key ingredient in personalization methods is finetuning pre-trained text-to-image models on a small set of personal images while relying on heavy regularization to

*Work done while working at Google. †Project page available at https : / / prompt - aligned .

github.io/.

maintain the model’s capacity. Doing so will preserve the model’s prior knowledge and allow the user to synthesize images with various prompts; however, it impairs capturing identifying features of the target subject. On the other hand, persisting on identification accuracy can hinder the promptalignment capabilities. Generally speaking, the trade-off between identity preservation and prompt alignment is a core challenge in personalization methods (see Fig. 2).

Content creators and AI artists frequently have a clear idea of the prompt they wish to utilize. It may involve stylization and other factors that current personalization methods struggle to maintain. Therefore, we take a different approach by focusing on excelling with a single prompt rather than offering a general-purpose method intended to perform well with a wide range of prompts. This approach enables both (i) learning the unique features of the subject from a few or even a single input image and (ii) generating richer scenes that are better aligned with the user’s desired prompt (see Fig. 1).

Our work is based on the premise that existing models possess knowledge of all elements within the target prompt except the new personal subject. Consequently, we leverage the pre-trained model’s prior knowledge to prevent personalized models from losing their understanding of the target prompt. In particular, since we know the target prompt during training, we show how to incorporate score-distillation guidance [34] to constrain the personalized model’s prediction to stay aligned with the pre-trained one. Therefore, we introduce a framework comprising two components: personalization, which teaches the model about our new subject, and prompt alignment which prevents it from forgetting elements included in the target prompt.

Our approach liberates content creators from constraints associated with specific prompts, unleashing the full potential of text-to-image models. We evaluate our method qualitatively and quantitatively. We show superior results compared with the baselines in multi- and single-shot settings, all without pre-training on large-scale data [2, 16], which can be difficult for certain domains. Finally, we show that our method can accommodate multi-subject personalization with minor modification and offer new applications such as drawing inspiration from a single artistic painting, and not just text (see Figure 3).

### 2. Related work

Text-to-image synthesis has marked an unprecedented progress in recent years [13, 30, 36, 38, 41], mostly due to large-scale training on data like LAION-400m [42]. Our approach uses pre-trained diffusion models [21] to extend their understanding to new subjects. We use the publicly available Stable-Diffusion model [38] for most of our experiments since baseline models are mostly open-source on SD. We further verify our method on a larger latent diffu-

|[Figure 10]|[Figure 11]|
|---|---|
| |[Figure 12]|

[Figure 13]

[Figure 14]

| |[Figure 15]<br><br>[Figure 16]|
|---|---|
| |[Figure 17]|
| | |

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Figure 2. Previous personalization methods struggle with complex prompts (e.g., “A sketch inspired by Vitruvian man”) presenting a trade-off between prompt-alignment and subject-fidelity. Our method, optimizes for both, without compromising either.

sion model variant [38].

Text-based editing methods rely on contrastive multimodal models like CLIP [35] as an interface to guide local and global edits [3, 5, 6, 14, 32]. Recently, Promptto-Prompt (P2P) [19] was proposed as a way to edit and manipulate generated images by editing the attention maps in the cross-attention layers of a pre-trained text-to-image model. Later, Mokady et al. [29] extended P2P for real images by encoding them into the null-conditioning space of classifier-free guidance [20]. InstructPix2Pix [7] uses an instruction-guided image-to-image translation network trained on synthetic data. Others preserve image-structure by using reference attention maps [31] or features extracted using DDIM [44] inversion [47]. Imagic [27] starts from a target prompt, finds text embedding to reconstruct an input image, and later interpolates between the two to achieve the final edit. UniTune [48], on the other hand, performs the interpolation in pixel space during the denoising backward process. In our work, we focus on the ability to generate images depicting a given subject, which may not necessarily maintain the global structure of an input image.

Early personalization methods like Textual Inversion [15] and DreamBooth [39] tune pre-trained text-2image models to represent new subjects, either by finding a new soft word-embedding [15] or calibrating modelweights [39] with existing words to represent the newly added subject. Later methods improved memory require-

ments of previous methods using Low-Rank updates [22, 28, 40, 46] or compact-parameter space [17]. In another axis, NeTI [1] and P+ [51] extend TI [15] using more tokens to capture the subject-identifying features better. Personalization can also be used for other tasks. ReVersion [23] showed how to learn relational features from reference images, and Vinker et al. [50] used personalization to decompose and visualize concepts at different abstraction levels. Chefer et. al [9] propose an interpretability method for text-to-image models by decomposing concepts into interpretable tokens. Another line-of-works pre-train encoders on large-data for near-instant, single-shot adaptation [2, 10, 16, 49, 53, 55, 56]. Single-image personalization has also been addressed in Avrahami et al. [4], where the authors use segmentation masks to personalize a model on different subjects. In our work, we focus on prompt alignment, and any of the previous personalization methods may be replaced by our baseline personalization method.

Score Distillation Sampling (SDS) emerged as a technique for leveraging 2D-diffusion models priors [38, 41] for 3D-generation from textual input. Soon, this technique found way to different applications like SVG generation [24, 25], image-editing [18], and more [45]. Other variant of SDS [34] aim to improve the image-generation quality of SDS, which suffers from over-saturation and blurriness [26, 52]. In our approach, we propose a framework that leverages score sampling to maintain alignment with the target prompt. Alternative score-sampling techniques can be considered to further boost the text-alignment of the personalized method.

Text-to-image alignment methods address text-related issues that arise in base diffusion generative models. These issues include neglecting specific text parts, attribute mixing, and more. Previous methods address these issues through attention-map re-weighting[12, 33, 54], latentoptimization [8, 37], or re-training with additional data [43]. However, none of these methods address prompt alignment of personalization methods, instead, they aim to enhance the base models in generating text-aligned images.

### 3. Preliminaries

Generative Diffusion Models. Diffusion models perform a backward diffusion process to generate an image. In this process, the diffusion model G progressively denoises a source xT ∼ N(0,1) to produce a real sample x0 from an underlying data distribution p(X). At each timestep t ∈ {0,1,...,T}, the model predicts a noise ϵˆ = G(xt,t,y;θ) conditioned on a prompt y and timestep t. The generative model is trained by maximizing the evidence lower bound (ELBO) using a denoising score matching objective [21]:

|[Figure 25]|[Figure 26]| |
|---|---|---|
| | |[Figure 27]|
| | | |

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

|[Figure 33]<br><br>[Figure 34]|
|---|

|[Figure 35]<br><br>[Figure 36]|
|---|

Figure 3. PALP for multi-subject personalization achieves coherent and prompt-aligned results. Our method works when the subject has only one image (e.g., the ”Wanderer above the Sea of Fog” artwork by Caspar David Friedrich).

L(x,y) = Et∼[0,T],ϵ∼N(0,1) ∥G(xt,t,y;θ) − ϵ∥22 . (1)

Here, xt = √α¯tx + √1 − α¯tϵ, and x is a sample from the real-data distribution p(X). Throughout this section, we will write Gθ(xt,y) = G(xt,t,y;θ).

Personalization. Personalization methods fine-tune the diffusion model G for a new target subject S, by optimizing Eq. (1) on a small set of images representing S. TextualInversion [15] optimizes new word embeddings to represent the new subject. This is done by pairing a generic prompt with the placeholder [V ], e.g., “A photo of [V]”, where [V ] is mapped to the newly-added word embedding. DreamBooth [39] calibrates existing word-embeddings to represent the personal subject S. This can be done by adjust-

[Figure 37]

[Figure 38]

- Figure 4. Method overview. We propose a framework consisting of a personalization path (left) and a prompt-alignment branch (right) applied simultaneously in the same training step. We achieve personalization by finetuning the pre-trained model using a simple reconstruction loss to denoise the new subject S. To keep the model aligned with the target prompt, we additionally use score sampling to pivot the prediction towards the direction of the target prompt y, e.g., ”A sketch of a cat.” In this example, when personalization and text alignment are optimized simultaneously, the network learns to denoise the subject towards a ”sketch” like representation. Finally, our method does not induce a significant memory overhead due to the efficient estimation of the score function, following [34].

ing the model weights, or using recent more efficient methods that employ a Low-Rank Adaptation (LoRA) [22] on a smaller subset of weights [28, 40, 46].

### 4. Prompt Alignment Method

#### 4.1. Overview

Our primary objective is to teach G to generate images related to a new subject S. However, unlike previous methods, we strongly emphasize achieving optimal results for a single textual prompt, denoted by y. For instance, consider a scenario where the subject of interest is one’s cat, and the prompt is y = “A sketch of [my cat] in Paris.” In this case, we aim to generate an image that faithfully represents the cat while incorporating all prompt elements, including the sketchiness and the Parisian context.

The key idea is to optimize two objectives: personalization and prompt alignment. For the first part, i.e., personalization, we fine-tune G to reconstruct S from noisy samples. One option for improving prompt alignment would be using our target prompt y as a condition when tuning G on S. However, this results in sub-optimal outcome since we have no images depicting y (e.g., we have no sketch images of our cat, nor photos of it in Paris). Instead, we push G’s noise prediction towards the target prompt y. In particular, we steer G’s estimation of S towards the distribution density p(x|y) using score-sampling methods (See Fig. 4). By employing both objectives simultaneously, we achieve two things: (1) the ability to generate the target subject S via the backward-diffusion process (personalization) while (2) ensuring the noise-predictions are aligned with text y. We next explain our method in detail.

#### 4.2. Personalization

We follow previous methods [15, 28, 39, 46], and fine-tune G on a small set of images representing S, which can be as small as a single photo. We update G’s weights using LoRA method [22], where we only update a subset of the network’s weights θLoRA ⊆ θ, namely, the self- and crossattention layers. Furthermore, we optimize a new word embedding [V ] to represent S.

#### 4.3. Prompt-Aligned Score Sampling

Long fine-tuning on a small image set could cause the model to overfit. In this case, the diffusion model predictions always steer the backward denoising process towards one of the training images, regardless of any conditional prompt. Indeed, we observed that overfitted models could estimate the inputs from noisy samples, even in a single denoising step.

In Fig. 5, we visualize the model prediction by analyzing its estimate of x0, the real-sample image. The real-sample estimation xˆ0 is derived from the model’s noise prediction via:

√1 − α¯tGθ(xt,y) √α¯t

xt −

. (2)

xˆ0 =

.

As can be seen from Fig. 5, the pre-trained model (before personalization) steers the image towards a sketch-like image, where the estimate xˆ0 has a white background, but the subject details are missing. Apply personalization without PALP overfits, where elements from the training images, like the background, are more dominant, and sketchiness fades away, suggesting miss-alignment with the prompt “A sketch” and over-fitting to the input image. Using PALP, the model prediction steers the backward-denoising process

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

(a) Input (b) base model (c) Over-fitted (d) Prompt aligned

- Figure 5. Visualization of xˆ0. We visualize the model estimation of xˆ0 given a pure-noise and the prompt ”A sketch of [V].” The base model (b) is not personalized to the target subject and predicts mainly the ”Sketch” appearance. Personalization methods (c) tend to overfit the input image where many image elements, including the background and the subject colors are restored, suggesting the model does not consider the prompt condition. Prompt aligned personalization (c) maintains the sketchiness and does not overfit (see cat-like shape).

towards a sketch-like image, while staying personalized, where a cat like shape is restored.

Our key idea is to encourage the model’s denoising prediction towards the target prompt. Alternatively, we push the model’s estimation of the real-sample, denoted by xˆ0, to be aligned with the prompt y.

In our example, where the target prompt is ”A sketch of [V],” we push the model prediction toward a sketch and prevent overfitting a specific image background. Together with the personalization loss, this will encourage the model to focus on capturing [V]’s identifying features rather than reconstructing the background and other distracting features.

To achieve the latter, we use Score Distillation Sampling (SDS) techniques [34]. In particular, given a clean target prompt yc, that doesn’t contain the placeholder ”[V]”. Then we use score sampling to optimize the loss:

L(ˆx0,yc), (3)

from Eq. (1). Note, we use the pre-trained network weights to evaluate Eq. (3), and omit all learned placeholder (e.g., [V ]). Therefore, this ensures that by minimizing the loss, we will stay aligned with the textual prompt y since the pretrained model possesses all knowledge about the prompt elements.

Poole et al. [34] found an elegant and efficient approximation of the score function using:

∂x ∂ϕ

∇LSDS(x) = w˜(t)(Gα (xt,t,yc;θ) − ϵ)

, (4)

where ϕ are the weights controlling the appearance of x, and w˜(t) is a weighting function. Here Gα denotes the classifier-free guidance prediction, which is an extrapolation of the conditional and unconditioned (y = ∅) noise prediction. The scalar α ∈ R+ controls the extrapolation via:

Gα (xt,t,yc;θ) = (1−α)·Gθ (xt,∅)+α·Gθ (xt,yc). (5)

In our case, x = xˆ0, and it is derived from G’s noise prediction as per Eq. (2) . Therefore, the appearance of xˆ0 is directly controlled by the LoRA weights and [V ].

#### 4.4. Avoiding Over-saturation and Mode Collapse

Guidance by SDS produces less diverse and over-saturated results. Alternative implementations like [26, 52] improved diversity, but the overall personalization is still affected (see Appendix A). We argue that the reason behind this is that the loss in Eq. (4) pushes the prediction xˆ0 towards the center of distribution of p(x|yc), and since the clean target prompt yc contains only the subject class, i.e., “A sketch of a cat”, then as a result, the loss will encourage xˆ0 towards the center of distribution of a general cat, not our one.

Instead, we found the Delta Denoising Score (DDS) [18] variant to work better. In particular, we use the residual direction between the personalized model’s prediction Gβθ

(ˆxt,yP), and the pre-trained one Gαθ (ˆxt,yc), where α and β are their guidance scales, respectively. Here yP is the personalization prompt, i.e., ”A photo of [V],” and yc is a clean prompt, e.g., ”A sketch of a cat.” The score can then be estimated by:

LoRA

∂xˆ0 ∂θLoRA

∇LPALP = w˜(t)(Gαθ (ˆxt,yc) − Gβθ

,

(ˆxt,yP))

LoRA

(6) which perturbs the denoising prediction towards the target prompt (see right part of Fig. 4). Our experiments found imbalanced guidance scales, i.e., α > β, to perform better. We have also considered two variant implementations: (1) in the first, we use the same noise for the two branches, i.e., the text-alignment and personalization branches, and (2) we use two i.i.d noise samples for the branches. Using the same noise achieves better text alignment than the latter variant.

On the Computational Complexity of PALP: the gradient in Eq. (6), is proportional to the personalization gradient with the scaling:

√1 − α¯t √α¯t ∇Gθ

∂xˆ0 ∂θLoRA ∝ −

LoRA

(xt,yP). (7)

(xt,yP) is also calculated as part of the derivative of Eq. (1), then, we do not need to back-propagate through the text-to-image model through the prompt-alignment branch. This may be useful to use guidance from bigger models to boost smaller model’s personalization performance. Finally, we also note that the scaling term is very high for large t-values. Re-scaling the gradient back by √α¯t/√1 − α¯t ensures uniform gradient update across all timesteps and improves numerical stability.

and since the gradient ∇Gθ

LoRA

Input Ours TI+DB [15, 39] CD [28] P+ [51] NeTI [1]

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

[Figure 58]

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

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Table 1. Qualitative comparison in multi-shot setting. Our method achieves state-of-the-art results on complex prompts, better-preserving identity, and prompt alignment. For TI [15]+DB [39], we use the same seed to generate our results, emphasizing the gain achieved by incorporating the prompt alignment path. For other baselines, we chose the best two out of eight samples.

0.800

0.35

0.775

0.34

0.750

base model TI + DB TI + DB w/ NFSD Ours (different noise) Ours (same noise) Ours w/ rescale

0.33

0.725

Personalization

Tex-Alignment

0.32

0.700

0.31

0.675

0.650

0.30

0.625

0.29

0.600

100 200 300 400 500 Steps

100 200 300 400 500 Steps

- Figure 6. Ablation study. We report image alignment (left) and text alignment (right) as a function of the number of fine-tuning steps. The base model results are the pre-trained model’s performance when using the class word to represent the target subject.

### 5. Results

Experimental setup: We use StableDiffusion (SD)v1.4 [38] for ablation and comparison purposes, as many official implementations of state-of-the-art methods are available in SD-v1.4. We further validate our method with larger text-to-image models (see Appendix B). Complete experimental configuration, including learning rate, number of steps, and batch size, appear in Appendix C.

Evaluation metric: for evaluation, we follow previous works [15, 39] and use CLIP-score [35] to measure alignment with the target (clean) prompt ytc (i.e., does not have the placeholder [V]). For subject preservation, we also use CLIP feature similarity between the input and generated images with the target prompt. We use VIT-B32 [11] trained by OpenAI on their proprietary data for both metrics. This ensures that the underlying CLIP used by SD-v1.4 differs from the one used for evaluation, which could compromise the validity of the reported metric.

Dataset: for multi-shot setting, we use data collected by previous methods [15, 28], with different subjects like animals, toys, personal items, and buildings. For those subjects, checkpoints of previous methods exist, allowing a fair comparison.

#### 5.1. Ablation studies

For ablation, we start with TI [15] and DB [39] as our baseline personalization method and gradually add different components contributing to our final method.

Early stopping: We begin by considering early stopping as a way to control the text-alignment. The lower the number of iterations, the less likely we are to hurt the model’s prior knowledge. However, this comes at the cost of subject fidelity, evident from Fig. 6. The longer we tune the model on the target subject, the more we risk overfitting the training set.

Adding SDS guidance: improves text alignment, yet it severely harms the subject fidelity, and image diversity is

Text-Alignment ↑ Image-Alignment ↑ Method Style Class Ambiance 1 Ambiance 2 Target Prompt

P+ 0.244 0.257 0.217 0.218 0.308 0.673 NeTI 0.235 0.264 0.22 0.214 0.310 0.695 TI+DB 0.237 0.279 0.22 0.216 0.319 0.716 Ours 0.245 0.272 0.23 0.224 0.340 0.681

- Table 2. Comparisons to prior work. Our method presents better prompt-alignment, without hindering personalization.

Method Text-Alignment ↑ Personalization ↑

P+ 68.5 % 61.2 % NeTI 63.2 % 70.3 % TI+DB 73.3 % 60.4 % Ours 91.2 % 72.1 %

- Table 3. User Study results. For text alignment, we report the percent of elements from the prompt that users found in the generated image. For personalization, users rated the similarity of the subject S and the main subject in the generated image.

substantially reduced (see Appendix A). Alternative distillation sampling guidance [26] improves on top of SDS; however, since the distillation sampling guides the personalization optimization towards the center of distribution of the subject class, it still produces less favorable results.

Replacing SDS with PALP guidance: improves text alignment by a considerable margin and maintains high fidelity to the subject S. We consider two variants: one where we use the same noise for personalization loss or sample a new one from the normal distribution. Interestingly, using the same noise helps with prompt alignment. Furthermore, scaling the score sampling equation Eq. (6) by √α¯t/√1 − α¯t further enhances the performance.

#### 5.2. Comparison with Existing Methods

We compare our method against multi-shot methods, including CustomDiffusion [28], P+ [51], and NeTI [1]. We further compare against TI [15] and DB [39] using our implementation, which should also highlight the gain we achieve by incorporating our framework with existing personalization methods. Our evaluation set contains ten different complex prompts that include at least four different elements, including style-change (e.g., ”sketch of”, ”anime drawing of”), time or a place (e.g., ”in Paris”, ”at night”), color palettes (e.g., ”warm,” ”vintage”). We asked users to count the number of elements that appear in the image (textalignment) and rate the similarity of the results to the input subject (personalization, see Tab. 2).

Our method achieves the best text alignment while maintaining high image alignment. TI+DB achieves the best image alignment. However, the reason for this is because TI+DB is prone to over-fitting. Indeed, investigating each element in the prompt, we find that the TI+DB achieves the best alignment with the class prompt (e.g., ”A photo of a

Input Ours Best of previous works

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

”3D Render as a chef”

”Vector art wearing a hat”

”A painting by Da Vinci”

”3D Render as a chef”

”Vector art wearing a hat”

”A painting by Da Vinci”

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

”Anime drawing” ”Pop art” ”A Caricature” ”Anime drawing” ”Pop art” ”A Caricature”

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

”3D Render as a chef”

”Vector art wearing a hat”

”A painting by Da Vinci”

”3D Render as a chef”

”Vector art wearing a hat”

”A painting by Da Vinci”

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

”Anime drawing” ”Pop art” ”A Caricature” ”Anime drawing” ”Pop art” ”A Caricature”

Table 4. Qualitative comparison against ProFusion [56], IP-Adapter [55] , E4T [16], and Face0 [49]. On the left, we show the results of our method on two individuals using a single image with multiple prompts. To meet space requirements, we report a single result among all previous methods, based on our preference, for a given subject and prompt. Full comparison appears in Tab. 9 Tab. 8.

cat”) while being significantly worse in the Style prompt (e.g., ”A sketch”). Our method has a slightly worse image alignment since we expect appearance change for stylized prompts. We validate this hypothesis with a user study and find that our method achieves the best user preference in prompt alignment and personalization (see Tab. 3). Full details on the user study appear in Appendix D.

#### 5.3. Applications

Single-shot setting: In a single-shot setting, we aim to personalize text-2-image models using a single image. This setting is helpful for cases where only a single image exists for our target subject (e.g., an old photo of a loved one). For this setting, we qualitatively compare our method with encoder-based methods, including IP-Adapter [55], ProFusion [56], Face0 [49], and E4T [16]. We use portraits of two individuals and expect previous methods to generalize to our selected images since all methods are pre-trained on

human faces. Note that E4T [16] and ProFusion [56] also perform test time optimization.

As seen from Tab. 4, our method is both prompt- and identity-aligned. Previous methods, on the other hand, struggle more with identity preservation. We note that optimization-based approaches [16, 56] are more identitypreserving, but this comes at the cost of text alignment. Finally, our method achieves a higher success rate, where the quality of the result is independent of the chosen seed.

Multi-concept Personalization: Our method accommodates multi-subject personalization via simple modifications. Assume we want to compose two subjects, S1 and S2, in a specific scene depicted by a given prompt y. To do so, we first allocate two different placeholders, [V1] and [V2], to represent the target subjects S1 and S2, respectively. During training, we randomly sample an image from

a set of images containing S1 and S2. We assign different personalization prompts yP for each subject, e.g., ”A photo of [V1]” or ”A painting inspired by [V2]”, depending on the context. Then, we perform PALP while using the target prompt in mind, e.g., ”A painting of [V1] inspired by [V2]”. This allows composing different subjects into coherent scenes or using a single artwork as a reference for generating art-inspired images. Results appear in Fig. 3; further details and results appear in Appendix E.

### 6. Conclusions

We have introduced a novel personalization method that allows better prompt alignment. Our approach involves finetuning a pre-trained model to learn a given subject while employing score sampling to maintain alignment with the target prompt. We achieve favorable results in both promptand subject-alignment and push the boundary of personalization methods to handle complex prompts, comprising multiple subjects, even when one subject has only a single reference image.

While the resulting personalized model still generalizes for other prompts, we must personalize the pre-trained model for different prompts to achieve optimal results. For practical real-time use cases, there may be better options. However, future directions employing promptaligned adapters could result in instant time personalization for a specific prompt (e.g., for sketches). Finally, our work will motivate future methods to excel on a subset of prompts, allowing more specialized methods to achieve better and more accurate results.

### References

- [1] Yuval Alaluf, Elad Richardson, Gal Metzer, and Daniel Cohen-Or. A neural space-time representation for text-toimage personalization. CoRR, abs/2305.15391, 2023. 3, 6, 7, 12, 13, 17
- [2] Moab Arar, Rinon Gal, Yuval Atzmon, Gal Chechik, Daniel Cohen-Or, Ariel Shamir, and Amit H. Bermano. Domainagnostic tuning-encoder for fast personalization of text-toimage models. CoRR, abs/2307.06925, 2023. 2, 3
- [3] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 1824, 2022, pages 18187–18197. IEEE, 2022. 2
- [4] Omri Avrahami, Kfir Aberman, Ohad Fried, Daniel CohenOr, and Dani Lischinski. Break-a-scene: Extracting multiple concepts from a single image. CoRR, abs/2305.16311, 2023. 3
- [5] Omri Avrahami, Ohad Fried, and Dani Lischinski. Blended latent diffusion. ACM Trans. Graph., 42(4):149:1–149:11,

2023. 2

- [6] Omer Bar-Tal, Dolev Ofri-Amar, Rafail Fridman, Yoni Kasten, and Tali Dekel. Text2live: Text-driven layered image

- and video editing. In Computer Vision - ECCV 2022 - 17th European Conference, Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part XV, pages 707–723. Springer, 2022. 2
- [7] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. Instructpix2pix: Learning to follow image editing instructions. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 1724, 2023, pages 18392–18402. IEEE, 2023. 2
- [8] Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM Transactions on Graphics (TOG), 42(4):1–10, 2023. 3
- [9] Hila Chefer, Oran Lang, Mor Geva, Volodymyr Polosukhin, Assaf Shocher, Michal Irani, Inbar Mosseri, and Lior Wolf. The hidden language of diffusion models. arXiv preprint arXiv:2306.00966, 2023. 3
- [10] Wenhu Chen, Hexiang Hu, Yandong Li, Nataniel Ruiz, Xuhui Jia, Ming-Wei Chang, and William W. Cohen. Subject-driven text-to-image generation via apprenticeship learning. CoRR, abs/2304.00186, 2023. 3
- [11] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7,

2021. OpenReview.net, 2021. 7

- [12] Weixi Feng, Xuehai He, Tsu-Jui Fu, Varun Jampani, Arjun Akula, Pradyumna Narayana, Sugato Basu, Xin Eric Wang, and William Yang Wang. Training-free structured diffusion guidance for compositional text-to-image synthesis. arXiv preprint arXiv:2212.05032, 2022. 3
- [13] Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman. Make-a-scene: Scenebased text-to-image generation with human priors. In Computer Vision - ECCV 2022 - 17th European Conference, Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part XV, pages 89–106. Springer, 2022. 2
- [14] Rinon Gal, Or Patashnik, Haggai Maron, Amit H. Bermano, Gal Chechik, and Daniel Cohen-Or. Stylegan-nada: Clipguided domain adaptation of image generators. ACM Trans. Graph., 41(4):141:1–141:13, 2022. 2
- [15] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit Haim Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. 1, 2, 3, 4, 6, 7, 12, 17
- [16] Rinon Gal, Moab Arar, Yuval Atzmon, Amit H. Bermano, Gal Chechik, and Daniel Cohen-Or. Encoder-based domain tuning for fast personalization of text-to-image models. ACM Trans. Graph., 42(4):150:1–150:13, 2023. 2, 3, 8, 18, 19
- [17] Ligong Han, Yinxiao Li, Han Zhang, Peyman Milanfar, Dimitris Metaxas, and Feng Yang. Svdiff: Compact parameter space for diffusion fine-tuning. arXiv preprint arXiv:2303.11305, 2023. 3

- [18] Amir Hertz, Kfir Aberman, and Daniel Cohen-Or. Delta denoising score. CoRR, abs/2304.07090, 2023. 3, 5
- [19] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross-attention control. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. 2
- [20] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. CoRR, abs/2207.12598, 2022. 2
- [21] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020. 2, 3
- [22] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. 3, 4, 12
- [23] Ziqi Huang, Tianxing Wu, Yuming Jiang, Kelvin C. K. Chan, and Ziwei Liu. Reversion: Diffusion-based relation inversion from images. CoRR, abs/2303.13495, 2023. 3
- [24] Shir Iluz, Yael Vinker, Amir Hertz, Daniel Berio, Daniel Cohen-Or, and Ariel Shamir. Word-as-image for semantic typography. ACM Trans. Graph., 42(4):151:1–151:11, 2023. 3
- [25] Ajay Jain, Amber Xie, and Pieter Abbeel. Vectorfusion: Text-to-svg by abstracting pixel-based diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 1724, 2023, pages 1911–1920. IEEE, 2023. 3
- [26] Oren Katzir, Or Patashnik, Daniel Cohen-Or, and Dani Lischinski. Noise-free score distillation. arXiv preprint arXiv:2310.17590, 2023. 3, 5, 7, 11, 12
- [27] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pages 6007–6017. IEEE, 2023. 2
- [28] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pages 1931–1941. IEEE, 2023. 3, 4, 6, 7, 17
- [29] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pages 6038–6047. IEEE, 2023. 2
- [30] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021. 2

- [31] Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot image-to-image translation, 2023. 2
- [32] Or Patashnik, Zongze Wu, Eli Shechtman, Daniel CohenOr, and Dani Lischinski. Styleclip: Text-driven manipulation of stylegan imagery. In 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021, pages 2065–2074. IEEE, 2021. 2
- [33] Quynh Phung, Songwei Ge, and Jia-Bin Huang. Grounded text-to-image synthesis with attention refocusing. arXiv preprint arXiv:2306.05427, 2023. 3
- [34] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. 2, 3, 4, 5, 11, 12
- [35] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. CoRR, abs/2103.00020, 2021. 2, 7
- [36] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. CoRR, abs/2102.12092,

2021. 1, 2

- [37] Royi Rassin, Eran Hirsch, Daniel Glickman, Shauli Ravfogel, Yoav Goldberg, and Gal Chechik. Linguistic binding in diffusion models: Enhancing attribute correspondence through attention map alignment, 2023. 3
- [38] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 10674–

10685. IEEE, 2022. 2, 3, 7

- [39] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pages 22500–22510. IEEE,

2023. 1, 2, 3, 4, 6, 7, 12, 17

- [40] Simo Ryu. Low-rank adaptation for fast text-to-image diffusion fine-tuning. [https : / / github . com / cloneofsimo / lora](https://github.com/cloneofsimo/lora3, 4
- [41] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models with deep language understanding. arXiv preprint arXiv:2205.11487, 2022. 1, 2, 3
- [42] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m:

- Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 2
- [43] Eyal Segalis, Dani Valevski, Danny Lumen, Yossi Matias, and Yaniv Leviathan. A picture is worth a thousand words: Principled recaptioning improves image generation. arXiv preprint arXiv:2310.16656, 2023. 3
- [44] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. 2
- [45] Kunpeng Song, Ligong Han, Bingchen Liu, Dimitris N. Metaxas, and Ahmed Elgammal. Diffusion guided domain adaptation of image generators. CoRR, abs/2212.04473,

2022. 3

- [46] Yoad Tewel, Rinon Gal, Gal Chechik, and Yuval Atzmon. Key-locked rank one editing for text-to-image personalization. In ACM SIGGRAPH 2023 Conference Proceedings, SIGGRAPH 2023, Los Angeles, CA, USA, August 6-10, 2023, pages 12:1–12:11. ACM, 2023. 3, 4
- [47] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pages 1921–1930. IEEE, 2023. 2
- [48] Dani Valevski, Matan Kalman, Yossi Matias, and Yaniv Leviathan. Unitune: Text-driven image editing by fine tuning an image generation model on a single image. arXiv preprint arXiv:2210.09477, 2022. 2
- [49] Dani Valevski, Danny Wasserman, Yossi Matias, and Yaniv Leviathan. Face0: Instantaneously conditioning a text-toimage model on a face. CoRR, abs/2306.06638, 2023. 3, 8, 18, 19
- [50] Yael Vinker, Andrey Voynov, Daniel Cohen-Or, and Ariel Shamir. Concept decomposition for visual exploration and inspiration. CoRR, abs/2305.18203, 2023. 3
- [51] Andrey Voynov, Qinghao Chu, Daniel Cohen-Or, and Kfir Aberman. P+: extended textual conditioning in text-toimage generation. CoRR, abs/2303.09522, 2023. 3, 6, 7, 12, 17
- [52] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. CoRR, abs/2305.16213, 2023. 3, 5
- [53] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. arXiv preprint arXiv:2302.13848, 2023. 3
- [54] Qiucheng Wu, Yujian Liu, Handong Zhao, Trung Bui, Zhe Lin, Yang Zhang, and Shiyu Chang. Harnessing the spatialtemporal attention of diffusion models for high-fidelity textto-image synthesis. CoRR, abs/2304.03869, 2023. 3
- [55] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. CoRR, abs/2308.06721, 2023. 3, 8, 18, 19
- [56] Yufan Zhou, Ruiyi Zhang, Tong Sun, and Jinhui Xu. Enhancing detail preservation for customized text-to-image

|[Figure 121]<br><br>[Figure 122]<br><br>[Figure 123]<br><br>[Figure 124]|
|---|

- Figure 7. Additional results for Multi-subject personalization. The reference images appear in Fig. 10.

|[Figure 125]|
|---|

|[Figure 126]|
|---|

- Figure 8. User study sample questions. On the left, we show a sample question in which the participants must count the prompt elements perceived from the generated sample. On the right, users were asked to rate, from 1 to 5, the similarity between the two main subjects, with one being the least similar. Participants answers were aggregated and normalized.

generation: A regularization-free approach. CoRR, abs/2305.13579, 2023. 3, 8, 18, 19

### A. Over-saturation and mode-collapse in SDS

We use prompt-aligned score sampling Eq. (6) to guide the model towards the target prompt y. We found this to be more effective than using SDS [34] or improved versions of it like NFSD [26]. In particular, SDS and NFSD produce over-saturated or less diverse results. We provide qualitative examples in Appendix A, and quantitative comparison appears in Sec. 5.1.

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

[Figure 137]

[Figure 138]

(a) SDS [34] (b) NFSD [26] (c) Ours

- Figure 9. Using SDS [34] guidance for prompt alignment produces over-saturated results. Alternative improvements like NFSD [26] producess less-diverse results. Our method produces diverse and prompt-aligned results. The input prompt is ”A digital art of [V] on a field with lightning in the background.”

### B. Additional Results

We provide additional qualitative comparison for multi-shot setting, the results appear in Tab. 7. Moreover, a full qualitative comparison and un-curated results of the single-shot experiment appear in Tab. 8, Tab. 9, and Fig. 12. For results obtained using a larger model, with improved text-encoder and bigger capacity, see Tab. 5 and Tab. 6.

### C. Implementation Details

All our experiments, except those conducted using a larger model, were conducted using TPU-v3, with a global batch size of 32. We use a learning rate of 5e-5. For LoRA [22], we use a rank r = 32 and only modify the self-and crossattention layers’ projection matrices. For most experiments, we use a classifier free-guidance scale of α = 15.0 and α = 7.5; for composition experiments, we use α = 7.5 and β = 1.0. For quantitative comparison, we fine-tuned the model for 500 steps. However, this may be sub-optimal depending on the subject and prompt complexity.

### D. User-Study Details

In the user study, we asked 30 individuals to rate prompt alignment and personalization performances of four methods, including P+ [51], NeTI [1], TI [15]+DB [39], and our method. Our test set includes six different subjects and ten prompts. We generated eight samples for each prompt and subject using the four methods. Then, we randomly picked a single photo and asked the participants to count the number of different prompt elements that appear in the generated image. We asked the users to rate the similarity between the main subject in the generated image and the sample. Then, we randomly divided the questions into three forms and shuffled them between the participants. Sample

questions appear in Fig. 8.

### E. Multi-subject Personalization

Multi-subject composition: For multi-subject personalization, we use two placeholder V1 and V2 to represent the subject S1 and S2. We use yP = ”A photo of [Vi]” as our personalization prompt. We use a descriptive prompt describing the desired scene for the target clean prompt. For example, ”A 19th century portrait of a [C1] and [C2]”, where C1 and C2 represent the class name of S1 and S2, respectively. We provide additional results in Fig. 7.

Art-inspired composition: For Art-inspired composition, we use ”A photo of [V1]” to describe the main subject, while ”An oil painting of [V2]” is used for the artistic image. The clean prompt used is ”An oil painting of [class name],” where ”[class name]” is the subject class (e.g., a cat or a toy). Furthermore, adding a description to the clean prompt improves alignment, for example, ”An oil painting of a cat sitting on a rock” for the” Wanderer above the Sea of Fog” artwork by Caspar David Friedrich. Finally, at test time, we use the target prompt ”An oil painting of [V1] inspired by [V2] painting” or a similar variant, with possibly an additional description of the desired scene. Results appear in Fig. 10. We further show that joint training is insufficient for this task, nor is pre-training on S prompting the artwork or artist name. Both cases produce miss-aligned results (see Fig. 11).

As can be seen from Fig. 5, the pre-trained model (before personalization) steers the image towards a sketch-like image, where the estimate xˆ0 has a white background, but the subject details are missing. Apply personalization without PALP overfits, where elements from the training images, like the background, are more dominant, and sketchiness

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

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

|[Figure 163]<br><br>[Figure 164]<br><br>[Figure 165]<br><br>[Figure 166]<br><br>[Figure 167]<br><br>[Figure 168]<br><br>[Figure 169]<br><br>|
|---|

|[Figure 170]<br><br>[Figure 171]<br><br>[Figure 172]<br><br>[Figure 173]|
|---|

- Figure 10. Art inspired composition. Our method blends the target subject and the reference paintings coherently. Further, we can produce diverse results by slightly modifying the prompt. For example, we can generate an image of <my toy> having a picnic, playing with a kite, or simply standing next to a lake (see top-row third column).

[Figure 174]

[Figure 175]

[Figure 176]

(a) NeTI [1] (b) w/o PALP Guidnace (c) Ours

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

- Figure 11. Art-inspired composition ablation. We consider two alternatives: (a) using a pre-trained personalized model and prompting the artwork name, or (b) jointly training on both subjects without PALP guidance. In both cases, the results are sub-optimal, where our method achieves more coherent results. Reference images appear in Fig. 10.

fades away, suggesting miss-alignment with the prompt “A sketch” and over-fitting to the input image. Using PALP, the model prediction steers the backward-denoising process towards a sketch-like image, while staying personalized, where a cat like shape is restored.

[Figure 183]

A lineart of [V]

[Figure 184]

[V] in the style of Vermeer

[Figure 185]

A sketch of [V]

Figure 12. Un-curated samples for single-shot setting. The input image appear in Tab. 9.

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

###### Table 5. Additional Results on Larger Text-to-Image Models. Reference images appear at the bottom.

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

[Figure 217]

[Figure 218]

[Figure 219]

###### Table 6. Additional Results on Larger Text-to-Image Models. Reference images appear at the bottom.

Input Ours TI+DB [15, 39] CD [28] P+ [51] NeTI [1]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

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

Table 7. Additional qualitative comparison in multi-shot setting.

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

###### Table 8. Single-shot setting - full qualitative comparison.

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

###### Table 9. Single-shot setting - full qualitative comparison.

