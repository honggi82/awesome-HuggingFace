## CoRe: Context-Regularized Text Embedding Learning for Text-to-Image Personalization

### Feize Wu1*, Yun Pang1*, Junyi Zhang1*, Lianyu Pang1*, Jian Yin1, Baoquan Zhao1, Qing Li2, Xudong Mao1†

1Sun Yat-sen University 2The Hong Kong Polytechnic University

# arXiv:2408.15914v1[cs.CV]28Aug2024

##### Abstract

Recent advances in text-to-image personalization have enabled high-quality and controllable image synthesis for userprovided concepts. However, existing methods still struggle to balance identity preservation with text alignment. Our approach is based on the fact that generating prompt-aligned images requires a precise semantic understanding of the prompt, which involves accurately processing the interactions between the new concept and its surrounding context tokens within the CLIP text encoder. To address this, we aim to embed the new concept properly into the input embedding space of the text encoder, allowing for seamless integration with existing tokens. We introduce Context Regularization (CoRe), which enhances the learning of the new concept’s text embedding by regularizing its context tokens in the prompt. This is based on the insight that appropriate output vectors of the text encoder for the context tokens can only be achieved if the new concept’s text embedding is correctly learned. CoRe can be applied to arbitrary prompts without requiring the generation of corresponding images, thus improving the generalization of the learned text embedding. Additionally, CoRe can serve as a test-time optimization technique to further enhance the generations for specific prompts. Comprehensive experiments demonstrate that our method outperforms several baseline methods in both identity preservation and text alignment. Code will be made publicly available.

### Introduction

Text-to-image personalization involves adapting a pretrained diffusion model to generate novel images based on user-provided concepts and text prompts. The goal of personalization techniques is to produce high-quality images that not only accurately preserve the concept’s identity but also align well with the text prompt. However, balancing the trade-off between identity preservation and text alignment remains a core challenge in personalization of diffusion models.

In this work, we focus on improving text alignment for text-to-image personalization. Our investigation is based on the fact that a precise semantic understanding of the prompts is the premise for aligning the generated images with the prompts. The semantic understanding of the prompts is man-

*Equal contribution. †Corresponding author (xudong.xdmao@gmail.com).

Input

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

A S* wearing a Santa hat, with a white beard, dressed as Santa Claus, is standing in front of a burning fireplace

A S* was planted with sunflowers on a wooden shelf, on a sunny day

Input

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

A S* dressed as Loki, with a tiny horned helmet and a green and gold outfit, standing on a throne in a grand Asgardian hall

A S* as Star-Lord, featuring a red leather jacket and a miniature blaster, standing on a distant alien planet with a colorful nebula sky

Input

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

A S* dressed as a Jedi, holding a glowing lightsaber, standing on a desert planet under the sun

A S* dressed as a wizard, holding a glowing staff, standing in a mystical forest filled with ancient ruins

Input

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

A S* in a suit and Mark Zuckerberg are A backpack in the style of S* enjoying a day at an amusement park

Figure 1: CoRe enables text-aligned personalized generations, allowing for high visual variability of the userprovided concept.

aged by the CLIP text encoder, which involves processing the text embeddings of all tokens and their interactions. Therefore, we aim to learn a proper text embedding for the new concept, which not only accurately represents the concept but also seamlessly integrates with existing tokens.

Instead of investigating the text embedding of the new concept itself (Alaluf et al. 2023; Pang et al. 2024b), we shift our focus to the context tokens surrounding the new concept in prompts. Here, the term text embedding refers to the input to the CLIP text encoder. Moreover, for clarity and following the terminology used in (Lu et al. 2022), we refer to the embeddings before and after the CLIP text encoder also as the input embedding and output embedding, respectively. As illustrated in Figure 2, consider a scenario where the object

Cosine similarity to each word of "{Dog} in the desert"

1.0

{Dog} in the desert {Cat} in the desert {Puppy} in the desert

0.9

0.8

CosineSimilarity

0.7

0.6

{Overfitted-S*} in the desert

0.5

{Dog}

{Puppy}

0.4

{Cat}

0.3

{Overfitted-S*}

{} in the desert

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Stable Diffusion

{Dog} in the desert

Output

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Stable Diffusion

{Puppy} in the desert

Output

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Textual Inversion

[Figure 36]

Input

Input {Overfitted-S*} in the desert

Output

- Figure 2: For the four similar prompts (“{} in the desert”), we show the cosine similarity between the output embeddings of each token (left), and the cross-attention map visualization of each token (right). Replacing “dog” with “puppy” or “cat” results

in similar output embeddings and attention maps for other tokens. In contrast, using the overfitted S∗ by Textual Inversion significantly alters the output embeddings and attention maps for other tokens.

token in a prompt is switched from “dog” to “cat”; the output embeddings of the other tokens largely remain consistent. However, using an overfitted text embedding by Textual Inversion (Gal et al. 2022) significantly alters the output embeddings of its context tokens. This alteration occurs because the overfitted embedding adversely affects the output of the context tokens within the text encoder. As shown, these inappropriate output embeddings subsequently lead to incorrect allocations in the attention maps for the context tokens.

Based on these observations, we introduce a new method named Context Regularization (CoRe), which enhances text embedding learning for a new concept by regularizing its context tokens. CoRe can improve the compatibility of the new concept’s text embedding, thereby facilitating a more precise semantic understanding of the prompt. As indicated in Figure 2, replacing the object token “dog” with a compatible embedding of the new concept should yield similar output embeddings for the context tokens. Therefore, we propose a regularization strategy that encourages the output embeddings of the context tokens to align with those from a reference prompt containing a super-category token. As the attention maps play a crucial role in generation, we also impose constraints on the attention maps for the context tokens. We avoid imposing constraints directly between the new concept and its super-category, due to the typically substantial differences between them.

As our proposed CoRe is applied only to the output embeddings and attention maps, without the need for generating images, it can be used with arbitrary prompts. Therefore, we construct a regularization prompt set that covers a broad range of prompts to improve the generalization of the new concept’s text embedding. During training, a prompt is randomly selected from this set for regularization purposes. Moreover, at test time, CoRe can serve as a test-time optimization approach to further refine the generations for specific prompts.

We demonstrate the effectiveness of our method by comparing it with four state-of-the-art personalization methods through both qualitative and quantitative evaluations. Our method shows superior performance in identity preservation and text alignment compared to the baselines, especially for prompts requiring high visual variability. Moreover, in addition to personalizing general objects, our method also works well for face personalization, generating more identity-preserved face images compared to three recent face personalization methods.

### Related Work

Text-to-Image Generation. Text-to-image generation involves creating visual images from textual prompts, a task that has seen significant advances with diffusion models (Sohl-Dickstein et al. 2015; Ho, Jain, and Abbeel 2020; Nichol and Dhariwal 2021). To achieve high-resolution textto-image generation, various methods have been developed. DALL-E 2 (Ramesh et al. 2022) proposes converting a CLIP text embedding into a CLIP image embedding, and then transforming this image embedding into the final image. Imagen (Saharia et al. 2022) employs a cascaded diffusion model that learns to generate images from low-resolution to high-resolution. Latent diffusion models (Rombach et al. 2022) utilize an autoencoder to map the image into a lowerdimensional latent representation, performing progressive denoising steps in this latent space.

Text-to-Image Personalization. Text-to-image personalization focuses on adapting pre-trained diffusion models to incorporate new concepts with a few user-provided images. Textual Inversion (Gal et al. 2022) inverts the new concept into the text embedding space, while DreamBooth (Ruiz et al. 2023) fine-tunes the entire U-Net to learn the new concept. Custom Diffusion (Kumari et al. 2023) fine-tunes the text embedding and a few parameters in the U-Net, enabling fast tuning for multi-concept customization. Re-

cent advancements in text-to-image personalization have significantly improved identity preservation (Voynov et al. 2023; Alaluf et al. 2023; Zhou et al. 2023) and text alignment (Tewel et al. 2023; Qiu et al. 2024). Some works (Arar et al. 2024; Huang et al. 2024b) enhance text alignment by optimizing generations for specific prompts at test time. Moreover, tuning-free approaches (Wei et al. 2023; Shi et al. 2023; Li, Li, and Hoi 2023; Ye et al. 2023) focus on accelerating the personalization process. Additionally, many studies (Xiao et al. 2023; Wang et al. 2024a; Li et al. 2023; Wang et al. 2024b) concentrate on the personalized synthesis of widely interested human faces.

Text Embedding Learning. Customizing a concept by inverting it into the text embedding space was first introduced in Textual Inversion (Gal et al. 2022). XTI (Voynov et al. 2023) extends this space to be more expressive by using multiple tokens, assigning one token per attention layer. NeTI (Alaluf et al. 2023) further expands the text embedding space to depend on both the denoising timestep and the UNet layer. E4T (Gal et al. 2023) employs an encoder-based tuning method that efficiently inverts the concept into the text embedding space. AttnDreamBooth (Pang et al. 2024b) suggests optimizing the text embedding for the new concept with very few steps, as it is prone to overfitting. A concurrent work, ClassDiffusion (Huang et al. 2024a), also utilizes a super-category token to guide the learning of text embeddings for the new concept. Our work differs in several aspects. First, we use context tokens to indirectly regularize the text embedding learning without including the super-category token, whereas concurrent work focuses on narrowing the gap between the new concept and its supercategory. Second, we further regularize the attention maps of the context tokens. Third, we construct a regularization prompt set to cover a broad range of prompts, making the learned text embeddings more generalizable.

Cross-Attention Control. The control of the crossattention maps has demonstrated the effectiveness in image synthesis for diffusion models (Chefer et al. 2023). Several studies (Jin et al. 2023; Nam et al. 2024; Ma et al. 2023; Zhang et al. 2024) have also investigated controlling the attention maps for text-to-image personalization. Custom Diffusion (Kumari et al. 2023) illustrates how incorrect attention maps can lead to failed compositions involving multiple concepts. ViCo (Hao et al. 2023) proposes regularizing the attention maps to focus on meaningful regions. BreakA-Scene (Avrahami et al. 2023) enhances the generation of multiple concepts by using segmentation masks to guide the learning of the attention maps.

### Preliminaries

Latent Diffusion Models. In Latent Diffusion Model (LDM) (Rombach et al. 2022), an encoder E transforms an image x into a latent representation z = E(x) in lowerdimensional space, and a decoder D reconstructs the image from this latent code, i.e., D(E(x)) ≈ x. Furthermore, a denoising diffusion probabilistic model (Ho, Jain, and Abbeel 2020) is utilized to generate latent codes within the autoencoder’s latent space. To create images from textual descrip-

tions, the model relies on a conditioning input vector c(y), which is derived from the given text prompt y. The training objective of LDM is expressed as follows:

Ldiffusion = Ez∼E(x),y,ε∼N(0,1),t ∥ε − εθ (zt,t,c(y))∥22 ,

(1) where the denoising network εθ is used to remove the noise added to the latent code given the noised latent zt, the timestep t and the conditioning vector c(y).

Textual Inversion. Given several image examples of a target concept, Textual Inversion (TI) (Gal et al. 2022) learns the concept by inverting it into the text embedding space. TI introduces a new token S∗ and a corresponding embedding v∗. During the learning process, v∗ is optimized to minimize the diffusion loss (Eq. 1) as follows:

Ez,y,ϵ,t ∥ε − εθ(zt,t,c(y,v))∥22 , (2)

v∗ = arg min

v

where c(y,v) denotes the the conditioning vector using the optimized text embedding v.

DreamBooth. DreamBooth (Ruiz et al. 2023) fine-tunes the entire U-Net of the diffusion model to learn the target concept. It employs a rarely used token to represent the concept and fixes its text embedding during optimization. Since the entire U-Net and possibly the text encoder are fine-tuned, DreamBooth usually achieves better identity preservation than Textual Inversion.

### Method

#### Text Embedding Learning with CoRe

To achieve text-aligned generations, we aim to learn an appropriate text embedding for the new concept that is compatible with and seamlessly integrates into existing tokens. This is because text-aligned generations depend on a precise semantic understanding of the prompt, which in turn depends on the correct interactions between the new concept and the other tokens. Instead of directly improving the new concept’s embedding, we focus on constraining the context tokens surrounding the new concept. Our method derives from two key insights. First, proper output embeddings of the context tokens can only be achieved if the new concept’s input embedding is correctly learned; otherwise, it adversely impacts the output embeddings of the context tokens within the text encoder. Second, when replacing the object token in a prompt with another, the output embeddings and attention maps of the context tokens should largely remain consistent. We verify these insights through experiments illustrated in Figure 2. For instance, in the prompt “dog in the desert”, replacing “dog” with an overfitted embedding by Textual Inversion significantly alters the output embeddings and attention maps of the other tokens. In contrast, replacing “dog” with “cat” maintains the consistency of the output embeddings and attention maps.

Based on these insights, we propose Context Regularization (CoRe) that enhances the text embedding learning for the new concept by regularizing its context tokens. For a training prompt containing the new concept, we construct

Diffusion U-Net

Random a templete

{Dog} in the desert

desert

{Dog} in the

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Cross-Attention

Cross-Attention

“{Dog} in the desert ”

“{} in the desert” “{} dives into the ocean” “{} wearing a hat”

TextEnocder

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

“ S* in the desert ”

•••

Share Weights

Prompt set

in the desert in the

S*

desert

S*

“ a photo of S* ”

a photo of S*

[Figure 46]

[Figure 47]

Trainable in the first stage

Context Embedding Regularization Context Attention Regularization

Trainable in the second stage

Diffusion U-Net

- Figure 3: Overview of the proposed CoRe. Our method enhances the text embedding learning for S∗ by regularizing its context tokens. Specifically, we randomly select a regularization prompt (e.g., “S∗ in the desert”) and a reference prompt (e.g., “Dog in the desert”) from the prompt set. During training, the proposed context embedding regularization and context attention regularization are applied together with the diffusion loss, which encourages the representations of the context tokens surrounding S∗ to align with those in the reference prompt. These regularization terms make the text embedding of S∗ more compatible with existing tokens.

a reference prompt by replacing the new concept token with a super-category token. We then enforce a similarity constraint on the output embeddings and attention maps of the context tokens between these two prompts. It is important to note that our context regularization can be used with arbitrary prompts because it is applied only to the output embeddings and attention maps, without the need for generating images. Therefore, we construct a regularization prompt set designed to cover a broad range of prompts, with details provided in the Appendix.

Context Embedding Regularization. Formally, we randomly select a prompt template (e.g., “a {} in the jungle”) from the regularization prompt set, and fill it with the new concept token and its super-category token, respectively, producing a pair of prompts y∗ and ycat (e.g., “a S∗ in the jungle” and “a [super-category] in the jungle”). The input embeddings of these two prompts, {vi} and {vi′}, are then fed into the text encoder E, producing corresponding output embeddings {E(vi)} and {E(vi′)}. We minimize the average cosine distance between these two sets of output embeddings:

n

1 n − 1

Lemb =

i=1,i̸=k

1 − cos(E(vi),E(vi′)) , (3)

where k is the index corresponding to S∗ and [supercategory], n is the length of the output embeddings, and cos(·,·) denotes the cosine similarity. Note that we avoid imposing constraints between S∗ and [super-category] due to the observed significant degradation in identity preservation, as the new concept and its super-category usually exhibit substantial differences.

Context Attention Regularization. As illustrated in Figure 2, the overfitted embedding of the new concept subsequently results in incorrect attention maps for the context tokens. Therefore, we utilize attention maps to further regularize the text embedding learning for the new concept. We introduce an additional regularization term that enforces a similarity constraint between the attention maps of the two prompts, y∗ and ycat. Formally, the output embeddings {E(vi)} and {E(vi′)} for y∗ and ycat are fed into the 16 different cross-attention layers of the U-Net, generating 16 attention maps {Mi1:16} and {Mi′1:16}, respectively. We minimize the average squared difference between the mean values of these attention maps as follows:

n

1 n − 1

µ(Mi1:16) − µ(Mi′1:16) 2, (4)

Lattn =

i=1,i̸=k

where µ(M1:16) denotes the mean of all values across the 16 attention maps, k is the index corresponding to S∗, and n is the length of the prompt.

Overall, the full optimization objective of our method is defined as:

Ldiffusion + λembLemb + λattnLattn. (5)

v∗ = arg min

vk

Embedding Rescaling. As identified in (Alaluf et al. 2023; Pang et al. 2024a), during optimization, the scale of the new concept’s text embedding tends to become excessively large, leading to significant degradation in text alignment. Inspired by (Alaluf et al. 2023), we propose rescaling the norm of the text embedding during optimization to mitigate this issue. Specifically, after one optimization step, we reset the norm of the updated embedding to match the norm

Input

Custom Diffusion NeTI OFT AttnDreamBooth Ours

|[Figure 48]|[Figure 49]|
|---|---|
| |[Figure 50]|

|[Figure 51]|[Figure 52]|
|---|---|
| |[Figure 53]|

|[Figure 54]|[Figure 55]|
|---|---|
| |[Figure 56]|

|[Figure 57]|[Figure 58]|
|---|---|
| |[Figure 59]|

|[Figure 60]|[Figure 61]|
|---|---|
| |[Figure 62]|

[Figure 63]

Two S* fight each other in a boxing ring with a seething crowd in the background

|[Figure 64]|[Figure 65]|
|---|---|
| |[Figure 66]|

|[Figure 67]|[Figure 68]|
|---|---|
| |[Figure 69]|

|[Figure 70]|[Figure 71]|
|---|---|
| |[Figure 72]|

|[Figure 73]|[Figure 74]|
|---|---|
| |[Figure 75]|

|[Figure 76]|[Figure 77]|
|---|---|
| |[Figure 78]|

[Figure 79]

A S* dressed as Spider-Man swings between tall buildings using webbing

|[Figure 80]|[Figure 81]|
|---|---|
| |[Figure 82]|

|[Figure 83]|[Figure 84]|
|---|---|
| |[Figure 85]|

|[Figure 86]|[Figure 87]|
|---|---|
| |[Figure 88]|

|[Figure 89]|[Figure 90]|
|---|---|
| |[Figure 91]|

|[Figure 92]|[Figure 93]|
|---|---|
| |[Figure 94]|

[Figure 95]

A steampunk S* with gears and pipes, exploring a retro factory

|[Figure 96]|[Figure 97]|
|---|---|
| |[Figure 98]|

|[Figure 99]|[Figure 100]|
|---|---|
| |[Figure 101]|

|[Figure 102]|[Figure 103]|
|---|---|
| |[Figure 104]|

|[Figure 105]|[Figure 106]|
|---|---|
| |[Figure 107]|

|[Figure 108]|[Figure 109]|
|---|---|
| |[Figure 110]|

[Figure 111]

A S* adorned with dragon-scale armor, standing in the midst of a battlefield with a determined gaze

|[Figure 112]|[Figure 113]|
|---|---|
| |[Figure 114]|

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

[Figure 127]

A S* inside a box, floating on the water

|[Figure 128]|[Figure 129]|
|---|---|
| |[Figure 130]|

|[Figure 131]|[Figure 132]|
|---|---|
| |[Figure 133]|

|[Figure 134]|[Figure 135]|
|---|---|
| |[Figure 136]|

|[Figure 137]|[Figure 138]|
|---|---|
| |[Figure 139]|

|[Figure 140]|[Figure 141]|
|---|---|
| |[Figure 142]|

[Figure 143]

A S* crossing a river in the jungle is depicted in a painting in the style of Monet

- Figure 4: Qualitative comparison. We present personalization results of our method and four baseline methods, including Custom Diffusion (Kumari et al. 2023), NeTI (Alaluf et al. 2023), OFT (Qiu et al. 2024), and AttnDreamBooth (Pang et al. 2024b). Our method demonstrates superior performance in both text alignment and identity preservation compared to these baselines, especially for the prompts that require high visual variability of the concept. from the previous step. The rescaled embedding is given by:

This yields an editable embedding but provides a coarse depiction of the concept identity. In the second stage, we freeze the text embedding and fine-tune all layers of the U-Net to precisely capture the concept identity.

v∗s ∥v∗s∥

∥v∗s−1∥, (6)

v∗s =

where s denotes the s-th optimization step. In practice, we apply this rescaling strategy only during the intermediate phase of the optimization, as we empirically find that rescaling at the beginning or end phases can lead to degraded identity preservation, likely due to the information loss introduced by rescaling.

#### Test-Time Optimization

At test time, our proposed method, CoRe, can optionally serve as a test-time optimization technique to enhance generation for specific prompts. Specifically, given a prompt for generation, we refine the output embeddings and attention maps associated with this prompt by performing a few additional optimization steps using CoRe. This refinement is done without employing the diffusion loss. Note that in our experiments, this test-time optimization strategy is not applied when comparing with the baselines to ensure a fair comparison.

#### Embedding-to-Identity Training Strategy

Solely optimizing the text embedding is insufficient to capture the concept identity. Inspired by (Roich et al. 2022; Pang et al. 2024b), we propose a two-stage training strategy. Initially, we employ CoRe to learn a text embedding for the new concept that is compatible with existing tokens.

Input CI PM FD Ours

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

S* wearing a red sweater

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

S* and Sergey Brin sit on a sofa

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

S* wearing a doctoral cap

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

Ice sculpture of S*

- Figure 5: Face personalization results of our method and three baseline methods, including Cross Initialization (CI) (Pang et al. 2024a), PhotoMaker (PM) (Li et al. 2023), and Face2Diffusion (FD) (Shiohara and Yamasaki 2024). Our method achieves more identity-preserved face generations compared to the baselines, especially when the input image is a side face.

### Experiments

Datasets. For a comprehensive evaluation, we collect 24 concepts from previous studies (Gal et al. 2022; Ruiz et al. 2023; Kumari et al. 2023). Following (Tewel et al. 2023), we categorize these concepts into two groups: animate objects (e.g., “cat” and “child doll”) and inanimate objects (e.g., “clock” and “berry bowl”). Accordingly, we use two sets of prompts for these two groups, respectively. Some prompts are shared across all concepts, including background change, concept color change, and artistic style, while others are specific to animate objects, such as action and outfit change.

Evaluation Setup. We compare our method against four recent baseline methods: Custom Diffusion (Kumari et al.

- 2023), NeTI (Alaluf et al. 2023), OFT (Qiu et al. 2024), and AttnDreamBooth (Pang et al. 2024b). For quantitative evaluation, we employ a set of 20 prompts, detailed in the Appendix, using the following metrics: (1) identity preservation, measured by the visual similarity between the generated and input images in the CLIP-I (Radford et al. 2021) and DINO (Caron et al. 2021) feature spaces; and (2) text alignment, measured by the CLIP-T similarity between the generated images and the prompts. Following (Zeng et al.
- 2024), the CLIP-I and DINO scores are exclusively calculated on foreground-masked images to eliminate background variations and better reflect concept identity similarity. Additionally, prompts involving stylization or outfit change are excluded from the CLIP-I and DINO score cal-

- Table 1: Quantitative comparison. CLIP-I and DINO evaluate identity preservation by measuring the similarity between the generated and input images. CLIP-T evaluates text alignment by measuring the similarity between the generated images and the text prompts.

Methods CLIP-T↑ CLIP-I↑ DINO↑ Custom Diffusion 0.2537 0.6706 0.5163 NeTI 0.2386 0.7104 0.5623 OFT 0.2397 0.7018 0.5612 AttnDreamBooth 0.2547 0.6918 0.5641 Ours 0.2568 0.7054 0.5842

- Table 2: User study. For each paired comparison, our method is preferred over the baselines.

Baselines Prefer Baseline Prefer Ours

Custom Diffusion 14.3% 85.7% NeTI 23.7% 76.3% OFT 28.4% 71.6% AttnDreamBooth 35.3% 64.7%

culations because these modifications can significantly alter the concept’s appearance. The implementation details of our method and the baselines are provided in the Appendix.

#### Results

Qualitative Evaluation. In Figure 4, we present a visual comparison of personalized generations for various concepts. We employ a set of complex prompts for evaluation, such as depicting the pets in a human-like posture and dressing (e.g., “S∗ dressed as Spider-Man swings between tall buildings”), complex spatial relationships (e.g., “S∗ inside a box, floating on the water”), and composition of multiple changes (e.g., “A steampunk S∗ with gears and pipes, exploring a retro factory”). As observed, Custom Diffusion fails to generate text-aligned images and sometimes discards the new concept in the generation. NeTI and OFT struggle to accurately adapt the given concept in new scenes. AttnDreamBooth achieves improved personalized generations, but still fails to generate identity-preserved and text-aligned images, especially for prompts requires high visual variability (e.g., “A cat∗ dressed as Spider-Man”). In contrast, the generations by our method accurately preserve the concept identity and align with the complex prompts. Additional qualitative results are provided in the Appendix.

Although our method is primarily designed for personalizing general objects, it also performs well in personalizing human faces. Figure 5 shows our personalization results on human faces compared with three specialized face personalization methods, including Cross Initialization (Pang et al. 2024a), PhotoMaker (Li et al. 2023), and Face2Diffusion (Shiohara and Yamasaki 2024). Our method demonstrates superior identity preservation compared to these baselines.

Input w/o CER w/o CAR w/o Rescale Full

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

A S* wearing a police cap in a police car

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

A S* dressed as a purple wizard on a desk in a medieval library

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

A S* wearing a sombrero

- Figure 6: Ablation study. We compare models trained without Context Embedding Regularization (w/o CER), without Context Attention Regularization (w/o CAR), and without embedding rescaling strategy (w/o Rescale). All submodules are essential for achieving identity-preserved and text-aligned personalized generations.

Quantitative Evaluation. We quantitatively evaluate each method using 24 concepts and 20 text prompts, generating 32 samples per prompt for each concept. The results are presented in Table 1. Note that prompts requiring high visual variability are excluded from quantitative evaluation due to the limitations of quantitative metrics in accurately assessing the quality of generated images for these prompts, for two main reasons. First, such prompts significantly alter the concept’s appearance, which makes them unsuitable for measuring identity similarity to the input images. Second, methods that neglect to incorporate the new concept in generations tend to achieve high text alignment scores, as these scores are calculated without considering the new concept. Consequently, using relatively simple prompts, our method achieves slightly higher CLIP-T scores than AttnDreamBooth. In terms of CLIP-I and DINO scores, our method outperforms AttnDreamBooth, likely due to the insufficient text embedding learning in AttnDreamBooth. NeTI achieves the highest CLIP-I score but ranks lowest in text alignment, indicating a tendency to overfit the new concept. Overall, the results demonstrate that our method achieves a superior balance between identity preservation and text alignment compared to the baselines.

User Study. We conduct a paired human preference study to compare CoRe with the baseline methods. In each question, we present two generated images, one from our method and one from a baseline, using the same prompt. Participants are asked to evaluate the generated images based on identity preservation and text alignment. We collect 1200 responses from 60 participants. As shown in Table 2, our method is clearly preferred over the baselines, indicating its superiority in identity preservation and text alignment.

Before Test-time optim. After

Before Test-time optim. After

Input

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

S* eating an ice cream in a bowl, with sunlight streaming through the window

S* splashing through a river wearing a detective hat

Input

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

A S* dressed in pink, lying quietly next to a child on the bed

A white S* playfully chasing a fox through a whimsical forest

Figure 7: Serving as a test-time optimization technique, CoRe enables previously omitted words to be reflected in the generated images.

#### Ablation Study

In this section, we ablate each sub-module of our method to demonstrate its contribution. Figures 6 shows the results of the ablation study. As shown, the absence of the contextual embedding regularization leads to degradation in both identity preservation and text alignment. The model without the context attention regularization tends to generate images that are similar to the input, indicating a potential overfit to the concept. Additionally, without applying the embedding rescaling strategy, the model exhibits slight degradation in both text alignment and identity preservation. Additional ablation study results can be found in the Appendix.

#### Test-Time Optimization

In this section, we evaluate the effectiveness of CoRe for test-time optimization. Given a specific prompt for generation, we perform an additional 10 optimization steps using CoRe to refine the output embeddings and attention maps for this prompt. As illustrated in Figure 7, this strategy helps to better align the generations with the prompts, allowing previously omitted words to be reflected in the new images. For example, in the second row, test-time optimization effectively replaces the unintended “dog” with the correct “child”, and retrieves the missing “fox”.

### Conclusions and Limitations

In conclusion, we proposed a personalization method named CoRe that enhances the text embedding learning for the new concept by regularizing context tokens. This method is based on the insight that appropriate output embeddings of context tokens are achievable only when the new concept’s text embedding is correctly learned. Our experimental results demonstrate that CoRe outperforms the baseline methods. As shown in Figure 7, our method still faces challenges with difficult compositions involving the learned concept and other objects, which is partly inherited from the pretrained model. CoRe can serve as a test-time optimization technique to enhance the generation of such difficult compositions.

### References

Alaluf, Y.; Richardson, E.; Metzer, G.; and Cohen-Or, D. 2023. A Neural Space-Time Representation for Text-toImage Personalization. arXiv preprint arXiv:2305.15391.

Arar, M.; Voynov, A.; Hertz, A.; Avrahami, O.; Fruchter, S.; Pritch, Y.; Cohen-Or, D.; and Shamir, A. 2024. PALP: Prompt Aligned Personalization of Text-to-Image Models. arXiv preprint arXiv:2401.06105.

Avrahami, O.; Aberman, K.; Fried, O.; Cohen-Or, D.; and Lischinski, D. 2023. Break-A-Scene: Extracting Multiple Concepts from a Single Image. arXiv preprint arXiv:2305.16311.

Caron, M.; Touvron, H.; Misra, I.; J´egou, H.; Mairal, J.; Bojanowski, P.; and Joulin, A. 2021. Emerging properties in self-supervised vision transformers. In ICCV.

Chefer, H.; Alaluf, Y.; Vinker, Y.; Wolf, L.; and CohenOr, D. 2023. Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models. In SIGGRAPH.

Gal, R.; Alaluf, Y.; Atzmon, Y.; Patashnik, O.; Bermano, A. H.; Chechik, G.; and Cohen-Or, D. 2022. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618.

Gal, R.; Arar, M.; Atzmon, Y.; Bermano, A. H.; Chechik,

- G.; and Cohen-Or, D. 2023. Encoder-based domain tuning for fast personalization of text-to-image models. TOG. Hao, S.; Han, K.; Zhao, S.; and Wong, K.-Y. K. 2023. ViCo: Detail-Preserving Visual Condition for Personalized Textto-Image Generation. arXiv preprint arXiv:2306.00971. Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising diffusion probabilistic models. In NeurIPS. Huang, J.; Liew, J. H.; Yan, H.; Yin, Y.; Zhao, Y.; and Wei, Y. 2024a. ClassDiffusion: More Aligned Personalization Tuning with Explicit Class Guidance. arXiv preprint arXiv:2405.17532. Huang, M.; Mao, Z.; Liu, M.; He, Q.; and Zhang, Y. 2024b. RealCustom: Narrowing Real Text Word for Real-Time Open-Domain Text-to-Image Customization. In CVPR, 7476–7485. Jin, C.; Tanno, R.; Saseendran, A.; Diethe, T.; and Teare, P.

2023. An Image is Worth Multiple Words: Learning Object Level Concepts using Multi-Concept Prompt Learning. arXiv preprint arXiv:2310.12274.

Kumari, N.; Zhang, B.; Zhang, R.; Shechtman, E.; and Zhu, J.-Y. 2023. Multi-concept customization of text-to-image diffusion. In CVPR.

Li, D.; Li, J.; and Hoi, S. C. H. 2023. BLIPDiffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing. arXiv preprint arXiv:2305.14720.

Li, Z.; Cao, M.; Wang, X.; Qi, Z.; Cheng, M.-M.; and Shan, Y. 2023. PhotoMaker: Customizing Realistic Human Photos via Stacked ID Embedding. arXiv preprint arXiv:2312.04461.

Lu, Y.; Liu, J.; Zhang, Y.; Liu, Y.; and Tian, X. 2022. Prompt Distribution Learning. In CVPR.

Ma, J.; Liang, J.; Chen, C.; and Lu, H. 2023. SubjectDiffusion:Open Domain Personalized Text-to-Image Generation without Test-time Fine-tuning. arXiv preprint arXiv:2307.11410.

Nam, J.; Kim, H.; Lee, D.; Jin, S.; Kim, S.; and Chang, S. 2024. DreamMatcher: Appearance Matching Self-Attention for Semantically-Consistent Text-to-Image Personalization. arXiv preprint arXiv:2402.09812.

Nichol, A.; and Dhariwal, P. 2021. Improved Denoising Diffusion Probabilistic Models. In ICML.

Pang, L.; Yin, J.; Xie, H.; Wang, Q.; Li, Q.; and Mao, X. 2024a. Cross Initialization for Personalized Text-to-Image Generation. In CVPR.

Pang, L.; Yin, J.; Zhao, B.; Wu, F.; Wang, F. L.; Li, Q.; and Mao, X. 2024b. AttnDreamBooth: Towards Text-Aligned Personalized Text-to-Image Generation. arXiv preprint arXiv:2406.05000.

Qiu, Z.; Liu, W.; Feng, H.; Xue, Y.; Feng, Y.; Liu, Z.; Zhang,

- D.; Weller, A.; and Sch¨olkopf, B. 2024. Controlling text-toimage diffusion by orthogonal finetuning. In NeurIPS.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning transferable visual models from natural language supervision. In ICML.

Ramesh, A.; Dhariwal, P.; Nichol, A.; Chu, C.; and Chen, M. 2022. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125.

Roich, D.; Mokady, R.; Bermano, A. H.; and Cohen-Or, D. 2022. Pivotal tuning for latent-based editing of real images. TOG.

Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Ommer, B. 2022. High-resolution image synthesis with latent diffusion models. In CVPR.

Ruiz, N.; Li, Y.; Jampani, V.; Pritch, Y.; Rubinstein, M.; and Aberman, K. 2023. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR.

Saharia, C.; Chan, W.; Saxena, S.; Li, L.; Whang, J.; Denton,

- E. L.; Ghasemipour, K.; Gontijo Lopes, R.; Karagol Ayan, B.; Salimans, T.; et al. 2022. Photorealistic text-to-image diffusion models with deep language understanding. In NeurIPS.

Shi, J.; Xiong, W.; Lin, Z.; and Jung, H. J. 2023. Instantbooth: Personalized text-to-image generation without testtime finetuning. arXiv preprint arXiv:2304.03411.

Shiohara, K.; and Yamasaki, T. 2024. Face2Diffusion for Fast and Editable Face Personalization. In CVPR, 6850– 6859.

Sohl-Dickstein, J.; Weiss, E. A.; Maheswaranathan, N.; and Ganguli, S. 2015. Deep Unsupervised Learning using Nonequilibrium Thermodynamics. In ICML.

Tewel, Y.; Gal, R.; Chechik, G.; and Atzmon, Y. 2023. KeyLocked Rank One Editing for Text-to-Image Personalization. In SIGGRAPH.

Voynov, A.; Chu, Q.; Cohen-Or, D.; and Aberman, K. 2023. P+: Extended Textual Conditioning in Text-to-Image Generation. arXiv preprint arXiv:2303.09522.

Wang, Q.; Bai, X.; Wang, H.; Qin, Z.; and Chen, A. 2024a. InstantID: Zero-shot Identity-Preserving Generation in Seconds. arXiv preprint arXiv:2401.07519.

Wang, Q.; Jia, X.; Li, X.; Li, T.; Ma, L.; Zhuge, Y.; and Lu,

- H. 2024b. StableIdentity: Inserting Anybody into Anywhere at First Sight. arXiv preprint arXiv:2401.15975.

Wei, Y.; Zhang, Y.; Ji, Z.; Bai, J.; Zhang, L.; and Zuo, W. 2023. ELITE: Encoding Visual Concepts into Textual Embeddings for Customized Text-to-Image Generation. arXiv preprint arXiv:2302.13848.

Xiao, G.; Yin, T.; Freeman, W. T.; Durand, F.; and Han, S. 2023. FastComposer: Tuning-Free Multi-Subject Image Generation with Localized Attention. arXiv preprint arXiv:2305.10431.

Ye, H.; Zhang, J.; Liu, S.; Han, X.; and Yang, W. 2023. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721.

Zeng, Y.; Patel, V. M.; Wang, H.; Huang, X.; Wang, T.C.; Liu, M.-Y.; and Balaji, Y. 2024. JeDi: Joint-Image Diffusion Models for Finetuning-Free Personalized Text-toImage Generation. In CVPR.

Zhang, X.; Wei, X.-Y.; Wu, J.; Zhang, T.; Zhang, Z.; Lei, Z.; and Li, Q. 2024. Compositional Inversion for Stable Diffusion Models. In AAAI.

Zhou, Y.; Zhang, R.; Sun, T.; and Xu, J. 2023. Enhancing Detail Preservation for Customized Text-to-Image Generation: A Regularization-Free Approach. arXiv preprint arXiv:2305.13579.

### A Analysis of the Context Embedding

In this section, we present the analysis of the context tokens’ output embeddings, supplementing Figure 2. As illustrated, the text embedding learned by CoRe exhibits greater similarity to prompts composed solely of existing tokens, in contrast to the overfitted text embedding produced by Textual Inversion. Additionally, we present the results from CoRe when the regularization prompt set is not employed, demonstrating that this strategy significantly enhances the generalization ability of the learned text embeddings.

### B Additional Qualitative Comparisons

In Figure 9, we present additional qualitative comparisons with four baseline methods, including NeTI (Alaluf et al.

- 2023), OFT (Qiu et al. 2024), AttnDreamBooth (Pang et al.
- 2024b), and the concurrent work, ClassDiffusion (Huang et al. 2024a).

### C Additional Qualitative Results

In Figures 10 and 11, we present additional qualitative results generated by our method using a variety of prompts.

### D Results for Test-Time Optimization

In Figure 12, we present additional results of employing CoRe as a test-time optimization technique.

### E Additional Ablation Study

The results of the quantitative ablation study are presented in Table3, which are consistent with the qualitative results in Figure 6. The absence of the context embedding regularization leads to degradation in both identity preservation and text alignment. The model without the context attention regularization achieves higher CLIP-I and DINO scores but a lower CLIP-T score, indicating a tendency to overfit the concept. The model without the embedding rescaling strategy exhibits slight degradation in both text alignment and identity preservation. Additionally, Figure 13 shows more qualitative results of the ablation study.

### F Prompts for Quantitative Evaluation

Following (Tewel et al. 2023), we categorize the concepts into two groups: animate objects (e.g., “cat” and “child doll”) and inanimate objects (e.g., “clock” and “berry bowl”). Accordingly, we utilize two corresponding sets of prompts for quantitative evaluation, as detailed in Table 4.

### G Implementation Details

Our implementation is based on the publicly available Stable Diffusion v2-1. In the first training stage, the text embedding of S∗ is initialized with a super-category token. The regularization prompt set is detailed in the subsequent section. We optimize the text embedding of S∗ using CoRe for 300 steps with a batch size of 6 and a learning rate of 5e-3. The hyper-parameters λemb and λattn are set to 1.5e-4 and 0.05, respectively. Additionally, we apply the embedding rescaling strategy during the intermediate phase of the optimization (from 120 to 180 steps), as we empirically find

Cosine similarity to each word of "{Dog} in the desert" {Dog} in the desert

1.0

0.9

{Cat} in the desert {Puppy} in the desert

CosineSimilarity

0.8

###### {Our-S*} in the desert

0.7

0.6

{Our-S* w/o prompt set} in the desert

0.5

{Overfitted-S*} in the desert

0.4

{} in the desert

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

Stable Diffusion

{Dog} in the desert

Output

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

Ours

[Figure 199]

{S*} in the desert

Output

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

Ours

w/o prompt set

[Figure 205]

Input

{S* w/o ps.} in the desert

Output

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

Textual Inversion

[Figure 211]

Input

Input {Overfitted-S*} in the desert

Output

Figure 8: The cosine similarity between the output embeddings of the prompt “{dog} in the desert” and the objectswitched prompts. Here, “w/o prompt set” means that the regularization prompt set is not used, applying the proposed regularization strategies directly to the training prompt “a photo of {}”.

that rescaling at the beginning or end phases can lead to degraded identity preservation, likely due to the information loss introduced by rescaling. In the second training stage, we fine-tune the entire U-Net for 1,000 steps with a batch size of 4 and a learning rate of 2e-6. For all the baseline methods, we utilize their official implementations and follow the hyper-parameters specified in their papers. All experiments are conducted on a single Nvidia A100 GPU.

Regularization Prompt Set. As previously mentioned, CoRe can be used with any prompt since it is applied only to the output embeddings and attention maps. To enhance the generalization of the new concept’s text embedding, we design a regularization prompt set that covers a broad range of prompts. Similar to the prompts used for quantitative evaluation in Section , we employ two distinct prompts sets for animate and inanimate concepts, each containing 100 prompts. Specifically, the prompt set for inanimate concepts com-

Table 3: Quantitative ablation study. CLIP-I and DINO are used to evaluate identity preservation, while CLIP-T evaluates text alignment.

Methods CLIP-T↑ CLIP-I↑ DINO↑ w/o CER 0.2504 0.6996 0.5796 w/o CAR 0.2486 0.7123 0.6060 w/o Rescale 0.2561 0.6968 0.5805 Full 0.2568 0.7054 0.5842

prises four types of prompts: background change (e.g.,“a {} in the jungle”), concept color change (e.g.,“a black {} seen from the top”), artistic style (e.g.,“an abstract painting of a {}”), and style composition (e.g.,“a {} style skyscraper”). For animate concepts, two additional types are included: action and outfit change. The complete list of these prompt sets are provided in Figures 16 and 15.

### H User Study

Figure 14 illustrates an example question from the user study. Participants are presented with a concept image and a corresponding text prompt, followed by two generated images: one generated by our method and another by a baseline method. They are tasked with selecting the image that more accurately preserves the concept’s identity and better aligns with the text prompt. The results are summarized in Table 2.

Input NeTI OFT ClassDiffusion ADB Ours

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

A S* dressed in overalls and a wide-brimmed hat, holding a sketchpad and sitting on a stool.

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

A S* covered by sand under the moonlight on a serene beach

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

A S* in a firefighter outfit

fights a fire on the street with a water gun

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

A grand S* adorned with golden threads and a vividly red elaborate headdress, set against a backdrop of ancient stone carvings

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

a S* wearing a sombrero, sitting on the snowy street

Figure 9: Additional qualitative comparisons with four baseline methods, including NeTI (Alaluf et al. 2023), OFT (Qiu et al. 2024), ClassDiffusion (Huang et al. 2024a), and AttnDreamBooth (ADB) (Pang et al. 2024b).

[Figure 277]

Input Sample

[Figure 278]

Input Sample

[Figure 279]

Input Sample

[Figure 280]

Input Sample

[Figure 281]

Input Sample

[Figure 282]

A S* is lifting a dumbbell at the gym

[Figure 283]

A S* as Captain America in a WWII battlefield

[Figure 284]

A S* as an astronaut holding an American flag on the moon

[Figure 285]

A S* in boots was walking its pet dog down the street with a leash

[Figure 286]

S* is dressed as Gandalf the Grey from The Hobbit, wearing a long grey robe, a pointed hat, and holding a wooden staff

[Figure 287]

A S* in a detective’s trench coat and hat, eating a ice cream on the street

[Figure 288]

S* is wearing Superman’s cape and the iconic blue suit with the ’S’ emblem, standing on a rooftop in Metropolis at dawn

[Figure 289]

A S* dressed as Iron Man, complete with a tiny suit of red and gold armor, standing confidently on a futuristic city rooftop

[Figure 290]

A black S* walking with his suitcase at the airport

[Figure 291]

A green S* presenting a poster at a conference with people around

[Figure 292]

A S* as Doctor Strange in a magical sanctum

[Figure 293]

S* as a cowboy in a dusty town showdown, surrounded by cacti and a saloon

[Figure 294]

A S* as Hulk, destroying a city, with Smoke in the background

[Figure 295]

A S* as a knight draped in armor, riding a brown horse and galloping through the lush fields

[Figure 296]

A Bronze cup with two ears, in shape of S*

[Figure 297]

An oil painting of S* dressed as a musketeer in an old French town

[Figure 298]

A S* as a Jedi, with a green lightsaber, standing on a starship bridge, gazing out at distant galaxies

[Figure 299]

At the Avengers headquarters, a S* dressed in a Captain America uniform patrols the area with authority

[Figure 300]

A purple S* typing a paper on a laptop

[Figure 301]

A painting of S* as a boatman propping a boat in the lake in the style of Monet

[Figure 302]

S* wears a Green Lantern suit, standing on an outer space station’s platform, with the vast starry sky and shining nebulae in the background

[Figure 303]

A red S* holding up his accepted paper in the jungle

[Figure 304]

A painting of S* as a vintage steampunk automaton, complete with gears and complex mechanical devices

[Figure 305]

S* wears Batman’s black cape, perched on the edge of a Gotham City skyscraper, with lightning in the night sky and city lights below

[Figure 306]

A S* atop a tall tower, looking out over a bustling metropolis illustrated in a vintage comic book style

Figure 10: Additional generated images by CoRe.

[Figure 307]

Input Sample

[Figure 308]

Input Sample

[Figure 309]

A S* wearing a cozy sweater and scarf, holding a cup of hot cocoa by the fireplace

[Figure 310]

A S* dressed in Hogwarts robes, casting spells with a magical wand

[Figure 311]

[Figure 312]

S* dressed in Hogwarts robes, with a magical wand and a magical book inside an ancient library filled with towering bookshelves

A S* in a full samurai armor, standing proudly with a katana

[Figure 313]

[Figure 314]

A S* in a full samurai armor, standing proudly with a katana

A S* in a whimsical fairy costume, with delicate wings and sparkling dust

[Figure 315]

Input Sample

[Figure 316]

A S* serves as a candle holder with a flame at the top

[Figure 317]

Input Sample

[Figure 318]

A S* in a suit is enjoying a breakfast with a fork in a park

[Figure 319]

A S* in the ruins of an old cathedral, with shattered stained glass windows casting colorful light on it

[Figure 320]

A S* is eating bread in front of the Eiffel Tower

[Figure 321]

Input Sample

[Figure 322]

[Figure 323]

A S* wearing a Hawaiian shirt and flower lei, joyfully playing a ukulele under the sea

A S* dressed as a chef, with a tiny hat and apron, standing in a toy kitchen surrounded by miniature pots, pans, and a deliciouslooking toy meal

[Figure 324]

A S* in a dimly lit museum hall, with other ancient artifacts and sculpture around it

[Figure 325]

A S* is being hugged by a little girl on a plush carpet in the living room

[Figure 326]

A S* wearing a safari hat and holding a tranquilizer gun, standing among prehistoric plants

[Figure 327]

A S* adorned in a pirate outfit, with a tricorn hat and a sword, standing on the deck of a pirate ship

[Figure 328]

A S* with wings, flying above the clouds at sunset, with a flock of birds soaring alongside it

[Figure 329]

A colorful S* in the style of Monet, surrounded by a lush impressionist garden with light reflecting off its glossy surface

[Figure 330]

A S* dressed as a purple wizard sits at a desk in a medieval library

[Figure 331]

A S* dressed as a royal prince, complete with a crown and cape, sitting on a miniature throne in a lavish toy castle

[Figure 332]

A S* in the intricate, decorative patterns of Islamic art

[Figure 333]

A gummy candy in the shape of S* sits on the table in a glass candy jar

[Figure 334]

A S* in a mystical forest glade, surrounded by glowing mushrooms and ethereal light

[Figure 335]

A curious S* peeking from beneath a canvas, with art supplies on a table above

[Figure 336]

A S* wearing a warrior’s armor, holding a sword, ready to defend the kingdom

Figure 11: Additional generated images by CoRe.

Input w/o TTO w/ TTO Input w/o TTO w/ TTO

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

Watercolor painting of a S* A red S* with a beautiful bowtie wearing a sombrero in a basket sitting on a suitcase at the airport

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

A S* eating a carrot in A S* wearing a cowboy hat the rainy streets and boots holding a lasso

standing in the Wild West Figure 12: Additional results for Test-Time Optimization (TTO).

Input w/o CER w/o CAR w/o Rescle Full

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

A S* as a Batman

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

A S* hopping on stepping stones

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

A S* as a witcher

Figure 13: Additional ablation study. We compare models trained without Context Embedding Regularization (w/o CER), without Context Attention Regularization (w/o CAR), and without embedding rescaling strategy (w/o Rescale).

[Figure 376]

Figure 14: An example question of the user study. Table 4: Prompt list for quantitative evaluation, where “{}” represents S∗ or “[V] class” in some baselines.

##### Text prompts for Animate Objects Text prompts for Inanimate Objects

a photo of a {} a photo of a {} a {} floats on a river a {} floats on a river a {} in Times Square a {} in Times Square a {} in the marketplace a {} in the marketplace a {} on a stone wall in the countryside a {} on a stone wall in the countryside a {} was buried at the bottom of the river a {} was buried at the bottom of the river a {} with the Eiffel Tower in the background a {} with the Eiffel Tower in the background a red {} in the garden a red {} in the garden a pink {} by the lake a pink {} by the lake a white {} seen from the bottom a white {} seen from the bottom a curious {} exploring ancient ruins by the beach a black {} seen from the back a brave {} crossing a shallow river in the wilderness a purple {} on the dining table a {} leaping across rooftops in a city a {} with a wheat field in the background a {} sitting on a floating island amidst the clouds, surrounded by flying birds

a {} in a snowy mountain landscape

a cake with cream shaped like a {} on top a {} depicted with rough, textured brushstrokes a {} reimagined as a character from a classic fairy tale a {} depicted in the style of a Renaissance painting a {} wearing a superhero costume, complete with cape and mask

a {} painted like a Picasso abstract

a {} painted in the style of an Abstract Expressionist painting

a {} painted in the style of an Abstract Expressionist painting

a {} in the style of Van Gogh’s Starry Night a {} in the style of Van Gogh’s Starry Night a {} with intricate Baroque carvings a {} with intricate Baroque carvings

[Figure 377]

[Figure 378]

###### Figure 15: Prompt set for animate concepts. Figure 16: Prompt set for inanimate concepts.

