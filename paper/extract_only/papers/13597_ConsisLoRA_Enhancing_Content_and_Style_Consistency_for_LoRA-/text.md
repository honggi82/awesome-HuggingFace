# arXiv:2503.10614v1[cs.CV]13Mar2025

## ConsisLoRA: Enhancing Content and Style Consistency for LoRA-based Style Transfer

Bolin Chen1, Baoquan Zhao1, Haoran Xie2, Yi Cai3, Qing Li4, Xudong Mao1* 1Sun Yat-sen University 2Lingnan University 3South China University of Technology 4The Hong Kong Polytechnic University

https://consislora.github.io

|Style<br><br>Content|[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]|
|---|---|
|[Figure 8]<br><br>[Figure 9]|[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]|

Figure 1. Style transfer results of our method. Given a content image and a style reference image, ConsisLoRA enables high-fidelity stylized generations that excel in both content preservation and style alignment.

### Abstract

Style transfer involves transferring the style from a reference image to the content of a target image. Recent advancements in LoRA-based (Low-Rank Adaptation) methods have shown promise in effectively capturing the style of a single image. However, these approaches still face significant challenges such as content inconsistency, style misalignment, and content leakage. In this paper, we comprehensively analyze the limitations of the standard diffusion parameterization, which learns to predict noise, in the context of style transfer. To address these issues, we introduce ConsisLoRA, a LoRA-based method that enhances both content and style consistency by optimizing the LoRA weights to predict the original image rather than noise. We also propose a two-step training strategy that decouples the learning of content and style from the reference image. To effectively capture both the global structure and local details of the content image, we introduce a stepwise loss transition strategy. Additionally, we present an inference guid-

∗Corresponding author (xudong.xdmao@gmail.com).

ance method that enables continuous control over content and style strengths during inference. Through both qualitative and quantitative evaluations, our method demonstrates significant improvements in content and style consistency while effectively reducing content leakage.

### 1. Introduction

Diffusion models have emerged as a powerful paradigm for text-to-image synthesis, achieving significant breakthroughs in controllable generation tasks, including personalized generation [13, 38], image editing [2, 16], and image stylization [8, 55]. Despite these advancements, style transfer remains challenging due to the inherently complex and underdetermined nature of style. The goal of style transfer is to transform a content image to match a desired style from a style reference image.

Diffusion models have been extensively applied to style transfer, utilizing methods such as fine-tuning-based approaches [43, 55] and tuning-free approaches [8, 26, 45]. Recently, LoRA-based techniques [24, 28, 41] have shown remarkable efficacy in capturing style from a single image.

Notably, B-LoRA [11] separates content and style within an image by jointly learning two distinct LoRAs: one for content and another for style. However, as illustrated in Fig. 2, current LoRA-based methods still encounter significant challenges. First, accurately capturing high-level structural and stylistic features remains difficult, often resulting in outputs that are inconsistent with the original content or suffer from style misalignment. Second, the precise separation of style and content continues to be challenging, sometimes leading to content leakage [45].

Previous studies on text-to-image personalization [38, 39] have revealed that DreamBooth-LoRA [39] tends to capture major concepts of the input image (often a part of the image) rather than its entire global structure. This limitation is particularly problematic for style transfer, which requires 1) learning global style information from the entire style image, and 2) capturing the global structure of the content image to ensure content-consistent generations. We attribute these issues to the inappropriate noise prediction loss used in existing LoRA-based methods [11, 41], which fails to adequately focus on global and high-level features.

To overcome these challenges, we introduce ConsisLoRA, a novel approach that optimizes LoRA weights by predicting the original image, where the predicted image is reconstructed from the predicted noise. This reformulated loss function significantly enhances both content and style consistency for LoRA-based style transfer. To further decouple the learning of style and content, we employ a twostep training strategy: initially learning a content-consistent LoRA, followed by learning a style LoRA while keeping the content LoRA fixed. Moreover, we propose a stepwise loss transition approach to capture both the overall structure and the fine details of the content image. We also introduce an inference guidance method that allows for continuous control of content and style strengths during inference.

To demonstrate the effectiveness of ConsisLoRA, we conduct a comprehensive evaluation comparing it against four state-of-the-art baseline methods through both qualitative and quantitative assessments. The results show that ConsisLoRA outperforms the baselines in terms of content preservation and style alignment, while effectively reducing the content leakage.

### 2. Related Work

Fine-tuning Diffusion Models. Recent advancements in text-to-image models [37, 40] have leveraged fine-tuning techniques for personalization, enabling diffusion models to generate images of new concepts from several provided images. Textual Inversion [13] optimizes text embeddings to learn new concepts, while DreamBooth [38] fine-tunes the entire U-Net of the diffusion model. To enhance fine-tuning efficiency, several parameter-efficient approaches have been proposed [14, 21, 25, 35]. Notably, LoRA [21], originally

developed for fine-tuning large language models, has gained popularity in fine-tuning diffusion models due to its effectiveness and parameter efficiency. In this context, the parameterization of ϵ-prediction [20] is commonly used for fine-tuning because of its ability to produce high-quality and diverse visual outputs. In this paper, we propose replacing ϵ-prediction with x0-prediction to improve content and style consistency in style transfer. Recently, a concurrent work [15] also employs x0-prediction for high-quality dense prediction by directly altering the output of U-Net to x0. In contrast, our approach derives the predicted image from the predicted noise, without modifying the output of U-Net.

Style Transfer. Style transfer, which involves transferring the visual style from a reference image to a target content image, remains a significant challenge in computer vision [10, 18]. Recent advancements in diffusion models have revolutionized the field of style transfer. These diffusion-based methods can be primarily classified into two main categories. The first approach [1, 7, 47, 52] involves learning the style representation by fine-tuning diffusion models, such as InST [55] and StyleDrop [43]. The second approach [4, 9, 17, 22, 29, 45, 51] explores tuning-free methods to accelerate the stylization process. In particular, IP-Adapter [53] and Style-Adapter [48] train lightweight adapters to inject style features into crossattention layers of U-Nets. Some methods achieve this by utilizing large-scale datasets for training [5, 26, 34, 49]. Additionally, there is a growing trend of research attempting to improve content preservation [6, 8, 23, 27, 46, 50, 57].

LoRA-based Style Transfer. Recently, LoRA-based methods [11, 24, 28, 32, 41, 42, 54] have demonstrated the effectiveness in style transfer. These methods often involve training two separate LoRAs to capture content and style, respectively. ZipLoRA [41] introduces a method that effectively merges independently trained style and content LoRAs, enabling the generation of any subject in any style. Pair Customization [24] jointly learn content and style LoRAs by capturing the stylistic differences between a pair of content and style images. B-LoRA [11] reveals that jointly training the LoRA weights of two specific blocks within the SDXL architecture can effectively separate content and style within a single image. Despite these advancements, challenges remain in generating stylized images that preserve content structure and align the desired style.

### 3. Preliminaries

Latent Diffusion Models. The Latent Diffusion Model (LDM) [37] utilizes an autoencoder to provide a lowdimensional latent space. The encoder E maps an image

Content image Style image

Output Style image Output Style image Output

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Content inconsistency Style misalignment Content leakage

Figure 2. Examples of three significant challenges encountered by existing LoRA-based methods: 1) Content inconsistency: the structure of the generated image is inconsistent with that of the content image; 2) Style misalignment: the style of the generated image does not align with that of the style image; 3) Content leakage: content from the style image undesirably leaks into the generated image.

x to a latent representation z = E(x), and the decoder D reconstructs the image from this latent representation, i.e., D(E(x)) ≈ x. The Denoising Diffusion Probabilistic Model (DDPM) [20] is employed to train the model within the latent space of the autoencoder.

Parameterizations of Diffusion Models. DDPM [20] introduces two parameterizations of the objective function for model training: ϵ-prediction and x0-prediction. In the context of LDM, the objective functions are defined as:

0,ϵ,t ∥ϵ − ϵθ(zt,t)∥22 , (1) Lx0

Lϵ = Ez

0,ϵ,t ∥z0 − zθ(zt,t)∥22 , (2)

= Ez

where the denoising networks ϵθ and zθ are tasked with predicting the added noise and the original latent, respectively, from the noised latent zt, given a specific timestep t. ϵprediction is typically used as the training objective, as it empirically yields high-quality and diverse visual outputs.

B-LoRA. After examining the SDXL architecture [33] for LoRA optimization, B-LoRA [11] finds that jointly optimizing the LoRA weights of two specific transformer blocks (W04 and W05) effectively separates content and style within a single image. Following DreamBooth-LoRA [39], the model is fine-tuned using the diffusion loss (Eq. 1) to reconstruct the input image. One trained, the two learned LoRAs can then be used independently or together for various stylization tasks, such as style transfer and text-based stylization.

### 4. Method

#### 4.1. Analysis of ϵ-prediction for Style Transfer

The ϵ-prediction loss, as defined in Eq.1, is commonly employed as the objective function for training [20, 37] or finetuning [11, 38] diffusion models. However, as illustrated in Fig. 2, when ϵ-prediction is applied to style transfer, it leads to three significant issues: 1) the structure of the generated image is inconsistent with that of the content image, 2) the

-prediction x0-prediction

0.35

0.30

AverageLoss

0.25

0.20

0.15

0.10

0.05

0.00

0-200 200-400 400-600 600-800 800-1000 Timestep Interval

Figure 3. Comparison of the average loss across various timestep intervals for different parameterizations of diffusion models.

style of the generated image does not align with that of the style image, and 3) content from the style image leaks into the generated image.

These problems can be attributed to the inherent focus of ϵ-prediction on low-level local details rather than on highlevel structure and style. In Fig. 3, we present the average loss values of ϵ-prediction at various timestep stages, demonstrating that the loss is high at small t and diminishes as t increases. This pattern occurs because at large t, the noised image approaches pure noise, simplifying the task for the model to predict the noise. Conversely, at small t, where the noised image closely resembles the original, the model must discern fine details to effectively predict the noise. Consequently, ϵ-prediction emphasizes low-level features at early timesteps and neglects high-level features at later timesteps. Given that style transfer requires capturing the global structure of the content image and the overall style of the style image, ϵ-prediction is suboptimal for this application.

#### 4.2. ConsisLoRA

Our approach builds upon B-LoRA [11], which jointly learns content and style LoRAs corresponding to two specific blocks within SDXL from a single image. We introduce ConsisLoRA, a LoRA-based method designed to enhance content and style consistency in style transfer. ConsisLoRA is based on three main ideas. First, we replace the

###### Training Style LoRA

“A [v]”

###### “An image in the style of [v]”

###### Style image Style image

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Content LoRA Style LoRA

Content LoRA Style LoRA

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

𝜃𝑠𝑠′

𝜃𝑠𝑐 𝜃𝑠𝑐 𝜃𝑠𝑠

Shared content LoRA

Training from scratch using 𝓛𝒙𝟎

𝓛𝜺 Loss transition 𝓛𝒙𝟎

Training Content LoRA

###### Inference

“A [c] in the style of [v]”

“A [c]” Content image

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Content LoRA Style LoRA

Content LoRA Style LoRA

[Figure 46]

[Figure 47]

𝜃𝑐𝑐 𝜃𝑐𝑠 𝜃𝑐𝑐 𝜃𝑠𝑠

|𝓛𝜺|
|---|

Loss transition 𝓛𝒙𝟎

- Figure 4. Method Overview. We replace the standard ϵ-prediction with x0-prediction for training both style and content LoRAs. (Bottomleft) For training the content LoRA, we propose a loss transition strategy to capture both the global structure and local details of the content image. (Top) To disentangle the learning of style and content from the style image, we introduce a two-step training strategy: first, learn a content-consistent LoRA using the proposed loss transition, and then, train a style LoRA while keeping the content LoRA fixed.

standard ϵ-prediction loss with x0-prediction loss to address the challenges detailed in Sec. 4.1. Second, we introduce a two-step training strategy that more effectively separates the content and style representations within the style image. Third, we propose a stepwise loss transition strategy to simultaneously capture the overall structure and fine details of the content image. An overview of the proposed ConsisLoRA is depicted in Fig. 4.

##### Content- and Style-Consistent LoRA. As analyzed in

- Sec. 4.1, the ϵ-prediction loss tends to focus on low-level local details rather than high-level structure and style, making it unsuitable for style transfer. To address this, we propose replacing the traditional ϵ-prediction (Eq. 1) with the

x0-prediction (Eq. 2) for optimizing both content and style LoRAs. It is important to note that we do not directly alter the output of U-Net from the predicted noise ϵθ to the predicted latent zθ. Instead, we derive the predicted latent from the predicted noise through

√1 − α¯tϵθ √α¯t

zt −

, (3)

zˆ0 =

where α¯t = ti=1(1−βi) and {βi} represents the variance schedule. Then, we minimize the difference between the

predicted latent zˆ0 and the original latent z0 as:

0,ϵ,t ∥z0 − zˆ0∥22 . (4)

= Ez

Lzˆ0

As shown in Fig. 3, in contrast to ϵ-prediction, the proposed loss exhibits a large value for large t and a small value for small t. This behavior arises because the x0-prediction loss is scaled by a factor of

√√1−α¯αt¯t, which becomes substantial at large timesteps. This indicates that x0-prediction more effectively emphasizes high-level features compared to ϵ-prediction, as these features are primarily determined at large timesteps [16].

Stepwise Loss Transition for Content LoRA. In Fig. 15 (see Appendix E), we compare the outputs of using ϵ-prediction and x0-prediction. As shown, while x0prediction more accurately captures the global structure of the content image, it occasionally fails to retain some local details. To address this, we propose a stepwise loss transition strategy for the content LoRA. Initially, we optimize the LoRA weights using ϵ-prediction for a subset of the training steps and subsequently switch to x0-prediction for the remaining steps. As demonstrated in Fig. 15, this approach effectively preserves both the global structure and local details. We also experimented a gradual transition from ϵ-prediction to x0-prediction (e.g., a linear change over timesteps), but observed no performance gains. Importantly, this stepwise loss transition is not applied to the style LoRA, as our empirical findings suggest that ϵ-prediction in style LoRA optimization leads to the inadvertent capture of

local content details, resulting in content leakage issues (see

- Sec. 5.4).

Disentangling Style and Content for Style LoRA. To effectively separate the learning of style and content from the reference image, our strategy begins by accurately learning a content LoRA, followed by learning a style LoRA while keeping the learned content LoRA fixed. We utilize our proposed loss transition training strategy to learn a content-consistent LoRA from the reference image. As demonstrated in Fig. 9, the jointly learned style LoRA tends to exhibit content leakage, likely due to two primary reasons: 1) simultaneous optimization of style and content LoRAs can lead them to learn shared features that are relevant to both style and content, and 2) the use of ϵ-prediction in the loss transition strategy causes the style LoRA to inadvertently capture local content details. To overcome these issues, we propose training the style LoRA separately from scratch using x0-prediction, while keeping the learned content LoRA fixed. Moreover, this approach of separate training allows for more focused style learning by using a stylespecific prompt, such as “An image in the style of [v]”, instead of a generic prompt like “A [v]” used in [11], guiding the LoRA to exclusively capture style attributes.

Image Stylization Applications. Similar to B-LoRA [11], our method supports a variety of image stylization applications, as illustrated in Fig. 6. Style transfer is enabled through the integration of both content and style LoRAs, facilitating the creation of images that accurately reflect the desired content and style. Utilizing only the content LoRA enables text-based image stylization, which is controlled by a style prompt. Conversely, employing only the style LoRA allows for the generation of style-consistent images with any text-described content.

#### 4.3. Controlling with Inference Guidance

Drawing on the classifier-free guidance [19], previous research has explored various inference guidance methods for tasks such as stylization [24], image editing [56], and compositional generation [30]. Inspired by these approaches, we introduce two guidance terms that allow for continuous control over content and style strengths during inference. Formally, after optimizing LoRAs over the content and style images, we obtain four distinct sets of LoRA weights: content and style LoRA weights from the content image (denoted as θcc and θcs), and content and style LoRA weights from the style image (denoted as θsc and θss). Our inference algorithm is defined as follows:

c,θss(zt,∅)

c,θss(zt,c) − ϵθc

ϵ˜ = ϵθc

c,θss(zt,c) + λcfg ϵθc

###### (zt,ccc) − ϵθc

(zt,ccs))

+ λcont(ϵθc

c

s

###### (zt,css) − ϵθs

(zt,csc)),

+ λsty(ϵθs

s

c

(5)

where λcfg ϵθc

c,θss(zt,∅) is the classifier-free guidance term [19] with the LoRA weights θcc and θss, {λcont,λsty} control the strengths of the guidance, and {ccc,ccs,css,csc} are the text conditioning vectors for the corresponding LoRAs. The content guidance term is defined as the difference between the noises of θcc and θsc, used to enhance the content strength from the content image. Similarly, the style guidance term enhances the style strength from the style image. Note that this inference guidance is not applied in our experiments when comparing with the baselines to ensure a fair comparison.

c,θss(zt,c) − ϵθc

### 5. Experiments

#### 5.1. Implementation and Evaluation Setup

Implementation Details. Our implementation is based on SDXL v1.0 [33], with both the model weights and text encoders frozen. The rank of LoRA weights is set to 64. All LoRAs are trained on a single image. For the content image, we initially train for 500 steps using ϵ-prediction, then switch to x0-prediction for an additional 1000 steps. For the style image, we first obtain its content LoRA using the above training strategy, and then separately train a new style LoRA for 1000 steps using x0-prediction. The entire training process takes approximately 12 minutes on a single 4090 GPU. More implementation details of our method and the baselines are provided in Appendix A.

Evaluation Setup. We compare our method with four state-of-the-art stylization methods, including StyleID [8], StyleAligned [17], ZipLoRA [41], and B-LoRA [11]. For a fair comparison, we collect 20 content images and 20 style images from different studies [8, 11, 38, 43, 45]. Using these images, we compose 400 pairs of content and style images for quantitative evaluation.

#### 5.2. Results

Qualitative Evaluation. In Fig. 5, we present a visual comparison of style transfer results between our method and the baselines. As shown, the outputs produced by B-LoRA, ZipLoRA, and StyleAligned exhibit structural inconsistencies with the content image, as the ϵ-prediction loss tends to capture broad concepts rather than the precise global structure. Moreover, as observed in the first and second rows, B-LoRA sometimes suffers from style misalignment and content leakage. ZipLoRA struggles to balance the merged content and style LoRAs, sometimes neglecting the style from the reference image. Although StyleID achieves good content preservation through DDIM inversion [44], it often fails to accurately capture the style of the reference image, thereby diminishing the stylistic impact. The outputs from StyleAligned show significant structural inconsistencies with the content image and occasionally incorporate

Content Style Ours B-LoRA ZipLoRA StyleID StyleAligned

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

- Figure 5. Qualitative comparison. We present style transfer results of our method and four baseline methods, including B-LoRA [11], ZipLoRA [41], StyleID [8], and StyleAligned [17]. Our method demonstrates superior performance in preserving the structure of the content image while accurately applying the style from the reference style image.

structural elements from the reference image. In contrast, our method generates content-consistent images with accurate stylization and effectively prevents content leakage. Fig. 6 shows more results of different stylization applications using our method. Additional qualitative evaluation is provided in Appendix B.

Quantitative Evaluation. We conduct a quantitative evaluation of each method in terms of style and content alignment. Style alignment between the generated and reference images is measured using the DreamSim distance [12] and CLIP score [36]. Content alignment between the generated and content images is assessed using the DINO score [3], DreamSim distance, and CLIP score. Each method is evaluated across 400 pairs of style and content

Table 1. Quantitative comparison. We measure style and content alignment using DreamSim (DS) distance and cosine similarities calculated over CLIP and DINO features.

Style Align. Content Align. DS↓ CLIP↑ DINO↑ DS↓ CLIP↑

Methods

StyleAlign [17] 0.591 0.645 0.441 0.561 0.647 StyleID [8] 0.653 0.638 0.679 0.494 0.693 ZipLoRA [41] 0.646 0.643 0.488 0.543 0.668 B-LoRA [11] 0.573 0.654 0.536 0.568 0.643 Ours 0.567 0.659 0.629 0.524 0.671

images, with the results detailed in Tab. 1. StyleID achieves the best performance in content alignment but ranks lowest in style alignment. This is consistent with the qualitative

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

Style

Content

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

[Figure 108]

[Figure 109]

Content input “Pixel art” “Ice” “Vintage” “Cyberpunk” “Sketch cartoon” “Gothic dark”

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

Style input “Dog” “Car” “Train” “Bench” “Classroom” “Bedroom”

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

- Figure 6. Results generated by ConsisLoRA for three image stylization tasks: (Top) Transferring the style from a reference image to the content of a target image; (Middle) Applying the style described by prompts to a content image; (Bottom) Generating objects described by prompts with the style extracted from a reference image.

observations, where StyleID often diminishes the stylistic impact. Besides this extreme case, our method outperforms all baselines in both style and content alignment. In particular, compared to B-LoRA, our method shows significant improvements in content alignment, especially evident in the DINO score. While B-LoRA achieves a CLIP score comparable to ours in style alignment, this score may be inflated due to content leakage from the reference image.

User Study. We also conducted a user study to evaluate our method. In this study, participants were presented with a content image, a reference image, and two stylized images: one generated by our method and the other by a baseline method. Participants were tasked with selecting the image that better aligns with the style of the reference image while preserving the content of the content image. We collected a total of 1,500 responses from 50 participants, as detailed in Tab. 2. The results demonstrate a strong preference for our method.

Table 2. User Study. Participants were presented with two images: one by our method and another by a baseline method. The results demonstrate a clear preference for our method.

Baselines Prefer Baseline Prefer Ours

B-LoRA [11] 24.3% 75.7% ZipLoRA [41] 11.7% 88.3% StyleID [8] 19.2% 80.8% StyleAligned [17] 10.4% 89.6%

Content and Style Decomposition. Given a single input image, we compare our method with B-LoRA for content and style decomposition, applying the content and style LoRAs separately, as illustrated in Fig. 7. When B-LoRA employs content LoRA with new styles described by text prompts, it struggles to preserve the global structure of the input image and fails to align the generated images with the specified styles in the prompts. Additionally, B-LoRA is unable to learn a disentangled style LoRA from the input

Input

Ours B-LoRA

Ours B-LoRA

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

Content LoRA

[Figure 128]

“A [c] in watercolor style”

“A [c] in vintage style”

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

Style LoRA

“A street in the style of [v]”

“A market in the style of [v]”

- Figure 7. Content and style decomposition. Our method achieves a more accurate and disentangled decomposition of content and style compared to the baseline method.

Input Original Content↑ Style↑ C&S↑

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

- Figure 8. Inference guidance. Increasing the strengths of content and style guidance correspondingly enhances their impact on the generated images. Zoom in for best view.

image, resulting in severe content leakage issues in the generated images. In contrast, our method effectively disentangles the content and style of the input image, demonstrating clear advantages in content and style decomposition. More decomposition results are provided in Appendix C.

#### 5.3. Inference Guidance

In this section, we evaluate the proposed inference guidance described in Sec. 4.3 for controlling content and style strengths during inference. As shown in Fig. 8, increasing the content and style strengths correspondingly enhances their impact on the generated images. Furthermore, a detailed comparison between our inference guidance and the approach of scaling LoRA weights is provided in Appendix F. We observe that our method more effectively preserves the content structure when adjusting content strength. In terms of adjusting style strength, both methods are capable of generating high-quality stylized images.

#### 5.4. Ablation Study

We conduct an ablation study to evaluate the effectiveness of each component of our method. Specifically, we assess three variants: 1) replacing x0-prediction with ϵ-prediction,

Input Var A Var B Var C Full

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

Figure 9. Ablation study. We evaluate three variants of our model: 1) replacing x0-prediction with ϵ-prediction (Var A), 2) removing the two-step training strategy for style LoRA (Var B), and 3) using x0-prediction alone instead of loss transition for content LoRA (Var C). Zoom in for best view.

2) removing the two-step training strategy for style LoRA, and 3) employing x0-prediction alone instead of loss transition for content LoRA. Fig. 9 presents a visual comparison of the stylized images generated by each variant. The results underscore the crucial role of each component. Without using x0-prediction, the model fails to capture both the global structure of the content image and style features from the style image. Removing the two-step training strategy for style LoRA leads to significant content leakage issues. Moreover, employing x0-prediction alone for content LoRA causes the model to struggle with capturing local details (e.g., the pictures hanging on the wall in the top row). More results of the ablation study are provided in Appendix D.

### 6. Conclusions and Limitations

In this study, we introduced ConsisLoRA, a style transfer method designed to address critical challenges faced by existing LoRA-based methods, such as content inconsistency, style misalignment, and content leakage. By optimizing LoRA weights to predict the original image rather than noise, our approach significantly enhances both style and content consistency. Our two-step training strategy effectively separates the learning of content and style, facilitating the disentanglement of these elements. Additionally, our stepwise loss transition strategy ensures the preservation of both global structures and local details of the content image. Despite these advancements, our approach does exhibit some limitations. First, similar to other LoRA-based methods, our content LoRA often neglects the color of objects, a factor that may be crucial for certain applications. Second, our method faces challenges in preserving the identity of individuals, due to the limited capacity of LoRAs. We aim to focus our efforts on enhancing identity preservation of individuals in our future work.

### References

- [1] Namhyuk Ahn, Junsoo Lee, Chunggi Lee, Kunhee Kim, Daesik Kim, Seung-Hun Nam, and Kibeom Hong. Dreamstyler: Paint by style inversion with text-to-image diffusion models. In AAAI, 2024. 2
- [2] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In CVPR, 2023. 1
- [3] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021. 6
- [4] Dar-Yen Chen, Hamish Tennent, and Ching-Wen Hsu. Artadapter: Text-to-image style transfer using multi-level style encoder and explicit adaptation. In CVPR, 2024. 2
- [5] Jingwen Chen, Yingwei Pan, Ting Yao, and Tao Mei. Controlstyle: Text-driven stylized image generation using diffusion priors. In ACM MM, 2023. 2
- [6] Hansam Cho, Jonghyun Lee, Seunggyu Chang, and Yonghyun Jeong. One-shot structure-aware stylized image synthesis. In CVPR, 2024. 2
- [7] Jooyoung Choi, Chaehun Shin, Yeongtak Oh, Heeseung Kim, and Sungroh Yoon. Style-friendly snr sampler for styledriven generation. arXiv preprint arXiv:2411.14793, 2024. 2
- [8] Jiwoo Chung, Sangeek Hyun, and Jae-Pil Heo. Style injection in diffusion: A training-free approach for adapting largescale diffusion models for style transfer. In CVPR, 2024. 1, 2, 5, 6, 7, 11, 12
- [9] Yingying Deng, Xiangyu He, Fan Tang, and Weiming Dong. Z*: Zero-shot style transfer via attention reweighting. In CVPR, 2024. 2
- [10] Alexei A. Efros and William T. Freeman. Image quilting for texture synthesis and transfer. In SIGGRAPH, 2001. 2
- [11] Yarden Frenkel, Yael Vinker, Ariel Shamir, and Daniel Cohen-Or. Implicit style-content separation using b-lora. In ECCV, 2024. 2, 3, 5, 6, 7, 11, 12
- [12] Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. Dreamsim: Learning new dimensions of human visual similarity using synthetic data. In NeurIPS, 2024. 6
- [13] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 1, 2
- [14] Ligong Han, Yinxiao Li, Han Zhang, Peyman Milanfar, Dimitris Metaxas, and Feng Yang. Svdiff: Compact parameter space for diffusion fine-tuning. arXiv preprint arXiv:2303.11305, 2023. 2
- [15] Jing He, Haodong LI, Wei Yin, Yixun Liang, Leheng Li, Kaiqiang Zhou, Hongbo Zhang, Bingbing Liu, and YingCong Chen. Lotus: Diffusion-based visual foundation model for high-quality dense prediction. In ICLR, 2025. 2
- [16] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt im-

- age editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 1, 4
- [17] Amir Hertz, Andrey Voynov, Shlomi Fruchter, and Daniel Cohen-Or. Style aligned image generation via shared attention. In CVPR, 2024. 2, 5, 6, 7, 11, 12
- [18] Aaron Hertzmann, Charles E. Jacobs, Nuria Oliver, Brian Curless, and David H. Salesin. Image analogies. In SIGGRAPH, 2001. 2
- [19] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 5
- [20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 2, 3
- [21] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 2
- [22] Jaeseok Jeong, Junho Kim, Yunjey Choi, Gayoung Lee, and Youngjung Uh. Visual style prompting with swapping selfattention. arXiv preprint arXiv:2402.12974, 2024. 2
- [23] Ruixiang Jiang and Changwen Chen. Diffartist: Towards aesthetic-aligned diffusion model control for training-free text-driven stylization. arXiv preprint arXiv:2407.15842,

2024. 2

- [24] Maxwell Jones, Sheng-Yu Wang, Nupur Kumari, David Bau, and Jun-Yan Zhu. Customizing text-to-image models with a single image pair. In SIGGRAPH Asia, 2024. 1, 2, 5
- [25] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In CVPR, 2023. 2
- [26] Wen Li, Muyuan Fang, Cheng Zou, Biao Gong, Ruobing Zheng, Meng Wang, Jingdong Chen, and Ming Yang. Styletokenizer: Defining image style by a single instance for controlling diffusion models. In ECCV, 2024. 1, 2
- [27] Kuan Heng Lin, Sicheng Mo, Ben Klingher, Fangzhou Mu, and Bolei Zhou. Ctrl-x: Controlling structure and appearance for text-to-image generation without guidance. In NeurIPS, 2024. 2
- [28] Chang Liu, Viraj Shah, Aiyu Cui, and Svetlana Lazebnik. Unziplora: Separating content and style from a single image. arXiv preprint arXiv:2412.04465, 2024. 1, 2
- [29] Jin Liu, Huaibo Huang, Chao Jin, and Ran He. Portrait diffusion: Training-free face stylization with chain-of-painting. arXiv preprint arXiv:2312.02212, 2023. 2
- [30] Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B. Tenenbaum. Compositional visual generation with composable diffusion models. In ECCV, 2022. 5
- [31] Mkshing. Ziplora-pytorch. https://github.com/ mkshing/ziplora-pytorch. 11
- [32] Ziheng Ouyang, Zhen Li, and Qibin Hou. K-lora: Unlocking training-free fusion of any subject and style loras. In CVPR,

2025. 2

- [33] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 3, 5, 11

- [34] Tianhao Qi, Shancheng Fang, Yanze Wu, Hongtao Xie, Jiawei Liu, Lang Chen, Qian He, and Yongdong Zhang. Deadiff: An efficient stylization diffusion model with disentangled representations. In CVPR, 2024. 2
- [35] Zeju Qiu, Weiyang Liu, Haiwen Feng, Yuxuan Xue, Yao Feng, Zhen Liu, Dan Zhang, Adrian Weller, and Bernhard Sch¨olkopf. Controlling text-to-image diffusion by orthogonal finetuning. In NeurIPS, 2024. 2
- [36] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 6
- [37] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2, 3
- [38] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, 2023. 1, 2, 3, 5
- [39] Simo Ryu. Low-rank adaptation for fast text-to-image diffusion fine-tuning. https : / / github . com / cloneofsimo/lora. 2, 3, 11
- [40] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. In NeurIPS, 2022. 2
- [41] Viraj Shah, Nataniel Ruiz, Forrester Cole, Erika Lu, Svetlana Lazebnik, Yuanzhen Li, and Varun Jampani. Ziplora: Any subject in any style by effectively merging loras. In ECCV,

2024. 1, 2, 5, 6, 7, 11, 12

- [42] Donald Shenaj, Ondrej Bohdal, Mete Ozay, Pietro Zanuttigh, and Umberto Michieli. Lora.rar: Learning to merge loras via hypernetworks for subject-style conditioned image generation. arXiv preprint arXiv:2412.05148, 2024. 2
- [43] Kihyuk Sohn, Nataniel Ruiz, Kimin Lee, Daniel Castro Chin, Irina Blok, Huiwen Chang, Jarred Barber, Lu Jiang, Glenn Entis, Yuanzhen Li, et al. Styledrop: Text-to-image generation in any style. arXiv preprint arXiv:2306.00983,

2023. 1, 2, 5

- [44] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 5
- [45] Haofan Wang, Matteo Spinelli, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen. Instantstyle: Free lunch towards style-preserving in text-to-image generation. arXiv preprint arXiv:2404.02733, 2024. 1, 2, 5
- [46] Haofan Wang, Peng Xing, Renyuan Huang, Hao Ai, Qixun Wang, and Xu Bai. Instantstyle-plus: Style transfer with content-preserving in text-to-image generation. arXiv preprint arXiv:2407.00788, 2024. 2
- [47] Ye Wang, Tongyuan Bai, Xuping Xie, Zili Yi, Yilin Wang, and Rui Ma. Sigstyle: Signature style transfer via personalized text-to-image models. In AAAI, 2025. 2
- [48] Zhouxia Wang, Xintao Wang, Liangbin Xie, Zhongang Qi, Ying Shan, Wenping Wang, and Ping Luo. Styleadapter:

- A single-pass lora-free model for stylized image generation. arXiv preprint arXiv:2309.01770, 2023. 2
- [49] Peng Xing, Haofan Wang, Yanpeng Sun, Qixun Wang, Xu Bai, Hao Ai, Renyuan Huang, and Zechao Li. Csgo: Content-style composition in text-to-image generation. arXiv preprint arXiv:2408.16766, 2024. 2
- [50] Youcan Xu, Zhen Wang, Jun Xiao, Wei Liu, and Long Chen. Freetuner: Any subject in any style with training-free diffusion. arXiv preprint arXiv:2405.14201, 2024. 2
- [51] Yifei Xu, Xiaolong Xu, Honghao Gao, and Fu Xiao. Sgdm: An adaptive style-guided diffusion model for personalized text to image generation. TMM, 2024. 2
- [52] Tao Yang, Rongyuan Wu, Peiran Ren, Xuansong Xie, and Lei Zhang. Pixel-aware stable diffusion for realistic image super-resolution and personalized stylization. In ECCV,

2024. 2

- [53] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 2

- [54] Gong Zhang, Kihyuk Sohn, Meera Hahn, Humphrey Shi, and Irfan Essa. Finestyle: Fine-grained controllable style personalization for text-to-image models. In NeurIPS, 2024. 2
- [55] Yuxin Zhang, Nisha Huang, Fan Tang, Haibin Huang, Chongyang Ma, Weiming Dong, and Changsheng Xu. Inversion-based style transfer with diffusion models. In CVPR, 2023. 1, 2
- [56] Zhixing Zhang, Ligong Han, Arnab Ghosh, Dimitris Metaxas, and Jian Ren. Sine: Single image editing with textto-image diffusion models. In CVPR, 2023. 5
- [57] Zhanjie Zhang, Quanwei Zhang, Guangyuan Li, Wei Xing, Lei Zhao, Jiakai Sun, Zehua Lan, Junsheng Luan, Yiling Huang, and Huaizhong Lin. Artbank: Artistic style transfer with pre-trained diffusion model and implicit style prompt bank. In AAAI, 2024. 2

### A. Implementation Details

Our model leverages SDXL v1.0 [33], with the model weights and text encoders fixed. We employ the Adam optimizer to tune the LoRA weights with a rank of 64. For the content image, training initially involves a learning rate of 2 × 10−4 for 500 steps using the ϵ-prediction loss, followed by a transition to the x0-prediction loss at a learning rate of 1 × 10−4 for an additional 1000 steps. For the style image, we first establish its content LoRA using the aforementioned strategy, then train a new style LoRA from scratch for 1000 steps using x0-prediction. The entire training process requires approximately 12 minutes on a single Nvidia 4090 GPU. For baselines, including B-LoRA [11], StyleID [8], and StyleAligned [17], we utilize their official implementations. In the absence of an official implementation for ZipLoRA [41], we rely on a communitydeveloped version [31]. Since StyleAligned focuses solely on consistent style generation, we follow [11, 17] to use DreamBooth-LoRA [39] to provide the content.

### B. Additional Qualitative Results

In Fig. 10, we present additional qualitative comparisons against the baseline methods using various pairs of style and content images. Our method demonstrates superior performance in both content preservation and style alignment compared to the baselines. Moreover, in Figs. 11 and 12, we provide additional qualitative results generated by ConsisLoRA for different image stylization applications.

### C. Content and Style Decomposition

In Fig. 13, we present an additional decomposition comparison between our method and B-LoRA. As illustrated, our method shows clear advantages in accurately capturing both the content structure and style features.

### D. Additional Ablation Study

As described in Sec. 5.4, we conduct an ablation study by removing each component of our method. Specifically, we evaluate three variants: 1) replacing x0-prediction with ϵprediction (w/o x0-prediction), 2) removing the two-step training strategy for style LoRA (w/o two-step training), and 3) using x0-prediction alone instead of loss transition for content LoRA (w/o loss transition). In Fig. 14, we provide additional ablation study results for each variant. The results demonstrate the crucial role of each component.

### E. Comparing ϵ-prediction with x0-prediction

In Fig. 15, we present a qualitative comparison of four different loss schemes used for training the content LoRA: 1) using ϵ-prediction only, 2) using x0-prediction only, 3) transitioning from x0-prediction to ϵ-prediction (x0 → ϵ),

and 4) transitioning from ϵ-prediction to x0-prediction (ϵ → x0). Similar to B-LoRA, ϵ-prediction alone does not effectively capture the global structure of the content image. In contrast, x0-prediction alone more accurately captures the global structure but falls short in retaining some local details. The transition of x0 → ϵ achieves a compromised performance between ϵ-prediction and x0-prediction. The proposed transition of ϵ → x0 achieves the best performance in content preservation, accurately capturing both global structure and local details. This suggests that directly switching the original pre-training loss to a new loss may make the model struggle to adapt.

### F. Inference Guidance

In Figs. 16 and 17, we compare the proposed inference guidance with the method of scaling LoRA weights for controlling the content and style strengths. When increasing the content strength, both approaches effectively enhance the impact from the content image. However, scaling LoRA weights sometimes leads to distortions in some local details. In terms of adjusting style strength, both methods exhibit comparable performance, successfully generating high-quality stylized images.

### G. Visualization of Attention Maps

- In Fig. 18, we visualize the attention maps corresponding to the style token “[v]” for both our method and B-LoRA. As shown, B-LoRA tends to focus more on certain local details compared to our method. This emphasis on local details contributes to issues of content leakage and style misalignment in B-LoRA.

H. Portrait Stylization

- In Fig. 19, we present the portrait stylization results for both our method and B-LoRA. Our method demonstrates superior performance in preserving human identity compared to B-LoRA.

Content Style Ours B-LoRA ZipLoRA StyleID StyleAligned

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

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

Figure 10. Additional qualitative comparison. We present style transfer results of our method and four baseline methods, including B-LoRA [11], ZipLoRA [41], StyleID [8], and StyleAligned [17]. Our method demonstrates superior performance in both content preservation and style alignment.

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

Style

Content

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

Figure 11. Additional style transfer results by ConsisLoRA.

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

Style

Content

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

[Figure 277]

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

Content input “Pixel art” “Ice” “Vintage” “Sketch cartoon” “Cyberpunk” “Gothic dark”

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

Style input “Dog” “Car” “Train” “Bench” “Classroom” “Bedroom”

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

Figure 12. Additional qualitative results by ConsisLoRA for three image stylization tasks, including style transfer (top), text-based image stylization (middle), and consistent style generation (bottom).

Ours B-LoRA

Ours B-LoRA

Input Content LoRA

Input Content LoRA

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

“A [c] in pixel art style”

“A [c] in gothic dark style”

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

Style LoRA

Style LoRA

“A playground in the style of [v]”

“A sofa in the style of [v]”

Figure 13. Additional results of content and style decomposition.

Content Image Style Image w/o x0-prediction w/o two-step training w/o loss transition Full

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

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

- Figure 14. Additional results of the ablation study. We evaluate three variants of our model: replacing x0-prediction with ϵ-prediction (w/o x0-prediction), 2) removing the two-step training strategy for style LoRA (w/o two-step training), and 3) using x0-prediction alone instead of loss transition for content LoRA (w/o loss transition).

Content Image Style Image ϵ-prediction x0-prediction x0 → ϵ ϵ → x0

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

- Figure 15. Qualitative comparison of four different loss schemes used for training the content LoRA: 1) using ϵ-prediction only, 2) using x0-prediction only, 3) transitioning from x0-prediction to ϵ-prediction (x0 → ϵ), and 4) transitioning from ϵ-prediction to x0-prediction (ϵ → x0).

Content Style Inference guidance strength (content↑)

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

- 0.0 1.0 2.0 3.0 4.0

Content Style LoRA weight scaling (content↑)

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

- 1.0 1.1 1.2 1.3 1.4

Content Style Inference guidance strength (style↑)

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

- 0.0 1.0 2.0 3.0 4.0

Content Style LoRA weight scaling (style↑)

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

- 1.0 1.1 1.2 1.3 1.4

Figure 16. Qualitative comparison between the proposed inference guidance and scaling LoRA weight.

Content Style Inference guidance strength (content↑)

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

- 0.0 1.0 2.0 3.0 4.0

Content Style LoRA weight scaling (content↑)

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

- 1.0 1.1 1.2 1.3 1.4

Content Style Inference guidance strength (style↑)

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

- 0.0 1.0 2.0 3.0 4.0

Content Style LoRA weight scaling (style↑)

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

- 1.0 1.1 1.2 1.3 1.4

Figure 17. Qualitative comparison between the proposed inference guidance and scaling LoRA weight.

Input Ours B-LoRA Ours B-LoRA

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

Attention map visualization of “[v]” “A street in the style of [v]”

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

Attention map visualization of “[v]” “A bedroom in the style of [v]”

Figure 18. We visualize the attention maps corresponding to the style token “[v]” and observe that B-LoRA tends to focus more on certain local details compared to our method. This focus contributes to the issues of content leakage and style misalignment.

Content Image Style Image

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

Ours

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

B-LoRA

Content Image Style Image

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

Ours

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

B-LoRA

Figure 19. Qualitative comparison of portrait stylization between our method and B-LoRA.

