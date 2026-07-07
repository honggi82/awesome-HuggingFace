## FaceStudio: Put Your Face Everywhere in Seconds

# arXiv:2312.02663v2[cs.CV]6Dec2023

Yuxuan Yan∗ Chi Zhang∗† Rui Wang Yichao Zhou Gege Zhang Pei Cheng Bin Fu Gang Yu

Tencent

{yuxuanyan, johnczhang, raywwang, yichaozhou, gretazhang, peicheng, brianfu, skicyyu}@tencent.com

https://icoz69.github.io/facestudio/

Multi-Human Image Synthesis Stylized Portrait Synthesis Novel View Synthesis

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

Mixing

+

Portrait Synthesis by Mixed Guidance

Exemplar Style

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Prompt: Cartoon style

Figure 1. Applications of our proposed framework for identity-preserving image synthesis. Our method can preserve the subject’s identity in the synthesized images with high fidelity.

### Abstract

This study investigates identity-preserving image synthesis, an intriguing task in image generation that seeks to maintain a subject’s identity while adding a personalized, stylistic touch. Traditional methods, such as Textual Inversion and DreamBooth, have made strides in custom image creation, but they come with significant drawbacks. These include the need for extensive resources and time for finetuning, as well as the requirement for multiple reference images. To overcome these challenges, our research introduces a novel approach to identity-preserving synthesis, with a particular focus on human images. Our model leverages a direct feed-forward mechanism, circumventing the need for intensive fine-tuning, thereby facilitating quick and

*Equal contributions. †Corresponding Author.

efficient image generation. Central to our innovation is a hybrid guidance framework, which combines stylized images, facial images, and textual prompts to guide the image generation process. This unique combination enables our model to produce a variety of applications, such as artistic portraits and identity-blended images. Our experimental results, including both qualitative and quantitative evaluations, demonstrate the superiority of our method over existing baseline models and previous works, particularly in its remarkable efficiency and ability to preserve the subject’s identity with high fidelity.

### 1. Introduction

In recent years, artificial intelligence (AI) has driven significant progress in the domain of creativity, leading to transformative changes across various applications. Particularly,

text-to-image diffusion models [41, 42] have emerged as a notable development, capable of converting textual descriptions into visually appealing, multi-styled images. Such advancements have paved the way for numerous applications that were once considered to be beyond the realms of possibility.

However, despite these advancements, several challenges remain. One of the most prominent is the difficulty faced by existing text-to-image diffusion models in accurately capturing and describing an existing subject based solely on textual descriptions. This limitation becomes even more evident when detailed nuances, like human facial features, are the subject of generation. Consequently, there is a rising interest in the exploration of identity-preserving image synthesis, which encompasses more than just textual cues. In comparison to standard text-to-image generation, it integrates reference images in the generative process, thereby enhancing the capability of models to produce images tailored to individual preferences.

In pursuit of this idea, various methods have been proposed, with techniques such as DreamBooth [43] and Textual inversion [14] leading the way. They primarily focus on adjusting pre-trained text-to-image models to align more closely with user-defined concepts using reference images. However, these methods come with their set of limitations. The fine-tuning process, essential to these methods, is resource-intensive and time-consuming, often demanding significant computational power and human intervention. Moreover, the requirement for multiple reference images for accurate model fitting poses additional challenges.

In light of these constraints, our research introduces a novel approach focusing on identity-preserving image synthesis, especially for human images. Our model, in contrast to existing ones, adopts a direct feed-forward approach, eliminating the cumbersome fine-tuning steps and offering rapid and efficient image generation. Central to our model is a hybrid guidance module, which guides the image generation of the latent diffusion model. This module not only considers textual prompts as conditions for image synthesis but also integrates additional information from the style image and the identity image. By employing this hybridguided strategy, our framework places additional emphasis on the identity details from a given human image during generations. To effectively manage images with multiple identities, we develop a multi-identity cross-attention mechanism, which enables the model to aptly associate guidance particulars from various identities with specific human regions within an image.

Our training method is intuitive yet effective. Our model can be easily trained with human image datasets. By employing images with the facial features masked as the style image input and the extracted face as the identity input, our model learns to reconstruct the human images while high-

lighting identity features in the guidance. After training, our model showcases an impressive ability to synthesize human images that retain the subject’s identity with exceptional fidelity, obviating the need for any further adjustments. A unique aspect of our method is its ability to superimpose a user’s facial features onto any stylistic image, such as a cartoon, enabling users to visualize themselves in diverse styles without compromising their identity. Additionally, our model excels in generating images that amalgamate multiple identities when supplied with the respective reference photos. Fig. 1 shows some applications of our model.

This paper’s contributions can be briefly summarized as follows:

- • We present a tuning-free hybrid-guidance image generation framework capable of preserving human identities under various image styles.
- • We develop a multi-identity cross-attention mechanism, which exhibits a distinct ability to map guidance details from multiple identities to specific human segments in an image.
- • We provide comprehensive experimental results, both qualitative and quantitative, to highlight the superiority of our method over baseline models and existing works, especially in its unmatched efficiency.

### 2. Related Work

Text-to-image diffusion models. Diffusion models have recently come to the forefront of generative model research. Their exceptional capability to generate high-quality and diverse images has placed them at a distinct advantage over their predecessors, namely the GAN-based and autoregressive image generation models. This new generation of diffusion models, having the capacity to produce stateof-the-art synthesis results, owes much of its success to being trained on image-text pairs data at a billion scale. The integration of textual prompts into the diffusion model serves as a vital ingredient in the development of various text-to-image diffusion models. Distinguished models in this domain include GLIDE [38], DALL·E 2 [41], Imagen [45], and StableDiffusion [42]. These models leverage text as guidance during the image generation process. Consequently, they have shown considerable proficiency in synthesizing high-quality images that closely align with the provided textual description. Compared to previous efforts [29, 39, 51, 69] on face image synthesis based on Generative Adversarial Networks (GANs) [25], diffusion models exhibit greater stability during training and demonstrate enhanced capability in effectively integrating diverse forms of guidance, such as texts and stylized images.

However, textual descriptions for guidance, while immensely useful, often fall short when it comes to the generation of complex and nuanced details that are frequently associated with certain subjects. For instance, generating

[Figure 27]

[Figure 28]

[Figure 29]

ei

eh

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

Clip Image Encoder

| | | | | |
|---|---|---|---|---|

[Figure 34]

U-Net

[Figure 35]

ef

[Figure 36]

[Figure 37]

Crop

Generated Image

[Figure 38]

Linear

ID Extractor

Addition

eg

ef

| |
|---|
| |
| |
| |
| |

[Figure 39]

Frozen modules

[Figure 40]

[Figure 41]

| | | | | |
|---|---|---|---|---|

"A man in the style of Manmaru"

Prior Model

[Figure 42]

Trainable modules

eT

ep

- Figure 2. Hybrid-Guidance Identity-Preserving Image Synthesis Framework. Our model, built upon StableDiffusion, utilizes text prompts and reference human images to guide image synthesis while preserving human identity through an identity input.

images of human faces proves challenging with this approach. While generating images of celebrities might be feasible due to the substantial presence of their photographs in the training data that can be linked to their names, it becomes an uphill task when it comes to generating images of ordinary people using these text-to-diffusion models.

encoder-based including Composer [21], ProFusion [68] MagiCapture [22], IP-Adapter [59], ELITE [52], DisenBooth [8], Face0 [50], PhotoVerse [9], AnyDoor [11], SingleInsert [54], etc. [24, 35, 56].

Alongside optimization-based methods, a number of tuning-free methods have been concurrently proposed, such as InstantBooth [47]. This method converts input images into a textual token for general concept learning and introduces adapter layers to obtain rich image representation for generating fine-grained identity details. However, it comes with the drawback of having to train the model separately for each category. Bansal et al.[6] put forward a universal guidance algorithm that enables diffusion models to be controlled by arbitrary guidance modalities without the need to retrain the network. Similarly, Yang et al.[57] propose an inpainting-like diffusion model that learns to inpaint a masked image under the guidance of example images of a given subject. Similar methods leveraged by inversion-based personalization including Cao et al. [7], Han et al. [17], Gu et al. [16], Mokady et al. [37]. Besides, Controlnet [67] is also an effective way to personalize. In addition, there are some solutions that use editing to achieve the purpose of maintaining identity, such as SDEdit [36], Imagic [26], etc. [18, 48]. Based on all the solutions mentioned above, the method of combining multiple objects while maintaining identity is also widely used in Break-a-scene [5], FastComposer [55], Cones [34] and MultiConceptCustomDiffusion [27].

Subject-driven image generation. Subject-driven image generation seeks to overcome the limitations posed by text-to-image synthesis models. Central to this novel research area is the inclusion of subject-specific reference images, which supplement the textual description to yield more precise and personalized image synthesis. To this end, several optimization-based methods have been employed, with popular ones being Low-Rank Adaptation (LoRA), DreamBooth [43], Textual Inversion [14], and Hypernetwork [3]. These methods typically involve fine-tuning a pre-trained text-to-image framework or textual embeddings to align the existing model with user-defined concepts, as indicated by a set of example images. There are several other methods derived from these works such as Unitune [49], HyperDreamBooth [44], EasyPhoto [53], FaceChain [33], etc. [4, 23, 60]. However, these methods pose some challenges. For one, they are time-consuming due to the need for model fine-tuning, which also requires multiple reference images to achieve accurate fitting. Overfitting also becomes a potential issue with these optimization-based methods, given their heavy reliance on example images. In response to these challenges, recent studies have proposed various improved methods. DreamArtist [13], for instance, mitigates the problem of overfitting by incorporating both positive and negative embeddings and jointly training them. Similarly, E4T [15] introduced an encoder-based domain-tuning method to accelerate the personalization of text-to-image models, offering a faster solution to model fitting for new subjects. There are large numbers of similar methods which are

Within the text-to-image community, one specific area of interest that has garnered significant attention is the generation of human images. Human-centric image generation, owing to its vast applicability and widespread popularity, constitutes a substantial proportion of posts in the community, such as Civitai [1]. In light of these considerations, our work primarily focuses on the preservation of human identity during the image generation process.

|V|
|---|

porate a human identity input in this module, which works with face pictures to guide the synthesis of human images. Intuitively, the stylized human pictures also referred to as style images, specify image content and style, while the identity input provides fine-grained identity guidance. More specifically, we deploy the CLIP vision encoder [40] to process the human images Ih, resulting in eh = CLIPV(Ih). Concurrently, the Arcface model [12] attends to a face image If, leading to ef = Arcface(If). These two derived embeddings are then combined, and a linear layer processes the combined data: ei = Linear(eh||ef), where || denotes the concatenation function. This design effectively extracts and disentangles the identity-related representations from the overall human image content. By doing so, it equips the model with the capability to specifically focus on human identities during the synthesis process. An intrinsic advantage of this separated design is the flexibility it offers. After training, by simply swapping the facial image in the condition module, we can seamlessly modify the identity within a human image, ensuring adaptability to diverse requirements.

| | | | |
|---|---|---|---|

|K|
|---|

Guidance

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

|Q|
|---|

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |

Feature

Attention

- (a) Baseline cross-attentions within the StableDiffusion model.

|Q|
|---|

|V|
|---|

|K|
|---|

| | |
|---|---|

| | |
|---|---|
| | |

| |
|---|
| |

| |
|---|
| |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| |
|---|
| |

| |
|---|
| |

| | |
|---|---|

| | |
|---|---|

| | |
|---|---|

| | |
|---|---|

| |
|---|
| |
| |
| |

ID 1

| | |
|---|---|

| | |
|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

ID 2

Region 1 Region 2

Attention

- (b) Enhanced cross-attentions optimized for multi-identity synthesis.

- Figure 3. Comparison between standard cross-attentions in singleidentity modeling (a) and the advanced cross-attentions tailored for multi-identity integration (b).

Few-shot Learning. It is a common practice in the field of few-shot learning literature to convert optimizationbased tasks into a feed-forward approach, which can significantly reduce computation time. Numerous previous studies in the realm of few-shot learning have adopted this approach to address computer vision tasks, including but not limited to classification [63–66], detection [58], and segmentation [10, 28, 30–32, 61, 62]. In this research, our objective aligns with this approach as we aim to integrate identity information into the generative process through a feed-forward module.

Incorporating Textual Guidance To further allow textual prompts as guidance for conditional image generation, a prior model [41] is employed to translate textual descriptions T into image embeddings eT. To achieve this, the prior model is pre-trained to map the CLIP text embeddings to their corresponding CLIP vision embeddings, resulting in eT = Prior(CLIPT(T)). Following this, a linear layer is deployed to integrate the textual guide with the identity information, formulated as ep = Linear(eT||ef). Given that the prior model’s output shares the embedding space with the CLIP vision encoder’s output, both branches in our framework use shared linear layer parameters for identity fusion. With these steps, the model is equipped with dual guiding mechanisms for image generation: human photographs under various styles and text prompts, both enriched with identity information. The two types of guidance are then merged linearly, using a hyper-parameter α, which provides the final guidance embedding eg: eg = αei + (1 − α)ep. Finally, this guidance embedding is fused into the U-Net with cross-attentions, as is done in StableDiffusion [42].

### 3. Method

In this section, we present the design and functionalities of our novel framework. Our method fundamentally builds on StableDiffusion [42], with several pivotal modifications, especially in the condition modules catering to hybridguidance image generation. We start by elaborating on our hybrid guidance design in the proposed condition module. Following that, we delve into the mechanism for managing multiple identities within images. Lastly, we discuss the training strategy of our models. The overview of our model structure is shown in Fig. 2.

#### 3.2. Handling Multiple Identities

#### 3.1. Hybrid Guidance Strategy

Our design can adeptly fuse multiple identities when given their respective face photos. This merging is done by blending their face embeddings, presented as ef = i=1 βieif, where eif is the facial embedding of the ith identity, and βi is a weight factor. Yet, a challenge arises when we intend to superimpose varying identities onto multiple humans within a single image. Given the standard process in StableDiffusion [42], guidance information, eg, is fused into the intermediate U-Net feature maps, F, using a cross-attention

Disentangling Identities from Style Images Given our research’s main aim is to keep human identities intact during image generation under various styles, it is indispensable to extract salient identity features from human images for conditional image generation. Building on this understanding, our first step is setting up an image-condition guidance module, which aims to take stylized human pictures as the main data for the condition modules. We additionally incor-

layer, represented as

##### Fˆ = Attention(Q,K,V ). (1)

Here, Q originates from flattened intermediate features F, while K and V come from the guidance embedding eg. In situations with several humans in the content images, we aim for each human’s generation to reference unique identity data. If there are N humans and N identities, with the aim of a 1-to-1 identity match, our approach ensures that features in F from the ith human region solely access information from the ith identity. This is denoted as

##### Fˆi = Attention(Qi,Ki,V i), (2)

where Qi is obtained from the specific features of the ith human, and Ki and V i are derived from the guidance vector of the ith identity. This feature selection operation can be easily implemented with the help of an off-the-shelf human instance segmentation model. An illustration of our design is shown in Fig. 3. The strength of our design lies in its precision in managing multiple identities, ensuring that each identity aligns with its corresponding human figure in the content image. For features in non-human regions, they can randomly choose guidance embeddings for guidance, since all guidance embeddings contain information about the stylized human image, with differences only in the identity inputs. Alternatively, the latent embeddings in these regions can be fixed, similar to the approach used in StableDiffusion [42] for inpainting tasks.

#### 3.3. Training

We train our model with a human image reconstruction task, conditioned on the input human image and the identity input. Specifically, raw images with the face region masked serve as the stylized human image, while the cropped face from the same image acts as the identity input. This strategy effectively separates identity information from the overall image content for guided image generation. Our model is optimized by only using the image-conditioned branch in our condition module, as the linear layer in the textconditioned branch shares the parameters with the one in the image-condition branch. This obviates the need for text annotations for human images, which are often hard to acquire on a large scale. We keep the parameters of both the CLIP vision encoder and the Arcface model frozen, focusing our optimization on the newly introduced linear layer and the U-Net model. In line with StableDiffusion [42], our U-Net model, denoted as εθ(), is trained to denoise latent representations produced by a pre-trained VAE encoder E(). This is captured by:

LDM := EE(x),ϵ∼N(0,1),t ∥ϵ − ϵθ(zt,t,eg)∥22 , (3)

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Raw image

[Figure 49]

[Figure 50]

[Figure 51]

Ours

###### w/o

ID input

[Figure 52]

[Figure 53]

[Figure 54]

Ours

- Figure 4. Influence of identity input on image construction. The addition of identity input proves to be effective in preserving the subject’s identity within the generated image.

ID Reference view Result Reference view Result

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

- Figure 5. Identity-preserving novel view synthesis experiment. Our method excels at generating new views of a subject while maintaining its identity.

where x is a sampled human image, t denotes the sampled de-noising step, and eg represents the guidance embedding generated by our condition module. We compute the MSE loss between the sampled noise ϵ and the estimated noise ϵθ(zt,t,eg) for optimization.

### 4. Experiments

#### 4.1. Implementation details.

The vision encoder utilized in the image-conditioned branch of our model combines three CLIP model [40] variants with different backbones. These are: CLIP-ViT-L/14, CLIP-RN101, and CLIP-ViT-B/32. The outputs from these individual models are concatenated to produce the final output of our vision encoder. Our approach primarily utilizes the DDPM configuration [20] as described in StableDiffusion [42] for training. Specifically, we incorporated a total of 1,000 denoising steps. For the inference stage, we use the

Input

Style image

Text condition and style image interpolation

image

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

- Figure 6. Hybrid-guidance experiments. In this experiment, we employ an approach that combines textual prompts and reference images for image synthesis, and the text prompt used here pertains to the cartoon style.

Style Image

Input ID A

ID Interpolation

Input ID B

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

- Figure 7. Identity mixing experiment. We generate facial images that combine multiple identities using a mixing ratio to control the influence of different IDs.

EulerA sampler [2] and set it to operate over 25 timesteps. To align with the training methodology of classifier-free guidance [19], we introduced variability by randomly omitting the conditional embeddings related to both style images and face images. Specifically, the probabilities for dropping these embeddings were set at 0.64 for style images and 0.1 for face images.

The primary dataset used for training was FFHQ [25], which is a face image dataset encompassing 70,000 images. To augment this, we also incorporated a subset of the LAION dataset [46] into our training phase, which aims to ensure the model retains the capability to generate generic, non-human images during the finetuning process. It’s worth noting that when non-human images are sampled for training, the face embedding in the conditional branch is set to zero. During training, we set the learning rate at 1e-6. The model was trained using 8 A100 GPUs, with a batch size of 256, and was trained for 100,000 steps.

#### 4.2. Results

We provide quantitative and qualitative results for comparison and analysis in this section.

Quantitative comparison with baselines. To evaluate our model’s ability to preserve identity during image generation, we conduct a quantitative comparison with baselines. We measure the Arcface feature cosine similarity [12] between the face regions of input reference human images and the generated images, with values in the range of (-1, 1). We consider both 1-shot and multi-shot cases, where only one reference image and 11 reference images are provided, respectively. The baselines we compare against include DreamBooth [43] and Textual Inversion [14], which are the most popular tuning-based methods, which are optimized with 1K and 3K iterations, respectively. We also incorporate our model variant that removes the identity input for comparison to measure the influence of identity information. For the multi-shot experiment, we compute the face similarity score between the generated image and each reference image and report the mean score for comparison.

Identity Baseline Ours Identity Baseline Ours

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

- Figure 8. Comparison of multi-human image synthesis. Our model’s effectiveness is evident when compared to our model variant that removes our proposed multi-human cross-attention mechanisms.

[Figure 133]

PhotoVerse Ours

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

- Figure 9. Comparison with state-of-the-art methods in identity-preserving text-to-image generation. This figure illustrates a comparative analysis of our model against state-of-the-art methods in the task of identity-preserving text-to-image generation. The evaluation is conducted using the same samples employed in FaceNet’s experiments. Notably, our model consistently achieves comparable or superior qualitative results across all the examples.

| |Face Similarity ↑|Time (s) ↓|
|---|---|---|
| |Single-Image Multi-Image<br><br>|Tuning Inference Sum|
|DreamBooth [43] Textual Inversion [14]|0.65 0.58 0.31 0.27<br><br>|405 2.8 407.8<br><br>3425 2.8 3427.8|
|Ours w/o ID input<br><br>|0.47|0 3.6 3.6|

Ours w/ text 0.71 0.61 0 5.3 + 0.04 N 5.3 + 0.04 N Ours w/ image 0.86 0.74 0 3.6 + 0.04 N 3.6 + 0.04 N

Table 1. Comparison of face similarity and generation time for identity-preserving image generation. Our methods, guided by both texts and images, exhibit remarkable advantages compared to baseline approaches in terms of both face similarity and generation time. The variable N represents the number of reference images per identity. Notably, the omission of identity guidance input in our design results in a substantial drop in performance.

We have conducted experiments on 11 different identities, and we report the average performance for comparison, and the results are shown in Table 1. Our model demonstrates superior performance in both 1-shot and multi-shot cases, highlighting the effectiveness of our design in preserving identities. Notably, our advantages stem from the inclusion of our identity-guidance branch, as demonstrated by the performance drop in the baselines that lack this guidance.

Influence of the identity input. To further investigate the influence of the identity input, we conducted an ablation study through qualitative comparisons. We compare our model with a baseline model that removes the identityguidance branch, acting as an image reconstruction model. The results are displayed in Fig. 4. The image reconstruction baseline roughly preserves image content but struggles with fine-grained identity information. In contrast, our model successfully extracts identity information from the identity-guidance branch, resulting in improved results for the face region.

Novel view synthesis. We conduct novel view synthesis experiments to validate our algorithm’s effectiveness in synthesizing images with large pose changes while preserving identity. Results are presented in Fig. 5. The result demonstrates that our algorithm can synthesize high-quality images with preserved identities even when faced with significant pose changes, which showcases the robustness of our design.

Dual guidance experiment. We explore the combined use of text-based and image-based guidance during inference by controlling the strength of both types of guidance using the hyperparameter, α, within the range [0,1]. The results, presented in Fig. 6, illustrate that our model can effectively utilize both types of guidance for image synthesis while preserving identity information. By adjusting α, we could see how the influence of each type of guidance changed.

Identity mixing experiment. Our model’s ability to mix identity information from different humans during image synthesis is showcased in Fig. 7. By controlling the mix ratio within the range [0,1], we assign weights to different identities. As is shown, our model effectively combines

identity information from different people and synthesizes new identities with high fidelity.

Multi-human image generation. One of our model’s unique features is synthesizing multi-human images from multiple identities. We present the results in Fig. 8, by comparing our design to a baseline using vanilla cross-attention mechanisms. As the result shows, our model effectively correlates different human regions with different identities with our proposed enhanced cross-attention mechanisms to differentiate between identities, while the baseline results in confused identities in the human regions.

More qualitative results. Fig.10 showcases a new image synthesis method, which is an extension of the image-to-image generation technique originally found in StableDiffusion[42]. This method has only minor variations from the primary pipeline discussed in the main paper. In this approach, the diffusion process begins with the noisy version of the raw human images’ latent representations, with the rest of the process unchanged. This specific modification ensures that the synthesized images retain a layout similar to the original images. Our results demonstrate that, despite these adjustments, the method successfully maintains the identity of subjects in the synthesized images. Additionally, Fig. 11 provides more qualitative results of our model when applied to a broader range of image styles. These results highlight the model’s adaptability to various artistic styles while still holding true to the core objective of identity preservation. This versatility is crucial for applications that require a balance between stylistic expression and the need to maintain recognizable features of the subject.

### 5. Conclusion

In this paper, we present an innovative approach to text-toimage generation, specifically focusing on preserving identity in the synthesized images. Our method significantly accelerates and enhances the efficiency of the image generation process. Central to our approach is the hybrid guidance strategy, which combines stylized and facial images with textual prompts, guiding the image generation process in a cohesive manner. A standout feature of our method is its

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

- Figure 10. Image-to-image synthesis with our proposed method. Our model preserves the identities of humans and the layout in the raw images.

ability to synthesize multi-human images, thanks to our developed multi-identity cross-attention mechanisms. Our extensive experimental evaluations, both qualitative and quantitative, have shown the advantages of our method. It surpasses baseline models and previous works in several key aspects, most notably in efficiency and the ability to maintain identity integrity in the synthesized images.

Limitation and Social Impacts. Compared to existing works like DreamBooth [43], which synthesize images of diverse subjects such as animals and objects, our model is

specifically tailored for identity-preserving generation, exclusively targeting human images. Our text-to-image generation research has two key societal impacts to consider: 1) Intellectual Property Concerns. The ability of our model to create detailed and stylized images raises potential issues with copyright infringement. 2) Ethical Considerations in Facial Generation. The model’s capability to replicate human faces brings up ethical issues, especially the potential for creating offensive or culturally inappropriate images. It’s crucial to use this technology responsibly and establish

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

[Figure 209]

[Figure 210]

[Figure 211]

###### Figure 11. More qualitative results. Our model obtains a balance between stylistic expression and the need to maintain recognizable features of the subject.

###### guidelines to prevent its misuse in sensitive contexts.

### References

- [1] Stable diffusion models, embeddings, loras and more. 3
- [2] Implementation of EulerAncestralDiscreteScheduler based on k-diffusion. 6
- [3] Implementation of Hypernetwork. 3
- [4] Yuval Alaluf, Elad Richardson, Gal Metzer, and Daniel Cohen-Or. A neural space-time representation for text-toimage personalization. arXiv preprint arXiv:2305.15391,

2023. 3

- [5] Omri Avrahami, Kfir Aberman, Ohad Fried, Daniel CohenOr, and Dani Lischinski. Break-a-scene: Extracting multiple concepts from a single image. arXiv preprint arXiv:2305.16311, 2023. 3
- [6] Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Universal guidance for diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 843–852,

2023. 3

- [7] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. arXiv preprint arXiv:2304.08465, 2023. 3
- [8] Hong Chen, Yipeng Zhang, Xin Wang, Xuguang Duan, Yuwei Zhou, and Wenwu Zhu. Disenbooth: Identitypreserving disentangled tuning for subject-driven text-toimage generation. arXiv preprint arXiv:2305.03374, 2023. 3
- [9] Li Chen, Mengyi Zhao, Yiheng Liu, Mingxu Ding, Yangyang Song, Shizun Wang, Xu Wang, Hao Yang, Jing Liu, Kang Du, et al. Photoverse: Tuning-free image customization with text-to-image diffusion models. arXiv preprint arXiv:2309.05793, 2023. 3
- [10] Xiaoyu Chen, Chi Zhang, Guosheng Lin, and Jing Han. Compositional prototype network with multi-view comparision for few-shot point cloud semantic segmentation. arXiv preprint arXiv:2012.14255, 2020. 4
- [11] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. arXiv preprint arXiv:2307.09481, 2023. 3
- [12] Jiankang Deng, J. Guo, and S. Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. Computer Vision and Pattern Recognition, 2018. 4, 6
- [13] Ziyi Dong, Pengxu Wei, and Liang Lin. Dreamartist: Towards controllable one-shot text-to-image generation via contrastive prompt-tuning. arXiv preprint arXiv:2211.11337, 2022. 3
- [14] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 2, 3, 6, 8
- [15] Rinon Gal, Moab Arar, Yuval Atzmon, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Encoder-based domain tuning for fast personalization of text-to-image models. arXiv preprint arXiv:2302.12228, 2023. 3

- [16] Jing Gu, Yilin Wang, Nanxuan Zhao, Tsu-Jui Fu, Wei Xiong, Qing Liu, Zhifei Zhang, He Zhang, Jianming Zhang, HyunJoon Jung, et al. Photoswap: Personalized subject swapping in images. arXiv preprint arXiv:2305.18286, 2023. 3
- [17] Ligong Han, Song Wen, Qi Chen, Zhixing Zhang, Kunpeng Song, Mengwei Ren, Ruijiang Gao, Yuxiao Chen, Di Liu 0003, Qilong Zhangli, et al. Improving tuning-free real image editing with proximal guidance. CoRR, 2023. 3
- [18] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 3
- [19] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 6
- [20] Jonathan Ho, Ajay Jain, and P. Abbeel. Denoising diffusion probabilistic models. Neural Information Processing Systems, 2020. 5
- [21] Lianghua Huang, Di Chen, Yu Liu, Yujun Shen, Deli Zhao, and Jingren Zhou. Composer: Creative and controllable image synthesis with composable conditions. arXiv preprint arXiv:2302.09778, 2023. 3
- [22] Junha Hyung, Jaeyo Shin, and Jaegul Choo. Magicapture: High-resolution multi-concept portrait customization. arXiv preprint arXiv:2309.06895, 2023. 3
- [23] Hyeonho Jeong, Gihyun Kwon, and Jong Chul Ye. Zero-shot generation of coherent storybook from plain text story using diffusion models. arXiv preprint arXiv:2302.03900, 2023. 3
- [24] Xuhui Jia, Yang Zhao, Kelvin CK Chan, Yandong Li, Han Zhang, Boqing Gong, Tingbo Hou, Huisheng Wang, and Yu-Chuan Su. Taming encoder for zero fine-tuning image customization with text-to-image diffusion models. arXiv preprint arXiv:2304.02642, 2023. 3
- [25] Tero Karras, S. Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. Computer Vision and Pattern Recognition, 2018. 2, 6
- [26] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6007–6017, 2023. 3
- [27] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1931–1941, 2023. 3
- [28] Lvlong Lai, Jian Chen, Chi Zhang, Zehong Zhang, Guosheng Lin, and Qingyao Wu. Tackling background ambiguities in multi-class few-shot point cloud semantic segmentation. Knowledge-Based Systems, 253:109508, 2022. 4
- [29] Lingzhi Li, Jianmin Bao, Hao Yang, Dong Chen, and Fang Wen. Faceshifter: Towards high fidelity and occlusion aware face swapping. arXiv preprint arXiv:1912.13457, 2019. 2
- [30] Weide Liu, Chi Zhang, Guosheng Lin, and Fayao Liu. Crnet: Cross-reference networks for few-shot segmentation. In IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2020. 4

- [31] Weide Liu, Chi Zhang, Henghui Ding, Tzu-Yi Hung, and Guosheng Lin. Few-shot segmentation with optimal transport matching and message flow. IEEE Transactions on Multimedia (TMM), 2021.
- [32] Weide Liu, Chi Zhang, Guosheng Lin, and Fayao Liu. Crcnet: Few-shot segmentation with cross-reference and regionglobal conditional networks. International Journal of Computer Vision (IJCV), 2022. 4
- [33] Yang Liu, Cheng Yu, Lei Shang, Ziheng Wu, Xingjun Wang, Yuze Zhao, Lin Zhu, Chen Cheng, Weitao Chen, Chao Xu, et al. Facechain: A playground for identity-preserving portrait generation. arXiv preprint arXiv:2308.14256, 2023. 3
- [34] Zhiheng Liu, Ruili Feng, Kai Zhu, Yifei Zhang, Kecheng Zheng, Yu Liu, Deli Zhao, Jingren Zhou, and Yang Cao. Cones: Concept neurons in diffusion models for customized generation. arXiv preprint arXiv:2303.05125, 2023. 3
- [35] Jian Ma, Junhao Liang, Chen Chen, and Haonan Lu. Subject-diffusion: Open domain personalized text-to-image generation without test-time fine-tuning. arXiv preprint arXiv:2307.11410, 2023. 3
- [36] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021. 3
- [37] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6038–6047, 2023. 3
- [38] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021. 2
- [39] Ivan Perov, Daiheng Gao, Nikolay Chervoniy, Kunlin Liu, Sugasa Marangonda, Chris Um´e, Mr Dpfks, Carl Shift Facenheim, Luis RP, Jian Jiang, et al. Deepfacelab: Integrated, flexible and extensible face-swapping framework. arXiv preprint arXiv:2005.05535, 2020. 2
- [40] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 4, 5
- [41] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125,

2022. 2, 4

- [42] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 4, 5, 8
- [43] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven

- generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500– 22510, 2023. 2, 3, 6, 8, 9
- [44] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman. Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models. arXiv preprint arXiv:2307.06949, 2023. 3
- [45] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022. 2
- [46] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, P. Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, R. Kaczmarczyk, and J. Jitsev. Laion5b: An open large-scale dataset for training next generation image-text models. Neural Information Processing Systems,

2022. 6

- [47] Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without testtime finetuning. arXiv preprint arXiv:2304.03411, 2023. 3
- [48] Yoad Tewel, Rinon Gal, Gal Chechik, and Yuval Atzmon. Key-locked rank one editing for text-to-image personalization. In ACM SIGGRAPH 2023 Conference Proceedings, pages 1–11, 2023. 3
- [49] Dani Valevski, Matan Kalman, Yossi Matias, and Yaniv Leviathan. Unitune: Text-driven image editing by fine tuning an image generation model on a single image. arXiv preprint arXiv:2210.09477, 2022. 3
- [50] Dani Valevski, Danny Wasserman, Yossi Matias, and Yaniv Leviathan. Face0: Instantaneously conditioning a text-toimage model on a face. arXiv preprint arXiv:2306.06638,

2023. 3

- [51] Yuhan Wang, Xu Chen, Junwei Zhu, Wenqing Chu, Ying Tai, Chengjie Wang, Jilin Li, Yongjian Wu, Feiyue Huang, and Rongrong Ji. Hififace: 3d shape and semantic prior guided high fidelity face swapping. arXiv preprint arXiv:2106.09965, 2021. 2
- [52] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. arXiv preprint arXiv:2302.13848, 2023. 3
- [53] Ziheng Wu, Jiaqi Xu, Xinyi Zou, Kunzhe Huang, Xing Shi, and Jun Huang. Easyphoto: Your smart ai photo generator. arXiv preprint arXiv:2310.04672, 2023. 3
- [54] Zijie Wu, Chaohui Yu, Zhen Zhu, Fan Wang, and Xiang Bai. Singleinsert: Inserting new concepts from a single image into text-to-image models for flexible editing. arXiv preprint arXiv:2310.08094, 2023. 3
- [55] Guangxuan Xiao, Tianwei Yin, William T Freeman, Fr´edo Durand, and Song Han. Fastcomposer: Tuning-free multisubject image generation with localized attention. arXiv preprint arXiv:2305.10431, 2023. 3

- [56] Xingqian Xu, Jiayi Guo, Zhangyang Wang, Gao Huang, Irfan Essa, and Humphrey Shi. Prompt-free diffusion: Taking” text” out of text-to-image diffusion models. arXiv preprint arXiv:2305.16223, 2023. 3
- [57] Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. Paint by example: Exemplar-based image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18381–18391,

2023. 3

- [58] Ze Yang, Chi Zhang, Ruibo Li, and Guosheng Lin. Efficient few-shot object detection via knowledge inheritance. IEEE Transactions on Image Processing (TIP), 2022. 4
- [59] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 3

- [60] Ge Yuan, Xiaodong Cun, Yong Zhang, Maomao Li, Chenyang Qi, Xintao Wang, Ying Shan, and Huicheng Zheng. Inserting anybody in diffusion models via celeb basis. arXiv preprint arXiv:2306.00926, 2023. 3
- [61] Chi Zhang, Guosheng Lin, Fayao Liu, Jiushuang Guo, Qingyao Wu, and Rui Yao. Pyramid graph networks with connection attentions for region-based one-shot semantic segmentation. In IEEE International Conference on Computer Vision (ICCV), 2019. 4
- [62] Chi Zhang, Guosheng Lin, Fayao Liu, Rui Yao, and Chunhua Shen. Canet: Class-agnostic segmentation networks with iterative refinement and attentive few-shot learning. In IEEE Conf. Computer Vision and Pattern Recognition (CVPR),

2019. 4

- [63] Chi Zhang, Yujun Cai, Guosheng Lin, and Chunhua Shen. Deepemd: Few-shot image classification with differentiable earth mover’s distance and structured classifiers. In IEEE Conf. Computer Vision and Pattern Recognition (CVPR ORAL), 2020. 4
- [64] Chi Zhang, Henghui Ding, Guosheng Lin, Ruibo Li, Changhu Wang, and Chunhua Shen. Meta navigator: Search for a good adaptation policy for few-shot learning. In IEEE International Conference on Computer Vision (ICCV), 2021.
- [65] Chi Zhang, Nan Song, Guosheng Lin, Yun Zheng, Pan Pan, and Yinghui Xu. Few-shot incremental learning with continually evolved classifiers. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2021.
- [66] Chi Zhang, Yujun Cai, Guosheng Lin, and Chunhua Shen. Deepemd: Differentiable earth mover’s distance for few-shot learning. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2022. 4
- [67] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 3
- [68] Yufan Zhou, Ruiyi Zhang, Tong Sun, and Jinhui Xu. Enhancing detail preservation for customized text-to-image generation: A regularization-free approach. arXiv preprint arXiv:2305.13579, 2023. 3
- [69] Yuhao Zhu, Qi Li, Jian Wang, Cheng-Zhong Xu, and Zhenan Sun. One shot face swapping on megapixels. In Proceedings

of the IEEE/CVF conference on computer vision and pattern recognition, pages 4834–4844, 2021. 2

