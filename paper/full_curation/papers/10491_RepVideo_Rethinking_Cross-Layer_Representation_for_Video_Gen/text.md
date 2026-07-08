## RepVideo: Rethinking Cross-Layer Representation for Video Generation

Chenyang Si†, Weichen Fan†, Zhengyao Lv, Ziqi Huang, Yu Qiao, and Ziwei Liu

### arXiv:2501.08994v1[cs.CV]15Jan2025

Abstract—Video generation has achieved remarkable progress with the introduction of diffusion models, which have significantly improved the quality of generated videos. However, recent research has primarily focused on scaling up model training, while offering limited insights into the direct impact of representations on the video generation process. In this paper, we initially investigate the characteristics of features in intermediate layers, finding substantial variations in attention maps across different layers. These variations lead to unstable semantic representations and contribute to cumulative differences between features, which ultimately reduce the similarity between adjacent frames and negatively affect temporal coherence. To address this, we propose RepVideo, an enhanced representation framework for text-to-video diffusion models. By accumulating features from neighboring layers to form enriched representations, this approach captures more stable semantic information. These enhanced representations are then used as inputs to the attention mechanism, thereby improving semantic expressiveness while ensuring feature consistency across adjacent frames. Extensive experiments demonstrate that our RepVideo not only significantly enhances the ability to generate accurate spatial appearances, such as capturing complex spatial relationships between multiple objects, but also improves temporal consistency in video generation. Project page: https://vchitect.github.io/RepVid-Webpage.

Index Terms—Video Generation, Diffusion Models, Video Diffusion Models, Transformer.

I. INTRODUCTION

# R

ECENTLY, the research community has witnessed remarkable advancements in generative models, particu-

larly in the domain of text-to-image generation (T2I) [1]– [5]. These breakthroughs have garnered significant attention, showcasing the potential of generative models in producing diverse and realistic images from textual descriptions. Notably, diffusion models [6]–[11] have emerged as a significant development, enabling the generation of high-fidelity images through the iterative diffusion of information within the model.

Chenyang Si is with the S-Lab, Nanyang Technological University, Singapore, 639798. E-mail: chenyang.si@ntu.edu.sg.

Weichen Fan is with the S-Lab, Nanyang Technological University, Singapore, 639798. E-mail: fanweichen2383@gmail.com.

Zhengyao Lv is with Shanghai Artificial Intelligence Laboratory, China. E-mail: cszy98@gmail.com.

Ziqi Huang is with the S-Lab, Nanyang Technological University, Singapore, 639798. E-mail: ziqi002@ntu.edu.sg.

Yu Qiao is with Shanghai Artificial Intelligence Laboratory, China. E-mail: qiaoyu@pjlab.org.cn.

Ziwei Liu is with the S-Lab, Nanyang Technological University, Singapore, 639798. E-mail: ziwei.liu@ntu.edu.sg.

† Equal contribution.

Inspired by these achievements, researchers are now exploring the application of diffusion models in the realm of text-tovideo (T2V) generation [12]–[26], aiming to generate visually compelling and contextually coherent videos with textual descriptions.

In the realm of video generation, producing coherent and high-quality videos is inherently complex and demanding. Unlike static images, videos require the generation of sequential frames that exhibit spatial fidelity and temporal continuity. However, the availability of high-quality video datasets is significantly limited compared to image datasets, making it challenging to obtain diverse and representative training data. Additionally, generating a single frame in a video involves not only capturing visual details but also preserving smooth transitions and consistent motion across frames. Consequently, video generation entails substantial computational requirements due to the increased number of frames and the need for effective modeling of temporal dependencies.

To overcome these challenges and advance the field of video generation, early video diffusion methods sought to extend state-of-the-art T2I models to text-to-video (T2V) generations, such as Make-A-Video [12], Imagen Video [13], Latent-Shift [14], MagicVideo [15], Video LDM [16], and Lavie [17]. These methods primarily adapt image diffusion models by incorporating temporal modules such as temporal attention [27], spatio-temporal attention [28], directed temporal attention [15], temporal shift [14], and 3D CNNs [16], and fine-tune them on RGB video data.

With the recent success of Diffusion Transformers (DiTs) [9], [10], [29] in text-to-image generation, which leverage transformers as the backbone of diffusion models, exploration has extended to text-to-video generation. A significant advancement is achieved with the OpenAI Sora [30], which significantly scales a spatial-temporal transformer model, marking a milestone in video generation technology. Subsequently, transformer-based approaches, such as CogVideoX [31], Vidu [32] and MovieGen [33], have further advanced video generation. These methods typically use a 3D VAE to compress input videos along spatial and temporal dimensions, yielding a video latent representation. This latent is then flattened into the token sequence. A transformer network captures spatial and temporal information within this sequence, effectively generating complex videos. Despite these significant successes, most studies have primarily focused on scaling up model training, i.e., model size, dataset scale, and compute budget, while offering limited insights into how representations directly impact the video generation process.

In this paper, we conduct an in-depth analysis of transformer

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

A 3D animation of a moon-like object approaching Earth.

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

Push upward at a low angle, slowly look up, a tiger with intense, fiery eyes, surrounded by flames.

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

The video presents a serene landscape during sunrise.

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

The video starts with a dark purple background, then glowing particles gradually form a heart.

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

A close-up of a gray-haired man in a beret, deep in thought at a Paris cafe.

- Fig. 1: The examples generated by RepVideo. RepVideo effectively generates diverse videos with enhanced temporal coherence and fine-grained spatial details.

representations within video diffusion models, focusing on their impact on spatial appearance and temporal consistency in video generation. Achieving both fine spatial detail and smooth temporal transitions is essential for producing highquality videos. To this end, we first investigate the spatial expressiveness of attention maps across transformer layers. Our findings reveal that the attention maps in each transformer layer exhibit substantial differences, with each layer focusing on distinct token information. While this enables the model to capture diverse spatial features, the lack of coordination across layers leads to fragmented feature representations, weakening the model’s ability to form coherent spatial semantics within individual frames.

Building on this, we further extend our investigation to examine how these attention mechanisms affect temporal consistency. We analyze the evolving similarity between adjacent frame features across transformer layers, as this similarity serves as a key indicator of temporal coherence. Our findings show that as layer depth increases, the similarity between adjacent frame features gradually decreases. This phenomenon is

attributed to the progressive feature differentiation introduced by attention mechanisms, which accumulates across layers. As a result, the diminished frame-to-frame similarity disrupts temporal coherence, leading to artifacts such as motion discontinuities or blur in the generated videos, ultimately impairing their overall quality.

Based on the above insights, we propose a novel architecture, termed RepVideo, to enhance the video representations for text-to-video diffusion models. The attention module of each transformer layer captures distinct local information, leading to a diverse set of intermediate feature representations. Hence, we explore leveraging these rich features to enhance the semantic consistency and quality of generated videos. Specifically, We introduce the feature cache module that aggregates the features from multiple adjacent transformer layers. We apply a mean aggregation strategy across all collected features to achieve a stable semantic representation. Finally, this aggregated representation is combined with the original transformer input through a gating mechanism, producing an enhanced feature input for each transformer layer. With the

guidance of the stable semantic representation, RepVideo can maintain consistent feature details across frames, alleviating inconsistencies between adjacent frames. This leads to enhanced temporal coherence, while also improving the model’s capacity for generating details, thereby resulting in higher quality and more visually consistent video generations, as shown in Fig. 1.

We conducted extensive experiments to evaluate the effectiveness of RepVideo. The experimental results demonstrate that RepVideo significantly enhances both temporal coherence and spatial detail generation, achieving competitive performance in qualitative and quantitative metrics.

Our contributions are summarized as follows:

- • We investigate the transformer representations in video diffusion models, revealing that substantial variations in attention maps across layers lead to fragmented spatial semantics and reduced temporal consistency, which negatively impact video quality.
- • We propose RepVideo, a framework that leverages a feature cache module and a gating mechanism to aggregate and stabilize intermediate representations, enhancing both spatial detail and temporal coherence.
- • Extensive experiments demonstrate that RepVideo achieves competitive performance in both temporal consistency and spatial quality, validating its effectiveness for video generation.

II. RELATED WORK A. Generative Models

Generative models have been extensively studied and have achieved significant advancements. Early research focused on the application of Generative Adversarial Networks (GANs), which excelled at generating visually realistic and semantically coherent images. For example, AttnGAN [34] introduced an attention mechanism to focus on specific image regions, enhancing fine-grained details. DM-GAN [35] incorporated a dynamic memory module to refine image synthesis quality, while DF-GAN [36] proposed a simple yet effective framework for one-stage high-resolution image generation. On the other hand, a different line of research for text-to-image generation has emerged with the framework of Vector Quantized Variational AutoEncoders (VQ-VAE). These methods tokenize images into discrete visual representations, forming the foundation for models such as DALL-E [2], CogView [3], M6 [4], and Parti [5]. These models leverage VQ-VAE [37] or VQGAN [1] to encode images and employ large-scale language models to map textual descriptions to visual tokens, enabling the generation of semantically aligned, high-quality images. More recently, Denoising Diffusion Probabilistic Models (DDPM) [6] have garnered significant attention for their ability to produce high-fidelity images with robust text-to-image alignment [7]– [11], [38]. Notable examples include GLIDE [39], which adopts a classifier-free guidance strategy for improved generation, and Imagen [40], which introduces an efficient UNet architecture for diffusion models. table Diffusion [8], [11] further optimizes computational efficiency by applying diffusion in the latent space while maintaining high image quality.

Building on these advancements, transformer-based diffusion models [9], [10], [29] have emerged as a powerful alternative to the UNet architecture, achieving superior performance in generating high-resolution and semantically consistent images.

B. Video Generative models

While significant advancements have been made in image generation, video generation from textual descriptions presents additional challenges. Unlike static images, videos require not only capturing spatial details but also modeling temporal dynamics and maintaining coherent transitions across frames. Similar to the development of image generation, early text-to-video (T2V) methods primarily employed Generative Adversarial Networks (GANs), such as those proposed in [41]–[43]. These early approaches focused on constrained domains, such as sequences of moving digits or specific human actions. However, generating videos in more complex and diverse domains introduces substantial challenges, as it requires modeling intricate spatial-temporal relationships and handling greater variability in visual content. As an alternative approach, auto-regressive methods have also been exploited for video generation. GODIVA [44] introduces a threedimensional sparse attention mechanism into VQ-VAE for an open-domain text-to-video generation. VideoGPT [45] adapts VQ-VAE and Transformer models to generate realistic samples from complex natural video datasets. TATS as proposed by Ge et al. [46], extends the 2D-VQGAN to a 3D-VQGAN for modeling videos and employs a hierarchical approach with a frame interpolation transformer to generate long, coherent, and high-quality videos. The paper by Wu et al. [47] introduces NUWA, a multimodal model that leverages auto-regressive architectures to facilitate the generation of videos and images from textual inputs. CogVideo [48], an autoregressive transformer model, leverages the pre-trained text-to-image generative model to the text-to-video generation. Phenaki [49] is capable of generating variable-length videos conditioned on a sequence of open domain text prompts. Diffusion models, a recent class of models, offer a promising framework for generation tasks, including T2V generation which has been the focus of recent research efforts [13]–[15], [17], [28], [50], [51]. Ho et al. [50] propose a video diffusion model by using factorized space-time attention blocks for joint image-video training. Imagen Video [13] extends the T2I diffusion model of Imagen [40] to generate high-fidelity videos through a cascade of video diffusion models. MagicVideo [15] introduces the directional attention and the adaptor module into Stable Diffusion [8] for generating videos. Make-A-Video [12] extends the pre-trained T2I diffusion model through a spatiotemporally factorized diffusion model. Tune-A-Video [28] finetunes the pre-trained text-to-image diffusion model on a single video for video synthesis. Video LDM [16] leverages pre-trained image modes and turns them into video generators by inserting temporal layers that learn to align images in a temporally consistent manner.Latent-Shift [14] propose a temporal shift module to leverage a T2I model as-is for T2V generation without adding any new parameters. LVDM [27] uses a hierarchical latent video diffusion model to generate longer videos beyond

the temporal training length. VideoFusion [51] presents a decomposed diffusion process that separates per-frame noise into shared base noise and residual noise, improving the generation of high-quality videos. Most recently, transformer-based diffusion models have marked a significant milestone in video generation. Models like Sora [30] have demonstrated remarkable performance, showcasing the potential of transformer diffusion architectures. Additionally, models such as CogVideoX [31], and MovieGen [33], have achieved impressive results, pushing the boundaries of T2V generation with enhanced spatialtemporal coherence and scalability. Despite showing some promising results, these methods provide limited insights into how intermediate representations directly influence the video generation process. In this paper, we conduct a comprehensive analysis of transformer representations within video diffusion models, focusing on their role in video quality and temporal coherence.

III. METHODOLOGY A. Preliminaries

1) Denoising Diffusion Probabilistic Models: Denoising Diffusion Probabilistic Models (DDPM), as introduced in [6], encompass a diffusion process and a denoising process for data modeling. The diffusion process in DDPM is represented by a sequence of T steps. At each step t, Gaussian noise is progressively injected into the data distribution x0 ∼ q(x0) through a Markov chain according to a variance schedule β1,...,βT:

q(x1:T|x0) =

T

q(xt|xt−1) (1)

t=1

q(xt|xt−1) = N(xt; 1 − βtxt−1,βtI) (2)

where the variances βt are kept constant throughout the diffusion process, as suggested by [6]. The diffusion process consists of T steps, where random noise is progressively added to each sample xt−1 to generate the next sample xt. The magnitude of the added noise at each step is determined by the predefined variance schedule βt.

The denoising process reverses the above diffusion process to the underlying clean data xt−1 given the noisy input xt:

T

pθ(xt−1|xt) (3)

pθ(x0:T) = p(xT)

t=1

pθ(xt−1|xt) = N(xt−1;µθ(xt,t),Σθ(xt,t)) (4)

The µθ and Σθ are estimated using a denoising model ϵθ. This denoising model is commonly implemented as a timeconditional UNet architecture, which incorporates residual blocks and self-attention layers. The denoising model ϵθ is trained to remove noise and enhance the fidelity of the generated samples. During training, the objective is to maximize a simplified version of the variational bound:

L = Ex,ϵ∼N(0,1),t[∥ϵ − ϵθ(xt,t)∥22] (5) where ϵ denotes the unscaled noise.

2) Latent Diffusion Models: Latent Diffusion Models [8] (LDMs) are a type of diffusion model that models the distribution of the latent space of images. The training process of latent image diffusion models consists of two stages.

In the first stage, an autoencoder is employed to learn a compressed and meaningful representation of input images. The autoencoder is composed of an encoder E and a decoder D. The encoder takes an input image x ∈ RH×W×3 and maps it to a lower-dimensional latent space, producing a latent representation z = E(x) ∈ Rh×w×c. Here, H and W represent the height and width of the input image, while h, w, and c represent the dimensions of the feature maps in the latent space. The decoder then reconstructs the original image from this latent representation, generating a reconstructed image x˜ = D(z) ∈ RH×W×3. In the LDMs, two commonly used architectures for the autoencoder component are VQGAN [1] and VAE [52]. In the latent space, the diffusion model of the second stage can be trained with the objective in Eq. 5, where the latent representation z replaces the original image x, i.e., Ez,ϵ∼N(0,1),t[∥ϵ−ϵθ(zt,t)∥22]. Compared to the highdimensional pixel space, LDMs offer advantages in terms of parameter count and memory consumption, while maintaining similar performance. This property makes LDMs a more efficient and practical choice for generative tasks.

B. Rethinking the Representation of Video Diffusion Models

Recently, diffusion transformers have demonstrated significant success in text-to-video generation, such as CogVideoX [31], and MovieGen [33]. These methods typically consist of three core components: a 3D VAE, the text encoder, and a transformer network, as shown in Fig 2. The 3D VAE is used to compress the video data along spatial and temporal dimensions, yielding a compact latent representation that allows efficient processing of higher video resolutions and larger numbers of frames, while significantly reducing GPU memory usage. The text encoder processes the input text prompt, converting it into a set of embeddings that capture the semantic meaning and guide the entire video generation process. The video latent representation is then flattened into a sequence of tokens, which, along with the text embedding tokens, are fed into the transformer network. By leveraging the transformer’s powerful attention mechanism, it can learn complex spatial and temporal relationships within the video sequence, ensuring that the generated frames are coherent, consistent, and aligned with the semantic information provided by the text prompt. By integrating these components, diffusion transformer-based models have shown remarkable improvements in generating high-resolution, long-duration videos that are both temporally consistent and semantically aligned with the input prompt. Despite these advancements, most studies have primarily focused on scaling up model training, such as increasing model size, training dataset scale, and computational resources, while providing limited insights into how intermediate representations directly influence the video generation process. This emphasis on scaling overlooks the potential benefits of understanding and optimizing the internal feature representations, which could

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

###### DiT DiT DiT

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

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

Video Encoder

TransformerBlock

TransformerBlock

TransformerBlock

Video Decoder

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

……

……

……

……

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

#### …

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

An adorable red panda playfully perched on a tree branch…

|[Figure 146]<br><br>[Figure 147]<br><br>Video Token Text Token|
|---|

Text Encoder

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

- Fig. 2: The architecture of recent transformer-based video diffusion models. These methods typically consist of three core components: a 3D VAE, the text encoder, and a transformer network.

Frame 1 Frame 2 Frame 3

- Fig. 3: The visualization of the attention distribution of each frame’s token across the entire token sequence. The results highlight significant variations in attention distributions across layers, with deeper layers focusing more on tokens from the same frame and exhibiting weaker attention to tokens from other frames.

lead to more efficient and coherent video generation without merely relying on increased model capacity.

the same frame, with relatively weaker attention to tokens from other frames.

This observation suggests that analyzing the attention maps of each frame’s token alone can effectively represent the overall attention characteristics of the sequence. Thus, we can focus on the self-attention maps of individual frame tokens to capture the key global attention features, simplifying the analysis while still providing meaningful insights into how attention is distributed across different spatial and temporal regions in the video. As shown in Fig 4, we visualize the attention maps for a single frame token across different layers. The visualization reveals that the attention maps from different layers exhibit substantial differences, with each layer attending to distinct regions and capturing different feature information. As the network depth increases, these attention mechanisms, also lead to progressive differentiation of features. While this enables the model to capture diverse spatial features, the lack of coordination across layers leads to fragmented feature representations, weakening the model’s ability to form coherent spatial semantics within individual frames.

To gain a comprehensive understanding of the role that intermediate representations play in video generation, we perform an in-depth analysis of transformer representations within video diffusion models, particularly focusing on their impact on spatial expressiveness and temporal consistency. We begin by analyzing the spatial expressiveness of attention maps across transformer layers. Recent models, as illustrated in Fig 2, integrate both video latent tokens and text embedding tokens into a unified token sequence, which is then processed by the transformer using a full attention mechanism to capture relationships among these tokens. To understand which regions in the attention maps contribute to semantic meaning, we visualize the attention distribution of each frame’s token across the entire token sequence, as shown in Fig 3. The results reveal significant variations in attention distributions across layers, with each layer focusing on distinct regions and learning different aspects of the features. Additionally, we find that as layer depth increases, the attention corresponding to each frame’s token becomes more concentrated on the tokens from

Building on this, we extend our investigation to examine

###### Frame Layer1 Layer2 Layer3 Layer4 Layer5 Layer6

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

- Fig. 4: The visualization of attention maps across transformer layers. Each layer attends to distinct regions, capturing diverse spatial features. However, the lack of coordination across layers results in fragmented feature representations, weakening the model’s ability to establish coherent spatial semantics within individual frames.

0.5

0.6

0.7

0.8

0.9

1

Similarity

Step 20

Layer 10 Layer 15 Layer 20 Layer 25 Layer 30

0.5

0.6

0.7

0.8

0.9

1

Similarity

Step50

Layer 10 Layer 15 Layer 20 Layer 25 Layer 30

0.5

0.6

0.7

0.8

0.9

1

Similarity

Step 5

Layer 10 Layer 15 Layer 20 Layer 25 Layer 30

0.5

0.6

0.7

0.8

0.9

1

Similarity

Step 35

Layer 10 Layer 15 Layer 20 Layer 25 Layer 30

- Fig. 5: The average similarity between adjacent frame features across diffusion layers and denoising steps. The similarity decreases as layer depth increases for a given denoising step, indicating greater differentiation in deeper layers. Additionally, similarity between adjacent frames declines as the denoising process progresses.

how these attention mechanisms affect temporal consistency. We analyze the evolving similarity between adjacent frame features across transformer layers, as this similarity serves as a critical indicator of temporal coherence. Fig 5 visualizes the average similarity between adjacent frame features across different diffusion layers at various denoising steps. The analysis reveals two key observations. First, for a given denoising step, the similarity between adjacent frame features decreases as layer depth increases. This suggests that deeper layers introduce progressively diverse features, leading to greater differentiation between frame features. Second, when comparing across denoising steps, the similarity between adjacent frames decreases as the denoising process progresses. For example, the average similarity is higher at earlier steps (e.g., Step 5) but gradually decreases at later steps (e.g., Step 5). This trend indicates that, while the denoising process enriches the video features with more diverse semantic content, it also increases variability between adjacent frames, reducing temporal coherence and potentially introducing motion artifacts in the generated videos.

C. Enhanced Representation for Video Diffusion Model

Capitalizing on the above discoveries, we introduce RepVideo, a simple and effective framework that leverages

enriched intermediate representations to enhance video generation in text-to-video diffusion models. Our approach draws inspiration from recent advancements in diffusion models that incorporate multiple text encoders, such as FLUX [53] and MovieGen [33]. These methods enhance the model’s ability to interpret text prompts by employing multiple encoders to capture different layers of information, such as semanticlevel and character-level understanding, thereby improving the alignment between generated content and textual descriptions. RepVideo builds upon this idea, aiming to create richer video representations to ultimately improving temporal consistency and semantic alignment in the generated videos.

To achieve this, we explore leveraging the rich features inherent in diffusion transformers to enhance the semantic consistency and quality of the generated videos. This approach eliminates the need to introduce additional networks, as is done with text encoders, thereby maintaining the model’s simplicity and computational efficiency. Firstly, we introduce a Feature Cache Module within the transformer, as illustrated in Fig 8. This module allows each transformer layer to store its output token sequences in the cache, enabling the Feature Cache Module to aggregate features from multiple adjacent transformer layers. More specifically, we store the output token

Frame Original Feature Aggregation Feature

[Figure 204]

[Figure 205]

[Figure 206]

- Fig. 6: The comparison of the original feature maps from a standard transformer layer with those obtained after aggregation in the Feature Cache Module. The aggregated features demonstrate more comprehensive semantic information and clearer structural details.

- Fig. 7: The comparison of adjacent frame similarity between original and aggregated features. The aggregated features from the Feature Cache Module exhibit higher similarity between adjacent frames compared to the original transformer layers, indicating improved temporal coherence.

sequence from the l-th transformer layer as follows:

[Figure 207]

Forigl → Mcache, (6)

Transformer Block

where Forigl represents the output token sequence from the lth layer. Thus, the Feature Cache Module contains the output from multiple layers, aggregating features across different transformer layers. Then, we compute the mean of the features stored in the Feature Cache Module as follows:

[Figure 208]

Gate

[Figure 209]

Transformer Block

[Figure 210]

[Figure 211]

Feature Cache

Gate

m

1 m

[Figure 212]

Fcachei , (7)

Transformer Block

Fmean =

i=1

[Figure 213]

Gate

where m represents the number of token sequences in the feature cache module. Fcachei is the i-th token sequences stored in the cache, i.e., Fcachei ∈ Mcache.

[Figure 214]

Transformer Block

We find that by merging these features, a more semantically enriched representation can be achieved. In the Fig 6, we present a comparison between the original feature maps produced by a standard transformer layer and those obtained after feature aggregation in the Feature Cache Module. It can be observed that the aggregated features capture more comprehensive semantic information and exhibit clearer structural information. Additionally, Fig. 7 illustrates the similarity between adjacent frame features for both sets of representations. The analysis reveals that the aggregated features from the Feature Cache Module exhibit higher similarity between adjacent frames compared to those from the original transformer layers. This suggests that the integration of multi-layer features not

###### Fig. 8: The architecture of the enhanced cross-layer representation framework.

only enriches semantic information but also maintains stronger temporal consistency throughout the frames, which is evident in the reduced variability and improved alignment between adjacent frames.

To use this enriched representation to enhance the original transformer features, we combine the aggregated features with

the original input through a gating mechanism:

###### Fenhl = gl · Forigl + (1 − gl)Fmeanl , (8)

where Fmeanl represents the aggregated features produced from the feature cache module at l-layer. Note that the feature cache module stores a different number of token sequences for each layer, with the number increasing as the layer depth increases. The gating factor gl is a learnable parameter within the l-th layer. This allows the model to dynamically adjust the influence of the enriched representations, effectively balancing the semantic enhancement provided by the cache while preserving necessary layer-specific details from the original feature map.

By using this enriched input, RepVideo enhances the ability of the diffusion model to maintain alignment between the textual input and the generated video, leading to smoother transitions, greater semantic coherence, and an overall improvement in video quality.

D. Training

To evaluate the effectiveness of the enhanced representation introduced in RepVideo, we implemented our model based on CogVideoX-2B [31]. This baseline was selected due to its robust architecture and proven performance in text-to-video generation tasks, providing a solid foundation for comparison. The training process was carefully designed to ensure fairness and consistency between the baseline and our proposed method.

Data Preparation. Since CogVideoX-2B [31] is pretrained on a large-scale corpus of videos, we constructed a curated internal dataset tailored for fine-tuning the model. The dataset, sourced from high-quality platforms underwent a rigorous preprocessing pipeline to ensure diversity and quality. First, long videos were segmented into shorter, manageable clips to emphasize focused events or actions. Related clips were then linked together to form coherent event sequences, preserving narrative consistency across frames. To refine the dataset further, static video filtering was applied to exclude clips lacking significant motion, ensuring an emphasis on dynamic content.

Additionally, aesthetic scoring was conducted to evaluate visual quality based on predefined criteria, prioritizing highquality and semantically rich video inputs. Dynamic estimation was employed to analyze movement patterns and overall clip dynamics, enhancing the dataset’s relevance for motioncentric video generation. A watermark classifier was applied to detect and annotate videos containing visible watermarks, maintaining the integrity and usability of the training data. This comprehensive data preparation process resulted in a highquality dataset of 1 million annotated video clips with detailed captions, covering a diverse range of categories.

Training Setup. Both the baseline CogVideoX-2B [31] and the enhanced RepVideo model were fine-tuned under identical training conditions to ensure fair comparisons. The models were trained for 50,000 steps with a batch size of 32. The AdamW optimizer was employed, using a learning rate of

1 × 10−5, which balanced efficient convergence with model stability. Training was conducted on 32 NVIDIA H100 GPUs, leveraging the computational resources required for largescale video generation tasks. A separate validation set, curated from the prepared dataset, was utilized to periodically assess model performance during training and mitigate the risk of overfitting. These measures ensured that the training process was rigorous and reproducible.

Enhanced Representation Implementation. The core innovation of RepVideo lies in its ability to aggregate intermediate transformer outputs to create stable and semantically enriched representations, improving both spatial fidelity and temporal consistency. This was achieved through the introduction of a Feature Cache Module, which aggregates features across every m transformer layers. By aggregating features from neighboring layers, the module effectively captures semantic details while mitigating inconsistencies across frames. Empirical evaluations demonstrated that setting m = 6 provided the optimal balance between computational efficiency and performance improvement. Unless otherwise specified, m = 6 is used throughout the experiments.

To integrate these enriched representations into the transformer layers, a gating mechanism was introduced. The gating mechanism dynamically combines the aggregated features with the original transformer outputs, using a learnable parameter to control their relative influence.

The proposed aggregation and gating mechanisms are lightweight, introducing minimal additional parameters and computational overhead. During training, these mechanisms demonstrated significant improvements in spatial consistency, temporal coherence, and semantic alignment, resulting in smoother and more visually compelling video outputs.

IV. EXPERIMENTS

To thoroughly evaluate the effectiveness of our proposed framework, we conducted a comprehensive set of experiments covering both quantitative and qualitative analyses. These experiments were designed to assess the performance of RepVideo across a variety of metrics, including spatial fidelity, temporal consistency, and semantic alignment with textual prompts. In particular, we benchmark our model against stateof-the-art methods in automated metrics and human preference evaluations, providing a comprehensive view of its advantages.

The experiments are organized into three parts: 1). Automated Evaluation: Quantitative metrics derived from VBench [54] are used to objectively compare RepVideo’s performance with existing models. 2). Human Evaluation: Human raters assess the generated videos based on alignment with prompts, temporal smoothness, and frame-wise quality, offering a complementary perspective. 3). Ablation Study: A detailed investigation into the contributions of the design of RepVideo, analyzing how spatial and temporal consistency are improved. The following sections delve into these evaluations, highlighting both the strengths and potential areas of improvement for our approach.

- TABLE I: Comparison with previous methods on VBench [54] .

Method Total Score Motion Smoothness Object Class Multiple Objects Spatial Relationship

LaVie [17] 77.08% 96.38% 91.82% 33.32% 34.09% VideoCrafter-2.0 [55] 80.44% 97.73% 92.55% 40.66% 35.86% OpenSoraPlan-v1.1 78.00% 98.28% 76.30% 40.35% 53.11% OpenSora-1.2 [56] 79.76% 98.50% 82.22% 51.83% 68.56% Gen-3 [57] 82.32% 99.23% 87.81% 53.64% 65.09% Gen-2 [58] 80.58% 99.58% 90.92% 55.47% 66.91% Pika-1.0 [59] 80.69% 99.50% 88.72% 43.08% 61.03% CogVideoX-2B [31] 80.91% 97.73% 83.37% 62.63% 69.90% CogVideoX-5B [31] 81.61% 96.92% 85.23% 62.11% 66.35% Vchitect-2.0 [60] 81.57% 97.76% 87.81% 69.35% 54.64%

RepVideo (Ours) 81.94% 98.13% 87.83% 71.18% 74.74%

- TABLE II: User study between CogVideoX-2B [31] and RepVideo.

Method Overall Score Spatial Appearance Temporal Consistency Video-text alignment

CogVideoX-2B [31] 25.54% 25.00% 25.93% 23.38% RepVideo 75.46% 75.00% 74.07% 76.62%

A. Automated Evaluation

Quantitative Evaluation. To evaluate the performance of our model, we adopt all metrics provided in VBench [54], [61]. As shown in Table I, we report the Total Score alongside several representative metrics. Specifically, Motion Smoothness evaluates the temporal stability of generated videos, while Object Class and Multiple Objects measure the ability to generate diverse and well-defined visual elements. Additionally, Spatial Relationship assesses the coherence in object positioning and interactions.

Compared to the baseline model CogVideoX-2B, our model, RepVideo-2B, achieves superior results with a higher Total Score. Notably, our model improves by 0.4% in Motion Smoothness and 4.46% in Object Class, highlighting its ability to maintain temporal consistency and generate fine-grained object details. Moreover, RepVideo achieves significant improvements in Spatial Relationship (+4.84%) and Multiple Objects (+8.55%), demonstrating enhanced spatial coherence and strong ability to handle complex object interactions.

Qualitative Evaluation. Figure 9 provides a qualitative comparison between our method and the baseline, CogVideoX2B [31], demonstrating the significant improvements achieved by our model. The results generated by our method are presented in the second column, while the baseline’s outputs are displayed in the first column. It is obvious that our approach produces more visually consistent and semantically accurate videos, capturing both the spatial and temporal relationships described in the provided prompts. For example, with the prompt “A litter of golden retriever puppies playing in the snow. Their heads pop out of the snow, covered in,” our model maintains temporal coherence, ensuring consistent appearance and motion of the puppies across frames. In contrast, the baseline struggles with stability, resulting in artifacts and inconsistencies. In the prompt “The video features a serene beach scene during sunset,” our method effectively captures the smooth motion of the sunset and spatial accuracy, while the baseline fails to understand the sunset. Similarly, for “A single

tree with a dense canopy of green leaves, standing prominently in the center of a complex circuit board,” our method preserves both the intricate details and spatial alignment across frames. The baseline, however, exhibits noticeable jitter and spatial inconsistencies. Finally, with “A single parrot with a vibrant orange head, a yellow body, and green wings and tail,” our model ensures vivid colors and smooth transitions, maintaining temporal stability. The baseline fails to generate coherent movements, leading to visual distortions. These qualitative results underscore the strength of our proposed RepVideo framework in generating high-quality videos with enhanced temporal coherence and spatial fidelity.

B. Human Evaluation

In addition to the automated evaluation, we conducted a comprehensive human evaluation to assess the performance of our model in comparison to state-of-the-art methods. Human evaluators were presented with pairs of videos generated by different models, each conditioned on the same text prompt. Evaluators were tasked with selecting their preferred video based on three key criteria, evaluated independently: videotext alignment, temporal consistency, and spatial appearance. These criteria were derived from the dimension design in VBench [61], ensuring a standardized and rigorous evaluation process.

The evaluation included two models: our proposed RepVideo and CogVideoX-2B [31]. For each criterion, evaluators conducted 50 pairwise comparisons between videos generated by our model and those produced by the competing method.

As summarized in Tab. II, our model achieved an average win ratio exceeding 50% across all three metrics, demonstrating its superiority in generating videos with higher semantic alignment, smoother temporal transitions, and enhanced visual quality. These results underscore the effectiveness of the enriched representations introduced in our framework,

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

The video features a single tree with a dense canopy of green leaves, standing prominently in the center of a complex circuit board. The circuit board is composed of numerous black lines and components, with orange lights scattered throughout, creating a contrasting yet harmonious visual effect.

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

The video captures a serene beach scene during sunset. The sky is painted with hues of orange and purple, and the sun is partially visible, casting a warm glow on the water. The waves are the main focus, with one large wave in the foreground and smaller waves in the background. The wave in the foreground is a deep blue-green color, with white foam at the crest, and it is breaking on the sandy beach.

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

The video features a single parrot with a vibrant orange head, a yellow body, and green wings and tail. The parrot is perched on a metal bar, which appears to be part of a railing or fence. The background is a blurred view of greenery and buildings, suggesting an outdoor setting. The parrot's movements are minimal, with occasional head turns and slight body shifts.

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

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

A litter of golden retriever puppies playing in the snow. Their heads pop out of the snow, covered in.

- Fig. 9: The qualitative comparison between our method and the baseline CogVideoX-2B [31]. The first row shows results from the baseline CogVideoX-2B [31], while the second row presents the results generated by RepVideo, demonstrating significant improvements in quality and coherence.

as evidenced by the clear preference expressed by human evaluators.

C. Ablation Study

To validate the effectiveness of our proposed method, we conducted two experiments that highlight how RepVideo

improves both spatial appearance and temporal consistency. These evaluations combine qualitative visualizations and quantitative metrics to demonstrate the advantages of our approach.

How RepVideo Improves Spatial Appearance. Figure 10 provides a layer-wise comparison of feature maps between CogVideoX-2B [31] and RepVideo. The results demonstrate

Frame Layer14 Layer18 Layer22 Layer26

CogVideoX-2BRepVideoCogVideoX-2B

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

A person clad in a space suit with a helmet and equipped with a chest light and arm device is seen closely examining and interacting with a variety of plants in a lush, indoor botanical setting.

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

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

RepVideo

A man wearing a hat and a dark suit walks from the corridor towards the room. The lighting casts a bluish tint over the scene, creating a suspenseful atmosphere.

- Fig. 10: The Layer-wise comparison of feature maps between CogVideoX-2B and RepVideo. The comparison shows that RepVideo consistently captures richer semantic information and maintains more coherent spatial details across layers compared to CogVideoX-2B [31].

that our model consistently captures richer semantic information and maintains more coherent spatial details as the layers deepen. For instance, in the example “A man wearing a hat and a dark suit walks from the corridor towards the room,” the feature maps generated by RepVideo clearly preserve the structure and contours of the man, ensuring that the generated video retains sharp and well-defined spatial attributes. The deeper layers of CogVideoX-2B [31], in contrast, display feature maps that are blurred and lack focus, failing to capture key semantic elements of the scene.

Similarly, in the example “A person clad in a space suit is seen interacting with a variety of plants in a lush, indoor botanical setting,” RepVideo’s feature maps demonstrate a superior ability to capture fine-grained details, such as the distinct outlines of the astronaut and the plants. As the layer depth increases, our model retains semantic consistency and spatial integrity, whereas CogVideoX-2B [31] struggles with spatial coherence, often leading to visual artifacts in the generated videos. These results underscore the importance of RepVideo’s enriched representations, which aggregate information across layers to stabilize spatial details and enhance semantic alignment.

The improved spatial appearance can be attributed to our feature aggregation mechanism, which leverages intermediate representations from consecutive layers. As shown in Fig-

ure 11, the visualization of the attention map demonstrates whether the objects in the frame are activated under different prompts. The attention map of our RepVideo clearly shows the boundary of different subjects mentioned in the prompts compared with CogVideoX, demonstrating that our aggregated features can help to strengthen the corresponding spatial regions. By combining these features, RepVideo reduces variability across layers and ensures that critical spatial information is preserved throughout the video generation process. This mechanism enhances the model’s ability to generate scenes that are visually consistent and aligned with the input prompts.

How RepVideo Improves Temporal Consistency. In addition to improving spatial details, RepVideo significantly enhances temporal consistency, a key factor in generating coherent videos. To quantify this improvement, we calculated the cosine similarity between consecutive frames, as shown in Figure 12. The x-axis represents the layer depth, while the y-axis indicates the cosine similarity between the nth frame and the (n+1)th frame. Higher similarity values suggest better temporal stability, with smaller variations between consecutive frames.

The results demonstrate that RepVideo achieves consistently higher cosine similarity scores across all layers compared to CogVideoX-2B [31]. For example, at deeper layers such as

##### CogVideoX-2B RepVideo

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

A young woman with long, flowing hair sits at a grand piano in a dimly lit room, her fingers gracefully dancing across the keys. She wears a flowing white dress that contrasts beautifully with the dark wood of the piano. The camera captures her intense concentration, her eyes closed as she loses herself in the music. The soft glow of a nearby lamp casts a warm light on her face, highlighting her serene expression. The room is adorned with vintage decor, including a framed painting and a vase of fresh flowers on a side table, adding to the intimate and timeless atmosphere.

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

The camera follows behind a white vintage SUV with a black roof rack as it speeds up a steep dirt road surrounded by pine trees on a steep mountain slope, dust kicks up from its tires, the sunlight shines on the SUV as it speeds along the dirt road, casting a warm glow over the scene. The dirt road curves gently into the distance, with no other cars or vehicles in sight. The trees on either side of the road are redwoods, with patches of greenery scattered throughout. The car is seen from the rear following the curve with ease, making it seem as if it is on a rugged drive through the rugged terrain. The dirt road itself is surrounded by steep hills and mountains, with a clear blue sky above with wispy clouds.

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

In a still frame, the Temple of Hephaestus, with its timeless Doric grace, stands stoically against the backdrop of a quiet Athens. The ancient structure, bathed in the soft glow of the setting sun, reveals intricate details of its columns and pediments. The sky, painted in hues of orange and pink, casts a serene light over the scene. Surrounding the temple, lush greenery and scattered ruins hint at the rich history of the area. In the distance, the modern city of Athens lies in peaceful contrast, its buildings and streets muted in the twilight, emphasizing the enduring presence of this classical marvel.

- Fig. 11: The comparison of attention maps between CogVideoX-2B and RepVideo. The comparison demonstrates that RepVideo could maintain more consistent semantic relationship compared to CogVideoX-2B [31].

Layer 25 and Layer 30, RepVideo maintains a strong frameto-frame similarity, ensuring that the generated videos are free from temporal artifacts like flickering or jittering. In contrast, CogVideoX-2B [31] shows a sharp decline in similarity as the layer depth increases, indicating weaker temporal consistency and a higher likelihood of motion discontinuities.

The enhanced temporal consistency achieved by RepVideo is due to its innovative feature cache module. By aggregating features from multiple adjacent layers, the module stabilizes intermediate representations and minimizes temporal variability. This approach ensures that each frame is not only semantically aligned with the input prompt but also consistent with preceding and succeeding frames. As a result, RepVideo produces videos with smooth transitions and coherent motion, even in complex scenarios involving dynamic objects or environments.

V. DISCUSSION

While RepVideo achieves notable advancements over existing methods, there are areas where our approach could be improved. A key limitation comes from our reliance on pretrained models, such as CogVideoX-2B [31]. These models,

while robust, carry inherent biases and constraints from their original training datasets, which can limit the diversity and adaptability of the generated videos, particularly in scenarios that fall outside the training data distribution.

Another area for improvement is the computational cost associated with the feature aggregation mechanism. Although the approach is relatively lightweight compared to adding new parameters, it may still present challenges for realtime applications or deployment in environments with limited resources. Finding ways to streamline this mechanism without compromising its effectiveness remains an important direction for future work.

Moreover, the method could perform better in generating human-centric content and understanding complex spatial relationships. As highlighted in our experiments, RepVideo occasionally struggles to maintain consistency in videos that require precise modeling of human actions or intricate object arrangements.

Future research could focus on developing real-time feature aggregation mechanism. Additionally, integrating our method with different text-to-video pretrained-model could be beneficial to the whole community. Addressing these limitations

Layer 5 Layer 10 Layer 15

Layer 20 Layer 25 Layer 30

###### Fig. 12: The cosine similarity between consecutive frames across layers.

could further expand the potential of RepVideo, enabling it to set new benchmarks in text-to-video generation tasks.

VI. CONCLUSION

This paper presented RepVideo, a framework designed to improve text-to-video diffusion models by addressing challenges in spatial appearance and temporal consistency. Our approach leverages enriched intermediate representations, combining feature aggregation and gating mechanisms to enhance semantic stability across frames and maintain fine-grained spatial details. Built on the solid foundation of CogVideoX2B [31], RepVideo demonstrates substantial improvements in generating visually consistent and semantically aligned videos.

Through extensive experiments, we validated the effectiveness of our framework. RepVideo outperforms state-of-the-art models across automated benchmarks and human evaluations, showing its ability to maintain temporal coherence in dynamic scenarios and to generate detailed, accurate spatial representations. These contributions represent a meaningful advancement in text-to-video generation, providing a strong platform for future innovation and application in this evolving field.

ACKNOWLEDGMENT

This study is supported by the Ministry of Education, Singapore, under its MOE AcRF Tier 2 (MOET2EP202210012), NTU NAP, and under the RIE2020 Industry Alignment Fund – Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s).

REFERENCES

- [1] P. Esser, R. Rombach, and B. Ommer, “Taming transformers for highresolution image synthesis,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 12873–12883.
- [2] A. Ramesh, M. Pavlov, G. Goh, S. Gray, C. Voss, A. Radford, M. Chen, and I. Sutskever, “Zero-shot text-to-image generation,” in International Conference on Machine Learning. PMLR, 2021, pp. 8821–8831.
- [3] M. Ding, Z. Yang, W. Hong, W. Zheng, C. Zhou, D. Yin, J. Lin, X. Zou, Z. Shao, H. Yang et al., “Cogview: Mastering text-to-image generation via transformers,” Advances in Neural Information Processing Systems, vol. 34, pp. 19822–19835, 2021.
- [4] J. Lin, R. Men, A. Yang, C. Zhou, M. Ding, Y. Zhang, P. Wang, A. Wang, L. Jiang, X. Jia et al., “M6: A chinese multimodal pretrainer,” arXiv preprint arXiv:2103.00823, 2021.
- [5] J. Yu, Y. Xu, J. Y. Koh, T. Luong, G. Baid, Z. Wang, V. Vasudevan, A. Ku, Y. Yang, B. K. Ayan et al., “Scaling autoregressive models for content-rich text-to-image generation,” arXiv preprint arXiv:2206.10789, 2022.
- [6] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in Neural Information Processing Systems, vol. 33, pp. 6840– 6851, 2020.
- [7] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” arXiv preprint arXiv:2010.02502, 2020.
- [8] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 10684–10695.
- [9] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4195–4205.
- [10] J. Chen, J. Yu, C. Ge, L. Yao, E. Xie, Y. Wu, Z. Wang, J. Kwok, P. Luo, H. Lu et al., “Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis,” arXiv preprint arXiv:2310.00426, 2023.
- [11] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. M¨uller, J. Penna, and R. Rombach, “Sdxl: Improving latent diffusion models for high-resolution image synthesis,” in The Twelfth International Conference on Learning Representations.
- [12] U. Singer, A. Polyak, T. Hayes, X. Yin, J. An, S. Zhang, Q. Hu, H. Yang, O. Ashual, O. Gafni et al., “Make-a-video: Text-to-video generation without text-video data,” arXiv preprint arXiv:2209.14792, 2022.

- [13] J. Ho, W. Chan, C. Saharia, J. Whang, R. Gao, A. Gritsenko, D. P. Kingma, B. Poole, M. Norouzi, D. J. Fleet et al., “Imagen video: High definition video generation with diffusion models,” arXiv preprint arXiv:2210.02303, 2022.
- [14] J. An, S. Zhang, H. Yang, S. Gupta, J.-B. Huang, J. Luo, and X. Yin, “Latent-shift: Latent diffusion with temporal shift for efficient text-tovideo generation,” arXiv preprint arXiv:2304.08477, 2023.
- [15] D. Zhou, W. Wang, H. Yan, W. Lv, Y. Zhu, and J. Feng, “Magicvideo: Efficient video generation with latent diffusion models,” arXiv preprint arXiv:2211.11018, 2022.
- [16] A. Blattmann, R. Rombach, H. Ling, T. Dockhorn, S. W. Kim, S. Fidler, and K. Kreis, “Align your latents: High-resolution video synthesis with latent diffusion models,” arXiv preprint arXiv:2304.08818, 2023.
- [17] Y. Wang, X. Chen, X. Ma, S. Zhou, Z. Huang, Y. Wang, C. Yang, Y. He, J. Yu, P. Yang et al., “Lavie: High-quality video generation with cascaded latent diffusion models,” arXiv preprint arXiv:2309.15103, 2023.
- [18] A. Blattmann, T. Dockhorn, S. Kulal, D. Mendelevitch, M. Kilian, D. Lorenz, Y. Levi, Z. English, V. Voleti, A. Letts et al., “Stable video diffusion: Scaling latent video diffusion models to large datasets,” arXiv preprint arXiv:2311.15127, 2023.
- [19] D. J. Zhang, J. Z. Wu, J.-W. Liu, R. Zhao, L. Ran, Y. Gu, D. Gao, and M. Z. Shou, “Show-1: Marrying pixel and latent diffusion models for text-to-video generation,” International Journal of Computer Vision, pp. 1–15, 2024.
- [20] X. Ma, Y. Wang, G. Jia, X. Chen, Z. Liu, Y.-F. Li, C. Chen, and Y. Qiao, “Latte: Latent diffusion transformer for video generation,” arXiv preprint arXiv:2401.03048, 2024.
- [21] Y. Guo, C. Yang, A. Rao, Z. Liang, Y. Wang, Y. Qiao, M. Agrawala, D. Lin, and B. Dai, “Animatediff: Animate your personalized textto-image diffusion models without specific tuning,” arXiv preprint arXiv:2307.04725, 2023.
- [22] W. Harvey, S. Naderiparizi, V. Masrani, C. Weilbach, and F. Wood, “Flexible diffusion modeling of long videos,” Advances in Neural Information Processing Systems, vol. 35, pp. 27953–27965, 2022.
- [23] H. Chen, M. Xia, Y. He, Y. Zhang, X. Cun, S. Yang, J. Xing, Y. Liu, Q. Chen, X. Wang et al., “Videocrafter1: Open diffusion models for high-quality video generation,” arXiv preprint arXiv:2310.19512, 2023.
- [24] C. Wang, J. Gu, P. Hu, S. Xu, H. Xu, and X. Liang, “Dreamvideo: High-fidelity image-to-video generation with image retention and text guidance,” arXiv preprint arXiv:2312.03018, 2023.
- [25] W. Menapace, A. Siarohin, I. Skorokhodov, E. Deyneka, T.-S. Chen, A. Kag, Y. Fang, A. Stoliar, E. Ricci, J. Ren et al., “Snap video: Scaled spatiotemporal transformers for text-to-video synthesis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 7038–7048.
- [26] Y. Jiang, T. Wu, S. Yang, C. Si, D. Lin, Y. Qiao, C. C. Loy, and Z. Liu, “Videobooth: Diffusion-based video generation with image prompts,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 6689–6700.
- [27] Y. He, T. Yang, Y. Zhang, Y. Shan, and Q. Chen, “Latent video diffusion models for high-fidelity video generation with arbitrary lengths,” arXiv preprint arXiv:2211.13221, 2022.
- [28] J. Z. Wu, Y. Ge, X. Wang, W. Lei, Y. Gu, W. Hsu, Y. Shan, X. Qie, and M. Z. Shou, “Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation,” arXiv preprint arXiv:2212.11565, 2022.
- [29] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. M¨uller, H. Saini, Y. Levi, D. Lorenz, A. Sauer, F. Boesel et al., “Scaling rectified flow transformers for high-resolution image synthesis,” in Forty-first International Conference on Machine Learning, 2024.
- [30] T. Brooks, B. Peebles, C. Holmes, W. DePue, Y. Guo, L. Jing, D. Schnurr, J. Taylor, T. Luhman, E. Luhman, C. Ng, R. Wang, and A. Ramesh, “Video generation models as world simulators,” 2024. [Online]. Available: https: //openai.com/research/video-generation-models-as-world-simulators
- [31] Z. Yang, J. Teng, W. Zheng, M. Ding, S. Huang, J. Xu, Y. Yang, W. Hong, X. Zhang, G. Feng et al., “Cogvideox: Text-to-video diffusion models with an expert transformer,” arXiv preprint arXiv:2408.06072, 2024.
- [32] F. Bao, C. Xiang, G. Yue, G. He, H. Zhu, K. Zheng, M. Zhao, S. Liu, Y. Wang, and J. Zhu, “Vidu: a highly consistent, dynamic and skilled text-to-video generator with diffusion models,” arXiv preprint arXiv:2405.04233, 2024.
- [33] A. Polyak, A. Zohar, A. Brown, A. Tjandra, A. Sinha, A. Lee, A. Vyas, B. Shi, C.-Y. Ma, C.-Y. Chuang et al., “Movie gen: A cast of media foundation models,” arXiv preprint arXiv:2410.13720, 2024.

- [34] T. Xu, P. Zhang, Q. Huang, H. Zhang, Z. Gan, X. Huang, and X. He, “Attngan: Fine-grained text to image generation with attentional generative adversarial networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 1316–1324.
- [35] M. Zhu, P. Pan, W. Chen, and Y. Yang, “Dm-gan: Dynamic memory generative adversarial networks for text-to-image synthesis,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 5802–5810.
- [36] M. Tao, H. Tang, F. Wu, X.-Y. Jing, B.-K. Bao, and C. Xu, “Dfgan: A simple and effective baseline for text-to-image synthesis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 16515–16525.
- [37] A. Van Den Oord, O. Vinyals et al., “Neural discrete representation learning,” Advances in neural information processing systems, vol. 30, 2017.
- [38] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen, “Hierarchical text-conditional image generation with clip latents,” arXiv preprint arXiv:2204.06125, 2022.
- [39] A. Q. Nichol, P. Dhariwal, A. Ramesh, P. Shyam, P. Mishkin, B. Mcgrew,

I. Sutskever, and M. Chen, “Glide: Towards photorealistic image generation and editing with text-guided diffusion models,” in International Conference on Machine Learning, 2022, pp. 16784–16804.

- [40] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. L. Denton, K. Ghasemipour, R. Gontijo Lopes, B. Karagol Ayan, T. Salimans et al., “Photorealistic text-to-image diffusion models with deep language understanding,” Advances in Neural Information Processing Systems, vol. 35, pp. 36479–36494, 2022.
- [41] Y. Li, M. Min, D. Shen, D. Carlson, and L. Carin, “Video generation from text,” in Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1, 2018.
- [42] G. Mittal, T. Marwah, and V. N. Balasubramanian, “Sync-draw: Automatic video generation using deep recurrent attentive architectures,” in Proceedings of the 25th ACM international conference on Multimedia, 2017, pp. 1096–1104.
- [43] Y. Pan, Z. Qiu, T. Yao, H. Li, and T. Mei, “To create what you tell: Generating videos from captions,” in Proceedings of the 25th ACM international conference on Multimedia, 2017, pp. 1789–1798.
- [44] C. Wu, L. Huang, Q. Zhang, B. Li, L. Ji, F. Yang, G. Sapiro, and N. Duan, “Godiva: Generating open-domain videos from natural descriptions,” arXiv preprint arXiv:2104.14806, 2021.
- [45] W. Yan, Y. Zhang, P. Abbeel, and A. Srinivas, “Videogpt: Video generation using vq-vae and transformers,” arXiv preprint arXiv:2104.10157, 2021.
- [46] S. Ge, T. Hayes, H. Yang, X. Yin, G. Pang, D. Jacobs, J.-B. Huang, and D. Parikh, “Long video generation with time-agnostic vqgan and timesensitive transformer,” in Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XVII. Springer, 2022, pp. 102–118.
- [47] C. Wu, J. Liang, L. Ji, F. Yang, Y. Fang, D. Jiang, and N. Duan, “N¨uwa: Visual synthesis pre-training for neural visual world creation,” in Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XVI. Springer, 2022, pp. 720–736.
- [48] W. Hong, M. Ding, W. Zheng, X. Liu, and J. Tang, “Cogvideo: Largescale pretraining for text-to-video generation via transformers,” arXiv preprint arXiv:2205.15868, 2022.
- [49] R. Villegas, M. Babaeizadeh, P.-J. Kindermans, H. Moraldo, H. Zhang, M. T. Saffar, S. Castro, J. Kunze, and D. Erhan, “Phenaki: Variable length video generation from open domain textual description,” arXiv preprint arXiv:2210.02399, 2022.
- [50] J. Ho, T. Salimans, A. Gritsenko, W. Chan, M. Norouzi, and D. J. Fleet, “Video diffusion models,” arXiv preprint arXiv:2204.03458, 2022.
- [51] Z. Luo, D. Chen, Y. Zhang, Y. Huang, L. Wang, Y. Shen, D. Zhao, J. Zhou, and T. Tan, “Videofusion: Decomposed diffusion models for high-quality video generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 10209–10218.
- [52] D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” arXiv preprint arXiv:1312.6114, 2013.
- [53] B. F. Labs, “Flux.1,” https://github.com/black-forest-labs/flux?tab= readme-ov-file, 2024.
- [54] Z. Huang, Y. He, J. Yu, F. Zhang, C. Si, Y. Jiang, Y. Zhang, T. Wu, Q. Jin, N. Chanpaisit et al., “Vbench: Comprehensive benchmark suite for video generative models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21807–21818.
- [55] H. Chen, Y. Zhang, X. Cun, M. Xia, X. Wang, C. Weng, and Y. Shan, “Videocrafter2: Overcoming data limitations for high-quality video

- diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 7310–7320.
- [56] B. Lin, Y. Ge, X. Cheng, Z. Li, B. Zhu, S. Wang, X. He, Y. Ye, S. Yuan, L. Chen, T. Jia, J. Zhang, Z. Tang, Y. Pang, B. She, C. Yan, Z. Hu, X. Dong, L. Chen, Z. Pan, X. Zhou, S. Dong, Y. Tian, and L. Yuan, “Open-sora plan: Open-source large video generation model,” arXiv preprint arXiv:2412.00131, 2024.
- [57] “Gen-3,” https://runwayml.com/research/introducing-gen-3-alpha, 2024.
- [58] “Gen-2,” https://research.runwayml.com/gen2, 2023.
- [59] “Pika 1.0,” https://www.pika.art/, 2023.
- [60] “Vchitect-2.0,” https://github.com/Vchitect/Vchitect-2.0, 2024.
- [61] Z. Huang, F. Zhang, X. Xu, Y. He, J. Yu, Z. Dong, Q. Ma, N. Chanpaisit, C. Si, Y. Jiang, Y. Wang, X. Chen, Y.-C. Chen, L. Wang, D. Lin, Y. Qiao, and Z. Liu, “Vbench++: Comprehensive and versatile benchmark suite for video generative models,” arXiv preprint arXiv:2411.13503, 2024.

Ziwei Liu is currently a Nanyang Assistant Professor at Nanyang Technological University, Singapore. His research revolves around computer vision, machine learning and computer graphics. He has published extensively on top-tier conferences and journals in relevant fields, including CVPR, ICCV, ECCV, NeurIPS, ICLR, ICML, TPAMI, TOG and Nature - Machine Intelligence. He is the recipient of Microsoft Young Fellowship, Hong Kong PhD Fellowship, ICCV Young Researcher Award, HKSTP Best Paper Award and WAIC Yunfan Award.

[Figure 375]

He serves as an Area Chair of CVPR, ICCV, NeurIPS and ICLR, as well as an Associate Editor of IJCV.

Chenyang Si received the B.S. degree from Zhengzhou University, Zhengzhou, China, in 2016, and the Ph.D. degree from the National Laboratory of Pattern Recognition (NLPR), Institute of Automation, Chinese Academy of Sciences (CASIA), Beijing, China, in 2021. Currently, he is a research fellow at Nanyang Technological University (NTU) Singapore. His research lies at the intersection of deep learning and computer vision, including visionbased human perception (pose and action), few-shot learning, self-supervised learning, semi-supervised

[Figure 376]

learning, network architecture, and image/video generation.

[Figure 377]

Weichen Fan received the bachelor’s degree from University of Electronic Science and Technology of China (UESTC), and the master’s degree from National University of Singapore (NUS). He is currently working toward the Ph.D. degree with MMLab@NTU, Nanyang Technological University, supervised by Prof. Ziwei Liu. His research interests include image/video generation, robotics, and robotic simulation.

[Figure 378]

Zhengyao Lv is currently a Ph.D. student at the University of Hong Kong, supervised by Prof. Kenneth K.Y. Wong. He received his Master’s degree in 2022 and his Bachelor’s degree in 2020 from Harbin Institute of Technology. His current research interests focus on visual generation.

[Figure 379]

Ziqi Huang is currently a Ph.D. student at MMLab@NTU, Nanyang Technological University (NTU), supervised by Prof. Ziwei Liu. She received her Bachelor’s degree from NTU in 2022. Her current research interests include visual generation and evaluation. She is awarded Google PhD Fellowship 2023.

