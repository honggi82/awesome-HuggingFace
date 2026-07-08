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

## Cobra: Efficient Line Art COlorization with BRoAder References

JUNHAO ZHUANG, Tsinghua University, China LINGEN LI, The Chinese University of Hong Kong, China XUAN JU, The Chinese University of Hong Kong, China ZHAOYANG ZHANG∗, Tencent ARC Lab, China CHUN YUAN†, Tsinghua University, China YING SHAN†, Tencent ARC Lab, China

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

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

# arXiv:2504.12240v3[cs.CV]6May2025

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

Reference Pool

|[Figure 118]|
|---|

|[Figure 119]|
|---|

|[Figure 120]|
|---|

|[Figure 121]|
|---|

|[Figure 122]|
|---|

|[Figure 123]|
|---|

|[Figure 124]|
|---|

|[Figure 125]|
|---|

|[Figure 126]|
|---|

|[Figure 127]|
|---|

|[Figure 128]|
|---|

|[Figure 129]|
|---|

|[Figure 130]|
|---|

|[Figure 131]|
|---|

|[Figure 132]|
|---|

|[Figure 133]|
|---|

|[Figure 134]|
|---|

|[Figure 135]|
|---|

|[Figure 136]|
|---|

|[Figure 137]|
|---|

200+ Images) …

(Long Context,

Single or Several Image(s) as Reference

[Figure 138]

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

Ref. Images

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

Ref. Image Ref. Image

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

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

Fig. 1. Cobra is a novel efficient long-context fine-grained ID preservation framework for line art colorization, achieving high precision, efficiency, and flexible usability for comic colorization. By effectively integrating extensive contextual references, it transforms black-and-white line art into vibrant illustrations.

The comic production industry requires reference-based line art colorization with high accuracy, efficiency, contextual consistency, and flexible control. A comic page often involves diverse characters, objects, and backgrounds, which complicates the coloring process. Despite advancements in diffusion models for image generation, their application in line art colorization remains limited, facing challenges related to handling extensive reference images, time-consuming inference, and flexible control. We investigate the necessity of extensive contextual image guidance on the quality of line art colorization. To address these challenges, we introduce Cobra, an efficient and versatile method that supports color hints and utilizes over 200 reference images while maintaining low latency. Central to Cobra is a Causal Sparse DiT

architecture, which leverages specially designed positional encodings, causal sparse attention, and Key-Value Cache to effectively manage long-context references and ensure color identity consistency. Results demonstrate that Cobra achieves accurate line art colorization through extensive contextual reference, significantly enhancing inference speed and interactivity, thereby meeting critical industrial demands. We release our codes and models on our project page: https://zhuang2002.github.io/Cobra/.

###### CCS Concepts: • Computing methodologies → Computer vision.

Additional Key Words and Phrases: Artificial Intelligence Generated Content, Computer Vision, Image Colorization

∗Project Lead. †Co-corresponding authors: Chun Yuan and Ying Shan.

###### ACM Reference Format:

Authors’ addresses: JUNHAO ZHUANG, Tsinghua University, China, zhuangjh23@ mails.tsinghua.edu.cn; LINGEN LI, The Chinese University of Hong Kong, China, lgli@link.cuhk.edu.hk; XUAN JU, The Chinese University of Hong Kong, China, juxuan. 27@gmail.com; ZHAOYANG ZHANG, Tencent ARC Lab, China, zhaoyangzhang@link. cuhk.edu.hk; CHUN YUAN, Tsinghua University, China, yuanc@sz.tsinghua.edu.cn; YING SHAN, Tencent ARC Lab, China, yingsshan@tencent.com.

JUNHAO ZHUANG, LINGEN LI, XUAN JU, ZHAOYANG ZHANG, CHUN YUAN, and YING SHAN. 2025. Cobra: Efficient Line Art COlorization with BRoAder References. ACM Trans. Graph. 1, 1 (May 2025), 11 pages. https: //doi.org/10.1145/nnnnnnn.nnnnnnn

#### 1 INTRODUCTION

Diffusion models [Rombach et al. 2022a; Zhang et al. 2023] have transformed image generation, inpainting [Ju et al. 2024; Zhuang et al. 2023], and editing [Brooks et al. 2023; Hertz et al. 2022], yet their application in multi-reference-based colorization, especially for industrial-scale tasks, remains underexplored.

Early solutions for line art colorization focused on palettes, color hints, and text control mechanisms. Palettes [Chang et al. 2015; Wang et al. 2022] provide consistency but limit flexibility in diverse comic styles. Color hint methods [Liang et al. 2024; Xian et al. 2018; Yun et al. 2023; Zhang et al. 2018] offer adaptability, yet they lack the automation required for rapid industrial applications. Text control [Bahng et al. 2018; Chang et al. 2022; Zhang et al. 2023] enables intuitive guidance but the text encoder is computationally expensive and sensitive to input clarity.

Existing reference-based colorization methods [Cao et al. 2023; Furusawa et al. 2017; Zhuang et al. 2024] address these challenges but are limited by the number of reference images, slow inference times, and an inability to handle complex, context-dependent comic pages. ScreenVAE [Xie et al. 2020] extracts style vectors from a single reference image for manga colorization, while ColorFlow [Zhuang et al. 2024] employs a three-stage framework with dual-branch networks, limited to 12 references. AniDoc [Meng et al. 2024] focuses on singlecharacter anime video colorization. However, comic pages feature diverse items, backgrounds, and character details that are contextdependent and dispersed across various pages. Ensuring color ID consistency requires long-context references. Current methods are inadequate for practical industrial applications due to their limited reference images, lengthy inference times, and lack of support for diverse input formats, such as color hint references.

To address these challenges, we introduce Cobra, a novel longcontext line art colorization framework that offers high precision, efficiency, and flexible usability. It supports over 200 reference images and color hints as input. Rich contextual references are crucial for comic art colorization, as shown in Tab. 4. Cobra enhances accuracy and efficiency through four key innovations: (1) Multi-Identity Consistency: By spatially concatenating clean reference image latents with noise latents, we use attention mechanisms to maintain consistency between reference images and target outputs. (2) Efficient Attention Design: We sparsify traditional full attention by eliminating pairwise attention computations among reference images, thereby reducing redundant interactions. Inspired by advances in language modeling [Pope et al. 2022; Vaswani et al. 2023; Yenduri et al. 2023], we further adopt a causal attention mechanism along with a KV-cache, which significantly reduces memory and computational costs. These design choices collectively improve inference efficiency without sacrificing colorization quality. (3) Flexible Position Encoding: The Localized Reusable Position Encoding allows reuse of local positional encodings without altering pre-trained DiT

##### 2D encodings, accommodating an arbitrary number of references within a constrained resolution. (4) Color Hint Integration: Cobra also supports the integration of color hints, ensuring flexible usage and enhancing the line art colorization process.

To comprehensively evaluate comic page line art colorization, we introduce Cobra-bench, which includes 30 comic chapters, each

with 50 line art images for colorization and 100 reference images. Extensive evaluations demonstrate that Cobra surpasses existing baselines in image quality, color ID accuracy, and inference efficiency, particularly with richer contextual information. Cobra also requires minimal inference modifications to adapt to anime video colorization. Our contributions are summarized as follows:

- • We introduce Cobra, an innovative framework designed for longcontext comic line art colorization, achieving high precision, efficiency, and flexible control, supporting both color hints and long reference contexts.
- • We present the Causal Sparse DiT architecture, featuring Localized Reusable Position Encoding and Causal Sparse Attention with KV-Cache. This design significantly reduces the computational complexity of traditional full attention mechanisms, enabling the processing of over 200 reference images with lower overhead and reduced latency.
- • We establish Cobra-bench, a comprehensive benchmark for multireference-based comic line art colorization. Extensive evaluations show that broader image references are crucial for high-quality comic colorization, with our method significantly outperforming existing techniques in image quality and color ID accuracy.

2 RELATED WORK 2.1 Image Colorization

Image colorization [Furusawa et al. 2017; Zhang et al. 2018] converts grayscale images, such as manga [Qu et al. 2006], line art [Kim et al. 2019], sketches [Zhang et al. 2023], and natural grayscale images [Zabari et al. 2023], into colored images. Previous methods rely on text [Chang et al. 2023; Weng et al. 2024; Zabari et al. 2023; Zhang et al. 2023] and palette controls [Utintu et al. 2024; Wang et al. 2022; Wu et al. 2023b; Xiao et al. 2020] or require manual color hints (e.g., scribbles) [Ci et al. 2018; Dou et al. 2021; Liu et al. 2018; Yun et al. 2023; Zhang et al. 2021, 2018], but these approaches only provide rough color styles and fail to ensure accurate color preservation for identities in black-and-white images. To improve controllability, reference-based colorization [Cao et al. 2023; Furusawa et al. 2017; Wang et al. 2023; Wu et al. 2023a,b; Zou et al. 2024], which uses reference images for guidance, has become popular. For instance, ColorFlow [Zhuang et al. 2024] uses a three-stage framework for retrieval, colorization, and upscaling, while AniDoc [Meng et al. 2024] employs a dual-branch 3D UNet framework with an animation character image as a reference for line art video colorization.

Despite these advancements, existing methods face limitations in practical applications due to the diverse objects and characters on a single comic page and challenges in retrieval accuracy. They often struggle with limited reference images, long inference times, and inadequate support for varied input formats like color hint references. To overcome these issues, we propose Cobra, which uses Localized Reusable Position Encoding and Causal Sparse Attention to enable efficient long-context reference, resulting in more accurate and faster comic line art colorization.

[Figure 217]

Cobra: Efficient Line Art COlorization with BRoAder References • 3

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

|[Figure 233]|
|---|

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

### …

4 × K Reference Image

Long-Context Reference

TimeStep = 0 MLP

× N Blocks

[Figure 250]

Localized Reusable Position Encoding

[Figure 251]

[Figure 252]

Causal Sparse DiT

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

❄ ❄ ❄ ❄ ❄ ❄

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

|[Figure 261]| |
|---|---|
| | |

[Figure 262]

[Figure 263]

ε

C

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

### …

[Figure 268]

[Figure 269]

|[Figure 270]|
|---|

KV Cache

KV Cache

KV Cache

Positional Encoding +

Top K Recall

+ +

GSRP

Reference Image Pool

🔥 🔥

🔥

[Figure 271]

[Figure 272]

[Figure 273]

Noise Latent

[Figure 274]

TimeStep = T, T-1, …,1 MLP

[Figure 275]

[Figure 276]

[Figure 277]

Query

[Figure 278]

🔥 🔥 🔥

[Figure 279]

[Figure 280]

Line Art Latent

Causal Sparse Attention (CSA)

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

ε

|[Figure 287]|
|---|

|[Figure 288]|
|---|

[Figure 289]

[Figure 290]

[Figure 291]

… 𝐾 𝐾

[Figure 292]

C

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

|[Figure 297]|
|---|

|[Figure 298]|
|---|

Crop

[Figure 299]

Hint Color Latent

[Figure 300]

Line Art Guider

𝑄𝑄

[Figure 301]

× N/2 Blocks

Concat Add

C

Hint Color (Optional) Hint Mask (Optional)

+

Tunning Weight Frozen Weight

Cross-Attention Lora Block

Self-Attention Causal Sparse Attention

[Figure 302]

🔥 ❄

Attention Map

[Figure 303]

- Fig. 2. The overview of Cobra. This figure depicts the framework of Cobra, which utilizes a large collection of retrieved reference images to guide the colorization of comic line art. The framework effectively manages an arbitrary number of contextual image references through localized reusable positional encoding, ensuring appropriate aspect ratios and resolutions. Additionally, the causal sparse DiT architecture processes long contextual references, enhancing identity preservation and color accuracy while reducing computational complexity. The integration of optional color hints further ensures user flexibility, culminating in high-quality coloring that is highly suitable for industrial applications.

attention restrict In-Context LoRA to generating a limited number of related images. Our approach overcomes these limitations by reusing localized positional encoding patches to handle any number of references and introducing Causal Sparse Attention with KV-Cache for efficient linear complexity relative to the number of reference images.

- 2.2 ldentity Preservation

Identity preservation enables users to provide reference images to generate results containing the given identities. Traditional approaches can be classified into tuning-based and plug-and-play methods. The tuning-based methods [Gal et al. 2022; Jiang et al. 2024; Ruiz et al. 2023] are designed to optimize the weight of the diffusion model to memorize the identity information, which is limited by the consuming online fine-tuning time, large efforts for manually collecting training samples, and the poor generalization ability. Plug-and-play approaches [Chen et al. 2024; Gal et al. 2023; Li et al. 2024; Wang et al. 2024; Ye et al. 2023] usually train an encoder to map reference images to visual representations and use them to guide the diffusion generation process. Although these methods are relatively easy to use, numerous studies [Ding et al. 2022; Ju et al. 2023] have demonstrated that the adapter-based approaches they rely on often lead to suboptimal performance. Recent research on In-Context LoRA [Huang et al. 2024] has demonstrated that the in-context learning capabilities of self-attention can effectively and naturally preserve identities, resulting in superior performance.

3 METHOD

We present Cobra, an efficient long-context fine-grained ID preservation framework designed for line art colorization, providing a robust solution suitable for industrial applications. As illustrated in Fig. 2, Cobra retrieves relevant images from a reference image pool, denoted as 𝑅 = {𝑟1,𝑟2, . . .,𝑟𝑛}. Each reference image 𝑟𝑖 contributes to the colorization of a given line art image 𝐿, guiding the coloring process to enhance the final output. Users can also incorporate color hints to further refine the results. The architecture of Cobra consists of three core components: Causal Sparse DiT, Localized Reusable Position Encoding, and Line Art Guider, as shown in Fig. 2.

However, the resolution limitations of 2D positional encodings in pre-trained diffusion model and the quadratic complexity of full

Key Key

Key

| | | | | |
|---|---|---|---|---|

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

T steps

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

Query

+ Causal Attention (w/ KV Cache)

+ Sparse Attention

1 steps

Full Attention Causal Sparse Attention

- Fig. 3. Illustration of the transition from Full Attention to Causal Sparse Attention. This figure highlights the reduction in computational complexity achieved by excluding pairwise calculations among reference images. Additionally, the application of unidirectional causal attention, along with the use of KV-Cache, further enhances computational efficiency while ensuring effective transmission of essential color ID information.

- 3.1 Causal Sparse DiT

Prior studies [Huang et al. 2024; Zhuang et al. 2024] demonstrate that concatenating images along the spatial dimension improves identity preservation by leveraging the strong contextual matching capabilities of self-attention. However, the high computational overhead of Full Attention limits its practical usage to only a few reference images during both training and inference.

In our framework, we define the sequence length of the line

art latent as 𝑆𝑙 and that of each reference image latent as 𝑆𝑟. The computational complexity of the full attention mechanism across the total time steps𝑇 is given by:

O(𝑇 × (𝑆𝑙2 + 2𝑁 × 𝑆𝑙 × 𝑆𝑟 + 𝑁2 × 𝑆𝑟2)). (1) As the number of reference images 𝑁 increases, this complexity can become substantial.

However, treating all reference images as complete images during attention calculations is inefficient. Instead, reference images should primarily provide color ID information for the line art, allowing us to eliminate pairwise computations among them. To mitigate this inefficiency, we transition from a full attention mechanism to a sparse attention mechanism by excluding calculations between reference images, as illustrated in Fig. 3. This adjustment reduces computational complexity to:

O(𝑇 × (𝑆𝑙2 + 2𝑁 × 𝑆𝑙 × 𝑆𝑟 + 𝑁 × 𝑆𝑟2)). (2) Furthermore,sincethereferenceimages are clean and pre-existing,

they do not require a full diffusion denoising process alongside the noise latents. To maintain independence among the reference image latents, we modify the bidirectional attention between the reference images and noise latents to unidirectional causal attention. This also ensures that the reference image latents preserve color information, which is then transferred to the coloring region. Reference image latents only require one denoise step at time step 0. We implement a KV-Cache to store layer-wise keys and values, providing conditional guidance for the noise latents during the diffusion denoising process and ensuring consistent preservation of color IDs, as shown in Fig. 3. By transitioning from Sparse Attention to Causal Sparse Attention, we further reduce the computational complexity to:

O(𝑇 × (𝑆𝑙2 + 𝑁 × 𝑆𝑙 × 𝑆𝑟) + 𝑁 × 𝑆𝑟2). (3) In summary, Causal Sparse DiT 𝐷𝑐𝑠 enhances the efficiency and effectiveness of the colorization process by leveraging optimized

attention mechanisms. This not only reduces computational complexity but also preserves essential color information.

- 3.2 Localized Reusable Position Encoding

Previous works have adopted two main strategies for spatial image concatenation: grid concatenation and horizontal/vertical concatenation. The grid concatenation restricts the number of reference images to specific counts, while horizontal or vertical concatenation often leads to extreme aspect ratios and resolutions, weakening the spatial correspondence between distant reference images and the target region. Moreover, we observe that the pre-trained PixArt-Alpha model—which supports aspect ratios ranging from 0.25 to 4.0 [Chen et al. 2023]—struggles to generate coherent results when more than 8 reference images are concatenated horizontally or vertically.

To address these limitations, we propose Localized Reusable Position Encoding, which enables the integration of an arbitrary number of reference images without modifying existing 2D position encodings. As illustrated in Fig. 2, we divide the line art image into four spatial patches (top-left, bottom-left, top-right, and bottom-right) and retrieve the top-𝑘 most similar reference images for each patch, resulting in four distinct reference sets. We patchify the noise latent and all reference latents, obtaining a noise feature 𝑧 ∈ R𝑑,ℎ,𝑤 and 𝑁 reference features 𝑟 ∈ R𝑑,ℎ2,𝑤2 . The positional encoding of shape (𝑑,ℎ, 2𝑤) is divided into five parts: one of shape (𝑑,ℎ,𝑤) for the central region, and four of shape (𝑑, ℎ2, 𝑤2 ) for the surrounding regions. The noise feature 𝑧 is combined with the central positional encoding, while each reference feature 𝑟 integrates its corresponding local position encoding based on its set. This is equivalent to concatenating the reference features and the noise feature along the spatial dimension, while reusing the local encodings to maintain their proximity to the central area in positional space. This approach allows for an arbitrary number of references, even exceeding 200.

During training, we randomly sample images from each reference set, keeping the total number of references fixed at 3, 6, or 12. This enhances the model’s adaptability to varying numbers and combinations of reference images.

- 3.3 Line Art Guider

We concatenate the line art images latents and color hint images latents as input to the Line Art Guider𝐺. The features from the Line Art Guider are integrated layer-wise into the main branch, enabling precise control over line art and facilitating flexible incorporation of color hints.

- 3.3.1 Self-Attention-Only Block. Given that the Line Art Guider is solely responsible for receiving image-type control conditions, we eliminate the Cross-Attention layers and retain only the SelfAttention layers. This approach reduces the model’s parameters without compromising its control effectiveness.
- 3.3.2 Line Art Style Augmentation. We observe that existing line art extraction models produce outputs with distinct styles. To enhance the robustness of the Line Art Guider to varying line art styles, we select two line art extractors[Li et al. 2017; Xiang et al. 2021] with significantly different styles. As shown in Fig. 4, during training, we employ line art style augmentation by randomly blending the

|[Figure 304]|
|---|

|[Figure 305]|[Figure 306]|[Figure 307]|[Figure 308]|
|---|---|---|---|

- Fig. 4. An example of line art style augmentation, demonstrating the blending of outputs from two distinct line art extractors. This strategy improves the robustness of the Line Art Guider to diverse artistic styles.

[Figure 309]

| |
|---|

[Figure 310]

| |
|---|

✓

[Figure 311]

❌

|[Figure 312]|
|---|

|[Figure 313]|
|---|

- Fig. 5. Hint Point Sampling Strategy. This method reduces ambiguity by limiting the RGB pixel value variance within hint points to 0.01, effectively preventing hint points from being placed at edge intersections during training. Additionally, we visualize 30,000 randomly sampled hint points to demonstrate their distribution.

two styles at varying proportions, thereby improving the Line Art Guider’s adaptability to diverse line art styles.

- 3.3.3 Hint Point Sampling Strategy. Color hint allows users to specify colors for specific regions directly. To avoid ambiguities during training, such as those illustrated in Fig. 5 where hint points fall on edge intersections, we propose a simple hint points sampling strategy. By constraining the variance of RGB pixels value within the hint points to no more than 0.01, we effectively exclude hint points sampled on edges during the training phase.

The optimization objective for Cobra can be expressed as follows:

L = 𝐸𝑡,𝜖∥𝜖 − 𝐷𝑐𝑠(𝐺(𝑍𝐿,𝑍𝐶,𝑀,𝑡),𝑍𝑅0:𝑁−1,𝑍𝑡,𝑡)∥22. (4) Here, 𝜖 represents Gaussian noise, 𝑍𝐿 is the latent of the line art image, 𝑍𝐶 is the latent of the hint color image, 𝑀 is the hint mask, 𝑍𝑅0:𝑁−1 denotes the latents of 𝑁 reference images, and 𝑍𝑡 is the latent of the original image after undergoing the forward diffusion process at the 𝑡-th timestep. Additionally, we adopt a Guided SuperResolution Pipeline (GSRP), following the methodology of ColorFlow [Zhuang et al. 2024], to rectify structural distortions caused by the imprecise reconstruction of the Variational Autoencoder.

- 4 EXPERIMENTS

- 4.1 Dataset and Benchmark

- 4.1.1 Training Dataset. We trained our model using a large-scale dataset of over 1.2 million color images from more than 50,000 comic chapter sequences sourced from various online repositories, including 1 and 2, all in the public domain. After filtering and cleaning, we used the CLIP image encoder [Radford et al. 2021] to annotate at least 12 relevant reference images per comic page based on image similarity, minimizing redundant computations during training.

- 1https://digitalcomicmuseum.com
- 2https://comicbookplus.com

Cobra: Efficient Line Art COlorization with BRoAder References • 5

4.1.2 Cobra-Bench. To evaluate Cobra comprehensively, we established Cobra-Bench, a benchmark of 30 comic chapters excluded from training. Each chapter includes 100 reference images and 50 line art pages, available in standard and shadowed forms to mimic traditional black-and-white comics. We used five metrics for quantitative assessment against baseline methods: three perceptual measures—CLIP Image Similarity (CLIP-IS)[Radford et al. 2021], Fréchet Inception Distance (FID)[Heusel et al. 2017], and Aesthetic Score (AS)[Schuhmann et al. 2022]—and two pixel-wise metrics—Peak Signal-to-Noise Ratio (PSNR)[Wikipedia contributors 2024] and Structural Similarity Index (SSIM) [Wang et al. 2004]. These metrics collectively provide a robust evaluation of colorization quality, assessing both aesthetic appeal and fidelity to the original content.

- 4.2 Implementation Details

Cobra is trained based on the pretrained PixArt-Alpha [Chen et al. 2023]. The trainable components of Cobra include the Line Art Guider and a LoRA applied to the pretrained Full Attention DiT, which facilitates the transition to our proposed Causal Sparse DiT. We conducted a total of 78,000 training steps with a learning rate of 1e-5 and a batch size of 16, utilizing the AdamW optimizer. The training resolution was set to 640×1024 (width × height), while the resolution of the reference images was 320×512 (width × height). Additionally, since Cobra does not rely on text conditions, we utilized the empty text output from the text encoder as input during both the training and inference phases.

- 4.3 Comparison with baselines

In this section, we conduct a quantitative comparative analysis of Cobra against leading state-of-the-art comic colorization methods. Specifically, we evaluate ColorFlow [Zhuang et al. 2024], a diffusionbased colorization model that leverages reference images, and MangaColorization v2 (MC-v2) [Maksim Golyadkin 2024], a GAN-based model that operates without references. Additionally, we include comparisons with a generative approache that offer similar functionalities. Specifically,we utilize a pre-trained IP-Adapter [Ye et al. 2023] to convert reference images into image prompts, combined with a pre-trained ControlNet [Zhang et al. 2023] that guides Stable Diffusion 1.5 [Rombach et al. 2022b] for the comic line art colorization. In our experiments, We fixed the number of reference images for Cobra at 24. Both ColorFlow and Cobra were configured with an inference step of 10, while the IP-Adapter was set to the official default of 50 steps. To ensure a fair comparison, all images were resized to a resolution of 512×800, consistent with ColorFlow’s default inference resolution, prior to evaluating the PSNR and SSIM.

4.3.1 Qualitative results. As illustrated in Fig. 6 and 7, the approach employed by the IP-Adapter, which maps reference images to coarse image prompts, results in the blending of color IDs, rendering it incapable of achieving reasonable colorization for line art images. Although MC-v2 is capable of comic colorization, its lack of reference information leads to inaccuracies in character coloring, alongside issues of excessive color saturation. Additionally, while ColorFlow can generally provide accurate colorization for line art, its limited number of reference images often results in the omission of critical color IDs, leading to erroneous colors for certain objects or characters.

[Figure 314]

- Fig. 6. Qualitative results of line art colorization, highlighting how Cobra outperforms other methods by accurately preserving color IDs and providing high-quality results, even in complex scenarios.

[Figure 315]

- Fig. 7. Qualitative results of line art with shadows, showcasing Cobra ’s superior ability to maintain color fidelity and enhance detail, demonstrating its robustness and effectiveness in real-world applications.

[Figure 316]

- Fig. 8. Interactive line art colorization using color hints, demonstrating how Cobra allows users to specify color adjustments in designated areas, enhancing control over the colorization process while maintaining overall stability.

Conversely, Cobra effectively references a richer set of contextual images, enabling robust and comprehensive extraction of color IDs. This capability facilitates a more refined and higher-quality result in comic line art colorization, demonstrating its superiority in the practical industrial applications.

4.3.2 Quantitative results. Tab. 1 summarizes the quantitative results. The IP-Adapter performs poorly in line art colorization, significantly trailing both ColorFlow and Cobra across all five evaluation metrics. In contrast, Cobra excels in identity preservation and image quality by utilizing richer contextual information. For line art with shadow colorization, the non-reference MC-v2 consistently underperforms compared to reference-based methods in both perceptual and pixel-wise metrics. Notably, Cobra achieves the best

performance across all evaluation criteria, underscoring its robustness in comic colorization. Additional results pertaining to line art colorization for an anime video are provided in the Appendix.

4.4 Contributions over ColorFlow

Cobra offers several improvements over ColorFlow. It supports arbitrary numbers of references via Localized Reusable Position Encoding, whereas ColorFlow is limited to 12 due to grid concatenation. Cobra also simplifies the architecture by directly concatenating multiple references with the noise latent and processing them through a Causal Sparse DiT, improving both efficiency and image quality. Additionally, Cobra supports color hints for enhanced user control.

As shown in Tab. 2, Cobra outperforms ColorFlow across all metrics with 12 references at 640×1024 resolution, while requiring

- Table 1. Quantitative comparison of Cobra with state-of-the-art comic colorization methods, including CLIP-IS [Radford et al. 2021], FID [Heusel et al. 2017], PSNR [Wikipedia contributors 2024], SSIM [Wang et al. 2004], and AS [Schuhmann et al. 2022] metrics for both line art and line art with shadows.

Method Reference-based

Line Art Line Art + Shadow

CLIP-IS↑ FID↓ PSNR↑ SSIM↑ AS↑ CLIP-IS↑ FID↓ PSNR↑ SSIM↑ AS↑

MC-v2 - - - - - 0.8635 44.09 15.57 0.7630 4.531 IP-Adapter ✓ 0.8284 76.01 8.111 0.5561 4.511 - - - - ColorFlow ✓ 0.9030 26.29 15.20 0.8045 4.630 0.9198 21.79 18.49 0.8631 4.657

Cobra ✓ 0.9176 20.98 16.08 0.8141 4.641 0.9264 18.84 18.96 0.8694 4.674

significantly less time and memory. Even with only 2 references Cobra still achieves superior results.

- Table 2. Quantitative comparison of ColorFlow and Cobra on colorization.

Method CLIP-IS FID↓ PSNR↑ SSIM↑ AS↑ Time(s) Mem(GB)

ColorFlow (12 refs) 0.9030 26.29 15.20 0.8045 4.630 1.03 36.4 Cobra (12 refs) 0.9132 21.86 15.94 0.8136 4.642 0.31 9.3

ColorFlow (2 refs) 0.8798 32.55 14.88 0.8013 4.591 - Cobra (2 refs) 0.8989 26.91 15.19 0.8120 4.611 - -

- 4.5 Colorization Using Color Hints

- Fig. 8 demonstrates interactive line art colorization using userprovided color hint points. When a large set of reference images doesn’t fully meet user needs, Cobra allows users to enhance colorization by adding color hints in specific areas. As shown in the interaction, when a user specifies a color hint, Cobra accurately adjusts the color in that region, demonstrating fidelity to user input and offering precise control for practical applications. The model maintains stability and consistency in areas not influenced by hints.

- 4.6 User Study

- Table 3. Results of the User Study. The table presents the voting rates for Cobra and ColorFlow based on contextual color IDs consistency, plausibility of object colors, and overall aesthetic quality.

Color IDs consistency↑ Color plausibility↑ Aesthetic quality↑ Cobra 79.1% 69.3% 73.2%

ColorFlow 20.9% 30.7% 26.8%

In the user study, we conducted a comparative analysis of Cobra and the suboptimal model ColorFlow, as evaluated in quantitative experiments, across three dimensions: contextual color IDs consistency, plausibility of object colors, and overall aesthetic quality. We collected over 4,000 valid votes, with the results presented in Tab. 3. These findings demonstrate a clear preference for Cobra in all assessed aspects and provide robust evidence of its superior performance in comic colorization.

- 4.7 Ablation Study

- Table 4. Impact of reference image count on the performance of Cobra in line art colorization, demonstrating consistent improvements as the number of reference images increases.

Number CLIP-IS↑ FID↓ PSNR↑ SSIM↑ AS↑

4 0.9083 23.18 15.61 0.8133 4.634 12 0.9132 21.86 15.94 0.8136 4.642 24 0.9176 20.98 16.08 0.8141 4.641 36 0.9183 20.64 16.13 0.8142 4.649

- Table 5. Performance comparison of three attention variants (Full, Sparse, and Causal Sparse Attention) with 24 reference images under FP16 precision.

Attention Type Time(s)/step↓ Computation(TFlops)↓ FID↓

Full Attention 1.99 38.2 Sparse Attention 0.81 14.7 21.07

Causal Sparse Attention 0.35 9.0 20.98

- 4.7.1 Impact of Reference Image Count. In this study, we investigate how the number of reference images impacts line art colorization quality using the Cobra-Bench evaluation framework. We evaluated Cobra with 4, 12, 24, and 36 reference images, as shown in Tab. 4. Our results show that as the number of reference images increases, colorization accuracy improves, with better preservation of small but important details like character accessories and eye color. These details, though occupying small areas, are critical for high-demand applications. Tab. 4 illustrates a steady improvement in all metrics as the number of reference images increases, highlighting the importance of having a larger and more diverse set of reference images to optimize comic line art colorization.
- 4.7.2 Effectiveness of Causal Sparse Attention. We evaluate three attention mechanisms—Full Attention, Sparse Attention, and Causal Sparse Attention (with KV-Cache)—under a fixed setting of 24 reference images and FP16 precision. As reported in Tab. 5, Causal Sparse Attention achieves significantly better inference efficiency than the other two, while maintaining comparable colorization quality. To assess scalability, we further measure inference latency with different numbers of reference images: 4, 16, 32, 64, and 128 (Fig. 9). Full Attention exhibits quadratic growth in inference time, becoming approximately 15× slower than Causal Sparse Attention at 64 references. Causal Sparse Attention also consistently outperforms

In this section, we will analyze the impact of the number of reference images on the quality of line art colorization and the effectiveness of Causal Sparse Attention in accelerating inference.

8

Full Attention

Sparse Attention

TimeperStep(s)

Causal Sparse Attention

6

4

| |
|---|

2

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0

0 20 40 60 80 100 120

Reference Image Number

- Fig. 9. Evaluation of inference time efficiency for Full Attention, Sparse Attention, and Causal Sparse Attention (with KV-Cache) across different counts of reference images.

|[Figure 317]|
|---|

|[Figure 318]|[Figure 319]|
|---|---|

|[Figure 320]|[Figure 321]|
|---|---|

Line Art Reference Image A Result A Reference Image B Result B

- Fig. 10. Limitation of Cobra . Cobra preserves color identity when references share the same character (Result A), but fails when references depict different characters (Result B).

Sparse Attention, requiring only one-third of its latency at 64 references. These results demonstrate the substantial efficiency of Causal Sparse Attention, making it a more favorable choice for practical line art colorization applications.

4.7.3 Limitation. As shown in Fig. 10, Cobra can handle slight variations in art style when transferring colors from references of the same character. However, the model is explicitly designed to transfer consistent color identities for the same character and fails to generalize when references depict different characters, limiting its ability to perform style transfer across identities.

- 5 CONCLUSION

In this work, we introduce Cobra, a novel long-context framework for comic line art colorization. By leveraging a rich set of reference images, our approach enhances colorization accuracy while preserving fine-grained identity details. With components like Causal Sparse DiT and Localized Reusable Position Encoding, Cobra significantly improves efficiency and color fidelity over existing methods. Extensive qualitative and quantitative evaluations demonstrate its superior performance, especially in industrial applications where user control and contextual richness are essential. Overall, Cobra represents a major advancement in comic colorization, providing robust solutions for diverse artistic styles and user needs.

ACKNOWLEDGMENTS

This work was supported by the National Key R&D Program of China(2022YFB4701400/4701402),SSTIC Grant(KJZD20230923115106012, KJZD20230923114916032, GJHZ20240218113604008) and Beijing Key Lab of Networked Multimedia.

REFERENCES

Hyojin Bahng, Seungjoo Yoo, Wonwoong Cho, David Keetae Park, Ziming Wu, Xiaojuan Ma, and Jaegul Choo. 2018. Coloring with words: Guiding image colorization through

text-based palette generation. In Proceedings of the european conference on computer vision (eccv). 431–447.

Tim Brooks, Aleksander Holynski, and Alexei A Efros. 2023. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 18392–18402.

Yu Cao, Xiangqiao Meng, PY Mok, Xueting Liu, Tong-Yee Lee, and Ping Li. 2023. AnimeDiffusion: Anime Face Line Drawing Colorization via Diffusion Models. arXiv preprint arXiv:2303.11137 (2023).

Huiwen Chang, Ohad Fried, Yiming Liu, Stephen DiVerdi, and Adam Finkelstein. 2015. Palette-based photo recoloring. ACM Trans. Graph. 34, 4 (2015), 139–1.

Zheng Chang, Shuchen Weng, Yu Li, Si Li, and Boxin Shi. 2022. L-CoDer: Languagebased colorization with color-object decoupling transformer. In European Conference on Computer Vision. Springer, 360–375.

Zheng Chang, Shuchen Weng, Peixuan Zhang, Yu Li, Si Li, and Boxin Shi. 2023. LCoIns: Language-based colorization with instance awareness. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 19221–19230. Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. 2023. Pixart-𝛼: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426 (2023).

Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. 2024. Anydoor: Zero-shot object-level image customization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6593–6602.

Yuanzheng Ci, Xinzhu Ma, Zhihui Wang, Haojie Li, and Zhongxuan Luo. 2018. Userguided deep anime line art colorization with conditional adversarial networks. In Proceedings of the 26th ACM international conference on Multimedia. 1536–1544. Ning Ding, Yujia Qin, Guang Yang, Fuchao Wei, Zonghan Yang, Yusheng Su, Shengding Hu, Yulin Chen, Chi-Min Chan, Weize Chen, et al. 2022. Delta tuning: A comprehensive study of parameter efficient methods for pre-trained language models. arXiv preprint arXiv:2203.06904 (2022).

Zhi Dou, Ning Wang, Baopu Li, Zhihui Wang, Haojie Li, and Bin Liu. 2021. Dual color space guided sketch colorization. IEEE Transactions on Image Processing 30 (2021), 7292–7304.

Chie Furusawa, Kazuyuki Hiroshiba, Keisuke Ogaki, and Yuri Odagiri. 2017. Comicolorization: semi-automatic manga colorization. In SIGGRAPH Asia 2017 Technical Briefs (Bangkok, Thailand) (SA ’17). Association for Computing Machinery, New York, NY, USA, Article 12, 4 pages. https://doi.org/10.1145/3145749.3149430

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. 2022. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618 (2022).

Rinon Gal, Moab Arar, Yuval Atzmon, Amit H Bermano, Gal Chechik, and Daniel CohenOr. 2023. Encoder-based domain tuning for fast personalization of text-to-image models. ACM Transactions on Graphics (TOG) 42, 4 (2023), 1–13.

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. 2022. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626 (2022).

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. 2017. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. Advances in Neural Information Processing Systems (NIPS) 30 (2017).

Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. 2024. In-context lora for diffusion transformers. arXiv preprint arXiv:2410.23775 (2024).

Yuming Jiang, Tianxing Wu, Shuai Yang, Chenyang Si, Dahua Lin, Yu Qiao, Chen Change Loy, and Ziwei Liu. 2024. Videobooth: Diffusion-based video generation with image prompts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6689–6700.

Xuan Ju, Xian Liu, Xintao Wang, Yuxuan Bian, Ying Shan, and Qiang Xu. 2024. Brushnet: A plug-and-play image inpainting model with decomposed dual-branch diffusion. arXiv preprint arXiv:2403.06976 (2024).

Xuan Ju, Ailing Zeng, Chenchen Zhao, Jianan Wang, Lei Zhang, and Qiang Xu. 2023. Humansd: A native skeleton-guided diffusion model for human image generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 15988– 15998.

Hyunsu Kim, Ho Young Jhoo, Eunhyeok Park, and Sungjoo Yoo. 2019. Tag2pix: Line art colorization using text tag with secat and changing loss. In Proceedings of the IEEE/CVF international conference on computer vision. 9056–9065.

Chengze Li, Xueting Liu, and Tien-Tsin Wong. 2017. Deep Extraction of Manga Structural Lines. ACM Transactions on Graphics (SIGGRAPH 2017 issue) 36, 4 (July 2017), 117:1–117:12.

Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, Ming-Ming Cheng, and Ying Shan. 2024. Photomaker: Customizing realistic human photos via stacked id embedding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8640–8650.

Zhexin Liang, Zhaochen Li, Shangchen Zhou, Chongyi Li, and Chen Change Loy. 2024. Control Color: Multimodal Diffusion-based Interactive Image Colorization. arXiv

preprint arXiv:2402.10855 (2024).

Yifan Liu, Zengchang Qin, Tao Wan, and Zhenbo Luo. 2018. Auto-painter: Cartoon image generation from sketch by using conditional Wasserstein generative adversarial networks. Neurocomputing 311 (2018), 78–87.

Maksim Golyadkin. 2024. Automatic colorization. https://github.com/qweasdd/mangacolorization-v2 [Online; accessed 4-Oct-2024].

Yihao Meng, Hao Ouyang, Hanlin Wang, Qiuyu Wang, Wen Wang, Ka Leong Cheng, Zhiheng Liu, Yujun Shen, and Huamin Qu. 2024. AniDoc: Animation Creation Made Easier. arXiv preprint arXiv:2412.14173 (2024).

Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Anselm Levskaya, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. 2022. Efficiently Scaling Transformer Inference. arXiv:2211.05102 [cs.LG] https: //arxiv.org/abs/2211.05102

Yingge Qu, Tien-Tsin Wong, and Pheng-Ann Heng. 2006. Manga colorization. ACM Transactions on Graphics (ToG) 25, 3 (2006), 1214–1220.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PMLR, 8748–8763.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022a. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022b. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 10684–10695.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. 2023. Dreambooth: Fine tuning text-to-image diffusion models for subjectdriven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 22500–22510.

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. 2022. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems 35 (2022), 25278–25294.

Chaitat Utintu, Pinaki Nath Chowdhury, Aneeshan Sain, Subhadeep Koley, Ayan Kumar Bhunia, and Yi-Zhe Song. 2024. SketchDeco: Decorating B&W Sketches with Colour. arXiv preprint arXiv:2405.18716 (2024).

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2023. Attention Is All You Need. arXiv:1706.03762 [cs.CL] https://arxiv.org/abs/1706.03762

Hanzhang Wang, Deming Zhai, Xianming Liu, Junjun Jiang, and Wen Gao. 2023. Unsupervised deep exemplar colorization via pyramid dual non-local attention. IEEE Transactions on Image Processing (2023).

Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, Anthony Chen, Huaxia Li, Xu Tang, and Yao Hu. 2024. Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519 (2024).

Yi Wang, Menghan Xia, Lu Qi, Jing Shao, and Yu Qiao. 2022. PalGAN: Image colorization with palette generative adversarial networks. In European Conference on Computer Vision. Springer, 271–288.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13, 4 (2004), 600–612.

Shuchen Weng, Peixuan Zhang, Yu Li, Si Li, Boxin Shi, et al. 2024. L-cad: Languagebased colorization with any-level descriptions using diffusion priors. Advances in Neural Information Processing Systems 36 (2024).

Wikipedia contributors. 2024. Peak signal-to-noise ratio — Wikipedia, The Free Encyclopedia. https://en.wikipedia.org/w/index.php?title=Peak_signal-to-noise_ratio& oldid=1210897995 [Online; accessed 4-March-2024].

Shukai Wu, Xiao Yan, Weiming Liu, Shuchang Xu, and Sanyuan Zhang. 2023a. Selfdriven dual-path learning for reference-based line art colorization under limited data. IEEE Transactions on Circuits and Systems for Video Technology (2023).

Shukai Wu, Yuhang Yang, Shuchang Xu, Weiming Liu, Xiao Yan, and Sanyuan Zhang. 2023b. FlexIcon: Flexible Icon Colorization via Guided Images and Palettes. In Proceedings of the 31st ACM International Conference on Multimedia. 8662–8673. Wenqi Xian, Patsorn Sangkloy, Varun Agrawal, Amit Raj, Jingwan Lu, Chen Fang, Fisher Yu, and James Hays. 2018. Texturegan: Controlling deep image synthesis with texture patches. In Proceedings of the IEEE conference on computer vision and pattern recognition. 8456–8465.

Xiaoyu Xiang, Ding Liu, Yiheng Zhu Xiao Yang, and Xiaohui Shen. 2021. Anime2Sketch: A Sketch Extractor for Anime Arts with Deep Networks. https://github.com/ Mukosame/Anime2Sketch.

Chufeng Xiao, Chu Han, Zhuming Zhang, Jing Qin, Tien-Tsin Wong, Guoqiang Han, and Shengfeng He. 2020. Example-Based Colourization Via Dense Encoding Pyramids. In Computer Graphics Forum, Vol. 39. Wiley Online Library, 20–33.

Minshan Xie, Chengze Li, Xueting Liu, and Tien-Tsin Wong. 2020. Manga filling style conversion with screentone variational autoencoder. ACM Transactions on Graphics (TOG) 39, 6 (2020), 1–15.

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. 2023. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721 (2023).

Gokul Yenduri, Ramalingam M, Chemmalar Selvi G, Supriya Y, Gautam Srivastava, Praveen Kumar Reddy Maddikunta, Deepti Raj G, Rutvij H Jhaveri, Prabadevi B, Weizheng Wang, Athanasios V. Vasilakos, and Thippa Reddy Gadekallu. 2023. Generative Pre-trained Transformer: A Comprehensive Review on Enabling Technologies, Potential Applications, Emerging Challenges, and Future Directions. arXiv:2305.10435 [cs.CL] https://arxiv.org/abs/2305.10435

Jooyeol Yun, Sanghyeon Lee, Minho Park, and Jaegul Choo. 2023. iColoriT: Towards propagating local hints to the right region in interactive colorization by leveraging vision transformer. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 1787–1796.

Nir Zabari, Aharon Azulay, Alexey Gorkor, Tavi Halperin, and Ohad Fried. 2023. Diffusing Colors: Image Colorization with Text Guided Diffusion. In SIGGRAPH Asia 2023 Conference Papers. 1–11.

Lvmin Zhang, Chengze Li, Edgar Simo-Serra, Yi Ji, Tien-Tsin Wong, and Chunping Liu.

2021. User-guided line art flat filling with split filling mechanism. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 9889–9898. Lvmin Zhang, Chengze Li, Tien-Tsin Wong, Yi Ji, and Chunping Liu. 2018. Two-stage

sketch colorization. ACM Transactions on Graphics (TOG) 37, 6 (2018), 1–14.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 3836–3847.

Junhao Zhuang, Xuan Ju, Zhaoyang Zhang, Yong Liu, Shiyi Zhang, Chun Yuan, and Ying Shan. 2024. ColorFlow: Retrieval-Augmented Image Sequence Colorization. arXiv preprint arXiv:2412.11815 (2024).

Junhao Zhuang, Yanhong Zeng, Wenran Liu, Chun Yuan, and Kai Chen. 2023. A task is worth one word: Learning with task prompts for high-quality versatile image inpainting. arXiv preprint arXiv:2312.03594 (2023).

Chengyi Zou, Shuai Wan, Marc Gorriz Blanch, Luka Murn, Marta Mrak, Juil Sock, Fei Yang, and Luis Herranz. 2024. Lightweight Deep Exemplar Colorization via Semantic Attention-Guided Laplacian Pyramid. IEEE Transactions on Visualization and Computer Graphics (2024).

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

[Figure 352]

Fig. 11. Additional results of colorization for comic line art.

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

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

Frame 1 Frame 5 Frame 9 Frame 13

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

- Fig. 12. Additional results of line art colorization for an anime video.

[Figure 413]

###### Fig. 13. Additional results of colorization for comic line art with shadow.

