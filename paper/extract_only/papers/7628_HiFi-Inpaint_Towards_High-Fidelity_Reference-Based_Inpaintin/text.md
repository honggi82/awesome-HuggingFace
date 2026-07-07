# arXiv:2603.02210v3[cs.CV]17Apr2026

[Figure 1]

[Figure 2]

[Figure 3]

## HiFi-Inpaint: Towards High-Fidelity Reference-Based Inpainting for Generating Detail-Preserving Human-Product Images

### Yichen Liu1,⋆, Donghao Zhou2,⋆, Jie Wang3, Xin Gao3, Guisheng Liu3, Jiatong Li3,†, Quanwei Zhang4, Qiang Lyu1, Lanqing Guo5, Shilei Wen3,§, Weiqiang Wang1,§, Pheng-Ann Heng2,§

1University of Chinese Academy of Sciences, 2The Chinese University of Hong Kong, 3ByteDance, 4Zhejiang University, 5UT Austin

⋆Equal contribution, †Project lead, §Corresponding author

### Abstract

Human-product images, which showcase the integration of humans and products, play a vital role in advertising, e-commerce, and digital marketing. The essential challenge of generating such images lies in ensuring the high-fidelity preservation of product details. Among existing paradigms, reference-based inpainting offers a targeted solution by leveraging product reference images to guide the inpainting process. However, limitations remain in three key aspects: the lack of diverse large-scale training data, the struggle of current models to focus on product detail preservation, and the inability of coarse supervision for achieving precise guidance. To address these issues, we propose HiFi-Inpaint, a novel high-fidelity reference-based inpainting framework tailored for generating human-product images. HiFi-Inpaint introduces Shared Enhancement Attention (SEA) to refine fine-grained product features and Detail-Aware Loss (DAL) to enforce precise pixel-level supervision using high-frequency maps. Additionally, we construct a new dataset, HP-Image-40K, with samples curated from self-synthesis data and processed with automatic filtering. Experimental results show that HiFi-Inpaint achieves state-of-the-art performance, delivering detail-preserving human-product images.

Note: Accepted by CVPR 2026 Project Page: https://correr-zhou.github.io/HiFi-Inpaint

### 1 Introduction

Recent advancements in diffusion models [6, 34, 39] have revolutionized image generation, enabling the creation of diverse visual content across real-world scenarios [10, 21, 49]. Among these, generating human-product images has emerged as a practical application in industries such as advertising, e-commerce, and digital marketing. This task takes textual descriptions and visual cues as inputs, generating images that seamlessly integrate humans with products. By creating high-quality human-product images, this task enables the automated production of commercial content at unprecedented speed and quality, unlocking new horizons for the real-world impact of image generation. Importantly, it reduces manual design effort while improving image quality and consistency across large-scale deployments.

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

Ref. Generated Image Ref. Generated Image Ref. Generated Image Ref. Generated Image

- Figure 1 HiFi-Inpaint enables high-fidelity reference-based inpainting. Our HiFi-Inpaint can seamlessly integrate product reference images into masked human images, generating high-quality human-product images with high-fidelity detail preservation. To avoid potential privacy and copyright concerns, we use AI-generated products and humans for presentation purposes in this paper. Zoom in for better view.

However, generating high-quality human-product images poses significant challenges and the most critical one lies in ensuring the high-fidelity preservation of product details. Specifically, generated human-product mages should faithfully depict the fine-grained features of products, such as shapes, colors, patterns, and texture, to meet the stringent requirements of real-world use. Otherwise, even subtle inaccuracies can undermine consumer trust and reduce the effectiveness of commercial efforts. Utilizing existing paradigms, such as image customization [40] or text-driven editing [12], to generate these highly demanding images often leads to inferior results. This is because they typically focus on global or high-level semantic manipulation with a free-form input manner, making it difficult to robustly preserve fine-grained details.

One promising paradigm is reference-based inpainting [53], which synthesizes coherent images by leveraging reference images to guide the inpainting process. This approach offers a more structured and targeted mechanism for integrating product-specific features while maintaining overall consistency. However, existing methods [24, 33, 44] still fall short in preserving fine-grained product details. This is largely because diffusionbased methods struggle to faithfully retain reference image details due to limited enforcement of strict spatial and appearance alignment. The denoising process tends to average or hallucinate content, resulting in inconsistencies in texture, shape, and branding elements that are critical for high-fidelity demands.

To address these limitations, we propose HiFi-Inpaint, a novel High-Fidelity reference-based Inpainting framework tailored for generating detail-preserving human-product images (Fig. 2). Our HiFi-Inpaint produces visually consistent and appealing human-product images using a concise text prompt, a masked human image, and a product reference image as inputs. First, to support robust model training, we construct a new dataset, HP-

Image-40K, consisting of 40,000+ samples generated via a self-synthesis pipeline followed by automatic filtering to ensure high-quality, diverse training data. We then design a high-frequency map-guided DiT framework that employs a token merging mechanism to effectively coordinate the integration of multiple image conditions, which additionally considers the injection of high-frequency maps. To further enhance visual fidelity, we introduce Shared Enhancement Attention (SEA), which utilizes shared dual-stream visual DiT blocks to refine visual tokens within masked regions. By replacing product image tokens with corresponding high-frequency map tokens in another branch, SEA helps to enhance fine-grained product features for the original branch of the base model. Finally, we propose a Detail-Aware Loss (DAL) that leverages high-frequency pixel-level supervision to guide the reconstruction of fine-grained product details in masked regions, complementing the shortcomings of relying solely on MSE loss in the latent space. In experiments, we demonstrate that our HiFi-Inpaint can achieve state-of-the-art performance in generating human-product images, especially excelling at preserving high-fidelity details.

In summary, our main contributions are as follows:

- • We propose a novel high-fidelity reference-based inpainting framework , HiFi-Inpaint, incorporating Shared Enhancement Attention (SEA) for enhancing fine-grained product features and Detail-Aware Loss (DAL) for providing precise pixel-level supervision.

- • We curate a large-scale and diverse dataset, HP-Image-40K, with samples curated from self-synthesis data and processed with automated filtering, to provide a solid foundation for model training.
- • Extensive experiments validate the effectiveness of HiFi-Inpaint, showing its superior ability to generate detail-preserving human-product images.

### 2 Related Works

- 2.1 Text-to-Image Generation

Text-to-image (T2I) generation has experienced significant progress in recent years, making it possible to synthesize images directly from textual input. Early methods primarily relied on Generative Adversarial Networks (GANs) [35, 38, 52], while auto-regressive transformers [4, 37, 55] later demonstrated their potential. The introduction of diffusion models has revolutionized T2I generation [12, 56], leading to rapid advancements in related applications, such as image customization [2, 19, 25, 40, 60], image editing [3, 20, 27, 28, 57–59], consistent image generation [43, 61], and controllable generation [9, 13, 14, 18, 26, 29, 31, 42, 47]. However, while T2I diffusion models have achieved remarkable success, generating high-quality human-product images remains a challenging task due to the difficulty of preserving intricate product details. In this study, we explore reference-based inpainting to enable precise and seamless integration of humans and products.

- 2.2 Image Inpainting

Image inpainting aims to restore missing or corrupted regions in an image while maintaining visual coherence. Classical methods relied on optimization techniques [5] or patch-based approaches [16] to fill in gaps based on surrounding context. Diffusion models have further advanced the field, offering powerful tools for inpainting by iteratively denoising images from latent representations [1, 32, 41]. These models can further incorporate additional conditions, providing better control over inpainting tasks [51, 54]. Among these, reference-based inpainting [24, 33, 44, 53] has emerged as a promising paradigm, leveraging reference images to guide the restoration process and ensure consistency in visual context. Despite these advancements, current reference-based inpainting methods still face limitations and thus achieve suboptimal results when applied to human-product images. This highlights the need for a more targeted end-to-end solution to generate detail-preserving human-product images.

- 3 Methodology

- 3.1 Overview

We propose HiFi-Inpaint, a high-fidelity reference-based inpainting framework for generating high-quality human-product images. Given a concise text prompt T, a masked human image Ih, and a product reference image Ip, the goal of our HiFi-Inpaint is to generate an image Ig that seamlessly integrates the visual content of Ip into the mask region of Ih while following the description of T. As illustrated in Fig. 2, we start by constructing a new dataset called HP-Image-40K, with samples synthesized by a pretrained text-to-image (T2I) model and processed by automatic filtering (Sec. 3.2). Then, we design a high-frequency map-guided DiT framework that employs a token merging mechanism and integrates Shared Enhancement Attention (SEA) for enhancing product features (Sec. 3.3). Furthermore, we introduce Detail-Aware Loss (DAL) to achieve high-frequency pixel-level supervision (Sec. 3.4). Below we delve into the details of the above improvements.

- 3.2 Dataset Construction: HP-Image-40K

Collecting real-world human-product image data is time-consuming and labor-intensive. To liberate model training from data constraints, we use a pretrained T2I model to synthesize desired samples and employ an automatic filtering process to process them, enabling the acquisition of large-scale and diverse data with minimal human intervention. Finally, we construct a new dataset called HP-Image-40K, consisting of 40,000+ high-quality samples to facilitate the training of our model. The entire procedure is detailed as follows:

Dataset Construc on

|[Figure 16]|
|---|

|[Figure 17]|
|---|

[Figure 18]

###### Shared Enhancement A en on

“A man is holding a bottle and smiling to the camera”

- Step 1: Diptych Synthesis

- Step 2: Diptych Segmenta on

- Step 3: Seman c Filtering

Intern-VL

Edge Detec on

[Figure 19]

| |
|---|

[Figure 20]

| |
|---|

|[Figure 21]|
|---|

[Figure 22]

| |
|---|

[Figure 23]

| |
|---|

“ZENLUX”

YOLOv8

[Figure 24]

Intern-VL

[Figure 25]

[Figure 26]

| |
|---|

[Figure 27]

| |
|---|

[Figure 28]

CLIP Similarity

Text Overlapping

High-Frequency Extrac on

- Step 4: Textual Filtering

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

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

“A diptych. Left: …

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Full A en on

Full A en on

Full A en on

Right: …” FLUX

Shared

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

FFN

FFN

FFN

Inputs

[Figure 67]

[Figure 68]

Text Encoder

VAE Encoder

|[Figure 69]|
|---|

&

[Figure 70]

[Figure 71]

❄ ❄

+

×

Mask

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

Dual-Stream Blocks

[Figure 89]

[Figure 90]

DiT Blocks

[Figure 91]

🔥

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

Detail-Aware Loss

[Figure 109]

Full A en on

|[Figure 110]|
|---|

|[Figure 111]|
|---|

[Figure 112]

❄

[Figure 113]

[Figure 114]

[Figure 115]

VAE Decoder

FFN

“ZENLUX”

Single-Stream Blocks

- Figure 2 Overview of HiFi-Inpaint. HiFi-Inpaint is a high-fidelity reference-based inpainting framework tailored for generating human-product images. To support model training, we construct HP-Image-40K, a large-scale dataset of human-product images, collected through a self-synthesis pipeline combined with automated filtering to ensure high-quality samples (Sec. 3.2). Furthermore, we introduce two key techniques: (i) Shared Enhancement Attention (SEA), designed to refine fine-grained product features by leveraging high-frequency map tokens within dual-stream visual DiT blocks (Sec. 3.3), and (ii) Detail-Aware Loss (DAL), developed to enforce precise pixel-level supervision by utilizing high-frequency information, enabling the reconstruction of intricate product and human details (Sec. 3.4).

- 1. Diptych Synthesis: We begin by utilizing FLUX.1-Dev [6] to generate diptych-format images, leveraging its capability to retain concept consistency within a generated image [8]. We design a dedicated prompt template: “A diptych. left: [product description] right: [product and human description]” to guide the model in producing semantically aligned diptychs, with the left side depicting a product and the right side showing the expected human-product image.
- 2. Diptych Segmentation: Next, we segment the diptychs to extract individual product images and human-product images for subsequent filtering. To simplify the process, we utilize an edge detection algorithm for efficient segmentation. Specifically, a Sobel filter [17] is applied to precisely locate the vertical boundary between the two halves, ensuring an accurate separation of the left-side product image and the right-side human-product image.
- 3. Semantic Filtering: Ensuring product consistency across the two segmented images is a crucial step in collecting high-quality data. For each image pair, we utilize YOLOv8 [46] to localize the product region in the image. We then compute the CLIP [36] similarity between the cropped image and the product image. Only pairs with high similarity scores are selected, ensuring highly consistent samples.
- 4. Textual Filtering: Textual content on products often represents key brand information, labels, or instructions. To further enhance data quality, we extract textual content from both the product image and the human-product image using InternVL [15]. We then compare the extracted strings by evaluating their overlapping degree. Only pairs with high textual consistency are retained, guaranteeing that the selected samples accurately preserve textual fidelity.

Finally, each sample contains a text prompt T, a masked human image Ih, a product image Ip, and a targeted human-product image Igt. Specifically, T is generated by driving InternVL [15] to describe It, and Ih is produced by masking the region of the detected product. All samples are carefully curated to ensure high quality, supplementing diverse and robust training data to support the generation of high-fidelity human-product images.

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

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

Original Canny Ours Original Canny Ours

Ref. Fixed Learnable Ref. Fixed Learnable

- Figure 3 Comparison with the Canny algorithm. While Canny detects all edges, leading to significant background clutter (red frame), the adopted algorithm highlights key elements like text and logos (blue frame), by being responsive to specific frequencies.

Figure 4 Comparison with fixed weighting of SEA.

Adopting a learnable weighting factor produces more harmonious and realistic results, whereas using a fixed one often leads to visual artifacts and conflicts across the inpainting region.

#### 3.3 High-Frequency Map-Guided DiT

We design a high-frequency map-guided DiT framework based on FLUX.1-Dev [6]. High-frequency extraction is first adopted, and the token merging mechanism is leveraged to integrate image conditions effectively. To improve visual fidelity, we introduce Shared Enhancement Attention (SEA), which refines the visual features within masked regions by incorporating high-frequency map tokens into dual-stream visual DiT blocks.

High-Frequency Extraction. As outlined in Algo. 1, we utilize a dedicated frequency-domain filtering method [22] to extract the high-frequency details of the product. Specifically, the input image is transformed to the frequency domain via the Discrete Fourier Transform (DFT), where a high-pass filter is applied by suppressing low-frequency components near the center using a circular mask of radius r. The filtered spectrum is then transformed back to the spatial domain using the inverse DFT, and the magnitude response is normalized to obtain the high-frequency map. This process is applied to both Ip and the masked product region of Igt, exhibiting more targeted results than conventional edge detection in this scenario (Fig. 3).

Algorithm 1 High-Frequency Extraction Require: Input image I Ensure: High-frequency detail image I′

- 1: Compute DFT: F ← DFT(I)
- 2: Shift zero-frequency to center: Fc ← fftshift(F)
- 3: Construct high-pass mask Mh (center radius r set to zero)
- 4: Apply mask: Fh ← Fc ⊙ Mh
- 5: Inverse shift: F−h1 ← ifftshift(Fh)
- 6: Inverse DFT: I′ ← |IDFT(F−h1)|
- 7: return I′

Token Merging Mechanism. Our base model (i.e., FLUX.1-Dev [6]) adopts the MMDiT architecture. During the training of MMDiT, textual tokens are encoded from the text prompt T, while noisy visual tokens are obtained by VAE encoding the ground-truth image Igt with noise added. To inject the conditions of the masked human image Ih and the product image Ip, we leverage a token merging mechanism (i.e., concatenate their encoded tokens with original noisy visual tokens for joint modeling), resulting in joint visual tokens z0 as

z0 = Concat E(Ih),E(Ip),N E(Igt),t , (1)

where Concat(·) denotes token concatenation, E(·) is the VAE image encoder and N(·,t) is the noise addition with a random timestep t. Furthermore, we extract the high-frequency map of the product image Ip, to obtain another token sequence called high-frequency visual tokens z′0 as

z′0 = Concat E(Ih),E H(Ip) ,N E(Igt),t , (2)

where H(·) denotes the extraction of the high-frequency map (i.e., Algo. 1). We follow [45] to utilize joint visual tokens z0 during training, where textual tokens c0 and noisy visual tokens z0 are first separately refined by single-stream blocks and then jointly processed by dual-stream blocks. This mechanism facilitates the model to effectively fuse visual information, enabling reference-based inpainting by capturing the relationships between multiple input images. As for high-frequency visual tokens z′, we leave the discussion about their usage to the following part.

Shared Enhancement Attention. To improve the model’s ability to retain fine-grained product details, such as shapes, patterns, and textures, we propose Shared Enhancement Attention (SEA) that explicitly leverages high-frequency information from the product image to enhance the product-specific visual features. For each dual-stream visual DiT block, we supplement another branch for high-frequency visual tokens, which leverages the same parameters of the original branch. Let Bi(·) indicate the i-th dual-stream visual DiT block and zi/z′i are the joint/high-frequency visual tokens processed by Bi(·). Specifically, SEA modifies the naive forward process zi = Bi(zi−1) to

zi = Bi(zi−1) + αi · Mask(Bi(z′i−1),Mds), (3)

where αi is a learnable weighting factor, showing better performance than simply fixing to 1 (Fig. 4), and Mask(·,Mds) denotes attention masking operation with Mds as the down-sampled masked regions of Ih. This masking constraint prevents unintended interference from irrelevant regions, ensuring that only the most relevant features are refined. Notably, such a parameter-sharing mechanism can maintain model compactness by introducing only one additional parameter. The design of SEA enables effective integration of high-frequency details with global contextual information, significantly enhancing the model’s ability to capture and preserve intricate product-specific features.

#### 3.4 Detail-Aware Training Strategy

To align with and further boost the improvements in the model architecture, we make an enhancement to the training strategy by incorporating high-frequency pixel-level supervision. This enhancement ensures the accurate preservation of fine-grained details in masked regions while complementing the model’s ability to maintain global consistency.

Detail-Aware Loss. To address the limitations of latent-level supervision, which struggles to provide precise guidance for capturing fine-grained details, we propose Detail-Aware Loss (DAL). This loss leverages highfrequency pixel-level supervision to complement latent-level objectives, encouraging the accurate reconstruction of intricate product details in masked regions. Specifically, DAL can be formulated as

LDA = H(ˆIgt) ⊙ M − H(Igt) ⊙ M

2 2

, (4)

where ˆIgt denotes the predicted ground truth, Igt is the real ground truth, and M is the original masked region of Ih. This formulation ensures that the supervision focuses specifically on the high-frequency components of the masked regions, which are captured by H(·). By emphasizing fine-grained details, DAL effectively guides the model to reconstruct these details that are otherwise challenging to recover through latent-level supervision alone.

Overall Loss Formulation. In addition, we also adopt MSE loss in the latent space to ensure stable global optimization. This latent-level supervision focuses on preserving the overall semantic and coherence of the generated image, while the pixel-level supervision provided by DAL refines the intricate details within the masked regions. The full formulation of the overall loss can be expressed as

LOverall = LMSE + LDA, (5)

where LMSE is the latent-level MSE loss to focus on the reconstruction of clean tokens of Igt. Finally, the model is trained with LOverall using flow matching [30]. Combining these complementary supervision signals, our training strategy achieves a balanced improvement in both global consistency and local detail fidelity.

### 4 Experiments

#### 4.1 Setups

Implementation Details. In our HiFi-Inpaint, we adopt FLUX.1-Dev [6] as the base model. We utilized an internal dataset consisting of approximately 14,000 samples, which was combined with our curated HPImage-40K dataset to support model training. We trained the model with a learning rate of 5 × 10−5 and a total batch size of 24 for 10,000 steps. All images were processed at a resolution of 1024 × 576 pixels. For

Table 1 Quantitative comparison. The results of automatic metrics demonstrate HiFi-Inpaint’s state-of-the-art performance. The best and second-best results are marked in bold and underlined.

|Method<br><br>|Text Alignment<br><br>|Visual Consistency|Generation Quality|
|---|---|---|---|
| |CLIP-T↑ (%)|CLIP-I↑ (%) DINO↑ (%) SSIM↑ (%) SSIM-HF↑ (%)<br><br>|LAION-Aes↑ Q-Align-IQ↑|
|Paint-by-Example [53] ACE++ [33] Insert Anything [44] FLUX-Kontext [7] HiFi-Inpaint (Ours)|31.6 34.9 35.3<br><br>36.6<br><br>36.1<br><br>|69.1 63.4 54.0 34.9<br><br>93.1 90.7 58.3 37.2<br><br>94.1 89.8 62.1 40.0 82.5 63.1 51.6 32.0<br><br>95.0 91.9 63.4 42.9<br><br><br>|4.09 4.06 4.18 4.00 4.20 3.89 4.54 3.74 4.40 4.36<br><br>|

comprehensive evaluation, we split 1,000 samples from HP-Image-40K to assess method performance. These samples were exclusively used for evaluation and were not included in the training process.

Compared Methods. We compare our method with four approaches capable of handling reference-based inpainting: (i) Paint-by-Example [53], which leverages CLIP feature representations to capture the appearance of the reference image and generates matching content in the target region; (ii) ACE++ [33], which is an instruction-based approach that integrates multi-modal inputs and employs a two-stage training scheme; (iii) Insert Anything [44], which is a framework using in-context editing and DiT for text-guided image insertion; and (iv) FLUX.1-Kontext-Dev (denoted as FLUX-Kontext) [7], which is an image editing model optimized for iterative, precise local and global edits. To ensure a fair comparison, all methods are evaluated at a fixed resolution of 1024 × 576 pixels and adhere to the same inference configurations and settings.

Evaluation Metrics. We evaluate the performance of each method from three perspectives: (i) Text Alignment: To assess how well generated images align with their text prompts, we compute the average CLIP-T score, which measures the similarity between each generated image and its corresponding text using the CLIP [36] model. (ii) Visual Consistency: We evaluate the visual consistency of generated images using multiple metrics. CLIP-I measures the similarity between generated images and reference images in the CLIP feature space, while DINO [11] provides another feature-based similarity assessment. Additionally, we use SSIM [48] to quantify structural similarity as we have access to the ground-truth human-product images in the test set. To specifically evaluate the ability of detail preservation, we introduce SSIM-HF, applying a high-pass filter to generated images before calculating SSIM. (iii) Image Quality: We assess the overall quality and aesthetics of the generated images using Q-Align-IQ [50] and LAION-Aes [23]. These metrics comprehensively evaluate both the technical and perceptual aspects of image quality, including sharpness, naturalness, and aesthetic appeal. For a more fine-grained evaluation, all metrics, except for CLIP-T, are calculated based on the generated content of the masked regions and the product reference images.

#### 4.2 Quantitative Comparison

We report comprehensive quantitative results across a variety of evaluation metrics in Tab. 1, showing HiFi-Inpaint achieves the overall state-of-the-art performance. In terms of text alignment, it achieves a competitive CLIP-T, indicating that the generated images are highly consistent with the input text prompts. For visual similarity, our method obtains the top CLIP-I (0.950) and DINO (0.919), demonstrating strong alignment with the product reference images in both global and local features. Structural similarity is also enhanced, as reflected by the highest SSIM (0.634) and SSIM-HF (0.429), further confirming HiFi-Inpaint’s ability to preserve fine-grained details. For aesthetic and image quality metrics, our method achieves the best or competitive results on LAION-Aes (4.40) and Q-Align-IQ (4.36), indicating that the outputs are not only faithful to the input but also visually pleasing and of high technical quality. In contrast, FLUX-Kontext exhibits relatively weak performance, with notably low CLIP-I (0.712) and DINO (0.631), reflecting poor textual and visual consistency. ACE++ and Insert Anything perform moderately well, with Insert Anything showing slightly better detail preservation (e.g., higher SSIM than ACE++), but both are consistently outperformed by our HiFi-Inpaint, especially in the metrics of visual consistency. Moreover, Paint-by-Example exhibits outdated performance of visual consistency in such a challenging scenario. In summary, these results highlight that our method delivers overall superior performance in terms of text alignment, visual consistency, and generation quality.

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

###### Ref. HiFi-Inpaint (Ours) ACE++ Insert Anything FLUX-Kontext Ref. HiFi-Inpaint (Ours) ACE++ Insert Anything FLUX-Kontext

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

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

| |
|---|

[Figure 148]

[Figure 149]

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

- Figure 5 Qualitative comparison. Compared to existing methods, our HiFi-Inpaint exhibits remarkable performance in generating high-quality human-product images, enabling high-fidelity preservation of product’s fine-grained details. Key differences are highlighted with red boxes to emphasize critical improvement areas. Zoom in for better view.

#### 4.3 Qualitative Comparison

The qualitative comparison, which illustrates the results of the four most competitive methods, is presented in Fig. 5. As we can observe, FLUX-Kontext often fails to perform successful inpainting, often generating a standalone product image instead. This is likely because general instruction-based editing is insufficient for the model to establish a meaningful relationship between the reference image and the masked region, leading to confusion regarding the visual elements in the inputs. Even in cases where object inpainting is successful, FLUX-Kontext still struggles to preserve fine details, resulting in noticeable inconsistencies in structure and texture. ACE++ demonstrates a better ability to associate the product with the masked region, effectively preserving the overall product shape and, to some extent, retaining text and patterns from the reference image. However, it still faces challenges in reconstructing fine-scale details such as small text or intricate logos. InsertAnything shows improved performance in detail preservation, maintaining more of the fine-grained details and patterns from the original product. Nevertheless, when the mask region is small, it tends to produce artifacts, which significantly degrade the quality of the generated results. In contrast,

- our HiFi-Inpaint achieves superior generation performance. HiFi-Inpaint is capable of generating realistic

Table 2 Quantitative ablation analysis. Each component of HiFi-Inpaint contributes to superior performance.

|Scheme<br><br>|Syn. Data DAL SEA|Text Alignment|Visual Consistency<br><br>|Generation Quality|
|---|---|---|---|---|
| | |CLIP-T↑ (%)<br><br>|CLIP-I↑ (%) DINO↑ (%) SSIM↑ (%) SSIM-HF↑ (%)<br><br>|LAION-Aes↑ Q-Align-IQ↑|
|A<br>B<br><br>C<br><br>D<br><br>E<br><br><br>|✓<br><br>✓ ✓<br><br>✓ ✓ ✓ ✓ ✓|35.4 35.8<br><br>36.2<br><br>35.9 36.1<br><br>|91.8 85.4 57.7 38.4<br><br>94.5 89.9 62.4 41.2<br><br>94.6 90.7 62.3 41.8<br><br><br>92.2 87.6 59.8 40.3 95.0 91.9 63.4 42.9<br><br><br>|4.29 4.40<br><br>4.32 4.23<br>4.33 4.28<br>4.34 4.47 4.40 4.36<br><br><br>|

and naturally composited images, with products seamlessly aligned to the background and their fine-grained details, including text, patterns, and branding elements, are faithfully preserved. Notably, HiFi-Inpaint is also able to maintain object shape and text details even when the mask region is small, further demonstrating its robustness for challenging scenarios.

#### 4.4 User Study

We conduct a user study to further assess overall generation performance from a human perspective. Specifically, we design a questionnaire with 11 groups of generated images, each paired with the corresponding text prompt, product reference image, and masked human image. Participants were asked to select the best result in each group based on three criteria: text alignment, visual consistency, and generation quality. A total of 31 valid responses were collected, and the averaged selection rates are presented in Tab. 3. The results reveal that our HiFi-Inpaint consistently outperforms existing methods across all three evaluation dimensions, providing better alignment with human preferences. Additionally, the findings from the user study are highly consistent with the quantitative evaluation of automatic metrics, further validating the effectiveness of our approach in delivering superior image generation quality.

Table 3 User study. The results on three criteria show that HiFi-Inpaint outperforms others in human preference.

|Method<br><br>|Text Align.↑ (%)|Visual Consis.↑ (%)<br><br>|Gen. Quality↑ (%)|
|---|---|---|---|
|ACE++ Insert Anything FLUX-Kontext HiFi-Inpaint (Ours)|20.3 24.9 18.4<br><br>36.4<br><br>|19.6 21.0 17.9<br><br>41.5|22.7 21.6 16.2<br><br>39.5|

#### 4.5 Ablation Analysis

To evaluate the effectiveness of each component, we conduct a systematic ablation study by disabling key components while keeping the overall model architecture and training settings. We report quantitative results for five ablation schemes in Tab. 2, where Scheme E denotes our final proposed method. Moreover, we highlight the key ablation comparisons by providing qualitative results in Fig. 6.

Ablation of Synthesized Data. We evaluate the improvements brought by the proposed HP-Image-40K dataset, as shown by the comparison of “Scheme A & Scheme B” and “Scheme D & Scheme E” in Tab. 2. As we can see, under otherwise identical conditions, incorporating our synthetic dataset for model training leads to significant gains in text alignment and visual consistency.

Ablation of Shared Enhancement Attention. We further assess the impact of Shared Enhancement Attention (SEA) by comparing “Scheme C & Scheme E. As illustrated in Fig. 6, the introduction of SEA enables more precise alignment of intricate details and patterns within the generated images. Furthermore, the quantitative results presented in Tab. 2 show its consistent improvements across multiple metrics, highlighting its comprehensive benefits.

Ablation of Detail-Aware Loss. We assess the contribution of Detail-Aware Loss (DAL) by comparing “Scheme B & Scheme C” in Tab. 2, showing that DAL effectively improves overall performance. As shown in Fig. 6, the model trained without DAL fails to preserve fine-grained elements such as text and intricate patterns, resulting in blurry or semantically incomplete product renderings. These results confirm that DAL is crucial for enhancing the model’s ability to retain critical visual details of products.

###### Ref. HiFi-Inpaint (Ours) w/o SEA w/o SEA & DAL Ref. HiFi-Inpaint (Ours) w/o SEA w/o SEA & DAL

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

- Figure 6 Qualitative ablation analysis. The results demonstrate the effectiveness of both Shared Enhancement Attention (SEA) and Detail-Aware Loss (DAL) in improving the quality of generated human-product images. Our HiFi-Inpaint, integrating these techniques, achieves the best overall performance with superior detail preservation. Zoom in for better view.

### 5 Conclusions

In this paper, we explore the paradigm of reference-based inpainting for generating high-quality human-product images. We introduce HiFi-Inpaint, a novel framework that leverages Shared Enhancement Attention (SEA) to capture fine-grained product features and Detail-Aware Loss (DAL) to enable precise pixel-level supervision. To facilitate training, we present HP-Image-40K, a high-quality dataset specifically designed for this task. Experiments demonstrate that our approach achieves superior performance, effectively preserving intricate product details while generating visually coherent images. Future work will focus on enhancing the diversity and realism of the generated images and extending our method to video generation.

### Acknowledgments

This work was supported in part by the NSFC under Grant No. 62472403, in part by the 2035 Key Research and Development Program of Ningbo City under Grant No. 2024Z123, and in part by the InnoHK initiative of the Innovation and Technology Commission of the Hong Kong Special Administrative Region Government via the Hong Kong Centre for Logistics Robotics. We thank Hao Yang and Ruibiao Lu for their helpful discussions and technical support.

### References

- [1] Omri Avrahami, Ohad Fried, and Dani Lischinski. Blended latent diffusion. ACM transactions on graphics (TOG), 42(4):1–11, 2023.
- [2] Jinbin Bai, Zhen Dong, Aosong Feng, Xiao Zhang, Tian Ye, and Kaicheng Zhou. Integrating view conditions for image synthesis. arXiv preprint arXiv:2310.16002, 2023.
- [3] Jinbin Bai, Wei Chow, Ling Yang, Xiangtai Li, Juncheng Li, Hanwang Zhang, and Shuicheng Yan. Humanedit: A high-quality human-rewarded dataset for instruction-based image editing. arXiv preprint arXiv:2412.04280, 2024.
- [4] Jinbin Bai, Tian Ye, Wei Chow, Enxin Song, Qing-Guo Chen, Xiangtai Li, Zhen Dong, Lei Zhu, and Shuicheng Yan. Meissonic: Revitalizing masked generative transformers for efficient high-resolution text-to-image synthesis. In The Thirteenth International Conference on Learning Representations, 2024.
- [5] Marcelo Bertalmio, Guillermo Sapiro, Vincent Caselles, and Coloma Ballester. Image inpainting. In Proceedings of the 27th annual conference on Computer graphics and interactive techniques, pages 417–424, 2000.
- [6] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.
- [7] Black Forest Labs. Flux. https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev, 2025.
- [8] Shengqu Cai, Eric Ryan Chan, Yunzhi Zhang, Leonidas Guibas, Jiajun Wu, and Gordon Wetzstein. Diffusion self-distillation for zero-shot customized image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18434–18443, 2025.
- [9] Ke Cao, Jing Wang, Ao Ma, Jiasong Feng, Zhanjie Zhang, Xuanhua He, Shanyuan Liu, Bo Cheng, Dawei Leng, Yuhui Yin, et al. Relactrl: Relevance-guided efficient control for diffusion transformers. arXiv preprint arXiv:2502.14377, 2025.
- [10] Pu Cao, Feng Zhou, Qing Song, and Lu Yang. Controllable generation with text-to-image diffusion models: A survey. arXiv preprint arXiv:2403.04279, 2024.
- [11] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.
- [12] Sixiang Chen, Jinbin Bai, Zhuoran Zhao, Tian Ye, Qingyu Shi, Donghao Zhou, Wenhao Chai, Xin Lin, Jianzong Wu, Chao Tang, et al. An empirical study of gpt-4o image generation capabilities. arXiv preprint arXiv:2504.05979, 2025.
- [13] SiXiang Chen, Jianyu Lai, Jialin Gao, Tian Ye, Haoyu Chen, Hengyu Shi, Shitong Shao, Yunlong Lin, Song Fei, Zhaohu Xing, et al. Postercraft: Rethinking high-quality aesthetic poster generation in a unified framework. arXiv preprint arXiv:2506.10741, 2025.
- [14] Sixiang Chen, Jianyu Lai, Jialin Gao, Hengyu Shi, Zhongying Liu, Tian Ye, Junfeng Luo, Xiaoming Wei, and Lei Zhu. Posteromni: Generalized artistic poster creation via task distillation and unified reward feedback. arXiv preprint arXiv:2602.12127, 2026.
- [15] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198, 2024.
- [16] Antonio Criminisi, Patrick Pérez, and Kentaro Toyama. Region filling and object removal by exemplar-based image inpainting. IEEE Transactions on image processing, 13(9):1200–1212, 2004.
- [17] Per-Erik Danielsson and Olle Seger. Generalized and separable sobel operators. In Machine vision for threedimensional scenes, pages 347–379. Elsevier, 1990.
- [18] Runze He, Bo Cheng, Yuhang Ma, Qingxiang Jia, Shanyuan Liu, Ao Ma, Xiaoyu Wu, Liebucha Wu, Dawei Leng, and Yuhui Yin. Plangen: Towards unified layout planning and image generation in auto-regressive vision language models, 2025. URL https://arxiv.org/abs/2503.10127.
- [19] Runze He, Yiji Cheng, Tiankai Hang, Zhimin Li, Yu Xu, Zijin Yin, Shiyi Zhang, Wenxun Dai, Penghui Du, Ao Ma, et al. Re-align: Structured reasoning-guided alignment for in-context image generation and editing. arXiv preprint arXiv:2601.05124, 2026.

- [20] Jiancheng Huang, Yi Huang, Jianzhuang Liu, Donghao Zhou, Yifan Liu, and Shifeng Chen. Dual-schedule inversion: Training-and tuning-free inversion for real image editing. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 660–669. IEEE, 2025.
- [21] Yi Huang, Jiancheng Huang, Yifan Liu, Mingfu Yan, Jiaxi Lv, Jianzhuang Liu, Wei Xiong, He Zhang, Liangliang Cao, and Shifeng Chen. Diffusion model-based image editing: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.
- [22] Bernd Jähne. Digital image processing. Springer, 2005.
- [23] LAION-AI. Laion-aesthetics-predictor v1, 2022. URL https://github.com/LAION-AI/aesthetic-predictor.
- [24] Yaowei Li, Xiaoyu Li, Zhaoyang Zhang, Yuxuan Bian, Gan Liu, Xinyuan Li, Jiale Xu, Wenbo Hu, Yating Liu, Lingen Li, et al. Ic-custom: Diverse image customization via in-context learning. arXiv preprint arXiv:2507.01926, 2025.
- [25] Jingyu Lin, Yongrong Wu, Zeyu Wang, Xiaode Liu, and Yufei Guo. Pair-id: A dual modal framework for identity preserving image generation. IEEE Signal Processing Letters, 2024.
- [26] Jingyu Lin, Guiqin Zhao, Jing Xu, Guoli Wang, Zejin Wang, Antitza Dantcheva, Lan Du, and Cunjian Chen. Difftv: Identity-preserved thermal-to-visible face translation via feature alignment and dual-stage conditions. In ACM Multimedia 2024, 2024.
- [27] Yunlong Lin, Zixu Lin, Kunjie Lin, Jinbin Bai, Panwang Pan, Chenxin Li, Haoyu Chen, Zhongdao Wang, Xinghao Ding, Wenbo Li, et al. Jarvisart: Liberating human artistic creativity via an intelligent photo retouching agent. arXiv preprint arXiv:2506.17612, 2025.
- [28] Yunlong Lin, Linqing Wang, Kunjie Lin, Zixu Lin, Kaixiong Gong, Wenbo Li, Bin Lin, Zhenxi Li, Shiyi Zhang, Yuyang Peng, et al. Jarvisevo: Towards a self-evolving photo editing agent with synergistic editor-evaluator optimization. arXiv preprint arXiv:2511.23002, 2025.
- [29] Run Ling, Ke Cao, Jian Lu, Ao Ma, Haowei Liu, Runze He, Changwei Wang, Rongtao Xu, Yihua Shao, Zhanjie Zhang, et al. Mofu: Scale-aware modulation and fourier fusion for multi-subject video generation. arXiv preprint arXiv:2512.22310, 2025.
- [30] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.
- [31] Ao Ma, Jiasong Feng, Ke Cao, Jing Wang, Yun Wang, Quanwei Zhang, and Zhanjie Zhang. Lay2story: extending diffusion transformers for layout-togglable story generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16102–16111, 2025.
- [32] Hayk Manukyan, Andranik Sargsyan, Barsegh Atanyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Hd-painter: high-resolution and prompt-faithful text-guided image inpainting with diffusion models. In The Thirteenth International Conference on Learning Representations, 2023.
- [33] Chaojie Mao, Jingfeng Zhang, Yulin Pan, Zeyinzi Jiang, Zhen Han, Yu Liu, and Jingren Zhou. Ace++: Instruction-based image creation and editing via context-aware content filling. arXiv preprint arXiv:2501.02487, 2025.
- [34] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.
- [35] Tingting Qiao, Jing Zhang, Duanqing Xu, and Dacheng Tao. Mirrorgan: Learning text-to-image generation by redescription. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1505–1514, 2019.
- [36] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
- [37] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International conference on machine learning, pages 8821–8831. Pmlr, 2021.

- [38] Scott Reed, Zeynep Akata, Xinchen Yan, Lajanugen Logeswaran, Bernt Schiele, and Honglak Lee. Generative adversarial text to image synthesis. In International conference on machine learning, pages 1060–1069. PMLR, 2016.
- [39] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022.
- [40] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500–22510, 2023.
- [41] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad Norouzi. Palette: Image-to-image diffusion models. In ACM SIGGRAPH 2022 conference proceedings, pages 1–10, 2022.
- [42] Guibao Shen, Luozhou Wang, Jiantao Lin, Wenhang Ge, Chaozhe Zhang, Xin Tao, Yuan Zhang, Pengfei Wan, Zhongyuan Wang, Guangyong Chen, et al. Sg-adapter: Enhancing text-to-image generation with scene graph guidance. arXiv preprint arXiv:2405.15321, 2024.
- [43] Quanjian Song, Donghao Zhou, Jingyu Lin, Fei Shen, Jiaze Wang, Xiaowei Hu, Cunjian Chen, and Pheng-Ann Heng. Scenedecorator: Towards scene-oriented story generation with scene planning and scene consistency. arXiv preprint arXiv:2510.22994, 2025.
- [44] Wensong Song, Hong Jiang, Zongxing Yang, Ruijie Quan, and Yi Yang. Insert anything: Image insertion via in-context editing in dit. arXiv preprint arXiv:2504.15009, 2025.
- [45] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. arXiv preprint arXiv:2411.15098, 2024.
- [46] ultralytics. Yolov8, 2023. URL https://github.com/ultralytics/ultralytics.
- [47] Zeyu Wang, Jingyu Lin, Yifei Qian, Yi Huang, Shicen Tian, Bosong Chai, Juncan Deng, Lan Du, Cunjian Chen, Yufei Guo, et al. Diffx: Guide your layout to cross-modal generative modeling. arXiv preprint arXiv:2407.15488, 2024.
- [48] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.
- [49] Yuxiang Wei, Yiheng Zheng, Yabo Zhang, Ming Liu, Zhilong Ji, Lei Zhang, and Wangmeng Zuo. Personalized image generation with deep generative models: A decade survey. arXiv preprint arXiv:2502.13081, 2025.
- [50] Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Yixuan Gao, Annan Wang, Erli Zhang, Wenxiu Sun, et al. Q-align: Teaching lmms for visual scoring via discrete text-defined levels. arXiv preprint arXiv:2312.17090, 2023.
- [51] Shaoan Xie, Zhifei Zhang, Zhe Lin, Tobias Hinz, and Kun Zhang. Smartbrush: Text and shape guided object inpainting with diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22428–22437, 2023.
- [52] Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, and Xiaodong He. Attngan: Fine-grained text to image generation with attentional generative adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1316–1324, 2018.
- [53] Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. Paint by example: Exemplar-based image editing with diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18381–18391, 2023.
- [54] Shiyuan Yang, Xiaodong Chen, and Jing Liao. Uni-paint: A unified framework for multimodal image inpainting with pretrained diffusion model. In Proceedings of the 31st ACM International Conference on Multimedia, pages 3190–3199, 2023.
- [55] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022.

- [56] Sicheng Zhang, Binzhu Xie, Zhonghao Yan, Yuli Zhang, Donghao Zhou, Xiaofei Chen, Shi Qiu, Jiaqi Liu, Guoyang Xie, and Zhichao Lu. Trade-offs in image generation: How do different dimensions interact? arXiv preprint arXiv:2507.22100, 2025.
- [57] Zhanjie Zhang, Quanwei Zhang, Wei Xing, Guangyuan Li, Lei Zhao, Jiakai Sun, Zehua Lan, Junsheng Luan, Yiling Huang, and Huaizhong Lin. Artbank: Artistic style transfer with pre-trained diffusion model and implicit style prompt bank. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pages 7396–7404,

- 2024.

[58] Zhanjie Zhang, Ao Ma, Ke Cao, Jing Wang, Shanyuan Liu, Yuhang Ma, Bo Cheng, Dawei Leng, and Yuhui Yin. U-stydit: Ultra-high quality artistic style transfer using diffusion transformers. arXiv preprint arXiv:2503.08157,

- 2025.

- [59] Ziyue Zhang, Quanjian Song, Yuxin Zhang, and Rongrong Ji. Objectadd: adding objects into image via a training-free diffusion modification fashion. arXiv preprint arXiv:2404.17230, 2024.
- [60] Donghao Zhou, Jiancheng Huang, Jinbin Bai, Jiaze Wang, Hao Chen, Guangyong Chen, Xiaowei Hu, and Pheng-Ann Heng. Magictailor: Component-controllable personalization in text-to-image diffusion models. arXiv preprint arXiv:2410.13370, 2024.
- [61] Donghao Zhou, Jingyu Lin, Guibao Shen, Quande Liu, Jialin Gao, Lihao Liu, Lan Du, Cunjian Chen, Chi-Wing Fu, Xiaowei Hu, et al. Identitystory: Taming your identity-preserving generator for human-centric story generation. arXiv preprint arXiv:2512.23519, 2025.

### A HP-Image-40K Dataset Statistics

To better demonstrate the diversity and comprehensiveness of the HP-Image-40K dataset, we provide detailed statistics in terms of mask area ratio and product categories. These statistics highlight the broad coverage of the dataset, making it a valuable resource for training robust and generalizable models across various real-world scenarios for generating high-quality human-product images.

Mask Area Ratio. The HP-Image-40K dataset includes a wide range of mask area ratios, as shown in Fig. 7. The mask area ratio, defined as the proportion of the mask area to the total image area, varies significantly across the dataset. This variation ensures that the dataset covers diverse object sizes and spatial distributions, from small, localized objects to large, prominent ones. Such mask area ratio diversity is critical for training models that can handle objects of different scales and spatial contexts, improving their performance across various application scenarios.

Product Categories. The HP-Image-40K dataset also features a rich variety of product categories, as visualized in the word cloud in Fig. 8. These categories include bottles, containers, jars, tubes, and dispensers, among others. This diversity not only reflects the dataset’s real-world applicability but also enriches the feature representation for model training. By exposing models to a broad spectrum of shapes, materials, and structural characteristics, the dataset enhances the model’s ability to generalize across multiple domains and adapt to different product types.

14000

12000

10000

Count

8000

6000

4000

2000

0

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7

Mask Area Ratio

Figure 7 Histogram of the mask area ratio in HPImage-40K. Our HP-Image-40K dataset exhibits a diverse range of mask area ratios, effectively covering various real-world scenarios.

[Figure 210]

Figure 8 Word cloud of the product categories in HP-Image-40K. The dataset encompasses a wide variety of product categories, providing the model with a wide range of shapes, materials, and structures for training.

### B Information of Internal Real-World Dataset

In addition to the synthetic HP-Image-40K dataset, we further construct a internal real-world dataset collected from publicly available internet images to evaluate model generalization under more realistic conditions. After preprocessing, the real-world dataset is aligned with the synthetic dataset in terms of image resolution and aspect ratio to ensure a fair comparison protocol.

Compared to the synthetic data, the real-world dataset exhibits substantially higher diversity and complexity. It contains a wide range of scenes (indoor and outdoor environments), diverse human subjects with varying poses and interactions, and products with more complex appearances, materials, textures, and branding details. The visual conditions also vary significantly in lighting, viewpoint, occlusion, and background clutter, making the learning problem more challenging than in the controlled synthetic setting.

For training, we utilize approximately 14,000 preprocessed real-world samples. For evaluation, we curate a separate test set consisting of 2,000 preprocessed real-world samples that are not used during training. This split enables a comprehensive assessment of the model’s robustness and generalization capability in complex real-world scenarios.

Table 4 Quantitative comparison on real-world data. The results of automatic metrics demonstrate HiFi-Inpaint’s overall state-of-the-art performance. The best and second-best results are marked in bold and underlined.

|Method<br><br>|Text Alignment<br><br>|Visual Consistency|Generation Quality|
|---|---|---|---|
| |CLIP-T↑ (%)|CLIP-I↑ (%) DINO↑ (%) SSIM↑ (%) SSIM-HF↑ (%)|LAION-Aes↑ Q-Align-IQ↑|
|Paint-by-Example [53] ACE++ [33] Insert Anything [44] FLUX-Kontext [6] HiFi-Inpaint (Ours)|27.1 28.2 28.9 29.0<br><br>29.7<br><br>|56.2 24.3 50.8 35.7 80.1 74.2 53.5 36.6 83.1 77.5 55.1 37.8 59.9 55.7 44.6 34.3<br><br>86.8 79.8 60.5 44.1|4.34 2.23<br><br>3.90 3.47<br><br>3.95 3.48<br><br>4.30 2.91<br><br><br>4.27 3.29<br>|

### C More Details of Setups

Training Configuration. We set the LoRA scaling factor α to 256, which is equal to the rank. Although our model supports reference images of arbitrary resolutions, we adopt a padding-then-resizing strategy for both training and evaluation to ensure consistent spatial alignment. Specifically, when a reference image does not match the target resolution of 1024×576, we first pad it to the target aspect ratio while preserving its original content, and then resize it to the fixed resolution. This strategy avoids geometric distortion and maintains structural integrity of the product appearance.

Baseline Adaptation. For fair comparison, all baselines are evaluated under the same input resolution (1024 × 576) and identical masked regions. Paint-by-Example, ACE++ and Insert Anything natively support reference-based inpainting with multi-image inputs. Therefore, we directly follow their official inference protocols and input formatting without additional modification or prompt engineering. FLUX-Kontext is an instruction-based image editing model that does not support explicit multi-image conditioning. To adapt it to our task, we concatenate the product reference image and the masked human image along the width dimension to form a single composite input. Following the official inpainting usage guidelines, we adopt the instruction prompt: “Change the object in the black square to the product in the left image.” This enables the model to interpret the left region as the reference product and perform object replacement within the masked area accordingly.

### D Evaluation on Real-World Data

In the main paper, experiments were conducted on synthetic data due to their well-controlled alignment with our task definition. To further verify the generalizability of the models, we additionally evaluate HiFi-Inpaint and other baselines on an internal real-world test set containing 2,000 diverse human–product samples, which presents significantly more variation in lighting conditions, pose configurations, and product appearance.

#### D.1 Quantitative Comparison

Tab. 4 reports quantitative comparisons across all metrics, showing that HiFi-Inpaint remains highly competitive under the more challenging real-world setting. For text alignment, HiFi-Inpaint achieves the best CLIP-T, indicating that the generated content generally stays faithful to textual instructions even when scenes exhibit greater complexity. In terms of visual similarity, our model attains the highest CLIP-I (86.8) and DINO (79.8), suggesting that HiFi-Inpaint can better preserve both global product identity and local appearance details when the alignment between reference and target is less constrained than in synthetic cases. Structural similarity follows a similar trend: HiFi-Inpaint obtains the top SSIM (60.5) and SSIM-HF (44.1), reflecting strong preservation of object structure and high-frequency characteristics such as text, logos, and fine patterns. For aesthetic and perceptual quality, HiFi-Inpaint delivers competitive results, ranking third on both LAION-Aes (4.27) and Q-Align-IQ (3.29). Although these scores are slightly lower than the best-performing baselines, they still indicate visually appealing outputs that remain technically coherent with the reference products. By comparison, FLUX-Kontext exhibits lower CLIP-I (59.9) and DINO (55.7), suggesting that it has more difficulty grounding the reference product under real-world conditions. ACE++ and Insert Anything achieve moderate to strong performance, with Insert Anything showing relatively good structural and detail preservation, yet both are still outperformed by HiFi-Inpaint on most visual consistency metrics.

###### Ref. HiFi-Inpaint (Ours) ACE++ Insert Anything FLUX-Kontext Ref. HiFi-Inpaint (Ours) ACE++ Insert Anything FLUX-Kontext

[Figure 211]

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

- Figure 9 Qualitative comparison on real-world data. Compared to existing methods, our HiFi-Inpaint exhibits remarkable performance in generating high-quality human-product images, enabling high-fidelity preservation of fine-grained details. The eyes have been obscured to protect the identity of real humans. Zoom in for better view.

Paint-by-Example lags behind recent methods in this challenging setup, especially on visual consistency and structural similarity. Overall, the real-world evaluation suggests that HiFi-Inpaint generalizes well beyond synthetic data and remains robust in preserving product fidelity under realistic variations, while maintaining competitive aesthetic and perceptual quality.

#### D.2 Qualitative Comparison

Qualitative comparisons of the four strongest competing methods are provided in Fig. 9. FLUX-Kontext frequently fails to perform correct inpainting, often generating an isolated product instead of integrating it into the masked region. This suggests that generic instruction-based editing offers limited capability for grounding the reference product within complex inputs. Even when successful, FLUX-Kontext tends to lose high-frequency details, leading to noticeable inconsistencies in structure and texture. ACE++ shows a stronger ability to associate the product with the masked region, preserving overall shape and partially retaining textual or patterned elements. However, fine-scale details such as small characters or intricate logos are often not accurately reconstructed. Insert Anything performs better in detail preservation but tends to introduce artifacts when the masked region becomes smaller, degrading realism and compositional quality. In contrast, HiFi-Inpaint produces clean, realistic, and naturally composited results. The model faithfully preserves product appearance, including text, patterns, and branding elements, while aligning the inpainted region seamlessly with the surrounding context. Importantly, HiFi-Inpaint remains robust even when the mask is small, maintaining structural integrity and fine-grained details without introducing noticeable artifacts.

### E Additional Results of Ablation Analysis

To further validate the effectiveness of key components in HiFi-Inpaint, we conduct a systematic ablation study, with additional qualitative results shown in Fig. 10. The examples show that removing individual components leads to noticeable degradation in the detail preservation performance. In contrast, the complete HiFi-Inpaint consistently produces superior results, faithfully preserving critical details and achieving seamless

###### Ref. HiFi-Inpaint (Ours) w/o SEA w/o SEA & DAL Ref. HiFi-Inpaint (Ours) w/o SEA w/o SEA & DAL

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

- Figure 10 Qualitative ablation analysis. The results demonstrate the effectiveness of both Shared Enhancement Attention (SEA) and Detail-Aware Loss (DAL) in improving the quality of generated human-product images. Our HiFi-Inpaint, integrating these techniques, achieves the best overall performance with superior detail preservation. Zoom in for better view.

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

Ref. Generated Image Ref. Generated Image Ref. Generated Image Ref. Generated Image Ref. Generated Image

w/o Human (Outdoor) w/o Human (Indoor) Full Body View Product interference Style Transfer

- Figure 11 Generalizability analysis of HiFi-Inpaint. We further evaluate our HiFi-Inpaint on several hard cases, demonstrating its potential to generalize to a broader range of scenarios. Zoom in for better view.

integration with the background. These results further highlight the contributions of our proposed techniques in tackling challenging inpainting tasks.

### F Generalizability Analysis

In Fig. 11, HiFi-Inpaint is further evaluated on a collection of challenging real-world cases designed to examine the model’s behavior beyond standard inpainting conditions. These examples cover a wide span of difficult scenarios, including images without humans in both outdoor and indoor environments, full-body human views with large pose variations, situations with product interference in the masked image, and cases requiring substantial style adaptation. As illustrated in Fig. 11, HiFi-Inpaint consistently produces coherent and context-aware completions across these heterogeneous inputs. Even when object scale, lighting conditions, or style distributions deviate significantly from the training distribution, the model is able to integrate the target product naturally into the scene while preserving its key visual attributes. Although certain extreme cases still reveal room for improvement, these results highlight the model’s potential to generalize toward a broader range of practical applications and more diverse deployment conditions.

