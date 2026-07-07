# TryOffDiff: Virtual-Try-Off via High-Fidelity Garment Reconstruction using Diffusion Models

Riza Velioglu1

- 1Machine Learning Group, CITEC Bielefeld University Germany
- 2Institute of Mathematics Technical University Berlin Germany

rvelioglu+bmvc@techfak.de

arXiv:2411.18350v2[cs.CV]7Aug2025

Petra Bevandic´1

pbevandic+bmvc@techfak.de

Robin Chan2

chan+bmvc@tu-berlin.de

Barbara Hammer1

bhammer+bmvc@techfak.de

[Figure 1]

Figure 1: Virtual Try-Off results from our method. Each row shows the input, prediction, and ground truth for samples from VITON-HD (cols 1-6) and Dress Code (cols 7-12). TryOffDiff synthesizes garments on a clean background in a standard pose, accurately preserving fine details like patterns and logos from a single reference image.

##### Abstract

This paper introduces Virtual Try-Off (VTOFF), a novel task generating standardized garment images from single photos of clothed individuals. Unlike Virtual TryOn (VTON), which digitally dresses models, VTOFF extracts canonical garment images, demanding precise reconstruction of shape, texture, and complex patterns, enabling robust evaluation of generative model fidelity. We propose TryOffDiff, adapting Stable Diffusion with SigLIP-based visual conditioning to deliver high-fidelity reconstructions. Experiments on VITON-HD and Dress Code datasets show that TryOffDiff outperforms adapted pose transfer and VTON baselines. We observe that traditional metrics such as SSIM inadequately reflect reconstruction quality, prompting our use of DISTS for reliable assessment. Our findings highlight VTOFF’s potential to improve e-commerce product imagery, advance generative model evaluation, and guide future research on high-fidelity reconstruction. Demo, code, and models are available at: https://rizavelioglu.github.io/tryoffdiff

© 2025. The copyright of this document resides with its authors. It may be distributed unchanged freely in print or electronic forms.

## 1 Introduction

High-quality, standardized product images are vital in e-commerce, particularly in fashion, where visuals drive consumer purchasing decisions [50, 56]. Producing such images is resource-intensive, requiring specialized equipment and extensive post-processing. While existing approaches like Virtual Try-On (VTON) [24] focus on enhancing customer experience by synthesizing images of people wearing selected garments—using a garment image and a reference person image—they do not address the creation of the garment images themselves. VTON advances human-centric generation, involving pose transfer, garment warping, and texture preservation[18], but assumes access to clean product shots. To address this gap, we introduce Virtual Try-Off (VTOFF): the task of extracting standardized garment images from photos of clothed individuals. To date, this inverse task has received little attention and has yet to be formally defined.

VTOFF addresses critical needs across multiple domains. In e-commerce, it offers a practical solution for generating catalog-ready garment images from diverse real-world photos. This reduces reliance on costly photography setups and manual editing, empowering smaller vendors to access professional-quality visuals. It supports customer-to-product retrieval by providing consistent garment images for recommendation systems [15]. Environmentally, VTOFF enhances visual clarity of garment details, aligning customer expectations with product appearance and potentially reducing product returns, which are a major contributor to the fashion industry’s environmental footprint [13, 20]. From a research perspective, VTOFF presents several complex challenges that advance the field of high-fidelity garment reconstruction. These include handling occlusions, deformations, and varying poses, as well as managing geometric and appearance transformations while preserving fine-grained details such as textures, patterns, and logos. The diversity of realworld photos, which vary in background, lighting, and camera quality, further introduces unique hurdles in domain adaptation and robust feature extraction, making VTOFF a compelling area for innovation. Moreover, VTON and VTOFF exist in a complementary relationship, as illustrated in Fig. 2. This cyclic connection enables: (1) improved training through consistency constraints in the loss function [54], (2) generation of synthetic data for both tasks [9, 40]. This synergy boosts model robustness and opens new avenues for fashion image generation.

[Figure 2]

Figure 2: VTON vs. VTOFF. Left: Virtual Try-On generates a dressed person from a garment image and a masked reference. Right: Virtual Try-Off reconstructs a canonical garment form from a photo of a clothed person. The two tasks form a cycle, where one’s output can serve as other’s input.

A key advantage of VTOFF over VTON is its evaluation clarity. VTON suffers from ill-defined target outputs, which complicates quality assessment. Generated images often exhibit stylistic variations, such as garments appearing tucked, untucked, or altered in fit (see Fig. 2), introducing plausible yet inconsistent results that obscure true garment fidelity [48]. Existing evaluation metrics, primarily designed for broad generative quality [22], are sensitive to irrelevant regions like backgrounds and fail to focus on garment-specific details,

making it hard to identify performance issues [11, 46]. In contrast, VTOFF provides a clear and precise target: the standardized garment image. This well-defined output enables more accurate assessment of reconstruction quality, particularly in capturing shape, texture, and patterns, making VTOFF an ideal task for evaluating generative models in fashion contexts. By bridging e-commerce solutions and research opportunities, VTOFF advances fashion technology. This paper lays the groundwork for its exploration. Our main contributions are:

- • We introduce VTOFF, a novel task to generate standardized garment images from photos of clothed individuals, addressing e-commerce needs and research challenges.
- • We leverage VTOFF’s well-defined targets to expose the limitations of conventional metrics and validate DISTS as a more reliable alternative.
- • We propose TryOffDiff, a framework that adapts pretrained diffusion models with pretrained image encoders for high-fidelity garment reconstruction.
- • Experiments on the VITON-HD and Dress Code datasets show TryOffDiff outperforms adapted baselines with minimal pre- and post-processing.

## 2 Related Work

This section reviews prior research relevant to the proposed task and method, covering garment reconstruction, image-based virtual try-on, view synthesis, pose transfer, and conditional diffusion models.

Garment Generation is central to VTOFF. Early work, such as TileGAN [60], tackled VTOFF using a two-stage GAN-based approach. While innovative for its time, TileGAN did not formally define VTOFF as a standalone problem, and its reliance on GANs has been eclipsed by diffusion models. Limited evaluation, restricted datasets, and unavailable code likely curbed its adoption, highlighting the need for a more comprehensive framework. Later efforts in garment synthesis deviate from VTOFF’s image-only focus. ARMANI [63] and DressCode [21] emphasize text-to-garment generation, producing garments (including 3D models in the latter) from textual descriptions. Similarly, SGDiff [44] and DiffCloth [64] leverage diffusion models—finetuning GLIDE, and Stable Diffusion, respectively—to synthesize garments, but they rely on multi-modal inputs like text combined with style images or sketches. These approaches, while advancing synthesis, sidestep VTOFF’s core challenge of reconstructing garments from real-world photos alone. Closer to VTOFF, FLDMVTON [54] incorporates a VTOFF-like component in its loss function to improve VTON training. However, it does not address VTOFF explicitly, limiting its relevance to our problem definition. As a result, prior work has overlooked the specific challenges of VTOFF, including occlusions, garment deformations, and diverse real-world photo conditions. To the best of our knowledge, we are the first to formally define VTOFF, explore its potential, and propose appropriate evaluation protocols.

Image-based Virtual Try-On aims to generate realistic images of a person wearing a specified garment, preserving their identity, pose, and body shape while capturing fine garment details. Early efforts, such as CAGAN[24] introduced the task using a cycle-GAN approach, while VITON[18] formalized it as a two-step supervised framework: warping the garment and blending it onto the person. CP-VTON[53] refined warping with a learnable thin-plate spline transformation, later enhanced by dense flow[19] and appearance flow [16]

for better pixel-level alignment. Despite progress, challenges with complex textures remain. Recent developments have shifted toward GAN-based and diffusion-based methods. FWGAN[12] and PASTA-GAN[57] leveraged GANs, but their instability prompted a move to diffusion models, such as IDM-VTON[8], OOTDiffusion[58], and CatVTON[9], which provide greater stability and detail retention. Notably, VTON and VTOFF differ fundamentally despite both involving garment manipulation. VTON models work with complete garment details, focusing on warping them to fit a target pose. Conversely, VTOFF models rely on partial, often occluded garment views from real-world images, requiring them to infer and reconstruct missing details from limited visual information.

Image-based Novel View Synthesis & Pose Transfer. NVS aims to generate realistic images from unseen viewpoints. Early approaches demanded extensive image sets per instance [28, 43, 47, 65, 66], whereas recent methods achieve synthesis from sparse views [23, 49]. However, NVS lacks the ability to handle pose variations, limiting its use in garment reconstruction. Pose transfer extends view synthesis by enabling pose alterations and inferring occluded regions. DiOr [10] employs a recurrent framework to sequentially dress subjects, while [36] leverages a GAN-based approach for enhanced pose transfer. Advanced methods like DreamPose [25] and PoCoLD [17] utilize diffusion models for poseguided image and video synthesis. ViscoNet [6] and PCDM [39] further improve control and detail fidelity in pose-driven synthesis. Unlike pose transfer, which preserves scene attributes like lighting, background and subject appearance, VTOFF demands specific standards, such as consistent views, uniform sizing, and e-commerce catalog styling.

Conditional Diffusion Models. Latent Diffusion Models [35] (LDMs) excel in synthesis, leveraging cross-attention [51] for precise conditioning on text [2, 3, 14] or images [33, 37, 38]. Text-guided extensions like ControlNet [62] and T2I-Adapter [31] add spatial accuracy, while IP-Adapter [59] separates text and image features for flexibility. Despite the advancements, applying these models directly to garment reconstruction poses significant challenges: text-guided approaches demand impractically detailed prompts for each sample to specify product attributes, while existing image-guided models lack specialized mechanisms needed to meet the strict requirements of standardized product photography, such as precise alignment, and detail preservation.

## 3 Methodology

We formalize the virtual try-off task, propose an evaluation framework with suitable metrics, and detail the TryOffDiff model architecture.

### 3.1 Virtual Try-Off

Problem Formulation. Let I ∈ RH×W×3 be an RGB image of a clothed person, with height H ∈ N and width W ∈ N. VTOFF aims to generate a standardized garment image G ∈ {0,...,255}H×W×3 from I, adhering to commercial catalog standards. Formally, we seek to train a generative model to approximate the conditional distribution P(G|C), where G and C denote garment images and reference images (serving as condition), respectively. Given a reference image I, the model should produce a sample Gˆ ∼ Q(G|C = I) closely matching a ground-truth garment image G ∼ P(G|C = I).

Datasets. VTOFF does not require new data collection. It leverages existing VTON datasets such as VITON-HD [7] and Dress Code [30]. These datasets provide input-target pairs (I,G)

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

(a) 82.4 / 20.6 (b) 96.8 / 17.9 (c) 88.3 / 20.3 (d) 86.0 / 70.3 (e) 75.0 / 8.2 (f) 86.4 / 24.7

- Figure 3: Metric suitability for VTON and VTOFF (SSIM↑ / DISTS↓). A reference image is compared to: (a) masked-out garment, (b) hue-jittered, (c) patch-wise color-jittered image; and a garment image is compared to: (d) plain white, (e) slightly rotated, (f) randomly posterized image. SSIM fails to penalize distortions, while DISTS better reflects judgment.

for training and evaluation, where I is an image of a clothed person (half-body or full-body) and G is the corresponding standardized garment image.

Performance Measures. Effective evaluation requires metrics that assess both reconstruction quality (pixel-level fidelity between Gˆ and G) and perceptual quality (visual realism per human judgment). Full-reference metrics like Structural Similarity Index Measure (SSIM) [55] and its multi-scale (MS-SSIM) and complex-wavelet (CW-SSIM) variants measure reconstruction but poorly align with human perception, as shown in prior studies [11, 46] and our own experiments (Fig. 3). For instance, Fig. 3d compares a garment image to a plain white image and still yields a high SSIM score (86/100), even higher than the garment image itself slightly rotated (Fig. 3e), highlighting a key failure mode. Indeed, we evaluated these metrics across the entire dataset, with results in Tab. 4 (appendix), confirming DISTS’s robustness over traditional metrics for VTOFF. Perceptual quality may be captured with no-reference metrics like Fréchet Inception Distance (FID) [22] and Kernel Inception Distance (KID) [4], as they compare feature distributions. However, these are sensitive to sample size and outliers, making them unsuitable for single-image evaluation. Additionally, their reliance on Inception features [45] misaligns with human judgment in assessing perceptual quality [42]. Given these limitations, we adopt Deep Image Structure and Texture Similarity (DISTS) [11] metric as our primary measure. DISTS compares images using VGG features [41], combining low-level structural and high-level textural features through a weighting scheme optimized via human-rated similarity. This yields a perceptual metric that better captures the quality of garment reconstructions in VTOFF.

### 3.2 TryOffDiff

TryOffDiff adapts Stable Diffusion [35] (v1.4), a latent diffusion model originally designed for text-conditioned image generation using CLIP [34], to perform image-guided generation by replacing text prompts with image features. A core challenge in image-guided generation is effectively incorporating visual features into the conditioning mechanism of the generative model. CLIP’s ViT [34] has become a popular choice for image feature extraction due to its general-purpose capabilities and joint embedding space for text and images. Recently, SigLIP [61] introduced modifications that improve performance, particularly for tasks requiring more detailed and domain-specific visual representations. Therefore, we use the SigLIP model as image feature extractor and retain the entire sequence of token representations in its final layer to preserve spatial information, which we find essential for the capture of fine-grained visual details and accurate garment reconstruction. For an input image I, the condition is derived as C(I) = (LN◦Linear◦SigLIP)(I) ∈ Rn×m, where SigLIP embeddings are linearly projected followed by layer normalization (LN) [1], cf. Fig. 4. These features

[Figure 9]

- Figure 4: TryOffDiff architecture. SigLIP [61] extracts features from a reference image, processed by adapter modules and embedded into pretrained Stable Diffusion v1.4 [35] by replacing text features in cross-attention layers, enabling image-guided generation. Joint training of adapter layers and diffusion model enables effective garment transformation.

are integrated into the denoising U-Net via cross-attention, with keys K = C(I)·Wk ∈ Rn×dk and values V = C(I)·WV ∈ Rn×dv, where Wk ∈ Rm×dk and Wv ∈ Rm×dv. This conditions the denoising process on the visual features of reference image I, enhancing alignment in the output. We train adapter modules and finetune denoising U-Net, keeping SigLIP and VAE encoder-decoder frozen to leverage pretrained capabilities.

## 4 Experiments

We evaluate our proposed TryOffDiff model on the VTOFF task against a range of adapted baselines, including virtual try-on methods, pose transfer approaches, and a general-purpose image-conditioning method. We describe experimental setup for reproducibility, report quantitative results, and present qualitative comparisons to demonstrate TryOffDiff’s ability to reconstruct fine-grained garment details across diverse inputs.

Datasets. We conduct experiments on VITON-HD [7], containing 13,679 high-resolution (1024×768) pairs of frontal half-body models and upper-body garments, and Dress Code [30], with 53,792 pairs (1024×768) of full-body models and upper-body, lower-body, and dress garments. We focus exclusively on upper-body garments, using 11,552 training and 1,990 test pairs for VITON-HD, and 13,563 training and 1,800 test upper-body pairs for Dress Code. Dataset cleaning details are in App. A.

Implementation Details. TryOffDiff builds on pretrained Stable Diffusion v1.4 [35], finetuning the denoising U-Net and training adapter layers from scratch. Input images are padded to a square aspect ratio and resized to 512×512 for SigLIP-B/16-512 and VAE compatibility. The adapter reduces SigLIP’s 1024 token embeddings (dimension 768) to 77 embeddings (dimension 768). Training uses a single NVIDIA A40 GPU for 150k iterations with MSE loss. Detailed parameters are in App. B.

Baseline Approaches. We adapt state-of-the-art methods from pose transfer, view synthesis, and virtual try-on as baselines, modifying each to approximate garment reconstruction

[Figure 10]

[Figure 11]

(a) Left to right: reference image, fixed pose heatmap derived from target image, initial model output, SAM prompts, and final processed output.

(b) Left to right: masked conditioning image, mask image, pose image, initial model output with SAM prompts, and final processed output.

[Figure 12]

[Figure 13]

[Figure 14]

(c) Left to right: masked garment, model image, masked model, initial model output with SAM prompts, and final output.

(d) Left to right: conditioning garment, blank model, mask image, initial output with SAM prompts, and final output.

(e) Left to right: starting image, image prompt, initial output, output with SAM prompts, and final output.

- Figure 5: Adapting baselines for VTOFF. (a) GAN-Pose [36] uses pose transfer; (b) ViscoNet [6] employs view synthesis; (c) OOTDiffusion [58] and (d) CatVTON [9] are recent virtual try-on methods; (e) IP-Adapter [59] is a general-purpose image-conditioned baseline.

functionality as closely as possible. In addition, we include IP-Adapter [59], a generalpurpose image-conditioning method, to evaluate diffusion-based image-guided generation in the VTOFF setting. All outputs are post-processed with Segment Anything (SAM) [27] using point prompts to isolate garments, then composited onto a white background, standardizing outputs and mitigating background-related artifacts that negatively impact evaluation metrics. These baseline approaches are illustrated in Fig. 5 and detailed in App. C.

Quantitative Results. Quantitative results on VITON-HD are shown in Tab. 1. TryOffDiff outperforms baselines across all metrics, achieving a DISTS score of 20.3, a 7.9-point improvement over CatVTON (28.2). Baseline rankings vary: GAN-Pose excels in fullreference metrics (e.g., SSIM 77.4), while CatVTON leads in no-reference metrics (e.g., FID 31.4). DISTS, balancing structure and texture, aligns with visual perception, ranking CatVTON highest among baselines.

We examine the robustness of our approach by evaluating its cross-dataset generalization. On Dress Code, TryOffDiff achieves a DISTS score of 21.6 for upper-body garments in the within-dataset setting (DC/DC, Tab. 2). Baselines were not evaluated on Dress Code

Method SSIM↑ MSSSIM ↑ CWSSIM ↑ LPIPS↓ FID↓ FDCLIP ↓ KID↓ DISTS↓

GAN-Pose [36] 77.4 63.8 32.5 44.2 73.2 30.9 55.8 30.4 ViscoNet [6] 58.5 50.7 28.9 54.0 42.3 12.1 25.5 31.2 OOTDiff. [58] 65.1 50.6 26.1 49.5 54.0 17.5 33.2 32.4 CatVTON [9] 72.8 56.9 32.0 45.9 31.4 9.7 17.8 28.2 IP-Adapter [59] 63.2 53.3 30.1 51.5 36.1 13.1 21.6 30.9 TryOffDiff 80.3 72.5 49.2 30.1 14.5 5.7 3.9 20.3

- Table 1: Quantitative comparison on VITON-HD-test. Performance of TryOffDiff and baseline methods on the VTOFF task, evaluated using standard similarity and perceptual metrics. TryOffDiff outperforms all baselines across the board, achieving the best results on both full-reference (SSIM variants, LPIPS) and perceptual (FID, KID, DISTS) measures.

Train/Test SSIM↑ MSSSIM ↑ CWSSIM ↑ LPIPS↓ FID↓ FDCLIP ↓ KID↓ DISTS↓

HD/HD 80.3 72.5 49.2 30.1 14.5 5.7 3.9 20.3 DC/HD 77.8 65.5 37.0 38.3 22.9 7.9 9.2 26.6

DC/DC 80.8 73.8 47.8 31.6 17.1 5.2 4.7 21.6 HD/DC 75.8 65.4 34.7 41.5 29.6 9.7 10.5 28.1

- Table 2: Within- and cross-dataset evaluation of TryOffDiff. Metrics for TryOffDiff trained and tested on VITON-HD (HD) or DressCode (DC, upper-body) in within-dataset (HD/HD, DC/DC) and cross-dataset (DC/HD, HD/DC) settings.

due to their poor performance in initial tests. As our main focus is on upper-body garments, results for lower-body garments and dresses are provided in App. E for completeness. Importantly, in the cross-dataset setting, TryOffDiff trained on Dress Code and tested on VITONHD (DC/HD) outperforms all VITON-HD baselines across all metrics (e.g., DISTS 26.6 vs. CatVTON’s 28.2), despite domain shifts (e.g., full-body to half-body references). This demonstrates the model’s strong generalization capabilities, although performance remains below the in-domain HD/HD setting (DISTS 20.3), which is expected under domain shift.

Qualitative Analysis. Qualitative results for VITON-HD are shown in both Fig. 1 and Fig. 6, while Dress Code examples appear in Fig. 1 (more in App. E). These examples align with our quantitative findings and highlight metric-specific strengths, as discussed in Sec. 3.1. GAN-Pose approximates garment color and shape but often misses small regions, reducing visual fidelity in no-reference metrics. ViscoNet produces realistic outputs but deforms shapes and favors long sleeves, with limited textural detail. OOTDiffusion preserves logos but struggles to maintain texture consistency. CatVTON excels at rendering natural textures and logos but often mismatches garment shapes, limiting full-reference performance. In contrast, TryOffDiff consistently reconstructs accurate shapes and textures, including occluded regions (e.g., high-cut bodysuits in Dress Code), colors, patterns, and logos, achieving superior DISTS scores. Its image conditioning mechanism enables precise recovery, outperforming baselines. Additional qualitative results are provided in App. E.

Ablations. We ablate TryOffDiff’s components on VITON-HD (Tab. 3). Using SigLIPB/16 over CLIP ViT-B/32 improves DISTS from 23.5 to 22.1 without pretraining, and to 20.3 with pretraining, due to enhanced feature extraction. The adapter (Linear+LN) boosts performance (e.g., DISTS 23.7 to 23.5 with CLIP), and pretraining Stable Diffusion v1.4 further improves all metrics (e.g., DISTS 22.1 to 20.3). These results confirm the importance of SigLIP and pretraining for high-fidelity garment reconstruction. Additional ablation analyses are in App. E.

Training Configuration Structural Similarity↑ Perceptual Quality↓ Img. Encoder Adapter Cond. Shape Pretrain SSIM MSSSIM CWSSIM LPIPS FID FDCLIP KID DISTS↓

CLIP ViT-B/32 - (257,1024) - 77.9 69.2 46.4 35.3 16.8 7.7 5.2 23.7 CLIP ViT-B/32 Linear+LN (77,768) - 78.8 70.5 46.5 34.3 16.3 7.4 4.7 23.5

SigLIP-B/16 Linear+LN (77,768) - 78.4 69.8 47.3 33.1 16.4 6.6 5.1 22.1 CLIP ViT-B/32 Linear+LN (77,768) ✓ 79.8 71.5 47.8 32.0 15.2 5.8 4.1 21.5

SigLIP-B/16 Linear+LN (77,768) ✓ 80.3 72.5 49.2 30.1 14.5 5.7 3.9 20.3

- Table 3: Ablation study on VITON-HD. Training configurations and metrics on VITONHD-test. “Pretrain" indicates U-Net initialization from Stable Diffusion (✓) or scratch (-). All ablations mirror Stable Diffusion-v1.4 architecture.

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

(a)

(b)

(c) ViscoNet (d)

(e)

(f)

(g)

(h) Target

Reference

Gan-Pose

OOTDiff.

CatVTON

IP-Adapter

TryOffDiff

Figure 6: Qualitative comparison on VITON-HD-test. In comparison to the baseline approaches, TryOffDiff is capable of generating garment images with accurate structural details as well as fine textural details.

## 5 Conclusion

In this paper, we introduced virtual try-off (VTOFF), a novel task designed to reconstruct standardized upper-body garment images from a single reference photo of a person wearing the garment. Unlike virtual try-on (VTON), VTOFF emphasizes precise garment reconstruction, making it ideal for evaluating generative models’ accuracy in capturing fine details such as patterns and logos. We proposed TryOffDiff, an adaptation of Stable Diffusion with SigLIP image conditioning to guide the generative process, replacing text-based inputs. Evaluated on VITON-HD and Dress Code (upper-body garments), it outperforms baselines adapted from pose transfer and VTON methods, achieving the best performance across all evaluation metrics. Cross-dataset experiments further demonstrate its robust generalization despite domain shifts (e.g., full-body to half-body references). Qualitative results confirm TryOffDiff’s strength in preserving garment shape, texture, and occluded regions, with minimal pre- and post-processing. While TryOffDiff performs well overall, challenges remain in reconstructing garments with highly complex structures, such as intricate logos and text. Looking ahead, VTOFF can be extended to multi-garment scenarios and enriched with more advanced conditioning or generation architectures, offering new avenues for evaluating generative models and advancing applications in e-commerce.

## Acknowledgment

This work has been funded by Horizon Europe program under grant agreement 101134447ENFORCE, and by the German federal state of North Rhine-Westphalia as part of the research funding program KI-Starter. We would like to thank UniZG-FER for providing access to their hardware.

## References

- [1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. stat, 1050:21, 2016.
- [2] Jason Baldridge, Jakob Bauer, Mukul Bhutani, Nicole Brichtova, Andrew Bunner, Kelvin Chan, et al. Imagen 3. arXiv, 2024. https://doi.org/nqr4.
- [3] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, et al. Improving image generation with better captions. preprint, 2023.
- [4] Mikołaj Bi´nkowski, Danica J Sutherland, Michael Arbel, and Arthur Gretton. Demystifying mmd gans. In ICLR, 2018.
- [5] Chaofeng Chen and Jiadi Mo. IQA-PyTorch: Pytorch toolbox for image quality assessment. https://github.com/chaofengc/IQA-PyTorch, 2022.
- [6] Soon Yau Cheong, Armin Mustafa, and Andrew Gilbert. Visconet: Bridging and harmonizing visual and textual conditioning for controlnet. In ECCVW, 2024.
- [7] Seunghwan Choi, Sunghyun Park, Minsoo Lee, and Jaegul Choo. Viton-hd: Highresolution virtual try-on via misalignment-aware normalization. In CVPR, 2021.
- [8] Yisol Choi, Sangkyung Kwak, Kyungmin Lee, Hyungwon Choi, and Jinwoo Shin. Improving diffusion models for virtual try-on. arXiv, 2024. https://doi.org/ np47.
- [9] Zheng Chong, Xiao Dong, Haoxiang Li, Shiyue Zhang, Wenqing Zhang, Xujie Zhang, Hanqing Zhao, and Xiaodan Liang. Catvton: Concatenation is all you need for virtual try-on with diffusion models. In ICLR, 2025.
- [10] Aiyu Cui, Daniel McKee, and Svetlana Lazebnik. Dressing in order: Recurrent person image generation for pose transfer, virtual try-on and outfit editing. In ICCV, 2021.
- [11] Keyan Ding, Kede Ma, Shiqi Wang, and Eero P Simoncelli. Image quality assessment: Unifying structure and texture similarity. IEEE TPAMI, 2020.
- [12] Haoye Dong, Xiaodan Liang, Xiaohui Shen, Bowen Wu, Bing-Cheng Chen, and Jian Yin. Fw-gan: Flow-navigated warping gan for video virtual try-on. In ICCV, 2019.
- [13] Daria Dzyabura, Siham El Kihal, John R Hauser, and Marat Ibragimov. Leveraging the power of images in managing product return rates. Mark. Sci., 2023.

- [14] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024.
- [15] Yuying Ge, Ruimao Zhang, Lingyun Wu, Xiaogang Wang, Xiaoou Tang, and Ping Luo. A versatile benchmark for detection, pose estimation, segmentation and reidentification of clothing images. In CVPR, 2019.
- [16] Yuying Ge, Yibing Song, Ruimao Zhang, Chongjian Ge, Wei Liu, and Ping Luo. Parser-free virtual try-on via distilling appearance flows. In CVPR, 2021.
- [17] Xiao Han, Xiatian Zhu, Jiankang Deng, Yi-Zhe Song, and Tao Xiang. Controllable person image synthesis with pose-constrained latent diffusion. In ICCV, 2023.
- [18] Xintong Han, Zuxuan Wu, Zhe Wu, Ruichi Yu, and Larry S Davis. Viton: An imagebased virtual try-on network. In CVPR, 2018.
- [19] Xintong Han, Xiaojun Hu, Weilin Huang, and Matthew R Scott. Clothflow: A flowbased model for clothed person generation. In CVPR, 2019.
- [20] Jochen Hartmann, Yannick Exner, and Samuel Domdey. The power of generative marketing: Can generative ai create superhuman visual marketing content? Int. J. Res. Mark., 2024.
- [21] Kai He, Kaixin Yao, Qixuan Zhang, Jingyi Yu, Lingjie Liu, and Lan Xu. Dresscode: Autoregressively sewing and generating garments from text guidance. TOG, 2024.
- [22] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, 2017.
- [23] Wonbong Jang and Lourdes Agapito. Nvist: In the wild new view synthesis from a single image with transformers. In CVPR, 2024.
- [24] Nikolay Jetchev and Urs Bergmann. The conditional analogy gan: Swapping fashion articles on people images. In ICCVW, 2017.
- [25] Johanna Karras, Aleksander Holynski, Ting-Chun Wang, and Ira KemelmacherShlizerman. Dreampose: Fashion image-to-video synthesis via stable diffusion. In ICCV, 2023.
- [26] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In NeurIPS, 2022.
- [27] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, 2023.
- [28] Tejas D Kulkarni, William F Whitney, Pushmeet Kohli, and Josh Tenenbaum. Deep convolutional inverse graphics network. In NeurIPS, 2015.
- [29] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019.

- [30] Davide Morelli, Matteo Fincato, Marcella Cornia, Federico Landi, Fabio Cesari, and Rita Cucchiara. Dress code: High-resolution multi-category virtual try-on. In CVPR, 2022.
- [31] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In AAAI, 2024.
- [32] Gaurav Parmar, Richard Zhang, and Jun-Yan Zhu. On aliased resizing and surprising subtleties in gan evaluation. In CVPR, 2022.
- [33] Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and JunYan Zhu. Zero-shot image-to-image translation. In SIGGRAPH, 2023.
- [34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021.
- [35] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.
- [36] Prasun Roy, Saumik Bhattacharya, Subhankar Ghosh, and Umapada Pal. Multi-scale attention guided pose transfer. PR, 2023.
- [37] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, et al. Palette: Image-to-image diffusion models. In SIGGRAPH, 2022.
- [38] Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi. Image super-resolution via iterative refinement. IEEE TPAMI, 2022.
- [39] Fei Shen, Hu Ye, Jun Zhang, Cong Wang, Xiao Han, and Wei Yang. Advancing poseguided image synthesis with progressive conditional diffusion models. In ICLR, 2024.
- [40] Le Shen, Yanting Kang, Rong Huang, and Zhijie Wang. Mfp-vton: Enhancing maskfree person-to-person virtual try-on via diffusion transformer. arXiv, 2025. https: //doi.org/pnd8.
- [41] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for largescale image recognition. In ICLR, 2015.
- [42] George Stein, Jesse Cresswell, Rasa Hosseinzadeh, Yi Sui, Brendan Ross, Valentin Villecroze, Zhaoyan Liu, Anthony L Caterini, Eric Taylor, and Gabriel Loaiza-Ganem. Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models. In NeurIPS, 2024.
- [43] Shao-Hua Sun, Minyoung Huh, Yuan-Hong Liao, Ning Zhang, and Joseph J Lim. Multi-view to novel view: Synthesizing novel views with self-learned confidence. In ECCV, 2018.
- [44] Zhengwentai Sun, Yanghong Zhou, Honghong He, and PY Mok. Sgdiff: A style guided diffusion model for fashion synthesis. In ACMMM, 2023.

- [45] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In CVPR, 2015.
- [46] Huixuan Tang, Neel Joshi, and Ashish Kapoor. Learning a blind measure of perceptual image quality. In CVPR, 2011.
- [47] Maxim Tatarchenko, Alexey Dosovitskiy, and Thomas Brox. Multi-view 3d models from single images with a convolutional network. In ECCV, 2016.
- [48] Lucas Theis, Aäron van den Oord, and Matthias Bethge. A note on the evaluation of generative models. In ICLR, 2016.
- [49] Dmitry Tochilkin, David Pankratz, Zexiang Liu, Zixuan Huang, Adam Letts, Yangguang Li, Ding Liang, Christian Laforte, Varun Jampani, and Yan-Pei Cao. Triposr: Fast 3d object reconstruction from a single image. arXiv, 2024. https: //doi.org/nq56.
- [50] Brandon Van Der Heide, Benjamin K. Johnson, and Mao H. Vang. The effects of product photographs and reputation systems on consumer behavior and product cost on ebay. Comput. Hum. Behav., 2013.
- [51] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017.
- [52] Riza Velioglu, Robin Chan, and Barbara Hammer. Fashionfail: Addressing failure cases in fashion object detection and segmentation. In IJCNN, 2024.
- [53] Bochao Wang, Huabin Zheng, Xiaodan Liang, Yimin Chen, Liang Lin, and Meng Yang. Toward characteristic-preserving image-based virtual try-on network. In ECCV, 2018.
- [54] Chenhui Wang, Tao Chen, Zhihao Chen, Zhizhong Huang, Taoran Jiang, Qi Wang, and Hongming Shan. Fldm-vton: Faithful latent diffusion model for virtual try-on. In IJCAI, 2024.
- [55] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Trans. Image Process., 2004.
- [56] Huosong Xia, Xiaoting Pan, Yanjun Zhou, and Zuopeng Justin Zhang. Creating the best first impression: Designing online product photos to increase sales. Decis. Support Syst., 2020.
- [57] Zhenyu Xie, Zaiyu Huang, Fuwei Zhao, Haoye Dong, Michael Kampffmeyer, and Xiaodan Liang. Towards scalable unpaired virtual try-on via patch-routed spatiallyadaptive gan. In NeurIPS, 2021.
- [58] Yuhao Xu, Tao Gu, Weifeng Chen, and Chengcai Chen. Ootdiffusion: Outfitting fusion based latent diffusion for controllable virtual try-on. In AAAI, 2025.

- [59] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv, 2023. https:// doi.org/np3v.
- [60] Wei Zeng, Mingbo Zhao, Yuan Gao, and Zhao Zhang. Tilegan: category-oriented attention-based high-quality tiled clothes generation from dressed person. Neural Comput. Appl., 2020.
- [61] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In ICCV, 2023.
- [62] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to textto-image diffusion models. In ICCV, 2023.
- [63] Xujie Zhang, Yu Sha, Michael C Kampffmeyer, Zhenyu Xie, Zequn Jie, Chengwen Huang, Jianqing Peng, and Xiaodan Liang. Armani: Part-level garment-text alignment for unified cross-modal fashion design. In ACMM, 2022.
- [64] Xujie Zhang, Binbin Yang, Michael C Kampffmeyer, Wenqing Zhang, Shiyue Zhang, Guansong Lu, Liang Lin, Hang Xu, and Xiaodan Liang. Diffcloth: Diffusion based garment synthesis and manipulation via structural cross-modal semantic alignment. In ICCV, 2023.
- [65] Bo Zhao, Xiao Wu, Zhi-Qi Cheng, Hao Liu, Zequn Jie, and Jiashi Feng. Multi-view image generation from a single-view. In ACM MM, 2018.
- [66] Tinghui Zhou, Shubham Tulsiani, Weilun Sun, Jitendra Malik, and Alexei A Efros. View synthesis by appearance flow. In ECCV, 2016.

## TryOffDiff: Virtual-Try-Off via High-Fidelity Garment Reconstruction using Diffusion Models

Supplementary Material

## A Dataset Cleaning

Upon closer inspection of VITON-HD, we identified 95 duplicate image pairs (0.8%) in the training set and 6 duplicate pairs (0.3%) in the test set. Additionally, we found 36 pairs (1.8%) in the training set that had been included in the original test split. To ensure the integrity of our experiments, we cleaned the dataset by removing all duplicates in both subsets as well as all leaked examples from the test set. The resulting cleaned dataset, contains 11,552 unique image pairs for training and 1,990 unique image pairs for testing. We provide the script for cleaning the dataset in our code repository.

## B Training Details

We train TryOffDiff by building on the pretrained Stable Diffusion v1.4 [35], focusing on finetuning the denoising U-Net and training adapter layers from scratch. As a preprocessing step, we pad the input reference image along the width for a square aspect ratio, then resize

- them to a resolution of 512×512 to match the expected input format of the pretrained SigLIP and VAE encoder. For training, we preprocess the garment images in the same way. We use SigLIP-B/16-512 as image feature extractor, which outputs 1024 token embeddings of dimension 768. Our adapter, consisting of linear and normalization layers, reduces these to n = 77 conditioning embeddings of dimension m = 768.

Training occurs over 150k iterations on a single NVIDIA A40 GPU, requiring approximately 5 days with a batch size of 16. We employ the AdamW optimizer [29], with an initial learning rate of 1e-4 that increases linearly from 0 during the first %5 warmup steps,

- then follows a linear decay to 0. As proposed in [26], we use the EulerDiscrete scheduler with 1,000 steps. We optimize using the standard Mean Squared Error (MSE) loss, which measures the difference between the added and the predicted noise at each step. This loss function is commonly employed in diffusion models to guide the model in learning to reverse the noising process effectively. During inference, we run TryOffDiff with a EulerDiscrete scheduler over 20 timesteps with a guidance scale of 2.0. On a single NVIDIA A40 GPU, this process takes 1.8 seconds per image and requires 9.8GB of memory.

## C Baselines

GAN-Pose [36] is a GAN-based pose transfer method that expects three inputs: a reference image, and pose heatmaps of the reference and target subject. Garment images from VITONHD are used to estimate the heatmap for a fixed, neutral pose. This setup enables the transfer of human poses from diverse reference images to a standardized pose, aligning the output to the typical view of product images.

ViscoNet [6] requires a text prompt, a pose, a mask, and multiple masked conditioning images as inputs. For the text prompt, we use a description such as “a photo of an e-commerce

clothing product”. We choose a garment image from VITON-HD to estimate a neutral pose as well as a generic target mask. Since ViscoNet is originally trained with masked conditioning images, we apply an off-the-shelf fashion parser [52] to mask the upper-body garment, which is then provided as input.

OOTDiffusion [58] takes a garment image and a reference image to generate a VTON output. To adapt this model for VTOFF, we again apply the fashion parser [52] to mask the upper-body garment to create the garment image. We select a reference image with a mannequin in a neutral pose as further input. An intermediate step involves masking the upperbody within the reference image, for which we use a hand-crafted masked version of the reference image.

CatVTON [9] is a model that generates a VTON image using a reference image and a conditioning garment image as inputs. An intermediate step incorporates upper-body masks

- to guide the try-on process. For adaptation to VTOFF, we replace the reference image with a plain white image and use a handcrafted mask in a neutral pose, enabling CatVTON to perform garment transfer independently of any specific person. IP-Adapter [59] integrates into Stable Diffusion-v1.5 without architecturla changes, enabling image-based conditioning via image prompts. For VTOFF, we adapt IP-Adapter to reconstruct standardized garment images using visual cues from a reference image. The generation is initialized from a mask-like image on a white background, serving as a neutral garment template. A reference image of a clothed person guides the synthesis, with a generic text prompt (“high quality product photo of a garment”) and a high image prompt weight (0.75) to ensure fidelity to garment details while maintaining a consistent, catalogstyle presentation.

## D Implementation Details

For evaluation, we use ‘IQA-PyTorch’ [5] to compute SSIM, MS-SSIM, CW-SSIM, and LPIPS, and the ‘clean-fid’ [32] library for FID, CLIP-FID, and KID. We employ the original implementation of DISTS [11] for evaluating perceptual image quality. For readability purposes, the values of SSIM, MS-SSIM, CW-SSIM, LPIPS, and DISTS presented in this paper are multiplied by 100, and KID is multiplied by 1000.

## E Additional Results

To further validate DISTS’s robustness over traditional metrics (as discussed in Sec. 3.1), we conducted experiments on the entire VITON-HD-test dataset, with averaged results reported in Tab. 4. The results confirm that widely used metrics for VTON, such as SSIM and its variants, fail to capture image similarity in a manner suitable for VTON and VTOFF. For instance, SSIM-based metrics assign the highest score to case (b)—a structurally correct but color-distorted image—while LPIPS similarly overestimates similarity in this case. In contrast, DISTS assigns its lowest score to case (e), which is arguably the most perceptually similar to the ground truth for garment-focused tasks. These findings highlight the inadequacy of traditional metrics for both VTON and VTOFF and establish DISTS as a more reliable measure for evaluating garment image similarity.

Tab. 5 presents evaluation metrics for TryOffDiff models, each trained separately on a distinct garment category from the Dress Code dataset: upper-body, lower-body, and dresses.

Case SSIM↑ MSSSIM ↑ CWSSIM ↑ LPIPS↓ DISTS↓

- (a) 82.59 71.31 40.31 35.05 21.43

- (b) 98.81 98.85 96.42 12.14 14.43

- (c) 90.24 78.72 58.91 33.77 19.76

- (d) 73.08 66.09 0 45.65 69.54

- (e) 81.48 77.15 72.35 17.42 10.04

- (f) 77.52 92.63 73.41 20.91 29.85

- Table 4: Quantitative evaluation of metric suitability. Average values of various metrics for different scenarios, as shown in Fig. 3. Metrics are calculated at a full resolution of 1024x768 using the VITON-HD-test dataset.

Garment Type SSIM↑ MSSSIM ↑ CWSSIM ↑ LPIPS↓ FID↓ FDCLIP ↓ KID↓ DISTS↓

Upper-body 80.8 73.8 47.8 31.6 17.1 5.2 4.7 21.6 Lower-body 81.1 76.8 55.5 30.0 22.6 9.4 5.9 21.1 Dresses 81.6 76.1 55.6 26.1 18.4 8.2 4.4 20.8

- Table 5: Evaluation metrics for TryOffDiff models by garment type. Models are trained independently on upper-body, lower-body, and dress categories from Dress Code dataset.

The results demonstrate robust performance across all categories, with structural metrics (SSIM, MS-SSIM, CW-SSIM) and perceptual metrics (DISTS, LPIPS) showing minimal variation. Dresses exhibit slightly improved DISTS and LPIPS scores, indicating that TryOffDiff effectively handles garments with complex global structures. These findings highlight the model’s adaptability and effectiveness across diverse garment types.

To further demonstrate the limitations of the widely used SSIM metric in both VTON and VTOFF tasks, we compare ground-truth images with predictions from two arbitrary models, as shown in Fig. 7. The results highlight that SSIM often fails to penalize perceptual artifacts. For example, in Fig. 7(c), the prediction is blurry and lacks detail, yet receives a higher SSIM score than Fig. 7(d), which preserves patterns and offers a more faithful reconstruction. Similar issues are observed across other examples. In contrast, DISTS more reliably reflects perceptual quality, assigning lower scores to visually degraded outputs and aligning better with human judgment.

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

(a) 81.9 / 36.2 (b) 80.3 / 24.2 (c) 81.5 / 40.4 (d) 75.3 / 25.0 (e) 81.7 / 39.7 (f) 80.3 / 19.4

- Figure 7: Metric suitability for VTOFF (SSIM↑ / DISTS↓). Each pair displays ground truth (left) and predictions (right) from two arbitrary models (a-c vs. d-f). High SSIM scores obscure poor quality in the top row, while DISTS effectively highlights variations, better reflecting human perception.

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

(a)

(b)

(c) ViscoNet (d)

(e)

(f)

(g)

(h) Target

Reference

Gan-Pose

OOTDiff.

CatVTON

IP-Adapter

TryOffDiff

- Figure 8: Qualitative comparison between baselines and TryOffDiff on VITON-HD. TryOffDiff more accurately reconstructs both structural and textural garment details compared to baseline methods.

[Figure 37]

[Figure 38]

[Figure 39]

#### Figure 9: Qualitative results from TryOffDiff on Dress Code (upper-body garments).

