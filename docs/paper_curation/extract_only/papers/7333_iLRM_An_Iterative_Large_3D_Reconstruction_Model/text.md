# arXiv:2507.23277v3[cs.CV]31May2026

## iLRM: An Iterative Large 3D Reconstruction Model

Gyeongjin Kang1 Seungtae Nam2 Seungkwon Yang2 Xiangyu Sun1 Sameh Khamis3 Abdelrahman Mohamed4* Eunbyung Park2

1Sungkyunkwan University 2Yonsei University 3Rembrand 4Meta

https://gynjn.github.io/iLRM/

[Figure 1]

Figure 1. The overall architecture of the iLRM. As the layer index increases, compact viewpoint tokens are iteratively refined by attending to multi-view image tokens and are finally decoded into 3D Gaussian primitives, enabling efficient and high quality 3D reconstruction.

### Abstract

Feed-forward 3D modeling has emerged as a promising approach for rapid and high-quality 3D reconstruction. In particular, directly generating explicit 3D representations, such as 3D Gaussian splatting, has attracted significant attention due to its fast and high-quality rendering. However, many state-of-the-art methods, primarily based on transformer architectures, suffer from severe scalability issues because they rely on full attention across image tokens from multiple input views, resulting in prohibitive computational costs as the number of views or image resolution increases. Toward a scalable and efficient feed-forward 3D reconstruction, we introduce an iterative Large 3D Reconstruction Model (iLRM) that generates 3D Gaussian representations through an iterative refinement mechanism, guided by three core principles: (1) decoupling the scene representation from input images to enable compact 3D repre-

*Work done while at Rembrand.

sentations; (2) decomposing global multi-view interactions into a two-stage attention scheme to reduce computational costs; and (3) injecting high-resolution information at every layer to achieve high-fidelity reconstruction. Experimental results on widely used datasets, such as RE10K and DL3DV, demonstrate that iLRM outperforms existing methods in both reconstruction quality and speed.

### 1. Introduction

Since the recent success of 3D Gaussian Splatting (3DGS) [30], significant progress has been made in leveraging this 3D representation for building generalizable feedforward 3D reconstruction models [5, 10, 11, 49, 57, 58, 63]. These methods typically train large neural networks to transform multi-view input images into feature representations, then regress Gaussian attributes. In contrast to per-scene 3D-GS optimization approaches [18, 30, 39, 40], these feed-forward models can reconstruct 3D scenes in a

single forward pass, offering near real-time performance. Moreover, the prior knowledge learned from large-scale datasets [14, 15, 35, 66] allows them to effectively generalize to unseen scenes. While their reconstruction quality often lags behind that of per-scene optimization methods, fast reconstruction speed and generalization capability mark a promising step toward the long-standing goal of achieving accurate and real-time 3D scene reconstruction.

Among the promising approaches, pixel-aligned Gaussian models [5, 48, 65] have emerged as the de facto standard, leveraging decades of advances developed for numerous image-based tasks. While these models have proven effective, they also exhibit certain limitations. In particular, since they generate per-pixel Gaussians directly from the input images, the image resolution determines the number of Gaussians produced, which can lead to an excessive number of redundant Gaussians. For example, given input images at 1K resolution across 200 viewpoints (a scale comparable to the bicycle scene in the mip-NeRF 360 dataset [3]), these methods would produce 200 million Gaussians, despite prior studies [9, 17, 32, 33] demonstrating that such scenes can be efficiently represented with around 0.5 million Gaussians. To mitigate this issue, several techniques have been proposed, such as Gaussian regularization [67] and feature fusion [54]. Alternatively, the network can also be designed to generate fewer Gaussians, for example by downsampling the output resolutions. However, these strategies still require processing high-resolution images and therefore do not address another fundamental limitation of these models: high computational and memory demands.

A significant portion of computational and memory overhead arises from modeling interactions across multiple input views in feed-forward 3D reconstruction models. For instance, GS-LRM [63] performs full attention over all image tokens from every input view, leading to a quadratic increase in complexity with respect to both the number of views and image resolution. MVSplat [11] and DepthSplat [57] construct and process cost volumes for each view, further contributing to the computational demands. While one might attempt to alleviate this burden by reducing the input image resolution or using a sparser set of views, such strategies risk discarding essential geometric and appearance information required for accurate reconstruction.

Beyond the computational complexity and the inefficiency of the generated representations, we also question whether the prevailing formulation, casting 3D reconstruction as a sequence-to-sequence problem that maps entire sets of image tokens to dense, pixel-aligned Gaussians, is fundamentally well-suited to the nature of the task. While this formulation has achieved impressive results [27, 63, 67], it remains primarily a one-shot 3D scene generation process. In contrast, the recent optimizationbased methods [30, 40] follow a fundamentally different

strategy: they treat reconstruction as an iterative refinement process, where each iteration involves rendering the current scene estimate, measuring reconstruction error, and updating the representation accordingly. This loop enables the model to progressively capture finer geometric and appearance details while ensuring strong 3D consistency. The success of these methods suggests that high-quality 3D reconstruction may benefit not only from expressive representations but also from feedback-driven iterative refinement, a trait largely absent in existing feed-forward 3D models.

In this paper, we introduce iLRM*, an iterative large 3D reconstruction model that effectively 1) incorporates the principles of feedback-driven refinement, while also 2) addressing the computational burden and representational inefficiencies inherent in existing feed-forward approaches. As illustrated in Fig. 1, the network (acting as an optimizer) transforms the embedding features (analogous to updating the 3D-GS representation) at each layer (analogous to each optimization step), based on multi-view image tokens (serving as gradient-like signals). This design allows the model to iteratively update the scene representation at every layer based on feedback from the multi-view input images, effectively mimicking the optimization process within a feedforward architecture. Through this process, the learned neural network jointly examines the input view images and the evolving scene representation to identify where and how to make targeted updates that improve reconstruction quality.

Another core design principle of our approach is to decouple the representation, later transformed into 3D Gaussians, from direct dependence on input images, addressing the computational complexity and redundancy that arise in architectures that generate pixel-aligned Gaussians directly from multi-view inputs. By decoupling the representation and input images, we can use low-resolution representations to produce a compact set of Gaussians while still leveraging high-resolution input images for detailed guidance.

In addition, we propose an efficient mechanism for modeling the interaction between the representations and the input images. A na¨ıve approach would involve computing full attention between all tokens across views, which quickly becomes computationally prohibitive. To overcome this, we initialize our representation using viewpoint embeddings, each tied to a specific input view. Interaction modeling is then split into two stages. First, we perform cross-attention between each viewpoint embedding and its corresponding image, which is highly efficient due to the one-to-one mapping. Next, we apply self-attention across all viewpoint embeddings. Importantly, since this second stage operates over a low-resolution representation space, it remains computa-

*By “feedback-driven,” we mean that at every update layer the current scene tokens are explicitly revised via cross-attention with each image tokens. We call this “iterative” because the scene representation is repeatedly updated layer by layer under static (unchanged) image evidence, rather than merely transformed by stacked self-attention blocks [25].

tionally tractable while facilitating global information exchange across views. Overall, this scalable design significantly reduces computational and memory overhead and allows for the incorporation of more viewpoints, thereby improving reconstruction fidelity.

We comprehensively evaluate the proposed method on large-scale datasets, RealEstate10K [66] and DL3DV [35]. The experimental results demonstrate that iLRM achieves superior rendering quality while substantially reducing both computational and memory overhead compared to recently proposed feed-forward Gaussian models. Moreover, in high-resolution and wide-coverage settings (540×960, 32 views), our method completes inference in 0.5 seconds, achieving comparable performance to an optimizationbased approach, which takes about 8 minutes.

### 2. Related Works

#### 2.1. Feed-forward 3D Gaussian Splatting

Feed-forward 3D Gaussian Splatting [5, 11, 41, 48, 49, 57, 58, 63] capitalizes on robust priors learned from extensive datasets to estimate Gaussian primitive parameters and synthesize novel view images using sparse input data. PixelSplat [5] and LatentSplat [55] predict Gaussians from image features using an epipolar line sampling method to enhance geometric accuracy, while MVSplat [11] and MVSGaussian [37] construct cost volumes through a plane-sweep stereo approach. In a further development, Flash3D [47] and DepthSplat [57] introduce a pre-trained depth estimation model [43, 59], which improves the robustness of the spatial positions of 3D Gaussians. In contrast, GSLRM [63] and Long-LRM [67] minimize reliance on explicit 3D priors by leveraging large-scale data-driven priors.

While demonstrating strong results, a major limitation of all the aforementioned approaches lies in their non-scalable architectural design, which restricts their ability to effectively leverage a large number of input views. Moreover, the one-shot generation strategy, which produces 3D representations in a single forward pass, often fails to capture complex geometric details and fine 3D consistency, making them suboptimal for high-quality 3D reconstruction. We address these limitations by proposing an iterative 3D reconstruction framework and scalable architectural designs.

Iterative refinements. Our work is also closely related to recent methods that adopt iterative refinement strategies, such as G3R [8] and Gen-Den [41]. Both utilize actual gradients to update their representations more precisely. While promising, these approaches require additional computational burden for rendering multiple images per training iteration, and relying solely on gradients may risk overlooking valuable information present in the raw input images. Nonetheless, exploring how to incorporate gradient information remains an interesting direction for future work.

- 2.2. 3D Representations from Embeddings Inspired by previous generative approaches [4, 20, 29], recent works [6, 19, 24] have investigated the synthesis of 3D representations directly from learnable embeddings, guided by input image supervision. This paradigm leverages the expressive capacity of latent spaces to encode rich geometric priors, which act as structural templates that guide the reconstruction process. Such approaches offer notable flexibility, allowing rendering from arbitrary viewpoints and adaptation to varying space scales and camera poses. However, both LRM [24] and Lara [6] are limited to object-centric representations, restricting scalability to complex scenes involving multiple objects or large spatial layouts. The recently proposed Quark [19] also utilizes learnable embeddings to fuse visual cues from multiple images, demonstrating compelling results, but its representation is confined to the target view [27, 37, 56], lacking an explicit and persistent 3D reconstruction.

In contrast to previous works, we construct scene-level explicit 3D representations from viewpoint embeddings by decoupling the generation of Gaussians from the input images. This separation enables iterative refinement of the embeddings using low-level visual features and provides flexible control over the density of the 3D representation, independent of the input image resolution.

- 3. Method

#### 3.1. Motivation and Problem Statement

Existing generalizable 3D Gaussian reconstruction methods process multi-view images in an end-to-end fashion, often employing techniques such as epipolar line sampling [5], plane-sweep stereo [10, 11, 57], or full-resolution attention [58, 63, 67] to enforce multi-view consistency. While effective, these strategies introduce significant computational and memory overhead, limiting their scalability.

To address these challenges, we propose iLRM, a novel feed-forward 3D reconstruction framework that decouples Gaussian generation from direct dependence on input images. Instead of generating pixel-aligned Gaussians, iLRM initializes viewpoint-centric embeddings as the basis for constructing the 3D scene. These embeddings are then iteratively refined via cross-attention with multi-view image features, enabling the model to efficiently fuse geometric and appearance cues across views.

We start with N multi-view images {Ii}Ni=1 and camera poses {Ci}Ni=1. Based on this setup, our goal is to train a model fθ that maps a set of viewpoints to 3D Gaussians, leveraging the associated multi-view images as visual cues throughout the reconstruction pipeline. More formally,

vWvN

fθ : {(Ci,Ii)}Ni=1  → {(µk,αk,Σk,ck)}H

k=1 , (1) where fθ is modeled as a feed-forward network with the model parameter θ. µk,αk,Σk,ck are attributes of 3D

[Figure 2]

Figure 2. The proposed scalable architectural designs by decoupling viewpoint and image tokens, and modeling the global interactions via cross- and self-attentions (N: # views, h = H/p, w = W/p).

Gaussians, representing the mean, opacity, covariance, and color, respectively, while Hv and Wv denote the height and width of the generated Gaussians for each camera viewpoint. It is important to note that they do not correspond to the resolution of the input images. We train our model using held-out target images along with their corresponding camera poses, enabling high-quality novel view synthesis.

#### 3.2. Architectural Design

We propose an end-to-end transformer that directly regresses 3D Gaussian parameters from viewpoint embeddings. To compensate for the absence of direct image input, we enrich these embeddings at each layer via crossattention with multi-view image features. The resulting embeddings are further refined through self-attention to capture global dependencies across viewpoints.

Viewpoint tokenization. Following previous works [27, 49, 63], we employ a Pl¨ucker ray embedding for each input view using the camera poses. Specifically, given the intrinsic, rotation, and translation, we construct the Pl¨ucker ray embeddings for each viewpoint. We then divide these viewpoint embeddings into non-overlapping patches of size p × p, and reshape each patch into a 1D vector, resulting in a tensor of shape HvWv/p2 × 6p2. Then, we encode it using a single linear layer to produce viewpoint to-

kens, Vi(0) ∈ RH

vWv/p2×d. Pl¨ucker coordinates inherently capture spatial variations across pixels and views, allowing them to effectively differentiate between patches. As a result, we do not utilize additional positional embeddings.

Multi-view image tokenization. For each input view image, which provides visual guidance to the reconstruction process, we extract both image features and corresponding pose information. Specifically, we divide an input image into non-overlapping patches and obtain two sets: RGB image patches and Pl¨ucker ray patches. These are then concatenated and linearly projected to construct the image patch tokens, Si ∈ RHW/p

2×d, Sij = Linear(concat(Iij,Pij)) ∈ Rd, (2)

2

2

represent the flattened j-th image and ray patches for the i-th view, respectively, and HW/p2 is the number of tokens for each input view image. Scalable multi-view context modeling. Fig. 2-(a) shows the typical feed-forward 3D methods [11, 57, 63] using transformer architecture, which perform full attention across multi-view images, incurring a quadratic increase in computational cost with respect to both the number of views and the image resolution. Fig. 2-(b) depicts our decoupling approach. Thanks to the decoupling technique, we can reduce the viewpoint resolution while still leveraging high-resolution multi-view images (Fig. 2-(c)). We further decrease the computation cost by two-stage multiview context modeling, per-view cross-attention and viewpoint self-attention (Fig. 2-(d)). For example, given 16 input images with a resolution of 256 × 256 and a patch size of 8, the relative computational cost follows the ratio (1:1:0.25:0.08, Fig. 2-(a):(b):(c):(d)), highlighting that our proposed method can accommodate more input views with significantly less computational burden.

where Iij ∈ R3p

,Pij ∈ R6p

[Figure 3]

Figure 3. Network architecture.

Update block. Given a set of viewpoint tokens, we formulate the problem as an iterative refinement process, where the viewpoint tokens are progressively updated through in-

[Figure 4]

Figure 4. Various mini-batch cross-attention schemes. We primarily adopt “Quarter Cross-attention” in our experiments.

teractions with multi-view image tokens, ultimately leading to more accurate and spatially consistent 3D Gaussian Splatting. As shown in Fig. 3, our model consists of multiple transformer modules, each comprising one crossattention layer followed by one self-attention layer.

V˜i(l−1) = cross-attn(l)(Vi(l−1),Si), (3) {Vi(l)}Ni=1 = self-attn(l)({V˜i(l−1)}Ni=1), (4)

where the superscript (l) denotes the layer index. In the cross-attention layers, the viewpoint tokens are refined by the visual information from their corresponding image tokens. In the self-attention layers, the viewpoint tokens interact with each other to enhance their representations using global contextual information. Note that we use separate model parameters for the update blocks at different layers.

Token uplifting. Standard cross-attention is typically applied between token sets of similar spatial resolutions. In our setting, however, we intentionally use lower-resolution (LR) viewpoint tokens compared to HR image tokens to improve scalability and efficiency, which may limit their ability to fully capture rich visual information. To bridge this gap, we propose a token uplifting strategy. Each LR viewpoint token is lifted by a linear query layer that expands its feature dimension by a factor of k, yielding a tensor of shape HvWv/p2 × dk, which is then reshaped to HvWvk/p2×d so that each original token corresponds to k finer-grained query tokens for better visual correspondence during cross-attention. After cross-attention with HR image tokens as keys and values, the resulting tensor is reshaped back to HvWv/p2×dk and projected to the original dimension d via a linear layer, yielding refined viewpoint tokens of shape HvWv/p2 × d. We set k = 2 to balance representational capacity and efficiency.

Mini-batch cross-attention. In our architecture, viewpoint tokens are iteratively updated at each layer based on the image tokens via cross-attention. The proposed decoupling design allows us to arbitrarily reduce the number of viewpoint tokens for improved scalability, whereas the resolution of image tokens remains fixed due to their spatial na-

ture. Consequently, the primary computational bottleneck in cross-attention lies in the high-resolution image tokens.

To address this, we propose several efficient crossattention schemes, as illustrated in Fig. 4, aimed at improving scalability without sacrificing performance. Our design is conceptually inspired by mini-batch gradient descent in optimization, where only a subset of data points is sampled in each iteration to reduce computational cost. Similarly, our mechanism selectively samples subsets of both image tokens and viewpoint tokens during cross-attention. While random token sampling (Fig. 4-(d)) is ideal in theory, it complicates efficient implementation. To mitigate this, we design structured sampling strategies that are simple to implement and demonstrate strong empirical performance.

Decoding into 3D Gaussians. After the final self-attention layer, we decode the final layer’s viewpoint tokens, Vi(L), into Gaussian parameters through a single linear layer and apply post-activation functions. For a detailed description, please refer to our supplementary materials.

Interpretation. Compared to standard self-attention, S(l) = S(l−1) + f(S(l−1)), our method applies evidenceconditioned updates, V (l) = V (l−1) +F(V (l−1),S), where the image tokens S are fixed and provide detailed visual guidance. This resembles a gradient descent iteration, V (l) ≈ V (l−1) − η∇V E(V (l−1);S), where E is an implicit objective function, making each layer a feedback correction step rather than a pure feature transformation. Our mini-batch variant further extends this view as Vmb(l) ≈ Vmb(l−1) − η∇VmbE(Vmb(l−1);Smb), where Vmb and Smb denote a subset of viewpoint tokens and their corresponding image tokens, respectively.

#### 3.3. Training Objectives

We rasterize 3D Gaussians from viewpoint tokens to obtain rendered images Iˆt, supervised against ground-truth images It via MSE and perceptual loss [7, 34]:

Ltotal =

t∈T

LMSE(Iˆt,It) + λLperceptual(Iˆt,It), (5)

where T is the set of target view indices and λ=0.5 balances the two loss terms.

### 4. Experiments

Datasets. We train our model on two large-scale datasets: RealEstate10K (RE10K) [66] and DL3DV [35], and evaluate it on three datasets, including ACID [36]. We adopt the RE10K split following [5] and the official split for DL3DV. We use an image resolution of 256×256 for the RE10K and ACID datasets, while for the DL3DV dataset, we use a resolution of 256×448 and 512×960. In addition, we employ the undistorted version of the DL3DV dataset at a resolution of 540×960, which originates from Long-LRM [67].

Implementation details. Our model consists of 12 update layers, each containing one cross-attention and one self-attention block. Inside each attention module, we adopt a pre-normalization method with LayerNorm [2] and QK-Norm [22] method with an RMSNorm [62] layer. Also, each block utilizes 12-head attention [50] and two GELU [21]-activated linear layers. We set the hidden dimension to d = 768, and use a patch size of p = 8. For more details, please refer to the supplementary material.

Evaluation. We compare our model against recent generalizable 3D reconstruction methods [5, 11, 41, 57, 63, 67] as well as optimization-based approach [30]. For evaluation, we follow the settings from [5, 60] for RE10K and [57, 67] for DL3DV. We denote our various viewpoint settings as (V,H/F,F), where V is the number of viewpoints, and the following entries indicate the resolutions of viewpoint and image tokens (H: half-resolution, F: full-resolution). For example, a setting of (2,H,F) indicates two viewpoints tokens with half-resolution and full-resolution image tokens. MC refers to our quarter mini-batch cross-attention (Fig. 4(c)). Note that our 2-view full-resolution setting (2,F,F) does not include token uplifting, as resolutions are identical. Additionally, when using more views than is required in the evaluation protocol, we sample extra views, ensuring there is no overlap with the target indices.

Method #Param (M) PSNR ↑ SSIM ↑ LPIPS ↓ # Gaussians Time (s)

pixelSplat 125 25.89 0.858 0.142 131,072 0.101 MVSplat 12 26.39 0.869 0.128 131,072 0.047 GS-LRM* 300 28.10 0.892 0.114 131,072 DepthSplat 354 27.47 0.889 0.114 131,072 0.065 Gen-Den 28 27.08 0.879 0.120 347,072 0.224 Ours (2,F,F) 171 28.65 0.900 0.110 131,072 0.025

Ours (4,H,F) 185 30.37 0.923 0.095 65,536 0.027 Ours-MC (4,H,F) 185 30.10 0.919 0.098 65,536 0.027 Ours (8,H,F) 185 31.57 0.935 0.082 131,072 0.028 Ours-MC (8,H,F) 185 31.24 0.933 0.084 131,072 0.029

Table 1. Quantitative comparisons on the RE10K dataset with various view configurations. Inference time is measured on a RTX 4090 GPU. * indicates closed-source methods. The time difference in the MC variant is negligible due to the short sequence length in inference. For a more comprehensive analysis of our mini-batch cross-attention, see Tab. 8.

[Figure 5]

Figure 5. Qualitative comparison on the RE10K dataset.

ACID DL3DV PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓

Method

MVSplat 28.15 0.841 0.147 22.65 0.737 0.191 DepthSplat 28.37 0.847 0.141 24.28 0.813 0.147 Gen-Den 28.61 0.847 0.141 22.92 0.750 0.188 Ours (2,F,F) 29.24 0.856 0.143 25.35 0.826 0.144

Ours (4,H,F) 30.10 0.877 0.138 27.90 0.877 0.122 Ours-MC (4,H,F) 29.90 0.873 0.141 27.68 0.881 0.127 Ours (8,H,F) 30.96 0.894 0.122 29.56 0.907 0.101 Ours-MC (8,H,F) 30.72 0.890 0.125 29.33 0.904 0.102

- Table 2. Cross-dataset generalization on the ACID and DL3DV (256×256) using a model trained on the RE10K dataset.

Method Views PSNR ↑ SSIM ↑ LPIPS ↓ # Gaussians Time (s) Memory (GB) MVSplat 6 22.93 0.775 0.193 688,128 0.279 5.87

DepthSplat

6 24.19 0.823 0.147 688,128 0.102 3.55 11 24.28 0.833 0.141 1,261,568 0.170 6.01 24 22.37 0.781 0.195 2,752,512 0.371 12.39

Ours

(6,H,F) 25.60 0.830 0.168 172,032 0.031 1.40 (11,H,F) 26.99 0.865 0.140 315,392 0.051 1.59 (24,H,F) 27.38 0.882 0.126 688,128 0.123 2.01

- Table 3. Quantitative comparisons on the DL3DV dataset under the 50-frame baseline setting (256×448). Inference time and memory consumption are measured on a RTX 4090 GPU.

[Figure 6]

Figure 6. Qualitative comparison on DL3DV dataset (256×448).

- 4.1. Results

In Tab. 1, 2 and Fig. 5, we compare our approach with feed-forward methods on the RE10K dataset and crossdataset generalization on ACID and DL3DV. Furthermore, we report results with an increased number of input views (4 and 8), which incur less than half of the computation time compared to the baseline (DepthSplat) while achieving superior performance. For the DL3DV dataset, our method consistently outperforms the baselines across various viewpoint and resolution configurations, including inference speed and memory usage, while achieving efficient scene representation with fewer Gaussians, as shown in Tab. 3, 4 and Fig. 6. While DepthSplat and our method are both trained under varying numbers of input views, our

[Figure 7]

- Figure 7. Qualitative comparison on undistorted DL3DV dataset under the wide-baseline setting (32 input images, 540×960, zero-shot).

[Figure 8]

- Figure 8. For the colored query patches in the reference viewpoint (red, yellow, green), we visualize top-3 attended tokens from other viewpoints throughout the iterative refinement process.

approach demonstrates enhanced scalability with respect to the increasing number of views in Tab. 3.

Method Views PSNR ↑ SSIM ↑ LPIPS ↓ # Gaussians Time (s) Memory (GB) DepthSplat 12 21.38 0.739 0.265 5,898,240 - OOM Ours (12,H,F) 24.35 0.781 0.256 1,474,560 0.415 3.53

- Table 4. Quantitative comparisons on the DL3DV dataset under the 100-frame baseline setting (512×960). Inference time and memory consumption are measured on a single RTX 4090 GPU. Since DepthSplat encounters out-of-memory (OOM) issue on the device, we evaluate its performance using a H100 GPU.

Method Views Time ↓ PSNR ↑ SSIM ↑ LPIPS ↓ 3D-GS 16 8min 21.48 0.753 0.252 Long-LRM 16 0.50sec 22.66 0.740 0.292 Ours (16,H,F) 0.19sec 22.91 0.766 0.295 3D-GS 32 8min 24.43 0.827 0.191 Long-LRM 32 0.84sec 23.97 0.778 0.267 Ours (32,H,F) 0.53sec 24.30 0.803 0.256 Long-LRM10 32 11sec 25.56 0.826 0.237 Ours10 (32,H,F) 4.5sec 25.67 0.844 0.230 Long-LRM (Unseen) 40 1.05sec 24.18 0.787 0.260 Ours (Unseen) (40,H,F) 0.76sec 24.54 0.811 0.248 Long-LRM (Unseen) 48 1.38sec 24.30 0.797 0.252 Ours (Unseen) (48,H,F) 1.04sec 24.78 0.820 0.240

- Table 5. Quantitative comparisons on the undistorted DL3DV dataset (540×960). We utilized flash attention v3 [46] using a H100 GPU. We re-evaluate Long-LRM [67] with their official checkpoint except for 16-view metrics (16-view weights are not released).

In the wide-coverage setting (Tab. 5), we evaluate performance using various numbers of high-resolution input images under full-frame coverage. For comparison, we also include optimization-based 3D-GS [30] trained for 30k iterations using the input images and camera poses. LongLRM10 means finetuning 10 epochs initialized from the Long-LRM’s generated Gaussians. Since our approach produces more compact 3D Gaussian representations (4× fewer), the finetuning process is significantly faster than the baseline. We further evaluate longer-context generalization ability (40 and 48 views) using a model trained with 32 views. Our method achieves better performance and faster inference across all metrics and scales more favorably with the number of views while maintaining compact scenes.

#### 4.2. Attention Visualization

We investigate how our method achieves global consistency throughout the iterative refinement process. Using the first input view as the reference, we select three query patches from its viewpoint embedding, and visualize top-3 attended tokens in the other viewpoint embeddings. For ease of visualization, we project the selected tokens onto the corresponding images via spatial upsampling. As shown in Fig. 8, attended tokens from other viewpoints gradually shifts toward geometrically and semantically corresponding regions as the layers go deeper, demonstrating the progressive, iterative refinement of the multi-view scene representation, which aligns our proposed design motivation.

# Params PSNR ↑ SSIM ↑ LPIPS ↓

12 layers (base) 185M 29.24 0.907 0.109 9 layers 139M 29.01 0.903 0.112 6 layers 94M 28.68 0.898 0.116 3 layers 48M 28.04 0.887 0.126

Table 6. Ablations on model size.

#### 4.3. Computational Costs of Training

We report detailed comparisons of computational costs during training in Tab. 8. The iteration time is measured under the same setting: half-resolution 8 viewpoints (8,H,F), and a batch size of 16 on a single RTX 4090 GPU. For memory comparison, to provide a clearer analysis, all models are run without gradient checkpointing on a single H100 GPU. Lastly, we present a theoretical comparison of FLOPs that further underscores the efficiency of our method, with only a marginal drop in performance. For detailed calculations of FLOPs, please refer to our supplementary material.

Method PSNR ↑ SSIM ↑ LPIPS ↓ Iteration (s) Memory (GB) GFLOPs

Baseline 30.39 0.923 0.095 1.51 62.5 3.83 w/ Half Cross-attn 30.25 0.922 0.096 1.13 47.4 1.71 w/ Quarter Cross-attn 30.08 0.919 0.098 0.94 39.0 0.81

Table 8. Quantitative comparison of our mini-batch cross-attention on the RE10K dataset, with iteration time and memory consumption measured during training.

#### 4.4. Ablations and Analysis

Tab. 6 presents the ablations on the number of update layers. All variants are trained under half-resolution 4 viewpoints setting (4,H,F), on the RE10K dataset with batch size 16. The results demonstrate consistent performance gains as the number of layers increases. From the perspective of the iterative refinement procedure, increasing the number of layers can be interpreted as introducing more optimization steps, which aligns with our intuition that deeper refinement leads to more accurate 3D representations.

In Tab. 7, we report the ablation results on architectural components. All experiments follow the model-size ablation training setup with a 12-layer baseline. More extensive ablation studies are provided in the supplementary material.

- 1) Iterative refinement. The cross-attention blocks in our model keep providing visual evidence (image) into the viewpoint tokens as part of the iterative refinement process. We validate this by replacing per-layer cross-attention with a single cross-attention in the first layer: the baseline has 12 layers (each with cross- then self-attention), while the variant has 1 cross-attention followed by 23 self-attention layers. The result shows that our consecutive cross-attention with image features plays a critical role in refining the viewpoint embeddings especially in terms of the LPIPS metric.
- 2) Resolution decoupling. Our design decouples image resolution from the viewpoint representation, so crossattention consumes high-resolution image features while

PSNR ↑ SSIM ↑ LPIPS ↓

Baseline (12 layers) 29.24 0.907 0.109 w/o iter. refinement 28.58 0.893 0.127

w/o resolution decoupling 28.47 0.891 0.123 w/o token uplifting 28.90 0.901 0.113

Table 7. Ablations on model architecture.

the scene tokens remain lightweight. When image features are constrained to the viewpoint resolution (prior approaches [63, 67]), performance drops, indicating that resolution decoupling is essential for simultaneously preserving compactness and high-fidelity reconstructions.

3) Token uplifting. Removing the token uplifting mechanism leads to a drop in performance across all metrics compared to baseline. This validates the importance of expanding low-resolution view tokens before cross-attention with high-resolution image tokens. Without this step, the model struggles to capture fine-grained spatial correspondences, resulting in a degraded reconstruction quality.

### 5. Limitations

One limitation of this work is the self-attention bottleneck across many input views. While our compact viewpoint embeddings substantially reduce the computational cost, challenges may arise as the number of input views increases considerably. In this study, aiming for scalable feedforward 3D models, we present the first implementation of the framework that iteratively refines 3D representations by leveraging high-resolution image information at every layer. Further development of more scalable alternatives [12, 13] would be a valuable direction for future research.

A second limitation is the reliance on accurate camera poses [42, 44] in static scenarios. Furthermore, because our primary goal is high fidelity novel view synthesis with rendering supervision, the recovered geometry may be less accurate than explicit geometry-supervised methods [51, 53]. Even so, our scalable and flexible structure provides a natural basis for relaxing these assumptions, as joint pose refinement [64] or even pose-free variants [23, 26, 28, 60] could be realized by training on suitable datasets and supervisions without modifying the core architectural design.

### 6. Conclusion

In this work, we present an iterative Large 3D Reconstruction Model (iLRM), a feed-forward architecture that reflects per-scene optimization-based schemes, by stacking multiple update layers composed of cross- and self-attention modules. By decoupling Gaussian representations from input images and splitting the update mechanism into perview interactions with image features and global aggregation over compact viewpoint embeddings, iLRM enables efficient, scalable, and high-quality 3D reconstruction across diverse scenes. We believe that iLRM lays a strong foundation for future research in feed-forward 3D reconstruction.

### Acknowledgements

This work was supported by Samsung Research Funding & Incubation Center of Samsung Electronics under Project Number SRFC-IT2401-01, the Artificial Intelligence Industrial Convergence Cluster Development Project funded by the Ministry of Science and ICT (MSIT, Korea) & Gwangju Metropolitan City, and the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korean government (MSIT) under the following projects: (No. RS-2024-00457882, AI Research Hub Project); (No. RS-2025-25441838, Development of a human foundation model for human-centric universal artificial intelligence and training of personnel); (No. RS-2020-II201361, Artificial Intelligence Graduate School Program (Yonsei University)); and (No. RS-202502653113, High-Performance Research AI Computing Infrastructure Support at the 2 PFLOPS Scale).

### References

- [1] Dejan Azinovi´c, Ricardo Martin-Brualla, Dan B Goldman, Matthias Nießner, and Justus Thies. Neural rgb-d surface reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6290– 6301, 2022. 14
- [2] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450,

2016. 6, 12

- [3] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5470–5479, 2022. 2, 14, 15, 16
- [4] Eric R Chan, Connor Z Lin, Matthew A Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas J Guibas, Jonathan Tremblay, Sameh Khamis, et al. Efficient geometry-aware 3d generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16123–16133, 2022. 3
- [5] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 1, 2, 3, 6, 15
- [6] Anpei Chen, Haofei Xu, Stefano Esposito, Siyu Tang, and Andreas Geiger. Lara: Efficient large-baseline radiance fields. In European Conference on Computer Vision, pages 338–355. Springer, 2024. 3
- [7] Qifeng Chen and Vladlen Koltun. Photographic image synthesis with cascaded refinement networks. In Proceedings of the IEEE international conference on computer vision, pages 1511–1520, 2017. 5
- [8] Yun Chen, Jingkang Wang, Ze Yang, Sivabalan Manivasagam, and Raquel Urtasun. G3r: Gradient guided generalizable reconstruction. In European Conference on Computer Vision, pages 305–323. Springer, 2024. 3

- [9] Yihang Chen, Qianyi Wu, Weiyao Lin, Mehrtash Harandi, and Jianfei Cai. Hac: Hash-grid assisted context for 3d gaussian splatting compression. In European Conference on Computer Vision, pages 422–438. Springer, 2024. 2
- [10] Yuedong Chen, Chuanxia Zheng, Haofei Xu, Bohan Zhuang, Andrea Vedaldi, Tat-Jen Cham, and Jianfei Cai. Mvsplat360: Feed-forward 360 scene synthesis from sparse views. arXiv preprint arXiv:2411.04924, 2024. 1, 3
- [11] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In European Conference on Computer Vision, 2025. 1, 2, 3, 4, 6, 12, 15, 16
- [12] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019. 8
- [13] Tri Dao and Albert Gu. Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. In International Conference on Machine Learning, 2024. 8
- [14] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. 2
- [15] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. Objaverse-xl: A universe of 10m+ 3d objects. Advances in Neural Information Processing Systems, 2024. 2
- [16] Johan Edstedt, Qiyu Sun, Georg B¨okman, M˚arten Wadenb¨ack, and Michael Felsberg. Roma: Robust dense feature matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19790– 19800, 2024. 15
- [17] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, Zhangyang Wang, et al. Lightgaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps. Advances in neural information processing systems, 37: 140138–140158, 2024. 2
- [18] Guangchi Fang and Bing Wang. Mini-splatting: Representing scenes with a constrained number of gaussians. In European Conference on Computer Vision, 2024. 1
- [19] John Flynn, Michael Broxton, Lukas Murmann, Lucy Chai, Matthew DuVall, Cl´ement Godard, Kathryn Heal, Srinivas Kaza, Stephen Lombardi, Xuan Luo, et al. Quark: Real-time, high-resolution, and general neural view synthesis. ACM Transactions on Graphics, 2024. 3
- [20] Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. 3
- [21] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016. 6
- [22] Alex Henry, Prudhvi Raj Dachapally, Shubham Pawar, and Yuxuan Chen. Query-key normalization for transformers. arXiv preprint arXiv:2010.04245, 2020. 6, 12

- [23] Sunghwan Hong, Jaewoo Jung, Heeseong Shin, Jisang Han, Jiaolong Yang, Chong Luo, and Seungryong Kim. Pf3plat: Pose-free feed-forward 3d gaussian splatting. arXiv preprint arXiv:2410.22128, 2024. 8
- [24] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400, 2023. 3
- [25] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In International conference on machine learning, pages 4651–4664. PMLR, 2021. 2
- [26] Hanwen Jiang, Hao Tan, Peng Wang, Haian Jin, Yue Zhao, Sai Bi, Kai Zhang, Fujun Luan, Kalyan Sunkavalli, Qixing Huang, and Georgios Pavlakos. Rayzer: A selfsupervised large view synthesis model. arXiv preprint arXiv:2505.00702, 2025. 8
- [27] Haian Jin, Hanwen Jiang, Hao Tan, Kai Zhang, Sai Bi, Tianyuan Zhang, Fujun Luan, Noah Snavely, and Zexiang Xu. Lvsm: A large view synthesis model with minimal 3d inductive bias. arXiv preprint arXiv:2410.17242, 2024. 2, 3, 4, 12
- [28] Gyeongjin Kang, Jisang Yoo, Jihyeon Park, Seungtae Nam, Hyeonsoo Im, Sangheon Shin, Sangpil Kim, and Eunbyung Park. Selfsplat: Pose-free and 3d prior-free generalizable 3d gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22012–22022,

2025. 8

- [29] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019. 3
- [30] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics,

2023. 1, 2, 6, 7, 12, 14, 15

- [31] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics, 36(4), 2017. 14, 15, 16
- [32] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko, and Eunbyung Park. Compact 3d gaussian representation for radiance field. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21719– 21728, 2024. 2
- [33] Joo Chan Lee, Jong Hwan Ko, and Eunbyung Park. Optimized minimal 3d gaussian splatting. arXiv preprint arXiv:2503.16924, 2025. 2
- [34] Zhengqi Li, Wenqi Xian, Abe Davis, and Noah Snavely. Crowdsampling the plenoptic function. In Computer Vision– ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16, pages 178–196. Springer, 2020. 5
- [35] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep

- learning-based 3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 2, 3, 6, 12, 13, 14, 16
- [36] Andrew Liu, Richard Tucker, Varun Jampani, Ameesh Makadia, Noah Snavely, and Angjoo Kanazawa. Infinite nature: Perpetual view generation of natural scenes from a single image. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021. 6
- [37] Tianqi Liu, Guangcong Wang, Shoukang Hu, Liao Shen, Xinyi Ye, Yuhang Zang, Zhiguo Cao, Wei Li, and Ziwei Liu. Mvsgaussian: Fast generalizable gaussian splatting reconstruction from multi-view stereo. In European Conference on Computer Vision, pages 37–53. Springer, 2024. 3
- [38] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 12
- [39] Tao Lu, Ankit Dhiman, R Srinath, Emre Arslan, Angela Xing, Yuanbo Xiangli, R Venkatesh Babu, and Srinath Sridhar. Turbo-gs: Accelerating 3d gaussian fitting for highquality radiance fields. arXiv preprint arXiv:2412.13547,

2024. 1

- [40] Saswat Subhajyoti Mallick, Rahul Goel, Bernhard Kerbl, Markus Steinberger, Francisco Vicente Carrasco, and Fernando De La Torre. Taming 3dgs: High-quality radiance fields with limited resources. In SIGGRAPH Asia 2024 Conference Papers, 2024. 1, 2
- [41] Seungtae Nam, Xiangyu Sun, Gyeongjin Kang, Younggeun Lee, Seungjun Oh, and Eunbyung Park. Generative densification: Learning to densify gaussians for high-fidelity generalizable 3d reconstruction. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26683– 26693, 2025. 3, 6, 15, 16
- [42] Linfei Pan, D´aniel Bar´ath, Marc Pollefeys, and Johannes L Sch¨onberger. Global structure-from-motion revisited. In European Conference on Computer Vision, pages 58–77. Springer, 2024. 8
- [43] Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, and Fisher Yu. Unidepth: Universal monocular metric depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10106–10116, 2024. 3
- [44] Johannes L. Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR),

2016. 8

- [45] Thomas Sch¨ops, Johannes L. Sch¨onberger, Silvano Galliani, Torsten Sattler, Konrad Schindler, Marc Pollefeys, and Andreas Geiger. A multi-view stereo benchmark with highresolution images and multi-camera videos. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2538–2547, 2017. 14
- [46] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. Flashattention-3: Fast and accurate attention with asynchrony and low-precision. Advances in Neural Information Processing Systems, 37: 68658–68685, 2024. 7, 14, 15
- [47] Stanislaw Szymanowicz, Eldar Insafutdinov, Chuanxia Zheng, Dylan Campbell, Joao F Henriques, Christian Rup-

- precht, and Andrea Vedaldi. Flash3d: Feed-forward generalisable 3d scene reconstruction from a single image. arXiv preprint arXiv:2406.04343, 2024. 3
- [48] Stanislaw Szymanowicz, Chrisitian Rupprecht, and Andrea Vedaldi. Splatter image: Ultra-fast single-view 3d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 2, 3
- [49] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. In European Conference on Computer Vision, 2025. 1, 3, 4
- [50] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 6
- [51] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025. 8, 13
- [52] Ruicheng Wang, Sicheng Xu, Yue Dong, Yu Deng, Jianfeng Xiang, Zelong Lv, Guangzhong Sun, Xin Tong, and Jiaolong Yang. Moge-2: Accurate monocular geometry with metric scale and sharp details. arXiv preprint arXiv:2507.02546,

2025. 14

- [53] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697– 20709, 2024. 8
- [54] Yunsong Wang, Tianxin Huang, Hanlin Chen, and Gim Hee Lee. Freesplat: Generalizable 3d gaussian splatting towards free view synthesis of indoor scenes. Advances in Neural Information Processing Systems, 37, 2025. 2
- [55] Christopher Wewer, Kevin Raj, Eddy Ilg, Bernt Schiele, and Jan Eric Lenssen. latentsplat: Autoencoding variational gaussians for fast generalizable 3d reconstruction. In European Conference on Computer Vision, pages 456–473. Springer, 2024. 3
- [56] Haofei Xu, Anpei Chen, Yuedong Chen, Christos Sakaridis, Yulun Zhang, Marc Pollefeys, Andreas Geiger, and Fisher Yu. Murf: Multi-baseline radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 3
- [57] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys. Depthsplat: Connecting gaussian splatting and depth. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 16453–16463, 2025. 1, 2, 3, 4, 6, 12, 15, 16
- [58] Yinghao Xu, Zifan Shi, Wang Yifan, Hansheng Chen, Ceyuan Yang, Sida Peng, Yujun Shen, and Gordon Wetzstein. Grm: Large gaussian reconstruction model for efficient 3d reconstruction and generation. arXiv preprint arXiv:2403.14621, 2024. 1, 3
- [59] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing

- the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 3, 14
- [60] Botao Ye, Sifei Liu, Haofei Xu, Xueting Li, Marc Pollefeys, Ming-Hsuan Yang, and Songyou Peng. No pose, no problem: Surprisingly simple 3d gaussian splats from sparse unposed images. arXiv preprint arXiv:2410.24207, 2024. 6, 8, 15
- [61] Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen, Brent Yi, Zhuoyang Pan, Otto Seiskari, Jianbo Ye, Jeffrey Hu, Matthew Tancik, et al. gsplat: An open-source library for gaussian splatting. Journal of Machine Learning Research, 26(34):1–17, 2025. 12
- [62] Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in Neural Information Processing Systems, 2019. 6
- [63] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, and Zexiang Xu. Gs-lrm: Large reconstruction model for 3d gaussian splatting. In European Conference on Computer Vision, 2025. 1, 2, 3, 4, 6, 8, 12, 15
- [64] Shangzhan Zhang, Jianyuan Wang, Yinghao Xu, Nan Xue, Christian Rupprecht, Xiaowei Zhou, Yujun Shen, and Gordon Wetzstein. Flare: Feed-forward geometry, appearance and camera estimation from uncalibrated sparse views. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21936–21947, 2025. 8
- [65] Shunyuan Zheng, Boyao Zhou, Ruizhi Shao, Boning Liu, Shengping Zhang, Liqiang Nie, and Yebin Liu. Gpsgaussian: Generalizable pixel-wise 3d gaussian splatting for real-time human novel view synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19680–19690, 2024. 2
- [66] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. arXiv preprint arXiv:1805.09817, 2018. 2, 3, 6, 12, 13, 15, 16
- [67] Chen Ziwen, Hao Tan, Kai Zhang, Sai Bi, Fujun Luan, Yicong Hong, Li Fuxin, and Zexiang Xu. Long-lrm: Longsequence large reconstruction model for wide-coverage gaussian splats. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025. 2, 3, 6, 7, 8, 12, 13, 14, 15

## iLRM: An Iterative Large 3D Reconstruction Model Supplementary Material

### A. Additional Implementation Details

We initialize model weights using a zero-mean normal distribution with a standard deviation of 0.02. Bias terms are omitted in all Linear and normalization layers. The model is trained using the AdamW [38] optimizer with hyperparameters β1 = 0.9 and β2 = 0.95. A weight decay of 0.05 is applied to all parameters except the weights of LayerNorm [2]. We use a cosine learning rate schedule with a peak learning rate of 2e-4 and a warmup of 2500 iterations. Our training setup largely follows the configuration proposed in [27, 63].

For the RealEstate10K (RE10K) [66] dataset, the 8-view half-resolution viewpoint setting (8, H, F) is trained on 8 H100 GPUs with a total batch size of 256 for 50,000 iterations. Similarly, the 4-view half-resolution viewpoint setting (4, H, F) is trained on 8 RTX 4090 GPUs with a total batch size of 128 for 100,000 iterations. The minibatch cross-attention variants were also trained with the equivalent computational budgets for each viewpoint setting. Lastly, the 2-view full-resolution viewpoint setting (2, F, F), which serves as the reference point, is trained on 8 H100 GPUs for 200,000 iterations.

There are two variants in the DL3DV dataset [35]. 1) For comparison with MVSplat [11] and DepthSplat [57], we initialize from the pretrained (8, H, F) model trained on the RE10K dataset, and finetune it on 8 H100 GPUs with a total batch size of 96 for 100,000 iterations for LR (256×448), and additional 50,000 iterations for HR (512×960). During training, the number of input viewpoints is randomly sampled between 6 and 11 to expose the model to varying numbers of viewpoints/images. Following this stage, the model is further finetuned under the high-resolution setting (512×960). 2) For comparison with LongLRM [67], which incorporates an undistortion preprocessing step, we adopt the training protocol described in the original work, using 8 H200 GPUs. The training resolution is scheduled in a curriculum of 256×256, 512×512, and 540×960.

Gaussian representations. After the final self-attention layer, the viewpoint features are decoded into Gaussian parameters using a single linear layer with an output dimension of 16. The Gaussian positions, denoted as µ, consist of 5 channels: 2 for the spatial xy offset and 3 for depth, z. The final depth is obtained by averaging the 3 depth channels. Opacity (α) is represented by a single channel. Covariance (Σ) is derived from 3 channels of scale and 4 channels of rotation. Finally, color (c) is represented using

- 3 channels. Higher-order spherical harmonics coefficients are not used in our method. The post-activation functions for each parameter follow the design of GS-LRM [63], ex-

cept for the spatial xy offset, for which we constrain the range to lie within a single pixel of viewpoint resolution. We utilize gsplat [61], an open-source library for Gaussian Splatting [30] for a rasterizer. In the post-prediction optimization, we use a learning rate of 6e−4 for the positions and 1e−3 for the other attributes.

Camera pose normalization. We normalize camera poses to align the scene into a consistent coordinate system and scale. First, we compute the average position and viewing directions (forward, down, and right) from the input camera extrinsics. These are used to build a new reference pose, which centers and aligns the scene. All camera extrinsics are then transformed into this reference frame. Finally, we scale the entire scene so that the largest camera distance is 1, ensuring the scene fits within a normalized space [63].

### B. Additional Architectural Details

We provide the detailed figure of our token uplifting module in Fig. 9. Note that, to balance the model’s representational capacity and computational efficiency, the length of the lowresolution viewpoint embeddings does not exceed that of the high-resolution image features.

[Figure 9]

Figure 9. Architectural details of token uplifting.

Tokenization and normalization. After tokenizing the viewpoints and multi-view images using linear layers, both types of tokens are passed through a LayerNorm [2]. In each cross-attention layer, only the viewpoint tokens are further processed with a pre-normalization layer. Additionally, after the query and key linear projections, both tokens are passed through an extra normalization layer, referred to as the QK-Norm [22].

### C. Additional Evaluation Details

When we utilize more input viewpoints (more than two in RealEstate10K [66] experiment compared to the baselines), we sample additional viewpoints/images evenly between the two endpoint indices, ensuring that these samples do not overlap with the target indices. For cross-dataset generalization on the DL3DV dataset, we use a baseline of 12 frames.

In wide-baseline setting, every 8th image in the sequence is reserved for the test split, while K-means clustering on camera positions and viewing directions is applied to the remaining images to select input views that ensure wide scene coverage [67].

PSNR ↑ SSIM ↑ LPIPS ↓ Baseline 29.24 0.907 0.109

w/o self-attention 23.33 0.755 0.220 w/ group-attention 29.02 0.904 0.112

w/ random init. 28.90 0.902 0.112 w/ LR-feature init. 28.35 0.894 0.121

Table 9. Additional ablations on model architecture.

### D. Additional Ablations on Model Architecture

We provide additional ablation studies and analyses in Tab. 9 under the same configuration as the ablations on model architecture in main script. All variants are trained under half-resolution 4 viewpoints setting (4,H,F), with a batch size of 16 on a single RTX 4090 GPU.

- 1) Self-attention. To ensure a fair comparison, we replaced all self-attention layers with cross-attention layers rather than simply removing them, maintaining a comparable parameter count. The performance dropped significantly, highlighting the essential role of self-attention in capturing global dependencies and enhancing multi-view awareness among viewpoint embeddings. Without selfattention, the model struggles to integrate contextual information across different viewpoints, resulting in poor convergence and reconstruction quality.
- 2) Group-attention This variant replaces the per-viewpoint cross-attention mechanism with a group-attention approach, where all viewpoint tokens and image tokens are concatenated and jointly processed through a cross-attention block. Unlike our default design, group-attention introduces global interactions across all views. While this mechanism can increase the expressive capacity between multiple viewpoints, it incurs quadratic complexity with respect to the number of views. However, the increased computational cost does not yield performance gains, suggesting that separating the

roles—using cross-attention for localized image-view interactions and self-attention for global refinement across viewpoints—leads to a more efficient and effective architecture, which is also validated as Alternating-Attention in VGGT [51].

3) Different initialization. We also investigate the different initialization methods of scene representation. For the random initizliation, we used a learnable embedding initialized with zero mean and 0.02 standard deviation, whereas for the LR feature variant, we used features extracted from low-resolution images. In the PSNR training curve, the LR feature variant rises more quickly in the early stages but is later surpassed by the random initialization variant. We believe this is because, in our iterative cross-attention architecture, high-resolution image features and camera information are continually provided by the cross-attention blocks. As a result, the learnable embedding can offer more flexible and informative parameters for guiding iterative updates, whereas the LR image features may introduce redundant and less discriminative information that limits longterm performance gains. Moreover, the use of LR features may bias the early attention stages toward coarse processing, which can hinder the model’s ability to fully refine fine details in later stages.

- E. Computational Costs of Training We provide a detailed theoretical calculation of the FLOPs for our mini-batch cross-attention mechanism. In this analysis, we limit the computation to a per-view, single crossattention operation, excluding our token uplifting strategy (as it introduces a constant cost across all variations). Given a viewpoint token of shape (Lv,D) and an image token of shape (Li,D), where Lv and Li denote the token lengths and D is the hidden dimension, the FLOPs for the crossattention operation are computed as:

4D2(Lv + Li) + 4LvLiD. Assuming a hidden dimension of D = 768, an image resolution of 256×256, and a viewpoint resolution of 128×128, with a patch size of 8 × 8, the token lengths are computed as Li = 1024 for the image tokens and Lv = 256 for the viewpoint tokens, based on the experimental configuration used in the RealEstate10K [66] dataset.

Thus, the computation becomes: baseline: 3.83 GFLOPs; half cross-attention: 1.71 GFLOPs; quarter cross-attention: 0.81 GFLOPs.

- F. Additional Quantitative Results

Wide-coverage baselines. We additionally evaluate our method on the recently released DL3DV [35] evaluation split, which comprises 51 scenes in our experiments. Our method achieves better reconstruction quality, faster inference, and stronger longer-context generalization, while its

compact 3D scene representations additionally enable fast rendering, as shown in Tab. 10.

Method Views Time ↓ PSNR ↑ SSIM ↑ LPIPS ↓ 3D-GS [30] 32 8min 25.09 0.838 0.175 Long-LRM [67] 32 0.84sec 23.54 0.776 0.270 Ours (32,H,F) 0.53sec 23.93 0.800 0.259 Long-LRM (Unseen) 16 0.50sec 20.65 0.707 0.328 Ours (Unseen) (16,H,F) 0.19sec 21.63 0.746 0.316 Long-LRM (Unseen) 40 1.05sec 23.76 0.785 0.262 Ours (Unseen) (40,H,F) 0.76sec 24.21 0.809 0.250 Long-LRM (Unseen) 48 1.38sec 23.88 0.795 0.255 Ours (Unseen) (48,H,F) 1.04sec 24.45 0.818 0.242

- Table 10. Quantitative comparisons on the undistorted DL3DV evaluation dataset (540×960). We utilized flash attention v3 [46] using a H100 GPU. Depth estimation. In addition to novel view synthesis, we further evaluate the rendered depth maps, which serve as a indicator measure for underlying geometric accuracy, on the DL3DV [35], Tanks&Temples (TNT) [31], and MipNeRF360 (Mip360) [3] dataset. Since these datasets do not provide ground-truth depth, we adopt the recent stateof-the-art monocular depth estimator, MoGe-2 [52], as a proxy to obtain pseudo depth. For each target view, we predict depth from the target image and its focal length, mask out invalid values in both rendered and pseudo depths, and then compute relative depth accuracy by comparing them using standard scale-invariant depth metrics. Tab. 11 shows that our method produces depth maps that are more consistent with the MoGe-2 predictions than those of the baseline, even though the baseline is additionally regularized using a pretrained depth estimation model [59] during training. We regard this evaluation as an indirect indicator of geometric quality in the absence of ground-truth depth. We also provide qualitative visualizations of the rendered depth maps in Fig. 10, where our method produces sharper, more detailed depth boundaries and fewer artifacts compared to the baseline. We attribute these improvements to our compact representation, which reduces redundancy and artifacts while better preserving fine geometric details.

Method

Views DL3DV-Benchmark DL3DV-Eval TNT&Mip360 Abs Rel ↓ RMSE ↓ Abs Rel ↓ RMSE ↓ Abs Rel ↓ RMSE ↓

Long-LRM [67] 16 0.718 1.427 0.768 1.631 0.603 1.199 Ours (16,H,F) 0.670 1.174 0.693 1.354 0.515 0.852

Long-LRM 32 0.759 1.571 0.805 1.788 0.645 1.345 Ours (32,H,F) 0.709 1.310 0.739 1.513 0.543 0.967

- Table 11. Quantitative comparison of depth estimation. We use the large MoGe-2 (ViT-L) variant as the pseudo-depth estimator.

Geometry estimation We evaluate the Chamfer Distance (CD) and F1-score with geometry datasets using NRGBD [1] and ETH3D [45]. For iLRM, ground-truth pointmaps are downsampled by half to match the generated point clouds. With Mip-NeRF360 dataset, we visualized the vertices after building mesh and removing flying points

[Figure 10]

Figure 10. Qualitative comparison of rendered depth maps. Examples from DL3DV (top two rows), Tanks&Temples (third row), and Mip-NeRF360 (bottom two rows) are shown.

(relative tolerance threshold of 0.04). iLRM shows better geometry with fewer points in Tab. 12 and Fig. 11.

NRGBD (16-view) NRGBD (32-view) ETH3D (16-view) ETH3D (32-view) CD ↓ F1-score ↑ CD ↓ F1-score ↑ CD ↓ F1-score ↑ CD ↓ F1-score ↑

Method

Long-LRM 0.53 0.52 0.43 0.59 2.75 0.32 2.69 0.39 Ours 0.50 0.60 0.52 0.69 2.06 0.50 1.15 0.54

Table 12. Pointmap estimation comparisons with a input resolution of 540 × 960 (# views). The F1-score threshold was set to 0.1.

[Figure 11]

Figure 11. Zero-shot colored vertices visualization.

Post-prediction optimization While the performance of zero-shot novel view synthesis (TNT, Mip360) in Tab. 13 is comparable to the baseline, our finer geometric estimation promote faster convergence during post-prediction optimization which reflects more reliable geometry in the initial 3D Gaussians. In 10-epoch optimization, our method already outperforms the baseline while using less than half

of the time, and with 20-epoch, still faster than the baseline, it achieves even higher accuracy. We attribute these gains to our compact scene representation and its stronger capacity for capturing underlying geometric structure. We provide qualitative comparisons on 10-epoch optimization in Fig. 12.

Tanks&Temples [31] Mip-NeRF360 [3] PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓

Method Views Time ↓

3D-GS [30] 32 8min 18.48 0.720 0.260 22.95 0.694 0.250 Long-LRM [67] 32 0.84sec 18.59 0.614 0.367 21.08 0.484 0.445 Ours (32,H,F) 0.53sec 18.58 0.631 0.385 21.09 0.495 0.466 Long-LRM10 32 11sec 19.23 0.663 0.348 22.05 0.554 0.414 Ours10 (32,H,F) 4.5sec 19.42 0.689 0.350 22.49 0.601 0.414 Ours20 (32,H,F) 8.6sec 19.62 0.704 0.338 22.85 0.622 0.397

Table 13. Quantitative comparisons on the Tanks&Temples and Mip-NeRF360 dataset (540×960). We adopt the settings reported in their paper for the post-prediction optimization of Long-LRM (learning rates of 5e−4 for position and 1e−3 for color). We utilized flash attention v3 [46] using a H100 GPU.

[Figure 12]

Figure 12. Qualitative comparison on Mip-NeRF360 (first four rows) and Tanks&Temples (bottom two rows) after 10-epoch of post-prediction optimization.

Input robustness. To assess the robustness of our model to imperfect camera pose estimates, we evaluate under translational camera pose perturbations on the 32-view DL3DV dataset. Gaussian noise is added to the camera translation vectors, and we report performance across varying noise to examine the degradation in rendering quality (Tab. 14).

Inference time. We compare the inference time of our

0 0.001 0.005 PSNR LPIPS PSNR LPIPS PSNR LPIPS

stand. dev.

Long-LRM 23.97 0.267 23.27 (-0.70) 0.287 (+0.020) 20.49 (-3.48) 0.381 (+0.114) iLRM 24.30 0.256 23.82 (-0.48) 0.270 (+0.014) 21.49 (-2.81) 0.352 (+0.096)

- Table 14. Robustness evaluation under translational camera pose perturbations on the 32-view DL3DV (540×960) dataset.

model across different numbers of input views at a resolution of 540×960, using a single H100 GPU with flash attention v3 [46]. As shown in Tab. 15, our method achieves lower latency than Long-LRM across all comparable settings. Notably, Long-LRM runs out of memory at 256 input views, whereas our model still completes inference within a practical time budget, indicating better scalability of the proposed compact viewpoint representation to large number of views.

Method 16 32 64 96 128 256 Long-LRM [67] 0.5 0.84 2.08 3.90 6.39 Out-of-memory Ours 0.19 0.53 1.66 3.37 5.61 20.92

- Table 15. Quantitative comparisons of inference time across different numbers of input views. All times are measured in seconds.

Varying baseline range. We compare our model against recent generalizable 3D reconstruction methods [11, 41, 57] on the RealEstate10K [66] dataset, with a particular focus on handling varying degrees of camera overlap [60]. These overlap categories are determined using the dense feature matching method RoMA [16]. As shown in Tab. 16, our method, which efficiently handles a large number of input viewpoints/images, achieves superior performance compared to existing approaches, especially in challenging cases with small viewpoint overlap.

Same number of Gaussians. We also validate the strength of our decoupling strategy in leveraging high-resolution images as visual cues while generating efficient and compact 3D Gaussians. As discussed in our motivation, previous methods [5, 11, 41, 57, 63] require downsampling the input images to reduce the number of generated Gaussians, inherently coupling image resolution with representation density. To demonstrate the flexibility of our approach, we conduct an experiment in which all methods generate the same number of Gaussians using 4 viewpoints at half resolution. Specifically, the baseline methods [11, 57] follow a (4,H,H) configuration, where both the number of viewpoints and image resolution are reduced. In contrast, our method adopts a (4,H,F) setting, where we preserve high-resolution image inputs while generating lowresolution Gaussians, thanks to our decoupled design. As shown in Tab. 17, our method surpasses the baselines in performance while requiring fewer computational resources in training, and faster inference speed, highlighting the practi-

Small Medium Large Average Method PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓

MVSplat [11] 20.37 0.725 0.250 23.81 0.814 0.172 27.47 0.885 0.115 24.01 0.812 0.175 DepthSplat [57] 22.82 0.798 0.193 25.38 0.851 0.145 28.32 0.900 0.104 25.59 0.852 0.145 Gen-Den [41] 21.10 0.744 0.234 24.57 0.828 0.162 28.26 0.895 0.108 24.77 0.826 0.164 Ours (2,F,F) 23.82 0.813 0.184 26.54 0.864 0.139 29.43 0.910 0.103 26.70 0.864 0.140

Ours (4,H,F) 27.65 0.887 0.127 29.13 0.908 0.108 30.73 0.926 0.092 29.22 0.908 0.108 Ours-MC (4,H,F) 27.41 0.882 0.131 28.87 0.904 0.111 30.44 0.927 0.095 28.96 0.904 0.112 Ours (8,H,F) 29.44 0.912 0.106 30.51 0.925 0.093 31.77 0.937 0.080 30.61 0.925 0.092 Ours-MC (8,H,F) 29.15 0.908 0.108 30.20 0.922 0.094 31.46 0.935 0.082 30.30 0.922 0.094

Table 16. Quantitative comparisons on the RE10K dataset under varying view overlap conditions.

Method Params (M) Train GPU (#) PSNR ↑ SSIM ↑ LPIPS ↓ # Gaussians Time (s) Memory (GB)

MVSplat [11] 12 H100 (1) 27.53 0.889 0.116 65,536 0.048 0.65 DepthSplat [57] 354 H100 (1) 28.08 0.898 0.107 65,536 0.062 2.49

- Ours 185 RTX 4090 (1) 29.24 0.907 0.109 65,536 0.027 1.22
- Ours 185 RTX 4090 (2) 29.82 0.916 0.101 65,536 0.027 1.22

- Table 17. Quantitative comparisons under the same number of Gaussians on the RE10K dataset. Inference time and memory consumption are measured only during the Gaussian generation stage, excluding the rendering process on a RTX 4090 GPU.

cal advantages of our design. This result demonstrates the efficiency and the representational ability of our architecture, which effectively utilizes high-resolution visual cues, leading to superior reconstruction quality under the same output density without requiring expensive hardware. To train the baseline methods effectively with a large batch size (similar to ours), we run them on a single H100 GPU. Our method and MVSplat [11] are trained with a batch size of 16, while DepthSplat [57] is trained with a batch size of 12 due to memory constraints.

Quarter resolution baselines. To evaluate different viewpoint configurations—specifically the resolution of each viewpoint—we additionally compare a quarter-resolution variant of the viewpoint inputs. All experiments in Tab. 18 are conducted using a single RTX 4090 GPU with a batch size of 16, 12 update layers, and trained for 100,000 iterations. While lowering the resolution of viewpoint inputs leads to a moderate drop in reconstruction quality, it significantly reduces the number of generated Gaussians, showing trade-off between accuracy and efficient representations.

Method PSNR ↑ SSIM ↑ LPIPS ↓ # Gaussians Ours (8,H,F) 30.39 0.923 0.095 131,072 Ours (4,H,F) 29.24 0.907 0.109 65,536 Ours (8,Q,F) 27.36 0.868 0.152 32,768 Ours (4,Q,F) 26.40 0.843 0.177 16,384

- Table 18. Quantitative comparisons of different viewpoint configurations on the RE10K dataset. Q denotes quarter resolution compared to the original image resolution.

### G. Additional Qualitative Results

We present additional qualitative results in Fig. 13 for the RealEstate10K (RE10K) [66] dataset and in Fig. 14, 15, 16 and 17 for the DL3DV [35], Tanks&Temples [31], and MipNeRF360 [3] dataset. Also, we provide additional attention visualization in Fig. 18. Further details for each example are provided in the corresponding captions in figures.

[Figure 13]

Figure 13. Qualitative comparison on the RE10K dataset (2 input images except for “Ours(4, H, F)” and “Ours(8, H, F)”, 256×256).

[Figure 14]

- Figure 14. Qualitative comparison on the DL3DV dataset under the 50-frame baseline setting (6 input images, 256×448).

[Figure 15]

- Figure 15. Qualitative comparison on the DL3DV dataset under the 100-frame baseline setting (12 input images, 512×960).
- Figure 16. Qualitative comparison under the wide-baseline setting (32 input images, 540×960, zero-shot). DL3DV (top three rows), Tanks&Temples (fourth row), and Mip-NeRF360 (bottom row) are shown.

[Figure 16]

[Figure 17]

###### Figure 17. Qualitative visualization of rendered color and depth maps from novel viewpoints (32 input images, 540×960, zero-shot). Scenes from DL3DV (top two example), Tanks&Temples (third example), and Mip-NeRF360 (bottom example) are shown.

[Figure 18]

- Figure 18. For the colored query patches in the reference viewpoint (red, yellow, green), we visualize top-3 attended tokens from other viewpoints throughout the iterative refinement process. For relatively easy cases with small camera motion or distinctive regions, the model identifies the correct correspondences in the early layers, whereas for more challenging cases with larger viewpoint changes, the attention gradually converges to geometrically consistent regions as the refinement progresses. Scenes from DL3DV (top two rows), Tanks&Temples (third row), and Mip-NeRF360 (bottom row) are shown.

