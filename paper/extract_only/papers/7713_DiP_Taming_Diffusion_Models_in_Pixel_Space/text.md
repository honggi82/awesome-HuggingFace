## DiP: Taming Diffusion Models in Pixel Space

Zhennan Chen1,2* Junwei Zhu2† Xu Chen2 Jiangning Zhang2 Xiaobin Hu3 Hanzhen Zhao3 Chengjie Wang2 Jian Yang1 Ying Tai1‡

1Nanjing University 2Tencent Youtu Lab 3National University of Singapore

https://github.com/NJU-PCALab/DiP

### Abstract

😞 Informa on loss , Non-end-to-end training

[Figure 1]

[Figure 2]

[Figure 3]

# arXiv:2511.18822v3[cs.CV]26Mar2026

[Figure 4]

Diffusion models face a fundamental trade-off between generation quality and computational efficiency. Latent Diffusion Models (LDMs) offer an efficient solution but suffer from potential information loss and non-end-to-end training. In contrast, existing pixel space models bypass VAEs but are computationally prohibitive for highresolution synthesis. To resolve this dilemma, we propose DiP, an efficient pixel space diffusion framework. DiP decouples generation into a global and a local stage: a Diffusion Transformer (DiT) backbone operates on large patches for efficient global structure construction, while a co-trained lightweight Patch Detailer Head leverages contextual features to restore fine-grained local details. This synergistic design achieves computational efficiency comparable to LDMs without relying on a VAE. DiP is accomplished with up to 10× faster inference speeds than previous method while increasing the total number of parameters by only 0.3%, and achieves an 1.79 FID score on ImageNet 256×256.

[Figure 5]

VAE

VAE

DiT

…

…

[Figure 6]

- (a) Vanilla Latent Diffusion Model
- (b) Vanilla Pixel Diﬀusion Model

[Figure 7]

😞 Longer sequences, higher complexity

[Figure 8]

| | | | | |[Figure 9]| | |patchfy|
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

[Figure 10]

[Figure 11]

[Figure 12]

Projectout

Projectin

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

unpatchfy

DiT

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

…

…

…

…

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

###### 😊 End-to-end training, low computa onal cost, high quality

patch detailer head

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

…

coarse structural mode ﬁne detail mode

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Projectin

[Figure 43]

[Figure 44]

[Figure 45]

patchify DiT

unpatchify

…

…

…

…

[Figure 46]

[Figure 47]

[Figure 48]

(c) Ours

Figure 1. Comparison of vanilla latent diffusion model, vanilla pixel diffusion model and our method. Vanilla LDMs utilize VAEs to balance computational efficiency and generation quality. Vanilla pixel diffusion models use small patches to pursue detailed generation quality. Our method achieves high-quality generation while maintaining efficient end-to-end training in pixel space.

### 1. Introduction

Diffusion models [19, 20, 27, 38, 41, 44, 47–49, 57, 61] have reshaped the landscape of generative visual content. With its outstanding generative capabilities of fidelity and diversity, they have established new state-of-the-art benchmarks across a multitude of tasks, including image synthesis [3, 6–8, 11, 53, 60, 68, 70, 74, 75], video generation [13, 36, 64, 71], and 3D object creation [65–67], decisively surpassing prior paradigms like Generative Adversarial Networks (GANs) [14, 26, 35, 40, 69, 76]. However, this generative prowess is underpinned by immense computational demands. Consequently, the inherent trade-off between generation quality and computational efficiency thus stands as one of the most critical challenges in the field of diffusion models today.

high-resolution images into a compact latent space, LDMs significantly reduce the computational complexity of the iterative denoising process, as shown in Figure 1(a). Nevertheless, this approach is not without its limitations, including potential information loss [4, 17, 28, 59] during VAE compression and a non-end-to-end training pipeline.

The most direct solution to eliminate the shortcomings of LDMs is to train a diffusion model in pixel space. However, existing pixel space diffusion models [5, 9, 22, 23, 29, 50], particularly those based on the powerful Transformer architecture [52], face a severe scalability issue. As shown in Figure 1(b), to capture fine-grained details, they typically rely on small input patches (e.g. 2×2 or 4×4), causing the input sequence length to grow quadratically with image resolution. This quadratic scaling renders high-resolution

To mitigate this challenge, Latent Diffusion Models (LDMs) [42] have emerged as the de facto standard. By employing a pre-trained autoencoder (VAE) [30] to compress

*Work done during the internship at Tencent. †Project Leader. ‡Corresponding Author.

[Figure 49]

Figure 2. Our method achieves the best FID score with minimal computational cost. (Note: LDM latency includes VAE. The methods marked with dashed lines ( ) are our estimated latency based on the sampling method in the corresponding paper, and should actually be greater than the marked values. The rest methods are the actual test results in the same hardware environment.)

training and inference computationally intractable, creating a formidable barrier to their practical application.

In this paper, we aim to resolve this critical trade-off in pixel diffusion models. We propose an efficient pixel space diffusion framework called DiP. As shown in Figure 1(c), for efficient global structure construction, we employ a DiT [38] backbone. Critically, we configure it to operate on large image patches (e.g., 16×16). This setting choice drastically reduces the input sequence length, aligning it with that of mainstream LDMs operating in latent space. Consequently, our model achieves computational efficiency comparable to LDMs while remaining entirely VAE-free, enabling it to effectively capture the global layout and semantic content of the image. However, operating on large patches alone inevitably leads to blurry outputs lacking high-frequency details. To address this, we introduce a lightweight Patch Detailer Head (only 0.3% increase in total parameters), which is not a post-processing module but an integral component co-trained with the DiT backbone. For each large patch, it receives contextual features from the DiT and leverages its strong local receptive fields to synthesize the missing high-frequency information. This synergistic design allows the DiT backbone to focus on the computationally demanding task of global consistency, while the efficient Patch Detailer Head specializes in local texture and detail restoration. As demonstrated in Figure 2, our approach sets a new state-of-the-art on the efficiencyquality frontier, achieving superior FID scores at significantly lower latency compared to existing methods. Our main contributions are summarized as follows:

- • We propose DiP, a new end-to-end pixel diffusion model framework that effectively alleviates the trade-off between generation quality and computational efficiency through synergistic global-local modeling.
- • We systematically validate the impact of different architectural designs of our framework, hoping to provide the community with a unified, principled framework.
- • On ImageNet generation benchmarks, our framework achieves state-of-the-art performance and lowest inference latency with low training costs.

### 2. Related Work

Latent Diffusion Models. Latent Diffusion Models (LDMs) [2, 3, 12, 39, 42, 57, 62] have become the de-facto paradigm for large-scale generative modeling due to their computational efficiency and scalability. By performing the diffusion process in a compressed latent space learned by a VAE, LDMs drastically reduce memory and computational costs. Architectural advancements within this paradigm, such as replacing the U-Net [43] backbone with a more scalable Transformer (DiT) [38], have further pushed the boundaries of generation quality. Despite their success, this efficiency comes at a cost: the VAE acts as an information bottleneck, imposing a hard ceiling on the final image fidelity and often introducing subtle reconstruction artifacts [46, 59]. Our work circumvents these limitations by proposing an equally efficient architecture that operates directly in pixel space, thereby eliminating the VAE-induced quality constraints.

Pixel Diffusion Models. Recent years have seen renewed interest in pixel space diffusion models that aim to maximize signal fidelity while addressing computational inefficiency. Early works such as ADM [9] and DDPM [20] demonstrated the power of diffusion but were constrained by the quadratic complexity of their backbones, rendering them impractical for high resolutions. Multi-scale and image patch-based methods [5, 10, 22, 23, 23] further enhance the generation effect by decomposing large images into small patches. However, these methods essentially simulate locality through brute-force training, which leads to extremely low efficiency. Concurrent work JiT [33] demonstrates that high-dimensional data in pixel space can be effectively modeled by predicting clean images. Recent work by PixelNerd [55] leverages a Transformer to process image features, which then conditions a NeRF-like coordinate network to act as a renderer for finely reconstructing each image patch, achieving impressive performance. Nevertheless, PixelNerd tightly couples the success of its method with this specific NeRF-like rendering mechanism, which may limit the exploration of a broader design space. We argue that the key to achieving efficient and high-quality pixel space generation lies not in relying on a specific structure like NeRF, but rather in the design principle of decoupling global struc-

ture construction from local detail refinement. Based on this insight, this paper aims to provide a more principled, efficient, and general solution for pixel space diffusion models.

### 3. Methods

#### 3.1. Preliminaries

A diffusion process gradually perturbs an initial data sample x0 ∼ q (x0) from the true data distribution into isotropic Gaussian noise:

xt = √α¯tx0 + √1 − α¯tϵ, where ϵ ∼ N(0,I), (1)

where αt = 1 − βt and α¯t = ti=1 αi. {βt}Ti=1 is a predefined variance schedule that controls the noise level at each

step. As t → T,α¯t → 0, and the distribution of xT converges to a standard normal distribution p(xT) ≈ N(0,I).

This discrete formulation can be generalized to a continuous-time setting via a stochastic differential equation (SDE):

dx = f(x,t)dt + g(t)dw, (2) where f(·,t) is the drift and g(t) is the diffusion coefficient.

The trajectory of this reverse process is governed by a corresponding probability flow ordinary differential equation (ODE):

dx = f(x,t) − g(t)2∇x log pt(x) dt. (3)

Learning to generate data is thus equivalent to learning the score function log pt(x) or the associated vector field of this ODE. To train a neural network for this task, several objectives have been proposed. DDPM trains a model ϵθ (xt,t) to predict the noise component θ from a noisy sample xt:

0,ϵ ∥ϵ − ϵθ (xt,t)∥2 . (4)

LDDPM = Et,x

Flow Matching (FM) [12] provides a simulation-free paradigm for directly learning the vector field. It defines a conditional probability path pt (x | x0) and a corresponding target vector field ut(x). A network vθ(x,t) is then trained to regress this field by minimizing the loss:

t(x|x0) ∥ut(x) − vθ(x,t)∥2 . (5)

LFM = Et,p

#### 3.2. Motivation

DiT models the long-range dependencies of an image by partitioning it into a sequence of patches, thereby forming a coherent global structure. However, while the self-attention mechanism excels at modeling macroscopic relationships between patches, it compresses the rich spatial information within each patch into a single, flattened token. This design introduces an inherent limitation: the model can adeptly

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

DiT-only

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

Ours

origin image

Init 20k steps 60k steps 100k steps

Overfit

- Figure 3. Overfitting the DiT-only model using a single image in pixel space leads to poor detail reconstruction. Introducing a local inductive bias achieves better reconstruction and accelerates convergence. Please zoom in for details.

(a) DiT-only (b) Ours

[Figure 59]

[Figure 60]

- Figure 4. The t-SNE visualization of feature space. In the ImageNet validation set, 100 samples were randomly selected from each of the 10 classes for feature visualization. Features are extracted using DiT-only and our method, with each class shown in a distinct color.

learn the coarse-level layout and arrangement of patches but struggles to model the fine-grained textures and highfrequency details within each patch, consequently limiting the upper bound of its image generation performance.

To empirically validate this, we conduct a preliminary experiment by overfitting a DiT model on a single highresolution image in the pixel space. As shown in Figure 3, the model successfully captures the global layout and color palette but fails to render fine textures and sharp edges, resulting in a blurry reconstruction. This result demonstrates that when a DiT architecture operates directly on images, it suffers from a lack of inductive bias [1, 16, 25, 58] at the local level, rendering it incapable of achieving precise pixel-level reconstruction within each patch.

This motivates our core design principle: to augment the global Transformer with a dedicated module that explicitly re-injects this missing inductive bias for local details. In this way, our model can leverage the computational efficiency afforded by large patch sizes while simultaneously generating high-quality images with fine-grained details. As shown in Figure 4, our method achieves tighter intra-class clusters and clearer inter-class separation, whereas vanilla DiT exhibits more mixed distributions. This means that the introduction of local inductive bias can more effectively integrate local textures and edge cues in pixel space and thus improves high-level semantic separability and feature con-

Patch Detailer Head

Patch Detailer Head

Patch Detailer Head

Patch Detailer Head

Projectout

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Projectin

Projectin

Projectin

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

DiT

DiT

DiT

…

…

…

…

…

…

…

…

…

…

…

…

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

(a) Post-hoc Refinement (b) Intermediate Injection

(c) Hybrid Injection

Figure 5. Patch Detailer Head with local inductive bias was placed at different locations in the model. The results in Sec. 4.3 show that all three methods offer gains compared to DiT-only.

sistency. Such improvements are expected to yield more stable structural alignment and better detail during generation process.

Down Sampling 𝒔𝒊 Up Sampling

Conv

Conv

Conv

Conv

Conv

Conv

Conv

Conv

Conv

Conv

SiLU

SiLU

SiLU

SiLU

SiLU

SiLU

SiLU

SiLU

SiLU

𝒑𝒊 𝛜𝐢

#### 3.3. Framework

Based on the above observation, we introduce a framework for high-quality image generation that operates directly in pixel space. DiP first employs a DiT to model the global structure and long-range dependencies of the image. Subsequently, a lightweight Patch Detailer Head refines the output at the patch level, introducing a local inductive bias to synthesize high-frequency details.

Figure 6. Patch Detailer Head framework. This design introduces the local inductive bias that DiT-only lacks with a low number of parameters, resulting in a high-quality image with rich detail.

- • Standard MLP. As a simple baseline, we used MLP that

takes the feature vector si and a flattened noisy patch pi as input. While straightforward, this design lacks any inherent spatial bias, treating all pixels within the patch as an unordered set.

- • Coordinate-based MLP. To introduce spatial awareness, a design inspired by NeRF can be adopted [55]. For each pixel within pi, we concatenate its normalized 2D coordinates. si is used to dynamically generate the weights of a small, coordinate-based MLP. This implicitly learns a continuous function of the image patch, but it lacks the strong priors for local texture and structure that convolutions provide.
- • Intra-Patch Attention. We explored using a small Transformer to operate on the pixels within each patch. Each P×P patch is treated as a sequence of P2 pixel tokens. This allows for complex, content-aware interactions between pixels but is computationally intensive and may not be as efficient as convolutions for learning local patterns.
- • Convolutional U-Net (Our Final Choice). We found that a lightweight convolutional U-Net provided the best performance. The hierarchical structure of downsampling and upsampling paths, combined with skip connections, is exceptionally well-suited for capturing multi-scale spatial features and ensuring local continuity. The inherent inductive biases of convolutions (locality and translation equivariance) are highly effective for denoising local textures and edges. As shown in Figure 6, we instantiate the Patch Detailer Head with a shallow U-Net, which includes 4 downsampling and 4 upsampling blocks. Each block consists of a sequence of Convolution, SiLU activation and pooling layer. The global feature vector si is

Global Structure Construction (DiT Backbone). Given a noisy image xt ∈ RH×W×3 at timestep t, we first partition it into a sequence of non-overlapping patches. Each patch has a size of P×P (we set P=16), resulting in a sequence of N=(H×W)/P2 patches. This patching strategy ensures our pixel space model maintains a computational footprint comparable to latent space DiT models. Along with a timestep embedding and positional embeddings, they are fed into a series of DiT blocks to produce a sequence of context-aware output features Sglobal ∈ RN×D, where D is the feature dimension.

Local Detail Refinement (Patch Detailer Head). The Patch Detailer Head operates independently and in parallel on each patch. For each patch i, it takes two inputs: the corresponding global context map si and the original noisy pixel patch pi ∈ R3×P×P, where si ∈ RD×1×1 is obtained by reshaping and expanding Sglobal. Its objective is to leverage the global context from Sglobal to accurately interpret the local noisy information in pi, ultimately predicting the corresponding noise component ϵi ∈ R3×P×P for that patch. After processing all N patches in parallel, the resulting sequence of predicted noise patches {ϵi}Ni=1 is reassembled into a full-resolution noise prediction map.

#### 3.4. Architecture Design

Exploring Patch Detailer Head Architectures. We investigated several architectures for the Patch Detailer Head, each embodying a different form of inductive bias. Our goal is to present a simple, effective and highly efficient design.

ImageNet 256×256

Method

FID↓ sFID↓ IS↑ Prec.↑ Rec.↑ Latency↓ Epochs NFE Params Latent Generative Models

LDM [42] 3.60 - 247.7 0.87 0.48 - 170 250x2 400M+86M DiT-XL [38] 2.27 4.60 278.2 0.83 0.57 2.09s 1400 250x2 675M+86M MaskDiT-G [73] 2.28 5.67 276.6 0.80 0.61 - 1600 79x2 675M+86M SiT-XL [34] 2.06 4.50 270.3 0.82 0.59 2.09s 1400 250x2 675M+86M FlowDCN-XL [54] 2.00 4.33 263.1 0.82 0.58 - 400 250x2 618M+86M

Pixel Generative Models CDM [21] 4.88 - 158.7 - - - 2160 4100 ADM [9] 3.94 6.14 215.8 0.83 0.53 15.80s 400 500 554M JetFormer-L [51] 6.64 - - 0.69 0.56 - 500 - 2.8B SiD [22] 2.77 - 211.8 - - - 800 250×2 2.0B VDM++ [29] 2.12 - 278.1 - - - - 250×2 2.46B RIN [24] 3.42 - 182.0 - - - 480 1000 410M Farmer/16 [72] 3.96 - 250.6 0.79 0.50 - 320 - 1.9B PixelFlow-XL/4 [5] 1.98 5.83 282.1 0.81 0.60 7.50s 320 120x2 677M DiP-XL/16 2.16 4.79 276.8 0.82 0.61 0.92s 160 100x2 631M DiP-XL/16 1.98 4.57 282.9 0.80 0.62 0.70s 320 75x2 631M DiP-XL/16 1.79 4.59 281.9 0.80 0.63 0.92s 600 100x2 631M

Table 1. Comparison of the performance of different methods on ImageNet 256×256 with Euler solver and CFG. Performance metrics are annotated with ↑ (higher is better) and ↓ (lower is better). Our method achieves the best FID score. Furthermore, compared to other pixel diffusion models, we achieve the best performance across all metrics with the lowest latency.

concatenated channel-wise with the downsampling output at the bottleneck. This design allows the global semantic information to guide the local refinement process effectively while keeping the parameter count minimal.

We provide experimental evidence for this part in Sec. 4.3. Furthermore, we present a preliminary theoretical analysis in Appendix to model the necessity and effectiveness of the Patch Detailer Head, aiming to offer deeper insights.

Placement of the Patch Detailer Head. Since the main weakness of DiT backbone being trained directly in pixel space is its lack of local awareness, a natural design question arises: does introducing Patch Detailer Head at different locations in the model also bring gains? As shown in Figure 5, we investigated three placement strategies:

- • Post-hoc Refinement. The Patch Detailer Head is placed only after the final DiT block. This creates a clean separation of concerns: the DiT is solely responsible for global modeling, and Patch Detailer Head is solely responsible for local refinement.
- • Intermediate Injection. The Patch Detailer Head is inserted between DiT blocks. The refined patch representations are then projected back and fed into the subsequent

DiT blocks.

• Hybrid Injection. Patch Detailer Head are placed both at an intermediate stage and at the end of the DiT.

Our experiments in Sec. 4.3 revealed that all three strategies yield comparable performance gains over the baseline DiT. However, the Post-hoc Refinement strategy has a unique advantage: by placing the Head at the end, we treat the standard DiT architecture as a fixed, black-box backbone. This approach requires no modification to the DiT’s internal structure, greatly simplifying implementation and potentially allowing for the use of pre-trained DiT checkpoints. Given its optimal balance between high performance and implementation simplicity, we adopt the postrefinement strategy as the final architecture.

### 4. Experiments

#### 4.1. Setup

Implementation Details. Our experiments are conducted on the class-conditional ImageNet dataset and original images are center-cropped and resized to 256×256 resolution. We set global batch size to 256. We use DDT [56], a variant of DiT, as our model backbone and apply an Exponential

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

- Figure 7. Qualitative samples from our model trained at 256 × 256 resolution with classifier-free guidance scale of 4.0. DiP demonstrates fine-grained detail, and high visual quality.

Moving Average (EMA) on the model weights with a decay factor of 0.9999. In Patch Detailer Head, the kernel size of the middle layers is set to 3, the padding to 1, and the kernel size of the last convolutional layer is set to 1. Unless otherwise specified, all samples were generated using the Euler-100 solver. More details are included in Appendix.

Evaluation Protocol. To ensure a comprehensive and rigorous assessment of our model’s generative capabilities, we adhere to the evaluation protocol established by ADM [9]. We employ a suite of standard quantitative metrics to measure performance across different dimensions. Specifically, we use the Fr´echet Inception Distance (FID) [18] to assess overall realism and fidelity, the Spatial FID (sFID) [37] to evaluate spatial and structural coherence, and the Inception Score (IS) [45] to measure class-conditional diversity. Furthermore, we report Precision (Prec.)/Recall (Rec.) [31] to respectively quantify the fidelity of individual samples and the model’s ability to cover the true data distribution. All metrics are calculated using 50,000 generated samples.

#### 4.2. Main Result

Performance. Table 1 presents a comprehensive comparison against recent SOTA methods with classifier-free guidance scheduling with guidance interval [32]. After 600 training epochs, DiP achieved an FID of 1.79 without

requiring a pre-trained VAE, surpassing potentially diffusion models such as DiT-XL (FID 2.27) and SiT-XL (FID

- 2.06), which require longer training times. DiP outperforms the previous best pixel-based model, PixelFlow-XL/4 (FID 1.98), and significantly exceeds others like ADM (FID
- 3.94) and VDM++ (FID 2.12). Even with a shorter training schedule of 160 epochs, our model reaches a competitive FID of 2.16, outperforming established models like DiT-XL that require much longer training.

Figure 7 presents qualitative samples of DiP at 256×256 resolution. These visualizations reveal rich detail, demonstrating the effectiveness of introducing local inductive bias. More visualization samples are provided in Appendix.

Computational Cost Comparison. DiP’s parameter count (631M) is significantly smaller than other pixel models, such as VDM++ (2.0B) and Farmer (1.9B). DiP reaches its best performance with only 320 epochs, which is over 4× more efficient than DiT-XL and SiT-XL (1400 epochs) and substantially faster than many other pixel-based methods like CDM (2160 epochs). In single-image inference speed tests, DiP (0.92s) is more than 2.2× faster than DiTXL (2.09s) and more than 8× faster than the previous best pixel model, PixelFlow-XL (7.50s). Furthermore, in 75step inference, DiP (0.70s) achieved the same FID score as PixelFlow-XL with a speed more than 10× faster.

ImageNet 256×256 FID↓ sFID↓ IS↑ Prec.↑ Rec.↑ Training Cost Latency Params Scaling Up DiT

Method

DiT-only (26 Layers, 1152 Hidden Dim) 5.28 6.56 243.8 0.74 0.55 84×8 GPU Hours 0.88s 629M DiT-only (32 Layers, 1152 Hidden Dim) 4.91 6.44 251.7 0.74 0.56 103×8 GPU Hours 1.05s 772M DiT-only (26 Layers, 1280 Hidden Dim) 4.28 6.26 249.6 0.77 0.56 103×8 GPU Hours 1.06s 776M DiT-only (26 Layers, 1536 Hidden Dim) 2.83 5.16 285.6 0.80 0.57 149×8 GPU Hours 1.49s 1.1B

###### Different Patch Detailer Head

Standard MLP 6.92 7.27 210.9 0.79 0.41 93×8 GPU Hours 0.91s 630M Intra-Patch Attention 2.98 5.16 275.0 0.80 0.56 96×8 GPU Hours 0.94s 630M Coordinate-based MLP 2.20 4.49 284.6 0.80 0.58 123×8 GPU Hours 0.95s 700M Convolutional U-Net 2.16 4.79 276.8 0.82 0.61 92×8 GPU Hours 0.92s 631M

Table 2. Impact of different design schemes on computational overhead and performance.

[Figure 103]

[Figure 104]

[Figure 105]

ImageNet 512×512

Method

FID↓ sFID↓ Prec.↑ Rec.↑ IS↑ Params Latent Generative Models

DiT-XL [38] 3.04 5.02 0.84 0.54 240.8 675M+86M MaskDiT-G [73] 2.50 5.10 0.83 0.56 256.3 675M+86M SiT-XL [34] 2.62 4.18 0.84 0.57 252.2 675M+86M FlowDCN-XL [54] 2.44 4.53 0.84 0.54 252.8 618M+86M

FID=2.16 FID=2.24 FID=3.53

(a) Post-hoc Refinement (b) Intermediate Injection (c) Hybrid Injection

###### Pixel Generative Models

- Figure 8. The t-SNE visualization of feature space. Features are extracted using Post-hoc Refinement, Intermediate Injection, and Hybrid Injection, with each class shown in a distinct color.

ADM [9] 3.85 5.86 0.84 0.53 221.7 554M SiD [22] 3.02 - - - 248.7 2.00B VDM++ [29] 2.65 - - - 278.1 2.46B RIN [24] 3.95 - - - 216.0 410M

#### 4.3. Analysis

DiP-XL/32 2.31 4.48 0.84 0.58 291.68 631M

In this section, we analyzed the trade-off between generation quality and computational cost during the development of DiP, and at the same time explained the rationality of the Patch Detailer Head we designed.

Table 3. Comparison of the performance of different methods on ImageNet 512×512 with CFG. Performance metrics are annotated with ↑ (higher is better) and ↓ (lower is better). Our method remains competitive at higher resolutions.

Patch Detailer Head vs. Scaling Up DiT. A common strategy to improve generative models is to increase the model size. However, our findings indicate that this is a suboptimal approach for pixel space diffusion models. As shown in Table 2, increasing the DiT’s depth from 26 to 32 layers yields only a marginal improvement (FID from 5.28 to 4.91) at a considerable cost in parameters and training time. It also means that the effectiveness of our Patch Detailer Head comes from the introduction of effective local inductive biases, rather than increasing network depth.

tures for the Patch Detailer Head to understand the importance of inductive bias in local patch refinement. (1) The Standard MLP performs poorly (6.92 FID), even worse than the DiT-only baseline. This is expected, as it lacks any spatial inductive bias, treating patch pixels as an unordered set and failing to capture crucial local structures. (2) The IntraPatch Attention shows a significant improvement over the MLP (FID 2.98). This indicates that content-aware relationships between pixels are valuable. Its training and inference costs are only slightly higher than our final choice, but its actual memory overhead is about twice that of the final solution. (3) The Coordinate-based MLP achieves 2.20 FID (we are based on a reproduction of [55]). By explicitly conditioning on pixel coordinates, it effectively introduces spatial awareness. However, it requires more parameters (700M) and a longer training time (123×8 GPU hours) compared to our final choice, and its implicit continuous representation may lack the strong, built-in priors for local patterns that convolutions provide. (4) The Convolutional U-Net increases the number of parameters by only 0.3% (from 629M

In contrast, widening the model proves more effective for quality improvement. For instance, scaling the hidden dimension to 1536 reduces the FID to 2.83. This substantial quality gain comes at a prohibitive cost: a 74.9% increase in parameters (from 629M to 1.1B), a 77.4% rise in training cost (from 84×8 to 149×8 GPU hours), and a 69.3% (from 0.88s to 1.49s) increase in inference latency. This highlights a critical challenge with monolithic scaling, where significant computational resources are required for performance. Experimental Results of Exploring Patch Detailer Head Architectures. We further investigated different architec-

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

- Figure 9. Qualitative samples from our model trained at 512×512 resolution with classifier-free guidance scale of 4.0. DiP showcases fine-grained detail and rich diversity at higher resolutions.

to 631M) and achieves the best FID score with the lowest computational cost among all Patch Detailer Head. Its success can be attributed to highly relevant inductive biases of convolutions. It is well-suited for capturing and preserving the continuity of local textures and edges, making it the most efficient and effective architecture for performing patch-level detail optimizations.

In summary, our experimental results clearly demonstrates that introducing an appropriate local inductive bias via a Patch Detailer Head is key to performance improvement over the scaling up DiT baseline. Among the architectures explored, the Convolutional U-Net strikes the optimal balance between best generation quality and minimal computational cost, making it our definitive choice.

Experimental Results of Placement of the Patch Detailer Head. We tested the effects of introducing Patch Detailer Head at different locations in the model. As shown in Figure 8, all three introduction modes showed significant improvements compared to DiT-only (FID 5.28). From the feature visualization results, Hybrid Injection performed worse in clustering than the other two, which may be due to multiple local inductive biases potentially disrupting the original structure, leading to performance degradation. Post-hoc Refinement achieves the best performance, a result attributed to the synergy between global build and local refinement, while its implementation is simple and easily extensible.

#### 4.4. Ablation Study

Performance on ImageNet 512×512. As shown in Table 3, on 512×512 resolution, DiP achieved the best FID score , surpassing previous methods. Figure 9 illustrates the sampling results at 512×512, demonstrating that DiP can also generate high-quality images. We present more qualitative results in Appendix.

[Figure 122]

[Figure 123]

###### (a) Different Patch Detailer Head Configuration (b) The Influence of Different Patch Size

Figure 10. (a). Performance differences between different Patch Detailer Head configurations. Depth is defined as the number of down/up-sampling stages, and width corresponds to the number of base channels in the convolutional layers. (b). Performance and computational overhead differences of different patch sizes.

Impact of Patch Detailer Head Configuration. Our findings reveal distinct trends for depth and width. As we increase the depth, we observe a consistent improvement in generation quality, although the gains exhibit diminishing returns and eventually saturate. This suggests that a multi-scale feature hierarchy is crucial for the Patch Detailer Head to effectively synthesize high-frequency details across. Conversely, blindly increasing the width will not lead to a sustained performance improvement. This indicates that the role of Patch Detailer Head is not to perform complex feature transformations, but rather to render specific details.

Impact of Patch Size. Small patch size offer superior performance but also significantly increase computational overhead. Our method can use large patch to shorten the input sequence length, making our model’s computational efficiency comparable to mainstream LDMs. As shown in Figure 10(b), at the higher 512×512 resolution, DiP maintains a significant performance margin over a DiTonly baseline using smaller patch size. This validates that our approach provides a robust solution for efficient, highresolution synthesis directly in pixel space.

### 5. Conclusion

In this paper, we addressed the fundamental trade-off between generation quality and computational efficiency in pixel diffusion models. We introduced DiP, a new endto-end pixel space diffusion framework that resolves this dilemma through a synergistic global-local modeling approach. By employing a DiT backbone on large image patches, we achieve computational efficiency comparable to LDMs for modeling global structures. This is complemented by a co-trained, lightweight Patch Detailer Head that expertly restores high-frequency details, effectively bypassing the need for a VAE. Our extensive experiments on the ImageNet benchmark demonstrate that achieves superior FID scores with significantly lower inference latency and training costs. In the future, we plan to apply the DiP framework to text to image and text to video tasks to further explore the capabilities of this solution.

### Acknowledgment

This work was supported by Natural Science Foundation of Jiangsu Province: BK20241198, the Gusu Innovation and Entrepreneur Leading Talents: No. ZXL2024362 and Natural Science Foundation of China: No. 62406135.

### References

- [1] Jie An, De Wang, Pengsheng Guo, Jiebo Luo, and Alex Schwing. On inductive biases that enable generalization in diffusion transformers. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2024. 3
- [2] BlackForest. Black forest labs; frontier ai lab, 2024. 2
- [3] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023. 1, 2
- [4] Junyu Chen, Han Cai, Junsong Chen, Enze Xie, Shang Yang, Haotian Tang, Muyang Li, Yao Lu, and Song Han. Deep compression autoencoder for efficient high-resolution diffusion models. arXiv preprint arXiv:2410.10733, 2024. 1
- [5] Shoufa Chen, Chongjian Ge, Shilong Zhang, Peize Sun, and Ping Luo. Pixelflow: Pixel-space generative models with flow. arXiv preprint arXiv:2504.07963, 2025. 1, 2, 5
- [6] Zhennan Chen, Rongrong Gao, Tian-Zhu Xiang, and Fan Lin. Diffusion model for camouflaged object detection. In ECAI 2023, pages 445–452. IOS Press, 2023. 1
- [7] Zhennan Chen, Yajie Li, Haofan Wang, Zhibo Chen, Zhengkai Jiang, Jun Li, Qian Wang, Jian Yang, and Ying Tai. Ragd: Regional-aware diffusion model for text-to-image generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19331–19341, 2025.
- [8] En Ci, Shanyan Guan, Yanhao Ge, Yilin Zhang, Wei Li, Zhenyu Zhang, Jian Yang, and Ying Tai. Describe, don’t dictate: Semantic image editing with natural language intent. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19185–19194, 2025. 1
- [9] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 1, 2, 5, 6, 7
- [10] Zheng Ding, Mengqi Zhang, Jiajun Wu, and Zhuowen Tu. Patched denoising diffusion models for high-resolution image synthesis. In The twelfth international conference on learning representations, 2023. 2
- [11] Nikai Du, Zhennan Chen, Shan Gao, Zhizhou Chen, Xi Chen, Zhengkai Jiang, Jian Yang, and Ying Tai. Textcrafter: Accurately rendering multiple texts in complex visual scenes. arXiv preprint arXiv:2503.23461, 2025. 1
- [12] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

2024. 2, 3

- [13] Tiehan Fan, Kepan Nan, Rui Xie, Penghao Zhou, Zhenheng Yang, Chaoyou Fu, Xiang Li, Jian Yang, and Ying Tai. Instancecap: Improving text-to-video generation via instanceaware structured caption. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 28974– 28983, 2025. 1
- [14] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020. 1
- [15] Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. 7
- [16] Anirudh Goyal and Yoshua Bengio. Inductive biases for deep learning of higher-level cognition. Proceedings of the Royal Society A, 478(2266):20210068, 2022. 3
- [17] Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Fei-Fei Li, Irfan Essa, Lu Jiang, and Jos´e Lezama. Photorealistic video generation with diffusion models. In European Conference on Computer Vision, pages 393–411. Springer, 2024. 1
- [18] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 6
- [19] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 1
- [20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 1, 2
- [21] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. Journal of Machine Learning Research, 23(47):1–33, 2022. 5
- [22] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. In International Conference on Machine Learning, pages 13213–13232. PMLR, 2023. 1, 2, 5, 7
- [23] Emiel Hoogeboom, Thomas Mensink, Jonathan Heek, Kay Lamerigts, Ruiqi Gao, and Tim Salimans. Simpler diffusion (sid2): 1.5 fid on imagenet512 with pixel-space diffusion. arXiv preprint arXiv:2410.19324, 2024. 1, 2
- [24] Allan Jabri, David Fleet, and Ting Chen. Scalable adaptive computation for iterative generation. arXiv preprint arXiv:2212.11972, 2022. 5, 7
- [25] Zahra Kadkhodaie, Florentin Guth, Eero P Simoncelli, and St´ephane Mallat. Generalization in diffusion models arises from geometry-adaptive harmonic representations. arXiv preprint arXiv:2310.02557, 2023. 3
- [26] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019. 1
- [27] Tero Karras, Miika Aittala, Tuomas Kynk¨a¨anniemi, Jaakko Lehtinen, Timo Aila, and Samuli Laine. Guiding a diffu-

- sion model with a bad version of itself. Advances in Neural Information Processing Systems, 37:52996–53021, 2024. 1
- [28] Maciej Kilian, Varun Jampani, and Luke Zettlemoyer. Computational tradeoffs in image synthesis: Diffusion, masked-token, and next-token prediction. arXiv preprint arXiv:2405.13218, 2024. 1
- [29] Diederik Kingma and Ruiqi Gao. Understanding diffusion objectives as the elbo with simple data augmentation. Advances in Neural Information Processing Systems, 36: 65484–65516, 2023. 1, 5, 7
- [30] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 1
- [31] Tuomas Kynk¨a¨anniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. Advances in neural information processing systems, 32, 2019. 6
- [32] Tuomas Kynk¨a¨anniemi, Miika Aittala, Tero Karras, Samuli Laine, Timo Aila, and Jaakko Lehtinen. Applying guidance in a limited interval improves sample and distribution quality in diffusion models. Advances in Neural Information Processing Systems, 37:122458–122483, 2024. 6, 7
- [33] Tianhong Li and Kaiming He. Back to basics: Let denoising generative models denoise. arXiv preprint arXiv:2511.13720, 2025. 2
- [34] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, pages 23–40. Springer, 2024. 5, 7
- [35] Mehdi Mirza and Simon Osindero. Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784, 2014. 1
- [36] Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai. Openvid-1m: A large-scale high-quality dataset for text-tovideo generation. arXiv preprint arXiv:2407.02371, 2024. 1
- [37] Charlie Nash, Jacob Menick, Sander Dieleman, and Peter W Battaglia. Generating images with sparse representations. arXiv preprint arXiv:2103.03841, 2021. 6
- [38] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205,

2023. 1, 2, 5, 7

- [39] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2
- [40] Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434, 2015. 1
- [41] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1

(2):3, 2022. 1

- [42] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 2, 5
- [43] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015. 2
- [44] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 1
- [45] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016. 6
- [46] Ivan Skorokhodov, Sharath Girish, Benran Hu, Willi Menapace, Yanyu Li, Rameen Abdal, Sergey Tulyakov, and Aliaksandr Siarohin. Improving the diffusability of autoencoders. arXiv preprint arXiv:2502.14831, 2025. 2
- [47] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015. 1
- [48] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.
- [49] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 1
- [50] Jiayan Teng, Wendi Zheng, Ming Ding, Wenyi Hong, Jianqiao Wangni, Zhuoyi Yang, and Jie Tang. Relay diffusion: Unifying diffusion process across resolutions for image synthesis. arXiv preprint arXiv:2309.03350, 2023. 1
- [51] Michael Tschannen, Andr´e Susano Pinto, and Alexander Kolesnikov. Jetformer: An autoregressive generative model of raw images and text. arXiv preprint arXiv:2411.19722,

2024. 5

- [52] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 1
- [53] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, Anthony Chen, Huaxia Li, Xu Tang, and Yao Hu. Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024. 1
- [54] Shuai Wang, Zexian Li, Tianhui Song, Xubin Li, Tiezheng Ge, Bo Zheng, and Limin Wang. Flowdcn: Exploring dcnlike architectures for fast image generation with arbitrary resolution. In Proceedings of the 38th International Conference on Neural Information Processing Systems, pages 87959– 87977, 2024. 5, 7

- [55] Shuai Wang, Ziteng Gao, Chenhui Zhu, Weilin Huang, and Limin Wang. Pixnerd: Pixel neural field diffusion. arXiv preprint arXiv:2507.23268, 2025. 2, 4, 7
- [56] Shuai Wang, Zhi Tian, Weilin Huang, and Limin Wang. Ddt: Decoupled diffusion transformer. arXiv preprint arXiv:2504.05741, 2025. 5, 7
- [57] Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Yujun Lin, Zhekai Zhang, Muyang Li, Yao Lu, and Song Han. Sana: Efficient high-resolution image synthesis with linear diffusion transformers. arXiv preprint arXiv:2410.10629, 2024. 1, 2
- [58] Tao Yang, Cuiling Lan, Yan Lu, and Nanning Zheng. Diffusion model with cross attention as an inductive bias for disentanglement. Advances in Neural Information Processing Systems, 37:82465–82492, 2024. 3
- [59] Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15703–15712, 2025. 1, 2
- [60] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 1

- [61] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022. 1
- [62] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 2
- [63] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 7
- [64] Xuying Zhang, Xiaoshuai Sun, Yunpeng Luo, Jiayi Ji, Yiyi Zhou, Yongjian Wu, Feiyue Huang, and Rongrong Ji. Rstnet: Captioning with adaptive attention on visual and non-visual words. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15465–15474,

2021. 1

- [65] Xuying Zhang, Yutong Liu, Yangguang Li, Renrui Zhang, Yufei Liu, Kai Wang, Wanli Ouyang, Zhiwei Xiong, Peng Gao, Qibin Hou, et al. Tar3d: Creating high-quality 3d assets via next-part prediction. arXiv preprint arXiv:2412.16919,

2024. 1

- [66] Xuying Zhang, Bo-Wen Yin, Yuming Chen, Zheng Lin, Yunheng Li, Qibin Hou, and Ming-Ming Cheng. Temo: Towards text-driven 3d stylization for multi-object meshes. In Proceedings of the ieee/cvf conference on computer vision and pattern recognition, pages 19531–19540, 2024.
- [67] Xuying Zhang, Yupeng Zhou, Kai Wang, Yikai Wang, Zhen Li, Shaohui Jiao, Daquan Zhou, Qibin Hou, and MingMing Cheng. Ar-1-to-3: Single image to consistent 3d object generation via next-view prediction. arXiv preprint arXiv:2503.12929, 2025. 1

- [68] Chen Zhao, Weiling Cai, Chenyu Dong, and Chengwei Hu. Wavelet-based fourier information interaction with frequency diffusion adjustment for underwater image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8281–8291,

2024. 1

- [69] Chen Zhao, Weiling Cai, Chengwei Hu, and Zheng Yuan. Cycle contrastive adversarial learning with structural consistency for unsupervised high-quality image deraining transformer. Neural Networks, 178:106428, 2024. 1
- [70] Chen Zhao, En Ci, Yunzhe Xu, Tiehan Fan, Shanyan Guan, Yanhao Ge, Jian Yang, and Ying Tai. Ultrahr-100k: Enhancing uhr image synthesis with a large-scale high-quality dataset. Advances in Neural Information Processing Systems, 2025. 1
- [71] Chen Zhao, Jiawei Chen, Hongyu Li, Zhuoliang Kang, Shilin Lu, Xiaoming Wei, Kai Zhang, Jian Yang, and Ying Tai. Luve: Latent-cascaded ultra-high-resolution video generation with dual frequency experts. arXiv preprint arXiv:2602.11564, 2026. 1
- [72] Guangting Zheng, Qinyu Zhao, Tao Yang, Fei Xiao, Zhijie Lin, Jie Wu, Jiajun Deng, Yanyong Zhang, and Rui Zhu. Farmer: Flow autoregressive transformer over pixels. arXiv preprint arXiv:2510.23588, 2025. 5
- [73] Hongkai Zheng, Weili Nie, Arash Vahdat, and Anima Anandkumar. Fast training of diffusion models with masked transformers. arXiv preprint arXiv:2306.09305, 2023. 5, 7
- [74] Dewei Zhou, You Li, Fan Ma, Xiaoting Zhang, and Yi Yang. Migc: Multi-instance generation controller for text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6818– 6828, 2024. 1
- [75] Dewei Zhou, Ji Xie, Zongxin Yang, and Yi Yang. 3dis: Depth-driven decoupled instance synthesis for text-to-image generation. arXiv preprint arXiv:2410.12669, 2024. 1
- [76] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycleconsistent adversarial networks. In Proceedings of the IEEE international conference on computer vision, pages 2223– 2232, 2017. 1

## DiP: Taming Diffusion Models in Pixel Space Supplementary Material

### A. Why Patch Detailer Head: A Theoretical Perspective

In this section, we try to provide a simplified theoretical analysis to further elucidate why we need local detail refinement in enhancing generation quality. From a general insight, we argue that DiT primarily focuses on the layout and arrangement of the dominant elements in the image, or in other words, the low-frequency signals of the global data. Consequently, it is less effective to learn local details and high-frequency signals. Through the refinement structure, we directly inject all signals from the global data into the learning process, which substantially improves the fine-grained processing of these high-frequency details.

Specifically, we follow the flow matching description of the diffusion process. Given an initial data sample x0 ∼ pdata(x0) ∈ Rd as the input, a Gaussian noise ϵ ∼ N(0,Id), and t ∈ [0,1], let

xt = (1 − t)x0 + tϵ. (6) In this paper, since all inputs are partitioned into patches of equal size, we first define the patch-level input as follows.

- Definition A.1 (Patch-level Input). For each input x0 ∈ Rd, we define the patch-level input as x(0s)

N s=1

, where x(0s) ∈ Rp

and x0 = x(1)0

⊤

,··· , x(0N)

⊤ ⊤

, Np = d.

It is natural to represent the patch-level input by a series of selection matrices P(s) Ns=1. For each s, P(s) ∈ Rp×d satisfies P(s) P(s) ⊤ = Ip and P(s)x0 = x(0s). The flow-based models try to minimize a loss function defined as LFM = Et,x

0,ϵ ∥f(xt,t) − (ϵ − x0)∥2 . Assuming that each patch is independent of one another, a patch-level predictor f(·,t) tries to estimate the patch-level objective field vˆ(s) = f x(ts),t for each patch-level noised input x(ts) = (1 − t)x(0s) + tϵ(s), where ϵ(s) = P(s)ϵ ∼ N(0,Ip). For the given LFM, the optimal predictor is the conditional expectation

vˆ(s),∗ = E ϵ(s) − x(0s) x(ts) .

However, in true generation tasks, each patch is not independent of others, because, for natural images, the boundaries between adjacent patches are typically continuous and smoothly varying (e.g., there is little difference between one patch of sky and another). The correlation between patches only weakens when an abrupt transition occurs in the image’s elements, such as at the boundary between sky and grass. Moreover, DiT’s attention-based structure allows a single patch to access partial information from all other patches. Although this information may be coarse, this remains a complex, coupled structure.

Therefore, for a DiT model, the estimate of vˆ(s) is not only based on x(ts) but also some other information from x(tl)

l̸=s

. Thus, we define the effective information below.

- Definition A.2 (Effective Information). For a patch-level noised input x(ts)

N s=1

N s=1

, we define EI(s) f; x(ts)

to represent the effective information used for a generation model f to estimate the patch-level vector field vˆ(s) for any s ∈ [N].

Assuming that each patch is independent of one another, the patch-level estimate vˆ(s) = f x(ts),t only uses x(ts) for prediction, which means EI(s) f; x(ts)

N s=1

= x(ts) . Thus the optimal predictor can be more generally formulated as

N s=1

vˆ(s),∗ = E ϵ(s) − x(0s) EI(s) f; x(ts)

. For attention-based generation models, we cannot accurately obtain the

effective information due to the complex coupling structure. However, based on some standard assumptions on the initial data distribution and some empirical observations, we can still give a brief formulation for the effective information.

- Assumption A.3 (Data Distribution). For the initial data distribution, we assume that pdata ∼ N(µ,Σ), where Σ = UΛU⊤, U = [u1,··· ,ud], and Λ = diag{λ1,··· ,λd}.
- Assumption A.4 (Eigenvalue Decay). There exists α > 1 such that for any i ∈ [d], the eigenvalues of Σ satisfies λi ≍ i−a.

Assumption A.3 and A.4 characterize the data distribution as a Gaussian distribution with a covariance of a series of fastdecay eigenvalues. The eigenvalue decay of covariance characterizes the differences in high- and low-frequency signals of the image information. This is consistent with the empirical observation that DiT can effectively learn low-frequency signals but has difficulty capturing high-frequency signals. Given b > 0, we can decompose the input x0 into low- and high-frequency components as

x0 = µ + x0,low + x0,high, (7)

where x0,low ∼ N(0,Σlow) and x0,high ∼ N(0,Σhigh). Σlow satisfies Σlow = UrΛrU⊤r = ri=1 λiuiu⊤i where λr > b and λr+1 ≤ b, and Σhigh = Σ − Σlow. Thus we can decompose the patch-level noised input x(ts) as

x(ts) = (1 − t)P(s)µ

Mean(s)

+(1 − t)P(s)x0,low

Low(s)

+(1 − t)P(s)x0,high

High(s)

+ tP(s)ϵ Noise(s)

(8)

We can assume that for the DiT model, the effective information is composed of the local patch itself and the low-frequency signals of other patches as below.

- Assumption A.5 (EI of DiT). Given DiT as the predictor, there exists β > 0 such that for any s ∈ [N], the effective information to estimate the patch-level vector vˆ(s) satisfies

N s=1

EI(s) DiT; x(ts)

= x(ts) ∪ x(t,llow)

, (9)

l̸=s

where

x(t,llow) = Mean(l) + Low(l) + Noise(l) (10) for all l ̸= s.

Our refinement structure directly injects all signals from the initial data x0 for prediction, which means that for DiP, the effective information satisfies

EI(s) DiP; x(ts)

N s=1

= EI(s) DiT; x(ts)

N s=1

∪ x(ts)

N s=1

= x(ts)

N s=1

. (11)

N s=1

N s=1

and vˆDiP(s) = E ϵ(s) − x(0s) EI(s) DiP; x(ts)

We define vˆDiT(s) = E ϵ(s) − x(0s) EI(s) DiT; x(ts)

as

the general near-optimal estimate of DiT and DiP, respectively. Then we obtain the main results below. Theorem A.6. Assume that Assumption A.3, A.4 and A.5 hold. Consider using DiT and DiP for the diffusion generation task as the predictor, respectively. The general near-optimal estimate vˆDiT(s) and vˆDiP(s) satisfy

vˆDiT(s) = P(s)BˆMˆ (xt − (1 − t)µ) − P(s)µ, (12) and

vˆDiP(s) = P(s)AM(xt − (1 − t)µ) − P(s)µ, (13) respectively, where

⊤

Mˆ = (1 − t)2Σlow + t2Id + (1 − t)2 P(s)

P(s)Σhigh P(s)

⊤

Bˆ = tId − (1 − t)B, B = Σlow + Σhigh P(s)

P(s),

M = (1 − t)2Σ + t2Id −1 , A = tId − (1 − t)Σ.

⊤

P(s)

−1

,

(14)

The denoising operator P(s)BˆMˆ and P(s)AM satisfies

d

r

t − (1 − t)λi (1 − t)2λi + t2

λi t

P(s)BˆMˆ ≍

viu⊤i +

viu⊤i + I1 + I2, (15)

i=r+1

i=1

and

d

t − (1 − t)λi (1 − t)2λi + t2

viu⊤i , (16)

P(s)AM =

i=1

(1−t)2λj+t2 u⊤i P(s) ⊤ P(s)uj viu⊤j and I2 = − di=r+1

(1−t)λi

respectively, where [v1,··· ,vd] = P(s)[u1,··· ,ud], I1 = − di=r+1

r j=1

t2 u⊤i P(s) ⊤ P(s)uj viu⊤j . Proof. See Appendix B.

(1−t)λi

d j=r+1

| |
|---|

Remark A.7. Theorem A.6 illustrates that our designed refinement mechanism exhibits a strong adaptive correction capability for high-frequency signals within the image. Specifically, (12) and (13) align with our objective to estimate the conditional expectation of ϵ − x0 through DiT and DiP. The first term of (12) and (13) represents the estimate of noise, while the second term means the estimate of original data. We focus on the first term regarded as the “denoising process”, with respective “denoising operator” (15) and (16) serve as a global representation characterization, performing denoising on different frequency-domain components derived from the original image.

- • When using only DiT for estimation, Equation (15) indicates that DiT can achieve a good adaptive fit for low-frequency signals (i ≤ r, dominant signals). However, for high-frequency components in the image, relying solely on DiT may not provide sufficient representational capacity for these components. Specifically,

- – The first term in (15) corresponds to the denoising process applied to the low-frequency signals. Although all these belong to the low-frequency regime, those with larger values of λi (i ≤ r) contain a stronger proportion of the original image content relative to noise. Consequently, as λi decreases, the denoising operator adaptively selects a larger

correction magnitude. (It can be readily demonstrated that t−(1−t)λ

i

(1−t)2λi+t2 exhibits a monotonic increase as λi decreases.)

- – The remaining three terms correspond to the denoising process applied to the high-frequency components. Among them, I1 represents the consistent influence exerted by the low-frequency signal on the high-frequency ones. Since the highfrequency components are associated with λi (i ≥ r + 1) that are significantly smaller than those of the low-frequency regime (Assumption A.4), this term can generally be regarded as o(1). The second term and I2—particularly the second term—point to the following fact: when t → 1 in the early stage of denoising, the magnitude of λi is much smaller than t, leading the model to apply only negligible corrections to the high-frequency components. This weakens the model’s ability to learn from this portion of the signal. Conversely, as t → 0 in the late stage of denoising, where the model aims to learn a sensitive compensatory mechanism to capture more fine-grained details from the original image, t becomes

much smaller than λi, causing the model’s corrections to high-frequency components to lose stability. As a result, these corrections may introduce inconsistencies with the previously learned representations, potentially affecting fine details of the final output.

- • In contrast, when using DiP for estimation, Equation (16) demonstrates that DiP can provide a robust adaptive correction for all signals. This is attributed to our refinement mechanism, which, at a low computational cost, enhances the effective information during the denoising process. Particularly for high-frequency signals, the refinement provides a powerful supplement to the information that DiT struggles to capture, which aligns with both our intuition and experimental results.

### B. Proof of Theorem A.6

Proof. Based on (11), we have vˆDiP(s) = E ϵ(s) − x(0s) x(ts)

following statistics to obtain the first term E ϵ(s) xt . Expectations:

N s=1

= E ϵ(s) xt − E x(0s) xt . We first obtain the

E ϵ(s) = 0, E[xt] = (1 − t)µ, (17) Covariances:

Cov ϵ(s),xt = Cov P(s)ϵ,(1 − t)x0 + tϵ = tP(s)Cov(ϵ,ϵ) = tP(s), (18)

and

Cov(xt) = Cov((1 − t)x0 + tϵ) = (1 − t)2Cov(x0) + t2Cov(ϵ) = (1 − t)2Σ + t2Id. (19) Then we use E[Y |X] = EY + Cov(Y,X)Cov(X,X)−1(X − EX) to obtain that

E ϵ(s) xt = tP(s) (1 − t)2Σ + t2Id −1 (xt − (1 − t)µ). (20)

Similarly, for the second term of vˆDiP(s) we have

Cov x(0s),xt = Cov P(s)x0,(1 − t)x0 + tϵ = (1 − t)P(s)Cov(x0,x0) = (1 − t)P(s)Σ, (21) Then we obtain

E x(0s) xt = P(s)µ + (1 − t)P(s)Σ (1 − t)2Σ + t2Id −1 (xt − (1 − t)µ). (22) Thus we have

vˆDiP(s) = E ϵ(s) xt − E x(0s) xt

= P(s) [tId − (1 − t)Σ] (1 − t)2Σ + t2Id −1 (xt − (1 − t)µ) − P(s)µ.

Letting M = (1 − t)2Σ + t2Id −1, [v1,··· ,vd] = P(s)[u1,··· ,ud], A = tId − (1 − t)Σ, we have

 

 

d

d

t − (1 − t)λi (1 − t)2λi + t2

vju⊤j

###### uiu⊤i

P(s)AM =

j=1

i=1

d

t − (1 − t)λi (1 − t)2λi + t2

viu⊤i .

=

i=1

(23)

(24)

, where x(t,llow) = (1 − t)P(l)µ + (1 − t)P(l)x0,low + tP(l)ϵ. We first use one vector to represent the condition x(ts) ∪ x(t,llow)

Similarly, based on Assumption A.5, we have vˆDiT(s) = E ϵ(s) − x(0s) x(ts) ∪ x(t,llow)

l̸=s

. We try

l̸=s

to construct an observation xˆt such that at s patch, P(s)xˆt = x(ts), and at l ̸= s patch P(l)xˆt = x(t,llow) . The following xˆt satisfies the requirement above

xˆt = (1 − t)µ + (1 − t)x0,low + tϵ + (1 − t) P(s)

⊤

P(s)x0,high. (25)

Now we use the same technique to obtain E ϵ(s) x ˆt . The covariance terms satisfy

Cov(ϵ(s),xˆt) = Cov P(s)ϵ,tϵ = tP(s), (26) and

⊤

Cov(xˆt) = Cov (1 − t)x0,low + tϵ + (1 − t) P(s)

P(s)x0,high

⊤

= (1 − t)2Cov(x0,low) + t2Cov(ϵ) + (1 − t)2 P(s)

P(s)Cov(x0,high) P(s)

⊤

⊤

= (1 − t)2Σlow + t2Id + (1 − t)2 P(s)

P(s)Σhigh P(s)

P(s).

Thus we have

⊤

P(s)

(27)

E ϵ(s) x ˆt = tP(s) (1 − t)2Σlow + t2Id + (1 − t)2 P(s)

⊤

P(s)Σhigh P(s)

⊤

P(s)

−1

(xˆt − (1 − t)µ). (28)

For E x(0s) x ˆt , we have

Cov x(0s),xˆt = Cov P(s)x0,low + P(s)x0,high,(1 − t)µ + (1 − t)x0,low + tϵ + (1 − t) P(s)

⊤

= Cov P(s)x0,low,(1 − t)x0,low + Cov P(s)x0,high, P(s)

P(s)x0,high

⊤

P(s) .

= (1 − t)P(s) Σlow + Σhigh P(s)

⊤

P(s)x0,high

Thus we obtain

E x(0s) x ˆt =P(s)µ + (1 − t)P(s) Σlow + Σhigh P(s)

⊤

P(s) ×

(1 − t)2Σlow + t2Id + (1 − t)2 P(s)

⊤

P(s)Σhigh P(s)

⊤

P(s)

−1

(xt − (1 − t)µ).

Finally we have

(29)

(30)

vˆDiT(s) = E ϵ(s) x ˆt − E x(0s) x ˆt

= P(s) [tId − (1 − t)B]Mˆ (xt − (1 − t)µ) − P(s)µ,

(31)

−1

where B = Σlow + Σhigh P(s) ⊤ P(s) and Mˆ = (1 − t)2Σlow + t2Id + (1 − t)2 P(s) ⊤ P(s)Σhigh P(s) ⊤ P(s)

. Letting Bˆ = tId − (1 − t)B, we have

 

  C ˆ1 + Cˆ2 D ˆ + Eˆ

d

−1

P(s)BˆMˆ =

vju⊤j

, (32)

j=1

where

r

###### Cˆ1 =

(t − (1 − t)λi)uiu⊤i

i=1

d

d

###### Cˆ2 = t

λiuiu⊤i − (1 − t)

λiuiu⊤i P(s)

i=r+1

i=r+1

⊤

P(s),

- Dˆ =

r

i=1

((1 − t)2λi + t2)uiu⊤i +

d

i=r+1

t2uiu⊤i ,

- Eˆ = (1 − t)2

d

λi P(s)

i=r+1

⊤

P(s)uiu⊤i P(s)

⊤

P(s).

(33)

We notice that Dˆ is a positive diagonal matrix and Dˆ −1Eˆ ≍ o(1) because Assumption A.4 shows that λp/λq ≍ (q/p)a ≍ o(1) for any 1 ≤ q ≤ r and p ≥ r + 1. Thus due to first-order Taylor expansion we have

D ˆ + Eˆ

−1

= Id + Dˆ −1Eˆ

−1 Dˆ −1 ≈ Id − Dˆ −1Eˆ D ˆ −1 = Dˆ −1 − Dˆ −1EˆDˆ −1 ≍ Dˆ −1. (34)

Therefore we obtain

We finish the proof.

P(s)BˆMˆ ≍

=

 

  C ˆ1 + Cˆ2 D ˆ −1

d

vju⊤j

j=1

d

r

t − (1 − t)λi (1 − t)2λi + t2

λi t

viu⊤i +

###### viu⊤i

i=r+1

i=1

d

r

(1 − t)λi (1 − t)2λj + t2

⊤

###### P(s)uj viu⊤j

u⊤i P(s)

−

i=r+1

j=1

I1

d

d

(1 − t)λi t2

⊤

###### P(s)uj viu⊤j

u⊤i P(s)

−

.

i=r+1

j=r+1

I2

(35)

| |
|---|

### C. More Implementation Details

DiT Architecture Input dim 256×256×3 Num. layers 26 Hidden dim. 1152 Num. heads 16

Patch Detailer Head Architecture DownSampling path 16→8→4→2→1 UpSampling path 1→2→4→8→16 DownSampling channel 3→64→128→256→512 Bottleneck (512+1152)→512 UpSampling channel 512→256→128→64→64 Output Layer 64→3

Optimization Optimizer AdamW Learning rate 0.0001 Weight decay 0 Batch size 256

Interpolants Diffusion sampler Euler Diffusion steps 100 Evaluation suite ADM

Table 4. Hyperparameter settings.

Hypermarameters. Table 4 reports the detailed hyperparameters of DiP, including the DiT Architecture, Patch Detailer Head Architecture, Optimization, and Interpolants.

Objective. DiP follows the training objectives of DDT [56]. It is trained using flow matching as the objective function and regularized using representation alignment techniques. Further improvements could be made by introducing adversarial loss [15], perceptual loss [63].

Sampler. We use Euler-Maruyama ODE sampler with 100 sampling steps by default. For DiT-only and DiP, we used the same inference hyperparameters.

Classifier-Free Guidance. In our experiments, we employ Interval-based Classifier-Free Guidance [32] (Interval-CFG). Specifically, we set the guidance scale to cfg=2.9. The guidance is activated exclusively within the normalized timestep interval of [0.11, 0.97].

### D. How to Preserving High-Frequency Signal: Patch or Image

While the theoretical analysis in Appendix A establishes the need for all-frequency raw signals to refine missing highfrequency details, we also focus on how can this information be injected in the most effective way. Specifically, we are interested in whether patch-level input is better than image-level input, or vise versa, as shown in Figure 11. Intuitively, the transformer structure in DiT has captured the long-distance dependencies, therefore we only need the specific high-frequency signals or details of the image. A toy experiment in Figure 12 verifies this intuition.

In Figure 12(a), through the patch-level input, the learned manifold (black) tightly adheres to the ground truth structure (orange), effectively capturing intricate branching patterns and sharp boundaries. In contrast, Figure 12(b) reveals that global processing leads to over-smoothing. The learned distribution is more dispersed and struggles to lock onto fine structural details. This suggests that we only need the refinement structure to dedicate its capacity to high-frequency sensing without being distracted by long-distance dependencies. On the contrary, with image-level input the network tends to average out features across a broader spatial regime, resulting in a loss of sharp details in high-frequency regions.

| | | | |
|---|---|---|---|
| | | | |
| | | |[Figure 124]|
| | | | |

[Figure 125]

Patch Detailer Head

Patch Detailer Head

DiT DiT

(a) Patch-level Input (b) Image-level input

Figure 11. Different input formats of Patch Detailer Head.

[Figure 126]

[Figure 127]

[Figure 128]

###### GT (a) Patch-level Input (b) Image-level input

Figure 12. Toy experiment. (a) Visualization of manifold fitting with Patch-level input. The model precisely captures high-frequency branches. (b) Visualization of manifold fitting with Image-level input. The model exhibits over-smoothing and fails to resolve fine details.

### E. Alternative Patch Detailer Head (PDH).

Details of the alternative PDH are in Fig. 13. The performance gap arises from the Convolutional U-Net’s built-in inductive biases and hierarchical architecture, which better capture spatial detail and preserve local continuity. By contrast, alternative variants lack spatial information or are less effective at modeling local patterns, as explained in Sec. 3.4 (paper).

Batch Size 𝑩, Patch Size 𝑷, Channel 𝑪, Hidden Size 𝑫𝒎𝒍𝒑/𝑫𝒂𝒕𝒕𝒏, Number of Encoding Frequencies 𝑳𝒇𝒓𝒆𝒒 Noisy Pixel Patch 𝒑𝒊, DiT Output 𝒔𝒊, Predicted Noise Patches 𝛜𝐢

𝒑𝒊 𝒔𝒊

𝒑𝒊 𝒔𝒊

(Dynamic Weight)

Cross

ReLU 𝒑𝒊

𝒔𝒊

-Attn

Point-wise MLP

Pixel Point

Token Sequence （𝐵,𝑃 , 𝐷 ）

Linear

##### Concat Vector

##### … 𝛜𝐢

𝛜𝐢

（𝐵, 𝑃 , 𝐶 + 𝐿 ）

（𝐵,𝑃 ×𝐶 + 𝐷 ）

Self-

Ann

Learnable Pos Embedding

𝛜𝐢

DCT Pos Encoding

𝒑𝒊

(a) Standard MLP (b) Coordinate-based MLP (c) Intra-Patch Attention

Figure 13. Details of other PDH.

### F. More Visualization Results

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

[Figure 139]

[Figure 140]

Figure 14. 256×256 samples. Class lable = “goldfish, Carassius auratus” (1). CFG = 4.0.

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

Figure 15. 256×256 samples. Class lable = “junco, snowbird” (13). CFG = 4.0.

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

Figure 16. 256×256 samples. Class lable = “chickadee” (19). CFG = 4.0.

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

Figure 17. 256×256 samples. Class lable = “tree frog, tree-frog” (30). CFG = 4.0.

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

Figure 18. 256×256 samples. Class lable = “mud turtle” (35). CFG = 4.0.

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

- Figure 19. 256×256 samples. Class lable = “teddy, teddy bear” (859). CFG = 4.0.

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

###### Figure 20. 256×256 samples. Class lable = “cauliflower” (938). CFG = 4.0.

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

###### Figure 21. 256×256 samples. Class lable = “potpie” (964). CFG = 4.0.

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

###### Figure 22. 256×256 samples. Class lable = “bolete” (997). CFG = 4.0.

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

- Figure 23. 512×512 samples. Class lable = “ptarmigan” (81). CFG = 4.0.

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

- Figure 24. 512×512 samples. Class lable=”jellyfish” (107). CFG=4.0.

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

Figure 25. 512×512 samples. Class lable=”Maltese dog, Maltese terrier, Maltese” (153). CFG=4.0.

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

Figure 26. 512×512 samples. Class lable = “lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens” (387). CFG = 4.0.

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

Figure 27. 512×512 samples. Class lable = “barn” (425). CFG = 4.0.

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

Figure 28. 512×512 samples. Class lable = “beacon, lighthouse, beacon light, pharos” (437). CFG = 4.0.

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

Figure 29. 512×512 samples. Class lable = “beer glass” (441). CFG = 4.0.

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

Figure 30. 512×512 samples. Class lable = “wool, woolen, woollen” (911). CFG = 4.0.

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

Figure 31. 512×512 samples. Class lable = “trifle” (927). CFG = 4.0.

