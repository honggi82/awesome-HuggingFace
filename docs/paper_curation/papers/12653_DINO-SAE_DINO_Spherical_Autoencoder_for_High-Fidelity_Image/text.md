# arXiv:2601.22904v2[cs.CV]9May2026

## Hyperspherical Autoencoder for High-Fidelity Image Reconstruction and Generation

#### Hun Chang∗

KAIST AI hun.mark.chang@kaist.ac.kr

#### Byunghee Cha∗

KAIST AI paulcha1025@kaist.ac.kr

Jong Chul Ye KAIST AI jong.ye@kaist.ac.kr

[Figure 1]

Figure 1: Representative high-fidelity reconstruction and generation samples from HAE.

### Abstract

Recent studies have explored using pretrained Vision Foundation Models (VFMs) such as DINO for generative autoencoders, showing strong generative performance. Unfortunately, existing approaches often suffer from limited reconstruction fidelity due to the loss of high-frequency details. In this work, we present the Hyperspherical Autoencoder (HAE), a framework that bridges semantic representation and pixel-level reconstruction. Our key insight is that while semantic information in contrastive representations is primarily directional, enforcing strict magnitude matching hinders the preservation of fine-grained details. To address this, we introduce a Directional Feature Alignment objective that enforces semantic consistency while allowing flexible feature magnitudes for detail retention, alongside a Hierarchical Convolutional Patch Embedding module to enhance local structure preservation. Furthermore, observing that SSL-based representations intrinsically lie on a hypersphere, we employ Riemannian Flow Matching to train a Diffusion Transformer (DiT) directly on this spherical latent manifold. Notably, our manifold-aware DiT exhibits highly efficient convergence, achieving an exceptional gFID of 1.96 alongside a reconstruction rFID of 0.78 and a PSNR of 25.2 dB, validating the advantages of our manifold-aware approach. Code is available at https://github.com/wkdgnsgo/HAE.

∗Equal contribution.

Preprint.

### 1 Introduction

Diffusion models [1, 2, 3] have revolutionized image generation, and a key factor behind their success is the use of latent diffusion model (LDM) [4], where generation is performed in the compact latent space of a pretrained autoencoder. Consequently, the design of the latent space plays a decisive role in determining the overall performance of diffusion models. Recent work has shown that aligning internal diffusion representations with pretrained Vision Foundation Models (VFMs) can accelerate convergence and improve semantic generation, suggesting that similar benefits may be obtained by incorporating VFM representations into latent autoencoders. Along this direction, methods such as RAE [5] leverage pretrained VFMs like DINOv2 [6] as encoders, enabling the latent space to capture richer semantic information. However, this comes with a critical trade-off: although VFM-based autoencoders improve semantic expressiveness, they often exhibit substantially degraded pixel-level reconstruction fidelity, e.g., lower PSNR, compared to standard VAEs.

We attribute this limitation to two primary factors: the aggressive downsampling in standard Vision Transformer (ViT) patch embeddings and the rigidity of feature alignment objectives. Existing methods typically employ Mean Squared Error (MSE) to align the student encoder with the VFM teacher. However, strictly enforcing both magnitude and directional alignment creates a gradient conflict between semantic preservation and pixel reconstruction. This over-constrained optimization landscape prevents the encoder from learning the subtle high-frequency features necessary for fidelity.

To address these challenges, we introduce the Hyperspherical Autoencoder (HAE), a framework designed to reconcile semantic abstraction with high-fidelity reconstruction. Through systematic analysis, we discover that the semantic integrity of VFM representations is predominantly encoded in their directional component, while magnitude plays a negligible role. Driven by this empirical finding, we adopt a Hierarchical Convolutional Patch Embedding to preserve local texture information, paired with a Directional Feature Alignment strategy using cosine similarity. Unlike MSE, our objective relaxes the magnitude constraint, providing the encoder with the necessary degrees of freedom to minimize reconstruction error while prioritizing semantic alignment via feature direction.

Building upon the insight that directional information dictates semantics, we extend this geometric perspective to the generative stage. Theoretically and empirically, representations derived from SelfSupervised Learning (SSL, e.g., DINO) naturally reside on a hyperspherical manifold rather than a flat Euclidean space. Consequently, applying standard Euclidean diffusion to these latents introduces a fundamental geometric mismatch. To resolve this, we model the generative dynamics directly on the hypersphere via Riemannian Flow Matching (RFM) [7]. By constraining the generative process to the spherical manifold and modeling geodesics, we eliminate redundant radial variations and focus solely on semantically meaningful angular dynamics, providing a geometrically principled and highly efficient formulation. Our contributions can be summarized as follows:

- • We propose a Hierarchical Convolutional Patch Embedding that mitigates the information bottleneck of standard ViT patchification, significantly enhancing local detail preservation.
- • We introduce Directional Feature Alignment, which alleviates the optimization conflict inherent in MSE-based distillation. This enables the encoder to simultaneously achieve high semantic alignment and highly competitive reconstruction fidelity (25.2 dB PSNR).
- • We empirically reveal that the semantic properties of VFM latents are strictly governed by their directional components. Guided by this geometric insight and the intrinsic spherical nature of SSL features, we formulate the generative process via Riemannian Flow Matching. This manifold-aware approach resolves the geometric mismatch of Euclidean baselines, significantly accelerating training convergence and enhancing generation quality.

### 2 Related Work

Representation Alignment. Recent work has shown that leveraging semantic representations can significantly improve the training efficiency of diffusion-based generative models. Following the introduction of REPA [8], which aligns intermediate features of DiT with pretrained DINOv2 representations, several extensions have been proposed. REG [9] further enhances this paradigm by introducing a class token into DiT and aligning it with the DINOv2 class token in feature space during generation, leading to even faster convergence. Beyond external representation models, a

number of self-contained approaches have demonstrated similar efficiency gains: SRA [10] aligns intermediate features at higher noise levels with later-layer features at lower noise levels, while LSEP [11] shows that merely enforcing linear separability of intermediate features via a classification probe can improve convergence. REPA-E [12] extends the range of representation alignment training to the Latent Autoencoder itself, jointly optimizing VAE parameters under the REPA objective, further improving convergence speed.

Semantic Latent Autoencoders. In parallel, substantial effort has been devoted to improving latent autoencoders by aligning their latent spaces with semantic representations. Methods such as VAVAE [13] and MAETok [14] explicitly align VAE latents with pretrained vision foundation models, achieving competitive or even improved rFID while dramatically accelerating gFID convergence. RAE [5] takes a more direct approach by replacing the VAE encoder entirely with a pretrained vision foundation model, eliminating the need for explicit alignment losses and fully exploiting pretrained representations. However, RAE and similar VFM-encoder-based tokenizers exhibit significantly degraded pixel-level reconstruction quality, especially suffering low PSNR. This limitation arises because pretrained vision foundation models prioritize semantic abstraction over fine-grained details, which are essential for accurate reconstruction. We identify the key architectural bottleneck causing this information loss and introduce a minimal modification that improves reconstruction fidelity while preserving the semantic strengths of VFM-based encoders.

Riemannian Flow Matching. Chen and Lipman [7] extended the idea of flow matching (FM) [15] beyond Euclidean spaces and proposed Riemannian Flow Matching (RFM), enabling Continuous Normalizing Flow (CNF) training on general Riemannian manifolds. M. In this setting, the model learns a time-dependent tangent vector field vθ(x,t) ∈ TxM that transports samples from a base distribution p to the target data distribution q.

Specifically, the RFM objective is defined as the regression loss

t ∥vθ(xt,t) − u(xt,t)∥2g , (1)

LRFM(θ) = Et,x

where ∥ · ∥g denotes the norm induced by the Riemannian metric, and u(xt,t) is the marginal target flow generating a probability path between p and q.

Following the conditional flow matching, the target vector field can be expressed as the marginalization of a conditional flow ut(x | x1):

pt(xt | x1)q(x1) pt(xt)

. (2)

ut(xt | x1)

dvolx

u(xt,t) =

1

M

A key distinction from standard FM lies in the design of the conditional flow on manifolds. RFM introduces the notion of a premetric d(x,x1) and constructs a conditional trajectory ψt(x0 | x1) that monotonically decreases this distance:

d(ψt(x0 | x1),x1) = κ(t)d(x0,x1), (3)

where κ(t) is a decreasing scheduler satisfying κ(0) = 1 and κ(1) = 0. Under this formulation, the conditional vector field admits the closed-form expression

d(x,x1) ∇d(x,x1) ∥∇d(x,x1)∥2g

dlog κ(t) dt

. (4)

ut(x | x1) =

In the special case where the premetric is chosen as the geodesic distance dg, the resulting conditional flow reduces to the constant-speed geodesic connecting the endpoints, which can be expressed analytically using the exponential and logarithmic maps:

(x0)). (5)

ψt(x0,x1) = expx

(κ(t)logx

1

1

This construction makes RFM simulation-free on simple manifolds, while remaining scalable to general geometries.

### 3 Hyperspherical Autoencoder (HAE)

Our Hyperspherical Autoencoder (HAE) is designed to equip pre-trained vision foundation models with high-fidelity reconstruction capabilities. Our approach addresses the information bottleneck in

[Figure 2]

- Figure 2: HAE architecture and training loss overview

Table 1: Comparison of autoencoder models on ImageNet 256×256.

Model rFID PSNR SD-VAE 0.62 26.04 RAE 0.59 18.94 MAETok 0.48 23.61 VAVAE 0.28 27.96 HAE (Ours) 0.78 25.20

Table 2: Effect of latent smoothing.

Setting rFID gFID PSNR

w/o Latent Smoothing 0.37 32.64 26.2 w/ Latent Smoothing 0.78 2.65 25.2

standard Vision Transformers (ViTs) and introduces a scale-decoupled alignment strategy to preserve high-frequency details without compromising semantic integrity. The overall architecture and its training objective is illustrated in Fig. 2, and the details are provided below.

#### 3.1 Architecture

Enhanced Patch Embedding. Standard ViT architectures, including DINOv3 [16], typically implement patch embedding using a single convolutional layer. We hypothesize that this aggressive, non-overlapping downsampling operation acts as a primary bottleneck, causing the irreversible loss of high-frequency spatial details essential for reconstruction tasks.

To mitigate this, we replace the standard single-layer embedding with a Hierarchical Convolutional Stem. Specifically, we design a four-stage Convolutional Neural Network (CNN) that progressively downsamples the input image. This enhanced patch embedding layer allows the model to capture fine-grained local features (e.g., edges, textures) in the early stages, which are typically discarded by the large-kernel convolution of standard ViTs. The output of this module matches the dimension of the Transformer blocks, serving as a rich input token sequence z0. For the subsequent encoding stages, we utilize the pre-trained DINOv3 Transformer backbone, keeping all parameters frozen to preserve the original semantic representations.

Decoder Architecture. For the decoding stage, we adopt the lightweight yet effective decoder architecture proposed in DC-AE [17]. This decoder is designed to efficiently upsample the latent tokens z back to the pixel space xˆ while minimizing computational overhead. Detailed architectural configurations are provided in Appendix B.

#### 3.2 Training

Directional feature alignment. Standard feature alignment via Mean Squared Error (MSE) strictly enforces both magnitude and directional matching. However, matching the exact magnitude of texture-invariant VFM features creates an over-constrained gradient conflict that degrades pixellevel reconstruction [18, 19]. To alleviate this, we adopt directional feature alignment using cosine similarity: Lalign = 1 − (zS · zT)/(∥zS∥2∥zT∥2), where zS and zT denote the student and teacher features, respectively. This isolates semantic alignment to the angular component, leaving feature magnitudes flexible for high-fidelity reconstruction. This relaxation resolves the reconstructionalignment trade-off and motivates our spherical generation formulation in Section 3.3.

We empirically verify that this relaxation preserves semantic integrity. Linear probing on ImageNet1K (Figure 3) shows HAE retains competitive accuracy (87% Top-1, 97% Top-5) compared to DINOv3 (89%, 98%). Furthermore, PCA visualizations of the feature maps (Figure 4) confirm that the student faithfully reproduces the teacher’s fine-grained semantic geometry. These results demonstrate that directional alignment effectively transfers semantic priors while supporting highfidelity reconstruction.

[Figure 3]

[Figure 4]

- Figure 3: Linear probing results on ImageNet-1K. HAE retains robust semantic information throughout training. Figure 4: PCA visualization of feature representations. HAE

faithfully preserves the teacher’s semantic structure.

Progressive Training Strategy. We train our model in four progressive stages to balance semantic alignment and reconstruction fidelity.

- Stage 1: Semantic-Structural Alignment. In the initial phase, we train the encoder—specifically focusing on the hierarchical patch embedding—alongside the decoder to establish a stable latent space. We employ a combination of the directional alignment loss, pixel-wise reconstruction loss, and perceptual loss:

LStage1 = λcosLalign + λL1∥x − xˆ∥1 + λlpipsLLPIPS(x,xˆ) (6)

where LLPIPS denotes the LPIPS loss for perceptual quality. Notably, this configuration—combining the hierarchical patch embedding with the cosine alignment objective—maximizes the model’s pure reconstruction capacity, yielding exceptionally high PSNR and low rFID scores prior to any generative regularization.

- Stage 2: Adversarial Adaptation with Stochastic Latent Smoothing. From this stage onwards, we freeze the entire encoder to preserve the established semantic latent space. To bridge the domain gap between strict reconstruction and downstream generative sampling, we introduce adversarial training coupled with a stochastic latent noise injection strategy.

Following RAE [20], we inject isotropic Gaussian noise into the latent representations with a randomized continuous scale: z′ = z+σϵ, where σ ∼ U(0,τ) is sampled per batch and ϵ ∼ N(0,I). Rather than constraining latents to the exact hyperspherical surface, this perturbation thickens the latent manifold by forming a local volumetric neighborhood around each semantic direction. Training the decoder to reconstruct from these perturbed latents encourages local smoothness and improves robustness to imperfect latent trajectories produced by the Diffusion Transformer (DiT) during inference.

This smoothing introduces a trade-off between pure reconstruction fidelity and downstream generative quality. As shown in Table 2, training without latent smoothing achieves stronger reconstruction metrics, with an rFID of 0.37 and a PSNR of 26.2 dB, but severely degrades generation quality, yielding a gFID above 32.64. In contrast, training with latent smoothing slightly sacrifices reconstruction fidelity, yielding an rFID of 0.78 and a PSNR of 25.2 dB, but substantially improves generative performance to an impressive gFID of 2.65. This demonstrates that latent smoothing is an essential and highly effective regularizer for enabling generative sampling while preserving competitive reconstruction quality.

[Figure 5]

- Figure 5: Magnitude robustness and directional sensitivity of HAE. (a) Ground Truth (GT). (b) Original reconstruction. (c–f) Reconstructions after fixing latent magnitudes to s = 8, 10, 16, and 20. (g) Reconstruction after injecting angular noise (g = 0.1) while preserving the original magnitude. The contrast between stable magnitude-rescaled reconstructions and degraded angle-perturbed reconstruction suggests that semantic and structural information is predominantly encoded in feature direction.

Coupled with this stochastic smoothing, we incorporate an adversarial loss [21, 20] to further enhance the overall reconstruction quality and compensate for potential smoothing artifacts. The final objective is formulated as:

LStage2 = LStage1 + λadvLGAN(x,xˆ) (7)

Detailed training configurations, including the specific discriminator architecture, are provided in Appendix B.

#### 3.3 Image Generation via HAE

Through empirical analysis, we observe that the reconstruction output of our trained autoencoder is primarily governed by the direction of latent features, being relatively insensitive to their magnitude as long as the values remain within a reasonable numerical bound.

To systematically validate this property, we rescale the latent representation z extracted by the encoder to fixed magnitudes along the channel dimension. Specifically, for each spatial location (i,j), we

compute z˜(ijs) = s · ∥zzij

ij∥2, s ∈ {8,10,16,20}.

As illustrated in Figure 5(c-f), the reconstructed images maintain high visual fidelity and semantic consistency despite large variations in the imposed scale, confirming the decoder’s strong robustness to magnitude shifts.

Conversely, to evaluate directional sensitivity, we inject a small Gaussian noise ϵ ∼ N(0,g2I) (with g = 0.1) strictly into the normalized directional vector. We re-normalize this perturbed vector and multiply it by the original magnitude ∥z∥2 to ensure the latent scale remains unchanged, decoding zˆnoise = ∥z∥2 · ∥z¯z¯++ϵϵ∥

. As shown in Figure 5(g), injecting even a minimal amount of angular noise catastrophically degrades the image quality.

2

This stark contrast provides empirical proof that the core semantic and structural information in our latent space is predominantly encoded within the directional (angular) component. Motivated by this foundational property, we propose to perform Riemannian Flow Matching [7] strictly on a patch-wise spherical manifold.

Formally, we model the latent space as a product manifold of N hyperspheres:

##### , where SRC = z ∈ RC ∥z∥2 = R . (8)

##### M = SRC × SRC × ··· × SRC

N times

Each patch-level latent x(i) ∈ RC is therefore constrained to lie on a sphere of fixed radius R. Instead of linear interpolation in Euclidean space, we define the interpolation path between x0 and x1 along the geodesic of the spherical manifold. For each patch index i, the geodesic interpolation at time t ∈ [0,1] is given by

sin tΩ(i) sin Ω(i)

sin (1 − t)Ω(i) sin Ω(i)

x(1i), (9) where the angular distance Ω(i) is defined as

x(0i) +

x(ti) =

Ω(i) = arccos ⟨x(0i),x(1i)⟩ R2

. (10)

The target velocity u(ti) is obtained by differentiating the geodesic with respect to time:

Ω(i) sin Ω(i)

d dt

u(ti) =

cos tΩ(i) x(1i) − cos (1 − t)Ω(i) x(0i) . (11)

x(ti) =

In practice, while the target velocity u(ti) is mathematically bound to the tangent space of the hypersphere, the neural network vθ(xt,t) outputs unconstrained vectors in the Euclidean space RC. Although the trained decoder is highly robust to magnitude variations, allowing the neural network to freely predict radial velocities introduces unnecessary variance and wastes the model’s capacity on semantically negligible magnitude shifts.

To explicitly enforce the manifold constraint and ensure training stability, we decompose the network’s prediction into tangent and radial components. We then apply a Riemannian Flow Matching objective on the tangent space, alongside a radial penalty to strictly suppress any velocity orthogonal to the manifold. As demonstrated in Table 6, calibrating this penalty is crucial for optimal performance; setting λrad = 0.1 effectively suppresses radial drift while maintaining generative quality, significantly outperforming unconstrained (λrad = 0.0) or overly restricted (λrad = 1.0) settings. The overall objective is formulated as:

N

2 2

2 2

1 N

vtan(i) − u(ti)

+ λrad vrad(i)

, (12)

L(θ) = Et,x

0, x1

i=1

where the radial component is computed as vrad(i) = ⟨vθ(i),x¯(ti)⟩x¯(ti) with the unit normal vector x¯(ti) = x(ti)/R, and the projected tangent velocity is vtan(i) = vθ(i) − vrad(i). The hyperparameter λrad controls the strength of the penalty for deviating from the spherical manifold.

This formulation aligns the generative dynamics with the intrinsic geometry of the autoencoder latent space, focusing the model on directional variations and improving training efficiency and stability. Detailed pseudocode for training and inference is provided in Appendix A.

### 4 Experiments

#### 4.1 Image Generation via Riemannian Flow Matching

Implementation. For downstream image generation, we train the LightningDiT-XL architecture utilizing 8 NVIDIA B200 GPUs. All LightningDiT-XL experiments were conducted on 8 NVIDIA B200 GPUs, requiring approximately 20 minutes per epoch. The model is optimized for 550 epochs using AdamW with a learning rate of 2 × 10−4, betas of (0.9,0.95), and a global batch size of 1024.

Crucially, because our generative process is modeled via Riemannian Flow Matching (RFM) directly along the data-driven spherical manifold, the training dynamics converge significantly faster than standard Euclidean flow matching. As a consequence, we empirically observed that using a conventional, slow Exponential Moving Average (EMA) decay rate (e.g., 0.9999) creates a bottleneck; the EMA weights fail to swiftly track the rapidly optimizing parameters, leading to an unwarranted performance penalty. To correctly align the moving average with our accelerated convergence rate, we adopt a faster EMA decay rate of 0.9995. We generate 50,000 images using the Riemannian Euler sampler with manifold projection for 50 steps.

Results. As shown in Table 3, HAE provides a remarkably effective latent space for downstream generative modeling. When paired with LightningDiT-XL, our model achieves a gFID of 2.65 without guidance, significantly outperforming competitive VAE-based foundations such as VAVAE (4.29) and the baseline RAE (4.28) under the exact same 80-epoch training budget and architecture (676M).

Notably, while RAE utilizes a larger architecture with specialized DDT heads (DiTDH-XL, 879M) to achieve its optimal performance, our approach remains highly competitive using a standard, lighter LightningDiT-XL (676M). This highlights the parameter efficiency of our hyperspherical formulation: by restricting diffusion dynamics strictly to the directional manifold, the model effectively bypasses the need to learn radial variations, achieving strong performance with fewer parameters.

- Figure 6 further highlights the training efficiency derived from our hyperspherical formulation. As explicitly shown in the gFID convergence curves, LightningDiT-XL trained on HAE latents exhibits an overwhelmingly faster convergence rate compared to other competitive baseline models.

Table 3: Comparison of generative models on ImageNet 256×256. We report gFID, IS, Precision, and Recall with and without classifier-free guidance (CFG). Lower gFID is better, while higher IS, Precision, and Recall indicate better performance.

Method Epochs #Params Generation@256 w/o guidance Generation@256 w/ guidance

gFID↓ IS↑ Prec.↑ Rec.↑ gFID↓ IS↑ Prec.↑ Rec.↑ Autoregressive

VAR [22] 350 2B 1.92 323.1 0.82 0.59 1.73 350.2 0.82 0.60 MAR [23] 800 943M 2.35 227.8 0.79 0.62 1.55 303.7 0.81 0.62

Euclidean Latent Diffusion & Flow Matching

DiT-XL/2 [3] 400 675M 9.62 121.5 0.67 0.67 2.27 278.2 0.83 0.57 SiT-XL/2 [24] 400 675M 8.61 131.7 0.68 0.67 2.06 270.3 0.82 0.59 REPA [8] 80 675M 7.90 122.6 0.70 0.65 - - - -

800 675M 5.78 158.3 0.70 0.68 - - - DDT [25] 400 675M 6.62 135.2 0.69 0.67 1.52 263.7 0.78 0.63 VAVAE [13] 80 675M 4.29 - - - - - - -

800 675M 2.17 205.6 0.77 0.65 - - - MAETok [14] 800 675M 2.56 224.5 - - 1.72 307.3 - REPA-E [12] 80 675M 3.46 159.8 0.77 0.63 - - - -

800 675M 1.70 217.3 0.77 0.66 - - - RAE [5]

+LightningDiT-XL/1 (DINOv2-B) 80

4.28 - - - - - - -

676M

+LightningDiT-XL/1 (DINOv2-S) 800 1.87 209.7 0.80 0.63 1.41 309.4 0.80 0.63

+DiTDH-XL/1 (DINOv2-B) 80

2.16 214.8 0.82 0.59 - - - 800 1.51 242.9 0.79 0.63 1.13 262.6 0.78 0.67

879M

Riemannian Latent Flow Matching RAE (DINOv2-B)+LightningDiT-XL/1+RJF [26] 80 677M 3.62 186.2 0.82 0.52 2.81 201.22 0.82 0.56 HAE (DINOv3-L)+LightningDiT-XL/1 (Ours) 80

2.65 202.1 0.82 0.55 - - - 550 1.96 249.2 0.72 0.69 1.90 252.7 0.71 0.70

677M

###### Patch Embedding Strategy rFID ↓ PSNR ↑

Method gFID ↓ Euclidean FM 15.95 Riemannian FM 2.65

Standard ViT (Baseline) 0.47 19.55 Hierarchical CNN (Ours) 0.37 26.20

Table 4: Ablation on generative dynamics: Euclidean Flow Matching (FM) vs. Riemannian Flow Matching (RFM).

Table 5: Ablation on patch embedding strategies. Utilizing the hierarchical CNN stem significantly improves high-frequency detail retention (PSNR). Models in this table are evaluated without latent smoothing to isolate the pure architectural impact.

Crucially, while the baseline RAE employs standard Euclidean Flow Matching within an unconstrained latent space, our approach utilizes Riemannian Flow Matching to model diffusion dynamics strictly on the hyperspherical manifold. This exact geometric alignment ensures that the generative model does not waste representational capacity on arbitrary radial variations, allowing it to reach high-fidelity generation in a fraction of the typical training budget. Consequently, our model converges approximately

- 4× faster than the identical DiT architecture trained on RAE latents. Overall, these results demonstrate that respecting the intrinsic geometry of the latent space simultaneously improves generative quality and significantly accelerates training convergence.

[Figure 6]

Figure 6: FID convergence comparison of various autoencoders and methods.

#### 4.2 Ablation Studies

Effect of Hierarchical Convolutional Patch Embedding. To isolate the contribution of our modified patch embedding layer, we compare our HAE against a baseline that strictly uses the standard single-layer ViT patchification of the frozen DINOv3. As shown in Table 5, avoiding modifications to the standard patch embedding results in a severe bottleneck for pixel-level reconstruction, yielding a PSNR of only 19.55 dB. In contrast, our Hierarchical CNN Patch Embedding effectively captures

and preserves high-frequency spatial details early in the network. This architectural adjustment produces a massive improvement in reconstruction fidelity, boosting the PSNR to 19.55 dB while simultaneously improving the rFID.

Impact of the Radial Penalty. As summarized in Table 6, we further investigate the role of the radial penalty term (λrad) during the early generative training phase. When λrad = 0, the network’s velocity predictions are unconstrained, frequently drifting off the hyperspherical tangent space. This introduces unnecessary radial variance, leading to training instability. By enforcing a strict manifold constraint (λrad > 0), we explicitly suppress velocities orthogonal to the sphere, ensuring that the model focuses its capacity entirely on semantically meaningful directional shifts.

λrad gFID ↓

- 0.0 4.40
- 1.0 5.70 0.1 3.90

Table 6: Ablation on radial penalty. Models in this study are trained for 20 epochs.

Riemannian vs. Euclidean Flow Matching. Finally, as reported in Table 4, we evaluate the necessity of Riemannian Flow Matching (RFM) compared to standard Euclidean Flow Matching (FM). To ensure a fair comparison, both models employ the LightningDiT-XL architecture and are evaluated at 80 epochs. Since the core structural and semantic information of our latent space is encoded in the feature direction (as proven in Section 3.3), standard Euclidean FM wastes modeling capacity on predicting arbitrary and redundant magnitude scales. By constraining the vector field to the geodesics of the spherical manifold, RFM inherently respects the geometry of the contrastive features. This results in more structurally coherent generation and a substantial improvement in performance, achieving an optimal gFID of 2.65 compared to the Euclidean baseline’s gFID of 15.95.

### 5 Conclusion

We presented the Hyperspherical Autoencoder (HAE) to resolve the fundamental tension between semantic abstraction and pixel-level fidelity in Vision Foundation Models (VFMs). We identify the aggressive single-layer patch embedding of standard Vision Transformers as the primary bottleneck responsible for the irreversible loss of high-frequency details.

To overcome this, HAE introduces a Hierarchical Convolutional Patch Embedding, successfully recovering and retaining crucial local textures. However, modifying this early stem inevitably shifts the feature space away from the pretrained teacher. To realign these representations without suffering from the over-constrained gradient conflict of standard mean-squared error (MSE), we propose Directional Feature Alignment via cosine similarity. This aligns the angular semantics with the pretrained DINOv3 weights while granting the decoder the magnitude flexibility needed to reconstruct intricate structural details.

Recognizing that this magnitude-robust latent space inherently forms a hyperspherical manifold, we formulated the downstream generative dynamics via Riemannian Flow Matching (RFM). By projecting the vector field onto the tangent space and penalizing redundant radial variations, we prevent the model from wasting capacity on arbitrary magnitude scaling. Additionally, stochastic latent smoothing thickens this manifold, effectively bridging the domain gap between deterministic reconstruction and continuous generative sampling.

Evaluations on ImageNet-1K at 256 × 256 confirm our framework’s superiority. HAE achieves a competitive reconstruction baseline of 0.78 rFID and 25.2 dB PSNR, empowering the downstream Diffusion Transformer to reach an optimal gFID of 1.96 with significantly accelerated convergence. Ultimately, HAE demonstrates that respecting the intrinsic geometry of the latent space—both architecturally and dynamically—provides a highly efficient and structurally coherent paradigm for next-generation visual synthesis.

### References

- [1] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising Diffusion Probabilistic Models. In NeurIPS, 2020.
- [2] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising Diffusion Implicit Models. In ICLR, 2021.
- [3] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023.
- [4] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-Resolution Image Synthesis with Latent Diffusion Models. In CVPR, 2022.
- [5] Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690, 2025.
- [6] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
- [7] Ricky T. Q. Chen and Yaron Lipman. Riemannian flow matching on general geometries. arXiv preprint arXiv:2302.03660, 2023.
- [8] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. In International Conference on Learning Representations, 2025.
- [9] Ge Wu, Shen Zhang, Ruijing Shi, Shanghua Gao, Zhenyuan Chen, Lei Wang, Zhaowei Chen, Hongcheng Gao, Yao Tang, Jian Yang, et al. Representation entanglement for generation: Training diffusion transformers is much easier than you think. In Adv. Neural Inform. Process. Syst., 2025.
- [10] Dengyang Jiang, Mengmeng Wang, Liuzhuozheng Li, Lei Zhang, Haoyu Wang, Wei Wei, Guang Dai, Yanning Zhang, and Jingdong Wang. No other representation component is needed: Diffusion transformers can provide representation guidance by themselves. In ICLR, 2026.
- [11] Junno Yun, Ya¸sar Utku Alçalar, and Mehmet Akçakaya. No alignment needed for generation: Learning linearly separable representations in diffusion models. arXiv preprint arXiv:2509.21565, 2025.
- [12] Xingjian Leng, Jaskirat Singh, Yunzhong Hou, Zhenchang Xing, Saining Xie, and Liang Zheng. Repa-e: Unlocking vae for end-to-end tuning with latent diffusion transformers. In ICCV, 2025.
- [13] Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.
- [14] Hao Chen, Yujin Han, Fangyi Chen, Xiang Li, Yidong Wang, Jindong Wang, Ze Wang, Zicheng Liu, Difan Zou, and Bhiksha Raj. Masked autoencoders are effective tokenizers for diffusion models. In ICML, 2025.
- [15] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.
- [16] Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.
- [17] Junyu Chen, Han Cai, Junsong Chen, Enze Xie, Shang Yang, Haotian Tang, Muyang Li, Yao Lu, and Song Han. Deep compression autoencoder for efficient high-resolution diffusion models. arXiv preprint arXiv:2410.10733, 2024.
- [18] Bowei Chen, Sai Bi, Hao Tan, He Zhang, Tianyuan Zhang, Zhengqi Li, Yuanjun Xiong, Jianming Zhang, and Kai Zhang. Aligning visual foundation encoders to tokenizers for diffusion models. arXiv preprint arXiv:2509.25162, 2025.
- [19] Hao Tang, Chenwei Xie, Xiaoyi Bao, Tingyu Weng, Pandeng Li, Yun Zheng, and Liwei Wang. Unilip: Adapting clip for unified multimodal understanding, generation and editing. arXiv preprint arXiv:2507.23278, 2025.
- [20] Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690, 2025.

- [21] Axel Sauer, Tero Karras, Samuli Laine, Andreas Geiger, and Timo Aila. Stylegan-t: Unlocking the power of gans for fast large-scale text-to-image synthesis. In International conference on machine learning, pages 30105–30118. PMLR, 2023.
- [22] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. In NIPS, 2024.
- [23] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. In NIPS, 2024.
- [24] Nanye Ma, Mark Goldstein, Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. 2024.
- [25] Shuai Wang, Zhi Tian, Weilin Huang, and Limin Wang. Ddt: Decoupled diffusion transformer. arXiv preprint arXiv:2504.05741, 2025.
- [26] Amandeep Kumar and Vishal M. Patel. Learning on the manifold: Unlocking standard diffusion transformers with representation encoders. arXiv preprint arXiv:2602.10099, 2026.
- [27] Jiayan Teng, Wendi Zheng, Ming Ding, Wenyi Hong, Jianqiao Wangni, Zhuoyi Yang, and Jie Tang. Relay diffusion: Unifying diffusion process across resolutions for image synthesis. arXiv preprint arXiv:2309.03350, 2023.
- [28] Ting Chen. On the importance of noise scheduling for diffusion models. arXiv preprint arXiv:2301.10972, 2023.
- [29] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. Simple diffusion: End-to-end diffusion for high resolution images. In International Conference on Machine Learning. PMLR, 2023.
- [30] Patrick Esser et al. Scaling rectified flow transformers for high-resolution image synthesis. In Proceedings of the 41st International Conference on Machine Learning, volume 235. PMLR, 2024.

### A Technical Appendices and Supplementary Material

In this appendix, we detail the training and inference procedures for Riemannian Flow Matching (RFM) on the HAE latent space.

#### A.1 Training Algorithm

- Algorithm 1 outlines the training of our manifold-aware DiT. We strictly constrain the generative process to a hypersphere of radius R = 32.0. Since our DINOv3-Large encoder inherently extracts

C = 1024 dimensional features, setting R = √2Γ((Γ(CC/+1)2)/2) ≈ 32.0 ensures unit variance per channel and optimal numerical scaling.

Furthermore, recent studies [27, 28, 29, 30] indicate that high-dimensional latent spaces (e.g., RAE [5]) suffer from under-corrupted signals at standard noise levels, impairing diffusion training. To compensate for our exceptionally high channel capacity, we adopt a dimension-dependent timeshifting schedule [30]. The shift factor, mathematically defined relative to a base dimension of 4096, evaluates precisely to α = (C × H × W)/4096 = (1024 × 162)/4096 = 8.0. This properly calibrates the noise schedule, dynamically allocating representational capacity to high-curvature flow regions.

Finally, to enforce the manifold constraint, the neural network’s unconstrained output is orthogonalized into tangent and radial components. A radial penalty λrad = 0.1 is applied to strictly suppress non-tangential velocity predictions. Crucially, because our formulation relies heavily on sensitive geometric operations—such as evaluating trigonometric functions (sine, cosine, and arccosine) for spherical interpolations and computing explicit tangent projections—we strictly execute all forward computations and optimization steps in single-precision floating-point format (FP32). This mitigates the accumulation of truncation errors inherent in lower-precision formats (e.g., BF16) and guarantees the numerical stability required to accurately navigate the hyperspherical manifold.

- Algorithm 1 Training Riemannian Flow Matching on HAE Require: Pre-trained Encoder E, DiT vθ, Radius R = 32.0, Time-shift α = 8.0, Penalty λrad = 0.1

- 1: Initialize DiT parameters θ
- 2: while not converged do
- 3: Sample image x ∼ pdata
- 4: Extract and project latent: z0 = R · ∥EE((xx))∥

2

▷ Spherical Projection

- 5: Sample and project noise: z1 ∼ N(0, I), z1 ← R · ∥zz1

1∥2

- 6: Sample tbase ∼ LogitNormal(0, 1), shift time: t ← 1+(αα·−tbase1)t

base

- 7: // Geodesic computations strictly evaluated in FP32
- 8: Angular distance: Ω = arccos ⟨zR0 , zR1 ⟩ ▷ Geodesic SLERP setup

- 9: Interpolate state: zt = R sin((1sin(Ω)−t)Ω) zR0 + sin(sin(Ω)tΩ) zR1

- 10: Target velocity: ut = Rsin(Ω)Ω cos(tΩ)zR1 − cos((1 − t)Ω)zR0

- 11: Predict velocity v = vθ(zt, t) and normal vector z¯t = ∥zzt

t∥2

- 12: Target decomp.: urad = ⟨ut, z¯t⟩z¯t, utan = ut − urad
- 13: Predict decomp.: vrad = ⟨v, z¯t⟩z¯t, vtan = v − vrad
- 14: Loss: L = Mean(∥vtan − utan∥22) + λrad · Mean(∥vrad∥22) ▷ RFM + Penalty
- 15: Update θ using ∇θL
- 16: end while
- 17: return Optimized parameters θ

#### A.2 Inference Algorithm

- Algorithm 2 outlines the inference process utilizing the Riemannian Euler sampler. The sampling trajectory strictly adheres to the geometry of the hypersphere via the exponential map (Rodrigues’ rotation formula). To maintain consistency with the training phase, the same time-shifting schedule (α = 8.0) is applied. At each step, we employ an Auto-Guidance mechanism (Classifier-Free Guidance); the model concurrently evaluates the conditioned and unconditioned states using a null token to extrapolate the guided velocity. The resulting velocity is then explicitly projected onto the tangent space before stepping forward to ensure manifold constraint satisfaction. Finally, before passing the generated latent to the pre-trained HAE decoder, we explicitly rescale the latent norm from the Flow Matching radius (R = 32.0) to a designated decoding scale of 14.0. As detailed in our ablation study (Appendix C), this rescaling step optimally calibrates the feature variance fed into the decoder, preventing numerical instability while preserving the network’s capacity to synthesize intricate high-frequency details.

- Algorithm 2 Inference with Riemannian Euler Sampler and Auto-Guidance

Require: Trained DiT vθ, Pre-trained Decoder D, Radius R = 32.0, Steps N, Time-shift α = 8.0, Guidance

scale w, Cond. y, Null token ∅, VAE scale svae = 14.0

- 1: Sample noise z1 ∼ N(0, I) and project: z1 ← R · ∥zz1

1∥2

- 2: Base time tbase = {1.0, 1.0 − N1 , . . . , 0.0}, shift time: ti = 1+(αα·−tbase1)t,i

base,i

- 3: for i = 1 to N do
- 4: ∆t = ti+1 − ti ▷ Negative time step for reverse generation
- 5: // Auto-Guidance (CFG) Step
- 6: Predict unconditioned velocity: vuncond = vθ(zti, ti, ∅)
- 7: Predict conditioned velocity: vcond = vθ(zti, ti, y)
- 8: Apply Auto-Guidance: v = vuncond + w · (vcond − vuncond)
- 9: Project to tangent space: v ← v − ⟨v, zRti ⟩zRti ▷ Remove radial component

- 10: Compute rotation angle θ = |∆t|·∥Rv∥2 and direction v¯ = sign(∆t)∥v∥v

2+ϵ

- 11: Exponential map: znext = zti cos(θ) + Rv¯ sin(θ) ▷ Rodrigues’ rotation
- 12: Re-normalize: zti+1 = R · ∥zznext

next∥2

- 13: end for
- 14: Rescale to target VAE scale z0 ← svae · zR0 and Decode xˆ = D(z0)

- 15: return xˆ

### B Autoencoder Architecture and Training Details

In this section, we provide the detailed architectural configurations and hyperparameter settings for training our HAE framework. As discussed in Section 3, the training process is divided into multiple stages to balance semantic alignment and reconstruction fidelity. Autoencoder training was performed on 8 NVIDIA B200 GPUs and required approximately 1 hour and 20 minutes per epoch. Here, we detail the core architectures and the foundational configuration used in Stage 1.

#### B.1 Hierarchical Convolutional Patch Embedding

To mitigate the irreversible loss of high-frequency spatial details caused by the aggressive single-layer downsampling of standard ViTs, we replace the standard patch embedding with a Hierarchical Convolutional Stem. This module progressively downsamples the input image by a factor of 16 through four convolutional stages, smoothly projecting the 3-channel RGB input into the 1024dimensional latent space required by the DINOv3-Large backbone.

The exact architectural specifications are detailed in Table 7. For the intermediate layers, we utilize Spatial Root Mean Square Normalization (TRMS2d) and SiLU activation functions to ensure stable gradient flow and robust feature extraction at the earliest stages.

#### B.2 Decoder Architecture

We adopt the highly efficient decoder architecture proposed in DC-AE [17], tailored to our specific hyperspherical latent space. The decoder progressively upsamples the latent representation z ∈

Table 7: Architectural configuration of the Hierarchical Convolutional Patch Embedding.

Layer Kernel Size Stride Output Channels Norm Activation Input Image - - 3 - -

- ConvLayer 1 7 × 7 2 64 TRMS2d SiLU
- ConvLayer 2 3 × 3 2 128 TRMS2d SiLU
- ConvLayer 3 3 × 3 2 256 TRMS2d SiLU
- ConvLayer 4 3 × 3 2 1024 None None

z×Wz×1024 back to the pixel space xˆ ∈ RH×W×3. The detailed layer-by-layer configuration is summarized in Table 8. Across all blocks, we utilize TRMS2d and SiLU activation. For the upsampling stages, we employ InterpolateConv blocks with channel matching, utilizing a duplicating shortcut connection to preserve structural integrity.

RH

Table 8: Architectural configuration of the HAE Decoder.

Stage Width (Channels) Depth (Blocks) Block Type Input Latent 1024 - -

- Stage 1 1024 3 EViT-GLU
- Stage 2 1024 3 EViT-GLU
- Stage 3 512 3 ResBlock
- Stage 4 512 5 ResBlock
- Stage 5 256 5 ResBlock Output Image 3 - Conv2D (ReLU)

#### B.3 Stage 1: Semantic-Structural Alignment Hyperparameters

In Stage 1, we optimize the hierarchical convolutional patch embedding alongside the decoder to establish a stable semantic latent space. The model is trained using the AdamW optimizer with a base learning rate of 1 × 10−5 for a total of 20 epochs. To ensure stable convergence, we employ Exponential Moving Average (EMA) and mixed-precision (BF16) training. The comprehensive hyperparameter configurations are detailed in Table 9.

Table 9: Hyperparameters for Stage 1 Autoencoder Training.

Hyperparameter Value

Optimization Settings Optimizer AdamW Learning Rate 1 × 10−5 Weight Decay 0.0 Optimizer Betas (0.9,0.999) Global Batch Size 512 Precision BF16 EMA Decay (βEMA) 0.9999 Total Epochs 20

Loss Objective Weights

Reconstruction Weight (λL1) 1.0 Perceptual Weight (λlpips) 1.0 Cosine Similarity Weight (λcos) 0.5

#### B.4 Stage 2: Adversarial Adaptation with Stochastic Latent Smoothing

Building upon the stable semantic space established in Stage 1, Stage 2 introduces adversarial training and stochastic latent smoothing to bridge the domain gap for downstream generative modeling. In this stage, we freeze the pre-trained DINOv3 encoder and train only the decoder alongside a feature-level

discriminator (DINO-Disc). Following StyleGAN-T [21] and RAE [20], this DINO-Disc performs discrimination in the feature space of a pre-trained DINO model to ensure logically consistent textures.

To ensure stable adversarial dynamics, we apply a two-stage warmup strategy. For the first two epochs, the discriminator is trained independently to adapt to the generated distribution, while the generator is optimized solely with reconstruction losses. From the third epoch onwards, the adversarial loss (λadv = 0.5) is fully activated. Specifically, we employ a hinge loss formulation for the adversarial objective (LGAN), updating the overall training loss as follows:

LStage2 = LStage1 + λadvLGAN(x,xˆ) (13)

Notably, we disable the cosine similarity loss (λcos = 0.0) in this stage, as the encoder is frozen and further directional constraints are unnecessary.

Crucially, we implement a progressive noise schedule for the stochastic latent smoothing. The spherical noise scale τ is initialized at 0.8 and linearly increased to 1.2 over the first 20 epochs, after which it remains constant until the end of training at epoch 40. This curriculum smoothing effectively thickens the latent manifold without abruptly collapsing the reconstruction capability. Both the generator (decoder) and the discriminator are optimized using AdamW with a reduced learning rate of 5 × 10−5 and modified momentum betas of (0.5,0.9) to stabilize the GAN training. The complete hyperparameter settings are summarized in Table 10.

Table 10: Hyperparameters for Stage 2 Autoencoder Training (Adversarial Adaptation).

#### Hyperparameter Value

Optimization Settings Optimizer (Generator & Discriminator) & AdamW Learning Rate 5 × 10−5 Weight Decay 0.0 Optimizer Betas (0.5,0.9) Global Batch Size 512 Precision BF16 EMA Decay (βEMA) 0.9975 Total Epochs 40 Discriminator Warmup Epochs 2

Stochastic Latent Smoothing Noise Scale (τ) Schedule 0.8 → 1.2 (Epoch 0–20)

1.2 (Epoch 20–40) Loss Objective Weights

Reconstruction Weight (λL1) 1.0 Perceptual Weight (λlpips) 1.0 Adversarial Weight (λadv) 0.5 Cosine Similarity Weight (λcos) 0.0

### C Ablation on Decoder Latent Scaling Factor

As demonstrated in Section 3.3, our fully trained decoder exhibits strong robustness to latent magnitude shifts. However, during the image generation phase, the specific scaling factor svae applied to the latents generated via Riemannian Flow Matching (RFM) significantly impacts the final visual quality. Because the RFM process outputs generative trajectories strictly on a hypersphere of radius R = 32.0, these uncalibrated vectors must be properly rescaled before being passed to the decoder to synthesize the final image.

To determine the optimal decoding scale for these generated latents, we conducted a systematic ablation study by sweeping the parameter svae from 8.0 to 20.0 during the inference stage. As illustrated by the generative metrics in Figure 7, the performance exhibits a clear U-shaped trajectory

with respect to the scaling factor. Extremely low scaling factors excessively compress the latent variance, yielding a poor gFID of 7.36 at svae = 8.0. In this constrained regime, the restricted magnitude fails to sufficiently excite the decoder’s feature maps, preventing the network from synthesizing intricate high-frequency details.

As the scale increases, the generative quality improves sharply, reaching a clear "sweet spot" at svae = 14.0 with an optimal gFID of 2.65. This qualitative progression is explicitly visualized in Figure 8. At lower scaling factors (e.g., svae = 8.0 and 10.0), the constrained feature magnitude prevents the decoder from forming coherent geometry, resulting in images that suffer from structural fragmentation and severe tearing artifacts. As the scale increases, the generative quality improves sharply, reaching a clear "sweet spot" at svae = 14.0. At this optimal scale, the model achieves the highest image quality, perfectly resolving fine textures and intricate high-frequency details. Conversely, excessively high scaling values (e.g., svae = 18.0 and 20.0) push the latent activations far beyond the decoder’s learned optimal range. This disruption in feature statistics causes the decoding process to over-smooth the outputs, leading to a noticeable loss of texture and increasingly blurry images. Thus, we adopted svae = 14.0 as the default rescaling factor during inference to optimally align the generated latent statistics with the decoder’s representational capacity.

[Figure 7]

- Figure 7: Effect of the latent scaling factor (svae) on generative performance. The model achieves an optimal gFID balance at svae = 14.0.

[Figure 8]

- Figure 8: Generated Golden Retriever samples across different scales: (a) 8.0, (b) 10.0, (c) 14.0, (d) 18.0, and (e) 20.0. As shown, svae = 14.0 best preserves the intricate high-frequency details.

### D Limitations

While HAE demonstrates strong training efficiency and competitive generation quality, several limitations remain. First, our generative model has currently been trained for only 550 epochs, and we have not yet fully explored longer training regimes. Nevertheless, even at this intermediate training budget, HAE with LightningDiT-XL already achieves performance comparable to RAE with LightningDiT-XL trained for 800 epochs, suggesting favorable convergence efficiency. Second, our Riemannian Flow Matching training currently requires full-precision computation for stable

optimization. In preliminary experiments, BF16 training led to numerical instability. As a result, our current implementation sacrifices some of the memory and throughput advantages typically provided by mixed-precision training. Finally, although our method substantially improves the efficiency of standard LightningDiT-XL on hyperspherical latents, it does not yet outperform the strongest specialized baselines such as RAE with DiTDH. In particular, DiTDH uses a heavier architecture, and future work should investigate whether the proposed hyperspherical formulation can be combined with such stronger transformer heads or further architectural scaling.

### E Uncurated Generated Samples

In this section, we present uncurated 256 × 256 samples generated by LightningDiT-XL to provide a comprehensive view of its generative performance. All images were generated using the Riemannian Euler sampler for 50 steps with a Classifier-Free Guidance (CFG) scale of 1.1.

[Figure 9]

- Figure 9: Uncurated 256 × 256 samples of class Goldfinch at CFG 1.1.

[Figure 10]

Figure 10: Uncurated 256 × 256 samples of class Macaw at CFG 1.1.

[Figure 11]

Figure 11: Uncurated 256 × 256 samples of class Golden Retriever at CFG 1.1.

[Figure 12]

Figure 12: Uncurated 256 × 256 samples of class Polar bear at CFG 1.1.

[Figure 13]

Figure 13: Uncurated 256 × 256 samples of class Jellyfish at CFG 1.1.

[Figure 14]

Figure 14: Uncurated 256 × 256 samples of class Monarch butterfly at CFG 1.1.

[Figure 15]

Figure 15: Uncurated 256 × 256 samples of class Balloon at CFG 1.1.

[Figure 16]

Figure 16: Uncurated 256×256 samples of class Cliff at CFG 1.1.

[Figure 17]

Figure 17: Uncurated 256 × 256 samples of class Daisy at CFG 1.1.

[Figure 18]

Figure 18: Uncurated 256 × 256 samples of class Volcano at CFG 1.1.

[Figure 19]

Figure 19: Uncurated 256 × 256 samples of class Hamburger at CFG 1.1.

[Figure 20]

Figure 20: Uncurated 256 × 256 samples of class Icecream at CFG 1.1.

