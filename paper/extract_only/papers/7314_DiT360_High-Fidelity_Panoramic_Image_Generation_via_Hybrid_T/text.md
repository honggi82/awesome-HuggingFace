# arXiv:2510.11712v1[cs.CV]13Oct2025

## DIT360: HIGH-FIDELITY PANORAMIC IMAGE GENERATION VIA HYBRID TRAINING

### Haoran Feng1,2∗ Dizhe Zhang1†∗ Xiangtai Li3 Bo Du4 Lu Qi1,4✉

1 Insta360 Research 2 Tsinghua University 3 Nanyang Technological University 4 Wuhan University

[Figure 1]

Figure 1: Visualization of DiT360’s results. The shown examples include text-to-panorama generation, inpainting, and outpainting, together with comparisons against existing methods.

ABSTRACT

In this work, we propose DiT360, a DiT-based framework that performs hybrid training on perspective and panoramic data for panoramic image generation. For the issues of maintaining geometric fidelity and photorealism in generation quality, we attribute the main reason to the lack of large-scale, high-quality, realworld panoramic data, where such a data-centric view differs from prior methods that focus on model design. Basically, DiT360 has several key modules for inter-domain transformation and intra-domain augmentation, applied at both the pre-VAE image level and the post-VAE token level. At the image level, we incorporate cross-domain knowledge through perspective image guidance and panoramic refinement, which enhance perceptual quality while regularizing diversity and photorealism. At the token level, hybrid supervision is applied across multiple modules, which include circular padding for boundary continuity, yaw loss for rotational robustness, and cube loss for distortion awareness. Extensive

∗ Equal contribution † Project lead ✉ Corresponding author

experiments on text-to-panorama, inpainting, and outpainting tasks demonstrate that our method achieves better boundary consistency and image fidelity across eleven quantitative metrics. Our code is available at https://github.com/Insta360Research-Team/DiT360.

- 1 INTRODUCTION

With the growing demand for spatial intelligence (Yang et al., 2025a; Chen et al., 2024; Wu et al., 2025), panoramic image generation (Lin et al., 2025) has become critical considering its ability to capture the full 360° field of view. Unlike conventional image generation on perspective views (Esser

- et al., 2024; Podell et al., 2023; Tian et al., 2024; Black Forest Labs, 2024), panoramic image generation remains challenging due to its unique characteristics such as severe distortions in polar regions, which in turn hinders its wider deployment in applications such as AR/VR (Wang et al., 2025b; Shi et al., 2025) and autonomous driving (Yue et al., 2025; Qi et al., 2019).

To address this issue, existing methods usually focus on specific model design based on equirectangular projection (ERP) (Ye et al., 2024; Huang et al., 2025; Kalischek et al., 2025; Zhang et al., 2024; Xie, 2025; Sun et al., 2025; Team et al., 2025; Ni et al., 2025; Sun et al., 2025), with or without the assistance of cubemaps (CP) (Bar-Tal et al., 2023; Li & Bansal, 2023; Shi et al., 2023; Tang et al., 2023; Park et al., 2025; Yang et al., 2025b), where both ERP and CP are common panoramic representations.

Despite the success achieved, these works still struggle with perceptual realism and geometric fidelity due to the scarcity of high-quality real-world panoramic data and the over-reliance on simulated one.

A heuristic solution is to exploit 360° data from media platforms such as YouTube, but its direct use for training is impractical since panoramic data requires domain-specific curation, including horizon correction and aesthetic filtering, which remain largely unexplored. Thus, one question raised: how can models be endowed with real-world knowledge when only limited panoramic data is available?

By this motivation, we propose DiT360, a DiT-based framework (Peebles & Xie, 2023), which adopts a hybrid training strategy that combines limited synthetic panoramic data with well-curated, high-quality perspective images to enhance photorealism and geometric fidelity simultaneously. To fully realize the merit of this hybrid paradigm, it is essential to leverage knowledge from the two domains at different representation levels. Accordingly, the DiT360 incorporates several key modules for inter-domain transformation and intra-domain augmentation, applied at both the pre-VAE image level and the post-VAE token level. At the image level, the focus is on regularization across different domains, where existing panoramic data is regularized through masking and inpainting to remove spatial-variant artifacts in polar regions, while perspective data is regularized into the panoramic space through projection-aware methods to provide photorealistic guidance. At the token level, the focus is on geometry-aware supervision in the latent space. Circular padding aims to address the boundary continuity problem of ERP images, where the left/right edge correspond to inherently periodic 0°/360° longitude. In addition, global rotational consistency is enforced through rotation-consistent yaw loss, while distortion-aware cube loss provides complementary supervision beyond ERP that guides the model toward consistent and high-fidelity panoramic representations.

The extensive experiments demonstrate that DiT360 with hybrid training can perform better than existing text-to-panorama methods in boundary consistency and image fidelity, as evidenced by both quantitative metrics and qualitative visualizations. For example, DiT360 achieves state-of-the-art performance on the Matterport3D validation set, surpassing prior methods across nine metrics like FID, Inception Score, and BRISQUE. Beyond text-to-panorama generation, DiT360 naturally supports inpainting and outpainting tasks without additional finetuning enabled by its built-in masking and inpainting strategy. Furthermore, our method can produce high-resolution and photo-realistic panoramic images benefited by high-quality perspective data. Our main contributions are summarized as follows:

- • We present DiT360, a DiT-based framework with hybrid training that leverages both perspective and panoramic data to preserve photorealism and geometric fidelity. Unlike prior approaches that

- primarily focus on model design, DiT360 emphasizes the effective utilization of multi-domain data to achieve superior generation quality.
- • The proposed hybrid paradigm is realized through multi-level mechanisms, where image-level regularization refines existing panoramas and leverages perspective data to enhance diversity and photorealism, while token-level supervision in the latent space enforces geometric consistency through rotation- and distortion-aware constraints.
- • Extensive quantitative and qualitative experiments on three tasks including text to image, inpainting and outpainting demonstrate that DiT360 outperforms existing methods in boundary consistency, image fidelity, and overall perceptual quality. The user study conducted further confirms that our method aligns better with human preferences.

- 2 RELATED WORK

Text-to-Image Diffusion Models. Diffusion models have replaced earlier approaches (Kingma & Welling, 2022; Goodfellow et al., 2020) as the dominant paradigm in image generation, achieving high-quality and diverse synthesis by reversing a gradual noising process (Dhariwal & Nichol, 2021; Qi et al., 2023; Nichol et al., 2022; Saharia et al., 2022; Rombach et al., 2022; Ramesh et al., 2022; Qi et al., 2024). Among these, the Latent Diffusion Model (LDM) (Rombach et al., 2022) wrapped with UNet structure introduced denoising in the latent space, enabling scalable high-resolution generation (Podell et al., 2023). More recently, transformer-based architectures (Peebles & Xie, 2023; Vaswani et al., 2017) have been adopted using explicit positional encoding and attention operation to further improve performance (Black Forest Labs, 2024; Esser et al., 2024; Yu et al., 2025; Ma

- et al., 2024), and are emerging as a new paradigm for better scalability and stronger results. We note that both UNet- and transformer-based structures benefit from the large-scale perspective datasets. Motivated by this, we leverage perspective data to compensate for the limited scale of panoramic data by inter-domain transformation and projection.

Panoramic image Generation. Early panoramic image generation mainly relied on outpaintingbased methods (Akimoto et al., 2022; Dastjerdi et al., 2022; Wang et al., 2022; 2023; Wu et al., 2023b;c; Lu et al., 2024; Qi et al., 2022), which reconstruct a full 360° view from partial observations, such as narrow field of view (NFoV), but often suffer from limited flexibility and content diversity. With advances in text-to-image generation, research has shifted towards text-to-panorama generation for more controllable results and can be broadly divided into two categories. The first kinds of approaches (Fang et al., 2023; H¨ollein et al., 2023; Yu et al., 2023; Bar-Tal et al., 2023; Lee et al., 2023; Li & Bansal, 2023; Shi et al., 2023; Tang et al., 2023; Park et al., 2025; Yang et al., 2025b) generate panoramic images by stitching multiple perspective views. However, they often suffer from limited perceptual realism because of repeated objects and poor geometric fidelity, such as discontinuities. To alleviate this problem, some work (Song et al., 2023; Ye et al., 2024; Huang

- et al., 2025; Kalischek et al., 2025) adopts cube mapping, which better aligns with the spherical geometry of panoramic images; yet discontinuities across cube faces remain unresolved, along with additional computational and temporal overhead. Another line of work (Chen et al., 2022; Shum et al., 2023; Zhang et al., 2023; Feng et al., 2023; Ai et al., 2024; Wang et al., 2024; Yang et al., 2024a; Zhang et al., 2024; Xie, 2025; Sun et al., 2025; Team et al., 2025; Ni et al., 2025; Wang et al., 2025a; Lu et al., 2025) trains models directly on equirectangular images, preserving global continuity and allowing the model to learn distortion patterns. However, these methods struggle to maintain boundary consistency that requires seamless alignment at the 0°/360° longitude and degrade in regions with severe polar distortion, leading to reduced geometric fidelity. Although recent works (Sun et al., 2025; Park et al., 2025; Zhang et al., 2024) attempt to alleviate these issues through alternative convolutional designs, they remain limited in practice and are less compatible with pretrained models. In addition, all these methods are constrained by the limited quality of panoramic datasets, often inheriting polar degradation and producing rendered-like appearances that lack perceptual realism. In contrast, we employ a hybrid training strategy that enables the generation of high-resolution panoramic images with high perceptual realism, producing sharp and detailed content (Qi et al., 2021) and strong geometric fidelity, ensuring correct polar distortion and seamless boundaries.

[Figure 2]

- Figure 2: Overview of the DiT360 hybrid training pipeline. For the perspective branch, we employ (a) perspective image re-projection to transfer perspective knowledge to panoramic domain. For the panoramic branch, we first apply (b) panoramic refinement to remove polar blurring and then introduce (c) position-aware circular padding, (d) rotation-consistent yaw loss and (e) distortionaware cube loss for token-level hybrid supervision.

- 3 METHOD

- As illustrated in Fig. 2, DiT360 is a novel framework for generating panoramic images, which improves photorealism and geometric fidelity through hybrid training at both the image and token levels. In the following sections, We first present the preliminaries and overall design of DiT360 in Sec. 3.1. We then introduce several key modules of the hybrid paradigm from two complementary perspectives: image-level regularization in Sec. 3.2, and token-level supervision in Sec. 3.3. Finally, we show that DiT360 natively supports extended generation tasks such as inpainting and outpainting without additional training, as detailed in Sec. 3.4.

- 3.1 DIT360

Revisit Diffusion Transformer (DiT). Recent diffusion models increasingly adopt the DiT architecture (Peebles & Xie, 2023), which uses a transformer (Dosovitskiy, 2020) to process latent sequences of post-VAE image tokens X ∈ RN×d, where N is the sequence length and d denotes the embedding dimension. To capture spatial structure, DiT employs Rotary Positional Embeddings (RoPE) (Su et al., 2024), which inject coordinate-dependent rotations into the image tokens, thereby allowing the model to effectively encode both relative and absolute positional information. In addition, DiT adopts a flow-based scheduler to progressively denoise the latent representation, typically conditioned on a text promp c. Its training objective is the standard denoising score-matching loss, computed as:

#### L = EX,c,ϵ,t ∥ϵ − ϵθ(Xt,c,t)∥22 , (1)

where Xt denotes the noise latent in timestep t, ϵ is the added Gaussian noise and ϵθ represents the predicted noise of the model.

Overview of DiT360. Building upon DiT, we introduce DiT360 for panoramic image generation. Fig. 2 illustrates the proposed framework, which adopts a hybrid paradigm to jointly exploit perspective and panoramic data through two training branches. The key modules enabling hybrid training are categorized into image-level regularization and token-level supervision. At the image level, perspective image guidance and panoramic refinement introduce cross-domain knowledge to enhance perceptual quality while regularizing diversity and photorealism. At the token level, hybrid supervision across multiple objectives is conducted, which includes circular padding for boundary continuity, yaw loss for rotational robustness, and cube loss for distortion awareness. Together, this hybrid design operates across multiple representation levels to achieve perceptual photorealism and geometric fidelity.

[Figure 3]

- Figure 3: Panoramic image refinement pipeline. The ERP panorama is converted into a cubemap, where pre-defined masks are applied to the central regions of the top and bottom faces. These masked regions are then reconstructed with an inpainting model and reprojected to ERP. In the figure, orange boxes represent blurry areas, and red dashed boxes indicate inpainted cubes.

- 3.2 IMAGE-LEVEL REGULARIZATION

- At the image level, we adopt a hybrid regularization strategy that improves generation quality through inter-domain transformation, combining refinement of existing panoramas with the transfer of photorealistic knowledge from perspective views.

Panoramic image refinement. The availability of high-quality, real-world panoramic datasets remains severely restricted, with Matterport3D (Chang et al., 2017) being one of the most widely adopted due to its large scale and high fidelity. Nevertheless, images in this dataset frequently exhibit blurring in the polar regions, as shown in Fig. 3, which in turn hampers the quality of downstream panoramic image generation. To mitigate blurring artifacts in the polar regions, we transform panoramic ERPs into cubemap representations, where well-established perspective-domain inpainting can be directly applied. We first fix a binary mask M for each blurred cube face I to localize the inpainting area. Specifically, for H = W = 1024, we mask out the central region:

M(u,v) =

- 0, if 256 ≤ u,v < 768,
- 1, otherwise,

(2)

where (u,v) denotes pixel coordinates. The masked image is then obtained as

#### Imask = I ⊙ M + (1 − M) · Imiss, (3)

where ⊙ denotes element-wise multiplication and Imiss is a white image of the same resolution. Finally, the inpainting model (Labs et al., 2025) is then applied to Imask to reconstruct the missing region and produce Iˆ, which is then transformed back into the ERP space to obtain blur-free, highfidelity panoramas.. This process serves as an image-quality regularization step, yielding clearer training images while constraining panoramas to retain inherent distortion characteristics, as illustrated in Fig. 3.

Perspective image guidance. In addition, we leverage high-quality realistic perspective images from the Internet to regularize the panoramic domain by transferring photorealistic knowledge. Specifically, as illustrated in Fig. 2a, a perspective image is regarded as a cubemap lateral face and then converted back into the ERP representation with a corresponding mask M. We restrict the re-projection to the lateral faces, as the top and bottom faces usually correspond to sky or ground regions, which require perspective images from uncommon viewing angles that are rarely covered in the dataset. During training, we directly apply the mean squre error (MSE) loss from Flux (Black Forest Labs, 2024) to the re-projected ERP, restricting it to the masked regions to avoid contamination from unrelated panoramic areas, yielding:

#### Lperspective = LMSE(ϵ ⊙ M,ϵˆθ ⊙ M), (4)

where ϵ and ϵˆθ denote the sampled noise and the reparameterized predicted noise, respectively. This strategy provides effective image-level guidance through cross-domain knowledge adaptation, exposing the model to more diverse scenes and thereby increasing the generation diversity. More importantly, the incorporated perspective knowledge regularizes the model toward photorealistic fidelity, which remains underexplored in prior works.

- 3.3 TOKEN-LEVEL SUPERVISION

At the token level, DiT360 adopts a hybrid training strategy that balances complementary supervision at the post-VAE token level, simultaneously enhancing boundary continuity, rotational robustness, distortion awareness and perceptual quality. Specifically, we introduce three mechanisms applied to noisy tokens of panoramic inputs: position-aware circular padding for seamless boundary coherence, yaw loss for global rotation consistency, and cube loss for precise supervision of ERP distortion patterns. Together, we propose a hybrid loss design to ensure fine-grained token-level supervision while maintaining balanced generation quality across multiple dimensions.

Position-aware Circular Padding. Panoramic images cover the full 360° horizontal field, making it critical to maintain continuity across image boundaries. To address this challenge, we propose a token-based circular padding mechanism specific to DiT-based frameworks that takes advantage of the inherent correspondence between explicit positional encoding and image content. This property ensures that latent tokens at the same spatial position generate consistent visual features, which we exploit to enhance boundary coherence without introducing additional architectural complexity. Specifically, as illustrated in Fig. 2c, after the VAE compression and the subsequent noise injection, we reshape the latent tokens Xt ∈ RN×d into Xt ∈ RH×W×d. We then apply a circular padding along the width dimension. Formally, let X0 and X−1 denote the first and last column features, respectively. We then concatenate them to obtain the padded tensor

#### X˜t = X−1, Xt, X0 ∈ RH×(W+2)×d. (5)

The same operation is applied to the positional encoding, which encourages the model to learn continuity specifically between adjacent columns across the boundary.

Rotation-consistent Yaw Loss. To enforce global rotational robustness, we introduce yaw loss that offers token-level supervision on the model’s behavior under spherical rotations along the yaw axis, as illustrated in Fig. 2d. Unlike the standard diffusion loss, the corresponding yaw loss captures the non-linear effects of yaw rotations and constrains the model to produce consistent predictions across different viewing angles. Formally, with the reparameterized noise ϵ and predicted noise ϵθ, we randomly select a yaw rotation angle a and define the rotated features as:

ϵyaw = Rotate(Xt − ϵ,a), ϵθ,yaw = Rotate(ϵθ,a), (6) where Rotate(·,a) denotes the equirectangular panorama rotated by angle a along the yaw axis. The yaw loss is then computed as the mean squared error (MSE) between the predicted and target rotated noise features:

#### Lyaw = E |ϵθ,yaw − ϵyaw|22 . (7)

Distortion-aware Cube Loss. To effectively model distortion patterns and preserve fine details, we introduce a cube loss based on the cubemap representation of panoramasas, as shown in Fig. 2e. Direct supervision on equirectangular projections often causes the model to reproduce similar distorted appearances rather than learn the precise structural patterns, which leads to incorrect generation details with dealing with polar-region distortions. To address this challenge, we project both sampled and predicted noise onto cube faces and apply face-wise supervision, thereby transferring model’s strength in perspective priors to the panoramic domain to preserve structural distortion patterns. Further analysis are provided in Sec. A. Formally, let Xt denote the noisy latent at time step t in the forward diffusion process, ϵ denote the reparameterized Gaussian noise, and ϵθ denote the noise predicted by the model. We define the cube-space features by applying a cube-mapping operation:

ϵcube = CubeMap(Xt − ϵ), ϵθ,cube = CubeMap(ϵθ), (8) where CubeMap(·) transforms an equirectangular panorama into six cube faces. Then, the cube loss is computed as the MSE between the predicted and target cube-space noise features:

#### Lcube = E |ϵθ,cube − ϵcube|22 . (9)

It is worth noting that we apply both cube and yaw losses directly in the latent noise space. While a natural alternative is to compute them in the latent token space—by predicting latents from noise and

- Table 1: Quantitative comparison results on text-to-panorama generation. We highlight the best result in bold and the second best with underline.

Methods FID↓ FIDclip↓ FIDpole↓ FIDequ↓ FAED↓ IS↑ CS↑ QAquality↑ QAaesthetic↑ BRISQUE↓ NIQE↓

PanFusion 124.87 120.75 182.09 108.12 11.06 1.30 28.35 3.83 3.56 27.38 4.31 MVDiffusion 108.19 117.26 - - 4.39 1.58 34.65 3.97 3.25 44.79 4.91 SMGD 46.72 45.04 65.69 34.84 3.29 1.40 31.14 4.05 3.77 30.35 4.75 PAR 47.72 47.26 76.93 27.39 2.97 1.34 33.85 3.91 3.54 32.26 4.38 WorldGen 67.11 62.97 79.32 33.45 3.29 1.40 34.61 4.30 3.59 32.31 4.82 Matrix-3D 60.91 56.70 77.21 26.73 3.08 1.56 34.59 4.48 3.78 16.37 3.95 LayerPano3D 62.82 60.34 80.37 38.67 2.98 1.50 34.40 4.73 3.93 33.91 3.79 HunyuanWorld 76.75 75.65 106.58 41.75 2.91 1.53 34.73 4.67 3.85 39.12 5.18

Ours 42.88 41.60 50.88 24.77 2.91 1.60 34.68 4.69 4.19 10.25 3.72

comparing with ground-truth latents—our experiments show that noise predictions already encode rich spatial and structural information due to the coupling of noise and semantics in the diffusion objective, making them suitable for spatial supervision. In addition, operating in the noise space aligns the auxiliary losses with the flow-based scheduler, thereby improving training stability.

Hybrid Loss Design. To better balance geometric fidelity and perceptual quality, we adopt a hybrid loss design that retains the MSE loss from Flux (Black Forest Labs, 2024) as the principal objective and augment it with yaw loss and cube loss described above. The overall training loss Lpano for the panoramic branch is then calculated as:

Lpano = LMSE + λ1Lcube + λ2Lyaw, (10)

where λ1 and λ2 represent the balancing coefficients. This token-level hybrid supervision ensures that general perceptual quality, rotational robustness and distortion fidelity are jointly preserved within a unified training framework.

- 3.4 MORE APPLICATIONS

Benefiting from the strong robustness of our method, we perform feature replacement via inversion (Feng et al., 2025) to enable image inpainting and outpainting without additional training. In addition, our model natively supports high-resolution generation, with all training conducted at 1024×2048 resolution. The results in Fig. 1 demonstrate the generalization capability of our approach beyond the primary generation task, with more analysis and results provided in Sec. B.

- 4 EXPERIMENTS

- 4.1 SETUP

DiT360 is developed on top of Flux (Black Forest Labs, 2024) with LoRA (Hu et al., 2021) incorporated into the attention layers. For improved perceptual realism and geometric fidelity, we design a hybrid training strategy that combines high-quality Internet landscape images with large-scale panoramas from Matterport3D (Chang et al., 2017). To assess the effectiveness of our approach, we adopt a diverse set of complementary metrics covering realism, diversity, text–image alignment, and perceptual quality, ensuring a comprehensive assessment of model performance. More detailed descriptions of the implementation, dataset preprocessing, and metric definitions are in Sec. C.

- 4.2 MAIN RESULTS AND COMPARISONS

In this section, we present our main experimental results and conduct a comprehensive comparison with representative baselines, including PanFusion (Zhang et al., 2024), MVDiffusion (Tang et al., 2023), SMGD (Sun et al., 2025), PAR (Wang et al., 2025a), WorldGen (Xie, 2025), Matrix-3D (Lu

- et al., 2025), LayerPano3D (Yang et al., 2024b), and HunyuanWorld (Team et al., 2025). These methods span diverse architectures, enabling a comprehensive evaluation of our approach. In addition to quantitative and qualitative comparisons, we also present a user study in Sec. E and further results in Sec. F.

[Figure 4]

- Figure 4: Qualitative comparisons on panorama generation. The representative artifacts are highlighted with red boxes. More complete results are provided in Sec. D.

Qualitative Comparisons. We provide qualitative comparisons with baseline methods in Fig. 4 and highlights artifacts with red boxes. SMGD (Sun et al., 2025) and PAR (Wang et al., 2025a) propose alternative paradigms—structural modifications or autoregression—but struggle with detail fidelity, often producing cluttered or less precise results. Moreover, insufficient data quality leads to pronounced distortions near the polar regions, resulting in poor perceptual realism. Recent advances in Diffusion Transformers (DiT) (Peebles & Xie, 2023) have led to their adoption as backbones in several panorama generation methods. Matrix-3D improves boundary alignment yet struggles with fine-grained details, suffering from limited geometric fidelity. LayerPano3D (Yang et al., 2024b) and HunyuanWorld (Team et al., 2025) leverage large amounts of synthetic data, which improves geometric fidelity to some extent, but results in render-like appearances that compromise perceptual realism; additionally, iterative denoising introduces further artifacts. In contrast, our method generates panoramas with high perceptual realism and geometric fidelity, producing sharp, detailpreserving images with strong lateral consistency and effectively mitigated distortions.

Quantitative Comparisons. We further conduct quantitative evaluations to validate the effectiveness of our approach, with results reported in Tab. 1. Our method ranks first on nearly all benchmarks and shows consistently strong performance across most metrics. Although our approach slightly underperforms the top methods on CLIP Score and the quality branch of Q-Align, the gaps are marginal and largely attributable to the fact that both metrics are designed for perspective images, which may

[Figure 5]

- Figure 5: Ablation results of different settings. Artifacts are marked by color-coded bounding boxes: red for spurious details, yellow for boundary discontinuities, and green for incorrect distortions.

- Table 2: Ablation study of different model components on text-to-panorama generation. Best results are in bold, second best are underlined, and “Pi guidance” denotes perspective image guidance.

Methods FID↓ FIDclip↓ FIDpole↓ FIDequ↓ FAED↓ IS↑ CS↑ QAquality↑ QAaesthetic↑ BRISQUE↓ NIQE↓

Flux + LoRA 46.69 45.90 66.03 28.91 3.23 1.51 34.39 4.40 3.97 17.02 3.97 w/ circular padding 43.71 42.36 61.32 27.51 3.04 1.54 34.44 4.51 3.98 13.61 3.82 w/ cube loss 44.40 43.75 60.16 26.30 3.01 1.57 34.62 4.41 3.92 15.68 3.89 w/ yaw loss 44.63 43.90 64.19 26.99 2.98 1.56 34.53 4.37 3.94 15.96 3.92 w/ pi guidance 46.03 44.92 63.72 27.81 2.95 1.48 34.42 4.54 4.02 16.94 3.83

Ours (w/ all) 42.88 41.60 50.88 24.77 2.91 1.60 34.68 4.69 4.19 10.25 3.72

not fully reflect the quality and fidelity of panoramas. Collectively, the results support our qualitative observations and demonstrate the effectiveness and robustness of our approach in generating high-quality panoramas.

- 4.3 ABLATION STUDY

To assess the contribution of each component, we conduct ablation studies using a combination of Flux (Black Forest Labs, 2024) and LoRA (Hu et al., 2021) as the baseline. We ablate four key modules: position-sensitive circular padding, distortion-sensitive cube loss, rotation-consistent yaw loss, and perspective image guidance and evaluate their impact in Tab. 2 and Fig. 5.

Circular padding significantly enhances consistency across image boundaries and also improves overall image quality, reflected in reductions of FID (Heusel et al., 2018) and BRISQUE (Mittal et al., 2012), because the identical positional encoding on the left and right edges allows the model to learn correct boundary correspondences.

Cube loss mainly refines fine-grained details and reduces artifacts by applying additional supervision on the cubemap representation, enabling the model to learn accurate panoramic distortions. This results in substantially fewer artifacts in the polar regions and thus largely improved IS and CS that are more related to the visual semantics.

Yaw loss improves global rotation consistency and structural coherence, explaining its superior performance on FAED (Oh et al., 2021) where the autoencoders used are pre-trained by panoramic images. This is because that we supervise the model on rotated tokens to explicitly enforce fullimage rotation consistency.

Perspective image guidance further enhances local details, enriches visual diversity and effectively mitigates detail-related artifacts, as evidenced in the QAquality and QAaesthetic metrics that are more sensitive to the visual style.

Overall, the components contribute to the perceptual realism and geometric fidelity, and their combination delivers the strongest performance, validating the effectiveness of our framework.

- 5 CONCLUSION

In this paper, we proposed DiT360, a framework for geometry-aware and photorealistic panoramic image generation, built upon a hybrid training strategy that combines limited high-quality panoramic data with large-scale perspective images to enhance both realism and generalization. To fully leverage this hybrid paradigm, we introduce multiple modules across different representation levels, where image-level regularization refines existing panoramas and leverages perspective data to enhance diversity and photorealism, while token-level supervision in the latent space enforces geometric consistency through rotation- and distortion-aware constraints. Extensive experiments on textto-panorama generation, inpainting, and outpainting demonstrate superior image fidelity, boundary consistency, and visual quality. By bridging perspective and panoramic domains across multiple representation levels, DiT360 establishes a strong baseline for future research in 3D scene generation and large-scale open-world environments.

REFERENCES

Hao Ai, Zidong Cao, Haonan Lu, Chen Chen, Jian Ma, Pengyuan Zhou, Tae-Kyun Kim, Pan Hui, and Lin Wang. Dream360: Diverse and immersive outdoor virtual scene creation via transformerbased 360 image outpainting. In IEEE TVCG, 2024. 3

Naofumi Akimoto, Yuhi Matsuo, and Yoshimitsu Aoki. Diverse plausible 360-degree image outpainting for efficient 3dcg background creation. In CVPR, 2022. 3

Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. Multidiffusion: Fusing diffusion paths for controlled image generation.(2023). In arXiv, 2023. 2, 3

Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024. Accessed: 2024-09-23. 2, 3, 5, 7, 9, 15

Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niessner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3d: Learning from rgb-d data in indoor environments. In 3DV, 2017. 5, 7, 15

Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In CVPR,

2024. 2 Zhaoxi Chen, Guangcong Wang, and Ziwei Liu. Text2light: Zero-shot text-driven hdr panorama generation. In ACM Trans. Graph., 2022. 3

Mohammad Reza Karimi Dastjerdi, Yannick Hold-Geoffroy, Jonathan Eisenmann, Siavash Khodadadeh, and Jean-Fran¸cois Lalonde. Guided co-modulated gan for 360 field of view extrapolation. In 3DV, 2022. 3

Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In NeurIPS, 2021. 3

Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. In arXiv, 2020. 4

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In arXiv, 2024. 2, 3

Chuan Fang, Yuan Dong, Kunming Luo, Xiaotao Hu, Rakesh Shrestha, and Ping Tan. Ctrl-room: Controllable text-to-3d room meshes generation with layout constraints. In arXiv, 2023. 3

Haoran Feng, Zehuan Huang, Lin Li, Hairong Lv, and Lu Sheng. Personalize anything for free with diffusion transformer. In arXiv, 2025. 7, 14

Mengyang Feng, Jinlin Liu, Miaomiao Cui, and Xuansong Xie. Diffusion360: Seamless 360 degree panoramic image generation based on diffusion models. In arXiv, 2023. 3

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. In CACM, 2020. 3

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In arXiv, 2015. 16

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In arXiv,

2018. 9, 15 Lukas H¨ollein, Ang Cao, Andrew Owens, Justin Johnson, and Matthias Nießner. Text2room: Extracting textured 3d meshes from 2d text-to-image models. In ICCV, 2023. 3 Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,

and Weizhu Chen. Lora: Low-rank adaptation of large language models. In arXiv, 2021. 7, 9, 15 Yukun Huang, Yanning Zhou, Jianan Wang, Kaiyi Huang, and Xihui Liu. Dreamcube: 3d panorama

generation via multi-plane synchronization. In arXiv, 2025. 2, 3

Nikolai Kalischek, Michael Oechsle, Fabian Manhardt, Philipp Henzler, Konrad Schindler, and Federico Tombari. Cubediff: Repurposing diffusion-based image models for panorama generation. In The Thirteenth ICLR, 2025. 2, 3

Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In arXiv, 2022. 3

Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas M¨uller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space. In arXiv, 2025. 5

Yuseung Lee, Kunho Kim, Hyunjin Kim, and Minhyuk Sung. Syncdiffusion: Coherent montage via synchronized joint diffusions. In NeurIPS, 2023. 3

Jialu Li and Mohit Bansal. Panogen: Text-conditioned panoramic environment generation for vision-and-language navigation. In NeurIPS, 2023. 2, 3

Xin Lin, Xian Ge, Dizhe Zhang, Zhaoliang Wan, Xianshun Wang, Xiangtai Li, Wenjie Jiang, Bo Du, Dacheng Tao, Ming-Hsuan Yang, et al. One flight over the gap: A survey from perspective to panoramic vision. In arXiv, 2025. 2

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In arXiv, 2019. 15 Yuanxun Lu, Jingyang Zhang, Tian Fang, Jean-Daniel Nahmias, Yanghai Tsin, Long Quan, Xun

Cao, Yao Yao, and Shiwei Li. Matrix3d: Large photogrammetry model all-in-one. In arXiv,

2025. 3, 7, 17 Zhuqiang Lu, Kun Hu, Chaoyue Wang, Lei Bai, and Zhiyong Wang. Autoregressive omni-aware outpainting for open-vocabulary 360-degree image generation. In AAAI, 2024. 3

Nanye Ma, Mark Goldstein, Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In arXiv, 2024. 3

Anish Mittal, Anush Krishna Moorthy, and Alan Conrad Bovik. No-reference image quality assessment in the spatial domain. In IEEE Trans. Image Process., 2012. 9, 16

Anish Mittal, Rajiv Soundararajan, and Alan C. Bovik. Making a “completely blind” image quality analyzer. In IEEE Signal Process. Lett., 2013. 16

Jinhong Ni, Chang-Bin Zhang, Qiang Zhang, and Jing Zhang. What makes for text to 360-degree panorama generation with stable diffusion? In arXiv, 2025. 2, 3

Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. In arXiv, 2022. 3

Changgyoon Oh, Wonjune Cho, Daehee Park, Yujeong Chae, Lin Wang, and Kuk-Jin Yoon. Bips: Bi-modal indoor panorama synthesis via residual depth-aided adversarial learning. In arXiv, 2021. 9, 15

Minho Park, Taewoong Kang, Jooyeol Yun, Sungwon Hwang, and Jaegul Choo. Spherediff: Tuningfree omnidirectional panoramic image and video generation via spherical latent representation. In arXiv, 2025. 2, 3

William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 2, 3, 4, 8

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In arXiv, 2023. 2, 3

Lu Qi, Li Jiang, Shu Liu, Xiaoyong Shen, and Jiaya Jia. Amodal instance segmentation with kins dataset. In CVPR, 2019. 2

Lu Qi, Jason Kuen, Jiuxiang Gu, Zhe Lin, Yi Wang, Yukang Chen, Yanwei Li, and Jiaya Jia. Multiscale aligned distillation for low-resolution detection. In CVPR, 2021. 3

Lu Qi, Jason Kuen, Yi Wang, Jiuxiang Gu, Hengshuang Zhao, Philip Torr, Zhe Lin, and Jiaya Jia. Open world entity segmentation. In TPAMI, 2022. 3

Lu Qi, Jason Kuen, Weidong Guo, Tiancheng Shen, Jiuxiang Gu, Jiaya Jia, Zhe Lin, and MingHsuan Yang. High-quality entity segmentation. In ICCV, 2023. 3

Lu Qi, Lehan Yang, Weidong Guo, Yu Xu, Bo Du, Varun Jampani, and Ming-Hsuan Yang. Unigs: Unified representation for image generation and segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 3

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In arXiv, 2021. 16

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. In arXiv, 2022. 3

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In arXiv, 2022. 3

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding, 2022. 3

Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. In arXiv, 2016. 15

Qingyu Shi, Lu Qi, Jianzong Wu, Jinbin Bai, Jingbo Wang, Yunhai Tong, and Xiangtai Li. Dreamrelation: Bridging customization and relation generation. In CVPR, 2025. 2

Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. In arXiv, 2023. 2, 3

Ka Chun Shum, Hong-Wing Pang, Binh-Son Hua, Duc Thanh Nguyen, and Sai-Kit Yeung. Conditional 360-degree image synthesis for immersive indoor scene decoration. In ICCV, 2023. 3

Liangchen Song, Liangliang Cao, Hongyu Xu, Kai Kang, Feng Tang, Junsong Yuan, and Yang Zhao. Roomdreamer: Text-driven 3d indoor scene synthesis with coherent geometry and texture. In arXiv, 2023. 3

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. In Neurocomputing, 2024. 4

Xiancheng Sun, Mai Xu, Shengxi Li, Senmao Ma, Xin Deng, Lai Jiang, and Gang Shen. Spherical manifold guided diffusion model for panoramic image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, 2025. 2, 3, 7, 8, 15

Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In arXiv, 2015. 16

Shitao Tang, Fuyang Zhang, Jiacheng Chen, Peng Wang, and Yasutaka Furukawa. Mvdiffusion: Enabling holistic multi-view image generation with correspondence-aware diffusion. In arXiv,

2023. 2, 3, 7

HunyuanWorld Team, Zhenwei Wang, Yuhao Liu, Junta Wu, Zixiao Gu, Haoyuan Wang, Xuhui Zuo, Tianyu Huang, Wenhuan Li, Sheng Zhang, Yihang Lian, Yulin Tsai, Lifu Wang, Sicong Liu, Puhua Jiang, Xianghui Yang, Dongyuan Guo, Yixuan Tang, Xinyue Mao, Jiaao Yu, Junlin Yu, Jihong Zhang, Meng Chen, Liang Dong, Yiwen Jia, Chao Zhang, Yonghao Tan, Hao Zhang, Zheng Ye, Peng He, Runzhou Wu, Minghui Chen, Zhan Li, Wangchen Qin, Lei Wang, Yifu Sun, Lin Niu, Xiang Yuan, Xiaofeng Yang, Yingping He, Jie Xiao, Yangyu Tao, Jianchen Zhu, Jinbao Xue, Kai Liu, Chongqing Zhao, Xinming Wu, Tian Liu, Peng Chen, Di Wang, Yuhong Liu, Linus, Jie Jiang, Tengfei Wang, and Chunchao Guo. Hunyuanworld 1.0: Generating immersive, explorable, and interactive 3d worlds from words or pixels. In arXiv, 2025. 2, 3, 7, 8, 16, 17

Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. In arXiv, 2024. 2

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 3

Chaoyang Wang, Xiangtai Li, Lu Qi, Xiaofan Lin, Jinbin Bai, Qianyu Zhou, and Yunhai Tong. Conditional panoramic image generation via masked autoregressive modeling. In arXiv, 2025a. 3, 7, 8

Guangcong Wang, Yinuo Yang, Chen Change Loy, and Ziwei Liu. Stylelight: Hdr panorama generation for lighting estimation and editing. In ECCV, 2022. 3

Hai Wang, Xiaoyu Xiang, Yuchen Fan, and Jing-Hao Xue. Customizing 360-degree panoramas through text-to-image diffusion models. In WACV, 2024. 3

Jionghao Wang, Ziyu Chen, Jun Ling, Rong Xie, and Li Song. 360-degree panorama generation from few unregistered nfov images. In arXiv, 2023. 3

Xiaoyuan Wang, Yizhou Zhao, Botao Ye, Xiaojun Shan, Weijie Lyu, Lu Qi, Kelvin CK Chan, Yinxiao Li, and Ming-Hsuan Yang. Holigs: Holistic gaussian splatting for embodied view synthesis. In NeurlPS, 2025b. 2

Diankun Wu, Fangfu Liu, Yi-Hsin Hung, and Yueqi Duan. Spatial-mllm: Boosting mllm capabilities in visual-based spatial intelligence. In arXiv, 2025. 2

Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Yixuan Gao, Annan Wang, Erli Zhang, Wenxiu Sun, Qiong Yan, Xiongkuo Min, Guangtao Zhai, and Weisi Lin. Q-align: Teaching lmms for visual scoring via discrete text-defined levels. In arXiv, 2023a. 16

Tianhao Wu, Chuanxia Zheng, and Tat-Jen Cham. Ipo-ldm: Depth-aided 360-degree indoor rgb panorama outpainting via latent diffusion model. In arXiv, 2023b. 3

Tianhao Wu, Chuanxia Zheng, and Tat-Jen Cham. Panodiffusion: 360-degree panorama outpainting via diffusion. In arXiv, 2023c. 3

Ziyang Xie. Worldgen: Generate any 3d scene in seconds. https://github.com/ ZiYang-xie/WorldGen, 2025. 2, 3, 7

Bangbang Yang, Wenqi Dong, Lin Ma, Wenbo Hu, Xiao Liu, Zhaopeng Cui, and Yuewen Ma. Dreamspace: Dreaming your room space with text-driven panoramic texture propagation. In IEEE VR, 2024a. 3

Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In CVPR, 2025a. 2

Liu Yang, Huiyu Duan, Yucheng Zhu, Xiaohong Liu, Lu Liu, Zitong Xu, Guangji Ma, Xiongkuo Min, Guangtao Zhai, and Patrick Le Callet. Omni2: Unifying omnidirectional image generation and editing in an omni model. In arXiv, 2025b. 2, 3

Shuai Yang, Jing Tan, Mengchen Zhang, Tong Wu, Yixuan Li, Gordon Wetzstein, Ziwei Liu, and Dahua Lin. Layerpano3d: Layered 3d panorama for hyper-immersive scene generation. In arXiv, 2024b. 7, 8

Weicai Ye, Chenhao Ji, Zheng Chen, Junyao Gao, Xiaoshui Huang, Song-Hai Zhang, Wanli Ouyang, Tong He, Cairong Zhao, and Guofeng Zhang. Diffpano: Scalable and consistent text to panorama generation with spherical epipolar-aware diffusion. In NeurIPS, 2024. 2, 3

Jason J Yu, Fereshteh Forghani, Konstantinos G Derpanis, and Marcus A Brubaker. Long-term photometric consistent novel view synthesis with diffusion models. In ICCV, 2023. 3

Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. In ICLR, 2025. 3

Jingtong Yue, Zhiwei Lin, Xin Lin, Xiaoyu Zhou, Xiangtai Li, Lu Qi, Yongtao Wang, and MingHsuan Yang. Roburcdet: Enhancing robustness of radar-camera fusion in bird’s eye view for 3d object detection. In ICLR, 2025. 2

Cheng Zhang, Qianyi Wu, Camilo Cruz Gambardella, Xiaoshui Huang, Dinh Phung, Wanli Ouyang, and Jianfei Cai. Taming stable diffusion for text to 360 panorama image generation. In CVPR,

2024. 2, 3, 7, 15, 17 Qinsheng Zhang, Jiaming Song, Xun Huang, Yongxin Chen, and Ming-Yu Liu. Diffcollage: Parallel generation of large content with diffusion models. In CVPR, 2023. 3

Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. Places: A 10 million image database for scene recognition. In IEEE Trans. Pattern Anal. Mach. Intell., 2017. 16

APPENDIX

- A EFFECT OF SUPERVISION ON POLAR DISTORTIONS

In this section, we further illustrate the effect of cube loss in addressing severe distortions around the polar regions. Figure 6 compares results generated from the same prompt without and with this supervision, showing that incorporating cube loss leads to clearer structures and fewer artifacts in the polar regions.

- B INPAINTING AND OUTPAINTING

DiT360 demonstrates native inpainting and outpainting capabilities without requiring additional training, thereby establishing a unified framework for panoramic image generation. Specifically, inspired by (Feng et al., 2025), we first perform inversion on the input image to obtain its initial

[Figure 6]

- Figure 6: Qualitative comparison of generated panoramas and their top/bottom cube faces without (left) and with (right) cube loss. Red boxes mark regions where polar artifacts are significantly reduced when supervision is applied.

noise representation. At the same time, we extract reference image tokens without positional encodings, along with the associated subject mask. During the early denoising steps, we employ a token replacement strategy. The tokens within the masked or extended regions are substituted with those from the reference image, while preserving the original positional encodings. This time-stepadaptive replacement mechanism ensures faithful reproduction of subject details and spatial consistency. It anchors subject identity in the early phase of generation and naturally guides the model toward coherent content completion. As a result, DiT360 produces consistent and semantically rich results in both inpainting and outpainting tasks. More results are provided in Fig. 7.

- C EXPERIMENT SETTINGS

Implementation Details. We developed DiT360 on top of Flux (Black Forest Labs, 2024), integrating LoRA (Hu et al., 2021) into the attention layers. The model was fine-tuned on 5 H20 GPUs using AdamW (Loshchilov & Hutter, 2019) with a learning rate of 2 × 10−5 for 20 epochs, a batch size per GPU of 1, and a gradient accumulation of 3. Our experiments revealed that the guidance scale plays a crucial role in convergence, with 1.0 yielding the most stable training. For inference, we set the guidance scale to 3.0 and employed 28 sampling steps.

Dataset. We adopt a hybrid training strategy that combines perspective and panoramic data. For the perspective branch, we curate 40k high-quality landscape images from the Internet, center-crop them to a 1:1 ratio, and project them onto random panoramic regions. For the panoramic branch, we follow PanFusion (Zhang et al., 2024) and utilize Matterport3D (Chang et al., 2017), a largescale RGB-D dataset comprising 10,800 panoramas across 90 building-scale scenes. To mitigate distortion, we refine the blurred polar regions and use 10k panoramas for training while reserving the remainder for validation, consistent with prior work.

Evaluation Metrics. Following prior work, we evaluate our method with a diverse set of complementary metrics. For realism, we adopt Fr´echet Inception Distance (FID) (Heusel et al., 2018) and its variants, including FIDclip for fair comparison by excluding blurred polar regions, and FIDpole and FIDequ following SMGD (Sun et al., 2025) to assess polar distortion and perspective projection quality. Since FID relies on an Inception network trained on perspective images and may not fully capture panoramic characteristics, we further employ Fr´echet Auto-Encoder Distance (FAED) (Oh et al., 2021), a variant tailored for panoramas. For diversity, we report Inception Score (IS) (Salimans

[Figure 7]

Figure 7: More results on inpainting and outpainting.

et al., 2016), replacing the standard Inception-v3 (Szegedy et al., 2015) with a ResNet pretrained on Places365 (He et al., 2015; Zhou et al., 2017) to better reflect the scene-centric nature of our data. For text–image alignment, we compute CLIP Score (CS) (Radford et al., 2021), and for perceptual quality, we report Q-Align (QA) (Wu et al., 2023a), BRISQUE (Mittal et al., 2012), and NIQE (Mittal et al., 2013), following HunyuanWorld (Team et al., 2025).

- D FULL COMPARISION

In this section, we present the complete qualitative comparison of text-to-panorama generation results. As shown in Fig. 8, our method demonstrates superior perceptual realism, producing sharper and more visually authentic panoramas. In addition, it achieves higher geometric fidelity by effectively handling distortions and preserving boundary continuity, whereas baseline methods often suffer from visible artifacts and structural inconsistencies.

[Figure 8]

Figure 8: The full qualitative comparison on panorama generation. We highlight representative artifacts with red boxes.

- E USER STUDY

To further evaluate human preference, we conducted a user study comparing our method with several representative baselines (Zhang et al., 2024; Lu et al., 2025; Team et al., 2025). The study focused

Table 3: User study results on text-to-panorama generation.

Methods Text Alignment↑ Boundary Continuity↑ Realism↑ Overall Quality↑

PanFusion 21.7% 19.6% 2.1% 0.3% Matrix-3D 24.1% 27.5% 23.7% 5.1% HunyuanWorld 25.9% 18.9% 10.4% 13.7% Ours 28.3% 34.0% 63.8% 80.9%

on four key aspects: text alignment, boundary continuity, realism, and overall quality. A total of 63 participants were asked to choose their preferred outputs from different methods on the test set. As shown in Tab. 3, our method received the highest preference across all metrics, clearly demonstrating its superior ability to generate realistic panoramic images with faithful alignment and coherent boundaries.

- F MORE RESULTS

We present additional results in Figs. 9 and 10 to further illustrate the performance of DiT360 on panoramic image generation. These examples demonstrate that the model consistently produces high-quality, semantically coherent, and visually detailed completions across a variety of scenes.

- G USE OF LARGE LANGUAGE MODELS

Large Language Models were used for minor grammar and style corrections only. All technical content, experiments, and conclusions were authored by the paper’s authors.

- H LIMITATIONS AND FUTURE WORK

Despite the strong performance of DiT360 on panoramic image generation tasks, several limitations remain. The model’s effectiveness is constrained by the diversity and scale of available datasets, leading to suboptimal results in certain scenarios, such as those containing high-resolution human faces or intricate scene details. Future work will focus on collecting larger and more diverse highquality datasets to further enhance the model’s generative capabilities and image resolution. Additionally, leveraging synthetic data to augment training samples can facilitate further advances in panoramic image generation. In the long term, extending the framework to three-dimensional scene generation and understanding represents a promising research direction.

[Figure 9]

##### Figure 9: More results on text-to-panorama generation.

[Figure 10]

##### Figure 10: More results on text-to-panorama generation.

