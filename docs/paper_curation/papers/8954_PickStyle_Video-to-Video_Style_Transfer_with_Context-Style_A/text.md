# arXiv:2510.07546v1[cs.CV]8Oct2025

[Figure 1]

PICKSTYLE: VIDEO-TO-VIDEO STYLE TRANSFER WITH CONTEXT-STYLE ADAPTERS

Soroush Mehraban1,2,3*, Vida Adeli1,2,3* Jacob Rommann1, Babak Taati2,3, Kyryl Truskovskyi1 1Pickford AI 2University of Toronto 3Vector Institute

* Equal contribution

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

FLATTEN

Control-A-

[Figure 7]

[Figure 8]

Video

ProblemDefinition

Artistic style transfer, Style degradation

[Figure 9]

[Figure 10]

## ?

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

FRESCO

Ours

Vector style Video

[Figure 16]

Color/details inconsistency PICKSTYLE

[Figure 17]

[Figure 18]

|[Figure 19]|[Figure 20]|[Figure 21]|
|---|---|---|

|[Figure 22]|[Figure 23]|
|---|---|

|[Figure 24]|
|---|

…AnimeOrigamiPixarLEGO

[Figure 25]

[Figure 26]

|[Figure 27]|[Figure 28]|
|---|---|

|[Figure 29]|
|---|

|[Figure 30]|[Figure 31]|
|---|---|

|[Figure 32]|
|---|

OurResults

|[Figure 33]|[Figure 34]|
|---|---|

|[Figure 35]|
|---|

|[Figure 36]|[Figure 37]|[Figure 38]|
|---|---|---|

|[Figure 39]|
|---|

|[Figure 40]|
|---|

|[Figure 41]|
|---|

|[Figure 42]|
|---|

|[Figure 43]|
|---|

|[Figure 44]|
|---|

Source Video Styled Video

Figure 1: PICKSTYLE addresses video-to-video style transfer by preserving motion and context while translating videos into diverse styles. Unlike prior methods that treat the task as artistic style transfer (color–texture statistics while ignoring geometric properties of the target style) and that often suffer from style degradation, visual inconsistency and temporal flicker, PICKSTYLE produces coherent translations across nine styles.

ABSTRACT

We address the task of video style transfer with diffusion models, where the goal is to preserve the context of an input video while rendering it in a target style specified by a text prompt. A major challenge is the lack of paired video data for supervision. We propose PICKSTYLE, a video-to-video style transfer framework that augments pretrained video diffusion backbones with style adapters and benefits from paired still image data with source–style correspondences for training. PICKSTYLE inserts low-rank adapters into the self-attention layers of conditioning modules, enabling efficient specialization for motion–style transfer while maintaining strong alignment between video content and style. To bridge the gap between static image supervision and dynamic video, we construct synthetic training clips from paired images by applying shared augmentations that simulate camera motion, ensuring temporal priors are preserved. In addition, we in-

Project page: https://pickstyle.pickford.ai/

troduce Context–Style Classifier-Free Guidance (CS–CFG), a novel factorization of classifier-free guidance into independent text (style) and video (context) directions. CS–CFG ensures that context is preserved in generated video while the style is effectively transferred. Experiments across benchmarks show that our approach achieves temporally coherent, style-faithful, and content-preserving video translations, outperforming existing baselines both qualitatively and quantitatively.

- 1 INTRODUCTION

Recent advances in video diffusion models enable the generation of realistic, temporally coherent videos (Wan et al., 2025; Kong et al., 2024; HaCohen et al., 2024). Following these advances, a growing body of research explores ways to add controllability to text-to-video diffusion models, enabling finer-grained guidance over the generated content (He et al., 2025; Burgert et al., 2025; Jiang et al., 2025). While style transfer has advanced significantly for images, improvements in the video domain remain limited. This limitation is largely due to the scarcity of well-curated paired video datasets spanning diverse styles, in contrast to the abundance of such resources for images.

To mitigate data limitations, several methods (Yang et al., 2024; 2023) leverage image priors to apply style transfer on key frames and subsequently integrate them into videos, yet achieving coherent motion and appearance remains a persistent challenge. StyleMaster (Ye et al., 2025) synthesizes training data by leveraging the illusion property of VisualAnagrams (Geng et al., 2024), generating image pairs that share a common style while differing in content. Building on the still-moving paradigm, it subsequently trains a motion adapter on frozen video representations. Nevertheless, two key limitations remain. First, the synthetic pairs primarily capture artistic variations and are insufficient to model more complex styles, such as LEGO. Second, training a motion adapter on frozen videos presupposes a separation between spatial and temporal attention, whereas recent architectures (Wan et al., 2025; HaCohen et al., 2024; Kong et al., 2024) increasingly adopt spatiotemporal attention mechanisms, making such a decoupling more challenging.

To address these limitations, we exploit GPT-4o’s (Achiam et al., 2023) strong style transfer capability to convert a Unity3D-rendered talk show into three distinct styles (anime, clay, and Pixar), thereby constructing a curated image dataset. We then augment this dataset with a subset of OmniConsistency (Song et al., 2025) to further increase stylistic diversity. To convert these image pairs into videos, we apply synthetic camera motions (e.g., zooming, sliding), creating sequences with simple movement and mitigating the risk of overfitting to static, motionless videos. Next, we keep the base model frozen and train a LoRA module on an auxiliary branch that conditions on RGB videos. Motivated by advances in training-free diffusion guidance approaches (Rajabi et al., 2025; Hong, 2024; Ahn et al., 2024), we further strengthen the context condition by extending classifierfree guidance to context–style classifier-free guidance (CS-CFG), which jointly emphasizes the text prompt for style and the video for contextual information during denoising. Our empirical results demonstrate that this approach significantly outperforms existing models, effectively transferring style while maintaining consistency with the conditioning video.

- 2 RELATED WORKS

Video style transfer with image prior. There are several models that leverage image-based diffusion models for video style transfer by extending them with temporal mechanisms. ControlVideo (Zhang et al., 2023b) adapts ControlNet from images to videos by adding full cross-frame self-attention and interleaved-frame smoothing, which allows strong structural fidelity under textand-condition guidance. However, it is heavily reliant on the quality of control signals (such as depth or edges), making it less robust when such guidance is noisy or unavailable. ReRender-AVideo (Yang et al., 2023) generates stylized key frames with hierarchical cross-frame constraints using an image diffusion model, and then propagates them to the full video through patch-based blending. This hybrid design balances efficiency and quality but can introduce blurred details or artifacts when large motion or scene changes occur. FRESCO (Yang et al., 2024) builds on image priors by enforcing spatial and temporal correspondences and introducing a feature blending mechanism that aggregates spatially similar regions and propagates them along optical flow paths. While this reduces flicker and improves motion stability, it remains sensitive to flow errors and

[Figure 45]

|𝜖𝜃|
|---|

…

|𝒞null|
|---|

|𝒯|
|---|

|𝒞<br><br>|
|---|

|𝜋|
|---|

ϵ ⨁

x𝑥t

𝑡

(𝑥𝑡,𝑡,∅, 𝒞) (𝑥𝑡,𝑡,∅,𝒞null)

(𝑥𝑡, 𝑡, 𝒯, 𝒞)

…

𝜖cond 𝜖null−text 𝜖null ∆text ∆context

Eq. 16

[Figure 46]

|𝑥𝑡−1|
|---|

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

𝑥0

Styled Video

- Figure 2: Training and inference pipeline of PICKSTYLE. In training (left), both the style image and the context image are transformed into video tokens and context tokens with synthetic camera motion using motion augmentation; video tokens are noised and denoised conditioned on context tokens by the DiT-based PICKSTYLE model with context-style adapters. In inference (right), a context video and a style description are encoded and iteratively denoised under text, context, and null conditions, where the proposed CS–CFG applies spatiotemporal permutation to the null context to generate the final styled video.

adds computational complexity. Despite their progress, all these image-based approaches still find it challenging to fully preserve the natural motion of the input video without noticeable flicker.

Video style transfer with video diffusion models. Models that build on video diffusion for style transfer include Control-A-Video (Chen et al., 2023), V-Stylist (Yue et al., 2025), and StyleMaster (Ye et al., 2025). Control-A-Video extends an image diffusion backbone with temporal layers and spatio-temporal attention, and incorporates motion-aware initialization and first-frame conditioning while also supporting per-frame controls such as edges, depth, or flow maps; this allows it to preserve structure and motion while applying styles described in the prompt, though its outputs are generally constrained to short clips and moderate resolutions. V-Stylist approaches the problem as a multi-agent pipeline: it parses the input video into shots, interprets an open-ended style request with an LLM, and renders each shot with a style-specific diffusion model and multiple ControlNets, guided by a self-refinement loop that balances style and structure. This design makes it effective for long and complex videos while producing strong style fidelity. StyleMaster, in contrast, integrates both local and global style cues into a video diffusion backbone, employs a motion adapter to enhance temporal consistency, and uses a tiled ControlNet for video-to-video translation; its styles are often more artistic, as they are grounded in a curated training dataset created using VisualAnagrams, which emphasizes distinctive painterly and creative effects.

- 3 PICKSTYLE

Our goal is to adapt text-to-video diffusion models for the task of video style transfer, where the content of an input video is preserved while its appearance is translated into a target style specified by a text prompt. A key challenge is the lack of paired video datasets for style transfer. To address this, we construct training data from pairs of images with different artistic or visual styles, which provide supervision for learning consistent appearance transformations.

- 3.1 PRELIMINARIES

Conditional Diffusion Models In conditional diffusion models, the forward process progressively corrupts a clean sample x0 into a noisy latent xt through

### q(xt | xt−1) = N xt;√αt xt−1,(1 − αt)I , (1)

until xT approximates Gaussian noise. The reverse process seeks to recover x0 by denoising in a stepwise manner, modeled as

pθ(xt−1 | xt,c), (2) where c denotes the conditioning signal (e.g., class label, text, or image). This transition is parameterized by a neural denoiser ϵθ(xt,t,c) that predicts the injected noise at each step. Training minimizes the conditional objective

0,t,ϵ ∥ϵ − ϵθ(xt,t,c)∥2 , (3) ensuring that the learned reverse dynamics generate samples consistent with the condition c.

Ex

Classifier Free Guidance. Classifier-free guidance (CFG) is a widely used sampling technique that enhances the alignment of conditional diffusion models with a given condition c without requiring an external classifier. Instead of relying solely on ϵθ(xt,t,c), the denoiser is jointly trained with and without conditions, yielding an unconditional branch ϵθ(xt,t,∅). During inference, the two predictions are interpolated as

### ϵˆθ(xt,t,c) = ϵθ(xt,t,∅) + ω ϵθ(xt,t,c) − ϵθ(xt,t,∅) , (4)

where ω > 1 is the guidance scale. This formulation strengthens the influence of the condition by amplifying its contribution relative to the unconditional estimate, thereby producing samples that more faithfully follow y while preserving sample diversity.

VACE Building on ACE (Han et al., 2024), VACE (Jiang et al., 2025) introduces multimodal input conditioning for text-to-video generation through the Video Condition Unit (VCU). Formally, VCU is defined as

V = (T ,F,M), (5) where T denotes the text prompt, F = {u1,u2,...,um} ∈ RC×T×H×W is the normalized video conditioning, and M = {m1,m2,...,mn} ∈ {0,1}T×H×W is a binary mask, with 1 indicating tokens that can be modified and 0 indicating tokens that remain fixed. The model then computes reactive frames Fc = F ⊙ M and inactive frames Fk = F ⊙ (1 − M), which are concatenated as C = [Fc;Fk] to form the final video conditioning input.

To inject the condition, VACE uses signals such as optical flow, depth maps, grayscale videos, scribbles, human 2D poses, and bounding boxes as F during training. Following ControlNet (Zhang

- et al., 2023a), it duplicates the pretrained text-to-video blocks into context blocks and trains them as a separate branch. These context blocks are fewer than the main blocks and skip certain layers, which makes the model more lightweight and improves convergence. The output of each context block is then added back to the corresponding DiT block in the main branch. While VACE incorporates diverse conditioning signals during training, RGB frames are always treated as inactive frames. As a result, the model can handle tasks such as inpainting and outpainting, but cannot encode RGB inputs as reactive frames, which limits its ability to perform tasks like style transfer.

- 3.2 TRAINING WITH IMAGE PAIRS

To enable the model to generalize from static image pairs to dynamic video content, we simulate motion during training. Specifically, we apply conventional data augmentations such as zooming in/out and sliding the crop window, which act as synthetic camera motions. For each image pair (source, style), we generate two corresponding video clips of length T frames, where both clips undergo identical augmentation trajectories. This ensures the paired clips exhibit aligned synthetic motion while differing in style, allowing the model to learn temporal consistency during style transfer.

- Fig. 2 shows our training and inference pipeline. We adapt pretrained VACE model built on N DiT blocks from Wan2.1, and adds M context blocks (M < N) to encode the additional condition. We finetune only the self-attention layers of the context blocks. Cross-attention layers, which handle text conditioning, are left untouched because the model already demonstrates strong language understanding. Restricting adaptation to self-attention layers avoids disrupting the pretrained textvideo alignment while still enabling the model to specialize in transferring motion and appearance across video domains. Formally, the standard QKV projections in self-attention layers are defined as:

### Qi = WQZi, Ki = WKZi, Vi = WV Zi, i ∈ {n,c}, (6)

where Zn,Zc are input features for noise and context tokens, and WQ,WK,WV ∈ Rd×d are shared projection matrices used across all branches. We introduce LoRA transformations exclusively on the context blocks:

∆Qc = BQAQZc, ∆Kc = BKAKZc, ∆Vc = BV AV Zc, (7) where AQ,AK,AV ∈ Rr×d and BQ,BK,BV ∈ Rd×r are low-rank matrices with r ≪ d. The QKV for the context blocks is then updated as:

Q′c = Qc + ∆Qc, Kc′ = Kc + ∆Kc, Vc′ = Vc + ∆Vc, (8) while the noise branch remains unchanged:

Q′n = Qn, Kn′ = Kn, Vn′ = Vn. (9)

- 3.3 CONTEXT–STYLE CLASSIFIER-FREE GUIDANCE (CS–CFG)

Let xt denote the noised latent at diffusion step t, and let ϵθ(xt,t; T ,C) be the noise-prediction network conditioned on a text prompt T (style) and a video-conditioning tensor C (context). We construct a “null” version of the context by independently permuting its temporal and spatial axes. Concretely, if C ∈ Rt×h×w×c is the encoded context tensor in latent space, we draw independent uniform permutations πT ∈ ST, πH ∈ SH, πW ∈ SW, where ST (resp. SH, SW) denotes the symmetric group of all permutations of {1,...,T} (resp. {1,...,H}, {1,...,W}). The null context tensor is then defined as

Cnull = πW · πH · πT · C, (10) with (πT · C)t,h,w,c = CπT(t),h,w,c and analogously for πH and πW. We then evaluate three forward passes:

ϵcond = ϵθ(xt,t; T , C), (11) ϵnull text = ϵθ(xt,t; ∅, C), (12)

ϵnull = ϵθ(xt,t; ∅, Cnull), (13) where ∅ denotes dropped text-conditioning (i.e., the classifier-free “null” token). CS–CFG factorizes the guidance into a style (text) direction and a context (video) direction:

∆text = ϵcond − ϵnull text, (14)

∆context = ϵnull text − ϵnull. (15) Given user-selected scales tguide≥0 (style) and cguide≥0 (context), the guided prediction is

ϵ = ϵnull text + tguide ∆text + cguide ∆context. (16)

- 3.4 NOISE INITIALIZATION STRATEGY

To enhance temporal coherence and preserve the context structure of the input video, we depart from the standard diffusion process that initializes sampling from pure Gaussian noise. Instead, we propose to initialize sampling from a partially noised version of the original video content C. Given a total of n denoising steps, we select a hyperparameter k ∈ [1,n], and construct xn−k by applying the forward noising process to C up to step n − k:

### xn−k ∼ q(xn−k | x0 = C). (17)

We then run the reverse process starting from xN−k down to x0 using the DPM++ (Lu et al., 2025) sampler:

xt−1 = DPM++ xt, ϵθ(xt,t;T ,C) , t = n − k,...,1, (18) where ϵθ(xt,t;T ,C) is the denoiser conditioned on the style prompt T and video content C.

By initializing from xn−k rather than pure Gaussian noise, the model retains spatial and motion structure from the original video content C, while still allowing sufficient stochasticity to adapt the style specified by T . The hyperparameter k controls the trade-off between style strength (larger k) and content/motion fidelity (smaller k).

- Table 1: Quantitative comparisons on Content and Style Alignment across baseline methods and our PICKSTYLE

Content Alignment Style Alignment DreamSim ↓ UMT ↑ CLIP ↑ CSD ↑ R Precision ↑ Top@1 Top@2 Top@3

Models

Control-A-Video Chen et al. (2023) 0.52 1.33 0.57 0.10 0.34 0.54 0.65 Rerender Yang et al. (2023) 0.41 2.47 0.55 0.13 0.27 0.39 0.54 FLATTEN Cong et al. (2024) 0.34 2.80 0.56 0.21 0.28 0.43 0.53 FRESCO Yang et al. (2024) 0.45 1.82 0.54 0.17 0.09 0.22 0.32

PICKSTYLE 0.34 3.33 0.57 0.37 0.75 0.85 0.91

- 4 EXPERIMENTS

Implementation details. We use the multi-node training framework of (Modal) with RDMA support to efficiently optimize the LoRA parameters. Our style adapter is trained on 32 H100 GPUs for 3000 steps with a learning rate of 5.6 × 10−4 and rank r = 128 on the Wan2.1-VACE-14B variant. During inference, we apply n = 20 denoising steps with tguide = 5 and cguide = 4 in CS–CFG. To further improve results, we use TeaCache (Liu et al., 2025) to accelerate generation and APG (Sadat

- et al., 2024) to mitigate oversaturation. Additional details are provided in the appendix.

Metrics. We evaluate our method based on Content Alignment, Style Alignment, and Video Quality. For content alignment, we compute frame-level similarity using the DreamSim (Fu et al., 2023) distance between corresponding frames in the original and generated videos, and report the final score by averaging across all frames. We further evaluate how well the generated video matches its high-level text description using UMTScore (Liu et al., 2023). For style alignment, we calculate the CLIP score (Hessel et al., 2021) between each generated frame and a textual style prompt, then average over frames to obtain the final score. We also compute the CSD score (Somepalli et al., 2024) by first averaging the similarity between each generated frame and the target style exemplars, and then averaging across frames to produce the overall style alignment score. We further evaluate top-k R Precision using Gemini (Team et al., 2023) by classifying the middle frame of each generated video against all candidate style prompts. For each frame, Gemini returns the top-k most likely styles in order, and we compute top-k precision for each frame, and averaging across frames to produce the final precision score. For Video quality, we use Motion smoothness, dynamic quality, and visual quality from VBench (Huang et al., 2024) benchmark. Motion smoothness leverages the motion priors in the AMT (Li et al., 2023) model to leverage the smoothness of generated videos. Dynamic quality uses RAFT (Teed & Deng, 2020) to estimate degree of dynamics, and Visual quality uses MUSIQ (Ke et al., 2021) on each frame to assess distortions such as over-exposure, noise, or blur.

Dataset. Our training dataset consists of paired images across multiple styles. We begin by extracting 250 diverse frames from an animated 3D talk show rendered in Unity3D, which serve as our source images. Using GPT-4o, we transform each frame into three distinct styles: Anime, Pixar, and Claymation. To ensure consistency in content between the generated samples and the originals, we manually refine the prompts for each case. This process yields a carefully curated dataset of 750 stylized samples, containing both the original reference frames and their three stylistic variants. To further enhance the diversity of training data, we incorporate six styles from OmniConsistency’s dataset (Song et al., 2025): 3D Chibi, Vector, LEGO, Rick & Morty, Origami, and Macaron, and we further augment our Claymation style using their samples.

- 4.1 COMPARISONS WITH OTHER METHODS

Quantitative comparison. Table 1 compares PICKSTYLE with prior approaches on both content and style alignment metrics. For content alignment, PICKSTYLE achieves the lowest DreamSim score (0.34) and the highest UMTScore (3.33), indicating stronger frame-level consistency and better alignment with high-level content descriptions than the baselines. On style alignment, PICKSTYLE reaches the highest CSD score (0.37). While CLIP score remains tied with Control-A-Video (0.57), PICKSTYLE achieves substantially higher R Precision across all top-k levels, demonstrating more accurate alignment with the target styles.

[Figure 51]

- Figure 3: Comparison on CSD Score and inference cost, per one second of generated video. Inference is evaluated on a single H100 GPU.

- Table 2: Quantitative comparisons on Video Quality metrics across baselines and our PICKSTYLE

Video Quality

Models

Overall MotionSmooth ↑ DynamicQuality ↑ VisualQuality ↑

Control-A-Video (Chen et al., 2023) 0.976 0.602 0.683 0.754 Rerender (Yang et al., 2023) 0.990 0.667 0.567 0.741 FLATTEN (Cong et al., 2024) 0.977 0.780 0.592 0.783 FRESCO (Yang et al., 2024) 0.993 0.632 0.623 0.716

PICKSTYLE 0.982 0.797 0.688 0.822

- Fig. 3 further shows that our method achieves both faster inference and better CSD score for style alignment, whereas Rerender and FRESCO rely on Ebsynth blending (Jamriˇska et al., 2019), which introduces the main bottleneck during inference.

Table 2 demonstrates that PICKSTYLE achieves a clear margin over existing approaches in both dynamic quality and visual quality, the two metrics most reflective of temporal coherence and perceptual fidelity. MotionSmooth remains nearly perfect for all methods, since they are derived from video-to-video models that inherently preserve motion trajectories, and the small numerical differences are therefore negligible. When aggregated, PICKSTYLE obtains the highest overall score, highlighting its effectiveness in generating temporally consistent and perceptually compelling video outputs compared to prior work.

Qualitative comparison. Fig. 4 presents a qualitative comparison of PICKSTYLE with Rerender, Control-a-Video, FLATTEN, and FRESCO on LEGO and Anime styles. The competing methods, which rely on depth maps or HED edges (Xie & Tu, 2015) as inputs, lack access to color information, often producing mismatched hues and noticeable color artifacts in their generated videos. In addition, Rerender and FRESCO, being image-based models, exhibit poor temporal consistency and suffer from frame-to-frame flickering. Finally, while the geometry constraints in these baselines sometimes succeed in forming LEGO-like structures in local regions such as the head, they frequently fail to propagate these stylistic details across the entire body. In contrast, PICKSTYLE consistently delivers faithful color reproduction, stable temporal coherence, and coherent geometry throughout the video. Additional qualitative comparison results across styles are provided in the Appendix and supplemental video.

Fig. 5 shows qualitative results on Unity3D animations that we collected and used to train Anime, Pixar, and Clay styles. Although this dataset differs from the photorealistic data used to train other styles, PICKSTYLE is still able to transfer styles such as LEGO, Rick & Morty, and Macaron from OmniConsistency, which were originally trained on photorealistic counterparts. This demonstrates that PICKSTYLE generalizes effectively across domains, handling both photorealistic and non-photorealistic inputs. Moreover, it highlights a practical application for animated content: instead of depending on high-quality outputs from 3D engines, one can rely on simple Unity3D renderings and leverage style transfer to achieve visually compelling results.

In Fig. 6, we further compare PICKSTYLE with VACE on Macaron style generation. Here, optical flows extracted using RAFT (Teed & Deng, 2020) serve as the input condition for VACE. Because these flows do not contain color information, VACE cannot preserve the lost appearance details in its outputs. In addition, since VACE was not originally designed for style transfer and is highly sensitive

Rerender Control-A-Video FLATTEN FRESCO PICKSTYLE

Source

[Figure 52]

[Figure 53]

|[Figure 54]|
|---|

|[Figure 55]|
|---|

|[Figure 56]|
|---|

|[Figure 57]|
|---|

|[Figure 58]|
|---|

|[Figure 59]|
|---|
|[Figure 60]|

|[Figure 61]|
|---|

|[Figure 62]|
|---|

|[Figure 63]|
|---|

|[Figure 64]|
|---|

|[Figure 65]|
|---|

LEGO

|[Figure 66]|
|---|

|[Figure 67]|
|---|

|[Figure 68]|
|---|
|[Figure 69]|

|[Figure 70]|
|---|

|[Figure 71]|
|---|

|[Figure 72]|
|---|
|[Figure 73]|

|[Figure 74]|
|---|

|[Figure 75]|
|---|

|[Figure 76]|
|---|

|[Figure 77]|
|---|

|[Figure 78]|
|---|

|[Figure 79]|
|---|

|[Figure 80]|
|---|

|[Figure 81]|
|---|

|[Figure 82]|
|---|

|[Figure 83]|
|---|

[Figure 84]

[Figure 85]

|[Figure 86]|
|---|
|[Figure 87]|

|[Figure 88]|
|---|

|[Figure 89]|
|---|

|[Figure 90]|
|---|

|[Figure 91]|
|---|

|[Figure 92]|
|---|

Anime

|[Figure 93]|
|---|

|[Figure 94]|
|---|

|[Figure 95]|
|---|

|[Figure 96]|
|---|

|[Figure 97]|
|---|

|[Figure 98]|
|---|

|[Figure 99]|
|---|

|[Figure 100]|
|---|

|[Figure 101]|
|---|

|[Figure 102]|
|---|

|[Figure 103]|
|---|

- Figure 4: Qualitative comparison of PICKSTYLE , Control-a-Video, Rerender, FRESCO, and FLATTEN in LEGO and anime styles.

PixarMacaronSource

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

[Figure 126]

ClayLEGORickMorty

- Figure 5: Qualitative evaluation of PICKSTYLE on a non-photorealistic example rendered in Unity3D.

to the input geometries, it struggles to capture the intended stylistic patterns and fails to achieve reliable style transfer. More extensive comparisons with alternative input modalities supported by VACE are provided in the Appendix.

- 4.2 ABLATION STUDIES

Effect of motion augmentation. Fig. 7 shows the effect of motion augmentation on videos generated by PICKSTYLE in anime and Pixar styles. For the anime samples both the video description and the style prompt are provided, while for the Pixar samples only the style prompt is given. When the video description is included the generated results achieve both good motion quality and faithful style transfer. Without motion augmentation however small background motions such as people walking on a treadmill are often missed, as the model pays less attention to fine motion details. The gap becomes larger when the video description is not provided. In the Pixar example the model without motion augmentation cannot fully preserve actions such as the jump at the end of the video

8

[Figure 127]

OpticalFlowVACE

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

Source

[Figure 134]

PSTYLEICK

|[Figure 135]|
|---|

|[Figure 136]|
|---|

|[Figure 137]|
|---|

|[Figure 138]|
|---|

|[Figure 139]|
|---|

|[Figure 140]|
|---|

- Figure 6: Comparison between PICKSTYLE and the VACE baseline in 3D Chibi style. VACE fails to capture the target style.

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

Source

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

w/o Motion Aug

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

w/ Motion Aug

Anime Pixar

- Figure 7: Effect of motion augmentation of generated video of PICKSTYLE .

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

Source CFG

CS-CFG (Zero Context) CS-CFG

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

- Figure 8: Effect of CS-CFG on the style transferring, evaluated on Clay style.

and focuses mostly on style transfer. With motion augmentation the model better captures both large scale and subtle motions even when detailed descriptions are not available.

[Figure 191]

[Figure 192]

[Figure 193]

Effect of CS–CFG. Fig. 8 highlights the effectiveness of CS-CFG in improving style transfer. With CFG, only the style guidance in text prompt influences the output, so while the video carries the intended clay style, it lacks fidelity to the original content. In this case, the model confuses the dog with a swan due to its generative prior and produces a hybrid appearance that diminishes contextual accuracy. An alternative design replaces the null video context in CS-CFG with zero pixels, which yields partial improvement over CFG but results in oversaturation and incomplete preservation of the clay style, as seen for instance in the person’s hand where fine details are lost. In contrast, CS-CFG leverages spatiotemporal permutation to better capture contextual cues, leading to sharper details, faithful clay-style transfer, and stronger adherence to the intended content.

- 5 LIMITATION

PICKSTYLE is built on Wan2.1 as the underlying generative backbone and therefore inherits artifacts and weaknesses present in that model. Typical issues include distortions in fine regions such as faces and hands, where the base model struggles to capture small details. As more advanced video backbones become available, the same pipeline can directly benefit from them, reducing such artifacts and further improving overall quality.

- 6 CONCLUSION

We introduced PICKSTYLE , a video-to-video style transfer framework built on VACE with context–style adapters and a novel CS–CFG mechanism. Despite being trained on a relatively limited dataset, PICKSTYLE effectively preserves motion and context while rendering diverse target styles. By leveraging synthetic motion-augmented training pairs and a noise initialization strategy, it achieves superior style fidelity, temporal stability, and perceptual quality compared to existing methods. Beyond quantitative improvements, PICKSTYLE consistently produces coherent color reproduction and faithful geometry across diverse styles while avoiding the temporal flicker and blending artifacts common in image-based approaches. These results highlight that even with constrained supervision, PICKSTYLE can deliver high-quality style transfer and establish a strong baseline for future research in controllable video stylization.

Acknowledgment. We gratefully acknowledge the support of Modal, whose computing credits facilitated the processing required for this work.

REFERENCES

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Donghoon Ahn, Hyoungwon Cho, Jaewon Min, Wooseok Jang, Jungwoo Kim, SeonHwa Kim, Hyun Hee Park, Kyong Hwan Jin, and Seungryong Kim. Self-rectifying diffusion sampling with perturbed-attention guidance. In European Conference on Computer Vision, pp. 1–17. Springer, 2024.

Ryan Burgert, Yuancheng Xu, Wenqi Xian, Oliver Pilarski, Pascal Clausen, Mingming He, Li Ma, Yitong Deng, Lingxiao Li, Mohsen Mousavi, et al. Go-with-the-flow: Motion-controllable video diffusion models using real-time warped noise. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 13–23, 2025.

Weifeng Chen, Jie Wu, Pan Xie, Hefeng Wu, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. Control-a-video: Controllable text-to-video generation with diffusion models. CoRR, 2023.

Yuren Cong, Mengmeng Xu, Christian Simon, Shoufa Chen, Jiawei Ren, Yanping Xie, Juan-Manuel P´erez-R´ua, Bodo Rosenhahn, Tao Xiang, and Sen He. FLATTEN: optical flow-guided attention for consistent text-to-video editing. In ICLR, 2024.

Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. DreamSim: Learning new dimensions of human visual similarity using synthetic data. arXiv preprint arXiv:2306.09344, 2023.

Daniel Geng, Inbum Park, and Andrew Owens. Visual anagrams: Generating multi-view optical illusions with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24154–24163, 2024.

Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. LTX-Video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.

Zhen Han, Zeyinzi Jiang, Yulin Pan, Jingfeng Zhang, Chaojie Mao, Chenwei Xie, Yu Liu, and Jingren Zhou. ACE: All-round creator and editor following instructions via diffusion transformer. arXiv preprint arXiv:2410.00086, 2024.

Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. CameraCtrl: Enabling camera control for video diffusion models. In The Thirteenth International Conference on Learning Representations, 2025.

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. CLIPScore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718, 2021.

Susung Hong. Smoothed energy guidance: Guiding diffusion models with reduced energy curvature of attention. Advances in Neural Information Processing Systems, 37:66743–66772, 2024.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21807–21818, 2024.

Ondˇrej Jamriˇska, S´ˇarka Sochorov´a, Ondˇrej Texler, Michal Luk´aˇc, Jakub Fiˇser, Jingwan Lu, Eli Shechtman, and Daniel S`ykora. Stylizing video by example. ACM Transactions on Graphics (TOG), 38(4):1–11, 2019.

Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. VACE: All-in-one video creation and editing. arXiv preprint arXiv:2503.07598, 2025.

Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. MUSIQ: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 5148–5157, 2021.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. HunyuanVideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Zhen Li, Zuo-Liang Zhu, Ling-Hao Han, Qibin Hou, Chun-Le Guo, and Ming-Ming Cheng. AMT: All-pairs multi-field transforms for efficient frame interpolation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9801–9810, 2023.

Feng Liu, Shiwei Zhang, Xiaofeng Wang, Yujie Wei, Haonan Qiu, Yuzhong Zhao, Yingya Zhang, Qixiang Ye, and Fang Wan. Timestep embedding tells: It’s time to cache for video diffusion model. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 7353– 7363, 2025.

Yuanxin Liu, Lei Li, Shuhuai Ren, Rundong Gao, Shicheng Li, Sishuo Chen, Xu Sun, and Lu Hou. FETV: A benchmark for fine-grained evaluation of open-domain text-to-video generation. Advances in Neural Information Processing Systems, 36:62352–62387, 2023.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-Solver++: Fast solver for guided sampling of diffusion probabilistic models. Machine Intelligence Research, pp. 1–22, 2025.

Modal. Modal: High-performance ai infrastructure. https://modal.com/. Accessed: 2025-09-23. Javad Rajabi, Soroush Mehraban, Seyedmorteza Sadat, and Babak Taati. Token perturbation guid-

ance for diffusion models. arXiv preprint arXiv:2506.10036, 2025.

Seyedmorteza Sadat, Otmar Hilliges, and Romann M Weber. Eliminating oversaturation and artifacts of high guidance scales in diffusion models. In The Thirteenth International Conference on Learning Representations, 2024.

Gowthami Somepalli, Anubhav Gupta, Kamal Gupta, Shramay Palta, Micah Goldblum, Jonas Geiping, Abhinav Shrivastava, and Tom Goldstein. Measuring style similarity in diffusion models. arXiv preprint arXiv:2404.01292, 2024.

Yiren Song, Cheng Liu, and Mike Zheng Shou. OmniConsistency: Learning style-agnostic consistency from paired stylization data. arXiv preprint arXiv:2505.18445, 2025.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Zachary Teed and Jia Deng. RAFT: Recurrent all-pairs field transforms for optical flow. In European conference on computer vision, pp. 402–419. Springer, 2020.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Saining Xie and Zhuowen Tu. Holistically-nested edge detection. In Proceedings of the IEEE international conference on computer vision, pp. 1395–1403, 2015.

Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. Rerender a video: Zero-shot text-guided video-to-video translation. In SIGGRAPH Asia 2023 Conference Papers, pp. 1–11, 2023.

Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. FRESCO: Spatial-temporal correspondence for zero-shot video translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8703–8712, 2024.

Zixuan Ye, Huijuan Huang, Xintao Wang, Pengfei Wan, Di Zhang, and Wenhan Luo. StyleMaster: Stylize your video with artistic generation and translation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 2630–2640, 2025.

Zhengrong Yue, Shaobin Zhuang, Kunchang Li, Yanbo Ding, and Yali Wang. V-Stylist: Video stylization via collaboration and reflection of mllm agents. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 3195–3205, 2025.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 3836–3847, 2023a.

Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. Controlvideo: Training-free controllable text-to-video generation. arXiv preprint arXiv:2305.13077, 2023b.

A APPENDIX

Supplemental Video. The supplemental video provides qualitative demonstrations that illustrate the effectiveness of our approach across various styles and scenarios. We strongly encourage readers to view the supplemental video for a more comprehensive understanding of the results.

- A.1 MORE IMPLEMENTATION DETAILS

Based on noise initilization strategy introduced in Sec. 3.3, we skip the first k denoising steps that controls the trade-off between style strength and motion fidelity. By trial and error, we choose different k values for each style presented in Table 3. For styles such as Vector that are more abstract, we use less k value and for styles such as Pixar that more resembles the input RGB, we use higher value. For R Precision, we employ Gemini-2.5-Flash as the style classifier.

Table 3: Step skip values used for different styles.

Style Step Skip Value

Vector 1 3D Chibi 2 Anime 3 Pixar 6 Clay 0 LEGO 2 Macaron 2 Origami 2 Rick & Morty 0

- A.2 MORE COMPARISON WITH VACE

Alternative conditions that VACE can use for style transfer include depth maps, shown in Fig. 9, and scribbles, shown in Fig. 10. However, because depth maps only provide relative geometry and scribbles capture edges, VACE is unable to perform effective style transfer in either case. Moreover, since these conditions are extracted from videos, they are prone to noise, which further degrades the quality of the generated output.

[Figure 194]

|[Figure 195]|
|---|

|[Figure 196]|
|---|

|[Figure 197]|
|---|

|[Figure 198]|
|---|

|[Figure 199]|
|---|

|[Figure 200]|
|---|

PSTYLEICKSource

DepthVACE

[Figure 201]

|[Figure 202]|
|---|

|[Figure 203]|
|---|

|[Figure 204]|
|---|

|[Figure 205]|
|---|

|[Figure 206]|
|---|

|[Figure 207]|
|---|

- Figure 9: Comparison between PICKSTYLE and the VACE baseline in Anime style when using Depth map as condition of VACE.

PSTYLEICKSource

[Figure 208]

[Figure 212]

ScribbleVACE

|[Figure 213]|
|---|

|[Figure 214]|
|---|

|[Figure 215]|
|---|

|[Figure 216]|
|---|

|[Figure 217]|
|---|

|[Figure 218]|
|---|

|[Figure 219]|
|---|

|[Figure 220]|
|---|

|[Figure 221]|
|---|

|[Figure 222]|
|---|

|[Figure 223]|
|---|

|[Figure 224]|
|---|

- Figure 10: Comparison between PICKSTYLE and the VACE baseline in LEGO style when using scribble as condition of VACE.

13

#### A.3 MORE QUALITATIVE COMPARISON

Additional qualitative comparisons are shown in Fig. 11 and Fig. 12, covering Pixar, 3D Chibi, Origami, Vector, Clay, Macaron, and Rick & Morty styles. Across these diverse cases, competing approaches frequently suffer from color artifacts, style distortion, and unstable temporal consistency. For instance, methods like Rerender and FRESCO often introduce flickering due to their image-based design, while Control-A-Video and FLATTEN struggle to maintain coherent color reproduction and consistent geometry when translating styles across frames. In contrast, PICKSTYLE produces results that remain faithful to the source video while accurately reflecting the intended target style, demonstrating stronger robustness across both simple and complex stylizations.

Rerender Control-A-Video FLATTEN FRESCO PICKSTYLE

Source

[Figure 228]

|[Figure 229]|
|---|

|[Figure 230]|
|---|

[Figure 231]

|[Figure 232]|
|---|

|[Figure 233]|
|---|
|[Figure 234]|
|[Figure 235]|

|[Figure 236]|
|---|
|[Figure 237]|

|[Figure 238]|
|---|

|[Figure 239]|
|---|

|[Figure 240]|
|---|

|[Figure 241]|
|---|

|[Figure 242]|
|---|

Pixar

|[Figure 243]|
|---|

|[Figure 244]|
|---|

|[Figure 245]|
|---|

|[Figure 246]|
|---|

|[Figure 247]|
|---|

|[Figure 248]|
|---|

|[Figure 249]|
|---|

|[Figure 250]|
|---|

|[Figure 251]|
|---|

[Figure 252]

[Figure 253]

|[Figure 254]|
|---|

|[Figure 255]|
|---|

Origami3DChibi

|[Figure 256]|
|---|

|[Figure 257]|
|---|

|[Figure 258]|
|---|

|[Figure 259]|
|---|

|[Figure 260]|
|---|

|[Figure 261]|
|---|

|[Figure 262]|
|---|

|[Figure 263]|
|---|

|[Figure 264]|
|---|

|[Figure 265]|
|---|

|[Figure 266]|
|---|

|[Figure 267]|
|---|

|[Figure 268]|
|---|

|[Figure 269]|
|---|

|[Figure 270]|
|---|

|[Figure 271]|
|---|

|[Figure 272]|
|---|
|[Figure 273]|

|[Figure 274]|
|---|
|[Figure 275]|

[Figure 276]

[Figure 277]

|[Figure 278]|
|---|

|[Figure 279]|
|---|

|[Figure 280]|
|---|

|[Figure 281]|
|---|

|[Figure 282]|
|---|

|[Figure 283]|
|---|

|[Figure 284]|
|---|

|[Figure 285]|
|---|

|[Figure 286]|
|---|

|[Figure 287]|
|---|

|[Figure 288]|
|---|

|[Figure 289]|
|---|

[Figure 290]

[Figure 291]

|[Figure 292]|
|---|

|[Figure 293]|
|---|

|[Figure 294]|
|---|

|[Figure 295]|
|---|

Vector

|[Figure 296]|
|---|
|[Figure 297]|

|[Figure 298]|
|---|

|[Figure 299]|
|---|

|[Figure 300]|
|---|

|[Figure 301]|
|---|

|[Figure 302]|
|---|

|[Figure 303]|
|---|

|[Figure 304]|
|---|

|[Figure 305]|
|---|

|[Figure 306]|
|---|

|[Figure 307]|
|---|

Figure 11: Qualitative comparison in Pixar, 3D Chibi, Origami, and Vector styles.

#### A.4 THE USE OF LARGE LANGUAGE MODELS (LLMS) We use GPT-5 to refine the writing, paraphrase content, and improve readability.

Rerender Control-A-Video FLATTEN FRESCO PICKSTYLE

##### Source

[Figure 308]

[Figure 309]

|[Figure 310]|
|---|

|[Figure 311]|
|---|

|[Figure 312]|
|---|

|[Figure 313]|
|---|

|[Figure 314]|
|---|

|[Figure 315]|
|---|

|[Figure 316]|
|---|

|[Figure 317]|
|---|

|[Figure 318]|
|---|

|[Figure 319]|
|---|

|[Figure 320]|
|---|

|[Figure 321]|
|---|

Clay

|[Figure 322]|
|---|

|[Figure 323]|
|---|

|[Figure 324]|
|---|

|[Figure 325]|
|---|

|[Figure 326]|
|---|

|[Figure 327]|
|---|

[Figure 328]

[Figure 329]

|[Figure 330]|
|---|

|[Figure 331]|
|---|

|[Figure 332]|
|---|

|[Figure 333]|
|---|

|[Figure 334]|
|---|

|[Figure 335]|
|---|

Rick&MortyMacaron

|[Figure 336]|
|---|

|[Figure 337]|
|---|

|[Figure 338]|
|---|

|[Figure 339]|
|---|

|[Figure 340]|
|---|

|[Figure 341]|
|---|

|[Figure 342]|
|---|

|[Figure 343]|
|---|

|[Figure 344]|
|---|

|[Figure 345]|
|---|

|[Figure 346]|
|---|

|[Figure 347]|
|---|

|[Figure 348]|
|---|

|[Figure 349]|
|---|

|[Figure 350]|
|---|

|[Figure 351]|
|---|

[Figure 352]

[Figure 353]

|[Figure 354]|
|---|

|[Figure 355]|
|---|

|[Figure 356]|
|---|

|[Figure 357]|
|---|

|[Figure 358]|
|---|

|[Figure 359]|
|---|

|[Figure 360]|
|---|

|[Figure 361]|
|---|

|[Figure 362]|
|---|

|[Figure 363]|
|---|

|[Figure 364]|
|---|

|[Figure 365]|
|---|

|[Figure 366]|
|---|

|[Figure 367]|
|---|

Figure 12: Qualitative comparison in Clay, Macaron, and Rick & Morty styles.

