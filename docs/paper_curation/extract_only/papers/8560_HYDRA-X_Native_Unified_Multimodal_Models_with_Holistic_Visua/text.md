## arXiv:2606.13289v1[cs.CV]11Jun2026

[Figure 1]

[Figure 2]

2026-06-12

### HYDRA-X: Native Unified Multimodal Models with Holistic Visual Tokenizers

Guozhen Zhang1,†,∗, Xuerui Qiu2,4,†,∗, Yutao Cui3,†, Tianhui Song3, Changlin Li3, Junzhe Li3, Tao Huang3, Xiao Zhang3, Yang Li3, Jianbing Wu3, Miles Yang3, Zhao Zhong3, Liefeng Bo3, Limin Wang1,5,‡ 1Nanjing University 2CASIA 3Tencent Hunyuan 4Zhongguancun Academy 5Shanghai AI Lab zgzaacm@gmail.com lmwang@nju.edu.cn

##### Abstract

Holistic visual tokenizers are fundamental to unified multimodal models (UMMs) as they map diverse visual inputs into a unified representation space. In this paper, we present HYDRA-X, the first UMM that unifies image and video tokenization within a single Vision Transformer (ViT). Our design is driven by two core challenges: efficiently injecting spatiotemporal reconstruction capability into a native ViT, and embedding image- and video-level semantic awareness into the latent space. To address the first, comprehensive ablations reveal two key findings: (1) framelevel causal temporal attention suffices for visual reconstruction, whereas full spatiotemporal attention degrades it; and (2) hierarchical temporal compression substantially outperforms singlestep alternatives. To tackle the second, we propose a lightweight decompressor that upsamples temporally compressed features under joint image-video teacher supervision, thereby enforcing complementary semantic structures within the compact latent space. Building on this holistic tokenizer, we further propose a principled improvement of the editing pipeline: source-target interaction should occur at the latent level inside the tokenizer rather than at the semantic level inside the LLM, substantially improving editing consistency and accelerating convergence. Instantiated at the 7B dense model, HYDRA-X achieves strong performance across image and video understanding and generation tasks, paving the way for future unified-tokenizer UMMs.

##### 1 Introduction

Unified multimodal models (UMMs) (Xie et al., 2025a; Liu et al., 2025b; Deng et al., 2025; Qiu et al., 2026; Zhou et al., 2024) have recently emerged as a powerful paradigm that jointly trains a single autoregressive backbone for both visual understanding and generation. A central design choice is how visual inputs are encoded: existing systems either deploy decoupled visual encoders that pair a ViT encoder with a separate VAE encoder for the two tasks (Deng

- et al., 2025; Ma et al., 2025c; Zhou et al., 2024), or adopt a unified visual tokenizer that maps diverse visual inputs into a single representation space shared by both tasks (Xie et al., 2025a; Liu et al., 2025b; Tong et al., 2026a; Qiu
- et al., 2026; Wu et al., 2025d; Ma et al., 2025a). The latter approach offers distinct architectural advantages: it eliminates the representational mismatch between heterogeneous encoders that the LLM must otherwise reconcile, and opens a pathway for the mutual reinforcement between understanding and generation.

While unified tokenization has been extensively explored for static images, a holistic tokenizer that binds images and videos into a single representation space has received much less attention. Existing video-capable UMMs typically adopt one of two ad-hoc strategies. The first paradigm relies on frame-wise tokenizers that apply an image semantic encoder independently to each frame (Tong et al., 2026a). Without any temporal interaction inside the tokenizer, the resulting representation cannot capture cross-frame dynamics such as motion or short-horizon causality, leaving the downstream LLM with disjoint per-frame features that carry no inherent video structure. The second paradigm employs cascaded designs that stack a 3D causal VAE before a semantic encoder (Xie et al., 2025a; Liu et al., 2025b). Although this packs the temporal axis, the VAE is trained in isolation without any semantic constraint, and may inadvertently discard information critical for understanding.

In this paper, we present HYDRA-X, the first UMM framework built upon HYDRA-XTOK, a unified visual tokenizer that handles both image and video encoding within a single Vision Transformer (ViT). Our overall design follows the image-only UMM framework HYDRA (Qiu et al., 2026), which compresses intermediate-layer ViT features into a compact latent and then reconstructs semantic feature from it. Extending this paradigm to jointly support images and videos, however, raises two core challenges: (a) efficiently injecting spatiotemporal reconstruction capability into a native ViT, and (b) embedding both image- and video-level semantic awareness into the shared latent space.

Our investigation of the first challenge yields two findings that run counter to conventional intuition. (1) Although full spatiotemporal attention is the most natural extension to video, it actively degrades reconstruction by disrupting

∗ Work done during internship at Tencent Hunyuan. † Equal contribution. ‡ Corresponding author.

[Figure 3]

Image Generation

Video Understanding

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

User: Describe the video. Hydra-X: a dark gray, vintage

[Figure 10]

Volvo station wagon driving down

[Figure 11]

a quiet urban street toward the

camera before making a right

turn. The setting is an older

neighborhood characterized by

[Figure 12]

multi-story residential buildings,

including a dark-shingled house

with a green bay window and a front porch. Image Editing

[Figure 13]

Image Understanding “Transferwatercolor-washthe imagestyle.”into a loose, flowing

[Figure 14]

[Figure 15]

User: What is the dome building in the picture?

[Figure 16]

HYDRA-X

Hydra-X: The dome building

in the image is the Rogers

Centre, a multi-purpose

stadium in Toronto, Canada.

“Remove the bear in the center of the

image, blending the background of trees

and foliage to make it seamless.”

Video Generation

” The sky gradually darkens as the sun sets, with the warm glow of

the sunset fading and the stars beginning to twinkle into view.”

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Figure 1: HYDRA-X is a native UMM that unifies image/video understanding, image/video generation, and instruction-guided image editing through one holistic tokenizer HYDRA-XTOK.

the locality and structure encoded during image pretraining. Surprisingly, frame-level causal temporal attention with a minimal temporal receptive field, attending only to the immediately preceding frame, comprehensively outperforms its global counterpart. (2) A single-step patchify substantially underperforms a hierarchical patchify that distributes temporal compression across multiple stages, indicating that the temporal axis benefits from progressive, multi-scale folding. Together, these two design choices enable HYDRA-XTOK to surpass the reconstruction fidelity of dedicated 3D-conv video VAEs such as Wan2.2-VAE (Wan et al., 2025).

To address the second challenge, we extend the established paradigm of semantic distillation (Qiu et al., 2026; Wu et al., 2025d; Ma et al., 2025a) from images to video, and uncover a fundamental asymmetry: while image latents can readily reuse existing semantic teachers, no available video encoder operates at the compressed temporal resolution of our latent, leaving the video stream without a natural source of semantic supervision. We resolve this asymmetry through a remarkably simple addition: a lightweight Decompressor that lifts the compressed latent back to its native temporal length, enabling direct distillation from pretrained image and video teachers (Tschannen et al., 2025; Wang et al., 2022) at full frame rate. Under this dual spatiotemporal supervision, the compact latent simultaneously preserves pixel-level fidelity and rich spatiotemporal semantic structure, substantially advancing both understanding and generation in UMMs.

Building on this holistic tokenizer, HYDRA-X unifies five UMM tasks within a single shared encoder, as shown in

- Figure 1: image/video generation, image/video understanding, and image editing. Yet editing in particular exposes a fundamental flaw in both HYDRA and cascaded designs: by feeding the LLM only post-encoder semantic features, they confine source-target interaction to the semantic level and forfeit the fine-grained structural information that resides at the latent. To resolve this, we propose a principled inversion of the design: HYDRA-XTOK jointly tokenizes source and target with cross-frame interaction, fusing structural details directly into the target before reaching the LLM. This early latent-level interaction substantially improves editing consistency and accelerates convergence.

Instantiated at the 7B scale on top of Qwen2.5-7B-Instruct (Yang et al., 2024a), HYDRA-X achieves strong performance across image and video understanding and generation tasks. More importantly, it elevates the visual tokenizer from a specialized image-processing component to a holistic image-and-video interface, laying a solid foundation for future unified-tokenizer UMM exploration.

##### 2 Related Work

###### 2.1 Visual Tokenizers for Unified Multimodal Models

A growing body of work unifies reconstruction and semantics within a single visual tokenizer. For images, RAE (Zheng et al., 2025; Tong et al., 2026b) freezes a semantic encoder and learns a pixel decoder, while several unified-tokenizer designs (Yue et al., 2025; Yao et al., 2025a; Ma et al., 2025a; Qu et al., 2025; Song et al., 2025; Lin

- et al., 2025b; Tang et al., 2025) co-train reconstruction and understanding within a single ViT. HYDRA (Qiu et al.,

2026) introduces a progressive ViT with a Generation–Semantic Bottleneck for compress-then-restore semantic distillation, which HYDRA-XTOK inherits. Aligning generative latents with semantic features has further been shown to mutually benefit both tasks (Wang et al., 2024a; Yu et al., 2024; Yao et al., 2025b; Ma et al., 2025b; Wang et al., 2025b; 2023). Joint image-and-video tokenization, however, remains largely under-explored: for video, 3D-convolutional VAEs (Yu et al., 2023; Wan et al., 2025) dominate but lack any semantic structure. A recent work, AToken (Lu et al., 2025), unifies images and videos within a single tokenizer for reconstruction and understanding, but emits task-specific output features for the two objectives and therefore does not yield a unified representation. To our knowledge, HYDRA-X is the first UMM framework to unify image and video within a single ViT-based tokenizer, augmenting HYDRA’s philosophy with explicit temporal causality, hierarchical patchify, and a Decompressor for spatiotemporal semantic awareness.

###### 2.2 Native Unified Multimodal Models

UMMs aim to handle visual understanding and generation within a single backbone, and existing systems can be broadly grouped into three families that differ in how tightly the two objectives share parameters and representations. Composite UMMs (Tong et al., 2025; Chen et al., 2025a; Lin et al., 2025a; Pan et al., 2025; Tang et al., 2025) bridge pretrained understanding and generation models via lightweight adapters or projection layers; this preserves the strengths of each specialised model but leaves the synergy between the two tasks shallow, as gradients rarely flow across the modality boundary and the two backbones never see a shared latent. Native UMMs instead train both objectives jointly from the start, and further bifurcate by their choice of visual representation. Quantised-token approaches (Team, 2024; Xie et al., 2024; Wang et al., 2024c; Zhou et al., 2024) cast visual generation as nexttoken prediction over a VQ codebook, which unifies the LLM interface but inherits the reconstruction loss and codebook-collapse pathologies of VQ tokenizers, capping the achievable visual fidelity. Decoupled designs (Ma

- et al., 2025c; Wu et al., 2025b; Chen et al., 2025b; Deng et al., 2025; Liao et al., 2025; Li et al., 2025; Fan et al.,

2025) side-step this ceiling by routing understanding through a semantic encoder and generation through a separately trained VAE; the price is a duplicated visual pathway whose two streams compete for LLM attention and whose representations must be re-aligned downstream. The most recent line we build on are unified-encoder UMMs such as TransNext (Tong et al., 2026a), Show-o2 (Xie et al., 2025a), and TUNA (Liu et al., 2025b), which share a single visual tokenizer across both tasks and recover the architectural cleanliness of composite systems while retaining joint optimisation. We extend this line in two directions: from images to a unified image-and-video tokenizer, and from independent per-input encoding to a tokenizer-stage source–target interaction tailored for editing.

###### 2.3 Image Editing in Unified Multimodal Models

Image editing is the canonical task in which a UMM must condition the target image on a structurally similar source image, and existing pipelines differ mainly in where this conditioning is injected. The first family relies on dedicated condition adapters: ControlNet-style branches (Zhang et al., 2023) attach a parallel encoder that injects spatially aligned source features into the generator, while reference-token streams as used in BAGEL (Deng et al., 2025) prepend the source as an extra context that the LLM attends to. Both families add either parameters or context length, and the source representation is shaped specifically for the generation head rather than shared with the understanding side. Closer to our setting, the unified-encoder UMMs Show-o2 (Xie et al., 2025a) and TUNA (Liu et al., 2025b) reuse a single tokenizer for both the source and the target, but still encode the two images independently; only their post-encoder semantic features are concatenated at the LLM input, so any cross-image alignment must be reconstructed by the LLM from two already-compressed semantic streams, with the fine-grained pre-bottleneck structure inaccessible. We instead place the source and target in the same temporal window of HYDRA-XTOK and process them in a single forward pass, allowing source–target interaction to begin at the latent level inside the tokenizer’s causal Sem-ViT and propagate before reaching the LLM. This reuses the temporal pathway already trained for video, removes any extra cross-image attention module, and exposes the LLM to a target representation that has already absorbed source structure.

##### 3 Preliminaries: Representation-Harmonized Tokenization

Our overall design follows the image-only UMM framework HYDRA (Qiu et al., 2026). At its core is a single ViT split into a Gen-ViT and a Sem-ViT, connected by a Generation–Semantic Bottleneck that supports generation and semantic perception within one backbone. Given an input image x∈RH×W×3, the Gen-ViT first produces a feature h rich in structural primitives, which the Bottleneck projects into a compact latent z∈RN×C suitable for generation. The Sem-ViT then un-projects z back into a high-dimensional semantic feature s, which is aligned with a pretrained semantic teacher T via distillation:

###### x −−−−−→Gen-ViT h −−−−−−→Bottleneck z −−−−−→Sem-ViT s ←−−−align T (x). (1)

The downstream LLM operates exclusively on the Sem-ViT output s for both understanding and generation, whereas the pixel decoder that reconstructs images from z is invoked only during tokenizer training. We retain this

Latents

###### Gen-ViT

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Transformer Layers

Temporal

Bottle-

Transformer

Spatiotemporal Patchify

[Figure 27]

Patchify

neck

Layers

𝑇 4

𝐻 16

𝑊 16

𝑇 2

𝐻 16

𝑊 16

𝑇

𝐻

𝑊

64 × 1 +

×

×

C × 1 +

×

×

C × 1 +

×

×

3 × 𝑇 × 𝐻 × 𝑊

4

16

16

Full Attention Tubelet Causal Attention

Causal Attention

[Figure 28]

[Figure 29]

[Figure 30]

QueryFrames

QueryFrames

QueryFrames

Key Frames Key Frames

Key Frames

- Figure 2: Spatiotemporal reconstruction design. (Top) The Gen-ViT folds a clip into a compact latent. (Bottom) Three ablated attention masks: Full attends across all space-time tokens, Causal masks future frames, and Tubelet restricts attention to a 2-frame window.

Table 1: Reconstruction ablation on ImageNet (256×256) and DAVIS (17×256×256). All three attention-mask baselines use the single-step 4× temporal patchify; only the ‘Ours’ row uses the hierarchical 2×2 schedule. Latency is measured per forward pass on a 17×512×512 video clip. Further reconstruction comparisons and visualizations are provided in Appendix H.1 and H.3

ImageNet DAVIS

Method Latency (s)(↓)

PSNR(↑) SSIM(↑) rFID(↓) PSNR(↑) SSIM(↑) rFVD(↓)

Full attention 0.49 31.10 0.8890 0.367 27.40 0.8277 16.20 Causal attention 0.45 31.38 0.8901 0.352 27.62 0.8283 14.05 Tubelet attention 0.17 31.42 0.8907 0.347 27.69 0.8287 13.69

Ours 0.25 31.73 0.8936 0.329 27.97 0.8307 11.19

overall design and extend it from images to videos through explicit temporal causality, hierarchical patchify, and a Decompressor introduced in Section 4.

##### 4 HYDRA-XTOK: Holistic Visual Tokenization in a Single ViT

HYDRA-XTOK is designed as the visual interface of HYDRA-X: before any token reaches the LLM, it must be compact enough for generation, faithful enough for reconstruction, and semantic enough for understanding. We initialize Gen-ViT and Sem-ViT from SigLIP2 (Tschannen et al., 2025); all UMM-side ablations use Qwen2.51.5B (Yang et al., 2024a). The tokenizer is trained with a reconstruction term and two semantic distillation terms:

LHYDRA-XTOK = Lrec + λLdist, (2)

where Lrec keeps the compact latent pixel-faithful, Ldist aligns Sem-ViT features with semantic features. Detailed recipes are in Appendix A.1.

###### 4.1 Spatiotemporal Reconstruction in a ViT

Existing ViT-based tokenizers that jointly handle images and videos reconstruction, such as AToken (Lu et al., 2025) and OmniTokenizer (Wang et al., 2024b), share two design choices: full spatiotemporal attention across all frames, and a single-step temporal patchify applied at the input that aggressively compresses the temporal axis. Both choices come at a cost. Full spatiotemporal attention scales quadratically with the clip length and tends to disrupt the per-frame structural prior inherited from image pretraining; the aggressive single-step patchify collapses fine-grained temporal details before any cross-frame reasoning. This naturally raises a critical question: are these design choices really necessary?

We answer this through a controlled ablation along the same two axes: (i) the temporal attention region, and (ii) the temporal patchify schedule. Following the common design in video VAEs, a clip x∈R3×(1+T)×H×W is encoded into an anchor image latent together with the remaining T frames compressed by a factor of 4, producing a compact

T 4 )×

H 16×

W

latent z ∈ RC×(1+

16 . The two axes are then ablated independently. For (i) we compare three attention masks (Fig. 2, bottom): Full attention, the standard choice in AToken and OmniTokenizer; Causal attention, with a causal mask across all preceding frames; and Tubelet attention, where causal attention is restricted to a 2-frame tubelet so each token attends only to its own frame and the immediately preceding one. For (ii) we compare the single-step 4× temporal patchify used by AToken and OmniTokenizer against a hierarchical schedule that applies

inputs latents semantics

[Figure 31]

# }

Decompressor Video

[Figure 32]

…

[Figure 33]

###### Teacher

[Figure 34]

Gen-ViT

[Figure 35]

Sem-ViT

Image

Teacher Spatiotemporal Distillation

- Figure 3: Spatiotemporal distillation. The uncompressed image latent is directly distilled by an image teacher; the 4× temporally-compressed video latent is first lifted to origin length T by a lightweight Decompressor before

distillation by a video teacher.

###### Table 2: Semantic-distillation ablation.

Design choices Vid. Und. Img. Und. Img. Gen. Edit

img Decomp Decomp Sem-ViT MVBench VideoMME AI2D MME GenEval ImgEdit distill w/ img w/ video bi-dir (↑) (↑) (↑) (↑) (↑) (↑)

29.8 27.4 45.1 989 67.5 2.35 ✓ 42.1 42.5 61.2 1339 70.6 2.72

- ✓ ✓ 44.7 44.3 62.7 1522 70.7 3.07

- ✓ ✓ 45.4 45.0 62.5 1501 72.0 3.20

✓ ✓ ✓ 43.1 43.7 62.0 1434 70.1 2.70

two consecutive 2× patchify stages (top of Fig. 2). During each temporal patchify stage, the anchor frame is zero-padded so that it goes through the same operation as the remaining frames.

- Table 1 reveals two principles that contradict the common choices. First, expanding the temporal receptive field beyond a 2-frame tubelet only degrades reconstruction: both full bidirectional and all-past causal attention perform worse than Tubelet attention. Second, distributing temporal compression across two patchify stages consistently outperforms a single-step counterpart at the same compression ratio. These results answer our opening question: aggressive spatiotemporal attention and single-step patchify are not only unnecessary but actively suboptimal.

- • Less attention is more. Restricting causal attention to a 2-frame tubelet yields the best reconstruction; wider receptive fields disturb local detail more than they help.
- • Hierarchical patchify outperforms single-step. Distributing temporal compression across two 2× patchify stages consistently improves reconstruction over a single 4× patchify, indicating that the temporal axis benefits from progressive, multi-scale folding.

4.2 Spatiotemporal Semantic Distillation via the Decompressor

Following HYDRA, we inject semantic structure into the latent by distilling the Sem-ViT output against pretrained teachers. Extending this recipe to video, however, reveals a fundamental asymmetry. For images, the Sem-ViT output has the same spatial resolution as a frame and can be aligned token-by-token with an off-the-shelf image teacher. For video, the Sem-ViT output is temporally compressed to 1 + T/4 tokens, while existing video encoders operate at the original frame rate. The video stream therefore receives no video-level semantic supervision under the standard distillation recipe.

We resolve this asymmetry by introducing a lightweight Decompressor, a small ViT module D that lifts the temporally compressed Sem-ViT output back to its native temporal length, producing dense per-frame semantic features that can be aligned with both image and video teachers (Fig. 3). The Decompressor is only used at tokenizer-training time and is discarded afterwards; the LLM still operates on the same compact Sem-ViT output s. Letting dcos(a,b)=1 − cos(a,b) denote the cosine distance, the full distillation loss combines an image-teacher term at s and a video-teacher term at the Decompressor output:

Ldist = dcos s0, Timg(x) + dcos D(s1:), Tvid(x) , (3)

where s0 is the leading uncompressed image token and s1: are the compressed video latents. For pure image batches, the video term in Eq. 3 is masked out. We ablate four design choices in Table 2: (i) whether to apply image distillation at the Sem-ViT output (img distill); (ii) whether to additionally distill the Decompressor output against an image teacher (Decomp w/ img); (iii) or against a video teacher (Decomp w/ video); and, as a cross-check of F1, (iv) whether the Sem-ViT uses bidirectional rather than tubelet attention (Sem-ViT bi-dir).

- Table 2 surfaces three principles. First, semantic distillation is indispensable: removing it collapses both image and video understanding. Second, the Decompressor is what unlocks video-level supervision: distilling it against a video teacher yields the strongest video understanding while preserving image-side performance, and the same configuration also delivers the best image generation and editing scores, consistent with the hypothesis that semantically richer latents accelerate the LLM’s convergence on generation and editing. Third, switching the

Image Token

Video Token

Text Token

Source image target image Source image target image

w/wo noise

w/wo noise

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

“What breed of dog is in

image video image editing

the picture?” “Generate a video of a girl swinging on a swing.” “Colorize the black and white line art.”

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Gen-ViT Gen-ViT

[Figure 46]

Gen-ViT Gen-ViT

[Figure 47]

Sem-ViT

Sem-ViT Sem-ViT

Gen-ViT

Text Tokenizer

###### UMM

UMM

Noise Scheduler

Structural interaction Structural interaction

[Figure 48]

[Figure 49]

Sem-ViT

[Figure 50]

[Figure 51]

Semantic interaction Semantic interaction

Previous Methods Ours

(a) HYDRA-XTok (b) Tokenizer-Stage Source-Target Interaction

Image understanding Image generation

Video understanding Video generation Image editing

Hydra-X

| | | |
|---|---|---|
| | | |

Language Head ℒNTP Visual Head ℒFM

(c) HYDRA-X

- Figure 4: HYDRA-X unifies five visual tasks through the holistic tokenizer HYDRA-XTOK. (a) HYDRA-XTOK encodes any image or video into a compact Gen-ViT latent and then into semantic features with Sem-ViT. (b) Previous editing pipelines (left) encode source and target with two independent branches; HYDRA-X (right) keeps Gen-ViT independent for faithful reconstruction but shares the Sem-ViT with tubelet causal attention, injecting structural interaction inside the tokenizer. (c) A shared backbone with two separate heads drives all five tasks.

Sem-ViT to bidirectional attention uniformly degrades every metric, mirroring F1: less attention is more even on the understanding side.

• A semantic latent lifts both understanding and generation. Dual image- and video-teacher distillation, enabled by the Decompressor, equips the compact latent with explicit spatiotemporal semantic structure, jointly improving understanding and generation.

- 5 HYDRA-X: Advancing Unified Multimodal Models with Holistic Tokenizers

###### 5.1 Overall Architecture

HYDRA-X follows the standard native UMM template (Xie et al., 2025a; Liu et al., 2025b; Qiu et al., 2026): text tokens and visual tokens produced by HYDRA-XTOK are interleaved into a single sequence and processed by a shared LLM backbone with two specialised heads, an autoregressive language head trained with next-token prediction and a vision head trained with flow matching (Lipman et al., 2022; Esser et al., 2024). Within this template, HYDRA-X unifies five tasks under one shared tokenizer HYDRA-XTOK (Fig. 4(a)): image generation (text → image), image understanding (image → text), video generation (text → video), video understanding (video → text), and image editing (source image with text instruction → target image).

As illustrated in Fig. 4(c), the same Gen-ViT serves all five tasks; the only task-dependent component is which head decode the LLM output. The model is trained end-to-end with the composite loss

LHYDRA-X = λ1LNTP + λ2LFM, (4)

where LNTP is the next-token prediction loss for text, LFM is the rectified flow matching loss for visual latents, and both loss weights λ1 and λ2 are set to 1 by default.

- Table 3: Source-target interaction (STI) ablation. HYDRA-X-STI tokenises the editing pair as a length-2 clip with Sem-ViT tubelet causal attention; HYDRA-X-Indep encodes the source and target independently. Recon-PSNR: PSNR of source reconstruction on ImgEdit.

Editing Img. Gen. Vid. Und. Img. Und.

Layout

ImgEdit(↑) Recon-PSNR(↑) GenEval(↑) MVBench(↑) VideoMME(↑) AI2D(↑) MME(↑)

HYDRA-X-Indep 2.80 20.74 70.51 45.3 45.0 62.2 1478.5 HYDRA-X-STI 3.20 27.65 71.97 45.4 45.0 62.5 1501.0

###### 5.2 Independent Encoding Bypasses the Latent

Among the five tasks, image editing is the only one whose input contains both a conditioning image and a target image. Conventional pipelines, including HYDRA (Qiu et al., 2026) and cascaded designs such as Show-o2 (Xie et al., 2025a) and TUNA (Liu et al., 2025b), tokenise the source xc and the target xt independently with the same tokenizer (Fig. 4(b), left):

[sc, st] = HYDRA-XTOK(xc), HYDRA-XTOK(xt) , sc ⊥ st inside the tokenizer. (5)

As a result, the source and target latents zc,zt never interact inside the tokenizer, and the LLM has to discover their cross-image alignment from scratch on top of two independent semantic streams. This is sufficient for high-level semantic edits but consistently fails on detail-faithful edits.

###### 5.3 Tokenizer-Stage Source-Target Interaction

A natural fix falls out of HYDRA-XTOK’s holistic design: since the Sem-ViT already applies tubelet causal attention for video modeling, we reuse the exact same mechanism for editing pairs by routing (xc,xt) through HYDRA-XTOK

- as a length-2 clip (Fig. 4(b), right). The Gen-ViT continues to encode the two images independently and the post-Bottleneck latents zc,zt remain reconstruction-faithful. The cross-image interaction is then injected exclusively
- at the Sem-ViT, which processes [zc;zt] with the same tubelet causal mask used for video: [sc, st] = Sem-ViT [zc;zt] , causal: sc attends only to zc, st attends to [zc;zt]. (6)

Note that for editing pairs we disable Gen-ViT’s cross-frame tubelet attention since the source and target are not temporally adjacent video frames; only Sem-ViT (the semantic stage) reuses the video tubelet causal mask. This asymmetric reuse is a deliberate choice: structural reconstruction benefits from independent encoding, while semantic alignment benefits from cross-image interaction.

- Table 3 compares HYDRA-X-Indep against HYDRA-X-STI, identical except for whether the editing pair is encoded independently or as a length-2 clip with Sem-ViT tubelet causal attention. STI raises Recon-PSNR, the PSNR of source reconstruction on ImgEdit (Ye et al., 2025) that directly probes editing consistency, by nearly 7 dB and lifts ImgEdit by 0.4. STI further yields consistent gains on most non-editing benchmark, with GenEval (+1.46) the most prominent, suggesting that the new latent-level coupling also enriches the Sem-ViT for generation. The Recon-PSNR jump directly validates our hypothesis from Section 5.2: editing’s consistency failure stems from latent-level isolation inside the tokenizer, not from LLM capacity or supervision.

• Latent-level interaction matters for editing. Reusing the Sem-ViT’s tubelet causal attention to fuse source and target inside the tokenizer, adds no parameters and no separate cross-image module yet substantially improves editing consistency.

##### 6 Main Results

Implementation. HYDRA-X is instantiated at two scales. The reported model uses Qwen2.5-7B-Instruct (Yang et al., 2024a) as the LLM backbone; a matched 1.5B variant is used for the methodological ablations in Sections 4–5. Following AToken (Lu et al., 2025), HYDRA-XTOK includes a symmetric ViT encoder/decoder pair augmented with

- 3D rotary position embeddings (3D RoPE) (Su et al., 2024) for joint spatiotemporal modelling. The Decompressor D in Eq. 3 is a lightweight 4× temporal upsampler that stacks two consecutive (temporal upsample → transformer block) stages; each temporal upsample is a 1×1 convolution doubling the channel dimension (C→2C) followed by a channel-to-time reshape, inverting the encoder’s hierarchical 2×2 temporal patchify. The bottleneck dimension is C=64. For distillation teachers, we use SigLIP-SO400M-patch16-naflex (Tschannen et al., 2025) as the image teacher Timg and InternVideo-Next-L (Wang et al., 2025a) as the video teacher Tvid.

###### Table 4: Evaluation on image understanding benchmarks. # Params. denotes the model size. Rows in gray indicate models with ≥ 14B parameters and are excluded from the ranking. Within each subgroup of the table, bold marks the best result and underline marks the second-best.

AI2D MME MMMU OCRBench MMB RWQA ChartQA DocVQA InfoVQA

Models # Params

test summary val test dev en test test val val Understanding-only Models

Qwen2.5-VL (Bai et al., 2025) 7B 84.3 2312.0 58.0 88.8 82.8 68.4 84.1 93.0 78.6 LLaVA-1.5 (Liu et al., 2023b) 7B 55.5 1510.7 35.7 31.8 62.3 54.8 17.9 – – LLaVA-OV (Li et al., 2024a) 7B 81.4 1998.1 48.8 62.2 80.8 66.3 80.0 87.5 68.8

Unified Multimodal Models

BLIP3-o (Chen et al., 2025a) 8B – 2329.7 50.6 83.1 83.5 69.0 78.0 – – TokenFlow-XL (Qu et al., 2025) 14B – 1922.1 43.2 – 68.9 56.6 – – – Ming-UniVision (Huang et al., 2025) 16B 82.8 2023.0 40.3 72.4 78.5 – – – – BAGEL (Deng et al., 2025) 14B 89.2 2388.0 55.3 73.3 85.0 72.8 78.5 – – Janus-Pro (Chen et al., 2025b) 7B 71.3 1567.1 41.0 59.0 79.2 58.0 25.8 – – VILA-U (Wu et al., 2024b) 7B – 1401.8 – – 66.6 – – – – Show-o2 (Xie et al., 2025a) 7B 78.6 1620.5 48.9 32.4 79.3 64.7 52.3 77.3 45.8 HYDRA (Qiu et al., 2026) 7B 85.1 2068.6 49.4 57.7 82.4 64.7 – – – HYDRA-X 7B 86.5 2350.0 51.5 84.5 84.0 68.7 86.5 81.7 59.1

###### Table 5: Evaluation on video understanding benchmarks. # Params. denotes the model size. Video-MME reports the w/o-subtitle score.

MVBench Video-MME LongVideoBench LVBench test w/o sub val test Proprietary Und.-only Models

Models # Params

GPT-4V (OpenAI, 2023) – 43.5 59.9 61.3 – GPT-4o (OpenAI, 2024) – – 71.9 66.7 48.9 Gemini-1.5-Flash (Team et al., 2024) – – 70.3 61.6 – Gemini-1.5-Pro (Team et al., 2024) – 54.2 75.0 64.0 33.1

Open-source Und.-only Models

VILA (Lin et al., 2024) 40B – 60.1 – – PLLaVA (Xu et al., 2024) 34B 58.1 – 53.2 – LongVA (Zhang et al., 2024b) 7B 49.2 52.6 51.8 – VideoLLaMA2 (Cheng et al., 2024) 7B 54.6 47.9 – – LLaVA-OV (Li et al., 2024a) 7B 56.7 58.2 56.5 26.9 IXC-2.5 (Zhang et al., 2024a) 7B 69.1 55.8 – –

Unified Multimodal Models

Show-o2 (Xie et al., 2025a) 1.5B 49.8 48.0 49.2 – TUNA (Liu et al., 2025b) 1.5B 54.4 49.1 49.7 27.4 Show-o2 (Xie et al., 2025a) 7B 55.8 57.4 55.5 – HYDRA-X 7B 59.1 60.0 59.5 30.0

- 6.1 Multimodal Understanding Image understanding. We benchmark on AI2D (Kembhavi et al., 2016), MME (Fu et al., 2023), MMMU (Yue

- et al., 2024), OCRBench (Liu et al., 2024b), MMBench (Liu et al., 2024a), RealWorldQA, ChartQA (Masry et al., 2022), DocVQA (Mathew et al., 2021), and InfoVQA (Mathew et al., 2022). Table 4 compares HYDRA-X against open-source UMMs at a similar scale. Overall, HYDRA-X matches or exceeds 7B native UMM baselines on most reported metrics, including OCR- and chart-heavy tasks where strong semantic retention is important. Video understanding. We evaluate on MVBench (Li et al., 2024c), Video-MME (Fu et al., 2025), LVBench (Wang
- et al., 2025c), and LongVideoBench (Wu et al., 2024a)(Table 5). HYDRA-X improves over the reported 1.5B and

7B unified baselines on the benchmarks where comparable numbers are available. It remains below the strongest dedicated or proprietary video LMMs on several metrics, but narrows the gap while using a single ViT tokenizer shared across understanding, generation, and editing. These results are consistent with the role of dual-teacher distillation in HYDRA-XTOK, which provides the compressed latent with both image- and video-level semantics.

- 6.2 Visual Generation

###### Table 6 jointly reports image generation on GenEval (Ghosh et al., 2023) and WISE (Niu et al., 2025), and video generation on VBench (Huang et al., 2024) for 17-frame outputs at 640 × 384, summarised by Quality Score (QS), Semantic Score (SS), and the aggregate Total score. Among 7B-scale unified baselines, HYDRA-X is the strongest entry on every reported GenEval and WISE column; compared against ≥ 14B unified models, it remains competitive on the Overall scores while using a 7B backbone. On VBench, HYDRA-X leads all unified entries on QS, SS, and Total, improving over the closest unified competitor (Show-o2-1.5B) by +1.87 QS, +3.26 SS, and

- Table 6: Comprehensive visual generation results. Image generation on GenEval (Ghosh et al., 2023) and WISE (Niu et al., 2025); video generation on VBench (Huang et al., 2024) reporting Quality Score (QS), Semantic Score (SS), and the aggregate Total score. † refers to using LLM rewriters. Rows in gray indicate models with ≥ 14B parameters and are excluded from the ranking. Qualitative results are in Appendix H.4.

Models # Params

GenEval WISE VBench Two Obj. Pos. Over. Cult. Space Over. QS SS Total Generation-only Models

SD3-Med (Esser et al., 2024) 2B 0.94 0.33 0.74 – – – – – – FLUX.1 [Dev] † (Labs et al., 2025) 12B 0.93 0.68 0.82 0.48 0.62 0.50 – – – CogVideoX (Yang et al., 2024b) 5B – – – – – – 82.75 77.04 81.61 Hunyuan Video (Kong et al., 2024) 13B – – – – – – 85.07 76.88 83.43

Unified Multimodal Models

TokenFlow-XL (Qu et al., 2025) 14B 0.60 0.16 0.55 – – – – – – SEED-X (Ge et al., 2024) 17B 0.58 0.19 0.49 – – – – – – Ming-UniVision (Huang et al., 2025) 16B 0.93 0.92 0.85 – – – – – – BAGEL † (Deng et al., 2025) 14B 0.95 0.78 0.88 0.44 0.68 0.52 – – – Show-o2 (Xie et al., 2025a) 1.5B 0.86 0.46 0.73 0.33 0.53 0.39 82.10 78.31 81.34 Harmon (Wu et al., 2025d) 1.5B 0.86 0.74 0.76 0.38 0.52 0.41 – – – Blip3-o† (Chen et al., 2025a) 8B – – 0.84 – – 0.50 – – – MUSE-VL (Xie et al., 2025b) 7B 0.64 0.25 0.57 – – – – – – Janus-Pro (Chen et al., 2025b) 7B 0.89 0.79 0.80 0.30 0.49 0.35 – – – VILA-U (Wu et al., 2024b) 7B – – – 0.26 0.37 0.31 76.26 65.04 74.01 HaploOmni (Xiao et al., 2025b) 7B – – – – – – – – 78.10 Emu3 (Wang et al., 2024c) 8B – – 0.66 – – – – – 80.96 Show-o2 † (Xie et al., 2025a) 7B 0.87 0.52 0.76 0.40 0.58 0.44 – – – HYDRA-X 7B 0.95 0.84 0.88 0.53 0.72 0.56 83.97 81.57 83.49

- Table 7: Image editing. ImgEdit-Bench: Ext.=Extract, Rm.=Remove, Over.=overall (mean of 9 categories). GEdit-Bench: G-SC=G-Semantic Consistency, G-PQ=G-Perceptual Quality, G-Over.=overall. Per-dimension breakdown is provided in Appendix 12.

ImgEdit-Bench GEdit-Bench

Models

Ext. Rm. Over. G-SC G-PQ G-Over. Generation-only Models

FLUX.1 Kontext [Pro] (Labs et al., 2025) 2.35 3.57 4.00 7.02 7.60 6.56 Qwen-Image (Wu et al., 2025a) 3.43 4.14 4.27 8.00 7.86 7.56

Unified Multimodal Models

OmniGen (Xiao et al., 2025a) 1.71 2.43 2.96 5.96 5.89 5.06 UniWorld-V1 (Lin et al., 2025a) 2.27 3.24 3.26 4.93 7.43 4.85 BAGEL (Deng et al., 2025) 1.70 2.62 3.20 7.36 6.83 6.52 OmniGen2 (Wu et al., 2025c) 1.77 3.20 3.44 7.16 6.77 6.41 HYDRA-X 4.04 4.38 4.34 7.80 7.24 7.17

+2.15 Total. Per-dimension VBench scores are provided in Appendix Table 13, where HYDRA-X additionally leads in semantic-heavy dimensions including Object Class, Human Action, and Scene. Together these results suggest that dual-teacher distillation transfers semantic structure into the latent while preserving its role in visual synthesis.

###### 6.3 Image Editing

- Table 7 reports editing on GEdit-Bench (Liu et al., 2025a) and ImgEdit-Bench (Ye et al., 2025). Among 7B-scale unified models, HYDRA-X leads on Ext.(4.04, +1.77), Rm.(4.38, +1.14), ImgEdit Over.(4.34, +0.90), and GEdit G-SC/G-Over.(7.80/7.17), also beating BAGEL-14B on every column. The largest gains land on Ext. and Rm.—both needing identity-faithful source preservation—validating the tokenizer-stage source–target interaction in Section 5.3. With a 7B backbone, HYDRA-X trails Qwen-Image-20B by only 0.20/0.39 on G-SC/G-Over.; per-dimension scores in Appendix Table 12.

##### 7 Conclusion

We presented HYDRA-X, the first native UMM framework that unifies image and video tokenization within a single ViT. Three counter-intuitive design choices in HYDRA-XTOK, frame-level causal tubelet attention, hierarchical temporal patchify, and a Decompressor for dual image-video teacher supervision, efficiently transform an image tokenizer into a video-and-image tokenizer. Rather than treating image editing as a purely LLM-side problem,

we elegantly repurpose our video temporal-causal mechanism to process source and target images as length-2 clips. This restores the fine-grained latent-level coupling that is fundamentally lost in prior independent-encoding pipelines. Through this unified design, the visual tokenizer transcends its traditional role as a static image encoder, emerging as a holistic image-and-video interface that unifies five tasks under one shared backbone.

##### References

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, Wesam Manassra, Prafulla Dhariwal, Casey Chu, Yunxin Jiao, and Aditya Ramesh. Improving image generation with better captions. OpenAI technical report, 2023. URL https://cdn.openai.com/ papers/dall-e-3.pdf.

Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025a.

Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025b.

Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024.

Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, Jiasen Lu, Taira Anderson, Erin Bransom, Kiana Ehsani, Huong Ngo, YenSung Chen, Ajay Patel, Mark Yatskar, Chris Callison-Burch, Andrew Head, Rose Hendrix, Favyen Bastani, Eli VanderBilt, Nathan Lambert, Yvonne Chou, Arnavi Chheda, Jenna Sparks, Sam Skjonsberg, Michael Schmitz, Aaron Sarnat, Byron Bischoff, Pete Walsh, Chris Newell, Piper Wolters, Tanmay Gupta, Kuo-Hao Zeng, Jon Borchardt, Dirk Groeneveld, Crystal Nam, Sophie Lebrecht, Caitlin Wittlif, Carissa Schoenick, Oscar Michel, Ranjay Krishna, Luca Weihs, Noah A. Smith, Hannaneh Hajishirzi, Ross Girshick, Ali Farhadi, and Aniruddha Kembhavi. Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 91–104, June 2025.

Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

Lijie Fan, Luming Tang, Siyang Qin, Tianhong Li, Xuan Yang, Siyuan Qiao, Andreas Steiner, Chen Sun, Yuanzhen Li, Tao Zhu, et al. Unified autoregressive visual generation and understanding with continuous tokens. arXiv preprint arXiv:2503.13436, 2025.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023.

Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 24108– 24118, 2025.

Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396, 2024.

Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21807–21818, 2024.

Ziyuan Huang, DanDan Zheng, Cheng Zou, Rui Liu, Xiaolong Wang, Kaixiang Ji, Weilong Chai, Jianxin Sun, Libin Wang, Yongjie Lv, et al. Ming-univision: Joint image understanding and generation with a unified continuous tokenizer. arXiv preprint arXiv:2510.06590, 2025.

Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In European conference on computer vision, pp. 235–251. Springer, 2016.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024a.

Dongxu Li, Yudong Liu, Haoning Wu, Yue Wang, Zhiqi Shen, Bowen Qu, Xinyao Niu, Guoyin Wang, Bei Chen, and Junnan Li. Aria: An open multimodal native mixture-of-experts model. arXiv preprint arXiv:2410.05993, 2024b.

- Han Li, Xinyu Peng, Yaoming Wang, Zelin Peng, Xin Chen, Rongxiang Weng, Jingang Wang, Xunliang Cai, Wenrui Dai, and Hongkai Xiong. Onecat: Decoder-only auto-regressive model for unified understanding and generation. arXiv preprint arXiv:2509.03498, 2025.

Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22195–22206, 2024c.

Weixin Liang, Lili Yu, Liang Luo, Srinivasan Iyer, Ning Dong, Chunting Zhou, Gargi Ghosh, Mike Lewis, Wen-tau Yih, Luke Zettlemoyer, and Xi Victoria Lin. Mixture-of-transformers: A sparse and scalable architecture for multi-modal foundation models. arXiv preprint arXiv:2411.04996, 2024.

Chao Liao, Liyang Liu, Xun Wang, Zhengxiong Luo, Xinyu Zhang, Wenliang Zhao, Jie Wu, Liang Li, Zhi Tian, and Weilin Huang. Mogao: An omni foundation model for interleaved multi-modal generation. arXiv preprint arXiv:2505.05472, 2025.

Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. Uniworld: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025a.

Haokun Lin, Teng Wang, Yixiao Ge, Yuying Ge, Zhichao Lu, Ying Wei, Qingfu Zhang, Zhenan Sun, and Ying Shan. Toklip: Marry visual tokens to clip for multimodal comprehension and generation. arXiv preprint arXiv:2505.05422, 2025b.

Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 26689–26699, 2024.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2023a. Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural

information processing systems, 36:34892–34916, 2023b.

Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025a.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pp. 216–233. Springer, 2024a.

Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12):220102, 2024b.

Zhiheng Liu, Weiming Ren, Haozhe Liu, Zijian Zhou, Shoufa Chen, Haonan Qiu, Xiaoke Huang, Zhaochong An, Fanny Yang, Aditya Patel, et al. Tuna: Taming unified visual representations for native unified multimodal models. arXiv preprint arXiv:2512.02014, 2025b.

Jiasen Lu, Liangchen Song, Mingze Xu, Byeongjoo Ahn, Yanjun Wang, Chen Chen, Afshin Dehghan, and Yinfei Yang. Atoken: A unified tokenizer for vision. arXiv preprint arXiv:2509.14476, 2025.

Chuofan Ma, Yi Jiang, Junfeng Wu, Jihan Yang, Xin Yu, Zehuan Yuan, Bingyue Peng, and Xiaojuan Qi. Unitok: A unified tokenizer for visual generation and understanding. arXiv preprint arXiv:2502.20321, 2025a.

Shijie Ma, Yuying Ge, Teng Wang, Yuxin Guo, Yixiao Ge, and Ying Shan. Genhancer: Imperfect generative models are secretly strong vision-centric enhancers. arXiv preprint arXiv:2503.19480, 2025b.

Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Xingkai Yu, et al. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 7739–7751, 2025c.

Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the association for computational linguistics: ACL 2022, pp. 2263–2279, 2022.

Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 2200–2209, 2021.

Minesh Mathew, Viraj Bagal, Rub`en P Tito, Dimosthenis Karatzas, Ernest Valveny, and C. V. Jawahar. InfographicVQA. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 1697–1706, 2022.

Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, et al. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025.

OpenAI. Gpt-4v(ision) system card, 2023. URL https://cdn.openai.com/papers/GPTV_System_

Card.pdf. OpenAI. Gpt-4o. https://openai.com/index/hello-gpt-4o/, 2024. Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu,

Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, et al. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025.

Xuerui Qiu, Yutao Cui, Guozhen Zhang, Junzhe Li, JiaKui Hu, Xiao Zhang, Yang Li, Songtao Liu, Miles Yang, Yu Shi, et al. Hydra: Unifying multi-modal generation and understanding via representation-harmonized tokenization. arXiv preprint arXiv:2603.15228, 2026.

Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 2545–2555, 2025.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael S. Bernstein, Alexander C. Berg, and Li Fei-Fei. Imagenet large scale visual recognition challenge. International Journal of Computer Vision, 115:211 – 252, 2014.

Wei Song, Yuran Wang, Zijia Song, Yadong Li, Haoze Sun, Weipeng Chen, Zenan Zhou, Jianhua Xu, Jiaqi Wang, and Kaicheng Yu. Dualtoken: Towards unifying visual understanding and generation with dual visual vocabularies. arXiv preprint arXiv:2503.14324, 2025.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

- Hao Tang, Chenwei Xie, Xiaoyi Bao, Tingyu Weng, Pandeng Li, Yun Zheng, and Liwei Wang. Unilip: Adapting clip for unified multimodal understanding, generation and editing. arXiv preprint arXiv:2507.23278, 2025.

Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.

Shengbang Tong, David Fan, Jiachen Zhu, Yunyang Xiong, Xinlei Chen, Koustuv Sinha, Michael Rabbat, Yann LeCun, Saining Xie, and Zhuang Liu. Metamorph: Multimodal understanding and generation via instruction tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025.

Shengbang Tong, David Fan, John Nguyen, Ellis Brown, Gaoyue Zhou, Shengyi Qian, Boyang Zheng, Th´eophane Vallaeys, Junlin Han, Rob Fergus, et al. Beyond language modeling: An exploration of multimodal pretraining. arXiv preprint arXiv:2603.03276, 2026a.

Shengbang Tong, Boyang Zheng, Ziteng Wang, Bingda Tang, Nanye Ma, Ellis Brown, Jihan Yang, Rob Fergus, Yann LeCun, and Saining Xie. Scaling text-to-image diffusion transformers with representation autoencoders. arXiv preprint arXiv:2601.16208, 2026b.

Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Chenting Wang, Yuhan Zhu, Yicheng Xu, Jiange Yang, Lang Lin, Ziang Yan, Yali Wang, Yi Wang, and Limin Wang. Internvideo-next: Towards general video foundation models without video-text supervision. arXiv preprint arXiv:2512.01342, 2025a.

Dianyi Wang, Wei Song, Yikun Wang, Siyuan Wang, Kaicheng Yu, Zhongyu Wei, and Jiaqi Wang. Autoregressive semantic visual reconstruction helps vlms understand better. arXiv preprint arXiv:2506.09040, 2025b.

Haochen Wang, Anlin Zheng, Yucheng Zhao, Tiancai Wang, Zheng Ge, Xiangyu Zhang, and Zhaoxiang Zhang. Reconstructive visual instruction tuning. arXiv preprint arXiv:2410.09575, 2024a.

Junke Wang, Yi Jiang, Zehuan Yuan, Bingyue Peng, Zuxuan Wu, and Yu-Gang Jiang. Omnitokenizer: A joint imagevideo tokenizer for visual generation. Advances in Neural Information Processing Systems, 37:28281–28295, 2024b.

Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, and Yu Qiao. Videomae v2: Scaling video masked autoencoders with dual masking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14549–14560, June 2023.

Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Ming Ding, Xiaotao Gu, Shiyu Huang, Bin Xu, et al. Lvbench: An extreme long video understanding benchmark. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 22958–22967, 2025c.

Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024c.

Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, et al. Internvideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191, 2022.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025a.

Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 12966–12977, 2025b.

Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025c.

Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. Advances in Neural Information Processing Systems, 37:28828–28857, 2024a.

Size Wu, Wenwei Zhang, Lumin Xu, Sheng Jin, Zhonghua Wu, Qingyi Tao, Wentao Liu, Wei Li, and Chen Change Loy. Harmonizing visual representations for unified multimodal understanding and generation. arXiv preprint arXiv:2503.21979, 2025d.

Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024b.

Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 13294–13304, 2025a.

Yicheng Xiao, Lin Song, Rui Yang, Cheng Cheng, Zunnan Xu, Zhaoyang Zhang, Yixiao Ge, Xiu Li, and Ying Shan. Haploomni: Unified single transformer for multimodal video understanding and generation. arXiv preprint arXiv:2506.02975, 2025b.

Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.

Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Show-o2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564, 2025a.

Rongchang Xie, Chen Du, Ping Song, and Chang Liu. Muse-vl: Modeling unified vlm through semantic discrete encoding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 24135–24146, 2025b.

Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. Pllava: Parameter-free llava extension from images to videos for video dense captioning. arXiv preprint arXiv:2404.16994, 2024.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024a.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024b.

Jingfeng Yao, Yuda Song, Yucong Zhou, and Xinggang Wang. Towards scalable pre-training of visual tokenizers for generation. arXiv preprint arXiv:2512.13687, 2025a.

Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 15703–15712, 2025b.

Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025.

Lijun Yu, Jos´e Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, et al. Language model beats diffusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023.

Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9556–9567, 2024.

Zhengrong Yue, Haiyu Zhang, Xiangyu Zeng, Boyu Chen, Chenting Wang, Shaobin Zhuang, Lu Dong, KunPeng Du, Yi Wang, Limin Wang, et al. Uniflow: A unified pixel flow tokenizer for visual understanding and generation. arXiv preprint arXiv:2510.10575, 2025.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023.

Pan Zhang, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Rui Qian, Lin Chen, Qipeng Guo, Haodong Duan, Bin Wang, Linke Ouyang, et al. Internlm-xcomposer-2.5: A versatile large vision language model supporting long-contextual input and output. arXiv preprint arXiv:2407.03320, 2024a.

Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024b.

Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun MA, Ziwei Liu, and Chunyuan Li. LLaVA-video: Video instruction tuning with synthetic data. Transactions on Machine Learning Research, 2025. ISSN 2835-8856. URL https://openreview.net/forum?id=EElFGvt39K.

Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690, 2025.

Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024.

##### A Training Details

###### A.1 Tokenizer Training Loss

HYDRA-XTOK is designed as the visual interface of HYDRA-X: before any token reaches the LLM, it must be compact enough for generation, faithful enough for reconstruction, and semantic enough for understanding. We initialize Gen-ViT and Sem-ViT from SigLIP2 (Tschannen et al., 2025). The tokenizer is trained with a reconstruction term and a semantic distillation term:

LHYDRA-XTOK = Lrec + λdistLdist,

where Lrec is the reconstruction term detailed below and Ldist aligns Sem-ViT features with the image and video teachers (Eq. 3).

To keep the compact latent both pixel-faithful and structurally stable, the reconstruction term Lrec encapsulates pixel-level recovery, perceptual fidelity, and latent space regularization. Specifically, it combines an L1 loss for

direct pixel-space reconstruction, an LPIPS perceptual loss Llpips, an adversarial GAN loss Lgan to refine texture realism, and a Kullback–Leibler (KL) divergence penalty that aligns the posterior with a standard normal prior. The comprehensive reconstruction objective is formulated as:

C

∑

Lrec = λ1∥x − xˆ∥1 + λpercLlpips + λganLgan − λKL

j=1

1 + ρj − µ2j − exp(ρj) , (7)

where x and xˆ are the original and reconstructed images, while µj and ρj are the mean and log-variance of the compressed latent.

###### A.2 Tokenizer Pre-training

HYDRA-XTOK is trained in three progressive stages to balance foundational representation learning with highfidelity generative quality:

- Stage 1: Foundation Training. Initialized with SigLIP-2, HYDRA-XTOK first undergoes training on ImageNet1.2M at 256 × 256 resolution. We then transition to mixed-resolution training, combining 256 × 256 videos with images ranging from 256 to 2048 pixels. This strategy empowers the tokenizer to generalize effectively to highresolution video. We optimize the model for 300k iterations using AdamW with a peak learning rate of 2 × 10−4, employing a hybrid SigLIP-2 / InternVideo teacher for distillation.
- Stage 2: Decoder Refinement. To enhance texture realism and perceptual fidelity, we freeze the encoder and exclusively fine-tune the 27-layer ViT decoder. Adversarial training (GAN loss) is incorporated in this stage to significantly improve visual reconstruction.
- Stage 3: Representation Harmonization. In the final stage, we first compute the channel-wise mean and standard deviation of the Gen-ViT latent features. We then freeze Gen-ViT and the decoder while unfreezing Sem-ViT. The Gen-ViT features are normalized before being fed into Sem-ViT and the decoder; during this process, only Sem-ViT is updated. This normalization eliminates feature heterogeneity between the two heads and establishes a unified, semantic-aware latent space capable of faithful reconstruction, which is crucial for downstream UMM tasks.

###### A.3 Native Unified Multimodal Models Pre-training

To cultivate the harmonized nature of HYDRA-X, we implement a three-stage progressive training strategy for the unified multimodal model. Detailed configurations and computational cost are summarised in Table 8.

- Stage 1: Unified Representation Alignment. To resolve the representation divergence at the input level, we freeze the LLM (Qwen2.5-7B-Instruct) and exclusively tune the vision components (projector, time-step embedding, and flow head). Utilizing 100M image–text pairs, this phase aligns the visual latent space with the linguistic domain, ensuring a coherent unified input representation.
- Stage 2: Comprehensive Multimodal Pre-training. We unlock all parameters to facilitate harmonized copromotion within a single unified stream. The model is jointly optimized on a balanced mix of 30M understanding samples and 30M generative samples (strategically filtered from Stage 1). We further incorporate approximately 2M image editing samples and 10M video samples into the joint training process. This full-parameter update ensures the compatibility of the learning process and allows the diverse tasks to mutually reinforce each other.

- Table 8: Training details and computational cost of our HYDRA-X. The HYDRA-XTOK pre-training takes an additional 24h on 256h GPUs. †Data Ratio denotes Text: Image Caption : Image Generation : Video Caption: Video SFT: Image SFT: Edit.

Setting Stage 1 Stage 2 Stage 3

Vision Head: 10−4 Vision Head: 5 × 10−5 Vision Head: 5 × 10−5 LR.

Sem-ViT: 5 × 10−5 LLM & Sem-ViT: 2 × 10−5 LLM & Sem-ViT: 2 × 10−5 Base Resolution 256 512 1024

Batch Size 1024 1024 1024

Tasks Image (Und. & Gen.) + Video & Edit + Text Und. Data Ratio† 0:1:3:0:0:0:0 0:2:6:1:1:0:0 1:0:3:3:0:1:3

Hardware 256 GPUs 512 GPUs 512 GPUs Training Step 50K 200K 20K

Time Cost ∼10h ∼96h ∼24h

- Table 9: Reconstruction comparison on ImageNet, DAVIS, and UCF. All methods are evaluated with a unified protocol using their official implementations: inputs are resized and center-cropped to 256×256 and metrics are computed with identical scripts. Compression ratios are reported separately along the spatial (fs) and temporal (ft) axes; image-only tokenizers have ft=1. Within each subgroup, bold marks the best result and underline marks the second-best. † indicates models trained strictly on the ImageNet-1.2M dataset.

Compression ImageNet DAVIS UCF

Method

Spatial Temporal PSNR(↑) SSIM(↑) rFID(↓) PSNR(↑) SSIM(↑) rFVD(↓) PSNR(↑) SSIM(↑) rFVD(↓) Generation-only Tokenizers

SD-VAE (Rombach et al., 2022) 8× 1× 26.26 0.745 0.606 – – – – – – RAE† (Zheng et al., 2025) 16× 1× 18.05 0.500 2.040 – – – – – – FLUX.1 [dev] (Labs et al., 2025) 8× 1× 32.86 0.917 0.176 – – – – – – Qwen-Image (Wu et al., 2025a) 8× 1× 32.18 0.899 1.459 – – – – – – VAVAE† (Yao et al., 2025b) 16× 1× 27.70 0.798 0.279 – – – – – – Wan2.2 (Wan et al., 2025) 16× 4× 31.25 0.878 0.749 27.64 0.820 14.78 36.11 0.961 4.15

Unified Tokenizers

OmniTokenizer (Wang et al., 2024b) 8× 4× 26.74 0.824 1.023 24.30 0.737 113.56 29.20 0.931 38.15 Vila-U (Wu et al., 2024b) 16× 1× 22.24 0.612 4.231 – – – – – – UniTok (Ma et al., 2025a) 16× 1× 25.34 0.742 0.362 – – – – – – AToken-So/C (Stage 3) (Lu et al., 2025) 16× 4× 29.72 0.848 0.209 26.60 0.784 29.19 34.66 0.953 7.77 HYDRA-XTOK † 16× 1× 32.96 0.905 0.154 – – – – – – HYDRA-XTOK (Stage 3) 16× 4× 32.04 0.898 0.465 28.19 0.835 11.61 36.88 0.967 3.11

- Stage 3: High-Quality Instruction Fine-tuning. The final stage focuses on high-fidelity refinement using curated datasets. For multimodal understanding (MMU), we employ 6M instruction-tuning samples sourced from LLaVA-OneVision (Li et al., 2024a) and Pixmo (Deitke et al., 2025), alongside 1.2M video instruction-tuning samples from LLaVA-Video (Zhang et al., 2025). For generation, we utilize 10M aesthetic-filtered images (derived from Stage 2) and 6M high-fidelity synthetic images. Additionally, we continue to train on high-quality image editing data to further enhance the model’s precise control capabilities.

###### A.4 Ablation Study Training Details

In our ablation studies, the evaluation covers three core capabilities with specific setups: (i) Multimodal Understanding: We train HYDRA-X using the LLaVA-1.5 multimodal understanding dataset (Liu et al., 2023a) combined with the LLaVA-Video SFT dataset (Zhang et al., 2025). (ii) Image Generation: We use Qwen2.5-1.5B (Yang

- et al., 2024a) as the base model, first training on 20M image-caption pairs, then further fine-tuning with the ImgEdit dataset for image editing capabilities. (iii) Image Reconstruction: We train on the ImageNet-1k (1.2M) dataset (Russakovsky et al., 2014) for 150k iterations and assess quality using rFID.

##### B Visual Reconstruction

We benchmark HYDRA-XTOK under a unified protocol against three families of tokenizers: image-only generative VAEs, video VAEs, and joint image–video tokenizers. All inputs are resized and centre-cropped to 256×256 and metrics are computed with identical scripts. Table 9 reports PSNR, SSIM, and rFID/rFVD on ImageNet, DAVIS, and UCF.

To isolate the effect of architecture from that of training data, we additionally report HYDRA-XTOK †, a controlled variant trained strictly on ImageNet-1.2M to match the data budget of RAE† and VAVAE†. Under this matched-data setting, HYDRA-XTOK † outperforms RAE† and VAVAE† by a large margin on every ImageNet metric (e.g.,

+5.26dB PSNR over VAVAE†). More strikingly, despite operating at twice the compression ratio of dedicated 8× image VAEs and using strictly less training data, HYDRA-XTOK † still exceeds the 8× image VAE, FLUX.1, on ImageNet PSNR (32.96 vs. 32.86) and rFID (0.154 vs. 0.176), indicating that the holistic ViT design rather than the data scale drives the gain. The fully trained HYDRA-XTOK is the strongest unified tokenizer on every video metric: it improves over the previous best AToken-So/C by +1.59dB DAVIS PSNR and +2.22dB UCF PSNR, while more than halving rFVD on both datasets (11.61 vs. 29.19 on DAVIS; 3.11 vs. 7.77 on UCF). On video benchmarks, HYDRA-XTOK also outperforms the dedicated 16× video VAE Wan 2.2 (+0.55dB DAVIS PSNR, +0.77dB UCF PSNR; rFVD reduced by 21% and 25% respectively), suggesting that a single holistic ViT with hierarchical patchify is a competitive alternative to cascaded image+video designs.

##### C Evaluation Details of Multi-modal Understanding Benchmarks

To comprehensively evaluate the perception and reasoning capabilities of HYDRA-X, we employ nine diverse benchmarks covering general understanding, expert knowledge, document/chart comprehension, and fine-grained visual perception. We benchmark on AI2D (Kembhavi et al., 2016) (test split), MME (Fu et al., 2023) (test split), MMMU (Yue et al., 2024) (val split), OCRBench (Liu et al., 2024b) (test split), MMBench (Liu et al., 2024a) (dev en split), RealWorldQA (test split), ChartQA (Masry et al., 2022) (test split), DocVQA (Mathew et al., 2021) (val split), and InfoVQA (Mathew et al., 2022) (val split). Table 4 compares HYDRA-X against open-source UMMs at a similar scale. Overall, HYDRA-X matches or exceeds 7B native UMM baselines on most reported metrics, including OCR- and chart-heavy tasks that simultaneously require fine-grained visual details (e.g., character strokes, table cells) and rich semantic structure (e.g., layout and relational reasoning), both of which HYDRA-XTOK’s compact latent is designed to preserve.

##### D Tokenizer-Stage Source–Target Interaction: Visual Evidence

Section 5.3 argues that tokenizer-stage source–target interaction (STI)—routing the source xc and target xt jointly through a shared Sem-ViT with tubelet causal attention rather than encoding them independently—is the missing ingredient for identity-faithful image editing. Quantitatively, this single change recovers nearly 7dB of sourcereconstruction PSNR while leaving the rest of the architecture and parameter count untouched (Table 3). Figure 5 provides the qualitative counterpart.

We compare two variants of HYDRA-X that differ only in this routing step: HYDRA-X-Indep encodes the source and target through two independent Sem-ViT branches—the conventional pipeline shared by BAGEL, OmniGen2, and similar systems—while HYDRA-X-STI routes the pair through a single shared Sem-ViT with tubelet causal attention, treating (xc,xt) as a length-2 clip. Both variants share the same Gen-ViT, the same LLM, and the same flow-matching head, with identical parameter count and training schedule.

The contrast is striking. In the still-life example (top row), HYDRA-X-Indep collapses into a fragmented mosaic where fruit positions, textures, and lighting are all hallucinated locally, whereas HYDRA-X-STI returns a near-pixelperfect reproduction. The car example (bottom row) makes the failure mode of independent encoding explicit: the Indep variant “re-imagines” the car as a different vehicle, removing the driver and passenger and erasing the on-screen text; the STI variant preserves the entire scene including the people inside and the visible plate. These observations confirm the mechanism analysed in Section 5.3: in the conventional pipeline, the latent already loses identity-sensitive information before the LLM ever reads it, so even a perfectly reasoning LLM cannot recover the source faithfully. STI fixes this bottleneck inside the tokenizer at zero parameter cost, which is precisely what enables the consistent margin on identity-sensitive editing dimensions reported in Tables 7 and 12.

##### E Limitations

First, the current scale of training data and model parameters remains a bottleneck, potentially limiting the model’s ability to capture the full complexity of high-dimensional video distributions. Second, resource constraints prevented us from exploring long video generation and video editing, both of which are natural extensions of our holistic encoder. Finally, for a fair comparison, we instantiate HYDRA-X only on a 7B dense LLM; pairing our tokenizer with more advanced backbones, such as MoE (Li et al., 2024b) or MoT (Liang et al., 2024), represents a clear path to further amplify cross-task performance gains.

##### F Broader Impacts

As HYDRA-X introduces strong text-to-image generation capabilities within a unified framework, we acknowledge potential downstream risks. These include the generation of misleading or fabricated visual content (e.g., deepfakes), which could be exploited for disinformation or impersonation. To mitigate such risks, we advocate for the

Input HYDRA-X-Indep HYDRA-X-STI

[Figure 52]

[Figure 53]

[Figure 54]

Input HYDRA-X-Indep HYDRA-X-STI

[Figure 55]

[Figure 56]

[Figure 57]

Input HYDRA-X-Indep HYDRA-X-STI

[Figure 58]

[Figure 59]

[Figure 60]

Input HYDRA-X-Indep HYDRA-X-STI

[Figure 61]

[Figure 62]

[Figure 63]

- Figure 5: Qualitative effect of tokenizer-stage source–target interaction. Source-image reconstruction produced by HYDRA-X-Indep (independent Sem-ViT encoding of source and target, the conventional pipeline) versus HYDRA-X-STI (joint encoding through tubelet causal attention, our proposal). The two variants share every other architectural component. HYDRA-X-STI preserves identity-sensitive details (object layout, characters, on-screen text) that HYDRA-X-Indep loses, despite both pipelines using the same LLM and the same number of parameters.

incorporation of content watermarking and provenance tracking mechanisms upon deployment, as well as adherence to responsible release practices such as gated model access and usage guidelines. We believe that advancing the scientific understanding of unified multimodal architectures carries substantial positive societal value, while the

associated risks can be effectively managed through community-driven safety standards.

##### G Additional Main Results

This section provides the full per-category breakdown of the benchmarks summarised in Section 6, complementing the condensed tables in the main paper.

GenEval. Table 10 reports the per-category breakdown on GenEval (Ghosh et al., 2023), covering single-object, two-object, counting, color, position, and color-attribute prompts. The breakdown helps locate the compositional dimensions where each model is strongest.

- Table 10: Detailed image generation results on the GenEval benchmark (Ghosh et al., 2023). Rows in gray indicate models with ≥ 14B parameters and are excluded from the ranking. † refers to methods using LLM rewriters.

Models Size Single Object Two Objects Count Colors Position Color Attribute Overall

Generation-only Models

SD3-Med (Esser et al., 2024) 2B 0.99 0.94 0.72 0.89 0.33 0.60 0.74 FLUX.1 [Dev] (Labs et al., 2025) 12B 0.98 0.93 0.75 0.93 0.68 0.65 0.82 DALL-E 3 (Betker et al., 2023) – 0.96 0.87 0.47 0.83 0.43 0.45 0.67

Unified Multimodal Models

TokenFlow-XL (Qu et al., 2025) 14B 0.95 0.60 0.41 0.81 0.16 0.24 0.55 SEED-X (Ge et al., 2024) 17B 0.97 0.58 0.26 0.80 0.19 0.14 0.49 Ming-UniVision (Huang et al., 2025) 16B 1.00 0.93 0.59 0.93 0.92 0.70 0.85 BAGEL† (Deng et al., 2025) 14B 0.98 0.95 0.84 0.95 0.78 0.77 0.88 MetaQuery-XL (Pan et al., 2025) 7B – – – – – – 0.80 Blip3-o† (Chen et al., 2025a) 8B – – – – – – 0.84 MUSE-VL (Xie et al., 2025b) 7B 0.98 0.64 0.54 0.72 0.25 0.31 0.57 Janus-Pro (Chen et al., 2025b) 7B 0.99 0.89 0.59 0.90 0.79 0.66 0.80 Show-o2† (Xie et al., 2025a) 7B 1.00 0.87 0.58 0.92 0.52 0.62 0.76 HYDRA† (Qiu et al., 2026) 7B 1.00 0.97 0.68 0.91 0.81 0.80 0.86 HYDRA-X 7B 0.99 0.95 0.83 0.90 0.84 0.75 0.88

WISE. Table 11 reports the per-category breakdown on WISE (Niu et al., 2025), which probes world knowledge across culture, time, space, biology, physics, and chemistry, and is therefore complementary to the geometric and compositional probes of GenEval.

- Table 11: Detailed image generation results on the WISE benchmark (Niu et al., 2025). Rows in gray indicate models with ≥ 14B parameters and are excluded from the ranking.

Models Size Culture Time Space Biology Physics Chemistry Overall

Generation-only Models

FLUX.1 [Dev] (Labs et al., 2025) 12B 0.48 0.58 0.62 0.42 0.51 0.35 0.50 SD3.5-Large (Esser et al., 2024) 8B 0.44 0.50 0.58 0.44 0.52 0.31 0.46

Unified Multimodal Models

BAGEL (Deng et al., 2025) 14B 0.44 0.55 0.68 0.44 0.60 0.39 0.52 VILA-U-7B (Wu et al., 2024b) 7B 0.26 0.33 0.37 0.35 0.39 0.23 0.31 Janus-Pro-7B (Chen et al., 2025b) 7B 0.30 0.37 0.49 0.36 0.42 0.26 0.35 Emu3-Gen-8B (Wang et al., 2024c) 8B 0.34 0.45 0.48 0.41 0.45 0.27 0.39 Show-o2 (Xie et al., 2025a) 7B 0.40 0.45 0.58 0.39 0.53 0.34 0.44 HYDRA (Qiu et al., 2026) 7B 0.52 0.53 0.68 0.47 0.58 0.38 0.53 HYDRA-X 7B 0.53 0.57 0.72 0.53 0.64 0.40 0.56

ImgEdit-Bench. Table 12 provides the full per-dimension breakdown on ImgEdit-Bench (Ye et al., 2025), spanning nine instruction-guided editing operations from object addition and removal to background replacement, style transfer, and compositional edits.

VBench. Table 13 expands the QS/SS/Total summary in the main paper to all fourteen VBench (Huang et al.,

- 2024) dimensions, separately probing visual quality, motion smoothness, dynamic degree, semantic correctness, and compositional reasoning.

- Table 12: Detailed image editing results on the ImgEdit-Bench (Ye et al., 2025). Editing dimensions: Add, Adj.(Alter), Ext.(Extract), Rep.(Replace), Rm.(Remove), Bg.(Background), Sty.(Style), Hyb.(Compose), Act.(Action). Rows in gray indicate models with ≥ 14B parameters and are excluded from the ranking.

Models Size Add Adj. Ext. Rep. Rm. Bg. Sty. Hyb. Act. Overall

Generation-only Models

FLUX.1 Kontext [Pro] (Labs et al., 2025) 12B 4.25 4.15 2.35 4.56 3.57 4.26 4.57 3.68 4.63 4.00 Qwen-Image (Wu et al., 2025a) 20B 4.38 4.16 3.43 4.66 4.14 4.38 4.81 3.82 4.69 4.27

Unified Multimodal Models

BAGEL (Deng et al., 2025) 14B 3.56 3.31 1.70 3.30 2.62 3.24 4.49 2.38 4.17 3.20 OmniGen (Xiao et al., 2025a) 3.8B 3.47 3.04 1.71 2.94 2.43 3.21 4.19 2.24 3.38 2.96 UniWorld-V1 (Lin et al., 2025a) 12B 3.82 3.64 2.27 3.47 3.24 2.99 4.21 2.96 2.74 3.26 OmniGen2 (Wu et al., 2025c) 4B 3.57 3.06 1.77 3.74 3.20 3.57 4.81 2.52 4.68 3.44 HYDRA-X 7B 4.49 4.27 4.04 4.41 4.38 4.30 4.77 3.43 4.32 4.34

- Table 13: Detailed video generation results on VBench (Huang et al., 2024). Column abbreviations: QS: Quality Score, SS: Semantic Score, SC: Subject Consistency, BC: Background Consistency, MS: Motion Smoothness, DD: Dynamic Degree, AQ: Aesthetic Quality, IQ: Imaging Quality, OC: Object Class, MO: Multiple Objects, HA: Human Action, C: Color, SR: Spatial Relationship, S: Scene.

Models Size QS SS SC BC MS DD AQ IQ OC MO HA C SR S Total

Generation-only Models

CogVideoX (Yang et al., 2024b) 5B 82.75 77.04 96.23 96.52 96.92 70.97 61.98 62.90 85.23 62.11 99.40 82.81 66.35 53.20 81.61 Hunyuan Video (Kong et al., 2024) 13B 85.07 76.88 97.22 97.60 99.05 71.94 60.28 67.24 83.48 66.71 94.40 89.79 72.13 54.46 83.43

Unified Multimodal Models

VILA-U (Wu et al., 2024b) 7B 76.26 65.04 – – – – – – – – – – – – 74.01 HaploOmni (Xiao et al., 2025b) 7B – – 96.40 97.60 96.80 65.30 – – – – – – – – 78.10 Emu3 (Wang et al., 2024c) 8B – – 95.32 97.69 98.93 79.27 59.64 – 86.17 44.64 77.71 – 68.73 37.11 80.96 Show-o2 (Xie et al., 2025a) 1.5B 82.10 78.31 97.28 96.78 98.25 40.83 65.15 67.06 94.81 76.01 95.20 80.89 62.61 57.67 81.34 HYDRA-X 7B 83.97 81.57 95.70 95.88 92.69 35.42 63.73 65.23 96.52 76.68 99.00 87.37 73.88 69.74 83.49

##### H Qualitative Comparisons

This section presents qualitative results across the five tasks supported by HYDRA-X. We compare against representative baselines drawn from both unified multimodal models and task-specialised systems, and organise the comparisons by task and resolution.

###### H.1 Image Reconstruction at 512×512

We first inspect reconstruction fidelity at the standard 512×512 resolution. The comparison spans three families of baselines: dedicated image VAEs (FLUX), unified tokenizers built into UMMs (MingTok, AToken), and the recently proposed RAE. The visual difference makes texture, fine-edge, and small-text fidelity directly comparable.

Input RAE MingTok Atoken FLUX HYDRA-X

|[Figure 64]|
|---|

|[Figure 65]|
|---|

|[Figure 66]|
|---|

|[Figure 67]|
|---|

|[Figure 68]|
|---|

|[Figure 69]|
|---|

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

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

Input RAE MingTok Atoken FLUX HYDRA-X

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

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

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

Input RAE MingTok Atoken FLUX HYDRA-X

|[Figure 88]|
|---|

|[Figure 89]|
|---|

|[Figure 90]|
|---|

|[Figure 91]|
|---|

[Figure 92]

[Figure 93]

|[Figure 94]|
|---|

|[Figure 95]|
|---|

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

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

Input RAE MingTok Atoken FLUX HYDRA-X

|[Figure 100]|
|---|

|[Figure 101]|
|---|

|[Figure 102]|
|---|

|[Figure 103]|
|---|

|[Figure 104]|
|---|

|[Figure 105]|
|---|

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

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

- Figure 6: Qualitative reconstruction comparison at 512×512. We compare HYDRA-X against RAE (Zheng

- et al., 2025), MingTok (Huang et al., 2025), AToken (Lu et al., 2025), and FLUX (Labs et al., 2025).

###### H.2 Image Reconstruction at 1280×768

To stress-test generalisation beyond the training resolution, we additionally compare reconstructions at a high resolution of 1280×768 and include the dedicated video VAE Wan 2.2 alongside the image-only baselines. This setting exposes how each tokenizer handles dense fine details such as text, foliage, and small structural elements when the spatial token budget is stretched.

WAN2.2 Atoken

[Figure 112]

[Figure 113]

Input

| |
|---|

| |
|---|

[Figure 114]

|[Figure 115]|
|---|

|[Figure 116]|
|---|

| |
|---|

FLUX HYDRA-X

|[Figure 117]|
|---|

[Figure 118]

[Figure 119]

| |
|---|

| |
|---|

|[Figure 120]|
|---|

|[Figure 121]|
|---|

WAN2.2 Atoken

[Figure 122]

[Figure 123]

| |
|---|

| |
|---|

Input

[Figure 124]

| |
|---|

|[Figure 125]|
|---|

|[Figure 126]|
|---|

FLUX HYDRA-X

[Figure 127]

[Figure 128]

|[Figure 129]|
|---|

| |
|---|

| |
|---|

|[Figure 130]|
|---|

|[Figure 131]|
|---|

- Figure 7: Qualitative reconstruction comparison at 1280×768. We compare HYDRA-X against Wan 2.2 (Wan et al., 2025), AToken (Lu et al., 2025), and FLUX (Labs et al., 2025).

###### H.3 Video Reconstruction at 512×512

Beyond static images, we visualise temporally consecutive frames reconstructed by HYDRA-X against the dedicated video VAE Wan 2.2 and the joint image–video tokenizer AToken. This helps assess whether HYDRA-XTOK’s holistic ViT preserves motion-sensitive cues such as object boundaries and inter-frame consistency.

Input Atoken WAN2.2 HYDRA-X

|[Figure 132]|
|---|

|[Figure 133]|
|---|

|[Figure 134]|
|---|

|[Figure 135]|
|---|

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 140]|
|---|

|[Figure 141]|
|---|

|[Figure 142]|
|---|

|[Figure 143]|
|---|

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

TimeStep

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 148]|
|---|

|[Figure 149]|
|---|

|[Figure 150]|
|---|

|[Figure 151]|
|---|

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 156]|
|---|

|[Figure 157]|
|---|

|[Figure 158]|
|---|

|[Figure 159]|
|---|

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Input Atoken WAN2.2 HYDRA-X

|[Figure 164]|
|---|

|[Figure 165]|
|---|

|[Figure 166]|
|---|

|[Figure 167]|
|---|

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 172]|
|---|

|[Figure 173]|
|---|

|[Figure 174]|
|---|

|[Figure 175]|
|---|

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

TimeStep

|[Figure 180]|
|---|

|[Figure 181]|
|---|

|[Figure 182]|
|---|

|[Figure 183]|
|---|

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 188]|
|---|

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

|[Figure 193]|
|---|

|[Figure 194]|
|---|

|[Figure 195]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Figure 8: Qualitative video reconstruction comparison. We compare HYDRA-X against Wan 2.2 (Wan et al.,

- 2025) and AToken (Lu et al., 2025).

###### H.4 Image Generation

We provide qualitative text-to-image samples produced by HYDRA-X spanning a diverse range of prompts—from realistic photography and stylised illustration to compositional and knowledge-driven scenes—to characterise the model’s coverage and aesthetic quality.

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

###### Figure 9: Qualitative image generation results from HYDRA-X.

###### H.5 Video Generation

We similarly present qualitative text-to-video samples covering varied subjects, scenes, and motion patterns, illustrating how the holistic latent supports temporally coherent synthesis under the same UMM backbone.

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

A majestic, photorealistic wide shot of snow-capped mountains under a bright, clear blue sky. Fluffy white clouds elegantly form

the word 'HYDRA-X' in the center of the sky. The text clouds blend naturally with scattered, regular clouds around them. Cinematic lighting, highly detailed, with gentle, realistic cloud movement and a slow, sweeping camera pan.

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

A grand, expansive vista of a dramatic sunset blending into a starry night sky. The top is a clear blue with the Milky Way and stars visible, while the bottom horizon glows with rich orange or red hues. Fluffy clouds are scattered across the scene.

Timelapse-style camera showing the gradual shift of the sky and stars. Gentle camera pan.

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

A wide-angle, cinematic shot of a vast golden desert with detailed, wind-swept sand dunes in the foreground. Above, a breathtakingly clear, deep blue night sky filled with bright stars and a vividly glowing Milky Way galaxy. In the distant horizon, a mysterious, massive dark silhouette resembling a giant hand or abstract rock formation rises towards the stars. Surreal

atmosphere, highly detailed, with a smooth, slow camera pan across the silent landscape.

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

A close-up view of a brawny humanoid werewolf-like creature with grey fur, standing in a blurry forest setting. The creature looks directly into the lens with an intense, weary expression. It holds its hands firmly around an ornate, branch-decorated arrow

with a red feather that is shot into its bloodied chest. Cinematic, dramatic lighting. Subtle chest movement as it breathes.

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

A smiling female astronaut floating in zero gravity inside a space station, her blonde hair drifting freely in the air, with a view of Earth and space through the window behind her.

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

A surreal dreamscape of a peaceful beach with glowing, translucent jellyfish floating in a pastel aurora sky, surrounded by giant

colorful flowers and gentle ocean waves.

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

A beautiful watercolor-style animation of a great white shark swimming gracefully underwater above a vibrant coral reef.

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

A magical fantasy landscape featuring a glowing cyan river, vibrant colorful flora, a majestic golden tree, and glowing mountains under a purple starry sky with two moons.

###### Figure 10: Qualitative video generation results from HYDRA-X.

###### H.6 Image Editing

Finally, we compare HYDRA-X against representative editing systems on a set of instruction-guided edits. The baselines include both unified multimodal models (BAGEL, OmniGen2) and editing-specialised generators (QwenImage-Edit, Step1X-Edit), allowing readers to gauge identity preservation, instruction adherence, and visual quality side-by-side.

#### Src HYDRA-X Bagel Qwen-Edit

#### Step1-X Omnigen2

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

Add a coffee cup on the table in the foreground.

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

Replace the architecture in the image with a large tree.

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

Replace the sliced steak in the image with a folded umbrella.

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

Change the tent in the picture from the forest to the beach.

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

Extract the colorful striped top worn by the person in the image.

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

Extract the navy blue T-shirt worn by the person in the image.

Figure 11: Qualitative editing comparison. We compare HYDRA-X against BAGEL (Deng et al., 2025), QwenImage-Edit (Wu et al., 2025a), Step1X-Edit (Liu et al., 2025a), and OmniGen2 (Wu et al., 2025c).

