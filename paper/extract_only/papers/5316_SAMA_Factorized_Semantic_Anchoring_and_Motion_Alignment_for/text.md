# arXiv:2603.19228v1[cs.CV]19Mar2026

## SAMA: Factorized Semantic Anchoring and Motion Alignment for Instruction-Guided Video Editing

##### Xinyao Zhang1,2*, Wenkai Dong1*, Yuxin Song1*†, Bo Fang1,3, Qi Zhang1, Jing Wang1,2, Fan Chen1, Hui Zhang1, Haocheng Feng1, Yu Lu4‡, Hang Zhou1, Chun Yuan2, Jingdong Wang1 1Baidu 2Tsinghua University 3City University of Hong Kong 4Zhejiang University

Project Page: https://cynthiazxy123.github.io/SAMA Email Address: songyuxinbb@outlook.com

### Abstract

Current instruction-guided video editing models struggle to simultaneously balance precise semantic modifications with faithful motion preservation. While existing approaches rely on injecting explicit external priors (e.g., VLM features or structural conditions) to mitigate these issues, this reliance severely bottlenecks model robustness and generalization. To overcome this limitation, we present SAMA (factorized Semantic Anchoring and Motion Alignment), a framework that factorize video editing into semantic anchoring and motion modeling. First, we introduce Semantic Anchoring which establish a reliable visual anchor by jointly predicting semantic tokens and video latents at sparse anchor frames, enabling purely instruction-aware structural planning. Second, Motion Alignment pre-trains the same backbone on motion-centric video restoration pretext tasks (cube inpainting, speed perturbation, and tube shuffle), enabling the model to internalize temporal dynamics directly from raw videos. SAMA is optimized with a two-stage pipeline: a factorized pre-training stage that learns inherent semantic-motion representations without paired video-instruction editing data, followed by supervised fine-tuning on paired editing data. Remarkably, the factorized pre-training alone already yields strong zero-shot video editing ability, validating the proposed factorization. SAMA achieves state-of-the-art performance among open-source models and is competitive with leading commercial systems (e.g. Kling-Omni). Code, models, and datasets will be released.

### 1 Introduction

Diffusion models have enabled interactive, instruction-guided image editing with impressive fidelity and controllability [1–9]. Extending this paradigm from single images to videos, however, remains substantially more challenging. A practical instruction-guided video editor must (i) apply fine-grained semantic changes that follow the instruction, while (ii) preserving temporally coherent motion of the edited subject, background, and camera. In current models, these two requirements often conflict: aggressive semantic changes induce localized artifacts, identity drift, and texture popping, whereas enforcing temporal consistency can dilute the intended edit and reduce instruction fidelity (Fig. 1 top). This tension has been widely observed in diffusion-based video editing and adaptation works [10–13].

To mitigate these issues, a prevailing trend in existing approaches is to rely on injecting explicit external priors, such as VLM-extracted semantic conditions [14, 15] or structural signals like skeletons and depth maps [16, 17]. We argue that this over-reliance reflects a significant bottleneck, which constrains the diffusion backbone from learning inherent semantic-motion representations for precise semantic editing and faithful motion alignment with the source video dynamics. Instead, we attribute the core difficulty of instruction-guided video editing to the lack of factorization between semantic

∗Equal Contribution †Project Leader ‡Corresponding Author

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Source video

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

SAMA

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

KlingOmni

Add a brown hat on the man's head.

Remove the football.

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

Change all person's clothes color into black. Turn into watercolor style.

Remove Instruct follow

VIE-Bench

[Figure 37]

[Figure 38]

[Figure 39]

Semantic prediction

Video prediction

Video prediction

10

[Figure 40]

Semantic

Style Quality Kling-Omni

Remove Preservation

[Figure 41]

[Figure 42]

Semantic

Motion

8

Pika

T2I Model

SAMA

6

Style Preservation

Remove Quality

InsV2V

VLM / Motion Extractor

###### VACE

4

[Figure 43]

Motion

Omni-Video

2

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Pretext perturbation

MiniMax

Style Instruct follow

Swap Instruct follow

0

Instruction Source video Source video Instruction

DiffuEraser

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

UniVideo

InstructX

Add Quality

Swap Preservation

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

SAMA (Ours)

Add Preservation

Swap Quality Add Instruct follow

[Figure 60]

[Figure 61]

[Figure 62]

Source video Motion Motion

[Figure 63]

Source video

Semantic Semantic

Replace the man's white dress shirt with a light blue short-sleeve polo shirt, maintaining the same position and pose within the video scene.

Remove the wolf-like dog on the right.

- Figure 1: Teaser and overview. Top: qualitative comparisons on VIE-Bench, comparing SAMA with representative open- and closed-source systems. Bottom left: illustration of SAMA’s semantic– motion training objectives. Bottom right: fine-grained VIE-Bench performance comparison.

structure planning and motion modeling [18–22]. Semantic edits are typically sparse and temporally stable: a small number of anchor frames is often sufficient to determine the desired visual modification. In contrast, motion coherence follows physical and temporal dynamics that can be learned from large-scale raw videos without explicit editing supervision.

Based on this observation, we propose SAMA (factorized Semantic Anchoring and Motion Alignment), a framework that encourages the model to learn semantic structure planning and motion modeling as two complementary capabilities. First, we introduce Semantic Anchoring which predicts semantic tokens together with video latents to support instruction-aware structural planning in the semantic space while retaining high-fidelity rendering in the latent space. Second, Motion Alignment strengthens temporal reasoning through motion-centric video restoration tasks, encouraging the backbone to internalize coherent temporal dynamics directly from raw videos.

To realize this factorized learning paradigm, we train SAMA with a two-stage strategy. In the first stage, a factorized pre-training process encourages the model to internalize semantic anchoring and motion dynamics as two complementary capabilities, without requiring paired instruction-guided video editing data. Remarkably, we find that this stage alone already induces strong zero-shot video editing behavior. This observation suggests that robust instruction-guided video editing can naturally emerge once a model learns to jointly reason about semantic intent and temporal dynamics. In the subsequent supervised fine-tuning stage, the model is trained on paired video editing datasets to resolve residual semantic–motion conflicts and improve visual fidelity. Consequently, SAMA

achieves state-of-the-art performance among open-source models while delivering results comparable to leading commercial systems (e.g. Kling-Omni [15], Runway [23]).

- • We propose a factorized perspective on instruction-guided video editing that separates semantic planning from motion modeling, reducing reliance on brittle external priors.
- • We introduce Semantic Anchoring and Motion Alignment via motion-centric video restoration pre-training, enabling the diffusion backbone to internalize robust semantic and temporal representations.
- • SAMA achieves state-of-the-art performance among open-source video editing models and is competitive with leading commercial systems. Code, models, and datasets will be publicly released.

### 2 Related Work

##### 2.1 Instruction-Guided Video Editing

Instruction-guided video editing aims to edit an input video following a text instruction, with the key challenge of preserving temporal consistency. Early diffusion-based attempts [24, 13, 25– 28, 10, 12] in instruction-guided video editing mainly follow zero-shot or one-/few-shot paradigms, where pretrained text-to-image diffusion models are repurposed for videos with additional temporal modeling to maintain consistency.

With the release of large-scale instruction-guided video editing datasets such as Señorita-2M [29], InsViE-1M [30], Ditto-1M [31], ReCo-Data [32], and OpenVE-3M [33], recent research has shifted toward data-driven video editing models trained end-to-end. Ditto [31] builds its large-scale synthetic data pipeline by combining a strong image editing model with an in-context video generation model, and then trains a model on Ditto-1M to improve instruction-guided and temporal consistency. OpenVE-3M [33] expands supervision across diverse editing categories, while ReCo-Data [32] focuses on region-aware instruction editing to improve local controllability.

Several recent works [34–41, 32] further explore unified and in-context formulations for video editing. UNIC [34] unifies different video editing tasks by converting the noisy video latents, source video tokens, and multi-modal condition tokens into a single sequence, so a Diffusion Transformer can learn editing behaviors in-context without task-specific adapters or DDIM inversion. VACE [36] explores a unified and controllable editing formulation that supports diverse edit operations, improving the generality and robustness of instruction-guided video editing. ICVE [39] proposes a low-cost pretraining strategy that uses unpaired video clips to learn general editing ability in-context, and then refines the model with a small amount of paired editing data. EditVerse [40] proposes a unified framework for image/video generation and editing by representing text, images, and videos in a shared token space, enabling strong in-context editing and supporting data-driven training with large-scale benchmarks. DiffuEraser [35] studies instruction-guided video object removal by integrating diffusion-based editing with temporal-consistent inpainting, aiming to erase targets while preserving coherent backgrounds across frames. ReCo [32] introduces a joint source-target video diffusion framework and applies region constraints to improve instruction-guided editing. VideoCoF [41] introduces a Chain-of-Frames “see–reason–edit” formulation that predicts where/how to edit across frames before generation, improving instruction-to-region alignment and temporal consistency without requiring user-provided masks.

Beyond editing-centric models, unified video understanding and generation frameworks such as Omni-Video [42], InstructX [14], UniVideo [43], and VINO [44] provide strong representations for video content and motion dynamics.

##### 2.2 Semantic Alignment on Image and Video Generation

Recent progress in image and video generation also benefits from semantic alignment between generative models and strong pretrained encoders. In image generation, REPA [45] aligns intermediate denoising features with clean features from a pretrained image encoder, which stabilizes training and improves generation quality. Following REPA, several works study how to apply representation alignment more effectively, including end-to-end VAE–diffusion training (REPA-E [46]), stage-

|x type token<br><br>| |
|---|
<br><br>source video<br><br>| |
|---|
<br><br>noisy semantic<br><br>+ token adding<br><br>| |
|---|
<br><br>noisy target<br><br>| |
|---|
<br><br>predicted target<br><br>predicted semantic|
|---|

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

###### …

| |
|---|

#### DiT Blocks

[Figure 68]

🔥

+ +

###### + +

+

+

+

0 … 0

…

0

1

2

2

2

source video tokens

noisy semantic tokens

noisy target video tokens

Projector 🔥

[Figure 69]

Text Encoder

VAE Encoder

VAE Encoder

Semantic Encoder

|[Figure 70]|
|---|

[Figure 71]

[Complete the missing regions in the video.] “A woman wearing a white sweater, with a long ponytail and ...”

|[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]|
|---|

|[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]<br><br>[Figure 78]<br><br>[Figure 79]|
|---|

Anchor Frames

- Stage 0 (I2I & T2V)
- Stage 1 (I2I & V2V) Anchor Frames

inpainting speed shuffle

[Figure 80]

[Figure 81]

[Figure 82]

Factorized Pre-training

Source Video Editing Instruction & Caption

Target Video

Pretext Perturbation

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

|[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]|
|---|

Transform the long, straight hair into a tightly braided updo with a single red ribbon.

|[Figure 91]|
|---|

|[Figure 92]<br><br>[Figure 93]<br><br>[Figure 94]|
|---|

Supervised Fine-tuning

Source Video Editing Instruction

Target Video

- Figure 2: Overall pipeline. SAMA first performs factorized pre-training (stage 0) on additional perturbed videos by completing a pretext task conditioned on the given captions. It then performs normal supervised fine-tuning (stage 1) on original source videos. Semantic Anchoring is incorporated in both stages to jointly facilitate semantic representation learning and instruction-guided video editing.

wise scheduling to avoid late-stage degradation (HASTE [47]), teacher-free self-alignment via self-distillation (SRA [48]).

Similar ideas have recently been extended to video generation. SemanticGen [49] first predicts compact semantic features and then generates VAE latents conditioned on them, which is more efficient for long videos. VideoREPA [50] distills spatio-temporal relational knowledge from video foundation models into text-to-video diffusion models via token-relation alignment. Beyond generation, this relational alignment idea has been adopted for video editing: FFP-300K [51] uses inter-frame relational distillation inspired by VideoREPA to better preserve source motion.

Positioning. Inspired by recent advances in semantic alignment for image/video generation, we apply semantic-alignment regularization to instruction-guided video editing. Our approach improves instruction following and temporal consistency, and accelerates DiT convergence during training, without heavy test-time optimization.

- 2.3 Self-supervised Learning for Video Representation Learning

Self-supervised learning learns spatiotemporal representations from unlabeled videos via pretext tasks. Motivated by this line of work, we adopt lightweight pretext tasks as motion-centric restoration objectives in our Motion Alignment (Sec. 3.3) to better capture coherent temporal dynamics. Prior works mainly fall into three categories: speed-based learning (e.g., SpeedNet [52], PRP [53], Pace Prediction [54]), spatiotemporal puzzles (e.g., Space-Time Cubic Puzzles [55]), and reconstructionbased objectives (e.g., masked video modeling and VideoMAE [56]).

- 3 Method

Preliminary We adopt a video diffusion transformer framework trained via the flow matching [57] paradigm. The main training objective is to minimize the expected flow matching loss, defined as:

0,x1∥vθ(xt,t) − (x1 − x0)∥22, (1)

LFM(θ) = Et,x

where x1 is the target video and x0 is the Gaussian prior. The network vθ learns to regress the vector field x1 − x0 from the intermediate state xt = tx1 + (1 − t)x0. This formulation corresponds to the

flow ordinary differential equation:

dx dt

= vθ(x,t). (2)

##### 3.1 SAMA

SAMA is built upon the video diffusion model Wan2.1-T2V-14B [58]. Given a source video Vs and an editing instruction y, the goal is to generate an edited target video Vt that follows y while preserving realistic spatiotemporal motion and non-edited content.

Latent tokenization. We encode videos into VAE latents following latent diffusion style formulations [59]. The source and target videos are represented as token sequences zs and zt. We form an in-context V2V input by concatenating the source and (noisy) target token sequences: z = [zs ; zt]. Type embeddings. To disambiguate token roles, we add a learned type embedding to each token: type id 0 for source-video latent tokens zs, type id 2 for target-video latent tokens zt, and type id 1 for semantic tokens introduced by Semantic Anchoring (Sec. 3.2). This convention is used consistently across all stages. We empirically observe that using type embeddings leads to faster convergence than the commonly used shifted RoPE scheme [60, 61], while minimally perturbing the backbone prior. We provide further discussion and supporting evidence in the Appendix.

SAMA internalizes two complementary capabilities within the diffusion backbone: Semantic Anchoring (SA) provides instruction-consistent anchors on sparse anchor frames to stabilize structural editing (see Sec. 3.2); Motion Alignment (MA) aligns the edited video with the source motion dynamics through motion-centric pretext supervision, improving temporal stability and mitigating semantic–motion conflicts (see Sec. 3.3). Building on these two capabilities, we further introduce a two-stage training strategy: we first learn strong inherent semantic–motion representations in a factorized pre-training stage, and then strengthen editing performance with paired supervision in an SFT stage (Sec. 3.4).

##### 3.2 Semantic Anchoring

Semantic Anchoring (SA) is introduced as an auxiliary objective throughout both the Factorized Pre-training Stage and the SFT Stage. For an image sample, the target image serves as the anchor. For a video sample, we uniformly sample N frames from the target video and treat them as sparse anchor frames. Each anchor frame is encoded by a SigLIP image encoder [62] to obtain patch-level semantic features. We then aggregate these features into a compact token set by pooling, producing M local semantic tokens that capture region-level semantics along with one global token that summarizes the overall content. All semantic tokens are finally projected by a lightweight two-layer MLP into the same embedding space as the VAE latent tokens.

Injecting semantic tokens into the denoising sequence. Let ˆs denote the projected semantic tokens extracted from the N anchor frames. We prepend ˆs to the target latent sequence and treat them as part of the denoising trajectory: we apply the same forward noising process to both semantic tokens and target latents, and feed the concatenated noisy sequence into the DiT. After denoising, we read out the positions corresponding to the semantic tokens and pass them through a semantic prediction head attached to the final DiT layer, yielding predicted semantic tokens s.

Objective. We supervise semantic prediction with an ℓ1 loss between the predicted tokens and the extracted anchor tokens:

Lsem = ∥ˆs − s∥1. (3) The overall training objective combines the flow-matching loss and the Semantic Anchoring loss:

L = LFM + λ · Lsem. (4)

##### 3.3 Motion Alignment

Motion Alignment (MA) is applied on video samples in the Factorized Pre-training Stage (Sec. 3.4). Given a source video Vs and instruction y, we apply a motion-centric transformation T only to the source video to obtain V˜s = T (Vs), while keeping the target side unchanged (i.e., always using the original target video without augmentation). This design forces the model to learn motion recovery

1 2 3 4 5 6 7 8

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

A B C D

Original Video

[Figure 103]

[Figure 104]

[Figure 105]

Cube Inpainting Speed Perturbation

###### Tube Shuffle

|[Figure 106]<br><br>3|[Figure 107]<br><br>4|
|---|---|

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

7 2 8 1

1 2

2 4

1

A

B D

3

C

drop

drop

Perturbed Video

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

5

6 7 8

6 7 8

5

5 3 6 4

drop

drop

##### Figure 3: Illustration of pretext perturbations.

and temporal reasoning from the source stream, improving robustness under fast motion and complex camera dynamics. Fig. 9 provides an illustration of the pretext perturbations.

Motion-centric transformations. We adopt three restoration-style perturbations inspired by selfsupervised learning for visual sequences [56, 63, 64]: (i) Cube Inpainting: mask a continuous temporal block in V˜s and recover missing content conditioned on the remaining frames; (ii) Speed Perturbation: temporally accelerate V˜s and learn to restore normal dynamics, improving robustness to motion-rate changes; (iii) Tube Shuffle: partition V˜s into a 2×2×2 spatio-temporal tube grid and randomly permute tubes, forcing the model to reason about spatio-temporal structure and restore consistent motion.

Prompting for pretext tasks. To make the objective explicit and unify the formulation across tasks, we prepend a short task token to the editing instruction:

Task token examples for Motion Alignment

- • [Complete the missing regions in the video.] (Cube Inpainting)
- • [Restore the video to normal playback speed.] (Speed Perturbation)
- • [Restore the correct spatio-temporal order of the video segments.] (Tube Shuffle)

Overall, MA encourages the backbone to internalize robust motion dynamics from the source stream while remaining fully compatible with the instruction-conditioned editing formulation.

##### 3.4 Training Strategies

SAMA is optimized with a two-stage training pipeline that mirrors our factorized view of instructionguided video editing.

- Stage 0: Factorized Pre-training. We start from a strong text-to-video prior and pre-train it on a mixture of instruction-based image editing pairs and large-scale text-to-video data [65, 66]. The image editing portion provides broad semantic coverage and improves general instruction grounding, while the text-to-video portion supplies diverse real-world motion patterns. During this stage, we apply SA to both image and video samples, and apply MA only to the video stream: (i) SA supervises semantic token prediction on N sparsely sampled anchor frames, encouraging instruction-consistent semantic anchoring while sharing the same diffusion backbone (Sec. 3.2); (ii) MA trains the model to restore temporally perturbed source videos with motion-centric pretext supervision, improving temporal stability and robustness under fast motion (Sec. 3.3). The overall objective at Stage 0 follows Eq. (4),

L = LFM + λ · Lsem, (5) where LFM is the flow matching loss in Eq. (1) and Lsem is the SA semantic prediction loss.

- Stage 1: Supervised Fine-tuning (SFT). We then perform supervised fine-tuning on paired video editing datasets [31, 33, 32], while mixing a small portion of image editing data to preserve general instruction-following behavior [67, 68]. In this stage, the model is trained on standard instructionguided video editing triplets (source video, instruction, target video), and we keep SA enabled

- Table 1: Statistics of training data for each stage. ⋆ denotes we use specific text-to-video data for Motion Alignment by solving pretext transformations.

Training stage Dataset # Pairs Type

- Stage 0

Factorized Pre-training

NHR-Edit [67] 720,087 image editing GPT-Image-Edit [69] 1,015,170 image editing X2Edit [70] 768,470 image editing Koala-36M [65] 1,532,716 text-to-video⋆ MotionBench [66] 53,879 text-to-video⋆

- Stage 1

NHR-Edit [67] 720,087 image editing Pico-Banana-400K [68] 257,730 image editing Ditto-1M [31] 3,936 video editing OpenVE-3M [33] 818,232 video editing ReCo-Data [32] 206,596 video editing

Supervised Fine-tuning

to maintain stable semantic anchoring on sparse anchor frames. Compared with Stage 0, Stage 1 focuses on aligning generation with paired editing supervision, improving edit fidelity and mitigating remaining semantic–motion conflicts observed in challenging motions and fine-grained edits.

This two-stage design separates the learning of semantic anchoring and motion alignment from scarce paired video-edit data. As a result, Stage 0 already provides strong zero-shot video editing capability, and Stage 1 further improves edit fidelity and benchmark performance with paired supervision.

### 4 Experiments

##### 4.1 Experimental Settings

Training data. As summarized in Tab. 1, we use NHR-Edit [67], GPT-image-edit [69], X2Edit[70], and Pico-Banana-400K [68] for image editing training. We additionally incorporate text-to-video Koala-36M [65] and MotionBench [66] for pretext motion alignment. Ditto-1M [31], OpenVE-

- 3M [33], and ReCo-Data [32] are employed for video editing. All datasets are additionally subjected to a VLM-based coarse filtering stage to remove low-quality or instruction-inconsistent samples. The detailed filtering criteria are provided in Appendix. Specifically, we only use the Style subset of Ditto1M [31], and the Local Change, Background, Style, and Subtitles categories from OpenVE-3M [33].

Implementation details. During training, we conduct two-stage training on mixed image and video data. The learning rate is 2 × 10−5 for both stages. The global batch size is 448 for images and 112 for videos, and we train at a resolution of 480p. We support multiple aspect ratios, including 1/2,2/3,3/4, and 1/1, as well as their reciprocals. We maintain an exponential moving average (EMA [71]) of model parameters with decay 0.9998 and update it every iteration. The loss weight λ (Eq. 4) is set to 0.1. Unless otherwise specified, we uniformly sample N sparse anchor frames for Semantic Anchoring (Sec. 3.2); for efficiency, we set N = 1 in all experiments. We use M local semantic tokens per anchor frame (plus one global token), and fix M = 64 throughout.

In the text-to-video data, we use no pretext task as well as three pretext tasks—Cube Inpainting, Speed Perturbation, and Tube Shuffle—with a sampling ratio of 1:2:3:4 (no-pretext : cube inpainting : speed perturbation : tube shuffle). Task-specific settings are deferred to Appendix.

Evaluation details. To evaluate SAMA, we compare it against current state-of-the-art methods, including closed-source and open-source systems. For closed-source models, we include Kling1.6 [72], Kling-Omni [15], Runway [23], MiniMax [73], and Pika [74]. For open-source methods, we compare with InsV2V [38], DiffuEraser [35], VACE [36], InsViE [30], Omni-Video [42], LucyEdit [37], UniVideo [43], InstructX [14], ICVE [39], Ditto [31], OpenVE-Edit [33], VINO [44], and ReCo [32]. We conduct experiments on three benchmarks: VIE-Bench [14], OpenVE-Bench [33], and ReCoBench [32]. We use different VLM judges for scoring across benchmarks: GPT-4o [75] for VIEBench [14], Gemini-2.5-Pro [76] for OpenVE-Bench [33], and Gemini-2.5-Flash-Thinking [77] for ReCo-Bench [32].

- Table 2: Comparison results on VIE-Bench. The best results are shown in bold. Gray shading indicates closed-source models.

Method

|Instruct follow<br><br>Preservation<br><br>Quality Avg.<br><br>|Instruct follow<br><br>Preservation<br><br>Quality Avg.|
|---|---|
|Add<br><br>|Swap / Change|

Kling1.6 6.000 8.230 5.576 6.602 9.000 9.060 8.333 8.800 Kling-Omni 9.333 9.589 8.622 9.181 9.495 9.448 8.638 9.194 Runway 8.607 8.913 7.823 8.447 9.580 8.628 9.275 9.161 Pika - - - - 7.542 7.847 6.837 7.408 InsV2V 3.552 5.891 3.402 4.281 5.304 6.428 4.971 5.567 VACE 3.938 6.696 3.929 4.854 6.171 7.552 6.199 6.640

- Omni-Video 5.699 6.135 6.294 6.242 4.733 4.856 4.656 4.748 UniVideo 8.567 9.422 7.978 8.656 8.886 8.962 8.200 8.683 InstructX 8.446 8.683 7.919 8.349 9.514 9.171 8.533 9.072

- SAMA 8.467 9.422 8.244 8.711 9.733 9.514 8.771 9.340 Remove Style / Tone Change

Kling1.6 8.440 8.800 7.520 8.253 - - - Kling-Omni 9.378 9.233 8.789 9.133 9.867 9.200 8.956 9.341 Runway 8.664 9.145 7.703 8.504 9.583 9.200 8.616 9.133 MiniMax 6.963 7.518 6.037 6.839 - - - DiffuEraser 6.346 6.807 5.576 6.243 - - - InsV2V 1.209 3.769 1.322 2.098 7.835 8.086 6.437 7.452 VACE 1.812 3.877 2.359 2.682 - - - Omni-Video 6.004 5.970 4.807 5.593 5.486 4.655 5.959 5.366 UniVideo 8.133 8.778 7.789 8.233 9.244 8.689 8.200 8.711 InstructX 8.627 8.668 7.672 8.322 9.650 9.099 8.839 9.196

- SAMA 9.533 9.189 8.711 9.144 9.644 9.356 8.778 9.259

- Table 3: Comparison results on OpenVE-Bench with Gemini 2.5 Pro. The best results are highlighted in bold. Gray shading indicates closed-source models.

Global Style

Background Change

Local Change

Local Remove

Local Add

Subtitle Edit

Creative Edit

Method # Params.

Runway - 3.72 2.62 4.18 4.16 2.78 3.62 3.64 VACE 14B 1.49 1.55 2.07 1.46 1.26 1.48 1.47 Omni-Video 1.3B 1.11 1.18 1.14 1.14 1.36 1.00 2.26 InsViE 2B 2.20 1.06 1.48 1.36 1.17 2.18 2.02 Lucy-Edit 5B 2.27 1.57 3.20 1.75 2.30 1.61 2.86 ICVE 13B 2.22 1.62 2.57 2.51 1.97 2.09 2.41 Ditto 14B 4.01 1.68 2.03 1.53 1.41 2.81 1.23 OpenVE-Edit 5B 3.16 2.36 2.98 1.85 2.15 2.91 2.31 UniVideo 20B 3.64 2.22 3.91 2.70 2.98 2.69 2.90 SAMA 14B 4.05 2.59 3.93 3.32 2.54 3.63 3.11

##### 4.2 Comparisons with State-of-the-Art Methods

Tab. 2 show that our method consistently outperforms existing open-source video-editing models across most metrics, while remaining competitive with state-of-the-art closed-source systems. In particular, SAMA achieves the best overall performance on Swap/Change and Remove, among all compared methods. Similar gains are observed on Tab. 3 on OpenVE-Bench [33] and Tab.4 on ReCo-Bench [32], where SAMA attains the top overall score and delivers strong results across multiple task categories, despite a few metrics where it is not the best-performing method.

Qualitative Comparisons. In the qualitative comparisons on VIE-Bench and ReCo-Bench (see Fig. 4), SAMA demonstrates stronger instruction adherence and temporal consistency across diverse

- Table 4: Comparison results on ReCo-Bench with Gemini-2.5-Flash-Thinking. The best results are shown in bold. Abbreviations: SA, semantic accuracy; SP, scope precision; CP, content preservation; AN, appearance naturalness; SN, scale naturalness; MN, motion naturalness; VF, visual fidelity;

TS, temporal stability; ES, edit stability. SEA/SV N/SV Q are category scores and S is the overall score.

Edit Accuracy Video Naturalness Video Quality Average Score SA SP CP AN SN MN VF TS ES SEA SV N SV Q S

Task Method

InsViE 2.60 2.79 2.78 2.33 3.98 3.74 3.71 3.91 3.58 2.60 3.10 3.46 3.05 Lucy-Edit 6.27 6.32 7.75 4.63 7.08 6.08 6.31 6.82 7.57 6.47 5.70 6.77 6.31

Ditto 7.46 7.24 6.30 6.30 8.85 8.30 8.13 8.55 9.03 6.70 7.57 8.41 7.56 UniVideo 9.39 9.27 9.69 7.27 9.23 8.80 8.44 8.89 9.75 9.40 8.31 8.99 8.90

Add

ReCo 8.65 8.40 9.22 6.39 8.78 8.28 8.02 8.61 9.61 8.54 7.55 8.61 8.23 SAMA 9.51 9.26 9.83 7.44 9.50 8.87 8.78 9.03 9.76 9.43 8.33 9.00 8.92

InsViE 1.89 2.38 2.48 2.58 5.25 5.05 3.76 4.00 3.52 2.10 3.91 3.49 3.17 Lucy-Edit 6.57 7.49 7.73 5.13 7.46 6.65 6.32 6.64 8.08 7.08 6.21 6.88 6.72

Ditto 4.95 4.83 4.79 5.81 8.63 8.10 7.55 7.95 8.71 4.56 7.21 7.96 6.58 UniVideo 9.03 9.68 9.73 7.73 9.30 8.92 8.57 8.91 9.80 9.40 8.39 8.90 8.90

Replace

ReCo 9.38 9.43 9.59 7.07 8.87 8.47 8.19 8.65 9.67 9.43 8.01 8.77 8.74 SAMA 9.58 9.82 9.82 7.77 9.35 8.98 8.55 8.80 9.72 9.71 8.60 8.98 9.10

InsViE 2.53 2.49 2.44 2.63 4.87 4.72 3.41 3.67 3.40 2.44 3.76 3.29 3.16 VACE 4.58 4.58 4.56 4.96 6.09 5.89 5.48 5.50 5.57 4.57 5.43 5.56 5.19

UniVideo 7.37 7.43 7.28 6.06 7.61 7.13 6.28 6.43 7.72 7.33 6.59 6.51 6.81 ReCo 7.43 7.43 7.17 6.20 7.43 7.30 6.48 6.63 7.68 7.28 6.90 6.82 7.00 SAMA 8.76 8.71 8.43 7.16 8.73 8.42 7.31 7.52 8.73 8.61 7.94 7.73 8.09

Remove

InsViE 7.59 8.86 8.49 6.77 9.14 9.28 7.13 6.40 8.99 8.17 8.21 7.35 7.91 Lucy-Edit 3.73 5.59 5.39 4.20 5.88 5.88 4.44 4.17 5.87 4.65 4.67 5.17 4.83

Ditto 9.10 9.36 9.26 8.25 9.51 9.58 8.33 8.33 9.77 9.20 9.07 8.77 9.01 UniVideo 8.10 9.82 9.50 8.56 9.65 9.84 8.91 8.57 9.88 8.95 9.23 9.00 9.06

Style

ReCo 9.11 9.82 9.54 8.43 9.55 9.70 8.61 8.35 9.87 9.42 9.19 8.90 9.17 SAMA 8.46 9.95 9.64 8.79 9.77 9.77 8.88 8.59 9.83 9.24 9.42 9.07 9.25

editing types. SAMA follows fine-grained instructions more reliably, correctly handling relative position cues (e.g., “on the left”) and attribute constraints (e.g., alternating light and dark hair). It also completes replacements (e.g., pigeon→squirrel, seal→crab) with consistent appearance over time. Motion-wise, SAMA better preserves temporal alignment (e.g., keeping the stroller aligned after removal) and maintains identity/details during stylization, while other methods may drift or blur. Overall, SAMA better grounds instruction semantics while maintaining coherent motion, leading to higher-quality and more stable edits. More qualitative results are provided in the Appendix.

##### 4.3 Zero-shot Video Editing

We evaluate SAMA in a zero-shot setting, where the model is trained w/o any video editing data and is directly prompted with editing instructions during inference. As show in Fig. 5, SAMA demonstrates strong zero-shot editing capabilities across Replace/Add/Remove/Style/Hybrid tasks, producing consistent edits over multiple frames while largely preserving non-edited content. Despite these encouraging results, we also observe several typical failure modes in the zero-shot setting: (i) attribute edits can be temporally inconsistent, e.g., the edited colors may vary across frames; (ii) newly added objects may appear slightly blurry; (iii) removal edits may leave residual ghosting.

##### 4.4 Ablation Study

Semantic Anchoring. We first observe that incorporating the semantic prediction objective accelerates the decrease of the diffusion loss, leading to faster DiT convergence. In addition, SA stabilizes training, as evidenced by a noticeably reduced loss variance (see Fig. 6b). We set the baseline by concatenating the source latent with the video latent, without SA or MA. We use the smaller

Source Video SAMA Kling UniVideo Source Video SAMA Kling UniVideo

[Figure 130]

[Figure 131]

VIE-Bench

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

Replace the blackswan with a white cat.

Add a woman with alternating light and dark hair sitting in the box.

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

Remove the woman. Remove the boy.

Source Video SAMA Ditto UniVideo Source Video SAMA Ditto UniVideo

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

ReCo-Bench

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

Add a golden crown hat on the head of the chipmunk on the left. Remove the woman on the left.

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

Replace the pigeon on the right with a squirrel. Replace the spotted baby seal on the sand with a red crab.

Figure 4: Qualitative comparisons with prior methods on VIE-Bench and ReCo-Bench.

Wan2.2-T2V-5B [78] for efficiency with type embeddings and train it on a subset of the Ditto-1M [31], obtaining the baseline results. Building on this baseline, adding SA leads to consistent mean score improvements across all tasks on VIE-Bench.

We further provide qualitative comparisons under the same number of training steps in Fig. 6a. As shown, the model with SA produces higher-quality edits at earlier training stages, whereas the baseline without it often yields incomplete or less accurate modifications. These results corroborate that SA facilitates faster convergence in practice.

Motion Alignment. We conduct a qualitative analysis on the effect of MA. We find that enabling MA improves temporal consistency under fast motion and alleviates motion blur. Representative qualitative results are shown in Fig. 7.

Table 5: Ablations of SAMA modules.

In the tennis case with large camera motion, with MA noticeably improves background sharpness (e.g., clearer on-screen text), while the baseline appears blurred. Similar improvements are observed in the car and the third example, where the baseline often loses background motion. Quantitative ablation results on MA are summarized in Tab. 6. On VIE-Bench,

instruct follow

preservation

quality Overall

method

baseline 6.575 6.261 6.100 6.312 w/ SA 7.002 6.744 6.342 6.696 w/ MA 6.969 6.620 6.544 6.711 SAMA 7.402 6.998 6.884 7.095

###### Source Video Stage0 Stage1

Source Video Stage0 Stage1

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

Add a A little girl with long brown hair beside the boy.

Change the color of the boat to yellow.

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

Remove the dancing man. Transform into illustration style.

[Figure 216]

Figure 5: Zero-shot qualitative results on VIE-Bench at two training stages.

Training Loss

###### Source video w/o SA w/ SA

[Figure 217]

w/o SA w/ SA

Replace the person with a white robot.

[Figure 218]

Loss

Change the football to a fireball.

[Figure 219]

Replace the man into a gorilla.

[Figure 220]

Replace the Goat with a leopard.

Step

(a) Visual results of SAMA w/ SA (right column) and w/o SA (middle column).

(b) Training loss curves.

##### Figure 6: Ablations for Semantic Anchoring (SA).

adding MA alone improves the overall score by 0.399 over the baseline. When combining SA and MA, the overall score further increases by 0.783, indicating that the two components are complementary.

SAMA w/o MA

SAMA w/ MA

SAMA w/o MA

SAMA w/ MA

Source Video

Source Video

SAMA w/o MA

SAMA w/ MA

Source Video

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

Replace the person into a woman with red dress.

Replace the car with a red convertible sports car.

Turn the goods box in the man's hand into treasure chest.

##### Figure 7: Qualitative comparison of SAMA w/ and w/o MA.

### 5 Conclusion

We presented SAMA, a factorized framework for instruction-guided video editing that separates semantic anchoring and motion alignment within a DiT. Semantic anchoring introduces an explict prior via semantic-token prediction at anchor frames, while motion alignment improves temporal coherence through motion-centric restoration pre-training on text-to-video data. Extensive experiments on VIE-Bench, OpenVE-Bench, and ReCo-Bench demonstrate state-of-the-art performance among open-source methods and competitive results against commercial systems. Moreover, SAMA exhibits strong zero-shot editing behavior, suggesting that robust instruction following can emerge from learning disentangled semantic and motion representations. Future work will focus on long-video editing, fast-motion scenarios, and stronger semantic tokenization to further reduce residual artifacts and temporal inconsistencies.

### References

- [1] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023.
- [2] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems, 36:31428–31449, 2023.
- [3] Haozhe Zhao, Xiaojian Shawn Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale. Advances in Neural Information Processing Systems, 37:3058–3093, 2024.
- [4] Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Anyedit: Mastering unified high-quality image editing for any idea. arXiv preprint arXiv:2411.15738, 2024.
- [5] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.
- [6] YuXin Song, Yu Lu, Haoyuan Sun, Huanjin Yao, Fanglong Liu, Yifan Sun, Haocheng Feng, Hang Zhou, and Jingdong Wang. Cologen: Progressive learning of concept-localization duality for unified image generation. arXiv preprint arXiv:2602.22150, 2026.
- [7] Yuying Ge, Sijie Zhao, Chen Li, Yixiao Ge, and Ying Shan. Seed-data-edit technical report: A hybrid dataset for instructional image editing. arXiv preprint arXiv:2405.04007, 2024.
- [8] Peng Wang, Yichun Shi, Xiaochen Lian, Zhonghua Zhai, Xin Xia, Xuefeng Xiao, Weilin Huang, and Jianchao Yang. Seededit 3.0: Fast and high-quality generative image editing. arXiv preprint arXiv:2506.05083, 2025.
- [9] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.
- [10] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 7623–7633, 2023.
- [11] Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2545–2555, 2025.
- [12] Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. Video-p2p: Video editing with cross-attention control. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8599–8608, 2024.
- [13] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15932–15942, 2023.
- [14] Chong Mou, Qichao Sun, Yanze Wu, Pengze Zhang, Xinghui Li, Fulong Ye, Songtao Zhao, and Qian He. Instructx: Towards unified visual editing with mllm guidance. arXiv preprint arXiv:2510.08485, 2025.
- [15] Kling Team, Jialu Chen, Yuanzheng Ci, Xiangyu Du, Zipeng Feng, Kun Gai, Sainan Guo, Feng Han, Jingbin He, Kang He, et al. Kling-omni technical report. arXiv preprint arXiv:2512.16776, 2025.

- [16] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, XIAOPENG ZHANG, Wangmeng Zuo, and Qi Tian. Controlvideo: Training-free controllable text-to-video generation. In The Twelfth International Conference on Learning Representations.
- [17] Weifeng Chen, Yatai Ji, Jie Wu, Hefeng Wu, Pan Xie, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. Control-a-video: Controllable text-to-video diffusion models with motion prior and reward feedback learning, 2024.
- [18] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.
- [19] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Leo Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Blog, 1(8):1, 2024.
- [20] Weijie Kong, Qi Tian, Zijian Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.
- [21] Niket Agarwal et al. Cosmos: World foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.
- [22] David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2018.
- [23] Runway. Runway gen-4. https://runwayml.com/, 2025.
- [24] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion features for consistent video editing. arXiv preprint arXiv:2307.10373, 2023.
- [25] Paul Couairon, Clément Rambour, Jean-Emmanuel Haugeard, and Nicolas Thome. Videdit: Zero-shot and spatially aware text-driven video editing. Transactions on Machine Learning Research, 2023.
- [26] Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. Rerender a video: Zero-shot text-guided video-to-video translation. In SIGGRAPH Asia 2023 Conference Papers, pages 1–11, 2023.
- [27] Yang Shen, Xiu-Shen Wei, Yifan Sun, Yuxin Song, Tao Yuan, Jian Jin, Heyang Xu, Yazhou Yao, and Errui Ding. Explanatory instructions: Towards unified vision tasks understanding and zero-shot generalization. arXiv preprint arXiv:2412.18525, 2024.
- [28] Yuren Cong, Mengmeng Xu, Christian Simon, Shoufa Chen, Jiawei Ren, Yanping Xie, JuanManuel Perez-Rua, Bodo Rosenhahn, Tao Xiang, and Sen He. Flatten: Optical flow-guided attention for consistent text-to-video editing. In Proceedings of the International Conference on Learning Representations (ICLR), 2024.
- [29] Bojia Zi, Penghui Ruan, Marco Chen, Xianbiao Qi, Shaozhe Hao, Shihao Zhao, Youze Huang, Bin Liang, Rong Xiao, and Kam-Fai Wong. Señorita-2m: A high-quality instruction-based dataset for general video editing by video specialists. arXiv preprint arXiv:2502.06734, 2025.
- [30] Yuhui Wu, Liyi Chen, Ruibin Li, Shihao Wang, Chenxi Xie, and Lei Zhang. Insvie-1m: Effective instruction-based video editing with elaborate dataset construction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16692–16701, 2025.
- [31] Qingyan Bai, Qiuyu Wang, Hao Ouyang, Yue Yu, Hanlin Wang, Wen Wang, Ka Leong Cheng, Shuailei Ma, Yanhong Zeng, Zichen Liu, et al. Scaling instruction-based video editing with a high-quality synthetic dataset. arXiv preprint arXiv:2510.15742, 2025.
- [32] Zhongwei Zhang, Fuchen Long, Wei Li, Zhaofan Qiu, Wu Liu, Ting Yao, and Tao Mei. Regionconstraint in-context generation for instructional video editing. arXiv preprint arXiv:2512.17650, 2025.
- [33] Haoyang He, Jie Wang, Jiangning Zhang, Zhucun Xue, Xingyuan Bu, Qiangpeng Yang, Shilei Wen, and Lei Xie. Openve-3m: A large-scale high-quality dataset for instruction-guided video editing. arXiv preprint arXiv:2512.07826, 2025.

- [34] Zixuan Ye, Xuanhua He, Quande Liu, Qiulin Wang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, Qifeng Chen, and Wenhan Luo. Unic: Unified in-context video editing. arXiv preprint arXiv:2506.04216, 2025.
- [35] Xiaowen Li, Haolan Xue, Peiran Ren, and Liefeng Bo. Diffueraser: A diffusion model for video inpainting. arXiv preprint arXiv:2501.10018, 2025.
- [36] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. arXiv preprint arXiv:2503.07598, 2025.
- [37] DecartAI Team. Lucy edit: Open-weight text-guided video editing. 2025.
- [38] Jiaxin Cheng, Tianjun Xiao, and Tong He. Consistent video-to-video transfer using synthetic dataset. arXiv preprint arXiv:2311.00213, 2023.
- [39] Xinyao Liao, Xianfang Zeng, Ziye Song, Zhoujie Fu, Gang Yu, and Guosheng Lin. Incontext learning with unpaired clips for instruction-based video editing. arXiv preprint arXiv:2510.14648, 2025.
- [40] Xuan Ju, Tianyu Wang, Yuqian Zhou, He Zhang, Qing Liu, Nanxuan Zhao, Zhifei Zhang, Yijun Li, Yuanhao Cai, Shaoteng Liu, et al. Editverse: Unifying image and video editing and generation with in-context learning. arXiv preprint arXiv:2509.20360, 2025.
- [41] Xiangpeng Yang, Ji Xie, Yiyuan Yang, Yan Huang, Min Xu, and Qiang Wu. Unified video editing with temporal reasoner. arXiv preprint arXiv:2512.07469, 2025.
- [42] Zhiyu Tan, Hao Yang, Luozheng Qin, Jia Gong, Mengping Yang, and Hao Li. Omni-video: Democratizing unified video understanding and generation. arXiv preprint arXiv:2507.06119, 2025.
- [43] Cong Wei, Quande Liu, Zixuan Ye, Qiulin Wang, Xintao Wang, Pengfei Wan, Kun Gai, and Wenhu Chen. Univideo: Unified understanding, generation, and editing for videos. arXiv preprint arXiv:2510.08377, 2025.
- [44] Junyi Chen, Tong He, Zhoujie Fu, Pengfei Wan, Kun Gai, and Weicai Ye. Vino: A unified visual generator with interleaved omnimodal context. arXiv preprint arXiv:2601.02358, 2026.
- [45] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024.
- [46] Xingjian Leng, Jaskirat Singh, Yunzhong Hou, Zhenchang Xing, Saining Xie, and Liang Zheng. Repa-e: Unlocking vae for end-to-end tuning of latent diffusion transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18262–18272, 2025.
- [47] Ziqiao Wang, Wangbo Zhao, Yuhao Zhou, Zekai Li, Zhiyuan Liang, Mingjia Shi, Xuanlei Zhao, Pengfei Zhou, Kaipeng Zhang, Zhangyang Wang, et al. Repa works until it doesn’t: Earlystopped, holistic alignment supercharges diffusion training. arXiv preprint arXiv:2505.16792, 2025.
- [48] Dengyang Jiang, Mengmeng Wang, Liuzhuozheng Li, Lei Zhang, Haoyu Wang, Wei Wei, Guang Dai, Yanning Zhang, and Jingdong Wang. No other representation component is needed: Diffusion transformers can provide representation guidance by themselves. arXiv preprint arXiv:2505.02831, 2025.
- [49] Jianhong Bai, Xiaoshi Wu, Xintao Wang, Xiao Fu, Yuanxing Zhang, Qinghe Wang, Xiaoyu Shi, Menghan Xia, Zuozhu Liu, Haoji Hu, et al. Semanticgen: Video generation in semantic space. arXiv preprint arXiv:2512.20619, 2025.
- [50] Xiangdong Zhang, Jiaqi Liao, Shaofeng Zhang, Fanqing Meng, Xiangpeng Wan, Junchi Yan, and Yu Cheng. Videorepa: Learning physics for video generation through relational alignment with foundation models. arXiv preprint arXiv:2505.23656, 2025.

- [51] Xijie Huang, Chengming Xu, Donghao Luo, Xiaobin Hu, Peng Tang, Xu Peng, Jiangning Zhang, Chengjie Wang, and Yanwei Fu. Ffp-300k: Scaling first-frame propagation for generalizable video editing. arXiv preprint arXiv:2601.01720, 2026.
- [52] Sagie Benaim, Ariel Ephrat, Oran Lang, Inbar Mosseri, William T Freeman, Michael Rubinstein, Michal Irani, and Tali Dekel. Speednet: Learning the speediness in videos. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9922–9931, 2020.
- [53] Yuan Yao, Chang Liu, Dezhao Luo, Yu Zhou, and Qixiang Ye. Video playback rate perception for self-supervised spatio-temporal representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6548–6557, 2020.
- [54] Jiangliu Wang, Jianbo Jiao, and Yun-Hui Liu. Self-supervised video representation learning by pace prediction. In European conference on computer vision, pages 504–521. Springer, 2020.
- [55] Dahun Kim, Donghyeon Cho, and In So Kweon. Self-supervised video representation learning with space-time cubic puzzles. In Proceedings of the AAAI conference on artificial intelligence, volume 33, pages 8545–8552, 2019.
- [56] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. Advances in neural information processing systems, 35:10078–10093, 2022.
- [57] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023.
- [58] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [59] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [60] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.
- [61] Yuxin Song, Wenkai Dong, Shizun Wang, Qi Zhang, Song Xue, Tao Yuan, Hu Yang, Haocheng Feng, Hang Zhou, Xinyan Xiao, et al. Query-kontext: An unified multimodal model for image generation and editing. arXiv preprint arXiv:2509.26641, 2025.
- [62] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.
- [63] Yuxin Song, Min Yang, Wenhao Wu, Dongliang He, Fu Li, and Jingdong Wang. It takes two: Masked appearance-motion modeling for self-supervised video transformer pre-training. arXiv preprint arXiv:2210.05234, 2022.
- [64] Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, and Yu Qiao. Videomae v2: Scaling video masked autoencoders with dual masking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14549–14560, 2023.
- [65] Qiuheng Wang, Yukai Shi, Jiarong Ou, Rui Chen, Ke Lin, Jiahao Wang, Boyuan Jiang, Haotian Yang, Mingwu Zheng, Xin Tao, et al. Koala-36m: A large-scale video dataset improving consistency between fine-grained conditions and video content. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8428–8437, 2025.
- [66] Wenyi Hong, Yean Cheng, Zhuoyi Yang, Weihan Wang, Lefan Wang, Xiaotao Gu, Shiyu Huang, Yuxiao Dong, and Jie Tang. Motionbench: Benchmarking and improving fine-grained video motion understanding for vision language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8450–8460, 2025.

- [67] Maksim Kuprashevich, Grigorii Alekseenko, Irina Tolstykh, Georgii Fedorov, Bulat Suleimanov, Vladimir Dokholyan, and Aleksandr Gordeev. Nohumansrequired: Autonomous high-quality image editing triplet mining. arXiv preprint arXiv:2507.14119, 2025.
- [68] Yusu Qian, Eli Bocek-Rivele, Liangchen Song, Jialing Tong, Yinfei Yang, Jiasen Lu, Wenze Hu, and Zhe Gan. Pico-banana-400k: A large-scale dataset for text-guided image editing. arXiv preprint arXiv:2510.19808, 2025.
- [69] Yuhan Wang, Siwei Yang, Bingchen Zhao, Letian Zhang, Qing Liu, Yuyin Zhou, and Cihang Xie. Gpt-image-edit-1.5 m: A million-scale, gpt-generated image dataset. arXiv preprint arXiv:2507.21033, 2025.
- [70] Jian Ma, Xujie Zhu, Zihao Pan, Qirong Peng, Xu Guo, Chen Chen, and Haonan Lu. X2edit: Revisiting arbitrary-instruction image editing through self-constructed data and task-aware representation learning. arXiv preprint arXiv:2508.07607, 2025.
- [71] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [72] Kuaishou. Kling1.6. https://app.klingai.com/, 2025.
- [73] Bojia Zi, Weixuan Peng, Xianbiao Qi, Jianan Wang, Shihao Zhao, Rong Xiao, and Kam-Fai Wong. Minimax-remover: Taming bad noise helps video object removal. arXiv preprint arXiv:2505.24873, 2025.
- [74] Pika: Idea-to-video platform (web product). https://pika.art/.
- [75] OpenAI. Hello gpt-4o. Blog post, May 2024.
- [76] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- [77] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [78] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [79] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025.

### A Discussion on Type Embeddings vs. Shifted RoPE

To distinguish token roles in our unified formulation, we add a learned type embedding to each token (source-video latents, target-video latents, and semantic tokens), and apply this design throughout all training stages. We adopt type embeddings because they provide an explicit yet lightweight way to encode token identity without altering the backbone’s positional encoding, and introduce a smaller perturbation to the pretrained prior than shifted RoPE. Empirically, type embeddings yield faster and more stable convergence than shifted RoPE, likely because they decouple token role from token position: positional encoding continues to capture spatiotemporal structure, while token identity is modeled separately. Additional evidence is provided in Fig. 8 and Tab. 6. Under the Wan2.2-T2V-5B [78] LoRA setting, training on the Ditto-1M replace subset [31] and evaluating on VIE-Bench replace [14], type embeddings converge faster and better preserve background content.

### B VLM-based Data Filtering Details

This appendix provides additional details on the training data used in our pipeline. For data filtering, we use Qwen2.5-VL-72B [79] as a VLM judge to score each sample from 1-10 via three inference turns, and average the scores. Scores are on a 1–10 scale. The detailed prompts are as follows. The judge assigns four scores: Instruction Following, Visual Quality, Content Preservation, and Motion Consistency (only for videos). We then filter samples with the following thresholds. For images, we use samples with Instruction Following ≥ 9, Visual Quality ≥ 9, and Content Preservation ≥ 9. For videos, we use samples with Instruction Following ≥ 8, Visual Quality ≥ 9, Content Preservation ≥ 8, and Motion Consistency ≥ 8.

Table 6: Ablations of Type Embeddings (TE) modules.

instruct follow

preservation

quality Overall

method

w/ PE 6.705 7.533 6.686 6.975 w/o PE 6.619 6.257 6.619 6.498

Prompt for VLM-based Video Editing Evaluation: Instruction Following and Visual Quality

You are an expert video editing evaluator. Please evaluate the editing quality. # User’s Editing Instruction: {instruction} # Videos to Compare:

- 1. First video: Original video (before editing)
- 2. Second video: Edited video (after editing) # Evaluation Criteria (Rate 1-10):

- 1. Instruction Following: How well does the edited video implement the user’s specific instruction?

- - 10: Perfectly follows all aspects
- - 7-9: Mostly follows with minor issues
- - 4-6: Partially follows, some aspects missing
- - 1-3: Does not follow the instruction

- 2. Visual Quality: Is the edited video visually coherent and natural-looking?

- - 10: Perfect quality, seamless editing
- - 7-9: High quality, minor artifacts
- - 4-6: Moderate quality, noticeable issues
- - 1-3: Poor quality, obvious problems

# Output Format: Return ONLY a JSON object with this exact structure: {

"instruction_following_score": <number between 1 and 10>, "visual_quality_score": <number between 1 and 10>, "explanation": "<brief explanation for the scores>"

} Important: Only output the JSON object, nothing else.

Prompt for VLM-based Video Editing Evaluation: Motion Consistency

You are an expert video editing evaluator. Please evaluate the editing quality. # User’s Editing Instruction: {instruction} # Videos to Compare:

- 1. First video: Edited video (after editing)
- 2. Second video: Original video (before editing) # Evaluation Criteria (Rate 1-10):

1. **Motion Consistency**: Does the edited video preserve the original motion of both the main object and the background?

Note: Any style, object, appearance changes are acceptable. You should only care about motion inconsistency which NOT caused by style, object, appearance change.

- - 10: Perfectly preserves all original motions
- - 7-9: Mostly preserves, minor motion deviations
- - 4-6: Noticeable motion inconsistencies, or noticeable motion delays
- - 4-6: Unnatural texture in the background of the edited video
- - 1-3: Most of motion mismatches
- - 1-3: In edited video, the first frame differs significantly from subsequent frames

# Output Format: Return ONLY a JSON object with this exact structure: {{

"motion_consistency_score": <number between 1 and 10>, "explanation": "<brief explanation for the scores>"

}} Important: Only output the JSON object, nothing else.

Prompt for VLM-based Video Editing Evaluation: Content Preservation

You are an expert video editing evaluator. Please evaluate the editing quality. Specifically, please evaluate the Content Preservation of the edited video only.

# User’s Editing Instruction: {instruction}

# Videos to Compare:

- 1. First video: Original video (before editing)
- 2. Second video: Edited video (after editing)

# Evaluation Criteria (Rate 1-10):

1. **Content Preservation**: Does the edit preserve all unrelated content from the original video?

- - 10: Perfectly preserves all unrelated content
- - 7-9: Mostly preserves, minor changes
- - 4-6: Some unrelated changes
- - 1-3: Changes much unrelated content

# Output Format: Return ONLY a JSON object with this exact structure: {{

"content_preservation_score": <number between 1 and 10>, "explanation": "<brief explanation for the scores>"

}}

###### Important: Only output the JSON object, nothing else.

Source Video W/ TE W/O TE Source Video W/ TE W/O TE

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

Replace the brown bear with a panda. Replace the car with a red convertible sports car.

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

Replace the cows with a wolf. Replace the person into a woman with red dress.

[Figure 263]

[Figure 264]

Figure 8: Illustration of Type Embedding (TE).

### C Task-Specific Settings for Text-to-Video Pretext Tasks

As mentioned in the main text, for text-to-video data, we use no pretext task together with three pretext tasks: Cube Inpainting, Speed Perturbation, and Tube Shuffle.

For Cube Inpainting, we use a masking ratio of 30%. For Speed Perturbation, we apply a 2× temporal acceleration. For Tube Shuffle, we divide each video into 2×2×2 spatiotemporal tubes and randomly shuffle them.

### D Pretext Prediction Visualization

We visualize the model predictions for the three motion-centric pretext tasks used in Motion Alignment, including Cube Inpainting, Speed Perturbation, and Tube Shuffle. As shown in Fig. 9, the model is able to (i) plausibly complete masked spatio-temporal regions, (ii) recover more natural motion dynamics from temporally perturbed inputs, and (iii) restore coherent spatio-temporal structure after tube permutation. These qualitative results indicate that the pretext objectives encourage the backbone to internalize motion cues and temporal reasoning, which benefits subsequent instruction-guided video editing.

### E More qualitative results

In this section, we present additional qualitative comparisons with other methods, showing that our method produces more consistent and visually appealing editing results across a wide range of scenarios. More detailed visual comparisons are provided in Figs. 10–12. The left column highlights examples where our method excels at semantic understanding and instruction grounding, while the right column presents cases emphasizing improved motion consistency and temporal alignment.

Source Video Input Video Generate Video Source Video Input Video Generate Video

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

[Complete the missing regions in the video.] In the scene, there are four people all wearing something similar to VR glasses and….

[Complete the missing regions in the video.] In the center of the screen, a man facing the camera with his head…

[Figure 285]

[Figure 286]

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

[Restore the correct spatio-temporal order of the video segments.] The scene is set in a classroom, with four people standing in the center of the …

[Restore the correct spatio-temporal order of the video segments.] A brown bear is walking through a rocky area.…

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

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

[Restore the video to normal playback speed while maintaining temporal coherence.] In the scene is a young white woman wearing a black top…

[Restore the video to normal playback speed while maintaining temporal coherence.] A male is holding a mobile phone in his left hand…

[Figure 329]

[Figure 330]

##### Figure 9: Illustration of Pretext Prediction.

Source Videos SAMA Ditto UniVideo Source Videos SAMA Ditto UniVideo

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

[Figure 353]

[Figure 354]

Remove the dogs.

Convert the grayish black car into an armored tank

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

Change the color of the man's T-shirt to purple.

Remove the woman on the left.

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

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

Transform the entire scene into pixel art style. Replace the man in the right into asunglasses.woman with long hair, wearing black shirt and

Source Videos SAMA Kling UniVideo Source Videos SAMA Kling UniVideo

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

Remove the bowl in the middle. Make it snowy.

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

Replace the blackswan with a white cat. Turn into watercolor style.

##### Figure 10: More qualitative results on VIE-Bench.

###### Source Video SAMA Ditto UniVideo Source Video SAMA Ditto UniVideo

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

Replace the blue sports car with an electric green eco-friendly car, ensuring it maintains the same position and pose within the video scene.

Replace the woman's yellow dress with an elegant navy blue evening gown, maintaining the same pose and position within the video scene.

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

Remove the person in a maroon sweater and glasses engaged in focused activity from the entire video sequence. The background must be reconstructed with temporal consistency, and all other video content must remain unchanged.

[Figure 468]

[Figure 469]

[Figure 470]

Replace the man's light gray hoodie with a sharp dark navy blue business suit, white shirt, and black tie, maintaining the same position and pose within the scene.

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

Remove the young woman with long, wavy brown hair and a serene expression from the entire video sequence. She is wearing a textured, off-white knit sweater with wide, ruffled sleeves, gazing upwards with her lips slightly parted and eyes softly closed, her head tilting slightly to the side while maintaining a relaxed posture with shoulders subtly leaning back. The background must be reconstructed with temporal consistency, and all other video content must remain unchanged.

[Figure 498]

Replace the background with a dynamic desert highway scene. Heat waves should shimmer above the asphalt, dust occasionally drifts across the road, and distant birds fly across the sky. The lighting should be bright and natural, with subtle shadows moving as if from a passing cloud. The man and SUV remain perfectly still.

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

Remove the majestic white horse adorned with a harness, moving gracefully along a cobblestone path, from the entire video sequence. The background must be reconstructed with temporal consistency, and all other video content must remain unchanged.

Completely remove a light blue cowboy hat with a wide, slightly curved brim and a smooth, rounded crown from the entire video, using temporally consistent background inpainting and ensuring all other video content remains unchanged.

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

"Replace the subtitles What’s locked in these frames could rewrite her story."" with ""Her fingers trace fragile film—soft light, a life unspooling."" at the bottom of the video with white text without border style."""

[Figure 554]

Remove the subtitles at the top of the video.

##### Figure 11: More qualitative results on OpenVE-Bench.

Source Videos SAMA Ditto UniVideo Source Videos SAMA Ditto UniVideo

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

Add a small flower wreath on top of the the seal's head. Add a tiny, red fez hat on the monkey's head.

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

Remove the wolf-like dog on the right. Replace the bulldog on the grass with a pot-bellied pig.

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

Convert the video into a Sketch style.

Remove the man in the green hoodie.

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

Remove the young boy standing at the chalkboard.

Remove the woman on the left.

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

Replace the man in the green hoodie with a fluffy St. Bernard dog sitting in the chair, paws on the desk.

Add a gorilla on the left, standing and trying to copy the dance moves.

[Figure 693]

[Figure 694]

##### Figure 12: More qualitative results on ReCo-Bench.

