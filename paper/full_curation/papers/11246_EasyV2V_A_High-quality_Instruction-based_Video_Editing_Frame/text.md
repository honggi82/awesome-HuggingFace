## EasyV2V: A High-quality Instruction-based Video Editing Framework

# arXiv:2512.16920v1[cs.CV]18Dec2025

Jinjie Mai1,2∗ Chaoyang Wang2 Guocheng Gordon Qian2 Willi Menapace2 Sergey Tulyakov2 Bernard Ghanem1 Peter Wonka2,1 Ashkan Mirzaei2∗†

1 KAUST 2 Snap Inc. ∗ Core contributors † Project lead

Project Page: https://snap-research.github.io/easyv2v/

w/ Text Instruction + Mask Instruction

Input Mask

EasyV2V

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

InputEasyV2V

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

(via Image Editing)

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Make them smile, shake hands, and kiss Turn her into a robot

+ Temporal Control + Reference

Edit Here! Input Reference Image EasyV2V

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

InputEasyV2V

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

A random frame transformed via image editing

The lamp smoothly turns on lighting up the room.

Make it into an anime style and change the sign to say “EasyV2V”

Figure 1. EasyV2V unifies data processing, architecture, and control for high-quality, instruction-based video editing with flexible inputs (text, masks, edit timing). The figure illustrates its versatility across diverse input types and editing tasks.

### Abstract

While image editing has advanced rapidly, video editing remains less explored, facing challenges in consistency, control, and generalization. We study the design space of data, architecture, and control, and introduce EasyV2V, a simple and effective framework for instruction-based video editing. On the data side, we compose existing experts with fast inverses to build diverse video pairs, lift image edit pairs into videos via single-frame supervision and pseudo pairs with shared affine motion, mine dense-captioned clips for video pairs, and add transition supervision to teach how edits unfold. On the model side, we observe that pretrained text-to-video models possess editing capability, motivating a simplified design. Simple sequence concatenation for conditioning with light LoRA fine-tuning suffices to train a strong model. For control, we unify spatiotemporal control via a single mask mechanism and support optional reference images. Overall, EasyV2V works with flexible inputs, e.g., video+text, video+mask+text, video+mask+reference+text, and achieves state-of-the-art

video editing results, surpassing concurrent and commercial systems.

### 1. Introduction

What makes a good instruction-based video editor? We argue that three components govern performance: data, architecture, and control. This paper analyzes the design space of these components and distills a recipe that works great in practice. The result is a lightweight model that reaches state-of-the-art quality while accepting flexible inputs.

Training-free video editors adapt pretrained generators but are fragile and slow [10, 26]. Training-based approaches improve stability, yet many target narrow tasks such as ControlNet-style conditioning [1, 20], video inpainting [72], or reenactment [8]. General instruction-based video editors handle a wider range of edits [9, 53, 64, 73], yet still lag behind image-based counterparts in visual fidelity and control. We set out to narrow this gap. Figure 2 motivates our design philosophy: modern video models already know how to transform videos. To unlock this emerging capability with minimal adaptation, we conduct a comprehensive investigation into data curation, architectural de-

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

A man playing the piano gradually transforms into a panda

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

A woman wearing a red tshirt that slowly turns blue

- Figure 2. A pretrained text-to-video model can mimic common editing effects without finetuning. This suggests that much of the “how” of video editing already lives inside modern backbones.

sign, and instruction-control.

Data. We see three data strategies. (A) One generalist model renders all edits and is used to self-train an editor [9, 53, 64]. This essentially requires a single teacher model that already solves the problem in high quality. (B) Design and train new experts for specific edit types, then synthesize pairs at scale [73]. This yields higher per-task fidelity and better coverage of hard skills, but training and maintaining many specialists is expensive and slows iteration and adaptation to future base models. We propose a new strategy: (C) Select existing experts and compose them. We focus on experts with a fast inverse (e.g., edge↔video, depth↔video) and compose more complex experts from them. This makes supervision easy to obtain, and experts are readily available as off-the-shelf models. It keeps costs low and diversity high by standing on strong off-the-shelf modules; the drawback is heterogeneous artifacts across experts, which we mitigate through filtering and by favoring experts with reliable inverses. Concretely, we combine offthe-shelf video/image experts for stylization, local edits, insertion/removal, and human animation; we also convert image edit pairs into supervision via two routes: single-frame training and pseudo video-to-video (V2V) pairs created by applying the same smooth camera transforms to source and edited images. Lifting image-to-image (I2I) data increases scale and instruction variety; its limitation is weak motion supervision, and shared camera trajectories reintroduce temporal structure without changing semantics. We further leverage video continuation: we derive V2V pairs from densely captioned text-to-video (T2V) datasets by sampling input clips outside each captioned interval and target clips within it, while converting the caption into an instruction using an LLM. Continuation data teaches action and transition edits that are scarce in typical V2V corpora, at the cost of more careful caption-to-instruction normalization. We collect and create an extensive set of V2V pairs, the most comprehensive among published works, and conduct a detailed study of how editing capability emerges from each training source.

Architecture. Using a pretrained video backbone, we study two strategies to inject the source video: channel con-

catenation and sequence concatenation. Channel concatenation is faster in practice because it uses fewer tokens, but we show that sequence concatenation consistently yields higher edit quality. The trade-off is efficiency versus separation: channel concatenation keeps context short but entangles source and target signals; sequence concatenation costs more tokens but preserves clean roles for each stream, improving instruction following and local detail. Our final design adds small, zero-init patch-embedding routes for the source video and an edit mask used for spatiotemporal control of the edit extent, reuses the frozen video VAE for all modalities, and fine-tunes only with LoRA [16]. Full finetuning can help when massive, heterogeneous data are available, but it risks catastrophic forgetting and is costly to scale. With < 10M paired videos, LoRA transfers faster, reduces overfitting, and preserves pretrained knowledge while supporting later backbone swaps; its main downside is a slight headroom loss when unlimited data and compute are present. We inject the mask by token addition, and concatenate tokens from the source and optional reference image along the sequence. This preserves pretraining benefits, keeps token budgets tight (by not introducing new tokens for the mask), and makes the model easily portable to future backbones. Addition for masks is simple and fast; while it carries less capacity than a dedicated mask-token stream, we find it sufficient for precise region and schedule control without context bloat. We use an optional reference frame at training and test time to leverage strong image editors when available. References boost specificity and style adherence when present; randomly dropping the reference during training keeps the model robust when they are absent or noisy.

Flexible control. Prior work explores control by skeletons, segmentation, depth, and masks. A key signal is still missing: when the edit happens. Users often want an edit to appear gradually (e.g., “set the house on fire starting at 1.5s, then let the flames grow gradually”). We unify spatial and temporal control with a single mask video. Pixels mark where to edit; frames mark when to edit, and how the effect evolves. Alternatives include keyframe prompts or token schedules, which are flexible but harder to author and to align with motion. A single mask video is direct, differentiable, and composes well with text and optional references. The cost is requiring a mask sequence, which we keep lightweight and editable.

Combining these design choices results in a unified editor, EasyV2V, which supports flexible input combinations, including video + text, video + mask + text, and video + mask + reference + text (see Fig. 1). Despite its simplicity, our recipe consistently improves edit quality, motion, and instruction following over recently published methods. In summary, our contributions are:

- • A clarified design space for instruction-based video editing and a consistent strategy across data, architecture, and control that achieves state-of-the-art results.
- • A reusable data engine built from composable experts with trivial inverses, lifted high-quality image edits, and video continuation, with per-expert/per-source ablations.
- • A lightweight architecture that minimally modifies a pretrained video backbone: zero-init patch-embeddings for source and mask, frozen VAE reuse, and LoRA finetuning, plus optional reference frames.
- • Temporal control as a first-class signal, unified with spatial control via a single mask video that schedules when edits start and how they evolve.

### 2. Related work

Instruction-based visual editing datasets. With the success of image and video generative models trained on largescale text-to-image and text-to-video datasets, recent work has focused on developing datasets for instruction-based image and video editing. Early approaches attempted to build large-scale paired I2I datasets, though with low success rates, requiring extensive automatic [4, 18] or manual [65] filtering. Later efforts improved success rates by leveraging task-specific models [13, 52, 57, 63, 70], leading to highly capable image editing systems [29, 47]. These models subsequently enabled the generation of higherquality paired I2I datasets [51]. Creating paired V2V datasets is inherently more challenging, as it requires editing multiple frames while maintaining temporal coherence and faithfully applying the intended modification. Early work [9] used LLMs and Prompt-to-Prompt [14] to synthesize video pairs, but results were limited by artifacts from the underlying editing method. Another dataset [17] provided video–object-mask pairs with corresponding captions but omitted edited videos. The current trend is to generate synthetic datasets using one previous general video editing model [9, 53, 64] or a collection of more specialized video editing models [73]. These efforts still lag behind image editing datasets in quality and diversity. We discuss strategies to mitigate this gap and provide extensive analysis comparing the effects of different approaches.

Instruction-based visual editing. Building on the success of diffusion models for image [12, 19, 28, 39, 41, 47] and video [5, 25, 45, 48, 58] generation, early visual editing methods used pretrained diffusion models in a training-free manner for image editing [7, 14, 21, 27, 30, 33, 49, 62], typically manipulating attention maps and latent spaces, or by noise inversion. These approaches generally produce low-quality outputs, have slow inference, and exhibit low success rates. Subsequent work showed that even relatively low-quality paired datasets can outperform training-free methods [4, 42]. Such models concatenate input image latents with noisy latents along the channel

dimension and fine-tune a pretrained image generator on synthetic paired I2I data produced by training-free approaches. Another line of research performs concatenation along the sequence dimension, improving quality at the cost of efficiency [6, 23, 55, 69]. Similar directions adopt LLM-style architectures for unified or token-based editing [11, 71]. Instruction-based video editing has been explored far less. Early attempts include training-free video editing pipelines [10, 26]. Some works adapt image generative models [34, 43], while others train on low-quality synthetic video pairs and are limited by training data quality [9, 14, 64, 68]. Additional methods include propagating edits from the first frame to subsequent frames [31] or designing architectures with task-specific conditioning for multiple editing tasks [61]. Another line of work explores task-specific editing, such as ControlNet-style [66] video generation [20] or pose/face conditioned human video synthesis [8]. Two concurrent frameworks recently appeared on arXiv, focusing on patch-wise concatenation [44] and LLM-style architectures [22]. In contrast, our method systematically studies different data sources for training V2V models, identifies those most critical for performance, and enables more controllable video generation through optional reference images and spatiotemporal edit control.

### 3. Method

Overview. We build on a modern video generative backbone [48] and introduce lightweight conditioning modules for video editing. A pivotal design choice is how to inject the source video condition. While one simple approach is to concatenate the source video latent and the noisy latent along the channel dimension [44], requiring only a modification of the initial patch embedding layer, we found this method struggles to learn edits efficiently. We therefore adopt a sequence-wise concatenation strategy, appending the source video tokens to the noisy latent token sequence. This approach provides a more robust and effective conditioning mechanism, as validated in our ablations (Tab. 3).

Concretely, we add separate patch-embedding layers for each control signal (source video and mask) and support an optional reference image at training and inference. After the encoding of inputs and patch-embedding, condition signals are injected either by element-wise addition (for the edit mask) or by sequence concatenation (for source video and optional reference image). To preserve the generative priors of the T2V backbone and ensure stable training, we adopt a parameter-efficient finetuning strategy. We freeze the original backbone weights and optimize only the newly introduced patch-embedding layers and low-rank adaptation (LoRA) weights added to the DiT’s attention layers. We found this approach essential, as full-model finetuning easily led to training instability and source video inconsis-

Latents

V2V w/ Transition

V2V

Trainable

[Figure 41]

InputVideoMaskTargetVideoReference

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Optional

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Embed.

Frozen

[Figure 51]

[Figure 52]

[Figure 53]

Patch

InputTargetMask

InputTargetMask

DiT

+Seq.Concat.

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

Velocity

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Embed. Patch

[Figure 70]

Patch

LoRA

VideoVAE

Denoising

[Figure 71]

Iterative

Insertion and Removal Dense-Captioned T2V

[Figure 72]

Instruct.

[Figure 73]

[Figure 74]

Embed. Patch

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

InputTargetMask

Noise

[Figure 84]

laughs waves hand

[Figure 85]

Input Target

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

Embed.

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

Mask

Edited Video

(a) Architecture (b) Input Formats

- Figure 3. (a) Overview of our video editing architecture. The frozen video VAE encodes control signals. Mask tokens are added to the input video tokens and concatenated with the noisy latent tokens and optional reference image. The DiT is trained using LoRA. (b) Input formats used for different dataset types during training. When object masks are available, the spatiotemporal mask input equals the input mask. Otherwise, the mask indicates transitions from the unedited to the edited video.

tency.

Input video conditioning. As shown in Fig. 3, we adopt a simple yet effective strategy to condition on the input video. Given a source video and a target video, each of shape V ∈ RNCHW, we encode them with a video VAE into latents Zsrc,Ztgt ∈ Rnchw. We use separate patch-embedding layers for Zsrc and Ztgt. After applying edit-mask conditioning (below), their tokens are concatenated along the sequence dimension. We place Zsrc first, followed by Ztgt, so the model can learn in-context video editing behavior that resembles video continuation, which is supported by our dense-caption video data.

Edit mask conditioning. The mask video M ∈ RNCHW is a binary indicator of where and when to edit the source video. It can specify (1) pixel-wise regions for inpainting or removal and/or (2) frame-wise intervals to control when an edit occurs and for how long, enabling both spatial and temporal controllability. We process the encoded mask latents Zmsk with a dedicated patch-embedding layer and inject them by addition into the source video tokens. This addition-based injection is a deliberate choice for computational efficiency. Because the mask is a low-frequency signal, its information can be effectively fused without appending it to the DiT’s input sequence. At inference time, if no mask video is provided, we default to blank masks, treating the task as a canonical, instruction-only video edit. Reference image conditioning. We optionally provide a reference image during both training and inference. The reference can be sampled from the target video during training. During inference, it can be produced by an external image-editing model by applying it to a frame from the source video or provided by the user. Because refer-

Table 1. Summary of datasets used for training. We collect ∼ 4.3M open-sourced and licensed data and curate ∼ 3.4M data, resulting in a total of ∼ 8M high-quality data for training.

Dataset Type Mask Reference # Pairs

GPT-Edit-1.5M [51] I2I ✗ ✗ ∼ 1.5M Ditto-1M [2] V2V ✗ ✗ ∼ 1M Se˜norita-2M [73] V2V ✗ ✗ ∼ 1.8M

Image Editing I2I ✗ ✗ ∼ 2M Human Animate V2V ✗ ✓ ∼ 60K Object Removal / Insertion V2V ✓ ✗ ∼ 110K Actor Transmutation V2V ✗ ✓ ∼ 4K Video Stylization V2V ✗ ✓ ∼ 90K Controllable Video Generation V2V ✗ ✗ ∼ 1.1M Human Action T2V ✗ ✓ ∼ 150K

Ours

ences can be imperfect (e.g., spurious zooms from QwenImage-Edit [36]), we apply random crops/rotations and randomly drop the reference during training so the model is robust to its absence or noise at test time. When present, the reference is encoded with the video VAE and Patch embeddingref, and its tokens are concatenated at the end of the sequence. This preserves a fixed token distance between Zsrc and Ztgt while positioning Zref closer to Ztgt for stronger guidance, leveraging off-the-shelf imageediting priors without sacrificing inference flexibility.

### 4. Data pipeline

#### 4.1. V2V data

We propose a data generation framework that utilizes various expert pipelines for generating paired V2V datasets. Our approach yields significantly stronger results than only using a single general editing pipeline. Figure 4 shows samples of each of our new datasets. Next, we detail the experts. Human animation. Previous V2V datasets [73] lack expert

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

Controllable Video Generation Human Animate Object Removal and Insertion Dense-captioned T2V

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

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

Make the person a circus ringmaster

Turn the pose video into a video showing ...

Remove the person closest to the camera Make her tilt her head to the left

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

Turn the depth video into a video showing ... Add powdered sugar falling on the dish

Make the person look like a cute little cat

Add another fish

Actor Transmutation Stylization

Image Editing w/o Transform.

[Figure 143]

Image Editing w/ Transform.

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

Add a red rose on the headstone

In the style of a cartoon

Turn the deer into a leopard

Make the water bright turquoise

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

In the style of a painting

Turn the cheetah into a polecat

Turn it into a whimsical cartoon style

Remove the clouds

Figure 4. Overview of our datasets. In each example, the left images are inputs and the right images are outputs.

annotations ensuring consistent human poses and facial expressions between input and edited videos. To overcome this, we use Wan Animate [8] to generate paired V2V data where the output video is conditioned on (1) the subject’s pose in the input video for pose consistency, (2) cropped facial regions for expression consistency, and (3) a reference image produced by an image editing model that modifies the first frame to alter the subject or scene. A large language model (LLM) generates diverse edit instructions involving actor swaps, clothing and style changes, and the addition or removal of accessories.

Object removal and insertion. We construct paired videos by removing selected objects from original clips. First, we apply an open-set object detection model [67] to the first frame of each video to detect candidate objects and generate corresponding bounding boxes and tags. Next, an LLM refines the tags by removing adjectives (e.g., color, material) and scenery-related labels (e.g., road, sky). For each video, we randomly sample a fixed number of objects, with sampling probability proportional to the bounding box area to prioritize salient objects. For each selected object, we use a video segmentation model [38] to obtain per-frame segmentation masks. A video inpainting model [72] then generates a new version of the video with the target object removed. Human annotators review the results and discard videos with noticeable inpainting artifacts. Finally, we provide the input and output videos, along with the object category and input video captions, to a vision-language model (VLM) [3] and prompt it to generate a detailed instructional caption describing the edit. By replacing the source and target videos and corresponding VLM-based changes to the instruction, this dataset is used for object insertion.

Actor transmutation. While the human animate dataset already includes high-quality examples of replacing one humanoid with another, a general editing model should also support actor transmutation for other categories such as animals. To achieve this, we leverage a zero-shot image editing model [27] and build a zero-shot V2V pipeline on top of

a pretrained video generative model [48]. We use this zeroshot V2V model to extend editing types beyond humanoidto-humanoid, including (1) quadruped-to-quadruped, (2) bipedal-to-bipedal, and (3) avian-to-avian transformations. We observe that the zero-shot editing model [27] performs noticeably better when the input video is also generated by the same video generative model. Consequently, for each video pair, we sample two actors from the same category, along with an action and a scene, and generate a video of “<actor1> performing <action> in <scene> ”. We then apply our zero-shot V2V pipeline to produce the edited version described as “<actor2> performing <action> in <scene> ”. The lists of actors, actions, and scenes for each category are generated by an LLM.

Video stylization. A general-purpose V2V model should be capable of applying global style changes to input videos. To create suitable datasets for this task, we extract edge maps from input videos to preserve structure and motion while removing appearance and style information. We then use an image editing model [29] to apply an LLM generated style transfer instruction to the first frame of the input video. This edited reference image, along with the edge video, are used as control signals to generate the stylized output video [1]. The style transfer instructions include art movements and styles, lighting conditions, aesthetics, photographic and cinematic styles, weather conditions, traditional arts and patterns, and color palettes and tones.

Controllable video generation. Video generation using control signals such as depth and edge maps has been widely studied [1, 20]. The datasets for training such models consist of video pairs that are simple and inexpensive to obtain, making them suitable as supporting data for training V2V models. These datasets enhance controllable generation capabilities and potentially promote emergent skills through data diversification. We include controllable video generation pairs in our training data, using control signals such as depth, HED edge, Canny edge, optical flow, human poses, noisy videos, and grayscale videos.

V2V transition data. To supervise how an edit unfolds over time and enable temporal mask control, we synthesize transition effects on top of paired videos V src and V tgt. Given an edit onset ti, we form a training target V ′ = [Vtsrc

0:ti, Vttgt

i:tN ] and derive a frame-wise mask that activates the edit after ti. We then apply transition operators like linear blending to ensure a natural transition effect centered at ti.

#### 4.2. I2I data

High-quality instructional V2V datasets are limited, while image editing benefits from mature models and largescale data. To bridge this gap, we train our instructional V2V editor with I2I edit pairs collected from open-source datasets [18, 51, 52, 70] and additional pairs synthesized from VLM-curated image–caption corpora. From each caption, an LLM produces diverse instruction-style edit prompts; instruction-following image editors [29, 36] generate the edited images. A VLM-based filter retains only successful, high-quality edits, yielding instruction-aligned I2I supervision without manual annotation.

I2I → V2V via affine transformations. Treating an I2I pair as a single-frame video is insufficient because it lacks temporal and motion cues. To address this, we convert each I2I pair into a pseudo video pair using a shared 2D affine camera trajectory. Specifically, we sample smooth sequences of small rotations, zooms, and translations by interpolating between a neutral and a randomly drawn target pose, with adjustments to prevent out-of-frame regions. The same trajectory is applied to both the source and target images, producing temporally consistent videos that differ only by the intended edit. This preserves the I2I supervision signal while introducing realistic temporal structure.

#### 4.3. Dense-captioned text-to-video data

Existing V2V datasets cover object manipulation, stylization, and animation but lack edits like changing human action. As a result, instruction-guided V2V models often fail to modify actions effectively. Dense-captioned textto-video datasets [54], however, provide diverse action descriptions at scale, enabling action-centric supervision. To curate T2V data for video editing, given a video V and a caption c localized to a temporal window [ti,tj] (e.g., “he sits down”), we slice V to form a source clip Vt

i−N:ti (frames preceding the caption) and a target clip Vt

i:ti+N (frames containing the caption), where N is the frame count used for training. We discard windows that are too long (tj − ti ≫ N), too short, or contain scene cuts. The caption c is then converted into an imperative instruction c′ using an LLM [56] (e.g., “make him sit down”). This yields canonical triples for training V2V editors: (source video, target video, instruction). Our approach scales efficiently to large captioned video corpora

while emphasizing action edits. Qualitative results (Figure 1) reveal that curated action edits enable EasyV2V to perform accurate and complex human action edits.

### 5. Implementation details

We adopt the pretrained video generation model, Wan-2.2TI2V-5B [48], as our base model for LoRA adaption and Wan-2.2-VAE with a spatiotemporal compression ratio of 4 × 16 × 16. We perform training and report our results based on a spatiotemporal resolution of 81×832×480 following Wan-2.1-FunControl. We also provide high-quality results fine-tuned on 81×1280×704 in the supplement. For LoRA training, we set the rank to 256 and adopt a constant learning rate of 1e−4 with AdamW [32] optimizer. We conduct our training on 32 NVIDIA H100s for our EasyV2V. All newly introduced parameters are zero-initialized. We perform random reference image dropout and video transition augmentation each with 50% probability. A complete list of all training data used is provided in Tab. 1.

### 6. Experiments

#### 6.1. Benchmarks and metrics

We evaluate our method on the latest EditVerseBench [22] which has 20 edit types. We evaluate on categories covered by our training data—excluding unsupported tasks such as camera-pose changes—resulting in 160 videos across 16 edit types. Following the benchmark’s protocol, we conduct a comprehensive quantitative evaluation using these metrics: Frame and video text alignment. We measure alignment at both frame and video levels using image-text [37] and video-text [50] encoders by computing cosine similarity between (i) each frame and the target prompt and (ii) a joint embedding of uniformly sampled frames and the prompt. Preference score. We report a preference-tuned image-text PickScore [24] that correlates with human aesthetic judgments, applied between frames and the target prompt. VLM quality assessment. We use OpenAI GPT-4o [47] to evaluate three sampled frame pairs on prompt following, edit quality, and background consistency (0-3 each), providing justifications and a total score of 9. In our analysis, we observe that the VLM score aligns most closely with human qualitative assessment so we designate the VLM score as our primary metric for comparison.

#### 6.2. Baseline comparison

We perform a comprehensive quantitative comparison against a diverse set of baseline methods in Table 2. With our primary VLM score of 7.73/9, without any guidance, we outperform not only the previously best published method but also concurrent work and a commercial solution. When a reference image is given, we can achieve better visual-text alignment performance.

Table 2. Benchmark comparison across methods on EditVerse Benchmark. Higher is better for all metrics (↑).

VLM evaluation Editing Quality ↑

Video Quality Pick Score ↑

Text Alignment ↑ Frame Video

Method

Attention Manipulation (Training-free)

TokenFlow [35] 5.02 19.59 25.10 22.49 STDF [59] 4.20 19.32 24.74 22.09 Instruction-Guided (w/ End-to-End Training)

Se˜norita-2M [73] (w/ Ref., Qwen-Image-Edit) 6.45 20.26 26.51 23.24 InsViE-1M [53] (w/ Ref) 4.36 19.25 25.06 21.28 InsV2V [9] 4.95 19.33 24.98 22.74 Ours (w/o Ref.) 7.73 20.36 27.59 24.46 Ours (w/ Ref., Qwen-Image-Edit) 7.36 20.54 27.50 24.31 Ours (w/ Ref., Flux-Kontext) 7.53 20.61 28.10 25.13 Closed-Source Commercial Models

Runway Aleph [40] 7.48 20.56 27.96 24.68 Concurrent unpublished work

Lucy Edit [44] 5.96 19.64 26.03 23.37 EditVerse [22] (code unavailable) 7.64 20.33 27.70 25.37

We provide a qualitative comparison with the baselines in Fig. 5. Since EditVerse [22] has not released its code, we take their gallery videos from their webpage. InsViE1M [53] supports only short horizontal videos and often fails to edit properly, producing severe visual artifacts. Se˜norita-2M [73] depends on the image editor for the first frame and, even with successfully edited first frames, shows motion mismatch and artifacts after the first frame. Lucy Edit Dev [44] supports limited categories of edit types and frequently exhibits motion mismatches. Compared to these methods, our model achieves higher-quality outputs and better instruction following. For example, in the first row, EditVerse fails to produce a “heavy fog” across the “whole video.” In the second, the tree branch between her fingers remains. In the third, the output lacks “visible pen lines.” In the last, EditVerse also removes the background trees. Please refer to our supplement for visualizations.

#### 6.3. Ablation study

Architecture. We ablate the architecture choice for input video conditioning and compare full model finetuning with LoRA. We evaluate the VLM score for each protocol trained with 20K and 40K steps in Tab. 3. We show that full model training tends to overfit, while LoRA tuning quickly transforms a T2V model into a V2V model. And “EmbedAdd.” refers to adopting our patch embedding addition strategy for both mask video latent and control video latent. Technically, this is similar to channel-wise concatenation of source video latent and noisy target latent except for the bias term. However, our results show that this remains suboptimal compared to sequence-wise concatenation of control latents. We also show that LoRA tuning via patch embedding addition performs worse than sequence concatenation.

Using I2I datasets. Since generating V2V editing datasets is much more expensive than creating I2I editing datasets, and because strong I2I models and high-quality opensource datasets already exist, we argue that V2V editing models should also be trained on I2I data. A simple approach is to treat image editing samples as single-frame

- Table 3. Ablation on architecture choice. “Full” denotes full model training, and “EmbedAdd.” refers to summing control and target latents after patch embedding.

Method VLM score @ 20K ↑ VLM score @ 40K ↑

Full w/ EmbedAdd. 4.67 4.57 Full w/ SeqCat. 3.66 3.94 LoRA w/ EmbedAdd. (Ours) 6.11 6.29 LoRA w/ SeqCat. (Ours) 7.05 7.47

- Table 4. Ablation study on using I2I data. Single Image treats I2I pairs as single-frame videos, while Affine Image applies affine transformations to I2I data to create V2V pairs.

Training Datasets VLM Eval. Video Quality Single Image Affine Image Video Edit Edit Quality Pick Score

✓ ✗ ✗ 5.52 19.49 ✓ ✓ ✗ 6.24 19.67 ✗ ✗ ✓ 6.69 19.90 ✓ ✓ ✓ 6.86 19.94

- Table 5. Ablation study on the effectiveness of our proposed video datasets using the VLM score as the metric.

Training on Datasets Edit types Se˜norita-2M [73] Stylization Human Animate Controllable Video Flow Edit Inpainting Dense Caption

Stylization 4.97 7.97 5.33 5.30 5.77 4.63 5.20 Animation 3.88 3.65 7.20 3.80 7.48 3.13 4.18 Change object 3.33 2.40 4.43 3.20 5.13 3.13 4.27 Control video 5.47 4.42 5.20 6.13 3.46 4.00 5.04 Actor transmutation 4.37 2.53 6.23 3.90 8.30 2.83 5.00 Edit w/ Mask 3.40 2.73 3.17 3.03 2.43 4.63 1.10 Change human action 4.97 4.50 5.03 3.53 6.50 2.50 6.87

videos, but this lacks motion. To narrow the domain gap, we apply a consistent sequence of affine transformations to the input and edited images to simulate pseudo V2V editing pairs. We perform this ablation study by sampling an equal number of I2I and V2V editing pairs and train different models for different dataset compositions. We test performance on a benchmark containing 100 videos from edit types supported by both datasets. As shown in Table 4, adding affine transformations improves downstream V2V editing performance, and jointly training on I2I datasets outperforms training solely on V2V datasets.

Using V2V and dense-captioned datasets. To examine the effect of data curated by our pipeline, we train different models on various V2V datasets using the same number of training steps. We then evaluate performance on a mini benchmark with 10 videos for each edit type. As shown in Table 5, training on each of our proposed V2V datasets significantly improves performance on certain edit types, which validates the effectiveness of our proposed data pipeline. The only exception is our Human Animate dataset, which is outperformed by our Actor Transmutation dataset according to the VLM score. Although both tasks are similar, the latter includes more diverse subjects. However, the Human Animate dataset remains useful for preserving human identities across edits and maintaining consistent facial expressions.

Impact of dataset size and generalization to unseen edits. We split a subset of our V2V training data to include three edit types only and ablate on training data size with

Input InsViE-1M Señorita-2M Lucy Edit (Concurrent) EditVerse (Concurrent) EasyV2V (Ours)

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

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

Add fog in the whole video. The lions are walking through heavy fog.

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

Change the trees to cloud.

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

Convert the video into a hand-drawn animation: Employ a frame-by-frame rotoscope effect where every shot appears sketchy with visible pen lines and watercolor textures.

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

Change the rabbit's fur to metallic silver

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

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

Remove all background people, including the blurred figures and anyone not in sharp focus, so that only the smiling girl in traditional clothing remains [...]

Figure 5. Qualitative comparison of our video editing results with the baselines.

- 9

VLMScore

Seen Edits Unseen Edits

Full Score

10K samples 100K samples

| |
|---|

1M samples

Figure 6. Ablation study on training data size and generalization.

- 10K, 100K, and 1M samples. As training data size increases, Figure 6 shows performance improves accordingly for both seen and unseen edits. We observe that training with only 10K examples already yields fair performance. Moreover, editing capability on seen tasks consistently enhances performance on unseen edit categories, validating that the inherent edit ability of a pretrained T2V model can be unlocked with our efficient tuning.

6

3

0

addobjectremoveobjectchangeobject combinedtasks animation stylization

### 7. Conclusion

We introduced EasyV2V, a lightweight, instruction-based video editor that achieves state-of-the- performance. First, we propose robust data engines that curate diverse training data, notably by “lifting” static image edits into dynamic pseudo-video pairs using shared affine motion, supplemented with video continuation supervision from text2video data. Second, we propose a parameter-efficient architecture which effectively conditions on the source video, enabling unified spatiotemporal control from a single mask with optional reference images. This combination of a novel data strategy and a minimal-tuning architecture provides a strong and effective recipe for high-quality video editing with extensive controllability.

Limitations. While EasyV2V is robust on a wide range of edit types, common to other concurrent diffusion-based video models, inference takes around one minute and precludes its use in real-time applications. This framework can be naturally extended to support more advanced control

###### abilities. For example, adding geometric cinematographic camera pose controls is a compelling direction to further broaden our creative applications.

## EasyV2V: A High-quality Instruction-based Video Editing Framework Supplementary Material

### A. Performance on image editing

During training, we occasionally sampled from imageediting datasets and treated each image–edit pair as a singleframe video, rather than exclusively applying affine transformations to synthesize pseudo-videos.

As shown in Table I, we evaluated EasyV2V on a recent image-editing benchmark [60], interpreting each image as a single-frame video with a resolution of 1 × 832 × 480. Impressively, although EasyV2V was not specifically designed for image editing, it surpassed all baselines on nearly all subtasks and achieved performance comparable to leading closed-source and commercial systems.

Notably, EasyV2V outperformed EditVerse by a margin of 0.54. The model also demonstrated strong results on both action and hybrid editing tasks, underscoring the effectiveness of our unified data pipeline that jointly leverages image-editing datasets and video–caption datasets featuring human actions. Figure I presents representative qualitative results, with additional examples provided in our supplementary materials.

### B. Model configuration B.1. Mask conditioning

We compare our mask-conditioned editing performance with WanVACE [20]. Note that as WanVACE is not an instruction-based editing model, we restrict our comparison to the edit types it supports via edit masks: inpainting, object removal, and object replacement. We evaluate two types of edit masks: pixel-wise spatial masks (indicating the edit region per frame) and frame-wise temporal masks (indicating which frames should be modified). While WanVACE utilizes an additional branch with complex context activation and injection for mask conditioning, we demonstrate that encoding masks using the video VAE is simple yet effective, particularly for temporal control.

We investigate different mask conditioning strategies for EasyV2V:

- • Video VAE, Zmsk + Zsrc: After encoding the mask video into mask latent Zmsk using the video VAE, we perform token addition to inject the mask condition into the source video latent. This is the default strategy for EasyV2V in the main paper.
- • Video VAE, Zmsk + Ztgt: After encoding the mask video into mask latent Zmsk using the video VAE, we perform token addition to inject the mask condition into the noisy target video latent.
- • Downsample, Zmsk +Zsrc: We apply spatial and temporal

average pooling to downsample the mask video from input resolution to latent resolution, then perform token addition to inject the mask condition into the source video latent.

• Video VAE, Seq Cat(Zsrc,Ztgt,Zmsk): After encoding the mask video into mask latent Zmsk using the video VAE, we perform sequence concatenation for all condition signals, including Zsrc,Ztgt,Zmsk, and Zref.

To evaluate spatial mask editing, we adopt the VLM prompt from the EditVerse evaluation protocol. To evaluate temporal mask editing, where the edit occurs after a certain timestamp, we modify the VLM prompt to incorporate edit timestamp awareness. Please refer to the last page for our complete VLM prompts.

As shown in Table II, EasyV2V achieves the best performance when using token addition to inject the mask condition into the source video latent. Both sequence concatenation and our token addition strategy outperform WanVACE on mask-conditioned video editing.

From the qualitative comparison in Figure II, we observe that WanVACE [20] has limited ability to generalize to diverse edit prompts and fails to adhere to both temporal and spatial masks in many cases compared to EasyV2V.

#### B.2. LoRA Rank

We employ LoRA [16] to mitigate divergence and study the effect of LoRA rank. We train the model for 20K steps to ablate the impact of LoRA rank on the EditVerse benchmark. We observe that performance improves as we increase the rank. We adopt a LoRA rank of 256, as performance begins to saturate between ranks 128 and 256. Table III reports the results for ranks 64, 128, and 256. Although the model with rank 256 performs best on most metrics, the gap between ranks 128 and 256 is marginal, and rank 64 is only slightly inferior. This supports our hypothesis that pre-trained video models serve as strong priors for video editing, and that a low-rank update is sufficient for robust performance.

#### B.3. Time and memory profiling

We profile the efficiency of EasyV2V on a single NVIDIA H100 GPU for training and inference under different strategies: (1) full model fine-tuning with sequence concatenation of source and target latents, (2) full model fine-tuning with token addition of source and target latents, (3) LoRA fine-tuning with sequence concatenation of source and target latents, and (4) LoRA fine-tuning with token addition of source and target latents. FlashAttention is enabled during both training and inference. A complete comparison is

Table I. Category-wise image-editing performance on ImgEdit Bench [60]. Scores are derived from a vision–language model (VLM) based on Prompt Compliance, Visual Naturalness/Seamlessness, and Physical/Detail Coherence (higher is better). Impressively, EasyV2V surpasses all baselines across most categories and approaches the performance of leading commercial systems, despite not being specifically designed for image editing.

###### Method Add Adjust Extract Replace Remove Background Style Hybrid Action Overall↑

MagicBrush 2.84 1.58 1.51 1.97 1.58 1.75 2.38 1.62 1.22 1.83 Instruct-P2P 2.45 1.83 1.44 2.01 1.50 1.44 3.55 1.20 1.46 1.88 AnyEdit 3.18 2.95 1.88 2.47 2.23 2.24 2.85 1.56 2.65 2.45 UltraEdit 3.44 2.81 2.13 2.96 1.45 2.83 3.76 1.91 2.98 2.70 ICEdit 3.58 3.39 1.73 3.15 2.93 3.08 3.84 2.04 3.68 3.05 Step1X-Edit 3.88 3.14 1.76 3.40 2.41 3.16 4.63 2.64 2.52 3.06 UniWorld-V1 3.82 3.64 2.27 3.47 3.24 2.99 4.21 2.96 2.74 3.26 BAGEL 3.81 3.59 1.58 3.85 3.16 3.39 4.51 2.67 4.25 3.42 OmniGen2 3.57 3.06 1.77 3.74 3.20 3.57 4.81 2.52 4.68 3.44 Kontext-dev 3.83 3.65 2.27 4.45 3.17 3.98 4.55 3.35 4.29 3.71 EasyV2V (Ours) 4.46 4.18 1.80 3.86 3.70 4.33 4.57 4.04 4.68 3.96

Closed-Source Commercial Models and Concurrent works

EditVerse 3.81 3.62 1.44 3.95 3.14 3.58 4.71 2.72 3.80 3.42 Ovis-U1 3.99 3.73 2.66 4.38 4.15 4.05 4.86 3.43 4.68 3.97 GPT-4o-Image 4.61 4.33 2.90 4.35 3.66 4.57 4.93 3.96 4.89 4.20

Input EasyV2V (Ours) Input EasyV2V (Ours)

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

Transfer the image into a traditional ukiyo-e woodblock-print style. Add a small sailboat floating near the end of the dock in the background, with its sails partially filled with wind.

[Figure 265]

[Figure 266]

[Figure 267]

|[Figure 268]|
|---|

Replace the mountain goat in the image with a rabbit. Change the color of the vehicle to red.

[Figure 269]

|[Figure 270]|
|---|

Remove the human standing in the foreground wearing a blue suit, who is positioned in front of the construction site.

[Figure 271]

[Figure 272]

Change the green foliage background in the picture to a coastal beach setting.

Figure I. Qualitative image-editing results from our video-editing model, which treats images as single-frame videos and achieves state-ofthe-art performance, showing that video editing aids image editing.

### provided in Table IV. C. Additional details on data pipelines

Human animation. We use the 14B-parameter pretrained Wan Animate [8] model as the expert editor, following its

###### Source Video Mask Video WanVACE EasyV2V

###### Source Video Mask Video WanVACE EasyV2V

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

Frame0(noedit)

Frame0

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

Frame40(editing)Frame80(edited)

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

Frame40Frame80

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

(b) Edit with temporal mask. Edit prompt: “change the woman to a robot.”

(a) Edit with spatial mask. Edit prompt: “remove the lamb.”

Figure II. Comparison of mask-guided editing performance with WanVACE using samples from EditVerseBench [22].

Table II. Comparison of video editing mask strategies.

Method VLM score (↑) Pixel-wise spatial mask Frame-wise temporal mask Average

Wan VACE [20] 4.13 6.87 5.50 Video VAE, Zmsk + Ztgt 5.50 7.40 6.45 Downsample, Zmsk + Zsrc 5.80 5.40 5.60 Video VAE, Seq Cat(Zsrc,Ztgt,Zmsk) 6.00 7.70 6.85 Video VAE, Zmsk + Zsrc (EasyV2V) 7.23 7.73 7.48

- Table III. Ablation study on LoRA rank and the use of the reference image.

Rank 64 Rank 128 Rank 256 Metric w/o Ref. w/ Ref. w/o Ref. w/ Ref. w/o Ref. w/ Ref.

Frame-Text Alignment 24.80 27.27 24.99 27.67 25.32 27.67 Video-Text Alignment 21.31 24.36 21.45 24.73 21.90 24.60 PickScore Video Quality 19.42 20.20 19.52 20.36 19.58 20.37 VLM Quality Score 6.17 7.02 6.20 7.12 6.48 7.22

- Table IV. Comparison of training and inference costs across different tuning strategies.

Metric Full Model w/ SeqCat. Full Model w/ EmbedAdd. LoRA w/ SeqCat. LoRA w/ EmbedAdd.

New Params 5B 5B 0.64B 0.64B Train (s / batch) 5.63 4.60 5.70 4.54 Train VRAM 62GB 62GB 37GB 32GB Inference (s / sample) 67.71 30.41 69.42 30.11

preprocessing for face crops and poses. For first-frame editing, we apply Flux Kontext Dev [29]. Edit prompts are generated by ChatGPT [47] using 150 prompts created from the following instruction:

You are a helpful assistant to help with the generation of video editing prompts, to edit videos of people. Below are samples of the prompts we're interested it:

→ → → →

# Fantasy & Mythical Creatures

- - "Make the person look like an elf"
- - "Make the person look like a goblin" # Professions & Archetypes
- - "Make the person look like a knight"
- - "Make the person look like a samurai" # Horror & Dark Styles
- - "Make the person look like a zombie"
- - "Make the person look like a vampire" # Animals & Hybrids
- - "Make the person look like a lion"
- - "Make the person look like a tiger" # Sci-Fi & Futuristic
- - "Make the person look like a robot"

- - "Make the person look like a cyborg" # Stylized & Surreal
- - "Make the person look like a

→ stained-glass figure"

- - "Make the person look like a chalk → drawing" # Accessories
- - "Give the person a pair of sunglasses"
- - "Give the person a hat"

Now, generate several prompts per category

→ mentioned above.

##### Actor transmutation.

We adapt FlowEdit [27], originally designed for image editing, on Wan 2.1 14B [48] for video editing. As described in the main paper, our prompt assets include three object categories: bipedals (e.g., clown, pirate, ninja, samurai, humanoid robot), quadrupeds (e.g., dog, cat, lion, cheetah, sheep), and avians (e.g., pigeon, duck, parrot, eagle, owl). We provide five examples per category to ChatGPT to generate extended lists. We also compile lists of scenes (e.g., jungle, mountain, beach, street, bedroom) and actions (e.g., walking, running, jumping, dancing).

##### Video stylization.

Similar to the prompts for our human animate dataset, we use ChatGPT to generate prompts for video stylization:

You are a helpful assistant to help with the generation of video stylization prompts, to edit and stylize in the wild videos. Below are samples of the prompts we're interested it:

→ → → →

# Distinct media / aesthetics

- - "In the style of a watercolor painting"
- - "In the style of a digital painting" # Comic / cartoon & animation houses
- - "In the style of a manga"
- - "In the style of a cartoon" # Art movements
- - "In the style of Brutalism"
- - "In the style of Impressionism" # Lighting
- - "Captured in the golden hour"
- - "Captured bathed in neon lights" # Cinematic & photographic framing styles
- - "Shot on 35 mm film grain"
- - "Shot in black-and-white film noir style" # Global traditional arts & patterns
- - "In the style of a Roman floor mosaic"

- "In the style of Persian miniature

→ painting"

# Color palette & tonal approaches

- - "Rendered in soft pastel hues"
- - "Rendered in muted earth tones"

Now, generate several prompts per category

→ mentioned above.

We end up generating 350 different stylization prompts. Controllable video generation.

We curate a 15K-sample dataset from in-the-wild videos through manual filtering, then apply a range of model-free and model-based video-to-video transformations to build a paired dataset for training our video editing model. These transformations include human pose estimation (DWPose), Canny and HED edge detection, RAFT large optical flow, random black rectangle masks (inpainting), random black borders (outpainting), depth prediction with Depth Anything V2, grayscale conversion, Gaussian blur, color negation, saturation/contrast/brightness adjustments, pixelation, wave warping, posterization, Gaussian noise, and color overlays.

##### Dense-captioned text-to-video data

We apply strict filtering criteria to ensure high-quality training pairs from dense-captioned datasets. First, we require videos to have at least 162 frames after downsampling to 15 fps (enabling 81 frames for both source and target clips). Second, we filter temporal segments based on: (i) start time must allow sufficient preceding frames (ti ≥ 81/fpsdownsample), (ii) segment duration must exceed 2 seconds to ensure meaningful actions, and (iii) no scene cuts within the segment interval. For instruction generation, we flatten all temporal segments from multiple videos into batches and process them simultaneously with Qwen-34B [56], discarding segments where the LLM returns empty strings (indicating unsuitable actions for video editing conversion). The LLM prompt instructs the model to convert action descriptions into imperative instructions starting with verbs like “make,” “let,” or “have,” while preserving all key details.

##### Image-to-image data

We employ a multi-stage pipeline to generate highquality instructional I2I pairs from image captions at scale. Given an image caption, an LLM (Qwen3-4BInstruct [56]) generates up to five diverse edit instructions spanning canonical edit types: add, remove, replace, change global, change local, change color, transform global, transform local, text, and other. Each instruction is then executed using instructionfollowing image editors (Qwen-Image-Edit [36] , or FluxKontext [29]), producing candidate edit pairs. To ensure quality, we apply VLM-based filter with Gemma-327B [46].

For I2I-to-V2V conversion, we generate smooth affine camera trajectories by sampling random target poses with bounded parameters: rotation angles in [−15◦,15◦], zoom factors in [0.66,1.0] (avoiding excessive zoom-out), and translation offsets within ±33% of frame dimensions. These parameters are linearly interpolated across frames and constrained via linear programming to ensure the transformed bounding box remains fully within the frame boundaries throughout the trajectory. The same trajectory is applied to both source and target images using perspective transforms, creating temporally coherent pseudo-videos with motion cues (zoom, pan, rotation) while preserving the edit signal. With 50% probability, trajectories are reversed to balance zoom-in and zoom-out motions, providing diverse camera movement patterns that enhance the model’s robustness to dynamic viewpoints during training.

### D. Impact of classifier free guidance

We perform an ablation study on the impact of classifierfree guidance (CFG) [15] during inference.

CFG Implementation. Following the standard CFG formulation, we guide the denoising process by interpolating between conditional and unconditional predictions. Given the predicted noise ϵθ from our diffusion model, the CFGguided prediction is computed as:

ϵˆθ = ϵθ(∅) + s · (ϵθ(c) − ϵθ(∅)) (1)

where c represents the conditioning signal, ∅ denotes the unconditional (null) condition, and s is the guidance scale. This can be reformulated as:

ϵˆθ = (1 − s) · ϵθ(∅) + s · ϵθ(c) (2)

When s = 1.0, the model performs purely conditional generation, while larger values of s amplify the influence of the conditioning signal.

In our framework, we support two CFG strategies depending on which conditions are used for guidance: Prompt-only CFG and Prompt + Reference CFG.

Prompt-only CFG. By default, we apply CFG only to the text prompt while keeping all visual conditions (source video and reference image) shared between conditional and unconditional branches:

ϵˆθ = ϵθ(cvis) + s · (ϵθ(cvis,cprompt) − ϵθ(cvis)) (3)

where cvis denotes visual conditions (source video and optionally reference image), and cprompt is the edit instruction. This approach maintains consistent visual context while allowing the text prompt to guide the editing direction.

Prompt + Reference CFG. Alternatively, we can apply CFG to both the text prompt and reference image:

ϵˆθ = ϵθ(csrc) + s · (ϵθ(csrc,cref,cprompt) − ϵθ(csrc)) (4)

Table V. Effect of the CFG scale for the edit prompt.

Inference w/o Ref. Inference w/ Ref.

CFG scale 1.0 3.0 5.0 7.0 1.0 3.0 5.0 7.0 Frame-Text Alignment 26.65 27.59 27.49 27.11 27.26 27.28 27.33 27.29 Video-Text Alignment 24.52 24.46 24.16 23.40 24.01 24.07 24.27 24.33 PickScore Video Quality 20.05 20.36 20.23 20.05 20.57 20.60 20.58 20.53 VLM Quality Score 6.79 7.73 7.48 6.98 7.14 7.30 7.28 7.21

Source Video CFG=1.0 CFG=3.0 CFG=5.0 CFG=7.0

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

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

Figure III. Effect of the CFG for the edit prompt. Edit prompt: “Transform the entire visual style of the video using a hand-drawn watercolor animation effect.”

where csrc is the source video (always present), cref is the reference image, and cprompt is the text prompt. This strategy applies guidance to both the reference appearance and text instruction simultaneously. Our method is uniquely suitable for this type of guidance because the reference image is an optional input to our model.

Experimental results. In Table V, we present results under different CFG scales when using only the edit prompt for CFG. We also provide a visualized example in Figure III. As shown, higher CFG scales generally improve text alignment but may reduce temporal consistency and video quality when the scale becomes too large.

In Table VI, we show results under different CFG scales when using both the edit prompt and reference image for CFG. For fair comparison, we report inference performance only when a reference image is provided. A visualized example is shown in Figure IV. The dual-condition CFG provides stronger control over both appearance and semantic alignment but requires careful tuning of the guidance scale to balance faithfulness to conditions and generation quality.

We observe that the best performance is achieved when CFG scales are between 3.0 and 5.0. We adopt a moderate CFG scale of s = 3.0 by default with prompt-only guidance. We believe that further fine-grained CFG hyperparameter tuning could yield even better performance.

### E. User study

We construct a custom benchmark comprising 160 horizontal and vertical videos spanning 18 distinct edit types, including actor transmutation, add effect, add object, animation, change action, change background, change color, change material, change object, change weather, complex

Table VI. Effect of the CFG scale for the edit prompt and reference image.

Inference w/ Ref.

CFG scale 1.0 3.0 5.0 7.0 Frame-Text Alignment 26.65 27.77 27.49 27.11 Video-Text Alignment 24.52 24.54 24.16 23.40 PickScore Video Quality 20.05 20.34 20.23 20.05 VLM Quality Score 6.79 7.69 7.47 6.98

Source Video

###### CFG=1.0 CFG=3.0 CFG=5.0 CFG=7.0

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

Ref.img

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

[Figure 329]

[Figure 330]

Figure IV. Visual illustration of the effect of the CFG scale for the edit prompt and reference image. Edit prompt: “Change the bird’s color to emerald green.” The top row shows the reference image generated by Qwen-Image-Edit.

edit, control video, edit with mask, freeform, local stylization, global stylization, remove object, and sim2real. We conduct a user study in which participants select the superior sample between outputs generated by two different methods, evaluating them along three dimensions: Instruction Alignment (adherence to the text prompt), Preservation of Unedited Regions (temporal consistency in unchanged areas), and Video Quality (overall visual fidelity). As shown in Figure V, EasyV2V outperforms all other methods across all three dimensions.

### F. Additional visualization

#### F.1. Robustness to Reference Image

We provide visualization results of EasyV2V based on the choice of reference image. By default, we derive the reference image by applying the image editing model to the first frame of the source video. We also present results using the first, middle, or last frame of the source video as the basis for the reference. As shown in Figure VI, our model is robust to the choice of reference image, indicating that EasyV2V effectively captures the identity of the reference image for video editing. Moreover, EasyV2V can rectify inconsistent zoom-in effects and human pose misalignments introduced by the image editing model. Notably, without an external reference image from the image editing model, EasyV2V achieves even better consistency with the source video; for instance, the background remains well preserved.

#### F.2. Comparison on Human Animate and Flow Edit Datasets

We curate the Human Animate dataset, which contains human-centric video edits, and the Flow Edit dataset, which focuses on actor transmutation edits. We provide comparisons in Figure VII between a model trained exclusively on the Human Animate dataset and one trained on the Flow Edit dataset. The model trained on the Human Animate dataset often achieves superior visual details, preserves poses more effectively, and generalizes better to unseen human-specific pose-to-video tasks. Facial expressions are also better preserved when training on the Human Animate dataset compared to the Flow Edit (actor transmutation) dataset.

#### F.3. Exhibition Gallery and Visualization

We present additional video editing examples on our visualization website included in the supplementary material. We strongly encourage readers to view the attached HTML file for comprehensive and diverse video results.

Capability for Human Action Editing. Leveraging a densely captioned video dataset during training, our model demonstrates a strong capability to follow text instructions for modifying human actions. Compared to concurrent works, which often struggle to alter human actions effectively, EasyV2V exhibits a unique proficiency in such edits, highlighting the effectiveness of our curated human action data.

Natural Edit Transitions. Although we employ a simple blending transition strategy during training, EasyV2V exhibits an emergent ability to produce natural transitions. We evaluate this using frame-wise temporal masks where the edit is restricted to the second half of the video. We observe smooth and realistic transitions at the timestamp where the source-to-target edit initiates.

High-Resolution Results. We train our model at a higher resolution of 81×1280×704 to further validate our method and data pipeline. The total training data is subsampled to approximately 6 million samples due to the computational cost of high-resolution training. Note that 81 × 1280 × 704 is the maximum supported resolution of Wan-2.2-TI2V5B [48]. We observe rapid convergence within a few training steps, confirming that our architecture design with lowrank tuning efficiently adapts a T2V model to V2V tasks. To the best of our knowledge, EasyV2V is the first instructionbased video editing model capable of editing ∼720P videos with a duration of 81 frames.

### G. VLM prompts for evaluation

VLM prompt we used for spatial mask editing evaluation:

Win Tie Loss

|81.9%|6.2%|11.9%|
|---|---|---|

vs. Señorita-2M: Quality

|56.9%|11.9%|31.2%|
|---|---|---|

vs. Señorita-2M: Preservation

|78.8%|8.8%|12.5%|
|---|---|---|

vs. Señorita-2M: Instruction

|68.8%|8.1%|23.1%|
|---|---|---|

vs. Lucy Edit: Quality

|45.0%|13.8%|41.2%|
|---|---|---|

vs. Lucy Edit: Preservation

|87.5%| |9.4%|
|---|---|---|

vs. Lucy Edit: Instruction

|94.4%| | |
|---|---|---|

vs. InsViE-1M: Quality

|83.1%|5.6%|11.2%|
|---|---|---|

vs. InsViE-1M: Preservation

|93.8%| | |
|---|---|---|

vs. InsViE-1M: Instruction

Figure V. Results of the user study. EasyV2V is the most preferred method across all evaluation criteria.

'You are a meticulous video editing quality evaluator. Your task is to provide a detailed assessment of a video edit by comparing the original image with the edited image based on a given text prompt.\n\

→ → → → →

Editing Prompt:\n{editing_prompt}\n\ Instructions:\n\ Analyze the provided image (the edited

video frame) and evaluate how well the "Editing Prompt" has been executed. You will evaluate the edit across three distinct criteria. For each criterion, provide a score from 0 (worst) to 3 (best) and a brief justification. Finally, provide the total score.\n\

→ → → → → → →

Your evaluation should focus on three key

→ aspects:\n\

- 1. Prompt Following (Score: 0-3) \n\ Question: Does the edit accurately and

completely fulfill the instructions in the "Editing Prompt"? \n\

→ →

Scoring Guide:\n\

- - 3: The prompt is perfectly and completely

→ followed.\n\

- - 2: The prompt is mostly followed but with

→ minor inaccuracies or omissions.\n\

- - 1: The prompt is poorly followed or only

→ partially executed.\n\

- - 0: The prompt is completely ignored or

→ the opposite was done. \n\

- 2. Edit Quality (Score: 0-3) \n\

Question: How is the visual quality of the edited area itself? Is it realistic, seamless, and free of artifacts (e.g., blurriness, distortion, unnatural textures)?\n\

→ → → →

Scoring Guide:\n\

- - 3: The edit is of high visual quality,

→ seamless, and artifact-free.\n\

- - 2: The edit is good but has minor,

→ noticeable artifacts.\n\

- - 1: The edit is of low quality with

→ significant, distracting artifacts.\n\

- - 0: The edited area is extremely poor,

→ garbled, or has completely failed.\n\ 3. Background Consistency (Score: 0-3) \n\ Question: Have the areas that should not

have been edited remained unchanged between the "Before" and "After" images? \n\

→ → →

Scoring Guide:\n\

- - 3: The areas that should not have been edited are perfectly preserved and stable. \n\

→ →

- - 2: There are minor, subtle, but noticeable changes or flickers in the areas that should not have been edited.\n\

→ → →

- - 1: There are significant and distracting changes in the areas that should not have been edited. \n\

→ →

- - 0: The areas that should not have been edited is completely or catastrophically altered. \n\

###### → →

VLM prompt we used for temporal mask editing evaluation:

Ref. img from beginning frame

Ref. img from middle frame

Ref. img from end frame

###### Source Video Wo Ref. img

[Figure 331]

[Figure 332]

[Figure 333]

Ref.img

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

Figure VI. Robustness of EasyV2V to the choice of reference image. Edit prompt: “Change the apron and blouse to a classic clown costume.” The top row shows reference images generated by Qwen-Image-Edit based on different frames from the source video, followed by videos generated by EasyV2V conditioned on these reference images. The source video is taken from Lucy Edit’s website [44].

You are a meticulous video editing quality evaluator. Your task is to provide a detailed assessment of a temporal video edit by comparing the original image with the edited image based on a given text prompt.

→ → → → →

Editing Prompt: {editing_prompt}

IMPORTANT TEMPORAL CONTEXT: This frame is from the FIRST HALF of the

video, where the edit SHOULD NOT have occurred yet. The frame should remain unchanged from the original.

→ → →

....

Your evaluation should focus on three key

→ aspects:

1. Temporal Consistency (Score: 0-3) Question: Does the frame correctly show NO

→ editing in the first half of the video? Scoring Guide:

- - 3: The frame is perfectly unchanged, showing no signs of the edit that should only appear in the second half.

→ →

- - 2: The frame is mostly unchanged but shows very minor, subtle hints of the edit.

→ →

- - 1: The frame shows partial editing when

→ it should be unchanged.

- - 0: The frame is fully edited when it

→ should be completely unchanged.

############# IMPORTANT TEMPORAL CONTEXT: This frame is from the TRANSITION PERIOD

(around the middle of the video), where the edit IS HAPPENING. The frame should show the edit in progress or just completed.

→ → → →

Instructions: Analyze the provided images and evaluate how well the temporal editing is progressing. You will evaluate the edit across three distinct criteria. For each criterion, provide a score from 0 (worst) to 3 (best) and a brief justification. Finally, provide the total score.

→ → → → → → →

Your evaluation should focus on three key

→ aspects:

1. Edit Progress (Score: 0-3) Question: Does the frame show appropriate

edit progression during this transition period?

→ →

Scoring Guide:

- - 3: The frame shows natural, smooth editing transition that aligns with the temporal position.

→ →

- - 2: The frame shows editing but the transition is slightly abrupt or unnatural.

→ →

- - 1: The frame shows poor editing

→ progression or timing.

- - 0: The frame shows no editing or

→ completely wrong timing.

.... IMPORTANT TEMPORAL CONTEXT:

[Figure 349]

Input

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

EasyV2V Trained w/ Our Actor Transmutation (Flow Edit) Dataset

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

Identity Changed!

[Figure 367]

[Figure 368]

Pose changed!

Pose changed!

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

Generalization Failed!

[Figure 373]

EasyV2V Trained w/ Our Human Animate Dataset

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

Identity Preserved

[Figure 379]

[Figure 380]

Aligned Pose

Aligned Pose

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

Successful Generalization

Figure VII. Comparison between models trained on the Flow Edit dataset and the human animate dataset.

This frame is from the SECOND HALF of the video, where the edit SHOULD HAVE been completed. The frame should show the fully edited result.

→ → →

Instructions: Analyze the provided images and evaluate

how well the "Editing Prompt" has been executed. You will evaluate the edit across three distinct criteria. For each criterion, provide a score from 0 (worst) to 3 (best) and a brief justification. Finally, provide the total score.

→ → → → → → →

Your evaluation should focus on three key

→ aspects:

1. Prompt Following (Score: 0-3)

Question: Does the edit accurately and completely fulfill the instructions in the "Editing Prompt"?

→ →

Scoring Guide:

- - 3: The prompt is perfectly and completely

→ followed.

- - 2: The prompt is mostly followed but with

→ minor inaccuracies or omissions.

- - 1: The prompt is poorly followed or only

→ partially executed.

- - 0: The prompt is completely ignored or

→ the opposite was done. #############

### References

- [1] VideoX-Fun: A more flexible framework that can generate videos at any resolution and creates videos from images. https://github.com/aigc-apps/VideoX-Fun,

2025. 1, 5

- [2] Qingyan Bai, Qiuyu Wang, Hao Ouyang, Yue Yu, Hanlin

- Wang, Wen Wang, Ka Leong Cheng, Shuailei Ma, Yanhong Zeng, Zichen Liu, Yinghao Xu, Yujun Shen, and Qifeng Chen. Scaling instruction-based video editing with a highquality synthetic dataset. arXiv, 2025. 4
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-VL technical report, 2025. 5
- [4] Tim Brooks, Aleksander Holynski, and Alexei A Efros. InstructPix2Pix: Learning to follow image editing instructions. CVPR, 2023. 3
- [5] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators.

2024. 3

- [6] Qi Cai, Jingwen Chen, Yang Chen, Yehao Li, Fuchen Long, Yingwei Pan, Zhaofan Qiu, Yiheng Zhang, Fengbin Gao, Peihan Xu, et al. HiDream-I1: A high-efficient image generative foundation model with sparse diffusion transformer. arXiv, 2025. 3
- [7] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. MasaCtrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In ICCV, 2023. 3
- [8] Gang Cheng, Xin Gao, Li Hu, Siqi Hu, Mingyang Huang, Chaonan Ji, Ju Li, Dechao Meng, Jinwei Qi, Penchong Qiao, Zhen Shen, Yafei Song, Ke Sun, Linrui Tian, Feng Wang, Guangyuan Wang, Qi Wang, Zhongjian Wang, Jiayu Xiao, Sheng Xu, Bang Zhang, Peng Zhang, Xindi Zhang, Zhe Zhang, Jingren Zhou, and Lian Zhuo. Wan-Animate: Unified character animation and replacement with holistic replication, 2025. 1, 3, 5, 11
- [9] Jiaxin Cheng, Tianjun Xiao, and Tong He. Consistent videoto-video transfer using synthetic dataset. ICLR, 2024. 1, 2, 3, 7
- [10] Paul Couairon, Cl´ement Rambour, Jean-Emmanuel HAUGEARD, and Nicolas THOME. VidEdit: Zero-shot and spatially aware text-driven video editing. TMLR, 2024. 1, 3
- [11] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, and Feng Li. Emerging properties in unified multimodal pretraining: The BAGEL model. arXiv, 2025. 3
- [12] Yu Gao, Lixue Gong, Qiushan Guo, Xiaoxia Hou, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao, Liyang Liu, Wei Liu, Yichun Shi, Shiqi Sun, Yu Tian, Zhi Tian, Peng Wang, Rui Wang, Xuanda Wang, Xun Wang, Ye Wang, Guofeng Wu, Jie Wu, Xin Xia, Xuefeng Xiao, Zhonghua Zhai, Xinyu Zhang, Qi Zhang, Yuwei Zhang, Shijia Zhao, Jianchao Yang, and Weilin Huang. Seedream 3.0 technical report, 2025. 3
- [13] Yuying Ge, Sijie Zhao, Chen Li, Yixiao Ge, and Ying Shan. SEED-Data-Edit technical report: A hybrid dataset for instructional image editing. arXiv, 2024. 3

- [14] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv, 2022. 3
- [15] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 14
- [16] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In ICLR, 2022. 2, 10
- [17] Jiahao Hu, Tianxiong Zhong, Xuebo Wang, Boyuan Jiang, Xingye Tian, Fei Yang, Pengfei Wan, and Di Zhang. VIVID10M: A dataset and baseline for versatile and interactive video local editing, 2025. 3
- [18] Mude Hui, Siwei Yang, Bingchen Zhao, Yichun Shi, Heng Wang, Peng Wang, Yuyin Zhou, and Cihang Xie. HQ-Edit: A high-quality dataset for instruction-based image editing. arXiv, 2024. 3, 6
- [19] Imagen-Team-Google. Imagen 3, 2024. 3
- [20] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. VACE: All-in-one video creation and editing. In ICCV, 2025. 1, 3, 5, 10, 12
- [21] Xuan Ju, Ailing Zeng, Yuxuan Bian, Shaoteng Liu, and Qiang Xu. PnP inversion: Boosting diffusion-based editing with 3 lines of code. ICLR, 2024. 3
- [22] Xuan Ju, Tianyu Wang, Yuqian Zhou, He Zhang, Qing Liu, Nanxuan Zhao, Zhifei Zhang, Yijun Li, Yuanhao Cai, Shaoteng Liu, Daniil Pakhomov, Zhe Lin, Soo Ye Kim, and Qiang Xu. EditVerse: Unifying image and video editing and generation with In-Context learning, 2025. 3, 6, 7, 12
- [23] Xuan Ju, Weicai Ye, Quande Liu, Qiulin Wang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, and Qiang Xu. FullDiT: Multi-task video generative foundation model with full attention, 2025. 3
- [24] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36:36652– 36663, 2023. 6
- [25] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv, 2024. 3
- [26] Max Ku, Cong Wei, Weiming Ren, Harry Yang, and Wenhu Chen. AnyV2V: A tuning-free framework for any video-tovideo editing tasks. TMLR, 2024. 1, 3
- [27] Vladimir Kulikov, Matan Kleiner, Inbar HubermanSpiegelglas, and Tomer Michaeli. FlowEdit: Inversion-free text-based editing using pre-trained flow models. ICCV,

2025. 3, 5, 13

- [28] Black Forest Labs. FLUX. https://github.com/ black-forest-labs/flux, 2024. 3
- [29] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas M¨uller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. FLUX.1 Kontext: Flow matching

for in-context image generation and editing in latent space,

2025. 3, 5, 6, 12, 13

- [30] Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. Video-P2P: Video editing with cross-attention control.

- CVPR, 2024. 3

[31] Shaoteng Liu, Tianyu Wang, Jui-Hsien Wang, Qing Liu, Zhifei Zhang, Joon-Young Lee, Yijun Li, Bei Yu, Zhe Lin, Soo Ye Kim, and Jiaya Jia. Generative video propagation.

- CVPR, 2025. 3

- [32] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 6
- [33] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. FateZero: Fusing attentions for zero-shot text-based video editing. ICCV,

2023. 3

- [34] Bosheng Qin, Juncheng Li, Siliang Tang, Tat-Seng Chua, and Yueting Zhuang. InstructVid2Vid: Controllable video editing with natural language instructions. In ICME, 2024. 3
- [35] Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K. Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. In CVPR, 2025. 7
- [36] Qwen Team. Qwen-Image: A unified foundation model for image generation and editing. arXiv, 2025. 4, 6, 13
- [37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 6
- [38] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, ChaoYuan Wu, Ross Girshick, Piotr Doll´ar, and Christoph Feichtenhofer. SAM 2: Segment anything in images and videos. arXiv, 2024. 5
- [39] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 3
- [40] Runway. Introducing runway aleph. https : / / runwayml . com / research / introducing runway-aleph. 7
- [41] Team Seedream, :, Yunpeng Chen, Yu Gao, Lixue Gong, Meng Guo, Qiushan Guo, Zhiyao Guo, Xiaoxia Hou, Weilin Huang, Yixuan Huang, Xiaowen Jian, Huafeng Kuang, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao, Liyang Liu, Wei Liu, Yanzuo Lu, Zhengxiong Luo, Tongtong Ou, Guang Shi, Yichun Shi, Shiqi Sun, Yu Tian, Zhi Tian, Peng Wang, Rui Wang, Xun Wang, Ye Wang, Guofeng Wu, Jie Wu, Wenxu Wu, Yonghui Wu, Xin Xia, Xuefeng Xiao, Shuang Xu, Xin Yan, Ceyuan Yang, Jianchao Yang, Zhonghua Zhai, Chenlin Zhang, Heng Zhang, Qi Zhang, Xinyu Zhang, Yuwei Zhang, Shijia Zhao, Wenliang Zhao, and Wenjia Zhu. Seedream 4.0: Toward next-generation multimodal image generation, 2025. 3
- [42] Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman.

- Emu edit: Precise image editing via recognition and generation tasks. In CVPR, 2023. 3
- [43] Uriel Singer, Amit Zohar, Yuval Kirstain, Shelly Sheynin, Adam Polyak, Devi Parikh, and Yaniv Taigman. EVE: Video editing via factorized diffusion distillation. ECCV, 2024. 3
- [44] Decart Team. Lucy edit: Open-weight text-guided video editing. https://d2drjpuinn46lb.cloudfront. net / Lucy _ Edit _ _High _ Fidelity _ Text _ Guided_Video_Editing.pdf, 2025. 3, 7, 17
- [45] Genmo Team. Mochi 1. https://github.com/ genmoai/models, 2024. 3
- [46] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ram´e, Morgane Rivi`ere, et al. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025. 13
- [47] OpenAI Team. GPT-4o system card, 2024. 3, 6, 12
- [48] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models, 2025. 3, 5, 6, 13, 15
- [49] Wen Wang, Yan Jiang, Kangyang Xie, Zide Liu, Hao Chen, Yue Cao, Xinlong Wang, and Chunhua Shen. Zero-shot video editing using off-the-shelf image diffusion models. arXiv, 2023. 3
- [50] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942, 2023. 6
- [51] Yuhan Wang, Siwei Yang, Bingchen Zhao, Letian Zhang, Qing Liu, Yuyin Zhou, and Cihang Xie. GPT-IMAGE-EDIT1.5M: A million-scale, GPT-generated image dataset, 2025. 3, 4, 6
- [52] Cong Wei, Zheyang Xiong, Weiming Ren, Xinrun Du, Ge Zhang, and Wenhu Chen. OmniEdit: Building image editing generalist models through specialist supervision. In ICLR,

2025. 3, 6

- [53] Yuhui Wu, Liyi Chen, Ruibin Li, Shihao Wang, Chenxi Xie, and Lei Zhang. InsViE-1M: Effective instruction-based video editing with elaborate dataset construction. ICCV,

2025. 1, 2, 3, 7

- [54] Ziyi Wu, Aliaksandr Siarohin, Willi Menapace, Ivan Skorokhodov, Yuwei Fang, Varnith Chordia, Igor Gilitschenski, and Sergey Tulyakov. Mind the time: Temporallycontrolled multi-event video generation. In Proceedings of

- the Computer Vision and Pattern Recognition Conference, pages 23989–24000, 2025. 6
- [55] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. OmniGen: Unified image generation. In CVPR, 2025. 3
- [56] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv,

2025. 6, 13

- [57] Ling Yang, Bohan Zeng, Jiaming Liu, Hong Li, Minghao Xu, Wentao Zhang, and Shuicheng Yan. EditWorld: Simulating world dynamics for instruction-following image editing. arXiv, 2024. 3
- [58] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. CogVideoX: Text-to-video diffusion models with an expert transformer. ICLR, 2025. 3
- [59] Danah Yatim, Rafail Fridman, Omer Bar-Tal, Yoni Kasten, and Tali Dekel. Space-time diffusion features for zero-shot text-driven motion transfer. In CVPR, 2024. 7
- [60] Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025. 10, 11
- [61] Zixuan Ye, Xuanhua He, Quande Liu, Qiulin Wang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, Qifeng Chen, and Wenhan Luo. UNIC: Unified in-context video editing, 2025. 3
- [62] Jaehong Yoon, Shoubin Yu, and Mohit Bansal. RACCooN: A versatile instructional video editing framework with autogenerated narratives. EMNLP, 2025. 3
- [63] Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. AnyEdit: Mastering unified highquality image editing for any idea. In CVPR, 2025. 3
- [64] Shoubin Yu, Difan Liu, Ziqiao Ma, Yicong Hong, Yang Zhou, Hao Tan, Joyce Chai, and Mohit Bansal. VEGGIE: Instructional editing and reasoning video concepts with grounded generation. ICCV, 2025. 1, 2, 3
- [65] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. MagicBrush: A manually annotated dataset for instructionguided image editing. In NeurIPS, 2023. 3
- [66] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models, 2023. 3
- [67] Youcai Zhang, Xinyu Huang, Jinyu Ma, Zhaoyang Li, Zhaochuan Luo, Yanchun Xie, Yuzhuo Qin, Tong Luo, Yaqian Li, Shilong Liu, et al. Recognize anything: A strong image tagging model. arXiv, 2023. 5
- [68] Zhenghao Zhang, Zuozhuo Dai, Long Qin, and Weizhi Wang. EffiVED: Efficient video editing via text-instruction diffusion models, 2024. 3
- [69] Zechuan Zhang, Ji Xie, Yu Lu, Zongxin Yang, and Yi Yang. In-Context Edit: Enabling instructional image editing with in-context generation in large-scale diffusion transformers. NeurIPS, 2025. 3

- [70] Haozhe Zhao, Xiaojian Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. UltraEdit: Instruction-based fine-grained image editing at scale. In NeurIPS, 2024. 3, 6
- [71] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model, 2024. 3
- [72] Bojia Zi, Weixuan Peng, Xianbiao Qi, Jianan Wang, Shihao Zhao, Rong Xiao, and Kam-Fai Wong. MiniMax-Remover: Taming bad noise helps video object removal. arXiv, 2025. 1, 5
- [73] Bojia Zi, Penghui Ruan, Marco Chen, Xianbiao Qi, Shaozhe Hao, Shihao Zhao, Youze Huang, Bin Liang, Rong Xiao, and Kam-Fai Wong. Se˜norita-2M: A high-quality instructionbased dataset for general video editing by video specialists. In NeurIPS, 2025. 1, 2, 3, 4, 7

