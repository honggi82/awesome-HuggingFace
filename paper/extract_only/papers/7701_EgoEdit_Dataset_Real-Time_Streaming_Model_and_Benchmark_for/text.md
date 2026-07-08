## arXiv:2512.06065v1[cs.CV]5Dec2025

### EgoEdit: Dataset, Real-Time Streaming Model, and Benchmark for Egocentric Video Editing

Runjia Li1,3,† Moayed Haji Ali1,2 Ashkan Mirzaei1 Chaoyang Wang1 Arpit Sahni1 Ivan Skorokhodov1 Aliaksandr Siarohin1 Tomas Jakab3 Junlin Han3 Sergey Tulyakov1 Philip Torr3 Willi Menapace1

1Snap Research 2Rice University 3University of Oxford

snap-research.github.io/EgoEdit

[Figure 1]

[Figure 2]

Source Target

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

...

Categories

Substitution

Insertion Removal

... ...

Streaming Inputs

Video2Pose

Combined

Reasoning

Stylization ...

Depth2Video Video2Depth

Transform the bowl [...] into an oval woven wicker basket containing a colorful assortment of fruit [...]

[Figure 7]

[Figure 8]

Add a golden retriever and a ball in the hand

[Figure 9]

[Figure 10]

Real-Time Editing Model

Model

Metamorphose the low-profile tire with scuffed sidewalls into an iron lantern with a dark, weathered metal frame and decorative filigree panels [...]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Transform screwdriver into a feather quill [...] Replace the electrical cabinet with a technician's workbench [...] gothic illustration style [...]

Real-Time Outputs

[Figure 15]

[Figure 16]

Scoring

VLM Video Quality Text Alignment Temporal Consistency

855 ms [ms]

[...] morph the hot-glue gun with protruding glue stick into a vial of phosphorescent moss and spores [...]

Dataset

Model Benchmark

Figure 1. We propose a framework for real-time egocentric video editing. Our system is composed of: EgoEditData, a manually curated dataset of 100k video editing pairs focusing on the egocentric case and featuring object substitution and removal under challenging hand occlusions, interactions, and large egomotion; EgoEdit, the first real-time autoregressive model for egocentric video editing running in real time on a single H100 with 855ms first-frame latency and enabling live augmented reality (AR) interactions; EgoEditBench, a comprehensive benchmark for evaluation of egocentric video editing systems.

#### Abstract

We study instruction-guided editing of egocentric videos for interactive AR applications. While recent AI video editors perform well on third-person footage, egocentric views present unique challenges — including rapid egomotion and frequent hand–object interactions — that create a significant domain gap. Moreover, existing offline editing pipelines suffer from high latency, limiting real-time interaction. To address these issues, we present a complete ecosystem for egocentric video editing. First, we construct EgoEditData, a carefully designed and manually curated dataset specifically designed for egocentric editing scenarios, featuring rich hand-object interactions, while explicitly preserving hands. Second, we develop EgoEdit, an instruction-following egocentric video editor that supports real-time streaming inference on a single GPU. Finally, we introduce EgoEditBench, an evaluation suite targeting in-

†Work done while interning at Snap Inc.

struction faithfulness, hand and interaction preservation, and temporal stability under egomotion. Across both egocentric and general editing tasks, EgoEdit produces temporally stable, instruction-faithful results with interactive latency. It achieves clear gains on egocentric editing benchmarks—where existing methods struggle—while maintaining performance comparable to the strongest baselines on general editing tasks. EgoEditData and EgoEditBench will be made public for the research community.

#### 1. Introduction

Altering the perceived world is central to augmented reality (AR). This technology empowers real-time experiences that immerse the user into new worlds, transform the surrounding environment, create virtual characters and objects, and let users interact with these elements. However, traditional AR experiences rely on graphics pipelines and significant expert effort to handcraft each application, tying the poten-

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Input Real Time Result Input Real Time Result

Performance Across Edit Tasks

Vlm Eval

Transform the banana into a water gun.

Transform the hand into a robot hand.

Change Color

Change Camera Pose

|[Figure 25]<br><br>Change<br><br>Change Object|[Figure 26]<br><br>Replace the paper<br><br>Material|
|---|---|

|[Figure 27]<br><br>with a flying squirrel.<br><br>Change|[Figure 28]<br><br>Background|
|---|---|

|[Figure 29]<br><br>Add|[Figure 30]<br><br>a leash in the hand. A dog is|
|---|---|

|[Figure 31]<br><br>attached to the leash and follo|[Figure 32]<br><br>ws.|
|---|---|

cups

|[Figure 33]<br><br>Add an orang<br><br>Change Weather|[Figure 34]<br><br>e cat that jumps to the wooden|
|---|---|

|[Figure 35]<br><br>table and then jumps to the|[Figure 36]<br><br>left armchair.<br><br>Add Object|
|---|---|

|[Figure 37]|[Figure 38]<br><br>Transform the|
|---|---|

|[Figure 39]<br><br>banana into a fish.|[Figure 40]|
|---|---|

n

ba

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Add Effect

|[Figure 45]|[Figure 46]<br><br>Transform the umbrella|
|---|---|

|[Figure 47]<br><br>a glowing energy sword.|[Figure 48]|
|---|---|

0 1 2 3 4 5 6 7 8 9

Combined Tasks

Replace the floor with an ocean with tall waves.

into

Stylization

Figure 2. In-the-wild video edits produced in real time by EgoEdit’s streaming variant EgoEdit-RT on a single H100 GPU. The model demonstrates strong generalization to out-of-distribution scenarios, producing compelling real-time results suitable for immersive AR experiences. Additional results are presented in Appx. E and the website.

Depth2Video

Sketch2Video

Pose2Video

Remove Object

Reasoning

11/14/25, 4:17 AM vlm_eval_radar_chart_v2.html

11/14/25, 4:18 AM vlm_eval_radar_chart.html

|AnyV2V EditVerse Lucy_Edit StreamDiffusionV2 TokenFlow EgoEdit (Ours) EgoEdit-RT (Ours)<br><br>|
|---|

ability of current editors [57] and the usefulness of existing datasets [66, 76, 79]. Moreover, AR requires not only edit fidelity but also real-time, low-latency responses suitable for interaction. Many high-quality diffusion editing pipelines remain too slow for this setting.

Performance Across Edit Tasks

Performance Across Edit Tasks

Performance Across Edit Tasks

Vlm Eval

Vlm Eval

Vlm Eval

Change Color Change Material

Combined Tasks Depth2Video

Change Color Change Material

Change Camera Pose

Change Background

Change Camera Pose

Change Background

Change Object

Change Background

Change Object

Add Effect

Change Object

Add Object

Change Camera Pose

Add Object

Change Weather

Video2Pose

Change Weather

Add Effect

Add Object

Add Effect

0 1 2 3 4 5 6 7 8 9

0 1 2 3 4 5 6 7 8 9

0 1 2 3 4 5 6 7 8 9

Combined Tasks

Pose2Video

Combined Tasks

Stylization

Video2Sketch

Stylization

We address this gap by targeting the egocentric editing regime end-to-end through data, a real-time generative model, and an evaluation framework as shown in Figure 1.

Depth2Video

Reasoning

Depth2Video

Sketch2Video

Video2Depth

Sketch2Video

Pose2Video

Remove Object Sketch2Video Stylization

Pose2Video

Reasoning Remove Object

Remove Object

Reasoning

EgoEditBench

|AnyV2V EditVerse Lucy_EditEditVerseBenchStreamDiffusionV2 TokenFlow EgoEdit(Ours) EgoEdit-RT (Ours)|
|---|

Data. High-quality paired editing data showing “before/after” editing examples are main drivers of editing quality [26, 39, 54]. We therefore focus on EgoEditData, a manually curated video editing dataset tailored to egocentric scenarios. Our data generation pipeline emphasizes the tasks most relevant to AR, focusing on the challenging removal and substitution of objects under active hand-object interaction, while explicitly preserving hand structure. We automatically identify actively interacted objects at which to target edits to ensure edit relevance, and conduct multi-stage human review to enforce consistently high visual quality. To ensure high instruction alignment, each pair is accompanied by precise, descriptive edit instructions. The resulting EgoEditData dataset comprises 49.7k video samples and 99.7k instruction–edit pairs and will be released to support research in this domain.

- Figure 3. Comparison of EgoEdit and EgoEdit-RT against baselines according to VLM score on EgoEditBench and EditVerseBench [26]. EgoEdit and its real-time variant EgoEdit-RT achieve superior results on egocentric editing tasks and perform competitively with the strongest baselines on general editing tasks. EditVerse is excluded from EgoEditBench as source code is unavailable. Streaming models are indicated in dashed lines.

tial achievable by this technology to the amount of available expert labor. The rapid progress of text-conditioned image [6, 30, 31] and video [28, 59, 70] generation however raises a natural question: can instruction-guided editing [3, 26, 57, 60, 64] serve as a direct engine for AR, enabling users to add, remove, or modify scene elements with simple language while they interact with the world?

ﬁle:///Users/mali5/Downloads/vlm_eval_radar_chart.html 1/1

ﬁle:///Users/mali5/Downloads/vlm_eval_radar_chart_v2.html 1/1

Real-Time Model. Building on this data, we train an instruction-following egocentric editor, EgoEdit, from a large video generator. To meet the latency demands of interactive use, we apply Self-Forcing distillation [23] to obtain a generator that runs in real-time with low latency on a single H100 GPU, enabling responsive, user-in-the-loop editing as shown in Figure 2.

Editing is especially appealing because it addresses central tasks at the core of AR, such as object insertion and removal, environmental restyling, camera and lighting adjustments, while preserving the core content of a scene. However, most recent editors [26, 57] and their training corpora [79] are targeted at exocentric content: third-person views with moderate motion, and low amounts of interaction. However, in AR, the camera is first-person and constantly moving, hands frequently occlude and manipulate objects, and object interactions are complex. These characteristics produce a distribution shift that limits the reli-

Evaluation. To standardize assessment in this setting, we introduce EgoEditBench, an automated benchmark for egocentric video editing. The benchmark targets instruction faithfulness, preservation of hands and manipulated

objects, and temporal consistency under typical egocentric scenarios, providing a basis for reproducible comparisons.

EgoEdit delivers temporally stable, instruction-faithful edits, and latency suitable for interactive use. In comparisons against recent video-editing baselines [11, 17, 57] (see Figure 3), we observe consistent qualitative and quantitative improvements in the egocentric case, while performing closely to concurrent state-of-the-art editing methods [26] in the exocentric case. Ablations highlight the enabling role of EgoEditData in achieving such performance.

In summary, we present a complete ecosystem for egocentric video editing comprised of:

- • EgoEditData: the first high-quality, manually curated dataset for egocentric video editing with 49.7k videos and 99.7k instruction–edit pairs intended for public release.
- • EgoEdit: a real-time egocentric video editing model that enables interactive AR scenarios on a single H100 GPU at 38.1fps with a first-frame latency of 855ms.
- • EgoEditBench: a benchmark that standardizes evaluation for egocentric editing with a focus on complex hand interactions and large egomotion typical of AR use cases.

#### 2. Related Work

Image & Video Editing. Text-conditioned diffusion initially enabled editing without additional training. Inversionbased methods reconstruct the source along the denoising trajectory and then steer it with the edit prompt [16, 25, 35, 38, 43, 44, 48, 61]. Attention-control methods modify or reweight cross/self-attention to preserve content while changing appearance [5, 22, 48, 58]. While broadly applicable, these approaches remain brittle for large structural edits and long-range consistency. InstructPix2Pix [3] overcame these limitations by collecting paired “before/after” image edit data, and training a conditional editor that directly edits the source image. Because paired video edit data are hard to obtain, early trained video editing approaches sidestepped supervision by coupling image editors with video modules [40, 55]. InsV2V [11], however, brought the InstructPix2Pix recipe to video via synthetic video edit pairs. Recent image editors improve fidelity by scaling both data and models, making use of large curated edit corpora [39, 65] paired with either transformer backbones driven by MLLM embeddings [4, 39, 60, 64] or unified architectures that perform understanding and generation in one model for “in-context” editing [13, 34, 65]. Video editors follow similar paradigms, with in-context editors unifying conditioning and instructions along the sequence: UNIC [72] composes tasks via composite token sequences with task-aware RoPE and condition bias; EditVerse [26] improves unified image/video editing and generation through careful data curation. In contrast to image editors, sequence concatenation increases inference cost sig-

nificantly for video editors [26]. Lucy Edit [57] reduces the cost of long sequences by channel-wise conditioning while retaining strong source-video control. Recent works aim at training-free real-time video editing [17, 27, 33]. While showing promise, they currently suffer a quality gap with trained methods [26, 57, 72].

Across both images and videos, edit quality tracks the scale and quality of paired data and model capacity. This motivates the construction of our curated egocentric editing dataset, where edit examples focus on egocentric motion and complex hand/object interactions and are not available in existing video editing datasets.

Editing Datasets. Editing quality closely follows the scale and curation of paired “before/after” data. For images, InstructPix2Pix [3] established the data-first recipe with 313k CLIP-filtered pairs generated via Prompt-toPrompt. Since then, most large corpora follow a common pattern: task-specialized synthetic pipelines that generate at scale [18, 39, 54, 62, 63, 75], paired with multistage filtering using CLIP [18, 75], MLLMs [63, 75], object detectors [75], heuristics, and human-in-the-loop passes [18, 39, 62, 63, 75], resulting in million-scale filtered datasets. Quality-focused variants tighten these stages with heavier MLLM and human curation [39] or use existing high-quality editing methods as data generators [62]. Complementary, carefully curated sets show that domain alignment and quality can result in improvements despite smaller sizes [77, 78]. For videos, the same principles now dominate. InsV2V [11] pioneered synthetic video edit pairs, demonstrating that scaled paired training transfers to the temporal setting. Follow-ups are based on propagation of edited frames [66, 76] or the creation of specialized pipelines mirroring the image editing domain [79]. EditVerse [26] extensively filtered existing datasets [63, 75, 79] resulting in a curated set of 232K video-edit samples.

Streaming Video Generation. State-of-the-art video generators [28, 59, 70] yield high-quality but suffer from low throughput, long first-frame latency, and limited clip length. The denoising process is long, and the full video must be generated before the first frame can be shown to the user. Recent methods create autoregressive generators capable of generating long videos by predicting a chunk of frames at a time. Diffusion forcing [7] and its variants [1, 8, 32, 67] divide the video into chunks and assigning distinct diffusion noise levels so the model can denoise autoregressively chunk-by-chunk. Causal distillation converts slow bidirectional teachers into few-step causal students. CausVid [74] distills a 50-step bidirectional model into a 4-step causal student using DMD [73], with chunk-wise generation and KV caching. Self-Forcing [23] addresses the exposure bias by rolling out the student at train time, allowing the model to self-correct its errors. APT2 [36] employs adversarial post-training to achieve 1-NFE per frame for interactive

generation. Our approach follows the Self-Forcing strategy to achieve interactive egocentric video editing.

#### 3. EgoEditData

Compared to conventional video editing, egocentric editing brings distinct challenges due to complex hand-object interactions, frequent occlusions, and large egomotion. Existing editing datasets rarely cover egocentric scenes or such intricate interactions, creating data barriers [77] for learningbased AR experiences. To address data scarcity and provide a foundation for egocentric editing, we present EgoEditData, a manually curated and high-quality egocentric video editing dataset, focusing on rich hand-object interactions.

###### 3.1. Data Curation Pipeline

Because data quality [39, 54, 78] and domain alignment [77] are main drivers of editing performance, our pipeline prioritizes quality over quantity through strict filtering, and emphasizes creation of editing pairs that depict the most challenging egocentric scenarios with rich handobject interaction. Starting from real egocentric videos [19, 20], we ensure that: (1) videos contain an object that is actively manipulated by the egocentric subject, (2) the edit target is the manipulated object, (3) the synthetic videos preserve realistic hand motion while reflecting the intended content change, (4) the instruction prompts are descriptive and accurate. The stages are detailed below, with additional implementation specifics in Appx. A.1.

Video selection. We consider videos from the Ego4D [19] and EgoExo4D [20] datasets. From Ego4D we select sequences coming from high-quality camera models only (see Appx. A.1). For EgoExo4D, we select egocentric cameras, and perform rectification of the videos. We also conduct filtering to reduce jittering and motion blur. 1.8% of videos are retained after this stage.

Hand mask segmentation. We first detect hands in each frame using a hand detection method [47]. Videos without visible hands are removed. For the remaining videos, detected hand regions provide visual prompts to SAM 2 [52], which yields fine-grained and temporally consistent hand masks across the sequence. We conduct human filtering to ensure hand masks are annotated correctly, with 49.6% of samples remained after filtering.

Object names extraction. We then identify the object that the hands manipulate using a vision language model. Qwen2.5-VL-32B [2] is prompted to name the interacted objects for each video. Videos where no meaningful handobject interaction is found are discarded.

Object mask segmentation. Given the identified object name, Grounded SAM [53] predicts an approximate object mask in each frame. Videos with low mask confidence in every frame are removed. To confirm real interaction, we compute the edge distance between the hand and object

masks and the distances between the object mask and hand skeleton keypoints [47], and filter out false positives. The coarse grounded masks then seed SAM 2 [52] to obtain finegrained and temporally consistent object masks. We conduct manual filtering to ensure object masks are extracted correctly, retaining 43.6% of the sequences. Curation at this stage ensures the expensive successive object editing stage is only run on correctly processed videos.

Object Editing. This stage creates target videos where the original object is replaced with a different one or removed entirely. For each segment, we prompt GPT-5 Mini [45] to propose diverse target objects for substitution, including both ordinary and imaginary items. Qwen-Image [64] then synthesizes a reference image for each proposal. Next, GPT-5 Mini [45] produces a scene-level description of the video assuming interaction with the target object. We feed the reference image, the scene-level prompt, and the object mask to Wan 2.1 VACE 14B [24] to generate the edited video. Object removal is treated as a special case with no target object. Although the conditioning is rich and the computation is heavy (0.112 fps on 8 H100 GPUs), Wan

- 2.1 VACE 14B [24] yields only a small fraction of results that meet our standard for dataset quality. Human annotators therefore remove imperfect results to ensure consistent quality, leaving 37.8% of the generated edits. Editing pairs construction. We form video editing pairs that include a source video, a target video, and a natural language instruction. For each real video and its edited variants, we permute pairs among all versions, including the original clip, and prompt GPT-5 Mini [45] to generate a precise and faithful description of the edit [62].
- 3.2. Statistics

After stages of curation and filtering, only 0.4% of the original videos from Ego4D and EgoExo4D are kept. The resulting pairs compose our EgoEditData dataset. It is composed of 10.9k original and 38.8k synthetic videos (70 hours long), for an average of 3.6 synthetic videos per original video, yielding a total of 99.7k editing pairs, each comprising a source video, target video, and editing instruction, with 93,422 of them derived from Ego4D [19] and 6,237 pairs from EgoExo4D [20]. EgoEditData is diverse. Additional details are presented in Appx. A.

- 4. EgoEdit

Consider a source video Xsrc and a textual instruction c specifying a desired edit. The goal is to produce a target video Xtgt reflecting the requested change. We focus on egocentric video editing. Unlike the traditional exocentric setting, which can be processed offline,

egocentric editing requires real-time results to support interactive experiences.

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

"Make the cutting board a frozen clock"

| | |
|---|---|
|Text Em|bedder|
| | |

|[Figure 55]|[Figure 56]| | |[Figure 57]|
|---|---|---|---|---|

Unpatchify

Patchify

DiT

- Figure 4. Architecture of EgoEdit. EgoEdit extends a video generation DiT model for video editing by performing channel-wise concatenation of the source and noisy target video inputs, avoiding the computational overheads of sequence-wise concatenation.

To address these challenges, we introduce EgoEdit, a video editing method tailored to the real-time egocentric setting. We first convert a pretrained video generator into a video editor by adding source video conditioning and finetuning on an editing corpus that includes EgoEditData.

Then, we distill the editor into an autoregressive realtime generator using bidirectional DMD [73] and autoregressive Self Forcing [23]. Section 4.1 introduces the Flow Matching framework, Section 4.2 describes the model, and Section 4.3 details the distillation procedure.

###### 4.1. Preliminaries: Flow Matching

We train our generators with Rectified Flow flow matching [37, 41], which learns a deterministic path from a noise distribution pn to the data distribution pd. Let X1 ∼ pd and X0 ∼ pn = N(0,I). We define a linear path Xt = (1−t)X0+tX1 for t ∈ [0,1], whose ground truth velocity is constant along the path, vt = dX

dt = X1 − X0. A neural network G(·) predicts the velocity from a noised input and a time value and is trained by minimizing:

t

t,X1∼pd,X0∼pn G(Xt, t)−(X1−X0) 22, (1)

LRF = Et∼p

where pt is a training distribution over t chosen as a logitnormal [15]. At inference time, an Euler solver integrates the learned ODE from X0 to X1 to produce a sample.

###### 4.2. Architecture

We base EgoEdit on a pretrained text-to-video generator trained on the latent space of a Wan 2.1 autoencoder [59] with a transformer backbone [46] (see Section B.1). As shown in Figure 4, the model receives Xt at time t, projects it to a sequence of tokens through a linear patchifier, processes the tokens with transformer blocks, and maps them to the predicted velocity vˆ through a final linear head. Text conditions c are provided through cross attention layers placed after each self attention block. The computation is expressed as vˆ = G(Xt | c).

We adapt the model to editing by replacing Xt with the noisy target video Xtgtt and by conditioning on the source video Xsrc, written as vˆ = G(Xtgtt | Xsrc;c). We consider

Recording

|[Figure 58]| |
|---|---|
| | |

[Figure 59]

[Figure 60]

[Figure 61]

AutoregressiveDenoising

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

##### DiT

[Figure 73]

[Figure 74]

[Figure 75]

Streaming Video Editing

Figure 5. Inference of EgoEdit. EgoEdit performs inference in a streaming fashion. A camera continuously acquires video sequences which are edited by the model in a chunk-by-chunk manner so that the edited video can be served to the user in a watch-asyou-generate fashion. Each blue arrow represents a model forward pass on a single video chunk for the case of a 3 steps model.

two main strategies to inject Xsrc. Sequencewise concatenation [26, 72] patchifies the source and concatenates its patches with those of the target along the sequence dimension. This approach is common [26, 64], but the longer token sequence increases the cost of self attention quadratically, which conflicts with real-time low latency operation. To avoid this growth, EgoEdit uses channel-wise concatenation [57], where Xsrc and Xtgtt are concatenated along channels before patchification, which keeps the cost close to the base model.

###### 4.3. Distillation Procedure

The edited model described above is accurate but slow at inference. 40 denoising steps with classifier-free guidance are required to produce a video, which corresponds to 80 model invocations (NFEs). In addition, the model generates the full clip at once, which delays the first visible frame until the process finishes. Interactive use calls for autoregressive generation with low latency, as illustrated in Figure 5. We therefore distill the editor into a real-time autoregressive model in two phases (see Appx. B.3).

Bidirectional DMD distillation. We follow DMD [73] to compress the 40-step model with classifier-free guidance into a 4-step model with distilled guidance. This reduces the NFEs from 80 to 4, facilitating subsequent distillation.

Self Forcing. Self Forcing [23] runs the causal model autoregressively on video streams and applies a DMD loss with score models based on the bidirectional teacher. The model learns to correct its own errors, which reduces exposure bias and enables low latency autoregressive inference. To minimize exposure bias while achieving low latency, we generate a chunk at a time, where each chunk is composed

Offline Models Streaming Models

Input EgoEdit Lucy Edit InsV2V Senorita-2M EgoEdit-RT StreamDiffusionV2 StreamDiffusion

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

|[Figure 84]|[Figure 85]|
|---|---|

|[Figure 86]|[Figure 87]|[Figure 88]|[Figure 89]|
|---|---|---|---|

|[Figure 90]|[Figure 91]|
|---|---|

|[Figure 92]|[Figure 93]|
|---|---|

|[Figure 94]|[Figure 95]|
|---|---|

|[Figure 96]|[Figure 97]|
|---|---|

|[Figure 98]|[Figure 99]|
|---|---|

Replace the white aida fabric with a carved oak folding mask featuring flame-like scrollwork [...] Change the background to a bright sewing nook [...] apply holographic interface overlays to the scene.

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

|[Figure 106]|
|---|

|[Figure 107]|
|---|

|[Figure 108]|[Figure 109]|
|---|---|

|[Figure 110]|[Figure 111]|[Figure 112]|[Figure 113]|
|---|---|---|---|

|[Figure 114]|[Figure 115]|
|---|---|

|[Figure 116]|[Figure 117]|
|---|---|

|[Figure 118]|[Figure 119]|
|---|---|

|[Figure 120]|[Figure 121]|
|---|---|

|[Figure 122]|[Figure 123]|
|---|---|

Turn the depth map into a video [...] the egocentric subject draws on a large white sketchpad resting on a wooden table.[...] trace bold outlines, filling paw prints and lettering "DOGS" while sketching dog silhouettes. [...]

|[Figure 124]|
|---|

|[Figure 125]|
|---|

|[Figure 126]|
|---|

|[Figure 127]|
|---|

|[Figure 128]|
|---|

|[Figure 129]|
|---|

|[Figure 130]|
|---|

|[Figure 131]|
|---|

|[Figure 132]|[Figure 133]|
|---|---|

|[Figure 134]|[Figure 135]|[Figure 136]|[Figure 137]|
|---|---|---|---|

|[Figure 138]|[Figure 139]|
|---|---|

|[Figure 140]|[Figure 141]|
|---|---|

|[Figure 142]|[Figure 143]|
|---|---|

|[Figure 144]|[Figure 145]|
|---|---|

|[Figure 146]|[Figure 147]|
|---|---|

Morph the turquoise gingham with white aida strip into a mask overgrown with phosphorescent fungi [...] it emits a teal‑green/cyan glow from the eyes and fungal tips with purple accents [...]

Figure 6. Qualitative comparison of EgoEdit and its real time streaming variant EgoEdit-RT against baselines on EgoEditBench. EgoEdit and EgoEdit-RT consistently perform better than their baselines. Note that Se˜norita-2M uses the first frame from EgoEdit for frame propagation. Additional results are presented in Appx. E and the website.

of three latent frames. Note that the employed Wan [59] autoencoder natively supports autoregressive operation.

Table 2 shows latency and throughput for the different model variants. Our Self Forcing distilled model can produce results in real-time at 38.1fps with a latency of 855ms for displaying the first frame on a single H100 GPU.

###### 4.4. EgoEditBench

Existing video editing benchmarks [10, 26, 56] are primarily built on third-person natural videos, making them unsuitable for evaluating egocentric video editing. To properly assess this setting, we follow [26] to construct EgoEditBench, a benchmark designed to evaluate editing performance across 15 egocentric tasks [26], including: Add Object, Add Effect, Remove Object, Change Object, Change Background, Change Camera Pose, Stylization, Reasoning, Depth-to-Video, Sketch-to-Video, Pose-to-Video, Video-toPose, Video-to-Sketch, Video-to-Depth, and a Combined Task of all previous tasks. To build this benchmark, we sample 100 unique source videos from a split of the Ego4D dataset [19] that was not used in constructing EgoEditData, ensuring maximum diversity. This is achieved by first extracting source-object names following the EgoEditData pipeline, and computing BERT [14] embeddings from the concatenation of each source-object name and its corresponding scene description. We then perform K-means clustering with 10 centroids, and select 10 samples per cluster, resulting in 100 diverse source videos. Conditioned on the source video, its caption, and the source object, we prompt GPT-5 [45] to generate task-specific instruction prompts for each of the 15 tasks. For the X-to-Video tasks,

we synthesize conditioning signals as follows: Canny edge maps using OpenCV for the Sketch2Video task, 2D poses using DWpose [69] for Pose2Video, and depth maps using Depth Anything [68] for Depth2Video. For Add Object and Remove Object, we sample 50 unique source videos per task and construct the corresponding instruction prompts following EgoEditData. For the Change Object task, we generate four instruction prompts for each unique source video with two focusing on object replacement and two combining object replacement with an added effect.

In total, EgoEditBench comprises 1700 source videos paired with instruction prompts, covering 15 diverse egocentric editing tasks. Evaluation is performed according to the EditVerseBench [26] protocol and metrics, with results averaged per task to ensure equal weighting across all tasks.

Additional implementation details for EgoEditBench can be found in Appx. C.

#### 5. Experiments

###### 5.1. Experimental Setup

Training details. We finetune our pretrained video generation model on EgoEditData and a corpus of additional 1.31M video and 3.5M image editing pairs to obtain the base video editing model. We finetune the model with a total batch size of 96 for 30k iterations using an AdamW [42] optimizer with lr 1e-5, weight decay of 0.1 and exponential moving average. Bidirectional DMD distillation is performed for 4.5k steps using an AdamW [42] optimizer with lr of 1e-6 for the model and 4e-7 for the critic, weight decay of 0.1 and exponential moving average. Successively,

EgoEditBench EditVerseBench [26]

Method Family

VLM ↑ PS ↑ TA ↑ TC ↑ VLM ↑ PS ↑ TA ↑ TC ↑ TokenFlow [49]

4.99 18.91 15.89 95.04 5.87 19.90 23.68 98.21 STDF [71] 4.59 18.69 15.64 93.96 6.64 19.54 24.33 96.96 Se˜norita-2M‡ [79]

Attention manipulation

7.52 18.85 16.25 95.86 6.99 19.32 23.07 98.33 AnyV2V‡ [29] 6.72 18.65 15.35 92.37 6.46 19.47 23.32 95.91 InsV2V [11]

First-frame propagation

5.24 18.81 14.92 94.01 5.71 19.08 22.49 96.39 Lucy Edit [57] 5.44 18.87 15.03 94.41 6.27 19.23 22.55 98.62 EditVerse† [26] — — — — 8.26 19.69 25.29 98.68 EgoEdit (ours) 7.76 19.21 16.89 96.70 8.00 19.61 24.40 98.54 StreamDiffusion [27]

Instruction-guided

4.32 18.92 14.15 86.83 4.33 18.76 19.01 93.41 StreamDiffusionV2 [17] 2.55 18.63 12.75 94.31 2.78 18.45 17.32 98.22 EgoEdit-RT (ours) 7.71 19.13 16.34 96.41 8.18 19.59 17.61 98.55

Streaming models

- Table 1. Quantitative comparison of the baseline models and our method on EgoEditBench and EditVerseBench benchmarks: “VLM” is VLM evaluation score, “PS” is Pick Score, “TA” is Text Alignment, “TC” is Temporal Consistency. Reference-based editing tasks from EditVerseBench—propagation, inpainting, reference insertion, and edit with mask—were excluded. “†” indicates closed-source models evaluated using their publicly released samples; “‡” indicates models utilizing the first frame generated by EgoEdit. EgoEdit-RT stands for the real-time streaming version of EgoEdit.

Self Forcing training is conducted for 3.5k steps using an AdamW [42] optimizer with lr of 1e-6 for the model and 4e7 for the critic, weight decay of 0.1 and exponential moving average, producing the final checkpoint. Additional training and dataset details are presented in the Appx. B.2.

Evaluation metrics and protocol. EgoEditBench serves as the main framework for evaluating egocentric video editing quality. Evaluation is supplemented by EditVerseBench [26] to provide references on the nonegocentric case. As our model does not support reference image conditioning, we remove the EditVerseBench tasks of Propagation, Inpainting, Reference Insertion, and Edit with Mask, which require such conditioning.

###### 5.2. Comparison to Baselines

Baselines selection. We select a range of baselines against which to compare that are based on attention manipulation [49, 71], propagation of the first frame [29, 79], or direct instruction-guided video to video translation [11, 26, 57]. Frame propagation methods [29, 79] receive as input the first frame edited by EgoEdit for fair comparison. Additionally, we select StreamDiffusion [17, 27] as representatives for real-time editing methods. For a fair comparison, all baseline models are evaluated using their recommended inference settings, including the number of sampling steps, guidance scale, resolution, and frame rate (following EditVerseBench [26]). For inversion-based methods that require a target video caption and cannot process natural instruction prompts, we use GPT-5 [45] to generate the target caption from the source prompt and instruction prompt. Complete inference details for each baseline are provided in Appx. D.1.

Results. As shown in Table 1 and Figure 6, EgoEdit produces state-of-the-art results on egocentric videos while still performing strongly on general editing, as demonstrated by EditVerseBench performance. In particular, on the challenging egocentric setting, our method drops only 0.24 points in VLM evaluation when switching from general editing tasks to egocentric ones, while Lucy Edit [57] and InsV2V drop respectively 0.83 and 0.47 points. Only Se˜norita-2M [79] and AnyV2V [29] retain their performance due to receiving the propagated first frame from EgoEdit. When compared to existing real-time streaming editors, our streaming variant EgoEdit-RT achieves markedly stronger performance across both benchmarks. Relative to the bidirectional full model, EgoEdit-RT delivers comparable results on all quantitative metrics. These findings highlight the strong real-time editing capabilities of EgoEdit-RT and validate the effectiveness of our distillation procedure. Additional qualitative results are provided in the supplementary material and the website.

###### 5.3. Ablations

We conduct ablations to investigate performance change during distillation and the effectiveness of EgoEditData.

Distillation. Table 2 compares the original non-distilled EgoEdit model to the one obtained after 4-step DMD distillation and the final real-time autoregressive generator obtained after Self Forcing training. We take first chunk latency as the main metric for assessing real-time method suitability. It shows the delay from the moment the user hits the “record” button on their camera to the moment the corresponding first edited frame is visualized on the screen. Due to the inability of standard methods to perform

###### No Distill. DMD [73] Self Forcing [23]

Is streaming? × × ✓ NFEs 80 4 4 VLM-Eval ↑ 7.76 7.31 7.71 First chunk size 81 frames 81 frames 9 frames Next chunk size N/A N/A 12 frames

First Chunk Latency [ms]

Recording 5062 5062 562 AE 1520 1520 217 Model 6850 343 75.7 Total 13432 6925 855

Throughput [fps]

Model 11.9 237 134 Model + AE 9.68 43.5 38.1

- Table 2. EgoEditBench VLM score, latency and throughput analysis of different distilled EgoEdit models on 1×H100 under resolution of 512×384px. We consider latency involved in recording the source video, running EgoEdit, and running the autoencoder (AE) for source video encoding and generated video decoding.

chunk-by-chunk generation, only the Self Forcing variant achieves sub-second latency compatible with interactive usage. When evaluated on EgoEditBench, the Self Forcing variant reaches comparable VLM scores to the bidirectional teacher model, while enabling interactive generation.

Contribution of EgoEditData. To better quantify how EgoEditData improves a model’s ability to adapt to egocentric editing, we vary the number of videos of EgoEditData included during training. Starting from the same textto-video checkpoint, we finetune different models for 10k iterations on our training editing data corpora, removing a certain percentage of original videos and corresponding edited versions from EgoEditData in a proportion from 0% to 100%. As shown in Table 3, performance on EgoEditBench steadily improves as more samples from EgoEditData are incorporated, highlighting the role of EgoEditData in enabling robust egocentric video editing. We also discuss how EgoEditData influences general editing performance on EditVerse in Appx. D.

% of EgoEditData 0% 25% 75% 100% VLM Evaluation↑ 4.87 7.12 7.52 7.85

- Table 3. Performance of EgoEdit when trained using progressively smaller subsets of EgoEditData. A trend is visible: the model performs better with more egocentric editing data included during training. Note that these results differ from Table 1 because all models are evaluated at the 10k-iteration checkpoint. Additional details are provided in Appx. D.

###### 5.4. In-the-Wild Evaluation

We conduct in-the-wild evaluation of the real-time version of EgoEdit to test robustness in the real-world usage and emerging capabilities. Results are presented in Figure 2. We observe that the model is able to perform complex editing tasks such as correctly preserving hands during interactions, modeling environment interactions such as the sidewalk becoming wet when sprayed with an imaginary water gun, modeling lighting effects induced by inserted objects, replacing fast-moving thrown objects, inserting animals that realistically interact with the environment by jumping over or navigating around objects, and correctly rotating inserted objects according to the substituted object orientation. Most excitingly, some instances of inserted objects react to user interactions, such as dogs being walked on a leash. We observe, however, that the amount of structural modifications that inserted objects can operate on the surrounding environment is limited: swords will not cut through furniture, and animals will not move real objects.

#### 6. Discussion

Limitations. While EgoEdit-RT performs comparably to EgoEdit on automated benchmarks and presents strong in the wild generation capabilities, we notice a qualitative gap which manifests as: (i) lower proficiency in out-ofdistribution editing instructions, (ii) less robust performance when editing objects becoming temporarily occluded, (iii) lower temporal consistency. EgoEdit possesses a first-frame latency of 855ms, which is sufficient but suboptimal for interactive usage. As shown in Table 2, such latency is dominated by the recording time of the first chunk of 3 latent frames, corresponding to 9 RGB frames. Further reduction in latency can be achieved by lowering the chunk size during Self Forcing training. Finally, EgoEdit operates at a resolution of 512×384px and a frame rate of 16 fps, slightly lower than the common 480p resolution.

Conclusions. We introduce a comprehensive framework for developing learning-based AR applications through egocentric video editing. To support this effort, we construct and publicly release EgoEditData, the first highquality and manually curated dataset of egocentric video edits, containing 99.7k editing pairs across 49.7k unique videos. Building on this dataset, we propose EgoEdit, the first real-time video editing model tailored to egocentric applications. Our model demonstrates strong generalization to in-the-wild videos, achieves state-of-the-art performance on egocentric editing tasks, and competitive results on general video editing benchmarks. Finally, we introduce EgoEditBench, a standardized benchmark designed to evaluate egocentric video editing performance under realistic hand–object interactions. Together, these contributions form a foundation and ecosystem for real-time, instructionguided AR generation, paving the way for future research in interactive generative systems for augmented reality.

#### References

- [1] Sand. ai, Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, W. Q. Zhang, Weifeng Luo, Xiaoyang Kang, Yuchen Sun, Yue Cao, Yunpeng Huang, Yutong Lin, Yuxin Fang, Zewei Tao, Zheng Zhang, Zhongshu Wang, Zixun Liu, Dai Shi, Guoli Su, Hanwen Sun, Hong Pan, Jie Wang, Jiexin Sheng, Min Cui, Min Hu, Ming Yan, Shucheng Yin, Siran Zhang, Tingting Liu, Xianping Yin, Xiaoyu Yang, Xin Song, Xuan Hu, Yankai Zhang, and Yuqiao Li. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025. 3
- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 4, 13
- [3] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. Instructpix2pix: Learning to follow image editing instructions. In CVPR, 2023. 2, 3
- [4] Qi Cai, Jingwen Chen, Yang Chen, Yehao Li, Fuchen Long, Yingwei Pan, Zhaofan Qiu, Yiheng Zhang, Fengbin Gao, Peihan Xu, Yimeng Wang, Kai Yu, Wenxuan Chen, Ziwei Feng, Zijian Gong, Jianzhuang Pan, Yi Peng, Rui Tian, Siyu Wang, Bo Zhao, Ting Yao, and Tao Mei. Hidream-i1: A high-efficient image generative foundation model with sparse diffusion transformer. arXiv preprint arXiv:2505.22705, 2025. 3
- [5] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. arXiv preprint arXiv:2304.08465, 2023. 3
- [6] Siyu Cao, Hangting Chen, Peng Chen, Yiji Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kipper Gong, Tianpeng Gu, Xiusen Gu, Tiankai Hang, Duojun Huang, Jie Jiang, Zhengkai Jiang, Weijie Kong, Changlin Li, Donghao Li, Junzhe Li, Xin Li, Yang Li, Zhenxi Li, Zhimin Li, Jiaxin Lin, Linus, Lucaz Liu, Shu Liu, Songtao Liu, Yu Liu, Yuhong Liu, Yanxin Long, Fanbin Lu, Qinglin Lu, Yuyang Peng, Yuanbo Peng, Xiangwei Shen, Yixuan Shi, Jiale Tao, Yangyu Tao, Qi Tian, Pengfei Wan, Chunyu Wang, Kai Wang, Lei Wang, Linqing Wang, Lucas Wang, Qixun Wang, Weiyan Wang, Hao Wen, Bing Wu, Jianbing Wu, Yue Wu, Senhao Xie, Fang Yang, Miles Yang, Xiaofeng Yang, Xuan Yang, Zhantao Yang, Jingmiao Yu, Zheng Yuan, Chao Zhang, Jian-Wei Zhang, Peizhen Zhang, Shi-Xue Zhang, Tao Zhang, Weigang Zhang, Yepeng Zhang, Yingfang Zhang, Zihao Zhang, Zijian Zhang, Penghao Zhao, Zhiyuan Zhao, Xuefei Zhe, Jianchen Zhu, and Zhao Zhong. Hunyuanimage 3.0 technical report. arXiv preprint arXiv:2509.23951, 2025. 2
- [7] Boyuan Chen, Diego Mart´ı Mons´o, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffu-

sion. In Neural Information Processing Systems (NeurIPS),

2024. 3

- [8] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, Weiming Xiong, Wei Wang, Nuo Pang, Kang Kang, Zhiheng Xu, Yuzhe Jin, Yupeng Liang, Yubing Song, Peng Zhao, Boyuan Xu, Di Qiu, Debang Li, Zhengcong Fei, Yang Li, and Yahui Zhou. Skyreelsv2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025. 3
- [9] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-sigma: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. arXiv preprint arXiv:2403.04692, 2024. 14
- [10] Yinan Chen, Jiangning Zhang, Teng Hu, Yuxiang Zeng, Zhucun Xue, Qingdong He, Chengjie Wang, Yong Liu, Xiaobin Hu, and Shuicheng Yan. Ivebench: Modern benchmark suite for instruction-guided video editing assessment. arXiv preprint arXiv:2510.11647, 2025. 6
- [11] Jiaxin Cheng, Tianjun Xiao, and Tong He. Consistent videoto-video transfer using synthetic dataset. In International Conference on Learning Representations (ICLR), 2024. 3, 7
- [12] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. Flashattention: Fast and memory-efficient exact attention with io-awareness. 2022. 14
- [13] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 3
- [14] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In North American Chapter of the Association for Computational Linguistics,

2019. 6

- [15] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. In Proceedings of the 41st International Conference on Machine Learning, pages 12606–12633. PMLR, 2024. 5, 14
- [16] Kunyu Feng, Yue Ma, Bingyuan Wang, Chenyang Qi, Haozhe Chen, Qifeng Chen, and Zeyu Wang. Dit4edit: diffusion transformer for image editing. In AAAI Conference on Artificial Intelligence, 2025. 3
- [17] Tianrui Feng, Zhi Li, Shuo Yang, Haocheng Xi, Muyang Li, Xiuyu Li, Lvmin Zhang, Keting Yang, Kelly Peng, Song Han, Maneesh Agrawala, Kurt Keutzer, Akio Kodaira, and Chenfeng Xu. Streamdiffusionv2: A streaming system for dynamic and interactive video generation. arXiv preprint arXiv:2511.07399, 2025. 3, 7, 17
- [18] Yuying Ge, Sijie Zhao, Chen Li, Yixiao Ge, and Ying Shan. Seed-data-edit technical report: A hybrid dataset for instructional image editing. arXiv preprint arXiv:2405.04007,

2024. 3

- [19] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, Miguel Martin, Tushar Nagarajan, Ilija Radosavovic, Santhosh Kumar Ramakrishnan, Fiona Ryan, Jayant Sharma, Michael Wray, Mengmeng Xu, Eric Zhongcong Xu, Chen Zhao, Siddhant Bansal, Dhruv Batra, Vincent Cartillier, Sean Crane, Tien Do, Morrie Doulaty, Akshay Erapalli, Christoph Feichtenhofer, Adriano Fragomeni, Qichen Fu, Abrham Gebreselasie, Cristina Gonz´alez, James Hillis, Xuhua Huang, Yifei Huang, Wenqi Jia, Weslie Khoo, J´achym Kol´aˇr, Satwik Kottur, Anurag Kumar, Federico Landini, Chao Li, Yanghao Li, Zhenqiang Li, Karttikeya Mangalam, Raghava Modhugu, Jonathan Munro, Tullie Murrell, Takumi Nishiyasu, Will Price, Paola Ruiz, Merey Ramazanova, Leda Sari, Kiran Somasundaram, Audrey Southerland, Yusuke Sugano, Ruijie Tao, Minh Vo, Yuchen Wang, Xindi Wu, Takuma Yagi, Ziwei Zhao, Yunyi Zhu, Pablo Arbel´aez, David Crandall, Dima Damen, Giovanni Maria Farinella, Christian Fuegen, Bernard Ghanem, Vamsi Krishna Ithapu, C. V. Jawahar, Hanbyul Joo, Kris Kitani, Haizhou Li, Richard Newcombe, Aude Oliva, Hyun Soo Park, James M. Rehg, Yoichi Sato, Jianbo Shi, Mike Zheng Shou, Antonio Torralba, Lorenzo Torresani, Mingfei Yan, and Jitendra Malik. Ego4d: Around the world in 3,000 hours of egocentric video. In Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 4, 6, 13, 14
- [20] Kristen Grauman, Andrew Westbury, Lorenzo Torresani, Kris Kitani, Jitendra Malik, Triantafyllos Afouras, Kumar Ashutosh, Vijay Baiyya, Siddhant Bansal, Bikram Boote, Eugene Byrne, Zach Chavis, Joya Chen, Feng Cheng, FuJen Chu, Sean Crane, Avijit Dasgupta, Jing Dong, Maria Escobar, Cristhian Forigua, Abrham Gebreselasie, Sanjay Haresh, Jing Huang, Md Mohaiminul Islam, Suyog Jain, Rawal Khirodkar, Devansh Kukreja, Kevin J Liang, JiaWei Liu, Sagnik Majumder, Yongsen Mao, Miguel Martin, Effrosyni Mavroudi, Tushar Nagarajan, Francesco Ragusa, Santhosh Kumar Ramakrishnan, Luigi Seminara, Arjun Somayazulu, Yale Song, Shan Su, Zihui Xue, Edward Zhang, Jinxu Zhang, Angela Castillo, Changan Chen, Xinzhu Fu, Ryosuke Furuta, Cristina Gonzalez, Prince Gupta, Jiabo Hu, Yifei Huang, Yiming Huang, Weslie Khoo, Anush Kumar, Robert Kuo, Sach Lakhavani, Miao Liu, Mi Luo, Zhengyi Luo, Brighid Meredith, Austin Miller, Oluwatumininu Oguntola, Xiaqing Pan, Penny Peng, Shraman Pramanick, Merey Ramazanova, Fiona Ryan, Wei Shan, Kiran Somasundaram, Chenan Song, Audrey Southerland, Masatoshi Tateno, Huiyu Wang, Yuchen Wang, Takuma Yagi, Mingfei Yan, Xitong Yang, Zecheng Yu, Shengxin Cindy Zha, Chen Zhao, Ziwei Zhao, Zhifan Zhu, Jeff Zhuo, Pablo Arbelaez, Gedas Bertasius, Dima Damen, Jakob Engel, Giovanni Maria Farinella, Antonino Furnari, Bernard Ghanem, Judy Hoffman, C.V. Jawahar, Richard Newcombe, Hyun Soo Park, James M. Rehg, Yoichi Sato, Manolis Savva, Jianbo Shi, Mike Zheng Shou, and Michael Wray. Ego-exo4d: Understanding skilled human activity from first- and thirdperson perspectives. In Computer Vision and Pattern Recognition (CVPR), 2024. 4, 13, 14

- [21] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity long video generation. arXiv, 2023. 14
- [22] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 3
- [23] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the traintest gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025. 2, 3, 5, 8, 15
- [24] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. arXiv preprint arXiv:2503.07598, 2025. 4, 13, 14
- [25] Xuan Ju, Ailing Zeng, Yuxuan Bian, Shaoteng Liu, and Qiang Xu. Pnp inversion: Boosting diffusion-based editing with 3 lines of code. In International Conference on Learning Representations (ICLR), 2024. 3
- [26] Xuan Ju, Tianyu Wang, Yuqian Zhou, He Zhang, Qing Liu, Nanxuan Zhao, Zhifei Zhang, Yijun Li, Yuanhao Cai, Shaoteng Liu, Daniil Pakhomov, Zhe Lin, Soo Ye Kim, and Qiang Xu. Editverse: Unifying image and video editing and generation with in-context learning. arXiv preprint arXiv:2509.20360, 2025. 2, 3, 5, 6, 7, 15, 16, 17, 19
- [27] Akio Kodaira, Chenfeng Xu, Toshiki Hazama, Takanori Yoshimoto, Kohei Ohno, Shogo Mitsuhori, Soichi Sugano, Hanying Cho, Zhijian Liu, Masayoshi Tomizuka, and Kurt Keutzer. Streamdiffusion: A pipeline-level solution for realtime interactive generation. In International Conference on Computer Vision (ICCV), 2025. 3, 7, 17
- [28] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, Kathrina Wu, Qin Lin, Junkun Yuan, Yanxin Long, Aladdin Wang, Andong Wang, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Hongmei Wang, Jacob Song, Jiawang Bai, Jianbing Wu, Jinbao Xue, Joey Wang, Kai Wang, Mengyang Liu, Pengyu Li, Shuai Li, Weiyan Wang, Wenqing Yu, Xinchi Deng, Yang Li, Yi Chen, Yutao Cui, Yuanbo Peng, Zhentao Yu, Zhiyu He, Zhiyong Xu, Zixiang Zhou, Zunnan Xu, Yangyu Tao, Qinglin Lu, Songtao Liu, Dax Zhou, Hongfa Wang, Yong Yang, Di Wang, Yuhong Liu, Jie Jiang, and Caesar Zhong. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2025. 2, 3, 14
- [29] Max Ku, Cong Wei, Weiming Ren, Harry Yang, and Wenhu Chen. Anyv2v: A tuning-free framework for any video-tovideo editing tasks. arXiv preprint arXiv:2403.14468, 2024. 7
- [30] Black Forest Labs. Flux, 2024. 2
- [31] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas M¨uller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv, 2025. 2

- [32] Runjia Li, Philip Torr, Andrea Vedaldi, and Tomas Jakab. Vmem: Consistent interactive video scene generation with surfel-indexed view memory. arXiv preprint arXiv:2506.18903, 2025. 3
- [33] Feng Liang, Akio Kodaira, Chenfeng Xu, Masayoshi Tomizuka, Kurt Keutzer, and Diana Marculescu. Looking backward: Streaming video-to-video translation with feature banks. arXiv preprint arXiv:2405.15757, 2024. 3
- [34] Chao Liao, Liyang Liu, Xun Wang, Zhengxiong Luo, Xinyu Zhang, Wenliang Zhao, Jie Wu, Liang Li, Zhi Tian, and Weilin Huang. Mogao: An omni foundation model for interleaved multi-modal generation. arXiv preprint arXiv:2505.05472, 2025. 3
- [35] Haonan Lin, Mengmeng Wang, Jiahao Wang, Wenbin An, Yan Chen, Yong Liu, Feng Tian, Guang Dai, Jingdong Wang, and Qianying Wang. Schedule your edit: A simple yet effective diffusion noise schedule for image editing. arXiv preprint arXiv:2410.18756, 2024. 3
- [36] Shanchuan Lin, Ceyuan Yang, Hao He, Jianwen Jiang, Yuxi Ren, Xin Xia, Yang Zhao, Xuefeng Xiao, and Lu Jiang. Autoregressive adversarial post-training for real-time interactive video generation. arXiv preprint arXiv:2506.09350,

2025. 3

- [37] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow Matching for Generative Modeling. In The Eleventh International Conference on Learning Representations (ICLR), 2023. 5, 14
- [38] Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. Video-p2p: Video editing with cross-attention control. In Computer Vision and Pattern Recognition (CVPR), 2024. 3
- [39] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, Guopeng Li, Yuang Peng, Quan Sun, Jingwei Wu, Yan Cai, Zheng Ge, Ranchen Ming, Lei Xia, Xianfang Zeng, Yibo Zhu, Binxing Jiao, Xiangyu Zhang, Gang Yu, and Daxin Jiang. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025. 2, 3, 4
- [40] Shaoteng Liu, Tianyu Wang, Jui-Hsien Wang, Qing Liu, Zhifei Zhang, Joon-Young Lee, Yijun Li, Bei Yu, Zhe Lin, Soo Ye Kim, and Jiaya Jia. Generative video propagation. In Computer Vision and Pattern Recognition (CVPR), 2025. 3
- [41] Xingchao Liu, Chengyue Gong, and qiang liu. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. In The Eleventh International Conference on Learning Representations (ICLR), 2023. 5, 14
- [42] I Loshchilov. Decoupled weight decay regularization. arXiv,

2017. 6, 7, 14, 15

- [43] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 3
- [44] Shen Nie, Hanzhong Allan Guo, Cheng Lu, Yuhao Zhou, Chenyu Zheng, and Chongxuan Li. The blessing of randomness: Sde beats ode in general diffusion-based image editing. In ICLR, 2024. 3

- [45] OpenAI. Gpt-5, 2025. 4, 6, 7, 13, 15, 16, 18
- [46] William Peebles and Saining Xie. Scalable Diffusion Models with Transformers. In International Conference on Computer Vision (ICCV), 2023. 5, 14
- [47] Rolandos Alexandros Potamias, Jinglei Zhang, Jiankang Deng, and Stefanos Zafeiriou. Wilor: End-to-end 3d hand localization and reconstruction in-the-wild. arXiv preprint arXiv:2409.12259, 2024. 4, 13
- [48] Chenyang QI, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. In International Conference on Computer Vision (ICCV), 2023. 3
- [49] Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K. Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. In Computer Vision and Pattern Recognition Conference (CVPR), 2025. 7
- [50] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. 2021. 14
- [51] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research (JMLR), 2022. 14
- [52] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, ChaoYuan Wu, Ross Girshick, Piotr Dollar, and Christoph Feichtenhofer. SAM 2: Segment anything in images and videos. In International Conference on Learning Representations (ICLR), 2025. 4, 13
- [53] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, Zhaoyang Zeng, Hao Zhang, Feng Li, Jie Yang, Hongyang Li, Qing Jiang, and Lei Zhang. Grounded sam: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159, 2024. 4, 13
- [54] Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. arXiv preprint arXiv:2311.10089, 2023. 2, 3, 4
- [55] Uriel Singer, Amit Zohar, Yuval Kirstain, Shelly Sheynin, Adam Polyak, Devi Parikh, and Yaniv Taigman. Video editing via factorized diffusion distillation. In European Conference on Computer Vision (ECCV 2024), 2025. 3
- [56] Shangkun Sun, Xiaoyu Liang, Songlin Fan, Wenxu Gao, and Wei Gao. Ve-bench: Subjective-aligned benchmark suite for text-driven video editing quality assessment. arXiv preprint arXiv:2408.11481, 2024. 6
- [57] Decart Team. Lucy edit: Open-weight text-guided video editing. arXiv, 2025. 2, 3, 5, 7, 17

- [58] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 3
- [59] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and Advanced Large-Scale Video Generative Models, 2025. 2, 3, 5, 6, 14
- [60] Peng Wang, Yichun Shi, Xiaochen Lian, Zhonghua Zhai, Xin Xia, Xuefeng Xiao, Weilin Huang, and Jianchao Yang. Seededit 3.0: Fast and high-quality generative image editing. arXiv preprint arXiv:2506.05083, 2025. 2, 3
- [61] Wen Wang, Yan Jiang, Kangyang Xie, Zide Liu, Hao Chen, Yue Cao, Xinlong Wang, and Chunhua Shen. Zero-shot video editing using off-the-shelf image diffusion models. arXiv preprint arXiv:2303.17599, 2024. 3
- [62] Yuhan Wang, Siwei Yang, Bingchen Zhao, Letian Zhang, Qing Liu, Yuyin Zhou, and Cihang Xie. Gpt-image-edit1.5m: A million-scale, gpt-generated image dataset. arXiv preprint arXiv:2507.21033, 2025. 3, 4, 14
- [63] Cong Wei, Zheyang Xiong, Weiming Ren, Xeron Du, Ge Zhang, and Wenhu Chen. Omniedit: Building image editing generalist models through specialist supervision. In International Conference on Learning Representations (ICLR),

2025. 3, 14

- [64] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-Image Technical Report. arXiv, 2025. 2, 3, 4, 5, 14
- [65] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, Ze Liu, Ziyi Xia, Chaofan Li, Haoge Deng, Jiahao Wang, Kun Luo, Bo Zhang, Defu Lian, Xinlong Wang, Zhongyuan Wang, Tiejun Huang, and Zheng Liu. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025. 3
- [66] Yuhui Wu, Liyi Chen, Ruibin Li, Shihao Wang, Chenxi Xie, and Lei Zhang. Insvie-1m: Effective instruction-based video editing with elaborate dataset construction. In International Conference on Computer Vision (ICCV), 2025. 2, 3

- [67] Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and Xingang Pan. Worldmem: Longterm consistent world simulation with memory, 2025. 3
- [68] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10371–10381, 2024. 6, 16
- [69] Zhendong Yang, Ailing Zeng, Chun Yuan, and Yu Li. Effective whole-body pose estimation with two-stages distillation. arXiv preprint arXiv:2307.15880, 2023. 6, 16
- [70] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. arXiv, 2024. 2, 3, 14
- [71] Danah Yatim, Rafail Fridman, Omer Bar-Tal, Yoni Kasten, and Tali Dekel. Space-time diffusion features for zero-shot text-driven motion transfer. In Computer Vision and Pattern Recognition (CVPR), 2024. 7
- [72] Zixuan Ye, Xuanhua He, Quande Liu, Qiulin Wang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, Qifeng Chen, and Wenhan Luo. Unic: Unified in-context video editing. arXiv preprint arXiv:2506.04216, 2025. 3, 5
- [73] Tianwei Yin, Micha¨el Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T. Freeman. Improved distribution matching distillation for fast image synthesis. arXiv preprint arXiv:2405.14867, 2024. 3, 5, 8
- [74] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. 2025. 3
- [75] Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Anyedit: Mastering unified high-quality image editing for any idea. In Computer Vision and Pattern Recognition Conference (CVPR). 3
- [76] Shoubin Yu, Difan Liu, Ziqiao Ma, Yicong Hong, Yang Zhou, Hao Tan, Joyce Chai, and Mohit Bansal. Veggie: Instructional editing and reasoning video concepts with grounded generation. arXiv preprint arXiv:2503.14350,

2025. 2, 3

- [77] Bohan Zeng, Ling Yang, Jiaming Liu, Minghao Xu, Yuanxing Zhang, Pengfei Wan, Wentao Zhang, and Shuicheng Yan. Editworld: Simulating world dynamics for instructionfollowing image editing. In ACM International Conference on Multimedia, 2025. 3, 4
- [78] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: a manually annotated dataset for instructionguided image editing. In International Conference on Neural Information Processing Systems (NeurIPS), 2023. 3, 4
- [79] Bojia Zi, Penghui Ruan, Marco Chen, Xianbiao Qi, Shaozhe Hao, Shihao Zhao, Youze Huang, Bin Liang, Rong Xiao, and Kam-Fai Wong. Se˜norita-2m: A high-quality instructionbased dataset for general video editing by video specialists. arXiv preprint arXiv:2502.06734, 2025. 2, 3, 7

# Appendix

For more video results, please check the website.

#### Appendix Contents

- A. Additional Dataset Details 13

- A.1. Pipeline Details . . . . . . . . . . . . . . . . 13
- A.2. Dataset statistics . . . . . . . . . . . . . . . 13

- B. Additional Method Details 14

- B.1. Text-to-Video Model . . . . . . . . . . . . . 14
- B.2. Training Details . . . . . . . . . . . . . . . 14
- B.3. Distillation Procedure Details . . . . . . . . 14

- C. Additional Benchmark Details 15

- C.1. Evaluation Tasks . . . . . . . . . . . . . . . 15
- C.2. EgoEditBench-Human Alignment . . . . . . 16

- D. Evaluation Details 16

- D.1. Baseline Details . . . . . . . . . . . . . . . 16
- D.2. Metrics Details . . . . . . . . . . . . . . . . 17

- E. Additional Results 17

- E.1. Additional In-the-Wild Results . . . . . . . . 17
- E.2. Additional Comparison to Baselines . . . . . 17
- E.3. Additional Distillation Ablation Results . . . 17
- E.4. Additional Dataset Ablation Results . . . . . 18

- F. Failed Experiments 18

#### A. Additional Dataset Details

###### A.1. Pipeline Details

Video selection. We select videos from Ego4D [19] and EgoExo4D [20] according to the following criteria. The video must be captured with one of the following camera models: GoPro Hero 4, GoPro Hero Black 7, GoPro Hero Black 8, GoPro Hero Black 9, GoPro Hero Silver 7, or GoPro Max, and it must be monocular rather than binocular. Furthermore, to ensure that the videos are sharp and visually informative, we apply additional filtering based on jitter scores and aesthetic scores.

Hand mask segmentation. We employ WiLoR [47] for hand detection, applying a confidence threshold of 0.75 on a frame-by-frame basis. From the detected frames, we select the three with the highest confidence scores and use their hand masks to generate point prompts for SAM2 [52], enabling the creation of dense and temporally consistent hand masks across frames.

Object names extraction. We employ Qwen2.5-VL32B [2] to extract object names. For inputting the video, we sub-sample the video with 2 fps into frames and ask the model what is held by the hand in the video.

Object mask segmentation. We use the extracted object names to prompt Grounded-SAM [53] to generate object masks for each frame. A mask confidence threshold of 0.4 and a text threshold of 0.35 are applied. Each object mask is then filtered based on its distance to the hand skeleton and the edge of the hand mask. If multiple objects are detected in a single frame, we select the mask closest to the hand mask. To refine the prompts, we exclude regions corresponding to the hand masks and use the remaining object mask areas from the top three frames with the highest confidence scores to generate point prompts for SAM2, enabling the creation of dense and consistent object masks.

Object Editing. We use the object masks to generate rectangular masks for each frame, expanding them with an additional margin. A Gaussian dilation with a kernel size of 50×50 px is then applied to these rectangular masks. After excluding the regions corresponding to the hand masks, the resulting final masks are used by VACE-14B [24] to produce the edited videos. The resolution of the generated videos is 1920×1104 px.

###### A.2. Dataset statistics

EgoEditData contains approximately 99,700 video editing pairs, each consisting of a source video, a target video, and a corresponding instruction prompt. Our dataset is built upon two egocentric video datasets with 6.4% of the videos originating from EgoExo4D [20], and the remaining 93.6% from Ego4D [19]. EgoEditData focuses on object manipulation in egocentric videos. The majority of samples correspond to the Change Object task, which prompts the model to replace an existing object in the video with an ordinary object, and contains 54,164 pairs. The second largest group consists of Change Object with Special Effects tasks, comprising 39,465 samples, where the source object is replaced with an imaginary object with special effects such as fire or frost. In addition, the dataset includes 3,651 samples for the Add Object task and 2,379 for the Remove Object tasks.

Figure 7a presents the word cloud of source objects descriptions, while Figure 7b shows the word cloud of target objects descriptions. Overall, thanks to the sampling of target object names through GPT-5 Mini [45] used during the Object Editing phase of dataset construction, the dataset contains 13,632 distinct target objects compared to 3,199 unique source objects, enhancing diversity. We also report in Figure 7c the distribution of the instruction prompt lengths. With a mean of 378 characters, most prompts contain detailed instructions specifying target object, scene details and style, while certain tasks not requiring such context such as Remove/Add Object have more concise prompts like “Remove/Add an {object}.”. Finally, Figure 8 shows the ten most frequent scenarios in EgoEditData as categorized by the original egocentric datasets, highlighting a balanced distribution across different types of scenes and scenarios.

[Figure 148]

11/19/25, 12:12 PM prompt_length_distribution.html

###### Instruction Prompt Length Distribution

- (I) Source Objects Word Cloud
- (II)(b)TargetTargetObjectsWordWordCloudCloud

[Figure 149]

[Figure 150]

###### Character Length Distribution Word Count Distribution

20k

Mean: 378 Mean: 60

10k

15k

Frequency

Frequency

Frequency

10k

5k

5k

0

0

0 200 400 600

0 20 40 60 80 100

Character Count

Characters Words

- (I) Source Objects Word Cloud
- (II) Target Objects Word Cloud

(a) Source Word Cloud

(c) Prompt Length Distribution

[Figure 151]

Figure 7. Overview of dataset statistics. (a) and (b) illustrate the word clouds for the source and target object descriptions respectively. While source object descriptions focus on everyday objects, target descriptions often contain descriptions of materials and special effects. (c) shows the distribution of prompt lengths. Short prompts in (c) correspond to instructions not requiring context such as “Remove/Add an {object}”.

11/19/25, 9:08 PM top_10_scenarios.html

###### Top 10 Scenarios

###### B.2. Training Details

Cooking

Cooking

27,482

Crafting

Crafting/knitting/sewing/drawing/painting

22,403

Training configuration.

Handyman

Handyman

10,137

We finetune the pretrained text to video model for 30k iterations with a total batch size of 96 videos using 48 H100 GPUs. We use AdamW [42] optimizer with lr 1e-5, a learning rate scheduler with linear warmup of 1000 steps, weight decay of 0.1, beta values of 0.99 and exponential moving average of 0.9999.

Plumber

plumber

10,084

Others

Others

10,084

Scenario

Electrician

Electrician

10,084

Construction

jobs related to construction/renovation company

10,084

Tiler

tiler

10,084

Cleaning

Cleaning / laundry

8,631

ﬁle:///Users/mali5/Downloads/prompt_length_distribution.html 1/1

Editing data corpora. We collect a corpus of editing data to supplement EgoEditData for the non-egocentric case. We consider the following publicly available editing datasets: GPT-Image-Edit-1.5M [62], ShareGPT-4oImage, Complex-Edit, HQEdit, OmniEdit [63], and UltraEdit. In addition, we employ internal data generation pipelines for the creation of additional editing training data. The pipelines employ Qwen-Image-Edit [64] to generate 2M image editing examples, and a combination of models including Wan VACE [24], Wan-Animate and MiniMaxRemover to generate an additional corpus of 210k video editing data containing edit categories including object addition and removal, object substitution, human edits. Generated editing pairs are assessed through a combination of automated and manual filtering. In addition, the pipeline synthesizes 1.1M video editing data pair by considering natural videos and pairing them with depth estimation, pose estimation, edge estimation, and optical flow estimation models. The training data is composed of multiple datasets. During training, we apply importance sampling with weights of 28% for EgoEditData, 52% for other video editing datasets, and 20% for image editing datasets. Within each category, individual datasets are weighted proportionally to their sizes.

Mechanic

Car mechanic

4,163

0 5k 10k 15k 20k 25k

Frequency

Frequency

Figure 8. Distribution of most frequent scenarios in EgoEditData according to the original datasets [19, 20] categorization.

#### B. Additional Method Details

###### B.1. Text-to-Video Model

ﬁle:///Users/mali5/Downloads/top_10_scenarios.html 1/1

State-of-the-art methods [28, 59, 70] in text-to-video generation commonly adopt the latent diffusion [21] paradigm, DiT-like [46] transformer backbones, and the Flow Matching [37, 41] framework. Our pretrained text-to-video model follows the same paradigm. We model videos in the latent space of a Wan 2.1 autoencoder [59] which performs an 8 × 8 × 4 compression along the height, width, and time dimensions respectively. Input and output linear projections perform 2×2 spatial patchification to further increase compression. The main backbone consists of a 10.7B transformer backbone composed of 32 identical transformer blocks with 4096 hidden dimensions and 32 heads. Each block employs self attention, followed by cross attention for text conditioning, and a final MLP, and uses modulation [9] for timestep conditioning. QK normalization [15] and Flash Attention [12] are used for every attention operation to improve stability and speed. Text tokens are extracted using a combination of pretrained T5 [51] and CLIP [50] text encoders. The model performs inference in 40 steps using an Euler solver and is trained at a resolution of 512 px.

###### B.3. Distillation Procedure Details

The distillation procedure is responsible for enabling realtime and low-latency execution of EgoEdit starting from the finetuned video editing method. The original model performs inference of 5s, 512×384px, 16 fps videos using 40 steps with classifier-free guidance, for a total of 80

model evaluations (NFEs). As every model evaluation takes 86ms on a single H100 GPU, the resulting throughput is of only 11.9fps, which additionally decreases to 9.68fps when considering autoencoder source video encoding and target video decoding times, far from the 16fps required for realtime performance (see Table 2).

The first phase of distillation thus performs step and guidance distillation using DMD to yield a model producing a 5s 512×384px 16fps video in 4 NFEs, for a 20× increase in throughput. While the model now possesses a throughput of 43.5fps after application of the autoencoder, the full video needs to be sampled before the first frame can be displayed, and the autoencoder needs to be run on the full source and target videos, creating a first frame latency of 6.93s, inhibiting interactive usage. We perform DMD distillation for 4500 iterations with a total batch size of 64 using 32 H100 GPUs. We use the AdamW [42] optimizer with lr 1e-6 for the generator and 4e-7 for the fake score model, 5 steps per generator update, weight decay of 0.1, beta values of 0.99 and exponential moving average of 0.99.

To enable interactive usage, Self Forcing [23] performs distillation of the bidirectional DMD model into an autoregressive model capable of producing videos in a chunk-bychunk fashion. In this way, the first frame can be displayed to the user right after the first chunk is sampled and decoded by the autoencoder, reducing first frame latency to 855ms. We use a configuration of Self Forcing where latent frames are denoised in chunks spanning three consecutive latent frames [23]. We conduct Self Forcing distillation starting directly from the DMD distilled checkpoint for 4500 iterations with a total batch size of 64 using 64 H100 GPUs. We use an AdamW [42] optimizer with lr 1e-6 for the generator and 4e-7 for the fake score model, 10 steps per generator update, weight decay of 0.1, beta values of 0.99 and exponential moving average of 0.99. We find that the model can quickly adapt to autoregressive modeling when initialized from the DMD checkpoint, so we skip the ODE initialization phase [23]. We use 7 chunks for generation and each chunk contains 3 latent frames with 21 latent frames in total. Following [23], to imitate the generation for longer videos, during training, we mask the first chunk of the latent frames when generating the last chunk and we use a window size of 5 chunks as condition during inference.

#### C. Additional Benchmark Details

###### C.1. Evaluation Tasks

Starting from 100 unique egocentric source videos, we construct a benchmark spanning 15 editing tasks. We ensure diversity by clustering source objects and scenario names using sentence embeddings and then uniformly sample source videos across clusters. Conditioned on each source video, its caption, and the source object, we use

a GPT-5 Mini [45] to produce task-specific instruction prompts. Below we include details on constructing the source video and instruction prompt for each editing task.

- • Remove Object: We aim to evaluate the model’s ability to remove certain objects from the video while keeping the other parts of the video consistent with the source video. We select 50 source videos from the EgoEditData and their instruction prompts. We include them in the benchmark and remove them from the training set.
- • Add Object: We aim to evaluate the model ability to insert a specified object into the scene while keeping the rest of the video consistent with the source. We select 50 source videos from the EgoEditData together with their instruction prompts. We use source videos from the Add Object task, where target object is absent and synthetically removed following the EgoEditData pipeline. We include these in the benchmark and exclude them from the training set.
- • Change Object: We evaluate the model’s ability to alter a specified object, either by modifying an attribute or replacing it with a new specified object while keeping the rest of the video consistent with the source. For each sampled source video, we create four instruction prompts: two that perform a pure replacement (object: a → b) and two that pair the replacement with an added effect on the new object (e.g., fire or glow). In total, we obtain 400 videos for evaluating the Change Object task. Instruction prompts are designed following EgoEditData pipeline.
- • Change Background: We evaluate the model’s ability to replace or edit the background while preserving foreground identity and motion. To construct instructions, we provide GPT-5 Mini [45] with a few source frames and the video caption and request an editing instruction prompt with a semantically compatible target background.
- • Change Camera Pose: We assess recomposition via a specified camera trajectory (pan/tilt/dolly/zoom) without altering scene events. Instruction prompts are generated by GPT-5 [45] Mini by prompting it with the video caption which contains a description of the camera pose in the source video.
- • Add Effect: We evaluate adding post-processing effects while preserving scene content. These effects usually operate as global filters (e.g., motion blur, VHS, glow, film grain) that are independent of the particular video. Accordingly, we construct instructions by prompting GPT-5 Mini—primed with a few in-context examples from EditVerseBench [26] and to ask it to propose a diverse pool of effect editing instructions, from which we randomly sample one per clip.
- • Stylization: We mirror the procedure of Add Effect by providing GPT-5 Mini with EditVerseBench [26] examples and ask it to propose diverse set of style editing in-

- structions. We sample one randomly per source video.
- • Reasoning: We evaluate edits that rely on spatial and temporal reasoning. Given the source video and its caption, we prompt GPT-5 Mini [45] to propose an editing instruction tied to an explicit anchor (event or timestamp) or disambiguating relations (e.g., leftof/behind/before/after). The instruction deliberately avoids explicitly naming a unique target and instead focuses on giving an instruction where the correct object must be inferred from context.
- • Depth-to-Video: We convert the source video into a depth map using Depth Anything [68], and construct the instruction prompt as (“Turn the depth map into a video with the following description: caption.”) where the caption is the source video caption describing the appearance of the scene.
- • Sketch-to-Video: We convert the source video into Canny edge maps using OpenCV, and construct the instruction prompt as (“Turn the canny edge map into a video with the following description: caption.”), where caption is the source video caption describing the scene’s appearance and layout.
- • Pose-to-Video: We convert the source video into 2D human poses using DWpose [69], and construct the instruction prompt as (“Turn the DWpose pose map into a video with the following description: ⟨caption⟩.”), where caption is the source video caption specifying the subject’s identity, attire, and overall scene appearance.
- • Video-to-Depth: We prompt the model to convert a video into a temporally consistent depth map. For this task, we use fixed instruction prompt (“Turn the video into a depth map.”).
- • Video-to-Sketch: We use a fixed instruction prompt of (“Turn the video into a Canny edge map.”).
- • Video-to-Pose: We use a fixed instruction prompt (“Turn the video into a DWpose pose map.”).
- • Combined (Multi-Task): We compose multiple editing prompts from the same source video (e.g., Pose-toVideo + Change Background + Stylization) by sampling a subset of instruction prompts and prompting a GPT-5 Mini [45] to compose a single instruction prompt that combines all of the tasks together.

###### C.2. EgoEditBench-Human Alignment

To evaluate the reliability of the VLM score employed in EgoEditBench, we conduct a study on EgoEditBench comparing VLM and human preference alignment. Given 30 randomly sampled benchmark element per category, and a baseline method, we conduct VLM evaluation of EgoEdit and the baseline following EgoEditBench protocol and, for each sample, assign VLM preference to the method with highest VLM score. Simultaneously we ask a human evaluator to express preference for each sample for our method or

the baseline. Results are shown in Table 4. VLM and user preferences are in high agreement, with 86.2% and 84.9% of cases on average, respectively when evaluated against LucyEdit and InsV2V. We thus rely on VLM score as the main benchmark metric.

#### D. Evaluation Details

###### D.1. Baseline Details

To ensure a fair comparison, we use each baseline’s default inference hyperparameters including the number of frames, frames-per-second (FPS), spatial resolution, and inference settings such as guidance scale and sampling steps. Below, we include the specific hyperparameters we used for evaluating each baseline:

- • TokenFlow. We rely on Stable Diffusion 1.5 as the backbone. We use 16 frames at a
- • STDF. We use 24 frames at 10 fps and 40 inference steps and 10 optimization steps. We use a guidance scale of 10 and inference at the resolution of 576 × 320.
- • SENORITA (Senorita-2M). We use 33 frames at 8 fps and 768 × 448 resolution, with guidance scale of 4 and 30 inference steps. We use CogVideoX as the backbone. To obtain the edited first frame, we use the first frame generated by our model instead of relying on pretrained ControlNets, following [26]. We crop and resize the first frame generated by our method to the default resolution of SENORITA.
- • AnyV2V. We use 16 frames at 8 fps and 512 × 512, guidance of 9, with 100 inversion steps plus 50 edit steps. Similarly, we use the first frame generated by our model instead of relying on pretrained ControlNets, following [26]. We crop and resize the first frame generated by our method to the default resolution of AnyV2V.
- • InsV2V. We use 32 frames at 15 fps and 384 × 384, with the default guidance for the video branch (1.2) and text branch (7.5), over 20 inference steps.
- • Lucy-Edit. We use 81 frames at 15 fps with 832 × 480 resolution, guidance scale of 5, and 50 inference steps.
- • EditVerse. Since EditVerse is a closed source model, we only compare on EditVerseBench, where we rely on its published samples. We omit the results on EgoEditBench since we do not have access to the model to generate the required results.
- • StreamDiffusion. We use the image-to-image editing pipeline for the video editing task. We process 81 frames and 16 fps at the resolution 832 × 480.
- • StreamDiffusionV2. We use the streaming setup at 832× 480 with an 81 frames sequence and 16 fps, using a very light 2–4 denoising steps.

VLM Mean Score Preference (VLM) Preference (User Study) Agreement (%) Task EgoEdit LucyEdit InsV2V vs LucyEdit vs InsV2V vs LucyEdit vs InsV2V LucyEdit InsV2V

Add Object 7.83 4.12 5.47 29 (100%) 29 (97%) 29 (100%) 29 (97%) 100.0 93.3 Change Camera Pose 7.11 6.93 7.30 20 (67%) 14 (47%) 27 (90%) 28 (93%) 70.0 40.0 Change Object 7.61 6.49 4.09 23 (77%) 26 (87%) 29 (97%) 30 (100%) 80.0 86.7 Change Background 7.28 5.10 4.46 29 (97%) 27 (90%) 28 (93%) 29 (97%) 90.0 86.7 Combined (Multi-Task) 7.96 6.36 4.66 30 (100%) 29 (97%) 29 (97%) 30 (100%) 96.7 96.7 Depth-to-Video 8.53 7.44 5.72 30 (100%) 30 (100%) 30 (100%) 30 (100%) 100.0 100.0 Add Effect 6.41 5.29 5.96 24 (80%) 19 (63%) 24 (80%) 27 (90%) 80.0 73.3 Video-to-Pose 7.90 4.63 4.39 30 (100%) 30 (100%) 28 (93%) 28 (93%) 93.3 93.3 Pose-to-Video 8.58 7.18 4.77 28 (93%) 30 (100%) 28 (93%) 29 (97%) 86.7 96.7 Reasoning 6.69 5.30 4.85 24 (80%) 23 (77%) 25 (83%) 29 (97%) 63.3 73.3 Remove Object 6.72 5.04 5.71 25 (83%) 23 (77%) 26 (87%) 29 (97%) 76.7 73.3 Sketch-to-Video 8.81 6.31 5.54 30 (100%) 30 (100%) 27 (90%) 30 (100%) 90.0 100.0 Stylization 7.88 4.68 6.70 29 (97%) 25 (83%) 29 (97%) 29 (97%) 93.3 80.0 Video-to-Depth 8.80 4.00 4.80 30 (100%) 30 (100%) 28 (93%) 28 (93%) 93.3 93.3 Video-to-Sketch 9.00 3.78 4.34 30 (100%) 30 (100%) 24 (80%) 26 (87%) 80.0 86.7

Overall 7.76 5.44 5.24 411 (91%) 395 (88%) 411 (91%) 431 (96%) 86.2 84.9

Table 4. Study on EgoEditBench comparing VLM and human preference alignment. We conduct VLM evaluation following EgoEditBench protocol and for each sample, assign VLM preference to the method with highest VLM score. Simultaneously we ask a human evaluator to express preference for a sample from one of two compared methods. We find VLM and user preference to be in agreement in 86.2% and 84.9% of cases on average, respectively when evaluated against LucyEdit and InsV2V.

###### D.2. Metrics Details

We closely follow [26] when computing metrics for our benchmark. We evaluate edit quality with a VLM-based judge, overall video quality with PickScore, text alignment at the frame and video levels, and temporal consistency with CLIP and DINO. Because frame- and video-level text alignment are highly correlated in our setting, we report only the video-level score for brevity. We also find the CLIP- and DINO-based consistency metrics to be strongly correlated, so we rely on the CLIP consistency metric for temporal consistency. Among all metrics, we found the VLM score to align well with human judgment (see Table 4), so we use it as our primary quality signal. For EgoEditBench, we use exactly the same evaluation prompts as [26] when computing the VLM editing-quality score.

#### E. Additional Results E.1. Additional In-the-Wild Results

We present in Figure 10 and Figure 9 additional in-the-wild qualitative examples produced by EgoEdit-RT in real-time on a single H100 GPU for the egocentric and exocentric cases respectively. EgoEdit-RT can perform complex edits such as transforming markers placed on the floor into pillars of the Golden Gate Bridge, model fluid effects, change a location into a haunted mansion, add animals that interact with the environment and the user, or perform stylization.

###### E.2. Additional Comparison to Baselines

We present additional qualitative results in Figure 11 comparing EgoEdit to baseline video editing methods. EgoEdit and its real-time variant EgoEdit-RT are capable of performing a range of editing tasks including object attribute changes, object replacement, object insertion, style transfer, background changes, and conversion to depth map. We observe concurrent offline methods [57] and real-time video editing methods [17, 27] to often fail in the egocentric case, with failure modes often manifesting as inability to produce any change over the source video, or modifying the input beyond what is requested in the editing instruction. In addition, Table 5 reports per-category VLM scores on EgoEditBench for our method and baselines.

###### E.3. Additional Distillation Ablation Results

Figure 12 shows additional qualitative results comparing variants of EgoEdit at different stages of distillation: the starting 40-step (80 NFEs) video editing checkpoint obtained after finetuning of the pretrained video generation model (EgoEdit), the 4-step (4 NFEs) variant obtained as a result of the DMD distillation phase (EgoEdit-DMD), and the real-time streaming variant obtained at the end of Self Forcing training (EgoEdit-RT). We observe similar qualitative performance among the distilled variants as shown in the figure. We observe EgoEdit-RT to occasionally present temporal shifts at the boundaries between different chunks of predicted frames. In addition, when qualitatively evalu-

Sketch2Vid Stylization Vid2Depth Vid2Sketch

Cam.Mov.

Change Chg.BG Combined Depth2Vid

Vid2Pose. Pose2Vid Reasoning Remove

Overall

Effect

Add

Method

TokenFlow 5.56 5.67 5.37 5.02 5.39 4.05 5.80 5.62 0.52 5.24 5.49 3.73 6.01 5.88 5.54 4.99 STDF 4.59 4.40 4.83 4.82 5.27 4.71 4.67 5.33 2.58 4.13 4.55 4.00 5.33 5.17 4.45 4.59 Se˜norita-2M 7.85 7.34 7.35 6.64 7.10 8.48 7.24 6.68 7.85 7.61 6.93 8.49 6.75 8.41 8.05 7.52 AnyV2V 7.72 7.55 7.22 6.58 6.42 7.14 7.32 6.66 2.62 7.52 7.21 5.29 6.58 7.35 7.56 6.72 InsV2V 5.67 7.26 3.99 4.42 4.85 5.60 6.30 4.39 4.69 4.41 5.52 5.55 6.73 4.79 4.40 5.22 Lucy Edit 4.31 6.92 6.25 4.67 6.15 7.31 5.16 4.49 7.13 5.68 4.81 6.52 4.65 3.88 3.72 5.46 StreamDiffusion 3.95 3.06 3.44 5.29 5.60 5.53 3.62 5.43 2.66 3.26 2.91 5.20 6.32 4.45 4.02 4.31 StreamDiffusionV2 1.63 1.76 2.42 2.21 3.62 4.66 2.42 3.52 2.45 1.10 1.02 3.33 3.19 3.53 1.37 2.55

EgoEdit 7.89 6.79 7.84 7.45 7.74 8.57 6.32 7.87 8.57 6.79 6.82 8.70 7.46 8.70 8.93 7.76 EgoEdit-DMD 7.91 6.76 8.04 5.36 7.29 8.01 5.92 8.07 5.60 7.16 7.46 7.76 6.45 8.95 8.96 7.42 EgoEdit-RT 7.79 6.67 8.09 7.57 7.83 8.52 6.40 8.48 8.31 7.54 4.01 7.92 7.43 8.89 8.98 7.83

Table 5. Breakdown of per-category EgoEditBench VLM scores for EgoEdit and baselines.

Method VLM ↑ PS ↑ TA ↑ TC ↑ EgoEdit 7.80 19.09 16.91 96.74 EgoEdit-DMD 7.42 18.95 16.52 96.87 EgoEdit-RT 7.83 19.04 16.49 96.49

- Table 6. Quantitative comparison of EgoEdit variants on EgoEditBench: “VLM” is VLM evaluation score, “PS” is Pick Score, “TA” is Text Alignment, “TC” is Temporal Consistency. EgoEditRT is the real-time version, and EgoEdit-DMD is the bidirectional DMD variant.

ated on in-the-wild cases, distilled variants (EgoEdit-DMD, EgoEdit-RT) exhibited a lower capacity of handling complex out-of-distribution editing instructions. In addition, Table 6 reports full evaluation scores on EgoEditBench for all distilled variants of EgoEdit.

###### E.4. Additional Dataset Ablation Results

In Figure 13, we show qualitative examples produced by different variants of EgoEdit trained on our training data mixture same as the main experiment (Table 1) with reduced portions of EgoEditData. As the portion of EgoEditData increases, we notice increase quality of edits and improved alignment to the source video. We additionally evaluate whether increasing the amount of data in EgoEditData can result in improved performance in non-egocentric benchmarks. Table 7 compares variants of EgoEdit trained with a data mixture same as the main experiment including versions of EgoEditData of reduced size, on the EditVerseBench benchmark, which mostly consists of exocentric videos. We find that the introduction of EgoEditData consistently improves the overall benchmark score, with particular regards to the Reasoning, Remove, Camera Movement, and Change categories. We speculate these improvements originate from strict quality filtering and descriptive captions, whose benefits extend beyond the egocentric case.

#### F. Failed Experiments

Usage of unfiltered data. We initially experiment with a video editing dataset corpora consisting of lightly filtered video editing pairs. In this setting, we notice weak video editing performance, with the model often reproducing artifacts encountered in low-quality video editing data pairs such as failure of an object in being replaced with a different object, or failure in adding an object. This motivates us to integrate extensive automated and manual curation into EgoEditData at the expense of dataset size, which resulted in increased editing performance.

Usage of detailed editing instructions. We initially experiment with simple editing instructions for EgoEditData that are obtained by filling templates with known source and target object names such as “Replace ⟨source object⟩ with ⟨target object⟩”, where source and target objects are represented by strings obtained during the Object names extraction and Object Editing stages of the data curation pipeline. We observe low editing performance when training on such prompts. Such captions suffer from several issues: (i) their variety is limited, (ii) object names extracted in the Object names extraction stage are often short, (iii) objects in generated target videos produced in the Object Editing stage might not faithfully correspond to the requested edited object description. After training on editing instructions using GPT-5 Mini [45], we observe increased ability to follow instruction prompts.

Sketch2Vid Stylization Overall

Cam.Mov.

Change Chg.BG Combined Depth2Vid Pose2Vid Reasoning Remove

Add

% of EgoEditData

0% 7.80 6.27 6.38 8.33 5.54 7.63 8.17 6.43 4.30 7.40 7.87 6.89 25% 7.23 6.43 6.82 8.23 7.42 6.00 8.63 6.73 5.00 6.63 8.20 7.00 75% 7.73 7.37 6.88 7.80 5.29 8.07 8.63 7.53 5.27 7.53 8.60 7.30 100% 7.92 7.60 7.46 8.27 7.17 8.13 8.57 8.37 7.00 7.17 8.77 7.79

- Table 7. EditVerseBench [26] results for our model trained on a data mixture where the number of samples retained in EgoEditData is varied. As the amount of retained samples in EgoEditData increases, non-egocentric editing performance as measured by EditVerseBench [26] increases, showing usefulness of EgoEditData beyond the egocentric case.

Input Real Time Result

Input Real Time Result

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

Transform the human into a robot with metallic skin and glowing eyes.

Transform the human into a devil with horns and a tail.

|[Figure 160]<br><br>Trans|[Figure 161]<br><br>form the human into a fairy|
|---|---|

|[Figure 162]<br><br>with translucent wings and spar|[Figure 163]<br><br>kles.|
|---|---|

|[Figure 164]<br><br>Tra|[Figure 165]<br><br>nsform the human into an|
|---|---|

|[Figure 166]<br><br>with white wings and a hal|[Figure 167]<br><br>o.|
|---|---|

w

angel

|[Figure 168]|[Figure 169]<br><br>A small dragon sitting|
|---|---|

|[Figure 170]<br><br>the human's shoulder.|[Figure 171]|
|---|---|

|[Figure 172]|[Figure 173]<br><br>A tiny fairy sitting on|
|---|---|

|[Figure 174]<br><br>the human's shoulder.|[Figure 175]|
|---|---|

on

th

|[Figure 176]|[Figure 177]<br><br>A small cat sitting on|
|---|---|

|[Figure 178]<br><br>the human's shoulder.|[Figure 179]|
|---|---|

|[Figure 180]|[Figure 181]<br><br>A fluffy rabbit sitting on|
|---|---|

|[Figure 182]<br><br>the human's shoulder.|[Figure 183]|
|---|---|

t

###### Figure 9. Exocentric video edits produced in real time by EgoEdit’s streaming variant EgoEdit-RT on a single H100 GPU.

Input Real Time Result Input Real Time Result

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

Turn the water into lava.

Morph the red stop sign into a green go sign.

|[Figure 192]<br><br>Add a hu|[Figure 193]<br><br>ge octopus on the roof [...]|
|---|---|

|[Figure 194]<br><br>water waves drowning the|[Figure 195]<br><br>building.|
|---|---|

|[Figure 196]|[Figure 197]<br><br>Transform the environment|
|---|---|

|[Figure 198]<br><br>into a haunted mansion.|[Figure 199]|
|---|---|

huge

t

[Figure 200]

[Figure 201]

|[Figure 202]|
|---|

|a red sports car in the center.|
|---|

|[Figure 203]|
|---|

|[Figure 204]<br><br>Transform the marke|[Figure 205]<br><br>rs into a [...] model of the|
|---|---|

|[Figure 206]<br><br>Gate Bridge, where each|[Figure 207]<br><br>marker is a pillar [...].|
|---|---|

Add

r.

Golden

|[Figure 208]|[Figure 209]<br><br>Replace the white pillow with|
|---|---|

|[Figure 210]<br><br>a dog that enjoys the petting.|[Figure 211]|
|---|---|

|[Figure 212]|[Figure 213]<br><br>Transform the umbrella into|
|---|---|

|[Figure 214]<br><br>a glowing energy sword.|[Figure 215]|
|---|---|

|[Figure 216]|[Figure 217]<br><br>Make the car covered|
|---|---|

|[Figure 218]<br><br>in rust and aged.|[Figure 219]|
|---|---|

|[Figure 220]|[Figure 221]<br><br>Replace the plastic|
|---|---|

|[Figure 222]<br><br>toy with a real duck.|[Figure 223]|
|---|---|

d

duck

|[Figure 224]|[Figure 225]<br><br>Replace the stop|
|---|---|

|[Figure 226]<br><br>with a yield sign.|[Figure 227]|
|---|---|

|[Figure 228]|[Figure 229]<br><br>Transform the building|
|---|---|

|[Figure 230]<br><br>into an ice palace.|[Figure 231]|
|---|---|

sign

g

|[Figure 232]|[Figure 233]<br><br>Add a dog walking|
|---|---|

|[Figure 234]<br><br>up the stairs.|[Figure 235]|
|---|---|

|[Figure 236]|[Figure 237]<br><br>Add a baby elephant|
|---|---|

|[Figure 238]<br><br>sitting in the trash can.|[Figure 239]|
|---|---|

ng

s

|[Figure 240]|[Figure 241]<br><br>Replace the plastic duck toy|
|---|---|

|[Figure 242]<br><br>with a real duck getting petted.|[Figure 243]|
|---|---|

|[Figure 244]|[Figure 245]<br><br>Replace the white|
|---|---|

|[Figure 246]<br><br>pillow with dumbbells.|[Figure 247]|
|---|---|

w

pi

|[Figure 248]|[Figure 249]<br><br>Replace the mouse|
|---|---|

|[Figure 250]<br><br>with a smartphone.|[Figure 251]|
|---|---|

|[Figure 252]|[Figure 253]<br><br>Change the plant to be|
|---|---|

|[Figure 254]<br><br>frozen with ice and mist.|[Figure 255]|
|---|---|

f

|[Figure 256]|[Figure 257]<br><br>Add a volcano exploding,|
|---|---|

|[Figure 258]<br><br>spreading lava everywhere.|[Figure 259]|
|---|---|

|[Figure 260]|[Figure 261]<br><br>Turn the|
|---|---|

|[Figure 262]<br><br>into lava.|[Figure 263]|
|---|---|

s

water

|[Figure 264]<br><br>Replac|[Figure 265]<br><br>e the floor with lava. Stones|
|---|---|

|[Figure 266]<br><br>present where the feet are|[Figure 267]<br><br>placed.|
|---|---|

|[Figure 268]|[Figure 269]<br><br>Turn the parking lot|
|---|---|

|[Figure 270]<br><br>into a water pond.|[Figure 271]|
|---|---|

are

|[Figure 272]<br><br>Add a|[Figure 273]<br><br>corgi and the corgi licks the|
|---|---|

|[Figure 274]<br><br>hand when the hand moves for|[Figure 275]<br><br>ward.|
|---|---|

|[Figure 276]|[Figure 277]<br><br>Replace the apple with a|
|---|---|

|[Figure 278]<br><br>apple with mists and fogs.|[Figure 279]|
|---|---|

h

frozen

|[Figure 280]<br><br>C|[Figure 281]<br><br>hange the building facade to|
|---|---|

|[Figure 282]<br><br>glass and steel modern style.|[Figure 283]|
|---|---|

|[Figure 284]|[Figure 285]<br><br>Make it snowy winter|
|---|---|

|[Figure 286]<br><br>with snow on the car.|[Figure 287]|
|---|---|

|[Figure 288]|[Figure 289]<br><br>Replace the gray|
|---|---|

|[Figure 290]<br><br>bag with a violin.|[Figure 291]|
|---|---|

|[Figure 292]|[Figure 293]<br><br>Transform the|
|---|---|

|[Figure 294]<br><br>into a waterfall.|[Figure 295]|
|---|---|

stairs

|[Figure 296]|[Figure 297]<br><br>Make the scene look|
|---|---|

|[Figure 298]<br><br>a watercolor painting.|[Figure 299]|
|---|---|

|[Figure 300]|[Figure 301]<br><br>Transform the|
|---|---|

|[Figure 302]<br><br>hand into glass.|[Figure 303]|
|---|---|

like

h

|[Figure 304]|[Figure 305]<br><br>Add a treehouse|
|---|---|

|[Figure 306]<br><br>on the branches.|[Figure 307]|
|---|---|

|[Figure 308]|[Figure 309]<br><br>Add a car driving|
|---|---|

|[Figure 310]<br><br>down the road.|[Figure 311]|
|---|---|

###### Figure 10. In-the-wild video edits produced in real time by EgoEdit’s streaming variant EgoEdit-RT on a single H100 GPU.

Offline Models Streaming Models

Input EgoEdit Lucy Edit InsV2V Senorita-2M EgoEdit-RT StreamDiffusionV2 StreamDiffusion

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

|[Figure 320]|[Figure 321]|
|---|---|

|[Figure 322]|[Figure 323]|[Figure 324]|[Figure 325]|
|---|---|---|---|

|[Figure 326]|[Figure 327]|
|---|---|

|[Figure 328]|[Figure 329]|
|---|---|

|[Figure 330]|[Figure 331]|
|---|---|

|[Figure 332]|[Figure 333]|
|---|---|

|[Figure 334]|[Figure 335]|
|---|---|

Change the patterned chart on the left to dark blue while keeping the embroidered cloth on the right unchanged.

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

|[Figure 344]|[Figure 345]|
|---|---|

|[Figure 346]|[Figure 347]|[Figure 348]|[Figure 349]|
|---|---|---|---|

|[Figure 350]|[Figure 351]|
|---|---|

|[Figure 352]|[Figure 353]|
|---|---|

|[Figure 354]|[Figure 355]|
|---|---|

|[Figure 356]|[Figure 357]|
|---|---|

|[Figure 358]|[Figure 359]|
|---|---|

Replace the kitchen with a small home office desk with a laptop and a desk lamp.

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

|[Figure 368]|[Figure 369]|
|---|---|

|[Figure 370]|[Figure 371]|[Figure 372]|[Figure 373]|
|---|---|---|---|

|[Figure 374]|[Figure 375]|
|---|---|

|[Figure 376]|[Figure 377]|
|---|---|

|[Figure 378]|[Figure 379]|
|---|---|

|[Figure 380]|[Figure 381]|
|---|---|

|[Figure 382]|[Figure 383]|
|---|---|

Transform the white-front smartphone with brown case into an open brass pocket compass with its face transformed into a pool of glowing molten rock, [...].

|[Figure 384]|
|---|

|[Figure 385]|
|---|

|[Figure 386]|
|---|

|[Figure 387]|
|---|

|[Figure 388]|
|---|

|[Figure 389]|
|---|

|[Figure 390]|
|---|

|[Figure 391]|
|---|

|[Figure 392]|[Figure 393]|
|---|---|

|[Figure 394]|[Figure 395]|[Figure 396]|[Figure 397]|
|---|---|---|---|

|[Figure 398]|[Figure 399]|
|---|---|

|[Figure 400]|[Figure 401]|
|---|---|

|[Figure 402]|[Figure 403]|
|---|---|

|[Figure 404]|[Figure 405]|
|---|---|

|[Figure 406]|[Figure 407]|
|---|---|

Add knife in hands.

|[Figure 408]|
|---|

|[Figure 409]|
|---|

|[Figure 410]|
|---|

|[Figure 411]|
|---|

|[Figure 412]|
|---|

|[Figure 413]|
|---|

|[Figure 414]|
|---|

|[Figure 415]|
|---|

|[Figure 416]|[Figure 417]|
|---|---|

|[Figure 418]|[Figure 419]|[Figure 420]|[Figure 421]|
|---|---|---|---|

|[Figure 422]|[Figure 423]|
|---|---|

|[Figure 424]|[Figure 425]|
|---|---|

|[Figure 426]|[Figure 427]|
|---|---|

|[Figure 428]|[Figure 429]|
|---|---|

|[Figure 430]|[Figure 431]|
|---|---|

Apply animation art style to the video.

|[Figure 432]|
|---|

|[Figure 433]|
|---|

|[Figure 434]|
|---|

|[Figure 435]|
|---|

|[Figure 436]|
|---|

|[Figure 437]|
|---|

|[Figure 438]|
|---|

|[Figure 439]|
|---|

|[Figure 440]|[Figure 441]|
|---|---|

|[Figure 442]|[Figure 443]|[Figure 444]|[Figure 445]|
|---|---|---|---|

|[Figure 446]|[Figure 447]|
|---|---|

|[Figure 448]|[Figure 449]|
|---|---|

|[Figure 450]|[Figure 451]|
|---|---|

|[Figure 452]|[Figure 453]|
|---|---|

|[Figure 454]|[Figure 455]|
|---|---|

Change the lounge to a quiet library with tall bookshelves and a warm reading lamp.

|[Figure 456]|
|---|

|[Figure 457]|
|---|

|[Figure 458]|
|---|

|[Figure 459]|
|---|

|[Figure 460]|
|---|

|[Figure 461]|
|---|

|[Figure 462]|
|---|

|[Figure 463]|
|---|

|[Figure 464]|[Figure 465]|
|---|---|

|[Figure 466]|[Figure 467]|[Figure 468]|[Figure 469]|
|---|---|---|---|

|[Figure 470]|[Figure 471]|
|---|---|

|[Figure 472]|[Figure 473]|
|---|---|

|[Figure 474]|[Figure 475]|
|---|---|

|[Figure 476]|[Figure 477]|
|---|---|

|[Figure 478]|[Figure 479]|
|---|---|

Morph the porous green pottery sponge into a leather pouch with a dense, moss-like overgrowth [...] bright neon-green phosphorescent pinpricks [...].

|[Figure 480]|
|---|

|[Figure 481]|
|---|

|[Figure 482]|
|---|

|[Figure 483]|
|---|

|[Figure 484]|
|---|

|[Figure 485]|
|---|

|[Figure 486]|
|---|

|[Figure 487]|
|---|

|[Figure 488]|[Figure 489]|
|---|---|

|[Figure 490]|[Figure 491]|[Figure 492]|[Figure 493]|
|---|---|---|---|

|[Figure 494]|[Figure 495]|
|---|---|

|[Figure 496]|[Figure 497]|
|---|---|

|[Figure 498]|[Figure 499]|
|---|---|

|[Figure 500]|[Figure 501]|
|---|---|

|[Figure 502]|[Figure 503]|
|---|---|

Turn the video into a depth map.

|[Figure 504]|
|---|

|[Figure 505]|
|---|

|[Figure 506]|
|---|

|[Figure 507]|
|---|

|[Figure 508]|
|---|

|[Figure 509]|
|---|

|[Figure 510]|
|---|

|[Figure 511]|
|---|

|[Figure 512]|[Figure 513]|
|---|---|

|[Figure 514]|[Figure 515]|[Figure 516]|[Figure 517]|
|---|---|---|---|

|[Figure 518]|[Figure 519]|
|---|---|

|[Figure 520]|[Figure 521]|
|---|---|

|[Figure 522]|[Figure 523]|
|---|---|

|[Figure 524]|[Figure 525]|
|---|---|

|[Figure 526]|[Figure 527]|
|---|---|

Change the lab bench to an electronics workbench with a soldering iron and parts trays.

- Figure 11. Qualitative comparison of EgoEdit and its real-time streaming variant EgoEdit-RT against baselines on EgoEditBench. EgoEdit and EgoEdit-RT consistently perform better than their baselines. Note that Se˜norita-2M uses the first frame from EgoEdit for frame propagation.

Input EgoEdit EgoEdit-DMD EgoEdit-RT

|[Figure 528]|
|---|

|[Figure 529]|
|---|

|[Figure 530]|
|---|

|[Figure 531]|
|---|

|[Figure 532]|[Figure 533]|
|---|---|

|[Figure 534]|[Figure 535]|[Figure 536]|[Figure 537]|
|---|---|---|---|

|[Figure 538]|[Figure 539]|
|---|---|

Transform the white shaker with blue cap into an insulated metal thermos with a tall, cylindrical, slightly tapered stainless-steel body in a matte silver finish [...], capped by a rounded black plastic screw-on lid [...].

|[Figure 540]|
|---|

|[Figure 541]|
|---|

|[Figure 542]|
|---|

|[Figure 543]|
|---|

|[Figure 544]|[Figure 545]|
|---|---|

|[Figure 546]|[Figure 547]|[Figure 548]|[Figure 549]|
|---|---|---|---|

|[Figure 550]|[Figure 551]|
|---|---|

Turn the video into a depth map.

|[Figure 552]|
|---|

|[Figure 553]|
|---|

|[Figure 554]|
|---|

|[Figure 555]|
|---|

|[Figure 556]|[Figure 557]|
|---|---|

|[Figure 558]|[Figure 559]|[Figure 560]|[Figure 561]|
|---|---|---|---|

|[Figure 562]|[Figure 563]|
|---|---|

Morph the bright green floral fabric into a small square, plain white dish towel with a waffle/looped weave that creates visible horizontal ribs [...] frayed edges with a few loose threads [...].

|[Figure 564]|
|---|

|[Figure 565]|
|---|

|[Figure 566]|
|---|

|[Figure 567]|
|---|

|[Figure 568]|[Figure 569]|
|---|---|

|[Figure 570]|[Figure 571]|[Figure 572]|[Figure 573]|
|---|---|---|---|

|[Figure 574]|[Figure 575]|
|---|---|

Replace the white-bezeled smartphone with a calendar app with a palm-sized clear glass vial topped by a silver metal cap, holding a pale bluish liquid with internal frosty swirls; dense white vapor exhales from beneath the cap [...].

Input EgoEdit EgoEdit-DMD EgoEdit-RT

|[Figure 576]|
|---|

|[Figure 577]|
|---|

|[Figure 578]|
|---|

|[Figure 579]|
|---|

|[Figure 580]|[Figure 581]|
|---|---|

|[Figure 582]|[Figure 583]|[Figure 584]|[Figure 585]|
|---|---|---|---|

|[Figure 586]|[Figure 587]|
|---|---|

Add blue cloth in the hands.

|[Figure 588]|
|---|

|[Figure 589]|
|---|

|[Figure 590]|
|---|

|[Figure 591]|
|---|

|[Figure 592]|[Figure 593]|
|---|---|

|[Figure 594]|[Figure 595]|[Figure 596]|[Figure 597]|
|---|---|---|---|

|[Figure 598]|[Figure 599]|
|---|---|

Transform the light-blue patterned fabric being snipped into a sock with a short ankle length and a thick, folded ribbed cuff in a clean, bright white color; the knit fabric should display fine vertical ribbing [...].

|[Figure 600]|
|---|

|[Figure 601]|
|---|

|[Figure 602]|
|---|

|[Figure 603]|
|---|

|[Figure 604]|[Figure 605]|
|---|---|

|[Figure 606]|[Figure 607]|[Figure 608]|[Figure 609]|
|---|---|---|---|

|[Figure 610]|[Figure 611]|
|---|---|

Transform the stainless-steel bowl with sprouts into a smooth, glossy white ceramic mug [...] holding a light brown, milky liquid [...].

|[Figure 612]|
|---|

|[Figure 613]|
|---|

|[Figure 614]|
|---|

|[Figure 615]|
|---|

|[Figure 616]|[Figure 617]|
|---|---|

|[Figure 618]|[Figure 619]|[Figure 620]|[Figure 621]|
|---|---|---|---|

|[Figure 622]|[Figure 623]|
|---|---|

Replace the paperback held open in both hands with a magazine with a white, glossy cover featuring a dark rectangular photo panel near the top [...] make it spiral-bound along the left edge with a visible coiled spine [...].

- Figure 12. Qualitative comparison of EgoEdit at different stages of distillation. EgoEdit indicates the original 80 NFEs model, EgoEditDMD represents the model after the 4-step DMD distillation, EgoEdit-RT represents the final real-time streaming model obtained after Self Forcing distillation.

Input 0% 25% 75% 100%

|[Figure 624]|
|---|

|[Figure 625]|[Figure 626]|
|---|---|

|[Figure 627]|
|---|

|[Figure 628]|[Figure 629]|
|---|---|

|[Figure 630]|
|---|

|[Figure 631]|[Figure 632]|
|---|---|

|[Figure 633]|
|---|

|[Figure 634]|[Figure 635]|[Figure 636]|[Figure 637]|
|---|---|---|---|

|[Figure 638]|
|---|

Swap the metal cup with brown powder for a clear, glossy mason-style glass jar with [...] a small side handle, topped by a threaded screw rim for a lid, [...] filled nearly to the top with an off-white creamy liquid [...].

|[Figure 639]|
|---|

|[Figure 640]|[Figure 641]|
|---|---|

|[Figure 642]|
|---|

|[Figure 643]|[Figure 644]|
|---|---|

|[Figure 645]|
|---|

|[Figure 646]|[Figure 647]|
|---|---|

|[Figure 648]|
|---|

|[Figure 649]|[Figure 650]|
|---|---|

|[Figure 651]|
|---|

|[Figure 652]|[Figure 653]|
|---|---|

|[Figure 654]|
|---|

|[Figure 655]|[Figure 656]|
|---|---|

|[Figure 657]|
|---|

|[Figure 658]|[Figure 659]|[Figure 660]|[Figure 661]|
|---|---|---|---|

|[Figure 662]|
|---|

|[Figure 663]|
|---|

|[Figure 664]|[Figure 665]|[Figure 666]|[Figure 667]|
|---|---|---|---|

|[Figure 668]|
|---|

Transform the aida fabric with embroidered motif into an off-white bandana with a smooth, slightly satiny texture, folded into a neat square; one edge features a short woven fringe with a visible hem and a few loose threads [...].

Transform the light-blue fabric [...] into an amber vial [...]: the glass [...] warm brown‑orange, and from the cork bursts [...] white, spark‑like crystalline flashes at the mouth that give way to a glowing, wispy green vapor [...].

|[Figure 669]|
|---|

|[Figure 670]|[Figure 671]|
|---|---|

|[Figure 672]|
|---|

|[Figure 673]|[Figure 674]|
|---|---|

|[Figure 675]|
|---|

|[Figure 676]|[Figure 677]|
|---|---|

Transform the white aida square into a bone mask resembling an aged human skull with glowing cyan-blue eye sockets and jaw, emitting pale blue frost. Place this in a sunlit sewing nook with a vintage sewing machine [...].

|[Figure 678]|
|---|

|[Figure 679]|[Figure 680]|[Figure 681]|[Figure 682]|
|---|---|---|---|

|[Figure 683]|
|---|

|[Figure 684]|
|---|

|[Figure 685]|[Figure 686]|
|---|---|

|[Figure 687]|
|---|

|[Figure 688]|[Figure 689]|
|---|---|

|[Figure 690]|
|---|

|[Figure 691]|[Figure 692]|
|---|---|

Transform the video into psychedelic poster art style.

|[Figure 693]|
|---|

|[Figure 694]|[Figure 695]|[Figure 696]|[Figure 697]|
|---|---|---|---|

|[Figure 698]|
|---|

|[Figure 699]|
|---|

|[Figure 700]|[Figure 701]|
|---|---|

|[Figure 702]|
|---|

|[Figure 703]|[Figure 704]|
|---|---|

|[Figure 705]|
|---|

|[Figure 706]|[Figure 707]|
|---|---|

Morph the white shaker bottle with blue cap into an ornate silver goblet [...] the bowl glows an intense electric blue from within while the outer rim and upper surface are rimed in white frost and feathery ice crystals [...].

|[Figure 708]|
|---|

|[Figure 709]|[Figure 710]|[Figure 711]|[Figure 712]|
|---|---|---|---|

|[Figure 713]|
|---|

Input 0% 25% 75% 100%

- Figure 13. Qualitative comparison of different variants of EgoEdit trained on a data mixture with reduced portions of EgoEditData. Percentages indicate the proportion of unique source video samples in EgoEditData retained for training.

