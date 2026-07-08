## arXiv:2605.18748v1[cs.CV]18May2026

[Figure 4]

# Aurora: Unified Video Editing with a Tool-Using Agent

##### Yongsheng Yu1, Ziyun Zeng1, Zhiyuan Xiao1, Zhenghong Zhou1, Hang Hua2, Wei Xiong3, Jiebo Luo1 1UofR, 2MIT-IBM, 3NVIDIA

Recent video editing models have converged on a unified conditioning design: a single diffusion transformer jointly consumes text, source video, and reference images, and one set of weights covers replacement, removal, style transfer, and reference-driven insertion. The design is flexible, but it assumes that the user already provides model-ready text, reference images, and spatial grounding for local edits, which real requests often omit. We present Aurora, an agentic video editing framework that pairs a tool-augmented vision-language model (VLM) agent with a unified video diffusion transformer. The VLM agent maps a raw user request to a structured edit plan aligned with the transformer’s conditioning channels, thereby resolving textual and visual underspecification before generation. We train the VLM agent with supervised data for complete edit planning and reference-image selection, together with preference pairs for robust tool use and instruction refinement. We introduce AgentEditBench to evaluate agent-enhanced video editing under textual and visual underspecification. Experiments on AgentEdit-Bench and two existing video editing benchmarks show that Aurora improves over instruction-only baselines and that the VLM agent transfers to compatible frozen video editing models.

Code: https://github.com/yeates/Aurora Homepage: https://yeates.github.io/Aurora-Page

Add a glowing 'Heineken neon sign' to the wall behind the man on the right. Replace the striped shirt held by the woman with a 'Burberry check scarf'.

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

|[Figure 12]|
|---|

Image Web Search Tool

|[Figure 13]|
|---|

Image Web Search Tool

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

|[Figure 28]|
|---|

|[Figure 29]|
|---|

Segmentation Tool

Image Web Search Tool

Replace the photograph of a woman walking on a beach with a print of

Remove only the young woman from the scene.

'The Great Wave off Kanagawa'

Figure 1 Aurora converts underspecified instructions into model-ready video-editing conditions. The VLM agent invokes web image search to recover the missing reference images for a Heineken neon sign, a Burberry check scarf, and a print of The Great Wave off Kanagawa, and invokes segmentation to recover the missing mask for a person removal.

1

Edited Video

###### Aurora Agent

[Figure 36]

Help me erase the pedestrian and replace the stop sign with the Uniqlo logo in the uploaded video:

[Figure 37]

<Image Web Search Tool>

Finetuned MLLM

[Figure 38]

[Figure 39]

Tool Trigger

<Segmentation Tool>

Video Editing Diffusion Model

[Figure 40]

[Figure 41]

[Figure 42]

rewrite

| | |
|---|---|
| | |

[Figure 43]

[Figure 44]

Uniﬁed Condition Tuple Multi-modal Context Encoder

task route

- Figure 2 Aurora pipeline. The Aurora VLM agent (a LoRA-finetuned Qwen3-VL-8B-Instruct [24]) parses the raw user

request, triggers tools, rewrites the instruction, and routes the task. The output is the unified conditioning tuple (𝑦′,𝑉𝑠𝑟𝑐, 𝑅+) that the Aurora video diffusion transformer consumes; the task label is part of the agent plan and supports the instruction rewrite, but is not fed to the video model.

### 1 Introduction

Recent video editing models increasingly accept several conditioning inputs in one model: text specifies the edit, the source video supplies content to preserve, and optional reference images or masks specify appearance and location [36, 16, 19, 39]. This design lets one set of weights support replacement, removal, style transfer, and reference-driven insertion, but it also assumes that each edit request is provided with the appropriate conditioning inputs, including a precise text instruction, a reference image that specifies the visual characteristics of a requested real-world entity or visual pattern, and a mask when the edit is spatially localized. However, in practice, user requests are often incomplete with respect to these conditioning channels. We categorize this incompleteness along two axes. Visual underspecification occurs when the instruction specifies an edit target or region without providing the visual input needed to ground it. For example, “Replace the striped shirt held by the woman with a Burberry check scarf” names a specific visual pattern but provides no reference image. Textual underspecification occurs when the desired edit is described only indirectly, as in “Change the blue item to a vehicle that runs much faster”, where the model must infer the concrete editing goal. Thus, unified conditioning video editing models provide channels for text, reference images, and masks, but they do not completely interpret how missing conditions should be inferred, grounded, or filled from underspecified user requests. Figure 1 shows underspecified instructions that need additional conditions.

This gap is best understood as a condition-construction problem. Unified video editing models already define the conditioning tuple they expect, but what remains unresolved is how to construct that tuple from a user’s raw request. The missing piece may be a more complete instruction, a reference image that specifies the visual characteristics of a requested real-world entity or visual pattern, or spatial grounding for a local edit. Each arises before video generation begins. Accordingly, we decouple condition construction from video generation.

We present Aurora, an agentic video editing framework built around this split between the VLM agent and video model, shown in Figure 2. Our design has two parts. First, a tool-augmented VLM agent maps a source video and a raw user request to a four-field edit plan: a rewritten instruction, a task label, an optional image-search query, and an optional mask phrase. The first two fields resolve textual underspecification through rewriting and task routing; the latter two recover missing visual conditions through web image search and grounded segmentation. When the source clip and the user’s text already specify the edit, the tool fields remain empty and the framework reduces to ordinary instruction-conditioned editing.

Second, Aurora uses a unified video diffusion transformer (DiT) whose conditioning paths match those plan fields. Built on Wan2.2-TI2V-5B [34], the Aurora video DiT consumes the resulting conditions along two paths: a shared multimodal context for the rewritten instruction, sampled source frames, and reference images, and a latent token sequence where source and reference tokens accompany the noisy video tokens. The video DiT itself does not rewrite instructions or call tools. It consumes the conditioning tuple produced by the VLM agent.

We also introduce AgentEdit-Bench, a video editing benchmark designed to evaluate this two-part framework jointly rather than each part in isolation. Each case is built around a request that is underspecified along either the textual or the visual axis, so the framework succeeds only if it fills in the missing visual input or instruction text with a concrete editing goal. This structure supports three evaluation protocols on the same source clips: our VLM agent with our video DiT, our VLM agent with any unified video editing model, and any video editing model under the raw user instruction alone. The first measures the full framework, the second isolates VLM

agent transfer across video editing models, and the third measures how much the VLM agent helps over an instruction-only baseline. Our contributions are as follows.

- • We present Aurora, an agentic video editing framework that pairs a tool-augmented VLM agent with a unified video diffusion transformer, so that raw user requests are mapped to more complete and model-ready conditions before generation.
- • We construct VLM agent training data for complete edit planning, reference-image selection, and DPO-based preference alignment.
- • We introduce AgentEdit-Bench, a benchmark designed to evaluate video editing enhanced by a toolaugmented VLM agent under textual and visual underspecification.

### 2 Related Work

#### 2.1 Instruction-conditioned and unified video editing

Early instruction-conditioned video editing methods adapted image-editing techniques to video by sharing attention across frames [23, 41, 9]. Senorita-2M [48] scales supervision with task-specialist synthesizers, and OpenVE [11] and Ditto [1] place VLMs in the data-construction pipeline to generate instructions and filter low-quality edits. Recent methods move toward unified video editing models that accept richer conditioning inputs, including instructions, source videos, reference images, masks, in-context examples, or reasoning frames [15, 19, 36, 16, 39]. As conditioning becomes richer, the bottleneck shifts from what the model can ingest to what the user can reliably provide. Their conditioning interfaces accept references and masks as inputs, but the burden of producing those inputs, when the raw request omits them, is left to the user or to upstream pipelines. Aurora closes this gap with a trained tool-augmented VLM agent that reads the raw request, retrieves a reference image when needed, grounds a mask when a spatial region must be localized, and rewrites the instruction into a model-ready form, then hands a complete conditioning tuple to the video editing model.

#### 2.2 Agentic visual generation and video agents

The closest related work is in image generation, where language-model agents retrieve external knowledge or reference images, decompose user requests, choose tools, and score intermediate results with visual feedback or learned rewards [7, 42, 20, 40, 45]. Their shared pattern is to keep planning outside the generator, so that the generator receives more complete conditions. Existing video-domain agents target complementary settings: generalist video generation, compositional text-to-video generation, model selection, or multi-step workflows over model libraries [44, 14, 33, 18, 13]. The Aurora VLM agent addresses a different problem in this design space: rather than coordinating a library of video models, it translates a raw instruction into the conditions required by a single unified video DiT.

### 3 Method

Aurora pairs a tool-augmented VLM agent with a unified video DiT. Aurora uses two separate Qwen-family checkpoints: the VLM agent is a LoRA-finetuned Qwen3-VL-8B-Instruct [24] that emits the edit plan, and the multimodal context encoder inside the video DiT is a frozen Qwen3.5-4B [25] that turns the plan and source video into cross-attention tokens. The two share no weights and play distinct roles throughout the paper. We write 𝜋𝜙 for the VLM agent, where 𝜙 collects the trainable parameters (the LoRA adapter on top of Qwen3VL-8B-Instruct described in Sec. 3.1). Given a source video 𝑉src ∈ ℝ𝐹×𝐻×𝑊×3, a raw user instruction 𝑦, and an optional user reference-image set 𝑅 = {𝑟1, . . . , 𝑟𝐾}, the target edited video 𝑉edit should realize the requested change while preserving the identity, motion, and background structure of 𝑉src wherever the instruction does not ask for a change. The method has three parts. The VLM agent first maps (𝑉src, 𝑦, 𝑅) to a complete edit plan and constructs concrete conditions from optional tool calls. The resulting conditions are then assembled into a unified conditioning tuple used by the video DiT. Finally, the video DiT is trained with flow matching, while the VLM agent is trained with supervised planning and preference alignment.

Ref. Header Header Image Header

Ref. Image

System Prefix Header Input Video Frames

Instruction

|V|
|---|

"Source Video: " "Reference :" "Editing Instruction: "

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

V 0 T

|T|
|---|

[Figure 45]

[Figure 46]

[Figure 47]

###### Qwen 3.5-4B MLLM

Context Projector

Multi-modal Context Tokens

Last Hidden States

Dropped Prefix

[Figure 48]

[Figure 49]

- Figure 3 Multimodal context encoder. A frozen Qwen3.5-4B MLLM encodes the instruction, sampled source frames, and reference images into a shared multimodal context that feeds DiT cross-attention.

- 3.1 Agent planning and condition construction The VLM agent first predicts a typed edit plan:

(𝑦′, 𝑐, 𝑞, 𝑚) = 𝜋𝜙(𝑉src, 𝑦, 𝑅), (1)

where 𝑦′ is the rewritten instruction used by the video DiT, 𝑐 ∈ C is a task label, 𝑞 ∈ Σ∗ ∪ {∅} is an optional image-search query, and 𝑚 ∈ Σ∗ ∪ {∅} is an optional mask phrase. We constrain the output space so that user-provided references suppress web image search, and a non-empty search query is allowed only for searcheligible tasks such as addition, replacement, and background change. The plan fields separate the two forms of underspecification introduced in Section 1. The rewritten instruction 𝑦′ standardizes the free-form user request into a model-ready form that stays consistent with the task label 𝑐. The image-search query 𝑞 retrieves a reference image that specifies the visual characteristics of a requested real-world entity or visual pattern, and the mask phrase 𝑚 localizes spatial regions described in the instruction.

Only 𝑞 and 𝑚 trigger external tools. We use T for these tool operators. Both produce reference-image assets of the same type as the user references in 𝑅, so the downstream video DiT does not need a separate mask branch. Tsearch denotes web image retrieval (via the Serper API [29]) followed by reference-image selection, returning a single selected image. Tmask denotes grounded segmentation followed by mask compositing onto 𝑉src, returning a masked image of the same resolution. A tool-execution stage converts the non-empty plan fields into these reference assets and merges them with any user references:

𝑅+ = 𝑅 ∪ Tsearch(𝑞) ∪ Tmask(𝑚,𝑉src). (2)

An empty plan field denotes no tool call: we adopt the convention Tsearch(∅) = ∅ and Tmask(∅, ·) = ∅, so 𝑞 = ∅ or 𝑚 = ∅ contributes nothing to 𝑅+. After condition construction, the inputs that the video DiT consumes are gathered into a conditioning tuple:

x = (𝑦′,𝑉src, 𝑅+). (3) We use x for this unified conditioning tuple throughout the method. The task label 𝑐 is not fed to the video DiT.

[Figure 50]

|{<br><br>"refined_text_instruction": string, "subtask": canonical_edit_type, "image_search": query | false, "mask": phrase | false,<br><br>} if "image_search" is not false|
|---|

Intermediate Plan JSON

[Figure 51]

|111<br><br>{<br><br>"refined_text_instruction": string, "subtask": canonical_edit_type, "image_search": path | false, "mask": path | false,<br><br>}|
|---|

Output Plan JSON

[Figure 52]

Candidate Images

Serper Image Search

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

GroundingDINO Box

SAM Segmentation

Mask Overlay

Segmentation Tool

[Figure 59]

Masked Image Path

|[Figure 60]|
|---|

Selected Image Path

Image Web Search Tool

[Figure 61]

Finetuned MLLM

if "mask" is not false

h"

Deterministic ﬂow Conditional ﬂow

Figure 4 Protocol for tool-augmented condition construction.

- Figure 4 summarizes this protocol, which turns an intermediate plan into complete conditions. When image search is triggered (the value image_search is not empty), a candidate set is retrieved via the Serper API [29] and a referenceimage selection step keeps a single image. When mask generation is triggered (the value mask is not empty), GroundingDINO [21] and Segment Anything [27] localize the target and compos-

ite a masked image onto 𝑉src. Both outputs are inserted into 𝑅+ as reference images. The VLM agent architecture itself is deliberately simple: Qwen3-VL-8B-Instruct with a lightweight LoRA adapter [24, 12], generating the four-field JSON

real timestep zero-timestep

Reference Index Embedding

[Figure 62]

[Figure 63]

[Figure 64]

| | | | |
|---|---|---|---|
| | | | |

Noisy Tokens Input Video

Multi-modal Context Tokens

###### WAN 2.2 VAE

[Figure 65]

[Figure 66]

[Figure 67]

CrossAttention

###### WAN 2.2-TI2V-5B DiT

+ + +

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

Predicted Velocity Dropped Token

- Figure 5 Aurora video DiT. The video DiT predicts denoising velocity from noisy video tokens while attending to source and reference tokens that are concatenated along the token-sequence dimension under zero-timestep modulation.

output directly. The key design choice is the fixed four-field output format, and VLM agent training does not update the downstream video DiT.

#### 3.2 Unified conditioning in the video DiT

Built on Wan2.2-TI2V-5B [34], the Aurora video DiT consumes the unified conditioning tuple x along two conditioning paths. A multimodal context sequence carries the rewritten instruction together with sampled source frames and reference images into cross-attention, while a latent token sequence carries noisy video tokens together with source and reference tokens into self-attention. Figures 3 and 5 summarize these two paths.

Multimodal context encoder. We use a frozen Qwen3.5-4B multimodal encoder [25] to encode the rewritten instruction 𝑦′, sampled source frames from 𝑉src, and reference images in 𝑅+. The final hidden states are mapped to the DiT cross-attention dimension with a zero-initialized linear projector:

Tctx = Linear MLLM(𝑦′,𝑉src, 𝑅+) . (4)

The multimodal context Tctx enters DiT cross-attention, so the instruction, source frames, and selected references share one multimodal context. This Qwen3.5-4B encoder is distinct from the VLM agent backbone introduced in Sec. 3.1; only an additional linear projector is trained, while the encoder weights stay frozen.

Latent token sequence and zero-timestep modulation. Let E denote the frozen Wan VAE encoder [34], Patchify(·) the DiT patch embedding, and z𝑡 the noisy video latent at flow-matching timestep 𝑡 (defined in Eq. (7)). To preserve unedited content and the identity of reference images, these visual conditions are concatenated along the token-sequence dimension via the operator [ · ; · ]:

Snoisy = Patchify(z𝑡), Ssrc = Patchify(E(𝑉src)), Sref = Patchify(E(𝑟𝑘)) + 𝑒𝑘 𝑟

𝑘∈𝑅+, S = [Snoisy;Ssrc;Sref],

(5)

where 𝑒𝑘 is a learnable reference-index embedding following [37]. If 𝑅+ = ∅, then Sref is empty. Each reference image is VAE-encoded at its own aspect ratio, and only the noisy video latent is decoded at output time. Mask conditioning has no separate branch: a binary mask is first composited as a masked image, then inserted into 𝑅+ like any other reference image.

Noisy tokens, source tokens, and reference tokens play different roles: noisy tokens move along the flowmatching trajectory, while source and reference tokens remain fixed across timesteps and provide conditioning. Wan2.2 DiT blocks receive timestep embeddings through AdaLN modulation [34]. We construct a real-timestep modulation 𝑎(𝑡) and a zero-timestep modulation 𝑎(0), then assign each token a selector 𝜂𝑖:

- 0, 𝑖 ∈ noisy tokens,
- 1, 𝑖 ∈ source or reference tokens.

𝑎𝑖 = (1 − 𝜂𝑖)𝑎(𝑡) + 𝜂𝑖𝑎(0), 𝜂𝑖 =

(6)

Conditioning tokens therefore do not change their modulation with the denoising noise level, while noisy tokens follow the normal flow-matching trajectory.

#### 3.3 Training objectives

Video DiT Training. Given conditioning tuple x, the video DiT predicts a conditional velocity field 𝑣𝜃(z𝑡, 𝑡;x) over VAE latents. Let z1 = E(𝑉edit) be the latent of the target video, let z0 ∼ N(0, 𝐼) be Gaussian noise, and set z𝑡 = (1 − 𝑡)z0 + 𝑡z1. The video DiT is trained with the standard flow-matching objective

LFM(𝜃) = 𝔼𝑡,z0,z1,x ∥𝑣𝜃(z𝑡, 𝑡;x) − (z1 − z0)∥22 , (7)

with samples from image editing, instruction-conditioned video editing, and reference-conditioned video editing mixed under the same objective. The model therefore sees empty-reference edits, user-reference edits, and masked-image edits as instances of one conditioning tuple. Prompt dropout and visual dropout train the unconditional and visual-negative branches that the inference-time guidance schedule uses (Appendix E); the training data mixture is described in Sec. 4.

VLM Agent Supervised Finetuning (SFT). After training the DiT to use unified conditions, we freeze it and train the VLM agent to construct those conditions. Agent SFT has two supervised tasks: complete edit planning and reference-image selection. Let Dplan contain VLM agent context 𝑠 = (𝑉src, 𝑦, 𝑅) and target plan 𝑝★ = (𝑦′, 𝑐, 𝑞, 𝑚). The planning loss is

LSFT(𝜙) = −𝔼(𝑠,𝑝★)∼Dplan log𝜋𝜙(𝑝★ | 𝑠). (8)

The supervision target is not a natural-language rationale. It is the complete edit plan 𝑝★ = (𝑦′, 𝑐, 𝑞, 𝑚) defined in Eq. (1); downstream tool execution then turns this plan into the conditioning tuple in Eq. (3). Reference-image selection uses a separate image-choice set and the same token-level negative log likelihood, but the output is a single candidate index such as Image 3. In this way, SFT directly supervises the inference decisions to rewrite text, route the task, generate a search query, select a reference image, or request a mask.

VLM Agent Preference Alignment. SFT teaches the VLM agent to produce valid plans, but it does not fully resolve borderline cases between similar plans. We therefore apply DPO [26] to plan-level preference pairs. Given VLM agent context 𝑠, chosen plan 𝑝+, rejected plan 𝑝−, current policy 𝜋𝜙, and frozen reference policy 𝜋ref, the objective is

𝜋𝜙(𝑝+ | 𝑠)𝜋ref(𝑝− | 𝑠) 𝜋ref(𝑝+ | 𝑠)𝜋𝜙(𝑝− | 𝑠)

LDPO(𝜙) = −𝔼(𝑠,𝑝+,𝑝−) log𝜎 𝛽 log

, (9)

where 𝜎 is the logistic function and 𝛽 is the DPO temperature. Preference pairs are built around five plan-level boundary cases: source-entity false triggers (image search called for an entity already visible in the source video), ambiguous mask phrases (phrase too vague to ground cleanly), false image-search triggers (search called for edits that need no real-world reference, such as style transfer), constraint-losing rewrites (rewrite drops user-specified constraints from the original prompt), and task-routing confusions (a similar but wrong edit category is chosen). Because the video DiT is frozen throughout VLM agent training, the preference stage sharpens the VLM agent’s decisions without mixing video sampling variance into the learning objective.

### 4 Data Construction

We construct two interdependent datasets. The first trains the unified video DiT to use source, text, reference, and masked-image conditions. The second trains the VLM agent to decide which of those conditions should be supplied at inference time.

Video editing model data. The video DiT is trained entirely on open-source datasets. Table 1 reports the video DiT training mixture. Its three subsets correspond to the conditioning signals each sample teaches: imageedit pairs provide diverse instruction-following supervision, instruction-based video edits provide temporal source conditioning, and reference-guided video edits provide identity, and masked-image conditions. A subset of the open-source video-edit sources is filtered and relabeled with Gemini Flash-Lite [10]; identity references are synthesized with FLUX.2-klein [2]; and masked images are curated separately. Additional source-by-source curation and filtering details appear in Appendix F.1.

UniVideo KiwiEdit Aurora (Ours) Source / Reference

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Add a Whole Foods Market logo to the wall behind the woman.

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Change the background to Andy’s room from Toy Story, featuring the iconic cloud wallpaper, while keeping the girl and her toy unchanged.

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

Remove the hand, wrist, and the yellow rose it is holding. You must also completely erase the sharp, detailed shadow of the hand and rose cast on the wall to the right.

- Figure 6 Qualitative comparison on AgentEdit-Bench. Rows show IP (named real-world entity) addition, IP background change, and localized removal. Target objects to edit are highlighted in (bold blue). Reference images come from Aurora’s web image search tool and are not provided to the baselines in this figure, since this is their default deployment when the

- user request is underspecified. The complementary setup, baselines paired with the same Aurora VLM agent, is reported quantitatively in the ✓ rows of Table 2 and qualitatively in Figures 12 and 13.

- Table 1 Video DiT training data. Counts report accepted training pairs after curation. † marks filtered sources, cap marks re-captioned sources, and syn marks reference synthesis, captioning, or masked-image generation.

Training subset Sources Count Image editing CrispEdit-2M [5], UltraEdit [47], TextEdit [32] 2,388,440 Instruction video editing ReCo [46]†, Ditto [1]†, OpenVE [11]†, EgoEdit [17]†, ROSE [22]cap,

EffectErase [8]cap, Ditto combined-task data [1]†

1,672,008

Reference-guided video editing OpenS2V [43]†syn, Ref VIE [19], SpatialVID [35]†, ROSE [22]syn, EffectErase [8]syn, HuMoSet [4]†

609,988

VLM agent supervised data. The VLM agent is trained with three supervision types, with examples illustrated in Figures 14, 15, and 16. Planning SFT starts from accepted editing pairs and degrades clean instructions into shorter or more ambiguous user requests, with the four-field plan as supervision. Reference-image selection

- uses retrieved candidate sets and supervises which single image should occupy the reference slot. We use 25K planning examples and 10K reference-image selection examples; detailed construction appears in Appendix F.

Preference alignment. Preference pairs target boundary cases that the supervised plans alone do not separate, covering the same five categories defined in Sec. 3.3: source-entity false triggers, ambiguous mask phrases, false image-search triggers, constraint-losing rewrites, and task-routing confusions. Each pair keeps the source and most plan fields fixed while changing one decision axis, so DPO refines the VLM agent’s decision boundaries. The final set contains 1.8K chosen/rejected pairs. The appendix provides illustrated examples.

### 5 Experiments

We organize the experiments around three questions: whether Aurora improves editing under underspecified user requests, which part of the framework is responsible for that gain, and how far the same VLM agent transfers across video editing models. The first question drives our main evaluation, in which AgentEdit-Bench is the primary benchmark for underspecified requests, while EditVerse-Bench [16] and OpenVE-Bench [11] act as sanity checks that the framework remains competitive when the instruction is already well specified. Ablation studies then address the other two questions, isolating the source of the gain and measuring its transfer across video editing models.

- Table 2 Main results on AgentEdit-Bench. Each case is scored by Gemini 2.5 Pro [6] against a rubric, and we report the score as a percentage of that rubric’s maximum (higher is better). Bold marks the best score per column.

###### Method Aurora Agent IP Replace IP Add IP Background Change Reasoning Removal Overall

UniVideo [36] × 67.1 63.4 65.6 77.3 66.5 67.0 UniVideo [36] ✓ 80.3 81.3 66.8 74.6 68.1 76.8 Kiwi-Edit [19] × 66.7 65.1 86.6 79.5 69.0 69.7 Kiwi-Edit [19] ✓ 71.6 73.1 78.3 84.3 57.8 71.7 Aurora (Ours) × 73.0 68.0 71.2 86.4 83.0 74.7 Aurora (Ours) ✓ 89.6 85.3 90.1 88.1 86.8 87.9

#### 5.1 Experiment setup

Benchmarks. EditVerse-Bench [16] is built from 100 source videos and 200 instruction-video pairs spanning 20 edit categories, with horizontal and vertical formats evenly balanced. Following the standard source-video protocol used in current open-source comparisons, we report results on the 120-case subset, which removes tasks that require extra controls such as depth, pose, or sketch inputs. OpenVE-Bench [11] contains 431 video-edit pairs across eight categories. Because our current training data does not cover the creative-edit or camera-edit categories, the table reports results on the remaining six categories.

Implementation details. Aurora video DiT uses a two-stage training recipe: a low-resolution warmup at 399,360 pixels followed by high-resolution finetuning at 921,600 pixels, both with 81 frames. Unless noted otherwise, all reported Aurora numbers use the Stage 2 checkpoint with 50 denoising steps. Benchmarking scores are judged by Gemini 2.5 Pro [6] at temperature=0, while video frame handling and scoring prompts follow the corresponding benchmark-specific evaluation protocol. Appendix E provides more details.

#### 5.2 AgentEdit-Bench

Design goal. Existing editing benchmarks mostly measure prompt following under model-ready instructions. AgentEdit-Bench instead targets requests whose conditions are incomplete: it contains 150 cases spanning five task types, each one built around a request that is underspecified along the textual or the visual axis. It is a VLM-as-judge benchmark: Gemini 2.5 Pro [6] scores each case with a rubric that measures instruction following, edit region localization, source preservation, visual quality, and temporal consistency. The benchmark maps the two underspecification axes to five task types: IP replacement, IP addition, IP background change, localized removal, and reasoning edits. Here, IP refers to a named real-world entity, such as a brand, product, character, or artwork. For the three IP tasks, the rubric also scores IP presence and IP identity match. Appendix G provides the composition, full scoring rubric, judge prompts, and examples.

Quantitative results. Table 2 compares Aurora against UniVideo [36] and Kiwi-Edit [19], and also measures each model with and without the Aurora VLM agent on identical source clips. Even with the agent disabled, the Aurora video DiT alone reaches 74.7 Overall and already outperforms UniVideo (67.0) and Kiwi-Edit (69.7) under the same raw-prompt input, so the gain on AgentEdit-Bench is not driven by the agent stage alone. Adding the VLM agent on top then raises Aurora from 74.7 to 87.9, and the gains are largest on the IP tasks, especially IP replacement and IP addition, where the raw request names an entity but does not provide the reference image the video DiT needs. The VLM agent closes that gap by selecting a concrete reference image before generation instead of leaving the model to infer the target appearance from text alone.

Qualitative comparison. Figure 6 shows that when the request omits the reference image needed for a requested real-world entity or visual pattern, or the spatial grounding needed for a local edit, raw-prompt baselines often synthesize a generic substitute or edit the wrong region, whereas Aurora resolves the missing condition before sampling. Appendix A extends this comparison with additional AgentEdit-Bench cases.

#### 5.3 Results on existing video editing benchmarks

- Table 5 Agent transfer ablation across video editing models on EditVerse-Bench [16] and OpenVE-Bench [11]. † denotes the five-category OpenVE-Bench average excluding Subtitle removal.

(a) OpenVE-Bench† Setting Overall

(b) OpenVE-Bench† Setting Overall

(c) EditVerse-Bench Setting Overall

(d) EditVerse-Bench Setting Overall

Kiwi-Edit 3.02 + Aurora agent 3.29

Aurora (Ours) 3.31 + Aurora agent 3.46

UniVideo 6.12 + Aurora agent 6.48

Aurora (Ours) 7.25 + Aurora agent 7.61

Δ +0.27

Δ +0.15

Δ +0.36

Δ +0.36

We next evaluate Aurora on existing video editing benchmarks whose inputs are already sufficiently specified. This section checks whether the agentic design that helps on AgentEdit-Bench also preserves strong general editing quality outside the underspecified setting. Table 3 reports Overall scores judged by Gemini 2.5 Pro [6] on the 120-case source-video subset of EditVerse-Bench [16] (0– 10 scale) and on OpenVE-Bench [11] (1–5 scale), whose task distribution differs from EditVerse-Bench and includes both scene-level and localized edits. On both benchmarks, the prompts and any required reference images are already supplied by the benchmark itself. We disable the VLM agent’s web image search and mask tools throughout this section; the VLM agent only rewrites the instruction. Aurora is the strongest open-source method on both benchmarks. The closed-source Runway Aleph [28] baseline reaches 3.51 Overall on OpenVE-Bench, 0.13 above Aurora, so Aurora closes most but not all of the open-vs.-closed gap on that benchmark. On OpenVE-Bench, our training data does not include subtitle removal, so the Subtitle column is reported as zero-shot generalization; even under this stricter setting, Aurora is the best in that column, and the 3.38 Overall is therefore a conservative estimate. For a like-for-like comparison against Kiwi-Edit [19], which does not natively support subtitle removal, Tab. 5(a)(b) reports a five-category average (†) that excludes Subtitle on the same source clips. Per-category breakdowns and additional qualitative comparisons appear in Tables 6–7 and Appendices B–C.

Table 3 Overall scores on existing video editing benchmarks. Gray rows denote closed-source baselines; “–” means the score is not reported.

###### Method EditVerse-Bench OpenVE-Bench

Runway Aleph [28] 7.17 3.51 EditVerse [16] 7.52 –

VACE [15] 4.60 1.55 InsViE [38] – 1.58 LucyEditDev [31] 4.65 2.12 Ditto [1] – 2.25 OpenVE-Edit [11] – 2.57 UniVideo [36] 6.12 3.10 Senorita-2M [48] 6.54 – Kiwi-Edit [19] 7.00 – Aurora (Ours) 7.61 3.38

#### 5.4 Ablation study

###### Table 4 Ablation of agent training stages on AgentEdit-Bench.

After establishing Aurora’s performance on both underspecified requests and existing benchmark instructions, we turn to ablations that isolate where the gain comes from and how far the VLM agent transfers across video DiTs.

###### Configuration Overall

Aurora, no agent 74.7 + agent, SFT only 85.0 + agent, SFT + DPO 87.9

VLM agent training stages. Table 4 ablates the VLM agent on Aurora’s own video DiT. Without an agent, the video DiT scores 74.7 on AgentEditBench; supervised planning raises this to 85.0, and DPO adds a further 2.9 points to 87.9. Supervised planning therefore carries most of the gain by teaching the VLM agent which conditioning channels each request needs, while DPO sharpens the boundary cases defined in Sec. 3.3 that SFT leaves ambiguous, such as false image-search triggers, ambiguous mask phrases, and task-routing confusions.

VLM agent transfer across video editing models. Table 5 tests whether the Aurora VLM agent still helps after we keep it fixed and swap only the video editing model. Each pair compares the same model with and without the VLM agent’s outputs. On OpenVE-Bench, adding the VLM agent improves Kiwi-Edit [19] from 3.02 to 3.29 and improves our video DiT from 3.31 to 3.46. The two OpenVE-Bench panels (†) report a five-category average that excludes Subtitle removal, so each model is evaluated on a category set it natively supports. On EditVerse-Bench, the same setup improves UniVideo from 6.12 to 6.48 and improves our video DiT from 7.25 to 7.61. Together with the transfer rows in Table 2, these results show that the VLM agent remains useful

when paired with frozen downstream video editing models. Both benchmarks already supply prompts and any required reference images or masks, so the VLM agent’s image search and mask tools are disabled here and its only job is to rewrite the instruction. The gains in Table 5 therefore show two things directly: the Aurora VLM agent is effective even as a pure instruction-rewriting module for unified video editing, and its benefit is not tied to a single video editing model but carries over to the other unified video editing models we test. This model-agnostic evaluation is made possible by the open-source releases of UniVideo [36] and Kiwi-Edit [19], which we thank. Appendix D extends this transfer study with qualitative illustrations in Figures 12 and 13.

### 6 Limitations

Aurora has two potential limitations. First, we align the VLM agent only with offline DPO (Sec. 4); online RL such as GRPO [30] over the full editing pipeline, where the reward would come from the actually edited clip rather than from the plan, is impractical under our compute budget. Few-step distillation of the video editing DiT, combined with joint online RL training [3] of the VLM agent and the distilled DiT, is a natural way to close this gap. Second, the 5B video DiT handles removal, stylization, and insertion of objects with limited motion reliably, yet adding a new subject that undergoes large and physically plausible motion remains difficult. We attribute this to both the motion-modeling capacity of a 5B backbone and the coverage of large-motion insertion cases in our editing data. We plan to address this in future work using a larger backbone and a broader training mixture.

Broader Impacts. Like other video editing models, Aurora can be misused for non-consensual identity manipulation, and the image-search tool adds copyright and likeness risks from retrieved references. Aurora should therefore be treated as a creative tool, not an evidence-producing one.

### 7 Conclusion

This paper presents Aurora, an agentic video editing framework for user requests that leave out information the video model needs. Aurora places a tool-augmented VLM agent in front of a unified video editing model. The VLM agent rewrites the instruction, assigns a task label, and, when needed, retrieves a reference image or grounds a mask before generation. We also introduce AgentEdit-Bench, a benchmark for textual and visual underspecification that measures the full framework and isolates the contribution of the VLM agent stage on the same source clips. Experiments show that adding the VLM agent raises the overall AgentEdit-Bench score from 74.7 to 87.9 with the same Aurora video DiT. On EditVerse-Bench and OpenVE-Bench, where the inputs are already sufficiently specified, Aurora remains strong and the VLM agent still helps through instruction rewriting alone. The transfer results further show that this benefit is not tied to a single video editing model and carries over to other unified video editing models. Progress in video editing depends not only on stronger video DiTs, but also on reliably turning raw user requests into model-ready conditions.

### References

- [1] Qingyan Bai, Qiuyu Wang, Hao Ouyang, Yue Yu, Hanlin Wang, Wen Wang, Ka Leong Cheng, Shuailei Ma, Yanhong Zeng, Zichen Liu, et al. Scaling instruction-based video editing with a high-quality synthetic dataset. arXiv preprint arXiv:2510.15742, 2025.
- [2] Black Forest Labs. FLUX.2 [klein]: Towards interactive visual intelligence. Blog post, January 2026. URL https:// bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence. Model page: https://bfl. ai/models/flux-2-klein.
- [3] Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Steven Hoi, Zhaohui Hou, Shijie Huang, Dengyang Jiang, Xin Jin, Liangchen Li, et al. Z-image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699, 2025.
- [4] Liyang Chen, Tianxiang Ma, Jiawei Liu, Bingchuan Li, Zhuowei Chen, Lijie Liu, Xu He, Gen Li, Qian He, and Zhiyong Wu. HuMo: Human-centric video generation via collaborative multi-modal conditioning. arXiv preprint arXiv:2509.08519, 2025.

- [5] Wei Chow, Linfeng Li, Lingdong Kong, Zefeng Li, Qi Xu, Hang Song, Tian Ye, Xian Wang, Jinbin Bai, Shilin Xu, Xiangtai Li, Junting Pan, Shaoteng Liu, Ran Zhou, Tianshu Yang, and Songhua Liu. EditMGT: Unleashing potentials of masked generative transformers in image editing. arXiv preprint arXiv:2512.11715, 2025.
- [6] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities, 2025.
- [7] Kaituo Feng, Manyuan Zhang, Shuang Chen, Yunlong Lin, Kaixuan Fan, Yilei Jiang, Hongyu Li, Dian Zheng, Chenyang Wang, and Xiangyu Yue. Gen-Searcher: Reinforcing agentic search for image generation. arXiv preprint arXiv:2603.28767, 2026.
- [8] Yang Fu, Yike Zheng, Ziyun Dai, and Henghui Ding. EffectErase: Joint video object removal and insertion for high-quality effect erasing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.
- [9] Michal Geyer, Omer Bar Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion features for consistent video editing. In ICLR, 2024.
- [10] Google DeepMind. Gemini 3.1 Flash-Lite. Model card, 2026. URL https://deepmind.google/models/gemini/ flash-lite/. Model card PDF: https://storage.googleapis.com/deepmind-media/Model-Cards/ Gemini-3-1-Flash-Lite-Model-Card.pdf.
- [11] Haoyang He, Jie Wang, Jiangning Zhang, Zhucun Xue, Xingyuan Bu, Qiangpeng Yang, Shilei Wen, and Lei Xie. OpenVE-3M: A large-scale high-quality dataset for instruction-guided video editing. arXiv preprint arXiv:2512.07826, 2025.
- [12] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. Iclr, 1(2):3, 2022.
- [13] Hang Hua, Ziyun Zeng, Yizhi Song, Yunlong Tang, Liu He, Daniel Aliaga, Wei Xiong, and Jiebo Luo. Mmig-bench: Towards comprehensive and explainable evaluation of multi-modal image generation models. Advances in Neural Information Processing Systems, 2026.
- [14] Kaiyi Huang, Yukun Huang, Xuefei Ning, Zinan Lin, Yu Wang, and Xihui Liu. Genmac: compositional text-to-video generation with multi-agent collaboration. In Proceedings of the AAAI Conference on Artificial Intelligence, 2026.
- [15] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. arXiv preprint arXiv:2503.07598, 2025.
- [16] Xuan Ju, Tianyu Wang, Yuqian Zhou, He Zhang, Qing Liu, Nanxuan Zhao, Zhifei Zhang, Yijun Li, Yuanhao Cai, Shaoteng Liu, et al. Editverse: Unifying image and video editing and generation with in-context learning. arXiv preprint arXiv:2509.20360, 2025.
- [17] Runjia Li, Moayed Haji-Ali, Ashkan Mirzaei, Chaoyang Wang, Arpit Sahni, Ivan Skorokhodov, Aliaksandr Siarohin, Tomas Jakab, Junlin Han, Sergey Tulyakov, Philip Torr, and Willi Menapace. EgoEdit: Dataset, real-time streaming model, and benchmark for egocentric video editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.
- [18] Zhengyang Liang, Daoan Zhang, Huichi Zhou, Rui Huang, Bobo Li, Yuechen Zhang, Shengqiong Wu, Xiaohan Wang, Jiebo Luo, Lizi Liao, et al. Univa: Universal video agent towards open-source next-generation video generalist. arXiv preprint arXiv:2511.08521, 2025.
- [19] Yiqi Lin, Guoqiang Liang, Ziyun Zeng, Zechen Bai, Yanzhe Chen, and Mike Zheng Shou. Kiwi-Edit: Versatile video editing via instruction and reference guidance. arXiv preprint arXiv:2603.02175, 2026.
- [20] Yunlong Lin, Linqing Wang, Kunjie Lin, Zixu Lin, Kaixiong Gong, Wenbo Li, Bin Lin, Zhenxi Li, Shiyi Zhang, Yuyang Peng, Wenxun Dai, Xinghao Ding, Chunyu Wang, and Qinglin Lu. JarvisEvo: Towards a self-evolving photo editing agent with synergistic editor-evaluator optimization. arXiv preprint arXiv:2511.23002, 2025.
- [21] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection. In European Conference on Computer Vision, 2024.
- [22] Chenxuan Miao, Yutong Feng, Jianshu Zeng, Zixiang Gao, Hantang Liu, Yunfeng Yan, Donglian Qi, Xi Chen, Bin Wang, and Hengshuang Zhao. ROSE: Remove objects with side effects in videos. Advances in Neural Information Processing Systems, 38, 2025.

- [23] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15932–15942, 2023.
- [24] Qwen Team. Qwen3-VL: A family of open multimodal large language models. Model card and blog post, 2025. URL https://qwenlm.github.io/blog/qwen3-vl/.
- [25] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen.ai/blog?id= qwen3.5.
- [26] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, 2023.
- [27] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. In ICLR, 2025.
- [28] Runway. Introducing runway aleph. https://runwayml.com/research/introducing-runway-aleph, 2025. Accessed:

- 2025-09-10.

[29] Serper. Serper: The world’s fastest and cheapest Google search API. https://serper.dev/, 2024. Accessed:

- 2026-04-25.

- [30] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [31] DecartAI Team. Lucy edit: Open-weight text-guided video editing. 2025.
- [32] Changyao Tian, Danni Yang, Guanzhou Chen, Erfei Cui, Zhaokai Wang, Yuchen Duan, Penghao Yin, Sitao Chen, Ganlin Yang, Mingxin Liu, Zirun Zhu, Ziqian Fan, Leyao Gu, Haomin Wang, Qi Wei, Jinhui Yin, Xue Yang, Zhihang Zhong, Qi Qin, Yi Xin, Bin Fu, Yihao Liu, Jiaye Ge, Qipeng Guo, Gen Luo, Hongsheng Li, Yu Qiao, Kai Chen, and Hongjie Zhang. InternVL-U: Democratizing unified multimodal models for understanding, reasoning, generation and editing. arXiv preprint arXiv:2603.09877, 2026.
- [33] Rong-Cheng Tu, Wenhao Sun, Zhao Jin, Jingyi Liao, Jiaxing Huang, and Dacheng Tao. Spagent: Adaptive task decomposition and model selection for general video generation and editing. IEEE Transactions on Image Processing, 2026.
- [34] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [35] Jiahao Wang, Yufeng Yuan, Rujie Zheng, Youtian Lin, Jian Gao, Lin-Zhuo Chen, Yajie Bao, Yi Zhang, Chang Zeng, Yanxi Zhou, Xiao-Xiao Long, Hao Zhu, Zhaoxiang Zhang, Xun Cao, and Yao Yao. SpatialVID: A large-scale video dataset with spatial annotations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.
- [36] Cong Wei, Quande Liu, Zixuan Ye, Qiulin Wang, Xintao Wang, Pengfei Wan, Kun Gai, and Wenhu Chen. Univideo: Unified understanding, generation, and editing for videos. arXiv preprint arXiv:2510.08377, 2025.
- [37] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, Ze Liu, Ziyi Xia, Chaofan Li, Haoge Deng, Jiahao Wang, Kun Luo, Bo Zhang, Defu Lian, Xinlong Wang, Zhongyuan Wang, Tiejun Huang, and Zheng Liu. Omnigen2: Towards instruction-aligned multimodal generation. arXiv preprint arXiv:2506.18871, 2025.
- [38] Yuhui Wu, Liyi Chen, Ruibin Li, Shihao Wang, Chenxi Xie, and Lei Zhang. Insvie-1m: Effective instruction-based video editing with elaborate dataset construction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16692–16701, 2025.
- [39] Xiangpeng Yang, Ji Xie, Yiyuan Yang, Yan Huang, Min Xu, and Qiang Wu. VideoCoF: Unified video editing with temporal reasoner. arXiv preprint arXiv:2512.07469, 2025.
- [40] Mingde Yao, Zhiyuan You, King-Man Tam, Menglu Wang, and Tianfan Xue. PhotoAgent: Agentic photo editing with exploratory visual aesthetic planning. arXiv preprint arXiv:2602.22809, 2026.
- [41] Danah Yatim, Rafail Fridman, Omer Bar-Tal, Yoni Kasten, and Tali Dekel. Space-time diffusion features for zero-shot text-driven motion transfer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8466–8476, 2024.

- [42] Ruijie Ye, Jiayi Zhang, Zhuoxin Liu, Zihao Zhu, Siyuan Yang, Li Li, Tianfu Fu, Franck Dernoncourt, Yue Zhao, Jiacheng Zhu, et al. Agent banana: High-fidelity image editing with agentic thinking and tooling. arXiv preprint arXiv:2602.09084, 2026.
- [43] Shenghai Yuan, Xianyi He, Yufan Deng, Yang Ye, Jinfa Huang, Bin Lin, Jiebo Luo, and Li Yuan. OpenS2V-Nexus: A detailed benchmark and million-scale dataset for subject-to-video generation. Advances in Neural Information Processing Systems, 38, 2025.
- [44] Zhengqing Yuan, Ruoxi Chen, Zhaoxu Li, Haolong Jia, Lifang He, Chi Wang, and Lichao Sun. Mora: Enabling generalist video generation via a multi-agent framework. arXiv preprint arXiv:2403.13248, 2024.
- [45] Ziyun Zeng, Hang Hua, and Jiebo Luo. Mira: Multimodal iterative reasoning agent for image editing. arXiv preprint arXiv:2511.21087, 2025.
- [46] Zhongwei Zhang, Fuchen Long, Wei Li, Zhaofan Qiu, Wu Liu, Ting Yao, and Tao Mei. Region-constraint in-context generation for instructional video editing. arXiv preprint arXiv:2512.17650, 2025.
- [47] Haozhe Zhao, Xiaojian Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. UltraEdit: Instruction-based fine-grained image editing at scale. Advances in Neural Information Processing Systems, 37, 2024.
- [48] Bojia Zi, Penghui Ruan, Marco Chen, Xianbiao Qi, Shaozhe Hao, Shihao Zhao, Youze Huang, Bin Liang, Rong Xiao, and Kam-Fai Wong. Señorita-2m: A high-quality instruction-based dataset for general video editing by video specialists. arXiv preprint arXiv:2502.06734, 2025.

Replace the glowing sphere on the table with an Amazon Echo Dot 4th Gen.

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Add a classic Coca-Cola logo to the red background behind the woman.

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

Remove the green bust of David sitting on the table, including the makeup brushes sticking out of its head. You must also completely remove its shadow and its reflection on the glossy, patterned table surface, accurately reconstructing the colorful pattern underneath.

- Figure 7 Extended qualitative comparison on AgentEdit-Bench. Reference images come from the Aurora VLM agent’s web image search tool and are not provided to the baselines, which see only the original underspecified request.

### A Extended qualitative comparison on AgentEdit-Bench

Figures 7, 8, and 9 extend Figure 6 with additional AgentEdit-Bench cases against the same two baselines (UniVideo [36] and Kiwi-Edit [19]) under their default deployment, in which they use the raw user prompt without a VLM agent. For IP cases, Aurora uses the retrieved reference image from our fine-tuned VLM agent, which is shown in the figure as a small inset overlaid on the first source frame. The cases are grouped by clip orientation: Figure 9 collects landscape clips covering replacement, removal, and addition; Figures 7 and 8 collect portrait clips covering the same five edit types. Across these cases, the two baselines either drop the named entity, substitute a generic one, or fail to remove the target cleanly. In contrast, Aurora more faithfully fulfills the raw user request via the VLM agent’s rewritten instruction and, when needed, tool calls that retrieve a reference image. The gains are clearest on edits that require preserving the identity of a real-world entity.

### B Qualitative comparison on EditVerse-Bench

- Table 6 expands the EditVerse-Bench Overall column from Table 3 into the full per-category breakdown. Figure 10 compares Aurora against four open-source baselines (LucyEditDev [31], VACE [15], Kiwi-Edit [19], UniVideo [36]) on three EditVerse-Bench [16] cases that together cover reference-conditioned insertion, background change (i.e., Bkg.), and reasoning. The first case is a landscape clip shown as the two distant frames sampled at 20% and 80% of the clip to check temporal consistency.

Replace the plain grey back cushion on the chair with a Marimekko Unikko patterned cushion.

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Remove the person’s lower body wearing light blue jeans and purple shoes dancing in the frame. You must also completely remove their large, dynamic shadow cast on the tiled pavement, ensuring the grid lines of the tiles are reconstructed seamlessly without any ghosting.

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

Replace the top-most photograph of the woman on the beach with a print of ‘The Great Wave off Kanagawa’ while maintaining the same size and tape placement.

Figure 8 Extended qualitative comparison on AgentEdit-Bench.

- Table 6 Per-category quantitative comparison on EditVerse-Bench [16]. Scores are obtained using Gemini 2.5 Pro [6]. Gray rows denote closed-source reference methods; bold marks the best open-source score per column. Overall scores are summarized in Table 3.

Method Add Remove Change Style ID Reason Bkg. Color Material Effect Weather Combined Overall

Runway Aleph [28] 6.68 7.40 7.70 7.80 6.45 7.85 7.93 7.95 6.70 6.03 6.40 7.13 7.17 EditVerse [16] 6.93 8.10 7.40 8.73 7.50 7.73 7.68 7.90 7.30 6.25 7.53 7.25 7.52

VACE [15] 2.90 2.28 4.85 7.83 4.18 4.98 5.85 6.30 4.00 3.23 4.80 3.98 4.60 LucyEditDev [31] 4.67 3.70 4.00 4.40 5.30 3.27 4.63 7.50 5.27 4.33 4.30 4.47 4.65 UniVideo [36] 6.00 6.20 5.73 7.80 7.00 5.70 6.97 5.87 4.80 5.30 6.27 5.80 6.12 Senorita-2M [48] 5.83 7.13 6.53 8.33 6.63 6.63 6.63 7.20 6.07 5.17 6.40 5.93 6.54 Kiwi-Edit [19] 6.73 6.85 7.53 8.43 6.75 7.38 6.93 8.10 6.33 6.13 7.18 5.65 7.00 Aurora (Ours) 7.35 8.00 7.03 8.63 7.68 7.70 7.93 7.73 7.15 6.58 7.93 7.55 7.61

C Qualitative comparison on OpenVE-Bench

- Table 7 expands the OpenVE-Bench Overall column from Table 3 into the full per-category breakdown. Figure 11 compares Aurora against UniVideo and Kiwi-Edit on four OpenVE-Bench [11] cases covering local removal, local addition, and local change. These cases are shown as the two distant frames (sampled at 25% and 75% of the clip) to check temporal consistency.

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

Replace the striped shirt held by the woman with a Burberry check pattern scarf, maintaining the way it hangs on the hanger.

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

Remove the white tennis racket resting vertically on the court. You must also completely erase the intricate shadow cast by its frame and strings onto the ground, replacing it with the natural, continuous texture of the clay surface.

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

Add a glowing Heineken neon sign to the empty wall space behind the man on the right.

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

Add a RE/MAX hot air balloon floating in the clear sky above the mountains.

Figure 9 Extended qualitative comparison on AgentEdit-Bench.

### D Qualitative VLM agent transfer across video DiTs

Figures 12 and 13 isolate the VLM agent’s contribution by holding the receiving video DiT fixed and toggling the VLM agent stage on and off. UniVideo [36] and Kiwi-Edit [19] are run twice on the same source clip and the same user request: once on the raw user prompt (their default deployment, labeled w/o agent), and once

LucyEditDev VACE KiwiEdit UniVideo Aurora (Ours) Source

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

Insert a car <image1> driving on the road.

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

Turn the grass into a reflective water surface.

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

Change the blue item to a vehicle that runs much faster.

Figure 10 Qualitative comparison on EditVerse-Bench [16].

- Table 7 Per-category quantitative comparison on OpenVE-Bench [11], judged by Gemini 2.5 Pro [6] on a 1–5 scale. † marks zero-shot generalization: Aurora is not trained on subtitle removal data. Overall scores are summarized in Table 3.

Method #Params. Global Style Bkg. Change Local Change Local Remove Local Add Subtitle Overall Runway Aleph [28] – 3.72 2.62 4.18 4.16 2.78 3.62 3.51 VACE [15] 14B 1.49 1.55 2.07 1.46 1.26 1.48 1.55 InsViE [38] 2B 2.20 1.06 1.48 1.36 1.17 2.18 1.58 Lucy-Edit [31] 5B 2.27 1.57 3.20 1.75 2.30 1.61 2.12 Ditto [1] 14B 4.01 1.68 2.03 1.53 1.41 2.81 2.25 OpenVE-Edit [11] 5B 3.16 2.36 2.98 1.85 2.15 2.91 2.57 Kiwi-Edit [19] 5B 3.64 2.64 3.83 2.63 2.36 – – UniVideo [36] 13B 3.75 2.38 3.78 2.93 2.87 2.91 3.10 Aurora (Ours) 5B 4.07 2.67 4.03 3.73 2.78 2.97† 3.38

on the VLM agent’s rewritten prompt together with the VLM agent’s retrieved reference image (labeled w/ agent). In each case, the third cell shows the first source frame with the retrieved reference image overlaid as

UniVideo KiwiEdit Aurora (Ours) Source

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

Remove the person with long, flowing red hair wearing a light brown, textured coat with a wide collar and tied waist belt from the entire video sequence. The background should be replaced with a clean, uniform light gray studio backdrop.

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

Insert a small, hand-painted ceramic vase onto the dark wooden surface, slightly behind and to the left of the white candle holder. The vase has a glossy white glaze with a delicate blue floral pattern around its midsection.

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

Replace the smooth white table with a luxurious marble surface table.

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

Replace the dark green Dodge Challenger muscle car on the left with a sleek metallic red Dodge Challenger muscle car.

Figure 11 Qualitative comparison on OpenVE-Bench [11].

a small inset. Across these examples, the w/o agent runs usually drop the named entity or replace it with a generic substitute, whereas the w/ agent runs follow the rewritten instruction and use the retrieved reference

w/o agent w/ agent Source / Reference w/o agent w/ agent Source / Reference

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

Add a Jellycat Bashful Bunny plush toy sitting inside the woven basket in the background.

Add a Batman Funko Pop figure on the desk to the left of the keyboard, ensuring it matches the scene’s purple and orange lighting.

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

Add a Kikkoman soy sauce bottle on the white countertop next to the wooden cutting board.

Replace the animal alphabet poster on the wall with a Disney Frozen movie poster.

Figure 12 Qualitative VLM agent transfer on UniVideo [36].

w/o agent w/ agent Source / Reference w/o agent w/ agent Source / Reference

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

Add a Smeg retro toaster on the empty wooden countertop to the right of the woman.

Replace the studded boots on the person’s feet with Nike Air Jordan 1 sneakers, matching the lighting and movement of the foot.

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

Replace the stuffed animal toy held by the girl with a Steiff classic teddy bear.

Replace the potted plant on the table with a Lego Bonsai Tree while preserving the shallow depth of field.

Figure 13 Qualitative VLM agent transfer on Kiwi-Edit [19].

image to better preserve the identity of the requested real-world entity.

### E Experimental details

Tables 8–9 summarize the Aurora video model and training recipe used for the reported results. Table 10 summarizes the per-benchmark inference settings and the VLM agent tool toggles for each evaluation.

Output frames per benchmark. The video DiT always denoises 81 frames internally, matching Wan’s 4𝑘+1 temporal constraint, but the number of frames written to disk differs across benchmarks so that each saved clip matches the protocol of its evaluator. On AgentEdit-Bench, the benchmark consumes the full 81-frame output, so we save all 81 frames. On EditVerse-Bench, the official evaluator expects 64-frame clips, so we keep the first 64 frames of the 81-frame output and discard the rest. On OpenVE-Bench, source clips have variable frame counts, so we temporal-resize each source to 81 frames before denoising and then temporal-resize the 81-frame output back to the source’s original frame count when saving, so that the saved edited clip has the same length as the input.

Inference-time guidance. At inference time, we evaluate three predictions from the same velocity field 𝑣𝜃 trained in Sec. 3.3: 𝑣+ with full text and visual conditions, 𝑣vneg with empty or negative text and visual conditions, and 𝑣∅ with no text and no visual conditions. The guided velocity is

𝑣ˆ = 𝑣∅ + 𝜆img(𝑣vneg − 𝑣∅) + 𝜆txt(𝑣+ − 𝑣vneg). (10)

When 𝜆img = 1, Eq. (10) algebraically reduces to the standard two-pass text CFG path. The unconditional and visual-negative branches that this formula evaluates are exactly the two dropout branches trained by the prompt dropout and visual dropout in Sec. 3.3.

Table 8 Aurora video DiT architecture hyperparameters.

###### Component Item Setting

Video DiT Backbone Wan2.2-TI2V-5B with patch size (1, 2, 2). VAE Model Wan2.2 VAE, frozen, with 16× spatial downsampling. MLLM encoder Model Frozen Qwen3.5-4B in bf16. Context projector Architecture Zero-initialized linear layer. Source conditioning Latent path VAE-encoded source video tokens are concatenated with noisy video

tokens along the token-sequence dimension. Source conditioning MLLM path Images are resized with maximum pixels 147456 before feeding into Qwen. Frames are sampled at 1 fps with at least 2 frames.

- Table 9 Aurora video DiT training parameters. Stage 1 is the low-resolution warmup stage; Stage 2 resumes from the Stage 1 checkpoint stream at the higher pixel budget.

Parameter Stage 1 Stage 2 Notes

Frames 81 81 Wan’s 4k+1 temporal constraint. Video pixel budget 399,360 921,600 Stage 1 is about 448 × 832; Stage 2 is about 708 × 1280 Reference VAE budget 921,600 921,600 Per reference image and source video frame. MLLM source budget 147,456 147,456 Per sampled source video frame. MLLM reference budget 147,456 147,456 Per reference image. MLLM video sampling 1 fps, min 2 frames 1 fps, min 2 frames Qwen video-processor override. Optimizer AdamW AdamW Learning rate and schedule 1 × 10−5, constant 1 × 10−5, constant Weight decay and grad clip 0.01 / 1.0 0.01 / 1.0 Precision and distributed recipe bf16, ZeRO-2 bf16, ZeRO-2 Effective global batch 128 128 Source conditioning mode token-sequence

concat

token-sequence concat

-

Prompt dropout 0.1 0.1 Trains the empty-text branch. Visual dropout given prompt drop 0.5 0.5 Produces the visual-negative and unconditional subsets.

- Table 10 Aurora per-benchmark inference settings. cfg_scale and image_cfg_scale correspond to 𝜆txt and 𝜆img in Eq. (10); when 𝜆img = 1.0 the pipeline falls back to the two-pass CFG path defined in Appendix E. The agent VLM tools (web image search and mask overlay) are disabled on the two already-specified benchmarks, so the agent only rewrites the instruction there.

###### Setting AgentEdit-Bench EditVerse-Bench OpenVE-Bench

Checkpoint Stage 2 Stage 2 Stage 2 Cases 150 (5 edit types) 120 (12 edit types) 358 (6 edit types)

cfg_scale (𝜆txt) 2.0 1.5 2.0 image_cfg_scale (𝜆img) 1.25 1.0 1.0 Two-pass CFG fallback when 𝜆img=1 disabled enabled enabled Denoising steps 50 50 50 Output frames 81 (save 81) 81 (save 64) 81, temporal-resize Max video pixels 921,600 921,600 921,600 Negative prompt empty empty empty

VLM agent base Qwen3-VL-8B-Instruct [24] + Aurora LoRA adapter Agent rewrite (𝑦′) used yes yes yes Agent web image search enabled disabled disabled Agent mask overlay enabled disabled disabled

Judge model Gemini 2.5 Pro [6] at temperature=0

### F Additional VLM agent data details

#### F.1 Data-construction details

Video editing model data. The mixture of training data in Table 1 is built so that instructions, source videos, and reference images all appear under the same conditioning format. Large open-source video-edit sources such as ReCo [46], Ditto [1], EgoEdit [17], and OpenVE [11] are judged by Gemini Flash-Lite [10] on source quality, target quality, edit localization, identity preservation, motion consistency, edit authenticity, and prompt alignment. Accepted samples keep the original instruction when it is faithful; otherwise the judge-provided rewrite is used. The “Ditto combined-task data” row of Table 1 is mined from same-source pairs in Ditto [1]. For each source video 𝑆 with two edited variants 𝑇𝑖 and 𝑇𝑗 produced by instructions 𝐴 and 𝐵, the model is trained on the pair whose input is 𝑇𝑖 and whose target is 𝑇𝑗, so that 𝑆 is never seen and the new instruction must describe the change from 𝑇𝑖 to 𝑇𝑗 using only what is currently visible. Three rule-based filters first discard candidates in which 𝐴 and 𝐵 are both style transfers, both global scene changes, or both manipulate the same object, since those collapse into a single edit. Gemini Flash-Lite [10] then inspects the three frames and accepts a candidate only when the two edits are visually distinct and the change from input to target can be written as one short instruction describable from the input alone. The final subset contains 64,782 pairs.

Planning SFT and reference-image selection. Planning SFT starts from accepted editing data with a clean instruction. We do not ask the teacher to write an explanation; instead, we ask what a real user would most likely omit and how the missing information should be restored into the four-field format. The clean instruction is degraded into a shorter, more ambiguous, more colloquial, or identity-hiding raw request, while the supervision target remains the plan defined in Eq. (1). Sampling is organized by task control rather than raw frequency. Reference-image selection uses another 10K supervised examples adapted from [7]. Each example contains a retrieved entity, web context, and several candidate images; the target is a discrete candidate index such as Image 3, so the VLM agent learns which image should populate the reference slot after retrieval.

- Figure 14 shows the additional planning SFT examples. Each cell pairs two source video frames with the colloquial user request and the four-field edit plan that the VLM agent is supervised to produce. The fifteen cells together illustrate when each plan field carries information versus when it remains false: an identity-bearing target or a named real-world entity triggers image_search; a localized removal triggers mask; a global style change and a combined edit leave both tool fields empty.
- Figure 15 shows the additional reference-image selection SFT subset summarized in Section 4. Once the planner has emitted a non-empty image_search query, web image search returns multiple candidates. The VLM agent’s task here is simple: pick the single image that is visually grounded and factually consistent with the named-entity constraints in the search prompt.

Preference alignment. Preference data targets cases that SFT does not isolate well. Each chosen/rejected pair keeps the source, raw request, and most plan fields fixed while varying one decision axis. The 1.8K-pair set covers the same five plan-level boundary cases defined in Sec. 3.3: source-entity false triggers, ambiguous mask phrases, false image-search triggers, constraint-losing rewrites, and task-routing confusions.

Figures 16 and 17 show the preference alignment data. Each panel keeps the user prompt and shared plan fields fixed and varies one preference axis at a time. The chosen and rejected plan JSONs are shown in full so the reader can see exactly which field flips between the two; that field is marked with a green bar and bold green text in the chosen panel and a red bar and bold red text in the rejected panel.

### G AgentEdit-Bench

AgentEdit-Bench is a 150-case video editing benchmark over five edit types. All source videos of benchmark are drawn from the Pexels stock library. Figure 18 shows nine cases covering all five edit types, with the named target or counterfactual cue and the edit region or physical effect highlighted in the instruction. Figure 19 reports the instruction word cloud and four count distributions: cases per edit type, instruction length, reasoning sub-style, and the physical effects that removal targets carry.

[Figure 200]

SOURCE VIDEO FRAMES

User prompt: “hey, i was just looking at this and wondering if you could maybe take that lifebuoy thing off the wall so it looks plain?”

Output plan JSON: {

refined_text_instruction: “Remove the red and white striped lifebuoy from the center of the tiled wall. Replace the space it occupied with the continuous pattern of the light gray square tiles...”

subtask: remove_object image_search: false mask: “red and white striped lifebuoy from the center of the tiled wall”

}

[Figure 201]

SOURCE VIDEO FRAMES

User prompt: “can u make this look real and not like pixel art anymore thx”

Output plan JSON: {

refined_text_instruction: “Convert this pixel art video back to a live-action, photorealistic format by systematically removing all pixel art characteristics. Restore real-world textures, natural lighting...”

subtask: global_style image_search: false mask: false

}

[Figure 202]

SOURCE VIDEO FRAMES

User prompt: “hey i was just thinking it would be super cool if you could maybe make those toy blocks look like thomas the tank engine.”

Output plan JSON: { refined_text_instruction: “Transform the colorful toy blocks into a

Thomas the Tank Engine toy.” subtask: replace_object image_search: “Thomas the Tank Engine toy” mask: false

}

[Figure 203]

SOURCE VIDEO FRAMES

User prompt: “swap bowl for nutella jar”

Output plan JSON: { refined_text_instruction: “Replace the white bowl filled with hazelnuts

held by hands in the center of the frame with a jar of Nutella.” subtask: replace_object image_search: “jar of Nutella” mask: false

}

[Figure 204]

SOURCE VIDEO FRAMES

User prompt: “change mask pattern”

Output plan JSON: { refined_text_instruction: “Transform the floral face mask into a

geometric patterned mask in gold and black.” subtask: change_color image_search: false mask: false

}

[Figure 205]

SOURCE VIDEO FRAMES

User prompt: “hey could you maybe change the background to 10 downing street and make it look like a still daylight shot instead of the stormy cliff?”

Output plan JSON: {

refined_text_instruction: “Replace the dynamic stormy coastal cliff with a static outdoor scene of 10 Downing Street under consistent daylight with no background motion.”

subtask: change_background image_search: “10 Downing Street” mask: false

}

[Figure 206]

SOURCE VIDEO FRAMES

User prompt: “make music video add flares change colors”

Output plan JSON: {

refined_text_instruction: “Transform the entire video into a cinematic music video aesthetic with slow-motion effects, lens flares, and a color grade that blends deep reds and electric blues into a moody...”

subtask: combined_tasks image_search: false mask: false

}

[Figure 207]

SOURCE VIDEO FRAMES

User prompt: “Put a kid in that shirt on the bench between the two people.”

Output plan JSON: {

refined_text_instruction: “Add a young boy with dark hair, wearing a white t-shirt with two black stripes on the sleeves, looking down, sitting on the bench between the adult and the young girl.”

subtask: add_object image_search: false mask: false

}

[Figure 208]

SOURCE VIDEO FRAMES

User prompt: “make photorealistic”

Output plan JSON: {

refined_text_instruction: “Convert this Abstract Art-animated video to a live-action, photorealistic format. Meticulously strip away the abstract shapes, stylized color fields, and gestural animations...”

subtask: global_style image_search: false mask: false

}

[Figure 209]

SOURCE VIDEO FRAMES

User prompt: “put darth vader on the right chair thx”

Output plan JSON: { refined_text_instruction: “Add Darth Vader sitting on the rightmost

chair.” subtask: add_object image_search: “Darth Vader” mask: false

}

Figure 14 Planning SFT examples.

[Figure 210]

SOURCE VIDEO FRAMES

User prompt: “swap duck joker”

Output plan JSON: {

refined_text_instruction: “Replace the yellow duck mascot on the right side of the frame with a Joker in a white lab coat, plaid shirt, and orange vest.”

subtask: replace_object image_search: “Joker” mask: false

}

[Figure 211]

SOURCE VIDEO FRAMES

User prompt: “Change the hair to long and light colored with that accessory on the end.”

Output plan JSON: { refined_text_instruction: “Change the shoulder-length brown hair into

long, silvery-white hair with a single red ribbon tied at the end.” subtask: change_color image_search: false mask: false

}

[Figure 212]

SOURCE VIDEO FRAMES

User prompt: “put dwight schrute in a blue shirt and green apron holding red stuff standin next to the lady lookin down pls”

Output plan JSON: {

refined_text_instruction: “Add Dwight Schrute wearing a blue shirt and a green apron, holding red produce, looking down, standing to the right of the woman in the field.”

subtask: add_object image_search: “Dwight Schrute” mask: false

}

[Figure 213]

SOURCE VIDEO FRAMES

User prompt: “swich the diner background to monica gellers kitchen from friends w the purple walls and teal cabinets pls thx”

Output plan JSON: {

refined_text_instruction: “Replace the dynamic 1950s diner background with Monica Geller’s kitchen from the TV show Friends, making sure to include its iconic purple walls and teal cabinets.”

subtask: change_background image_search: “Monica Geller’s kitchen from Friends” mask: false

}

[Figure 214]

SOURCE VIDEO FRAMES

User prompt: “hey, the room feels a bit cramped so could you maybe take out that table and just fill in the floor so it looks empty?”

Output plan JSON: {

refined_text_instruction: “Remove the dark brown rectangular wooden table frame located in the center of the room. Replace it by continuing the pattern of the polished parquet flooring...”

subtask: remove_object image_search: false mask: “dark brown rectangular wooden table frame located in the center

of the room” }

#### G.1 Scoring rubric and judge prompt

Each case is scored on the seven axes in Table 11; IP edits use all seven, and reasoning and removal use only axes 1–5. The removal subset adds an anti-replacement clause to axis 1: an output that substitutes a new object for the target is capped at 1, which prevents hallucinated replacements from scoring as clean inpainting. Separating IP Presence from IP Identity Match distinguishes an edit that places something in the right region from one that misses the region entirely. Per-edit-type numbers use the native axis count; the

Search prompt: “A macro photo of the orchid mantis (Hymenopus coronatus) on a flower. A nearby informational card lists the scientific family name and the specific country where the holotype was collected; both must be correct.”

[Figure 215]

- Selected: image_2.

Search prompt: “A detailed 3D visualization of the metal-organic framework MIL-53. A data sidebar on the screen lists its specific BET surface area in square meters per gram and the city of the university in France where it was first developed; both must be correct.”

[Figure 216]

Selected: image_2.

Search prompt: “A macro photograph of a ‘Chuvash Khushpu’ headpiece resting on a pedestal. A small archival tag identifies the village where this specific piece was acquired and lists the precise number of silver coins sewn onto it.”

[Figure 217]

Selected: image_1.

Search prompt: “A photo of an exhibit at the Nobel Peace Center focusing on the organization ‘Nihon Hidankyo’. A wall caption identifies the year the organization was founded and the city where its first general assembly was held.”

[Figure 218]

- Selected: image_3.

Search prompt: “A cinematic shot of the ‘Dungeon Meshi’ character ‘Izutsumi’ in her transformed beast-girl form. A side panel lists her archetype and her listed height; both must be correct.”

[Figure 219]

Selected: image_3.

Search prompt: “A detailed shot of the ‘Jingdezhen Imperial Kiln Museum’ in China, showcasing its brick-vaulted halls. A signage panel lists the architect of record and the year construction was completed.”

[Figure 220]

Selected: image_2.

Figure 15 Reference-image selection data examples.

- Table 11 AgentEdit-Bench scoring rubric. IP edits use all seven axes (max 21); reasoning and removal use axes 1–5 only (max 15).

##### # Axis What it measures

- 1 Instruction Following Whether the requested edit happened in the named region with the named target. The removal subset adds an anti-replacement clause: an output that replaces the target with a new object is capped at 1.
- 2 Edit Region Localization Whether the change stays inside the specified region.
- 3 Source Preservation Whether unedited regions, motion, geometry, and lighting are preserved.
- 4 Visual Quality Realism, lighting and shadow match, scale and perspective, absence of paste-on artifacts.
- 5 Temporal Consistency Whether the edited entity would remain stable across the clip.

- 6 IP Presence Whether the named entity is visible and recognizable in roughly the right region.
- 7 IP Identity Match Whether the visible entity matches the specific real-world identity (brand colors, logo, signature shape).

cross-type aggregate normalizes every case to a 0–21 scale and we report it as a percentage.

The judge runs with temperature=0, and scores each case on three frames sampled at indices 0, 𝑛/3, and 2𝑛/3. Per-axis scores are averaged across the three frames and the raw judge text is persisted next to the parsed scores. Figure 20 shows the verbatim prompts.

User prompt: “make that place look like an old vintage postcard with sepia tones and blurry edges like an old camera.”

[Figure 221]

|✓ Chosen<br><br>refined_text_instruction: “Transform the video into a vintage Hollywood postcard style with soft sepia tones, film grain, and a slightly blurred edge, as if seen through an old camera lens.”<br><br>subtask: global_style image_search: false mask: false|
|---|

|x Rejected<br><br>refined_text_instruction: “Transform the video into a vintage Hollywood postcard style with soft sepia tones, film grain, and a slightly blurred edge, as if seen through an old camera lens.”<br><br>subtask: combined_tasks image_search: false mask: false|
|---|

User prompt: “hey i was looking at this and i think it would be cool if you could maybe take out the guy sitting on the right.”

[Figure 222]

|✓ Chosen<br><br>refined_text_instruction: “Remove the person on the<br><br>right.” subtask: remove_object image_search: false<br><br>mask: “person on the right”|
|---|

|x Rejected<br><br>refined_text_instruction: “Remove the person on the<br><br>right.” subtask: remove_object image_search: false<br><br>mask: “person”|
|---|

User prompt: “add tomatoes bowl”

[Figure 223]

|✓ Chosen<br><br>refined_text_instruction: “Add small red cherry tomatoes into the clear bowl.”<br><br>subtask: add_object<br><br>image_search: false mask: false|
|---|

|x Rejected<br><br>refined_text_instruction: “Add small red cherry tomatoes into the clear bowl.”<br><br>subtask: add_object<br><br>image_search: “NatureSweet Cherubs tomatoes” mask: false|
|---|

- Figure 16 DPO preference examples (1/2).

User prompt: “change that guy in the center to an old man wearing different clothes.”

[Figure 224]

|✓ Chosen<br><br>refined_text_instruction: “Replace the man wearing a blue baseball cap and a grey sweater in the center of the frame with an elderly man with a long white beard, a black hat, and a black robe.”<br><br>subtask: replace_object<br><br>image_search: false mask: false|
|---|

|x Rejected<br><br>refined_text_instruction: “Replace the man wearing a blue baseball cap and a grey sweater in the center of the frame with an elderly man with a long white beard, a black hat, and a black robe.”<br><br>subtask: replace_object<br><br>image_search: “the man wearing a blue baseball cap and a grey sweater”<br><br>mask: false|
|---|

User prompt: “hey i was just thinking it would be cool if you could maybe make that dress pink and put a red flower floating above her.”

[Figure 225]

|✓ Chosen<br><br>refined_text_instruction: “Change the green garment to pink and add a single floating red rose near the top of the scene.”<br><br>subtask: combined_tasks image_search: false mask: false|
|---|

|x Rejected<br><br>refined_text_instruction: “Change the green outfit to<br><br>pink and add a rose near the top.” subtask: combined_tasks image_search: false mask: false|
|---|

User prompt: “add snow animation”

[Figure 226]

|✓ Chosen<br><br>refined_text_instruction: “Apply a dynamic, snowy animation style to this video, ensuring smooth, frame-by-frame consistency. The final output should feature realistic snowfall, icy textures, and a wintery atmosphere...”<br><br>subtask: global_style image_search: false mask: false|
|---|

|x Rejected<br><br>refined_text_instruction: “Apply a snowy animation<br><br>style.” subtask: global_style image_search: false mask: false|
|---|

- Figure 17 DPO preference examples (2/2).

[Figure 227]

ip_object_replace

Replace the glass bowl of salad on the left of the counter with a red Le Creuset Dutch oven.

[Figure 228]

ip_object_add

Add a red KitchenAid stand mixer on the counter in the background.

[Figure 229]

ip_object_replace

Replace the white armchair in the background with a classic Eames Lounge Chair.

[Figure 230]

ip_object_add

Add a Goodyear blimp flying in the empty blue sky to the left of the kite.

[Figure 231]

ip_object_add

Add a bottle of Hendrick’s Gin on the white counter to the left of the tray.

[Figure 232]

ip_background_change

Change the view through the window to a New York City skyline at sunset, keeping the interior unchanged.

[Figure 233]

[Figure 234]

[Figure 235]

reasoning

removal

removal

Turn the lake into a perfect mirror for the passing clouds above.

Remove the white tennis racket on the court, including the intricate shadow cast by its frame and strings on the clay surface.

Remove the ballerina in the middle of the room and reconstruct the bright window, wall, and barre behind her.

- Figure 18 AgentEdit-Bench examples. Each case is one source clip plus one user request; we highlight the named target or counterfactual cue (bold blue) and the edit region or physical effect (italic green).

[Figure 236]

(a) Instruction word cloud, generic edit verbs and stop words removed.

[Figure 237]

(b) Cases per edit type, instruction length, reasoning sub-style, and removal physical effects.

- Figure 19 AgentEdit-Bench distribution. The vocabulary in (a) is dominated by scene and entity nouns, and the four count plots in (b) show the case mix and the substructure inside the reasoning and removal subsets.

Judge prompt: 7-axis IP rubric (IP edit types)

You are a meticulous video editing quality evaluator. Compare the BEFORE frame and the AFTER frame against the editing instruction and a named target entity, and score the edit on seven axes. Editing instruction: {prompt} Target entity: {target_entity} Edit region: {edit_region} For each axis, return an integer from 0 (worst) to 3 (best) with a brief justification. Do not exceed 3.

- 1. Instruction Following (0–3). Was the requested edit performed in the named region with the named target?

3: requested edit performed correctly in the right region. 2: edit performed but with a minor inaccuracy or omission. 1: edit partially performed or applied to the wrong region. 0: instruction ignored, or the opposite was done.

- 2. Edit Region Localization (0–3). Was the change confined to the specified region?
- 3. Source Preservation (0–3). Are subject motion, geometry and lighting outside the edit region preserved?
- 4. Visual Quality (0–3). Realism, seamless integration, lighting and shadow match, scale and perspective.
- 5. Temporal Consistency (0–3). Inferred from the AFTER frame, would the edited entity remain stable across the clip without ghosting or pop-in?
- 6. IP Presence (0–3). Is the named entity actually visible and recognizable in roughly the right region?
- 7. IP Identity Match (0–3). Does the visible entity match the specific real-world identity, brand colors, logo, and signature shape? You may rely only on your internal world knowledge of the brand or product.

Return your evaluation in exactly this format: Instruction Following: [score] - [one-sentence justification] Edit Region Localization: [score] - [one-sentence justification] Source Preservation: [score] - [one-sentence justification] Visual Quality: [score] - [one-sentence justification] Temporal Consistency: [score] - [one-sentence justification] IP Presence: [score] - [one-sentence justification] IP Identity Match: [score] - [one-sentence justification] Total: [sum of the seven scores]

Judge prompt: 5-axis non-IP rubric (reasoning, removal)

You are a meticulous video editing quality evaluator. Compare the BEFORE frame and the AFTER frame against the editing instruction, and score the edit on five axes. Editing instruction: {prompt} For each axis, return an integer from 0 (worst) to 3 (best) with a brief justification. Do not exceed 3.

- 1. Instruction Following (0–3). Was the requested edit performed? 3: edit performed correctly. 2: edit performed with a minor inaccuracy. 1: edit partially performed or applied to the wrong region. 0: instruction ignored, or the opposite was done. Removal-only clause : if the model replaced the target with a new object instead of removing it,

give at most 1.

- 2. Edit Region Localization (0–3). Was the change confined to the specified region or, for global edits, to the implied scope?
- 3. Source Preservation (0–3). Are subject motion, geometry and lighting outside the edit region preserved?
- 4. Visual Quality (0–3). Realism, seamless integration, lighting and shadow match, scale and perspective.
- 5. Temporal Consistency (0–3). Inferred from the AFTER frame, would the result remain stable across the clip?

Return your evaluation in exactly this format: Instruction Following: [score] - [one-sentence justification] Edit Region Localization: [score] - [one-sentence justification] Source Preservation: [score] - [one-sentence justification] Visual Quality: [score] - [one-sentence justification] Temporal Consistency: [score] - [one-sentence justification] Total: [sum of the five scores]

- Figure 20 VLM-as-judge prompts. The two variants share the body and differ in axis count and the removal-only antireplacement clause.

