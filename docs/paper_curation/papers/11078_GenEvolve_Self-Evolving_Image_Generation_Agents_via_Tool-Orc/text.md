# arXiv:2605.21605v2[cs.CV]22May2026

## GenEvolve: Self-Evolving Image Generation Agents via Tool-Orchestrated Visual Experience Distillation

Sixiang Chen1,2 Zhaohu Xing1 Tian Ye1 Xinyu Geng3 Yunlong Lin Jianyu Lai1,2 Xuanhua He3 Fuxiang Zhai1 Jialin Gao4,† Lei Zhu1,3

1The Hong Kong University of Science and Technology (Guangzhou) 2Meituan 3The Hong Kong University of Science and Technology 4National University of Singapore Project Repo: https://ephemeral182.github.io/GenEvolve/

### Abstract

Open-ended image generation is no longer a simple prompt-to-image problem. High-quality generation often requires an agent to combine a model’s internal generative ability with external resources. As requests become more diverse and demanding, we aim to develop a general image-generation agent that can selfevolve through trajectories and use tools more effectively across varied generation challenges. To this end, we propose GenEvolve, a self-evolving framework based on Tool-Orchestrated Visual Experience Distillation. In GenEvolve, each generation attempt is modeled as a tool-orchestrated trajectory, where the agent gathers evidence, selects references, invokes generation skills, and composes them into a prompt-reference program. Unlike existing agentic generation methods that mainly rely on image-level scalar rewards, GenEvolve compares multiple trajectories for the same request and abstracts best-worst differences into structured visual experience, provided only to a privileged teacher branch. Inspired by onpolicy self-distillation, Visual Experience Distillation provides dense token-level supervision, helping the student internalize better search, knowledge activation, reference selection, and prompt construction. We further construct GenEvolve-Data and GenEvolve-Bench. Experiments on public benchmarks and GenEvolve-Bench show substantial gains over strong baselines, achieving state-of-the-art performance among current image-generation frameworks.

### 1 Introduction

Modern image generators are increasingly powerful, but open-ended image generation is not solved by fidelity alone. Real requests require deciding what the generator already knows, what external facts and references to acquire, which internal generation knowledge to activate, and how to translate these signals into instructions a downstream generator can follow. Thus, high-quality generation is becoming less a one-shot prompt-to-image task than an agentic process of planning, tool orchestration, and feedback-driven adaptation.

This shift is most visible in complex and grounded generation scenarios. A request may involve current or long-tail factual knowledge, reference-specific appearance, multi-source visual evidence, professional design constraints, or implicit user intent that cannot be captured by a single rewritten prompt. Strong generators may possess substantial internal knowledge and visual priors, but they do not decide when to search, how to use internal knowledge, which references are useful, or how failures should guide future behavior. Thus, the key challenge is not simply improving local abilities such as text rendering, layout, counting, or attribute binding. Rather, it is to build a general image-generation agent that can coordinate internal generative knowledge with external tools and learn how to use them

Preprint.

[Figure 1]

- Figure 1: Results of GenEvolve. Top: Representative generation results by our self-evolving agent across diverse open-ended and complicated requests covering architecture, creative transfer, scientific illustration, street scenes, and more, using both Nano Banana Pro and Qwen-Image-Edit as downstream generators. Bottom: Quantitative comparison on (a) our GENEVOLVE-BENCH (KScore

+ four judge dimensions and Knowledge-/Quality-Anchored tracks) and (b) the external WISE benchmark, where GenEvolve consistently outperforms SOTA direct generators and recent agentic baselines.

through interaction with the generator. Such coordination requires more than exposing a tool list: the agent must learn when a request needs factual lookup, what queries should be issued, which retrieved images should serve as references, which generation knowledge should be activated via skills, and how these signals should be bound into a generator-facing program.

Recent agentic generation systems have begun to explore this direction. GenAgent treats image generators as invokable tools for multi-turn reasoning, tool use, judgment, and reflection [20]. Gen-Searcher and ORIG improve factual grounding through search- or retrieval-augmented generation [14, 40], while GEMS and Mind-Brush introduce memory, reusable skills, or research-style workflows [18, 17]. Maestro and CRAFT further refine generation with critic feedback, verifier agents, or constraintdriven correction [41, 21]. These systems show the value of search, tools, memory, and iterative refinement, but they usually address only part of the generation process: acquiring external evidence, wrapping a black-box generator, or evolving prompts at inference time. It therefore remains underexplored how to train an open image-generation agent whose tool use, reference selection, knowledge activation, prompt-reference program construction, and generator interaction are optimized together.

Therefore, we propose GenEvolve, a self-evolving framework for image-generation agents based on Tool-Orchestrated Visual Experience Distillation. GenEvolve models each generation attempt as a tool-orchestrated visual trajectory, in which the agent gathers textual evidence, retrieves and selects visual references, invokes callable generation knowledge, and synthesizes a prompt-reference program z = (g,R), where g is a targeted prompt and R is a small set of selected reference images. A reference-conditioned generator then produces the final image, which is evaluated together with the trajectory that produced it via reward calculation and diagnostics. Thus, the learning target is not merely a prompt, but a complete generation trajectory linking tool decisions, generator-facing instructions, generated outcomes, and feedback.

To make this formulation trainable and measurable, we construct GenEvolve-Data and GenEvolveBench. GenEvolve-Data goes beyond ordinary prompt-rewriting corpora by providing toolorchestrated trajectories that teach the agent how to acquire external evidence, activate internal generation knowledge, and construct prompt-reference programs. It further provides filtered GT image cases that make visual feedback meaningful for self-evolution. GenEvolve-Bench evaluates final image quality across Knowledge-Anchored and Quality-Anchored settings, covering both external grounding and quality-sensitive generation requirements.

On top of this trajectory data, GenEvolve turns visual outcomes into structured experience for improving the agent. Existing agentic generation methods can optimize trajectories with image-level scalar rewards, but such rewards indicate which trajectory is better without explaining which decisions caused the improvement. GenEvolve instead compares multiple trajectories for the same request and abstracts best-worst differences into visual experience. Inspired by on-policy self-distillation, this experience is provided only to a privileged teacher branch, while the student acts under the normal inference context. Combined with group-relative policy optimization, Visual Experience Distillation provides dense token-level supervision for better tool orchestration and generator-facing program synthesis. As illustrated in Figure 1, GenEvolve produces high-quality images across diverse openended requests and consistently outperforms strong direct generators and recent agentic baselines on both our GENEVOLVE-BENCH and the external WISE benchmark. Therefore, our contributions are summarized as follow:

- • We propose GenEvolve, which reformulates open-ended image generation as an agentic trajectory learning problem, where a general image-generation agent learns to coordinate internal generative knowledge with external tools, including factual search, visual reference retrieval, callable generation knowledge, prompt-reference program synthesis, image generation, and experience internalization.
- • We first introduce a self-evolving post-training mechanism that compares multiple trajectories for the same request and abstracts best-worst trajectory differences into structured visual experience. The token-level distillation objective builds on established on-policy self-distillation losses, while our contribution is the visual experience construction, retrieval, and teacher-only conditioning for image-generation agents.
- • We construct a trajectory dataset and diagnostic benchmark for general image-generation agents, evaluating both final image quality and agentic behaviors such as tool use, reference selection, skill routing, and prompt-reference faithfulness.
- • Experiments show that GenEvolve achieves the best performance on GenEvolve-Bench and the public benchmark, outperforming raw generators, agentic baselines and further improving with a stronger generator. These results demonstrate the effectiveness and transferability of the learned prompt-reference programs and tool-orchestrated policy.

### 2 Related Work

Image generation models. Image generation has evolved from standalone text-to-image generators to integrated multimodal generation systems. Diffusion and latent diffusion models established high-fidelity prompt-conditioned synthesis [31, 32, 30, 9, 10], while diffusion transformers and their successors, including DiT, PixArt-α, Stable Diffusion 3, FLUX, Hunyuan-DiT, and Nano Banana Pro, further improve scalability, text understanding, and generation quality [28, 8, 13, 22, 39, 16]. In parallel, unified multimodal models such as Chameleon, Emu3, Show-o, BAGEL, OmniGen2, HunyuanImage 3.0, and BLIP3-o explore shared or hybrid architectures for multimodal understanding and generation [38, 45, 48, 11, 47, 6, 7]. Despite their strong rendering ability and multimodal flexibility, these models remain primarily generators: they do not explicitly decide when to acquire missing facts, which references to trust, or which generation knowledge to activate.

Agentic image generation. Agentic generation systems augment image models with planning, retrieval, tool use, judging, or refinement. GenAgent enables multi-turn reasoning, tool invocation, judgment, and reflection around image generators [20]. Mind-Brush [17], Gen-Searcher, and ORIG focus on research/search/retrieval-augmented generation for implicit, dynamic, or factual knowledge [17, 14, 40]. GEMS introduces memory and skills [18], while Maestro and CRAFT use critic/verifier feedback or constraint-driven correction to iteratively improve prompts [41, 21]. These systems show the value of search, tools, memory, and refinement, but they often emphasize one

component of the broader generation process or wrap a generator with an external workflow. Recent commercial systems such as Nano Banana Pro, built on Gemini, point toward tighter integration of reasoning, real-world knowledge, grounding, and visual synthesis [16]. Inspired by this direction, GenEvolve trains an open image-generation agent that coordinates external tools and internal generation knowledge along visual trajectories, and uses visual experience distillation to improve the coupling between the agent policy and downstream generator behavior.

On-policy distillation. On-policy distillation has become a promising post-training paradigm for language models and agents, with variants including OPSD, OPCD, Skill-SD, SDPO, and HDPO [51, 49, 43, 19, 12]. OPSD uses privileged context to supervise on-policy generations [51]; OPCD distills useful in-context knowledge into model parameters [49]; SDPO converts rich feedback into dense self-distillation signals [19]; and Skill-SD summarizes multi-turn agent trajectories into training-only skills with an importance-weighted sampled-token reverse-KL objective [43]. GenEvolve is motivated by this general teacher-only self-distillation principle, but changes the privileged signal and task character: instead of ground-truth reasoning traces or text-agent skills, the teacher receives visual experience extracted from tool-orchestrated image-generation trajectories, helping the student internalize better search, knowledge activation, reference selection, and promptreference program synthesis.

### 3 Tool-Orchestrated Visual Trajectory Formulation

We formalize each generation attempt as a tool-orchestrated visual trajectory. Given a user request x, the agent does not directly generate an image, merely rewrite the prompt, or only retrieve external evidence. Instead, it decides when to acquire external information, which visual references to trust, when to activate internal generation knowledge, and how to synthesize these signals into a prompt-reference program. This makes the generation process observable and trainable, covering both external tool use and internal knowledge activation before generation.

At turn t, the agent observes the interaction history and samples an action: ht = (x,a1,o1,...,at−1,ot−1), at ∼ πθ(a | ht), ot ∼ Tat

(1)

(o | ht),

where at is either a tool call or the final answer, and ot is the corresponding observation. The final answer is a prompt-reference generation program z = (g,R), where g is a targeted generation prompt and R is a small set of selected reference images. A reference-conditioned generator renders yˆ = G(g,R). A complete trajectory is therefore

τ = (x,a1,o1,...,aT,oT,z,y,r,dˆ ), (2) where r is a scalar reward and d contains visual diagnostics. The trajectory-level objective is

Ex,τ[R(ˆy,z,x)]. (3)

max

θ

In GenEvolve, however, reward is not the only learning signal. For the same request, multiple trajectories may produce different visual outcomes. GenEvolve compares high- and low-reward trajectories and converts their differences into structured visual experience M, which is provided only to a privileged teacher branch during self-distillation.

This formulation differs from prior agentic generation systems in both optimization scope and supervision source. Many existing methods expose external interfaces such as search, retrieval, judging, or prompt correction around black-box or loosely coupled generators. GenEvolve instead treats the whole generation process as the learnable object: external tool use, internal generationknowledge activation, reference selection, and prompt-reference synthesis are modeled as trajectory decisions. By distilling visual experience into the student policy, GenEvolve teaches not only which trajectory is better, but which orchestration behaviors should be reused for future requests.

### 4 GenEvolve-Data and GenEvolve-Bench

Before introducing the learning algorithm, we first define the data substrate that enables toolorchestrated visual trajectory learning. As shown in Figure 2, GenEvolve-Data is constructed

###### GenEvolve-Data Construction and GenEvolve-Bench

[Figure 2]

###### PromptPool Teacher Trajectory VLM Filtering GT Images & Splits GenEvolve-Bench

[Figure 3]

Generatedimages

Filtered images

###### High-qualityData

###### Knowledge-Anchored

[Figure 4]

Search

Facts, entities, Events, places

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Ref-Image

Search

Quality-Anchored Diversity, balance, Clarity, difficulty

[Figure 13]

Judgeinspects:

…

~3200 GT image cases

Knowledge-Anch. Quality-Anch.

Query Knowledge

[Figure 14]

trajectory

SFT

[Figure 15]

[Figure 16]

[Figure 17]

KScore

Final

Train +Eval (w/o. GT Image)

References

[Figure 18]

###### Kept ~70%

program

###### 60% 40%

Evaluates

[Figure 19]

Self-evolution+Bench

Aesthetic Faithful

[Figure 20]

Final Program

open-ended complicated

/

~20000valid trajectories

[Figure 21]

Visual Consis Text

85% train held-out

~20000prompts(16 categories)

cases.

Representative Sample

###### Prompt

###### Thinking&ToolCall

###### Thinking&ToolCall

[Figure 22]

<think>\nOkay, the first image search gave me Antikythera

<think>\nGot it, let's tackle this. First, I need to get visual references for the key subjects here: the Antikythera mechanism main fragment,

Generate an image in the style of a watercolor

mechanism references. IMG_001 looks like a clear shot of the corroded green bronze fragment, that's good for the first reference.

museum concept sketch. In the center, depict the

Tool Response &

the Bismuth crystal's structure and iridescent colors, right? Wait also,

main fragment of the Antikythera mechanism made of heavily corroded green bronze. Directly

Selection <thinking>

Now I need a visual reference for the bismuth crystal with the stair-

the style is watercolor museum concept sketch…, let's do an image

search for \"Antikythera mechanism main corroded green bronze

step geometric structure and iridescent colors. So next I should

beneath it, acting as a pedestal, is a massive

fragment\" to get the correct look of that. Then another one for \"Bismuth crystal geometric stair step iridescent\" to get the crystal's

call image_search for that query. Let's do that

Bismuth crystal showing its signature geometric

now.\n</think>\n<tool_call>\n{\"name\": \"image_search\", \"arguments\": {\"query\": \"massive bismuth crystal geometric

stair-step structure and iridescent metallic colors. In the upper left corner of the background, there

appearance. \n</think>\n<tool_call>\n{\"name\": \"image_search\",

\"arguments\": {\"query\": \"Antikythera mechanism main fragment

stair step structure iridescent metallic\"}}\n</tool_call>

should be a frosted glass plaque with the words

heavily corroded green bronze\"}}\n</tool_call>

'ANCIENT TECH' cleanly engraved into it.

###### Thinking&ToolCall

[Figure 23]

###### Tool Response &

[Figure 24]

[Figure 25]

<think>\nGreat, now I have references for both key objects: IMG_001 for the Selection <thinking>

On-demand Always Loaded

Antikythera fragment, IMG_004 for the bismuth crystal. Now I need to get the

[Figure 26]

Instruction Description Name

relevant skill guidance. Let's list the skills needed:\n1. spatial_layout: we have

specific positions for all elements (center: fragment, directly beneath: bismuth pedestal, upper left: plaque) so that's needed.\n2. aesthetic_drawing: we need

- Des._1
- Des._2
- Des._3 Des._n

- Instruction_1
- Instruction_2
- Instruction_3 Instruction_n

- skill_1
- skill_2
- skill_3 skill_n

Targeted Prom.

[Figure 27]

watercolor museum concept sketch style, so lighting, texture, style guidance

here.\n3. text_rendering: the engraved \"ANCIENT TECH\" on frosted glass

Tool Response & Final <thinking>

Tool Response &

plaque, need guidance for that.\n4. physical_material_consistency: need to correctly render corroded green bronze, iridescent bismuth, frosted glass

[Figure 28]

Knowledge activation

materials. ... .\n</think>\n<tool_call>\n{\"name\": \"query_knowledge\",

\"arguments\": {\"skill_name\": \"spatial_layout\"}}\n</tool_call>

Scalable internal knowledge activation

- Figure 2: Overview of GenEvolve-Data and GenEvolve-Bench. The top row presents the construction pipeline: diverse prompts are converted into tool-orchestrated teacher trajectories, audited by VLM-based checks, used to generate and filter GT image cases, and split for supervised training, self-evolution, and held-out evaluation. The bottom row illustrates a representative case, showing how the agent retrieves visual references, activates generation knowledge, and composes a prompt-reference program for grounded image generation.

as a complete generation pipeline rather than a prompt-rewriting corpus: diverse prompts are solved by teacher agents through tool use, audited by VLM filters, rendered into GT image cases, and split for supervised cold start, self-evolution, and held-out evaluation.

Prompt pool. We construct natural user requests from structured recipes specifying the task family, missing external evidence, visual anchor, dominant generation requirement, and difficulty. The pool contains two complementary tracks. Knowledge-Anchored prompts require external grounding for entities, events, places, objects, or visual facts, while Quality-Anchored prompts emphasize quality-sensitive generation requirements such as text layout, spatial composition, counting, anatomy, material consistency, aesthetics, and creative transfer. These recipe fields are used for coverage control and stratified splitting, but are not exposed to the agent as task labels.

Teacher trajectories. Each validated prompt is converted into a teacher trajectory through a real multi-turn tool loop. We use Seed2.0 and Gemini 3 Pro as teacher models, leveraging their multimodal understanding, reasoning, and agentic capabilities to issue textual search queries, retrieve visual references, activate generation knowledge, and synthesize the final prompt-reference program [34, 15]. The tool order is request-dependent: knowledge-heavy cases may begin with factual lookup, reference-sensitive cases rely more on image search, and quality-anchored cases may activate generation knowledge for text, layout, pose, material, or style control. Each trajectory records the tool observations, selected references, intermediate rationale, and final program used for generation.

Trajectory filtering. We audit teacher trajectories before using them for training. Programmatic checks remove incomplete tool loops, invalid reference selections, raw URL or ID leakage, missing ordinal reference wording, and underspecified final programs. A VLM judge then reviews whether the selected references support the requested visual details, whether the collected evidence is actually used, and whether the final program integrates the required constraints. This produces a high-quality trajectory set for SFT cold start.

GT images and splits. For self-evolution and evaluation, high-quality teacher programs are rendered into GT image cases using Nano Banana Pro, which is built on Gemini 3 Pro Image and is designed for high-quality image generation/editing with strong text rendering, visual control, and real-world knowledge [16]. A second visual filter checks prompt compliance, reference usage, visual coherence,

###### On-policy Tool-Orchestrated Visual Trajectory Deployment

Generation Preparation Conditioned Generation Rewards &Diagnostics

GenEvolveAgent

Prompt-Reference Program (synthesize 𝑧 = (𝑔, 𝑅))

Reference-Conditioned

UserRequest

Search

ImageReward

Generator 𝐺

Tool Orchestration

[Figure 29]

Targeted generation

[Figure 30]

Ref-Image

instruction 𝑔

Search

Text Reward

[Figure 31]

Selected

references 𝑅

Query

Knowledge

Knowledge Diagnostics

…

Request + facts+ refs +

Render the final image

knowledge + guards

conditioned on 𝑧 = (𝑔,𝑅)

[Figure 32]

Search and Ref-Image Search: external tools Query Knowledge: internal generation knowledge

[Figure 33]

Self-Evolving Closed-Loop

Improved GenEvolveAgent

Current Visual Experience Distillation

Best / Worst

Vsiual Experience Slots

Roll out𝑲visual trajectories

(UpdatedStudentPolicyπθ)

StudentPolicyπθ

Trajectory Comparison

###### Privileged Teacher

[Figure 34]

[Figure 35]

Search Strategy

- … Traj. 1

[Figure 36]

- … Traj. 2

[Figure 37]

- … Traj. 3

[Figure 38]

[Figure 39]

[Figure 40]

Normalcontext +

𝜏+

Visualexperience

[Figure 41]

Knowledge Actvation

AspectsFive

[Figure 42]

vs.

Reference Selection

[Figure 43]

[Figure 44]

[Figure 45]

𝜏−

Student

Prompt Construction

Normal inference

[Figure 46]

Failure Avoidance

context

Image/ Text reward+diagnostics

Asynchronous Training

Vsiual Experience Self-distillation + GRPO

Betterpolicy Bettertrajectories Betterexperience ℒ = ℒGRPO + λ ℒSDL

- Figure 3: Overview of GenEvolve. The student agent orchestrates external search, visual references, and internal generation knowledge to produce a prompt-reference program z = (g,R). During training, multiple trajectories are judged with image/text rewards; best-worst differences are converted into visual experience and injected only into a privileged teacher. GRPO provides trajectory-level optimization, while Visual Experience Self-Distillation supplies token-level guidance, forming a self-evolving loop of better policy, trajectories, and experience.

and image quality. The surviving cases are exported into two views: an SFT view that preserves full tool-loop trajectories without exposing GT images, and a visual-feedback view that contains the user request, GT image, and metadata for self-evolution and benchmark evaluation. This design enables supervised cold start while preventing the self-evolving agent from simply copying teacher outputs.

GenEvolve-Bench. GenEvolve-Bench is the held-out evaluation split produced by the same pipeline. It evaluates open-ended image-generation agents under a unified KScore [14] protocol, with results reported on both Knowledge-Anchored and Quality-Anchored subsets. The benchmark is designed to test whether agents can combine external evidence, selected visual references, and quality-aware generation control, rather than merely follow generic text-to-image prompts. Overall details and construction statistics of our data are provided in Appendix A.

### 5 Method: GenEvolve

#### 5.1 Tool-Orchestrated Visual Trajectories

Given the trajectory data and GT image cases in Section 4, GenEvolve trains an image-generation agent whose output is produced through a multi-turn visual trajectory rather than a single prompt rewrite. For a user request x, the agent samples τ = (a1,o1,...,aT,oT,z), where each at is a tool call or the final answer, ot is the corresponding observation, and z is an executable prompt-reference generation program. Following a ReAct-style interface, search planning, reference acquisition, internal-knowledge activation, and prompt-reference program construction become explicit trajectory decisions.

The action space contains three tool families. search(q) gathers textual evidence for visible facts, image_search(q) retrieves candidate visual references, and query_knowledge(skill_name) activates internal generation knowledge. We instantiate such internal knowledge as compact callable generation skills, covering text rendering, layout, counting, anatomy, attribute binding, material consistency, aesthetics, and creative transformation. Static generation knowledge remains available to the deployed student through tool calls, while dynamic visual experience is used only during training. The full tool protocol and skill taxonomy are provided in the appendix.

#### 5.2 SFT Cold Start for Tool-Orchestrated Agents

GenEvolve-Data first provides supervised trajectories to cold-start the base MLLM into a toolorchestrated image-generation agent. This stage teaches the model to follow the visual trajectory formulation: when to call tools, how to retrieve and select references, when to activate internal generation knowledge, and how to output a valid prompt-reference program z = (g,R).

Each SFT example contains a user request, a multi-turn tool trajectory, and a final program:

τ⋆ = (a1,o1,...,aT,oT,z⋆), z⋆ = (g⋆,R⋆). (4) We optimize assistant-side trajectory tokens under the observed tool history:

1 i,t mi,t i t

mi,t log πθ(yi,t⋆ | h⋆i,t), (5)

LSFT(θ) = −

where h⋆i,t includes previous tool observations and mi,t masks valid assistant tokens. After this cold start, GRPO and Visual Experience Distillation further optimize the initialized policy with

generated-image feedback, forming the self-evolving stage shown in Fig. 3.

###### 5.3 Prompt-Reference Program and Generation Feedback The final trajectory output is a prompt-reference generation program

z = (g,R), R = {r1,...,rk}, (6)

where g is a targeted generation instruction and R is an ordered set of selected reference images. The instruction refers to selected images by ordinal phrases such as “the first reference image”, rather than raw URLs or retrieval IDs. Program synthesis binds constraints from the user request, retrieved facts, selected references, activated internal generation knowledge, and failure-avoidance experience:

C = Cuser ∪ Cfact ∪ Cref ∪ Cknow ∪ Cavoid. (7) A reference-conditioned generator then produces

yˆ = G(g,R). (8)

For trajectory-level optimization, we follow recent work Gen-Searcher [14] and use dual reward feedback: an image-side reward evaluates the generated image, while a text-side reward evaluates the agent’s final program. Specifically, Rimg follows the KScore-style image judge over faithfulness, visual correctness, text accuracy, and aesthetics. Different from a generic fluency or prompt-quality score, our Rtext is designed as a program sufficiency reward: it checks whether z = (g,R) contains enough grounded facts, ordinal reference bindings, activated generation knowledge, and executable generation constraints for a strong generator to reproduce the intended image. The final reward is

R = (1 − α)Rimg + αRtext. (9)

For each request, GenEvolve samples K trajectories and applies standard group-relative policy optimization over the mixed reward:

Ri − R¯ σR + ϵadv

. (10)

Ai =

We optimize the clipped GRPO surrogate over assistant-side trajectory tokens, following prior GRPObased policy optimization [35, 14, 1]. This reward-driven term identifies which visual trajectories are better, while the visual experience distillation term in Section 5.5 provides denser token-level guidance about why the better trajectory should be preferred. The full GRPO objective is given in Appendix B.6, and the concrete training hyper-parameters (clip ratios, KL coefficient, batch sizes, etc.) are summarised in Table 10.

#### 5.4 Visual Experience Extraction

Scalar rewards indicate which trajectories are better, but not why they are better. GenEvolve therefore converts generated outcomes into structured visual experience. For the K trajectories sampled for the same request, we identify the best and worst trajectories:

R(τi), τ− = arg min

R(τi), ∆ = R(τ+) − R(τ−). (11)

τ+ = arg max

i

i

- Table 1: Main comparison on GenEvolve-Bench. KScore follows the Gen-Searcher [14] visual judge. Know.-Anch. and Qual.-Anch. denotes the Knowledge-Anchored and Quality-Anchored tracks. Best results are in bold; underlined values mark the best open-generator agent result.

Judge dimensions Benchmark overall Method Generator

Faith. Vis. Text Aesth. KScore (All) Know.-Anch. Qual.-Anch. Direct generator baselines raw

Lumina-Image 2.0 [29] Lumina-Image 2.0 0.1044 0.0000 0.3308 0.2694 0.1697 0.1528 0.1915 BAGEL [11] BAGEL 0.1212 0.0059 0.3721 0.4082 0.2041 0.1684 0.2504 SD-3.5-Large [36] SD-3.5 0.1456 0.0135 0.3872 0.4865 0.2235 0.1943 0.2612 FLUX.1-dev [22] FLUX.1 0.1574 0.0059 0.4150 0.5556 0.2396 0.2097 0.2784 FLUX.2 Klein 4B [4] FLUX.2 0.2525 0.0059 0.3847 0.5648 0.2380 0.2004 0.2865 Z-Image-Turbo [5] Z-Image 0.2837 0.0396 0.4369 0.6187 0.2808 0.2340 0.3413 FLUX.2 Klein 9B [4] FLUX.2 0.3662 0.0210 0.4192 0.6599 0.2787 0.2327 0.3382 Z-Image [5] Z-Image 0.3333 0.0278 0.4352 0.5429 0.2728 0.2203 0.3407 Qwen-Image [46] Qwen-Image 0.3729 0.0623 0.4226 0.6751 0.2987 0.2384 0.3768 Nano Banana Pro [16] Nano Banana Pro 0.7761 0.2837 0.6178 0.9158 0.5298 0.5160 0.5477

Agentic image-generation workflows agent

Gen-Searcher 8B [14] Qwen-Image-Edit-2511 0.5284 0.1050 0.4768 0.6377 0.3493 0.3293 0.3745 Gen-Searcher 8B [14] Nano Banana Pro 0.7465 0.3378 0.6198 0.9036 0.5481 0.5472 0.5492 GenEvolve (open generator) Qwen-Image-Edit-2511 0.5303 0.1338 0.4907 0.6347 0.3663 0.3410 0.3990 GenEvolve (strong generator) Nano Banana Pro 0.7970 0.3832 0.6218 0.9222 0.5739 0.5669 0.5830

If ∆ ≥ δmin, the best-worst pair is summarized into five experience slots:

M = {Msearch,Mknow,Mref,Mprompt,Mfail}. (12)

These slots capture the search strategy, internal-knowledge activation, reference selection, promptreference construction, and failure-avoidance lesson that distinguish the stronger trajectory from the weaker one. They are extracted by a VLM judge from the complete tool trajectories, final programs, judge rationales and corresponding diagnostics. Pairs caused only by protocol failures, such as missing references, are discarded because they do not provide reusable visual strategy.

The resulting experience buffer is prompt-keyed and used only during training. Each entry is attached to the source prompt that produced it and stores the corresponding prompt embedding, computed with Qwen3-Embedding-0.6B. For a new prompt x, the teacher retrieves experience by source-prompt similarity rather than by matching the compact experience text:

cos(e(x),e(xj)), Mx = {Ms(xj) : s ∈ S,xj = x˜}. (13)

x˜ = arg max

xj∈B

This source-prompt bundle retrieval returns a coherent set of lessons from one historical case and avoids mixing unrelated slot entries. The detailed slot schema, memory update rule, and fallback strategy are provided in Appendix B.5.

#### 5.5 Visual Experience Distillation

GenEvolve uses visual experience as training-only privileged context. The student receives the normal inference context c(x), while the teacher view receives the same context patched with retrieved experience:

cE(x) = Patch(c(x),Mx). (14) The teacher does not generate a separate trajectory; it re-scores the same student-sampled tokens under cE(x), so inference remains unchanged while training obtains dense token-level guidance. Since student and teacher score the same tokens under different contexts, we draw inspiration from Skill-SD and KL-estimator analyses [43, 33, 37, 24] and use an importance-weighted sampled-token reverse-KL objective for stable cross-context distillation.

For token yi,t, define the student probability and the detached experience-conditioned teacher probability as

pi,t = πθS(yi,t | hi,t), qi,t = sg πθE(yi,t | h˜i,t) , (15) and let ℓi,t = log pi,t − log qi,t. The sampled-token reverse-KL estimator is

k3(ℓi,t) = exp(−ℓi,t) − 1 + ℓi,t. (16)

|Create an image of the Harley J. Earl Trophy sitting perfectly in the center of a wet, reflective winner's podium. Ensure the specific silver shape and wooden base of the real trophy are accurately represented and interact realistically with the lighting. Directly behind the podium, place a large blackboard. On the blackboard, draw a highly detailed,<br><br>creative chalk-style blueprint of the<br><br>Daytona International Speedway track layout.|
|---|

|[Figure 47]|
|---|

|[Figure 48]|
|---|

|[Figure 49]|
|---|

|[Figure 50]|
|---|

|[Figure 51]|
|---|

|[Figure 52]|
|---|

|[Figure 53]|
|---|

|Draw the Svalbard Global Seed Vault in the style of a Studio Ghibli animated film. Make sure the distinctive wedge-shaped entrance jutting out of the snow is clearly recognizable, but rendered in that soft, painted anime background aesthetic with dramatic lighting.|
|---|

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
|---|

|Design a vintage 1970s screen-printed poster honoring Junko Tabei. In the center, illustrate her wearing the exact climbing gear she wore during her famous Everest<br><br>summit. Behind her, draw exactly crossed ice axes arranged in a symmetric form. At<br><br>the very bottom of the poster, include a<br><br>large banner that says 'FIRST WOMAN ON EVEREST' followed by the exact year she achieved this feat.|
|---|

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

|[Figure 66]|
|---|

|[Figure 67]|
|---|

|Prompt|
|---|

|Qwen-Image|
|---|

|Z-Image|
|---|

|Nano Pro|
|---|

|GenSearch (Qwen)|
|---|

|GenEvolve (Qwen-Edit)|
|---|

|GenEvolve (Nano Pro)|
|---|

|Ground-truth|
|---|

- Figure 4: Visual comparison on representative GenEvolve-Bench cases. Orange marks external or uncommon knowledge requirements, while blue marks internal generation-knowledge requirements; GenEvolve substantially improves both Qwen-based and Nano Banana Pro generation frameworks.

Because tokens are sampled by the old student policy under the plain context, the SDL term uses the on-policy importance ratio

ρoni,t = min

The experience-conditioned SDL loss is

πθS(yi,t | hi,t) sg[πθS

,ρmax . (17)

(yi,t | hi,t)]

old

1 i,t mEi,t i,t

mEi,t min ρoni,tk3(ℓi,t),ctok , (18)

LSDL =

where mEi,t selects assistant tokens with non-empty retrieved experience. The final objective is

LGenEvolve = LGRPO + λSDLLSDL. (19) Unlike OPSD-style methods [51, 49] that rely on privileged answers or reasoning traces, GenEvolve distills visual experience from best-worst generated trajectories to improve tool use, knowledge activation, reference selection, and prompt-reference synthesis. Together with GRPO, this closes the self-evolving loop: the updated student produces stronger trajectories, yielding better visual experience for later updates. In practice we further restrict mEi,t to the agent’s crucial tokens, within each sequence, keep only the top 10% ranked by |log πθE − log πθS|; this concentrates the teacher signal on the few choices where the experience-conditioned policy disagrees most with the student. Full SDL hyper-parameters and a token-level case study are provided in Appendix B.7 (Figure 10, Table 10).

### 6 Experiments

We evaluate GenEvolve on GenEvolve-Bench using the same held-out prompts and judge rubrics. The main comparison covers direct image generators, recent agent frameworks, and GenEvolve paired open and strong downstream generators.

Setup. The agent backbone is Qwen3-VL-8B-Instruct [1]. We first cold-start it with SFT on the supervised split of GenEvolve-Data using LLaMA-Factory [52], and then perform on-policy self-evolution on the self-evolution split (full SFT and GRPO+SDL hyper-parameters are listed in Appendix Tables 9 and 10). For each prompt, the agent samples six trajectories, each producing a prompt-reference program z = (g,R). We use Qwen-Image-Edit as the default open-source generator and Nano Banana Pro as a representative strong proprietary generator, allowing us to evaluate both open-generator performance and transfer to stronger closed-source generation. Generated images are scored by a KScore-style visual judge, and the final programs are scored by a program-sufficiency text judge. The mixed reward drives GRPO, while best-worst trajectory pairs provide visual experience for distillation.

All methods are evaluated on the held-out GenEvolve-Bench prompts with the same judge rubric. We report KScore and its four dimensions: faithfulness, visual correctness, text accuracy, and aesthetics.

- Table 2: External evaluation on WISE [26]. Scores are WiScore by category; higher is better. Baselines include direct generators and agentic image-generation systems. GenEvolve is evaluated with the same rollout-and-generate pipeline used for the open-generator setting.

Model Cultural Time Space Biology Physics Chemistry Overall Direct generator baselines raw

Emu3 [44] 0.34 0.45 0.48 0.41 0.45 0.27 0.39 FLUX.1-schnell [3] 0.39 0.44 0.50 0.31 0.44 0.26 0.40 SD-3-Medium [13] 0.42 0.44 0.48 0.39 0.47 0.29 0.42 SD-3.5-Medium [36] 0.43 0.50 0.52 0.41 0.53 0.33 0.45 SD-3.5-Large [36] 0.44 0.50 0.58 0.44 0.52 0.31 0.46 FLUX.1-dev [22] 0.48 0.58 0.62 0.42 0.51 0.35 0.50 Hunyuan-Image 3.0 [6] 0.58 0.57 0.70 0.56 0.63 0.31 0.57 UniWorld-V2 [23] 0.60 0.61 0.70 0.53 0.64 0.32 0.58 Qwen-Image [46] 0.62 0.63 0.77 0.57 0.75 0.40 0.62 NextFlow-RL [50] 0.63 0.63 0.77 0.58 0.67 0.39 0.62 LongCat-Image [25] 0.66 0.61 0.72 0.66 0.72 0.49 0.65 DeepGen1.0 [42] 0.72 0.81 0.70 0.67 0.82 0.66 0.73 GPT-4o [27] 0.81 0.71 0.89 0.83 0.79 0.74 0.80

Agentic image-generation workflows agent

GenAgent [20] 0.78 0.67 0.78 0.72 0.77 0.55 0.72 Gen-Searcher-8B + Qwen-Image [14] 0.80 0.71 0.82 0.76 0.74 0.75 0.77 Mind-Brush [17] 0.83 0.69 0.84 0.71 0.85 0.68 0.78 GenEvolve + Qwen-Image-Edit 0.84 0.74 0.87 0.83 0.81 0.83 0.82

Full rubric weights, judge backbone, and calibration settings are listed in Appendix D.2 and Table 11. To assess transfer beyond our constructed benchmark, we additionally evaluate GenEvolve on the publicly released WISE benchmark [26], a commonly used knowledge-intensive image-generation benchmark; results are presented in Section 6 and protocol details are provided in Appendix D.3.

Main comparisons. Table 1 reports the main comparison on GenEvolve-Bench. Since the benchmark includes both Knowledge-Anchored and Quality-Anchored prompts, strong performance requires factual grounding, reference use, and quality-sensitive prompt-reference construction, not only generic image quality. Direct generators remain limited: Qwen-Image reaches only 0.2987 KScore, while the stronger Nano Banana Pro reaches 0.5298 but still benefits from agent-side orchestration on demanding grounded cases. Under the same open-generator setting, GenEvolve improves over Gen-Searcher, a recent trained search-grounded agentic baseline, raising KScore from 0.3493 to 0.3663 and visual correctness from 0.1050 to 0.1338. Although Qwen-Image-Edit is not a particularly strong renderer, this gain shows the benefit of better agent-side orchestration, including external search, internal knowledge activation, reference selection, and prompt-reference synthesis. With the stronger Nano Banana Pro, the learned agent is further unlocked: GenEvolve reaches the best overall KScore of 0.5739 and improves most dimensions over both raw Nano Banana Pro and Gen-Searcher with the same generator. This suggests generator-transferable orchestration rather than overfitting to one renderer. Figure 4 provides representative visual comparisons.

External Generalization on WISE. To verify that the policy learned by GenEvolve generalizes beyond our constructed benchmark, we evaluate it on the publicly released WISE benchmark [26], which targets knowledge-intensive image generation across six categories: cultural, time, space, biology, physics, and chemistry. Following the standard protocol, the agent receives only the WISE prompt, produces a prompt-reference program through the normal rollout interface, and the output image is generated by Qwen-Image-Edit. Generated images are scored under the WISE threedimension protocol with GPT-4o-2024-05-13 as the judge, and results are aggregated by the official WiScore script.

- Table 2 compares GenEvolve with both direct generators and recent agentic image-generation systems. GenEvolve attains the best overall WiScore of 0.82, surpassing the strongest direct baseline (GPT-4o,

- Table 3: Component ablation on GenEvolve-Bench. Except for the generator-only row, all variants use Qwen-Image-Edit as the downstream generator.

Variant KScore (All) Faith. Vis. Text Aesth. Know.-Anch. Qual.-Anch.

raw Qwen-Image only 0.2987 0.3729 0.0623 0.4226 0.6751 0.2384 0.3768 base Untuned Qwen3-VL workflow 0.3317 0.4781 0.1061 0.4571 0.5867 0.3109 0.3587 SFT only 0.3480 0.4791 0.1121 0.4934 0.5785 0.3303 0.3709 SFT + GRPO w/o visual experience 0.3548 0.5027 0.1138 0.4926 0.6197 0.3277 0.3898 full GenEvolve 0.3663 0.5303 0.1338 0.4907 0.6347 0.3410 0.3990

0.80) and all agentic baselines (e.g., GenAgent 0.72, Gen-Searcher-8B 0.77, Mind-Brush 0.78). The improvement is most pronounced on chemistry (0.83) and biology (0.83), where factual grounding through tool-orchestrated trajectories is most beneficial. Figure 1 (b) further visualizes these categorywise gains. These results show that the tool-orchestrated visual trajectory and Visual Experience Distillation transfer to an out-of-distribution knowledge-intensive benchmark, rather than overfitting to GenEvolve-Bench prompts or judges.

Ablation study. Table 3 studies the contribution of each training stage. Direct Qwen-Image reaches 0.2987 KScore, and an untuned Qwen3-VL workflow improves to 0.3317 by using the same tool interface and reference-conditioned generation. SFT cold start further raises KScore to 0.3480, showing that curated trajectories teach more reliable tool use and prompt-reference construction. Adding GRPO without visual experience improves KScore to 0.3548, but scalar rewards alone provide limited credit assignment for long tool trajectories. The full GenEvolve reaches 0.3663, with the highest visual correctness of 0.1338 and the best scores on both Knowledge-Anchored and Quality-Anchored tracks. This shows that Visual Experience Distillation provides complementary token-level guidance beyond SFT and scalar-reward GRPO.

### 7 Conclusion

We present GenEvolve, a self-evolving framework for image-generation agents via Tool-Orchestrated Visual Experience Distillation. Instead of a single prompt-to-image call, GenEvolve formulates generation as a visual trajectory where an agent coordinates external evidence, visual references, internal generation knowledge, prompt-reference construction, generation, and feedback. By comparing best-worst trajectories, GenEvolve extracts structured visual experience and distills this teacher-only guidance into the student policy. Experiments on GenEvolve-Bench show improved agentic behavior and final image quality, demonstrating its value compared with traditional generation.

### References

- [1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.
- [2] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.
- [3] Black Forest Labs. FLUX.1 [schnell]. Hugging Face model card, 2024. URL https:// huggingface.co/black-forest-labs/FLUX.1-schnell. Accessed: 2026-05-20.
- [4] Black Forest Labs. FLUX.2 [klein]. https://huggingface.co/black-forest-labs/ FLUX.2-klein-4B, 2026. FLUX.2 [klein] model family; compact image generation and editing models. Accessed: 2026-05-07.
- [5] Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Steven Hoi, Zhaohui Hou, Shijie Huang, Dengyang Jiang, Xin Jin, Liangchen Li, et al. Z-image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699, 2025.
- [6] Siyu Cao, Hangting Chen, Peng Chen, Yiji Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kipper Gong, Tianpeng Gu, Xiusen Gu, et al. Hunyuanimage 3.0 technical report. arXiv preprint arXiv:2509.23951, 2025.
- [7] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025.
- [8] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.
- [9] SiXiang Chen, Jianyu Lai, Jialin Gao, Tian Ye, Haoyu Chen, Hengyu Shi, Shitong Shao, Yunlong Lin, Song Fei, Zhaohu Xing, et al. Postercraft: Rethinking high-quality aesthetic poster generation in a unified framework. arXiv preprint arXiv:2506.10741, 2025.
- [10] Sixiang Chen, Jianyu Lai, Jialin Gao, Hengyu Shi, Zhongying Liu, Tian Ye, Junfeng Luo, Xiaoming Wei, and Lei Zhu. Posteromni: Generalized artistic poster creation via task distillation and unified reward feedback. arXiv preprint arXiv:2602.12127, 2026.
- [11] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.
- [12] Ken Ding. Hdpo: Hybrid distillation policy optimization via privileged self-distillation. arXiv preprint arXiv:2603.23871, 2026.
- [13] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206, 2024.
- [14] Kaituo Feng, Manyuan Zhang, Shuang Chen, Yunlong Lin, Kaixuan Fan, Yilei Jiang, Hongyu Li, Dian Zheng, Chenyang Wang, and Xiangyu Yue. Gen-searcher: Reinforcing agentic search for image generation. arXiv preprint arXiv:2603.28767, 2026.
- [15] Google AI for Developers. Gemini api release notes: Gemini 3 pro preview. https:// ai.google.dev/gemini-api/docs/changelog, November 2025. Official release note for gemini-3-pro-preview. Accessed: 2026-05-07.
- [16] Google DeepMind. Introducing nano banana pro. https://blog.google/technology/ai/ nano-banana-pro/, November 2025. Google DeepMind product release for the Nano Banana Pro image generation and editing model built on Gemini 3 Pro. Accessed: 2026-05-06.

- [17] Jun He, Junyan Ye, Zilong Huang, Dongzhi Jiang, Chenjue Zhang, Leqi Zhu, Renrui Zhang, Xiang Zhang, and Weijia Li. Mind-brush: Integrating agentic cognitive search and reasoning into image generation. arXiv preprint arXiv:2602.01756, 2026.
- [18] Zefeng He, Siyuan Huang, Xiaoye Qu, Yafu Li, Tong Zhu, Yu Cheng, and Yang Yang. Gems: Agent-native multimodal generation with memory and skills. arXiv preprint arXiv:2603.28088, 2026.
- [19] Jonas Hübotter, Frederike Lübeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, et al. Reinforcement learning via self-distillation. arXiv preprint arXiv:2601.20802, 2026.
- [20] Kaixun Jiang, Yuzheng Wang, Junjie Zhou, Pandeng Li, Zhihang Liu, Chen-Wei Xie, Zhaoyu Chen, Yun Zheng, and Wenqiang Zhang. Genagent: Scaling text-to-image generation via agentic multimodal reasoning. arXiv preprint arXiv:2601.18543, 2026.
- [21] V Kovalev, A Kuvshinov, A Buzovkin, D Pokidov, and D Timonin. Craft: Continuous reasoning and agentic feedback tuning for multimodal text-to-image generation. arXiv preprint arXiv:2512.20362, 2025.
- [22] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.
- [23] Zongjian Li, Zheyuan Liu, Qihui Zhang, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Yang Ye, Wangbo Yu, Yuwei Niu, and Li Yuan. Uniworld-V2: Reinforce Image Editing with Diffusion Negative-aware Finetuning and MLLM Implicit Feedback. arXiv preprint arXiv:2510.16888,

2025. URL https://arxiv.org/abs/2510.16888.

- [24] Kezhao Liu, Jason Klein Liu, Mingtao Chen, and Yiming Liu. Rethinking kl regularization in rlhf: From value estimation to gradient optimization. arXiv preprint arXiv:2510.01555, 2025.
- [25] Meituan LongCat Team, Hanghang Ma, Haoxian Tan, Jiale Huang, Junqiang Wu, Jun-Yan He, Lishuai Gao, Songlin Xiao, Xiaoming Wei, Xiaoqi Ma, Xunliang Cai, Yayong Guan, and Jie Hu. LongCat-Image Technical Report. arXiv preprint arXiv:2512.07584, 2025. URL https://arxiv.org/abs/2512.07584.
- [26] Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, et al. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025.
- [27] OpenAI. Introducing 4o Image Generation. OpenAI blog, 2025. URL https://openai.com/ index/introducing-4o-image-generation/. Accessed: 2026-05-20.
- [28] William Peebles and Saining Xie. Scalable diffusion models with transformers. arXiv preprint arXiv:2212.09748, 2022.
- [29] Qi Qin, Le Zhuo, Yi Xin, Ruoyi Du, Zhen Li, Bin Fu, Yiting Lu, Jiakang Yuan, Xinyue Li, Dongyang Liu, Xiangyang Zhu, Manyuan Zhang, Will Beddow, Erwann Millon, Victor Perez, Wenhai Wang, Conghui He, Bo Zhang, Xiaohong Liu, Hongsheng Li, Yu Qiao, Chang Xu, and Peng Gao. Lumina-image 2.0: A unified and efficient image generative framework. arXiv preprint arXiv:2503.21758, 2025.
- [30] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.
- [31] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [32] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022.

- [33] John Schulman. Approximating kl divergence. http://joschu.net/blog/kl-approx. html, 2020. Blog post.
- [34] Bytedance Seed. Seed2. 0 model card: Towards intelligence frontier for real-world complexity. Technical report, Technical report, Bytedance, 2025. URL https://lf3-static. bytednsdoc. com ..., 2026.
- [35] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [36] Stability AI. Stable diffusion 3.5 large. https://huggingface.co/stabilityai/ stable-diffusion-3.5-large, 2024. Official model card for Stable Diffusion 3.5 Large. Accessed: 2026-05-07.
- [37] Yunhao Tang and Rémi Munos. On a few pitfalls in kl divergence gradient estimation for rl. arXiv preprint arXiv:2506.09477, 2025.
- [38] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.
- [39] Tencent Hunyuan Team. Hunyuan-dit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding. arXiv preprint arXiv:2405.08748, 2024.
- [40] Yang Tian, Fan Liu, Jingyuan Zhang, Wei Bi, Yupeng Hu, and Liqiang Nie. Open multimodal retrieval-augmented factual image generation. arXiv preprint arXiv:2510.22521, 2025.
- [41] Xingchen Wan, Han Zhou, Ruoxi Sun, Hootan Nakhost, Ke Jiang, Rajarishi Sinha, and Sercan Ö Arık. Maestro: Self-improving text-to-image generation via agent orchestration. arXiv preprint arXiv:2509.10704, 2025.
- [42] Dianyi Wang, Ruihang Li, Feng Han, Chaofan Ma, Wei Song, Siyuan Wang, Yibin Wang, Yi Xin, Hongjian Liu, Zhixiong Zhang, Shengyuan Ding, Tianhang Wang, Zhenglin Cheng, Tao Lin, Cheng Jin, Kaicheng Yu, Jingjing Chen, Wenjie Wang, Zhongyu Wei, and Jiaqi Wang. DeepGen 1.0: A Lightweight Unified Multimodal Model for Advancing Image Generation and Editing. arXiv preprint arXiv:2602.12205, 2026. URL https://arxiv.org/abs/2602.12205.
- [43] Hao Wang, Guozhi Wang, Han Xiao, Yufeng Zhou, Yue Pan, Jichao Wang, Ke Xu, Yafei Wen, Xiaohu Ruan, Xiaoxin Chen, et al. Skill-sd: Skill-conditioned self-distillation for multi-turn llm agents. arXiv preprint arXiv:2604.10674, 2026.
- [44] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, Yingli Zhao, Yulong Ao, Xuebin Min, Tao Li, Boya Wu, Bo Zhao, Bowen Zhang, Liangdong Wang, Guang Liu, Zheqi He, Xi Yang, Jingjing Liu, Yonghua Lin, Tiejun Huang, and Zhongyuan Wang. Emu3: Next-token prediction is all you need. CoRR, abs/2409.18869, 2024. doi: 10.48550/ARXIV.2409.18869. URL https://doi.org/10.48550/arXiv.2409.18869.
- [45] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.
- [46] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.
- [47] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025.
- [48] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.

- [49] Tianzhu Ye, Li Dong, Xun Wu, Shaohan Huang, and Furu Wei. On-policy context distillation for language models. arXiv preprint arXiv:2602.12275, 2026.
- [50] Huichao Zhang, Liao Qu, Yiheng Liu, Hang Chen, Yangyang Song, Yongsheng Dong, Shikun Sun, Xian Li, Xu Wang, Yi Jiang, Hu Ye, Bo Chen, Yiming Gao, Peng Liu, Akide Liu, Zhipeng Yang, Qili Deng, Linjie Xing, Jiyang Liu, Zhao Wang, Yang Zhou, Mingcong Liu, Yi Zhang, Qian He, Xiwei Hu, Zhongqi Qi, Jie Shao, Zhiye Fu, Shuai Wang, Fangmin Chen, Xuezhi Chai, Zhihua Wu, Yitong Wang, Zehuan Yuan, Daniel K. Du, and Xinglong Wu. NextFlow: Unified Sequential Modeling Activates Multimodal Understanding and Generation. arXiv preprint arXiv:2601.02204, 2026. URL https://arxiv.org/abs/2601.02204.
- [51] Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. arXiv preprint arXiv:2601.18734, 2026.
- [52] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, and Zheyan Luo. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd annual meeting of the association for computational linguistics (volume 3: system demonstrations), pages 400–410, 2024.

### A GenEvolve-Data Construction

#### A.1 Prompt Pool Recipes

GenEvolve-Data is constructed to support three stages of our framework: supervised cold start, selfevolution, and held-out evaluation. Unlike ordinary prompt-rewriting data, each example is designed as a complete visual generation problem in which an agent must acquire missing evidence, select visual references, activate generation knowledge when needed, and synthesize a prompt-reference program. We therefore begin from recipe-controlled prompt generation rather than unconstrained LLM sampling. Each recipe specifies the task family, missing factual information, expected visual anchor, dominant visible requirement, optional secondary constraints, and difficulty. These fields are used for coverage control and auditing, but are not exposed to the agent as task labels.

Table 4: Prompt recipe fields used to construct the prompt pool.

Field Role Examples Task family Controls the high-level generation sce-

factual scene, text-critical generation, multi-object layout, body/anatomy, attribute binding, material, creative transfer, mixed hard case

nario

Factual gap Specifies information that may require search

entity names, official labels, dates, variants, visual text, event-specific details

Visual anchor Specifies appearance cues that benefit from image search

entity shape, facade, costume, product geometry, pose, material, style

Dominant requirement Defines the main visible quality challenge

text rendering, layout, counting, material, anatomy, aesthetics

Secondary requirements

Adds realistic multi-constraint composition

additional object relations, style constraints, reference binding, readable text

Difficulty Controls expected trajectory depth medium or hard; rare easy prompts are retained only after validation

The current pool contains 19,990 valid prompts after deduplication: 11,999 Knowledge-Anchored prompts and 7,991 Quality-Anchored prompts. The average prompt length is about 65 words, with 13,333 hard prompts, 6,654 medium prompts, and 3 easy prompts. Table 5 summarizes the concrete configuration used by the current implementation.

Table 5: Concrete GenEvolve-Data configuration.

Component Value Notes Prompt pool 19,990 valid prompts Constructed from 19,991 pre-dedup valid candidates; 9

invalid candidates removed by length filtering. Prompt tracks 11,999 / 7,991 Knowledge-Anchored / Quality-Anchored prompts.

Categories 16 categories Architecture, city streets, public figures, products, vehicles, events, science, artifacts, text/layout, spatial, counting, anatomy, attributes, material, aesthetics, creative transfer.

Search labels 19,990 image search; 13,972 text search

All prompts require image search. Text search includes all Knowledge-Anchored prompts and 1,973 QualityAnchored prompts.

Dominant quality tags 8 visible requirement types Material 3,000; creative transfer 3,000; text 3,000; anatomy 3,000; attribute binding 2,999; counting 2,995; spatial layout 1,000; aesthetics 996.

Difficulty 13,333 hard; 6,654 medium; 3 easy

Difficulty controls expected search/reference/knowledgeactivation depth.

Self-evolution/eval format {id, question,

Teacher trajectory, selected references, and teacher prompt are removed from the rollout input.

gt_image}

Evaluation metadata track, category, difficulty, search flags

Also includes expected reference targets and qualityrequirement tags for analysis.

#### A.2 Teacher Trajectory Generation

Each validated prompt is converted into a teacher trajectory through a real multi-turn tool loop. At each turn, the teacher emits reasoning and either one tool call or a final answer. The system executes the tool, appends the observation, and continues until a valid prompt-reference program is produced or the trajectory times out. We use strong multimodal teacher models, including Seed2.0 and Gemini

###### 3 Pro, to generate these trajectories because they provide strong reasoning, reference understanding, and tool-use capabilities [34, 15].

Accepted trajectories must contain meaningful tool use rather than only a final prompt. Image search is required because the downstream generator is reference-conditioned. Generation-knowledge calls are encouraged when the prompt contains visible quality challenges, but the system does not force every trajectory to query all knowledge types. This preserves realistic tool-order diversity: some trajectories begin with factual lookup, some start from reference search, and others activate generation knowledge only after inspecting the request and retrieved evidence.

#### A.3 Filtering Rubric

Filtering combines hard programmatic checks and VLM-based judgment. Hard checks reject incomplete trajectories, missing image search, invalid reference counts, invalid local image paths, unparseable JSON, invalid skill names, missing ordinal reference wording, meaningless reasoning, unsafe content, and raw URL leakage in the final prompt. These rules remove format and protocol failures before semantic review.

The VLM filter then scores six dimensions: prompt suitability, reference grounding, trajectory process quality, skill integration, final prompt faithfulness, and supervised training value. A trajectory is kept only if the final program remains faithful to the user request, the selected references support the claimed visual details, and the collected evidence is actually used. In the full run, 13,379 of 19,320 structurally valid trajectories were kept (69.2%). Average skill integration was 4.70/5, while reference grounding was the hardest dimension at 3.98/5. Common failures include hallucinated reference content, contradicted image details, duplicate references, unused tool results, and final prompts that copy the user request without meaningful synthesis.

#### A.4 GT Image Generation and Filtering

For self-evolution and evaluation, we render GT images from high-quality teacher prompt-reference programs and their selected references. We use Nano Banana Pro as the GT image generator because of its strong instruction following, reference-conditioned editing, text rendering, and real-world visual knowledge [16]. These GT images are not unique ground-truth answers to the raw user prompts; instead, they provide strong visual realizations that make image-level feedback and evaluation meaningful.

We generated 4,321 successful GT images from 4,379 attempts (98.7%) and retained 3,175 after filtering (73.5%). The image filter checks generation-prompt compliance, reference utilization, visual coherence, and image quality. Images that ignore selected references, fail required text, contradict grounded details, or exhibit severe visual artifacts are discarded.

#### A.5 Supervised and Self-Evolution Export

GenEvolve-Data is exported into two complementary views. The supervised view preserves the full tool-loop conversation and all images shown to the model, including candidate references, so that the student learns evidence acquisition, candidate comparison, reference selection, internal-knowledge activation, and prompt-reference program construction. During supervised training, only assistantside tokens are optimized; user prompts and tool observations serve as context and are masked from the loss. The supervised split contains 8,800 training examples and 200 evaluation examples.

The self-evolution view removes the teacher trajectory and teacher final program. Each example contains the raw prompt, GT image path, and metadata needed for reward evaluation. This prevents the agent from copying teacher actions and forces it to produce its own tool-orchestrated rollout. The current split uses 3,175 filtered GT image cases: a 2,575-case self-evolution training pool and a about

600-case evaluation pool. The training pool is further divided into 2,446 optimization cases and 129 internal validation cases.

#### A.6 Coverage and Construction Statistics

- Figure 5 visualizes the two-track category hierarchy, and Figure 6 summarizes the major filtering and split statistics. Together, these figures show that GenEvolve-Data covers both externally grounded generation and quality-sensitive generation requirements, while maintaining a held-out benchmark split with no exact overlap with the self-evolution training pool.

Architecture

TransferCreative

LightingAesthetic

CityStreets

Materials

Attributes

Events/ Sports

QualityAnchored

Anatomy/ Pose

###### GenEvolve Data

Artifacts

Counting

KnowledgeAnchored

Spatial Comp.

Products

TextLayout

FiguresPublic Science

Vehicles

- Figure 5: Category hierarchy of GenEvolve-Data. The prompt pool is organized into KnowledgeAnchored and Quality-Anchored tracks, each covering eight diagnostic categories used for coverage control, split stratification, and benchmark analysis.

0 5000 10000 15000 20000

Number of cases

Prompt pool Valid trajectories

Filtered trajectories SFT trajectories

from recipe-controlled requests to supervised agent traces

19,990 19,320

13,379 8,800

(a) Tool-trajectory data

| | |
|---|---|
| | |

0 1000 2000 3000 4000 5000

Number of cases

GT generation attempts Successful generation

Filtered images Self-evolution images

Held-out benchmark

from generated GT images to self-evolution and evaluation splits

4,379 4,321

3,175 2,575

594

(b) Visual-feedback cases

GenEvolve-Data Construction Statistics

- Figure 6: GenEvolve-Data construction statistics. The left panel summarizes prompt-to-trajectory filtering for supervised learning, and the right panel summarizes GT image generation, image filtering, self-evolution images, and held-out benchmark cases.

### B Additional Method Details

This section provides implementation details for the rollout protocol, prompt-reference program schema, experience memory, retrieval, GRPO loss, and experience-conditioned self-distillation. These details complement the main method while keeping the core paper concise.

#### B.1 Tool-Orchestrated Rollout Protocol

At inference time, GenEvolve exposes a small and auditable action space. Each assistant turn must either emit one tool call or terminate with a parseable final answer. The environment executes the requested tool, appends the observation, and resumes the agent. The current runtime uses the tools in Table 6.

Table 6: Runtime tools used by the image-generation agent.

Tool Input Observation Main role search Textual query Snippets for names, dates, labels, event

External factual evidence

facts, official terms, and other visible facts

image_search Visual query Candidate images with IDs, titles, source URLs, cached paths, and previews

Visual reference retrieval

query_knowledge One enum skill name Static generation guidance for the re-

Internal generationknowledge activation

quested knowledge type

The agent is not forced to call tools in a fixed order. Knowledge-Anchored prompts often use search before image_search; Quality-Anchored prompts may skip text search when the visual anchor is sufficient; and complex prompts may call several generation-knowledge tools. The final answer must select one or two reference images and synthesize a prompt-reference program. Invalid tool names, invalid knowledge names, missing references, raw URL leakage, and unparseable final JSON are treated as trajectory failures during data construction and evaluation.

Table 7: Callable internal generation-knowledge modules. We instantiate them using the common “skill” interface, but conceptually they serve as on-demand generation knowledge for visible failure modes.

Module Visible failure mode Module Visible failure mode text text rendering unreadable, misplaced, or mis-

layout spatial layout wrong positions, depth, scale, or occlusion

spelled text

body anatomy/body coherence

style aesthetic drawing weak lighting, camera, composition, or atmosphere bind attribute binding attributes leak between objects create creative drawing weak style transfer or concept fu-

malformed hands, pose, face, or body structure

sion

physics physical/material consistency

impossible support, shadow, reflection, or material behavior

count quantity counting wrong number of instances or uncountable layout

#### B.2 Prompt-Reference Program Schema

The final executable object is a prompt-reference generation program z = (g,R). The instruction g is natural language consumed by a reference-conditioned generator, while R is an ordered list of local reference images. The instruction must refer to selected references by ordinal phrases such as “the first reference image”, rather than raw image IDs or URLs. This makes the program independent of transient retrieval IDs and aligns it with generator APIs that receive images as ordered inputs.

{

}

"gen_prompt": "... the first reference image ...", "reference_images": [

{"img_id": "IMG_001", "role": "identity/shape reference"}, {"img_id": "IMG_004", "role": "style or layout reference"}

], "selected_skills": ["spatial_layout", "text_rendering"], "rationale": "Short explanation of evidence, references, and knowledge."

The selected skills are not passed to the generator as separate hidden controls. They record which internal generation knowledge was queried and should be reflected in g. During training, this field supports tool-call supervision, diagnostic analysis, and visual-experience slot construction.

#### B.3 Visual Experience Extraction — Real Training Cases

To make the experience extraction process concrete, we present three complete best-worst trajectory comparisons extracted during self-evolution training. For each case, we show the original user request, the full tool-call sequence and thinking excerpts from both the best and worst trajectories, the generated images, and the five experience slots distilled from the comparison. All content is taken verbatim from the training logs; only tool observations are abbreviated for space.

These cases illustrate three distinct failure modes that Visual Experience Distillation is designed to address. Case 1 demonstrates how a subtle search query mistake — including an attribute keyword that introduces noise — propagates into a completely wrong factual grounding and ruins the entire generation. Case 2 shows that retrieving correct facts is necessary but insufficient: skipping generation-knowledge skills causes the agent to produce a syntactically valid but visually broken prompt, because text layout and typographic structure require explicit skill guidance that cannot be replaced by a style reference image. Case 3 shows how omitting one critical skill (spatial_layout) while correctly calling others leads to cascading layout and attribute failures — when two distinct objects must be spatially separated, vague positioning descriptions cause the generator to merge or overlap them.

Together, these cases demonstrate that the experience extraction mechanism captures actionable lessons across the full trajectory decision space: search query formulation (Case 1), skill routing and activation (Cases 2–3), reference selection (all cases), prompt-reference program synthesis (all cases), and failure avoidance patterns (all cases). The extracted experience slots are then injected only into the privileged teacher branch during SDL, providing dense token-level guidance that complements the scalar reward signal from GRPO.

###### User Request

“Draw a futuristic, cyberpunk version of the Crucible Theatre’s main stage setup used for the World Snooker Championship. The iconic, intimate circular seating layout should remain recognizable. The snooker table should be centered, constructed from polished dark mahogany and thick, glowing transparent glass. The table felt must match the national flag stripes of the 2023 World Snooker Champion. Above the table, add a translucent holographic scoreboard displaying ‘147’ in bright cyan digits.”

###### Best Trajectory R

###### Turn 1 — Think & Tool Call: search("2023 World Snooker Championship winner nationality")

“...First, I need to find out who won the 2023 World Snooker Championship, right? Because the table felt has to match their national flag’s stripes. So first, use the search tool for that query...”

→ Observation: Search returns Luca Brecel from Belgium.

###### Turn 2 — Think & Tool Call: image_search("Crucible Theatre Sheffield World Snooker Championship main stage setup circular seating")

“...the 2023 winner is Luca Brecel from Belgium, so his national flag is the Belgian flag, which has vertical black, yellow, red stripes. Got that part. Now I need a visual reference of the Crucible Theatre main stage setup...”

→ Observation: Returns images of the iconic circular seating layout. Selects IMG_001.

- Turn 3 — Think & Tool Call: query_knowledge("creative_drawing")

- “...IMG_001 looks like it shows the actual stage setup with the circular seating, that’s perfect for reference. Now, what skills do I need? 1. Creative drawing: because we need a cyberpunk futuristic version of the real setup, keeping the circular seating recognizable...”

- Turn 4 — Think & Tool Call: query_knowledge("spatial_layout")

“...I need to make sure the elements are positioned correctly: circular seating in the background, snooker table centered in the midground, holographic scoreboard above the table...”

- Turn 5 — Think & Tool Call: query_knowledge("text_rendering")

“...the holographic scoreboard has to show ‘147’ in bright cyan digits, so I need to make sure the text is rendered correctly...”

###### Turn 6 — Think & Final Answer:

“...reference the first reference image for the iconic circular seating layout. Then, cyberpunk futuristic style, midground has the centered snooker table: polished dark mahogany frame, thick glowing transparent glass sides, felt matching the vertical black-yellow-red stripes of the Belgian national flag...Above the table, centered in the upper third of the frame, a translucent holographic scoreboard displaying ‘147’ in bold cyan digits...”

Final program (gen_prompt): “Futuristic cyberpunk version of the Crucible Theatre main stage setup, using the iconic circular seating layout from the first reference image for recognizability. Depth layering: background = the circular seating from the first reference, reimagined with neon accent lighting in cyberpunk style; midground = centered snooker table, with polished dark mahogany frame, thick glowing transparent glass sides, and table felt matching the vertical black-yellow-red stripes of the Belgian national flag; foreground = subtle neon haze. Above the table...a translucent holographic scoreboard displays the text ‘147’ in bright bold sans-serif cyan glowing digits...”

Reference: IMG_001 — note: “Iconic circular seating layout of the Crucible Theatre main stage, to retain recognizability in the cyberpunk reimagining.”

###### Worst Trajectory R

###### Turn 1 — Think & Tool Call: search("2023 World Snooker Championship winner national flag")

“...first, let’s confirm the 2023 winner. Let’s call search with query ‘2023 World Snooker Championship winner national flag’. Then, after that, do image search for Crucible Theatre main stage...”

→ Observation: Search returns mixed results due to the keyword “flag” causing noise. The agent incorrectly concludes the winner is Wu Yize from China.

###### Turn 2 — Think & Tool Call: image_search("Crucible Theatre Sheffield World Snooker Championship main stage circular seating layout")

“...2023 winner is Wu Yize, Chinese, so Chinese national flag has red background, five yellow stars. Got that. Now I need to get a visual reference of the Crucible Theatre main stage setup...”

- Turn 3 — Think & Tool Call: query_knowledge("creative_drawing")

“...IMG_002 looks like a good visual of the actual venue seating from a fan’s perspective. Now I need to query the relevant skills. First, creative_drawing because we need a cyberpunk futuristic version of the real venue...”

- Turn 4 — Think & Tool Call: query_knowledge("attribute_binding")

“...we have multiple elements with specific distinct attributes: the snooker table has polished dark mahogany construction, thick glowing transparent glass edges, felt matching Chinese flag stripes (red with yellow stars)...attribute binding will help make sure each attribute is correctly assigned...”

- Turn 5 — Think & Tool Call: query_knowledge("text_rendering")

“...there’s a holographic scoreboard displaying ‘147’ in bright cyan digits, so I need to make sure that text is rendered correctly...”

- Turn 6 — Think & Final Answer:

“...reference the first reference image for the Crucible Theatre intimate circular seating layout...cyberpunk futuristic style: neon accents, dark moody lighting...snooker table constructed from polished dark mahogany with thick glowing transparent glass edges, its felt exactly matching the red background and five yellow star stripes of the Chinese national flag...”

Final program (gen_prompt): “Futuristic cyberpunk reinterpretation of the Crucible Theatre main stage...Centered on the stage: a snooker table constructed from polished dark mahogany with thick glowing transparent glass edges, its felt exactly matching the red background and five yellow star stripes of the Chinese national flag. Centered directly above the snooker table: a translucent holographic scoreboard displaying the text ‘147’ in bold bright cyan glowing sans-serif digits...”

Critical failures: (1) The search query included the word “flag”, causing keyword-matching noise that led to an incorrect entity identification (Wu Yize instead of Luca Brecel). (2) The entire downstream trajectory — reference selection, skill integration, prompt construction — was built on a wrong factual premise, producing Belgian → Chinese flag stripes. (3) Despite correct tool use for creative_drawing, attribute_binding, and text_rendering, the text reward is 0.0 because the flag constraint is factually wrong.

###### Extracted Experience Slots (Delta

- S1 Search strategy: When a constraint requires an attribute of a specific entity (e.g., a champion’s flag), search for the entity first (“2023 World Snooker Championship winner”), then deduce the attribute — rather than including the attribute keyword (“flag”) in the search query, which causes keyword-matching noise.
- S2 Knowledge activation: Call spatial_layout to explicitly define the positioning and depth layering of the environment (circular seating) versus the central subject (snooker table), preventing the environment from being omitted.
- S3 Reference selection: Explicitly map the reference image to a specific depth layer (e.g., “background = circular seating from the first reference”) so the layout is preserved in the cyberpunk reimagining.
- S4 Prompt construction: Structure the prompt using explicit depth layering (background, midground, foreground) to separate the environment from the central subject, preventing the model from focusing entirely on the subject and omitting the venue.
- S5 Failure avoidance: Do not include visual attribute keywords (e.g., “flag”) in the initial factual search query — this biases the search engine toward irrelevant articles and can cause incorrect entity resolution.

[Figure 68]

[Figure 69]

Best (R = 1.0): Belgian flag stripes (correct) Worst (R = 0.225): Chinese flag colors (wrong )

- Figure 7: Case 1 generated images. The search query “winner nationality” (best) vs. “winner national flag” (worst) led to completely different factual grounding and flag stripe colors on the snooker table felt.

###### Case 2 — User Request

“Create a retro-futuristic 1970s-style travel poster featuring the French Aérotrain I80. The poster should show the hovertrain gliding on its inverted T-shaped concrete track. In bold vintage typography, include the exact top speed it reached during its world record run, along with the year that record was set. The text should be styled like a classic transport advertisement.”

###### Best Trajectory R

###### Turn 1 — Think & Tool Call: search("Aérotrain I80 world record top speed year set")

“...first, I need to find two key things: the exact top speed of the Aérotrain I80’s world record run and the year it was set, right? Because the user wants that exact text on the poster...”

→ Observation: Returns speed 430.4km/h (267mph) set in 1974.

###### Turn 2 — Think & Tool Call: image_search("Aérotrain I80 on inverted T concrete track")

“...the record speed is 430.4 km/h set in 1974. Good. Now I need a visual reference of the Aérotrain I80 and its inverted T-shaped track to make the poster accurate...”

→ Observation: Returns clear images. Selects IMG_001.

###### Turn 3 — Think & Tool Call: query_knowledge("text_rendering") ← critical skill

“...IMG_001 is a clear shot of the Aérotrain I80 on its track. Now I need to get guidance on text rendering for 1970s vintage transport advertisement typography, right? Because the user wants bold vintage typography with the exact speed and year...”

###### Turn 4 — Think & Final Answer:

“...the gen_prompt should be retro-futuristic 1970s travel poster style...Bold vintage 1970s transport advertisement

typography centered in the lower third of the poster: first line reads ‘AÉROTRAIN I80’ in large bold rounded retro sans-serif font, second line reads ‘430.4km/h • 267mph’, third line reads ‘WORLD SPEED RECORD • 1974’...”

Final program (gen_prompt): “Retro-futuristic 1970s-style travel poster, featuring the Aérotrain I80 gliding on its inverted T-shaped concrete track as shown in the first reference image. Bold vintage 1970s transport advertisement typography centered in the lower third of the poster: first line reads ‘AÉROTRAIN I80’ in large, bold, rounded retro sans-serif font, second line reads ‘430.4km/h • 267mph’ in slightly smaller bold matching font, third line reads

‘WORLD SPEED RECORD • 1974’...Vibrant, saturated 1970s color palette, warm golden hour lighting, clean retro composition...” Skills called: text_rendering ✓ Reference: IMG_001 (Aérotrain on track). Judge: all pass.

###### Worst Trajectory R

Turn 1 — Think & Tool Call: search("Aérotrain I80 official world record top speed year of record run")

“...first I should do a text search for that, because the user wants the exact speed and year for the text...”

→ Observation: Same facts: 430.4km/h, 1974. Turn 2 — Think & Tool Call: image_search("Aérotrain I80 gliding on inverted T shaped concrete track")

“...the world record top speed is 430.4km/h set in 1974. Perfect. Now I need visual references: first, an image of the Aérotrain I80 on its track...” Turn 3 — Think & Tool Call: image_search("1970s retro-futuristic travel poster vintage transport advertisement typography")

“...IMG_001 is perfect, it shows the Aérotrain I80 clearly on its T-shaped track. Now I need a reference for 1970s retro-futuristic travel poster style with vintage transport typography...” Turn 4 — Think & Final Answer: ← no query_knowledge called at all

“...I have both references I need. First, IMG_001 for the Aérotrain. Second, IMG_007 for the retro-futuristic travel poster style...Use the 1970s vintage travel poster aesthetic, warm saturated color palette, and bold retro typography style from the second reference image...”

Final program (gen_prompt): “Retro-futuristic 1970s-style travel poster, featuring the Aérotrain I80 gliding on its inverted T-shaped concrete track as shown in the first reference image. Use the 1970s vintage travel poster aesthetic...from the second reference image. Prominent bold vintage transport advertisement text across the poster reads: ‘Aérotrain I80 | Official World Speed Record: 430.4km/h (267mph) | 1974’...”

Skills called: none. Missing: text_rendering — no guidance on typography decomposition or spatial placement of text. Critical failures:

- 1. Skipped text_rendering — crammed all text into a single long string “Aérotrain I80 | Official World Speed Record: 430.4km/h (267mph) | 1974” instead of decomposing into separate lines with spatial anchors → text_rendering=fail.
- 2. Used a second style reference image instead of skill guidance — relied on the model to mimic typography from a reference rather than applying explicit font/layout rules → aesthetic=fail, attribute=fail.
- 3. Same correct facts as best, but without text_rendering guidance the poster is unreadable: spatial_layout=fail.

###### Case 2 — Extracted Experience Slots (Delta

- S1 Search strategy: Execute parallel searches for both the historical event (speed/year) and the physical design (inverted T-track) to ground both text and visuals.
- S2 Knowledge activation: Call text_rendering to obtain guidance on line-level text decomposition, spatial placement, and typographic styling — a style reference image alone cannot replace this.
- S3 Reference selection: Select one reference for the vehicle/infrastructure accuracy; do not add a second “style” reference as a substitute for skill guidance.
- S4 Prompt construction: Decompose text into explicit separate lines (e.g. first line, second line) and assign a specific spatial location (e.g. “lower third”) instead of a single concatenated string.
- S5 Failure avoidance: Cramming multiple text elements into one line with separators (“|”) causes the generator to render garbled, overlapping text — always decompose into spatially anchored lines.

[Figure 70]

[Figure 71]

Worst (R = 0.40): garbled text, broken poster

Best (R = 0.875): clear typography, correct layout

- Figure 8: Case 2 generated images. Both trajectories retrieved the same correct facts (430.4km/h, 1974). The best trajectory called text_rendering and decomposed text into explicit lines with spatial anchors. The worst skipped all skills and crammed text into one string, resulting in unreadable typography.

###### Case 3 — User Request

“Generate a street view with two famous European housing complexes side by side. On the left, feature the colorful, uneven apartment building designed by Friedensreich Hundertwasser in Vienna. On the right, feature the tilted yellow houses designed by Piet Blom in Rotterdam. Above the entrance of the left building, add a large wooden sign reading ‘ORGANIC’. Above the right building, place a sleek metal sign reading

‘GEOMETRIC’.”

###### Best Trajectory R

Turn 1 — Think & Tool Call: image_search("Friedensreich Hundertwasser Vienna colorful uneven apartment building exterior")

“...first, I need to get visual references for both housing complexes. Let’s start with an image search for Hundertwasser building Vienna...”

- Turn 2 — Think & Tool Call: image_search("Piet Blom Rotterdam tilted yellow houses exterior")

“...IMG_001 is the clear exterior shot, that’s perfect for the left building. Now I need the reference for the Piet Blom Rotterdam tilted yellow houses...”

- Turn 3 — Think & Tool Call: query_knowledge("spatial_layout") ← critical skill

“...now I have both building references. First, spatial layout: because we have left/right arrangement of buildings, signs above each entrance, street view perspective...”

- Turn 4 — Think & Tool Call: query_knowledge("text_rendering")

“...I need text rendering guidance because there are two signs with specific text, different materials...”

- Turn 5 — Think & Tool Call: query_knowledge("physical_material_consistency")

“...I need physical material consistency guidance to make sure the building materials, sign materials are correct...”

- Turn 6 — Think & Final Answer:

“...spatial layout: foreground is paved grey asphalt street with a narrow concrete sidewalk running between the two buildings...Midground left side of the frame: the colorful uneven Hundertwasser apartment building...Midground right side: the tilted yellow cube houses...Centered directly above the main entrance of this left building: a large rough natural wood sign...reading ‘ORGANIC’...”

Final program (gen_prompt): “Realistic street view photograph, soft overcast daylight...Foreground: paved grey asphalt street with a narrow concrete sidewalk running between the two buildings...Midground left side of the frame: the colorful uneven Hundertwasser apartment building, fully matching the architectural details...of the first reference image...Centered directly above the main entrance of this left building: a large rough natural wood sign, with bold black sans-serif text carved into the wood reading ‘ORGANIC’...Midground right side...the tilted yellow cube houses...a sleek brushed steel sign...reading ‘GEOMETRIC’...Both buildings are at equal height, spaced 10 feet apart with the sidewalk between them...”

Skills called: spatial_layout ✓, text_rendering ✓, physical_material_consistency ✓ Judge: spatial=pass, text=pass, attribute=pass, material=pass.

###### Worst Trajectory R

Turn 1 — Think & Tool Call: image_search("Friedensreich Hundertwasser Vienna colorful uneven apartment building exterior")

“...first, I need to get visual references for both housing complexes. Let’s start with image search for Hundertwasser Vienna building...” Turn 2 — Think & Tool Call: image_search("Piet Blom Rotterdam tilted yellow houses exterior street view")

“...IMG_003 looks like a clear, standard exterior shot. Now I need to get the reference for the Piet Blom Rotterdam tilted yellow houses...”

- Turn 3 — Think & Tool Call: query_knowledge("text_rendering")

“...now I have both building references. Next, I need guidance for text rendering because there are two specific signs: wooden ‘ORGANIC’ above the left building, metal ‘GEOMETRIC’ above the right...”

- Turn 4 — Think & Final Answer: ← skipped spatial_layout and physical_material

“...I have all the info I need...the gen_prompt needs to follow the text rendering guidance, plus spatial layout: left is Hundertwasser building, right is cube houses side by side...”

Final program (gen_prompt): “Photorealistic sunny daytime European street view, two famous housing complexes placed side by side at equal width. On the left, copy the colorful uneven facade...from the first reference image. On the right, copy the tilted geometric shape...from the second reference image. Above the main entrance of the left building, place a large weathered wooden sign...spelling out ‘ORGANIC’...Above the main entrance of the right building, place a sleek brushed metal sign...spelling out ‘GEOMETRIC’...Foreground has a cobblestone street...”

Skills called: text_rendering ✓ Missing: spatial_layout, physical_material_consistency Critical failures:

- 1. Missing spatial_layout — used vague “placed side by side at equal width” without frame-relative coordinates

→ buildings overlap or merge, spatial_layout=fail.

- 2. Without precise spatial anchors, text signs float or attach to the wrong building → text_rendering=fail, attribute_binding=fail.
- 3. Missing physical_material_consistency — sign materials (wood vs metal) not properly grounded → physical_material=partial.

[Figure 72]

[Figure 73]

Best (R = 0.80): correct layout, both signs legible Worst (R = 0.40): merged buildings, text failure

- Figure 9: Case 3 generated images. The best trajectory called spatial_layout and used framerelative coordinates (“midground left/right side of the frame, spaced 10 feet apart”). The worst skipped spatial_layout and used vague “side by side at equal width,” causing the buildings to merge and text signs to fail.

###### Case 3 — Extracted Experience Slots (Delta

- S1 Search strategy: Include the requested camera angle in all image search queries to ensure consistent perspective references for the street view.
- S2 Knowledge activation: Call spatial_layout to obtain guidance on using absolute frame-relative coordinates text_rendering alone cannot prevent spatial overlap or attribute leakage between adjacent buildings.
- S3 Reference selection: Explicitly state the intended spatial position (e.g. “placed on the left side of the scene”) in the reference image notes.
- S4 Prompt construction: Use absolute frame-relative coordinates (e.g. “Midground left side”) and explicitly mount text signs to these specific spatial zones with depth layering.
- S5 Failure avoidance: Strictly partition the prompt using depth layers and absolute horizontal positions to isolate the architectural and textual attributes — vague “side by side” causes attribute bleeding between buildings.

#### B.4 Prompt-Keyed Visual Experience Memory

For each prompt, the self-evolution pipeline samples multiple rollouts. After image generation and judging, the system identifies the best and worst rollout by scalar reward. If the reward gap exceeds a minimum margin, the comparison is added to a pending buffer. Comparisons caused only by missing references or other protocol failures are ignored because they do not provide a reusable visual strategy.

At each memory update, high-gap comparisons are summarized into five dimensions: search strategy, knowledge activation, reference selection, prompt construction, and failure avoidance. The memory is prompt-keyed rather than global. Each entry is attached to the source prompt that produced the best-worst comparison and stores a source-prompt embedding matched to the generated strategy key and other information. Each slot keeps a capacity-limited buffer and trims by reward gap and recency when full.

- Table 8: Prompt-keyed visual experience slots used to construct the privileged teacher context.

Slot Extracted contrast Teacher-side effect

- S1 SEARCH STRATEGY Factual gap, query decomposition, evidence verification, and stopping condition

Guides whether to search, what to verify, and when evidence is sufficient

- S2 KNOWLEDGE ACTIVATION

Called, missed, or unnecessary generation knowledge explaining the reward gap

Biases query_knowledge calls toward relevant visible failure modes

- S3 REFERENCE SELECTION

Useful, redundant, or harmful retrieved image candidates and their intended roles

Improves filtering, deduplication, and ordinal reference binding

- S4 PROMPT CONSTRUCTION

Program structure that carries evidence, reference roles, and knowledge guidance into generation

Improves prompt-reference program synthesis

- S5 FAILURE AVOIDANCE

Recurring low-reward visual patterns for similar requests

Adds guards for text, counting, material, anatomy, grounding, and style failures

#### B.5 Source-Prompt Bundle Retrieval

When constructing teacher context for a current prompt, the system retrieves by source-prompt similarity rather than by experience-text similarity. Let B be the set of historical source prompts stored in the memory. The retrieved source prompt is

cos(e(x),e(xj)). (20)

x˜ = arg max

xj∈B

The teacher receives the entries from all five slots that share the same source prompt x˜. This bundle retrieval avoids mixing unrelated lessons from different cases. If no entry exists, no teacher context is produced and the SDL term is skipped for that row.

#### B.6 GRPO Rollout Loss

For each prompt x, the behavior policy samples K complete visual rollouts. Each rollout contains tool-call tokens, final-answer tokens, selected references, the generated image, and a scalar

reward. Environment observations and generated images are not optimized directly; the policy loss is computed only on assistant tokens with mask mi,t.

Let Ri be the final mixed reward of rollout i. The group-relative advantage is

K

K

Ri − R¯ σR + ϵadv

1 K

1 K

, R¯ =

(Rj − R¯)2. (21)

Ai =

Rj, σR =

j=1

j=1

For a sampled token yi,t, the policy ratio is

(yi,t | hi,t))). (22) The clipped GRPO surrogate minimized by the agent is

ui,t(θ) = exp(log πθ(yi,t | hi,t) − sg(log πθ

old

1 i,t mi,t i,t

mi,t min ui,t Ai,clip(ui,t,1 − ϵℓ,1 + ϵh) Ai + βrefKref, (23)

LGRPO = −

where Kref denotes the reference-policy regularization term when enabled. The same final reward supplies the group advantage for all optimized assistant tokens in a rollout, including tool decisions and final prompt-reference program construction.

#### B.7 Experience-Conditioned SDL Contexts

Experience-conditioned SDL uses two contexts for the same sampled assistant tokens. The student receives the normal inference context. The teacher receives the same context with the retrieved source-prompt experience bundle inserted into the system prompt before the tool definitions. Teacher and student share model weights; the teacher view is detached and privileged only during training. The returned policy is the student policy and does not require dynamic experience slots at inference.

The sampled-token reverse-KL estimator and the importance correction follow prior self-distillation and KL-estimator work [43, 33, 37, 24]; the GenEvolve-specific component is the construction and retrieval of visual experience for the teacher branch. For token yi,t in rollout i, define

pi,t = πθS(yi,t | hi,t), qi,t = sg πθE(yi,t | h˜i,t) , ℓi,t = log pi,t − log qi,t. (24) The sampled-token KL estimator is

k3(ℓi,t) = exp(−ℓi,t) − 1 + ℓi,t. (25) Let mEi,t select valid assistant tokens from rows with real teacher context, and let

πθS(yi,t | hi,t) sg πθS

ρoni,t = min

,ρmax (26) be the clipped student-centered on-policy importance ratio. The implemented SDL loss is

(yi,t | hi,t)

old

1 i,t mEi,t i,t

mEi,t min ρoni,tk3(ℓi,t),ctok . (27)

LSDL =

SDL is applied on the same on-policy responses used by the group-relative rollout loss, so it adds dense token-level guidance without introducing a separate offline imitation dataset. Compared with skill-conditioned self-distillation for text agents, the teacher conditioning object here is a promptkeyed visual experience bundle extracted from generated-image best-worst trajectory pairs. The SDL coefficient and clipping constants are kept fixed across all reported ablations.

Token-level evidence of teacher guidance. To verify that experience-conditioned SDL indeed provides actionable token-level guidance rather than merely matching the student distribution, we instrument a representative held-out rollout and inspect the teacher-vs-student log-probabilities at the assistant tokens where the SDL loss is applied. We use a request to render the Wuppertal Schwebebahn (a historical suspended monorail in Germany) in a stylised illustration that simultaneously requires (i) factual visual identity of a real-world landmark, (ii) the exact number of visible carriages from a specified viewpoint, and (iii) a non-photorealistic style/layout transfer. The retrieved experience bundle for this prompt focuses on verifying the exact count of visible components from the specified perspective using image references before applying quantity counting, which couples three skills the agent must coordinate: factual search, visual reference selection, and quantity counting. Figure 10 shows two complementary effects of the teacher signal on this trajectory:

###### (a) Teacher opposes student — re-routes to experience-recommended token

###### (b) Teacher supports student — boosts probability of correct decision

T0 'spatial'

0.53 0.96

→ switch from filler to action 'call'

T0 'let' → 'call'

T4 'count'

0.50 0.79

→ scope to both visible sides

T2 'spatial' → 'both'

T4 'reference'

0.59 0.79

→ anchor on 'factual' identity

T0 'correct' → 'factual'

T0 'count'

0.64 0.79

→ route to spatial_layout skill

T0 'shape' → 'layout'

T0 'search'

0.62 0.85

→ begin with a search query

T0 'first' → 'query'

T4 'should'

0.49 0.82

→ verify wheel as factual detail

T0 'wheel' → 'factual'

- T0 'number'

T1 'after'

- T1 'exactly'

0.52 0.84

→ find evidence first

- T0 'confirm' → 'find'

T4 'gen' → 'reference'

- T1 'driving' → 'visible'

- T2 'view' → 'profile'

0.44 0.68

→ anchor on reference image

0.63 0.79

→ refine to 'visible from camera'

T4 'matching'

0.55 0.85

→ specify 'profile' view

T0 'side'

0.78 1.00

→ specify spatial composition

T4 'both' → 'spatial'

pstudent

T0 'specifically'

0.74 0.91

pteacher (color = role)

−4 −3 −2 −1 0 1 2 Δlogp = logpteacher − logpstudent

0.2 0.4 0.6 0.8 1.0

top-1 probability (arrow: pstudent → pteacher)

###### Token role (color):

Skill name

Tool / search keyword

Planning verb

Decision modifier

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Figure 10: Token-level evidence of experience-conditioned SDL guidance. Representative tokens from a single held-out rollout illustrate the two complementary effects of the teacher signal under the prompt-keyed experience bundle. The case asks for a stylised rendering of the Wuppertal Schwebebahn that must respect a real landmark’s identity, layout and a specified visible-carriage count; the bundle instructs the agent to verify the count from a specified perspective using image references before invoking quantity counting. (a) Teacher opposes the student: at decision tokens where the student commits to a generic or off-target word (e.g., “shape”, “correct”, “first”, “gen”), the teacher concentrates its mass on an experience-recommended alternative (e.g., “layout”, “factual”, “query”, “reference”), producing a large negative ∆log p. Bar color encodes the role of the teacherrecommended token: skill name, tool/search keyword, planning verb, or decision modifier. (b) Teacher supports the student: at tokens where the student is already on-policy but uncertain, the

teacher boosts the same top-1 token from pstudent (hollow circle) to pteacher (filled circle, color = role). Both effects appear at multiple turns (T0–T4) and consistently track the case-specific bundle (factual identity, reference-based counting, spatial composition), confirming that SDL provides dense token-level guidance complementary to the trajectory-level GRPO reward.

- • (a) Teacher opposes student. On tokens where the student commits to a less effective lexical choice, the teacher concentrates its mass on a different, experience-recommended token, producing

a strongly negative ∆log p = log pteacher − log pstudent. The opposed tokens span multiple turns (T0, T1, T2, T4) and three decision aspects of this case. (i) Search vs free reasoning at task entry: at T0 the teacher rewrites planning fillers into explicit tool calls (“let” → “call”, “first” → “query”, “confirm” → “find”), enforcing the bundle’s instruction to gather factual evidence before describing the scene. (ii) Skill activation: at T0 the teacher replaces the generic descriptor “shape” with the named skill “layout” (routing the agent into spatial_layout), and at T4 it pushes “gen” → “reference” and “both” → “spatial” when the student is composing the final program, anchoring the description on the retrieved reference and on spatial composition. (iii) Grounded modifiers: the teacher tightens loose adjectives into the bundle’s grounding cues (“correct” → “factual”, “wheel” → “factual”, “view” → “profile”, “driving” → “visible”), forcing the agent to express the condition in the specific visual form the bundle prescribes.

- • (b) Teacher supports student. On tokens where the student is already moving in the correct direction but uncertain, the teacher boosts the same top-1 token’s probability, producing a positive

gap pteacher − pstudent > 0. The boosts coincide with the same case-specific concepts: at T0 the teacher reinforces the routing into the spatial_layout and quantity_counting skills (‘spatial’: 0.527→0.961, ‘count’: 0.637→0.793) and the choice of search as the first action (‘search’: 0.622→0.848); at T4 it strengthens the binding to the retrieved image (‘reference’: 0.588→0.785) and the explicit count statement in the final program (‘count’: 0.499→0.785). The reinforced tokens cover skill names, search keywords, planning verbs, and decision modifiers, matching the experience-bundle pattern observed in (a).

The two patterns together show that experience-conditioned SDL acts as a fine-grained controller: it re-routes the agent at the few decision tokens where free-form generation would diverge from the

experience-distilled policy, while simultaneously sharpening the agent’s confidence on the many tokens where the student is already correct. Both effects in this case track the same underlying experience bundle (factual identity, reference-based counting, spatial composition), confirming that the SDL signal is content-specific rather than a generic regulariser, and that it complements the trajectory-level GRPO reward with dense token-level guidance.

### C Training Details

#### C.1 Supervised Stage Configuration

The supervised stage cold-starts Qwen3-VL-8B-Instruct into a tool-orchestrated image-generation agent. We fine-tune the language-policy part on supervised GenEvolve-Data trajectories using a longcontext multimodal training stack, while keeping the visual encoder fixed. Only assistant-side tokens are optimized, including reasoning, tool calls, and final prompt-reference programs; user prompts and tool observations are used as context and masked from the loss. Concrete hyper-parameters, optimizer settings, batch sizes, and the LLaMA-Factory configuration we use are summarised in Table 9.

Table 9: Supervised trajectory-tuning configuration. We list the concrete values used for the released checkpoint; the same setup is used for all SFT-based baselines reported in the main paper.

Setting Value Notes Backbone Qwen3-VL-8B-Instruct [1] Base multimodal agent policy. Framework LLaMA-Factory [52] Long-context multimodal SFT with

DeepSpeed ZeRO-3 sharding.

Trainable part Language-policy parameters freeze_vision_tower=true, freeze_multi_modal_projector=true; visual encoder fixed.

Data split GenEvolve-Data SFT split 8,800 training and 200 held-out trajectories. Sequence length cutoff_len=32,768 Long-context to fit multi-turn tool trajec-

tories. Optimizer AdamW (default in LLaMA-Factory) weight_decay=1×10−6. Learning rate 1×10−5 cosine schedule with

warmup_ratio=0.02. Epochs / batch 2 epochs, 16 GPUs, micro-bsz=2, grad-

Effective global batch size = 32 sequences/step.

accum=1

Precision / kernel bf16, FlashAttention-2 Activation checkpointing on. Loss masking Assistant-token only User prompts and tool-response tokens

are masked from the SFT loss.

#### C.2 Self-Evolution and SDL Configuration

Self-evolution starts from the supervised trajectory checkpoint. For each update, the agent samples multiple on-policy rollouts per prompt, and each rollout produces a prompt-reference program whose image is rendered by the reference-conditioned generator. The generated images and final programs are scored by image and text judges, respectively (see Appendix D.2 and Table 11 for the judge backbone and rubric). Best-worst trajectory pairs with sufficient reward gaps are mined into promptkeyed visual experience, which is retrieved only for the privileged teacher branch during SDL. The deployed policy is always the student branch and does not use runtime visual-experience memory. The full set of GRPO, SDL, reward, and visual-experience-memory hyper-parameters used to train the released GenEvolve checkpoint is given in Table 10. Concretely, SDL is restricted to special tokens, and within each sequence we keep only the top 10% of those tokens by |log πθE − log πθS| (SDL_TOP_K_FRAC=0.1); together with the importance-ratio cap ρmax = 2 this isolates the few decision tokens where the experience-conditioned teacher disagrees most strongly with the student.

#### C.3 Self-Evolution Training Dynamics

- Figure 11 visualizes the two core training signals during self-evolution: the mixed reward and the Visual Experience Distillation (SDL) loss.

- Table 10: Self-evolution (GRPO + experience-conditioned SDL) configuration. We list the concrete values used to train the released GenEvolve checkpoint; all ablations in the main paper share the same setup unless stated otherwise.

Setting Value Notes Framework and infrastructure RL framework rLLM/verl FSDP actor with SGLang rollout back-

end. Hardware 1 node × 8 GPUs fsdp_size=8, parameter+optimizer offload enabled. Initialization SFT checkpoint (Table 9) Cold start from the supervised trajectory checkpoint.

Rollout sampling Prompt batch / rollouts 8 prompts/step × 6 rollouts each train_batch_size=8, n=6 (group

size for GRPO advantage). Sampling temperature=0.7, top-p=0.95, top-k=−1 Same val sampling. Max prompt / response 6,144 / 30,000 tokens Multi-turn tool-orchestrated rollouts. Tool-call budget MAX_LLM_CALL_PER_RUN=11 Allows search + image-search + skill

queries plus final answer. Generator Qwen-Image-Edit-2511 (open) / Nano Ba-

References capped at QWEN_EDIT_MAX_REF_IMAGES=2.

nana Pro (strong)

Reward (mixed image + text) Image judge Gemini 3.1 Pro Preview, KScore protocol Faithfulness/Visual/Text/Aesthetic with

weights 0.1:0.4:0.4:0.1. Text judge Gemini 3.1 Pro Preview, program-sufficiency 5-bin scoring on {0, 0.25, 0.5, 0.75, 1}. Final reward R = 0.5 Rimg + 0.5 Rtext GEN_REWARD_TEXT_COEF=0.5. GRPO objective Algorithm Group-relative PPO surrogate adv_estimator=grpo, low-variance

KL. Learning rate 1×10−6 (actor) Cosine, no warmup. Clip ratios ϵℓ = 0.20, ϵh = 0.28 Asymmetric high clip [35].

KL regularizer βkl = 10−3 (KL controller) use_kl_loss=False, kl_coef=1e-3. Aggregation seq-mean-token-sum Same aggregation for SDL, following

Gen-Searcher [14].

SDL (Visual Experience Distillation) SDL coefficient λSDL = 2.0 actor.sdl_loss_coef=2.0. Importance-ratio cap ρmax = 2.0 sdl_is_clip=2.0; on-policy studentcentered ratio capped per token, lowvariance KL estimator k3 clamped to [−10, 10]. The cap rarely fires (sdl_rho_clip_frac≈ 0).

Visual experience memory Bundle summarizer Gemini 3.1 Pro Preview temperature=0.0,

max_tokens=8192, timeout=90s, RPM cap 80.

###### Min reward gap δmin = 0.20 EXPERIENCE_MIN_REWARD_GAP=0.20; best/worst pair retained only if |∆R| ≥ δmin.

Comparisons / step Up to 8 pairs (one per prompt group) EXPERIENCE_MAX_COMPARISONS=8,

###### TOP_GROUPS_PER_STEP=8.

Bundle schema 1 bundle per comparison Each bundle stores retrieval_key={trigger, source_prompt_summary} plus decision_guidance (focus + 6 imperative bullet lists).

Buffer capacity 500 bundles, FIFO + reward-gap eviction EXPERIENCE_BUFFER_CAPACITY=500. Prompt embedder Qwen3-Embedding-0.6B (CPU) max_length=512, last-token pool with

L2 normalization, cosine similarity for nearest-bundle retrieval.

###### (a) Reward

###### (b) Visual Experience Distillation Loss

5.5

0.66

SDL loss

5.2

0.64

5.0

MeanReward

4.8

0.62

SDLLoss

4.5

0.60

4.2

0.58

4.0

3.8

Mean reward

0.56

0 25 50 75 100 125 150 175 200

0 25 50 75 100 125 150 175 200

Training Step

Training Step

- Figure 11: Self-evolution training dynamics. (a) Mean reward across training steps. The smoothed curve (window=25) shows a steady upward trend, indicating that the agent progressively produces higher-quality tool-orchestrated trajectories and prompt-reference programs. (b) SDL loss across training steps. The decreasing trend indicates that the student policy gradually converges toward the experience-conditioned teacher distribution, internalizing the visual experience extracted from best-worst trajectory comparisons. Translucent dots show per-step raw values; the dashed line is a linear trend fit.

Reward progression. The mean reward increases steadily over training, reflecting improved generation quality as measured by both the image-side KScore judge and the text-side program-sufficiency judge. Per-step variance is expected because each batch contains diverse prompts spanning knowledgeanchored and quality-anchored tracks with different tool-use demands. Despite this variance, the linear trend confirms consistent improvement: the agent learns to issue more targeted search queries, select more relevant visual references, activate appropriate generation knowledge, and synthesize better prompt-reference programs.

SDL loss. The SDL loss measures the reverse-KL divergence between the student policy and the experience-conditioned teacher on the same on-policy tokens. Its decreasing trend indicates that the student progressively absorbs the privileged strategic guidance provided by the retrieved decision guide. Notably, SDL loss does not collapse to zero: this is expected because the teacher always sees the latest retrieved experience while the student operates under the plain inference context, maintaining a constructive gap that continues to provide learning signal throughout training. The joint decrease in SDL loss and increase in reward confirms that the two objectives are complementary: GRPO identifies which trajectories are better at the trajectory level, while SDL provides dense token-level guidance about why the better trajectory should be preferred.

### D Evaluation Details

#### D.1 GenEvolve-Bench Categories

GenEvolve-Bench primarily evaluates final generated-image quality under two complementary prompt tracks: Knowledge-Anchored and Quality-Anchored. Knowledge-Anchored cases emphasize externally grounded entities, events, places, products, artifacts, public figures, and other visual facts. Quality-Anchored cases emphasize visible generation requirements such as text-critical generation, spatial layout, anatomy/body coherence, attribute binding, quantity counting, physical/material consistency, aesthetics, and creative transformation. The benchmark metadata also records category, difficulty, search flags, expected reference targets, and quality-requirement tags, which are used for subset analysis rather than as the main benchmark score.

#### D.2 Reward Rubric

For final image evaluation, we follow the KnowGen-style KScore protocol adopted by recent agentic image-generation work, especially Gen-Searcher [14]. The visual judge compares the generated image against the user request and the fixed GT image associated with the same held-out case. We intentionally reuse the same four-dimensional rubric for raw generators and agent-produced prompt-reference programs, instead of designing a GenEvolve-specific evaluator.

The visual judge is implemented as a single LLM call per sample using Gemini 3.1 Pro Preview. The system prompt enforces a strict 3-level scale per dimension ({0,0.5,1} for faithfulness/visual correctness/text accuracy/aesthetics), the prompt also instructs the judge to first list the 2–5 hard constraints of the request before scoring. We use temperature=0.0, max_tokens=8192, and a deterministic JSON output schema.

Let sf,sv,st,sa ∈ [0,1] denote faithfulness, visual correctness, text accuracy, and aesthetics. The aggregate image score, reported as KScore, is

Simg = 0.1sf + 0.4sv + 0.4st + 0.1sa. (28)

The two large weights on visual correctness and text accuracy reflect the benchmark’s focus on grounded, externally checkable details rather than generic prompt fluency. When the prompt does not require any readable text the judge sets text_accuracy_na=true; in that case the score is renormalised over the remaining three dimensions before the weighted sum.

Text-side program-sufficiency judge (training only). During self-evolution we additionally call the same Gemini 3.1 Pro Preview backend with a different system prompt that scores the agent’s final prompt-reference program (without seeing the image) on a 5-bin scale {0,0.25,0.5,0.75,1}. The text judge measures whether the program contains enough grounded facts, ordinal reference bindings, activated generation knowledge, and executable constraints for a strong generator to reproduce the intended image. The training reward is the equally weighted mixture

R = (1 − α)Simg + α Stext, α = 0.5, (29)

controlled by the GEN_REWARD_TEXT_COEF environment variable. In the main tables we report only KScore so that raw generators and agentic systems are directly comparable; the text reward is used only during GRPO+SDL training.

Table 11: Visual judge dimensions used in the GenEvolve-Bench evaluation protocol.

Dimension Weight What the judge checks Faithfulness 0.1 Semantic alignment with the request, including required entities, style, scene intent,

relations, and explicit constraints.

Visual correctness 0.4 Correctness of grounded visual details, reference identity, spatial structure, anatomy, material behavior, counting, and verifiable visual cues. Text accuracy 0.4 Correct generation of required text, labels, logos, signage, numbers, and other readable

visual symbols. Aesthetics 0.1 Overall coherence, visual appeal, composition, and absence of severe generation artifacts.

#### D.3 External WISE Evaluation Protocol

We complement the appendix with the protocol details for the external WISE evaluation, whose results are reported in the main text (Section 6, Table 2) and visualized alongside our internal benchmark in Figure 1. The evaluation uses the original WISE release [26] and its six category groups: culture, time, space, biology, physics, and chemistry. The agent receives only the WISE prompt and produces a prompt-reference program through the same rollout interface used for GenEvolve-Bench. The output image is generated by Qwen-Image-Edit, scored under the WISE three-dimension protocol using GPT-4o-2024-05-13 as the judge, and aggregated by the official WiScore script. Missing or failed generations are counted as zero rather than skipped, ensuring that the reported WiScore reflects end-to-end agent reliability rather than only generation quality on completed cases. Directgenerator baselines are taken from the WISE leaderboard or original papers, and agentic baselines are reproduced under the same protocol with their released checkpoints.

### E Additional Qualitative Results

To further demonstrate the generality and effectiveness of our self-evolved agent policy across varied open-ended generation challenges, we provide additional qualitative results in Figures 12 and 13, paired respectively with Nano Banana Pro and Qwen-Image-Edit as downstream generators. The same trained agent policy is used in both settings; only the final reference-conditioned generator differs.

These examples are sampled from the held-out evaluation split and span all eight callable generation skills: spatial layout, text rendering, quantity counting, attribute binding, anatomy and pose, creative drawing, physical material consistency, and aesthetic drawing. Each generation involves the agent autonomously deciding which factual or visual evidence to retrieve, which references to select, which skills to activate, and how to compose the prompt-reference program for the downstream generator. The visual diversity (architecture, creative transfer, scientific illustration, street scenes, anatomy, material physics, and quantity-anchored compositions) reflects the breadth of skills the agent learns to coordinate. Together, the two galleries also illustrate the generator-transferability of the learned tool-orchestrated policy: the same trajectories yield strong open-source results with Qwen-Image-Edit while transferring to higher-fidelity outputs when paired with the stronger Nano Banana Pro generator.

### F Prompt and Template Details

#### F.1 Agent Final Answer Template

The agent’s final <answer> must contain a single parseable JSON object. The natural-language gen_prompt must reference selected images by ordinal phrase and must not contain IMG_### identifiers or raw URLs. The reference_images list is sorted by img_id ascending so that the ordinal phrases in gen_prompt resolve unambiguously.

{

"gen_prompt": "A detailed generator-facing prompt that refers to selected images using only ordinal phrases such as ’the first reference image’.",

"reference_images": [ {"img_id": "IMG_001", "note": "what to copy from this image"}, {"img_id": "IMG_004", "note": "what to copy from this image"}

] }

#### F.2 Judge Output Template

The reward judge returns scalar subscores and diagnostics. Diagnostics are used for analysis and experience construction; they are not directly optimized as independent rewards.

{

"faithfulness": 0.0, "visual_correctness": 0.0, "text_accuracy": 0.0, "aesthetics": 0.0, "overall": 0.0, "failure_tags": ["wrong_count", "weak_reference_use"], "skill_diagnostics": {

"text_rendering": "pass", "spatial_layout": "partial", "quantity_counting": "fail"

} }

#### F.3 Experience Bundle Template

Each best-vs-worst comparison is summarized into one compact bundle rather than five independent slot entries. A bundle has two parts: a retrieval_key (trigger + source_prompt_summary) used as the embedding key for nearest-bundle lookup, and a decision_guidance block containing one short decision_focus and six lists of imperative action-level bullets. Bullets are derived first from observed best-vs-worst differences; when no clear difference is visible for a category, we fall back to the best trajectory’s behaviour as a default plan, marked with the literal prefix Standard:.

##### {

[Figure 74]

- Figure 12: Additional qualitative results of GenEvolve paired with Nano Banana Pro. The agent autonomously orchestrates search, reference selection, and skill activation to produce high-fidelity images across diverse categories. Examples cover spatial layout, text rendering, quantity counting, attribute binding, anatomy/pose, creative transfer, material physics, and aesthetic drawing skills.

[Figure 75]

- Figure 13: Additional qualitative results of GenEvolve paired with Qwen-Image-Edit. Using the same trained agent policy as in Figure 12, paired here with an open-source downstream generator. The consistent quality across the two generators demonstrates that GenEvolve learns generatortransferable tool orchestration rather than overfitting to one specific renderer.

"retrieval_key": { "trigger": "When/why to retrieve this bundle (8-25 words, second-person, no named entities).", "source_prompt_summary": "What image type is being requested (8-25 words, mid-level visual-role phrases)."

}, "decision_guidance": {

"decision_focus": "Single most important pattern (1 sentence).", "recommended_tool_plan": ["...", "..."], "search_query_guidance": ["...", "..."], "skill_routing_guidance": ["...", "..."], "reference_selection_guidance": ["...", "..."], "prompt_program_guidance": ["...", "..."], "failure_guards": ["...", "..."]

} }

#### F.4 Representative Implementation Prompt Excerpts

This section reports representative implementation prompt templates used in the current codebase. We typeset long raw templates in breakable prompt boxes so that the original placeholders and JSON schemas remain readable without overflowing the page. Deployment-specific endpoints, API keys, and private paths are omitted.

#### Prompt (Bundle Summarizer (Best/Worst to One Decision Guide))

You are an experience summarizer for a tool-using image-generation agent. You will receive one contrastive best-vs-worst pair for the SAME user prompt:

- - the original user prompt,
- - the best trajectory and worst trajectory (with all tool calls + thinking + final answer),
- - their final gen-prompt programs,
- - which static skills were called via query_knowledge,
- - rewards and per-dimension judge diagnostics.

Your task is NOT to write a case report. Your task is NOT to output Bad/Good/Rule. Your task is to extract a TARGETED DECISION GUIDE from the best-vs-worst

differences.

The guide must be useful for future prompts that match the same retrieval_key. It must directly guide agent decisions on the action-token level:

tool ordering, search query decomposition, skill routing, reference selection, and the final prompt-reference program.

The agent has 3 tools: search (text), image_search, query_knowledge (static

skills). Static skill names: spatial_layout, aesthetic_drawing, text_rendering, creative_drawing, anatomy_body_coherence, attribute_binding, physical_material_consistency, quantity_counting.

## Comparison Metadata prompt_hash: {prompt_hash} reward_gap: {reward_gap:.3f}

## User Prompt {user_prompt}

## BEST Rollout outcome: {best_outcome} called_skills: {best_called_skills} final_gen_prompt: {best_gen_prompt} trajectory: {best_trajectory}

## WORST Rollout outcome: {worst_outcome} called_skills: {worst_called_skills} final_gen_prompt: {worst_gen_prompt} trajectory: {worst_trajectory}

## Output schema (return ONLY this JSON, no prose, no code fences) {

"retrieval_key": { "trigger": "...",

"source_prompt_summary": "..."

}, "decision_guidance": {

"decision_focus": "...", "recommended_tool_plan": ["...", "..."], "search_query_guidance": ["...", "..."], "skill_routing_guidance": ["...", "..."], "reference_selection_guidance": ["...", "..."], "prompt_program_guidance": ["...", "..."], "failure_guards": ["...", "..."]

}

} ## retrieval_key requirements

- - trigger (8-25 words, second person): combine a decision-risk phrase (factual identity, exact text, exact count, multi-object composition, material physics, spatial layout, style transfer, anatomical accuracy, reference binding, readable typography) with the constraint pattern that pairs with it. No named entities. No third-person framing.
- - source_prompt_summary (8-25 words): describe the requested image type with 2-4 mid-level visual-role phrases (product-label design, museum-style object composition, scientific anatomical diagram, etc.). No named entities.

## decision_guidance requirements

- PRIORITY 1: every bullet should derive from an actual BEST/WORST difference.
- PRIORITY 2: when BEST and WORST do NOT differ for a category, fall back to the BEST trajectory’s behaviour as a default plan and prefix the bullet with the literal "Standard:". If neither priority yields signal, output [] for that category.

Style: imperative, action-space language only (name the tool, the static skill, the reference role, or the final-program element); 12-22 words per bullet; no case framing ("the better trajectory ..."); no named entities; avoid generic phrases ("use better references", "write a detailed prompt").

Field meaning:

- - decision_focus: the single most important behavioural pattern (1 sentence).
- - recommended_tool_plan: when to call search vs image_search vs query_knowledge, in what order.
- - search_query_guidance: how to decompose / phrase queries by axis (factual identity, style, material, typography, count).
- - skill_routing_guidance: which static skill names to activate for which constraint types; cite skills only when the trajectory used them.
- - reference_selection_guidance: how to assign roles to retrieved images (identity / material / typography / layout reference); when to skip refs.
- - prompt_program_guidance: how the final gen-prompt should bind references, anchor spatial relations, and prevent attribute bleed.
- - failure_guards: concrete failure modes visible in WORST and how to avoid them; if WORST shows none, extract from BEST as "Standard:".

Output the JSON object only. No explanation.

#### Prompt (Query-Key Generation (New Prompt to Retrieval Key))

You generate a mid-level retrieval key for an image-generation user prompt. The key will be used to look up previously-stored agent experience bundles.

User prompt: {user_prompt}

## Output schema (return ONLY this JSON, no prose, no code fences) {

"trigger": "...", "source_prompt_summary": "..."

} ## Field requirements

- - trigger (8-25 words, second person): combine a decision-risk phrase with the constraint pattern that pairs with it. No generic descriptors. No named entities.
- - source_prompt_summary (8-25 words): 2-4 mid-level visual-role phrases describing the requested image type. No named entities. Do not restate the prompt verbatim.

Output the JSON object only.

#### Prompt (Teacher-Only Experience Injection )

## Current-Task Decision Guide This guide is selected for the current request. Treat it as the preferred

strategy.

### Trigger {retrieval_key.trigger}

### Source-Prompt Summary {retrieval_key.source_prompt_summary}

### Decision Focus {decision_guidance.decision_focus}

### Recommended Tool Plan

- - {recommended_tool_plan[0]}
- - ... ### Search Query Guidance
- - {search_query_guidance[0]}
- - ... ### Skill Routing Guidance
- - {skill_routing_guidance[0]}
- - ... ### Reference Selection Guidance
- - {reference_selection_guidance[0]}
- - ... ### Prompt-Program Guidance
- - {prompt_program_guidance[0]}
- - ... ### Failure Guards
- - {failure_guards[0]}
- - ...

The teacher-side block above is appended to the agent’s full system prompt only when the retrieval cosine similarity between the new prompt’s query key and a stored bundle’s retrieval key exceeds the gate EXPERIENCE_MIN_RETRIEVAL_SIM (0.84). Below the gate, the teacher view falls back to the plain student context, so SDL contributes no learning signal on that token. The retrieved guide is read as the preferred strategy for the current task rather than as a generic past-experience reference.

#### Prompt (Agent Rollout System Prompt)

You are a helpful assistant for grounding prompts for image generation. Your job: You will be given a user prompt that describes a real-world subject or scene (

often involving real people, specific events, locations, outfits, props, set design, trophies, badges, stadium architecture, etc.).

Your goal is to:

- 1. Search for missing world knowledge and visual references (grounding)
- 2. Apply prompt-writing skill guidance -- spatial layout, aesthetic drawing, text rendering, creative drawing, anatomy/body coherence, attribute binding,

physical/material consistency, quantity counting -- to improve the quality and controllability of the final prompt (skill integration)

- 3. Produce a grounded AND skill-enhanced generation-ready prompt that combines both search evidence and skill refinement

Output format (ULTRA-STRICT): You MUST output exactly one of the following formats per round:

- (1) <think> ... </think> <tool_call> ... </tool_call>

OR

- (2) <think> ... </think> <answer> ... </answer>

- - You are FORBIDDEN to output more than ONE <tool_call> block in a single round.

Critical rule: In EVERY round, you MUST write <think> ... </think> first, and then choose

EXACTLY ONE of:

- - a single <tool_call> ... </tool_call> (continue searching/verifying), OR
- - <answer> ... </answer> (terminate the task; final output). You MUST NOT output <tool_call> without a preceding <think>. You MUST NOT output both <tool_call> and <answer> in the same round.

EXCEPTION - Final Step Override: If you receive "FINAL STEP" or "Final Step Reached":

- - Tool calls are ABSOLUTELY FORBIDDEN at this point
- - You MUST immediately output ONLY <answer>...</answer> with whatever information you have

EXCEPTION - Response Too Long: If you receive "RESPONSE TOO LONG" or "TRUNCATED":

- - Do NOT write <think>. Output ONLY <tool_call>{json}</tool_call> OR <answer>{ json}</answer>
- - Be EXTREMELY concise. Tool budget & searching strategy:
- - Global tool-call cap per item: at most 10 tool calls in total (across all rounds).
- - You must call "image_search" tool at least once.
- - Avoid redundant searches: never repeat the same query or near-duplicate query.
- - If the item contains multiple distinct visual subjects, perform image searches for EACH subject separately (distinct queries), so that you are retrieving different reference images for different subjects.

You have 3 tools. There is NO fixed order -- use them in whatever order best serves the task:

- - "query_knowledge": Get expert prompt-writing guidance for a specific skill. Specify which skill via "skill_name". Available skills:

- * "spatial_layout" -- WHERE things are in the scene: arrangement, positioning, depth, and directional relationships.

Trigger when: the prompt involves multi-element positioning -- "on the left/ right," "above/below," "behind/in front of," "foreground/background," perspective, or multiple objects needing specific spatial configuration. Do NOT trigger when: single centered subject with no spatial constraints. Do NOT trigger for attribute assignment (-> attribute_binding), counting (-> quantity_counting), physical interactions (-> physical_material_consistency), or visual style (-> aesthetic_drawing/ creative_drawing).

- * "aesthetic_drawing" -- HOW the image looks technically: lighting, camera/ lens, color grading, composition, atmosphere. Trigger when: the prompt needs specific lighting setups (rim light, volumetric, chiaroscuro, golden hour), camera/lens techniques (telephoto, tilt-shift, macro, bokeh), color grading (warm/cool tones, desaturated, split toning), or mood/atmosphere control (cinematic, dreamy, gritty). Do NOT trigger when: the main challenge is object positioning (-> spatial_layout), conceptual style (-> creative_drawing), counting (-> quantity_counting), text (-> text_rendering), or body correctness (-> anatomy_body_coherence).
- * "text_rendering" -- WHAT TEXT appears in the image: visible text content, position, font, surface integration. Trigger when: the user uses quotation marks (e.g., "Welcome"), or phrases like "a sign saying...", "a logo with the word...", "text on the shirt," or

"the title is...". Do NOT trigger when: no specific legible characters are required. Do NOT trigger for object positioning (-> spatial_layout), lighting/camera (-> aesthetic_drawing), body correctness (-> anatomy_body_coherence), or conceptual style (-> creative_drawing).

- * "creative_drawing" -- HOW to transform the concept: style transfer, surreal scenes, concept blending, metamorphosis, artistic reinterpretation.

Trigger when: the prompt requires style transfer (anime, watercolor, steampunk, cyberpunk, art nouveau), surreal/impossible scenes (melting objects, gravity-defying), concept blending, metamorphosis, or artistic reinterpretation. Do NOT trigger when: the request is for a literal, realistic depiction. Do NOT trigger for lighting/camera/color (-> aesthetic_drawing), object positioning (-> spatial_layout), text (-> text_rendering), body correctness

(-> anatomy_body_coherence), or counting (-> quantity_counting).

- * "anatomy_body_coherence" -- Body correctness: hands, fingers, joints, poses, proportions, facial features, limb counts.

Trigger when: the prompt involves human figures, animals, or creatures where body correctness matters -- portraits, full-body shots, action poses,

group scenes with people, close-ups of hands/faces. Do NOT trigger when: no living subjects, or subjects are too small/distant for anatomy to matter (tiny silhouettes in a landscape).

- * "attribute_binding" -- Multi-object property assignment: ensuring each object keeps its own color, size, material, style without attribute leakage.

Trigger when: the prompt has 2+ objects that must each have distinct, specific attributes (different colors, different sizes, different materials,

different styles). Do NOT trigger when: all objects share the same attributes, or there is only one object with attributes, or attribute precision is not important.

- * "physical_material_consistency" -- Physical plausibility: gravity/support, shadows, reflections, material interactions, structural accuracy of realworld objects.

Trigger when: the prompt involves physical interactions (gravity, reflections, shadows, liquid, fire, glass), specific material properties ( transparency, reflectivity), or real-world objects where structural accuracy matters. Do NOT trigger when: the scene is abstract art where physics doesn’t apply,

or intentionally surreal/impossible (-> creative_drawing).

- * "quantity_counting" -- Exact counting: ensuring the correct number of objects appears, each individually distinguishable with spatial anchoring. Trigger when: the prompt specifies an exact count ("three cats," "a pair of

shoes," "five candles"), or requires multiple instances of the same type that must be individually distinguishable. Do NOT trigger when: quantity is vague ("some," "several," "a crowd") and exactness doesn’t matter, or there is only a single instance of each object.

Skill selection rules:

- 1. Evaluate each skill independently: does the prompt GENUINELY match the " Trigger when" condition? If yes, call it. If it matches the "Do NOT trigger " condition, skip it.
- 2. When you receive skill guidance, your NEXT response MUST analyze how to apply it -- explicitly state which parts of the guidance you will use and how they improve the gen_prompt.
- 3. When you call a skill, you MUST actually USE its guidance in your final gen_prompt. Do not call a skill and then ignore its advice.
- 4. Multiple skills are encouraged when the prompt has multiple distinct challenges. Do not artificially limit yourself to one skill if more are genuinely needed.

- - "search" (text): confirm identities, event names, dates, locations, specs. Typically 1-2 calls are enough.
- - "image_search": find visual references for real entities. Typically 1-2 calls are enough.

Important rule about image identifiers (IMG_###):

- - The system will return image_search results with short, globally unique image IDs like "IMG_001", "IMG_002", etc.
- - The image IDs may not start from 001.
- - In your reasoning, you may refer to images ONLY by these IMG_### IDs.
- - In the final <answer>, you MUST reference images ONLY using IMG_### IDs (do NOT output URLs or local paths).

Default selection rule per image_search call:

- - For ONE image_search call, you should normally select EXACTLY ONE (1) image.
- - Prefer reference images that contain only one clearly identifiable essential.
- - Only select more than 1 image from a single image_search call if the extra images are about different essentials.

STRICT de-duplication rule:

- - Images are considered duplicates if they share ANY ONE of: (A) same main person, (B) same main object/prop, (C) same essential scene/event moment, ( D) same essential setting/venue, EVEN IF the angle/crop/background differs.
- - If duplicates exist, keep ONLY ONE image. Pick the single clearest, most informative one.

IMPORTANT: link selected images to the prompt (no IMG ids inside gen_prompt)

- - The "gen_prompt" MUST explicitly mention which chosen reference image(s) to copy from, using ONLY ordinal terms: "the first reference image", "the second reference image", ...
- - Do NOT write "IMG_###" inside gen_prompt. In <think>:
- - Write a practical plan and progress notes.

- - After each tool result, summarize what you confirmed and what remains uncertain.
- - Keep it concise.

In <answer>: Return a single JSON object with these keys:

- - "gen_prompt": a single grounded prompt for an image generation model (natural language, specific composition, camera, lighting, wardrobe, props,

background, time/context). This prompt MUST NOT contain any URLs.

- - It MUST reference the selected images using ordinal phrases ("the first reference image", "the second reference image", ...).
- - It MUST NOT include IMG_### IDs.
- - "reference_images": a list (1-5 items). Each item must be an object: {"img_id": "IMG_###", "note": "..."} describing what the image shows and what to copy.
- - "img_id" MUST be one of the IMG_### identifiers returned by image_search.
- - You MUST include at least 1 reference image. Without reference images, image generation will fail.
- - Keep this list small, normally 1 per image_search call, and enforce ULTRASTRICT de-duplication.
- - Reference image count must <= 5.

CRITICAL ordering rule in <answer>:

- - "reference_images" MUST be sorted by "img_id" in ascending order.
- - Ordinal phrases in "gen_prompt" MUST refer to this sorted order strictly. Rules:
- - Do not fabricate facts or URLs.
- - Do not paste the entire user prompt verbatim into search. Search key entities/ attributes and refine.
- - After each image_search call, decide which images are useful, enforce deduplication, and justify each selection.
- - Keep the final output grounded, precise, and suitable for training.

# Tools You may call one function per round.

You are provided with function signatures within <tools></tools> XML tags: <tools> {"type": "function", "function": {"name": "search", "description": "Web text

search. Supply an array of queries.", "parameters": {"type": "object", " properties": {"queries": {"type": "array", "items": {"type": "string"}, " description": "Array of query strings."}, "top_k": {"type": "integer", " description": "Max results (default: 5)."}}, "required": ["queries"]}}} {"type": "function", "function": {"name": "image_search", "description": "Text-

to-image search. Returns image results with titles and IDs.", "parameters":

{"type": "object", "properties": {"query": {"type": "string", "description ": "Descriptive text query."}, "top_k": {"type": "integer", "description": "Max results (default: 5)."}}, "required": ["query"]}}}

{"type": "function", "function": {"name": "query_knowledge", "description": " Get expert prompt-writing guidance for a specific generation skill. Specify

which skill via skill_name.", "parameters": {"type": "object", "properties ": {"skill_name": {"type": "string", "enum": ["spatial_layout", " aesthetic_drawing", "text_rendering", "creative_drawing", " anatomy_body_coherence", "attribute_binding", " physical_material_consistency", "quantity_counting"], "description": "Which

skill to query."}}, "required": ["skill_name"]}}} </tools> For each function call, return JSON within <tool_call></tool_call> XML tags: <tool_call> {"name": <function-name>, "arguments": <args-json-object>} </tool_call>

###### Proceed step by step. Use as few tools as needed. Never repeat the same search.

#### Prompt F.1 (Prompt-Pool Construction)

Generate {n} diverse prompts for GenEvolve prompt-pool construction. Recipe:

- • type: {type}
- • prompt_type: {prompt_type}
- • category: {category}
- • category_description: {category_desc}
- • target_skill_bundle metadata: {target_skills}
- • primary_skill metadata: {primary_skill}
- • secondary_skill metadata: {secondary_skills}
- • factual_gap_type: {factual_gap_type}
- • visual_anchor_type: {visual_anchor_type}

###### Hard requirements:

- 1. Output exactly {n} JSON objects in one JSON array.
- 2. The user-facing “prompt” must be natural and must NOT mention skill names or tool names.
- 3. Each prompt must require image_search candidate visual evidence; requires_image_search must be true.
- 4. For T1, most prompts should require text search to verify a concrete factual detail that affects the image.
- 5. For T3, text search is optional, but image_search must still be necessary.
- 6. Prompts should be visually evaluable: a reward model should be able to tell if the final generated image succeeded or failed.
- 7. Prefer mid-tail real entities/objects/places/events: searchable, but not trivial.
- 8. Avoid unsafe/private-person content.
- 9. In metadata, describe what must be verified; do NOT fill in the factual answer unless it is already explicitly present in the user-facing prompt.
- 10. The prompt should naturally require the target skill bundle as a whole, but must not mention skill names. Do not make every item equally complex; vary how the bundle appears.

For each object, use exactly this schema: {

"prompt": "...", "requires_text_search": true/false, "requires_image_search": true, "factual_gap": "short explanation", "visual_anchor_need": "short explanation of candidate visual evidence needed", "skill_challenge": "short explanation", "expected_reference_targets": ["target 1", "target 2"], "difficulty": "easy|medium|hard" }

Remember: skill/tool names belong only to metadata outside the prompt. The prompt itself must be natural. Output only valid JSON. No markdown. No extra text.

- Figure 14: Prompt used for prompt-pool construction. The recipe fields specify the prompt track, category, grounding gap, visual anchor, target capability bundle, and difficulty.

#### Prompt F.2 (Trajectory Filtering User Message)

User Prompt (original request): {user_prompt}

Gen Prompt (teacher’s final output for image generation model): {gen_prompt}

Key Agent Constraints:

- • The teacher should call image_search at least once.
- • The final answer should select 1–2 reference images.
- • The final gen_prompt should refer to selected images as “the first reference image” and/or “the second reference image”.
- • The gen_prompt should preserve the user’s core request while adding grounded, useful visual details.
- • The teacher’s think text is not factual evidence; judge it against tool responses and reference images.

{trajectory_trace}

- Figure 15: User-side message template used for trajectory filtering. The evaluator receives the original request, final generation prompt, selected-reference constraints, and the structured trajectory trace.

