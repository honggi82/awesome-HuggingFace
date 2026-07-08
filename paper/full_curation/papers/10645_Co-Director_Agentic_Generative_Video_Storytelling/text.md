## arXiv:2604.24842v1[cs.AI]27Apr2026

[Figure 1]

[Figure 2]

# Co-Director: Agentic Generative Video Storytelling

##### Yale Song1 *, Yiwen Song1, Nick Losier1, Nathan Hodson1, Ye Jin1, Rhyard Zhu1, Yan Xu1, Daniel Vlasic1, Carina Claassen1, Jasmine Leon1, Khanh G. LeViet1, Zack Chomyn1, Joe Timmons1, Brett Slatkin1, Scott Penberthy1 and Tomas Pfister1

1Google Inc.

While diffusion models generate high-fidelity video clips, transforming them into coherent storytelling engines remains challenging. Current agentic pipelines automate this via chained modules but suffer from semantic drift and cascading failures due to independent, handcrafted prompting. We present Co-Director, a hierarchical multi-agent framework formalizing video storytelling as a global optimization problem. To ensure semantic coherence, we introduce hierarchical parameterization: a multi-armed bandit globally identifies promising creative directions, while a local multimodal self-refinement loop mitigates identity drift and ensures sequence-level consistency. This balances the exploration of novel narrative strategies with the exploitation of effective creative configurations. For evaluation, we introduce GenAd-Bench, a 400-scenario dataset of fictional products for personalized advertising. Experiments demonstrate that Co-Director significantly outperforms state-of-the-art baselines, offering a principled approach that seamlessly generalizes to broader cinematic narratives.

Project Page: https://co-director-agent.github.io/

### 1. Introduction

Generative video storytelling aims to transform abstract ideas—such as a simple tagline and sparse reference visuals—into cohesive video narratives, like a fully realized short film. It has diverse applications, ranging from democratizing high-end content creation (Divya and Mirza, 2024) to automating filmmaking (Zhang et al., 2025) and digital advertising (Baek, 2023). Realizing this potential requires jointly optimizing multiple objectives: narrative coherence, visual consistency, and mise-en-scène—the visual language dictating how camera angles, color palettes, and metaphors deliver the intended message (Zettl, 1973).

Recent advancements in video diffusion demonstrate remarkable high-fidelity generation. Highprofile collaborations, such as Google DeepMind’s work with Darren Aronofsky on Ancestra (Aronofsky and McNitt, 2025) and On This Day... 1776 (Aronofsky, 2026), showcase the artistic ceiling when backed by a 200-person creative team. While representing the pinnacle of human-AI collaboration, such resources remain out of reach for independent creators. Without them, individuals must manually bridge the gap between LLM scripting and fragmented generative models, correcting for the consistency errors and identity drift inherent in current architectures (Elmoghany et al., 2025).

To mitigate this, the community has pivoted toward agentic pipelines where LLMs orchestrate the multi-step generative process (Huang et al., 2025; Lin et al., 2023; Shi et al., 2025; Wei et al., 2025; Wu et al., 2025b). However, these systems remain susceptible to “cascading failures,” where early errors propagate and break long-horizon consistency (Zhou et al., 2024). This reflects the classical credit assignment problem (Minsky, 1961), as terminal failures are difficult to trace back to specific prompts. Furthermore, existing systems rely on static, handcrafted prompt templates lacking

Corresponding author(s): yalesong@google.com

a unified creative vision. Without a mechanism to map script-level changes to downstream generation, the resulting visual output often feels disjointed. Crucially, these approaches fail to capture the serendipity essential to creative production. Human directors often explore counter-intuitive ideas to find “creative gold.” Static, chained pipelines are fundamentally ill-equipped for such exploration, remaining anchored to fixed templates unable to adapt to the non-linear nature of creative success.

In this work, we present Co-Director, a multi-agent framework for automating video storytelling. To the best of our knowledge, Co-Director is the first system to formalize generative storytelling as a global optimization problem. Unlike prior agentic pipelines that rely on linear, "waterfall" chains prone to semantic drift, we introduce a hierarchical parameterization strategy steered by Multi-Armed Bandit (MAB) (Bubeck and Cesa-Bianchi, 2012). Rather than relying on fixed prompt templates, Co-Director utilizes MAB to sample abstract creative trajectories and dynamically injects these into the system prompts of sub-agents. This top-down steering ensures that all-agents—from script-writing to video generation—are orthogonally aligned toward a unified creative intent. This formulation formalizes the creative process as a search for the optimal balance between exploration—venturing into novel, unconventional narrative territories—and exploitation—refining historically effective stylistic configurations. This allows Co-Director to move beyond simple self-correction toward a principled exploration of the creative latent space, identifying the configuration that optimally balances visual fidelity with narrative effectiveness (Escalas, 2004).

Furthermore, we argue that evaluating video storytelling requires a shift from open-ended cinematic metrics towards rigorous, constraint-driven frameworks. We use digital advertising as our demonstration scenario because it demands both sophisticated storytelling (much like short cinematic film) and strict adherence to objective requirements: asset preservation, audience alignment, and value proposition communication (Teixeira et al., 2012). To this end, we introduce GenAd-Bench, a dataset of 400 scenarios across 200 fictional products. By utilizing fictional entities, we respect copyright constraints while preventing models from defaulting to memorized training priors. We provide a multimodal LLM (MLLM)-based evaluation suite covering Visual Asset Fidelity, Demographic Alignment, Visual Quality, and Marketing Appeal—anchored in the AIDA hierarchy (Barry and Howard, 1990; Strong, 1925).

Our primary contributions are summarized as follows:

- • We propose Co-Director, a multi-agent architecture for video storytelling that achieves global coherence by integrating multi-armed bandit optimization with a local, MLLM feedback-driven refinement loop.
- • We introduce the hierarchical parameterization of creative intent, enabling the consistent propagation of visual and narrative styles across discrete generative sub-agents.
- • We present GenAd-Bench to rigorously evaluate generative video storytelling across 400 scenarios, eliminating memorization bias via the use of fictional brand assets.
- • We demonstrate our approach significantly outperforming existing foundation models and agentic baselines in asset preservation, contextual alignment, and visual quality.
- • We open-source our implementation of Co-Director and the GenAd-Bench dataset to facilitate future research in generative video storytelling.

### 2. Related Work

Foundation Models for Video Synthesis While large-scale diffusion models (ByteDance, 2026; Google, 2026d; Kuaishou, 2026; Wan et al., 2025) achieve unprecedented cinematic realism, maintaining long-range consistency remains challenging (Elmoghany et al., 2025). Specialized architectures

Input Multi-Armed Bandit (MAB)

Orchestrator Agent (Φorch )

Factored Reward Signal

|Warm Start|[Figure 5]<br><br>Creative Strategy Narrative Mode Aesthetic Archetype<br><br>Informational<br><br>Transformational<br><br>Comparative Clarity/ Energy<br><br>Kinetic Grit<br><br>Cinematic Premium<br><br>[Figure 6]<br><br>Analy -tical<br><br>Vigne -tte<br><br>Drama|
|---|---|
|θ| |

User Prompt (puser)

[Figure 7]

Structuredprompt(puser)

R = (rcs, rnm, raa)

𠆢

|[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>Ref. Visuals (Iref)<br><br>Annotatedref.visuals(Iref)<br><br>Creative configuration (θ)<br><br>Optimi Lo<br><br>𠆢|zatio op|
|---|---|
| | |

n

(θcs ) (θnm ) (θaa )

Post-prod. Agent (Φpost )

Pre-production Agent (Φpre ) Production Agent (Φprod )

MLLM Judge

| | |
|---|---|
| | |

Keyframe Agent (Φframe )

Keyframes (K )

Creative Brief (B )

Visual Assets (V )

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Video Agent (Φvideo )

[Figure 16]

[Figure 17]

Clips (C )

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Storyline (L )

Storyboard (S )

Audio Agent (Φaudio )

Audio (A )

Video Editing Final Video (F )

- Figure 1 | Co-Director multi-agent pipeline: (Top): The Orchestrator Agent (Φ𝑜𝑟𝑐ℎ) utilizes MAB to navigate a factored creative action space 𝜃 = (𝜃𝑐𝑠, 𝜃𝑛𝑚, 𝜃𝑎𝑎). (Bottom): The Pre-Production Agent (Φ𝑝𝑟𝑒) synthesizes a creative brief (B), storyline (L), and visual assets (V) to establish a consistent storyboard (S), which the Production Agent (Φ𝑝𝑟𝑜𝑑) translates into synchronized keyframes (K), video clips (C), and audio (A). An MLLM Judge evaluates the final video (𝐹) to generate a factored

reward signal 𝑅 = (𝑟𝑐𝑠, 𝑟𝑛𝑚, 𝑟𝑎𝑎), which is propagated back to the MAB to iteratively refine the creative configuration over 𝑇 optimization loops.

like Phantom (Liu et al., 2025) and MoC (Cai et al., 2025) introduce advanced techniques to maintain character consistency, but they remain fundamentally standalone generators. Furthermore, these models serve as powerful building blocks rather than complete end-to-end solutions, lacking the high-level planning and reasoning required to create complex narratives in the resulting video output (Elmoghany et al., 2025).

Agentic Frameworks for Video Storytelling To bridge the gap between clip generation and narrative production, recent multi-agent systems utilize LLMs to decompose scripts into shot lists and prompts (Lin et al., 2023; Wu et al., 2025b), targeting domains like personalized vlogs (Hou et al.,

- 2025), music videos (Tang et al., 2025), and short films (Mu et al., 2026; Wei et al., 2025; Xiao et al., 2025; Zhao et al., 2024). However, these “waterfall” pipelines are susceptible to cascading errors. While frameworks like AniMaker (Shi et al., 2025) and MAViS (Wang et al., 2026) incorporate local consistency checks, they struggle with global credit assignment (Minsky, 1961) across the agentic stack. To resolve this, Co-Director moves beyond the local-refinement paradigm by introducing a Multi-Armed Bandit (MAB) formulation (Bubeck and Cesa-Bianchi, 2012; Lai and Robbins, 1985) to perform a principled exploration-exploitation of the creative latent space. This global, hierarchical parameterization resolves the “semantic collisions” between sub-agents that plague current pipelines, preventing identity drift and ensuring horizontal cohesiveness.

Benchmarks for Video Storytelling As generative capabilities evolve, evaluation methodologies must adapt to measure higher-order reasoning. Generic video benchmarks (Huang et al., 2024; Sun et al., 2025) focus on prompt alignment and broad aesthetics, lacking the visual and narrative qualities demanded by professional workflows. Furthermore, even recent narrative-focused benchmarks, such as MovieBench (Wu et al., 2025a) and ViStoryBench (Zhuang et al., 2025), strictly evaluate isolated generative sub-tasks using meticulously pre-defined scripts. Consequently, current literature lacks evaluations for true end-to-end storytelling: the complex process of autonomously transforming a simple, abstract idea into a cohesive video narrative. Our work addresses this gap by introducing GenAd-Bench designed specifically for end-to-end generative workflows.

### 3. Approach

#### 3.1. Hierarchical Multi-Agent Architecture

We formulate Co-Director as a hierarchical multi-agent system orchestrating generative video workflows. While the framework is domain-agnostic, we instantiate it to solve video advertising, which comes with strict narrative and visual constraints.

Orchestrator Agent (Φ𝑜𝑟𝑐ℎ) This agent manages the global state and execution flow of all sub-agents. Before downstream generation, it performs multimodal data ingestion: 1) parsing the user prompt 𝑝𝑢𝑠𝑒𝑟 into structured constraints ˆ𝑝𝑢𝑠𝑒𝑟 (brand, product, and demographics (Baardman et al., 2021)); 2) annotating reference visuals I𝑟𝑒𝑓 with captions and semantic roles (e.g., product, logo, protagonist), yielding Iˆ𝑟𝑒𝑓. Post-initialization, Φ𝑜𝑟𝑐ℎ drives a global optimization loop for 𝑇 iterations. At iteration 𝑡, it interfaces with a multi-armed bandit (Section 3.3) to sample a creative configuration 𝜃(𝑡), forming a global conditioning context Θ(𝑡) = (ˆ𝑝𝑢𝑠𝑒𝑟, Iˆ𝑟𝑒𝑓, 𝜃(𝑡)),1 which is then propagated downstream.

Pre-Production Agent (Φ𝑝𝑟𝑒) This agent transforms the structured constraints Θ into a descriptive storyboard S through four sequential modules:

- • Creative Brief (B): Acting as the “deep research” phase, this module performs contextual enrich-

ment, expanding product/demographic constraints within ˆ𝑝𝑢𝑠𝑒𝑟 into localized cultural resonance and environmental context. This becomes a creative brief B.

- • Storyline (L): Conditioned on B and 𝜃, this module constructs the storyline L that acts as a scene-by-scene script establishing plot, character actions, pacing, and transitions.
- • Visual Assets (V): This module generates visual assets V for entities in L (e.g., characters,

props, environments) only if they are absent from the reference visuals Iˆ𝑟𝑒𝑓 ∈ Θ. For characters, it synthesizes context-aligned wardrobes dictated by L, compositing the base character and garments into multi-angle reference collages.

- • Storyboard (S): Expands L into a storyboard S = {𝑠0, . . . , 𝑠𝑁−1} modulated by 𝜃. Each scene 𝑠𝑖 ∈ S defines semantic descriptors (actions, environment), camera kinematics (scale, movements), temporal constraints, and entity presence flags linked to V. It also formulates global audio directives (voiceover, tempo, mood), inspired by consumer psychology literature (Alpert and Alpert, 1990; Gorn, 1982).

Production Agent (Φ𝑝𝑟𝑜𝑑) This agent translates the storyboard S into multimodal artifacts via three specialized sub-agents:

- • Keyframe Agent (Φ𝑓𝑟𝑎𝑚𝑒): Direct text-to-video generation is prone to identity drift (Elmoghany et al., 2025). To mitigate this, Φ𝑓𝑟𝑎𝑚𝑒 establishes a visual prior by generating first-frame images to condition downstream synthesis, modulated by aesthetic constraints in 𝜃 (Section 3.3). Formally,

for each scene, it processes descriptors 𝑠𝑖, assigned assets V, and configuration 𝜃 to output the keyframe set K:

K = {𝑘0, 𝑘1, . . . , 𝑘𝑁−1} where 𝑘𝑖 = Φ𝑓𝑟𝑎𝑚𝑒(𝑠𝑖, V, 𝜃) (1)

- • Video Agent (Φ𝑣𝑖𝑑𝑒𝑜): This sub-agent transforms K into a sequence of video clips C. For each scene, it conditions a video diffusion model on the initial keyframe 𝑘𝑖, parameterized by local descriptors 𝑠𝑖 and creative configuration 𝜃 to dictate temporal dynamics like pacing and camera kinematics (Section 3.3). Formally, this yields the clip set C:

C = {𝑐0, 𝑐1, . . . , 𝑐𝑁−1} where 𝑐𝑖 = Φ𝑣𝑖𝑑𝑒𝑜(𝑠𝑖, 𝑘𝑖, 𝜃) (2)

1We omit superscript (𝑡) hereafter for clarity.

- • Audio Agent (Φ𝑎𝑢𝑑𝑖𝑜): This sub-agent synthesizes the acoustic elements specified in S. Rather than per-scene generation, it processes global voiceover and background music from the planning phase.

Acoustic properties (e.g., tempo, genre, mood) are conditioned on 𝜃. Φ𝑎𝑢𝑑𝑖𝑜 employs text-to-speech and text-to-audio models to generate and mix these discrete elements into a unified global track A. Formally:

A = Φ𝑎𝑢𝑑𝑖𝑜(S, 𝜃) (3)

Post-Production Agent (Φ𝑝𝑜𝑠𝑡) This module consolidates clips C and audio A into the final video 𝐹. It concatenates C based on temporal constraints defined in storyboard S, then multiplexes A onto the sequence to synchronize visual pacing with acoustic cues.

𝐹 = Φ𝑝𝑜𝑠𝑡(S, C, A) (4)

- 3.2. Local Optimization via Agentic Self-Refinement

In sequential generative pipelines, early errors propagate and amplify downstream. To prevent this, Co-Director implements feedback descent (Lee et al., 2025; Yuksekgonul et al., 2024) to self-refine intermediate artifacts. LLMs and MLLMs iteratively evaluate and refine artifacts based on textual feedback. This loop terminates upon reaching a score threshold or iteration budget, returning the best attempt. We apply this local optimization to storyline L and keyframes K, reserving final video evaluation for a global reward signal (Section 3.3)

Storyline Before Φ𝑝𝑟𝑒 generates storyboard S, an LLM assesses storyline L on hook quality, narrative cohesion, product integration, emotional engagement, and prompt adherence. It outputs a score with textual feedback and rewrites the prompt to emphasize areas needing improvement. If the score falls below threshold 𝜏L, the agent regenerates L using this revised prompt, ensuring a coherent narrative foundation before expansion into S.

Keyframe An MLLM evaluates the keyframe sequence K = {𝑘0, . . . , 𝑘𝑁−1} jointly rather than individually. This joint contextualization evaluates sequence-level consistency for character and product identities, alongside environmental continuity. Comparing K against ˆ𝑝𝑢𝑠𝑒𝑟 and V, it scores visual consistency, narrative flow, product appeal, and prompt adherence. If the aggregate score falls below threshold 𝜏K, the MLLM isolates issues and indexes problematic keyframes in K. It then provides actionable feedback and refined prompts, allowing Φ𝑓𝑟𝑎𝑚𝑒 to selectively regenerate flagged keyframes while preserving accepted ones.

- 3.3. Global Optimization via Multi-Armed Bandits

Because the chained pipeline is non-differentiable, standard gradient updates cannot achieve end-toend optimization. Instead, Co-Director uses hierarchical parameterization, treating the pipeline as a black box steered by centralized parameters governing global creative directions. We formulate tuning these parameters as a Multi-Armed Bandit (MAB) problem (Bubeck and Cesa-Bianchi, 2012; Lai and Robbins, 1985). Operating at this top level, rather than tuning individual prompts independently, ensures semantic coherence across sub-agents while balancing the exploration of novel strategies with the exploitation of proven configurations.

MAB Preliminaries In a standard MAB framework, an agent iteratively selects an action (an “arm”), observes a reward, and updates its policy to maximize long-term payout. The central challenge is the exploration-exploitation tradeoff: balancing testing untried arms for potentially higher rewards against exploiting the best-known option. In Co-Director, a single “pull” corresponds to a full pipeline execution, the “arms” are creative directions 𝜃, and the “reward” is the terminal video evaluation score.

Hierarchical Parameterization We define the action space (the “arms”) over three orthogonal axes of video production grounded in marketing and media theory. The creative direction 𝜃 is a tuple comprising three categorical variables: 𝜃 = (𝜃𝑐𝑠, 𝜃𝑛𝑚, 𝜃𝑎𝑎).

- • Creative Strategy (𝜃𝑐𝑠): This controls what the ad is fundamentally trying to communicate (Laskey et al., 1989). The bandit selects among an Informational strategy (focusing on functional utility), a Transformational strategy (driving psychological and lifestyle appeal), or a Comparative strategy (positioning against a known standard or competitor).
- • Narrative Mode (𝜃𝑛𝑚): This controls how the strategy is delivered over time (Escalas, 2004). The bandit selects among an Analytical mode (an argument-based delivery without a story arc), a Vignette mode (atmospheric slices-of-life prioritizing mood over plot), or a Narrative Drama mode (a character-driven sequence of conflict and resolution).
- • Aesthetic Archetype (𝜃𝑎𝑎): This controls the sensory realization of the video (Zettl, 1973) via four configurations: Clarity/Energy (high-key lighting, fast-cut motion, high-tempo audio); Cinematic Premium (chiaroscuro lighting, slow reflective camera work, orchestral audio); Minimalist Focus (bright backgrounds, micro-movements, ASMR-style foley); and Kinetic Grit (low-key lighting, handheld/FPV drone motion, electronic synth audio).

At execution step 𝑡, the MAB selects an arm combination 𝜃(𝑡). While downstream LLMs can interpret these creative parameters, independent interpretation risks semantic drift. To ensure coherence, orchestrator Φ𝑜𝑟𝑐ℎ centrally synthesizes 𝜃(𝑡) into agent-specific instructions, dynamically injecting them into placeholders within the agents’ system prompts.

Reward To evaluate 𝜃(𝑡), Co-Director uses an MLLM evaluator processing the final video 𝐹(𝑡) alongside 𝑝𝑢𝑠𝑒𝑟 and I𝑟𝑒𝑓 to yield two metrics (full prompt in Appendix K). First, an aggregate quality score (temporal coherence, visual quality, consistency) selects the best video across 𝑇 executions. Second, it computes dimension-specific strategic efficacy scores crucial for MAB optimization. This decouples execution from strategy, assessing if the creative direction was effective independent of downstream diffusion artifacts. This yields a factored reward signal 𝑅(𝑡) = (𝑟𝑐𝑠, 𝑟𝑛𝑚, 𝑟𝑎𝑎) ∈ [0, 100]3 to update the bandit’s policy.

Optimization Policy and Warm Start Co-Director models the search using the Upper Confidence Bound (UCB1) algorithm (Auer et al., 2002), which uses the disentangled scalars from 𝑅(𝑡) to independently update the expected value of each selected arm. By decoupling the reward updates, the MAB analytically bypasses the credit assignment problem (Minsky, 1961), isolating exactly which creative strategies drove the video’s success or failure. Furthermore, because unguided exploration is prohibitively expensive in video generation, we introduce an LLM-driven warm start. Prior to the first execution, an LLM analyzes the user’s demographic and product domain to initialize the expected values of the arms, safely biasing early exploration toward theoretically sound configurations.

### 4. GenAd-Bench: Evaluating Generative Video Advertising

To construct GenAd-Bench, we developed a human-in-the-loop pipeline using Gemini 3 Pro (Google,

- 2026b) for text synthesis and Nano Banana Pro (Google, 2026c) for visual assets. We generated 50 fictional brands with four products each, spanning consumer goods to industrial equipment. For each product, we authored two contrasting demographics—stereotypical and unconventional—yielding 400 unique scenarios defined by a six-point prompt (brand, product, gender, age, location, interest). Accompanying visual assets (logos, product references) were generated and manually verified (15% required regeneration) to ensure uniqueness and prevent visual memorization or copyright conflicts.

#### The resulting benchmark enforces strict targeting constraints and multimodal conditioning,

requiring models to synthesize textual demographic intent with specific reference imagery. By evaluating paired scenarios across a global geographic distribution, GenAd-Bench explicitly tests conceptual flexibility and exposure to ingrained biases. Furthermore, by utilizing fictional entities, the benchmark prevents models from defaulting to memorized training priors, ensuring that performance is a measure of reasoning rather than retrieval. We partition the data into half (200 scenarios each) for hillclimbing and validation.

We use Gemini 3 Pro (Google, 2026b) as an MLLM-as-a-Judge to evaluate generated videos; see Appendix L for the rubric prompts. The MLLM simultaneously processes visual frames and the audio track to score four dimensions (0–100):

- • Visual Asset Fidelity (VAF): Measures faithfulness to user-provided logos and products. The MLLM assesses visual-semantic similarity, ensuring core identity preservation across complex camera angles, occlusions, and dynamic lighting.
- • Demographic Alignment (DA): Assesses targeting success across four variables (Gender, Age, Location, Interest). The MLLM penalizes generic aesthetics while rewarding tailored casting, localized environmental context, and narrative tone.
- • Marketing Appeal (MA): Evaluates advertisement persuasiveness via the AIDA hierarchy (Barry and Howard, 1990; Strong, 1925) and engagement theory (Teixeira et al., 2012). It scores visual hooks, value proposition clarity, and emotional resonance.
- • Visual Quality (VQ): A prompt-agnostic metric scanning for diffusion artifacts, object morphing, and temporal flickering. It ensures videos meet broadcast viability by penalizing unnatural motion or violations of basic physical laws.

While utilizing an MLLM for evaluation raises potential bias concerns, we benchmark these metrics against a human study (Section 5.2). We find that the MLLM serves as a reliable proxy, showing alignment that approaches the human-human agreement ceiling.

### 5. Experiments

We evaluate Co-Director on GenAd-Bench against state-of-the-art baselines, demonstrating its superior storytelling capabilities, followed by ablation studies validating the necessity of local self-refinement and global MAB optimization.

Implementation Details We instantiate Co-Director with Gemini 3 Pro (Google, 2026b), Nano Banana Pro (Google, 2026c), Veo 3.1 (Google, 2026d), Gemini 2.5 Pro TTS (Google, 2026a), and Lyria 2 (Google, 2025). MAB optimization is set to 𝑇 = 4 iterations (see App. F for search efficiency trade-offs), and local refinement is capped at 3 retries. Final videos feature 4 shots totaling 12 seconds.

Baselines We evaluate Co-Director against monolithic models, commercial platforms, and multiagent frameworks, listed in Table 1. To isolate architectural efficacy, we compare against analogous agentic pipelines AniMaker and MovieAgent using the same underlying models. Since these require scene-by-scene scripts, Gemini 3 Pro expands our six-point prompts into the necessary formats. Finally, we compare Co-Director against a Base Agentic Pipeline (no optimizations) and a Random Search Baseline (𝑇 = 4) that samples the creative space without MAB or warm-up. Baseline details are in Appendix E.

- Table 1 | Evaluation on GenAd-Bench. All metrics are scaled [0, 100]. VAF: Visual Asset Fidelity, DA: Demographic Alignment, MA: Marketing Appeal, VQ: Visual Quality.

Method VAF ↑ DA ↑ MA ↑ VQ ↑ Avg. ↑ Proprietary Models

Creatify (Creatify AI, 2026) 23.2 16.2 19.5 29.5 22.1 HeyGen (HeyGen, 2026) 42.9 59.3 39.5 45.0 46.7 Kling 3.0 Omni (Kuaishou, 2026) 62.0 70.3 56.0 45.3 58.4 Veo 3.1 (Google, 2026d) 60.0 80.8 63.2 50.5 63.6 Wan 2.6 (Wan et al., 2025) 67.0 71.5 62.5 58.9 65.0

Open-Source Models

LTX-2.3 (HaCohen et al., 2024) 23.5 56.0 30.6 24.4 33.6 AniMaker (Shi et al., 2025) 53.1 81.3 60.3 53.9 62.2 MovieAgent (Wu et al., 2025b) 61.2 81.3 66.4 52.4 65.3

Base Agentic Pipeline (𝑇 = 1) 68.5 78.4 67.1 59.9 68.5 Random Search Baseline (𝑇 = 4) 77.0 85.8 75.6 64.1 75.7 Co-Director (𝑇 = 4) 82.1 91.4 82.0 70.2 81.4

#### 5.1. Main Results

As reported in Table 1, Co-Director achieves the highest average performance (81.4). We include specialized commercial platforms (Creatify, HeyGen) to establish the current industry baseline; their reliance on "talking head" avatar synthesis with static overlays fundamentally limits their performance on complex narratives, highlighting the necessity of Co-Director’s dynamic storytelling. Crucially, the performance gap between Co-Director and prior agentic baselines (AniMaker, MovieAgent) stems strictly from agentic architectural differences. By utilizing the same LLM (Gemini 3 Pro) to generate their requisite input scripts, we ensured a highly competitive and equitable starting point. Furthermore, self-refinement (Section 3.2) and MAB optimization (Section 3.3) drive Co-Director’s superior Visual Quality (VQ: 70.2) over its base model, Veo 3.1 (VQ: 50.5). Although VQ is promptagnostic, Co-Director’s optimization proactively improves suboptimal clips, effectively elevating the foundation model’s baseline and ensuring superior temporal consistency and physical realism over raw, single-pass generation.

To isolate the impact of our orchestration and global optimization, we evaluate Co-Director against both external baselines and internal architectural variants. Using identical foundation models, our Base Agentic Pipeline (𝑇 = 1) averages 68.5. This proves our core orchestration alone significantly outperforms prior linear open-source pipelines like AniMaker (62.2) and MovieAgent (65.3), even before iterative refinement. Scaling to 𝑇 = 4 via Random Search (75.7), akin to Best-of-N (Snell

- et al., 2024), yields noticeable improvements. Crucially, replacing random sampling with our MAB formulation and warm-up (Co-Director, 𝑇 = 4) drives the average to 81.4, demonstrating that our ultimate performance gains stem from a combination of superior reasoning, planning, and principled global optimization.

As shown in Figure 2, comparing cumulative best performance reveals that Co-Director ’s Warm-Up + MAB provides an efficiency advantage, achieving high-quality results in the first iteration where Random Search fails to ground the creative intent. While Random Search shows a steadier upward trajectory over long horizons, this reflects the variance of undirected sampling; it eventually "discovers" strong configurations by chance, whereas the MAB rapidly isolates an optimal plateau. This suggests the system effectively converges on a strategically sound local optimum early, avoiding the prohibitive computational costs of the exhaustive search required for Random Search to reach

###### Visual Asset Fidelity (VAF)

###### Demographic Alignment (DA)

###### Marketing Appeal (MA)

###### Visual Quality (VQ)

###### Average

87.5

94

74

Random Search

Random Search

Random Search

Random Search

Random Search

85.0

84

Warm-Up + MAB

Warm-Up + MAB

Warm-Up + MAB

Warm-Up + MAB

Warm-Up + MAB

72

85.0

92

82.5

82

70

82.5

90

80

80.0

68

80.0

88

Score

78

77.5

66

77.5

76

86

75.0

64

75.0

74

84

72.5

62

72.5

72

82

70.0

60

70

70.0

2 4 6 8 10

2 4 6 8 10

2 4 6 8 10

2 4 6 8 10

2 4 6 8 10

Iteration

Iteration

Iteration

Iteration

Iteration

- Figure 2 | MAB Optimization Efficiency. Cumulative best performance over 10 iterations comparing Co-Director’s MAB optimization against an undirected Random Search.

comparable parity.

#### 5.2. MLLM Metric Validation via Human Evaluation

To validate the MLLM-as-a-Judge metrics, we conducted a human study on a 50-scenario subset. Each video was evaluated by 5 independent raters along the four dimensions (VAF, DA, MA, VQ) on a 5-point Likert scale.

Table 2 reports Krippendorff’s Alpha (𝛼), Cohen’s Kappa (𝜅), Pearson (𝑟), Spearman (𝜌), and Mean Absolute Error (MAE), establishing human agreement as the theoretical upper bound. The MLLM demonstrates strong alignment, approaching this ceiling in complex dimensions like DA and MA. Visual Quality (VQ) exhibits lower correlation due to differing evaluation mechanics: humans viewing real-time playback often find transient diffusion artifacts less visually disruptive. The MLLM, conversely, processes frames analytically, strictly penalizing brief structural or temporal inconsistencies. Thus, while serving as a highly aligned proxy for narrative evaluation, the MLLM’s VQ score acts as a strict physical quality gate rather than a direct mirror of subjective human perception.

- Table 2 | Human agreement and correlation statistics across 50 scenarios. Human-human agreement serves as the theoretical ceiling for MLLM-human alignment.

Human-Human (Ceiling) MLLM-Human

Dimension

𝛼 𝜅 𝑟 𝜌 MAE ↓ 𝛼 𝜅 𝑟 𝜌 MAE ↓

VAF 0.606 0.465 0.733 0.635 0.644 0.521 0.330 0.601 0.522 1.016 DA 0.641 0.523 0.751 0.617 0.548 0.592 0.438 0.647 0.512 0.770 MA 0.620 0.489 0.737 0.673 0.596 0.562 0.386 0.570 0.499 0.848 VQ 0.578 0.463 0.705 0.675 0.642 0.469 0.346 0.345 0.317 0.945

𝛼 (Krippendorff’s Alpha): Calculated over the full rater pool. 𝜅 (Cohen’s Kappa) & MAE: Average of all pairwise individual comparisons. 𝑟 (Pearson) & 𝜌 (Spearman): Calculated against the Human MOS.

Perceptual Comparison (MOS) The Mean Opinion Scores (MOS) in Table 3 corroborate our findings through automatic evaluation. Co-Director achieves the highest overall perceptual quality (3.96), outperforming the strongest monolithic baseline, Veo 3.1 (3.71), and significantly exceeding specialized agentic systems like AniMaker (3.07) and MovieAgent (3.22). Notably, Co-Director establishes state-of-the-art performance across all four individual dimensions. This confirms that our framework’s global optimization effectively raises the narrative and visual ceiling of the underlying generative backbones.

- Table 3 | Mean Opinion Scores (MOS). Average of 5 human raters per video on a [1–5] scale.

Method VAF DA MA VQ Avg. AniMaker 3.34 3.71 2.72 2.50 3.07 MovieAgent 3.64 4.06 2.85 2.31 3.22 Wan 2.6 3.84 3.65 3.13 3.29 3.48 Kling 3.0 Omni 3.93 4.13 3.07 3.21 3.59 Veo 3.1 3.90 4.20 3.40 3.32 3.71 Co-Director (Ours) 4.22 4.41 3.65 3.58 3.96

- Table 4 | Ablation results (𝑇 = 4). S-Ref: Storyline and K-Ref: Keyframe Refinement.

# S-Ref. K-Ref. MAB Logic VAF DA MA VQ Avg. (1) ✓ ✓ Ours 79.8 88.6 80.0 67.5 79.0

| | |
|---|---|
|(2) ✓ ✓ Cold Start<br>(3) ✓ ✓ Scalar Reward<br><br><br>|78.2 87.8 77.7 65.6 77.3 74.8 87.9 77.8 65.3 76.4|
|(4) × ✓ Ours<br>(5) ✓ × Ours<br>(6) × × Ours<br><br><br>|74.0 86.2 73.7 62.0 74.0 70.0 82.5 71.5 62.0 71.5 68.1 78.6 69.0 62.0 69.4|
|(7) ✓ ✓ None<br>(8) × × None<br>|66.2 75.3 67.6 63.6 68.2 65.4 74.8 66.1 62.5 67.2|

#### 5.3. Ablation Studies

MAB Logic We compare our MAB search logic (1) against two variants. Reverting to a cold start (2) degrades performance, as lacking strong initialization produces higher-variance trajectories during early exploration. Crucially, collapsing the multi-dimensional arm-specific reward into a single scalar (3)—using the holistic score meant for final video selection—reduces VAF by 5.0 points. This validates our factored reward design. By leveraging the MLLM evaluator to decouple the overall video score from arm-specific rewards, the MAB isolates the exact parameters driving success (see App. K).

Self-Refinement Loops Removing Storyline Refinement (4) primarily penalizes semantic metrics like MA and DA, validating its impact on narrative coherence. Removing Keyframe Refinement (5) causes a steep 9.8-point VAF drop, confirming its vital role in preserving product visual identity. Stripping all local refinement and global optimization yields a “barebones” single-pass baseline (8) (67.2 Avg.). Our complete architecture outperforms this linear performance floor by a relative 17.6%.

#### 5.4. Qualitative Results and Discussion

To complement our quantitative analysis, Appendix B provides qualitative results showcasing generation artifacts across GenAd-Bench scenarios. These examples demonstrate Co-Director’s ability to interpret abstract ideas into cohesive visual narratives with precise scene instructions. Our observations highlight three primary strengths: (1) Strategic Adaptability: evoking expected atmospheres for stereotypical prompts while maintaining consistency in unconventional contexts; (2) Identity Preservation: achieving character and product consistency across dynamic poses, varied angles, and dramatic environmental transitions; and (3) Logical Robustness: local self-refinement correcting LLM hallucinations like misidentified product categories (App. H). Furthermore, evaluation on ViStoryBench-Lite (Zhuang et al., 2025) (App. C) confirms Co-Director’s core consistency and creative steering generalize to non-advertising cinematic narratives, outperforming specialized base-

lines in style and character fidelity. We encourage readers to consult the appendix and supplementary videos for a complete trace of pipeline reasoning and rendered outputs.

### 6. Conclusion

We present Co-Director, a hierarchical multi-agent framework that formulates video storytelling as a global optimization problem. By navigating a creative latent space via MAB-driven steering, our system achieves high-fidelity narrative consistency that traditionally requires large-scale production teams. Furthermore, to evaluate this capability, we introduce GenAd-Bench, a rigorous dataset designed to assess end-to-end generative workflows under strict targeting and visual constraints. Experimental results confirm that Co-Director outperforms both monolithic models and existing agentic pipelines, offering a principled, highly efficient approach to autonomous visual storytelling that generalizes beyond advertising to broader cinematic narratives.

### References

- J. I. Alpert and M. I. Alpert. Music influences on mood and purchase intentions. Psychology & Marketing, 7(2):109–133, 1990.

D. Aronofsky. On this day... 1776. https://www.imdb.com/title/tt39466407, 2026.

- D. Aronofsky and E. McNitt. Ancestra. https://www.imdb.com/title/tt37344561, 2025.

- P. Auer, N. Cesa-Bianchi, and P. Fischer. Finite-time analysis of the multiarmed bandit problem. Machine learning, 47(2):235–256, 2002.

L. Baardman, E. Fata, A. Pani, and G. Perakis. Dynamic creative optimization in online display advertising. Available at SSRN 3863663, 2021.

T. H. Baek. Digital advertising in the age of generative ai. Journal of Current Issues & Research in Advertising, 44(3):249–251, 2023.

T. E. Barry and D. J. Howard. A review and critique of the hierarchy of effects in advertising. International Journal of advertising, 9(2):121–135, 1990.

- S. Bubeck and N. Cesa-Bianchi. Regret analysis of stochastic and nonstochastic multi-armed bandit problems. arXiv preprint arXiv:1204.5721, 2012.

ByteDance. Seedance 2.0. https://seedance2.ai/, 2026.

- S. Cai, C. Yang, L. Zhang, Y. Guo, J. Xiao, Z. Yang, Y. Xu, Z. Yang, A. Yuille, L. Guibas, et al. Mixture of contexts for long video generation. arXiv preprint arXiv:2508.21058, 2025.

Creatify AI. Creatify. https://creatify.ai, 2026.

- V. Divya and A. U. Mirza. Transforming content creation: The influence of generative ai on a new frontier. Exploring the frontiers of artificial intelligence and machine learning technologies, 143:1–30, 2024.

M. Elmoghany, R. Rossi, S. Yoon, S. Mukherjee, E. M. Bakr, P. Mathur, G. Wu, V. D. Lai, N. Lipka, R. Zhang, et al. A survey on long-video storytelling generation: architectures, consistency, and cinematic quality. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7023–7035, 2025.

- J. E. Escalas. Narrative processing: Building consumer connections to brands. Journal of consumer psychology, 14(1-2):168–180, 2004.

Google. Lyria. https://docs.cloud.google.com/vertex-ai/generative-ai/docs/ model-reference/lyria-music-generation, 2025.

- Google. Gemini 2.5 Pro TTS. https://docs.cloud.google.com/text-to-speech/docs/ gemini-tts, 2026a.
- Google. Gemini 3 Pro. https://docs.cloud.google.com/vertex-ai/generative-ai/ docs/models/gemini/3-pro, 2026b.

Google. Gemini 3 Pro Image. https://docs.cloud.google.com/vertex-ai/ generative-ai/docs/models/gemini/3-pro-image, 2026c.

Google. Veo 3.1. https://docs.cloud.google.com/vertex-ai/generative-ai/docs/ models/veo/3-1-generate, 2026d.

- G. J. Gorn. The effects of music in advertising on choice behavior: A classical conditioning approach. Journal of marketing, 46(1):94–101, 1982.

- Y. HaCohen, N. Chiprut, B. Brazowski, D. Shalem, D. Moshe, E. Richardson, E. Levin, G. Shiran, N. Zabari, O. Gordon, et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.

HeyGen. Heygen. https://www.heygen.com/, 2026.

- X. Hou, B. Ma, J. Cheng, X. Ren, K. Yu, W. Li, T. Zheng, and Q. Lu. Personavlog: Personalized multimodal vlog generation with multi-agent collaboration and iterative self-correction. arXiv preprint arXiv:2508.13602, 2025.

K. Huang, Y. Huang, X. Wang, Z. Lin, X. Ning, P. Wan, D. Zhang, Y. Wang, and X. Liu. Filmaster: Bridging cinematic principles and generative ai for automated film generation. arXiv preprint arXiv:2506.18899, 2025.

Z. Huang, Y. He, J. Yu, F. Zhang, C. Si, Y. Jiang, Y. Zhang, T. Wu, Q. Jin, N. Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.

Kuaishou. Kling video 3.0 omni. https://app.klingai.com/global/omni/new, 2026. T. L. Lai and H. Robbins. Asymptotically efficient adaptive allocation rules. Advances in applied

mathematics, 6(1):4–22, 1985. H. A. Laskey, E. Day, and M. R. Crask. Typology of main message strategies for television commercials. Journal of advertising, 18(1):36–41, 1989.

- Y. Lee, J. Boen, and C. Finn. Feedback descent: Open-ended text optimization via pairwise comparison. arXiv preprint arXiv:2511.07919, 2025.

- L. Liu, T. Ma, B. Li, Z. Chen, J. Liu, G. Li, S. Zhou, Q. He, and X. Wu. Phantom: Subject-consistent video generation via cross-modal alignment. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14951–14961, 2025.

H. Lin, A. Zala, J. Cho, and M. Bansal. Videodirectorgpt: Consistent multi-scene video generation via llm-guided planning. arXiv preprint arXiv:2309.15091, 2023.

- M. Minsky. Steps toward artificial intelligence. Proceedings of the IRE, 49(1):8–30, 1961.

C. Mu, X. He, Q. Yang, W. Chen, J. Yao, H. Liu, Z. Yi, B. Zhao, X. Chen, R. Ma, et al. The script is all you need: An agentic framework for long-horizon dialogue-to-cinematic video generation. arXiv preprint arXiv:2601.17737, 2026.

H. Shi, Y. Li, X. Chen, L. Wang, B. Hu, and M. Zhang. Animaker: Multi-agent animated storytelling with mcts-driven clip generation. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers, pages 1–11, 2025.

C. Snell, J. Lee, K. Xu, and A. Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.

- E. K. Strong. The psychology of selling and advertising. McGraw-Hill book Company, Incorporated, 1925.

- K. Sun, K. Huang, X. Liu, Y. Wu, Z. Xu, Z. Li, and X. Liu. T2v-compbench: A comprehensive benchmark for compositional text-to-video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8406–8416, 2025.

- X. Tang, X. Lei, C. Zhu, S. Chen, R. Yuan, Y. Li, C. Oh, G. Zhang, W. Huang, E. Benetos, et al. Automv: An automatic multi-agent system for music video generation. arXiv preprint arXiv:2512.12196, 2025.

- T. Teixeira, M. Wedel, and R. Pieters. Emotion-induced engagement in internet video advertisements. Journal of marketing research, 49(2):144–159, 2012.

T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu, H. Zhao, J. Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

- Q. Wang, Z. Huang, R. Jia, P. Debevec, and N. Yu. Mavis: A multi-agent framework for long-sequence video storytelling. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2273–2295, 2026.

- Z. Wei, M. Li, Z. Zhang, R. Yuan, P. Hui, H. Qu, J. Evans, M. Agrawala, and A. Rao. Hollywood town: Long-video generation via cross-modal multi-agent orchestration. arXiv preprint arXiv:2510.22431, 2025.

- W. Wu, M. Liu, Z. Zhu, X. Xia, H. Feng, W. Wang, K. Q. Lin, C. Shen, and M. Z. Shou. Moviebench: A hierarchical movie level dataset for long video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 28984–28994, 2025a.

W. Wu, Z. Zhu, and M. Z. Shou. Automated movie generation via multi-agent cot planning. arXiv preprint arXiv:2503.07314, 2025b.

J. Xiao, C. Yang, L. Zhang, S. Cai, Y. Zhao, Y. Guo, G. Wetzstein, M. Agrawala, A. Yuille, and L. Jiang. Captain cinema: Towards short movie generation. In The Fourteenth International Conference on Learning Representations, 2025.

M. Yuksekgonul, F. Bianchi, J. Boen, S. Liu, Z. Huang, C. Guestrin, and J. Zou. Textgrad: Automatic"

differentiation" via text. arXiv preprint arXiv:2406.07496, 2024. H. Zettl. Sight, sound, motion; applied media aesthetics. 1973.

- R. Zhang, B. Yu, J. Min, Y. Xin, Z. Wei, J. N. Shi, M. Huang, X. Kong, N. L. Xin, S. Jiang, et al. Generative ai for film creation: A survey of recent advances. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 6267–6279, 2025.

C. Zhao, M. Liu, W. Wang, W. Chen, F. Wang, H. Chen, B. Zhang, and C. Shen. Moviedreamer: Hierarchical generation for coherent long visual sequence. arXiv preprint arXiv:2407.16655, 2024.

- Y. Zhou, D. Zhou, M.-M. Cheng, J. Feng, and Q. Hou. Storydiffusion: Consistent self-attention for long-range image and video generation. Advances in Neural Information Processing Systems, 37: 110315–110340, 2024.

C. Zhuang, A. Huang, Y. Hu, J. Wu, W. Cheng, J. Liao, H. Wang, X. Liao, W. Cai, H. Xu, et al. Vistorybench: Comprehensive benchmark suite for story visualization. arXiv preprint arXiv:2505.24862, 2025.

### A. Appendix Overview

To assist in navigating the supplemental materials, this section provides a directory of the contents contained within this appendix. Each section covers a distinct component of our results, dataset, or methodology:

- • Section B: Qualitative Results on GenAd-Bench — Visual galleries demonstrating CoDirector’s ability to preserve brand identity across diverse narrative arcs.
- • Section C: Evaluation on General Video Storytelling — Performance metrics and qualitative comparisons on the ViStoryBench benchmark (Zhuang et al., 2025).
- • Section D: GenAd-Bench Details — Details on dataset construction and descriptive statistics.
- • Section E: Baseline Implementation Details — Standardized configurations, prompt templates, and foundation models used for all evaluated baselines.
- • Section F: MAB Optimization Efficiency — Analysis of sample efficiency and the convergence of our multi-armed bandit creative steering.
- • Section G: Creative Direction Examples — Walkthroughs of hierarchical parameterization and how aesthetic archetypes influence visual output.
- • Section H: Local Optimization via Self-Refinement — Evidence of the agentic loop’s ability to correct structural hallucinations and identity drift.
- • Section I: Storyline Evaluation Prompt — The system prompt and logic used to evaluate narrative coherence in a storyline.
- • Section J: Keyframe Evaluation Prompt — The system prompt used to assess visual consistency and character preservation in keyframes.
- • Section K: Final Video Evaluation Prompt — The system prompt and rubric for scoring end-to-end video quality and creative alignment.
- • Section L: GenAd-Bench Evaluation System Prompts — The complete set of automated scoring rubrics for the proposed benchmark.

### B. Qualitative Results on GenAd-Bench

This section presents qualitative results that demonstrate Co-Director’s ability to translate six-point product prompts and reference visuals (brand logos and products) into cohesive and high-fidelity visual narratives.

Figure 3 shows synthesized storylines alongside their corresponding keyframes across four distinct scenarios. These examples highlight the framework’s ability to ground abstract marketing concepts—such as a functional ’bellows effect’ or a psychological shift toward safety—into precise scene instructions with rich emotional arcs. Furthermore, they showcase the model’s robust preservation of character and product identity across dynamic physical poses, varied camera angles, and dramatic environmental transitions.

To further illustrate the pipeline’s versatility, Figure 4 provides keyframe sequences from 10 additional scenarios, spanning both stereotypical and unconventional product contexts. Co-Director consistently evokes the appropriate atmosphere for expected settings, such as rustic or softly lit environments, while maintaining structural and temporal consistency in unconventional scenarios. This includes displaying the correct object manipulation in unexpected spaces (e.g., a high-tech workspace or a library) and handling complex lighting conditions like neon urban streets.

### C. Evaluation on General Video Storytelling

In the main paper, we chose video advertising as our main demonstration scenario because it highlights a novel aspect of our framework: the ability to generate a full video narrative from a single, highlevel idea, rather than needing a detailed, scene-by-scene script. Existing agentic systems, such as AniMaker (Shi et al., 2025) and MovieAgent (Wu et al., 2025b), require structured, scene-byscene scripts provided a priori. In contrast, our framework synthesizes full cohesive narratives from a brief tagline (e.g., our six-point prompt). While MAViS (Wang et al., 2026) attempts a similar idea-to-video pipeline, their evaluation relies on a closed set of 10 custom prompts, precluding reproducible comparison. This underscores the critical importance of our proposed GenAd-Bench. Existing visual narrative datasets, such as MovieBench (Wu et al., 2025a) and ViStoryBench (Zhuang

- et al., 2025), evaluate isolated generative steps (e.g., text-to-image or image-to-video synthesis) rather than the complete video storytelling pipeline. GenAd-Bench fills this void by providing a rigorous, reproducible standard for evaluating the entire trajectory of video storytelling—from abstract conceptualization to final rendered output.

Despite the architectural mismatch between our end-to-end framework and traditional staged benchmarks, we evaluate Co-Director on ViStoryBench-Lite (Zhuang et al., 2025) to demonstrate its generality across domains. Because ViStoryBench evaluates storyboard generation—providing strict shot-by-shot scripts and character reference images as inputs, and expecting static image outputs rather than video—we ablate our pipeline accordingly. Specifically, we bypass our script generation and video synthesis modules, isolating and deploying only the Keyframe Agent, the local self-refinement loop, and the global MAB-steered optimization. This allows us to prove that Co-Director’s core consistency and creative direction mechanisms remain highly effective even when constrained to rigid, pre-defined narrative structures. To ensure a rigorous evaluation, we deliberately select agentic video generation pipelines from the official ViStoryBench leaderboard as our baselines, as their modular architectures provide the most direct and equitable comparison to our own.

As shown in Table 5, Co-Director achieves strong performance in visual consistency, securing the highest scores in Style (Cross: 0.499, Self: 0.743) and Character (Cross: 0.499, Self: 0.593) consistency metrics. Furthermore, it yields the highest average Prompt Alignment (3.166), demonstrating superior control over camera dynamics and local context compared to prior agentic baselines like AniMaker and MovieAgent. While Co-Director trails in Quality & Diversity (e.g., Inception Score: 10.900) and On-stage Character Count Matching (OCCM: 66.215), these metrics fundamentally conflict with cohesive storytelling. Inception Score favors high visual variance, naturally opposing our model’s strict cross-scene consistency. Likewise, OCCM enforces a rigid “closed-world” character

Qux-Uniform (ID:23, Type: Unconventional) Corge-Top (ID:30, Type: Stereotypical)

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

A focused 21-year-old Latina caddy navigates the prestigious Santa Maria Golf Club in Panama City, using the professional composure of her Qux-Uniform to seamlessly transition from physical labor in 91-degree heat to a high-stakes networking moment with a corporate executive.

In the oppressive, humid heat of Burleigh Heads, a professional athlete performs a high-intensity weightlifting set, demonstrating how his premium performance gear utilizes a 'bellows effect' to manage extreme moisture and maintain structural integrity despite the punishing Gold Coast climate.

- Scene 1: A dynamic shot showing shimmering heat distortion across the emerald fairway as a background golfer visibly struggles, wiping sweat from his brow; in the foreground, the Latina caddy stands in sharp, cool contrast, her Qux-Uniform looking impeccably dry and crisp in the 91-degree Panama sun.
- Scene 2: In a medium shot, she confidently strides across the fairway carrying a heavy golf bag, her Qux-Uniform appearing crisp and dry despite the 90% humidity.
- Scene 3: She provides a precise yardage reading and a strategic tip to a wealthy executive, who nods in impressed recognition of her professional poise and expertise.
- Scene 4: The scene ends with a firm, professional handshake between the caddy and the executive by the final hole, the character's empowered expression signaling a successful networking connection.

- Scene 1: A tight, high-quality shot of Lachlan, 42, gripping a cold steel barbell at an outdoor beachside gym; intense sun glare and visible heat haze shimmer off the equipment, emphasizing the punishing humidity as sweat glints on his skin against the Gold Coast skyline.
- Scene 2: Lachlan executes a powerful, explosive clean-and-jerk; the camera captures the raw power of the lift as he drives the weight overhead, emphasizing the garment's flexibility and resilience during peak physical exertion.
- Scene 3: In a slow-motion side profile, Lachlan drops the weights and breathes heavily; he pulls the shirt fabric away from his chest to demonstrate the 'bellows effect,' showing the material snap back and circulate air without sticking, even in 82% subtropical humidity.
- Scene 4: Lachlan looks directly into the lens with an authoritative, satisfied nod, the Corge-Top logo subtly visible as the frame fades to a clean, modern product lockup.

Grault-Clutch (ID:35, Type: Stereotypical) Garply-Lantern (ID:38, Type: Stereotypical)

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

A sophisticated woman transitions from the cold, damp Moscow street into the warm, cultured embrace of an evening at the Bolshoi Theatre, her Grault-Clutch serving as the anchor of her poise through its craftsmanship and functional luxury.

A sophisticated woman in Nelspruit transforms a dark, humid bushveld campsite into a warm, secure family sanctuary using the Garply-Lantern, moving from the uncertainty of the wild night to the fortified comfort of home.

- Scene 1: An extreme close-up captures the sharp glint of the Grault-Clutch’s gold hardware as freezing Moscow rain beads on its textured surface, creating a high-impact visual hook against the grey slush.
- Scene 2: In the damp March wind, her gloved fingers expertly engage the turn-lock clasp with a crisp mechanical click, emphasizing the product's craftsmanship as she prepares to enter the theatre.
- Scene 3: Stepping through the heavy doors, she smoothly retrieves a physical theatre invitation from the clutch’s interior, her expression softening into serene intellectual command as she transitions into the golden light.
- Scene 4: In the warm glow of the foyer, the camera lingers on a clear close-up of the delicate geometric 'G' logo on the Grault-Clutch as it catches the golden light, ensuring the brand identity is the final, memorable focal point of her transition.

- Scene 1: In the humid Nelspruit dusk, a 42-year-old woman fumbles urgently with a jammed tent zipper while her head snaps toward the thicket at the sudden sound of a snapping branch, her breath hitching in the encroaching mist.
- Scene 2: She clicks on the premium Garply-Lantern, and a wide, warm golden glow instantly pushes back the shadows, revealing a beautifully organized glamping setup and a set dinner table.
- Scene 3: Close-up on her face as her expression softens from professional intensity to relaxed peace, her hand resting confidently on the lantern's tactile, high-end frame.
- Scene 4: The lantern sits centrally on a wooden camp table, illuminating the entire tent interior with a reassuring light while the dark South African bush remains safely beyond the canvas.

- Figure 3 | Qualitative results on GenAd-Bench. Co-Director translates complex product prompts into cohesive visual narratives. The framework maintains character and product identity across dynamic poses and varied angles (top) and preserves precise product details during dramatic environmental and lighting transitions (bottom).

Wandom-Yarn (ID:40, Type: Stereotypical) FredBaz-Syrup (ID:47, Type: Stereotypical)

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

Wandom-Glue (ID:41, Type: Unconventional) Plugh-Oil (ID:51, Type: Stereotypical)

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

Xyzzy-Bear (ID:52, Type: Stereotypical) BobBaz-Sand (ID:62, Type: Unconventional)

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

EveFoo-Helmet (ID:66, Type: Unconventional) EveFoo-Lock (ID:67, Type: Unconventional)

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

FooTrent-Kit (ID:75, Type: Stereotypical) LoremBaz-Pillow (ID:86, Type: Unconventional)

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

[Figure 103]

- Figure 4 | Additional qualitative results on GenAd-Bench. Co-Director effectively generates diverse visual narratives across both stereotypical and unconventional product scenarios. It successfully evokes expected atmospheres for stereotypical prompts, such as the rustic warmth of FredBaz-Syrup or the gentle nursery lighting of Xyzzy-Bear. The pipeline shows robustness to unconventional contexts, maintaining consistency in character/product identity and complex lighting (e.g., neon urban streets in EveFoo-Helmet) while executing precise object manipulation in unexpected environments like a library (LoremBaz-Pillow) or a high-tech workspace (Wandom-Glue).

count, naturally penalizing Co-Director for organically generating unprompted background actors (e.g., pedestrians in a “bustling city”) that enhance cinematic realism.

These quantitative metrics are directly corroborated by qualitative observations. As illustrated in Figure 5, Co-Director reliably generates temporally coherent frame sequences across diverse narratives, successfully preserving both character identity and background spatial structures over time. A closer examination of a specific sequence (Figure 6) further highlights the framework’s strong prompt alignment and character preservation. While minor anatomical artifacts, such as occasional left-right hand confusion, can still manifest as an artifact of the underlying diffusion process, the framework’s overall visual fidelity and narrative continuity remain exceptionally high.

Ultimately, these results illustrate that Co-Director is not narrowly overfitted to marketing content, but is instead a highly capable engine for generalized visual storytelling. Crucially, video advertising should not be viewed as a restrictive niche; rather, it is a demanding subset of general storytelling characterized by rigid, testable constraints on brand fidelity, demographic alignment, and runtime. By mastering these rigorous constraints, Co-Director demonstrates a foundational narrative competency that seamlessly transfers to broader cinematic and storytelling domains.

### D. GenAd-Bench Details

To rigorously evaluate video storytelling, we introduce GenAd-Bench. This appendix details the dataset’s construction process, comprehensive demographic and category statistics, our data split strategy, and qualitative examples.

#### D.1. Dataset Construction Pipeline

Current video evaluation benchmarks predominantly rely on generic, short-horizon prompts (e.g., “a dog running on grass”) and evaluate well-known entities, allowing models to rely heavily on parametric memory. GenAd-Bench circumvents this by utilizing a human-in-the-loop pipeline to synthesize 200 fictitious products across 50 fictitious brands.

We utilized Gemini 3 Pro Google (2026b) to generate the structural text metadata and Nano Banana Pro (Google, 2026c) to generate the visual reference assets. The construction followed a hierarchical schema:

- 1. Brand & Product Synthesis: The LLM generated 50 unique brands and assigned exactly 4 distinct products to each. Products were explicitly prompted to span a wide range of form factors, including small consumer goods, large industrial equipment, and intangible offerings (e.g., software and subscription services).
- 2. Paired Scenario Generation: For each of the 200 products, the LLM generated two contrasting

#### Table 5 | Evaluation on ViStoryBench-Lite.

Style Consistency

Character Consistency

Quality & Diversity OCCM

Prompt Alignment Cross Self Cross Self Aesthetics Inception Scene Camera GCA LCA Avg

Method

MMStoryAgent 0.261 0.661 0.385 0.598 5.910 8.090 58.236 2.965 2.452 1.643 1.500 2.012 Vlogger 0.299 0.497 0.373 0.519 4.242 8.827 79.145 1.749 3.143 2.429 2.350 2.404 AniMaker 0.305 0.558 0.423 0.593 5.603 9.942 72.034 3.622 2.932 3.388 2.850 3.128 MovieAgent 0.346 0.539 0.433 0.582 5.318 12.044 67.289 3.406 3.052 3.404 2.650 3.033 Co-Director 0.499 0.743 0.499 0.593 5.286 10.900 66.215 3.412 3.288 2.999 2.966 3.166

StoryStoryStoryStoryStoryStoryStoryStoryStoryStory55082729325260647901

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

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

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

[Figure 153]

[Figure 154]

[Figure 155]

- Figure 5 | Qualitative results on ViStoryBench. Five consecutive frames are displayed for each generated story (01, 08, 27, 29, 32, 52, 55, 60, 64, 79). Across these diverse examples, Co-Director maintains robust temporal consistency, successfully preserving both character identity and background spatial coherence throughout the sequences.

[Figure 157]

- [Shot 1] Totto-chan stands in front of the ticket inspector uncle with an expectant expression, her right hand outstretched. The ticket inspector uncle smiles and shakes his head, holding a stack of tickets in his left hand. In the background, the ticket box next to the train door is half full.

- [Shot 2] Totto-chan stands beside the train door, looking focused, her eyes fixed on the ticket box, her right hand gently touching it. In the background, the train door is open with passengers boarding and alighting

- [Shot 3] Totto-chan’s mother holds Totto-chan’s hand as they walk down the street. Totto-chan looks cheerful, while her mother appears slightly worried. Their shadows stretch long on the ground.

- [Shot 4] Totto-chan and her mother stand at the school gate. Totto-chan looks surprised, pointing at the school name plate with her right hand, while her mother smiles at her. In the background, the two trees forming the gate stand out prominently.

- [Shot 5] Totto-chan stands in the center of the schoolyard, looking astonished, her hands covering her mouth. In the background, the six train cars are neatly arranged, gleaming in the sunlight.

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

- Figure 6 | Qualitative results on ViStoryBench (Story 28). The generated sequence demonstrates strong prompt alignment and robust character consistency across frames, though minor anatomical artifacts (e.g., left-right hand confusion) occasionally occur.

Brand Logo Product Pair 1 Product Pair 2 Product Pair 3 Product Pair 4

[Figure 163]

[Figure 164]

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

[Figure 261]

[Figure 262]

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

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

- Figure 7 | GenAd-Bench reference visuals. The reference set comprises brand logos and product images, with each brand featuring four product pairs specifically curated to represent both stereotypical and unconventional target demographics.

target demographics to test conceptual flexibility: a Stereotypical scenario (aligned with traditional marketing tropes) and an Unconventional scenario (subverting traditional expectations).

- 3. Visual Asset Generation: The text-to-image model generated a distinct vector-style logo for each of the 50 brands. Subsequently, it generated product reference images conditioned on both the textual product description and the respective brand logo to ensure visual-semantic integration.
- 4. Manual Verification & Quality Control: To ensure high-fidelity inputs for the product reference generation stage, we first manually verified and corrected all brand logos. Following the generation of these product reference visuals, approximately 15% were regenerated to address improper logo integration, logical inconsistencies, or unintended visual memorization (e.g., designs too closely resembling products existing in the real-world). This targeted refinement ensures the uniqueness and brand accuracy of the primary visual assets before they are propagated downstream.

#### D.2. Taxonomy and Distribution Statistics

GenAd-Bench contains 400 evaluation scenarios distributed across a diverse range of industries and global contexts.

Macro-Category Distribution The dataset challenges models to render various physical properties, including reflective metals (Jewelry), transparent liquids (Perfumes), and complex dynamics (Drones, Apparel). As illustrated in Figure 8a, the 200 products span 24 distinct macro-categories. High-frequency categories such as Home Goods (24), Food (20), Fashion (20), and Tech (20), test foundational object generation, while specialized domains like Automotive (8), Logistics (4), and Baby (4) probe the model’s ability to synthesize niche, domain-specific contexts without hallucination.

Demographic and Geographic Parity To ensure the multi-armed bandit optimization does not merely exploit ingrained training biases, GenAd-Bench explicitly enforces balanced demographic and geographic distributions. As shown in Figure 8d, the dataset maintains exact gender parity: 50% (200) of the target personas are Male, and 50% (200) are Female. Crucially, because paired scenarios invert the demographic targets for the exact same product, gender bias is neutralized at the product level. Target age brackets (Figure 8b) uniformly represent four key life stages: Gen Z (18–24), Millennials (25–34), Gen X (35–50), and Seniors (50+). Finally, the framework demands robust cultural and environmental alignment beyond traditional Western-centric backdrops; the scenarios are grounded in 183 uniquely prompted global locations. While the dataset spans all six major continents, it deliberately concentrates on the two most diverse global consumer markets: Europe (158 scenarios) and Asia (108 scenarios). Within these clusters, geographic representation remains highly granular. The European scenarios span from primary Western hubs (e.g., France, UK, Germany) to Nordic and Eastern European locales, while the Asian distribution covers major East Asian centers (Japan, China, South Korea) alongside robust representation across Southeast Asia, India, and the Middle East.

#### D.3. Dataset Split

We partitioned the 400-scenario dataset into two equal 200-scenario splits. The Hillclimbing Set is reserved strictly iterative system optimization and is excluded from the evaluations presented in this work. All performance metrics are reported solely on the Validation Set, which is stratified into two sub-challenges to assess Co-Director’s generalization capabilities across 45 unique brands:

• In-Domain (80% of Val / 160 Scenarios): Contains 80 products from 40 brands that share macro-categories with the Hillclimbing set. This evaluates a system’s ability to extrapolate optimized strategies to entirely novel products and brands within familiar industries.

###### (a) Macro-Category Distribution

###### (b) Target Demographic Age

115

106 105

24

Home Goods

100

Scenarios

20

Food

74

20

Fashion

50

20

Tech

16

Personal Care

0

18-24 25-34 35-50 50+

12

Travel

8

Tools

8

Apparel

###### (c) Geographic Distribution

8

Beverage

8

Automotive

158

150

4

Education

Scenarios

108

4

Cosmetics

100

66

4

Jewelry

50

4

Baby

19 25 24

4

Health

0

EU Asia NA Africa SA Oceania

4

Pets

4

Furniture

###### (d) Target Gender Parity

4

Electronics

4

Finance

Male 50.0%

4

Toys

4

Hobbies

4

Outdoor

4

Logistics

50.0%

4

Appliances

Female

0 5 10 15 20 25

Number of Products

- Figure 8 | Dataset Statistics for GenAd-Bench. The benchmark enforces rigorous categorical, demographic, and geographic diversity to prevent generative collapse. (a) The macro-category distribution spans a wide array of physical domains. (b) Target age demographics cover four distinct life stages. (c) Scenarios are situated across 183 unique global locations, grouped here into six continental macro-regions. (d) Target gender distributions are perfectly balanced at the product level to neutralize latent generation biases.

• Out-of-Domain (20% of Val / 40 Scenarios): Contains 20 products from 5 brands drawn from distinct industries (e.g., Pets, Jewelry) that are absent from the Hillclimbing set. This tests a system’s broader generalization capabilities when handling product categories for which no category-specific optimization has occurred.

### E. Baseline Implementation Details

Monolithic and Commercial Platforms To evaluate the monolithic foundation models and specialized commercial platforms—LTX-2.3 (HaCohen et al., 2024), Kling 3.0 Omni (Kuaishou, 2026), Veo

- 3.1 (Google, 2026d), Wan 2.6 (Wan et al., 2025), Creatify (Creatify AI, 2026), and HeyGen (HeyGen, 2026)—we employ a standardized zero-shot prompting strategy. Each system is queried using the template: “Generate a video advertisement. [SIX-POINT PROMPT]. Reserve the last 1 second for an ending scene highlighting the provided brand logo image.” This ensures a uniform evaluation setting that tests each baseline’s ability to interpret the core product brief and integrate branding constraints. All the baseline methods natively generate audio given text prompt.

Multi-Agent Frameworks We compare against two structurally analogous agentic pipelines, AniMaker (Shi et al., 2025) and MovieAgent (Wu et al., 2025b), to isolate the architectural efficacy of our multi-agent system from the underlying generative backbones. Because these baselines require structured, multi-shot scripts rather than abstract product briefs, we introduced a Script Generator module. This module leverages Gemini 3 Pro (Google, 2026b) to decompose the initial prompt into a structured JSON script. The resulting script explicitly dictates the narrative arc, scene-specific visual prompts, voice-over text, and precise shot durations, enforced by a strict 12-second total video length constraint. To maintain a controlled comparison against our framework, these baselines are instantiated using identical foundation models: Veo 3.1 (Google, 2026d) for video generation and Gemini 2.5 Pro TTS (Google, 2026a) for voiceovers

### F. Extended Discussion on MAB Optimization Efficiency

While standard multi-armed bandit (MAB) formulations often analyze asymptotic convergence over thousands of iterations, video generation imposes extreme computational and financial constraints. An exhaustive grid search of our defined creative action space (𝜃𝑐𝑠 × 𝜃𝑛𝑚 × 𝜃𝑎𝑎) would require 36 discrete, end-to-end video generation pipelines per prompt. While this brute-force approach guarantees finding the global optimum, it renders the system entirely unscalable and defeats the purpose of an accessible agentic framework.

Therefore, the goal of our MAB formulation is not infinite-horizon exploration, but rather sample efficiency under resource constraints. We achieve this through two mechanisms:

- 1. Factored Rewards: By utilizing a factored reward signal 𝑅 = (𝑟𝑐𝑠, 𝑟𝑛𝑚, 𝑟𝑎𝑎), a single execution step 𝑡 simultaneously updates the expected values across all three independent creative axes. The system is not searching 36 distinct paths, but rather evaluating 10 independent parameters.
- 2. LLM-Driven Warm Start: We initialize the expected values using an LLM, safely biasing early exploration toward theoretically sound configurations and pruning the vast majority of sub-optimal combinations before iteration even begins.

Consequently, as demonstrated in our empirical results (Fig. 2), an iteration budget of 𝑇 = 4 is sufficient for the MAB to rapidly converge on a highly effective creative configuration. This approach clearly outperforms Random Search, capturing the benefits of a global search without incurring the computational overhead of exhaustive sampling.

### G. Creative Direction Examples

#### G.1. Horizontal Directing through Hierarchical Parameterization

Co-Director achieves global coherence through horizontal directing—the process of synthesizing disparate creative parameters into a unified, high-level vision. Rather than passing raw categorical arms as static strings, the Orchestrator Agent (Φ𝑜𝑟𝑐ℎ) generates a top-down creative direction that is decomposed into agent-specific technical directives. These directives are then routed to their respective sub-agents via dynamic prompt injection; specifically, the synthesized creative direction is injected into the system prompt of each agent. This ensures that the Storyline, Keyframe, and Video instructions are orthogonally aligned toward a single strategic goal, preventing semantic drift by grounding each agent’s execution in a centralized creative intent.

Table 6 illustrates an example. In Iteration 1, the bandit selects an Informational strategy (𝜃𝑐𝑠) delivered via a Vignette narrative mode (𝜃𝑛𝑚) and Clarity/Energy (𝜃𝑎𝑎) aesthetics. This results in

directives that prioritize "functional utility" and "vibrant suburban transit," utilizing high-temporal motion and fast cuts to emphasize the efficiency of the vehicle’s sliding doors and safety frame within an Osaka carpool context.

In contrast, in Iteration 2, the bandit pivots to a Transformational strategy (𝜃𝑐𝑠) using an Analytical mode (𝜃𝑛𝑚) and Cinematic Premium (𝜃𝑎𝑎). The instructions are immediately overridden to focus on a "logical manifesto" of access and protection, delivered through a "Premium Noir" aesthetic. This configuration utilizes Chiaroscuro lighting and low-temporal, reflective cinematography to connect the vehicle’s technical specifications to the user’s social identity and psychological state of "unshakable composure." This horizontal consistency across different narrative and sensory modalities demonstrates that Co-Director is capable of high-level creative reasoning, ensuring that the "what," "how," and "feel" of the advertisement remain in strict strategic alignment.

#### G.2. Visual Realization via Aesthetic Archetypes

To evaluate the qualitative impact of our hierarchical parameterization, we analyze the visual output of two scenarios across four distinct Aesthetic Archetypes (𝜃𝑎𝑎). As shown in Figure 9, the MABselected configuration acts as a "Digital Director," re-shooting the same product prompt to meet specific strategic intents:

- • Clarity/Energy (the "Pop" Look): This maximizes immediate product recognition. By flooding the scene with high-key, low-contrast illumination, the system eliminates shadows and emphasizes vibrancy. In the Grault-Heel example, this manifests as clean, energetic neons; in the Con-Kibble case, it produces a clinical, trustworthy brightness.
- • Cinematic Premium (the "Movie" Look): This targets perceived prestige via chiaroscuro (highcontrast) lighting. By sculpting depth through dramatic light and shadow, the product is elevated to a "hero" status. Note the sophisticated golden-hour rim lighting on the footwear and the moody, atmospheric shadows in the K9 training station.
- • Minimalist Focus (the "Gallery" Look): Here, the system prioritizes material essentialism. By stripping away environmental noise and utilizing macro-framing against pure backgrounds, the viewer is forced to engage with the physical "truth" of the object—such as the intricate crystal weave of a heel or the nutritional texture of a kibble pellet.
- • Kinetic Grit (the "Raw" Look): This emphasizes modern authenticity over polished perfection. Through low-key lighting and unstable handheld or FPV drone motion cues, the system captures a sense of "verité" intensity. This is visible in the gritty, urban shadows of the Ibiza nightlife and the tactical, unpolished urgency of the professional K9 unit.

Crucially, despite these shifts in aesthetic "vibe," the underlying reference visual assets remain consistent. The brand logos and structural features of the assets are preserved across every narrative arc, proving that our agentic steering does not compromise brand integrity.

### H. Local Optimization via Agentic Self-Refinement

Building on the foundations of feedback descent (Lee et al., 2025; Yuksekgonul et al., 2024), we implement an agentic self-refinement loop to ensure the logical and visual integrity of intermediate artifacts. While Section 3.2 detailed the adaptation of this framework for our pipeline, this section provides qualitative evidence of its performance. Specifically, we demonstrate how the system identifies and corrects structural hallucinations in storylines and enforces character consistency across generated keyframes.

#### Table 6 | MAB Parameter Synthesis into Creative Directions. This table illustrates how specific

combinations of Creative Strategy (𝜃𝑐𝑠), Narrative Mode (𝜃𝑛𝑚), and Aesthetic Archetype (𝜃𝑎𝑎) yield differentiated directives across the Story, Keyframe, and Video agents. Example: Foo-Family by Foo Corp (ID: 2, Type: Stereotypical).

Pre-Production Agent (Storyline) Keyframe Agent Video Agent Iteration 1 — 𝜃𝑐𝑠: Informational — 𝜃𝑛𝑚: Vignette — 𝜃𝑎𝑎: Clarity/Energy

Develop a script based on the Informational Creative Strategy (Laskey et al., 1989), focusing on the functional utility of the Foo-

Generate high-key, low-contrast imagery consistent with Zettl’s (1973) Media Aesthetics to suggest transparency and vital-

Execute a high-temporal motion edit characterized by fast cuts and high-arousal transitions (Zettl, 1973). The cinematography

|Family vehicle. The narrative must follow the ’Slice of Life’ mode (Escalas, 2004), presenting a series of atmospheric vignettes of adult women (aged 35-50) in Osaka, Japan, engaged in a carpool routine. Each scene must highlight a specific product attribute: the ease of the dual sliding doors for loading equipment, the spacious multipassenger seating for adult commuters, and the reinforced safety frame. Avoid a traditional plot arc; instead, create a consistent ’vibe’ of efficient, safe suburban transit. Use text overlays to provide factual evidence of safety ratings and door clearance measurements to appeal to the logical advantages of the vehicle.|ity. The visual style should be clean, modern, and bright, featuring the Foo-Family vehicle in suburban Osaka settings. Focus on the textures of the vehicle’s interior and the mechanical precision of the sliding doors. Characters should be Japanese women aged 35-50, dressed in contemporary professional-casual attire, interacting with the car in a bright, airy environment. Use a color palette of whites, soft blues, and metallic silver to emphasize safety and cleanliness. Ensure the Foo Corp logo is clearly visible on the vehicle.|should utilize dynamic camera movements, including quick tracking shots of the sliding doors in motion, rapid dolly-ins to safety sensors, and whip-pans between adult passengers to create a sense of energy and efficiency. The frame rate should be standard, but the editing rhythm must be upbeat and synchronized with a high-tempo, optimistic audio track. Use bright, even lighting throughout to maintain a high-key aesthetic. The camera should emphasize the vehicle’s accessibility and safety features through close-ups of functional components, maintaining a fast-paced, vibrant flow that reflects the vitality of the Osaka carpool lifestyle.|
|---|---|---|
|Iteration 2 — 𝜃𝑐𝑠:|Transformational — 𝜃𝑛𝑚: Analytical — 𝜃𝑎𝑎: C|inematic Premium<br><br>|
| | | |

Develop a script based on an argumentbased narrative mode (Escalas, 2004) that delivers a transformational creative strategy (Laskey et al., 1989). The script must function as a logical manifesto rather than a story, using direct address to the viewer. Focus on three logical pillars: ’The Logic of Access’ (sliding doors), ’The Logic of Protection’ (safety features), and ’The Logic of Community’ (multi-passenger capacity). Each point must be framed to evoke a psychological state of ’unshakable composure’ for the Osaka-based female lead (age 3550). The tone is premium, intellectual, and emotionally resonant, connecting the vehicle’s specs to the user’s social identity as a reliable pillar of her suburban community.

Generate high-fidelity scenes utilizing Chiaroscuro lighting (Zettl, 1973). Visuals must feature high contrast with deep, dramatic shadows and sharp, intentional highlights that define the Foo-Family vehicle’s silhouette and sliding door mechanism. The setting is a refined suburban Osaka environment at dusk, emphasizing a ’Premium Noir’ aesthetic. Surfaces should show realistic reflections of city lights. Characters are adult females (3550) with elegant, stoic expressions, dressed in high-end professional or suburban attire. Ensure the Foo Corp logo and vehicle branding are visible but integrated into the high-contrast environment.

Execute cinematography with lowtemporal motion (Zettl, 1973). Use slow, deliberate dolly-in and tracking shots that glide alongside the vehicle at a reflective pace. The motion of the sliding doors must be captured in smooth, weighted slow-motion to emphasize engineering precision. Avoid rapid cuts or handheld jitter; every frame must feel stable and composed. The camera should focus on the textures of the vehicle and the calm, confident faces of the adult passengers. Audio must be a cinematic, ambient orchestral score—heavy on strings and low-frequency resonance—to reinforce the high-end, transformational feel. The pacing must be rhythmic and analytical, matching the logical progression of the argument-based script.

Iteration 3 — 𝜃𝑐𝑠: Comparative — 𝜃𝑛𝑚: Narrative Drama — 𝜃𝑎𝑎: Minimalist Focus

Develop a three-act narrative script centered on a 40-year-old female driver in a bright, suburban Osaka neighborhood. Act 1 (Conflict): The protagonist struggles with the ’urban norm’—a standard sedan with hinged doors parked in a narrow space, making it impossible to load large school event supplies and for her adult friends to enter. Act 2 (Resolution): Introduce the Foo-Family vehicle. Highlight the unique value proposition of the motorized sliding doors that glide open effortlessly in the same tight space. Act 3 (Transformation): The group of adult women sits comfortably inside the spacious cabin, sharing a moment of relief and safety. The script must emphasize the contrast between the cramped standard and the Foo-Family’s accessibility.

Apply a Zettl-inspired ’Applied Media Aesthetics’ style. Visuals must be high-key with bright, clean, and airy backgrounds that evoke a pristine Osaka suburban morning. Focus on extreme detail and tactile textures: the brushed metal of the Foo-Family logo, the smooth track of the sliding door, and the premium fabric of the interior seating. Use a minimalist color palette with soft whites, silvers, and light blues. Every frame should look hyper-real and clinical yet inviting. All human subjects must be adult females aged 35-50, dressed in sophisticated, contemporary Japanese fashion.

Execute a cinematography plan focused on micro-movements and static precision. Use slow-motion macro shots to capture the mechanical elegance of the sliding door opening. Camera movement should be limited to subtle, buttery-smooth dollies or slow pans that emphasize the vehicle’s length and safety features. Avoid handheld or shaky movements. The visual rhythm should be calm and deliberate, allowing the viewer to focus on the ’ASMR’ quality of the car’s operation—the click of a seatbelt, the whisper-quiet slide of the door, and the tactile touch of the dashboard. Ensure the lighting remains consistently high-key throughout all transitions

[Figure 322]

[Figure 323]

Grault Design elevates with Grault-Heel (high-heeled footwear intended for performance or social events), targeting Female partygoers aged 18-24 in Ibiza, Spain who are interested in partying and glamour in nightclubs.

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

Clarity / Energy

Cinematic Premium

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

Minimalist Focus

Kinetic Grit

[Figure 340]

[Figure 341]

Consectetur Co feeds with Con-Kibble (dry nutritional pellets formulated for canine consumption), targeting Male officers aged 18-24 in Brussels, Belgium who are interested in K9 unit nutrition in police stations.

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

Clarity / Energy

Cinematic Premium

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

Minimalist Focus

Kinetic Grit

- Figure 9 | Visual Guide to Aesthetic Archetypes (𝜃𝑎𝑎): This comparison shows how the same product is "re-shot" by the system to match different moods. (a) Clarity/Energy ("The Pop Look"): Everything is bright and easy to see. It uses "high-key" lighting with no dark shadows to make the product look clean, vibrant, and exciting. (b) Cinematic Premium ("The Movie Look"): This uses "Chiaroscuro" (dramatic light and deep shadows). It makes the product look expensive and legendary, like a hero in a high-end film. (c) Minimalist Focus ("The Gallery Look"): The background is deleted or made clean. The camera zooms in to show off tiny textures (like crystals or kibble pellets) in a calm, "museum-like" setting. (d) Kinetic Grit ("The Raw Look"): This feels like a real-world documentary. The lighting is dark and "moody," and the camera feels like it’s being held by a person (handheld) or a drone, creating a sense of raw intensity. Examples: Grault-Heel by Grault Design (ID: 34, Type: Stereotypical) and Con-Kibble by Consectetur Co. (ID: 96, Type: Unconventional).

#### H.1. Storyline Self-Refinement

Following the multi-agent architecture described in Section 3.1, the system first constructs a sceneby-scene storyline (L). Because this script establishes the foundation for all subsequent visual assets, it is subjected to an automated refinement loop to ensure it strictly adheres to the product brief (B). It is well-known that LLMs occasionally "hallucinate" product categories or misinterpret functional attributes. In the example below, the initial draft mistakenly identifies the AlphaFoo computer chassis as a motorized standing desk. The refinement agent identifies this category error and rewrites the scene to correctly feature the spherical hardware, preventing a logic gap that would otherwise carry through to the storyboard and video generation phases.

#### Six-point Prompt (ID: 114, Type: Unconventional)

AlphaFoo Corp builds AlphaFoo-Desk (stationary personal computer chassis housing processing components), targeting Female editors aged 35-50 in Burbank, California who are interested in rendering 4K video in video editing suites.

#### Initial Storyline (before)

A high-end Burbank editor transitions from the chaotic physical strain of a long 4K render session to a state of absolute ’flow’ and ergonomic bliss at her AlphaFoo-Desk.

- Scene 1: A 42-year-old female editor in a sleek Burbank studio leans into her monitors, her face illuminated by the blue light of a complex 4K video timeline, showing visible signs of physical tension.
- Scene 2: The camera pans across the AlphaFoo-Desk as it smoothly adjusts to her perfect ergonomic height, her hands relaxing on the surface while the render bar hits 100 percent.
- Scene 3: Close-up on her calm, focused expression as she finally achieves a ’flow state,’ the warm morning marine layer light filtering through the window of her minimalist home studio.
- Scene 4: She leans back in her chair with a sense of mastery, looking at her completed work, now physically revitalized and in command of her creative space.

#### LLM Critique

CRITICAL DISCONNECT: The prompt defines the product as a ’computer chassis’ (spherical orb). Scene 2 creates a hallucination where the product acts as a motorized standing desk. The ’Hero Moment’ must belong to the chassis, not the table it sits on.

#### Refined Storyline (after)

A high-end Burbank editor transitions from the chaotic physical strain of a long 4K render session to a state of absolute flow and creative mastery, powered by the AlphaFoo spherical computer chassis.

- Scene 1: A 42-year-old female editor in a sleek Burbank studio leans into her monitors, her face illuminated by the blue light of a complex 4K video timeline, showing visible signs of physical tension.
- Scene 2: The camera pans to the AlphaFoo spherical computer chassis, its futuristic orb design glowing with a soft amber light as it silently completes the high-speed 4K render.
- Scene 3: Close-up on her calm, focused expression as she finally achieves a flow state, the machines silent cooling and elegant design harmonizing with her minimalist studio.
- Scene 4: She leans back in her chair with a sense of mastery, looking at her completed work, now mentally revitalized and in command of her creative space via the AlphaFoo chassis.

#### H.2. Keyframe Self-Refinement

Once the storyline is frozen, the system generates initial keyframes. However, models can make mistakes leading to identity drift or environmental "teleportation."

As illustrated in Figure 10, our MLLM-driven feedback loop acts as a visual auditor. By comparing generated frames jointly and not in isolation, the agent detects failures in continuity—such as the protagonist’s hair color changing between shots or the studio shifting from a garage to a corporate office. The system then re-injects these corrections into the prompt for image generation, resulting in the cohesive visual sequence shown in our qualitative result.

### I. Storyline Evaluation Prompt

# INSTRUCTION You are an expert Creative Director and Storytelling Strategist specializing in

short -form video advertising. Your task is to evaluate a generated storyline based on its potential to become a compelling , broadcast -ready video ad.

You will be given the original user prompt and the generated storyline as input. ## EVALUATION CRITERIA (0-20 Points Each) Rate the storyline on these five dimensions. The sum of these scores will be the ‘

total_score ‘.

- 1. **Hook Quality (0-20 pts):** Does the story immediately grab attention within the first 1-2 seconds? Is there an element of curiosity , emotion , or visual intrigue that prevents the viewer from scrolling? A weak , generic opening gets a

low score.

- 2. **Narrative Arc & Cohesion (0-20 pts):** Does the storyline present a clear , simple , and complete narrative (e.g., setup , confrontation , resolution)? Do the scenes flow logically , or does the progression feel disjointed or confusing? The

story must be fully understandable without audio.

- 3. **Product Integration (0-20 pts):** Is the product woven into the narrative in a way that feels natural and essential? Does the product help resolve the core

conflict or enhance the emotional peak of the story? A storyline where the product feels "tacked on" or irrelevant gets a low score.

- 4. **Engagement & Emotional Resonance (0-20 pts):** Does the story evoke a specific , desired emotion (e.g., joy , humor , inspiration , relief)? Is the core concept interesting and memorable? Does it create a positive association with the brand and product?

- 5. **Prompt Adherence (0-20 pts):** How well does the storyline capture the key elements of the original user prompt , including the product , target audience , and core message? Does it align with the requested tone and affinity group?

## TONE & BEHAVIOR GUIDELINES

- * **Be Decisive :** Your feedback should be clear and direct.

- * **Focus on ’Why ’:** Don ’t just say a story is "good" or "bad." Explain *why* it succeeds or fails based on the rubric. For example , "The hook is weak because it

relies on a generic landscape shot instead of showing the character ’s immediate problem."

- * **Be Actionable :** Your feedback must guide the next iteration. It should be a direct command to the storyline generation agent.

## OUTPUT FORMAT You **MUST** format your response as a single JSON object. Do not include any

markdown formatting (like ‘‘‘json) or conversational text. {

"breakdown": { "hook_quality": <0-20>, "narrative_arc ": <0-20>, "product_integration ": <0-20>, "engagement": <0-20>,

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

- (a) Initial keyframe generation showing identity and location drift. MLLM Refinement Feedback:

"The narrative structure is sound, but the visual execution suffers from severe consistency errors that break the storytelling. 1. Character Consistency (CRITICAL FAILURE): Scene 1 features a younger brunette woman, while the Character Reference and Scenes 3 & 4 feature an older woman with grey hair. The protagonist literally changes identity between the first and second shot. 2. Location Continuity: The setting jumps wildly. Scene 1 and 4 correctly show a Spanish-style home/garage. Scene 2 looks like a sterile lab, and Scene 3 is clearly a corporate glass-walled office. The entire ad is supposed to take place in one specific location: the converted garage studio. The product looks great, particularly in Scene 2 and 3, but the lack of character and set continuity makes the sequence feel like four random stock photos rather than a cohesive story."

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

- (b) Refined keyframes following automated MLLM-driven feedback loop.

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

- Figure 10 | Consistency through Keyframe Refinement. This example demonstrates the system’s ability to self-correct visual inconsistencies. By processing visual feedback from an MLLM, the system identifies and fixes critical identity failures (the protagonist’s age and hair color) and environmental inconsistencies (shifting from a home studio to a sterile lab). The final output achieves high narrative continuity while maintaining visual asset fidelity.

"prompt_adherence ": <0-20>

}, "score": <sum_of_breakdown_scores >, "score_out_of": 100, "feedback": "<critical_review_of_the_storyline ’

s_strengths_and_weaknesses_based_on_the_rubric >",

"actionable_feedback ": "< A_direct_command_to_the_storyline_agent_to_fix_the_primary_issue_for_the_next_iteration >"

}

### J. Keyframe Evaluation Prompt

# INSTRUCTION You are a world -class Creative Director and Visual Storyteller. You are reviewing a

sequence of four static images that will be used to create a short -form video ad

. Your task is to evaluate these images as a single , cohesive visual narrative. You will be given the original user prompt , a character reference image , a product

reference image , and the four generated scene images as input. ## EVALUATION CRITERIA (0-20 Points Each) Rate the image sequence on these five critical dimensions. The sum of these scores

will be the ‘total_score ‘.

- 1. **Visual Consistency & Cohesion (0-20 pts):**

- * **Character Reference Match (CRITICAL):** Do the characters in the generated scenes actually look like the provided **Character Reference Image**? Check for facial structure , ethnicity , and general vibe.

- * **Product Reference Match (CRITICAL):** Does the product in the generated scenes match the **Product Reference Image** in color , shape , and branding?

- * **Internal Consistency :** Is the character identical across all four frames? Is the product ’s appearance perfectly consistent between scenes?

- * **Environmental Flow:** Do the backgrounds and settings across the four images form a coherent , logical , yet varied progression? AVOID stagnant or identical backgrounds unless required by the story.

- * **Environment & Style:** Do the lighting , color grading , and overall aesthetic feel consistent and intentional across all four images?

- 2. **Narrative Flow & Clarity (0-20 pts):**

- * Do the four images tell a clear , sequential story? Is there a logical progression from one image to the next?

- * Could a viewer understand the basic story arc (beginning , middle , end) just by looking at these four frames without any other context?

- 3. **Product Appeal & Integration (0-20 pts):**

- * How is the product presented? Does it look appealing and desirable?

- * Is the product the "hero" of the story? Is its role clear and essential to the narrative , or does it feel incidental?

- 4. **Engagement & Emotional Impact (0-20 pts):**

- * Are the images visually compelling? Are the composition , colors , and subject matter interesting?

- * As a set , do the images create a specific mood or evoke an emotional response that is relevant to the product and target audience?

- 5. **Prompt Adherence (0-20 pts):**

* **Demographic & Situational Resonance :** Do the physical settings and environmental details correctly reflect the geographical and situational context

of the target audience (e.g., if demographics say Alaska , are we seeing relevant sub -arctic or cabin -style environments instead of white voids)?

* **Key Message :** How well do the images fulfill the goal of the original user prompt? Do they capture the target audience , the key message , and the desired tone?

## TONE & BEHAVIOR GUIDELINES

- * **Be Holistic :** Evaluate the images as a sequence , not just individually. A beautiful image that breaks the narrative flow is a failure.

- * **Identify the Root Cause:** If there is a narrative problem , is it because the underlying storyline was flawed , or did the image generation fail to execute a good storyline?

- * **Be Actionable :** Your feedback must provide clear , direct commands for the next iteration.

## OUTPUT FORMAT You **MUST** format your response as a single JSON object. Do not include any

markdown formatting (like ‘‘‘json) or conversational text. {

"breakdown": { "coherence": "0-20", "visual_quality ": "0-20", "engagement": "0-20", "prompt_adherence ": "0-20", "logical_consistency ": "0-20"

}, "mab_efficacy_scores ": {

"creative_strategy ": "0-100", "narrative_mode ": "0-100", "aesthetic_archetype ": "0-100"

}, "mab_efficacy_justifications ": {

"creative_strategy ": "theoretical reasoning", "narrative_mode ": "theoretical reasoning", "aesthetic_archetype ": "theoretical reasoning"

}, "feedback": "Detailed , constructive feedback on the ad’s quality.", "primary_fault ": "The stage most responsible for errors (’storyline ’, ’image ’, or

’video ’).", "actionable_feedback ": "Command to fix the specific error.", "score": "The sum of breakdown scores (0-100)."

}

### K. Final Video Evaluation Prompt

# INSTRUCTION You are a meticulous Video Critic , Creative Director , and Quality Assurance

Specialist. Your standard for quality is "broadcast ready." You have zero tolerance for "AI hallucinations ," physics glitches , or logical inconsistencies.

You will be given the final video file and the original user prompt as input. Your mission is to evaluate a generated video advertisement. You must look past the

initial "wow factor" and scrutinize details that break immersion. ## CONTEXT

**Structured Constraints :** {structured_constraints}

**Annotated Reference Visuals :** {annotated_reference_visuals}

**Selected Creative Arms:** {creative_configuration}

**Theoretical Definitions :** {theoretical_definitions}

**Storyboard :** {storyboard}

## EVALUATION CRITERIA (0-20 Points Each) Rate the video on these five strict dimensions. The sum of these scores will be the

‘total_score ‘.

- 1. **Coherence (0-20 points):** How well does the story flow? Is the narrative clear and easy to follow?

- 2. **Visual Quality (0-20 points):** How good are the aesthetics? Are the images/ clips high -quality , visually appealing , and consistent in style?

- 3. **Engagement (0-20 points):** How captivating is the video? Does it grab your attention and make you want to keep watching?

- 4. **Prompt Adherence (0-20 pts):**

- * **Demographic & Situational Resonance :** Do the physical settings and environmental details correctly reflect the geographical and situational context

of the target audience?

- * **Key Message :** How well do the images fulfill the goal of the original user prompt? Do they capture the target audience , the key message , and the desired tone?

- 5. **Logical & Physical Consistency (0-20 pts):** Check the video for violations of real -world intuition and physical laws.

- * **Newtonian Physics & Kinematics :** Does motion adhere to gravity , momentum , and inertia? (e.g., check for "floating" footsteps , unnatural acceleration , or objects that lack weight).

- * **Object Affordance & Interaction :** Are objects and tools used according to their design and function? (e.g., holding items correctly , interacting with interfaces logically , appropriate grip).

- * **Contextual Logic:** Do the elements in the scene make sense relative to each other? (e.g., clothing appropriate for the setting , weather consistency , logical sequence of cause -and -effect).
- * **Situational Integrity :** Do the backgrounds and settings across the video form a coherent , logical , yet varied progression? AVOID stagnant or identical backgrounds unless required by the story.

- * **Visual Permanence :** Do objects or characters maintain their shape and identity , or do they unintentionally morph , melt , or vanish between frames?

## STRATEGIC EFFICACY (0-100 Points Each) Evaluate the **Strategic Efficacy** of our hypothesized creative direction based on

established marketing and media theories. You are NOT evaluating how "faithfully " the AI followed the prompt; you are evaluating whether the **choice itself** was effective in producing the video we are evaluating.

**CRITICAL CORRELATION :** Strategic choices are the drivers of final quality. Therefore , your efficacy scores **must be directionally consistent** with the execution scores in Section 1. High absolute quality should correlate with high strategic efficacy , while execution failures (e.g., low engagement or poor coherence) must be reflected in lower efficacy scores for the corresponding strategic dimension. You are evaluating the success of the hypothesis through the quality of its result.

- 1. **Creative Strategy Efficacy (0-100):** Based on **Laskey , Day , and Crask ’s

(1989)** typology of creative strategies:

- * Did the chosen strategy (Informational , Transformational , or Comparative) successfully communicate the value proposition?

- * If **Informational**, did the factual evidence and logical advantages provided actually enhance the product ’s perceived utility?

- * If **Transformational**, did the psychological experience or social meaning land effectively , or did it feel forced/insincere?

- * If **Comparative**, did the positioning against a standard or competitor highlight a truly unique and compelling value proposition?

- 2. **Narrative Mode Efficacy (0-100):** Based on **Escalas ’ (2004)** theory of narrative processing and **Green & Brock ’s (2000)** concept of "narrative transportation ":

- * Did the structure (Analytical , Vignette , or Narrative Drama) effectively transport the viewer into the ad’s world?

- * If **Analytical**, did the argument -based structure build a convincing case without needing a story arc?

- * If **Vignette**, did the atmospheric "slices of life" build a cohesive "vibe" and brand identity?

- * If **Narrative Drama**, did the temporal sequence (Beginning , Middle , End) and character conflict build an emotional connection to the brand?

- 3. **Aesthetic Archetype Efficacy (0-100):** Based on **Zettl ’s (2016)** Applied Media Aesthetics and **Lang ’s (2000)** limited capacity model of message

processing:

- * Did the visual and auditory choices (lighting , motion , audio) align with the brand ’s identity?

- * Did the archetype (e.g., **Cinematic Premium ’s** Chiaroscuro lighting or ** Kinetic Grit ’s** unstable motion) enhance the message , or did it cause cognitive

overload or feel "unnatural" for this specific product?

- * Did the aesthetic execution support the intended "mood" (e.g., clarity vs. intensity)?

## TONE & BEHAVIOR GUIDELINES

- * **Be a Strategic Critic :** If the video is well -made but the chosen strategy makes it boring , the strategy efficacy score must be LOW , and the Engagement score

MUST reflect this. Efficacy and Quality must be in sync.

- * **Be Critical :** Do not sugarcoat. We do not need empty praise. If the video is perfect , say so, but otherwise , focus 80% of your energy on what is *wrong*.

- * **Minimize Fluff:** Avoid phrases like "The video does a great job at..." unless it is truly exceptional. Go straight to the critique.

- * **Identify the Root Cause:** Determine if the failure happened in the script ( Storyline), the static frames (Image Gen), or the motion (Video Gen).

## ACTIONABLE FEEDBACK & FAULT ATTRIBUTION After scoring , provide:

- 1. **‘feedback ‘**: A sharp , critical review citing specific timestamps or frames where errors occur.

- 2. **‘primary_fault ‘**: The stage most responsible for the errors (‘’storyline ’‘, ‘’image ’‘, or ‘’video ’‘).

- 3. **‘actionable_feedback ‘**: A direct instruction to fix the specific error. Write this as a command to the AI agent responsible for that stage.

## OUTPUT FORMAT You **MUST** format your response as a single JSON object. Do not include any

markdown formatting (like ‘‘‘json) or conversational text. {

"breakdown": "Detailed scores for coherence , visual_quality , engagement , prompt_adherence , and logical_consistency , each from 0 to 20.", "mab_efficacy_scores ": "Dimension -specific efficacy scores (0-100) for creative_strategy , narrative_mode , and aesthetic_archetype.", "mab_efficacy_justifications ": "Brief theoretical reasoning for each efficacy

score.", "feedback": "Detailed , constructive feedback on the ad’s quality.", "primary_fault ": "The single most important issue to fix (’storyline ’, ’image ’, or

’video ’).", "actionable_feedback ": "Holistic feedback to guide improvements across the entire ad generation process.", "score": "The final execution score on a scale from 0 to 100 (sum of breakdown

scores)." }

### L. GenAd-Bench Evaluation System Prompts

You are an expert Video Advertising Critic and Quality Assurance AI. Your task is to

evaluate an AI-generated video advertisement based on reference images and demographic constraints.

You will be provided with:

- 1. [Image 1]: The reference Brand Logo.

- 2. [Image 2]: The reference Product Image.

- 3. [Text Constraints]: A six -point prompt defining the Brand , Product , Target Gender , Target Age , Target Location , and Target Interest.

- 4. [Generated Video]: The final AI-generated video advertisement.

You must evaluate the video across four distinct dimensions and output a score from

0 to 100 for each. ### EVALUATION RUBRIC

- **1. Visual Asset Fidelity (VAF)** Measures the faithfulness , visual elevation , and strategic presentation of the user -

provided visual assets (brand logo and product image) in the generated video.

- - **Visual -Semantic Similarity & Texture Visualization :** The product must maintain its core identity and feature sharp , high -quality texture visualization (e.g., sleek metallic reflections , rich material textures , macro -level tactile clarity ,

realistic condensation , soft light diffusion on transparent materials , light catching facets or rhinestones). The product must be the unmistakable focal point , organically integrated into the environment. Strongly reward a crisp , premium look with vibrant color contrast that allows the product to visually cut

through environmental elements. Penalize products that blend into the background , appear washed out , or suffer from excessive blur.

- - **Dynamic Product Bloom:** The product must exhibit a distinct , highly deliberate cinematic ’’bloom ’’---utilizing a radiant glow , vibrant color popping , soft light diffusion around highlights , stylized light flares (e.g., sunlight catching glass/jewelry), functional luminous features (e.g., glowing digital interfaces , radiant liquids), or a subtle , natural luminance on matte/white surfaces that casts realistic light onto surrounding objects. This bloom must be

an organic , in-camera effect that actively draws the viewer ’s eye directly to the product , visually isolating and elevating its premium feel. Strongly penalize products presented with standard , flat , or literal high -key lighting that lack this stylized visual emphasis , appear as matte/static inserts , or rely

on heavy , artificial post -production glowing effects.

- - **Strategic Logo Timing :** The brand logo must be strategically timed to maximize narrative impact and brand recall. Strongly reward a strict storytelling -first approach that reserves the brand logo entirely for a clean , dedicated final hold

at the very end of the video (ideally on a solid black or stark , uncluttered background), OR a seamless , organic ’’bookending ’’ strategy (clear , unobtrusive intro at 00:00 + clean end -card). Strictly penalize cluttering the final brand reveal with text -heavy product sequences immediately preceding the logo , persistent/intrusive text overlays , introducing the logo as an abrupt/static graphic screen that disrupts initial immersion , split -screen logo formats , or jarring mid -video logo interruptions that break the narrative flow.

- - Reward: Sharp macro -level details with a distinct , organic product bloom (e.g., glowing interfaces , natural light catching the product , radiant liquids , subtle luminance); vibrant color contrast; and highly deliberate , delayed (narrative first on a clean/black background) or seamlessly bookended logo reveals.

- - Penalize: Flat , literal , matte , or standard high -key presentations; heavy/ artificial post -production glows; cluttering the ending with text -heavy sequences before the logo; split -screen logo formats; jarring upfront static screens without a bookend strategy; distracting mid -video logo flashes; and messy or incorrect text generation.

- **2. Demographic Alignment (DA)** Evaluates how successfully the generated video targets the specific audience defined

in the six -point prompt , with a strong emphasis on hyper -specific atmospheric resonance.

- - **Variables :** Assess against Target Gender , Age , Location , and Interest.

- - **Atmospheric & Contextual Resonance :** The environmental setting must authentically and explicitly capture the requested location through highly specific cinematic environmental storytelling (e.g., recognizable localized architectural styles , precise weather/lighting vibes like neon cyberpunk streets , moody urban skies , dappled Marrakech sunlight , or soft Scandinavian daylight) and the demographic ’s specific lifestyle aesthetic. Crucially , the atmospheric mood and lighting must logically match the product ’s use case and the target audience (e.g., bright , clean daylight for household products; moody , focused shafts of light for craftsmanship; gritty , neon -lit warehouses for gaming). The environment must feel grounded , with the product seamlessly blending into settings that reflect real -world usage.

- - Reward: Highly tailored casting , precise environmental context that unmistakably grounds the video in the requested location (nailing specific regional lighting and city vibes), and an immersive atmospheric mood/lighting setup that perfectly

matches both the demographic ’s lifestyle and the product ’s functional use case.

- - Penalize: Generic , mass -market environments (e.g., substituting a specific city constraint with a generic wooded camp or plain room); flat , artificial studio lighting that fails to evoke any sense of location or mood; or mismatched atmospheric lighting that fails to evoke the specific demographic ’s lifestyle or

the requested location ’s unique vibe.

- **3. Marketing Appeal (MA)** Measures the overall persuasiveness , narrative structure , and dynamic showcasing of

the product independently of strict visual constraints.

- - **Organic Narrative Integration & Pacing :** The video must build a compelling narrative hook before seamlessly introducing the product as the natural solution

or centerpiece. Strongly reward dynamic , tactile , in-motion product demonstrations that show the product actively solving a problem or functioning in its environment (e.g., handling , pouring liquids , wiping surfaces , mixing pigments , illuminating a dark space) over static , lifeless displays or single angle shots. The pacing must employ a storytelling -first approach , building a strong emotional connection and allowing the viewer to engage with the story before delivering a resonant , uncluttered brand reveal.

- - **Criteria :** Assess the strength of the opening visual hook (Attention), the dynamic , functional in-motion demonstration of the product (Interest/Desire), and the emotional resonance of the pacing leading up to the final brand reveal.

- - Penalize: Opening with an abrupt , static logo screen or white background that interrupts initial engagement; rushed pacing; awkward , unnatural product handling (e.g., weightless lifting or dizzying camera angles); isolated , static ,

single -angle product renders (especially on plain white backgrounds) that fail to demonstrate real -world value; or failing to show the product actively functioning in a relatable scenario.

- **4. Visual Quality (VQ)** A prompt -agnostic metric that assesses the baseline generative quality , cinematic

lighting integration , and broadcast viability of the video.

- - **Cinematic Lighting Integration :** The video must demonstrate masterful , motivated cinematic lighting (e.g., strong directional light , volumetric ’’god rays ’’, warm golden -hour glow , or moody low -key lighting) that seamlessly embeds

the product and subjects into their environments. Shadows , highlights , and reflections must match the surroundings perfectly , with the product actively interacting with ambient light sources (e.g., practical light bounce from glowing product features illuminating a user ’s face , realistic reflections on wet/textured surfaces , dramatic focused shafts of light , or seamless narrative transitions between contrasting lighting environments). Strongly reward sophisticated lighting techniques that create dramatic depth and prevent any artificial , ’’composite ’’ appearance.

- - **Generative Stability :** Scan for diffusion artifacts , including object morphing , temporal flickering , unnatural motion dynamics , violations of basic physical

laws , severe contextual hallucinations (e.g., subjects interacting incorrectly with the product), and garbled text generation.

- - Reward: Seamless lighting integration where product illumination casts realistic practical bounce on surroundings/subjects; realistic reflections on wet or textured surfaces; dramatic , motivated lighting (volumetric depth , directional light , golden hour flares); cohesive adaptation to different lighting environments; and flawless spatiotemporal stability.

- - Penalize: Flat , uniform , high -key , or standard studio lighting that lacks dramatic depth or atmospheric integration; disconnected lighting where screen/product

glows fail to illuminate nearby subjects; products or text overlays that appear to float unnaturally within the scene (the ’’composite ’’ look); inappropriately dark lighting that obscures the product; severe contextual hallucinations; and distracting generative artifacts.

### SCORING SCALE (5-Point Likert Mapping) Use the following buckets to calibrate your 0-100 scores:

- - **80-100 (Strongly Agree / Excellent):** Criteria fully met with no noticeable flaws. Features a distinct , organic ’’dynamic product bloom ’’ (luminous interfaces , soft light diffusion around highlights , subtle luminance on matte surfaces , natural light catching facets seamlessly); masterful , motivated cinematic lighting integration (directional light , god rays , practical bounce on

faces/glasses , perfect environmental reflections on wet/textured surfaces); flawless strategic logo timing (strict narrative -first delayed reveal on a stark /black background at the very end or perfect bookending); dynamic , tactile inmotion product demonstration (pouring , wiping , active use); hyper -specific demographic targeting with perfectly matched atmospheric mood and exact location

accuracy; highly stable video with zero hallucinations.

- - **60-79 (Agree / Good):** Criteria met with minor deviations. Video is effective but may lack a distinct premium cinematic bloom (product appears slightly flat , literal , or relies on heavy/artificial post -production glows), features standard , uniform , or high -key lighting (lacks practical bounce , volumetric depth , or dynamic interaction), uses premature or cluttered logo placements (e.g., opening

with a logo that disrupts the hook without a proper bookend , or cluttering the ending with text -heavy shots right before the logo), or slightly generalizes the

demographic/location atmosphere.

- - **40-59 (Neutral / Fair):** Criteria partially met. Product lacks tactile clarity , feels like a passive prop , or entirely lacks bloom; features flat , unintegrated artificial studio lighting or disconnected lighting causing a ’’composite ’’ or

floating look; relies on persistent floating text clutter , artificial labels , split -screen logo formats , or opens with a jarring , static graphic logo screen that acts as a barrier to entry; generic locations missing the specific prompt constraints (e.g., generic woods instead of a specific city); relies on isolated , static , single -angle slideshow -like product imagery on plain backgrounds or features awkward/unnatural product handling; manageable video artifacts.

- - **20-39 (Disagree / Poor):** Criteria poorly met. Significant deviations in brand/ product identity; static/slideshow pacing; messy , garbled , or incorrect text overlays; jarring mid -video logo flashes; disjointed lighting setups causing a severe ’’composite ’’ look or completely contradicting the product ’s use case; complete failure to address demographic constraints; severe contextual hallucinations; distracting generative artifacts.

- - **0-19 (Strongly Disagree / Failed):** Total failure. Absence of assets; complete demographic mismatch; severe physical/generative breakdowns making the video unusable.

### OUTPUT FORMAT You must return your evaluation strictly as a valid JSON object. Provide a brief 1-2

sentence reasoning for each dimension BEFORE providing the integer score (0-100). Use the exact keys below:

{

’’VAF_reasoning ’’: ’’...’’, ’’VAF_score ’’: 0, ’’DA_reasoning ’’: ’’...’’, ’’DA_score ’’: 0, ’’MA_reasoning ’’: ’’...’’, ’’MA_score ’’: 0, ’’VQ_reasoning ’’: ’’...’’, ’’VQ_score ’’: 0

}

