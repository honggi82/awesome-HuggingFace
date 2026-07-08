## Mind-Brush: Integrating Agentic Cognitive Search and Reasoning into Image Generation

# arXiv:2602.01756v1[cs.CV]2Feb2026

#### Jun He*1 Junyan Ye*13 Zilong Huang1 Dongzhi Jiang4 Chenjue Zhang2 Leqi Zhu3 Renrui Zhang4 Xiang Zhang1 Weijia Li12†

Github: https://github.com/PicoTrex/Mind-Brush

Dataset: https://huggingface.co/datasets/PicoTrex/Mind-Brush

[Figure 1]

Figure 1. We introduce Mind-Brush, an agentic framework that synergizes active search with explicit reasoning for image generation. By decomposing user intent, retrieving multimodal evidence, and inferring latent requirements, our agent effectively bridges the cognitive gaps and interpretative biases prevalent in existing models. Furthermore, we propose Mind-Bench, a comprehensive benchmark designed to evaluate model performance on up-to-date long-tail concepts and multimodal reasoning tasks, thereby probing the boundaries of unified understanding and generation capabilities.

### Abstract

While text-to-image generation has achieved unprecedented fidelity, the vast majority of existing models function fundamentally as static text-topixel decoders. Consequently, they often fail to grasp implicit user intentions. Although emerging unified understanding-generation models have improved intent comprehension, they still struggle to accomplish tasks involving complex knowledge

*Jun He and Junyan Ye contribute equally to this work.

1Sun Yat-sen University 2Tsinghua Shenzhen International Graduate School, Tsinghua University 3Shanghai Artificial Intelligence Laboratory 4MMLab, The Chinese University of Hong Kong. Correspondence to: Weijia Li <liweij29@mail.sysu.edu.cn>.

Preprint. February 3, 2026.

reasoning within a single model. Moreover, constrained by static internal priors, these models remain unable to adapt to the evolving dynamics of the real world. To bridge these gaps, we introduce Mind-Brush, a unified agentic framework that transforms generation into a dynamic, knowledgedriven workflow. Simulating a human-like ’thinkresearch-create’ paradigm, Mind-Brush actively retrieves multimodal evidence to ground out-ofdistribution concepts and employs reasoning tools to resolve implicit visual constraints. To rigorously evaluate these capabilities, we propose Mind-Bench, a comprehensive benchmark comprising 500 distinct samples spanning real-time news, emerging concepts, and domains such as mathematical and Geo-Reasoning. Extensive ex-

periments demonstrate that Mind-Brush significantly enhances the capabilities of unified models, realizing a zero-to-one capability leap for the Qwen-Image baseline on Mind-Bench, while achieving superior results on established benchmarks like WISE and RISE.

### 1. Introduction

Recent advancements in image generation have democratized visual creation, enabling the seamless translation of imagination into high-fidelity imagery (Rombach et al., 2022; Esser et al., 2024; Gong et al., 2025). However, fundamentally, the vast majority of existing models function primarily as static text-to-pixel decoders (Podell et al., 2023; Cai et al., 2025). Confined to mapping explicit user instructions to pixels, these models often fail to grasp implicit, high-level user intentions, thereby diverging significantly from the human artistic creation process. While emerging unified multimodal understanding-generation models, such as GPT-Image (OpenAI, 2024) and Bagel (Deng et al., 2025), demonstrate promising capabilities in comprehending user intent and incorporating world knowledge, their performance remains constrained in tasks demanding complex mathematical or knowledge-intensive reasoning. This limitation suggests that monolithic architectures may struggle to encompass the full spectrum of capabilities required for such intricate, end-to-end tasks.

Moreover, constrained by the temporal knowledge cutoff inherent in pre-training data, the cognitive boundaries of current image generation models remain static. Consequently, they struggle to adapt to the evolving dynamics of the real world, resulting in significant capability gaps when handling real-time news or novel IP concepts (Son et al., 2025; Li et al., 2025a). In the realm of Large Language Models (LLMs), researchers have successfully transcended these boundaries by integrating retrieval capabilities through agentic designs (Chen et al., 2024; Yu et al., 2024), as exemplified by Search-o1 (Li et al., 2025c). While recent proprietary models such as Nano Banana Pro (Comanici et al., 2025) and FLUX-2 Max (Black Forest Labs, 2026b) have demonstrated integrated search and reasoning capabilities, a significant gap persists within the open-source community. Specifically, there is a notable absence of open-source models capable of interacting with the open world, performing complex reasoning, and executing active planning.

Recently, agentic approaches for image generation have emerged (Jiang et al., 2026), such as T2I-Copilot (Chen et al., 2025a) and PromptSculptor (Xiang et al., 2025), which focus on elaborating concise instructions into detailrich descriptions. Think-Then-Gen (Kou et al., 2026) advances this further by leveraging LLMs to decompose user

queries into sequential drawing steps. Nevertheless, these efforts remain largely confined to standard T2I benchmarks like GenEval++ (Ye et al., 2025a), prioritizing prompt refinement over intricate cognitive tasks such as mathematical derivation or commonsense reasoning. Crucially, due to the absence of external tools, their reasoning relies solely on internalized knowledge. Consequently, these methods falter when tasks demand factual verification of real-time events or evolving contexts.

To bridge this gap, we introduce Mind-Brush, a unified agentic framework that shifts image generation from a static mapping paradigm to a dynamic, knowledge-driven workflow. Rather than treating generation as single-step inference, Mind-Brush orchestrates a cognitive process: it actively retrieves multimodal evidence to ground out-ofdistribution concepts and employs logical reasoning to deduce implicit visual constraints, thereby realizing the unification of agentic understanding and generation. By effectively simulating the human artist’s “Think-Research-Create” workflow, Mind-Brush enables high-fidelity generation requiring real-time knowledge and handles tasks involving complex reasoning.

Furthermore, existing image generation benchmarks, such as GenEval (Ghosh et al., 2023) and ImgEdit (Ye et al., 2025b), primarily prioritize the evaluation of instruction following. While datasets like WISE (Niu et al., 2025) and RISE (Zhao et al., 2025) extend this scope to probe internal knowledge recall and rudimentary reasoning, they fall short of evaluating capabilities that require active information retrieval and complex reasoning. To address this limitation, we introduce Mind-Bench, a comprehensive benchmark specifically designed to assess generative performance under conditions necessitating complex reasoning and external knowledge acquisition. As illustrated in Figure 1, MindBench encompasses 500 distinct samples across 10 diverse categories, spanning challenging scenarios from real-time news and emerging IP concepts to complex mathematical and geographical reasoning. This benchmark fills a critical void in evaluating image synthesis tasks that demand realtime knowledge and deep reasoning, particularly for unified understanding-generation models.

Our main contributions are summarized as follows:

- 1. We propose Mind-Brush, a novel agentic framework that unifies intent analysis, multi-modal search, and knowledge reasoning to enable a ’think-researchcreate’ paradigm for image generation.
- 2. We propose Mind-Bench, a benchmark tailored to evaluate generative capabilities involving dynamic external knowledge and complex reasoning. Experimental results reveal critical limitations in current unified multimodal models regarding real-time awareness and

- logical deduction.
- 3. Mind-Brush substantially elevates the accuracy of the Qwen-Image baseline from 0.02 to 0.31 on MindBench, while significantly outperforming existing baselines on established benchmarks including the knowledge-driven WISE (+25.8 % WiScore) and reasoning-driven RISEBench (+27.3 % Accuracy).

### 2. Related Work

#### 2.1. Agent for Image Generation

Multimodal LLMs are increasingly serving as agentic decision-makers to align vague user intents with precise image synthesis(Yan et al., 2025; Zhang et al., 2025; Su et al., 2025). A primary stream focuses on prompt optimization: T2I-Copilot (Chen et al., 2025a) and PromptSculptor (Xiang et al., 2025) employ multi-agent collaboration to refine concise instructions into detailed descriptions, while ImAgent introduces test-time policy scaling for semantic alignment. Parallel works target precise control: MCCD (Li et al.,

- 2025b) utilizes agents to decouple multi-object attributes, while AgentStory (Zhou et al., 2025) and CREA(Venkatesh et al., 2025) ensure narrative consistency and creative editing, respectively. To transcend static knowledge boundaries, Think-Then-Generate introduces a ”think-then-generate” paradigm to explicitize internal visual logic. Concurrently, World-to-Image (Son et al., 2025) and IA-T2I (Li et al.,

- 2025a) attempt to employ simple image retrieval as visual cues to supplement out-of-distribution (OOD) concepts. However, current solutions remain fragmented. Reasoningbased methods are confined by closed training data, failing on real-time events; conversely, retrieval-based approaches often treat external evidence as shallow visual cues without deep logical integration. A unified workflow synergizing active multimodal search with explicit reasoning remains absent, limiting performance on complex, knowledgeintensive tasks.

#### 2.2. Image Generation Model

The ultimate goal of Unified Multi-Modal Models (UMMs) is to consolidate cross-modal understanding and generation within a single architecture. Pioneering works (e.g., Chameleon (Team, 2024), Emu3 (Wang et al., 2024)) integrated image generation into the LLM paradigm via visual signal discretization. However, their reliance on VQ-VAE (Van Den Oord et al., 2017) introduced lossy compression, which fundamentally restricted generation fidelity. Subsequent research (e.g., Transfusion (Zhou et al., 2024), Show-o (Xie et al., 2024)) attempted to mitigate this by unifying autoregressive text prediction and bidirectional image diffusion within a shared Transformer backbone. Yet, this approach triggered irreconcilable modal conflicts. Even

with recent architectural innovations—such as Mixture-ofTokens (MoT) or Mixture-of-Experts (MoE) employed in Bagel and OneCat—balancing the dual objectives of understanding and generation remains a formidable challenge. Consequently, state-of-the-art methods (e.g., OmniGen2 (Wu et al., 2025b), BLIP-o3 (Chen et al., 2025b)) have shifted toward a decoupled strategy, utilizing powerful multimodal large language models (MLLMs) to guide external diffusion heads, thereby achieving superior performance.

#### 2.3. Image Generation Benchmarks

Evaluating Unified Multi-Modal Models (UMMs) necessitates a comprehensive assessment of both comprehension and synthesis. Mainstream benchmarks primarily assess text-image alignment and instruction following capabilities. For instance, GenEval (Ghosh et al., 2023) evaluates compositional integrity, quantifying the model’s ability to bind attributes (e.g., counts, positions) explicitly stated in prompts. However, these benchmarks are largely confined to shallow, explicit text comprehension, neglecting deeper conceptual or reasoning capabilities. To address this, subsequent benchmarks incorporate extensive world knowledge. WISE (Niu et al., 2025) and PhyBench (Meng et al., 2024) assess domain-specific knowledge (e.g., culture, physics), while RISEBench (Zhao et al., 2025) focuses on logical reasoning, evaluating the translation of causal and spatio-temporal semantics into visual representations. Nevertheless, existing methods predominantly assess Internalized Parametric Memory and there is a notable absence of benchmarks evaluating multimodal reasoning or Out-of-Distribution (OOD) concepts, failing to distinguish whether a model is merely retrieving stored static knowledge or actively performing reasoning for real-time scenarios.

### 3. Mind-Brush 3.1. Problem Formulation

We formalize the inference workflow of Mind-Brush as a Hierarchical Sequential Decision-Making Process, defined by the tuple M = ⟨S,A,π,E⟩. This framework generates a structured cognitive trajectory to bridge the gap between abstract intent and visual realization.

- • Cognitive State (S): Let st = {I,Iimg,Et} denote the state at step t. It encapsulates the original user inputs (instruction I and optional reference image Iimg) and the dynamic evidence buffer Et, which accumulates retrieved knowledge and reasoning chains.
- • Action Space (A): The set of operators available to the agent. We distinguish between the Meta-Action aplan (Cognitive Gap Detection), which identifies cognitive gaps Qgap, and Execution Actions aexec ∈

[Figure 2]

Figure 2. The overall framework of Mind-Brush. The user input first undergoes intent decomposition to identify potential knowledge deficits and formulate a question list. Based on specific requirements, the system dynamically executes specialized tools—such as utilizing active search or logical reasoning—to effectively bridge cognitive gaps. Finally, the consolidated evidence is organized into a final instruction via a concept review process to guide precise image generation, ensuring alignment with the user’s authentic intent.

{asearch,areason}, which actively acquire multimodal evidence.

• Execution Policy (π): The Intent Analysis module functions as a high-level policy π(aplan|s0). It assesses the initial state to formulate a deterministic execution path based on the identified Qgap.

The inference process evolves as a context-aware trajectory. As shown in Figure 2, the system does not follow a rigid workflow; instead, it dynamically adapts to the user’s request. By evaluating the specific nature of cognitive gaps in the initial state such as factual deficits or logical conflicts, the planner infers the optimal structure for evidence accumulation, routing the execution through specialized Search or Reasoning branches. This effectively aligns the inference computation with the intrinsic complexity of the user intent.

Ultimately, our objective is to generate the optimal target image x∗ based on the final converged state sT. This state contains the consolidated Master Prompt Pmaster and verified visual references Iref, transforming static generation into a dynamic explicit evidence accumulation process.

#### 3.2. Cognitive Gap Detection

User instructions often contain implicit constraints and longtail concepts exceeding the model’s parametric knowledge boundaries. To address this, we introduce the Cognitive Gap Detection strategy, integrated within the Intent Analysis Agent (Aintent) as a meta-planner, to bridge this cognitive divide. Specifically, it maps the text instruction I and optional image Iimg into a structured semantic space via the 5W1H (What, When, Where, Why, Who, and How) paradigm (Cao et al., 2024), establishing a multimodal

“Ground Truth” to determine signal dominance. Subsequently, the module executes a rigorous gap analysis by detecting specific entities or logical dependencies that require external verification. Information absent from internal knowledge is formalized into a set of explicit atomic questions, denoted as Qgap. Based on the composition of Qgap, the system instantiates a dynamic execution policy π, routing the workflow to the appropriate factual grounding or logical reasoning branch defined in the action space.

#### 3.3. Adaptive Knowledge Completion

To bridge the identified cognitive gaps, Mind-Brush employs a Internal Logical Derivation mechanism. Unlike rigid single-pathway systems, the execution policy π flexibly composes the retrieval and reasoning tools based on the complexity of Qgap.

External Knowledge Anchoring. For gaps involving OOD entities or dynamic events, the framework activates the Cognition Search Agent (Asearch). It first utilizes a Keyword Generator to synthesize the user’s multimodal inputs (I,Iimg) and the identified gaps Qgap, producing precise textual queries Qtxt and initial visual queries Qimg. Upon retrieving factual documents Tref from open-world knowledge bases, the system performs a dual-update operation:

I′ = Inject(I,Tref),Q′img = Calibrate(Qimg,Tref) (1) where the retrieved concepts are injected back into the user instruction (I′) to update the textual context, while simultaneously calibrating visual queries (Q′img) to ensure that subsequently retrieved reference images Iref align with validated facts.

#### Internal Logical Derivation. For gaps requiring com-

plex deduction—such as solving mathematical problems in Iimg or inferring spatial relations from retrieved data—the system triggers the CoT Knowledge Reasoning Agent (Areasoning). This engine functions as a logic processor that ingests the user instruction, the input image, and crucially, the accumulated search evidence (Esearch = Tref ∪ Iref). It performs multi-step reasoning to resolve implicit conflicts or interpret retrieved visual data, producing explicit conclusions Rcot.

The final evidence set E = Esearch ∪ Rcot forms a comprehensive, logically consistent cognitive context for generation.

#### 3.4. Constrained Generation

The accumulation of external information introduces the risk of redundancy or irrelevance. Therefore, the final phase focuses on Information Consolidation and Conditional Synthesis. First, the Concept Review Agent (Areview) serves as a consolidation mechanism to filter noise from the disjointed evidence stream E. It synthesizes the verified facts and logical conclusions with the user’s original creative intent, rewriting them into a structured Master Prompt Pmaster. This prompt explicitly articulates visual attributes that were previously implicit or unknown. Subsequently, the Unified Image Generation Agent (Ageneration) executes the visual synthesis. Distinct from standard T2I models, Ageneration is conditioned on both the text-aligned Pmaster and adaptive visual cues Vin. Specifically, based on user intent, the mechanism dynamically selects between generation and editing modes to determine the visual conditioning source Vin (i.e., from Iref or Iimg). These constraints effectively guide the model to achieve high fidelity to the user’s creative vision while strictly adhering to the factual and logical boundaries established during the knowledge acquisition phase.

### 4. Mind-Bench

#### 4.1. Motivation and Task Definition

Complex image generation transcends simple text-topixel translation, necessitating a “Research-then-Create” paradigm akin to human artistry. However, current evaluation benchmarks often prioritize direct generation capabilities based on static knowledge. While some extend to general world knowledge, they remain relatively simplistic, suffering from a lack of temporal sensitivity and depth in multimodal reasoning. To probe the boundaries of “cognitive generation,” we propose Mind-Bench, a comprehensive benchmark comprising 500 samples designed to objectively evaluate generation capabilities dependent on dynamic external knowledge and user intent reasoning.

To systematically assess these capabilities, we categorize Mind-Bench into two primary clusters covering 10 diverse

[Figure 3]

Figure 3. Overview of Checklist-based Strict Accuracy (CSA) evaluation pipeline in Mind-Bench.

sub-domains, as shown in Figure 1.

Knowledge-Driven Tasks: This category includes five sub-domains: Special Events (e.g., breaking news scenes), Weather (real-time meteorological conditions), Character (specific IP character), Object (long-tail artifacts), and World Knowledge (Common sense in general situations). These tasks comprehensively evaluate the model’s ability to retrieve and integrate external information for precise visual grounding. The core challenge lies in mitigating hallucinations regarding Out-of-Distribution (OOD) entities.

Reasoning-Driven Tasks: This category encompasses: Life Reasoning (common sense inference), Geo Reasoning (spatial and map understanding), Math (geometric and algebraic visualization), Science & Logic (physical states and abstract logic), and Poem (imagery derived from literary metaphor). The core challenge lies in the model’s capacity to deduce implicit constraints from ostensibly simple instructions—determining whether the model can genuinely comprehend the latent reasoning results required for accurate generation. More details of task description can be found in the appendix B.1.

#### 4.2. Benchmark Construction

Mind-Bench is constructed through a rigorous HumanMachine Collaborative Pipeline to ensure multidimensional complexity and factual reliability.

First, distinct from random web-crawling, we recruited 6 graduate students in AI to carefully curate high-difficulty prompts. For each prompt, annotators manually collected strongly correlated multimodal evidence (e.g., official news reports, authoritative reference images) to establish objective factual anchors.

Subsequently, annotators utilized LLMs to generate candi-

- Table 1. Quantitative comparison of different models on Mind-Bench. The table is divided into proprietary (top) and open-source (bottom) models. The best performing model is highlighted in bold. The symbol ‘-’ indicates that the model is not applicable to I2I tasks.

Knowledge-Driven Reasoning-Driven

Model Name

Overall SE Weather MC IP WK SL Poem Life Reason GU Math

GPT-Image-1(OpenAI, 2024) 0.32 0.06 0.22 0.02 0.16 0.32 0.10 0.24 0.10 0.12 0.17 GPT-Image-1.5(OpenAI, 2025) 0.36 0.18 0.22 0.04 0.30 0.34 0.08 0.34 0.10 0.02 0.21 FLUX 2 Pro(Black Forest Labs, 2026b) 0.38 0.12 0.08 0.00 0.20 0.44 0.64 0.18 0.04 0.02 0.21 FLUX 2 Max(Black Forest Labs, 2026a) 0.44 0.12 0.10 0.04 0.38 0.40 0.50 0.20 0.02 0.06 0.23 Nano Banana(Google DeepMind, 2025b) 0.30 0.10 0.12 0.00 0.30 0.32 0.36 0.20 0.04 0.08 0.18 Nano Banana Pro(Google DeepMind, 2025a) 0.50 0.36 0.40 0.16 0.56 0.62 0.68 0.30 0.16 0.46 0.41

SDXL(Podell et al., 2023) 0.04 0.00 0.04 0.00 0.00 0.00 0.00 - - - 0.01 SD-3.5 M(Stability AI, 2024b) 0.02 0.00 0.00 0.00 0.02 0.00 0.00 - - - 0.01 SD-3.5 L(Stability AI, 2024a) 0.04 0.00 0.02 0.00 0.02 0.00 0.06 - - - 0.01 FLUX 1 dev(Labs, 2024) 0.04 0.00 0.00 0.00 0.02 0.02 0.04 - - - 0.02 FLUX 1 Kontext(Labs et al., 2025) 0.02 0.00 0.00 0.00 0.02 0.00 0.00 - - - 0.01 FLUX 1 Krea(Labs, 2024) 0.04 0.00 0.04 0.00 0.02 0.00 0.02 - - - 0.02 Bagel(Deng et al., 2025) 0.02 0.00 0.00 0.00 0.00 0.02 0.02 0.02 0.00 0.08 0.02 Echo-4o(Ye et al., 2025a) 0.04 0.00 0.00 0.00 0.00 0.02 0.06 0.02 0.02 0.02 0.02 DraCo(Jiang et al., 2025b) 0.02 0.00 0.02 0.00 0.00 0.02 0.02 0.04 0.02 0.06 0.02 Z-Image(Cai et al., 2025) 0.02 0.00 0.08 0.02 0.00 0.00 0.00 - - - 0.02 Qwen-Image(Wu et al., 2025a) 0.08 0.00 0.04 0.00 0.00 0.04 0.00 0.04 0.00 0.00 0.02 Mind-Brush (Ours) 0.54 0.16 0.62 0.18 0.40 0.26 0.54 0.10 0.16 0.14 0.31

date fine-grained evaluation checklist items based on the collected evidence. These items underwent strict human verification to eliminate redundancies and ensure executability. This process yields a final set of samples, each equipped with the input instruction, multimodal reference evidence, and a rigorous evaluation checklist.

#### 4.3. Evaluation Criterion

Previous image generation benchmarks typically employ CLIP scores or MLLM ratings to assess generation quality across abstract perceptual dimensions. However, these metrics exhibit significant limitations in accuracy: they predominantly operate at a coarse semantic level (e.g., verifying the presence of a generic object) but fail to distinguish identity-level details or specific factual attributes.

To accurately reflect model usability in complex cognitive tasks, we propose Checklist-based Strict Accuracy (CSA) as the core metric, as illustrated in Figure 3. This criterion employs an MLLM judge to scrutinize the generated image against the checklist under a Holistic Pass Criterion: a sample is deemed correct only if all sub-items are verified as “Pass”. For a dataset with N samples, the accuracy is defined as:

 

  (2)

|Ci|

N

1 N

VQA(Igen(i) ,c(ji)) = 1

I

AccCSA =

i=1

j=1

where I(·) is the indicator function and VQA(I,c) returns 1 if image I satisfies item c. This metric offers a rigorous evaluation, effectively penalizing generations that are partially correct but logically flawed.

### 5. Experiments

#### 5.1. Benchmarks and Evaluation Protocols

To comprehensively evaluate the capabilities of current methods in understanding user intent and generating longtail concepts, we selected three benchmarks with distinct focuses. First, our proposed Mind-Bench assesses generation capabilities dependent on dynamic external knowledge and multi-step reasoning using Checklist-based Strict Accuracy (CSA). We also include WISE, which focuses on complex semantic understanding and world knowledge integration evaluated via WiScore, and RISEBench, which evaluates joint text-image analysis capabilities across four reasoning dimensions (Instruction Reasoning, Appearance Consistency, Visual Plausibility, and Accuracy). Further details regarding the evaluation protocols and implementation can be found in the Appendix A.4.

#### 5.2. Experimental Settings

Baselines. We compare Mind-Brush against current mainstream proprietary UMMs, including GPT-Image1 (OpenAI, 2024), GPT-Image-1.5 (OpenAI, 2025), Nano Banana (Google DeepMind, 2025b), Nano Banana Pro (Google DeepMind, 2025a), FLUX-2 Pro (Black Forest Labs, 2026b), and FLUX-2 Max (Black Forest Labs, 2026a). Additionally, we compare it with state-of-the-art (SOTA) open-source T2I models or UMMs, including FLUX.1 dev (Labs, 2024), FLUX 1 Kontext (Labs et al., 2025), FLUX 1 Krea (Labs, 2024), Stable Diffusion (SD) 3.5 Large (Stability AI, 2024a), SD 3.5 Medium (Stability AI,

- 2024b), SD-XL (Podell et al., 2023), Bagel (Deng et al.,
- 2025), Echo-4o (Ye et al., 2025a), DraCo (Jiang et al.,

- Table 2. Quantitative comparison of different models on WISE and RISEBench. The table is divided into proprietary (top) and open-source (bottom) models. Within each group, the best performing model is highlighted in bold. The symbol ‘-’ indicates that the model is not applicable to I2I tasks.

WISE RISEBench

Model Name

Cultural Time Space Bio Phys Chem Overall Instr. Reas. App. Consis. Vis. Plaus. Accuracy

GPT-Image-1(OpenAI, 2024) 0.81 0.71 0.89 0.83 0.79 0.74 0.80 62.8 80.2 94.9 28.9 Nano Banana(Google DeepMind, 2025b) 0.89 0.87 0.95 0.89 0.89 0.79 0.89 61.2 86.0 91.3 32.8 Nano Banana Pro(Google DeepMind, 2025a) 0.89 0.80 0.89 0.88 0.86 0.85 0.87 77.0 85.5 94.4 47.2

FLUX.1-dev(Labs, 2024) 0.48 0.58 0.62 0.42 0.51 0.35 0.50 26.0 71.6 85.2 1.9 FLUX.1-Canny(Labs, 2024) - - - - - - - 20.2 13.1 77.5 0.0 SD-XL-base(Podell et al., 2023) 0.43 0.48 0.47 0.44 0.45 0.27 0.43 - - - SD-3.5-medium(Stability AI, 2024b) 0.43 0.50 0.52 0.41 0.53 0.33 0.45 - - - SD-3.5-large(Stability AI, 2024a) 0.44 0.50 0.58 0.44 0.52 0.31 0.46 - - - BAGEL (w/ CoT)(Deng et al., 2025) 0.76 0.69 0.75 0.65 0.75 0.58 0.70 45.9 73.8 80.1 11.9 BAGEL(Deng et al., 2025) 0.44 0.55 0.68 0.44 0.60 0.39 0.52 36.5 53.5 73.0 6.1 Qwen-Image(Wu et al., 2025a) 0.62 0.63 0.77 0.57 0.75 0.40 0.62 49.9 71.0 91.5 19.4 GenAgent (Jiang et al., 2026) 0.78 0.67 0.78 0.72 0.77 0.55 0.72 - - - Mind-Brush (Ours) 0.83 0.69 0.84 0.71 0.85 0.68 0.78 61.5 79.4 86.5 24.7

- 2025b), Z-Image (Cai et al., 2025), GenAgent (Jiang et al.,

- 2026) and Qwen-Image (Wu et al., 2025a). All baselines are evaluated using their official default settings.

Implementation Details. To ensure a fair comparison, Mind-Brush maintains consistent experimental settings across all benchmarks. We employ Qwen-Image-Edit-2512 as the image-guided T2I model and Qwen-Image as the prompt-guided T2I model. For the backbone MLLM, we uniformly utilize GPT-5.1 across all agents. Regarding search tools, we use the Google Search API for retrieval, setting the limit for text search results to 2 and truncating web content at 2000 words to ensure sufficient textual evidence without incurring excessive token costs. The limit for image search results is set to 5. All experiments involving open-source models were conducted on 8 NVIDIA A100 80G GPUs.

#### 5.3. Comparison with State of the Art

Table 1 presents the quantitative comparison results on Mind-Bench. Experimental results demonstrate that MindBrush achieves a significant improvement in overall generation accuracy compared to open-source T2I and UMM models, surpassing SD-3.5 Large and Qwen-Image by 30.0% and 29.0%, respectively. Notably, while the baseline accuracy of Qwen-Image on Mind-Bench is considerably lower than that of proprietary methods, our Mind-Brush framework empowers these base models to match or even exceed the performance of several proprietary UMMs. Specifically, in tasks heavily reliant on search, such as Special Events(SE) and Character(MC). In Geo Understanding (GU), which demands cross-modal reasoning, Mind-Brush outperforms the vast majority of proprietary models and matches the performance of Nano Banana Pro. Furthermore, our framework achieves a 47.6% improvement in overall accuracy compared to GPT-Image-1.5. This validates the ef-

Table 3. Ablation Study on Knowledge and Reasoning Modules

KnowledgeDriven

ReasoningDriven Overall

Ablation Setting

Baseline 0.02 0.02 0.02 + Areasoning 0.11 0.21 0.17 + Asearch 0.30 0.20 0.25

Mind-Brush (Ours) 0.38 0.24 0.31

fectiveness of the textual and visual references provided by our retrieval and reasoning modules, achieving a unification of understanding and generation on open-source models through active knowledge acquisition and reasoning. The experimental results also indicate that Mind-Bench poses a significant challenge to current mainstream models.

Mind-Brush also delivers outstanding performance on WISE, which emphasizes world knowledge, and RISEBench, which focuses on reasoning logic. As shown in Table 2, on WISE, Mind-Brush outperforms all proprietary image generation models, boosting the overall WiScore by 25.8% compared to Qwen-Image and reaching 0.78, matching the performance of GPT-Image-1. On RISEBench, our method achieves a score of 61.5 in Instruction Reasoning, surpassing Nano Banana and exceeding Bagel by 68.5%. Furthermore, our overall accuracy approaches that of GPT-Image-1. These results further corroborate that Mind-Brush effectively integrates multimodal inputs to infer implicit user intents, thereby realizing accurate image generation.

Figure 4 illustrates the qualitative comparison between Mind-Brush and competing baselines on Mind-Bench. In the context of Knowledge-Driven tasks, Mind-Brush effectively leverages search tools to retrieve pertinent visual references, thereby achieving accurate synthesis of Out-ofDistribution concepts that baseline models fail to recognize. Regarding Reasoning-Driven tasks, the Knowledge Rea-

[Figure 4]

Figure 4. Qualitative Comparison of different models on Mind-Bench, including knowledge-driven (upper part) and reasoning-driven (lower part) tasks. The green border indicates that the generated result matches the facts, while the red border indicates the opposite.

soning Agent dissects implicit attribute features within concise user instructions—such as spatial relationships in the output or the mathematical logic underlying input questions. This capability allows the system to effectively manifest the user’s intricate intent in the generated imagery, ensuring logical coherence alongside visual fidelity.

#### 5.4. Ablation Study

To validate the efficacy of the individual components within Mind-Brush, we conducted comprehensive ablation studies on the core Reasoning and Search tools using the MindBench dataset. As reported in Table 3, the incorporation of the Knowledge Reasoning Agent yielded significant accuracy gains on Reasoning-Driven tasks compared to the baseline. Conversely, the Cognition Search Agent effectively compensated for the model’s cognitive deficits regard-

ing unknown concepts, resulting in an accuracy improvement of 0.28 on Knowledge-Driven tasks over the baseline. Furthermore, the simultaneous deployment of both agents demonstrates a strong synergistic effect. This combination not only bridges cognitive gaps regarding OOD concepts but also accurately deciphers the user’s authentic generative intent. Consequently, the full framework achieves accuracy improvements of 0.17 and 0.06 compared to configurations using solely the Reasoning Agent or the Search Agent, respectively. More ablation study of can be found in the appendix A.3.

### 6. Conclusion

We introduced Mind-Brush, a training-free agentic framework that transforms text-to-image generation from pas-

sive decoding into an active cognitive workflow. By orchestrating intent analysis, multimodal grounding, and explicit chain-of-thought reasoning, Mind-Brush effectively bridges the gap between vague user intents and precise, factually grounded visual synthesis. To rigorously evaluate this, we established Mind-Bench, a benchmark designed to stress-test models on knowledge-intensive and reasoningdependent tasks. Empirical results demonstrate that our framework significantly outperforms state-of-the-art models, validating the synergy of active retrieval and logical deduction. We believe this shift towards an Agentic Generative Paradigm paves the way for next-generation systems capable of complex problem-solving in visual synthesis.

### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

### References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Black Forest Labs. Flux 2 max: Next generation image synthesis. https://bfl.ai/models/flux-2-max, 2026a. Accessed: 2026-01-26.

Black Forest Labs. Flux 2 pro: State-of-the-art quality at maximum speed. https://bfl.ai/models/ flux-2, 2026b. Accessed: 2026-01-26.

Cai, H., Cao, S., Du, R., Gao, P., Hoi, S., Hou, Z., Huang, S., Jiang, D., Jin, X., Li, L., et al. Z-image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699, 2025.

Cao, Y., Lan, Y., Zhai, F., and Li, P. 5w1h extraction with large language models, 2024. URL https://arxiv. org/abs/2405.16150.

Chen, C.-Y., Shi, M., Zhang, G., and Shi, H. T2i-copilot: A training-free multi-agent text-to-image system for enhanced prompt interpretation and interactive generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 19396–19405, 2025a.

Chen, J., Xu, Z., Pan, X., Hu, Y., Qin, C., Goldstein, T., Huang, L., Zhou, T., Xie, S., Savarese, S., et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025b.

Chen, X., Wu, Z., Liu, X., Pan, Z., Liu, W., Xie, Z., Yu, X., and Ruan, C. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025c.

Chen, Z., Liu, K., Wang, Q., Liu, J., Zhang, W., Chen, K., and Zhao, F. Mindsearch: Mimicking human minds elicits deep ai searcher. arXiv preprint arXiv:2407.20183, 2024.

Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Deng, C., Zhu, D., Li, K., Gou, C., Li, F., Wang, Z., Zhong, S., Yu, W., Nie, X., Song, Z., et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

Esser, P., Kulal, S., Blattmann, A., Entezari, R., M¨uller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

Ghosh, D., Hajishirzi, H., and Schmidt, L. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.

Gong, L., Hou, X., Li, F., Li, L., Lian, X., Liu, F., Liu, L., Liu, W., Lu, W., Shi, Y., et al. Seedream 2.0: A native chinese-english bilingual image generation foundation model. arXiv preprint arXiv:2503.07703, 2025.

Google DeepMind. Gemini image pro: High-quality image generation. https://deepmind.google/ models/gemini-image/pro/, 2025a. Accessed: 2026-01-26.

Google DeepMind. Gemini image: High-quality image generation. https://deepmind.google/models/ gemini-image/flash/, 2025b. Accessed: 202601-26.

Jiang, D., Guo, Z., Zhang, R., Zong, Z., Li, H., Zhuo, L., Yan, S., Heng, P.-A., and Li, H. T2i-r1: Reinforcing image generation with collaborative semantic-level and token-level cot. arXiv preprint arXiv:2505.00703, 2025a.

Jiang, D., Zhang, R., Li, H., Zong, Z., Guo, Z., He, J., Guo, C., Ye, J., Fang, R., Li, W., et al. Draco: Draft as cot for text-to-image preview and rare concept generation. arXiv preprint arXiv:2512.05112, 2025b.

Jiang, K., Wang, Y., Zhou, J., Li, P., Liu, Z., Xie, C.-W., Chen, Z., Zheng, Y., and Zhang, W. Genagent: Scaling text-to-image generation via agentic multimodal reasoning, 2026. URL https://arxiv.org/abs/2601.

18543.

Kou, S., Jin, J., Zhou, Z., Ma, Y., Wang, Y., Chen, Q., Jiang, P., Yang, X., Zhu, J., Yu, K., et al. Think-thengenerate: Reasoning-aware text-to-image diffusion with llm encoders. arXiv preprint arXiv:2601.10332, 2026.

Labs, B. F. Flux. https://github.com/ black-forest-labs/flux, 2024.

Labs, B. F., Batifol, S., Blattmann, A., Boesel, F., Consul, S., Diagne, C., Dockhorn, T., English, J., English, Z., Esser, P., Kulal, S., Lacey, K., Levi, Y., Li, C., Lorenz, D., M¨uller, J., Podell, D., Rombach, R., Saini, H., Sauer, A., and Smith, L. Flux.1 kontext: Flow matching for incontext image generation and editing in latent space, 2025. URL https://arxiv.org/abs/2506.15742.

Li, C., Sun, J., Feng, Y., Zhai, M., Chang, Y., and Zhang, K. Ia-t2i: Internet-augmented text-to-image generation. arXiv preprint arXiv:2505.15779, 2025a.

Li, M., Hou, X., Liu, Z., Yang, D., Qian, Z., Chen, J., Wei, J., Jiang, Y., Xu, Q., and Zhang, L. Mccd: Multi-agent collaboration-based compositional diffusion for complex text-to-image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 13263– 13272, 2025b.

Li, X., Dong, G., Jin, J., Zhang, Y., Zhou, Y., Zhu, Y., Zhang, P., and Dou, Z. Search-o1: Agentic search-enhanced large reasoning models. arXiv preprint arXiv:2501.05366, 2025c.

Meng, F., Shao, W., Luo, L., Wang, Y., Chen, Y., Lu, Q., Yang, Y., Yang, T., Zhang, K., Qiao, Y., et al. Phybench: A physical commonsense benchmark for evaluating textto-image models. arXiv preprint arXiv:2406.11802,

- 2024.

Niu, Y., Ning, M., Zheng, M., Jin, W., Lin, B., Jin, P., Liao, J., Feng, C., Ning, K., Zhu, B., et al. Wise: A world knowledge-informed semantic evaluation for textto-image generation. arXiv preprint arXiv:2503.07265,

- 2025.

OpenAI. Gpt-image-1: Models and capabilities for image generation. https://platform.openai.com/ docs/models/gpt-image-1, 2024. Accessed: 2026-01-29.

OpenAI. Gpt-image-1.5: Enhanced visual reasoning and creative generation. https://platform.openai. com/docs/models/gpt-image-1.5, 2025. Accessed: 2026-01-29.

Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., M¨uller, J., Penna, J., and Rombach, R. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Son, M. H., Oh, J., Mun, S. B., Roh, J., and Choi, S. World-to-image: Grounding text-to-image generation with agent-driven world knowledge. arXiv preprint arXiv:2510.04201, 2025.

Stability AI. Stable diffusion 3.5 large. https://huggingface.co/stabilityai/ stable-diffusion-3.5-large, 2024a.

Stability AI. Stable diffusion 3.5 medium. https://huggingface.co/stabilityai/ stable-diffusion-3.5-medium/, 2024b.

Su, Z., Xia, P., Guo, H., Liu, Z., Ma, Y., Qu, X., Liu, J., Li, Y., Zeng, K., Yang, Z., et al. Thinking with images for multimodal reasoning: Foundations, methods, and future frontiers. arXiv preprint arXiv:2506.23918, 2025.

Sun, K., Fang, R., Duan, C., Liu, X., and Liu, X. T2ireasonbench: Benchmarking reasoning-informed textto-image generation. arXiv preprint arXiv:2508.17472, 2025.

Team, C. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

Van Den Oord, A., Vinyals, O., et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.

Venkatesh, K., Dunlop, C., and Yanardag, P. Crea: A collaborative multi-agent framework for creative image editing and generation. arXiv preprint arXiv:2504.05306, 2025.

Wang, X., Zhang, X., Luo, Z., Sun, Q., Cui, Y., Wang, J., Zhang, F., Wang, Y., Li, Z., Yu, Q., et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.

Wu, C., Li, J., Zhou, J., Lin, J., Gao, K., Yan, K., Yin, S.-m., Bai, S., Xu, X., Chen, Y., et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025a.

Wu, C., Zheng, P., Yan, R., Xiao, S., Luo, X., Wang, Y., Li, W., Jiang, X., Liu, Y., Zhou, J., et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025b.

Xiang, D., Xu, W., Chu, K., Ding, T., Shen, Z., Zeng, Y., Su, J., and Zhang, W. Promptsculptor: Multi-agent based text-to-image prompt optimization. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 774– 786, 2025.

Xie, J., Mao, W., Bai, Z., Zhang, D. J., Wang, W., Lin, K. Q., Gu, Y., Chen, Z., Yang, Z., and Shou, M. Z. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.

Yan, Z., Lin, K., Li, Z., Ye, J., Han, H., Wang, Z., Liu, H., Lin, B., Li, H., Xu, X., et al. Unified multimodal model as auto-encoder. arXiv preprint arXiv:2509.09666, 2025.

Ye, J., Jiang, D., Wang, Z., Zhu, L., Hu, Z., Huang, Z., He, J., Yan, Z., Yu, J., Li, H., et al. Echo-4o: Harnessing the power of gpt-4o synthetic images for improved image generation. arXiv preprint arXiv:2508.09987, 2025a.

Ye, Y., He, X., Li, Z., Lin, B., Yuan, S., Yan, Z., Hou, B., and Yuan, L. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025b.

Yu, T., Zhang, S., and Feng, Y. Auto-rag: Autonomous retrieval-augmented generation for large language models. arXiv preprint arXiv:2411.19443, 2024.

Zhang, R., Jiang, D., Zhang, Y., Lin, H., Guo, Z., Qiu, P., Zhou, A., Lu, P., Chang, K.-W., Qiao, Y., et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pp. 169–186. Springer, 2024.

Zhang, X., Guo, J., Zhao, S., Fu, M., Duan, L., Hu, J., Chng, Y. X., Wang, G.-H., Chen, Q.-G., Xu, Z., et al. Unified multimodal understanding and generation models: Advances, challenges, and opportunities. arXiv preprint arXiv:2505.02567, 2025.

Zhao, X., Zhang, P., Tang, K., Zhu, X., Li, H., Chai, W., Zhang, Z., Xia, R., Zhai, G., Yan, J., et al. Envisioning beyond the pixels: Benchmarking reasoning-informed visual editing. arXiv preprint arXiv:2504.02826, 2025.

Zhou, C., Yu, L., Babu, A., Tirumala, K., Yasunaga, M., Shamis, L., Kahn, J., Ma, X., Zettlemoyer, L., and Levy, O. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024.

Zhou, T., Duan, Z., Chen, C., Zhou, W., Wang, Y., and Li, Y. Agentstory: A multi-agent system for story visualization with multi-subject consistent text-to-image generation. In Proceedings of the 2025 International Conference on Multimedia Retrieval, pp. 1894–1902, 2025.

Zhuo, L., Zhao, L., Paul, S., Liao, Y., Zhang, R., Xin, Y., Gao, P., Elhoseiny, M., and Li, H. From reflection to perfection: Scaling inference-time optimization for text-to-image diffusion models via reflection tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 15329–15339, 2025.

Appendix A. Additional Implementary Details of

Mind-Brush

- A.1. Workflow of Mind-Brush

Algorithm 1 formally delineates the inference workflow of the Mind-Brush framework. Unlike static generation paradigms, Mind-Brush operates as a dynamic, agentic cognitive system designed to align vague user intents with objective reality.

The workflow initiates with the reception of a User Instruction (Iinst) and an optional User Image (Iimg), alongside the initialization of core foundation models and toolsets. In the Intent Analysis phase, the system structurally decomposes the input using the 5W1H paradigm to pinpoint cognitive gaps (Qgap) and formulates a dynamic execution strategy (Splan).

Driven by this strategy, the system branches into active knowledge acquisition. If Cognition Search is triggered, the agent performs a two-stage retrieval process: gathering textual evidence (Tref) to ground factual concepts and subsequently refining visual queries to retrieve semantically accurate reference images (Iref). Conversely, for tasks necessitating deep logic or cross-modal understanding, the Knowledge Reasoning agent is activated. Crucially, this agent ingests the user image (Iimg) alongside the instruction and gathered evidence, employing Chain-of-Thought (CoT) reasoning to derive explicit logical conclusions (Rcot) and resolve implicit constraints.

Finally, the Concept Review agent acts as a convergence hub, synthesizing the user’s original intent with the disjointed streams of multimodal evidence (E) into a coherent Master Prompt (Pmaster). This prompt, enriched with factual and logical precision, guides the Image Generator (Gθ) to synthesize the final high-fidelity image (Ifinal), ensuring rigorous alignment with both user creativity and real-world logic.

- A.2. Additional Experimental Results

Due to space limitations in the manuscript, we conducted additional evaluations of Mind-Brush on the GenEval++ (Ye et al., 2025a) and Imagine-Bench (Ye et al., 2025a) benchmarks. GenEval++ is an enhanced instruction-following benchmark that utilizes Accuracy as its metric. It increases instruction complexity and compositional difficulty across various task categories (e.g., Color, Count, Position) to mitigate metric saturation. Imagine-Bench is a benchmark dedicated to creative generation, evaluating models on task categories such as Attribute Shift and Spatiotemporal Anomalies. It employs a Comprehensive Score to assess semantic understanding and visual quality in surreal fantasy scenarios.

Algorithm 1 Inference Workflow of Mind-Brush Framework

- 1: Input: User Instruction Iinst, User Image Iimg (Optional)
- 2: Initialization: Large Language Model LLMϕ, Image Generator Gθ, Search Tools; Evidence set E ← ∅, Visual References Iref ← ∅
- 3: Decompose Iinst via 5W1H paradigm and identify cognitive gaps Qgap (Aintent)
- 4: Formulate execution strategy Splan based on Qgap
- 5: if Splan requires Cognition Search then
- 6: Generate text queries Qtxt and initial visual queries Qv
- 7: Tref ← Searchtxt(Qtxt)k
- 8: Update Evidence E ← E ∪ Tref
- 9: Q′v ← Φrefine(Qv,Tref)
- 10: Iref ← Searchimg(Q′v)k
- 11: end if
- 12: if Splan requires Knowledge Reasoning then
- 13: Retrieve relevant context from E, Iinst and Iimg
- 14: Rcot ← Φreason(Iinst,Iimg,Qgap,E)
- 15: Update Evidence E ← E ∪ Rcot
- 16: end if
- 17: Synthesize disjointed streams {Iinst,Iimg,E} and resolve ambiguities (Areview)
- 18: Pmaster ← Rewrite(Iinst,Iimg,E) to generate Master Prompt
- 19: Ifinal ← Gθ(Pmaster,Iref) conditioned on visual references
- 20: return High-fidelity Generated Image Ifinal

Experimental results demonstrate that Mind-Brush achieves impressive performance on these challenging benchmarks. As shown in Table 4, compared to Agentic methods such as PromptEnhancer and GenAgent, our method outperforms the best performance baseline (GenAgent) on GenEval++. Specifically, Mind-Brush improves Accuracy by 41.7% on the Pos/Count task and 13.3% on the Multi-Count task, reaching performance levels close to GPT-4o. Table 5 presents the performance comparison on Imagine-Bench; our results show improvements in Score of 8.2% and 2.6% on the Spatiotemporal and Hybridization tasks, respectively, compared to GenAgent. Extensive experimental results validate the superior capability of our framework in handling complex long-tail instructions and open-world creative generation tasks.

#### A.3. Additional Ablation Study

To rigorously evaluate the generalizability and robustness of the Mind-Brush framework, we conducted ablation studies involving diverse configurations of the Multimodal Large Language Model (MLLM) backbone and the image genera-

- Table 4. Quantitative Comparison of different methods on GenEval++. The best performing model (excluding close-sourece model GPT4o) is highlighted in bold and underline indicates the second best.

Method Color Count Color/Count Color/Pos Pos/Count Pos/Size Multi-Count Overall

FLUX.1-dev (Labs, 2024) 0.400 0.600 0.250 0.250 0.075 0.400 0.300 0.325 Qwen-Image (Wu et al., 2025a) 0.875 0.725 0.725 0.600 0.475 0.725 0.550 0.668

GPT4o (Achiam et al., 2023) 0.900 0.675 0.725 0.625 0.600 0.800 0.850 0.739 Janus Pro 7B (Chen et al., 2025c) 0.450 0.300 0.125 0.300 0.075 0.350 0.125 0.246 T2I-R1 (Jiang et al., 2025a) 0.675 0.325 0.200 0.350 0.075 0.250 0.300 0.311 Bagel (Deng et al., 2025) 0.325 0.600 0.250 0.325 0.250 0.475 0.375 0.371

PromptEnhancer (Xiang et al., 2025) 0.500 0.625 0.225 0.375 0.125 0.450 0.375 0.382 ReflectionFLow (Zhuo et al., 2025) 0.400 0.625 0.275 0.275 0.200 0.425 0.325 0.361 GenAgent (Jiang et al., 2026) 0.775 0.775 0.650 0.800 0.600 0.725 0.750 0.725

Mind-Brush (Ours) 0.775 0.700 0.775 0.750 0.850 0.775 0.850 0.782

- Table 5. Quantitative Comparison of different methods on Imagine. The best performing model (excluding close-sourece model GPT4o) is highlighted in bold and underline indicates the second best.

Method Attribute Shift Spatiotemporal Hybridization Multi-Object Overall

FLUX.1-dev (Labs, 2024) 5.298 6.350 7.053 5.973 6.072 Qwen-Image (Wu et al., 2025a) 6.771 7.193 8.130 7.500 7.329

GPT4o (Achiam et al., 2023) 8.540 9.180 8.570 7.980 8.560 Janus Pro 7B (Chen et al., 2025c) 5.300 7.280 6.730 6.040 6.220 T2I-R1 (Jiang et al., 2025a) 5.850 7.700 7.360 6.680 6.780 Bagel (Deng et al., 2025) 5.370 6.930 6.500 6.410 6.200

PromptEnhancer (Xiang et al., 2025) 5.489 6.213 7.327 6.493 6.281 GenAgent (Jiang et al., 2026) 7.613 7.547 8.343 7.763 7.794

Mind-Brush (Ours) 7.416 8.167 8.557 7.533 7.862

tion engine. The quantitative results, presented in Table 6, confirm the framework’s efficacy across distinct computational regimes and disentangle the contributions of the agentic intelligence substrate versus the visual synthesis capabilities.

Quantitative Comparison with Different MLLM Baselines. To isolate the contribution of the multimodal foundation model that drives the entire agentic workflow, we evaluated the framework’s performance by replacing the MLLM backbone across all constituent agents while holding the generation model constant. We first tested a fully open-source configuration utilizing Qwen3-VL-235B as the universal backbone for all agents and Qwen-Image as the visual generator. Despite relying on accessible open-weights models, this configuration achieves an overall CSA score of 0.24, surpassing the proprietary baseline GPT-Image-1.5 (0.21) by a relative margin of 14.3%. This empirical evidence suggests that the collaborative agentic workflow can effectively compensate for the capacity gaps between opensource models and state-of-the-art proprietary baselines. Furthermore, by upgrading the framework’s backbone from

Qwen3-VL to GPT-5.1 (while keeping Qwen-Image fixed), the overall performance significantly improves from 0.24 to 0.31, yielding a 29.2% relative gain. This finding is pivotal, demonstrating that the intelligence level of the underlying MLLM is the dominant factor. A more capable backbone enhances the precision of every agentic step—from evidence retrieval to constraint verification—directly translating into higher generation fidelity even without altering the image decoder.

Quantitative Comparison with Different Image Generation Models. We further investigated the framework’s capacity to empower and scale with different visual executors. Integrating the legacy GPT-Image-1 into the Mind-Brush framework (driven by the GPT-5.1 backbone) results in a substantial performance leap, where the overall CSA score increases from the baseline of 0.17 to 0.34, representing a 100% relative improvement. This confirms that MindBrush serves as a powerful meta-architecture capable of unlocking the latent potential of weaker generators. Notably, under the same agentic backbone (GPT-5.1), the GPTImage-1 based configuration outperforms the Qwen-Image

variant. This indicates that while the agentic backbone is crucial for intent alignment and factuality, the framework effectively utilizes the superior visual synthesis capabilities of stronger generation engines, ensuring that performance scales consistently with the quality of both the intelligence substrate and the visual decoder.

#### A.4. Evaluation Protocols

To ensure a rigorous and standardized assessment of model performance across diverse cognitive dimensions, we strictly adhere to the official evaluation protocols defined in the respective benchmarks. All automated evaluations for existing benchmarks utilize the official LLM settings as the core judge to maintain consistency with prior literature.

WISE Evaluation. For the WISE benchmark, which necessitates deep semantic comprehension and world knowledge integration, we report the WiScore following the official definition. This composite metric provides a quantitative measure of knowledge-image alignment by aggregating three sub-dimensions on a discrete 3-point scale (s ∈ {0,1,2}): Consistency (Scon), Realism (Sreal), and Aesthetic Quality (Saes). Formally, the WiScore is calculated as a weighted linear combination:

WiScore = α1 · S¯con + α2 · S¯real + α3 · S¯aes (3)

The weights are rigorously calibrated to α1 = 0.7, α2 = 0.2, and α3 = 0.1. This weighting strategy prioritizes the model’s fidelity to implicit semantic constraints (Consistency) while simultaneously penalizing violations of physical laws (Realism) and low artistic quality (Aesthetic), thereby offering a holistic view of generation quality grounded in world knowledge.

RISEBench Evaluation. RISEBench targets reasoninginformed visual editing, where precision is paramount. Following its official guidelines, we conduct a fine-grained evaluation using GPT-4.1 to score results on a scale of 1 to 5 across three critical dimensions: Instruction Reasoning (Sir), which measures the accurate execution of complex logical directives; Appearance Consistency (Sac), which assesses the preservation of task-irrelevant visual attributes; and Visual Plausibility (Svp), which evaluates the coherence of the edited output. Distinct from standard aggregation methods, RISEBench imposes a strict All-or-Nothing success criterion. A sample is deemed successfully solved if and only if it achieves perfect scores across all applicable dimensions. The final accuracy is defined as:

1 N

AccRISE =

N

I Sir(i) = 5 ∧ Sac(i) = 5 ∧ Svp(i) = 5

i=1

(4)

where I(·) denotes the indicator function. This rigorous metric effectively penalizes partial failures, ensuring that high performance reflects robust reasoning capabilities rather than successful approximations.

Mind-Bench Evaluation. For our proposed Mind-Bench, to rigorously quantify the model’s ability to bridge cognitive gaps without hallucination, we employ the Checklistbased Strict Accuracy (CSA). To mitigate potential selfevaluation biases inherent in GPT-series models, we utilize Gemini-3.0-Pro as the expert evaluator. Each test sample is associated with a human-verified set of atomic factual claims C = {c1,...,ck}. The evaluator performs a binary verification ({0,1}) for each claim. We adopt a Holistic Pass Criterion, where a generation is considered correct only if it satisfies all constraints in the checklist:

 

  (5)

|Ci|

N

1 N

VQAM(Igen(i) ,c(ji)) = 1

AccCSA =

I

j=1

i=1

This stringent protocol ensures that the reported accuracy reflects comprehensive understanding and precise grounding of external knowledge, rather than superficial semantic overlap.

### B. Additional Details of Mind-Bench

#### B.1. Additional task description of Mind-Bench

Due to space limitations in the main manuscript, we provide a comprehensive specification of the task taxonomy within Mind-Bench in this section. Table 7 presents a detailed overview, outlining the sample distribution, input modalities, and precise definitions for each of the 10 distinct task categories, structured across Knowledge-driven and Reasoning-driven domains to fully elucidate the benchmark’s evaluation scope.

#### B.2. Comparison with Existing Benchmarks

To contextualize the contributions of Mind-Bench, Table 8 presents a comprehensive comparison with existing T2I benchmarks, highlighting our distinct advantages in temporality, modality, and evaluation rigor. Uniquely, MindBench serves as the sole platform for assessing real-time information retrieval, diverging from traditional benchmarks like GenEval and WISE that rely on static, frozen knowledge distributions to explicitly target dynamic concepts. Beyond simple text alignment, it incorporates multimodal contexts to evaluate complex reasoning, matching the depth of specialized benchmarks like T2I-ReasonBench while offering superior breadth across 10 distinct task categories. Furthermore, by adopting the Checklist-based Strict Accuracy (CSA) metric instead of subjective scalar

- Table 6. Ablation study of different MLLM backbones and image generation model. The best performing model is highlighted in bold.

Knowledge-Intensive Reasoning-Intensive

MLLM model Generation model

Overall SE Weather MC IP WK Life Reason GU Math SL Poem

- - GPT-Image-1 0.32 0.06 0.22 0.02 0.16 0.24 0.10 0.12 0.32 0.10 0.17

- - GPT-Image-1.5 0.36 0.18 0.22 0.04 0.30 0.34 0.10 0.02 0.34 0.08 0.21

- - Nano Banana 0.30 0.10 0.12 0.00 0.30 0.20 0.04 0.08 0.32 0.36 0.18

- - Nano Banana Pro 0.50 0.36 0.40 0.16 0.56 0.30 0.16 0.46 0.62 0.68 0.41

###### Mind-Brush

Qwen3-VL-235B Qwen-Image 0.42 0.06 0.44 0.10 0.36 0.06 0.14 0.18 0.20 0.44 0.24 GPT-5.1 Qwen-Image 0.54 0.16 0.62 0.18 0.40 0.10 0.16 0.14 0.26 0.54 0.31 GPT-5.1 GPT-Image-1 0.64 0.18 0.56 0.10 0.50 0.28 0.10 0.06 0.50 0.48 0.34

scoring, Mind-Bench ensures a precise, objective standard for validating the capabilities of agentic generation systems.

#### B.3. Data Sources and Copyright Statement

The construction of Mind-Bench strictly adheres to the copyright policies of its data sources. Data is primarily derived from publicly available academic datasets or public information sources permitting non-commercial use:

- 1. Wikipedia: Reference images and textual descriptions for tasks involving News, Historical Events, Specific Characters, Trendy IPs, World Knowledge, and Geographical Understanding are primarily sourced from Wikipedia1. These contents are utilized under the Creative Commons Attribution-ShareAlike 3.0 Unported License (CC BY-SA 3.0).
- 2. Historical Weather Data: Data for Weather tasks, including temperature and historical weather conditions, is sourced from world-weather.info2.
- 3. Life Reasoning Data: Image data for life reasoning tasks is curated from recipetineats3.
- 4. Mathematical Data: Images and queries for Math tasks are primarily adapted from the MathVerse image generation benchmark (Zhang et al., 2024), adhering to its academic citation license.

Redistribution Policy: We prioritize respecting the intellectual property rights of all data owners. For content licensed under compatible open-source agreements (e.g., CC BYSA), we include the data directly in our release. For sources that restrict secondary redistribution, our benchmark release will provide metadata and direct URLs to the original content, allowing users to download the data independently in compliance with the respective terms of service.

- 1https://www.wikipedia.org/
- 2https://world-weather.info
- 3https://www.recipetineats.com

### C. Additional Visualization of Inference Process

To provide a more intuitive understanding of our framework, we present additional qualitative examples (from fig. 5 to fig. 24) that visualize the complete step-by-step cognitive trajectory of Mind-Brush, ranging from initial intent analysis to the final image synthesis across diverse scenarios.

Table 7. Overview of Mind-Bench. Task distribution, types, and per-task definitions across Knowledge and Reasoning domains. Task Number Type Definition Knowledge-driven News 50 T2I Retrieve and generate images of specific news events based on the provided temporal

and spatial contexts. Weather 50 T2I Retrieve and generate images of meteorological conditions for specific times and locations. Character 50 T2I Retrieve and generate images of specific personas, celebrities, or fictional characters from user input. IP 50 T2I Retrieve and generate images of products and artifacts associated with well-known intellectual properties. World Knowledge 50 T2I Retrieve and generate images corresponding to specific factual and historical information about the world.

Reasoning-driven Life Reasoning 50 I2I Reason and generate images related to daily life tasks and their outcomes based on

provided lifestyle imagery. Geo Understanding 50 I2I Reason, retrieve, and generate images of specific locations based on input map imagery and spatial contexts. Math 50 I2I Reason visual mathematical problems and generate images rendering the step-bystep results. Science 50 T2I Reason and generate images depicting scientific phenomena, physical states, or experimental processes. Poem 50 T2I Reason and generate visual scenes that embody the specific imagery and metaphors of poems given the poet and emotion.

Table 8. Comparison between Mind-Bench and existing T2I benchmarks.

###### Benchmark Up-to-date Reasoning modality Sample Num Task Num Metric Type

GenEval (Ghosh et al., 2023) No Text ∼550 6 Scoring GenEval++ (Ye et al., 2025a) No Text ∼280 7 Accuracy WISE (Niu et al., 2025) No Text 1,000 25 Scoring T2I-ReasonBench (Sun et al., 2025) No Text + Image 800 4 Scoring RISEBench (Zhao et al., 2025) No Text + Image 360 4 Scoring + Accuracy

###### Mind-Bench (Ours) Yes Text + Image 500 10 Accuracy (CSA)

[Figure 5]

###### Figure 5. A generation process of Mind-Brush in Special Events task of Mind-Bench.

[Figure 6]

###### Figure 6. A generation process of Mind-Brush in Special Events task of Mind-Bench.

[Figure 7]

[Figure 8]

[Figure 9]

###### Figure 9. A generation process of Mind-Brush in Character task of Mind-Bench.

[Figure 10]

###### Figure 10. A generation process of Mind-Brush in Character task of Mind-Bench.

[Figure 11]

###### Figure 11. A generation process of Mind-Brush in Object task of Mind-Bench.

[Figure 12]

###### Figure 12. A generation process of Mind-Brush in Object task of Mind-Bench.

[Figure 13]

###### Figure 13. A generation process of Mind-Brush in World Knowledge task of Mind-Bench.

[Figure 14]

###### Figure 14. A generation process of Mind-Brush in World Knowledge task of Mind-Bench.

[Figure 15]

###### Figure 15. A generation process of Mind-Brush in Life Reasoning task of Mind-Bench.

[Figure 16]

###### Figure 16. A generation process of Mind-Brush in Life Reasoning task of Mind-Bench.

[Figure 17]

[Figure 18]

[Figure 19]

###### Figure 19. A generation process of Mind-Brush in Geo Math task of Mind-Bench.

[Figure 20]

###### Figure 20. A generation process of Mind-Brush in Math task of Mind-Bench.

[Figure 21]

###### Figure 21. A generation process of Mind-Brush in Science & logic task of Mind-Bench.

[Figure 22]

###### Figure 22. A generation process of Mind-Brush in Science & logic task of Mind-Bench.

[Figure 23]

###### Figure 23. A generation process of Mind-Brush in Poem task of Mind-Bench.

[Figure 24]

###### Figure 24. A generation process of Mind-Brush in Poem task of Mind-Bench.

