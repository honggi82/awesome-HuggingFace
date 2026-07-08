## ViLoMem: Agentic Learner with Grow-and-Refine Multimodal Semantic Memory

Weihao Bo1,2 Shan Zhang3 Yanpeng Sun4 Jingjing Wu2 Qunyi Xie2 Xiao Tan2

Kunbin Chen2 Wei He2 Xiaofan Li2 Na Zhao4 Jingdong Wang2 * Zechao Li1† 1Nanjing University of Science and Technology 2Baidu Inc 3AIML, Adelaide University 4Singapore University of Technology and Design

Project page: https://weihao-bo.github.io/ViLoMeo-page/

# arXiv:2511.21678v2[cs.AI]2May2026

#### Abstract

[Figure 1]

[Figure 2]

The formula for calculating the area of a triangle 𝐀 = 𝟏𝟐×𝒃𝒂𝒔𝒆× 𝒉𝒆𝒊𝒈𝒉 . Right triangles use legs as height; non-right triangles must draw altitude. Like this…

Calculatethearea oftriangleABC?

MLLMs exhibit strong reasoning on isolated queries, yet they operate de novo—solving each problem independently and often repeating the same mistakes. Current memory agents mainly reuse past trajectories. However, this approach suffers from brevity bias and gradually loses essential domain knowledge. Most critically, even in multimodal tasks, it records only a single-modality trace, failing to capture how visual attention and logical reasoning jointly produce the solution. This is fundamentally misaligned with human cognition: semantic memory is both multimodal and integrated, preserving visual and abstract knowledge through coordinated but distinct representational streams. We thus introduce ViLoMem, a dual-stream memory framework that constructs compact, schema-based memory. It separately encodes visual distraction patterns and logical reasoning errors, enabling MLLMs to learn from their successful and failed experiences. Following a grow-and-refine principle, the system incrementally accumulates and updates multimodal semantic knowledge—preserving stable, generalizable strategies while avoiding catastrophic forgetting. Across six multimodal benchmarks, ViLoMem consistently improves pass@1 accuracy and reduces repeated visual and logical errors. Ablations confirm that dual-stream memory is necessary and scalable, supporting long-term learning with efficiency ensured by our retrieval design; learning from visually intensive experiences leads to more faithful multimodal reasoning over time.

5+12+13

= 30 ???

A

- 12
- 13

A

5

13 Got it, let me think.

B C

5 12

B C

[Figure 3]

[Figure 4]

[Figure 5]

I need to help me determine the height, and then I need to calculate the area.

Logic Memory

[Figure 6]

Logic Memory

[Figure 7]

[Figure 8]

A B

Area of triangle: 𝐀 = 𝟏𝟐×𝒃𝒂𝒔𝒆×𝒉𝒆𝒊𝒈𝒉 .

Area of triangle: 𝐀 = 𝟏𝟐×𝒃𝒂𝒔𝒆×𝒉𝒆𝒊𝒈𝒉 .

[Figure 9]

O

4

[Figure 10]

[Figure 11]

2

[Figure 12]

Visual Memory

Visual Memory

D C

6

I am certain that the area of triangle ODC is 𝟏 𝟐×𝟐×𝟔 = 𝟔

Calculatethearea oftriangleODC ?

A

I think I understand how to calculate the area of a triangle.

A B

13

O

5 12

4

B C

D C

6

Figure 1. Multimodal Semantic Memory Enables Progressive Learning. When solving multimodal problems, early attempts may contain both logical and visual errors; through feedback, the model refines its logical memory for question-appropriate theorem application and its visual memory to avoid perceptual traps—improving by integrating the where to look with the how to reason.

and complex scientific problem solving . Yet despite their growing capability, current MLLMs approach each problem de novo—solving every query in isolation, repeatedly re-deriving the same insights and re-committing familiar errors[15, 16, 34, 58]. Although recent memory-augmented models attempt to mitigate this by storing past interactions [33, 53], these memories capture only high-level logical summaries while discarding the visual grounding and perceptual cues essential for multimodal reasoning.

#### 1. Introduction

Recent research has demonstrated that MLLMs’ visual perception ability remains fundamentally weaker than their linguistic reasoning, with low-level perceptual failures identified as a primary bottleneck for high-level multimodal reasoning tasks [26, 32, 35, 56]. In mathematical multimodal problem-solving in particular, diagram-perception errors exceed logical reasoning errors, and visual mistakes fre-

Multimodal Large Language Models (MLLMs) [5, 43, 57, 59] have achieved impressive progress in scene understanding [24, 25, 31, 39], visual question answering [20, 21],

*Project Leader †Corresponding Author.

quently persist in intermediate reasoning steps even when the final answer is correct [55, 56].

This indicates visual attention errors directly cause downstream logical hallucinations that creates a cascading failure pattern [41, 60]. Our ablation studies further confirm this phenomenon: across six multimodal problemsolving benchmarks, the proportion of visual error summaries consistently exceeds that of logical memory errors (Fig. 4). Therefore, when solving problems paired with images, it is essential for models to maintain accurate visual attention to task-relevant regions, avoiding perceptual distractions that propagate into flawed logical inferences.

Logic-only memory is insufficient for multimodal problem solving. While logical theorems and rules are general (e.g., applying the base–height formula for area computation), effective reasoning also requires aligning these abstract rules with their correct visual counterparts (e.g., the shape of triangles). As illustrated in Fig. 1, triangles exhibit diverse visual configurations, and early attempts may contain both logical and visual errors. Through feedback, the model refines its logical memory for question-appropriate theorem application and its visual memory to avoid perceptual traps, attending to task-relevant regions. This progressive learning mirrors the human cognitive system, where semantic memory maintains multimodal representations that integrate visual experience with abstract reasoning [11].

We thus introduce ViLoMem, a dual-stream memory framework that separately models visual distraction patterns and logical hallucination errors as structured schemas, coordinating them through unified retrieval. Following a growand-refine principle, ViLoMem avoids the detail erosion caused by iterative rewriting by filtering similar error patterns and using tailored add/skip and retrieval strategies to incrementally accumulate multimodal semantic knowledge. Specifically, we design custom retrieval strategies for visual and logical streams. For the visual stream, direct imagesimilarity search is insufficient; the key requirement is helping the model identify question-specific “visually trapped regions”. To achieve question-aware attention, we generate cross-modal attention maps guided by keywords (previously observed visual mistakes), enabling the model to highlight regions associated with known error patterns relevant to the current question. For the logical stream, instead of directly retrieving query semantically similar logics, the model first analyzes the problem to identify its underlying subject and reasoning requirements—supporting precise positioning of the task type and precise selection of the relevant logical schema.

Overall, ViLoMem automatically attributes successes or failures to the visual or logical stream and updates the corresponding schemas without human supervision. It enables progressive mistake reduction and cross-domain knowledge transfer in multimodal tasks.

#### 2. Related Work

##### 2.1. Context Engineering

Recent advancements in agent self-improvement have prominently featured context engineering, a paradigm that refines model behavior by strategically modifying input prompts rather than altering the model’s underlying weights[1, 8, 29, 42]. These methods primarily leverage natural language feedback, enabling a model to analyze its own performance based on execution traces, reasoning steps, or validation signals and then iteratively revise its operational context [2, 30, 37, 49]. This approach has given rise to several influential frameworks. For instance, ReAct [47] pioneered the integration of reasoning and acting within a synergistic loop. Building on this, Reflexion [30] introduced a mechanism for agents to reflect on past failures, using verbal reinforcement to enhance subsequent planning and decision-making. Other works have focused on optimizing the prompts themselves; TextGrad [49] proposed a novel method to generate gradient-like textual feedback for prompt refinement, while GEPA [2] demonstrated that an evolutionary approach to prompt optimization based on execution traces can achieve performance surpassing that of traditional reinforcement learning in certain scenarios. However, these approaches are limited by their ephemeral nature; the context is constructed for single interactions, preventing long-term knowledge accumulation. Furthermore, they often suffer from a brevity bias [17], where iterative refinement strips away crucial details, hindering performance on complex, knowledge-intensive tasks.

##### 2.2. Long-term Memory

To address the limitations of transient context, a parallel line of research has focused on equipping agents with long-term memory, enabling them to learn from experience and retain knowledge persistently[3, 13, 28, 45, 58]. This vision is rooted in the cognitive science principle that true, deep learning extends beyond formal training and arises from the continuous accumulation of experience [6, 14, 36, 40]. Research in this area explores various architectures for building durable memory systems. For example, Dynamic Cheatsheet [33] constructs an external memory that explicitly stores successful and unsuccessful strategies from past inferences, allowing the agent to consult its history. Similarly, ACE [53] develops an incremental “context playbook” through a generate-reflect-curate cycle, which is designed to avoid the simplification and catastrophic forgetting associated with simple iterative rewriting. The mechanisms for populating these memories are also diverse, ranging from learning through early, formative experiences [52] and reinforcement learning-based exploration [50] to interactive learning from noisy, real-time human feedback [4, 46].

However, these frameworks exhibit a critical blind spot:

they are overwhelmingly logic-centric, capturing reasoning patterns while neglecting the visual dimension of multimodal tasks. In contrast, the human brain adopts a huband-spoke semantic memory architecture. Visual–semantic associations and error patterns are encoded in the inferotemporal and perirhinal cortex (visual spoke), while abstract reasoning rules and logical error patterns are maintained in the temporal–parietal cortex (logic spoke)[9, 22, 23]. The anterior temporal lobe (ATL) serves as the central hub that integrates these modality-specific representations into unified conceptual knowledge. Inspired by this architecture, our AI system implements an error-aware multimodal semantic memory, where visual and logical error patterns are stored in separate modality-specific modules, integrated through a semantic hub, and monitored by an executive verifier that detects redundant visual–logical information and modulates attention to prevent recurring mistakes in multimodal scientific reasoning tasks.

#### 3. Method

We propose ViLoMem, a plug-in dual-stream memory framework for multimodal reasoning in large language models, featuring a closed-loop Memory Cycle that enables the agent to continuously learn from its reasoning and perception errors—facilitating progressive, lifelong learning.

Problem Formulation. Consider a sequence of multimodal inputs (x1,x2,...,xn), where each input xi = (Ii,qi) consists of an image Ii and a question text qi. The system maintains two memory banks: a logic memory MLi = {mL1 ,mL2 ,...,mL|L|} storing textual reasoning guidelines, and a visual memory MVi = {(mV1 ,I1V ),(mV2 ,I2V ),...,(mV|V |,I|VV |)} storing visual guidelines paired with source images.

As illustrated in Figure 2(a), the cycle operates as follows: given problem xi, the system performs parallel Retrieval from both memory banks to obtain relevant mem-

ories RiL and RiV . These retrieved memories are then fed to the Solver for Utilization, which generates a candidate

answer y˜i. The Verifier evaluates this answer against the ground truth yi. Upon detecting an error (y˜i ̸= yi), the system activates the Generation process to update both memory banks in parallel, yielding MLi+1 and MVi+1. This mechanism enables the agent to progressively refine its capabilities through iterative self-correction.

Core Operations. We define several key operations used throughout the framework. Let ϕT(·) and ϕM(·) denote text and multimodal embedding functions, respectively. The cosine similarity between two embeddings is computed as:

u · v ∥u∥∥v∥

Sim(u,v) =

(1)

For problem analysis during retrieval, we employ an LLM to extract structured information from the question

and reasoning trace:

###### ai = AnalyzeL(qi,y˜i) (2)

The process identifies the problem’s subject domain and key concepts. An enriched query is then constructed by combining the original question with this analysis:

q˜i = [qi;ai] (3)

##### 3.1. Memory Generation

When errors are detected, the system activates a parallel memory-generation framework, as illustrated in Figure 2(b). This framework conducts detailed error attribution and constructs structured memory units corresponding to two distinct error types.

###### 3.1.1. Visual Memory Generation

The visual analysis module, powered by an MLLM, simultaneously identifies the error type and generates corrective guidance. Given the original image Ii, question qi, erroneous reasoning trace y˜i, and ground truth yi, the module produces both error indicator and corresponding guideline within a single model invocation, formally expressed as:

###### (eVi ,giV ) = AnalyzeGenerateV (Ii,qi,y˜i,yi), (4)

where eVi ∈ True,False indicates whether the error originates from visual misinterpretation (e.g., object confusion, overlooked visual symbols, or spatial relationship misunderstandings), and giV denotes the generated Visual Guideline—an instruction prescribing the correct observation strategy. All information is stored in a structured JSON dictionary for persistent memory updating. For example, when addressing shape and attribution-related errors in 3D solid objects, the guideline may state:

“When an object has a uniform, reflective, or metallic-looking surface—even if it appears matte under diffuse lighting—treat it as metallic if it matches the visual style of other known metallic objects in the scene.”

Before storage, a similarity check is performed against existing memories in MVi using text embeddings. The system computes similarity scores sVj = Sim(ϕT(giV ),ϕT(mVj )) for all mVj ∈ MVi . If maxj sVj > τV (where τV is a similarity threshold), a merge operation consolidates the knowledge:

MVi+1 = MVi \{(mVj∗,IjV∗)}∪{(MergeV (mVj∗,giV ),IjV∗)},

(5) where j∗ = arg maxj sVj . Otherwise, a new memory entry is created: MVi+1 = MVi ∪ {(giV ,Ii)}.

|[Figure 13]<br><br>Retrieved Memory<br><br>[Figure 14]<br><br>Verifier Solver<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>（a) Memory Cycle<br><br>Utilization<br><br>Action<br><br>[Figure 19]<br><br>[Figure 20]<br><br>Logic Memory<br><br>Visual Memory<br><br>Retrieval<br><br>Generation<br><br>[Figure 21]<br><br>Question|
|---|

|Logic Memory Retrieval<br><br>Logic Memory Visual Memory<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>（c) Memory Retrieval<br><br>Visual Memory Retrieval<br><br>Load Memory DB<br><br>Similarity Search<br><br>Image Similarity Search<br><br>Query Text Filter<br><br>Attention Heatmap<br><br>Question<br><br>[Figure 27]<br><br>Text Query Image Query<br><br>[Figure 28]<br><br>Logic Retriever Visual Retriever<br><br>[Figure 29]<br><br>[Figure 30]|
|---|

|(b) Memory Generation<br><br>[Figure 31]<br><br>[Figure 32]<br><br>LLM<br><br>Logic Analysis<br><br>[Figure 33]<br><br>Logic Memory Generation<br><br>[Figure 34]<br><br>MLLM<br><br>Visual Analysis<br><br>[Figure 35]<br><br>[Figure 36]<br><br>Visual Memory Generation<br><br>Merge / Create Check for Similarity<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>Logic Memory<br><br>Visual Memory<br><br>[Figure 40]<br><br>Verifier|
|---|

- Figure 2. Overview of the ViLoMem framework. (a) Memory Cycle: A closed-loop learning mechanism where both logical and visual memories are retrieved and utilized by the solver. Retrieval is conditioned on the textual question and its paired image. The solver then performs reasoning steps (actions), which are evaluated by the verifier to filter redundant or invalid trajectories. The remaining trajectories are used to update both memory streams according to their respective types. (b) Memory Generation: An error-attribution framework that employs an LLM for logical analysis and an MLLM for visual analysis, producing structured memory schemas through similarity-based merge and create operations. (c) Memory Retrieval: Specialized dual-stream retrieval mechanism. Visual memories undergo a two-stage process involving image-embedding retrieval followed by question-specific retrieval, since visual information must be conditioned on both image content and the textual query. Logical memories are retrieved through problem analysis and text-embedding similarity.

###### 3.1.2. Logical Memory Generation

scores sLj = Sim(ϕT(giL),ϕT(mLj )) are computed for all mLj ∈ MLi , and the memory bank is updated accordingly:

In parallel, the logic analysis module, powered by an LLM, examines the reasoning chain for non-visual errors such as computational mistakes, formula misapplications, or logical fallacies. This module focuses solely on textual reasoning without accessing visual information. As formalized in Equation (6), the module produces both error classification and guideline in a single model invocation:

 

MLi \ {mLj∗} ∪ {mLnew} sLj∗ > τL MLi ∪ {giL} sLj∗ ≤ τL MLi otherwise,

MLi+1 =

(7)



where j∗ = arg maxj sLj , mLnew = MergeL(mLj∗,giL), and the update is triggered when eLi = Logical and giL ̸= ∅.

(eLi ,giL) = AnalyzeGenerateL(qi,y˜i,yi), (6)

where eLi ∈ {Logical,Non-Logical} classifies whether the error involves reasoning failures, and giL represents the abstracted Logic Guideline. The model outputs a structured text response containing error type, analysis summary, and guideline fields. For example, when encountering a geometry error arising from incorrect assumptions (i.e., textual biases), the generated guideline may state:

##### 3.2. Memory Retrieval and Utilization

When addressing a new problem xi = (Ii,qi), the solver initiates parallel retrieval procedures from both memory banks, as illustrated in Figure 2(c). Unlike conventional single-stage retrieval, our framework employs specialized strategies for each memory type: visual memory uses a twostage multimodal-to-text pipeline, while logical memory leverages problem analysis to construct enriched queries.

“In geometry problems involving perpendicular bisectors, remember that only points lying on the perpendicular bisector segment are guaranteed to be equidistant from the endpoints of the segment. Do not assume a point lies on the bisector unless this is explicitly stated or can be proven from the given construction. Always verify the position of intersection points relative to the bisector before applying the equidistance property.”

###### 3.2.1. Visual Memory Retrieval

Visual memory retrieval employs a two-stage pipeline that progressively refines candidates from visual similarity to semantic relevance.

Stage 1: Image Embedding Similarity. The system first employs multimodal embeddings to compute visual similarity between the query image Ii and all stored memory images. For each memory (mVj ,IjV ) ∈ MVi , the similarity is computed as sMj = Sim(ϕM(Ii),ϕM(IjV )). This rapidly

This guideline then undergoes the same similarity check and merge/create process as visual memory. Similarity

Table 1. Main results across six multimodal reasoning benchmarks. Baseline metrics for Qwen3 series models are sourced from official reports, while GPT-4.1 baselines are from OpenCompass. Metrics marked with * indicate self-evaluated results where official reports are unavailable or show substantial discrepancies. Models with “(step)” and “(+ ViLoMem)” are prompted by step-by-step reasoning.

Method MMMU MathVista MathVision HallusionBench MMStar RealWorldQA

GPT-4.1 (baseline) 74.00 70.40 46.12* 58.50 69.80 73.72 GPT-4.1 (step) 74.16 74.27 47.47 74.44 70.43 72.03 GPT-4.1 (+ ViLoMem) 77.26 76.88 53.95 75.29 72.43 74.38

Qwen3-VL-235B-A22B-Instruct (baseline) 78.70 84.90 61.28* 63.20 78.40 79.30 Qwen3-VL-235B-A22B-Instruct (step) 75.97 83.66 62.17 74.58 76.16 78.66 Qwen3-VL-235B-A22B-Instruct (+ ViLoMem) 79.40 84.98 62.83 75.21 78.31 77.22

Qwen3-VL-8B-Instruct (baseline) 66.38* 77.20 48.13* 61.10 70.91 71.50 Qwen3-VL-8B-Instruct (step) 65.52 77.80 48.35 73.08 70.22 70.85 Qwen3-VL-8B-Instruct (+ ViLoMem) 69.90 77.87 49.34 73.19 72.13 73.59

recalls a set of top-kM candidate memories:

and ranking by similarity score:

CiV = {(mVj ,IjV ) | j ∈ TopK({sMj },kM)} (8)

Stage 2: Text Embedding Filtering . Visual similarity alone is insufficient for semantic matching. The system subsequently performs text-based reranking using the enriched query q˜i from Equation (3). For each candidate guideline mVj ∈ CiV , text similarity is computed as sTj = Sim(ϕT(˜qi),ϕT(mVj )). The final retrieved visual memories are obtained by filtering with threshold τV and selecting top-kV by similarity score:

RiV = {mVj | j ∈ TopK({sTj | sTj ≥ τV },kV )} (9)

This two-stage process ensures that the retrieved visual memories are both semantically relevant to the current problem and specifically address common visual pitfalls encountered when interpreting similar images.

Focusing on where to look via visual attention maps. Beyond textual guidelines, we further introduce an auxiliary visual representation of memory cues. Leveraging the retrieved visual memory and its associated error patterns, the system generates question-aware attention maps that highlight historically error-prone regions in the query image Ii. These attention maps serve as supplementary visual inputs alongside the original image, providing explicit spatial guidance that directs the model’s focus toward task-relevant areas while avoiding known perceptual traps. Experimental results demonstrate that this visual augmentation yields additional performance improvements (refer to Section 4.3).

###### 3.2.2. Logical Memory Retrieval

Logical memory retrieval is a text-based semantic matching process. The system constructs an enriched query q˜i using Equations 2-3 to capture both the problem text and structured domain information. For each memory mLj ∈ MLi , text embedding similarity is computed as sLj = Sim(ϕT(˜qi),ϕT(mLj )). The top-kL most relevant guidelines are retrieved by applying similarity threshold τL

RiL = {mLj | j ∈ TopK({sLj | sLj ≥ τL},kL)} (10)

###### 3.2.3. Solution Generation with Dual Memory

Finally, the solver generates the answer by conditioning on both the original inputs and the retrieved memories from the visual and logical streams:

y˜i = Gen(Ii,qi,RiL,RiV ), (11)

where Gen denotes the MLLM solver that integrates visual perception, question understanding, and dual-stream memory guidance. The retrieved logical guidelines RiL provide structured and context-relevant reasoning frameworks, while the visual guidelines RiV supply explicit perceptual priors. Together, they enable more robust and accurate multimodal reasoning.

#### 4. Experiments 4.1. Experimental Setup

Tasks and Datasets. We evaluate ViLoMem on three multimodal reasoning benchmarks that are particularly sensitive to cumulative visual–logical errors: (1) Hallucination and real-world robustness, which emphasize language hallucination, visual illusion, and spatial grounding; (2) Multimodal mathematical reasoning, which couples logic reasoning with visual grounding; and (3) Vision-dependent knowledge, which requires expert-level visual understanding across multiple disciplines.

HallusionBench [18] diagnoses intertwined language hallucination and visual illusion through 1,129 controlpaired questions; RealWorldQA [44] assesses spatial reasoning over 765 natural scenes; MathVista (mini) [26] and MathVision (mini) [38] test visual-grounded mathematical reasoning across diverse diagrams and competitionstyle problems; MMMU (val) [48] covers 1050 collegelevel questions across six academic domains (Art & Design, Business, Science, Health & Medicine, Humani-

2025/11/14 17:30 Visual Memory Cases

|Question|Logic Memory|Visual Memory|Attention Map|
|---|---|---|---|
|CASE 1<br><br>Subtract all large rubber spheres. Subtract all big shiny cylinders. How many objects are left?|In geometry problems involving perpendicular bisectors, remember that only points lying on the perpendicular bisector segment are guaranteed to be equidistant from the endpoints...|When an object has a uniform, reflective, or metallic-looking surface —even if it appears matte —treat it as metallic... When assessing material properties like 'rubber' versus 'shiny/metallic',<br><br>compare the object's surface texture and reflectivity to other<br><br>clearly labeled objects in the scene...|[Figure 41]|
|CASE 2<br><br>Find the value of the square in the figure.|When identifying angle pairs formed by a transversal intersecting parallel lines, always verify the exact position of the angles: consecutive interior angles lie between the parallel lines on the same side...|Always verify the exact numerical value written inside each geometric region of a diagram, as misreading a digit (e.g., 6 vs. 12) can invalidate the entire logical pattern...|[Figure 42]|
|CASE 3<br><br>If D is the midpoint of line segment AB, then is D the orthocenter of triangle ABC? Please answer yes or no.|When solving problems involving data interpretation from tables or charts, ensure you are addressing the actual question by referencing the relevant data directly, rather than applying unrelated mathematical or theoretical concepts...|/ (No visual memory needed)|[Figure 43]|
|CASE 4<br><br>Does this figure mainly depict a hen and eggs, no potatoes?|When a question asks whether two colors are different in the context of an optical illusion, always consider whether it is asking about objective color values (e.g., RGB or physical properties) rather than perceived appearance...|When comparing the color of objects on a gradient background, verify that perceived differences are not caused by the background's luminance shift by isolating or masking the background to assess the objects' true color...|[Figure 44]|

- Figure 3. Visual memory generation and retrieval examples. Each case shows the original error, the extracted visual pattern, and successful retrieval in analogous scenarios.

ties & Social Science, Tech & Engineering); and MMStar [7] offers 1,500 high-quality samples evaluating visiondependent reasoning across 18 fine-grained dimensions.

Models and Implementation. To assess the effectiveness and generalizability of ViLoMem, we evaluate it across models of varying scale and accessibility: the proprietary GPT-4.1 as a strong closed-source baseline, the open-source Qwen3-VL-235B-A22B-Instruct as a state-of-the-art large multimodal model, and Qwen3-VL-8B-Instruct as a smaller model to test whether memory benefits extend to resourceconstrained settings. For memory generation, we employ Qwen3-235B-A22B-Instruct for logical memory (pure language reasoning analysis) and Qwen3-VL-235B-A22BInstruct for visual memory (image-grounded error attribution). Memory retrieval uses Qwen3-Embedding for text similarity and Qwen2.5-VL-Embedding for image similarity, enabling efficient semantic matching. Additional implementation details are provided in the Appendix.

Evaluation Metrics. We report pass@1 accuracy using VLMEvalKit [12]. When rule-based matching detects potential errors, we apply an LLM-as-a-judge mechanism for verification, enhancing scoring accuracy and reducing false negatives from format variations.

- 4.2. Main Results on Multimodal Benchmarks

ﬁle:///Users/bwh/Documents/CVPR26_VL_Memory/ﬁgures/visual_memory_showcase.html 1/1

Table 1 shows evaluations across six multimodal benchmarks covering mathematical reasoning, hallucination robustness, and visual knowledge understanding. We compare three MLLMs under three configurations: Baseline:

Table 2. Ablation study on the contribution of dual stream memory components. We evaluate GPT-4.1 with different memory configurations on two representative benchmarks.

Method MMMU MathVista

GPT-4.1 (baseline) 74.00 70.40 GPT-4.1 (step) 74.16 74.27

GPT-4.1 (w/o logic memory) 76.64 75.59 GPT-4.1 (w/o visual memory) 76.88 75.66

GPT-4.1 (+ ViLoMem) 77.26 76.88 GPT-4.1 (+ ViLoMem & attention) 78.21 76.87

following the official default prompting setup; Step: using explicit step-by-step reasoning prompts; and +ViLoMem: integrating our dual-stream memory framework. The comparison between Step and +ViLoMem highlights the effectiveness of memory in mitigating de novo reasoning and promoting experience-driven problem solving.

ViLoMem achieves consistent improvements across all models, with particularly notable gains on mathematical reasoning benchmarks. This result aligns with our motivation, as mathematical reasoning tasks demand more visually grounded chains of thoughts. Prior studies have shown that visual perception errors significantly degrade reasoning accuracy [26, 55]. By tracking visual errors and integrating them with logical reasoning, ViLoMem effectively enhances overall mathematical reasoning performance. Among the three MLLMs, GPT-4.1 shows the largest improvement—particularly on MathVision (+6.48) and MathVista (+2.61)—owing to its stronger contextual learning ability and superior capacity to utilize and interpret past errors for solving similar problems. Smaller models benefit more substantially from memory augmentation: Qwen3-VL-8B-Instruct achieves notable gains on MMMU (+4.38) and RealWorldQA (+2.74), indicating that structured memory provides complementary knowledge beyond the model’s limited parametric capacity.

Among the evaluated tasks, improvements on knowledge-intensive benchmarks are moderate, as these tasks primarily rely on factual recall rather than multi-step reasoning. Moreover, manual inspection of the stored memory information from both streams reveals two primary performance bottlenecks. First, when the solver exhibits strong textual bias, over-relying on linguistic reasoning while paying limited attention to visual cues, the resulting reasoning traces contain insufficient visual information for the verifier to generate effective visual memory. Second, when the solver struggles to perceive complex diagrams and generates low-quality visual descriptions, the verifier finds it difficult to identify clear visual errors and tends to attribute all errors to the logical stream, often resulting in mixed memory updates. Therefore, a promising direction for future work is to design more specialized mechanisms to further enhance the decoupling the dual memory streams.

[Figure 45]

Figure 4. Analysis of dual stream memory usage patterns across six benchmarks. (a) Memory generation and retrieval statistics show that visual errors dominate generation (59% to 93%), while retrieval operations significantly exceed generation events. (b) Cross task dependency analysis reveals balanced utilization of both memory streams during retrieval across diverse tasks and models.

- Table 3. Cross model memory transfer analysis. For each solver, we replace its self-generated memory with memories generated by the other two models on the same benchmark.

Method MMMU MathVista

GPT-4.1 (step) 74.16 74.27 GPT-4.1 (+ ViLoMem) 77.26 76.88 GPT-4.1 (+ ViLoMem Cross) 78.21 76.58

Qwen3-VL-235B (step) 75.97 83.66 Qwen3-VL-235B (+ ViLoMem) 79.40 84.98 Qwen3-VL-235B (+ ViLoMem Cross) 79.26 84.21

Qwen3-VL-8B (step) 65.52 77.80 Qwen3-VL-8B (+ ViLoMem) 69.90 77.87 Qwen3-VL-8B (+ ViLoMem Cross) 71.26 79.20

Case Study: Figure 3 illustrates the operation of dualstream memory in practice. Cases 1, 2, and 4 expose a key limitation of logical memory: it retrieves guidelines irrelevant to the visual context (e.g., recalling perpendicular bisector principles when material discrimination is required). Visual memory effectively addresses this gap by identifying surface reflectivity (Case 1), numerical digits in diagrams (Case 2), and background luminance for color perception (Case 4). The attention maps confirm that retrieved visual cues guide the model toward task-relevant regions (Case 2/ 4). Case 3 highlights the plausibility of our memory generation process: when a problem can be solved without visual cues (the question already providing complete visual descriptions), logical memory alone suffices. Overall, visual memory supports perception-intensive tasks, while logical memory governs reasoning-driven problems.

##### 4.3. Ablation Study

We validate the necessity of dual-stream memory by selectively disabling each component on GPT-4.1. As shown in

Table 2, removing either stream consistently degrades performance, confirming that both memory types are essential. Removing logical memory leads to larger drops on MathVista, where systematic reasoning and formula-related errors frequently recur. In contrast, removing visual memory produces comparable degradation across both benchmarks, indicating that visual distraction errors are pervasive in multimodal reasoning tasks. The gap between the single-stream variants and the full ViLoMem model demonstrates their complementarity: the visual and logical streams capture distinct, rather than redundant, error patterns. Augmenting visual memory with question-aware attention maps yields notable gains on MMMU, but only marginal improvements on MathVista, because diagram-based tasks require more fine-grained visual understanding, e.g., smaller-scale vertex attention and higher spatial precision. More detailed analyses are provided in the Appendix.

##### 4.4. Memory Usage Analysis

Figure 4 analyzes memory usage patterns across all benchmarks. Visual memory generation dominates the error collection, accounting for 59%–93% of stored cases in Fig-

- ure 4(a), demonstrating that visual perception remains the primary bottleneck in multimodal reasoning. Despite this generation asymmetry, both streams contribute comparably during retrieval, indicating effective memory reuse. Fig-
- ure 4(b) further confirms consistent dual-stream coordination across all three MLLMs, as reflected by the distribution of translucent retrieval points along the diagonal, indicating balanced contributions from both visual and logical streams. Moreover, our memory mechanism is not biased toward any specific model, as all three models exhibit similar patterns of memory utilization.

- Table 4. Cross benchmark memory generalization analysis. For each benchmark, we exclude its task specific memory and merge memories from all other benchmarks as the retrieval source. Results demonstrate that while cross domain memories provide partial benefits, task aligned memories remain essential for optimal performance.

Method MMMU MathVista MathVision HallusionBench MMStar RealWorldQA

Qwen3-VL-8B (baseline) 66.38* 77.20 48.13* 61.10 70.91 71.50 Qwen3-VL-8B (step) 65.52 77.80 48.35 73.08 70.22 70.85

Qwen3-VL-8B (+ ViLoMem) 69.90 77.87 49.34 73.19 72.13 73.59 Qwen3-VL-8B (+ ViLoMem Cross) 65.14 76.10 50.00 70.66 70.93 71.63

##### 4.5. Cross-Model Memory Transfer

To evaluate the reusability and composability of the dualstream memory framework, we conduct cross-model memory transfer experiments where each solver retrieves memories generated by other models. As shown in Table 3, the 8B model benefits most from cross-model memories (+1.36 on MMMU, +1.33 on MathVista), surpassing its selfgenerated performance, indicating that memories distilled from stronger models encode higher-quality error patterns and generalization strategies. In contrast, larger models show comparable or slightly reduced performance, as their reasoning capabilities already yield near-optimal memory formation. These results highlight that dual-stream memory supports effective knowledge distillation from stronger to weaker models, enabling collaborative learning without explicit fine-tuning or ensembling.

##### 4.6. Cross-Benchmark Memory Generalization

We assess memory transferability across task domains using Qwen3-VL-8B-Instruct. For each target benchmark in Table 4, we exclude its task-specific memory bank and instead retrieve from memories accumulated across all other benchmarks. The results reveal substantial heterogeneity: MathVision and RealWorldQA benefit from cross-domain memories, as both require strong spatial reasoning. In contrast, tasks with large domain gaps, such as MathVista and HallusionBench (diagram-grounded vs. natural image reasoning), exhibit conflicts in memory utilization. Overall, the persistent gap between cross-domain and ViLoMem underscores that task-aligned memories are essential for optimal performance, validating our design choice to maintain distinct memory banks for different domains.

##### 4.7. Memory Scalability

To stress-test long-term memory growth, we construct a progressive memory pool by sequentially accumulating memories from four math-domain benchmarks: MathGlance [32], MathVista [26], MathVision [38], and MathVerse [54]. This sequence follows a visual-to-reasoning progression: we first build memory on diagram perceptual tasks (MathGlance) and then transition to visual grounded reasoning tasks. The resulting pool contains ∼3k samples and 150k memory tokens. We then evaluate on the unseen WeMath [27]. This setup simulates a realistic long-horizon

Table 5. Memory scalability on WeMath. Memory is progressively accumulated from four math-domain benchmarks (MathGlance→MathVista→MathVision→MathVerse) and evaluated on the unseen WeMath. “Direct” denotes memory generated directly on WeMath itself.

#Tokens (samples) 15k (0.1k) 50k (0.5k) 87k (1k) 105k (2k) 150k (3k) Direct WeMath Acc. (%) 72.53 72.82 73.91 74.33 74.58 73.85

deployment where the memory pool grows incrementally across diverse tasks.

As shown in Table 5, accuracy on WeMath consistently improves as memory scales from 15k to 150k tokens (72.53→74.58), demonstrating effective long-term memory scaling without catastrophic forgetting. Notably, crossbenchmark progressive memory (74.58) even outperforms memory generated directly on WeMath itself (73.85, “Direct”), validating that abstract reasoning patterns accumulated across diverse math tasks transfer effectively to unseen problems.

#### 5. Conclusion

We introduce ViLoMem, a dual-stream memory framework that separately models visual distraction patterns and logical hallucination errors for multimodal large language models. Inspired by human semantic memory systems, ViLoMem coordinates visual and logical memory streams through specialized retrieval strategies and grow-and-refine update mechanisms. Comprehensive evaluations across six multimodal benchmarks demonstrate consistent improvements, with particularly pronounced gains on mathematical reasoning tasks where visual-logical coupling is most acute. Ablation studies confirm that both memory streams are complementary; joint operation enables synergistic error correction. Further analyses reveal heterogeneous crossdomain transfer behavior—task-aligned domains benefit from shared memory, whereas domain-mismatched tasks exhibit mild interference. Moreover, cross-model transfer experiments highlight that our memory can distill error patterns and reasoning strategies from stronger models to smaller ones, demonstrating its potential as a lightweight knowledge-sharing mechanism without explicit fine-tuning. By enabling progressive error reduction without catastrophic forgetting, ViLoMem builds a foundation for continual learning in multimodal reasoning.

#### Acknowledgment

This work was supported by National Natural Science Foundation of China (Grant No. 62425603).

#### References

- [1] Rishabh Agarwal, Avi Singh, Lei Zhang, Bernd Bohnet, Luis Rosias, Stephanie Chan, Biao Zhang, Ankesh Anand, Zaheer Abbas, Azade Nova, et al. Many-shot in-context learning. Advances in Neural Information Processing Systems, 37:76930–76966, 2024. 2
- [2] Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J Ryan, Meng Jiang, et al. Gepa: Reflective prompt evolution can outperform reinforcement learning. arXiv preprint arXiv:2507.19457, 2025. 2
- [3] Keivan Alizadeh, Seyed Iman Mirzadeh, Dmitry Belenko, S Khatamifard, Minsik Cho, Carlo C Del Mundo, Mohammad Rastegari, and Mehrdad Farajtabar. Llm in a flash: Efficient large language model inference with limited memory. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12562–12584, 2024. 2
- [4] Ali Ayub, Chrystopher L Nehaniv, and Kerstin Dautenhahn. Interactive continual learning architecture for long-term personalization of home service robots. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 11289–11296. IEEE, 2024. 2
- [5] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 1
- [6] Yuxuan Cai, Yipeng Hao, Jie Zhou, Hang Yan, Zhikai Lei, Rui Zhen, Zhenhua Han, Yutao Yang, Junsong Li, Qianjun Pan, et al. Building self-evolving agents via experiencedriven lifelong learning: A framework and benchmark. arXiv preprint arXiv:2508.19005, 2025. 2
- [7] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Mmstar: Are we on the right way for evaluating large visionlanguage models. arXiv preprint arXiv:2403.20330, 5, 2024. 6
- [8] Qizhou Chen, Taolin Zhang, Xiaofeng He, Dongyang Li, Chengyu Wang, Longtao Huang, et al. Lifelong knowledge editing for llms with retrieval-augmented continuous prompt learning. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 13565– 13580, 2024. 2
- [9] Alex Clarke and Lorraine K Tyler. Object-specific semantic coding in human perirhinal cortex. Journal of Neuroscience, 34(14):4766–4775, 2014. 3
- [10] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 1

- [11] Julien Dirani and Liina Pylkk¨anen. Meg evidence that modality-independent conceptual representations contain semantic and visual features. Journal of Neuroscience, 44(27),

2024. 2

- [12] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM international conference on multimedia, pages 11198–11201, 2024. 6, 4
- [13] Yue Fan, Xiaojian Ma, Rujie Wu, Yuntao Du, Jiaqi Li, Zhi Gao, and Qing Li. Videoagent: A memory-augmented multimodal agent for video understanding. In European Conference on Computer Vision, pages 75–92. Springer, 2024. 2
- [14] Jizhan Fang, Xinle Deng, Haoming Xu, Ziyan Jiang, Yuqi Tang, Ziwen Xu, Shumin Deng, Yunzhi Yao, Mengru Wang, Shuofei Qiao, et al. Lightmem: Lightweight and efficient memory-augmented generation. arXiv preprint arXiv:2510.18866, 2025. 2
- [15] Jinyuan Fang, Yanwen Peng, Xi Zhang, Yingxu Wang, Xinhao Yi, Guibin Zhang, Yi Xu, Bin Wu, Siwei Liu, Zihao Li, et al. A comprehensive survey of self-evolving ai agents: A new paradigm bridging foundation models and lifelong agentic systems. arXiv preprint arXiv:2508.07407, 2025. 1
- [16] Huan-ang Gao, Jiayi Geng, Wenyue Hua, Mengkang Hu, Xinzhe Juan, Hongzhang Liu, Shilong Liu, Jiahao Qiu, Xuan Qi, Yiran Wu, et al. A survey of self-evolving agents: On path to artificial super intelligence. arXiv preprint arXiv:2507.21046, 1, 2025. 1
- [17] Shuzheng Gao, Chaozheng Wang, Cuiyun Gao, Xiaoqian Jiao, Chun Yong Chong, Shan Gao, and Michael Lyu. The prompt alchemist: Automated llm-tailored prompt optimization for test case generation. arXiv preprint arXiv:2501.01329, 2025. 2
- [18] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14375–14385, 2024. 5
- [19] Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. Glm-4.5 v and glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv preprint arXiv:2507.01006, 2025. 1
- [20] Xin Jiang, Hao Tang, Junyao Gao, Xiaoyu Du, Shengfeng He, and Zechao Li. Delving into multimodal prompting for fine-grained visual classification. In Proceedings of the AAAI conference on artificial intelligence, pages 2570–2578,

2024. 1

- [21] Xin Jiang, Hao Tang, and Zechao Li. Global meets local: Dual activation hashing network for large-scale fine-grained image retrieval. IEEE Transactions on Knowledge and Data Engineering, 36(11):6266–6279, 2024. 1
- [22] Philipp Kuhnke, Curtiss A Chapman, Vincent KM Cheung, Sabrina Turker, Astrid Graessner, Sandra Martin, Kathleen A

- Williams, and Gesa Hartwigsen. The role of the angular gyrus in semantic cognition: a synthesis of five functional neuroimaging studies. Brain Structure and Function, 228 (1):273–291, 2023. 3
- [23] Matthew A Lambon Ralph, Karen Sage, Roy W Jones, and Emily J Mayberry. Coherent concepts are computed in the anterior temporal lobes. Proceedings of the National Academy of Sciences, 107(6):2717–2722, 2010. 3
- [24] Xiaofan Li, Yifu Zhang, and Xiaoqing Ye. Drivingdiffusion: Layout-guided multi-view driving scenarios video generation with latent diffusion model. In European Conference on Computer Vision, pages 469–485. Springer, 2024. 1
- [25] Xiaofan Li, Chenming Wu, Yanpeng Sun, Jiaming Zhou, Delin Qu, Yansong Qu, Weihao Bo, Haibao Yu, and Dingkang Liang. Fvar: Visual autoregressive modeling via next focus prediction. arXiv preprint arXiv:2511.18838,

2025. 1

- [26] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In The Twelfth International Conference on Learning Representations. 1, 5, 6, 8
- [27] Runqi Qiao, Qiuna Tan, Guanting Dong, MinhuiWu MinhuiWu, Chong Sun, Xiaoshuai Song, Jiapeng Wang, Zhuoma Gongque, Shanglin Lei, Yifan Zhang, et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning? In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 20023–20070, 2025. 8
- [28] C´esar Santos, Fumio Machida, and Ermeson Andrade. Experimental investigation of memory-related software aging in llm systems. Journal of Systems and Software, page 112653, 2025. 2
- [29] Rulin Shao, Jacqueline He, Akari Asai, Weijia Shi, Tim Dettmers, Sewon Min, Luke Zettlemoyer, and Pang W Koh. Scaling retrieval-based language models with a trillion-token datastore. Advances in Neural Information Processing Systems, 37:91260–91299, 2024. 2
- [30] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36:8634–8652, 2023. 2
- [31] Yanpeng Sun, Jing Hao, Ke Zhu, Jiang-Jiang Liu, Yuxiang Zhao, Xiaofan Li, Gang Zhang, Zechao Li, and Jingdong Wang. Descriptive caption enhancement with visual specialists for multimodal perception. arXiv preprint arXiv:2412.14233, 2024. 1
- [32] Yanpeng Sun, Shan Zhang, Wei Tang, Aotian Chen, Piotr Koniusz, Kai Zou, Yuan Xue, and Anton van den Hengel. Mathglance: Multimodal large language models do not know where to look in mathematical diagrams. arXiv e-prints, pages arXiv–2503, 2025. 1, 8, 3
- [33] Mirac Suzgun, Mert Yuksekgonul, Federico Bianchi, Dan Jurafsky, and James Zou. Dynamic cheatsheet: Testtime learning with adaptive memory. arXiv preprint arXiv:2504.07952, 2025. 1, 2

- [34] Zhen Tan, Jun Yan, I-Hung Hsu, Rujun Han, Zifeng Wang, Long Le, Yiwen Song, Yanfei Chen, Hamid Palangi, George Lee, et al. In prospect and retrospect: Reflective memory management for long-term personalized dialogue agents. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8416–8439, 2025. 1
- [35] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9568–9578, 2024. 1
- [36] Boshi Wang, Weijian Xu, Yunsheng Li, Mei Gao, Yujia Xie, Huan Sun, and Dongdong Chen. Improving code localization with repository memory. arXiv preprint arXiv:2510.01003, 2025. 2
- [37] Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen, and Sercan O Arik. Astute rag: Overcoming imperfect retrieval augmentation and knowledge conflicts for large language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 30553–30571, 2025. 2
- [38] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 37:95095–95169, 2024. 5, 8
- [39] YuAn Wang, Xiaofan Li, Chi Huang, Wenhao Zhang, Hao Li, Bosheng Wang, Xun Sun, and Jun Wang. Faithfusion: Harmonizing reconstruction and generation via pixel-wise information gain. arXiv preprint arXiv:2511.21113, 2025. 1
- [40] Rong Wu, Xiaoman Wang, Jianbiao Mei, Pinlong Cai, Daocheng Fu, Cheng Yang, Licheng Wen, Xuemeng Yang, Yufan Shen, Yuxin Wang, et al. Evolver: Self-evolving llm agents through an experience-driven lifecycle. arXiv preprint arXiv:2510.16079, 2025. 2
- [41] Shengqiong Wu, Hao Fei, Liangming Pan, William Yang Wang, Shuicheng Yan, and Tat-Seng Chua. Combating multimodal llm hallucination via bottom-up holistic reasoning. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 8460–8468, 2025. 2
- [42] Yingsheng Wu, Yuxuan Gu, Xiaocheng Feng, Weihong Zhong, Dongliang Xu, Qing Yang, Hongtao Liu, and Bing Qin. Extending context window of large language models from a distributional perspective. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 7288–7301, 2024. 2
- [43] Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, Huazuo Gao, Yiyang Ma, Chengyue Wu, Bingxuan Wang, et al. Deepseek-vl2: Mixture-ofexperts vision-language models for advanced multimodal understanding. arXiv preprint arXiv:2412.10302, 2024. 1
- [44] xAI. Grok-1.5 vision preview. https://x.ai/news/ grok-1.5v, 2024. 5
- [45] Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem: Agentic memory for llm agents. arXiv preprint arXiv:2502.12110, 2025. 2

- [46] Yutao Yang, Jie Zhou, Junsong Li, Qianjun Pan, Bihao Zhan, Qin Chen, Xipeng Qiu, and Liang He. Reinforced interactive continual learning via real-time noisy human feedback. arXiv preprint arXiv:2505.09925, 2025. 2
- [47] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, 2022. 2
- [48] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9556–9567,

2024. 5

- [49] Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Zhi Huang, Carlos Guestrin, and James Zou. Textgrad: Automatic” differentiation” via text. arXiv preprint arXiv:2406.07496, 2024. 2
- [50] Guibin Zhang, Hejia Geng, Xiaohang Yu, Zhenfei Yin, Zaibin Zhang, Zelin Tan, Heng Zhou, Zhong-Zhi Li, Xiangyuan Xue, Yijiang Li, et al. The landscape of agentic reinforcement learning for llms: A survey. Transactions on Machine Learning Research, . 2
- [51] Jiarui Zhang, Mahyar Khayatkhoei, Prateek Chhikara, and Filip Ilievski. Mllms know where to look: Training-free perception of small visual details with multimodal llms. In The Thirteenth International Conference on Learning Representations, . 4
- [52] Kai Zhang, Xiangchao Chen, Bo Liu, Tianci Xue, Zeyi Liao, Zhihan Liu, Xiyao Wang, Yuting Ning, Zhaorun Chen, Xiaohan Fu, et al. Agent learning via early experience. arXiv preprint arXiv:2510.08558, 2025. 2
- [53] Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, et al. Agentic context engineering: Evolving contexts for self-improving language models. arXiv preprint arXiv:2510.04618, 2025. 1, 2
- [54] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pages 169–186. Springer, 2024. 8
- [55] Shan Zhang, Aotian Chen, Yanpeng Sun, Jindong Gu, YiYu Zheng, Piotr Koniusz, Kai Zou, Anton Van Den Hengel, and Yuan Xue. Primitive vision: Improving diagram understanding in mllms. In International Conference on Machine Learning, pages 74732–74755. PMLR, 2025. 2, 6, 1
- [56] Shan Zhang, Aotian Chen, Kai Zou, Jindong Gu, Yuan Xue, and Anton van den Hengel. Hierarchical process reward models are symbolic vision learners. arXiv preprint arXiv:2512.03126, 2025. 1, 2
- [57] Yichi Zhang, Zhuo Chen, Lingbing Guo, Yajing Xu, Min Zhang, Wen Zhang, and Huajun Chen. Abstractive visual understanding of multi-modal structured knowledge: A

- new perspective for mllm evaluation. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 12323–12332, 2025. 1
- [58] Zeyu Zhang, Quanyu Dai, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. A survey on the memory mechanism of large language modelbased agents. ACM Transactions on Information Systems, 43

(6):1–47, 2025. 1, 2

- [59] Zijia Zhao, Yuqi Huo, Tongtian Yue, Longteng Guo, Haoyu Lu, Bingning Wang, Weipeng Chen, and Jing Liu. Efficient motion-aware video mllm. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24159– 24168, 2025. 1
- [60] Chenyue Zhou, Mingxuan Wang, Yanbiao Ma, Chenxu Wu, Wanyi Chen, Zhe Qian, Xinyu Liu, Yiwei Zhang, Junhao Wang, Hengbo Xu, et al. From perception to cognition: A survey of vision-language interactive reasoning in multimodal large language models. arXiv preprint arXiv:2509.25373, 2025. 2
- [61] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 1

## ViLoMem: Agentic Learner with Grow-and-Refine Multimodal Semantic Memory

### Supplementary Material

#### 6. Additional Results and Ablation Study

##### 6.1. Integration with more models

To verify the flexibility of ViLoMem, we extend our evaluation beyond the main experiments to recent reasoningenhanced models, including GLM-4.1v [19], InternVL338B [61], and Gemini 2.5 [10]. As shown in Table 6, ViLoMem demonstrates robust adaptability across different architecture designs and inference regimes, consistently improving performance over both baseline and step-bystep configurations. This pattern echoes our observations in the main paper that visual perception remains a dominant bottleneck for multimodal reasoning [26, 55] and that decoupling visual distraction from logical hallucination yields complementary gains across tasks. Notably, models equipped with “thinking” or long-chain reasoning capabilities exhibit superior compatibility with the step-by-step format required for memory retrieval: their extended inference process allows for tighter integration of retrieved visual and logical guidelines into the reasoning chain, enabling them to correct potential errors before they propagate. These results suggest that ViLoMem is particularly well-suited to models with strong deliberative reasoning, while still offering consistent benefits to smaller or less capable solvers.

##### 6.2. Attention Mechanism Ablation

Table 7 presents the ablation study of the attention mechanism. In general, the integration of attention maps yields consistent performance gains across hallucination and general reasoning benchmarks (e.g., HallusionBench, MMStar), corroborating the critical importance of visual memory in refining perceptual grounding. However, we observe a performance plateau or marginal decline on mathematicscentric datasets (MathVista and MathVision). We attribute this limitation to two primary factors: (1) Visualization Precision: Current attention visualization methods struggle to faithfully preserve fine-grained geometric structures and chart details, which are essential for mathematical reasoning. (2) Contextual Interpretation: While serving as an auxiliary image to enhance visual context, the attention map imposes higher demands on the model’s intrinsic capability to interpret heatmap overlays. The benefit of this enriched context is contingent on the model’s ability to align these explicit visual cues with the raw image features without information loss.

##### 6.3. Additional Case Study

Figure 5 summarizes representative qualitative cases. For many vision-intensive questions (e.g., traffic-light color, visible portion of the sun, object localization, and opticalillusion setups), logical memory is either not retrieved or fails to offer useful guidance, while visual memory provides concrete viewing strategies such as checking the actual illuminated region, reading tiny objects and relative positions from the viewer’s frame, or isolating targets from distracting backgrounds. In these cases, attention maps concentrate on the queried regions (e.g., the active light, visible solar arc, or relevant segments), so that the retrieved visual guidelines directly steer the solver toward task-relevant evidence.

For geometry and chart-reading tasks, visual and logical memories are complementary: logical memory provides reusable rules for measurement and graph interpretation, while visual memory focuses on concrete inspection behaviors such as aligning with gridlines, following step edges, or checking true line orientation under strong illusions. Together, these cases highlight a clear division of labor: visual memory governs “where to look” and mitigates systematic perceptual traps, whereas logical memory refines “how to reason” once the correct visual evidence has been attended.

##### 6.4. Comparison with Existing Memory Methods

We benchmark ViLoMem against state-of-the-art memory mechanisms [33, 53]. While the original DynamicCheetsheet [33] employs cumulative memory, its unbounded context growth is infeasible for our large-scale setting (approx. 1,000 cases per benchmark), so we adopt the retrieval-based configuration from the open-source Dynamic-Cheetsheet codebase, which follows the similar methodology as ACE [53]. For a fair multimodal comparison, we replicate the official prompt structure and use the same MLLM for both memory generation and inference. In this setup, the retrieval module relies purely on text similarity without image-aware matching.

Experimental results in Table 7 show that this direct adaptation of logical memory methods is suboptimal in multimodal settings and can even underperform the baseline, especially for smaller models. In practice, such textonly retrieval often surfaces visually dissimilar examples with similar questions, resurfacing prior misperceptions as salient “hints” that misdirect attention away from the correct regions of the current problem. Qualitative inspection further reveals that Dynamic-Cheetsheet and ACE are tai-

- Table 6. Additional evaluation results on GLM4.1v, InternVL3-38B, and Gemini2.5-flash across six multimodal reasoning benchmarks. Models with “(step)” and “(+ ViLoMem)” are prompted by step-by-step reasoning. Results demonstrate consistent improvements from ViLoMem across diverse model architectures.

Method MMMU (dev) MathVista (mini) MathVision (mini) HallusionBench MMStar RealWorldQA

GLM4.1v (baseline) 69.14 72.57 56.88 73.08 72.90 73.33 GLM4.1v (step) 70.29 73.47 58.22 72.77 73.40 72.54 GLM4.1v (+ ViLoMem) 71.52 73.97 61.51 74.02 73.47 72.68

InternVL3-38B (baseline) 62.92 70.80 35.53 67.40 69.33 71.99 InternVL3-38B (step) 64.18 71.90 35.56 71.50 67.80 72.42 InternVL3-38B (+ ViLoMem) 65.97 73.80 36.84 72.34 69.73 73.20

Gemini2.5-flash (baseline) 72.18 81.10 53.21 72.67 72.07 76.99 Gemini2.5-flash (step) 71.90 81.41 53.94 76.34 72.40 71.50 Gemini2.5-flash (+ ViLoMem) 72.86 83.40 58.22 78.33 73.20 76.42

- Table 7. Comprehensive ablation study and comparison with existing memory methods across six multimodal reasoning benchmarks. We compare ViLoMem with attention mechanism variants and the Dynamic-Cheetsheet [33] baseline adapted for multimodal tasks.

Method MMMU (dev) MathVista (mini) MathVision (mini) HallusionBench MMStar RealWorldQA GPT-4.1

baseline 74.00 70.40 46.12* 58.50 69.80 73.72 step 74.16 74.27 47.47 74.44 70.43 72.03 + dynamic-cheetsheet 70.95 73.87 48.68 75.30 68.68 70.13 + ViLoMem 77.26 76.88 53.95 75.29 72.43 74.38 + ViLoMem & attention 78.21 76.87 50.66 75.73 71.76 74.38

Qwen3-VL-235B-A22B-Instruct

baseline 78.70 84.90 61.28* 63.20 78.40 79.30 step 75.97 83.66 62.17 74.58 76.16 78.66 + dynamic-cheetsheet 72.13 83.25 60.06 70.62 75.49 77.11 + ViLoMem 79.40 84.98 62.83 75.21 78.31 77.22 + ViLoMem & attention 78.14 83.87 60.86 75.95 78.46 77.88

Qwen3-VL-8B-Instruct

baseline 66.38* 77.20 48.13* 61.10 70.91 71.50 step 65.52 77.80 48.35 73.08 70.22 70.85 + dynamic-cheetsheet 63.39 74.92 46.81 68.39 69.12 69.98 + ViLoMem 69.90 77.87 49.34 73.19 72.13 73.59 + ViLoMem & attention 67.52 77.07 48.72 74.87 72.67 73.46

lored to code- or logic-centric schemas: even when driven by an MLLM, they mainly produce fine-grained corrections of specific visual details (digits, colors, marks) rather than robust guidance on how to inspect diagrams. These detaillevel cues lack stable visual grounding and easily conflict with the actual image, inducing additional hallucinations that smaller models are particularly vulnerable to. This contrast highlights the need for ViLoMem’s decoupled visual stream and question-aware retrieval, which explicitly organize and retrieve perception-oriented error patterns instead of repurposing logic-only memories.

Retrieval Efficiency at Scale. We further analyze how the two-stage retrieval pipeline scales with growing memory.

- Table 8 compares accuracy and per-case latency (ms) on WeMath for: no memory (S0), visual-only retrieval (S1),

Table 8. Two-stage retrieval analysis on WeMath. S0: no memory (baseline); S1: visual retrieval; S2: text retrieval; S⊕: S1 + S2 (two-stage). Latency in ms per case.

Stage / Acc. % (Latency ms) S0 S1 S2 S⊕

87k tokens 72.07 72.94 (19.1) 73.22 (22.8) 73.91 (20.4) 150k tokens 72.07 73.72 (60.7) 73.41 (115) 74.58 (61.6)

tency increases from 22.8ms to 115ms due to full-corpus text matching, while S⊕ latency only rises from 20.4ms to 61.6ms, a 22.3× reduction compared to S2 alone. Due to S1 first narrows the candidate set via efficient image embedding search (1024-dim), allowing S2 to rerank only the top candidates. Crucially, S⊕ also achieves the highest accuracy (74.58%), as the two stages provide complementary filtering: S1 captures visual similarity while S2 refines by semantic relevance.

text-only retrieval (S2), and two-stage retrieval (S⊕ = S1 + S2). As memory grows from 87k to 150k tokens, S2 la-

- Table 9. Efficiency comparison between ViLoMem and DynamicCheetsheet (DC) on MathVista. ViLoMem achieves superior accuracy with substantially lower retrieval cost and storage.

Method Acc. (%) Retrieval (ms) Storage (MB) #Memory (tokens) Baseline 70.40 0 0 0 DC 73.87 325 18.44 221K ViLoMem 76.88 120 6.12 73K

We further compare the computational efficiency of ViLoMem against Dynamic-Cheetsheet (DC) on MathVista in Table 9. ViLoMem achieves higher accuracy (+3.01) while requiring significantly lower retrieval latency (−63.1%) and storage cost (−66.8%). This efficiency stems from our selective memory update strategy: ViLoMem only generates memory entries for incorrect answers, whereas DC updates memory for every case, leading to a bloated memory pool (221K vs. 73K tokens) with proportionally higher retrieval overhead.

##### 6.5. Failure Case Analysis

To understand the limitations of ViLoMem, we analyze cases on MMMU (1,041 samples, GPT-4.1 as solver) where the baseline answers correctly but ViLoMem fails. Overall, ViLoMem achieves a net gain of +8.86% over baseline; however, we identify 66 regression cases where memory retrieval hurts performance. Among these, 33.3% (22 cases) are caused by generic visual memory: the retrieved visual cues are only weakly relevant to the specific image and question, lacking image-specific and questionspecific adaptation, which distracts reasoning despite being knowledge-correct. The remaining 66.7% (44 cases) are caused by empty retrieval: no memory is retrieved at all (0 matched entries), meaning the model receives no memoryaugmented guidance and instead relies on a step-by-step prompt that may differ from the baseline’s default prompting. Notably, no failures occur when both visual and logical memory are successfully retrieved, suggesting that the dual-stream retrieval mechanism is reliable when sufficient memory coverage exists.

##### 6.6. Visual Memory Characterization

As shown in Table 2, ViLoMem achieves 77.26% on MMMU with GPT-4.1, with both visual and logical memory contributing to the overall gain. To understand which types of tasks benefit most from each memory stream, we further analyze the 835 subject-matched samples across MMMU’s six academic disciplines. Table 10 reports the improvement from the full system over baseline (∆(B→VLM)), and the independent contribution of each memory stream: ∆(vis) = ViLoMem − w/o visual, and ∆(log) = ViLoMem − w/o logic.

The results reveal clear discipline-dependent memory utilization patterns:

Table 10. Per-discipline memory contribution analysis on MMMU (GPT-4.1, 835 subject-matched samples). ∆(vis) and ∆(log) measure the independent contribution of visual and logical memory, respectively. Bold indicates the dominant memory stream.

Discipline N ∆(B→VLM) ∆(vis) ∆(log)

Tech & Engineering 205 +12.7 +9.8 +5.4 Health & Medicine 99 +10.1 +1.0 +8.1 Humanities & Social Sci. 137 +9.5 +0.7 +0.0 Science 136 +8.1 +3.7 +4.4 Art & Design 93 +3.2 +1.1 +4.3 Business 165 +0.0 −3.6 −4.2

Visual memory dominates in Tech & Engineering (∆(vis)=+9.8 vs. ∆(log)=+5.4). At the subject level, Energy & Power (+20.0), Math (+19.4), Agriculture (+14.3), and Mechanical Engineering (+12.5) benefit most from visual memory. These subjects feature specialized diagrams (circuit schematics, engineering drawings, mathematical plots, microscopic images) that require recognizing domain-specific visual patterns beyond the model’s general vision capability.

Logical memory dominates in Health & Medicine (∆(log)=+8.1 vs. ∆(vis)=+1.0). Diagnostics & Laboratory Medicine shows the largest overall gain (+24.2 from baseline), but the improvement is primarily driven by logical memory (+9.1), indicating that medical reasoning chainsrather than visual perception—are the main bottleneck.

Both streams are ineffective for Business (∆(vis)=−3.6, ∆(log)=−4.2). The model’s baseline visual encoder already handles standard business charts and tables well, and the retrieved memory introduces noise that harms performance. This highlights that memory augmentation is most valuable when the task exceeds the model’s inherent capabilities.

##### 6.7. Results on Visual Perception Benchmarks

On the math perception benchmark MathGlance [32], ViLoMem generates 166 visual and 16 logic memory entities, achieving +1.87% average accuracy improvement over the no-memory baseline on plane/solid geometry and graph tasks. Note that purely perceptual benchmarks such as OCRBench and DocVQA are not directly applicable to our framework, as they use scoring-based evaluation (character matching) rather than binary correctness judgments, which prevents the error-driven memory generation process.

#### 7. Additional Experimental Details

This section provides additional implementation details that complement the experimental setup.

Model Deployment. For open-source models, we deploy most checkpoints using vLLM for efficient batched inference. Due to its scale, Qwen3-VL-235B-A22B-Instruct is accessed via its official API instead of local deploy-

ment, and all proprietary models (e.g., GPT-4.1, Gemini 2.5 flash) are evaluated through their corresponding APIs. For API-based evaluations, certain images or prompts may be flagged as unsafe by the provider’s safety filters and thus rejected, which introduces a small amount of noise into the reported scores.

Decoding Hyperparameters. Unless otherwise specified, we use a temperature of 0.7 and a maximum generation length of 8,192 tokens for all models. Within our memory pipeline, the maximum generation length is set to 1,024 tokens for problem analysis and 2,048 tokens for memory generation to balance expressiveness and efficiency. Baseline evaluations directly feed benchmark questions to the models without additional prompts, whereas the Step configuration prepends a simple step-by-step system prompt; the full template is shown in Figure 6.

Attention Map Generation. Attention maps are generated following the training-free small-detail perception framework of Zhang et al. [51], instantiated with Qwen2.5-VL3B as the backbone model. This setup produces token-level saliency over input images, which we overlay as heatmaps to visualize and interpret visual memory retrieval.

Evaluation Protocol. We adopt VLMEvalKit [12] as the primary evaluation framework. When automatic matching fails or produces ambiguous results (e.g., due to formatting variations), we further apply Math-Verify and an LLMas-a-judge protocol to reduce sensitivity to output formatting. The judge model is Qwen3-8B-Instruct, which assesses whether a model’s response is semantically correct with respect to the reference answer.

#### 8. Prompt Templates

We provide the full prompt templates used in our framework, including the step-by-step reasoning prompt used in the Step configuration (Figure 6), the Problem Analysis Prompt (Figure 7), the Logical Memory Generation Prompt

- (Figure 8), and the Visual Memory Generation Prompt
- (Figure 9), together with the LLM-as-a-judge verification prompt (Figure 10).

|Question|Logic Memory|Visual Memory|Attention Map|
|---|---|---|---|
|CASE 1<br><br>Is the traffic light green?|/|When counting illuminated traffic lights, verify the color of each light individually and confirm it is part of a traffic signal assembly, not a decorative or non-traffic light fixture.|[Figure 46]|
|CASE 2<br><br>What percent of the sun is showing?|When calculating the range of a data set, always double-check that the minimum and maximum values are correctly identified by reviewing all data points, especially when values repeat or are close in magnitude....|When identifying 'tiny thing' or relative positions, always verify object size and spatial layout from the viewer’s perspective; matte finish and right-side positioning must be visually confirmed, not assumed....|[Figure 47]|
|CASE 3<br><br>Move the ruler to measure the length of the pencil to the nearest inch. The pencil is about (_) inches long.|When solving geometry problems involving squares inscribed in circles or angles formed by intersecting chords, verify key relationships such as the diagonal of the square equaling the circle's diameter and the angle measure being half the sum of the intercepted arcs.....|Always verify the exact starting point of the object being measured on the ruler; do not assume alignment with 0 cm unless visually confirmed, as the object may begin at a nonzero mark....|[Figure 48]|
|CASE 4<br><br>Are blue lines in the image parallel? Yes or No|When applying the Corresponding Angles Postulate, always verify that the angles lie on the same side of the transversal and in matching positions at each intersection.....|When assessing parallelism of lines surrounded by diagonal patterns, use a ruler or grid overlay to verify true orientation and spacing, as the background can induce a perceptual illusion of verticality or parallelism.<br><br>When comparing line lengths in an image, measure or visually align them directly rather than relying on assumptions about known optical illusions, as the actual lengths may differ from the illusion's typical setup....|[Figure 49]|
|CASE 5<br><br>Where is the sheep?|/|When identifying objects relative to others in a scene, always verify both the object’s identity and its spatial position (left/right/front/back) by comparing their actual locations in the image, not assumptions based on size or context.|[Figure 50]|
|CASE 6<br><br>What is the position of the sink relative to the refrigerator?|When a question asks for the direction of object A relative to object B, ensure the reference frame is correctly oriented: "A in relation to B" means you start from B and determine where A lies, not the reverse. Always double-check the subject and reference point in directional questions to avoid reversing the relationship.|When identifying objects relative to others in a scene, always verify both the object’s identity and its spatial position (left/right/front/back) by comparing their actual locations in the image, not assumptions based on size or context.|[Figure 51]|
|CASE 7<br><br>What is the value of the smallest individual bar in the whole chart?|When interpreting values on a logarithmic scale, remember that tick marks represent powers of 10, and values between 102 and 103 range from 100 to 1000; always verify whether a bar exceeds a linear threshold like 100 by estimating its actual value, not just its position relative to 102...|On a logarithmic scale, a bar ending exactly at a labeled tick (e.g., 102) represents that exact value, not a value greater than it; always check if the bar exceeds the tick mark to determine if it's strictly larger....|[Figure 52]|
|CASE 8<br><br>When does the function value first reach 2?|When matching a correctly identified element (e.g., a region, value, or object) to a multiplechoice option, always verify that the letter of the choice corresponds to the correct labeled item in the diagram or question, not just the reasoning outcome. Double-check the mapping between your conclusion and the answer options to avoid selection errors....|When interpreting a step function graph, always verify the exact y-value of each horizontal segment by aligning it with the yaxis gridlines, rather than assuming values based on adjacent steps or integer patterns.<br><br>When identifying the y-intercept on a graph, always trace the curve to where it crosses the y-axis (x=0) and read the exact y-value at that point, not the vertex or any nearby grid line....|[Figure 53]|
|CASE 9<br><br>Does this figure mainly depict a hen and eggs, no potatoes?|When a question asks whether two colors are different in the context of an optical illusion, always consider whether it is asking about objective color values (e.g., RGB or physical properties) rather than perceived appearance; remember that illusions affect perception, not necessarily reality.|When comparing the color of objects on a gradient background, verify that perceived differences are not caused by the background’s luminance shift by isolating or masking the background to assess the objects’ true color...|[Figure 54]|

###### Figure 5. Showcase of representative cases demonstrating ViLoMem’s memory generation and retrieval process across different types of multimodal reasoning tasks.

###### Prompt: Step-by-Step Reasoning

Objective: Solve the given problem using a step-by-step process. Expected Output Structure:

- Step 1:
- Step 2:

... Step n: Final Answer: \boxed{answer}

Question:

Figure 6. The step-by-step reasoning system prompt used in the Step configuration.

###### Prompt: Problem Analysis

Objective: Analyze the following problem to identify its subject area and the key concepts, principles, formulas, or laws required for its solution. This analysis will be used to retrieve relevant guiding principles from a knowledge base. Instructions:

- • Do not solve the problem.
- • First, identify the primary subject (e.g., Physics, Chemistry, Biology, Mathematics).
- • Then, list the core concepts or principles involved (e.g., Newton’s Second Law, Conservation of Energy, Stoichiometry, Pythagorean theorem).
- • Keep the analysis concise and focused. Problem: {question} Output Format: Subject: <The primary subject> Key Concepts: <A brief list of key concepts>

Figure 7. The prompt template for analyzing the problem to identify its subject and key concepts.

###### Prompt: Logical Memory Generation

Objective: Analyze the provided incorrect reasoning process for a scientific or mathematical problem. Your goal is to classify the error and, if it is a logical error, generate a high-quality, actionable guideline (a “memory”) to prevent similar mistakes in the future. Context:

- • Problem: {question}
- • Incorrect Reasoning Steps: {reasoning steps}

- • Correct Answer (for reference): {gold answer} Instructions:

- 1. Analyze the Mistake: Carefully review the Incorrect Reasoning Steps against the Problem and Correct Answer to pinpoint the primary mistake.
- 2. Categorize the Error: Classify the error into one of two types:

- • Logical: Any error in the reasoning process itself. This includes calculation mistakes, misapplication of a formula or theorem, logical fallacies, or conceptual misunderstandings. These errors can be identified from the text of the reasoning alone.
- • Non-Logical: An error that stems purely from misinterpreting the visual information in an image. This kind of error can only be confirmed by looking at the image (e.g., misidentifying a shape, reading a value from a graph incorrectly).

- 3. Generate the Guideline (Memory):

- • Only if the error type is Logical, you must generate a guideline.

- • If the error type is Non-Logical, the guideline must be left empty. Guideline Quality Requirements:

- • Be Specific and Concrete: The guideline must target the specific principle, theorem, formula, or reasoning pattern that was misused. Name the concept directly.
- • Be Actionable: Frame it as a clear instruction, a warning, or a rule of thumb (e.g., “Always check...”, “Remember to differentiate between...”, “When X occurs, apply Y...”).
- • Be Generalizable: The advice should be abstracted from the specific numbers and context of this single problem so it can apply to a whole class of similar problems.
- • Keep it Concise: The guideline should be one to two sentences long. Guideline Examples (Good, Specific Examples):
- • (Physics): “When applying the conservation of energy to rolling objects, always include both translational and rotational kinetic energy in the equation.”
- • (Math): “In geometry problems involving tangents to a circle, remember that the radius to the point of tangency is perpendicular to the tangent line.”
- • (Chemistry): “For stoichiometry calculations, always ensure the chemical equation is correctly balanced before determining the molar ratios.”

Output Format (use this exact structure): error type: <“Logical” or “Non-Logical”> analysis summary: <A brief, one-sentence summary of what went wrong.> guideline: <Your 1-2 sentence guideline if the error is “Logical”, otherwise leave this field empty.>

Figure 8. The prompt template for generating logical memories.

###### Prompt: Visual Memory Generation

###### Objective:

You are an expert in visual reasoning and error analysis. Your task is to first describe the provided image objectively, then analyze an incorrect reasoning process to determine if the error stems from misinterpreting that image. If a visual error is found, you must generate a concise, actionable guideline (a “visual memory”) to prevent this mistake in the future.

###### Context:

- • Problem: {question}
- • Incorrect Reasoning Steps: {reasoning steps}

- • Correct Answer (for reference): {gold answer} Attached Image: <image> Thinking Process and Final Output: Your response must follow a strict two-stage process. The first stage is your internal “thought process” which you will write out. The second stage is the final JSON output.

###### Stage 1: Internal Thought Process (Write this out first)

- 1. Describe the Image: Begin by providing an objective, detailed description of the attached image. List all key elements, labels, values, geometric shapes, and their relationships. This description will serve as the “ground truth” for your analysis.
- 2. Analyze for Discrepancies: Compare your image description and the image itself against the text in Incorrect Reasoning Steps. Identify any contradictions, misinterpretations, or omissions.

###### Stage 2: Final JSON Output (Provide ONLY this JSON block as the final answer) After completing your thought process, generate a JSON object based on your analysis. The JSON should adhere to the following structure and guidelines. Guidelines for guideline Generation:

- • The guideline MUST be about how to correctly interpret a specific visual pattern or element.
- • It must be a rule that can be applied to other, similar-looking problems.
- • It should be concise (one to two sentences). Guideline Examples (Good, Specific Visual Memories):
- • (Physics/Diagrams): “In a free-body diagram, always verify that all forces, including friction and normal force, are accounted for before applying Newton’s laws.”
- • (Geometry): “When an angle appears to be a right angle in a diagram, do not assume it is 90 degrees unless it is explicitly marked with a square symbol.”
- • (Chemistry/Molecules): “For complex organic molecules, double-check the placement of double bonds and functional groups as they dictate the molecule’s reactivity.”
- • (Biology/Graphs): “When reading a bar chart, pay close attention to the Y-axis scale and units to avoid misinterpreting the magnitude of the results.” Avoid these types of guidelines (Bad, Non-Visual or Too Vague):
- • “The model made a calculation error.” (This is a logical error, not visual)
- • “You need to look at the image more carefully.” (Not actionable)
- • “The reasoning about the physics was wrong.” (Too general) Final Output Format (use this exact JSON structure):

{

"is_visual_error": true/false, "analysis_summary": "A brief, one-sentence summary of the visual

misinterpretation.",

"guideline": "Your 1-2 sentence visual guideline. Provide this only if is_visual_error is true, otherwise it should be null."

}

Figure 9. The prompt template for generating visual memories.

###### Prompt: LLM-as-a-Judge Verification

Objective: You are an expert answer verification judge. Your task is to determine whether a model prediction matches the gold answer.

###### Core Principle (Critical Rule):

- • All decisions are based only on the gold answer; ignore the quality of the reasoning.
- • If the extracted final answer from the prediction exactly matches the gold answer, set verified=true; otherwise, set verified=false.
- • Do not consider whether the prediction’s reasoning is correct or sensible.
- • Do not give partial credit for “close” answers (e.g., 2 ̸= 9, C ̸= A). Verification Steps:

- 1. Identify the gold answer format. Determine whether the gold answer is:

- • a single letter (A/B/C/D/E) for multiple-choice questions (compare letters only);
- • a number for numerical questions (compare numeric values, ignoring formatting such as 7 vs. 7.0);
- • a text span for open-ended questions (compare semantic meaning, allowing minor wording differences).

- 2. Extract the final answer from the prediction.

- • For multiple-choice questions, locate the final chosen letter (often after “Final Answer:” or “Answer:”) and compare it with the gold letter.
- • For numerical questions, locate the final numeric value, ignoring units and extra text, then compare it to the gold number.
- • For text answers, extract the final answer phrase and compare its semantic meaning with the gold text (e.g., “Yes, the baby is crawling to the right.” matches “Yes”).

- 3. Apply the strict matching rule.

- • Only compare the final extracted answers.
- • Do not use external knowledge to judge whether an answer is reasonable.
- • If the extracted answer and the gold answer match under the appropriate format, output verified=true; otherwise, output verified=false. Input Fields:
- • Question: {question}
- • Gold Answer: {gold answer}

- • Choices (optional, for multiple choice): {choices text}

- • Prediction: {prediction} Output Format (JSON, exact structure):

{

"reasoning": "Step 1: Extract answer from prediction: [extracted_value].

- Step 2: Compare with gold: [gold_value].
- Step 3: Match result: [yes/no].", "verified": true or false

}

Figure 10. The LLM-as-a-judge prompt template used to verify whether a model prediction matches the gold answer, independent of reasoning quality.

