## CutClaw: Agentic Hours-Long Video Editing via Music Synchronization

##### Shifang Zhao1,2, Yihan Hu2, Ying Shan3, Yunchao Wei1†, and Xiaodong Cun2†

- 1 Beijing Jiaotong University
- 2 GVC Lab, Great Bay University 3 ARC Lab, Tencent

# arXiv:2603.29664v1[cs.CV]31Mar2026

https://github.com/GVCLab/CutClaw

[Figure 1]

[Figure 2]

[Figure 3]

###### User Instruction

###### Hours-long Video Footage Music Footage

Showcase the Joker'ssignaturetraits:

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

lip-licking,

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

head tilts, and unsettling smile. Sync his movements

…

…

and laughter to the music rhythm….

0:00 0:30

[Figure 13]

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

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

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

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

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

###### Music Synchronization FollowingInstruction Visually Appealing

0:00

0:30

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

Fig. 1: We present an automated music-driven video editing system that transforms hours-long footage into high-quality short videos based on user instructions and given music rhythm, so that the resulting video demonstrates precise music synchronization, faithful instruction following, and visually appealing aesthetics.

Abstract. Editing the video content with audio alignment forms a digital human-made art in current social media. However, the time-consuming and repetitive nature of manual video editing has long been a challenge for filmmakers and professional content creators alike. In this paper, we introduce CutClaw, an autonomous multi-agent framework designed to edit hours-long raw footage into meaningful short videos that leverages the capabilities of multiple Multimodal Language Models (MLLMs) as an agent system. It produces videos with synchronized music, followed by instructions, and a visually appealing appearance. In detail, our approach begins by employing a hierarchical multimodal decomposition that captures both fine-grained details and global structures across visual and

† Corresponding authors.

audio footage. Then, to ensure narrative consistency, a Playwriter Agent orchestrates the whole storytelling flow and structures the long-term narrative, anchoring visual scenes to musical shifts. Finally, to construct a short edited video, Editor and Reviewer Agents collaboratively optimize the final cut via selecting fine-grained visual content based on rigorous aesthetic and semantic criteria. We conduct detailed experiments to demonstrate that CutClaw significantly outperforms state-of-the-art baselines in generating high-quality, rhythm-aligned videos. The code is available at: https://github.com/GVCLab/CutClaw.

### 1 Introduction

Videos are inherently multimodal, weaving together visual and auditory streams. Consequently, Audio-driven video editing4 represents the most transformative stage of storytelling, fusing sight and sound into organic harmony. Moving beyond simple temporal concatenation, cinematic editing is inherently a complex multimodal alignment problem. In practice, distilling hours of untrimmed video into a concise output requires traversing a massive search space to retrieve sparse, salient segments that simultaneously advance the global storyline and strictly adhere to local auditory dynamics. Balancing the dual constraints of maintaining global narrative coherence and ensuring fine-grained visual-audio harmony renders professional editing a highly labor-intensive process that is heavily dependent on human aesthetic intuition.

Despite recent progress, existing automated video editing frameworks typically neglect the critical role of audio, falling into three suboptimal paradigms. e.g., Template-based methods [1,3,5] force clips into rigid, predefined temporal slots and overlay background music; lacking audio-visual synchronization and semantic awareness, they yield repetitive outputs devoid of narrative progression. Highlight detection methods [26] optimize for local visual salience but are audioagnostic, treating clips in isolation and failing to construct a globally coherent narrative. Text-based approaches [12] prioritize linguistic semantics by aligning visuals with transcripts, yet neglect the underlying musical structure, disrupting both kinetic rhythm and affective energy. Consequently, these methods optimize audio, video, and text instruction independently, struggling to achieve the holistic multimodal alignment required to satisfy the dual constraint of global storytelling and fine-grained visual-audio harmony.

To build a system capable of practical audio-visual storytelling entails three fundamental technical challenges. (i) Context Length Limitation. The dense visual information required for fine-grained understanding across hours-long raw footage physically surpasses the context window length of current MLLMs (Multimodal Language Models) [9,17].(ii) Context-Grounded Storytelling. Crafting a cohesive visual story requires reconciling external user instructions with the intrinsic semantics of the raw video and audio. It is highly challenging to synthesize

4 In this work, “editing” and “cutting” are used interchangeably to denote the temporal selection and assembly of raw video segments.

a narrative logic that strictly executes creative intent without decoupling from the native context and subjects of the source materials. (iii) Fine-Grained CrossModal Alignment. Achieving organic visual-audio harmony demands fine-grained temporal grounding to synchronize musical shifts with a holistic understanding of visual plot, aesthetics, and emotion.

To address these challenges, we introduce CutClaw, an autonomous MLLMpowered multi-agent framework that mimics a professional post-production workflow through a collaborative, coarse-to-fine hierarchy. To overcome the context length limitation, a Bottom-Up Multimodal Footage Deconstruction module abstracts both raw video and audio into structured semantic units of visual scenes and musical sections, enabling both narrative comprehension and fine-grained analysis. To achieve context-grounded storytelling, a Playwriter agent acts as a global planner. Using the musical structure as an invariant temporal anchor, it aligns user instructions with the abstracted scenes to synthesize a narrative that executes creative intent while respecting the source material’s intrinsic plot. Finally, to achieve fine-grained cross-modal alignment, Editor agent and Reviewer agent collaboratively perform top-down hierarchical visual grounding. Guided by the summarized script, the Editor localizes precise segments, and the Reviewer enforces a multi-criteria validity gate to rigorously evaluate plot relevance, visual aesthetics, and instruction following, thereby guaranteeing organic audio-visual harmony.

Our key contributions are summarized as:

- – We tackle the novel task of audio-driven video editing, formally modeling it as a joint optimization problem that simultaneously satisfies instructiondriven storytelling and fine-grained rhythmic harmony.
- – We introduce CutClaw, an MLLM-powered multi-agent framework that tackles the computationally intractable search space of hours-long footage. It integrates bottom-up multimodal deconstruction with a collaborative agentic workflow, where a Playwriter orchestrates music-anchored narrative planning, while Editor and Reviewer agents collaboratively execute precise segment selection.
- – Extensive experiments and user studies demonstrate that CutClaw significantly outperforms state-of-the-art baselines in visual quality, instruction following, and rhythmic harmony.

### 2 Related Work

AI-assisted Video Editing. Video editing has evolved from optimizationbased heuristics to data-driven frameworks. Early pioneering works, such as Write-A-Video [24] and ESA [7], formulated editing as an energy minimization problem to align shots with themed cues. Recent generative methods [8,18] have shifted towards constructing visual sequences driven by high-level instructions or subtitle narratives [12]. However, these methods are fundamentally limited to assembling pre-segmented clips, rely on explicit scripts for narrative structure, and critically neglect the rhythmic guidance of the music modality. In contrast,

CutClaw directly processes raw, untrimmed footage without manual scripts, formulating editing as a hierarchical narrative construction that simultaneously guarantees semantic storytelling and fine-grained audio-visual harmony.

Video Temporal Grounding and Highlight Detection. Video Temporal Grounding(VTG) and Highlight Detection serve as fundamental prerequisites for editing by determining where to cut within raw footage. VTG aims to localize specific segments based on natural language queries; conventional approaches [10,15] rely on pretrained feature encoders, while recent methods [25] leverage MLLMs to enhance instruction understanding. Similarly, Highlight Detection has evolved from using visual saliency scores [23,27,29] to incorporating textual prompts [22, 26] for better alignment with user preferences. However, both streams of research face significant limitations in professional editing contexts: they struggle to effectively model the long-term context of raw footage and lack precise control over the duration of retrieved results. Consequently, these methods are ill-suited for high-precision audio-visual synchronization tasks, where visual cuts must rigorously align with musical beats and rhythmic patterns. To bridge this gap, we take a step to deal with hours-long video footage with both textual and musical input.

Agents for Video Generation and Editing. The advent of MLLMs has catalyzed the use of multi-agent collaborations in the video domain [13, 32]. Recent frameworks employ agents for various settings, ranging from generative role-playing in ViMax [11] to non-linear editing in EditDuet [21] and targeted video trimming [30]. However, these systems face critical bottlenecks in scalability and precision. They are constrained by context windows when processing hours-long footage and fail to achieve audio-visual synchronization due to coarse LLM planning. CutClaw overcomes these limitations by pairing a Hierarchical Decomposition strategy for long-context processing with Audio-Anchor Alignment for precise multi-modal synchronization.

### 3 Method

#### 3.1 Problem Formulation

Given raw video footage, a target music track, and a text instruction as multimodal inputs, we formulate video editing as an agent-driven segment extraction and assembly problem. By leveraging multiple specialized models and agents, our framework extracts and synchronizes relevant clips to ensure the final output strictly follows the narrative instruction while achieving organic audio-visual harmony.

Formally, given raw video footage V, the background music track M, and the user instructions I, the target edited video is recomposed by a trimed timeline E = (c1,...,cN), which consists of a sequence of clips and each clip ci = (tini ,touti ) represents a continuous segment extracted from the original video footage V. We

[Figure 220]

- Fig. 2: The whole workflow of the CutClaw. The multi-modal footage is first Deconstructed, and then, the shot plan is generated by the Playwriter, scene retrieval and editing by the Editor, and quality validation by the Reviewer. optimize a timeline E∗ to maximize a joint objective function:

E∗ = arg max

E

λvQvis(E) + λnQnarr(E)+

λcQcond(E,I) + λsQsync(E,M) ,

(1)

where Qvis (Visual Quality) ensures aesthetic appeal and protagonist prominence; Qnarr (Narrative Flow) enforces coherent storytelling between adjacent clips; Qcond (Semantic Alignment) measures the fidelity of selected content to the instructions I; and Qsync (Rhythmic Alignment) encourages visual cuts to synchronize with musical beats in M. Instead of brute-force searching, we approximate the solution via a hierarchical search space analysis and pruning strategy. As shown in Fig. 2, we first discretize the high-dimensional footage into structured semantic (Sec. 3.2), effectively reducing the solution space. Subsequently, the Playwriter (Sec. 3.3) leverages audio-visual correlations to constrain the search scope to localized candidate pools, enabling the Editor (Sec. 3.4) and Reviewer (Sec. 3.5) to perform efficient fine-grained retrieval and rigorous rejection sampling to finalize the timeline E∗.

#### 3.2 Bottom-Up Multimodal Footage Deconstruction

The raw footage V and background music M are continuous, high-dimensional streams, making direct timeline optimization computationally intractable. To address this, we perform a bottom-up deconstruction to discretize these inputs into structured semantic units, establishing a finite, searchable candidate space for the subsequent hierarchical planning.

Raw Video Footage

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

Showcase the Joker's signature traits ....

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

63 64 65 66

Section-Level Structure

Scenes Caption Instruction

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

(1) Shot Structure Parsing

Shot 0 Shot 5

(1) Structural Scene Allocation

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

Section Proposal

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

Theme:

A haunting character study of the Joker as an agent of pure chaos, capturing his most iconic and unsettling moments.

Subtitles

Shot 4

Shot 1

Emotion:

Unsettling menace blended with the dark fascination of chaos.

Environment:

Environment:

01:40:27,830 -> 01:40:32,010 [Jim Gordon] The doctor says you're in agonizing pain, but you won't accept medication.

Location: Hospital Room

Location: Hospital Bed

Time: Daylight

Time: Daylight

Narrative logic:

The scenes synchronize with the musical arc. Sparse verses match intimate, tense confrontations, while the anthemic chorus drives the escalation toward explosive chaos and the systematic dismantling of order.

Character:

Character:

01:40:27,830 --> 01:40:32,010 [Jim Gordon] That you're refusing to accept skin grafts.

ID 1: Patient in Bed

ID 1: Person in Foreground

Related Scenes:

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

Cinematography:

Cinematography:

...

Shot: Medium Shot, Static

Shot: Close-up, Static

01:40:38,150 --> 01:40:42,430 [Harvey Dent] Remember that name you all had for me when I was at Internal Affairs?

Composition: Centered composition with the character, Eye-level

Composition: Over-the-shoulder, Eye-level

63 64

Time Duration: 23s

Narrative:

Narrative:

Action: Lying

Action: Speaking

- 01:40:52,570 [Harvey Dent] Say it. Say it!
- 01:41:05,130 --> 01:41:07,310 [Jim Gordon] Two-Face, Harvey Two-Face.

Emotion: Serious, Somber

Emotion: Somber, Intense

[Figure 320]

[Figure 321]

[Figure 322]

Context: To establish the presence and serious demeanor of a key character in a formal setting.

Context: To capture the character's emotional response and internal conflict during a serious conversation.

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

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

Keypoint-Level Music Structure

Shot-level Caption

(2) Semantic Scene Aggregation

(2) Keypoint-Aligned Shot Planning

[Figure 349]

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

Shot Plan

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

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

Content:

Disguised as a nurse, the Joker stands before the burning hospital, taunts authorities, and activates a bomb. The scene captures the explosive destruction as he revels in the chaos.

Scene-level Caption

Emotion:

Scene 0

Triumphant, building intensity with gospel power.

Visual Score: ⭐ 5/5 (Class: Content, Usable) Character: Narrative:

Harvey Dent (Subject), Jim Gordon (Interactor).

Production-Orientedrodrod tiontion rientedriented

Time Duration: 2.9s

Story Beat: Theatrical destruction of Gotham General

Harvey Dent lies in a hospital bed, his face bandaged and his body in agony, refusing skin grafts and medication. Jim Gordon visits him, expressing sorrow for Rachel's death and confronting Dent about his transformation ....

Hospital.

Identity-Awareee tt tt ee

Related Scene:

64

Visual:

Wide shot of burning hospital with Joker; medium shot of detonator activation; close-up of smiling face lit by flames.

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

Narrative-Coherentarratarrat ee ereere tt

Emotion: Cinematography:

Somber (Start), Tense (Dialogue), Tragic (Climax)

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

Close-up of face to Reveal of disfigurement .

- Fig. 3: Left: Video Shots Aggregation. We first perform shot detection on the entire video and conduct a caption for a detailed understanding. Then, we use an LLM to aggregate similar content for a scene-level description. Right: The workflow of Playwriter. Playwriter generate the whole storyline according to the input, and gives the detailed shot plan of each specific shot of music.

Video Shots Aggregation: From Shot to Scene Effective editing requires both fine-grained and coarse-grained narrative comprehension. To reconcile these granularity requirements within the context window limits of MLLMs [2], we propose a hierarchical aggregation strategy (Fig. 3 Left). Specifically, we discretize the footage V into atomic shots S defined as fundamental visual units bounded by camera cuts, which are subsequently aggregated into scenes Z forming contiguous, spatio-temporally coherent shot sequences.

Shot Parsing and Scene Aggregation. To instantiate this hierarchy, we first obtain the atomic shots S using boundary detection [6]. For each shot si, we extract semantic attributes A(si) covering cinematography, character dynamics, and environment via an MLLM [2]. To group these individual shots into the defined scenes, we compute a transition similarity Sim(si,si+1) = α⊤ vi,i+1 between adjacent shots. Here, vi,i+1 denotes the attribute-wise similarity vector derived from the LLM [20] features, and α represents the weight vector balancing the importance of different attributes. A scene boundary is induced whenever this similarity score drops below a predefined threshold τ, effectively partitioning the continuous footage into discrete, meaningful narrative blocks.

Character-Aware Grounding. To ensure narrative consistency involving recurring protagonists, we implement a identity injection. We first analyze the dialogue to infer character identities H (names and roles). These identities are injected as textual conditioning into the MLLM [28] during scene analysis. This grounds the generated descriptive summary D(zj) in specific personas (e.g., replacing “a man” with “Joker”), facilitating reliable cross-scene character tracking.

Structural Audio Parsing To maximize Rhythmic Alignment (Qsync), we convert the continuous music waveform M into a discrete grid of potential

cut points. We employ a hierarchical strategy that bridges micro-level rhythm (beats) with macro-level musical form (sections), providing the Playwriter with rigid temporal anchors.

Hierarchical Keypoint Detection. We first extract perceptually salient Sound Keypoints K on a discrete time axis T [4]. We identify three types of candidates: (i) Downbeats Kdb (bar-level accents); (ii) Pitch Changes Kpc (melodic transitions); and (iii) Spectral Energy Changes Kse (timbral transitions). We form a unified candidate pool K0 = Kdb ∪ Kpc ∪ Kse and apply temporal filtering Φ(·) (e.g., peak de-duplication) to obtain robust boundaries K = Φ(K0).

Structure-Guided Refinement. To organize these keypoints, we use an MLLM [28] to partition the track into coarse structural units U = {uj}Mj=1 (e.g., verse, chorus). Within each unit uj, we score the contained keypoints t ∈ K ∩ uj to retain only the most significant boundaries. The significance score is computed as a weighted sum of cue intensities:

score(t) = β⊤ i(t), where i(t) = [intdb(t),intpc(t),intse(t)]⊤. (2)

where int∗(t) denotes the intensity of each respective type at time t, and β is the weight vector. Finally, we generate structure-aligned captions, describing local rhythm, emotion, and energy to guide the visual matching.

#### 3.3 Playwriter: Music-Anchored Script Synthesis

Given the decomposed semantic scenes Z and structural audio units U, Playwriter [9] utilizes the musical structure U as the invariant temporal anchor for storytelling(Fig. 3 Right). By strictly grounding the visual narrative progression onto this auditory skeleton, the Playwriter enforces Rhythmic Alignment (Qsync) while optimizing for Instruction Fidelity (Qcond) and Narrative Flow (Qnarr). It utilizes structural scene allocation and keypoint-aligned shot planning to map the video scenes Z onto the musical structure U, generating a shot plan subject to strictly formalized execution rules to guarantee validity:

- 1. Disjoint Resource Allocation (Non-Overlap): To prevent temporal redun-

dancy, the Playwriter strictly partitions the scene Z. Let Zuj ⊂ Z denote the subset of scenes allocated to the j-th musical unit. For any two distinct

units uj,uk ∈ U, we enforce: Zuj ∩ Zuk

= ∅. This exclusive assignment ensures that no source material is reused across different narrative blocks, logically satisfying the global non-overlap constraint by construction.

- 2. Structural Temporal Anchoring (Music Duration): To enforce the duration

constraint, the generated shot plan Pj for each unit uj inherits the fixed temporal topology of the audio. The total planned duration is strictly anchored

Duration(p) ≡ |uj|. Below, we give the details workflow.

to the audio interval length: p∈P

j

Structural Scene Allocation The first stage constructs a global mapping between musical structural units and visual scenes. Let U = {uj}Mj=1 denote the set of musical units derived in Sec. 3.2. The agent generates a structure proposal P that assigns a subset of candidate scenes Zuj ⊂ Z to each unit uj. The allocation is formulated as a conditional generation task:

= Φmacro(uj,I | Z), (3)

Zuj

where Φmacro represents the LLM-based [9] planning function conditioned on the user instruction I. To satisfy the hard temporal constraints, we enforce a strict disjoint set requirement:

= ∅, ∀j ̸= k. (4)

Zuj ∩ Zuk

If the generated proposal violates this condition (i.e., a scene is reused across different musical sections), the system rejects P and triggers a regeneration with negative constraints.

Keypoint-Aligned Shot Planning The second stage refines the allocation into a sequence of executable specifications. For each unit uj, let {k1,...,kL} be the set of fine-grained musical segments contained within its temporal scope. The agent generates a shot plan consisting of specifications {p1,...,pL}.

Critically, rather than outputting final timestamps, each specification pi = (τi,zid,di) serves as a retrieval constraint for the subsequent editing phase:

τi: The target duration constraint derived directly from the audio segment ki,

ensuring rhythmic synchronization (Qsync). zid: The source scene index selected from Zuj

, which restricts the retrieval search space to the allocated narrative block.

di: A semantic visual description (e.g., specific plot or emotion) that guides the content matching within scene zid.

This hierarchical binding transforms the global optimization problem into a series of local retrieval tasks. By explicitly binding the i-th shot to a specific scene zid, the Playwriter effectively prunes the search space for the downstream Editor, ensuring that the final clip selection is conducted successfully.

#### 3.4 Editor: Top-Down Hierarchical Visual Grounding

Operating within the structural shot plan constrained by the Playwriter, the Editor performs fine-grained temporal grounding to determine the precise continuous coordinates of the final timeline E∗. We instantiate the Editor as a ReAct [31] agent designed to iteratively maximize the local energy terms of the joint objective function (Eq. 1), specifically targeting Visual Quality (Qvis).

As shown in Fig. 4, for each retrieval specification pi = (τi,zid,di) generated by the Playwriter, the Editor navigates the candidate pool through a hypothesisand-verification loop. Its goal is to identify a specific clip ci = [tin,tout] such that the duration constraint |ci| ≈ τi is met, while maximizing local utility. The Editor has 3 main actions:

Action 1 SNR Action 2 FGST

|[Figure 408]<br><br>[Figure 409]<br><br>[Figure 410]<br><br>[Figure 411]<br><br>[Figure 412]<br><br>01:52:45 -<br><br>Captions<br><br>Shot 2|Draft<br><br>Temporal<br><br>[Figure 413]<br><br>[Figure 414]<br><br>[Figure 415]<br><br>[Figure 416]<br><br>[Figure 417]<br><br>[Figure 418]<br><br>|
|---|---|
| |01:52:07 -<br><br>Event<br><br>|

01:51:58 - 01:52:06 01:52:07 - 01:52:45

Zoom in

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

###### Shot 0 Shot 1

Scene 64

|[Figure 428]<br><br>[Figure 429]<br><br>[Figure 430]<br><br>[Figure 431]<br><br>[Figure 432]<br><br>[Figure 433]<br><br>[Figure 434]<br><br>[Figure 435]<br><br>[Figure 436]<br><br>[Figure 437]<br><br>[Figure 438]<br><br>[Figure 439]<br><br>[Figure 440]<br><br>[Figure 441]| |
|---|---|
|Externa|[Figure 442]<br><br>l MLLM|

[Figure 443]

[Figure 444]

Retrieval Corresponding Shot

Shot

Environment

01:52:45

Cinematography Character Narrative

Cinema tography

Character Emotion

Action 3 Commit

[Figure 445]

find others

find others

|[Figure 446]<br><br>[Figure 447]<br><br>[Figure 448]<br><br>[Figure 449]<br><br>[Figure 450]<br><br>[Figure 451]<br><br>[Figure 452]<br><br>[Figure 453]<br><br>[Figure 454]<br><br>[Figure 455]<br><br>[Figure 456]<br><br>[Figure 457]<br><br>[Figure 458]<br><br>[Figure 459]<br><br>|
|---|

if failed

Review Check Length Shot Length Satisfied

Review Check Aesthetic Score CinematographyEmotion

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

Review Check Protagonist Ratio

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

- Action 1: Semantic Neighborhood Retrieval. This action initializes the local search space Ωi by retrieving all shots belonging to the assigned scene zid. To address potential content scarcity or segmentation noise in the visual candidate, we incorporate an Adaptive Ex-

pansion mechanism. If the primary search space Ωi = {s | s ∈ zid} fails to yield a high-confidence candidate, the Editor expands the scope to the semantic neighborhood:

Ωi′ = Ωi ∪ {s | s ∈ Neighbor(zid,∆)}. (5)

This fallback strategy prevents retrieval dead-ends by aggregating shots from adjacent structural units, ensuring the agent maintains a sufficient material pool for optimization.

- Action 2: Fine-Grained Shot Trimming. To maximize the objective terms Qvis and Qsync, the Editor employs a VLM-driven analysis tool to perform dense temporal grounding within the candidate shots. For a candidate shot s ∈ Ωi, the agent seeks a sub-segment ci ⊂ s that maximizes a weighted local score:

c∗i = arg max c⊂s,|c|=τi

(α · Saes(c) + β · Rprot(c | H)). (6)

Here, Saes represents the aesthetic score (contributing to Qvis), and Rprot denotes the Protagonist Presence Ratio (contributing to Qcond), where α and β are the respective balancing weights. The presence ratio is computed by cross-referencing frame content with the character identity set H established in Sec. 3.2. If the current segment yields a suboptimal score, the agent heuristically shifts the temporal window based on VLM feedback until a high-fidelity clip is secured.

- Action 3: Commit. The Editor submits the trimmed candidate ci to the Reviewer (Sec. 3.5). Upon receiving an approval signal, the clip is rendered and committed to the final timeline E∗. Otherwise, the Editor triggers a backtracking mechanism to explore alternative intervals within Ωi.

Fig. 4: Editor and Reviewer are used to perform segment selection and validation. SNR stands for Semantic Neighborhood Retrieval, and FGST stands for Fine-Grained Shot Trimming.

#### 3.5 Reviewer: Multi-Criteria Validity Gate

To ensure the final timeline E∗ adheres to both narrative intent and structural constraints, we introduce the Reviewer to operate as a discriminatory gate. As shown in Fig. 4, this module audits every candidate clip ci proposed by the Editor through a rigorous rejection sampling mechanism. The reviewer checks the consistency of the edited video from the following aspects:

Semantic Identity Verification. To enforce narrative consistency (Qcond), the Reviewer validates that the visual subject strictly aligns with the target identity defined in H. By computing a Protagonist Presence Ratio via hierarchical MLLM [2] sampling, we filter out false positives where the character is merely a background extra, occluded, or unrecognizable. This ensures that the protagonist remains the primary visual focus throughout the sequence, distinguishing the main characters from crowd elements.

Temporal and Structural Integrity. To maintain the topological validity of the timeline, we enforce hard constraints on sequencing. The Reviewer verifies Non-Overlap (∩Eprev = ∅) to prevent content duplication and checks Duration Fidelity to ensure the visual cut points align precisely with the rhythmic grid of the music track M. Any violation of these constraints triggers an immediate rejection to preserve the global structure.

Perceptual Quality Assurance. To maximize aesthetic appeal (Qvis), the module audits low-level visual saliency. It rejects shots exhibiting significant quality degradation, ensuring that every committed segment meets broadcastlevel viewing standards. Upon detection of any violation, the Reviewer returns structured feedback, prompting the Editor to backtrack and explore alternative intervals within the semantic neighborhood.

If the candidate clip does not meet the requirements, the Reviewer will notify the editor to select the relevant scene around the current one. By reviewing each candidate clip in the optimized timeline, we obtain the final edited video.

4 Experiment

#### 4.1 Evaluation and Implementation

Benchmark. To rigorously evaluate our framework, we establish a diverse benchmark specifically designed for agentic video editing tasks. Our dataset comprises 10 distinct source pairs, collected from 5 feature-length films and 5 long-duration VLOGs, with raw footage lengths ranging from 1 to 3 hours. This collection accumulates to approximately 24 hours of total footage, ensuring a robust assessment across both professionally cinematographed content and unscripted, naturalistic recordings. The corresponding auditory inputs consist of 10 segmented music tracks spanning a wide spectrum of genres, including Pop, Jazz, OST, Rock, and R&B with target edit durations varying from 20 seconds to one minute.

To test the system’s semantic adaptability, we formulate two distinct instruction categories: (1)Character-Centric Instructions, which constrain the edit to focus exclusively on a single protagonist, thereby challenging the agent’s ability

to maintain identity consistency; and (2)Narrative-Centric Instructions, which demand the inclusion of multiple characters or complex interactions to convey a cohesive visual story. In total, this benchmark yields 20 unique evaluation cases (10 pairs × 2 instruction types), covering a broad range of visual styles and narrative requirements.

Metrics. We evaluate our framework via automated quantitative analysis and a subjective user study. In the automated regime, Visual Quality and Instruction Follow are scored by GPT-5.2 [16] based on aesthetic integrity and semantic alignment, respectively. Conversely, given the high temporal precision required for audio-visual alignment, which remains challenging for MLLMs, AV Harmony is quantified via detecting the minimum temporal offset ∆t between audio onsets (downbeats, pitch) and video scenes, strictly rewarding alignments within a perceptual threshold (e.g., ∆t ≤ 0.1s). The user study mirrors these three metrics to capture human perceptual preference and exclusively evaluates a fourth dimension, Human-Likeness, which benchmarks the naturalness of the model’s editing pacing and logic against professional human editors.

Baselines. We benchmark our framework against three representative methods covering different editing paradigms. NarratoAI [12] serves as a baseline for subtitle-driven editing; it is a mainstream open-source framework that processes full-video subtitles to generate clips based on textual instructions. Note that NarratoAI is inapplicable to VLOG scenarios due to the scarcity of speech and subtitles in such footage. UVCOM [26] and Time-R1 [25] represent state-of-theart approaches in moment retrieval and temporal grounding, respectively. Since both models typically handle fixed-length short videos, we adapt them for longform footage by first segmenting the source video, then selecting the top-5 clips with the highest confidence. Finally, we trim the selected segments to match the target duration, discarding excess frames.

Implementation Details. For the core agentic framework, we employ MiniMax-

M2.1 [14] to power the Editor and Reviewer agents, while Gemini3-Pro [9] serves as the Playwriter. In the preprocessing stage, we utilize PySceneDetect [6] for shot boundary detection and Whisper-v3-turbo [19] for Automatic Speech Recognition (ASR) to extract subtitles. For multimodal understanding, Qwen3VL-30B-A3B [2] and Qwen3-Omni-30B-A3B [28] are deployed for visual and music captioning, respectively. To optimize computational efficiency during inference, video footage is downsampled to a short-side resolution of 360p at a frame rate of 2 FPS.

#### 4.2 Main Results

As presented in Tab. 1, CutClaw achieves superior performance across all quantitative metrics. Validating our coarse-to-fine editing strategy, our method consistently surpasses the strongest baselines in Visual Quality. Furthermore, CutClaw leads the Instruction Follow metric, particularly excelling in object-oriented instruction, which demonstrates its precise visual content localization capabilities. Finally, its dominant performance in AV Harmony confirms that the resulting cuts are rigorously and rhythmically aligned with the input music. Fig. 6 shows

Instruction Input

“A montage of Rose's evolution on the Titanic: from trapped aristocrat to free spirit. Key shots: despair at the ship's rail, joyous dancing in 3rd class, defiant spitting, and survival in freezing water. Focus on her red hair, ocean backdrop, and the transition from tight corsets to loose clothing.”

Input

[Figure 468]

[Figure 469]

Music

Input VIdeo Footage

Iter 0 Get Shot Plan

Shot IDX: 1 Time Duration: 5.2s Content: This is the discovery of the "Ghost" or Gravitational Anomaly. Cooper is in his daughter Murph's bedroom, observing how the dust floating in the air is settling in a specific, non-random pattern on the floor due to a manipulation of gravity. Emotion: Awe, disbelief and suspenseful. Visuals: Transitions from a medium shot to a close-up with soft, warm lighting. The scene ends with a close-up shot of Cooper’s superising face, moving at a slow pace. Related Scene: 15

The iteration initiates with the target scene defined by the Shot Plan, while potentially extending the search to adjacent scenes. The subsequent search is initialized based on the most promising shot identified in this process.

Scene15

[Figure 470]

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

(Target_Scene=15)

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

- Iter 1 SNR

- Iter 2 FGST

Iter 4 Commit (Draft_Shot=[00:19:54 - 00:20:00])

[Figure 492]

Iter 7 Check Protagonist Ratio

(Draft_Shot=[00:19:54 - 00:20:00])

Iter 5 Check Length (Draft_Shot=[00:19:54 - 00:20:00])

[Figure 493]

Result list of shots returned. Promising candidates found:

- - [00:19:18 - 00:19:26]: Kitchen, man at refrigerator (potential match for drawing).
- - [00:19:46 - 00:20:03]: A candle flickers in a dark, dusty room, low light that illuminates floating dust particles.

Result Analysis show:

- - [00:19:18 – 00:19:20] (2.0s):

- - Visual Composition: [Hard Cut] Close-up. Dark kitchen setting with minimal lighting through blinds. Slight motion blur observed.
- - Semantic Analysis: High tension (Intensity: Medium). The shot functions primarily to establish emotional distress.
- - Quality Assessment: Score 3 (Consistent low lighting).

- - [00:19:20 – 00:19:23] (3.0s):

- - Visual Composition: [Hard Cut] Medium shot containing two subjects (older man, young girl). Environment features hanging lamp and appliances.
- - Semantic Analysis: Melancholic mood. The narrative focus shifts to revealing character emotion through body language.
- - Quality Assessment: Score 3 (Consistent low lighting).

- - [00:19:23 – 00:19:26] (3.0s):
- - Visual Composition: [Hard Cut] Side profile tracking shot. Man moves toward a window source. Higher contrast with warm glow.
- - Semantic Analysis: Mysterious mood. The scene utilizes lighting changes to build suspense.
- - Quality Assessment: Score 4 (Sharp focus, good composition). Recommended for action sequences.

- Iter 3 FGST

00:19:12 00:20:22

(Time_Range=[00:19:18 - 00:19:26])

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

The Editor finds that this time range lacks the desired narrative, so it shifts to another.

[Figure 499]

(Time_Range=[00:19:46 - 00:20:07])

[Figure 500]

Result Analysis show:

[Figure 501]

- - [00:19:46 – 00:19:54] (8.0s):

- - Visual Composition: [Hard Cut] Close-up. A candle flickers in a dark, dusty room, low light that illuminates floating dust particles.
- - Semantic Analysis: High tension (Intensity: Medium). Eerie, Isolated, Mysterious.
- - Quality Assessment: Score 3 (Consistent low lighting).

- - [00:19:54 – 00:20:04] (10.0s):

- - Visual Composition: [Hard Cut] The man is in profile, facing left, and the woman is partially visible on the right.
- - Semantic Analysis: Warm lighting, stable shot, slight motion blur
- - Quality Assessment: Score 4

- - [00:20:04 – 00:20:07] (3.0s):

The Reviewer successfully captured the protagonist's absence. Although the protagonist appears in the draft shot, the poor cinematic quality (persistent side profile and bad lighting) caused it to fail.

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

- Visual Composition: [Hard Cut] Close-up of a woman holding a pillow in a dark room. She is looking slightly to the side, and the lighting is dim with a warm tone.

- - Semantic Analysis: Mysterious mood. The scene utilizes lighting changes to build suspense.
- - Quality Assessment: Score 3 (Low lighting, slight motion blur, acceptable composition.) Result Analysis show:
- - [00:20:07 – 00:20:10] (3.0s):

- - Visual Composition: [Hard Cut] Medium shot of a man, looking at a glowing object in a dimly lit, dusty room with bookshelves in the background
- - Semantic Analysis: Tense, builds suspense and mystery
- - Quality Assessment: Score 4 (Warm lighting, shallow depth of field, stable shot, slight grain)

- - [00:20:10 – 00:20:12] (2.0s):

- - Visual Composition: [Hard Cut] Close-up of a man's face, looking down with a concerned expression in a dimly lit room.
- - Semantic Analysis: Reveals character emotion and internal conflict.
- - Quality Assessment: Score 4 (Sharp focus on the face, warm lighting, stable shot).

- - [00:20:12 – 00:20:17] (5.0s):
- - Visual Composition: [Hard Cut] Close-up of a dusty floor with a glowing object in the background, out of focus.
- - Semantic Analysis: Maintains tension and focus on the environment.
- - Quality Assessment: Score 4 (Shallow depth of field, warm lighting, stable shot).

00:19:12 00:20:00

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

✅

Iter 6 Check Aesthetic Score

❌

- (Draft_Shot=[00:19:54 - 00:20:00]) ✅

Iter 11 Check Aesthetic Score

- (Draft_Shot=[00:20:07 - 00:20:12])✅

- Iter 8 FGST (Time_Range=[00:20:07 - 00:20:17])

- Iter 9 Commit (Draft_Shot=[00:20:07 - 00:20:12])

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

The Editor continues attempting to search neighboring time ranges. While the narrative is satisfactory, better cinematic visual quality is needed.

[Figure 518]

[Figure 519]

00:20:07 00:20:12

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

Iter 10 Check Length (Draft_Shot=[00:20:07 - 00:20:12])

✅

(Draft_Shot=[00:20:07 - 00:20:12])

✅

Iter 12 Check Protagonist Ratio

- Fig. 5: A sample execution of a single-shot cutting, utilizing footage from movie “Interstellar” and the music “Moon.” Actions performed by the Playwright, Editor, and Reviewer are color-coded in blue, yellow, and green, respectively. The

orange background traces the execution path leading to the final clip selection.

the qualitative comparison against baseline methods. When considering the overall edited quality, baseline methods exhibit rigid segment selection, completely failing to align with the musical structure. While NarratoAI loosely follows user instructions at the cost of severe visual degradation, UVCOM and Time-R1 maintain visual quality but lack logical narrative connections across shots. Additional video results are available in the supplementary material. To further

Object-centric Instruction:

MusicInstruction InputInput

"A vibrant, rhythmic showcase of the seamless transitions between the stoic Dr. Atsuko Chiba and her effervescent alter-ego Paprika, utilizing the track’s mellow jazz-hop beat to synchronize the kaleidoscope of colorful transformations, shattering glass, and gravitydefying flights across the dreamscape."

Input

Soothing Nostalgic Playful Joyful Intense Melancholic Urgent Hopeful Tender Peaceful

[Figure 529]

| |🎵 Temporal Synchronization: Precisely aligns visual cuts with the musical rhythm.<br><br>🎭 Affective Consistency: Visual content effectively mirrors the melody's emotional tenor.<br><br>🎯 Semantic Fidelity: Generates accurate object-centric content guided by user instructions.|
|---|---|
| | |

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

CutClawTime-R1UVCOMNarratoAI

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

[Figure 554]

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

| |🔇 Temporal Misalignment: Lacks auditory perception, failing to synchronize visual cuts with the musical beat.<br><br>🧩 Semantic Discrepancy: Struggles to ground object-centric instructions, resulting in low fidelity to the target entities.|
|---|---|
| | |

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

Narrative-driven Instruction:

MusicInstruction Input

“A deconstruction of the Hollywood Ending, using the song’s cynical yet tender lyrics to juxtapose the idealized ‘Epilogue’ sequence with the gritty reality of the couple’s arguments and separation, illustrating that while they were each other's muse, real life—like the song suggests—is far more complicated than a movie script.”

| |🧠 Abstract Semantic Following: Interprets high-level descriptive instructions to grasp complex narrative themes.<br><br>🔍 Precise Event Retrieval: Accurately localizes clips depicting specific interaction dynamics (e.g., arguments and separation).<br><br>🎼 Affective Sequencing: Orchestrates the editing structure to align with the soundtrack.|
|---|---|
| | |

Tender Dreamy Resigned Explosive Wistful

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

CutClawUVCOMNarratoAI

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

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

|🗑 Semantic Noise: Includes extraneous clips that are weakly related to the core narrative instructions.<br><br>🔭 Narrow Event Scope: Disproportionately focuses on conflict scenes, neglecting broader context required by the user.<br><br>🚫 Logical Disjointedness: Fails to synthesize the retrieved clips into a coherent, meaningful sequence.|
|---|

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

[Figure 756]

[Figure 757]

[Figure 758]

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

[Figure 766]

[Figure 767]

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

[Figure 774]

| |✅ High-Quality Extraction: Maintains the completeness of individual video segments.<br><br>❌ Semantic Inability: Completely fails to ground complex instructions, resulting in the retrieval of irrelevant clips.|
|---|---|
| | |

[Figure 775]

[Figure 776]

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

[Figure 782]

[Figure 783]

[Figure 784]

Time-R1

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

[Figure 789]

[Figure 790]

[Figure 791]

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

[Figure 796]

[Figure 797]

[Figure 798]

[Figure 799]

[Figure 800]

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

- Fig. 6: Qualitative comparison between CutClaw and baseline methods. The two cases utilize full-length footage from the films “Paprika” and “La La Land”, paired with the musical tracks “Luv(sic) Pt.2” and “Norman F**king Rockwell”, respectively. Shot boundary detection is performed using PySceneDetect [6].

illustrate our method, we provide an execution sample in Fig. 5, detailing the collaborative workflow among the Playwriter, Editor, and Reviewer agents.

Ablation Study To validate the effectiveness of individual components within CutClaw, we conducted an ablation study by systematically removing the Editor,

Table 1: Comparision. We report the performance scores across three metrics: Visual Quality, Instruction Follow (Obj/Nar), and AV Harmony.

Visual Quality Instruction Follow AV Harmony Film Vlog Avg. Obj Nar Avg. Film Vlog Avg.

Method

NarratoAI 75.7 - 75.7 56.0 72.0 64.0 84.9 - 84.9 UVCOM 71.2 73.6 72.4 60.8 64.5 62.6 78.9 79.7 79.3 Time-R1 73.3 72.6 72.9 51.9 71.0 61.5 77.0 75.8 76.4

CutClaw 79.2 76.0 77.6 66.6 73.4 70.0 85.7 87.3 86.5

Table 2: Ablation Study. We report the performance impact of ablating the Editor, Reviewer, or Audio Context across three metrics.

Visual Quality Instruction Follow AV Harmony Film Vlog Avg. Obj Nar Avg. Film Vlog Avg.

Method

w/o Audio 77.3 73.8 75.5 63.4 74.3 68.9 78.0 76.5 77.2 w/o Editor 78.6 75.4 77.0 59.7 71.5 65.6 84.8 86.0 85.4 w/o Reviewer 78.1 74.0 76.0 66.6 72.9 69.8 85.1 89.4 87.2

###### CutClaw 79.2 76.0 77.6 66.6 73.4 70.0 85.7 87.3 86.5

Table 3: User Study Results. We report the percentage of user votes across four metrics: Visual Quality, Instruction Following, Audio-Visual Harmony, and HumanLikeness. For each metric, we break down performance by instruction type (Narrative/Object) and video type (Film/Vlog). Avg. denotes the average score. Our method outperforms baselines across all categories. * Note that NarratoAI cannot deal with the VLOG as it does not have dense subtitles.

Visual Quality Instruction Follow Audio-Visual Harmony Human-Like

Method

Nar Obj Film Vlog Avg Nar Obj Film Vlog Avg Nar Obj Film Vlog Avg Nar Obj Film Vlog Avg

NarratoAI* 11.6% 11.2% 27.3% - 11.4% 14.8% 10.8% 28.7% - 12.8% 11.2% 12.4% 25.3% - 11.8% 12.0% 8.4% 23.3% - 10.2% UVCOM 18.0% 16.8% 7.3% 23.6% 17.4% 18.8% 13.2% 7.3% 22.0% 16.0% 17.2% 13.2% 6.0% 22.8% 15.2% 18.8% 15.6% 9.3% 23.6% 17.2% Time-R1 25.2% 17.6% 18.0% 22.4% 21.4% 22.4% 19.6% 16.7% 24.0% 21.0% 23.2% 16.8% 17.3% 18.4% 20.0% 24.8% 22.8% 16.7% 25.6% 23.8%

CutClaw 45.2% 54.4% 47.3% 54.0% 49.8% 44.0% 56.4% 47.3% 53.6% 50.2% 48.4% 57.6% 51.3% 57.6% 53.0% 44.4% 53.2% 50.7% 50.4% 48.8%

Reviewer, and Audio Context, with results detailed in Tab. 2. We first observe that replacing the Audio’s beat-aware analysis with fixed-length segmentation causes AV Harmony to drop significantly from 86.5 to 77.2, confirming its necessity for rhythmic alignment. Similarly, removing the Reviewer leads to a decline in Visual Quality from 77.6 to 76.0, as the system loses the feedback loop required to refine low-quality candidates and transition mismatches. Finally, substituting the Editor with a random clip selector degrades performance across both Visual Quality and Instruction Following, reducing the average score from 70.0 to 65.6. This demonstrates that the Editor’s hierarchical structuring is foundational for preserving narrative coherence and semantic accuracy.

User Study To complement the objective metrics, we conducted a user preference study to assess the subjective quality of the generated videos. We recruit 25 participants to evaluate the results. The questionnaire consists of 80 evaluation items, asking participants to vote for the best method across four dimensions: Visual Quality, Instruction Follow, Audio-Visual Harmony, and Human-Likeness. In total, we collected 2,000 user opinions, providing a statistically robust basis for our analysis. As illustrated in Tab. 3, CutClaw outperforms all baselines by a significant margin across all categories. Specifically, our method receive 49.8% of the votes for Visual Quality and 53.0% for Audio-Visual Harmony, which is more than double the votes received by the second-best method, Time-R1 (21.4% and 20.0%, respectively). Notably, in the Human-Like metric, CutClaw secured 48.8% of user preference, highlighting its ability to mimic professional human editing logic better than existing automated solutions. These results align consistently with our quantitative findings, confirming the superiority of our approach in real-world viewer evaluations.

#### 4.3 Limitation

Our framework still faces limitations. First, while we ensure strong narrative flow, the system lacks advanced visual hooks, such as generated visual effects or specific monologue highlights that are crucial for engaging content. Future iterations could integrate generative video models to synthesize these expressive elements. Second, the multi-stage pipeline processing extensive raw footage results in high inference latency. Optimizing the pipeline for speed or employing

coarse-to-fine processing strategies to enable real-time feedback remains a key direction for future research.

### 5 Conclusion

We presented CutClaw, an autonomous multi-agent framework designed to automate the complex task of professional video editing from hours-long raw footage. By addressing the critical challenges of processing long contexts and achieving precise audio-visual consistency, our approach bridges the gap between simple clip assembly and instruction-aligned, music-driven storytelling. The core innovation of our framework lies in its hierarchical decomposition strategy, which transforms continuous high-dimensional data into structured semantic units. This structure allows our specialized agents to collaborate effectively: the Playwriter anchors the narrative to musical structure, the Editor performs fine-grained visual grounding, and the Reviewer enforces rigorous aesthetic and continuity constraints. Our extensive experimental results demonstrate that CutClaw significantly outperforms state-of-the-art baselines across key metrics, including Visual Quality, Instruction Following, and AV Harmony.

### Acknowledgements

This work was financially supported in part by the National Natural Science Foundation of China (Project No. 62506064) and Guangdong Provincial Regional Joint Fund (Project No. 2024A1515110052). The computational resources are supported by SongShan Lake HPC Center (SSL-HPC) in Great Bay University.

### References

- 1. Adobe: Adobe premiere pro (2026), https://www.adobe.com/products/ premiere.html
- 2. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631

(2025)

- 3. Blackmagic Design: Davinci resolve 20 (2026), https://www.blackmagicdesign. com/products/davinciresolve
- 4. Böck, S., Korzeniowski, F., Schlüter, J., Krebs, F., Widmer, G.: madmom: a new Python Audio and Music Signal Processing Library. In: Proceedings of the 24th ACM International Conference on Multimedia. pp. 1174–1178. Amsterdam, The Netherlands (10 2016). https://doi.org/10.1145/2964284.2973795
- 5. ByteDance: Capcut (2026), https://www.capcut.com/
- 6. Castellano, B., collaborators: Pyscenedetect (2025), https://github.com/ Breakthrough/PySceneDetect
- 7. Chen, Y., Wang, W., Zheng, T., Wen, X., Yang, H., Zhang, Y.: Esa: Energybased shot assembly optimization for automatic video editing. arXiv preprint arXiv:2511.02505 (2025)

- 8. Cheng, D., Zhan, H., Zhao, X., Liu, G., Li, Z., Xie, J., Song, Z., Feng, W., Peng, B.: Text-to-edit: Controllable end-to-end video ad creation via multimodal llms. arXiv preprint arXiv:2501.05884 (2025)
- 9. Google: Gemini3 (2025), https://deepmind.google/models/gemini/
- 10. Guo, Y., Liu, J., Li, M., Liu, Q., Chen, X., Tang, X.: Trace: Temporal grounding video llm via causal event modeling. arXiv preprint arXiv:2410.05643 (2024)
- 11. HKUDS: Vimax: Agentic video generation (2025), https://github.com/HKUDS/ ViMax
- 12. linyqh: Narratoai (2025), https://github.com/linyqh/NarratoAI
- 13. Liu, C., Wu, H., Zhong, Y., Zhang, X., Wang, Y., Xie, W.: Intelligent grimmopen-ended visual storytelling via latent diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6190– 6200 (2024)
- 14. MiniMax AI: Minimax-m2.1: a sota model for real-world dev & agents. (2025), https://github.com/MiniMax-AI/MiniMax-M2.1
- 15. Mu, F., Mo, S., Li, Y.: Snag: Scalable and accurate video grounding. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 18930–18940 (2024)
- 16. OpenAI: Gpt-5.2 (2025), https://openai.com/index/introducing-gpt-5-2/
- 17. OpenAI: Chatgpt (2026), https://chatgpt.com/
- 18. Pardo, A., Wang, J.H., Ghanem, B., Sivic, J., Russell, B., Heilbron, F.C.: Generative timelines for instructed visual assembly. arXiv preprint arXiv:2411.12293

(2024)

- 19. Radford, A., Kim, J.W., Xu, T., Brockman, G., McLeavey, C., Sutskever, I.: Robust speech recognition via large-scale weak supervision. In: International conference on machine learning. pp. 28492–28518. PMLR (2023)
- 20. Reimers, N., Gurevych, I.: Making monolingual sentence embeddings multilingual using knowledge distillation. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics (11 2020), https://arxiv.org/abs/2004.09813
- 21. Sandoval-Castaneda, M., Russell, B., Sivic, J., Shakhnarovich, G., Caba Heilbron, F.: Editduet: A multi-agent system for video non-linear editing. In: Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers. pp. 1–11 (2025)
- 22. Sun, H., Zhou, M., Chen, W., Xie, W.: Tr-detr: Task-reciprocal transformer for joint moment retrieval and highlight detection. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 38, pp. 4998–5007 (2024)
- 23. Sun, M., Farhadi, A., Seitz, S.: Ranking domain-specific highlights by analyzing edited videos. In: European conference on computer vision. pp. 787–802. Springer

(2014)

- 24. Wang, M., Yang, G.W., Hu, S.M., Yau, S.T., Shamir, A., et al.: Write-a-video: computational video montage from themed text. ACM Trans. Graph. 38(6), 177– 1 (2019)
- 25. Wang, Y., Wang, Z., Xu, B., Du, Y., Lin, K., Xiao, Z., Yue, Z., Ju, J., Zhang, L., Yang, D., et al.: Time-r1: Post-training large vision language model for temporal video grounding. arXiv preprint arXiv:2503.13377 (2025)
- 26. Xiao, Y., Luo, Z., Liu, Y., Ma, Y., Bian, H., Ji, Y., Yang, Y., Li, X.: Bridging the gap: A unified video comprehension framework for moment retrieval and highlight detection. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 18709–18719 (2024)

- 27. Xiong, B., Kalantidis, Y., Ghadiyaram, D., Grauman, K.: Less is more: Learning highlight detection from video duration. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 1258–1267 (2019)
- 28. Xu, J., Guo, Z., Hu, H., Chu, Y., Wang, X., He, J., Wang, Y., Shi, X., He, T., Zhu, X., et al.: Qwen3-omni technical report. arXiv preprint arXiv:2509.17765 (2025)
- 29. Xu, M., Wang, H., Ni, B., Zhu, R., Sun, Z., Wang, C.: Cross-category video highlight detection via set-based learning. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 7970–7979 (2021)
- 30. Yang, L., Chen, Z., Li, X., Jia, P., Long, L., Yang, J.: Agent-based video trimming. arXiv preprint arXiv:2412.09513 (2024)
- 31. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K.R., Cao, Y.: React: Synergizing reasoning and acting in language models. In: The eleventh international conference on learning representations (2022)
- 32. Zhu, Z., Lin, K.Q., Shou, M.Z.: Paper2video: Automatic video generation from scientific papers. arXiv preprint arXiv:2510.05096 (2025)

