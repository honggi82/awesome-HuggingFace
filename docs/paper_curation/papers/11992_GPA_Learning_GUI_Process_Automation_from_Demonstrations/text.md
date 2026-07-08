# arXiv:2604.01676v2[cs.CV]4Apr2026

## GPA: Learning GUI Process Automation from Demonstrations

#### Zirui Zhao∗ Jun Hao Liew Yan Yang Wenzhuo Yang Ziyang Luo Doyen Sahoo Silvio Savarese Junnan Li∗

Salesforce AI Research https://www.salesforceairesearch.com/gpa

### Abstract

GUI Process Automation (GPA) is a lightweight but general vision-based Robotic Process Automation (RPA), which enables fast and stable process replay with only a single demo. Addressing the fragility of traditional RPA and the non-deterministic risks of current vision language model-based GUI agents, GPA introduces three core benefits: (1) Robustness via Sequential Monte Carlo-based localization to handle rescaling and detection uncertainty; (2) Deterministic & Reliability safeguarded by readiness calibration; and (3) Privacy through fast, fully local execution. This approach delivers the adaptability, robustness, and security required for enterprise workflows. It can also be used as an MCP/CLI tool by other agents with coding capabilities so that the agent only reasons and orchestrates while GPA handles the GUI execution. We conducted a pilot experiment to compare GPA with Gemini 3 Pro (with CUA tools) and found that GPA achieves higher success rate with 10× faster execution speed in finishing long-horizon GUI tasks.

### 1 Introduction

The automation of repetitive tasks on Graphical User Interfaces (GUIs) is a critical driver of enterprise productivity. Historically, this domain has been dominated by Robotic Process Automation (RPA). While effective, traditional RPA imposes a heavy implementation burden: it typically requires skilled developers to manually define selectors (e.g., HTML tags, accessibility IDs) and script logic to handle edge cases. Furthermore, these scripts are notoriously brittle; a minor update to a website’s layout or a change in screen resolution can break the rigid coordinate or metadata-based rules, requiring costly maintenance [11, 15]. Beyond fragility, many tasks remain out of reach for RPA entirely: workflows involving dynamic content, complex conditional logic, or unstructured visual data cannot be expressed as deterministic rules.

Recently, large Vision Language Model (VLM)-based GUI agents have emerged as a more flexible alternative [36, 38]. These agents can interpret high-level instructions and interact with diverse interfaces. However, their reliance on generative probability introduces a fundamental flaw for mission-critical workflows: they are non-deterministic. The stochastic nature of probabilistic next-token prediction means that an agent may perform correctly nine times but “hallucinate” an action on the tenth. For real-world enterprise use cases that prioritise reliability, this inherent uncertainty is difficult to bound in practice. Additionally, VLM agents offer limited controllability: the same high-level instruction can yield different action sequences across runs, making it hard to audit, constrain, or predict behaviour. They also often suffer from high latency and data privacy risks, as they typically require streaming sensitive screenshots to external cloud-based APIs for inference [4, 13, 29].

In this work, we present GUI Process Automation (GPA), a system that combines the ease of demonstration-based learning with robust geometric matching. Unlike RPA, GPA requires no coding or selector inspection; the user records a single demonstration of the task, from which the system compiles the interaction trace into a structured workflow

∗Correspondence to {zirui.zhao, junnan.li}@salesforce.com. GPA is a proof of concept project for research. For collaboration, product development, or enterprise application, feel free to contact us.

GPA GUI Agents Traditional RPA Setup

Manual scripting and selector labelling Runtime AI Tiny local models only LLM call per action No AI Reliability Deterministic, reliable replay Non-deterministic Fragile under UI drift

Learn from a single demonstration

Prompt engineering

Screenshots sent to external providers

Privacy Fully local

Local

Variables Auto-extracted at build time Ad hoc via prompting Hard-coded

- Table 1: Comparison of GPA with VLM-based GUI agents and traditional RPA. GPA combines demonstrationbased workflow construction with deterministic local execution.

with ordered steps, action types, neighbor nodes, and parameterisable fields. Unlike VLM agents, GPA executes this workflow deterministically and efficiently, runs entirely locally to preserve data privacy, and is safeguarded by readiness checking—a confidence-gating mechanism that verifies the UI state before each action is executed.

The main technical challenge is then UI grounding [7]: given a recorded workflow step, reliably locating the corresponding interactive element on the current screen despite layout changes. Appearances shift due to rendering updates, dynamic content, and window rescaling, while element identifiers may be absent or inconsistent across sessions. Addressing this robustly requires moving beyond pixel coordinates or fragile metadata lookups. In summary, the key contributions of GPA are:

- • Robustness via Geometric Context: We employ a stable recording and building process with a multi-stage retrieval process culminating in a context-guided Sequential Monte Carlo (SMC) [8, 10] inference procedure. If the primary detector misses the target element, or if the application window is rescaled, the system uses neighboring nodes from the demo graph to infer the target’s location. This mimics human intuition, in which a user locates a button based on its position relative to surrounding text or icons.
- • Reliability: By utilising statistical calibration and a pre-computed null distribution, GPA executes actions only when confidence exceeds a strictly defined threshold. This eliminates the random failures associated with generative AI models and safeguards the system.
- • Privacy and Performance: The architecture relies on lightweight local models (e.g., IconCLIP [23]) rather than massive cloud models. We also trained an accurate UI detector to identify interactive elements. Together, these components ensure high-quality, low-latency execution and guarantee that sensitive visual data never leaves the local machine. In a pilot study, GPA achieves 100% success while running roughly 10× faster on average than Gemini’s computer-use agent baseline.

### 2 GPA: GUI Process Automation

A GUI workflow or GUI RPA is a sequence of GUI actions, such as clicks, text inputs, and hotkeys, performed in a fixed order to accomplish a repeatable task. GPA operates in two phases: (1) a demonstration phase, in which a user performs the task once while the system records each (screenshot, action) pair, and (2) an execution phase, in which the system automatically replays the workflow with new inputs. We address the UI grounding challenge by representing each UI state as a graph and formulating grounding as a graph-matching problem that combines visual similarity, geometric relationships, and statistical calibration.

- As shown in Figure 1, in the demonstration phase, the system records one user run as a sequence of screenshot-action pairs. For each step, it parses the screenshot into UI elements, builds a UI graph, and stores the target element together with nearby nodes in a reusable workflow template. In the execution phase, GPA processes the workflow step by step. For each step, it builds a UI graph from the current screen and finds the target element from the demonstration. A finite state machine orchestrates all the components, checking readiness of each step, control execution, and handle errors.

screenshot

[Figure 1]

Computer

Workﬂow template

Recording

execute/retry/ done/abort

UI Parser

Description: … Var: shell_cmd, …

Description: workflow open terminal and run commands …

[Figure 2]

Finite State Machine

UI Parser

UI Detector

KNN

[Figure 3]

- Step 1: Click Var:None

- Step 2: Click Var:None

Click(x, y)

ready?

Current UI graph

SMC

UI Graph

Click(x, y)

[Figure 4]

Fetch step info

Click(x’, y’)

…

Step i: Action Variable

UI graph for step i

…

[Figure 5]

Generate Step information, extract variables, etc

Write(“ls”)

Step N: Write( {shell_cmd}) Var:shell_cmd

LLM

Workﬂow template

(a) Demonstration phase

(b) Execution phase.

Figure 1: Two phases of GPA: demonstration phase (a) and execution phase (b). In the demonstration phase, a user interacts with the local desktop environment while the recorder captures screenshots and actions, parses them into step subgraphs, and applies LLM-based post-processing for cleanup and variable extraction. In the execution phase, GPA receives the latest observation from the local computer, loads the current recorded step, parses the screenshot into a UI graph, and combines the parsed graph, recorded workflow graph, and action in SMC localization. A finite state machine orchestrates all the components, checking readiness of each step, control execution, and handle errors.

#### 2.1 Problem Formulation

We formulate UI grounding as a localization problem, relying on two assumptions. First, nearby UI elements often preserve their relative layout across runs, even when the window is resized or slightly rearranged. Second, even if the target itself changes appearance or is missed by the detector, some nearby elements usually remain stable enough to serve as context for geometric inference.

We represent each UI state as a graph G = (V,E), where each node v ∈ V stores a bounding box bv, OCR text tv, and an icon embedding ev. Edges connect spatially nearby elements. For each demonstrated action, we form a demo graph Gd consisting of the target node vtarget and its nearby nodes {vi}Mi=1. Given a runtime graph Gr, the goal is to predict the click location of the target on the current screen.

To handle layout changes, we infer a latent variable θ = [x,y,sx,sy], where (x,y) is the target location and (sx,sy) models horizontal and vertical scaling between the demonstration and runtime screens1. Let Ztarget be the event of observing the target node in the runtime screenshot, and Zneighbor,i be observing the matching instance of the target’s i-th neighbor in the runtime screenshot at the same geometric relation to the target. Z = Ztarget ∪ Zneighbor,1 ∪ Zneighbor,2 ∪ ... denote the overall observation event across all neighbor nodes. For example, in Figure 3, the Ztarget denotes observing the matching instance of target checkbox, where Zneighbor,1 denotes observing the “Edit” text next to the target checkbox at the runtime screenshot. We seek the posterior

[Figure 6]

p(θ | Z) ∝ p(Z | θ)p(θ), (1)

Figure 2: Example UI graph with detected elements and edges. The highlighted node is the target, and nearby nodes helps localizing the target.

where p(θ) is a prior over plausible locations and scales, and p(Z | θ) is the likelihood of the observing the same demo graph in the runtime screenshot under hypothesis θ. Exact inference is difficult because context-candidate correspondences are ambiguous and the posterior can be multi-modal. We therefore use Sequential Monte Carlo (Section 2.3) [8, 10] to approximate the posterior and obtain a click prediction with a confidence score for readiness check.

1We use execution screen, runtime screen or current screen interchangeably.

The UI graph is constructed from screenshots in three steps: (1) extracting UI elements using a finetuned icon detector2 and OCR for text, (2) computing visual features for each element using IconCLIP [23], and (3) connecting spatially nearby elements via k-nearest neighbors to form edges. The resulting graph captures both visual appearance and spatial layout, allowing the retrieval algorithm to exploit neighboring context when matching across UI states. See Appendix B.1 for details.

We compute similarity between nodes based on their type. For textual elements, we combine fuzzy string matching with icon embedding similarity; for non-textual elements (icons, images), we use icon embedding similarity alone. This hybrid approach is robust to both visual changes and OCR errors (see Appendix B.2 for the full formulation).

#### 2.2 Demonstration and Workflow Building

During the demonstration phase, the user performs the task once while the recorder captures each (screenshot, action) pair as a keyframe. A background preprocessing thread immediately parses each incoming screenshot into a UI graph as keyframes arrive, so that processing overlaps with the user’s interaction and the workflow is ready to compile as soon as the recording ends.

For each step, GPA identifies the target UI element at the recorded click coordinates and selects nearby nodes using k-nearest-neighbour graph traversal, forming a step subgraph that encodes the target together with its local neighbors. For steps following a scroll action, the screenshot is preprocessed to mask regions outside the scrollable container before node matching, ensuring that neighbor nodes are drawn from within the scrolled area rather than from unrelated parts of the screen.

After recording, an LLM analyses the full step sequence to assign natural-language descriptions, identify parameterisable fields (e.g., form values, search terms) as typed workflow variables, and generate a workflow name and title. The result is a workflow template—a structured file containing the ordered steps, each with its step subgraph, action type, and variable bindings. At execution time, the runner loads this template and replays it against the live UI (see Appendix D for an example).

#### 2.3 Execution

- At runtime, GPA processes the workflow one step at a time. For each step, the runner captures an observation (screenshot and window metadata) from the environment, parses it into a UI graph, retrieves the target element from the demonstration subgraph, and checks whether the current screen is ready to act. If the confidence score exceeds a threshold, the action is generated and executed; otherwise the step is retried. To reduce per-step latency, a lightweight precheck pipeline speculatively processes the next step in the background while the environment is settling after the current action. Grounding as Sequential Monte Carlo Localization

We first attempt a direct match using appearance similarity between the target node vt in the demo graph and nodes in runtime graph Gr. We rank matching candidates by visual and textual similarity and return the top candidate when it is both high-confidence and unambiguous. Ambiguity is measured by the normalised entropy of the softmax distribution over the top-k candidate scores (see Appendix A.2): low entropy indicates a clear winner, while high entropy triggers the full SMC procedure. This fast path handles the common case where the UI is largely unchanged and avoids running SMC unnecessarily.

Hard cases arise when target matches are ambiguous or having low confidence, motivating us to use neighboring nodes in the demo graph to help locate the target element. We solve for θ via the posterior defined in Section 2.1. Given a demo graph, we can get target coordinate tdemo and neighbor node coordinates {cdemoi }Mi=1, and precompute neighbor-to-target displacement vectors ri = cdemoi − tdemo = [rix,riy]⊤. For each point θ, the predicted neighbor node location on the current screen is cˆi(θ) = [x,y] + [sxrix, syriy].

2The model is available at https://huggingface.co/Salesforce/GPA-GUI-Detector

1.0

Normalizedlikelihood

[Figure 7]

|[Figure 8]<br><br>wapp= 0.95<br><br>wapp= 0.95 wapp= 0.95<br><br>wapp= 0.95|
|---|

|[Figure 9]<br><br>wapp<br><br>|= 1.|
|---|
<br><br>00<br><br>wapp= 0.30 wapp= 0.20|
|---|

|[Figure 10]<br><br>wapp= 0.95<br><br>wapp= 0.95<br><br>wapp= 0.95<br><br>wapp= 0.95|
|---|

|[Figure 11]|
|---|

Screenshot

0.8

Runtime

0.6

w

0.4

0.2

0.0

|[Figure 12]|
|---|

|[Figure 13]|
|---|

|[Figure 14]|
|---|

|[Figure 15]|
|---|

DemoGraph

(b) p(Zneighbor,1 | · ), p(Zneighbor,2 | · ): the per-node likelihoods of observing neighbor elements around the target checkbox

(a) p(Ztarget | · ), the likelihood of observing target checkbox in runtime screenshot

(c) p(Z | · ), the joint likelihood

Figure 3: SMC likelihood visualization. The target to click is the checkbox next to “Edit”. (a) shows the likelihood of directly matching the target checkbox. The pointwise maximum over candidates forms the likelihood surface, showing high ambiguity since matching candidates are all identical. (b) shows the likelihood of matching the neighbor nodes of target, which helps us to differentiate the checkbox. (c) shows the joint likelihood, combining the results of (a) and (b).

For each node v ∈ Vdemo in demo graph, we can find a set of matching candidates in the runtime graph Cv obtained by appearance filtering (text match when available, otherwise icon embedding similarity). Instead of committing to a single correspondence, we jointly consider all candidates with explicit missing-node mass. For each neighbor node, we take the best-matching candidate (weighted Gaussian) and combine it with the missing mass:

wapp(v,c) N pc | cˆv(θ),σv2I , (2)

p(Zv | θ) = pmiss + max c∈Cv

where pc is the matching candidate center, wapp(n,c) is the appearance similarity, and pmiss allows nodes to be absent or unmatched. N pc | cˆv(θ),σv2I evaluates how likely the matching candidate of node i has the same displacement rn to θ as in the demo graph. The joint likelihood weights neighbor nodes by locality wloc(n) (closer neighbor nodes are more reliable to infer the location of target UI element; See Appendix A.1) and add a scale prior:

log p(Z | θ)+log p(θ) =

wloc(v) log p(Zv | θ)+log p(sx)+log p(sy), (3)

v∈Vdemo

|s)| | |
|---|---|---|
| | | |

p(

s

1 exp(µ)

Figure 4: Mixture prior over scale s: one mode favors no resizing (s = 1), and the other favors proportional rescaling.

where the scale prior for each axis is a mixture of two log-normals (as shown in Figure 4): one component centered at s = 1 (no resizing) and another at the observed window-size ratio (proportional rescaling). This allows the model to handle both unchanged and rescaled windows. Full details of the scale prior parameterization and geometric tolerance are in Appendix B.3.

We approximate the posterior p(θ | Z) using a tempered Sequential Monte Carlo (SMC) sampler with N weighted particles. Particles are initialized by back-projecting top-K context-candidate matches into target proposals with Gaussian jitter. The sampler then applies likelihood tempering: gradually “fading in” the likelihood from the prior to the full posterior, with adaptive temperature steps, importance reweighting, resampling, and Metropolis–Hastings rejuvenation at each stage. This standard SMC sampler procedure prevents particle degeneracy and handles the multi-modal posterior that arises from ambiguous context-candidate correspondences. The final prediction ˆt is the mean of the densest particle cluster. The likelihood computation, SMC iterations, etc, are all very fast in runtime with vectorized operations (in less than 0.2 sec). Full algorithmic details are in Appendix A.4.

#### Readiness Checking

Before executing each action, GPA gates on a confidence score C = p˜(Z | θ) × Cspatial derived from the SMC retriever. Likelihood confidence p˜(Z | θ) measures how well the predicted position explains the observed demo-runtime matches. It is different from the likelihood of individual particles but a overall confidence of clusters. Spatial confidence Cspatial

[Figure 16]

Figure 5: Process of SMC. The blue points are particles that gradually converge to the target prediction.

measures how tightly particles agree on the target location, based on the posterior predictive variance. The multiplicative combination ensures that both the evidence quality and posterior certainty must be high: strong likelihood with a scattered posterior (ambiguous) or a tight posterior with poor likelihood (confidently wrong) both yield low confidence. GPA proceeds only when C exceeds a threshold; otherwise the step is retried (see Appendix A.3 for details).

[Figure 17]

EXECUTE

No steps remaining

Step remaining

Ready

DONE

Check readiness & generation action

CHECK DECIDE

Not ready, attempts remaining

Sleep 1s, re-observe

Run out retries

FAILED

RETRY

(b) DECIDE state logic. Scroll-to-find steps first check target visibility; regular steps follow the bounded retry path directly.

(a) Top-level step-execution FSM.

- Figure 6: Step-execution FSM and the DECIDE state detail. The top-level FSM is shared across all steps, while the decision logic adds a small scroll-specific branch when needed.

#### Execution Control

Grounding and readiness checking determine where GPA should act and whether the current screen matches the expected state. A step-level execution controller then decides what to do next. As shown in Figure 6a, the top-level controller follows a simple bounded-retry finite state machine (FSM). For each workflow step, GPA alternates between EXECUTE and DECIDE. The EXECUTE state parses the current UI, checks readiness, and generates an action candidate. The DECIDE state then chooses among three outcomes: finish the step, retry after re-observing the screen, or fail the step after exhausting the retry budget. This structure prevents the workflow from drifting after a transient mismatch.

The DECIDE state contains the only step-specific branch. For regular steps, shown in Figure 6b, the logic is minimal: if the action is ready, GPA finishes the step; otherwise, it either retries or fails depending on the remaining attempt budget. A special case is scroll-to-find step, which is a scroll action whose purpose is to reveal a later target element that is currently off-screen. For scroll-to-find steps, DECIDE first checks whether the target is already visible. If so, GPA skips the scroll and completes the step immediately. If the target is not yet visible, GPA checks whether the scroll action itself is ready. A ready scroll action is executed and the next screen is re-evaluated on the following iteration; otherwise, GPA follows the same bounded retry-or-fail rule as in the regular case. In this way, scroll-specific logic refines the decision state without changing the top-level FSM.

This finite-state controller complements the SMC-based matcher: the matcher provides localization and confidence, while the controller turns that confidence into safe retry, execute, or fail decisions. GPA further reduces latency with a lightweight precheck pipeline that speculatively processes upcoming steps in the background while the environment settles after the current action (see Appendix C).

### 3 Pilot Experiments

We conduct a small-scale pilot study of 16 desktop GUI tasks to compare GPA against a VLM-based baseline: the Gemini computer-use agent (gemini-3-pro) that executes a demonstrated workflow description. We categorize tasks by the length of the recorded demonstration (number of steps): simple tasks have short demos and hard tasks have long demos. For each task, we measure (i) wall-clock runtime and (ii) success rate.

The task set covers a mix of common desktop productivity and enterprise workflows. The simple tasks (average 10.8 steps) include drafting an email, downloading a receipt from email, downloading a utility receipt followed by counting, flight searching, and flight booking. The hard tasks (average 27.27 steps) include setting a Google Calendar event, setting a calendar event followed by drafting an email, flight searching and booking, two Agentforce tasks, reimbursement submission with and without receipt retrieval, two SAP ERP form-filling tasks, and two HR workflows for interview scheduling and candidate information entry. This mix lets us evaluate both short routine automations and longer multi-stage enterprise processes.

| |GPA Gemini 3 pro<br><br>|
|---|---|
|Task level|Avg demo steps Success (%) Time (s) Success (%) Time (s)<br><br>|
|Simple Hard Avg<br><br>|10.80 100 17.84 93.2 210.66 27.27 100 40.96 87.64 383.24 22.13 100 33.74 89.38 329.31<br><br>|

- Table 2: Pilot results: GPA vs. Gemini GUI agent. Success rate and runtime aggregated by task difficulty (defined by demonstration length).

#### Experimental protocol

For each task, GPA is given a single recorded demonstration and then replays the workflow using its visual graph retrieval, context-node-based matching, readiness checks, and bounded retry logic. The Gemini baseline is given the same demonstrated video and process theml into textual form, and use it to guide its action autoregressively during execution. We report average runtime and success rate over the tasks in each difficulty group. Although this is a pilot-scale study, it is sufficient to test whether the practical trends are consistent with the design goals of GPA.

#### Results

- As shown in Table 2, GPA achieves 100% success across both difficulty levels while being roughly 10× faster (33.74s vs. 329.31s). Gemini’s success drops from 93.2% to 87.64% as tasks grow longer. Three factors explain the gap:

- • Reliability. GPA follows a fixed demonstrated procedure and acts only when the readiness checker confirms a confident match. Bounded retries handle transient delays; unresolvable mismatches trigger explicit failure rather than guessing. Gemini must infer the correct action at every step, and each inference carries a small misidentification risk that compounds over the trajectory.
- • Latency. Each GPA step involves only on-device screenshot capture, OCR parsing, and local subgraph retrieval (milliseconds to low seconds). Gemini incurs network round-trip and VLM inference latency per step, which accumulates to >383s on ∼27-step hard tasks versus ∼41s for GPA.
- • Scaling with length. GPA’s fixed-plan execution adds only a cheap matching operation per step with no additional drift risk. The generative agent may misread a label, click an adjacent element, or miss a page load at each step—over 27 steps these per-step failure probabilities compound, explaining the ∼6pp success drop while GPA remains at 100%.

### 4 Related Work

From RPA to visual GUI automation. Classical robotic process automation (RPA) is designed for stable, repetitive workflows and is attractive in practice because it can automate legacy systems without requiring API-level integration.

- At the same time, prior work has emphasized that RPA is most effective when processes are structured, rule-based, and relatively invariant, and that coupling automation logic to application structure or surface behavior makes such systems brittle under software evolution and interface changes [11, 15]. Earlier work in visual GUI automation and testing partially addressed this problem by replacing DOM- or selector-centric interaction with screenshot matching [6] and image-based scripting [3]. However, these methods still rely on low-level visual correspondence and offer limited semantic understanding, which leaves them sensitive to appearance changes and ambiguous interface states.

Foundation-model GUI agents. Recent work reframes GUI automation as a multimodal decision-making problem over screenshots, structured interface cues, and grounded actions. Recent surveys organize the space by perception, grounding, planning, memory, and execution across web, mobile, and desktop settings [36, 38]. The line first matured on the web through datasets and evaluation suites such as Mind2Web [9], WebArena [48], and VisualWebArena [21], which exposed the gap between text-only agents and visually grounded multimodal agents. End-to-end web agents such as WebVoyager [16] and AutoWebGLM [22] showed that large multimodal models can act directly on live websites, while SeeClick and ScreenSpot [7] highlighted GUI grounding as a central bottleneck for more general visual interaction. Subsequent environments expanded evaluation from browser automation to full computer use, including OmniACT [20], OSWorld [40], AndroidWorld [34], WindowsAgentArena [5], and ScreenSpot-Pro [24], shifting attention toward long-horizon, cross-application tasks under realistic interface variability.

At the system level, early foundation-model GUI agents such as CogAgent [17], Ferret-UI [44], AppAgent [47], MobileAgent [37], and UFO [46] established screenshot-native interaction for web, mobile, and Windows environments. More recent work has diversified into stronger open-source grounding/action models, including UGround [14], OS-ATLAS [39], Aguvis [42], and TinyClick [32], and into increasingly capable execution frameworks such as AutoGLM [28], Agent S [1], Agent S2 [2], UI-TARS [33], PC-Agent [27], Mobile-Agent-v3 [43], Mobile-Agent-v3.5 [41], AppAgentX [19], and UFO2 [45], which combine hierarchical planning, specialist modules, reinforcement learning, memory, or hybrid GUI/API control to improve long-horizon execution. In parallel, productized computer-use systems such as Anthropic’s computer use [4], OpenAI’s Operator [30] and CUA [29], Google’s Project Mariner [12] and Gemini 2.5 Computer Use [13], and open-source platforms such as OpenClaw [31] show that GUI agents are rapidly moving from benchmark prototypes toward deployable assistants and automation substrates.

Learning from demonstration. Another major direction reduces the burden on zero-shot planning by using demonstrations to specify the desired workflow. Early work such as HILC [18] showed that users can teach GUI tasks through demonstration, with follow-up questions helping resolve ambiguity in one-shot task specification. More recently, LearnAct [26] introduces a demonstration-based framework for mobile GUI agents together with the LearnGUI benchmark, using dedicated modules to parse demonstrations, retrieve relevant prior experience, and execute actions in new contexts. Instruction Agent [25] similarly leverages a single expert demonstration to extract step-by-step instructions and constrains execution with verification and backtracking, improving performance on complex longhorizon tasks. This literature is especially relevant for settings where the goal is not unconstrained autonomy, but dependable reproduction of user-preferred workflows.

### 5 Conclusions & Limitations

We propose GPA, a novel RPA framework that requires only a single demonstration. Powered by mature algorithms for grounding UI elements during process replay, we show that many GUI operations may not require expensive LLM agents to perform. GPA can serve as a powerful tool for GUI automation on local desktops, in browsers, in enterprise software, and in other domains where GUI elements are well-defined and stable. It can also be viewed as a computer-use

skill with command-line (CLI) support, enabling integration with shell scripting and composition of multiple workflows for even more complex and flexible use cases.

We also acknowledge several limitations. GPA is a record-and-replay system: it does not have reasoning or decisionmaking capabilities. It executes workflows as recorded and cannot adapt to situations that require judgment. For example, selecting a date on a calendar widget requires reasoning about the current state (e.g., which month is currently displayed, how many clicks are needed to navigate to the target month). GPA cannot perform this reasoning, so date pickers will only work correctly if the same date as in the recording is being selected.

GPA can be extended to support fully automated, no human-in-the-loop operation: LLM agents could record workflows and perform self-healing when a workflow becomes stale due to UI updates. We are also interested in precondition tracking for tool use, as the preconditions required for a tool call may not always be satisfied, requiring we have a reasonable state estimation of the software we are operationg on. This represents a general challenge for tool design in AI agent systems.

### References

- [1] Saaket Agashe, Jiuzhou Han, Shuyu Gan, Jiachen Yang, Ang Li, and Xin Eric Wang. Agent s: An open agentic framework that uses computers like a human. In International Conference on Learning Representations, 2025.
- [2] Saaket Agashe, Kyle Wong, Vincent Tu, Jiachen Yang, Ang Li, and Xin Eric Wang. Agent s2: A compositional generalistspecialist framework for computer use agents, 2025.
- [3] Emil Alégroth, Robert Feldt, and Lisa Ryrholm. Visual gui testing in practice: Challenges, problems and limitations. Empirical Software Engineering, 20(3):694–744, 2015.
- [4] Anthropic. Introducing computer use, a new claude 3.5 sonnet, and claude 3.5 haiku. https://www.anthropic.com/news/3-5models-and-computer-use, 2024. Accessed 2026-03-25.
- [5] Rogerio Bonatti, Dan Zhao, Francesco Bonacci, Dillon Dupont, Sara Abdali, Yinheng Li, Yadong Lu, Justin Wagle, Kazuhito Koishida, Arthur Bucker, Lawrence Jang, and Zack Hui. Windows agent arena: Evaluating multi-modal os agents at scale. In International Conference on Machine Learning, 2024.
- [6] Tsung-Hsiang Chang, Tom Yeh, and Robert C. Miller. Gui testing using computer vision. In Proceedings of the 28th ACM SIGCHI Conference on Human Factors in Computing Systems, pages 1535–1544. ACM, 2010.
- [7] Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Li YanTao, Jianbing Zhang, and Zhiyong Wu. Seeclick: Harnessing gui grounding for advanced visual gui agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9313–9332, Bangkok, Thailand, 2024. Association for Computational Linguistics.
- [8] Pierre Del Moral, Arnaud Doucet, and Ajay Jasra. Sequential monte carlo samplers. Journal of the Royal Statistical Society Series B: Statistical Methodology, 68(3):411–436, 2006.
- [9] Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web. In Neural Information Processing Systems, 2023.
- [10] Petar M Djuric, Jayesh H Kotecha, Jianqui Zhang, Yufei Huang, Tadesse Ghirmai, Mónica F Bugallo, and Joaquin Miguez. Particle filtering. IEEE signal processing magazine, 20(5):19–38, 2003.
- [11] Tom Roar Eikebrokk and Dag Håkon Olsen. Robotic process automation and consequences for knowledge workers; a mixed-method study. In Responsible Design, Implementation and Use of Information and Communication Technology, volume 12066 of Lecture Notes in Computer Science, pages 114–125. Springer, 2020.
- [12] Google. Google introduces gemini 2.0: A new ai model for the agentic era. https://blog.google/innovation-and-ai/models-andresearch/google-deepmind/google-gemini-ai-update-december-2024/, 2024. Accessed 2026-03-25.

- [13] Google. Introducing the gemini 2.5 computer use model. https://blog.google/innovation-and-ai/models-and-research/googledeepmind/gemini-computer-use-model/, 2025. Accessed 2026-03-25.
- [14] Boyu Gou, Ruohan Wang, Boyuan Zheng, Yanan Xie, Cheng Chang, Yiheng Shu, Huan Sun, and Yu Su. Navigating the digital world as humans do: Universal visual grounding for gui agents. In International Conference on Learning Representations, 2025.
- [15] Geert Haerens and Herwig Mannaert. On evolvability issues of robotic process automation (rpa). In PATTERNS 2020: The Twelfth International Conference on Pervasive Patterns and Applications, pages 25–30. IARIA, 2020.
- [16] Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. Webvoyager: Building an end-to-end web agent with large multimodal models. In Annual Meeting of the Association for Computational Linguistics, 2024.
- [17] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxuan Zhang, Juanzi Li, Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang. Cogagent: A visual language model for gui agents. In Computer Vision and Pattern Recognition, 2024.
- [18] Thanapong Intharah, Daniyar Turmukhambetov, and Gabriel J. Brostow. Help, it looks confusing: Gui task automation through demonstration and follow-up questions. In Proceedings of the 22nd International Conference on Intelligent User Interfaces, pages 233–243, Limassol, Cyprus, 2017. ACM.
- [19] Wenjia Jiang, Yangyang Zhuang, Chenxi Song, Xu Yang, Joey Tianyi Zhou, and Chi Zhang. Appagentx: Evolving gui agents as proficient smartphone users, 2025.
- [20] Raghav Kapoor, Yash Parag Butala, Melisa Russak, Jing Yu Koh, Kiran Kamble, Waseem Alshikh, and Ruslan Salakhutdinov. Omniact: A dataset and benchmark for enabling multimodal generalist autonomous agents for desktop and web. In European Conference on Computer Vision, 2024.
- [21] Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Chong Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Ruslan Salakhutdinov, and Daniel Fried. Visualwebarena: Evaluating multimodal agents on realistic visual web tasks, 2024.
- [22] Hanyu Lai, Xiao Liu, Iat Long Iong, Shuntian Yao, Yuxuan Chen, Pengbo Shen, Hao Yu, Hanchen Zhang, Xiaohan Zhang, Yuxiao Dong, and Jie Tang. Autowebglm: A large language model-based web navigating agent. In Knowledge Discovery and Data Mining, 2024.
- [23] Kaixin Li. Iconclip. https://huggingface.co/likaixin/IconClip-ViT-B-32. Accessed: 2026-3-25.
- [24] Kaixin Li, Ziyang Meng, Hongzhan Lin, Ziyang Luo, Yuchen Tian, Jing Ma, Zhiyong Huang, and Tat-Seng Chua. Screenspotpro: Gui grounding for professional high-resolution computer use. In ACM Multimedia, 2025.
- [25] Yinheng Li, Hailey Hultquist, Justin Wagle, and Kazuhito Koishida. Instruction agent: Enhancing agent with expert demonstration, 2025.
- [26] Guangyi Liu, Pengxiang Zhao, Liang Liu, Zhiming Chen, Yuxiang Chai, Shuai Ren, Hao Wang, Shibo He, and Wenchao Meng. Learnact: Few-shot mobile gui agent with a unified demonstration benchmark, 2025.
- [27] Haowei Liu, Xi Zhang, Haiyang Xu, Yuyang Wanyan, Junyang Wang, Ming Yan, Ji Zhang, Chunfeng Yuan, Changsheng Xu, Weiming Hu, and Fei Huang. Pc-agent: A hierarchical multi-agent collaboration framework for complex task automation on pc, 2025.
- [28] Xiao Liu, Bo Qin, Dongzhu Liang, Guang Dong, Hanyu Lai, Hanchen Zhang, Hanlin Zhao, Iat Long Iong, Jiadai Sun, Jiaqi Wang, Junjie Gao, Junjun Shan, Kangning Liu, Shudan Zhang, Shuntian Yao, Siyi Cheng, Wentao Yao, Wenyi Zhao, Xinghan Liu, Xinyi Liu, Xinying Chen, Xinyue Yang, Yang Yang, Yifan Xu, Yu Yang, Yujia Wang, Yulin Xu, Zehan Qi, Yuxiao Dong, and Jie Tang. Autoglm: Autonomous foundation agents for guis, 2024.
- [29] OpenAI. Computer-using agent. https://openai.com/index/computer-using-agent/, 2025. Accessed 2026-03-25.
- [30] OpenAI. Introducing operator. https://openai.com/index/introducing-operator/, 2025. Accessed 2026-03-25.

- [31] OpenClaw. Openclaw documentation. https://docs.openclaw.ai/, 2026. Accessed 2026-03-25.
- [32] Pawel Pawlowski, Krystian Zawistowski, Wojciech Lapacz, Adam Wiacek, Marcin Skorupa, Sebastien Postansque, and Jakub Hoscilowicz. Tinyclick: Single-turn agent for empowering gui automation. In Interspeech, 2025.
- [33] Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, et al. Ui-tars: Pioneering automated gui interaction with native agents, 2025.
- [34] Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, Daniel Toyama, Robert Berry, Divya Tyamagundlu, Timothy Lillicrap, and Oriana Riva. Androidworld: A dynamic benchmarking environment for autonomous agents. In International Conference on Learning Representations, 2025.
- [35] Bernard W. Silverman. Density estimation for statistics and data analysis. Chapman & Hall, 1986.
- [36] Fei Tang, Haolei Xu, Hang Zhang, Siqi Chen, Xingyu Wu, Yongliang Shen, Wenqi Zhang, Guiyang Hou, Zeqi Tan, Yuchen Yan, Kaitao Song, Jian Shao, Weiming Lu, Jun Xiao, and Yueting Zhuang. A survey on (m)llm-based gui agents, 2025.
- [37] Junyang Wang, Haiyang Xu, Jiabo Ye, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. Mobile-agent: Autonomous multi-modal mobile device agent with visual perception, 2024.
- [38] Shuai Wang, Weiwen Liu, Jingxuan Chen, Yuqi Zhou, Weinan Gan, Xingshan Zeng, Yuhan Che, Shuai Yu, Xinlong Hao, Kun Shao, Bin Wang, Chuhan Wu, Yasheng Wang, Ruiming Tang, and Jianye Hao. Gui agents with foundation models: A comprehensive survey, 2024.
- [39] Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, and Yu Qiao. Os-atlas: A foundation action model for generalist gui agents, 2024.
- [40] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. In Neural Information Processing Systems, 2024.
- [41] Haiyang Xu, Xi Zhang, Haowei Liu, Junyang Wang, Zhaozai Zhu, Shengjie Zhou, Xuhao Hu, Feiyu Gao, Junjie Cao, Zihua Wang, Zhiyuan Chen, Jitong Liao, Qi Zheng, Jiahui Zeng, Ze Xu, Shuai Bai, Junyang Lin, Jingren Zhou, and Ming Yan. Mobile-agent-v3.5: Multi-platform fundamental gui agents, 2026.
- [42] Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tianbao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, and Caiming Xiong. Aguvis: Unified pure vision agents for autonomous gui interaction. In International Conference on Machine Learning, 2025.
- [43] Jiabo Ye, Xi Zhang, Haiyang Xu, Haowei Liu, Junyang Wang, Zhaoqing Zhu, Ziwei Zheng, Feiyu Gao, Junjie Cao, Zhengxi Lu, Jitong Liao, Qi Zheng, Fei Huang, Jingren Zhou, and Ming Yan. Mobile-agent-v3: Foundamental agents for gui automation, 2025.
- [44] Keen You, Haotian Zhang, Eldon Schoop, Floris Weers, Amanda Swearngin, Jeffrey Nichols, Yinfei Yang, and Zhe Gan. Ferret-ui: Grounded mobile ui understanding with multimodal llms. In European Conference on Computer Vision, 2024.
- [45] Chaoyun Zhang, He Huang, Chiming Ni, Jian Mu, Si Qin, Shilin He, Lu Wang, Fangkai Yang, Pu Zhao, Chao Du, Liqun Li, Yu Kang, Zhao Jiang, Suzhen Zheng, Rujia Wang, Jiaxu Qian, Minghua Ma, Jian-Guang Lou, Qingwei Lin, Saravan Rajmohan, and Dongmei Zhang. Ufo2: The desktop agentos, 2025.
- [46] Chaoyun Zhang, Liqun Li, Shilin He, Xu Zhang, Bo Qiao, Si Qin, Minghua Ma, Yu Kang, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang, and Qi Zhang. Ufo: A ui-focused agent for windows os interaction. In North American Chapter of the Association for Computational Linguistics, 2025.
- [47] Chi Zhang, Zhao Yang, Jiaxuan Liu, Yucheng Han, Xin Chen, Zebiao Huang, Bin Fu, and Gang Yu. Appagent: Multimodal agents as smartphone users. In International Conference on Human Factors in Computing Systems, 2025.

###### [48] Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. Webarena: A realistic web environment for building autonomous agents. In International Conference on Learning Representations, 2024.

### A SMC Retriever Details

- A.1 Locality Weight Bandwidth Each context node i contributes to the log-likelihood weighted by

wloc(i) = exp −

∥vi∥2 2σloc2

, (4)

where vi = cdemoi − tdemo is the demo displacement vector. The bandwidth σloc controls how rapidly the weight decays with distance from the target: a small σloc trusts only nearby nodes, while a large one incorporates distant ones.

Motivation.

In dense interfaces (e.g. a spreadsheet), nearby nodes are abundant and distant ones are unreliable because the layout may differ across regions. In sparse interfaces (e.g. a dialog with a few buttons), even distant nodes provide useful geometric constraints. An effective σloc must therefore adapt to the spatial distribution of nodes.

Adaptive bandwidth. We apply Silverman’s rule [35] using the RMS distance from the target to context node centres as the scale estimate:

σloc = clamp 1.06 σˆ n−1/5, σmin, σmax , (5)

where σˆ = n1 ni=1 ∥vi∥2 is the root-mean-square (RMS) distance from the target to the n demo nodes—i.e., the square root of the average squared displacement magnitude. Unlike the per-axis standard deviation (which is

translation-invariant and only measures the spread of context node positions regardless of the target), RMS distance is target-centric: it reflects how far nodes actually are from the target. When the target sits at the edge of the context node cluster, σˆ is larger and distant nodes receive more weight; when the target is surrounded by nodes, σˆ is smaller and the locality weight is more selective. The n−1/5 factor comes from Silverman’s asymptotically optimal bandwidth rate; it means that with more nodes we can afford to be more local.

Behaviour.

- • Dense UI (20 nodes within ∼100px of target): RMS distance σˆ ≈ 70px, σloc ≈ 1.06 × 70 × 20−0.2 ≈ 41px. Only nearby nodes contribute significantly.
- • Sparse UI (3 nodes, avg. 500px from target): σˆ ≈ 500px, σloc ≈ 1.06 × 500 × 3−0.2 ≈ 425px. Even distant nodes receive meaningful weight.
- • Target at edge (same nodes as dense, but target far from cluster centre): RMS distance increases, giving a larger σloc so nodes in the far part of the cluster are not prematurely down-weighted.
- • Edge cases: fewer than 2 nodes ⇒ fallback constant (σloc = 1000px); all nodes at target position ⇒ σˆ → 0, clamped to σmin = 30px.

- A.2 Entropy-Based Ambiguity Detection

The fast-path direct match bypasses the SMC particle filter when the top candidate is clearly the best. This decision requires an ambiguity measure over the ranked candidate scores.

#### Normalised entropy.

Given the top-k candidate similarity scores s1 ≥ s2 ≥ ··· ≥ sk, we form a softmax distribution at temperature τ and compute the normalised entropy:

k

1 log keff

exp(sj/τ)

. (6)

H = −

pj log pj, pj =

k ℓ=1 exp(sℓ/τ)

j=1

The normalisation uses the effective candidate count keff = max(|{j : pj > 0.01}|, 2) rather than the raw k. This prevents dilution from irrelevant tail candidates: when only the top two candidates carry meaningful probability mass, keff = 2 regardless of how many weak candidates exist. Without this guardrail, normalising by log k (e.g. k = 5) would reduce H and allow ambiguous top-2 pairs to pass as “clear” simply because three negligible candidates inflate the denominator. A match is deemed unambiguous when H < Hthr.

Role of temperature. The temperature τ controls sensitivity to score differences. Since similarity scores lie in [0,1], a small τ (e.g. 0.02) amplifies even modest gaps into sharply peaked distributions, while a large τ (e.g. 0.5) flattens them. At τ = 0.02:

- • Scores [0.95,0.40,0.30]: H ≈ 0.00 (unambiguous).
- • Scores [0.92,0.90,0.30]: H ≈ 0.84 (ambiguous — the top two are too close).
- • Scores [0.85,0.84,0.83,0.82]: H ≈ 0.90 (ambiguous — four near-equal candidates).

#### Comparison with score-gap heuristic.

The simpler margin test s1 − s2 > ∆ only examines the top two candidates and is sensitive to score scale (a gap of 0.06 means different things at s1 = 0.95 vs. s1 = 0.50). Entropy naturally accounts for (i) the number of close competitors, not just the runner-up; (ii) the relative magnitudes of all top-k scores; and (iii) the overall peakedness of the distribution. For instance, a candidate leading by 0.05 over three close competitors (H high) is more ambiguous than one leading by 0.05 over a single runner-up with all others far behind (H low), a distinction the margin test cannot make.

#### Two-stage gating.

The direct match requires both a minimum score threshold (s1 > smin, default 0.9) and low entropy (H < Hthr, default 0.5). The score threshold ensures the top candidate is genuinely a good match; the entropy threshold ensures it is clearly better than alternatives. Only when both conditions hold does GPA skip SMC.

#### A.3 Confidence Scoring

The final confidence C = p˜(Z | θ) × Cspatial is the product of two independent measures that capture complementary aspects of retrieval quality.

- A.3.1 Likelihood Confidence Likelihood confidence measures how well the predicted target position explains the observed node-candidate matches under the likelihood model. Per-node score. For each node v ∈ Vdemo, define three log-scale quantities evaluated at the predicted hypothesis θˆ:

- • log pmatch,n: the best joint score (geometry + appearance) among all candidates for node v, i.e. maxc∈C

v −∥pc − cˆv(θˆ)∥2/(2σv2) + log wapp(v,c) .

- • log pbest_sim,n: the appearance-only ceiling—the best log wapp(n,c) among the winning candidate, representing the score if geometry were perfect.
- • log pmiss,v = log pmiss: the missing-node baseline.

The per-node confidence normalises the match quality relative to the best achievable:

log pmatch,v − log pmiss,v log pbest_sim,v − log pmiss,v

cv = clip

, 0, 1 . (7)

When log pmatch,v ≥ log pbest_sim,v (near-perfect geometry), cv → 1. When log pmatch,v ≤ log pmiss,v (worse than the missing baseline), cv = 0. When the best candidate’s similarity itself falls below the missing baseline (log pbest_sim,v < log pmiss,v), the denominator is non-positive and cv is set to 0—matching to a very weak candidate should not contribute positively.

Weighted aggregation. The overall likelihood confidence is a locality-weighted average:

wloc(v)cv v∈Vdemo wloc(v)

p˜(Z | θ) = v∈V

. (8)

demo

This gives more influence to nearby nodes whose spatial predictions are more reliable. A high p˜(Z | θ) indicates that the predicted position is geometrically consistent with multiple context node observations; a low value signals that nodes cannot find their expected matches near the prediction, often because the UI layout has changed substantially.

#### A.3.2 Spatial Confidence

Spatial confidence measures how tightly the particle population agrees on the target location, independent of the context-candidate likelihood.

#### Posterior predictive variance.

After SMC convergence, we have N weighted particles {(θ(j),w(j))} with positions x(j) = (θx(j),θy(j)). We compute the weighted mean and covariance:

µ =

j

w(j)x(j), Σ =

j

w(j)(x(j) − µ)(x(j) − µ)⊤. (9)

Under an isotropic Gaussian approximation with variance σ¯2 = 12 tr(Σ), the probability that the true position lies within radius r of the mean is:

r2 2σ¯2

. (10)

Cspatial = 1 − exp −

This is the CDF of a Rayleigh distribution, which arises as the distribution of distances from the mean of a 2-D isotropic Gaussian.

Density-adaptive radius. The acceptance radius adapts to context node density:

r = rbase + α · σloc, (11)

where rbase = 50px and α = 0.2. In sparse UIs (σloc large), fewer nodes provide weaker constraints, so particles naturally spread more. A fixed radius would over-penalise this expected spread and produce artificially low confidence. Scaling r by σloc ensures that the acceptance region grows with the spatial scale of the problem.

#### Behaviour.

- • Dense UI (σloc ≈ 40px, r ≈ 58px): nearly the base radius; tight convergence required.
- • Moderate UI (σloc ≈ 200px, r ≈ 90px): tolerates wider particle spread.
- • Sparse UI (σloc ≈ 500px, r ≈ 150px): substantial tolerance for the limited evidence.
- • Tight convergence (σ¯ ≈ 5px, r = 50): Cspatial ≈ 1.0.
- • Wide scatter (σ¯ ≈ 100px, r = 50): Cspatial ≈ 0.12.
- • Degenerate (all particles at same point): tr(Σ) = 0 ⇒ Cspatial = 1.

Why multiplicative combination. The product C = p˜(Z | θ) × Cspatial requires both factors to be high for the final confidence to be high:

- • High p˜(Z | θ), low Cspatial: nodes match well individually, but particles are scattered across multiple modes ⇒ ambiguous posterior, low confidence (correct: the prediction is unreliable).

- • Low p˜(Z | θ), high Cspatial: particles converge tightly, but to a position where nodes do not match ⇒ confidently wrong, low confidence (correct: the prediction is likely incorrect).
- • High p˜(Z | θ), high Cspatial: strong context evidence and tight particle agreement ⇒ high confidence.

This decomposition separates two failure modes that the readiness checker must detect: observation mismatch (context confidence) and posterior uncertainty (spatial confidence).

#### A.4 SMC Algorithm

Rather than directly sampling the often sharply peaked and multi-modal posterior, the SMC sampler uses likelihood tempering to define a sequence of intermediate targets:

πβ(θ) ∝ p(Z | θ)β p(θ), 0 = β0 < β1 < ··· < βS = 1. (12)

Intuitively, β “fades in” the likelihood: πβ=0 is the prior and πβ=1 is the true posterior. We choose βs+1 adaptively to maintain a desired effective sample size (ESS) under the incremental importance weights, with an optional cap on ∆β to ensure gradual annealing.

At each stage, we apply importance reweighting, resampling, and a Markov chain Monte Carlo (MCMC) rejuvenation step targeting πβ

. This combination prevents particle degeneracy and improves mixing between modes. Concretely:

s

Algorithm 1 Tempered SMC Sampler for Static UI Localization Require: Context-to-target displacement vectors {ri} and candidate sets {Ci} Ensure: Predicted click position ˆt and confidence

- 1: Initialize particles {θ(j)}Nj=1 from proposal (fallback: uniform over screen bounds)
- 2: Set β0 ← 0
- 3: for s = 0 to S − 1 do
- 4: Choose βs+1 ∈ (βs,1] to meet an ESS target (optionally cap ∆β)
- 5: Incremental reweight: w˜(j) ∝ w(j) · p(Z | θ(j))(β

s+1−βs)

- 6: Normalize: w(j) ← w˜(j)/ k w˜(k)
- 7: Resample: draw {θ(j)} with replacement according to w (optionally ESS-gated)
- 8: Rejuvenate (MCMC): for each particle, run L MH steps targeting πβ

s+1

(θ) ∝ p(Z | θ)β

s+1

p(θ)

- 9: If confidence > cmin or (β=1 and confidence stable): break
- 10: end for
- 11: Optional refinement: with β = 1, repeat resample+MH for a few additional steps to tighten concentration
- 12: Output ˆt as the densest particle cluster mean (grid-based clustering)

The Metropolis-Hasting (MH) proposal is a symmetric Gaussian random walk. Step sizes are scaled by the current particle spread (so particles contract instead of re-scattering near β=1) and modulated by an intensity factor. During tempering (β < 1), intensity decays to progressively focus particles. After β reaches 1, intensity resets with a floor so that MH retains enough step size to merge surviving sub-clusters against the full posterior. The loop may exit early when confidence exceeds a threshold or when β=1 and confidence has stabilized.

### B UI Graph and Similarity Details

#### B.1 UI Graph Construction

The UI graph is constructed from screenshots in three steps. First, we extract UI elements from the screenshot using computer vision detectors. We finetuned OmniParser for icon detection and OCR for text extraction. Each detected element becomes a node with its bounding box coordinates. Second, we compute visual features for each element using IconCLIP (ViT-B-32) [23] to produce icon embeddings. Third, we build the graph structure by connecting spatially

nearby elements. We use k-nearest neighbors based on element centers to create edges, typically with k = 5. This creates a graph where edges represent spatial proximity rather than semantic relationships.

The resulting graph captures both the visual appearance and spatial layout of the UI. Elements that are close together in the interface are connected by edges. This structure allows the retrieval algorithm to use context from neighboring elements when matching across different UI states.

#### B.2 Similarity Computation

We compute similarity between nodes based on their type and content. For textual elements such as buttons with text or text fields, we use fuzzy string matching. The fuzzy matching uses the Levenshtein distance ratio (via RapidFuzz) with length-adaptive tolerance that provides more forgiveness for OCR errors in short strings, returning 1.0 for exact matches and smoothly degrading for partial matches. We then combine this text similarity with icon embedding similarity:

s(vd,vc) = 0.9 · stext(vd,vc) + 0.1 · cos(ed,ec), (13)

where ed,ec are icon embeddings. For non-textual elements such as pure icons or images, we use only icon embedding similarity: s(vd,vc) = cos(ed,ec).

###### B.3 Scale Prior and Geometric Tolerance The scale prior for each axis is a mixture of two log-normals:

p(sx) = w LogNormal(sx; 0, σs,x) + (1 − w)LogNormal(sx; µx, σs,x), (14)

and analogously for sy with mean µy = log(Hlive/Hdemo). The identity component (µ = 0, weight w) captures the case where no resizing occurs; the ratio component (µx = log(Wlive/Wdemo), weight 1−w) captures a full proportional rescaling of the window. Using per-axis ratios instead of a single isotropic mean accounts for independent horizontal and vertical rescaling. Penalties are applied symmetrically in log-space so that shrink and expand are equally penalized. The y-axis prior uses a tighter σs,y since vertical layout is typically more stable. Hard bounds are enforced on (sx,sy).

The geometric tolerance is size-aware:

##### σi = σbase + α di + β min(wi,hi), (15)

where di = ∥cdemoi − tdemo∥2 is the distance from the context node to the target in the demonstration and (wi,hi) is the context node element size. This makes localization stricter for small UI elements and more tolerant for large ones, while also accounting for the fact that distant nodes have inherently less precise geometric constraints.

### C Precheck Pipeline

After executing an action, the environment typically needs time to settle (e.g., waiting for a page to load or an animation to finish). The precheck pipeline exploits this idle time by speculatively processing upcoming steps in a background thread, so that when the runner advances to the next step, a cached result may already be available.

The pipeline operates as follows. When the runner finishes step N, it submits the current observation to the pipeline and dispatches the action to the environment. While the environment executes, the pipeline processes step N +1 (and optionally step N +2) using the last available observation. When the environment returns a new observation, the runner calls collect() to retrieve any completed results. If the precheck result for step N+1 has sufficiently high confidence, the runner uses it directly, skipping redundant parsing and retrieval. Otherwise, the cached result is discarded and the step is processed normally with the fresh observation.

Runner PrecheckModule Environment StepActionHandler

submit(result, idx, obs) env.step(action)

- process(step N +1) background

- StepResult(N +1)

process(step N +2)

- StepResult(N +2)

observation

collect()

cached results

try_get(idx) StepResult (if confident)

- Figure 7: Precheck pipeline sequence diagram. While the environment executes the current action (gray bar), the pipeline speculatively processes upcoming steps in a background thread. The runner retrieves cached results via try_get, using them directly if confidence is sufficient.

### D Workflow Template Example

Below is an example workflow template generated from a single demonstration of drafting an email. The template contains metadata (name, title, description), parameterisable variables with placeholder defaults, and an ordered list of steps with natural-language action descriptions. Variables such as {{recipient_email}} are automatically extracted during the build phase and substituted at execution time.

workflow.yaml

- 1 workflow_id: ’20260313_110944_3’

- 2 workflow_name: draft_email

- 3 workflow_title: Draft Email

- 4 description: Compose and save a draft email with recipient address, subject line,

- 5 and message content

- 6 running_config:

- 7 variable_values:

- 8 recipient_email: email address

- 9 subject: email subject

- 10 email_content: email content

- 11 injected_steps: []

- 12 category: ’’

- 13 steps:

- 14 - step_number: 1

- 15 Action: Click on Mail icon

- 16 id: c2c927c9-2056-4935-b225-93275e9e655d

- 17 - step_number: 2

- 18 Action: Click on Compose button

- 19 id: 0559d739-6660-4ec6-b4c7-1493cac585bf

- 20 - step_number: 3

- 21 Action: ’Type text: {{recipient_email}}’

- 22 id: dddef290-3c3f-42a3-a266-5730582fe6f8

- 23 - step_number: 4

- 24 Action: ’Press key: tab’

- 25 id: 88f6250f-3f02-44e4-ac0b-099174fac6e8

- 26 - step_number: 5

- 27 Action: ’Press key: tab’

- 28 id: d6174e85-e256-4deb-bc12-f654d151b722

- 29 - step_number: 6

- 30 Action: ’Type text: {{subject}}’

- 31 id: d18f4cf2-1dd8-4a04-81df-9b8124880f57

- 32 - step_number: 7

- 33 Action: ’Press key: tab’

- 34 id: 8e6de124-c15e-45f3-bac7-1de89e0a1547

- 35 - step_number: 8

- 36 Action: ’Type text: {{email_content}}’

- 37 id: 30c9f2a6-54fb-4827-b197-7d82a2e65517

- 38 - step_number: 9

- 39 Action: ’Press hotkey: cmd + s’

- 40 id: b0c473e9-393e-47c7-93b5-da8adabcc55b

- 41 - step_number: 10

- 42 Action: Click on close button

- 43 id: bd0e41cc-216d-4ea4-88b0-aecf7c390121

The accompanying metadata file stores build-time configuration, extracted variable descriptions, and references to the step data used during execution:

metadata.json

- 1 {

- 2 "format_version": "2.0",

- 3 "workflow_id": "20260313_110944_3",

- 4 "config_file": "storage/workflows/.../workflow.yaml",

- 5 "data_file": "steps_data.json",

- 6 "step_count": 10,

- 7 "created_at": "2026-03-18T19:59:42.605361",

- 8 "workflow_metadata": {

- 9 "variables": [

- 10 {

- 11 "name": "recipient_email",

- 12 "default_value": "email address",

- 13 "description": "Email address of the recipient"

- 14 },

- 15 {

- 16 "name": "subject",

- 17 "default_value": "email subject",

- 18 "description": "Subject line of the email"

- 19 },

- 20 {

- 21 "name": "email_content",

- 22 "default_value": "email content",

- 23 "description": "Body content of the email message"

- 24 }

- 25 ],

- 26 "metadata": {

- 27 "scale_factor": 2.0,

- 28 "builder_config": {

- 29 "ui_parser_config": {

- 30 "text_feature": "sentence_e5",

- 31 "vision_feature": "iconclip",

- 32 "ui_extractors": [

- 33 "batch_omniparser",

- 34 "ocrmac_detector"

- 35 ]

- 36 },

- 37 "knn_k": 8

- 38 }

- 39 }

- 40 }

- 41 }

The referenced steps_data.json is keyed by step UUID and contains the per-step data used during execution. Its structure (with embeddings and lists abbreviated) is shown below:

steps_data.json (abbreviated)

- 1 {

- 2 "<step_uuid>": {

- 3 "step_id": "<uuid>",

- 4 "step_number": 1,

- 5 "action_type": "click",

- 6 "description": "Click on Mail icon",

- 7 "step_subgraph": {

- 8 "target_element_id": 7,

- 9 "image_size": [W, H],

- 10 "click_coordinates": [x, y],

- 11 "ui_graph": {

- 12 "G": {

- 13 "directed": false,

- 14 "nodes": [

- 15 {

- 16 "pos": [x, y, w, h],

- 17 "icon_emb": [/* 512-d vector */],

- 18 "text_emb": [/* 384-d vector or null */],

- 19 "attrs": {"type": "icon|text", "content": "..."},

- 20 "id": 0

- 21 }

- 22 /* ... more nodes ... */

- 23 ],

- 24 "links": [

- 25 {"source": 0, "target": 10}

- 26 /* ... kNN edges ... */

- 27 ]

- 28 }

- 29 },

- 30 "window_bounds": [x, y, w, h],

- 31 "knn_k": 8,

- 32 "scale_factor": 1.0,

- 33 "offset_x": -9.0,

- 34 "offset_y": -2.59

- 35 },

- 36 "variables": [],

- 37 "values": {},

- 38 "pause_duration": 1.73,

- 39 "active_app_name": "Dock"

- 40 }

- 41 /* ... one entry per step ... */

- 42 }

