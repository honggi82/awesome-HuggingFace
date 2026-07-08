### Learn2Fold: Structured Origami Generation with World Model Planning

Yanjia Huang1,2∗ Yunuo Chen1∗ Ying Jiang1∗ Jinru Han1 Zhengzhong Tu2 Yin Yang3 Chenfanfu Jiang1

## arXiv:2603.29585v2[cs.GR]2Apr2026

[Figure 1]

Figure 1. Teaser. From simple planes to complex articulated forms, Learn2Fold plans origami folding sequences that respect geometric constraints and anticipate future consequences, enabling robust generalization across unseen crease patterns.

#### Abstract

The ability to transform a flat sheet into a complex threedimensional structure is a fundamental test of physical intelligence. Unlike cloth manipulation, origami is governed by strict geometric axioms and hard kinematic constraints, where a single invalid crease or collision can invalidate the entire folding sequence. As a result, origami demands long-horizon constructive reasoning that jointly satisfies precise physical laws and high-level semantic intent. Existing approaches fall into two disjoint paradigms: optimization-based methods enforce physical validity but require dense, precisely specified inputs, making them unsuitable for sparse natural language descriptions, while generative foundation models excel at semantic and perceptual synthesis yet fail to produce long-horizon, physicsconsistent folding processes. Consequently, generating

* Equal contribution. 1 UCLA, 2 Texas A&M University, 3 University of Utah. yanjia_0812@tamu.edu, yunuoch@math.ucla.edu, anajymua@gmail.com, jinruhan1219@g.ucla.edu, tzz@ tamu.edu, yin.yang@utah.edu, cffjiang@ucla.edu

valid origami folding sequences directly from text remains an open challenge. To address this gap, we introduce Learn2Fold, a neuro-symbolic framework that formulates origami folding as conditional program induction over a crease-pattern graph. Our key insight is to decouple semantic proposal from physical verification. A large language model generates candidate folding programs from abstract text prompts, while a learned graph-structured world model serves as a differentiable surrogate simulator that predicts physical feasibility and failure modes before execution. Integrated within a lookahead planning loop, Learn2Fold enables robust generation of physically valid folding sequences for complex and out-of-distribution patterns, demonstrating that effective spatial intelligence arises from the synergy between symbolic reasoning and grounded physical simulation.

#### 1. Introduction

Recent advances in generative AI have enabled the synthesis of increasingly complex visual content, including im-

ages, videos, and 3D assets [6, 8, 22, 25, 27]. However, most of these successes focus on generating static or perceptual representations, where physical feasibility and execution constraints are either ignored or only weakly enforced. Extending generative models beyond visual plausibility toward physically executable processes remains an open and largely unexplored challenge. This challenge becomes particularly pronounced in tasks that require longhorizon reasoning under strict geometric and topological constraints. While recent progress in deformable object manipulation, such as cloth folding [19, 21, 23, 38, 41], has demonstrated impressive results, these settings benefit from the inherent compliance and error tolerance of amorphous materials. Garments can accommodate local inaccuracies through smoothing and deformation, allowing learning-based methods to recover from imprecise actions. In contrast, origami folding operates under a fundamentally different regime. Origami is the art of transforming a flat sheet into a three-dimensional structure through a sequence of folds, governed by strict geometric axioms and topological constraints [18, 26]. A single misplaced crease does not merely introduce a local artifact, but can violate surface topology or render all subsequent folding steps mathematically infeasible. As a result, origami demands precise coordination of discrete topological changes and continuous geometric motions over long horizons, with little tolerance for error.

In this work, we adopt origami folding as a challenging and principled testbed for studying constraint-aware generative planning. Digitally representing and generating origami processes requires modeling both a structured crease pattern and the progressive, constraint-driven folding dynamics that transform a flat sheet into a valid 3D shape. Despite its conceptual simplicity, origami exposes the core limitations of existing generative approaches and serves as a rigorous benchmark for evaluating long-horizon spatial reasoning under hard physical constraints.

Prior work on origami generation can be broadly categorized into learning based methods and optimization based approaches. Generative models, including large language models and vision language models [36, 39, 40, 46], are trained on large scale multimodal data such as origami videos, images, and textual instructions. These models can produce descriptive tutorials or high level folding guidance conditioned on text prompts or images. However, they typically fail to generate physically executable origami processes, as they optimize for approximate visual plausibility rather than exact physical feasibility, often hallucinating geometries that appear visually coherent but violate folding constraints. In contrast, traditional optimization based methods [12, 18, 35] formulate origami generation as a constrained optimization problem, employing techniques such as circle packing or tuck folding algorithms to mathemat-

ically guarantee that a target mesh can be folded from a single sheet. These approaches produce simulation ready, physically grounded crease patterns, but require precise 3D mesh inputs, making them difficult to apply to sparse inputs such as a single image or a text prompt. This raises a key question: can we retain the physical rigor and simulation ready representations of computational origami while leveraging the powerful priors of large language and vision language models to reconstruct executable origami processes from enriching semantic descriptions?

To bridge these gaps, we introduce Learn2Fold, a neuro-symbolic framework that formulates origami folding as constraint-aware program induction. Our key insight is that robust generation requires separating proposal from verification. Instead of blindly decoding a sequence, Learn2Fold operates in a propose, verify loop. We leverage a Large Language Model (LLM) to propose high-level structured action tokens, utilizing its semantic planning capabilities. However, acknowledging that LLMs lack intrinsic physics grounding, we integrate a learned GraphStructured World Model for lookahead planning. This world model acts as a differentiable surrogate simulator, allowing the system to imagine the geometric consequences of actions and prune branches that lead to invalid states before execution. We propose a symbolic simulator that performs final constraint verification, complementing neural proposal and learned lookahead with exact geometric feasibility checks.

Our contributions are summarized as follows:

- • We propose Learn2Fold, a novel framework for origami process generation that integrates a Large Language Model (LLM) for high-level structured action proposal with a learned Graph-Structured World Model for physics-aware lookahead planning and verification.
- • A scalable, simulation-driven data curation engine for origami that generates large-scale folding transitions using counterfactual perturbations and propose a new origami dataset, OrigamiCode dataset containing structured folding programs and verified transitions for learning origami folding dynamics.
- • We validate the effectiveness of the proposed method through comprehensive experiments, demonstrating robust generalization to out-of-distribution physically valid and executable origami generation.

#### 2. Related Work

##### 2.1. Structured and Constraint-Aware Generation

Recent generative models have demonstrated remarkable proficiency in synthesizing high fidelity assets, ranging from static 3D shapes [17, 20, 42, 43] to dynamic video sequences [4, 14, 30, 32]. However, modeling progressive shape formation processes like origami folding still

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

###### CP Graph State Card

Expert Steps

[Figure 6]

Llama

World Model

Perturbations

Model

[Figure 7]

[Figure 8]

###### Level0Sim

Exploration

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Training

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

# Learn2Fold

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

State Card

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Progress Angle Type

Goal

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

WM Rollouts

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

Airplane

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

Score

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

- 1. Pre-Crease [2, 3]
- 2. Pre-Crease [0, 1]
- 3. Fold [0, 3]
- 4. ……

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

Execute Select NoPath

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

No Idx

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

Inference

- Figure 2. Overview of Learn2Fold. Learn2Fold formulates origami folding as constraint-aware sequential program generation. During training, a symbolic Level-0 simulator enables scalable data generation and supervision for both a language-based proposal model and a learned world model. At inference time, Learn2Fold combines LM proposals with world-model rollouts and MPC to robustly plan folding sequences under hard constraints.

remains an open challenge. Unlike one-shot generation methods that directly predict a final geometry, origami folding is intrinsically an executable, long-horizon action sequence. This process operates on a complex hybrid discrete-continuous state space. This task involves discrete topological changes such as face layering, connectivity updates, coupled with continuous kinematic transformations. Crucially, this generation process is governed by strict physical validity. Every folding step must satisfy hard geometric and topological constraints, such as flat-foldability and self-intersection avoidance; a minor violation in early steps compounds, rendering the final result physically invalid. Consequently, this setting demands structured generation paradigms rather than unstructured end-to-end inference. To address similar structural challenges, recent works have adopted intermediate representations, such as scene graphs or layouts [16, 24, 44], to anchor object relations and reduce spurious outputs. Another line of research integrates constraint-aware decoding or verifier-guided search to ensure validity [2, 28, 45]. For instance, recent structural synthesis models like BrickGPT [28] rely on reactive rollback to filter out physically unstable steps. While these assembly generation systems effectively combine auto-regressive proposals with rollback mechanisms, straightforward back-

tracking becomes computationally prohibitive for complex folding sequences. Distinguishing our work from these approaches, we propose a CP-grounded folding program equipped with diagnostic feedback. Instead of binary success or failure checks, our model performs causal attribution to identify why a fold failed, enabling efficient planning and recovery even on out-of-distribution crease patterns.

##### 2.2. Computational Origami

Origami folding is fundamentally governed by rigorous mathematical rules concerning develop ability and flatfoldability like Kawasaki’s and Maekawa’s theorems[3, 7, 15]. To simulate these complex behaviors computationally, researchers have developed kinematic models that treat creases as rotational hinges. Early works focused on rigid origami, modeling the mesh as discrete rigid facets connected by joints [34, 35]. To alleviate this, more recent approaches, such as the bar-and-hinge model used in Origami Simulator [9], introduce compliance to approximate the elastic deformation of paper, enabling real-time folding visualization. While these simulators provide ground truth physics, they are purely forward-process tools where they calculate the geometric consequence of a given fold but do not posses the agency to plan a sequence or reason about

hight-level semantic goals.

The problem of generation a crease patter (CP) for a target 3D shape has traditionally been formulated as a geometric optimization problem. Pioneering systems like TreeMaker [18] and Origamizer [34] use circle packing or tuck-folding algorithms to mathematically guarantee that a specific mesh can be folded from a single sheet. However, these methods are strictly geometry-centric and deterministic. They lack the flexibility to handle ambiguous semantic descriptions and are often sensitive to topological errors, where a slight violation in the CP graph renders the entire optimization infeasible. Unlike these optimization-based solvers which require a perfect final mesh as input, our approach treats generation as a sequential decision-making process. This allows for robust recovery from intermediate errors and generalization to out-of-distribution patterns via previous trained structures.

##### 2.3. World Models

World models learn action-conditioned dynamics to enable planning via imagined rollouts. This paradigm spans from classical latent-dynamics methods in model-based RL [10, 11, 29] to recent foundation-scale video simulators that model physics in rich visual domains [4, 14, 31]. However, pixel-based or latent world models do not directly enforce hard discrete geometric constraints, nor do they naturally produce structured, executable programs. Furthermore, collecting action-labeled interaction data for specialized domains like origami remains prohibitive [1]. In our work, we learn a state-level world model over CP-graph states, supervised by scalable synthetic transitions from a deterministic constraint engine. Crucially, our training data includes near-boundary perturbations, exposing the model to both feasible and infeasible outcomes. This learned dynamics model enables efficient model-predictive lookahead, allowing the system to verify action feasibility and recover from proposal errors on out-of-distribution crease patterns.

[Figure 247]

- Figure 3. Deriving Expert Trajectories from Videos. We show one data source for obtaining expert folding trajectories. In-thewild instructional videos are processed into State Cards and folding steps, which are then augmented through perturbation and exploration for training.

#### 3. Method

We target physically valid generation for Computational Origami: at inference time, our agent augments its base proposal policy with a graph-based world model that imagines future manifold states and converts them into validity scores for planning (Fig. 2). Our approach, Learn2Fold, tightly couples three components: ❶ a Canonicalized Graph Representation that ensures structural invariance; ❷ a Generative Proposal Policy that suggests candidate folds based on semantic goals; and ❸ a Graph-based World Model that rolls out short-horizon geometric futures. At test time, we do not only rely on the policy’s likelihood; instead, the world model’s predictions are fused at the score level via model predictive control (MPC) to rank candidate actions, ensuring strict geometric feasibility without sacrificing generative flexibility.

In the following sections, Sec. 3.1 details the canonicalized state representation. Sec. 3.2 formalizes the languageconditioned proposal policy. Sec. 3.3 introduces the graph world model, which acts as a differentiable surrogate simulator. Finally, Sec. 3.4 describes the MPC planning strategy that integrates these signals for robust action selection.

##### 3.1. State Representation and Canonicalization

We formulate the origami folding process as a sequential manipulation of a graph-structured manifold. An origami instance is represented by a tuple Ot = (G,st), where G denotes its static topology and st denotes a dynamic state.

Static Graph Topology. The crease pattern (CP) is a planar graph G = (V,E) with points V = {vi ∈ [0,1]2}N

v

i=1 and edges E = {ej}N

j=1. Each edge may carry an initial

e

crease type label z0j ∈ {M, V, U} (M: mountain, V: valley, U: unknown).

Canonicalization. Raw CP data often contains arbitrary vertex indexing, which hinders learning. To ensure permutation invariance and robust generalization, we apply a deterministic canonicalization process Φ : G → G∗. Specifically, we (i) reindex vertices via lexicographical sorting of coordinates, and (ii) reindex edges based on the sorted endpoint indices. To further eliminate orientation bias, we augment the training data by applying dihedral symmetries (rotations and reflections) to V prior to canonicalization. This ensures that structurally identical patterns map to the same index space.

Dynamic State. We track the folding status using a state vector st = αt,ρt,zt,ψt,bt,t , where αt ∈ [−π,π]|E| are signed dihedral angles, ρt ∈ [0,1]|E| are progress ratios, zt ∈ {M, V, U}|E| are crease types, ψt is the global frame angle, bt is the MV-flip flag, and t is the step counter.

##### 3.2. Policy Learning via Language Models

We frame origami folding as a conditional program induction task. The goal is to learn a policy πθ(at | Ct) that generates a valid folding operation at given the current context Ct.

Unified Token Space. The action space of folding is inherently hybrid, requiring the selection of discrete graph elements (e.g., target edges) and continuous parameters (e.g., fold angles). To leverage the reasoning capabilities of Transformer-based LLMs, we unify these modalities into a homogeneous vocabulary Σ = Σops∪Σgraph∪Σgeo. Continuous geometric parameters are quantized into discrete bins Σgeo, while canonicalized graph indices are mapped to semantic tokens Σgraph. This formulation transforms the complex control problem into an autoregressive sequence modeling task, enabling the model to capture joint dependencies between topological intent and geometric specifications.

Context and Objective. The policy is conditioned on a context Ct = (g;G∗,st), where g denotes the high-level semantic goal. By operating on the canonicalized graph G∗, the policy learns structure-invariant folding motifs (e.g., “rabbit-ear fold”) rather than overfitting to instance-specific identifiers (e.g., vertex indices). We train the model using Maximum Likelihood Estimation (MLE) on expert demonstrations D:

Lpolicy(θ) = E(C,a∗)∼D −

log πθ(at,k | Ct,at,<k) ,

k

(1) where at,k denotes the k-th token of the action sequence at step t. This supervised pre-training instills the grammar of valid folding operations.

##### 3.3. Graph-Based World Model

While the policy proposes plausible actions, ensuring strict physical feasibility requires rigorous verification. To enable efficient lookahead planning without computationally expensive mesh-based simulations, we learn a differentiable world model Mϕ that acts as a surrogate simulator.

Residual Graph Dynamics. Unlike pixel-based world models [5] which lack explicit geometric constraints, our model operates directly on the graph state st. We formulate the transition as a sparse residual update:

∆ˆst, mˆ t, cˆt+1 = Mϕ(G∗,st,at), sˆt+1 = st + ∆ˆst ⊙ expand( ˆmt),

(2)

where mˆ t ∈ [0,1]|E| is a locality mask and cˆt+1 ∈ [0,1]|E| estimates per-edge constraint violation likelihood.

expand(·) broadcasts the per-edge mask to all state channels.

##### 3.4. Inference via Graph-Guided MPC

At test time, we perform a constrained lookahead search on the CP graph G = (V,E). At each step t, our proposal policy πθ generates candidate structured actions, which are filtered by a hard verifier (Level-0 simulator) and ranked by the learned world model.

Candidate Sampling. We sample K candidate actions from the proposal distribution using nucleus sampling:

###### At = {a(tk)}Kk=1, a(tk) ∼ πθ(· | Ct). (3)

Hard Verification (Level-0). Each candidate is first evaluated by a deterministic constraint kernel:

(˜s(t+1k) ,vt(k),rt(k),m(tk)) = LEVEL0SIM(G∗,st,a(tk)), (4)

where vt(k) ∈ {0,1} indicates fold validity, rt(k) denotes the reason for invalidity, and m(tk) ∈ {0,1}|E| is the affected-edge mask. We discard invalid candidates and retain Avalidt = {a(tk) ∈ At | vt(k) = 1}.

World-Model Rollout. For each valid candidate, the world model predicts residual state updates and a soft violation mask:

∆ˆs(tk), cˆ(t+1k) = Mϕ(G∗,st,a(tk)), sˆ(t+1k) = st + ∆ˆs(tk), (5)

where cˆ(t+1k) ∈ [0,1]|E| estimates per-edge constraint violation likelihood (a soft counterpart of mt).

Action Selection. We choose the action maximizing a fused objective of proposal likelihood, goal progress, and feasibility:

1 |a(tk)|

log πθ(a(tk) | Ct)

a∗t = arg max

a(tk)∈Avalidt

− λgoal Ugoal(ˆs(t+1k) ) + λcst log ϵ + 1 − ∥cˆ(t+1k) ∥∞ .

(6) Here λgoal,λcst balance goal pursuit and constraint satis-

faction, and ϵ > 0 avoids numerical instability.

Failure and Re-sampling. In the case when Avalidt = ∅ or maxk J(k) < τ, we construct a negative constraint from the predicted violation mask (e.g., top-M edges with highest cˆ) and re-sample candidates under the updated constraint set.

[Figure 248]

- Figure 4. Overview of the OrigamiCode Benchmark Dataset. The dataset features diverse categories, structured representations, sequential folding data, detailed statistics, and benchmark downstream tasks.

[Figure 249]

[Figure 250]

[Figure 251]

##### 3.5. Dataset Construction

To support learning structured origami folding behaviors, we construct the OrigamiCode dataset (Fig. 4), a large-scale collection of procedural folding sequences derived from canonical crease pattern (CP) representations. The dataset covers 25 common origami object classes, each defined by a parameterized CP graph (Fig. 5) and an associated folding specification stored in a structured JSON format. Unlike previous datasets dominated by simple shapes, our benchmark is carefully stratified into three difficulty tiers based on step count and non-local dependency: Simple (10 categories): Basic rigid folding structures with minimal layering (e.g., Airplanes, Hearts, Cups). Intermediate (10 categories): Standard models requiring moderate spatial planning and box-pleating (e.g., Boats, Flowers). Complex (5 categories): High-frequency folding sequences with intricate appendage management and strict circle-packing constraints (e.g., Cranes, Dragons).

Butterfly CP graph Airplane CP graph Hat CP graph

Figure 5. Crease Pattern (CP) Graph. We created CP graph for each case to represent folding sequence.

sion for learning fold prediction, sequence generation, and origami manipulation tasks.

#### 4. Experiments

##### 4.1. Experiment Setup

Implementation Details We train the world model (WM) using large-scale synthetic folding data generated by the Level-0 simulator. Specifically, we collect approximately 76,000 transitions through expert demonstrations and constraint-guided perturbations, and train the WM with supervised learning for 50 epochs, which takes about 30 hours on a single NVIDIA RTX Pro 6000 GPU. The language model (LM) is a lightweight decoder-only transformer fine-tuned to generate structured folding actions under a fixed JSON schema. It is trained using roughly 104 expert folding steps augmented with simulator-verified perturbations, and converges within 6 hours using LoRA adapters on the same hardware. At inference time, Learn2Fold runs in a model predictive control (MPC) loop, where the LM proposes N = 8 candidate actions per step, the simulator

The CP graph encodes the planar crease topology, including vertex connectivity, edge types (mountain or valley), and boundary constraints, while the JSON specification defines the ordered folding operations and geometric parameters required to generate valid intermediate states. Our data curation pipeline derives expert trajectories from instructional sources and further augments them through perturbation and exploration. This process produces a large set of physically consistent folding trajectories, where each sequence records the evolving mesh geometry, crease states, and fold parameters at every step. In total, we collect 5,760 origami process sequences and 75,000 trajectories in the OrigamiCode dataset, which provides structured supervi-

[Figure 252]

[Figure 253]

###### Figure 6. Learn2Fold results.

[Figure 254]

- Figure 7. Folding with Reasoning. Learn2Fold incrementally constructs origami folding programs in CP-graph space. At each step, multiple candidate actions are evaluated through worldmodel rollouts, infeasible options are discarded, and the best action is selected for execution, enabling robust folding and recovery under hard constraint

filters invalid ones, and the WM scores the remaining candidates via short-horizon rollouts to select the final action. All experiments are conducted with fixed random seeds for reproducibility.

Dataset. To rigorously evaluate topological generalization, we curate a held-out OrigamiCode benchmark dataset of 25 distinct origami categories that span the full spectrum of folding complexity. This taxonomy allows us to disentangle basic instruction following from complex physical

reasoning. Each instance provides a canonicalized CP and a ground-truth program. Following a standard train–test split, 80% of the data is used for training, while the remaining 20% is reserved for evaluation.

Baselines. We compare our approach against three representative methods. First, we evaluate BrickGPT [28], a reactive baseline adapted from assembly synthesis that employs a physics-aware rollback mechanism to filter unstable steps through trial-and-error execution and is trained on the proposed OrigamiCode dataset. Second, we benchmark against GPT-5.1 and GPT-5.2, the latest state-ofthe-art foundation models. These general-purpose agents are provided with in-context examples to output structured folding programs, representing the upper bound of unconstrained semantic planning without specialized geometric modules. Finally, we compare these against Learn2Fold (Ours), which generates actions under explicit graph-based lookahead verification.

Metrics. We evaluate performance at both the step level and the trajectory level. At the step level, we report Precision, Recall, and F1 to measure how accurately each method predicts structured folding actions under a unified action schema, capturing both the correctness and coverage of discrete decisions. At the trajectory level, we report Category Success Rate (Cat-SR) and Edge-IoU to evaluate long-horizon execution performance and structural alignment, respectively. Cat-SR is defined as the fraction of folding sequences that successfully complete the target origami within each category and is macro-averaged across categories to mitigate class imbalance. Edge-IoU measures whether a predicted action affects the correct set of creases by computing the intersection-over-union between the predicted affected-edge set and the simulator-derived ground truth.

##### 4.2. Quantitative Evaluation

In the absence of standard benchmarks for origami process generation, following [28], we construct a custom test set comprising 3,840 text prompts spanning 25 categories. From this set, we select 1,150 cases for validation and perform two independent runs per prompt for each method, yielding 7,680 results per method. As shown in Table 1, our method outperforms all baselines across all metrics in both step-level accuracy and trajectory-level success. At the step level, Learn2Fold achieves a Precisionµ/Recallµ/F1µ of 0.766/0.711/0.739, substantially exceeding the strongest baseline (GPT-5.1, F1µ = 0.266), corresponding to a +47.3 point absolute improvement in F1. In contrast, LLM-based baselines exhibit a pronounced precision–recall imbalance. For example, GPT-5.2 achieves a relatively high Recallµ (0.358) but very low Precisionµ (0.124), suggesting that

[Figure 255]

- Figure 8. Qualitative comparison of folding behaviors across methods. Learn2Fold produces concise, physically feasible folding trajectories on both simple and complex origami tasks. Baseline methods frequently fail due to invalid actions, early termination, or inability to recover from long-horizon errors, especially on complex crease patterns.

- Table 1. Main comparison across methods. ∗ indicates prompted models, and † indicates finetuned models.

Method Prec. ↑ Rec. ↑ F1 ↑ Edge-IoU ↑ Cat-SR ↑ Gemini* 0.2874 0.4213 0.3420 0.1126 0.4942

- GPT-5.1* 0.2625 0.2996 0.2663 0.0937 0.6753

- GPT-5.2* 0.1243 0.3575 0.1648 0.1322 0.1600 BrickGPT† 0.3969 0.2250 0.2461 0.0505 0.5455 Ours 0.7661 0.7113 0.7394 0.5820 0.8912

while many relevant actions are proposed, they are often imprecise or misaligned with the required structural context since LLMs operate at a coarse semantic level, lacking detailed visual guidance. As a result, they can outline plausible folding intentions but cannot resolve the fine-grained, step-specific details. As for BrickGPT, while it benefits from explicit rollback-based execution and achieves higher precision than LLM-based baselines, it still suffers from limited recall, indicating that its reactive trial-and-error strategy produces coarse and incomplete folding actions and fails to consistently recover the full sequence of required steps.

##### 4.3. Qualitative Study

Fig. 8 presents qualitative comparisons between Learn2Fold and baseline methods on representative examples from the same test set described in Sec. 4.2. LLM-based baselines typically fail after only a few steps. While the initial actions are often semantically plausible, errors quickly accumulate due to the lack of explicit geometric state tracking. As a result, many predicted

steps are either structurally incorrect or misaligned with the underlying crease pattern, causing premature termination of the folding process. BrickGPT exhibits improved stability in the early stages of execution. In several examples, the first few predicted actions (e.g., the first three to four steps) are valid and physically feasible, benefiting from its rollback-based mechanism. However, as folding sequences grow longer, BrickGPT struggles to maintain long-term consistency, as it fails to capture fine-grained dependencies between distant steps, resulting in incorrect or incomplete long-horizon folding processes and limiting its ability to represent detailed step-by-step origami procedures for complex models. While Learn2Fold consistently produces coherent and fine-grained folding sequences across the entire trajectory. By explicitly modeling the folding state and verifying feasibility at each step, Learn2Fold maintains structural consistency and accurately captures the intended origami process, even for long and intricate folding sequences. These qualitative results corroborate the quantitative findings and highlight the advantage of explicit state modeling for reliable origami process generation.

##### 4.4. Ablation

We conduct an ablation study of the key components of our framework to evaluate the contribution of each proposed component to the final origami process generation’s performance. We progressively ablate the system by comparing three configurations: (i) an LLM-only proposer, (ii) LLM augmented with a learned world model (LM+WM), and (iii) the full system Learn2Fold that further incorporates the Level0Sim constraint kernel (LM+WM+Level0Sim). These variants are evaluated under both in-distribution (IID)

- Table 2. Ablations on IID (top) and OOD (bottom). Blue and teal indicate the best and second-best results.

Method Step Valid ↑ Traj SR ↑ Goal Dist ↓

LM 70.8% ± 45.5% 22.2% ± 45.5% 0.796 ± 0.194 LM+WM 54.2% ± 49.8% 25.0% ± 43.3% 0.759 ± 0.214 Full (Ours) 64.2% ± 41.8% 33.3% ± 47.1% 0.855 ± 0.196

Method Step Valid ↑ Traj SR ↑ Goal Dist ↓

LM 47.6% ± 29.2% 20.7% ± 55.5% 0.633 ± 0.192 LM+WM 32.3% ± 28.7% 17.8% ± 51.7% 0.560 ± 0.248 Full (Ours) 41.2% ± 32.3% 27.7% ± 50.1% 0.487 ± 0.353

and out-of-distribution (OOD) CP holdout settings using step-level validity, trajectory success rate (Traj SR), and final goal distance. Incorporating the world model alters the decision-making behavior by introducing short-horizon lookahead. Compared to the LLM-only baseline, LM+WM exhibits a modest improvement in trajectory-level success (IID Traj SR: 22.2% → 25.0%), and consistently reduces the final goal distance in both IID (0.796 → 0.759) and OOD settings (0.633 → 0.560). However, this improvement comes with a decrease in step-level validity, indicating that the world model prioritizes global progress over local action safety, occasionally selecting actions that are locally risky but potentially beneficial for long-horizon objectives. Adding Level0Sim consistently improves long-horizon performance. The full system achieves the highest trajectory success in both IID and OOD settings, while recovering step-level validity and further reducing final goal distance under distribution shift. Overall, the ablation indicates complementary roles of the LLM proposer, the world model, and Level0Sim, whose combination is required for robust long-horizon folding.

#### 5. Conclusion

In this work, we present Learn2Fold, a neuro-symbolic framework for physically valid origami process generation that unifies semantic reasoning with rigorous geometric constraint enforcement. By formulating origami folding as a constraint-aware program induction over a CP graph, Learn2Fold addresses a fundamental limitation of prior generative models: the inability to reliably generate long-horizon, executable action sequences under strict topological and kinematic constraints. We view origami not merely as an application, but as a principled testbed for future spatial reasoning systems, where it exposes core challenges in reasoning over structured space: discrete topological decisions coupled with continuous geometry, irreversible constraints, and long-horizon dependency.

#### References

- [1] Bo Ai, Stephen Tian, Haochen Shi, Yixuan Wang, Tobias Pfaff, Cheston Tan, Henrik I. Christensen, Hao Su, Jiajun Wu, and Yunzhu Li. A review of learning-based dynamics models for robotic manipulation. Science Robotics, 10(106): eadt1497, 2025. 4
- [2] Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. Guided open vocabulary image captioning with constrained beam search. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 936–945, Copenhagen, Denmark, 2017. Association for Computational Linguistics. 3
- [3] Marshall Bern and Barry Hayes. The complexity of flat origami. In Proceedings of the Seventh Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 175–183. Society for Industrial and Applied Mathematics, 1996. 3
- [4] Jake Bruce, Michael Dennis, Ashley Edwards, Jack ParkerHolder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, Yusuf Aytar, Sarah Bechtle, Feryal Behbahani, Stephanie Chan, Nicolas Heess, Lucy Gonzalez, Simon Osindero, Sherjil Ozair, Scott Reed, Jingwei Zhang, Konrad Zolna, Jeff Clune, Nando de Freitas, Satinder Singh, and Tim Rockt¨aschel. Genie: Generative interactive environments, 2024. 2, 4
- [5] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Forty-first International Conference on Machine Learning, 2024. 5
- [6] Zhaoxi Chen, Guangcong Wang, and Ziwei Liu. Scenedreamer: Unbounded 3d scene generation from 2d image collections. IEEE transactions on pattern analysis and machine intelligence, 45(12):15562–15576, 2023. 2
- [7] Erik D Demaine and Joseph O’Rourke. Geometric Folding Algorithms: Linkages, Origami, Polyhedra. Cambridge University Press, 2007. 3
- [8] Jun Gao, Tianchang Shen, Zian Wang, Wenzheng Chen, Kangxue Yin, Daiqing Li, Or Litany, Zan Gojcic, and Sanja Fidler. Get3d: A generative model of high quality 3d textured shapes learned from images. Advances in neural information processing systems, 35:31841–31854, 2022. 2
- [9] Amanda Ghassaei, Erik D Demaine, and Neil Gershenfeld. Fast, interactive origami simulation using gpu compute shaders. In Proceedings of the 7th International Meeting on Origami in Science, Mathematics and Education (OSME7), pages 1151–1166, 2018. 3
- [10] Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels, 2019. 4
- [11] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. In International Conference on Learning Representations (ICLR), 2020. 4
- [12] Can He, Lingxiao Meng, Zhirui Sun, Jiankun Wang, and Max Q. H. Meng. Fabricfolding: Learning efficient fabric folding without expert demonstrations, 2023. 2

- [13] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. Iclr, 1(2):3, 2022. 12
- [14] Yanjia Huang, Xianshun Jiang, Xiangbo Gao, Mingyang Wu, and Zhengzhong Tu. Vistav2: World imagination for indoor vision-and-language navigation, 2025. 2, 4
- [15] Thomas C. Hull. The combinatorics of flat folds: A survey. In Origami3: Third International Meeting of Origami Science, Mathematics, and Education, pages 29–38. A K Peters, 2002. 3
- [16] Justin Johnson, Agrim Gupta, and Li Fei-Fei. Image generation from scene graphs, 2018. 3
- [17] Yushi Lan, Fangzhou Hong, Shangchen Zhou, Shuai Yang, Xuyi Meng, Yongwei Chen, Zhaoyang Lyu, Bo Dai, Xingang Pan, and Chen Change Loy. Ln3diff++: Scalable latent neural fields diffusion for speedy 3d generation. IEEE Transactions on Pattern Analysis and Machine Intelligence, page 1–18, 2025. 2
- [18] Robert J Lang. Origami Design Secrets: Mathematical Methods for an Ancient Art. CRC Press, 2 edition, 2011. 2, 4
- [19] Robert Lee, Jad Abou-Chakra, Fangyi Zhang, and Peter Corke. Learning fabric manipulation in the real world with human videos. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 3124–3130. IEEE,

2024. 2

- [20] Weiyu Li, Xuanyang Zhang, Zheng Sun, Di Qi, Hao Li, Wei Cheng, Weiwei Cai, Shihao Wu, Jiarui Liu, Zihao Wang, Xiao Chen, Feipeng Tian, Jianxiong Pan, Zeming Li, Gang Yu, Xiangyu Zhang, Daxin Jiang, and Ping Tan. Step1x-3d: Towards high-fidelity and controllable generation of textured 3d assets, 2025. 2
- [21] Yinxiao Li, Yonghao Yue, Danfei Xu, Eitan Grinspun, and Peter K Allen. Folding deformable objects using predictive simulation and trajectory optimization. In 2015 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 6000–6006. IEEE, 2015. 2
- [22] Yangguang Li, Zi-Xin Zou, Zexiang Liu, Dehu Wang, Yuan Liang, Zhipeng Yu, Xingchao Liu, Yuan-Chen Guo, Ding Liang, Wanli Ouyang, et al. Triposg: High-fidelity 3d shape synthesis using large-scale rectified flow models. arXiv preprint arXiv:2502.06608, 2025. 2
- [23] Yiming Liu, Lijun Han, Enlin Gu, and Hesheng Wang. Learning a general model: Folding clothing with topological dynamics. arXiv preprint arXiv:2504.20720, 2025. 2
- [24] Yunlong Liu, Shuyang Li, Pengyuan Liu, Yu Zhang, and Rudi Stouffs. From pixels to predicates structuring urban perception with scene graphs. arXiv preprint arXiv:2512.19221, 2025. 3
- [25] Yuanxun Lu, Jingyang Zhang, Shiwei Li, Tian Fang, David McKinnon, Yanghai Tsin, Long Quan, Xun Cao, and Yao Yao. Direct2. 5: Diverse text-to-3d generation via multi-view 2.5 d diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8744– 8753, 2024. 2
- [26] Jeremy Maitin-Shepard, Marco Cusumano-Towner, Jinna Lei, and Pieter Abbeel. Cloth grasp point detection based

- on multiple-view geometric cues with application to robotic towel folding. In 2010 IEEE International Conference on Robotics and Automation, pages 2308–2315, 2010. 2
- [27] Gimin Nam, Mariem Khlifi, Andrew Rodriguez, Alberto Tono, Linqi Zhou, and Paul Guerrero. 3d-ldm: Neural implicit 3d shape generation with latent diffusion models. arXiv preprint arXiv:2212.00842, 2022. 2
- [28] Ava Pun, Kangle Deng, Ruixuan Liu, Deva Ramanan, Changliu Liu, and Jun-Yan Zhu. Generating physically stable and buildable brick structures from text. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14798–14809, 2025. 3, 7, 12
- [29] Rafael Rafailov, Tianhe Yu, Aravind Rajeswaran, and Chelsea Finn. Offline reinforcement learning from images with latent space models, 2020. 4
- [30] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents, 2022. 2
- [31] Marc Rigter, Tarun Gupta, Agrin Hilmkil, and Chao Ma. Avid: Adapting video diffusion models to world models,

2024. 4

- [32] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models, 2022. 2
- [33] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025. 12
- [34] Tomohiro Tachi. Simulation of rigid origami. In Origami4: Fourth International Meeting of Origami Science, Mathematics, and Education, pages 175–187. AK Peters/CRC Press, 2009. 3, 4
- [35] Tomohiro Tachi. Freeform variations of origami. Journal for Geometry and Graphics, 14(2):203–215, 2010. 2, 3
- [36] ByteDance Seed Team. Rt-2: Vision-language-action models transfer web knowledge to robotic control, 2023. 2
- [37] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 12
- [38] Gemini Robotics Team. Gemini robotics: Bringing ai into the physical world, 2025. 2
- [39] OpenAI Team. Gpt-4 technical report, 2024. 2
- [40] Qwen Team. Qwen3 technical report, 2025. 2
- [41] Tongxuan Tian, Haoyang Li, Bo Ai, Xiaodi Yuan, Zhiao Huang, and Hao Su. Diffusion dynamics models with generative state estimation for cloth manipulation. arXiv preprint arXiv:2503.11999, 2025. 2
- [42] Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts, David Pankratz, Dmitry Tochilkin, Christian Laforte, Robin Rombach, and Varun Jampani. Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion. In European Conference on Computer Vision, pages 439–457. Springer, 2024. 2
- [43] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and

- diverse text-to-3d generation with variational score distillation. Advances in neural information processing systems, 36: 8406–8441, 2023. 2
- [44] Danfei Xu, Yuke Zhu, Christopher B Choy, and Li Fei-Fei. Scene graph generation by iterative message passing. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5410–5419, 2017. 3
- [45] Kun Yan, Lei Ji, Huaishao Luo, Ming Zhou, Nan Duan, and Shuai Ma. Control image captioning spatially and temporally. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 2014–2025, Online, 2021. Association for Computational Linguistics. 3
- [46] Jingyi Zhang, Jiaxing Huang, Sheng Jin, and Shijian Lu. Vision-language models for vision tasks: A survey, 2024. 2

### Appendix

#### A. More Model and Implementation Details

Learn2Fold is implemented as a neuro-symbolic planning system over a canonicalized crease-pattern graph representation. Given a high-level semantic goal and the current origami state, the system maintains a unified graph-state interface that is shared across data generation, model training, and inference-time planning. Concretely, each instance is represented by a canonicalized crease-pattern graph together with its dynamic folding state, so that structurally equivalent patterns can be processed in a consistent index space. On top of this representation, we train a lightweight decoder-only language model to generate structured folding actions under a fixed JSON schema, which serves as the proposal policy during inference. To ground the proposal process in physical feasibility, we additionally train a graph-structured world model on large-scale synthetic folding transitions produced by the symbolic Level-0 simulator. The world model is optimized with supervised learning to predict short-horizon state evolution and soft constraintviolation signals, allowing the planner to estimate future failure modes before execution. In our implementation, the world model is trained on approximately 76,000 transitions collected from expert demonstrations and constraint-guided perturbations for 50 epochs, which takes about 30 hours on a single NVIDIA RTX Pro 6000 GPU. The language model is trained on roughly 104 expert folding steps augmented with simulator-verified perturbations, and converges within 6 hours on the same hardware using LoRA adapters [13]. At inference time, Learn2Fold operates in a model predictive control loop: at each step, the language model proposes N = 8 candidate actions, the Level-0 simulator deterministically filters out invalid candidates, and the world model performs short-horizon lookahead scoring over the remaining valid actions to select the final execution step. All experiments are conducted with fixed random seeds for reproducibility.

#### B. Evaluation Protocol

This section provides additional details on the evaluation setup used in the main paper. To assess both structured action prediction and long-horizon folding reliability, we evaluate all methods on a held-out benchmark constructed from 25 origami categories spanning three difficulty tiers: simple, intermediate, and complex. As described in the main paper, the benchmark is designed to cover a broad range of folding complexity, from basic rigid-folding patterns with minimal layering to long-horizon sequences involving nonlocal dependencies and strict structural constraints. Following the dataset split used throughout the paper, 80% of the OrigamiCode data is used for training and the remaining

20% is reserved for evaluation. Each evaluation instance consists of a canonicalized crease pattern together with its target semantic goal and reference folding program. The final test suite contains 3,840 text prompts spanning the 25 categories, from which we select 1,150 cases for validation. For each evaluated method, we perform two independent runs per prompt in order to reduce variance due to stochastic decoding and obtain a more stable estimate of long-horizon generation performance.

We compare Learn2Fold against two classes of baselines. The first class consists of prompted foundation models, including GPT-5.1, GPT-5.2 [33], and Gemini [37], which are prompted to generate structured folding programs from the same held-out benchmark inputs. These baselines are provided with in-context demonstrations and are evaluated under the same structured output interface as our method whenever possible, so that their predictions can be parsed by the same evaluator. The second class is BrickGPT [28], a finetuned reactive baseline adapted from assembly synthesis, which uses rollback-based trial-and-error execution to reject unstable steps and is trained on the same OrigamiCode dataset as described in the main paper. This evaluation design allows us to compare semantic-only proposal systems, reactive execution-based baselines, and our full graph-guided planning system under a common benchmark protocol.

We report both step-level and trajectory-level metrics. At the step level, we measure Precision, Recall, and F1 on structured action tokens under a unified action schema. These metrics quantify how accurately a method predicts the discrete content of each folding action, including operation type and associated structured arguments, while also capturing coverage over the full program. In the main paper, these are reported as the evaluator’s category-averaged micro scores. At the trajectory level, we report Category Success Rate (Cat-SR) and Edge-IoU. Cat-SR is defined as the fraction of folding sequences that successfully complete the target origami within each category and is then macroaveraged across categories to mitigate class imbalance. Edge-IoU measures structural alignment between predicted and reference actions by computing the intersection-overunion between the affected-edge set induced by the predicted action and the simulator-derived ground-truth mask. Together, these metrics distinguish local action correctness from global execution success and allow us to separately assess symbolic fidelity, structural grounding, and longhorizon robustness.

For Learn2Fold, evaluation follows the same inference procedure described in the main paper. At each step, the language model proposes multiple candidate actions, the Level-0 simulator discards invalid candidates, and the learned world model scores the remaining valid actions through short-horizon lookahead before one action is se-

lected for execution. A trajectory is rolled out autoregressively until the program terminates or the method fails to produce a valid continuation. For the ablation experiments, we use the same held-out protocol and compare three configurations: LM only, LM+WM, and the full LM+WM+Level0Sim system. These variants are evaluated under both IID and OOD CP holdout settings using step validity, trajectory success rate, and final goal distance, following the setup described in Sec. 4.4 of the main paper. This shared protocol ensures that gains from the world model and the symbolic verifier are measured under the same benchmark conditions rather than under separate task definitions.

Unless otherwise noted, all experiments are conducted with fixed random seeds. The same canonicalized CP representation and unified action schema are used for training, decoding, parsing, and evaluation, which avoids representation mismatch across methods and allows all predictions to be assessed under a common symbolic interface. In the supplementary material, we additionally provide more qualitative examples, representative failure cases, and example structured outputs to further clarify how the above protocol is instantiated in practice.

