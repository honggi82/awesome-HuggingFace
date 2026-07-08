## LoopViT: Scaling Visual ARC with Looped Transformers

Wen-Jie Shu1 Xuerui Qiu2 Rui-Jie Zhu3 Harold Haodong Chen1 Yexin Liu1 Harry Yang1 1HKUST 2CASIA 3UC Santa Cruz wenjieshu2003@gmail.com

# arXiv:2602.02156v1[cs.CV]2Feb2026

### Abstract

Recent advances in visual reasoning have leveraged vision transformers to tackle the ARC-AGI benchmark. However, we argue that the feed-forward architecture, where computational depth is strictly bound to parameter size, falls short of capturing the iterative, algorithmic nature of human induction. In this work, we propose a recursive architecture called Loop-ViT, which decouples reasoning depth from model capacity through weight-tied recurrence. Loop-ViT iterates a weight-tied Hybrid Block, combining local convolutions and global attention, to form a latent chain of thought. Crucially, we introduce a parameter-free Dynamic Exit mechanism based on predictive entropy: the model halts inference when its internal state “crystallizes” into a low-uncertainty attractor. Empirical results on the ARC-AGI-1 benchmark validate this perspective: our 18M model achieves 65.8% accuracy, outperforming massive 73M-parameter ensembles. These findings demonstrate that adaptive iterative computation offers a far more efficient scaling axis for visual reasoning than simply increasing network width. The code is available at https://github.com/WenjieShu/LoopViT.

### 1. Introduction

A core facet of intelligence is visual reasoning: inferring an underlying rule from a handful of examples and executing it in a novel setting. Importantly, the required reasoning depth varies widely across instances, ranging from simple rule discovery to multi-step execution. As illustrated in Fig. 2, the Abstraction and Reasoning Corpus (ARC-AGI) [6, 8] operationalizes this setting through visual grid puzzles that demand precise, compositional transformations (e.g., recursive filling, object relocation, or gravity-like dynamics) from only 2–4 demonstration pairs. Unlike conventional vision benchmarks that reward dataset-scale statistical learning of textures or semantics [11, 25, 9], ARC emphasizes step-wise procedural reasoning. While humans solve these tasks through iterative hypothesis testing [19, 22], the benchmark remains challenging for deep learning systems that lack mechanisms for multi-step deliberation [29, 28].

[Figure 1]

Figure 1. Acc-Params comparisons with recurrent and vision methods. The vertical axis is accuracy (ARC-AGI-1); the horizontal axis is Parameters (memory cost). Our Loop-ViT outperforms previous methods while requiring significantly cheaper Params.

Historically, ARC reasoning has relied on methods that serialize 2D grids into 1D sequences: program synthesis and Large Language Models (LLMs) convert grids to text to exploit linguistic priors [40, 4, 37, 3, 24, 5, 27], while recurrent models [39, 20] process discrete grid tokens in a recurrent fashion. Both approaches, however, discard the spatial topology essential for visual reasoning. In contrast, the Vision ARC (VARC) framework [18] demonstrated that vanilla Vision Transformers (ViTs) [12] can solve ARC tasks directly from pixels, establishing that language is not necessary: pure visual representations suffice for ARC-style visual reasoning.

Yet a key limitation persists: feed-forward ViTs scale inefficiently with reasoning complexity. As shown in Fig. 1, simply increasing model capacity via depth or width yields diminishing returns, implying a structural mismatch for puzzles demanding recursion. We attribute this to the fact that visual reasoning is rarely a single-pass perceptual decision; it resembles an iterative latent deliberation process where

an internal state is repeatedly updated. A feed-forward network, however, implements a fixed computation graph that forces a dynamic derivation into a static mapping. Our results (Sec. 4) suggest that decoupling depth from parameter count via recurrence is a more effective scaling axis, allowing models to adapt computational effort (“Time”) rather than solely relying on raw capacity (“Space”).

To address this gap, we introduce Loop-ViT, a looped Vision Transformer tailored for pure visual reasoning. LoopViT replaces a stack of distinct Transformer layers with a weight-tied recurrent core executed for multiple iterations, decoupling computational depth from parameter count. This design encourages learning a reusable state-transition operator (a “thought step”) rather than a collection of non-robust, task-specific heuristics. To better match the local, cellularupdate nature of many ARC transformations, the recurrent core is implemented as a Hybrid Block that combines depthwise convolutions with self-attention [34, 42]. Finally, we introduce a Dynamic Exit mechanism driven by predictive entropy [33]: as predictions crystallize (i.e., the output distribution stabilizes and entropy decays), Loop-ViT halts early on easier tasks, reducing average compute without compromising accuracy on hard reasoning problems.

Empirically, iterative computation proves a more efficient scaling axis than model capacity, as shown in Fig. 1. Specifically: (I) Pareto efficiency: Loop-ViT improves the empirical Pareto frontier for visual reasoning when considering accuracy, compute, and parameters; (II) Scalable performance: a 3.8M parameter Loop-ViT (Small) reaches 60.1% score on ARC-1, surpassing the 18M VARC baseline (54.5%) with roughly one-fifth of the parameters, and scaling the number of core layers further to 18M parameters (Large) improves to 65.8%, outperforming even large ensembles of feed-forward experts; (III) Iterative refinement: across iterations we observe consistent step-wise error decay and attention dynamics that shift from broad exploration to focused execution, suggesting an emergent deliberation process.

Our contributions are summarized as follows: • Introducing Looped Transformers to Vision: We propose Loop-ViT, the first looped Vision Transformer, establishing iterative recurrence as a powerful new paradigm for abstract visual reasoning.

- • A Performant and Efficient Design: Our architecture features a weight-tied Hybrid Block that aligns with recursive algorithms for robust reasoning, and a Dynamic Exit mechanism that enables adaptive “thinking time” without extra parameters, significantly improving the accuracy-FLOPs trade-off.
- • Empirical Superiority over Parameter Scaling: We demonstrate that scaling through iteration is more effective than scaling parameters for abstract reasoning. Loop-ViT outperforms a larger state-of-the-art feed-forward model while

###### ARC-1 ARC-2

x y x y x y

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

xinfer yinfer xinfer yinfer xinfer yinfer

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Figure 2. Illustration of ARC-AGI-1 and ARC-AGI-2 benchmarks. The left two columns display tasks from ARC-AGI-1, characterized by visual priors such as “Object Cohesion” and “Pattern Completion”. These tasks primarily test perceptual generalization. The right column showcases an ARC-AGI-2 task, exemplifying higher-order algorithmic challenges such as “Symbolic Interpretation”, “Compositional Reasoning”, and “Contextual Rule Application”. For each task, the top rows show the few-shot demonstrations (Training) used to infer the rule, and the bottom row shows the query input (Inference).

using 5× fewer parameters.

### 2. Related Work

We situate Loop-ViT within the broader landscape of visual reasoning, distinguishing it from language-centric approaches and clarifying the role of recurrence in modern architectures. Fig. 3 visually summarizes the evolution of these paradigms.

Paradigms of Abstract Reasoning: Language vs. Vision. Historically, ARC reasoning has relied on language-based methods (Fig. 3 (A-B)), which serialize 2D grids into 1D sequences. This is typically done via JSON or ASCII representations for Large Language Models (LLMs) [40, 3], or as discrete tokens for recurrent models [39, 20]. Although these methods exploit powerful linguistic priors, the serialization process inevitably discards the spatial topology essential for many visual puzzles. Recently, the Vision ARC (VARC) framework [18] shifted this paradigm to pure vision (Fig. 3 (C)), treating reasoning as an image-to-image translation task. This formulation allows the use of powerful Vision Transformers (ViTs) and standard data augmentations. However, standard ViTs are feed-forward universal approximators and lack an inherent inductive bias for the iterative algorithm execution required by complex ARC tasks. Our work retains the visual formulation of VARC but fundamentally alters the computational graph from a static pass to a dynamic loop (Fig. 3 (D)).

Looped Transformers and Algorithmic Generalization. Reusing layer parameters across depth, often termed “weight tying,” allows neural networks to implement iterative algorithms instead of static pattern matching. In NLP, architectures like Universal Transformers [10] and AL-

BERT [21] established that such loops improve parameter efficiency and generalization. Recent work on “thinking models” [1, 26, 45, 31, 43] further suggests that recurrence supports a latent Chain-of-Thought, enabling models to adapt their computation based on task complexity. Modern largescale studies [14, 46] indicate that scaling this “thinking time” in the latent space can achieve results comparable to much larger, deeper models. In computer vision, recurrent processing has traditionally focused on refining continuous signals, such as optical flow estimation in RAFT [38]. Earlier efforts also applied recurrent ResNets to synthetic maze-solving tasks [2, 32], demonstrating the potential for algorithmic generalization. However, these methods often relied on convolutional priors tailored to specific, narrow domains. Loop-ViT extends this recurrent philosophy to modern Vision Transformers, treating weight-tied loops as a scalable primitive for abstract visual reasoning across diverse tasks.

Adaptive Computation Mechanisms. A unique advantage of recurrent architectures is the ability to decouple computation from parameter count. Classical “soft halting” approaches (ACT [15], PonderNet [1]) treat halting as a probabilistic variable, requiring complex auxiliary losses. In contrast, “hard halting” strategies [41, 30] rely on learned routing policies to adjust depth. Loop-ViT implies a simpler, parameter-free entropy exit. By monitoring the stabilization of predictive entropy (“crystallization”), our model halts when the internal state reaches a stable attractor, ensuring logical consistency without extra supervision or parameters.

### 3. Method

Fig. 4 illustrates the overall architecture of Loop-ViT. As depicted in Fig. 4 (B), our model introduces iterative state refinement through a weight-tied core. We first formalize this Global Recurrent Architecture and the weight-tied layer reuse (Sec. §3.1). We then detail the Hybrid Encoder Block, which integrates convolutional and attention mechanisms for heterogeneous processing of image and task tokens (Sec. §3.2). Besides, we introduce the Dynamic Exit strategy, which leverages predictive entropy to adaptively halt computation during inference. Finally, we describe the stable training protocol (Sec. §3.3)

#### 3.1. Loop-ViT Architecture

Let emb(·) : X → RM×d be the input embedding function that maps a visual grid x and task-specific context c to a sequence of M tokens with hidden dimension d. Let Mθ(·) : RM×d → RM×d be a core transformer trunk parameterized by θ. As illustrated in Fig. 4 (A), a standard non-looped vision transformer stacks L distinct layers such that the total function F is:

F(·) := head ◦ LθL ◦ ··· ◦ Lθ1 ◦ emb(·), (1)

[Figure 20]

- A
- B
- C
- D

Figure 3. Comparison of input representations and inference paradigms for ARC. (A) LLMs operate on a 1D textual token sequence obtained by serializing the ARC grids into a prompt (e.g., JSON/ASCII). (B) Recurrent token models also take a 1D sequence, but with a discrete grid-tokenization that pads the grid to a fixed canvas and inserts special boundary tokens (e.g., PAD/EOS), yielding a fixed-length token stream. (C) VARC follows a vision formulation, encoding the grid as a 2D spatial input processed in a single forward pass. (D) Ours combines the vision input with looped/iterative inference, repeatedly refining internal representations and predictions across multiple steps, bridging spatial inductive bias and recurrent computation.

where head(·) is the output projection layer. In contrast, our proposed Loop-ViT reuses the same core trunk Mθ for T iterations. Let t ∈ {1,...,Tmax} be the number of loop steps. As shown in the unrolled view of Fig. 4 (B), the state zt evolves through the recursive application of the transition operator:

zt+1 = Mθ(zt + et), z0 = emb(·), (2)

where et is a learned step-dependent embedding that disambiguates the computation progress. The final output is given by F(t)(·) = head(zt). This formulation forces the model to learn a unified, step-wise “transition rule” that is robust enough to be applied repeatedly. Crucially, scaling the computational duration T does not increase the parameter count, allowing the model to emulate complex algorithmic simulations with high efficiency. For inference iterations exceeding the training budget, we adopt an identity extrapolation for the step embeddings: et = eT

for all t > Ttrain.

train

#### 3.2. Hybrid Encoder Block

We hypothesize that ARC tasks require two distinct modes of processing: local pattern matching (e.g., continuing a line or filling a region) and global rule induction (e.g., detecting

[Figure 21]

- Figure 4. The overall pipeline of the proposed LoopViT. (A) Comparison of the standard VARC pipeline versus our Loop-ViT pipeline. Loop-ViT introduces iterative state refinement through a weight-tied core. (B) Detailed unrolled view of the Loop-ViT recurrence, where the state zt acts as a dynamic memory. (C) Structure of the Hybrid Transformer Block, employing RMSNorm and Rotary Positional Embeddings. (D) The Heterogeneous Feed-Forward Network (ConvGLU), which splits processing pathways to apply depth-wise convolution solely to image tokens while preserving task tokens, reconciling local spatial updates with global rule induction.

symmetry or gravity). To support this, our Hybrid Block explicitly fuses the strengths of convolutions and attention. The depth-wise convolution in the FFN acts as a cellular automaton update rule, processing local neighborhoods to maintain spatial consistency. Simultaneously, the global attention mechanism broadcasts rule information across the entire grid, enabling long-range reasoning.

The core trunk Mθ consists of L hybrid encoder layers. Each layer balances global relational reasoning with local spatial updates.

Internal Layer Structure. As shown in Fig. 4 (C), each layer consists of Multi-Head Self-Attention (MHSA) followed by a Heterogeneous ConvGLU as the Feed-Forward Network (FFN). To better capture spatial relationships in ARC grids, the MHSA employs Rotary Positional Embed-

dings (RoPE) [35]. Given an input sequence Z ∈ RM×d, we first project it into query, key, and value manifolds for each head h:

##### Qh,Kh,Vh = ZWhQ,ZWhK,ZWhV . (3)

The RoPE operator fR is applied to Qh and Kh to inject relative positional information. The output of a single head Oh is then computed as:

Oh = Softmax

fR(Qh)fR(Kh)T √dh

Vh, (4)

where WhQ,WhK,WhV are learnable projections and dh is the head dimension. The final MHSA output integrates all

heads via concatenation and a linear projection WO: MHSA(Z) = Concatenate(O1,...,OH)WO. (5)

To ensure numerical stability during deep recurrence, we adopt a pre-norm configuration with RMSNorm [44]. The full layer transition is defined as:

Z′ = Z + MHSA(RMSNorm(Z)) (6) Zout = Z′ + ConvGLU(RMSNorm(Z′)). (7)

Heterogeneous Processing. The primary engine of our visual induction is the Heterogeneous ConvGLU, as illustrated in Fig. 4 (D). Recognizing that task-level context tokens and spatial image patches require distinct inductive biases, we apply depthwise convolutions selectively. For a sequence Z, we first compute the gated and value representations:

[Xgate,Xval] = Linear1(Z). (8)

We then partition Xgate into task tokens Gtask and image tokens Gimg. While Gtask bypasses the spatial operator to preserve abstract rules, we reshape Gimg to a 2D grid and apply a 3 × 3 depthwise convolution (DW-Conv) to capture local connectivity:

Gˆimg = Flatten(DW-Conv(Reshape(Gimg))). (9)

The augmented gate Xˆgate is then reassembled via concatenation: Xˆgate = [Gtask,Gˆimg]. The final output is derived as:

ConvGLU(Z) = Linear2(σ(Xˆgate) ⊙ Xval), (10)

where σ denotes the activation function (e.g., SiLU). This dual-track prior facilitates the decomposition of visual reasoning: MHSA facilitates global task induction, while ConvGLU executes local spatial transformations.

#### 3.3. Dynamic Exit via Entropy-Based Prediction Crystallization

- A key insight of recurrent vision is that reasoning depth should be adaptive rather than fixed; ideally, computation should cease once a solution “crystallizes.” This design is motivated by the variable complexity of ARC tasks: while simple geometric transformations may stabilize in few iterations, complex algorithmic puzzles require prolonged refinement to resolve logical ambiguity.

To exploit this, we introduce an inference-time Dynamic Exit mechanism based on the magnitude of predictive entropy. Let Pt = softmax(head(zt)) be the predicted probability distribution over the grid categories at step t. We quantify the model’s confidence through the average pixelwise Shannon entropy:

N

1 N

Ht = −

i=1

C

Pt,i(c)log Pt,i(c), (11)

c=1

where N is the number of pixels and C is the number of color categories. During inference, generation halts at step t if the entropy falls below a confidence threshold Ht < τ, where τ = 0.05. If the threshold is not met, computation continues until reaching a hard limit of Tmax iterations. Once halted, the state is “frozen” (zk = zt,∀k > t), effectively bypassing further core layer computations. This entropy-based strategy requires no additional parameters and provides a principled measure of when the model has reached a stable attractor in its latent space.

#### 3.4. Training Strategy

We employ a training protocol designed to foster these stable recurrent dynamics.

Fixed-Depth Training. The model is trained on the ARC and RE-ARC datasets. Crucially, we do not use dynamic halting during training. Instead, we unroll the core Mθ for a fixed number of steps T (e.g., T = 12) and apply supervision on the final output. This procedure ensures that the model learns a robust transition rule Mθ that converges to the correct solution within the allocated budget, rather than overfitting to early exits. The objective is a standard per-pixel cross-entropy loss:

Loffline = CrossEntropy(PT,Ygt). (12)

Test-Time Training (TTT). During evaluation, we finetune the shared weights θ on the few-shot demonstrations of each specific task. We generate augmented views (rotations, flips, color permutations) of the support examples to create a task-specific batch. This adaptation phase specializes the general-purpose “thought step” into a dedicated algorithm for the current puzzle, further sharpening the convergence profile.

### 4. Experiments

This section evaluates Loop-ViT on the ARC-AGI. We test the hypothesis that visual reasoning is effectively modeled as a recurrent state transition rather than a fixed-depth feedforward process. The evaluation is structured around three main findings: (i) Global Performance and Efficiency, comparing Loop-ViT against LLMs and state-of-the-art vision baselines; (ii) Structural Scaling Laws, exploring the interaction between space (parameters) and time (iterations); and (iii) Step-wise Attention Dynamics, analyzing how internal attention patterns evolve across reasoning steps.

#### 4.1. Experimental Setup

Datasets and Benchmarks. Primary evaluation is conducted on the ARC-AGI-1 benchmark [7]. Following stateof-the-art pixel-based methodologies [18], we augment the training split with synthetic samples from the RE-ARC generator [17]. We report Pass@2 accuracy in percentage (%).

Table 1. Performance comparison on ARC-AGI benchmark. LoopViT demonstrates superior parameter efficiency, with a modest 11.2M parameters outperforming a 73M-parameter ensemble. Best results are bold, the second results are underlined.

Model #Params ARC-AGI-1 ARC-AGI-2

Large Language Models (LLMs) Deepseek R1 [16] 671B 15.8 1.3 Claude 3.7 8k [13] N/A 21.2 0.9 o3-mini-high [13] N/A 34.5 3.0 GPT-5 [13] N/A 44.0 1.9 Grok-4-thinking [13] 1.7T 66.7 16.0 Bespoke (Grok-4) [5] 1.7T 79.6 29.4

Recurrent Models

HRM [39] 27M 40.3 5.0 TRM [20] 7M 44.6 7.8

Vision Models VARC [18] 18M 54.5 8.3 VARC (ensemble) [18] 73M 60.4 11.1 Loop-ViT (Small) 3.8M 60.1 10.0 Loop-ViT (Medium) 11.2M 63.8 11.5 Loop-ViT (Large) 18M 65.8 14.2

human results avg.human [23] – 60.2 – best.human – 98.0 100.0

Results on the ARC-AGI-2 set are also provided to assess out-of-distribution generalization.

Implementation Framework. A two-stage training pipeline is adopted: (i) offline pre-training on augmented datasets; (ii) Test-Time Training (TTT) [36] on few-shot demonstrations. During TTT, the model specializes its weights to the specific task through local augmentations (e.g., rotations and flips).

Model Configurations. We analyze three variants designated as Small (3.8M params), Medium (11.2M params), and Large (18M params). We set Tmax ∈[20,28] for the Small variant and Tmax ∈[4,8] for Medium/Large, depending on specific model scale and task complexity.

#### 4.2. Pillar 1: Performance and Parameter Efficiency

Table 1 summarizes the performance of Loop-ViT. Our results confirm that dedicated visual reasoning architectures can achieve comparable or superior results to massive Large Language Models [16, 13] with a fraction of the parameter count.

Loop-ViT demonstrates a significant Recurrence Dividend. By recycling weights across iterations, Loop-ViT (Large) achieves 65.8% on ARC-1, surpassing the 73M VARC ensemble. This result establishes that scaling iter-

ative computation (“Time”) is more effective for algorithmic induction than increasing raw layer counts (“Space”).

#### 4.3. Pillar 2: Ablation Study

To evaluate the effectiveness of our design choices, we conduct a series of ablation experiments. We focus on two critical axes: (i) the structural trade-offs between parameter budget (space) and computational depth (time), and (ii) the necessity of spatial inductive biases in the recurrent core.

Space-Time Joint Scaling. Fig. 6 explores the trade-off between core block depth (B) and the number of unrolled loop steps (T). The diverging trajectories reveal two regimes: (i) For low-capacity cores (B = 2), increasing T yields the most significant gains as weight-tied recurrence emulates the expressive depth of larger models; (ii) For high-capacity cores (B = 10), performance continues to scale with T up to the computational limit, reaching a peak of 63.9%. Inductive Bias: Hybrid vs. Vanilla. The Hybrid Block (DW-Conv + MHSA) is compared against a standard Vanilla Transformer across core depths. As shown in Fig. 7, the Hybrid architecture maintains a consistent lead. This suggests that local spatial priors are a fundamental requirement for grounding abstract reasoning in grid-based visual domains, regardless of model depth.

Impact of Dynamic Exit. The effectiveness of adaptive halting is evaluated by comparing our dynamic-step model (constrained to T ∈ [4,8]) against a fixed 6-step baseline. As shown in Fig. 8, the Dynamic Exit mechanism achieves higher accuracy while using lower average inference compute on ARC-AGI-1. This is critical for weight-tied recurrent models, where inference compute scales approximately linearly with the number of executed iterations.

Efficiency vs. Difficulty. We further analyze the correlation between inference steps and task difficulty using the Loop-ViT variant with B = 2 (approximately 5M parameters) in Fig. 10. The results demonstrate a clear trend: samples that exit early (e.g., at Step 5) yield high accuracy (83.33%), whereas those requiring more iterations (Step 8) are inherently more challenging (45.80%). This validates that our entropy-based stopping criterion serves as an effective proxy for solution confidence, enabling the model to “fast-track” easier problems (Step 5) while instinctively reserving deeper computation for complex reasoning tasks (Step 8). This behavior mirrors human cognitive resource allocation, spending more time only when necessary.

#### 4.4. Mechanistic Insights

Beyond quantitative performance, we conduct a qualitative analysis to understand the internal refinement process of Loop-ViT. By visualizing prediction crystallization and attention evolution, we aim to uncover how the recurrent state converges toward logically consistent solutions.

Input Loop1 Loop2 Loop3 Loop4 Loop5 Loop6

[Figure 22]

[Figure 23]

[Figure 24]

Ground Truth

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

- Figure 5. Iterative Prediction Refinement in Loop-ViT. (Top) The model’s output progressively approaches the ground truth through successive iterations. (Middle) Pixel-wise difference maps between consecutive steps show decreasing prediction volatility. (Bottom) Entropy measurements demonstrate the stabilization of the model’s confidence. This “crystallization effect” reveals how recurrent processing enables gradual convergence to logically consistent solutions.

1 2 3 4 5 6

Loop steps T

30

40

50

60

Accuracy(%)

Joint Scaling on ARC-AGI-1

B = 2 B = 4 B = 6 B = 8 B = 10

- Figure 6. Joint scaling of core block depth (B) and loop steps (T) on ARC-AGI-1. We vary the number of layers in the recurrent core (B) and the number of unrolled iterations (T). Each line represents a fixed core depth. The performance multiplier provided by recurrence is most evident in lower-capacity core models (e.g.,

66

Vanilla Transformer Hybrid (Ours)

64

63.2

62.1

62.1

62

###### Accuracy(%)

61.2

60.1

60

58.8

58

56.2

56

55.0

54

52

B=2 B=4 B=6 B=8

Figure 7. Ablation of Inductive Bias: Hybrid vs. Vanilla Core across different core block depths (B) on ARC-AGI-1. We compare our Hybrid architecture (incorporating depth-wise convolutions in the FFN) against a standard Vanilla Transformer core. The Hybrid core consistently maintains a significant accuracy gap over the Vanilla baseline across all depths. This persistent advantage indicates that injecting local spatial priors is essential for grounding abstract reasoning in the image domain, and this requirement is not diminished by simply increasing model depth.

- B = 2), where increasing T from 1 to 6 yields a massive performance leap. Performance continues to scale with T even for deeper cores, demonstrating that computational time can effectively compensate for limited parameter space.

Prediction Crystallization. As illustrated in Fig. 5, LoopViT’s predictions undergo a systematic “crystallization” process. Quantitative analysis in Fig. 9 shows a synchronized decay in both prediction volatility (L2 difference) and uncertainty (entropy). The L2 difference drops precipitously in early iterations, suggesting a rapid commitment to the task geometry. The steady reduction in mean entropy Ht indicates the resolution of logical ambiguities.

Step-wise Attention Dynamics. We visualize the evolution of self-attention patterns across the loop in Fig. 11. In earlier steps, the attention matrices are relatively dense, reflecting a stage of Global Scanning where the model integrates information from task demonstrators. As processing progresses, the attention shifts toward highly sparse and localized patterns. These later steps focus precisely on the grid transitions required for the predicted rule, corresponding to a stage of

66

Fixed-6 Baseline

11M

Entropy Early Exit

64

Accuracy(%)(ARC-AGI-1)

VARC-ViT

62

8M

60

5M

58

56

VARC-18M VARC-66M

54

52

30 40 50 60 70 80 90 100

Compute (GFLOPs)

- Figure 8. Accuracy–Compute–Params comparisons. The horizontal axis is total inference compute, the vertical axis is Accuracy, and the circle radius corresponds to model Parameters. For LoopViT, GFLOPs accounts for unrolled recurrence and is computed using the average executed steps under entropy-based halting. Our Entropy Early Exit strategy (orange) consistently surpasses the fixed-step baselines (grey) across all model scales (B=2, 4, 6), establishing a stronger accuracy–compute Pareto frontier. Feedforward VARC baselines are included for comparison.

[Figure 33]

[Figure 34]

- Figure 9. Quantitative Diagnostics of Recurrent Convergence. We monitor the evolution of (top) the L2-normalized difference δt and (bottom) the average pixel-wise Shannon entropy Ht across Loop-ViT iterations. Solid lines and shaded regions represent the mean and variance across the validation set, respectively. The synchronized decay of both prediction volatility and information uncertainty confirms that the model’s internal state adheres to a stable trajectory toward a deterministic logical attractor, empirically validating our dynamic exit criterion.

Local Execution. This transition from exploratory to focused attention mirrors the deliberative strategies observed in human visual reasoning.

### 5. Conclusion

In this work, we introduced Loop-ViT, a recurrent vision architecture that challenges the paradigm of purely feedforward visual reasoning. By decoupling reasoning depth

250

83.3%

- Pass@1 Accuracy

- Pass@2 Accuracy

NumberofSamples

80

200

Accuracy(%)

Sample Count

75.0%

150

64.7%

60

55.1%

100

58.6%

52.8%

45.8%

50

40

42.8%

0

Step 5 Step 6 Step 7 Step 8

- Figure 10. Efficiency vs. Task Difficulty Analysis. Using the Loop-ViT variant (B = 2), we stratify the test set by the number of inference steps Loop-ViT requires before exiting. “Early Exit” (Step 5) samples achieve significantly higher accuracy (83.33%) compared to those requiring the full depth (Step 8, 45.80%). This confirms that the dynamic exit mechanism successfully identifies and solves simpler instances with minimal compute, while allocating more resources to harder tasks.

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Loop1 Loop2 Loop3 Loop4 Loop5 Loop6

Layer0 Layer2 Layer4 Layer6 Layer8 Layer10

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Ddemo xinfer yinfer

[Figure 48]

0.0 1.0

- Figure 11. Evolution of Attention Patterns Across Processing Steps. We visualize the average self-attention maps across LoopViT’s recurrent steps. Early steps exhibit broad attention that analyzes the full input context. Later steps develop focused, sparse patterns that precisely track the algorithmic operations needed to solve the ARC task. This shift from global scanning to localized execution mirrors human reasoning strategies.

from model capacity, we demonstrated that iterative computation is a more effective scaling axis than parameter width for abstract induction. Our design rests on two complementary pillars: a weight-tied Hybrid Block that aligns architectural inductive bias with the cellular nature of ARC transformations, and a Dynamic Exit mechanism driven by predictive entropy that enables the model to actively crystallize its latent state. Our results show that this simple approach significantly outperforms larger feed-forward baselines. We hope Loop-ViT serves as a strong baseline for future research on more complex reasoning tasks.

### References

- [1] Andrea Banino, Jan Balaguer, and Charles Blundell. Pondernet: Learning to ponder, 2021.
- [2] Arpit Bansal, Avi Schwarzschild, Eitan Borgnia, Zeyad Emam, Furong Huang, Micah Goldblum, and Tom Goldstein. End-to-end algorithm synthesis with recurrent networks: Logical extrapolation without overthinking, 2022.
- [3] Jeremy Berman. How I came in first on ARC-AGI-Pub using Sonnet 3.5 with evolutionary test-time compute. Substack,

2024. Accessed: 2025-10-13.

- [4] Jeremy Berman. How I got a record 53.6% on ARC-AGI. Substack, 2024. Accessed: 2025-10-13.
- [5] Jeremy Berman. How I got the highest score on ARC-AGI again swapping Python for English. Substack, 2025.
- [6] Franc¸ois Chollet. On the measure of intelligence. arXiv preprint arXiv:1911.01547, 2019.
- [7] Franc¸ois Chollet. On the measure of intelligence. arXiv:1911.01547, 2019.
- [8] Francois Chollet, Mike Knoop, Gregory Kamradt, and Bryan Landers. Arc prize 2024: Technical report. arXiv preprint arXiv:2412.04604, 2024.
- [9] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3213–3223, 2016.
- [10] Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. Universal transformers. arXiv preprint arXiv:1807.03819, 2018.
- [11] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.
- [12] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.
- [13] ARC Prize Foundation. ARC-AGI benchmarking: Leaderboard and dataset for the ARC-AGI benchmark. https: //arcprize.org/leaderboard, 2025. Accessed: 2025-11-01.
- [14] Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. Scaling up test-time compute with latent reasoning: A recurrent depth approach. arXiv preprint arXiv:2502.05171, 2025.
- [15] Alex Graves. Adaptive computation time for recurrent neural networks, 2017.
- [16] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv:2501.12948, 2025.

- [17] Michael Hodel. Addressing the abstraction and reasoning corpus via procedural example generation. arXiv:2404.07353, 2024.
- [18] Keya Hu, Ali Cy, Linlu Qiu, Xiaoman Delores Ding, Runqian Wang, Yeyin Eva Zhu, Jacob Andreas, and Kaiming He. Arc is a vision problem! arXiv preprint arXiv:2511.14761, 2025.
- [19] Aysja Johnson, Wai Keen Vong, Brenden M Lake, and Todd M Gureckis. Fast and flexible: Human program induction in abstract reasoning tasks. arXiv preprint arXiv:2103.05823, 2021.
- [20] Alexia Jolicoeur-Martineau. Less is more: Recursive reasoning with tiny networks. arXiv:2510.04871, 2025.
- [21] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. Albert: A lite bert for self-supervised learning of language representations. arXiv preprint arXiv:1909.11942, 2019.
- [22] Solim LeGris, Wai Keen Vong, Brenden M Lake, and Todd M Gureckis. H-arc: A robust estimate of human performance on the abstraction and reasoning corpus benchmark. arXiv preprint arXiv:2409.01374, 2024.
- [23] Solim LeGris, Wai Keen Vong, Brenden M. Lake, and Todd M. Gureckis. H-ARC: A robust estimate of human performance on the abstraction and reasoning corpus benchmark. arXiv:2409.01374, 2024.
- [24] Wen-Ding Li, Keya Hu, Carter Larsen, Yuqing Wu, Simon Alford, Caleb Woo, Spencer M. Dunn, Hao Tang, Wei-Long Zheng, Yewen Pu, and Kevin Ellis. Combining induction and transduction for abstract reasoning. In ICLR, 2025.
- [25] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014.
- [26] Kerong Liu et al. Looped transformers are better at learning learning algorithms. In ICLR, 2024.
- [27] Matthew V Macfarlane and Cl´ement Bonnet. Searching latent program spaces. arXiv:2411.08706, 2024.
- [28] Arseny Moskvichev, Victor Vikram Odouard, and Melanie Mitchell. The ConceptARC benchmark: Evaluating understanding and generalization in the ARC domain. arXiv:2305.07141, 2023.
- [29] Rolf Pfister and Hansueli Jud. Understanding and benchmarking artificial intelligence: Openai’s o3 is not agi. arXiv preprint arXiv:2501.07458, 2025.
- [30] David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, PC Conway, and P Adam. Mixture-of-depths: Dynamically allocating compute in transformer-based language models. arXiv preprint arXiv:2404.02258, 2024.
- [31] Nikunj Saunshi, Nishanth Dikkala, Zhiyuan Li, Sanjiv Kumar, and Sashank J Reddi. Reasoning with latent thoughts: On the power of looped transformers. arXiv preprint arXiv:2502.17416, 2025.
- [32] Avi Schwarzschild, Eitan Borgnia, Arjun Gupta, Furong Huang, Uzi Vishkin, Micah Goldblum, and Tom Goldstein. Can you learn an algorithm? generalizing from easy to hard problems with recurrent networks. Advances in Neural Information Processing Systems, 34:6695–6706, 2021.

- [33] Claude E Shannon. A mathematical theory of communication. The Bell system technical journal, 27(3):379–423, 1948.
- [34] Dai Shi. Transnext: Robust foveal visual perception for vision transformers, 2024.
- [35] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 2024.
- [36] Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei A. Efros, and Moritz Hardt. Test-time training with selfsupervision for generalization under distribution shifts. In ICML, 2020.
- [37] Hao Tang, Keya Hu, Jin Zhou, Sicheng Zhong, Wei-Long Zheng, Xujie Si, and Kevin Ellis. Code repair with LLMs gives an exploration-exploitation tradeoff. In NeurIPS, 2024.
- [38] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In ECCV, 2020.
- [39] Guan Wang, Jin Li, Yuhao Sun, Xing Chen, Changling Liu, Yue Wu, Meng Lu, Sen Song, and Yasin Abbasi Yadkori. Hierarchical reasoning model. arXiv:2506.21734, 2025.
- [40] Ruocheng Wang, Eric Zelikman, Gabriel Poesia, Yewen Pu, Nick Haber, and Noah D. Goodman. Hypothesis search: Inductive reasoning with language models. In ICLR, 2024.
- [41] Guanyu Xu, Jiawei Hao, Li Shen, Han Hu, Yong Luo, Hui Lin, and Jialie Shen. LGViT: Dynamic early exiting for accelerating vision transformer. In ACM MM, 2023.
- [42] Weihao Yu, Mi Luo, Pan Zhou, Chenyang Si, Yichen Zhou, Xinchao Wang, Jiashi Feng, and Shuicheng Yan. Metaformer is actually what you need for vision, 2022.
- [43] Boyi Zeng, Shixiang Song, Siyuan Huang, Yixuan Wang, He Li, Ziwei He, Xinbing Wang, Zhiyu Li, and Zhouhan Lin. Pretraining language models to ponder in continuous space. arXiv preprint arXiv:2505.20674, 2025.
- [44] Biao Zhang and Rico Sennrich. Root mean square layer normalization, 2019.
- [45] H Zhou et al. Looped transformers for length generalization. In ICLR, 2025.
- [46] Rui-Jie Zhu, Zixuan Wang, Kai Hua, Tianyu Zhang, Ziniu Li, Haoran Que, Boyi Wei, Zixin Wen, Fan Yin, He Xing, Lu Li, Jiajun Shi, Kaijing Ma, Shanda Li, Taylor Kergan, Andrew Smith, Xingwei Qu, Mude Hui, Bohong Wu, Qiyang Min, Hongzhi Huang, Xun Zhou, Wei Ye, Jiaheng Liu, Jian Yang, Yunfeng Shi, Chenghua Lin, Enduo Zhao, Tianle Cai, Ge Zhang, Wenhao Huang, Yoshua Bengio, and Jason Eshraghian. Scaling latent reasoning via looped language models, 2025.

