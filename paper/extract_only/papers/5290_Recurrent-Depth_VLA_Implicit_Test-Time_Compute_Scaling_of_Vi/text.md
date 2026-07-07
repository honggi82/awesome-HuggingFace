# Recurrent-Depth VLA: Implicit Test-Time Compute Scaling of Vision–Language–Action Models via Latent Iterative Reasoning

## arXiv:2602.07845v1[cs.RO]8Feb2026

Yalcin Tur1, Jalal Naghiyev2,†, Haoquan Fang3,4,†, Wei-Chuan Tsai3 Jiafei Duan3,4,∗, Dieter Fox3,4,∗, Ranjay Krishna3,4,∗

1Stanford University 2Technical University of Munich 3University of Washington 4Allen Institute for Artificial Intelligence †Co-second authors ∗Co-advising rd-vla.github.io

Abstract—Current Vision–Language–Action (VLA) models rely on fixed computational depth, expending the same amount of compute on simple adjustments and complex multi-step manipulation. While Chain-of-Thought (CoT) prompting enables variable computation, it scales memory linearly and is ill-suited for continuous action spaces. We introduce Recurrent-Depth VLA (RD-VLA), an architecture that achieves computational adaptivity via latent iterative refinement rather than explicit token generation. RD-VLA employs a recurrent, weight-tied action head that supports arbitrary inference depth with a constant memory footprint. The model is trained using truncated backpropagation through time (TBPTT) to efficiently supervise the refinement process. At inference, RD-VLA dynamically allocates compute using an adaptive stopping criterion based on latent convergence. Experiments on challenging manipulation tasks show that recurrent depth is critical: tasks that fail entirely (0% success) with single-iteration inference exceed 90% success with four iterations, while simpler tasks saturate rapidly. RDVLA provides a scalable path to test-time compute in robotics, replacing token-based reasoning with latent reasoning to achieve constant memory usage and up to 80× inference speedup over prior reasoning-based VLA models.

I. INTRODUCTION

Humans do not operate with a fixed computational budget. When performing trivial maneuvers, such as adjusting a grip or nudging an object, we rely on near-reflexive, low-effort responses. However, when environments become cluttered or actions require long-horizon foresight, human cognition adaptively reallocates resources—slowing down to process sensory evidence and refine internal models before acting [41]. This ability to scale ”compute” to task complexity is a hallmark of efficient reasoning that remains a significant challenge for modern robotics. Currently, most Vision-Language-Action (VLA) models [22, 4, 25] are limited to a fixed computational depth, processing every control step with the same number of parameters regardless of difficulty; a simple gripper adjustment receives the same compute as high-precision navigation in a cluttered space.

While generalist VLAs built upon Multimodal Large Language Models (MLLMs) exhibit impressive visual understand-

ing, they often remain brittle and fail to fully leverage the latent reasoning capabilities of their backbone models. To bridge this gap, reasoning-centric VLAs [16, 25, 51, 49] have emerged. These models incorporate intermediate thinking processes—typically via supervised Chain-of-Thought (CoT) which categorized into textual reasoning (generating pseudoCoT labels) or visual reasoning (generating sub-goal images or 2D traces) [53, 35, 25]. However, a fundamental limitation persists: these models perform reasoning at the token level. By tethering the ”thinking” process to specific task settings and explicit sequences, these methods are often inefficient or misaligned with the requirements of continuous robotic control. But unlike language modeling, reasoning for physical manipulation is inherently subtler and less conducive to tokenization; consequently, attempting to verbalize these dynamics carries significant overhead, requiring the curation of custom reasoning datasets and scaling memory linearly with the length of the reasoning chain.

Beyond these constraints, a fundamental architectural limitation persists: these models perform reasoning in the output space. This requires iterative transitions between the model’s high-dimensional latent space and a discretized, lowbandwidth output space (e.g., text, depth bins, or coordinates). This cyclical re-projection is fundamentally non-ideal, as it forces the model to collapse continuous internal states into lossy, discretized representations only to re-encode them for subsequent reasoning steps. The resulting information bottlenecks and quantization noise inherently limit reasoning fidelity, constraining the model’s deliberation to the resolution of its output vocabulary rather than the continuous physical dynamics of the environment.

To address this, we look toward the iterative nature of biological systems. Human cognition is fundamentally characterized by recurrent neural dynamics, where the repeated recruitment of the same neural circuitry transforms initial sensory inputs into progressively refined, deliberative representations [24]. An architectural analogy in neural networks is the recursive reuse of layers, in which a model iteratively

processes information until a refined decision is reached. This iterative process differs from Diffusion Policies [8], which generate actions through multi-step refinement. Diffusion Policies operate by iteratively denoising an action trajectory in the output space. While effective for modeling multi-modal distributions, this process is fundamentally a generative sampling technique rather than a deliberative one: it refines the action signal but does not scale or enrich the underlying representation of the scene.

We introduce Recurrent-Depth VLA (RD-VLA), a new class of VLA model that enables adaptive test-time compute for visuomotor control. Unlike existing reasoning-centric methods, our approach enables adaptive behavior emerges naturally from its recurrent architecture, eliminating the need for curated reasoning traces or explicit supervision of iteration counts. Inspired by recent advances in latent reasoning and looped transformer architectures [56, 14], we extend the principle of latent iterative refinement to the VLA domain. Our core insight is that robotic reasoning can occur entirely within the representation space, bypassing the need to decode intermediate tokens or perform computationally expensive denoising iterations in action space. RD-VLA achieves this by decomposing the action head into three stages: an initial encoding stack, a weight-tied recurrent block for iterative refinement, and a final projection layer. By recursively updating hidden states through the shared recurrent block, the model progressively refines its internal representations within a fixed-dimensional latent space. This design enables arbitrarily deep computation at test time while maintaining a constant memory footprint, with minimal parameter overhead due to weight reuse.

When coupled with an adaptive stopping criterion (Figure 3), RD-VLA dynamically allocates computation on a per-sample basis, yielding a scalable and resource-efficient framework for high-precision robotic control at inference. To

- our knowledge, RD-VLA is the first VLA model to support scaling test-time computation through implicit latent-space reasoning via a weight-tied recurrent core. We further observe task-dependent convergence behavior: the required unrolled depth emerges naturally from the execution context, reflecting the underlying complexity of the action being performed. This property enables adaptive inference-time stopping and execution schedules that modulate the action horizon based on the model’s internal reasoning trajectory, resulting in inference speeds up to an order of magnitude faster than prior action-reasoning approaches. Empirically, RD-VLA outperforms strong end-to-end VLAs and all reasoning-centric VLAs on the LIBERO benchmark [27], achieving 93.0% success with fixed iterations and 92.5% with uncertaintybased adaptive computation. On the CALVIN benchmark [31], RD-VLA achieves an average task length of 3.39 and a task-5 success rate of 45.3%, demonstrating strong longhorizon generalization. Finally, RD-VLA transfers effectively to real-world robotic settings, achieving robust performance on challenging long-horizon tasks such as bread toasting and towel folding.

II. RELATED WORK

- A. Vision-Language-Action Models

Large language models (LLMs) [1, 11, 19] and vision language models (VLMs) [28, 43, 21, 55] have shown strong problem-solving capabilities and deep semantic understanding of visual and textual data. Leveraging large-scale Internet data, they generalize to unseen tasks, though acquiring comparable real-world robotic data remains challenging. Despite advances in assembling large robotic datasets [12, 32], the Open XEmbodiment dataset [34] is the largest, featuring 22 robots, 527 skills, and 160,266 tasks from 21 institutions.

Using this dataset, Transformer-based general robot policies like Octo [39] and RT-1 [5] were trained. However, models trained from scratch for robotic tasks struggled to generalize to new environments, tasks, and objects. RT-2 [57] addressed these issues by using a large (55B-parameter) pretrained vision-language model to generate actions, while OpenVLA [22] emerged as a leading open-source alternative. Building on these, π0 [4] introduced a generalist policy using flowmatching for multi-modal actions, with π0.5[18] providing a more efficient, high-performance iteration for real-time deployment.

- B. Reasoning and Efficient-Compute VLA Models

As Vision-Language-Action (VLA) models scale in parameter count, balancing the tradeoff between computational demand and the real-time requirements of robotic control has become increasingly critical. Recent research has split into two primary directions: optimizing the efficiency of existing backbones and introducing structured reasoning stages to move beyond reactive policies.

To mitigate the high latency of large Transformer backbones, several works explore dynamic resource allocation. TinyVLA [44] focuses on data-efficient distillation to create high-performance, small-scale models. To reduce redundant computation in invariant visual scenes, VLA-Cache [46] introduces an adaptive token caching mechanism that reuses textual and visual features across control steps. Alternatively, DeeRVLA [48] treats model depth as a dynamic variable, activating model segments based on task difficulty. This enables rapid early-exiting during trivial movements while preserving full capacity for high-entropy manipulations.

A second direction aims to improve performance by grounding actions in explicit reasoning via mid-level representations. Mobility-VLA [9] employs long-context VLMs to reason over topological graphs for navigation, while TraceVLA [53] utilizes visual trace prompting to enhance spatial-temporal awareness. Several recent works have integrated explicit reasoning directly into the control loop. Embodied Chain of Thought (ECoT) [49] leverages embodied chain-of-thought to generate textual justifications before action emission. ThinkAct [16] utilizes reinforced visual latent planning, and MolmoAct [25] introduces Action Reasoning Models (ARMs) that generate depth-aware perception tokens and editable trajectories. Similarly, CoT-VLA [51] demonstrates that visual chain-of-thought

[Figure 1]

- Fig. 1: Recurrent-Depth VLA. (Left) Previous reasoning VLAs (e.g., ThinkAct, MolmoAct) generate explicit reasoning tokens in output space, requiring expensive autoregressive decoding. (Center) Our approach performs iterative refinement entirely in latent representation space, bypassing token generation overhead. (Right) RD-VLA achieves comparable performance to autoregressive reasoning baselines on LIBERO-10 while being substantially faster due to the efficiency of latent reasoning with adaptive compute.

[Figure 2]

- Fig. 2: Recurrent-Depth VLA Architecture. The Prelude (P) grounds learned queries via cross-attention to mid-layer VLM features. The weight-tied Recurrent Core (R) iteratively refines a noisy latent scratchpad over K iterations, cross-attending to final-layer VLM representations and proprioception. The Coda (C) decodes the converged state into actions. Recurrence depth K adapts dynamically at inference based on task complexity.

reasoning about spatial constraints and object relationships significantly improves generalization in out-of-distribution environments.

C. Recurrent Transformers

Recurrent Transformers is an architecture where all or some portion of layers in transformers are recurred, which means that representation of the later layer gets reinjected back into the earlier layers. The initial work focused on recurring only one transformer layer [10], while later works explored vari-

- ous architectures and methods to train recurrent transformers [14, 13, 15, 30]. Although various research mostly focused on

inspecting inductive biases of recurrent architectures on toy scales, and in algorithmic setting [42, 20, 36, 29], recent results had shown that these model could scale to Foundation Models sizes [14, 30, 56]. Recurrent transformers allow models to scale computation by repeating layers, enabling several useful properties: test-time scaling, where the number of computation steps can be increased at inference; uncertainty quantification for test-time scaling, since operating at the representation level makes it possible to define metrics that reflect model confidence; adaptive compute [14], where prior work uses measures such as cosine distance and KL divergence between representations across recurrent iterations to adjust the number

of computation steps based on uncertainty, and we illustrate this behavior in simulation in Figure 3; and no need for specialized CoT data, since recurrent transformers provide an architectural mechanism for iterative reasoning without requiring curated chain-of-thought supervision, which can be particularly difficult to obtain in robotics.

[Figure 3]

- Fig. 3: Case study for adaptive computation. In a LIBERO rollout, the model dynamically selects different numbers of iterations before terminating, depending on the execution state. It uses fewer iterations (7–9) at steps 1 and 30, which correspond to simpler motions like navigation and placing, and more iterations (about 14) at steps 10 and 25, where the actions are more complex, such as grasping.

III. METHOD

We introduce Recurrent-Depth VLA (RD-VLA), a framework designed to decouple computational depth from the fixed architectural constraints of pretrained vision-language backbones. While standard VLA architectures typically utilize fixed-depth MLP heads or compute-intensive iterative processes in the output space such as diffusion or flow-matching action heads [4], RD-VLA shifts the computational burden to a weight-tied recurrent transformer core operating entirely within a continuous latent manifold. This design allows the model to scale its test-time compute by unrolling the recurrent block to an arbitrary depth r, enabling dynamic allocation of compute based on task complexity (Figure 3).

A. Architectural Backbone and Token Flow

The RD-VLA action head is a modular framework designed to be backbone-agnostic, capable of being integrated with any Vision-Language Model (VLM) that produces dense latent representations. For the purposes of this work, we instantiate the framework using a VLM based on Qwen2.5-0.5B [47, 40] from MiniVLA [2], which utilizes the Prismatic [21] training recipe with a frozen DINOv2 [33] and SigLIP [50] fused vision encoder from MiniVLA [2].

The frozen vision encoder generates 256 vision tokens per image observation (512 for wrist and the main camera), which are projected into a Qwen2-0.5B (24-layer) LLM backbone fine-tuned via LoRA. Following the architectural design principles of OpenVLA-OFT [23], we augment the VLM input sequence with a set of 64 dedicated learned latent embeddings. These tokens serve as specialized grounded placeholders that attend to the multi-modal context during the LLM’s forward pass.

After the VLM execution, we extract the hidden states and partition them into two distinct sets:

- • Task/Vision representations (hvis ∈ R512×D): Capturing spatial and semantic scene information.
- • Latent-specific representations (hlat ∈ R64×D): Extracted from the latent token positions to provide compressed, task-aligned features.

These representations are combined to form a static conditioning manifold. Specifically, the recurrent head Rθ at each iteration k attends via cross attention to a concatenated context vector [h(24)vis+lat;p], effectively grounding the latent reasoning process in both observation and the high-level semantic tokens.

B. Recurrent-Depth Architecture

We introduce a framework that decouples computational depth from the fixed architectural constraints of pretrained vision-language backbones. While standard VLA architectures typically utilize fixed-depth heads, RD-VLA shifts the computational burden to a weight-tied recurrent transformer core operating within a continuous latent manifold. Following the Huggin approach [14], we partition the architecture into a functional triplet: the Prelude, the Recurrent Core, and the Coda (Fig. 2). The Prelude and Coda serve as non-recurrent interface layers that map representations into and out of a dedicated latent manifold optimized for iterative reasoning.

The process begins with the Prelude (Pϕ), a non-recurrent interface that consumes K = 8 learned queries. First K queries self attend to each other bidirectionally. Then by performing cross-attention over the VLM’s middle-layer visual

features h(12)vis+lat, the Prelude transforms these queries into a grounded latent foundation:

### Spre = Pϕ(Queries,h(12)vis+lat) ∈ RK×D (1)

Parallel to this, we initialize a latent scratchpad S ∈ RK×D from a high-entropy truncated normal distribution to serve as the evolving state for the reasoning process:

### S0 ∼ TruncNormal(0,γinit · σinit) (2)

This noisy initialization transforms S into a blank workspace that the model must iteratively “clean” and refine. This ensures that the model learns a stable refinement operator rather than overfitting to a specific starting point.

1) Latent Iterative Reasoning via Input Injection: The core iterative refinement occurs within the Recurrent Core (Rθ), a weight-tied transformer block. To maintain representational stability and prevent the model from losing its grasp on

the physical observations over long unrolls (representational collapse), we utilize a persistent Input Injection strategy. At every step k, the recurrent block observes both the current state of the scratchpad Sk−1 and the static foundation Spre provided by the Prelude.

Specifically, for each iteration k = 1...r, the previous scratchpad state Sk−1 is concatenated with the fixed Spre along the feature dimension to form a 2D-dimensional context. This is mapped back to the manifold dimension via a learned adapter and normalized:

### xk = RMSNorm(γadapt · Wadapt[Sk−1;Spre]) (3)

where Wadapt ∈ RD×2D. The scratchpad state is then updated through the weight-tied recurrent block:

### Sk = Rθ(xk,[h(24)vis+lat;p]) (4)

Similar to Prelude during this update, Rθ first performs bidirectional self-attention across K, and then does gated cross-attention where the queries are derived from xk and the keys/values are derived from a concatenated conditioning manifold [h(24)vis+lat;p]. This manifold consists of the 64 taskaligned latent tokens and 512 vision tokens from the VLM’s final layer and the robot’s current proprioception p.

2) Coda and Action Projection: Once the recurrence reaches the desired depth r, the converged scratchpad Sr is processed by the non-recurrent Coda (Cψ). The Coda performs the final decoding pass by moving the representation

- out of the latent manifold, attending to the self, and high-level

VLM features (h(24)vis+lat). Finally, an output projection layer maps these refined features to the robot’s action space:

a = Wout · RMSNorm(Cψ(Sr,[h(24)vis ;h(24)lat ;p])) (5) where Wout is the final linear layer producing the control commands a. This architecture ensures that any intermediate state Sk is a valid representation, while Sk+1 represents a strictly more refined iteration of the action plan.

3) Training with Randomized Recurrence: To promote convergence to a steady state independent of initialization depth, we sample the number of iterations N during training from a heavy-tailed log-normal Poisson distribution:

### τ ∼ N(ln(µrec) − 0.125,σ2), N ∼ Poisson(eτ) + 1 (6)

where µrec = 32. We utilize TBPTT, where gradients are propagated through only the final d = 8 iterations, while earlier steps are computed with gradients detached. This forces the network to learn to iteratively refine the scratchpad from any noisy initialization into a stable manifold, ensuring Sk+1 is a strictly better refinement of Sk. This enables the model to scale compute dynamically during inference without retraining.

C. Adaptive Computation

Leveraging the properties of the model, we implement an adaptive computation mechanism at inference. Rather than specifying a fixed iteration count, we utilize the model’s own internal convergence as a proxy for reasoning certainty.

We define a stopping criterion based on the KullbackLeibler (KL) divergence between the action distributions of consecutive iterations. Approximating KL via Mean Squared Error (MSE) in the action space, the inference loop terminates at step k∗ when:

### ||ak − ak−1||22 < δ (7)

where ak is the predicted action chunk at step k and δ is a convergence threshold (e.g., 1e−3). This allows the model to self-regulate: terminating instantly for trivial movements while allocating extended compute for complex situations.

D. Adaptive Execution

Adaptive computation determines how long to recur. At the same time adaptive execution determines how many actions to execute. We observe that instances requiring deep recurrence (k∗ > 8) often correspond to states of high uncertainty. In these regimes, executing a long horizon of actions is dangerous, as small errors in the initial plan compound over time.

We propose two strategies to couple the depth of reasoning with the execution of action.

- 1) Threshold-Based Adaptive Execution: This method

modulates the execution horizon using a binary reasoning threshold τ. We hypothesize that high iteration counts imply higher epistemic uncertainty. Consequently, if convergence requires k∗ > τ steps, we truncate the horizon to a shorter duration Hshort to mitigate compounding errors, while retaining Hlong for confident predictions (k∗ ≤ τ):

Hexec =

Hlong if k∗ ≤ τ Hshort if k∗ > τ

(8)

- 2) Linear Decay Execution: To provide a continuous

scaling mechanism, we implement a linear decay schedule that reduces the execution horizon inversely to the reasoning depth. Given a base iteration budget τbase, the horizon Hexec decreases by one step for every additional iteration required to converge:

### Hexec(k∗) = max(Hmin,Hmax − max(0,k∗ − τbase)) (9)

This approach forces the agent to replan more frequently as computational demand increases, favoring safety over efficiency in complex states.

IV. EXPERIMENTS

We evaluate our approach in both simulation and realworld settings. Simulation experiments are conducted on two widely used manipulation benchmarks, LIBERO [27] and CALVIN [31], while real-world experiments are performed on a bimanual YAM manipulator. Our evaluation is designed to answer the following questions:

- IV-A How does our approach scale with recurrent computation?
- IV-B Is adaptive computation necessary for robotic manipulation tasks?
- IV-C What is the most effective strategy for adaptive computation at inference time?

TABLE I: Comparison on the LIBERO [27] benchmark. Bold* indicates the best performance, Bold the second best, and Italics the third best. “Params” denotes backbone scale in Billions. Here we compare three types of VLAs. End-to-end (E2E) VLAs directly predict robot actions without intermediate reasoning steps. Token reasoning VLAs first perform explicit reasoning by generating tokens before producing action outputs. Finally, latent reasoning, our approach, performs iterative reasoning in a latent space using a recurrent structure before emitting actions.

Method Params Spat. Obj. Goal Long Avg.

SmolVLA [38] 2.2 93.0 94.0 91.0 77.0 88.8 OpenVLA [22] 7 84.7 88.4 79.2 53.7 76.5 WorldVLA [7] 7 87.6 96.2 83.4 60.0 81.8 π0-FAST [4] 3 96.4 96.8 88.6 60.2 85.5

E2E VLAs

CoT-VLA [51] 7 87.5 91.6 87.6 69.0 81.1 FlowVLA [54] 8.5 93.2 95.0 91.6 72.6 88.1 SpatialVLA [35] 4 88.2 89.9 78.6 55.5 78.1 ThinkAct [16] 7 88.3 91.4 87.1 70.9 84.4 Fast-ThinkAct [17] 3 92.0 97.2 90.2 79.4 89.7 TraceVLA [53] 7 84.6 85.2 75.1 54.1 74.8 MolmoAct [25] 7 87.0 95.4 87.6 77.2 86.6

Token Reasoning

Latent Reasoning RD-VLARD-VLA (Fixed)(Adaptive) 0.50.5 92.088.6 99.098.8 96.096.8 84.885.8 93.092.5

- IV-D Is representation-level reasoning for action prediction more effective than token-level reasoning?
- IV-E How well does our approach perform in real-world deployment?

In this section, we evaluate Recurrent-Depth VLA (RDVLA) across three dimensions: (1) the scaling behavior of latent recurrence on standard benchmarks, (2) the efficacy of adaptive test-time compute, and (3) performance comparisons against state-of-the-art baselines. We utilize the LIBERO [27] and CALVIN [31] benchmarks for quantitative analysis and conduct real-world evaluations to assess physical robustness.

- A. Performance Scaling via Recurrent Computation

We first establish the relationship between computational depth and task success rates by evaluating RD-VLA with a fixed number of recurrent iterations Ninf ∈ {1,...,32} on all task suites of LIBERO. As shown in Figure 4, performance exhibits a clear log-linear improvement with increased recurrence. Starting from near-random performance at Ninf = 1 (8.4% average), the model achieves substantial gains at each doubling of compute: 40.5% at Ninf = 2 (382% increase), 84.1% at Ninf = 4 (108% increase), and 92.6% at Ninf = 8 (10% increase). Performance saturates between 8 and 12 iterations, with the peak of 93.1% achieved at Ninf = 24. Notably, scaling beyond Ninf = 12 yields diminishing returns, suggesting that for the given task distribution, the model shows an increasing trend of its performance on different tasks by scaling up the compute budget at test-time.

- B. Necessity of Task-dependent Computation

Beyond aggregate performance, we observe that individual tasks exhibit distinct convergence profiles, reflecting that different tasks have different requirements for computation complexity. Figure 5 illustrates this task-dependent behavior on selected Long-horizon tasks from LIBERO.

Critically, the number of required iterations emerges naturally from the task context rather than being prescribed.

TABLE II: Comprehensive Ablation of Recurrent-Depth VLA. We compare fixed computational depths against adaptive strategies (Binary, Linear Decay, and Pure KL). For adaptive runs, we report Mean Iterations (k¯) and Standard Deviation (σ) to illustrate reasoning convergence.

Strategy & Threshold τ k¯ σ Spatial Object Goal Long Avg.

- Rec=1 1.0 0.00 9.0 12.2 11.4 1.0 8.4

- Rec=2 2.0 0.00 38.0 61.2 47.6 15.0 40.5 Rec=4 4.0 0.00 79.2 93.0 89.2 74.8 84.1 Rec=8 8.0 0.00 93.0 97.8 94.2 85.2 92.6 Rec=12 12.0 0.00 92.0 99.0 96.0 84.8 93.0 Rec=16 16.0 0.00 91.6 99.0 95.2 85.2 92.8 Rec=24 24.0 0.00 92.4 99.2 94.2 86.6 93.1 Rec=32 32.0 0.00 91.4 98.8 93.8 84.4 92.1

Fixed Recurrence

- τ = 1e−4 11.04 1.20 91.2 98.2 96.0 79.8 91.3

- τ = 2e−4 9.71 1.11 90.6 97.2 96.0 80.8 91.2 τ = 5e−4 7.93 1.03 88.6 98.8 96.8 85.8 92.5 τ = 1e−3 6.61 0.89 88.6 97.8 94.6 84.8 91.5 τ = 5e−3 4.27 0.41 83.2 93.6 87.4 61.8 81.5 τ = 1e−2 3.36 0.19 74.6 88.6 81.6 43.6 72.1

Binary Adaptation

- τ = 1e−4 11.55 1.03 91.2 97.2 96.2 80.0 91.2

- τ = 2e−4 9.86 1.10 91.2 98.4 96.6 82.0 92.1 τ = 5e−4 7.90 1.05 88.8 97.6 94.4 82.0 90.7 τ = 1e−3 6.62 0.90 89.0 98.4 94.6 83.8 91.5 τ = 5e−3 4.26 0.42 84.2 94.4 90.6 62.2 82.9 τ = 1e−2 3.36 0.19 76.2 87.2 81.6 42.6 71.9

Linear Decay

- τ = 1e−4 10.58 1.40 91.2 98.2 93.8 81.8 91.3

- τ = 2e−4 9.25 1.12 92.0 98.6 95.0 82.0 91.9 τ = 5e−4 7.66 0.95 90.8 99.4 93.2 82.0 91.4 τ = 1e−3 6.57 0.79 89.8 98.2 93.8 79.6 90.4 τ = 5e−3 4.30 0.40 86.2 91.4 87.2 50.8 78.9 τ = 1e−2 3.38 0.18 75.8 82.2 81.8 33.8 68.4

Pure KL (Threshold)

Some tasks (e.g., Task 4) achieve near-perfect performance with just two iterations, while others (e.g., Task 5) require three or more iterations before any meaningful success. This observation reveals the phenomenon that different tasks might have different optimal iteration counts in the latent reasoning process. This further motivates our adaptive computation strategy: rather than using a fixed compute budget, the model should dynamically allocate iterations based on the difficulty

[Figure 4]

- Fig. 4: Performance across LIBERO benchmarks for different numbers of recurrences. All task categories show consistent improvement with increased computational depth, with models converging between 8–12 iterations on average.

[Figure 5]

- Fig. 5: Performance on selected 5 Long tasks across recurrence steps. Each task exhibits distinct convergence behavior: Task 4 jumps from 6% at iteration 1 to nearly 80% at iteration 2, while Task 5 remains at 0% through iteration 2 and only reaches ∼70% at iteration 3. This demonstrates the task-dependent and emergent adaptive behavior of our model.

of the current state.

- C. Adaptive Computation Strategies

We evaluate whether the model can dynamically calibrate its own compute budget using the KL-divergence stopping criterion described in Section III-C. Table II presents comprehensive ablation studies comparing fixed computational depths against three adaptive strategies: Binary Adaptation, Linear Decay, and Pure KL thresholding.

Results demonstrate that adaptive computation maintains parity with the best fixed-depth models while reducing average inference cost. With τ = 5 × 10−4, the Binary Adaptation strategy achieves 92.5% success rate (comparable to the 93.0% peak of fixed recurrence at N = 12) using an average of only k¯ = 7.93 iterations, which is a 34% reduction in compute. Figure 6 illustrates the distribution of adaptive exits across task categories, showing that the model dynamically adjusts

TABLE III: Comparison on the CALVIN ABC→D benchmark. We report tasks completed in a row (↑) and average episode length (↑).

CALVIN ABC→D Params Tasks completed in a row ↑ Avg. len ↑

1 2 3 4 5

OpenVLA[22] 7 91.3 77.8 62.0 52.1 43.5 3.27 VLAS [52] 7 87.2 64.2 40.9 28.1 19.6 2.40 LCB [37] 7 73.6 50.2 28.5 16.0 9.9 1.78

DeeR [48] 3 86.2 70.1 51.8 41.5 30.4 2.82 RoboFlamingo [26] 3 82.4 61.9 46.6 33.1 23.5 2.48 GR-1 [45] 3.1 85.4 71.2 59.6 49.7 40.1 3.06 SuSIE [3] 1.3 87.0 69.0 49.0 38.0 26.0 2.69

RT-1 [6] – 53.3 22.2 9.4 3.8 1.3 0.90 RD-VLA (Ours) 0.5 91.4 79.5 67.9 54.9 45.3 3.39

the number of recurrent steps based on the difficulty of the current state.

Notably, all three adaptive strategies perform comparably at matched compute budgets, suggesting that the key insight is not the specific stopping criterion but rather the principle of condition-dependent computation allocation. However, Binary Adaptation with τ = 5 × 10−4 achieves the best balance between efficiency and performance.

- D. Performance against Other Baselines

We compare RD-VLA against state-of-the-art VLA methods on the LIBERO and CALVIN benchmarks. Table I presents results on LIBERO, where methods are grouped into three categories: end-to-end VLAs, token-level reasoning methods, and our latent reasoning approach.

RD-VLA achieves state-of-the-art performance of 93.0% with fixed recurrence, significantly outperforming all prior methods, including the strong Fast-ThinkAct baseline (89.7%). Remarkably, our model achieves this with only 0.5B parameters which is 14× smaller than 7B token-reasoning methods. The adaptive variant maintains competitive performance (92.5%) while providing the efficiency benefits of dynamic compute allocation.

Table III presents results on the CALVIN ABC→D benchmark, which evaluates long-horizon task chaining. RD-VLA achieves the highest average chain length of 3.39, outperforming OpenVLA [22] (3.27) and all other baselines. This validates that latent refinement effectively extends the model’s sequential planning capabilities.

- E. Real-world Experiments

To validate the capability of our method in the real-world setting, we deployed RD-VLA on a bimanual robot arm (bimanual YAM) across four household tasks: placing a cube in a bowl, wiping a dish, folding a towel, and toasting bread. We train and compare our model in each task, both using fixed 8 iterations and Pure KL with threshold τ = 10−4. Figure 7 presents task progression scores compared to Diffusion Policy and π0.5 baselines. RD-VLA with fixed 8 iterations demonstrates superior robustness across all tested scenarios, reaching nearly 100% completion on dish wiping and significantly outperforming baselines on the remaining tasks. The adaptive

[Figure 6]

- Fig. 6: Histograms of zero-shot, per-token adaptive exits based on KL divergence between consecutive steps (τ = 10−4). The distributions show task-dependent iteration counts: Spatial tasks require more iterations (µ = 11.8) than Object (µ = 9.8) or Goal (µ = 9.0) tasks, reflecting their relative complexity.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Put cube into bowl Wipe the dish Fold towel Toast the bread

0.0

0.2

0.4

0.6

0.8

1.0

TaskProgression

Bimanual-Arm YAM Tasks

Diffusion Pi-05 RecurrentVLA-8 RecurrentVLA-adaptive

[Figure 7]

“Put cube into bowl” “Wipe the dish” “Fold towel” “Toast the bread”

[Figure 8]

[Figure 9]

[Figure 10]

- Fig. 7: Performance on real-world tasks compared to π0.5 and Diffusion Policy baselines. RD-VLA variants consistently outperform baselines across all tasks, with the fixed 8-iteration model achieving near-perfect performance on dish wiping.

a relatively small backbone (0.5B parameters). We expect that scaling this architecture to larger backbones and training on more diverse datasets will yield significant performance gains.

A key limitation observed in our experiments is the boundary of depth generalization. While performance scales predictably with the number of recurrent steps up to some optimal number of iterations, extending recurrence beyond this number of iterations may lead to state saturation or performance degradation rather than continued refinement. Addressing this problem—perhaps through architectural innovations or specific training protocols remains an open challenge for scaling latent reasoning in robotics.

Recurrent architectures inherently expose internal state dynamics that can serve as proxies for model uncertainty. This offers a suite of test-time intervention capabilities, such as adaptive computation or uncertainty-aware execution. For instance, the system could autonomously halt execution or request operator assistance if the variance between recurrent states exceeds a safety threshold. While we demonstrated the viability of these mechanisms, the design space for such interventions is vast. We leave the specific implementation of such mechanisms to future work, focusing here on the architectural foundation that makes them possible.

While this work highlights the latency and memory limitations of token-based Chain-of-Thought (CoT) reasoning for high-frequency control, our Recurrent-Depth approach offers a complementary pathway. Future research could investigate hybrid approaches where recurrent depth is modulated pertoken to enhance CoT reasoning capabilities in embodied agents.

variant (RD-VLA-adaptive, Pure KL with threshold τ = 10−4) maintains competitive performance, matching or closely trailing the fixed-strategy model while achieving the highest score on cube placement. This demonstrates the viability of dynamic computation in physical settings, even when slightly trading off performance on complex manipulation tasks like towel folding.

VI. CONCLUSION AND FUTURE WORK

We introduced Recurrent-Depth VLA (RD-VLA), a novel architecture that shifts robotic reasoning from the discrete output space to the continuous latent space. This work serves as the first implementation of latent iterative reasoning for visuomotor policies, demonstrating that effective test-time compute scaling can be achieved without the memory and latency overhead of autoregressive Chain-of-Thought generation. Our experiments provide substantial evidence that the model successfully utilizes recurrent iterations to refine its internal state, with performance scaling log-linearly with com-

V. DISCUSSION AND LIMITATIONS

Our primary objective was to investigate latent iterative reasoning in robotic control, rather than to hyper-optimize a specific model for state-of-the-art dominance. While RD-VLA achieves competitive or superior performance on standard benchmarks such as LIBERO [27], we emphasize that these results were obtained with minimal hyperparameter tuning and

pute depth. Crucially, this architecture unlocks new capabilities for embodied agents: the ability to think longer for harder tasks and the capacity to measure its own uncertainty through latent convergence. We aimed to open a new design space for efficient, reasoning-capable robotic policies. We believe that optimizing the regimes for adaptive compute and exploring the scaling laws of latent recurrence represent promising avenues for future research. By validating the efficacy of iterative latent refinement, RD-VLA provides a foundation for the next generation of resource-efficient and robust foundation models in robotics.

REFERENCES

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Suneel Belkhale and Dorsa Sadigh. Minivla: A better vla with a smaller footprint, 2024. URL https://github.com/ Stanford-ILIAD/openvla-mini.
- [3] Kevin Black, Mitsuhiko Nakamoto, Pranav Atreya, Homer Walke, Chelsea Finn, Aviral Kumar, and Sergey Levine. Zero-shot robotic manipulation with pretrained image-editing diffusion models, 2023. URL https://arxiv. org/abs/2310.10639.
- [4] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy

Groom, Karol Hausman, Brian Ichter, et al. π0: A visionlanguage-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

- [5] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.
- [6] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Tomas Jackson, Sally Jesmonth, Nikhil J Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, KuangHuei Lee, Sergey Levine, Yao Lu, Utsav Malla, Deeksha Manjunath, Igor Mordatch, Ofir Nachum, Carolina Parada, Jodilyn Peralta, Emily Perez, Karl Pertsch, Jornell Quiambao, Kanishka Rao, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Kevin Sayed, Jaspiar Singh, Sumedh Sontakke, Austin Stone, Clayton Tan, Huong Tran, Vincent Vanhoucke, Steve Vega, Quan Vuong, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-1: Robotics transformer for realworld control at scale, 2023. URL https://arxiv.org/abs/ 2212.06817.
- [7] Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, et al. Worldvla: Towards autoregressive

- action world model. arXiv preprint arXiv:2506.21539, 2025.
- [8] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 44(10-11):1684–1704, 2025.
- [9] Hao-Tien Lewis Chiang, Zhuo Xu, Zipeng Fu, Mithun George Jacob, Tingnan Zhang, Tsang-Wei Edward Lee, Wenhao Yu, Connor Schenck, David Rendleman, Dhruv Shah, et al. Mobility vla: Multimodal instruction navigation with long-context vlms and topological graphs. arXiv preprint arXiv:2407.07775, 2024.
- [10] Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. Universal transformers,

2019. URL https://arxiv.org/abs/1807.03819.

- [11] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- [12] Frederik Ebert, Yanlai Yang, Karl Schmeckpeper, Bernadette Bucher, Georgios Georgakis, Kostas Daniilidis, Chelsea Finn, and Sergey Levine. Bridge data: Boosting generalization of robotic skills with crossdomain datasets. arXiv preprint arXiv:2109.13396, 2021.
- [13] Khashayar Gatmiry, Nikunj Saunshi, Sashank J. Reddi, Stefanie Jegelka, and Sanjiv Kumar. On the role of depth and looping for in-context learning with task diversity,

2024. URL https://arxiv.org/abs/2410.21698.

- [14] Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. Scaling up test-time compute with latent reasoning: A recurrent depth approach, 2025. URL https://arxiv.org/abs/2502. 05171.
- [15] Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space, 2025. URL https://arxiv.org/abs/2412.06769.
- [16] Chi-Pin Huang, Yueh-Hua Wu, Min-Hung Chen, YuChiang Frank Wang, and Fu-En Yang. Thinkact: Visionlanguage-action reasoning via reinforced visual latent planning. arXiv preprint arXiv:2507.16815, 2025.
- [17] Chi-Pin Huang, Yunze Man, Zhiding Yu, Min-Hung Chen, Jan Kautz, Yu-Chiang Frank Wang, and Fu-En Yang. Fast-thinkact: Efficient vision-language-action reasoning via verbalizable latent planning. arXiv preprint arXiv:2601.09708, 2026.
- [18] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al.

π0.5: A vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

- [19] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, De-

- vendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.
- [20] Alexia Jolicoeur-Martineau. Less is more: Recursive reasoning with tiny networks, 2025. URL https://arxiv. org/abs/2510.04871.
- [21] Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. Prismatic vlms: Investigating the design space of visuallyconditioned language models. In Forty-first International Conference on Machine Learning, 2024.
- [22] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.
- [23] Moo Jin Kim, Chelsea Finn, and Percy Liang. Finetuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025.
- [24] V A Lamme and P R Roelfsema. The distinct modes of vision offered by feedforward and recurrent processing. Trends Neurosci, 23(11):571–579, November 2000.
- [25] Jason Lee, Jiafei Duan, Haoquan Fang, Yuquan Deng, Shuo Liu, Boyang Li, Bohan Fang, Jieyu Zhang, Yi Ru Wang, Sangho Lee, et al. Molmoact: Action reasoning models that can reason in space. arXiv preprint arXiv:2508.07917, 2025.
- [26] Xinghang Li, Minghuan Liu, Hanbo Zhang, Cunjun Yu, Jie Xu, Hongtao Wu, Chilam Cheang, Ya Jing, Weinan Zhang, Huaping Liu, Hang Li, and Tao Kong. Visionlanguage foundation models as effective robot imitators,

2024. URL https://arxiv.org/abs/2311.01378.

- [27] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning, 2023. URL https://arxiv.org/abs/2306.03310.
- [28] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023.
- [29] Sean McLeish, Arpit Bansal, Alex Stein, Neel Jain, John Kirchenbauer, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, Jonas Geiping, Avi Schwarzschild, and Tom Goldstein. Transformers can do arithmetic with the right embeddings, 2024. URL https://arxiv.org/abs/2405. 17399.
- [30] Sean McLeish, Ang Li, John Kirchenbauer, Dayal Singh Kalra, Brian R. Bartoldson, Bhavya Kailkhura, Avi Schwarzschild, Jonas Geiping, Tom Goldstein, and Micah Goldblum. Teaching pretrained language models to think deeper with retrofitted recurrence, 2025. URL https://arxiv.org/abs/2511.07384.
- [31] Oier Mees, Lukas Hermann, Erick Rosete-Beas, and Wolfram Burgard. Calvin: A benchmark for languageconditioned policy learning for long-horizon robot manipulation tasks, 2022. URL https://arxiv.org/abs/2112. 03227.
- [32] Russell Mendonca, Shikhar Bahl, and Deepak Pathak.

- Structured world models from human videos. arXiv preprint arXiv:2308.10901, 2023.
- [33] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin ElNouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
- [34] Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024.
- [35] Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, et al. Spatialvla: Exploring spatial representations for visual-language-action model. arXiv preprint arXiv:2501.15830, 2025.
- [36] Nikunj Saunshi, Stefani Karp, Shankar Krishnan, Sobhan Miryoosefi, Sashank J. Reddi, and Sanjiv Kumar. On the inductive bias of stacking towards improving reasoning,

2024. URL https://arxiv.org/abs/2409.19044.

- [37] Yide Shentu, Philipp Wu, Aravind Rajeswaran, and Pieter Abbeel. From llms to actions: Latent codes as bridges in hierarchical robot control, 2025. URL https://arxiv.org/ abs/2405.04798.
- [38] Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Aractingi, Caroline Pascal, Martino Russi, Andres Marafioti, et al. Smolvla: A vision-language-action model for affordable and efficient robotics. arXiv preprint arXiv:2506.01844, 2025.
- [39] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024.
- [40] Qwen Team. Qwen2.5: A party of foundation models, September 2024. URL https://qwenlm.github.io/blog/ qwen2.5/.
- [41] Tom Verguts, Eliana Vassena, and Massimo Silvetti. Adaptive effort investment in cognitive and physical tasks: A neurocomputational model. Frontiers in Behavioral Neuroscience, 9:57, 2015.
- [42] Guan Wang, Jin Li, Yuhao Sun, Xing Chen, Changling Liu, Yue Wu, Meng Lu, Sen Song, and Yasin Abbasi Yadkori. Hierarchical reasoning model, 2025. URL https: //arxiv.org/abs/2506.21734.
- [43] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- [44] Junjie Wen, Yichen Zhu, Jinming Li, Minjie Zhu, Zhibin

- Tang, Kun Wu, Zhiyuan Xu, Ning Liu, Ran Cheng, Chaomin Shen, et al. Tinyvla: Towards fast, data-efficient vision-language-action models for robotic manipulation. IEEE Robotics and Automation Letters, 2025.
- [45] Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kong. Unleashing large-scale video generative pretraining for visual robot manipulation, 2023. URL https: //arxiv.org/abs/2312.13139.
- [46] Siyu Xu, Yunke Wang, Chenghao Xia, Dihao Zhu, Tao Huang, and Chang Xu. Vla-cache: Towards efficient vision-language-action model via adaptive token caching in robotic manipulation. arXiv preprint arXiv:2502.02175, 2025.
- [47] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.
- [48] Yang Yue, Yulin Wang, Bingyi Kang, Yizeng Han, Shenzhi Wang, Shiji Song, Jiashi Feng, and Gao Huang. Deervla: Dynamic inference of multimodal large language models for efficient robot execution. Advances in Neural Information Processing Systems, 37:56619–56643, 2025.
- [49] Michał Zawalski, William Chen, Karl Pertsch, Oier Mees, Chelsea Finn, and Sergey Levine. Robotic control via embodied chain-of-thought reasoning. arXiv preprint arXiv:2407.08693, 2024.
- [50] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pretraining. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.
- [51] Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, et al. Cot-vla: Visual chainof-thought reasoning for vision-language-action models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1702–1713, 2025.
- [52] Wei Zhao, Pengxiang Ding, Min Zhang, Zhefei Gong, Shuanghao Bai, Han Zhao, and Donglin Wang. Vlas: Vision-language-action model with speech instructions for customized robot manipulation, 2025. URL https: //arxiv.org/abs/2502.13508.
- [53] Ruijie Zheng, Yongyuan Liang, Shuaiyi Huang, Jianfeng Gao, Hal Daum´e III, Andrey Kolobov, Furong Huang,

- and Jianwei Yang. Tracevla: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies. arXiv preprint arXiv:2412.10345, 2024.
- [54] Zhide Zhong, Haodong Yan, Junfeng Li, Xiangchen Liu, Xin Gong, Tianran Zhang, Wenxuan Song, Jiayi Chen, Xinhu Zheng, Hesheng Wang, et al. Flowvla: Visual chain of thought-based motion reasoning for vision-language-action models. arXiv preprint arXiv:2508.18269, 2025.
- [55] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing visionlanguage understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.
- [56] Rui-Jie Zhu, Zixuan Wang, Kai Hua, Tianyu Zhang, Ziniu Li, Haoran Que, Boyi Wei, Zixin Wen, Fan Yin, He Xing, Lu Li, Jiajun Shi, Kaijing Ma, Shanda Li, Taylor Kergan, Andrew Smith, Xingwei Qu, Mude Hui, Bohong Wu, Qiyang Min, Hongzhi Huang, Xun Zhou, Wei Ye, Jiaheng Liu, Jian Yang, Yunfeng Shi, Chenghua Lin, Enduo Zhao, Tianle Cai, Ge Zhang, Wenhao Huang, Yoshua Bengio, and Jason Eshraghian. Scaling latent reasoning via looped language models, 2025. URL https://arxiv.org/abs/2510.25741.
- [57] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-languageaction models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023.

