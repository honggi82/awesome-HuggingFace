# arXiv:2602.05885v2[cs.LG]6Feb2026

## DR. KERNEL: Reinforcement Learning Done Right for Triton Kernel Generations

Wei Liu1, Jiawei Xu3, Yingru Li2, Longtao Zheng4, Tianjian Li2, Qian Liu2, Junxian He1 1HKUST 2TikTok 3CUHK(SZ) 4NTU

#### Abstract

High-quality kernel is critical for scalable AI systems, and enabling LLMs to generate such code would advance AI development. However, training LLMs for this task requires sufficient data, a robust environment, and the process is often vulnerable to reward hacking and lazy optimization. In these cases, models may hack training rewards and prioritize trivial correctness over meaningful speedup. In this paper, we systematically study reinforcement learning (RL) for kernel generation. We first design KERNELGYM, a robust distributed GPU environment that supports reward hacking check, data collection from multi-turn interactions and long-term RL training. Building on KERNELGYM, we investigate effective multi-turn RL methods and identify a biased policy gradient issue caused by self-inclusion in GRPO. To solve this, we propose Turn-level Reinforce-Leave-One-Out (TRLOO) to provide unbiased advantage estimation for multi-turn RL. To alleviate lazy optimization, we incorporate mismatch correction for training stability and introduce Profiling-based Rewards (PR) and Profiling-based Rejection Sampling (PRS) to overcome the issue. The trained model, DR. KERNEL-14B, reaches performance competitive with Claude-4.5-Sonnet in Kernelbench. Finally, we study sequential test-time scaling for DR. KERNEL-14B. On the KernelBench Level-2 subset, 31.6% of the generated kernels achieve at least a 1.2× speedup over the Torch reference, surpassing Claude-4.5-Sonnet (26.7%) and GPT-5 (28.6%). When selecting the best candidate across all turns, this 1.2× speedup rate further increases to 47.8%. All resources, including environment, training code, models, and dataset, are included in github.com/hkust-nlp/KernelGYM.

47.8

|25.1| | |7.3|
|---|---|---|---|
| | | | |
| | | | |
| | | | |

31.6

28.6

|18.8| | |3.0|
|---|---|---|---|
| | | | |
| | | | |
| | | | |

25.6 26.7

|16.5| | |12.0|
|---|---|---|---|
| | | | |
| | | | |
| | | | |

|13.5| | |11.0|
|---|---|---|---|
| | | | |
| | | | |
| | | | |

|16.9| | |1.2|
|---|---|---|---|
| | | | |
| | | | |
| | | | |

D .K -14B + STTS (Best)

D .K -14B + STTS

D .K -14B Claude-4.5-Sonnet GPT-5

Level 1

Level 2

Level 3

| |
|---|

| |
|---|

| |
|---|

Figure 1: Rate of generated kernels achieving at least a 1.2× speedup over the Torch reference on KernelBench across three level subsets. DR. KERNEL-14B is competitive with Claude-4.5-Sonnet and GPT-5, and applying sequential test-time scaling further improves DR. KERNEL-14B, surpassing both models on two of the three subsets.

#### 1 Introduction

Efficient GPU kernels are critical for scalable AI systems. Seminal works like FlashAttention (Dao, 2024) and FlashInfer (Ye et al., 2025) have demonstrated that specialized kernels are essential for unlocking the full efficiency of modern LLMs. However, developing such

###### Hacked_Kernel.py

###### Lazy_Optimization.py

16

50

@triton.jit def sum_channels_kernel(…):

@triton.jit def _layernorm_forward_rows_kernel(…):

14

| |
|---|

…

…

40

Fast@1.2(%)

class ModelNew(nn.Module):

12

Fast@1(%)

class ModelNew(nn.Module):

def __init__(self, …): self.conv_trans = nn.ConvTranspose3d(…) self.max_pool1 = nn.MaxPool3d(…) self .max_pool2 = nn.MaxPool3d(…)

def __init__(…): … self.ln = nn.LayerNorm(…)

Fast@1w/o Hacking Check

10

30

Fast@1

Fast@1.2

8

| |
|---|

def forward(…): # (thinking via annotations…) if self.training:

| |
|---|

20

def forward(self, x): x = self. conv_trans(x) x = self.max_pool1(x) x = self.max_pool2(x) … sum_channels_kernel[grid](…) return y

6

| |
|---|

pass # # (…) return self.ln(x)

4

10

| |
|---|

0 50 100 150 200 250 300

Training Step

- Figure 2: Left: The plot uses a dual y-axis to compare two metrics. We report results from two models: Fast@1 of the model trained without reward hacking check (§ 3.3) (w/o hacking check), and Fast@1 / Fast@1.2 of the model trained with hacking check enabled. Evaluation is done using the same standard for all variants with hacking check. Multi-turn RL is run on Qwen3-8B-Base after cold-start SFT, using TRLOO for advantage estimation (§4.2) and KERNELGYM as the execution environment (§3). Right: Representative cases illustrating reward hacking and lazy optimization. In Hacked Kernel.py, the model emits a Triton kernel to satisfy the “@triton.jit” heuristic but never calls it, and additionally skips the real computation under the default training mode, inflating the measured speedup. In Lazy Optimization.py, the model replaces only a trivial sub-operation (channel summation) with a kernel while leaving the remaining computation in Torch, missing the larger gains from fusion.

kernels remains difficult. It requires deep expertise spanning both algorithms and GPU hardware intricacies. While Domain-Specific Languages (DSLs) like Triton and TileLang (Wang et al., 2025) have lowered the entry barrier compared to CUDA, achieving peak performance still requires significant manual engineering. This difficulty makes kernel development a natural candidate for automation.

Kernel generation is characterized by easily accessible optimization objectives. Correctness can be verified through execution, and efficiency can be measured via profiling, making these tasks naturally suited for RL, which, therefore, offers a promising approach to enhancing the ability of LLMs to generate kernel code. However, these optimization benefits come with potential pitfalls. During training, models can devolve into reward hacking, exploiting measurement loopholes or executing invalid operations that appear fast but result in meaningless optimizations. Alternatively, models may generate correct but trivial kernel implementations that fail to deliver meaningful speedup, a phenomenon we refer to as lazy optimization, as shown in Figure 2. Previous works only partially addressed these challenges and remain RL not fully explored for this field. For instance, AutoTriton (Li et al., 2025) optimizes solely for correctness, neglecting the crucial objective of speedup. TritonRL (Woo et al., 2025) identifies the risk of reward hacking but relies on imprecise LLM-as-a-judge mechanisms rather than rigorous execution-based verification. Similarly, while CudaLLM (CudaLLM Team, 2025) collects valuable data, it stops short of full-scale RL training, reporting only correctness metrics. Furthermore, most prior efforts are limited to single-turn generation while kernel optimization can be recursively refined for multiple rounds. Kevin (Baronio et al., 2025) attempts multi-turn RL, yet it is constrained by a small-scale dataset of only 280 samples split from the KernelBench benchmark.

In this work, we systematically study RL training for kernel code generation. We follow the task definiton of previous works (Ouyang et al., 2025; Li et al., 2025; Baronio et al., 2025), where a Torch reference code is provided and models are asked to optimize the implementation via kernel code. We focus on Triton, a Pythonic, high-level GPU programming language that is more amenable to current LLMs than low-level CUDA. Moreover, Triton’s abstraction makes execution-based safeguards against reward hacking more tractable during training, making it an ideal testbed for developing RL methodologies for kernel generation. Following the adage, “A workman must first sharpen his tools if he is to do his work well,” we first build a robust environment, KERNELGYM. Unlike prior ad-hoc solutions, KERNELGYM is a scalable, distributed serving system tailored for long-horizon RL: it provides strict fault isolation to tolerate frequent CUDA runtime failures, and exposes granular environment

###### KernelGYM

###### Evaluation

[Figure 1]

Register/Unregister

###### Toolkits

[Figure 2]

[Figure 3]

###### Correctness: ✅ Speedup: 1.2x Hacking Check: Pass

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Hacking Check

GPU Node 1

Profiler

[Figure 8]

###### Server

[Figure 9]

###### GPU Worker 1

[Figure 10]

Interface

Data Collection

Subprocess

[Figure 11]

[Figure 12]

Eval Tasks

Correctness Speed Test

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Eval Results

[Figure 17]

FastAPI

[Figure 18]

[Figure 19]

Backend

[Figure 20]

[Figure 21]

…

Worker Monitor

Eval Results Eval Tasks

Multi-Turn Trajectories

[Figure 22]

Triton

[Figure 23]

RL Training

GPU Worker K

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

###### Scheduled Subprocess

CUDA

[Figure 29]

Tasks

…

[Figure 30]

[Figure 31]

[Figure 32]

TRLOO PR PRS

[Figure 33]

Redis Scheduler

Multi-Turn RL

[Figure 34]

Eval Results

[Figure 35]

Task Manager

TileLang

[Figure 36]

[Figure 37]

: Heartbeat : Restart Signal

[Figure 38]

- Figure 3: Overview of KERNELGYM and our training framework. Left: We study RL training methods for kernel generation, including multi-turn RL with TRLOO, profilingbased rewards (PR), and profiling-based rejection sampling (PRS). Right: The architecture of KERNELGYM: a server-worker split distributed design. The server side (interface + task manager) receives evaluation jobs and schedules to registered distributed GPU workers; each job runs in an isolated subprocess; toolkits produce structured signals for training, parallel evaluation and data collections.

feedback, including execution profiling and hacking checks, to enable rigorous evaluation and high-quality data collection from multi-turn interactions.

Equipped with KERNELGYM, we investigate effective multi-turn RL methods for LLMs. We identify that standard GRPO can induce biased policy-gradient updates; to address this, we propose Turn-level Reinforce-Leave-One-Out (TRLOO), an unbiased advantage estimator for multi-turn RL. Furthermore, we alleviate “lazy optimization” from stability and optimization objective prospectives. We find mismatch correction contributes to the training stability. And we further propose Profiling-based Rewards (PR) and Profiling-based Rejection Sampling (PRS) to explicitly incentivize alleviating performance bottlenecks. Finally, we study sequential test-time scaling (STTS) to maximize the inference capability of our trained models. Experimental results demonstrate KERNELGYM with modular design and hacking check enables long-term RL training. Our final multi-turn RL method, DR. KERNEL, yields strong DR. KERNEL-14B model which achieves substantial gains on two KernelBench (Ouyang et al., 2025) subsets, reaching performance competitive with frontier models like Claude-4.5-Sonnet. And for DR. KERNEL-14B with STTS, on the KernelBench Level-2 subset, 31.6% of the generated kernels achieve at least a 1.2× speedup over the Torch reference, surpassing Claude-4.5-Sonnet (26.7%) and GPT-5 (28.6%). When selecting the best candidate across all turns, the 1.2× speedup rate further increases to 47.8%.

#### 2 Pitfalls in Kernel Generation

Correctness and speedup are the two objectives of kernel generation. However, triton kernel generation is particularly vulnerable to reward hacking (Baronio et al., 2025; Woo et al., 2025): the model can produce outputs that appear correct and fast under the evaluation while being actually meaningless. A simple example is copying the Torch reference implementation, it passes correctness checks but yields only ∼1.0× speedup, providing an easy way to harvest reward without learning kernel generation. Another common failure mode occurs when Triton kernels are emitted but never executed in the kernel entry function, resulting in misleading timing measurements. As shown in Figure 2 (right), although the model generates the kernel implementation for LayerNorm, the kernel is never actually executed.

Beyond hacking, we observe a second bottleneck that is specific to performance-oriented kernel generation. In Figure 2 (Left), we validate Fast@1 and Fast@1.2 on KernelBench Level

- 2 during training. Fast@1 / Fast@1.2 represent the fraction of kernels passing correctness checks with at least 1 × /1.2× speedup over the Torch reference. Fast@1 improves steadily, while the stricter Fast@1.2 saturates quickly (around ∼100 steps). Unlike standard code generation, where functional correctness is often the end goal, kernel generation ultimately targets meaningful speedup. Many KernelBench (Ouyang et al., 2025) tasks admit trivialbut-correct implementations (e.g., local rewrites such as swapping a simple activation) that do not address the true runtime bottlenecks. As shown in Figure 2 (right), the model leverages the kernel for the simple summation operation, while leaving other operations to the Torch implementation, thereby overlooking the potential benefits of more advanced fusion optimizations. Since these solutions still yield only ∼ 1× speedup, the policy tends to exploit such low-hanging fruits to improve Fast@1, without producing kernels that clear the higher bar required by Fast@1.2. We refer to this behavior as lazy optimization. Cases are provided in Figure 2 (Right) and Appendix E.1.

These two issues are also evident and remain unresolved in previous works. For instance, in AutoTriton (Li et al., 2025), despite implementing a rule-based reward that assigns 0 reward to code without the “@triton.jit” decorator, the model still encounters hacking cases, such as neglecting to call the function, as shown in Figure 2. In our evaluation, code generated by their released model still exhibits approximately 10% hacking cases in the KernelBench level-1 subset. In addition to reward hacking, the issue of lazy optimization persists. As shown in Table 1, while AutoTriton achieves a notable 30.6% performance on the Fast@1 metric in the Kernelbench level-2 subset, its performance on the stricter Fast@1.2 metric quickly saturates to just only 9.2%.

Next, to make RL for kernel generation practically feasible, we start with a robust execution environment with hacking checks and torch profiler (§ 3), then introduce an unbiased multi-turn RL estimator (§ 4), and finally improve training stability and align optimization objectives toward real speedup to alleviate lazy optimization (§ 5).

- 3 KERNELGYM: A Gym for Kernel Generations

###### 3.1 Design Principles

Training LLM agents for iterative kernel generation requires an environment that can (1) evaluate correctness and performance at scale, (2) remain resilient against frequent CUDA runtime failures, and (3) provide granular feedback for RL optimization. To meet these requirements under constrained GPU resources, we architect KERNELGYM as a scalable, distributed serving system that strictly decouples agent clients from kernel execution (Figure 3, right). This separation keeps the client implementation lightweight—agents focus on policy learning, while the system handles scheduling, resource management, and failure recovery.

Concretely, the design of KERNELGYM follows four principles tailored to GPU workloads: (i) Serialized execution: profiling is highly sensitive to contention, so we enforce a oneGPU-one-task policy to prevent context pollution and ensure reliable timing; (ii) Elastic scalability: GPU workers can be added or removed dynamically without interrupting training; (iii) Fault isolation and self-recovery: unsafe generated kernels frequently trigger illegal memory access or unrecoverable CUDA errors, hence failures must be isolated at the task level and recovered automatically to maintain long-horizon availability; (iv) Rich environmental feedback: coarse signals such as pass/fail or a single speedup value are insufficient for RL, so KERNELGYM exposes structured feedback (e.g., profiling summaries and reward-hacking detection) to support optimization and data collection.

###### 3.2 Architecture and Modular Design

Server KERNELGYM adopts a server–worker architecture. The server serves as the central coordinator, consisting of an Interface (FastAPI) and a Task Manager (Redis + scheduler). The Interface exposes REST APIs for task submission/querying and worker registration. The Task Manager uses Redis to maintain persistent task/worker states and dispatches tasks to available workers with timeout-based re-queuing to sustain throughput.

GPU Worker and Monitor Kernel evaluations are executed by distributed GPU workers, where each GPU is treated as an independent worker instance. Each worker pulls scheduled tasks from the server and runs them sequentially using the configured backend/toolkits (§3.3). To contain CUDA/runtime failures from generated kernels without corrupting longrunning processes, each evaluation runs in a fresh spawned subprocess, while the parent worker remains CUDA-clean and continues serving subsequent tasks. A worker monitor tracks liveness (e.g., heartbeat/process health), automatically restarts failed workers, and reassigns unfinished tasks to healthy workers to maintain RL training stability.

###### 3.3 Backends and Toolkits

Backends Following Ouyang et al. (2025), KERNELGYM runs the generated kernel code and evaluates it on two basic toolkits: correctness and speedup against a Torch reference. In this work we mainly use a Triton backend, but the same interface can also support other kernel languages (e.g., CUDA, TileLang). More broadly, KERNELGYM can be extended to other GPU tasks by adding other toolkits that define how to run the code and how to leverage the output.

For correctness, the backend compares the generated code with a reference implementation under a fixed test protocol (e.g., multiple randomized inputs) and returns a discrete status such as pass, mismatch, runtime error, or compilation error. For performance, the backend measures running time using a consistent timing procedure (e.g., warmup followed by repeated runs) and reports the speedup relative to the baseline. The performance metric would only be measured on the correct kernel code.

Hacking Check As discussed in §2, reward hacking is a major failure mode in performanceoriented kernel RL. To mitigate such behaviors, KERNELGYM implements an executionbased hacking check that filters suspicious candidates from optimization.

Concretely, KERNELGYM instruments Triton’s launch path to record executed Triton kernels and measures end-to-end runtime in both train and eval modes. Motivated by Figure 2 (right), where the model branches on self.training to bypass execution of kernel code and inflate speedup, we mark a candidate as incorrect if it executes no Triton kernel in either mode.

Profiler Beyond scalar feedbacks, KERNELGYM exposes profiling summaries to provide richer and more reliable feedback, which mainly serves as informative context for multiturn optimization where the models recursively optimize the code during multi-turn RL training or test-time scaling. For incorrect candidates, the profiler returns structured failure diagnostics (e.g., exception type and traceback) to help the model localize and fix errors in subsequent turns. For correct executions, it provides kernel-level summaries of the executed code, alongside the default correctness and performance measurements, enabling the model to identify unoptimized operators. These signals are also used during training to reduce the lazy optimization behavior in §2. We operationalize this idea via Profiling-based Rewards and Profiling-based Rejection Sampling (§5). The examples for profiling feedback are shown in Figure 10 (Appendix E.1).

#### 4 Multi-Turn RL with KERNELGYM

Kernel generation naturally lends itself to multi-turn refinement. Just as human developers iterate by writing, executing, and revising kernels based on runtime profiling, LLMs can improve solutions through repeated propose–evaluate–refine cycles. Previous work like AlphaEvolve (Novikov et al., 2025) has demonstrated that such interaction with an execution environment can even lead to the discovery of fundamental algorithms.

Motivated by this, KERNELGYM enables long-term multi-turn RL for kernel generation: at each turn, the model proposes a revised kernel conditioned on the history, and KERNELGYM executes it to provide immediate feedback. This setup is distinct from recent agentic RL with multi-step tool use (Wei et al., 2025; Jin et al., 2025), where learning is typically driven

55

50

50

40

45

Fast@1(%)

Fast@1(%)

40

30

35

AutoTriton

20

w/o Hacking Check

w/ Single Turn

30

w/ Single Turn

w/ = 0

w/ = 0

w/ GRPO

10

w/ GRPO

25

w/ TRLOO

w/ TRLOO

0 50 100 150 200 250 300

1 2 3

Training Step

Turn Number

- Figure 4: Fast@1 on KernelBench Level 2. Left: Fast@1 at turn 3 over training steps. Right: Fast@1 across turns (evaluated at the selected checkpoint). Since all methods besides AutoTriton achieve their best performance at turn 3, we select checkpoints based on turn 3 performance. For AutoTriton we use their released model.

by a sparse, single-outcome reward. In contrast, our environment yields dense, turn-level rewards after each execution.

###### 4.1 Cold-Start Data Collections

To alleviate the data scarcity for kernel code and teach models basic kernel-generation skills (e.g., tiling, fusion, etc.), we distill multi-turn trajectories from a proprietary model (e.g., GPT-5) interacting with KERNELGYM. The prompting template is provided in Appendix F.1.

We start from 8K kernel-generation queries from CUDALLM-SFT (CudaLLM Team, 2025) and use GPT-5 to generate 5-turn Triton implementations. At each turn, the generated code is executed in KERNELGYM and the model receives feedbacks from the environment (§ 3.3), including correctness status, error diagnostics for failed runs or runtime, and profiling summaries for successful kernels. This feedback is appended to the next-turn query, prompting the model to refine the implementation conditioned on the full interaction history.

###### 4.2 Multi-Turn RL

Reward Design As in cold-start collection, we provide the model with environment feedback and optimize it in a multi-turn RL setting. Following prior work (Baronio et al., 2025; Woo et al., 2025), we combine correctness and speedup to define the per-turn reward for the response i at the t-th turn yi,t as:

Ri,t = C(yi,t) + C(yi,t) · speedupi,t . (1)

Here C(yi,t) is a binary correctness reward, and speedupi,t is computed from runtime measurements. We clip the speedup term to improve training stability and reduce the

impact of anomalous evaluations: speedupi,t = min TTreference

, 3 . This clipping prevents

kernel

rare timing artifacts from producing excessively large rewards, and is also consistent with the observation that speedups beyond 3× are uncommon given the current capabilities of LLMs in such tasks.

Multi-Turn Advantage We use a reward-to-go formulation to compute turn-level returns, assigning credit to each turn based on subsequent rewards in the interaction. This is natural for multi-turn kernel refinement: earlier turns influence later turns through accumulated code and environment feedback. Specifically, we define the return at turn t as

T

γt′−t Ri,t′, (2)

### ∑

Gi,t =

t′=t

where γ ∈ (0,1] is a discount factor and Ri,t is the per-turn reward. We fix γ = 1 in this work.

After computing Gi,t, we form turn-level advantages using a GRPO-style in-batch mean baseline. For each prompt (i.e., a fixed kernel task specification), we sample K independent

rollouts. At turn t, some rollouts may be invalid (e.g., masked out or terminated early); we denote Gt as the set of valid rollouts for a given prompt at turn t, and Nt = |Gt| ≤ K. We compute average returns within each turn group:

1

Gj,t, AGRPOi,t = Gi,t − G¯t, i ∈ Gt. (3)

G¯t =

### Nt ∑

j∈Gt

Self-Inclusion Issue in GRPO We identify that the in-group mean baseline in Eq. (3) suffers from self-inclusion: G¯t includes Gi,t itself for any i ∈ Gt. Since Gi,t depends on the current action yi,t through rewards from turn t onward (i.e., Ri,t:T), the baseline can become action-dependent, violating the standard requirement for an unbiased REINFORCE baseline (Sutton et al., 1999; Sutton & Barto, 2018). For the mean-centering form, this manifests as a biased policy-gradient estimator within each prompt–turn group:

1

E[gˆGRPO] = 1 −

Nt ∇θJ(θ). (4)

That is, the update is systematically shrunk by a factor depending on the (effective) group size. We provide a detailed derivation in Appendix A.

TRLOO To remove self-inclusion, we propose Turn-level REINFORCE Leave-One-Out (TRLOO), a multi-turn adaptation of the Leave-One-Out (Kool et al., 2019; Ahmadian et al., 2024) baseline. For each group Gt and sample i ∈ Gt with Nt > 1, define

1

G¯t(−i) =

Gj,t, ATRLOOi,t = Gi,t − G¯t(−i). (5)

Nt −1 ∑

j∈Gt, j̸=i

t−1 Gi,t − G¯t . Because G¯t(−i) excludes Gi,t, it does not depend on the current action yi,t under independent rollouts, yielding an unbiased turn-level advantage estimator for multi-turn RL.

Equivalently, ATRLOOi,t = NNt

Beyond unbiasedness, we claim TRLOO is beneficial for hard tasks with sparse positive rewards, where successful trajectories are rare. First, it avoids self-penalization in GRPO: under mean-centering, a rare high-return sample contributes to the G¯t and thus partially suppresses its advantage by subtracting the baseline. TRLOO excludes Gi,t from the baseline, so rare successes obtain a larger learning signal. Second, TRLOO is robust to varying group sizes. In multi-turn refinement, later turns may have fewer valid samples due to context limits or early termination, making N1

larger in Eq. (4). TRLOO removes this self-inclusion effect and preserves the correct scale across varying group sizes, improving sample efficiency when positive feedback is scarce.

t

###### 4.3 Empirical Results

We report empirical results of multi-turn RL training under different design choices. We use the Qwen3-8B-Base (Team, 2025) model after training with our cold-start data. During training, we sample 16 rollouts per question. And each turn in a trajectory becomes a training sample (Baronio et al., 2025). We evaluate on KernelBench (Ouyang et al., 2025) Level 21, whose difficulty is better matched to current LLM capabilities. Additional experimental details are provided in §6.1. We measure performance using the standard KernelBench metric Fast@1 and sample 8 candidates per question.

Our default setting uses TRLOO with a maximum of 3 turns, enables Hacking Check, and sets γ = 1.0 for return computation; we denote this run as w/ TRLOO. To isolate the effect of each component, we compare against the following variants: w/o Hacking Check disables the hacking-check module in KERNELGYM while keeping other settings unchanged; w/ Single

1Level 2 is not necessarily harder than Level 1 or Level 3 for current LLMs. Level 1 often requires outperforming highly optimized primitives such as GEMM, while Level 3 includes more complex network-level kernels.

TRLOO

20.0

+ MRS

+ MRS + PR

+ MRS + PR + PRS

17.5

Fast@1.2(%)

15.0

12.5

10.0

7.5

5.0

2.5

0 50 100 150 200 250 300

Training Step

1.2

TRLOO

+ MRS

+ MRS + PR

1.0

+ MRS + PR + PRS

Entropy

0.8

0.6

0.4

0 50 100 150 200 250 300

Training Step

- Figure 5: Left: Fast@1.2 at turn 3 over training steps. While MRS stabilizes training, profilingbased methods (PR and PRS) are required to significantly improve the stricter Fast@1.2 metric. Right: Entropy over training steps. While MRS improves training stability, PR and PRS further enhance stability on top of MRS. Additional training dynamics are shown in Figure 7.

Turn sets the maximum number of turns to 1; γ = 0 sets the discount factor to zero to ablate reward-to-go credit assignment; and w/ GRPO replaces TRLOO with GRPO.

- Figure 4 shows that TRLOO under the default setting achieves the best overall performance. The variant without the hacking check shows that the hacking check is necessary for effective RL training. Otherwise, training would saturate after only ∼ 50 steps. Compared to the GRPO variant, w/ TRLOO attains higher Fast@1 at every turn and exhibits a more stable learning curve, whereas GRPO saturates after roughly 200 training steps. Relative to singleturn training, multi-turn RL yields substantial gains as the number of turns increases, and it also improves first-turn quality. We attribute this to reward-to-go credit assignment: since early turns condition all subsequent refinements, they are incentivized to produce higher-quality intermediate kernels. This effect is corroborated by the γ = 0 ablation, which substantially degrades first-turn performance because the first-turn advantage no longer incorporates the impact on later interactions. Additionally, we observe that, unlike our model, the baseline AutoTriton fails to refine the kernel through multi-turn feedback.

#### 5 From Stability to Effectiveness: Overcoming Lazy Optimization

With KERNELGYM and TRLOO, long-term RL training becomes feasible, yet the “lazy optimization” issue persists as discussed in §2 and Figure 2. We systematically investigate this bottleneck through two hypotheses. Hypothesis 1: the saturation is caused by optimization instability arising from training–inference mismatch. Hypothesis 2: the optimization objective remains misaligned with meaningful speedup, incentivizing low-impact solutions.

###### 5.1 Hypothesis 1: Training Instability

Our initial speculation was that this premature saturation stemmed from training instability. Training–inference mismatch (Yao et al., 2025; Liu et al., 2025) is a fundamental challenge in RL for LLMs, where discrepancies between the rollout (inference) and training engines induce off-policy drift. Theoretically, this drift can lead to gradient variance and reward collapse, preventing the model from reaching higher performance peaks.

To investigate this, we monitor training dynamics via entropy, gradient norms, and perplexity (Figure 7). As observed, the multi-turn RL for kernel generation run exhibits excessively high values across these metrics, clearly indicating training instability. Following Liu et al. (2025), we adopt geometric Mismatch Rejection Sampling (MRS) to mitigate this drift. We compute the geometric-mean importance ratio

w = exp |T 1| ∑

t∈T

πtrain(at | st) πrollout(at | st)

log

, (6)

retaining samples only if w ∈ [0.999,1.001]. Additionally, we enforce a strict token-level veto: the entire sequence is rejected if the likelihood ratio πtrain/πrollout for any single token drops below 10−4.

As visualized in Figure 7, MRS successfully stabilizes the training dynamics. However, Figure 5 reveals a critical insight: while mismatch correction prevents early collapse (smoothing the learning curve), it does not fundamentally lift the performance ceiling of Fast@1.2. This indicates that while Hypothesis 1 effectively accounts for the training instability, addressing it alone does not fully resolve the performance saturation. Consequently, this directs our focus to the optimization objective itself.

###### 5.2 Hypothesis 2: Misaligned Objective

Given that improving stability alone is insufficient, we speculate that the standard reward signal fails to distinguish between trivial improvements and meaningful bottlenecks. While a kernel may be correct and achieve some speedup, it might still fail to address the true performance bottlenecks. To move from producing merely correct kernels to effective ones, we must make rewards bottleneck-aware.

Profiling-based Rewards (PR) A key symptom of this misalignment is the tendency for models to optimize trivial sub-operations (e.g., replacing a simple summation operation) without affecting the dominant bottlenecks in the computation. As demonstrated in the case study of lazy optimization versus better fusion (Figure 10), in the lazy optimization case, the model-generated kernel accounted for only 0.014% of the total CUDA execution time, indicating that the kernel optimization did not affect the main bottlenecks. In contrast, with better fusion, the model generated kernels that covered 86.15% of the total CUDA runtime, resulting in better and more meaningful speedup.

To formalize this intuition, we leverage the profiling toolkit in KERNELGYM to isolate the runtime contribution of the generated kernels (Tgenerated) from the overall CUDA execution time (Ttotal). We define the profiling ratio as:

Tgenerated Ttotal

PRi,t =

. (7)

Intuitively, PRi,t assigns higher credit when the candidate optimizes kernels that dominate the end-to-end runtime. We then augment the per-turn reward with this signal (applied only to correct kernels):

Ri,t = C(yi,t) + C(yi,t) · speedupi,t +C(yi,t) · PRi,t . (8) This encourages the model to focus on kernel optimizations that contribute significantly to performance, explicitly driving learning toward optimizations with larger real speedup. Besides, since PRi,t is bounded in [0,1], the speedup term naturally dominates, preventing the model from maximizing coverage via inefficient code.

Profiling-based Rejection Sampling (PRS) Even with bottleneck-aware rewards, the exploration process can still be dominated by a high volume of low-impact (“lazy”) samples. To further filter the training distribution, we introduce profiling-based rejection sampling (PRS). For each sample (i, t), we retain it with probability:

PRi,t −τ s

pi,t = clip

, 0, 1 , (9)

where τ is a cutoff threshold and s controls the softness of the filter. In our experiments, we fix τ = 0.3 and s = 0.1. We ablate the design choice of PRS in Appendix D.

###### 5.3 Empirical results

- Figure 5 confirms this staged diagnosis. MRS improves training stability but does not, by itself, raise the Fast@1.2 ceiling. Furthermore, adding PR and PRS substantially lifts Fast@1.2. And the stability is even further improved by PR and PRS as shown in Figure 5 (Right) and Figure 7.

- Table 1: Performance across levels under different Fast thresholds. As we treat samples with reward hacking as incorrect, our evaluation is more strict than the original Kernelbench. ∗The Cold-Start-8B refers to Qwen3-8B-Base after training with our cold-start data. We contain our DR. KERNEL-14B with sequential test-time scaling (STTS) using context management 6.3 as reference. DR. KERNEL-14B-STTS† reports the results from selecting the best turn across all history turns.

LEVEL1 LEVEL2 LEVEL3 Fast1 Fast1.2 Fast1.5 Fast2 Fast1 Fast1.2 Fast1.5 Fast2 Fast1 Fast1.2 Fast1.5 Fast2 GPT-5 19.5 16.5 12.5 11.0 46.7 28.6 13.1 3.0 21.0 12.0 4.0 2.0

Model

Claude-4.5-Sonnet 15.5 13.5 11.0 8.5 50.0 26.7 9.2 1.8 21.0 11.0 5.0 4.0 Deepseek-V3.2-Thinking 7.5 5.5 4.5 4.0 11.0 6.5 2.5 0.5 2.0 1.0 0.0 0.0

GLM-4.7 19.4 17.2 13.1 10.4 30.0 20.5 8.5 3.5 5.0 2.0 2.0 2.0 Qwen3-8B 5.8 4.8 4.1 3.4 13.0 5.6 2 1.1 5.7 0.2 0.0 0.0

Qwen3-32B 6.1 4.9 4.3 4 14.0 9.4 2.4 0.2 3.5 0.0 0.0 0.0 Qwen3-Coder-A30BA3 6.0 5.2 5.1 3.8 12.6 5.0 1.5 0.3 7.0 1.0 0.0 0.0

AutoTriton 4.5 3.6 2.8 2.1 30.6 9.2 2.6 0.5 7.5 0.0 0.0 0.0 Cold-Start-8B∗ 7.5 6.6 5.0 4.3 8.8 5.6 1.8 0.4 0.5 0.0 0.0 0.0

DR. KERNEL-8B 15.9 12.8 10.9 8.4 46.0 20.0 5.0 1.5 10.8 1.0 0.0 0.0 DR. KERNEL-14B 20.3 16.9 13.2 11.6 49.2 25.6 7.4 2.1 8.8 1.2 0.2 0.0

DR. KERNEL-14B-STTS 24.1 18.8 15.3 12.8 59.8 31.6 9.6 3.0 17.1 3.0 0.2 0.0 DR. KERNEL-14B-STTS† 39.3 25.1 20.4 17.6 80.9 47.8 23.6 12.5 29.8 7.3 0.5 0.0

#### 6 Experiments

###### 6.1 Setup

We evaluate on KernelBench (Ouyang et al., 2025) across all three levels. We follow the official Torch backend in Kernelbench and their implementations of correctness and speedup measurement. We furhter conduct hacking checks when we evaluate kernels. Therefore, our evaluations are stricter than the original Kernelbench. We follow the standard metrics in Kernelbench Fast@p, p = [1,1.2,1.5,2], where it is the ratio of samples are both correct and achieve ×p speedup. Both the evaluations and training are conducted on NVIDIA H100. We experiment with Qwen3-8B-Base and Qwen-14B-Base. We sample each question for 8 samples with 3 max turns. We set the max tokens as 32768 and the max generated tokens per turn as 8192. To keep a fair comparison, we report results at turn 3 for our models and for most baselines, as turn 3 typically yields the best average performance. For baselines whose best average performance is achieved at an earlier turn, we instead report their best-performing turn.

We first perform cold-start supervised fine-tuning on our collected 8K 5-turn trajectories, using a learning rate of 1 × 10−6, batch size 256, for 4 epochs. After cold-start training, we run multi-turn RL on the RL queries from cudaLLM (CudaLLM Team, 2025). The queries used for both SFT and RL cover basic PyTorch operators, Transformer components, more complex compositions, and LLM-generated tasks. For RL, we use a learning rate of 1 × 10−6, train for 300 rollout steps, sample 16 rollouts per prompt with max turns to 3 and a rollout batch size of 16. We implement the training pipeline with asynchronous inference.

###### 6.2 Results

Main Results. Table 1 summarizes performance across KernelBench levels under different Fast thresholds. We compare DR. KERNEL against AutoTriton (Li et al., 2025), open-source models with strong coding/reasoning abilities, and proprietary models. AutoTriton is a very relevant baseline since it is released, Triton-based, and reports Fast@p.

Overall, DR. KERNEL achieves the strongest performance among open-source baselines and is competitive with frontier models on Level 1 and Level 2. In particular, DR. KERNEL-14B attains high Fast@1.2 on both Level 1 and Level 2, indicating that it improves not only any speedup (Fast@1) but also the stricter and meaningful speedup. This contrasts with prior

##### Last-turn Performance

30

Fast@1.2(%)

Context Overflow

25

20

Vanilla Extrapolation

15

Context Management

10

1 2 3 4 5 6 7 8 9 10 11 12 13 14

Turn

###### Best-of-History Performance

50

Fast@1.2(%)

40

Context Overflow

30

20

Vanilla Extrapolation

Context Management

10

1 2 3 4 5 6 7 8 9 10 11 12 13 14

Turn

- Figure 6: Test-time scaling with DR. KERNEL-14B. We report Fast@1.2 for the last turn (left) and best-of-history (right). Vanilla extrapolation increases the number of turns by appending all previous turns to the prompt. Context management stores the full history externally, but only includes the top-w turns (by reward, w=4) in the prompt to fit context length.

approaches such as AutoTriton, which achieves a strong Fast@1 on Level 2 but delivers substantially smaller gains under stricter thresholds (e.g., Fast@1.2).

Comparing against our cold-start model, DR. KERNEL shows that multi-turn RL contributes materially to performance gains, especially on the stricter metrics (e.g., Fast@1.2 improves from 5.6 → 20.0 on Level 2). While DR. KERNEL improves Level 3 Fast@1 relative to opensource baselines, performance at stricter thresholds on Level 3 remains limited, suggesting that further scaling of training data and model capacity is likely required to close the gap to frontier models on the hardest subset.

Finally, test-time scaling further amplifies DR. KERNEL’s performance. With sequential testtime scaling (STTS) via context management (§6.3), DR. KERNEL-14B-STTS boosts Fast@1.2 from 16.9 → 18.8 on Level 1 and from 25.6 → 31.6 on Level 2; with best-turn selection across history turns (DR. KERNEL-14B-STTS†), Fast@1.2 further rises to 25.1 (Level 1) and 47.8 (Level 2), surpassing frontier models such as GPT-5 and Claude-4.5-Sonnet on these metrics. STTS also materially improves Level 3 (Fast@1.2: 1.2 → 3.0/7.3), narrowing the gap and partially compensating for the hardest subset; given the substantial model-size disparity to frontier models, these test-time scaled results remain informative despite leveraging TTS.

Analysis We conduct a comprehensive analysis that includes reward hacking ratio (Appendix C) and case studies (Appendix E.2). Please refer to these sections for details.

###### 6.3 Test-time Scaling for Kernel Generation

We study sequential test-time scaling (STTS) by increasing the number of multi-turn refinement steps at inference time. We use DR. KERNEL-14B with a maximum context length of 32,768 tokens and evaluate two strategies: vanilla extrapolation and context management. We report two metrics: (i) Last-turn Fast@1.2, i.e., the Fast@1.2 achieved by the final generated kernel at turn T; and (ii) Best-of-history Fast@1.2, i.e., the best Fast@1.2 obtained among turns {1, . . . , T}. With STTS, our model even outperforms GPT-5/Claude-4.5-Sonnet in Kernelbench level-2 subset.

Vanilla extrapolation We directly extrapolate the number of refinement turns beyond training (trained with up to 3 turns) by appending the entire interaction history to the prompt at each turn. Figure 6 shows that increasing turns initially improves both last-turn and best-of-history performance. However, as T grows, prompt length scales linearly and may approach the context limit, which can degrade performance.

Context management To scale T without unbounded prompt growth, we store all turns in an external memory and maintain a fixed in-context window. Concretely, at each turn we select the top-w turns with the highest rewards from the accumulated history and only

- Table 2: Fast performance across levels under different thresholds (evaluated under torch.compile). Since torch.compile provides a strong optimized baseline, Fast@1 remains meaningful (unlike eager mode, where trivial “lazy” optimizations can inflate Fast@1).

LEVEL1 LEVEL2 LEVEL3 Fast1 Fast1.2 Fast1.5 Fast2 Fast1 Fast1.2 Fast1.5 Fast2 Fast1 Fast1.2 Fast1.5 Fast2 GPT-5 18.6 8.0 6.5 5.5 22.1 3.6 1.5 1.0 14.0 4.0 3.0 1.0

Model

Claude-4.5-Sonnet 10.0 2.2 2.0 1.8 20.5 3.0 0.0 0.0 12.0 3.5 0.5 0.0

DR. KERNEL-8B 16 3.0 1.5 8.4 20.6 0.8 0.0 0.0 7.2 2.3 0.0 0.0 DR. KERNEL-14B 17.8 5 3.3 2.5 23.5 1.9 0.0 0.0 9.2 3 0.0 0.0

append these selected turns as the prompt history for generating the next turn (we use w=4). As shown in Figure 6, context management yields consistently stronger best-ofhistory performance and continues to improve as turns scale. Last-turn performance can be slightly lower at small T, since vanilla extrapolation can condition on the full history, but as T increases, context management becomes strictly more reliable and surpasses the best performance achievable by vanilla extrapolation.

###### 6.4 Results on torch.compile

torch.compile is an advanced PyTorch feature that captures PyTorch programs into a compiled computation graph and optimizes execution via operator fusion, code generation, and scheduling, among other compiler passes. While most prior work evaluates modelgenerated kernels only under Torch eager execution, we further validate both our models and frontier models under torch.compile, providing a substantially stronger and more practical assessment of speedups.

As shown in Table 2, DR. KERNEL remains effective under the more challenging torch.compile setting and stays competitive with frontier models across the three levels. Because torch.compile already applies compiler optimizations, the headroom for additional gains is smaller than in eager mode; consequently, the absolute Fast@p numbers are generally lower for all models, including both ours and frontier models. Importantly, Fast@1 under torch.compile is also a stricter target: trivial “lazy” changes that may yield marginal improvements in eager execution typically do not surpass the optimized compiled baseline. These results suggest that our training methods generalize beyond eager-mode artifacts, and further scaling of data and model capacity is a promising direction to obtain larger improvements even on top of torch.compile.

#### 7 Conclusion

We investigate RL for Triton kernel generation and identify the challenges of reward hacking and lazy optimization, which are common in prior works. To address these, we develop the KERNELGYM environment with hacking checks and profiling tools, propose unbiased multiturn RL methods incorporating mismatch correction, profiling-based rewards/rejection sampling, and explore sequential test-time scaling. Our approach enhances meaningful speedup, advancing RL training for kernel generation.

#### 8 Limitations and Future Work

While our approach demonstrates progress in RL-based training for Triton kernel generation, we acknowledge several areas that warrant further investigation.

Data Scaling and Pre-training From a data perspective, resource constraints limited our supervised fine-tuning (SFT) phase to 8,000 cold-start samples. Given the relative scarcity of high-quality kernel programming data in the pre-training corpora of current Large Language Models (LLMs), our results suggest that the ”data floor” for this domain is quite

high. Future work could involve larger-scale data collection to facilitate domain-specific pre-training or continual pre-training (middle-training), which would also provide a more robust foundation for subsequent RL optimization.

Model Capacity Our observations with DR. KERNEL-8B and DR. KERNEL-14B confirm that larger models possess a superior capacity for kernel generation. This scaling effect is particularly critical in Reinforcement Learning, where the model must rely on its own generations to explore the solution space and update its policy. We expect that migrating these methods to even larger parameter scales will accelerate the development.

Path Toward Production-Ready Automation Although our methods achieve performance improvements that rival or exceed frontier models, the field remains in an exploratory stage. While current models can generate high-quality code snippets, they are not yet capable of fully autonomous, end-to-end kernel generation for production environments. We hope that our contributions, specifically the KERNELGYM environment and the DR. KERNEL training framework, serve as a catalyst for future research.

#### References

Arash Ahmadian, Chris Cremer, Matthias Gall´e, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Ust¨ un,¨ and Sara Hooker. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms, 2024. URL https://arxiv.org/ abs/2402.14740.

Carlo Baronio, Pietro Marsella, Ben Pan, Simon Guo, and Silas Alberti. Kevin: Multi-turn rl for generating cuda kernels, 2025. URL https://arxiv.org/abs/2507.11948.

CudaLLM Team. CudaLLM: Training language models to generate high-performance cuda kernels. https://github.com/ByteDance-Seed/cudaLLM, 2025. GitHub repository. Latest commit Aug 18, 2025 (commit 7d280a6). Accessed 2026-01-20.

Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations (ICLR), 2024.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning, 2025. URL https://arxiv.org/abs/2503.09516.

Wouter Kool, Herke van Hoof, and Max Welling. Buy 4 REINFORCE samples, get a baseline for free!, 2019. URL https://openreview.net/forum?id=r1lgTGL5DE.

Shangzhan Li, Zefan Wang, Ye He, Yuxuan Li, Qi Shi, Jianling Li, Yonggang Hu, Wanxiang Che, Xu Han, Zhiyuan Liu, et al. Autotriton: Automatic triton programming with reinforcement learning in llms. arXiv preprint arXiv:2507.05687, 2025.

Jiacai Liu, Yingru Li, Yuqian Fu, Jiawei Wang, Qian Liu, and Yu Shen. When speed kills stability: Demystifying RL collapse from the training-inference mismatch, September

2025. URL https://richardli.xyz/rl-collapse.

Alexander Novikov, Ngˆan Vu,˜ Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco J. R. Ruiz, Abbas Mehrabian, M. Pawan Kumar, Abigail See, Swarat Chaudhuri, George Holland, Alex Davies, Sebastian Nowozin, Pushmeet Kohli, and Matej Balog. Alphaevolve: A coding agent for scientific and algorithmic discovery, 2025. URL https: //arxiv.org/abs/2506.13131.

Anne Ouyang, Simon Guo, Simran Arora, Alex L Zhang, William Hu, Christopher Re, and Azalia Mirhoseini. Kernelbench: Can LLMs write efficient GPU kernels? In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum? id=yeoN1iQT1x.

Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. URL http://incompleteideas.net/book/the-book-2nd. html.

Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In S. Solla, T. Leen, and K. Muller¨ (eds.), Advances in Neural Information Processing Systems, volume 12. MIT Press, 1999. URL https://proceedings.neurips.cc/paper files/paper/1999/file/ 464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf.

Qwen Team. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388. Lei Wang, Yu Cheng, Yining Shi, Zhengju Tang, Zhiwen Mo, Wenhao Xie, Lingxiao Ma,

Yuqing Xia, Jilong Xue, Fan Yang, and Zhi Yang. Tilelang: A composable tiled programming model for ai systems, 2025. URL https://arxiv.org/abs/2504.17577.

Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese. Browsecomp: A simple yet challenging benchmark for browsing agents, 2025. URL https: //arxiv.org/abs/2504.12516.

Jiin Woo, Shaowei Zhu, Allen Nie, Zhen Jia, Yida Wang, and Youngsuk Park. Tritonrl: Training llms to think and code triton without cheating, 2025. URL https://arxiv.org/ abs/2510.17891.

Feng Yao, Liyuan Liu, Dinghuai Zhang, Chengyu Dong, Jingbo Shang, and Jianfeng Gao. Your efficient rl framework secretly brings you off-policy rl training, August 2025. URL https://fengyao.notion.site/off-policy-rl.

Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci, Vinod Grover, Arvind Krishnamurthy, and Luis Ceze. Flashinfer: Efficient and customizable attention engine for llm inference serving. arXiv preprint arXiv:2501.01005, 2025. URL https://arxiv.org/abs/2501.01005.

#### A Derivation: Self-Inclusion Causes a Scaled Gradient in GRPO

Setup For a given prompt question, fix a turn group Gt with |Gt| = N. For each rollout i ∈ Gt, let si,t ≜ (hi,:t−1, xi,t) and yi,t ∼ πθ(· | si,t), where hi,:t−1 is all turn history before t, xi,t is the prompt with environmental feedback in turn t. Let Gi,t be the reward-to-go return at turn t and define the in-group mean G¯t = N1 ∑j∈Gt Gj,t. GRPO mean-centering uses AGRPOi,t = Gi,t − G¯t.

For any random variable Z that does not depend on the sampled action yi,t (conditional on si,t),

Eyi,t∼πθ(·|si,t)[∇θ log πθ(yi,t | si,t) · Z] = Z · Eyi,t[∇θ log πθ(yi,t | si,t)] = 0, (10) where we use Ey[∇θ log πθ(y | s)] = ∇θ πθ(y | s) dy = ∇θ1 = 0. Policy Gradient in GRPO Consider the per-turn GRPO estimator within this group:

1

∇θ log πθ(yi,t | si,t) (Gi,t − G¯t) . (11)

N ∑

gˆ =

i∈Gt

Taking expectation and expanding the baseline term,

1

1

N ∑

E[gˆ] =

E[∇θ log πθ(yi,t | si,t) Gi,t] −

i

1

1

N ∑

E[∇θ log πθ(yi,t | si,t) Gi,t] −

=

i

E[∇θ log πθ(yi,t | si,t) G¯t]

N ∑

i

1

N ∑

N ∑

E ∇θ log πθ(yi,t | si,t)

Gj,t . (12)

i

j

For j ̸= i, Gj,t is independent of yi,t under independent rollouts, hence we can apply the score-function identity in Eq. (10) to obtain

E ∇θ log πθ(yi,t | si,t) Gj,t = 0 (j ̸= i). Therefore only the j = i term remains:

1

1

1 N

N ∑

N ∑

E[gˆ] =

E[∇θ log πθ(yi,t | si,t) Gi,t] −

E ∇θ log πθ(yi,t | si,t)

Gi,t

i

i

1 N ·

1

N ∑

E[∇θ log πθ(yi,t | si,t) Gi,t] . (13)

= 1 −

i

The last term corresponds to the unbiased REINFORCE gradient for this prompt–turn group (up to the outer averaging convention), hence the GRPO mean baseline induces a shrinkage

factor (1 − N1 ) due to self-inclusion.

Leave-one-out removes self-inclusion. Using the leave-one-out baseline G¯t(−i) =

1

N−1 ∑j̸=i Gj,t, the baseline term no longer contains Gi,t and is independent of yi,t under independent rollouts. By Eq. (10), the expected contribution of the baseline term becomes zero, yielding an unbiased estimator.

#### B Training Dynamics

In this section, we analyze the training dynamics. As shown in Figure 7, multi-turn RL for kernel generation exhibits significant instability, characterized by elevated entropy, perplexity, and gradient norms, even when using unbiased advantage estimation. Incorporating mismatch correction (i.e., Mismatch Rejection Sampling) effectively stabilizes the training process. Furthermore, the introduction of PR and PRS provides additional stability, leading to smoother training.

1.2

TRLOO

+ MRS

+ MRS + PR

1.0

+ MRS + PR + PRS

Entropy

0.8

0.6

0.4

0 50 100 150 200 250 300

Training Step

TRLOO

+ MRS

2.8

+ MRS + PR

+ MRS + PR + PRS

2.6

2.4

VLLM-PPL

2.2

2.0

1.8

1.6

1.4

0 50 100 150 200 250 300

Training Step

0.25

TRLOO

+ MRS

+ MRS + PR

+ MRS + PR + PRS

0.20

GradientNorm

0.15

0.10

0.05

0 50 100 150 200 250 300

Training Step

TRLOO

+ MRS

2.8

+ MRS + PR

+ MRS + PR + PRS

2.6

2.4

FSDP-PPL

2.2

2.0

1.8

1.6

1.4

0 50 100 150 200 250 300

Training Step

- Figure 7: The training dynamics of TRLOO, TRLOO + Mismatch Rejection Sampling (MRS), TRLOO + MRS + Profiling-based Reward (PR) and TRLOO + MRS + PR + Profiling-based Rejection Sampling (PRS). We analyze the training dynamics via the lens of entropy, gradient norm, VLLM-PPL, and FSDP-PPL, which are also monitored by Liu et al. (2025).

C Hacking Ratio

We analyze the changes in the hacking ratio during training for DR. KERNEL-14B on the Kernelbench level-2 subset. With the hacking check in KERNELGYM, the hacking ratio steadily decreases from approximately 20% at the start to around 3%. We also examine the hacking ratio on the Kernelbench level-1 subset; compared to AutoTriton, which exhibits a hacking ratio of around 10%, DR. KERNEL-14B experiences hacking in only 1.7% of cases.

0 50 100 150 200 250 300

Step

5.0

7.5

10.0

12.5

15.0

17.5

20.0

HackingRatio(%)

Dr. Kernel-14B

- Figure 8: The hacking ratio of DR. KERNEL-14B. With the hacking check in KERNELGYM, the hacking ratio decreases from approximately 20% at the start to only around 3% in the Kernelbench level-2 subset.

#### D Ablations of PRS

We perform an ablation study to evaluate the effect of the softness sampling design choice in PRS. Our default setting for DR. KERNEL is the combination of TRLOO, MRS, PR, and

PRS, and we compare it to a variant, DR. KERNELw/o s in PRS, where kernels with PR ≥ τ are kept directly, and kernels with PR < τ are discarded outright. In contrast, in the variant with softness, kernels with PR ∈ [0.3,0.4) are probabilistically retained, based on PR, τ, and s. In this ablation, we set τ = 0.3 as the fixed configuration.

As shown in Figure 9, DR. KERNEL outperforms the variant w/o s in PRS. The variant still performs better and shows improved stability compared to the baseline w/o PR & PRS. This ablation demonstrates that the softness sampling mechanism enhances the robustness of PRS, helping it balance correctness and meaningful speedup by selectively retaining relatively low-quality samples that might otherwise be discarded. This ability to retain such samples contributes to a more effective exploration-exploitation trade-off and facilitates more stable kernel generation over time.

20

Dr. Kernel

Dr. Kernel w/o s in PRS

18

Dr. Kernel w/o PR & PRS

16

Fast@1.2(%)

14

12

10

8

6

4

0 50 100 150 200 250 300

Training Step

- Figure 9: Comparison of DR. KERNEL with and without softness in PRS. DR. KERNEL outperforms the variant w/o s in PRS, while both variants show better stability compared to the baseline w/o PR & PRS.

#### E Cases Studies

###### E.1 Lazy Optimization vs. Better Fusion

We show the cases of profiling feedback from lazy optimization and better fusion cases. As shownin Figure 10, in the lazy optimization case, where only a trivial summation operation is replaced, the model-generated kernel accounts for only 0.014% of the total CUDA execution time. In contrast, with better fusion, the model generates more meaningful kernels, achieving significantly better speedup and increasing the CUDA runtime fraction to 86.15% of the total runtime.

###### E.2 Trajectories from DR. KERNEL

We conduct qualitative studies on DR. KERNEL-14B using multi-turn inference across three turns. As shown in Figure 11, in the first turn, DR. KERNEL identifies the LayerNorm operation and generates a kernel by fusing different operations into a single Triton kernel. After receiving feedback from KERNELGYM, it recognizes that certain configurations, such as block size, number of wraps, and stages, are under-explored, and applies ”autoconfig” to select better configurations. By the third turn, DR. KERNEL identifies the optimal configuration based on the running hardware, further improving the performance by adjusting the configuration in ”autoconfig.” This case demonstrates that after RL training, DR. KERNEL-14B handles basic kernel writing and also adapts effectively to environment feedback, showcasing its ability to improve over time.

###### E.3 Case for Better Fusion

We further study the case where the lazy optimization issue is alleviated, as shown in the profiling summary in Figure 10 (right). In contrast to the lazy optimization case, the example generated by DR. KERNEL-8B addresses lazy optimization by converting most operations

###### Profiling_Lazy_Optimization.json

###### Profiling_Better_Fusion.json

Server feedback (status/metrics/errors): {

Server feedback (status/metrics/errors): {

"task_id": "parallel_task_000336_b4243221", "status": "completed", "compiled": true, "correctness": true, ”hacking": false, "reference_runtime": 47.4, "kernel_runtime": 47.0, "speedup": 1.01, "metadata": {

… "reference_runtime": 47.5, "kernel_runtime": 22.8, "speedup": 2.08, "metadata": {

"hardware": "NVIDIA H100 80GB HBM3", "profiling": {

"kernels": [ {

… "hardware": "NVIDIA H100 80GB HBM3", "profiling": {

"name": "max_pool3d_k2s2_per_channel_kernel", "cuda_time_us": 96169.16, "cpu_time_us": 0.0, "count": 5

"kernels": [ {

}, {

"name": "void convolveNd_dgrad_float_engine<float, 3…", "cuda_time_us": 27522.25, "cpu_time_us": 0.0, "count": 5

"name": "sm90_xmma_dgrad_implicit_gemm_...", "cuda_time_us": 6663.94, "cpu_time_us": 0.0, "count": 5

}, {

}, {

"name": "void cudnn::engines_precompiled::setTensor5d…", …

"name": "max_pool3d_k3s3_per_channel_kernel", …

}, … {

}, … {

"name": "sum_channels_kernel", "cuda_time_us": 34.8, "cpu_time_us": 0.0, "count": 5

"name": "sum_channels_kernel", …

], "total_cuda_time_us": 118674.11, "total_cpu_time_us": 243259.05, "kernel_count": 8, "memory_stats": {

}

], "total_cuda_time_us": 237500.34, "total_cpu_time_us": 479674.12, "memory_stats": {

… }

... }

}, … "triton_profiler_matches": "[…'max_pool3d_k2s2_per_channel_kernel grid…’,

}, … "triton_profiler_matches": "[…,'sum_channels_kernel grid…']", "custom_kernel_cuda_time_in_profiling_us": 34.82,

'max_pool3d_k3s3_per_channel_kernel…', 'sum_channels_kernel…’]”, "custom_kernel_cuda_time_in_profiling_us": 102242.61, "total_kernel_run_time_in_profiling_us": 118674.10, "custom_kernel_cuda_time_coverage": "Custom kernel CUDA time: 102242.61us / Total time:

"total_kernel_run_time_in_profiling_us": 237500.34, "custom_kernel_cuda_time_coverage": "Custom kernel CUDA time: 34.82us / Total time:

237500.35us, Coverage: 0.014%", …

118674.11us, Coverage: 86.15%", …

- Figure 10: Profiling feedback for cases with lazy optimization and better fusion. We omit some profiling items for brevity. The original code for Profiling Lazy Optimization.json is shown in Figure 2 (right), and the original code for Profiling Better Fusion.json can be found in Appendix E.3. In the lazy optimization case, where only a trivial summation operation is replaced, the model-generated kernel accounts for only 0.014% (i.e PR = 0.00014) of the total CUDA execution time. In contrast, with better fusion, the model generates more meaningful kernels, achieving significantly better speedup and increasing the CUDA runtime fraction to 86.15% i.e PR = 0.8615 of the total runtime.

###### 1st Turn Trajectory (2)

###### Original_Torch_Implementation.py

###### 1st Turn Trajectory (1)

while start < D: offs = start + tl.arange(0, BLOCK_N) mask = offs < D ptrs = x_ptr + row * stride_b + offs * stride_d vals = tl.load(ptrs, mask=mask, other=0.0) vals_f32 = vals.to(tl.float32) block_sum = tl.sum(tl.abs(vals_f32), axis=0) sum_val += block_sum start += BLOCK_N

Thinking, analysis, and plan

import torch import torch.nn as nn

- - What the original PyTorch code does
- - It computes mean(abs(x), dim=1, keepdim=True) for a 2D tensor x of shape [B, D]. …
- - Where the time goes in the PyTorch version …
- - So you have multiple passes over memory and intermediate tensors, which for large [B, D] are

memory-bandwidth bound.

- - What we can do better with Triton
- - Fuse abs, reduction, and scaling into a single kernel that:
- - Reads each element of a row once to accumulate the L1 norm (sum of abs).
- - Computes the inverse mean (1/mean) from that sum.
- - Makes a second pass over the row to write out x / mean.
- - This removes materialization of abs(x) and mean, and reduces kernel launch overhead.
- - Memory access is simple and coalesced: process one row per program, iterate over columns in blocks.
- - Kernel structure
- - Grid: one program per row (grid = (B,)).
- - Loop over columns in chunks of BLOCK_N (e.g., 128/256/512/1024). …
- - Tuning
- - BLOCK_N: choose based on D (next power-of-two up to a cap), e.g., 128–1024.
- - num_warps: 4 or 8 is typical; 4 is fine for many cases.
- - This is a simple, robust choice; autotuning is possible but overkill here.
- - Edge cases and correctness …

class Model(nn.Module): """ Simple model that performs L1 normalization. """ def __init__(self): """ Initializes the L1 normalization layer. """ super(Model, self).__init__()

# mean = sum / D ; inv_mean = 1 / mean # guard D > 0 (host should ensure); compute in f32 inv_mean = 1.0 / (sum_val / D)

# Second pass: y = x * inv_mean start = 0 while start < D:

offs = start + tl.arange(0, BLOCK_N) mask = offs < D in_ptrs = x_ptr + row * stride_b + offs * stride_d out_ptrs = y_ptr + row * stride_b + offs * stride_d vals = tl.load(in_ptrs, mask=mask, other=0.0) vals_f32 = vals.to(tl.float32) out_vals = vals_f32 * inv_mean # cast back to input dtype out_cast = out_vals.to(vals.dtype) tl.store(out_ptrs, out_cast, mask=mask) start += BLOCK_N

def forward(self, x: torch.Tensor) -> torch.Tensor: """ Applies L1 normalization to the input tensor.

def _choose_block_n(D: int) -> int: # next power-of-two up to 1024, minimum 64 if D <= 64:

Args: x (torch.Tensor): Input tensor of shape (..., dim, ...).

return 64 p2 = 1 << (int(math.ceil(math.log2(max(1, D))))) return min(max(64, p2), 1024)

Code (Triton-optimized L1 row normalization) ```python …

Returns: torch.Tensor: Output tensor with L1 normalization applied, same shape as input. """ return x / torch.mean(torch.abs(x), dim=1, keepdim=True)

class ModelNew(nn.Module): def __init__(self): super().__init__()

@triton.jit def _l1_norm_rows_kernel(

- x_ptr, # *const T
- y_ptr, # *T B: tl.constexpr, # number of rows D: tl.constexpr, # number of cols stride_b: tl.constexpr, # row stride in elements stride_d: tl.constexpr, # col stride in elements BLOCK_N: tl.constexpr

def forward(self, x: torch.Tensor) -> torch.Tensor: # Fallback to PyTorch if not on CUDA or Triton not available if (not x.is_cuda) or (not _HAS_TRITON):

# match original: mean over dim=1, keepdim=True, then divide

- return x / torch.mean(torch.abs(x), dim=1, keepdim=True)

assert x.dim() == 2, f"Expected 2D tensor, got shape {tuple(x.shape)}" B, D = x.shape if D == 0:

raise ValueError("Normalization dimension D must be > 0") if not x.is_contiguous():

- x = x.contiguous()
- y = torch.empty_like(x)

BLOCK_N = _choose_block_n(D) grid = (B,) _l1_norm_rows_kernel[grid](

x, y, B, D, x.stride(0), x.stride(1), BLOCK_N=BLOCK_N, num_warps=4, # reasonable default

)

- return y

batch_size = 32768 # choose dim so total <2^31 dim = 65535

):

# program id = row index row = tl.program_id(0) # guard: if grid is larger than B (shouldn't happen with correct grid), early return if row >= B: return

def get_inputs(): x = torch.rand(batch_size, dim) return [x]

# Accumulate sum of abs over this row in float32 sum_val = tl.zeros((), dtype=tl.float32)

# First pass: compute L1 norm = sum |x| start = 0

def get_init_inputs(): return []

```

###### 2nd Turn Trajectory (1)

###### 2nd Turn Trajectory (2)

###### 3rd Turn Trajectory

Deep dive analysis, thinking, and optimization plan

Deep analysis, reasoning, and optimization plan

# First pass: compute sum |x| in f32 sum_val = tl.zeros((), dtype=tl.float32) start = 0 while start < D:

- 1) What the original PyTorch model does

- ...

- 2) Performance characteristics of the PyTorch version

- …

- 3) First Triton version (your feedback context)

- - …
- - This removes intermediate tensors and kernel launches, gives a small speedup (~1.04x on H100 for B=32768, D=65535).

- 4) Where the time is spent and how to improve

- …

- 5) Optimization plan

- - Autotune BLOCK_N, num_warps, num_stages for the given D.
- - Provide a small set of configs that cover 128–1024 block sizes, warps 4–8, stages 2–4.
- - Key on D so the best config is cached per shape.
- - Minor micro-optimizations:
- - Compute inv_mean as D / sum (avoid division earlier).
- - Keep accumulation in float32; cast on store.
- - Ensure contiguous layout.
- - Use while loop over cols; it’s fine and compiles well.
- - Keep API and correctness identical.

- 6) Correctness and edge cases

- …

- 7) Expected result

- 1) Restating the operation

- …

- 2) Performance characteristics and bottlenecks

- …

- 3) What we did previously and results

- - Fused kernel with two passes, float32 accumulation, coalesced access.
- - Added autotuning over BLOCK_N, num_warps, num_stages.
- - Achieved ~1.21x speedup on H100 for B=32768, D=65535.

- 4) Where we can squeeze more performance

- - Increase instruction-level parallelism and memory pipeline depth:
- - Larger BLOCK_N (up to 2048/4096) can improve vectorization and reduce loop iterations for very large D.
- - Increase num_warps (up to 16) and num_stages (up to 5) to better hide memory latency on H100.
- - Minor arithmetic tweak:
- - Compute inv_mean = D / sum to avoid an earlier division.
- - Keep the kernel simple to allow the compiler to generate efficient code (contiguous, unit stride along columns).

- 5) Design decisions

- ...

- 6) Correctness and edge cases

- …

- 7) Expected outcome

offs = start + tl.arange(0, BLOCK_N) mask = offs < D ptrs = x_ptr + row * stride_b + offs * stride_d vals = tl.load(ptrs, mask=mask, other=0.0) vals_f32 = vals.to(tl.float32) block_sum = tl.sum(tl.abs(vals_f32), axis=0) sum_val += block_sum start += BLOCK_N

# Compute inv_mean = D / sum ; avoid early division inv_mean = (D * 1.0) / sum_val

# Second pass: y = x * inv_mean start = 0 while start < D:

offs = start + tl.arange(0, BLOCK_N) mask = offs < D in_ptrs = x_ptr + row * stride_b + offs * stride_d out_ptrs = y_ptr + row * stride_b + offs * stride_d vals = tl.load(in_ptrs, mask=mask, other=0.0) vals_f32 = vals.to(tl.float32) out_vals = vals_f32 * inv_mean out_cast = out_vals.to(vals.dtype) tl.store(out_ptrs, out_cast, mask=mask) start += BLOCK_N

- On H100, expanding the search space (BLOCK_N up to 2048, warps up to 16, stages up to 5) should find a configuration that is equal or faster than the previous one, improving the speedup while preserving correctness. Final improved code (Triton-optimized, with expanded autotuning)

- … Improved Triton implementation with autotuning ```python …

class ModelNew(nn.Module): def __init__(self): super().__init__()

```python import torch import torch.nn as nn

def forward(self, x: torch.Tensor) -> torch.Tensor: … y = torch.empty_like(x) grid = (B,) _l1_norm_rows_kernel[grid](

try: import triton import triton.language as tl _HAS_TRITON = True

@triton.autotune( configs=[

triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2), triton.Config({'BLOCK_N': 256}, num_warps=4, num_stages=2), triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=3), triton.Config({'BLOCK_N': 1024}, num_warps=8, num_stages=4),

except Exception: _HAS_TRITON = False

x, y, B, D, x.stride(0), x.stride(1), # Meta-parameters are selected by autotune; no need to pass BLOCK_N/num_warps here

], key=['D'], # tune per column size

@triton.autotune( configs=[

) @triton.jit def _l1_norm_rows_kernel(

) return y

triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2), triton.Config({'BLOCK_N': 256}, num_warps=4, num_stages=2), triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=3), triton.Config({'BLOCK_N': 1024}, num_warps=8, num_stages=4),

- x_ptr, # *const T
- y_ptr, # *T B, # number of rows (int) D, # number of cols (int) stride_b, # row stride in elements (int) stride_d, # col stride in elements (int) BLOCK_N: tl.constexpr,

``` What changed and why it should be faster

triton.Config({'BLOCK_N': 2048}, num_warps=16, num_stages=4), # If D is very large, 4096 can help reduce loop iters; include it sparingly triton.Config({'BLOCK_N': 4096}, num_warps=16, num_stages=5),

- - Autotuning over BLOCK_N, num_warps, and num_stages lets Triton pick the best configuration for the given D on the current GPU.
- - Increased warps (up to 8) and pipelining (up to 4 stages) help hide memory latency and improve throughput on H100.
- - Slight arithmetic tweak (inv_mean = D / sum) avoids an earlier division.
- - Everything else remains fused and memory-efficient: two passes, no intermediate tensors, coalesced access. This should preserve correctness while extracting a bit more performance compared to the fixed-parameter kernel.

], key=['D'], # tune per column size

):

) @triton.jit def _l1_norm_rows_kernel(

# program id = row index row = tl.program_id(0) if row >= B: return

…

- Figure 11: The case study of DR. KERNEL-14B in LayerNorm operation from Kernelbench Level-1 subset. The speedup of the three turns are 1.04, 1.21, and 1.45, respectively. In this case, DR. KERNEL-14B effectively handles basic kernel writing and adapts to environmental feedback over multiple turns.

into Triton kernels, which account for 86.15% of the total CUDA runtime. However, we note that a convolution operation remains in the implementation. During training, we observed that the model attempted to implement this operation with Triton. Despite Triton’s potential for kernel optimization, convolution operations are difficult to implement effectively, as they are highly optimized by libraries such as cuDNN. Consequently, our small-sized models with limited data struggle to implement this operation as effectively as cuDNN, and instead, the model focuses on fusing other kernels. This behavior highlights one of the root causes of lazy optimization and emphasizes the importance of a well-established optimization objective in RL-based kernel generation, such as a bottleneck-aware profiling-based reward.

###### Dr.Kernel-8B-Better-Fusion.py

… class ModelNew(nn.Module): def __init__(self, …): … def forward(self, x):

# ConvTranspose3d using PyTorch/cuDNN x = self.conv_transpose(x) # x shape: [N, out_channels, D, H, W] … if use_triton:

# First pooling: kernel=2, stride=2, padding=0

D_out1 = (D - 2) // 2 + 1 H_out1 = (H - 2) // 2 + 1 W_out1 = (W - 2) // 2 + 1

x_pooled1 = torch.empty((N, C, D_out1, H_out1, W_out1), dtype=x.dtype, device=x.device)

grid1 = (N, C, D_out1 * H_out1 * W_out1) max_pool3d_k2s2_per_channel_kernel[grid1](

x, x_pooled1, N, C, D, H, W, D_out1, H_out1, W_out1,

x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4), x_pooled1.stride(0), x_pooled1.stride(1), x_pooled1.stride(2), x_pooled1.stride(3),

x_pooled1.stride(4), num_warps=4, num_stages=2 )

# Second pooling: kernel=3, stride=3, padding=0

D_out2 = (D_out1 - 3) // 3 + 1 H_out2 = (H_out1 - 3) // 3 + 1 W_out2 = (W_out1 - 3) // 3 + 1

x_pooled2 = torch.empty((N, C, D_out2, H_out2, W_out2), dtype=x.dtype, device=x.device) grid2 = (N, C, D_out2 * H_out2 * W_out2) max_pool3d_k3s3_per_channel_kernel[grid2](…) … grid_sum = (N * D_out2 * H_out2 * W_out2,) sum_channels_kernel[grid_sum]( x_pooled2, y, N, C, D_out2, H_out2, W_out2, x_pooled2.stride(0), x_pooled2.stride(1), x_pooled2.stride(2), x_pooled2.stride(3),

x_pooled2.stride(4), y.stride(0), y.stride(2), y.stride(3), y.stride(4), num_warps=4, num_stages=2 )

return y

- Figure 12: The case study of better fusion from DR. KERNEL-8B, which corresponds to the profiling feedback in Figure 10 (right).

#### F Prompt Template

- F.1 Prompt Template for Cold-Start Data Distillation We show the template for cold-start data distillaion in Figure 13.
- F.2 Prompt Template for SFT and RL We show the template for both SFT and RL in Figure 14.

###### 1st Turn Prompt Template for Training

###### Prompt Template After 1st Turn

You are looking at this PyTorch code and thinking it could be optimized with Triton.

Server feedback from the evaluation environment for your last implementation: {kernelgym_feedbacks}

Here's the PyTorch code: ```python {reference_code} ``` You need to create a Triton version with the entry point called `{entry_point}New`. Please firstly analyze this code and think hard how you can optimize it.

Based on the above server feedback, please improve the implementation:

- - If there are errors/crashes/illegal memory access: identify the root cause and fix it; prevent recurrence.
- - If there is no speedup or performance regresses: optimize the bottlenecks to achieve a clear speedup.
- - If there is already a speedup: further improve performance without degrading correctness.
- - Please output your thinking, plan, analysis, and the final code.

**Please output and show your thinking, plan, analysis etc., before your coding, which should be as more as possible.**

Figure 13: The prompt template for cold-start data distillation.

###### 1st Turn Prompt Template for Training

###### Example Kernel Code

###### Example Reference Code

import torch import torch.nn as nn import torch.nn.functional as F

[user] You write custom Triton kernels to replace the pytorch operators in the given architecture to get speedups.

import torch import torch.nn as nn import torch.nn.functional as F import triton import triton.language as tl

class Model(nn.Module): def __init__(self) -> None:

You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom Triton kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.

@triton.jit def add_kernel(

super().__init__() def forward(self, a, b):

- x_ptr, # Pointer to first input
- y_ptr, # Pointer to second input out_ptr, # Pointer to output n_elements, # Total number of elements in input/output BLOCK_SIZE: tl.constexpr,

return a + b def get_inputs(): # randomly generate input tensors based on the model architecture

):

# Each program handles a contiguous block of data of size BLOCK_SIZE block_start = tl.program_id(0) * BLOCK_SIZE # Create a range of offsets [0..BLOCK_SIZE-1] offsets = block_start + tl.arange(0, BLOCK_SIZE) # Mask to ensure we don't go out of bounds mask = offsets < n_elements # Load input values

- a = torch.randn(1, 128).cuda()
- b = torch.randn(1, 128).cuda() return [a, b]

Here's an example to show you the syntax of inline embedding

def get_init_inputs(): # randomly generate tensors required for initialization based on the model architecture return []

- x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
- y = tl.load(y_ptr + offsets, mask=mask, other=0.0) # Perform the elementwise addition out = x + y # Store the result tl.store(out_ptr + offsets, out, mask=mask)

custom Triton kernels in torch: The example given architecture is: ``` {example_ref} ``` The example new arch with custom Triton kernels looks like this: ``` {example_kernel} ```

###### Prompt Template for Training After 1st Turn

def triton_add(x: torch.Tensor, y: torch.Tensor): """ This function wraps the Triton kernel call. It:

- 1. Ensures the inputs are contiguous on GPU.
- 2. Calculates the grid (blocks) needed.
- 3. Launches the Triton kernel.

""" assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."

Now you have received the server feedback for your last implementation. Based on that and all your previous responses, improve the implementation.

- x = x.contiguous()
- y = y.contiguous()

# Prepare output tensor out = torch.empty_like(x)

Here is the server feedback. Please refer to this feedback to improve the implementation:

# Number of elements in the tensor n_elements = x.numel() BLOCK_SIZE = 128 # Tunable parameter for block size

You are given the following architecture: ``` {ref_code} ```

{KernelGYM Feedback} Return an improved Triton implementation named `ModelNew` as a single ```python``` block. Let's think step by step.

# Determine the number of blocks needed grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

# Launch the Triton kernel add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE) return out

Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Let's think step by step.

class ModelNew(nn.Module): def __init__(self) -> None:

super().__init__()

def forward(self, a, b): # Instead of "return a + b", call our Triton-based addition return triton_add(a, b)

Figure 14: The prompt template for both SFT and RL. The task instruction, example code and reference code are shown in the first-turn prompt. And for the later turns, prompt asks model to refine the kernel implementaion based on all hisotory information and KERNELGYM feedbacks.

