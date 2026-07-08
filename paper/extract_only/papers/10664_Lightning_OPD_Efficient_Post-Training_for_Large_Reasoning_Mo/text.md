# arXiv:2604.13010v2[cs.LG]8May2026

[Figure 1]

## Lightning OPD: Efficient Post-Training for Large Reasoning Models with Offline On-Policy Distillation

###### Yecheng Wu, Song Han, Han Cai

NVIDIA https://github.com/jet-ai-projects/Lightning-OPD

Abstract: On-policy distillation (OPD) is an effective post-training paradigm for large language models but requires a live teacher server throughout training, resulting in substantial infrastructure overhead. We investigate whether OPD can be performed offline by precomputing teacher log-probabilities once over SFT rollouts and reusing them during training. We find that naively doing so fails to reliably match standard OPD, and trace the root cause to a previously overlooked condition we term teacher consistency, requiring that the same teacher be used for both supervised fine-tuning and OPD. Violating this condition introduces a gradient bias that degrades performance for both offline and online OPD. Building on this insight, we propose Lightning OPD, an offline on-policy distillation framework that enforces teacher consistency and eliminates the need for a live teacher server entirely. We prove that, under teacher consistency, Lightning OPD shares the same optimum as standard OPD, with bounded gradient discrepancy and an implicit regularization effect that helps prevent policy drift. Experiments on math reasoning and code generation show that Lightning OPD achieves comparable performance to standard OPD while delivering 4.0× higher training efficiency. Starting from an SFT-initialized Qwen3-8B-Base model, Lightning OPD reaches 69.9% on AIME 2024 in just 30 GPU hours. Lightning OPD further scales to MoE architectures, training Qwen3-30B-A3B to 71.0% on AIME 2024 on a single 8×H100 node, substantially lowering the barrier for academic research on LLM post-training. Our code is released at https://github.com/jet-ai-projects/Lightning-OPD.

85

85

|SFT OPD Lightning OPD<br><br>56.7<br><br>65.4<br><br>68.1<br><br>31.5<br><br>39.3 40.3<br><br>72<br><br>20<br><br>3.6×|
|---|

|SFT OPD Lightning OPD<br><br>63.7<br><br>68.5 69.9<br><br>36.8<br><br>41.2<br><br>43.9<br><br>120<br><br>30<br><br>4.0×|
|---|

Accuracy(%)

Accuracy(%)

65

65

45

45

25

25

AIME24 LiveCodebench v6 GPU Hours

AIME24 LiveCodebench v6 . GPU Hours

Student Model: Qwen3-4B-Base

Student Model: Qwen3-8B-Base

Lightning OPD

OPD

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

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Rollout

Logprobs

Serve Teacher Train Student

Offline Dataset

Train Student

4.0× Higher Training Efficiency

Figure 1 | (Top) Performance (Pass@1, %) and training cost of Lightning OPD compared to standard OPD and the SFT baseline on Qwen3-4B-Base and Qwen3-8B-Base models. Lightning OPD achieves comparable performance to standard OPD on both math and coding benchmarks across both scales, while eliminating the need for a live teacher server during training. At the 8B scale, Lightning OPD achieves a state-of-the-art 69.9% on AIME 2024 in just 30 GPU hours, delivering 4.0× higher training efficiency than standard OPD. (Bottom) Intuitive comparison of GPU resource allocation. Standard OPD requires co-hosting the student and teacher, fragmenting GPU resources. Lightning OPD collects rollouts and teacher log-probabilities offline, dedicating all GPUs to student training.

Corresponding author(s): Han Cai (hcai@nvidia.com). © 2026 NVIDIA. All rights reserved.

### 1. Introduction

Large language models (LLMs) have achieved remarkable progress across tasks such as mathematical reasoning, code generation, and multi-step agent planning [1, 2, 3, 4, 5]. This success is underpinned by carefully designed post-training pipelines [6], which typically consist of supervised fine-tuning (SFT) on high-quality data [7], followed by a reinforcement learning (RL) stage to elicit stronger reasoning capabilities. On-Policy Distillation (OPD) [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] has emerged as a particularly effective alternative to the RL stage. It trains a student model to match a stronger teacher’s token-level distribution using dense per-token advantage signals. Compared to Reinforcement Learning from Verifiable Rewards (RLVR) [19, 20, 1, 21], OPD provides richer supervision, offers greater training stability, and incurs significantly lower training costs, while achieving competitive or superior performance across a wide range of tasks [9, 22, 10, 14, 23, 5].

However, standard OPD requires the teacher to score every student rollout during training, which introduces a persistent infrastructure bottleneck. A dedicated multi-GPU teacher server must run in parallel with the training job, leading to substantial compute overhead and making large-scale experiments costly and difficult to reproduce, particularly for academic researchers without access to extensive serving infrastructure.

A natural question is whether the benefits of on-policy supervision can be preserved while eliminating the need for a live teacher server. On-policy training is defined by the student’s current rollout distribution, which evolves at every gradient step, making the teacher appear indispensable. However, recent empirical studies suggest that RL-trained models remain surprisingly close to their SFT initialization: reasoning trajectories in RL models are largely a reweighted subset of those present in the SFT model [24], and on-policy updates are inherently biased toward solutions that minimize KL divergence from the reference policy [25]. We observe a similar phenomenon in OPD training, where the student’s distribution exhibits only modest drift from the SFT reference throughout the OPD stage. This observation suggests a practical offline alternative [26]: precompute the teacher’s log-probabilities once over SFT rollouts prior to training and reuse these values throughout the OPD process, thereby eliminating the need for a live teacher server.

In practice, however, naively applying this offline precomputation fails to reliably match the performance of standard OPD. Investigating the root cause, we find that the issue does not primarily stem from the offline approximation itself, but rather from a more fundamental condition that has been overlooked in prior OPD work, which we term teacher consistency. Unlike RLVR, where model behavior is shaped solely by a reward signal, OPD involves two distinct teachers: one used during the SFT stage to generate training trajectories, and another used during the OPD stage to provide the reference distribution. Teacher consistency requires these two teachers to be the same model. In practice, existing pipelines often violate this condition by following conventions inherited from RLVR, where SFT datasets are curated using whichever teacher produces the highest-quality demonstrations, without regard to the teacher used during OPD. For example, Thinking Machines Lab [9] trains a Qwen3-8B-Base model on OpenThoughts-3 [7], whose trajectories are generated by QwQ-32B, while using Qwen3-32B as the OPD teacher, resulting in a mismatch that our analysis predicts to be detrimental. We show that such teacher inconsistency introduces a gradient bias that degrades performance for both offline and online OPD, with a more pronounced effect on the offline variant. These findings establish teacher consistency as an important design principle for any OPD pipeline, rather than specific to the offline setting.

With teacher consistency established as the key condition, we propose Lightning OPD (Lightning On-Policy Distillation), an offline distillation framework that naturally arises from enforcing this principle. In the SFT stage, the base model is fine-tuned on trajectories generated by a chosen teacher 𝜋𝑇 to obtain the reference policy 𝜋ref. In the OPD stage, rollouts are sampled from 𝜋ref, and the same teacher’s log-probabilities are precomputed once over these fixed responses, eliminating the need for a live teacher server during training. We provide a rigorous theoretical analysis showing that, under teacher consistency, Lightning OPD provably shares the same optimum as standard OPD. Moreover, the gradient discrepancy between the two remains bounded throughout training, and the offline objective introduces an implicit regularization effect that naturally prevents policy drift without requiring any explicit penalty.

We evaluate Lightning OPD on math and code reasoning tasks across diverse student–teacher model pairs, including Qwen3-4B with Qwen3-8B and Qwen3-8B with Qwen3-32B as the teacher [22]. Lightning OPD achieves comparable performance to standard OPD across all benchmarks while bringing significant training speedup by removing the need for parallel teacher serving infrastructure. Concretely, starting from an

SFT-initialized Qwen3-8B-Base model, Lightning OPD reaches 69.9% on AIME 2024 in just 30 GPU hours, achieving 4.0× higher training efficiency than standard OPD. We further demonstrate that Lightning OPD scales to Mixture-of-Experts (MoE) architectures, where standard OPD faces significant challenge due to the memory overhead of co-hosting both student and teacher models. On a single 8×H100 node, Lightning OPD trains a Qwen3-30B-A3B model to 71.0% on AIME 2024 and 60.8% on LiveCodeBench v5, significantly lowering the barrier for academic research on LLM post-training.

Our main contributions are summarized as follows:

- • We identify teacher consistency as an important design principle for effective OPD, requiring that SFT teacher and OPD teacher be the same model. We show that violating this condition introduces a gradient bias that degrades performance for both online and offline OPD.
- • We propose Lightning OPD, an offline on-policy distillation framework that enforces teacher consistency and eliminates the need for a live teacher server by precomputing teacher log-probabilities once over SFT rollouts. We prove that, under teacher consistency, Lightning OPD shares the same optimum as standard OPD, with bounded gradient discrepancy and an implicit regularization effect that naturally prevents policy drift.
- • We validate Lightning OPD on math and code reasoning tasks across diverse student–teacher pairs, from dense models (4B, 8B) to Mixture-of-Experts (30B-A3B). Starting from an SFT-initialized Qwen38B-Base model, Lightning OPD achieves 69.9% on AIME 2024 in just 30 GPU hours, delivering 4.0× higher training efficiency than standard OPD. On a 30B MoE model, Lightning OPD reaches 71.0% on AIME 2024 on a single 8×H100 node, significantly lowering the barrier for academic research on LLM post-training.

### 2. Related Work

LLM Post-Training. Post-training is now central to capable LLMs, typically combining supervised fine-tuning (SFT) on high-quality demonstrations [6, 7, 27] with reinforcement learning (RL) for further capability elicitation. Existing RL methods span outcome-based optimization against sparse verifiable rewards [28, 29, 30, 31, 32, 33, 20, 21] and process-based optimization with dense intermediate supervision [34, 35, 36, 37, 19], together driving recent reasoning advances [1, 38, 39, 2, 3, 4, 5, 40, 22, 41]. Recent work increasingly studies the co-design of SFT and RL [42, 43, 44, 45, 46, 47]. From the OPD perspective, our contribution is to show that SFT and OPD must share the same teacher, making teacher consistency a principled co-design constraint.

On-Policy Distillation. Knowledge distillation transfers capabilities from a large teacher to a smaller student by matching output distributions [48]. Standard offline KD uses fixed teacher-generated data [49, 50], whereas on-policy distillation (OPD) aligns the student to the teacher on student-generated rollouts [8, 9], often yielding stronger post-training gains [22, 10]. Recent OPD variants study reward extrapolation [10], self-distillation [11, 13, 12], entropy-aware divergence [18], reinforcement-aware distillation [51], failure modes and recipes [17, 16], controllable reasoning [52], black-box or privileged settings [53, 54], and broader posttraining validation [14]. Our Lightning OPD removes the live teacher server while preserving dense per-token supervision, and highlights a design constraint largely ignored in prior OPD practice that the SFT-stage and OPD-stage teachers must be the same model.

Off-Policy Reinforcement Learning. Off-policy RL decouples data collection from optimization [55, 56]. This motivates offline RL algorithms [57, 58, 59], their use in LLM alignment [60, 61, 62], and recent off-policy reasoning methods [19, 20, 63, 64]. Lightning OPD is similar only in optimizing over a fixed dataset but receives dense token-level teacher supervision rather than sparse rewards, so its central challenge is teacher consistency rather than OOD value estimation. We provide a more detailed comparison in Appendix D.

### 3. Methodology

##### 3.1. Preliminaries

Let 𝜋𝑇 denote a fixed teacher model and 𝜋𝜃 a trainable student. Given a prompt 𝑞, a response 𝑥 = (𝑎1,...,𝑎𝑇) is generated autoregressively as

∏︁𝑇

𝜋𝜃(𝑥 | 𝑞) =

𝜋𝜃(𝑎𝑡 | 𝑠𝑡), (1)

𝑡=1

where 𝑠𝑡 = (𝑞,𝑎1,...,𝑎𝑡−1). Let 𝜋ref denote the SFT-initialized student. The per-token OPD advantage is 𝐴𝑡(𝜃) = log 𝜋𝑇(𝑎𝑡 | 𝑠𝑡) − log 𝜋𝜃(𝑎𝑡 | 𝑠𝑡), (2)

which is positive where the teacher is more confident than the student and negative otherwise. Standard OPD [8, 9, 22] optimizes

[︃ 𝑇

𝐴𝑡(𝜃)]︃, (3)

∑︁

𝐽on(𝜃) = E𝑞∼𝑝, 𝑥∼𝜋

𝜃

𝑡=1

while Lightning OPD fixes the rollout distribution to 𝜋ref and optimizes

[︃ 𝑇

𝐴𝑡(𝜃)]︃. (4)

∑︁

𝐽off(𝜃) = E𝑞∼𝑝, 𝑥∼𝜋

ref

𝑡=1

Both objectives share the same advantage function and differ only in the response distribution. Following standard OPD practice [8, 9, 22], the advantage 𝐴𝑡(𝜃) is treated as a fixed scalar (stop-gradient) when computing parameter updates; throughout our analysis, ∇𝐽on and ∇𝐽off denote the resulting advantageweighted∑︀ policy gradients (formalized in Appendix A.1). Defining the per-trajectory gradient 𝑓(𝑥;𝜃) :=

[𝑤(𝑥;𝜃) · 𝑓(𝑥;𝜃)] where 𝑤(𝑥;𝜃) = 𝜋𝜃(𝑥)/𝜋ref(𝑥), and the offline gradient ∇𝐽off(𝜃) = E𝑥∼𝜋

𝑡 𝐴𝑡(𝜃) · ∇log 𝜋𝜃(𝑎𝑡 | 𝑠𝑡), an IS decomposition gives ∇𝐽on(𝜃) = E𝑥∼𝜋

ref

[𝑓(𝑥;𝜃)] is the special case 𝑤 ≡ 1.

ref

##### 3.2. Lightning On-Policy Distillation

Following the common practice of LLM post-training [6, 1], Lightning OPD consists of two stages. We describe each stage below and highlight where Lightning OPD departs from standard OPD.

- Stage 1: Supervised Fine-Tuning. Given a base model 𝜋base, a teacher 𝜋𝑇, and a prompt dataset 𝒬SFT, we collect a supervised dataset from teacher-generated trajectories:

𝒟SFT =

{︀(︀

𝑞𝑖, 𝑥𝑖

)︀

⃒ 𝑥𝑖 ∼ 𝜋𝑇(· | 𝑞𝑖), 𝑞𝑖 ∈ 𝒬SFT }︀

. (5)

The base model is then fine-tuned on 𝒟SFT via maximum likelihood estimation to obtain the reference policy 𝜋ref:

𝜋ref = arg max

𝜃

E(𝑞,𝑥)∼𝒟

SFT

[︃ 𝑇

∑︁

𝑡=1

log 𝜋𝜃(𝑎𝑡 | 𝑠𝑡)]︃. (6)

While standard OPD may use any high-quality SFT dataset regardless of its source, Lightning OPD requires that 𝒟SFT be generated by the same teacher 𝜋𝑇 used in the OPD stage. This teacher consistency condition is a prerequisite for the offline approximation to be sound, as we formally establish in Section 3.

- Stage 2: Offline On-Policy Distillation. Starting from a task-specific prompt dataset 𝒬OPD, the OPD stage proceeds in two phases. In the preprocessing phase, rollouts are sampled from 𝜋ref and the teacher is queried once to precompute and store per-token log-probabilities, forming the offline dataset:

𝒟OPD = {︁(︀

⃒ 𝑥𝑗 ∼ 𝜋ref(· | 𝑞𝑗), 𝑞𝑗 ∈ 𝒬OPD }︁. (7)

)︀

𝑞𝑗, 𝑥𝑗, {log 𝜋𝑇(𝑎𝑗𝑡 | 𝑠𝑗𝑡)}𝑇𝑡=1𝑗

In contrast, standard OPD requires a live teacher server throughout training, querying it at every step for log-probabilities on freshly sampled rollouts from the current policy 𝜋𝜃. In the training phase, the student

Stage 1: SFT Stage

Teacher Model πT Prompt Dataset

SFT Dataset

DSFT = Base Model πbase

| | |
|---|---|
| | |

#### {(qi, xi)|xi ∼ πT( ⋅ |qi)}

SFT Model πref

QSFT

Stage 2: Oﬄine OPD Stage Training Phase

Preprocessing Phase

OPD dataset with precomputed teacher logprob

Current Student Model πθ

SFT Model πref

#### = {qj, xj,{logπT(atj,stj)}Tj

t=1}

| | |
|---|---|
| | |

DOPD

Update Policy πθ

Prompt Dataset

QOPD

Reverse-KL Advantage Computation

At(θ) = logπT − logπθ

Teacher Model πT

Figure 2 | Overview of Lightning OPD. In the SFT stage, the teacher 𝜋𝑇 generates trajectories on 𝒬SFT then the base model 𝜋base is fine-tuned on these trajectories to obtain the SFT model 𝜋ref. OPD stage proceeds in two phases. In the preprocessing phase, rollouts are sampled from 𝜋ref on 𝒬OPD and the teacher is queried once to compute and store per-token log-probabilities log 𝜋𝑇(𝑎𝑡 | 𝑠𝑡), forming the offline dataset 𝒟OPD. In the training phase, the student 𝜋𝜃 is initialized from 𝜋ref and trained on 𝒟OPD. At each step, the per-token advantage 𝐴𝑡 = log 𝜋𝑇(𝑎𝑡 | 𝑠𝑡) − log 𝜋𝜃(𝑎𝑡 | 𝑠𝑡) is computed by reading log 𝜋𝑇 from 𝒟OPD while computing log 𝜋𝜃 online. During this process, we do not need a live teacher model. In contrast, standard OPD requires a live teacher server throughout training to compute advantages at every gradient step. By eliminating this requirement, Lightning OPD reduces total training cost of OPD by 4.0× at the 8B scale (from 120 to 30 GPU hours).

is initialized from 𝜋ref and trained on 𝒟OPD with no teacher server required. At each step, the advantage is computed as 𝐴𝑡(𝜃) = log 𝜋𝑇(𝑎𝑡 | 𝑠𝑡) − log 𝜋𝜃(𝑎𝑡 | 𝑠𝑡), where the teacher term is read directly from 𝒟OPD and the student term is computed online. This eliminates the primary infrastructure bottleneck of standard OPD and makes high-quality on-policy distillation accessible with minimal compute overhead. The full procedure is given in Algorithm 1.

##### 3.3. Theoretical Analysis

All proofs are deferred to Appendix A. The analysis rests on three standard assumptions, with a fourth for teacher-mismatch analysis.

- Assumption 3.1 (Bounded Absolute Advantage). There exists 𝜎𝐴 < ∞ such that for all 𝜃 ∈ Θ:

E𝑥∼𝜋

ref[︁(︀∑︀𝑇

𝑡=1 |𝐴𝑡(𝜃)|)︀2]︁ ≤ 𝜎𝐴2 , where 𝐴𝑡(𝜃) = log 𝜋𝑇(𝑎𝑡 | 𝑠𝑡) − log 𝜋𝜃(𝑎𝑡 | 𝑠𝑡).

- Assumption 3.2 (Support Coverage). For all 𝜃 encountered during optimization and all prompts 𝑞: supp(𝜋𝜃(· | 𝑞)) ⊆ supp(𝜋ref(· | 𝑞)).
- Assumption 3.3 (Bounded Score Function). There exists 𝐺 < ∞ such that ‖∇log 𝜋𝜃(𝑎𝑡 | 𝑠𝑡)‖2 ≤ 𝐺 for all 𝜃 ∈ Θ and all 𝑡.
- Assumption 3.4 (Bounded Teacher Mismatch). Let Δ𝑡 = log 𝜋𝑇SFT(𝑎𝑡 | 𝑠𝑡) − log 𝜋𝑇OPD(𝑎𝑡 | 𝑠𝑡). There exists

ref[︁(︀∑︀𝑇

𝑡=1 |Δ𝑡|)︀2]︁ ≤ 𝜎Δ2 . When teacher consistency holds (𝜋𝑇SFT = 𝜋𝑇OPD), Δ𝑡 = 0 everywhere and 𝜎Δ = 0.

𝜎Δ < ∞ such that E𝑥∼𝜋

Assumptions 3.1–3.3 are standard: 3.1 is an 𝐿2 condition on absolute advantages, automatically satisfied under advantage clipping (|𝐴𝑡| ≤ 𝜏 gives 𝜎𝐴 ≤ 𝑇𝜏); 3.2 holds naturally when 𝜋𝜃 is initialized from 𝜋ref; and

- 3.3 is standard in policy gradient analyses. Assumption 3.4 quantifies the teacher mismatch and is only needed for the teacher consistency results (Theorems 3.8–3.9).

- Theorem 3.5 (Gradient Discrepancy Bound). Under Assumptions 3.1–3.3, ‖∇𝐽on(𝜃) − ∇𝐽off(𝜃)‖2 ≤ 𝐺 · 𝜎𝐴 · √︀𝜒2(𝜋𝜃 ‖𝜋ref), (8)

Algorithm 1 Lightning On-Policy Distillation (Lightning OPD) Require: Base model 𝜋base, teacher 𝜋𝑇, prompt datasets 𝒬SFT, 𝒬OPD, learning rate 𝜂

- 1: // Stage 1: SFT
- 2: Collect 𝒟SFT = {(𝑞,𝑥) | 𝑞 ∈ 𝒬SFT, 𝑥 ∼ 𝜋𝑇(· | 𝑞)}
- 3: Fine-tune 𝜋base on 𝒟SFT to obtain 𝜋ref
- 4: // Stage 2, Phase 1: Preprocessing
- 5: for each prompt 𝑞𝑗 ∈ 𝒬OPD do
- 6: Sample 𝑥𝑗 ∼ 𝜋ref(· | 𝑞𝑗); store log 𝜋𝑇(𝑎𝑗𝑡 | 𝑠𝑗𝑡) for all 𝑡
- 7: end for
- 8: Form 𝒟OPD = {(𝑞𝑗,𝑥𝑗,{log 𝜋𝑇(𝑎𝑗𝑡 | 𝑠𝑗𝑡)}𝑡𝑇=1𝑗 )}
- 9: // Stage 2, Phase 2: Training
- 10: Initialize 𝜋𝜃 ← 𝜋ref
- 11: for each training step do
- 12: Sample mini-batch from 𝒟OPD
- 13: Compute 𝐴𝑡(𝜃) = log 𝜋𝑇(𝑎𝑡 | 𝑠𝑡) − log 𝜋𝜃(𝑎𝑡 | 𝑠𝑡), clip to [−𝜏,𝜏]
- 14: Update 𝜃 ← 𝜃 + 𝜂 · ∇𝐽off(𝜃)
- 15: end for

where 𝜒2(𝜋𝜃‖𝜋ref) = E𝑥∼𝜋

[𝑤(𝑥;𝜃)2] − 1.

ref

At initialization 𝜋𝜃 = 𝜋ref, 𝜒2 = 0 and the two gradients coincide exactly; the bound grows with drift but remains small under standard KL regularization. When the teacher is representable, the two methods share a common fixed point:

- Theorem 3.6 (Shared Fixed Point). The online objective satisfies 𝐽on(𝜃) = −KL(𝜋𝜃 ‖𝜋𝑇) ≤ 0, with global maximum at 𝜃* ∈ arg min𝜃∈Θ KL(𝜋𝜃 ‖𝜋𝑇). When 𝜋𝑇 ∈ ΠΘ, 𝐴𝑡(𝜃*) = 0 almost surely, and 𝜃* is a shared zero of both the online and offline OPD updates.

When 𝜋𝑇 ∈/ ΠΘ, Theorem 3.5 ensures the offline update stays close to the online one as long as drift remains small, suggesting comparable capacity-limited behavior in practice. Moreover, the two updates are related by a covariance correction:

- Theorem 3.7 (Gradient Decomposition). The offline gradient decomposes as ∇𝐽off(𝜃) = ∇𝐽on(𝜃) −

Cov𝜋

ref

[𝑤(𝑥;𝜃), 𝑓(𝑥;𝜃)], where 𝑓(𝑥;𝜃) =

∑︀

𝑡 𝐴𝑡(𝜃) · ∇log 𝜋𝜃(𝑎𝑡 | 𝑠𝑡).

The covariance term vanishes at initialization (𝑤 ≡ 1) and grows with drift, empirically acting as a trust-region effect that stabilizes training without an explicit KL penalty (Figure 3a). The three results above assume the SFT and OPD stages use the same teacher. We formalize this as teacher consistency and show that violating it introduces a bias in both paradigms:

- Theorem 3.8 (Teacher Consistency and the Offline-Online Gap). Let 𝜋𝑇SFT and 𝜋𝑇OPD denote the teachers used in the SFT and OPD stages. Under Assumptions 3.1–3.4,

‖∇𝐽on(𝜃) − ∇𝐽off(𝜃)‖2 ≤ 𝐺 ·(𝜎𝐴 + 𝜎Δ) · √︀𝜒2(𝜋𝜃 ‖𝜋ref). (9)

Additionally, when 𝜎Δ > 0, the mismatched offline gradient can carry a persistent bias bounded by ‖∇𝐽off(𝜃)− ∇𝐽off𝛿=0(𝜃)‖2 ≤ 𝐺𝜎Δ relative to the consistent offline gradient, independent of 𝜒2.

- Theorem 3.9 (Teacher Consistency and Standard OPD). Let ∇𝐽on𝛿=0(𝜃) denote the standard OPD gradient under a consistent teacher (𝜋𝑇SFT = 𝜋𝑇OPD). At initialization 𝜃 = 𝜃ref,

‖∇𝐽on(𝜃ref) − ∇𝐽on𝛿=0(𝜃ref)‖2 ≤ 𝐺 · 𝜎Δ. (10)

Remark 3.10. Theorems 3.8–3.9 show that 𝐺𝜎Δ biases the gradient of both paradigms. Teacher consistency is therefore an important design requirement: when 𝜎Δ = 0, Theorems 3.5–3.6 apply in full and Lightning OPD matches standard OPD.

### 4. Experiments

##### 4.1. Experimental Setup

Models. We train two student models under the Lightning OPD pipeline, covering different model scales from the Qwen3 model family [22]. The first uses Qwen3-4B-Base as the student with Qwen3-8B as the teacher. The second uses Qwen3-8B-Base as the student with Qwen3-32B as the teacher. Both pipelines follow the two-stage procedure described in Section 3.2. The base model is first fine-tuned on teacher-generated trajectories to obtain 𝜋ref, which is then used to sample rollouts and precompute teacher log-probabilities for the OPD stage.

Training Data. The SFT stage uses prompts from OpenThoughts-3 [7], with responses generated by the respective teacher model. For the OPD stage, we train on two domains. Mathematical reasoning uses DAPO-Math-17k [21], which provides 17K competition-level math problems spanning a wide range of difficulty. Code generation uses a sampled 30K subset of EpiCoder-func-380k [65], which provides diverse function-level code synthesis problems. For each prompt, we sample a single response from 𝜋ref and precompute the corresponding teacher log-probabilities once prior to training, with no teacher server required during the OPD stage.

Benchmarks. For math reasoning, we evaluate on AIME 2024 [66], AIME 2025 [67], and HMMT 2025 [68]. For code reasoning, we evaluate on LiveCodeBench v5 and v6 [69]. In all evaluations, we set the temperature to 0.6, top-𝑝 to 0.95, and the maximum generation length to 32,768 for math benchmarks and 40,960 for coding benchmarks. We sample 32 solutions per problem for math benchmarks and 4 solutions per problem for code benchmarks, reporting the average pass@1.

Training Settings. The SFT stage is implemented using LlamaFactory [70] and the OPD stage is implemented using slime [71]. The OPD stage is trained for 150 steps, which we find sufficient for convergence as shown in Figure 3b. Standard OPD and Lightning OPD share identical training settings in the OPD stage, differing only in the source of rollouts that standard OPD samples rollouts online from the current student while Lightning OPD reuses rollouts precomputed from 𝜋ref before training begins. Full hyperparameter details are provided in Appendix B.

##### 4.2. Main Results

- Table 1 presents evaluation results across five benchmarks for both 4B and 8B model scales. The central finding is that Lightning OPD, despite eliminating the live teacher server entirely during training, achieves performance on par with standard OPD across all settings, and in several cases marginally exceeds it. This validates our theoretical analysis that under teacher consistency, the offline approximation preserves the same performance optimum as standard OPD. The gains from the OPD stage over the SFT baseline are substantial and consistent across both math and code benchmarks, confirming that on-policy distillation provides strong and transferable post-training improvements. At the 4B scale, compared to ExOPD [10], a recent OPD baseline, Lightning OPD achieves substantially better results, reaching 68.1% compared to 61.0% on AIME 2024, and the gap widens on code generation where Lightning OPD reaches 40.3% on LCB v6 against ExOPD’s 29.0%. At the 8B scale, Lightning OPD achieves 69.9% on AIME 2024 and 49.5% on LiveCodeBench v5. Together, these results demonstrate the effectiveness of Lightning OPD as a general and efficient post-training framework across model scales and task domains.

- 4.3. Training Cost

- Table 2 compares the training cost of standard OPD and Lightning OPD. Lightning OPD achieves a 3.6× speedup at the 4B scale, reducing total GPU hours from 72 to just 20, and a 4.0× speedup at the 8B scale, bringing the full pipeline down from 120 to just 30 GPU hours. We also provide a per-phase breakdown of the Lightning OPD pipeline. The actual OPD training stage consumes only a small fraction of this budget, with the remaining cost split between rollout collection and teacher logprob precomputation, both of which are

- Table 1 | Pass@1 on math and code reasoning benchmarks. Lightning OPD achieves comparable performance to standard OPD across all benchmarks at both model scales, while requiring no live teacher server during training. At the 4B scale, Lightning OPD substantially outperforms ExOPD [10], achieving 68.1% compared to 61.0% on AIME 2024 and 40.3% compared to 29.0% on LCB v6. At the 8B scale, Lightning OPD reaches 69.9% on AIME 2024 and 49.5% on LCB v5. Bold indicates the best result within each model scale.

Method

Math Reasoning Code Generation

AIME 2024 AIME 2025 HMMT 2025 Avg. LCB v5 LCB v6 Avg. Student: Qwen3-4B-Base Teacher: Qwen3-8B

SFT 56.7 52.1 34.0 47.6 33.8 31.5 32.6 ExOPD [10] 61.0 56.0 34.4 50.5 – 29.0 – OPD 65.4 57.9 39.9 54.4 44.2 39.3 41.8

- Lightning OPD 68.1 58.4 39.8 55.4 42.8 40.3 41.5 Student: Qwen3-8B-Base Teacher: Qwen3-32B

SFT 63.7 51.7 36.9 50.8 44.7 36.8 40.8 OPD 68.5 59.0 39.4 55.6 47.3 41.2 44.2

- Lightning OPD 69.9 59.2 41.9 57.0 49.5 43.9 46.7

- Table 2 | Training cost (GPU hours) of standard OPD vs. Lightning OPD. Lightning OPD achieves a 3.6× speedup at 4B and a 4.0× speedup at 8B, requiring only 20 and 30 GPU hours to train a reasoning model respectively. The lower panel breaks down Lightning OPD costs by stage, showing that the actual OPD training consumes only a moderate fraction of the total budget, highlighting how the minimal infrastructure requirement of Lightning OPD makes the training highly efficient.

Method Qwen3-4B-Base Qwen3-8B-Base

OPD 72 120 Lightning OPD 20 30

Speedup 3.6× 4.0× Lightning OPD breakdown

⌞ Rollout collection 10 10 ⌞ Teacher logprob precompute 2 4 ⌞ OPD training 8 16

one-time offline operations that require no specialized infrastructure. This stands in sharp contrast to standard OPD, which demands a dedicated multi-GPU teacher server running continuously throughout training. The minimal infrastructure requirement of Lightning OPD makes high-quality on-policy distillation accessible to practitioners without large-scale training and serving systems.

##### 4.4. Scaling to Mixture-of-Experts

We further apply Lightning OPD to Qwen3-30B-A3B-Base, a 30B-parameter MoE model with 3B active parameters, using Qwen3-30B-A3B-Thinking-2507 as the teacher. We follow the same two-stage pipeline and training settings as the 8B experiments in Section 4.1. Standard OPD is infeasible at this scale on a single 8×H100 node, as co-hosting both a 30B student and a 30B teacher for training and scoring exceeds available GPU memory. Lightning OPD eliminates this bottleneck by precomputing teacher log-probabilities offline, allowing all GPUs to be dedicated to student training. As shown in Table 3, Lightning OPD reaches 71.0% on AIME 2024 and 60.8% on LiveCodeBench v5, achieving state-of-the-art performance among open MoE models at such model scale.

- Table 3 | Lightning OPD applied to a Mixture-of-Experts architecture (Qwen3-30B-A3B). By precomputing teacher log-probabilities offline, Lightning OPD enables on-policy distillation of a 30B MoE model on a single 8×H100 node, achieving state-of-the-art performance on math and code reasoning tasks. In contrast, OPD runs out of memory as it requires co-hosting both a 30B teacher and a 30B student. Bold indicates improvement over the SFT baseline.

Method

Math Reasoning Code Generation

AIME 2024 AIME 2025 HMMT 2025 Avg. LCB v5 LCB v6 Avg. Student: Qwen3-30B-A3B-Base Teacher: Qwen3-30B-A3B-Thinking-2507

SFT 66.8 63.2 44.6 58.2 39.4 33.0 36.2 OPD (OOM) ✗ ✗ ✗ ✗ ✗ ✗ ✗ Lightning OPD 71.0 66.3 48.3 61.9 60.8 54.4 57.6

- Table 4 | Teacher consistency ablation (AIME 2024 pass@1, %). Rows indicate the SFT-stage teacher and columns indicate the OPD-stage teacher. The consistent setting (diagonal, bold) always achieves the best performance for both standard OPD and Lightning OPD. Teacher inconsistency degrades Lightning OPD more severely than standard OPD (up to 7 points at 8B), because the fixed rollout distribution compounds the gradient bias when the SFT and OPD teachers differ.

Standard OPD OPD Teacher OPD Teacher SFT Teacher Qwen3-8B QwQ-32B SFT Teacher Qwen3-32B QwQ-32B Student: Qwen3-4B-Base Student: Qwen3-8B-Base

Qwen3-8B 65.4 62.4 Qwen3-32B 68.5 64.8 QwQ-32B 61.2 62.8 QwQ-32B 65.0 66.5

Lightning OPD OPD Teacher OPD Teacher SFT Teacher Qwen3-8B QwQ-32B SFT Teacher Qwen3-32B QwQ-32B Student: Qwen3-4B-Base Student: Qwen3-8B-Base

Qwen3-8B 68.1 62.5 Qwen3-32B 69.9 63.1 QwQ-32B 59.3 63.1 QwQ-32B 62.1 68.7

4.5. Ablation Study

Teacher consistency. To empirically verify teacher consistency, we cross SFT-stage and OPD-stage teacher choices by introducing QwQ-32B [72] alongside Qwen3-32B at the 8B scale and Qwen3-8B at the 4B scale, yielding a full grid of teacher combinations for both standard OPD and Lightning OPD in Tables 4. The results uniformly confirm our theoretical prediction that the teacher-consistent setting on the diagonal always achieves the best performance, while mismatched teachers consistently degrade accuracy. Notably, teacher inconsistency imposes a larger penalty on Lightning OPD than on standard OPD. At the 8B scale, mismatching from Qwen3-32B SFT to QwQ-32B OPD drops Lightning OPD by 6.8 points but standard OPD by only 3.7 points. This asymmetry arises because a mismatched SFT teacher corrupts 𝜋ref in two roles simultaneously for Lightning OPD, both as the reference distribution and as the source of fixed rollouts, whereas standard OPD refreshes rollouts from the current student at every step and can partially recover. Teacher consistency is consequently a more critical design requirement for Lightning OPD than for standard OPD.

- 5. Conclusion

We presented Lightning OPD, an offline on-policy distillation framework for efficient LLM post-training. By precomputing teacher log-probabilities once over SFT rollouts, Lightning OPD reduces the infrastructure of OPD to a single standard training job while preserving the dense per-token supervision that makes OPD effective. We further identified teacher consistency as a necessary condition for OPD to work well in general, proving that mismatched SFT and OPD teachers introduce an irreducible gradient bias that causes both

paradigms to converge to a suboptimal fixed point, and that under consistency Lightning OPD provably shares the same optimum as standard OPD. Empirically, Lightning OPD trained an 8B reasoning model to 69.9% on AIME 2024 in just 30 GPU hours, achieving 4.0× speedup over standard OPD while matching its performance across all benchmarks. We hope these results encourage broader adoption of on-policy distillation in settings where a persistent teacher server is impractical, and that the teacher consistency principle provides a useful design guideline for future OPD research. More generally, our results suggest that much of the practical benefit of on-policy distillation can be retained without the conventional deployment burden of maintaining a live teacher throughout training.

### References

- [1] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [2] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.
- [3] Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, SH Cai, Yuan Cao, Y Charles, HS Che, Cheng Chen, Guanduo Chen, et al. Kimi k2. 5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.
- [4] NVIDIA. Nvidia nemotron 3: Efficient and open intelligence. arXiv preprint arXiv:2512.20856, 2025.
- [5] Bangjun Xiao, Bingquan Xia, Bo Yang, Bofei Gao, Bowen Shen, Chen Zhang, Chenhong He, Chiheng Lou, Fuli Luo, Gang Wang, et al. Mimo-v2-flash technical report. arXiv preprint arXiv:2601.02780, 2026.
- [6] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, 2022.
- [7] Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina, Jean Mercat, Trung Vu, Zayne Sprague, et al. Openthoughts: Data recipes for reasoning models. arXiv preprint arXiv:2506.04178, 2025.
- [8] Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In The Twelfth International Conference on Learning Representations, 2024.
- [9] Kevin Lu and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism, 2025. https://thinkingmachines.ai/blog/on-policy-distillation.
- [10] Wenkai Yang, Weijie Liu, Ruobing Xie, Kai Yang, Saiyong Yang, and Yankai Lin. Learning beyond teacher: Generalized on-policy distillation with reward extrapolation. arXiv preprint arXiv:2602.12125, 2026.
- [11] Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. arXiv preprint arXiv:2601.18734, 2026.
- [12] Idan Shenfeld, Mehul Damani, Jonas Hübotter, and Pulkit Agrawal. Self-distillation enables continual learning. arXiv preprint arXiv:2601.19897, 2026.
- [13] Jonas Hübotter, Frederike Lübeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, et al. Reinforcement learning via self-distillation. arXiv preprint arXiv:2601.20802, 2026.
- [14] Zhuolin Yang, Zihan Liu, Yang Chen, Wenliang Dai, Boxin Wang, Sheng-Chieh Lin, Chankyu Lee, Yangyi Chen, Dongfu Jiang, Jiafan He, Renjie Pi, Grace Lam, Nayeon Lee, Alexander Bukharin, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Nemotron-cascade 2: Post-training llms with cascade rl and multi-domain on-policy distillation. 2026.
- [15] Mingyang Song and Mao Zheng. A survey of on-policy distillation for large language models. arXiv preprint arXiv:2604.00626, 2026.
- [16] Siyuan Chen, Daya Guo, Yichun Tan, Xiaohan Liang, Hao Zhou, Bo Zheng, and Dejian Yang. Rethinking on-policy distillation of large language models: Phenomenology, mechanism, and recipe. arXiv preprint arXiv:2604.13016, 2026.

- [17] Yuqian Fu et al. Revisiting on-policy distillation: Empirical failure modes and simple fixes. arXiv preprint arXiv:2603.25562, 2026.
- [18] Yiren Wang et al. Entropy-aware on-policy distillation of language models. arXiv preprint arXiv:2603.07079, 2026.
- [19] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [20] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [21] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.
- [22] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [23] Aohan Zeng et al. GLM-5: From vibe coding to agentic engineering. arXiv preprint, 2026.
- [24] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025.
- [25] Idan Shenfeld, Jyothish Pari, and Pulkit Agrawal. Rl’s razor: Why online reinforcement learning forgets less. arXiv preprint arXiv:2509.04259, 2025.
- [26] Miao Rang, Zhenni Bi, Hang Zhou, Hanting Chen, An Xiao, Tianyu Guo, Kai Han, Xinghao Chen, and Yunhe Wang. Revealing the power of post-training for small language models via knowledge distillation. arXiv preprint arXiv:2509.26497, 2025.
- [27] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, volume 35, 2022.
- [28] Jian Hu. REINFORCE++: A simple and efficient approach for aligning large language models. arXiv preprint arXiv:2501.03262, 2025.
- [29] Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting REINFORCE-style optimization for learning from human feedback in LLMs. arXiv preprint arXiv:2402.14740, 2024.
- [30] Ziniu Li, Tian Xu, Yushun Zhang, Yang Yu, Ruoyu Sun, and Zhi-Quan Luo. ReMax: A simple, effective, and efficient reinforcement learning method for aligning large language models. arXiv preprint arXiv:2310.10505, 2023.
- [31] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding R1-Zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.
- [32] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.
- [33] MiniMax, Aonian Shan, Bangwei Gong, Bo Yang, et al. MiniMax-M1: Scaling test-time compute efficiently with lightning attention. arXiv preprint arXiv:2506.13585, 2025.
- [34] Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Yuchen Zhang, Jiacheng Chen, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, et al. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456, 2025.
- [35] Yufeng Yuan, Yu Yue, Ruofei Zhu, Tiantian Fan, and Lin Yan. What’s behind PPO’s collapse in long-CoT? value optimization holds the secret. arXiv preprint arXiv:2503.01491, 2025.
- [36] Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, Tiantian Fan, Zhengyin Du, Xiangpeng Wei, et al. VAPO: Efficient and reliable reinforcement learning for advanced reasoning tasks. arXiv preprint arXiv:2504.05118, 2025.

- [37] Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva Reddy, Aaron Courville, and Nicolas Le Roux. VinePPO: Refining credit assignment in RL training of LLMs. arXiv preprint arXiv:2410.01679, 2024.
- [38] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chengqi Zhao, Chenggang Deng, Chengpeng Zhang, et al. DeepSeek-V3 technical report. arXiv preprint arXiv:2412.19437, 2024.
- [39] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, et al. Kimi k1.5: Scaling reinforcement learning with LLMs. arXiv preprint arXiv:2501.12599, 2025.
- [40] Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model. arXiv preprint arXiv:2503.24290, 2025.
- [41] An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, et al. Qwen2.5-Math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122, 2024.
- [42] Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, and Yue Zhang. Learning to reason under off-policy guidance. arXiv preprint arXiv:2504.14945, 2025.
- [43] Lu Ma, Hao Liang, Meiyi Qiang, Lexiang Tang, Xiaochen Ma, Zhen Hao Wong, Junbo Niu, Chengyu Shen, Runming He, Yanhao Li, Bin Cui, and Wentao Zhang. Learning what reinforcement learning can’t: Interleaved online fine-tuning for hardest questions. arXiv preprint arXiv:2506.07527, 2026.
- [44] Wenhao Zhang, Yuexiang Xie, Yuchang Sun, Yanxi Chen, Guoyin Wang, Yaliang Li, Bolin Ding, and Jingren Zhou. On-policy RL meets off-policy experts: Harmonizing supervised fine-tuning and reinforcement learning via dynamic weighting. arXiv preprint arXiv:2508.11408, 2025.
- [45] Liang Chen, Xueting Han, Li Shen, Jing Bai, and Kam-Fai Wong. Beyond two-stage training: Cooperative SFT and RL for LLM reasoning. arXiv preprint arXiv:2509.06948, 2025.
- [46] Mingyang Liu, Gabriele Farina, and Asuman Ozdaglar. UFT: Unifying supervised and reinforcement fine-tuning. arXiv preprint arXiv:2505.16984, 2025.
- [47] Zeyu Huang, Tianhao Cheng, Zihan Qiu, Zili Wang, Yinghui Xu, Edoardo M. Ponti, and Ivan Titov. Blending supervised and reinforcement fine-tuning with prefix sampling. arXiv preprint arXiv:2507.01679, 2025.
- [48] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. NIPS Deep Learning Workshop, 2015.
- [49] Yoon Kim and Alexander M Rush. Sequence-level knowledge distillation. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, 2016.
- [50] Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. MiniLLM: Knowledge distillation of large language models. In The Twelfth International Conference on Learning Representations, 2024.
- [51] Zhihui Xu et al. Reinforcement-aware knowledge distillation for LLM reasoning. arXiv preprint arXiv:2602.22495, 2026.
- [52] Kun Liang, Clive Bai, Xin Xu, Chenming Tang, Sanwoo Lee, Weijie Liu, Saiyong Yang, and Yunfang Wu. Orbit: On-policy exploration-exploitation for controllable multi-budget reasoning. arXiv preprint arXiv:2601.08310, 2026.
- [53] Tianzhu Ye, Li Dong, Zewen Chi, Xun Wu, Shaohan Huang, and Furu Wei. Black-box on-policy distillation of large language models. arXiv preprint arXiv:2511.10643, 2025.
- [54] Emiliano Penaloza, Dheeraj Vattikonda, Nicolas Gontier, Alexandre Lacoste, Laurent Charlin, and Massimo Caccia. Privileged information distillation for language models. arXiv preprint arXiv:2602.04942, 2026.
- [55] Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643, 2020.
- [56] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3-4):229–256, 1992.
- [57] Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. Conservative Q-learning for offline reinforcement learning. In Advances in Neural Information Processing Systems, volume 33, 2020.

- [58] Ilya Kostrikov, Ashvin Nair, and Sergey Levine. Offline reinforcement learning with implicit q-learning. In International Conference on Learning Representations, 2022.
- [59] Scott Fujimoto, David Meger, and Doina Precup. Off-policy deep reinforcement learning without exploration. In International Conference on Machine Learning, 2019.
- [60] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, volume 36, 2023.
- [61] Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Keming Lu, Chuanqi Tan, Chang Zhou, and Jingren Zhou. Scaling relationship on learning mathematical reasoning with large language models. arXiv preprint arXiv:2308.01825, 2023.
- [62] Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, and Tong Zhang. RAFT: Reward ranked finetuning for generative foundation model alignment. Transactions on Machine Learning Research, 2023.
- [63] Daniel Ritter, Owen Oertell, Bradley Guo, Jonathan Chang, Kianté Brantley, and Wen Sun. LLMs can learn to reason via off-policy RL. arXiv preprint arXiv:2602.19362, 2026.
- [64] Yao Lu, Dengdong Fan, Jianzheng Nie, Fan Xu, Jie Chen, Bin Zhou, and Yonghong Tian. PCL-Reasoner-V1.5: Advancing math reasoning with offline reinforcement learning. arXiv preprint arXiv:2601.14716, 2026.
- [65] Yaoxiang Wang, Haoling Li, Xin Zhang, Jie Wu, Xiao Liu, Wenxiang Hu, Zhongxin Guo, Yangyu Huang, Ying Xin, Yujiu Yang, Jinsong Su, Qi Chen, and Scarlett Li. Encompassing diversity and complexity in code generation. arXiv preprint arXiv:2501.04694, 2025.
- [66] AI-MO. AIME 2024. https://huggingface.co/datasets/AI-MO/aimo-validation-aime, 2024.
- [67] OpenCompass. AIME 2025. https://github.com/open-compass/opencompass, 2025.
- [68] Mislav Balunović, Jasper Dekoninck, Ivo Petrov, Nikola Jovanović, and Martin Vechev. MathArena: Evaluating LLMs on uncontaminated math competitions. arXiv preprint arXiv:2505.23281, 2025.
- [69] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. LiveCodeBench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024.
- [70] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. LlamaFactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (System Demonstrations), 2024.
- [71] THUDM. slime: An SGLang-native post-training framework for RL scaling. https://github.com/THUDM/slime, 2025.
- [72] Qwen Team. Qwq-32b: Embracing the power of reinforcement learning, March 2025.
- [73] Yang Chen, Zhuolin Yang, Zihan Liu, Wenliang Dai, Boxin Wang, Sheng-Chieh Lin, Chankyu Lee, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Nemotron-cascade: Scaling cascaded reinforcement learning for general-purpose reasoning models. arXiv preprint arXiv:2512.13607, 2025.

### A. Proofs

We provide complete proofs of all theoretical results stated in Section 3.3. We use Assumptions 3.1–3.3 (and Assumption 3.4 for the teacher-mismatch results) as stated in Section 3.3. We begin with a fundamental preliminary result and then prove each theorem in order.

Notation. Let Ω be a finite vocabulary. A prompt 𝑞 ∈ Ω* is drawn from 𝑝(𝑞). A response 𝑥 = (𝑎1,...,𝑎𝑇) is generated as 𝜋𝜃(𝑥 | 𝑞) =

∏︀𝑇

𝑡=1 𝜋𝜃(𝑎𝑡 | 𝑠𝑡), where 𝑠𝑡 = (𝑞,𝑎1,...,𝑎𝑡−1). The sequence-level IS ratio is 𝑤(𝑥;𝜃) := 𝜋𝜃(𝑥 | 𝑞)/𝜋ref(𝑥 | 𝑞).

##### A.1. Preliminary: Importance-Sampling Decomposition

We begin with a fundamental identity relating the online and offline gradients via IS reweighting. This is used in all subsequent proofs.

Proposition A.1 (IS Decomposition). Under Assumption 3.2, the online gradient equals the IS-reweighted offline expectation:

[︃𝑤(𝑥;𝜃) ·

𝐴𝑡(𝜃) · ∇log 𝜋𝜃(𝑎𝑡 | 𝑠𝑡)]︃.

∑︁𝑇

∇𝐽on(𝜃) = E𝑥∼𝜋

ref

𝑡=1

∑︀

The offline gradient ∇𝐽off(𝜃) = E𝑥∼𝜋

[

𝑡 𝐴𝑡(𝜃) · ∇log 𝜋𝜃(𝑎𝑡 | 𝑠𝑡)] is the special case 𝑤 ≡ 1. Proof. Since Ω is a finite vocabulary, all expectations are finite sums. For any function 𝑓(𝑥):

ref

𝜋𝜃(𝑥) 𝜋ref(𝑥) · 𝑓(𝑥) = E𝑥∼𝜋

∑︁

∑︁

[𝑓(𝑥)] =

𝜋𝜃(𝑥)𝑓(𝑥) =

[𝑤(𝑥;𝜃)𝑓(𝑥)],

𝜋ref(𝑥) ·

E𝑥∼𝜋

ref

𝜃

𝑥

𝑥

where the second equality requires 𝜋ref(𝑥) > 0 wherever 𝜋𝜃(𝑥) > 0, which holds by Assumption 3.2. Applying this with 𝑓(𝑥) =

∑︀

𝑡 𝐴𝑡(𝜃) · ∇log 𝜋𝜃(𝑎𝑡 | 𝑠𝑡) (stop-gradient applied to 𝐴𝑡(𝜃)) gives the result.

| |
|---|

∑︀

Remark on the surrogate gradient. The per-trajectory gradient 𝑓(𝑥;𝜃) =

𝑡 𝐴𝑡(𝜃) · ∇log 𝜋𝜃(𝑎𝑡 | 𝑠𝑡) treats the advantage 𝐴𝑡(𝜃) as a fixed scalar, i.e., stop-gradient is applied to 𝐴𝑡. This is the standard update rule used in all OPD implementations [8, 9, 22] and yields an advantage-weighted policy gradient. We denote these surrogate gradients by ∇𝐽on and ∇𝐽off throughout for notational consistency with the OPD literature.

##### A.2. Proof of Theorem 3.5

∑︀𝑇

Proof. Define the per-trajectory gradient 𝑓(𝑥;𝜃) :=

𝑡=1 𝐴𝑡(𝜃) · ∇log 𝜋𝜃(𝑎𝑡 | 𝑠𝑡). By Proposition A.1: ∇𝐽on(𝜃) − ∇𝐽off(𝜃) = E𝑥∼𝜋

[(𝑤(𝑥;𝜃) − 1)𝑓(𝑥;𝜃)]. Applying the Cauchy–Schwarz inequality to the expectation:

ref

[(𝑤 − 1)2] · √︁E𝜋

[(𝑤 − 1)𝑓]‖2 ≤ √︀E𝜋

[‖𝑓‖22].

‖E𝜋

ref

ref

ref

∑︀

##### First factor. Since E𝜋

𝑥 𝜋𝜃(𝑥) = 1: E𝜋

[𝑤(𝑥;𝜃)] =

ref

[︀

]︀

[︀

]︀ − 2E𝜋

[︀

]︀ − 1 = 𝜒2(𝜋𝜃 ‖𝜋ref).

(𝑤 − 1)2

𝑤2

𝑤2

= E𝜋

[𝑤] + 1 = E𝜋

ref

ref

ref

ref

Second factor. By Assumption 3.3, ‖∇log 𝜋𝜃(𝑎𝑡 | 𝑠𝑡)‖2 ≤ 𝐺 for each 𝑡. Applying the triangle inequality:

∑︁𝑇

∑︁𝑇

‖𝑓(𝑥;𝜃)‖2 =

𝐴𝑡(𝜃) · ∇log 𝜋𝜃(𝑎𝑡 | 𝑠𝑡)

|𝐴𝑡(𝜃)|.

≤ 𝐺 ·

⃦

⃦

𝑡=1

𝑡=1

2

Therefore:

⎡ ⎣

(︃ 𝑇

∑︁

[︀‖𝑓(𝑥;𝜃)‖22]︀ ≤ 𝐺2 · E𝜋

###### E𝜋

ref

ref

𝑡=1

where the last inequality uses Assumption 3.1.

)︃2⎤ ⎦ ≤ 𝐺2 𝜎𝐴2 ,

|𝐴𝑡(𝜃)|

##### Combining.

‖∇𝐽on(𝜃) − ∇𝐽off(𝜃)‖2 ≤ √︀𝜒2(𝜋𝜃 ‖𝜋ref) · 𝐺𝜎𝐴. □

| |
|---|

Corollary A.2 (Zero Gap at Initialization). When 𝜋𝜃 = 𝜋ref, 𝜒2(𝜋𝜃 ‖𝜋ref) = 0, so ‖∇𝐽on(𝜃)−∇𝐽off(𝜃)‖2 = 0. Under standard smoothness conditions, the gap grows at rate 𝑂(𝜂 𝑘) after 𝑘 gradient steps with step size 𝜂.

[𝑤2] = 1 and 𝜒2 = 0. The growth rate follows from a first-order Taylor expansion of 𝜒2(𝜋𝜃−𝜂𝑔‖𝜋ref) around 𝜃.

Proof. At 𝜋𝜃 = 𝜋ref, 𝑤(𝑥;𝜃) = 1 for all 𝑥, so E𝜋

ref

| |
|---|

Remark on 𝜒2 versus KL. A bound via Pinsker’s inequality yields ‖∇𝐽on−∇𝐽off‖2 ≤ 𝑀𝑇𝐺√︀

2KL(𝜋𝜃‖𝜋ref),

where 𝑀 = sup𝑡 |𝐴𝑡(𝜃)|. This is vacuous in practice because 𝑀 diverges whenever 𝜋𝜃(𝑎𝑡|𝑠𝑡) → 0. Theorem 3.5 avoids this by using the 𝐿2 constant 𝜎𝐴 ≪ 𝑀𝑇 and replacing KL with 𝜒2. Although 𝜒2 ≥ KL, advantage clipping and KL regularization keep 𝜒2 small throughout training.

- A.3. Proof of Theorem 3.6 Proof. We first establish the identity 𝐽on(𝜃) = −KL(𝜋𝜃 ‖𝜋𝑇). By definition:

[︃ 𝑇

]︃ = E𝑥∼𝜋

]︂ = −KL(𝜋𝜃 ‖𝜋𝑇),

[︂log

𝜋𝑇(𝑥 | 𝑞) 𝜋𝜃(𝑥 | 𝑞)

∑︁

(︀

)︀

𝐽on(𝜃) = E𝑥∼𝜋

log 𝜋𝑇(𝑎𝑡 | 𝑠𝑡) − log 𝜋𝜃(𝑎𝑡 | 𝑠𝑡)

𝜃

𝜃

𝑡=1

∏︀

where we used the autoregressive factorization 𝜋𝜃(𝑥 | 𝑞) =

𝑡 𝜋𝜃(𝑎𝑡 | 𝑠𝑡). Since KL ≥ 0, we have 𝐽on(𝜃) ≤ 0 with equality if and only if 𝜋𝜃 = 𝜋𝑇 on supp(𝜋𝜃). Therefore, any 𝜃* ∈ arg min𝜃∈Θ KL(𝜋𝜃 ‖𝜋𝑇) is a global maximizer of 𝐽on.

Achievable case (𝜋𝑇 ∈ ΠΘ). When 𝜋𝑇 ∈ ΠΘ, the global maximizer 𝜃* achieves 𝜋𝜃* = 𝜋𝑇, so 𝐴𝑡(𝜃*) = log 𝜋𝑇(𝑎𝑡 | 𝑠𝑡) − log 𝜋𝜃*(𝑎𝑡 | 𝑠𝑡) = 0 for all 𝑡 and all (𝑞,𝑥). Both the online and offline OPD updates vanish:

[︃

0 · ∇log 𝜋𝜃*(𝑎𝑡 | 𝑠𝑡)]︃ = 0, ∇𝐽off(𝜃*) = E𝜋

[︃

0 · ∇log 𝜋𝜃*(𝑎𝑡 | 𝑠𝑡)]︃ = 0.

∑︁

∑︁

∇𝐽on(𝜃*) = E𝜋

ref

𝜃*

𝑡

𝑡

Thus 𝜃* is a shared fixed point of both standard OPD and Lightning OPD.

Capacity-limited case (𝜋𝑇 ∈/ ΠΘ). When 𝜋𝑇 cannot be exactly represented in ΠΘ, the shared fixedpoint argument above does not directly apply. However, the irreducible approximation error 𝜀approx = min𝜃∈Θ KL(𝜋𝜃 ‖𝜋𝑇) > 0 lower-bounds both methods, and Theorem 3.5 ensures that the per-step gradient discrepancy remains controlled by the policy drift 𝜒2(𝜋𝜃 ‖𝜋ref). As confirmed empirically in Figure 3a, this drift stays small throughout training, so the offline and online updates remain close in practice.

| |
|---|

Heuristic error decomposition. Combining Theorems 3.5 and 3.6, we can informally decompose the final student’s divergence from the teacher as:

+ 𝑂(︁𝐺𝜎𝐴√︀𝜒2(𝜋𝜃‖𝜋ref))︁ ⏟ ⏞

KL(𝜋𝜃

final ‖𝜋𝑇) ≈ 𝜀approx ⏟ ⏞

+ 𝜀opt ⏟ ⏞

,

optimisation error (→0 with training)

model capacity (irreducible)

offline-online distribution gap

where 𝜀approx = min𝜃∈Θ KL(𝜋𝜃‖𝜋𝑇). Switching from offline to online rollouts reduces only the third term, not 𝜀approx, which is the dominant term determined by model capacity. This decomposition is not a formal bound for the surrogate gradient dynamics, but provides useful intuition for why Lightning OPD performs comparably to standard OPD in practice.

##### A.4. Proof of Theorem 3.7 (Gradient Decomposition)

∑︀

Proof. Recall 𝑓(𝑥;𝜃) =

𝑡 𝐴𝑡(𝜃) · ∇log 𝜋𝜃(𝑎𝑡 | 𝑠𝑡). By Proposition A.1: ∇𝐽on(𝜃) = E𝜋

[𝑤(𝑥;𝜃) · 𝑓(𝑥;𝜃)]. Using the covariance decomposition E[𝑤𝑓] = E[𝑤]E[𝑓] + Cov[𝑤,𝑓] and E𝜋

ref

[𝑤] = 1: ∇𝐽on(𝜃) = E𝜋

ref

[𝑤(𝑥;𝜃), 𝑓(𝑥;𝜃)]. Rearranging gives ∇𝐽off(𝜃) = ∇𝐽on(𝜃) − Cov𝜋

[𝑓] + Cov𝜋

[𝑤, 𝑓] = ∇𝐽off(𝜃) + Cov𝜋

ref

ref

ref

[𝑤,𝑓].

ref

Regularization effect of the covariance term. At 𝜋𝜃 = 𝜋ref, 𝑤 ≡ 1 is constant so Cov[𝑤,𝑓] = 0 (the zero vector) and ∇𝐽off = ∇𝐽on exactly. As 𝜋𝜃 drifts from 𝜋ref, 𝑤 becomes large on trajectories that 𝜋𝜃 over-weights relative to 𝜋ref; on those same trajectories 𝑓 (a gradient vector) also tends to have large magnitude. The covariance vector Cov[𝑤,𝑓] therefore grows in norm with drift and, when projected onto the drift direction 𝜃 − 𝜃ref, opposes further movement away from 𝜋ref. This produces a restoring-force effect: ∇𝐽off subtracts a component aligned with the drift direction from ∇𝐽on, implicitly penalizing deviation from 𝜋ref without requiring an explicit KL regularization term.

| |
|---|

##### A.5. Proof of Theorem 3.8

We consider the case where the SFT-stage teacher 𝜋𝑇SFT and the OPD-stage reference teacher 𝜋𝑇OPD differ. The SFT model 𝜋ref was trained on data generated by 𝜋𝑇SFT, so its distribution concentrates on trajectories preferred by 𝜋𝑇SFT. However, the advantage function 𝐴𝑡(𝜃) = log 𝜋𝑇OPD(𝑎𝑡 | 𝑠𝑡) − log 𝜋𝜃(𝑎𝑡 | 𝑠𝑡) reflects 𝜋𝑇OPD. This mismatch introduces an irreducible additive bias.

Proof. Define 𝛿 := 𝜒2(𝜋𝑇SFT‖𝜋𝑇OPD) and Δ𝑡 := log 𝜋𝑇SFT(𝑎𝑡 | 𝑠𝑡) − log 𝜋𝑇OPD(𝑎𝑡 | 𝑠𝑡). The advantage under teacher inconsistency decomposes as:

)︀ ⏟ ⏞

(︀

log 𝜋𝑇SFT(𝑎𝑡 | 𝑠𝑡) − log 𝜋𝜃(𝑎𝑡 | 𝑠𝑡)

𝐴𝑡(𝜃) =

−Δ𝑡.

consistent advantage 𝐴cons𝑡 (𝜃)

∑︀

𝑡 𝐴cons𝑡 (𝜃) · ∇log 𝜋𝜃(𝑎𝑡 | 𝑠𝑡) and 𝑓Δ(𝑥;𝜃) = −∑︀

Correspondingly, define 𝑓cons(𝑥;𝜃) =

𝑡 Δ𝑡 · ∇log 𝜋𝜃(𝑎𝑡 | 𝑠𝑡),

so 𝑓 = 𝑓cons + 𝑓Δ. By Proposition A.1 and the triangle inequality:

‖∇𝐽on(𝜃) − ∇𝐽off(𝜃)‖2 = ‖E𝜋

[(𝑤 − 1)𝑓]‖2 ≤ ‖E𝜋

[(𝑤 − 1)𝑓cons]‖2

+‖E𝜋

[(𝑤 − 1)𝑓Δ]‖2

.

ref

ref

ref

⏟ ⏞

⏟ ⏞

(I)

(II)

- Term (I): Consistent part. This is identical to the proof of Theorem 3.5 with the consistent advantage, giving:

(I) ≤ 𝐺𝜎𝐴 · √︀𝜒2(𝜋𝜃 ‖𝜋ref).

- Term (II): Mismatch part. By the triangle inequality and Assumption 3.3, ‖𝑓Δ(𝑥;𝜃)‖2 ≤ 𝐺 · ∑︀

𝑡 |Δ𝑡|. Applying Cauchy–Schwarz:

⎯ ⎸ ⎷E𝜋

)︃2⎤ ⎦ ≤ 𝐺𝜎Δ · √︀𝜒2(𝜋𝜃‖𝜋ref),

⎡ ⎣

(︃

∑︁

(II) ≤ √︀𝜒2(𝜋𝜃‖𝜋ref) · 𝐺 ·

|Δ𝑡|

ref

𝑡

where the last step applies Assumption 3.4. Combining Terms (I) and (II):

‖∇𝐽on(𝜃) − ∇𝐽off(𝜃)‖2 ≤ 𝐺 · (𝜎𝐴 + 𝜎Δ) · √︀𝜒2(𝜋𝜃 ‖𝜋ref). At 𝜋𝜃 = 𝜋ref (where 𝑤 ≡ 1), 𝜒2 = 0 and the online-offline gap is exactly zero, regardless of teacher mismatch.

Offline gradient bias. Separately, the mismatched offline gradient carries a persistent bias relative to the consistent offline gradient. Writing ∇𝐽off = E𝜋

[𝑓cons + 𝑓Δ] and ∇𝐽off𝛿=0 = E𝜋

[𝑓cons], their difference is E𝜋

ref

ref

[𝑓Δ], which can be nonzero when 𝜎Δ > 0. By the triangle inequality, Jensen’s inequality, and Assumption 3.4:

ref

[︃

]︃

∑︁

⃦∇𝐽off − ∇𝐽off𝛿=0⃦

2 = ‖E𝜋

[𝑓Δ]‖2 ≤ E𝜋

[‖𝑓Δ‖2] ≤ 𝐺 · E𝜋

|Δ𝑡|

≤ 𝐺𝜎Δ.

ref

ref

ref

𝑡

This bias is independent of 𝜒2 and persists even at initialization, corrupting the offline update direction throughout training.

| |
|---|

##### A.6. Proof of Theorem 3.9

Proof. Let ∇𝐽on𝛿=0(𝜃) denote the standard OPD gradient under a consistent teacher (𝜋𝑇SFT = 𝜋𝑇OPD), and let ∇𝐽on(𝜃) denote the gradient under the mismatched teacher 𝜋𝑇OPD ̸= 𝜋𝑇SFT. For any iterate 𝜃 reachable from 𝜋ref, their difference is:

[︃

)︀ · ∇log 𝜋𝜃(𝑎𝑡 | 𝑠𝑡)]︃

∑︁

(︀

∇𝐽on(𝜃) − ∇𝐽on𝛿=0(𝜃) = E𝑥∼𝜋

𝐴OPD𝑡 (𝜃) − 𝐴SFT𝑡 (𝜃)

𝜃

𝑡

[︃

Δ𝑡 · ∇log 𝜋𝜃(𝑎𝑡 | 𝑠𝑡)]︃,

∑︁

= −E𝑥∼𝜋

𝜃

𝑡

where Δ𝑡 = log 𝜋𝑇SFT(𝑎𝑡 | 𝑠𝑡) − log 𝜋𝑇OPD(𝑎𝑡 | 𝑠𝑡). By the triangle inequality and Assumption 3.3, ‖∑︀

𝑡 Δ𝑡 · ∇log 𝜋𝜃(𝑎𝑡 | 𝑠𝑡)‖2 ≤ 𝐺 · ∑︀

𝑡 |Δ𝑡|. Taking the norm outside the expectation and applying Jensen’s inequality:

⎯ ⎸ ⎷E𝑥∼𝜋

⎡ ⎣

)︃2⎤ ⎦.

[︃

]︃

(︃

∑︁

∑︁

⃦∇𝐽on(𝜃) − ∇𝐽on𝛿=0(𝜃)⃦

|Δ𝑡|

|Δ𝑡|

2 ≤ 𝐺 · E𝑥∼𝜋

≤ 𝐺 ·

𝜃

𝜃

𝑡

𝑡

Since 𝜃 is reachable from 𝜋ref, by Assumption 3.2 𝜋𝜃 ≪ 𝜋ref. Applying the IS identity (Proposition A.1):

⎡ ⎣

)︃2⎤ ⎦ = E𝑥∼𝜋

⎡ ⎣𝑤(𝑥;𝜃)(︃

)︃2⎤ ⎦.

(︃

∑︁

∑︁

|Δ𝑡|

|Δ𝑡|

E𝑥∼𝜋

ref

𝜃

𝑡

𝑡

∑︀

𝑡 |Δ𝑡|)2] ≤ 𝜎Δ2 by Assumption 3.4. Substituting:

At initialization 𝜃 = 𝜃ref, we have 𝑤 ≡ 1, so this reduces exactly to E𝜋

[(

ref

⃦∇𝐽on(𝜃ref) − ∇𝐽on𝛿=0(𝜃ref)⃦

2 ≤ 𝐺𝜎Δ. □

| |
|---|

Remark (extension beyond initialization). For iterates 𝜃 with bounded drift (𝜒2(𝜋𝜃 ‖𝜋ref) ≤ 𝐶), we expect the bound to remain of similar order. Intuitively, Δ𝑡 depends only on the two teachers and the token sequence, not on 𝜃, so the covariance between 𝑤 and (

∑︀|Δ𝑡|)2 stays small when drift is small. The term 𝐺𝜎Δ thus biases the gradient of standard OPD whenever 𝜎Δ > 0, degrading the effective convergence point relative to the fixed point of Theorem 3.6.

### B. Implementation Details

Tables 5 and 6 list the full hyperparameter configurations for the SFT and OPD stages respectively. For the OPD stage, the same hyperparameters are used for both the math and code domains. We set the maximum response length to 4,096 tokens during OPD training. Although the evaluation generation length is significantly longer (up to 40,960 tokens for code benchmarks), we find that training with 4,096 tokens already achieves optimal performance while offering substantially better training efficiency; increasing the rollout length beyond this threshold does not improve results. For code generation, the OPD stage is initialized from the math-trained OPD checkpoint rather than the SFT checkpoint directly. We find this consistently outperforms initializing code OPD from the SFT model, consistent with prior findings that math reasoning training provides a stronger initialization for code training [73].

Table 5 | SFT stage hyperparameters.

Hyperparameter 4B Scale 8B Scale

Training steps 3000 3000 Global batch size 256 128 Max sequence length 16384 16384 Learning rate 8 × 10−5 8 × 10−5 LR schedule cosine cosine Warmup ratio 0.1 0.1 Packing ✓ ✓ DeepSpeed stage ZeRO-0 ZeRO-1

Table 6 | OPD stage hyperparameters for standard OPD and Lightning OPD. The same configuration is used for both the math and code domains.

Hyperparameter 4B Scale 8B Scale

Training steps 150 150 Global batch size 256 256 Max response length 4096 4096 Learning rate 2 × 10−6 2 × 10−6 LR schedule constant constant Weight decay 0.1 0.1

- Adam 𝛽1 0.9 0.9
- Adam 𝛽2 0.98 0.98 Rollout temperature 0.8 0.8 Rollout top-𝑝 1.0 1.0 Advantage clip range [−10,10] [−10,10] Tensor parallel size 2 4

### C. Additional Experimental Analysis

##### C.1. Training Dynamics

Figure 3 examines the internal dynamics of Lightning OPD across both stages of training. Figure 3a tracks the per-token importance weight 𝑤𝑡 = 𝜋𝜃/𝜋ref throughout OPD training. The mean drops from 1 to 0.94 within the first 20 steps and then plateaus, while the standard deviation rises sharply in the same early phase before stabilizing at a moderate level. The mean remaining close to 0.94 indicates that the student policy stays near the reference distribution throughout training, and the standard deviation stabilizing below 0.1 indicates that per-token distributional shift is consistently small. Both quantities remaining bounded confirms the implicit regularization of Theorem 3.7, which shows that fixing rollouts to 𝜋ref automatically constrains both the magnitude and spread of policy drift without any explicit KL penalty. Figure 3b shows the AIME

0.15

Importance Weight Mean

1.00

Importance Weight Std

ImportanceWeightMean

0.12

ImportanceWeightStd

0.98

0.09

0.96

0.06

0.94

0.03

0.92

0.00

0 30 60 90 120 150

Training Step

(a) Importance weight dynamics.

72

| |68.1| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| |56.7| | | | |
| | | | | | |

AIME2024Score(%)

68

64

60

56

0 30 60 90 120 150

Training Step

(b) AIME 2024 score vs. training step.

72

67.7 68.1

65.2 65.6

AIME2024Score(%)

62.3

64

60.7

65.4

64.4 64.6

63.6

61.1

56

56.4 56.0 56.0 56.7

54.7

48

SFT

51.0

42.3

OPD

Lightning OPD

40

500 1000 1500 2000 2500 3000

SFT Steps

(c) AIME 2024 score vs. SFT steps.

Figure 3 | Training dynamics of Lightning OPD (Qwen3-4B-Base student). (a) The mean importance weight 𝑤𝑡 = 𝜋𝜃/𝜋ref drops rapidly to ≈ 0.94 within the first 20 steps and plateaus, with its standard deviation stabilizing concurrently, validating the implicit regularization in Theorem 3.7. (b) AIME 2024 pass@1 rises steeply in the first 50 OPD steps and saturates thereafter, confirming that 150 steps is a sufficient training budget. (c) AIME 2024 pass@1 of the SFT model, standard OPD, and Lightning OPD as a function of SFT checkpoint quality. All three improve consistently with more SFT steps, and both OPD variants provide a large, stable gain on top of the SFT baseline at every checkpoint.

2024 score throughout OPD training. The score converges remarkably fast, with the student capturing nearly all of its performance gain within the first 50 steps and remaining stable thereafter, justifying our choice of 150 steps as a sufficient training budget. Figure 3c shows the effect of SFT checkpoint quality on the final model. All three curves improve consistently with more SFT steps, and both OPD variants provide a large, stable gain on top of the SFT baseline at every checkpoint. The relative ordering of the three methods remains consistent across all SFT budgets, indicating that Lightning OPD is robust to the choice of SFT training length.

### D. Extended Discussion

We discuss how Lightning OPD relates to two superficially similar paradigms, offline RL and offline knowledge distillation, and clarify why neither subsumes our approach.

Relation to Offline RL. Lightning OPD resembles offline RL in that both optimize over a fixed dataset, but the resemblance is superficial and a direct application of offline RL techniques to OPD would fail for reasons that offline RL methods are not designed to address. The central challenge of offline RL is OOD action overestimation arising from sparse reward signals, which offline RL methods address through conservatism mechanisms such as value pessimism or policy constraint. Neither problem exists in Lightning OPD, where the teacher supplies dense per-token log-probability supervision for all token sequences, leaving no OOD region and no sparse reward to cause high-variance estimation. Conservatism is therefore neither necessary nor applicable. The real obstacle to offline OPD is teacher inconsistency. When 𝜎Δ > 0, the irreducible bias 𝐺𝜎Δ is a structural property of the gradient field itself, not an estimation artifact from limited data coverage, and no importance sampling correction or conservatism mechanism can remove it. Beyond the challenge, the two paradigms also differ in the nature of their solutions. Offline RL fixed points are data-limited, shaped by the coverage of the behavior policy, so better data coverage leads to better policies. Theorem 3.6 and our empirical results suggest that Lightning OPD’s performance ceiling is primarily capacity-limited, determined by model capacity ΠΘ relative to the teacher rather than by the rollout distribution. Improving the rollout distribution cannot push past the teacher’s own capability, and the right lever is model capacity, not data coverage. In summary, Lightning OPD is not a variant of offline RL applied to distillation. It is a principled offline approximation to an on-policy distillation objective, one whose unique challenge is teacher inconsistency rather than distributional coverage, and one that recovers standard OPD when teacher consistency is enforced.

Relation to Offline Knowledge Distillation. Lightning OPD also differs fundamentally from offline (off-policy) knowledge distillation [49, 50], despite the shared use of precomputed teacher signals. Offline KD

trains the student on teacher-generated sequences, so the student only receives supervision on trajectories the teacher would produce, never on its own mistakes. Lightning OPD instead collects rollouts from the student’s own policy 𝜋ref and evaluates the teacher’s per-token log-probabilities on these student-generated sequences. This means the teacher provides corrective signals on exactly the distribution the student will encounter during inference, which is the core advantage of on-policy methods over off-policy ones [8]. The distinction is preserved in Lightning OPD even though teacher log-probabilities are precomputed. Empirically, the OPD stage provides substantial gains over the SFT baseline across all benchmarks, confirming that on-policy supervision on student rollouts extracts significantly more from the teacher than offline KD on teacher-generated data alone. This is consistent with Theorem 3.6: Lightning OPD’s performance is primarily capacity-limited, depending on how well 𝜋𝜃 can approximate 𝜋𝑇, rather than being restricted to the support of teacher-generated data.

Relation to Rang et al. [26]. Rang et al. [26] also precompute teacher signals over student-generated responses and train without a live teacher server. While the high-level motivation of offline on-policy distillation is shared, the two approaches differ in a fundamental way: the coupling between SFT and distillation. Rang et al. treat SFT and knowledge distillation as independent sequential stages: SFT is performed on independently curated data, and the distillation teacher is a separate model with no constraint linking the two stages. No analysis is provided for when or why the offline approximation is reliable. A central finding of Lightning OPD is that the two stages must be considered holistically. We identify teacher consistency as an important design principle, requiring that the SFT data be generated by the same teacher used for OPD, and show that violating it introduces a gradient bias that degrades both offline and online OPD (Theorems 3.8–3.9). This bias is a structural property of the gradient field itself, not an artifact of the offline approximation, and no amount of data or training can remove it. Enforcing teacher consistency is sufficient for offline OPD to share the same fixed point as online OPD when the teacher is representable (Theorems 3.5, 3.6, 3.7), providing formal guarantees that are absent in prior work. Beyond the design principle, the two approaches also differ in the distillation paradigm. Rang et al. formulate distillation as supervised learning with a composite cross-entropy and KL loss using soft teacher logits, whereas Lightning OPD formulates it as policy gradient optimization where the per-token advantage log 𝜋𝑇 − log 𝜋𝜃 drives the gradient update. The policy gradient formulation enables direct integration with standard OPD and RLVR post-training infrastructure, and is what makes the theoretical analysis (gradient discrepancy bounds, shared fixed points, implicit regularization) tractable.

Evaluation Templates. We use the following prompt templates for evaluation. For mathematical reasoning benchmarks, the prompt follows the Qwen3 chat format:

<|im_start|>user Question: {problem} Please reason step by step, and put your final answer within \boxed{}. <|im_end|> <|im_start|>assistant

For code generation benchmarks, the prompt includes a system message and a task description following LiveCodeBench conventions:

<|im_start|>system You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step. <|im_end|> <|im_start|>user You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.

Question: {question} Read the inputs from stdin solve the problem and write the answer to stdout

(do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT. # YOUR CODE HERE

<|im_end|> <|im_start|>assistant

### E. Limitations

Our experiments focus on mathematical reasoning and code generation, both of which benefit from well-defined verifiable evaluation metrics. Extending Lightning OPD to broader post-training tasks such as multi-turn agent interactions, tool use, and open-ended instruction following remains an open direction, as these settings introduce challenges including multi-turn distribution shift and the difficulty of precomputing meaningful teacher signals over long interaction trajectories. Additionally, teacher consistency requires that SFT training data be generated by the same teacher used for OPD; when adopting a new teacher, this necessitates regenerating the SFT dataset, which can be resource-intensive for large teacher models and partially offsets the training-time savings, though it remains a one-time cost amortized over multiple experiments.

