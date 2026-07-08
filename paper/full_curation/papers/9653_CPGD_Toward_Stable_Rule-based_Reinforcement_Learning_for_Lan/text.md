arXiv:2505.12504v1[cs.LG]18May2025

# CPGD: Toward Stable Rule-based Reinforcement Learning for Language Models

Zongkai Liu1,2∗ Fanqing Meng4* Lingxiao Du3* Zhixiang Zhou2* Chao Yu1† Wenqi Shao2,3† Qiaosheng Zhang2,3†

1Sun Yat-Sen University 2Shanghai Innovation Institute 3Shanghai AI Laboratory 4Shanghai Jiao Tong University

## Abstract

Recent advances in rule-based reinforcement learning (RL) have significantly improved the reasoning capability of language models (LMs) with rule-based rewards. However, existing RL methods—such as GRPO, REINFORCE++, and RLOO—often suffer from training instability, where large policy updates and improper clipping can lead to training collapse. To address this issue, we propose Clipped Policy Gradient Optimization with Policy Drift (CPGD), a novel algorithm designed to stabilize policy learning in LMs. CPGD introduces a policy drift constraint based on KL divergence to dynamically regularize policy updates, and leverages a clip mechanism on the logarithm of the ratio to prevent excessive policy updates. We provide theoretical justification for CPGD and demonstrate through empirical analysis that it mitigates the instability observed in prior approaches. Furthermore, we show that CPGD significantly improves performance while maintaining training stability. Our implementation balances theoretical rigor with practical usability, offering a robust alternative for RL in the post-training of LMs. We release our code at https://github.com/ModalMinds/MM-EUREKA.

## 1 Introduction

Rule-based reinforcement learning (RL) has emerged as a key approach for eliciting reasoning capabilities in language models (LMs) [1]. It leverages simple, efficient reward functions derived from deterministic rules, effectively mitigating reward hacking [2] while activating reasoning abilities of models [1, 3, 4, 5]. This has sparked a line of research focused on developing more effective RL algorithms for both textual and general multimodal reasoning tasks. Notable methods include GRPO [1], REINFORCE++ [6], RLOO [7, 8], and GRPO variants such as DAPO [9], Dr.GRPO [10], and GPG [11]. However, we observe that these RL methods often suffer from training instability, which we attribute to the use of importance-sampling ratios in their loss functions. Although PPOclip loss [12] is commonly adopted to mitigate extreme policy updates, its one-sided nature fails to constrain large ratios when the advantage is negative—potentially causing gradient explosions dominated by poor samples, leading to catastrophic training collapse. We theoretically show that incorporating the importance-sampling ratio in the loss can amplify the policy shift, and our empirical results confirm that this can lead to training collapse in existing RL methods.

To address this issue, we propose Clipped Policy Gradient Optimization with Policy Drift (CPGD), an algorithm that replaces the PPO-clip loss with a policy gradient loss [13] to avoid instability caused by directly involving policy ratios in the loss function. To ensure proximal optimization, we introduce both a clip mechanism and a policy drift regularizer, constraining optimization within a local region

∗Equal contribution †Corresponding Authors: {zhangqiaosheng, shaowenqi}@pjlab.org.cn; yuchao3@mail.sysu.edu.cn

Preprint. Under review.

and mitigating over-optimization that may impair reasoning behaviors as shown in Section 4.2. Furthermore, we develop a novel KL estimator that ensures correct gradient directions while avoiding the potential numerical instability associated with the commonly used k3 estimators [14]. We also incorporate weighted advantages to dynamically adjust the influence of each sample, further enhancing model performance.

We theoretically prove the convergence of CPGD and empirically demonstrate its superior training stability and performance. As shown in Table 1, models trained with CPGD consistently outperform those trained with other RL algorithms and strong open-source baselines across standard multimodal reasoning benchmarks. Notably, CPGD improves the overall performance over the base model by +11.0% across all benchmarks. Specially, CPGD achieves +21.8% gain on the in-domain benchmark MMK12 [15], and improves by +8.5% and +11.4% on the out-of-distribution benchmarks MathVista [16] and MathVision [17], respectively.

## 2 Related work

RL for training reasoning models. RL has become a key method for improving reasoning in LMs [1, 18]. While early methods rely on PPO [12], its high computational cost has driven interest in alternatives like DPO [19], which simplifies training but depends on offline data. Recent RL methods such as GRPO, RLOO, and REINFORCE++ aim to balance stability and efficiency. Notably, DeepSeek R1 [1] shows that pure RL can elicit self-reflection and reasoning in LMs without supervised pretraining. Recently, several concurrent works have proposed GRPO variants to address its limitations. For instance, Dr.GRPO [10] identifies optimization bias in GRPO that favors longer response among incorrect ones. DAPO [9] incorporates multiple improvements, including decoupled clipping thresholds, token-level losses, and an online filtering strategy. GPG [11], in contrast, adopts a minimalist design by discarding both clipping and KL regularization, relying solely on the policy gradient loss [13]. However, none of these approaches fundamentally resolve the training instability issue to existing RL methods, which is the primary focus of this work.

Large reasoning model. Recently, a surge of reasoning models has emerged, driven by the principle of test-time scaling laws, which demonstrate that models with explicit reasoning processes achieve superior performance [20]. Leading models in this area include DeepSeek R1 [1], OpenAI’s oseries [18], Qwen series [21, 22], and Kimi k1.5 [23]. However, their training pipelines and datasets remain undisclosed. This has motivated a wave of academic research within the open-source community, including parallel efforts such as OpenR1 [24], TinyZero [25], LMM-R1 [26], R1-V [27], Reason-RFT [28], and MM-Eureka [15]. These works primarily focus on constructing high-quality datasets and complete training pipelines. They commonly adopt GRPO to enhance reasoning capabilities but do not specifically investigate improvements to the RL algorithms themselves.

## 3 Preliminaries

### 3.1 Problem formulation

We denote an LM by πθ, where θ ∈ Rd represents the model parameters. Given a prompt x = [x1,...,xm] ∈ D, the model generates a response y = [y1,...,yn] by sampling from the conditional distribution πθ(·|x), with both xi and yi drawn from a finite vocabulary V. In this work, we focus on transformer-based LMs that generate responses autoregressively, such that πθ(y|x) =

- n i=1 πθ(yi|x,y<i), where y<i = [y1,...,yi−1] and y<1 is an empty sequence.

RL in post-training is typically modeled as a Markov decision process (MDP), defined by a tuple M = (S,A,P,R,ρ), where S is the state space, A is the action space, P is the transition kernel, R is the deterministic reward function, and ρ is the initial state distribution. For LMs, two MDP formulations are widely considered: token-level MDP and response-level MDP.

In a token-level MDP, each token is treated as a single action. At the time step t, the state st = [x,y<t] includes the prompt and the tokens generated so far. The action at = yt is sampled according to yt ∼ πθ(·|x,y<t), where the action space A is equal to the vocabulary V. The environment transitions deterministically to st+1 = [x,y<t+1]. The reward is defined as R(st,at) = R([x,y<t],yt), and ρ is induced by the prompt distribution in D.

In a response-level MDP, the full response is treated as an individual action: a = y ∼ πθ(·|x). The state is defined solely by the prompt s = x, and the episode terminates after one step. Thus, the transition kernel is omitted in the single-turn dialogue setting. The reward is R(s,a) = R(x,y), with ρ again determined by D.

### 3.2 Rule-based reinforcement learning

This work focuses on verifiable tasks, where the outcome reward is determined by the final accuracy. Specifically, a response y receives a reward of 1 if it is the correct answer to the prompt x, and 0 otherwise. We denote this reward function as Ro to emphasize its nature as an outcome-based reward. Within this setting, REINFORCE-style algorithms are favored as they reduce computational cost by forgoing critic networks. Notable methods include REINFORCE++ [6], RLOO [7, 8], and GRPO [1].

REINFORCE++: REINFORCE++ enhances the standard REINFORCE framework by integrating key optimizations from PPO [12], improving both stability and efficiency. The objective is defined as:

|y|

πθ(yi|x,y<i) πθ

1 |y|

AR++i ,

LR++(θ;θold) = Ex∼D,y∼π

min

θold(·|x)

(yi|x,y<i)

old

i=1

πθ(yi|x,y<i) πθ

AR++i ,

clip1+1−ϵϵ

(yi|x,y<i)

old

where ϵ ∈ [0,1], clipba(x) := max(min(x,b),a), and

|y|

(yj|x,y<j) πref(yj|x,y<j)

πθ

AR++i := GlobalNorm G(x,y≤i) , G(x,y≤i) := Ro(x,y) − β

old

ln

.

j=i

old(yj|x,y<j)

Here, ln πθ

πref(yj|x,y<j) is the token-level KL penalty, constraining divergence from the reference policy πref, typically the initial model. GlobalNorm(x) = x−mean({x

′∈ batch})

std({x′∈ batch}) is the normalization operation across the global batch for all prompts.

RLOO: The primary distinction between RLOO and REINFORCE++ lies in their computation of the advantage value. RLOO first generates a group of K responses {y(k)}Kk=1 for each prompt x and computes the advantage using a leave-one-out strategy to reduce the gradient variance:

1 K − 1 k

′) ≤i ).

ARLOOi,k := GlobalNorm G ˜(x,y≤(ki)) , G˜(x,y≤(ki)) := G(x,y≤(ki)) −

G(x,y(k

′̸=k

GRPO: GRPO introduces a group-based advantage and employs an external KL regularization via the k3 estimator [14], which approximates DKL(p,q) = i(qi/pi − 1 − lnqi/pi). The loss is:

|y(k)|

K

1 K

1 |y(k)|

− β · Miθ,ref(x,y(k))

LGRPO(θ;θold) = Ex∼D,{y(k)}Kk=1∼πθold(·|x)

i=1

k=1

πθ(yi(k)|x,y<i(k)) πθ

πθ(yi(k)|x,y<i(k)) πθ

AGRPOk , where

AGRPOk ,clip1+1−ϵϵ

+ min

(yi(k)|x,y<i(k))

(yi(k)|x,y<i(k))

old

old

AGRPOk := GroupNorm(Ro(x,y(k))) = Ro(x,y(k)) − mean({Ro(x,y(k))}Kk=1) std({Ro(x,y(k))}Kk=1)

,

πref(yi(k)|x,y<i(k)) πθ(yi(k)|x,y<i(k))

πref(yi(k)|x,y<i(k)) πθ(yi(k)|x,y<i(k))

Miθ,ref(x,y(k)) :=

− 1 − ln

.

## 4 The proposed method

This section introduces our RL algorithm, Clipped Policy Gradient Optimization with Policy Drift (CPGD), designed to improve the stability of RL training. In Section 4.1, we present the CPGD

algorithm along with its theoretical guarantees, and highlight potential limitations of the standard PPO-clip loss. In Section 4.2, we provide empirical evidence of instability in existing methods and analyze its possible causes, showing how CPGD addresses them for more stable training. Finally, Section 4.3 describes the practical implementation of CPGD, striking a balance between theoretical soundness and practical implementation.

- 4.1 Clipped Policy Gradient Optimization with Policy Drift (CPGD) Under the response-level MDP assumption, CPGD aims to maximize the following formula:

,πθ|x) , (1) where

LCPGD(θ;θold) = Ex∼D Ey∼π

θold(·|x) Φθ(x,y) − α · DKL(πθ

old

πθ(y|x) πθ

πθ(y|x) πθ

(y|x) · ACPGD(x,y),clipln(1ln(1−−ϵϵ)) ln

ACPGD(x,y) ,

Φθ(x,y) := min ln

(y|x)

old

old

ACPGD(x,y) := Ro(x,y) − Ey′∼πθ(·|x) Ro(x,y′) , DKL(πθ˜,πθ|x) := Ey∼π

πθ˜(y|x) πθ(y|x)

.

θ˜(·|x) ln

Hereinafter, we term the KL divergence between the old and current policies as policy drift, and between the current and reference policies as reference constraint.

CPGD differs from the standard PPO-clip loss in two key aspects: (1) A different policy optimization objective is used, where the policy gradient loss is adopted with the clip mechanism. (2) A policy drift is introduced, imposing a forward KL divergence penalty between the old and current policies.

Why use the policy gradient objective? In the original PPO objective, although the importancesampling ratio corrects for the distribution mismatch between the old and current policies, it simultaneously introduces high variance. As empirically demonstrated in Section 4.2, such variance can destabilize training and even cause training collapse, while using a policy gradient loss without the ratio substantially improves training stability. Proposition 1 further provides a theoretical explanation for this phenomenon, showing that the use of the policy ratio amplifies policy drift, causing the updated policy to exceed the intended bounds.

Why introduce the policy drift and clip mechanism? The introduction of the clip mechanism and policy drift is designed to ensure proximal policy updates, which are critical for theoretical convergence guarantees in Theorem 1. The clip mechanism enforces local updates by zeroing gradients when the policy ratio exceeds a specified threshold, while policy drift introduces a corrective gradient to constrain the policy ratio within a stable range. Notably, the clip mechanism alleviates the need for a large penalty coefficient on the policy drift term: when the ratio remains within bounds, the small drift coefficient allows the algorithm to focus on optimizing the primary objective Φ; when the ratio exceeds the range, the gradient of the primary objective becomes zero, prompting the algorithm to rely on the policy drift signal to prevent further deviation—particularly those caused by optimizer momentum (e.g., Adam) or neural network generalization effects.

0(y|x) πθold(y|x) − 1| = ϵ. Consider updating θ0 using either (i) the PPO-clip objective, resulting in parameter θ1PPO, or (ii) the CPGD objective with α = 0 (denoted as CPG), yielding parameter θ1CPG. Then, there exists a constant ηmax > 0 such that for any learning rate η ∈ (0,ηmax), the following inequality holds:

Proposition 1. Let θ0 be a parameter such that the importance-sampling ratio satisfies | πθ

(y|x) πθ

(y|x) πθ

πθPPO

πθCPG

(y|x) πθ

πθ

0

(y|x) − 1 >

(y|x) − 1 >

(y|x) − 1 = ϵ.

1

1

old

old

old

After one update step, both PPO and CPG increase the importance-sampling ratio deviation from the old policy, but PPO does so more aggressively than CPG.

The following theorem further presents that CPGD enjoys the convergence guarantee, indicating its theoretical rationality. See Appendix A for the proofs of Proposition 1 and Theorem 1.

### Theorem 1. Let {πθ

k}∞k=0 denote the sequence of policies generated by the CPGD update rule (Equation 1). Then, the sequence πθ

converges.

k

GRPO w/ drift

GRPO w/o clip

GRPO

CPG

CPGD

RLOO

PG

PGD

REINFORCE++

GRPO w/ dual clip

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.20

1000

ResponseLength

ClippingFraction

0.6

800

0.15

Accuracy

600

0.4

0.10

400

0.2

0.05

200

0.0

0.00

0

0 250 500 750 1000 1250

0 250 500 750 1000 1250

0 250 500 750 1000 1250

Training Steps

Training Steps

Training Steps

Figure 1: Accuracy, clipping fraction and response length curves throughout training.

### 4.2 Training collapse

Several studies suggest that the reference constraint may hinder policy improvement [9, 29]. However, we observe that removing this KL term leaves the PPO-clip loss alone insufficient to effectively constrain large policy shifts, which can lead to training collapse. While such collapse may be partially mitigated through techniques such as early stopping or small learning rates, it remains a latent instability that undermines the reliability of continued training. In this subsection, we investigate this phenomenon of training collapse and demonstrate that CPGD effectively prevents it.

Figure 1 presents training curves on the MMK12 dataset [15] for RLOO, REINFORCE++, GRPO, GRPO w/o clip (i.e., GRPO without the clip mechanism), GRPO w/ dual clip (i.e., the policy ratio is additionally clipped to no more than a constant—3.0 in our case—when advantage is negative [30]), GRPO w/ drift (i.e., GRPO with policy drift), PG (basic policy gradient), CPG (PG with the clip mechanism), PGD (PG with the policy drift), and CPGD, all without the reference constraint. We use QwenVL2.5-7B [31] as the base model. All algorithms share the same hyperparameters: a training and rollout batch size of 128, 8 responses per prompt, a learning rate of 1e−6, one PPO epoch, and ten training episodes. As shown in Figure 1, almost all baselines experience training collapse.

As shown in Figure 1, methods such as REINFORCE++, RLOO, GRPO w/o clip, and GRPO exhibit highly unstable policy ratio dynamics, leading to training collapse in mid stages. In contrast, GRPO w/ dual clip, GRPO w/ drift, PG, CPG, PGD, and CPGD maintain stable training curves. GRPO w/ dual clip mitigates instability by globally constraining the policy ratio, while the PG series sidesteps ratio-induced variance by excluding it from the loss computation. These comparisons indicate that incorporating policy ratios in the loss can introduce high variance during fluctuations, and that simple one-sided clipping fails to recover from extreme ratios, ultimately causing collapse. Although dual clip mechanism stabilizes training, it may introduce new issues: frequent zero-gradient updates and ineffective learning under negative advantages due to the zero-gradient clipped large ratios. Additionally, GRPO w/ drift demonstrates that incorporating policy drift effectively constrains the policy ratio within a reasonable range, thereby preventing training collapse.

On the other hand, while prior work suggests clipping may be unnecessary due to the low proportion of clipped ratios [8, 11], our findings suggest otherwise. Despite only ~1% of ratios being clipped, training performance diverges significantly with and without clipping. Specifically, methods like PG and PGD—though stable without ratio terms—suffer from response length collapse, degenerating into trivial outputs (e.g., only emitting tokens like <think>) that exploit the format reward function without performing meaningful reasoning. This highlights the model’s vulnerability to reward hacking, likely due to overly aggressive updates. These results reveal the necessity of the proximal policy updates.

### 4.3 Implementation

In this subsection, we design a practically implementable loss function in per-token form based on the CPGD update formulation (Equation 1), aiming to strike a balance between theoretical rigor and empirical applicability. Our CPGD loss is straightforward to be integrated into widely-used large model training frameworks such as OpenRLHF [32] and veRL [33]. The practical loss function is given by

1 |D|

JCPGD(θ) = −

1

K k=1 |y(k)|

(x,{y(k)}Kk=1)∈D

|y(k)|

i=1

Φiθ(x,y(k))−α·Eθiold,θ(x,y(k)) , (2)

where the per-token policy optimization term is

πθ(yi|x,y<i) πθ

πθ(yi|x,y<i) πθ

ACPGDω (x,y) , and

(yi|x,y<i) ·ACPGDω (x,y),clipln (1+ln (1−ϵϵi)

Φiθ(x,y):=min ln

i) ln

(yi|x,y<i)

old

old

′))}Kk′=1 ,

ACPGDω (x,y(k)) := ω(x) · Ro(x,y(k)) − mean {Ro(x,y(k

sg(πθ(yi|x,y<i)) πθ

Eθiold,θ(x,y) := min

#### (yi|x,y<i) − 1,c · lnπθ(yi|x,y<i).

old

Here, sg(·) denotes the operation that prevents gradient computation, ω(x) is a per-prompt weighting factor, and c > 0 is a constant. We provide the following clarifications regarding the differences between the theoretical update formulation (Equation 1) and the practical loss (Equation 2):

- (I) Policy optimization term: In the theoretical update (Equation 1), the policy optimization term is written in the form of joint distribution. But in the practical implementation (Equation 2), it is decomposed into token level using the decomposability of the logarithm function. Specifically, the clipping threshold ϵi can be set the same for all tokens, ensuring that each token shares the same clip

range. Alternatively, a tight-to-loose schedule can be employed such as ϵi = λϵ + (1 − λ)ϵ · i/|y(k)|, which assigns smaller thresholds to earlier tokens that usually have higher variance.

- (II) Policy drift: Similar to the policy optimization term, policy drift also leverages the decomposability of the logarithm function, but applies the following further transformations:

DKL(πθ

old

,πθ|x) = Ey∼π

θold(·|x) ln

πθ

old

(y|x) πθ(y|x)

= Ey∼π

θold(·|x)

|y|

i=1

ln

πθ

old

(yi|x,y<i) πθ(yi|x,y<i)

(3)

= Ey∼π

θold(·|x)

|y|

i=1

πθ(yi|x,y<i) πθ

old

(yi|x,y<i) − 1 − ln

πθ(yi|x,y<i) πθ

old

(yi|x,y<i)

. (4)

Equations 3 and 4 correspond to the k1 and k3 KL estimators proposed by Schulman [14]. In practice, particularly when using gradient optimizers such as Adam, we prefer the k3 estimator over k1, as k1 fails to effectively constrain the policy drift, while the gradient direction of k3 dynamically adjusts based on the relative magnitude between the current and old policies:

∇θ ln

πθ

old

(yi|x,y<i) πθ(yi|x,y<i)

= −∇θ lnπθ(yi|x,y<i),

∇θ

πθ(yi|x,y<i) πθ

old

(yi|x,y<i) − 1 − ln

πθ(yi|x,y<i) πθ

old

(yi|x,y<i)

=

πθ(yi|x,y<i) πθ

old

(yi|x,y<i) − 1 ∇θ lnπθ(yi|x,y<i).

(5)

However, Equation 4 involves the policy ratio, which can potentially lead to training collapse as discussed in Section 4.2. To mitigate this issue, we clip the policy ratio to be no greater than c + 1. Importantly, this clipping is not applied directly to the KL divergence estimator in Equation 4, but rather to its gradient (Equation 5). This design ensures that when the ratio exceeds the threshold, the policy drift term continues to provide a gradient that reduces the ratio: when sg(π

θ(yi|x,y<i))

πθold(yi|x,y<i) − 1 > c, ∇θEθiold,θ(x,y) = c · ∇θ lnπθ(yi|x,y<i).

In contrast, if clipping were applied to the estimator itself, the resulting gradient −∇θ lnπθ(yi|x,y<i) would further increase the ratio once it exceeds the threshold, exacerbating training instability.

- (III) Weighted advantage: In the view of the response level, each prompt can be viewed as a distinct task. Consequently, we can introduce a per-prompt weighting factor ω(x) to assign different levels of importance to different prompts. (1) Equal weight: when ω(x) = 1, ACPGDω reduces to the original unweighted form. (2) STD weight: when ω(x) = 1/std({R(x,y(k))}k), ACPGDω is the same as

AGRPO. (3) Clip-filter-like weight: when ω(x) = min(cω, #{x∈D|std(#{R{x∈D}

o(x,y(k))}k)̸=0}), cω > 0, similar weighting strategies have also been explored in concurrent work [11], with an analogous effect to online filtering [34], amplifying the gradient contribution of samples with non-zero advantage.

## 5 Experiments

### 5.1 Experiments setup

RL baselines, dataset and implementation details. We compare our CPGD with several widely used RL algorithms, including GRPO [1], REINFORCE++ [6] and RLOO [8] on the MMK12 training dataset [15], which contains 15,616 multimodal math problems with verified answers. All RL algorithms use QwenVL2.5-7B as the base model, trained under the same hyperparameters: rollout and training batch sizes of 128, 8 sampled responses per prompt (temperature 1.0), a learning rate of 1e−6, one PPO epoch, and five training episodes. No reference policy constraint is applied during training, and final performance is reported using the last checkpoint. In our system prompt, reasoning steps and final answers are explicitly marked using <think> and <answer> tags, respectively (see Appendix B).

Benchmarks, model baselines and Overall metric. We evaluate all algorithms on six widely used benchmarks: MathVista (testmini) [16], MathVerse (testmini) [35], MathVision (test) [17], OlympiadBench (EN-OE split) [36], WeMath [37] and MMK12 [15]. MathVista covers visual QA, logic, algebra, and geometry; MathVerse focuses on mathematically grounded visual understanding; and MathVision extends to abstract visual reasoning. OlympiadBench targets graduate-level competition problems, while WeMath enables fine-grained diagnostic analysis via hierarchically annotated tasks. MMK12 provides 500 multiple-choice questions per subject across math, physics, chemistry, and biology for cross-domain performance evaluation.

We also include several multimodal models as baselines. We evaluate open-source models of comparable model size, trained with various strategies, including QwenVL2.5-7B [31], InternVL2.5-8B [38], InternVL2.5-MPO-8B [39], R1-OneVision [40], OpenVLThinker [41], and MM-Eureka [15], which collectively represent the average performance of this model size across the evaluated benchmarks. We further evaluate the leading closed-source models such as GPT-4o [42] and OpenAI-o1 [18] to represent the most outstanding performance that the current state-of-the-art model can achieve on these benchmarks.

To capture overall model performance across N benchmarks, we define an Overall metric by normalizing each score against a strong baseline, QwenVL2.5-7B: Overall := N1 Nj=1 Xj/XjQwen, where Xj and XjQwen are the model and baseline scores on benchmark j.

### 5.2 Main results

Table 1: Performance comparison of various 7B/8B models and leading closed-source models. Top performer is in bold and second-best is underlined (excl. OpenAI-o1/GPT-4o).

Model MathVista MathVerse MathVision Olypamid WeMath MMK12 Overall Leading models

GPT-4o 63.8 50.2 30.4 35.0 68.8 49.9 1.16 OpenAI-o1 73.9 57.0 60.3 68.0 98.7 73.9 1.83

### Similar-size models

InternVL2.5-8B 64.4 39.5 19.7 12.3 53.5 45.6 0.81 QwenVL2.5-7B 68.2 47.9 25.4 20.2 62.1 53.6 1.00 InternVL2.5-MPO-8B 68.9 35.5 21.5 7.8 53.5 34.5 0.75 R1-Onevision (7B) 64.1 47.1 23.5 17.3 61.8 39.8 0.91 OpenVLThinker (7B) 70.2 47.9 25.3 20.1 64.3 60.6 1.03 MM-Eureka (7B) 73.0 50.3 26.9 20.1 66.1 64.5 1.07

### Different RL algorithms on QwenVL2.5-7B

RLOO 68.6 48.3 23.0 19.5 65.8 61.3 1.01 REINFORCE++ 63.9 45.5 18.2 17.8 66.7 64.3 0.96 GRPO 70.3 51.4 25.9 18.5 67.4 65.1 1.06 CPGD (clip-filter-like) 73.4 51.4 25.9 21.5 70.2 67.3 1.10 CPGD (STD weight) 74.0 50.6 28.3 21.4 68.3 65.3 1.11

Table 1 presents a comprehensive comparison across multiple multimodal mathematical benchmarks. Closed-source models GPT-4o and OpenAI-o1 demonstrate strong performance across all tasks, with

- o1 achieving the highest scores overall, notably excelling on MathVision (60.3), Olypamid (68.0) and WeMath (98.7), establishing the current performance upper bound. Among similar-size open models, MM-Eureka shows competitive results. MM-Eureka achieves strong results on MathVista (73.0), MathVision (26.9) and a strong result on MMK12 (64.5). However, our proposed CPGD consistently outperforms all similar-size baselines, achieving top or near-leading scores across all benchmarks, reflecting the effectiveness of our proposed RL algorithm.

We further analyze different RL algorithms under the same setting as ours, including the base model, the training dataset, and the hyperparameters. Among baseline methods, GRPO outperforms RLOO and REINFORCE++ on most benchmarks, particularly on MathVerse (51.4) and MathVision (25.9). However, our proposed CPGD method significantly outperforms all baselines, achieving the best performance. Both variants of CPGD (using either clip-filter-like weights or STD-based weights) yield over a +10% improvement in overall performance compared to the base model QwenVL2.5-7B. Notably, CPGD (STD weight) achieves a +21.8% gain on the in-domain benchmark MMK12, and further demonstrates strong generalization with +8.5% and +11.4% improvements on the out-ofdistribution benchmarks MathVista and MathVision, respectively. These results demonstrate that CPGD serves as a strong and robust alternative for RL in LM training.

### 5.3 Ablation study

Table 2: Results of ablation studies. Top performer is in bold and second-best is underlined.

Model MathVista MathVerse MathVision Olypamid WeMath MMK12 Overall CPGD (STD weight) 74.0 50.6 28.3 21.4 68.3 65.3 1.11 Ablation study on the components (using STD weight)

PG 67.8 42.0 22.5 8.0 58.6 65.9 0.89 PGD 64.2 41.1 20.8 7.5 58.3 67.3 0.86 CPG 72.7 52.3 27.6 20.8 70.7 66.2 1.11

Ablation study on the weighting factor

unprocessed rewards 69.1 40.2 21.8 3.5 59.7 67.2 0.85 equal weight 73.1 51.1 27.2 20.8 67.9 65.8 1.09 clip-filter-like weight 73.4 51.4 25.9 21.5 70.2 67.3 1.10

Ablation study on the reference constraint (using STD weight) w/ reference constraint 71.8 50.0 21.0 21.2 69.8 65.8 1.05

Component ablation. We conduct ablation on key components of our method by comparing variants: PG (basic policy gradient), PGD (PG + policy drift), CPG (PG + clip mechanism), and CPGD. Results show that the clip mechanism plays the most critical role, as seen by the performance drop from CPG/CPGD to PG/PGD across nearly all benchmarks. This aligns with our observation in Section 4.2 that clipping mitigates the response length collapse issue, which otherwise can impair test-time computation and reasoning capabilities. In contrast, adding policy drift has a relatively smaller effect. This is because CPGD’s objective lacks a potentially unstable importance-sampling ratio and already benefits from proximal updates via clipping, making policy drift mainly serve as a safeguard against excessive ratio deviation.

Weighting factor ablation. We further ablate different weighting strategies. We additionally include a baseline that uses raw unprocessed rewards as advantages, which results in significant performance degradation. This confirms that subtracting the group mean is crucial for stable and effective learning. This approach prevents over-penalization of all responses in the failure cases, which may otherwise trigger a squeezing effect [43], where the Softmax output head unintentionally reallocates probability mass to unexpected tokens, resulting in undesirable behaviors. Both clip-filter-like weight and STD weight outperform equal weighting, which we attribute to their ability to assign greater emphasis to samples with non-zero advantages. This targeted weighting encourages the model to focus more on informative training signals, thereby contributing to the improved performance.

Reference constraint ablation. Removing the reference constraint consistently improves performance, which echoes findings from recent studies [9, 10, 29], suggesting that such constraints may overly restrict policy improvement, and thus hinder overall optimization.

## 6 Discussion

- 6.1 Importance sampling

Importance sampling is a valuable technique for correcting the sampling distribution when the learned policy and the behavior policy differ significantly, thereby improving sample efficiency. While we omit the importance-sampling ratio to reduce variance, we do not suggest discarding it entirely. In fact, we use a single PPO epoch during training, a widely recommended default [6, 15]. In our view, importance sampling can be omitted with one epoch but should be reintroduced when using more:

ACPGDω (x,y) ← clip1+1−ϵϵ

sg(πθ(m−1)(yi|x,y<i)) πθ

old

(yi|x,y<i))

ACPGDω (x,y), m = 1,...,M, where πθ(m) denotes the updated policy after the m-th PPO epoch, and πθ(0) = πθ

old

, and thus the final updated policy is θnew = θ(M) after total M epochs. Here, the truncated importance sampling weight is applied to correct the off-policy distribution. Notably, we use θ(m) rather than the real-time θ to avoid instability caused by frequent updates within a single PPO epoch. This also ensures consistency with our proposed method. However, maintaining πθ(m) may incur additional cost, which we leave for future work to optimize.

- 6.2 Forward KL divergence vs. reverse KL divergence

Our policy drift adopts the forward KL divergence DKL(πθ

old

,πθ|x) instead of the reverse KL divergence DKL(πθ,πθ

old|x). While forward KL has been explored before [12], it is considered less effective than PPO-clip. In contrast, reverse KL is more commonly used in theory because it is closely related to mirror descent and has strong convergence guarantees [44, 45].

Although these two KL forms are different in how they are calculated, they often lead to similar results in practice [46]. This is because both are used to control policy updates. In fact, the difference between their gradients turns out to be small when the policy ratio is small, which is usually the case during training as shown in Figure 1:

∇θDKL(πθ,πθ

old|x) − ∇θDKL(πθ

old

,πθ|x)≈Ey∼π

θold(·|x)

- 1

- 2

πθ(y|x) πθ

old

(y|x) − 1

2

∇θ lnπθ(y|x) .

This approximation holds because xlnx ≈ x − 1 + 21(x − 1)2 when x is close to 1. Despite their similarity, we prefer forward KL for two main reasons: (1) It avoids importance sampling, which

reverse KL requires; and (2) It can be cleanly split into per-token terms (see Equation 4), which is not possible with reverse KL due to the importance weights.

- 6.3 Exploitation vs. exploration

Recent work [47] claims that the performance ceiling of a model is determined by its base model, casting a pessimistic view on the role of RL. While we do not fully agree or disagree, we offer a more nuanced view: the exploration capability is largely determined by the base model.

In RL training for LMs, the set of possible responses is constrained by what the base model can generate. RL helps it pick the best ones, boosting metrics like Maj@K. In other words, pretraining and SFT shape what the model can explore, while RL enhances the model’s exploitation ability.

This work mainly aims to improve RL stability, but advancing LM reasoning capability requires improving both RL and earlier stages like SFT to expand the model’s exploration range. Encouraging active exploration may be key to unlocking further improvements in model performance.

## 7 Conclusion

We identify a critical source of instability in existing RL methods for LMs: the use of asymmetric clipping on importance-sampling ratios, which can result in training collapse. To address this, we

propose CPGD, a principled alternative that avoids direct dependence on policy ratios while enforcing proximal updates through the clip mechanism and policy drift. CPGD further incorporates a stable KL estimator and a weighted advantage strategy to improve learning robustness. Theoretically grounded and empirically validated, CPGD demonstrates superior stability and performance across multimodal math benchmarks, offering a strong and stable RL solution for training LMs.

## References

- [1] DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025.
- [2] Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization, 2022.
- [3] Stanislas Polu and Ilya Sutskever. Generative language modeling for automated theorem proving, 2020.
- [4] Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, and Steven Chu Hong Hoi. Coderl: Mastering code generation through pretrained models and deep reinforcement learning. Advances in Neural Information Processing Systems, 35:21314–21328, 2022.
- [5] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36:8634–8652, 2023.
- [6] Jian Hu, Jason Klein Liu, and Wei Shen. Reinforce++: An efficient rlhf algorithm with robustness to both prompt and reward models, 2025.
- [7] Wouter Kool, Herke van Hoof, and Max Welling. Buy 4 REINFORCE samples, get a baseline for free! In Deep Reinforcement Learning Meets Structured Prediction, ICLR 2019 Workshop. OpenReview.net, 2019.
- [8] Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting reinforce-style optimization for learning from human feedback in llms. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, pages 12248–12267. Association for Computational Linguistics, 2024.
- [9] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, et al. Dapo: An open-source llm reinforcement learning system at scale, 2025.
- [10] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective, 2025.
- [11] Xiangxiang Chu, Hailang Huang, Xiao Zhang, Fei Wei, and Yong Wang. Gpg: A simple and strong reinforcement learning baseline for model reasoning, 2025.
- [12] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017.
- [13] Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.
- [14] John Schulman. Approximating kl divergence, 2020. URL http://joschu. net/blog/kl-approx. html, 2023.
- [15] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, Ping Luo, Yu Qiao, Qiaosheng Zhang, and Wenqi Shao. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning, 2025.
- [16] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts, 2024.
- [17] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset, 2024.
- [18] OpenAI. Introducing openai o1. https://openai.com/o1/, 2024. Accessed: 2024-10-02.
- [19] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2023.

- [20] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling, 2025.
- [21] Qwen Team. Qwq-32b: Embracing the power of reinforcement learning, March 2025.
- [22] Qwen Team. Qvq: To see the world with wisdom, December 2024.
- [23] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, et al. Kimi k1.5: Scaling reinforcement learning with llms, 2025.
- [24] Hugging Face. Open r1: A fully open reproduction of deepseek-r1, January 2025.
- [25] Jiayi Pan, Junjie Zhang, Xingyao Wang, Lifan Yuan, Hao Peng, and Alane Suhr. Tinyzero. https://github.com/Jiayi-Pan/TinyZero, 2025. Accessed: 2025-01-24.
- [26] YingZhe Peng, Gongrui Zhang, Xin Geng, and Xu Yang. Lmm-r1. https://github.com/TideDra/ lmm-r1, 2025. Accessed: 2025-02-13.
- [27] Liang Chen, Lei Li, Haozhe Zhao, Yifan Song, and Vinci. R1-v: Reinforcing super generalization ability in vision-language models with less than $3. https://github.com/Deep-Agent/R1-V, 2025. Accessed: 2025-02-02.
- [28] Huajie Tan, Yuheng Ji, Xiaoshuai Hao, Minglan Lin, Pengwei Wang, Zhongyuan Wang, and Shanghang Zhang. Reason-rft: Reinforcement fine-tuning for visual reasoning, 2025.
- [29] Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, and Heung-Yeung Shum Xiangyu Zhang. Openreasoner-zero: An open source approach to scaling reinforcement learning on the base model. https: //github.com/Open-Reasoner-Zero/Open-Reasoner-Zero, 2025.
- [30] Deheng Ye, Zhao Liu, Mingfei Sun, Bei Shi, Peilin Zhao, Hao Wu, Hongsheng Yu, Shaojie Yang, Xipeng Wu, Qingwei Guo, et al. Mastering complex control in moba games with deep reinforcement learning. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 6672–6679, 2020.
- [31] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond, 2023.
- [32] Jian Hu, Xibin Wu, Zilin Zhu, Xianyu, Weixun Wang, Dehao Zhang, and Yu Cao. Openrlhf: An easy-to-use, scalable and high-performance rlhf framework, 2024.
- [33] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework, 2024.
- [34] Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, Jiarui Yuan, Huayu Chen, Kaiyan Zhang, Xingtai Lv, Shuo Wang, Yuan Yao, Xu Han, Hao Peng, Yu Cheng, Zhiyuan Liu, Maosong Sun, Bowen Zhou, and Ning Ding. Process reinforcement through implicit rewards, 2025.
- [35] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, and Hongsheng Li. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems?, 2024.
- [36] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems, 2024.
- [37] Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, Zhuoma GongQue, Shanglin Lei, Zhe Wei, Miaoxuan Zhang, Runfeng Qiao, Yifan Zhang, Xiao Zong, Yida Xu, Muxi Diao, Zhimin Bao, Chen Li, and Honggang Zhang. We-math: Does your large multimodal model achieve human-like mathematical reasoning?, 2024.
- [38] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling, 2025.
- [39] Weiyun Wang, Zhe Chen, Wenhai Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Jinguo Zhu, Xizhou Zhu, Lewei Lu, Yu Qiao, and Jifeng Dai. Enhancing the reasoning ability of multimodal large language models via mixed preference optimization, 2024.

- [40] Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, Bo Zhang, and Wei Chen. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization, 2025.
- [41] Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, and Kai-Wei Chang. Openvlthinker: An early exploration to complex vision-language reasoning via iterative self-improvement, 2025.
- [42] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card, 2024.
- [43] Yi Ren and Danica J. Sutherland. Learning dynamics of LLM finetuning. In The Thirteenth International Conference on Learning Representations, 2025.
- [44] Matthieu Geist, Bruno Scherrer, and Olivier Pietquin. A theory of regularized markov decision processes. In International conference on machine learning, pages 2160–2169. PMLR, 2019.
- [45] Lior Shani, Yonathan Efroni, and Shie Mannor. Adaptive trust region policy optimization: Global convergence and faster rates for regularized mdps. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 5668–5675, 2020.
- [46] Chloe Ching-Yun Hsu, Celestine Mendler-Dünner, and Moritz Hardt. Revisiting design choices in proximal policy optimization, 2020.
- [47] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model?, 2025.

Appendix

- A Proofs

- A.1 Proof for Proposition 1

0(y|x) πθold(y|x) − 1| = ϵ. Consider updating θ0 using either (i) the PPO-clip objective, resulting in parameter θ1PPO, or (ii) the CPGD objective with α = 0, yielding parameter θ1CPG. Then, there exists a constant ηmax > 0 such that for any learning rate η ∈ (0,ηmax), the following inequality holds:

Proposition 2. Let θ0 be a parameter such that the importance-sampling ratio satisfies | πθ

(y|x) πθ

(y|x) πθ

πθPPO

πθCPG

(y|x) πθ

πθ

0

(y|x) − 1 >

(y|x) − 1 >

(y|x) − 1 = ϵ.

1

1

old

old

old

After one update step, both PPO and CPG increase the importance-sampling ratio deviation from the old policy, but PPO does so more aggressively than CPG.

(y|x)

πθCPG 1

πθold(y|x), where θ1CPG = θ0 + η∇θLˆCPG(x,y;θ0) is the single gradient ascent step on the empirical CPGD objective (Equation 1) without the policy drift term. The gradient of the objective takes the form:

Proof. Consider f(η) =

∇θLˆCPG(x,y;θ) = ACPGD(x,y)∇θ lnπθ(y|x). Thus, for the case where πθ0(y|x)

πθold(y|x) = 1 + ϵ and ACPGD(x,y) > 0, the directional derivative of f at η = 0 satisfies:

∇θπθ

(y|x) πθ

,∇θLˆCPG(x;θ0)⟩ > 0.

f′(0) = ⟨

0

(y|x)

old

Hence, there exists a constant η1 > 0 such that for any η ∈ (0,η1), we have f(η) > f(0). Similarly, when πθ0(y|x)

πθold(y|x) = 1 − ϵ and ACPGD(x,y) < 0, there exists η2 > 0 such that f(η) < f(0) for any η ∈ (0,η2). Therefore, for any 0 < η < min(η1,η2), the following holds:

(y|x) πθ

πθCPG

(y|x) πθ

πθ

(y|x) − 1| = ϵ. (6)

0

|

(y|x) − 1| > |

1

old

old

(y|x)

###### (y|x)

πθPPO 1

πθCPG 1

πθold(y|x), where θ1PPO = θ0 + η∇θLˆPPO(x,y;θ0) and ∇θLˆPPO(x,y;θ) = ACPGD(x,y)∇θπθ(y|x)

Next, define g(η) =

πθold(y|x) −

.

(y|x)

πθ

old

For the case where πθ0(y|x)

πθold(y|x) = 1 + ϵ and ACPGD(x,y) > 0, we have: g′(0) = ∇θπθ

(y|x) πθ

πθ(y|x) πθ

,ACPGD(x,y) · (1 −

0

) · ∇θ lnπθ(y|x) < 0.

(y|x)

(y|x)

old

old

Hence, there exists a constant η3 > 0 such that g(η) < g(0) for any η ∈ (0,η3). Similarly, for the case where πθ0(y|x)

πθold(y|x) = 1 − ϵ and ACPGD(x,y) < 0, there exists a constant η4 > 0 such that g(η) > g(0) for any η ∈ (0,η4). Therefore, for any 0 < η < min(η3,η4), we have

(y|x) πθ

(y|x) πθ

πθPPO

πθCPG

(y|x) − 1|. (7) Therefore, by letting ηmax = min(η1,η2,η3,η4), the proof is complete.

|

(y|x) − 1| > |

1

1

old

old

| |
|---|

- A.2 Proof for Theorem 1

Theorem 2. Let {πθ

k}∞k=0 denote the sequence of policies generated by the CPGD update rule (Equation 1). Then, the sequence πθ

converges.

k

Proof. First, denote LCPGD(θ;θk) = Ex∼D g(θ;θk,x) , and rewrite g as

πθ(y|x) πθ

g(θ;θk,x) = Ey∼π

(y|x) − αDKL(πθ

,πθ|x)

θk(·|x) Ro(x,y)ln

k

k

πθ(y|x) πθ

πθ(y|x) πθ

θk(·|x) ReLU ln

−Ey∼π

,ln(1 − ϵ),ln(1 + ϵ) Ro(x,y) . Here, we omit the baseline Ey∼π

(y|x) − clip ln

(y|x)

k

k

θk(·|x)[Ro(x,y)]. Then, denoting θk+1 the point such that LCPGD(θk+1;θk) ≥ LCPGD(θk;θk), we obtain

Ey∼π

θk+1(·|x) Ro(x,y) − Ey∼π

θk(·|x) Ro(x,y)

(y|x) πθ

πθ

=Ey∼π

k+1

(y|x) − 1 Ro(x,y) ≥Ey∼π

θk(·|x)

k

(y|x) πθ

πθ

k+1

(y|x) · Ro(x,y)

θk(·|x) ln

k

=g(θk+1;θk,x) − g(θk;θk,x) + αDKL(πθ

k+1|x)

,πθ

k

(y|x) πθ

(y|x) πθ

πθ

πθ

θk(·|x) ReLU ln

+ Ey∼π

k+1

k+1

(y|x) − clip ln

,ln(1 − ϵ),ln(1 + ϵ) Ro(x,y) .

(y|x)

k

k

Denoting the overall expected return by η(πθ) = Ex∼D,y∼π

θ(·|x) Ro(x,y) , we integrate over x to conclude

Pinsker inequality

α 2 ∥πθ

k∥21. Because η(πθ

) ≥ αEx∼D DKL(πθ

) − η(πθ

k+1|x)

≥

k+1 − πθ

η(πθ

,πθ

k+1

k

k

) = η∗. Thus, taking the limit of k on both sides of the following equation,

) is bounded, there exists a η∗ such that limk→∞ η(πθ

k

k

α 2 ∥πθ

k∥21 ≤ η(πθ

0 ≤

k+1 − πθ

) − η(πθ

), we can obtain limk→∞ ∥πθ

k+1

k

k∥1 = 0. Since the parameter space Θ is compact, the sequence {πθ

k+1 − πθ

k} converges to some limit point πθ

.

∗

| |
|---|

## B Prompt setting

Table 3: Prompt setting.

SYSTEM: Solve the question. The user asks a question, and you solves it. You first thinks about the reasoning process in the mind and then provides the user with the answer. The answer is in latex format and wrapped in $...$. The final answer must be wrapped using the \boxed{} command. The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively, i.e., <think>Since 1+1 = 2, so the answer is 2. </think><answer>The answer is $\boxed{2}$ </answer>, which means the final answer assistant’s output should start with <answer> and end with </answer>.

USER: <image>{{question}}

We follow the prompt format from DeepSeek-R1, where reasoning steps and final answers are explicitly marked using <think> and <answer> tags, respectively. The full prompt template is provided in Table 3.

## C Limitations

While this work introduces a stable and effective RL method for LMs training, it has several limitations: (1) For the weighted advantage component, we conducted only preliminary experiments and did not thoroughly explore the impact of different weighting factors. Our results suggest that using non-uniform weights yields better performance than trivial equal weighting, but further investigation is needed. (2) Our study focuses on on-policy training; we leave off-policy settings—where importance sampling is typically required—for future work. Ensuring training stability in the presence of importance sampling remains an open question. (3) All experiments were conducted on standard academic-scale models (7B parameters). We did not evaluate our method on larger models (e.g., 100B+), which would require significant computational resources.

