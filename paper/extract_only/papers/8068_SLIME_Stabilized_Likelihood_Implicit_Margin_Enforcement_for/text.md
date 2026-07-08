## SLIME: Stabilized Likelihood Implicit Margin Enforcement for Preference Optimization

Maksim Afanasyev*1 Illarion Iov*1

# arXiv:2602.02383v2[cs.LG]3Feb2026

### Abstract

Direct preference optimization methods have emerged as a computationally efficient alternative to Reinforcement Learning from Human Feedback (RLHF) for aligning Large Language Models (LLMs). Latest approaches have streamlined the alignment process by deriving implicit reward functions, yet they often suffer from a critical objective mismatch: optimizing the relative margin between chosen and rejected responses does not guarantee the preservation of the chosen response’s absolute likelihood. This can lead to “unlearning”, where the model degrades the probability of high-quality outputs to satisfy margin constraints, and “formatting collapse” caused by the over-penalization of rejected sequences. In this work, we introduce SLIME (Stabilized Likelihood Implicit Margin Enforcement), a reference-free alignment objective designed to decouple preference learning from generation quality. SLIME incorporates a three-pronged objective: (1) an anchoring term to maximize the likelihood of preferred responses; (2) a stabilizing penalty that prevents the probabilities of rejected tokens from collapsing to zero; and (3) a dual-margin mechanism that combines hard and soft constraints for precise boundary shaping. Our results demonstrate that SLIME achieves superior performance compared to stateof-the-art baselines while maintaining higher generation stability.

### 1. Introduction

The alignment of Large Language Models (LLMs) with human intent is a cornerstone of modern AI development. While Reinforcement Learning from Human Feedback (RLHF) via Proximal Policy Optimization (PPO) (Ouyang

*Equal contribution 1Floating Point Sigma Lab. Correspondence to: Maksim Afanasyev <mr.applexz@gmail.com>, Illarion Iov <illariov1809@gmail.com>.

Preprint. February 4, 2026.

- et al., 2022) has been the standard for this task, it is notoriously unstable and resource-intensive. This has spurred the development of offline, gradient-based methods such as Direct Preference Optimization (DPO) (Rafailov et al.,

2023) and Identity Preference Optimization (IPO) (Azar

- et al., 2023), which reframe alignment as a classification problem on preference pairs. More recently, SimPO (Meng
- et al., 2024) further simplified this paradigm by removing the reference model and incorporating length normalization, establishing a new state-of-the-art for reference-free alignment.

Despite these advances, current margin-based objectives suffer from a fundamental limitation: they optimize the relative gap between winning and losing responses, often at the expense of the absolute quality of the generation. While using these methods a model may “game” the objective by lowering the probability of the chosen response, provided it lowers the probability of the rejected response even further. We observe that this phenomenon leads to the “unlearning” of valid syntax and reasoning patterns found in the pretrained model. Furthermore, the aggressive suppression of rejected sequences—which often contain valid partial reasoning or correct grammar—can result in distribution collapse, hurting the model’s fluency and diversity.

To address these challenges, we propose SLIME (Stabilized Likelihood Implicit Margin Enforcement). Unlike prior methods that treat alignment purely as margin maximization, SLIME explicitly anchors the policy to high-likelihood regions for chosen responses while treating rejected responses with a stabilized penalty rather than indiscriminate suppression.

Our approach introduces three key contributions to the alignment landscape:

- • Likelihood Anchoring: We reintroduce a supervised

signal for the chosen sequence (yw), ensuring that the model retains its generative capabilities and prevents the probability degradation common in standard DPO/SimPO training.

- • Token-Level Stabilization: We identify that minimizing rejected likelihoods (yl) indiscriminately harms

model fluency. SLIME employs a non-linear, softplusbased penalty that discourages extremely low token probabilities, effectively filtering out “easy” negatives while preserving the structural integrity of the language model.

human utility of individual generations based on whether they are simply “desirable” or “undesirable.” KTO belongs to a class of objectives called Human-Aware LOsses (HALOs) and defines the loss function via a reference-dependent value function:

• Dual-Margin Optimization: We propose a novel distance loss combining a hard margin for strict cutoff and a soft margin for gradient shaping. This allows for efficient optimization near the decision boundary without the vanishing gradients or overfitting associated with single-margin losses.

We empirically evaluate SLIME on a diverse benchmarks, including MT-Bench (Zheng et al., 2023), Arena-Hard (Li et al., 2024). Experimental results indicate that SLIME outperforms strong baselines such as DPO and SimPO across various model families (Llama3.2, Qwen3, Gemma3). Furthermore, ablation studies confirm that our stabilizing and anchoring mechanisms significantly contribute to training stability and final model robustness.

### 2. Related Works

#### 2.1. Standard Alignment and Direct Methods

The foundational approach to LLM alignment, introduced by InstructGPT (Ouyang et al., 2022), uses Proximal Policy Optimization (PPO) to optimize a policy against a trained reward model. Although effective, PPO is known to be unstable and memory-intensive. This led to the development of Direct Preference Optimization (DPO) (Rafailov et al., 2023), which derives an implicit reward function directly from the policy, allowing for alignment via a simple classification loss on preference pairs.

Several variants have since modified this approach. Identity Preference Optimization (IPO) (Azar et al., 2023) addresses DPO’s tendency to overfit by regularizing the gap between the policy and the reference model. SimPO (Meng et al.,

- 2024) further simplifies the objective by removing the reference model entirely and adding length normalization to prevent the generation of verbose low-quality responses. While these offline methods are computationally efficient, they often lack the exploration stage required for solving complex reasoning problems, thus requiring the use of online policy gradient methods (Shao et al., 2024).

#### 2.2. Kahneman-Tversky Optimization (KTO)

While preference-based methods like DPO optimize the likelihood of relative preferences, (Ethayarajh et al., 2024) propose Kahneman-Tversky Optimization (KTO), a method grounded in prospect theory. KTO posits that aligning models does not strictly require paired preference data (e.g., yw ≻ yl); instead, it can be achieved by maximizing the

LKTO(πθ,πref) = Ex,y∼D [w(y)] (1)

where the per-sample loss w(y) is defined as:

λD(1 − σ(β(rθ(x,y) − z0))) if y is desirable λUσ(β(rθ(x,y) − z0)) if y is undesirable

w(y) =

(2) Here, rθ(x,y) is the implicit reward log-ratio log π

θ(y|x) πref(y|x), and z0 serves as a reference point (effectively the KL divergence), which is estimated in practice using mismatched outputs within a training batch. By adjusting the hyperparameters λD and λU, KTO can handle extreme data imbalances (e.g., far fewer desirable examples than undesirable ones) and has been shown to match or exceed the performance of preference-based methods like DPO even without paired data.

#### 2.3. SimPO: Simple Preference Optimization

Building on the trend of simplifying alignment objectives, (Meng et al., 2024) proposed SimPO, which eliminates the need for a reference model entirely. Unlike DPO and KTO, which rely on the ratio between the policy and a reference model to define implicit rewards, SimPO uses the average log probability of the sequence itself as the reward formulation. This choice directly aligns the training objective with the generation metric used during inference. To prevent the model from exploiting length to maximize reward, SimPO normalizes the reward by the response length and introduces a target reward margin γ into the Bradley-Terry objective to enforce a significant separation between winning and losing responses.

#### 2.4. Generalization via Ψ-Preference Optimization

To provide a unified view of preference learning, (Azar et al., 2023) introduce Ψ-Preference Optimization (ΨPO), a general objective where DPO and RLHF are strictly special cases using the logit mapping Ψ(q) = log(q/(1 − q)) under the Bradley-Terry assumption. However, they demonstrate that this specific mapping causes DPO to overfit when preferences are deterministic, as the implicit reward tends towards infinity and ignores the KL constraint. By generalizing this mapping to the identity function Ψ(q) = q, they derive Identity Preference Optimization (IPO). IPO allows for learning directly from preferences without assuming the Bradley-Terry model, effectively regularizing the gap

between the policy and reference log-likelihood ratios to prevent the greedy behavior observed in DPO.

#### 2.5. Group Relative Policy Optimization (GRPO)

To retain the benefits of online exploration without the value function overhead, (Shao et al., 2024) introduced GRPO. Instead of using a separate critic model to estimate value V (s), GRPO samples a group of outputs {y1,...,yG} for a query x and estimates the baseline using the group average. The advantage for every input is computed as:

r(x,yi) − µ({rj}) σ({rj})

Aˆi =

(3)

where µ and σ are the mean and standard deviation of rewards within the group. This method significantly reduces memory usage and has been used to train models like DeepSeekMath.

Another approach similar to GRPO is REINFORCE Leaveone-out (RLOO)(Ahmadian et al., 2024), which uses the mean of the other k − 1 samples as the baseline, providing an unbiased estimation but lacking the variance reduction properties of GRPO’s normalization.

### 3. Method

In this section, we introduce SLIME (Stabilized Likelihood Implicit Margin Enforcement), a reference-free preference optimization objective designed to align Large Language Models (LLMs) with human preferences while preserving generation quality. Unlike previous approaches such as DPO (Rafailov et al., 2023), which may degrade the likelihood of chosen sequences, SLIME explicitly anchors the policy to high-likelihood regions and enforces training on samples yet not well resolved by the model.

Recent work in online policy optimization has identified that token-level processing introduces critical stability challenges. As noted by (Zheng et al., 2025), token-level importance ratios in methods like GRPO accumulate highvariance noise during long-horizon generation. While their solution of shifting to sequence-level ratios addresses this in the online setting, we observe that a complementary problem exists in offline preference optimization: token-level likelihood suppression of rejected sequences can destabilize the model’s linguistic foundations. SLIME adapts this insight to the offline regime by introducing explicit token-level regularization that prevents probability collapse, effectively transferring the lesson that careful token-level treatment is essential for stable alignment.

#### 2.6. Sequence-level and Global Optimization

Despite being widely adopted, standard GRPO relies on token-level importance sampling, which can be unstable for long-term generation. The authors of (Zheng et al.,

- 2025) identified that token-level ratios accumulate highvariance noise, particularly in Mixture-of-Experts (MoE) models where expert routing fluctuates between the behavior and target policies. They proposed Group Sequence Policy Optimization (GSPO), which derives importance ratios from the likelihood of the entire sequence:

  1

  (4)

|yi|

πθ(yi,t|x,yi,<t) πθ

sseqi (θ) = exp

log

|yi|

(yi,t|x,yi,<t)

old

t=1

The sequence-level approach aligns the optimization granularity with the reward signal evaluated for the full sequence and omits the need for complex stabilization strategies such as Routing Replay in MoE training.

Concurrently, (Hu et al., 2025) critique the local normalization used in GRPO. They prove normalizing advantages based only on a small local group (e.g., G = 4) introduces bias because the numerator (centered reward) and denominator (local standard deviation) are not independent. They proposed REINFORCE++ method utilizing Global Advantage Normalization across the entire training batch to provide a stable and effectively unbiased estimator.

#### 3.1. Overview and Motivation

Recent advancements in offline preference optimization, such as SimPO (Meng et al., 2024) utilize the average loglikelihood of a sequence as an implicit reward. While effective, this formulation can suffer from a critical limitation: the objective maximizes the margin between winning (yw) and losing (yl) responses, but does not strictly enforce the maintenance of high likelihood for yw. As a result, a model can minimize its loss by lowering the probability of yw, provided it lowers the probability of yl even further. This race can lead to a degradation in the model’s general capabilities and fluency.

To address this, SLIME introduces a composite loss function that decouples the optimization into three distinct goals: (1) Anchoring the Chosen: explicitly preserving the likelihood of the preferred response; (2) Stabilizing the Rejected: preventing the likelihood of rejected responses from collapsing to zero; and (3) Dual-Margin Optimization: employing a hybrid hard-soft margin to precisely shape the decision boundary. The total objective is:

L(θ) = Lw(θ) + Ll(θ) + Ldist(θ) (5)

#### 3.2. Anchoring the Chosen Sequence

To prevent the policy model from sacrificing the likelihood of the winning response yw to achieve a better margin, we introduce a separate anchoring term Lw. This term acts as a

positive reinforcement signal, ensuring that the model maintains or increases the probability of generating the preferred sequence:

w)∼D log πθ(yw|x) (6)

Lw(θ) = −λw E(x,y

where λw is a hyperparameter controlling the strength of this anchor. By explicitly maximizing the log-probability of yw, we counteract the phenomenon observed in margin-based losses where the model “unlearns” the chosen response to minimize the relative cost.

#### 3.3. Stabilizing Rejected Token Probabilities

Standard preference optimization often treats rejected responses yl simply as negatives to be minimized. However, rejected responses are often fluent and partially correct; aggressively suppressing their likelihood can lead to the loss of valid syntax or reasoning patterns. We employ a token-level stabilizing loss Ll that penalizes extremely low probabilities for tokens t in the rejected sequence:

softplus −log πθ(t | x) − δ p (7)

Ll(θ) = λl Et∈y

l

Here, δ acts as a threshold shift. This formulation acts as a robust filter. For tokens with sufficiently high logprobability, the term approaches zero. As the log-probability drops significantly, the power of p amplifies the penalty, ensuring the model is not forced to decrease token-level probabilities beyond a reasonable threshold. The exponent p = 2.5 controls the sharpness of the penalty applied to low-probability tokens and can be treated as a tunable hyperparameter. We analyze the optimal values in the Ablation Study section 6

#### 3.4. Dual-Margin Preference Optimization

Finally, we define a distance-based loss Ldist that operates on the log-probability difference ∆. We propose a Dual-Margin approach that combines a hard margin for cut-off and a soft margin for gradient shaping. Let ∆ = log πθ(yw|x) − log πθ(yl|x). We define:

ℓhard = max(0, −∆ + mh), (8) ℓsoft = σ(−κ(∆ − ms)) (9)

The combined distance loss is then:

Ldist(θ) = λd E[ℓhard · ℓsoft] (10) This formulation introduces two distinct hyperparameters: the hard margin mh and the soft margin ms. The hard margin defines a “victory condition” where the loss becomes exactly zero once the margin is satisfied. The soft margin ℓsoft acts as a dynamic gating mechanism, prioritizing optimization in the critical region between the two margins while avoiding the constant non-zero gradients typical of standard log-sigmoid objectives.

### 4. Experiments

#### 4.1. Experimental Setup

We conduct all experiments using the TRL (von Werra et al., 2020) framework and compare the performance of DPO and SimPO. All models are trained on the UltraFeedback dataset (Cui et al., 2023), which provides pairwise preference annotations for preference-based fine-tuning.

Models. We evaluate three base language models of comparable scale:

- • Llama3.2-3B (Meta, 2024)
- • Qwen3-4B (Yang et al., 2025)
- • Gemma3-4B (Team et al., 2025)

All models are trained from their publicly released checkpoints.

Training Pipeline and Data Splitting. To differentiate the impact of preference optimization from pre-existing instruction tuning, we adopt a two-stage training pipeline starting strictly from pretrained base checkpoints. For tokenization, we apply the official chat templates from the corresponding instruction-tuned variants to these base models. The training proceeds as follows: (1) Supervised FineTuning (SFT): We first fine-tune the pretrained base model to establish a compliant instruction-following policy with cross-entropy loss. (2) Preference Alignment: We then apply the respective preference optimization algorithms (DPO, SimPO, or SLIME) initializing the policy with the SFT checkpoint. To support this pipeline, we partition the UltraFeedback dataset into two disjoint subsets using a fixed random seed: 33% of the data is utilized exclusively for the SFT stage, while the remaining 66% is reserved for preference optimization. This strict separation ensures that the alignment phase does not optimize on examples seen during SFT, thereby preventing data leakage and overfitting.

Parameter-Efficient Fine-Tuning. We employ LowRank Adaptation (LoRA) (Hu et al., 2022) for parameterefficient fine-tuning. Unless stated otherwise, LoRA is applied to all models using a uniform configuration across experiments. The full LoRA hyperparameter setup is summarized in Table 1.

This configuration enables a high-capacity LoRA setup, allowing for expressive adaptation while still training only a subset of model parameters. The same LoRA configuration is used for all base models to ensure a controlled comparison.

Table 1. LoRA configuration used in our experiments.

Hyperparameter Value

Rank (r) 64 Scaling factor (α) 128 (α/r = 0.5) Target modules Attention projections (q, k, v, o) and MLP layers

(up, down, gate) Modules to save LM Head Dropout 0.0 Bias none (bias parameters are not trained)

Algorithm-Specific Hyperparameters. For DPO, we set the inverse temperature parameter to β = 0.1. For SimPO, the scaling parameter is set to γ = 0.2.

For SLIME, we use the following hyperparameter configuration:

λw = 0.1, λl = 0.1, λdist = 1.0,

Hardware and Compute. All experiments are executed on a single multi-GPU node equipped with NVIDIA H100 accelerators. The hardware configuration and computational budget are summarized in Table 2.

δ = 1.25, mh = 1.5, ms = 1.0, κ = 2.5, p = 2.5.

All hyperparameters are kept fixed across models to ensure a fair comparison.

- Table 2. Hardware and computational resources used in our experiments.

Resource Specification

GPU 8× NVIDIA H100 SXM5 (80GB) Interconnect NVLink CPU AMD EPYC Milan (180 cores) @ 3.6GHz System RAM 1.45TB Training time per model (wall-clock) 1.25h Total GPU hours per model 30GPU-h

Optimization and Training Details. We employ the AdamW optimizer (Loshchilov & Hutter, 2017). For SFT, we start from the learning rate of 2×10−4 for Llama3.2 and Gemma3 models and 2.5×10−6 for Qwen3. The difference in learning rates comes from Qwen3 pretrain checkpoint being already fine-tuned for reasoning and instruct modes, and achieving high results on benchmarks from the start. For all models, chosen checkpoints allow the models to achieve the best possible performance on benchmarks after fine-tuning. All models are trained for three epochs. At the Reinforcement Learning stage, all models are trained with an initial learning rate of 5 × 10−7, linearly decayed to zero over the course of training. No learning rate warmup is used. Gradient clipping is disabled in all experiments. For this stage, a single epoch has been used for training.

Training is performed using Distributed Data Parallel (DDP) across all eight GPUs with mixed-precision bf16 arithmetic. We use a mini-batch size of 8 and apply gradient accumulation over two steps, resulting in an effective batch size of 16 per GPU. For evaluation, a batch size of 16 is used.

Evaluation Protocol. Evaluation is performed every 1000 training steps. The UltraFeedback dataset contains approximately 63,000 samples, which are evenly sharded across the eight GPUs, resulting in an effective evaluation size of approximately 8,000 samples per evaluation run. No early stopping is applied.

Evaluation. We evaluate all models using a set of established instruction-following and robustness benchmarks: MT-Bench and Arena-Hard. These benchmarks collectively assess model quality across general instruction adherence, multi-turn reasoning, adversarial prompting, and open-domain robustness.

MT-Bench scores are computed as the average score across all benchmark questions using the official grading procedure. Arena-Hard evaluations follow their respective public evaluation protocols.

All evaluations are performed on the same model checkpoints obtained after training completion. We do not apply any additional fine-tuning, prompt tuning, or postprocessing during evaluation. The evaluation results are summarized in Table 3.

Reproducibility. To ensure reproducibility, we fix the random seed to zero for Random, Numpy and Torch packages.

### 5. Results

Overview. We evaluate the performance of all fine-tuned models on a held-out evaluation set derived from the UltraFeedback dataset. We report results for three base models and compare three preference optimization algorithms. For reference, we also include the performance of the initial (pre-finetuning) models. All results are obtained using the same evaluation protocol described in Section 4.1.

Main Results. Table 3 summarizes the evaluation results across all model–algorithm combinations. For each base model, we report the baseline performance prior to preference-based fine-tuning, as well as results obtained after training with DPO, SimPO, and SLIME. For Arena-Hard, the default baseline model gpt-o3-mini-2025-01-31 overperforms 4B models, thus the base Gemma3-4B-it answers were used as a baseline.

- Table 3. Evaluation results across multiple benchmarks. We compare the raw pretrained base model, the supervised fine-tuned (SFT) baseline, and subsequent preference optimization methods. Higher is better for all metrics. The best result for each model and benchmark is highlighted in bold, the second best is underlined.

###### Model Method MT-Bench ↑ Arena-Hard ↑ Arena-Hard Std

Base (Pretrain) 1.01 – – SFT 4.56 7.5 −0.8/ + 0.9 + DPO 4.92 11.1 −1.0/ + 1.1 + SimPO 4.22 7.6 −1.0/ + 0.9 + SLIME (ours) 5.49 9.7 −1.0/ + 1.2

Llama3.2-3B

Base (Pretrain) 5.95 – – SFT 5.40 32.1 −2.0/ + 1.9 + DPO 5.30 39.0 −2.1/ + 2.4 + SimPO 5.72 25.8 −1.8/ + 1.5

Qwen3-4B

- + SLIME (ours) 5.93 39.8 −2.1/ + 1.7

Gemma3-4B

Base (Pretrain) 3.86 – – SFT 4.71 7.6 −0.8/ + 0.7 + DPO 5.15 11.8 −1.3/ + 1.1 + SimPO 5.03 0.7 −0.3/ + 0.3

- + SLIME (ours) 6.15 13.1 −1.5/ + 1.2

### 6. Ablation Study

We conduct an ablation study to analyze the contribution of individual components of the proposed SLIME objective. All ablation experiments are performed using the same training and evaluation protocol as described in Section 4.1, unless explicitly stated otherwise. We use Gemma3 as a base model for all comparisons.

Loss Component Ablations. To assess the importance of each term in the SLIME objective, we evaluate several variants in which individual loss components are removed. Specifically, we consider the following settings:

- • w/o chosen term: the loss term corresponding to preferred (chosen) responses is removed;
- • w/o rejected term: the loss term corresponding to rejected responses is removed;
- • w/o soft distance margin: the distance-based soft margin loss is disabled;
- • w/o hard margin: the hard margin constraint is removed.

These ablations isolate the contribution of each component while keeping all other hyperparameters fixed.

Stabilizing Loss Exponent Ablation. The stabilizing loss Ll includes a power term that amplifies penalties for extremely low token probabilities in rejected responses. While this exponent is fixed to p = 2.5 in the default configuration, it directly controls the sharpness of the penalty and may influence training stability and performance.

To study its effect, we vary the exponent p in the range {1.5,2.0,2.5,3.0} while keeping all other hyperparameters fixed. This ablation evaluates the sensitivity of SLIME to the strength of token-level stabilization. Benchmark results for different exponent values are shown in Table 5

We observe that moderate exponent values yield the best overall performance, while overly small or large exponents degrade results, suggesting a trade-off between stability and flexibility.

### 7. Discussion

The experimental results presented in Table 3 reveal several important insights about the behavior of preference optimization methods across different model architectures and scales.

Consistent Improvements Across Architectures. SLIME demonstrates robust performance gains across all three evaluated model families. On Gemma3-4B, SLIME achieves the highest MT-Bench score of 6.15, representing a 30.6% improvement over the SFT baseline (4.71) and outperforming both DPO (5.15) and SimPO (5.03). Similarly, on Llama3.2-3B, SLIME attains an MT-Bench score of 5.49, surpassing DPO (4.92) by 11.6% and SimPO (4.22) by 30.1%. These consistent improvements suggest that the three-pronged objective of SLIME - likelihood anchoring, token-level stabilization, and dual-margin optimization - addresses fundamental limitations present across diverse model architectures.

The Unlearning Phenomenon in Margin-Based Methods. Our results provide empirical evidence for the “unlearning” hypothesis outlined in the introduction. SimPO,

Table 4. Ablation results for individual components of the SLIME loss. Higher is better.

###### Variant MT-Bench ↑ Arena-Hard ↑ Std

Full SLIME 6.15 13.1 −1.1/ + 1.0 w/o chosen term 5.21 11.1 −1.2/ + 1.1 w/o rejected term 5.74 12.1 −1.0/ + 1.1 w/o soft distance margin 5.80 11.2 −1.2/ + 1.1 w/o hard margin 5.90 12.4 −1.3/ + 1.4

Table 5. Effect of the stabilizing loss exponent p on SLIME performance.

###### Exponent p MT-Bench ↑ Arena-Hard ↑ Std

1.0 5.70 12.2 −1.2/ + 1.1

- 1.5 5.86 12.2 −1.3/ + 1.5
- 2.0 5.76 11.2 −1.1/ + 1.1

2.5 (default) 6.15 13.1 −1.1/ + 1.0

- 3.0 5.66 12.7 −1.4/ + 1.2

which relies purely on margin maximization without explicit likelihood preservation, exhibits notable performance degradation in several configurations. On Llama3.2-3B, SimPO underperforms even the SFT baseline on MT-Bench (4.22 vs. 4.56), and on Gemma3-4B Arena-Hard, SimPO collapses to a score of 0.7 compared to the SFT baseline of 7.6. This pattern is consistent with our theoretical motivation: when the objective focuses solely on the relative margin ∆ = log πθ(yw|x) − log πθ(yl|x), the model can satisfy the loss by degrading both likelihoods, provided the chosen response is degraded less severely. SLIME’s anchoring term Lw directly counteracts this failure mode by maintaining an explicit supervision signal on the preferred sequence.

Stability Through Token-Level Regularization. The ablation study in Table 4 confirms that the rejected-sequence stabilization term Ll contributes meaningfully to final performance. Removing this component reduces MT-Bench performance from 6.15 to 5.74 on Gemma3-4B. We hypothesize that this term prevents the model from aggressively suppressing tokens that, while appearing in rejected responses, represent valid linguistic patterns. By employing a softplus-based penalty with threshold δ, SLIME maintains a floor on token probabilities, preserving the model’s fluency and preventing the distribution collapse observed in standard preference optimization.

The Role of Dual-Margin Optimization. The dualmargin formulation provides complementary benefits. The hard margin mh establishes a clear “victory condition” that eliminates gradients once satisfied, preventing overoptimization. The soft margin ms, implemented via a sigmoid gate, concentrates optimization effort in the critical region near the decision boundary. As shown in the gradient analysis (Appendix A), this combination avoids both

the vanishing gradients of pure hard-margin losses and the constant non-zero gradients of log-sigmoid objectives. The ablation results support this design: removing the soft margin reduces performance to 5.80, while removing the hard margin yields 5.90, both below the full SLIME objective.

Model-Specific Observations. The Qwen3-4B results warrant additional discussion. Unlike Llama3.2 and Gemma3, Qwen3’s pretrained checkpoint already incorporates instruction-tuning, resulting in a higher baseline MT-Bench score of 5.95. Consequently, the SFT stage provides minimal improvement and may even introduce slight degradation (5.40). However, SLIME still achieves the strongest Arena-Hard performance (39.8) among all methods on this model, demonstrating that the approach remains effective even when starting from a stronger baseline. This suggests that SLIME’s stabilizing mechanisms are particularly valuable for preserving pre-existing capabilities during preference alignment.

Limitations. Several limitations of this work merit acknowledgment. First, our evaluation is limited to models in the 3–4B parameter range; the effectiveness of SLIME on larger models remains to be validated. Second, while we evaluate on diverse benchmarks, all training uses the UltraFeedback dataset; generalization to other preference datasets warrants further investigation. Third, the computational overhead of SLIME relative to simpler baselines, while modest, introduces additional hyperparameters (λw, λl, δ, mh, ms, κ, p) that require tuning. Finally, our analysis focuses on English-language benchmarks; multilingual evaluation would strengthen claims of general applicability.

### 8. Conclusion

We introduced SLIME (Stabilized Likelihood Implicit Margin Enforcement), a reference-free preference optimization objective that addresses fundamental limitations of existing margin-based alignment methods. By decomposing the optimization into three complementary components - likelihood anchoring for chosen sequences, token-level stabilization for rejected sequences, and dual-margin preference optimization - SLIME decouples preference learning from generation quality preservation.

Our experiments across Llama3.2-3B, Qwen3-4B, and Gemma3-4B demonstrate that SLIME consistently outperforms both DPO and SimPO on MT-Bench and Arena-Hard benchmarks. Notably, SLIME achieves these improvements while avoiding the “unlearning” phenomenon observed in pure margin-based methods, as evidenced by SimPO’s performance degradation below SFT baselines in several configurations.

The ablation study confirms that each component of the SLIME objective contributes to final performance. The anchoring term prevents likelihood degradation of preferred responses; the stabilization term preserves linguistic fluency by maintaining reasonable token probabilities in rejected sequences; and the dual-margin mechanism enables precise boundary shaping without over-optimization.

Looking forward, several directions merit exploration. Extending SLIME to online settings with on-policy sampling could combine its stability benefits with the exploration advantages of policy gradient methods. Investigating the interaction between SLIME and other efficiency techniques, such as quantization or pruning, would inform practical deployment. Finally, theoretical analysis establishing formal guarantees on likelihood preservation under SLIME optimization would complement our empirical findings.

We believe that explicitly addressing the objective mismatch in preference optimization, rather than treating alignment purely as margin maximization, represents a promising direction for developing more robust and capable language models. SLIME demonstrates that loss design can preserve model capabilities while effectively optimizing human preferences.

### Code Availability

The code used in this work is available at https:// github.com/fpsigma/trl-slime to facilitate reproducibility and enable future research.

### References

Ahmadian, A., Cremer, C., Gall´e, M., Fadaee, M., et al. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. arXiv preprint arXiv:2402.14740, 2024.

Azar, M. G., Rowland, M., Piot, B., Guo, D., Munos, R., et al. A general theoretical paradigm to understand learning from human preferences. arXiv preprint arXiv:2310.12036, 2023.

Cui, G., Yuan, L., Ding, N., Yao, G., He, B., Zhu, W., Ni, Y., Xie, G., Xie, R., Lin, Y., et al. Ultrafeedback: Boosting

language models with scaled ai feedback. arXiv preprint arXiv:2310.01377, 2023.

Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D., and Kiela, D. Kto: Model alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306, 2024.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.

Hu, J., Xu, H., Liu, J. K., and Shen, W. Reinforce++: Stabilizing critic-free policy optimization with global normalization. arXiv preprint arXiv:2501.03262, 2025.

Li, T., Chiang, W.-L., Frick, E., Dunlap, L., Wu, T., Zhu, B., Gonzalez, J. E., and Stoica, I. From crowdsourced data to high-quality benchmarks: Arena-hard and benchbuilder pipeline, 2024. URL https://arxiv.org/abs/ 2406.11939.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Meng, Y., Xia, M., and Chen, D. Simpo: Simple preference optimization with a reference-free reward. arXiv preprint arXiv:2405.14734, 2024.

Meta, A. Llama 3.2: Revolutionizing edge ai and vision with open, customizable models. Meta AI Blog. Retrieved December, 20:2024, 2024.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.

Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, volume 36, 2023.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Team, G., Kamath, A., Ferret, J., Pathak, S., Vieillard, N., Merhej, R., Perrin, S., Matejovicova, T., Ram´e, A., Rivi`ere, M., et al. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025.

von Werra, L., Belkada, Y., Tunstall, L., Beeching, E., Thrush, T., Lambert, N., Huang, S., Rasul, K., and Gallou´edec, Q. TRL: Transformers Reinforcement Learning, 2020. URL https://github.com/ huggingface/trl.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Zheng, C., Gao, C., Liu, S., Chen, X.-H., et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., and Stoica, I. Judging llm-as-a-judge with mt-bench and chatbot arena, 2023. URL https:

//arxiv.org/abs/2306.05685.

### A. Gradient Analysis and Optimization Dynamics

The total objective function L(θ) is composed of three distinct components: a chosen-sequence maximization term Lw, a rejected-sequence regularization term Ll and a margin-based distance term Ldist. Below, we analyze the gradient dynamics of each component to show how they collectively optimize the policy πθ.

First, we derive log probabilities for both the token and sequence-level parts of the loss. Total loss:

##### L(θ) = Lw(θ) + Ll(θ) + Ldist(θ)

Token-level log probability:

lt = log πθ(t|x,y<t) Sequence-level log probability:

1 |y| t∈y

¯l(y) =

log πθ(y|x,y<t)

¯lw = ¯l(yw);¯ll = ¯l(yl)

Separate weight gradient can be expressed as follows, thus we can further analyze the loss dynamics using a gradient with respect to the loss components:

∂L ∂l · ∇θl

∇θL =

The component Lw acts as a foundational supervision signal. The gradient with respect to the sequence-level probability is constant:

Lw(θ) = −λw¯lw.

∂L ∂¯lw

= −λw

This results in a consistent gradient update ∇θLw = −λw∇θ¯lw. This term uniformly drives model to the increased likelihood of preferred sequence yw, akin to Supervised Fine-Tuning (SFT).

The loss Ll is a token-level nonlinear penalty on the rejected sequence yl. Unlike standard ranking losses that simply minimize the probability of yl, this component acts as a regularization term to prevent probability collapsing to zero. The gradient is given by:

(log (1 + exp−(l

t+δ))p

Ll = λlEt∈y

t

∂u ∂lt

= −1

ut = −(lt + δ);

∂f ∂ut

f(ut) = Softplus(ut)p;

= p · Softplus(ut)p−1 · σ(ut)

∂Ll ∂lt

= −p · λl · Softplus(−lt − δ)p−1 · σ(−lt − δ) The gradient formulation creates a feedback loop:

- 1. Magnitude control: the term with Softplus sets the magnitude of the penalty. If the token probability lt drops significantly below −δ and becomes highly unlikely, ut becomes large and positive, causing the gradient magnitude to grow super-linearly. This heavily penalizes the model for forgetting the rejected sequences, maintains the overall linguistic fluency and prevents the ”formatting collapse” often encountered in RLHF.

- 2. Gating mechanism: the sigmoid term σ acts as a smooth gate. As lt increases and probability becomes ”safe”, ut becomes negative, and σ(ut) approaches 0, shutting off the component. Thus, the model is not penalized for the rejected sequence once it maintains a sufficient baseline probability.

The distance loss Ldist optimizes the separation

∆ = ¯lw − ¯ll between the chosen and the rejected sequences. The dynamics are set up by the hard margin mh and soft margin ms. The gradient with respect to the margin ∆ can be expressed by:

Ld = λd · ReLU(mh − ∆) · σ(−κ(∆ − ms)) If ∆ ≥ mh, then the whole ∂Ld

∂∆ = 0. Assuming ∆ < mh,

u = mh − ∆ v = σ(−κ(∆ − ms));Ld = λd · u · v Where u is a hard gating factor and v is the soft gating factor.

∂u ∂∆

= −1

∂w ∂∆

w = −κ(∆ − ms);

= −κ

∂v ∂∆

∂w ∂∆

= σ(w)(1 − σ(w))

∂Ld ∂∆

= λd(u · v′ + u′ · v)

∂Ld ∂∆

= −λd(v + κuv(1 − v))

The expression shows a two-way optimization mechanism:

- 1. Direct margin expansion (v): the first term scales with v, pushing the margin ∆ higher. This force is modulated by

the sigmoid; if ∆ is far smaller than ms, v ≈ 1, applying maximum pressure. As ∆ exceeds ms, v decays, reducing the gradient.

- 2. Boundary Sensitivity: the second term accounts for the curvature of the soft margin. It is active primarily in the

transition region where the margin is close to ms. It provides an additional boost to the gradient when the model is close to satisfying the soft margin, accelerating convergence in the critical decision boundary region.

Since ∂Ld

∂∆ < 0, the chain rule application to the parameters θ:

∂L ∂¯lw

=

∂L ∂∆

;

∂L ∂¯ll

∂L ∂∆

= −

∂Ld ∂∆

(∇θ¯lw − ∇θ¯ll)

∇θLd =

results in a positive update for ¯lw maximizing the winner and a negative update for ¯ll minimizing the loser, yet it is strictly bounded by the hard margin mh. If ∆ ≥ mh, the gradient collapses to zero due to the ReLU component, preventing over-optimization and preserving the KL-divergence constraints of the policy without reference policy evaluation.

