# arXiv:2601.00423v1[cs.LG]1Jan2026

## E-GRPO: High Entropy Steps Drive Effective Reinforcement Learning for Flow Models

Shengjun Zhang, Zhang Zhang, Chensheng Dai, Yueqi Duan† Tsinghua University

{zhangsj23, z-z23}@mails.tsinghua.edu.cn, duanyueqi@tsinghua.edu.cn

#### Abstract

Recent reinforcement learning has enhanced the flow matching models on human preference alignment. While stochastic sampling enables the exploration of denoising directions, existing methods which optimize over multiple denoising steps suffer from sparse and ambiguous reward signals. We observe that the high entropy steps enable more efficient and effective exploration while the low entropy steps result in undistinguished roll-outs. To this end, we propose E-GRPO, an entropy aware Group Relative Policy Optimization to increase the entropy of SDE sampling steps. Since the integration of stochastic differential equations suffer from ambiguous reward signals due to stochasticity from multiple steps, we specifically merge consecutive low entropy steps to formulate one high entropy step for SDE sampling, while applying ODE sampling on other steps. Building upon this, we introduce multi-step group normalized advantage, which computes group-relative advantages within samples sharing the same consolidated SDE denoising step. Experimental results on different reward settings have demonstrated the effectiveness of our methods. Our code is available at https: //github.com/shengjun-zhang/VisualGRPO.

#### 1. Introduction

Recent advances in generative models have significantly propelled the field of visual content creation, enabling a wide array of applications ranging from artistic design and entertainment to medical imaging and virtual reality. Stateof-the-art diffusion models [13, 29, 33] and flow-based approaches [19, 21] have achieved remarkable fidelity in generating high-quality images and videos [5, 8, 26].

In large language models, reinforcement learning has demonstrated its effectiveness on the alignment with human preferences, including Proximal Policy Optimization (PPO) [30], Direct Policy Optimization (DPO) [28], and Group Relative Policy Optimization (GRPO) [32]. Thus, reinforcement learning from human feedback (RLHF) [4, 9]

has been employed in post-training stages for visual generation. Since GRPO simplifies the architecture by eliminating the value network, using intra-group relative rewards to compute advantages directly, recent works [20, 38] integrate this into flow models with stochastic differential equations (SDE). To enhance the efficiency of sampling, some methods [12, 17, 22, 40] introduce a mixture of SDE sampling and ODE sampling, while others [10, 18] propose a tree-based structure for less sampling steps.

Despite these advancements, existing GRPO-based methods apply policy optimization across multiple denoising timesteps, resulting in sparse and ambiguous reward signals that hinder effective alignment. We observe that only high-entropy timesteps contribute meaningfully to training dynamics. As shown in Figure 1 (b), stochastic exploration via SDE at timesteps with higher noise level possess larger entropy. We visualize the generated images under different circumstances, where high-entropy timesteps yield diverse images with distinguishable reward variations, while lowentropy timesteps produce less reward differences, which are similar to those induced by adding 10% random noise to the final image. This phenomenon implies that reward models struggle to discern subtle trajectory deviations in lowentropy regimes. Furthermore, we implement GRPO with SDE sampling on four settings: (i) the first 4 timesteps, (ii) the first 8 timesteps, (iii) the second 8 timesteps, and (iv) all 16 steps. Notably, optimization on the first half timesteps performs even better than on all timesteps, which indicates that the second half timesteps are largely uninformative.

To address this limitation, we propose E-GRPO, an entropy-aware SDE sampling strategy for more effective exploration during GRPO training. An intuitive approach would be to employ multi-step continuous SDE sampling to broaden exploration. However, this introduces cumulative stochasticity results in ambiguous reward attribution across steps, so that a beneficial exploration in one step may be penalized due to suboptimal downstream trajectory deviations, leading to optimization in the opposing direction. Instead, we consolidate multiple low-entropy SDE steps into a single effective SDE step while keeping the remaining steps

[Figure 1]

[Figure 2]

(b)

[Figure 3]

(a) (c)

- Figure 1. The influence of entropy for sampling results. (a) We visualize the generation images with different SDE sampling strategy, including one-step SDE on step 2, one-step SDE on step 6, and merged-step SDE on step 6. We also report the variance of clip score for generated images. Samples from the initial steps and merged steps share higher differences, while posterior steps generate undistinguishable samples, whose variance is similar to small perturbation on original images. (b) We report the entropy of SDE sampling on each timestep with different merged steps. More merged steps indicate higher entropy and larger exploration scope in RL training. (c) We visualize the training reward curves on models trained on all timesteps, the first half timesteps, and the second half timesteps.

deterministic as ODE sampling, thereby preserving highentropy exploration only where informative and ensuring reliable reward attribution. Building upon this, we introduce multi-step group normalized advantage, which computes group-relative advantages within samples sharing the same consolidated SDE step. This mechanism provides dense and trustworthy reward signals, enhancing the alignment of generative trajectories with human preferences.

We conduct experiments on both single-reward settings and multi-reward settings and evaluation on in-domain and out-of-domain matrices. Experimental results demonstrate the effectiveness and efficiency of our method. Our main contribution can be summarized as follows:

- 1. We provide a comprehensive entropy-based analysis of denoising timesteps in GRPO training process, revealing that effective alignment can be achieved by optimizing exclusively at high-entropy steps.
- 2. We propose E-GRPO, an entropy-aware SDE sampling strategy for GRPO training of flow models, which consolidates multiple low-entropy steps into a single highentropy SDE step, thereby expanding meaningful exploration while eliminating reward attribution ambiguity.
- 3. We conduct extensive experiments under both singlereward and multi-reward settings, clearly demonstrating that E-GRPO consistently outperforms prior meth-

ods, validating the efficacy and robustness of targeted, entropy-guided stochastic optimization.

#### 2. Related Works

RL Alignment for Image Generation. Reinforcement Learning from Human Feedback (RLHF) [3, 24] and Reinforcement Learning with Verifiable Rewards (RLVR) [16] have emerged as powerful paradigms for aligning large language models (LLMs) with human preferences [1, 3, 31]. Early frameworks based on Proximal Policy Optimization (PPO) [30] rely on a value model to guide policy updates, whereas recent approaches such as Group Relative Policy Optimization (GRPO) [7, 23, 32] achieve greater stability and efficiency by leveraging relative group-wise comparisons instead of absolute rewards. These advancements in language alignment have inspired increasing interest in transferring RL techniques to align visual generative models with human preferences. In the visual generation domain, diffusion [13, 33] and flow matching models [19, 21, 25] have demonstrated strong generative capabilities through iterative denoising processes [26, 29]. To enhance alignment with human feedback, recent studies have adapted RLHF to these models. DiffusionDPO [34], and D3PO [39] extend Direct Preference Op-

timization (DPO) [28] to diffusion models. However, these methods suffer from distribution shifting because no new samples are generated during the training process. While DanceGRPO [38] and Flow-GRPO [20] reformulate deterministic ODE-based sampling into stochastic SDE trajectories, enabling GRPO-style policy updates in visual domains. Building upon this foundation, GranularGRPO [40] refines timestep granularity for more precise and dense credit assignment across denoising steps, and TempFlow-GRPO [12] introduces temporally-aware weighting to alleviate the limitations of uniform optimization across timesteps. MixGRPO [17] further improves training efficiency through a hybrid ODE–SDE sampling mechanism, while BranchGRPO [18] enhances exploration efficiency via branching rollouts and structured pruning. Despite these advancements, existing GRPO frameworks for flow models typically optimize uniformly across all timesteps, overlooking the heterogeneity of exploration potential during the denoising process and suffering from sparse or noisy reward signals. Our work addresses these challenges by leveraging step-wise entropy as a measure of exploration capacity, enabling optimization on high entropy steps to improve both stability and efficiency.

Entropy-Guided Exploration and Alignment. Early work in reinforcement learning (RL) has recognized the importance of entropy as a mechanism for promoting effective exploration. In particular, strategies such as policy entropy regularization have been widely used to stabilize learning and encourage diverse behavior [2]. For example, Soft Actor-Critic (SAC) [11] explicitly maximizes the expected reward while also maximizing policy entropy, resulting in more robust and sample-efficient exploration. More recently, entropy-based insights have been applied to large language models (LLMs) in the context of reinforcement learning for reasoning. Study shows that a small fraction of high-entropy tokens disproportionately drives policy improvement, highlighting the significance of tokenlevel uncertainty in guiding exploration [35]. Complementary work further formalizes entropy as a lens for understanding exploration dynamics, demonstrating that highentropy regions correspond to critical decision points that are most informative for learning [6]. Inspired by these findings, we investigate whether similar entropy-driven patterns arise in flow matching models, and propose an entropyaware GRPO framework that prioritizes informative denoising steps, leading to more efficient and stable alignment with human preferences.

#### 3. Methods

##### 3.1. Preliminary

To enable exploration in reinforcement learning, flow-based Group Relative Policy Optimization (GRPO) converts de-

terministic ODE sampling:

dxt = vθ(xt,t)dt (1) into an equivalent SDE:

σt2 2t

xt + (1 − t)vθ(xt,t) ∆t

xt+∆t =xt + vθ(xt,t) +

√

∆tϵ, (2)

+ σt

with ϵ ∼ N(0,I) and σt = a 1−t t. With SDE sampling, flow-based GRPO integrates online reinforcement learning into flow matching models by framing the reverse sampling as a Markov Decision Process (MDP) with states st = (xt,t), actions at = xt−1 ∼ πθ(·|st), and terminal rewards R(x0,c) for prompt c. The policy optimizes

JFlow-GRPO(θ) = Ec∼C, {x(i)}Gi=1∼πθold(·|c) [f(r,A,θ,ϵ)]. The clipped surrogate objective f(r,A,θ,ϵ) is defined as:

T−1

G

1 T

1 G

t=0

i=1

min rt(i)A(i),clip(rt(i),1 − ϵ,1 + ϵ)A(i) ,

(i) t−1|x(ti),c)

where rt(i)(θ) = pθ(x

, and pθ(x(t−i)1 | x(ti),c) is

pθold(x(ti−)1|x(ti),c)

the policy function for output x(i) at timestep t − 1. The group-normalized advantages A(i) is formulated as:

R(x(0i),c) − mean{R(x(0j),c)}Gj=1 std{R(x(0j),c)}Gj=1

A(i) =

. (3)

Following the practices of previous methods [17, 38], the KL-regularization item is omitted in the objective function.

##### 3.2. Entropy Analysis

In flow-based Group Relative Policy Optimization, the reverse sampling process from an SDE is framed as a Markov Decision Process (MDP). To derive the entropy of the reverse SDE step, we start from the given forward SDE and apply Bayes’ theorem. The forward SDE is given by:

√

∆tϵ, (4)

xt+∆t = xt + µθ(xt,t)∆t + σt

where ϵ ∼ N(0,I) injects stochasticity, and the drift term is:

σt2 2t

µθ(xt,t) = vθ(xt,t) +

xt + (1 − t)vθ(xt,t) .

The transition probability of the forward SDE is a Gaussian distribution:

###### pf(xt+∆t | xt) = N xt+∆t | xt + µθ(xt,t)∆t, σt2∆tI .

[Figure 4]

- Figure 2. E-GRPO sampling strategy. First, we generate a set of anchor noise latents corresponding to different timesteps. For

each active SDE timestep ti, merged steps Ti is selected based on entropy analysis. We generate a group of results based on the specific SDE sampling of merged steps, and compute the advantage within each group.

Reverse SDE via Bayes’ Theorem. The reverse transition probability pr(xt | xt+∆t), which corresponds to the policy πθ in GRPO, can be derived using Bayes’ theorem:

pf(xt+∆t | xt)p(xt) p(xt+∆t)

pr(xt | xt+∆t) =

.

For a Gaussian process, the reverse transition is also a Gaussian distribution:

pr(xt | xt+∆t) = N xt | µ˜θ(xt+∆t,t), σ˜t2∆tI ,

where µ˜θ is the reverse drift and σ˜t is the reverse diffusion coefficient. For linear Gaussian SDEs, the diffusion coefficient is the same in both directions when the process is time-reversible, where σ˜t = σt. For the reverse drift µ˜θ, the log of the forward transition probability is:

- 1

- 2

log det(2πσt2∆tI) −

log pf = −

- 1

- 2σt2∆t ∥xt+∆t − xt − µθ(xt,t)∆t∥2

Taking the derivative with respect to xt, we find the reverse drift is formulated as:

µ˜θ(xt+∆t,t) = xt+∆t − µθ(xt,t)∆t + σt2∆t∇xt

log p(xt).

Entropy of the Reverse SDE Step. The entropy of a multivariate Gaussian distribution N(µ,Σ) is given by:

d 2

log (2πe)d det(Σ) (5)

h(y) =

where d is the dimension of the random variable y. For the reverse SDE step, the covariance matrix is given by:

Σr = σt2∆tI. (6)

[Figure 5]

Figure 3. Ambiguous reward signal. For consecutive multistep SDE sampling, the advantage is corresponding to multiple timesteps, which may results in wrong optimization direction on the specific timestep. Our merged-step SDE sampling not only enlarges the exploration scope, but also eliminate ambiguous reward by aligning the final advantage to one merged SDE step.

The determinant of this diagonal matrix is (σt2∆t)d. Substituting this into the entropy formula:

h(t) =

=

=

- 1

- 2

log (2πe)d(σt2∆t)d (7)

- 1

- 2

dlog(2πe) + dlog(σt2∆t) (8)

d 2

log 2πeσt2∆t (9)

Substituting σt = a 1−t t, we get:

d 2

t 1 − t · ∆t (10)

log 2πe · a2 ·

h(t) =

##### 3.3. Entropy-aware GRPO

To address the sparse and ambiguous reward attribution of uniform optimization across timesteps, we propose an entropy-aware GRPO (E-GRPO) framework, which integrates an entropy-driven step merging strategy and multistep group normalized advantage estimation. The core design prioritizes meaningful exploration by consolidating low-entropy SDE steps into informative sampling events.

###### 3.3.1. Entropy-Driven Step Merging Strategy

Given a sequence of denoising timesteps {tT,...,t1,t0}, the relation of timesteps and the entropy is formulated as:

tk 1 − tk

eh(t

k) ∝

(tk − tk−1). (11)

Practically, flow models adjust time shift to balance quality and efficiency:

stˆk 1 + (s − 1)tˆk

tk =

,

where tˆk = Tk . Substituting this into (11), we get:

s2Tk (T − k)[T + (s − 1)k][T + (s − 1)(k − 1)]

eh(t

k) ∝

.

We define an adaptive entropy threshold τ to classify timesteps into high-entropy ones {tT,··· ,tM+1} with eh(t

k) ≥ τ and low-entropy ones {tM,··· ,t0} with eh(t

k) < τ. For a low-entropy timestep tm, we can introduce multi-step SDE sampling on consecutive timesteps {tm,...,tm−l}. As shown in Figure 3, this introduces cumulative stochasticity results in ambiguous reward attribution across steps, so that a beneficial exploration in one step may be penalized due to suboptimal downstream trajectory deviations, leading to optimization in the opposing direction. Thus, we consolidate consecutive timesteps into a single equivalent SDE step to eliminate ambiguous reward signals. Merging l consecutive low-entropy SDE steps requires preserving the total diffusion effect while reducing step count. For the consolidated timesteps, the time interval is ∆t = tm − tm−l. Substituting ∆t to (4), we have:

,tm)(tm−tm−l)+σt tm − tm−l ϵ. According to (6), the reverse SDE step is formulated as:

###### xt

= xt

+µθ(xt

m−l

m

m

Σ = σt2(tm − tm−l)I.

Thus, the entropy for the merged timestep is given by:

tm 1 − tm

eh(t

k) ∝

(tm − tm−l)

s2Tml (T − m)[T + (s − 1)m][T + (s − 1)(m − l)]

∝

,

where eh(t

k) is an increasing function of l.

Instead of using a uniform l for all low-entropy timesteps, we propose an adaptive strategy to select an optimal l for each low-entropy timestep, where l is determined such that the entropy of the merged step just exceeds the threshold τ. This design avoids excessively large entropy of a single merged step, which would make it difficult to find a proper optimization direction under limited exploration attempts. The adaptive selection of l ensures that each merged step maintains a moderate entropy level—sufficient to retain meaningful exploration signals while preventing the entropy from becoming too high to guide effective optimization. By aligning the merged entropy with the predefined threshold, we balance the efficiency gain from step merging and the reliability of reward-guided exploration.

Algorithm 1 Entropy-aware GRPO (E-GRPO)

Input: Initial policy θold, prompt set C, total timesteps T, active SDE sampling timesteps {tT,··· ,tN}, merging step count {lT,··· ,lN}, clipping coefficient ϵ, trajectory count {G(T),··· ,G(N)}

Output: Optimized policy θ

- 1: for iteration = 1 to K (total iterations) do
- 2: for c ∼ C (sample prompt) do
- 3: for N ≤ n ≤ T do
- 4: Tn ← {tn,tn−1,··· ,tn−l

n}

- 5: Generate G(n) trajectories with Tn
- 6: Compute rewards {R(x(0j,t)

n

,c)}G

(n)

j=1

- 7: A(ti)

n

← R(x

(i) 0,tn,c)−mean{R(x(0j,t)n,c)}Gj=1(n)

std{R(x(0j,t)n,c)}Gj=1(n)

- 8: rt(i)

n

(θ) ← p

θ(x(tin)−l

n

|x(tin),c) pθold(x(tin)−l

n

|x(tin),c)

- 9: end for
- 10: Construct clipped surrogate: f(r,A,θ,ϵ)
- 11: Update θ by minimizing JE-GRPO(θ)
- 12: Set θold ← θ
- 13: end for
- 14: end forreturn Optimized policy θ

###### 3.3.2. Policy Optimization Objective

To resolve reward attribution ambiguity, we extend GRPO’s group normalization to merged steps by defining mergegrouped samples. We designate a set of active SDE timesteps {tT,··· ,tN}, where each timestep tn (with N ≤ n ≤ T) is associated with a merging step count ln determined by the entropy-driven strategy in Section 3.3.1. For a given prompt c, we generate Gn trajectories for each active timestep tn, where all Gn trajectories share the same consolidated merged timesteps Tn ≜ {tn,tn−1,··· ,tn−l

n}.

Within each merge group Tn, we first compute the advantage estimates using the Gn trajectories, ensuring reward signals are attributed consistently to the merged timesteps. The advantage of the i-th trajectory at state xt

n

estimated over the merge group Tn is given by:

R(x(0i,t)

,c) − mean{R(x(0j,t)

,c)}G

n

j=1 std{R(x(0j,t)

A(ti)

, (12)

n

n

=

,c)}G

n

n

j=1

n

where x(0j,t)

denotes the j-th generated results with active SDE timestep tn. The final clipped surrogate objective f(r,A,θ,ϵ), adapted to merge-grouped samples, is then formulated as:

n

G(n)

T

1 Tˆ

1 G(n)

min rt(i)

n

i=1

n=N

A(ti)

,clip(rt(i)

,1 − ϵ,1 + ϵ)A(ti)

n

n

n

,

- Table 1. Evaluation Results. Comparison between different methods. The best and second best results in each column are bolded and underline respectively.

Training Reward Model: HPS Training Reward Models: HPS&CLIP HPS CLIP PickScore ImageScore HPS CLIP PickScore ImageScore

Method

FLUX.1-dev [15] 0.311 0.388 0.231 1.089 0.311 0.388 0.231 1.089 DanceGRPO [38] 0.353 0.375 0.228 1.233 0.331 0.389 0.227 1.128 MixGRPO [17] 0.378 0.358 0.225 1.266 0.363 0.399 0.230 1.436 GranularGRPO [40] 0.385 0.355 0.229 1.313 0.377 0.400 0.236 1.490 BranchGRPO [18] 0.358 0.365 0.231 1.311 0.342 0.384 0.230 1.243 TempFlowGRPO [12] 0.382 0.357 0.231 1.264 0.310 0.388 0.230 1.106 Ours 0.391 0.355 0.232 1.324 0.382 0.401 0.237 1.494

where Tˆ = T − N and the ratio rt(i)

is given by:

n

pθ(x(ti)

| x(ti)

,c) pθ

rt(i)

. (13)

n−ln

n

(θ) =

(x(ti)

| x(ti)

n

,c)

old

n−ln

n

Building on the clipped surrogate objective of GRPO, EGRPO restricts optimization to consolidated high-entropy steps. The objective function is modified to:

JE-GRPO(θ) = Ec∼C,{x(i)

tn}Gi=1(n)∼πθold(·|c),N≤n≤Tf(r,A,θ,ϵ).

Our strategy is illustrated in Algorithm 1. We first Compute h(tk) for all timesteps using (10) and determine τ. Then, we cluster consecutive low-entropy steps Ti for timestep ti. We generate trajectories using ODE for other steps and consolidated SDE for merged steps. Finally, we estimate advantages A(ti)

and ratio via (12) and (13) so that reward signals are attributed consistently to the merged timesteps, and update pθ by minimizing JE-GRPO(θ).

n

#### 4. Experiments

##### 4.1. Experimental Settings

Dataset and Model. We conduct our experiments on the HPD dataset [36], a large-scale dataset for human preference evaluation, containing approximately 103,000 text prompts for training and 400 prompts for testing. For our experiments, we adopt FLUX.1-dev [15] as the backbone flow matching model, consistent with prior works such as DanceGRPO [38] and MixGRPO [17].

Evaluation Settings. To assess alignment with human preferences, we employ several representative reward models, each capturing different aspects of generated images. HPSv2.1 [36] and PickScore [14] are both trained on large-scale human preference data, thus reflecting human judgments of overall image quality and text–image consistency. CLIP Score [27] primarily measures the semantic alignment between the generated image and the input prompt. ImageReward [37] focuses on the perceptual quality and aesthetic

[Figure 6]

Figure 4. Comparison of Training Reward Curves. The reward curve of E-GRPO demonstrates faster and more stable improvement during training compared to baseline methods. This indicates that exploration guided by high-entropy steps can enhance learning efficiency while mitigating noise in the reward signal.

appeal of the image, providing a complementary perspective to preference-based metrics.

Sampling Strategy. Following DanceGRPO, images generated in training are sampled from the same initialized noise to form rollout groups. We set the total number of sampling steps to T = 16 and the parameter a in the equa-

t 1 − t

to 0.7. During training, the entropy threshold τ for step merging is set as 2.2.

tion σt = a

Training Details. We first train the model using only the HPS-v2.1 reward model to estimate the upper performance bound of our approach. To enhance robustness and mitigate potential reward hacking, we further train the model with both HPS-v2.1 and CLIP as joint reward signals. Training is only conducted on the first half sampling steps.

Optimization Details. All experiments are performed on 8 × NVIDIA A800 GPUs with a batch size of 1. We employ the AdamW optimizer with a learning rate 2 × 10−6

FLUX DanceGRPO BranchGRPO MixGRPO G2RPO Ours

[Figure 7]

“A papaya fruit dressed as a sailor.”

[Figure 8]

“A lemon with a McDonald's hat.”

[Figure 9]

“A spoon dressed up with eyes

and a smile”

[Figure 10]

“The Mona Lisa wearing hea-

dphones and listening to Lana Del Rey on a phone, depicted with photorealistic high detail.”

Figure 5. Visualization Comparisons. Comparison between E-GRPO with other baseline methods. E-GRPO better integrates semantics and fine-grained details.

and a weight decay of 1 × 10−4. Mixed-precision training is enabled using the bfloat16 format. The total number of training iterations is 300.

##### 4.2. Main Experiments

As shown in Tab. 1, we evaluate our method against several recent methods, including the baseline FLUX.1-dev [15], DanceGRPO [38], MixGRPO [17], BranchGRPO [18], TempFlowGRPO [12] and GranularGRPO [40]. When trained with the single HPS-v2.1 reward, our method achieves a new state-of-the-art performance, surpassing DanceGRPO by 10.8% on the HPS metric. This demonstrates that our entropy-guided exploration effectively identifies high-value denoising steps, leading to more precise and stable policy optimization.

However, as discussed in DanceGRPO [38], training solely with HPS-v2.1 can lead to reward hacking, resulting in overly saturated visual results that do not align with genuine human preferences. To address this, we follow prior works [17, 38, 40] and adopt a joint reward scheme using both HPS-v2.1 and CLIP score as reward during training. Under this more robust multi-reward setting, our approach not only maintains its SOTA performance on the in-domain HPS metric but also achieves substantial improvements on out-of-domain metrics. In particular, compared with DanceGRPO, our method improves ImageReward by 32.4% and

PickScore by 4.4%, highlighting that entropy-guided optimization promotes broader generalization across reward models and effectively mitigates reward hacking.

Figure 5 presents qualitative comparisons among FLUX.1-dev, DanceGRPO, BranchGRPO, MixGRPO, G2RPO, and our proposed E-GRPO. As shown in the first row (prompt: “A papaya fruit dressed as a sailor”), EGRPO generates a composition that naturally integrates the papaya’s structure with human-like attire, yielding images of higher aesthetic quality and greater realism. In contrast, baseline methods either misinterpret the prompt (e.g., DanceGRPO generates a person holding a papaya) or produce visually incoherent results (e.g., MixGRPO and G2RPO). In the third row (prompt: “A spoon dressed up with eyes and a smile”), E-GRPO produces expressive and visually consistent humanized faces while preserving the metallic texture of the spoon, whereas other methods generate unrealistic facial blending or lose material fidelity. These results highlight that E-GRPO achieves superior semantic grounding and visual coherence, leading to images that more faithfully reflect textual intent and human aesthetic preference.

Figure 4 illustrates the reward trajectories during training. Compared with prior work, our method exhibits faster early-stage reward growth and smoother convergence, achieving a higher final reward. This indicates that the

- Table 2. Comparison of Step Merging Strategies. Quantitative results comparing different step merging strategies during training. The proposed entropy-aware adaptive merging consistently achieves higher scores on HPS, CLIP, PickScore, and ImageScore, indicating better semantic alignment and generation quality. The best results in each column are bolded

.

Merging Strategies HPS CLIP PickScore ImageScore

2-step 0.382 0.290 0.232 1.223 4-step 0.374 0.302 0.230 1.216 6-step 0.337 0.372 0.226 1.298

Adaptive 0.391 0.355 0.232 1.324

entropy-guided step selection stabilizes optimization by focusing updates on the most informative denoising steps, improving both efficiency and reliability.

##### 4.3. Ablation Studies

In order to evaluate the effectiveness of the proposed method, we conduct a series of ablation experiments to understand the design of E-GRPO.

Step Merging Strategies. We evaluate several step merging strategies to verify the effectiveness of our entropybased adaptive merging scheme during training. As shown in Tab. 2, our method consistently outperforms the naive 2-step, 4-step, and 6-step merging baselines across almost all evaluation metrics, demonstrating both efficiency and robustness. Compared with fixed merging strategies that combine multiple steps regardless of their exploration level, our entropy-aware adaptive merging dynamically adjusts the merging behavior to maintain comparable exploration across steps, leading to more accurate and efficient optimization.

Step Entropy Analysis. To validate the rationality of the entropy-based analysis and effectiveness of the proposed entropy-aware GRPO method, we conduct experiments by training models on different subsets of denoising steps. Specifically, we train separate models using (1) the first 4 steps, (2) the first 8 steps, (3) the last 8 steps, and (4) all steps. As shown in Fig. 1(c), training on the first 8 highentropy steps achieves the best performance, followed by using the first 4 steps. In contrast, training on all steps yields similar results to the first 4-step case but with substantially higher computational cost. When the model is trained on the last 8 (low-entropy) steps, the performance drops dramatically. These results indicate that focusing training on early high-entropy steps is sufficient to achieve strong performance, while involving too many later low-entropy steps introduces unnecessary noise and inefficiency. Therefore, we adopt the first 8 denoising steps as our default training configuration. Tab. 3 further provides quantitative results for different subsets of training steps.

Table 3. Comparison of Different Training Denoising Steps. Models trained on high-entropy (early) steps achieve higher alignment scores with lower computational cost, confirming that highentropy steps contribute most to effective optimization. The highest score in each column is bolded. Note that the CLIP score shows an unexpected deviation, which is caused by training solely with the HPS reward, leading to a certain degree of reward hacking as discussed earlier.

Merging Strategies HPS CLIP PickScore ImageScore

First 4 Steps 0.370 0.348 0.231 1.252 First 8 Steps 0.391 0.355 0.232 1.324 Second 8 Steps 0.357 0.381 0.231 1.250 Full Steps 0.366 0.359 0.231 1.169

#### 5. Conclusion

This work addresses the critical challenge of sparse and ambiguous reward signals in existing Group Relative Policy Optimization (GRPO)-based methods for flow models, which stem from uniform optimization across all denoising timesteps. Through entropy analysis, we reveal a key insight that high-entropy timesteps contribute meaningfully to effective exploration and human preference alignment, while low-entropy timesteps yield undistinguished rollouts that hinder reward discrimination. To tackle this limitation, we propose E-GRPO, an entropy-aware framework that integrates two core innovations, including an adaptive entropy-driven step merging strategy and multi-step group normalized advantage estimation. The step merging strategy consolidates consecutive low-entropy SDE steps into single high-entropy SDE steps, while retaining ODE sampling for other steps, eliminating reward attribution ambiguity caused by cumulative stochasticity. The multi-step group normalized advantage ensures dense and reliable reward signals by computing relative advantages within samples sharing the same consolidated step. Extensive experiments on the HPD dataset with FLUX.1-dev as the backbone validate the efficacy of E-GRPO.

Limitations and Future Works. A critical bottleneck in advancing visual generative models lies in the design and alignment of reward signals. Rewards serve as the cornerstone of guiding reinforcement learning paradigms toward generating high-quality, human-preferred content, yet existing reward formulations often fail to fully align with nuanced human preferences—such as aesthetic appeal, semantic consistency, and contextual appropriateness. This misalignment not only leads to suboptimal generation outcomes but also renders models vulnerable to reward hacking: models may exploit loopholes in the reward function to maximize scores without genuinely meeting human expectations. As a result, the development of more robust and effective reward models remains an essential direction for future research in visual RL-driven generation.

Acknowledgments. This work was supported in part by the Beijing Natural Science Foundation under Grant L252011, and by the National Natural Science Foundation of China under Grant 62576185.

#### References

- [1] Marwa Abdulhai, Isadora White, Charlie Snell, Charles Sun, Joey Hong, Yuexiang Zhai, Kelvin Xu, and Sergey Levine. Lmrl gym: Benchmarks for multi-turn reinforcement learning with language models, 2023. 2
- [2] Zafarali Ahmed, Nicolas Le Roux, Mohammad Norouzi, and Dale Schuurmans. Understanding the impact of entropy on policy optimization. In Proceedings of the 36th International Conference on Machine Learning, pages 151–160. PMLR,

2019. 3

- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. 2
- [4] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023. 1
- [5] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 1
- [6] Daixuan Cheng, Shaohan Huang, Xuekai Zhu, Bo Dai, Wayne Xin Zhao, Zhenliang Zhang, and Furu Wei. Reasoning with exploration: An entropy perspective on reinforcement learning for llms, 2025. 3
- [7] DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao

- Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. 2
- [8] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

2024. 1

- [9] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models. Advances in Neural Information Processing Systems, 36:79858–79885, 2023. 1
- [10] Xiaolong Fu, Lichen Ma, Zipeng Guo, Gaojing Zhou, Chongxiao Wang, ShiPing Dong, Shizhe Zhou, Ximan Liu, Jingling Fu, Tan Lit Sin, et al. Dynamic-treerpo: Breaking the independent trajectory bottleneck with structured sampling. arXiv preprint arXiv:2509.23352, 2025. 1
- [11] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor, 2018. 3
- [12] Xiaoxuan He, Siming Fu, Yuke Zhao, Wanli Li, Jian Yang, Dacheng Yin, Fengyun Rao, and Bo Zhang. Tempflow-grpo: When timing matters for grpo in flow models. arXiv preprint arXiv:2508.04324, 2025. 1, 3, 6, 7
- [13] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 1, 2
- [14] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. In Advances in Neural Information Processing Systems, pages 36652–36663. Curran Associates, Inc., 2023. 6
- [15] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 6, 7

- [16] Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James Validad Miranda, Alisa Liu, Nouha Dziri, Xinxi Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Christopher Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, and Hannaneh Hajishirzi. Tulu 3: Pushing frontiers in open language model post-training. In Second Conference on Language Modeling, 2025. 2
- [17] Junzhe Li, Yutao Cui, Tao Huang, Yinping Ma, Chun Fan, Miles Yang, and Zhao Zhong. Mixgrpo: Unlocking flowbased grpo efficiency with mixed ode-sde. arXiv preprint arXiv:2507.21802, 2025. 1, 3, 6, 7
- [18] Yuming Li, Yikai Wang, Yuying Zhu, Zhongyu Zhao, Ming Lu, Qi She, and Shanghang Zhang. Branchgrpo: Stable and efficient grpo with structured branching in diffusion models. arXiv preprint arXiv:2509.06040, 2025. 1, 3, 6, 7
- [19] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 1, 2
- [20] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025. 1, 3
- [21] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 1, 2
- [22] Yifu Luo, Penghui Du, Bo Li, Sinan Du, Tiantian Zhang, Yongzhe Chang, Kai Wu, Kun Gai, and Xueqian Wang. Sample by step, optimize by chunk: Chunk-level grpo for text-to-image generation. arXiv preprint arXiv:2510.21583,

2025. 1

- [23] OpenAI, :, Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie, Alex Karpenko, Alex Tachard Passos, Alexander Neitz, Alexander Prokofiev, Alexander Wei, Allison Tam, Ally Bennett, Ananya Kumar, Andre Saraiva, Andrea Vallone, Andrew Duberstein, Andrew Kondrich, Andrey Mishchenko, Andy Applebaum, Angela Jiang, Ashvin Nair, Barret Zoph, Behrooz Ghorbani, Ben Rossen, Benjamin Sokolowsky, Boaz Barak, Bob McGrew, Borys Minaiev, Botao Hao, Bowen Baker, Brandon Houghton, Brandon McKinzie, Brydon Eastman, Camillo Lugaresi, Cary Bassin, Cary Hudson, Chak Ming Li, Charles de Bourcy, Chelsea Voss, Chen Shen, Chong Zhang, Chris Koch, Chris Orsinger, Christopher Hesse, Claudia Fischer, Clive Chan, Dan Roberts, Daniel Kappler, Daniel Levy, Daniel Selsam, David Dohan, David Farhi, David Mely, David Robinson, Dimitris Tsipras, Doug Li, Dragos Oprica, Eben Freeman, Eddie Zhang, Edmund Wong, Elizabeth Proehl, Enoch Cheung, Eric Mitchell, Eric Wallace, Erik Ritter, Evan Mays, Fan Wang, Felipe Petroski Such, Filippo Raso, Florencia Leoni, Foivos Tsimpourlas, Francis Song, Fred von Lohmann, Freddie Sulit, Geoff Salmon, Giambattista Parascandolo, Gildas Chabot, Grace Zhao, Greg Brockman, Guillaume Leclerc, Hadi Salman, Haiming Bao, Hao Sheng, Hart Andrin, Hessam Bagherinezhad, Hongyu Ren, Hunter Light-

- man, Hyung Won Chung, Ian Kivlichan, Ian O’Connell, Ian Osband, Ignasi Clavera Gilaberte, Ilge Akkaya, Ilya Kostrikov, Ilya Sutskever, Irina Kofman, Jakub Pachocki, James Lennon, Jason Wei, Jean Harb, Jerry Twore, Jiacheng Feng, Jiahui Yu, Jiayi Weng, Jie Tang, Jieqi Yu, Joaquin Qui˜nonero Candela, Joe Palermo, Joel Parish, Johannes Heidecke, John Hallman, John Rizzo, Jonathan Gordon, Jonathan Uesato, Jonathan Ward, Joost Huizinga, Julie Wang, Kai Chen, Kai Xiao, Karan Singhal, Karina Nguyen, Karl Cobbe, Katy Shi, Kayla Wood, Kendra Rimbach, Keren Gu-Lemberg, Kevin Liu, Kevin Lu, Kevin Stone, Kevin Yu, Lama Ahmad, Lauren Yang, Leo Liu, Leon Maksin, Leyton Ho, Liam Fedus, Lilian Weng, Linden Li, Lindsay McCallum, Lindsey Held, Lorenz Kuhn, Lukas Kondraciuk, Lukasz Kaiser, Luke Metz, Madelaine Boyd, Maja Trebacz, Manas Joglekar, Mark Chen, Marko Tintor, Mason Meyer, Matt Jones, Matt Kaufer, Max Schwarzer, Meghan Shah, Mehmet Yatbaz, Melody Y. Guan, Mengyuan Xu, Mengyuan Yan, Mia Glaese, Mianna Chen, Michael Lampe, Michael Malek, Michele Wang, Michelle Fradin, Mike McClay, Mikhail Pavlov, Miles Wang, Mingxuan Wang, Mira Murati, Mo Bavarian, Mostafa Rohaninejad, Nat McAleese, Neil Chowdhury, Neil Chowdhury, Nick Ryder, Nikolas Tezak, Noam Brown, Ofir Nachum, Oleg Boiko, Oleg Murk, Olivia Watkins, Patrick Chao, Paul Ashbourne, Pavel Izmailov, Peter Zhokhov, Rachel Dias, Rahul Arora, Randall Lin, Rapha Gontijo Lopes, Raz Gaon, Reah Miyara, Reimar Leike, Renny Hwang, Rhythm Garg, Robin Brown, Roshan James, Rui Shu, Ryan Cheu, Ryan Greene, Saachi Jain, Sam Altman, Sam Toizer, Sam Toyer, Samuel Miserendino, Sandhini Agarwal, Santiago Hernandez, Sasha Baker, Scott McKinney, Scottie Yan, Shengjia Zhao, Shengli Hu, Shibani Santurkar, Shraman Ray Chaudhuri, Shuyuan Zhang, Siyuan Fu, Spencer Papay, Steph Lin, Suchir Balaji, Suvansh Sanjeev, Szymon Sidor, Tal Broda, Aidan Clark, Tao Wang, Taylor Gordon, Ted Sanders, Tejal Patwardhan, Thibault Sottiaux, Thomas Degry, Thomas Dimson, Tianhao Zheng, Timur Garipov, Tom Stasi, Trapit Bansal, Trevor Creech, Troy Peterson, Tyna Eloundou, Valerie Qi, Vineet Kosaraju, Vinnie Monaco, Vitchyr Pong, Vlad Fomenko, Weiyi Zheng, Wenda Zhou, Wes McCabe, Wojciech Zaremba, Yann Dubois, Yinghai Lu, Yining Chen, Young Cha, Yu Bai, Yuchen He, Yuchen Zhang, Yunyun Wang, Zheng Shao, and Zhuohan Li. Openai o1 system card, 2024. 2
- [24] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730– 27744, 2022. 2
- [25] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4195–4205, 2023. 2
- [26] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint

- arXiv:2307.01952, 2023. 1, 2
- [27] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, pages 8748–8763. PMLR, 2021. 6
- [28] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023. 1, 3
- [29] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 2
- [30] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 1, 2
- [31] Lior Shani, Aviv Rosenberg, Asaf Cassel, Oran Lang, Daniele Calandriello, Avital Zipori, Hila Noga, Orgad Keller, Bilal Piot, Idan Szpektor, Avinatan Hassidim, Yossi Matias, and R´emi Munos. Multi-turn reinforcement learning from preference human feedback, 2024. 2
- [32] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 1, 2
- [33] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 1, 2
- [34] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8228–8238, 2024. 2
- [35] Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, and Junyang Lin. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning, 2025. 3
- [36] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341,

2023. 6

- [37] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for textto-image generation. In Advances in Neural Information Processing Systems, pages 15903–15935. Curran Associates, Inc., 2023. 6

- [38] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025. 1, 3, 6, 7
- [39] Kai Yang, Jian Tao, Jiafei Lyu, Chunjiang Ge, Jiaxin Chen, Weihan Shen, Xiaolong Zhu, and Xiu Li. Using human feedback to fine-tune diffusion models without any reward model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8941–8951, 2024. 2
- [40] Yujie Zhou, Pengyang Ling, Jiazi Bu, Yibin Wang, Yuhang Zang, Jiaqi Wang, Li Niu, and Guangtao Zhai. G2rpo: Granular grpo for precise reward in flow models. arXiv preprint arXiv:2510.01982, 2025. 1, 3, 6, 7

## E-GRPO: High Entropy Steps Drive Effective Reinforcement Learning for Flow Models

### Supplementary Material

#### 6. Ablation Study on the Entropy Threshold τ

In our main paper, we introduce an adaptive entropy threshold τ to separate timesteps into high-entropy and lowentropy groups. During sampling, consecutive low-entropy steps are merged until their entropy reaches the threshold. This threshold serves as a critical hyperparameter in our entropy-driven step-merging strategy. To claim the effectiveness of the proposed method and assess its sensitivity to τ, we conducted a series of experiments with different threshold values. Specifically, we trained E-GRPO with τ set to 0 (meaning all steps are treated equally and no merging ooccurs), 1.8, 2.0, 2.2 (our default setting), and 2.6 under the HPS reward configuration. The results are summarized in Table 4.

As shown in Table 4, the model behaves noticeably differently under varying threshold values. As τ increases, the achievable HPS score also improves, indicating the effectiveness of entropy as a guidance signal during training. However, when τ becomes excessively large, a long sequence of steps may be merged, occasionally combining steps that still contain useful entropy or gradient information. This leads to overly coarse updates and, consequently, a slight degradation in performance. Notably, our default choice of τ = 2.2 strikes an effective balance between leveraging entropy for guidance and avoiding excessive merging, yielding the best overall performance in our experiments.

#### 7. Additional Visualizations 7.1. More Quality Results

To further demonstrate the superiority of our proposed EGRPO, we provide additional qualitative comparisons with

Table 4. Ablation study on the entropy threshold τ. Results are reported under the HPS reward setting. A threshold of τ = 0 corresponds to the baseline without step merging. Our default choice (τ = 2.2) achieves the overall best performance. Best results in each column are highlighted in bold.

Threshold (τ) HPS CLIP PickScore ImageScore 0 (No Merging) 0.384 0.349 0.230 1.297

- 1.8 0.383 0.352 0.232 1.293
- 2.0 0.384 0.344 0.231 1.269 2.2 (Ours) 0.391 0.355 0.233 1.324 2.6 0.388 0.355 0.233 1.320

baseline methods in Figure 6 and Figure 7. As illustrated in these figures, E-GRPO consistently produces results that are more faithful to the text prompts. For example, under the prompt “ An award-winning portrait of a lemon in a muted, space age style reminiscent of the 1930s.” EGRPO successfully generates a portrait that combines a space-age aesthetic with the intended compositional structure. Likewise, for the prompt “A lot of buildings on each side of the road, with a very curvy road in the middle.” our method captures the “curvy” characteristic more accurately and achieves higher aesthetic quality compared with baseline methods. These results further validate that by focusing on high-entropy steps, E-GRPO enables more effective exploration and better alignment with complex human preferences.

##### 7.2. Failure Cases

Despite the robustness of E-GRPO, we observe several recurring failure patterns when handling challenging prompts. Reward Hacking. As discussed in the main paper, using only the HPS reward tends to produce overly saturated images, making the CLIP reward necessary as a counterbalance. Nevertheless, reward hacking still occurs in some cases. For instance, in the prompts shown in Figure 8, such as “A jellyfish sleeping in a space station pod. ” and “The image depicts alien flowers and plants surrounded by visceral exoskeletal formations in front of mythical mountains with dramatic contrast lighting, created with surreal hyper detailing in a 3D render. ”, the model occasionally introduces human faces or humanoid shapes that should not be present. These artifacts reflect the model’s tendency to exploit biases in the reward models, a limitation that is common across many RL-based training frameworks. Improving reward model reliability will be crucial for advancing RL in visual generation.

Overall, these observations highlight several key challenges faced by RL-based visual generation systems. Future research may explore solutions guided by these identified limitations.

[Figure 11]

###### Figure 6. Additional visualization comparisons between E-GRPO and other baseline methods.

[Figure 12]

###### Figure 7. Additional visualization comparisons between E-GRPO and other baseline methods.

[Figure 13]

###### Figure 8. Failure cases of E-GRPO

