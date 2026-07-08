## REASONING WITH EXPLORATION: AN ENTROPY PERSPECTIVE

### Daixuan Chengα,β,ε† Shaohan Huangβ† Xuekai Zhuγ† Bo Daiε Wayne Xin Zhaoα‡ Zhenliang Zhangε‡ Furu Weiβ αRUC βMSRA γSJTU εBIGAI

ABSTRACT

Balancing exploration and exploitation is a central goal in reinforcement learning (RL). Despite recent advances in enhancing large language model (LLM) reasoning, most methods lean toward exploitation, and increasingly encounter performance plateaus. In this work, we revisit entropy—a signal of exploration in RL—and examine its relationship to exploratory reasoning in LLMs. Through empirical analysis, we uncover positive correlations between high-entropy regions and three types of exploratory reasoning actions: (1) pivotal tokens that determine or connect logical steps, (2) reflective actions such as self-verification and correction, and (3) rare behaviors under-explored by the base LLMs. Motivated by this, we introduce a minimal modification to standard RL with only one line of code: augmenting the advantage function with an entropy-based term. Unlike traditional maximum-entropy methods which encourage exploration by promoting uncertainty, we encourage exploration by promoting longer and deeper reasoning chains. Notably, our method achieves significant gains on the Pass@K metric—an upper-bound estimator of LLM reasoning capabilities—even when evaluated with extremely large K values, pushing the boundaries of LLM reasoning.

# arXiv:2506.14758v4[cs.CL]8Nov2025

###### PPO/GRPO

|%|

%! &" ',&#" %!!"# &" ',&#"

%! &" ',&#" %!!"# &" ',&#"

!min

)",clip

,1 − /,1 + / )"

"&'

###### PPO/GRPO with Entropy Advantage

|%|

%! &" ',&#" %!!"# &" ',&#"

%! &" ',&#" %!!"# &" ',&#"

!min

(*"+,(ℋ")),clip

,1 − 4,1 + 4 (*"+,(ℋ"))

"&'

###### Pass@K Performance

[Figure 1]

[Figure 2]

GRPO w/ Entropy Advantage

PPO w/ Entropy Advantage

AIME2025

###### GRPO

###### PPO

Number of Samples K

Figure 1: Top: We augment the advantage in PPO (Schulman et al., 2017b) or GRPO (Shao et al., 2024) with a minimal per-token entropy-based term. Bottom: Our entropy-based advantage effectively encourages exploratory reasoning in LLMs, achieving superior Pass@K performance even with extremely large K values.

Email: daixuancheng6@gmail.com †Core Contributors.‡ Corresponding Authors.

- 1 INTRODUCTION

Recent reinforcement learning methods for large language models (LLMs), particularly those using verifiable rewards (RLVR; Lambert et al., 2024), typically rely on signals that reflect output accuracy to guide training. These approaches have proven effective in enhancing reasoning by reinforcing correct outputs and discouraging incorrect ones (Guo et al., 2025). However, as training progresses under purely accuracy-driven objectives, these benefits often diminish. LLMs tend to converge on narrow and over-optimized behaviors, gradually losing their incentive to explore alternative strategies. This lack of exploration weakens the model’s capacity for sustained, multi-step reasoning, causing performance to plateau or even regress, especially in complex or underspecified settings (Yu et al., 2025; Cui et al., 2025b).

In traditional RL, exploration plays a vital role alongside exploitation by encouraging the policy model to explore alternative strategies and avoid overfitting. A common metric for measuring exploration is entropy, which quantifies uncertainty in the policy’s action distribution (Haarnoja et al., 2018; Ziebart et al., 2008). Motivated by this, we investigate the relationship between entropy and exploratory reasoning in LLMs, and uncover strong correlations: (1) Pivotal tokens that guide or connect reasoning steps—such as first, because, and however—consistently exhibit higher entropy; (2) Reflective actions (Shah et al., 2025), such as self-verification and error correction, tend to emerge under high-entropy conditions; (3) During RL training, rare or under-explored solutions also coincide with elevated entropy. Together, these findings suggest entropy can be a valuable signal for recognizing exploratory reasoning behaviors in LLMs.

Based on these findings, we propose incorporating entropy as an auxiliary term to encourage exploratory reasoning during RL training. While traditional maximum entropy methods encourage exploration by promoting uncertainty (O’Donoghue et al., 2016), our approach takes a different path to balance exploration and exploitation: we introduce a clipped, gradient-detached entropy term into the advantage function of standard RL algorithms. Clipping ensures that the entropy term neither dominates nor reverses the sign of the original advantage, while gradient detachment preserves the original optimization direction. This design amplifies exploratory reasoning behaviors that emerge under uncertainty while maintaining the original policy gradient flow. Moreover, because of the intrinsic tension between entropy and confidence, the entropy-based term naturally diminishes as confidence increases—encouraging exploration in early stages while avoiding over-encouragement as training progresses. Furthermore, our method is extremely simple, requiring only one line of code to seamlessly integrate into existing RLVR training pipelines (Sheng et al., 2024).

We validate our method on mainstream RLVR algorithms, GRPO (Shao et al., 2024) and PPO (Schulman et al., 2017b), and observe distinct benefits. First, it amplifies exploratory reasoning behaviors—such as the use of pivotal tokens and reflective actions—by decreasing the policy’s uncertainty at these decision points. Second, it encourages the generation of longer, more exploratory responses without increasing the repetition rate, enabling coherent multi-step reasoning. Consequently, our method consistently improves Pass@1 accuracy across different benchmarks. We further increase the number of attempts K per question to evaluate Pass@K—a metric recently regarded as an upper-bound estimator of reasoning capability (Yue et al., 2025a). As shown in Figure 1, our method achieves substantial improvements even at large K, pushing the boundaries of LLM reasoning. In summary, the key contributions of this work are as follows:

- • We investigate and reveal a strong correlation between entropy and exploratory reasoning in LLMs, showing that pivotal tokens, reflective actions, and rare behaviors emerge with higher entropy.
- • We propose a minimal yet effective method that augments the standard RL advantage with a clipped, gradient-detached entropy term, encouraging exploration by fostering longer and deeper reasoning chains while preserving the original policy optimization direction.
- • We validate our method on mainstream RLVR algorithms: GRPO and PPO, achieving consistent gains on Pass@1 and substantial improvements on Pass@K, pushing the boundaries of reasoning.

- 2 PRELIMINARY ANALYSIS: ENTROPY AND EXPLORATORY REASONING

We examine entropy—a core signal of exploration in RL (Schulman et al., 2017a; Haarnoja et al., 2018; Nachum et al., 2017)—and its relationship with exploratory reasoning in LLMs. We begin by

###### Entropy Visualization: Tokens in Bold Show Higher Entropy Entropy Comparison: Exploratory Reasoning vs. Others

###### Rare

Nowwesolveasystemofequationswithlogarithms: !"#! $! = !"#" &#" = 10 First, let'ssimplifyeachequationthroughtheapplicationoflogarithmic identities:

|Pivotal Token<br><br>logical connectors: First, Because…|
|---|

|0.4<br><br>0.6<br><br>0.8<br><br>1.0<br><br>1.2 Pivotal token<br><br><br>Other token| |0.4<br><br>0.6<br><br>0.8<br><br>1.0<br><br>1.2 Reflective action<br><br><br>Other action| |0.2<br><br>0.4<br><br>0.6<br><br>0.8 behavior<br><br>Other behavior|
|---|---|---|---|---|

Avg.Entropy

1. !"#! $! = & !"#! $ = 10 2. !"#" &#" = 4!"#" & = 10 Thisconvertsourlogarithmic system intothefollowingsystem oflinear equations:… Recall that!"#$ + = &'(%

|Reflection<br><br>self-verification and correction|
|---|

!($),hence werewritethesystem:… From the first equation: & = 10, … Substitute , = ," + into& = 10,:…Thus,&$ = 25. Let’s verify if this is correct usingPython:… Answer:25

|Rare behavior<br><br>under-explored solutions|
|---|

Figure 2: Entropy Visualization and Comparison between Exploratory Reasoning and Others. We categorize tokens/actions/behaviors based on their role in the reasoning process. In the visualization, tokens with higher entropy appear in bold and larger sizes, with colors denoting different reasoning roles. In the comparison, we show average entropy values across different categories.

visualizing token-level entropy in the responses of Qwen2.5-Base-7B (Yang et al., 2024a) on mathematical reasoning tasks (MAA, 2025). As shown in Figure 2, we observe that high-entropy tokens consistently correspond to different reasoning dynamics compared to low-entropy ones. Based on this observation, we categorize exploratory reasoning-related content—including both tokens and sentences—to support the following analysis1.

Pivotal Tokens Figure 2 shows that pivotal reasoning tokens (e.g., first, recall, thus) tend to have higher entropy. These tokens serve as logical connectors, marking decision points where the model determines the flow and structure of reasoning. To quantify this observation, we compute the average entropy of commonly occurring pivotal tokens across all responses and compare it to that of the remaining tokens. These include causal terms (e.g., because, therefore), contrastive markers (e.g., however, although), sequential terms (e.g., first, then), and reasoning verbs (e.g., suggest, demonstrate). Results on the right of Figure 2 confirm a statistically significant increase in entropy for these pivotal tokens. Similar observations have also been noted in concurrent work (Wang et al., 2025a; Qian et al., 2025), where such tokens are referred to as forking tokens or information peaks.

Reflective Actions Reflection is a form of meta-cognition that involves examining generated information, evaluating the underlying reasoning, and adapting future behavior accordingly (Shah et al., 2025). In this work, we focus on self-reflection, where the model assesses and comments on its own outputs. This is illustrated in the visualization in Figure 2, where the LLM assigns higher entropy to sentences such as “Let’s verify if this is correct...”.

[Figure 3]

Base RL Other RL Rare

To quantify this behavior, we segment each response into sentences, compute the average entropy for each one, and use regular expressions to identify reflective actions—specifically, sentences containing keywords such as “verify” or “check”. As shown in the comparison in Figure 2, these reflective sentences consistently exhibit higher average entropy, suggesting that self-reflection tends to occur under greater uncertainty. To the best of our knowledge, this is the first analysis linking entropy to self-reflection in LLMs.

t-SNE2

t-SNE1

Rare Behaviors Emergent During RL We further examine whether under-explored or emergent behaviors—those rarely exhibited by the base model—are associated with distinct entropy patterns during RL. In the visualization (Figure 2), such behaviors include

Figure 3: Behavior Clustering. t-SNE projection of response embeddings. Base denotes the pre-RL model outputs; RL Other and RL Rare represent common and rare behaviors after RL, respectively.

1We also conduct a parallel analysis using a fine-tuned LLM—DeepSeek-R1-Distill-Qwen-1.5B (Guo et al., 2025)—on code reasoning tasks, yielding consistent observations (see Appendix A).

converting logarithmic systems into systems of linear equations, which are less frequently observed in the base model’s outputs. To quantify this, we perform RL training on the base model (see Section 4 for configurations), and define rare behaviors as sentences that semantically isolated from the base model’s output distribution. We embed all response sentences using SBERT (Reimers & Gurevych, 2019) and, for each RL-generated sentence, compute the average distance to its k = 5 nearest base-model neighbors. Sentences in the top 10% of this distance metric are labeled as rare. Behavior clusters are visualized in Figure 3. As shown in the comparison in Figure 2, these rare behaviors exhibit higher entropy, revealing a strong correlation between semantic novelty and predictive uncertainty.

- 3 METHOD

Our analysis reveals a strong correlation between entropy and exploratory reasoning in LLMs, motivating us to actively encourage high-entropy actions during training. To this end, we propose an advantage shaping method that augments the per-token advantage with a term based on its entropy. This entropy-based term serves as a robust, self-regulating signal that guides learning without altering the original gradient flow of the base RL algorithm.

Let q denote a question sampled from a dataset D, and let o = (o1,o2,...,o|o|) be the corresponding output response generated by a policy model πθ. Our method is compatible with mainstream policy optimization algorithms such as Proximal Policy Optimization (PPO; Schulman et al., 2017b) and Group Relative Policy Optimization (GRPO; Shao et al., 2024). We begin by briefly reviewing these methods before introducing our entropy-based advantage shaping method.

- 3.1 RL BASELINES: PPO AND GRPO

Proximal Policy Optimization (PPO) PPO optimizes the policy by maximizing the following clipped surrogate objective:

 

 

|o|

, (1)

JPPO(θ) = Eq∼D, o∼π

min[ρt(θ)At, clip(ρt(θ),1 − ϵlow,1 + ϵhigh)At]

θold(O|q)





t=1

θ(ot|q,o<t)

where ρt(θ) = π

πθold(ot|q,o<t) denotes the likelihood ratio between the current and old policy models, and At is the advantage, typically computed using Generalized Advantage Estimation (GAE; Schulman et al., 2015. We omit the length normalization term, as our implementation adopts a token-level policy loss without per-response normalization. The loss is averaged across all tokens in a training batch to mitigate implicit length bias (Liu et al., 2025; Zeng et al., 2025). The clipping range ϵlow and ϵhigh stabilizes policy updates by preventing excessively large changes. While standard PPO uses symmetric clipping (i.e., ϵlow = ϵhigh), recent work (Yu et al., 2025) suggests that slightly increasing ϵhigh can help avoid entropy collapse.

The gradient of the PPO objective is (we omit min and clip operations under the single-update-perrollout assumption (Shao et al., 2024)):

 . (2)

 

|o|

∇θJPPO(θ) = Eq∼D, o∼π

At∇θ log πθ(ot | q,o<t)

θold(O|q)

t=1

Group Relative Policy Optimization (GRPO) GRPO is an alternative to GAE-based PPO that avoids learning a separate value function by using the average reward of multiple sampled outputs, produced in response to the same question, as the baseline. Formally, for each question q, a group of G outputs {o1,o2,...,oG} is sampled from the old policy πθ

, a reward model is then used to score

old

the outputs, yielding G rewards {r1,r2,...,rG} correspondingly. These scores are then normalized as:

ri − mean({r1,r2,...,rG}) std({r1,r2,...,rG})

. (3)

r˜i =

Recently, GRPO has been widely used in outcome-supervised settings (Guo et al., 2025), where the normalized reward is assigned at the end of each output oi, and every token in oi receives the same advantage, i.e., Ai,t = r˜i . The policy is then optimized using the PPO objective in Equation 2 with these group-relative advantages. A KL penalty term between the trained policy and a reference policy may be added to the loss (Schulman, 2020).

- 3.2 ENCOURAGING EXPLORATORY REASONING VIA ENTROPY-BASED ADVANTAGE

Entropy-Based Advantage Shaping To encourage exploratory reasoning, we propose an entropy-guided advantage shaping method. The key idea is to inject an entropy-based term into the advantage function during policy optimization.

For each token ot in an output o, the entropy of the current policy over the vocabulary V is:

πθ(v | q,o<t)log πθ(v | q,o<t). (4)

Ht = −

v∈V

We then define an entropy-based advantage term ψ(Ht) and use it to shape the advantage:

ψ(Ht) = min α · Htdetach, |At|

, where α > 0 and κ > 1, (5)

κ

Ashapedt = At + ψ(Ht). (6) Here, α is the scaling coefficient, and κ controls the clipping threshold. This clipping ensures that the entropy-based term ψ(Ht) ≤ |A

t|

κ , so it does not dominate the advantage. Moreover, when At < 0, this constraint ensures that adding the entropy-based term does not reverse the sign of the advantage—thus preserving the original optimization direction. Crucially, the entropy term Htdetach is detached from the computational graph during backpropagation, acting as a fixed offset to the original advantage. This adjusts the magnitude of the update without altering the gradient flow. As a result, the policy gradient retains a format similar to that of PPO in Equation 2, where only the advantage At is replaced with the shaped one:

 

 . (7)

|o|

∇θJPPOshaped(θ) = Eq∼D, o∼π

(At + ψ(Ht))∇θ log πθ(ot | q,o<t)

θold(O|q)

t=1

Our shaping method can be seamlessly integrated into existing RL training pipelines using only a single line of code. Specifically, after computing the advantages with PPO or GRPO, we add the entropy-based advantage term before calculating the policy loss, as follows2:

### Entropy-Based Advantage Shaping (PyTorch Implementation)

# Compute advantages as in PPO or GRPO adv = compute advantages(...)

# Apply entropy-based term for advantage shaping adv += torch.min(alpha * entropy.detach(), adv.abs()/kappa) # Use the shaped advantages to compute the policy loss loss = compute policy loss(adv, ...)

Robustness of Entropy-Based Advantage: Avoiding Over-Encouragement Prior work (Chen et al., 2025) attempts to enhance reasoning by rewarding the policy based on the frequency of reasoning-like tokens, but this leads to reward hacking—the policy model repeatedly generates such tokens to exploit the reward without performing true reasoning. In contrast, our method naturally avoids such over-encouragement due to the intrinsic tension between entropy and confidence. As shown in Figure 4, our method initially assigns high advantage to tokens with high-entropy distributions but gradually reduces the entropy-based advantage as model confidence increases over training iterations.

2To be specific, this corresponds to a one-line code insertion in the update policy function of verl/workers/actor/dp actor.py file when using the veRL framework (Sheng et al., 2024).

Formally, let k denote the training iteration and t denote the token position within the output response. The policy model parameters are updated via gradient ascent:

θk+1 = θk + η ∇θJ (θk), (8) where η is the learning rate, and the policy gradient ∇θJ (θk) (Equation 7) uses the shaped advantage Ashapedk,t = Ak,t + ψ(Hk,t) which is positively correlated with the detached entropy Hk,tdetach (Equation 5). When the original advantage Ak,t > 0, higher entropy leads to a stronger update on the selected token ot, largely increasing its likelihood πθ(ot | ·) and thus sharpening the output distribution. According to the entropy definition in Equation 4, a sharper distribution lowers entropy, which in turn reduces the entropybased advantage ψ(Ht) and weakens subsequent updates. This self-regulating effect is empirically validated in Figure 7.

High Entropy ℋ! ↑

Probability

[Figure 4]

High #(ℋ!) ↑

!!Vocabulary

Policy Update

Boost & '! ( Confidence ↑

Probability

[Figure 5]

Entropy ℋ! ↓

#(ℋ!) ↓

!! Vocabulary

Figure 4: Dynamics of Entropy-Based Advantage. High entropy initially largely amplifies the advantage, accelerating confidence gain and leading to reduced entropy-based shaping in subsequent steps.

Comparison with Entropy Regularization In traditional RL, it is common to add an entropy regularizer to the gradient to prevent the policy from becoming overly deterministic (O’Donoghue et al., 2016). Practically, this means adding an entropy loss term to the policy loss. To clarify the distinction between our method and entropy regularization, we present a comparison in Table 1.

Entropy regularization explicitly adds an entropy term t Ht to the objective, scaled by a coefficient β. Since Ht depends on the current policy πθ, this introduces an additional gradient component ∇θHt, encouraging higher-entropy policies during training.

In contrast, our method modifies the advantage function by adding a clipped entropy term Htdetach, which is detached from the computation graph. As a result, ∇θHtdetach = 0, and the entropy term influences optimization only through the adjusted advantage values. Thus, our method preserves the original RL optimization dynamics. This makes it fundamentally distinct from—and even orthogonal to—entropy regularization.

Entropy Regularization Entropy-Based Adv. Shaping Training Objective J = JPPO + β t Ht J = JPPO(Ashapedt ) Policy Gradient t At∇θ log πθ(ot) + β t ∇θHt t Ashapedt ∇θ log πθ(ot) Entropy Gradient Flow ∇θHt ̸= 0 ∇θHtdetach = 0

- Table 1: Comparison of gradient behavior between entropy regularization and our entropy-based advantage shaping. We present simplified expressions that omit PPO’s min/clip operations and batch normalization. JPPO(Ashapedt ) denotes the PPO objective computed with shaped advantages.

- 4 EXPERIMENT SETTINGS

Backbone Models We conduct experiments on two base models: the general-purpose Qwen2.5Base-7B (Yang et al., 2024a) and its domain-adapted variant Qwen2.5-Math-Base-7B (Yang et al., 2024b). We also initially attempted RL training from Llama-series LLMs (AI@Meta, 2024) using vanilla GRPO or PPO, but observed that the LLMs abandoned intermediate reasoning chains within just a few training iterations. This observation aligns with Gandhi et al. (2025) that Llama LLMs inherently lack reasoning behaviors and likely require pre-training on reasoning traces prior to RL training.

RL Training Configuration Our training data are sourced from DAPO (Yu et al., 2025). We use output reward that assigns +1 for correct final answers and -1 otherwise. We conduct experiments

on GRPO and PPO using the veRL framework (Sheng et al., 2024). To build strong baselines, we adopt several techniques from DAPO and VAPO (Yue et al., 2025b), including Clip-Higher, Tokenlevel Loss, Critic-Pretraining, and Group-Sampling. Detailed hyperparameters are in Appendix B. Building on these RL baselines, we apply our proposed entropy-based advantage. We fix κ to 2 throughout all experiments, and set α to 0.4 for GRPO and 0.1 for PPO.

Evaluation Benchmarks and Metrics We evaluate on AIME 2025/2024 (MAA, 2025), AMC 2023 (MAA, 2023) and MATH500 (Hendrycks et al., 2021), using a rollout temperature of 0.6, a maximum response length of 8K tokens, and top-p sampling with p = 0.95. Each dataset is evaluated multiple times, and we report the average Pass@1 accuracy. Following Yue et al. (2025a), we also assess reasoning ability boundaries using the Pass@K metric: for each question, if at least one of K sampled model outputs passes verification, Pass@K = 1; otherwise 0. To mitigate variance, we adopt the unbiased estimation method proposed by Chen et al. (2021). For the small and challenging benchmark AIME 2024/2025 (30 examples per year), we scale K to a large value of 256. For the larger and less challenging benchmarks AMC 2023 (83 examples) and MATH500 (500 examples), we set K = 128 and K = 16, respectively, because LLMs already achieve near-perfect results with small K, and their size makes large K computationally expensive.

- 5 RESULTS

As shown in Table 2, our method consistently outperforms the baselines across benchmarks and RL algorithms, achieving superior average performance even compared to strong existing approaches (Cui et al., 2025a; Liu et al., 2025; Chu et al., 2025). Moreover, this advantage extends to Pass@K—a metric for estimating the reasoning capacity of LLMs. As shown in Figure 5, our method continues to deliver improvements even at large K values, where most baselines plateau.

On benchmarks such as AIME2024, AMC2023, and MATH500, we observe a similar phenomenon reported in Yue et al. (2025a): although RL-trained models consistently outperform their base models in terms of average Pass@1 performance, the base models can surpass RL-finetuned ones in Pass@K when K becomes sufficiently large. This indicates that conventional RL fine-tuning may inadvertently limit the exploratory capacity of the model. Our method effectively mitigates this issue. Notably, on AIME2025—the most challenging benchmark in our evaluation, released after the training data cutoff of the base models—our method not only outperforms the RL baselines but also exceeds the performance ceiling of the base model. This highlights the potential of our approach to break through the inherent limitations of base models and push the boundaries of LLM reasoning.

AIME25 AIME24 AMC23 MATH500

Pass@256 Pass@1 Pass@256 Pass@1 Pass@128 Pass@1 Pass@16 Pass@1

Qwen2.5-Base 50.0 2.2 66.7 5.2 90.4 28.3 88.8 54.4 + GRPO 50.0 10.7 46.7 11.9 91.6 55.6 65.4 55.3 + GRPO w/ Entropy Adv. 53.3 11.8 56.7 12.6 91.6 57.8 74.0 58.5 ∆ +3.3 +1.1 +10.0 +0.7 +0.0 +2.2 +8.6 +3.2

+ PPO 43.3 7.9 46.7 14.2 85.5 51.8 68.4 57.9 + PPO w/ Entropy Adv. 56.7 11.7 50.0 16.8 88.0 56.1 75.2 60.9 ∆ +13.4 +3.8 +3.3 +2.6 +2.5 +4.3 +6.8 +3.0

Qwen2.5-Math 50.7 4.4 70.0 10.7 90 34.4 88.6 47.5 Qwen2.5-Math-Ins† - - - 13.3 - 50.6 - 79.8 Eurus-2-PRIME - - - 26.7 - 57.8 - 79.2 Oat-Zero† - - - 30.0 - 55.4 - 80.6 GPG† - - - 33.3 - 65.0 - 80.0

+ GRPO 57.4 16.3 83.3 30.9 92.8 66.9 94.6 83.0 + GRPO w/ Entropy Adv. 63.6 17.6 80.0 33.7 95.2 69.8 94.8 83.1 ∆ +6.2 +1.3 -3.3 +2.8 +2.4 +2.9 +0.2 + 0.1

- Table 2: Pass@K and Pass@1 Performance. †: results from Chu et al. (2025). “+ GRPO” and “+ PPO” indicate RL training from the base models, while “w/ Entropy Adv.” denotes applying our entropy-based advantage to the corresponding RL algorithms. ∆ denotes the performance difference between without and with applying our method.

AIME25 AIME24 AMC23 MATH500

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Qwen2.5-Base,GRPOQwen2.5-Math,GRPOQwen2.5-Base,PPO

RL RL w/ Entropy Advantage

Number of Samples K

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Figure 5: Pass@K Performance of the LLMs with different RL algorithms.

- 6 ANALYSIS

We conduct a detailed analysis to understand the impact of our method on RL training for LLMs. Specifically, we track key metrics—including reward, response length, entropy, and our entropybased advantage—throughout training (Figure 6 and 7), as well as reasoning dynamics during testing (Figure 8 and 9). Furthermore, we provide a comprehensive comparison between our method and traditional entropy regularization.

- 6.1 RL TRAINING PROCESS

Training Reward As shown on the left of Figure 8, we observe steady upward trends across all three methods. Notably, RL with Entropy-based Advantage yields slightly higher rewards in the later stages of training, indicating a stronger and more sustained improvement over time.

Response Length Figure 8 (middle) indicates that while the response length of RL baseline shows a steady increase before 1000 steps, it slightly declines thereafter. In contrast, augmenting the RL baseline with our entropy-based advantage sustains the upward trend in response length beyond 1000 steps, surpassing both the RL baseline and RL with Entropy Regularizer. This may reflect stronger reasoning capabilities, as the LLM tends to spend more time (i.e., generate more tokens) to explore and reach the correct answer (Guo et al., 2025).

Overall Entropy Both the RL baseline and our method exhibit a decreasing trend throughout training, reflecting increasing model confidence. However, neither shows signs of entropy collapse. This is likely due to the use of the “clip-higher” technique in the RL baseline, which prevents the gradients of low-probability tokens from being clipped. Specifically, at step 2000, the entropy of the RL baseline is 0.34, and that of our method is 0.17. As a reference, in an ablation where “cliphigher” is removed, entropy drops to 0.03—a level typically considered as entropy collapse (Yu et al., 2025; Cui et al., 2025b).

In contrast, although adding an entropy regularizer to the RL training objective increases entropy during training, it shows a sudden spike after step 1500, indicating unstable optimization. The

###### Training Reward Response Length Entropy

[Figure 18]

[Figure 19]

[Figure 20]

#ResponseTokens(K)

Avg.Entropy

1.14

Reward

RL

0.34

RL w/ Entropy Reg.

0.17

0.03 (RL w/o Clip-Higher)

RL w/ Entropy Adv.

0 500 1000 1500 2000

0 500 1000 1500 2000

0 500 1000 1500 2000

Policy Update Step

Policy Update Step

Policy Update Step

Figure 6: Metrics during RL training. The RL baseline is GRPO; “RL w/ Entropy Reg.” applies entropy regularization; “RL w/ Entropy Adv.” applies entropy-based advantage shaping; “RL w/o Clip-Higher” removes the clip-higher technique from the RL baseline (i.e., ϵhigh = ϵlow = 0.2). “Entropy Adv. Ratio” denotes the ratio of entropy-based advantage to the original advantage.

AIME25 AIME24 AMC23 MATH500 Pass@256 Pass@1 Pass@256 Pass@1 Pass@128 Pass@1 Pass@16 Pass@1

RL w/ Entropy Reg. 50.0 9.3 50.0 16.0 90.4 54.3 70.4 57.4 RL w/ Entropy Adv. 53.3 11.8 56.7 12.6 91.6 57.8 74.0 58.5

- Table 3: Comparison of model performance trained with RL (i.e., GRPO) using entropy regularization vs. entropy-based advantage shaping

corresponding testing performance comparison between our method and entropy regularization is shown in Table 3, highlighting our method’s superiority in promoting stable training while improving reasoning performance.

Our method does not aim to increase token entropy uniformly. Instead, it promotes exploratory reasoning behaviors by amplifying high-entropy actions and gradually builds confidence (reflected by lower entropy) at these points. We further analyze this in the next subsection on exploratory reasoning dynamics.

[Figure 21]

EntropyAdv.Ratio(%)

Entropy-Based Advantage Figure 7 shows the ratio of the entropy-based advantage to the original advantage. As training progresses and the model gains confidence, the entropy-based advantage decreases. This supports our hypothesis that the intrinsic tension between model confidence and entropy naturally encourages exploration in uncertain regions, while gradually reducing the entropy-based advantage once sufficient confidence is achieved—thereby preventing over-encouragement.

0 500 1000 1500 2000

Policy Update Step

Figure 7: Ratio of Entropy-Based Advantage to the original advantage (i.e., ψ(H

t)

|At| ).

- 6.2 EXPLORATORY REASONING DYNAMICS

We further analyze the reasoning dynamics of the RL-trained models on the testing benchmarks to validate whether encouraging high-entropy actions during training can effectively enhance the model’s exploratory reasoning capabilities.

Pivotal Tokens and Reflective Actions As shown in Figure 8, applying our entropy-based advantage successfully reinforces the model’s ability to generate pivotal tokens and reflective actions. These regions exhibit much lower entropy, indicating increased model confidence when producing such actions. Consequently, we observe significantly higher counts of pivotal tokens and reflective actions compared to the baseline.

- 702
- 703
- 704
- 705
- 706
- 707
- 708
- 709
- 710
- 711
- 712
- 713
- 714
- 715
- 716
- 717
- 718
- 719
- 720
- 721
- 722
- 723
- 724
- 725
- 726
- 727
- 728
- 729
- 730
- 731
- 732
- 733
- 734
- 735
- 736
- 737
- 738
- 739
- 740
- 741
- 742
- 743
- 744
- 745
- 746
- 747
- 748
- 749
- 750
- 751
- 752
- 753
- 754
- 755

Response from RL w/ Bonus (Part 3) Total Length: 3023 Tokens Trying n = 8

If n = 8, the median is the average of a4 and a5. Since the median is a positive integer not in the list, a4+a5

2 →= 9. The list looks like: a1,a2,a3,9,9,9,a6,a7 The sum is:

a1 + a2 + a3 + 9 + 9 + 9 + a6 + a7 = 30

Since 9 is the mode, we need to balance the remaining values. Let’s try:

• a1 = a2 = a3 = 1:

1 + 1 + 1 + 9 + 9 + 9 + a6 + a7 = 30 3 + 27 + a6 + a7 = 30 30 + a6 + a7 = 30 a6 + a7 = 0 (not valid since a6 and a7 must be positive)

After testing several more combinations, let’s try n = 6 again with a different strategy:

Response from RL 725 Tokens To ﬁnd the sum of the squares of all items in the list that satisﬁes the given properties, let’s analyze the requirements step-by-step:

• Let’s try a1 = a2 = 1, a3 = 2, a4 = 9, and a5 and a6 need to be chosen such that the sum is 30 and 9 is the unique mode. The list looks like: 1,1,2,9,a5,a6 The sum is:

###### Pivotal Token

###### Reflective Action

###### Length and Repetition

1 + 1 + 2 + 9 + a + a = 30 13 +

|0.0<br><br>0.1<br><br>0.2<br><br>0.3<br><br>0.4<br><br>#ReflectiveActions(K)<br><br>5 6<br><br>+ a5<br><br>|+ a6<br><br>6 = 9.<br><br>+ 7|
|---|
<br><br>+ = 30 a5 + a 17<br><br>cannot be Let’s try<br><br>2 + 9 + 10 =<br><br>condition of all the items|
|---|

[Figure 22]

[Figure 23]

1.2

[Figure 24]

1. The sum of the integers in the list must be equal to

RL w/ Entropy Adv.

RepetitionRate(Dashed)

|9 is a5<br><br>Mode|
|---|

30. (...) Let’s guess the list might be something like:

#ResponseTokens(K)

#ReflectiveActions(K)

RL w/ Entropy Adv.

RL w/ Entropy Adv.

a Since

1.0

#PivotalTokens(K)

the unique mode, a5 and a6 can other values:

Avg.Entropy

Avg.Entropy

- • Suppose the list has an odd number of elements (...) Let’s try another conﬁguration:
- • Let’s try a list with sum 30: [1, 9, 9, 9, 7]: Sum is 30 correct.
- • Let’s check the median is not in the list and unique mode is 9 is unique correctly.
- • The ﬁnal valid list sum correctly would be: Let’s do correct summing and checking properties correctly: (...)

• If = 7 and a6 = 10:

0.7

1 + 1 + 30 This works. The list is: 1,1,2,9,7,10

RL

RL

0.5

- • M e

|is 9 is|
|---|

(unique)

- • Median 2+72 = 4.5 (not in list, integer satisﬁed)

RL

0.2

Now, we need to ﬁnd the sum of the squares in the list:

0 1000 2000 0 1000 2000 0 1000 2000

RL RL w/ Entropy Adv.

RL

RL w/ Entropy Adv.

Policy Update Step

Policy Update Step Policy Update Step

12 + 12 + 22 + 92 + 72 + 102 1 + 1 + 4 + 81 + 49 + 100 1 + 1 + 4 + 81 + 49 + 100 = 236 Answer:

Figure 8: Comparison of Pivotal Tokens (left), Reflection Actions (middle), and Response Length and Repetition (right) on the testing task between the baseline RL algorithm—GRPO (grey)—and RL with entropy advantage (purple/blue/orange), evaluated across training process.

One possible valid list correctly is: [1, 9, 9, 9, 7]: Sums correctly. Correct sum properties and correctly summed properties correctly would sum correctly.

|236|
|---|

Answer: 251

Response from RL w/ Entropy Adv. 3023 Tokens To solve this problem, we will systematically analyze the given conditions and construct the list of positive integers that meets all the criteria.

Question and Ground Truth Question: A list of positive integers has the following properties:

- • The sum of the items in the list is 30
- • The unique mode of the list is 9
- • The median of the list is a positive integer that does not appear in the list itself. Find the sum of the squares of all the items in the list. Ground Truth: 236

1. Sum of the items in the list is 30: (...) Step-by-Step Analysis:

- Step 1: Understanding the Mode (...)
- Step 2: Understanding the Median (...)
- Step 3: Setting Up the List (...) Case Analysis:

Response from RL 725 Tokens To ﬁnd the sum of the squares of all items in the list that satisﬁes the given properties, let’s analyze the requirements step-by-step:

- Case 1: n is odd: If n is odd, (...)
- Case 2: n is even If n is even, (...)

Constructing the List Let’s start by trying small values for n and see if we can satisfy all conditions.

1. The sum of the integers in the list must be equal to

30. (...) Let’s guess the list might be something like:

- Trying n = 5 If n = 5, the median is (...) Let’s try some values: (...)
- Trying n = 6 (...)
- Trying n = 7 (...)
- Trying n = 8 (...)

- • Suppose the list has an odd number of elements (...) Let’s try another conﬁguration:
- • Let’s try a list with sum 30: [1, 9, 9, 9, 7]: Sum is 30 correct.
- • Let’s check the median is not in the list and unique mode is 9 is unique correctly.
- • The ﬁnal valid list sum correctly would be: Let’s do correct summing and checking properties correctly: (...)

14

After testing several more combinations, let’s try n = 6 again with a different strategy: (...) This works. The list is: (...) Now, we need to ﬁnd the sum of the squares of all the items in the list: (...) Answer:

One possible valid list correctly is: [1, 9, 9, 9, 7]: Sums correctly. Correct sum properties and correctly summed properties correctly would sum correctly.

|236|
|---|

Answer: 251

- Figure 9: Response Comparison between RL-trained LLMs with and without entropy-based advantage shaping. Certain portions are omitted and the completion version is in Appendix C.

Response from RL w/ Entropy Adv. 3023 Tokens To solve this problem, we will systematically analyze the given conditions and construct the list of positive integers that meets all the criteria.

20

Response Length and Repetition Rate On the right side of Figure 8, we also observe a substantial increase in response length across testing benchmarks. Additionally, we record the n-gram-based repetition rate of generated responses and find that our method yields much longer responses while maintaining a repetition rate comparable to that of the RL baseline, demonstrating its ability to scale effectively at test time without increasing redundancy.

1. Sum of the items in the list is 30: (...) Step-by-Step Analysis:

- Step 1: Understanding the Mode (...)
- Step 2: Understanding the Median (...)
- Step 3: Setting Up the List (...) Case Analysis:

Case Study Figure 9 presents example responses from the RL-trained models. Compared to the baseline, our method produces more accurate and mathematically rigorous solutions. The model explicitly lists problem constraints, performs systematic case analysis (e.g., odd vs. even list lengths), and dynamically adjusts its approach when initial attempts fail. For instance, it iterates through candidate values (e.g., n = 5,6,...) while ensuring constraints are satisfied at each step. This structured

- Case 1: n is odd: If n is odd, (...)
- Case 2: n is even If n is even, (...)

Constructing the List Let’s start by trying small values for n and see if we can satisfy all conditions.

10

- Trying n = 5 If n = 5, the median is (...) Let’s try some values: (...)
- Trying n = 6 (...)
- Trying n = 7 (...)
- Trying n = 8 (...)

After testing several more combinations, let’s try n = 6 again with a different strategy: (...) This works. The list is: (...) Now, we need to ﬁnd the sum of the squares of all the items in the list: (...) Answer:

|236|
|---|

20

and persistent reasoning process leads to valid final answers, whereas the baseline often overlooks key conditions and produces incorrect solutions.

- 7 RELATED WORK

Exploration in Reinforcement Learning Exploration has long been a central theme in RL (Li et al., 2025), addressed through theoretical frameworks (Cai et al., 2020; Agarwal et al., 2020; Ishfaq et al., 2021), as well as empirical heuristics (Burda et al., 2019; Pathak et al., 2017; Raileanu & Rockt¨aschel, 2020; Henaff et al., 2022). Motivated by the use of entropy to guide exploration (Haarnoja et al., 2018; Schulman et al., 2017b; Ziebart et al., 2008), we investigate its role in LLM reasoning by treating entropy as an advantage-shaping signal to reinforce exploratory reasoning behaviors. A concurrent work (Gao et al., 2025) also studies exploration-driven reasoning but adopts a different approach by designing custom metrics rather than using entropy. Other concurrent studies incorporate an entropy regularizer (He et al., 2025; Wang et al., 2025b) to the training objective, while our method focuses on the advantage function, providing an orthogonal perspective.

Training Signals in Reinforcement Fine-Tuning Reinforcement fine-tuning of LLMs can leverage supervised and/or unsupervised training signals (Shao et al., 2025). Supervised methods, such as RLHF (Ouyang et al., 2022) and RLVR, rely on reward signals derived from human feedback or verifiable correctness, and have proven effective in aligning model behavior and solving deterministic tasks. In contrast, unsupervised approaches reduce dependence on human annotations by leveraging consistency-based signals (Prasad et al., 2024; Zuo et al., 2025) or entropy minimization (Zhang et al., 2025; Agarwal et al., 2025). Chen et al. (2025) incorporates uncertainty-aware weighting into RL using semantic entropy. Our work focuses on unsupervised signals with a specific emphasis on exploration, employing entropy to shape the advantage and encourage exploratory reasoning.

- 8 CONCLUSION

This work investigates reasoning with exploration to encourage longer and deeper reasoning chains in LLMs, through the lens of entropy. We begin by analyzing the relationship between entropy and exploratory reasoning, revealing that pivotal tokens, reflective actions, and rare behaviors consistently align with regions of higher entropy. Motivated by these findings, we introduce a minimal modification to standard RL algorithms by augmenting the advantage function with a clipped, gradient-detached entropy term. This design fosters exploration by promoting longer and deeper reasoning, while preserving the original policy optimization direction. We validate our method on mainstream RLVR algorithms, GRPO and PPO, and observe substantial improvements in Pass@K across diverse benchmarks—highlighting a promising direction for exploration-aware LLM training.

ACKNOWLEDGMENTS

The first author would like to thank Hongzhao Xie, Yexin Li, and Yuxian Gu for helpful discussions.

REFERENCES

Alekh Agarwal, Sham Kakade, Mikael Henaff, and Wen Sun. Pc-pg: Policy cover directed exploration for provable policy gradient learning. In Advances in Neural Information Processing Systems, 2020.

Shivam Agarwal, Zimin Zhang, Lifan Yuan, Jiawei Han, and Hao Peng. The unreasonable effectiveness of entropy minimization in llm reasoning. arXiv preprint arXiv:2505.15134, 2025.

AI@Meta. Llama 3 model card, 2024. URL https://github.com/meta-llama/llama3/ blob/main/MODEL_CARD.md.

Yuri Burda, Harrison Edwards, Amos Storkey, and Oleg Klimov. Exploration by random network distillation. In Proceedings of the 7th International Conference on Learning Representations, 2019.

Qi Cai, Zhuoran Yang, Chi Jin, and Zhaoran Wang. Provably efficient exploration in policy optimization. In Proceedings of the 37th International Conference on Machine Learning, 2020.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Zhipeng Chen, Yingqian Min, Beichen Zhang, Jie Chen, Jinhao Jiang, Daixuan Cheng, Wayne Xin Zhao, Zheng Liu, Xu Miao, Yang Lu, et al. An empirical study on eliciting and improving r1-like reasoning models. arXiv preprint arXiv:2503.04548, 2025.

Xiangxiang Chu, Hailang Huang, Xiao Zhang, Fei Wei, and Yong Wang. Gpg: A simple and strong reinforcement learning baseline for model reasoning. arXiv preprint arXiv:2504.02546, 2025.

Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, et al. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456, 2025a.

Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, et al. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025b.

Kanishk Gandhi, Ayush Chakravarthy, Anikait Singh, Nathan Lile, and Noah D Goodman. Cognitive behaviors that enable self-improving reasoners, or, four habits of highly effective stars. arXiv preprint arXiv:2503.01307, 2025.

Jingtong Gao, Ling Pan, Yejing Wang, Rui Zhong, Chi Lu, Qingpeng Cai, Peng Jiang, and Xiangyu Zhao. Navigate the unknown: Enhancing llm reasoning with intrinsic motivation guided exploration. arXiv preprint arXiv:2505.17621, 2025.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In Proceedings of the 35th International Conference on Machine Learning. PMLR, 2018.

Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, et al. Skywork open reasoner 1 technical report. arXiv preprint arXiv:2505.22312, 2025.

Mikael Henaff, Roberta Raileanu, Minqi Jiang, and Tim Rockt¨aschel. Exploration via elliptical episodic bonuses. In Advances in Neural Information Processing Systems, 2022.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

HuggingFace. Math-verify, 2025. URL https://github.com/huggingface/ Math-Verify.

Haque Ishfaq, Qiwen Cui, Viet Nguyen, Alex Ayoub, Yang Zhuoran, Zhaoran Wang, Doina Precup, and F. Lin Yang. Randomized exploration for reinforcement learning with general value function approximation. In Proceedings of the 38th International Conference on Machine Learning, 2021.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. CoRR, abs/2403.07974, 2024. URL https: //doi.org/10.48550/arXiv.2403.07974.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. T\” ulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.

Yexin Li, Pring Wong, Hanfang Zhang, Shuo Chen, and Siyuan Qi. Cae: Repurposing the critic as an explorer in deep reinforcement learning. arXiv preprint arXiv:2503.18980, 2025.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.

Michael Luo, Sijun Tan, Roy Huang, Ameen Patel, Alpay Ariyak, Qingyang Wu, Xiaoxiang Shi, Rachel Xin, Colin Cai, Maurice Weber, Ce Zhang, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepcoder: A fully open-source 14b coder at o3-mini level. https://github.com/ agentica-project/rllm, 2025. Notion Blog.

MAA. American mathematics competitions - amc. https://maa.org/, 2023. MAA. American invitational mathematics examination - aime. https://maa.org/, 2025. Ofir Nachum, Mohammad Norouzi, Kelvin Xu, and Dale Schuurmans. Bridging the gap between

value and policy based reinforcement learning. Advances in neural information processing systems, 30, 2017.

Brendan O’Donoghue, R´emi Munos, Koray Kavukcuoglu, and Volodymyr Mnih. Pgq: Combining policy gradient and q-learning. CoRR, 2016.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35: 27730–27744, 2022.

Deepak Pathak, Pulkit Agrawal, Alexei A. Efros, and Trevor Darrell. Curiosity-driven exploration by self-supervised prediction. In Proceedings of the 34th International Conference on Machine Learning. PMLR, 2017.

Archiki Prasad, Weizhe Yuan, Richard Yuanzhe Pang, Jing Xu, Maryam Fazel-Zarandi, Mohit Bansal, Sainbayar Sukhbaatar, Jason Weston, and Jane Yu. Self-consistency preference optimization. arXiv preprint arXiv:2411.04109, 2024.

Chen Qian, Dongrui Liu, Haochen Wen, Zhen Bai, Yong Liu, and Jing Shao. Demystifying reasoning dynamics with mutual information: Thinking tokens are information peaks in llm reasoning. arXiv preprint arXiv:2506.02867, 2025.

Roberta Raileanu and Tim Rockt¨aschel. Ride: Rewarding impact-driven exploration for procedurally-generated environments. In Proceedings of the 8th International Conference on Learning Representations, 2020.

Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bertnetworks. arXiv preprint arXiv:1908.10084, 2019.

John Schulman. Approximating kl divergence, 2020. URL http://joschu.net/blog/ kl-approx.html.

John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. Highdimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.

John Schulman, Xi Chen, and Pieter Abbeel. Equivalence between policy gradients and soft qlearning. arXiv preprint arXiv:1704.06440, 2017a.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv: 1707.06347v2, 2017b.

Darsh J Shah, Peter Rushton, Somanshu Singla, Mohit Parmar, Kurt Smith, Yash Vanjani, Ashish Vaswani, Adarsh Chaluvaraju, Andrew Hojel, Andrew Ma, et al. Rethinking reflection in pretraining. arXiv preprint arXiv:2504.04022, 2025.

Rulin Shao, Shuyue Stella Li, Rui Xin, Scott Geng, Yiping Wang, Sewoong Oh, Simon Shaolei Du, Nathan Lambert, Sewon Min, Ranjay Krishna, Yulia Tsvetkov, Hannaneh Hajishirzi, Pang Wei Koh, and Luke Zettlemoyer. Spurious rewards: Rethinking training signals in rlvr, 2025. URL https://arxiv.org/abs/2506.10947.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, et al. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. arXiv preprint arXiv:2506.01939, 2025a.

Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Lucas Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, et al. Reinforcement learning for reasoning in large language models with one training example. arXiv preprint arXiv:2504.20571, 2025b.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. CoRR, abs/2412.15115, 2024a.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren, and Zhenru Zhang. Qwen2.5-math technical report: Toward mathematical expert model via self-improvement. CoRR, abs/2409.12122, 2024b.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025a.

Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, et al. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks. arXiv preprint arXiv:2504.05118, 2025b.

Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerlzoo: Investigating and taming zero reinforcement learning for open base models in the wild. arXiv preprint arXiv:2503.18892, 2025.

Qingyang Zhang, Haitao Wu, Changqing Zhang, Peilin Zhao, and Yatao Bian. Right question is already half the answer: Fully unsupervised llm reasoning incentivization. arXiv preprint arXiv:2504.05812, 2025.

Brian D Ziebart, Andrew L Maas, J Andrew Bagnell, Anind K Dey, et al. Maximum entropy inverse reinforcement learning. In Aaai, volume 8, pp. 1433–1438. Chicago, IL, USA, 2008.

Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui, Xuekai Zhu, Haozhan Li, Yuchen Zhang, Xinwei Long, Ermo Hua, et al. Ttrl: Test-time reinforcement learning. arXiv preprint arXiv:2504.16084, 2025.

Minghan Chen, Guikun Chen, Wenguan Wang, and Yi Yang. Seed-grpo: Semantic entropy enhanced grpo for uncertainty-aware policy optimization. arXiv preprint arXiv:2505.12346, 2025.

- A EXTENDED PRELIMINARY ANALYSIS

Section 2 analyzes Qwen2.5-Base on mathematical domain. We further validate our findings using DeepSeek-R1-Distill-Qwen-1.5B on code domain; the evaluation benchmark is LiveCodeBench (Jain et al., 2024), and the RL training configuration follows DeepCoder (Luo et al., 2025). DeepSeek-R1-Distill-Qwen-1.5B is fine-tuned from Qwen2.5-Math-1.5B on large-scale, high-quality long-chain reasoning data distilled from DeepSeek-R1-671B, and already exhibits strong long-chain reasoning capabilities. Figure 10 shows consistent results: pivotal tokens, reflective actions, and rare behaviors exhibit higher entropy. This demonstrates that our entropy-related observations hold across domains, even with LLMs that already demonstrate strong reasoning performance.

|0.35<br><br>0.38<br><br>0.40<br><br>0.43<br><br>0.45 Pivotal<br><br>token<br><br>Other token|
|---|

|0.48<br>0.49<br>0.50<br>0.51<br>0.52 Rare behavior<br><br><br>Other behavior|
|---|

|0.1<br>0.2<br>0.3<br>0.4<br>0.5 action<br><br><br>Other action|
|---|

Reflective

Avg.Entropy

Figure 10: Entropy Comparison between Exploratory Reasoning and Others using DeepSeekR1-Distill-Qwen-1.5B on code domain.

- B EXPERIMENT SETTINGS

Training Data and Reward Function For Qwen2.5-Base-7B, we use DAPO’s original prompt template and verifier; for Qwen2.5-Math-Base-7B, we append the suffix \nPlease reason step by step, and put your final answer within \boxed{}.to each question and employ Math-Verify (HuggingFace, 2025) as the verifier.

RL Training Configuration For both GRPO and PPO, we use the hyperparameters in Table 4, without using entropy or KL losses. For PPO only, the critic is trained with a learning rate of 2e−6, with a 10-step pre-training applied before initiating policy updates. We use GAE with λ = 1.0 and γ = 1.0.

Hyperparameter Value Optimizer AdamW Policy learning rate 1e−6 Training batch size 512 prompts Samples per prompt 8 Mini-batch size 32 prompts Policy updates per rollout 16 Max response length 8K tokens Rollout temperature 1.0 Clip range ϵlow, ϵhigh 0.2, 0.28

Table 4: RL Hyperparameters

- C CASE STUDY

### Question and Ground Truth

Question: A list of positive integers has the following properties: (1) The sum of the items in the list is 30; (2) The unique mode of the list is 9; (3) The median of the list is a positive integer that does not appear in the list itself. Find the sum of the squares of all the items in the list.

### Ground Truth: 236

Response from RL Baseline Total Length: 725 Tokens To find the sum of the squares of all items in the list that satisfies the given properties, let’s analyze the requirements step-by-step:

- 1. The sum of the integers in the list must be equal to 30.
- 2. The unique mode of the list is 9.
- 3. The median of the integer list is a positive integer that does not appear in the list. Let’s denote the list as a1,a2,...,an where a1 ≤ a2 ≤ ··· ≤ an.

The mode of the list being unique and equal to 9 implies that 9 must appear the most frequently in the list; no other integer in the list appears as frequently as 9.

The median of the list is a positive integer not found in the list. Depending on whether the list has an odd or even number of elements, the median will be the middle element in the sorted list or the average of two middle elements, respectively.

Now let’s consider the possible list structures based on these properties. Since 9 must be the unique mode, 9 must appear more frequently than any other integer in the list. Let’s consider the possible sizes of the list and the appearances of 9.

Let’s guess the list might be something like:

- • Suppose the list has an odd number of elements, let n = 2k + 1. The median would be the element in the middle position.
- • Suppose the median is m, and m ̸= a1,a2,...,an. Let’s consider the list with these properties:
- • The list might include several 9’s. Let’s start by considering the minimal configurations:
- • Suppose the list has exactly three 9’s because then 9 appears with the highest frequency and other numbers must sum correctly. Thus the list will look like: 9, 9, 9, ...along with other integers summing up correctly.

We need to find a configuration where the sum is 30, and the median does not appear, unique mode is 9, and the median is integers in the list.

Let’s try constructing such a list:

- • Let’s say the list has a number of 9’s (let’s assume 3 × 9 = 27), then we need the rest to sum to 3 ≤ 3.
- • Let’s try constructing a specific list configuration manually. Suppose the list is; [1, 9, 9, 9, 7]: Sum is 1 + 9 + 9 + 9 + 7 = 35 which is not correct. Let’s try another configuration:
- • Let’s try a list with sum 30: [1, 9, 9, 9, 7]: Sum is 30 correct.
- • Let’s check the median is not in the list and unique mode is 9 is unique correctly.
- • The final valid list sum correctly would be: Let’s do correct summing and checking properties correctly: Sum correctly would be 30: One possible valid list correctly is: [1, 9, 9, 9, 7]: Sums correctly. Correct sum properties and correctly summed properties correctly would sum correctly. Answer: 251

To solve this problem, we will systematically analyze the given conditions and construct the list of positive integers that meets all the criteria.

- 1. Sum of the items in the list is 30: Sum = 30
- 2. The unique mode of the list is 9: Mode = 9 and 9 appears more frequently than any other number
- 3. The median of the list is a positive integer that does not appear in the list itself: Median is a positive integer and Median ∈/ list

Let’s start by denoting the list of positive integers as a1,a2,...,an where n is the number of elements in the list, and the list is sorted in ascending order (a1 ≤ a2 ≤ ··· ≤ an).

Step-by-Step Analysis:

- Step 1: Understanding the Mode

Since the mode is 9 and it is unique, 9 must appear more frequently than any other number in the list. Let’s say 9 appears k times. Therefore, k is the highest frequency of any number in the list.

- Step 2: Understanding the Median

The median is the middle value of the list when it is sorted. Since the median is a positive integer that does not appear in the list, it must be the average of the two middle numbers if n is even, or it must be the middle number if n is odd.

- Step 3: Setting Up the List

Let’s start by assuming the smallest possible values for the other numbers while maintaining the mode as 9.

### Case Analysis:

- Case 1: n is odd If n is odd, the median is the middle number an+1

2

. Since the median is a positive integer not in the list, an+1

2

̸= 9.

- Case 2: n is even

If n is even, the median is the average of the two middle numbers a

+a n 2

n 2

+1

2 . Since the median is a positive integer not in the list, a

+a n 2

n 2

+1

2 ̸= 9.

Constructing the List Let’s start by trying small values for n and see if we can satisfy all conditions.

### Trying n = 5

- If n = 5, the median is a3. Since a3 is a positive integer not in the list, a3 ̸= 9. The list looks like: a1,a2,9,a4,a5 The sum is:

a1 + a2 + 9 + a4 + a5 = 30

Since 9 is the mode, a1,a2,a4, and a5 cannot all be 9 (as 9 would not be the unique mode). Let’s try some values:

- • If a1 = a2 = a4 = a5 = 1 (smallest possible values):

- 1 + 1 + 9 + 1 + 1 = 13 (not equal to 30)

• If a1 = a2 = a4 = 1 and a5 = 2:

- 1 + 1 + 9 + 1 + 2 = 14 (not equal to 30)

#### • If a1 = a2 = 1 and a4 = a5 = 2:

- 1 + 1 + 9 + 2 + 2 = 15 (not equal to 30)

• If a1 = 1 and a2 = a4 = a5 = 2:

- 1 + 2 + 9 + 2 + 2 = 16 (not equal to 30)

We need to find a better combination. Let’s try increasing the values while maintaining the mode and unique mode properties.

### Trying n = 6

- If n = 6, the median is the average of a3 and a4. Since the median is a positive integer not in

- the list, a3+a4

2 ̸= 9. The list looks like: a1,a2,a3,9,9,a6 The sum is:

a1 + a2 + a3 + 9 + 9 + a6 = 30

Since 9 is the mode, a1,a2,a3, and a6 cannot all be different. Let’s try some values:

- • If a1 = a2 = 1 and a3 = a6 = 2:

- 1 + 1 + 2 + 9 + 9 + 2 = 24 (not equal to 30)

• If a1 = 1 and a2 = a3 = 2 and a6 = 3:

- 1 + 2 + 2 + 9 + 9 + 3 = 26 (not equal to 30)

• If a1 = 1 and a2 = 2 and a3 = a6 = 3:

- 1 + 2 + 3 + 9 + 9 + 3 = 27 (not equal to 30)

- • If a1 = 2 and a2 = 2 and a3 = a6 = 3: 2 + 2 + 3 + 9 + 9 + 3 = 28 (not equal to 30)
- • If a1 = 1 and a2 = 3 and a3 = 5 and a6 = 6: 1 + 3 + 5 + 9 + 9 + 6 = 33 (not equal to 30)

After testing several combinations, let’s try a different approach by starting with a known mode and ensuring the sum is met.

Trying n = 7

- If n = 7, the median is a4. Since the median is a positive integer not in the list, a4 ̸= 9. The list looks like: a1,a2,a3,9,9,9,a7 The sum is:

a1 + a2 + a3 + 9 + 9 + 9 + a7 = 30

Since 9 is the mode, we need to balance the remaining values. Let’s try:

- • a1 = a2 = a3 = 1: 1 + 1 + 1 + 9 + 9 + 9 + a7 = 30

30 + a7 = 30 a7 = 0 (not valid since a7 must be positive)

- • a1 = a2 = 1 and a3 = 2: 1 + 1 + 2 + 9 + 9 + 9 + a7 = 30

31 + a7 = 30 a7 = −1 (not valid since a7 must be positive) After testing several more combinations, let’s try a different approach by starting with a known mode and ensuring the sum is met while satisfying all conditions.

### Trying n = 8

- If n = 8, the median is the average of a4 and a5. Since the median is a positive integer not in

- the list, a4+a5

2 ̸= 9. The list looks like: a1,a2,a3,9,9,9,a6,a7 The sum is:

a1 + a2 + a3 + 9 + 9 + 9 + a6 + a7 = 30

Since 9 is the mode, we need to balance the remaining values. Let’s try:

• a1 = a2 = a3 = 1:

1 + 1 + 1 + 9 + 9 + 9 + a6 + a7 = 30 3 + 27 + a6 + a7 = 30 30 + a6 + a7 = 30 a6 + a7 = 0 (not valid since a6 and a7 must be positive)

After testing several more combinations, let’s try n = 6 again with a different strategy:

• Let’s try a1 = a2 = 1, a3 = 2, a4 = 9, and a5 and a6 need to be chosen such that the sum is 30 and 9 is the unique mode. The list looks like: 1,1,2,9,a5,a6 The sum is:

1 + 1 + 2 + 9 + a5 + a6 = 30 13 + a5 + a6 = 30 a5 + a6 = 17

Since 9 is the unique mode, a5 and a6 cannot be 9. Let’s try other values:

• If a5 = 7 and a6 = 10:

1 + 1 + 2 + 9 + 7 + 10 = 30 This works. The list is: 1,1,2,9,7,10

- • Mode is 9 (unique)
- • Median is 2+72 = 4.5 (not in list, integer condition satisfied)

Now, we need to find the sum of the squares of all the items in the list:

12 + 12 + 22 + 92 + 72 + 102 1 + 1 + 4 + 81 + 49 + 100 1 + 1 + 4 + 81 + 49 + 100 = 236 Answer:

|236|
|---|

