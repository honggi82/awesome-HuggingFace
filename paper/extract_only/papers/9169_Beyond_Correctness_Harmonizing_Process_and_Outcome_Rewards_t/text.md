## Beyond Correctness: Harmonizing Process and Outcome Rewards through RL Training

### Chenlu Ye1,2* Zhou Yu1 Ziji Zhang1 Hao Chen1 Narayanan Sadagopan1

# arXiv:2509.03403v2[cs.LG]15May2026

Jing Huang1 Tong Zhang2 Anurag Beniwal1 1Amazon 2University of Illinois Urbana-Champaign

### Abstract

Reinforcement Learning with Verifiable Rewards (RLVR) improves final-answer accuracy on reasoning tasks, but it does not reliably improve reasoning quality. Because outcome rewards only assess final answers, they also reward spurious successes: flawed reasoning can still receive maximal reward when it accidentally reaches the correct outcome. This outcome reward hacking creates biased gradients, making current RLVR insufficient for learning faithful reasoning. Process Reward Models (PRMs) provide step-wise supervision, but directly optimizing PRMs or naively combining them with outcome rewards is unstable under distribution shift during RL training process. We introduce PRocess cOnsistency Filter (PROF), a data curation method that uses PRM– ORM consistency for sample selection rather than direct reward optimization. PROF keeps correct responses with strong process support and incorrect responses with weak process support while maintaining a balanced training ratio. Experiments show that PROF consistently improves both final-answer accuracy and intermediate reasoning quality over strong baselines, with less dependence on strong PRMs. Codes and training recipes are available at https:// github.com/amazon-science/PROF-GRPO.

### 1 Introduction

Verifiable rewards have attracted substantial attention because they can reliably improve performance on reasoning tasks with easily verifiable outcomes, such as mathematical and coding problems (Cobbe et al., 2021; Jaech et al., 2024; Shao et al., 2024; Xiong et al., 2025b). However, success on these

*Email: chenluy3@illinois.edu

Problem: There are 4 coins of 1, 2, 3, 5 cruzeiros, which weigh 1, 2, 3, 5 grams respectively. One of them is counterfeit, differing in weight (but not known whether it is heavier or lighter). How can the counterfeit coin be identified with the minimum number of weighings using a balance scale without weights?

- Step 1: Introduction. We need to identify a counterfeit coin with an unknown weight difference from ...
- Step 2 - 6: The Flawed Weighing Logic and Code.

- 1. Weigh coins of 1 gram and 2 grams against coins of 3 grams and 5 grams. 2. Based on the result of the first weighing: - If the left side is heavier, then the

- counterfeit coin is either 1 gram or 3 grams. - If the right side is heavier, then the counterfeit coin is either

2 gram or 5 grams. - If both sides are equal, then the

- counterfeit coin is either 2 gram or 5 grams. 3. For the second weighing, ... Let’s illustrate this with code and ensure that the steps are correct ... output: (1, ’heavier’). Step 7, 8, 9: Summary and Final Answer. Here is a summary of the steps: 1. Weigh coins of 1 gram and 2 grams against coins of 3 grams and 5 grams... 2. For the second weighing, weigh the 1 gram coin against a known genuine coin (3 grams)... Thus, the minimum number of weighings required to

|2|
|---|

identify the counterfeit coin is

.

Table 1: An example of reaching a correct final answer through flawed reasoning. The first weighing ({1g,2g} vs. {3g,5g}) is invalid because the two sides are not equal in nominal weight, so all subsequent deductions are unsupported even though the final answer happens to be correct.

tasks is usually measured only by the final answer, while in many applications we also care about the quality of the reasoning process itself, especially its faithfulness, validity, and interpretability. Throughout this paper, we use reasoning quality as an umbrella term for these process-level properties. Optimizing the verifier is therefore not the same as optimizing reasoning quality. Because verifiers only assess final outcomes, Outcome Reward Models (ORMs) are too sparse and coarse to distinguish

flawed reasoning within correct answers or valid reasoning within incorrect answers. For instance, the training example in Table 1 has fundamentally invalid reasoning but still arrives at the correct answer. To theoretically analyze this challenge, we define a latent state variable z, where z = 1 denotes a valid intermediate reasoning process (i.e.,

no error). Let απ = P(z = 1|π) represent the probability of generating valid reasoning. Given that an incorrect process (z = 0) may coincidentally yield a correct answer (r = 1) with a small probability ϵ (i.e., P(r = 1|z = 0) ≤ ϵ), the expected reward can be decomposed as:

Eπ[r] =

P(r = 1|z)P(z|π) ≈ απ + ϵ(1 − απ)

z∈0,1

= (1 − ϵ)απ + ϵ.

While the ideal objective is to maximize απ, the term ϵ(1−απ) introduces gains from spurious successes. During training, samples where r = 1 despite z = 0 generate biased gradients that inadvertently reinforce flawed reasoning paths, allowing the policy to increase outcome reward without improving the latent reasoning quality. This creates a process-outcome mismatch: final-answer correctness no longer reliably reflects reasoning quality, especially the faithfulness of the underlying reasoning process. We refer to the resulting optimization failure as outcome reward hacking: the model is rewarded for exploiting weaknesses in outcome-only supervision rather than for producing faithful reasoning. This misalignment leads to unfaithful reasoning, a limitation increasingly observed in recent studies (Baker et al., 2025; Chen

- et al., 2025b). Consequently, relying solely on final answer accuracy is insufficient; ensuring reasoning quality, faithfulness, and interpretability in Chain of Thought (CoT) is crucial for the safety and practical utility of LLMs (Zhu et al., 2025; Lyu et al., 2023; Yeo et al., 2024). To empirically support this process-outcome mismatch and the resulting reasoning-quality gap, we later analyze 2k samples from Qwen2.5-Math-7B and find that 26.28% of correct responses still contain flawed reasoning, as judged by Claude. Within this flawed-correct subset, PROF identifies and filters 65.88% (Figure 1).

This process-outcome mismatch shows that current RLVR alone cannot solve the reasoning-quality gap: outcome rewards are necessary for verifiability, but insufficient for supervising how the answer is reached. This has motivated a flurry of recent work on training Process Reward Models (PRMs) and using them in RL training (Lightman et al., 2023; Zhang et al., 2025; Zou et al., 2025), since PRMs provide dense and fine-grained feedback

[Figure 1]

Figure 1: Within correct responses that contain flawed reasoning, PROF filters 65.88% and leaves 34.12%.

over intermediate reasoning processes. In other words, if we want to optimize reasoning quality rather than final correctness alone, some form of PRM-style process supervision is necessary. However, directly using PRMs as rewards introduces a second failure mode. Although these PRMs achieve excellent performance on PRM benchmarks, directly combining PRM and ORM in the reward function can lead to reward hacking. Notably, since PRMs are often trained offline, applying them to online training suffers from distribution shift. Especially in boundary cases where the policy encounters difficult problems and produces rarely seen responses, PRMs often fail to judge them correctly, thus leading to severe reward hacking when they are used as explicit reward signals during RL training (Michaud et al., 2020; Tien et al., 2022). Even when some works (Zha et al., 2025; Cui et al., 2025) attempt to co-train the policy and PRMs online, they can only train PRMs in implicit ways that lack accurate process scores, such as implicit generative rewards or alignment between process rewards and outcomes. Therefore, instead of training another PRM for a specific dataset or base model, we focus on how to robustly integrate a pre-trained PRM into online training, i.e., how to harmonize accurate but coarse-grained ORMs with fine-grained but noisy Process Reward Models (PRMs) in Reinforcement Learning (RL).

We develop a PRocess cOnsistency Filter (PROF) framework, an online data curation strategy based on process-outcome consistency. PROF oversamples responses at training time and then ranks and filters them by PRM–ORM consistency. Specifically, it removes samples where the process and outcome signals conflict, such as correct responses derived from flawed reasoning or incorrect responses that contain sound reasoning steps. By using PRMs for filtering rather than as direct

[Figure 2]

Figure 2: Visualization of Algorithm 1. Rectangle length denotes trajectory-level process score. PROF ranks correct and incorrect groups separately by PRM– ORM consistency: in the correct group, higherconsistency samples are kept; in the incorrect group, lower-consistency (or random) samples are kept to maintain a balanced correct/incorrect ratio.

optimization targets, PROF injects process supervision into RLVR while avoiding the instability of explicit PRM reward maximization. Furthermore, because correct and incorrect responses have different consistency distributions, we rank each group separately to maintain a balanced training ratio. PROF is a modular framework that can be combined with RL algorithms like Group Relative Policy Optimization (GRPO) for online training.

We conduct extensive experiments to validate the improvement of PROF on both outcome accuracy and reasoning quality using both Qwen (Yang et al., 2024) and LLaMA (Dubey et al., 2024) models. To summarize, we highlight our key contributions as follows:

- • We identify a fundamental reasoning-quality gap in current RLVR. Because outcome-only rewards can reward spurious successes, current RLVR can improve final-answer accuracy without reliably improving faithful reasoning, a failure mode we characterize as outcome reward hacking. We support this processoutcome mismatch with both theoretical analysis and empirical evidence.
- • We propose PROF, a consistency-based data curation framework that robustly injects PRM supervision into RLVR. Rather than directly optimizing PRM scores or naively blending PRM and ORM rewards, PROF uses PRM– ORM consistency for ranking and filtering, allowing it to remove conflicting trajectories while maintaining a balanced correct/incorrect training ratio.
- • Extensive experiments and ablations on both Qwen and LLaMA models show that PROF consistently improves both final-answer accuracy and intermediate reasoning quality over

strong baselines, with less dependence on strong PRMs. Under matched compute cost and matched rollout-group size, PROF still achieves larger gains by almost 2%. We further demonstrate robustness to different offthe-shelf PRMs, generality beyond GRPO, and the importance of filtering correct and incorrect responses separately.

### 2 Related Work

Reasoning-Quality Gaps and Faithfulness of Chain-of-Thought. A growing literature documents process-outcome mismatch in language models: final-answer correctness can diverge substantially from reasoning quality, especially from the faithfulness of a model’s verbalized reasoning. Turpin et al. (2023) show that CoT explanations can omit biasing features and rationalize incorrect predictions, while Lyu et al. (2023) argue that standard CoT does not guarantee a faithful explanation of how the answer is produced. Subsequent work measures this reasoning-quality gap more directly: Nguyen et al. (2024) report a significant disparity between answer accuracy and CoT faithfulness in multi-hop question answering, and Paul et al. (2024) use causal mediation analysis to show that LLMs do not reliably use their generated intermediate steps when producing the final answer. Beyond faithfulness, Yeo et al. (2024) advocate evaluating reasoning explanations along multiple axes including robustness and utility, and Jacovi et al. (2024) show that even dedicated verifiers struggle to detect logical errors and contradictions inside reasoning chains. Recent monitoring work extends these concerns to reasoning models themselves: Baker et al. (2025); Chen et al. (2025b) show that model-generated reasoning often fails to transparently reveal the cues or considerations that drive behavior. Relative to this line of work, we focus on the RLVR setting and study how process supervision can reduce process-outcome mismatch during online training by filtering trajectories whose final outcomes and reasoning quality are inconsistent.

Process-Supervised Reward Models for FineGrained Feedback. RLHF focuses on trajectorylevel comparison under the Bradley-Terry model. For reasoning-related tasks, Yang et al. (2024) uses final-answer correctness to construct preference pairs and trains Bradley-Terry reward models for mathematical reasoning. A more widely used approach, termed Outcome Reward Models (ORMs),

trains a classifier to predict whether the final answer is correct based on the reasoning history. However, Lightman et al. (2023) show that ProcessSupervised Reward Models (PRMs), which evaluate each intermediate step of a reasoning chain, significantly outperform ORMs, especially for data selection tasks such as best-of-n sampling (Lightman et al., 2023). Their approach, however, requires human annotators to label each intermediate step. Wang et al. (2023) proposes using Monte-Carlo estimation of the Q value to determine labels automatically. Many follow-up works improve PRMs through generative reward modeling, advanced training techniques such as RL, and refined engineering practices (Xiong et al., 2024b; Zhang et al., 2025; Khalifa et al., 2025; Zhao et al., 2025; Xiong

- et al., 2025c). Our work does not focus on improving PRMs themselves; instead, we use PRMs to supervise the intermediate steps of CoT trajectories for data filtering. We mainly use Qwen2.5Math-PRM-7B from Zhang et al. (2025) because it is trained on the Qwen distribution and achieves strong performance on ProcessBench (Zheng et al.,

- 2024).

Sample Filtering in Reinforcement Learning for LLM. A key challenge in applying reinforcement learning to LLM applications is the imperfection of reward signals. These signals stem from a learned reward model, such as Reinforcement Learning from Human Feedback (RLHF), or are sparse, delivered only at the end of a trajectory (e.g. RLVR). In RLHF, the reward model is trained on humanannotated pairwise comparisons, typically using a Bradley-Terry model (Bradley and Terry, 1952). Due to inherent human disagreement and finite training data, the model develops shortcuts that RL algorithms can exploit (Lin et al., 2023; Eisenstein

- et al., 2023) to chase for a fake high reward. Consequently, these rewards may not fully align with the underlying intended goals, leading to reward hacking.

Data filtering has proven effective in mitigating reward hacking across RL-based LLM training. In RLHF, prior work filters preference pairs by reward gap (Yuan et al., 2024; Dong et al., 2024; Xiong

- et al., 2024a; Zhang et al., 2024) or combines reward with response length (Kim et al., 2024; Yu
- et al., 2025a) to retain samples that are more reliable under the learned reward model.

Filtering is also useful in RLVR despite the reward being available only at the final outcome. Re-

jection sampling fine-tuning discards incorrect trajectories and often approaches stronger RL baselines (Dong et al., 2023; Chen et al., 2025a; Xiong et al., 2025a). Other methods filter prompts by difficulty (Yang et al., 2024), remove zero-gradient prompts via dynamic sampling (Yu et al., 2025b), or over-sample and retain subsets that improve reward variance or the balance between correct and incorrect responses (Xiong et al., 2025a; Xu et al., 2025). In contrast to these methods, which mainly rely on coarse outcome-level signals, our approach uses process-supervised reward models (PRMs) (Lightman et al., 2023) to filter trajectories based on reasoning quality at the level of intermediate steps and their consistency with ORMs.

### 3 Formulation and Algorithm

An LLM defines a policy distribution: given a prompt x, it assigns density π(a|x) to each response a. For mathematical reasoning tasks with a binary verifiable reward, there exists a verifier mapping prompt-response pairs (x,a) to a scalar reward ro(x,a) ∈ {−1,1}. For each prompt, we generate a group of responses together with their verifier outcomes, denoted by {(ai,rio)}Gi=1.

Algorithm 1 Process Consistency Filter (PROF)

- 1: Input: Number of rollouts n, policy update size m, roll-

outs {a1, . . . , an}, outcome rewards {r1o, . . . , rno}, step number regularization parameter λ, Hλ > 0.

- 2: Obtain process rewards for each rollout ai with Hi steps: (ri1, . . . , riHi) and compute trajectory-wise consistency

ripro =

1 Hi

Hi

h=1

rih − λI(Hi = 1 or Hi ≥ Hλ) · rio.

(1)

- 3: Divide rollouts into correct group G+ = {a+1 , . . . , a+n+}

with rio = 1 and incorrect group G− = {a−1 , . . . , a−n−} with rio = −1, where n+ + n− = n.

- 4: Compute kept number k+ ∈ [n+], k− ∈ [n−] in each group such that k+ + k− = m and k+k− is maximized.
- 5: Rank G+ and G− by rpro separately, and for the correct

group keep the samples K+ = {a+i |rank(rpro(a+i )) ≤ k+}. For the incorrect group, we have two sub-algorithms:

- • PROF-POS: randomly pick k− samples from G−;
- • PROF-BOTH: keep K− = {a−i |rank(rpro(a−i )) ≥ n− − k−}.

- 6: Output: The kept trajectories K+ ∪ K− with final kept size m.

GRPO. (Shao et al., 2024) proposes this policy gradient algorithm that simplifies the Proximal Policy Optimization (PPO) (Schulman et al., 2017) by only computing the advantage based on the out-

come rewards in a group. Instead of maintaining and updating another value network, GRPO computes the advantage by standardizing the outcome rewards within a group: for i = 1,...,n,

ro(x, ai) − mean {ro(x, aj)}nj=1

Ai =

,

std {ro(x, aj)}nj=1 + δ

where ro(x,ai) is the outcome reward for a given response and δ > 0 is a small constant for nu-

merical stability. Let ati denote the t-th token and a<ti denote (a1i,...,ati−1). This advantage is then incorporated into a clipped surrogate objective, which is optimized to update the policy from πθold to πθ:

|ai|

n

πθ(ati|x, a<ti ) πθold(ati|x, a<ti )

1 n

JGRPO(θ) =Ex∼D

min

Ai,

t=1

i=1

πθ(ati|x, a<ti ) πθold(ati|x, a<ti )

clip

, 1 − ϵ, 1 + ϵ Ai .

Although this approach stabilizes the online policy optimization and is efficient, the sparse reward signal limits further improvement in intermediate reasoning quality.

Process Reward Model (PRM). For a response a composed of multiple reasoning steps a = (a1,...,aH), we follow previous works (Zheng et al., 2024; Zhang et al., 2025; Zou et al., 2025) to use a newline as a sign for a new step. For each step ah, the PRM score rh maps it, the previous steps and the prompt (x,a≤h) to a scalar rh(x,a≤h), where we use the short-hand notation a≤h = (a1,...,ah).

Our Method PROF: Process Consistency Filter Framework We propose PROF in Algorithm 1 to robustly incorporate PRM–ORM consistency after the rollout phase, and also visualize it in Figure 2. First, we generate n samples and obtain outcome rewards. Then, we call the PRM to generate step-level rewards for each rollout and compute the trajectory-wise consistency score rpro by taking the mean over step-level rewards and adding a steplength regularization in equation 1, where λ is the regularization parameter and Hλ is the threshold for the penalized step number. This regularization ensures that samples with no step segments or over-long steps are discarded in the correct group. The samples are divided into two subgroups: G+ contains correct samples with ro = 1, and G− contains incorrect samples with ro = −1. Inspired by (Xu et al., 2025), the numbers to keep in each subgroup, k+ and k−, are chosen to maximize the outcome-reward variance of the final kept samples

k+k−/(k+ + k−)2. Since k+ + k− = m is fixed, k+k− = k+(m − k+) should be maximized, and the maximum is attained when k+ is closest to m/2 under the constraint k+ ≤ n+,k− ≤ n−. This implies that the ratio of correct and incorrect responses should be balanced. After that, for the correct group, we use rpro to rank and keep the top k+ samples. For the incorrect group, PROF-POS randomly filters samples, while PROF-BOTH uses rpro to rank and keep the bottom k− samples. Finally, we collect the kept m trajectories for policy update.

False Positives Are Frequent and Filterable. We provide empirical evidence in Figure 1 to justify the practical motivation. On 2k samples from Qwen2.5-Math-7B, we find that 26.28% of correct responses still exhibit flawed reasoning, as judged by Claude. Crucially, within this flawed-correct subset, when PROF filters the bottom half of correct responses by PRM consistency, it identifies and removes 65.88% of these flawed responses. This confirms that process-outcome mismatch is a critical bottleneck and that PROF effectively filters problematic samples to improve gradient quality.

### 4 Experiments

#### 4.1 Setup

We focus on mathematical reasoning tasks in this work. For online training, we use the Numina-Math prompt set (Beeching et al., 2024), which contains nearly 860k math problems with ground-truth answers ranging from Chinese high school exercises to US and international mathematics olympiad problems. We use Qwen2.5-Math-1.5B-base and Qwen2.5-Math-7B-base (Yang et al., 2024) as the training base models. For the PRM, we mainly use Qwen2.5-Math-PRM-7B (Zhang et al., 2025) to generate process rewards. We also experiment with a weaker PRM, Skywork-PRM-1.5B (He et al., 2024b), to study the robustness of PROF to PRM quality. More details are provided in Appendix A. Model performance is evaluated on five benchmarks: Math500 (Hendrycks et al., 2021), Minerva Math (Lewkowycz et al., 2022), Olympiad Bench (He et al., 2024a), AMC20231, and AIME20242. We mainly use average@16 for evaluation, i.e., accuracy averaged over 16 responses per prompt under temperature 1.0. The models are allowed to

- 1https://huggingface.co/datasets/math-ai/amc23
- 2https://huggingface.co/datasets/math-ai/aime24

generate 4096 tokens.

#### 4.2 Main Results

We summarize our main results in Table 2, where Blend denotes a common way that mixes the PRM with outcome rewards (Zha et al., 2025; Cui et al., 2025; Zou et al., 2025). Following (Zou et al.,

- 2025), the PRMs are averaged over steps for each response, weighted by a parameter β, and added to outcome rewards. We use parameter β = 0.8 according to Table 5 of (Zou et al., 2025). Our main findings are as follows.

PROF Improves Accuracy under Standard and Matched-Cost Comparisons. As shown in Table 2, our proposed methods, PROF-POS and PROF-BOTH, consistently outperform GRPO and Blend-PRM-GRPO across benchmarks and base models. For models starting from Qwen2.5-Math1.5B-base, PROF-POS and PROF-BOTH achieve average accuracies of 40.2% and 39.6%, surpassing the standard GRPO baseline (37.2%) and Blend-PRM-GRPO (35.3%). A similar trend is observed with Qwen2.5-Math-7B-base, where PROFPOS and PROF-BOTH achieve 50.6% and 51.7% average accuracies, significantly above GRPO’s 49.9% and Blend-PRM-GRPO’s 47.3%. Moreover, for LLaMA-3.2-3B-instruct, whose policy distribution differs from the Qwen family, Blend performs even worse than GRPO, while PROF-POS still outperforms the baseline by 1.8%. The learning dynamics in Figure 4 corroborate these findings, illustrating that PROF steadily maintains a consistent performance advantage over both GRPO and Blend-PRM-GRPO throughout training, with faster convergence and higher final accuracy than GRPO.

To further address efficiency and fairness concerns, we increase the rollout group size n and policy update group size m, and compare GRPOn16m16 with PROF-n16m8 on Qwen2.5-Math7B-base under matched compute cost. As shown in Figure 3, PROF achieves larger gains than GRPO at the same cost level. We compute average cost as Inference +3× Train + PRM, where the factor 3 is a rough FLOPs proxy for training relative to a forward pass. We further aggregate all five benchmarks by base-model pass rate into four difficulty levels: Level 1 (p > 0.5), Level 2 (0.25 < p ≤ 0.5), Level 3 (0 < p ≤ 0.25), and Level 4 (p = 0). PROF’s gain is especially pronounced on harder problems, plausibly because easier problems usually involve shorter and simpler

[Figure 3]

[Figure 4]

- Figure 3: Matched-cost comparison on Qwen2.5-Math7B-base: overall average accuracy averaged over 5 benchmarks (left) and Level-4 hardest bucket (right). Average cost is computed as Inference +3× Train + PRM.

[Figure 5]

[Figure 6]

- Figure 4: Learning dynamics from Qwen2.5-Math-1.5Bbase (left) and Qwen2.5-Math-7B-base (right), compared with GRPO and Blend-PRM-GRPO. The y-axis is average@16 accuracy further averaged over Math500, Minerva Math, and Olympiad Bench.

reasoning with fewer flaws, making improvements smaller and more sensitive to PRM noise, whereas harder problems rely much more on PRM’s ability to distinguish trajectory quality. Due to space constraints, Figure 3 visualizes only Level 4 in the main text, while matched-cost curves for Levels 1–3 are provided in Appendix Figure 8.

Filtration Method is Much More Robust than Blending. We plot the entropy loss and response length curves of GRPO, Blend-PRM-GRPO, and PROF in Figure 7. Blend-PRM-GRPO suffers from severe reward hacking because its entropy collapses quickly toward zero. Simultaneously, its response length in the right plot increases uncontrollably, indicating that the model has learned to game the PRM by over-generating verbose and repetitive steps to obtain a higher averaged process reward. As a result, Blend-PRM-GRPO’s test accuracy even falls below GRPO. In contrast, PROF maintains a gradual and slightly faster decrease in entropy loss together with controlled responselength growth. This illustrates that our filtration method effectively leverages the PRM signal while staying robust to reward hacking. Below, we further analyze the quality of intermediate reasoning steps.

4.3 PROF Improves Reasoning Process Quality

PROF Improves Reasoning Consistency. To evaluate the quality of intermediate steps, we adopt Monte Carlo (MC) estimation, a common way to

|Model Algorithm|Math500 Minerva Math Olympiad Bench AIME24 AMC23 Average<br><br>|
|---|---|
|Qwen2.5-Math1.5B-base<br><br>Base GRPO Blend PROF-POS PROF-BOTH<br><br>|39.9 11.4 19.1 3.5 23.6 19.5 70.3 29.1 33.0 9.0 44.5 37.2 67.6 27.8 31.1 7.7 42.5 35.3<br><br>72.6 31.3 36.1 10.6 50.3 40.2<br><br>73.2 30.0 36.1 9.6 49.1 39.6<br><br><br>|
|Qwen2.5-Math7B-base<br><br>Base GRPO Blend PROF-POS PROF-BOTH<br><br>|42.0 12.8 19.2 12.9 30.0 23.4<br><br>81.6 37.2 45.5 20.6 64.4 49.9<br><br>81.7 36.7 45.0 15.2 58.0 47.3 81.4 36.6 45.0 24.8 64.2 50.6 83.1 39.0 47.8 17.5 70.9 51.7<br><br><br>|
|LLaMA-3.23B-instruct t<br><br>Base GRPO Blend PROF-POS PROF-BOTH<br><br>|30.0 8.8 6.1 2.3 10.6 11.6 50.5 18.8 17.9 5.0 25.6 23.6 37.2 13.1 9.9 1.0 17.2 15.7 52.4 19.5 19.8 6.7 28.6 25.4 49.0 18.0 17.3 5.4 23.9 22.7<br><br>|

- Table 2: Performance across Math500, Minerva Math, Olympiad Bench, AMC2023, and AIME2024. Blend denotes Blend-PRM-GRPO. Reported accuracy is average@16 under temperature 1.0, with each method tuned to its best setting.

estimate the probability of reaching correct final answers (Wang et al., 2023; Xiong et al., 2024a; Luo

- et al., 2024). For this analysis, we select problemresponse pairs from test prompts where our method and GRPO both produced the correct final answer. Both models were initialized from Qwen2.5-Math7B-base. To estimate the value of each reasoning step, we generate eight independent completions from that point using a temperature of 1.0, and the resulting empirical success rate serves as the MC value. In Figure 5 (left), the average MC estimates across all five benchmarks are consistently higher for our model. The specific improvement gaps are 9.2% on Math500, 37.4% on Minerva Math, 15.9% on Olympiad Bench, 9.2% on AMC2023, and 11.1% on AIME2024, which are much larger than the outcome-accuracy gap in Table 2.

PROF Reduces Flawed Reasoning within Correct Responses. As a more direct faithfulness metric, we audit correct responses on the test set with Claude Sonnet 4.6 and ask whether the reasoning process contains any flaw (e.g., logical or arithmetic errors), even when the final answer is correct. The audit prompt is provided in Appendix A.2. In

- Figure 5 (right), the flawed-reasoning rate within correct responses decreases from 8% for GRPO to 6% for PROF. This complements Figure 1: Figure 1 measures flawed-reasoning prevalence in base-model outputs before RL (about 30%, specifically 26.28%), whereas Figure 5 reports the same notion after training, where both methods fall below 10% and PROF remains lower. We also note that Claude-based auditing is still an approximate signal of reasoning quality and cannot fully replace careful human judgment on step granularity, subtle unsupported jumps, or the level of detail. There-

[Figure 7]

[Figure 8]

Figure 5: Reasoning-process quality of PROF vs. GRPO. Left: Monte Carlo step-value scores across five benchmarks. Right: flawed-reasoning rate within correct test responses judged by Claude Sonnet 4.6 (GRPO: 8%, PROF: 6%).

fore, we additionally provide qualitative response comparisons in Figures 12 and 13. These examples consistently show that PROF produces concrete and verifiable intermediate deductions, GRPO tends to skip key steps, and Blend-PRM-GRPO is often verbose but less reliable in core calculations.

Additional process metrics on Math500 (step counts and averaged PRM scores) are moved to Appendix Figure 10. The key takeaway is that PROF improves process quality under both MCbased estimation and direct flaw auditing.

### 5 Ablations

#### 5.1 Robustness to PRM Capability

To showcase PROF’s robustness to PRM quality, we use a weaker and smaller Skywork-PRM-1.5B (He et al., 2024b) while training from Qwen2.5Math-7B-base. The results in Table 3 validate that when using a weaker PRM, Blend achieves lower accuracies, while PROF still maintains performance close to the model trained with the 7B PRM. This finding further corroborates the robustness of our algorithm.

|Algorithm<br><br>|Math500 Minerva Math Olympiad Bench AIME24 AMC23 Average|
|---|---|
|GRPO Blend (PRM-7B) PROF (PRM-7B) Blend (PRM-1.5B) PROF-POS (PRM-1.5B) PROF-BOTH (PRM-1.5B)<br><br>|81.6 37.2 45.5 20.6 64.4 49.9<br>81.7 36.7 45.0 15.2 58.0 47.3 83.1 39.0 47.8 17.5 70.9 51.7<br><br><br>81.1 37.8 44.1 11.7 62.8 47.5<br>82.9 39.4 47.4 19.2 66.1 51.0<br><br>83.2 38.8 47.8 17.5 65.0 50.5<br>|

- Table 3: Test accuracy (average@16, temperature 1.0) for Qwen2.5-Math-7B-base trained with different PRMs, averaged across five benchmarks.

|Method|Average score<br><br>|
|---|---|
|Raft++-n4 Raft++-n8 PROF-Raft++ (n8m4)<br><br>|35.27 37.75 39.29|

- Table 4: RAFT++ ablation under different rollout budgets.

the rightmost plot in Figure 6 shows that PROF w/o separation has over 2% gap between rewards before and after filtering, indicating disproportionate removal of negative samples. A likely reason is that incorrect responses often contain several locally correct steps, which can inflate averaged PRM scores and blur process-outcome consistency. Separating correct and incorrect groups alleviates this bias.

[Figure 9]

[Figure 10]

We then compare three variants: PROF-POS (consistency filtering on correct group only), PROFNEG (incorrect group only), and Filter-Random (random filtering on both groups) (Xu et al., 2025). As shown in Figure 6, PROF-POS and PROFBOTH are the best-performing strategies across both 1.5B and 7B settings; PROF-BOTH is typically more sample-efficient, PROF-NEG is weaker, Filter-Random is only slightly above GRPO, and w/o separation is the worst. These results suggest that preserving quality in correct responses is the dominant factor, while consistency control on incorrect responses is secondary. More filtration ablations are provided in Appendix B.

[Figure 11]

- Figure 6: Left two: averaged accuracy on Math500, Minerva Math, and Olympiad Bench for PROF variants from Qwen2.5-Math-1.5B-base and Qwen2.5-Math-7Bbase. Right: reward gap after vs. before filtering, with and without separation.

#### 5.2 Generality beyond GRPO: RAFT++

To demonstrate that PROF is a general filtration framework, we extend our experiments to RAFT++ (Xiong et al., 2025a), a rejection-sampling-based online training paradigm that only trains on positive samples. We compare PROF-Raft++ against standard RAFT++ baselines with different rollout budgets in Table 4. PROF-Raft++ not only outperforms the standard Raft++-n4 baseline, but also significantly surpasses Raft++-n8. Since RAFT++ only uses positive samples and does not involve negative samples, this comparison is primarily influenced by the number and quality of positive trajectories. Therefore, PROF’s priority-based filtration is algorithm-agnostic and consistently identifies high-quality reasoning paths that lead to better policy improvement, regardless of the underlying RL objective.

This ablation highlights a practical trade-off between PROF-BOTH and PROF-POS. PROFBOTH usually converges faster by using consistency signals from both groups, while PROF-POS can be more robust when PRM reliability is weaker or distribution shift is larger, since it avoids tightly shaping the incorrect group with noisy estimates. In both cases, improving correct trajectories is the main driver, and filtering incorrect trajectories mainly affects efficiency and stability.

### 6 Conclusion and Future Work

This work introduces Process Consistency Filter (PROF), a data curation technique that filters generated responses based on PRM–ORM consistency while maintaining a balanced correct/incorrect ratio. We demonstrate that PROF consistently improves final-answer accuracy and shapes the policy to generate more detailed and fine-grained intermediate reasoning steps. PROF is also a general filtration framework rather than one tied to a spe-

#### 5.3 Separating Correct and Incorrect

We first test a no-separation variant (PROF w/o separation) that ranks all rollouts together. To mitigate PRM scale bias, we center each step score by subtracting the batch mean. Even with centering,

cific PRM or RL objective. Thus, using pre-trained PRMs in our experiments is not a limitation; instead, it highlights the robustness of our algorithm to different PRMs and suggests that training a taskspecific PRM for each base model is unnecessary. Exploring stronger or more diverse PRMs, and extending PROF to other reasoning tasks such as coding (Jimenez et al., 2023) and web navigation (Zhou et al., 2023), remains important future work.

Broader Impact and Ethics Statement Our work contributes to AI safety by enhancing the faithfulness and interpretability of chain-of-thought reasoning, mitigating the risk of misleading hallucinations. However, we acknowledge two potential risks. First, the reliance on oversampling and dense process reward computation increases computational overhead and environmental impact compared to standard baselines. Second, our filtration mechanism depends on pre-trained Process Reward Models (PRMs); if these PRMs harbor biases toward specific reasoning patterns or languages, our method may inadvertently amplify such biases by filtering out diverse but valid solutions. We encourage future research to address these efficiency and fairness challenges.

### Limitations

Although PROF can effectively improve robustness to PRM noise and increase reasoning-step quality, our method requires more computation than Blend or vanilla GRPO because it first oversamples and then filters. How to balance efficiency and reasoning quality remains an important direction for future work. Finally, we acknowledge the use of AI assistants (e.g., ChatGPT) for grammatical error correction and polishing of the manuscript.

### References

Bowen Baker, Joost Huizinga, Leo Gao, Zehao Dou, Melody Y Guan, Aleksander Madry, Wojciech Zaremba, Jakub Pachocki, and David Farhi. 2025. Monitoring reasoning models for misbehavior and the risks of promoting obfuscation. arXiv preprint arXiv:2503.11926.

Edward Beeching, Shengyi Costa Huang, Albert Jiang, Jia Li, Benjamin Lipkin, Zihan Qina, Kashif Rasul, Ziju Shen, Roman Soletskyi, and Lewis Tunstall. 2024. Numinamath 7b cot. https://huggingface. co/AI-MO/NuminaMath-7B-CoT.

Ralph Allan Bradley and Milton E Terry. 1952. Rank analysis of incomplete block designs: I. the method

of paired comparisons. Biometrika, 39(3/4):324– 345.

Huayu Chen, Kaiwen Zheng, Qinsheng Zhang, Ganqu Cui, Yin Cui, Haotian Ye, Tsung-Yi Lin, Ming-Yu Liu, Jun Zhu, and Haoxiang Wang. 2025a. Bridging supervised learning and reinforcement learning in math reasoning. arXiv preprint arXiv:2505.18116.

Yanda Chen, Joe Benton, Ansh Radhakrishnan, Jonathan Uesato, Carson Denison, John Schulman, Arushi Somani, Peter Hase, Misha Wagner, Fabien Roger, and 1 others. 2025b. Reasoning models don’t always say what they think. arXiv preprint arXiv:2505.05410.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, and 1 others. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, and 1 others. 2025. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456.

Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, and Tong Zhang. 2023. Raft: Reward ranked finetuning for generative foundation model alignment. arXiv preprint arXiv:2304.06767.

Hanze Dong, Wei Xiong, Bo Pang, Haoxiang Wang, Han Zhao, Yingbo Zhou, Nan Jiang, Doyen Sahoo, Caiming Xiong, and Tong Zhang. 2024. Rlhf workflow: From reward modeling to online rlhf. arXiv preprint arXiv:2405.07863.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, and 1 others. 2024. The llama 3 herd of models. arXiv e-prints, pages arXiv–2407.

Jacob Eisenstein, Chirag Nagpal, Alekh Agarwal, Ahmad Beirami, Alex D’Amour, DJ Dvijotham, Adam Fisch, Katherine Heller, Stephen Pfohl, Deepak Ramachandran, and 1 others. 2023. Helping or herding? reward model ensembles mitigate but do not eliminate reward hacking. arXiv preprint arXiv:2312.09244.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, and 1 others. 2024a. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008.

Jujie He, Tianwen Wei, Rui Yan, Jiacai Liu, Chaojie Wang, Yimeng Gan, Shiwen Tu, Chris Yuhao Liu,

Liang Zeng, Xiaokun Wang, Boyang Wang, Yongcong Li, Fuxiang Zhang, Jiacheng Xu, Bo An, Yang Liu, and Yahui Zhou. 2024b. Skywork-o1 open series.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874.

Alon Jacovi, Yonatan Bitton, Bernd Bohnet, Jonathan Herzig, Or Honovich, Michael Tseng, Michael Collins, Roee Aharoni, and Mor Geva. 2024. A chain-of-thought is as strong as its weakest link: A benchmark for verifiers of reasoning chains. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, and 1 others. 2024. Openai o1 system card. arXiv preprint arXiv:2412.16720.

Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. 2023. Swe-bench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770.

Muhammad Khalifa, Rishabh Agarwal, Lajanugen Logeswaran, Jaekyeom Kim, Hao Peng, Moontae Lee, Honglak Lee, and Lu Wang. 2025. Process reward models that think. arXiv preprint arXiv:2504.16828.

Sunnie SY Kim, Q Vera Liao, Mihaela Vorvoreanu, Stephanie Ballard, and Jennifer Wortman Vaughan. 2024. " i’m not sure, but...": Examining the impact of large language models’ uncertainty expression on user reliance and trust. In Proceedings of the 2024 ACM conference on fairness, accountability, and transparency, pages 822–835.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, and 1 others. 2022. Solving quantitative reasoning problems with language models. Advances in neural information processing systems, 35:3843–3857.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2023. Let’s verify step by step. In The Twelfth International Conference on Learning Representations.

Yong Lin, Hangyu Lin, Wei Xiong, Shizhe Diao, Jianmeng Liu, Jipeng Zhang, Rui Pan, Haoxiang Wang, Wenbin Hu, Hanning Zhang, and 1 others. 2023. Mitigating the alignment tax of rlhf. arXiv preprint arXiv:2309.06256.

Liangchen Luo, Yinxiao Liu, Rosanne Liu, Samrat Phatale, Meiqi Guo, Harsh Lara, Yunxuan Li, Lei Shu, Yun Zhu, Lei Meng, Jiao Sun, and Abhinav Rastogi. 2024. Improve mathematical reasoning in language models by automated process supervision. Preprint, arXiv:2406.06592.

Qing Lyu, Shreya Havaldar, Adam Stein, Li Zhang, Delip Rao, Eric Wong, Marianna Apidianaki, and Chris Callison-Burch. 2023. Faithful chain-ofthought reasoning. In The 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (IJCNLPAACL 2023).

Eric J Michaud, Adam Gleave, and Stuart Russell. 2020. Understanding learned reward functions. arXiv preprint arXiv:2012.05862.

Minh-Vuong Nguyen, Linhao Luo, Fatemeh Shiri, Dinh Phung, Yuan-Fang Li, Thuy-Trang Vu, and Gholamreza Haffari. 2024. Direct evaluation of chainof-thought in multi-hop reasoning with knowledge graphs. In Findings of the Association for Computational Linguistics: ACL 2024.

Debjit Paul, Robert West, Antoine Bosselut, and Boi Faltings. 2024. Making reasoning matter: Measuring and improving faithfulness of chain-of-thought reasoning. In Findings of the Association for Computational Linguistics: EMNLP 2024.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, and 1 others. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2025. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, pages 1279–1297.

Jeremy Tien, Jerry Zhi-Yang He, Zackory Erickson, Anca D Dragan, and Daniel S Brown. 2022. Causal confusion and reward misidentification in preference-based reward learning. arXiv preprint arXiv:2204.06601.

Miles Turpin, Julian Michael, Ethan Perez, and Samuel R. Bowman. 2023. Language models don’t always say what they think: Unfaithful explanations in chain-of-thought prompting. In Advances in Neural Information Processing Systems 36 (NeurIPS 2023).

Peiyi Wang, Lei Li, Zhihong Shao, RX Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. 2023. Math-shepherd: Verify and reinforce llms stepby-step without human annotations. arXiv preprint arXiv:2312.08935.

Wei Xiong, Chengshuai Shi, Jiaming Shen, Aviv Rosenberg, Zhen Qin, Daniele Calandriello, Misha Khalman, Rishabh Joshi, Bilal Piot, Mohammad Saleh, and 1 others. 2024a. Building math agents with multiturn iterative preference learning. arXiv preprint arXiv:2409.02392.

Wei Xiong, Jiarui Yao, Yuhui Xu, Bo Pang, Lei Wang, Doyen Sahoo, Junnan Li, Nan Jiang, Tong Zhang, Caiming Xiong, and 1 others. 2025a. A minimalist approach to llm reasoning: from rejection sampling to reinforce. arXiv preprint arXiv:2504.11343.

Wei Xiong, Hanning Zhang, Nan Jiang, and Tong Zhang. 2024b. An implementation of generative prm.

Wei Xiong, Hanning Zhang, Chenlu Ye, Lichang Chen, Nan Jiang, and Tong Zhang. 2025b. Selfrewarding correction for mathematical reasoning. arXiv preprint arXiv:2502.19613.

Wei Xiong, Wenting Zhao, Weizhe Yuan, Olga Golovneva, Tong Zhang, Jason Weston, and Sainbayar Sukhbaatar. 2025c. Stepwiser: Stepwise generative judges for wiser reasoning. Preprint, arXiv:2508.19229.

Yixuan Even Xu, Yash Savani, Fei Fang, and Zico Kolter. 2025. Not all rollouts are useful: Downsampling rollouts in llm reinforcement learning. arXiv preprint arXiv:2504.13818.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, and 1 others. 2024. Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122.

Wei Jie Yeo, Ranjan Satapathy, Rick Siow Mong Goh, and Erik Cambria. 2024. How interpretable are reasoning explanations from prompting large language models? In Findings of the Association for Computational Linguistics: NAACL 2024.

Ping Yu, Weizhe Yuan, Olga Golovneva, Tianhao Wu, Sainbayar Sukhbaatar, Jason Weston, and Jing Xu. 2025a. Rip: Better models by survival of the fittest prompts. arXiv preprint arXiv:2501.18578.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, and 1 others. 2025b. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476.

Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Sainbayar Sukhbaatar, Jing Xu, and Jason Weston. 2024. Self-rewarding language models. arXiv preprint arXiv:2401.10020, 3.

Kaiwen Zha, Zhengqi Gao, Maohao Shen, ZhangWei Hong, Duane S Boning, and Dina Katabi. 2025. Rl tango: Reinforcing generator and verifier together for language reasoning. arXiv preprint arXiv:2505.15034.

Chuheng Zhang, Wei Shen, Li Zhao, Xuyun Zhang, Lianyong Qi, Wanchun Dou, and Jiang Bian. 2024. Policy filtration in rlhf to fine-tune llm for code generation.

Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. 2025. The lessons of developing process reward models in mathematical reasoning. arXiv preprint arXiv:2501.07301.

Jian Zhao, Runze Liu, Kaiyan Zhang, Zhimu Zhou, Junqi Gao, Dong Li, Jiafei Lyu, Zhouyi Qian, Biqing Qi, Xiu Li, and Bowen Zhou. 2025. Genprm: Scaling test-time compute of process reward models via generative reasoning. Preprint, arXiv:2504.00891.

Chujie Zheng, Zhenru Zhang, Beichen Zhang, Runji Lin, Keming Lu, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. 2024. Processbench: Identifying process errors in mathematical reasoning. arXiv preprint arXiv:2412.06559.

Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, and 1 others. 2023. Webarena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854.

Dawei Zhu, Xiyu Wei, Guangxiang Zhao, Wenhao Wu, Haosheng Zou, Junfeng Ran, Xun Wang, Lin Sun, Xiangzheng Zhang, and Sujian Li. 2025. Chain-ofthought matters: improving long-context language models with reasoning path supervision. arXiv preprint arXiv:2502.20790.

Jiaru Zou, Ling Yang, Jingwen Gu, Jiahao Qiu, Ke Shen, Jingrui He, and Mengdi Wang. 2025. Reasonflux-prm: Trajectory-aware prms for long chain-of-thought reasoning in llms. arXiv preprint arXiv:2506.18896.

### A Additional Experimental Details and Results

- A.1 Main Experiments

The implementations are based on the verl framework (Sheng et al., 2025), and we follow most of its parameter settings. Specifically, we use the AdamW optimizer with learning rate 1 × 10−6. We adopt the clip-higher trick (Yu et al., 2025b), which clips the sampling ratio πθ/πold to an asymmetric range (1 − ϵlow,1 + ϵhigh). Specifically, we set ϵlow = 0.2,ϵhigh = 0.28 for models initialized from Qwen2.5-Math-1.5B-base and maintain ϵhigh = ϵlow = 0.2 for other cases. In each iteration, we sample 1024 prompts and roll out n = 4 responses per prompt for GRPO and n = 8 responses for PROF. Note that the policy update number for all algorithms is m = 4. For the regularization of step numbers in Algorithm 1, we take λ = 10 and Hλ = 30. For the rollout stage, we use a temperature of 1.0 and a top-p value of 1.0. We set the KL loss coefficient to 0.001 and entropy loss coefficient to 0.001. All the models are trained with 8 H100 GPUs. We set the training mini-batch size as 256 and allow the models to generate 4096 tokens per prompt.

- A.2 Prompt Template

We present the template used by the LLM to audit whether a correct response still contains reasoning flaws in Table 5.

### B Additional Experimental Results

In this section, we include additional ablation studies and evaluation results for a more comprehensive understanding of the PROF framework.

- B.1 Matched-Cost Results for Difficulty Levels 1–3
- B.2 Effect of Rollout Numbers

We study the scale of rollout numbers n with fixed policy-update number m = 4 by varying n = 4,8,12,16. The lower-right plot in Figure 9 presents the test accuracy averaged over all five benchmarks for PROF-BOTH (Both) and PROFPOS (Correct) initialized from Qwen2.5-Math-7Bbase. We observe that performance first increases and then decreases as n grows, revealing a tradeoff between enhancing process reasoning quality and avoiding reward hacking. Notably, PROF-POS decreases later (after n = 12) because it only leverages PRM influence in the correct group, indicating

that PROF-POS is more robust when PRM influence becomes stronger, such as when the rankingand-filtering scale increases.

- B.3 Additional Process Metrics on Math500
- B.4 Variants of Filtration Methods

In this subsection, we investigate different ways of computing the consistency score rpro, in addition to taking the mean of PRM scores over steps. Here, Mean denotes averaging over steps in Algorithm 1; Minimum and Sum denote taking the minimum and the sum over steps; and Ratio denotes filtering while preserving the original positive/negative sample distribution instead of balancing it. As shown in Table 6, Minimum (50.9%), Sum (50.6%), and Ratio (50.6%) all underperform Mean. This suggests that the mean provides a more stable estimate of reasoning consistency: unlike the minimum, it is less sensitive to a single poorly scored step, and unlike the sum, it avoids bias toward longer trajectories. Additionally, balancing the correct/incorrect ratio lets consistency-based filtering select the better-supported group without breaking class balance.

- B.5 Effect of Step Number

To verify that PROF does not help merely by increasing the number of steps, we evaluate FilterNstep, which ranks and filters samples by shorter step counts instead of lower PRM–ORM consistency.

From Table 6, we find that Ratio scores only 50.6% on average and cannot compete with balanced filtering (PROF), which further corroborates the importance of maintaining a balanced correct/incorrect proportion. Additionally, because PROF increases the number of intermediate reasoning steps, we compare against simple step-length filtering to verify that the gain does not come merely from longer responses. As shown in Figure 11 and Table 7, Filter-Nstep mainly manipulates step length, exhibits an unreasonable increase followed by a sudden drop, and yields inferior average accuracy.

### C Additional Examples

[Figure 12]

[Figure 13]

- Figure 7: Entropy loss (left) and response length (right) of the models initialized from Qwen2.5-Math-7B-base.

[Figure 14]

[Figure 15]

[Figure 16]

- Figure 8: Matched-cost comparison on Qwen2.5-Math-7B-base for difficulty Level 1 (p > 0.5), Level 2 (0.25 < p ≤ 0.5), and Level 3 (0 < p ≤ 0.25). Average cost is computed as Inference +3× Train + PRM.

[Figure 17]

- Figure 9: The averaged accuracy across all five benchmarks over rollout sizes n = 4,8,12,16 for filtering both correct and incorrect groups with PRM consistency (Both) and only the correct group with PRM consistency (Correct).

[Figure 18]

[Figure 19]

- Figure 10: Supplementary process metrics on Math500 for PROF vs. GRPO: number of reasoning steps (left) and averaged Qwen2.5-Math-PRM-7B scores (right).

[Figure 20]

Figure 11: The number of reasoning steps during training time for PROF-GRPO and Filter-Nstep initialized from Qwen2.5-Math-7B-base.

[Figure 21]

- Figure 12: A Minerva-Math example comparing distinct intermediate reasoning patterns of PROF-BOTH, vanilla GRPO, and Blend-PRM-GRPO. PROF-BOTH presents concrete and correct deduction steps. GRPO’s solution skips detailed deductions and contains flaws in calculation precision and final rounding. Blend-PRM-GRPO is long-winded and makes a major mistake in computing the power.

[Figure 22]

- Figure 13: A Math500 example to compare distinct intermediate reasoning patterns of PROF-BOTH, vanilla GRPO and Blend-PRM-GRPO. PROF-BOTH presents concrete and correct deduction steps. PROF-BOTH’s solution shows how to find the divisors and summation in detail, and is easy to follow. GRPO skips all core reasoning. Blend-PRM-GRPO has inefficient and excessively tedious steps.

Prompt for Auditing Reasoning Flaws in a Candidate Solution System Your task is to audit a candidate solution to a math problem and determine whether the reasoning process contains any flaw or error, even if the final answer is correct. User You must evaluate the solution step by step and focus on the validity of the reasoning process, not writing style. A solution can be flawed even when the final answer is correct. Reasoning flaws include:

- • random guessing;
- • skipping or jumping over intermediate steps;
- • logical errors;
- • arithmetic errors;
- • misuse of definitions, theorems, or formulas.

Do not penalize: minor wording issues or overly specific solution styles. Important rules:

- • Judge the reasoning process itself, not just the final answer.
- • If the final answer is correct but any earlier step is invalid or unsupported, mark the reasoning as flawed.
- • Identify the earliest step where the reasoning first becomes flawed.
- • Once an earliest flawed step is found, later steps may be unreliable because they can depend on that error.

You will be given a math problem and a candidate solution. If the candidate solution does not explicitly label steps, first segment it into natural reasoning steps before judging. Return your judgment in exactly the following format: <final_answer_correct>yes / no / unclear</final_answer_correct> <reasoning_flawed>yes / no</reasoning_flawed> <earliest_flawed_step>step number or none</earliest_flawed_step> <error_type>arithmetic | algebra | logical | unsupported_jump | random_guessing | theorem_misuse | missing_case | contradiction | answer_first_rationalization | none</error_type> <brief_explanation> A concise explanation of why the reasoning is valid, or why the earliest flawed step is flawed. </brief_explanation> Now evaluate the following example. [Problem] {{prompt}}

[Candidate Solution] {{responses}}

Table 5: Prompt for auditing flawed reasoning in candidate solutions via LLM-as-a-judge.

|Algorithm<br><br>|Math500 Minerva Math Olympiad Bench AIME24 AMC23 Average|
|---|---|
|Mean Minimum Sum Ratio|83.1 39.0 47.8 17.5 70.9 51.7 82.9 38.3 46.7 20.8 65.9 50.9 82.4 38.1 47.4 17.7 67.5 50.6 81.4 36.6 45.0 24.8 65.2 50.6<br><br>|

Table 6: Performance of different filtration ways in PROF starting from Qwen2.5-Math-7B-base.

|Algorithm|Math500 Minerva Math Olympiad Bench AIME24 AMC23 Average<br><br>|
|---|---|
|PROF-BOTH Filter-Nstep|83.1 39.0 47.8 17.5 70.9 51.7 81.5 35.5 45.9 16.3 58.6 47.6<br><br>|

Table 7: Performance of filtration variants besides PROF-BOTH for Qwen2.5-Math-7B-base, averaged over all five benchmarks. Ratio preserves the original correct/incorrect proportion, and Filter-Nstep ranks and filters by the number of step segments.

