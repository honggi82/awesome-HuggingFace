## arXiv:2512.10739v2[cs.CL]12Dec2025

[Figure 1]

2025-12-15

# Long-horizon Reasoning Agent for Olympiad-Level Mathematical Problem Solving

###### Songyang Gao1* Yuzhe Gu1,2* Zijian Wu1,3* Lingkai Kong1,2* Wenwei Zhang1*†

Zhongrui Cai1 Fan Zheng4 Tianyou Ma1 Junhao Shen1,2 Haiteng Zhao1 Duanyang Zhang5 Huilun Zhang6 Kuikun Liu1 Chengqi Lyu1 Yanhui Duan1 Chiyu Chen1 Ningsheng Ma1 Jianfei Gao1 Han Lyu1 Dahua Lin1,3 Kai Chen1† 1Shanghai AI Laboratory 2Shanghai Jiao Tong University 3MMLab, The Chinese University of Hong Kong 4ICMAT, Spanish National Research Council 5The High School Affiliated to Renmin University of China 6Ren Hui Academy of Beijing

{gaosongyang,guyuzhe,zhangwenwei,chenkai}@pjlab.org.cn

Large Reasoning Models (LRMs) have expanded the mathematical reasoning frontier through Chain-of-Thought (CoT) techniques and Reinforcement Learning with Verifiable Rewards (RLVR), capable of solving AIME-level problems. However, the performance of LRMs is heavily dependent on the extended reasoning context length. For solving ultra-hard problems like those in the International Mathematical Olympiad (IMO), the required reasoning complexity surpasses the space that an LRM can explore in a single round. Previous works attempt to extend the reasoning context of LRMs but remain prompt-based and built upon proprietary models, lacking systematic structures and training pipelines. Therefore, this paper introduces Intern-S1-MO, a long-horizon math agent that conducts multi-round hierarchical reasoning, composed of an LRM-based multi-agent system including reasoning, summary, and verification. By maintaining a compact memory in the form of lemmas, Intern-S1-MO can more freely explore the lemma-rich reasoning spaces in multiple reasoning stages, thereby breaking through the context constraints for IMO-level math problems. Furthermore, we propose OREAL-H, an RL framework for training the LRM using the online explored trajectories to simultaneously bootstrap the reasoning ability of LRM and elevate the overall performance of Intern-S1-MO. Experiments show that InternS1-MO can obtain 26 out of 35 points on the non-geometry problems of IMO2025, matching the performance of silver medalists. It also surpasses the current advanced LRMs on inference benchmarks such as HMMT2025, AIME2025, and CNMO2025. In addition, our agent officially participates in CMO2025 and achieves a score of 102/126 under the judgment of human experts, reaching the gold medal level.

### 1. Introduction

Reasoning is a highly intellectual human activity that requires the integration of deductive logic, pattern recognition, and creative problem decomposition to address complex challenges, which is regarded as a significant milestone towards Artificial General Intelligence (AGI) [31]. In recent years, large reasoning models (LRMs) have made substantial progress in mathematical reasoning, driven primarily by techniques such as Chain-of-Thought (CoT) [32, 43] and Reinforcement Learning from Verifiable Rewards (RLVR) [29, 39, 40]. Along with the increasing reasoning capabilities of LRMs, a clear trend is that LRMs are being allocated more thinking budgets for more difficult problems to support exploration of larger solution spaces and trial-and-error processes [1, 45].

However, hardware and data limitations have made unlimited scaling of context length infeasible. Currently, state-of-the-art (SOTA) reasoning models typically support a maximum context length of only 64k or 128k tokens [2, 6, 36], insufficient for ultra-challenging problems such as those in International Mathematical

* Equal contribution, † Corresponding author

[Figure 2]

- Figure 1: The motivation (a) and performance(b) of Intern-S1-MO. As problem difficulty increases, both the average human thinking time and the model token consumption per problem grow exponentially (a), already reaching concerning limits under current development trends. Intern-S1-MO enables LRMs to use about 512K tokens to solve a single problem, achieving state-of-the-art performance on challenging mathematical benchmarks (b).

Olympiads (IMO) 1. Figure 1(a) illustrates the logarithmic growth of the required context length with increasing difficulty of the problem, highlighting the mismatch between the existing capacity limits and practical demands. Although resource investment can marginally raise this context ceiling, developing a cost-effective paradigm to meet context requirements is more compelling [13, 15].

Some studies have explored multi-round interaction [21] or parallel decoding [41] to perform long logical deduction in mathematical reasoning. Furthermore, Huang & Yang [12] introduced self-reflective with prompt engineering, allowing models to identify flaws in intermediate reasoning steps and refine the results. Nevertheless, these approaches still confine problem-solving to a single reasoning cycle (even with internal iterations) rather than building cumulatively upon prior reasoning trajectories, which limits their capacity to leverage historical explorations for further in-depth deduction [33]. Alternatively, formal language–based search [4, 27, 46] shows some promise: by maintaining a structured repository to store and reuse intermediate results, they reduce reliance on model context length. However, the proof verification and state traversal demand extensive iterations, leading to high computational and search overhead. Moreover, formal systems require translating informal descriptions into formal logic, introducing additional costs and hindering the interaction between AI and humans.

Proprietary LRMs [7, 25] have reported impressive results on the International Mathematical Olympiad 2025 (IMO2025) problems, yet the research community lacks access to their methodologies and models. In this work, we present Intern-S1-MO, a math reasoning agent framework unconstrained by context length, which solves complex reasoning problems through hierarchical decomposition. This strategy closely aligns with human problem-solving patterns. Intern-S1-MO achieves unlimited exploration capability through lemma memory management. Specifically, after each single-round reasoning, the agent compresses its current reasoning history into concise sub-lemmas with a structured memory repository, which enables the agent to recover historical exploration outcomes in subsequent steps. We furthermore design process verification and revision mechanisms to certify the quality of the lemma repository. Notably, Intern-S1-MO enables adaptive control of its reasoning budget: it initiates multi-round exploration only for challenging tasks, ensuring efficient resource allocation.

To support the bootstrapping and online improvement of Intern-S1-MO, we additionally introduce the OREAL-H framework, enabling the agent to enhance its performance on complex problems with online reinforcement learning (RL). Starting from the basic formulation of Outcome Reward Reinforcement Learning (OREAL) [18], OREAL-H exploits the additional reward signal produced by the outcome process verifier (OPV) that is continuous and accelerates training, and is modified for the Hierarchical Markov Decision Process (MDP) formulation to suit the multi-agent setting of Intern-S1-MO.

Extensive experimental results show that Intern-S1-MO establishes new state-of-the-art results across

1https://imo2025.au

multiple mathematical reasoning benchmarks. As shown in Figure 1(b), on commonly used inference benchmarks like AIME2025 and HMMT2025, it achieves a 96.6% and 95% pass@1 score, respectively, surpassing the current advanced LRMs. To evaluate the performance of Intern-S1-MOon more difficult, Olympiad-level problems, we test it on the 5 non-geometry problems of IMO2025, and it obtains 26 out of 35 points, surpassing the silver medalist-level performance (21 points) of humans. We also test it on the Chinese National High School Mathematics Olympiad (CNMO2025), which is the preliminary round of the Chinese Mathematics Olympiad (CMO2025)2. CNMO2025 comprises 14 high-school math competition problems (excluding geometry problems), on which our system scores 232.4 out of 260 points. Additionally, in order to evaluate in a real-world environment, we officially participate in CMO2025 and conduct the test under the same time limit and grading standards as human contestants. After evaluation by human experts, our system received a score of 102 out of 126, largely exceeding the gold medal threshold of 78 points. Overall, our contributions are as follows:

- • We explore multi-round complex reasoning scenarios and propose a multi-agent system, Intern-S1-MO, which effectively extends the reasoning depth of current LRMs by the lemma-based memory management.
- • We contribute an RL framework, termed OREAL-H, for optimizing the multi-round performance of Intern-S1-MO on high-difficulty mathematical problems.
- • Experiments prove that our Intern-S1-MO outperforms current advanced LRM like Gemini 2.5 Pro. Specifically, it can match the performance of silver medalists in IMO2025, gold medalists in CMO2025, and get SOTA in benchmarks like AIME2025, HMMT2025, and CNMO2025.

### 2. Related Work

###### 2.1. Mathematical Reasoning Agents

Recent advancements in large reasoning models have significantly enhanced their performance on mathematical reasoning tasks; however, systematic exploration and reflection are still areas that require further investigation. A notable approach involves the use of tree search methods—such as Tree-of-Thoughts [37] and Monte Carlo Tree Search [41]—to facilitate parallel search during inference. While these methods broaden the search landscape, they often lack depth and struggle to effectively decompose complex problems[3, 31]. Other research has focused on augmenting LLMs with external tools to ground reasoning in computation or verified knowledge [8, 11, 29]. Yet, these tools typically serve to enhance the existing reasoning process rather than fundamentally restructure it. More recent efforts propose structured reasoning frameworks that integrate planning, exploration, and reflection to iteratively refine solutions [38]. These methods outperform standard chain-of-thought prompting on challenging problems, but they usually rely on carefully designed prompts and sometimes human-provided hints. Importantly, they shift reasoning from single-path generation to structured problem solving. Yet, training math agents—where exploration and reflection are optimized through learning signals—remains an emerging area [26]. Recent initiatives have introduced structured reasoning frameworks that integrate exploration and reflection to iteratively refine solutions [12]. These methods have been shown to outperform traditional methods on challenging problems. However, they often depend on meticulously crafted prompts and, at times, hints provided by humans.

###### 2.2. Reinforcement Learning for Math Agents

Reinforcement learning (RL) for mathematical reasoning has primarily focused on outcome rewards, where feedback is based solely on final answer correctness. Despite this sparse signal, methods like ARTIST [42], ToRL [16], and rStar2-Agent [28] exhibit emergent agentic behaviors—such as adaptive tool use, self-correction, and context-aware reasoning. Scaling studies (e.g., ZeroTIR [19]) further show that increased training effort leads to more sophisticated tool-integrated strategies. Nevertheless, current math agents remain limited: their decisions are mostly confined to choosing when to retry within a fixed reasoning template—rather than engaging in strategic planning or deep exploration. Critically, they lack summarization and cross-episode awareness. While approaches like TTRL [47] and Satori [30] introduce basic reflection or meta-actions, they operate within isolated reasoning episodes and do not support cumulative knowledge transfer across inferences. Process-aware RL and verifier-guided training (e.g., Prover-Verifier Games [14]) aim to provide intermediate supervision with predefined rules or code execution, and are not well-suited for complex reasoning scenarios.

2https://www.cms.org.cn

[Figure 3]

- Figure 2: The agentic framwork of Intern-S1-MO. In each reasoning round, the reasoner agent tries to solve the question, and the summarizer agent compresses the current reasoning history into a series of lemmas, which will be added to the memory system after being verified by the verifier agent. Except for the first round, the lemma library will be input into the reasoning agent along with the question. In the final round, the solution generated by the reasoner agent undergoes a modification loop, which improves the quality of the solution based on feedback from the verifier agent, until the verification is passed or the maximum number of loop rounds is reached.

In this paper, we use a process verifier to judge the rigor of natural language proofs, which provides a more flexible feedback signal.

### 3. Building Hierarchical Reasoning Agents

To extend the exploration of reasoning, we designed a hierarchical mathematical reasoning agent tailored for complex competition-level mathematical problems, as shown in Figure 2. By enabling recursive subproblem solving, it specifically addresses the aforementioned reasoning limitations constrained by context length. We give a case example in Appendix F.

Decomposing Sub-Problems for Lemma Search Decomposing complex problems into manageable sub-lemmas is a defining feature of human problem-solving for high-difficulty mathematics, as it breaks long-chain logical reasoning into incremental steps. We first observe that state-of-the-art models already exhibit a degree of reasonable decomposition capability for mathematical problems, though this ability is often undermined by a premature conclusion bias: when reasoning budgets are exhausted, models tend to rush toward incomplete or incorrect final answers instead of acknowledging partial progress. To mitigate this, we refine the model via prompt engineering and targeted training, explicitly enabling it to produce partial deductive progress in single-turn attempts (e.g., deriving intermediate sub-lemmas without forcing a full problem solution). This adjustment aligns the model’s behavior with human iterative reasoning and lays the groundwork for cumulative exploration, the complete style requirements are presented in the Appendix A.

Summarizing Exploration for Memory Maintenance The model’s reasoning processes for complex problems often include redundant exploratory efforts and trial-and-error content. While this content aids in generating intermediate conclusions, it adds little value to subsequent deductive steps. Such facts enable us to extract only the essential components that drive progress, specifically, validated intermediate lemmas from each reasoning turn and store them in a structured lemma library. This library encourages the agent to reuse historical conclusions during new exploration rounds, allowing for deeper deductions based on prior lemmas rather than reprocessing redundant information. Notably, summarizing compelling exploration is as complex as the exploration process itself, as it requires distilling and checking the logical validity independently. Therefore, we allocate a dedicated reasoning turn after each exploration step to update the lemma library. This computational cost is necessary to ensure the library remains useful for long-chain reasoning.

Verifying Theorems to Mitigate Error Propagation Advanced reasoning models can self-reflect, but if they rely on erroneous historical premises, they will expend significant resources trying to validate questionable results. Such a problem is compounded by error propagation, so that a flawed intermediate conclusion can mislead subsequent deductive directions, leading to circular reasoning or invalid proofs. Fortunately, the verification of lemmas is comparatively more tractable than that of the complete problem. We address this by integrating a theorem verifier that uses parallel sampling to compute confidence scores for each lemma. Specifically, for each lemma, we make the theorem verifier perform n parallel verifications, and the proportion of those correctly identified is used as the confidence score. We believe this improves the reliability of theorem verification, avoiding some false positives or false negatives.

Verifying Process for Final Proof Completion Verifying the validity of final solutions is crucial for obtaining reliable performance feedback, both in evaluation scenarios and reinforcement learning loops. To achieve this, we utilize the process verifier from OPV [34], whose evaluations demonstrate that their verifier achieves an F1-score greater than 85% on ProcessBench [44], surpassing the performance of o1-mini. In practice, the verifier serves two main functions: (1) enhancing robustness through test time scaling by aggregating verification results across multiple runs, and (2) providing high-quality feedback signals for iterative revision and reinforcement learning training to further optimize the agent’s reasoning precision.

### 4. RL training for Evolution of math agents

###### 4.1. Preliminaries

We model the agentic mathematical reasoning process as a Hierarchical Markov Decision Process, denoted ℳ = ⟨𝒮,𝒰,𝒱,𝑟,𝑅,𝛾⟩, where 𝒮 is the state space (problem context + reasoning trace + verification feedback), 𝒰 the high-level meta-action space (e.g., “extract lemmas”, “invoke verification”, “commit answer”), and 𝒱 the low-level token vocabulary. The agent alternates between high-level decisions and low-level generation: at each round 𝑡, it executes a reasoning action 𝑢𝑡 with token sequence 𝑣𝑡 = (𝑣𝑡,1,...,𝑣𝑡,𝑇

) ∼ 𝜋𝜃𝐿(·|𝑠𝑡) to produce a reasoning segment. This output is summarized and verified by an external module, yielding natural language feedback which induces an intermediate proxy reward 𝑟𝑡 ∈ R. Upon termination after several rounds, a sparse final reward 𝑅 indicates correctness of the solution. The training objective is to maximise expected final reward:

𝑡

𝜑 ,𝜋𝜃𝐿 [𝑅]. (1) Leveraging the conditional structure of the hierarchical policy, the per-round advantage can be estimated

𝐽(𝜃,𝜑) = E𝜋𝐻

via a high-level critic 𝑉 (𝑠𝑡), updated to satisfy:

###### 𝑉 (𝑠𝑡) ← E[𝑟𝑡 + 𝛾𝑉 (𝑠𝑡+1)], (2)

where 𝑠𝑡+1 is the state after applying 𝑢𝑡. The advantage for round 𝑡 is then 𝐴𝑡 = 𝑟𝑡 + 𝛾𝑉 (𝑠𝑡+1) − 𝑉 (𝑠𝑡). On low-level, we can then perform an online policy gradient conditioned on this advantage, aggregating token-level log-likelihoods within the round:

∇𝜃𝐽 = E[︃ 𝐾

∑︁

𝐴𝑡 ·

𝑡=1

∇𝜃 log 𝜋𝜃𝐿(𝑣𝑡,𝜏 |𝑠𝑡,𝑣𝑡,<𝜏)]︃. (3)

∑︁𝑇𝑡

𝜏=1

Reward Function As mentioned in Section 3, we employ a Process Verifier (PV) to assess the logical rigor of complex mathematical proofs. Specifically, the PV examines the agent’s final solution and outputs natural language feedback identifying the indices of steps containing logical fallacies. We estimate the PV’s confidence via a multi-round voting mechanism. In particular, for problems amenable to outcome supervision, the final reward 𝑅 is set to 0 if the final answer is incorrect. We further discuss the role of these supervision signals for RL steps in Section 4.3.

###### 4.2. Cloning Success Trajectory for Cold Start

To prime the agent’s adherence to structured reasoning formats and internalise the iterative agentic workflow, we initialize policies via behavioural cloning on filtered trajectories — retaining only rounds 𝑡 where the output admits a well-formed lemma summary (e.g., syntactically valid, non-empty, logically segmented). Let 𝒟init = {(𝑠𝑡,𝑣𝑡)} denote such transitions. The token-level pretraining objective is:

log 𝜋𝜃𝐿(𝑣𝑡,𝜏 |𝑠𝑡,𝑣𝑡,<𝜏)]︃. (4)

[︃ 𝑇

∑︁

𝑡

ℒRFT(𝜃) = −E(𝑠

𝑡,𝑣𝑡)∼𝒟init

𝜏=1

Notably, we continuously augment 𝒟init with question-answer pairs that are filtered by outcome-based scoring, without previous thinking. We observe that the model exhibits emergent generalization: patterns learned from these simplified trajectories boost agentic solving of the same problems, thereby improving the efficiency of positive trajectory discovery during online RL.

###### 4.3. Oreal with Conjugate Reward under Process Judgement

We adopt the reinforcement learning framework of Oreal for policy optimization, and introduce two critical adaptations tailored to our Hierarchical MDP setting: (1) credit assignment across high-level reasoning actions is non-trivial due to delayed rewards; (2) the Process Verifier (PV) introduces a continuous, noisy reward signal that deviates from the binary outcome supervision assumed in the RLVR setting.

###### 4.3.1. Progress-Conditioned Advantage via Lemma Dependency Graphs

Existing RLVR training predominantly targets outcome verification (e.g., final answer correctness), which proves insufficient for complex mathematical tasks requiring high process supervision. To align optimization with granular reasoning fidelity, we assign sparse reward signals across reasoning rounds, akin to performing round-level temporal differencing to minimize advantage estimation variance.

To rigorously quantify intermediate progress, we introduce a lemma dependency graph by aggregating reasoning states across multiple rollouts of the same problem. This graph structure captures the probabilistic contribution of specific lemmas to the final proof. Such mechanism functions as a computationally efficient surrogate to Monte Carlo Tree Search (MCTS), providing high-quality value estimation without the prohibitive overhead of extensive search. An example of the lemma graph is presented in Appendix. E

Within this topology, the value of a specific lemma 𝑙 is not isolated but structurally coupled with the proof’s progression. We define the value of a lemma recursively as the expected value of its subsequent derived lemmas, effectively backpropagating the success probability from the final answer to intermediate steps:

𝑣(𝑙) = E𝑙′∈Succ(𝑙) [𝑣(𝑙′)], (5) where Succ(𝑙) denotes the set of valid lemmas derived directly from 𝑙 in the dependency graph.For the policy optimization, we anchor credit to rounds that yield verifiable advances. Specifically, for a reasoning round 𝑡 that generates a set of candidate lemmas ℒ𝑡, we adopt an optimistic value estimation strategy. We define the state value of round 𝑡 as the maximum value among its generated candidates, 𝑉 (𝑠𝑡) = max𝑙∈ℒ

𝑣(𝑙). The round-level advantage is then computed via the temporal difference error between the best potential of the current round and the next:

𝑡

𝐴𝑡 = 𝑟𝑡 + 𝛾 max

𝑙′∈ℒ𝑡+1

𝑣(𝑙′) − max𝑙 ∈ ℒ𝑡𝑣(𝑙), (6)

where 𝑟𝑡 represents the immediate step reward (e.g., syntactic validity or solving a sub-goal) and 𝛾 is the discount factor.

For intermediate rounds yielding no new lemmas (𝐶𝑡 = 0), the advantage is effectively masked. These formulation ensures that the gradient estimation is driven by the most promising reasoning path discovered at each step, decoupling optimization intensity from trajectory length and effectively filtering out noise from suboptimal branches.

###### 4.3.2. Conjugate Reward Modeling for Noisy Process Verification

Process Verification (PV) offers valuable insight into the internal logical consistency of a generated solution by subjecting its intermediate steps to multiple stochastic checks. However, unlike final-answer correctness—which is deterministic—PV feedback is inherently noisy: a solution passing 𝑘 out of 𝑛 verification rounds does not guarantee superior reasoning quality, as passes may arise from lucky sampling or superficial plausibility rather than deep correctness. Directly using the empirical ratio 𝑘/𝑛 as a reward signal risks amplifying this noise, leading to unstable or misguided policy updates that overfit to verification artifacts rather than genuine mathematical rigor.

To address this, we adopt a Bayesian perspective and model the latent reasoning quality 𝑝 ∈ [0,1] as a random variable. We place a uniform prior 𝑝 ∼ Beta(1,1), encoding no initial assumption about solution validity. After observing 𝑘 successful verifications in 𝑛 independent PV trials, the conjugate Beta-Bernoulli update yields the posterior:

𝑝 | (𝑘,𝑛) ∼ Beta(𝑘 + 1,𝑛 − 𝑘 + 1). (7)

Instead of using point estimates (e.g., posterior mean), we define the reward as the probability that this solution is strictly better than a canonical “completely invalid” baseline—one that fails all 𝑛 checks (𝑘 = 0). Let 𝑝1 ∼ Beta(𝑘 + 1,𝑛 − 𝑘 + 1) represents the quality of the current solution and 𝑝0 ∼ Beta(1,𝑛 + 1) that of the baseline. The reward is then:

###### 𝑅(𝑘,𝑛) = P(𝑝1 > 𝑝0) = ∫︁ 1

∫︁ 1

I(𝑝1 > 𝑝0) · 𝑓Beta(𝑘+1,𝑛−𝑘+1)(𝑝1) · 𝑓Beta(1,𝑛+1)(𝑝0)𝑑𝑝1𝑑𝑝0. (8)

0

0

This formulation provides a principled, probabilistically calibrated reward that accounts for uncertainty in the verification process. It naturally suppresses spurious signals from low-pass outcomes while preserving strong gradients for high-confidence valid solutions.

In practice, we fix 𝑛 = 4, balancing verification cost and signal fidelity. Under this setting, 𝑅(4,4) ≈ 5.5, corresponding to a 99.5% dominance probability over the 𝑅(0,4) = 0 baseline, with smoothly interpolated rewards for intermediate cases (𝑘 = 1,2,3). By grounding the reward in a relative, distributional comparison rather than raw counts, our conjugate reward model effectively denoises PV feedback, ensuring that policy optimization aligns with latent reasoning quality rather than stochastic verification artifacts. This enables stable and meaningful reinforcement learning even in the presence of imperfect process-level supervision. The complete RL training process is demonstrated in Algorithm 1.

### 5. Experiment

###### 5.1. Experiment Setup

Implementation. We collect a set of problems from Art of Problem Solving (AoPS)3 and in-house datasets as cold-start data, whose domain across middle school, university, and competition-level, including both solution-based and proof-based questions. We generate candidate trajectories using a variant of Intern-S1 [2], then employ the CompassVerifier [17] and OPV [34] as the judger for solution-based and proof-based questions, respectively. Simultaneously, we chose a portion of the challenging problems as RL data, based on the pass rate of Intern-S1 on those data. Finally, built on Intern-S1 [2], we developed Intern-S1-MO, the multi-agent system

3https://artofproblemsolving.com/community

- Table 1: Overall evaluation results for Intern-S1-MO and each baseline. Here, the AIME2025 and HMMT2025 scores for the baseline models (first six rows) are from their respective technical reports or corresponding results in Matharena. For IMO2025, we report the pass@4 score, while the remaining benchmarks report the pass@1 score. Bold represents the best performance.

###### Model HMMT2025 AIME2025 CNMO2025 IMO2025

Gemini2.5-pro 82.5 83 157.5 14 o3-high 77.5 88.9 138.5 12.5 Grok4 92.5 91.7 84 4 GPT-OSS-120B 90 92.5 130 11 DeepSeek-R1-0528 76.67 87.5 113.5 6.5 Qwen3-235B-A22B 60.4 81.5 109 14

Intern-S1-mini-MO 79.2 87.3 176.3 17 Intern-S1-MO 95 96.6 232.4 26

solving complex reasoning problems through hierarchical decomposition. Subsequently, by distilling it, we built a lite system based on Intern-S1-Mini, called Intern-S1-mini-MO.

Evaluation. We use some well-established mathematical datasets for evaluation, such as AIME2025 [20], HMMT2025 Feb [10], IMO2025, CNMO2025, and CMO2025. For CNMO2025 and IMO2025, we only evaluate the non-geometric parts. Referring to the approach of MathArena [3], we build an evaluation system that scores the answer based on fine-grained grading points (details in Appendix D), and we employ it in the evaluation for IMO2025 and CNMO2025. For each sample, we perform 16 independent rollouts and use the unbiased pass@1 [5] as the metric, except for IMO2025, which we use pass@4. For CMO2025, we officially participate in the competition and conduct the test under the same time limit and grading standards as human contestants. We report the CMO2025 results separately at Section 5.4.

Baseline. We conduct evaluations against several baselines, including Gemini2.5-pro [6], o3-high [23], Grok4 [35], GPT-OSS-120B [24], DeepSeek-R1-0528 [9], and Qwen3-235B-A22B [36]. For some benchmarks (AIME2025 and HMMT2025), we report the scores of such baseline models from their respective technical reports or corresponding results from Matharena.

###### 5.2. Overall Results

The quantitative results, summarised in Table 1, reveal a distinct performance hierarchy where our proposed framework significantly outperforms current state-of-the-art baselines. The parameter-efficient variant, InternS1-mini-MO, exhibits exceptional reasoning density. It surpasses all closed-source and open-weights baselines on the highly challenging CNMO2025 benchmark (scoring 176.3 compared to Gemini 2.5 Pro’s 157.5) and achieves a score of 17 on IMO2025. This result suggests that the performance gains are primarily attributable to our architectural innovations, and offers compelling evidence that complex mathematical reasoning can be achieved with favorable inference-time efficiency.

Analyzing performance deltas across benchmarks reveals a qualitative divergence in problem-solving requirements. On relatively standard competition sets like HMMT2025 and AIME2025, the gap between strong baselines and our method is present but narrower. We hypothesize that performance in these regimes is partially saturated by models capable of pattern matching and heuristic retrieval from pre-training data. On CNMO2025 and IMO2025, whose problems demand the construction of novel proof paths and the synthesis of auxiliary lemmas. Intern-S1-MO excels here precisely because it maintains a persistent logical state across rounds. Unlike single-pass models that must restart reasoning from scratch upon failure, our agent accumulates partial progress (e.g., establishing a necessary inequality or isolating a geometric invariant), effectively simulating the "scratchpad" utility used by human experts.

The performance on IMO2025 warrants specific contextualization. A score of 26 places Intern-S1-MO within the top percentile of global human competitors, outperforming the national team averages of most participating countries. Preliminary error analysis indicates that the remaining deficit largely stems from problems requiring highly idiosyncratic transformations or "spark-of-insight" constructions that elude systematic search. Collectively, these findings demonstrate that while parameter scale provides a necessary foundation,

- Table 2: Ablation study results. Here, “Single-round with Agents” means that only one round of inference is performed in the agent system, which is the left part of Fig 2. “+ Multi-round Reasoning” means performing a full multi-round reasoning, but without providing scores for intermediate lemmas and a revised final loop. “+ Theorem Verifier” means providing the confidence score for the intermediate lemma, that is, which is the left and middle part of Fig 2. “+ Process Verifier” means the overall inference workflow. And “+ OReal-H” means the agents are trained by the RL algorithm introduced in Section 4.

Model HMMT2025 AIME2025 CNMO2025

Single-round with Agents 70.8 81.9 178.0 + Multi-round Reasoning 85.4 91.0 201.7 + Theorem Verifier 86.3 93.3 203.0 + Process Verifier 89.1 94.0 215.2 + OReal-H 95.0 96.6 232.4

the transition from competency to mastery in Olympiad-level mathematics requires a structured, verifiable cognitive architecture capable of sustained, multi-step deduction.

5.3. Ablation Study

To better understand the contribution of each key component in Intern-S1-MO, we conduct a systematic ablation study. Due to the limited number of problems in IMO2025 (only five), which brings the volatile results, we compare the evaluation results on HMMT2025, AIME2025, and CNMO2025.

As described in Section 3 and Section 4, the architecture of Intern-S1-MO integrates several components, including multi-round reasoning with lemma search and summary, lemma verification, process validation, and an RL framework for training the LRM using the online explored trajectories. However, it is crucial to disentangle their individual impacts to validate design choices and assess whether performance gains stem from architectural sophistication or synergistic interactions among modules. Therefore, we incrementally build up the full agent system from a simplified baseline, called “Single-round with Agents”, which means that only one round of inference is performed in the agent system. Then we progressively add the corresponding component.

As shown in Table 2, we add each component step by step, where “Single-round with Agents” represents the left part of Fig 2, “+ Multi-round Reasoning” represents the left and middle part of Fig 2 without providing scores for intermediate lemmas, “+ Theorem Verifier” represents the reasoning pattern with the scored lemma, “+ Process Verifier” represents the overall inference workflow, and “+ OReal-H” represents the agent system trained by the RL algorithm.

The gradual addition of the modules steadily increases the pass@1 scores of Intern-S1-MO on each benchmarks, proving every constituent component within the proposed framework serves a non-redundant function. Ultimately, compared to the initial baseline, our method improves the score in CNMO2025 from 178.0 to 232.4 and also achieves gains on HMMT2025 and AIME2025.

5.4. CMO2025 Official Participant

- Table 3: CMO2025 evaluation results. CMO2025 have six questions in total, each with a maximum score of 21 points, for a total of 126 points. The scores for each question are listed below. Our system achieved full marks on four questions and partial solves the other two questions.

###### Total Score P1 P2 P3 P4 P5 P6

102 21 21 9 21 21 9

To evaluate Intern-S1-MO in a real-world environment, we officially participate in CMO2025. Similar to human students, our system completed six questions in two days, with a limit of 4.5 hours per day to solve three questions and submit the solutions to the committee immediately. These solutions are scored by human experts using the same standards as those used for human contestants.

We participated in the competition using an extended search budget based on test-time scaling, achieving better results within the given time constraints. For each problem, we performed a 256-shot parallel search over up to 12 rounds. For intermediate lemmas, a lemma verifier provided multiple rounds of 8-shot feedback to help assess and refine their correctness. Upon obtaining candidate solutions, we applied an 8-shot refinement procedure comprising 24 rounds, in which, at each round, the OPV verifier identified informalities or gaps in the proof, which the policy model subsequently revised.

As shown in Table 3, our system achieves a score of 102 out of 126, exceeding the gold medal threshold of 78 points. This signifies that Intern-S1-MO not only matches the logical rigor and reasoning ability of top-tier high school math olympiad participants but also transcends the limitations of human problem-solving patterns by independently exploring to discover novel solution methods.

### 6. Conclusion

This paper aims to address the critical bottleneck in large reasoning models (LRMs) for complex mathematical reasoning: the inherent limitation of context length, which has hindered progress in solving ultra-challenging tasks such as International Mathematical Olympiad (IMO) problems. To this end, this paper introduces InternS1-MO, an LRM-driven multi-agent system that conducts multi-round hierarchical reasoning, which conducts reasoning, summary, and verification at each round. By maintaining a compact memory in the form of lemmas, Intern-S1-MO can more freely explore the lemma-rich reasoning spaces in multiple reasoning rounds, which significantly extends the 64K constraints of LRMs by about 8 times. We further propose OREAL-H, an RL framework for training the LRM to simultaneously bootstrap the reasoning ability of the LRM and elevate the overall performance of Intern-S1-MO. Intern-S1-MO can now solve problems that require humans to think about 1.5 hours, which eventually obtains 26 out of 35 points on the non-geometry problems of IMO2025, matching the performance of silver medalists. We wish the work paves the way for future research that adopts LRMs for mathematical research.

### References

- [1] Pranjal Aggarwal and Sean Welleck. L1: Controlling how long a reasoning model thinks with reinforcement learning. ArXiv, abs/2503.04697, 2025. URL https://api.semanticscholar.org/CorpusID:

276813519. 1

- [2] Lei Bai, Zhongrui Cai, Yuhang Cao, Maosong Cao, Weihan Cao, Chiyu Chen, Haojiong Chen, Kai Chen, Pengcheng Chen, Ying Chen, et al. Intern-s1: A scientific multimodal foundation model. arXiv preprint arXiv:2508.15763, 2025. 1, 5.1
- [3] Mislav Balunovi’c, Jasper Dekoninck, Ivo Petrov, Nikola Jovanovi’c, and Martin T. Vechev. Matharena: Evaluating llms on uncontaminated math competitions. ArXiv, abs/2505.23281, 2025. URL https: //api.semanticscholar.org/CorpusID:278996037. 2.1, 5.1, D
- [4] Luoxin Chen, Jinming Gu, Liankai Huang, Wenhao Huang, Zhicheng Jiang, Allan Jie, Xiaoran Jin, Xing Jin, Chenggang Li, Kaijing Ma, et al. Seed-prover: Deep and broad reasoning for automated theorem proving. arXiv preprint arXiv:2507.23726, 2025. 1
- [5] Mark Chen. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021. 5.1
- [6] Google DeepMind. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. ArXiv, abs/2507.06261, 2025. URL https://api. semanticscholar.org/CorpusID:280151524. 1, 5.1
- [7] Google DeepMind. Advanced version of gemini with deep think officially achieves gold-medal standard at the international mathematical olympiad, 2025. URL https://deepmind.google/discover/blog/ advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/. 1

- [8] Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Minlie Huang, Nan Duan, and Weizhu Chen. Tora: A tool-integrated reasoning agent for mathematical problem solving. ArXiv, abs/2309.17452,

2023. URL https://api.semanticscholar.org/CorpusID:263310365. 2.1

- [9] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 5.1
- [10] Harvard–MIT Mathematics Tournament. Hmmt february problem archive. https://www.hmmt.org/ www/tournaments/testing. Accessed 9 May 2025. 5.1
- [11] Kaixuan Huang, Jiacheng Guo, Zihao Li, Xiang Ji, Jiawei Ge, Wenzhe Li, Yingqing Guo, Tianle Cai, Hui Yuan, Runzhe Wang, Yue Wu, Ming Yin, Shange Tang, Yangsibo Huang, Chi Jin, Xinyun Chen, Chiyuan Zhang, and Mengdi Wang. Math-perturb: Benchmarking llms’ math reasoning abilities against hard perturbations. ArXiv, abs/2502.06453, 2025. URL https://api.semanticscholar.org/CorpusID:

276249117. 2.1

- [12] Yichen Huang and Lin F Yang. Gemini 2.5 pro capable of winning gold at imo 2025. arXiv preprint arXiv:2507.15855, 2025. 1, 2.1
- [13] Zixuan Ke, Fangkai Jiao, Yifei Ming, Xuan-Phi Nguyen, Austin Xu, Do Xuan Long, Minzhi Li, Chengwei Qin, PeiFeng Wang, Silvio Savarese, Caiming Xiong, and Shafiq Joty. A survey of frontiers in llm reasoning: Inference scaling, learning to reason, and agentic systems. Trans. Mach. Learn. Res., 2025, 2025. URL https://api.semanticscholar.org/CorpusID:277781085. 1
- [14] Jan Hendrik Kirchner, Yining Chen, Harri Edwards, Jan Leike, Nat McAleese, and Yuri Burda. Proververifier games improve legibility of llm outputs, 2024. URL https://arxiv.org/abs/2407.13692. 2.2
- [15] Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yutao Zhu, Yongkang Wu, Ji-Rong Wen, and Zhicheng Dou. Webthinker: Empowering large reasoning models with deep research capability. ArXiv, abs/2504.21776, 2025. URL https://api.semanticscholar.org/CorpusID:278207550. 1
- [16] Xuefeng Li, Haoyang Zou, and Pengfei Liu. Torl: Scaling tool-integrated rl. arXiv preprint arXiv:2503.23383, 2025. 2.2
- [17] Shudong Liu, Hongwei Liu, Junnan Liu, Linchen Xiao, Songyang Gao, Chengqi Lyu, Yuzhe Gu, Wenwei Zhang, Derek F Wong, Songyang Zhang, et al. Compassverifier: A unified and robust verifier for llms evaluation and outcome reward. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 33454–33482, 2025. 5.1
- [18] Chengqi Lyu, Songyang Gao, Yuzhe Gu, Wenwei Zhang, Jianfei Gao, Kuikun Liu, Ziyi Wang, Shuaibin Li, Qian Zhao, Haian Huang, et al. Exploring the limit of outcome reward for learning mathematical reasoning. arXiv preprint arXiv:2502.06781, 2025. 1, 27
- [19] Xinji Mai, Haotian Xu, Zhong-Zhi Li, Xing W, Weinong Wang, Jian Hu, Yingying Zhang, and Wenqiang Zhang. Agent rl scaling law: Agent rl with spontaneous code execution for mathematical problem solving,

2025. URL https://arxiv.org/abs/2505.07773. 2.2

- [20] Mathematical Association of America. American invitational mathematics examination (aime) problems and solutions. https://maa.org/student-programs/amc/. Accessed 9 May 2025. 5.1
- [21] Sumeet Ramesh Motwani, Chandler Smith, Rocktim Jyoti Das, Markian Rybchuk, Philip Torr, Ivan Laptev, Fabio Pizzati, Ronald Clark, and Christian Schröder de Witt. Malt: Improving reasoning with multi-agent llm training. ArXiv, abs/2412.01928, 2024. URL https://api.semanticscholar.org/CorpusID:

274446212. 1

- [22] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling. arXiv preprint arXiv:2501.19393, 2025. B.1

- [23] OpenAI. Introducing openai o3 and o4-mini, 2025. URL https://openai.com/index/ introducing-o3-and-o4-mini/. 5.1
- [24] OpenAI. gpt-oss-120b & gpt-oss-20b model card, 2025. URL https://arxiv.org/abs/2508.10925. 5.1
- [25] OpenAI. Openai imo 2025 proofs, 2025. URL https://github.com/aw31/ openai-imo-2025-proofs. 1
- [26] Aske Plaat, Annie Wong, Suzan Verberne, Joost Broekens, Niki van Stein, and Thomas H.W. Back. Multistep reasoning with large language models, a survey. 2024. URL https://api.semanticscholar. org/CorpusID:271218853. 2.1
- [27] Z. Z. Ren, Zhihong Shao, Jun-Mei Song, Huajian Xin, Haocheng Wang, Wanjia Zhao, Liyue Zhang, Zhe Fu, Qihao Zhu, Dejian Yang, Z. F. Wu, Zhibin Gou, Shirong Ma, Hongxuan Tang, Yuxuan Liu, Wenjun Gao, Daya Guo, and Chong Ruan. Deepseek-prover-v2: Advancing formal mathematical reasoning via reinforcement learning for subgoal decomposition. ArXiv, abs/2504.21801, 2025. URL https: //api.semanticscholar.org/CorpusID:278207693. 1
- [28] Ning Shang, Yifei Liu, Yi Zhu, Li Lyna Zhang, Weijiang Xu, Xinyu Guan, Buze Zhang, Bingcheng Dong, Xudong Zhou, Bowen Zhang, Ying Xin, Ziming Miao, Scarlett Li, Fan Yang, and Mao Yang. rstar2-agent: Agentic reasoning technical report, 2025. URL https://arxiv.org/abs/2508.20722. 2.2
- [29] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Jun-Mei Song, Mingchuan Zhang, Y. K. Li, Yu Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. ArXiv, abs/2402.03300, 2024. URL https://api.semanticscholar.org/CorpusID:267412607. 1, 2.1
- [30] Chuming Shen, Wei Wei, Xiaoye Qu, and Yu Cheng. Satori-r1: Incentivizing multimodal reasoning with spatial grounding and verifiable rewards. arXiv preprint arXiv:2505.19094, 2025. 2.2
- [31] Jiankai Sun, Chuanyang Zheng, Enze Xie, Zhengying Liu, Ruihang Chu, Jianing Qiu, Jiaqi Xu, Mingyu Ding, Hongyang Li, Mengzhe Geng, et al. A survey of reasoning with foundation models: Concepts, methodologies, and outlook. ACM Computing Surveys, 57(11):1–43, 2025. 1, 2.1
- [32] Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, and Ee-Peng Lim. Plan-andsolve prompting: Improving zero-shot chain-of-thought reasoning by large language models. In Annual Meeting of the Association for Computational Linguistics, 2023. URL https://api.semanticscholar. org/CorpusID:258558102. 1
- [33] Pengyuan Wang, Tian-Shuo Liu, Chenyang Wang, Yidi Wang, Shu Yan, Cheng-Xing Jia, Xu-Hui Liu, Xin-Wei Chen, Jia-Cheng Xu, Ziniu Li, and Yang Yu. A survey on large language models for mathematical reasoning. ArXiv, abs/2506.08446, 2025. URL https://api.semanticscholar.org/CorpusID:279261286. 1
- [34] Zijian Wu, Lingkai Kong, Wenwei Zhang, Songyang Gao, Yuzhe Gu, Zhongrui Cai, Tianyou Ma, Yuhong Liu, Wang Zhi, Runyuan Ma, Guangyu Wang, Wei Li, Conghui He, Dahua Lin, and Kai Chen. OPV: Outcome-based process verifier for efficient long chain-of-thought verification. 2025. 3, 5.1
- [35] xAI. Grok 4, 2025. URL https://x.ai/news/grok-4. 5.1
- [36] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 1, 5.1
- [37] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. ArXiv, abs/2305.10601, 2023. URL https://api.semanticscholar.org/CorpusID:258762525. 2.1
- [38] Yurun Yuan and Tengyang Xie. Reinforce llm reasoning through multi-agent reflection. ArXiv, abs/2506.08379, 2025. URL https://api.semanticscholar.org/CorpusID:279260670. 2.1

- [39] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? ArXiv, abs/2504.13837, 2025. URL https://api.semanticscholar.org/CorpusID:277940134. 1
- [40] Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerlzoo: Investigating and taming zero reinforcement learning for open base models in the wild. ArXiv, abs/2503.18892, 2025. URL https://api.semanticscholar.org/CorpusID:277940848. 1
- [41] Dan Zhang, Sining Zhoubian, Yisong Yue, Yuxiao Dong, and Jie Tang. Rest-mcts*: Llm self-training via process reward guided tree search. ArXiv, abs/2406.03816, 2024. URL https://api.semanticscholar. org/CorpusID:270285630. 1, 2.1
- [42] Jianyi Zhang, Yufan Zhou, Jiuxiang Gu, Curtis Wigington, Tong Yu, Yiran Chen, Tong Sun, and Ruiyi Zhang. Artist: Improving the generation of text-rich images with disentangled diffusion models and large language models, 2024. URL https://arxiv.org/abs/2406.12044. 2.2
- [43] Zhuosheng Zhang, Aston Zhang, Mu Li, and Alexander J. Smola. Automatic chain of thought prompting in large language models. ArXiv, abs/2210.03493, 2022. URL https://api.semanticscholar.org/ CorpusID:252762275. 1
- [44] Chujie Zheng, Zhenru Zhang, Beichen Zhang, Runji Lin, Keming Lu, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. ProcessBench: Identifying Process Errors in Mathematical Reasoning. arXiv e-prints, art. arXiv:2412.06559, December 2024. doi: 10.48550/arXiv.2412.06559. 3
- [45] Denny Zhou, Nathanael Scharli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Olivier Bousquet, Quoc Le, and Ed H. Chi. Least-to-most prompting enables complex reasoning in large language models. ArXiv, abs/2205.10625, 2022. URL https://api.semanticscholar.org/ CorpusID:248986239. 1
- [46] Yichi Zhou, Jianqiu Zhao, Yongxin Zhang, Bohan Wang, Siran Wang, Luoxin Chen, Jiahui Wang, Haowei Chen, Allan Jie, Xinbo Zhang, Haocheng Wang, Luong Ngoc Trung, Rong Ye, Phan Nhat Hoang, Huishuai Zhang, Peng Sun, and Hang Li. Solving formal math problems by decomposition and iterative reflection. ArXiv, abs/2507.15225, 2025. URL https://api.semanticscholar.org/CorpusID:280271297. 1
- [47] Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui, Xuekai Zhu, Haozhan Li, Yuchen Zhang, Xinwei Long, Ermo Hua, et al. Ttrl: Test-time reinforcement learning. arXiv preprint arXiv:2504.16084,

2025. 2.2

### The Use of Large Language Models (LLMs)

We used LLMs solely for language polishing. The scientific ideas, methodology, analyses, and conclusions were entirely developed by the authors, while the LLMs assisted only in improving clarity and readability of the text.

### A. System Prompts for Math Agents

Our workflow primarily comprises iterative policy lemma search and summarisation, alongside corresponding lemma and final answer verification. Following the final answer verification, the policy model will undergo iterative refinement based on feedback. The prompts for these five actions are presented as follows:

###### A.1. Lemma Search

- Listing 1: Lemma Search

**Objective:** Your task is to provide a rigorous mathematical proof and solution for the given problem.

˓→ The problem is expected to be challenging. Your primary goal is to demonstrate a ˓→ deep and correct understanding of the problem through logical, step-by-step ˓→ reasoning.

**Guiding Principles:**

- 1. **Rigor is Paramount:**

- * Every step in your proof must be logically sound and clearly justified.

- * The final answer is secondary to the correctness of the derivation. A correct ˓→ answer resulting from a flawed or incomplete proof will be considered a failure.

- 2. **Embrace Partial Solutions:**

- * It is understood that a complete solution may not be found in a single attempt.

- * If you cannot provide a complete solution, you must provide any significant ˓→ partial results that you can prove with full rigor.

- * **Do not guess or provide solutions with logical gaps.** Instead, focus on what ˓→ you *can* prove.

- * Examples of valuable partial results include:

- * Proving a key lemma.

- * Solving one or more cases of a proof by cases.

- * Establishing a critical property of the mathematical objects involved.

- * For an optimization problem, proving an upper or lower bound.

- * Clearly state which parts of the problem you have solved and which remain open. ˓→ Acknowledging the limits of your solution is a critical part of the task.

- 3. **Mathematical Formatting:**

* All mathematical variables, expressions, equations, and relations must be ˓→ formatted using TeX. For example: ‘Let $G$ be a group and let $H$ be a subgroup of ˓→ $G$.‘

**Output Format:** Your response MUST be structured into the following sections, in this exact order.

---

**1. Summary**

**a. Verdict:**

- * Begin by stating clearly whether you have found a complete or a partial solution.

- * **For a complete solution:** State the final answer. (e.g., "I have found a complete ˓→ solution. The answer is...")

- * **For a partial solution:** Clearly state the main rigorous conclusion(s) you have ˓→ proven (for example: "I have not found a complete solution, but I have rigorously

˓→ proven the following:"). Your output must strictly follow the Markdown and LaTeX ˓→ formatting guidelines below: - **Format for Proven Lemmas:**

- All **proven lemmas** and their proofs should be placed together inside a single ˓→ ‘\boxed{}‘ environment.

- - Use ‘---‘ horizontal lines to separate different lemmas.

- - Each lemma should begin with ‘**Lemma X:**‘, where ‘X‘ is a positive integer.

- - State each lemma concisely and formally, using LaTeX as appropriate.

- - The proof should immediately follow, starting with ‘**Proof X:**‘.

- - Each step of the proof should use an unordered list (‘*‘), and each step should

˓→ begin with ‘**Step Y:**‘.

- **Format for Unproven Lemmas:**

- All **unproven lemmas** should be placed together in a separate ‘\boxed{}‘ ˓→ environment.

- - Each lemma should begin with ‘**Lemma X:**‘.

- - If all key steps are already provided in the "Provided Lemmas" section or have

˓→ been fully proven (i.e., **no new unproven lemmas are found**), simply include ‘** ˓→ Lemma -1**‘ in this box.

- - **Example Output Format:** ‘‘‘ \boxed{

**lemma n+1**:{lemma n+1}

- **proof n+1**:

- *step 1:{step 1}

- *step 2:{step 2}

- *step 3:{step 3}

---

- **lemma n+2**:{lemma n+2}

- **proof n+2**:

- *step 1:{step 1}

... } \boxed{

- **withoutproof**:

**lemma -1** } ‘‘‘

- After outputting the lemmas, you should end your response immediately without ˓→ proceeding to the subsequent sections.

- **b. Method Sketch:**

- * Provide a high-level, conceptual outline of your logical argument. This should be ˓→ clear enough for an expert to grasp your approach without reading the full proof.

- * Include:

- * A narrative of your overall strategy.

- * The full and precise mathematical statements of any key lemmas or major ˓→ intermediate results you proved.

- * A description of any key constructions or case splits that form the backbone of ˓→ your argument.

- **2. Detailed Solution**
- * Present the full, step-by-step mathematical proof of your results.

- * This section should contain *only* the rigorous proof itself, free from any commentary, ˓→ reflections on your process, or alternative approaches you considered.

- * The level of detail must be sufficient for an expert to verify the correctness of your ˓→ reasoning without needing to fill in any gaps.

###### A.2. Lemma Summarization

- Listing 2: Lemma Summarization

You are a top-tier mathematical research assistant, proficient in the logical analysis and

˓→ argumentation of high-level competitive mathematics.

Your core task is to conduct an in-depth analysis of a solution approach generated by a ˓→ large language model for problems at the International Mathematical Olympiad (IMO) ˓→ level, identifying and extracting all key lemmas.

During this analysis, you must rigorously distinguish between propositions **newly ˓→ proposed** by the model and **universal lemmas** already provided by us. Your final ˓→ output **shall only contain** those lemmas appearing in the model’s solution ˓→ approach but not provided in the universal lemma repository.

**The input comprises three sections:**

- 1. ‘### Problem ###‘: The mathematical problem requiring resolution.

- 2. ‘### Provided Lemmas ###‘: A set of known, proven lemmas for reference during problem˓→ solving.

- 3. ‘### Model’s Thinking Process ###‘: The reasoning process generated by the large ˓→ language model to solve the problem.

**Your output must adhere to the following principles and format:**

- #### **A. Extraction Principles**

- 1. **Novelty**: Extract only lemmas first introduced or proven within the ‘Model’s ˓→ Thinking Process‘. Do not include lemmas from the ‘Provided Lemmas‘ if the model ˓→ utilises them.

- 2. **Classification**: Extract only new lemmas satisfying the following conditions:

* **Proven Lemmas**: Propositions explicitly stated or implicitly utilised within ˓→ the ‘model’s problem-solving approach‘, accompanied by a complete or core proof.

- #### **B. Strict Formatting Requirements** Your output must strictly adhere to the following Markdown and LaTeX formatting.

1. **Format for Proven Lemmas:**

- * Each **proven lemma** and its proof must be placed within a separate, non-nested ˓→ ‘<lemma>...</lemma>‘ environment, with the opening and closing tags each occupying a ˓→ distinct line. The number of ‘<lemma>...</lemma>‘ environments must match the ˓→ number of lemmas extracted in this round. Note that input lemmas may not be ˓→ presented in this format.

- * Each lemma must begin on a new line with the text ‘\n**Lemma X (Lemma X):**‘, ˓→ where ‘X‘ is a positive integer numbering. The opening line of this lemma must ˓→ occupy a complete line. You must strictly adhere to this format. If the lemma has ˓→ any additional names, annotations, or other descriptions, you may append explanatory ˓→ text enclosed in brackets after ‘\n**Lemma X (Lemma X):**‘, e.g., ‘\n**Lemma 2 ( ˓→ Lemma 2):**(Dilworth’s Theorem)‘.

- * Subsequently, the remainder of this line must strictly state the content of the ˓→ lemma. This requires a complete exposition of the lemma extracted from the ‘Model ˓→ Problem-Solving Approach‘. As the ‘Model Problem-Solving Approach‘ frequently ˓→ introduces entirely new symbols and notations, you must rigorously provide their ˓→ definitions. Should these definitions involve existing lemmas, you must also specify ˓→ which particular definitions from those existing lemmas are being referenced.

- * The statement of the lemma should employ concise, formal mathematical language, ˓→ utilising LaTeX where appropriate.

- * This is immediately followed by the proof, commencing on a new line with ‘**Proof ˓→ X:**‘.

- * Each step of the proof should begin with an unordered list ‘*‘ and be prefixed ˓→ with ‘**Step Y:**‘. Each step should occupy a separate line. Where an existing lemma

- ˓→ is referenced in the proof, you must explicitly state which existing lemma is being ˓→ referenced and how it is being applied.
- * The positive integer numbering for each round should be the largest number in the ˓→ general lemma library incremented by 1. Note that some lemmas in the general lemma ˓→ library are corrected versions of others. These corrected lemmas share the same ˓→ numbering as the original lemma, but are marked with the suffix ‘-fixed‘.

Below is a sample input-output pair: ### Problem Statement (Problem) ### {Problem} ### Provided Lemmas (Provided Lemmas) ### <lemma>

**Lemma 1 (Lemma 1)**:

**Proof 1 (Proof 1):**: </lemma>

---...

--<lemma>

**Lemma n**:

**Proof n:**: </lemma>

### Model’s Thinking Process ### {Thinking}

--####‘DESIREDOUTPUT: <lemma>

**Lemma n+1:** lemma n+1

- **Proof n+1:**:

- * **Step 1:** step 1

- * **Step 2:** step 2

- * **Step 3:** step 3 </lemma> <lemma>

- **Lemma n+2:** lemma n+2

- **Proof n+2:**:

- * **Step 1:** step 1

- * **Step 2:** step 2

- * **Step 3:** step 3 </lemma>

...

###### A.3. Lemma Verify

- Listing 3: Lemma Summarization

You are a mathematics and logic expert. Your task is to evaluate the correctness of a ˓→ newly proposed lemma. This lemma relates to the main mathematical problem and may ˓→ rely on a provided library of existing lemmas.

Your goal is to meticulously check the proof of the new lemma, step by step, to identify ˓→ the index of the first incorrect step. The index starts at 0 for the first step. If ˓→ the proof is entirely correct, you should output -1.

Instructions:

- - You will be given:

- 1. The Main Question: The overarching problem providing context.

- 2. Provided Lemmas: A library of existing statements assumed to be correct.

- 3. The New Lemma and Its Proof: The student’s work to be evaluated, with the proof ˓→ skeleton broken down into steps.

- - A key part of your evaluation is to verify that any use of a lemma from the Provided ˓→ Lemmas library is correctly applied and that its preconditions are satisfied. The ˓→ logical inferences within the proof must be sound and either self-evident, derived ˓→ from the main question’s conditions, or justified by one of the provided lemmas.

- - You must perform a step-by-step check of the entire solution. Present this analysis as a ˓→ Detailed Verification Log:

- - Use a numbered list; each item corresponds to a step in the student’s proof.

- - For correct steps, provide a brief justification.

- - For steps with errors or gaps, provide a detailed explanation.

- - Do not use a table.

- - Finally, at the conclusion of your response, always include a First Error, formatted as ˓→ ‘\box{{STEPk}}‘, where ‘k‘ denotes the index of the first incorrect step. For ˓→ instance, if step 2 is incorrect, respond with ‘\box{{STEP2}}‘. Should all steps be ˓→ correct, respond with ‘\box{{STEP-1}}‘.

- - The new lemma to verify is guaranteed to possess both a proposition and a proof skeleton. ˓→ Should any component be missing (thus rendering it an invalid lemma), directly ˓→ output ‘FORMAT_ERROR‘ followed by a description of the observed error. In such ˓→ instances, no further output is required; omit the ‘\box{{STEP}}‘ indicator.

- --### Question ###: {Question}

### Historical Lemma Repository (Provided Lemmas) ### {ProvidedLemmas}

### New Proof to Verify ### {NewLemmatoVerify}

### Detailed Verification Log and First Error ###:

###### A.4. Final Answer Verify

- Listing 4: Final Answer Verify

You are a mathematics and educational expert tasked with evaluating the correctness of a

˓→ student’s answer. The student’s solution is broken down into steps, and your goal is ˓→ to identify the index of the first incorrect step. The index starts at zero for the ˓→ first step. If all steps are correct, you should output -1.

Instructions:

- - You will receive a question along with the student’s answer, divided into steps. Each ˓→ step is presented in a separate paragraph.

- - You are encouraged to express your internal reasoning within <think>...</think> tags. ˓→ After presenting your thinking process, you **must** write a summary of your ˓→ evaluation. Finally, at the end of your response, always include an integer within ˓→ \\box{{STEP}}. For example, if step 2 is incorrect, respond with \\box{{STEP2}}. If

- ˓→ all steps are correct, respond with \\box{{STEP-1}}.
- - The student’s answer may involve a number of indexed lemmas with their proofs. If you ˓→ found any of them incorrect, you should report that. For example, if lemma 3 is ˓→ incorrect, respond with \\box{{LEMMA3}} instead of \\box{{STEP3}}. If all lemmas are ˓→ correct, you should them detect any incorrect steps. If everything is fine, respond ˓→ with \\box{{STEP-1}}. Do not count steps inside a lemma. The step index depends on ˓→ the detailed solution part.

- - Some steps may initially appear incorrect but are later corrected in subsequent steps. ˓→ If a reflection or revision is both accurate and reasonable, the step should be ˓→ considered correct. If there are multiple reflections, consider only the final one.

- - In cases where the problem is ambiguous, consider all possible interpretations and ˓→ determine if the student’s response aligns with any of them.

- - Evaluate the entire solution, as some intermediate steps might seem incorrect initially ˓→ but are rectified later, such as dismissing an extraneous root. Ensure you consider ˓→ the entire context and, if necessary, review the steps more than once.

- - The errors to identify can be very subtle, sometimes hiding in the inexplicit ˓→ applications of theorems or conditions. So you should actively checking every small ˓→ logical inferences at a small granularity carefully, either in natural language or ˓→ in formulas.

- - If an error does not affect the overall reasoning, or an gap can be recovered by your ˓→ effort, you should not report them as incorrect.

- - To help you identify the possible errors, every first time you checking a step, you ˓→ should repeat it in case you missed subtle information. Then you should check its ˓→ validity by examing its logical inferences within the step/sentences/subsentences ˓→ one by one.

- - Every step should have solid logical basis. Guessing without proof is not allowed.

- ---

**Question**: {question}

**Reference Answer**: {reference}

**Student’s Answer**: {response}

**First Error**:

###### A.5. Self-improve with Verify Feedback

- Listing 5: Self-improve with Verify Feedback

You are a mathematics and logic expert. Your task is to improve a solution to a math

˓→ Olympaid problem given a presented solution trial some comments about the solution. Your goal is to get a concised, improved solution that solve all errors reasonably pointed

˓→ out by comments, fill all gaps mentioned by comments and defend other issues that ˓→ is defendable. Besides, you should compress the complexity of the solution and try ˓→ your best to make it highschool-level.

Instructions:

- - You will be given:

- 1. The Main Question: The overall problem providing context.

- 2. Provided Solution: The student’s work to be evaluated, with a verdict and the ˓→ complete solution divided into steps.

- 3. Previous Comments: Prior attempts to detect specific types of errors. Treat these as ˓→ helpful guidance rather than authoritative. If they flag something, fix or defend.

- - You must present a improved solution with SAME FORMAT. Typically a solution comes with a ˓→ summary section and a detailed solution section.

- - You are free to decide the idea/approach of your improved solution. You can just fix ˓→ specific issues, restructure certain parts of the previous answer, or even discard ˓→ the original solution if considered as unfixable.

- - You are adviced to use highschool level of math. If you choose to use university level, ˓→ then you should treat the readers as smart hihgschool students with no backgrounds, ˓→ then provide the specific introduction of certain knowledge needed.

- - If the solution to improve contains NO USEFUL INFORMATION (e.g. simply admitting its ˓→ failure). Then you should just return "I have not found a complete solution".

- --### Question {Question}

### Solution to Improve {SolutiontoVerify}

### Previous Comments {PreviousCheckingEfforts}

### Your Improved Solution

### B. Implementation Details

- B.1. Inference Budget

Our agentic system is a scalable framework that allows for custom inference budgets based on problem difficulty. Theoretically, a higher inference budget leads to better performance, which aligns with the core logic of the TTS (test time scaling) strategy [22].

To control evaluation costs, we set some default inference budgets. Specifically, we set the maximum number of inference rounds for the reasoner and summarizer agent to 8, the number of parallel verifications for the theorem verifier to 4 for each lemma, and the maximum number of rounds for final iterative revision based on the process verifier to 8. For the reasoner and summarizer agent, the max length of output is set to 64k.

- B.2. Hyperparameters Details

During training iterations, each batch consists of 64 questions, with 16 rollouts per question. The max length of each rollout trajectory is set to 65536 tokens. Then the correctness of each response is averaged to calculate the pass rate, and questions with an overall pass rate of 0 or 1 are discarded.

For optimization, the policy model is trained with a learning rate of 5𝑒−7. Both models employ a cosine annealing learning rate schedule, decaying to 1/5 of the initial learning rate over time. We optimize both models using the AdamW optimizer. The KL coefficient 𝛽 is set to 0.01.

#### C. OReal-H Algorithm The complete RL training procedure is described in Algorithm 1.

Algorithm 1 Evolution of Hierarchical Math Agents via Lemma Dependency Graphs Require: Policy 𝜋𝜃, Verifier (PV) 𝑉𝜑, Dataset 𝒟, Hyperparameters 𝛾,𝛼 Require: Conjugate Reward function 𝑅conj(𝑘,𝑛) (Eq. 7)

- 1: while not converged do
- 2: Initialize batch buffer ℬ ← ∅
- 3: for each problem 𝑞 ∈ Batch(𝒟) do
- 4: // Step 1: Multi-round Rollout
- 5: Sample 𝐾 trajectories {𝜏(𝑖)}𝐾𝑖=1 using 𝜋𝜃(·|𝑞)
- 6: // Step 2: Process Verification
- 7: Evaluate each trajectory 𝜏(𝑖) via 𝑉𝜑
- 8: Compute final reward 𝑅(𝑖) ← 𝑅conj(𝑘,𝑛)
- 9: // Step 3: Construct Lemma Dependency Graph
- 10: Initialize graph 𝒢 = (𝒱,ℰ) with lemmas from all {𝜏(𝑖)}
- 11: Identify terminal nodes where 𝑅(𝑖) ̸= 0
- 12: Backpropagate values recursively to compute lemma values:
- 13: 𝑣(𝑙) ← E𝑙′∈Succ(𝑙)[𝑣(𝑙′)] ◁ Eq. (4)
- 14: // Step 4: Compute Progress-Conditioned Advantage
- 15: for each trajectory 𝜏(𝑖) and step 𝑡 = 1...𝑇 do
- 16: if 𝐶𝑡(𝑖) = 1 then ◁ Valid progress round
- 17: Estimate state value: 𝑉 (𝑠𝑡) ← max𝑙∈ℒ

𝑡

𝑣(𝑙)

- 18: Compute next-state value: 𝑉 (𝑠𝑡+1) ← max𝑙′∈ℒ𝑡+1 𝑣(𝑙′)
- 19: Calculate advantage 𝐴(𝑡𝑖) using TD error:
- 20: 𝐴(𝑡𝑖) ← 𝑟𝑡 + 𝛾𝑉 (𝑠𝑡+1) − 𝑉 (𝑠𝑡) ◁ Eq. (5)
- 21: else
- 22: 𝐴(𝑡𝑖) ← 0 ◁ Mask non-progress rounds
- 23: end if
- 24: Add (𝑠𝑡,𝑢𝑡,𝐴(𝑡𝑖)) to ℬ
- 25: end for
- 26: end for
- 27: Optimization via OReal Loss [18]
- 28: end while

### D. Grading Details

Automated evaluation of complex mathematical proofs presents substantial challenges. LLMs often exhibit excessive sensitivity to lexical phrasing while occasionally overlooking missing logical reasoning. To bridge the gap between automated evaluation pipelines and human experts, we designed a fine-grained grading scheme tailored to the nature of the problem.

Calculation-Centric Evaluation (HMMT, AIME) For datasets primarily focused on final answers, such as HMMT and AIME, we only employ Final Answer Accuracy as the sole metric. A response is awarded full score if and only if the extracted final answer matches the ground truth exactly; otherwise, it receives zero.

Proof-Oriented Evaluation (CNMO, IMO) For Olympiad-level proof problems (e.g., CNMO and IMO), we adopt a rubric-based scoring logic inspired by MathArena [3], with critical modifications to ensure rigor.

The key difference here is their grading schemes often list only the necessary sub-propositions, lacking explicit constraints on the derivation of conclusions. When used with LLM-based judges, this ambiguity frequently leads to significant false positives. To rectify this, we augmented the grading scheme by explicitly coupling sub-propositions with their corresponding conclusion requirements. A representative example of our revised grading scheme is shown in FIgure. D.

[

{

"desc": "1 point should be given for rigorously describing a construction for $n$=3. ˓→ Should prove that $k=2$ is impossible, and that $k=0,1,3$ is possible.",

"points": 1, "title": "Describing a construction for $k=0,1,3$ for $n=3$"

}, {

"desc": "1 point should be given for just finding the answer $k=0,1,3$ is valid for ˓→ all $n$. Providing a specific construction is sufficient to earn points.", "points": 1, "title": "Reaching the answer $k=0,1,3$ for all $n$"

}, {

"desc": "Stating and proving that the perimeter sides ($x=1, y=1, x+y=n+1$) contain a ˓→ total of $3n-3$ points.", "points": 1, "title": "Making an statement about the boundary points"

}, {

"desc": "Arguing that any sunny line can cover at most two points on the perimeter ˓→ sides, so for $n>3$, there must be at least one non-sunny line covering a complete ˓→ boundary line.", "points": 2, "title": "Proving the existence of a non-sunny line covering a complete boundary line"

}, {

"desc": "Stating and proving that if a non-sunny line contains one of the 3 perimeter ˓→ sides ($x=1, y=1, x+y=n+1$), the problem can be to reduce for $n-1$ without changing ˓→ the answer.", "points": 1, "title": "Reducing the problem from $n$ to $n-1$ given a boundary line"

}, {

"desc": "Finishing by summarizing the final answer that for any $n$, the possible ˓→ values of $k$ are 0, 1, and 3.", "points": 1, "title": "Finishing"

} ]

Example of the Refined Grading Scheme for IMO2025 P1. This JSON structure outlines the specific proof obligations, point allocation, and partial credit policies used to guide the LLM judge.

To mitigate the inherent stochasticity of LLM judges, we implement an ensemble evaluation protocol. Each generated solution is evaluated in parallel across 𝑁 = 8 independent runs. For grading points worth more than 1 point, the model is awarded partial credit if it provides a valid partial proof. The final score for a solution is calculated as the arithmetic mean of the total scores obtained across the eight evaluation runs.

### E. Lemma Graph

[Figure 4]

Figure 3: Example of a lemma graph. Red nodes mark all lemmas contributing to the final conclusion, with the numbers below indicating the number of rollout in which each lemma appeared.

### F. Case Example

##### Case Study: AMO-Bench P1

Problem Statement: Divide each side of an equilateral triangle 𝐴𝐵𝐶 into four equal parts. Through each division point, draw lines parallel to the other two sides, defining 15 lattice points. Find the smallest integer 𝑛 such that, no matter which 𝑛 points are chosen, there always exist three points that form an isosceles triangle.

◁ Reasoning Process & Lemma Accumulation

- Round 1

- Lemma 1: The 15 lattice points obtained by dividing each side of an equilateral triangle 𝐴𝐵𝐶 into four equal parts can be uniquely represented by integer pairs (𝑖, 𝑗), where

𝑖 ≥ 0, 𝑗 ≥ 0, 𝑖 + 𝑗 ≤ 4.

Let 𝑃 = (𝑖1, 𝑗1) and 𝑄 = (𝑖2, 𝑗2) be any two points. Their squared Euclidean distance satisfies

|𝑃𝑄|2 = Δ𝑖2 + Δ𝑖 Δ𝑗 + Δ𝑗2, where Δ𝑖 = 𝑖1 − 𝑖2, Δ𝑗 = 𝑗1 − 𝑗2.

- Proof 1:

- • Step 1: Establish a coordinate system with vertex 𝐴 as the origin, and basis vectors u = (1, 0) and

v =

(︀12,

√3

2 )︀

.

- • Step 2: The Cartesian coordinates of any point (𝑖, 𝑗) are 𝑖u + 𝑗v = (𝑖 + 2𝑗 ,

√3

2 𝑗).

- • Step 3: For two points (𝑖1, 𝑗1) and (𝑖2, 𝑗2), calculate the difference vector and apply the Euclidean distance formula. Simplification yields

|𝑃𝑄|2 = (Δ𝑖 + Δ2𝑗 )2 +

(︀√23Δ𝑗

)︀2

= Δ𝑖2 + Δ𝑖 Δ𝑗 + Δ𝑗2.

(The confidence score is 0.75.) Lemma 2: Let 𝑃 = (𝑎, 0) and 𝑄 = (𝑏, 0) be two lattice points on the same side 𝐴𝐵 (where 0 ≤ 𝑎 < 𝑏 ≤ 4). Then, any lattice point 𝑋 = (𝑖, 𝑗) (satisfying 𝑖, 𝑗 ≥ 0, 𝑖 + 𝑗 ≤ 4) satisfies

|𝑋𝑃| = |𝑋𝑄| ⇐⇒ 2𝑖 + 𝑗 = 𝑎 + 𝑏.

- Proof 2:

- • Step 1: Based on Lemma 1, write |𝑋𝑃|2 = (𝑖 − 𝑎)2 + (𝑖 − 𝑎)𝑗 + 𝑗2 and |𝑋𝑄|2 = (𝑖 − 𝑏)2 + (𝑖 − 𝑏)𝑗 + 𝑗2.
- • Step 2: Subtract the two equations and cancel common terms to obtain (𝑖−𝑎)2−(𝑖−𝑏)2+(𝑖−𝑎)𝑗−(𝑖−𝑏)𝑗 = 0.
- • Step 3: Simplify to (𝑏 − 𝑎)(2𝑖 + 𝑗 − 𝑎 − 𝑏) = 0. Since 𝑎 ̸= 𝑏, we obtain the required linear relation 2𝑖 + 𝑗 = 𝑎 + 𝑏.

(The confidence score is 0.0.)

......

- Lemma 10: Among all 15 lattice points, the cardinality of the largest point set containing no isosceles triangles is 5, and only sets of the form “all 5 lattice points of one side” can achieve this upper bound.

###### Proof 10:

- • Step 1: By Lemma 8, taking all 5 lattice points of any single side yields an isosceles-free set of size 5.
- • Step 2: If we attempt to construct an isosceles-free set of size 6, there are two cases:

- – It contains internal lattice points: This contradicts Lemma 6.
- – It contains only boundary lattice points: This contradicts Lemma 9.

Thus, no isosceles-free set of size 6 exists.

- • Step 3: Therefore, the maximum cardinality is 5, and the only maximal construction is the set of all lattice points on a single side.

(The confidence score is 0.0.)

- Round 2 Can not summary any new lemmas.

- Round 3

- Lemma 11: In the set of triangular lattice points, for any two points 𝑃 and 𝑄 chosen from two different sides, there exists a unique third point 𝑅 (located on the third side or in the interior) such that |𝑅𝑃| = |𝑅𝑄|.

- Proof 11:

- • Step 1: Let 𝑃 and 𝑄 be located on sides 𝐴𝐵 and 𝐴𝐶, respectively. Denote their coordinates as 𝑃 = (𝑎, 0) and 𝑄 = (0, 𝑏), where 0 ≤ 𝑎, 𝑏 ≤ 4.
- • Step 2: According to Lemma 1, calculate the difference in squared distance between the two points: |𝑅𝑋|2 − |𝑅𝑌 |2 = (𝑖 − 𝑎)2 + (𝑖 − 𝑎)𝑗 + 𝑗2 − (𝑖)2 − (𝑖)(𝑗 − 𝑏) − (𝑗 − 𝑏)2,

and simplify to obtain the equidistance condition:

2𝑖 + 𝑗 = 𝑎 + 𝑏.

- • Step 3: Under the constraints 𝑖, 𝑗 ≥ 0, 𝑖 + 𝑗 ≤ 4, the equation 2𝑖 + 𝑗 = 𝑎 + 𝑏 has a unique integer solution (𝑖0, 𝑗0) within the triangle. If this solution satisfies 𝑗0 > 0, then the point 𝑅 = (𝑖0, 𝑗0) is located on side 𝐴𝐶; if 𝑗0 = 0, then 𝑅 is located on side 𝐵𝐶. In any case, 𝑅 and 𝑃, 𝑄 form an isosceles triangle.
- • Step 4: It can similarly be proven that if 𝑃, 𝑄 are located on any two different sides, a unique point 𝑅 satisfying the equidistance condition can be found.
- • Step 5: In summary, for any two points on different sides, there must exist a unique third point equidistant to them, completing the proof.

(The confidence score is 0.0.) Lemma 12:

In any set consisting only of points on the three sides, if the cardinality of the set is 6, then the distribution of points across the sides must be such that no more than two points lie on any single side; otherwise, an isosceles triangle must exist.

- Proof 12:

- • Step 1: Let set 𝑆 contain 𝑥 points from the sides, where 𝑥 ≥ 3. If any side contains at least three points, select any two points 𝑃, 𝑄 on that side. By Lemma 4, there exists a third point 𝑅 (not on that side) such that |𝑅𝑃| = |𝑅𝑄|, thus forming an isosceles triangle.
- • Step 2: Therefore, to avoid an isosceles triangle, the number of points on any single side must be at most 2.
- • Step 3: Assume the cardinality of 𝑆 is 6. The distribution across the three sides must be (2, 2, 2) or (2, 3, 1) (partitioned by the three sides). If (2, 3, 1) occurs, the side containing three points must generate an isosceles triangle, a contradiction. Thus, the only possible distribution is (2, 2, 2).
- • Step 4: For the (2, 2, 2) configuration, let two points be located on different sides, denoted (𝑎, 0) and (0, 𝑏). By Lemma 11, a unique third point 𝑅 satisfying |𝑅𝑃| = |𝑅𝑄| can be determined. This point must lie on the third side, and thus must be in the set 𝑆, resulting in an isosceles triangle.
- • Step 5: Therefore, any set of size 6 consisting only of boundary points must contain an isosceles triangle, completing the proof.

(The confidence score is 0.0.)

Round 4 Lemma 13:

In any set of 6 points consisting only of points on the three sides (i.e., containing no internal points), the arrangement wherein at most two points are on any single side cannot avoid an isosceles triangle. In other words, if a point set 𝑆 ⊆ {(𝑖, 𝑗) | 𝑖 ≥ 0, 𝑗 ≥ 0, 𝑖 + 𝑗 ≤ 4} contains 6 points and no internal points, then there must exist three points 𝑃, 𝑄, 𝑅 ∈ 𝑆 such that |𝑅𝑃| = |𝑅𝑄|, thereby forming an isosceles triangle.

- Proof 13:

- • Step 1: Let 𝑥, 𝑦, 𝑧 be the number of points in 𝑆 on sides 𝐴𝐵, 𝐴𝐶, 𝐵𝐶, respectively, such that 𝑥 + 𝑦 + 𝑧 = 6. If the number of points on any side is ≥ 3, select any two points 𝑃, 𝑄 on that side. By Lemma 4 (two points on the same side must have a third equidistant point), there exists a point 𝑅 ̸= 𝑃, 𝑄 (located on another

- side or internally) such that |𝑅𝑃| = |𝑅𝑄|. If 𝑅 ∈ 𝑆, an isosceles triangle is formed. If 𝑅 ∈/ 𝑆, then 𝑆 contains at most all 5 points of that side, and since 𝑥 ≥ 3 and 𝑥 ≤ 5, the possible 6-point distributions must be unbalanced, such as (3, 2, 1), (3, 1, 2), (4, 1, 1), (5, 1, 0), etc. Upon checking these distributions, the equidistant point 𝑅 corresponding to 𝑃, 𝑄 can always be found on another side, and since that side has at least one point selected into 𝑆, 𝑅 must be in 𝑆, resulting in an isosceles triangle.
- • Step 2: Therefore, to avoid an isosceles triangle, the number of points on each side must be ≤ 2. Since 𝑥 + 𝑦 + 𝑧 = 6, the only possible distribution is (2, 2, 2).
- • Step 3: Assume the two points on side 𝐴𝐵 are 𝑃 = (𝑎, 0), 𝑄 = (𝑏, 0) (𝑎 < 𝑏). The equidistant point satisfies the line equation 2𝑖 + 𝑗 = 𝑎 + 𝑏 (Lemma 2). Consulting the table (from Lemma 4), this equation has at least one solution 𝑅 within the triangle, and 𝑅 must be on another side (if 𝑎 + 𝑏 is odd, 𝑅 is not on 𝐴𝐵). Similarly, the point pair on side 𝐴𝐶 yields the equation 𝑖 + 2𝑗 = 𝑐 + 𝑑, and the pair on side 𝐵𝐶 yields 𝑖 − 𝑗 = 𝑒 + 𝑓 − 4 (a corollary of Lemma 4).
- • Step 4: Since there are exactly two points on each side, there must exist a pair of points such that their equidistant point 𝑅 falls onto a selected point on another side. Specifically, if 𝑎 + 𝑏 is odd, the unique integer solution to 2𝑖 + 𝑗 = 𝑎 + 𝑏 lies on 𝐵𝐶, 𝑅 = (𝑎+2𝑏−1, 𝑎+2𝑏+1), and since 𝐵𝐶 contains exactly two points, one of them must be 𝑅. If 𝑎 + 𝑏 is even, the midpoint (𝑎+2𝑏, 0) is on 𝐴𝐵, but since only two points were selected on 𝐴𝐵, 𝑎+2𝑏 must not equal 𝑎 or 𝑏. The equidistant point must then lie on 𝐵𝐶 or 𝐴𝐶, and similarly, it must fall into the already selected point set. Consequently, an isosceles triangle must appear.

- • Step 5: In summary, any set of 6 points containing no internal points must include an isosceles triangle.

(The confidence score is 0.0.)

###### Final Answer:

The Triangular Lattice We place the equilateral triangle 𝐴𝐵𝐶 of side length 4 such that its vertices are:

3)︁. Let the basis vectors for the lattice be:

𝐴 = (0, 0), 𝐵 = (4, 0), 𝐶 = (︁2, 2

√

)︁.

u = (1, 0), v = (︁

√3 2

- 1

- 2,

Every point obtained by the construction is a lattice point, representable by integer coordinates (𝑖, 𝑗):

2 𝑗)︁, 𝑖, 𝑗 ∈ Z≥0, 𝑖 + 𝑗 ≤ 4. (1) Thus, the 15 points constitute the set:

(𝑖, 𝑗) = 𝑖u + 𝑗v = (︁𝑖 + 2𝑗 ,

√3

###### ℒ = {(𝑖, 𝑗) | 𝑖, 𝑗 ≥ 0, 𝑖 + 𝑗 ≤ 4}.

For two points 𝑃 = (𝑖1, 𝑗1) and 𝑄 = (𝑖2, 𝑗2), the squared Euclidean distance is obtained by substituting coordinates from (1) and simplifying:

|𝑃𝑄|2 = (𝑖1 − 𝑖2)2 + (𝑖1 − 𝑖2)(𝑗1 − 𝑗2) + (𝑗1 − 𝑗2)2. (2) Interior Points Force an Isosceles Triangle The interior points of ℒ (those not on the boundary) are:

𝐼1 = (1, 1), 𝐼2 = (1, 2), 𝐼3 = (2, 1).

Using the distance formula (2), one verifies that any interior point possesses **exactly four distinct squared distances** to the remaining fourteen lattice points:

|Point<br><br>|Possible |𝑃 − 𝑄|2|
|---|---|
|(1, 1)<br>(1, 2) (2, 1)<br>|1, 3, 4, 7 1, 3, 4, 7 1, 3, 4, 7<br><br>|

Since an interior point can be at at most four different distances from all other points of ℒ, consider a set 𝑆 ⊆ ℒ that contains an interior point 𝐼, with |𝑆| = 6. The five other points of 𝑆 yield five distances from 𝐼. By the **Pigeonhole Principle**, two of these distances must be equal. These two points, together with 𝐼, form an isosceles triangle. Consequently, **any six-point set that contains an interior point necessarily contains an isosceles triangle**.

Six Boundary Points Also Contain an Isosceles Triangle We now restrict our attention to the case where all six chosen points lie on the boundary of the triangle. The boundary points lie on the three sides:

𝐴𝐵 : (𝑖, 0), 𝑖 = 0, 1, 2, 3, 4, 𝐴𝐶 : (0, 𝑗), 𝑗 = 0, 1, 2, 3, 4, 𝐵𝐶 : (𝑖, 4 − 𝑖), 𝑖 = 0, 1, 2, 3, 4,

(3)

A Useful Fact: Equidistant Third Point Let 𝑃 = (𝑎, 0) and 𝑄 = (𝑏, 0) be two distinct points on side 𝐴𝐵 with 𝑎 < 𝑏. A point 𝑅 = (𝑖, 𝑗) is equidistant from 𝑃 and 𝑄 if and only if:

2𝑖 + 𝑗 = 𝑎 + 𝑏. (4) Within the constraints of the lattice (i.e., 𝑖, 𝑗 ≥ 0, 𝑖 + 𝑗 ≤ 4), the unique integer solution for 𝑅 is:

𝑅 = {︃(0, 𝑎 + 𝑏), 𝑎 + 𝑏 ≤ 4 (lies on 𝐴𝐶), (𝑎 + 𝑏 − 4, 8 − (𝑎 + 𝑏)), 𝑎 + 𝑏 ≥ 4 (lies on 𝐵𝐶).

(5)

This demonstrates that **any two points on a side determine a single lattice point on a different side that is equidistant from them**. This property holds symmetrically for the other two sides. Distribution Case Analysis Let 𝑆 be a set of six boundary points. We define the counts of points on each side as:

𝑥 = |𝑆 ∩ 𝐴𝐵|, 𝑦 = |𝑆 ∩ 𝐴𝐶|, 𝑧 = |𝑆 ∩ 𝐵𝐶|. Since the three vertices are counted in two sets each, we have the inequality:

𝑥 + 𝑦 + 𝑧 = |𝑆| + (number of chosen vertices) ≥ 6. (6)

As each side contains only five points, 𝑥, 𝑦, 𝑧 ≤ 5. Up to permutation, the only possibilities for the distribution (𝑥, 𝑦, 𝑧) that satisfy 𝑥 + 𝑦 + 𝑧 ≥ 6 and 𝑥, 𝑦, 𝑧 ≤ 5 are:

(5, 1, 0), (4, 2, 0), (4, 1, 1), (3, 3, 0), (3, 2, 1), (2, 2, 2).

(7)

We prove that in every case, an isosceles triangle is unavoidable by exhibiting a pair of points on the same side whose equidistant third point (given by (5)) also belongs to 𝑆.

- • Case (5, 1, 0) (or permutation): All five points of side 𝐴𝐵 are present; the sixth point lies on 𝐴𝐶, say (0, 𝑘) with 1 ≤ 𝑘 ≤ 4. Take the pair (0, 0) and (𝑘, 0) on 𝐴𝐵. Their sum is 𝑘 ≤ 4, so by (5) the third equidistant point is (0, 𝑘), which is exactly the chosen point on 𝐴𝐶. An isosceles triangle appears.
- • Case (4, 2, 0) (or permutation): Four points on 𝐴𝐵 and two on 𝐴𝐶. Let the two points on 𝐴𝐶 be (0, 𝑐) and (0, 𝑑) with 𝑐 < 𝑑. The pair (0, 0) and (𝑑, 0) on 𝐴𝐵 has sum 𝑑 ≤ 4; the third equidistant point is (0, 𝑑), which must belong to 𝑆. An isosceles triangle is formed.
- • Case (4, 1, 1) (or permutation): Four points on 𝐴𝐵, one on 𝐴𝐶, and one on 𝐵𝐶. If the point on 𝐴𝐶 is (0, 𝑘) with 𝑘 ≤ 4, the pair (0, 0) and (𝑘, 0) on 𝐴𝐵 yields the third point (0, 𝑘) on 𝐴𝐶. Since (0, 𝑘) ∈ 𝑆, an isosceles triangle occurs.
- • Case (3, 3, 0) (or permutation): Three points on 𝐴𝐵 and three on 𝐴𝐶. Let the points on 𝐴𝐵 be (𝑎, 0), (𝑏, 0), (𝑐, 0) and the points on 𝐴𝐶 be (0, 𝑝), (0, 𝑞), (0, 𝑟). The three sums 𝑎 + 𝑏, 𝑎 + 𝑐, 𝑏 + 𝑐 are distinct numbers in {1, . . . , 7}. If any of these sums equals 𝑝, 𝑞, or 𝑟, we are done by (5). The three numbers 𝑝, 𝑞, 𝑟 are also distinct. Since there are three possible pairs on 𝐴𝐵 and three chosen points on 𝐴𝐶, by the

**Pigeonhole Principle** applied to the set of sums/targets, one of the pairs on 𝐴𝐵 must have its equidistant point coinciding with one of the chosen points on 𝐴𝐶. Hence, an isosceles triangle exists.

- • Case (3, 2, 1) (or permutation): Three points on 𝐴𝐵, two on 𝐴𝐶, and one on 𝐵𝐶. If we consider pairs on 𝐴𝐶 or 𝐵𝐶, similar logic applies. The most direct approach is to consider the two points on 𝐴𝐶 and the single point on 𝐵𝐶. The unique equidistant point must fall onto one of the three points on 𝐴𝐵, proving that an isosceles triangle is unavoidable.
- • Case (2, 2, 2): Two points on each side. Let the points on 𝐴𝐵 be (𝑎, 0), (𝑏, 0) with 𝑎 < 𝑏. If 𝑎 + 𝑏 ≤ 4, the third equidistant point is (0, 𝑎 + 𝑏) on 𝐴𝐶. Since 𝐴𝐶 already contains two points, one of them must be (0, 𝑎 + 𝑏). If 𝑎 + 𝑏 ≥ 5, the third equidistant point is (𝑎 + 𝑏 − 4, 8 − (𝑎 + 𝑏)) on 𝐵𝐶. Since 𝐵𝐶 contains two points, one of them must coincide with this third point. Thus, an isosceles triangle is unavoidable in the (2, 2, 2) case.

Since all six possibilities lead to an isosceles triangle, **any set of six boundary points contains an isosceles triangle**. Sharpness and Conclusion The five points on a single side, for example:

###### {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)},

are collinear. They contain no non-degenerate triangle at all, and therefore no isosceles triangle. Hence, a set of five points can avoid the required configuration. Every set of six of the fifteen lattice points necessarily contains three points forming an isosceles triangle, while a set of five points may avoid it. Therefore, the smallest integer 𝑛 with the required property is:

|6|
|---|

.

A Failure Case before RL Training Distance Formula Let 𝑃𝑗1,𝑘1 and 𝑃𝑗2,𝑘2 be two points. We set Δ𝑗 = 𝑗2 − 𝑗1 and Δ𝑘 = 𝑘2 − 𝑘1. From the coordinate definition, 𝑃𝑗,𝑘 = (︁

)︁, we have:

√3 𝑘 8

2𝑗+𝑘

8 ,

√3 Δ𝑘 8

2Δ𝑗 + Δ𝑘 8

. Hence, the squared distance is:

Δ𝑥 =

, Δ𝑦 =

(2Δ𝑗 + Δ𝑘)2 + 3Δ𝑘2 64

|𝑃𝑗1,𝑘1𝑃𝑗2,𝑘2|2 = Δ𝑥2 + Δ𝑦2 =

(4)

(︀

)︀ 64

Δ𝑗2 + Δ𝑗Δ𝑘 + Δ𝑘2

Δ𝑗2 + Δ𝑗Δ𝑘 + Δ𝑘2 16

4

=

=

.

Thus, the scaled squared distances are the integers:

Δ𝑗2 + Δ𝑗Δ𝑘 + Δ𝑘2, (Δ𝑗, Δ𝑘) ̸= (0, 0), |Δ𝑗| + |Δ𝑘| ≤ 4. (5) A short enumeration of all admissible (Δ𝑗, Δ𝑘) pairs gives the set of possible scaled squared distances:

###### { Δ𝑗2 + Δ𝑗Δ𝑘 + Δ𝑘2 } = {1, 3, 4, 7, 9, 12, 13, 16}. (6)

Consequently, there are exactly eight distinct distances (up to the scaling factor 161 ) between points of ℒ. Isosceles-Free Sets

Three points form an isosceles triangle if and only if one of them is equidistant from the other two. Therefore, a set 𝒮 ⊆ ℒ contains **no** isosceles triangle precisely when, for every 𝑃 ∈ 𝒮, the distances {|𝑃𝑄| : 𝑄 ∈ 𝒮 ∖ {𝑃}} are all different. Since (6) lists only eight possible distances, a point can be the apex of at most eight distinct isosceles triangles. Hence, by the Pigeonhole Principle, a set of ten points must contain an isosceles triangle. This yields the preliminary bound 𝑛 ≤ 10. An Isosceles-Free Set of Size Eight Consider the set 𝑆 defined as the union of all points on side 𝐴𝐵 and three non-adjacent points on side 𝐴𝐶:

. (8)

𝑆 = {(𝑗, 0) | 𝑗 = 0, 1, 2, 3, 4} ⏟ ⏞

###### ∪ {(0, 𝑘) | 𝑘 = 2, 3, 4} ⏟ ⏞

the side 𝐴𝐵

three points on 𝐴𝐶 not including (0,0) and (0,1)

𝑆 contains 5 + 3 = 8 points. We verify that 𝑆 is isosceles-free.

- • Case 1: Apex 𝑃 = (𝑖, 0) ∈ 𝑆 (a point on 𝐴𝐵). The distances from 𝑃 to the three points of 𝐴𝐶 are proportional to 𝑖2 − 𝑖𝑘 + 𝑘2 for 𝑘 ∈ {2, 3, 4}. For each fixed 𝑖 ∈ {0, . . . , 4}, a direct check confirms these three numbers are distinct. Distances to other points of 𝐴𝐵 are (𝑖−𝑖′)2

16 with 𝑖′ ̸= 𝑖, which are also distinct. All distances from 𝑃 to the other points of 𝑆 are thus different.

- • Case 2: Apex 𝑃 = (0, 𝑘) ∈ 𝑆 (a point on 𝐴𝐶). The same calculation, with the roles of 𝑖 and 𝑘 interchanged, shows that the distances from 𝑃 to the points of 𝐴𝐵 are pairwise distinct. Distances to the other two points

16 with 𝑘′ ̸= 𝑘, which are also distinct. All distances from 𝑃 are different. Thus, no point of 𝑆 is equidistant from two others, proving that 𝑆 contains **no** isosceles triangle. Consequently, the maximal size of an isosceles-free subset is ≥ 8, implying 𝑛 ≥ 9. No Isosceles-Free Set of Size Nine Assume, for contradiction, that 𝒯 ⊆ ℒ is isosceles-free and |𝒯 | = 9. Contradiction via Internal Points and Boundaries

###### of 𝐴𝐶 are (𝑘−𝑘′)2

- • Boundary Intersection Constraint: If 𝒯 contains points on all three sides (𝐴𝐵, 𝐴𝐶, and 𝐵𝐶), at least one side must contain three points. The distances from vertex 𝐴 to the five points of side 𝐵𝐶 are proportional to {1, 1, 12, 13, 13}. Hence, any three points of 𝐵𝐶 contain two with the same distance from 𝐴; together with 𝐴, they form an isosceles triangle, contradicting the hypothesis. Therefore, 𝒯 is contained in the union of at most two sides.
- • Reduction: By symmetry, we may assume 𝒯 ⊆ 𝐴𝐵 ∪ 𝐴𝐶.
- • Exclusion of (0, 1): If (0, 1) ∈ 𝒯 , then |(0, 1)(0, 0)|2 = |(0, 1)(1, 0)|2 = 161 . Thus, (0, 1) is equidistant from (0, 0) and (1, 0), forming an isosceles triangle. Hence, (0, 1) ∈/ 𝒯 .

- • Admissible Set: The only points of 𝐴𝐶 that can coexist with all five points of 𝐴𝐵 without creating a duplicate distance are (0, 2), (0, 3), and (0, 4). Therefore, 𝒯 must be a subset of the eight-point set 𝑆 defined in (8), i.e.,

###### 𝒯 ⊆ 𝑆.

Impossibility of a Ninth Point Since |𝑆| = 8, a set 𝒯 of nine points must contain a point outside 𝑆. The remaining points of ℒ are the interior points (1, 1), (1, 2), (2, 1) and the excluded boundary point (0, 1).

- • If (0, 1) is added, we immediately form an isosceles triangle (as shown above).
- • If an interior point, (1, 1), is added, then by (4),

1 16

|(1, 1)(1, 0)|2 = |(1, 1)(2, 0)|2 =

.

Thus, (1, 1) together with (1, 0) and (2, 0) forms an isosceles triangle, which contradicts the assumption that 𝒯 is isosceles-free. The same phenomenon occurs for the other interior points.

Consequently, any ninth point forces the appearance of an isosceles triangle. No isosceles-free subset of ℒ can have nine points; the maximal size is eight. Conclusion

The largest possible cardinality of a subset of the fifteen lattice points that contains no three points forming an isosceles triangle is 8. Therefore, the smallest integer 𝑛 such that **every** choice of 𝑛 points necessarily contains an isosceles triangle is:

𝑛 = 8 + 1 = 9.

|9|
|---|

