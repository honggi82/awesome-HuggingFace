## arXiv:2510.25992v3[cs.CL]27Feb2026

[Figure 1]

# Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning

###### Yihe Deng2*† , I-Hung Hsu1*, Jun Yan1, Zifeng Wang1, Rujun Han1, Gufeng Zhang3, Yanfei Chen1, Wei Wang2, Tomas Pfister1 and Chen-Yu Lee1

1Google Cloud AI Research, 2UCLA, 3Google Cloud

Large Language Models (LLMs) often struggle with problems that require multi-step reasoning. For small-scale open-source models, Reinforcement Learning with Verifiable Rewards (RLVR) fails when correct solutions are rarely sampled even after many attempts, while Supervised Fine-Tuning (SFT) tends to overfit long demonstrations through rigid token-by-token imitation. To address this gap, we propose Supervised Reinforcement Learning (SRL), a framework that reformulates problem solving as generating a sequence of logical “actions”. SRL trains the model to generate an internal reasoning monologue before committing to each action. It provides smoother rewards based on the similarity between the model’s actions and expert actions extracted from the SFT dataset in a step-wise manner. This supervision offers richer learning signals even when all rollouts are incorrect, while encouraging flexible reasoning guided by expert demonstrations. As a result, SRL enables small models to learn challenging problems previously unlearnable by SFT or RLVR. Moreover, initializing training with SRL before refining with RLVR yields the strongest overall performance. Beyond reasoning benchmarks, SRL generalizes effectively to agentic software engineering tasks, establishing it as a robust and versatile training framework for reasoning-oriented LLMs.

###### AMC23 AIME24 AIME25 Minerva Average

60

Model Qwen2.5-7B-Instruct S1K-7B SFT (R1 reasoning) SFT (R1 outline) RL(VR) SFT (outline) -> RL(VR) SRL SRL -> RLVR

| |
|---|

50

| |
|---|

| |
|---|

| |
|---|

40

| |
|---|

| |
|---|

| |
|---|

30

| |
|---|

20

10

0

AMC23 (Avg@32)

AMC23 (Greedy)

AIME24 (Avg@32)

AIME24 (Greedy)

AIME25 (Avg@32)

AIME25 (Greedy)

Minerva Average

Figure 1 | Performance of our method (SRL) against baselines on math reasoning benchmarks, with all models trained on the challenging s1k dataset (Muennighoff et al., 2025). Our key observations are: (1) Directly applying SFT on this dataset leads to performance degradation compared to the base model. (2) While RLVR can improve generalization over SFT, the gains are marginal. (3) Our proposed SRL method substantially outperforms these baselines, and the SRL → RLVR pipeline achieves the highest performance, overcoming the challenges of training on difficult data.

### 1. Introduction

Large Language Models (LLMs) have demonstrated remarkable versatility across diverse domains, ranging from mathematical problem-solving (Wang et al., 2025) and search-result integration (Hsu

Corresponding author(s): Correspondence: yihedeng@g.ucla.edu, ihung.hsu@gmail.com, chenyulee@google.com

* Equal contribution. † This work was done while the author interned at Google Cloud AI Research. This work has no implications of any Google products.

- et al., 2024; Jin et al., 2025) to tool-use (Qian et al., 2025; Yin et al., 2025) and long-horizon agentic interaction (Li et al., 2025c; Tan et al., 2025; Xie et al., 2024). A pivotal recent advancement involves the use of reinforcement learning (RL) to bolster complex reasoning capabilities (Ahmadian et al., 2024; Lambert et al., 2024; Shao et al., 2024). By optimizing models through reward signals tied to verifiable outcomes—such as final answer accuracy—RL provides a scalable framework for eliciting sophisticated problem-solving behaviors, including self-reflection (Guo et al., 2025; Xie et al., 2025).

The effectiveness of these outcome-based RL methods fundamentally depends on the policy model’s ability to discover correct solutions within a limited rollout budget (Brown et al., 2024). However, given practical computational constraints, this learning paradigm struggles on challenging problems from the training data, where the model’s success rate is effectively zero (when the pass@𝑘 rate remains zero even after sampling 𝑘 rollouts). Such cases are increasingly common in tasks requiring complex, multi-step reasoning (Wang et al., 2024; Yue et al., 2025). For these problems, an incorrect intermediate step can derail the entire reasoning chain for a 7B-scale LLM, resulting negative learning signals regardless of any partially correct solutions. Furthermore, naively penalizing all incorrect final outputs can further introduce training instability and hinder progress, making these difficult reasoning tasks largely intractable for standard outcome-based RL methods (Xiong et al., 2025).

An alternative approach is imitation learning, commonly implemented via Supervised Fine-Tuning (SFT) on expert demonstrations (Ross et al., 2011). While SFT can instill valuable reasoning behaviors, its next-token prediction objective enforces rigid, token-level imitation, limiting the model’s ability to generalize beyond the training data. This problem becomes particularly pronounced when training data are modest in scale and when the model itself is relatively less capable. Under such conditions, long and complex demonstrations often lead to overfitting and shallow reasoning behaviors (Chu

- et al., 2025a; Li et al., 2025b), as illustrated by the performance decline in our Figure 1. Consequently, both SFT and outcome-based RL struggle on challenging reasoning tasks, leaving a critical gap for training small open-source models to effectively learn difficult problems.

To address this gap, we introduce Supervised Reinforcement Learning (SRL), a framework that reformulates problem-solving as a sequential decision-making process. Rather than optimizing for a final answer or imitating an entire expert trajectory, SRL trains the model to reproduce the sequence of key actions underlying expert reasoning, following an RL-style objective. Specifically, expert demonstrations are decomposed into a series of intermediate actions, each representing a meaningful decision step. During training, the model first generates an internal monologue to articulate its reasoning and then commits to an “action”. At every step, SRL provides a reward based on the similarity between the model’s predicted action and the corresponding expert action, thereby providing fine-grained, efficiently computable supervision that scales to large datasets.

Our work makes the following contributions:

- • We propose SRL, a novel framework designed to enable effective learning on difficult reasoning tasks where SFT and RLVR struggle, by providing dense and smooth rewards based on similarity with expert actions.
- • We demonstrate the effectiveness and robustness of SRL through extensive experiments on challenging mathematical reasoning and agentic software engineering benchmarks. Our results show that SRL significantly outperforms strong baselines across both domains (5.1 & 5.3).
- • Through detailed analysis, we show that granular guidance is vital to SRL’s reward and its impact on model behavior. We observe that SRL induces flexible and sophisticated reasoning patterns, such as interleaved planning and verification, which improve solution quality without simply increasing output length (5.2).

| | | | | | | | |
|---|---|---|---|---|---|---|---|

Query:

Step by step

Teacher Response

Rollouts

Teacher Response

| | | | | | | | |
|---|---|---|---|---|---|---|---|

| | |
|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

[Figure 2]

| | | | | | | | |
|---|---|---|---|---|---|---|---|

[Step1, Step2] [Step3] Token by token

| | | | | | | | |
|---|---|---|---|---|---|---|---|

[Figure 3]

Rollouts

r1

| | | | | | | | |
|---|---|---|---|---|---|---|---|

[Figure 4]

| | | | | | |
|---|---|---|---|---|---|

| | |
|---|---|

Model

Model

Model . . .

. . .

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

<think> … </think>

[Figure 5]

| | | | | | | | |
|---|---|---|---|---|---|---|---|

[Figure 6]

rk

| | | | | | |
|---|---|---|---|---|---|

| | |
|---|---|

<think> … </think>

(a) RL(VR) (b) SFT (c) SRL

Figure 2 | Illustration of SRL as compared to RL(VR) and SFT. (a) RL(VR) takes a query as input and performs k rollouts. The final answer correctness is used as the reward. (b) SFT uses both a query x and a complete teacher response y as input, training with a per-token loss to maximize the probability 𝑝(y|x). (c) SRL also uses a query and a teacher response. It breaks the response into step actions and, at each step, uses the previous steps as context. The model generates a next step action along with its step-wise inner thoughts, and the reward 𝑟𝑘 is based on the similarity between the model’s and the teacher’s action.

### 2. Related Work

#### 2.1. SFT (Distillation) for LLM Reasoning.

Distilling reasoning into smaller models via SFT on teacher-generated long Chain-of-Thought (CoT) rationales has proven highly effective for transferring complex problem-solving skills (Huang et al., 2024; Li et al., 2023; Min et al., 2024; Yeo et al., 2025), as exemplified by the small models distilled from DeepSeek R1 (Guo et al., 2025). Research indicates this process is surprisingly data-efficient, with small, high-quality datasets often being sufficient (Muennighoff et al., 2025; Ye et al., 2025). Given the success, research has focused on the underlying factor for effective SFT distillation (Chen et al., 2025a). Some emphasized the logical structure of the reasoning trace rather than its semantic correctness (Luo et al., 2025; Stechly et al., 2025), as models can learn from demonstrations with factual errors (Li et al., 2025a). Moreover, significant challenges remain in the student-teacher gap where the student fails to learn from overly complex data (Li et al., 2025b), and the risk of teacher hacking, where the student overfits to a teacher’s specific flaws (Tiapkin et al., 2025). Ultimately, distillation from a teacher model imposes a performance ceiling on the student (Huang et al., 2024).

#### 2.2. RL for LLM Reasoning.

The development of DeepSeek-R1 (Guo et al., 2025) showed the effectiveness of rule-based RL for enhancing the reasoning capabilities of LLMs. This approach utilizes a scalable reward system based on final answer correctness, exemplified by the Group Relative Policy Optimization (GRPO) algorithm (Shao et al., 2024) and parallel algorithms (Ahmadian et al., 2024; Lambert et al., 2024; Xie et al., 2025). Building on this foundation, subsequent research has introduced numerous algorithmic refinements. For example, Dr. GRPO (Liu et al., 2025) mitigates bias by removing variance normalization, while DAPO (Yu et al., 2025) introduces a token-level loss and relaxes the policy update constraint by increasing the clipping threshold. Other notable advancements include modifications to clipping methods, normalization techniques, the KL divergence loss, and dynamic sampling strategies (Chen et al., 2025b; Chu et al., 2025b; Zhang and Zuo, 2025; Zhang et al., 2025). Despite these algorithmic variations, these approaches primarily rely on the final outcome’s reward signal. A critical challenge arises when the rollouts fail to identify a correct solution trajectory, particularly for

difficult queries. DAPO (Yu et al., 2025), for instance, addresses this by filtering out instructions that do not yield any successful rollouts.

### 3. Preliminaries

- A Large Language Model (LLM) is formally defined by a probability distribution 𝑝𝜃 over sequences of tokens, parameterized by a set of model weights 𝜃. Given an input prompt, represented as a token sequence x = [𝑥1, . . . , 𝑥𝑛], the model generates a response sequence y = [𝑦1, . . . , 𝑦𝑚]. The response is produced autoregressively, where the generation of the token 𝑦𝑗 at any step 𝑗 is conditioned on the initial prompt x and all preceding tokens in the generated sequence, (𝑦1, . . . , 𝑦𝑗−1). The joint

probability of the entire response sequence y given the prompt x is thus factorized as: 𝑝𝜃(y|x) = 𝑚 𝑗=1 𝑝𝜃(𝑦𝑗|x, 𝑦1, . . . , 𝑦𝑗−1).

Supervised Fine-Tuning (SFT). SFT is typically employed to specialize LLM for downstream applications or domains. It is also commonly used to establish a cold start for subsequent RL training phases that requires certain reply format/pattern, such as RL for reasoning (Deng et al., 2025) or tool use (Feng et al., 2025). Specifically, the process utilizes a dataset D = {(x(𝑖), y(𝑖))}𝑖𝑁=1, where each x(𝑖) is an input prompt and y(𝑖) is the corresponding desired model output. The primary objective is to update the parameters 𝜃 to maximize the conditional probability of generating the target response y(𝑖) given the input prompt x(𝑖). This goal is formally achieved by minimizing the negative log-likelihood loss function: LSFT(𝜃) = − 𝑖 𝑁=1 log 𝑝𝜃(y(𝑖)|x(𝑖)) over the entire dataset. By minimizing this loss, the model learns to produce responses that are closely aligned with the exact words demonstrated in the labeled training examples.

Reinforcement Learning (RL). Recent literature on improving model reasoning capability has focused on RL with verifiable reward (RLVR), where the policy model receives reward signals purely based on the final answer correctness. Building on this principle, Group Relative Policy Optimization (GRPO) (Shao et al., 2024) involves sampling a group of 𝐺 response trajectories, {o𝑖}𝐺𝑖=1, from the previous policy model, 𝜃𝑜𝑙𝑑, for each input query x. The objective function for GRPO is:

∑︁|o𝑖|

##### ∑︁𝐺

𝑝𝜃(𝑜𝑖,𝑡 | x, o𝑖,<𝑡) 𝑝𝜃𝑜𝑙𝑑(𝑜𝑖,𝑡 | x, o𝑖,<𝑡)

𝑝𝜃(𝑜𝑖,𝑡 | x, o𝑖,<𝑡) 𝑝𝜃𝑜𝑙𝑑(𝑜𝑖,𝑡 | x, o𝑖,<𝑡)

1

1 |o𝑖|

𝐴ˆ𝑖,𝑡, clip

, 1 − 𝜖, 1 + 𝜖 𝐴 ˆ𝑖,𝑡

min

𝔼

𝐺

𝑡=1

𝑖=1

− 𝛽 𝔻KL 𝑝𝜃 ∥ 𝑝ref . (1)

The hyperparameter 𝜖 > 0 defines the clipping range for the policy update ratio, and the coefficient 𝛽 > 0 modulates the influence of the KL-divergence penalty against the policy update. The term 𝜃𝑜𝑙𝑑 refers to the policy from the previous iteration. The advantage function, 𝐴ˆ𝑖,𝑡 = (˜𝑟𝑖 − mean(˜𝑟))/std(˜𝑟), is defined as the group-level normalized reward.

A key challenge for these RL algorithms emerges when input queries are either too easy or too

hard, resulting in uniform correctness within policy rollouts {o𝑖}𝐺𝑖=1. In such cases, the advantage estimate 𝐴ˆ𝑖,𝑡 vanishes, yielding an uninformative policy gradient and preventing model updates. A common strategy to mitigate this is to dynamically sample the batches, filtering out samples and re-sampling until the data sample satisfies 0 < |{o𝑖|is_correct(o𝑖)}| < 𝐺 (Yu et al., 2025).

Step2 Context:

Expert Trajectory:

|Question: … Solution: Given a rational number between 0 and 1, ….. These are 8 distinct primes.|
|---|

Given a rational number between 0 and 1, we need to write it as a fraction in lowest terms \(\frac{a}{b}\) and calculate the product of the resulting numerator and denominator. We need to determine how many such rational numbers exist where the product \(a \times b = 20!\).

Instruction:

|Complete a single step of partial solution. You should first draft your thinking process as an inner monologue…|
|---|

[Figure 7]

- 1. **Prime Factorization of 20!**:

- The prime factors of 20! are 2, 3, 5, 7, 11, 13, 17, and 19. These are 8 distinct primes.

- 2. **Coprime Pairs**:

- For \(a \times b = 20!\) and \(\gcd(a, b) = 1\), each prime factor of 20! must be assigned entirely to either \(a\) or \(b\). This results in \(2^8 = 256\) coprime pairs \((a, b)\).

- 3. **Counting Pairs \(a < b\)**: ……

| |
|---|
| |
| |
| |
| |

- [Step1]

- [Step2]

<think> …… </think> <solution>

2. Get Coprime Pairs: For a \times b = 20! with \gcd(a,b)=1, each of the 8 distinct prime factors of 20! must be partitioned between them. This gives two choices for each prime, resulting in 2 ^ 8 = 256 coprime pairs. </solution>

r2

Figure 3 | Given a solution trajectory, we take each summarized step as an action to be learned and take the partial solution before the step as the context of our newly created data. The model is then prompted to generate its thinking process followed by the action for the current step. A reward (𝑟2 in the figure) is then calculated based on the similarity between the model’s and the expert’s action.

### 4. Methodology

#### 4.1. The challenge of hard reasoning problem

RL with verifiable reward is a prominent technique for enhancing the reasoning capabilities of LLMs. The strategy is to close the gap between a model’s potential performance across multiple attempts (pass@𝑘) (Brown et al., 2024; Yue et al., 2025). However, this paradigm falters on problems where the model’s pass@𝑘 rate is already near zero. For this set of difficult problems, which we term Dhard, positive reward signals are too sparse for RLVR to be effective (Xiong et al., 2025). Moreover, simply penalizing incorrect outputs can be detrimental to model performance (Xiong et al., 2025; Yu et al., 2025), creating a significant challenge for improving model reasoning.

Formally, we define Dhard = {x(𝑖), 𝑎(𝑖)}𝑖𝑁=1 as the set of problems (x, 𝑎) where policy model’s success rate is low with 𝑘 samples: 1𝑘 𝑘𝑗=1 𝕀 ExtractAnswer(y(𝑗)) == 𝑎 ≤ 𝜖, where each solution attempt y(𝑗) is sampled from the policy 𝑝𝜽(·|x) and 𝜖 > 0 is a small constant.

Due to the scarcity of successful trajectories, standard RL with verifiable reward struggles on

Dhard. Such data is further difficult to be learned by SFT, due to its limited amount and complexity in teacher reasoning trajectories (Li et al., 2025b).

#### 4.2. Supervised Reinforcement Learning (SRL)

To address the challenge of learning from Dhard, we introduce Supervised Reinforcement Learning (SRL), a framework that decomposes complex problem-solving as a sequential decision-making process, and thus can be easily learned on how to properly operate step-wise. Instead of generating a monolithic solution, the model learns to take actions similar to the expert while producing their own inner reasoning process in a step-by-step manner. The whole framework is illustrated in Figure 3. 1

1Empirically, we found that providing the subsquent step title (e.g., “2. **Coprime Pairs**” in Figure 3) as the additional context for the learner to predict the rest of the step content can further boost the performance.

Action-based problem formulation. Given an expert solution trajectory y that leads to a correct final answer, we decompose y into a sequence of tuples: y = {ystep

𝑛}𝑛𝑁=1. Each step represents a logical action: the concrete action to be operated. This formulation is domain-agnostic; for instance, an action in mathematical reasoning could be an algebraic manipulation, while for a software agent, it could be a command executed in a code repository.

Step-wise training data construction. To create training data for SRL, we leverage a powerful teacher model, 𝜽𝑒𝑥𝑝𝑒𝑟𝑡, to generate solution trajectories. From a single complete solution with 𝑁 steps, we construct 𝑁 −1 partial trajectories. For each step 𝑘 ∈ {1, . . . , 𝑁 −1}, we create a new input prompt xstep

. This process transforms one expert solution into a rich set of training instances that teach the model to proceed correctly from various intermediate states.

= [x, ystep1, . . . , ystep

𝑘−1], where the model’s task is to predict the subsequent step, ystep

𝑘

𝑘

Learning with a sequence similarity reward with own inner monologue. Given a partial context xstep

containing the problem and a partial solution, the policy model 𝑝𝜽 is prompted to generate the

𝑘

subsequent action step with their own inner monologue ythink′ , which is encapsulated by “<think>” tags. We then provide a dense reward based on the quality of the generated logical action ystep′

. The prediction can be formally specified as: y′ ∼ 𝑝𝜽(·|xstep

𝑘

𝑘) = [ythink′ , ystep′

].

𝑘

To guide training, we consider the reward function that measures the similarity between the generated action: 𝑅(ystep′

𝑘) = 2𝑇𝑀, where

, ystep

𝑘

- • T (Total elements): This is the total number of elements in both sequences combined. It is calculated as the sum of the lengths of the two sequences: 𝑇 = |𝑆1| + |𝑆2|.
- • M (Matched elements): The total count of elements found in all non-overlapping matching blocks between the two sequences. The algorithm first finds the longest contiguous matching subsequence and then recursively searches for more matches in the segments to the left and right of that block. If we represent the set of all such matching blocks as a list of tuples (𝑖, 𝑗, 𝑛), where 𝑛 is the length of the matching block, then 𝑀 is the sum of all lengths 𝑛: 𝑀 = (𝑖,𝑗,𝑛)∈MatchingBlocks 𝑛.

Combining these definitions, we can calculate the similarity ratio 𝑅 ∈ [0, 1] as:

2 (𝑖,𝑗,𝑛)∈MatchingBlocks 𝑛 |𝑆1| + |𝑆2|

𝑅 =

While the choice of similarity function can be adapted to specific data distributions or task requirements, we employ Python’s difflib.SequenceMatcher for our implementation. Furthermore, a negative reward is assigned if the generated output y′ fails to adhere to the required format. Hence, the final reward used is:

𝑘) if y′ follows format, −1 otherwise.

𝑅(ystep′

, ystep

𝑟(ystep′ 𝑘, ystep

𝑘) =

𝑘

The policy 𝑝𝜽 is then optimized using this reward signal with the GRPO objective function defined in Equation 1. Notably, our reward is computed only on the logical action, not the internal monologue. This grants the model flexibility to develop its own internal reasoning style while ensuring its external actions align with the expert’s strategy. This design provides dense, step-level feedback and allows for rapid reward calculation, making the SRL framework both effective and scalable.

Dynamic sampling for SRL. As our reward signal 𝑟 ∈ [0, 1] ∪ {−1} is dense, we generalize the dynamic sampling strategy previously designed for outcome accuracy and implement it to filter samples with less meaningful updates. Specifically, a sample should be filtered out if its rollouts yield rewards with near-zero variance, providing a weak advantage and thus weak learning signal. With

the sequence similarity reward in SRL, we retain a sample if the standard deviation of the reward scores of its rollouts exceeds a threshold 𝜖 > 0:

√︄

𝐺 𝑖=1(𝑟(o𝑖, y) − ¯𝑟)2

> 𝜖

𝐺

where 𝐺 is the number of generated rollouts, 𝑟(o𝑖, y) is the sequence similarity reward for the 𝑖-th rollout o𝑖 given the expert trajectory y, and ¯𝑟 is the mean reward for the sample. To maintain a consistent batch size of 𝐵, we continuously sample and filter until the batch is filled.

### 5. Experiments

#### 5.1. Main Results: Math Reasoning

Setup. We finetue Qwen2.5-7B-Instruct (Yang et al., 2024) on the s1K-1.1 dataset (Muennighoff et al., 2025). This dataset contains 1,000 diverse and challenging questions, each accompanied by a detailed reasoning trace and a final solution generated by DeepSeek R1. The solutions from DeepSeek R1 are formatted with structured, numbered steps (e.g., “1. Title of Step 1”). We leverage this structure to generate intermediate training targets by parsing these solutions and treating each complete step as a ground-truth continuation. Any data points that do not adhere to this format are excluded.The detailed data preparation process can be found in Appendix B.3. We hold out 60 questions from the dataset to form our validation set.

Baselines. We benchmark our proposed methods against several baselines, all initialized from the Qwen2.5-7B-Instruct model. These baselines include:

- 1. SFT on either the complete reasoning traces (R1 reasoning) or the final solutions from the s1K-1.1 dataset (R1 outline)
- 2. s1K-7B, the official distilled model released by the dataset’s authors
- 3. RLVR, which we implement using the GRPO algorithm. To ensure fair comparison, we implement additional dynamic sampling as in Yu et al. (2025), which removes samples with all correct or incorrect rollouts.
- 4. SFT → RLVR, the same RLVR algorithm applied after an initial SFT phase
- 5. R3 (Xi et al., 2024), which applies a reverse curriculum RL learning strategy by starting near the end of the demonstration solution and working backward with step-by-step decomposition to make the sparse final reward easier to find 2.

Our proposed method, SRL, is likewise evaluated both as a standalone technique and in a sequential configuration where it precedes RLVR (SRL then RLVR). All models are trained for up to 30 epochs, and for each method, we select the checkpoint with the best performance on the validation set.

Evaluation. We evaluate all models on the following four competition-level mathematical reasoning benchmarks: AMC23, AIME24, AIME253 and Minerva Math (Lewkowycz et al., 2022). Our evaluation protocol for all benchmarks follows the package released by Qwen2.5-Math4 and report the accuracy of greedy sampling. The evaluation prompt can be viewed in Appendix B.2.2. In addition, for AMC23, AIME24 and AIME25, we report the average@32 score with a temperature of 1.0 for all baselines to ensure a more robust evaluation, referenceing the evaluation in Yu et al. (2025).

2We re-implement this method by reformulating the data from s1K-1.1. 3Data available on HuggingFace (https://huggingface.co/datasets): AI-MO/aimo-validation-amc,

AI-MO/aimo-validation-aime, and math-ai/aime25 4https://github.com/QwenLM/Qwen2.5-Math

Performance. The performance results of our models are summarized in Table 1. Consistent with the officially released S1K-7B model, our model trained with SFT on the same dataset exhibited a notable performance degradation. In contrast, methods based on RL maintained generalization on the evaluation benchmarks. Specifically, while RLVR maintained the performance, SRL provided a substantial boost of 3.0% on average. Furthermore, applying RLVR after SRL training yielded a 3.7% increase on average, leveraging only 1k training data.

Comparatively, while the baseline R3 also use teacher demonstrations to structured a step-by-step learning, it exhibits distinct performance characteristics. As shown in Table 1, R3 performs competitively on AMC23 but lags behind SRL on the more challenging AIME24 and AIME25 benchmarks (e.g., SRL achieves 16.7% vs. R3’s 13.3% on AIME24). This divergence highlights a critical difference between SRL and R3’s objective, which relies on a reverse curriculum to facilitate exploration towards a sparse outcome reward. In contrast, SRL employs dense, intrinsic supervision, rewarding the model for the quality of its step-wise “actions” rather than just the final answer. Consequently, for complex reasoning tasks where outcome-based exploration is very challenging, SRL provides a more robust learning signal, effectively teaching the model how to reason and act.

- Table 1 | Evaluation results across competition-level math benchmarks. We take Qwen2.5-7B-Instruct as the base model and report the performance of different training schemes (SFT, RLVR via GRPO, and SRL) using the same set of training data. The bold numbers indicate the best results among the open-source models and the underscored numbers represent the second-best results.

AMC23 AIME24 AIME25

Model

Minerva Math Average Avg@32 Greedy Avg@32 Greedy Avg@32 Greedy

Base Model Qwen2.5-7B-Instruct 49.3 50.0 10.5 13.3 7.5 6.7 34.9 24.6 Training with SFT

S1K-7B 24.1 25.0 2.2 3.3 3.7 3.3 20.2 11.7 SFT (R1 reasoning) 26.8 40.0 3.9 10.0 5.4 10.0 20.2 16.6 SFT (R1 outline) 36.2 27.5 5.1 3.3 3.8 6.7 31.6 16.3

Training with Outcome-Reward RL

RL(VR) 52.0 47.5 11.1 10.0 7.4 10.0 33.8 24.5 SFT (outline) → RL(VR) 37.6 35.0 4.9 3.3 4.5 6.7 30.1 17.4 R3 52.5 57.5 10.0 13.3 7.4 10.0 35.3 26.6

Training with SRL

SRL 51.5 50.0 13.2 16.7 7.1 13.3 36.4 27.6 SRL → RLVR 52.1 57.5 13.3 20.0 8.6 10.0 36.4 28.3

#### 5.2. Analysis: Math Reasoning

Effect of dynamic sampling in SRL. In Table 2, we analyze the impact of the dynamic sampling component in SRL, based on thresholding the standard deviation of sequence similarity rewards within rollouts. For both models, we train until the training reward converges and select checkpoint based on validation scores. Our results are consistent the findings of DAPO (Yu et al., 2025), which stated that removing samples that provide a zero learning signal is critical in the effectiveness of the RL training loop, showing non-trivial d performance improvement.

Disentangling the impact of guidance granularity in SRL. To isolate the effects of guidance granularity from the benefits of sequence similarity rewards, we evaluated our multi-step method against two simplified, one-step baselines:

- Table 2 | The effect of dynamic filtering on SRL. Filtering out samples with less meaningful updates provides non-trivial performance improvement. DS stands for dynamic sampling.

Model

AMC23 AIME24 AIME25

Minerva Math Average Avg@32 Greedy Avg@32 Greedy Avg@32 Greedy

SRL w/out DS 48.5 52.5 11.1 13.3 6.8 6.7 33.8 24.7 SRL w/ DS 51.5 50.0 13.2 16.7 7.1 13.3 36.4 27.6

- Table 3 | Model Performance of different reward functions and density. For sequence similarity reward, we implement it with the entire expert output as a one-step supervision. The model benefits from our multi-step decomposition on the small set of challenging training data.

Model

AMC23 AIME24 AIME25

Minerva Math Average Avg@32 Greedy Avg@32 Greedy Avg@32 Greedy

Final Answer Reward One-step (RLVR) 52.0 47.5 11.1 10.0 7.4 10.0 33.8 24.5 Sequence Similarity Reward

One-step 51.8 52.5 11.7 13.3 6.9 10.0 34.9 25.9 Multi-step (SRL) 51.5 50.0 13.2 16.7 7.1 13.3 36.4 27.6

- • Final-answer reward: This baseline utilizes RLVR trained with GRPO, rewarding the model solely based on the correctness of the final answer.
- • Holistic sequence similarity reward: The model generates a complete solution in a single step. This entire solution is then evaluated for similarity against the complete ground-truth trajectory.

As shown in Table 3, the results underscore the value of fine-grained guidance for sequence similarity reward. While the holistic sequence-similarity reward improved performance on average, providing granular, step-by-step guidance led to markedly superior performance across benchmarks.

Model size variation. To evaluate the scalability of our approach, we apply our method to Qwen2.53B-Instruct (Yang et al., 2024) using the identical data setup from Table 1. As summarized in Table 4, our method yields a consistent performance boost, improving the average score over base model by 3.1% and surpassing RL(VR). Notably, while the limited capacity of the smaller base model restricts improvements on extremely difficult benchmarks (e.g., AIME24 & AIME25), we observe significant gains on more tractable tasks like AMC23 and Minerva Math.

- Table 4 | Evaluation results across competition-level math benchmarks using smaller base model. We take Qwen2.5-3B-Instruct as the base model in this table to test the generalization of our method.

AMC23 AIME24 AIME25

Model

Minerva Math Average Avg@32 Greedy Avg@32 Greedy Avg@32 Greedy

Qwen2.5-3B-Instruct 32.5 45.0 3.3 3.3 1.7 3.3 25.7 16.4 RL(VR) 35.0 42.5 6.7 3.3 6.7 3.3 26.5 17.7 SRL → RLVR 37.5 47.5 6.7 3.3 6.7 6.7 28.3 19.5

Interleaved reasoning behaviors. Beyond achieving superior performance, models trained with SRL exhibit distinctive and adaptive reasoning trajectories. While conventional models typically generate a single, monolithic reasoning block at the outset, our SRL-finetuned models—particularly those trained with RLVR—dynamically interleave reasoning steps throughout the generation process. This

behavior manifests in three primary modes: (1) Structured Planning, where the model outlines a comprehensive roadmap before execution; (2) Dynamic Adjustment, where the model inserts iterative reasoning blocks to refine its trajectory mid-solution; and (3) Reflective Verification, where the model pauses to evaluate its logic and verify the answer before committing to a final output. These emergent patterns are illustrated in Example 1 and Example 2 (see Appendix), which showcase the model’s transition from static generation to an integrated, "think-and-act" reasoning process. For further qualitative analysis and studies, please refer to Appendix C.

#### Example 1: Interleaved thinking and self-verification

Question: What is the least positive integer value of 𝑥 for which the inequality 3𝑥 > 2𝑥 + 1 is true? <think> To solve the inequality 3𝑥 > 2𝑥 + 1, we can start by isolating 𝑥. Let’s subtract 2𝑥 from both sides of the inequality. </think> Subtract 2𝑥 from both sides:

3𝑥 − 2𝑥 > 2𝑥 + 1 − 2𝑥 This simplifies to:

𝑥 > 1

<think> We can verify this by substituting 𝑥 = 2 into the original inequality to ensure it holds true. </think> The inequality 𝑥 > 1 tells us that 𝑥 must be greater than 1. The smallest positive integer greater than 1 is 2. Therefore, the least positive integer value of 𝑥 that satisfies the inequality is 2. Substitute 𝑥 = 2 into the original inequality:

3(2) > 2(2) + 1 This simplifies to:

6 > 5

This is true, so 𝑥 = 2 is indeed a solution. Therefore, the least positive integer value of 𝑥 for which the inequality 3𝑥 > 2𝑥 + 1 is true is

.

|2|
|---|

#### 5.3. Extension: Software Engineering Agentic Reasoning

Task. We extend our SRL framework to the domain of software engineering, training agents to resolve real-world programming issues. These tasks are commonly evaluated on benchmarks like SWE-Bench (Jimenez et al., 2023), which requires agents to perform complex, multi-turn interactions with large codebases and reason deeply about code functionality.

However, unlike in math domain, the direct application of online RL to software engineering is impeded by significant practical challenges. These include handling long context windows, highlatency environment feedback, and slow patch verification (Golubev et al., 2025; Wei et al., 2025). Consequently, these obstacles have hindered the development of stable and scalable end-to-end RL methods, leading to a prevailing approach of collecting expert agent trajectories and distilling them into a policy via SFT (Pan et al., 2024; Yang et al., 2025).

Setup. We apply SRL to further fine-tune Qwen2.5-Coder-7B-Instruct (Hui et al., 2024), a model already specialized for coding tasks. We use a dataset from Yang et al. (2025), which consists of 5,000 expert agent trajectories. These trajectories were generated by claude-3-7-sonnet-20250219 (Anthropic, 2025) and subsequently verified to ensure they produce correct code patches.

Each trajectory is composed of multiple steps defined by the agent’s interactions with the coding environment. As the example below illustrates, a single step contains natural language reasoning followed by an executable action:

I'll help you implement the necessary changes to fix the issue with the `OriginValidator ` not properly handling wildcard (*) in allowed_origins. Let 's follow the steps you outlined.

## Step 1: Find and read relevant code

First , let's explore the repository structure to locate the `OriginValidator ` class mentioned in the PR description. This is an extra long line added to demonstrate how the automatic line wrapping feature from the listings package works.

<function=bash > <parameter=command >find /testbed -type f -name "*.py" | grep -v "__pycache__" | sort </paramete r> </function >

In line with our SRL formulation (Section 4.2), we define the “action” as the environment-consumable command (e.g., the bash call). Following this decomposition, we process the full trajectories to create 134k step-wise training instances. For validation, we hold out 30 full trajectories, from which we curate a validation set of 650 step-wise instances.

Evaluation. We evaluate our model’s patch generation performance by measuring its resolve rate (%) under two distinct configurations, following Wei et al. (2025): (1) Oracle file editing evaluation: The model is provided with the oracle code files to repair. This configuration isolates and measures the model’s core patch generation capability; (2) End-to-end evaluation: This setting uses the Agentlessmini agent scaffold (Wei et al., 2025) to first identify the file(s) to modify and subsequently generate the patch. It tests the model’s fault localization and code repair abilities in conjunction.

We compare our SRL-trained model against two crucial baselines: the original base model (Qwen2.5-Coder-Instruct) and SWE-Gym-7B (Pan et al., 2024). Since SWE-Gym-7B is an SFT-based model finetuned from the same base model, this provides a direct, fair comparison between SFT and our SRL training methodology. As shown in Table 5, SRL substantially outperforms both baselines. In the oracle setting, SRL achieves a 14.8% resolve rate, representing a 74% relative improvement over the strong SWE-Gym-7B baseline. The performance gain is consistent when evaluating in the challenging end-to-end setting, where SRL can obtain twice the performance.

- Table 5 | Performance of SRL on SWE-Bench-Verified. Results in the table are using greedy decoding.

#### Oracle File Edit End-to-End

Qwen2.5-Coder-Instruct (Base) 5.8 3.2 SWE-Gym-7B 8.4 4.2 SRL (ours) 14.8 8.6

#### 5.4. Discussion and Limitations

Despite the strong performance of SRL, its effectiveness is subject to several inherent constraints. Primarily, the method relies heavily on the availability of high-quality, decomposable expert trajectories, as it assumes a well-defined “action” granularity where complex reasoning can be cleanly partitioned into discrete steps. This dependency suggests that SRL may be less effective in domains where expert demonstrations are sparse or in tasks that require holistic intuition rather than a sequential chain-of-thought structure. Furthermore, the success of the self-reflective loop is fundamentally tied to the student model’s initial proficiency; the model must already possess a baseline instructionfollowing capability to generate rollout samples that are coherent enough to yield non-trivial reward signals. Without this initial competence, the learning process may suffer from excessively sparse rewards. Consequently, future research will focus on adapting SRL to domains with ambiguous action

definitions, exploring techniques to alleviate the dependency on highly structured expert data, and investigating the variance in performance when applying different similarity functions to guide the learning process.

### 6. Conclusion

In conclusion, we introduced Supervised Reinforcement Learning (SRL), a novel method designed to teach LLMs complex reasoning skills from expert demonstrations, particularly for problems that are too difficult for conventional RL or SFT approaches. By breaking down expert solutions into manageable steps and leveraging a dense sequence similarity reward, SRL provides effective, granular guidance that bridges the gap between imitation learning and reinforcement learning. Our empirical results demonstrate that SRL not only significantly outperforms baseline methods in both mathematical reasoning and software engineering tasks but also enables a powerful curriculum learning strategy when combined with RLVR. This work establishes SRL as a robust and generalizable technique for unlocking a model’s potential to learn from challenging, multi-step problems, paving the way for training more capable and versatile AI agents.

### 7. Acknowledgments

We thank Jinwei Xing, Jinsung Yoon, and members from Google Cloud AI Research for their valuable feedback during the preparation of the paper.

### References

- A. Ahmadian, C. Cremer, M. Gallé, M. Fadaee, J. Kreutzer, O. Pietquin, A. Üstün, and S. Hooker. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. arXiv preprint arXiv:2402.14740, 2024.

Anthropic. Introducing claude 3.7 sonnet, 2025., 2025. URL https://www.anthropic.com/news/ claude-3-7-sonnet.

- B. Brown, J. Juravsky, R. Ehrlich, R. Clark, Q. V. Le, C. Ré, and A. Mirhoseini. Large language monkeys: Scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787, 2024.

- X. Chen, Z. Sun, W. Guo, M. Zhang, Y. Chen, Y. Sun, H. Su, Y. Pan, D. Klakow, W. Li, et al. Unveiling the key factors for distilling chain-of-thought reasoning. arXiv preprint arXiv:2502.18001, 2025a.
- Y. Chen, Y. Ge, R. Wang, Y. Ge, J. Cheng, Y. Shan, and X. Liu. Grpo-care: Consistency-aware reinforcement learning for multimodal reasoning. arXiv preprint arXiv:2506.16141, 2025b.

T. Chu, Y. Zhai, J. Yang, S. Tong, S. Xie, D. Schuurmans, Q. V. Le, S. Levine, and Y. Ma. Sft memorizes, rl generalizes: A comparative study of foundation model post-training. arXiv preprint arXiv:2501.17161, 2025a.

- X. Chu, H. Huang, X. Zhang, F. Wei, and Y. Wang. Gpg: A simple and strong reinforcement learning baseline for model reasoning. arXiv preprint arXiv:2504.02546, 2025b.
- Y. Deng, H. Bansal, F. Yin, N. Peng, W. Wang, and K.-W. Chang. Openvlthinker: An early exploration to complex vision-language reasoning via iterative self-improvement. arXiv preprint arXiv:2503.17352, 2025.

J. Feng, S. Huang, X. Qu, G. Zhang, Y. Qin, B. Zhong, C. Jiang, J. Chi, and W. Zhong. Retool: Reinforcement learning for strategic tool use in llms. arXiv preprint arXiv:2504.11536, 2025.

- A. Golubev, M. Trofimova, S. Polezhaev, I. Badertdinov, M. Nekrashevich, A. Shevtsov, S. Karasik, S. Abramov, A. Andriushchenko, F. Fisin, S. Skvortsov, and B. Yangel. Training long-context, multiturn software engineering agents with reinforcement learning. arXiv preprint arXiv:2508.03501, 2025.

D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

- I.-H. Hsu, Z. Wang, L. Le, L. Miculicich, N. Peng, C.-Y. Lee, and T. Pfister. CaLM: Contrasting large and small language models to verify grounded generation. In Findings of the Association for Computational Linguistics: ACL 2024, 2024.

Z. Huang, H. Zou, X. Li, Y. Liu, Y. Zheng, E. Chern, S. Xia, Y. Qin, W. Yuan, and P. Liu. O1 replication journey–part 2: Surpassing o1-preview through simple distillation, big progress or bitter lesson? arXiv preprint arXiv:2411.16489, 2024.

- B. Hui, J. Yang, Z. Cui, J. Yang, D. Liu, L. Zhang, T. Liu, J. Zhang, B. Yu, K. Dang, et al. Qwen2. 5-coder technical report. arXiv preprint arXiv:2409.12186, 2024.
- C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, and K. Narasimhan. Swe-bench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770, 2023.

B. Jin, H. Zeng, Z. Yue, J. Yoon, S. Arik, D. Wang, H. Zamani, and J. Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

N. Lambert, J. Morrison, V. Pyatkin, S. Huang, H. Ivison, F. Brahman, L. J. V. Miranda, A. Liu, N. Dziri, S. Lyu, et al. Tulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.

- A. Lewkowycz, A. Andreassen, D. Dohan, E. Dyer, H. Michalewski, V. Ramasesh, A. Slone, C. Anil,

- I. Schlag, T. Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in neural information processing systems, 2022.

- D. Li, S. Cao, T. Griggs, S. Liu, X. Mo, E. Tang, S. Hegde, K. Hakhamaneshi, S. G. Patil, M. Zaharia, et al. Llms can easily learn to reason from demonstrations structure, not content, is what matters! arXiv preprint arXiv:2502.07374, 2025a.

L. H. Li, J. Hessel, Y. Yu, X. Ren, K.-W. Chang, and Y. Choi. Symbolic chain-of-thought distillation: Small models can also" think" step-by-step. arXiv preprint arXiv:2306.14050, 2023.

- Y. Li, X. Yue, Z. Xu, F. Jiang, L. Niu, B. Y. Lin, B. Ramasubramanian, and R. Poovendran. Small models struggle to learn from strong reasoners. arXiv preprint arXiv:2502.12143, 2025b.
- Z. Li, Y. Hu, and W. Wang. Encouraging good processes without the need for good answers: Reinforcement learning for llm agent planning. arXiv preprint arXiv:2508.19598, 2025c.

- Z. Liu, C. Chen, W. Li, P. Qi, T. Pang, C. Du, W. S. Lee, and M. Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.

- Y. Luo, Y. Song, X. Zhang, J. Liu, W. Wang, G. Chen, W. Su, and B. Zheng. Deconstructing long chain-of-thought: A structured reasoning optimization framework for long cot distillation. arXiv preprint arXiv:2503.16385, 2025.

- Y. Min, Z. Chen, J. Jiang, J. Chen, J. Deng, Y. Hu, Y. Tang, J. Wang, X. Cheng, H. Song, et al. Imitate, explore, and self-improve: A reproduction report on slow-thinking reasoning systems. arXiv preprint arXiv:2412.09413, 2024.

N. Muennighoff, Z. Yang, W. Shi, X. L. Li, L. Fei-Fei, H. Hajishirzi, L. Zettlemoyer, P. Liang, E. Candès,

and T. Hashimoto. s1: Simple test-time scaling, 2025. URL https://arxiv.org/abs/2501.19393. J. Pan, X. Wang, G. Neubig, N. Jaitly, H. Ji, A. Suhr, and Y. Zhang. Training software engineering

agents and verifiers with swe-gym. arXiv preprint arXiv:2412.21139, 2024.

C. Qian, E. C. Acikgoz, Q. He, H. WANG, X. Chen, D. Hakkani-Tür, G. Tur, and H. Ji. ToolRL: Reward is all tool learning needs. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

- S. Ross, G. Gordon, and D. Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pages 627–635. JMLR Workshop and Conference Proceedings, 2011.

- Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

K. Stechly, K. Valmeekam, A. Gundawar, V. Palod, and S. Kambhampati. Beyond semantics: The unreasonable effectiveness of reasonless intermediate tokens. arXiv preprint arXiv:2505.13775, 2025.

- Z. Tan, J. Yan, I.-H. Hsu, R. Han, Z. Wang, L. Le, Y. Song, Y. Chen, H. Palangi, G. Lee, A. R. Iyer, T. Chen, H. Liu, C.-Y. Lee, and T. Pfister. In prospect and retrospect: Reflective memory management for long-term personalized dialogue agents. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025.

G. Team. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. CoRR, abs/2507.06261, 2025.

- D. Tiapkin, D. Calandriello, J. Ferret, S. Perrin, N. Vieillard, A. Ramé, and M. Blondel. On teacher hacking in language model distillation. arXiv preprint arXiv:2502.02671, 2025.

- C. Wang, Y. Deng, Z. Lyu, L. Zeng, J. He, S. Yan, and B. An. Q*: Improving multi-step reasoning for llms with deliberative planning. arXiv preprint arXiv:2406.14283, 2024.

- P.-Y. Wang, T.-S. Liu, C. Wang, Y.-D. Wang, S. Yan, C.-X. Jia, X.-H. Liu, X.-W. Chen, J.-C. Xu, Z. Li, and Y. Yu. A survey on large language models for mathematical reasoning. arXiv preprint arXiv:2506.08446, 2025.

- Y. Wei, O. Duchenne, J. Copet, Q. Carbonneaux, L. Zhang, D. Fried, G. Synnaeve, R. Singh, and S. I. Wang. Swe-rl: Advancing llm reasoning via reinforcement learning on open software evolution. arXiv preprint arXiv:2502.18449, 2025.
- Z. Xi, W. Chen, B. Hong, S. Jin, R. Zheng, W. He, Y. Ding, S. Liu, X. Guo, J. Wang, H. Guo, W. Shen,

- X. Fan, Y. Zhou, S. Dou, X. Wang, X. Zhang, P. Sun, T. Gui, Q. Zhang, and X. Huang. Training large language models for reasoning through reverse curriculum reinforcement learning. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, 2024.

- J. Xie, K. Zhang, J. Chen, T. Zhu, R. Lou, Y. Tian, Y. Xiao, and Y. Su. Travelplanner: A benchmark for real-world planning with language agents. In Forty-first International Conference on Machine Learning, 2024.

- T. Xie, Z. Gao, Q. Ren, H. Luo, Y. Hong, B. Dai, J. Zhou, K. Qiu, Z. Wu, and C. Luo. Logic-rl: Unleashing llm reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2502.14768, 2025.

- W. Xiong, J. Yao, Y. Xu, B. Pang, L. Wang, D. Sahoo, J. Li, N. Jiang, T. Zhang, C. Xiong, and H. Dong. A minimalist approach to llm reasoning: from rejection sampling to reinforce. arXiv preprint arXiv:2504.11343, 2025.

- A. Yang, B. Yang, B. Hui, B. Zheng, B. Yu, C. Zhou, C. Li, C. Li, D. Liu, F. Huang, G. Dong, H. Wei, H. Lin, J. Tang, J. Wang, J. Yang, J. Tu, J. Zhang, J. Ma, J. Xu, J. Zhou, J. Bai, J. He, J. Lin, K. Dang, K. Lu, K. Chen, K. Yang, M. Li, M. Xue, N. Ni, P. Zhang, P. Wang, R. Peng, R. Men, R. Gao, R. Lin, S. Wang, S. Bai, S. Tan, T. Zhu, T. Li, T. Liu, W. Ge, X. Deng, X. Zhou, X. Ren, X. Zhang, X. Wei,

- X. Ren, Y. Fan, Y. Yao, Y. Zhang, Y. Wan, Y. Chu, Y. Liu, Z. Cui, Z. Zhang, and Z. Fan. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.

J. Yang, K. Leret, C. E. Jimenez, A. Wettig, K. Khandpur, Y. Zhang, B. Hui, O. Press, L. Schmidt, and

- D. Yang. Swe-smith: Scaling data for software engineering agents. arXiv preprint arXiv:2504.21798, 2025.

Y. Ye, Z. Huang, Y. Xiao, E. Chern, S. Xia, and P. Liu. Limo: Less is more for reasoning. arXiv preprint arXiv:2502.03387, 2025.

- E. Yeo, Y. Tong, M. Niu, G. Neubig, and X. Yue. Demystifying long chain-of-thought reasoning in llms. arXiv preprint arXiv:2502.03373, 2025.
- F. Yin, Z. Wang, I.-H. Hsu, J. Yan, K. Jiang, Y. Chen, J. Gu, L. Le, K.-W. Chang, C.-Y. Lee, H. Palangi, and T. Pfister. Magnet: Multi-turn tool-use data synthesis and distillation via graph translation. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025.

Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, Y. Yue, W. Dai, T. Fan, G. Liu, L. Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Y. Yue, Z. Chen, R. Lu, A. Zhao, Z. Wang, S. Song, and G. Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025.

J. Zhang and C. Zuo. Grpo-lead: A difficulty-aware reinforcement learning approach for concise mathematical reasoning in language models. arXiv preprint arXiv:2504.09696, 2025.

- X. Zhang, J. Wang, Z. Cheng, W. Zhuang, Z. Lin, M. Zhang, S. Wang, Y. Cui, C. Wang, J. Peng, et al. Srpo: A cross-domain implementation of large-scale reinforcement learning on llm. arXiv preprint arXiv:2504.14286, 2025.

### A. Illustration of SRL on SWE Tasks.

In Figure 4, we illustrate how we approach the SWE tasks with SRL. We take two consecutive actionobservation pairs from the expert trajectories in the given SFT data as context. We prompt the LLM to first think in monologues and then conclude with its action. Sequence similarity score is thus computed between model action and expert action in the trajectory.

Past Expert Action

Past Observation Error importing generated parsers, run `python setup.py antlr` to regenerate.

Past Expert Action <function=bash> <parameter=command>cd /testbed && python reproduce.py</parameter> </function>

Past Observation File created successfully at: /testbed/reprodu ce.py

<function=str_replace_editor> <parameter=command>create< /parameter> <parameter=path>/testbed/rep roduce.py</parameter> … </function>

GitHub Issue

Context

Model Thoughts

Expert Action <function=bash> cd /testbed && pip install -e . </function>

Model Action <function=bash> cd /testbed && python setup.py antlr </function>

<think>

The error message indicates that the generated parsers are missing or outdated…

LLM

</think>

Reward r = similarity(model action, expert action)

- Figure 4 | Illustration of applying SRL to SWE tasks. We take two rounds of the past expert actions and corresponding observations as context and prompt the LLM to think before reaches the next action. Reward is then calculated via comparing model action with the expert action.

### B. Experiment Details

#### B.1. Computation and Parameter Details

We use VeRL as the training framework for GRPO and implement SRL based on it. In Table 6, 7 and 8, we detail the hyperparameters that we used for SFT, GRPO and SRL. The batch size for GRPO is set to be smaller (128) than SRL (512) as the filter rate is high based on final answer correctness, for most of the questions are very difficult. We maintain a smaller batch size to avoid repeated sampling and achieve better performance. We further lay out the system prompts we used for SRL. Experiments were conducted on GPU clusters to the similar level of NVIDIA H100 80GB GPU.

- Table 6 | Supervised fine-tuning hyperparameters.

Data type bf16 Learning rate 5e-6

Global batch size 64

Scheduler Cosine Warmup ratio 0.3

Num train epochs 3

- Table 7 | GRPO hyperparameters.

Batch size 128 Max grad norm 1.0

Data type bf16 Learning rate 5e-7 Warmup ratio 0.0

Rollout temperature 1.0

Rollout number 8 KL loss coeff 0

- Table 8 | SRL hyperparameters.

Batch size 512 Max grad norm 1.0

Data type bf16 Learning rate 5e-7 Warmup ratio 0.0

Rollout temperature 1.0

Rollout number 8 KL loss coeff 0

- B.2. Prompt Details

- B.2.1. System Prompt for SRL Learning

System Prompt for SRL

You are a helpful assistant for solving mathematical problems. A user will provide a math problem, which may ↩→ include a partial solution. Your task is to continue the solution by providing the very next logical ↩→ step.

A user will ask you to solve a task. You should first draft your thinking process (inner monologue). Then, ↩→ generate the solution.

Your response format must follow the template below: <think> Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as

↩→ long as you want until you are confident to generate a correct solution. </think> Provide only the single, next step to continue the solution. Do not solve the entire problem.

- B.2.2. System Prompt for Evaluation

System Prompt for Math Evaluation

Please reason step by step, and put your final answer within \boxed{}. You can draft your thinking process ↩→ inside <think> </think> tags anywhere. This is your "scratch paper".

- B.3. Task Decomposition Details

We conduct the task decomposition for math via a data re-writing process. We first decide a format that is easy to parse trajectory into steps:

Template Format for Task Decomposition in Math

SOLUTION:

- 1. **{{STEP TITLE}}** {{STEP ACTION}}
- 2. **{{STEP TITLE}}** {{STEP ACTION}}
- 3. ....

Then we prompt Gemini-2.5-flash (Team, 2025) to conduct data conversion using the below prompt:

Prompt Used for Data Rewrite in Math

### ROLE You are an expert Math Tutor and Educational Content Formatter. Your goal is to take a raw math problem and

↩→ its solution, and rewrite it into a strictly structured, easy-to-follow, step-by-step guide.

### INPUT DATA I will provide you with:

- 1. The Math Question.
- 2. The Original Solution.

### OUTPUT FORMATTING RULES You must adhere to the following format strictly:

``` SOLUTION:

- 1. **{{STEP TITLE}}** {{STEP ACTION}}
- 2. **{{STEP TITLE}}** {{STEP ACTION}}
- 3. ... [Continue until solved] ``` ### CONTENT GUIDELINES

- 1. **Step Titles:** Must be bolded (**Title**). They should be short, imperative verb phrases (e.g., "Identify ↩→ the variables," "Isolate x," "Substitute values").
- 2. **Step Actions:** The equations and operations to execute according to the step title
- 3. Please follow the original thoughts and solutions provided in the ``Original Solution''. Do not solve the ↩→ task by your own.

### EXAMPLE [[TWO SHOT EXAMPLE PUT HERE]]

### CURRENT TASK

**Question:** [INSERT QUESTION HERE]

**Original Solution:** [INSERT SOLUTION HERE]

For the data after conversion cannot pass our formatting check, we will discard them.

### C. Supplementary Studies

In this section, we provide further analysis to supplement our experimental results in Section 5.2. Specifically, we aim to illustrate the mechanisms behind the performance improvements of our SRLfinetuned model. While recent work often associates improved reasoning with longer generation sequences, we first demonstrate that our model’s gains are not a result of increased verbosity. In addition, we show that the model develops a more sophisticated internal structure. Through the following examples of structured planning and post-hoc self-verification, we visualize how the model leverages specific reasoning patterns to navigate complex problem-solving tasks more effectively

[Figure 8]

- Figure 5 | Reasoning length distribution for base model and model trained with SRL.

than the base model.

Reasoning length. We investigate whether the performance gains from SRL can be simply attributed to an increase in reasoning length. In Figure 5, we plot the distribution of reasoning lengths (i.e., the number of words in the solution) for both the base model and our SRL-finetuned model. We observe no significant difference between the two distributions. This finding suggests that the performance improvement stems from enhanced planning and higher-quality reasoning, rather than merely an increased token budget at inference time.

Qualitative Case Studies. The first example demonstrates a priori structured planning, while the second showcases the model’s ability to perform mid-stream or post-hoc self-verification.

#### Example 2: Planning

Question: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make? <think> To determine Josh’s profit, we need to follow these steps:

- 1. Calculate the new value of the house after the repairs.
- 2. Subtract the initial cost of the house and the repair costs from the new value to find the profit. </think> First, let’s calculate the new value of the house after the repairs. The value of the house increased by 150% after the repairs. The initial cost of the house is $80,000, and the repairs cost $50,000. The total cost before the value increase is:

80, 000 + 50, 000 = 130, 000 The value of the house increased by 150% of its initial value of $80,000. So, the increase in value is:

150% × 80, 000 = 1.5 × 80, 000 = 120, 000 Therefore, the new value of the house is:

80, 000 + 120, 000 = 200, 000

Now, we subtract the total cost (initial cost of the house plus the repair costs) from the new value of the house to find the profit:

200, 000 − 130, 000 = 70, 000 So, Josh’s profit is:

|70000|
|---|

#### Example 3: Self-verification

Question: Which one of these constellations is not located along the Milky Way in the sky? Answer Choices: (A) Perseus (B) Cygnus (C) Scorpius (D) Leo

First, let’s recall which constellations are along the Milky Way. Perseus, Cygnus, and Scorpius are all well-known constellations that lie along the Milky Way. However, Leo is not one of them; it is located in the northern sky and does not align with the Milky Way. Therefore, the correct answer is (D) Leo.

<think> To double-check, I can recall the general locations of these constellations:

- - Perseus is in the northern sky and is part of the Perseus Arm of the Milky Way.
- - Cygnus is also in the northern sky and is part of the Orion Arm of the Milky Way.
- - Scorpius is in the southern sky and is part of the Sagittarius Arm of the Milky Way.
- - Leo is in the northern sky and is not aligned with the Milky Way. </think> The answer is (D) Leo.

### D. LLM Usage

Throughout the preparation of this manuscript, we utilized a LLM as an assistant with refinement of writing. Its role was strictly limited to improving grammar, rephrasing sentences for clarity, and correcting spelling. All intellectual contributions, ideas, and conclusions presented in this paper are entirely the work of the authors.

