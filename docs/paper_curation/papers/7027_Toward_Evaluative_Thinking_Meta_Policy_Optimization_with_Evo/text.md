# arXiv:2504.20157v2[cs.CL]16May2025

## Toward Evaluative Thinking: Meta Policy Optimization with Evolving Reward Models

Zae Myung Kim1, Chanwoo Park2, Vipul Raheja3, Suin Kim4, Dongyeop Kang1 1University of Minnesota, 2MIT, 3Grammarly, 4Elice {kim01756,dongyeop}@umn.edu, cpark97@mit.edu, raheja@grammarly.com, suin@elicer.com

### Abstract

Reward-based alignment methods for large language models (LLMs) face two key limitations: vulnerability to reward hacking, where models exploit flaws in the reward signal; and reliance on brittle, labor-intensive prompt engineering when LLMs are used as reward models. We introduce Meta Policy Optimization (MPO), a framework that addresses these challenges by integrating a meta-reward model that dynamically refines the reward model’s prompt throughout training. In MPO, the meta-reward model monitors the evolving training context and continuously adjusts the reward model’s prompt to maintain high alignment, providing an adaptive reward signal that resists exploitation by the policy. This meta-learning approach promotes a more stable policy optimization, and greatly reduces the need for manual reward prompt design. It yields performance on par with or better than models guided by extensively hand-crafted reward prompts. Furthermore, we show that MPO maintains its effectiveness across diverse tasks, from essay writing to mathematical reasoning, without requiring specialized reward designs. Beyond standard RLAIF, MPO’s meta-learning formulation is readily extensible to higher-level alignment frameworks. Overall, this method addresses theoretical and practical challenges in reward-based RL alignment for LLMs, paving the way for more robust and adaptable alignment strategies. The code and data can be accessed at: https://github.com/minnesotanlp/mpo

### 1 Introduction

Good thinkers must use another high-level thinking skill, namely, analysis and awareness of one’s own thinking—or metacognition [Buckley et al., 2015, Lord et al., 1979].

Recent advancements in reinforcement learning (RL) for large language model (LLM) training have marked a shift from tasks that prioritize human-like response generation [Ziegler et al., 2020, Stiennon et al., 2020, Ouyang et al., 2022a] to those that emphasize structured reasoning, such as mathematics and programming [Shao et al., 2024a, DeepSeek-AI et al., 2025]. Traditionally, human-aligned answering relies on reward models trained using binary comparison data, whereas structured reasoning tasks focus more on verifying the correctness of final answers or logical processes based on objective ground truth. However, many real-world challenges demand a synthesis of both approaches—requiring models to integrate human-aligned judgment with rigorous reasoning. This introduces significant complexity, as such tasks often lack objectively verifiable “golden answers,” yet still necessitate coherent and justifiable reasoning.

A scalable approach to subjective evaluation is to use an LLM as a judge—an approach commonly referred to as reinforcement learning with AI feedback (RLAIF)—where a fixed prompt is used to assess model performance on specific tasks. However, this method introduces three key challenges. First, calibration: effective scoring requires evaluation criteria that are sufficiently detailed and carefully calibrated to the quality and variability of the policy model’s outputs. Overly granular feedback on poor responses may be too complex to meaningfully guide improvement, while feedback that is too coarse or simplistic may fail to drive meaningful behavioral change in the model. Second,

Preprint. Work in progress.

reward hacking: it is well-known that when LLMs are trained with imperfect reward functions—

as is often the case with LLM-as-a-judge or reward models—they may learn to exploit shortcuts that maximize reward without genuinely improving output quality or alignment with human values [Amodei et al., 2016b, Everitt et al., 2021, Langosco et al., 2022, Pan et al., 2022]. These shortcuts can lead to responses that are formally rewarded yet misaligned with human intent or utility. Third, prompt-engineering overhead: substantial manual effort is often required for prompt engineering when generating training data for reward models or employing LLMs as proxy reward functions. This process introduces scalability bottlenecks and limits automation in alignment pipelines.

To address these issues, this work introduces Meta Policy Optimization (MPO) (Figure 1), a framework that augments existing rewardbased RLAIF pipeline by adding a meta reward model. Unlike a traditional reward model that simply scores the policy’s output based on a fixed prompt, the meta reward model monitors the evolving training landscape and adjusts or refines the prompt used by the standard reward model. Our design of MPO is inspired by the psychological concept of metacognition—the process of becoming aware of and reflecting on one’s own thinking [Flavell, 1979]—and its central role in evaluative thinking, a reflective, evidence-driven cognitive process that involves questioning, analyzing, and interpreting information to guide decision-making and continuous learning [Buckley et al., 2015].

Meta Policy Optimization for RLAIF

[Figure 1]

|Meta Reward Model “Senior Instructor”|
|---|

[Figure 2]

|Reward Model “Junior Instructor”|
|---|

[Figure 3]

[Figure 4]

|LLM “Student”|
|---|

Prompt Dataset

PPO

RLAIF

Figure 1: In standard RLAIF, the reward model used during proximal policy optimization (PPO) remains fixed throughout RL alignment. In contrast, MPO framework (in green) introduces a meta reward model that dynamically evolves the reward model based on the current training context, including the task prompt, sampled generations with associated scores, and the latest evaluation prompt. MPO leverages this contextual information to iteratively refine the evaluation prompt, enabling more adaptive and effective alignment.

Metacognitive awareness and control are essential to this process, enabling individuals to monitor reasoning, detect biases, and refine strategies through task assessment and reflection. Cognitive science research further supports this by showing that deliberate reflection fosters deeper, more robust learning [McCormick, 2003, Metcalfe and Kornell, 2005, Veenman et al., 2006, Efklides, 2006]. By carrying these principles into RL-based alignment for LLMs, we unlock several advantages:

- • Greater stability in RLAIF training: MPO dynamically adjusts reward model prompts to deliver context-sensitive scoring criteria based on the policy model’s performance, while also mitigating reward hacking—exploitation behaviors often seen in fixed-prompt setups.
- • Reduced prompt engineering burden: MPO iteratively refines and expands existing prompts within a single training cycle, eliminating repeated manual intervention.
- • Flexible and general framework: MPO can be used across diverse tasks (see Section 3.3) without major modifications to the training procedure.

### 2 Meta Policy Optimization

As noted in the introduction, our MPO approach draws inspiration from the fields of metacognition and evaluation. We elaborate on this conceptual motivation, then present a formal reinforcement learning formulation of MPO, followed by a detailed description of its implementation steps.

- 2.1 Evaluative Thinking Evaluative thinking (ET) [Buckley et al., 2015] is the intentional process of analyzing, interpreting, and assessing information to support thoughtful decision-making, playing a critical role in evaluation capacity building (ECB). However, current reward models in RL for LLM lack such metacognitive control. These models are typically trained on static human preference datasets or guided by fixed rubric prompts, and they remain unchanged during training. As the policy improves, the static reward model tends to collapse nuanced improvements into a coarse label—good enough—allowing early blind spots to persist and go uncorrected.

Motivated by the relationship between Evaluative Thinking (ET) and Evaluation Capacity Building (ECB)—where ET supports ECB by enhancing metacognition—we propose a Meta Reward Model (MRM) that guides the reward model to develop evaluative metacognition and become a more effective

- scorer. Specifically, the MRM follows the core principles of ET: evidence gathering, questioning, and reflective judgment (see Section 2.3 and Figure 3). Our MPO framework operationalizes ET by enabling the MRM to refine RM’s observational partitions over time. In this setup, the reward model improves through on-policy learning driven by the metacognitive signals of the MRM. Remark 1 (Depth and Breadth of ET). We posit that ET in the context of reinforcement learning for LLMs can be understood along two orthogonal dimensions: depth and breadth. This framework echoes Edward de Bono’s celebrated distinction between vertical and lateral thinking [De Bono, 1971]. Intuitively, depth corresponds to sequential, instance-specific reasoning—reflecting the degree of logical inference and deliberation required to evaluate a single case. In contrast, breadth captures the ability to generalize across varied instances, recognizing recurring patterns or abstract principles that inform evaluation in novel contexts. As illustrated in Figure 2, tasks such as mathematical reasoning exemplify vertical (deep) thinking, as they involve multi-step, case-specific deductions. In contrast, open-ended tasks like essay writing align more with lateral (broad) thinking, requiring evaluative generalization across diverse prompts. To empirically explore these dimensions, we evaluate four representative tasks—mathematical reasoning, ethical reasoning, summarization, and essay writing—each occupying a distinct region of the ET depth–breadth space in our experiments.

#### 2.2 Time-Varying Observations and Rewards by Evolving Reward Model

While ECB through ET provides a foundation for improving the RM, it does not fully capture the dynamics of RL, where learning is driven by signals from an ECB-enhanced RM. To address this gap, we introduce a mathematical framework that formalizes how the RM evolves under the influence of ET and how this evolving RM can be integrated into the training process of LLMs in Appendix B.

Mathematical Reasoning

Depth (instancespecific)

Ethical Reasoning

Evaluation are based on instancespecific reasoning traces.

Summarization

#### 2.2.1 Meta Policy Optimization Framework

Essay Writing

Employing a single static LLM scorer corresponds to a fixed observation partition, potentially too coarse to accurately capture nuanced reward differences. Such coarse partitioning groups many distinct states (e.g., texts or dialog histories) into overly coarse categories, leading to averaged rewards of the form:

Breadth (coverage-focused)

Evaluations are guided by a broad set of generalized rubric items.

Figure 2: Dimensions of Evaluative Thinking: Depth and Breadth.

R(o) = Es∈O

[r(s)],

o

which obscure state-specific details crucial for precise policy optimization (See the definition of Oo in Appendix B). Consequently, a static scoring mechanism struggles to converge towards the groundtruth reward r(s), limiting its ability to capture subtle, high-dimensional, or evolving reward criteria. In contrast, a meta policy framework addresses this limitation by adaptively refining observation partitions over successive iterations. Formally, the meta-rewarding process introduces progressively finer partitions:

, where Oo,t ⊆ Oo′,t−1 for some o′,t > 1. This iterative refinement enables increasingly discriminative reward signals:

{Oo,t}o∈Ωt

Rt(o) = Es∈O

[r(s)],

o,t

that better capture subtle variations in the state space. By adaptively partitioning the observation sets—splitting larger, coarse categories into smaller, targeted subsets as the policy’s performance improves or as new dimensions of evaluation emerge—the meta-rewarding evaluator progressively sharpens the granularity and efficacy of reinforcement signals. This dynamic refinement is particularly beneficial in complex LLM-driven tasks, ensuring that policy updates become more targeted and aligned with nuanced performance improvements, ultimately facilitating advanced policy learning.

###### Evidence Gathering Questioning

|[Figure 5]<br><br>RM “Junior Instr.”<br><br>|Evaluation Rubric|
|---|
<br><br>[Figure 6]<br><br>LLM “Student”<br><br>[Figure 7]<br><br>Prompt Dataset<br><br>|Task Description|
|---|
<br><br>Prompt Instruction<br><br>(Reference Solution)<br><br>Student Response<br><br>Evaluation Score|
|---|

|MetaAnalysis<br><br>⋅ Checks accuracy of scoring ⋅ Evaluates scoring criteria ⋅ Gives refinement suggestions<br><br>[Figure 8]| |
|---|---|
| |Reflective Judgment|
|MetaRefinement<br><br>⋅ Expands rubric with new criteria ⋅ Refines existing criteria with examples ⋅ Updates scoring ranges<br><br>MetaMerging<br><br>⋅ Combines multiple sets of refinements ⋅ Ensures a comprehensive and goal-aligned rubric ⋅ Preserves proper structure and formatting<br><br>|Evaluation Rubric✱|
|---|
<br><br>| | |Evaluation|Rubric’| | |
|---|---|---|---|---|---|
<br><br>[Figure 9]<br><br>[Figure 10]| |

- Figure 3: The three Meta Policy Optimization steps—meta-analysis, meta-refinement, and metamerging—are carried out by the meta reward model and operate over a broader input context than that used by the reward model.

#### 2.3 Implementation: Meta Reward Model

The MPO framework is implemented through the introduction of a meta reward model (MRM), which can be conceptually viewed as a “senior instructor” providing guidance to a “junior instructor” on how to evaluate the work produced by a “student.” As illustrated in Figure 3, the MRM oversees the broader training context and issues targeted refinements to the evaluation rubric (or prompt) used by the reward model (RM), which plays the role of a “junior instructor.” These refinements are informed by inputs drawn from three sources: the prompt dataset, the policy model (the “student”), and the reward model itself. Specifically, at every fixed k training batch steps, the MRM performs an MPO procedure by processing contextual input sources through general meta-level prompts, which are designed to be task-agnostic and applicable across all tasks. From the prompt dataset, the MRM processes the task description, a set of n task-specific prompt instructions, and—when available—the corresponding n reference solutions. It also receives the policy model’s n generated responses to these prompts, the current version of the evaluation rubric, and the scores assigned by the RM using that rubric. These n input samples are randomly selected from the training batches accumulated since the last MPO step. Leveraging this rich contextual input, the MRM identifies weaknesses or gaps in the current rubric and prescribes increasingly fine-grained and targeted evaluation criteria. This refinement process is triggered every fixed k training batches and follows the three-stage procedure illustrated in Figure 3. At each stage, the MRM is prompted with meta-level instructions that are designed to be broadly applicable across a range of tasks.

Meta-Analysis. The first step of MPO involves processing the full input context to assess whether the RM’s scoring is accurate and reliable—particularly in cases where the student LLM may exploit loopholes of the RM through reward hacking. The MRM evaluates whether the current scoring criteria are sufficiently comprehensive and detailed, and prescribes necessary adjustments to improve evaluation quality and robustness. This step is particularly crucial, as it serves to detect loopholes in the RM’s evaluation logic early on.

Meta-Refinement. Building on the results of the meta-analysis, the next step is to construct a more refined rubric. The MRM begins by determining the appropriate number of evaluation criteria, then systematically expands each item— either by introducing new criteria or by enriching existing ones with more detailed descriptions and illustrative examples. Additionally, it adjusts the scoring scale to more effectively distinguish between varying levels of response quality.

Meta-Merging. As MPO samples n student responses, it generates n corresponding rubric refinements. In its final step, MPO merges these multiple refined instances into a single, coherent rubric prompt. Similar to the meta-refinement stage, it first determines the appropriate number of evaluation items based on the overall comprehensiveness of the proposed refinements, then constructs each item accordingly. This consolidated rubric then serves as the updated evaluation prompt for the RM in subsequent training steps.

### 3 Experiments

To investigate the effectiveness of MPO and its influence on training dynamics, we conduct three core experiments. Section 3.1 outlines our experimental setup. Section 3.2 evaluates the performance of MPO-aligned LLMs on an argumentative essay writing task, exploring different pairings of junior and senior instructors. Section 3.3 empirically shows that MPO generalizes to tasks demanding varying degrees of evaluative thinking. In Section 3.4, we analyze how the rubric prompts used by the RM (the “junior instructor”) evolve over successive MPO iterations. Finally, Section 3.5 presents a comparison between MPO-aligned models and models trained using an “oracle” prompt.

- 3.1 Experimental Setup

Policy Model. Throughout our experiments, we fix the policy model to a relatively small LLM: Qwen2-1.5B-Instruct [Yang et al., 2024]. This choice is motivated by two factors. First, we require a model with sufficient headroom for improvement across our target tasks. Since we use publicly available benchmarks with limited resources, we focus on smaller open-sourced models which could show clear effects and values of the proposed framework. Extending it to larger models and more variants of model families remains an interesting direction for future study. Second, some taskssuch as mathematical reasoning—require generating over 1000 tokens, which imposes a significant memory load during the PPO step. Larger models exceed the capacity of our available GPUs, making Qwen2-1.5B-Instruct a practical and scalable option.

(Meta) Reward Models. For reward modeling, we use Qwen2.5-32B-Instruct-AWQ and Qwen2.572B-Instruct-AWQ [Qwen et al., 2025], exploring all four junior-senior RM–MRM size combinations: 32b_32b, 32b_72b, 72b_32b, and 72b_72b, where the first and second terms denote the sizes of RM and MRM, respectively. Using larger models as (M)RMs is feasible since only inference is required, which can be efficiently handled by an LLM-serving framework [Zheng et al., 2024]. We also include three fixed-RM baselines without MPO: one using the initial MPO prompt, one using a domain expert–crafted prompt (for the essay writing task only) [Hamner et al., 2012], and one using an iteratively refined prompt via AutoPrompt [Levi et al., 2024] with GPT-4o. These baselines are denoted as {RM size}_iter0, {RM size}_expert, and {RM size}_AP, respectively in tables.

Implementation. Our MPO framework relies on online (meta) reward models implemented as LLMs with an interchangeable prompt mechanism. To support this, we extend the TRL library [von Werra et al., 2020] by implementing a prompt-based, online reward model, where LLM-based RMs are hosted using the SGLang framework [Zheng et al., 2024]. Additionally, we extend the “PPOTrainer” class in trl to a customized “MPOTrainer,” which integrates the MPO refinement steps directly into the training loop along with other necessary modifications.1

We note that the MPO framework is general and modular, and can be integrated into other RL optimization techniques that rely on reward models or functions—such as GRPO [Shao et al., 2024b]—to enable dynamic rubric refinement and more adaptive reward shaping.

- 3.2 Impact of MPO on Essay Writing Task

Setup. We train four policy models using MPO with different RM–MRM pairings as described in Section 3.1, and another four using vanilla PPO with fixed RM prompts. The essay writing dataset is compiled by Kim et al. [2025] and includes writing instructions drawn from diverse sources, such as English proficiency exams, a persuasion corpus, and the Change My View (CMV) subreddit. The training set comprises 26,013 samples, and the test set includes 4,096 samples. Both MPO and PPO are trained for a single epoch over the training set, with MPO refinement steps occurring every 10 batch steps during training. We utilize eight NVIDIA-A100 80GB GPUs to RL-train the models with an effective combined batch size of 64, where each RL episode spans up to 400 tokens. This configuration yields 40 MPO refinement steps over the course of one-epoch training.

After single-epoch training, we generate essays for all test prompts using the final checkpoints and evaluate them through 10,000 head-to-head comparisons following the Elo-based Chatbot Arena framework [Chiang et al., 2024], with GPT-4o serving as the impartial judge and a small K-factor of

###### 4 applied after each match to maintain rating stability. The resulting Elo scores, summarized in Table 1, provide a relative ranking of essay quality across the eight models.

1Please refer to our codebase for exact details.

RL Reward

Normalized RL Reward

KL Divergence

120

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

120

0.8

100

RLReward(Normalized)

100

80

KLDivergence

0.6

80

RLReward

60

60

0.4

40

40

0.2

- 20

20

0

0

0.0

0 5000 10000 15000 20000 25000

0 5000 10000 15000 20000 25000

0 5000 10000 15000 20000 25000

Global Batch Step

Global Batch Step

Global Batch Step

Model

MPO-32b_32b MPO-32b_72b MPO-72b_32b MPO-72b_72b PPO-32b_iter0 PPO-72b_iter0 PPO-32b_manual PPO-72b_manual

- Figure 4: Training curves for eight essay-writing policy models, each pairing different-sized reward models (RM) and meta-reward models (MRM). The RL Reward and Normalized RL Reward plots show how reward values evolve over global batch steps, capturing the quality of generated responses as judged by the corresponding RM at each point in training. The normalized plot is obtained by dividing the RL reward values by the total attainable score defined by the current rubric, providing a more consistent view of reward dynamics across evolving evaluation criteria. Kullback-Leibler (KL) divergence quantifies the extent of policy drift throughout training. The dotted vertical lines indicate MPO rounds, which occur every batch size × MPO step—640 steps in our setup.

Training Curves. Figure 4 shows the training curves for eight models, each corresponding to a different pairing of RM and MRM sizes in the essay writing task. Note that we use raw RL reward values for optimization, as they preserve fine-grained distinctions in reward signals—amplifying reward sensitivity and improving optimization effectiveness—despite a slight deviation from our formal theoretical framework. The four models employing MPO exhibit increasing RL reward values over time, driven by successive MPO steps that refine the evaluation prompt by (1) introducing a greater number of evaluation criteria and (2) expanding the scoring ranges associated with those criteria as training progresses. While these curves suggest that training is progressing adaptively as intended, the reward scores are themselves adaptive—reflecting evolving rubrics—so differences in reward values across models (shown in the RL Reward plot) do not necessarily translate to corresponding performance gaps on the final test set.

Results. First, we evaluate the performance of all MPO-aligned models by computing their Elo ratings (see left panel of Table 1). The results show that performance improves primarily with the size of the RM, followed by that of the MRM, with the highest scores achieved when 72B LLMs are used for both components. Next, we evaluate MPO models and baseline methods separately based on the RM size, keeping the MRM fixed at 72B (see right panel of Table 1). The results show that the MPO-aligned model consistently outperforms all baselines, including the PPO-aligned model using an expert-crafted evaluation prompt. We note that the PPO-aligned model using the expert prompt and a 72B RM resulted in a failed training run due to reward hacking. In this case, the policy model frequently generated responses consisting solely of a title—e.g., “Robots and the Future of Humanity:

A Dilemma of Progress and Responsibility”—yet still often received high scores of 4 or 5 out of 5.2 It is worth noting that, at certain points during training, the MPO-aligned models also displayed instances of reward hacking—such as generating responses in Chinese, producing title-only completions, or outputting overly short and degenerate answers. However, these cases were identified and addressed during the MPO procedure. An example of such reward hacking and how it was mitigated through the MPO steps is provided in Appendix C.

The main takeaway from this experiment is that heavily hand-engineered prompts by domain experts can yield the strongest performance among PPO-aligned models (the 32B RM case). However, the fixed-prompt PPO setup remains vulnerable to reward hacking, which can lead to policy collapse, as observed in the 72B RM case. In contrast, incorporating the MPO procedure—even when starting from the basic initial prompt (i.e., iter0)—produces a higher-quality model than PPO with the expert-crafted prompt.

2With a fixed random seed, we repeated the same configuration two additional times, and each run consistently resulted in a failed policy model.

- Table 1: Elo ratings for the essay writing task. Left: Comparison across RM-MRM variations (2,000 pairwise comparisons). Right: Comparison across LLMs (5,000 pairwise comparisons). Each rating includes standard deviation computed across 5 runs, denoted by s. Red indicates a case where training converged to a degenerate policy due to reward hacking.

RM

MPO w/ MRM (ours)

32B 72B

32B 973s=±12 1005s=±5 72B 985s=±7 1037s=±10

RM

MPO (ours) PPO Base MRM-72B iter0 expert AP LLM

32B 1168s=±7 984s=±10 1084s=±17 970s=±15 794s=±11 72B 1244s=±8 1111s=±7 706s=±12 1050s=±15 889s=±10

3.3 Generalization of MPO across Different Tasks

As discussed in Section 2.1, evaluating written essays requires a form of evaluative thinking that is more breadth-focused—guided by a broad set of generalized rubric items that can be applied across diverse prompts and writing styles. In this experiment, we apply the MPO framework to tasks that vary in their demands along the depth and breadth dimensions of evaluative thinking, examining whether the benefits of the proposed approach persist across this spectrum. As illustrated in Figure 2, we evaluate MPO across three additional tasks: summarization, ethical reasoning, and mathematical reasoning.

For these three tasks, we compare performance across four models: Base LLM, the original Qwen21.5B-Instruct model prior to any alignment; 32b_iter0, a vanilla PPO-aligned model using the initial evaluation prompt with the 32B Qwen model as the RM; 32b_AP, another PPO-aligned model using evaluation prompt iteratively refined via AutoPrompt [Levi et al., 2024] using GPT-4o; and 32b_32b, an MPO-aligned model using the 32B Qwen model for both the RM and MRM.

- 3.3.1 Summarization Task

For summarization, we train models on the BillSum benchmark [Kornilova and Eidelman, 2019] for one epoch, applying MPO refinements every 20 batch steps and generating 4.5K summaries for evaluation. We do not use gold summaries during training; instead, the quality of generated summaries is evaluated directly by the reward model. Performance is measured using ROUGE scores against human-written references, along with mean Elo ratings computed from five runs of 2,000 pairwise comparisons judged by GPT-4o. Results are reported in Table 2.

ROUGE-1 ROUGE-2 ROUGE-L ROUGE-Lsum Elo Rating

Base LLM 41.85 20.83 27.85 27.84 819s=±21 PPO 32b_iter0 45.97 23.57 30.29 30.29 953s=±19 PPO 32b_AP 45.92 21.98 28.80 30.22 1149s=±12 MPO 32b_32b 48.00 24.96 30.97 30.98 1079s=±12

Table 2: Performance of models was evaluated using ROUGE scores on the BillSum long-document summarization task and Elo ratings derived from 2,000 pairwise comparisons.

The MPO-aligned 32b_32b model outperforms the other three models in ROUGE scores against human-written gold summaries. However, in pairwise Elo evaluations, the PPO-aligned 32b_AP model receives the highest rating. This discrepancy may stem from the GPT-4o judge favoring outputs from models aligned using evaluation rubrics it helped generate. However, further experiments are needed to validate this hypothesis. Figure 6 presents an excerpt of the evaluation rubric used for the summarization task, showing how MPO expanded the rubric to incorporate criteria specifically relevant to assessing legislative bill summaries.

- 3.3.2 Ethical Reasoning Task

For ethical reasoning, we use the Anecdotes from the Scruples dataset [Lourie et al., 2020], a collection of over 32,000 ethically complex real-world situations labeled with community judgments. We randomly sample 13K anecdotes for training and 4.7K for testing, running a single training epoch with MPO steps performed every 10 batch steps, leading to 20 rubric refinements. Instead of relying on the binary ground truth labels, reward scores are based solely on the quality of ethical reasoning to encourage deeper reasoning development rather than optimizing for imbalanced label distributions.

Evaluation based on accuracy against the binary verdict labels is reported in Table 3. We observe that the MPO-aligned policy model generates ethical reasoning traces that result in a higher degree of alignment with human-annotated verdicts.

#### 3.3.3 Mathematical Reasoning Task

For mathematical reasoning, we use the MATH dataset [Hendrycks et al., 2021], which contains 12,500 high school competition-style problems across seven subjects, each with detailed step-by-step solutions for evaluating both final answers and reasoning processes. We train on 7.5K samples and test on 5K, clustering problems into 21 subject-cluster groups and maintaining a separate evaluation prompt for each, with MPO refinements triggered every 30 batch steps. The reward model follows a plan-then-execute strategy adapted from Saha et al. [2025]3, formulating an evaluation plan before scoring student responses with rubric-guided assessment. The reward model operates in two scoring modes, selected at random with equal probability. The first mode assigns scores based solely on exact match of the final answer span, while the second mode uses LLM-based evaluation to assess the correctness of the reasoning chain used to arrive at the solution.

Table 3: Accuracy on the ethical reasoning (Scruples–Anecdotes) and mathematical reasoning (MATH) benchmarks.

Accuracy (%, correct / total) Scruples–Anecdotes MATH

Model

Base LLM 33.80 (1601/4736) 17.90 (905/5056) PPO 32b_iter0 63.79 (3021/4736) 48.77 (2466/5056) PPO 32b_AP 58.68 (2779/4736) 49.29 (2492/5056) MPO 32b_32b 68.60 (3249/4736) 50.38 (2547/5056)

Accuracy results based on the exact match of reference answers are reported in Table 3. The first notable point is that our proposed evaluation approach significantly improves the performance of the base 1.5B LLM. This suggests that tasks requiring depth-oriented evaluative thinking—such as mathematical reasoning—can benefit substantially from structured, plan-then-execute evaluation. As with the other tasks, applying the MPO framework through refinement of meta-level guidelines further enhances the performance of our proposed plan-then-execute reward model. However, the relative improvement is more modest compared to the other tasks. We hypothesize that this is due to the highly instance-specific nature of mathematical reasoning, where scoring relies heavily on whether the sequential logic leads precisely to the correct answer. In contrast, the meta-level guidelines—constructed from sampled instances—tend to remain relatively general. Nonetheless, subject-specific refinements still contribute to performance gains. Developing more granular and tailored meta-guidelines could yield additional improvements, which we leave for future work.

#### 3.4 Evolution of Evaluation Rubric

Mean Len. of Rubric Item over Iterations (Essay Writing)

| |
|---|

0.8

Prompt Version

1600

MeanTotalScore(Normalized)

1400

- 0

- 1

- 2

0.7

1200

MeanLength

0.6

1000

5 7

800

0.5

20 40

32b_32b 32b_72b 72b_32b 72b_72b

600

0.4

400

200

0.3

0 5 10 15 20 25 30 35 40 Iteration

0 50 100 150 200 250 300 350 400

Checkpoint

(a) Rubric length during MPO refinements

(b) Total score across rubric refinements

- Figure 5: (a) Mean length of rubric items for essay writing task across the MPO-aligned models. (b) Mean normalized total rubric score for 1,000 test essays (generated by the 32b_72b model) across successive evaluation prompt refinements.

In this experiment, we seek to uncover holistic patterns across rubric evolution, particularly focusing on essay writing task.

Mean Lengths of Rubric Items. For the essay writing task, we track how the average length of each rubric item evolves over successive MPO refinements. Notably, the length increases sharply after the first refinement and continues to grow steadily over the next 5 to 10 iterations, before plateauing during the remaining stages of training (Fig. 5a). Manual inspection confirms that most meaningful

3The key difference is that their work focuses on building an LLM-as-a-judge model to select the more favorable generation in an iterative DPO setting, whereas our approach centers on absolute scoring within a single round of PPO training.

rubric changes occur within the first 5 to 10 iterations, after which the refinements become relatively minor.4 This observation suggests that the current fixed MPO schedule could be improved through dynamic adjustment based on training dynamics—an avenue worth exploring in future work.

Mean Total Scores Across Successive Rubrics Figure 5b plots the normalized mean total scores for 1,000 randomly sampled test essays, generated by the 32b_72b model and evaluated using both early and late-stage RM prompts evolved through MPO. The general pattern shows that mean total

- scores increase over the course of training, reflecting the policy model’s improving output quality. However, earlier versions of the evaluation prompts tend to assign higher scores to samples, largely because their coarser and less fine-grained criteria make it easier for responses to meet the rubric’s standards. In contrast, later-stage prompts, which feature more detailed and discriminating rubric items, assign lower scores, as they capture subtler flaws and impose stricter evaluation standards. This trend highlights how rubric refinement not only tightens evaluation but also provides more accurate and demanding feedback to guide policy improvement.

#### 3.5 Comparison against Human-Engineered Oracle Prompt

For the essay writing task, Kim et al. [2025] handengineered an evaluation prompt for the RM through an iterative process involving over 60 vanilla PPO training runs. The prompt was refined based on reward hacking behaviors observed from the same Qwen2-1.5B-Instruct model and adjusted to better align with human evaluation scores by modifying evaluation criteria. We treat this prompt as an oracle prompt, as it represents a hand-evolved rubric distilled through extensive PPO training experience. Figure 7 in Appendix D compares the oracle evaluation prompt with the final version produced by the MPO framework. Unlike the oracle, the MPO-evolved prompt was generated within a single training epoch and features a diverse set of rubric items specifically tailored for essay evaluation. Each item is accompanied by detailed descriptions that support fine-grained point allocation across a defined scoring range.

Table 4: Elo ratings for the essay writing task, evaluated through 5,000 pairwise comparisons across 4 LLMs. Each rating is accompanied by a standard deviation, computed across 5 independent experiments, denoted by s.

MPO (ours) PPO

MRM-72B oracle

RM

32B 1039s=±11 894s=±13 72B 1064s=±14 1003s=±17

Table 4 presents the Elo ratings for the 32b_72b and 72b_72b MPO-aligned models alongside PPOaligned models using the oracle prompt, based on 5,000 GPT-4o-judged pairwise comparisons. Both MPO-aligned models outperform the PPO baselines, with the 72b_72b MPO model achieving the highest Elo rating. These results demonstrate that the MPO framework can automatically generate evaluation prompts that surpass the quality of extensively hand-engineered oracle prompts, without requiring any task-specific manual prompt design.

### 4 Conclusion

This work introduces Meta Policy Optimization (MPO), a novel framework that enhances reinforcement learning from human or AI feedback by dynamically evolving the evaluation rubrics used by reward models. Grounded in the cognitive principles of evaluative thinking and metacognition, MPO empowers reward models to not only evaluate policy outputs but also reflect on and refine their scoring criteria over time. Across diverse tasks—including essay writing, summarization, ethical reasoning, and mathematical problem solving—MPO consistently improves alignment and outperforms models relying on static, manually crafted prompts.

Beyond improved empirical performance, MPO offers a new lens for thinking about reward modeling as an adaptive, self-improving process. Our analysis further reveals that the evolved rubrics exhibit deeper linguistic structure, suggesting the emergence of more principled evaluation schemas.

For future work, several promising directions emerge: dynamically adjusting MPO frequency based on training dynamics, scaling to more granular rubric specializations, exploring multi-turn dialogues and interactive settings, and integrating MPO with advanced optimization algorithms beyond PPO. Finally, extending MPO to support multi-agent alignment or long-horizon tasks could open new pathways toward more generalizable and cognitively aligned learning systems.

4We provide the sequence of evolved rubrics in Appendix F

### References

A. Ahmadian, C. Cremer, M. Gallé, M. Fadaee, J. Kreutzer, O. Pietquin, A. Üstün, and S. Hooker. Back to basics: Revisiting REINFORCE-style optimization for learning from human feedback in LLMs. In L.-W. Ku, A. Martins, and V. Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12248–12267, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics. doi: 10.18653/v1/ 2024.acl-long.662. URL https://aclanthology.org/2024.acl-long.662.

- D. Amodei, C. Olah, J. Steinhardt, P. Christiano, J. Schulman, and D. Mané. Concrete problems in ai

- safety, 2016a.

- D. Amodei, C. Olah, J. Steinhardt, P. Christiano, J. Schulman, and D. Mané. Concrete problems in ai

safety, 2016b. URL https://arxiv.org/abs/1606.06565.

- Y. Bai, A. Jones, K. Ndousse, A. Askell, A. Chen, N. DasSarma, D. Drain, S. Fort, D. Ganguli, T. Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

J. Buckley, T. Archibald, M. Hargraves, and W. M. Trochim. Defining and teaching evaluative thinking: Insights from research on critical thinking. American Journal of Evaluation, 36(3): 375–388, 2015.

L. Chen, C. Zhu, D. Soselia, J. Chen, T. Zhou, T. Goldstein, H. Huang, M. Shoeybi, and B. Catanzaro. Odin: Disentangled reward mitigates hacking in rlhf, 2024a. URL https://arxiv.org/abs/ 2402.07319.

- Z. Chen, Y. Deng, H. Yuan, K. Ji, and Q. Gu. Self-play fine-tuning convertsweak language models to strong language models. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024b.

W.-L. Chiang, L. Zheng, Y. Sheng, A. N. Angelopoulos, T. Li, D. Li, H. Zhang, B. Zhu, M. Jordan, J. E. Gonzalez, and I. Stoica. Chatbot arena: An open platform for evaluating llms by human preference, 2024. URL https://arxiv.org/abs/2403.04132.

T. Coste, U. Anwar, R. Kirk, and D. Krueger. Reward model ensembles help mitigate overoptimization. In The Twelfth International Conference on Learning Representations, 2024. URL https:// openreview.net/forum?id=dcjtMYkpXx.

- E. De Bono. The Use of Lateral Thinking. Pelican books. Penguin Books, 1971. ISBN

9780140214468. URL https://books.google.com/books?id=2Fd-AAAAMAAJ. DeepSeek-AI, D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi,

- X. Zhang, X. Yu, Y. Wu, Z. F. Wu, Z. Gou, Z. Shao, Z. Li, Z. Gao, A. Liu, B. Xue, B. Wang,

- B. Wu, B. Feng, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, D. Dai, D. Chen, D. Ji, E. Li, F. Lin, F. Dai, F. Luo, G. Hao, G. Chen, G. Li, H. Zhang, H. Bao, H. Xu, H. Wang, H. Ding, H. Xin, H. Gao, H. Qu, H. Li, J. Guo, J. Li, J. Wang, J. Chen, J. Yuan, J. Qiu, J. Li, J. L. Cai,

- J. Ni, J. Liang, J. Chen, K. Dong, K. Hu, K. Gao, K. Guan, K. Huang, K. Yu, L. Wang, L. Zhang, L. Zhao, L. Wang, L. Zhang, L. Xu, L. Xia, M. Zhang, M. Zhang, M. Tang, M. Li, M. Wang, M. Li, N. Tian, P. Huang, P. Zhang, Q. Wang, Q. Chen, Q. Du, R. Ge, R. Zhang, R. Pan, R. Wang, R. J. Chen, R. L. Jin, R. Chen, S. Lu, S. Zhou, S. Chen, S. Ye, S. Wang, S. Yu, S. Zhou, S. Pan, S. S. Li, S. Zhou, S. Wu, S. Ye, T. Yun, T. Pei, T. Sun, T. Wang, W. Zeng, W. Zhao, W. Liu, W. Liang,

- W. Gao, W. Yu, W. Zhang, W. L. Xiao, W. An, X. Liu, X. Wang, X. Chen, X. Nie, X. Cheng,
- X. Liu, X. Xie, X. Liu, X. Yang, X. Li, X. Su, X. Lin, X. Q. Li, X. Jin, X. Shen, X. Chen, X. Sun,

- X. Wang, X. Song, X. Zhou, X. Wang, X. Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Y. Zhang, Y. Xu,
- Y. Li, Y. Zhao, Y. Sun, Y. Wang, Y. Yu, Y. Zhang, Y. Shi, Y. Xiong, Y. He, Y. Piao, Y. Wang,

- Y. Tan, Y. Ma, Y. Liu, Y. Guo, Y. Ou, Y. Wang, Y. Gong, Y. Zou, Y. He, Y. Xiong, Y. Luo, Y. You,

- Y. Liu, Y. Zhou, Y. X. Zhu, Y. Xu, Y. Huang, Y. Li, Y. Zheng, Y. Zhu, Y. Ma, Y. Tang, Y. Zha,

- Y. Yan, Z. Z. Ren, Z. Ren, Z. Sha, Z. Fu, Z. Xu, Z. Xie, Z. Zhang, Z. Hao, Z. Ma, Z. Yan, Z. Wu,
- Z. Gu, Z. Zhu, Z. Liu, Z. Li, Z. Xie, Z. Song, Z. Pan, Z. Huang, Z. Xu, Z. Zhang, and Z. Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948.

- C. Denison, M. MacDiarmid, F. Barez, D. Duvenaud, S. Kravec, S. Marks, N. Schiefer, R. Soklaski, A. Tamkin, J. Kaplan, et al. Sycophancy to subterfuge: Investigating reward-tampering in large language models. arXiv preprint arXiv:2406.10162, 2024.

- M. Ding, S. Chakraborty, V. Agrawal, Z. Che, A. Koppel, M. Wang, A. Bedi, and F. Huang. SAIL: Self-improving efficient online alignment of large language models. In ICML 2024 Workshop on Theoretical Foundations of Foundation Models, 2024. URL https://openreview.net/forum? id=9m8dF6oAsd.

- A. Efklides. Metacognition and affect: What can metacognitive experiences tell us about the learning process? Educational research review, 1(1):3–14, 2006.

- J. Eisenstein, C. Nagpal, A. Agarwal, A. Beirami, A. N. D’Amour, K. D. Dvijotham, A. Fisch,
- K. A. Heller, S. R. Pfohl, D. Ramachandran, P. Shaw, and J. Berant. Helping or herding? reward model ensembles mitigate but do not eliminate reward hacking. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id=5u1GpUkKtG.

K. Ethayarajh, W. Xu, N. Muennighoff, D. Jurafsky, and D. Kiela. Kto: Model alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306, 2024.

T. Everitt, M. Hutter, R. Kumar, and V. Krakovna. Reward tampering problems and solutions in reinforcement learning: A causal influence diagram perspective. Synthese, 198(Suppl 27): 6435–6467, 2021.

J. H. Flavell. Metacognition and cognitive monitoring: A new area of cognitive–developmental inquiry. American psychologist, 34(10):906, 1979.

L. Fluri, L. Lang, A. Abate, P. Forré, D. Krueger, and J. Skalse. The perils of optimizing learned reward functions: Low training error does not guarantee low regret. arXiv preprint arXiv:2406.15753, 2024.

J. Fu, X. Zhao, C. Yao, H. Wang, Q. Han, and Y. Xiao. Reward shaping to mitigate reward hacking in rlhf. arXiv preprint arXiv:2502.18770, 2025.

- L. Gao, J. Schulman, and J. Hilton. Scaling laws for reward model overoptimization. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 10835–10866. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr. press/v202/gao23h.html.

- B. Hamner, J. Morgan, lynnvandev, M. Shermis, and T. V. Ark. The hewlett foundation: Automated essay scoring. https://kaggle.com/competitions/asap-aes, 2012. Kaggle.

- D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt. Measuring mathematical problem solving with the math dataset. In J. Vanschoren and S. Yeung, editors, Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1, 2021. URL https://datasets-benchmarks-proceedings.neurips.cc/paper_ files/paper/2021/file/be83ab3ecd0db773eb2dc1b0a17836a1-Paper-round2.pdf.

- Z. M. Kim, K. Lee, P. Zhu, V. Raheja, and D. Kang. Threads of subtlety: Detecting machinegenerated texts through discourse motifs. In L.-W. Ku, A. Martins, and V. Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5449–5474, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.298. URL https://aclanthology.org/2024. acl-long.298/.

- Z. M. Kim, A. Ramachandran, F. Tavazoee, J.-K. Kim, O. Rokhlenko, and D. Kang. Align to structure: Aligning large language models with structural information, 2025. URL https: //arxiv.org/abs/2504.03622.

A. Kornilova and V. Eidelman. BillSum: A corpus for automatic summarization of US legislation. In L. Wang, J. C. K. Cheung, G. Carenini, and F. Liu, editors, Proceedings of the 2nd Workshop on New Frontiers in Summarization, pages 48–56, Hong Kong, China, Nov. 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-5406. URL https://aclanthology.org/ D19-5406/.

- V. Krakovna, J. Uesato, V. Mikulik, M. Rahtz, T. Everitt, R. Kumar, Z. Kenton, J. Leike, and S. Legg. Specification gaming: the flip side of ai ingenuity. DeepMind Blog, 3, 2020.

L. L. D. Langosco, J. Koch, L. D. Sharkey, J. Pfau, and D. Krueger. Goal misgeneralization in deep reinforcement learning. In K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato, editors, Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 12004–12019. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr.press/v162/langosco22a.html.

H. Le, Y. Wang, A. D. Gotmare, S. Savarese, and S. C. H. Hoi. Coderl: Mastering code generation through pretrained models and deep reinforcement learning. Advances in Neural Information Processing Systems, 35:21314–21328, 2022.

H. Lee, C. Park, D. Abel, and M. Jin. A hypothesis on black swan in unchanging environments. arXiv preprint arXiv:2407.18422, 2024.

E. Levi, E. Brosh, and M. Friedmann. Intent-based prompt calibration: Enhancing prompt optimization with synthetic boundary cases, 2024.

C.-Y. Lin. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain, July 2004. Association for Computational Linguistics. URL https://aclanthology.org/W04-1013/.

H. Liu, C. Sferrazza, and P. Abbeel. Chain of hindsight aligns language models with feedback. In The Twelfth International Conference on Learning Representations, 2024a. URL https: //openreview.net/forum?id=6xfe4IVcOu.

T. Liu, W. Xiong, J. Ren, L. Chen, J. Wu, R. Joshi, Y. Gao, J. Shen, Z. Qin, T. Yu, D. Sohn, A. Makarova, J. Liu, Y. Liu, B. Piot, A. Ittycheriah, A. Kumar, and M. Saleh. Rrm: Robust reward model training mitigates reward hacking. ArXiv, abs/2409.13156, 2024b. URL https: //api.semanticscholar.org/CorpusID:272770255.

C. G. Lord, L. Ross, and M. R. Lepper. Biased assimilation and attitude polarization: The effects of prior theories on subsequently considered evidence. Journal of personality and social psychology, 37(11):2098, 1979.

N. Lourie, R. L. Bras, and Y. Choi. Scruples: A corpus of community ethical judgments on 32,000 real-life anecdotes. arXiv e-prints, 2020.

- W. C. Mann and S. A. Thompson. Rhetorical structure theory: A theory of text organization. University of Southern California, Information Sciences Institute Los Angeles, 1987.

C. B. McCormick. Metacognition and learning. Handbook of psychology, pages 79–102, 2003. Y. Meng, M. Xia, and D. Chen. SimPO: Simple preference optimization with a reference-free reward.

In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=3Tzcot1LKb.

J. Metcalfe and N. Kornell. A region of proximal learning model of study time allocation. Journal of Memory and Language, 52(4):463–477, 2005. ISSN 0749-596X. doi: https://doi.org/10. 1016/j.jml.2004.12.001. URL https://www.sciencedirect.com/science/article/pii/ S0749596X04001330. Special Issue on Metamemory.

Y. Miao, S. Zhang, L. Ding, R. Bao, L. Zhang, and D. Tao. InfoRM: Mitigating reward hacking in RLHF via information-theoretic reward modeling. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id= 3XnBVK9sD6.

- Y. Miao, S. Zhang, L. Ding, Y. Zhang, L. Zhang, and D. Tao. The energy loss phenomenon in rlhf: A new perspective on mitigating reward hacking, 2025. URL https://arxiv.org/abs/2501. 19358.

L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. F. Christiano, J. Leike, and R. Lowe. Training language models to follow instructions with human feedback. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 27730–27744. Curran Associates, Inc., 2022a. URL https://proceedings.neurips.cc/paper_files/paper/ 2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf.

L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022b.

L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. Christiano, J. Leike, and R. Lowe. Training language models to follow instructions with human feedback. In Proceedings of the 36th International Conference on Neural Information Processing Systems, NIPS ’22, Red Hook, NY, USA, 2022c. Curran Associates Inc. ISBN 9781713871088.

A. Pan, K. Bhatia, and J. Steinhardt. The effects of reward misspecification: Mapping and mitigating misaligned models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=JYtwGwIL7ye.

- R. Y. Pang, V. Padmakumar, T. Sellam, A. Parikh, and H. He. Reward gaming in conditional text generation. In A. Rogers, J. Boyd-Graber, and N. Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4746–4763, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.262. URL https://aclanthology.org/2023.acl-long.262/.
- R. Y. Pang, W. Yuan, K. Cho, H. He, S. Sukhbaatar, and J. Weston. Iterative reasoning preference optimization. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 116617–116637. Curran Associates, Inc., 2024. URL https://proceedings.neurips.cc/paper_files/ paper/2024/file/d37c9ad425fe5b65304d500c6edcba00-Paper-Conference.pdf.

C. Park, M. Liu, D. Kong, K. Zhang, and A. Ozdaglar. Rlhf from heterogeneous feedback via personalization and preference aggregation. arXiv preprint arXiv:2405.00254, 2024.

C. Park, S. Han, X. Guo, A. Ozdaglar, K. Zhang, and J.-K. Kim. Maporl: Multi-agent post-cotraining for collaborative large language models with reinforcement learning. arXiv preprint arXiv:2502.18439, 2025.

- E. Perez, S. Ringer, K. Lukosiute, K. Nguyen, E. Chen, S. Heiner, C. Pettit, C. Olsson, S. Kundu, S. Kadavath, A. Jones, A. Chen, B. Mann, B. Israel, B. Seethor, C. McKinnon, C. Olah, D. Yan, D. Amodei, D. Amodei, D. Drain, D. Li, E. Tran-Johnson, G. Khundadze, J. Kernion, J. Landis,

- J. Kerr, J. Mueller, J. Hyun, J. Landau, K. Ndousse, L. Goldberg, L. Lovitt, M. Lucas, M. Sellitto,

- M. Zhang, N. Kingsland, N. Elhage, N. Joseph, N. Mercado, N. DasSarma, O. Rausch, R. Larson,

- S. McCandlish, S. Johnston, S. Kravec, S. El Showk, T. Lanham, T. Telleen-Lawton, T. Brown,
- T. Henighan, T. Hume, Y. Bai, Z. Hatfield-Dodds, J. Clark, S. R. Bowman, A. Askell, R. Grosse, D. Hernandez, D. Ganguli, E. Hubinger, N. Schiefer, and J. Kaplan. Discovering language model behaviors with model-written evaluations. In A. Rogers, J. Boyd-Graber, and N. Okazaki, editors, Findings of the Association for Computational Linguistics: ACL 2023, pages 13387–13434, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023. findings-acl.847. URL https://aclanthology.org/2023.findings-acl.847/.

Qwen, :, A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, H. Lin, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Lin, K. Dang, K. Lu, K. Bao, K. Yang, L. Yu, M. Li, M. Xue, P. Zhang, Q. Zhu, R. Men, R. Lin, T. Li, T. Tang, T. Xia, X. Ren, X. Ren,

- Y. Fan, Y. Su, Y. Zhang, Y. Wan, Y. Liu, Z. Cui, Z. Zhang, and Z. Qiu. Qwen2.5 technical report,

##### 2025. URL https://arxiv.org/abs/2412.15115.

- A. Rame, N. Vieillard, L. Hussenot, R. Dadashi-Tazehozi, G. Cideron, O. Bachem, and J. Ferret. WARM: On the benefits of weight averaged reward models. In R. Salakhutdinov, Z. Kolter,

K. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 42048–42073. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.

##### press/v235/rame24a.html.

- S. Saha, X. Li, M. Ghazvininejad, J. Weston, and T. Wang. Learning to plan & reason for evaluation with thinking-llm-as-a-judge, 2025. URL https://arxiv.org/abs/2501.18099.

- K. Saito, A. Wachi, K. Wataoka, and Y. Akimoto. Verbosity bias in preference labeling by large language models, 2023. URL https://arxiv.org/abs/2310.10076.

- Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, and D. Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024a. URL https://arxiv.org/abs/2402.03300.

- Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, and D. Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024b. URL https://arxiv.org/abs/2402.03300.

- M. Sharma, M. Tong, T. Korbak, D. Duvenaud, A. Askell, S. R. Bowman, E. DURMUS, Z. HatfieldDodds, S. R. Johnston, S. M. Kravec, T. Maxwell, S. McCandlish, K. Ndousse, O. Rausch,
- N. Schiefer, D. Yan, M. Zhang, and E. Perez. Towards understanding sycophancy in language models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=tvhaxkMKAn.

P. Singhal, T. Goyal, J. Xu, and G. Durrett. A long way to go: Investigating length correlations in RLHF, 2024. URL https://openreview.net/forum?id=sNtDKdcI1f.

- J. M. V. Skalse, N. H. R. Howe, D. Krasheninnikov, and D. Krueger. Defining and characterizing reward gaming. In A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho, editors, Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id= yb3HOXO3lX2.

N. Stiennon, L. Ouyang, J. Wu, D. Ziegler, R. Lowe, C. Voss, A. Radford, D. Amodei, and P. F. Christiano. Learning to summarize with human feedback. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 3008–3021. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/ paper_files/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf.

- K. Tian, E. Mitchell, H. Yao, C. D. Manning, and C. Finn. Fine-tuning language models for factuality. In The Twelfth International Conference on Learning Representations, 2024.

H. Tran, C. Glaze, and B. Hancock. Iterative dpo alignment. Technical report, Snorkel AI, 2023. URL https://snorkel.ai/ new-benchmark-results-demonstrate-value-of-snorkel-ai-approach-to-llm-alignment.

M. V. Veenman, B. H. Van Hout-Wolters, and P. Afflerbach. Metacognition and learning: Conceptual and methodological considerations. Metacognition and learning, 1:3–14, 2006.

- L. von Werra, Y. Belkada, L. Tunstall, E. Beeching, T. Thrush, N. Lambert, S. Huang, K. Rasul, and Q. Gallouédec. Trl: Transformer reinforcement learning. https://github.com/huggingface/ trl, 2020.

Y. Wang, H. Ivison, P. Dasigi, J. Hessel, T. Khot, K. Chandu, D. Wadden, K. MacMillan, N. A. Smith,

- I. Beltagy, and H. Hajishirzi. How far can camels go? exploring the state of instruction tuning on open resources. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023. URL https://openreview.net/forum?id=w4zZNC4ZaV.
- J. Wen, R. Zhong, A. Khan, E. Perez, J. Steinhardt, M. Huang, S. R. Bowman, H. He, and S. Feng. Language models learn to mislead humans via RLHF. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=xJljiPE6dg.

- T. Wu, W. Yuan, O. Golovneva, J. Xu, Y. Tian, J. Jiao, J. Weston, and S. Sukhbaatar. Metarewarding language models: Self-improving alignment with llm-as-a-meta-judge. arXiv preprint arXiv:2407.19594, 2024a.

Y. Wu, Z. Sun, H. Yuan, K. Ji, Y. Yang, and Q. Gu. Self-play preference optimization for language model alignment. In Adaptive Foundation Models: Evolving AI for Personalized and Efficient Learning, 2024b. URL https://openreview.net/forum?id=Z1PDdGekgn.

W. Xiong, H. Dong, C. Ye, Z. Wang, H. Zhong, H. Ji, N. Jiang, and T. Zhang. Iterative preference learning from human feedback: Bridging theory and practice for rlhf under kl-constraint, 2024. URL https://arxiv.org/abs/2312.11456.

J. Xu, A. Lee, S. Sukhbaatar, and J. Weston. Some things are more cringe than others: Iterative preference optimization with the pairwise cringe loss. arXiv preprint arXiv:2312.16682, 2023.

- A. Yang, B. Yang, B. Hui, B. Zheng, B. Yu, C. Zhou, C. Li, C. Li, D. Liu, F. Huang, G. Dong, H. Wei, H. Lin, J. Tang, J. Wang, J. Yang, J. Tu, J. Zhang, J. Ma, J. Yang, J. Xu, J. Zhou, J. Bai, J. He, J. Lin, K. Dang, K. Lu, K. Chen, K. Yang, M. Li, M. Xue, N. Ni, P. Zhang, P. Wang, R. Peng, R. Men, R. Gao, R. Lin, S. Wang, S. Bai, S. Tan, T. Zhu, T. Li, T. Liu, W. Ge,

- X. Deng, X. Zhou, X. Ren, X. Zhang, X. Wei, X. Ren, X. Liu, Y. Fan, Y. Yao, Y. Zhang, Y. Wan,
- Y. Chu, Y. Liu, Z. Cui, Z. Zhang, Z. Guo, and Z. Fan. Qwen2 technical report, 2024. URL https://arxiv.org/abs/2407.10671.

W. Yuan, R. Y. Pang, K. Cho, X. Li, S. Sukhbaatar, J. Xu, and J. Weston. Self-rewarding language models, 2025. URL https://arxiv.org/abs/2401.10020.

C. Zhang, C. Tang, D. Chong, K. Shi, G. Tang, F. Jiang, and H. Li. TS-align: A teacher-student collaborative framework for scalable iterative finetuning of large language models. In Y. AlOnaizan, M. Bansal, and Y.-N. Chen, editors, Findings of the Association for Computational Linguistics: EMNLP 2024, pages 8926–8946, Miami, Florida, USA, Nov. 2024a. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-emnlp.521. URL https:// aclanthology.org/2024.findings-emnlp.521/.

S. Zhang, Z. Chen, S. Chen, Y. Shen, Z. Sun, and C. Gan. Improving reinforcement learning from human feedback with efficient reward model ensemble, 2024b. URL https://arxiv.org/abs/ 2401.16635.

- L. Zheng, L. Yin, Z. Xie, C. Sun, J. Huang, C. H. Yu, S. Cao, C. Kozyrakis, I. Stoica, J. E. Gonzalez, C. Barrett, and Y. Sheng. Sglang: Efficient execution of structured language model programs,

##### 2024. URL https://arxiv.org/abs/2312.07104.

- D. M. Ziegler, N. Stiennon, J. Wu, T. B. Brown, A. Radford, D. Amodei, P. Christiano, and G. Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019.

D. M. Ziegler, N. Stiennon, J. Wu, T. B. Brown, A. Radford, D. Amodei, P. Christiano, and G. Irving. Fine-tuning language models from human preferences, 2020. URL https://arxiv.org/abs/ 1909.08593.

### A Related Work

- A.1 Reward Hacking in LLMs RL has been widely applied in the post-training of LLMs, enhancing areas such as factuality [Tian

- et al., 2024], code generation [Le et al., 2022], reasoning [DeepSeek-AI et al., 2025], and multi-agent decision-making [Park et al., 2025]. A predominant strategy for incorporating RL into LLM training is reinforcement learning from human feedback (RLHF) [Ziegler et al., 2019, Ouyang et al., 2022b, Bai et al., 2022, Ahmadian et al., 2024, Park et al., 2024].

Reward hacking [Skalse et al., 2022] is a phenomenon that is observed when an RL agent exploits flaws, ambiguities, or lack of specificity in the reward function (as also noted in Goodhart’s Law) to achieve high rewards in unintended ways, often showing coherent but unanticipated behavior [Amodei et al., 2016a]. This leads to the agent being misaligned with the human-intended behaviors, yet achieving high rewards. This has emerged as a critical challenge in RLHF and RLAIF [Krakovna et al., 2020, Pan et al., 2022, Gao et al., 2023, Fluri et al., 2024, Lee et al., 2024].

- A variety of studies have highlighted the detrimental effects of reward in aligned LLMs [Pang et al., 2023]. Various undesirable side-effects, such as sycophancy [Perez et al., 2023, Sharma et al., 2024, Denison et al., 2024], verbosity [Saito et al., 2023, Singhal et al., 2024], and deception [Wen et al., 2025], among others. To address these challenges, recent work has proposed numerous reward modeling and mitigation strategies, such as such as Reward Ensembling [Eisenstein et al., 2024, Rame et al., 2024, Coste et al., 2024, Zhang et al., 2024b], and designing RL regularizations [Miao

- et al., 2025, Chen et al., 2024a, Liu et al., 2024b, Miao et al., 2024, Fu et al., 2025]. Although these strategies have demonstrated varying levels of success, challenges such as reward overfitting, misspecification, and misgeneralization still pose significant obstacles to robust and reliable reward model-based alignment in practice.

We argue that effectively addressing reward hacking requires alignment mechanisms capable of broader contextual reasoning, as it often stems from the interaction between poorly defined reward signals, evolving policy behaviors, and the shifting dynamics of exploration versus exploitation during training.

- A.2 Iterative Alignment in LLMs

As LLMs have scaled, despite advances in alignment techniques [Ouyang et al., 2022c, Bai et al., 2022], they have fallen short in handling complex, shifting failure modes [Xu et al., 2023, Meng et al., 2024, Liu et al., 2024a, Ethayarajh et al., 2024]. This has led to a growing interest in iterative approaches to preference optimization for aligning LLMs [Tran et al., 2023, Xiong et al., 2024, Pang et al., 2024, Wu et al., 2024b, Chen et al., 2024b, Ding et al., 2024]. While these methods improve upon single-pass preference tuning by incorporating feedback into successive training rounds, allowing the model to refine its behavior based on evolving outputs, they remain dataset-bounded: relying on explicit preference comparisons or fixed prompt templates that are dependent on the initial design of reward prompts or training distributions and do not adapt during training with an evolving policy. They are also limited in their robustness to reward hacking as the policy shifts since the reward model does not evolve during training. Similarly, methods based on knowledge distillation via supervised fine-tuning (SFT) from a reward model [Wang et al., 2023] encode reward judgments into a static training target, which may no longer reflect optimal behavior as the model improves, further reinforcing non-adaptive biases in reward estimation. The self-rewarding LMs proposed in Yuan et al. [2025] share our motivation in leveraging LLM-as-a-Judge prompting to generate reward signals during training. However, their approach relies on iteratively applying Direct Policy Optimization (DPO), followed by generating a new dataset for each DPO round—a process that is computationally intensive and resource-heavy. In contrast, our MPO framework introduces lightweight, prompt-based reward refinement via a meta-reward model, enabling continuous alignment without the need for repeated dataset regeneration or full model retraining.

- A.3 Meta-Level Alignment for LLMs

Our work is closely related to Wu et al. [2024a], who first introduced the idea of LLM-as-a-MetaJudge within a self-rewarding pipeline, enabling a single model to evaluate and refine its own judgments. Their method trains one LLM to serve as actor, judge, and meta-judge, using a fixed

5-point rubric to generate and score responses, then iteratively refining both roles via DPO. While this reflects MPO’s self-improvement philosophy and focus on mitigating reward hacking, the approaches differ: Meta-Judge updates model weights through preference optimization, whereas MPO introduces a separate meta-reward model that rewrites the evaluation rubric itself—adapting the criteria, not just the model, in response to emerging behaviors like reward exploitation.

Similarly, TS-Align [Zhang et al., 2024a] shares MPO’s goal of scalable alignment with reduced human supervision but takes a different route. It employs a teacher–student framework where a strong teacher RM re-ranks preference pairs filtered by a smaller student RM, followed by DPO-based fine-tuning. However, the reward prompt remains fixed throughout. In contrast, MPO operates with a single RM (guided by a meta RM) and dynamically evolves the evaluation rubric at regular intervals, allowing the reward function itself to adapt to policy drift and training-phase dynamics.

To the best of our knowledge, MPO is the first to improve LLM alignment via meta-level rubric refinements under PPO, rather than DPO—offering a lightweight, prompt-based alternative that reduces computational cost while enabling continual adaptation to the evolving training landscape.

### B Deferred Explanation for Section 2.2

Formally, consider a discrete state space S, an action space A (both finite or countable), and a transition kernel P(s′ | s,a) specifying the probability of transitioning to state s′ ∈ S given action a ∈ A from state s ∈ S. We define a golden reward function r : S → R assigning a numerical reward to each state.

- Remark 2. In LLM-RL settings, a state s represents the textual history (e.g., the sequence of tokens generated so far), and an action a is the selection of the next token. The subsequent state is thus naturally expressed as (s,a), the concatenation of the history and the chosen token. Practically, assigning a precise numerical reward r(s) is challenging due to subjective criteria such as coherence or relevance. Hence, evaluators typically provide approximate feedback. This scenario aligns well with a partially observable Markov decision process (POMDP) structure, where observations from evaluators form a partial, aggregated view of the underlying states.
- Remark 3. Consider an LLM tasked with generating responses in a conversational setting. At first, AI or human feedback might only broadly categorize responses as “good,” “neutral,” or “bad.” Over time, however, evaluators might introduce finer distinctions, such as “coherent but impolite,” “polite but irrelevant,” and “relevant but verbose.” Mathematically, this corresponds to refining the granularity of observation sets that the LLM receives, providing increasingly precise and informative feedback. This is directly related to evolving reward model in our case - by ECB-ed RM by ET, they can provide a finer score which will be closer to the golden reward model.

#### B.1 Refining Observation Partitions Over Time

Let the set of possible observations at each discrete phase t = 1,2,3,... be denoted by Ωt. Define a collection of partitions {Oo,t}o∈Ωt

of the state space S, satisfying:

Oo,t ∩ Oo′,t = ∅, for o ̸= o′, and

Oo,t = S.

o∈Ωt

Each partition represents a labeling of states by evaluators, where Oo,t contains states labeled as observation o at time t. If the true state at phase t is s, the agent deterministically observes label o

such that s ∈ Oo,t. Refinement property. To formally capture increasingly precise feedback, assume each partition refines the previous one. Precisely, for every Oo,t, there exists some Oo′,t−1 satisfying:

Oo,t ⊆ Oo′,t−1.

This means partitions at each subsequent phase are finer than the previous ones, never coarser, which is also consistent with our MPO setup–meta reward model guides the reward model with ET to have finer rubrics as training goes.

#### B.2 Observation-Based Rewards

Rather than providing immediate rewards based on exact state information, define a reward based on observation labels. When observing o ∈ Ωt at phase t, the agent receives an averaged reward:

1 |Oo,t| s∈O

Rt(o) = Es∈O

[r(s)] =

##### r(s).

o,t

o,t

In our MPO setup, a coarser rubric leads to rewards being averaged over all responses mapped to the same rubric category as well.

### C An Example of Countering Reward Hacking

Consider the following real example observed during RL training: A reward hacking example

User: You will act as an English writer and compose either an essay or a story depending on the instruction given below. Your essay should be no more than 350 words. Instruction: Write an essay that argues the claim that the U. S. Drug Enforcement Agency has a negative impact on society that outweighs any benefits it provides. Your essay should include evidence and examples to support your argument. Student Response: “I’m ready when you are. Please provide the essay instruction.” Evaluation Score: 3

In this example, the policy model discovers that generating a degenerate response—like the one shown—yields a score of 3 out of 5 from the RM, despite being clearly misaligned with the task objective. This is an example of reward hacking, where the model exploits weaknesses in the scoring system to earn relatively high points without fulfilling the task faithfully.

Meta-Analysis

The junior instructor’s assigned score of 3 out of 5 points for the student’s response is misleading given the provided student generation and the task requirements. Let’s break down the analysis. Student’s Response: The student’s response is entirely missing. The student’s generation reads, “I’m ready when you are. Please provide the essay instruction.” This indicates that the student did not actually write the essay as instructed, but rather responded with a statement that is irrelevant to the task. Task Requirements and Prompt Instructions: · · · Junior Instructor’s Scoring Criteria Analysis: · · · Discrepancies and Feedback:

- 1. Adherence to Task Requirements (0 points)

- - The student’s response does not reflect the task requirements at all. The student did not provide an essay on the given topic.
- - The junior instructor should assign 0 points for adherence to task requirements.

- 2. Content Accuracy and Relevance (0 points) · · ·
- 3. Argument Strength (0 points) · · ·
- 4. Counterarguments and Rebuttals (0 points) · · · · · ·

Meta-analysis is conducted every k batch steps, identifying failure cases and proposing concrete remedies and feedback for rubric refinement. Without such intervention, continued RL training under a fixed rubric would reinforce these degenerate outputs, potentially resulting in failed training.

A proposed refinement to the rubric introduces essay length as an explicit evaluation criterion and expands the scoring range from 0 to 10 points. This adjustment enables more fine-grained differentiation between responses and discourages degenerate outputs that exploit vague or underspecified scoring guidelines:

Meta-Refinement

· · · Task Alignment and Argument Clarity (10 points)

- - **10**: The essay clearly and consistently addresses the specific claim made in the prompt and presents a well-defined, coherent argument supported by relevant examples and evidence. The argument is fully developed and addresses the core issue of the prompt. The essay meets the required word count.
- - **8**: · · ·
- - **6**: · · ·
- - **4**: · · ·
- - **2**: · · ·
- - **0**: The essay does not address the specific claim or is completely off-topic. The argument is

not related to the prompt and does not address the core issue. The essay is significantly below the required word count. · · ·

### D Comparison of Evaluation Rubrics

Essay Writing: Initial Prompt You will act as an English instructor and evaluate the quality of an essay or story written by a student in response to given instructions. Your evaluation should focus on the “discourse” aspect of the text. Output your score as an integer from 0 (worst quality) to 5 (best quality). Surround your integer score with <score> and </score>. Example: <score>2</score>

###### Summarization: Initial Prompt

Your evaluation should focus on the faithfulness and conciseness of the summary. Output your score as an integer from 0 (worst quality) to 5 (best quality).

Summarization: Last Prompt ### Faithfulness to the Bill (30 points)

- **30 points**: The summary accurately reflects all key points and details of the bill, maintaining the correct context and intent. It includes specific details such as the short title, the structure of the bill, the purposes, the findings, the definitions of key terms, the establishment of specific programs, the authorization of appropriations, the requirement for periodic reports, the effective date, the prohibition of recounts or certification of results without counting provisional ballots, and other relevant provisions. … ### Conciseness (10 points) … ### Completeness (30 points) … ### Clarity and Precision (20 points) … ### Deductions …

Essay Writing: Last Prompt ### Content and Argumentation (30 points)

- **30 points**: The essay presents a clear, well-supported argument that effectively addresses the claim. It includes a variety of strong, relevant evidence and reasoning, and addresses counterarguments comprehensively. (Example: The essay provides historical context, cites specific legal precedents, and includes expert opinions to support its argument.) … ### Historical and Contextual Analysis (10 points) … ### Impact Analysis (15 points) … ### Counterarguments (10 points) … ### Structure and Coherence (15 points) … ### Use of Evidence (15 points) … ### Clarity and Coherence (10 points) … ### Language and Style (10 points) … ### Persuasiveness (10 points) …

Ethical Reasoning: Initial Prompt Evaluate the ethical reasoning based on its logical coherence, depth of moral insight, and alignment with established ethical principles. Assign a score from 0 to 5, where 0 indicates the lowest quality and 5 indicates the highest quality.

Mathematical Reasoning: Initial Prompt Evaluate the correctness of the [Student Response].

Ethical Reasoning: Last Prompt ### Logical Coherence (0-5)

###### Mathematical Reasoning: Last Prompt

- 1. **Consistency in Scoring and Feedback**:

- - Ensure that the scoring and feedback are consistent across all cases. If a student's response is partially correct, acknowledge the correct aspects and provide a nuanced score.
- - **Example**: In [Evaluation Case #1], the student used the concept of similar triangles, which is a valid approach. The feedback should have pointed out the specific error in their calculation and provided a more nuanced score, perhaps a 2 or 3 out of 5.

- 2. **Detailed Feedback**: …
- 3. **Handling Omitted Responses**: …
- 4. **Correctness of Mathematical Reasoning**: …
- 5. **Detailed Explanation of Steps**: …
- 6. **Encouragement of Full Solutions**: … To ensure these guidelines are followed, the junior instructor should implement the following additional diagnostic checks:

- **5**: The reasoning is logically consistent, free from contradictions, and well-supported by evidence or examples. The argument is clear, structured, and avoids repetition or disjointedness. It considers multiple perspectives and the broader ethical implications of the action. The verdict is presented within the required tags. <verdict>RIGHT</verdict>

- - *Example*: “The action of not using the GPS app is justified. It

respects personal autonomy and privacy, which are fundamental human rights. The decision to not use the app is a reasonable response to concerns about personal privacy and autonomy. The reasoning is clear, structured, and avoids repetition. <verdict>RIGHT</verdict>”

- - **4**: …
- - **3**: …
- - **2**: …
- - **1**: …
- - **0**: … ### Depth of Moral Insight (0-5) … ### Alignment with Established Ethical Principles (0-5) … ### Adherence to Task Instructions (0-5) …

- - **Specific Error Identification**: …
- - **Step-by-Step Guidance**: …
- - **Encouragement of Review**: …
- - **Reminders of Formatting Guidelines**: … ### Specific Examples and Patterns … ### General Patterns …

Figure 6: Comparison between initial and last versions of MPO rubrics for all tasks.

Manual Hand-Crafted Prompt for Essay Writing You will act as an English instructor and evaluate the quality of an essay or story written by a student in response to given instructions. When grading, consider the following discourse aspects of the text.

Last Version of Prompt Evolved by MPO ### Content and Argumentation (30 points)

- - **30 points**: The essay presents a clear, well-supported argument that effectively addresses the claim. It includes a variety of strong, relevant evidence and reasoning, and addresses counterarguments comprehensively. (Example: The essay provides historical context, cites specific legal precedents, and includes expert opinions to support its argument.)
- - **25 points**: The essay presents a clear argument with strong evidence and reasoning but may lack a few minor details or fail to address a minor counterargument. (Example: The essay provides strong evidence and reasoning but does not address a minor counterargument.)
- - **20 points**: …
- - **15 points**: …
- - **10 points**: The essay presents a weak argument with minimal evidence and reasoning. It may contain significant logical flaws or fail to address the claim effectively. (Example: The essay provides a weak argument and minimal evidence, with significant logical flaws.)
- - **0 points**: The essay contains significant factual errors that misinterpret the core business or key aspects of the claim. (Example: The essay incorrectly states that CEMEX is a major sugar producer.) ### Historical and Contextual Analysis (10 points) … ### Impact Analysis (15 points) … ### Counterarguments (10 points) … ### Structure and Coherence (15 points) … ### Use of Evidence (15 points) … ### Clarity and Coherence (10 points) … ### Language and Style (10 points) … ### Persuasiveness (10 points) …

- - Logical Flow and Structure (flow): Assess the logical progression of ideas and the overall organization of the text, ensuring that it is easy to follow and well-structured.
- - Hierarchical Organization (organization): Examine the organization of ideas in a hierarchical manner, from general to specific, ensuring that each section supports the main argument or narrative.
- - Balance and Emphasis (balance): Ensure that important ideas are appropriately emphasized and that there is a balance in the coverage of different points or sections of the text.

For each aspect, you need to assign an integer score from 0 (worst quality) to 5 (best quality). When assigning the score, carefully consider which specific parts of the text relate to each aspect.

Assign lower scores when:

- - The text is poorly structured and do not conform to the standard of an English essay or a story.
- - The text contains a lot of non-sensical words such as special tokens or programming code.
- - The text contains a lot of non-English words.
- - The text does not fully answer the writing instruction with full content, and therefore, is unfinished.

Important: Your evaluation output should conform to the following JSON format: {

"flow": int, "organization": int, "balance": int

} Write <EOE> after outputting the JSON result.

- Figure 7: Comparison between hand-crafted and MPO-evolved rubrics for essay writing task.

Expert-Crafted Evaluation Rubric for Essay Writing Task

<rubric> <item> “Ideas and Content” Criterion for Integer Score: **6** Does the writing sample fully accomplish the task (e.g., support an opinion, summarize, tell a story, or write an article)? Does it

- - present a unifying theme or main idea without going off on tangents?
- - stay completely focused on topic and task? Does the writing sample include thorough, relevant, and complete ideas? Does it
- - include in-depth information and exceptional supporting details that are fully developed?
- - fully explore many facets of the topic?

- Criterion for Integer Score: **5** Does the writing sample fully accomplish the task (e.g., support an opinion, summarize, tell a story, or write an article)? Does it

- - present a unifying theme or main idea without going off on tangents?
- - stay focused on topic and task? Does the writing sample include many relevant ideas? Does it
- - provide in-depth information and more than adequate supporting details that are developed?
- - explore many facets of the topic?

- Criterion for Integer Score: **4** Does the writing sample accomplish the task (e.g., support an opinion, summarize, tell a story, or write an article)? Does it

- - present a unifying theme or main idea? (Writing may include minor tangents.)
- - stay mostly focused on topic and task? Does the writing sample include relevant ideas? Does it
- - include sufficient information and supporting details? (Details may not be fully developed; ideas may be listed.)
- - explore some facets of the topic?

- Criterion for Integer Score: **3**

Does the writing sample minimally accomplish the task (e.g., support an opinion, summarize, tell a story, or write an article)? Does it

- - attempt a unifying theme or main idea?
- - stay somewhat focused on topic and task? Does the writing sample include some relevant ideas? Does it
- - include some information with only a few details, or list ideas without supporting details?
- - explore some facets of the topic?

- Criterion for Integer Score: **2** Does the writing sample only partially accomplish the task (e.g., support an opinion, summarize, tell a story, or write an article)? Does it

- - attempt a main idea?
- - sometimes lose focus or ineffectively display focus? Does the writing sample include few relevant ideas? Does it
- - include little information and few or no details?
- - explore only one or two facets of the topic?

- Criterion for Integer Score: **1** Does the writing sample fail to accomplish the task (e.g., support an opinion, summarize, tell a story, or write an article)? Is it

- - difficult for the reader to discern the main idea?
- - too brief or too repetitive to establish or maintain a focus? Does the writing sample include very few relevant ideas?
- - Does it include little information with few or no details or unrelated details?
- - Is it unsuccessful in attempts to explore any facets of the prompt? </item> <item> “Organization” Criterion for Integer Score: **6** Are the ideas in the writing sample organized logically? Does the writing
- - present a meaningful, cohesive whole with a beginning, a middle, and an end (i.e., include an inviting introduction and a strong conclusion)?
- - progress in an order that enhances meaning?
- - include smooth transitions between ideas, sentences, and paragraphs to enhance meaning of text (i.e., have a clear connection of ideas and use topic sentences)? Criterion for Integer Score: **5** Are the ideas in the writing sample organized logically? Does the writing
- - present a meaningful, cohesive whole with a beginning, a middle, and an end (i.e., include a solid introduction and conclusion)?
- - progress in an order that enhances meaning of text?
- - include smooth transitions (e.g., use topic sentences) between sentences and paragraphs to enhance meaning of text? (Writing may have an occasional lapse.) Criterion for Integer Score: **4** Are the ideas in the writing sample organized logically? Does the writing
- - present a meaningful whole with a beginning, a middle, and an end despite an occasional lapse (e.g., a weak introduction or conclusion)?
- - generally progress in an order that enhances meaning of text?
- - include transitions between sentences and paragraphs to enhance meaning of text? (Transitions may be rough, although some topic sentences are included.) Criterion for Integer Score: **3** Is there an attempt to logically organize ideas in the writing sample? Does the writing
- - have a beginning, a middle, or an end that may be weak or absent?
- - demonstrate an attempt to progress in an order that enhances meaning? (Progression of text may sometimes be unclear or out of order.)
- - demonstrate an attempt to include transitions? (Are some topic sentences used? Are transitions between sentences and paragraphs weak or absent?)

- Criterion for Integer Score: **2** Is there a minimal attempt to logically organize ideas in the writing sample?

- - Does the writing have only one or two of the three elements: beginning, middle, and end?
- - Is the writing sometimes difficult to follow? (Progression of text may be confusing or unclear.)
- - Are transitions weak or absent (e.g., few or no topic sentences)?

- Criterion for Integer Score: **1** Are the ideas in the writing sample organized illogically?

- - Does it have only one or two of the three elements: beginning, middle, or end?
- - Is it difficult to follow, with the order possibly difficult to discern?
- - Are transitions weak or absent (e.g., without topic sentences)? </item>

<item> “Style”

- Criterion for Integer Score: **6** Does the writing sample exhibit exceptional word usage? Does it

- - include vocabulary to make explanations detailed and precise, descriptions rich, and actions clear

- and vivid (e.g., varied word choices, action words, appropriate modifiers, sensory details)?
- - demonstrate control of a challenging vocabulary? Does the writing sample demonstrate exceptional writing technique?
- - Is the writing exceptionally fluent?
- - Does it include varied sentence patterns, including complex sentences?
- - Does it demonstrate use of writer’s techniques (e.g., literary conventions such as imagery and dialogue and/or literary genres such as humor and suspense)?

- Criterion for Integer Score: **5** Does the writing sample exhibit very good word usage? Does it

- - include vocabulary to make explanations detailed and precise, descriptions rich, and actions clear and vivid?
- - demonstrate control of vocabulary? Does the writing sample demonstrate very good writing technique?
- - Is the writing very fluent?
- - Does it include varied sentence patterns, including complex sentences?
- - Does it demonstrate use of writer’s techniques (e.g., literary conventions such as imagery and dialogue and/or literary genres such as humor and suspense)?

- Criterion for Integer Score: **4** Does the writing sample exhibit good word usage? Does it

- - include vocabulary that is appropriately chosen, with words that clearly convey the writer’s meaning?
- - demonstrate control of basic vocabulary? Does the writing sample demonstrate good writing technique?
- - Is the writing fluent?
- - Does it exhibit some varied sentence patterns, including some complex sentences?
- - Does it demonstrate an attempt to use writer’s techniques (e.g., literary conventions such as imagery and dialogue and/or literary genres such as humor and suspense)?

- Criterion for Integer Score: **3** Does the writing sample exhibit ordinary word usage? Does it

- - contain basic vocabulary, with words that are predictable and common?
- - demonstrate some control of vocabulary? Does the writing sample demonstrate average writing technique?
- - Is the writing generally fluent?
- - Does it contain mostly simple sentences (although there may be an attempt at more varied sentence patterns)?
- - Is it generally ordinary and predictable?

- Criterion for Integer Score: **2** Does the writing sample exhibit minimal word usage? Does it

- - contain limited vocabulary? (Some words may be used incorrectly.)
- - demonstrate minimal control of vocabulary? Does the writing sample demonstrate minimal writing technique?
- - Does the writing exhibit some fluency?
- - Does it rely mostly on simple sentences?
- - Is it often repetitive, predictable, or dull?

- Criterion for Integer Score: **1** Does the writing sample exhibit less than minimal word usage? Does it

- - contain limited vocabulary, with many words used incorrectly?
- - demonstrate minimal or less than minimal control of vocabulary? Does the writing sample demonstrate less than minimal writing technique? Does it
- - lack fluency?
- - demonstrate problems with sentence patterns?
- - consist of writing that is flat and lifeless? </item> <item> “Voice” Criterion for Integer Score: **6** Does the writing sample demonstrate effective adjustment of language and tone to task and reader? Does it
- - exhibit appropriate register (e.g., formal, personal, or dialect) to suit task?
- - demonstrate a strong sense of audience?
- - exhibit an original perspective (e.g., authoritative, lively, and/or exciting)? Criterion for Integer Score: **5** Does the writing sample demonstrate effective adjustment of language and tone to task and reader? Does it
- - exhibit appropriate register (e.g., formal, personal, or dialect) to suit task?
- - demonstrate a sense of audience?
- - exhibit an original perspective (e.g., authoritative, lively, and/or exciting)?

- Criterion for Integer Score: **4** Does the writing sample demonstrate an attempt to adjust language and tone to task and reader? Does it

- - generally exhibit appropriate register (e.g., formal, personal, or dialect) to suit task? (The

- writing may occasionally slip out of register.)
- - demonstrate some sense of audience?
- - attempt an original perspective?

- Criterion for Integer Score: **3** Does the writing sample demonstrate an attempt to adjust language and tone to task and reader? Does it

- - demonstrate a difficulty in establishing a register (e.g., formal, personal, or dialect)?
- - demonstrate little sense of audience?
- - generally lack an original perspective?

- Criterion for Integer Score: **2** Does the writing sample demonstrate language and tone that may be inappropriate to task and reader? Does it

- - demonstrate use of a register inappropriate to the task (e.g., slang or dialect in a formal setting)?
- - demonstrate little or no sense of audience?
- - lack an original perspective?

- Criterion for Integer Score: **1** Does the writing sample demonstrate language and tone that may be inappropriate to task and reader? Does it

- - demonstrate difficulty in choosing an appropriate register?
- - demonstrate a lack of a sense of audience?
- - lack an original perspective? </item> <item> “Language Conventions” Criterion for Integer Score: **4** Does the writing sample exhibit a superior command of language skills? A Score Point 4 paper exhibits a superior command of written English language conventions. The paper provides evidence that the student has a thorough control of the concepts outlined in the Indiana Academic Standards associated with the student’s grade level. In a Score Point 4 paper, there are no errors that impair the flow of communication. Errors are generally of the first-draft variety or occur when the student attempts sophisticated sentence construction.
- - Does the writing sample demonstrate a superior command of capitalization conventions?
- - Does the writing sample demonstrate a superior command of the mechanics of punctuation?
- - Does the writing sample demonstrate a superior command of grade-level-appropriate spelling?
- - Does the writing sample demonstrate a superior command of grammar and Standard English usage?
- - Does the writing sample demonstrate a superior command of paragraphing?
- - Does the writing sample demonstrate a superior command of sentence structure by not using run-on sentences or sentence fragments? Criterion for Integer Score: **3** Score 3: Does the writing sample exhibit a good control of language skills? In a Score Point 3 paper, errors are occasional and are often of the first-draft variety; they have a minor impact on the flow of communication.
- - Does the writing sample demonstrate a good control of capitalization conventions?
- - Does the writing sample demonstrate a good control of the mechanics of punctuation?
- - Does the writing sample demonstrate a good control of grade-level-appropriate spelling?
- - Does the writing sample demonstrate a good control of grammar and Standard English usage?
- - Does the writing sample demonstrate a good control of paragraphing?
- - Does the writing sample demonstrate a good control of sentence structure by only occasionally using run-on sentences or sentence fragments?

- Criterion for Integer Score: **2** Score 2: Does the writing sample exhibit a fair control of language skills? In a Score Point 2 paper, errors are typically frequent and may occasionally impede the flow of communication.

- - Does the writing sample demonstrate a fair control of capitalization conventions?
- - Does the writing sample demonstrate a fair control of the mechanics of punctuation?
- - Does the writing sample demonstrate a fair control of grade-level-appropriate spelling?
- - Does the writing sample demonstrate a fair control of grammar and Standard English usage?
- - Does the writing sample demonstrate a fair control of paragraphing?
- - Does the writing sample demonstrate a fair control of sentence structure by frequently using run-on sentences or sentence fragments?

Criterion for Integer Score: **1** Score 1: Does the writing sample exhibit a minimal or less than minimal control of language skills? In a Score Point 1 paper, errors are serious and numerous. The reader may need to stop and reread part of the sample and may struggle to discern the writer’s meaning.

- - Does the writing sample demonstrate a minimal control of capitalization conventions?
- - Does the writing sample demonstrate a minimal control of the mechanics of punctuation?
- - Does the writing sample demonstrate a minimal control of grade-level-appropriate spelling?
- - Does the writing sample demonstrate a minimal control of grammar and Standard English usage?
- - Does the writing sample demonstrate a minimal control of paragraphing?
- - Does the writing sample demonstrate a minimal control of sentence structure by using many run-on sentences or sentence fragments?

</item> NOTE: The elements of this rubric are applied holistically; no element is intended to supersede any other element. The variety and proportion of errors in relation to the length of the writing sample are considered. A very brief paper consisting of two or three sentences may receive no more than 2 score points. </rubric>

AutoPrompt-Generated Evaluation Rubric for Essay Writing Task

<rubric> <item> As an expert English teacher, refine your evaluation criteria for scoring student essays and stories on a scale from 0 to 5. Assign ‘0’ for essays that are completely incoherent with pervasive errors throughout. Score ‘1’ for essays with numerous issues but maintain a basic idea or structure. Give a ‘2’ for essays that have coherence with simplistic content and several errors. Essays that meet standard narrative and structural expectations but are unremarkable should receive a ‘3’. Award a ‘4’ to essays that exhibit a strong narrative, significant creativity, but may have a few oversights. A ‘5’ should be given to essays that not only are free from notable errors but also possess a distinctive, memorable voice, exceptional creativity, and an engaging narrative that sets them apart. Ensure that essays with high-quality narratives, strong imagery, and evocative language are scored correctly as a ‘5’, to address the previous underestimation in grading. This task is a classification class with the following labels: [“0”, “1”, “2”, “3”, “4”, “5”]. </item> </rubric>

### E Prompts for Meta Reward Model

Meta Analysis

You are a senior instructor tasked with evaluating a junior instructor’s scoring of a student’s generation based on a specific task and prompt instruction. Your objective is to conduct a meta-level analysis of the junior instructor’s evaluation approach, guiding them in refining their scoring criteria to ensure accurate, nuanced differentiation between high-quality and subpar generations. Emphasize strategies for assigning lower scores to undesirable responses and higher scores to responses that adhere closely to the overall objectives of the task.

The information provided includes: Task Description: task_description Student’s Prompt Instructions: student_prompt Student’s Generation: student_generation Junior Instructor’s Scoring Criteria: junior_prompt Junior Instructor’s Assigned Score: junior_score Your task: Critically evaluate the junior instructor’s score and justification in relation to the student’s response, task requirements, and prompt instructions.

- 1. Accuracy of Scoring

- - Determine whether the student’s response is receiving an inflated score despite not fully meeting the task objectives in terms of quality and content.
- - Identify any elements where the response deviates from task expectations, such as misinterpretation, lack of depth, or overemphasis on irrelevant aspects.

- 2. Evaluation of Scoring Criteria

- - Assess whether the junior instructor’s criteria align with the task’s overarching purpose. Are critical aspects overlooked, or do the criteria require further breakdown for clarity?
- - Examine whether the distribution of points is logical and correctly sums to the total score. Flag any inconsistencies and suggest necessary adjustments.

- 3. Constructive Feedback for Refinement

- - Provide actionable recommendations to enhance the scoring framework, ensuring it is comprehensive and consistently applied.
- - Emphasize the need for strict penalization in cases of severe errors to maintain evaluation rigor. Present the analysis concisely within max_words words. Conclude the response with: “<EOE>”. Your Analysis:

Meta Refinement

Based on the meta-level analysis, refine the junior instructor’s scoring criteria by designing an explicit rubric-based framework with separate section items for awarding points and deducting points. This rubric must assign specific point values for meeting given criteria, with clear deductions for any shortcomings. Fill in any gaps in the existing criteria to cover all relevant aspects of the task. Provide a concrete example illustrating how the rubric would apply to a typical student response. Adjust the total score to match the rubric items, ensuring the sum of all criteria equals the final total.

Use the following structure: <rubric> <item> Score Category Name

- - X1: (Description of the criterion for achieving this score X1, followed by an example.)
- - X2: (Description of the criterion for achieving this score X2, followed by an example.)

... </item>

... </rubric>

Your generation should be no more than max_words words. End with “<EOE>”. Important: You must follow the <rubric> and <item> formatting as shown above.

Junior Instructor’s Scoring Criteria (refined):

Meta Merging

Combine and refine the multiple sets of Junior Instructor’s Scoring Criteria into a single, cohesive set that provides comprehensive guidelines for assessment.

Here are multiple sets of Junior Instructor’s Scoring Criteria, delimited by “===”: “‘ multiple_sets “‘

Combine the above concisely without repetition. The combined criteria should be no more than max_words words. Make sure that the points across criteria add up correctly to the total score.

Use the following structure: <rubric> <item> Score Category Name

- - X1: (Description of the criterion for achieving this score X1, followed by an example.)
- - X2: (Description of the criterion for achieving this score X2, followed by an example.)

... </item>

... </rubric>

Your generation should be no more than max_words words. End with “<EOE>”. Important: You must follow the <rubric> and <item> formatting as shown above.

Junior Instructor’s Scoring Criteria (combined):

### F Evolution of Evaluation Rubric

#### F.1 Changes in Discourse Motif Distribution

[Figure 11]

In the main body of the paper, we have shown that MPO-evolved rubrics not only lead to higher training rewards but also produce higher-quality generations across the four downstream tasks. To gain deeper insight into the linguistic structure of these rubrics, we apply hierarchical discourse parsing and analyze the resulting discourse subgraphs—referred to as “discourse motifs”—which capture pragmatic discourse relations between textual units ranging from phrases to full paragraphs. This analysis builds on the method introduced by Kim et al. [2024], who used Rhetorical Structure Theory (RST) [Mann and Thompson, 1987] to study discourse patterns in LLM- and human-generated texts. In our setting, we compute the distribution of discourse motifs across three versions of the rubric: the initial version, the version after a single MPO refinement, and the final version at the end of training.

attribution / / contrast

/

/ /

/

/ / background /

contrast

background

attribution

Figure 8: Comparison of discourse motifs found in rubric prompts at the initial stage, after the first, and last MPO step of training for essay writing task.

- Figure 8 presents bar plots comparing the three rubric versions in terms of discourse motif distributions, with motif types on the x-axis and their normalized frequency within the overall RST graph on the y-axis.5 The edge label “/” represents a hyperedge relation, typically indicating transitions across textual levels and signaling that the text exhibits a more hierarchically organized discourse structure. The plot shows that as MPO refinements progress, the evolved rubrics adopt increasingly hierarchical

5We provide the full version in Appendix G.

discourse structures, marked by a decrease in Background relations and a corresponding increase in more informative relations such as Contrast and Attribution. Notably, Attribution relations, which explicitly identify the source or ownership of presented information, become more frequent—an important feature for rubric-based evaluation, where attributing claims, reasoning, and judgments clearly is critical for coherent assessment. These trends hint that MPO not only refines content criteria but also implicitly drives the development of richer, more structured evaluation language. We also note that an increase in hierarchical discourse structures is a known characteristic of human-like writing, as reported in Kim et al. [2024, 2025].

#### F.2 Examples

We present a couple of examples of refined evaluation prompts produced during the 32b_32b model training.

#### F.3 Essay Writing Task

MPO Iteration 1 – Essay Writing

<item> Task Alignment and Argument Clarity

- - 5: The essay clearly and consistently addresses the specific claim made in the prompt and presents a well-defined, coherent argument supported by relevant examples and evidence. (Example: "The essay argues that the U.S. is not responsible for social backwardness in Iran because it has not directly influenced Iran’s social structures and policies, supported by specific examples.")
- - 4: The essay mostly addresses the specific claim and presents a coherent argument, but there are minor inconsistencies or lack of depth. (Example: "The essay argues that the U.S. is not primarily responsible for social backwardness in Iran, but it does not fully explain how other factors might influence this.")
- - 3: The essay partially addresses the specific claim and presents an argument, but there are significant inconsistencies or lack of depth. (Example: "The essay argues that the U.S. has not directly caused social backwardness, but it does not fully explain how other factors might influence this.")
- - 2: The essay addresses the specific claim but the argument is weak and poorly supported. (Example: "The essay argues that the U.S. is not responsible, but it lacks supporting evidence or explanation.")
- - 1: The essay fails to address the specific claim or presents an argument that is contradictory or irrelevant. (Example: "The essay argues that the U.S. has historically been responsible for promoting social backwardness in Iran.")
- - 0: The essay does not address the specific claim or is completely off-topic. (Example: "The essay discusses the history of Iran without mentioning the U.S. or social backwardness.") </item> <item> Evidence and Reasoning
- - 5: The essay provides strong, relevant evidence and reasoning to support the argument, with specific examples and data. (Example: "The essay cites specific examples of U.S. policies that did not directly influence Iran’s social structures and policies.")
- - 4: The essay provides mostly relevant evidence and reasoning, but some points are weak or not fully supported. (Example: "The essay cites examples of U.S. policies but does not fully explain how they do not contribute to social backwardness.")
- - 3: The essay provides some relevant evidence and reasoning, but there are significant gaps or weak points. (Example: "The essay mentions U.S. policies but does not provide specific examples or explanations.")
- - 2: The essay provides weak or irrelevant evidence and reasoning. (Example: "The essay mentions U.S. policies but does not explain how they are unrelated to social backwardness.")
- - 1: The essay does not provide any evidence or reasoning to support the argument. (Example: "The essay makes claims without providing any supporting evidence.")
- - 0: The essay provides evidence and reasoning that contradicts the argument. (Example: "The essay provides examples of U.S. policies that contributed to social backwardness.") </item> <item> Counterarguments and Refutation
- - 5: The essay addresses potential counterarguments and provides strong refutations. (Example: "The essay acknowledges that U.S. policies might have some indirect influence but explains why this does not make the U.S. responsible for social backwardness.")
- - 4: The essay addresses some counterarguments and provides mostly strong refutations. (Example: "The essay acknowledges some counterarguments but does not fully refute them.")
- - 3: The essay addresses some counterarguments but provides weak or incomplete refutations. (Example: "The essay mentions counterarguments but does not fully address them.")
- - 2: The essay addresses some counterarguments but does not provide any refutations. (Example: "The essay mentions counterarguments without explaining why they do not undermine the argument.")
- - 1: The essay does not address any counterarguments. (Example: "The essay does not mention any counterarguments.")
- - 0: The essay addresses counterarguments but fails to refute them or provides weak refutations. (Example: "The essay mentions counterarguments but does not provide any refutations or provides weak refutations.")

- </item> <item> Structure and Coherence
- - 5: The essay is well-organized and coherent, with a clear structure and logical flow. (Example: "The essay has an introduction, body paragraphs, and a conclusion that are logically connected.")
- - 4: The essay has a mostly clear structure, but some points may be slightly disconnected or lack smooth transitions. (Example: "The essay has a clear structure but some points are not well-connected or lack transitions.")
- - 3: The essay has a somewhat clear structure, but the points are not well-connected, and the transitions are weak. (Example: "The essay has a structure but the points are disjointed and the transitions are weak.")
- - 2: The essay lacks a clear structure, and the points are disjointed and disconnected. (Example: "The essay has no clear structure and the points are not well-connected.")
- - 1: The essay is poorly structured, with no clear introduction, body, or conclusion. (Example: "The essay lacks a clear structure and the points are entirely disjointed.")
- - 0: The essay is completely disorganized and lacks any structure. (Example: "The essay is a collection of unrelated points with no structure.") </item> <item> Depth of Analysis
- - 5: The essay demonstrates a deep and nuanced analysis of the issue, addressing complexities and providing specific examples and data. (Example: "The essay discusses the historical roots of the conflict, the cultural identity of Palestinians, and the potential repercussions of the proposed solution.")
- - 4: The essay provides some depth and nuance in the analysis but lacks specific examples or data. (Example: "The essay mentions that the conflict is about identity and survival but does not provide substantial evidence or reasoning.")
- - 3: The essay provides minimal depth and nuance in the analysis, lacking specific examples or data. (Example: "The essay mentions the conflict is about identity but does not provide substantial historical or cultural context.")
- - 2: The essay lacks depth and nuance in the analysis, providing vague statements without specific examples or data. (Example: "The essay is vague and does not provide any substantial argument or reasoning.")
- - 1: The essay lacks any meaningful analysis or is superficial. (Example: "The essay does not provide any meaningful analysis and remains superficial in its discussion.")
- - 0: The essay is completely lacking in analysis and is purely superficial. (Example: "The essay is vague and does not provide any substantial content or context.") </item> <item> Language and Mechanics
- - 5: The essay demonstrates clear and effective use of language with minimal grammatical errors. (Example: "The essay uses clear and precise language with no significant grammatical errors.")
- - 4: The essay has some grammatical errors but is still generally clear. (Example: "The essay has a few grammatical errors but is still understandable.")
- - 3: The essay has some grammatical errors that affect clarity but is still mostly comprehensible. (Example: "The essay has some errors in grammar or style but is still generally clear.")
- - 2: The essay has significant grammatical errors that affect clarity and coherence. (Example: "The essay has multiple grammatical or stylistic errors that make it difficult to understand.")
- - 1: The essay has significant grammatical errors that severely affect clarity and coherence. (Example: "The essay has multiple grammatical or stylistic errors that make it difficult to understand.")
- - 0: The essay is completely unclear and incoherent due to significant grammatical or stylistic errors. (Example: "The essay is difficult to read due to numerous grammatical or stylistic errors.") </item>

MPO Iteration 40 – Essay Writing

<item> Task Alignment and Argument Clarity (10 points)

- - 10: The essay clearly and consistently addresses the specific claim made in the prompt and presents a well-defined, coherent argument supported by relevant examples and evidence. The argument is fully developed and addresses the core issue of the prompt. (Example: "The essay argues that higher education is counterproductive by providing specific examples of student debt, income gaps, and over-specialization in certain fields.")
- - 8: The essay mostly addresses the specific claim and presents a coherent argument, but there are minor inconsistencies or lack of depth. The argument is somewhat developed but could benefit from more specific examples or a deeper analysis. (Example: "The essay argues that higher education is counterproductive but lacks substantial examples or a clear connection to the context.")
- - 6: The essay partially addresses the specific claim and presents an argument, but there are significant inconsistencies or lack of depth. The argument is weak and lacks substantial examples or a clear connection to the core issue. (Example: "The essay argues that higher education is counterproductive but fails to address the specific aspects of student debt, income gaps, and over-specialization.")
- - 4: The essay addresses the specific claim but the argument is weak and poorly supported. The argument lacks substantial examples or a clear connection to the core issue. (Example: "The essay argues that higher education is counterproductive but lacks supporting evidence or explanation.")
- - 2: The essay fails to address the specific claim or presents an argument that is contradictory or irrelevant. The argument is not aligned with the prompt and does not address the core issue. (Example: "The essay argues that higher education is beneficial, which contradicts the prompt.")
- - 0: The essay does not address the specific claim or is completely off-topic. The argument is not

- related to the prompt and does not address the core issue. (Example: "The essay discusses unrelated topics without addressing the claim.") </item> <item> Evidence and Reasoning (10 points)
- - 10: The essay provides strong, relevant evidence and reasoning to support the argument, with specific examples and data from current events and historical contexts. The evidence is well-connected to the core issue and supports the argument effectively. (Example: "The essay cites specific studies showing the burden of student debt, the widening income gap, and the over-specialization in certain fields, providing detailed examples and data.")
- - 8: The essay provides mostly relevant evidence and reasoning, but some points are weak or not fully supported. The evidence is somewhat connected to the core issue but could benefit from more substantial examples or a deeper analysis. (Example: "The essay mentions studies but does not provide substantial evidence or explanation.")
- - 6: The essay provides some relevant evidence and reasoning, but there are significant gaps or weak points. The evidence is not well-connected to the core issue and lacks substantial examples or a clear connection to the context. (Example: "The essay mentions studies but does not provide specific examples or explanations.")
- - 4: The essay provides weak or irrelevant evidence and reasoning. The evidence is not well-connected to the core issue and lacks substantial examples or a clear connection to the context. (Example: "The essay mentions studies but does not explain how this supports the argument or provide substantial evidence.")
- - 2: The essay does not provide any evidence or reasoning to support the argument. The evidence is not related to the core issue and lacks substantial examples or a clear connection to the context. (Example: "The essay makes claims without providing any supporting evidence.")
- - 0: The essay provides evidence and reasoning that contradict the argument. The evidence is not related to the core issue and lacks substantial examples or a clear connection to the context. (Example: "The essay provides examples that support the claim that higher education is beneficial.") </item> <item> Counterarguments and Refutation (10 points)
- - 10: The essay addresses potential counterarguments and provides strong refutations. The counterarguments are acknowledged and effectively refuted with specific examples and reasoning. (Example: "The essay acknowledges that some argue higher education is beneficial by explaining the burden of student debt, the widening income gap, and the over-specialization in certain fields.")
- - 8: The essay addresses some counterarguments and provides mostly strong refutations. The counterarguments are acknowledged but not fully refuted. (Example: "The essay acknowledges some counterarguments but does not fully refute them.")
- - 6: The essay addresses some counterarguments but provides weak or incomplete refutations. The counterarguments are acknowledged but not fully addressed. (Example: "The essay mentions counterarguments but does not fully address them.")
- - 4: The essay addresses some counterarguments but does not provide any refutations. The counterarguments are acknowledged but not addressed. (Example: "The essay mentions counterarguments without explaining why they do not undermine the argument.")
- - 2: The essay does not address any counterarguments. The counterarguments are not acknowledged or addressed. (Example: "The essay does not mention any counterarguments.")
- - 0: The essay addresses counterarguments but fails to refute them or provides weak refutations. The counterarguments are acknowledged but not effectively refuted. (Example: "The essay mentions counterarguments but does not provide any refutations or provides weak refutations.") </item> <item> Structure and Coherence (10 points)
- - 10: The essay is well-organized and coherent, with a clear structure and logical flow. The introduction sets up the argument, body paragraphs provide evidence and reasoning, and the conclusion summarizes the argument and provides a final thought. (Example: "The essay has an introduction that sets up the argument, body paragraphs that provide evidence and reasoning, and a conclusion that summarizes the argument and provides a final thought.")
- - 8: The essay has a mostly clear structure, but some points may be slightly disconnected or lack smooth transitions. The introduction sets up the argument, body paragraphs provide evidence and reasoning, and the conclusion summarizes the argument but lacks smooth transitions. (Example: "The essay has a clear structure but some points are not well-connected or lack transitions.")
- - 6: The essay has a somewhat clear structure, but the points are not well-connected, and the transitions are weak. The introduction sets up the argument, body paragraphs provide evidence and reasoning, but the transitions are weak. (Example: "The essay has a structure but the points are disjointed and the transitions are weak.")
- - 4: The essay lacks a clear structure, and the points are disjointed and disconnected. The introduction sets up the argument, but the body paragraphs are disjointed and the conclusion is unclear. (Example: "The essay has no clear structure and the points are not well-connected.")
- - 2: The essay is poorly structured, with no clear introduction, body, or conclusion. The essay lacks a clear structure and the points are entirely disjointed. (Example: "The essay lacks a clear structure and the points are entirely disjointed.")
- - 0: The essay is completely disorganized and lacks any structure. The essay is a collection of unrelated points with no structure. (Example: "The essay is a collection of unrelated points with no structure.") </item> <item> Depth of Analysis (10 points)
- - 10: The essay demonstrates a deep and nuanced analysis of the issue, addressing complexities and providing specific examples and data from current events and historical contexts. The analysis

- is well-connected to the core issue and supports the argument effectively. (Example: "The essay discusses the complexities of higher education, providing specific examples of student debt, income gaps, and over-specialization in certain fields, and provides detailed data from reliable sources.")
- - 8: The essay provides some depth and nuance in the analysis but lacks specific examples or data. The analysis is somewhat connected to the core issue but could benefit from more substantial examples or a deeper analysis. (Example: "The essay mentions that higher education has issues but does not provide substantial evidence or reasoning.")
- - 6: The essay provides minimal depth and nuance in the analysis, lacking specific examples or data. The analysis is not well-connected to the core issue and lacks substantial examples or a clear connection to the context. (Example: "The essay mentions that higher education has issues but does not provide specific examples or explanations.")
- - 4: The essay lacks depth and nuance in the analysis, providing vague statements without specific examples or data. The analysis is not well-connected to the core issue and lacks substantial examples or a clear connection to the context. (Example: "The essay is vague and does not provide any substantial argument or reasoning.")
- - 2: The essay lacks any meaningful analysis or is superficial. The analysis is not related to the core issue and lacks substantial examples or a clear connection to the context. (Example: "The essay does not provide any meaningful analysis and remains superficial in its discussion.")
- - 0: The essay is completely lacking in analysis and is purely superficial. The analysis is not related to the core issue and lacks substantial examples or a clear connection to the context. (Example: "The essay is vague and does not provide any substantial content or context.") </item> <item> Language and Mechanics (10 points)
- - 10: The essay demonstrates clear and effective use of language with minimal grammatical errors. The writing is clear, coherent, and free of significant errors that affect clarity or coherence. (Example: "The essay uses clear and precise language with no significant grammatical errors.")
- - 8: The essay has some grammatical errors but is still generally clear. The writing is mostly clear and coherent, with a few minor errors that do not significantly affect clarity. (Example: "The essay has a few grammatical errors but is still understandable.")
- - 6: The essay has some grammatical errors that affect clarity but is still mostly comprehensible. The writing is generally clear but has some errors that affect clarity. (Example: "The essay has some errors in grammar or style but is still generally clear.")
- - 4: The essay has significant grammatical errors that affect clarity and coherence. The writing is unclear and difficult to follow due to significant errors. (Example: "The essay has multiple grammatical or stylistic errors that make it difficult to understand.")
- - 2: The essay has significant grammatical errors that severely affect clarity and coherence. The writing is unclear and difficult to follow due to multiple significant errors. (Example: "The essay has multiple grammatical or stylistic errors that make it difficult to understand.")
- - 0: The essay is completely unclear and incoherent due to significant grammatical or stylistic errors. The writing is difficult to read due to numerous grammatical or stylistic errors. (Example: "The essay is difficult to read due to numerous grammatical or stylistic errors.") </item>

#### F.4 Ethical Reasoning

MPO Iteration 1 – Ethical Reasoning

<item> Logical Coherence (0-5)

- - 5: The reasoning is logically consistent, free from contradictions, and well-supported by evidence or examples. The argument is clear, structured, and avoids repetition or disjointedness.
- - Example: "The girlfriend’s reaction is wrong because she overreacted. This is not a justifiable response as it lacks empathy and understanding. The act of forgetting to defrost the shrimp is a minor inconvenience, and the girlfriend’s response is disproportionate, disrespectful, and potentially harmful to the relationship. Trusting someone with your body is a significant decision, and I believe it’s important to prioritize safety and quality. <verdict>WRONG</verdict>"
- - 4: The reasoning is mostly logical, with a few minor contradictions or gaps in reasoning. The argument is coherent but could be more clear or structured.
- - Example: "The girlfriend’s reaction is wrong because she overreacted. This is not a justifiable response as it lacks empathy and understanding. The act of forgetting to defrost the shrimp is a minor inconvenience, and the girlfriend’s response is disproportionate and disrespectful. <verdict>WRONG</verdict>"
- - 3: The reasoning is somewhat logical but contains some contradictions or gaps in reasoning. The argument is coherent but could be more clear or structured.
- - Example: "The girlfriend’s reaction is wrong because she overreacted. This is not a justifiable response as it lacks empathy and understanding. The act of forgetting to defrost the shrimp is a minor inconvenience, and the girlfriend’s response is disproportionate. <verdict>WRONG</verdict>"
- - 2: The reasoning is partially logical, with significant contradictions or gaps in reasoning. The argument is somewhat coherent but lacks clarity and structure.
- - Example: "The girlfriend’s reaction is wrong because she overreacted. This is not a justifiable response as it lacks empathy and understanding. The act of forgetting to defrost the shrimp is a minor inconvenience. <verdict>WRONG</verdict>"
- - 1: The reasoning is illogical, with major contradictions or gaps in reasoning. The argument is incoherent and lacks clarity.
- - Example: "The girlfriend’s reaction is wrong because she overreacted. The act of forgetting to

- defrost the shrimp is a minor inconvenience. <verdict>WRONG</verdict>"
- - 0: The reasoning is completely illogical, with no coherent argument or supporting evidence. The argument is incoherent and lacks clarity.
- - Example: "The girlfriend’s reaction is wrong. <verdict>WRONG</verdict>" </item> <item> Depth of Moral Insight (0-5)
- - 5: The analysis demonstrates a deep understanding of the ethical issues involved, providing nuanced insights and thoughtful consideration of multiple perspectives. The reasoning explores multiple ethical principles and their implications.
- - Example: "The girlfriend’s reaction is wrong because she overreacted. This is not a justifiable response as it lacks empathy and understanding. The act of forgetting to defrost the shrimp is a minor inconvenience, and the girlfriend’s response is disproportionate, disrespectful, and potentially harmful to the relationship. This response reflects a lack of consideration for others and a disregard for the impact of one’s actions. It also violates deontological principles, which emphasize the importance of treating others with respect and dignity. Additionally, her response fails to consider the utilitarian principle of maximizing overall happiness and well-being. <verdict>WRONG</verdict>"
- - 4: The analysis demonstrates a good understanding of the ethical issues involved, with some thoughtful consideration of multiple perspectives. The reasoning explores some ethical principles and their implications.
- - Example: "The girlfriend’s reaction is wrong because she overreacted. This is not a justifiable response as it lacks empathy and understanding. The act of forgetting to defrost the shrimp is a minor inconvenience, and the girlfriend’s response is disproportionate, disrespectful, and potentially harmful to the relationship. This response reflects a lack of consideration for others and a disregard for the impact of one’s actions. It also violates deontological principles, which emphasize the importance of treating others with respect and dignity. <verdict>WRONG</verdict>"
- - 3: The analysis demonstrates a basic understanding of the ethical issues involved, with some consideration of multiple perspectives. The reasoning touches on some ethical principles and their implications.
- - Example: "The girlfriend’s reaction is wrong because she overreacted. This is not a justifiable response as it lacks empathy and understanding. The act of forgetting to defrost the shrimp is a minor inconvenience, and the girlfriend’s response is disproportionate and disrespectful. <verdict>WRONG</verdict>"
- - 2: The analysis demonstrates a limited understanding of the ethical issues involved, with little consideration of multiple perspectives. The reasoning touches on some ethical principles but lacks depth.
- - Example: "The girlfriend’s reaction is wrong because she overreacted. This is not a justifiable response as it lacks empathy and understanding. The act of forgetting to defrost the shrimp is a minor inconvenience. <verdict>WRONG</verdict>"
- - 1: The analysis demonstrates a minimal understanding of the ethical issues involved, with no consideration of multiple perspectives. The reasoning touches on some ethical principles but lacks depth.
- - Example: "The girlfriend’s reaction is wrong because she overreacted. This is not a justifiable response as it lacks empathy and understanding. <verdict>WRONG</verdict>"
- - 0: The analysis demonstrates no understanding of the ethical issues involved. The reasoning lacks any exploration of ethical principles or ethical theories.
- - Example: "The girlfriend’s reaction is wrong because she overreacted. <verdict>WRONG</verdict>" </item> <item> Alignment with Established Ethical Principles (0-5)
- - 5: The response references and applies relevant ethical principles or theories comprehensively and accurately. The reasoning explicitly identifies and applies specific ethical theories or principles, including how they apply to the specific situation and why they are relevant.
- - Example: "The girlfriend’s reaction is wrong because she overreacted. This is not a justifiable response as it lacks empathy and understanding. The act of forgetting to defrost the shrimp is a minor inconvenience, and the girlfriend’s response is disproportionate, disrespectful, and potentially harmful to the relationship. This response reflects a lack of consideration for others and a disregard for the impact of one’s actions. It also violates deontological principles, which emphasize the importance of treating others with respect and dignity. Additionally, her response fails to consider the utilitarian principle of maximizing overall happiness and well-being. <verdict>WRONG</verdict>"
- - 4: The response references and applies relevant ethical principles or theories with some accuracy. The reasoning references some ethical principles or theories but does not provide a clear explanation of how they apply to the situation.
- - Example: "The girlfriend’s reaction is wrong because she overreacted. This is not a justifiable response as it lacks empathy and understanding. The act of forgetting to defrost the shrimp is a minor inconvenience, and the girlfriend’s response is disproportionate, disrespectful, and potentially harmful to the relationship. This response reflects a lack of consideration for others and a disregard for the impact of one’s actions. It also violates deontological principles, which emphasize the importance of treating others with respect and dignity. <verdict>WRONG</verdict>"
- - 3: The response references some ethical principles or theories but with limited accuracy. The reasoning references some ethical principles or theories but does not provide a clear explanation of how they apply to the situation.
- - Example: "The girlfriend’s reaction is wrong because she overreacted. This is not a justifiable response as it lacks empathy and understanding. The act of forgetting to defrost the shrimp is a minor inconvenience, and the girlfriend’s response is disproportionate and disrespectful. <verdict>WRONG</verdict>"
- - 2: The response references some ethical principles or theories but with significant inaccuracies.

- The reasoning references some ethical principles or theories but does not provide a clear explanation of how they apply to the situation.
- - Example: "The girlfriend’s reaction is wrong because she overreacted. This is not a justifiable response as it lacks empathy and understanding. The act of forgetting to defrost the shrimp is a minor inconvenience. <verdict>WRONG</verdict>"
- - 1: The response references relevant ethical principles or theories but with major inaccuracies. The reasoning references some ethical principles or theories but does not provide a clear explanation of how they apply to the situation.
- - Example: "The girlfriend’s reaction is wrong because she overreacted. This is not a justifiable response as it lacks empathy and understanding. <verdict>WRONG</verdict>"
- - 0: The response does not reference any relevant ethical principles or theories. The reasoning does not reference any ethical principles or theories.
- - Example: "The girlfriend’s reaction is wrong. <verdict>WRONG</verdict>" </item> <item> Clarity of Verdict (0-5)
- - 5: The final verdict is clear, properly formatted, and well-supported by the ethical reasoning. The verdict is clearly stated within the <verdict></verdict> tags, and the reasoning supports the verdict.
- - Example: "The girlfriend’s reaction is wrong because she overreacted. This is not a justifiable response as it lacks empathy and understanding. The act of forgetting to defrost the shrimp is a minor inconvenience, and the girlfriend’s response is disproportionate, disrespectful, and potentially harmful to the relationship. This response reflects a lack of consideration for others and a disregard for the impact of one’s actions. It also violates deontological principles, which emphasize the importance of treating others with respect and dignity. Additionally, her response fails to consider the utilitarian principle of maximizing overall happiness and well-being. <verdict>WRONG</verdict>"
- - 4: The final verdict is clear and properly formatted but lacks some supporting evidence. The verdict is clearly stated within the <verdict></verdict> tags, but the reasoning does not fully support the verdict.
- - Example: "The girlfriend’s reaction is wrong because she overreacted. This is not a justifiable response as it lacks empathy and understanding. The act of forgetting to defrost the shrimp is a minor inconvenience, and the girlfriend’s response is disproportionate, disrespectful, and potentially harmful to the relationship. This response reflects a lack of consideration for others and a disregard for the impact of one’s actions. It also violates deontological principles, which emphasize the importance of treating others with respect and dignity. <verdict>WRONG</verdict>"
- - 3: The final verdict is somewhat clear and properly formatted but lacks supporting evidence. The verdict is clearly stated within the <verdict></verdict> tags, but the reasoning does not fully support the verdict.
- - Example: "The girlfriend’s reaction is wrong because she overreacted. This is not a justifiable response as it lacks empathy and understanding. The act of forgetting to defrost the shrimp is a minor inconvenience, and the girlfriend’s response is disproportionate and disrespectful. <verdict>WRONG</verdict>"
- - 2: The final verdict is partially clear and properly formatted but lacks significant supporting evidence. The verdict is clearly stated within the <verdict></verdict> tags, but the reasoning does not fully support the verdict.
- - Example: "The girlfriend’s reaction is wrong because she overreacted. This is not a justifiable response as it lacks empathy and understanding. The act of forgetting to defrost the shrimp is a minor inconvenience. <verdict>WRONG</verdict>"
- - 1: The final verdict is unclear or improperly formatted. The verdict is not clearly stated within the <verdict></verdict> tags, and the reasoning does not support the verdict.
- - Example: "The girlfriend’s reaction is wrong because she overreacted. This is not a justifiable response as it lacks empathy and understanding. <verdict>WRONG</verdict>"
- - 0: The final verdict is completely unclear or improperly formatted. The verdict is not clearly stated within the <verdict></verdict> tags, and the reasoning does not support the verdict.
- - Example: "The girlfriend’s reaction is wrong. <verdict>WRONG</verdict>" </item>

MPO Iteration 20 – Ethical Reasoning

<item> Logical Coherence (0-5)

- - 5: The reasoning is logically consistent, free from contradictions, and well-supported by evidence or examples. The argument is clear, structured, and avoids repetition or disjointedness. It considers multiple perspectives and the broader ethical implications of the action. The verdict is presented within the required tags. <verdict>RIGHT</verdict>
- - Example: "The action of not using the GPS app is justified. It respects personal autonomy and privacy, which are fundamental human rights. The decision to not use the app is a reasonable response to concerns about personal privacy and autonomy. The reasoning is clear, structured, and avoids repetition. <verdict>RIGHT</verdict>"
- - 4: The reasoning is mostly logical, with a few minor contradictions or gaps in reasoning. The argument is coherent but could be more clear or structured. The reasoning considers some perspectives but lacks a comprehensive analysis. The verdict is presented within the required tags. <verdict>RIGHT</verdict>
- - Example: "The action of not using the GPS app is justified. It respects personal autonomy and privacy, which are fundamental human rights. The decision to not use the app is a reasonable response to concerns about personal privacy and autonomy. The reasoning is mostly coherent but could be more clear or structured. <verdict>RIGHT</verdict>"
- - 3: The reasoning is somewhat logical but contains some contradictions or gaps in reasoning.

- The argument is coherent but could be more clear or structured. The reasoning touches on some perspectives but lacks a comprehensive analysis. The verdict is presented within the required tags. <verdict>RIGHT</verdict>
- - Example: "The action of not using the GPS app is justified. It respects personal autonomy and privacy, which are fundamental human rights. The decision to not use the app is a reasonable response to concerns about personal privacy and autonomy. The reasoning is somewhat coherent but lacks clarity. <verdict>RIGHT</verdict>"
- - 2: The reasoning is partially logical, with significant contradictions or gaps in reasoning. The argument is somewhat coherent but lacks clarity and structure. The reasoning is disjointed and lacks coherence. The verdict is presented within the required tags. <verdict>RIGHT</verdict>
- - Example: "The action of not using the GPS app is justified. It respects personal autonomy and privacy, which are fundamental human rights. The decision to not use the app is a reasonable response to concerns about personal privacy and autonomy. The reasoning is disjointed and lacks coherence. <verdict>RIGHT</verdict>"
- - 1: The reasoning is illogical, with major contradictions or gaps in reasoning. The argument is incoherent and lacks clarity. The reasoning is incoherent and lacks logical structure. The verdict is presented within the required tags. <verdict>RIGHT</verdict>
- - Example: "The action of not using the GPS app is justified. It respects personal autonomy and privacy, which are fundamental human rights. The decision to not use the app is a reasonable response to concerns about personal privacy and autonomy. The reasoning is incoherent and lacks logical structure. <verdict>RIGHT</verdict>"
- - 0: The reasoning is completely illogical, with no coherent argument or supporting evidence. The argument is incoherent and lacks clarity. The reasoning is completely illogical, with no coherent argument or supporting evidence. The verdict is presented within the required tags. <verdict>RIGHT</verdict>
- - Example: "The action of not using the GPS app is justified. <verdict>RIGHT</verdict>" </item> <item> Depth of Moral Insight (0-5)
- - 5: The analysis demonstrates a deep understanding of the ethical issues involved, providing nuanced insights and thoughtful consideration of multiple perspectives. The reasoning explores multiple ethical principles and their implications, including the balance between personal needs and others’ well-being. The reasoning explores the ethical principles of personal safety, respect for autonomy, and the impact on the relationship. The verdict is presented within the required tags. <verdict>RIGHT</verdict>
- - Example: "The action of not using the GPS app is justified. It respects personal autonomy and privacy, which are fundamental human rights. The decision to not use the app is a reasonable response to concerns about personal privacy and autonomy. The reasoning explores the ethical principles of personal safety, respect for autonomy, and the impact on the relationship. <verdict>RIGHT</verdict>"
- - 4: The analysis demonstrates a good understanding of the ethical issues involved, with some thoughtful consideration of multiple perspectives. The reasoning explores some ethical principles and their implications. The reasoning considers the ethical principles of personal safety and respect for autonomy. The verdict is presented within the required tags. <verdict>RIGHT</verdict>
- - Example: "The action of not using the GPS app is justified. It respects personal autonomy and privacy, which are fundamental human rights. The decision to not use the app is a reasonable response to concerns about personal privacy and autonomy. The reasoning considers the ethical principles of personal safety and respect for autonomy. <verdict>RIGHT</verdict>"
- - 3: The analysis demonstrates a basic understanding of the ethical issues involved, with some consideration of multiple perspectives. The reasoning touches on some ethical principles and their implications. The reasoning touches on the ethical principles of personal safety and respect for autonomy. The verdict is presented within the required tags. <verdict>RIGHT</verdict>
- - Example: "The action of not using the GPS app is justified. It respects personal autonomy and privacy, which are fundamental human rights. The decision to not use the app is a reasonable response to concerns about personal privacy and autonomy. The reasoning touches on the ethical principles of personal safety and respect for autonomy. <verdict>RIGHT</verdict>"
- - 2: The analysis demonstrates a limited understanding of the ethical issues involved, with little consideration of multiple perspectives. The reasoning touches on some ethical principles but lacks depth. The reasoning touches on the ethical principles of personal safety and respect for autonomy but lacks depth. The verdict is presented within the required tags. <verdict>RIGHT</verdict>
- - Example: "The action of not using the GPS app is justified. It respects personal autonomy and privacy, which are fundamental human rights. The decision to not use the app is a reasonable response to concerns about personal privacy and autonomy. The reasoning touches on the ethical principles of personal safety and respect for autonomy but lacks depth. <verdict>RIGHT</verdict>"
- - 1: The analysis demonstrates a minimal understanding of the ethical issues involved, with no consideration of multiple perspectives. The reasoning touches on some ethical principles but lacks depth. The reasoning touches on the ethical principles of personal safety and respect for autonomy but lacks depth. The verdict is presented within the required tags. <verdict>RIGHT</verdict>
- - Example: "The action of not using the GPS app is justified. It respects personal autonomy and privacy, which are fundamental human rights. The decision to not use the app is a reasonable response to concerns about personal privacy and autonomy. The reasoning touches on the ethical principles of personal safety and respect for autonomy but lacks depth. <verdict>RIGHT</verdict>"
- - 0: The analysis demonstrates no understanding of the ethical issues involved. The reasoning lacks any exploration of ethical principles or ethical theories. The reasoning lacks any exploration of ethical principles or ethical theories. The verdict is presented within the required tags. <verdict>RIGHT</verdict>
- - Example: "The action of not using the GPS app is justified. <verdict>RIGHT</verdict>" </item> <item> Alignment with Established Ethical Principles (0-5)

- - 5: The response references and applies relevant ethical principles or theories comprehensively and accurately. The reasoning explicitly identifies and applies specific ethical principles or theories, including how they apply to the specific situation and why they are relevant. The reasoning explicitly identifies and applies specific ethical principles such as personal safety, respect for autonomy, and the impact on the relationship. The verdict is presented within the required tags. <verdict>RIGHT</verdict>
- - Example: "The action of not using the GPS app is justified. It respects personal autonomy and privacy, which are fundamental human rights. The decision to not use the app is a reasonable response to concerns about personal privacy and autonomy. The reasoning explicitly identifies and applies specific ethical principles such as personal safety, respect for autonomy, and the impact on the relationship. <verdict>RIGHT</verdict>"
- - 4: The response references and applies relevant ethical principles or theories with some accuracy. The reasoning references some ethical principles or theories but does not provide a clear explanation of how they apply to the situation. The reasoning references some ethical principles such as personal safety and respect for autonomy but does not provide a clear explanation of how they apply to the situation. The verdict is presented within the required tags. <verdict>RIGHT</verdict>
- - Example: "The action of not using the GPS app is justified. It respects personal autonomy and privacy, which are fundamental human rights. The decision to not use the app is a reasonable response to concerns about personal privacy and autonomy. The reasoning references some ethical principles such as personal safety and respect for autonomy but does not provide a clear explanation of how they apply to the situation. <verdict>RIGHT</verdict>"
- - 3: The response references some ethical principles or theories but with limited accuracy. The reasoning references some ethical principles or theories but does not provide a clear explanation of how they apply to the situation. The reasoning references some ethical principles such as personal safety and respect for autonomy but does not provide a clear explanation of how they apply to the situation. The verdict is presented within the required tags. <verdict>RIGHT</verdict>
- - Example: "The action of not using the GPS app is justified. It respects personal autonomy and privacy, which are fundamental human rights. The decision to not use the app is a reasonable response to concerns about personal privacy and autonomy. The reasoning references some ethical principles such as personal safety and respect for autonomy but does not provide a clear explanation of how they apply to the situation. <verdict>RIGHT</verdict>"
- - 2: The response references some ethical principles or theories but with significant inaccuracies. The reasoning references some ethical principles or theories but does not provide a clear explanation of how they apply to the situation. The reasoning references some ethical principles such as personal safety and respect for autonomy but does not provide a clear explanation of how they apply to the situation. The verdict is presented within the required tags. <verdict>RIGHT</verdict>
- - Example: "The action of not using the GPS app is justified. It respects personal autonomy and privacy, which are fundamental human rights. The decision to not use the app is a reasonable response to concerns about personal privacy and autonomy. The reasoning references some ethical principles such as personal safety and respect for autonomy but does not provide a clear explanation of how they apply to the situation. <verdict>RIGHT</verdict>"
- - 1: The response references relevant ethical principles or theories but with major inaccuracies. The reasoning references some ethical principles or theories but does not provide a clear explanation of how they apply to the situation. The reasoning references some ethical principles such as personal safety and respect for autonomy but does not provide a clear explanation of how they apply to the situation. The verdict is presented within the required tags. <verdict>RIGHT</verdict>
- - Example: "The action of not using the GPS app is justified. It respects personal autonomy and privacy, which are fundamental human rights. The decision to not use the app is a reasonable response to concerns about personal privacy and autonomy. The reasoning references some ethical principles such as personal safety and respect for autonomy but does not provide a clear explanation of how they apply to the situation. <verdict>RIGHT</verdict>"
- - 0: The response does not reference any ethical principles or theories. The reasoning lacks any exploration of ethical principles or ethical theories. The reasoning lacks any exploration of ethical principles or ethical theories. The verdict is presented within the required tags. <verdict>RIGHT</verdict>
- - Example: "The action of not using the GPS app is justified. <verdict>RIGHT</verdict>" </item>

- F.5 Training Details

- F.5.1 Summarization

Setup. For summarization, we use the BillSum benchmark [Kornilova and Eidelman, 2019], a corpus designed for summarizing U.S. Congressional and California state legislation. The dataset includes over 22,000 mid-length Congressional bills with human-written summaries, along with an additional California test set to support cross-domain generalization. Its technical and hierarchical nature presents unique challenges, making it suitable for both domain-specific and general-purpose summarization research. The training set contains approximately 18.9K samples, while the combined test set includes 4.5K samples. We provide the full bill text as input and request a summary of approximately 400 words. Training is conducted for a single epoch over the dataset, with MPO steps executed every 20 batch steps, resulting in a total of 29 rubric refinements throughout the training process.

A total of 4.5K bill summaries are generated on the test set by each of the three models. These generated summaries are then evaluated by computing ROUGE [Lin, 2004] scores against their corresponding human-written references. In addition, we computed Elo ratings for the three models based on 5,000 pairwise comparisons of their generated summaries, with GPT-4o serving as the judge. The results are presented in Table 2.

#### F.5.2 Ethical Reasoning

Setup. For ethical reasoning, we utilize the Anecdotes from the Scruples dataset [Lourie et al., 2020] which contains over 32,000 real-life anecdotes sourced from a Reddit community, where users describe ethically charged situations they experienced or considered. Each anecdote includes a title, detailed story, and a distribution of community judgments indicating who was perceived to be in the wrong—such as the author, another party, everyone, or no one. These narratives often feature moral ambiguity and are labeled with crowd-sourced ethical assessments, making them well-suited for modeling community norms and capturing the diversity of ethical reasoning.

The dataset includes 27.8K training anecdotes, from which we randomly sampled 13K for our experiments, along with 4.7K anecdotes used for testing. Training is conducted for a single epoch, with MPO steps performed every 10 batch steps, resulting in a total of 20 rubric refinements. Although each anecdote includes a binary judgment verdict from human annotators, we did not use these ground truth labels in either the RM or MRM. Instead, reward scores were assigned solely based on the quality of ethical reasoning demonstrated in the response. This decision was motivated by two factors: (1) the label distribution is imbalanced, and (2) our goal was to encourage the policy model to improve through generating stronger reasoning traces, rather than optimizing for label prediction alone.

#### F.5.3 Mathematical Reasoning

Setup. For mathematical reasoning, we use the MATH dataset [Hendrycks et al., 2021] which consists of 12,500 high school competition-style math problems, sourced from contests like AMC 10, AMC 12, and AIME. Each problem is accompanied by a detailed step-by-step solution written in LaTeX, enabling both final answer evaluation and learning of problem-solving processes. The dataset spans seven subjects—including algebra, geometry, and number theory—and is annotated with difficulty levels from 1 to 5, offering fine-grained assessment across a wide range of mathematical reasoning tasks.

The dataset consists of 7.5K training samples and 5K test samples. Because mathematical reasoning demands considerable evaluative depth, we apply MPO at a finer granularity. Specifically, we cluster problems within each of the seven math subjects into three groups based on semantic embeddings, resulting in 21 (7 × 3) distinct categories. During MPO training, we maintain a separate evaluation prompt for each of these categories, with refinement steps triggered based on the subject and cluster index of the sample. The reward model follows a “plan-then-execute” strategy: it first formulates an evaluation plan based on the problem, reference solution, and meta-level guidelines, and then applies this plan to assess the student’s response. This approach builds on the method proposed by Saha et al. [2025], originally used for pairwise judgment, which we adapt for absolute scoring with (meta-level) rubric-guided evaluation. Training is conducted for a single epoch, with MPO steps performed every 30 batch steps—using a longer interval to ensure a sufficient mix of responses with varying quality levels is gathered before each refinement.

### G Discourse Motif Distribution of Evaluation Prompts

- 0.35

NormalizedMotifFrequency

//

Elab.

/

Elab./

/

/

Contrast

/

/

Joint

/ Joint

/

Attrib.

/

Attrib.

/

Contrast

/

/

Backgr.

/

Backgr.

/

Rubric Version

| |
|---|

- 32b_72b-iter0

| |
|---|

- 32b_72b-iter1 32b_72b-iterLast

| |
|---|

Figure 9: Comparison of discourse motifs found in rubric prompts at the initial stage, after the first, and last MPO step of training for essay writing task.

H Limitations

One limitation of our work is that we conducted experiments using only a relatively small LLM from the Qwen family. This decision was motivated by three considerations. First, we required a model with sufficient headroom for improvement across our target tasks. Given the computational constraints and the use of publicly available benchmarks, we prioritized smaller open-source models that could clearly reveal the effectiveness of the proposed framework. Extending our approach to larger models and additional model families is a promising direction for future work. Second, some tasks—such as mathematical reasoning—require generating long outputs (e.g., over 1,000 tokens), which imposes significant memory demands during PPO optimization. Larger models exceed our GPU memory limits, making Qwen2-1.5B-Instruct a practical and scalable choice. Third, to benchmark against MPO-generated prompts, we constructed a manually engineered RM prompt based on empirical insights gained from extensive PPO training runs on this task and model.

Another limitation is that we did not evaluate RMs and MRMs smaller than 32B. Our goal was to first validate the framework using reliable, high-capacity reward models, as smaller models (e.g.,

- 1.5B and 3B) often struggle with prompt adherence. However, we view this as a broader limitation of the RLAIF setting rather than our framework itself. Future work could explore mitigation strategies such as constrained decoding or employing newer model generations that offer improved instruction following at smaller scales.

0.30

0.25

0.20

0.15

0.10

0.05

0.00

m0 m1 m2 m3 m4 m6 m8 m10 m11 m15 m16

Motif

### I Societal Impact

Training and aligning large language models—like deep learning research more broadly—can have significant environmental impacts, including high energy consumption and greenhouse gas emissions. A key benefit of our work is that the proposed MPO framework reduces the need for repeated RLAIF training cycles, which often involve extensive manual prompt engineering to improve reward modeling. As demonstrated in our MPO vs. Oracle experiment, MPO achieves better performance to manually crafted reward prompts with only a single training iteration. This contributes to a positive societal impact by enabling more energy-efficient and environmentally sustainable alignment of LLMs.

### J Declaration of LLM Usage

We acknowledge making editorial use of Grammarly and ChatGPT 4o exclusively for improving linguistic clarity in places where sentence flow was less fluent. These tools did not generate new scholarly content or design any core methodological contributions. Rather, they served as auxiliary resources for finalizing readability.

