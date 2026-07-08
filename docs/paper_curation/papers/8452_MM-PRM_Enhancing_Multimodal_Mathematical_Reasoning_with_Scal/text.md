arXiv:2505.13427v2[cs.AI]5Jun2025

# MM-PRM: Enhancing Multimodal Mathematical Reasoning with Scalable Step-Level Supervision

Lingxiao Du1∗ Fanqing Meng1,3∗ Zongkai Liu2∗ Zhixiang Zhou2∗ Ping Luo4 Qiaosheng Zhang1,2† Wenqi Shao1,2†

1Shanghai AI Laboratory 2Shanghai Innovation Institute 3Shanghai Jiao Tong University 4The University of Hong Kong

## Abstract

While Multimodal Large Language Models (MLLMs) have achieved impressive progress in vision-language understanding, they still struggle with complex multistep reasoning, often producing logically inconsistent or partially correct solutions. A key limitation lies in the lack of fine-grained supervision over intermediate reasoning steps. To address this, we propose MM-PRM, a process reward model trained within a fully automated, scalable framework. We first build MM-Policy, a strong multimodal model trained on diverse mathematical reasoning data. Then, we construct MM-K12, a curated dataset of 10,000 multimodal math problems with verifiable answers, which serves as seed data. Leveraging a Monte Carlo Tree Search (MCTS)-based pipeline, we generate over 700k step-level annotations without human labeling. The resulting PRM is used to score candidate reasoning paths in the Best-of-N inference setup and achieves significant improvements across both in-domain (MM-K12 test set) and out-of-domain (OlympiadBench, MathVista, etc.) benchmarks. Further analysis confirms the effectiveness of soft labels, smaller learning rates, and path diversity in optimizing PRM performance. MM-PRM demonstrates that process supervision is a powerful tool for enhancing the logical robustness of multimodal reasoning systems. We release all our codes and data at https://github.com/ModalMinds/MM-PRM.

## 1 Introduction

The rapid advancement of Large Language Models (LLMs) [1, 2, 3, 4, 5, 6, 7, 8, 9] has significantly improved performance on a wide range of natural language processing tasks, including general reasoning and mathematical problem-solving. In parallel, the development of Multimodal Large Language Models (MLLMs) [10, 11, 12, 13, 14, 15, 16, 17, 18] has unlocked new capabilities in vision-language understanding, showing promising results in areas such as image captioning and visual question answering (VQA). However, despite their impressive capabilities in perception and basic reasoning, MLLMs still struggle with complex multi-step reasoning tasks, particularly in mathematics. These shortcomings often manifest as broken logical chains, inaccurate intermediate steps, or cases that produce incorrect intermediate steps while still occasionally arriving at the correct final answer—a phenomenon that introduces high false-positive rates and undermines interpretability.

To address this issue, reward modeling [19, 20, 21, 22] has emerged as a promising paradigm. Reward models (RMs) play a central role in reinforcement learning from human feedback (RLHF) [23, 24, 25, 26, 27, 28], and can also be used at inference time to select among multiple candidate responses using Test-Time Scaling (TTS) [29, 30, 31, 32, 33, 34, 35, 36, 37] strategies such as Best-of-N (BoN). In general, reward models for reasoning tasks can be broadly categorized into two types: Outcome Reward Models (ORMs) and Process Reward Models (PRMs). ORMs [19, 20] provide

∗Equal contribution †Corresponding Authors: shaowenqi@pjlab.org.cn; zhangqiaosheng@pjlab.org.cn

Preprint. Under review.

scalar feedback only on the final answer, overlooking the quality of the intermediate reasoning steps. This limits their ability to guide the model toward robust reasoning paths. In contrast, PRMs [38, 39, 21, 31, 40, 41] offer a more fine-grained approach by evaluating each reasoning step, enabling more accurate and interpretable feedback.

Some recent works have explored process reward models for mathematical reasoning in pure text. PRM800k [39] manually constructs a large-scale dataset with step-level correctness labels, which is hard to scale. MathShepherd [31] adopts Monte Carlo estimation to label reasoning steps by evaluating whether continuations from a given step lead to the correct answer, but its efficiency is relatively low. OmegaPRM [41] introduces a Monte Carlo Tree Search (MCTS)-based framework that enables efficient automated generation of process supervision data. However, all these works focus on mathematical reasoning in pure text. In the field of multimodal mathematical reasoning, how to design an efficient framework for process supervision data generation and stably train process reward models remains a challenging problem.

To address these challenges, we propose MM-PRM, a powerful process reward model that effectively handles both in-domain and out-of-domain problems. Specifically, we first collect a large-scale, highquality reasoning dataset and refine its format to train a policy model called MM-Policy with strong reasoning abilities and well-structured output. Inspired by OmegaPRM, we then establish an efficient multimodal process annotation framework. We first collect 10k high-quality K-12-level multimodal mathematical reasoning samples, named MM-K12. Starting with only 10k math problems as seed data, our framework automatically generates over 700k step-level annotations without any human supervision. Finally, we train MM-PRM with these data and evaluate its performance on multiple benchmarks using the BoN approach.

MM-PRM demonstrates strong performance and generalization across multiple benchmarks. Although we only use process data generated from K-12-level math problems for training, MM-PRM achieves remarkable results with MM-Policy on benchmarks such as MathVista that improves from 62.93% to 67.60%. In addition, even though MM-PRM is trained on data produced by MM-Policy, it still delivers competitive results on other models like InternVL2.5-8B and InternVL2.5-78B. For instance, on the self-collected MM-K12 test set, InternVL2.5-8B improves from 27.01% to 37.80%, and on MathVerse, InternVL2.5-78B improves from 50.18% to 54.47%. These results highlight MM-PRM’s ability to generalize both across datasets and across model size, despite being trained on limited and fixed data sources.

Finally, we provide a detailed discussion on key factors in PRM training, including learning rate and the choice between soft labels and hard labels. We find that the following factors are essential for stable PRM training: (1) using a small learning rate to ensure stable optimization; (2) adopting soft labels instead of hard thresholding to reduce noise and preserve step-wise uncertainty.

In summary, our contributions are as follows:

- • We collect and release MM-K12, a curated multimodal math dataset containing 10,000 seed problems and 500 test problems, all verified to contain unique, checkable answers.
- • We build a policy model named MM-Policy using a large corpus of mathematical reasoning data, and develop a fully automated, MCTS-based pipeline for scalable process supervision generation. Leveraging MM-Policy and only 10k seed problems from MM-K12, our framework produces over 700k step-level annotations without human supervision. Based on this, we train MM-PRM, which demonstrates strong performance across multiple benchmarks—for example, boosting accuracy on the MM-K12 test set from 33.92% to 42.80%, on MathVista from 62.93% to 67.60%, and on OlympiadBench from 15.41% to 24.00%.
- • We provide a detailed discussion on the settings of key parameters such as the type of process labels and the learning rate. We summarize the key factors for stable PRM training, which can help the community train PRM models more effectively.

## 2 Related work

### 2.1 Mathematical reasoning and reward modeling in LLMs

Mathematical reasoning has become a focal point in evaluating the deep logical abilities of LLMs. [19, 42]. Unlike general language tasks, mathematical problems demand precise multi-step reasoning

and logical coherence, motivating approaches like Chain-of-Thought (CoT) prompting[43] and Self-Consistency sampling [44]. Supervised fine-tuning (SFT) with structured solution datasets further reinforces reasoning performance, as demonstrated by models like Qwen [2], InternLM [7], and Gemini [3]. More recently, reinforcement learning (RL)-based optimization, exemplified by OpenAI’s o1 [45] and DeepSeek R1 [9], has emerged as a powerful strategy that surpassing SFT-only baselines.

However, despite these advances, current LLMs still frequently produce logically inconsistent reasoning steps or false-positive solutions. To address this issue, reward modeling techniques have emerged. Traditional Outcome Reward Models (ORMs)[19, 20], which assign rewards based solely on the final answer, fail to detect flawed intermediate reasoning. Process Reward Models (PRMs)[39, 31, 40, 41] overcome this limitation by explicitly providing step-level supervision, significantly improving logical coherence. Representative works such as PRM800K [39], MathShepherd [31], and MiPS [40] have demonstrated the effectiveness of PRMs in enhancing mathematical reasoning. During the preparation of this manuscript, we became aware of a concurrent effort, URSA [46], which also explores PRM training. While both approaches share some methodological similarities in PRM construction, our work places greater emphasis on building a scalable PRM pipeline and analyzing training dynamics, whereas URSA focuses more on integrating PRMs into RL.

Overall, PRMs have become an essential approach for improving the robustness and logical consistency of LLM reasoning through fine-grained, step-level evaluation.

- 2.2 Process supervision data construction

Training PRMs requires large-scale, high-quality step-level annotations that indicate the correctness of each intermediate reasoning step. However, most publicly available datasets focus solely on final answers, making it challenging to obtain supervision signals for reasoning quality. Existing process supervision construction methods can be broadly categorized into three classes:

Manual Annotation. The earliest efforts, such as PRM800K [39], relied on human experts to label each reasoning step. While this yields high-quality annotations, the approach is costly, timeconsuming, and difficult to scale—particularly for domains like mathematics where expert knowledge is required.

Monte Carlo Estimation. Works like MathShepherd [31] and MiPS [40] automate supervision by sampling multiple rollouts from each reasoning prefix and estimating step correctness based on the proportion of successful completions. If continuations from a given step often lead to correct answers, that step is considered reliable. This method is simple but suffers from label instability due to high variance in sampled rollouts—especially on long or complex reasoning chains.

Monte Carlo Tree Search (MCTS). To address these limitations, OmegaPRM [41] introduces a structured, search-based alternative. Using a divide-and-conquer MCTS algorithm, it efficiently identifies the first error in a reasoning chain and constructs a tree of reasoning paths with dynamically updated statistics. This approach improves annotation stability, better handles deep reasoning tasks, and generates large volumes of fine-grained supervision from limited seed data without human input.

Each method reflects a trade-off between cost, scalability, and label fidelity. Among them, MCTSbased approaches have recently emerged as the most effective strategy for generating robust and scalable process supervision, especially for multi-step mathematical reasoning tasks.

- 3 Method

- 3.1 Overview

We present a scalable and fully automated framework for training PRMs tailored to multimodal mathematical reasoning. Our approach addresses two critical bottlenecks in prior work: the lack of step-level supervision and the inefficiency of data generation in complex reasoning tasks. To overcome these challenges, we introduce a structured three-stage pipeline that enables fine-grained reward modeling without any human annotation. As shown in Figure 1, the framework comprises three interconnected stages as follows:

[Figure 1]

- Figure 1: Overview of our automated three-stage framework for training MM-PRM: supervised policy model construction, MCTS-based process supervision data generation, and step-level reward model training.

- (1) Policy Model Construction, where a multimodal policy model is trained to generate high-quality reasoning traces following the CoT paradigm.
- (2) Process Supervision Data Generation, where we use a MCTS-based engine, OmegaPRM [41], to efficiently identify reasoning flaws and produce step-level reward labels at scale.
- (3) Process Reward Model Training, where a PRM is trained to evaluate each reasoning step and provide dense feedback.

This end-to-end design ensures that process supervision can be generated, modeled, and applied in a fully closed loop. It significantly improves reasoning quality and robustness, especially in tasks requiring long logical chains. In Sections 3.2–3.3 below, we elaborate on these three stages.

### 3.2 Policy model construction

The policy model serves as the foundation of our framework, responsible for generating candidate reasoning trajectories given multimodal math problems. These trajectories are later evaluated and labeled to form step-level supervision for training the PRM. Ensuring that the policy model produces logically coherent and structurally complete outputs is thus essential for the effectiveness of the entire system.

To train the policy model, we curated a large-scale, high-quality dataset of mathematical problems spanning a wide range of topics and difficulty levels. Our dataset integrates samples from over a dozen public math datasets, including R-CoT [47], MAVIS [48], MathV360K [49], NuminaMath [50], and DART-Math [51], with problems ranging from elementary school arithmetic to advanced geometry and statistics. The full list of data sources and sample counts used for training is provided in Appendix A.

Once collected, all data undergo rigorous cleaning and format standardization. Visual and textual content are paired explicitly, and reasoning traces are reformatted to follow a structured CoT schema, with each logical step clearly marked using structured tags such as <step></step>, and the final conclusion labeled with <answer></answer>. To enhance quality and clarity, we leveraged a strong instruction-tuned language model (i.e., Qwen2.5-72B-Instruct [2]) to parse original solutions and restructure them into coherent, modular steps. This structured representation not only enhances model learnability but also lays the foundation for generating step-level reward labels in the next stage. The prompt used for solution restructuring is provided in Appendix B.

With this cleaned and annotated corpus of over 5 million examples, we fine-tuned a strong open-source multimodal model, InternVL2.5-8B [16], using supervised learning. This ensures that the model learns to produce logically sound and well-structured outputs that conform to the CoT reasoning pattern.

The resulting model not only delivers high-quality reasoning trajectories for downstream data annotation but also supports inference-time use cases such as BoN selection with reward-based reranking. Its strong generative ability and logical consistency form the cornerstone of our scalable process supervision framework.

### 3.3 Process supervision data generation

To enable fine-grained supervision for step-level reasoning, we adopt an automated process annotation pipeline based on the OmegaPRM [41] framework. OmegaPRM introduces a MCTS-based mechanism for efficiently identifying and labeling intermediate reasoning steps with confidence estimates. Although originally developed for textual mathematical reasoning, we adapt and extend this framework to handle multimodal inputs.

Our process begins with collecting a curated dataset of 10,000 multimodal math problems from realworld named MM-K12—consisting of 5,000 fill-in-the-blank and 5,000 multiple-choice questions. These problems span a range of curriculum topics from elementary to high school and serve as seed instances for process supervision generation. All examples in MM-K12 are carefully filtered to ensure that each question includes meaningful visual input and has a unique, verifiable answer, making them well-suited for structured reasoning and reward modeling. In addition, MM-K12 also provides an independent test set of 500 problems, constructed under the same criteria, which we use to evaluate in-distribution performance later. For each problem, the policy model produces multiple candidate solutions following the CoT paradigm, and these reasoning paths form the raw material for subsequent reward annotation.

To evaluate the correctness of each intermediate step, we follow OmegaPRM’s hierarchical rollout and search protocol. Specifically, we generate multiple completions (rollouts) from partial prefixes and estimate the correctness of a given step based on whether its downstream completions reach the correct final answer. By applying binary search, the algorithm efficiently pinpoints the earliest step at which the reasoning begins to deviate. These supervision signals are then aggregated within a structured state-action tree, which records the Monte Carlo (MC) estimates and other statistics at each reasoning state. In our implementation, we maintain the full multimodal context—including both textual and visual components—throughout the tree construction and search process.

Importantly, our adaptation retains the efficiency of OmegaPRM’s divide-and-conquer search while enabling reward supervision for reasoning steps that are conditioned on complex visual stimuli. Through this pipeline, we generate over 700,000 step-level annotations from only 10k seed questions, without requiring manual labeling. The resulting dataset provides dense, high-quality process supervision aligned with real-world multimodal reasoning.

### 3.4 Process reward model training

With large-scale step-level supervision in place, we proceed to train a PRM that can assess the quality of reasoning steps given a multimodal context. The PRM is designed to serve as a fine-grained critic, assigning a reward score to each intermediate step conditioned on its preceding reasoning context that enables both test-time scaling and potential RL applications.

### 3.4.1 Labeling strategy

A central design decision in PRM training lies in how to formulate supervision signals from the MC estimations [41, 32]. Rather than adopting hard binary label (e.g., yˆ = 1[MC(s) > τ]), we use soft label, directly taking the MC scores as continuous supervision targets.

This choice is motivated by the observation that the MC score reflects more than the correctness of an intermediate step. It also encodes factors such as problem difficulty, step criticality, and distributional uncertainty in the policy model’s rollouts. For instance, a reasoning step within a highly ambiguous or visually complex problem may yield lower MC scores even if the logic is fundamentally sound. In such cases, hard-thresholding may misrepresent the step’s quality, introducing noise into training. By

contrast, soft labels preserve the probabilistic nuance and enable smoother learning dynamics, more details will be discussed in section 5.3

Formally, for each reasoning step xt in a path x = [x1,x2,...,xT], we assign a supervision target yˆt = MC(x<t) ∈ [0,1], where MC(x<t) denotes the estimated probability that a correct final answer can be reached from this partial path.

### 3.4.2 Model design and training objective

To model the prediction task, we treat the PRM as a classifier operating on each reasoning step. Given a multimodal input q and a generated reasoning trace [x1,x2,...,xT], we interleave a special marker token, denoted σ, after each step, producing an input sequence of the form:

#### [q,x1,σ,x2,σ,...,xT,σ].

In our implementation, σ is instantiated as the token <prm>. At each occurrence of σ, the model is tasked with producing a scalar confidence score indicating the likelihood that the immediately

preceding step is logically correct. Formally, let zYes(i) and zNo(i) denote the unnormalized logits for binary labels “Yes” (correct) and “No” (incorrect) at the i-th occurrence of σ. The model’s predicted probability is computed via softmax:

exp(zYes(i) ) exp(zYes(i) ) + exp(zNo(i))

p(i) =

.

The training objective is to minimize the cross-entropy loss between the predicted scores p(i) and the soft labels yˆ(i), across all scoring points:

T

y ˆ(i) · log p(i) + (1 − yˆ(i)) · log(1 − p(i)) .

LPRM = −

i=1

This formulation guides the model to make fine-grained assessments of reasoning steps, assigning higher confidence to those with stronger evidence of correctness.

## 4 Experiments

### 4.1 Experiments setup

To validate the effectiveness of our proposed process reward modeling framework, we conduct a series of experiments, carefully configured to ensure fair, scalable, and reproducible results.

Policy model construction. Our policy model(i.e., MM-Policy) is initialized from the multimodal backbone InternVL 2.5-8B and fine-tuned using approximately 4 million cleaned, structured math problems. The model is trained for 1 epoch with a batch size of 128 and a learning rate of 4e-5, updating only the language module while keeping the vision encoder frozen.

Process supervision data generation. We adapt the OmegaPRM[41] pipeline for multimodal reasoning and apply it to MM-K12 (10k samples). Using MCTS-based structured rollouts, we generate approximately 747,000 step-level annotations. The sampling parameters are tuned for balance between diversity and efficiency: temperature = 1.0, topk = 50, topp = 0.9, exploration coefficient cpuct = 0.125, and up to 200 search steps or 1,000 total rollouts per problem.

Process reward model training. We initialize the PRM from the fine-tuned policy model and train it for 1 epochs with a batch size of 512 and a learning rate of 4e-6.

### 4.2 Evaluation strategies and benchmarks

To assess the effectiveness of the proposed MM-PRM in improving reasoning quality, we adopt the BoN evaluation protocol. For each test problem, the policy model generates N = 16 candidate reasoning paths independently. The PRM then scores each path step-by-step, producing a sequence of floating-point values representing the predicted quality of each intermediate step, the path with the highest score is selected as the final answer.

Since PRM outputs a vector of step-wise confidence scores for each candidate path, a crucial component of our evaluation is the aggregation function [40] used to compress this vector into a scalar. We explore a diverse set of aggregation functions, including Min, Average, Max, SumLogPr (i.e., sum of log-probabilities), SumLogOdds (i.e., sum of log-odds), and MeanOdds (i.e., mean odds), each function captures different aspects of path quality. Additionally, a Random baseline is used for comparison, where the final answer is randomly sampled from the same set of 16 candidates. Formal definitions of all aggregation functions are provided in Appendix C.

We evaluate performance using answer accuracy, defined as the proportion of final selected answers that match the ground truth. This metric directly reflects MM-PRM’s utility in guiding the selection of correct reasoning paths. To comprehensively evaluate our model’s performance and generalization, we conduct experiments on a range of multimodal math benchmarks, including MMK12 (test set), OlympiadBench (OE_MM_maths_en_COMP) [52], MathVista(testmini) [53], MathVerse(testmini) [54], and MathVision(test) [55]. The MM-K12 test set serves as an in-distribution evaluation. For out-of-distribution assessment, we use the OE_MM_maths_en_COMP split of OlympiadBench, which contains open-ended multimodal questions from international math competitions, closely related in format to MM-K12 but significantly harder. To further test generalization, we include MathVista, which covers a wide range of visual mathematical tasks; MathVerse, which emphasizes understanding structured visual content; and MathVision, which targets abstract visual reasoning. These benchmarks provide a diverse and rigorous setting to measure both performance and generalization of our process reward modeling framework.

### 4.3 Quantitative results

Table 1: Performance improvements across various benchmarks when applying the MM-PRM to different models.

### Model MM-K12 OlympiadBench MathVista MathVerse MathVision

MM-Policy 33.92 15.41 62.93 42.99 21.74 +MM-PRM 42.80 24.00 67.60 46.27 27.11

- +8.88 +8.59 +4.67 +3.28 +5.37 InternVL2.5-8B 27.01 5.23 56.43 36.26 10.04

- +MM-PRM 37.80 15.33 63.50 42.56 19.41

+10.79 +10.10 +7.07 +6.30 +9.37 InternVL2.5-26B 28.01 14.46 60.02 37.83 20.76

- +MM-PRM 38.00 24.67 64.50 44.19 25.63

- +9.99 +10.21 +4.25 +6.36 +4.87

InternVL2.5-38B 40.34 29.57 68.32 47.94 29.70 +MM-PRM 52.40 32.67 71.10 52.61 32.99

### +12.06 +3.10 +2.78 +4.67 +3.29

InternVL2.5-78B 42.24 30.98 69.48 50.18 31.50 +MM-PRM 48.80 34.67 73.20 54.47 33.26

### +6.56 +3.69 +3.72 +4.29 +1.76

We evaluate the effectiveness of MM-PRM by applying it to a range of policy models and testing its impact across multiple multimodal math benchmarks.

Across all models, MM-PRM yields substantial performance improvements. For example, when applied to MM-Policy on the MM-K12 test set, accuracy improves from 33.92% to 42.80%, and similar gains are observed with InternVL2.5-8B, where performance rises from 27.01% to 37.80%. These results confirm that MM-PRM is highly effective at identifying high-quality reasoning paths. Beyond in-domain settings, we observe that MM-PRM also generalizes well to larger models and more challenging datasets. As shown in Table 1, applying MM-PRM to InternVL2.5-78B improves accuracy on OlympiadBench from 30.98% to 34.67%, and on MathVerse from 50.18% to 54.47%. Despite being trained only on the small MM-K12 seed dataset and with a fixed policy model(i.e., MMPolicy), MM-PRM consistently enhances reasoning accuracy across diverse benchmarks and models. This demonstrates the potential of scalable step-level reward modeling to improve mathematical

[Figure 2]

[Figure 3]

Question Solution

[Figure 4]

|[Figure 5]<br><br>( = 0.83) Identify the relationship between the angles formed by the parallel lines and the transversal. Since AB CD and ∠A = 118 , We can use the property of parallel lines and transversals to determine the measure of ∠ACD. The angles on a straight line sum up to 180. So: ∠ACD = 180 − ∠A = 180 - 118 = 62<br><br>|
|---|
| |
|( = 0.68) Use the angle bisector property. Since CE bisect ∠ACD, it divides ∠ACD into two equal angles. Therefore: ∠ACE =<br><br>∠ACD 2 = 622 = 31<br><br>[Figure 6]|

֯

֯ ֯ ֯ ֯ ֯

֯

֯

|( = 0.02) Determine ∠AEC using the exterior angle theorem. In △AEC, ∠AEC is an exterior angle and is equal to the sum of the two non-adjacent interior angles ∠A and ∠ACE . However, since ∠AEC is directly opposite ∠ACE and ∠A is the exterior angle, we can use the fact that the exterior angle is the sum of the two non-adjacent interior angles: ∠AEC = ∠A − ∠ACE = 118 - 31 = 87<br><br>[Figure 7]|
|---|

|As shown in the figure, it is known that AB CD,CE bisect ∠ACD, and intersects AB at E, ∠A = 118. Then ∠AEC = ?<br><br>|
|---|

֯ ֯ ֯

|( = 0.01) Answer: The Final Answer is 87<br><br>[Figure 8]|
|---|

֯

֯

##### Figure 2: Qualitative example of MM-PRM accurately identifying error steps in multimodal reasoning process.

reasoning in a model-agnostic and data-efficient manner. Detailed evaluation results across all aggregation functions are provided in Appendix D.

- 4.4 Qualitative results

To further illustrate the behavior of MM-PRM, we present a qualitative example involving a geometry problem with parallel lines and an angle bisector. As shown in Figure 2, the policy model generates a four-step reasoning path, which ultimately leads to an incorrect final answer.

The PRM assigns high scores to the first two steps (0.83 and 0.68), indicating that they are logically sound. In contrast, Step-3 receives a very low score (0.02), signaling that the model has identified a significant reasoning error at this point. This flawed step leads directly to an incorrect conclusion in Step-4.

This example demonstrates that MM-PRM is capable of detecting localized logical errors within a reasoning chain, such fine-grained judgment is crucial in selecting high-quality reasoning trajectories and filtering out those with subtle but critical flaws.

- 5 Discussion

[Figure 9]

##### Figure 3: (a) Effect of the number of candidate reasoning paths on answer accuracy under Best-of-N inference. Increasing the number of candidates allows the PRM to select higher-quality reasoning trajectories. (b) Effect of learning rate on PRM performance. Small learning rates yield better accuracy, with performance peaking at 4e-6

### 5.1 Candidate path’s impact on PRM performance

Since the PRM operates purely as a selector in the BoN framework, its performance is inherently bounded by the diversity and quality of candidate reasoning paths produced by the policy model. In other words, PRM cannot improve a flawed generation in BoN—it can only choose among the available options. Therefore, the number of reasoning paths generated per problem directly affects its potential to identify a correct and coherent solution.

To study this effect, we vary the number of generated reasoning paths N ∈ {2,22,23,24} and measure the corresponding answer accuracy under the aggregation strategy MeanOdds. As shown in Figure 3(a), increasing N consistently improves MM-PRM’s performance across both test sets. On the MM-K12 test set, accuracy improves from 38.6% at N = 2 to 42.8% at N = 16, with gains tapering off beyond N = 8. In contrast, on OlympiadBench, accuracy increases more steadily—from 18.4% to 24.0%—as N grows. This suggests that for harder, more diverse tasks, having a larger pool of reasoning paths is critical for PRM to identify valid solutions.

### 5.2 Learning rate

As noted in [39], finetuning a PRM shifts the language models’ objective from generation to discrimination, making learning rate a critical factor. Smaller learning rates are often preferred to maintain stability and preserve pretrained knowledge.

We evaluate MM-PRM trained under different learning rates on the MM-K12 test set using MeanOdds aggregator. As shown in Figure 3(b), performance peaks at 4e-6—about one-tenth the learning rate typically used in supervised fine-tuning—then drops sharply at higher values. This confirms that a moderate, conservative learning rate leads to better training, while overly large values degrade accuracy.

##### 5.3 Soft label vs. Hard label Table 2: The performance comparisons of soft label vs. hard label.

Labeling Strategy Min Average Max SumLogPr SumLogOdds MeanOdds

Soft Label 37.4 43.0 43.4 42.0 43.2 42.8 Hard Label 36.8 34.4 35.6 36.0 33.8 37.0

As discussed in section 3.4.1, we adopt soft label—i.e., real-valued MC scores—as supervision for step-level reward modeling. Unlike hard label, soft label retain uncertainty and allow the model to learn more nuanced representations of reasoning quality.

To assess this design choice, we compare soft label with hard label thresholding, where steps with MC > 0 are treated as correct, and others as incorrect, following the protocol in [41, 56].

As shown in Table 2, soft-label training consistently outperforms hard-label training across all aggregation strategies. For instance, under the Average aggregator, soft labels yield 43% accuracy on MM-K12 test set, compared to 34.4% with hard labels. Similar improvements are observed with SumLogOdds (43.2% vs. 33.8%) and MeanOdds (42.8% vs. 37.0%).

## 6 Conclusion

In this work, we introduce MM-PRM, a process reward model built upon a scalable framework for multimodal mathematical process reward modeling that enables step-level supervision without human annotation. By leveraging a multimodal policy model and an MCTS-based data generation pipeline, we construct over 700k process-level labels from only 10k K-12 math problems in our MM-K12 dataset. Our trained PRM significantly improves reasoning accuracy in BoN inference across both standard and challenging benchmarks, and demonstrates strong generalization to new datasets and models. Extensive analysis further confirms the importance of soft labeling, conservative learning rates, and sufficient path diversity. MM-PRM highlights the value of process supervision for enhancing multimodal mathematical problem-solving.

## References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.
- [3] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [4] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [5] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [6] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad AlDahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- [7] InternLM Team. Internlm: A multilingual language model with progressively enhanced capabilities. https://github.com/InternLM/InternLM-techreport, 2023.
- [8] Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, et al. Internlm2 technical report. arXiv preprint arXiv:2403.17297, 2024.
- [9] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [10] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.
- [11] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024.
- [12] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.
- [13] Weiyun Wang, Zhe Chen, Wenhai Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Jinguo Zhu, Xizhou Zhu, Lewei Lu, Yu Qiao, and Jifeng Dai. Enhancing the reasoning ability of multimodal large language models via mixed preference optimization. arXiv preprint arXiv:2411.10442, 2024.
- [14] Zhangwei Gao, Zhe Chen, Erfei Cui, Yiming Ren, Weiyun Wang, Jinguo Zhu, Hao Tian, Shenglong Ye, Junjun He, Xizhou Zhu, et al. Mini-internvl: a flexible-transfer pocket multi-modal model with 5% parameters and 90% performance. Visual Intelligence, 2(1):1–17, 2024.
- [15] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. Science China Information Sciences, 67(12):220101, 2024.
- [16] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024.
- [17] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023.
- [18] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.
- [19] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.
- [20] Fei Yu, Anningzhe Gao, and Benyou Wang. Ovm, outcome-supervised value models for planning in mathematical reasoning. arXiv preprint arXiv:2311.09724, 2023.

- [21] Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process-and outcome-based feedback. arXiv preprint arXiv:2211.14275, 2022.
- [22] Yuhang Zang, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Ziyu Liu, Shengyuan Ding, Shenxi Wu, Yubo Ma, Haodong Duan, Wenwei Zhang, et al. Internlm-xcomposer2. 5-reward: A simple yet effective multi-modal reward model. arXiv preprint arXiv:2501.12368, 2025.
- [23] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.
- [24] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [25] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [26] Xin Lai, Zhuotao Tian, Yukang Chen, Senqiao Yang, Xiangru Peng, and Jiaya Jia. Step-dpo: Step-wise preference optimization for long-chain reasoning of llms. arXiv preprint arXiv:2406.18629, 2024.
- [27] Weiyun Wang, Zhe Chen, Wenhai Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Jinguo Zhu, Xizhou Zhu, Lewei Lu, Yu Qiao, et al. Enhancing the reasoning ability of multimodal large language models via mixed preference optimization. arXiv preprint arXiv:2411.10442, 2024.
- [28] Richard Yuanzhe Pang, Weizhe Yuan, He He, Kyunghyun Cho, Sainbayar Sukhbaatar, and Jason Weston. Iterative reasoning preference optimization. Advances in Neural Information Processing Systems, 37:116617–116637, 2024.
- [29] Hanze Dong, Wei Xiong, Bo Pang, Haoxiang Wang, Han Zhao, Yingbo Zhou, Nan Jiang, Doyen Sahoo, Caiming Xiong, and Tong Zhang. Rlhf workflow: From reward modeling to online rlhf. arXiv preprint arXiv:2405.07863, 2024.
- [30] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.
- [31] Peiyi Wang, Lei Li, Zhihong Shao, RX Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. arXiv preprint arXiv:2312.08935, 2023.
- [32] Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. The lessons of developing process reward models in mathematical reasoning. arXiv preprint arXiv:2501.07301, 2025.
- [33] Xidong Feng, Ziyu Wan, Muning Wen, Stephen Marcus McAleer, Ying Wen, Weinan Zhang, and Jun Wang. Alphazero-like tree-search can guide large language model decoding and training. arXiv preprint arXiv:2309.17179, 2023.
- [34] Jikun Kang, Xin Zhe Li, Xi Chen, Amirreza Kazemi, Qianyi Sun, Boxing Chen, Dong Li, Xu He, Quan He, Feng Wen, et al. Mindstar: Enhancing math reasoning in pre-trained llms at inference time. arXiv preprint arXiv:2405.16265, 2024.
- [35] Qianli Ma, Haotian Zhou, Tingkai Liu, Jianbo Yuan, Pengfei Liu, Yang You, and Hongxia Yang. Let’s reward step by step: Step-level reward model as the navigators for reasoning. arXiv preprint arXiv:2310.10080, 2023.
- [36] Ye Tian, Baolin Peng, Linfeng Song, Lifeng Jin, Dian Yu, Lei Han, Haitao Mi, and Dong Yu. Toward self-improvement of llms via imagination, searching, and criticizing. Advances in Neural Information Processing Systems, 37:52723–52748, 2024.
- [37] Dan Zhang, Sining Zhoubian, Ziniu Hu, Yisong Yue, Yuxiao Dong, and Jie Tang. Rest-mcts*: Llm self-training via process reward guided tree search. Advances in Neural Information Processing Systems, 37:64735–64772, 2024.
- [38] Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, and Weizhu Chen. Making large language models better reasoners with step-aware verifier. arXiv preprint arXiv:2206.02336, 2022.
- [39] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.
- [40] Zihan Wang, Yunxuan Li, Yuexin Wu, Liangchen Luo, Le Hou, Hongkun Yu, and Jingbo Shang. Multi-step problem solving through a verifier: An empirical analysis on model-induced process supervision. arXiv preprint arXiv:2402.02658, 2024.

- [41] Liangchen Luo, Yinxiao Liu, Rosanne Liu, Samrat Phatale, Meiqi Guo, Harsh Lara, Yunxuan Li, Lei Shu, Yun Zhu, Lei Meng, et al. Improve mathematical reasoning in language models by automated process supervision. arXiv preprint arXiv:2406.06592, 2024.
- [42] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.
- [43] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.
- [44] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.
- [45] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.
- [46] Ruilin Luo, Zhuofan Zheng, Yifan Wang, Yiyao Yu, Xinzhe Ni, Zicheng Lin, Jin Zeng, and Yujiu Yang. Ursa: Understanding and verifying chain-of-thought reasoning in multimodal mathematics. arXiv preprint arXiv:2501.04686, 2025.
- [47] Linger Deng, Yuliang Liu, Bohan Li, Dongliang Luo, Liang Wu, Chengquan Zhang, Pengyuan Lyu, Ziyang Zhang, Gang Zhang, Errui Ding, et al. R-cot: Reverse chain-of-thought problem generation for geometric reasoning in large multimodal models. arXiv preprint arXiv:2410.17885, 2024.
- [48] Renrui Zhang, Xinyu Wei, Dongzhi Jiang, Ziyu Guo, Shicheng Li, Yichi Zhang, Chengzhuo Tong, Jiaming Liu, Aojun Zhou, Bin Wei, et al. Mavis: Mathematical visual instruction tuning with an automatic data engine. arXiv preprint arXiv:2407.08739, 2024.
- [49] Wenhao Shi, Zhiqiang Hu, Yi Bin, Junhua Liu, Yang Yang, See-Kiong Ng, Lidong Bing, and Roy Ka-Wei Lee. Math-llava: Bootstrapping mathematical reasoning for multimodal large language models. arXiv preprint arXiv:2406.17294, 2024.
- [50] Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q Jiang, Ziju Shen, et al. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. Hugging Face repository, 13:9, 2024.
- [51] Yuxuan Tong, Xiwen Zhang, Rui Wang, Ruidong Wu, and Junxian He. Dart-math: Difficulty-aware rejection tuning for mathematical problem-solving. Advances in Neural Information Processing Systems, 37:7821–7846, 2024.
- [52] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008, 2024.
- [53] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.
- [54] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pages 169–186. Springer, 2024.
- [55] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 37:95095–95169, 2024.
- [56] Jun Wang, Meng Fang, Ziyu Wan, Muning Wen, Jiachen Zhu, Anjie Liu, Ziqin Gong, Yan Song, Lei Chen, Lionel M Ni, et al. Openr: An open source framework for advanced reasoning with large language models. arXiv preprint arXiv:2410.09671, 2024.
- [57] Shuai Peng, Di Fu, Liangcai Gao, Xiuqin Zhong, Hongguang Fu, and Zhi Tang. Multimath: Bridging visual and mathematical reasoning for large language models. arXiv preprint arXiv:2409.00147, 2024.
- [58] Lei Li, Yuqi Wang, Runxin Xu, Peiyi Wang, Xiachong Feng, Lingpeng Kong, and Qi Liu. Multimodal arxiv: A dataset for improving scientific comprehension of large vision-language models. arXiv preprint arXiv:2403.00231, 2024.
- [59] Wenwen Zhuang, Xin Huang, Xiantao Zhang, and Jin Zeng. Math-puma: Progressive upward multimodal alignment to enhance mathematical reasoning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 26183–26191, 2025.

- [60] Wenqi Zhang, Zhenglin Cheng, Yuanyu He, Mengna Wang, Yongliang Shen, Zeqi Tan, Guiyang Hou, Mingqian He, Yanna Ma, Weiming Lu, et al. Multimodal self-instruct: Synthetic abstract image and visual reasoning instruction using language model. arXiv preprint arXiv:2407.07053, 2024.
- [61] Erfei Cui, Yinan He, Zheng Ma, Zhe Chen, Hao Tian, Weiyun Wang, Kunchang Li, Yi Wang, Wenhai Wang, Xizhou Zhu, Lewei Lu, Tong Lu, Yali Wang, Limin Wang, Yu Qiao, and Jifeng Dai. Sharegpt-4o: Comprehensive multimodal annotations with gpt-4o, 2024.
- [62] Cyrile Delestre Tom Agonnoude, 2024.
- [63] Aida Amini, Saadia Gabriel, Peter Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. Mathqa: Towards interpretable math word problem solving with operation-based formalisms. arXiv preprint arXiv:1905.13319, 2019.
- [64] PawanKrd. math-gpt-4o-200k. Hugging Face repository, 2024.
- [65] Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mammoth: Building math generalist models through hybrid instruction tuning. arXiv preprint arXiv:2309.05653, 2023.

Appendix

- A Policy Model training data

- Table 3 summarizes the datasets and sample counts used to train the policy model. Table 3: Data sources and sample counts used for policy model training.

Dataset Sample Count R-CoT [47] 201,781 MAVIS [48] 29,551 multimath-300k [57] 224,747 Multimodal ArXiv [58] 99,772 MathV360K [49] 338,721 Math-PUMA [59] 1,111,939 Multi-modal-Self-instruct [60] 64,807 MMPR [13] 1,055,796 ShareGPT-4o [61] 57,289 table-vqa [62] 80,054 MathQA [63] 27,923 GSM8K [19] 7,413 DART-Math [51] 574,305 math-gpt-4o-200k [64] 196,032 NuminaMath-CoT [50] 807,883 MathInstruct [65] 257,755 Total 5,135,768

- B Prompt template for data cleaning

The prompt template used for data cleaning is shown below. Before generating logical steps and the final answer, the language model is first instructed to explicitly identify the key knowledge points involved in the problem, along with brief explanations. This two-stage prompting strategy enhances the model’s understanding of the question prior to solution restructuring.

Using the information provided, identify and summarize the key knowledge points required to solve the problem and rewrite the original answer with a detailed reasoning process based on the input answer. Provide a clear explanation of each knowledge point by stating its name followed by a colon (:), and then presenting the detailed Explanation on the same line. When explaining the knowledge point, DO NOT reference or describe the original question, answer, or answer details. Focus solely on explaining the knowledge points. Besides, rewrite the original answer to include detailed reasoning and answer. Do not change the final answer and always refer to the input answer.

Output Format: In order to answer this question, we first need to have the following prior knowledge:

- {{Substitute with name of knowledge point 1}}: {{Substitute with

- Explanation of knowledge point 1}},

{{Substitute with name of knowledge point 2}}: {{Substitute with

- Explanation of knowledge point 2}},

... We answer this based on prior knowledge as follows: Solution: Refined answer with detailed reasoning. Use Step 1, Step 2 to divide the steps. Remember do not change the final answer and always refer to the input answer. Answer: The Final Answer is {{Substitute with final answer}}.

Input Information: Question: {question}

-------------------------------Answer: {answer}

--------------------------------

## C Aggregation function definitions

In the BoN inference setup, each reasoning path is assigned a vector of step-level scores predicted by the PRM. To compare and rank these paths, we apply an aggregation function to compress each score vector into a single scalar value, following the design considerations discussed in [40]. The following aggregation strategies are used in our experiments:

- • Min: Computes the minimum of all step scores:

score(rj) = min{p(1j),p(2j),...,p(Tj)

j

}

- • Max: Computes the maximum of all step scores:

score(rj) = max{p(1j),p(2j),...,p(Tj)

j

}

- • Average: Computes the arithmetic mean of all step scores:

score(rj) =

1 Tj

Tj

i=1

p(ij)

- • SumLogPr: Computes the sum of log-probabilities of all step scores:

score(rj) =

Tj

i=1

log p(ij) = log

Tj

i=1

p(ij)

- • SumLogOdds: Computes the sum of log-odds of all step scores:

score(rj) =

Tj

i=1

log

p(ij) 1 − p(ij)

- • MeanOdds: Computes the mean of odds-transformed step scores:

1 Tj

score(rj) =

Tj

p(ij) 1 − p(ij)

i=1

Here, rj denotes the j-th reasoning path consisting of Tj steps, and p(ij) ∈ (0,1) is the predicted correctness probability of the i-th step in that path.

## D Full evaluation results

### D.1 MM-Policy

Table 4: Performance comparison of various aggregators on various benchmarks for MM-Policy. Top performer is in bold.

Benchmark Random Min Average Max SumLogPr SumLogOdds MeanOdds

MM-K12 33.9 37.4 43.0 43.4 42.0 43.2 42.8 OlympiadBench 15.4 23.3 20.0 24.7 22.7 24.0 24.0 MathVista 62.9 66.0 67.2 67.7 66.4 67.7 67.6 MathVerse 43.0 46.0 48.0 46.1 47.3 48.0 46.3 MathVision 21.7 25.8 26.7 26.9 25.4 26.3 27.1

### D.2 InternVL-8B

Table 5: Performance comparison of various aggregators on various benchmarks for InternVL-8B. Top performer is in bold.

Benchmark Random Min Average Max SumLogPr SumLogOdds MeanOdds

MM-K12 27.0 34.8 36.0 35.8 31.2 34.2 37.8 OlympiadBench 5.2 2.7 15.3 16.7 2.0 1.3 15.3 MathVista 56.4 63.4 62.8 62.6 61.0 62.0 63.5 MathVerse 36.3 41.8 42.8 41.1 42.0 42.8 42.6 MathVision 10.0 6.1 18.4 20.6 3.8 4.8 19.4

### D.3 InternVL-26B

Table 6: Performance comparison of various aggregators on various benchmarks for InternVL-26B. Top performer is in bold.

Benchmark Random Min Average Max SumLogPr SumLogOdds MeanOdds

MM-K12 28.0 38.0 37.2 35.4 32.0 33.0 38.0 OlympiadBench 14.5 16.0 23.3 22.0 16.7 16.7 24.7 MathVista 60.0 63.9 64.6 64.4 62.8 64.2 64.5 MathVerse 37.8 42.0 44.2 44.3 40.1 41.5 44.2 MathVision 20.8 23.3 24.1 24.9 20.6 22.0 25.6

### D.4 InternVL-38B

- Table 7: Performance comparison of various aggregators on various benchmarks for InternVL-38B. Top performer is in bold.

Benchmark Random Min Average Max SumLogPr SumLogOdds MeanOdds

MM-K12 40.3 46.8 51.0 49.8 42.8 46.6 52.4 OlympiadBench 29.6 29.3 34.0 34.7 28.0 30.0 32.7 MathVista 68.3 70.5 71.1 70.5 69.2 70.1 71.1 MathVerse 47.9 50.4 51.9 51.1 51.1 52.7 52.6 MathVision 29.7 29.1 32.6 33.6 30.6 31.3 33.0

### D.5 InternVL-78B

- Table 8: Performance comparison of various aggregators on various benchmarks for InternVL-78B. Top performer is in bold.

Benchmark Random Min Average Max SumLogPr SumLogOdds MeanOdds

MM-K12 42.2 46.2 47.2 47.2 45.0 46.8 48.8 OlympiadBench 31.0 33.3 35.3 34.7 33.3 34.7 34.7 MathVista 69.5 71.7 73.4 72.4 73.9 73.6 73.2 MathVerse 50.2 53.4 55.0 54.2 54.3 55.1 54.5 MathVision 31.5 30.1 33.5 33.6 31.8 32.4 33.3

## E Limitations

While our proposed framework demonstrates strong performance and generalization across multiple benchmarks, it also has several limitations: (1) Due to computational constraints, we conduct training only on the InternVL series with 8B parameters, without exploring larger models or architectures from other model families. This restricts our ability to fully assess how PRM training behavior scales with model size or generalizes across different backbones. (2) The seed data used for process supervision generation is limited in diversity, as it consists solely of K-12 math problems. As a result, the PRM may be less exposed to advanced mathematical domains or visual formats beyond the scope of standard educational settings. We leave broader model coverage and more diverse seed data construction as promising directions for future work.

