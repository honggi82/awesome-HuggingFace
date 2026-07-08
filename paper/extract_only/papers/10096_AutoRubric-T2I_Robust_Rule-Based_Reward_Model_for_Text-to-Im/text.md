# arXiv:2605.17602v2[cs.AI]20May2026

## AutoRubric-T2I: Robust Rule-Based Reward Model for Text-to-Image Alignment

Kuei-Chun Kao1 Daixuan Huo1 Yuanhao Ban1,2 Cho-Jui Hsieh1,2 1University of California, Los Angeles 2Arena johnson0213@g.ucla.edu, chohsieh@cs.ucla.edu

Project Page: AutoRubric-T2I Code

### Abstract

Aligning Text-to-Image (T2I) generation models with human preferences increasingly relies on image reward models that score or rank generated images according to prompt alignment and perceptual quality. Existing reward models are commonly trained as Bradley-Terry (BT) preference models on large-scale human preference corpora, making them costly to train, difficult to adapt, and opaque in their evaluation criteria. Meanwhile, Vision-Language Model (VLM) judges can provide more fine-grained assessments through textual rubrics, but their manually designed or heuristically generated scoring rules may fail to reliably reflect human preferences. In this paper, we propose AutoRubric-T2I, the first rubric learning framework in T2I that automatically synthesizes and selects explicit rubrics for guiding VLM judges. AutoRubric-T2I first synthesizes reasoning traces from preference pairs into candidate rubrics, then uses a VLM judge to score paired images under each rubric, producing pairwise rubric-score differences for preference learning. To remove noisy and redundant rules, we further employ a ℓ1-Regularized Logistic Regression Refiner, which selects the Top-N most discriminative rubrics. Extensive evaluations show that AutoRubric-T2I produces high-quality, interpretable reward signals using less than 0.01% of the annotated preference data, substantially reducing the need for large-scale reward-model training. On image reward benchmarks such as MMRB2, AutoRubric-T2I outperforms strong reward model baselines. We further validate AutoRubric-T2I as an RL reward on downstream T2I tasks, including TIIF and UniGenBench++, where it improves generation quality over scalar reward models using the Flow-GRPO pipeline on diffusion models.

### 1 Introduction

Recent advancements in T2I generation have made human preference alignment a central objective. Image reward models provide a practical mechanism for this alignment by learning to predict human judgments over generated images. In the T2I setting, an image reward model typically takes a text prompt together with one or more generated images as input and outputs either scalar reward scores or pairwise preferences indicating which image better satisfies the prompt and human quality expectations. These models are widely used for candidate ranking, best-of-N selection, data filtering, reinforcement fine-tuning (RFT), and automatic evaluation of T2I systems [13, 18, 14].

Existing image reward models mainly fall into two categories. The first category consists of learned BT preference models trained on large-scale human preference corpora, such as ImageReward [32], PickScore [15], HPSv2 [29], and HPSv3 [20]. These models can capture real-world human preferences and are effective for global image ranking, but they require massive annotated datasets and expensive fine-tuning. Moreover, because they usually compress multiple evaluation dimensions into a single scalar score, they provide limited transparency and may overlook fine-grained visual errors such as incorrect object counts, missing attributes, distorted anatomy, or violated spatial relations [9, 16, 8]. The second category consists of VLM-based judges and question-answering-based evaluators, which evaluate images through textual prompts, visual questions, or rubrics [9, 16, 10, 26].

Preprint.

These judges can assess fine-grained visual correctness when properly instructed, but their criteria are typically manually specified or heuristically generated rather than learned from human preferences.

- As a result, their judgments may not reliably correlate with actual human preference; for example, prompted VLM judges can be up to 10% worse than learned BT reward models on HPS or PickScore preference datasets in Table 1.

These limitations motivate rubric-based reward modeling, which replaces implicit scalar rewards with multi-dimensional evaluation criteria [31, 24, 6, 38, 19]. Rubrics make the reward signal more interpretable by decomposing human preferences into explicit rules. Recent works have explored rubrics as reward signals for LLM alignment and post-training [31, 24, 6, 11, 33, 17], and emerging T2I work has begun to study rubric rewards for image generation [4]. However, existing rubric-based approaches often rely on manually designed or heuristically generated rubrics, leaving open the question of how to automatically derive, select, and refine rubrics that better align with human preferences.

To address this gap, we propose AutoRubric-T2I, the first rubric learning framework that automatically derives and refines an explicit rubric set for guiding off-the-shelf VLM judges in T2I reward modeling. Instead of fine-tuning a reward model, AutoRubric-T2I learns which rubrics are most predictive of human preferences and iteratively improves them through failure analysis. This design preserves the interpretability of rubric-based evaluation while avoiding the cost and opacity of training a dense scalar reward model.

To achieve this, we formulate automated rubric learning as a sparse logistic regression problem within an infinite-dimensional space. We then introduce an iterative block coordinate descent method that dynamically adds new coordinates to the working set and employs ℓ1-regularization to assign weights and prune redundant coordinates (rubrics). This formulation is related to sparse function approximation and coordinate-selection methods such as orthogonal matching pursuit and sparse random features [21, 12, 37]. To enhance efficiency, we integrate a hard-pair mining algorithm for rubric refinement, ensuring that only the most informative coordinates are prioritized during the learning process.

Our main contributions are as follows:

- • Sparse Rubric Learning for T2I Reward Modeling: We introduce AutoRubric-T2I, a framework that learns a compact, weighted set of natural-language rubrics from image preference data, enabling interpretable VLM-based reward modeling without fine-tuning.
- • Failure-Driven Rubric Refinement: We formulate rubric selection as an ℓ1-regularized logistic regression problem over VLM-scored rubric features and iteratively expand the rubric pool through curriculum-bucketed hard-pair mining.
- • Strong Preference Prediction and Downstream Alignment: AutoRubric-T2I achieves strong preference prediction on MMRB2 among open-source reward models and improves downstream RFT on T2I tasks such as TIIF and UniGenBench++ using Flow-GRPO.

### 2 Related Work

#### 2.1 Text-to-Image Preference Alignment and Reward Modeling

Aligning text-to-image (T2I) models with human preferences commonly relies on reward models trained from human preference data. Many image reward models are trained from pairwise comparisons, often with a Bradley-Terry style objective, but are deployed as pointwise scorers that assign a scalar reward to each prompt-image pair. Large-scale preference datasets and models such as PickScore [15] enabled automatic ranking of generated images according to human judgments. Subsequent reward models, including ImageReward [32], HPSv2 [29], and HPSv3 [20], further improved visual preference modeling by capturing visual quality, aesthetics, and text-image correspondence. Recent work also explores alternative reward formulations, such as generative reward modeling in RewardDance [30] and UnifiedReward [26].

Despite their effectiveness, scalar reward models compress multi-dimensional human preferences into a single implicit score. This makes the learned reward difficult to interpret and vulnerable to reward hacking: a T2I policy may exploit superficial visual features such as brightness, contrast, saturation, or aesthetic style while ignoring prompt-specific semantic constraints [2, 8]. AutoRubric-T2I addresses this limitation by replacing an opaque scalar reward with an explicit weighted set of natural-language rubrics, allowing the reward signal to remain interpretable.

#### 2.2 Automated Rubric Generation

Rubric-based evaluation decomposes open-ended human preferences into explicit criteria, improving interpretability over monolithic scalar rewards. Recent work has explored automatic rubric generation to reduce the need for manually written evaluation rules. OpenRubrics [19] derives rubrics by contrasting preferred and rejected responses, while AutoRule [24] uses chain-of-thought prompting over preference examples to extract candidate rules. Other methods improve rubric coverage or specificity through refinement, decomposition, or differentiation, such as Chasing the Tail [38], RubricHub [17], Auto-Rubric [31], and RRD [23].

Our approach builds upon these insights but introduces a rubric learning framework for image reward modeling. To the best of our knowledge, AutoRubric-T2I is the first method to learn a sparse, weighted, global set of natural-language rubrics for T2I reward modeling directly from image preference data. Instead of relying only on LLM prompting heuristics for rubric refinement, we pair curriculum-based hard-pair mining with an ℓ1-regularized logistic regression refiner. This statistically prunes the rubric space, selects the Top-N most discriminative rubrics, and assigns learned weights that align the final rubric reward with human preferences.

#### 2.3 Reinforcement Learning from Rubric-Based Rewards

Reinforcement Learning with Verifiable Rewards has shown strong results in domains with objective correctness signals, such as mathematics and code generation [7, 28]. For more open-ended generation, recent work has proposed using rubrics as intermediate reward specifications. In language model alignment, Rubrics as Rewards [6] converts rubric-based feedback into scalar rewards for RL, while OnlineRubrics [22] updates evaluation criteria online to reduce criteria staleness.

In the T2I setting, prior works such as DDPO [1] and DanceGRPO [34] have demonstrated the effectiveness of RL for improving T2I models with scalar rewards. RubricRL [4] further applies rubric-based rewards to RFT by dynamically generating prompt-specific visual checklists during training. In contrast, AutoRubric-T2I focuses on learning a global rubric set offline from preference data. This distinction is important: RubricRL relies on per-prompt rubric construction during the RL loop, whereas our method learns a compact, reusable, and weighted rubric set before deployment. As a result, AutoRubric-T2I can serve as a training-free VLM-based reward model at inference time or as a fixed reward signal for downstream RFT.

### 3 Preliminaries

#### 3.1 Standard Reward Modeling

In standard Text-to-Image Reinforcement Learning from Human Feedback (RLHF), a scalar Reward Model (RM) rθ(x,y) is trained to predict human preference given a text prompt x and a generated image y. The objective is typically to minimize the Bradley-Terry ranking loss over a dataset of preference pairs (yw,yl):

w,yl)∼D [log σ (rθ(x,yw) − rθ(x,yl))], (1) where yw and yl denote the preferred and rejected images, respectively, and σ is the sigmoid function.

LRM(θ) = −E(x,y

#### 3.2 Reward Hacking in Text-to-Image Generation

Fine-tuning a T2I policy πϕ to maximize Ey∼π

ϕ(·|x)[rθ(x,y)] can improve reward-model alignment, but it can also induce reward hacking. Since standard image reward models compress semantic fidelity, object correctness, spatial layout, and perceptual quality into a single scalar, the learned reward may capture spurious shortcuts rather than true prompt satisfaction. In practice, we observe that standard RMs often over-emphasize aesthetic proxies, such as bright lighting, high contrast, sharp details, or human-centered compositions.

- Figure 1 shows an example after 500 steps of RFT. Although the prompt only asks for a conical chef hat hidden behind a spherical snowball, the HPSv3-optimized policy introduces an unnecessary human subject and still receives a high HPSv3 score. This suggests that the scalar reward is partially exploited through human-centered, visually appealing artifacts rather than by prompt satisfaction. In contrast, the policy optimized with AutoRubric-T2I preserves the intended objects and spatial relations. The rubric-level scores further reveal that the HPSv3-optimized image performs well on superficial visual quality but fails on prompt details and structure, illustrating how explicit rubrics can reduce reward hacking. We show the detail training dynamics in Appendix I.

[Figure 1]

- Figure 1: Reward hacking in scalar reward optimization. HPSv3 optimization attains a high scalar reward while violating prompt-specific constraints, whereas AutoRubric-T2I favors the rubric-aligned generation.

### 4 Methodology

In this section, we introduce AutoRubric-T2I. Section 4.1 formulates rubric learning as an infinitedimensional sparse logistic regression problem and motivates a working-set optimization strategy. Section 4.2 describes the practical implementation, including seed rubric generation, sparse rubric selection, hard-pair mining, and failure-driven rubric refinement.

#### 4.1 Formulation

In our framework, each rubric rj is parameterized by a natural language prompt. To evaluate a specific rubric rj on an image y conditioned on the input prompt x, we employ a VLM-as-a-judge (e.g., Gemini or the Qwen-3 series) to output a continuous scalar score s(rj,x,y) ∈ [0,1]. In practice, the score is the predicted probability of the yes token. Thus, s(rj,x,y) := Pθ yes | rj,x,y , where Pθ denotes the probability distribution of our VLM-based model.

Our objective is to identify a set of N natural language rubrics, R = {r1,r2,...,rN}, and a corresponding set of weights w ∈ RN, such that their weighted combination best explains the observed preference data. The final reward score for a given prompt-image pair (x,y) is defined as:

sR,w(x,y) :=

N j=1

wjs(rj,x,y).

To determine the optimal rubric-weight combination, we leverage a preference dataset Dtrain = {(x(i),ya(i),yb(i),z(i))}Mi=1 containing M human preference pairs. Here, x(i) is the text prompt, ya(i) and yb(i) are two generated images, and z(i) ∈ {1,−1} indicates the user preference (1 if ya is preferred). Notably, our framework requires only a small amount of data (e.g., M = 256). We seek the combination that minimizes the logistic loss:

M

log σ z(i) sR,w(x(i),ya(i)) − sR,w(x(i),yb(i)) , where σ denotes the sigmoid function.

min

R,w

i=1

While optimizing w is a standard linear logistic regression problem, learning the set R is inherently intractable. Since the space of possible natural-language rubrics is infinite, we let J := {1,2,...,∞} denote the indices of all possible rubrics. Selecting the top-N rubrics is equivalent to solving the optimization problem with an ℓ0 constraint, which we relax using an ℓ1 penalty:

 z(i)

 , (2)

M

wj∆s(ji)

λ∥w∥1 +

log σ

min

w

j∈J

i=1

where ∆s(ji) := s(rj,x(i),ya(i)) − s(rj,x(i),yb(i)) represents the score differential when applying rubric rj to the i-th training pair.

We solve this infinite-dimensional sparse recovery problem using a block coordinate descent method.

- At each iteration t, we generate a finite set of additional candidate rubrics (coordinates) using the current model’s failure cases from J, append them to the current working set Rt, and minimize Equation (2) with respect to the current working set of coordinates:

 z(i)

 , (3)

M

wj∆s(ji)

λ∥wRt∥1 +

min

log σ

wRt

j∈Rt

i=1

where wRt is the finite-dimensional sub-vector of w corresponding to indices in Rt. Postoptimization, we prune rubrics with zero weights to maintain a compact set.

This block coordinate descent approach has been widely used in sparse recovery problems; for instance, [37] demonstrated that such algorithms converge when the working set is augmented randomly. Furthermore, our approach draws inspiration from Orthogonal Matching Pursuit (OMP) [21, 12], which utilizes greedy strategies to select coordinates. In the following section, we instantiate this idea with a greedy strategy that prioritizes high-impact rubrics generated from hard failure pairs.

[Figure 2]

- Figure 2: Overview of AutoRubric-T2I. Our framework first constructs a seed rubric pool through diversityaware seed selection and rubric generation. It then iteratively scores training pairs, selects discriminative rubrics with sparse logistic regression, mines hard pairs, and proposes new rubrics to refine the final weighted rubric set.

#### 4.2 Detailed Procedure

We now describe the practical pipeline that instantiates the formulation above. Algorithm 1 and scores candidate rubrics with a VLM judge, solves the ℓ1-regularized problem in Eq. (3), evaluates the retained Top-N rubric set on a validation split, and expands the working set via new rubrics from curriculum-mined hard pairs. We provide the implementation details in the Appendix F.

- Figure 2 summarize the full procedure. Starting from a seed rubric set R0, each refinement round

#### 4.2.1 Seed Data Selection and Initial Rubric Generation

Before iterative refinement begins, we construct an initial working set of rubrics R0 from an informative seed data Dtrain. In our default setting, Dtrain contains 256 preference pairs.

Diversity-Aware Seed Data Selection. Naively sampling seed preference pairs may over-represent redundant prompts or visually trivial failures. Following FiFA [36], we use a proxy reward model to estimate the preference margin of each pair and cluster text prompts for semantic coverage. We select 256 seed pairs using a composite score favoring both high-margin preference signals and prompt-level diversity.

T2I-Adapted CoT Rubric Generation. Given the selected seed pairs, we generate the initial candidate rubrics using a VLM-based chain-of-thought prompting procedure adapted to text-to-image evaluation. For each seed pair, the VLM is asked to: (1) inspect the prompt and both images, (2) explain the visual differences that justify the human preference label, and (3) extract objective, deterministic rubric statements that could be reused across examples. The resulting statements are aggregated and deduplicated to form the initial working set R0.

- 4.2.2 Working-Set Rubric Scoring and Sparse Selection At refinement round t, we score all candidate rubrics in Rt on the training pairs, computing VLM

score differences ∆s(ji) = s(rj,x(i),ya(i)) − s(rj,x(i),yb(i)) as features for Eq. (3). We solve the ℓ1-regularized logistic regression over the current working set; the ℓ1 penalty assigns zero weights to redundant or weakly predictive rubrics. We retain the Top-N rubrics with the largest positive weights: Rtretained = {rj ∈ Rt : wj > 0}Top-N, whose weights wretainedt define the ensembled rubric reward. We use the liblinear solver with C = 1.0.

- 4.2.3 Curriculum-Bucketed Hard-Pair Mining

After obtaining the retained rubric set, we identify preference pairs that are incorrectly ranked by the current rubric reward. For a pair where ya is preferred over yb, the model misranks the pair if

rj∈Rtretained

wj∆s(ji) < 0.

These misranked examples reveal failure modes not yet captured by the current rubric set and serve as the source for generating new candidate rubrics.

Rather than sampling failures uniformly, we introduce a curriculum-bucketed hard-pair selector that partitions misranked pairs into three categories: (1) wrong-small margin pairs (below the 30th percentile of absolute margin), which involve subtle distinctions the current rubric set misses; (2) wrong-large margin pairs, indicating severe failures where the rubric set confidently contradicts human preference; and (3) high-reward wrong pairs, where both images receive high scores yet the ranking is incorrect, requiring finer-grained rubrics similar in spirit to [38]. Across refinement rounds, we shift the sampling ratio: early rounds emphasize large-margin errors to expose major missing dimensions, while later rounds focus on high-reward wrong cases to discover finer-grained rubrics. Pairs selected more than four times are excluded to avoid noisy or unlearnable examples.

- 4.2.4 VLM-Driven Rubric Generation from Failure Cases

For each sampled hard pair, we generate new candidate rubrics via a two-stage prompting procedure. First, in failure diagnosis, the VLM receives the text prompt, both images, and the current rubric set Rtretained, and diagnoses which missing visual or semantic dimension explains the human preference. Second, in rubric extraction, the VLM produces objective, reusable, and visually grounded rubric statements conditioned on the diagnosis. The newly extracted rubrics are deduplicated and appended to form Rt+1 = Rtretained ∪ Rtnew. The next round re-scores this expanded set and re-solves Eq. (3), progressively expanding the rubric space while using the ℓ1 refiner to maintain a compact, weighted global rubric set.

- 5 Experiments

We evaluate AutoRubric-T2I along two axes: RQ1: How does our learned rubric reward compare against fine-tuned RMs and existing rubric baselines on preference benchmarks? RQ2: Can the learned rubrics provide a robust signal for downstream T2I-RL?

#### 5.1 Experimental Setup

Models and Baselines. For VLM judges, we use Qwen3-VL-8B, Qwen3-VL-32B, and Gemini-3Flash. Baselines include: CLIP-based & scalar RMs (CLIPScore, ImageReward, PickScore, HPSv2); fine-tuned VLM RMs (HPSv3, UnifiedReward on Qwen2.5-VL-7B); zero-shot VLM judges in pairwise and pointwise modes; and rubric-based methods AutoRule [24] and AutoRubric [31]. Note that these methods are originally developed for text, while here we adapted them to T2I. (See the details in Appendix D)

Rubric Generation. We use Gemini-3-Flash to generate reasoning chains for rubric generation, including all rubric-based baselines.

| |MMRB2 (Out-of-domain)| |In-domain|
|---|---|---|---|
|Model|EvalMuse OneIG-Bench R2I-Bench RealUnify WISE<br><br>|Overall|PickScore HPSv3|

###### VLM-as-a-judge (Pairwise)

Qwen3-VL-8B 59.2 62.8 57.4 62.4 50.9 59.4 62.2 60.4 Qwen3-VL-32B 63.3 68.3 60.9 65.6 59.0 64.1 63.8 62.6 Gemini-3-Flash 69.8 73.2 70.7 78.5 65.0 70.8 70.3 65.6

VLM-as-a-judge (Pointwise) Qwen3-VL-8B 29.8 (60.3) 27.0 (58.5) 25.8 (56.3) 23.1 (55.5) 17.7 (54.1) 26.5 (58.0) 35.4 (58.8) 24.6 (55.6) Qwen3-VL-32B 25.9 (58.7) 21.6 (57.6) 20.3 (55.5) 21.5 (53.8) 15.3 (53.2) 22.4 (56.9) 40.1 (60.3) 28.7 (57.5) Gemini-3-Flash 36.7 (64.3) 38.4 (63.8) 38.6 (63.4) 34.1 (61.8) 30.9 (61.8) 36.6 (62.1) 52.1 (66.6) 42.5 (61.4) CLIP-Based

CLIPScore 54.4 51.1 53.1 46.2 55.0 52.6 60.8 48.6 ImageReward 51.0 51.4 58.6 52.7 57.7 53.0 61.1 58.6 HPSv2 51.0 53.6 63.3 60.2 55.0 54.6 70.5 65.6 PickScore 57.2 54.3 67.2 53.8 57.7 57.4 63.8 65.3

###### Fine-Tuned (Qwen2.5-VL-7B)

HPSv3 54.0 60.4 68.0 68.9 56.7 59.4 67.3 74.0 UnifiedReward 56.9 62.1 56.8 67.8 56.0 59.8 68.8 65.8

###### VLM-as-a-judge (Pointwise) Qwen3-VL-8B

AutoRule (on HPSv3) 56.9 60.4 64.1 62.4 55.0 59.1 61.1 62.8 AutoRule (on PickScore) 50.8 61.5 60.2 58.1 57.7 56.4 59.8 57.5 AutoRubric (on HPSv3) 54.1 61.9 61.7 59.1 52.3 57.5 57.1 58.1 AutoRubric (on PickScore) 53.1 56.1 59.4 59.1 68.5 57.0 54.8 57.1 AutoRubric-T2I (on HPSv3) 58.5 67.3 64.1 65.6 60.4 62.5 61.7 63.9 AutoRubric-T2I (on PickScore) 60.8 66.2 65.6 61.3 55.9 62.4 63.2 61.5

###### VLM-as-a-judge (Pointwise) Qwen3-VL-32B

AutoRule (on HPSv3) 64.1 68.0 64.8 64.5 61.3 65.0 60.5 62.7 AutoRule (on PickScore) 66.2 65.8 60.9 66.7 57.7 63.5 56.6 60.4 AutoRubric (on HPSv3) 61.3 64.4 67.2 66.7 62.2 63.4 62.5 60.6 AutoRubric (on PickScore) 60.0 57.9 57.0 59.1 58.6 58.8 58.8 56.7 AutoRubric-T2I (on HPSv3) 64.2 67.3 68.4 69.1 64.5 65.4 63.1 65.1 AutoRubric-T2I (on PickScore) 66.9 68.8 67.8 73.4 62.4 67.7 64.9 62.5

###### VLM-as-a-judge (Pointwise) Gemini-3-Flash

AutoRule (on HPSv3) 67.7 68.4 67.2 72.0 62.2 67.6 67.2 62.6 AutoRule (on PickScore) 66.4 65.8 62.5 68.8 55.0 64.7 68.3 59.4 AutoRubric (on HPSv3) 61.2 62.8 66.6 74.2 58.7 63.8 65.0 62.2 AutoRubric (on PickScore) 58.7 57.1 60.2 58.1 58.6 58.6 59.9 56.2 AutoRubric-T2I (on HPSv3) 70.2 71.3 70.8 78.7 64.9 70.8 69.0 70.0 AutoRubric-T2I (on PickScore) 70.7 71.9 71.1 79.5 66.0 71.4 70.3 66.8

- Table 1: Full comparison across MMRB2 out-of-domain and in-domain benchmarks. Within each VLM pointwise block, bold and underline indicate the best and second-best scores, respectively. Parentheses report tie-adjusted scores: (#wins + 0.5 ∗ #ties)/#total.

Datasets. We evaluate on out-of-distribution (OOD) image generation reward benchmarks including MMRB2 [10]. We also report in-domain performance on the test splits of HPSv3 and PickScore. For downstream T2I RL, we fine-tune SD-3.5-Medium [3] using Flow-GRPO [18] and evaluate on TIIF [27] and UniGenBench++ [25] datasets. (See the details in the Appendix E.)

#### 5.2 Preference Benchmark Evaluation

We first evaluate whether AutoRubric-T2I produces human-aligned preference judgments. As shown in Table 1, raw pointwise VLM judges are unreliable without explicit guidance: Qwen3VL-8B achieves only 26.5% overall accuracy on MMRB2, compared with 59.4% when used as a direct pairwise judge. By equipping the same Qwen3-VL-8B pointwise judge with learned rubrics, AutoRubric-T2I improves the overall MMRB2 accuracy to 62.5% and 62.4% when learned from HPSv3 and PickScore preferences, respectively. This outperforms both direct pairwise Qwen3-VL-8B evaluation and existing rubric-generation baselines such as AutoRule and AutoRubric. Notably, AutoRubric-T2I also surpasses fine-tuned scalar reward models on this out-of-domain benchmark, including HPSv3 and UnifiedReward, which obtain 59.4% and 59.8% overall accuracy. With stronger judges, AutoRubric-T2I further improves Qwen3-VL-32B to 67.7% and Gemini-3-Flash to 71.4%, exceeding their corresponding direct pairwise VLM baselines.

On in-domain PickScore and HPSv3 test sets, fine-tuned scalar reward models remain strong because they are trained directly on large-scale data from these distributions. For example, HPSv3 achieves 74.0% on the HPSv3 test set. Nevertheless, AutoRubric-T2I remains competitive without updating the VLM or training a dense reward model: Qwen3-VL-8B with AutoRubric-T2I reaches 63.2% on PickScore and 63.9% on HPSv3, while Gemini-3-Flash with AutoRubric-T2I reaches 70.3% and 70.0%, respectively. These results demonstrate that AutoRubric-T2I converts VLMs into effective pointwise reward models, which are more convenient for downstream ranking and RFT than pairwise preference evaluators.

Model Attr. Relation Reason. Attr.+Rel. Attr.+Reason. Rel.+Reason. Text Gen. Style Real Complex Overall TIIF (short)

SD-3.5-Medium 78.0 76.6 66.8 64.8 60.4 59.6 45.7 70.0 66.0 65.3 + HPSv3 83.5 75.5 65.8 64.5 57.4 60.7 62.0 76.7 73.5 68.8 + AutoRule (HPSv3) 85.0 72.4 70.0 68.2 58.8 57.5 55.2 80.0 74.6 69.1 + AutoRubric-T2I (HPSv3) 84.5 76.2 74.6 67.3 58.3 67.1 65.2 77.3 73.8 71.6 + PickScore 82.5 77.0 76.2 67.6 56.4 56.1 61.1 66.7 70.9 68.3 + AutoRule (PickScore) 85.5 74.2 65.8 68.1 60.2 59.2 64.2 66.7 70.5 68.3 + AutoRubric-T2I (PickScore) 84.0 77.2 74.6 67.8 61.8 58.0 69.7 70.0 74.2 70.8

###### TIIF (long)

SD-3.5-Medium 75.0 73.5 66.5 62.9 60.9 57.5 30.3 63.3 74.2 62.7 + HPSv3 78.0 70.9 70.0 71.2 55.6 58.4 50.7 56.7 67.2 64.3 + AutoRule (HPSv3) 79.5 69.9 69.7 64.4 65.2 58.5 49.8 73.3 74.2 67.2 + AutoRubric-T2I (HPSv3) 78.5 69.8 69.6 69.6 64.0 61.4 49.8 73.3 75.4 67.9 + PickScore 80.5 76.1 66.4 67.7 57.3 61.9 48.4 60.0 76.9 66.1 + AutoRule (PickScore) 80.5 74.7 64.3 69.7 58.6 63.8 43.9 63.3 68.7 65.3 + AutoRubric-T2I (PickScore) 79.5 73.8 71.6 68.8 63.8 64.8 54.8 66.7 77.6 69.0

- Table 2: Evaluation results for T2I RL on TIIF. We report results on both short and long prompts, comparing scalar reward models, AutoRule rewards, and AutoRubric-T2I on Qwen3-VL-8B.

Model Style World Know. Attribute Action Relation Compound Grammar Logic Layout Text Overall UniGenBench++ (short)

SD-3.5-Medium 87.6 87.0 70.1 61.1 66.8 57.3 62.3 28.9 73.1 15.5 61.0

- + HPSv3 81.2 88.6 67.3 55.5 67.8 54.4 53.2 29.8 72.8 17.2 58.8

- + AutoRule (HPSv3) 90.2 89.2 72.7 59.9 71.3 60.9 60.4 31.2 74.6 10.3 62.1

+ AutoRubric-T2I (HPSv3) 90.0 88.6 70.5 62.6 71.8 58.5 58.6 32.6 76.9 16.7 62.7 + PickScore 84.8 89.6 70.5 59.9 69.0 55.2 60.4 30.7 75.8 13.8 61.0

- + AutoRule (PickScore) 88.2 88.3 68.0 61.5 68.8 59.3 60.4 33.0 75.4 16.7 62.0

+ AutoRubric-T2I (PickScore) 90.4 88.5 71.6 61.8 70.1 59.6 61.2 32.2 74.6 14.1 62.4

UniGenBench++ (long)

SD-3.5-Medium 91.7 88.4 80.4 58.6 70.4 62.2 63.2 41.7 74.6 9.2 64.0 + HPSv3 82.7 89.6 79.6 57.6 69.2 60.1 60.7 42.7 73.3 10.3 62.6 + AutoRule (HPSv3) 91.4 89.9 82.3 62.3 70.6 66.3 64.5 52.0 75.6 8.7 66.3 + AutoRubric-T2I (HPSv3) 90.7 89.3 82.2 61.5 71.0 66.9 66.8 50.1 78.3 12.5 66.9 + PickScore 86.7 87.9 81.1 59.9 69.3 64.7 63.5 45.1 75.1 10.3 64.4

- + AutoRule (PickScore) 89.9 89.9 81.6 61.8 70.0 66.5 65.5 45.6 77.8 9.2 65.8

+ AutoRubric-T2I (PickScore) 93.5 90.2 82.2 63.7 72.2 70.1 64.4 52.9 76.6 10.9 67.7

- Table 3: Evaluation results for T2I RL on UniGenBench++. We report results on both short and long prompts, comparing scalar reward models, AutoRule rewards, and AutoRubric-T2I on Qwen3-VL-8B.

#### 5.3 Downstream T2I Reinforcement Learning

To verify whether the learned rubrics can serve as a dense and reliable training signal, we apply AutoRubric-T2I to fine-tune SD-3.5-Medium [3] with Flow-GRPO [18]. Since reward evaluation is repeatedly invoked during RL, we use Qwen3-VL-8B guided by the learned AutoRubric-T2I rubric set as a reward function. The final reward is computed as a weighted sum of rubric-level scores, where each score is the predicted probability of the yes token multiplied by its learned rubric weight. We provide implementation details in the Appendix B.

As shown in Tables 2 and 3, AutoRubric-T2I consistently improves downstream T2I RL over both scalar reward models and generic rubric-based rewards. On TIIF, AutoRubric-T2I improves SD-3.5Medium from 65.3% to 71.6% on short prompts and from 62.7% to 67.9% on long prompts under the HPSv3 setting. Under the PickScore setting, it reaches 70.8% and 69.0% overall, outperforming both PickScore and AutoRule. The gains are particularly strong in reasoning, relation composition, and text generation, suggesting that explicit rubrics provide more targeted feedback than scalar rewards.

The same trend holds on UniGenBench++. AutoRubric-T2I improves the base model from 61.0% to 62.7% on short prompts and from 64.0% to 66.9% on long prompts under the HPSv3 setting. With PickScore-derived rubrics, it further reaches 62.4% and 67.7% overall. Compared with scalar reward optimization, our method yields stronger improvements in fine-grained categories such as relation, compound reasoning, layout, and text rendering. These results demonstrate that AutoRubric-T2I

provides a more reliable optimization signal for T2I RL by decomposing preference alignment into explicit, learnable visual criteria instead of relying on a single opaque scalar reward.

### 6 Discussion and Analysis

#### 6.1 Ablation Study: Deconstructing AutoRubric-T2I

We ablate the key components of AutoRubric-T2I using Qwen3-VL-8B as the pointwise judge and evaluate on the out-of-domain MMRB2 and the in-domain PickScore and HPSv3. Starting from AutoRule [24], we progressively add our proposed components: ℓ1-based rubric selection, positivityconstrained weights, multi-round refinement, hard-pair evolution, and cluster-based initialization.

As shown in Table 4, simply adding an ℓ1 refiner with unrestricted weights provides little improvement, since negative weights make the learned rubric set difficult to interpret: satisfying a rubric may decrease the final reward. Enforcing positive-only weights improves performance, suggesting that rubrics are more effective when they serve as additive criteria whose satisfaction consistently indicates higher preference. Multi-round refinement further improves results, but randomly sampled pairs bring only modest gains. In contrast, mining hard pairs from the previous round leads to larger improvements, showing that difficult preference pairs expose missing or ambiguous criteria and guide the model toward more useful rubrics. Finally, cluster-based initialization gives the best overall performance, improving MMRB2 from 59.1% to 62.5% on HPSv3 and from 56.4% to 62.4% on PickScore. This confirms that both the initial rubric coordinates and the subsequent hard-pair refinement trajectory are important for building a robust rubric reward.

Method PickScore HPSv3 MMRB2 Qwen3-VL-8B on HPSv3

AutoRule 61.1 62.8 59.1 + L1 refiner w/ negative weights 60.5 62.5 59.2 + positive-only weights 62.2 63.4 60.1 + multi-round refinement w/ random pairs 61.6 63.1 60.3 + hard-pair evolution 61.7 63.5 61.3 + cluster initialization 61.7 63.9 62.5

Qwen3-VL-8B on PickScore

AutoRule 59.8 57.5 56.4 + L1 refiner w/ negative weights 59.9 57.7 56.8 + positive-only weights 60.4 58.3 58.2 + multi-round refinement w/ random pairs 60.3 58.6 59.4 + hard-pair evolution 61.6 60.1 61.0 + cluster initialization 63.2 61.5 62.4

Table 4: Ablation study of AutoRubric-T2I. Starting from AutoRule, we progressively add positive L1-based rule selection, multi-round refinement, hard-pair evolution, and cluster-based initialization.

#### 6.2 Qualitative and Human Evaluation of the RL Policy

While Section 5.3 provides useful indicators of policy improvement, human perception remains the gold standard for T2I evaluation. Figure 5, 6 qualitatively compares the base model, scalar-reward optimization, AutoRule-based optimization, and AutoRubric-T2I. Scalar rewards and generic rubric rewards can improve visual appeal, but they may still miss fine-grained prompt constraints. In contrast, AutoRubric-T2I better preserves requested objects, relations, and scene structure, suggesting that learned rubrics provide more targeted feedback during RL.

We further conduct a 4-way human evaluation with 30 annotators over 20 prompts. Annotators select the best image among outputs from the base model, scalar reward model, AutoRule reward, and AutoRubric-T2I. AutoRubric-T2I achieves a selection rate of 44.8%, substantially above the random baseline of 25% with a 95% confidence interval of [40.7%, 48.9%]. The improvement is statistically significant (p < 0.001), demonstrating that AutoRubric-T2I yields generations that are more frequently preferred by human evaluators. See the details in Appendix L.

### 7 Conclusion

We introduced AutoRubric-T2I, a fine-tuning-free reward modeling framework that learns a compact, weighted set of natural-language rubrics for text-to-image preference alignment. Instead of relying on opaque scalar reward models or manually designed rubrics, AutoRubric-T2I uses VLM-scored rubric features, ℓ1-regularized sparse selection, and hard-pair-driven refinement to derive interpretable criteria directly from image preference data.

Experiments show that AutoRubric-T2I improves preference prediction on out-of-domain benchmarks such as MMRB2, outperforming existing rubric-generation baselines and competitive fine-tuned scalar reward models. When used as a reward signal for Flow-GRPO, AutoRubric-T2I further improves downstream T2I generation on TIIF and UniGenBench++, producing images with stronger prompt alignment and better structural fidelity. These results suggest that learned, weighted rubrics provide a practical and interpretable alternative to scalar reward models for robust T2I alignment.

### References

- [1] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023.
- [2] Lichang Chen, Chen Zhu, Davit Soselia, Jiuhai Chen, Tianyi Zhou, Tom Goldstein, Heng Huang, Mehrdad Farajtabar, and Hongyang Li. Odin: Disentangled reward mitigates hacking in rlhf. arXiv preprint arXiv:2402.07319, 2024.
- [3] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.
- [4] Xuelu Feng, Yunsheng Li, Ziyu Wan, Zixuan Gao, Junsong Yuan, Dongdong Chen, and Chunming Qiao. Rubricrl: Simple generalizable rewards for text-to-image generation. arXiv preprint arXiv:2511.20651, 2025.
- [5] Google DeepMind. Gemini 3 system card. https://deepmind.google/technologies/ gemini/, 2025. Accessed: 2026-04-23.
- [6] Anisha Gunjal, Anthony Wang, Elaine Lau, Vaskar Nath, Yunzhong He, Bing Liu, and Sean Hendryx. Rubrics as rewards: Reinforcement learning beyond verifiable domains. arXiv preprint arXiv:2507.17746, 2025.
- [7] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Runxin Zhang, Runze Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [8] Yunqi Hong, Kuei-Chun Kao, Hengguang Zhou, and Cho-Jui Hsieh. Understanding reward hacking in text-to-image reinforcement learning. arXiv preprint arXiv:2601.03468, 2026.
- [9] Yushi Hu, Benlin Liu, Jungo Kasai, Yizhong Wang, Mari Ostendorf, Ranjay Krishna, and Noah A. Smith. Tifa: Accurate and interpretable text-to-image faithfulness evaluation with question answering. arXiv preprint arXiv:2303.11897, 2023.
- [10] Yushi Hu, Reyhane Askari-Hemmat, Melissa Hall, Emily Dinan, Luke Zettlemoyer, and Marjan Ghazvininejad. Multimodal rewardbench 2: Evaluating omni reward models for interleaved text and image. arXiv preprint arXiv:2512.16899, 2025.
- [11] Zenan Huang, Yihong Zhuang, Guoshan Lu, Zeyu Qin, Haokai Xu, Tianyu Zhao, Ru Peng, Jiaqi Hu, Zhanming Shen, Xiaomeng Hu, et al. Reinforcement learning with rubric anchors. arXiv preprint arXiv:2508.12790, 2025.
- [12] Prateek Jain, Ambuj Tewari, and Inderjit Dhillon. Orthogonal matching pursuit with replacement. Advances in neural information processing systems, 24, 2011.

- [13] Dongfu Jiang, Max Ku, Tianle Li, Yuansheng Ni, Shizhuo Sun, Rongqi Fan, and Wenhu Chen. Genai arena: An open evaluation platform for generative models. Advances in Neural Information Processing Systems, 37:79889–79908, 2024.
- [14] Dongzhi Jiang, Ziyu Guo, Renrui Zhang, Zhuofan Zong, Hao Li, Le Zhuo, Shilin Yan, PhengAnn Heng, and Hongsheng Li. T2i-r1: Reinforcing image generation with collaborative semantic-level and token-level cot. arXiv preprint arXiv:2505.00703, 2025.
- [15] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36:36652–36663, 2023.
- [16] Baiqi Li, Zhiqiu Lin, Deepak Pathak, Jiayao Li, Yixin Fei, Kewen Wu, Tiffany Ling, Xide Xia, Pengchuan Zhang, Graham Neubig, and Deva Ramanan. Genai-bench: Evaluating and improving compositional text-to-visual generation. arXiv preprint arXiv:2406.13743, 2024.
- [17] Sunzhu Li, Jiale Zhao, Miteto Wei, Huimin Ren, Yang Zhou, Jingwen Yang, Shunyu Liu, Kaike Zhang, and Wei Chen. Rubrichub: A comprehensive and highly discriminative rubric dataset via automated coarse-to-fine generation. arXiv preprint arXiv:2601.08430, 2026.
- [18] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025.
- [19] Zhen Liu, Yixin Wang, Jianfei Chen, and Jun Zhu. Openrubrics: Contrastive rubric generation for reward models. arXiv preprint arXiv:2505.14826, 2025.
- [20] Yuhang Ma, Xiaoshi Wu, Keqiang Sun, and Hongsheng Li. Hpsv3: Towards wide-spectrum human preference score. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15086–15095, 2025.
- [21] Yagyensh Chandra Pati, Ramin Rezaiifar, and Perinkulam Sambamurthy Krishnaprasad. Orthogonal matching pursuit: Recursive function approximation with applications to wavelet decomposition. In Proceedings of 27th Asilomar conference on signals, systems and computers, pages 40–44. IEEE, 1993.
- [22] Keivan Rezaei, Xuechen He, and Percy Liang. Onlinerubrics: Dynamic rubric elicitation for online reinforcement learning. arXiv preprint arXiv:2507.09832, 2025.
- [23] Yifan Shen, Xiang Li, Wei Zhang, and Yang Liu. Rrd: Recursive rubric decomposition for scalable reward modeling. arXiv preprint arXiv:2601.05743, 2026.
- [24] Tevin Wang and Chenyan Xiong. Autorule: Reasoning chain-of-thought extracted rule-based rewards improve preference learning. arXiv preprint arXiv:2506.15651, 2025.
- [25] Yibin Wang, Zhimin Li, Yuhang Zang, Jiazi Bu, Yujie Zhou, Yi Xin, Junjun He, Chunyu Wang, Qinglin Lu, Cheng Jin, et al. Unigenbench++: A unified semantic evaluation benchmark for text-to-image generation. arXiv preprint arXiv:2510.18701, 2025.
- [26] Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang. Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236, 2025.
- [27] Xinyu Wei, Jinrui Zhang, Zeqing Wang, Hongyang Wei, Zhen Guo, and Lei Zhang. Tiif-bench: How does your t2i model follow your instructions? arXiv preprint arXiv:2506.02161, 2025.
- [28] Xumeng Wen, Zihan Liu, Shun Zheng, Shengyu Ye, Zhirong Wu, Yang Wang, Zhijian Xu, Xiao Liang, Junjie Li, Ziming Miao, et al. Reinforcement learning with verifiable rewards implicitly incentivizes correct reasoning in base llms. arXiv preprint arXiv:2506.14245, 2025.
- [29] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023.
- [30] Xiaoshi Wu, Yiming Li, Keqiang Zhang, and Hongsheng Li. Rewarddance: Scaling visual reward modeling via generative next-token prediction. arXiv preprint arXiv:2504.12345, 2025.

- [31] Lipeng Xie, Sen Huang, Zhuo Zhang, Anni Zou, Yunpeng Zhai, Dingchao Ren, Kezun Zhang, Haoyuan Hu, Boyin Liu, Haoran Chen, et al. Auto-rubric: Learning from implicit weights to explicit rubrics for reward modeling. arXiv preprint arXiv:2510.17314, 2025.
- [32] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023.
- [33] Ran Xu, Tianci Liu, Zihan Dong, Tony Yu, Ilgee Hong, Carl Yang, Linjun Zhang, Tao Zhao, and Haoyu Wang. Alternating reinforcement learning for rubric-based reward modeling in non-verifiable llm post-training. arXiv preprint arXiv:2602.01511, 2026.
- [34] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.
- [35] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [36] Yongjin Yang, Sihyeon Kim, Hojung Jung, Sangmin Bae, SangMook Kim, Se-Young Yun, and Kimin Lee. Automated filtering of human feedback data for aligning text-to-image diffusion models. arXiv preprint arXiv:2410.10166, 2024.
- [37] Ian E Yen, Ting-Wei Lin, Shou-De Lin, Pradeep Ravikumar, and Inderjit S Dhillon. Sparse random feature algorithm as coordinate descent in hilbert space. Advances in Neural Information Processing Systems, 27, 2014.
- [38] Junkai Zhang, Zihao Wang, Lin Gui, Swarnashree Mysore Sathyendra, Jaehwan Jeong, Victor Veitch, Wei Wang, Yunzhong He, Bing Liu, and Lifeng Jin. Chasing the tail: Effective rubricbased reward modeling for large language model post-training. arXiv preprint arXiv:2509.21500, 2026.

### Table of Contents of Appendix

- A AutoRubric-T2I Pipeline Algorithm 14
- B RL Fine-Tuning Setup 14
- C Relation to Prior Rubric-Based Methods 14
- D Baseline Overview 15
- E Dataset and Benchmark Details 16
- F Details of Hyperparameter for AutoRubric-T2I 16
- G Qualitative Examples 17
- H Runtime and Data-needed Analysis 18
- I Training Dynamics 18
- J Limitations and Broader Impact 20
- K Prompt 20
- L Quantitative Human Evaluation 25
- M Optimized Rubric Set 25

- A AutoRubric-T2I Pipeline Algorithm This is the pseudo-code algorithm for AutoRubric-T2I, which is specifically discussed in Section 4. Algorithm 1 Working-set refinement pipeline for AutoRubric-T2I. Require: Train data Dtrain, valid data Dvalid, VLM judge, seed rubrics R0, Top-N, rounds R, hard-pair K

- 1: Init: best_acc ← −1, Rbest ← ∅, wbest ← ∅
- 2: for t = 0, 1, . . . , R − 1 do
- 3: Score rubrics in Rt on Dtrain with the VLM judge to obtain ∆s(ji)
- 4: Solve Eq. (3) over Rt and retain Top-N positive-weight rubrics (Rtretained, wretainedt )
- 5: Evaluate (Rtretained, wretainedt ) on Dvalid
- 6: if validation accuracy > best_acc then
- 7: best_acc ← validation accuracy
- 8: (Rbest, wbest) ← (Rtretained, wretainedt )
- 9: end if
- 10: if t < R − 1 then
- 11: Mine K hard pairs misranked by (Rtretained, wretainedt ) using curriculum buckets
- 12: Generate new rubrics Rtnew from hard-pair failure diagnoses
- 13: Update working set: Rt+1 ← Rtretained ∪ Rtnew
- 14: end if
- 15: end for
- 16: return (Rbest, wbest)

- B RL Fine-Tuning Setup

Algorithm overview. Downstream policy optimization follows Flow-GRPO [18], a GRPO adapted to flow-matching text-to-image samplers. For each prompt p in a mini-batch, the policy generates K rollouts {x(1),...,x(K)} by integrating the flow ODE for Ttrain steps; each rollout receives a scalar reward r(p,x(k)) and the group-relative advantage is computed as A(k) = (r(k) − µp)/(σp + ε), with µp and σp the in-group mean and standard deviation. We adopt the public Flow-GRPO implementation without algorithmic modification; only the reward signal is replaced (see below).

Backbones and training data. We fine-tune Stable Diffusion 3.5-Medium (SD3.5-M) on the prompt set used by Flow-GRPO [18], which is itself derived from the Pick-a-Pic prompt distribution [15]; this matches the training-prompt setting of prior T2I-RL work and isolates the contribution of the reward signal. No human preference labels from these prompts are used during RL—only the rubric reward defined below.

Integration with AutoRubric-T2I. At each rollout x, we evaluate the retained rubric set R(final) = {(ρj,wj)}Nj=1 produced by Section 4.2: for every rule ρj, the VLM judge is asked the binary question “Does this image satisfy this rule?” and we read the probability of the “Yes” token, pj(x,p) = PVLM(YES | x,p,ρj). The reward fed to Flow-GRPO is the learned weighted sum

rAutoRubric(x,p) =

N

j=1

wj pj(x,p), (4)

where the weights wj are exactly the ℓ1-logistic-regression coefficients fit on the 256 preference pairs (no re-tuning at RL time). Because pj ∈ [0,1] and the weights are obtained from a discriminative fit on held-out preferences, rAutoRubric is bounded, interpretable per-dimension, and scale-compatible with the group-relative advantage in Flow-GRPO. Substituting Eq. (4) for the scalar reward model is the only change to the upstream pipeline.

Hyperparameters. Table 5 lists the full RL configuration. LoRA rank/α, optimizer, learning rate, group size K=24, effective prompt batch, EMA, and evaluation cadence follow the Flow-GRPO defaults [18]. Training is run on 1 × 8 H200 GPUs; we report results between 100–500 steps, with checkpoints saved and evaluated every 25 steps.

- C Relation to Prior Rubric-Based Methods

AutoRubric-T2I differs from existing rubric-based methods in both rubric granularity and learning mechanism. Unlike prompt-adaptive methods such as RubricRL [4], which generate a new rubric

for each prompt during the RL loop, AutoRubric-T2I learns a compact global rubric set offline from preference data, making the reward signal reusable, cacheable, and consistent across evaluation and RFT. Compared with static rule extraction methods such as AutoRule [24], which primarily use CoT prompting to generate candidate rules, AutoRubric-T2I explicitly optimizes which rubrics should be retained and how they should be weighted. Specifically, rubric selection is formulated as the sparse logistic regression problem in Eq. (3), where VLM-scored rubric features are fitted to human preference labels and the ℓ1 penalty prunes redundant rules. Finally, our curriculum-bucketed hard-pair mining expands the rubric pool from failure cases of the current retained rubric set, closing the loop between rubric evaluation, failure discovery, and rubric refinement.

###### SD3.5-M

Backbone SD3.5-Medium Mixed precision fp16 Resolution 512 × 512 LoRA target / rank / α attention Q/K/V/out, r=32, α=64 Optimizer AdamW Learning rate 3×10−4

Train timesteps Ttrain 10 Eval timesteps Teval 40 CFG scale 4.5 Rollouts per prompt K 24 Effective prompt batch 48 Train batch size (per GPU) 9 Test batch size 16 Gradient accumulation steps N/2 (auto) Inner epochs 1 Timestep fraction (PPO mask) 0.99 KL β 0.01 EMA on policy weights yes Reported steps 100–500 Eval frequency every 25 steps

- Table 5: Training hyperparameters. K is the number of rollouts per prompt used to compute the group-relative advantage. Effective prompt batch is the number of unique prompts processed per outer iteration. KL β is the coefficient on the KL-to-reference penalty.

### D Baseline Overview

We summarize the baselines. AutoRule [24] and Auto-Rubric [31] are designed for text-domain preference learning, while RubricRL [4] targets T2I generation. These methods all replace opaque scalar rewards with explicit evaluation criteria, but differ in how rubrics are generated, selected, and used. We provide quantitative comparisons of data and computational cost in Appendix H.

AutoRule. AutoRule [24] extracts rule-based rewards from LLM reasoning chains in the text domain. It prompts a reasoning model to explain why preferred responses are better, extracts rule-like statements from these explanations, and merges them into a compact rule set. During RL, each response is evaluated against all rules by an LLM verifier, and the rule scores are averaged with uniform weights. However, it does not learn discriminative rule weights or iteratively refine rules on held-out preference errors.

Auto-Rubric. Auto-Rubric [31] is a training-free text-domain framework that generates candidate rubrics from preference pairs and revises them when their judgments disagree with ground-truth labels. It then compresses the resulting rubric pool by selecting semantically diverse rubrics in embedding space and applies the final set through majority voting with an LLM judge. Unlike AutoRubric-T2I, its selection criterion emphasizes semantic diversity rather than preference-discriminative power, and it is not designed for visual preference evaluation.

RubricRL. RubricRL [4] is a concurrent T2I method that generates prompt-specific visual checklists directly from each input prompt. For example, a prompt mentioning three red apples may induce rubrics about object count, color, and placement. Each image is scored by a VLM judge against

these generated criteria, and the binary scores are uniformly averaged as the reward for GRPO. RubricRL therefore depends on per-prompt rubric generation, uses uniform weights, and mainly captures constraints explicitly stated in the prompt, rather than learning a global preference-aligned rubric set from human preference data.

Fine-tuned reward models. ImageReward [32], HPSv2/HPSv3 [29, 20], PickScore [15], and UnifiedReward [26] are learned reward models trained on large-scale human preference data. They map a prompt-image pair to a scalar reward and are widely used for ranking, filtering, and RFT. However, because they compress multiple preference dimensions into one opaque score, they provide limited interpretability and can be vulnerable to reward hacking.

Zero-shot VLM judge. We also compare with zero-shot VLM judging, where a model such as Qwen3-VL is directly prompted to evaluate an image given a text prompt, without learned rubrics or preference-specific calibration. This baseline requires no reward-model training or rubric construction, but its criteria remain implicit and can be inconsistent. AutoRubric-T2I instead makes these criteria explicit and learns which rubrics are most predictive of human preferences.

### E Dataset and Benchmark Details

Training preference corpora. We instantiate AutoRubric-T2I on two human-preference corpora used independently as source distributions: HPDv3 [20] and PickScore (Pick-a-Pic) [15]. From each corpus, we select only |Dtrain|=256 preference pairs for seed-rubric generation and an additional held-out split for validation (used to monitor the best Top-N rule set across refinement rounds).

Diversity-aware seed selection. The 256 seed pairs are not sampled uniformly. As described in Section 4.2 of the main paper, we follow a FiFA-inspired [36] two-factor selection that combines (i) a preference-margin signal from a proxy reward model and (ii) a prompt-diversity signal obtained by clustering the text prompts. This design directly follows the ablation reported in Section 6 of the main paper, which shows that swapping cluster-based diversity-aware selection for AutoRule-style random sampling drops MMRB2 accuracy from 62.4% to 60.3% on HPSv3 (and produces a similar gap on PickScore), confirming that the clustering step is responsible for a measurable share of the final rubric quality rather than being a cosmetic preprocessing step.

Evaluation preference benchmarks. We evaluate rubric quality on three held-out preference test sets, summarized in Table 6. For HPDv3 [20] and PickScore [15], we use the official test splits released with each corpus (no overlap with the 256 seed pairs). MMRB2 (Multimodal RewardBench 2) [10] is a recent omni reward-model benchmark covering four subtasks—text-to-image, image editing, interleaved generation, and multimodal reasoning (“thinking-with-images”)—with 1,000 expert-annotated preference pairs per subtask drawn from 23 frontier models across 21 source tasks.

Generative T2I Benchmarks. For T2I generative quality assessment on RL post-training, we evaluate on two benchmarks: TIIF [27] and UniGenBench++ [25]. For TIIF, we evaluate instruction fidelity across categories, providing a graded measure of instruction-following capacity. We report both long and short prompt variants evaluation. For UniGenBench++, we probe semantic consistency with both long and short prompt variants, measuring coherence with brief versus detailed textual descriptions

Benchmark # Samples PickScore (official test) [15] 12,864 HPDv3 (official test) [20] 34,344 MMRB2 (T2I subtask) [10] 1,000

Table 6: Evaluation benchmarks. Cases denote the number of preference pairs used at test time. HPDv3 and PickScore use official test splits.

### F Details of Hyperparameter for AutoRubric-T2I

This appendix consolidates every non-trivial hyperparameter used in the AutoRubric-T2I pipeline reported in the main paper. The same configuration is used for both source corpora (HPDv3 and PickScore).

Models. The VLM judge that scores (image, prompt, rubric) triples is Qwen3-VL-8B-Instruct, served via vLLM. The rule generator (seed CoT vision reasoner, rule extractor/merger, and hard-pair diagnoser) uses Gemini-3-Flash [5] with temperature=0.1 and thinking_budget=1024. The VLM judge is queried with temperature=0.0 and at most 16 output tokens.

Refinement loop. For our best reported runs on both HPDv3 and PickScore, we run the iterative pipeline of Section 4.2 for R=10 rounds on |Dtrain|=256 preference pairs and retain the Top-N rules with N=20. Each round mines 16 hard pairs and discards any pair that has been selected more than

- 4 times across rounds (stale-pair cap), preventing degenerate loops on label noise. Validation uses the same source corpus’s held-out split.

Sparse rubric selection. The ℓ1-regularized logistic regression in Eq. (3) is fit with scikit-learn’s liblinear solver: penalty l1, regularization strength C=1.0, no intercept, and a fixed random state of 42. Coefficients with magnitude below 10−4 are treated as zero; positive coefficients are sorted in

descending order and the top N=20 define R(retainedt) .

Curriculum-bucketed hard-pair mining. Misranked pairs under the current retained rubric set are partitioned into the three buckets defined in Section 4.2—small-margin, large-margin, and highreward wrong—using a margin percentile of 0.3 and a reward quantile of 0.7. The 16 hard pairs sampled in round r are drawn from these buckets with phase-dependent weights (wsmall,wlarge,whigh):

 

(0.6, 0.4, 0.0) r < 3 (early) (0.5, 0.3, 0.2) 3 ≤ r < R−1 (mid) (0.3, 0.3, 0.4) r ≥ R−1 (late).

(wsmall,wlarge,whigh) =



Early rounds emphasize large-margin errors that expose missing preference dimensions; later rounds shift mass to high-reward wrong cases that drive finer-grained rubrics for distinguishing strong generations.

Summary table. Table 7 consolidates the values above for reproducibility.

Component Hyperparameter Value Models

VLM judge backbone Qwen3-VL-8B-Instruct Rule generator Gemini-3-Flash

VLM judge temperature 0.0 VLM judge max output tokens 16 Rule-generator temperature 0.1 Rule-generator thinking budget 1024

Sampling

# rounds R 10 # seed pairs |Dtrain| 256 Retained rules per round (Top-N) 20 Hard pairs per round 16

Refinement loop

Solver liblinear Regularization strength C 1.0 Intercept disabled Non-zero coefficient threshold 10−4

ℓ1 logistic regression

Margin percentile (small/large split) 0.3 Reward quantile (high-reward bucket) 0.7 Stale-pair cap (max repeats) 4

Hard-pair mining

- Table 7: Hyperparameters of the AutoRubric-T2I pipeline used to produce the main results on HPDv3 and PickScore. Values are shared across both source corpora.

### G Qualitative Examples

We present qualitative examples of images generated by models fine-tuned with AutoRubric-T2I rewards via the Flow-GRPO pipeline. Figure 5 and 6 show representative prompt–image pairs from UniGenBench++ and TIIF, illustrating improvements in prompt faithfulness, compositional accuracy, and visual quality compared to the base model.

### H Runtime and Data-needed Analysis

A key motivation of AutoRubric-T2I is to reduce the data and training cost of conventional reward modeling. CLIP-based reward models such as HPSv2 [29] and PickScore [15] require 137K–798K human preference pairs and multi-GPU training. Recent VLM-based reward models [26] reduce the annotation requirement but still require gradient-based training on 10K–100K+ preference pairs. In contrast, AutoRubric-T2I uses only 256 preference pairs and requires no neural reward model training. Its main cost is a one-time rubric refinement stage.

One-time rubric refinement cost. Our final pipeline runs for 10 refinement rounds. In each round, we mine 16 hard preference pairs and use them to generate new candidate rubrics. The VLM scoring is performed with Qwen3-8B [35] served by vLLM on 4× NVIDIA A6000 GPUs. Rule generation uses the Gemini API [5] with temperature=0.1 and thinking_budget=1024. The ℓ1-regularized logistic regression for rubric selection runs on CPU and takes less than one second per round.

Table 8 summarizes the wall-clock cost. Since the final setting keeps Top-20 rubrics and uses 10 rounds, the total number of VLM scoring calls is substantially smaller than earlier settings. The entire refinement stage finishes in approximately 2–4 hours on 4× A6000 GPUs. This cost is incurred only once and the resulting weighted rubric set can be reused for downstream evaluation and RL.

Component Hardware Est. Time VLM scoring (∼180K calls) Qwen3-8B / vLLM on 4× A6000 1.5–3 hours Validation scoring (∼20K–40K calls) Qwen3-8B / vLLM on 4× A6000 15–30 min Rule generation (320 calls) Gemini API ∼15–30 min Logistic regression (10 fits) CPU <1 min Total 2–4 hours

- Table 8: Wall-clock time breakdown for AutoRubric-T2I rubric refinement under the final setting: 10 rounds, 256 preference pairs, Top-20 rubrics, and 16 hard pairs per round.

Cost during RL. At deployment time, each generated image is independently scored by the 20 selected rubrics, which requires 20 VLM forward passes. For a representative RL setting with 10K prompts and 4 rollouts per prompt, this corresponds to 40K generated images and 800K rubric-scoring calls. These calls are parallel across images and rubrics, so throughput scales directly with additional vLLM replicas.

Method #Pref. Pairs Training Cost Calls / Image FLOPs / Image Open-Source? Interpretable?

CLIP-based RM 137K–798K 8–32 GPUs × days 1 ∼1.5 × 1011 Yes No VLM-based RM 10K–100K+ 8–64 GPUs × hrs–days 1 ∼2.4 × 1013 Varies No Zero-shot VLM Judge 0 None 1 ∼2.6 × 1013 Yes Partial RubricRL 0 None 10 API No Yes, uniform AutoRubric-T2I 256 2–4 hrs on 4× A6000 20 ∼5.1 × 1014 Yes Yes, learned weights

- Table 9: End-to-end cost comparison. AutoRubric-T2I trades higher inference cost for substantially lower annotation cost, no gradient-based reward-model training, and interpretable per-rubric scores with learned weights. FLOPs are estimated by 2P × T for open-weight models.

Summary. AutoRubric-T2I shifts the cost of reward modeling from large-scale annotation and reward-model training to a lightweight, one-time rubric refinement procedure. It requires only 256 preference pairs, runs in a few hours on 4× A6000 GPUs, and produces an interpretable weighted rubric set. Although per-image inference is more expensive than scalar reward models, the cost enables fine-grained diagnostics, rubric-level reward shaping, and improved robustness against scalar reward hacking.

### I Training Dynamics

We provide training curves of T2I-RL referenced in Section 3.2. In Figure 3, we visualize normalized training and evaluation rewards together with reward standard deviation. AutoRubric-T2I achieves steadily improving training and evaluation rewards while maintaining substantially lower reward variance. In contrast, HPSv3 exhibits much larger reward dispersion throughout training, suggesting a noisier optimization signal, whereas PickScore has extremely low reward variance, indicating limited sensitivity for distinguishing high-capability generations. Qualitative results throughout RL training are visualized in Figure 4, where scalar reward shows clear visual evidence of reward hacking.

[Figure 3]

Figure 3: Training dynamics of scalar and rubric-based T2I rewards.

[Figure 4]

- Figure 4: The evolution of generation quality of RL using AutoRubrics and other scalar reward models. The visual quality of scalar reward models degrades notably while the reward increases.

[Figure 5]

- Figure 5: Qualitative comparison of downstream T2I RL policies. AutoRubric-T2I better preserves promptspecific objects, relations, and fine-grained details compared with the base model, scalar reward optimization, and AutoRule-based rubric rewards.

### J Limitations and Broader Impact

Domain specificity of learned weights. The ℓ1-regularized weights are fit to the preference distribution of the training corpus (e.g., HPSv3 or PickScore), so the rubric texts remain broadly applicable, but their relative importance may shift on out-of-domain prompts such as typographyheavy or highly stylized images. Because rubric refinement requires only 256 pairs, the natural remedy is to re-fit the logistic regression on a small in-domain sample with the rubric set fixed, or to learn prompt-conditional weights.

RL training integration. AutoRubric-T2I can be directly integrated into Flow-GRPO as a drop-in reward by using the learned weighted sum of per-rubric scores. Beyond this simple scalarization, the explicit per-rubric decomposition provides a promising opportunity for rubric-aware advantage estimation, enabling finer-grained credit assignment during T2I reinforcement learning. However, how to fully exploit both rubric-level rewards and conventional scalar rewards for more effective advantage estimation remains an important direction for future work.

Broader impact. By replacing opaque scalar reward models with human-readable rubrics and learned weights, AutoRubric-T2I makes the criteria that drive T2I optimization inspectable and auditable, which helps diagnose and correct biases (e.g., systematic under-weighting of culturally diverse aesthetics) directly from the weights. The low data requirement and open-source infrastructure further lower the barrier for groups without large annotation pipelines or proprietary APIs, and we do not foresee negative societal impacts beyond those inherent to T2I generation broadly.

### K Prompt

Figures 7–10 present the complete set of prompt templates used at all stages of the AutoRubric-T2I pipeline: vision reasoner and rule extractor/merger for seed rubric generation, VLM judge templates, and hard-pair diagnosis and rule extraction prompts used during iterative refinement.

[Figure 6]

##### Figure 6: Qualitative examples from downstream RL fine-tuning with AutoRubric-T2I rewards. Each row shows a text prompt and the corresponding generated image, demonstrating improved prompt alignment, object placement, attribute accuracy, and overall visual quality after RL training.

[Figure 7]

##### Figure 7: Seed rubric generation, stage 1: vision reasoner that produces a step-by-step preference rationale for each image pair.

[Figure 8]

Figure 8: Seed rubric generation, stages 2-3: rule extractor rule merger.

[Figure 9]

Figure 9: VLM judge templates: Yes/No binary scoring.

[Figure 10]

##### Figure 10: Hard-pair refinement prompt.

### L Quantitative Human Evaluation

To supplement the quantitative human evaluation in Section 6.2, we provide a screenshot of the user survey interface used in our study. The survey was conducted with 30 graduate-student annotators. Each annotator was asked to evaluate 20 T2I prompts, resulting in 600 total human judgments.

For each question, annotators were shown a text prompt at the top, and four anonymized image candidates labeled A–D. These candidates correspond to outputs from the base model, scalar-reward optimization, AutoRule-based optimization, and AutoRubric-T2I, with the display order randomized to reduce positional bias. Annotators were instructed to select the single image that best satisfied the text prompt, considering both visual quality and prompt alignment. In particular, they were encouraged to focus on whether the generated image correctly captured the requested objects, spatial relations, attributes, scene layout, and other fine-grained constraints in the prompt.

Figure 11 shows an example survey question. The instruction explicitly asks users to vote for the best image according to the text prompt, and each question requires exactly one choice among the four candidates. This simple forced-choice design avoids requiring annotators to assign calibrated numerical scores, while still providing a direct measure of human preference among RL policies.

[Figure 11]

Figure 11: Screenshot of the human evaluation survey interface. Annotators were asked to choose the best image according to the text prompt shown in the question title.

### M Optimized Rubric Set

This section reports the final weighted rubric sets produced by the AutoRubric-T2I pipeline for each (VLM judge, source corpus) configuration. Rules are ranked in decreasing order of their fitted ℓ1-regularized logistic-regression weights (w¯); rules with effectively zero weight are pruned from the displayed Top-N set.

[Figure 12]

##### Figure 12: Optimized rubric set for Qwen-3-VL-8B trained on HPSv3 preference pairs (round 3).

[Figure 13]

##### Figure 13: Optimized rubric set for Qwen-3-VL-8B trained on PickScore preference pairs (round 6).

[Figure 14]

##### Figure 14: Optimized rubric set for Qwen-3-VL-32B trained on HPSv3 preference pairs (round 3).

[Figure 15]

##### Figure 15: Optimized rubric set for Qwen-3-VL-32B trained on PickScore preference pairs (round 6).

