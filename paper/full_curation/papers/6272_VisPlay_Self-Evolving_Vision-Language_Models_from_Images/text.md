[Figure 1]

## VisPlay: Self-Evolving Vision-Language Models from Images

# arXiv:2511.15661v2[cs.CV]20Nov2025

Yicheng He1* Chengsong Huang2* Zongxia Li3* Jiaxin Huang2 Yonghui Yang4 1University of Illinois Urbana-Champaign 2Washington University in St. Louis 3University of Maryland 4National University of Singapore

yh84@uiuc.edu chengsong@wustl.edu zli12321@umd.edu

#### Abstract

Reinforcement learning (RL) provides a principled framework for improving Vision-Language Models (VLMs) on complex reasoning tasks. However, existing RL approaches often depend on human-annotated labels or task-specific heuristics to define verifiable rewards—both costly and limited in scalability. We introduce VisPlay, a self-evolving RL framework that enables VLMs to autonomously improve their reasoning capabilities from massive unlabeled image data. Starting from a single base VLM, VisPlay assigns the model into two interacting roles: an Image-Conditioned Questioner that formulates challenging yet answerable visual questions, and a Multimodal Reasoner that generates silver responses. These roles are jointly trained using Group Relative Policy Optimization (GRPO), which uses diversity and difficulty rewards to balance the difficulty of generated questions with the quality of silver answers. VisPlay scales efficiently across two model families. Trained on Qwen2.5-VL and MiMo-VL, VisPlay achieves consistent improvements in visual reasoning, compositional generalization, and hallucination reduction across eight benchmarks including MM-Vet and MMMU, and establishes a scalable path toward self-evolving multimodal intelligence. Our project page is available at https://bruno686. github.io/VisPlay/.

#### 1. Introduction

Self-evolving mechanisms [8, 32] represent a promising frontier for advancing artificial intelligence. The training of state-of-the-art (SoTA) models has traditionally relied on large volumes of expert curated tasks and labels. However, the reliance on human annotation is not only costly, laborintensive, and difficult to scale, but also presents a funda-

*Core Contribution.

55

Avg.Accuracy(%)

GRPO w/ data

VisPlay w/o data

50

45

Along the Evolution

Figure 1. Illustration of the average accuracy improvement (averaged over seven datasets) through successive evolutions (Evo 1 to Evo 5) on Qwen2.5-VL-3B-Instruct, compared to a baseline trained on Vision-47K with GRPO, demonstrating the effectiveness of our VisPlay .

mental bottleneck to advancing intelligence toward capabilities that could surpass itself without human signal guidance [45]. Self-evolution offers a compelling alternative by equipping models with the capacity to independently generate, refine, and learn from their own experiences such as through self-play or synthetic data generation.

Motivated by these advantages, the research community has increasingly explored self-evolution, most notably in the context of Large Language Models (LLMs). A line of works have demonstrated how LLMs can autonomously enhance their complex reasoning and coding faculties, often by generating their own tasks or data [16, 26, 51]. However, the self-evolution paradigm remains largely underexplored for Vision-Language Models (VLMs) [2, 22, 35]. Unlike LLMs, which rely solely on text, developing self-evolving VLMs poses additional challenges due to their dependence on the visual modality. In a world where human annotation is costly and time-consuming, yet vast amounts of visual data are freely available online, self-evolving VLMs present a promising direction to continual improvement without human signals and directly from the abundant visual content on the internet [3, 33].

In this paper, we introduce VisPlay, a self-evolving RL framework that enables VLMs to autonomously improve their reasoning capabilities using only raw, unannotated images. The framework utilizes a single base VLM that alternates between two roles: the Image-Conditioned Questioner, which generates diverse and challenging questions conditioned on an input image, and the Multimodal Reasoner, which produces silver responses based on both the image and the generated question. Both roles are jointly optimized using Group Relative Policy Optimization (GRPO) [34], where designed rewards encourage a balance between question difficulty and answer quality without requiring external supervision. The Image-Conditioned Questioner learns to generate challenging yet answerable questions grounded in visual inputs, while the Multimodal Reasoner learns to produce accurate, detailed, and grounded responses. This self-evolving framework enables the VLM to progressively improve its visual reasoning abilities through iterative co-improvement of the Questioner and the Reasoner as Figure 1.

We apply our self-evolving RL framework to train three state-of-the-art (SoTA) VLMs and observe consistent performance gains across diverse visual reasoning benchmarks.

Our main contributions are:

- • We propose VisPlay, a self-evolving RL framework for Vision-Language models.
- • We apply VisPlay to three strong models—Qwen2.5VL-3B, Qwen2.5-VL-7B [35], and MiMo-VL-7B [43]. We run extensive evaluations over three major domains—General Visual Understanding, Visual Mathematics, and Hallucination Detection. All models show consistent gains in accuracy after several iterations.
- • We run extensive ablation studies to further validate the contribution of Image-Conditioned Questioner and the Multimodal Reasoner component to further show how VisPlay progressively strengthens multimodal reasoning across vision-language tasks.

#### 2. Method

##### 2.1. Preliminary

Reinforcement Learning with Verifiable Rewards (RLVR) [20] is a paradigm for training VLMs in domains where the correctness of model outputs can be verified. A rule-based verifier v : X → {0,1} assigns a binary reward to each generation xi:

ri = v(xi) =

1, if xi satisfies a correctness rule, 0, otherwise.

(1)

Such verifiable rewards are effective in tasks like mathematical reasoning, multiple choice, and code genera-

tion [52], where correctness can be objectively evaluated. GRPO [34] provides a practical RL algorithm without a value function by using relative rewards among multiple samples from the same prompt. Given a prompt p, a policy πθ

produces G complete responses {x1,...,xG} with corresponding rewards {r1,...,rG}. Rewards are normalized within the group to compute response-level advantages:

old

ri − mean(r1,...,rG) std(r1,...,rG) + εnorm

Aˆi =

, (2)

where εnorm is a small constant for stability.

The policy is then optimized using a clipped surrogate objective, regularized by a KL term to constrain policy drift:

1 G

LGRPO(θ) = −

clip

G

πθ(xi) πθ

Aˆi,

min

(xi)

old

i=1

πθ(xi) πθ

,1−ϵ, 1+ϵ A ˆi

(xi)

old

(3)

+ β KL(πθ ∥πθ

).

old

GRPO operationalizes RLVR principles to improve reasoning and generation quality in VLMs by rewarding responses with positive relative advantages while limiting policy deviation

##### 2.2. Pipeline Overview

We introduce VisPlay, a self-play reinforcement learning framework designed to evolve VLMs without humanannotated data. As illustrated in Figure 2, the framework operates as a closed-loop system involving two agents evolved from the same base model: an Image-Conditioned Questioner and a Multimodal Reasoner. The process begins with the Questioner taking an image as input to generate a visual query. Subsequently, the Reasoner receives both the image and the generated query to produce a response. Both the Questioner and the Reasoner are initialized from a shared pretrained backbone. The two agents co-evolve through iterative interactions: the Questioner is trained to generate more challenging questions, while the reasoner is trained to solve more and more challenging questions. The complete process is described in Algorithm 1.

##### 2.3. Image-Conditioned Questioner Training

The Questioner is an autoregressive policy denoted by Qθ. Conditioned on an input image I, it samples a group of

G questions {xi}Gi=1 ∼ Qθ(·|I), which are evaluated to produce scalar rewards {ri}Gi=1. These rewards are used to compute group-normalized advantages and to update Qθ with a GRPO objective. We next define reward components that constitute each ri.

[Figure 2]

- Figure 2. An illustration of our VisPlay framework, depicting the co-evolution of the Image-Conditioned Questioner and Multimodal Reasoner. Top: During the Questioner training stage, the Image-Conditioned Questioner is optimized via GRPO to produce challenge questions. The reward stems from the uncertainty of the frozen Multimodal Reasoner, computed by the consistency of its multiple generated answers. Bottom: In the Reasoner training stage, the Multimodal Reasoner is trained via GRPO on a curated set of challenging questions from the now-frozen Image-Conditioned Questioner, leveraging pseudo-labels from its own majority voting.

Pseudo-Label Generation. Since self-evolving VLMs learn without relying on labeled data, ground-truth answers for the Questioner’s generated questions are unavailable. Therefore, we introduce a method to approximate the corresponding answers. Given an image I and the generated question x, we introduce a Reasoner Sϕ that samples m responses {yj}mj=1.1 We define the empirical frequency of a candidate answer y as pˆ(y|x,I) = m1 mj=1 {y j = y}, and derive the pseudo-label via majority voting [15]: y˜ = arg maxy pˆ(y|x,I). We then define the confidence score for this pseudo-label as:

###### conf(x,I) = pˆ(˜y|x,I). (4)

Intuitively, conf(x,I) measures the Reasoner’s certainty about the pseudo-label: high values indicate stable and consistent predictions, whereas values near 0.5 reflect strong uncertainty. We therefore treat the degree of uncertainty (i.e., how close conf(x,I) is to 0.5) as a proxy for the model-perceived difficulty of the generated question.

Uncertainty Reward. The confidence score quantifies the Reasoner’s uncertainty, which we use as a proxy for

1The reasoner is evolved from the same base model as the Questioner. See Section 2.4 for details.

the model-perceived difficulty of the generated question. To encourage questions that probe the Reasoner’s limits, we compute the reward based on the confidence score c = conf(x,I). We define the uncertainty reward to penalize deviations from the point of maximum uncertainty:

###### runc(x,I) = 1 − 2c − 1 . (5)

This formulation yields a maximal reward of 1 when c = 0.5 and decreases linearly to 0 as the reasoner’s response distribution becomes deterministic (i.e., c → 1).

Diversity Regularization. To prevent the model from collapsing [5, 6, 9, 28, 54] into generating repetitive questions for a given image I, we introduce a redundancy penalty within its generated group XI. We cluster these generated questions based on pairwise similarity (BLEU score) to identify duplicates. For a question xi belonging to a cluster Ck(I) ⊆ XI, the regularization term is:

rdiv(xi,I) = λ |Ck(I)| G

, (6)

where Ck(I) denotes the cluster of similar questions for image I, and G is the total number of generated questions for

that image.

Format Constraint. We enforce a hard filter to ensure structural validity. Specifically, we require the generated question to be strictly enclosed within <question> tags. Any output failing to meet this format requirement is assigned zero reward. We denote this validity indicator as:

valid(x) =

1, if x is wrapped in <question> tags, 0, otherwise.

(7)

Final Questioner Reward. For each generated question xi conditioned on image I, we integrate the uncertainty signal and diversity regularization into a unified scalar objective:

ri = valid(xi) · ReLU runc(xi,I) − rdiv(xi,I) . (8)

This composite reward incentivizes the Questioner to generate challenging yet non-redundant questions while strictly filtering out malformed outputs. The ReLU function stabilizes GRPO updates by preventing spurious negative values from skewing the reward normalization across the group.

##### 2.4. Multimodal Reasoner Training

The training of the Multimodal Reasoner Sϕ builds upon the advancements of the Image-Conditioned Questioner. In each iteration, the Image-Conditioned Questioner functions produce challenging samples that serve as training targets. The Multimodal Reasoner then learns from these automatically curated samples, improving its visual reasoning ability without any external supervision.

Curated Dataset Construction. Following the update of the Image-Conditioned Questioner, we generate a diverse pool of N candidate questions {xi}Ni=1 per image by sampling xi ∼ Qθ(· | I). For each xi, we obtain m response samples from the current Multimodal Reasoner and compute the pseudo-label y˜i and confidence score ci = conf(xi,I). To focus on training samples that offer high information gain, we enforce an informative filter that retains pairs (xi,y˜i) with moderate confidence:

τlow ≤ ci ≤ τhigh, (9)

where τlow and τhigh are thresholds set to 0.25 and 0.75, respectively. This criterion effectively discards trivial samples where the model is already certain (ci > 0.75) as well as highly unstable or noisy generations (ci < 0.25). The final curated training set S is formed by collecting all retained pairs across images, up to a budgeted size, to optimize the Multimodal Reasoner via GRPO.

Algorithm 1: VisPlay: Self-Evolving RL for Vision-Language Models

Input: Initial models Qθ,Sϕ; Image dataset I; Group size G; Reasoner samples m; Dataset budget N; Thresholds τlow = 0.25,τhigh = 0.75.

Output: Evolved models Qθ and Sϕ.

- 1 for each self-play iteration do

- 2 for each image batch I ∈ I do

- 3 Sample question group {xi}Gi=1 ∼ Qθ(· | I);
- 4 for each question xi do

- 5 Sample m answers {yj}mj=1 ∼ Sϕ(· | I,xi);
- 6 Compute confidence ci ← conf(xi,I) via majority vote (Eq. 4);
- 7 Compute uncertainty reward runc ← 1 − |2ci − 1|;
- 8 Compute diversity penalty rdiv via clustering (Eq. 6);
- 9 Final reward: ri ← valid(xi) · ReLU runc − rdiv ;
- 10 end
- 11 Update Qθ via GRPO using rewards {ri}Gi=1;
- 12 end
- 13 Initialize curated dataset S ← ∅;
- 14 for each image I ∈ I do

- 15 Generate N candidate questions via Qθ ;
- 16 for each candidate xk do

- 17 Obtain pseudo-label y˜k and confidence ck from Sϕ;
- 18 if τlow ≤ ck ≤ τhigh then

- 19 Add (I,xk,y˜k) to S;
- 20 end
- 21 end
- 22 end
- 23 for each minibatch (I,x,y˜) ∈ S do

- 24 Sample G answers {yj}Gj=1 ∼ Sϕ(· | I,x);
- 25 Compute binary rewards rj ← (y j = y˜);
- 26 Update Sϕ via GRPO using rewards {rj}Gj=1;
- 27 end
- 28 end

Per-Sample Verifiable Reward. For a question xi ∈ S with pseudo-label y˜i, the Multimodal Reasoner generates a group of G candidate answers {yj}Gj=1. Each sampled answer receives the binary reward

rj =

1, if yj = y˜i, 0, otherwise.

(10)

[Figure 3]

- Table 1. Comprehensive results on visual reasoning benchmarks. Each base model is evaluated against two settings: a VisPlay ( challenger) baseline, in which the Reasoner is trained on questions produced by an untrained Challenger, and our iterative VisPlay framework. The highest performance reached during training for each model is emphasized in bold. We take accuracy as the metric.

General Visual Understanding Visual Math Hallucination Methods MMMU

MM RealWorld VisNum Math MATH Hallusion

Avg.

-Vet QA Bench Verse -Vision Bench Qwen2.5-VL-3B-Instruct

Base Model 19.95 36.24 49.28 27.08 26.14 20.23 32.81 30.61 VisPlay ( challenger) 23.34 43.58 57.78 29.33 33.50 23.39 64.88 33.77

[Figure 4]

- VisPlay (Iter 1) 29.40 48.62 67.06 30.01 29.67 22.57 91.80 44.16
- VisPlay (Iter 2) 33.37 44.50 65.62 29.64 32.36 24.67 94.95 44.87
- VisPlay (Iter 3) 37.11 38.07 71.90 39.15 35.15 29.97 90.54 47.27

Qwen2.5-VL-7B-Instruct

Base Model 23.10 44.95 57.52 32.57 33.78 24.05 66.88 40.41 VisPlay ( challenger) 35.24 45.87 69.67 32.41 35.13 26.22 78.13 38.33

[Figure 5]

- VisPlay (Iter 1) 28.94 46.33 62.61 28.65 33.88 26.91 80.34 44.53
- VisPlay (Iter 2) 27.07 42.66 60.92 27.08 36.32 25.00 67.72 40.97
- VisPlay (Iter 3) 38.27 46.33 69.67 32.57 39.14 31.15 92.32 48.61

MiMo-VL-7B-SFT

Base Model 30.22 59.17 78.17 44.80 41.80 25.33 87.17 43.56 VisPlay ( challenger) 27.54 58.72 63.27 49.66 38.78 24.80 57.83 39.63

[Figure 6]

- VisPlay (Iter 1) 25.67 56.42 69.54 51.59 40.20 25.13 87.59 43.16
- VisPlay (Iter 2) 34.07 55.96 78.69 51.18 41.65 28.45 86.65 45.58
- VisPlay (Iter 3) 28.24 56.88 71.50 52.69 46.02 29.44 74.55 45.69

These rewards are group-normalized to produce advantages Aˆj as in Eq. 2 (with the Reasoner’s rewards), and Sϕ is updated by minimizing LGRPO(ϕ) as in Eq. 3.

#### 3. Experiments

##### 3.1. Benchmarks and Evaluation Protocol

We use existing image datasets from Vision-47K [12, 25], which contains 47K web images collected from diverse domains, e.g. including charts, medical images, exams, textbooks, and driving simulations.2 We train three backbone models using VisPlay—Qwen2.5-VL-3BInstruct, Qwen2.5-VL-7B-Instruct, and Mimo-7B-SFT.3 We run evaluation across three multimodal domains [25].

• General Visual Understanding. We measure performance on four established benchmarks. MM-Vet [47] provides a unified LLM-based score across recognition, OCR, and visual math tasks. MMMU [48] evaluates cross-modal reasoning and subject knowledge through 11.5K college-level, four-choice questions spanning six academic disciplines. RealWorldQA [41] contains roughly 700 real-world images paired with spatially grounded questions. VisNumBench [40] focuses on vi-

- 2We only use the images without the questions and answers. Details of the dataset breakdowns are in supplement materials.
- 3The detailed training configurations are provided in the supplementary material.

sual number sense, covering around 1.9K questions involving numerical attributes and estimation tasks.

- • Multimodal Mathematical Reasoning. MathVerse [50] consists of 2.6K diagram-centric questions spanning geometry and functions, provided in multiple visualtext formats. MATH-Vision [37] includes around 3K competition-level problems across 16 subjects and five difficulty tiers.
- • Visual Hallucination Detection. HallusionBench [14] is used to analyze model errors, distinguishing between language-only hallucinations and visual-illusion errors, with a simple yes/no evaluation format.

##### 3.2. Main Results

We use LLM-as-a-judge to assess the correctness of the answers to ensure more robust evaluation [13, 23]. We present the outputs of the Multimodal Reasoner and analyze its reasoning ability progression in Table 1. We summarize the main findings of our experimental results below.

• VisPlay consistently improves overall performance across different models. All models trained with VisPlay consistently surpass both their corresponding base models and the Base Challenger over successive training iterations. Qwen2.5-VL-3B shows a remarkable improvement, with the average score increasing from 30.61 at baseline to 44.16 after the first iteration and reaching 47.27 at the third. Qwen2.5-VL-7B and MiMo-VL-7B

[Figure 7]

[Figure 8]

[Figure 9]

- Figure 3. Changes in question difficulty (orange, left axis) and problem-solving accuracy (blue, right axis) during Image-Conditioned Questioner and Multimodal Reasoner training across three VLMs.

- Table 2. Analysis of model performance and data quality. The shaded column indicates the estimated accuracy of the self-generated pseudo-labels for each question set, as determined using ChatGLM-Flash.

###### Performance of Evaluated Model (vs. Ground Truth)

Base Model Reasoner (Iter 1) Reasoner (Iter 2) Reasoner (Iter 3) Pseudo-Label Acc.

- DIter 1 39.0 44.0 45.5 49.0 72.0

- DIter 2 37.5 42.5 44.0 47.5 65.0

- DIter 3 36.0 40.0 41.5 45.0 61.0

follow similar upward trends, improving from 40.41 to

- 48.61 and 43.56 to 45.69, respectively. These results demonstrate the robust generalization ability and scalability of the proposed self-evolving framework across different models and model sizes.

- • Performance gains across diverse task types. VisPlay shows improvements across general visual understanding tasks, visual reasoning or math benchmarks, and are more robust to hallucination. For Qwen2.5-VL-3B, the Hallucination score rises from 32.81 to 94.95 by the second iteration, showing a substantial enhancement in factual grounding. Similar patterns are observed in other models—reasoning benchmarks consistently improve without compromising accuracy on general understanding tasks—demonstrating that VisPlay effectively strengthens both task-specific reasoning and cross-domain multimodal generalization.
- • Iterative co-evolution between the Questioner and Reasoner drives improvement. Performance trajectories across iterations highlight the co-evolution between the Questioner and Reasoner. As the Questioner generates more diverse and challenging queries, the Reasoner—trained with GRPO using high-quality silver supervision—learns to handle increasingly complex reasoning steps. This iterative loop allows both components to reinforce each other, leading to continual improvement in reasoning quality, generalization, and robustness. The results indicate that the co-evolutionary design of VisPlay provides a scalable path toward self-improving multimodal intelligence.

##### 3.3. Performance Comparison with HumanAnnotated Data

We conduct a performance comparison between models trained with VisPlay and those trained using human-curated image–question–answer pairs from the Vision-47K dataset under standard GRPO for one epoch, as shown in Table 3 for Qwen2.5-VL-3B and 7B. Although this experiment is not an ablation study in the strict sense, it provides a clear view of how our fully automated training pipeline performs relative to conventional supervised training. Overall, we observe that models trained with VisPlay achieve competitive average accuracy compared with those trained on real, human-written data. While performance on several task categories differs slightly, the general trend indicates that the self-evolving process can produce training signals of sufficient quality to improve base VLMs capabilities. These findings suggest that even in settings where human annotations are costly, limited, or unavailable, our framework can still serve as an effective and scalable alternative, enabling VLMs to develop stronger generalization abilities without depending on manual supervision.

##### 3.4. Co-Evolution Dynamics of Two Roles

• The Evolution of Question Difficulty and Solution Accuracy. To analyze the co-evolution dynamics of the two roles, we examine the changes in question difficulty (orange, left axis) and problem-solving accuracy (blue, right axis) across three VLMs during the first training iteration (Figure 3). Question difficulty is operationalized as the Reasoner’s model-perceived difficulty, derived from the

- Table 3. Performance comparison between VisPlay and standard GRPO training with human-labeled data. Although VisPlay relies entirely on self-generated supervision, it achieves competitive overall accuracy and significantly reduces hallucination. This demonstrates that our self-evolving framework can meaningfully enhance VLM performance even in the absence of human-annotated datasets.

General Visual Understanding Visual Math Hallucination Methods MMMU

MM RealWorld VisNum Math MATH Hallusion

Avg.

-Vet QA Bench Verse -Vision Bench Qwen2.5-VL-3B-Instruct

Standard GRPO 40.3 49.5 63.0 36.7 42.8 29.9 67.4 47.1

- VisPlay (Iter 3) 37.1 38.1 71.9 39.2 35.2 30.0 90.5 47.3

Qwen2.5-VL-7B-Instruct Standard GRPO 39.8 51.8 66.6 43 53.2 33.8 66.6 50.7

- VisPlay (Iter 3) 38.3 46.3 69.7 32.6 39.1 31.2 92.3 48.6

- Table 4. Examples of challenging questions generated by the self-evolving Vision-Language model across three training iterations. The questions progressively increase in complexity, illustrating the growth in difficulty of the Questioner’s outputs over iterations. Images on the left and right correspond to the visual context for each question. The questions are raised by Qwen2.5-VL-3B-Instruct.

###### Challenging Examples from Self-Evolving Trained Vision-Language Model

[Figure 10]

[Figure 11]

- Question (Iter 1) Approximately how many lung fields are visible in the X-ray image?

Which skeletal structure most likely belongs to a bird with hollow bones?

- Question (Iter 2) On a thoracic x-ray, the right lobe of the lung is more spread out compared to the left lobe. If the right lobe is given a score of 1 and the left lobe is given a score of 0, what is the difference in scores between the right and left lung lobes?

On which figure does the long neck of the dinosaur have the greatest horizontal angle with the vertical axis?

- Question (Iter 3) On which rib is the line approximately 2.5 cm above the image’s midpoint?

Which skeletal structure is most likely to have evolved secondary to flying abilities and which is less likely to have this trait?

confidence score defined in Eq. 4 in Section 2.3. Across all models, a consistent co-evolution pattern emerges. The Image-Conditioned Questioner’s difficulty curves exhibit a general upward trend, with initial increments followed by sustained growth, indicating the role’s ability to progressively formulate more challenging visual questions. Concurrently, the Multimodal Reasoner’s accu-

racy curves, despite minor fluctuations, show a complementary upward trajectory. This means that as question difficulty rises, the Reasoner adapts and enhances its problem-solving capability, with both metrics reinforcing each other’s improvement over iterations. Such mutual reinforcement validates VisPlay’s core mechanism, where the two interacting roles drive scalable self-evolution in

multimodal reasoning.

• The Evolution of Capabilities and Data Accuracy. Building on the interaction between question difficulty and solution accuracy, we further analyze how the Reasoner’s capabilities and the quality of pseudo-labeled data evolve across iterations. For each training iteration, the Questioner generates questions for the same 200 images, and Reasoners from each iteration attempt to answer them. As shown in Table 2, the Reasoner’s accuracy steadily improves across iterations (e.g., from 44.0 to 49.0 on first-iteration questions), while the estimated accuracy of pseudo-labels slightly declines (from 72 to 61), reflecting increasing question difficulty. These trends highlight the co-evolution of model reasoning ability and data complexity during self-improving training.

##### 3.5. Case Study on Question Difficulty Evolution

Table 4 presents example questions generated by the selfevolving Vision-Language Models across three training iterations. Iteration 1 questions focus on direct observation, such as counting or identifying objects. Iteration 2 introduces relational and comparative reasoning, requiring the model to assess differences or evaluate spatial angles. Iteration 3 further increases complexity with multi-step reasoning and inference, including precise localization and causal relationships. This progression demonstrates a systematic increase in question difficulty, providing increasingly challenging training signals that encourage the model to adapt and improve its reasoning capabilities. Such a design ensures that both the Questioner and Reasoner co-evolve, progressively enhancing the overall performance of the system.

#### 4. Related Work

Post-Training for Vision-Language Models Recent research in post-training of vision-language models (VLMs) has shifted from supervised fine-tuning (SFT) toward reinforcement learning (RL) paradigms, driven by the increasingly strong capabilities of pre-trained VLMs. Earlier works such as LLaVA [17, 27] primarily rely on SFT to align a language model backbone with a visual information via a projection layer, enabling multimodal instructionfollowing and visual reasoning. However, as base model quality improves, RL-based post-training has emerged as a more powerful alternative. In particular, R1-style training [10, 18, 53] has gained attention for its ability to enhance reasoning and visual understanding without explicit supervision. A key insight behind this success is that RL becomes effective only when the base model is sufficiently capable to self-explore reasoning trajectories [1, 7, 21, 22]. Despite these advances, most existing RL-based VLM approaches still depend on annotated multimodal datasets, which are costly and difficult to scale [11, 12, 24, 31, 42,

- 49]. To reduce reliance on human supervision, several re-

cent works explore VLM self-play paradigms in games. Vision-Zero [38] and Game-RL [36] train VLMs with simulated game data to improve their general reasoning ability. Nonetheless, these methods often continue to depend on external models or tools for training data generation. While large language models (LLMs) have demonstrated selfevolving learning dynamics without any human or modellabeled data, extending such paradigms to VLMs remains more challenging due to the additional visual modality.

Self-Evolving In Large Language Models Recent work has focused on enabling large language models (LLMs) to self-evolve their reasoning capabilities with minimal to zero human supervision. Various approaches have been proposed to achieve this. General unsupervised self-training frameworks like Genius [44] and Deep Self-Evolving Reasoning [29] aim for advanced reasoning. A common theme is data-free training or starting from zero data, as explored in R-Zero [16], Absolute zero [51], and Language self-play [19]. Other methods adapt self-play for specific goals or environments. For example, SPICE [26] improves reasoning in corpus environments, Search Self-play [30] pushes capabilities via search, and SPELL [46] focuses on evolving long-context models. Additionally, some methods utilize interactions between multiple agents to collectively bootstrap reasoning abilities, as seen in Socratic-Zero [39] and Multi-Agent Evolve [4].

#### 5. Limitation

While our work introduces a scalable, self-evolving framework, we acknowledge two primary limitations that suggest directions for future research. First, due to computational constraints, our experiments were limited to the Qwen2.5VL and MiMo-VL families. The scalability and effectiveness of VisPlay on significantly larger VLMs (e.g., ≥ 10B parameters) is still an important open question. Second, our framework lacks a definitive verification method for the self-generated data. While our GRPO policy indirectly optimizes for quality, developing more robust, automated methods to verify data faithfulness and prevent error accumulation is a key area for future investigation.

#### 6. Conclusion

We present VisPlay, a self-evolving RL framework that enables vision-language models to autonomously improve from unlabeled images. By decomposing a VLM into an Image-Conditioned Questioner and a Multimodal Reasoner and optimizing them via GRPO, our method balances challenge and accuracy without human supervision. Experiments show consistent gains in reasoning, compositional generalization, and hallucination reduction across multiple benchmarks. VisPlay demonstrates that scalable,

self-improving multimodal intelligence is achievable. By iteratively generating and learning from its own experiences, a model can refine its capabilities beyond humanlabeled data. This framework opens avenues for richer multimodal interactions and cross-domain adaptation, pointing toward intelligence systems that can continually evolve autonomously. Our results suggest a promising path toward truly autonomous vision-language systems that improve themselves over time.

#### References

- [1] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond, 2023. 8
- [2] Florian Bordes, Richard Yuanzhe Pang, Anurag Ajay, Alexander C. Li, Adrien Bardes, Suzanne Petryk, Oscar Ma˜nas, Zhiqiu Lin, Anas Mahmoud, Bargav Jayaraman, Mark Ibrahim, Melissa Hall, Yunyang Xiong, Jonathan Lebensold, Candace Ross, Srihari Jayakumar, Chuan Guo, Diane Bouchacourt, Haider Al-Tahan, Karthik Padthe, Vasu Sharma, Hu Xu, Xiaoqing Ellen Tan, Megan Richards, Samuel Lavoie, Pietro Astolfi, Reyhane Askari Hemmat, Jun Chen, Kushal Tirumala, Rim Assouel, Mazda Moayeri, Arjang Talattof, Kamalika Chaudhuri, Zechun Liu, Xilun Chen, Quentin Garrido, Karen Ullrich, Aishwarya Agrawal, Kate Saenko, Asli Celikyilmaz, and Vikas Chandra. An introduction to vision-language modeling, 2024. 1
- [3] Xiuwei Chen, Wentao Hu, Hanhui Li, Jun Zhou, Zisheng Chen, Meng Cao, Yihan Zeng, Kui Zhang, Yu-Jie Yuan, Jianhua Han, Hang Xu, and Xiaodan Liang. C2-evo: Coevolving multimodal data and model for self-improving reasoning, 2025. 1
- [4] Yixing Chen, Yiding Wang, Siqi Zhu, Haofei Yu, Tao Feng, Muhan Zhang, Mostofa Patwary, and Jiaxuan You. Multiagent evolve: Llm self-improve through co-evolution. arXiv preprint arXiv:2510.23595, 2025. 8
- [5] Zhipeng Chen, Xiaobo Qin, Youbin Wu, Yue Ling, Qinghao Ye, Wayne Xin Zhao, and Guang Shi. Pass@ k training for adaptively balancing exploration and exploitation of large reasoning models. arXiv preprint arXiv:2508.10751, 2025. 3
- [6] Daixuan Cheng, Shaohan Huang, Xuekai Zhu, Bo Dai, Wayne Xin Zhao, Zhenliang Zhang, and Furu Wei. Reasoning with exploration: An entropy perspective. arXiv preprint arXiv:2506.14758, 2025. 3
- [7] Xiangxiang Chu, Jianlin Su, Bo Zhang, and Chunhua Shen. Visionllama: A unified llama backbone for vision tasks,

2024. 8

- [8] Jeff Clune. Ai-gas: Ai-generating algorithms, an alternate paradigm for producing general artificial intelligence. arXiv preprint arXiv:1905.10985, 2019. 1
- [9] Runpeng Dai, Linfeng Song, Haolin Liu, Zhenwen Liang, Dian Yu, Haitao Mi, Zhaopeng Tu, Rui Liu, Tong Zheng, Hongtu Zhu, et al. Cde: Curiosity-driven exploration for efficient reinforcement learning in large language models. arXiv preprint arXiv:2509.09675, 2025. 3

- [10] DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. 8
- [11] Kaixuan Fan, Kaituo Feng, Haoming Lyu, Dongzhan Zhou, and Xiangyu Yue. Sophiavl-r1: Reinforcing mllms reasoning with thinking reward, 2025. 8
- [12] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms, 2025. 5, 8, 1
- [13] Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, Saizhuo Wang, Kun Zhang, Yuanzhuo Wang, Wen Gao, Lionel Ni, and Jian Guo. A survey on llm-as-ajudge, 2025. 5

- [14] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large visionlanguage models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14375–14385, 2024. 5, 1
- [15] Chengsong Huang, Langlin Huang, Jixuan Leng, Jiacheng Liu, and Jiaxin Huang. Efficient test-time scaling via selfcalibration. arXiv preprint arXiv:2503.00031, 2025. 3
- [16] Chengsong Huang, Wenhao Yu, Xiaoyang Wang, Hongming Zhang, Zongxia Li, Ruosen Li, Jiaxin Huang, Haitao Mi, and Dong Yu. R-zero: Self-evolving reasoning llm from zero data. 2025. 1, 8
- [17] Mengzhao Jia, Zhihan Zhang, Wenhao Yu, Fangkai Jiao, and Meng Jiang. Describe-then-reason: Improving multimodal mathematical reasoning through visual comprehension training, 2024. 8
- [18] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Searchr1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516,

2025. 8

- [19] Jakub Grudzien Kuba, Mengting Gu, Qi Ma, Yuandong Tian, and Vijai Mohan. Language self-play for data-free training. arXiv preprint arXiv:2509.07414, 2025. 8
- [20] Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. Tulu 3: Pushing frontiers in open language model posttraining. arXiv preprint arXiv:2411.15124, 2024. 2
- [21] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation, 2022. 8
- [22] Zongxia Li, Xiyang Wu, Hongyang Du, Fuxiao Liu, Huy Nghiem, and Guangyao Shi. A survey of state of the art large vision language models: Benchmark evaluations and challenges. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR) Workshops, pages 1587– 1606, 2025. 1, 8
- [23] Zongxia Li, Xiyang Wu, Ishani Mondal, Alexa Siu, Jordan Lee Boyd-Graber, and Ani Nenkova. LLM-as-a-judge failures at automating the identification of poor quality outputs in free-form texts. In Proceedings of The 5th New Frontiers in Summarization Workshop, pages 1–16, Hybrid, 2025. Association for Computational Linguistics. 5
- [24] Zongxia Li, Xiyang Wu, Guangyao Shi, Yubin Qin, Hongyang Du, Tianyi Zhou, Dinesh Manocha, and Jordan Lee Boyd-Graber. Videohallu: Evaluating and mitigating multi-modal hallucinations on synthetic video understanding, 2025. 8
- [25] Zongxia Li, Wenhao Yu, Chengsong Huang, Rui Liu, Zhenwen Liang, Fuxiao Liu, Jingxi Che, Dian Yu, Jordan Boyd-Graber, Haitao Mi, et al. Self-rewarding visionlanguage model via reasoning decomposition. arXiv preprint arXiv:2508.19652, 2025. 5, 1

- [26] Bo Liu, Chuanyang Jin, Seungone Kim, Weizhe Yuan, Wenting Zhao, Ilia Kulikov, Xian Li, Sainbayar Sukhbaatar, Jack Lanchantin, and Jason Weston. Spice: Self-play in corpus environments improves reasoning. arXiv preprint arXiv:2510.24684, 2025. 1, 8
- [27] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023. 8
- [28] Rui Liu, Dian Yu, Tong Zheng, Runpeng Dai, Zongxia Li, Wenhao Yu, Zhenwen Liang, Linfeng Song, Haitao Mi, Pratap Tokekar, et al. Vogue: Guiding exploration with visual uncertainty improves multimodal reasoning. arXiv preprint arXiv:2510.01444, 2025. 3
- [29] Zihan Liu, Shun Zheng, Xumeng Wen, Yang Wang, Jiang Bian, and Mao Yang. Deep self-evolving reasoning. arXiv preprint arXiv:2510.17498, 2025. 8
- [30] Hongliang Lu, Yuhang Wen, Pengyu Cheng, Ruijin Ding, Haotian Xu, Jiaqi Guo, Chutian Wang, Haonan Chen, Xiaoxi Jiang, and Guanjun Jiang. Search self-play: Pushing the frontier of agent capability without supervision. arXiv preprint arXiv:2510.18821, 2025. 8
- [31] Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai Yang, Xingzhong Xu, Xin Geng, and Xu Yang. Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl,

2025. 8

- [32] J¨urgen Schmidhuber. G¨odel machines: Fully self-referential optimal universal self-improvers. In Artificial general intelligence, pages 199–226. Springer, 2007. 1
- [33] Atharva Sehgal, Patrick Yuan, Ziniu Hu, Yisong Yue, Jennifer J. Sun, and Swarat Chaudhuri. Self-evolving visual concept library using vision-language critics, 2025. 1
- [34] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 2
- [35] Qwen Team. Qwen2.5-vl, 2025. 1, 2
- [36] Jingqi Tong, Jixin Tang, Hangcheng Li, Yurong Mou, Ming Zhang, Jun Zhao, Yanbo Wen, Fan Song, Jiahao Zhan, Yuyang Lu, Chaoran Tao, Zhiyuan Guo, Jizhou Yu, Tianhao Cheng, Zhiheng Xi, Changhao Jiang, Zhangyue Yin, Yining Zheng, Weifeng Ge, Guanhua Chen, Tao Gui, Xipeng Qiu, Qi Zhang, and Xuanjing Huang. Game-rl: Synthesizing multimodal verifiable game data to boost vlms’ general reasoning, 2025. 8
- [37] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset, 2024. 5, 1
- [38] Qinsi Wang, Bo Liu, Tianyi Zhou, Jing Shi, Yueqian Lin, Yiran Chen, Hai Helen Li, Kun Wan, and Wentian Zhao. Vision-zero: Scalable vlm self-improvement via strategic gamified self-play, 2025. 8
- [39] Shaobo Wang, Zhengbo Jiao, Zifan Zhang, Yilang Peng, Xu Ze, Boyu Yang, Wei Wang, Hu Wei, and Linfeng Zhang. Socratic-zero: Bootstrapping reasoning via data-free agent co-evolution. arXiv preprint arXiv:2509.24726, 2025. 8
- [40] Tengjin Weng, Jingyi Wang, Wenhao Jiang, and Zhong Ming. Visnumbench: Evaluating number sense of

- multimodal large language models. arXiv preprint arXiv:2503.14939, 2025. 5, 1
- [41] xAI. Realworldqa: Real-world spatial understanding benchmark. https://x.ai/blog/grok-1.5v-andrealworldqa, 2024. CC BY-ND 4.0 license. Benchmark dataset released with Grok-1.5 Vision. 5, 1
- [42] Jiaer Xia, Yuhang Zang, Peng Gao, Yixuan Li, and Kaiyang Zhou. Visionary-r1: Mitigating shortcuts in visual reasoning with reinforcement learning, 2025. 8
- [43] LLM-Core-Team Xiaomi. Mimo-vl technical report, 2025. 2
- [44] Fangzhi Xu, Hang Yan, Chang Ma, Haiteng Zhao, Qiushi Sun, Kanzhi Cheng, Junxian He, Jun Liu, and Zhiyong Wu. Genius: A generalizable and purely unsupervised selftraining framework for advanced reasoning. arXiv preprint arXiv:2504.08672, 2025. 8
- [45] Yuqing Yang, Yan Ma, and Pengfei Liu. Weak-to-strong reasoning. arXiv preprint arXiv:2407.13647, 2024. 1
- [46] Ziyi Yang, Weizhou Shen, Ruijun Chen, Chenliang Li, Fanqi Wan, Ming Yan, Xiaojun Quan, and Fei Huang. Spell: Selfplay reinforcement learning for evolving long-context language models. arXiv preprint arXiv:2509.23863, 2025. 8
- [47] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023. 5, 1
- [48] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556– 9567, 2024. 5, 1
- [49] Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao. R1-vl: Learning to reason with multimodal large language models via stepwise group relative policy optimization, 2025. 8
- [50] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, and Hongsheng Li. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems?, 2024. 5, 1
- [51] Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, and Gao Huang. Absolute zero: Reinforced self-play reasoning with zero data. arXiv preprint arXiv:2505.03335, 2025. 1, 8
- [52] Yulai Zhao, Haolin Liu, Dian Yu, Sunyuan Kung, Meijia Chen, Haitao Mi, and Dong Yu. One token to fool llm-as-ajudge. arXiv preprint arXiv:2507.08794, 2025. 2
- [53] Tong Zheng, Hongming Zhang, Wenhao Yu, Xiaoyang Wang, Runpeng Dai, Rui Liu, Huiwen Bao, Chengsong Huang, Heng Huang, and Dong Yu. Parallel-r1: Towards parallel thinking via reinforcement learning. arXiv preprint arXiv:2509.07980, 2025. 8
- [54] Yujun Zhou, Zhenwen Liang, Haolin Liu, Wenhao Yu, Kishan Panaganti, Linfeng Song, Dian Yu, Xiangliang Zhang,

Haitao Mi, and Dong Yu. Evolving language models without labels: Majority drives selection, novelty promotes variation. arXiv preprint arXiv:2509.15194, 2025. 3

[Figure 12]

## VisPlay: Self-Evolving Vision-Language Models from Images

### Supplementary Material

#### A.1 Detailed Training Dataset and Benchmarks

Training Dataset We use the image data from Vision47K dataset [12, 25], which contains 47,000 web-sourced images covering a wide variety of domains. The dataset includes charts, medical images, educational exams, textbook illustrations, and driving simulation frames. For our purposes, we exclusively use the images themselves, omitting any associated questions and answers. The dataset consists of approximately 10K charts, 8K medical images, 12K educational images (from exams and textbooks), 7K driving scenes, and 10K miscellaneous images from various domains. All images were standardized to a resolution of 224×224 pixels for model training.

Evaluation emphasizes spatial reasoning and contextual understanding.

• VisNumBench [40]: Focused on visual number sense, includes roughly 1.9K questions involving numerical attributes, comparisons, and estimations.

Multimodal Mathematical Reasoning Two specialized benchmarks:

- • MathVerse [50]: 2.6K diagram-centric questions covering geometry, functions, and algebra, provided in multiple visual-text formats.
- • MATH-Vision [37]: Approximately 3K competitionlevel problems across 16 subjects and five difficulty tiers. Focuses on integrating visual information into advanced mathematical reasoning.

Backbone Models and Training We trained three backbone models using VisPlay:

- • Qwen2.5-VL-3B-Instruct4: 3 billion parameters, finetuned with multimodal instruction data to enhance reasoning over visual-text tasks.
- • Qwen2.5-VL-7B-Instruct5: 7 billion parameters, trained under the same protocol with extended batch sizes and longer training schedules to improve complex reasoning and generalization.
- • Mimo-VL-7B-SFT6: 7 billion parameters, optimized with supervised fine-tuning on multimodal datasets for better alignment with human instructions.

General Visual Understanding Four established benchmarks are used:

- • MM-Vet [47]: Evaluates recognition, OCR, and visual math abilities using a unified LLM-based scoring metric. The dataset contains over 5,000 test samples with detailed scoring for each subtask.
- • MMMU [48]: Cross-modal reasoning benchmark with 11.5K college-level multiple-choice questions spanning six academic disciplines. Each question is image-based and designed to test subject knowledge and reasoning ability.
- • RealWorldQA [41]: Contains approximately 700 realworld images paired with spatially grounded questions.

- 4https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
- 5https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- 6https://huggingface.co/XiaomiMiMo/MiMo-VL-7B-SFT

Visual Hallucination Detection HallusionBench [14] is used to identify model errors caused by either languageonly hallucinations or visual illusions. Evaluation is conducted in a simple yes/no format, enabling precise measurement of hallucination rates and error types.

#### A.2 Training Configuration

Image-based Questioner Configuration. The Questioner is trained using a vision–language model with a maximum context window of 8192 tokens. The training set consists of 47K multimodal samples (Vision-SR1-47K), and evaluation is performed on the MMStar benchmark. Each sample uses problem, answer, and images as the prompt, label, and image fields, respectively. During rollouts, the Questioner generates 8 candidate questions per input. For the 3B model, we train for 20 steps, and for the 7B model, we train for 10 steps. In this setup, four GPUs run the vLLM service to provide reward signals, while the other four GPUs are used for training. The model parameters are loaded from a specified Questioner checkpoint, and all checkpoints are saved under the designated experiment directory. Validation before training is disabled to maximize efficiency during early-stage learning.

Multimodal Reasoner Configuration. The Solver is trained using chain-of-thought reinforcement learning. Its output length is capped at 4096 tokens, and prompts are constructed using a dedicated Jinja template to enforce a consistent reasoning format. Training uses the self-play

###### Image-Conditioned Questioner Prompt Template

You are an intelligent Question Generator. Your task is to create a question based on the given image. Requirements (must follow exactly):

1. Analyze the image carefully and understand all details. 2. Generate exactly one question that is directly related to the image. 3. Choose the question type from only one of the following: - multiple choice (Yes/No or four options labeled A, B, C, D; only one correct answer) - numerical (requires a specific numeric answer) - regression (requires predicting a continuous value, such as a measurement, quantity, or coordinate) 4. The question must require analysis or reasoning, not just description. 5. Output must be strictly in format < question > X < /question >, with nothing else: Strict rules: - Do not use any other labels, punctuation, or formatting. - Do not add commentary, explanations, or extra text. Example of correct output: < question > How many clubs are there < /question >

###### Multimodal Reasoner Prompt Template

Please reason step by step carefully based on the question: + content + and the image. After completing your reasoning, you MUST output the final, clean, and concise answer strictly inside + \boxed{} + . The final answer MUST appear inside \boxed{}, and nowhere else. If there is no boxed answer, your response is considered incorrect.

###### LLM-as-Judge Prompt Template

You are an answer evaluation assistant. Your task is to judge whether two answers are substantially equivalent. When evaluating, you should ignore superficial differences such as format, spaces, punctuation, case, etc., and focus on whether they are consistent in core content, logical meaning and information expression. The judgment criteria should be lenient and inclusive, as long as the expressed meaning is basically the same, it is considered equivalent.

dataset produced by the Questioner, while evaluation again uses MMStar. To ensure stable learning under long sequences, we adopt a conservative micro-batch size of 1 for both updates and experience rollouts. The rollout engine supports up to 20K batched tokens per forward pass. The number of training steps for the Solver is set to be the same as for the Questioner.

#### A.3 Prompt Templates

The following three prompt templates define the core interaction structure used in our self-evolving Vision-Language Model. In this setup, multiple specialized roles—such as a question generator, a multimodal reasoner, and an evaluator—are orchestrated to form an autonomous learning loop. Each template specifies precise behavioral constraints that allow the model to generate tasks, solve them with step-bystep reasoning, and assess answer consistency. Together, these components establish a controlled self-play environment that enables the model to iteratively refine its reasoning capabilities without relying on external supervision.

