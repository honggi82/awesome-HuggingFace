## arXiv:2604.04917v3[cs.CV]18Jun2026

### Vero: An Open RL Recipe for General Visual Reasoning

Gabriel Sarch∗ Linrong Cai∗ Qunzhong Wang Haoyang Wu Danqi Chen Zhuang Liu†

Princeton University

###### Models Data: 600K & 1.6M Code Project Page

[Figure 1]

Vero-600k (Ours) OpenMMReasoner-74k

ViRL-39k LLaVA-OV-1.5-RL-70k Vero Initial RL Model

Captioning & Instruction Following

Grounding, Counting & Search

Overall

Overall

+2.9

(30 evals)

57

(30 evals)

74.4

71.5

70

+5.4

+3.5

Performance

###### Vero-600k

Performance

66.1

+3.6

65.8

Chart & OCR

STEM

&

63.2

62.3

###### VeroEval

50

60.7

+4.9

59.6

60

57.8

Spatial & Action

Knowledge & Recognition

52.9

50

42

Qwen3-VL-I-8B

Qwen2.5-VL-7B

MiMo-VL-7B

Qwen3-VL-T-8B

Qwen3.5-9B

0 600

RL FLOPs

Qwen3I-8B Vero-Qwen3I-8B Qwen3.5-9B Vero-Qwen3.5-9B

70.5

80.5

60.7

75.0

76.7

87.4

78.1

73.3

68.2

58.2

71.0

83.5 83.3

63.4

69.5

67.2

64.0

53.8

78.2

52.3

58.5

57.3

62.6

61.2

Grounding, Counting & Search (8 evals)

Chart & OCR (6 evals)

Knowledge & Recognition (4 evals)

Spatial & Action (5 evals)

STEM (4 evals)

Captioning & IF (3 evals)

Figure 1. Vero achieves state-of-the-art performance across six task categories using a fully open RL recipe. Top left: Training curves versus RL FLOPs for Vero-600K compared with existing open RL datasets, all finetuned from Qwen2.5-VL-7B-Instruct; dashed lines indicate training beyond one epoch. Reported is mean ± sem (3 seeds). Top center: Summary of the broad task categories targeted in Vero-600K and VeroEval. Top right: Overall performance of Vero (Avg@5) against each initial RL model across the 30 VeroEval benchmarks. Bottom row: Same as top right, but displaying per-category scores for Vero trained on Qwen3-VL-8B-Instruct and Qwen3.5-9B.

###### Abstract

What does it take to build a visual reasoner that works across charts, science, spatial understanding, and open-ended tasks? The strongest vision-language models (VLMs) suggest that broad visual reasoning is within reach, yet their closed data and reinforcement learning (RL) pipelines make their gains difficult to study, reproduce, or extend. We introduce Vero, a family of fully open VLMs that match or exceed existing open-weight models across diverse visual reasoning tasks. We scale RL data and rewards across six broad task categories, constructing Vero-600K, a 600K-sample dataset from 59 datasets, and designing task-routed rewards that handle heterogeneous answers. Across VeroEval, our 30-benchmark suite, Vero-600K outperforms existing RL datasets under controlled comparisons. Applied to five starting models, Vero variants gain 2.9–5.4 points on average over their initial models. Notably, Vero-Qwen3I-8B, trained on the Instruct model, surpasses Qwen3-VL8B-Thinking by 3.8 points on average without additional distillation. Systematic ablations reveal that different task categories elicit distinct reasoning patterns and that broad gains depend on learning them jointly rather than in isolation. All data, code, and models are publicly available.

∗ Project Leads † Corresponding Author 1

G

r

o

u

n

d

Grounding, Counting & Search

###### Knowledge & Recognition VQA

###### STEM

i

F

n

I

g

&

,

C

g

o

n

u

i

n

n

CoSyn-Math

o

t

i

i

n

t

p

g & S e a r c h

AerialVG

IconQA

a

Geo170K

C

Objects365-QA

KVG

VisualWebInstruct

PixMo

PopVQA

TQA

TallyQA

VCR

GeomVerse

1

k

GroundUI

Visual7W

0

0

0

- 0

k

1 0 0 k

- 1

1

RAVEN MMK12 GeoQA+

OS-ATLAS

VQAv2

RefCOCOg

KVQA

C h a r t

MultiHop

GQA

We-Math 2.0 Std We-Math 2.0 Pro AI2D

Visual Probe

VizWiz

M

Total 600k RL Samples

1 0 0 k

E

OOD-VQA

A-OKVQA Indoor-QA ViQuAE

& O C R

- S
- T

Pixel Reasoner Chart & OCR

PathVQA

VQA-RAD

###### Spatial & Action GameQA

CoSyn-Table

1

k

0

0

ArxivQA

0

0

###### Captioning & IF

k

ChartQA

Spatial-SSRL

PixMo-AskModelAnything

A

ECD-VQA

- Visual Jigsaw 2D

- Visual Jigsaw 3D

Q

V

PixMo-CapQA

EvoChart

n

o

PixMo-Cap

S

i

t

InfographicVQA

Magma-AITW

i

p

n

a

g

t

o

i

- MM-RLVR-IFEval

- MMIF-23K

a

c

CoSyn-Chart

ST-VQA

l & A c t i o n

e

R

&

- d

g

- e

ReachQA

Magma-Mind2Web

e

l

w

- n
- o

K

CoSyn-Diagram

Robo2VLM

Flickr30K

- Figure 2. Composition of Vero-600K. The inner ring shows six task categories, each allocated 100K samples (600K total), and the outer ring shows their 59 constituent datasets. The categories represent real-world use cases and cover distinct visual reasoning capabilities (Sections 5 and 7). Categories are uniformly sampled to balance learning across tasks.

##### 1 Introduction

Vision-language models (VLMs) are increasingly expected to reason across a wide range of visual tasks, from chart and scientific interpretation to spatial understanding and open-ended questions. Reinforcement learning (RL) has emerged as a key driver of this progress, with methods such as PPO (Schulman et al., 2017) and GRPO (Shao et al., 2024) enabling models to learn from their own generations through reward signals. Recent multimodal models such as GPT-5 (Singh et al., 2025), Qwen3-VL (Bai et al., 2025a), and Kimi K2.5 (Kimi Team et al., 2026) demonstrate that RL drives substantial improvement in multimodal reasoning.

Yet the strongest existing visual reasoning models are products of proprietary RL pipelines with non-public data and undisclosed reward designs. Models such as Qwen3-VL (Bai et al., 2025a) release weights and are widely adopted, but do not release RL training code or datasets. Accompanying technical reports often omit detailed ablations of design choices, making it difficult to systematically study what drives performance. Meanwhile, fully open efforts such as OpenMMReasoner (Zhang et al., 2026) and VL-Rethinker (Wang et al., 2025) focus primarily on visual math, covering only a narrow subset of visual tasks. However, as we show in Sections 5 and 7, training on a single task category does not generalize to other visual capabilities, in both task performance and chain-of-thought behavior. More broadly, applying RL across heterogeneous visual reasoning tasks is challenging, as diverse task mixtures induce interference, weak transfer, and optimization imbalance unless the training distribution and rewards are carefully designed (Teh et al., 2017; Schaul et al., 2019; Hessel et al., 2019). This leaves a central question: what does it take to train a broadly capable visual reasoner?

We show that a single-stage RL recipe with diverse and high-quality data suffices. We introduce Vero, a family of fully open VLMs trained with RL on top of existing models to perform strongly across diverse visual tasks. Our recipe centers on careful dataset selection, sample filtering, and task-balanced mixing: we build Vero-600K, a 600K-sample training set from 59 datasets spanning six core task categories (Chart & OCR; STEM; Spatial & Action; Knowledge & Recognition; Grounding, Counting & Search; and Captioning & Instruction Following), and pair it with task-routed reward functions.1 No additional warm start, no staged RL, and no proprietary data. Alongside the training data, we assemble VeroEval, a comprehensive evaluation suite of 30 benchmarks spanning all six categories. Figure 2 summarizes the composition of Vero-600K.

1We also release Vero-1.6M, a 1.6M extension built with the same process, while all experiments use Vero-600K unless specified.

Through systematic ablations of dataset selection, sample filtering, mixture strategies, and reward design, we find that data diversity is the critical ingredient. Different task categories elicit qualitatively distinct reasoning patterns that transfer poorly in isolation: for example, STEM tasks trigger elevated backtracking while grounding tasks suppress introspective behaviors in favor of directed visual search. Producing a generally capable model therefore requires broad task coverage, so that the model learns these distinct reasoning patterns jointly rather than in isolation. We additionally find that (1) uniform mixture weighting across task categories outperforms schemes based on accuracy, reasoning length, or image size; (2) multi-task training necessitates an expressive, task-routed reward design; and (3) including open-ended tasks is necessary to preserve visual chat ability during RL.

Vero achieves strong overall performance across model families (Figure 1). Training on six different initial models yields consistent improvements, with gains of +2.9 to +5.4 points over five post-trained models, averaged over 30 benchmarks. Applied directly to the pretrained-only Qwen3.5-9B-Base, our recipe yields +12.9 points, reaching 73.0 overall, second only to Vero-Qwen35-9B, without any SFT or distillation warm start. Vero-Qwen35-9B, trained from Qwen35-9B, reaches 74.4 overall and improves over its initial model by +2.9, with gains on 25 of 30 benchmarks and all six category averages. Among 8B models, Vero-Qwen3I-8B reaches 66.1 overall and outperforms Qwen3-VL-8B-Thinking by +3.8 overall and on 25 of 30 benchmarks, while Vero-Qwen3T-8B improves over Qwen3-VL-8B-Thinking by +3.5 overall and on 22 of 30 benchmarks. Compared with distillation warm start baselines, Vero-Qwen25-7B exceeds OpenMMReasoner-7B by +5.6 overall despite OpenMMReasoner using a 874K example teacher distillation warm start. We release all data, code, and models to facilitate future research.

##### 2 Related Work

Vision-language models. Vision-language models excel on multimodal tasks, including proprietary systems such as GPT-5 (Singh et al., 2025) and Gemini (Team et al., 2023, 2024; Comanici et al., 2025), open-weight families such as Qwen (Bai et al., 2025b,a), GLM (Hong et al., 2025), and Kimi (Kimi Team, 2025), and fully open releases of data, code, and weights such as Molmo (Deitke et al., 2025; Clark et al., 2026) and LLaVA (Liu et al., 2023; An et al., 2025). These models are expected to handle a wide range of tasks. While little is publicly known about proprietary model post-training, recent open-weight models have explored techniques such as RL with curriculum sampling (Hong et al., 2025) and mixed on-policy RL (Yue et al., 2025b), yet the factors that drive their performance across diverse tasks remain unclear. Our work targets this gap by providing a fully open multi-domain RL recipe for general visual understanding.

Reasoning and thinking for VLMs. Chain-of-thought reasoning enables models to use additional test-time compute through step-by-step problem decomposition (Wei et al., 2022; Zhang et al., 2024). The two dominant approaches for training reasoning models are distillation, where a strong teacher generates reasoning traces for supervised fine-tuning (Xu et al., 2025; Yao et al., 2025; Sarch et al., 2025), and reinforcement learning, which optimizes against outcome-based rewards without requiring a fixed teacher (DeepSeek-AI et al., 2025). Recent works apply RL to visual reasoning (Yu et al., 2025a; Wang et al., 2025; Zhang et al., 2026; Feng et al., 2025), but primarily in narrow domains, leaving the effect of RL-trained reasoning on broad visual understanding underexplored. We show that RL with careful reward and data design consistently outperforms narrowly trained baselines across diverse visual task categories.

RL recipes and training data design for VLMs. Several works provide recipes for RL-based visual reasoning training. OpenMMReasoner (Zhang et al., 2026) combines teacher distillation and GSPO (Zheng et al., 2025a) over multimodal reasoning benchmarks, OneThinker (Feng et al., 2025) uses a distillation warm start before RL, VL-Rethinker (Wang et al., 2025) addresses training instability via selective sample replay and forced rethinking, and Perception-R1 (Yu et al., 2025a) designs discriminative rewards for perceptual tasks. These efforts primarily target visual math or narrow perceptual domains and provide only limited ablations of dataset selection, sample filtering, and reward design. Our recipe centers on Vero-600K, which spans six task categories with 600K data points from 59 datasets, includes a routed reward system, and provides systematic ablations of design choices, all released publicly to support open VLM research.

Data Sourcing Task Categories

Selection and Filtering Data Mixing

RL Training

|Heuristic Selection<br><br>1<br><br>Manual Quality Control<br><br>2<br><br>Question Filtering<br><br>3<br><br>Answer Filtering<br><br>4|
|---|

|Policy<br><br>π<br><br>|Reward score|
|---|
<br><br>|
|---|

250+

Datasets

- Figure 3. Vero-600K data curation pipeline. Starting from over 250 candidate datasets, we assign each to one of six task categories and apply multi-stage selection and filtering: heuristic screening (size, resolution, answer format), manual quality control, LLM-based question filtering for ambiguity and verifiability, and answer filtering for stable reward computation. The retained data are combined into a uniformly weighted mixture across task categories and used for on-policy RL training with task-routed rewards.

##### 3 Vero

###### 3.1 Task Categories

We consider the problem of training a Vision-Language Model (VLM) πθ via reinforcement learning to maximize expected reward across a diverse set of visual reasoning tasks. Given a visual input v (an image or

set of images) and a text query q, the model generates a structured response y = (z, a) ∼ πθ(· | v, q), where z denotes the reasoning or thinking content and a denotes the final answer. We verify the final answer a against the ground-truth answer y∗. The RL training objective is:

E(v,q,y∗)∼D E(z,a)∼πθ(·|v,q) [R(a, y∗)] , (1)

max

θ

where D is the training data distribution. A central challenge is constructing D to span a broad range of visual reasoning capabilities, so that the resulting policy generalizes across diverse tasks.

Figure 2 provides an overview of our training data composition. We organize our training data into six task categories, each targeting a distinct visual reasoning capability. This taxonomy is motivated by two observations. First, we find empirically (Section 5 and Section 7) that training on any single category fails to transfer reliably to others and elicits distinct chain-of-thought behaviors, suggesting that these categories exercise different reasoning strategies and skills. Second, while existing VLM evaluation frameworks organize benchmarks along similar axes, e.g., Qwen2.5-VL (Bai et al., 2025b) separates document understanding, mathematical reasoning, and grounding, and Kimi K2.5 (Kimi Team et al., 2026) distinguishes reasoning from perception, these categorizations are typically adopted by convention rather than validated empirically, and they are designed for evaluation rather than training. Our taxonomy refines and extends these axes to cover a broader set of visual reasoning tasks, and we validate its effectiveness for multi-task RL (Section 5).

Concretely, we define six categories: STEM (13 datasets) covers mathematical diagram reasoning, scientific figure interpretation, and medical image understanding, with answers that are typically numeric or symbolic. Spatial & Action (8 datasets) targets embodied reasoning, UI navigation, and 3D spatial understanding, requiring reasoning about spatial transformations and action sequences. Knowledge & Recognition (12 datasets) spans visual question answering that combines object, scene, and entity recognition with external or commonsense knowledge. Chart & OCR (9 datasets) focuses on extracting and reasoning over structured information in documents, charts, tables, and infographics. Grounding, Counting & Search (11 datasets) requires spatially localizing objects via bounding boxes, counting entity instances, and searching among visual distractors. Captioning & Instruction Following (6 datasets) encompasses open-ended image description and following prompt instructions.

###### Chart & OCR

###### Spatial & Action

Q: What was the crude suicide rate in Brunei in 2015?

Q: Is there any obstacle blocking the robot from reaching drawer? A. Cannot be determined B. No C. Yes D. Partially reachable

|[Figure 2]|
|---|

|[Figure 3]|
|---|

A: 1.3

A: B

###### STEM

###### Knowledge & Recognition

Q: In the figure provided, lines AB, CD, and EF are parallel (AB || CD, CD || EF). A transversal intersects these lines creating angles labeled x, y, and z. If the ratio of angle y to angle z is 3:7 (y : z = 3 : 7), what is the measur…

Q: What kind of bird is this? A: Seagull

|[Figure 4]|
|---|

|[Figure 5]|
|---|

A: 126

###### Grounding, Counting, and Search

###### Captioning & Instruction Following

Q: Where are the Surveillance Cameras located? Output the bounding box(es) in a JSON array format.

Q: Assume this is an image you are about to post on Twitter. Please provide a short, upbeat caption describing it.

|[Figure 6]|
|---|

|[Figure 7]|
|---|

A: [{"bbox_2d": [550, 34, 570, 54], "label": "Surveillance camera"}, {"bbox_2d": [315, 134, 324, 141],…

A: Step into serenity with this stunning blend of modern elegance and rustic charm! The open glass doors invite t…

- Figure 4. Examples from each task category illustrate the breadth of Vero-600K. We show representative samples of our training data from the six categories, highlighting the diversity of visual inputs, question formats, and answer types covered by our training set.

###### 3.2 Vero-600K: Sourcing, Filtering, and Mixtures of Broad Tasks

We construct Vero-600K, a multi-task RL training set of 600K samples from 59 datasets spanning six task categories (Section 3.1). Figure 3 summarizes the Vero-600K data curation pipeline, and Figure 4 shows representative examples from all six task categories.

- Step 1. Dataset sourcing and selection. We start from over 250 candidate datasets drawn from instructiontuning and RL collections (e.g., FineVision (Wiedmann et al., 2025)) and recently released task-specific sources (e.g., Visual Jigsaw (Wu et al., 2025a)), then apply dataset-level filtering. Each dataset is assigned to the task category that best reflects its primary skill, based on manual inspection and its utility in prior work.

Heuristic selection. We discard datasets with fewer than 1K examples, average image resolution below 200K pixels (retaining five low-resolution datasets for question quality), or binary questions to mitigate guessing.

Manual selection. For each candidate, we inspect ∼50 examples against three criteria: correctness (<5% annotation error rate in image–question–answer triples), unambiguity (each question admits a single verifiable answer), and verifiability (the answer format is compatible with our reward functions). Of ∼100 datasets passing heuristic screening, 59 were retained. For a small number of datasets, we additionally rewrite questions to fix prompt clarity (e.g., GameQA, Magma) or drop high-error subsets.

Figure D1 (Appendix) compares dataset selection strategies and shows our filtering has significant gains compared to taking a random subset from the candidate pool or sampling from the STEM or Chart subsets of the FineVision (Wiedmann et al., 2025) training set. On STEM, our selected mixture achieves an average benchmark gain of +4.6, compared to +2.9 for FineVision and −0.2 for random sampling from all candidates. On Chart & OCR, the gains are +3.4, +1.9, and +2.5, respectively.

Knowl.&

Spatial&

Chart&

Action

Cnt.&

Bench.

Recog.

Search

Grnd.,

Knowl.&

STEM

Spatial&

OCR

Avg.

Chart&

Action

Cnt.&

Recog.

Search

Grnd.,

STEM

OCR

equal ratios +8.6 +6.2 +5.6 +1.8 +5.6 +5.8 ratio ∝ (1 − acc.)α +6.8 +6.5 +4.3 +2.4 +5.2 +5.2 ratio ∝ areaα +7.0 +5.3 +4.1 +1.4 +6.2 +5.2 ratio ∝ lengthα +7.5 +6.4 +4.5 +1.7 +3.8 +4.8 w/o Knowl. & Recog. +6.4 +6.5 +4.8 +1.9 +4.7 +4.9

Unfiltered 60.0 45.4 56.3 62.5 54.7 Q. Filtering 60.1 43.6 58.2 63.0 54.1 A. Canonic. 59.9 45.5 – 64.6 –

Table 1. Filtering generally helps remove ambiguous samples from noisy datasets. Effect of question filtering and answer filtering on Qwen2.5-

- VL-7B-Instruct. This table shows category average scores on VeroEval.

Table 2. Equal task ratios perform best overall. Weighting schemes: equal ratios (uniform), difficulty-weighted by inverse accuracy, image-area-weighted by mean input resolution, reasoninglength-weighted by mean chain-of-thought length, and ablation dropping Knowledge & Recognition. Values are absolute score changes (∆) on VeroEval when using Qwen3-VL-8B-Instruct as the initial model for RL.

- Step 2. Data filtering. After dataset-level filtering, many individual examples remain ambiguous, unanswerable, or incompatible with our rewards. We apply additional steps to filter individual prompts.

Question filtering. We use Qwen3-VL-235B-A22B-Instruct (Bai et al., 2025a) to remove ambiguous, imageirrelevant, or unverifiable questions. The model scores each datapoint on five criteria: (1) relevance, whether the image depicts what the question refers to; (2) ambiguity, whether the question is too vague or not a genuine question; (3) language, whether the question is in English; (4) verifiability, whether a single objectively correct answer can be derived from visible content; and (5) numeric precision, whether the required precision is visually unambiguous. Any triggered criterion removes the datapoint. We provide details in Appendix A.3.

Answer filtering. We normalize ground-truth answers using text-only Qwen3-235B-A22B-Instruct (Bai

- et al., 2025a) to ensure stable reward computation. Numeric answers are stripped of units and currency symbols, converted to decimal form, and evaluated as expressions. Samples with unsupported notation are filtered. Multiple-choice answers are normalized to a single canonical letter. Except for the captioning and instruction following task category, samples with multi-value answers, non-reducible symbolic expressions, or ambiguous descriptions requiring semantic matching are removed. A full list of answer filtering rules is in Appendix A.4. We do not apply it to Spatial & Action or Grounding tasks, since answers are already standardized for all datasets in these categories.

Effects of filtering are mixed across categories (Table 1): question filtering yields a clear gain on Spatial & Action (+1.9 pts) but slightly hurts Grounding, Counting & Search (−0.6 pts), while answer filtering substantially improves Knowledge & Recognition (+2.1 pts) but is flat or marginally negative on other categories. Despite this variance, we apply both steps to all applicable task categories, as they generally help remove ambiguous samples from noisy datasets and the largest gains outweigh the small regressions.

- Step 3. Data mixtures. In our multi-task RL setting, the task category sampling distribution governs how training signal is allocated across skills. We investigate four task category weighting schemes (uniform, difficulty-weighted, image-size-weighted, reasoning-length-weighted), where the number of samples per batch is determined by a ratio proportional to the metric (e.g., difficulty). Uniform sampling achieves the highest benchmark average gain (+5.8 pts over the base model), outperforming alternative schemes. Alternatives yield gains on individual categories but at the cost of others (Table 2). We use uniform task category weighting, as it achieves the best overall performance.

VeroEval evaluation suite. We introduce VeroEval, a challenging evaluation suite for broad visual reasoning. We curate a suite of 30 benchmarks spanning the six visual reasoning categories defined in Section 3, with three to eight benchmarks per category. We select benchmarks according to three criteria: (i) difficulty: we

favor benchmarks on which current frontier models have room for improvement, while retaining established benchmarks (e.g., ChartQA, ScreenSpot) for comparability with prior work; (ii) annotation quality: we include only benchmarks with well-defined evaluation protocols and reliable ground-truth labels; and (iii) intracategory diversity: within each category we include benchmarks that test complementary sub-skills (e.g., within Chart & OCR: chart reasoning, infographic understanding, and scientific figure interpretation). The full benchmark list appears in Appendix Table A2.

###### 3.3 Training Vero with Reinforcement Learning

Algorithmic details. At its core, RL maximizes the expected reward of the model’s response y given a visual input v and query q. Our RL algorithm builds on Group Relative Policy Optimization (GRPO) (Shao et al.,

- 2024) and integrates algorithmic advances from GSPO (Zheng et al., 2025a), among others (Yu et al., 2025b).

GSPO (Zheng et al., 2025a) replaces the independent per-token importance ratios of GRPO with a sequencelevel ratio. For each response yi in a group of G rollouts, the sequence-average log-probability difference ∆¯ i = |yi| ∑t(log πθ(yi,t) − log πθold(yi,t)) is used to form a token-level ratio si,t(θ) = exp(sg(∆¯ i) + log πθ(yi,t) − sg(log πθ(yi,t))), where sg denotes stop-gradient. The GSPO objective is then:

- 1

|yi|

G

1 G

1 |yi|

#### ∑

#### ∑

min si,t(θ) Ai, clip si,t(θ), 1−εlow, 1+εhigh Ai , (2)

J (θ) =

t=1

i=1

where Ai = (ri − µg)/(σg + ϵ) is the normalized group advantage; note that in our setting Ai,t = Ai for all t, i.e., the advantage is constant across tokens within a response. We adopt asymmetric clip-higher (Yu

- et al., 2025b) (εhigh > εlow), remove the KL penalty (Yu et al., 2025b; Liu et al., 2025b) to allow less-restricted updates, and apply a soft overlong penalty (Yu et al., 2025b) that linearly ramps before the context limit. Reward formulation. The total reward for a response y is (α = 0.2):

R(y, y∗) = (1 − α) Racc(y, y∗) + α Rfmt(y) + Roverlong(y), (3)

Overlong penalty. To discourage excessively long responses, we use the soft penalty from Yu et al. (2025b) as a linear ramp in the buffer zone [Lmax − B, Lmax] (B = 2048, Lmax = max_tokens, and λ = 1.0):

Roverlong(y) = min −|y|−(LBmax−B) λ, 0 , (4)

Format reward. Rfmt requires the response to follow the format <think>. . .</think><answer>. . .</answer> with non-empty think content; responses that violate this structure receive Rfmt = 0. Given valid structure, Rfmt = 1 by default. For discrete symbolic answer types (string match, multiple choice, numeric, list match, counting, ordering, search, web action), a single valid \boxed{...} in the answer block is additionally required for Rfmt = 1; its absence or the presence of multiple \boxed{...} expressions reduces Rfmt to 0.5.

Multi-task reward. For Racc(y, y∗), we detail below the ten reward functions corresponding to the task types in our dataset (Figure 5). We show in Section 4.2 that our reward design outperforms simple alternatives.

- • String match (∈ {0,1}): normalized exact-string equality.
- • Multiple choice (∈ {0,1}): extracts a single letter (A–Z) and compares it to the predicted letter.
- • Numeric (∈ {0,1}): symbolic parsing via MATH-VERIFY (Kydlícˇek, 2025), with optional tolerance.
- • List string match (∈ {0,1}): any-match across a set of strings, handling synonym-equivalent answers.
- • Ordering (∈ [0,1]): full reward for exact list order and partial reward (discounted by a factor of 0.2) for correct set with wrong order. Adapted from Visual Jigsaw (Wu et al., 2025a).
- • Web action (∈ [0,1]): weighted match over structured JSON fields (ACTION, MARK, VALUE), with score equal to the fraction of non-null gold fields correctly predicted. Adapted from ViGoRL (Sarch et al., 2025).

Overview of Reward Types

[Figure 8]

Short Answer

Web Action

Binary String Match

Ordering Binary Order Score

Discrete Field Match

Numeric Binary Math Verify

[Figure 9]

[Figure 10]

Order points from closest to farthest from the camera

Compute the length of the AC side of the ABC right triangle.

Where does the solid blue line drop to its lowest?

[Figure 11]

Goal: Go to display settings.

GT: week 2 Pred: Week 1

GT: 1, 4, 2, 6, 5, 3 Pred: 1, 4, 5, 2, 6, 3

GT: Click Element: 4 Pred: Click Element: 2

GT: 12*√(3) Pred: (12)*√(3)

Instruction Following

Open Ended

Grounding Discrete IoU / F1

Pointing Binary Point in Box

Discrete Rules

Discrete LLM Judge

[Figure 12]

[Figure 13]

[Figure 14]

Locate the 'CONTACT US'. Output a point.

Tell me about this image in exactly two paragraphs

[Figure 15]

Where is the Clock? Output a bbox.

Describe the image in detail.

GT: [{"bbox_2d": [473, 210, … Pred: [{"bbox_2d": [478, 215, …

GT: [{"bbox_2d": [893, 40, 9… Pred: [{"point_2d": [963, 22]}]

GT: {"functions": ...} Pred: This picture contains ...

GT: This image captures a ... Pred: The image shows ...

- Figure 5. Accuracy verifiers for our multi-task reward. Each card illustrates a verifier used in Vero, with an example visual question, ground-truth answer, and model prediction. Verifiers include binary rewards (string match, multiple choice (not displayed), list string match (not displayed), numeric, ordering, point-in-box) and graded rewards (IoU/F1 for grounding, field match for web actions, rule-based checks for instruction following, and LLM-as-judge for open-ended responses). This task-routed design enables accurate reward computation across diverse answer formats.

- • Grounding (∈ [0, 1]): optimal Hungarian matching of predicted and ground-truth bounding boxes, scoring IoU/F1 with threshold 0.5. Bounding box coordinates are normalized to the [0,1000] range (Qwen-style). Adapted from Perception-R1 (Yu et al., 2025a).
- • Clicking (∈ [0,1]): checks whether the predicted click point falls within the ground-truth bounding box region, with coordinates in the same normalized space. Adapted from ViGoRL (Sarch et al., 2025).
- • Instruction following (∈ [0,1]): proportion of programmatically defined output constraints satisfied (e.g., length limits, format requirements, keyword inclusions). We use the constraint checks from MMIFeval (Ding et al., 2025) and RLVR-IFeval (Pyatkin et al., 2025).
- • LLM-as-judge (∈ [0, 1]): We adapt the judge setup from OLMo3 (OLMo Team et al., 2025). We use Qwen332B with thinking disabled to score the response against an optional reference answer. The judge prompt instructs the model to score 1–10 and explicitly penalizes self-evaluative language and meta-commentary to reduce reward hacking. See Appendix B.3 for the full prompt.

##### 4 Experiments

Evaluation settings. We use the following decoding setups. Qwen2.5-VL and MiMo-VL trained models follow the Qwen2.5-VL (Bai et al., 2025b) recommended decoding setup. Qwen3-VL trained models follow the decoding setup reported in the Qwen3-VL report (Bai et al., 2025a). Tables C2 and C3 summarize the model-family-specific sampling parameters and the shared runtime settings. We use Qwen3-32B with thinking disabled as the evaluation LLM judge when an LLM judge is required. For benchmarks requiring a VLM judge, we use Qwen3-VL-32B-Instruct. For judges, we use sampling parameters set to Temperature=0.7, TopP=0.8, TopK=20, and MinP=0.

We evaluate all models using the lmms-eval (Zhang et al., 2025a) framework, following the official evaluation protocols specified by each benchmark’s authors. Full benchmark-specific choices and metric details are provided in Appendix C. Table 3 reports Avg@5 for Vero variants, with superscript ± values denoting the standard error of the mean across runs. For the Qwen3.5-9B-Base model which exhibited weak instructionfollowing behavior, we extracted predicted answers using GPT-5.4-mini for ChartQA-Pro, EvoChart, and CountQA, and via the LaTeX boxed format for InfoVQA, ChartQA, GameQA-lite, and MathVision.

###### Vero (Ours) Open Weights Models Fully Open RL Recipes Propr.

Open MM Reas-7B

Vero Qw35-9B Base

One Thinker 8B

LLaVA OV1.5 RL-8B

Qw3VL 8B-Thk

Qw3VL 8B-Ins

Qw35 9B

Qw25VL 7B-Ins

Vero Qw35-9B

Vero Qw3I-8B

Vero Qw3T-8B

Vero Qw25-7B

Vero Mi-7B

MiVL RL-7B

Mo2-O 7B

GPT-5 Nano

Qw3vl 8B + Distill 340k

Qw25vl 7B + Distill 874k

Qw35 9B Base

LLaVA OV1.5 Ins

MiVL 7B

MiVL 7B SFT

RL Initial Model

Qw3VL 8B Think

Qw35 9B

Qw3VL 8B Inst

Qw25VL 7B Inst

SFT Only N/A

SFT N/A N/A N/A N/A

###### Chart & OCR

ChartQA-Pro 71.2±+0.15.1 70.4±+0.25.3 61.4+±17.10.2 62.3±+0.23.4 52.0±+0.28.7 61.6+±10.50.2 44.3† 58.9† 66.1† 43.3† 61.1† - 48.3† 56.0† 31.4† 56.0† ChartQA 91.7±+0.00.4 92.1±+0.14.0 91.7±+0.12.1 90.7±+0.12.1 90.3±+0.13.0 90.6±+0.10.5 89.6 88.6 91.3† 87.3 94.4 87.4 89.8† 88.0† 75.2† 80.1† InfoVQA 90.9±−0.00.4 91.8+±14.20.1 87.5±+0.14.4 88.3±+0.22.3 81.6±−0.11.0 87.3±+0.15.4 83.1 86.0 91.3† 82.6 90.1 76.6 82.8† 74.9† 60.3† 67.9† CharXivReason 77.4±+0.24.4 72.3+±10.60.2 54.3±+0.67.9 60.3±+0.37.3 45.8±+0.33.3 64.3+±11.20.5 46.4 53.0 73.0 42.5 60.9† - 46.1 47.5† 35.6† 51.2† ChartMuseum 70.3±+0.23.5 67.4+±12.00.3 47.5±+0.27.5 51.6±+0.37.2 33.0±+0.56.2 47.7±+0.47.5 40.0 44.4 66.8† 26.8 48.7† - 35.2† 35.2† 30.3† 48.0† EvoChart 81.3±+0.11.1 81.4±+0.17.2 74.7+±10.70.3 75.7±+0.12.7 64.6±+0.21.8 73.4±+0.22.4 64.0† 73.0† 80.2† 62.8† 73.4† - 64.4† 67.4† 51.0† 63.3† Category Avg 80.5±+0.02.3 79.2±+0.18.9 69.5±+0.28.3 71.5±+0.14.2 61.2±+0.13.6 70.8±+0.06.3 61.2 67.3 78.1 57.6 71.4 - 61.1 61.5 47.3 61.1

###### STEM

MMMU-ProStd 71.3±+0.21.2 69.9±+0.37.1 59.0±+0.33.1 59.5±−0.20.9 42.8±+0.34.5 58.7±+0.22.6 55.9 60.4 70.1 38.3 59.4† 39.9 44.1 53.6† 31.9† 61.3† MMMU-ProVis 69.6+±10.90.2 67.4+±12.90.2 56.6+±14.50.1 57.9±+0.24.5 40.0±+0.27.7 53.1±+0.25.4 42.1† 53.4† 58.7† 32.3† 49.8† 35.7 40.6 48.3† 16.0† 53.1† MathVision 79.0±+0.20.1 71.1+±16.70.2 59.4±+0.25.5 63.2±+0.10.5 28.4±+0.13.3 59.4±+0.22.6 53.9 62.7 78.9† 25.1 58.8† 34.4 43.6 48.6† 21.3† 61.7† MathVistatestmini 86.8±+0.41.1 85.9+±19.70.2 81.0±+0.23.8 80.7±−0.30.7 74.3±+0.36.1 79.7±−0.10.9 77.2 81.4 85.7 68.2 80.4† 72.3 79.5 77.6 53.6† 70.2† Category Avg 76.7±+0.13.3 73.6+±14.10.2 64.0±+0.06.7 65.3±+0.10.8 46.4±+0.25.4 62.7±+0.12.4 57.3 64.5 73.3 41.0 62.1 45.6 52.0 57.0 30.7 61.6

###### Spatial & Action

Blink 70.9±+0.23.5 69.7±+0.44.6 68.3±−0.20.8 66.6±+0.31.9 58.6±+0.12.2 63.3±+0.10.9 69.1 64.7 67.4† 56.4 64.5† - 60.8† 61.9† 56.4† 59.3† ERQA 55.8±+0.20.2 52.5±+0.74.5 47.6±+0.41.9 46.2±−0.50.5 42.4±+0.60.6 40.9±+0.41.4 45.8 46.8 55.5 41.8† 43.5† - 39.2† 43.2† 43.5† 45.5† GameQALite 75.3+±13.90.2 68.3+±18.50.2 52.5+±18.50.3 54.9+±15.10.3 45.3+±19.20.1 52.0±+0.15.8 34.0† 39.8† 61.4† 26.1† 49.8† - 29.2† 41.4† 29.6† 45.9† EmbSpatial 84.0±+0.21.0 81.8±+0.23.6 79.4±+0.20.9 80.8±−0.10.3 69.8±−0.10.9 71.4±−0.11.1 78.5 81.1 83.0 70.7† 70.2† - 71.0† 72.0† 68.1† 74.2† CV Bench 89.0±+0.11.5 89.9±+0.22.6 88.1±+0.12.6 87.9±+0.11.9 81.4±+0.11.0 83.7±+0.11.1 85.5† 86.0† 87.5† 80.4† 83.5† 82.9 81.1† 82.9† 81.7† 82.5† Category Avg 75.0±+0.04.0 72.4±+0.36.8 67.2±+0.14.6 67.3±+0.23.6 59.5±+0.14.4 62.3±+0.11.6 62.6 63.7 71.0 55.1 62.3 - 56.3 60.3 55.9 61.5

###### Knowledge & Recognition

RealWorldQA 80.1±−0.00.2 78.3±+0.46.5 74.2±+0.42.7 71.4±−0.22.1 68.7±+0.10.2 70.0±+0.42.4 71.5 73.5 80.3 68.5 68.6† 68.4 69.9† 68.2† 73.3† 65.9† SimpleVQAEn 49.5±+0.54.8 53.9+±12.20.3 45.9±+0.31.7 43.9±−0.31.0 50.1±+0.35.6 46.4±+0.24.5 44.2† 44.9† 44.6† 44.5† 40.9† - 43.8† 43.1† 30.8† 36.6† FVQA 32.4±+0.31.0 26.5±−0.21.5 24.3±−0.41.7 22.1±−0.22.1 26.6±+0.14.7 27.9±−0.31.9 26.0 24.2 31.4† 21.9 31.8† - 21.4† 22.8† 17.7† 29.4† MM-Vet v2 81.0±+0.84.5 84.1+±11.50.3 71.0±+0.33.4 82.4±+0.47.9 66.1±+0.43.2 78.5+±18.80.4 67.6 74.5 76.5† 62.9 61.2† - 62.6† 66.3† 60.8† 71.5† Category Avg 60.7±+0.02.5 60.7±+0.17.2 53.8±+0.21.5 54.9±+0.20.6 52.8±+0.13.3 55.7±+0.15.9 52.3 54.3 58.2 49.5 50.6 - 49.4 50.1 45.6 50.9

Grounding, Counting & Search CountBenchQA 95.5±+0.20.0 94.1±+0.23.7 90.0±+0.41.2 92.1±+0.52.5 83.1±−0.42.8 84.2±−0.31.1 88.8† 89.6† 95.5† 85.9† 86.4† 86.8 83.7† 89.2† 89.4† 75.4† CountQA 60.7±−0.21.1 48.6+±12.30.4 34.6±+0.26.1 37.0±+0.45.2 24.1±+0.23.2 26.5±+0.10.9 28.5† 31.8† 61.8† 20.9† 27.4† - 22.0† 28.8† 32.1† 25.7† MME-RealWorld-Lite 61.7±+0.20.7 61.8±+0.15.1 57.5+±10.40.2 54.4±+0.19.6 53.6±+0.19.1 52.8±+0.29.0 47.1† 44.8† 61.0† 44.5† 48.7† - 53.9† 42.6† 44.4† 49.8† VStarBench 90.5±+0.30.4 88.6±+0.33.8 88.7±+0.56.5 82.9±+0.66.5 84.8±+0.04.7 83.8±+0.01.6 82.2† 76.4† 90.1 80.1† 84.3 79.1 81.7† 66.5† 73.8† 71.2† AerialVG 45.3+±11.20.1 42.3+±40.10.0 30.2±−0.12.0 32.3+±19.70.2 29.1±+0.06.8 15.5±−0.14.6 32.2† 12.6† 34.1† 22.3† 22.2† - 11.9† 24.8† - VisualProbe 55.0±+0.42.8 55.9±−0.20.2 52.0±+0.54.3 46.3±+0.47.0 52.1±+0.06.1 51.6±+0.04.8 47.7† 39.3† 52.2† 46.0† 52.2† - 45.9† 21.3† 34.8† 41.5† ScreenSpot 90.8±+0.35.3 93.9+±61.00.1 92.6±+0.16.0 91.7±+0.26.2 90.2±+0.04.6 90.6±+0.12.7 86.6† 85.5† 85.5† 85.6† 87.3† - 63.5† 84.2† 75.8† ScreenSpotPro 64.2±−0.21.0 68.0+±58.60.1 61.8±+0.27.2 48.7±+0.42.1 39.6+±15.70.0 36.9±+0.04.1 54.6 46.6 65.2 23.9† 37.4† - 5.8† 20.7† 19.3† Category Avg 70.5±+0.12.3 69.1+±23.00.1 63.4±+0.14.9 60.7±+0.17.4 57.1±+0.06.0 55.2±+0.02.2 58.5 53.3 68.2 51.1 55.7 - 46.0 47.3 - Captioning & IF

MM-MTBench 83.8±+0.45.7 85.8+±14.30.4 81.0±+0.36.6 71.4±−0.66.4 62.6±+0.53.7 75.1±−0.70.6 74.4 77.8 78.1† 58.9 79.2† - 37.2† 60.4† 33.3† 72.7† MIABench 92.0±+0.12.2 93.5±+0.23.7 93.4±+0.12.3 92.4±+0.30.9 87.2±+0.15.4 90.7±+0.22.3 91.1 91.5 89.8† 81.8 88.4† - 66.6† 81.4† 77.9† 92.0† MMIFEval 86.4±+0.34.3 84.0+±11.30.3 76.0±+0.56.8 78.3±+0.33.3 66.6+±13.00.4 78.7±+0.49.5 69.2 75.0 82.1† 53.6 66.1 - 39.7† 53.5† 54.4† 78.0† Category Avg 87.4±+0.14.1 87.8±+0.19.8 83.5±+0.25.3 80.7±−0.30.7 72.1±+0.37.3 81.5±+0.23.7 78.2 81.4 83.3 64.8 77.9 - 47.8 65.1 55.2 80.9

Overall Avg 74.4±+0.02.9 73.0+±12.90.0 66.1±+0.05.4 65.8±+0.13.5 57.8±+0.14.9 63.2±+0.13.6 60.7 62.3 71.5 52.9 62.4 - 52.2 55.7 - -

- Table 3. Vero achieves state-of-the-art performance across six task categories on VeroEval. Vero columns show Avg@5 over initial models trained with RL on our dataset; superscript ± values report the standard error of the mean across runs, and +x / -x deltas indicate improvement/decline over the respective initial model. In model names, Qw denotes Qwen and Mi denotes MiMo. † indicates results evaluated by us. All other results are taken from official technical reports.

Vero-600k (Ours) ViRL-39k OpenMMReasoner-74k LLaVA-OV-1.5-RL-70k

Spatial & Action

Knowledge & Recognition

Grounding, Counting & Search

Captioning & IF

Chart & OCR

STEM

(5 evals)

(3 evals)

(6 evals)

(4 evals)

(4 evals)

(8 evals)

60

54

57

68

62

50

| | | |
|---|---|---|
| | | |

| | | |
|---|---|---|
| | | |

| | | |
|---|---|---|
| | | |

| | | |
|---|---|---|
| | | |

| | | |
|---|---|---|
| | | |

| | | |
|---|---|---|
| | | |

53

48

44

55

56

43

46

42

32

42

50

36

0 300 600

0 300 600

0 300 600

0 300 600

0 300 600

0 300 600

RL FLOPs

- Figure 6. Per-category RL training curves. Evaluation score vs. RL FLOPs for RL runs using Vero-600K and three prior open RL datasets, all starting from Qwen2.5-VL-7B-Instruct under identical RL settings. Curves report mean and SEM across seeds where shown. Dashed segments indicate training beyond one epoch. Vero-600K leads on five of six categories throughout training; on STEM it remains within 2 points of the best prior dataset.

Baselines. We compare against: (1) base VLMs without native <think> tokens (Qwen2.5-VL-7B-Instruct (Bai et al., 2025b), Qwen3-VL-8B-Instruct (Bai et al., 2025a), Qwen35-9B, Molmo2-O-7B (Clark et al., 2026)), (2) models trained to do native CoT with <think> tokens (Qwen3-VL-8B-Thinking (Bai et al., 2025a), MiMo-VL-

- 7B-RL (Yue et al., 2025b)), (3) existing fully open RL-trained models and recipes (VL-Rethinker-7B (Wang et al.,

- 2025), LLaVA-OV-1.5-RL (An et al., 2025), OpenMMReasoner-7B (Zhang et al., 2026), OneThinker-8B (Feng et al., 2025)), and (4) a proprietary reasoning model gpt-5-nano-2025-08-07 (Singh et al., 2025) with medium reasoning effort. For baseline results, we prioritize scores from official technical reports and benchmark leaderboards. When published results are unavailable, we evaluate the models ourselves (indicated by †) and follow the published benchmark guidelines.

- 4.1 Evaluation Results on VeroEval We report results in Table 3 and highlight the following observations.

Strong performance with a fully open RL recipe. Vero-Qwen35-9B achieves the highest overall average in Table 3, reaching 74.4 on VeroEval. It improves over Qwen35-9B by +2.9 overall, wins 25 of 30 benchmarks, and improves all six category averages: Chart & OCR (+2.3), STEM (+3.3), Spatial & Action (+4.0), Knowledge & Recognition (+2.5), Grounding, Counting & Search (+2.3), and Captioning & IF (+4.1). Among 8B models, Vero-Qwen3I-8B and Vero-Qwen3T-8B reach 66.1 and 65.8 overall, respectively, outperforming Qwen3-

- VL-8B-Thinking by +3.8 and +3.5 overall. Against distillation warm start baselines using the same initial model, Vero-Qwen25-7B is higher than OpenMMReasoner-7B by +5.6 overall, wins 5 of 6 category averages, and has its largest category margins on Captioning & IF (+24.3) and Grounding, Counting & Search (+11.1). Vero-Qwen3I-8B is higher than OneThinker-8B (Feng et al., 2025) by +10.4 overall and wins all six category averages. Vero uses no additional teacher distillation in either comparison.

Consistent gains across initial models. Vero training yields improvements across six different initial models. Over the five post-trained initial models, overall gains range from +2.9 to +5.4. Vero-Qwen3I-8B improves over Qwen3-VL-8B-Instruct by +5.4 overall and outperforms Qwen3-VL-8B-Thinking on 25 of 30 benchmarks despite using no distillation. Vero-Qwen3T-8B improves over Qwen3-VL-8B-Thinking by +3.5 overall and on 22 of 30 benchmarks, and Vero-Qwen25-7B improves over Qwen2.5-VL-7B-Instruct by +4.9 overall and on 27 of 30 benchmarks. The largest improvement comes from applying RL directly to the pretrained-only Qwen3.5-9B-Base: +12.9 overall, reaching 73.0, the second-highest score in Table 3, with no SFT or distillation warm start. Gains concentrate on Grounding, Counting & Search (+23.0) and Captioning & IF (+9.8, the table’s best at 87.8), capabilities largely absent from the base checkpoint despite prompting attempts2, while reasoning categories also improve substantially (STEM +14.1).

- 2Qwen35-9B-Base produced parseable coordinate outputs under prompting, but the predicted coordinates were poorly grounded.

###### (a) SFT vs. RL

Knowl.&

Spatial&

Chart&

Overall

Cap.&

Action

Cnt.&

Recog.

Search

Grnd.,

STEM

OCR

Avg.

IF

Base 57.6 41.0 55.1 49.4 50.1 64.8 52.4 FineVis SFT 54.8 37.4 52.1 45.3 40.1 52.2 46.2 Vero SFT 52.5 40.1 58.1 50.8 52.7 64.1 52.8 Vero RL 61.9 46.7 59.4 53.5 55.0 70.6 57.2

###### (b) Reward design

###### (c) RL algorithm (1/4 epoch, 5 task categories)

Knowl.&

Knowl.&

Spatial&

Spatial&

Entropy

Chart&

Chart&

Overall

Cap.&

Action

Action

Cnt.&

Cnt.&

Recog.

Recog.

Search

Search

Grnd.,

Grnd.,

STEM

STEM

OCR

OCR

Avg.

Avg.

Avg.

IF

Base 57.6 41.0 55.1 49.4 50.1 64.8 52.4 Math Ver. 61.4 46.1 58.5 50.1 51.0 34.3 51.8 Ours 61.9 46.7 59.4 53.5 55.0 70.6 57.2

DAPO 58.9 45.3 57.1 49.7 52.2 54.3 0.22±0.15 GRPO 59.2 44.4 58.1 48.2 53.0 54.3 0.50±0.11 GSPO 59.0 45.4 58.4 50.4 53.0 54.7 0.58±0.11

- Table 4. Ablation studies. All results on Qwen2.5-VL-7B-Instruct. Tables (a)–(c) report absolute scores. All runs are trained 1 epoch on the 600k mixture unless otherwise specified.

Vero-MiMo-7B, trained on MiMo-VL-7B-SFT with our fully open recipe, improves by +3.6 overall. It is +0.8 overall above MiMo-VL-7B-RL, which trains on the same initial model but uses a proprietary RL recipe with non-public data. Vero-MiMo-7B is higher on STEM (+0.6), Knowledge & Recognition (+5.1), and Captioning & IF (+3.6), tied on Spatial & Action, and lower on Chart & OCR and Grounding, Counting & Search.

Improvements not limited to a single domain. Unlike prior open RL-trained VLMs that focus primarily on STEM, Vero yields improvements across all six task categories. For example, Vero-Qwen3I-8B improves over its initial model on Chart & OCR (+8.3), STEM (+6.7), Spatial & Action (+4.6), Knowledge & Recognition (+1.5), Grounding, Counting & Search (+4.9), and Captioning & IF (+5.3), demonstrating that multi-task RL produces broadly capable models rather than specialists.

Consistent advantage over prior open RL datasets across training. Figure 6 compares RL runs using Vero-600K against runs using three prior open RL datasets, ViRL-39k (Wang et al., 2025), OpenMMReasoner74k (Zhang et al., 2026), and LLaVA-OV-1.5-RL-70k (An et al., 2025). All runs start from Qwen2.5-VL-7BInstruct and use identical RL settings over 600 steps, averaged over three runs with different random seeds. Even in the early training regime (first ∼150 steps), where all datasets are still within their first epoch, Vero-600K already leads or remains within seed variance of the best prior dataset on every category. By the end of training, Vero-600K reaches the highest score on five of six categories, with the largest margins on Captioning & IF (+13.7 over the next best) and Grounding, Counting & Search (+6.7). On STEM, where prior datasets largely concentrate their dataset selection, Vero-600K remains competitive (45.3 vs. 47.0 for ViRL-39k and 46.2 for OpenMMReasoner-74k), trailing by fewer than 2 points despite learning across all six categories.

###### 4.2 Ablations

Multi-task RL requires more expressive reward design. We compare our multi-route reward design against math_verify (Kydlícˇek, 2025), a widely used reward that performs extraction, parsing, and grading. Results in Table 4(b) show that our reward design, which routes answers through type-specific comparisons (exact match, numeric tolerance, set matching, and LLM-judge evaluation), achieves stronger performance than math_verify across task categories. math_verify lacks the flexibility to handle the diverse answer formats.

Our data benefits most from RL, yet even with SFT alone it outperforms strong SFT baselines. We compare SFT and RL training on our dataset in Table 4(a). The SFT model is trained to directly output the final answer without chain-of-thought or <think> tokens. SFT on our dataset produces gains on most tasks and outperforms SFT on a recent post-training dataset FineVision (Wiedmann et al., 2025). However, RL (GSPO with our multi-route reward) yields more consistent improvements across all task categories.

Δ vs Qwen2.5-7B-VL-Instruct

Δ vs Qwen3-8B-VL-Instruct

|+3.7|+0.2|+0.3|-1.8|-0.7|-4.4|+3.2|+4.8|
|---|---|---|---|---|---|---|---|
|+2.1|+4.6|+3.5|+3.0|+1.2|-2.1|+4.2|+5.7|
|+0.2|+0.0|+3.1|-0.7|-1.7|-6.3|+1.9|+5.0|
|+0.9|+0.3|-1.1|+4.3|-1.5|-2.6|+0.6|+3.6|
|-3.2|-3.3|-4.3|+0.9|+4.0|-7.7|+2.8|+3.8|
|-35.5|-21.2|-19.5|-15.8|-23.8|+2.4|+0.3|+4.2|

|+5.3|+3.6|+4.7|+4.8|+2.4|-2.0|+3.2|+6.7|
|---|---|---|---|---|---|---|---|
|+1.5|+7.0|+6.6|+4.5|+3.2|+0.9|+5.2|+5.8|
|+2.9|+2.7|+4.4|+2.4|+1.9|+0.6|+2.8|+3.5|
|+1.9|+3.7|+2.8|+1.4|+2.4|+2.6|+2.7|+1.6|
|-3.9|-0.8|-1.2|+0.4|+3.6|-5.1|+2.5|+3.2|
|-13.9|-12.8|-8.9|-6.9|-11.3|+4.9|+1.9|+4.7|

Chart & OCR

STEM

Evaluation

Spatial & Action

Knowl. & Recog.

Grnd., Cnt. & Search

Captioning IF

STEM Spatial & Action

STEM Spatial & Action

Mix Ful

Chart & OCR

Mix Comp. Ctrl.

Captioning IF

Mix Ful

Chart & OCR

Mix Comp. Ctrl.

Captioning IF

Knowl. & Recog. Grnd., Cnt. & Search

Knowl. & Recog. Grnd., Cnt. & Search

Training

- Figure 7. Diverse task mixing eliminates negative cross-task transfer. Each row shows a model trained on a single task category (or mixture of all categories). Values are absolute score changes relative to the base model. Single-task training yields selective transfer, while mixing achieves consistent gains.

GSPO outperforms GRPO and DAPO and leads to more stable entropy. We compare three RL algorithms (DAPO, GRPO, and GSPO) using the same base model (Qwen2.5-VL-7B-Instruct), reward design, and training dataset. Consistent with recent findings (Zhang et al., 2026), results in Table 4(c) show that GSPO achieves the highest average score (54.7) across all task categories, outperforming both GRPO (54.3) and DAPO (54.3). GSPO also maintains substantially more stable entropy throughout training (0.58 ± 0.11) compared to GRPO (0.50 ± 0.11) and especially DAPO (0.22 ± 0.15), suggesting that GSPO’s sequence-level clipping better preserves exploration capacity and avoids premature policy collapse observed with alternative algorithms.

##### 5 Data Diversity & Cross-Task Transfer

We study cross-task generalization by training models on each individual task category (100k samples, 1 epoch) and evaluating across all six categories. We compare against a model trained on a mixture of all categories with the same total number of training samples (100k, compute-controlled) and the full dataset (600k). We report results on two base models in Figure 7.

Single-task training frequently produces neutral or negative transfer on non-target tasks. On Qwen2.5-VL, nearly all single-task-category models degrade Grounding, Counting & Search performance (e.g., −3.2 from Chart & OCR, −3.3 from STEM, −4.3 from Spatial & Action), and training on Captioning & Instruction Following alone reduces performance across all other categories (−2.1 to −7.7). Conversely, training on any non-captioning task category severely degrades Captioning & Instruction Following (−15.8 to −35.5 on Qwen2.5-VL). These patterns hold on Qwen3-VL, where single-task-category models similarly hurt Grounding (−0.8 to −5.1 for non-grounding domains) and Captioning & Instruction Following (−6.9 to −13.9). However, in certain task categories, we do observe selective positive transfer: STEM training improves Chart & OCR (+3.6 on Qwen3-VL), and Spatial & Action training yields strong gains on STEM (+3.5 on Qwen2.5-VL, +6.6 on Qwen3-VL).

Diverse task mixing eliminates negative cross-task transfer. Even with the same compute budget, the mixed model achieves positive gains across all categories on both base models (+0.3 to +4.2 on Qwen2.5-VL; +1.9 to +5.2 on Qwen3-VL), avoiding the catastrophic losses seen with single-domain training. Training on the full 600k mixture further amplifies these gains. These patterns are consistent across both base models, suggesting that multi-task RL training is important for producing broadly capable models. We provide additional analysis and discussion in Sections 7 and 8, suggesting that reasoning behaviors may not transfer readily across the six task categories because their associated chain-of-thought patterns are largely distinct.

MM-MTBench MIA-Bench MMIF-Eval Average

+9.2

Spatial & Action 1,983.3 ± 50.8

+10

+7.2

+5.8 0

+1.0

Chart & OCR 1,592.7 ± 32.5

ΔScore(vsBase)

- -60
- -50
- -40
- -30
- -20
- -10

-8.3

STEM 1,576.1 ± 39.6

-17.1

-18.6

-22.9

-24.2

Captioning & IF 413.8 ± 13.1

-36.5

Grnd., Cnt., & Search 124.9 ± 12.6

-38.0

Knowledge & Recog. 75.8 ± 2.9

-54.6

0 Average Number of Reasoning Words 2500

+ Ans. tag + Sys

+ Captioning & IF with LLM Judge

+ Boxed

- Figure 8. RL on different task categories leads to varying reasoning lengths. Average reasoning length (in words) on the validation set, measured after training Qwen3-VL-

Figure 9. Open-ended RL training is important for maintaining visual chat quality. Answer tag parsing alone sharply reduces Captioning & Instruction Following performance, while adding system guidance and the Captioning & Instruction Following training category restores and improves visual chat quality. All experiments are run on Qwen2.5-VL-7B-Instruct.

- 8B-Instruct for 1000 steps on each task category data (100k) and evaluating on the same category. Error bars denote the standard error of the mean.

Vero-Q25-7B Vero-Mi-7B Vero-Q3T-8B CharXiv Reasoning

CountBenchQA

MMIFEval

70

100

90

| | | |
|---|---|---|
| | | |

| | | |
|---|---|---|
| | | |

| | | |
|---|---|---|
| | | |

Performance

50

85

65

30

70

40

0 200k 400k 600k

0 200k 400k 600k

0 200k 400k 600k Data Seen

- Figure 10. Performance improves with greater training data exposure. We plot benchmark performance as each model sees progressively more samples from the fixed 600K RL training mixture over the course of a single training pass.

Task categories elicit markedly different reasoning lengths. Figure 8 summarizes average reasoning length for Qwen3-VL-8B-Instruct trained on each task category. Spatial & Action has the longest responses at 1983.3 ± 50.8 words, followed by Chart & OCR at 1592.7 ± 32.5 and STEM at 1576.1 ± 39.6. Captioning & Instruction Following is much shorter at 413.8 ± 13.1, while Grounding, Counting & Search and Knowledge & Recognition are shortest at 124.9 ± 12.6 and 75.8 ± 2.9, respectively. The gap between Spatial & Action and Knowledge & Recognition is more than 26× larger, which suggests that long chain-of-thought behavior is concentrated in tasks that require multi-step state tracking or structured analytical decomposition.

Broader exposure to the mixed training distribution yields continued gains. Figure 10 tracks performance during a single pass over the fixed 600K-sample mixture for three Vero variants (Vero-Qwen25-7B, VeroMiMo-7B, and Vero-Qwen3T-8B) on CharXiv Reasoning, CountBenchQA, and MMIFEval, so later points reflect greater exposure to diverse RL samples rather than additional epochs. From the 100k checkpoint to the final checkpoint, all nine model–benchmark curves improve, with a mean gain of +4.3 points. The largest late-stage gains appear on CharXiv Reasoning (mean +5.7 pp across models; up to +7.3 pp for Vero-MiMo-7B) and MMIFEval (mean +5.5 pp; up to +6.1 pp for Vero-Qwen25-7B), while CountBenchQA improves more modestly (mean +1.6 pp), having largely plateaued by the 100k checkpoint. The same late-stage trend holds on the broader evaluation suite (not shown), with continued gains on benchmarks such as ScreenSpot-Pro, GameQA Lite, and ChartMuseum and near-saturation on MMMU-Pro Vision. These trends indicate that exposing the policy to more of the diverse training distribution remains beneficial over long training.

##### 6 Visual Chat Quality

In Section 5, we showed that training on task categories other than open-ended visual question answering and instruction following can substantially degrade performance on these capabilities. Here, we describe steps taken to mitigate this degradation and, in many cases, to further improve the model’s ability to produce fluent, instruction-adherent responses while preserving structured reasoning.

RL without open-ended prompts leads to visual chat deficits. We start from a baseline trained on the five task categories other than Captioning & Instruction Following (500k examples), using the answer format adopted in prior visual reasoning work (Meng et al., 2025; Zhang et al., 2026). We then incrementally introduce components of our open-ended instruction-following design to assess their individual contributions to preserving and improving visual chat quality (Figure 9). With answer tag parsing only, Captioning & IF quality drops severely (64.8→26.8), as the model collapses to producing terse answers for all queries. Adding system prompt guidelines and LATEX boxed formatting partially recovers performance (47.7), but still falls well short of the base model. Only when we include the full Captioning & Instruction Following task category with LLM-judge rewards does the model recover and surpass the base model on all three open-ended benchmarks (70.6 vs. 64.8 average), while simultaneously maintaining improved scores on other task categories. This demonstrates that explicitly training on open-ended prompts with judge-based rewards prevents RL from narrowing the model’s output distribution to short structured answers and instead teaches it to flexibly switch between structured reasoning and fluent, instruction-adherent generation.

Mitigating reward hacking with judge guidelines. While adding open-ended and instruction-following tasks improves the model’s ability to generate conversational responses, it also introduces the risk of reward hacking from the LLM judge. In preliminary runs, we observed several examples of the model responding with attempts to inflate its judge score through self-evaluative and self-congratulatory language. We highlight representative examples below:

- • “This description exhaustively documents every distinguishable visual element, spatial relationship, and stylistic detail observable in the image—without inferring purpose, user intent, or contextual meaning. It includes all necessary factual anchors ... to allow independent verification by another observer.”
- • “Note: Since the question only asks for description—and does not request analysis ...—this response fully satisfies the prompt. It provides complete, self-contained evidence of the image’s visual reality ... Therefore, no \boxed{} element is added here.”
- • “End of response. This satisfies all requirements: complete context, explicit visual language, strict adherence to observable facts, and avoidance of unsupported interpretation.”

These statements serve no informational purpose and are directed at the judge rather than the user. They assert compliance with evaluation criteria (“satisfies all requirements”), claim exhaustiveness (“exhaustively documents every ... detail”), and preemptively justify formatting choices (“no \boxed{} element is added here”). Beyond self-evaluation, the responses often exhibited over-specification as a form of score inflation:

- • Fabricated measurements: “15px vertical gap between username and password fields”, “diameter ~16px”.
- • Invented hex codes: “Pure #FF0000 (no transparency)”, “Gradient from #00668A (top) to #005A7A (bottom)”.
- • Unnecessary font specifications: “sans-serif, 14px, left-aligned, with a small asterisk”.

These details cannot be reliably determined from a screenshot and serve primarily to create an impression of thoroughness for the judge. We find that including strict judge prompt guidelines suffices to overcome reward hacking (see Listing B.3). We include explicit Automatic Failure Conditions that assign a score of 1 to any response containing self-evaluative statements. This penalty makes reward hacking through metacommentary a losing strategy, incentivizing the model to produce informative responses instead.

Training Training

Grnd.,Cnt.&SearchKnowledge&Recog.

Grnd.,Cnt.&SearchKnowledge&Recog.

Action

Action

Captioning&IFChart&OCR

Captioning&IFChart&OCR

0.04 0.28 0.52 0.76 1.00

Spatial&

Spatial&

All

All

STEMMix

STEMMix

Logical Coherence Sequential Organization

Visual Ref. / Grounding

Strategy Selection Syst. Regional Synth.

Selective Attention Compositionality Goal Management Forward Chaining Knowledge Struct. Align. Visual Foraging

Conceptual-Level Proc. Temporal Organization Ordinal Organization

Context Awareness Mental Imagery Sim.

Verification Decomp. & Integration

Network Organization Pattern Recognition Repr. Restructuring

Self-Evaluation Context Alignment

Causal Organization Abstraction Productivity

Perception-then-Reas.

Hierarchical Org. Spatial Organization

Backtracking Arithmetic Calculation Backward Chaining

Self-Awareness Adaptive Detail Mgmt.

- Figure 11. Single-task training elicits a range of behavioral profiles, many of which are distinct. Average behavior presence rates for Vero on Qwen3-VL-8B-Instruct, aggregated over six validation task categories after balanced subsampling. 34 high-level cognitive behaviors, sorted by overall prevalence. Single-task training elicits distinct high-level strategies, and these differences carry through to the skill level.

##### 7 Chain-of-Thought Behaviors

Benchmark accuracy alone does not characterize how the trained models arrive at their answers. To complement the performance results in Section 5, we analyze the generated reasoning traces at two levels: high-level cognitive behaviors and lower-level recurring skills. Together, these analyses quantify whether models trained on different task categories exhibit consistent differences in their intermediate reasoning traces.

###### 7.1 High-Level Cognitive Foundations

Experimental Setup. We evaluate model behavior using the cognitive framework of Kargupta et al. (2025), which defines 28 textual reasoning behaviors, supplemented with six behaviors for visual analysis. We compute the presence rate of each behavior for every trained model from Section 5 across all six validation task categories. Behavior presence rate on Vero trained on Qwen3-VL-8B-Instruct is shown in Figure 11, and we report additional base models in the Appendix D.2.

Each task category elicits a distinct cognitive behavioral profile. The behavioral profiles reveal that training on each task category elicits different cognitive behaviors. Captioning often uses mental imagery simulation (0.64 vs. 0.57 cross-task-category average in Qwen3), chart-trained models trigger systematic

Captioning IF 81±2.5%

Captioning IF

Chart & OCR 82±2.1%

Chart & OCR

Grnd., Cnt., & Search 80±3.9%

Grnd., Cnt., & Search

Knowl. & Recog. 59±3.7%

Knowl. & Recog.

Spatial & Action 77±3.3%

Spatial & Action

STEM 84±1.9%

STEM

0% 25% 50% 75% 100%

- Figure 12. Task category skills are largely distinct. Each bar shows the predicted classification distribution (as proportions) from a logistic regression probe trained on skill embeddings, with the values indicating per-category accuracy. The probe achieves 0.77 overall accuracy, with STEM (0.84) and Chart & OCR (0.82) yielding the most distinctive skills. Knowledge & Recognition is the least separable (0.59), with notable confusion toward Grounding, Counting & Search, reflecting shared visual grounding operations across these categories.

regional synthesis (0.74 vs. 0.68), and spatial reasoning often uses perception-then-reasoning sequencing (0.84 vs. 0.73). Grounding tasks more often suppress introspective behaviors, with self-awareness dropping to 0.49 (vs. 0.73), redirecting capacity toward directed visual search. In contrast, task categories requiring multi-step integration elicit higher-order behaviors, with STEM tasks showing elevated backtracking (0.48 vs. 0.27). The mixed-task-category setting increases strategy selection (0.80 vs. 0.71), indicating that the model first selects a reasoning approach before executing it.

###### 7.2 Skill Analysis

High-level behaviors provide a coarse summary of the traces. To obtain a more granular view, we additionally analyze recurring skills extracted from the same reasoning traces.

Experimental Setup. We extract task-category-specific skills from model reasoning traces following Didolkar et al. (2025). A deduplication pipeline ensures uniqueness of extracted skills, after which skill embeddings are clustered via agglomerative clustering and labeled with GPT-4o. We train a logistic regression probe on the resulting skill embeddings of the model trained on the same domain as the evaluation task category (Qwen3-Embedding-8B, 4,096-d) with 800 skills per task category, evaluated via 5-fold Stratified Group cross-validation with mean centering and ℓ2 normalization. We report the task confusion matrix in Figure 12.

Each task category cultivates a largely distinct skill repertoire. The probe achieves 0.77 overall accuracy (Figure 12), confirming that skill distributions are task-category-specific. STEM, chart, captioning, and grounding tasks yield the most distinctive skills: chart behaviors center on data-reading operations (e.g., cross-reference axes in visual data), while grounding behaviors reflect feature binding and localization reasoning (e.g., systematic visual scanning). Knowledge skills are least separable (0.59 accuracy), frequently confused with grounding (0.11 confusion rate).

Skill behavior presence rate. Figure 13 further shows that the prominent low-level skills vary distinctly across the six task categories. For example, the model heavily relies on mathematical concepts like "apply triangle angle sum" and "apply arc length formula" for STEM tasks, whereas it shifts to terms like "extract labels" and "compare axis ranges" for Chart & OCR. Similarly, Grounding, Counting & Search emphasizes grounding skills like "locate reference object" and "determine relative position," highlighting how the model dynamically adapts its skill set to the specific domain.

Figure D4 shows that these differences remain pronounced even when evaluation is held fixed within the same task category, indicating that training changes not only accuracy but also the composition of the reasoning process. Captioning & Instruction Following concentrates on communicative and descriptive operations, including Focus On Key Attributes, Analyze Visual Composition, and Balance Clarity & Impact.

Captioning & Instruction Following Chart & OCR Grounding, Counting & Search

[Figure 16]

[Figure 17]

[Figure 18]

Knowledge & Recognition Spatial & Action STEM

[Figure 19]

[Figure 20]

[Figure 21]

- Figure 13. Single-task training elicits distinct behavioral skills. Word clouds of extracted low-level skills reveal that models develop highly specialized strategies adapted to their specific training domains. For instance, STEM training fosters formula-driven logic (e.g., “Apply Arc Length Formula”); Grounding models prioritize localized perception (e.g., “Determine Relative Position”).

Chart & OCR instead emphasizes structured visual extraction. Grounding, Counting & Search favors localization-oriented behaviors such as Assess Visual Indicators and Visual Verification, whereas Knowledge & Recognition more often combines visual evidence with general world understanding through skills such as Context Analysis and Infer from Conventions. Spatial & Action highlights state tracking and forward simulation, including Map Obs. to Answers and Mental Simulation, while STEM more consistently activates analytical operations such as Diagram Analysis and Infer Structural Relationships. These examples suggest that each task category induces a skill-level signature rather than a generic increase in reasoning activity.

##### 8 Discussion

We advocate for fully open-source reinforcement learning recipes for vision-language models. When training pipelines are proprietary, the choices that govern performance remain hidden, preventing the field from reliably diagnosing failure modes, attributing gains to individual components, or verifying that training procedures are safe and well understood. Progress in this area requires studying models under conditions that reflect frontier-scale development, which is only possible when the full recipe is accessible. Building on prior efforts toward open training and data pipelines (Tong et al., 2024; OLMo Team et al., 2025; An et al., 2025; Guha et al., 2026; Clark et al., 2026; Zhang et al., 2026), we argue that open recipes are essential for reproducibility, mechanistic understanding, and sustained scientific progress.

We show that a transparent, single-stage RL pipeline, paired with a diverse, carefully filtered data mixture and task-routed rewards, improves performance across a broad suite of visual reasoning benchmarks. Closed systems preclude such analysis, limiting the field to black-box evaluations that reveal what a model can do but not why. We hope our work demonstrates the viability of open frontier recipes and encourages the community to prioritize transparency alongside performance.

Multi-task reinforcement learning and data diversity. A central result of our study is that, for visual reasoning reinforcement learning, breadth of task coverage matters at least as much as algorithmic sophistication. A simple single-stage RL pipeline, when paired with a diverse mixture of task categories and task-routed rewards, is sufficient to produce broad gains across visual reasoning benchmarks. Prior multi-task RL (Teh et al., 2017; Schaul et al., 2019; Hessel et al., 2019) work emphasizes that heterogeneous tasks often create optimization challenges, including interference and imbalance. Our findings do not contradict this view. Rather, they suggest that in visual reasoning, these challenges can be mitigated by data diversity and task-aware reward design. Under this regime, multi-task RL yields positive transfer across many tasks instead of the negative transfer often observed under narrower or less balanced training setups.

A useful interpretation of our findings is that each visual task category induces a different behavioral regime. In our setting, STEM tasks tend to elicit reflective and backtracking-heavy reasoning, whereas grounding and visual search tasks favor shorter, more directed perceptual strategies. This helps explain why RL on a narrow category transfers poorly outside that category: the model is not merely learning answers, but also adapting its policy over latent reasoning behaviors.

Our ablations on mixture design further reinforce that multi-task RL should be viewed as a distributiondesign problem rather than only an optimizer-design problem. The fact that uniform weighting across categories outperforms alternatives based on dataset size, reasoning length, or base accuracy suggests that broad capability emerges from maintaining balanced exposure to distinct behavioral modes. This echoes findings from instruction tuning in LLMs, where task diversity and mixture balancing are central determinants of generalization. FLAN and the Flan Collection (Longpre et al., 2023), in particular, showed that scaling the number and diversity of tasks and carefully balancing them can matter as much as model scale itself. In this sense, our results extend the instruction-tuning lesson into on-policy multimodal RL.

Relation to human multi-task reasoning. Our findings also have informative parallels to cognitive science and human multi-task reasoning. Human cognition does not rely on a single fixed reasoning strategy across all tasks. Classic task-switching work shows that changing tasks incurs measurable switch costs, consistent with the idea that distinct tasks recruit different task sets and control policies (Rogers and Monsell, 1995). Theories of cognitive control similarly emphasize that goals bias processing pathways in task-specific ways, while multiple-demand accounts argue that flexible intelligence depends on recombining partially shared control resources across diverse tasks (Miller and Cohen, 2001). From this perspective, further investigation may examine how our task categories may be eliciting different internal “task sets” in the model.

Our behavioral findings are especially reminiscent of metacognitive and visual-attention accounts. The elevated backtracking observed in STEM tasks resembles metacognitive monitoring and control, where an agent evaluates its own intermediate state and revises its strategy when uncertainty or error is detected. Conversely, grounding and search tasks appear closer to classic visual search models, where performance depends less on extended verbal reflection and more on directed allocation of attention to candidate regions or objects. Our results therefore support the view that visual reasoning in VLMs is not monolithic, but composed of multiple cognitive-style behaviors whose usefulness depends on the task.

Limitations. A few limitations remain. First, while our results show that diversity is critical, they do not establish the optimal taxonomy of task categories or the minimal set for broad transfer. For example, we do not include video or multi-turn tasks in our data mixture. Second, our behavioral analyses are descriptive rather than causal: we observe task-specific differences in reasoning traces, but do not yet identify the exact mechanisms by which those behaviors improve accuracy. Third, our analyses mostly focus on small models, including 7B-9B parameter models. Further work should study larger models and more diverse task sets.

Conclusions. We presented Vero, a fully open vision-language reasoning family trained with single-stage RL on Vero-600K, a 600K-sample dataset from 59 datasets spanning six capability categories. Our central finding is that breadth, not specialization, drives general visual reasoning: training jointly across diverse task categories with balanced, uniform exposure and task-routed rewards yields consistent positive cross-category transfer, both in benchmark performance and in the chain-of-thought behaviors models acquire. These results recast multi-task RL as a problem of distribution design rather than optimizer design alone, and they show that visual reasoning is not a monolithic skill but a composite of distinct, task-dependent behavioral modes that a single training mixture can elicit together. By releasing all datasets, training code, and models, we aim to make this recipe a reproducible foundation for future work on open visual reasoning, and to invite further study of which task taxonomies, scales, and behaviors most effectively compose into general capability.

##### Acknowledgements

This work was supported by Princeton Research Computing resources, including the Della high-performance computing cluster. We thank Princeton Language and Intelligence (PLI) for their support of this research, as well as Google’s TPU Research Cloud (TRC) program for providing additional compute.

##### References

Manoj Acharya, Kushal Kafle, and Christopher Kanan. Tallyqa: Answering complex counting questions. In AAAI, 2019.

Xiang An, Yin Xie, Kaicheng Yang, Wenkang Zhang, Xiuwei Zhao, Zheng Cheng, Yirui Wang, Songcen Xu, Changrui Chen, Didi Zhu, et al. Llava-onevision-1.5: Fully open framework for democratized multimodal training. arXiv preprint arXiv:2509.23661, 2025.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025a.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang,

Jun Tang, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b. Kerem Berke. keremberke/indoor-scene-classification via datasets at hugging face, 2024. Ali Furkan Biten, Ruben Tito, Andres Mafla, Lluis Gomez, Marçal Rusiñol, Ernest Valveny, C. V. Jawahar, and

Dimosthenis Karatzas. Scene text visual question answering. In ICCV, 2019. Jie Cao and Jing Xiao. An augmented benchmark dataset for geometric question answering through dual parallel text encoding. In COLING, 2022. Kaiyuan Chen, Shuangyu Xie, Zehan Ma, Pannag R Sanketi, and Ken Goldberg. Robo2vlm: Improving visual question answering using large-scale robot manipulation data. In NeurIPS, 2025. Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Li YanTao, Jianbing Zhang, and Zhiyong Wu. Seeclick: Harnessing gui grounding for advanced visual gui agents. In ACL, 2024.

Xianfu Cheng, Wei Zhang, Shiwei Zhang, Jian Yang, Xiangyuan Guan, Xianjie Wu, Xiang Li, Ge Zhang, Jiaheng Liu, Yuying Mai, et al. Simplevqa: Multimodal factuality evaluation for multimodal large language models. In ICCV, 2025a.

Zixu Cheng, Jian Hu, Ziquan Liu, Chenyang Si, Wei Li, and Shaogang Gong. V-star: Benchmarking video-llms on video spatio-temporal reasoning. arXiv preprint arXiv:2503.11495, 2025b.

Christopher Clark, Jieyu Zhang, Zixian Ma, Jae Sung Park, Mohammadreza Salehi, Rohun Tripathi, Sangho Lee, Zhongzheng Ren, Chris Dongjoo Kim, Yinuo Yang, et al. Molmo2: Open weights and data for vision-language models with video understanding and grounding. arXiv preprint arXiv:2601.10611, 2026.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, et al. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. Nature, 2025.

Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, Jiasen Lu, Taira Anderson, Erin Bransom, Kiana Ehsani, Huong Ngo, YenSung Chen, Ajay Patel, Mark Yatskar, Chris Callison-Burch, Andrew Head, Rose Hendrix, Favyen Bastani, Eli VanderBilt, Nathan Lambert, Yvonne Chou, Arnavi Chheda, Jenna Sparks, Sam Skjonsberg, Michael Schmitz, Aaron Sarnat, Byron Bischoff, Pete Walsh, Chris Newell, Piper Wolters, Tanmay Gupta, Kuo-Hao Zeng, Jon Borchardt, Dirk Groeneveld, Crystal Nam, Sophie Lebrecht, Caitlin Wittlif, Carissa Schoenick, Oscar Michel, Ranjay Krishna, Luca Weihs, Noah A. Smith, Hannaneh Hajishirzi, Ross Girshick, Ali Farhadi, and Aniruddha Kembhavi. Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models. In CVPR, 2025.

Aniket Didolkar, Nicolas Ballas, Sanjeev Arora, and Anirudh Goyal. Metacognitive reuse: Turning recurring llm reasoning into concise behaviors. arXiv preprint arXiv:2509.13237, 2025.

Shengyuan Ding, Shenxi Wu, Xiangyu Zhao, Yuhang Zang, Haodong Duan, Xiaoyi Dong, Pan Zhang, Yuhang

Cao, Dahua Lin, and Jiaqi Wang. Mm-ifengine: Towards multimodal instruction following. In ICCV, 2025. Mengfei Du, Binhao Wu, Zejun Li, Xuan-Jing Huang, and Zhongyu Wei. Embspatial-bench: Benchmarking

spatial understanding for embodied tasks with large vision-language models. In ACL, 2024.

Kaituo Feng, Manyuan Zhang, Hongyu Li, Kaixuan Fan, Shuang Chen, Yilei Jiang, Dian Zheng, Peiwen Sun, Yiyuan Zhang, Haoze Sun, et al. Onethinker: All-in-one reasoning model for image and video. arXiv preprint arXiv:2512.03043, 2025.

Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. In ECCV, 2024.

Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, and Lingpeng Kong. G-llava: Solving geometric problem with multi-modal large language model. In ICLR, 2025.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In CVPR, 2017.

Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina, Jean Mercat, Trung Vu, Zayne Sprague, et al. Openthoughts: Data recipes for reasoning models. In ICLR, 2026.

Danna Gurari, Qing Li, Abigale J. Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P. Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In CVPR, 2018.

Wei He, Zhiheng Xi, Wanxu Zhao, Xiaoran Fan, Yiwen Ding, Zifei Shan, Tao Gui, Qi Zhang, and Xuanjing Huang. Distill visual chart reasoning ability from llms to mllms. In Findings of EMNLP, 2025.

Xuehai He, Yichen Zhang, Luntian Mou, Eric Xing, and Pengtao Xie. Pathvqa: 30000+ questions for medical visual question answering. arXiv preprint arXiv:2003.10286, 2020.

Matteo Hessel, Hubert Soyer, Lasse Espeholt, Wojciech Czarnecki, Simon Schmitt, and Hado Van Hasselt. Multi-task deep reinforcement learning with popart. In AAAI, 2019.

Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, et al. GLM-4.5V and GLM-4.1V-Thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv preprint arXiv:2507.01006, 2025.

Muye Huang, Han Lai, Xinyu Zhang, Wenjun Wu, Jie Ma, Lingling Zhang, and Jun Liu. Evochart: A benchmark and a self-training approach towards real-world chart understanding. In AAAI, 2025.

Drew A. Hudson and Christopher D. Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, 2019.

Yiming Jia, Jiachen Li, Xiang Yue, Bo Li, Ping Nie, Kai Zou, and Wenhu Chen. Visualwebinstruct: Scaling up multimodal instruction data through web search. In EMNLP, 2025.

Priyanka Kargupta, Shuyue Stella Li, Haocheng Wang, Jinu Lee, Shan Chen, Orevaoghene Ahia, Dean Light, Thomas L. Griffiths, Max Kleiman-Weiner, Jiawei Han, Asli Celikyilmaz, and Yulia Tsvetkov. Cognitive foundations for reasoning and their manifestation in llms. arXiv preprint arXiv:2511.16660, 2025.

Mehran Kazemi, Hamidreza Alvari, Ankit Anand, Jialin Wu, Xi Chen, and Radu Soricut. Geomverse: A systematic evaluation of large models for geometric reasoning. arXiv preprint arXiv:2312.12241, 2023.

Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. ReferItGame: Referring to objects in photographs of natural scenes. In EMNLP, 2014.

Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In ECCV, 2016.

Aniruddha Kembhavi, Minjoon Seo, Dustin Schwenk, Jonghyun Choi, Ali Farhadi, and Hannaneh Hajishirzi. Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension. In CVPR, 2017.

Kimi Team. Kimi-VL technical report. arXiv preprint arXiv:2504.07491, 2025. Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, et al. Kimi K2.5: Visual agentic intelligence. arXiv preprint

arXiv:2602.02276, 2026. Hynek Kydlíˇcek. Math-verify: Math verification library. https://github.com/huggingface/Math-Verify, 2025. Xin Lai, Junyi Li, Wei Li, Tao Liu, Tianjian Li, and Hengshuang Zhao. Mini-o3: Scaling up reasoning patterns and interaction turns for visual search. In ICLR, 2026. Jason Lau, Soumya Gayen, Asma Ben Abacha, and Dina Demner-Fushman. A dataset of clinically generated visual questions and answers about radiology images. Scientific Data, 2018.

Paul Lerner, Olivier Ferret, Camille Guinaudeau, Hervé Le Borgne, Romaric Besançon, Jose G. Moreno, and Jesus Lovon. ViQuAE, a Dataset for Knowledge-based Visual Question Answering about Named Entities. In SIGIR, 2022.

Ang Li, Charles Wang, Deqing Fu, Kaiyu Yue, Zikui Cai, Wang Bill Zhu, Ollie Liu, Peng Guo, Willie Neiswanger, Furong Huang, Tom Goldstein, and Micah Goldblum. Zebra-cot: A dataset for interleaved vision-language reasoning. In ICLR, 2026.

Kaixin Li, Ziyang Meng, Hongzhan Lin, Ziyang Luo, Yuchen Tian, Jing Ma, Zhiyong Huang, and Tat-Seng Chua. Screenspot-pro: Gui grounding for professional high-resolution computer use. In MM, 2025.

Lei Li, Yuqi Wang, Runxin Xu, Peiyi Wang, Xiachong Feng, Lingpeng Kong, and Qi Liu. Multimodal arxiv: A

dataset for improving scientific comprehension of large vision-language models. In ACL, 2024. Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. Junli Liu, Qizhi Chen, Zhigang Wang, Yiwen Tang, Yiting Zhang, Chi Yan, Dong Wang, Xuelong Li, and Bin

Zhao. Aerialvg: A challenging benchmark for aerial visual grounding by exploring positional relations. In ICCV, 2025a.

Yuhong Liu, Beichen Zhang, Yuhang Zang, Yuhang Cao, Long Xing, Xiaoyi Dong, Haodong Duan, Dahua Lin, and Jiaqi Wang. Spatial-ssrl: Enhancing spatial understanding via self-supervised reinforcement learning. In CVPR, 2026.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. In COLM, 2025b.

Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V Le, Barret Zoph, Jason Wei, et al. The flan collection: Designing data and methods for effective instruction tuning. In ICML, 2023.

Pan Lu, Liang Qiu, Jiaqi Chen, Tony Xia, Yizhou Zhao, Wei Zhang, Zhou Yu, Xiaodan Liang, and Song-Chun Zhu. Iconqa: A new benchmark for abstract diagram understanding and visual language reasoning. In NeurIPS, 2021.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In ICLR, 2024.

Xinyu Ma, Ziyang Ding, Zhicong Luo, Chi Chen, Zonghao Guo, Derek F. Wong, Xiaoyi Feng, and Maosong Sun. Deepperception: Advancing r1-like cognitive visual perception in mllms for knowledge-intensive visual grounding. arXiv preprint arXiv:2503.12797, 2025.

Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of ACL, 2022.

Ahmed Masry, Mohammed Saidul Islam, Mahir Ahmed, Aayush Bajaj, Firoz Kabir, Aaryaman Kartha, Md Tahmid Rahman Laskar, Mizanur Rahman, Shadikur Rahman, Mehrad Shahmohammadi, et al. Chartqapro: A more diverse and challenging benchmark for chart question answering. In Findings of ACL, 2025.

Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In WACV, 2022.

Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, Ping Luo, Yu Qiao, Qiaosheng Zhang, and Wenqi Shao. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2503.07365, 2025.

Earl K Miller and Jonathan D Cohen. An integrative theory of prefrontal cortex function. Annual review of neuroscience, 2001.

OLMo Team, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, Faeze Brahman, Finbarr Timbers, Hamish Ivison, et al. Olmo 3. arXiv preprint arXiv:2512.13961, 2025.

Roni Paiss, Ariel Ephrat, Omer Tov, Shiran Zada, Inbar Mosseri, Michal Irani, and Tali Dekel. Teaching clip to count to ten. In ICCV, 2023.

Bryan A. Plummer, Liwei Wang, Chris M. Cervantes, Juan C. Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. IJCV, 2017.

Valentina Pyatkin, Saumya Malik, Victoria Graf, Hamish Ivison, Shengyi Huang, Pradeep Dasigi, Nathan Lambert, and Hannaneh Hajishirzi. Generalizing verifiable instruction following. In NeurIPS, 2025.

Penghui Qi, Zichen Liu, Xiangxin Zhou, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Defeating the training-inference mismatch via fp16. arXiv preprint arXiv:2510.26788, 2025.

Yusu Qian, Hanrong Ye, Jean-Philippe Fauconnier, Peter Grasch, Yinfei Yang, and Zhe Gan. Mia-bench: Towards better instruction following evaluation of multimodal llms. In ICLR, 2025.

Runqi Qiao, Qiuna Tan, Peiqing Yang, Yanzi Wang, Xiaowan Wang, Enhui Wan, Sitong Zhou, Guanting Dong, Yuchen Zeng, Yida Xu, Jie Wang, Chong Sun, Chen Li, and Honggang Zhang. We-math 2.0: A versatile mathbook system for incentivizing visual mathematical reasoning. arXiv preprint arXiv:2508.10433, 2025.

Robert D Rogers and Stephen Monsell. Costs of a predictible switch between simple cognitive tasks. Journal of experimental psychology: General, 1995.

Pragya Paramita Sahu, Abhishek Raut, Jagdish Singh Samant, Mahesh Gorijala, Vignesh Lakshminarayanan, and Pinaki Bhaskar. Pop-vqa – privacy preserving, on-device, personalized visual question answering. In WACV, 2024.

Naganand Yadati Sanket Shah, Anand Mishra and Partha Pratim Talukdar. Kvqa: Knowledge-aware visual question answering. In AAAI, 2019.

Gabriel Herbert Sarch, Snigdha Saha, Naitik Khandelwal, Ayush Jain, Michael J Tarr, Aviral Kumar, and Katerina Fragkiadaki. Grounded reinforcement learning for visual reasoning. In NeurIPS, 2025.

Tom Schaul, Diana Borsa, Joseph Modayil, and Razvan Pascanu. Ray interference: a source of plateaus in deep reinforcement learning. arXiv preprint arXiv:1904.11455, 2019.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In ECCV, 2022.

Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In ICCV, 2019.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, et al. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. HybridFlow: A flexible and efficient RLHF framework. In EuroSys, 2025.

Xiangxi Shi and Stefan Lee. Benchmarking out-of-distribution detection in visual question answering. In WACV, 2024.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.

Alex Su, Haozhe Wang, Weiming Ren, Fangzhen Lin, and Wenhu Chen. Pixel reasoner: Incentivizing pixel-space reasoning via curiosity-driven reinforcement learning. In NeurIPS, 2025.

Jayant Sravan Tamarapalli, Rynaa Grover, Nilay Pande, and Sahiti Yerramilli. Countqa: How well do mllms count in the wild? arXiv preprint arXiv:2508.06585, 2025.

Liyan Tang, Grace Kim, Xinyu Zhao, Thom Lake, Wenxuan Ding, Fangcong Yin, Prasann Singhal, Manya Wadhwa, Zeyu Leo Liu, Zayne Sprague, et al. Chartmuseum: Testing visual reasoning capabilities of large vision-language models. In NeurIPS, 2025.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.

Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, Steven Bohez, Konstantinos Bousmalis, Anthony Brohan, Thomas Buschmann, Arunkumar Byravan, Serkan Cabi, Ken Caluwaerts, Federico Casarini, Oscar Chang, Jose Enrique Chen, Xi Chen, Hao-Tien Lewis Chiang, Krzysztof Choromanski, David D’Ambrosio, Sudeep Dasari, Todor Davchev, Coline Devin, Norman Di Palo, Tianli Ding, Adil Dostmohamed, Danny Driess, Yilun Du, Debidatta Dwibedi, Michael Elabd, Claudio Fantacci, Cody Fong, Erik Frey, Chuyuan Fu, Marissa Giustina, Keerthana Gopalakrishnan, Laura Graesser, Leonard Hasenclever, Nicolas Heess, Brandon Hernaez, Alexander Herzog, R. Alex Hofer, Jan Humplik, Atil Iscen, Mithun George Jacob, Deepali Jain, Ryan Julian, Dmitry Kalashnikov, M. Emre Karagozler, Stefani Karp, Chase Kew, Jerad Kirkland, Sean Kirmani, Yuheng Kuang, Thomas Lampe, Antoine Laurens, Isabel Leal, Alex X. Lee, Tsang-Wei Edward Lee, Jacky Liang, Yixin Lin, Sharath Maddineni, Anirudha Majumdar, Assaf Hurwitz Michaely, Robert Moreno, Michael Neunert, Francesco Nori, Carolina Parada, Emilio Parisotto, Peter Pastor, Acorn Pooley, Kanishka Rao, Krista Reymann, Dorsa Sadigh, Stefano Saliceti, Pannag Sanketi, Pierre Sermanet, Dhruv Shah, Mohit Sharma, Kathryn Shea, Charles Shu, Vikas Sindhwani, Sumeet Singh, Radu Soricut, Jost Tobias Springenberg, Rachel Sterneck, Razvan Surdulescu,

Jie Tan, Jonathan Tompson, Vincent Vanhoucke, Jake Varley, Grace Vesom, Giulia Vezzani, Oriol Vinyals, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Fei Xia, Ted Xiao, Annie Xie, Jinyu Xie, Peng Xu, Sichun Xu, Ying Xu, Zhuo Xu, Yuxiang Yang, Rui Yao, Sergey Yaroshenko, Wenhao Yu, Wentao Yuan, Jingwei Zhang, Tingnan Zhang, Allan Zhou, and Yuxiang Zhou. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025.

Yee Teh, Victor Bapst, Wojciech M Czarnecki, John Quan, James Kirkpatrick, Raia Hadsell, Nicolas Heess, and Razvan Pascanu. Distral: Robust multitask reinforcement learning. In NeurIPS, 2017.

Jingqi Tong, Jixin Tang, Hangcheng Li, Yurong Mou, Ming Zhang, Jun Zhao, Yanbo Wen, Fan Song, Jiahao Zhan, Yuyang Lu, Chaoran Tao, Zhiyuan Guo, Jizhou Yu, Tianhao Cheng, Zhiheng Xi, Changhao Jiang, Zhangyue Yin, Yining Zheng, Weifeng Ge, Guanhua Chen, Tao Gui, Xipeng Qiu, Qi Zhang, and Xuanjing Huang. Game-rl: Synthesizing multimodal verifiable game data to boost vlms’ general reasoning. arXiv preprint arXiv:2505.13886, 2025.

Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Jairam Vedagiri Iyer, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, et al. Cambrian-1: A fully open, visioncentric exploration of multimodal llms. In NeurIPS, 2024.

Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. VL-Rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. In NeurIPS, 2025.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. In NeurIPS, 2024a.

Peng Wang, Qi Wu, Chunhua Shen, Anthony Dick, and Anton Van Den Hengel. Fvqa: Fact-based visual question answering. TPAMI, 2018.

Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, et al. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. In NeurIPS, 2024b.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. In NeurIPS, 2022.

Luis Wiedmann, Orr Zohar, Amir Mahla, Xiaohan Wang, et al. FineVision: Open data is all you need. arXiv preprint arXiv:2510.17269, 2025.

Penghao Wu, Yushan Zhang, Haiwen Diao, Bo Li, Lewei Lu, and Ziwei Liu. Visual jigsaw post-training improves mllms. arXiv preprint arXiv:2509.25190, 2025a.

Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, and Yu Qiao. Os-atlas: A foundation action model for generalist gui agents. In ICLR, 2025b.

xAI. Realworldqa: A benchmark for real-world spatial understanding. https://huggingface.co/datasets/ xai-org/RealworldQA, 2024. Accessed: 2025-04-26.

Guowei Xu, Peng Jin, Ziang Wu, Hao Li, Yibing Song, Lichao Sun, and Li Yuan. Llava-cot: Let vision language models reason step-by-step. In ICCV, 2025.

Jianwei Yang, Reuben Tan, Qianhui Wu, Ruijie Zheng, Baolin Peng, Yongyuan Liang, Yu Gu, Mu Cai, Seonghyeon Ye, Joel Jang, Yuquan Deng, Lars Liden, and Jianfeng Gao. Magma: A foundation model for multimodal ai agents. In CVPR, 2025a.

Yue Yang, Ajay Patel, Matt Deitke, Tanmay Gupta, Luca Weihs, Andrew Head, Mark Yatskar, Chris CallisonBurch, Ranjay Krishna, Aniruddha Kembhavi, and Christopher Clark. Scaling text-rich image understanding via code-guided synthetic multimodal data generation. In ACL, 2025b.

Yuwei Yang, Zeyu Zhang, Yunzhong Hou, Zhuowan Li, Gaowen Liu, Ali Payani, Yuan-Sen Ting, and Liang Zheng. Effective training data synthesis for improving mllm chart understanding. In ICCV, 2025c.

Huanjin Yao, Jiaxing Huang, Wenhao Wu, Jingyi Zhang, Yibo Wang, Shunyu Liu, Yingjie Wang, Yuxin Song, Haocheng Feng, Li Shen, and Dacheng Tao. Mulberry: Empowering mllm with o1-like reasoning and reflection via collective monte carlo tree search. In NeurIPS, 2025.

Kaining Ying, Fanqing Meng, Jin Wang, Zhiqian Li, Han Lin, Yue Yang, Hao Zhang, Wenbo Zhang, Yuqi Lin, Shuo Liu, et al. Mmt-bench: A comprehensive multimodal benchmark for evaluating large vision-language models towards multitask agi. In ICML, 2024.

En Yu, Kangheng Lin, Liang Zhao, Jisheng Yin, Yana Wei, et al. Perception-R1: Pioneering perception policy with reinforcement learning. In NeurIPS, 2025a.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, et al. DAPO: An open-source LLM reinforcement learning system at scale. In NeurIPS, 2025b.

Weihao Yu, Zhengyuan Yang, Lingfeng Ren, Linjie Li, Jianfeng Wang, Kevin Lin, Chung-Ching Lin, Zicheng Liu, Lijuan Wang, and Xinchao Wang. Mm-vet v2: A challenging benchmark to evaluate large multimodal models for integrated capabilities. arXiv preprint arXiv:2408.00765, 2024.

Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. In ACL, 2025a.

Zihao Yue, Zhenru Lin, Yifan Song, Weikun Wang, Shuhuai Ren, et al. MiMo-VL technical report. arXiv preprint arXiv:2506.03569, 2025b.

Rowan Zellers, Yonatan Bisk, Ali Farhadi, and Yejin Choi. From recognition to cognition: Visual commonsense reasoning. In CVPR, 2019.

Chi Zhang, Feng Gao, Baoxiong Jia, Yixin Zhu, and Song-Chun Zhu. Raven: A dataset for relational and analogical visual reasoning. In CVPR, 2019.

Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, et al. Lmms-eval: Reality check on the evaluation of large multimodal models. In Findings of NAACL, 2025a.

Kaichen Zhang, Keming Wu, Zuhao Yang, Bo Li, Kairui Hu, Bin Wang, Ziwei Liu, Xingxuan Li, and Lidong Bing. Openmmreasoner: Pushing the frontiers for multimodal reasoning with an open and general recipe. In CVPR, 2026.

Yi-Fan Zhang, Huanyu Zhang, Haochen Tian, Chaoyou Fu, Shuangqing Zhang, Junfei Wu, Feng Li, Kun Wang, Qingsong Wen, Zhang Zhang, et al. Mme-realworld: Could your multimodal llm challenge high-resolution real-world scenarios that are difficult for humans? In ICLR, 2025b.

Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-ofthought reasoning in language models. TMLR, 2024.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025a.

Longtao Zheng, Zhiyuan Huang, Zhenghai Xue, Xinrun Wang, Bo An, and Shuicheng Yan. Agentstudio: A toolkit for building general virtual agents. In ICLR, 2025b.

Yuke Zhu, Oliver Groth, Michael Bernstein, and Li Fei-Fei. Visual7w: Grounded question answering in images. In CVPR, 2016.

# Appendix

This appendix provides detailed dataset documentation, training configurations, evaluation protocols, and supplementary analyses supporting the main paper:

- • § A presents detailed information on the 59 training datasets and 30 evaluation benchmarks in VeroEval, including question filtering prompts, answer filtering rules, and data mixture weighting schemes.
- • § B describes the training setup, including the system prompt, RL hyperparameters, reward formulation, the LLM-judge prompt, and supervised fine-tuning baselines.
- • § C details the protocols, decoding settings, and benchmark-specific choices for each model family.
- • § D reports additional analyses on image size distributions, cognitive behavior definitions and presence rates, behavioral skill extraction, and skill-level probe experiments across task categories.
- • § E points to representative reasoning traces from Vero across all six task categories and highlights how the model adapts its reasoning strategies to different domains.

- A Dataset Details 27

- A.1 Training Dataset . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- A.2 Evaluation Datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- A.3 Question Filtering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- A.4 Answer Filtering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- A.5 Data Mixture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32

- B Training Details 33

- B.1 System Prompt . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
- B.2 Reinforcement Learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
- B.3 Reward . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
- B.4 Supervised Fine-tuning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37

- C Evaluation Details 37
- D Additional Analyses 38

- D.1 Image Size . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
- D.2 Behavioral Definitions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
- D.3 Cognitive Behaviors . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39
- D.4 Behavioral Skill Extraction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39
- D.5 Skill Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40

- E Reasoning Traces 42

##### A Dataset Details

###### A.1 Training Dataset

- In Table A1, we provide additional details on each data source retained in our final RL training mixture.

Dataset Ret. Answer type Reward type(s) Grounding, Counting & Search AerialVG (Liu et al., 2025a) 12,634 bbox coordinates grounding GroundUI (Zheng et al., 2025b) 12,064 click coordinates clicking MultiHop (Li et al., 2026) 6,316 integer count counting Objects365-QA (Shao et al., 2019) 12,632 bbox coordinates grounding OOD-VQA (Shi and Lee, 2024) 5,028 integer count counting OS-ATLAS (Wu et al., 2025b) 9,515 click coordinates clicking Pixel Reasoner (Su et al., 2025) 4,337 short text or count search PixMo (Deitke et al., 2025) 12,631 integer count counting RefCOCOg (Kazemzadeh et al., 2014) 6,882 bbox coordinates grounding TallyQA (Acharya et al., 2019) 12,631 integer count counting Visual Probe (Lai et al., 2026) 5,330 short text or count search Chart & OCR CoSyn-Chart (Yang et al., 2025b) 11,514 numeric or short text numeric, string match CoSyn-Diagram (Yang et al., 2025b) 7,433 numeric or short text numeric, string match CoSyn-Table (Yang et al., 2025b) 12,226 numeric or short text numeric, string match ArxivQA (Li et al., 2024) 12,225 MC or numeric multiple choice, numeric ChartQA (Masry et al., 2022) 12,224 numeric or short text numeric, string match ECD-VQA (Yang et al., 2025c) 12,224 numeric or short text numeric, string match EvoChart (Huang et al., 2025) 12,223 numeric or short text numeric, string match InfographicVQA (Mathew et al., 2022) 12,223 numeric or short text numeric, string match ReachQA (He et al., 2025) 7,708 numeric or short text numeric, string match Knowledge & Recognition A-OKVQA (Schwenk et al., 2022) 2,744 short text or numeric list string match, numeric GQA (Hudson and Manning, 2019) 6,120 multiple-choice option multiple choice IconQA (Lu et al., 2021) 12,755 MC, numeric, or text MC, numeric, string match Indoor-QA (Berke, 2024) 2,547 short text list string match KVG (Ma et al., 2025) 12,753 click coordinates clicking KVQA (Sanket Shah and Talukdar, 2019) 6,689 short text or numeric list string match, numeric PopVQA (Sahu et al., 2024) 12,753 short text or numeric list string match, numeric VCR (Zellers et al., 2019) 12,752 multiple-choice option multiple choice ViQuAE (Lerner et al., 2022) 1,859 short text or numeric list string match, numeric Visual7W (Zhu et al., 2016) 12,751 multiple-choice option multiple choice VizWiz (Gurari et al., 2018) 3,526 short text list string match VQAv2 (Goyal et al., 2017) 12,751 short text or numeric list string match, numeric Spatial & Action GameQA (Tong et al., 2025) 18,847 short text or symbol string match Magma-AITW (Yang et al., 2025a) 10,800 structured action JSON web action Magma-Mind2Web (Yang et al., 2025a) 5,298 structured action JSON web action Robo2VLM (Chen et al., 2025) 2,350 multiple-choice option multiple choice Spatial-SSRL (Liu et al., 2026) 18,847 MC, ordered number list multiple choice, number list ST-VQA (Biten et al., 2019) 6,168 multiple-choice option multiple choice

- Visual Jigsaw 2D (Wu et al., 2025a) 18,845 ordered number list number list
- Visual Jigsaw 3D (Wu et al., 2025a) 18,845 ordered number list number list

STEM CoSyn-Math (Yang et al., 2025b) 16,048 numeric answer numeric AI2D (Kembhavi et al., 2016) 2,194 multiple-choice option multiple choice Geo170K (Gao et al., 2025) 16,047 multiple-choice option multiple choice GeomVerse (Kazemi et al., 2023) 8,895 numeric answer numeric GeoQA+ (Cao and Xiao, 2022) 5,665 multiple-choice option multiple choice MMK12 (Meng et al., 2025) 6,869 MC or numeric multiple choice, numeric PathVQA (He et al., 2020) 1,108 short text string match RAVEN (Zhang et al., 2019) 8,021 multiple-choice option multiple choice TQA (Kembhavi et al., 2017) 11,373 multiple-choice option multiple choice VisualWebInstruct (Jia et al., 2025) 16,042 MC, numeric, or text MC, numeric, string match VQA-RAD (Lau et al., 2018) 678 short text str. match We-Math 2.0 Pro (Qiao et al., 2025) 2,841 MC, numeric, or text MC, numeric, string match We-Math 2.0 Std (Qiao et al., 2025) 4,219 MC, numeric, or text MC, numeric, string match Captioning & Instruction Following PixMo-AskAnything (Deitke et al., 2025) 16,667 open-ended response LLM-as-judge PixMo-CapQA (Deitke et al., 2025) 16,667 free-form caption LLM-as-judge PixMo-Cap (Deitke et al., 2025) 16,667 free-form caption LLM-as-judge MM-RLVR-IFEval 16,667 instruction-following text IF MMIF-23K (Ding et al., 2025) 16,667 instruction-following text IF & LLM-as-judge Flickr30K (Plummer et al., 2017) 16,667 free-form caption LLM-as-judge

- Table A1. Retained training datasets used in our RL mixture. Retained sizes are taken from the composition source used to generate the training-data figure. Captioning and instruction-following retained sizes are rounded display values.

###### A.2 Evaluation Datasets

- In Table A2, we summarize the 30 evaluation benchmarks in VeroEval and give a short description of what each benchmark tests.

Benchmark Description Chart & OCR ChartQA-Pro (Masry et al., 2025) diverse chart question answering ChartQA (Masry et al., 2022) chart reasoning and question answering InfoVQA (Mathew et al., 2022) infographic question answering CharXiv (Wang et al., 2024b) scientific chart understanding ChartMuseum (Tang et al., 2025) chart visual reasoning EvoChart (Huang et al., 2025) real world chart understanding STEM MMMU-Pro Standard (Yue et al., 2025a) multidisciplinary multimodal MCQA MMMU-Pro Vision (Yue et al., 2025a) vision focused multidisciplinary MCQA MathVision (Wang et al., 2024a) multimodal mathematical reasoning MathVistatestmini (Lu et al., 2024) visual mathematical reasoning Spatial & Action Blink (Fu et al., 2024) fine grained visual perception ERQA (Team et al., 2025) embodied reasoning for robotics GameQALite (Tong et al., 2025) game logic reasoning EmbSpatial (Du et al., 2024) embodied spatial understanding CVBench (Tong et al., 2024) 2D and 3D visual understanding Knowledge & Recognition RealWorldQA (xAI, 2024) real world understanding SimpleVQA (Cheng et al., 2025a) factual visual question answering FVQA (Wang et al., 2018) knowledge intensive visual question answering MM-Vet V2 (Yu et al., 2024) perceptual multimodal capabilities Grounding, Counting & Search CountBenchQA (Paiss et al., 2023) object counting CountQA (Tamarapalli et al., 2025) counting in the wild MMERealWorld (Zhang et al., 2025b) high resolution real world reasoning VStarBench (Cheng et al., 2025b) high resolution visual search AerialVG (Liu et al., 2025a) aerial visual grounding VisualProbe (Lai et al., 2026) high resolution visual search ScreenSpot (Cheng et al., 2024) GUI grounding ScreenSpotPro (Li et al., 2025) high resolution GUI grounding Captioning & Instruction Following MM-MTBench (Ying et al., 2024) multitask multimodal chat evaluation MIABench (Qian et al., 2025) multimodal instruction following MMIFEval (Ding et al., 2025) verifiable multimodal instruction following

- Table A2. Evaluation benchmarks in VeroEval, organized by task category. The right column gives a short description of what each benchmark evaluates.

Dataset preprocessing. In addition to model-based filtering, we apply lightweight dataset preprocessing for recurring annotation artifacts or formatting inconsistencies.

- • Answer normalization and prompt cleanup. We strip trailing answer instructions from prompts and normalize answers into short, verifiable forms when possible. For example, for chart subsets we remove trailing instructions, and extract final option letters or short answers from templated solutions.
- • Dataset-specific question rewrites and format conversions. Some datasets receive deterministic rewrites instead of removal. In GameQA, we prepend or insert short clarifications for subsets such as 2D Turing Machine. For open-ended chart reasoning subsets, such as ReachQA, we rewrite long answers into a single verifiable query. GeoQA+ is translated to English and reduced to a single multiplechoice letter, KVG is converted from bbox-markup prompts into point-clicking supervision, and VizWiz retains only answerable, non-yes/no questions with confident answers.
- • Creation of MM-RLVR-IFEval. We construct the MM-RLVR-IFEval training data to create a multimodal version of IF-RLVR (Pyatkin et al., 2025). We sample prompts and images in equal proportions from A-OKVQA (Schwenk et al., 2022), pixmo-ask-model-anything-images (Deitke et al., 2025), pixmocap-qa-images (Deitke et al., 2025), and cambrian (Tong et al., 2024). For each record, we sample between 1 and 10 random, conflict-checked instruction-following constraints drawn from the verifiable instruction sets of IF-RLVR (Pyatkin et al., 2025) and MMIF (Ding et al., 2025), and append them as bullet-point requirements directly to the prompt. We then use Qwen3-235B-Instruct to rephrase both the base question and the attached constraints into more natural language, while preserving all entities, keywords, numbers, special tokens, and instruction semantics.

###### Prompt for Model-Based Question Filtering

Given the image and the question, your task is to independently evaluate the following criteria and set each corresponding flag. A flag should be "true" when the issue is present (i.e., the item should be filtered on that

criterion) and "false" otherwise. Provide one concise overall explanation in "reason" summarizing the main driver(s) for any "true" flags. Evaluation criteria:

- 1. Relevance Filter ( relevance_filter ) — Is the image related to the question?

• Set "true" if the image does not depict what the question refers to, or the entities/attributes asked about are absent. Examples omitted for brevity

- 2. Ambiguity/Vagueness Filter ( ambiguous_filter ) — Is the question too vague, unclear, or not actually a question?

• Set "true" for unclear referents (“this”, “that”), incomplete/elliptical prompts, or non-question content (e.g.,

raw lists/tables with no query). Examples omitted for brevity

- 3. Language Filter ( language_filter ) — Is the question not in English?

• Set "true" if any required reading/understanding is in a language other than English.

- 4. Verifiability / Single-Answer Filter ( verifiable_filter ) — Can the question be answered with a single, objectively verifiable answer solely from visible content in the image?

• Set "true" if the answer would require external knowledge, speculation, non-visible attributes, prediction-

s/counterfactuals, or if multiple plausible answers exist from the same visual evidence. Examples omitted for brevity

- 5. Numeric Precision / Readability Filter ( number_precision_filter ) — Does the question demand a numeric precision that the visual cannot unambiguously support?

- • Set "true" if exact integers/decimals or derived metrics (e.g., average annual growth rate, percentage change) require precise values that are not explicitly labeled or legibly recoverable from the chart/axes/points.

- • Even if the question uses approximation language (“about,” “approximately,” “nearest”), set "true" if the underlying precise value cannot be confidently determined.

- • Things that can be visually estimated but may have slight ambiguity in numerical answer, mark "true" .

Examples omitted for brevity Decision guidance:

- • Multiple flags may be "true" simultaneously.

- • If any of the above flags is "true" , the item is considered filtered for that dimension. Use "reason" to summarize the primary cause(s).

###### Output Format (JSON):

{

"relevance_filter": "true", "ambiguous_filter": "false", "language_filter": "false", "verifiable_filter": "true", "number_precision_filter": "true", "reason": "Briefly explain the main reason(s) these filter(s) were triggered."

}

###### A.3 Question Filtering

Question filtering prompt. We provide the prompt for model-based question filtering in Listing A.2; the exact, copyable versions of all prompts (including dataset-specific filtering variants not shown here) are available in our code repository.3 The model flags samples based on five independent boolean filter flags: relevance_filter, ambiguous_filter, language_filter, verifiable_filter, and number_precision_-

filter. We remove a sample if any of the flags are returned as "true". For certain datasets, we apply lightweight dataset-specific rules that ignore particular filter triggers when those triggers arise from known dataset characteristics rather than genuine annotation problems. For example, for Knowledge & Recognition, we instruct the model to not flag ambiguous_filter if it requires external knowledge.

Question filtering examples. We illustrate four representative examples caught by our question filtering pipeline. These questions appear well-formed but are unsuitable for reliable reward computation. Our filtering pipeline successfully identifies such cases, enabling us to curate a high-quality training dataset.

Unsupported numeric precision. One example is a pie chart from EvoChart that displays category names but no percentage labels. The question asks for the proportion of “Virtual and Augmented Reality,” with a ground-truth answer of 20.76%. Since the chart provides no numeric annotations, the precise target cannot be visually verified. Our filtering pipeline flags such questions as requiring unsupported numeric precision.

Question–image mismatch. A second example is a fluorescence microscopy image of chromosomes with X and Y chromosome paint labels. The question asks “What is the country of citizenship of the subject of this image?” Since the image contains no human subject, the question is entirely irrelevant to the visual content. Our filtering detects such question–image mismatches and removes them.

Ambiguous reference. A third example is a cemetery scene containing multiple distinct structures, including gravestones, Celtic crosses, a round tower, and an angel statue. The question asks “In what year was the place in this image created?” with a ground-truth answer of 1832. However, “the place” is ambiguous: it could refer to the cemetery, the tower, or any individual gravestone, each potentially having a different creation date. Our filtering correctly flags this question as unanswerable due to the ambiguous reference.

Hidden external knowledge. A fourth example is a portrait painting paired with the question “A part of what collection is the painting in this image?” The ground-truth answer references specific museum collections (Gemäldegalerie Alte Meister, Hessen Kassel Heritage), but this provenance information is not visible in the image. For task categories other than Knowledge & Recognition, our pipeline filters such questions as requiring external knowledge that cannot be verified from pixels alone.

###### A.4 Answer Filtering

As explained in Section 3.2, we perform answer filtering on individual training examples to normalize the answer format before reward computation and remove answers that cannot be reliably verified by our reward functions. Below, we provide additional details on the common rules and examples for answer filtering.

Answer filtering rules by answer type. Because ground-truth answers in our source datasets are stored in heterogeneous formats, we apply type-specific answer filtering before reward computation. An LLM-based classifier first assigns each ground truth to one of four answer types (multiple-choice, numeric, string, or None (unresolvable)) and then a rule-based normalizer rewrites the answer into a standard form that our reward verifiers can consume.

Multiple-choice. Ground truths expressed as labeled options (e.g., “a) 67.37”, “Option (C)”, “Figure (2)”, “3.”) are normalized to a single uppercase letter (A, B, C, ...). The normalizer handles parenthesized letters, numbered options mapped positionally to letters, and text options that reference labeled figures or graphs. This is the most common reformatting rule, applied to the majority of reformatted samples.

3https://github.com/zlab-princeton/vero

Reason Description Example Single-ground-truth datasets Multi-value answers (∼300) Ground truth contains multiple distinct values that

AC=4, BD=4

cannot be reduced to a single verifiable target.

Ambiguous text labels (∼300) Descriptive phrases requiring fuzzy or semantic matching beyond exact string comparison.

“Isosceles triangle”

Unsupported notation (∼300) Scientific notation or symbolic algebraic expressions outside our numeric parser’s scope.

b = a cosC

Empty / invalid GT Ground truth is missing, empty, or malformed, making reward computation impossible.

“Empty case”

Unit mismatch Unit is inconsistent with the question context or cannot be cleanly stripped to a numeric value.

Mass answered in cm

Out-of-range values Numeric ground truth falls outside the valid range for the quantity asked about.

r = 1.215 (correlation)

Vector / complex answers Multi-component quantities that cannot be reduced to a single scalar.

(3, −2,5); 2+3i

Non-standard units Numeric value mixed with a unit descriptor our parser does not handle.

“1.70 million”

Non-task questions Prompt requests instruction or explanation rather than a verifiable answer.

“Explain how to solve...”

Multi-annotator datasets (e.g., VQAv2, VizWiz, A-OKVQA) Inconsistent answers (closed) Annotators gave mutually exclusive responses

- annotator 1: “white,blue”,
- annotator 2: “red”

with no dominant consensus for a single-answer question.

Inconsistent answers (open) Responses span unrelated concepts with no dominant semantic cluster.

“fish”, “float”, “tow”

Answer–question type mismatch

Ground-truth type does not match what the question semantically requires.

Question asks about a person

→ Ground truth = “Italy” Unanswerable markers Annotators flagged the question as unanswerable;

“unanswerable” tag

remaining answers are insufficient.

Open-ended descriptions Free-form description question with no single canonical answer for exact matching.

“Please describe this photo”

Composite questions Multiple sub-questions whose interleaved answers cannot be parsed into a single target.

“What is this? What color?”

Positional descriptors Ground truth is a spatial reference rather than an identifying entity.

“right”, “in the back”

- Table A3. Common reasons for answer filtering. The top section covers single-ground-truth datasets (frequency estimates from a manual sample); the bottom section covers multi-annotator datasets.

Numeric. Numeric ground truths are stripped of surrounding units, currency symbols, degree markers, and LaTeX formatting to yield a plain decimal value. For example, “$327,000” becomes 327000, “60◦” becomes 60, and “8 V” becomes 8. Thousand separators are removed, fractions are converted to decimals (e.g., “8/3” → 2.6667), and currency prefixes are dropped (e.g., “$222.14” → 222.14).

String. Free-form text answers undergo lowercasing and whitespace normalization to enable case-insensitive

exact matching at reward time (e.g., “Coronal” → coronal).

None (unresolvable). Answers that the classifier cannot confidently assign to any of the above types—such as multi-part answers (e.g., “(1) 3, (2) 120”), coordinate tuples (e.g., “(5.2,0)”), or single ambiguous tokens—are assigned type None and filtered from the training set.

Common reasons for answer filtering. Answers that cannot be reliably verified by our programmatic reward functions are removed during answer filtering. Table A3 summarizes the primary filtering reasons. The top section lists reasons common to single-ground-truth datasets, in decreasing order of frequency. For datasets with multi-annotator ground truths (e.g., VQAv2, VizWiz, A-OKVQA), additional reasons arise from annotator disagreement and question–answer alignment issues, shown in the bottom section.

###### A.5 Data Mixture

In Section 3.2 of the main paper, we examine how the task-category sampling distribution affects RL training. Here we describe the procedure used to construct each weighting scheme reported for that experiment.

Per-domain statistics. We collect three statistics for each domain d. Two of these, accuracy and reasoning length, require a profiling run: we train the base model (Qwen2.5-VL-7B-Instruct) for one epoch on a 100K-sample subset using uniform category weights and measure:

- • Accuracy (accd): average reward on the held-out verification set for domain d.
- • Reasoning length (Ld): mean number of tokens on the held-out verification set inside the <think> block.
- • Image area (Ad): mean pixel area of the input images (before any resizing), computed directly from the original training set.

Weighting schemes. We run the data mixture experiment on five domains: Chart & OCR, Grounding, Counting & Search, Knowledge & Recognition, Spatial & Action, and STEM. Each non-uniform scheme defines a per-domain ratio rd proportional to a power-law function of one of the profiling statistics. The exponent α controls how aggressively the distribution deviates from uniform. We tune α so that the ratio between the most- and least-weighted domains equals 1.6, a moderate spread that allows meaningful reallocation without starving any single category:

maxd rd mind rd

= 1.6.wwwwwwwwwww (5)

Concretely, the four weighting schemes and the ablation without Knowledge & Recognition are:

- 1. Equal ratios (uniform): rd = 0.20 for all five domains.
- 2. Difficulty-weighted (rd ∝ (1 − accd)α, α = 0.475): Up-weights domains where the model performs poorly after the profiling run. Spatial & Action receives the largest share (0.273) due to its low initial accuracy, while STEM receives the smallest (0.170).
- 3. Reasoning-length-weighted (rd ∝ Lαd, α = 0.144): Up-weights domains whose responses require longer chains of thought. Chart & OCR and Spatial & Action receive the largest shares (∼0.23 each), while

Knowledge & Recognition receives the smallest (0.148). We also evaluate the inverse scheme (rd ∝ L−α

d ), which favors domains with shorter reasoning traces.

- 4. Image-area-weighted (rd ∝ Aαd, α = 0.443): Up-weights domains with larger input images. Grounding, Counting & Search receives the largest share (0.244), while Spatial & Action receives the smallest (0.153).
- 5. Without Knowledge & Recognition: Sets rd = 0 for Knowledge & Recognition and distributes weight

equally among the remaining four domains (rd = 0.25 each). This ablation tests whether the lowest-gain category can be dropped without harming overall performance.

##### B Training Details

- B.1 System Prompt We provide the system prompt for Vero during training and evaluation in Listing B.1.

###### System Prompt for Vero

You are a helpful, conversational assistant tasked with answering a question about an image. Your response must include two parts:

- 1. Reasoning: A detailed, free-flowing chain of thought enclosed in <think> and </think> tags.

- 2. Final Answer: A clear, conversational response enclosed in <answer> and </answer> tags, using \boxed{} notation when the question has a definitive answer.

###### Reasoning Instructions

- • The reasoning section must be inside <think> ... </think> tags.

- • The reasoning should resemble a stream of consciousness: explore, test hypotheses, backtrack if necessary, reflect, and refine.
- • Let the reasoning flow naturally while progressing toward a conclusion.
- • Use reasoning strategies such as:

- • Planning: outline possible approaches before committing.
- • Exploration: consider multiple image regions or interpretations, even unlikely ones.
- • Evaluation: compare alternatives and verify against visual evidence.
- • Reflection: revisit earlier ideas if they may still be viable.

- • Thoroughly examine and cross-check relevant image regions before narrowing down.
- • If the image is ambiguous, make a reasonable inference based on visual and contextual cues.
- • End the reasoning once you are confident in the conclusion.

###### Final Answer Instructions

- • The answer section must be enclosed in <answer> ... </answer> tags.

- • The <answer> section should stand on its own as a response to the user: it must provide necessary context and justification so that a reader can understand and verify the conclusion without reading <think> .

• Do NOT refer to the <think> section (avoid phrases like “as explained above” or “from the reasoning”).

- • Boxed result:
- • If the question has a definitive, concise answer (a number, word, phrase, or label), include a conversational, natural response followed by exactly one boxed result using LaTeX: \boxed{final_result} .

- • If the question is open-ended, subjective, or does not yield a concise final result, omit the boxed notation.

###### Format Example

<think> Detailed reasoning goes here... </think> <answer> Self-contained response goes here... Following the response, if a concise final result exists, include: \boxed{final_result}. If open-ended or no concise result, respond naturally without \boxed. </answer>

###### B.2 Reinforcement Learning

GSPO algorithm and objective. The GSPO objective is a clipped surrogate loss aggregated as the mean-ofsequence-means (seq-mean-token-mean). Given a group of G rollouts {yi}iG=1 for a prompt (v, q), define the per-response sequence-average log-probability difference:

|yi|

1 |yi|

∆¯ i =

#### ∑

log πθ(yi,t | v, q, yi,<t) − log πθold(yi,t | v, q, yi,<t) . (6)

t=1

The sequence-level importance ratio at token t is then formed by routing the gradient through the sequence average while keeping the token-level log-prob differentiable:

si,t(θ) = exp sg(∆¯ i) + log πθ(yi,t) − sg log πθ(yi,t) , (7) where sg denotes stop-gradient. The GSPO objective is:

|yi|

G

1 G

1 |yi|

#### ∑

#### ∑

min si,t(θ) Ai, clip si,t(θ), 1−εlow, 1+εhigh Ai , (8)

J (θ) =

t=1

i=1

where the normalized group advantage is:

ri − µg σg + ϵ

1 G

, µg =

Ai =

G

rj, σg = std {rj}Gj=1 . (9)

#### ∑

j=1

Training hyperparameters. We detail the RL training hyperparameters in Table B1 and per-model configurations in Table B2. All models are trained for 2,343 steps using VeRL (Sheng et al., 2025) with FSDP2 on 8 GPUs. In preliminary experiments, we found that Qwen models exhibit slightly more stable training under fp16, following Qi et al. (2025), while MiMo-VL trains stably in bf16.

Hyperparameter Value

Framework VeRL FSDP strategy fsdp2 Rollouts per prompt (G) 8 Train batch size 256 PPO mini-batch size 128 Learning rate 1 × 10−6 LR warmup steps 40

Clip lower (εlow) 0.0003 Clip upper (εhigh) 0.0004 KL coefficient 0 Rollout temperature 1.0

- Table B1. RL training hyperparameters for all Vero models. Training uses VeRL with GSPO (Zheng et al., 2025a), asymmetric clipping (εlow < εhigh), and no KL penalty to allow less-restricted policy updates.

###### Base model GPUs Steps Ctx. Max px. Dtype Coords

MiMo-VL-7B-SFT 8×H100 2,000 28,672 30722 bf16 absolute Qwen2.5-VL-7B-Inst. 8×H100 2,000 24,576 30722 fp16 absolute Qwen3-VL-8B-Inst. 8×H200 2,000 36,864 40962 fp16 norm.0–1k Qwen3-VL-8B-Think. 8×H200 2,000 36,864 40962 fp16 norm.0–1k

- Table B2. Base-model training configurations for the Vero variants, including context length, maximum image resolution, precision, and coordinate format. Qwen-family models use fp16 for improved training stability (Qi et al., 2025).

- B.3 Reward The total reward for a response y is:

R(y, y∗) = (1 − α) Racc(y, y∗) + α Rfmt(y) + Roverlong(y), (10)

with α = 0.2. For tasks combining programmatic instruction-following with open-ended judgment, the blended accuracy score is:

R˜acc(y, y∗) = w Rinst(y) + (1 − w) Rjudge(y), w = 0.5. (11) The LLM judge produces a score on a 1–10 scale, normalized to [0,1] as (s − 1)/9.

Overlong penalty. To discourage excessively long responses, we use the soft penalty from Yu et al. (2025b) as a linear ramp in the buffer zone [Lmax − B, Lmax]:

|y| − (Lmax − B) B

Roverlong(y) = min −

λ, 0 , (12)

where B = 2048, Lmax = max_tokens, and λ = 1.0.

Format reward. Rfmt requires the response to follow the format <think>. . .</think><answer>. . .</answer> with non-empty think content; responses that violate this structure receive Rfmt = 0. Given valid structure, Rfmt = 1 by default. For discrete symbolic answer types (string match, multiple choice, numeric, list match, counting, ordering, search, web action), a single valid \boxed{...} in the answer block is additionally required for Rfmt = 1; its absence or the presence of multiple \boxed{...} expressions reduces Rfmt to 0.5. For grounding and clicking, the presence of multiple \boxed expressions similarly reduces Rfmt to 0.5.

Training judge. We include our training judge prompt in Listing B.3. We adapt the LLM judge prompt from OLMo3 (OLMo Team et al., 2025). For training-time LLM-as-judge rewards, we use Qwen3-32B served via vLLM with thinking disabled. We set judge temperature to 0.7 and judge max tokens to 1,024.

###### Prompt for LLM Judge Reward

Please act as an impartial judge and evaluate the quality of the answer provided by an AI assistant to the conversation history leading up to the answer displayed below. Judge whether the provided answer is good by comparing it to the reference answer. Notes:

- • Besides comparing to the reference answer, your evaluation should consider factors such as the naturalness, coherence, helpfulness, relevance, accuracy, creativity, appropriate level of detail, and how well the response satisfies the user’s explicit constraints or accurately follows their instructions.
- • The AI answer may use \boxed{} exactly once for a definitive concise answer (number, word, phrase, or

label), or not at all if the question is open-ended or subjective. Penalize if it is used on intermediate results or explanations.

- • Note that sometimes the reference answer is not the only answer. So any valid variation of the reference answer is also acceptable.
- • The conversation involves an image that is not shown to you. Use the reference answer as ground truth for any visual content.
- • Automatic Failure Conditions (Score = 1): The following violations require an automatic score of 1 under any circumstance, regardless of the overall quality of the answer. No partial credit may be given if any of these appear.

- 1. Notes to the judge or self-talk: Any meta commentary, internal reasoning, notes that are directed towards the judge, or reflective statements about how the answer was constructed automatically results in a score of 1.

Examples omitted for brevity

- 2. Self-evaluative or compliance-asserting statements: Any claim about the answer’s correctness, completeness, quality, adherence to constraints, or deservingness of a high score automatically results in a score of 1. Do not consider such claims as mitigating factors.

Examples omitted for brevity Judges must explicitly check for these violations. If any instance is present, the score must be 1.

- • Unnatural Penalty Condition (Score Reduction Required): The score must be reduced if the answer includes gratuitous verbosity, repetition, rhetorical padding, inflated phrasing, or stylistically unnatural language that does not add informational value. Explanations, intermediate reasoning steps, and brief summaries are permitted when they directly support the answer and are proportionate to the complexity of the question.
- • For context, provided below is the Conversation History, AI Answer, and Reference Gold Answer.

[Conversation History START] {input} [Conversation History END]

[AI Answer START] {output} [AI Answer END]

[Reference Gold Answer START] {label} [Reference Gold Answer END]

Please adhere to the following format.

- • Respond in JSON format.
- • Begin your evaluation by providing a short explanation in the "REASONING" key.

- • Be as objective as possible. After providing your short explanation, please output a score on a scale of 1 to 10 in the "SCORE" key.

[Your judgement] Respond in JSON format: {"REASONING": "[...]", "SCORE": "<your-score>"}

###### Hyperparameter Value

Epochs 1 Weight decay 0.01 Warmup ratio 0.03 Batch size 128 Scheduler Cosine Precision bf16 Flash attention fa2 Max sequence length 32768

Table C1. SFT hyperparameters.

Setting Value Max new tokens 16,384 Temperature 0.6 Top-p 1.0 Max image pixels 4096 × 4096

Setting Value Max new tokens 16,384 Temperature 1.0 Top-p 0.95 Top-k 20 Presence penalty 1.5 Max image pixels 4096 × 4096

Table C2. Evaluation settings for Qwen2.5-VL and MiMo-VL.

Table C3. Evaluation settings for Qwen3-VL.

Math_verify reward. In Section 4.2, we ablate our reward design with a math_verify baseline (Table 4(b)). The baseline replaces our full reward router with a single unified verifier built on the open-source MATH-VERIFY library (Kydlícˇek, 2025). Our task-routed reward outperforms math_verify on every category, improving the overall average from 51.8 to 57.2 (+5.4), with the largest gain on Captioning & Instruction Following (70.6 vs. 34.3). This highlights the need for task-specific reward routing in multi-task RL.

The math_verify verifier performs reward computation as follows: 1. Case-insensitive string match. We first try naive string matching. If the lowercased, whitespace-stripped prediction equals the lowercased ground truth, the reward is 1. 2. Symbolic parsing and verification. Both the ground truth and the prediction are passed to MATH-VERIFY’s parse function, which includes robust parsing of numerical, symbolic, and multiple choice answers embedded in text. The parsed representations are then compared via MATHVERIFY’s verify function to robustly match the prediction with the ground truth after normalization. If verify returns True, the reward is 1. Otherwise, the reward is 0.

###### B.4 Supervised Fine-tuning

We use supervised fine-tuning as a baseline in our SFT vs. RL ablation (Table 4(a), Section 4.2). Table C1 summarizes the shared SFT hyperparameters. For each baseline, we sweep learning rates over {1e−6,1e−7,5e−7,5e−6}. SFT on our data outperforms FineVision SFT (52.8 vs. 46.2), but RL yields substantially larger gains across all categories (57.2 overall, +4.8 over the base model).

##### C Evaluation Details

We use two decoding setups depending on the model trained. Qwen2.5-VL and MiMo-VL trained models follow the Qwen2.5-VL (Bai et al., 2025b) recommended decoding setup. Qwen3-VL trained models follow the decoding setup reported in the Qwen3-VL report (Bai et al., 2025a). In both cases, evaluation uses one sampled decode per example. Tables C2 and C3 summarize the model-family-specific sampling parameters and the shared runtime settings. We use Qwen3-32B with thinking disabled as the evaluation LLM judge when an LLM judge is required. For benchmarks requiring a VLM judge, we use Qwen3-VL-32B-Instruct. For judges, we use sampling parameters set to Temperature=0.7, TopP=0.8, TopK=20, and MinP=0. In the main results table, Vero variants are reported as Avg@5 with SEM across runs. Baseline scores are single reported or evaluated values unless otherwise indicated.

Benchmark-specific choices. We use the Standard and Vision splits of MMMU-Pro and MathVistatestmini, the English subset of SimpleVQA, and GameQALite, where Lite denotes sampling 2,633 examples. For AerialVG, we report mean IoU. For MM-MTBench, we rescale to a 0 to 100 scale using score100 = 100 · (x − 1)/9. For VisualProbe, we report the average of easy, medium, and hard subsets.

##### D Additional Analyses

###### D.1 Image Size

Figure D2 shows clear variation in average image area across task categories. Captioning & Instruction Following and Grounding, Counting & Search use the largest images at 1.53 ± 0.11 million and 1.50 ± 0.07 million pixels, followed by Chart & OCR at 1.18 ± 0.03 million pixels. Knowledge & Recognition and STEM fall in the middle at 0.91 ± 0.05 million and 0.81 ± 0.05 million pixels, while Spatial & Action uses the smallest images at 0.52 ± 0.01 million pixels.

###### D.2 Behavioral Definitions

Prompt adaptation. Following Kargupta et al. (2025), we conduct an automated annotation of the reasoning traces using Qwen3-32B (Bai et al., 2025a) as a strong evaluator model. We extend their taxonomy to better capture multimodal reasoning strategies by augmenting the original 28 text-centric behaviors with six supplementary visual-analysis capabilities: arithmetic-calculation, mental-imagery-simulation, perception-thenreasoning, systematic-regional-synthesis, visual-foraging, and visual-reference-or-grounding. In total, the evaluator monitors for 34 distinct cognitive capabilities.

In accordance with the original protocol, we formulate an individualized prompt per capability. Each prompt provides a precise definition, explicit criteria for behavioral evidence, and few-shot examples for in-context learning. Guided by these instructions, the evaluator model assesses the reasoning trace and outputs its final analysis in a structured JSON format, consisting of a final score indicating presence or absence and a supporting explanation.

To improve efficiency and reliability, we adapted the protocol by making two key simplifications. First, we removed the requirement for the evaluator to perform exact span identification. The original protocol required the evaluator model to pinpoint the exact textual spans or sentences within the reasoning trace where the cognitive behavior manifested. We relaxed this constraint, instructing the model to provide a single, holistic judgment for the entire trace. Second, we eliminated the original 0–2 continuous grading scale (where 0 indicated absent, 1 indicated partially present, and 2 indicated fully present) in favor of a strictly binary system (0 for absent, 1 for present).

We generate annotations with greedy decoding (T=0, max_tokens=2,048, seed=42). To mitigate benchmarksize imbalance within each validation domain, we downsample larger benchmarks to match the smallest benchmark in that domain. In addition to the Qwen3 results shown in Figure 11 of the main paper, we report the corresponding Qwen2.5 results in Figure D3.

All Candidates

FineVision Vero Dataset Curation

| |
|---|

| |
|---|

Captioning & IF 1,528,126 ± 109,932 Grnd., Cnt., & Search 1,495,429 ± 71,409

All Candidates

FineVision Vero Dataset Curation

Chart & OCR 1,182,367 ± 27,398 Knowledge & Recog. 913,072 ± 46,330

| |
|---|

| |
|---|

Captioning & IF 1,528,126 ± 109,932 Grnd., Cnt., & Search 1,495,429 ± 71,409

-0.2

STEM

+2.9

+4.6

STEM 811,990 ± 46,615 Spatial & Action 516,976 ± 9,391

Chart & OCR 1,182,367 ± 27,398 Knowledge & Recog. 913,072 ± 46,330

-0.2

+2.5

STEM

+2.9

Chart & OCR

+1.9

+4.6

STEM 811,990 ± 46,615 Spatial & Action 516,976 ± 9,391

0 2.0M Average Image Area (pixels)

+3.4

+2.5

Chart & OCR

+1.9

0 1 2 3 4 5

0 2.0M Average Image Area (pixels)

+3.4

Average benchmark gain

0 1 2 3 4 5

Figure D1. Ablation on dataset filtering. Each model is trained for 1 epoch with GSPO and a math verify (Kydlícˇek, 2025) reward on 100k examples from a single task category. We report the average benchmark gain over the base model (Qwen2.5-VL-7B-Instruct) for each category.

Figure D2. Average image area (pixels) by task category. The horizontal bars represent the mean image resolution for datasets within each specific domain. Error bars on each bar denote the standard error of the mean for that category.

Average benchmark gain

Training Training

Grnd.,Cnt.&SearchKnowledge&Recog.

Grnd.,Cnt.&SearchKnowledge&Recog.

Action

Action

Captioning&IFChart&OCR

Captioning&IFChart&OCR

0.01 0.24 0.47 0.70 0.93

Spatial&

Spatial&

All

All

STEMMix

STEMMix

Logical Coherence Selective Attention

Context Alignment

Network Organization Verification Abstraction

Sequential Organization Forward Chaining Compositionality Spatial Organization Knowledge Struct. Align. Decomp. & Integration Visual Ref. / Grounding Goal Management

Strategy Selection Causal Organization Mental Imagery Sim.

Context Awareness Self-Awareness Repr. Restructuring

Conceptual-Level Proc. Temporal Organization Perception-then-Reas.

Adaptive Detail Mgmt. Syst. Regional Synth. Arithmetic Calculation

Hierarchical Org. Ordinal Organization Pattern Recognition Visual Foraging

Self-Evaluation Productivity Backward Chaining Backtracking

Figure D3. High-level cognitive behavior presence rates across task categories for single-task and mixed-task trained Qwen2.5 models. Presence rate is indexed by color intensity, where green represents high cognitive presence and red represents low presence, mapping the relationship between training data and emergent reasoning capabilities. The plot reveals that a model’s emergent cognitive behaviors are highly dependent on its specific training domain.

###### D.3 Cognitive Behaviors

We include the cognitive behavioral presence rate of Qwen2.5-VL-7B-Instruct in Figure D3. The results further confirm our findings in Section 7.1 in the main paper, as the captioning-trained model consistently uses more mental imagery simulation (0.33 vs. 0.19 cross-domain average in Qwen2.5), and chart-trained models again demonstrate elevated systematic regional synthesis (0.24 vs. 0.16). Additionally, we found that compared to Qwen3 (Figure 11 in the main paper), several behaviors are much weaker in Qwen2.5: backtracking stays near zero (∼0.01 vs. 0.12–0.48 in Qwen3), self-evaluation ranges from 0.10–0.19 (vs. 0.46–0.94), and strategy selection from 0.15–0.34 (vs. 0.57–0.80). Both models share high logical coherence (∼0.91 vs. ∼0.98) and selective attention (∼0.87 vs. ∼0.98), but Qwen2.5 covers a narrower range overall, indicating that the stronger base model develops a wider set of reasoning behaviors through RL training.

###### D.4 Behavioral Skill Extraction

Building on the concept of meta-cognitive reuse (Didolkar et al., 2025), we formalize a comprehensive pipeline to discover, consolidate, and quantify fundamental reasoning skills directly from raw model traces. Our objective is to transition from analyzing task-specific execution steps (e.g., "counting three red cars") to cataloging domain-agnostic cognitive strategies (e.g., "systematic spatial enumeration"). To achieve this, we design a three-step pipeline consisting of extraction, deduplication, and annotation.

Extraction. In Stage 1, the model ingests a single multimodal reasoning trace to identify reusable, generalizable strategies. We constrain the model’s output so that every provisional name distinctly encodes the action, target, and goal (e.g., behavior_relative_camera_distance_comparison), proactively prohibiting problem-specific entities or narrow qualifiers. In Stage 2, we prevent lexical explosion by autoregressively maintaining a global behavior codebook. Qwen3 evaluates each new candidate against the existing codebook, mapping it into one of four rigid relationships: an exact equivalent (mapped to an existing skill), a subtype (discarded), a more general replacement (overwrites a narrower entry), or a distinct new skill (appended). This trace-by-trace reconciliation acts as a real-time semantic filter, continuously compressing the codebook and reducing redundant phrasing.

Deduplication. Because traces are processed independently, semantically identical behaviors often emerge under varying names. To finalize the codebook, we embed the concatenated “name: description” of every extracted behavior using OpenAI’s text-embedding-3-small and group them via agglomerative hierarchical clustering. To filter out non-reusable skills, we discard any behavior cluster appearing fewer than 10 times. For surviving clusters, GPT-4o synthesizes a single canonical name and a comprehensive description. Finally, human annotators manually inspect these definitions against a sample of source traces to verify that each cluster’s semantic boundary is sensible and accurately reflects the underlying cognitive actions.

Annotation. To quantify capability prevalence while preventing domain imbalance, we uniformly subsample benchmarks to match the smallest dataset within each domain before re-annotating. We frame this behaviormapping as a multi-label classification task. For each instance, Qwen3-32B evaluates the reasoning trace holistically against the canonical dictionary, returning a structured JSON object. For every behavior, the model provides a brief justification detailing its manifestation followed by a strictly binary presence score (1 for present, 0 for absent). We explicitly chose binary scoring over a continuous scale to minimize calibration variance and eliminate the subjectivity of scoring "partial" manifestations, yielding a robust, dense matrix of capability profiles across all models.

###### D.5 Skill Experiments

###### Skill behavior presence rate.

Figure D4 shows the presence rate per model and task category, highlighting the 20 skills with the largest variance across training conditions in each category. As shown, different training task categories can produce substantial differences in skill presence rates, even when models are evaluated on the same task category. This variance indicates that the underlying cognitive strategies a model employs are deeply coupled with its specific training distribution. Rather than developing a universal reasoning pathway, models adapt their problem-solving approaches to the distinct domains they were trained on.

Furthermore, this reveals why multi-task RL training is notoriously difficult: each task demands a fundamentally distinct cognitive profile. For example, optimizing a model to excel at captioning requires reinforcing generative skills like "Define Narrative Structure", which differ drastically from the strategies required for chart and OCR tasks, such as "Axis Analysis". Consequently, we conclude that the cross-domain transfer of low-level reasoning skills is not guaranteed. A highly diverse, mixed-domain training curriculum is strictly necessary to equip a multimodal model with a balanced repertoire of cognitive skills and prevent reasoning blind spots during generalization.

Logistic regression probe. As shown in the main paper, skills extracted per task category are largely linearly separable, which further supports the claim that distinct task domains necessitate fundamentally different cognitive profiles. Here we provide the details of constructing the probe. We select the annotated skills from the within-task reasoning traces, where the model is trained and evaluated on the same category domain, and embed the canonical behaviors using Qwen3-Embedding-8B. We subsample the reasoning traces to 800 per domain to maintain a balanced distribution. We then train a pipeline consisting of (i) per-fold mean centering via StandardScaler(with_std=False), (ii) ℓ2 normalization, and (iii) multinomial logistic regression with a maximum of 2,000 iterations. The pipeline is evaluated using 5-fold Stratified Group K-Fold cross-validation.

###### Qwen2.5-VL-7B-Instruct Trained on Vero Task Subset

###### Captioning & IF

###### Chart & OCR

###### Grnd., Cnt. & Search

GrndCn&Search

GrndCn&Search

GrndCn&Search

Know&Recog

Know&Recog

Know&Recog

Spaa&Acon

Spaa&Acon

Spaa&Acon

Char&OCR

Char&OCR

Char&OCR

Cap&F

Cap&F

Cap&F

STEM

STEM

STEM

MxA

MxA

MxA

Adjective Selection For Visual Description

Prioritize Explicit Information Over Inference Targeted Information Search Extract Numerical Information From Visuals Align Output With Instruction Format

Concluding From Evidence Synthesize Visual And Contextual Information

Contextual Inference Focus On Key Attributes

Final Confirmation

Tone Consistency Identify Central Visual Element

Visual Analysis Contextual Inference Assess Visual Clarity

Visual Data Interpretation Prioritize Explicit Labels Over Visual Cues

Adopt Perspective Analyze Visual Composition

Visual Label Association Extract Relevant Attribute From Data

Analyze Structural Elements Structural Analysis

Implied Meaning Construct Narrative

Chart Legend Interpretation Extract Textual Data Visual Comparison

Prioritize Clear Evidence Estimate Position Visual Estimation

Define Narrative Structure Balance Clarity And Impact Use Imagery And Metaphor

Axis Analysis Option Selection Based On Criteria

Select Option

Synthesize Information Extract Key Data Points

Associative Mapping Assess Visual Indicators

Visual Feature Identification Consider Data Granularity Aggregate Handling Validate Task Goal

Analyze Atmosphere Color Analysis Infer Functional Purpose Analyze Contrasts

Match To Options Elimination

Output In Required Format Extract Contextual Information Assume Standard Dimensions

Final Verification Locate Relevant Section By Heading

Identify Core Concept Describe Shape And Texture

Mapping

Determine Pixel Coordinates

###### Knowledge & Recog.

###### Spatial & Action

###### STEM

GrndCn&Search

GrndCn&Search

GrndCn&Search

Know&Recog

Know&Recog

Know&Recog

Spaa&Acon

Spaa&Acon

Spaa&Acon

Char&OCR

Char&OCR

Char&OCR

Cap&F

Cap&F

Cap&F

STEM

STEM

STEM

MxA

MxA

MxA

Visual Analysis

Spatial Relationship Analysis Focus On Salient Features Map Observations To Answer Choices Apply Domain Knowledge

Justify Answer With Evidence Avoid Unnecessary Complexity Prioritize Clarity And Relevance

Analyze Compositional Elements Prioritize Evidence Based Decision

Analyze Visual Layout Synthesize Information

Contextual Inference Infer Meaning From Context

Evidence Based Conclusion Question Scope Management

Analyze Image Structures Make Informed Assumption

Consistency Verification Synthesize Multiple Clues Constraint Handling

Contextual Inference Translate Spatial Relationship To Directional Language

Infer Context From Visual And Contextual Clues

Balance Precision With Generality Associate Visual Elements With Meaning

Consider Reference Perspective Analysis Label Handling Visual Feature Identification Classify Elements By Type

Select Best Fit Option Select Valid Solution Based On Constraints

Conciseness Optimization Use Domain Knowledge Compare Elements

Option Alignment Evaluation Question Intent Analysis

Visual Verification Process Of Elimination

Assess Information Availability Conclude Based On Insufficient Data

Plausibility Evaluation Foreground Background Analysis

Diagram Analysis Prioritize Prominent Visual Element

Assess Relevance Systematic Visual Scanning

Cross Validation Consistency Verification

Infer Structural Relationships

Pattern Recognition Handle Missing Information

Identify Categories Enumerate Elements

Filter Irrelevant Information Exhaustive Option Verification

Interpret Lighting And Atmosphere

Eliminate Invalid Options

Distractor Detection

###### Qwen3-VL-8B-Instruct Trained on Vero Task Subset

###### Captioning & IF

###### Chart & OCR

###### Grnd., Cnt. & Search

GrndCn&Search

GrndCn&Search

GrndCn&Search

Know&Recog

Know&Recog

Know&Recog

Spaa&Acon

Spaa&Acon

Spaa&Acon

Char&OCR

Char&OCR

Char&OCR

Cap&F

Cap&F

Cap&F

STEM

STEM

STEM

MxA

MxA

MxA

Focus On Key Attributes Tone Consistency

Align Output With Instruction Format

Final Confirmation

Final Verification Validate Task Goal

Visual Analysis Contextual Inference Assess Visual Clarity

Contextual Inference Adjective Selection For Visual Description

Consider Data Granularity

Identify Central Visual Element Synthesize Information Adopt Perspective Balance Clarity And Impact Implied Meaning

Label Verification Exhaustive Verification

Prioritize Clear Evidence Assume Standard Conventions

Visual Feature Identification

Visual Verification

Mapping Scope Constraint

Visual Scanning Visual Estimation

Define Narrative Structure Analyze Visual Composition

Assess Data Relevance And Availability

Elimination Visual Segmentation Spatial Analysis Reasoned Assumption

Value Matching Apply Domain Knowledge

Filter Irrelevant Information Analyze Contrasts Link Elements To Theme

Consider Alternative Interpretations Validate Data Type Consistency

Select Option Assess Visual Indicators

Infer Implicit Meaning Constraint Verification

Criteria Reinterpretation Eliminate Irrelevant Information

Check For Duplicates Iterative Refinement

Validate Consistency And Completeness Constrain Output Length Infer Design Intent Elimination

Consistency Assessment Infer From Context Cross Reference Cross Validation

Cross Reference Extract Contextual Information

Assess For Trick Question

Knowledge & Recog.

###### Spatial & Action

###### STEM

GrndCn&Search

GrndCn&Search

GrndCn&Search

Know&Recog

Know&Recog

Know&Recog

Spaa&Acon

Spaa&Acon

Spaa&Acon

Char&OCR

Char&OCR

Char&OCR

Cap&F

Cap&F

Cap&F

STEM

STEM

STEM

MxA

MxA

MxA

Visual Verification Infer Context From Visual And Contextual Clues

Map Observations To Answer Choices Question Scope Management Cross Validation Plausibility Evaluation Visual Feature Identification Consider Reference Label Handling

Contextual Inference

Synthesize Multiple Clues Infer Meaning From Context

Use Domain Knowledge

Compare Elements Make Informed Assumption

Select Valid Solution Based On Constraints Constraint Handling Question Intent Analysis

Elimination Systematic Visual Scanning

Select Best Fit Option Option Alignment Evaluation

Hypothesis Testing Pattern Recognition Contextual Analysis

Classify Elements By Type Evaluate Candidate Positions

Diagram Analysis Consider Alternative Solutions

Consistency Verification Eliminate Invalid Options

Visual Comparison Iterative Refinement Iterative Verification

Cross Validation Iterative Hypothesis Testing

Reexamine And Verify Consider Alternative Hypotheses Systematic Visual Scanning

Process Of Elimination Filter Irrelevant Information

Assess Relevance Identify Visual Focus

Identify Categories Iterative Hypothesis Testing

Infer Structural Relationships

Identify Structural Patterns Infer From Common Conventions Causal Reasoning

Visual Verification Exhaustive Option Verification

Interpret Ambiguity Rule Application

Distractor Detection Manage Uncertainty And Assumptions

Disambiguation Interpret Question Intent

Mental Simulation Object Identity Differentiation

Assume Missing Data

0.00 0.25 0.50 0.75 1.00

Figure D4. Behavioral skill analysis presence rates for models trained on individual task categories and one trained on all categories (mix all). Training on individual task categories impacts the emergence of fine-grained skills within each domain, with green indicating higher skill presence and red indicating lower. For instance, chart skills center on reading operations like "Axis Analysis", while spatial skills center on physical state reasoning like "Mental Simulation".

###### Full Catalog of Behavior-Description Pairs

Table D1 summarizes the high-level cognitive behavior codebook used in the behavioral analyses. Table D2 then lists the finer-grained skill definitions that support the behavioral skill analysis and probe construction.

Behavior Definition Abstraction Extract general principles from specific instances. Adaptive Detail Management

Behavior Definition Network Organization Arrange concepts as interconnected nodes with

Adjust the level of detail based on reasoning requirements.

multiple pathways and relationship types. Ordinal Organization Arrange elements according to relative rank or position. Pattern Recognition Recognize recurring structures across different contexts.

Arithmetic Calculation Extract, manipulate, and compute numerical values to reach a verifiable solution within a reasoning trace.

Backtracking Identify unproductive paths and return to earlier decision points. Backward Chaining Start with goals and work backward to identify

Perception Then Reasoning

Separate cognitive labor into two distinct stages: exhaustive information extraction and abstract logical operations.

prerequisites. Causal Organization Arrange elements through cause-effect

Productivity Generate novel combinations using a finite set of elements.

relationships. Compositionality Build complex ideas from simpler components. Conceptual Level Processing

Representational Restructuring

Reformulate problems to reveal new insights. Selective Attention Focus on relevant information while filtering out distractions. Self Awareness Assess one’s own knowledge state, capabilities, and task solvability.

Reason with abstract concepts before translating to linguistic forms.

Context Alignment Select appropriate organizational patterns based on context.

Context Awareness Recognize how the situational context shapes which reasoning strategies and goals are appropriate.

Self Evaluation Assess the quality, correctness, efficiency, and progress of one’s reasoning and make adjustments as needed.

Decomposition And Integration

Break problems into subparts and synthesize solutions.

Sequential Organization

Arrange steps in linear order where sequence matters.

Forward Chaining Start with initial conditions and work toward goals. Goal Management Establish, maintain, and adjust goals throughout the reasoning process.

Spatial Organization Arrange elements according to spatial relationships and configurations. Strategy Selection Choose the most appropriate reasoning approaches based on task requirements and domain.

Hierarchical Organization

Arrange concepts in nested, tree-like structures with parent-child relationships.

Systematic Regional Synthesis

Iteratively traverse multiple elements or regions of an image in a deliberate sequence, and synthesize information across them.

Knowledge Structure Alignment

Match reasoning organization to domain knowledge structure.

Temporal Organization Arrange elements along a timeline with

Logical Coherence Maintain consistency in reasoning across steps and contexts.

before/after relationships. Verification Check reasoning steps against established criteria. Visual Foraging Strategically manage the acquisition of multiple

Mental Imagery Simulation

Generate internal representations that preserve the properties of a stimulus in the absence of direct sensory input.

targets or information points within a complex environment.

Visual Reference Or Grounding

Establish a persistent and precise link between linguistic symbols and localized visual elements.

Table D1. Summary of high-level behavior definitions. A glossary detailing the high-level cognitive behaviors.

##### E Reasoning Traces

We provide extensive reasoning trace examples from Vero across all six task categories on our project website: https://vero-reasoning.github.io. Specifically, we showcase three representative examples per category to concretely illustrate how the model adapts its cognitive approach to different domains. These traces exhibit structured chain-of-thought reasoning with dynamic metacognitive behaviors, such as self-verification and backtracking, alongside precisely grounded visual perception.

Beyond merely highlighting these task-specific strategies, the qualitative examples validate the overarching efficacy of our open multi-task RL recipe. We demonstrate how training on a highly diverse, mixeddomain curriculum equips the model with a versatile toolkit for general-purpose visual reasoning. Whether navigating complex spatial constraints or performing abstract STEM deduction, our examples confirm that Vero seamlessly bridges raw visual inputs with long-horizon logical planning, fundamentally enhancing its capacity to handle real-world multimodal challenges.

Skill Definition Captioning & Instruction Following Adjective Selection For Visual Description

Skill Definition Grounding, Counting & Search Analyze Structural Elements Examines object structure to determine composition. Assess For Trick Question Recognize riddles or tricks via phrasing/data inconsistencies. Assess Visual Clarity Recognize limitations of visual evidence and detail. Assess Visual Indicators Draw conclusions based on the absence of specific indicators. Associative Mapping Associate text/marks with functional roles like brand names. Assume Standard Conventions

Choosing descriptive adjectives that reflect style and function to capture a scene.

Adopt Perspective Use vivid language to convey a specific perspective without using

pronouns. Analyze Atmosphere Establish emotional connection to a setting via environmental interaction. Analyze Contrasts Use visual contrast (color, structure) to highlight key information. Analyze Visual Composition Assess visual elements like focal points to determine mood or message. Balance Clarity And Impact Blend concrete imagery with abstract language for vivid descriptions. Constrain Output Length Limit the response to a specific word count while remaining coherent. Constraint Verification Systematically verifying that all specified constraints are met. Construct Narrative Construct a surreal narrative using imaginative and metaphorical

Assume standard response forms when multiple interpretations exist.

Assume Standard Dimensions Account for device scaling by estimating relative coordinates. Check For Duplicates Verify that an element is the only one matching a requirement. Color Analysis Analyze color properties to determine object relationships. Concluding From Evidence Draw final conclusions based on cumulative observed evidence. Cross Reference Cross-reference command language with UI labels to identify elements. Determine Pixel Coordinates Estimate coordinates based on spreadsheet layout and row headers. Estimate Position Estimate positions relative to surrounding components. Extract Contextual Information

language. Define Narrative Structure Organize narrative by subject, then setting, then secondary subjects. Describe Shape And Texture Incorporate descriptions of shape and texture to convey visual traits. Extract Key Data Points Extract specific data points (city names, deals) from visual labels. Focus On Key Attributes Prioritize features impacting user experience like comfort and aesthetics. Identify Central Visual Element

Identify and extract contextual relationship between text elements.

Final Confirmation Confirm interpretation by eliminating contradictions with requirements. Match To Options Select options most consistent with evidence despite limitations. Output In Required Format Translate element locations into structured output like JSON. Prioritize Clear Evidence Exclude elements that cannot be reliably determined. Reasoned Assumption Make assumptions to resolve visual ambiguities. Select Option Select the best-matching object even when multiple candidates exist. Spatial Analysis Determine roles in a scene based on action and environment. Synthesize Visual And Contextual Information

Choose representative visual elements to focus on in creative output.

Identify Core Concept Extract core technical concepts or mechanisms from the text. Implied Meaning Convey the presence of an element indirectly by referencing related

phenomena. Infer Design Intent Interpret technical styles like blueprints to infer purpose or context. Infer Functional Purpose Deduce system purpose by analyzing labels, actors, and interactions. Infer Implicit Meaning Identify implicit assumptions embedded in diagrams or explanations. Link Elements To Theme Select descriptive words that align with specific visual features. Tone Consistency Ensure language aligns with the formal or thematic nature of the subject. Use Imagery And Metaphor Combine visual analysis with context for symbolic interpretation. Validate Consistency And Completeness

Combine observations into a coherent, supported explanation.

Visual Estimation Include partially visible elements in counts if identifiable. Visual Scanning Divide images into quadrants to check for target objects. Visual Segmentation Divide fields into spatial regions to identify/count elements.

Cross-reference visual elements with textual data to confirm entities.

STEM Apply Domain Knowledge Apply field-specific knowledge when context is missing. Assume Missing Data Make reasonable assumptions for ambiguous parameters in calculations. Avoid Unnecessary Complexity

Chart & OCR Aggregate Handling Differentiate between group data and specific subgroup data. Align Output With Instruction Format

Adopt symmetric/extreme arrangements to simplify reasoning.

Deduce if numerical or categorical answers are expected.

Consider Alternative Solutions Evaluate various theorems to determine relevance to a problem. Consistency Verification Verify derived values against original problem constraints. Constraint Handling Evaluate if options adhere to structural/quantitative constraints. Contextual Inference Infer answers by evaluating outcomes of element comparisons. Cross Validation Identify errors by recomputing values and comparing results. Diagram Analysis Reevaluate geometric relationships to ensure correct properties. Distractor Detection Identify and disregard elements not required by constraints. Exhaustive Option Verification Verify if values result in duplicate elements in a set. Filter Irrelevant Information Disregard unnecessary parameters to focus on essentials. Infer Meaning From Context Determine task nature by analyzing materials and instructions. Infer Structural Relationships Deduce logical groupings by analyzing patterns and connections. Iterative Hypothesis Testing Test alternative hypotheses about overlapping elements. Justify Answer With Evidence Confirm answers by checking consistency across reasoning lines. Manage Uncertainty And Assumptions

Assess Data Relevance And Availability

Recognize when data lacks context to answer a question directly.

Axis Analysis Analyze chart axes to determine variables and their relationships. Chart Legend Interpretation Analyze legends to determine how visual elements encode info. Consider Alternative Interpretations

Re-evaluate requirements when there is a data mismatch.

Consider Data Granularity Weigh label granularity against the need for precision. Consistency Assessment Verify if visual attributes are applied consistently across regions. Criteria Reinterpretation Restate problems in actionable terms (e.g., "least change" as "flattest"). Eliminate Irrelevant Information

Exclude entries that do not correspond to valid target categories.

Exhaustive Verification Cross-validate answers with the explicit intent of the question. Extract Numerical Information From Visuals

Avoid definitive claims when evidence is ambiguous.

Estimate or read values directly from labels on charts/graphs.

Option Alignment Evaluation Assess if choices relate directly to the main subject. Prioritize Clarity And Relevance

Extract Relevant Attribute From Data

Identify connections matching specific attributes like color or type.

Weigh options supported by explicit actions in the image.

Extract Textual Data Extract info from text sources without additional calculation. Final Verification Re-examine prompts to ensure solutions align with requirements. Infer From Context Use context to infer expected trends (e.g., survival phases vs scarcity). Locate Relevant Section By Heading

Prioritize Prominent Visual Element

Focus on primary focal points rather than background elements.

Navigate visual resources using identifiers to focus attention.

Process Of Elimination Eliminate choices whose definitions do not match context. Question Intent Analysis Clarify intent to match reasoning strategy to the question. Select Best Fit Option Choose the choice numerically closest to a calculated value. Select Valid Solution Based On Constraints

Mapping Map visual arrangements to categories and time periods. Option Selection Based On Criteria

Eliminate solutions that do not make sense (e.g. negative angles).

Apply distinguishing steps to select a single candidate from many.

Spatial Relationship Analysis Infer spatial relationships using common layout conventions. Structural Analysis Recognize functional groups via atom bonding arrangements. Synthesize Multiple Clues Classify objects by synthesizing multiple distinctive features. Visual Feature Identification Associate visual features with traits of known categories. Visual Verification Integrate visual and text info to resolve ambiguities.

Differentiate between explicit data and assumed background knowledge.

Prioritize Explicit Information Over Inference

Prioritize Explicit Labels Over Visual Cues

Favor directly labeled terms over interchangeable associated terms.

Scope Constraint Clarify inclusion/exclusion based on roles (reference vs data points). Targeted Information Search Restate requirements to maintain focus on relevant data subsets. Validate Data Type Consistency

Spatial & Action Classify Elements By Type Determine action eligibility based on position or status. Consider Alternative Hypotheses

Match interpretation to data type (e.g., using % if only % available).

Evaluate alternative interpretations when outcomes do not match.

Validate Task Goal Cross-check annotations against task requirements for correctness. Value Matching Compare values to a target to assess proximity or criteria match. Visual Data Interpretation Infer magnitude by comparing spatial positions on a graph. Visual Label Association Check if labels are directly associated or provided via a legend.

Consider Reference Focus on specific reference objects for correct comparisons. Eliminate Invalid Options Narrow locations by identifying occupied rows/columns. Enumerate Elements List all elements in a specific position within an arrangement. Evaluate Candidate Positions Select positions fulfilling functional requirements of patterns. Evidence Based Conclusion Acknowledge when info is insufficient for spatial determination. Focus On Salient Features Identify prominent elements to determine importance. Foreground Background Analysis

Knowledge & Recognition Analyze Compositional Elements

Break down visual components to understand scene and context.

Apply layering principles to determine relative positions.

Analyze Image Structures Evaluate structures to determine orientation and standard features. Analyze Visual Layout Observe spatial relationships to understand object connections. Assess Information Availability

Handle Missing Information Deduce region assignments based on elimination/requirements. Identify Categories Identify element categories for further classification. Interpret Ambiguity Clarify ambiguous terms using common conventions. Label Handling Adopt provided labels even if personal inference differs. Map Observations To Answer Choices

Indicate when external knowledge is required for a definitive answer.

Assess Relevance Identify environmental elements relevant to context or purpose. Associate Visual Elements With Meaning

Link objects to real-world usage to infer setting themes.

Translate structural observations into required solution formats.

Balance Precision With Generality

Provide accurate answers that account for regional variability.

Mental Simulation Simulate moves to assess if they achieve objectives. Object Identity Differentiation Use comparative statements to assess opposite relationships. Perspective Analysis Express relationships using different reference frames. Plausibility Evaluation Evaluate scenario plausibility based on environment constraints. Question Scope Management Align question scope with relevant visual entities. Reexamine And Verify Backtrack and re-execute simulations upon detecting errors. Rule Application Compare outcomes against constraints to select approaches. Systematic Visual Scanning Scan fields in order (clockwise/rows) to prevent double-counting. Translate Spatial Relationship To Directional Language

Causal Reasoning Link symptoms to underlying causes based on known patterns. Compare Elements Compare characteristics to determine similarities and differences. Conciseness Optimization Include relevant info without exceeding word limits. Conclude Based On Insufficient Data

Exclude conclusions not supported by available evidence.

Contextual Analysis Deducing a person’s role from actions and environment. Disambiguation Clarify meanings of terms used in misleading or unclear ways. Elimination Rejecting interpretations that do not align with evidence. Hypothesis Testing Test subject identity by evaluating traits against candidates. Identify Structural Patterns Analyze shape and posture to distinguish object categories. Identify Visual Focus Analyze element positions to determine prominence or relevance. Infer Context From Visual And Contextual Clues

Interpret terms like "above" based on visual composition.

Analyze elements like street signs to understand scenes.

Infer From Common Conventions

Use typical structures (e.g., posters) to infer missing info.

Interpret Lighting And Atmosphere

Analyze light and environmental cues to infer time of day.

Interpret Question Intent Adjust understanding based on phrasing to align with intent. Iterative Refinement Propose and revise hypotheses based on new observations. Iterative Verification Establish confidence by repeatedly verifying data via checks. Label Verification Match icons and labels to process steps based on position. Make Informed Assumption Use real-world knowledge when direct evidence is absent. Pattern Recognition Identify context by recognizing familiar terminology and structures. Prioritize Evidence Based Decision

Select plausible answers based on cumulative visual analysis.

Synthesize Information Combine visual and text info for a coherent conclusion. Use Domain Knowledge Apply specific expertise to verify structures against theory. Visual Analysis Establish visual criteria to determine if elements meet tasks. Visual Comparison Use cues like bottle height to infer relative volume.

###### Table D2. Summary of skill definitions. A comprehensive glossary providing operational definitions for the fine-grained skills evaluated across all six task categories.

