## arXiv:2603.13224v2[cs.CV]9May2026

[Figure 1]

2026-5-12

# Visual-ERM: Reward Modeling for Visual Equivalence

###### Ziyu Liu*1,2, Shengyuan Ding*2,3, Xinyu Fang2, Xuanlang Dai2, Penghui Yang2, Jianze Liang2, Jiaqi Wang2, Kai Chen2, Dahua Lin2,4 and Yuhang Zang†2

1Shanghai Jiao Tong University, 2Shanghai AI Laboratory, 3Fudan University, 4CUHK

Vision-to-code tasks require models to reconstruct structured visual inputs, such as charts, tables, and SVGs, into executable or structured representations with high visual fidelity. While recent Large Vision Language Models (LVLMs) achieve strong results via supervised fine-tuning, reinforcement learning remains challenging due to misaligned reward signals. Existing rewards either rely on textual rules or coarse visual embedding similarity, both of which fail to capture fine-grained visual discrepancies and are vulnerable to reward hacking. We propose the Visual Equivalence Reward Model (Visual-ERM), a multimodal generative reward model that provides fine-grained, interpretable, and task-agnostic feedback to evaluate vision-to-code quality directly in the rendered visual space. Integrated into Reinforcement Learning (RL), Visual-ERM improves Qwen3-VL-8B-Instruct by +8.4 on chart-to-code and yields consistent gains on table and SVG parsing (+2.7, +4.1 on average), and further strengthens test-time scaling via reflection and revision. We also introduce VisualCritic-RewardBench (VC-RewardBench), a benchmark for judging fine-grained image-to-image discrepancies on structured visual data, where Visual-ERM at 8B decisively outperforms Qwen3-VL-235B-Instruct and approaches leading closedsource models. Our results suggest that fine-grained visual reward supervision is both necessary and sufficient for vision-to-code RL, regardless of task specificity.

##### 1. Introduction

Large Vision Language Models (LVLMs) have made rapid progress in multimodal understanding [3, 12, 25]. Among the resulting capabilities, vision-to-code (Fig. 1 top) stands out as a particularly important primitive: it converts structured visual inputs such as charts, tables, and SVGs into executable code [47] or markdown [24]. Vision-to-code has become a key primitive for downstream uses including UI-to-code generation [33], scientific document parsing, and knowledge management.

Most existing work improves vision-to-code with supervised fine-tuning (SFT) [47, 48], which is dataintensive and often generalizes poorly across tasks. Reinforcement learning (RL) is a natural alternative [18, 37], but its effectiveness depends on the reward signal. A reliable reward for vision-to-code should be visual (judge the rendered output rather than the source code), fine-grained (penalize element-level errors in layout, axes, and content), interpretable (localize what is wrong rather than emit an opaque scalar), and task-agnostic (cover charts, tables, and SVGs within a single model). As shown at the bottom of Fig. 1, current rewards fall short on these axes. Text-based rules such as edit distance and Tree-Edit-Distance Similarity (TEDS) operate on predicted code or markup, so they are not visual: layout, alignment, and other rendering errors are invisible to them. Vision-encoder similarities such as DINO [34] are visual but not fine-grained or interpretable: their encoders are pre-trained for semantic invariance and tolerate small spatial or structural shifts that change the meaning of a chart or table, while collapsing evidence into a single scalar (Sec. 2). Neither family is task-agnostic across chart, table, and SVG within a single model, leaving a gap between the reward and the underlying notion of visual equivalence. We present the detailed analysis in Sec. 2.

We address this gap with the Visual Equivalence Reward Model (Visual-ERM), a multimodal generative reward model that scores vision-to-code outputs by reasoning over the rendered image directly. We train Visual-ERM on a discrepancy-annotated corpus of reference–prediction image pairs assembled from two

† Corresponding authors: Yuhang Zang (zangyuhang@pjlab.org.cn)

* Equal contribution. Code is at https://github.com/InternLM/Visual-ERM

Visual-ERM: Reward Modeling for Visual Equivalence

#### Teaser图（NeurIPS）

###### Vision-to-Code

Input Image Code&Markdown Rendered Image

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

import numpy as np\n... plt.savefig('fig.png',dpi=80)

Chart-toCode

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Chart <table><tr><td>Year</td><td>Region</td><td>...</table>

SVG

Table-toMarkdown

Chart SVG Parse

[Figure 15]

Render

[Figure 16]

[Figure 17]

SVG-toCode

<svg width="512" height="512" viewBox="0 0 512 512">...</svg>

[Figure 18]

Table

Table

###### Reward Signal

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Visual Fine-grained Interpretable Task-agnostic

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Text-Based Reward (e.g., TEDS, edit distance) Vision-Based Reward (e.g., DINO, CLIP) Visual-ERM (ours)

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

- Figure 1: Overview of vision-to-code reward modeling and Visual-ERM. Top: vision-to-code parses a structured visual input (chart, table, or SVG) into code or markup, which is rendered back to an image so that quality can be judged in the visual space. Bottom: we compare reward signals along four properties: visual, fine-grained, interpretable, and task-agnostic. Text-based rewards (e.g., TEDS, edit distance) ignore visual cues; vision-encoder similarities (e.g., DINO [34], CLIP [29]) capture visual content but remain coarse and opaque. Visual-ERM combines all four properties, providing a reliable supervisor across vision-to-code tasks.

complementary sources: targeted edits that inject pre-defined error types and natural inferences from weaker LVLMs, with fine-grained annotations distilled from a stronger proprietary model. At deployment, the same model serves two roles: it converts per-error severities into a bounded scalar that, paired with a render-success term, supplies the reward for GRPO-based RL, and it returns the structured discrepancy list as natural-language feedback for test-time scaling.

We evaluate Visual-ERM along three axes. On RL, we integrate it into vision-to-code pipelines and observe that Visual-ERM boosts Qwen3-VL-8B-Instruct [3] by +8.4 points on chart-to-code and yields consistent gains on table-to-markdown (+2.7) and SVG-to-code (+4.1). To assess the reward signal itself, we introduce VisualCriticRewardBench (VC-RewardBench), a benchmark for fine-grained image-to-image discrepancy judgment on structured visuals; VC-RewardBench exposes consistent weaknesses of current LVLMs, and Visual-ERM at 8B improves over Qwen3-VL-235B-Instruct, narrowing the gap to leading closed-source models. For test-time scaling, Visual-ERM provides interpretable feedback that drives reflection and revision, yielding further gains.

Our contributions are: 1) We analyze existing reward paradigms for vision-to-code and identify four properties a reliable reward must satisfy, namely visual, fine-grained, interpretable, and task-agnostic; we further show that existing rewards each violate at least two of these properties and induce reward hacking under RL. 2) Guided by this analysis, we propose Visual-ERM, a multimodal generative reward model that scores rendered outputs with fine-grained, interpretable, and task-agnostic signals, and we integrate it into both RL and test-time scaling. 3) We release VisualCritic-RewardBench, a benchmark of 1,335 instances for fine-grained image-to-image discrepancy judgment. 4) Visual-ERM improves Qwen3-VL-8B-Instruct by +8.4 on chart-tocode, +2.7 on table-to-markdown, and +4.1 on SVG-to-code, with further gains from reflection-and-revision at inference time.

##### 2. Analysis of Reward Signals for Vision-to-Code

RL is the most promising route to scale vision-to-code beyond imitation, but RL is only as reliable as its reward. We first present the limitations of existing reward families in Fig. 2.

Setup. Let 𝑥 denote the rendered reference image and 𝑦ˆimg the rendered prediction. We define visual equivalence as the property that 𝑦ˆimg matches 𝑥 at every level a downstream user would inspect, including layout, axes, labels, and numeric content. We probe each reward in two complementary ways: qualitative case studies on held-out chart and table examples, and RL training under each reward followed by downstream evaluation.

Text-based rules are blind to rendering. TEDS, edit distance, and related metrics compare predicted code or

分析图（1x3）

Visual-ERM: Reward Modeling for Visual Equivalence

Analysis of Reward Signals

[Figure 36]

Original:<table><t r><td>...</td><td >...</td><td>...</ td></tr></table>

Text-Based

(a) RL Training Curve (b) Chat-to-Code

TEDS Score: 0.92

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Vision-blind Reward Bias

TEDs F1 Attribute matching

0.9

###### PerformancePerformance

[Figure 42]

GT Text

[Figure 43]

Parsed:<table><tr

0.8

>...</td><td>...<t d>...</td></tr></t able>

###### Reward Hacking

Visual-ERM: (1) layout error (2) numeric error

[Figure 44]

RewardScore

[Figure 45]

Edit Distance

[Figure 46]

[Figure 47]

0.7

- 64

68

72

76

80

- 65

Pred Text

visionbased

baseline

ours

0.6

[Figure 48]

[Figure 49]

DINO Score: 0.99

(c) Table-to-Markdown

Vision-Based

[Figure 50]

Faithful Optimization

0.5

[Figure 51]

Coarse-grained Uninterpretable

80

[Figure 52]

75

vison-based text-based ours

GT Image

0.4

DINO

[Figure 53]

Visual Similarity

70

[Figure 54]

Visual-ERM: (1) Colorbar numeric differ (2) Bubbles is positioned higher (3) The Y-axis error

0.3

[Figure 55]

[Figure 56]

[Figure 57]

60

0 20 40 60 80 100 120 140

visionbased

textbased

ours Pred Image

baseline

Training Steps

- Figure 2: Analysis of existing rewards and downstream impact. Left: a text-based reward (TEDS) and a vision-based reward (DINO) assign near-perfect scores to predictions that Visual-ERM identifies as visually wrong. Right: (a) RL training curves where text- and vision-based rewards plateau via reward hacking while Visual-ERM keeps improving; (b, c) downstream gains on Chart-to-Code and Table-to-Markdown.

markup tokens against a reference sequence and never inspect 𝑥 or 𝑦ˆimg. A prediction that is textually close but renders to the wrong figure therefore receives a near-perfect score, and a prediction that renders correctly via different yet equivalent code is penalized. The top-left case in Fig. 2 shows a table-to-markdown output with TEDS= 0.92 whose rendered version exhibits both header-hierarchy and numeric-cell errors, all of which Visual-ERM localizes.

Vision-encoder similarities are tuned for the wrong invariance and are uninterpretable. Vision foundation models such as DINO [6] and CLIP [29] are pre-trained to be invariant to translation, scale, and other transformations that preserve semantic content. This invariance is at odds with vision-to-code, where a small spatial shift or a permuted bar order changes the underlying data, and the resulting scalar also hides which element is wrong. The bottom-left case in Fig. 2 shows a chart with DINO similarity above 0.99 for which Visual-ERM enumerates colorbar, bubble-position, and y-axis errors.

Reward hacking under RL. We optimize a chart-to-code policy against each reward and report the proxy reward score over training in Fig. 2(a). The text-based and vision-based curves rise quickly and saturate, consistent with reward hacking: the policy maximizes the proxy without improving the rendered output. Optimizing against Visual-ERM keeps improving throughout training, suggesting that its supervision remains aligned with rendered-image fidelity. The next paragraph translates these proxy trends into rendered-task performance.

Downstream impact. Fig. 2(b, c) evaluates the resulting policies on Chart-to-Code and Table-to-Markdown. Policies trained with text-based or vision-based rewards do not consistently improve over the SFT baseline, and underperform it on Table-to-Markdown, indicating that the proxy gains in Fig. 2(a) are illusory. Optimizing against Visual-ERM yields the largest gains in our comparison on both tasks.

Summary. Together, the failure cases and the RL probes point to four properties a vision-to-code reward must satisfy: (i) be visual, operating in the rendered image space, since text-only rewards are blind to rendering; (ii) be fine-grained and locally sensitive, since coarse global similarity hides element-level errors; (iii) be interpretable, since a single scalar invites reward hacking and gives the policy no signal about what to fix; and (iv) be task-agnostic, since the same failures recur across charts, tables, and SVGs. Sec. 3 instantiates these four properties in Visual-ERM.

##### 3. Methods

The analysis in Sec. 2 defines four properties for a vision-to-code reward: it must operate in the rendered image space, be fine-grained and locally sensitive, expose interpretable evidence, and remain task-agnostic. Visual-ERM instantiates these properties through three coupled components, each visualized in Fig. 3: (i) a discrepancy-annotated dataset 𝒟reward and a single LVLM trained on it; (ii) a deployment-time pipeline that

Framework 图（NeurIPSVisual-ERM:） RewardModelingforVisualEquivalence

###### Visual-ERM Framework

- （a）
- （b）

Visual Perception & Grounding Reasoning & Rewarding

###### Model Training

[Figure 58]

"structure_error_count": 1, "data_error_count": 1, "text_error_count": 1, "style_error_count": 1, "errors": [ { "category": "text_error", "severity": 2, "location": "Z-axis label", "description": "Z-axis label text differs (Reference: 'Values' while Generated: 'Count'), which changes the displayed measure wording." }, ......]

[Figure 59]

[Figure 60]

Original

[Figure 61]

Finetune

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

Render

Base

{“category”, “severity”, “location”, “description”}

Visual-ERM

Training Data (Image Pair & Discrepancy annotation)

[Figure 68]

[Figure 69]

[Figure 70]

RL Pipeline

Reference Model 1

[Figure 71]

1

[Figure 72]

Bar Chart Shape(-3)

Reinforce ment Learning

[Figure 73]

[Figure 74]

Body mismatch(-2)

Bar Chart Color(-2) Z-axis Label(-1)

[Figure 75]

[Figure 76]

Multi-TaskJudgement

[Figure 77]

Stars replaced by square blocks (-3)

[Figure 78]

[Figure 79]

· · ·

Rewards

· · · Visual-

[Figure 80]

[Figure 81]

Visual-ERM provides visual-consistency and cross-modal evaluation. Visual-ERM provides fine-grained feedback.

[Figure 82]

[Figure 83]

ERM

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

Test-Time Scaline

VisualERM

[Figure 89]

[Figure 90]

Feedback

TestTime Scaling

Header hierarchy incorrect (-3) Numeric values error (-1) Symbol error (-1)

[Figure 91]

[Figure 92]

[Figure 93]

00

Infer

[Figure 94]

Refine VisualERM

Feedback

[Figure 95]

[Figure 96]

[Figure 97]

Visual-ERM’s reward outputs are highly interpretable.

[Figure 98]

- Figure 3: (a) Training data and model training. From paired ground-truth and rendered images, we obtain finegrained discrepancy annotations (category, severity, location, description) and fine-tune a base LVLM into Visual-ERM. (b) Deployment. Visual-ERM produces unified judgments across chart, SVG, and table tasks, yielding scalar rewards for RL and natural-language feedback for test-time scaling.

converts Visual-ERM outputs into a scalar reward for RL; and (iii) the same outputs reused as natural-language feedback for test-time scaling. We also release VC-RewardBench, a diagnostic benchmark for fine-grained image-to-image discrepancy judgment.

Notation. We retain the symbols introduced in Sec. 2. Let 𝑚∈{Chart, Table, SVG} index the task and ℛ𝑚(·) its renderer. A vision-to-code policy 𝜋𝜃(𝑦 | 𝑥) takes the reference image 𝑥 as input and emits structured text 𝑦, which is rendered to 𝑦ˆimg = ℛ𝑚(𝑦). We write 𝑓𝜃

for Visual-ERM; 𝑦⋆ denotes ground-truth code or markup.

ERM

###### 3.1. Visual-ERM: Data, Annotation, and Training

The training pipeline shown in Fig. 3(a) consists of three stages: image-pair construction, fine-grained discrepancy annotation, and supervised fine-tuning.

Image-pair construction. We curate pairs (𝑥,𝑦ˆimg) that span the discrepancies a downstream policy is likely to produce, using two complementary sources. (1) Targeted Edit: high-capacity LVLMs perturb 𝑦⋆ to inject a pre-defined taxonomy of error types, providing systematic coverage of structural, data, textual, and stylistic failures. (2) Natural Inference: weaker LVLMs predict 𝑦 directly, so the resulting errors match the distribution that an RL policy actually emits during rollout. Each candidate 𝑦 is rendered with ℛ𝑚 to obtain 𝑦ˆimg and paired with the reference image 𝑥.

Fine-grained discrepancy annotation. For every pair we collect a structured annotation 𝑎 = {(category,severity,

location,description)𝑘}𝐾𝑘=1 that localizes each discrepancy and grades its impact. A natural concern is whether large open-source LVLMs already produce such annotations directly; we find they do not. Even Qwen3-VL-

235B-Instruct [3] misses subtle structural and textual deviations, as quantified in Sec. 4.2. We therefore use a stronger model, GPT-5-mini [35], as a bootstrap teacher that proposes candidate discrepancies, which are then filtered by rendering-consistency checks (proposals referring to regions absent in 𝑦ˆimg are discarded) and by agreement with the ground-truth code 𝑦⋆ at the structural level. The resulting reward dataset is 𝒟reward = {(𝑚, 𝑥, 𝑦ˆimg, 𝑎)}. We stress that GPT-5-mini is used only to label the training set 𝒟reward and plays no role at evaluation time (Sec. 3.4). The teacher therefore acts as a scalable labeler for training only, not as both annotator and judge; any residual annotator bias in 𝒟reward is bounded by Visual-ERM’s downstream RL gains (Sec. 4.1), which are measured on benchmarks independent of the training labels.

Training objective. Visual-ERM is a conditional generator 𝑓𝜃

(𝑎 | 𝑥,𝑦ˆimg) that produces the structured discrepancy list and per-error severities used in Sec. 3.2. We supervise it by negative log-likelihood:

ERM

)︀]︁. (1)

###### [︁ − log 𝑓𝜃

(︀

###### ℒ(𝜃ERM) = E(𝑚,𝑥,𝑦^

𝑎 | 𝑥, 𝑦ˆimg

img,𝑎)∼𝒟reward

ERM

For an annotation sequence 𝑎 = (𝑎1,...,𝑎𝑇), this expands to the token-level form

img,𝑎)[︁ −

)︀]︁, (2)

∑︁𝑇

(︀

ℒ(𝜃ERM) = E(𝑚,𝑥,𝑦^

𝑎𝑡 | 𝑥, 𝑦ˆimg, 𝑎<𝑡

log 𝑓𝜃

ERM

𝑡=1

where 𝑎<𝑡 = (𝑎1,...,𝑎𝑡−1). A single 𝑓𝜃

is trained on all 𝑚, supporting the task-agnostic property in Sec. 2.

ERM

###### 3.2. Visual-ERM as an RL Reward

Pipeline. Following Fig. 3(b), each rollout draws 𝑦 ∼ 𝜋𝜃(· | 𝑥), renders 𝑦ˆimg = ℛ𝑚(𝑦), and queries Visual-ERM on (𝑥,𝑦ˆimg) to obtain a discrepancy set ℰ = {𝑒𝑘}𝐾𝑘=1 with severities 𝑠𝑘 ≥0. We pair this with a render-success reward (RSR) so that unrenderable code is excluded before fidelity is scored.

∑︀𝐾 𝑘=1 𝑠𝑘, normalize within the task batch 𝒯 :

Reward design. We aggregate severities into Sverm =

Sverm max𝑗∈𝒯 S(verm𝑗) + 𝜖

, (3)

̃︀Sverm =

and convert to a bounded reward

(︀

)︀

. (4) The overall RL reward is

1 − ̃︀Sverm, 0, 1

𝑟verm = clip

𝑟 = 𝑟rsr + 𝑟verm, (5) where 𝑟rsr = 1 if ℛ𝑚(𝑦) succeeds and 0 otherwise. Policy optimization. We optimize 𝜋𝜃 with a Group Relative Policy Optimization (GRPO)-based [32] objective using the reward in Eq. 5 and a Kullback–Leibler (KL) anchor to a reference policy 𝜋ref:

E𝑥∼𝒟[︁E𝑦∼𝜋

[︀

𝑟rsr(ℛ𝑚(𝑦))

max

𝜃(·|𝑥)

𝜃

(︀

)︀]︀

(6)

1 − 𝑆̃︀verm(𝑥, ℛ𝑚(𝑦)), 0, 1

+ clip

− 𝛽 KL(𝜋𝜃(· | 𝑥)‖𝜋ref(· | 𝑥))]︁,

where 𝛽 controls the strength of KL regularization and 𝒟 is the training distribution over reference images 𝑥. The training and downstream behavior of policies optimized under Eq. 6 are reported in Fig. 2 and Sec. 4.

###### 3.3. Visual-ERM for Test-Time Scaling

A second use of Visual-ERM, shown in Fig. 3(b), is to provide feedback for iterative self-refinement at inference time. The policy first produces an initial prediction

𝑦(0) ∼ 𝜋𝜃(· | 𝑥), (7)

which is rendered into 𝑦ˆimg(0) = ℛ𝑚(𝑦(0)) and evaluated by Visual-ERM:

ERM(︁𝑥, 𝑦ˆimg(0))︁, (8)

(︀

)︀

𝑟(0), 𝑓(0)

= 𝑓𝜃

where 𝑟(0) is the bounded reward from Eq. 5 and 𝑓(0) is the structured discrepancy description. If 𝑟(0) falls below a threshold, the policy revises its solution conditional on the previous draft and the feedback,

𝑦(𝑡+1) ∼ 𝜋𝜃(︁· | 𝑥, 𝑦(𝑡), 𝑓(𝑡))︁, (9)

and the loop repeats for up to 𝑇 steps. The interpretability of 𝑓(𝑡), rather than 𝑟(𝑡) alone, is what enables this revision step; a scalar-only reward could not localize what to fix. We evaluate this capability in Sec. 4.3; gains are bounded by the policy’s editing ability, since TTS reuses 𝜋𝜃 without weight updates.

Visual-ERM-Bench

[Figure 99]

Multi-Model Consensus

Model n

Metric

[Figure 100]

Filtering

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

Model 2

- Model 1
- Model 2

Single

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

Model 1

Checking

[Figure 112]

Visual-ERMBench

F1 Metric (TP,FP,FN)

[Figure 113]

[Figure 114]

[Figure 115]

Category

Location Severity Description

Y-axis label mismatch

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

Global

###### · · ·

[Figure 121]

[Figure 122]

Legend shifted right

[Figure 123]

[Figure 124]

Correlation Score

[Figure 125]

Chart Table SVG

[Figure 126]

[Figure 127]

[Figure 128]

Merging:

[Figure 129]

import numpy as np\n...... plt.savefig('figure.png', dpi=80)\nplt.clf()

Model n

###### · · ·

[Figure 130]

Z-axis label text differs

[Figure 131]

LVLM Pre-Annotation

Evaluation

Human Annotation

- Figure 4: VC-RewardBench. We construct VC-RewardBench by first leveraging several advanced proprietary models for preliminary annotation, followed by manual consolidation and filtering, resulting in 1,335 high-quality instances.

###### 3.4. VisualCritic-RewardBench

Existing reward benchmarks predominantly evaluate vision–language alignment [16], whereas the bottleneck for vision-to-code is image-to-image reconstruction fidelity. We therefore introduce VisualCritic-RewardBench (VC-RewardBench), a diagnostic suite that targets fine-grained image-to-image discrepancies in structured visuals.

VC-RewardBench contains 1,335 high-quality annotated instances, constructed following Fig. 4. Each instance pairs a reference image 𝑥 with a corrupted counterpart 𝑦ˆimg and a fine-grained discrepancy annotation 𝑎 covering type, location, description, and severity. We adopt a multi-model consensus protocol: independent annotations from GPT-5-mini [35], Gemini-2.5-Pro, and Gemini-3-Pro [8] reduce architecture-specific bias, and PhD-level reviewers consolidate the labels into a single expert-vetted set. Example cases are provided in Sec. D.

Because VC-RewardBench mixes structured fields (e.g., counts) with free-form content (e.g., descriptions), exact-match accuracy is unsuitable. We adopt an LLM-as-Judge protocol: a judge LLM matches predicted discrepancies 𝐴ˆ to ground-truth annotations 𝐴⋆ to identify TP/FP/FN and compute Precision/Recall/F1. Since models also output per-discrepancy severities, we sum severities per instance and report the Pearson correlation with the ground-truth totals to measure overall scoring consistency, denoted as 𝑆𝑐. VC-RewardBench is a diagnostic for 𝑓𝜃

itself, separate from the downstream RL and TTS evaluations in Sec. 4.

ERM

##### 4. Experiments

We evaluate Visual-ERM along three axes. (i) RL utility: whether using Visual-ERM as the reward in RL yields stronger policies across three vision-to-code tasks (Sec. 4.1). (ii) Reward quality: how well Visual-ERM judges fine-grained image-to-image discrepancies on VC-RewardBench (Sec. 4.2). (iii) Test-time utility: whether Visual-ERM’s critiques improve on test-time scaling (Sec. 4.3).

###### 4.1. Reinforcement Learning with Visual-ERM

Experimental Details and Benchmarks. We train Visual-ERM on top of Qwen3-VL-8B-Instruct [2]. To evaluate its utility as a reward model, we run GRPO [32] on three vision-to-code tasks: (1) Chart-to-Code, (2) Table-to-Markdown, and (3) SVG-to-Code, using Qwen3-VL-8B-Instruct as the policy model. For Chart-to-Code, we use ChartMimic [41] under both the direct (reproduce the input chart) and customized (generate a new chart under given style/data constraints) settings. For Table-to-Markdown, we report table-level metrics on OmniDocBench-v1/v1.5 [26] and olmOCRBench [28]. For SVG-to-Code, we evaluate on UniSVG [15]. We additionally compare against two stronger vision-to-code SFT policies, VinciCoder [47] and JanusCoderV8B [36]. Please refer to Sec. A for more experimental details.

Results on Chart-to-Code. We start with the Chart-to-Code task. Using Qwen3-VL-8B-Instruct [3] and VinciCoder-8B-SFT [47] as policies, we run GRPO with Visual-ERM as the reward. Tab. 1 shows that VisualERM-guided RL improves Qwen3-VL-8B-Instruct by +11.8 and +4.9 average points on ChartMimic-v2 direct and customized [41]. Starting from the already strong VinciCoder-8B-SFT, the same recipe still delivers +10.3 and +9.8 under the two settings.

- Table 1: Evaluation Results on Chart-to-Code Tasks. We evaluate on ChartMimic(Direct) and ChartMimic(Customized). ChartMimic(Direct) requires generating Python code that reproduces the input chart, while ChartMimic(Customized) requires generating code for a new chart that matches the given chart’s style and data constraints. Exec_rate denotes the execution success rate; Low and High denote the low-level and high-level scores, respectively.

Model

ChartMimic [41] (Direct) ChartMimic [41] (Customized)

Avg↑ Exec_rate↑ Low↑ High↑ Overall↑ Exec_rate↑ Low↑ High↑ Overall↑

Baseline

InternVL3.5-8B [38] 62.2 41.9 48.9 45.4 79.7 53.0 63.3 58.2 51.8 Qwen3-VL-8B-Instruct [3] 82.2 62.7 72.7 67.7 86.8 66.3 76.8 71.6 69.6 VinciCoder-8B-SFT [47] 86.2 68.2 77.5 72.9 86.2 52.3 71.5 61.9 67.4 JanusCoderV-8B [36] 80.6 65.8 73.2 69.5 80.7 66.7 74.2 70.4 70.0 ChartMaster-7B [37] 93.3 74.5 82.1 78.3 88.8 59.5 74.2 66.8 72.6

Qwen3-VL-8B-Instruct [3]

+ RL (DINO-based Reward) 90.8 71.1 81.9 76.5 91.7 71.2 80.3 75.8 76.1 + RL (Visual-ERM) 92.5 74.4 84.6 79.5 91.5 71.8 81.1 76.5 78.0 ∆vs. Qwen3-VL-8B-Instruct +10.3 +11.9 +11.9 +11.8 +4.7 +5.5 +4.3 +4.9 +8.4 VinciCoder-8B-SFT [47]

+ RL (DINO-based Reward) 90.8 73.1 82.3 77.7 91.5 57.0 77.5 67.3 72.5 + RL (Visual-ERM) 94.3 78.0 88.5 83.2 95.0 60.9 82.6 71.7 77.5 ∆vs. VinciCoder-8B-SFT +8.1 +9.8 +11.0 +10.3 +8.8 +8.6 +11.1 +9.8 +10.1

- Table 2: Evaluation Results on Table-to-Markdown. TEDS-S denotes TEDS-Structure-Only, and TA represents the table subtask of olmOCRBench. For Edit-Dist, where lower values indicate better performance, we use (100 − Edit-Dist) when computing the Average.

OmniDocBench [26] OmniDocBench-v1.5 [26] olmOCRBench [28]

Model

Avg↑ TEDS↑ TEDS-S↑ Edit-Dist↓ TEDS↑ TEDS-S↑ Edit-Dist↓ TA↑

Baseline

- Qwen2.5-VL-7B [4] 74.4 80.1 58.9 69.3 74.1 58.7 71.1 64.5 InternVL3.5-8B [38] 71.0 77.6 44.9 62.9 70.4 44.1 74.1 66.7 Qwen3-VL-8B-Instruct [3] 78.9 83.9 23.2 72.7 77.2 26.9 75.3 76.8

- Qwen3-VL-8B-Instruct [3]

+ RL (DINO-based Reward) 62.2 69.5 37.0 61.1 67.4 37.9 71.7 65.3 + RL (TEDS-based Reward) 79.2 82.9 31.6 73.0 76.6 35.3 78.6 74.8

+ RL (Visual-ERM) 81.4 86.3 20.7 75.4 80.4 24.2 78.1 79.5 ∆vs. Qwen3-VL-8B-Instruct +2.5 +2.4 +2.5 +2.7 +3.2 +2.7 +2.8 +2.7

We also compare Visual-ERM against the DINO-based RL reward in Tab. 1, and observe three advantages. (1) Stronger downstream policies: Visual-ERM-guided RL yields substantially larger gains than DINO-based RL on both Qwen3-VL-8B-Instruct and VinciCoder-8B-SFT backbones. DINO rewards reduce supervision to a fixed visual embedding space, where matching patch-level feature similarity prioritizes semantic alignment and global appearance while under-penalizing small but functionally critical deviations, so the proxy only loosely tracks human-perceived fidelity. (2) Cross-modal coverage: DINO rewards are unimodal and weak at penalizing errors carried by rendered text, allowing policies to improve the proxy while degrading textual faithfulness; Visual-ERM combines visual perception with cross-modal grounding, scoring reconstructions on both visual structure and rendered text. (3) Fine-grained interpretable feedback: trained as a generative judge, Visual-ERM emits decomposed discrepancy descriptions rather than a single scalar, which directly supports test-time scaling via reward-guided refinement (Sec. 4.3).

Results on Table-to-Markdown Parsing. We next turn to the Table-to-Markdown task. Starting from Qwen3VL-8B-Instruct, we run GRPO with Visual-ERM as the reward and compare against two alternative signals: a rule-based Tree-Edit-Distance Similarity (TEDS) score, and a DINO-based feature similarity. Tab. 2 shows that Visual-ERM-based RL delivers consistent improvements, with an overall gain of +2.7; the gains span both textual-recognition and structural-reconstruction metrics, consistent with the cross-modal coverage of Visual-ERM’s feedback.

Both alternatives fall short on this task. With TEDS, the training reward rises steadily, yet the policy improves only marginally on the target TEDS metric and slightly degrades on others, a pattern consistent with reward shortcutting in a purely textual/structural space that ignores visual cues. DINO-based RL fares worse: unlike charts, where DINO features still carry useful visual cues, tables are dominated by precise text and layout, and DINO-based RL not only fails to yield gains on Table-to-Markdown but also degrades performance across benchmarks.

- Table 3: Results on UniSVG. SSIM/LPIPS measure pixellevel and perceptual similarity, CLIP measures semantic alignment, and Score aggregates them.

Table 4: Test-Time Scaling on Chart-to-Code. Visual-ERM enables iterative self-reflection and revision; for the reflection turns study, see Tab. 6.

###### ChartMimic [41]

UniSVG [15] (ISVGEN) SSIM↑ LPIPS↓ CLIP↑ Score↑ Baseline

Model

Avg↑ Direct↑ Cust.↑

Model

Qwen3-VL-8B-Instruct [3] 67.7 71.6 69.6 + Reflection (self) 61.9 69.5 65.7 + Reflection (Visual-ERM) 75.6 79.5 77.6

Qwen3-VL-8B 60.9 60.0 73.3 64.2 JanusCoderV-8B 58.1 61.7 72.6 62.8

+ RL (Visual-ERM) 79.5 76.5 78.0 + Reflection (self) 75.2 76.9 76.1 + Reflection (Visual-ERM) 80.3 82.0 81.1

VinciCoder-8B-SFT 81.1 19.2 92.5 87.9 + RL (DINO) 76.9 23.3 92.7 86.3 + RL (Visual-ERM) 85.2 12.6 95.2 91.6

- Table 5: Evaluation Results on VC-RewardBench. We evaluate a range of proprietary and open-source models on

Visual-ERM-Bench. 𝐹1ℎ denotes 𝐹1ℎ𝑎𝑟𝑑, the strict-match F1 score; 𝐹1𝑠 denotes 𝐹1𝑠𝑜𝑓𝑡, the soft-match F1 score; and 𝑆𝑐 denotes the correlation score, measuring the overall agreement between the predicted scores and the ground-truth labels.

Chart Table SVG AVG

Model

𝐹1ℎ 𝐹1𝑠 𝑆𝑐 𝐹1ℎ 𝐹1𝑠 𝑆𝑐 𝐹1ℎ 𝐹1𝑠 𝑆𝑐 𝐹1ℎ 𝐹1𝑠 𝑆𝑐

Proprietary

GPT-4o [1] 22.8 28.3 48.5 32.9 35.7 49.5 13.0 19.3 50.3 25.0 29.5 56.5 GPT-5.2 [35] 30.1 32.6 64.8 39.3 40.6 54.6 28.5 32.2 61.1 32.7 35.0 58.9 Gemini-2.5-Pro [8] 33.7 37.5 61.8 46.4 48.0 49.9 29.3 34.3 63.3 37.8 40.9 59.1 Gemini-3-Flash [10] 38.5 41.3 62.8 48.1 50.1 45.6 33.3 67.5 64.3 40.6 43.4 53.4

Open-source

Qwen2.5-VL-7B [4] 3.9 5.4 11.2 2.3 3.1 12.6 1.9 7.5 37.9 2.8 5.1 15.2 InternVL3.5-8B [38] 2.5 5.7 11.0 9.9 10.9 31.7 6.1 13.1 48.9 6.7 9.6 32.5 Qwen3-VL-8B-Instruct [3] 3.3 3.5 3.8 7.0 7.8 21.4 6.1 9.4 27.1 5.3 6.5 17.5 Qwen3-VL-235B-Instruct [3] 28.0 31.8 47.2 35.7 37.4 56.2 19.4 22.8 51.5 29.5 32.4 56.2

Visual-ERM 39.9 42.8 61.2 56.4 57.6 74.8 28.3 32.6 59.6 42.1 44.7 58.4 ∆vs. Qwen3-VL-8B-Instruct +36.6 +39.3 +57.4 +49.4 +49.8 +53.4 +22.2 +23.2 +32.5 +36.8 +38.2 +40.9

Results on SVG-to-Code Parsing. We use VinciCoder-8B-SFT as policy and DINO-based RL as the featuresimilarity baseline. Compared with charts and tables, SVG reconstruction relies more on visual geometry and styling cues and less on textual content. Tab. 3 shows that Visual-ERM-based RL delivers consistent gains on the backbone. Notably, while DINO-based RL leads to performance degradation on the strong VinciCoder-8B-SFT baseline, Visual-ERM maintains its effectiveness. This suggests that Visual-ERM provides more robust and precise guidance than standard feature-similarity rewards, which may fail to provide meaningful gradients for highly optimized policies.

###### 4.2. Reward-Model Evaluation on VC-RewardBench

In Sec. 4.1, the selected benchmarks measure vision-to-code parsing quality but do not directly probe whether a model can judge reconstruction fidelity or surface fine-grained discrepancies. We therefore evaluate on VC-RewardBench, which targets fine-grained discrepancy detection and interpretable feedback across visionto-code tasks. Tab. 5 reports the results on VC-RewardBench. Built on Qwen3-VL-8B-Instruct, Visual-ERM improves 𝐹1ℎ/𝐹1𝑠/𝑆𝑐 over its base by +36.8/+38.2/+40.9. Even Qwen3-VL-235B-Instruct struggles with fine-grained visual and textual discrepancies, whereas Visual-ERM at 8B matches or surpasses the strongest proprietary baselines on 𝐹1ℎ/𝐹1𝑠 and remains competitive on 𝑆𝑐. We read this as evidence that fine-grained discrepancy detection and fidelity judgment come from reward-model specialization of a general-purpose LVLM, not from scale alone.

###### 4.3. Ablation Studies

Test-Time Scaling with Visual-ERM. Visual-ERM’s feedback is also useful at inference time. Rather than emitting only a scalar score, it returns localized discrepancy descriptions that a policy can act on. We insert Visual-ERM into the decoding loop: it critiques each candidate, and the policy revises its output before finalizing. With three reflection rounds on Chart-to-Code, this loop adds +8.0 Avg over the base Qwen3-VL-8B-Instruct and a further +3.1 on top of the Visual-ERM-RL-tuned policy (Tab. 4). Qualitative cases are provided in Fig. 5.

###### Ablation of the Reflection Rounds in Test-Time Scaling. In Tab. 4, we use a three-round reflection and

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

X/Y limits and plotted geometry differ, Color mapping differs ......

Inference Reflection

GT Image Pred Image

Pred Image Refined Image

Refined Image

GT Image

- Figure 5: Test-Time Scaling Cases. Using Visual-ERM’s fine-grained feedback, we enable an inference, reflection and refinement loop. Predictions are generated as text and rendered as images for visualization.

- Table 6: Ablation on the number of reflection rounds in test-time scaling. We evaluate test-time scaling with different numbers of reflection and revision rounds d.

ChartMimic [41] (Direct) ChartMimic [41] (Customized)

Model

Avg↑ exec_rate↑ average↑ gpt_score↑ overall↑ exec_rate↑ average↑ gpt_score↑ overall↑

Qwen3-VL-8B-Instruct [3] (+RL on Visual-ERM) 92.5 74.4 84.6 79.5 91.5 71.8 81.1 76.5 78.0

- + Reflection (𝑤 Visual-ERM) 2 rounds 92.8 75.2 84.5 79.8 92.0 74.4 82.8 78.6 79.2

- + Reflection (𝑤 Visual-ERM) 3 rounds 92.3 74.8 85.8 80.3 94.7 77.1 86.8 82.0 81.1

- + Reflection (𝑤 Visual-ERM) 4 rounds 92.0 74.5 85.4 80.0 94.2 76.9 86.1 81.5 80.7

revision pipeline by default. To study how the number of reflection rounds affects performance, we conduct an ablation where we vary the number of rounds while keeping the rest of the inference setup identical. Results are summarized in Tab. 6. Overall, increasing the number of reflection rounds yields consistent improvements, with diminishing returns beyond three rounds.

Additional analysis. We provide extended results in the appendix, such as an evaluation on general VQA benchmarks to verify that Visual-ERM-guided RL preserves broad multimodal competence (Sec. B.1), an ablation on multi-task data mixing for both reward modeling (Sec. B.2.1) and downstream RL (Sec. B.2.2), a robustness study across LLM judges on VC-RewardBench (Sec. B.3), and an ablation on reward design with and without the render-success term (Sec. B.4).

- 5. Related Work

Reward Models. To enable effective RL [19, 32], reward models (RMs) provide feedback that guide policy optimization. RMs can take several forms: (1) Bradley–Terry (BT) models that learn a scalar reward from pairwise comparisons and are often instantiated as discriminative rankers [5, 44, 50]; (2) generative RMs that produce natural language critiques or judgments which can be mapped to rewards [14, 20, 39, 43]; and (3) thinking/agentic RMs that perform multi-step evaluation, e.g., decomposing criteria, self-reflection, or invoking tools before returning a final score [9, 17, 27]. Most prior RMs are developed for text-centric generation (e.g., writing and dialogue) and do not support visual-to-code tasks, where quality is mainly determined by visual fidelity rather than text. Therefore, we propose Visual-ERM, a visual equivalence reward model for visual-to-code tasks.

Visual-to-Code Tasks. Visual-to-code spans a family of practical structured perception tasks that convert images into executable or structured representations. Chart-to-Code aims to parse charts into Python programs that can faithfully reproduce the original plots [37, 46, 48]. Table-to-Markdown converts tabular images into structured formats such as Markdown or HTML [18, 24, 45]. SVG-to-Code translates vector graphics into code representations [15, 42]. Such structured outputs facilitate downstream use and improve usability in real-world applications.

RL for Visual-to-Code Tasks. Despite its practical importance, visual-to-code remains challenging. Supervised fine-tuning (SFT) typically relies on large-scale, high-quality datasets [11, 48, 49], which are costly to curate. RL has been explored as an alternative, yet existing reward designs often fall into two extremes: (i) textual rule-based rewards [18], which score string-level or structural proxies in the text space without directly leveraging the visual evidence, and thus may introduce modality bias; and (ii) visual-encoder similarity-based rewards, such as DINO-based [34] similarity [31, 37, 47], which compare representations extracted by vision encoders but are often coarse-grained and offer limited interpretability. Motivated by these limitations, we propose Visual-ERM, a cross-modal reward model that provides fine-grained, interpretable, and task-agnostic feedback for Visual-to-Code.

##### 6. Conclusion

We propose the Visual Equivalence Reward Model (Visual-ERM), a generative reward model that evaluates vision-to-code outputs in visual space, providing fine-grained, interpretable, and task-agnostic supervision. We further introduce VisualCritic-RewardBench to directly assess image-to-image discrepancy judgment across vision-to-code tasks. Experiments show that Visual-ERM is an effective supervisor for both reinforcement learning and test-time scaling, consistently improving vision-to-code performance across multiple tasks.

##### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 5
- [2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xiong-Hui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Rongyao Fang, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Li Ying Meng, Xuancheng Ren, Xin yi Ren, Sibo Song, Yu-Chen Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yihe Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Botao Zheng, Humen Zhong, Jingren Zhou, Fanxi Zhou, Jingren Zhou, Yuanzhi Zhu, and Keming Zhu. Qwen3-vl technical report. ArXiv, abs/2511.21631, 2025. 4.1
- [3] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 1, 1, 3.1, 4.1, 1, 2, 4, 5, 6, A.1
- [4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 2, 5
- [5] Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, Xiao wen Dong, Haodong Duan, Qi Fan, Zhaoye Fei, Yang Gao, Jiaye Ge, Chenya Gu, Yuzhe Gu, Tao Gui, Aijia Guo, Qipeng Guo, Conghui He, Yingfan Hu, Ting Huang, Tao Jiang, Penglong Jiao, Zhen Jin, Zhikai Lei, Jiaxing Li, Jingwen Li, Linyang Li, Shuaibin Li, Wei Li, Yining Li, Hong wei Liu, Jiangning Liu, Jiawei Hong, Kaiwen Liu, Kui-Jie Liu, Xiaoran Liu, Chen Lv, Haijun Lv, Kai Lv, Li Ma, Runyuan Ma, Zerun Ma, Wenchang Ning, Linke Ouyang, Jiantao Qiu, Yuan Qu, Fukai Shang, Yunfan Shao, Demin Song, Zifan Song, Zhihao Sui, Peng Sun, Yu Sun, Huanze Tang, Bin Wang, Guoteng Wang, Jiaqi Wang, Jiayu Wang, Rui Wang, Yudong Wang, Ziyi Wang, Xing Wei, Qizhen Weng, Fan Wu, Yingtong Xiong, Chao Xu, Rui Ze Xu, Hang Yan, Yirong Yan, Xiaogui Yang, Haochen Ye, Huaiyuan Ying, Jia Yu, Jing Yu, Yuhang Zang, Chuyu Zhang, Li Zhang, Pan Zhang, Peng Zhang, Ruijie Zhang, Shuo Zhang, Songyang Zhang, Wenjian Zhang, Wenwei Zhang, Xingcheng Zhang, Xinyue Zhang, Hui Zhao, Qian Zhao, Xiaomeng Zhao, Fen-Fang Zhou, Zaida Zhou, Jingming Zhuo, Yi-Ling Zou, Xipeng Qiu, Yu Qiao, and Dahua Lin. Internlm2 technical report. ArXiv, abs/2403.17297, 2024. 5
- [6] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021. 2
- [7] Lei Chen, Xuanle Zhao, Zhixiong Zeng, Jing Huang, Liming Zheng, Yufeng Zhong, and Lin Ma. Breaking the sft plateau: Multimodal structured reinforcement learning for chart-to-code generation. arXiv preprint arXiv:2508.13587, 2025. A.2
- [8] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 3.4, 5, A.2

- [9] Shengyuan Ding, Xinyu Fang, Ziyu Liu, Yuhang Zang, Yuhang Cao, Xiangyu Zhao, Haodong Duan, Xiaoyi Dong, Jianze Liang, Bin Wang, et al. Arm-thinker: Reinforcing multimodal generative reward models with agentic tool use and visual reasoning. arXiv preprint arXiv:2512.05111, 2025. 5
- [10] Google. Gemini 3.1 Pro: A smarter model for your most complex tasks, February 2026. 5
- [11] Yi Gui, Zhen Li, Yao Wan, Yemin Shi, Hongyu Zhang, Bohua Chen, Yi Su, Dongping Chen, Siyuan Wu, Xing Zhou, et al. Webcode2m: A real-world dataset for code generation from webpage designs. In Proceedings of the ACM on Web Conference 2025, pages 1834–1845, 2025. 5
- [12] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv:2412.16720, 2024. 1
- [13] Lingjie Jiang, Shaohan Huang, Xun Wu, Yixia Li, Dongdong Zhang, and Furu Wei. Viscodex: Unified multimodal code generation via merging vision and coding models. arXiv preprint arXiv:2508.09945,

2025. A.2

- [14] Dahyun Kim, Chanjun Park, Sanghoon Kim, Wonsung Lee, Wonho Song, Yunsu Kim, Hyeonwoo Kim, Yungi Kim, Hyeonju Lee, Jihoo Kim, Changbae Ahn, Seonghoon Yang, Sukyung Lee, Hyunbyung Park, Gyoungjin Gim, Mikyoung Cha, Hwalsuk Lee, and Sunghun Kim. Solar 10.7b: Scaling large language models with simple yet effective depth up-scaling. ArXiv, abs/2312.15166, 2023. 5
- [15] Jinke Li, Jiarui Yu, Chenxing Wei, Hande Dong, Qiang Lin, Liangjing Yang, Zhicai Wang, and Yanbin Hao. Unisvg: A unified dataset for vector graphic understanding and generation with multimodal large language models. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 13156–13163,

2025. 4.1, 3, 5, A.2

- [16] Lei Li, Yuancheng Wei, Zhihui Xie, Xuqing Yang, Yifan Song, Peiyi Wang, Chenxin An, Tianyu Liu, Sujian Li, Bill Yuchen Lin, et al. Vl-rewardbench: A challenging benchmark for vision-language generative reward models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24657–24668,

2025. 3.4

- [17] Renhao Li, Jianhong Tu, Yang Su, Hamid Alinejad-Rokny, Derek F Wong, Junyang Lin, and Min Yang. One model to critique them all: Rewarding agentic tool-use via efficient reasoning. arXiv preprint arXiv:2510.26167, 2025. 5
- [18] Jun Ling, Yao Qi, Tao Huang, Shibo Zhou, Yanqin Huang, Jiang Yang, Ziqi Song, Ying Zhou, Yang Yang, Heng Tao Shen, et al. Table2latex-rl: High-fidelity latex code generation from table images via reinforced multimodal language models. arXiv preprint arXiv:2509.17589, 2025. 1, 5
- [19] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual-rft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025. 5
- [20] Ziyu Liu, Yuhang Zang, Shengyuan Ding, Yuhang Cao, Xiaoyi Dong, Haodong Duan, Dahua Lin, and Jiaqi Wang. Spark: Synergistic policy and reward co-evolving framework. arXiv preprint arXiv:2509.22624,

2025. 5

- [21] Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the association for computational linguistics: ACL 2022, pages 2263–2279, 2022. B.1
- [22] Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1697–1706, 2022. B.1
- [23] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209,

2021. B.1

- [24] Junbo Niu, Zheng Liu, Zhuangcheng Gu, Bin Wang, Linke Ouyang, Zhiyuan Zhao, Tao Chu, Tianyao He, Fan Wu, Qintong Zhang, et al. Mineru2. 5: A decoupled vision-language model for efficient high-resolution document parsing. arXiv preprint arXiv:2509.22186, 2025. 1, 5

- [25] OpenAI. Openai o3-mini system card, 2025. 1
- [26] Linke Ouyang, Yuan Qu, Hongbin Zhou, Jiawei Zhu, Rui Zhang, Qunshu Lin, Bin Wang, Zhiyuan Zhao, Man Jiang, Xiaomeng Zhao, et al. Omnidocbench: Benchmarking diverse pdf document parsing with comprehensive annotations. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24838–24848, 2025. 4.1, 2
- [27] Hao Peng, Yunjia Qi, Xiaozhi Wang, Zijun Yao, Bin Xu, Lei Hou, and Juanzi Li. Agentic reward modeling: Integrating human preferences with verifiable correctness signals for reliable reward systems. arXiv preprint arXiv:2502.19328, 2025. 5
- [28] Jake Poznanski, Aman Rangapur, Jon Borchardt, Jason Dunkelberger, Regan Huff, Daniel Lin, Christopher Wilhelm, Kyle Lo, and Luca Soldaini. olmocr: Unlocking trillions of tokens in pdfs with vision language models. arXiv preprint arXiv:2502.18443, 2025. 4.1, 2
- [29] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR,

2021. 1, 2

- [30] Juan A Rodriguez, Shubham Agarwal, Issam H Laradji, Pau Rodriguez, David Vazquez, Christopher Pal, and Marco Pedersoli. Starvector: Generating scalable vector graphics code from images. arXiv preprint arXiv:2312.11556, 2023. A.2
- [31] Juan A Rodriguez, Haotian Zhang, Abhay Puri, Aarash Feizi, Rishav Pramanik, Pascal Wichmann, Arnab Mondal, Mohammad Reza Samsami, Rabiul Awal, Perouz Taslakian, et al. Rendering-aware reinforcement learning for vector graphics generation. arXiv preprint arXiv:2505.20793, 2025. 5
- [32] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 3.2, 4.1, 5
- [33] Chenglei Si, Yanzhe Zhang, Ryan Li, Zhengyuan Yang, Ruibo Liu, and Diyi Yang. Design2Code: Benchmarking multimodal code generation for automated front-end engineering. In ACL, 2025. 1
- [34] Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104,

2025. 1, 1, 5

- [35] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025. 3.1, 3.4, 5, A.2
- [36] Qiushi Sun, Jingyang Gong, Yang Liu, Qiaosheng Chen, Lei Li, Kai Chen, Qipeng Guo, Ben Kao, and Fei Yuan. Januscoder: Towards a foundational visual-programmatic interface for code intelligence. arXiv preprint arXiv:2510.23538, 2025. 4.1, 1
- [37] Wentao Tan, Qiong Cao, Chao Xue, Yibing Zhan, Changxing Ding, and Xiaodong He. Chartmaster: Advancing chart-to-code generation with real-world charts and chart similarity reinforcement learning. arXiv preprint arXiv:2508.17608, 2025. 1, 1, 5
- [38] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. 1, 2, 5
- [39] Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang. Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236, 2025. 5
- [40] Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, et al. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. Advances in Neural Information Processing Systems, 37:113569–113697, 2024. B.1

- [41] Cheng Yang, Chufan Shi, Yaxin Liu, Bo Shui, Junjie Wang, Mohan Jing, Linran Xu, Xinyu Zhu, Siheng Li, Yuxiang Zhang, et al. Chartmimic: Evaluating lmm’s cross-modal reasoning capability via chart-to-code generation. arXiv preprint arXiv:2406.09961, 2024. 4.1, 4.1, 1, 4, 6
- [42] Yiying Yang, Wei Cheng, Sijin Chen, Xianfang Zeng, Fukun Yin, Jiaxu Zhang, Liao Wang, Gang Yu, Xingjun Ma, and Yu-Gang Jiang. Omnisvg: A unified scalable vector graphics generation model. arXiv preprint arXiv:2504.06263, 2025. 5
- [43] Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Sainbayar Sukhbaatar, Jing Xu, and Jason E. Weston. Self-rewarding language models. ArXiv, abs/2401.10020, 2024. 5
- [44] Yuhang Zang, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Ziyu Liu, Shengyuan Ding, Shenxi Wu, Yubo Ma, Haodong Duan, Wenwei Zhang, et al. Internlm-xcomposer2. 5-reward: A simple yet effective multi-modal reward model. arXiv preprint arXiv:2501.12368, 2025. 5
- [45] Jiarui Zhang, Yuliang Liu, Zijun Wu, Guosheng Pang, Zhili Ye, Yupei Zhong, Junteng Ma, Tao Wei, Haiyang Xu, Weikai Chen, et al. Monkeyocr v1. 5 technical report: Unlocking robust document parsing for complex patterns. arXiv preprint arXiv:2511.10390, 2025. 5
- [46] Zhihan Zhang, Yixin Cao, and Lizi Liao. Enhancing chart-to-code generation in multimodal large language models via iterative dual preference learning. arXiv preprint arXiv:2504.02906, 2025. 5
- [47] Xuanle Zhao, Deyang Jiang, Zhixiong Zeng, Lei Chen, Haibo Qiu, Jing Huang, Yufeng Zhong, Liming Zheng, Yilin Cao, and Lin Ma. Vincicoder: Unifying multimodal code generation via coarse-to-fine visual reinforcement learning. arXiv preprint arXiv:2511.00391, 2025. 1, 4.1, 4.1, 1, 5, A.1, A.5
- [48] Xuanle Zhao, Xianzhen Luo, Qi Shi, Chi Chen, Shuo Wang, Zhiyuan Liu, and Maosong Sun. Chartcoder: Advancing multimodal large language model for chart-to-code generation. arXiv preprint arXiv:2501.06598,

2025. 1, 5, A.2

- [49] Xu Zhong, Elaheh ShafieiBavani, and Antonio Jimeno Yepes. Image-based table recognition: data, model, and evaluation. arXiv preprint arXiv:1911.10683, 2019. 5
- [50] Banghua Zhu, Evan Frick, Tianhao Wu, Hanlin Zhu, and Jiantao Jiao. Starling-7b: Improving llm helpfulness & harmlessness with rlaif, November 2023. 5

##### Appendix

In the appendix, we provide additional materials to support the main paper and facilitate a deeper understanding of Visual-ERM.

First, in Sec. A, we provide detailed descriptions of the models, datasets, benchmark construction, experimental setups and computational efficiency throughout the paper, aiming to ensure reproducibility.

Second, Sec. B presents additional experimental analyses and ablation studies, including evaluations on general VQA benchmarks, investigations of multi-task reward modeling, robustness analyses of different judge models, further studies on reward design, test-time scaling and so on.

Third, in Sec. C, we list the full prompt templates used in our experiments, covering reward model data generation, Visual-ERM inference across different vision-to-code tasks, and the evaluation protocol of VC-RewardBench.

Finally, Sec. D provides representative qualitative case studies from chart-to-code, table-to-markdown, and SVG-to-code tasks, illustrating typical visual discrepancies and the fine-grained feedback produced by Visual-ERM.

##### A. More Details

###### A.1. Models

This work involves multiple models of different roles and scales. For clarity and reproducibility, we summarize the models used throughout the paper and their corresponding usage in each stage of the pipeline.

Visual-ERM (Reward Model). We build Visual-ERM on top of Qwen3-VL-8B-Instruct [3]. While Qwen3-VL-8BInstruct already exhibits strong multimodal perception and reasoning abilities, we find that it is not a reliable image-to-image discrepancy judge out of the box—especially for structured visuals where the key differences are often text- and layout-centric. This limitation is empirically reflected in Sec. 4.2, where the base model shows poor discrimination on fine-grained visual mismatches.

Starting from this 8B backbone, we train Visual-ERM on a large-scale reward-modeling corpus spanning three vision-to-code domains: Chart-to-Code (104K), Table-to-Markdown (125K), and SVG-to-Code (111K). This training equips Visual-ERM with task-agnostic yet fine-grained judgment capabilities. Notably, despite its relatively small size, Visual-ERM substantially outperforms Qwen3-VL-235B-Instruct as a judge and achieves performance competitive with strong proprietary models, demonstrating that targeted reward model training can be more effective than simply scaling up a general-purpose LVLM.

Policy models for RL. For reinforcement learning, we consider two types of policy backbones. The first is Qwen3-VL-8B-Instruct, which serves as our default policy model across tasks. In this setting, we use Visual-ERM as the reward model and optimize Qwen3-VL-8B-Instruct with RL to improve its vision-to-code generation quality.

In addition, to test whether Visual-ERM can also improve stronger specialized parsers, we use VinciCoder8B-SFT [47] as an alternative policy backbone on Chart-to-Code and SVG-to-Code. VinciCoder-8B-SFT provides a stronger supervised baseline for these tasks, and applying Visual-ERM-guided RL on top of it allows us to evaluate the reward model’s transferability and its ability to refine already-competitive policies.

For Table-to-Markdown, we do not include VinciCoder-8B-SFT because it does not have a comparable task-specific SFT baseline for tables; therefore, we only use Qwen3-VL-8B-Instruct as the policy model for RL in this setting.

###### A.2. Datasets

We use multiple datasets throughout the paper for reward-model training, RL policy optimization, and benchmark construction. For completeness, we summarize them here. The annotations and training data will be released publicly in the future, and qualitative examples are provided in Sec. D.

### Training Data Geneartion (Neurips)

###### (a) Training Data Gen

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

Systematic Coverage

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

Text:

[Figure 153]

Targeted Edit

[Figure 154]

[Figure 155]

Category Location Severity Description

Code & Markdown

[Figure 156]

Model Router

Propri etary

Model Pools

GT

GT Pred

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

SVG Chart

[Figure 161]

[Figure 162]

[Figure 163]

Render image

[Figure 164]

GT

Natural Inference

[Figure 165]

Empirical Realism

Pred

Diverse Coverage

Annotation Schema

Table

Pred

Positive & Negative Pairs

GT Data (Image & Text)

- Figure 6: Training Data Generation. We construct our Visual-ERM’s training data as shown in Figure. Reward-model training data.

We first collect a pool of images and their corresponding ground-truth annotations from public vision-to-code datasets spanning charts, tables, and SVG-style graphics [7, 13, 15, 30, 48]. However, these raw pairs are not directly suitable for training a reward model, because Visual-ERM requires supervision on image-to-image discrepancies rather than only paired input-output examples. We therefore construct training instances by synthesizing realistic error patterns and converting them into rendered visual differences.

Concretely, we create imperfect predictions via two complementary routes: (i) error injection on groundtruth annotations using a strong yet cost-effective model (GPT-5-mini) as well as Qwen3-VL-235B-Instruct, and (ii) model-generated predictions by running Qwen3-VL-8B-Instruct on a subset of ground-truth images to obtain naturally occurring parsing errors. We then render each prediction back into an image, forming an (Original Image, Generated Image) pair that reflects the types of distortions encountered in practice. Given each image pair, we use GPT-5-mini to produce structured annotations, including fine-grained error descriptions (category, location, severity), as illustrated in Sec. D. This procedure yields three reward-modeling datasets: 104K instances for Chart-to-Code, 125K for Table-to-Markdown, and 111K for SVG-to-Code. The trainning data generation pipeline is shown in Fig. 6.

RL training data. The RL data are drawn from the same underlying sources, but require substantially less processing. Unlike reward-model training, RL does not require free-form error annotations: during training, Visual-ERM directly compares the original image with the re-rendered prediction and outputs an image-to-image reward. As a result, we only need the ground-truth images, and in many cases the ground-truth parsing text is not necessary. In our experiments, we use 11K images for Chart-to-Code RL, 40K for Table-to-Markdown RL, and 10K for SVG-to-Code RL.

VC-RewardBench construction data. We construct VC-RewardBench using a similar discrepancy-centric philosophy, but with substantially stricter quality control to ensure reliability as an evaluation benchmark. We begin with a candidate pool of 4.5K image pairs and obtain annotations from three state-of-the-art proprietary models: GPT-5-mini [35], Gemini2.5-Pro, and Gemini3-Pro [8]. We then manually consolidate, filter, and curate the collected annotations to resolve inconsistencies and remove ambiguous or low-quality cases. The final benchmark contains 1,335 carefully vetted examples, which we use to evaluate fine-grained image-to-image judging ability in structured visual domains.

###### A.3. VC-RewardBench Statistics

This section reports detailed statistics of VC-RewardBench. The benchmark contains 1,335 curated examples spanning three structured-visual domains: 595 chart instances, 442 SVG instances, and 298 table instances.

Beyond instance counts, VC-RewardBench provides fine-grained error annotations that characterize the discrepancy types between the original image and the re-rendered prediction. For the chart subset, the annotations include 231 text_error, 619 style_error, 462 data_error, and 62 structure_error. For the table subset, we annotate 884 text_error, 284 numeric_error, and 185 layout_error. For the SVG subset, the error annotations contain 151 style_error, 218 structure_error, 360 shape_error, and 12 text_symbol_error.

Overall, these statistics indicate that VC-RewardBench covers a diverse range of realistic failure modes. In particular, chart examples are dominated by style- and data-related discrepancies, tables primarily exhibit textcentric recognition errors with a non-trivial portion of numeric and layout issues, and SVG examples emphasize geometric shape_error and structural composition mismatches. This diversity makes VC-RewardBench a challenging and diagnostic benchmark for evaluating fine-grained image-to-image judging capability.

###### A.4. Experimental Setups

We implement supervised training and RL fine-tuning using LLaMA-Factory and the VERL framework. For training Visual-ERM, we use a learning rate of 1 × 10−5 with a batch size of 32. We train for 3 epochs over the reward-modeling dataset. For reinforcement learning, we adopt GRPO with a learning rate of 1 × 10−6 and a batch size of 256. For each prompt, we sample 8 rollouts per GRPO update step to estimate the policy gradients.

###### A.5. Computational Efficiency and Training Latency

- A primary concern in scaling reinforcement learning (RL) is the computational overhead introduced by reward models, particularly as their parameter count increases. We systematically benchmark the training latency of three reward supervision paradigms under identical experimental configurations: (1) Text-based rules (e.g., TEDS); (2) Vision-encoder similarity (DINOv2-large, 0.3B), following the framework in [47]; and (3) our proposed Visual-ERM (8B).

While text-based rewards are predictably the most efficient at 0.15 hours per step, they yield the least effective policy performance. Notably, Visual-ERM demonstrates a distinct computational advantage: despite its 8B parameter scale, it achieves a per-step latency of 0.17 hours, outperforming the 0.24 hours required by the 0.3B DINOv2-large baseline. This efficiency gap arises because DINO-based rewards [47], necessitate segmenting high-resolution images into multiple sub-patches to ensure scoring precision, which introduces significant computational overhead. In contrast, Visual-ERM’s unified multimodal architecture processes structured visuals natively, providing high-fidelity reward signals without the need for additional pre-processing, thereby maintaining high training throughput.

- B. Additional Experimental Analyses

###### B.1. Evaluation on General VQA Benchmarks

While Visual-ERM substantially improves the policy model on a diverse set of vision-to-code tasks, it is important to examine whether these gains come with any trade-offs on broader multimodal capabilities. In particular, reinforcement learning with task-specific rewards may induce distributional shifts or over-specialization, potentially degrading performance on general-purpose visual question answering (VQA) benchmarks.

To assess the generality of the learned policy, we evaluate the RL-trained model on a suite of established VQA benchmarks that are closely related to the structured visual domains considered in this work. Concretely, we select multiple chart- and document-centric benchmarks that require fine-grained visual perception, text recognition, and cross-modal reasoning, thereby serving as a diagnostic for whether Visual-ERM-guided optimization preserves general visual understanding skills beyond code generation. These benchmarks include ChartQA [21], CharXiv_DQ [40], CharXiv_RQ [40], DocVQA [23] and InfoVQA [22].

Tab. 7 summarizes the results. Overall, Visual-ERM-guided RL does not degrade general VQA performance: the average score remains essentially stable and even shows slight improvements for two of the three RL policies. Concretely, compared to the base Qwen3-VL-8B-Instruct (AVG 78.3), RL on Chart-to-Code reaches 78.4 (+0.1) and RL on SVG-to-Code reaches 78.5 (+0.2), while RL on Table-to-Markdown is comparable at 78.1 (-0.2). At the benchmark level, the most consistent change is on CharXiv-RQ, where all RL variants improve over the base (46.0 → 47.2 / 46.7 / 46.4), suggesting better fine-grained reasoning over chart-centric queries. Meanwhile, document-centric performance is preserved: DocVQA remains nearly unchanged (95.6 → 95.6 / 95.5 / 95.8), and InfoVQA is stable with a small gain for the SVG RL policy (83.1 → 83.6). These results indicate that the gains on vision-to-code do not arise from sacrificing general multimodal competence; instead, Visual-ERM

- Table 7: Results on general VQA benchmarks. We evaluate the policy model trained with Visual-ERM-guided RL on multiple general-purpose benchmarks (with an emphasis on chart- and document-centric VQA) to examine whether its improved vision-to-code performance comes at the cost of general multimodal capability.

Models ChartQA_TEST CharXiv_DQ CharXiv_RQ DocVQA_VAL InfoVQA_VAL AVG Qwen3-VL-8B-Instruct 82.9 83.8 46.0 95.6 83.1 78.3 + RL on Chart-to-Code 82.6 83.5 47.2 95.6 83.1 78.4 + RL on Table-to-Markdown 81.5 83.8 46.7 95.5 83.1 78.1 + RL on SVG-to-Code 83.2 83.5 46.4 95.8 83.6 78.5

- Table 8: Effect of Multi-Task Data Mixing. We compare reward models trained on mixed multi-task data vs. single-task data.

Chart Table SVG AVG

Model

𝐹1ℎ 𝐹1𝑠 𝑆𝑐 𝐹1ℎ 𝐹1𝑠 𝑆𝑐 𝐹1ℎ 𝐹1𝑠 𝑆𝑐 𝐹1ℎ 𝐹1𝑠 𝑆𝑐

Visual-ERM

+ Chart-Data-Only 40.1 42.6 58.1 17.2 20.4 10.5 11.3 15.3 50.1 26.3 29.3 23.7 + Table-Data-Only 17.1 20.1 49.3 53.6 54.9 68.8 14.1 19.0 55.2 31.7 34.2 56.6 + SVG-Data-Only 16.4 21.0 56.0 11.4 17.6 37.0 26.4 31.0 57.3 17.5 22.6 44.2 + Mix-Data 39.9 42.8 61.2 56.4 57.6 74.8 28.3 32.6 59.6 42.1 44.7 58.4

provides a largely non-destructive learning signal that improves structured visual parsing while maintaining broad chart/document understanding.

- B.2. Evaluation of the Effect of Multi-Task Data Mixing

- B.2.1. Effect of Multi-Task Data Mixing on VC-RewardBench

Visual-ERM is trained to support multiple tasks that share partial failure modes but also exhibit task-specific characteristics. To examine whether joint training on mixed data is beneficial or induces negative transfer, we ablate the reward model’s training data composition by comparing a unified model trained on all three data sources with task-specific variants trained on a single data type. Results on VC-RewardBench are reported in Tab. 8.

Mixing chart, table, and SVG data yields the best overall performance. We attribute this to cross-task transfer of error patterns (e.g., recognition and layout behaviors learned from tables can generalize to charts due to overlapping failure modes) leading to a mutually beneficial effect under joint training. In Sec. B.2.2, we further validate this finding by comparing the downstream RL performance of reward models trained on different data sources.

As shown in Tab. 8, reward models trained on single-task data exhibit a strong task bias. For example, Chart-Data-Only performs well on the Chart subtask (𝐹1ℎ = 40.1,𝐹1𝑠 = 42.6), but performs much worse on Table and SVG tasks (e.g., Table 𝐹1ℎ = 17.2, SVG 𝐹1ℎ = 11.3). Similarly, although Table-Data-Only achieves strong results on the Table subtask (𝐹1ℎ = 53.6, 𝑆𝑐 = 68.8), it generalizes poorly to Chart (𝐹1ℎ = 17.1). In contrast, the mixed-data RM (Mix-Data) achieves more balanced and consistently strong performance across all three task categories. It maintains nearly the same performance on Chart (𝐹1ℎ = 39.9 vs. 40.1), while significantly improving on Table and SVG (Table: 𝐹1ℎ = 56.4, SVG: 𝐹1ℎ = 28.3). As a result, it attains the best average performance (AVG: 𝐹1ℎ = 42.1,𝐹1𝑠 = 44.7,𝑆𝑐 = 58.4). These results indicate that multi-task training does not introduce obvious negative transfer; instead, it provides positive transfer by sharing structured understanding and code generation abilities, leading to a more generalizable reward model.

- B.2.2. Effect of Multi-Task Data Mixing on RL

To further validate this phenomenon in a practical setting, we apply RMs trained with different datasets to actual RL training and evaluate the effects in real usage. We select the Table-to-Markdown task for this experiment, and the results are shown in Tab. 9. As shown in Tab. 9, both the Table-only RM and the mixed-data RM lead to stable improvements in the Table-to-Markdown RL scenario. Compared with the baseline, RM (table-data-only) improves TEDS and TEDS-S on both benchmarks and reduces Edit-Dist, resulting in an overall average gain of +1.8. More importantly, the mixed-data RM achieves further improvements over the single-task RM on most

- Table 9: Ablation Study of Multi-Task Data Mixing. We perform RL on the table parsing task using different reward models: one reward model is trained only on table data, while the other reward model is jointly trained on table, chart, and SVG data. TEDS-S denotes TEDS-Structure-Only, and TA represents the table subtask of olmOCRBench. For Edit-Dist, where lower values indicate better performance, we use (100 − Edit-Dist) when computing the Average.

OmniDocBench OmniDocBench-v1.5 olmOCRBench

Model

Avg↑ TEDS↑ TEDS-S↑ Edit-Dist↓ TEDS↑ TEDS-S↑ Edit-Dist↓ TA↑

Baseline Qwen3-VL-8B-Instruct 78.9 83.9 23.2 72.7 77.2 26.9 75.3 76.8 RL with Different Reward Model

+ RM (table-data-only) 79.5 84.4 21.6 74.4 79.4 25.0 79.4 78.6 ∆ +0.6 +0.5 +1.6 +1.7 +2.2 +1.9 4.1 +1.8

###### + RM (mix-data,Visual-ERM) 81.4 86.3 20.7 75.4 80.4 24.2 78.1 79.5 ∆ +2.5 +2.4 +2.5 +2.7 +3.2 +2.7 +2.8 +2.7

metrics. On OmniDocBench, it reaches 𝑇𝐸𝐷𝑆 = 81.4 and 𝑇𝐸𝐷𝑆-𝑆 = 86.3, while further reducing Edit-Dist to 20.7. On olmOCRBench-v1.5, it also achieves higher TEDS/TEDS-S and lower Edit-Dist. Overall, it yields an average improvement of +2.7. Although the mixed RM shows a slight drop on the TA metric compared to the Table-only RM (78.1 vs. 79.4), the overall gains are larger, suggesting that a richer multi-task reward signal provides a more robust optimization direction for RL, improving both final performance and generalization on real Table-to-Markdown tasks.

###### B.3. Ablation on Different Judge Models

VC-RewardBench evaluates a reward model’s ability to identify fine-grained image-to-image discrepancies and vision-to-code reconstruction errors. In addition to objective metrics (e.g., scalar scores, types and severities), VC-RewardBench also includes free-form discrepancy descriptions. Since these descriptions are open-ended and may use different wording to refer to the same underlying issue, an exact string match is insufficient for reliable evaluation.

LLM-assisted matching protocol. To evaluate such free-form outputs in a scalable and consistent manner, we employ an LLM-as-Judge to perform error-level alignment between the reward model’s predicted error list and the ground-truth error list. Specifically, given the predicted errors and the annotated errors, the judge determines for each predicted item whether it (i) correctly matches a ground-truth error, (ii) is a hallucinated/unsupported claim, or (iii) corresponds to an error that exists but is described incorrectly. Symmetrically, the judge also identifies ground-truth errors that are missed by the model. Based on this alignment, we compute two variants of F1: (i) a strict matching criterion (𝐹1ℎ, hard) that requires close semantic and attribute-level agreement, and (ii) a relaxed criterion (𝐹1𝑠, soft) that tolerates minor paraphrases or partial matches, reflecting a more lenient notion of equivalence.

Why judge ablations matter. Using an LLM introduces a potential dependency on the judge’s calibration and reasoning style. However, in VC-RewardBench the judge is only asked to perform relatively constrained matching (rather than open-ended preference ranking), which we hypothesize to be less sensitive to the specific judge model. To validate this, we rerun the same Visual-ERM predictions on VC-RewardBench while varying the judge among several widely-used API models, and report the resulting scores in Tab. 10.

Results and analysis. Tab. 10 shows that the evaluation is highly stable across judges. First, the 𝑆𝑐 is identical for all judges (e.g., Chart: 61.2, Table: 74.8, SVG: 59.6, AVG: 58.4), as expected since it is computed deterministically and does not rely on the LLM matcher. Second, the judge-dependent metrics (𝐹1ℎ and 𝐹1𝑠) vary only modestly: on average, 𝐹1ℎ ranges from 40.8 to 43.0 (a span of 2.2 points) and 𝐹1𝑠 ranges from 42.4 to 45.1 (a span of 2.7 points) across all judges. At the task level, the variability remains limited—for instance, Table 𝐹1𝑠 ranges from 55.7 to 57.8—indicating that different judges largely agree on which predicted errors are correct versus hallucinated or missing.

Overall, these results suggest that VC-RewardBench is not overly sensitive to the choice of LLM judge. Since the judge is used for structured error-to-error matching under well-specified criteria, replacing the judge model

- Table 10: Ablation on LLM Judges. VC-RewardBench uses LLM-assisted judging to verify answer correctness. To assess the robustness of this evaluation protocol, we compare multiple LLM-as-Judge models and report the resulting consistency across judges.

Judger

Chart Table SVG AVG

𝐹1ℎ 𝐹1𝑠 𝑆𝑐 𝐹1ℎ 𝐹1𝑠 𝑆𝑐 𝐹1ℎ 𝐹1𝑠 𝑆𝑐 𝐹1ℎ 𝐹1𝑠 𝑆𝑐

GPT-5-mini 39.9 42.8 61.2 56.4 57.6 74.8 28.3 32.6 59.6 42.1 44.7 58.4 GPT-5.2 40.9 45.4 61.2 55.0 57.1 74.8 24.5 31.6 59.6 40.8 45.1 58.4 Gemini2.5-Pro 40.8 42.1 61.2 54.8 55.7 74.8 27.1 28.7 59.6 41.2 42.4 58.4 Gemini3-Flash 41.5 43.7 61.2 56.5 57.8 74.8 28.5 32.1 59.6 43.0 44.9 58.4

- Table 11: Ablation Study of Reward Design. RSR denotes the Render-Success Reward. TEDS-S denotes TEDS-StructureOnly, and TA represents the table subtask of olmOCRBench. For Edit-Dist, where lower values indicate better performance, we use (100 − Edit-Dist) when computing the Average.

OmniDocBench OmniDocBench-v1.5 olmOCRBench

Model

Avg↑ TEDS↑ TEDS-S↑ Edit-Dist↓ TEDS↑ TEDS-S↑ Edit-Dist↓ TA↑

Qwen3-VL-8B-Instruct 78.9 83.9 23.2 72.7 77.2 26.9 75.3 76.8 GRPO (w/o RSR) 80.9 85.6 21.8 75.2 79.9 25.9 79.0 79.0 GRPO (w RSR) 81.4 86.3 20.7 75.4 80.4 24.2 78.1 79.5

yields consistent conclusions, supporting the robustness of our evaluation protocol.

###### B.4. Ablation Study on Reward Design

We investigate two reward designs. The first directly uses the scalar score produced by the reward model as the reward signal. To improve training stability, the second design augments the reward with an additional render-success reward that indicates whether the generated output can be successfully rendered, which is similar to a format reward. To combine the render-success reward with the reward-model score in a principled manner, we further apply score normalization before aggregation.

We conduct an ablation study comparing these two reward formulations and report the results in Tab. 11. While both designs yield comparable performance gains, the formulation incorporating a render-success reward achieves a slightly larger improvement, which we attribute to its enhanced training stability. Specifically, we observe that during the initial stages of training, the model occasionally produces code that cannot be successfully rendered, leading to intermittent zero rewards. However, the policy rapidly learns to navigate these formatting constraints, allowing the optimization focus to transition quickly from basic renderability to the primary objective of preserving fine-grained visual characteristics. This phased learning trajectory ensures that the model effectively prioritizes structural and semantic fidelity once the fundamental format requirements are mastered.

###### B.5. More Test-Time Scaling Results

In the main paper, we evaluate reflection-based test-time scaling (TTS) on Chart-to-Code. To further validate the effectiveness and generality of this inference-time refinement strategy, we additionally conduct TTS experiments on SVG-to-Code, with results reported in Tab. 12. We report SSIM and CLIPScore (higher is better) and LPIPS (lower is better).

The table contains two independent comparisons. The first compares Qwen3-VL-8B-Instruct with and without Visual-ERM-guided reflection. Reflection yields consistent improvements across all metrics: CLIPScore increases from 73.3 to 74.6, SSIM remains comparable (60.9 vs. 60.4), and LPIPS decreases from 60.0 to 58.1 (improved). Overall, the aggregated score improves from 64.2 to 65.2 (+1.0), demonstrating that Visual-ERM can provide actionable feedback for inference-time refinement even without RL.

The second comparison evaluates TTS on the Visual-ERM-guided RL policy (Qwen3-VL-8B-Instruct + RL

Prompt for Distillation Reward Modeling Datasets (Chart) — Part1

You are an **Experienced Specialist for Data Visualization**. You will be provided with **two images**:

- 1. **Original Image**: a chart rendered using ground-truth Matplotlib code.
- 2. **Generated Image**: a chart rendered using AI-generated Matplotlib code for the **Original Image**. Your task is to **compare the Generated Image against the Original Image** and identify **all visual discrepancies** that affect:

- * correctness of the data,
- * layout / structure, or
- * aesthetic consistency (when it impacts readability or overall fidelity). You are evaluating **only what is visible in the two images**, not the underlying code.

---

- ### Step 1: Compare the two images Carefully compare the Original and Generated images:

- * Look at chart types, subplots, axes, data shapes, colors, legends, labels, ticks, and annotations.
- * Focus on differences that a developer would need to fix in their Matplotlib code. If there is **no meaningful difference** between the two images (ignoring tiny pixel-level differences such as anti-aliasing), then you should report **zero errors**.

---

- ### Step 2: Classify each discrepancy into one category For **every discrepancy you find**, assign it to **exactly one** of the following categories:

- #### A. `structure_error` (Layout & Chart Type)

**Definition:** Problems with the overall structure or composition of the figure.

**Typical cases:**

- 1. Wrong chart type (e.g., pie vs. donut, bar vs. line, radar vs. line).
- 2. Wrong number of subplots (e.g., 1 subplot instead of 2).
- 3. Subplots arranged in wrong rows/columns or in the wrong positions.
- 4. Subplots missing or extra.
- 5. Axes that should be shared/aligned in the Original are not shared/aligned in the Generated (or vice versa).

- #### B. `data_error` (Content & Geometry)

**Definition:** The visualized data itself is incorrect or distorted.

**Typical cases:**

- 1. **Geometry mismatch:** The shape of the curve, polygon, or bar pattern is clearly different (e.g., an increasing line becomes flat or decreasing).
- 2. **Value distortion:** Bar heights, pie slice angles, or point positions do not match the original trend or relative ratios.
- 3. **Scale / limits mismatch:** Different xlim/ylim or axis scales that significantly change how the data appears.
- 4. **Missing / extra data:** Missing series, hallucinated extra lines, wrong number of groups or categories.

- #### C. `text_error` (Labels & Annotations)

**Definition:** Problems with any visible text in the chart. Text includes: titles, subtitles, axis labels, tick labels, legend text, annotations, and text boxes.

**Typical cases:**

- 1. Missing or incorrect titles, axis labels, tick labels, or legend entries.
- 2. Misplaced labels (e.g., attached to the wrong axis, wrong subplot, or wrong data series).
- 3. Text overlapping with other elements in a way that harms readability.
- 4. Obvious typos or formatting differences that clearly reduce readability (e.g., extremely small font where the original is normal).

- #### D. `style_error` (Aesthetics & Appearance)

**Definition:** Visual style differences that may not directly change the data values, but affect the overall look and clarity.

**Typical cases:**

- 1. **Color mismatch:** Different color palette or clearly wrong color mapping for data series / categories.
- 2. **Markers & lines:** Wrong line style (dashed vs. solid), marker shape, or line width.
- 3. **Grid / background:** Missing or extra gridlines, wrong background color, frame visibility changes.
- 4. Other stylistic differences that noticeably change the visual appearance, even if the data is still correct.

---

- ### Step 3: Assign severity for each error For each error, assign a **severity**:

- **1 (Minor)**: Purely aesthetic or very small differences. The chart remains correct and easy to understand (e.g., slightly different shade of color, small font size difference that doesn’t hurt readability).
- **2 (Moderate)**: Affects readability or requires correction for confident use (e.g., missing axis label, legend partially blocking data, noticeably confusing color choices).
- **3 (Critical)**: The chart is **factually wrong or structurally broken**, or seriously misleading (e.g., wrong chart type, missing subplot, reversed trend, wrong scale that changes interpretation, missing key data series).

---

- Figure 7: Prompt for distillation. The prompt used to distill GPT-5-mini to construct reward-modeling training data for Visual-ERM. We show the Chart-specific prompt here; due to its length, this table includes only the first part. 21

Prompt for Distillation Reward Modeling Datasets (Chart) — Part2

### Step 4: Output format (very important) You must output **ONLY** a single valid JSON object, with **no additional text**, **no explanations**, and **no Markdown code fences**.

Use exactly this JSON schema: {

"structure_error_count": int, "data_error_count": int, "text_error_count": int, "style_error_count": int, "errors": [

{

"category": "structure_error | data_error | text_error | style_error", "severity": 1 | 2 | 3, "location": "Specific location (e.g., 'Left subplot title', 'Red line data', 'Legend')", "description": "Concise description of the error."

} ]

}

**Additional constraints:**

- 1. If a category has **no errors**, its `*_error_count` **must be 0**.
- 2. The `errors` list can be **empty** if the Generated Image is visually consistent with the Original Image (ignoring tiny pixel-level differences).
- 3. The numeric counts (`*_error_count`) must exactly match the number of errors you report for each corresponding category in the `errors` list.

- Figure 8: Prompt for distillation. The prompt used to distill GPT-5-mini to construct reward-modeling training data for Visual-ERM. We show the Chart-specific prompt here; due to its length, this table includes only the second part.

Table 12: Test-Time Scaling on SVG-toCode. We evaluate TTS on the SVG-to-Code task. Leveraging Visual-ERM as an evaluator, the model performs multiple rounds of self-reflection and revision, leading to improved parsing performance.

UniSVG(ISVGEN) SSIM↑ LPIPS↓ CLIP↑ Score↑

Model

Qwen3-VL-8B-Instruct 60.9 60.0 73.3 64.2 + Reflection (Visual-ERM) 60.4 58.1 74.6 65.2

Qwen3-VL-8B-Instruct (+RL on Visual-ERM) 64.3 49.7 77.5 69.4 + Reflection (Visual-ERM) 63.9 49.2 78.1 69.8

on Visual-ERM). In this setting, applying reflection further boosts performance: LPIPS stays essentially stable (49.7 vs. 49.2, improved), and CLIPScore increases from 77.5 to 78.1. The overall score rises from 69.4 to 69.8 (+0.4). These results indicate that reflection provides additional gains on top of RL, and the RL-trained policy benefits from a stronger starting point, achieving the best absolute performance in both CLIPScore and overall score.

###### B.6. Scalability of Visual-ERM

Discrepancy Patterns. The primary categories defined in our training are macro-level abstractions that encompass nearly the entire spectrum of errors in visual-to-code tasks. We observe that almost all fine-grained discrepancies, including those not explicitly seen during training, are effectively combinations or specific instances of these core failure modes. By grounding the reward model in these fundamental dimensions, Visual-ERM develops a generalized capability to identify and penalize diverse error patterns regardless of their novelty.

Our data generation strategy further reinforces this adaptability through two synergistic modes: 1) Systematic Coverage (Targeted Edit): We utilize high-capacity models to deliberately perturb structured representations, ensuring that the reward model maps the complete theoretical space of structural and semantic

failures. 2) Empirical Realism (Natural Inference): We sample authentic errors directly from weaker models to align the training distribution with the messy, unpredictable failure modes encountered in practice.

Together, these modes ensure that Visual-ERM does not merely memorize a fixed list of labels but instead masters the underlying logic of visual-to-code equivalence. This allows the model to remain robust and precise when evaluating complex, hybrid, or previously unencountered discrepancies.

Broadening Task Scope. While our current evaluation focuses on structured visuals such as charts, tables, and SVGs, Visual-ERM is designed to capture atomic discrepancy patterns that are fundamentally shared across nearly all vision-to-code domains. We argue that the core challenges in tasks like UI reconstruction or mathematical diagramming are not unique, but are instead composed of the same structural and semantic building blocks mastered by our model. For instance, the spatial layout and color fidelity critical for UI reconstruction are directly mirrored in our evaluation of complex chart and table structures.

In essence, Visual-ERM does not just learn task-specific rules; it learns the universal rules of visual-to-code equivalence. Because these "unseen" tasks rely on the same underlying principles of preserving shape, color, and structure, Visual-ERM can be naturally extended to a wider range of vision-to-code scenarios without requiring fundamental architectural changes.

##### C. Prompts

Visual-ERM relies on several designed prompting templates at multiple stages of the pipeline. Specifically, we use prompts for: (i) distilling GPT-5-mini to construct reward-modeling training data for Visual-ERM, (ii) Visual-ERM inference across different vision-to-code tasks (Chart-to-Code, Table-to-Markdown, and SVG-toCode), and (iii) the LLM-assisted judging protocol used in VC-RewardBench for evaluating subjective, free-form error descriptions. We provide the full prompts in this section for reproducibility.

###### C.1. Distillation Prompt

To train Visual-ERM, we first build a reward-modeling dataset by distilling GPT-5-mini. Each training instance consists of an image pair: an Original Image and a Generated Image rendered from a candidate output (e.g., model-generated code/markdown). The distillation target is a structured comparison: the model must identify and describe all visually meaningful discrepancies between the two images, focusing strictly on what is observable in the rendered results rather than the underlying code.

We construct these image pairs via an automated process that combines (i) error injection using a stronger yet cost-effective model (GPT-5-mini) to synthesize diverse, realistic failure patterns, and (ii) lightweight policy inference to mimic errors produced by smaller models in practice. The resulting pairs, together with a detailed instruction prompt, are provided to GPT-5-mini to generate high-quality supervision signals for reward modeling. Figures 7 and 8 show the Chart-specific distillation prompt (split into two parts due to length).

Prompt overview (Chart). The prompt casts the judge as an “experienced specialist for data visualization” and explicitly defines the evaluation objective: compare the Generated Image against the Original Image and report discrepancies that affect (i) correctness of the data, (ii) layout/structure fidelity, and (iii) visual consistency/readability. Importantly, it instructs the judge to output zero errors if no meaningful difference exists.

Error taxonomy and severity. To ensure consistent and fine-grained supervision, the prompt requires every discovered discrepancy to be assigned to exactly one of four categories: structure_error (layout and chart type), data_error (content/geometry and value distortion), text_error (titles, labels, legends, ticks, and annotations), and style_error (aesthetic and appearance). For each error, the judge also assigns a severity level on a threepoint scale (minor / moderate / critical), which captures how strongly the discrepancy impacts fidelity or interpretation. This taxonomy is designed to reflect common failure modes in chart rendering and to provide training targets that are both interpretable and actionable.

Structured JSON output. Beyond natural-language descriptions, the prompt enforces a strict output format. As shown in Fig. 8, the judge must output only a single valid JSON object following a fixed schema, including: (i) per-category error counts, and (ii) an explicit list of error items, each annotated with category, severity, a

Prompt for Visual-ERM — Chart

PROMPT_CHART2CODE_JUDGEMENT = """\ You are an **Experienced Specialist for Data Visualization**. You will be provided with **two images**:

- 1. **Original Image**: a chart rendered using ground-truth Matplotlib code.
- 2. **Generated Image**: a chart rendered using AI-generated Matplotlib code for the **Original Image**.

Your task is to **compare the Generated Image against the Original Image** and identify **all visual discrepancies**, then summarize them in a strict JSON format.

Additionally, You should assign a **severity score** for each error:

- - 1 (minor): small errors that barely affect readability or understanding.
- - 2 (medium): errors that affect partial understanding and require manual correction for reliable use.
- - 3 (severe): structural or key-content errors that break reliable alignment or can significantly mislead.

You MUST output a single JSON object **ONLY**.

- - Do NOT include any extra text.
- - The JSON schema MUST match exactly: {

"structure_error_count": int, "data_error_count": int, "text_error_count": int, "style_error_count": int, "errors": [

{

"category": "structure_error | data_error | text_error | style_error", "severity": 1 | 2 | 3, "location": "Specific location (e.g., 'Left subplot title', 'Red line data', 'Legend')", "description": "Concise description of the error."

} ]

} """

- Figure 9: Visual-ERM inference prompt. The prompting template used by Visual-ERM at inference time. This prompt is specialized for the Chart-to-Code setting.

localized location description (e.g., “legend”, “left subplot title”), and a concise description. The prompt additionally specifies consistency constraints (e.g., counts must match the number of listed errors; the error list must be empty when the two images are visually consistent). These constraints substantially reduce annotation ambiguity and make the distilled outputs directly usable for reward-model training and evaluation.

Why this design. This prompt design provides two practical benefits. First, it produces fine-grained supervision that decomposes visual discrepancies into interpretable components (structure/data/text/style) rather than collapsing them into a single opaque score. Second, the enforced JSON format enables scalable filtering, aggregation, and auditing of the distilled labels, which is critical for building large reward-modeling datasets with reliable quality control.

###### C.2. Visual-ERM Inference Prompts

After constructing the distillation dataset, we train our reward model, Visual-ERM, to produce fine-grained, structured feedback for image-to-image visual equivalence. Since this behavior is learned through supervised fine-tuning, Visual-ERM does not require an overly restrictive prompt at inference time. Instead, we adopt lightweight and efficient prompting templates that primarily (i) specify the comparison setup (Original vs. Generated), (ii) request exhaustive discrepancy discovery conditioned on the task, and (iii) enforce a strict JSON output schema.

We design a task-specific prompt for each vision-to-code setting—Chart-to-Code, Table-to-Markdown, and

Prompt for Visual-ERM — Table

PROMPT_TABLE2MARKDOWN_JUDGEMENT = """\ You are a table parsing quality auditor. You will be given two table images:

- - Image 1: the original table screenshot (original)
- - Image 2: the parsed / rendered table image (parsed)

Your job is to compare the original vs. parsed images and identify all discrepancies in the parsed table relative to the original, then summarize them in a strict JSON format.

For every discrepancy, assign exactly ONE of the following categories:

- 1) layout_error (structure/layout errors)
- 2) text_error (text recognition errors)
- 3) numeric_error (numeric/symbol/unit errors)

Assign a severity score for each error:

- - 1 (minor): small errors that barely affect readability or understanding
- - 2 (medium): errors that affect partial understanding and require manual correction for reliable use
- - 3 (severe): structural or key-content errors that break reliable alignment or can significantly mislead

You MUST output a single JSON object ONLY.

- - Do NOT include any extra text.
- - Do NOT wrap the JSON in markdown code fences like ```json.
- - The JSON schema MUST match exactly:

{

"layout_error_count": int, "text_error_count": int, "numeric_error_count": int, "errors": [

{

"type": "layout_error | text_error | numeric_error", "description": "A short description of what is wrong and where it is wrong", "severity": 1|2|3

} ]

} """

- Figure 10: Visual-ERM inference prompt. The prompting template used by Visual-ERM at inference time. This prompt is specialized for the Table-to-Markdown setting.

SVG-to-Code. These prompts share the same overall structure and interface, but differ in the error taxonomy and schema fields to reflect domain-specific failure modes (e.g., chart type/axes/legend for charts, layout and numeric/text errors for tables, and shape/style/text-symbol discrepancies for SVGs). The complete prompts are provided in Fig. 9, Fig. 10, and Fig. 11.

Chart-to-Code prompt (Fig. 9). The chart prompt frames Visual-ERM as a data visualization specialist and focuses the comparison on discrepancies that matter for faithful chart reproduction. It explicitly decomposes errors into structure_error, data_error, text_error, and style_error, covering (i) global composition (chart type, subplot arrangement, axes sharing), (ii) geometric/value fidelity (bar heights, line trends, point positions, scale mismatches), (iii) textual correctness (titles, tick labels, legends, annotations), and (iv) appearance-level mismatches that may affect readability. This taxonomy is particularly suitable for charts because visually minor differences (e.g., small font changes) can be less important than semantic distortions (e.g., wrong scale or swapped series), and the explicit severity field further encourages Visual-ERM to surface high-impact errors prominently. In practice, the chart prompt also encourages localized descriptions (e.g., “legend”, “x-axis ticks”, “left subplot line”) which makes the feedback more actionable for downstream debugging and revision (such as test-time scaling).

Prompt for Visual-ERM — SVG

PROMPT_IMG2SVG_JUDGEMENT = """\ You are an expert QA Specialist for Vector Graphics & Icon Generation. You will be provided with two images to compare visually:

- - Image 1 (Ground Truth): The original icon/graphic rendered from correct SVG code.
- - Image 2 (Prediction): An icon/graphic rendered from AI-generated SVG code attempting to reproduce Image 1.

Your job is to compare the original vs. parsed images and identify all discrepancies in the parsed image relative to the original, then summarize them in a strict JSON format.

Assign a severity score for each error:

- - 1 (minor): small errors that barely affect readability or understanding
- - 2 (medium): errors that affect partial understanding and require manual correction for reliable use
- - 3 (severe): structural or key-content errors that break reliable alignment or can significantly mislead

You MUST output a single JSON object ONLY.

- - Do NOT include any extra text.
- - Do NOT wrap the JSON in markdown code fences like ```json.
- - The JSON schema MUST match exactly:

{

"structure_error_count": int, "shape_error_count": int, "style_error_count": int, "text_symbol_error_count": int, "other_error_count": int, "errors": [

{

"category": "structure_error | shape_error | style_error | text_symbol_error | other_error", "severity": 1 | 2 | 3, "location": "Visual location (e.g., 'Train Wheel', 'Eye Contour', 'Canvas Background')", "description": "Concise description of the visual mismatch."

} ]

} """

- Figure 11: Visual-ERM inference prompt. The prompting template used by Visual-ERM at inference time. This prompt is specialized for the SVG-to-Code setting.

Table-to-Markdown prompt (Fig. 10). The table prompt casts Visual-ERM as a table parsing quality auditor and emphasizes structural alignment and textual/numeric accuracy. Compared to charts, tables have less ambiguity in global structure but are highly sensitive to layout (row/column spans, merged cells, alignment) and cell-level content. Accordingly, the prompt uses a concise three-way categorization: layout_error, text_error, and numeric_error. This design reflects typical failure patterns in table parsing: slight layout shifts can cascade into many downstream cell mismatches, while OCR-style errors often manifest as character substitutions or missing tokens. Separating numeric_error from text_error is also important in practice, since numerical discrepancies (units, decimal points, sign errors) are often more harmful even when visually subtle. The enforced JSON schema provides a predictable interface for aggregating errors and computing metrics, while keeping the prompt lightweight enough for efficient inference.

SVG-to-Code prompt (Fig. 11). The SVG prompt targets vector-graphic reconstruction, where fidelity hinges on geometric primitives and styling attributes. To capture these characteristics, the prompt introduces categories beyond those used for raster-like structured visuals: shape_error (incorrect geometry or missing/extraneous components), style_error (stroke width, fill, color, opacity), text_symbol_error (glyphs, labels, icons) and structure_error (global composition and grouping). This taxonomy reflects the fact that SVG failures often involve precise contours and layering, which may not be well described by “data/text/style” alone.

Requiring a location field further encourages spatial grounding, making the output interpretable and enabling targeted correction during RL or test-time scaling.

VC-RewardBench Evaluation Prompt

prompt = f""" You are a strict “error alignment evaluator.” Your task is: compare the error list from PRED with the error list from GT, determine which ones refer to the same specific error point, and output structured JSON.

[Data Category] {category}

[Allowed Error Types (you may only match within these types)] {allowed_types}

[Matching Rules (VERY IMPORTANT)]

- 1) A match is only allowed when pred.type == gt.type OR pred.category == gt.category. Otherwise, it must be judged as not matching.
- 2) A “match” requires that the descriptions point to the same specific error point (same location/object/cell/chart element), and that the error phenomenon is the same or highly consistent.
- 3) If the pred description is more generalized than gt but clearly refers to the same error point, it can be considered match_level="partial"; otherwise it is "no".
- 4) Each pred can match at most one gt, and each gt can match at most one pred (1-to-1 matching).
- 5) Do not match randomly: it is better to leave unmatched than to make an incorrect match.

[Your output MUST be strictly valid JSON, with no extra text] The JSON output format is: {

"matches": [ {

"pred_id": 0, "gt_id": 3, "match_level": "yes" | "partial"

}

], "unmatched_pred": [1, 2], "unmatched_gt": [0, 4], "notes": "Optional, at most one sentence"

}

[PRED Error List] {json.dumps(pred_list, ensure_ascii=False, indent=2)}

[GT Error List] {json.dumps(gt_list, ensure_ascii=False, indent=2)} """.strip()

- Figure 12: VC-RewardBench evaluation prompt. For the subjective, free-form error descriptions in VC-RewardBench, we use an LLM-assisted judge to perform error matching and verification; the prompt used in this evaluation protocol is shown here.

Common design choices across tasks. Across all three prompts, we adopt several shared choices that improve usability and robustness: (i) the judge is instructed to compare only what is visible in the rendered images, preventing reliance on spurious code cues; (ii) the output must be a single valid JSON object with a fixed schema, which reduces post-processing ambiguity and supports scalable training/evaluation; and (iii) the severity score provides a unified notion of error impact, enabling downstream pipelines (e.g., reflection-and-revision) to prioritize the most critical issues first. Together, these prompts serve as a lightweight interface that activates Visual-ERM’s learned capabilities while remaining consistent across heterogeneous vision-to-code domains.

###### C.3. VC-RewardBench Evaluation Prompt

A key component of VC-RewardBench is the evaluation of free-form error descriptions, which summarize the visual mismatches between the original image and the image re-rendered from the parsed output. Since these descriptions are subjective and highly open-ended, rule-based string matching is brittle and fails to account for paraphrases, partial matches, or differences in granularity. We therefore use an LLM-as-Judge to perform verification: it determines which predicted errors are correctly matched to ground-truth errors, which are hallucinated, and which are only partially aligned.

Figure 12 shows the prompt used for this evaluation. The prompt is intentionally designed to be constrained and deterministic: it restricts matching to a predefined set of error types, enforces strict type/category consistency, and requires 1-to-1 alignment between predicted and ground-truth error items. It also explicitly encourages conservative decisions (preferring “unmatched” over uncertain matches) and outputs a single JSON object with match indices and match levels (yes vs. partial). These constraints reduce judge arbitrariness and make the resulting metrics more robust across different judge models (as further validated in Tab. 10).

##### D. Case Study

###### D.1. Data Cases

In this section, we provide representative qualitative examples to illustrate the data format and error annotations used in our pipeline. The reward-model training data and VC-RewardBench examples follow a similar structure—both are built from (Original Image, Re-rendered Image) pairs and accompanied by fine-grained error descriptions. Compared to the training data, VC-RewardBench is labeled by stronger proprietary models and further curated through human filtering, making it more reliable for evaluation. We showcase examples from VC-RewardBench across three domains (Chart, Table, and SVG), as shown in Fig. 13, Fig. 14, and Fig. 15.

Chart cases (Fig. 13). The chart examples highlight typical failure modes where likely textual and structural text are insufficient, and the discrepancies must be judged directly in the rendered space. In the first example, the annotations capture multiple error types: a high-severity data_error indicating mismatched bar magnitudes and/or incorrect 𝑧-axis scaling, a style_error where the color mapping collapses into a single palette (losing per-category distinction), and a text_error that flags unexpected or misplaced labels (e.g., additional axis/title tokens that alter readability). This case demonstrates that visually faithful reconstruction requires jointly preserving data geometry, semantic encodings (color-to-category), and text placement. In the second example, the dominant issues are again data_error: the predicted histogram overlays bars incorrectly and uses an inconsistent axis range, which changes the implied distribution. The annotations emphasize that even when the global chart type appears correct, local numeric scaling and bin/value placement errors can lead to critical semantic distortion.

Table cases (Fig. 14). The table examples illustrate that errors often concentrate on OCR- and layout-related details, where small differences translate into large structural mistakes in the parsed table. In the first example, the annotations include multiple text_error instances on header strings (missing characters, missing units) as well as numeric_error cases where punctuation or symbols change the value semantics (e.g., comma vs. dot in decimals, letter O vs. digit 0). Additionally, a high-severity layout_error captures column swapping, which is particularly harmful because it can propagate to many cell assignments and invalidate the table schema. In the second example, the errors focus on fine-grained text rendering and alignment: header tokens are misspelled, spacing/formatting changes introduce mismatched column names, and significance markers are rendered inconsistently. The layout_error further reflects misalignment of numeric entries across columns, a common structured-visual failure that is hard to detect in the text space but clearly visible after rendering.

SVG cases (Fig. 15). The SVG examples emphasize geometry- and structure-centric discrepancies that arise in vector-graphic reconstruction. In the first case, the annotations include structure_error due to an extra inner border and an altered frame composition, along with shape_error describing changed glyph geometry (stroke thickness and curve profiles) and subtle symbol differences (e.g., rounded vs. rectangular caps). In the second case, the prediction introduces hallucinated protrusions and fails to reproduce continuous elements (e.g., a missing uninterrupted horizontal band), while also exhibiting inconsistent spacing of repeated bars—a combination of structure_error and shape_error. In the third case, the mismatches include

[Figure 166]

[Figure 167]

GT Image Reference Image Annotation：

[{"type":"data_error","severity":3,"description":"Bar heights and z-axis range differ from the reference (e.g., reference shows values up to ~120 for West March, generated peaks at ~90) — data values and scale are inconsistent."}, {"type":"style_error","severity":2,"description":"Reference uses distinct colors per region (North purple, South blue, East green, West gold); generated image renders all bars in a single gold color, losing the region color mapping."}, {"type":"text_error","severity":1,"description":"Generated image includes axis labels 'Month' and 'Region' that are not present in the reference, altering textual layout/labels of the chart."}, {"type":"data_error","severity":3,"description":"The relative magnitudes of the data are completely wrong."}]

[Figure 168]

[Figure 169]

GT Image Reference Image

Annotation：

[{"type":"data_error","severity":3,"description":"The histogram data is incorrectly visualized. The original chart shows frequency counts up to 5 with distinct distribution peaks for different categories. The generated chart displays overlapping bars with a uniform height of 1.0, and the y-axis scale is incorrect (0-1.0 instead of 0-5)."}, {"type":"data_error","severity":3,"description":"In the reactor temperature distribution, the temperatures corresponding to the data and their frequencies are completely incorrect."}, {"type":"data_error","severity":1,"description":"In the reactor temperature distribution, the x-axis range should extend to 1000 rather than 800."}]

29

- Figure 13: Case study. We show representative examples from the Chart-to-Code reward-model training data and the benchmark.

Annotation：

[Figure 170]

[{"type":"text_error","severity":2,"description":" Header cell 'EGS Projec' is missing final 't' (should be 'EGS Project')"}, {"type":"text_error","severity":2,"description":" Header 'Capacity (MW)' is missing the trailing 'e' unit from original 'Capacity (MWe)'"}, {"type":"text_error","severity":2,"description":" Header 'Depth' is missing the unit '(km)' present in the original"}, {"type":"text_error","severity":1,"description":"P roject name 'Soultz (E U)' shows an extra space in '(E U)' vs original '(EU)'"}, {"type":"numeric_error","severity":1,"descriptio n":"Soultz capacity rendered as '1,5' (comma) instead of '1.5' (dot) in Capacity column"}, {"type":"numeric_error","severity":1,"descriptio n":"Desert Peak capacity rendered as '11-5O' (letter O) instead of '11-50' (zero)"}, {"type":"layout_error","severity":3,"description": "Row 'Landau' has Type and Country columns swapped (shows 'Germany (EU)' under Type and 'Commercial' under Country)"}, {"type":"numeric_error","severity":1,"descriptio n":"Cooper Basin depth rendered as 'B.3' (letter B) instead of '4.3'"}]

GT Image

[Figure 171]

Reference Image

[Figure 172]

[Figure 173]

GT Image Reference Image Annotation：

[{"type":"text_error","severity":2,"description":"Column header 'Responses' is misspelled as 'Respones' in the parsed table"}, {"type":"text_error","severity":2,"description":"Column header 'Year h=0' rendered as 'Year h =0' (letter O used instead of digit 0)"},

- {"type":"text_error","severity":1,"description":"Column header 'Year h=1' rendered as 'Yearh=1' (missing space)"},
- {"type":"text_error","severity":2,"description":"Column header 'Year h=4' rendered as 'Year h-4' (equals sign replaced by dash / formatting changed)"}, {"type":"layout_error","severity":3,"description":"Δh Short-term interest rate: OLS and IV numeric cells are misaligned/swapped (Year h=2 shows 1.02 in OLS and 0.45 in IV instead of vice versa) and corresponding standard errors are similarly misaligned"}, {"type":"text_error","severity":1,"description":"Δh Long-term interest rate (OLS, Year h=0) printed as '0.34 **' (two asterisks) in parsed but original shows three asterisks '0.34***'"}]

30

- Figure 14: Case study. We show representative examples from the Table-to-Markdown reward-model training data and the benchmark.

[Figure 174]

[Figure 175]

- [{"type":"structure_error","severity":2,"description":"Prediction introduces an extra thin inner border (double-border) that is not present in the ground truth."}, {"type":"structure_error","severity":2,"description":"Ground truth has a thick black outer bezel/frame; prediction replaces it with a thin outer border altering the frame structure."}, {"type":"shape_error","severity":2,"description":"Glyph geometry differs — letters are bulkier and proportions are changed (strokes/curves not matching ground-truth shapes)."}, {"type":"shape_error","severity":2,"description":"The dot symbol is circular in the ground truth but square in the prediction. Similarly, the hyphen has rounded caps in the ground truth but is rectangular in the prediction."}]

GT Image Reference Image Annotation：

GT Image Reference Image Annotation：

- [{"type":"structure_error","severity":3,"description":"The ground-truth continuous dark horizontal bar across the middle of the strings is missing/broken in the prediction (no uninterrupted band spanning all strings)."},

[Figure 176]

[Figure 177]

- {"type":"structure_error","severity":2,"description":"An extra pointed spike/extrusion appears on the rightmost yellow border that is not present in the ground truth (hallucinated shape)."}, {"type":"shape_error","severity":2,"description":"Widths and spacing of the inner bars are inconsistent and uneven compared to the evenly spaced, uniform-width strings in the ground truth."},
- {"type":"structure_error","severity":3,"description":"The frame is severely malformed with a jagged, protruding shape instead of a smooth continuous curve."}]

[Figure 178]

[Figure 179]

###### GT Image Reference Image Annotation：

[{"type":"structure_error","severity":3,"description":"Ground truth has a distinct large black outer circle placed outside the square; prediction shows a much smaller/partially overlapping black ring that is clipped against the square instead of being a separate, detached circle."}, {"type":"structure_error","severity":3,"description":"Ground truth square has an open/terminated topright segment to accommodate the external dot; prediction renders the square corner closed (no deliberate opening), causing the dot to overlap the corner."}, {"type":"style_error","severity":2,"description":"Blue inner circle is mis-centered and partly clipped/overlapping the square in the prediction; its size/placement relative to the outer ring differs from the ground truth."}]

31

- Figure 15: Case study. We show representative examples from the SVG-to-Code reward-model training data and the benchmark.

[Figure 180]

[Figure 181]

[Figure 182]

Visual-ERM

[Figure 183]

[Figure 184]

Structure_error (-2) The orientation and direction of the axes are different. Data_error (-3) The data series appear to be inverted or swapped. Style_error (-2) The legend is positioned outside the plot area to the right in the Generated image.

GT Image

Pred Image Dino_similarity: 0.975 (very high)

[Figure 185]

[Figure 186]

[Figure 187]

Visual-ERM

[Figure 188]

[Figure 189]

Structure_error (-3) The generated chart uses a fixed 2-column grid layout with an empty bottom-right slot ... Data_error (-3) The relative areas of the blocks are incorrect. Style_error (-2) The color is dark green, whereas in the original it is teal/turquoise

GT Image Pred Image

Dino_similarity: 0.974 (very high)

- Figure 16: Further Case Analysis. We present additional cases to highlight the limitations of current reward models.

incorrect global composition (outer ring size/placement), an incorrect top-right termination detail, and a mis-centered inner circle, reflecting intertwined structure_error and style_error. These cases show that SVG fidelity depends critically on precise spatial relationships, path geometry, and layering, which motivates an image-to-image judge that can reason over fine-grained rendered differences.

Overall, these examples demonstrate that Visual-ERM and VC-RewardBench capture diverse, realistic discrepancy patterns across domains, and that the error annotations provide interpretable and actionable supervision signals for both training and evaluation.

- D.2. Further Case Analysis We provide additional cases to demonstrate the limitations of existing reward models, as illustrated in Fig. 16.

32

