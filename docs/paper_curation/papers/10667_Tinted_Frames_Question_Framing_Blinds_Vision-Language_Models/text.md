# arXiv:2603.19203v2[cs.CV]20Mar2026

## Tinted Frames: Question Framing Blinds Vision-Language Models

Wan-Cyuan Fan1,3, Jiayun Luo†1,3, Declan Kutscher†2, Leonid Sigal1,3, Ritwik Gupta2

- 1 University of British Columbia
- 2 University of California, Berkeley 3 Vector Institute for AI

Abstract. Vision-Language Models (VLMs) have been shown to be blind, often underutilizing their visual inputs even on tasks that require visual reasoning. In this work, we demonstrate that VLMs are selectively blind. They modulate the amount of attention applied to visual inputs based on linguistic framing even when alternative framings demand identical visual reasoning. Using visual attention as a probe, we quantify how framing alters both the amount and distribution of attention over the image. Constrained framings, such as multiple choice and yes/no, induce substantially lower attention to image context compared to open-ended, reduce focus on task-relevant regions, and shift attention towards uninformative tokens. We further demonstrate that this attention misallocation is the principal cause of degraded accuracy and cross-framing inconsistency. Building on this mechanistic insight, we introduce a lightweight prompt-tuning method using learnable tokens that encourages the robust, visually grounded attention patterns observed in open-ended settings, improving visual grounding and improving performance across framings.

### 1 Introduction

Complex, multi-modal reasoning tasks have been the driving force behind contemporary vision-language model (VLM) development. As these models tackle increasingly difficult realworld datasets, it is critical that their reasoning and responses are appropriately grounded in visual evidence. However, despite their impressive performance on simple benchmarks, recent research has revealed that the visual capability of these systems is largely a function of text priors and biases. VLMs are “blind” and exhibit distinct failures in visual grounding, raising fundamental questions about whether they are reasoning over the image or merely leveraging powerful language priors to generate plausible answers.

Recent work characterizes these issues as a problem of visual disengagement and structural bias. Studies [32,40] have shown that VLMs assign little attention to visual tokens, generating responses driven primarily by textual context rather than visual evidence. This lack of attention is not uniformly distributed. Models frequently allocate disproportionately high attention weights to visual attention sink tokens [20,21,29], semantically meaningless background tokens, while also exhibiting severe spatial biases. For instance, artifacts from positional encodings (i.e., RoPE [37]) and causal attention mechanisms can create effective “blind spots” [39,42,51], resulting in the neglect of specific image regions regardless of their semantic importance. However, these analyses have typically been done holistically, averaging observations across heterogeneous benchmarks. This perspective implies that such blindness is a static, inherent flaw of the model architecture. While there is ample evidence that prompting impacts model accuracy [15,25,34], there is little evidence that suggests that visual perception process itself is impacted by this, or that a simple question framing can induce such behavior.

†Equal contribution. Corresponding author wancyuan@cs.ubc.ca. ⋆Project Page

###### 2 Fan, Luo, Kutscher, et al.

10-3

[Figure 1]

10

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

8

6

MCQ: What’s the color of the large chair? A: white, B: black. VLM: B

Y/N Q: Is the large chair white? VLM: No

Open-ended Q: What’s the color of the large chair? VLM: white

Input Image

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

4

2

Y/N Q: Is there one scarf in the image? VLM: No

MCQ: How many scarves in the image? A: 2, B: 1. VLM: A

Open-ended Q: How many scarves in the image? VLM: 1

Input Image

0

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

- Fig. 1: VLM grounding changes as a function of question framing. Attention maps reveal that while the model actively attends to the target object (the chair) during open-ended generation, it exhibits disengagement and misallocation when the same question is posed as a Yes/No or MCQ task. Note that we employ attention rollout rather than averaging attention weights across layers and token. By recursively rolling out attention matrices, we trace information propagation from inputs of early layers to output embeddings. Qwen2.5-VL-7B is used for the visualization. The top 3 tokens with the highest attention are highlighted in red boxes, and the minimum and maximum values of the linear colormap are set to the same value for all images.

In this work, we demonstrate that VLMs are selectively blind. They decide how much to look at an image based on the textual framing of the question—such as open-ended, Yes/No, or multiple-choice—despite different framings requiring the same visual concepts to answer correctly. We study this phenomenon mechanistically, utilizing attention rollout [2] to estimate the visual information propagation and visualize the attention map where the model attends when making output decisions. We hypothesize that alternative framings impact model performance indirectly via deviations in visual attention.

Since open-ended, Yes/No, and multiple-choice question (MCQ) are the three dominant framings used to evaluate VLMs across major benchmarks, this framing-dependent behavior has direct implications for how we assess model capabilities.

We conduct a three-part analysis. First we establish and quantify the impact of framing on the accuracy by introducing cross-framing inconsistency as a diagnostic measure. By posing semantically equivalent questions across all formats, we find that models which correctly answer open-ended questions frequently fail their constrained counterparts, especially for tasks involving object grounding.

Second, we find that framing impacts both amount and distribution of attention. We find that constrained framings trigger a shift in visual attention strategy, reducing overall attention on the image and redirecting attention away from task-relevant regions. Finally, through intervention on attention, we confirm that indeed the impact of the framing on the accuracy is induced by the shift in visual attention.

Armed with these findings, we propose a lightweight prompt-tuning mitigation strategy that learns a small set of soft tokens to realign the visual attention of constrained framings to match the robust patterns of open-ended setting. Our mitigation strategy restores visual grounding and yields consistent improvements across multiple models and benchmarks without modifying model weights.

### 2 Related Work

Vision-language models (VLMs) [1,5,22,30,38,48,50] have rapidly advanced from simple captioning to complex multimodal reasoning, but how do we know they truly understand what they see? Since contemporary VLMs output free-form language, it is difficult to disentangle visual understanding and reasoning from linguistic shortcuts [26]. Answering this question requires looking beyond output accuracy, into whether these models truly ground their reasoning in what they see, and what factors might cause that grounding to break down.

Visual Grounding in VLMs. Visual grounding, the ability to localize textual concepts within an image, has been long-standing goal in computer vision. Classical architectures such

- as object detectors [7,17,33] are explicitly trained to produce spatial localizations. More recently, vision transformers have been shown to develop interpretable attention patterns that correlate with object boundaries [8]. VLMs embed these vision encoders but are trained end-to-end for language generation. Therefore, spatial grounding is an implicit learning task as opposed to a primary objective. Yet, recent works [13, 21] have shown that this implicit grounding is often unreliable. VLMs can produce correct answers while attending to irrelevant regions, suggesting that strong benchmark performance does not guarantee genuine visual understanding. In this work, we establish a mechanistic link between question framing, visual attention, and output quality.

Visual Disengagement and Bias in VLMs. A growing amount of work documents visual shortcomings of VLMs. Studies [40,49] have shown that these models often allocate much lower attention to visual content than to textual ones, potentially generating responses driven by language priors rather than visual evidence. Others [21,29] have identified that models disproportionately attend to semantically meaningless visual tokens when performing visual reasoning tasks, further diluting visual engagement on the area of interest. Beyond attention allocation, systematic spatial biases from rotary position embedding (RoPE), causal attention masks, and data distribution create effective blind spots [39], causing models to neglect certain image regions regardless of semantic importance. These findings paint a picture of visual blindness as a general and static property. In this work, we find that visual disengagement is dynamic and conditional on linguistic framing. VLMs attend to images well under open-ended framings, but do not under alternatives. Therefore, this work reframes existing findings from “the model cannot see” to “the model decides not to see.”

Prompt Sensitivity. While human-in-the-loop evaluation [9] offers a direct measure of model quality based on human preference, it does not scale to systematic probing of specific visual capabilities. Visual capabilities are benchmarked via targeted probes—partially due to convenience of evaluation. An implicit assumption in existing benchmarks is that framing is a neutral container: a model that understands the scene should answer correctly regardless of how the question is asked. But is this assumption warranted? VLMs are known to be sensitive to how questions are phrased [10]. Prior work has documented a range of within-format perturbations: MCQ option ordering effects [31], yes-bias [24], negation bias [3], and paraphrase inconsistency [10]. These studies vary the surface wording while keeping the question format fixed. Framing, by contrast, is a stronger structural shift: it changes the format itself (e.g., from open-ended to Yes/No or MCQ) while preserving the underlying semantic question. This axis of sensitivity has received comparatively little attention. Moreover, existing studies [10,36] primarily measure sensitivity at the output level, through accuracy drops and answer distribution shifts. This works explores the mechanism of how framing reshapes models’ visual processing flow.

F

Q

X A Y

4 Fan, Luo, Kutscher, et al.

Q: Core Semantic Question

F: Question Framing

X: Visual Input A: Visual Attention Y: Final Prediction

- Fig. 2: Illustration of impact of question framing. We hypothesize that question framing influences model predictions through visual attention. Framing alters attention allocation (F→A), which in turn degrades prediction quality (A→Y).
- 3 Hypothesis on Framing-Attention Influence

We illustrate an ideal processing chain for VLMs in Fig. 2. When performing visual question answering, both text and image modalities should impact visual attention, which in turn affects the final prediction. The semantics of the question are often independent of the question framing. Therefore, there should be no latent factor affecting visual attention.

However, we posit that framing directly impacts visual attention (F→A). Predictions follow from visual attention (A→Y), but framing affects silently degrade the quality of this attention. Together, these form a joint pathway (F→A→Y). Therefore, a latent relationship exists between framing and the final prediction (F→Y). For a robust VLM, none of these pathways should exist; their presence reveals that current models rely on shallow, framingdependent heuristics rather than genuine visual understanding.

We investigate the impact and existence of each pathway in turn. Sec. 4 examines the overall effect of framing on predictions (F→Y). Sec. 5 analyzes the pathway through visual attention (F→A→Y). Finally, in Sec. 6, we present a prompt-tuning method that realign the visual attention to restore robust predictions.

- 4 Cross-Framing Inconsistency (F→Y)

Q: Core Semantic Question

F: Question Framing

X: Visual Input A: Visual Attention Y: Final Prediction

Before examining the internal mechanisms behind framing effects, we first ask a simpler question: does question framing affect the model’s final prediction (F→Y in Fig. 2)? To study this, we use open-ended generation as the anchor. Among the three standard evaluation formats, open-ended questioning provides a natural anchor: without candidate options to select from, the model must generate the answer through free-form reasoning, making it less likely to succeed relying on the prior knowledge alone. If a model answers an open-ended question correctly but fails when the same question is reframed as Yes/No or MCQ, this inconsistency is unlikely to stem from a fundamental lack of visual understanding and is more likely driven by the framing itself. We formalize this as cross-framing inconsistency: the rate at which a model fails to maintain the correct answer, as per the open-ended question, under Yes/No and MCQ framings.

Evaluation Protocol. Our protocol is illustrated in Fig. 3 (left). We evaluate on GQA [18], a general VQA benchmark, and SeedBench [23], which contains diverse visual reasoning tasks. For SeedBench, we remove the original multiple-choice options to form open-ended questions. We query the model with these open-ended questions and retain only correctly answered samples. We then use GPT-5.1 to construct semantically equivalent Yes/No reformulations from the correct answer and re-query the model, measuring whether the correct answer is preserved. Thus, the inconsistency rate is calculated based on cases where the open-ended question was answered correctly, but either the Yes/No or MCQ counterpart was incorrect. Details of the rephrasing procedure and evaluation are provided in the supplementary material.

[Figure 18]

30

50

| | | |GQ|A|
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |S|eedBen|ch|
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | |Multi-<br><br>Singl<br><br>Holist|Obj Ground e-Obj Groun<br><br>ic|ing ding|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

MCQ: Which side of the car is the person sitting on? A. Back B. Hood

LVLM: B

25

InconsistencyRate(%)

40

20

Y/NonQ:theIshoodthe personof the car?sitting LVLM: No

30

15

20

Open-ended Q: Which side of the car is the person sitting on?

10

[Figure 19]

Cross-Framing Inconsistency

10

5

VLM

Hood

0

0

Qwen2.5VL 7B

Gemma3 12B

GLM4.1V 9B-Base

LLaVA-OV1.5 8B

Spatial Relation

Instance Interaction

Instances Counting

Instance Attributes

Instance Location

Instance Identity

Scene Understand

- Fig. 3: Tested open-source VLMs have a significant inconsistency rates across framings and task types. (Left) Cross-framing inconsistency is evaluated by reframing questions with an LLM. (Right) With Qwen2.5-VL-7B, inconsistency is up to 26% on GQA and 38% on SeedBench.

Results. As shown in Fig. 3 (right), the results reveal a surprising degree of inconsistency across all tested VLMs. On GQA, Qwen2.5-VL [6], Gemma3 [38], GLM4.1V [48] exhibit more than 15% cross-framing inconsistency, meaning the model fails to preserve its own correct answers under constrained framing for nearly one in six questions. The task-level breakdown on SeedBench using Qwen2.5-VL-7B is particularly revealing:

Finding 1: Tasks requiring object grounding exhibit the highest inconsistency rates, with multiple-object grounding tasks such as spatial relation and counting being the most affected, suggesting that constrained framing is most damaging where visual grounding matters most.

These results confirm the existence of the connection F→Y, establishing that framing alters predictions.

### 5 Impact of Framing on Visual Attention (F→A→Y)

Having confirmed that framing alters predictions (F→Y), we now investigate the specific internal mechanisms that contribute to this. Specifically, we ask whether and how framing reshapes the model’s visual attention (F→A), and whether any such attention shift is a primary driver of prediction failures (A→Y)?

Choice of datasets and models. We conduct our analysis on two benchmarks selected for their spatial annotations, which enable precise mapping of attention to semantic regions. GQA [18] is a general-purpose visual question answering benchmark built upon the Visual Genome dataset, providing dense semantic annotations including bounding boxes for target objects and scene graph representations that capture spatial relationships between visual entities. V∗ [45] is a high-resolution visual grounding benchmark consisting of around 300 carefully curated samples that require fine-grained spatial reasoning in MCQ format, with bounding box annotations for target regions.

To isolate the impact of task framing from variations in question content, we employ a controlled generation approach. For each sample, we generate three distinct framing variants: open-ended, Yes/No, and MCQ, ensuring that the underlying visual reasoning required remains constant while only the output format changes. We utilize GPT-5.1 to rephrase the original samples into these target formats. For GQA, we leverage the ground-truth scene graph and object annotations to prompt the GPT, ensuring that generated Yes/No and MCQ distractors are factually consistent without requiring visual access. Detailed prompt templates and our human verification process are provided in the Supplementary Material. After filtering, we curate a final dataset of 10k unique semantic queries for GQA and the full 300 samples for V∗. With three framing variants per query, this yields 30k samples for GQA and 900 for V∗. We denote the resulting framing-controlled datasets as GQAF and V∗F to distinguish them from the original benchmarks.

###### Visual Energy

###### Box Attention

###### Sink Attention

###### Entropy

- 4%
- 5%
- 6%
- 7%
- 8%
- 9%

6.5

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

20%

40%

%ofVisualAttention

%ofVisualAttention

%ofTotalAttention

6.0

Entropy(bits)

15%

30%

GQAF

| | |
|---|---|
| | |

5.5

V F

10%

20%

5.0

5%

10%

Open-ended Yes/No MCQ

Open-ended Yes/No MCQ

Open-ended Yes/No MCQ

Open-ended Yes/No MCQ

0.6

0.25

Open-Ended

VisualEnergy

BoxAttention

Yes/No

0.20

MCQ

0.4

0.15

0.2

0.10

0.0

0.05

0 5 10 15 20 25

0 5 10 15 20 25

Layer Index

Layer Index

- Fig. 4: Visual energy drops significantly on non-open-ended framings. (Top) There is a significant drop in attention applied to the portion of the image containing the object of interest, and a corresponding increase in attention to sink tokens, for yes/no and MCQ framing. (Bottom) Per-layer values of visual energy and box attention of Qwen2.5-VL-7B on GQAF.

We primarily focus our analysis on Qwen2.5-VL 7B [6], with extended results covering Gemma3 [38] and LLaVA-OneVision1.5 [4] in the supplementary.

Visual attention aggregation. To quantify the visual reliance of VLMs in generation, we employ attention rollout [2] rather than simple attention averaging across layers and tokens. By recursively “rolling out” attention matrices, we trace visual information propagation from input visual tokens to output embeddings, accounting for both direct attention and indirect pathways via residual connections. Crucially, we apply receptive field normalization to preserve causality during attention map aggregation, as required for autoregressive transformers [2].

Formally, let Watt(ℓ) ∈ RN×N be the raw attention matrix at layer ℓ, where rows correspond to query tokens and columns to key tokens. Following [2], we account for residual connections by defining the adjusted attention matrix as A(ℓ) = 0.5Watt(ℓ) + 0.5I, where I denotes the identity matrix.

To address the bias arising from causal masking as discussed in previous work [44], we then apply receptive field normalization. Specifically, we scale the key tokens by A(ℓ) based on their receptive field size S to ensure unbiased probability mass propagation. We then re-normalize the rows to ensure the resulting matrix is row stochastic matrix before the recursive rollout:

R(ℓ) = N(A(ℓ) · diag(S)) · R(ℓ−1) , (1)

where R(0) = I. The operator N(·) performs row-wise normalization, ensuring that each row sums to 1. The final cumulative product R(L) represents a valid stochastic matrix capturing the effective information flow between all token pairs, where L denotes total number of layers for rollout. To quantify visual reliance, we extract the specific sub-matrix connecting the generated output tokens (queries) to the input visual tokens (keys). We define the final Visual Energy by aggregating the probability mass within this sub-matrix; a higher total summation indicates a stronger reliance on visual content during generation.

#### 5.1 Framing Reshapes Visual Attention (F→A)

With the analysis framework built, we examine how question framing affects the model’s visual attention strategy. We characterize this along three dimensions: the overall degree of visual engagement, the spatial allocation of attention relative to task-relevant regions, and the dispersion of attention across the image.

Visual energy and spatial allocation. As shown in Fig. 4 (top), constrained framings consistently exhibit lower overall visual energy compared to open-ended generation across both GQAF and V*F, indicating reduced reliance on visual content. Beyond this overall reduction, the spatial distribution of attention shifts dramatically. As reported in Fig. 4 (top), attention on sink tokens, positions with low semantic relevance identified in prior work [21], increase for both Yes/No and MCQ. In contrast the attention within target area (Box attention) drops from 19% on Open-ended to around 12% on Yes/No and 13% on MCQ, corresponding to a relative drop of 40% on GQAF. This distribution is even more different on V*F which requires higher grounding skill. The relative drop from open-ended to Yes/no or MCQ is roughly 50%. The model does not simply attend less to the image; it actively redirects attention away from task-relevant regions toward unrelated regions. Furthermore, the entropy of the attention distribution increases under constrained framings, indicating that the remaining visual attention becomes more diffuse and less focused on any specific region.

Finding 2: Constrained framings reduce overall visual energy, redirect attention from target regions to unrelated regions, and produce more dispersed attention, confirming that question framing impacts visual attention F→A in profound ways.

Layer-wise analysis Early layers exhibit similar attention patterns across all framings (Fig. 4). The divergence emerges in the middle layers (approximately layers 12–22), which prior work [19] has identified as cross-modal interaction layers, where visual and textual representations are jointly processed. In these layers, both visual energy and bounding box attention drop significantly for Yes/No and MCQ framings compared to open-ended, and this gap persists through the remaining layers.

Decomposing the framing effect. A question prompt consists of two components: the question itself (e.g., “How many dogs are there?” vs. “Is there a dog?”) and appended instructions (e.g., “Answer with Yes or No”). To understand what drives the attention shift, we disentangle these two sources of variation. As illustrated in Fig. 5 (top), we separately vary the question framing while holding instructions fixed, and vary instructions while holding the question fixed. The coefficient of quartile variation (CQV) for both visual energy and bounding box attention, shown in Fig. 5 (bottom), reveals that variation from changing the question framing is ≈ 3 times larger than variation from changing instructions alone.

#### 5.2 Connecting Attention to Prediction (A→Y)

We now test whether this attention distortion directly drives prediction errors (A→Y). The correlations observed in previous sections between framing and attention do not imply a direct link to accuracy: the model may simply find constrained framings easier to resolve and naturally allocate less visual energy to target area attention without any cost to performance. In other words, the drop in visual engagement could be an efficient strategy rather than a harmful one. To distinguish these alternatives, we perform attention steering, directly intervening on the model’s attention maps under constrained framings to restore them toward open-ended levels, and measure whether accuracy recovers.

Attention steering. Ideally, one would copy/intervene the full attention maps from an open-ended forward pass onto a constrained one. In practice, however, question tokens differ across framings, so the attention maps collected have different query dimensions. We therefore intervene on attention properties, magnitude and spatial distribution, using a multiplierbased scheme. We study two complementary interventions that together disentangle the contributions of visual engagement magnitude and spatial allocation:

###### Question Instruction

Visual Energy

Box Attention

How many dogs are there? Is there a dog in the image? How many dogs are there? A: 2, B: 1

Yes/No

74%

Provide a short answer.

MCQ

72%

=0.986

=0.759

Accuracy

(a) Illustration of varying the question framing

70%

| |
|---|

Question Instruction

| |
|---|

=0.986

68%

Please answer yes or no.

=0.943

| |
|---|

Is there a dog in the image? Please answer the question shortly. Return yes or no.

66%

| |
|---|

| |
|---|

100% 125% 150% 175% 200% Multiplier

100% 125% 150% 175% 200% Multiplier

(b) Illustration of varying the instruction (e.g. Yes/No)

V*F GQAF Accuracy Y/N MCQ Y/N MCQ

V*F GQAF Variation VE Box VE Box

Baseline 0.655 0.706 0.815 0.741 VE Intervention 0.682 0.722 0.817 0.743 Box Intervention 0.675 0.735 0.826 0.756

Question 0.146 0.171 0.206 0.109 Instruction 0.054 0.046 0.054 0.029

Fig. 6: (Top) Performance gains on V*F when multiplier increases with Spearman’s rank correlation coefficient listed. (Bottom) Attention steering results on V*F and GQAF.

- Fig. 5: (Top) Illustration of question/instruction variation. (Bottom) Coefficient of quartile variation on VE and Box for varying framing and instruction.

Visual Energy (VE) steering Given a VQA question in different framings, we first compute the ratio of visual energy between open-ended and constrained framings from attention rollout as the multiplier. Then, we perform inference for the constrained framing again: for each head and layer, we scale up the attention weights on all image tokens (for every query token after the image) by this multiplier, while proportionally reducing the attention on non-image tokens to maintain a valid probability distribution. The spatial distribution within image tokens is preserved; only the total visual energy changes, isolating the effect of how much the model attends to the image content.

Box attention steering With the task-relevant image regions identified using the ground truth bounding boxes, we compute a separate multiplier as the ratio of bounding-box attention from the open-ended rollout to that of the constrained framing. During inference for the constrained framing, we scale up attention weights of image tokens within the bounding-box by the computed multiplier while keeping the total visual energy constant. This isolates the complementary question: does where the model attends within the image matter?

Results. We report accuracy after steering in Fig. 6 (bottom). On V*F, which demands fine-grained grounding, both interventions yield clear gains: VE steering improves both Y/N and MCQ by +2.7 and +1.6 pts, while Box steering improves Y/N and MCQ by +2.0 and +2.9 pts. On GQAF, a more general reasoning benchmark, VE steering barely helps (≈ +0.2 pts for both framings), yet Box steering still delivers consistent improvement (≈ +1.3 pts for both). We further visualize performance change via incremental multiplier increases in Fig. 6 (top) on V*F. We show that accuracy improves monotonically as attention is steered closer to open-ended levels, with high Spearman correlations. Together, these results indicate that spatial misallocation, where attention falls within the image, matters more universally than total visual energy, especially for grounding-heavy tasks where precise localization is critical.

Finding 3: Attention steering confirms that changes induced by framing in the attention directly impact model output A→Y: restoring attention under constrained framings recovers accuracy. Spatial allocation (where the model looks) yields universal gains, while visual energy magnitude (how much it looks) primarily benefits groundingheavy tasks, revealing that framing-induced errors stem from qualitatively different attention failures depending on the task.

|VisionLanguageModel<br><br>[Figure 20]|
|---|

[Figure 21]

VisionLanguageModel

What’s the coat pattern of the cat?

Instructionfine-tuningloss

Attentionalignmentloss

Calico

Input image

Is the coat pattern of the cat tabby?

[Figure 22]

Yes

What’s the coat pattern of the cat?

Reframing

What’s the coat pattern of the cat? A. Tabby B. Calico

B

[Figure 23]

Input prompt

Output

Attention Rollout

Calico

Calico

[Figure 24]

[Figure 25]

[Figure 26]

GT Answer

Frozen module Learnable tokens

GT Answers

- Fig. 7: Learning to re-align attention. Given a VQA sample, we reframe the question/answer pairs into alternate framing with a frozen LLM. We append sets of learned tokens to constrained framings at the end of the input sequence. During training, we add an attention alignment objective on the attention rollout to align the attention from open-ended samples with constrained framings.

Takeaways. Taken together, Sections 3–4 establish a three-link pathway from question framing to model failure. First, constrained framings (Y/N, MCQ) can cause prediction errors, with the largest inconsistency happening on grounding-heavy tasks where precise visual localization is essential (Finding 1: F→Y). Second, we further find that constrained framings reduce overall visual energy, redirect engagement away from task-relevant regions (Finding 2: F→A). Third, directly steering visual attention back to open-ended levels recovers accuracy, confirming that spatial allocation matters universally while magnitude primarily benefits grounding tasks (Finding 3: A→Y). Combining these findings clearly shows that framing-induced errors are not superficial, but are induced by the fundamental shift of how visual information is processed by the VLM. In the next section, armed with these insights, we propose a simple mitigation strategy that improves robustness across question framings.

### 6 Attention Realignment via Prompt Tuning

The above observations that constrained framings suppress visual energy and redirect attention away from task-relevant regions, yet steering attention back recovers accuracy, lead to a natural hypothesis regarding mitigation. Specifically, (1) we hypothesize that the open-ended attention pattern, where the model reasons correctly, can serve as a reliable supervisory signal for realigning constrained framings. Further, (2) since the failure originates

- at the prompt level rather than from a fundamental model deficiency, lightweight prompt tuning with a small number of learnable tokens should be able to restore proper attention behavior without modifying any model weights.

Realigning attention via prompt tuning. As illustrated in Fig. 7, our method operates on training triplets constructed from each sample. Given an image and a question with a ground-truth answer, we use Qwen3-32B [5] to rewrite the question into all three framings, including open-ended, yes/no, multiple-choice, along with their corresponding answers. For the yes/no and multiple-choice framings, we append K learnable tokens to the input sequence; the open-ended input remains unchanged. All three framings are fed into a frozen VLM in parallel to produce output logits and attention rollouts.

Once the outputs are obtained, we jointly optimize two objectives. The first one is a standard cross-entropy loss for next-token prediction, applied to the triplets, preserving the original question answering capability. The second is an attention alignment loss that

Visual Energy

###### Box Attention

Cross-Framing Inconsistency

InconsistencyRate(%)

| | |F| | |
|---|---|---|---|---|
| | |GQA| | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

GQA

GQAF

18%

%ofVisualAttention

Baseline

%ofTotalAttention

Ours

20%

30%

16%

10%

14%

20%

0%

Openended

Yes/No MCQ

Openended

Yes/No MCQ

Qwen2.5VL 7B

Gemma3 12B

GLM4.1V 9B-Base

LLaVA-OV1.5 8B

40%

InconsistencyRate(%)

| | | | | | |SeedBench| |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

###### V F

V F

%ofVisualAttention

%ofTotalAttention

30%

8%

30%

20%

6%

20%

10%

4%

0%

Openended

Yes/No MCQ

Openended

Yes/No MCQ

Spatial Relation

Instance Interaction

Instances Counting

Instance Attributes

Instance Location

Instance Identity

Scene Understand

- Fig. 8: (Left) Performance comparison after appending ours learned soft tokens on input prompts on GQAF and V*F. (Right) Cross-framing inconsistency comparison between baseline (Qwen2.5-VL-7B) and ours.

encourages the constrained framings to mimic the open-ended attention pattern. Inspired by the attention distillation objective [14], our alignment loss operates on the attention rollouts and consists of (1) an L2 loss on visual energy magnitude, penalizing the difference in total attention over image tokens; and (2) a KL divergence on the normalized visual attention distributions (summing to one), encouraging spatial alignment. The alignment loss is computed between the open-ended framing and both constrained framings.

Implementation and training details. We fine-tune the VLM on 10K randomly sampled VQA pairs from the LLaVA [27] instruction tuning set4. The model is trained for 1 epoch using the AdamW optimizer [28] with β1 = 0.9 and β2 = 0.999. We employ a peak learning rate of 2E − 4 with a 5% linear warmup, followed by a cosine decay schedule. We use a batch size of 1 with gradient accumulation over 16 steps. We weight each sample by the model’s own confidence in the ground-truth answer under teacher forcing. Specifically, we compute the average probability assigned to each ground-truth token at the corresponding position, yielding a continuous weight. Samples where the model is already confident in the correct answer produce attention patterns that are more likely to be reliable supervisory signals, and thus receive higher weight; samples where the model is uncertain contribute proportionally less. We used K = 8 learned tokens during fine-tuning. We study the effect of the loss components and the number of learnable tokens K in our ablation experiments (see Sec. 6.1).

#### 6.1 Experiments

Models and benchmarks. We evaluate our method on five VLMs spanning diverse architectures: Qwen2.5-VL-7B [6], Qwen3-VL-8B [5], LLaVA-OneVision-1.5-8B [4], Gemma312B [38], and GLM4.1V-9B-Base [48]. For each model, the learnable tokens are trained on 10K randomly sampled examples (with early stopping) from the LLaVA SFT training set, as described above. We evaluate downstream performance on 7 benchmarks grouped into three categories: general and reasoning (RealWorldQA† [46], MME [12], MMMU-Pro‡ [47]), alignment (HallusionBench‡ [16], POPE [24]), and fine-grained grounding (HRBench8k [43], V∗ [45]). We note that MME and POPE are yes/no benchmarks, while the rest are multiplechoice QAs. Since we focus on constrained framings, we note that † denotes the multiple-choice subset, and ‡ denotes the vision split of MMMU-Pro and HallusionBench. Also, we do not incorporate the tools used for VLMs when testing on V∗. In addition, we revisit the GQA, Seedbench, GQAF, and V*F evaluation setup from Sec. 5 to measure whether our method recovers visual attention and reduces cross-framing inconsistencies.

4 llava-1.5-665k-instructions

General & Reasoning Alignment Fine-grained grounding RealWorldQA† MME MMMU-Pro‡ Hallusion‡ POPE HRBench8k V*

Methods

Gemma3-12B 61.87 2133.5 27.92 64.56 84.63 48.25 48.50 + Ours 61.87 2180.0 28.09 66.14 84.98 49.50 49.79

GLM4.1V-9B-Base 69.41 2208.1 32.22 63.72 88.52 51.62 58.80 + Ours 69.41 2237.8 31.41 67.51 87.24 52.25 61.80

LLaVA-OV1.5-8B 70.09 2276.2 30.52 66.04 88.82 58.00 62.23 + Ours 69.63 2290.3 30.92 66.14 89.12 58.50 62.66

- Qwen2.5VL-7B 63.70 2199.5 31.45 71.50 87.68 54.13 66.95 + Ours 66.44 2269.7 31.50 68.77 88.65 54.13 69.53

- Qwen3VL-8B 70.78 2390.8 32.72 71.08 89.26 61.38 72.96 + Ours 71.92 2430.5 32.89 72.34 89.41 61.25 75.11

- Table 1: Performance impact of our method. † denotes the multiple-choice subset, and ‡ denotes the vision split.

Improving visual attention and prediction consistency across framings. Before investigating downstream accuracy, we first verify the effectiveness of our proposed method on restoring the attention for constrained framings. As the results shown in Fig. 8 (Left), after applying our learnable tokens on Yes/No and MCQ, visual energy under constrained framings recovers substantially and bounding box attention increases, indicating that the model redirects its focus back to task-relevant image regions. We also revisit the inconsistency evaluation in Fig. 8 (right). One can see that the cross-framing inconsistency rate drops generally across all VLMs on GQA. On the Seedbench task breakdown using Qwen2.5-VL-7B, we observe significant inconsistency reductions by 20% on instance interaction and 15% on counting categories that demand fine-grained visual grounding. These results confirm that the learnable tokens successfully realign the attention distribution of constrained framings toward the open-ended pattern, motivating the benchmark evaluation that follows.

Improving the performance across benchmarks. We evaluate whether re-aligned attention translates into downstream accuracy gains, as hypothesized in Sec. 3. We append learned tokens to the input sequence according to the framing type of each benchmark (i.e., yes/no or MCQ). In Tab. 1, we report results across 7 benchmarks and 5 VLMs spanning

- 7B to 12B. Overall, our methods yields consistent improvements across most model and benchmark combinations. The gains are most pronounced on fine-grained grounding tasks: on V∗, Qwen2.5-VL-7B improves by 2.5pp, while HRBench8k sees steady improvements for Gemma3 and GLM4.1V. The results resonate the findings that tasks demanding spatial localization benefit most from restored visual attention. General reasoning benchmarks show modest but positive gains overall. The consistency of the improvements models confirms the effectiveness of our methods.

Model ablation. To verify that the performance gains stem from attention alignment rather than additional instruction tuning with cross-entropy loss, we first ablate the loss components on Qwen2.5-VL-7B. The results are shown in Tab. 2. Removing the attention alignment loss yields marginal gains over the baseline, indicating that cross-entropy loss alone is insufficient and barely recovers the performance of the baseline after adding additional learnable tokens; explicit attention realignment is the primary driver of improvement. Furthermore, removing the cross-entropy loss causes significant performance drops, as the additional learnable tokens are initialized randomly and the CE loss is essential to keep the model’s question-answering capability functioning normally. The full model achieves the best results across most metrics.

###### Tokens RWQA† MMMU-Pro‡ POPE V* Avg

###### Ablation RWQA† MMMU-Pro‡ POPE V* Avg

Baseline 63.70 31.45 87.68 66.95 62.45 Ours 66.44 31.50 88.65 69.53 64.03 w/o Attn Loss 64.84 32.89 87.05 65.67 62.61 w/o CE Loss 64.61 31.27 83.40 63.95 60.81

4 63.70 31.33 88.43 66.09 62.39 8 66.44 31.50 88.65 69.53 64.03 12 63.93 31.50 88.36 67.81 62.90 16 63.93 30.92 88.66 67.38 62.72

- Table 2: Ablation on loss functions and learnable token count. RWQA denotes RealWorldQA. † denotes the multiple-choice subset, and ‡ denotes the vision split. 7 Conclusion

In this paper, we establish that visual attention within VLMs is significantly impacted by question framing. Through our mechanistic analysis of the processing pathways in a VLM, we establish a latent relationship between framing and downstream task performance. This work reframes visual blindness as a dynamic behavior controllable by a user rather than a static architectural limitation. With this results, we implement a prompt-tuning method that realigns attention under constrained framings, yielding consistent improvement across models and benchmarks without modifying models’ weights.

### Acknowledgements

This work was funded, in part, by the Vector Institute for AI, Canada CIFAR AI Chairs, NSERC Canada Research Chair (CRC), AML-TN UBC, and NSERC Discovery and Discovery Accelerator Supplement Grants. Resources used in preparing this research were provided, in part, by the Province of Ontario, the Government of Canada through CIFAR, the Digital Research Alliance of Canada5, companies6 sponsoring the Vector Institute, and Advanced Research Computing at the University of British Columbia. Additional hardware support was provided by John R. Evans Leaders Fund CFI grant and Compute Canada under the Resource Allocation Competition award. Ritwik Gupta and Declan Kutscher were supported in part by funding from the Department of Defense, The House Fund, and BAIR’s industrial alliance programs. Additional compute was provided by the Department of Defense’s High Performance Computing Modernization Program. We are immensely grateful to Bicheng Xu from UBC and Stephanie Fu from UCB for sharing their valuable suggestions in paper writing and experiments.

- 5 alliancecan.ca
- 6 https://vectorinstitute.ai/#partners

### References

- 1. Abdin, M., Aneja, J., Behl, H., Bubeck, S., Eldan, R., Gunasekar, S., Harrison, M., Hewett, R.J., Javaheripi, M., Kauffmann, P., et al.: Phi-4 technical report. arXiv preprint arXiv:2412.08905

(2024)

- 2. Abnar, S., Zuidema, W.: Quantifying attention flow in transformers. arXiv preprint arXiv:2005.00928 (2020)
- 3. Alhamoud, K., Alshammari, S., Tian, Y., Li, G., Torr, P.H., Kim, Y., Ghassemi, M.: Visionlanguage models do not understand negation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 29612–29622 (2025)
- 4. An, X., Xie, Y., Yang, K., Zhang, W., Zhao, X., Cheng, Z., Wang, Y., Xu, S., Chen, C., Zhu, D., et al.: Llava-onevision-1.5: Fully open framework for democratized multimodal training. arXiv preprint arXiv:2509.23661 (2025)
- 5. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631 (2025)
- 6. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al.: Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923 (2025)
- 7. Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., Zagoruyko, S.: End-to-end object detection with transformers. In: European conference on computer vision. pp. 213–229. Springer

(2020)

- 8. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 9650–9660 (2021)
- 9. Chiang, W.L., Zheng, L., Sheng, Y., Angelopoulos, A.N., Li, T., Li, D., Zhu, B., Zhang, H., Jordan, M., Gonzalez, J.E., et al.: Chatbot arena: An open platform for evaluating llms by human preference. In: Forty-first International Conference on Machine Learning (2024)
- 10. Chou, S.H., Chandhok, S., Little, J., Sigal, L.: Mm-r3: On (in-) consistency of vision-language models (vlms). In: Findings of the Association for Computational Linguistics: ACL 2025. pp. 4762–4788 (2025)
- 11. Duan, H., Yang, J., Qiao, Y., Fang, X., Chen, L., Liu, Y., Dong, X., Zang, Y., Zhang, P., Wang, J., et al.: Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In: Proceedings of the 32nd ACM International Conference on Multimedia. pp. 11198–11201 (2024)
- 12. Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., et al.: Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394 (2023)
- 13. Fu, S., Bonnen, T., Guillory, D., Darrell, T.: Hidden in plain sight: Vlms overlook their visual representations. arXiv preprint arXiv:2506.08008 (2025)
- 14. Fuller, A., Yassin, Y., Wen, J., Kyrollos, D.G., Ibrahim, T., Green, J.R., Shelhamer, E.: Lookwhere? efficient visual recognition by learning where to look and what to see from selfsupervision. arXiv preprint arXiv:2505.18051 (2025)
- 15. Gu, J., Han, Z., Chen, S., Beirami, A., He, B., Zhang, G., Liao, R., Qin, Y., Tresp, V., Torr, P.: A systematic survey of prompt engineering on vision-language foundation models. arXiv preprint arXiv:2307.12980 (2023)
- 16. Guan, T., Liu, F., Wu, X., Xian, R., Li, Z., Liu, X., Wang, X., Chen, L., Huang, F., Yacoob, Y., et al.: Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 14375–14385 (2024)
- 17. He, K., Gkioxari, G., Dollár, P., Girshick, R.: Mask r-cnn. In: Proceedings of the IEEE international conference on computer vision. pp. 2961–2969 (2017)
- 18. Hudson, D.A., Manning, C.D.: Gqa: A new dataset for real-world visual reasoning and compositional question answering. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 6700–6709 (2019)
- 19. Jiang, Z., Chen, J., Zhu, B., Luo, T., Shen, Y., Yang, X.: Devils in middle layers of large visionlanguage models: Interpreting, detecting and mitigating object hallucinations via attention lens. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 25004–25014 (2025)
- 20. Kaduri, O., Bagon, S., Dekel, T.: What’s in the image? a deep-dive into the vision of vision language models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 14549–14558 (2025)

- 21. Kang, S., Kim, J., Kim, J., Hwang, S.J.: See what you are told: Visual attention sink in large multimodal models. arXiv preprint arXiv:2503.03321 (2025)
- 22. Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Zhang, P., Li, Y., Liu, Z., et al.: Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326 (2024)
- 23. Li, B., Wang, R., Wang, G., Ge, Y., Ge, Y., Shan, Y.: Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125 (2023)
- 24. Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W.X., Wen, J.R.: Evaluating object hallucination in large vision-language models. In: Proceedings of the 2023 conference on empirical methods in natural language processing. pp. 292–305 (2023)
- 25. Liang, H., Huang, R., Du, Y., Hu, Y., Su, W., Snoek, C.G.: Prompt-robust vision-language models via meta-finetuning. In: The Fourteenth International Conference on Learning Representations

(2026)

- 26. Lin, Z., Chen, X., Pathak, D., Zhang, P., Ramanan, D.: Revisiting the role of language priors in vision-language models. arXiv preprint arXiv:2306.01879 (2023)
- 27. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. Advances in neural information processing systems 36, 34892–34916 (2023)
- 28. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101

(2017)

- 29. Luo, J., Fan, W.C., Wang, L., He, X., Rahman, T., Abolmaesumi, P., Sigal, L.: To sink or not to sink: Visual information pathways in large vision-language models. arXiv preprint arXiv:2510.08510 (2025)
- 30. Meta, A.: Llama 3.2: Revolutionizing edge ai and vision with open, customizable models. Meta AI Blog. Retrieved December 20, 2024 (2024)
- 31. Pezeshkpour, P., Hruschka, E.: Large language models sensitivity to the order of options in multiple-choice questions. In: Findings of the Association for Computational Linguistics: NAACL

2024. pp. 2006–2017 (2024)

- 32. Rahmanzadehgervi, P., Bolton, L., Taesiri, M.R., Nguyen, A.T.: Vision language models are blind. In: Proceedings of the Asian Conference on Computer Vision. pp. 18–34 (2024)
- 33. Ren, S., He, K., Girshick, R., Sun, J.: Faster r-cnn: Towards real-time object detection with region proposal networks. Advances in neural information processing systems 28 (2015)
- 34. Schmalfuss, J., Chang, N., VS, V., Shen, M., Bruhn, A., Alvarez, J.M.: Parc: A quantitative framework uncovering the symmetries within vision language models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 25081–25091 (2025)
- 35. Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., Dao, T.: Flashattention-3: Fast and accurate attention with asynchrony and low-precision. Advances in Neural Information Processing Systems 37, 68658–68685 (2024)
- 36. Shah, M., Balaji, S., Sarkhel, S., Dey, S., Venugopal, D.: Analyzing the sensitivity of vision language models in visual question answering. In: Proceedings of the Fourth Workshop on Generation, Evaluation and Metrics (GEM2). pp. 431–438 (2025)
- 37. Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., Roformer, Y.L.: Enhanced transformer with rotary position embedding., 2021. DOI: https://doi. org/10.1016/j. neucom (2023)
- 38. Team, G., Kamath, A., Ferret, J., Pathak, S., Vieillard, N., Merhej, R., Perrin, S., Matejovicova, T., Ramé, A., Rivière, M., et al.: Gemma 3 technical report. arXiv preprint arXiv:2503.19786

(2025)

- 39. Tian, X., Zou, S., Yang, Z., Zhang, J.: Identifying and mitigating position bias of multi-image vision-language models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 10599–10609 (2025)
- 40. Tong, P., Brown, E., Wu, P., Woo, S., Iyer, A.J.V., Akula, S.C., Yang, S., Yang, J., Middepogu, M., Wang, Z., et al.: Cambrian-1: A fully open, vision-centric exploration of multimodal llms. Advances in Neural Information Processing Systems 37, 87310–87356 (2024)
- 41. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems 30

(2017)

- 42. Wang, C., Guo, J., Li, H., Tian, Y., Nie, Y., Xu, C., Han, K.: Circle-rope: Cone-like decoupled rotary positional embedding for large vision-language models. arXiv preprint arXiv:2505.16416

(2025)

- 43. Wang, W., Ding, L., Zeng, M., Zhou, X., Shen, L., Luo, Y., Yu, W., Tao, D.: Divide, conquer and combine: A training-free framework for high-resolution image perception in multimodal large language models. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 39, pp. 7907–7915 (2025)

- 44. Wang, Z., Zhang, H., Li, X., Huang, K.H., Han, C., Ji, S., Kakade, S.M., Peng, H., Ji, H.: Eliminating position bias of language models: A mechanistic approach. arXiv preprint arXiv:2407.01100

(2024)

- 45. Wu, P., Xie, S.: V*: Guided visual search as a core mechanism in multimodal llms. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13084–13094

(2024)

- 46. xAI: Grok-1.5 vision preview (April 2024), https://x.ai/news/grok-1.5v
- 47. Yue, X., Zheng, T., Ni, Y., Wang, Y., Zhang, K., Tong, S., Sun, Y., Yu, B., Zhang, G., Sun, H., et al.: Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 15134–15186 (2025)
- 48. Zeng, A., Lv, X., Zheng, Q., Hou, Z., Chen, B., Xie, C., Wang, C., Yin, D., Zeng, H., Zhang, J., et al.: Glm-4.5: Agentic, reasoning, and coding (arc) foundation models. arXiv preprint arXiv:2508.06471 (2025)
- 49. Zhang, X., Shen, C., Yuan, X., Yan, S., Xie, L., Wang, W., Gu, C., Tang, H., Ye, J.: From redundancy to relevance: Enhancing explainability in multimodal large language models. arXiv preprint arXiv:2406.06579 17 (2024)
- 50. Zhu, J., Wang, W., Chen, Z., Liu, Z., Ye, S., Gu, L., Tian, H., Duan, Y., Su, W., Shao, J., et al.: Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479 (2025)
- 51. Zhu, Y., Bai, X., Chen, K., Xiang, Y., Guan, W., Yu, J., Zhang, M.: From bias to balance: Exploring and mitigating spatial bias in lvlms. arXiv preprint arXiv:2509.21984 (2025)

Abstract. This supplementary provides additional implementation details, quantitative results, and qualitative analyses supporting the main paper. Specifically, Sec. A covers implementation details, including the cross-framing inconsistency pipeline and GPT prompts for question reframing (Sec. A.1), curation details and human evaluation of GQAF and V∗F (Sec. A.2), training and resource usage (Secs. A.3 and A.4), and evaluation protocols for all benchmarks (Sec. A.5). Sec. B presents additional quantitative results, including ablation studies on learnable token placement and confidence-based loss weighting (Sec. B.1), and extended visual attention analysis across additional VLM families, Gemma3, GLM-4.1V, LLaVA-OneVision-1.5, and Qwen3VL (Sec. B.2). Sec. C provides additional qualitative results, and Sec. D discusses limitations and future directions.

### A Implementation Details

#### A.1 Cross-Framing Inconsistency

As discussed in Sec. 4, we leverage cross-framing inconsistency to verify whether question framing affects the model’s final prediction (F→Y in Fig. 2), and the evaluation pipeline is illustrated in Fig. 3 (left). Here, we discuss more implementation details. For data curation, SeedBench [23] is a MCQ benchmark, so we remove the possible answer choices to obtain the open-ended question and use the content of the correct option as the ground truth answer. GQA [18] is largely open-ended, but a small portion of it consists of Yes/No questions. We filter out the Yes/No questions in GQA to obtain a fully open-ended generation benchmark. We then perform inference on both datasets across different VLMs and evaluate the performance. For samples which were answered correctly, we use GPT-5.1 (gpt-5.1-2025-11-13) to reframe the questions into Yes/No or MCQ (while for SeedBench we use the original options), and perform inference on the questions in different framings to get the final inconsistency rate. The detailed system prompt for GPT-5.1 is provided below. We study the quality of question reframing using GPT in the following data curation section.

Prompt for Reframing Open-ended Question to MCQ/YN questions Given the following question and its correct answer, create TWO new framings:

Original Question: {original_question} Correct Answer: {correct_answer} [If options exist:] Available Options: {options}

##### 1. Yes/No:

- – [If affirm:] Convert to a binary yes/no question where the answer is ‘yes’ (affirming that the answer is {correct_answer}).
- – [If negate:] Convert to a binary yes/no question where the answer is ‘no’. Replace the correct answer ({correct_answer}) with a distinctly incorrect or unlikely alternative that is clearly contradicted by the original answer. Avoid any ambiguous or ‘near-miss’ alternatives. Keep the same target object that the original question refers to if possible do not change the subject or object being asked about.
- – Turn the original question into a yes/no format that tests the same knowledge. For example, if the original question is “What is the person near the garbage bin wearing?” with answer “a coat”, the yes/no question could be “Is the person near the garbage bin wearing a coat?”
- – Must be a binary question (Is/Are/Does/Do/Can/Could/etc.)

##### 2. MCQ (Multiple Choice Question):

- – [If options exist:] Create an MCQ version with the same options: {options}. The correct answer should be {correct_answer}.
- – [If no options:] Create an MCQ version with 4 options. The correct answer should be {correct_answer}, and provide 3 easy negative distractors.

[If scene graph available:] Avoid using objects from the scene graph as negative options as they may create ambiguity. Scene graph: {scene_graph}.

- – Should be same open-ended question (What/Which/Where/Who/How/etc.)
- – Provide exactly 4 options as a list

##### IMPORTANT RULES:

- – Yes/No question must test the SAME knowledge as the original
- – MCQ must test the SAME knowledge as the original
- – Both should be answerable from the same visual information

Output ONLY valid JSON in this exact format: {

"yes_no": { "question": "Is/Are/Does/Do... question", "answer": "{yes|no}"

}, "mcq": {

"question": "What/Which/Where... question", "options": ["option1", "option2", "option3", "option4"], "answer_text": "{correct_answer}"

} }

##### Prompt for Reframing YN Question to Open-ended/MCQ questions Given the following yes/no question and answer, create TWO new framings:

Original Question: {original_question} Correct Answer: {original_answer} Correct Full Answer (for reference): {original_full_answer}

##### 1. MCQ (Multiple Choice Question):

- – Convert to a WH-question (What/Which/Where/Who/How/etc.) that asks for identification
- – Provide 4 plausible answer options as a list (the correct answer should be one of them and the answer should not be yes/no)
- – Example: “What color is the helmet?” with options [“light blue”, “red”, “yellow”, “green”]
- – [If scene graph available:] Avoid using objects from the scene graph as negative options as they may create ambiguity. Scene graph: {scene_graph}.

##### 2. Open-ended (short answer):

- – Convert to a WH-question (What/Which/Where/Who/How/etc.) expecting a brief specific answer
- – Should ask the same thing as MCQ but without providing options
- – Example: “What color is the helmet in the middle?” → Answer: “light blue”

##### IMPORTANT RULES:

- – MCQ and Open-ended should use WH-questions (What/Which/Where/Who/How/When/Why) and the answer should not be yes/no
- – Both formats should test the SAME knowledge as the original question

Output ONLY valid JSON in this exact format: {

"mcq": { "question": "What/Which/Where... question", "options": ["option1", "option2", "option3", "option4"], "answer_text": "<correct answer>"

}, "open_ended": {

[Figure 27]

[Figure 28]

Input Image and question Human Evaluation

Input Image and question Human Evaluation

[Figure 29]

[Figure 30]

Y/N Q: Is the person sitting on the hood of the car?

Open-ended Q: Which side of the car is the person sitting on?

Correctness Spatial Consistency

Correctness

Synthetic Answer: Yes

Synthetic Answer: Hood

[Figure 31]

[Figure 32]

MCQ: Which side of the car is the person sitting on? A. Back B. Hood

MCQ: Which side of the car is the person sitting on? A. Back B. Hood

Correctness Exclusivity

Correctness Exclusivity

Y/N Q: Is the person sitting on the hood of the car?

Open-ended Q: Which side of the car is the person sitting on?

GT Answer: Synthetic Answer: B Yes

GT Answer: Synthetic Answer: B Hood

Fig. A1: Human evaluation pipeline for open-ended and Yes/No question reframing. Note that for MCQ question reframing, we can use a rule-based approach (simply remove the options) to convert it into open-ended and then perform reframing to Yes/No.

"question": "What/Which/Where... question", "answer": "<correct answer>"

} }

#### A.2 Curation Details of GQAF, and V*F

As mentioned in Sec. 5, to isolate the impact of task framing from variations in question content, we curate two datasets where each sample has three distinct framing variants: openended, Yes/No, and MCQ. We ensure that the underlying visual reasoning required remains constant while only the output format changes. Again, we utilize GPT-5.1 to rephrase the original samples into these target formats. Now, we discuss how we curate datasets including GQAF and V*F in detail and use human evaluation to further verify their quality.

GQA contains both open-ended and yes/no questions. We leverage the ground-truth scene graph when prompting GPT to ensure that generated MCQ distractors are reasonable without requiring visual access. We provide the prompt that we used for reframing.

For V*, a multiple-choice benchmark, we remove the answer choices to obtain a purely open-ended question, and then follow the same protocol as used in GQA to reframe it into a Yes/No question.

After prompting and reframing, we filter out samples not in the correct JSON format. We curate a final dataset of 10k unique semantic queries for GQA and the full 300 samples for V∗. With three framing variants per query, we obtain 30k and 900 samples for GQA and V∗, respectively.

Human evaluation To verify the quality of the reframed questions and answers, we conduct a human evaluation, with results shown in Sec. A.2. Recall that the dataset reframing process covers three settings: (1) open-ended → MCQ and Yes/No, (2) Yes/No → open-ended and MCQ, and (3) MCQ → open-ended and Yes/No. Since an MCQ without its answer options is naturally open-ended, we handle this direction via a simple rule-based method that strips the options from the question. The resulting open-ended question can then be further converted to Yes/No using the same pipeline. This reduces the number of conversion types that require evaluation to four: open-ended → Yes/No, open-ended → MCQ, Yes/No → open-ended, and Yes/No → MCQ.

We now describe the human evaluation criteria for each of the four conversions. The full human evaluation interface is shown in Fig. A1.

Open-ended → Yes/No. Annotators assess both answer correctness and spatial consistency—whether the target object or region referred to in the question remains the same after reframing. For instance, if an open-ended question asks “What color is the cake?” but the reframed Yes/No question asks “Is the candy purple?”, the target object has shifted, which violates our requirement that the core concept of the question be preserved across formats. Open-ended → MCQ. Annotators assess answer correctness and additionally evaluate option exclusivity—whether the designated correct option is unambiguously the best choice

Open-ended Yes/No MCQ

From ↓ / To →

Correctness Correctness Spatial Cons. Correctness Exclusivity

Open-ended — 96.5 96.9 94.8 95.7 Yes/No 92.2 — — 94.8 93.7 MCQ rule-based 92.6 95.4 — —

- Table A1: Human evaluation results for question reframing across different conversion types. MCQ

→ Open-ended is handled via rule-based conversion and requires no human evaluation.

among all provided options. This is necessary because an LLM may inadvertently generate multiple plausible correct answers, introducing ambiguity. Yes/No → Open-ended and Yes/No → MCQ. For both conversions, annotators assess answer correctness. For MCQ, we also evaluate option exclusivity as aforementioned.

The paired human evaluation results are shown in Sec. A.2. We collect 100 samples with corresponding human feedback for each pair (500 samples in total). We use the CloudResearch platform for human studies, with a total of 197 human evaluators involved, coming from 6 countries, including USA, UK, Ireland, Australia, New Zealand, and Canada, whose native language is English. Across all conversion directions, correctness scores remain consistently high, demonstrating that the reframing process is robust. Beyond correctness, we evaluate two format-specific properties: spatial consistency, which measures whether target object or task-relevant region remains unchanged after reframing even the generated answer is correct, and exclusivity, which measures whether the correct answer in a reframed MCQ question remains unambiguously correct (i.e., the correct option is the best valid option be considered). Both properties are well-maintained, with spatial consistency scores of 96.9% and 95.4% and exclusivity scores of 93.7% and 95.7%.

#### A.3 Training

Following the discussion in Sec. 6, we fine-tune the VLMs on 10K randomly sampled VQA pairs from the LLaVA [27] instruction tuning set. The model is trained for 1 epoch using the AdamW optimizer [28] with β1 = 0.9 and β2 = 0.999. We employ a peak learning rate of 2 × 10−4 with a 5% linear warmup, followed by a cosine decay schedule. We use a batch size of 1 with gradient accumulation over 16 steps. All models, except for GLM, are trained on a single NVIDIA L40. GLM is trained on a single 80GB H100-SXM.

We weight each sample by the model’s own confidence in the ground-truth answer under teacher forcing [41]. Specifically, we compute the average probability assigned to each groundtruth token at the corresponding position, yielding a continuous weight. Samples where the model is already confident in the correct answer produce attention patterns that are more likely to be reliable supervisory signals, and thus receive higher weight; samples where the model is uncertain contribute proportionally less. During training, we jointly apply cross-entropy loss and attention alignment loss. Since the ranges of attention alignment from both mass and distribution are relatively small, we scale the attention loss by a factor of 5 to match the range of the cross-entropy loss. During training, the maximum image size is limited to 728 (longest side) for efficiency while the aspect ratio remains unchanged. For the reframing module in the training pipeline, we use Qwen3-32B locally. Gradient checkpointing is disabled for all models since the gradient backpropagation on the attention map may be incorrect when the gradient checkpointing is on in PyTorch. Training is done with bf16 and FlashAttention [35]. We use K = 8 learned tokens during fine-tuning.

#### A.4 Resource Usage

We report the training cost and parameter analysis of our methods in Tab. A2. We are only training the learnable tokens for each framing. With K = 8, the trainable parameters

are relatively low at around 60K. We train all models using one epoch with early-stopping applied, and the training time cost from smallest model at around three hours to largest model to six hours on one single GPU (L40 or H100). VRAM usage is high as the gradient graph are stored backpropogation back to input embedding space.

Trainable Parameters

Time (min)

VRAM (peak)

VIRT (peak)

Qwen2.5VL-7B 57 K 187 33.8 GB 53.1 GB Qwen3VL-8B 66 K 190 42.7 GB 65.9 GB Gemma3-12B 61 K 380 39.6 GB 59.7 GB GLM-4.1V-8B-Base 66 K 332 76.9 GB 116.1 GB LLaVA-OneVision-1.5-8B 66 K 259 36.5 GB 54.6 GB

- Table A2: Resource usage statistics for prompt tuning. Tested on 1 NVIDIA L40 with batch size 1 and gradient accumulation 16. VIRT stands for virtual memory size. GLM-4.1V-8B-Base is tested on 1 NVIDIA H100.

#### A.5 Evaluations

GQA [18] is a general visual question answering benchmark containing open-ended and yes/no questions. When evaluating the score, we follow the official script and perform simple string matching for both yes/no and open-ended questions.

SeedBench [23] is a comprehensive benchmark containing video question answering, singleimage, and multi-image question answering tasks. In this paper, we focus on the single-image task and subsample the single-image portion accordingly. For evaluation, SeedBench is MCQ-type, and we follow the official setting using an answer ranking strategy to obtain the final predicted option. Specifically, we compute the likelihood of each option instead of appending an instruction at the end of the question asking the model to return the letter of the correct option. This evaluation method disentangles the model’s instruction-following capability from its ability to answer the given questions.

RealWorldQA [46] is released by xAI alongside Grok-1.5 Vision and evaluates basic realworld spatial understanding. Each question is appended with options and an instruction for controlling the output format. RealWorldQA is a mixture of MCQ and open-ended questions. As we are testing the improvement of our soft tokens on yes/no and MCQ questions while leaving open-ended question inference untouched, we use only the MCQ portion of the benchmark. We directly use the official questions as input and apply string matching between the ground-truth letter and the predicted letter.

MME [12] is a VQA benchmark in yes/no question format designed to test the hallucination of LVLMs. We follow VLMEvalKit [11] and append a short instruction “please return yes or no.” to constrain the output of VLMs, then perform string matching for the final performance score.

MMMU-Pro [47] is an enhanced benchmark for VLMs designed to assess true understanding capabilities across multiple modalities. This benchmark provides VQA samples in three formats: standard textual questions with 4 options, standard textual questions with 10 options, and a purely vision-based format where the question and image are presented together in a screenshot. We evaluate our models on the vision split of the benchmark, where the entire question is posed in image format. Since the entire benchmark is MCQ-framing, we append a short instruction following VLMEvalKit to enforce the output to be the letter of the correct option, and perform string matching for the final performance.

General & Reasoning Alignment Fine-grained grounding RealWorldQA† MME MMMU-Pro‡ Hallusion‡ POPE HRBench8k V*

Position

Prefix 61.64 2272.2 28.15 65.51 88.28 51.62 66.52 Infix 66.44 2269.7 31.50 68.77 88.65 54.13 69.53 Postfix 63.47 2262.4 31.50 65.83 88.57 54.25 67.38

- Table A3: Ablation study on learnable token positioning. † denotes the multiple-choice subset, and ‡ denotes the vision split.

Strategy

General & Reasoning Alignment Fine-grained grounding RealWorldQA† MME MMMU-Pro‡ Hallusion‡ POPE HRBench8k V*

Baseline 63.70 2199.5 31.45 71.50 87.68 54.13 66.95 Confidence weighting 66.44 2269.7 31.50 68.77 88.65 54.13 69.53 Equal weighting 64.84 2205.6 30.25 65.51 89.09 52.50 68.67

- Table A4: Ablation study on learnable token weighting strategies. † denotes the multiple-choice subset, and ‡ denotes the vision split.

HallusionBench [16] is, similar to MME, a VQA benchmark in yes/no question format. This benchmark contains two parts: VQA and text QA. We focus on the VQA split, and similar to MME, a short instruction is appended at the end of each question before passing it to VLMs.

POPE [24] is a yes/no benchmark for testing the hallucination of VLMs. Similar to MME, we append an instruction and perform string matching for the final performance.

HRBench8k [43] and V* [45] are both VQA benchmarks featuring high-resolution image inputs. The target objects or task-relevant regions are relatively small, thus requiring strong grounding skills for VLMs to answer correctly. HRBench8k contains 800 VQA pairs and V* has around 300 samples; we include both the OCR and GPT-hard splits of V*.

### B Additional Quantitative Results

#### B.1 Ablation Study

Position of the Learnable Tokens As mentioned in Sec. 6, we append a set of learnable tokens to yes/no and MCQ questions to enforce VLMs to attend to the image after our attention alignment training. In this ablation, we study the impact of the position of the learnable tokens. Recall that given a VQA sample, VLMs process the input image and question into a sequence of tokens ordered as: image tokens, question tokens, and instruction tokens. Within this sequence, there are three positions to place our learnable tokens: (1) prefix: between image and question tokens, (2) infix: between question and instruction tokens, and (3) postfix: after the instruction tokens at the end of the sequence. We conduct an ablation study to analyze the impact of the position, with results shown in Tab. A3. One can see that among the three positions, infix achieves the best overall performance. Our hypothesis is that since the visual attention variation is primarily driven by the question framing, placing learnable tokens immediately after the question tokens allows them to learn adjustments conditioned on the given question, while also having a more direct impact on the question compared to placing them at the end of the sequence.

Weighting Strategy for Alignment Loss Across Samples In our prompt tuning framework, instead of applying the attention alignment loss to all training samples equally, we adopt a confidence-based weighting scheme. The motivation is that some open-ended questions may be answered incorrectly, and using attention maps from such samples would

###### Gemma3-12B

###### V F Visual Energy

###### V F Box Attention

###### GQAF Visual Energy

GQAF Box Attention

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

1.4

- 75%
- 76%
- 77%
- 78%
- 79%

- 75%
- 76%
- 77%
- 78%
- 79%

%ofVisualAttention

%ofVisualAttention

8

%ofTotalAttention

%ofTotalAttention

1.2

6

1.0

0.8

4

0.6

2

Open-ended Yes/No MCQ

Open-ended Yes/No MCQ

Open-ended Yes/No MCQ

Open-ended Yes/No MCQ

###### GLM-4.1V-9B-Base

###### V F Visual Energy

###### V F Box Attention

###### GQAF Visual Energy

GQAF Box Attention

40%

- 1%

- 1%

- 1%
- 2%

- 2%

- 2%

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

- 8%
- 9%
- 10%
- 11%
- 12%

%ofVisualAttention

%ofVisualAttention

%ofTotalAttention

%ofTotalAttention

35%

30%

30%

25%

25%

20%

20%

Open-ended Yes/No MCQ

Open-ended Yes/No MCQ

Open-ended Yes/No MCQ

Open-ended Yes/No MCQ

###### LLaVA-OneVision-1.5-8B

###### V F Visual Energy

###### V F Box Attention

###### GQAF Visual Energy

GQAF Box Attention

45%

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

6%

40%

%ofVisualAttention

%ofVisualAttention

18%

%ofTotalAttention

%ofTotalAttention

40%

15%

35%

30%

4%

30%

12%

20%

25%

2%

10%

20%

Open-ended Yes/No MCQ

Open-ended Yes/No MCQ

Open-ended Yes/No MCQ

Open-ended Yes/No MCQ

Qwen3-VL-8B

###### V F Visual Energy

###### V F Box Attention

###### GQAF Visual Energy

GQAF Box Attention

35%

- 0.5%
- 1.0%

- 1.5%
- 2.0%

- 2.5%

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

35%

14%

%ofVisualAttention

%ofVisualAttention

%ofTotalAttention

%ofTotalAttention

30%

30%

12%

25%

20%

10%

25%

15%

8%

20%

10%

Open-ended Yes/No MCQ

Open-ended Yes/No MCQ

Open-ended Yes/No MCQ

Open-ended Yes/No MCQ

Fig. A2: Visual energy drops significantly on non-open-ended framings.

introduce inaccurate and noisy training signals. During training, we can use ground-truth labels to assess correctness and apply either a hard threshold or a soft threshold, where the latter reflects the model’s confidence in its own output and thus potentially yields more reliable attention maps. To study the effectiveness of confidence weighting, we conduct an ablation comparing training with confidence weighting against training with equal weighting across all samples. As shown in Tab. A4, confidence weighting to filter out low-confidence samples consistently improves performance across benchmarks compared to equal weighting, verifying the effectiveness of this design choice.

#### B.2 Visual Attention Analysis for Other VLMs

In Sec. 5, we mainly focus on the behavioral study of Qwen2.5VL-7B. Here we further provide analysis on multiple models, including Gemma3 [38], GLM-4.1V [48], LLaVA-OneVision1.5 [4], and Qwen3VL [5]. The visual attention analysis and layer-wise analysis results are shown in Fig. A2 and Fig. A3, respectively. One can see that the overall visual attention behavior across framings is similar to Qwen2.5VL-7B in the main paper on V*F and GQAF, with higher bounding box attention for open-ended questions and lower attention for MCQ and yes/no questions. On Gemma3, while the bounding box attention trends are similar to our previous observation, we see that yes/no questions can potentially exhibit higher visual energy than open-ended question framings. The same behavior can be observed on GQA as well.

###### Gemma3-12B

0.008

0.8

VisualEnergy

BoxAttention

0.6

0.006

0.4

0.004

V F Open-Ended

V F Yes/No

0.2

0.002

V F MCQ

0.0

0.000

0 10 20 30 40

0 10 20 30 40

Layer Index

Layer Index

###### GLM-4.1V-9B-Base

0.6

V F Open-Ended

VisualEnergy

BoxAttention

V F Yes/No

0.04

V F MCQ

0.4

0.02

0.2

10 15 20 25 30 35 40

10 15 20 25 30 35 40

Layer Index

Layer Index

###### LLaVA-OneVision-1.5-8B

0.06

0.6

V F Open-Ended

VisualEnergy

BoxAttention

V F Yes/No

0.04

0.4

V F MCQ

0.02

0.2

0.00

0 5 10 15 20 25 30 35

0 5 10 15 20 25 30 35

Layer Index

Layer Index

Qwen3-VL-8B

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0.6

V F Open-Ended

0.06

VisualEnergy

BoxAttention

V F Yes/No

V F MCQ

0.4

0.04

0.2

0.02

0.0

0 5 10 15 20 25 30 35

0 5 10 15 20 25 30 35

Layer Index

Layer Index

Fig. A3: Visual energy drops significantly on non-open-ended framings.

- C Dataset Examples and Qualitative Results

We first provide some data samples of open-ended and Yes/No question reframing using GPT-5.1 in Fig. A4 and Fig. A5, respectively. Data are randomly sampled from GQAF; we showcase the original question in its open-ended or Yes/No framing, which is subsequently reframed into the other two formats.

Additionally, we provide a qualitative comparison of inference with and without our learned tokens in Fig. A6. Given a VQA sample, we perform inference using a Qwen2.5VL-7B baseline alongside our approach with extra learned tokens, visualizing the attention rollout during output generation. As shown, the attention maps from our approach focus significantly more on task-relevant regions, whereas the baseline often spreads to irrelevant background areas. The Diff map highlights the percentage-wise change, further demonstrating that our learned tokens effectively shift attention toward the target area during inference.

- D Limitation

While our study provides a comprehensive mechanistic analysis of framing effects across five diverse and prominent VLM families, the architectural landscape of multimodal models is rapidly evolving. Future research could extend this investigation to other emerging architectures, such as Mamba-based VLMs or Mixture-of-Experts (MoE) models, to determine if these structural paradigms inherently mitigate or exhibit similar framing-induced attention shifts.

Original Questions and Answers Reframed Questions and Answers

[Figure 33]

Open-eneded Q: On which side is the tap? A: left

Y/N Q: Is the tap on the right? A: No

MCQ: On which side is the tap located? A. left B. right C. center D.both sides Ans: A

[Figure 34]

Open-eneded Q: What is the material of the chair? A: leather

Y/N Q: Is the chair made of leather? A: Yes

MCQ: What material is the chair made of? A. leather B. fabric C. wood D. plastic Ans: A

[Figure 35]

Open-eneded Q: Is the black umbrella closed or is it open? A: open

MCQ: What is the condition of the black umbrella? A. closed B. open C. partially open D. collapsed Ans: B

Y/N Q: Is the black umbrella closed? A: No

###### Fig. A4: Qualitative examples of open-ended question reframing.

Original Questions and Answers Reframed Questions and Answers

[Figure 36]

[Figure 37]

Y/N Q: Is the man to the right of the bike standing on a ski? A: no

Open-ended Q: What is the man to the right of the bike standing on? A: skateboard

MCQ: What is the man to the right of the bike standing on? A. a skateboard B. a ski C. a surfboard D. a stool Ans: A

[Figure 38]

[Figure 39]

Open-ended Q: What material is the traffic cone to the left of the skateboard made of? A: rubber

Y/N Q: Is the traffic cone to the left of the skateboard made of metal? A: no

MCQ: What material is the traffic cone to the left of the skateboard made of? A. metal B. plastic C. rubber D. wood Ans: C

[Figure 40]

[Figure 41]

Open-ended Q: Where is the cup located in the image? A: on the right

MCQ: Where is the cup located in the image? A. on the left B. on the right C. in the center D. in the foreground Ans: B

Y/N Q: Is the cup on the right? A: yes

###### Fig. A5: Qualitative examples of Yes/No question reframing.

Questions and GT Answers Baseline

Ours Diff

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Q: Which of the following best describes the items visible in the image? A. Both a snowboard and a collar B. Only a collar C. Only a snowboard A: B

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

Q: Is the dog black? A: Yes

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Q: Are both the gate and the door made of the same material? A: Yes

[Figure 61]

[Figure 62]

###### Fig. A6: Qualitative comparison of Baseline and Ours (Qwen2.5-VL-7B) .

