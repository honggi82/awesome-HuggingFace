## See Less, See Right: Bi-directional Perceptual Shaping For Multimodal Reasoning

Shuoshuo Zhangϕπ* Yizhen Zhangϕ* Jingjing Fuπ Lei Songπ Jiang Bianπ Yujiu Yangϕ† Rui Wangπ† π Microsoft Research Asia ϕ Tsinghua University

zss24@mails.tsinghua.edu.cn, yang.yujiu@sz.tsinghua.edu.cn, ruiwa@microsoft.com

# arXiv:2512.22120v2[cs.CV]5Feb2026

#### Abstract

###### Input Data

External Tool–Dependent Paradigm

[Figure 1]

###### Q: What is the label of the curve which decreases the most from 0 to 50?

###### RL Reward Signal

Large vision–language models (VLMs) often benefit from intermediate visual cues, either injected via external tools or generated as latent visual tokens during reasoning, but these mechanisms still overlook fine-grained visual evidence (e.g., polylines in charts), generalize poorly across domains, and incur high inference-time cost. In this paper, we propose Bi-directional Perceptual Shaping (BiPS), which transforms question-conditioned masked views into bidirectional where-to-look signals that shape perception during training. BiPS first applies a KL-consistency constraint between the original image and an evidencepreserving view that keeps only question-relevant regions, encouraging coarse but complete coverage of supporting pixels. It then applies a KL-separation constraint between the original and an evidence-ablated view where critical pixels are masked so the image no longer supports the original answer, discouraging text-only shortcuts (i.e., answering from text alone) and enforcing fine-grained visual reliance. Across eight benchmarks, BiPS boosts Qwen2.5VL-7B by 8.2% on average and shows strong out-of-domain generalization to unseen datasets and image types. Code is available at https://github.com/zss02/BiPS.

[Figure 2]

Outcome GT

[Figure 3]

[Figure 4]

Iterative Tool × n

[Figure 5]

|[Figure 6]<br><br>[Figure 7]|
|---|

[Figure 8]

- 1st Round
- 2nd Round

[Figure 9]

|[Figure 10]<br><br>[Figure 11]|
|---|

[Figure 12]

[Figure 13]

The green curve.

Intrinsic Native Perception Paradigm (Ours)

Intrinsic Latent CoT Paradigm

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

RL Reward Signal Bi-directional RL Reward Signal

|[Figure 18]<br><br>[Figure 19]|
|---|

|[Figure 20]<br><br>[Figure 21]|
|---|

|[Figure 22]<br><br>[Figure 23]|
|---|

Outcome GT

Outcome GT

Guiding Latent Tokens

Guiding Goals Separating distractors

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

<latent token>The red curve. The blue curve.

Figure 1. Illustration of different paradigms. Within each colored box, the top row illustrates the training stage and the bottom row shows the inference stage. Prior approaches are limited by shape-rigid inference-time tools and domain-specific solutions that generalize poorly.

of fine-grained visual cues is still a bottleneck [2, 20, 35]. If perception slips, downstream reasoning can rely on incomplete or misleading cues, yielding plausible but evidencemismatched answers and degrading VQA performance.

#### 1. Introduction

Large vision–language models (VLMs) are increasingly serving as a unified interface for both visual and languagebased reasoning [1, 7]. Among real-world applications, visual question answering (VQA) is a widely deployed, highimpact task: a system must parse a natural-language query, localize the pertinent visual evidence, and produce an answer whose reasoning remains tied to that evidence. Despite rapid progress, the perceptual capability of VLMs, including the identification, localization, and accurate reading

To mitigate this perceptual bottleneck, a complementary line of work augments VLMs with external visual tools (e.g., cropping, masking, segmentation) that produce evidence-focused intermediate visual cues at inference time [11, 13, 24, 33, 42, 53, 57]. Recent efforts further collect step-by-step “visual chain-of-thought” traces, where the model is guided by intermediate boxes, tool-use trajectories, or auxiliary images and then trained to reproduce these visual reasoning steps [6, 11, 33, 50, 57]. In practice, such mechanisms improve grounding and answer accuracy

*Equal contribution. Work done during internship at Microsoft. †Corresponding author.

across tasks such as chart understanding, image-based math problems, and natural-image VQA with sparse cues.

Despite their utility, these approaches face three practical limitations. (i) Shape rigidity. Focused regions are typically rectangular crops or coarse masks, which miss irregular or fragmented evidence, such as thin polylines and intersections in charts, lesion contours in medical images, or nonconvex polygons in geometry diagrams. (ii) Scenariospecific solutions. Both custom tools and training pipelines that teach models to emit task-tailored latent visual tokens at inference time are tightly coupled to particular layouts or domains, limiting generalization. (iii) Inference-time overhead. Whether implemented via external tools or learned visual hints (e.g., boxes, masks, latent visual tokens), generating intermediate cues at inference introduces extra steps and computation and increases the risk of cascading errors. As illustrated in Fig. 1, we take a different route: programmatically generating perfect, ground-truth visual cues not as inference-time crutches, but as training signals. Rather than teaching the model to output specific visual cues, we use these cues to shape the model’s internal policy, biasing it toward grounding its answers in visually supported content.

In this paper, we propose Bi-directional Perceptual Shaping (BiPS), a two-stage training-time approach integrated into the Group Relative Policy Optimization (GRPO) framework [34], which shapes a VLM’s perception by pulling predictions toward an evidence-preserving view and pushing them away from an evidence-ablated view. BiPS first builds a question-conditioned evidencepreserving view that keeps regions needed to answer the query while masking distractors, and applies a consistency constraint based on the Kullback–Leibler (KL) divergence to align the model’s predictions on this view with those on the original image, encouraging coarse but complete coverage of supporting pixels. It then constructs a complementary evidence-ablated view that finely removes critical pixels so the answer changes or becomes unanswerable, and adds a KL-based separation term that pushes predictions on the original and ablated views apart, discouraging textonly shortcuts (i.e., answering from text alone) and enforcing fine-grained visual reliance. This bidirectional shaping yields visually grounded decisions while requiring no extra annotations or customized parsers at inference.

Realizing BiPS requires precise question-conditioned supervision in the form of paired views, including an evidence-preserving view and an evidence-ablated view for each image and question pair. Naive pixel-level masking or cropping is common, but it remains coarse and shape-constrained. Complex, multi-panel charts naturally carry the fine-grained evidence (e.g., thin polylines, layered marks, and small symbols), making charts a rich source of training signals. We therefore build a programmatic data pipeline for chart data that generates the required evidence-

preserving and evidence-ablated views. In this work, we instantiate the pipeline using ECD [49], a synthetic corpus of complex multi-panel figures paired with executable rendering code. Because each figure is generated by code, every object (marks, layers, axes, legends) has explicit provenance, which enables exact edits to synthesize the two complementary views. This pipeline yields 13K synthetic training examples. Fine-tuned solely on this set, BiPS already generalizes well: across eight benchmarks spanning realworld figure datasets (e.g., CharXiv [43], ChartQAPro [26]) and out-of-domain general VQA (e.g., MathVista [22], MMStar [3]), average accuracy improves by +7.3% over the base model (Qwen2.5-VL-7B [1]). Adding 39k mathspecialized examples with standard GRPO further increases the average gain to 8.2%.

The contributions are summarized as follows:

- • We turn inference-time visual cues into training signals that internalize perception. BiPS applies a KL consistency term toward an evidence-preserving view and a KL separation term from an evidence-ablated view. These signals teach the model to capture visual evidence, including fine and irregular details, yielding evidencefaithful predictions without test-time overhead.
- • We build a programmatic data construction pipeline that uses executable chart scripts to synthesize precise paired views (preserve and ablate) without human labeling. Fine-tuned solely on chart-derived cues, BiPS extends well to real-world figures and general VQA, indicating strong out-of-domain generalization.
- • Experiments show that BiPS delivers substantial, consistent performance gains, with strong cross-domain generalization and high data efficiency. Trained on only 13K chart samples, it boosts Qwen2.5-VL-7B by 7.3% on average across eight widely used chart and general VQA benchmarks (e.g., CharXiv, MathVista, MMStar), surpassing specialized models trained on vastly larger datasets. Adding 39K math-focused samples with GRPO further increases the average improvement to 8.2%.

#### 2. Related Works

##### 2.1. Multimodal Reasoning

Recent advances in reinforcement learning (RL) for language models have sparked growing efforts to extend RL to multimodal reasoning [23, 54]. Recent studies apply RL to strengthen visual understanding or reasoning across domains, including single-image [5, 16, 29, 40, 48] reasoning for spatial perception, multi-image reasoning [55] for crossscene consistency, video reasoning [8] for temporal causality, and chart reasoning [4] for quantitative alignment.

Complementary to advances in language reasoning, a growing body of work improves VLMs by injecting intermediate visual cues at inference time. Typical

Stage 1: Consistency Stage

𝑂

Original Question

[Figure 28]

Forward

x

𝜋

[Figure 29]

𝑂

Policy Model

Minimize Difference

𝐿

…

𝜋

𝑂

Evidence-Preserving View

Copy

Forward

Q: How spread is the blue curve … in Subplot (a)?

𝑂 𝑂

𝑟

𝐴

[Figure 30]

Rollout

Group Norm

Policy Model

Rule-based Verifier

𝑟

𝐴

[Figure 31]

… 𝑂

…

…

𝑟

𝐴

Original Image

Copy

Forward

𝑂

𝜋

Original Question

[Figure 32]

###### Maximize Difference

[Figure 33]

𝑂

𝐿

Policy Model

Forward

…

𝜋x

𝑂

Evidence-Ablated View

Stage 2: Separation Stage

- Figure 2. Overview of the Bi-directional Perceptual Shaping (BiPS) framework. BiPS employs a two-stage training curriculum built on the GRPO framework. Stage 1 (Consistency Stage) minimizes the KL-divergence (Lcons) between the original policy (πθ) and the policy on an evidence-preserving view (π˜θ). Stage 2 (Separation Stage) maximizes the KL-divergence (Lsep) between the original policy (πθ) and the policy on an evidence-ablated view, forcing the model to ground its reasoning in visual evidence.

strategies highlight question-relevant content via bounding boxes [33], question-conditioned crops [24], or masks that suppress distractors [57], biasing predictions toward the appropriate visual evidence. Building on this idea, subsequent methods design task-aware tools or specialized modules for tighter evidence localization, such as chart-specific annotation tools [11, 53] and visual sketchpad [13]. Pushing this paradigm further, recent studies supervise models with labeled evidence and train them to reproduce coordinates, masks, or tool selections so that downstream answers can be conditioned on the predicted evidence [6, 11, 32, 33, 36, 37, 45, 57, 58]. However, these designs inherit several limitations: rectangular or coarse masks struggle with fine-grained or irregular structures, task-specific engineering reduces generalization, and multi-step pipelines introduce nontrivial inference overhead.

informative detailed signals inside the figure.

##### 2.2. Chart Understanding And Reasoning

Unlike general multimodal reasoning on natural scenes [9] or geometric diagrams [21, 22, 41], chart understanding targets structured quantitative graphics where numerical relations are encoded by axes and marks, demanding precise value perception and visual–numerical correspondence. Early chart-domain studies mainly focused on lowlevel perception tasks such as chart element detection and text extraction [25, 30], while recent benchmarks shift toward reasoning-centric tasks [26, 38, 43] that require interpreting implicit patterns, making comparisons, and executing multi-step computations over visualized data. To enhance such reasoning, subsequent studies convert charts into structured symbolic programs or code for executable reasoning [18, 47, 51, 56], while others explore multimodal feedback, reflective learning, and reinforcement-based optimization [4, 15, 27] to improve reliability. Nonetheless, chart reasoning still faces challenges such as limited chartto-code accuracy [25, 46], weak generalization to diverse layouts, and difficulty handling high-frequency curve fluctuations or subtle oscillatory patterns, which often cause quantitative misalignment. Addressing these challenges de-

A recent work removes the need for explicit intermediate images by encouraging models to reason through latent internal processes [50], but such approach remain confined to specific tasks. ChiP [10] and PAPO [44] inject Gaussian noise or random masks as negative perturbations that penalize reliance on incorrect images to avoid visual hallucinations and text-only shortcuts, yet these methods overlook

complex, multi-panel charts along with their executable rendering code. Charts serve as an ideal substrate for this purpose: their elements (marks, axes, legends) are semantically structured, fine-grained, and explicitly linked to the underlying code. This enables editing at the code level rather than pixels, yielding semantically precise and fine-grained control over visual evidence. As shown in Figure 3, our pipeline consists of the following three main stages.

Original Data Reformulation

[Figure 34]

|Prompt: Rewrite the question and verify …|
|---|

|Question: How spread is the blue curve … in Subplot (a)?<br><br>Answer: The blue … Code: import numpy …|
|---|

|Question: How spread is … in Subplot (a)? A. 2 … Answerable: Yes|
|---|

Difficulty Filtering

Question Reformulation and Validation. The original dataset’s reasoning questions are open-ended and difficult to verify by predefined rules. Following a typical RLVR setting [29, 34], we refine this data by employing an auxiliary LLM arbitrator, GPT5-mini, to convert the original questions into a multiple-choice format. This arbitrator is provided with the chart’s source code and metadata to ensure that the reformulated question remains answerable and that the ground-truth option is correct. This step provides verifiable supervision compatible with downstream RL training and ensures the overall quality of our base question set.

Code Editting

[Figure 35]

|Evidence-Preserving View Code Edit: import numpy …|
|---|

[Figure 36]

|Evidence-Ablated View Code Edit: import numpy …|
|---|

- Figure 3. Overview of our data generation pipeline. This pipeline programmatically edits chart source code to generate the

paired Evidence-Preserving (Ipres) and Evidence-Ablated (Iabl) views used for bi-directional training.

Difficulty Filtering. To focus the training on non-trivial examples, we filter out questions that are too “easy” for the base model (Qwen2.5-VL-7B-Instruct [1]). We perform 8 rollouts for each validated question. Any question that the base model answers correctly in all rollouts is considered “easy” and is discarded from our training set, allowing our method to concentrate on more challenging reasoning tasks.

mands finer-grained perception and robust visual reasoning.

#### 3. Method

To address the core perceptual failures of VLMs, namely their tendency to be distracted by irrelevant visual information and their inability to focus on fine-grained evidence, we propose BiPS. As illustrated in Figure 2, BiPS is implemented as a two-stage training curriculum consisting of a Consistency Stage followed by a Separation Stage. The Consistency Stage guides the model to maintain consistent predictions when only evidence-relevant regions are retained, whereas the Separation Stage drives the model to diverge when the critical regions are removed. Both stages are implemented as bidirectional KL-based objectives, optimized within the GRPO [34] framework.

Code Editing and Counterpart Rendering. This is the core stage where we generate the paired visual counterparts required for BiPS training. For each filtered question q and its corresponding chart-rendering code C, we employ the LLM arbitrator guided by structured prompts to identify and modify the relevant code components.

- • Evidence-Preserving View (Ipres): To construct the minimal evidence view used in the Consistency Stage, the arbitrator removes code segments unrelated to answering q and executes the remaining script to render an image containing only the necessary visual elements.
- • Evidence-Ablated View (Iabl): To create the complementary view for Separation Stage, the arbitrator identifies code segments that provide the key evidence and removes them while keeping general contextual structures such as axes, legends, and layout. The resulting image omits fine-grained cues yet preserves the global context.

##### 3.1. Programmatic Data Construction Pipeline

To enable BiPS to learn evidence-aware grounding, we require training supervision that precisely distinguishes relevant from irrelevant visual content. Such supervision takes the form of paired visual views—an evidence-preserving view that retains the regions needed to answer the question, and an evidence-ablated view where critical regions are removed. While masking or cropping images is a common way to approximate such pairs, these operations are often coarse and shape-limited (e.g., rectangles), failing to capture fine-grained or irregular evidence.

This programmatic process produces a large-scale dataset of (I,q,Ipres,Iabl) tuples, providing semantically precise and well-aligned supervision for BiPS. After this pipeline, we obtained a final set of 13K high-quality training samples (Detailed data statistics are available in Appendix).

We overcome this limitation by constructing the paired views programmatically in the code domain. Our data generation pipeline builds upon ECD [49], which provides

##### 3.2. Bi-directional KL Constraints

Our core methodology comprises two complementary KL constraints operating on the paired views constructed in Section 3.1. The first constraint provides the primary positive guidance for evidence localization, while the second provides a negative-space regularization to ensure that this localization is robust and visually grounded.

###### 3.2.1. Focusing via Consistency

To teach the model to ignore distractions and focus on the correct region, we use a Consistency constraint, which enforces that the model’s policy on the full image I should be consistent with its policy on the evidence-preserving view Ipres. We apply this constraint by minimizing the KLdivergence between these two distributions:

Lcons = E(I,q,r) I(r=1) min ccons, DKL πθ(·|I,q) sg π ˜θ(·|Ipres,q) .

(1)

Here πθ is the model’s answer distribution; π˜θ denotes a target distribution computed on the evidence-preserving view. sg[·] indicates stop-gradient so that the Ipres branch serves as a fixed target; I(r=1) restricts supervision to validated correct samples; and ccons clips the KL term for stability.

The forward direction DKL πθ ∥π˜θ pulls probability mass on I toward evidence-supported answers on Ipres, encouraging the policy to base decisions on preserved evidence and treat extraneous regions as redundant.

###### 3.2.2. Robustness via Separation

On its own, Lcons is insufficient as it is susceptible to “shortcut learning”. A model may satisfy the consistency Lcons by exploiting surrounding text (OCR) or language priors to produce identical answers for the original and evidencepreserving images, without attending to fine-grained details (e.g., curves).

To ensure the model’s focus is truly visually grounded, we introduce a complementary Separation constraint, which acts as a regularizer, forcing the model to learn that the visual signal is indispensable. It achieves this by enforcing that the model’s policy on the full image I must be divergent from its policy on the evidence-ablated View Iabl. We maximize the KL-divergence between these distributions:

Lsep = E(I,q) min csep, DKL πθ(·|I,q) sg π ˜θ(·|Iabl,q) .

(2)

where csep is a clipping hyperparameter. This objective penalizes similarity between the two policies until the divergence exceeds csep, breaking text-only shortcuts and promoting fine-grained grounding.

##### 3.3. A Coarse-to-Fine Training Curriculum

Optimizing Lcons (an attractive force) and Lsep (a repulsive force) simultaneously can be challenging due to potentially conflicting gradients. We therefore devise a two-stage curriculum that decouples these objectives.

Base GRPO Training Objective. Before detailing the stages, we define the base GRPO [34] objective, which extends Proximal Policy Optimization (PPO) by normalizing rewards across rollouts within the same group to stabilize training. The training objective is:

LGRPO = −E(I,q) min rt(θ)At, clip(rt(θ), 1 − ϵ, 1 + ϵ)At − γDKL πθ∥πref .

(3)

Here, At denotes the group-relative advantage, ϵ is the clipping threshold, and γ controls the KL penalty strength.

- Stage 1: Consistency Stage. In the first stage, we train the model on the primary task of evidence localization using the consistency constraint:

- LStage1 = LGRPO + α Lcons (4)

where α is the consistency constraint coefficient. This stage establishes the foundational, coarse-grained skill of what to focus on.

Stage 2: Separation Stage. Building on the Stage 1 checkpoint, we now introduce the separation constraint using a view that removes fine-grained visual evidence, ensuring the learned focus is robust and truly visually grounded:

- LStage2 = LGRPO − β Lsep (5)

where β is the separation constraint coefficient.

This coarse-to-fine curriculum first applies the positive signal (Lcons) and then the regularizer (Lsep) to ensure the learned policy is both accurate and grounded. We demonstrate the superiority of this curriculum over joint training and the reversed order in our ablation studies (Sec. 4.3).

#### 4. Experiments 4.1. Experiment Setting

Implementation. We use Qwen2.5-VL-7B [1] as the base model. Stage 1 is trained for 5 epochs on 7K samples containing evidence-preserving views Ipres. Subsequently, Stage 2 trains for 3 epochs on 13K samples including evidence-ablated views Iabl, producing our BiPS-Chart model. To further enhance general reasoning capabilities,

[Figure 37]

###### Table 1. Evaluation on chart understanding and general perception & reasoning benchmarks. Models with are chart-specialized;

[Figure 38]

models with are math-specialized; † denotes our models. Blue numbers denote chart-related data, and red numbers denote math-related or perception-related data. Avg. is the arithmetic mean over available metrics. Models trained on MathVision (marked with ⋆) omit their scores on MathVision. Best scores are in bold, and second-best scores are underlined. Baselines are reproduced using VLMEvalKit to ensure strict fairness.

Chart Understanding and Reasoning General Perception and Reasoning

Model Data

Avg. CharXiv ChartQAPro ChartMuseum Evochart MathVista MathVision MathVerse-VO MMStar

Closed-Source Models GPT-4o - 47.1 37.7 42.2 49.8 63.8 31.2 40.6 65.1 47.2 Claude-3.7-Sonnet - 64.2 – 60.3 70.7 74.5 58.6 52.0 68.8 –

###### Open-Source General Models

Qwen2.5-VL-7B - 42.5 36.6 26.8 52.0 68.2 25.2 41.1 62.1 44.3 InternVL3-8B - 37.6 36.9 28.2 55.0 70.4 26.3 33.9 68.2 44.6

Multimodal Reasoning Models ChartLlama-13B 160K 14.2 – – 9.5 – – – – – ChartGemma-3B 163K 12.5 6.8 12.2 30.6 – – – – – TinyChart-3B 1.3M 8.3 13.3 12.5 25.5 – – – – – R1-OneVision-7B 67K + 98K 33.8 36.1 27.2 35.1 64.1 29.9 40.0 52.2 39.8 Vision-R1-7B⋆ 73K + 137K 42.5 39.6 28.5 59.9 73.2 - 47.7 64.8 – DeepEyes-7B 14K + 33K 42.9 38.1 28.1 65.6 70.8 26.5 44.9 63.0 47.5 BigCharts-R1-7B 1.7M 41.3 - - - - - - - – Chart-R1-7B 258K 46.2 44.0 31.7 64.7 67.5 20.6 28.1 61.1 45.5 GRPO 13K + 39K 45.4 50.2 32.9 68.0 74.3 27.3 42.6 64.6 50.7 BiPS-Chart-7B† 13K 49.4 51.9 33.5 68.2 73.5 27.2 44.4 64.9 51.6 ∆ over base model ↑ 6.9 ↑ 15.3 ↑ 6.7 ↑ 16.2 ↑ 5.3 ↑ 2.0 ↑ 3.3 ↑ 2.8 ↑ 7.3 BiPS-General-7B† 13K+39K 50.6 51.8 34.0 68.7 75.0 28.6 45.3 65.7 52.5 ∆ over base model ↑ 8.1 ↑ 15.2 ↑ 7.2 ↑ 16.7 ↑ 6.8 ↑ 3.4 ↑ 4.2 ↑ 3.6 ↑ 8.2

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

we re-optimize the Stage 2 checkpoint for 3 epochs on 39K samples from ViRL39k [40] using standard GRPO, yielding the final BiPS-General. All models are optimized with AdamW (lr = 1 × 10−6) on 8×H100 GPUs. We set the constraint coefficients to α = 0.01 and β = 0.02, and the clipping thresholds to ccons = 1.0 and csep = 0.2.

Benchmark. We evaluate our models on a comprehensive suite of benchmarks spanning two categories. To assess chart understanding and reasoning capabilities, we report scores on CharXiv [43], ChartQAPro [26], ChartMuseum [38], Evochart [14] and ECD-Bench [49]. For general perception and reasoning, We use MathVista [22], MathVision [41], MathVerse-VO [52], and MMStar [3].

Baseline. We evaluate our method against diverse baselines, including closed-source models like GPT-4o [17] and Claude-3.7-Sonnet as high-level SOTA references. Our open-source comparisons include general models such as InternVL3-8B [59] and Qwen2.5-VL-7B [1]. We also compare against specialized reasoning models, including chart-focused systems (ChartLlama [12], ChartGemma [28], Chart-R1 [4], BigCharts-R1 [27]) and broader

multimodal reasoners (R1-OneVision [48], Vision-R1 [16], DeepEyes [58]) that handle both chart and math reasoning.

##### 4.2. Main Results

Table 1 reports performance across a diverse set of chart understanding and general multimodal reasoning benchmarks. Overall, our approach delivers substantial improvements over the baseline Qwen2.5-VL-7B, boosting the average score by +7.3 points (from 44.3 to 51.6) with BiPSChart-7B and by a total of +8.2 points (to 52.5) with BiPSGeneral-7B. This reflects clear and consistent gains in both chart-centric and general reasoning performance.

BiPS-Chart-7B first illustrates the power of our method, achieving significant gains on both chart-specific and general out-of-domain benchmarks. Trained on only 13K chart samples, it delivers substantial gains on challenging chart reasoning benchmarks. On ChartXiv, performance increases from 42.5 (Qwen2.5-VL-7B) to 49.4 (+6.9), and on Evochart from 52.0 to 68.2 (+16.2). Crucially, BiPSChart-7B also exhibits strong OOD generalization, boosting performance on unseen general reasoning tasks such as MathVista (+5.3) and MMStar (+2.8). These improvements surpass other chart-specialized models that rely on

- Table 2. Ablation study on the components of BiPS. We analyze the contribution of the consistency constraint (Lcons) and the separation constraint (Lsep) when added to the GRPO baseline.

###### Method CharXiv ECD ChartMuseum

Qwen2.5-VL-7B 42.5 19.0 26.0 GRPO 44.3 35.6 30.8

GRPO with Lcons 47.2 36.3 31.3 GRPO with Lsep 47.7 38.3 31.8 Ours 49.4 39.9 33.5

far larger training sets, including TinyChart-3B, BigChartsR1-7B, and Chart-R1-7B, which are trained on hundreds of thousands to millions of chart examples but achieve lower scores on these key benchmarks. This comparison underscores that BiPS, by enhancing the model’s core visual perception capabilities, enables a more data-efficient acquisition of chart reasoning and promotes robust generalization across diverse chart types.

Building on these gains, BiPS-General-7B incorporates 39K math-focused samples to further enhance the model’s reasoning capacity. This integration leads to consistent improvements across chart and general benchmarks. BiPSGeneral-7B reaches 50.6 on ChartXiv, 68.7 on Evochart, 75.0 on MathVista, and 65.7 on MMStar, surpassing both the baseline and the chart-only BiPS-Chart-7B model. This demonstrates that the introduction of math-specific data further compounds these gains by explicitly strengthening the model’s abstract and numerical reasoning abilities.

To verify that the improvements stem from our specific perceptual shaping pipeline rather than merely applying RL to the data, we compare BiPS-General-7B against a baseline trained with standard GRPO. This baseline is fine-tuned on the exact same combined dataset (programmatic samples + ViRL39k) but treats all data uniformly without the proposed two-stage perceptual shaping curriculum. While standard GRPO yields substantial gains over the base model, BiPSGeneral consistently outperforms it across all benchmarks. Notably, on complex chart reasoning tasks like CharXiv, our method surpasses the standard GRPO baseline by a significant margin (+5.2). This performance gap highlights a critical insight: simply optimizing reasoning via RL is insufficient if the underlying visual grounding is flawed. By explicitly shaping the model’s perception through our programmatic curriculum, BiPS ensures that the RL process operates on high-fidelity visual signals, thereby amplifying the effectiveness of the optimization.

##### 4.3. Ablation Study

Impact of Bi-directional KL Constraints. Table 2 presents the component-wise ablation of BiPS, where individual constraints are added on top of the GRPO baseline. Adding either constraint consistently improves performance

- Table 3. Analysis of the Training Curriculum. We compare our two-stage curriculum against alternative optimization strategies: Joint Training (optimizing Lcons and Lsep simultaneously) and a Reversed Order (Stage 1: Lsep, Stage 2: Lcons).

Curriculum Charxiv ECD ChartMuseum

Joint Training 46.4 36.7 31.5 Reversed Order 46.8 39.2 31.3 Ours 49.4 39.9 33.5

- Table 4. Impact of the Counterpart Generation Strategy. We compare our programmatic code-editing pipeline against a common baseline that uses random masking.

###### Method CharXiv ECD ChartMuseum

Random Masking 44.8 37.6 31.8 Ours 49.4 39.9 33.5

50

Consistency constraint - α (when β = 0) Separation constraint - β (when α = 0)

| |
|---|

48

Accuracy(%)

46

44

42

40

0 0.01 0.02 0.04 0.08

Coefficient value of α and β

Figure 4. Accuracy on CharXiv with respect to the weighting coefficients of the consistency α and separation β constraints.

across benchmarks. Integrating the consistency constraint Lcons leads to notable gains on CharXiv (+2.9%), indicating that the coarse-grained focusing stage effectively guides the model toward relevant visual evidence. Incorporating the separation constraint Lsep yields more improvements on ECD-Bench (+2.7%) and CharXiv (+3.4%), showing that fine-grained visual grounding suppresses shortcut reliance and strengthens visual reasoning. Combining both stages achieves the best overall results (e.g., 39.9% on ECD-Bench and 49.4% on CharXiv), demonstrating that coarse-grained focusing and fine-grained grounding work synergistically to build a robust and well-aligned perceptual capability.

Analysis of the Training Curriculum. We evaluate the Coarse-to-Fine Training Curriculum against two alternatives: Joint Training (optimize both constraints simultaneously) and Reversed Order (apply Lsep before Lcons). As shown in Table 3, our curriculum attains the best results across all three benchmarks. Specifically, both two-stage

Input Input

Question: How many plots show the intersection of 𝐻(𝑓) and 𝐻(𝛽) ?

Question: What is the highest accuracy obtained on the Fashion-MNIST dataset across all plots?

[Figure 53]

[Figure 54]

Responses

Responses

Qwen: Exponential Plot: intersect at some point on the graph. Sigmoid Plot: do not intersect. Cubic Plot: intersect at one point on the graph. Left-censored: intersect at one point on the graph. The answer is 3.

Qwen: To determine the highest accuracy … Plot (a) VGG: The highest accuracy is 93.4%; Plot (b) ResNet18: 93.0%; Plot (c) ReActNet (ResNet14 as backbone): 94.6%; Plot (d) ReActNet (MobileNet as backbone): 94.6%.

Final Answer: 94.6%

BiPS: <think>To determine how many … the exponential and left-censored plots show the intersection. </think>\boxed{2}

BiPS: <think> To find the highest accuracy … Plot (a) VGG: The accuracy is 93.4%. Plot (b) ResNet18: 93.0%; Plot (c) ReActNet (ResNet14 as backbone): 93.8%; Plot (d) ReActNet (MobileNet as backbone): 94.0%. </think>\boxed{94.0}

Figure 5. Case study on ChartXiv comparing Qwen2.5-VL-7B and our BiPS-Chart. BiPS yields more visually grounded answers.

mantically faithful counterparts for the bi-directional KL constraints. Programmatic editing explicitly isolates the key-evidence and question-unrelated code components, ensuring that Lsep is computed on a pure distraction view. In contrast, random masking drops patches blindly and fails to produce a clean distraction view: if it mostly hides irrelevant regions, the example becomes easier and the prediction gets closer to Ipres, making KL maximization counterproductive; when it also hides task-critical evidence, the masked input becomes a mixture of missing-evidence and residual distractions rather than a pure distractor, so the KL is computed against an ill-defined target and yields noisy gradients. By ensuring clean, semantically controlled counterparts, our programmatic generation provides precise supervision for both Lcons and Lsep, leading to more stable optimization and stronger perceptual grounding.

variants outperform Joint Training on CharXiv and ECDBench (e.g., ours is +3.0% / +3.2% over Joint), suggesting that simultaneously optimizing the guidance (Lcons) and regularization (Lsep) objectives can introduce competing gradient directions and higher update variance under the GRPO policy optimization. Decoupling them into successive stages allows the model to first form a stable, evidencealigned policy before enforcing the grounding constraint, effectively reducing such interference. The training order also matters: our coarse-to-fine schedule consistently surpasses the reversed variant (49.4% vs. 46.8% on CharXiv). We attribute this to the fact that establishing coarse-grained perceptual focus first provides a well-anchored representation aligned with relevant evidence, upon which finegrained grounding can add a discriminative margin and suppress residual shortcuts. In contrast, applying the grounding constraint too early may over-regularize before a reliable focus is established, forcing divergence along under-defined directions and leading to slower or unstable convergence.

Effect of KL Constraint Coefficient. Figure 4 shows the performance impact of varying the consistency (α) and separation (β) constraint coefficients. Both objectives outperform the baseline (coefficient=0) over a wide range, confirming they provide complementary gains without requiring sensitive tuning. Peak accuracy is observed at the moderate coefficients α = 0.01 and β = 0.02, each obtained with the other fixed to 0. Conversely, large coefficients (e.g., 0.08) degrade performance. We attribute this to the auxiliary consistency and separation losses dominating the GRPO objective and increasing update variance.

Impact of Counterpart Generation Strategy. We compare our programmatic code-editing strategy against a random masking baseline. Following prior work [39, 44], this baseline randomly masks out 60% of the image patches to create the alternative view. As shown in Table 4, our programmatic code-editing strategy significantly outperforms the random masking baseline across all benchmarks. These improvements highlight the importance of generating se-

Case Study. Figure 5 presents two representative cases from ChartXiv [43]. In both cases, Qwen2.5-VL-7B produces plausible but incorrect answers by relying on textual or statistical cues rather than interpreting the visual structure. In the intersection-count example, it hallucinates extra curve crossings without actually tracing the plotted functions, while in the multi-plot accuracy example it overfits to frequently occurring numerical patterns (e.g., 94.6%) instead of reasoning over per-panel maxima. In contrast, BiPS yields correct and visually grounded responses. This improvement suggests the bidirectional perceptual shaping encourages the model to attend to the structural alignment and cross-plot relationships that are critical for reasoning, rather than inferring from superficial patterns or numeric priors. Additional cases can be found in the Appendix.

#### 5. Conclusion

In this paper, we proposed BiPS, a training-time framework that turns question-conditioned visual cues into perceptual shaping signals for VLMs. BiPS imposes two complementary KL constraints that pull predictions on the original image toward an evidence-preserving view and push them away from an evidence-ablated view. These signals encourage coarse coverage of supporting regions and fine-grained reliance on visual evidence, improving perception without generating any visual cues at inference. BiPS establishes a new paradigm for multimodal reasoning with three benefits: 1) strong generalization and data efficiency, where using only 13K chart samples yields significant, wide-ranging gains on both chart-specific benchmarks and general VQA; 2) improved fine-grained perception, as demonstrated on chart benchmarks whose complex layouts and thin polylines demand precise visual interpretation; and 3) inference efficiency with no additional test-time overhead.

#### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 1, 2, 4, 5, 6
- [2] Mahtab Bigverdi, Zelun Luo, Cheng-Yu Hsieh, Ethan Shen, Dongping Chen, Linda G Shapiro, and Ranjay Krishna. Perception tokens enhance visual reasoning in multimodal language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 3836–3845, 2025. 1
- [3] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37:27056–27087, 2024. 2, 6
- [4] Lei Chen, Xuanle Zhao, Zhixiong Zeng, Jing Huang, Yufeng Zhong, and Lin Ma. Chart-r1: Chain-of-thought supervision and reinforcement for advanced chart reasoner. arXiv preprint arXiv:2507.15509, 2025. 2, 3, 6

- [5] Shuang Chen, Yue Guo, Zhao yu Su, Yafu Li, Yulun Wu, Jiacheng Chen, Jiayu Chen, Weijie Wang, Xiaoye Qu, and Yu Cheng. Advancing multimodal reasoning: From optimized cold start to staged reinforcement learning. ArXiv, abs/2506.04207, 2025. 2
- [6] Xinyan Chen, Renrui Zhang, Dongzhi Jiang, Aojun Zhou, Shilin Yan, Weifeng Lin, and Hongsheng Li. Mint-cot: Enabling interleaved visual tokens in mathematical chain-ofthought reasoning. ArXiv, abs/2506.05331, 2025. 1, 3
- [7] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198, 2024. 1
- [8] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. ArXiv, abs/2503.21776, 2025. 2
- [9] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models. ArXiv, abs/2306.13394, 2023. 3
- [10] Jinlan Fu, Shenzhen Huangfu, Hao Fei, Xiaoyu Shen, Bryan Hooi, Xipeng Qiu, and See-Kiong Ng. Chip: Cross-modal hierarchical direct preference optimization for multimodal llms. arXiv preprint arXiv:2501.16629, 2025. 3
- [11] Xingyu Fu, Minqian Liu, Zhengyuan Yang, John Corring, Yijuan Lu, Jianwei Yang, Dan Roth, Dinei Florencio, and Cha Zhang. Refocus: Visual editing as a chain of thought for structured image understanding. arXiv preprint arXiv:2501.05452, 2025. 1, 3
- [12] Yucheng Han, Chi Zhang, Xin Chen, Xu Yang, Zhibin Wang, Gang Yu, Bin Fu, and Hanwang Zhang. Chartllama: A multimodal llm for chart understanding and generation. arXiv preprint arXiv:2311.16483, 2023. 6
- [13] Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A Smith, and Ranjay Krishna. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. Advances in Neural Information Processing Systems, 37:139348–139379, 2024. 1, 3
- [14] Muye Huang, Han Lai, Xinyu Zhang, Wenjun Wu, Jie Ma, Lingling Zhang, and Jun Liu. Evochart: A benchmark and a self-training approach towards real-world chart understanding. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 3680–3688, 2025. 6
- [15] Muye Huang, Lingling Zhang, Jie Ma, Han Lai, Fangzhi Xu, Yifei Li, Wenjun Wu, Yaqiang Wu, and Jun Liu. Chartsketcher: Reasoning with multimodal feedback and reflection for chart understanding. ArXiv, abs/2505.19076, 2025. 3
- [16] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749,

2025. 2, 6

- [17] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Weli-

- hinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 6
- [18] Caijun Jia, Nan Xu, Jingxuan Wei, Qingli Wang, Lei Wang, Bihui Yu, and Junnan Zhu. Chartreasoner: Code-driven modality bridging for long-chain reasoning in chart question answering. ArXiv, abs/2506.10116, 2025. 3
- [19] Chengzu Li, Wenshan Wu, Huanyu Zhang, Yan Xia, Shaoguang Mao, Li Dong, Ivan Vuli´c, and Furu Wei. Imagine while reasoning in space: Multimodal visualization-ofthought. arXiv preprint arXiv:2501.07542, 2025. 3
- [20] Junteng Liu, Weihao Zeng, Xiwen Zhang, Yijun Wang, Zifei Shan, and Junxian He. On the perception bottleneck of vlms for chart understanding. arXiv preprint arXiv:2503.18435,

2025. 1

- [21] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. In Annual Meeting of the Association for Computational Linguistics, 2021. 3
- [22] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023. 2, 3, 6
- [23] Ruilin Luo, Zhuofan Zheng, Yifan Wang, Xinzhe Ni, Zicheng Lin, Songtao Jiang, Yiyao Yu, Chufan Shi, Ruihang Chu, Jin Zeng, et al. Ursa: Understanding and verifying chain-of-thought reasoning in multimodal mathematics. arXiv preprint arXiv:2501.04686, 2025. 2
- [24] Yunze Man, De-An Huang, Guilin Liu, Shiwei Sheng, Shilong Liu, Liang-Yan Gui, Jan Kautz, Yu-Xiong Wang, and Zhiding Yu. Argus: Vision-centric reasoning with grounded chain-of-thought. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 14268–14280, 2025. 1, 3
- [25] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq R. Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. ArXiv, abs/2203.10244, 2022. 3
- [26] Ahmed Masry, Mohammed Saidul Islam, Mahir Ahmed, Aayush Bajaj, Firoz Kabir, Aaryaman Kartha, Md Tahmid Rahman Laskar, Mizanur Rahman, Shadikur Rahman, Mehrad Shahmohammadi, et al. Chartqapro: A more diverse and challenging benchmark for chart question answering. arXiv preprint arXiv:2504.05506, 2025. 2, 3, 6
- [27] Ahmed Masry, Abhay Puri, Masoud Hashemi, Juan A Rodriguez, Megh Thakkar, Khyati Mahajan, Vikas Yadav, Sathwik Tejaswi Madhusudhan, Alexandre Pich´e, Dzmitry Bahdanau, et al. Bigcharts-r1: Enhanced chart reasoning with visual reinforcement finetuning. arXiv preprint arXiv:2508.09804, 2025. 3, 6
- [28] Ahmed Masry, Megh Thakkar, Aayush Bajaj, Aaryaman Kartha, Enamul Hoque, and Shafiq Joty. Chartgemma: Visual instruction-tuning for chart reasoning in the wild. In Proceedings of the 31st International Conference on Computational Linguistics: Industry Track, pages 625–643, 2025. 6
- [29] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi,

- Wenhai Wang, Junjun He, et al. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2503.07365, 2025. 2, 4
- [30] Nitesh Methani, Pritha Ganguly, Mitesh M. Khapra, and Pratyush Kumar. Plotqa: Reasoning over scientific plots. 2020 IEEE Winter Conference on Applications of Computer Vision (WACV), pages 1516–1525, 2019. 3
- [31] QwenTeam. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 2
- [32] Gabriel Sarch, Snigdha Saha, Naitik Khandelwal, Ayush Jain, Michael J Tarr, Aviral Kumar, and Katerina Fragkiadaki. Grounded reinforcement learning for visual reasoning. arXiv preprint arXiv:2505.23678, 2025. 3
- [33] Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning. Advances in Neural Information Processing Systems, 37:8612–8642, 2024. 1, 3
- [34] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 2, 4, 5
- [35] Mong Yuan Sim, Wei Emma Zhang, Xiang Dai, and Biaoyan Fang. Can vlms actually see and read? a survey on modality collapse in vision-language models. In Findings of the Association for Computational Linguistics: ACL 2025, pages 24452–24470, 2025. 1
- [36] Alex Su, Haozhe Wang, Weiming Ren, Fangzhen Lin, and Wenhu Chen. Pixel reasoner: Incentivizing pixel-space reasoning with curiosity-driven reinforcement learning. ArXiv, abs/2505.15966, 2025. 3
- [37] Zhaochen Su, Linjie Li, Mingyang Song, Yunzhuo Hao, Zhengyuan Yang, Jun Zhang, Guanjie Chen, Jiawei Gu, Juntao Li, Xiaoye Qu, et al. Openthinkimg: Learning to think with images via visual tool reinforcement learning. arXiv preprint arXiv:2505.08617, 2025. 3
- [38] Liyan Tang, Grace Kim, Xinyu Zhao, Thom Lake, Wenxuan Ding, Fangcong Yin, Prasann Singhal, Manya Wadhwa, Zeyu Leo Liu, Zayne Sprague, et al. Chartmuseum: Testing visual reasoning capabilities of large vision-language models. arXiv preprint arXiv:2505.13444, 2025. 3, 6
- [39] Fei Wang, Wenxuan Zhou, James Y Huang, Nan Xu, Sheng Zhang, Hoifung Poon, and Muhao Chen. mdpo: Conditional preference optimization for multimodal large language models. arXiv preprint arXiv:2406.11839, 2024. 8
- [40] Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vl-rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint arXiv:2504.08837, 2025. 2, 6
- [41] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 37:95095–95169, 2024. 3, 6

- [42] Yaoting Wang, Shengqiong Wu, Yuecheng Zhang, Shuicheng Yan, Ziwei Liu, Jiebo Luo, and Hao Fei. Multimodal chain-of-thought reasoning: A comprehensive survey. arXiv preprint arXiv:2503.12605, 2025. 1
- [43] Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, et al. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. Advances in Neural Information Processing Systems, 37:113569–113697, 2024. 2, 3, 6, 9
- [44] Zhenhailong Wang, Xuehang Guo, Sofia Stoica, Haiyang Xu, Hongru Wang, Hyeonjeong Ha, Xiusi Chen, Yangyi Chen, Ming Yan, Fei Huang, et al. Perception-aware policy optimization for multimodal reasoning. arXiv preprint arXiv:2507.06448, 2025. 3, 8
- [45] Mingyuan Wu, Jingcheng Yang, Jize Jiang, Meitang Li, Kaizhuo Yan, Hanchao Yu, Minjia Zhang, Chengxiang Zhai, and Klara Nahrstedt. Vtool-r1: Vlms learn to think with images via reinforcement learning on multimodal tool use. arXiv preprint arXiv:2505.19255, 2025. 3
- [46] Zhengzhuo Xu, Sinan Du, Yiyan Qi, Chengjin Xu, Chun Yuan, and Jian Guo. Chartbench: A benchmark for complex visual reasoning in charts. ArXiv, abs/2312.15915, 2023. 3
- [47] Cheng Yang, Chufan Shi, Yaxin Liu, Bo Shui, Junjie Wang, Mohan Jing, Linran Xu, Xinyu Zhu, Siheng Li, Yuxiang Zhang, et al. Chartmimic: Evaluating lmm’s cross-modal reasoning capability via chart-to-code generation. arXiv preprint arXiv:2406.09961, 2024. 3
- [48] Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615, 2025. 2, 6
- [49] Yuwei Yang, Zeyu Zhang, Yunzhong Hou, Zhuowan Li, Gaowen Liu, Ali Payani, Yuan-Sen Ting, and Liang Zheng. Effective training data synthesis for improving mllm chart understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2653–2663,

2025. 2, 4, 6, 1

- [50] Zeyuan Yang, Xueyang Yu, Delin Chen, Maohao Shen, and Chuang Gan. Machine mental imagery: Empower multimodal reasoning with latent visual tokens, 2025. 1, 3
- [51] Liang Zhang, Anwen Hu, Haiyang Xu, Mingshi Yan, Yichen Xu, Qin Jin, Ji Zhang, and Fei Huang. Tinychart: Efficient chart understanding with program-of-thoughts learning and visual token merging. In Conference on Empirical Methods in Natural Language Processing, 2024. 3
- [52] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pages 169–186. Springer, 2024. 6
- [53] Shuoshuo Zhang, Zijian Li, Yizhen Zhang, Jingjing Fu, Lei Song, Jiang Bian, Jun Zhang, Yujiu Yang, and Rui Wang. Pixelcraft: A multi-agent system for high-fidelity visual reasoning on structured images. arXiv preprint arXiv:2509.25185, 2025. 1, 3

- [54] Xinchen Zhang, Xiaoying Zhang, Youbin Wu, Yanbin Cao, Renrui Zhang, Ruihang Chu, Ling Yang, and Yujiu Yang. Generative universal verifier as multimodal meta-reasoner. arXiv preprint arXiv:2510.13804, 2025. 2
- [55] Yizhen Zhang, Yang Ding, Shuoshuo Zhang, Xinchen Zhang, Haoling Li, Zhong-zhi Li, Peijie Wang, Jie Wu, Lei Ji, Yelong Shen, et al. Perl: Permutation-enhanced reinforcement learning for interleaved vision-language reasoning. arXiv preprint arXiv:2506.14907, 2025. 2
- [56] Xuanle Zhao, Xianzhen Luo, Qi Shi, Chi Chen, Shuo Wang, Wanxiang Che, Zhiyuan Liu, and Maosong Sun. Chartcoder: Advancing multimodal large language model for chart-tocode generation. ArXiv, abs/2501.06598, 2025. 3
- [57] Jinliang Zheng, Jianxiong Li, Sijie Cheng, Yinan Zheng, Jiaming Li, Jihao Liu, Yu Liu, Jingjing Liu, and Xianyuan Zhan. Instruction-guided visual masking. Advances in neural information processing systems, 37:126004–126031,

2024. 1, 3

- [58] Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing ”thinking with images” via reinforcement learning. ArXiv, abs/2505.14362, 2025. 3, 6
- [59] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 6

## See Less, See Right: Bi-directional Perceptual Shaping For Multimodal Reasoning

### Supplementary Material

#### 6. Implementation Details

- 6.1. Data Statistics

We provide a quantitative breakdown of the data generation pipeline described in the main paper. The construction of our training dataset involves a rigorous filtering and synthesis process to ensure the quality and difficulty of the reasoning tasks. The statistics for each stage are summarized

- in Table 5 and detailed below:

- • Stage 1: Sampling and Reformulation. We initially randomly sampled 50,000 raw chart-code pairs from the ECD [49] dataset. After the Question Reformulation and Validation phase, where the LLM arbitrator (GPT-5-mini) converted open-ended questions into verified multiplechoice formats, approximately 30K valid samples were retained.
- • Stage 2: Difficulty Filtering. To ensure the model learns from non-trivial examples, we filtered the dataset using the base model (Qwen2.5-VL-7B-Instruct). Approximately 10K “easy” samples that answered correctly in all 8 rollouts, were discarded, leaving roughly 20K challenging samples.
- • Stage 3: Code Editing. In this final stage, we performed programmatic editing to generate visual counterparts. We

successfully generated the Evidence-Ablated View (Iabl) for 13K samples. Within this subset, we further successfully synthesized the Evidence-Preserving View (Ipres) for approximately 7K instances.

Consequently, the final high-quality dataset used for BiPS training comprises 13K samples.

Table 5. Statistics of the Data Generation Pipeline. The table tracks the number of samples retained after each processing stage.

Pipeline Stage Sample Count Initial Sampling (from ECD) 50K After Reformulation & Validation ∼30K After Difficulty Filtering ∼20K Final Training Set (Success in Iabl Generation) ∼13K

– subset containing Ipres ∼7K

6.2. Training Details

We detail the hyperparameter configurations for RL training

- in Table 6. Specifically, we employ the AdamW optimizer with a learning rate of 1 × 10−6 and keep the vision tower

unfrozen. The reward is composed of 0.1 for correct formatting and 0.9 for the correct prediction.

Table 6. Hyperparameters for Reinforcement Learning.

###### Hyperparameter Value

Batch Size 256 Learning Rate 1 × 10−6 Optimizer AdamW Freeze Vision Tower False Max Response Length 2,048 KL Divergence Coefficient 0.01 Rollout Number 8 Temperature 0.85

#### 7. Additional Results

##### 7.1. Case Study

As shown in Figure 6, beyond in-domain chart benchmarks, we further evaluate cross-domain transfer on visual counting. The baseline fails due to incomplete object-level reasoning, whereas our model explicitly tracks and subtracts objects to arrive at the correct answer. The stronger performance in this setting indicates improved cross-domain generalization.

Figure 7 illustrates an example of an evidence-preserving view and an evidence-ablated view. This case clearly shows that simple operations such as cropping are insufficient for generating the two views, since the relevant visual evidence can be fine-grained and the associated meta-information may be sparse in the image. Moreover, the evidencepreserving view must retain all visual elements related to the question, including the necessary meta-information, whereas the evidence-ablated view provides a more precise, fine-grained modification that removes only the key visual element (e.g., the target line).

##### 7.2. Qwen3-VL-Thinking Results and Discussion

We further evaluate BiPS in the thinking-mode setting by fine-tuning Qwen3-VL-8B-Thinking on 13K chart samples, with hyperparameters following the same configuration as described in Section 4.1 and Section 6. We still obtain significant improvements on charts and generalization to nonchart domains, which shows that BiPS provides perceptual improvements that complement the model’s strong reasoning capabilities.

BiPS: <think> To solve the problem, we need to follow these steps:

[Figure 55]

Question: Subtract all big red rubber cylinders. Subtract all blue objects. How many objects are left?

1. Identify all the objects in the image. 2. Subtract all big red rubber cylinders. Since there are no big red rubber cylinders in the image, this step does not change the count of objects. 3. Subtract all blue objects. The blue objects in the image are a blue cube and a blue sphere. After removing the blue objects, we are left with the following objects: - Yellow cylinder - Two gray spheres - Red cube Purple sphere - Red cylinder - Silver sphere Counting these, we have 7 objects left. </think> \boxed{7}

Qwen2.5VL-7B-Instruct: There are no big red rubber cylinders in the image. There are two blue objects: a blue cube and a blue reflective sphere. Subtracting these two blue objects leaves us with 6 objects. Final value: 6

[Figure 56]

[Figure 57]

- Figure 6. Cross-domain case on visual counting. The baseline fails due to incomplete object reasoning, whereas BiPS correctly tracks and subtracts objects to obtain the right answer.

[Figure 58]

Original image

Question: What is the cumulative increase in the performance index of 'Material Application' across the six steps?

[Figure 59]

Evidence-Preserving View

[Figure 60]

Evidence-Ablated View

- Figure 7. Evidence-Preserving and Evidence-Ablated views.

report [31]). This controlled sweep accounts for decoding sensitivity and maintains consistency with prior evaluations, and we report the best score within this fixed grid for the base model. The prompts follow those used in Qwen3VL [31]. For MMStar, we remove the duplicate instruction “Please select the correct answer from the options above.” when the question already includes the hint: “Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.” We find that RL fine-tuning on chart samples negatively impacts optical character recognition (OCR) performance on text. This effect can be partially alleviated by adding a system prompt that biases the model toward OCR: “You are a vision-language model. For image-based problems, always prioritize accurate reading of all visible text, symbols, and numbers.” We adopt this prompt for MathVerse, where a subset of examples present the question directly within the image. We use the same prompt across all models for a fair comparison.

Discussion. While BiPS delivers consistently strong indomain gains and robust OOD generalization on Qwen2.5VL-7B, its behavior on the more advanced reasoning model Qwen3-VL-8B-Thinking reflects a more challenging generalization setting. BiPS continues to yield substantial in-domain improvements, while largely maintaining crossdomain performance with moderate OOD gains on most benchmarks. This trend is expected: BiPS is trained exclusively on chart data, and transferring fine-grained perceptual supervision to diverse visual domains becomes inherently harder under strong reasoning priors. Nevertheless, BiPS preserves cross-domain performance, indicating that the learned visual grounding signal remains transferable, although its magnitude is naturally bounded when the target domains differ substantially from the training distribution.

Setting. We use temperature = 1.0, top-p = 0.95, top-k = 20 and presence penalty = 1.2 for the fine-tuned model by default. For the base Qwen3-VL-8B-Thinking model, we keep temperature, top-p, and top-k unchanged, and sweep a small, pre-specified grid of presence penalties: 1.2 (matching the fine-tuned setting) and 1.5 (following the technical

Table 7. Effect of BiPS on Qwen3-VL-8B-Thinking

Model CharXiv ChartQAPro ChartMuseum Evochart MathVista MathVision MathVerse MMStar

Qwen3-VL-8B-Thinking 53.0 54.1 40.4 72.2 81.0 62.2 77.2 75.3 GRPO 54.3 54.6 43.0 74.0 80.4 60.8 76.2 75.3 Ours 58.1 56.8 44.1 75.5 80.4 63.9 77.4 76.3

Extensibility. Beyond chart-centric training, BiPS naturally generalizes as a transferable perceptual supervision mechanism. While chart data alone already yields consistent cross-domain improvements, stronger and more uniform gains can be expected by extending BiPS to multidomain training, for example by 1) mixing heterogeneous data, where non-chart domains are optimized with standard GRPO objectives, or 2) constructing bidirectional views across multiple domains and jointly optimizing them under the BiPS framework. BiPS can be extended to non-chart domains through construction of training views. For example, for natural images, recent visual chain-of-thought pipelines [36] suggest that segmentation tools such as SAM can automatically generate semantic masks, enabling edited views without human annotation. Similarly, procedural domains like Mazes [19] offer precise rendering control, allowing exact synthesis of counterfactual views similar to ours. We leave a systematic exploration of such hybrid and multi-domain settings to future work.

#### 8. Prompts Question Reformulation and Validation

- • System: You are an expert in data analysis and question generation. Task: You will analyze a chart-related questionanswer pair for correctness and potentially rewrite it as a multiple-choice question. Given:

- – Chart metadata and code snippets
- – A problem/question about the chart
- – An answer to that problem Your task:

- 1. Analyze correctness: Determine if both the question and answer are factually correct based on the chart data.
- 2. Generate output:

- – If correct: Rewrite as a multiple-choice question with 3–4 options.
- – If incorrect: Explain the error(s) without rewriting.
- – If uncertain: Explain what information is missing or unclear.

Guidelines:

- – Ensure options are plausible but only one is correct.
- – Include at least 3 options, preferably 4.
- – Distractors should reflect realistic misconceptions.
- – Keep questions clear and unambiguous.
- – Use data directly from provided metadata/code.

###### • User: Chart Metadata: {code} Original Problem: {question} Provided Answer: {answer}

Please analyze the correctness of this questionanswer pair and generate the appropriate output according to the format specified.

###### Evidence-Preserving View

- • System: You are an expert in chart code editing and data visualization. You will receive chart code (Matplotlib/Seaborn/Plotly/Altair) and a question. Your goal is to minimize edits while removing irrelevant elements and preserving layout: figure size, subplot grid, spacing, suptitle, legend order/length, trace order, color assignment, and axis links. Editing Principles:

- – Preserve all layout structure; if hiding content, keep axes and series positions.
- – Never change plotting library; only minimal imports for placeholders are allowed.
- – Keep legend/trace counts; for removed series use placeholders (e.g., NaNs, transparent marks, or “legendonly”).
- – Be careful not to let the model derive the answer directly from the remaining elements; keep the necessary distractors.
- – Maintain axis limits when feasible to avoid scale drift.

- • Decision Rules:

- – Subplot-specific questions: Keep only referenced subplots; blank others but preserve axes.
- – Legend/category-specific questions: Keep only mentioned categories; others become

- placeholders.
- – Series/trace-specific questions: Keep only targeted lines/bars/points; blank others.
- – Global comparison or vague questions: Do not edit.
- – If uncertain, set should edit = false.

- • Post-edit requirements:

- – Subplot grid unchanged; all axes preserved.
- – Legend length and order unchanged.
- – Placeholders inserted for every removed subplot/series.
- – Axis ranges preserved when appropriate.
- – Output must be JSON only.

- • User: Chart Code: {code} Question: {problem str} Please determine whether the chart code should be edited to remove irrelevant elements according to the rules above.

Evidence-Ablated View

- • System: You are an expert in chart code obfuscation for evaluation/red-teaming. Given chart code (Matplotlib/Seaborn/Plotly/Altair) and a question, your task is to make the question unanswerable by removing/blanking decisive evidence while preserving layout. Objectives (priority order):

- 1. Ensure unanswerability: blank all chart elements that allow a definitive answer.
- 2. Preserve layout: keep figure size, subplot grid, spacing, legend structure, series order, and color assignments.
- 3. Minimize edits: hide or blank evidence without refactoring or adding new content.

- • Decisive Evidence (to be blanked):

- – Any subplot targeted or compared by the question.
- – Legend/categories mentioned or implied by the question/options.
- – Series/traces/marks revealing values, trends, peaks, ranks, or comparisons.
- – Numeric cues: labels, annotations, thresholds, reference lines.
- – Axes information that allows inference once geometry is hidden.
- – If unsure, treat as decisive (favor overblanking).

- • Blanking Tactics (by library):

– Matplotlib: Replace data with NaNs, set in-

visible while keeping legend handles, or use dummy Line2D. For bars/scatter: empty/NaN arrays or alpha=0. For entire subplots: ax.cla(); ax.set axis off().

- – Plotly: Keep trace but hide geometry via visible=’legendonly’ or empty x/y while keeping showlegend=True.
- – Altair: Keep encodings and legend domain; blank via opacity=0 or empty filters.

###### • Decision Rules:

- – Options provided: blank all optionreferenced elements.
- – Comparisons/ranking/extremes: blank all compared candidates.
- – Single-target lookup: blank the target’s geometry and any revealing annotation.
- – Global comparisons: blank decisive evidence across all involved candidates.
- – Trend/correlation: blank scatter points and trend/regression lines.
- – Threshold questions: blank values and relevant threshold lines.

###### • Post-edit Requirements:

- – Subplot grid unchanged; axes preserved.
- – Legend length/order preserved (dummy placeholders allowed).
- – Placeholders inserted for all blanked series/subplots.
- – Axis ranges preserved when applicable; code must run.
- – Remaining visuals must not allow a human to answer the question.

###### • User: Chart Code: {code} Question: {problem str} Your task: Make the question unanswerable by blanking all decisive evidence while preserving layout.

