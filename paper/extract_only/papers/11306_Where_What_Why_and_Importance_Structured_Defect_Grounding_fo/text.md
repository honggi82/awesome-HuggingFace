# arXiv:2606.06113v2[cs.CV]11Jun2026

## Where, What, Why, and Importance: Structured Defect Grounding for Text-to-Image Feedback

Huaisong Zhang1,2,*,§ Hao Yu1,* Yuxuan Zhang2,3,4,§ Jiahe Wang1 Xinrui Chen1 Haoxiang Cao2,5,§ Feng Lu1 Wendong Zhang2,† Changqian Yu2,‡ Chun Yuan1,† 1Tsinghua University 2Kolors Team, Kuaishou Technology 3University of British Columbia 4Vector Institute 5South China Normal University

### Abstract

Despite generating increasingly photorealistic images, text-to-image (T2I) models still exhibit localized, subtle, and structurally complex failures. Diagnosing these failures requires instance-level feedback that answers where a defect occurs, what type it is, why it is defective, and its importance to overall image quality. While recent dense-feedback methods move beyond scalar supervision, their heatmapcentric representations still formulate diagnosis as pixel-field regression, making it difficult to localize variable-cardinality defects and bind semantic reasons to individual failures. To address this representation bottleneck, we propose Structured Defect Grounding (SDG), which casts T2I diagnosis as structured set prediction by modeling each defect as a (location, type, reason, importance) tuple. To make this formulation trainable and measurable, we introduce SDG-30K, a 30K-image dataset with box-grounded annotations across four modern T2I generators, together with a dedicated evaluation protocol, SDG-Eval. Building on this structured representation, we further present a diagnosis-to-alignment framework in which a Vision-Language Model (VLM) serves as the SDG detector, and BoxFlow-GRPO converts predicted defect sets into box-derived, importance-weighted spatial rewards for diffusion model alignment. Extensive experiments show that our SDG detector outperforms leading proprietary VLMs on structured defect grounding, while SDG-guided rewards consistently improve T2I alignment and support localized image refinement. These results establish SDG as a unified, instance-level interface for diagnosing, evaluating, and enhancing modern generative models.

### 1 Introduction

Text-to-image (T2I) models [Esser et al., 2024, Betker et al., 2023, Labs, 2024, Cai et al., 2025, Ma et al., 2025a, Team et al., 2025] produce increasingly photorealistic images, yet their failures often remain localized, subtle, and structurally heterogeneous, such as malformed text, implausible geometry, and semantic mismatches. Evaluating such outputs with scalar preference scores [Wu et al., 2023b,a, Xu et al., 2023, Kirstain et al., 2023] collapses these defects into a single global value. Although useful for ranking images, scalar feedback cannot answer the four key questions required for defect diagnosis: where the error occurs, what type of defect it is, why the region is defective, and its importance to overall image quality. This limitation reveals a mismatch between global image-level supervision and the localized, instance-level nature of T2I failures.

Recent work attempts to address this need by moving beyond scalar supervision toward dense feedback. RichHF [Liang et al., 2024] introduces artifact and misalignment heatmaps, while ImageDoctor [Guo et al., 2025] augments heatmap-based diagnosis with VLM reasoning. However,

*Equal contribution. ‡Project lead. †Corresponding authors. §Work done during internship in Kolors Team, Kuaishou Technology.

Preprint.

###### Generated Image

###### SDG Structured Feedback (Ours)

###### Heatmap Feedback

<think>

[Figure 1]

[Figure 2]

###### T2I Prompt

- 1. Prompt Understanding: The caption describes a "petite American ballerina" …
- 2. Defect Spotting: Ballerina stands upright on board; Both hands show fused; Subtle distortion in eye and mouth alignment.
- 3. Localization: Box 1 (Artifact): The defect is located on the raised hand on the image-right side. Box 2 (Artifact): … Box 3 (Artifact): … Box 4 (Misalignment): …

Petite American ballerina in a sparkling lace tutu, riding a glowing neon long skateboard, upside-down star jump, backlit with smoke, explosion, and fire twirling

- A1 Box: [600,48,641,100] Label: Artifact Description: fused fingers on raised hand Importance: 22/100

- A2 Box: [223,345,285,385] Label: Artifact Description: malformed fingers on extended hand Importance: 22/100

- A3 Box: [474,220,530,281] Label: Artifact Description: subtle facial distortion Importance: 32/100

[Figure 3]

[Figure 4]

No output

T2I Generated Image

Artifact

Misalignment

[Figure 5]

Textual Feedback: The ballerina is not performing a star jump. The ballerina’s anatomy is distorted, with an unnaturally long left arm, a deformed right hand, and misshapen feet; The skateboard also appears bent and slightly melted.

M1 Box: [223,48,657,840] Label: Misalignment Description: upright pose contradicts “upside-

down star jump” Importance: 95/100

Region size tied to annotation radius Struggle with large-area defects Hard to bind reasons to specific regions Not natively consumable by VLMs

[Figure 6]

[Figure 7]

Box boundary matches defect extent Handles large misalignments as explicit instances Each instance carries its own reason Native structured output for VLMs

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

- Figure 1: Comparison of dense feedback paradigms for T2I defect diagnosis. Left: a T2I generated image with its prompt. Middle: heatmap-style feedback produces separate artifact and misalignment maps with textual feedback. Right: SDG reasons via a chain-of-thought (CoT) trace, then outputs structured defects with bounding boxes, types, descriptions, and importance scores.

although heatmaps provide dense signals, they still formulate defect diagnosis as pixel-field regression rather than instance-level understanding. As illustrated in Figure 1, this heatmap-centric paradigm exposes representation bottlenecks: spatially, point-derived maps depend on annotator-chosen radii rather than true defect extents [Liang et al., 2024]; semantically, continuous severity fields cannot bind defect types, reasons, or importance scores to individual failures; and architecturally, pixel-level maps are not native outputs of autoregressive VLMs and often require additional decoders or regression heads. These limitations suggest that the key challenge is not merely to make feedback dense, but to represent defects as spatially grounded, semantically explicit, and VLM-compatible instances.

Motivated by this bottleneck, we move from continuous pixel-field regression to structured instance prediction. We introduce Structured Defect Grounding (SDG), which models each defect as a structured (location, type, reason, importance) tuple and formulates diagnosis as predicting a variable-length set of such tuples. SDG unifies two defect types, artifacts (i.e., image-intrinsic visual flaws) and misalignments (i.e., prompt-conditioned semantic errors), within a single instance space, naturally supporting images with multiple heterogeneous defects. To make this formulation trainable and measurable, we construct SDG-30K, a 30,096-image dataset with box-grounded artifact and misalignment annotations across four modern T2I generators, and define SDG-Eval, a dedicated protocol for structured defect-set evaluation.

Beyond diagnosis, SDG also provides a natural interface for model alignment: boxes define spatial support, defect types and reasons provide semantic diagnoses, and importance scores calibrate reward strength. We convert SDG predictions into box-derived, importance-weighted spatial reward maps and post-train diffusion models with our proposed BoxFlow-GRPO, enabling spatially targeted alignment beyond scalar preference optimization. The code, model weights, and dataset are available at https:

//github.com/nianbai006/SDG. Overall, this paper makes following main contributions:

- • We introduce Structured Defect Grounding (SDG), an instance-level representation that formulates dense T2I diagnosis as variable-cardinality set prediction over structured (location, type, reason, importance) tuples.
- • We construct SDG-30K, a 30,096-image box-grounded defect dataset across four modern T2I generators, and define SDG-Eval for image-level and defect-level evaluation.
- • We develop a diagnosis-to-alignment framework where a VLM-based SDG detector predicts structured defect sets and BoxFlow-GRPO converts them into importance-weighted spatial rewards for diffusion model alignment.
- • Extensive Experiments demonstrate that our SDG detector outperforms leading proprietary VLMs on structured defect grounding, while SDG-guided rewards improve T2I alignment and support more faithful, actionable image refinement.

[Figure 14]

[Figure 15]

[Figure 16]

###### SDG-30K Dataset Construction Detector Training Downstream Applications

###### Cold Start

BoxFlowGRPO

Box-derived Spatial Reward

Advantage Map

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

⚡

[Figure 25]

Image Generation

Flux1.dev

[Figure 26]

[Figure 27]

[Figure 28]

coordinate

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

SDG Detect

[Figure 33]

[Figure 34]

Flux2 Z-Image LongCat SANA

Pick-a-Pic Prompts

Qwen3VL-4B

jitter

SDG-30K

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Box Label Short Desc. 112 annotators

###### Composite-Reward GRPO

Human Annotation

Two-Round QC

𝐴 (ℎ,𝑤)

Policy Optimization

[Figure 39]

𝑅

Defect-Guided Image Refinement

𝑅

[Figure 40]

[Figure 41]

Desc. Expansion

Short Desc. Long Desc.

Policy model

…

Format check

Reward

𝑅

[Figure 42]

[Figure 43]

|Box Box|Label Label| |
|---|---|---|
|Desc. Desc.|Imp. Imp.| |
| | | |

[Figure 44]

[Figure 45]

###### Gemini 3 Pro Enhancement

[Figure 46]

- 1. Prompt Understanding
- 2. Defect Spotting
- 3. Localization

[Figure 47]

[Figure 48]

𝑅

CoT Distillation

SDG Model

GPT Image

[Figure 49]

SDG-30K

Gemini 3 Pro

T2I prompt

[Figure 50]

Policy Optimization

Importance Scoring

[Figure 51]

CoT Box Label Long Desc. Imp.

1-100

- Figure 2: Overview of the SDG framework. Left: SDG-30K construction combines human box-level defect annotation across four T2I generators with Gemini 3 Pro enhancement. Middle: Twostage SDG detector training via SFT with coordinate jitter followed by GRPO with a format-gated composite reward. Right: Downstream applications. BoxFlow-GRPO converts SDG detections into box-derived spatial rewards for diffusion alignment; defect-guided refinement feeds box overlays and text feedback to GPT-Image-1.5 for image refinement.

### 2 Related Work

From scalar evaluation to dense T2I feedback. Most T2I evaluators produce scalar scores [Wu et al., 2023b,a, Xu et al., 2023, Kirstain et al., 2023]. RichHF [Liang et al., 2024] introduces heatmapbased dense feedback, ImageDoctor [Guo et al., 2025] predicts heatmaps via a VLM-plus-decoder architecture, and HEIE [Yang et al., 2025] and MagicMirror [Wang et al., 2025a] further enriches feedback with hierarchical explanations and fine-grained artifact taxonomy, respectively. Beyond T2I evaluation, HAD [Wang et al., 2024] and AbHuman [Fang et al., 2024] demonstrate box-level supervision for human-body artifact detection, and LEGION [Kang et al., 2025] combines mask-level localization with natural-language explanations. However, they do not provide a unified instance-level formulation that jointly localizes artifact and misalignment with localized descriptions.

Structured spatial reasoning in VLMs. Modern VLMs increasingly support explicit spatial structure within an autoregressive generation framework. Qwen2.5-VL [Bai et al., 2025b] supports box- and point-based grounding with structured outputs, while Qwen3-VL [Bai et al., 2025a] further strengthens image-grounded reasoning and spatial understanding. At a finer granularity, SimpleSeg [Song et al., 2026] reformulates segmentation as point-sequence generation entirely in language space. These advances motivate our use of VLMs to generate structured defect instances.

RL for diffusion alignment and image refinement. RL has become a key tool for aligning diffusion models with human preferences. DDPO [Black et al., 2023] frames denoising as a multi-step MDP and applies policy gradients, Diffusion-DPO [Wallace et al., 2024] adapts direct preference optimization to diffusion training, and Flow-GRPO [Liu et al., 2025] applies group relative policy optimization [Shao

- et al., 2024] to flow-matching models. ImageDoctor [Guo et al., 2025] proposes extending scalar rewards to spatially varying dense reward maps. On the correction side, ReflectionFlow [Zhuo et al., 2025] and HumanRefiner [Fang et al., 2024] demonstrate that defect-guided refinement can improve generation quality. To our knowledge, our work is the first to realize spatially dense advantages in diffusion RL, and further uses structured dense feedback to guide image refinement.

### 3 SDG-30K: Dataset and Evaluation

#### 3.1 Task Formulation

Given a generated image I and its prompt T, the goal is to predict a variable-cardinality set of structured defect instances {(bi,ti,ri,si)}Ki=1. Here, K ∈ N denotes the number of defects in the image. The i-th instance consists of (1) a quantized bounding box bi ∈ {0,...,1000}4; (2) a defect type ti ∈ {artifact, misalignment}; (3) a free-form natural-language description ri; and (4) an integer importance score si ∈ {1,...,100} reflecting the defect’s perceptual impact on the overall image quality. The detailed definitions and distinctions of these defect types are provided in Appendix A.

- Table 1: Comparison with related datasets. SDG-30K is the first to jointly cover artifact and misalignment within a unified instance-level annotation space with natural-language reasons.

Dataset Images Spatial Ann. Defect Types NL Reason Instance-Level Generators RichHF-18K [Liang et al., 2024] 18K Heatmap (point-derived) Art. + Mis. × × SD-family HAD [Wang et al., 2024] 37K Bounding box Artifact (human body) × ✓ SDXL, DALL-E, MJ AbHuman [Fang et al., 2024] 56K Bounding box Artifact (human body) × ✓ SDXL MagicMirror [Wang et al., 2025a] 343K Taxonomy label Artifact × × FLUX, SD3, Kolors, MJ, etc. LEGION [Kang et al., 2025] 12K Pixel mask Artifact ✓ ✓ SD, FLUX, GLIDE, etc. SDG-30K (Ours) 30K Bounding box Art. + Mis. ✓ ✓ FLUX, Z-Image, LongCat, SANA

(a) Defect Density

###### (b) Image Composition

###### (c) Importance Distribution

SANA-1.5

SANA-1.5

11

49

8

32

- 3

7

5

- 4

20

51

23

1

3.22

Avg.Defects/Image

- 0

- 1

- 2

- 3

LongCat

LongCat

24

34

17

25

17

39

- 33
- 34

4

1.63

1.30

Z-Image

Z-Image

28

26

25

21

18

37

5

1.15

0.58 0.53 0.50

0.35

FLUX.2

FLUX.2

38

33

15

14

15

39

37

6

FLUX.2 Z-Image LongCat SANA-1.5

0 20 40 60 80 100

0 20 40 60 80 100

Images (%)

Defect Distribution (%)

Misalignment

Artifact

Clean

Mis. only Both

Critical (90 100)

Moderate (40 69) Minor (15 39)

Negligible (1 14)

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Art. only

Major (70 89)

| |
|---|

| |
|---|

- Figure 3: SDG-30K dataset statistics across four generators. (a) Average number of defects per image by type. (b) Image composition showing the proportion of clean, artifact-only, misalignment-only, and both-type images. (c) Distribution of defect importance scores across five severity tiers.

#### 3.2 Evaluation Metrics

We define SDG-Eval, an evaluation protocol that scores SDG at both image and defect levels for each defect type t ∈ {artifact,misalignment}; full definitions are provided in Appendix C. At the image level, DetTypeF1 measures whether a defect type is present on each test image, while ClnAcc reports true-negative accuracy on images with no ground-truth instance of that type (GT=0). At the defect level, predictions are matched to ground-truth instances of the same type via class-aware Hungarian matching by IoU. We report BoxF1@0.1 and BoxF1@0.5 for localization, DescCos@0.1 using Qwen3-Embedding-0.6B for description alignment, and ImpAcc@0.1 as normalized importance accuracy over matched pairs.

#### 3.3 Dataset Construction

Figure 2 (left) illustrates the dataset construction pipeline. SDG-30K contains 30,096 images at 1024×1024 resolution, generated from Pick-a-Pic prompts [Kirstain et al., 2023] using four T2I generators (∼7.8K each): FLUX.2-dev [Labs, 2024], Z-Image-Turbo [Cai et al., 2025], LongCatImage [Ma et al., 2025a], and SANA-1.5-1.6B [Team et al., 2025].

112 annotators (∼1,085 person-hours) examined each prompt–image pair, drew bounding boxes, assigned top-level labels, and wrote concise Chinese descriptions (≤30 characters). Two rounds of review resolved disagreements. To quantify inter-annotator agreement, 16 held-out annotators who did not participate in the original labeling independently re-annotated the test set. The resulting BoxF1@0.5 of 0.278 (artifact) and 0.409 (misalignment) against the primary annotations serves as a human upper bound for localization (Table 2). Detailed annotation rules are in Appendix A.

Human annotations provide boxes, labels, and short Chinese descriptions but lack reasoning traces and importance scores. We use Gemini 3 Pro [Team et al., 2023] to augment each sample with three components: (1) description expansion: from Chinese to detailed English; (2) reasoning trace distillation: as a three-step CoT trace (prompt understanding, defect spotting, localization); and (3) importance scoring: based on a multi-criteria rubric. Details are in Appendix B.2. Table 1 compares SDG-30K with related datasets.

#### 3.4 Statistics and Splits

Dataset split. The split follows the Pick-a-Pic prompt partition (prompt-disjoint): 28,945 training and 1,151 test images, ensuring that no prompt appears in both splits.

Defect frequency and image composition. Figure 3 summarizes the dataset statistics. Artifact instances are more frequent than misalignment across all four generators, with SANA-1.5 exhibiting the highest artifact frequency (3.22 per image). Of all images, 25.1% are defect-free, 46.3% artifact-

only, 5.4% misalignment-only, and 23.2% both. This confirms that localized failures remain common in recent T2I generators, motivating dense, instance-level feedback.

Importance distribution. Importance scores estimate each defect’s impact on image quality and prompt faithfulness, enabling failure prioritization and reward weighting. They are centered around the Moderate tier (40–69), mostly between 30 and 80 (Figure 3c), with rare extremes and balanced coverage for training.

### 4 Structured Defect Grounding Framework

In this section, we instantiate SDG as a diagnosis-to-alignment framework built around the structureddefect representation. We first present the SDG detector, which produces structured defect sets that inherently align with VLM output formats. We then introduce BoxFlow-GRPO to convert SDG’s outputs into box-derived spatial rewards for diffusion alignment.

#### 4.1 SDG Detector

We formulate defect diagnosis as a structured vision-language generation task. Given a generated image and its prompt, SDG first produces a reasoning trace R and then emits a structured defect set D, where each instance specifies its location, type, reason, and importance.

The SDG detector is trained in two stages (Figure 2, middle). Supervised fine-tuning (SFT) teaches the model to follow the defect-grounding instruction and emit the required structured format, while group relative policy optimization (GRPO) [Shao et al., 2024] further improves localization, description consistency, and importance estimation under a format-validity gate.

Cold Start. We fine-tune Qwen3-VL-4B-Instruct on the SDG-30K training split (Section 3.3). For each image-prompt pair (I,T), the target sequence concatenates the Gemini-distilled reasoning trace and the structured defect set, denoted as y = [R;D]. We optimize the model with the standard teacher-forced negative log-likelihood over the target tokens:

|y|

LSFT = −

t=1

log πθ(yt | I,T,y<t). (1)

To reduce sensitivity to exact coordinate values, we apply coordinate jitter to the <answer> segment during SFT data loading. For each box [x0,y0,x1,y1] in normalized [0,1000] space, each coordinate is independently perturbed by δ ∼ U(−10,10), then clamped to [0,1000] with valid box ordering enforced. The <think> segment is kept unchanged because it captures high-level visual reasoning rather than exact numerical coordinates. Since jitter offsets are resampled across epochs, the model observes multiple plausible coordinate variants of the same defect, improving tolerance to minor spatial variation and providing a more robust SFT initialization for subsequent GRPO training.

Composite-Reward GRPO. After SFT, we apply GRPO to directly optimize the structured output using rewards for spatial accuracy, description consistency, and importance estimation. For each prompt, we sample S=8 responses, compute the composite reward Rs for each response, and form group-normalized advantages As = (Rs − R¯)/σR. The policy is then optimized with the clipped GRPO objective:

LGRPO = −E[min(ρsAs, clip(ρs,1−ϵ,1+ϵ)As)] + β KL(πθ ∥πref), (2)

where ρs = πθ(ys | I,T)/πold(ys | I,T) is the importance ratio, πold is the rollout policy, πref is the fixed reference policy, and β=0.01.

The composite reward is gated by a format-validity check:

R =

λloc Rloc + λdesc Rdesc + λimp Rimp, if Format(y) = true, Rfail, otherwise,

(3)

where λloc,λdesc,λimp ≥ 0 with λloc + λdesc + λimp = 1 control the relative importance of each reward component, and Rfail < 0 is a fixed penalty for malformed outputs. The format predicate Format(y) verifies that the response contains well-formed reasoning and answer delimiters, a parseable JSON defect list, and geometrically valid bounding boxes (i.e., x0 < x1 and y0 < y1).

The three reward components are defined as follows. Rloc measures spatial grounding accuracy: predicted and ground-truth boxes are matched via the Hungarian algorithm with Distance-IoU (DIoU) cost, and the reward reflects match quality with explicit penalties for false negatives and false positives. Rdesc measures description consistency via embedding cosine similarity (Qwen3-Embedding-0.6B), linearly mapped to [0,1] and averaged over matched pairs. Rimp measures importance estimation accuracy via a clipped absolute-error metric. In practice, we set λloc=0.6, λdesc=0.25, λimp=0.15, and Rfail=−1. Detailed formulas, boundary-case handling are provided in Appendix D.

#### 4.2 SDG-Guided BoxFlow-GRPO

Once the SDG detector is trained, we translate its dense feedback into spatially varying rewards to guide the reinforcement learning of diffusion models. The closest prior work is ImageDoctor’s DenseFlow-GRPO [Guo et al., 2025], which extends Flow-GRPO [Liu et al., 2025] with a heatmap signal. In its public training code1, each image is assigned an image-level scalar reward R, normalized across the prompt group into a scalar advantage A; the predicted heatmap H ∈ [0,1]H×W enters only as a multiplicative mask on the policy-gradient loss, L ∝ −A · ρ · (1−H), where ρ is the importance ratio. The gradient signal is therefore still driven by an image-level scalar, with the heatmap merely down-weighting locations the detector flags as defective rather than contributing a true per-location advantage at the latent grid.

Motivated by this gap, we implement BoxFlow-GRPO around two design choices: box-derived spatial rewards and spatially normalized per-location advantages.

Given a base scalar reward R from any predefined reward model2 and the defect bounding boxes detected by our SDG model, we construct a spatially varying reward map in the latent space. For each latent spatial location (h,w), weighted masks Wart,Wmis aggregate the predicted importance of all defect boxes covering that location, and the per-location reward subtracts type-specific penalties from the scalar reward:

Wtype(h,w) = max

sˆk/100,

k∈Btype(h,w)

(4)

αart = cart · σR(group), αmis = cmis · σR(group), RD(h,w) = R − αartWart(h,w) − αmisWmis(h,w).

where Btype(h,w) denotes the set of boxes of a given type covering location (h,w) and sˆk ∈ {1,...,100} is the importance predicted by SDG for box k (Section 3.2). We set Wtype(h,w) = 0 when no box of that type covers (h,w). We set cart=0.5 and cmis=0.05, making the penalties adaptive to the prompt-group reward standard deviation. This formulation ensures that high-importance defects receive proportionally stronger spatial penalties, while minor defects exert lighter corrections.

After constructing RD, per-location advantages are computed by normalizing across the K samples within each prompt group at each spatial location:

RD(k)(h,w) − µD(h,w) σD(h,w) + ϵ

A(Dk)(h,w) =

, (5)

where µD(h,w) and σD(h,w) are the mean and standard deviation of {RD(k)(h,w)}Kk=1 computed over the K=8 samples in the prompt group, and ϵ is a small constant for numerical stability. Defining

the per-location likelihood ratio ρ(tk)(h,w) = πϕ(x(t−k)1,h,w | x(tk),c)/πϕ

(x(t−k)1,h,w | x(tk),c), the BoxFlow-GRPO objective is (KL regularization omitted for brevity):

old

1 KTHW k,t,h,w

min ρ(tk)(h,w)A(Dk)(h,w), clip ρ(tk)(h,w),1−ε,1+ε A(Dk)(h,w)

JBoxFlow(ϕ) =

(6) where T is the number of denoising steps and H,W are the latent spatial dimensions. Compared with a scalar-advantage implementation, this objective preserves spatial variation in both the advantage and the likelihood ratio.

1https://github.com/EthanG97/ImageDoctor 2We use UnifiedReward-2.0 (UR2) [Wang et al., 2025b] in our experiments

### 5 Experiments

We evaluate SDG from three complementary perspectives: (1) defect grounding quality on SDG-30K, (2) diffusion alignment with structured-feedback rewards via BoxFlow-GRPO, and (3) defect-guided image refinement driven by structured feedback.

- 5.1 Defect Grounding Results

- 5.1.1 Setup

Implementation. We fine-tune Qwen3-VL-4B-Instruct on 16 GPUs with DeepSpeed ZeRO-2 for 3 epochs (effective batch size 16, learning rate 3×10−5, cosine schedule). The vision encoder is frozen throughout SFT; Table 4 shows that unfreezing it degrades grounding quality at this dataset scale. For GRPO, we train on 16 GPUs with S=8 sampled responses per prompt (temperature 1.0, top-p 0.85) for 2 epochs at learning rate 5×10−6.

Baselines. We compare SDG (SFT and GRPO variants) against zero-shot GPT-5.4 and Gemini 3 Pro, both prompted with the same structured output format but without task-specific training. A human reference from 16 independent re-annotators provides a localization upper bound.

#### 5.1.2 Main Results

- Table 2: Defect grounding results on SDG-30K test set. Human row: localization upper bound from 16 independent re-annotators. Bold: best; underline: second best.

Artifact Misalignment Image-Level Defect-Level Image-Level Defect-Level All GT=0 GT>0 All GT=0 GT>0

Model DetTypeF1 ClnAcc BoxF1@0.1 BoxF1@0.5 DescCos@0.1 ImpAcc@0.1 DetTypeF1 ClnAcc BoxF1@0.1 BoxF1@0.5 DescCos@0.1 ImpAcc@0.1

GPT-5.4 0.736 0.118 0.237 0.035 0.868 0.812 0.654 0.513 0.457 0.292 0.837 0.820 Gemini 3 Pro 0.733 0.076 0.337 0.200 0.885 0.824 0.696 0.672 0.464 0.307 0.891 0.851 SDG (SFT) 0.776 0.697 0.402 0.255 0.904 0.883 0.636 0.799 0.499 0.376 0.893 0.892 SDG (GRPO) 0.772 0.560 0.404 0.263 0.904 0.887 0.675 0.732 0.511 0.387 0.888 0.893 Human 0.792 0.774 0.462 0.278 – – 0.712 0.818 0.563 0.409 – –

Quantitative analysis. Table 2 presents defect grounding results for artifact and misalignment separately. GRPO achieves the strongest BoxF1@0.5 (0.263/0.387) and importance accuracy (0.887/0.893), while SFT remains competitive and attains the highest clean-image accuracy (0.697/0.799). Compared with zero-shot VLM baselines, both SDG variants substantially reduce localization error: GPT-5.4 obtains reasonable image-level artifact detection (F1 0.736) but weak precise artifact localization (BoxF1@0.5 0.035), and Gemini 3 Pro improves localization (0.200/0.307) but remains below SDG. GRPO narrows the gap to the human reference on both artifact (0.263 vs. 0.278) and misalignment (0.387 vs. 0.409), while maintaining high description cosine similarity.

Qualitative comparison. Figure 4 shows qualitative comparisons on SDG-30K across three representative cases: an artifact-only image (top), a misalignment-only image (middle), and a clean image (bottom). SDG produces instance-level bounding boxes with per-defect descriptions, accurately localizing defects in the first two rows and correctly recognizing the clean case as defect-free. In contrast, ImageDoctor tends to predict heatmap responses on faces and hands even when they are anatomically correct, and struggles to detect prompt-conditioned misalignments.

Cross-dataset generalization. To evaluate generalization, we test SDG on the RichHF-18K test set without any fine-tuning on this dataset, comparing against ImageDoctor [Guo et al., 2025] which is trained on RichHF-18K. Since RichHF-18K provides heatmap annotations, we threshold ImageDoctor heatmaps at two operating points (0.10 and 0.33) for image-level evaluation.

As shown in Table 3, SDG achieves substantially higher misalignment F1 (0.655 vs. 0.250/0.007), demonstrating that structured defect grounding generalizes to unseen data and captures promptconditioned misalignments more effectively than heatmap-based methods. ImageDoctor achieves higher artifact F1 at the loose threshold (0.952), as expected from in-domain training, but its misalignment recall is notably poor (0.143/0.004), suggesting that heatmap-based representations struggle with prompt-conditioned misalignment in this setting.

ImageDoctor Misalignment

ImageDoctor Artifact

Original

GT SDG (ours)

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

Artifact

| |
|---|

| |
|---|

[Figure 59]

male woodland elf stalking through tall grass

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

| |
|---|

| |
|---|

Misalignment

Mario wearing a skull hat, glowing eyes

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Clean

A little girl lost in the forest

- Figure 4: Qualitative comparison on SDG-30K. Rows from top to bottom: artifact-only, misalignmentonly, and clean images.

- Table 3: Image-level defect detection on RichHF-18K. ImageDoctor is trained in-domain; SDG is zero-shot.

Artifact Misalignment Method Prec. Rec. F1 Prec. Rec. F1 ImageDoctor @0.10

| | |
|---|---|
| | |

0.965 0.939 0.952

| | |
|---|---|
| | |

0.983 0.143 0.250 ImageDoctor @0.33 0.977 0.762 0.856 1.000 0.004 0.007 SDG (ours) 0.970 0.698 0.812 0.951 0.500 0.655

Table 4: Key ablation results on SDG30K. Full metrics are in Table 7.

Variant Art. BoxF1@0.5 Mis. BoxF1@0.5

SFT 0.255 0.376 GRPO 0.263 0.387 SFT w/o step 1 0.218 0.329 SFT w/o step 3 0.253 0.281 GRPO w/o CoT 0.288 0.352 SFT unfreeze ViT 0.203 0.305 SFT w/o jitter 0.253 0.360

- 5.1.3 Ablation Study

- Table 4 summarizes the main ablation trends, with full metrics in Appendix E. GRPO improves localization over SFT (artifact/misalignment BoxF1@0.5: 0.263/0.387 vs. 0.255/0.376), confirming that policy optimization refines spatial precision. Removing CoT steps hurts prompt-conditioned misalignment more than artifact detection, and removing the reasoning trace from GRPO lowers misalignment BoxF1@0.5 from 0.387 to 0.352. Unfreezing the vision encoder substantially degrades localization, while coordinate jitter mainly improves image-level robustness (full results in Table 7).

5.2 Downstream Applications

- 5.2.1 BoxFlow-GRPO

Setup. We apply the dense reward construction of Section 4.2 to FLUX.1-dev, following FlowGRPO [Liu et al., 2025]. The scalar base reward is UnifiedReward-2.0 (UR2) [Wang et al., 2025b]; SDG detections convert it into a dense reward by subtracting importance-weighted artifact and misalignment penalties at covered latent locations. Training uses Pick-a-Pic [Kirstain et al., 2023] prompts held out from SDG-30K (prompt-disjoint), for 500 optimization steps at 512×512 resolution on 8 GPUs with LoRA (rank 64, α=128) and learning rate 3×10−4. We evaluate on DrawBench [Saharia et al., 2022] using PickScore [Kirstain et al., 2023], CLIPScore [Hessel et al., 2021], HPSv3 [Ma et al., 2025b], DeQA [You et al., 2025], and the real-image probability P(real) from Forensic-Chat [Lin

- et al., 2025], obtained by a 2-class softmax over its “real” vs. “fake” token logits.

- Results. As shown in Table 5, baseline RL variants tend to drift toward more illustration- or animelike outputs after training, a shortcut that can increase reward-model scores without preserving photographic realism. This reward hacking is reflected by the drop in P(real) for all baselines

FlowGRPO UR2

DenseFlow-GRPO ImageDoctor

BoxFlow-GRPO UR2+SDG（ours）

Original Fixed ImageDoctor SDG (ours)

Base

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

a cloud in the shape of a hamburger

Three cats and one dog sitting on the grass.

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

The decisive moment taken on a Leica camera, by Henri Cartier-Bresson

A storefront with 'Google Research Pizza Cafe' written on it.

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

A Ford fiesta mark 2 in tuning version

A stack of three books: a green book on top, a red book in the middle, and a blue book on the bottom.

- Figure 5: Qualitative comparison of BoxFlowGRPO.

Figure 6: Qualitative comparison of defect-guided image refinement via GPT-Image-1.5.

- Table 5: Downstream performance of BoxFlow-GRPO and baselines. Baseline RL variants can improve reward-model scores by drifting toward illustration- or anime-like outputs, reflected by lower P(real); Only BoxFlow-GRPO improves all five reported dimensions. Parenthesized values are relative change vs. Base (green: improvement, red: regression); Avg is the mean relative change.

Method Reward PickScore CLIPScore HPSv3 DeQA P(real) Avg Base – 22.84 0.912 11.75 0.876 0.211 –

Flow-GRPO

UR2 22.78(-0.3%) 0.908(-0.4%) 11.27(-4.1%) 0.882(+0.7%) 0.149(-29.4%) -6.7% ImageDoctor 23.17(+1.4%) 0.937(+2.7%) 12.23(+4.1%) 0.857(-2.2%) 0.201(-4.7%) +0.3%

DenseFlow-GRPO ImageDoctor 22.88(+0.2%) 0.945(+3.6%) 11.57(-1.5%) 0.864(-1.4%) 0.199(-5.7%) -1.0% BoxFlow-GRPO (ours) UR2+SDG 22.91(+0.3%) 0.915(+0.3%) 12.14(+3.4%) 0.877(+0.1%) 0.228(+8.1%) +2.4%

relative to Base. BoxFlow-GRPO instead achieves the best average relative change (+2.4%) and the highest P(real) (0.228, above Base) while maintaining competitive preference and quality metrics. Figure 5 provides a qualitative example where BoxFlow-GRPO reduces both artifacts and prompt misalignment while preserving photographic realism.

- 5.2.2 Defect-Guided Image Refinement

Setup. SDG first diagnoses defects and provides GPT-Image-1.5 with a box overlay and structured text feedback. We compare it with Fixed caption-only editing and ImageDoctor heatmap-based feedback, using the same editor for all methods. Two annotators independently and blindly compare paired outputs on 873 valid samples retained after GPT-Image-1.5 filtering, assigning Good/Same/Bad labels following HunyuanImage 3.0 [Cao et al., 2025], where Good means SDG is preferred.

Table 6: GSB rates between SDG and baseline methods over 873 valid samples.

Comparison Good (%) Same (%) Bad (%)

SDG vs ImageDoctor 11.00 85.11 3.90 SDG vs Fixed 10.31 86.94 2.75

- Results. As shown in Table 6, SDG obtains higher Good than Bad rates against both ImageDoctor (11.00% vs. 3.90%) and Fixed (10.31% vs. 2.75%). The high Same rate (85.1–86.9%) reflects the strong GPT-Image-1.5 editor and the already high quality of many inputs. Figure 6 shows that SDG can still enable targeted semantic correction, e.g., identifying that an image depicts a modern Ford Fiesta rather than the prompted Mark 2. Additional qualitative refinement results are provided in Appendix E.5.

- 6 Conclusion

We presented Structured Defect Grounding (SDG), an instance-level formulation for dense textto-image (T2I) feedback that models each defect as a (location, type, reason, importance) tuple and

casts diagnosis as variable-cardinality structured set prediction. This formulation supports SDG-30K, a 30K-image box-grounded dataset across four modern T2I generators, and SDG-Eval for structured defect-set evaluation. It also enables a diagnosis-to-alignment pipeline in which a VLM-based detector predicts defect sets and BoxFlow-GRPO converts them into dense spatial rewards for diffusion alignment. Experiments demonstrate that SDG outperforms leading proprietary VLMs in zero-shot defect grounding, improves T2I alignment, and supports actionable defect-guided image refinement, making it a practical interface for evaluating and improving modern generative models.

### References

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025a.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b.

James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.

Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023.

Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Steven Hoi, Zhaohui Hou, Shijie Huang, Dengyang Jiang, Xin Jin, Liangchen Li, et al. Z-image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699, 2025.

Siyu Cao, Hangting Chen, Peng Chen, Yiji Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kipper Gong, Tianpeng Gu, Xiusen Gu, et al. Hunyuanimage 3.0 technical report. arXiv preprint arXiv:2509.23951, 2025.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

Guian Fang, Wenbiao Yan, Yuanfan Guo, Jianhua Han, Zutao Jiang, Hang Xu, Shengcai Liao, and Xiaodan Liang. Humanrefiner: Benchmarking abnormal human generation and refining with coarse-to-fine pose-reversible guidance. In European Conference on Computer Vision, pages 201–217. Springer, 2024.

Yuxiang Guo, Jiang Liu, Ze Wang, Hao Chen, Ximeng Sun, Yang Zhao, Jialian Wu, Xiaodong Yu, Zicheng Liu, and Emad Barsoum. Imagedoctor: Diagnosing text-to-image generation via grounded image reasoning. arXiv preprint arXiv:2510.01010, 2025.

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A referencefree evaluation metric for image captioning. In Proceedings of the 2021 conference on empirical methods in natural language processing, pages 7514–7528, 2021.

Hengrui Kang, Siwei Wen, Zichen Wen, Junyan Ye, Weijia Li, Peilin Feng, Baichuan Zhou, Bin Wang, Dahua Lin, Linfeng Zhang, et al. Legion: Learning to ground and explain for synthetic image detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18937–18947, 2025.

Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Picka-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36:36652–36663, 2023.

Harold W Kuhn. The hungarian method for the assignment problem. Naval research logistics quarterly, 2(1-2):83–97, 1955.

##### Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.

Youwei Liang, Junfeng He, Gang Li, Peizhao Li, Arseniy Klimovskiy, Nicholas Carolan, Jiao Sun, Jordi Pont-Tuset, Sarah Young, Feng Yang, et al. Rich human feedback for text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19401–19411, 2024.

Kaiqing Lin, Zhiyuan Yan, Ruoxin Chen, Junyan Ye, Ke-Yue Zhang, Yue Zhou, Peng Jin, Bin Li, Taiping Yao, and Shouhong Ding. Seeing before reasoning: A unified framework for generalizable and explainable fake image detection. arXiv preprint arXiv:2509.25502, 2025.

Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025.

Hanghang Ma, Haoxian Tan, Jiale Huang, Junqiang Wu, Jun-Yan He, Lishuai Gao, Songlin Xiao, Xiaoming Wei, Xiaoqi Ma, Xunliang Cai, et al. Longcat-image technical report, 2025a. URL https://arxiv.org/abs/2512.07584.

Yuhang Ma, Xiaoshi Wu, Keqiang Sun, and Hongsheng Li. Hpsv3: Towards wide-spectrum human preference score. pages 15086–15095, 2025b.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. volume 35, pages 36479–36494, 2022.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Tianhui Song, Haoyu Lu, Hao Yang, Lin Sui, Haoning Wu, Zaida Zhou, Zhiqi Huang, Yiping Bao, Y Charles, Xinyu Zhou, et al. Towards pixel-level vlm perception via simple points prediction. arXiv preprint arXiv:2601.19228, 2026.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Meituan LongCat Team, Hanghang Ma, Haoxian Tan, Jiale Huang, Junqiang Wu, Jun-Yan He, Lishuai Gao, Songlin Xiao, Xiaoming Wei, Xiaoqi Ma, et al. Longcat-image technical report. arXiv preprint arXiv:2512.07584, 2025.

Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8228–8238, 2024.

Jia Wang, Jie Hu, Xiaoqi Ma, Hanghang Ma, Yanbing Zeng, and Xiaoming Wei. Magicmirror: A large-scale dataset and benchmark for fine-grained artifacts assessment in text-to-image generation. arXiv preprint arXiv:2509.10260, 2025a.

Kaihong Wang, Lingzhi Zhang, and Jianming Zhang. Detecting human artifacts from text-to-image models. arXiv preprint arXiv:2411.13842, 2024.

Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang. Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236, 2025b.

Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023a.

Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score: Better aligning text-to-image models with human preference. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2096–2105, 2023b.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023.

Fan Yang, Ru Zhen, Jianing Wang, Yanhao Zhang, Haoxiang Chen, Haonan Lu, Sicheng Zhao, and Guiguang Ding. Heie: Mllm-based hierarchical explainable aigc image implausibility evaluator. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 3856–3866, 2025.

Zhiyuan You, Xin Cai, Jinjin Gu, Tianfan Xue, and Chao Dong. Teaching large language models to regress accurate image quality scores using score distribution. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 14483–14494, 2025.

Le Zhuo, Liangbing Zhao, Sayak Paul, Yue Liao, Renrui Zhang, Yi Xin, Peng Gao, Mohamed Elhoseiny, and Hongsheng Li. From reflection to perfection: Scaling inference-time optimization for text-to-image diffusion models via reflection tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15329–15339, 2025.

### A Annotation Guidelines

This appendix describes the annotation protocol used to construct the SDG-30K dataset. The annotation task was conducted as data curation for generated images rather than as a user study. Annotators were compensated above the local minimum wage in the data-collection location.

#### A.1 Interface and Workflow

The annotation interface consists of two synchronized panels: a raw image canvas on which annotators draw the final bounding boxes and a metadata panel showing the generation prompt together with annotation controls.

Each image is processed in two passes. In the initial annotation pass, annotators examine the promptimage pair, draw defect boxes from scratch, assign top-level labels, and write concise descriptions for each confirmed defect. In the global scan pass, annotators re-examine the full image to identify missed defects, adjust box boundaries, and improve annotation completeness. No machine-generated candidate boxes or defect descriptions are shown during manual annotation.

#### A.2 Defect-Type Taxonomy

We define two top-level defect categories, each with fine-grained subtypes.

Misalignment captures inconsistencies between the generation prompt and image content. Subtypes include missing objects, extra objects, attribute mismatches (color, count, material), spatial-relation errors, action mismatches, and style mismatches. For example, a prompt requesting “two cats on a red sofa” paired with an image showing three cats constitutes a count-based misalignment.

Artifact captures visual plausibility defects independent of the prompt. Subtypes include anatomical distortions (e.g., malformed hands), geometric deformations, texture abnormalities, edge and contour defects, text garbling, and lighting inconsistencies. For example, fused fingers or impossible joint angles constitute anatomical artifacts regardless of the prompt content.

The taxonomy explicitly separates semantic mismatch from visual corruption: green eyes when blue were requested is misalignment, while geometrically distorted eyes is an artifact. Intentional artistic stylization (e.g., cubist deformation) is not labeled as a defect; only non-intentional structural or perceptual inconsistencies qualify.

#### A.3 Boxing and Description Principles

Although bounding boxes are coarser than pixel masks, they provide a practical trade-off between spatial specificity, annotation efficiency, and compatibility with VLM outputs. Bounding boxes should tightly cover the defective location while excluding unnecessary background. Each box corresponds to one concrete defect: when a large defect location contains multiple independent issues, annotators split them into separate boxes; when multiple small detections reflect a single coherent failure mode, they may be merged.

For each box, annotators provide a concise reason statement (target length ≤30 Chinese characters) that specifically describes the defect. Vague comments such as “looks wrong” are prohibited; descriptions must reference concrete visual evidence (e.g., “left hand has six fingers” or “hat mentioned in prompt is absent”).

#### A.4 Clean Images

The annotation platform required at least one bounding box per submission. For images without defects, annotators drew a placeholder box and assigned an explicit no-problem tag. During postprocessing, these samples are converted to empty defect sets, preserving them as valid negative examples for training and evaluation.

#### A.5 Prompt Interpretation Protocol

To ensure the correctness of prompt interpretation, annotators were instructed to consult search engines whenever a prompt involved unfamiliar named entities, cultural references, artistic styles, products, landmarks, or other domain-specific concepts. External search was used only to clarify the semantic background of the prompt, rather than to provide defect labels directly. This protocol reduces annotation noise caused by incomplete prior knowledge and improves the consistency of prompt-conditioned misalignment annotation.

#### A.6 Placeholder Box Handling

Due to a platform constraint, each sample required at least one submitted box. For truly clean images, annotators drew a placeholder box and assigned an explicit “no-problem” tag. In post-processing, these samples are mapped to empty defect sets so that clean images remain label-consistent during training and evaluation.

#### A.7 Importance Scoring Rubric

Each defect instance receives an importance score (integer from 1 to 100) reflecting how much it affects overall image quality and caption faithfulness, assessed via a rubric-guided Gemini protocol. The four criteria are considered in order of priority:

- 1. Visual prominence — how easily a typical viewer can spot the defect at normal viewing distance.
- 2. Semantic impact — whether the defect changes the meaning, identity, or key content relative to the prompt.
- 3. Area coverage — larger defects affecting more of the image score higher.
- 4. Location — defects on the main subject or focal area score higher than those in the background.

Scores are grouped into five tiers: Critical (90–100), Major (70–89), Moderate (40–69), Minor (15–39), and Negligible (1–14). Importance annotations are obtained by prompting Gemini with the image, the original prompt, and the human-annotated defect set, and are used both as SFT supervision and as a GRPO reward signal.

### B Prompt Templates

This appendix provides the full prompt templates used in SDG. The same user prompt (Appendix B.1) is used for SFT training, GRPO training, and inference. The system prompt uses the default Qwen chat template (“You are a helpful assistant.”).

#### B.1 SFT / GRPO / Inference User Prompt

SDG User Prompt

You are an AI image quality evaluator. You will be given one image to analyze. ### Definitions

Misalignment: Areas where the image content does NOT match the text caption, including:

- - Missing objects: Objects mentioned in caption but not present in image
- - Extra objects: Objects present in image but not mentioned in caption
- - Wrong attributes: Incorrect color, size, material, count, or other properties
- - Wrong spatial relationships: Incorrect positions, orientations, or arrangements Artifact: Visual defects in the generated image, including:
- - Distorted anatomy: Malformed hands, extra/missing limbs, wrong number of fingers
- - Duplicated/missing parts: Repeated or absent body parts, objects
- - Warped geometry: Perspective errors, impossible shapes
- - Texture issues: Melted, smeared, or overly smooth textures
- - Unnatural edges: Jagged, broken, or blurry boundaries

- - Garbled text: Unreadable or malformed text/letters
- - Lighting inconsistencies: Wrong shadows, reflections, or light sources Text Caption: {caption}

Goal: Produce a detailed analysis of the image quality and output bounding boxes with severity scores for all detected issues. ### Strict Output Rules Output TWO blocks in this exact order:

- 1) <think> - Your detailed analysis
- 2) <answer> - JSON list of bounding boxes

### Think Format (STRICT) <think>

- ### Step 1: Caption Understanding

- Briefly summarize what the caption requires (subject, key attributes, actions, setting, style/composition if mentioned).

- ### Step 2: Visual Analysis & Defect Spotting (Issue Summary)

- - Describe the quality issues you observe in the image.
- - You MAY merge similar issues into a single bullet.
- - Each bullet MUST include:

- (a) the issue category (artifact or misalignment, or both if needed)
- (b) what is affected (object/part)
- (c) concrete visual evidence (what specifically looks wrong/missing/mismatched)

- Do NOT mention numeric coordinates.

- ### Step 3: Localization (Box-by-Box Grounding)

- - Provide a detailed, precise localization statement for EACH defect instance (one line per box).
- - Do NOT mention numeric coordinates.
- - Each localization line MUST include all of the following information in natural language:

- 1) Anchor: the exact object/part involved
- 2) Position: image-based cues (image-left/right, upper/lower, center, near the border)
- 3) Interaction cue (when applicable): holding/touching/overlapping/merging/etc.
- 4) Scale description: tiny localized detail, part-sized area, large area, or whole image
- 5) Shape/orientation: compact, elongated, runs along a boundary, wraps around an object

- - Do NOT invent new defects; each line must correspond to exactly one defect instance. </think>

### Answer Format (for <answer>) [{“box_2d”: [x0, y0, x1, y1], “label”: “misalignment”|“artifact”, “description”: “brief description of the issue”, “importance”: N}]

Bounding box coordinates are in normalized 0–1000 space: [x0, y0, x1, y1]. If there are no issues, output an empty list. ### Importance Scoring For EACH box, assign an integer importance score from 1 to 100:

- - 90–100: Critical –- immediately obvious, ruins the image.
- - 70–89: Major –- clearly visible at normal viewing distance.
- - 40–69: Moderate –- noticeable on closer inspection.
- - 15–39: Minor –- only visible on careful examination.
- - 1–14: Negligible –- barely perceptible. ### Description Style Guide
- - 18 to 45 words per description.
- - Must mention the affected object/part.
- - Must include at least one concrete evidence phrase.
- - Do NOT use vague words like “weird/strange/bad”.
- - Do NOT include numeric coordinates in description. Now analyze the image and produce your output:

#### B.2 Gemini Distillation Prompt

During data preparation, we use Gemini 3 Pro to translate Chinese defect descriptions into English, expand them with richer detail, and assign importance scores. The model receives the image, caption, and human-annotated defect boxes as hidden context. The full prompt is shown below.

Gemini Distillation Prompt

You are an AI image quality evaluator. You will be shown an image and asked to identify quality issues. Hidden context (DO NOT reveal): The human annotations below are Ground Truth. You MUST keep the same number of boxes, the same coordinates, and the same labels in the final JSON. You must behave as if you discovered the issues independently from the image and caption. Ground Truth annotations (DO NOT modify coordinates or labels; you may rewrite/expand description in English):

- - Artifact boxes: {artifact_bbox_text}
- - Misalignment boxes: {misalignment_bbox_text} Text Caption: {caption}

====== GLOBAL CONSTRAINTS (MUST FOLLOW) ======

- 1) NEVER mention annotations, boxes, ground truth, translation, or any external hints.
- 2) Do NOT invent extra defects or remove any defect. The final JSON must contain exactly the same set of boxes/labels as provided.
- 3) IMPORTANT: box_2d uses RELATIVE coordinates on a 0–1000 scale.

- - Format: [x_min, y_min, x_max, y_max] (xyxy)
- - You MUST output these coordinates exactly as given in the final JSON.

- 4) Do NOT write numeric coordinates in the <think> block.
- 5) Output MUST contain exactly TWO blocks in this order:

(1) <think> ... </think> (2) <answer> ... </answer>

- 6) <answer> MUST be a JSON list in xyxy format: [{“box_2d”:[x0,y0,x1,y1],“label”:“artifact”|“misalignment”, “description”:“...”,“importance”:N}, ...]

====== IMPORTANCE SCORING (FIELD: “importance”) ====== For EACH box, assign an integer importance score from 1 to 100:

- - 90–100: Critical –- immediately obvious, ruins the image.
- - 70–89: Major –- clearly visible at normal viewing distance.
- - 40–69: Moderate –- noticeable on closer inspection.
- - 15–39: Minor –- only visible on careful examination.
- - 1–14: Negligible –- barely perceptible. Scoring criteria (in order of priority):

- a) Visual prominence –- How easily can a typical viewer spot this defect at normal viewing distance?
- b) Semantic impact –- Does this defect change the meaning, identity, or key content of the image relative to the caption?
- c) Area coverage –- Larger defects affecting more of the image score higher.
- d) Location –- Defects on the main subject or focal area score higher than those in background/periphery.

====== THINKING FORMAT (STRICT) ====== <think>

- ### Step 1: Caption Understanding
- ### Step 2: Visual Analysis & Defect Spotting (Issue Summary)
- ### Step 3: Localization (Box-by-Box Grounding) </think>

====== DESCRIPTION STYLE GUIDE (FOR <answer>) ====== For EACH box in the final JSON, rewrite description into English with richer detail:

- - 18 to 45 words per description.
- - Must mention the affected object/part.
- - Must include at least one concrete evidence phrase.
- - Do NOT use vague words like “weird/strange/bad”.
- - Do NOT mention translation or ground truth.

====== FEW-SHOT EXAMPLES ======

- [EXAMPLE A: NON-EMPTY GT] Caption: a rock musician sticking tongue out holding a sign that says hail Satan <think>

- ### Step 1: Caption Understanding

- The caption asks for a rock musician sticking their tongue out while holding a sign reading “hail Satan”.

- ### Step 2: Visual Analysis & Defect Spotting

- - Misalignment: The person does not convincingly match the specified celebrity identity.
- - Misalignment: The sign text is not rendered as the requested phrase.
- - Artifact: Ring edges blend into skin and local finger texture appears unnaturally bumpy.

- ### Step 3: Localization

- - Box 1 (artifact): Ring–skin contact on image-right hand, tiny localized detail.
- - Box 2 (artifact): Mid-finger ring edge, compact boundary location.
- - Box 3 (artifact): Below rings on image-right fingers, tiny skin patch.
- - Box 4 (artifact): Image-left hand ring-to-finger junction, tiny localized.
- - Box 5 (misalignment): Upper-central face location, part-sized.
- - Box 6 (misalignment): Central sign surface with lettering, part-sized. </think> <answer> [{“box_2d”:[777,604,796,627], “label”:“artifact”, “description”:“Ring appears fused into finger with smeared metal-to-skin boundary...”, “importance”:45}, ...] </answer>

- [EXAMPLE B: EMPTY GT] Caption: a girl with pink curly hair wearing a witch hat <think> ... No defects identified. </think> <answer> [] </answer>

Now analyze the image and respond with exactly TWO blocks (<think> then <answer>).

- C Evaluation Metric Details

Let T = {artifact,misalignment} and let D be the evaluation set. For image d, denote the groundtruth and predicted defect sets as G(d) = {(bi,ti,ri,si)}N

##### i=1 and P(d) = {(ˆbj,tˆj,rˆj,sˆj)}M

j=1. For

d

d

type t, let Gt(d) and Pt(d) be the subsets whose type is t. Image-level metrics. We define binary indicators yt,d = 1[|Gt(d)| > 0] and yˆt,d = 1[|Pt(d)| > 0]. DetTypeF1 is the type-specific image-level F1:

yt,dyˆt,d d∈D yˆt,d

yt,dyˆt,d d∈D yt,d

2PrtRet Prt + Ret

Prt = d∈D

, Ret = d∈D

. (7)

, DetTypeF1t =

Clean-image accuracy is the true-negative rate on Dt− = {d ∈ D | yt,d = 0}:

1 |Dt−|

1[ˆyt,d = 0]. (8)

ClnAcct =

d∈Dt−

Defect-level metrics. For each d, we compute a class-aware Hungarian matching [Kuhn, 1955] between G(d) and P(d) under the constraint tg = tp. Let Mτ,t be the union of matched pairs of type t whose IoU is at least τ. For τ ∈ {0.1,0.5}, localization precision, recall, and F1 are

BoxPrt = |Mτ,t| d∈D |Pt(d)|

, BoxRet = |Mτ,t| d∈D |Gt(d)|

2|Mτ,t| d∈D(|Gt(d)| + |Pt(d)|)

, BoxF1t =

. (9)

For each valid matched pair (g,p) ∈ Mτ,t, DescCos is the mean cosine similarity between Qwen3Embedding-0.6B embeddings of rg and rˆp, and ImpAcc is normalized absolute-error accuracy:

|sg − sˆp| 100

1 |Mτ,t|

1 |Mτ,t|

. (10)

⟨rg,rˆp⟩, ImpAcct =

1 −

DescCost =

(g,p)∈Mτ,t

(g,p)∈Mτ,t

### D Experimental Details

This appendix provides the complete reward computation and training details for the GRPO stage described in Section 4.1.

- D.1 Reward Formulation Composite reward. The composite reward is gated by a format check:

R =

0.6Rdiou + 0.25Rdesc + 0.15Rimp, if format valid, −1, otherwise.

(11)

The format gate verifies that the response contains valid <think> tags, a parseable JSON defect list in <answer>, and properly ordered xyxy coordinates.

Grounding accuracy (Rdiou). Predicted and ground-truth boxes are matched using the Hungarian algorithm with DIoU as the cost metric, yielding optimal one-to-one assignments. For each matched pair (i,j) ∈ M, the spatial reward is the DIoU score of that pair.

Edge cases are handled as follows:

- • Correct rejection (both sets empty): Rdiou = 0.3.
- • Miss (ground-truth defects exist but no predictions): Rdiou = −0.8.
- • False alarm (predictions on a clean image): Rdiou = −0.3.
- • Unmatched boxes: each receives a penalty of −0.5.

The final score is normalized by max(|G|,|P|,1) and clipped to [−1,1].

Description consistency (Rdesc). For each matched pair (i,j) ∈ M, we compute the cosine similarity between the predicted and ground-truth descriptions using Qwen3-Embedding-0.6B. The raw similarity is linearly transformed to [0,1] via:

sˆij = clip

sim(ri,rˆj) − 0.5 0.4

, 0, 1 . (12)

The description reward is the sum of transformed similarities over matched pairs, divided by max(|G|,|P|,1). Unmatched boxes contribute zero, implicitly penalizing over- or under-prediction.

Importance estimation (Rimp). For each matched pair (i,j) ∈ M, the importance reward is:

rimp(i,j) = clip 1 −

|sˆj − si| 50

, 0, 1 , (13)

where si and sˆj are the ground-truth and predicted importance scores. This provides a continuous reward that decreases linearly with absolute error, reaching zero when the error exceeds 50 points.

- D.2 GRPO Objective

The full GRPO objective optimizes the policy using clipped importance ratios and KL regularization:

LGRPO = −E[min(ρsAs, clip(ρs,1 − ϵ,1 + ϵ)As)] + β KL(πθ ∥πref), (14) where

πθ(ys | I,T) πold(ys | I,T)

(15)

ρs =

is the importance ratio for sampled response ys, ϵ is the clipping range, and β=0.01 is the KL regularization coefficient.

#### D.3 SFT Hyperparameters

All generated images are resized so that the longer side is at most 1024 pixels. We use DeepSpeed ZeRO-2 with bfloat16 mixed precision on 16 GPUs. The learning rate is 3 × 10−5 with a cosine schedule and 5% warmup. We train for 1 epoch with per-device batch size 1 and gradient accumulation steps 1, yielding an effective batch size of 16; the 3× pre-baked jitter augmentation effectively exposes the model to three passes over each example within this single epoch. The maximum sequence length is 5,100 tokens, and the vision encoder is frozen throughout.

For coordinate jitter augmentation, each coordinate receives an independent random offset sampled uniformly from [−10,+10] in the [0,1000] normalized space, with clamping to ensure valid box constraints. The offsets are resampled during SFT data loading across epochs, so the effective training corpus is not identical from one epoch to the next.

#### D.4 GRPO Hyperparameters

We use 16 GPUs with DeepSpeed ZeRO-2 and a learning rate of 5 × 10−6. The KL coefficient is β=0.01. For each prompt, S=8 candidate responses are sampled via colocated vLLM rollout with temperature 1.0 and top-p 0.85 (max completion length 4,096 tokens). We train for 2 epochs with per-device batch size 4.

#### D.5 Compute Resources

All detector training experiments are run on GPUs. A single SDG detector SFT run takes approximately 2 hours on 16 GPUs, while a single detector GRPO run takes approximately 36 hours on 16 GPUs. For diffusion alignment, one BoxFlow-GRPO run takes approximately 24 hours on 16 GPUs. The defect-guided image refinement experiments use GPT-Image-1.5 through an external API; their wall-clock time depends primarily on API throughput and the allowed request concurrency rather than local GPU compute. The estimated wall-clock time for the reported experiments is about 7 days in total. The full research project required additional compute beyond this estimate because it included preliminary studies and failed experimental runs that are not reported in the paper.

#### D.6 Data, Code, and Model Availability

We provide code, model weights, reproduction instructions, and a sampled subset of SDG-30K at https://github.com/REPLACE_WITH_REPO. The complete SDG-30K dataset is undergoing release review and will be publicly released with documentation, annotation schema, data splits, and license and usage notices once approved.

#### D.7 Existing Assets and Licenses

We use existing assets only for research dataset construction, training, evaluation, or API-based editing, and cite their original sources throughout the paper. Pick-a-Pic prompts are used under the MIT License. Qwen3-VL-4B-Instruct and Qwen3-Embedding-0.6B are released under Apache-2.0. The T2I generators used for SDG-30K follow their respective public licenses or terms: FLUX.2dev is governed by the FLUX Non-Commercial License, Z-Image-Turbo and LongCat-Image are released under Apache-2.0, and SANA-1.5 is released under NSCL v2-custom / NVIDIA License. Gemini 3 Pro, GPT-5.4, and GPT-Image-1.5 are accessed only through their official API terms. Public baselines and evaluation resources, including ImageDoctor, Flow-GRPO, UnifiedReward-2.0, PickScore, CLIPScore, HPSv3, DeQA, Forensic-Chat, DrawBench, and RichHF-18K, are cited and used according to their public releases or provider terms. We do not redistribute third-party weights, prompts, images, or code except where allowed by their licenses; released SDG assets will include attribution, license notices, intended-use documentation, and pointers to the original sources.

#### D.8 LLM Usage Declaration

LLMs and VLMs are used as core components of this work. Gemini 3 Pro is used during data preparation for description expansion, reasoning-trace distillation, and importance scoring; Qwen3VL-4B-Instruct is fine-tuned as the SDG detector; Qwen3-Embedding-0.6B is used for descriptionsimilarity evaluation and reward computation; and GPT-Image-1.5 is used for defect-guided image

refinement. We also used general-purpose LLMs for manuscript writing assistance, including language polishing and wording refinement. All technical claims, experimental results, and final text were reviewed and edited by the authors.

### E Extended Experimental Results

#### E.1 SDG Detector Ablation

Table 7 reports the full ablation results summarized in Section 5.1.

Table 7: Ablation study on SDG-30K test set. Table header follows Table 2. “–” indicates the component is ablated and the corresponding metric is undefined. Ablation groups are separated by dashed lines: training stage, CoT steps, output component, and architecture/augmentation.

Artifact Misalignment Image-Level Defect-Level Image-Level Defect-Level All GT=0 GT>0 All GT=0 GT>0

Variant DetTypeF1 ClnAcc BoxF1@0.1 BoxF1@0.5 DescCos@0.1 ImpAcc@0.1 DetTypeF1 ClnAcc BoxF1@0.1 BoxF1@0.5 DescCos@0.1 ImpAcc@0.1 SFT

0.776 0.697 0.402 0.255 0.904 0.883 0.636 0.799 0.499 0.376 0.893 0.892 GRPO 0.772 0.560 0.404 0.263 0.904 0.887 0.675 0.732 0.511 0.387 0.888 0.893 SFT w/o step 1 0.719 0.793 0.357 0.218 0.894 0.880 0.597 0.798 0.436 0.329 0.875 0.892 SFT w/o step 3 0.731 0.537 0.387 0.253 0.897 0.884 0.496 0.840 0.364 0.281 0.846 0.878 GRPO (no think) 0.776 0.640 0.417 0.288 0.899 0.881 0.640 0.677 0.487 0.352 0.879 0.883 SFT w/o desc 0.757 0.718 0.386 0.250 – 0.884 0.609 0.782 0.468 0.358 – 0.894 SFT w/o importance 0.761 0.745 0.402 0.254 0.898 – 0.594 0.793 0.448 0.357 0.870 – SFT unfreeze ViT 0.659 0.638 0.329 0.203 0.894 0.880 0.584 0.799 0.427 0.305 0.856 0.880 SFT w/o jitter 0.751 0.684 0.396 0.253 0.898 0.884 0.611 0.786 0.462 0.360 0.878 0.894

| | |
|---|---|
| | |
| | |
| | |

#### E.2 Extended Comparison with ImageDoctor on SDG-30K

- Figure 7 extends the comparison in Figure 4 with six additional cases covering both artifact and misalignment defects, as well as images where the targeted defect type is absent. For ImageDoctor we show the artifact and misalignment heatmap heads separately; for SDG we overlay the predicted bounding boxes with per-instance labels. SDG consistently grounds prompt-conditioned misalignments (e.g., “Nucleosome” depicted as a double helix, “Dumbo” drawn without clown makeup) that ImageDoctor’s misalignment head misses, while avoiding the spurious face/hand activations that ImageDoctor’s artifact head produces on clean regions (rows 5–6).

ImageDoctor Misalignment

ImageDoctor Artifact

Original

GT SDG (ours)

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

|[Figure 98]|
|---|

A high quality photograph of a Nucleosome

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

ghostly image of a soldier in a cemetry

|[Figure 104]|
|---|

[Figure 105]

[Figure 106]

[Figure 107]

|[Figure 108]|
|---|

the witch conjure, masterpiece, dark fantasy, horror, creepy, beautiful composition, oil canvas by joe dante and denis villeneuve and gregory crewdsoni and john singer sargent, artstation, 4 k

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

|[Figure 113]|
|---|

still screen of Dumbo, Pennywise the Clown as an elephant, in the style of Disney

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

A still image of a young man, who is looking out from behind a tree, wearing a dark cloak

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

Twerking elephant, anthro, anthropomorphic, twerk, back view, cartoon, illustration, humor

- Figure 7: Extended qualitative comparison on SDG-30K. Columns: original image, ground-truth SDG annotations, ImageDoctor artifact heatmap, ImageDoctor misalignment heatmap, and SDG (ours) predictions. Red text in the captions highlights the prompt span corresponding to the misalignment.

#### E.3 Full SDG Output Example

We provide the complete SDG model output for the example shown in Figure 1. The model receives the generated image and the prompt, and produces a structured reasoning trace (<think>) followed by a JSON defect set (<answer>).

SDG Full Output for Figure 1

Prompt: Psychedelic smoke, explosion, fire twirling, backlit, twisting, curled, petite American ballerina, wearing ballerina sparkling lace tutu, riding long glowing neon skateboard, star jump upside down, 8K, HD, highly detailed, rendered in octane

<think>

- ### Step 1: Caption Understanding

- - The caption describes a “petite American ballerina” in a “sparkling lace tutu” riding a “long glowing neon skateboard”.
- - Key action: “star jump upside down”.
- - Setting/Style: “psychedelic smoke”, “explosion”, “fire twirling”, “backlit”, “twisting”, “curled”, “8K”, “highly detailed”, “rendered in octane”.

- ### Step 2: Visual Analysis & Defect Spotting (Issue Summary)

- - Misalignment (Action/Pose): The most prominent issue is the ballerina’s pose. The caption explicitly requests a “star jump upside down”. However, the generated image shows the ballerina standing upright on the skateboard, balancing on one leg with the other extended. She is neither upside down nor performing a star jump.
- - Artifact (Anatomy – Hands): Both hands show generation artifacts. The hand raised high in the air (image-right) appears to have fused or indistinct fingers. The hand extended to the side (image-left) also lacks clear finger definition and looks somewhat malformed or “melted”.
- - Artifact (Anatomy – Face): The facial features, while detailed, show some subtle distortion typical of AI generation, particularly in the alignment of the eyes and mouth, which appear slightly uncanny or warped.

- ### Step 3: Localization (Box-by-Box Grounding)

- - Box 1 (Artifact): The defect is located on the raised hand on the image-right side; it is a small localized region where the fingers are visible but appear fused and lack proper anatomical separation.
- - Box 2 (Artifact): The defect is on the extended hand on the image-left side; it is a small localized region at the end of the arm where the fingers are blurry and indistinct, failing to resolve into a clear hand shape.
- - Box 3 (Artifact): The defect is centered on the ballerina’s face; it is a part-sized region where the facial geometry (eyes, nose, mouth) appears slightly distorted and lacks the crisp symmetry expected of a high-quality render.
- - Box 4 (Misalignment): The defect encompasses the entire figure of the ballerina; it is a large area covering the subject who is standing upright on the board, contradicting the prompt’s instruction for an “upside down” and “star jump” pose. </think> <answer>

[

{

"box_2d": [600, 48, 641, 100], "label": "artifact", "description": "On the raised image-right hand, the fingers

appear fused and indistinct, lacking clear anatomical separation and joint definition, resulting in a distorted, unnatural shape.",

"importance": 22

}, {

"box_2d": [223, 345, 285, 385], "label": "artifact", "description": "On the extended image-left hand, the fingers

are malformed and blurry, showing a lack of distinct digits and a melted appearance rather than a properly articulated hand.",

"importance": 22

}, {

"box_2d": [474, 220, 530, 281], "label": "artifact", "description": "The facial features in the upper center show

subtle distortion, with the eyes and mouth appearing slightly warped or asymmetrical, failing to maintain a consistent realistic geometry.",

"importance": 32

}, {

"box_2d": [223, 48, 657, 840], "label": "misalignment", "description": "The ballerina is depicted standing upright on

the skateboard, which directly contradicts the caption’s explicit requirement for an ’upside down’ star jump action.",

"importance": 95 }

] </answer>

#### E.4 Extended BoxFlow-GRPO Qualitative Comparison

- Figure 8 extends the qualitative comparison in Figure 5 with additional DrawBench prompts. Compared to FlowGRPO (UR2) and DenseFlow-GRPO (ImageDoctor), BoxFlow-GRPO (UR2+SDG) better respects fine-grained prompt attributes (e.g., correct color binding in “a red book and a yellow vase”, coherent subject composition in “Darth Vader playing with raccoon”) while preserving photographic realism, avoiding the illustration/anime drift that baseline RL variants exhibit.

FlowGRPO UR2

DenseFlow-GRPO ImageDoctor

BoxFlow-GRPO UR2+SDG（ours）

Base

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

A red book and a yellow vase.

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

Darth Vader playing with raccoon in Mars during sunset.

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

Rainbow coloured penguin.

- Figure 8: Extended qualitative comparison of BoxFlow-GRPO on DrawBench prompts. Columns: Base (FLUX.1-dev), FlowGRPO with UR2 reward, DenseFlow-GRPO with ImageDoctor heatmap reward, and BoxFlow-GRPO with UR2+SDG structured reward (ours).

- E.5 Extended Defect-Guided Refinement Results
- Figure 9 extends the refinement comparison in Figure 6 with additional GPT-Image-1.5 editing cases. Across these examples, SDG feedback provides localized, instance-level guidance that helps the editor make targeted corrections, such as removing an extra lion cub when the prompt asks for a single cub, identifying an incorrect webpage title and replacing it with the correct title, and eliminating a large poster-like obstruction from a city-skyline image.

Original Fixed ImageDoctor SDG (ours)

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

charcoal drawing of a baby lion culb, wildlife photography, photorealistic

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

a branded page for a research group on game theory and AI

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

wide angle shot - full view, Design a bold and beautiful Brutalism Poster wallpaper of a city skyline

- Figure 9: Extended qualitative comparison of defect-guided image refinement via GPT-Image-1.5. Columns: original image, Fixed caption-only editing, ImageDoctor heatmap/text-feedback editing, and SDG (ours) box-structured-feedback editing.

### F Limitations and Broader Impact

#### F.1 Limitations

SDG currently focuses on two defect types, artifact and misalignment. Other quality dimensions, such as aesthetics, style, composition, safety, and cultural appropriateness, fall outside the current label space and would require additional annotation guidelines and evaluation metrics. The dataset is constructed from four contemporary T2I generators and Pick-a-Pic prompts; although this provides broad coverage, performance may change for other generators, domains, resolutions, or prompt distributions.

Our importance scores are distilled from Gemini 3 Pro under a fixed rubric. These scores provide a scalable severity signal, but they may differ from human preferences and may inherit biases from the teacher model. The SDG detector can also miss subtle defects, hallucinate defects in clean locations, or produce boxes that are too coarse for very small or highly diffuse failures. Finally, BoxFlow-GRPO assumes that box-derived penalties can be meaningfully projected onto latent spatial locations; this approximation may be less reliable when the latent grid does not align cleanly with visible defect locations.

#### F.2 Broader Impact

Structured defect feedback can make T2I systems more interpretable by exposing localized failure modes, supporting dataset auditing, and enabling targeted image refinement. These properties can help users diagnose generation errors rather than relying only on scalar preference scores.

The same capabilities also carry risks. Better diagnosis and refinement may improve the realism of synthetic images, which could be misused for deceptive or harmful content. Our release includes code, model weights, and sampled data, while the complete dataset is undergoing internal review before public release. Dataset and model releases should preserve annotation provenance, document intended use, and include safeguards for controlled release where appropriate. Because SDG can make localized judgments about generated people or scenes, downstream deployments should also consider fairness and bias in the underlying prompts, generators, annotators, and teacher models.

