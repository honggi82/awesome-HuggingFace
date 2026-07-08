# arXiv:2606.00828v1[cs.CV]30May2026

## RoboStressBench: Benchmarking VLM Robustness to Physical Visual Stress in Embodied Scenes

Leyi Wu1,3,∗, Yifan Zhao1,∗, Jinjie Zhang1,∗, Suzeyu Chen1,3,∗, Wosong Chen1,3, Zhifei Chen1, Tianshuo Xu1, Qingchun He1, Hongxin Hu1, Haojian Huang1,3, Yangkai Wei3, Wenqian Li3, Yinchuan Li3, Ying-Cong Chen1,2,† 1HKUST(GZ) 2HKUST 3Knowin

lwu398@connect.hkust-gz.edu.cn; yingcongchen@ust.hk

Evaluation on VLMs

Stress Taxonomy

[Figure 1]

###### RoboStressBench

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Low -Contrast Blend

(with three Axes)

Dark Absorptive

Transp arent

Specular Confusion

Complex Texture

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Qwen GPT InternVL Gemini Molmo

I = F ( M , V , L , G )

|[Figure 16]|
|---|

| |
|---|

|[Figure 17]|
|---|

| |
|---|

| |
|---|

16 VLMs across 5 model families

Physically Grounded Visual Stress from Image Formation

[Figure 18]

[Figure 19]

Material

[Figure 20]

Overall Performance

- A.HumanCurated

Filtering

FromUnconstrainedDatasets HumanStress

Annotation

[Figure 21]

[Figure 22]

[Figure 23]

- B.Stress

Synthesis

NominalHighStress

[Figure 24]

- C.RealWorld

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Extreme ViewPoint

Truncated out-of-frame

Small Scale

[Figure 30]

[Figure 31]

Stress-Wise Robustness

[Figure 32]

Models

|[Figure 33]|
|---|

| |
|---|

|[Figure 34]|
|---|

[Figure 35]

[Figure 36]

1. Visual Question Answering (Multi-Choice)

- Axis 2: Task
- Axis 3: Benchmark Construction

[Figure 37]

ViewPoint

[Figure 38]

[Figure 39]

Reasoning Ability

[Figure 40]

[Figure 41]

Local Underexposure

Local Overexposure

Global Underexposure

Global Overexposure

[Figure 42]

[Figure 43]

[Figure 44]

Planning Ability

[Figure 45]

|[Figure 46]|
|---|

| |
|---|

|[Figure 47]|
|---|

|[Figure 48]|
|---|

Self-CapturedScene

Lighting

Collection

[Figure 49]

Axis1:Stress

###### 2. Grounding Tasks

[Figure 50]

[Figure 51]

Bounding Box

[Figure 52]

[Figure 53]

[Figure 54]

Placement Grounding

Stacked Layout

Cluttered Layout

Non-Rigid Deform

Occlusion

[Figure 55]

[Figure 56]

| |
|---|

[Figure 57]

[Figure 58]

|[Figure 59]|
|---|

Point

|[Figure 60]|
|---|

|[Figure 61]|
|---|

Target Grounding

Geometry

Figure 1: Overview of RoboStressBench. RoboStressBench evaluates VLM robustness under physical visual stress in embodied scenes. We organize visual stress according to four imageformation factors: Material, Viewpoint, Lighting, and Geometry. The benchmark is constructed from human-curated filtering, stress synthesis, and real-world collection, and supports task-aligned evaluation through multiple-choice visual question answering and grounding tasks. We further evaluate diverse VLM families to analyze overall performance and stress-wise robustness.

###### Abstract

Vision-Language Models (VLMs) have shown strong visual understanding capabilities and are increasingly deployed in embodied AI systems, where reliable perception under real-world conditions is essential. However, existing benchmarks generally assess VLMs using clean images or isolated perturbations rather than stresses caused by physical scene formation. This design has two limitations: it covers only a narrow subset of everyday visual stresses, and some perturbations rarely appear in realistic embodied scenes. This gap points to a more fundamental

∗ Equal contribution. Authors are listed in random order. † Corresponding author.

Preprint.

question: how can we define visual stress in a principled way that captures the diverse factors encountered in real physical environments? To address this question, we formulate visual perception from an inverse graphics perspective and introduce RoboStressBench, a benchmark for systematically evaluating VLM robustness to physical visual stress in embodied scenes. Inspired by the physical rendering equation, RoboStressBench decomposes visual stress into four physically grounded dimensions: Material (M), Viewpoint (V ), Lighting (L), and Geometry (G). This design enables RoboStressBench to cover a broad range of visual stresses that commonly arise in real-world environments, while allowing controlled analysis of their effects on VLM capabilities such as visual recognition, reasoning, and planning. Through comprehensive evaluations of state-of-the-art VLMs, we identify stress-specific failure modes and reveal that different physical factors degrade different embodied capabilities, which are often obscured by aggregate accuracy. We further introduce a stress-aware agentic solver that detects visual stressors and invokes visual-editing skills before reasoning, improving robustness in challenging high-stress scenarios. Overall, RoboStressBench provides a principled evaluation framework for diagnosing and improving VLM perception under real-world physical stress, supporting the development of more reliable embodied AI systems. The project webpage is RoboStressBench Page.

###### 1 Introduction

Recent Vision-Language Models (VLMs) [1, 2, 3, 4, 5, 6] have achieved strong general visual understanding and zero-shot reasoning capabilities, making them increasingly attractive for embodied AI applications [7, 8, 9, 10]. However, for embodied agents to operate reliably in the real world, their visual perception must robustly handle a range of visual challenges. We refer to these challenges as physical visual stress: visual degradation caused by physically plausible changes in scene appearance, where task-relevant evidence is weakened, distorted, or obscured. For example, a robot may need to recognize a transparent cup, localize a partially occluded tool, or make a decision under low illumination, specular reflection, or an unusual viewpoint. As shown in Table 1, model accuracy drops on the same scene-question pairs after physically grounded stress editing, demonstrating the impact of physical visual stress on VLM reliability.

Existing benchmarks leave physical visual stress under-characterized in two ways (see Fig. 2 for an overview). General VLM benchmarks [11, 12, 13, 14] primarily evaluate broad abilities; visually challenging cases appear only incidentally and are rarely annotated with their underlying physical stress factors. Robustness-oriented benchmarks [15, 16, 17] explicitly evaluate degraded inputs, but often rely on ImageNet-C-style corruptions [18], such as noise, pixelation, or algorithmic blur. These digital perturbations are useful for robustness testing, but only partially reflect the physical visual stresses encountered in embodied scenes. As a result, existing evaluations do not provide a principled way to diagnose how physical scene factors affect VLM reliability.

To address this gap, we introduce RoboStressBench, a benchmark for evaluating VLM robustness under physically grounded visual stress in embodied scenes. Inspired by inverse graphics, we abstract image formation as I = F(M,V,L,G) and organize stress into four dimensions: Material (M), Viewpoint (V ), Lighting (L), and Geometry (G). These dimensions provide an interpretable framework for diagnosing whether failures arise from surface appearance, camera pose, illumination, or spatial structure. We construct RoboStressBench through three complementary sources: filtering, synthesis, and collection. We filter naturally occurring stress cases from existing datasets, synthesize targeted stress variants from nominal images for rare or hard-to-isolate categories, and collect additional real-world examples from Internet-sourced and self-captured images. This pipeline balances natural realism, stress diversity, and factor-level controllability.

Using RoboStressBench, we evaluate 16 state-of-the-art VLMs across five model families, including Qwen [2], InternVL [3], Molmo [4], GPT [6], and Gemini [5]. Our results show that physical visual stress affects models unevenly: geometry stress strongly degrades localization and spatial reasoning, while material and lighting stress more often affect recognition and state understanding. These task-stress interactions reveal failure modes that are hidden by aggregate accuracy, motivating stress-aware evaluation beyond a single overall score.

###### Robustness Benchmark RoboStressBench (Ours)

###### General Benchmark

✅ Nominal Images

[Figure 62]

Gaussian Noise Impulse Noise

M Material

Viewpoint

V

|[Figure 63]|
|---|

|[Figure 64]|
|---|

|[Figure 65]|
|---|

|[Figure 66]|
|---|

|[Figure 67]|
|---|

|[Figure 68]|
|---|

Miss real world stress

Shot Noise Pixelation

Lighting Geometry

L G

❓ Incidental Stress

[Figure 69]

|[Figure 70]|
|---|

|[Figure 71]|
|---|

|[Figure 72]|
|---|

|[Figure 73]|
|---|

|[Figure 74]|
|---|

|[Figure 75]|
|---|

Stress exists but no annotations

Rare in real scenes ⚠ and inadequate

Real stress + explicit annotations

[Figure 76]

[Figure 77]

[Figure 78]

###### RoboStressBench: real-world physical stress + careful annotation + task-aligned evaluation

Figure 2: Motivation for RoboStressBench. Existing benchmarks either lack explicit stress annotation or rely on artificial perturbations, whereas RoboStressBench provides realistic physical stress with careful annotations.

###### Qwen Family

###### InternVL, Molmo, Gemini3.1, GPT5.5

Global

Global

Global Overexposure Underexposure

Global Overexposure Underexposure

Stacked Layout

Stacked Layout

|Local Overexposure<br><br>|Non<br><br>Occlusion<br><br>|
|---|---|
|Truncated out of Frame<br><br>|Specular Confusion<br><br>|

|Local Overexposure<br><br>|Non<br><br>Occlusion<br><br>|
|---|---|
|Truncated out of Frame<br><br>|Specular Confusion<br><br>|

Ov

Ov

n Rigid Deform

n Rigid Deform

Local Underexposure

Local Underexposure

Cluttered Layout

Cluttered Layout

Extreme Viewpoint

Extreme Viewpoint

Small Scale

Small Scale

Transparent

Transparent

Tru o

Tru o

Complex Texture Dark Absorptive

Low Contrast Blend

Complex Texture Dark Absorptive

Low Contrast Blend

Qwen3.5-4B Qwen3.5-9B Qwen3.5-27B

- Qwen3.5-35B-A3B

- Qwen3.6-27B

Qwen3-VL-4B Qwen3-VL-8B Qwen3-VL-30B-A3B

InternVL3.5-4B InternVL3.5-8B InternVL3.5-14B

Molmo2-4B Molmo2-8B

Gemini-3.1

GPT-5.5

Qwen3.6-35B-A3B

Figure 3: RoboStressBench evaluation results. We visualize the performance of all evaluated VLMs across RoboStressBench stress dimensions. Comprehensive numerical results are reported in Table 2.

As a proof-of-concept intervention enabled by this diagnosis, we further introduce StressDART, a stress-aware test-time solver that detects the dominant stress factor, applies targeted visual rectification, and reasons over the original and rectified images. StressDART yields modest robustness gains without model fine-tuning, suggesting that explicit stress diagnosis can guide test-time interventions while also highlighting the need for content-preserving rectification.

In summary, our contributions are as follows:

- • We introduce RoboStressBench, a benchmark and evaluation protocol for diagnosing VLM robustness in embodied scenes. RoboStressBench provides a physically grounded way to characterize visual difficulty, covering common real-world stressors caused by material appearance, camera viewpoint, illumination, and scene geometry.
- • We construct an approximately 7.2K visual stress dataset through human-annotated filtering, controlled synthesis, and real-world data collection, balancing realism, diversity, and controllability.
- • We provide a systematic diagnostic analysis of VLM robustness under physical visual stress, revealing task-stress interactions and stress-specific failure modes that are obscured by aggregate accuracy.
- • We propose StressDART, a modular stress-aware agentic solver that detects visual stress, applies targeted visual rectification, and performs reasoning on the processed input. Experiments show that explicit stress diagnosis improves robustness under challenging physical conditions.

###### 2 Related Work

From Visual Corruption to Physical Stress. Investigating how visual inputs challenge model robustness has motivated extensive research into stress and perturbations. Early work characterized visual vulnerability through worst-case perturbations

[19, 20, 21]. ImageNet-C/P extended this view to non-adversarial corruptions, organizing stress into controllable families [18]. Another line of work studies natural distribution shifts that better reflect deployment conditions. These benchmarks cover real-world shifts such as background, and rotation, as well as hard natural images, rendition/sketch shifts [22, 23]. More recent efforts, such as ImageNet-3DCC [24] and ImageNet-D [25], move toward physically plausible or high-level controllable stress. However, existing taxonomies often focus on isolated stress families and lack a unified physical account. RoboStressBench addresses this by grounding visual stress in image formation and provides interpretable dimensions for benchmark-

Table 1: Effect of physical visual stress on the paired editing subset. We compare VLM accuracy on the same scene-question pairs before and after stress editing, showing the impact of physically grounded stress on model performance.

Accuracy (%) Nom. Stress Drop Qwen3VL 51.0 35.5 -15.5

Family

- Qwen3.5 53.5 36.8 -16.8
- Qwen3.6 64.3 40.1 -24.1 InternVL3.5 10.0 9.9 -0.1 Molmo2 12.2 11.5 -0.7

ing, failure attribution, and stress-aware VLM reasoning.

Robustness Evaluation for Multimodal Understanding. Robustness benchmarking has evolved from image classification to increasingly complex perception tasks. In classification, ImageNetC/P [18] established a standard protocol to measure the models’ robustness. This protocol was later extended to object detection [26] and semantic segmentation robustness [27]. Recent multimodal benchmarks have made robustness evaluation more relevant to VLMs. The Visual Robustness Benchmark for VQA [15] evaluates VQA models and MLLMs under realistic visual corruptions with robustness-oriented metrics. Res-Bench [16] focuses on MLLM resolution robustness, measuring performance stability and volatility across dynamic input resolutions. VLM-RobustBench directly evaluates VLMs under a wide range of augmentations across visually grounded and reasoningoriented datasets [17]. Some works such as R-Bench [28], Eva-VLA [29] and DarkEQA [30], study multimodal robustness under real-world corruptions and physical variations. However, existing benchmarks rarely diagnose VLM failures through the physical image-formation factors. RoboStressBench fills this gap by evaluating VLMs along four interpretable stress dimensions.

Embodied Benchmarks for Vision-Language Models. Embodied VLM evaluation has gone beyond image QA, evolving from testing whether embodied agents can answer questions to evaluating whether visual evidence can guide what to localize, how to reason spatially, and where to act. OpenEQA [31] and RoboVQA [32] exemplify this question-answering paradigm, testing situated understanding over scene observations, visual memory, task progress, and robot experience.

Closer to action, the RoboRefIt [33] dataset supports grounding language to manipulable objects and grasp targets, while RoboSpatial [34] and RefSpatial-Bench [35] extend evaluation to roboticsoriented 2D/3D spatial reasoning and multi-step referring in robot-centered scenes. More actioncentric benchmarks [36, 37, 38] evaluate decision-relevant visual outputs In parallel, broader evaluations [39, 40, 41, 42] extend this trajectory along temporal, memory, planning, and agent-level dimensions.

However, task-level scores often conflate perception, reasoning, and planning errors. RoboStressBench complements them by diagnosing failures along physical image-formation axes.

###### 3 Preliminaries

Image Formation and Visual Stress. We use physically based rendering as a conceptual basis for defining physical visual stress. The rendering equation [43] models the outgoing radiance at a surface point x along direction ωo as

Lo(x,ωo) =

fr(x,ωi,ωo)Li(x,ωi)max(0,ωi · n)dωi, (1)

Ω

where fr is the Bidirectional Reflectance Distribution Function (BRDF), Li denotes incident radiance from direction ωi, and n is the surface normal at x. Although this equation is not a complete camera model, it highlights several physical factors that shape image appearance, including material reflectance, illumination, viewing direction, and surface geometry. Following the inverse graphics perspective, we abstract image formation as

###### I = F(M,V,L,G), (2)

where M, V , L, and G denote Material, Viewpoint, Lighting, and Geometry, respectively. These factors correspond to interpretable components of image formation: M is associated with reflectance properties such as fr, L with incident illumination such as Li, V with viewing direction ωo, and G with spatial structure such as surface position and normal (x,n). We define physical visual stress as physically plausible states of these factors that make task-relevant visual evidence less accessible to VLMs while leaving the underlying scene semantics unchanged. RoboStressBench instantiates this abstraction as material, viewpoint, lighting, and geometry stress, covering phenomena such as transparency, low illumination, unusual camera poses, occlusion, and clutter.

[Figure 79]

[Figure 80]

[Figure 81]

###### RoboStress Bench

- Figure 4: Overview of RoboStressBench’s statistical distributions. (Left) Word distribution of prompt suites; (Middle) Data distribution across 16 sub-stress types; and (Right) Data distribution across different tasks.

###### 4 RoboStressBench: Benchmarking Physical Visual Stress in Embodied Scenes

We first introduce the stress taxonomy (Sec. 4.1) and then describe the dataset curation pipeline (Sec. 4.2). Fig. 1 provides an overview of RoboStressBench, Fig. 4 summarizes the dataset statistics, and Fig. 5 illustrates the stress categories and the overall curation pipeline.

###### 4.1 Stress Taxonomy

RoboStressBench organizes visual stress using a physically grounded taxonomy based on the image formation abstraction I = F(M,V,L,G). We define four primary stress dimensions: Material (M), Viewpoint (V ), Lighting (L), and Geometry (G). Each dimension is further divided into fine-grained stress categories for controlled diagnosis of VLM perception and reasoning.

Material Stress. Material stress arises from surface appearance properties that obscure object identity, boundaries, or semantic cues. We consider five material-related stress types: dark absorptive, where objects or surfaces absorb most incident light and lose visible detail; low-contrast blend, where the target visually blends into the background due to similar color, texture, or brightness; complex texture, where highly patterned surfaces interfere with recognition; transparent, where refraction or background visibility changes object appearance; and specular confusion, where mirror-like or glossy reflections introduce misleading visual evidence.

Viewpoint Stress. Viewpoint stress is caused by camera pose, object scale, or framing conditions that make an object depart from its canonical appearance. We define three viewpoint-related stress types: extreme viewpoint, covering unusual observation angles such as top-down, low-angle, or side views; truncated out-of-frame, where the target is partially outside the image boundary; and small scale, where the target occupies only a small image region and becomes difficult to recognize or localize.

Lighting Stress. Lighting stress is caused by illumination conditions that suppress, saturate, or unevenly distort visual evidence. We define four lighting-related stress types: global overexposure, where excessive illumination washes out most of the scene; local overexposure, where strong light, glare, or highlights saturate specific regions; global underexposure, where the entire scene is too dark to reveal sufficient detail; and local underexposure, where shadows or uneven lighting obscure only part of the image.

Geometry Stress. Geometry stress arises from spatial structure, deformation, occlusion, and object arrangement. We consider four geometry-related stress types: occlusion, where the target is partially blocked by another object or scene element; non-rigid deform, where object shape changes due to bending, folding, compression, or related transformations; stacked layout, where objects are piled or layered vertically and support relations become ambiguous; and cluttered layout, where dense object arrangements make segmentation and spatial reasoning difficult.

This taxonomy enables two-level diagnosis: dimension-level analysis across Material, Viewpoint, Lighting, and Geometry, and category-level analysis within each dimension. As a result, RoboStress-

Human-Curated Filtering of Open Benchmarks

Demonstrated example

Target & Placement Grounding

Planning QA

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

Reasoning QA

Q: The robot's task is to pick up the potato chips to the right of the cola can and place them in the middle of the table. Has the task been completed now? Options: 1. Yes 2. No A: No

Material Viewpoint Lighting Geometry G: Non-rigid Deform

Data Type & Example

Stress Synthesis

Open Benchmarks Augmented

###### Self-Collected Data Augmented

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

Reasoning QA Planning QA

Target Grounding

Q: Locate the teal ballpoint pen in the image (the physical object on the table, not any mirror reflection of it).

Placement Grounding Q: Please point out the free space in the direction the logo of the closest white bottle to the viewer is facing. A: [(587, 903)]

|[Figure 91]<br><br>[Figure 92]<br><br>[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]<br><br>|
|---|

| |
|---|

A: [509, 441, 560, 717]

No Stress M: Specular Confusion

Data Type & Example

No Stress L: Local Over-exposure

Data Type & Example

Demonstrated example

Human-Curated Filtering of Self-Collected Data

###### Reasoning QA

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

Target Grounding

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

Q: Locate the mirror in the image.

Target Grounding

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

A: [3, 9, 632, 733]

Material Viewpoint Lighting Geometry

M: Specular Confusion Data Type & Example

- Figure 5: Stress categories and curation pipeline. Overview of the four stress dimensions and three data sources in RoboStressBench.

Bench can identify not only whether a model fails under stress, but also which physical factor and fine-grained stress pattern are associated with the failure.

###### 4.2 Dataset Curation

RoboStressBench is curated from three complementary sources to balance realism, diversity, and controllability. First, we select naturally occurring stress cases from existing unconstrained datasets [33, 34, 35, 36, 38, 39, 44, 45]. Second, we synthesize targeted stress variants from nominal images for categories that are rare or difficult to isolate in real data. Third, we collect additional real-world examples from Internet-sourced and self-captured images. Since physical stress factors often co-occur in real scenes, RoboStressBench supports multi-label stress annotation and records the dominant stress dimension for factor-level analysis.

RoboStressBench supports both visual question answering (VQA) and grounding tasks. We retain original questions or grounding annotations when available and manually verified; otherwise, annotators create task-specific questions, answers, or grounding labels. For synthesized grounding examples, we transfer annotations when the nominal and stressed images remain pixel-aligned after resizing, and re-label the target region otherwise. Detailed dataset statistics and annotation protocols are provided in Appendix A. All examples and annotations are provided in the supplementary material.

###### 5 StressDART: Test-Time Stress Detection and Rectification for Robust Visual Reasoning

RoboStressBench reveals that many VLM failures under physical visual stress are tied to identifiable scene factors, such as poor illumination, specular surfaces, occlusion, or unusual viewpoints. This motivates a test-time strategy that first diagnoses the dominant stressor and then applies a targeted operation to recover task-relevant visual evidence. We therefore propose StressDART, a stress-aware solver for Detection And Rectification at Test time. As shown in Fig. 6, StressDART requires no model fine-tuning and consists of three stages: stress detection, stress rectification, and final reasoning.

Given an image I and a question Q, StressDART first uses a Stress Detector to predict the stress condition relevant to the task:

s,c = D(I,Q), (3)

[Figure 130]

StressDART Test-time Solver

[Figure 131]

[Figure 132]

Input: image I + Question Q

[Figure 133]

[Figure 134]

[Figure 135]

Stress Detector Stress Rectifier Predict:

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

Rectified Image Editing

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

Q

[Figure 145]

[Figure 146]

∈

Coarse Stress s {M,V,L,G} & Sub-Stress Tags

What is the object in the very upper-left corner?

[Figure 147]

[Figure 148]

Preserve original when unsafe

[Figure 149]

[Figure 150]

∈

S {L} & Global Underexposure

[Figure 151]

[Figure 152]

[Figure 153]

comic sans ms

[Figure 154]

Stress Reasoner

[Figure 155]

ANSWER

[Figure 156]

dizzy.png

Rectified Image Original Image

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

The object in the upper-left corner is a pencil holder.

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

Image with Stress

[Figure 166]

[Figure 167]

ANSWER

DirectInference

[Figure 168]

- 1. Detect

- 2. Rectify

- 3. Reason

The object in the upperleft corner is a glass cup.

[Figure 169]

[Figure 170]

- Figure 6: Overview of StressDART. Given a stressed image and a question, StressDART first detects the dominant visual stress, then applies targeted rectification to recover task-relevant evidence, and finally reasons over both the original and rectified images to produce the answer.

where s ∈ {M,V,L,G} denotes the coarse stress dimension and c denotes a fine-grained stress category, such as transparent, global underexposure, occlusion, or small scale. This explicit diagnosis allows subsequent processing to be conditioned on why the image is difficult.

Next, a Stress Rectifier selects a category-specific visual operation ϕc and applies it to the input image:

I˜ = ϕc(I), (4)

where I˜is the rectified image. For example, underexposure may trigger illumination enhancement, overexposure may trigger highlight recovery, and small-scale targets may trigger cropping or zooming. For stressors that cannot be safely corrected, the rectifier preserves the original image or applies only conservative transformations.

Finally, the Reasoner answers the original question using both the original and rectified visual evidence:

A = R(I,I,Q,s,c˜ ), (5)

where R is the VLM reasoner and A is the predicted answer. Providing both I and I˜ preserves the original task context while allowing the model to exploit recovered visual cues. By separating diagnosis, rectification, and reasoning, StressDART provides an interpretable test-time framework for improving VLM robustness under physical visual stress.

###### 6 Experiments

We evaluate RoboStressBench from three complementary perspectives. First, we benchmark a broad set of state-of-the-art VLMs to characterize their robustness under physical visual stress. Second, we analyze performance across Material, Viewpoint, Lighting, and Geometry to identify how different image-formation factors affect performance. Third, we evaluate StressDART to test whether explicit stress diagnosis and targeted visual rectification can improve robustness at test time.

###### 6.1 Experimental Settings

Evaluation Protocol. We evaluate models on multiple-choice and grounding tasks. For multiplechoice questions, we report exact-match accuracy over the predicted option. For grounding tasks, we evaluate point predictions by checking whether the point falls inside the ground-truth mask, and evaluate box predictions using IoU-based metrics. The grounding scores in Table 2 average pointbased grounding accuracy and box-based IoU@0.95. Additional grounding metrics and evaluation details are provided in Appendix B.1.

Models. We evaluate a broad collection of open-source and closed-source VLMs. For open-source models, we include representative families: Qwen3-VL [1] with 4B, 8B, and 30B-A3B variants; Qwen3.5 [2] with 4B, 9B, 27B, and 35B-A3B variants; Qwen3.6 [46] with 27B and 35B-A3B variants; InternVL3.5 [3] with 4B, 8B, and 14B variants; and Molmo2 [4] with 4B and 8B variants. For commercial models, we evaluate Gemini-3.1 [5] and GPT-5.5 [6]. In total, our evaluation covers 16 VLMs across 5 model families. In StressDART, we use Qwen3-VL-4B [1] as both the Stress

Stress Dimensions Task Dimensions

Model Size Overall

Material Viewpoint Geometry Lighting Grounding Reasoning Planning Dark L-Con C-Tex Tran. Spec. Extr. Trun. Small Occl. Non-R Stack Clust G-Ovr L-Ovr G-Und L-Und Plc. Tgt. Spa. Sta. Plan

Qwen3VL 4B 43.2 50.6 49.8 53.7 44.0 32.4 38.1 62.0 36.6 53.4 27.9 16.8 30.6 34.5 57.7 38.4 45.6 34.1 31.9 65.2 49.4 53.8 Qwen3VL 8B 49.7 58.9 57.6 59.9 49.0 38.4 52.4 69.8 45.8 62.8 33.3 21.5 36.9 48.6 66.6 46.6 53.0 45.3 36.2 73.4 58.8 64.2 Qwen3VL 30B-A3B 55.9 64.7 65.0 63.3 65.2 42.2 57.1 70.8 56.2 67.9 36.3 25.2 41.6 58.2 64.5 50.3 58.0 39.4 41.1 71.9 68.6 99.8

Qwen3.5 4B 49.8 59.4 59.1 61.5 47.3 40.4 49.5 65.6 51.6 61.9 32.3 22.4 37.9 42.6 66.4 45.9 50.8 39.4 37.1 72.6 59.6 68.8 Qwen3.5 9B 50.7 61.5 60.4 60.3 41.6 40.8 54.7 69.5 54.2 63.1 31.4 23.4 39.7 50.0 66.6 51.6 53.6 45.2 37.8 73.7 58.9 65.5

###### Qwen3.5 27B 58.0 65.3 66.0 66.4 65.9 46.3 64.6 73.2 53.4 68.9 38.2 27.6 44.4 63.5 71.3 60.1 56.7 57.2 44.9 77.1 63.3 77.0

- Qwen3.5 35B-A3B 58.1 66.5 69.0 64.6 60.7 45.3 56.6 73.2 56.7 69.1 38.0 27.1 45.8 61.8 71.0 55.3 59.2 50.3 42.5 79.0 62.9 92.9

- Qwen3.6 27B 57.3 63.3 64.5 68.8 60.7 49.1 60.4 73.2 50.8 68.7 39.2 28.3 42.5 59.6 70.6 57.6 55.2 54.7 45.0 78.3 67.1 68.8

###### Qwen3.6 35B-A3B 55.8 63.7 66.6 62.6 51.8 44.6 60.8 74.1 55.2 70.1 36.4 25.4 42.5 58.8 71.3 53.4 58.0 52.1 39.7 78.1 67.1 81.1

InternVL3.5 4B 32.1 41.2 44.5 27.6 13.4 28.3 40.6 55.0 32.7 45.5 9.5 11.7 23.2 27.2 59.5 34.2 37.6 32.5 13.6 65.2 41.1 46.2 InternVL3.5 8B 32.9 43.6 45.1 26.5 13.6 28.1 42.5 54.3 36.1 46.9 8.6 12.4 24.3 33.5 59.1 37.8 37.0 33.9 12.8 67.9 41.5 50.3 InternVL3.5 14B 29.9 37.8 41.7 24.5 11.8 24.8 37.3 53.4 31.9 42.7 9.5 12.0 19.8 26.4 55.8 28.2 35.4 29.5 9.2 67.6 40.0 45.9

Molmo2 4B 31.5 39.0 42.6 24.9 13.6 26.5 36.3 51.6 31.7 46.1 9.7 13.3 23.6 32.1 56.7 33.8 38.2 36.7 12.8 62.9 40.3 44.5 Molmo2 8B 35.2 47.0 48.1 29.0 16.9 29.5 41.0 52.5 42.1 50.8 10.9 14.7 27.5 36.3 58.1 37.6 41.4 36.5 18.1 63.6 39.2 54.3

Gemini-3.1 – 44.8 48.8 51.8 51.8 45.7 31.1 42.9 57.5 48.5 58.7 28.8 27.3 31.1 45.6 54.1 44.3 46.6 47.0 33.2 70.0 41.6 56.3 GPT-5.5 – 46.2 53.5 58.3 58.8 39.1 31.1 50.0 62.8 47.4 58.3 26.9 21.8 33.1 54.4 65.9 41.8 53.4 54.6 30.5 80.3 38.9 57.0

Table 2: Overall evaluation results on RoboStressBench. We report overall accuracy, performance across 16 fine-grained stress categories, and performance across five task dimensions. The best, second-best, and third-best results in each column are highlighted.

###### Qwen3.5-4B

###### Qwen3.5-9B

###### Qwen3.5-27B

###### Qwen3.5-35B-A3B

###### Qwen3.6-27B

###### Qwen3.6-35B-A3B

###### Qwen3-VL-4B

###### Qwen3-VL-8B

Global

Global

Global

Global

Global

Global

Global

Global

Global Overexposure Underexposure

Global Overexposure Underexposure

Global Overexposure Underexposure

Global Overexposure Underexposure

Global Overexposure Underexposure

Global Overexposure Underexposure

Global Overexposure Underexposure

Global Overexposure Underexposure

Stacked Layout

Stacked Layout

Stacked Layout

Stacked Layout

Stacked Layout

Stacked Layout

Stacked Layout

Stacked Layout

Local Overexposure

Local Overexposure

Local Overexposure

Local Overexposure

Local Overexposure

Local Overexposure

Local Overexposure

Local Overexposure

Occlusion

Occlusion

Occlusion

Occlusion

Occlusion

Occlusion

Occlusion

Occlusion

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Non Rigid Deform

Non Rigid Deform

Non Rigid Deform

Non Rigid Deform

Non Rigid Deform

Non Rigid Deform

Non Rigid Deform

Non Rigid Deform

Local Underexposure

Local Underexposure

Local Underexposure

Local Underexposure

Local Underexposure

Local Underexposure

Local Underexposure

Local Underexposure

Cluttered Layout

Cluttered Layout

Cluttered Layout

Cluttered Layout

Cluttered Layout

Cluttered Layout

Cluttered Layout

Cluttered Layout

Extreme Viewpoint

Extreme Viewpoint

Extreme Viewpoint

Extreme Viewpoint

Extreme Viewpoint

Extreme Viewpoint

Extreme Viewpoint

Extreme Viewpoint

Small Scale

Small Scale

Small Scale

Small Scale

Small Scale

Small Scale

Small Scale

Small Scale

Transparent

Transparent

Transparent

Transparent

Transparent

Transparent

Transparent

Transparent

Truncated out of Frame

Specular Confusion

Truncated out of Frame

Specular Confusion

Truncated out of Frame

Specular Confusion

Truncated out of Frame

Specular Confusion

Truncated out of Frame

Specular Confusion

Truncated out of Frame

Specular Confusion

Truncated out of Frame

Specular Confusion

Truncated out of Frame

Specular Confusion

Complex Texture Dark Absorptive

Low Contrast Blend

Complex Texture Dark Absorptive

Low Contrast Blend

Complex Texture Dark Absorptive

Low Contrast Blend

Complex Texture Dark Absorptive

Low Contrast Blend

Complex Texture Dark Absorptive

Low Contrast Blend

Complex Texture Dark Absorptive

Low Contrast Blend

Complex Texture Dark Absorptive

Low Contrast Blend

Complex Texture Dark Absorptive

Low Contrast Blend

###### Qwen3-VL-30B-A3B

###### InternVL3.5-4B

###### InternVL3.5-8B

###### InternVL3.5-14B

###### Molmo2-4B

###### Molmo2-8B

###### Gemini-3.1

###### GPT-5.5

Global

Global

Global

Global

Global

Global

Global

Global

Global Overexposure Underexposure

Global Overexposure Underexposure

Global Overexposure Underexposure

Global Overexposure Underexposure

Global Overexposure Underexposure

Global Overexposure Underexposure

Global Overexposure Underexposure

Global Overexposure Underexposure

Stacked Layout

Stacked Layout

Stacked Layout

Stacked Layout

Stacked Layout

Stacked Layout

Stacked Layout

Stacked Layout

Local Overexposure

Local Overexposure

Local Overexposure

Local Overexposure

Local Overexposure

Local Overexposure

Local Overexposure

Local Overexposure

Occlusion

Occlusion

Occlusion

Occlusion

Occlusion

Occlusion

Occlusion

Occlusion

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Non Rigid Deform

Non Rigid Deform

Non Rigid Deform

Non Rigid Deform

Non Rigid Deform

Non Rigid Deform

Non Rigid Deform

Non Rigid Deform

Local Underexposure

Local Underexposure

Local Underexposure

Local Underexposure

Local Underexposure

Local Underexposure

Local Underexposure

Local Underexposure

Cluttered Layout

Cluttered Layout

Cluttered Layout

Cluttered Layout

Cluttered Layout

Cluttered Layout

Cluttered Layout

Cluttered Layout

Extreme Viewpoint

Extreme Viewpoint

Extreme Viewpoint

Extreme Viewpoint

Extreme Viewpoint

Extreme Viewpoint

Extreme Viewpoint

Extreme Viewpoint

Small Scale

Small Scale

Small Scale

Small Scale

Small Scale

Small Scale

Small Scale

Small Scale

Transparent

Transparent

Transparent

Transparent

Transparent

Transparent

Transparent

Transparent

Truncated out of Frame

Specular Confusion

Truncated out of Frame

Specular Confusion

Truncated out of Frame

Specular Confusion

Truncated out of Frame

Specular Confusion

Truncated out of Frame

Specular Confusion

Truncated out of Frame

Specular Confusion

Truncated out of Frame

Specular Confusion

Truncated out of Frame

Specular Confusion

Complex Texture Dark Absorptive

Low Contrast Blend

Complex Texture Dark Absorptive

Low Contrast Blend

Complex Texture Dark Absorptive

Low Contrast Blend

Complex Texture Dark Absorptive

Low Contrast Blend

Complex Texture Dark Absorptive

Low Contrast Blend

Complex Texture Dark Absorptive

Low Contrast Blend

Complex Texture Dark Absorptive

Low Contrast Blend

Complex Texture Dark Absorptive

Low Contrast Blend

- Figure 7: Per-model dimension profiles on RoboStressBench. Each panel shows one model’s scores over the 16 dimensions; see Table 2 for the raw numbers.

Detector and the final Reasoner, and instantiate the Stress Rectifier with Qwen-Image-Edit [47] to produce rectified visual inputs at test time.

Implementation Details. For open-source models, we use their official inference pipelines with deterministic greedy decoding, setting the maximum generation length to 64 new tokens and disabling sampling (temperature = 0.0, top-p = 1.0). For commercial models, we query the official APIs using the same image-question format and the same generation budget (maximum output tokens = 64, temperature = 0.0, top-p = 1.0). All models are evaluated with a unified instruction template and are constrained to produce answers in the required format.

- 6.2 Main Benchmark Results Table 2 reports the overall performance of all evaluated models on RoboStressBench. Fig. 3 and

- Fig. 7 further visualize model capabilities.

- Takeaway 1: Physical visual stress remains challenging for current VLMs. Across all evaluated models, performance on RoboStressBench remains far from saturated. The best overall result is achieved by Qwen3.5-35B-A3B [2] with only 58.1% accuracy, while strong commercial models such as Gemini-3.1 [5] and GPT-5.5 [6] obtain 44.8% and 46.2%, respectively. These results indicate that current VLMs still struggle when recognition, reasoning, or planning depends on visually degraded evidence. Strong general-purpose visual understanding therefore does not necessarily translate into reliable performance under physically stressed scene conditions.
- Takeaway 2: Scaling improves average performance but does not remove stress-specific weaknesses. Within the same model family, larger variants generally improve average performance,

Placement Grounding

###### Target Grounding

###### Planning MCQ

###### Spatial MCQ

###### State Understanding MCQ

100

100

100

100

100

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

80

80

80

80

80

Accuracy(%)

60

60

60

60

60

40

40

40

40

40

20

20

20

20

20

0

0

0

0

0

Material Geometry Lighting Viewpoint

Lighting Viewpoint Material Geometry

Geometry Lighting Viewpoint Material

Viewpoint Lighting Material Geometry

Geometry Material Viewpoint Lighting

###### Qwen3VL-30B-A3B Qwen3.5-35B-A3B Qwen3.6-27B InternVL3.5-8B Molmo2-8B Gemini-3.1 GPT-5.5

- Figure 8: Task-dependent sensitivity to physical visual stress. For each task format, we visualize model accuracy across Material, Viewpoint, Lighting, and Geometry stress.

but the gains are uneven. For example, Qwen3.5 [2] improves from 49.8% with the 4B model to 58.1% with the 27B model, yielding an 8.3% gain; Qwen3VL [1] also improves from 43.2% with the 4B model to 55.9% with the 30B-A3B model. However, scaling does not consistently eliminate stress-specific failures: larger models still show low scores on the most challenging stress categories, and the InternVL3.5-14B [3] variant even underperforms InternVL3.5-4B [3] in overall accuracy. This suggests that physical stress introduces failure modes that are not fully resolved by increasing model scale alone.

- 6.3 Stress-wise Analysis To identify where VLMs fail, we further break down performance across stress types. For each task,

- Fig. 8 plots model accuracy as a function of stress category.

- Takeaway 3: Stress sensitivity is task-dependent. Physical stress affects VLM capabilities unevenly, and the dominant failure factor changes with the evaluated ability. As shown in Fig. 8, Geometry stress is especially harmful for localization-oriented tasks: placement grounding, target grounding, and spatial MCQ generally reach their lowest accuracies under Geometry, suggesting that occlusion, clutter, and ambiguous spatial structure directly weaken object localization and spatial relation reasoning. In contrast, Planning MCQ does not follow the same Geometry-dominant pattern, several models remain relatively robust under Geometry but degrade more under Material or Viewpoint stress. State Understanding MCQ also shows a different profile, with noticeable drops under Lighting for some models. These results indicate that physical stress does not simply reduce overall image quality, but selectively disrupts different VLM capabilities depending on the task.

6.4 StressDART Results

Table 3: Results and ablation of StressDART. We evaluate StressDART using Qwen3-VL-4B as the base model and compare different visual inputs to the final reasoner.

Method Reasoner Input Acc.

Qwen3-VL-4B Original 43.2% StressDART Rectified only 48.9% StressDART Original + Rectified 49.0%

We next evaluate whether explicit stress diagnosis and targeted rectification can improve testtime reasoning. Using Qwen3-VL-4B [1] as the base model, we report the results of StressDART in Table 3. We also ablate the visual input to the final reasoner by comparing two settings: using only the rectified image, and using both the original and rectified images.

- Takeaway 4: StressDART improves robustness through test-time rectification. StressDART improves over the Qwen3-VL-4B base model [1] in both input settings, showing that explicit stress diagnosis and targeted rectification can help recover task-relevant visual evidence without updating model parameters. The rectified-only setting already provides most of the gain, suggesting that visual rectification is the main source of improvement. Providing both the original and rectified images yields the best accuracy, indicating that the original image can serve as a useful reference when visual editing introduces uncertainty or slightly changes local details. Overall, StressDART provides a practical test-time robustness improvement, while also pointing to future opportunities for more precise stress diagnosis and more content-preserving rectification.

###### 7 Conclusion

We presented RoboStressBench, a physically grounded benchmark for evaluating VLM robustness under visual stress in embodied scenes. RoboStressBench organizes visual stress by four image-formation factors: Material, Viewpoint, Lighting, and Geometry. This design enables more interpretable diagnosis of model failures than treating degradation as arbitrary image corruption. The benchmark is built through human-annotated filtering, controlled stress synthesis, and real-world data collection. It covers diverse stress conditions across VQA and grounding tasks. Our evaluation of 16 VLMs shows that current models remain far from saturated under physical visual stress. It also shows that scaling alone does not eliminate stress-specific weaknesses. We further introduced StressDART, a test-time detect-and-rectify framework that improves robustness through stress diagnosis and targeted visual rectification. We hope RoboStressBench supports future research on VLMs that can perceive, reason, and act reliably under challenging real-world visual conditions.

###### References

- [1] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge et al., “Qwen3-vl technical report,” arXiv preprint arXiv:2511.21631, 2025.
- [2] Qwen Team, “Qwen3.5: Towards native multimodal agents,” February 2026. [Online]. Available: https://qwen.ai/blog?id=qwen3.5
- [3] W. Wang, Z. Gao, L. Gu, H. Pu, L. Cui, X. Wei, Z. Liu, L. Jing, S. Ye, J. Shao et al., “Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency,” arXiv preprint arXiv:2508.18265, 2025.
- [4] C. Clark, J. Zhang, Z. Ma, J. S. Park, M. Salehi, R. Tripathi, S. Lee, Z. Ren, C. D. Kim, Y. Yang et al., “Molmo2: Open weights and data for vision-language models with video understanding and grounding,” arXiv preprint arXiv:2601.10611, 2026.
- [5] Google DeepMind, “Gemini 3.1 pro model card,” 2026, accessed: 2026-04-30. [Online]. Available: https://deepmind.google/models/model-cards/gemini-3-1-pro/
- [6] OpenAI, “GPT-5.5 system card,” 2026, accessed: 2026-04-30. [Online]. Available: https://openai.com/index/gpt-5-5-system-card/
- [7] J. Liu, H. Chen, P. An, Z. Liu, R. Zhang, C. Gu, X. Li, Z. Guo, S. Chen, M. Liu et al., “Hybridvla: Collaborative diffusion and autoregression in a unified vision-language-action model,” arXiv preprint arXiv:2503.10631, 2025.
- [8] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi et al., “Openvla: An open-source vision-language-action model,” arXiv preprint arXiv:2406.09246, 2024.
- [9] D. Driess, F. Xia, M. S. Sajjadi, C. Lynch, A. Chowdhery, B. Ichter, A. Wahid, J. Tompson, Q. Vuong, T. Yu et al., “Palm-e: An embodied multimodal language model,” arXiv preprint arXiv:2303.03378, 2023.
- [10] B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid et al., “Rt-2: Vision-language-action models transfer web knowledge to robotic control,” in Conference on Robot Learning. PMLR, 2023, pp. 2165–2183.
- [11] W. Yu, Z. Yang, L. Li, J. Wang, K. Lin, Z. Liu, X. Wang, and L. Wang, “Mm-vet: Evaluating large multimodal models for integrated capabilities,” arXiv preprint arXiv:2308.02490, 2023.
- [12] Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu et al., “Mmbench: Is your multi-modal model an all-around player?” in European conference on computer vision. Springer, 2024, pp. 216–233.
- [13] B. Li, R. Wang, G. Wang, Y. Ge, Y. Ge, and Y. Shan, “Seed-bench: Benchmarking multimodal llms with generative comprehension,” arXiv preprint arXiv:2307.16125, 2023.

- [14] C. Fu, P. Chen, Y. Shen, Y. Qin, M. Zhang, X. Lin, J. Yang, X. Zheng, K. Li, X. Sun et al., “Mme: A comprehensive evaluation benchmark for multimodal large language models,” arXiv preprint arXiv:2306.13394, 2023.
- [15] F. Ishmam, I. Tashdeed, T. A. Saadat, H. Ashmafee, A. R. M. Kamal, and A. Hossain, “Visual robustness benchmark for visual question answering (vqa),” in Proceedings of the Winter Conference on Applications of Computer Vision, 2025, pp. 6623–6633.
- [16] C. Li, Z. Wang, Y. Sheng, X. Zhu, Y. Hao, and X. Wang, “Res-bench: Benchmarking the robustness of multimodal large language models to dynamic resolution input,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 37, 2026, pp. 31545–31553.
- [17] R. Saxena, A. Suglia, and P. Minervini, “Vlm-robustbench: A comprehensive benchmark for robustness of vision-language models,” arXiv preprint arXiv:2603.06148, 2026.
- [18] D. Hendrycks and T. Dietterich, “Benchmarking neural network robustness to common corruptions and perturbations,” arXiv preprint arXiv:1903.12261, 2019.
- [19] C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan, I. Goodfellow, and R. Fergus, “Intriguing properties of neural networks,” arXiv preprint arXiv:1312.6199, 2013.
- [20] I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and harnessing adversarial examples,” arXiv preprint arXiv:1412.6572, 2014.
- [21] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, “Towards deep learning models resistant to adversarial attacks,” arXiv preprint arXiv:1706.06083, 2017.
- [22] A. Barbu, D. Mayo, J. Alverio, W. Luo, C. Wang, D. Gutfreund, J. Tenenbaum, and B. Katz, “Objectnet: A large-scale bias-controlled dataset for pushing the limits of object recognition models,” Advances in neural information processing systems, vol. 32, 2019.
- [23] P. W. Koh, S. Sagawa, H. Marklund, S. M. Xie, M. Zhang, A. Balsubramani, W. Hu, M. Yasunaga, R. L. Phillips, I. Gao et al., “Wilds: A benchmark of in-the-wild distribution shifts,” in International conference on machine learning. PMLR, 2021, pp. 5637–5664.
- [24] O. F. Kar, T. Yeo, A. Atanov, and A. Zamir, “3d common corruptions and data augmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 18963–18974.
- [25] C. Zhang, F. Pan, J. Kim, I. S. Kweon, and C. Mao, “Imagenet-d: Benchmarking neural network robustness on diffusion synthetic object,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21752–21762.
- [26] C. Michaelis, B. Mitzkus, R. Geirhos, E. Rusak, O. Bringmann, A. S. Ecker, M. Bethge, and W. Brendel, “Benchmarking robustness in object detection: Autonomous driving when winter is coming,” arXiv preprint arXiv:1907.07484, 2019.
- [27] C. Kamann and C. Rother, “Benchmarking the robustness of semantic segmentation models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 8828–8838.
- [28] C. Li, J. Zhang, Z. Zhang, H. Wu, Y. Tian, W. Sun, G. Lu, X. Min, X. Liu, W. Lin et al., “R-bench: Are your large multimodal model robust to real-world corruptions?” IEEE Journal of Selected Topics in Signal Processing, 2025.
- [29] H. Liu, S. Ruan, J. Long, J. Wu, J. Hou, H. Tang, T. Jiang, W. Zhou, and W. Yao, “Eva-vla: Evaluating vision-language-action models’ robustness under real-world physical variations,” arXiv preprint arXiv:2509.18953, 2025.
- [30] Y. Park, H. Ha, W. Jo, and T.-H. Oh, “Darkeqa: Benchmarking vision-language models for embodied question answering in low-light indoor environments,” arXiv preprint arXiv:2512.24985, 2025.

- [31] A. Majumdar, A. Ajay, X. Zhang, P. Putta, S. Yenamandra, M. Henaff, S. Silwal, P. Mcvay, O. Maksymets, S. Arnaud et al., “Openeqa: Embodied question answering in the era of foundation models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 16488–16498.
- [32] P. Sermanet, T. Ding, J. Zhao, F. Xia, D. Dwibedi, K. Gopalakrishnan, C. Chan, G. DulacArnold, S. Maddineni, N. J. Joshi et al., “Robovqa: Multimodal long-horizon reasoning for robotics,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 645–652.
- [33] Y. Lu, Y. Fan, B. Deng, F. Liu, Y. Li, and S. Wang, “Vl-grasp: A 6-dof interactive grasp policy for language-oriented objects in cluttered indoor scenes,” in Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2023, pp. 976–983.
- [34] C. H. Song, V. Blukis, J. Tremblay, S. Tyree, Y. Su, and S. Birchfield, “Robospatial: Teaching spatial understanding to 2d and 3d vision-language models for robotics,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 15768–15780.
- [35] E. Zhou, J. An, C. Chi, Y. Han, S. Rong, C. Zhang, P. Wang, Z. Wang, T. Huang, L. Sheng, and S. Zhang, “Roborefer: Towards spatial referring with reasoning in vision-language models for robotics,” arXiv preprint arXiv:2506.04308, 2025.
- [36] W. Yuan, J. Duan, V. Blukis, W. Pumacay, R. Krishna, A. Murali, A. Mousavian, and D. Fox, “Robopoint: A vision-language model for spatial affordance prediction in robotics,” in Proceedings of The 8th Conference on Robot Learning, vol. 270. PMLR, 2025, pp. 4005–4020.
- [37] X. Hao, Y. Tang, L. Zhang, Y. Ma, Y. Diao, Z. Jia, W. Ding, H. Ye, and L. Chen, “Roboafford++: A generative ai-enhanced dataset for multimodal affordance learning in robotic manipulation and navigation,” arXiv preprint arXiv:2511.12436, 2025.
- [38] Y. Yuan, H. Cui, Y. Chen, Z. Dong, F. Ni, L. Kou, J. Liu, P. Li, Y. Zheng, and J. Hao, “From seeing to doing: Bridging reasoning and decision for robotic manipulation,” in International Conference on Learning Representations, 2026.
- [39] K. Chen, S. Xie, Z. Ma, P. R. Sanketi, and K. Goldberg, “Robo2vlm: Improving visual question answering using large-scale robot manipulation data,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025.
- [40] L. Qiu, Y. Ge, Y. Chen, Y. Ge, Y. Shan, and X. Liu, “Egoplan-bench2: A benchmark for multimodal large language model planning in real-world scenarios,” arXiv preprint arXiv:2412.04447, 2024.
- [41] J. Yang, S. Yang, A. W. Gupta, R. Han, L. Fei-Fei, and S. Xie, “Thinking in space: How multimodal large language models see, remember, and recall spaces,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 10632–10643.
- [42] R. Yang, H. Chen, J. Zhang, M. Zhao, C. Qian, K. Wang, Q. Wang, T. V. Koripella, M. Movahedi, M. Li, H. Ji, H. Zhang, and T. Zhang, “Embodiedbench: Comprehensive benchmarking multimodal large language models for vision-driven embodied agents,” in Proceedings of the 42nd International Conference on Machine Learning, vol. 267. PMLR, 2025, pp. 70576–70631.
- [43] J. T. Kajiya, “The rendering equation,” in Proceedings of the 13th annual conference on Computer graphics and interactive techniques, 1986.
- [44] M. Du, B. Wu, Z. Li, X. Huang, and Z. Wei, “Embspatial-bench: Benchmarking spatial understanding for embodied tasks with large vision-language models,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), 2024, pp. 346–355.
- [45] Y. Tang, L. Zhang, S. Zhang, Y. Zhao, and X. Hao, “Roboafford: A dataset and benchmark for enhancing object and spatial affordance learning in robot manipulation,” in Proceedings of the 33rd ACM International Conference on Multimedia, ser. MM ’25. New York, NY, USA: Association for Computing Machinery, 2025, p. 12706–12713. [Online]. Available: https://doi.org/10.1145/3746027.3758209

- [46] Qwen Team, “Qwen3.6-35B-A3B: Agentic coding power, now open to all,” April 2026. [Online]. Available: https://qwen.ai/blog?id=qwen3.6-35b-a3b
- [47] C. Wu, J. Li, J. Zhou, J. Lin, K. Gao, K. Yan, S. ming Yin, S. Bai, X. Xu, Y. Chen, Y. Chen, Z. Tang, Z. Zhang, Z. Wang, A. Yang, B. Yu, C. Cheng, D. Liu, D. Li, H. Zhang, H. Meng, H. Wei, J. Ni, K. Chen, K. Cao, L. Peng, L. Qu, M. Wu, P. Wang, S. Yu, T. Wen, W. Feng, X. Xu, Y. Wang, Y. Zhang, Y. Zhu, Y. Wu, Y. Cai, and Z. Liu, “Qwen-image technical report,”

2025. [Online]. Available: https://arxiv.org/abs/2508.02324

- [48] Pexels, “Free stock photos, royalty free stock images & copyright free pictures,” https://www. pexels.com/, 2026, accessed: 2026-05-03.
- [49] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li, J. Yang, H. Su et al., “Grounding dino: Marrying dino with grounded pre-training for open-set object detection,” in European conference on computer vision. Springer, 2024, pp. 38–55.
- [50] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., “Segment anything,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4015–4026.
- [51] Google Gemini, “Nano Banana Pro: Gemini ai image generator & photo editor,” https://gemini. google/us/overview/image-generation/, 2026, accessed: 2026-05-07.

Appendix

- A RoboStressBench Details

- A.1 Data Sources

RoboStressBench is constructed from three types of source data: existing public benchmarks, Internetsourced real-world images, and self-collected images. For existing public benchmarks, we use samples from EmbSpatial-Bench [44], RefSpatial-Bench [35], RoboAfford-Eval [45], RoboSpatialHome [34], ManipulationVQA [39], VABench-P [38], Where2Place [36], and RoboRefit [33]. We retain the license and usage terms of each original source. For Internet-sourced examples, we mainly collect images from Pexels [48] and follow the Pexels License, which allows free use and modification while restricting redistribution as standalone stock content. Self-collected images are captured by ourselves in diverse physical environments and are used to cover naturally occurring stress cases that are difficult to obtain from public datasets alone.

- A.2 Annotation Protocol

Six trained annotators with background knowledge in embodied AI and vision-language models perform the annotations. For examples from existing public benchmarks, annotators first identify images that already exhibit physical visual stress and assign both coarse stress dimensions and fine-grained stress tags according to the Material, Viewpoint, Lighting, and Geometry taxonomy. When the original task annotations remain valid, we directly reuse the original questions and answers after manual verification. For images that do not originally contain clear stress but are suitable for controlled augmentation, we generate stressed variants and then manually check whether the original questions can still be reused. If a question becomes ambiguous or no longer matches the edited image, annotators revise the question or answer accordingly.

For Internet-sourced and self-collected images, we use a vocabulary-driven annotation pipeline. We first define a fixed object and scene vocabulary to guide candidate collection. Then, GroundingDINO [49] and SAM [50] are used to generate candidate object annotations, which are ranked by confidence. Annotators manually inspect the candidates, remove noisy or ambiguous cases, assign stress labels, and write task-specific questions and answers. This process ensures that each example is physically meaningful, visually grounded, and aligned with the intended evaluation task. The annotation interface is shown in Fig. 9.

- A.3 Grounding Annotation Normalization

For grounding tasks, we normalize all point and bounding-box annotations to a unified coordinate range of [0,1000]. During evaluation, models are explicitly prompted to output grounding results in the same normalized coordinate system. For point-based grounding, the model outputs a point coordinate (x,y); for box-based grounding, it outputs a bounding box (x1,y1,x2,y2), where all coordinates are represented in the [0,1000] range. This normalization makes grounding evaluation independent of the original image resolution and aspect ratio. It also allows different VLMs to follow a unified output format when images have different sizes, avoiding ambiguity caused by pixel-coordinate conventions.

- A.4 Controlled Stress Synthesis

Controlled stress synthesis complements naturally occurring data by increasing coverage of stress categories that are rare, ambiguous, or difficult to isolate in real-world images. Starting from a nominal image, we generate a stressed counterpart by editing one intended physical factor from our taxonomy—Material, Viewpoint, Lighting, or Geometry—while preserving the task-relevant scene content. We use Gemini-3-Pro-Image [51] and Qwen-Image-Edit [47] as image editors, but treat them only as controlled perturbation tools: each edit prompt explicitly specifies the target stress category, the allowed visual change, and the scene elements that must remain unchanged.

Each synthesis job is defined by an edit profile. An edit profile contains: (i) a nominal image and its original task annotation; (ii) a target stress category; (iii) an editing instruction describing the desired physical change; (iv) preservation constraints for task-relevant objects, layout, camera pose, lighting

[Figure 171]

- Figure 9: Annotation interface for RoboStressBench. Annotators inspect each image, assign coarse stress dimensions and fine-grained stress tags, verify or revise task questions and answers, and check grounding annotations when applicable.

consistency, and photorealism; and (v) an annotation policy specifying whether the original label can be reused. For grounding tasks, we resize the nominal and stressed images to the same resolution and check whether the target region remains pixel-aligned. If alignment is preserved, we transfer the original normalized point or bounding box annotation. If the edit changes the target location, shape, visibility, or surrounding evidence, annotators re-label the stressed image. All generated samples are manually verified for three conditions: the intended stress is visually present, the task semantics remain valid, and the final annotation is correct.

Figures 10–13 illustrate representative synthesis protocols. We organize these examples by how the edit is controlled, rather than by enumerating every stress axis or sub-category. The selected cases cover three common control modes used throughout the pipeline: temporary spatial guides, language-only spatial edits, and appearance-factor edits. Across all modes, the principle is the same: introduce a controlled physical stress, preserve task-relevant semantics, and decide annotation reuse only after post-edit validation.

[Figure 172]

[Figure 173]

Image-editing Prompt (Geometry / Cluttered Layout):

A bright red rectangular OUTLINE on

| |
|---|

the input is an editing guide only: inside it, lock the same stacked object, pose,

and alignment; outside, you may make

the table slightly more cluttered. The output must be a clean photorealistic

picture, and the red guide must not

appear in the final image.

(a) (b) (c)

- Figure 10: Controlled synthesis with a temporary bounding-box guide. (a) Nominal inputs with rasterized red rectangles used only as editing guides. The guides indicate regions whose target object, pose, and alignment should be preserved during editing. (b) Stressed outputs after adding surrounding clutter; cyan boxes visualize annotations that are reused only when post-edit alignment is verified. (c) Example edit prompt specifying the target stress, the protected region, and the requirement that the guide must not appear in the final image.

[Figure 174]

[Figure 175]

(a) (b) (c)

| |
|---|

Image-editing Prompt

(Geometry / Non-rigid Deform):

Add a crumpled white cloth to the image on the left side of the table.

Keep all other objects, their positions,

and the overall layout unchanged. Preserve the original camera angle,

perspective, lighting, shadows, and

photorealistic style. The edited result should look like a real photo.

- Figure 11: Language-only synthesis for spatial and non-rigid stress. (a) Nominal scene without rasterized guides. (b) Stressed output with an inserted deformable foreground object; boxes indicate annotator-verified regions after editing. (c) Example prompt that specifies the non-rigid object, its placement, and preservation constraints for other objects, camera pose, lighting, and photorealistic style.

Region-guided preservation. Figure 10 illustrates a guide-based protocol for cases where the target annotation should remain stable while the surrounding scene becomes more challenging. We rasterize the original bounding box as a temporary editing guide on the reference image and instruct the editor to keep the object inside the guide fixed while adding clutter outside the guide. The guide is used only during synthesis and is never included in the final evaluation image. This protocol is useful when the desired stress affects the nearby spatial context, such as clutter or background complexity, rather than the target object itself. When the edited image remains aligned with the nominal image, the original grounding annotation and question can be reused.

Language-only spatial edits. Figure 11 shows a complementary protocol for edits that are easier to specify with natural language than with a rasterized guide. The prompt describes the inserted or modified object, its spatial placement, and the scene elements that must remain unchanged. This mode is useful for non-rigid geometry stress, deformable foreground insertions, and other cases where exact pixel preservation is not assumed. Because the inserted object may change the visible scene layout or object extent, annotators verify the edited content and update grounding labels whenever the target region, visibility, or spatial evidence changes.

Appearance-factor edits. Figures 12 and 13 illustrate controlled edits to appearance factors such as illumination and surface material. These edits do not aim to change the object layout; instead, they modify visual evidence by altering how the existing scene is seen. For lighting stress, the prompt can introduce a concentrated highlight, glare region, shadow pocket, or exposure change while preserving global geometry and object identity. For material stress, the prompt can replace a plain surface with a dense texture, decal, or typographic pattern that follows the scene perspective and lighting. In both cases, annotators check that the edited appearance is physically plausible, that the intended stress

[Figure 176]

[Figure 177]

Image-editing Prompt

(Lightning / Local Overexposure):

Increase illumination from the upperright side and create a bright,

concentrated overexposed region on the

foreground surface. Preserve the original scene structure, geometry, colors, and

material details as much as possible. The

edited result should look like a real photograph.

(a) (b) (c)

- Figure 12: Appearance-factor synthesis via local overexposure. (a) Nominal scene before editing. (b) Stressed output with a bright, localized overexposed region produced by a directional lighting change, while the overall scene structure and task-relevant objects are preserved where possible. (c) Example prompt describing the lighting manipulation and preservation constraints.

(a) (b) (c)

Image-editing Prompt

(Material / Complex Texture): Replace the plain surface with a complex patterned texture applied flat across the main plane. The texture should contain dense geometric or ornamental details and follow the perspective naturally. Preserve the scene structure. The edited result should look like a real photograph.

|[Figure 178]|
|---|

[Figure 179]

- Figure 13: Appearance-factor synthesis via complex texture. (a) Nominal scene before editing. (b) Stressed output with dense texture or typographic patterns applied to a surface in a perspectiveand lighting-consistent manner, while keeping the manipulation layout stable when possible. (c) Example prompt describing the material change and the requirement to preserve scene structure and photorealism.

category is satisfied, and that task-relevant evidence remains valid. Grounding annotations are reused only when the target remains aligned; otherwise, the stressed image is re-annotated.

Scope of examples. The examples above are representative synthesis profiles, not an exhaustive catalog of all fine-grained stress categories. Many sub-categories follow the same workflow with different prompt instantiations: specify the target stress, preserve task-relevant content, generate the stressed variant, and verify annotation validity. The selected examples cover the main control modes in our pipeline—region-guided preservation, language-only spatial editing, and appearance-factor editing—and therefore illustrate the full synthesis and quality-control procedure.

###### A.5 Dataset Statistics

RoboStressBench contains 7183 examples in total. Among them, 2927 examples are filtered from existing unconstrained datasets, 2596 examples are generated through controlled stress synthesis, and 1660 examples are collected from additional real-world sources, including Internet-sourced images and images captured by ourselves. This combination allows the benchmark to include both naturally occurring stress cases and controlled high-stress variants.

In terms of stress distribution, RoboStressBench includes 2785 Material examples, 1292 Viewpoint examples, 1753 Lighting examples, and 3327 Geometry examples. For Material stress, the dataset contains 711 dark absorptive, 761 low-contrast blend, 551 complex texture, 575 transparent, and 495 specular-confusion examples. For Viewpoint stress, it contains 212 extreme-viewpoint, 665 truncatedout-of-frame, and 496 small-scale examples. For Lighting stress, it contains 364 global-overexposure, 575 local-overexposure, 521 global-underexposure, and 319 local-underexposure examples. For Geometry stress, it contains 1205 occlusion, 579 non-rigid-deformation, 865 stacked-layout, and 1658 cluttered-layout examples. Note that a single example may be associated with multiple stress

Table 4: Detailed grounding results. We report point-based grounding accuracy and box-based grounding metrics, including IoU@0.50, IoU@0.95, and mean accuracy averaged over IoU thresholds from 0.50 to 0.95 with a step of 0.05, following the standard COCO-style protocol.

###### Model Size Point-acc IoU@0.50 IoU@0.95 mAcc

Qwen3VL 4B 50.1 80.4 26.6 68.6 Qwen3VL 8B 54.4 79.7 24.7 65.8 Qwen3VL 30B-A3B 53.5 82.0 30.4 70.0 Qwen3.5 4B 52.0 82.0 25.7 67.7 Qwen3.5 9B 56.9 82.6 25.5 68.8

###### Qwen3.5 27B 64.3 83.1 34.3 72.4

- Qwen3.5 35B-A3B 58.9 82.0 32.1 70.5

- Qwen3.6 27B 63.3 83.3 34.5 72.4

###### Qwen3.6 35B-A3B 59.9 80.6 28.0 68.7 InternVL3.5 4B 37.7 18.8 0.5 8.4 InternVL3.5 8B 37.1 27.6 0.5 11.0 InternVL3.5 14B 28.0 7.9 0.0 2.2 Molmo2 4B 38.2 3.4 0.0 0.9 Molmo2 8B 49.5 4.4 0.1 1.3 Gemini-3.1 – 58.3 60.3 18.8 45.3 GPT-5.5 – 60.4 80.3 15.0 61.2

tags; consequently, the per-tag counts are reported independently and their sum may exceed the total number of examples.

RoboStressBench also covers multiple evaluation tasks. Specifically, it contains 949 placementgrounding examples, 3411 target-grounding examples, 1369 spatial-reasoning multiple-choice examples, 633 state-understanding multiple-choice examples, and 821 planning multiple-choice examples. These task types are designed to evaluate complementary embodied capabilities, including object localization, target grounding, spatial relation reasoning, object-state understanding, and high-level planning under physical visual stress. We provide representative examples from RoboStressBench in Fig. 14–Fig. 18. All examples and annotations of RoboStressBench are provided in the supplementary material.

###### B Additional Experimental Details

###### B.1 Detailed Grounding Results

RoboStressBench contains both point-based and box-based grounding tasks. For point-based grounding, a prediction is considered correct if the predicted point falls inside the ground-truth mask. For box-based grounding, we follow COCO-style IoU evaluation and report IoU@0.50, IoU@0.95, and the mean accuracy averaged over thresholds from 0.50 to 0.95 with a step size of 0.05. In Table 2, the grounding score is computed as the average of point-based grounding accuracy and box-based IoU@0.95. Table 4 reports the separate point-based results and the complete box-based grounding metrics.

###### B.2 Compute Resources

All experiments in RoboStressBench are conducted in an inference-only setting, without model fine-tuning or parameter updates. For open-source VLMs, we run evaluation on 8 NVIDIA H100 GPUs, each with 80 GB memory, using the official inference implementations of each model. All

Stress: M Dark Absorptive, Specular Confusion

Stress: L Local Overexposure

[Figure 180]

[Figure 181]

[Figure 182]

Q: From your perspective, which object in the image is the most distant?

[Figure 183]

A. Rug. B. sofa C. night stand D. plant

Answer: (C). Night Stand

[Figure 184]

[Figure 185]

How are window and desk positioned in relation to each other in the image?

Stress: M

Complex Texture,

- A. The window is connected to the desk.
- B. The window is blocking the desk.
- C. The window is on the right side of the desk.
- D. The window is above the desk.

Occlusion

Q: Which object, in relation to your current position, holds the nearest placement in the image?

A. Television B. Window C. Coffee Table D. Door Answer: (C)

Answer: (D)

In the image, how do the positions of picture and sofa interact with each other?

[Figure 186]

[Figure 187]

Stress: M Complex Texture

Stress: G Cluttered Layout

- A. The picture is inside the sofa.
- B. The picture is blocking the sofa.
- C. The picture is at the right side of the sofa.
- D. The picture is above the sofa.

Stress: L Local Overexposure

Answer: (D)

### Stress: L Local Overexposure

What is the spatial configuration between mirror and cabinet in relation to each other within the image?

[Figure 188]

[Figure 189]

Answer: (A)

- A. The mirror is above the cabinet.
- B. The mirror is outside the cabinet.
- C. The mirror is under the cabinet.
- D. The mirror is at the right side of the cabinet.

- Figure 14: Representative examples from RoboStressBench.We show several physically stressed examples with their questions, answers, and stress annotations.

Stress: M Complex Texture

Stress: G Occlusion

Q: What is the spatial arrangement of window and picture in the image concerning each other?

Local Overexposure

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

- A. The window is above the picture.,
- B. The window is inside the picture.
- C. The window is connected to the picture.
- D. The window is on the right side of the picture.

Answer: (D)

[Figure 194]

[Figure 195]

Stress: M Low Contrast Blend

Q: What is the spatial relationship between chair and bed in the image?

Stress: L Global Overexposure

- A. The chair is blocking the bed.
- B. The chair is inside the bed
- C. The chair is on the right side of the bed.
- D. The chair is on the left side of the bed.

Q: Which object from the list is situated at the shortest distance from your point of view within the image?

A. Blinds B. Night Stand C. Pillow D. Window Answer: (C)

Answer: (C)

Stress: G No Rigid Deform

[Figure 196]

[Figure 197]

In the image from ext2, which colored point is CLOSEST to the camera?

Stress: G Stacked Layout

- A. Yellow
- B. Red
- C. Blue,
- D. Green,

Stress: L Local Overexposure

Answer: (E) E. Purple

#### Stress: M Low Contrast Blend

[Figure 198]

Q: Find a few points in the vacant space behind the book. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image.

[Figure 199]

- Figure 15: Representative examples from RoboStressBench.We show several physically stressed examples with their questions, answers, and stress annotations.

Stress: G Truncated out of Frame

Stress: G Cluttered Layout

Q: Locate some places within the free space inside the cabinet below the clothes. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image.

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

Stress: G

Q: Is the robot's grasp of the can stable?

Cluttered Layout

- A. No
- B. Partially stable
- C. Yes
- D. Cannot be determined

Occlusion

Q: In the image from ext2, which colored point is FARTHEST from the camera?

A. Red B. Purple C. Green D. Blue. E. Yellow

Answer: (C)

Answer: (D)

Q: The robot task is to place the marker on the table. Which colored arrow correctly shows the direction the robot will move next?

[Figure 206]

[Figure 207]

Stress: V Small Scale

- A. Blue
- B. Purple,
- C. Green,
- D. Red

Stress: G Occlusion

Answer: (D) E. Yellow

##### Stress: G Occlusion

###### Stress: M Dark Absorptive

[Figure 208]

[Figure 209]

Q: Is the robot's grasp of the can stable?

- A. No,
- B. Partially stable,
- C. Yes,
- D. Cannot be determined

Answer: (A)

- Figure 16: Representative examples from RoboStressBench.We show several physically stressed examples with their questions, answers, and stress annotations.

###### Stress: L Local Underexposure

Stress: M Complex Texture

[Figure 210]

[Figure 211]

Q: What is the spatial relation of the microwave relative to the photo in the image?

[Figure 212]

[Figure 213]

A. Up B. Down C. left D. Right E. Connect F. uncertain

Answer: (B)

Q. What is the spatial relation of the heater relative to the photo in the image?

[Figure 214]

Stress: M Dark Absorptive Occlusion

[Figure 215]

Q: What is the spatial relation of the console table relative to the plastic bin in the image?

A. Up B. Down C. left D. Right E. Connect F. uncertain

###### Stress: G Occlusion

- A. Up
- B. Down
- C. left
- D. Right
- E. Connect
- F. uncertain

Stress: V Truncated out of Frame

Answer: (C)

Answer: (B)

[Figure 216]

[Figure 217]

Q: Which object is farthest from us in the image?

Stress: V Small Scale

- A. television
- B. vase
- C. lamp
- D. clock
- E. plant

Stress: M Dark Absorptive

Answer: (D)

Stress: L Global Overexposure

Stress: M Dark Absoptive

[Figure 218]

[Figure 219]

Q: What is the spatial relation of the mouse relative to the keyboard in the image?

- A. Up
- B. Down
- C. left
- D. Right
- E. Connect
- F. uncertain

Answer: (B)

- Figure 17: Representative examples from RoboStressBench.We show several physically stressed examples with their questions, answers, and stress annotations.

Stress: M Dark Absorptive Low Contrast Blend

Stress: L Global Underexposure

[Figure 220]

[Figure 221]

Q: What is the spatial relation of the Keyboard relative to the Screen in the image?

- A. Up
- B. Down
- C. left
- D. Right
- E. Connect
- F. uncertain

Answer: (B)

[Figure 222]

[Figure 223]

Q: What is the spatial relation of the shower curtain relative to the cabinet in the image?

Stress: G Non Rigid Deform

Stress: M Complex Texture

- A. Up
- B. Down
- C. left
- D. Right
- E. Connect
- F. uncertain

Stress: G Cluttered Layout

Answer: (D)

[Figure 224]

Q: What is the spatial relation of the Pen relative to the Keyboard in the image?

[Figure 225]

- A. Up
- B. Down
- C. left
- D. Right
- E. Connect
- F. uncertain

Stress: G Answer: (B) Cluttered Layout

Stress: V Extreme ViewPoint

- Figure 18: Representative examples from RoboStressBench.We show several physically stressed examples with their questions, answers, and stress annotations.

models are evaluated with deterministic greedy decoding, setting the maximum generation length to 64 new tokens and disabling sampling (temperature = 0.0, top-p = 1.0), as described in Sec. 6. The full open-source model evaluation takes approximately 48 GPU-hours in total.

For StressDART, the base reasoner is Qwen3-VL-4B [1], and the Stress Rectifier is implemented with Qwen-Image-Edit [47]. This rectification step introduces additional test-time image-editing cost, requiring approximately 150 GPU-hours for the evaluated subset. Closed-source models, including Gemini-3.1 [5] and GPT-5.5 [6], are evaluated through official APIs and therefore do not consume local GPU resources.

###### C Limitations

RoboStressBench is designed as a diagnostic benchmark for physical visual stress in embodied scenes, but it still has several limitations. First, although our Material–Viewpoint–Lighting–Geometry taxonomy is physically grounded and interpretable, it is not intended to exhaust all possible sources of visual difficulty in real-world embodied environments. Although RoboStressBench supports multi-label stress annotation, stress axes are not perfectly orthogonal in real scenes; factors such as viewpoint and geometry or lighting and material appearance can still be entangled, making fine-grained attribution challenging.

Second, our dataset construction combines human-curated filtering, controlled stress synthesis, and additional real-world collection. While this design balances realism, diversity, and controllability, it may still introduce source bias from the datasets and scenes we sample, as well as artifacts from generative editing for synthesized stress cases. We reduce this risk through manual verification and re-annotation when necessary, but synthetic examples cannot fully replace naturally occurring physical stress.

Third, the current benchmark focuses on image-based VQA and grounding tasks. These tasks capture important perception, spatial reasoning, and planning-related abilities, but they do not fully evaluate closed-loop embodied behavior, long-horizon interaction, or temporal robustness in dynamic scenes. Extending RoboStressBench to video observations, multi-view interaction, and real robot execution would provide a more complete picture of embodied robustness.

Finally, StressDART is an initial test-time intervention rather than a fully optimized robustness framework. Its results show that explicit stress diagnosis and targeted rectification can substantially improve performance. Nevertheless, some negative flips still occur when visual editing changes taskrelevant cues or when the diagnosed stress does not match the true failure mode. Future work should investigate more reliable stress detectors, content-preserving rectification methods, and reasoning strategies that can better decide when to trust the original image, the rectified image, or both. We hope these limitations will motivate future research on more realistic, temporally grounded, and action-aware robustness evaluation for embodied VLMs.

###### D Broader Impacts

RoboStressBench aims to support the development of more reliable VLMs for embodied AI by exposing failures under physically plausible visual stress. Such evaluation can benefit robotics systems that must operate in challenging real-world environments, including low illumination, occlusion, reflective materials, unusual viewpoints, and cluttered scenes. By providing stress annotations and task-level evaluation, RoboStressBench can help researchers diagnose when VLM perception is unreliable and develop targeted robustness improvements before deployment.

At the same time, the benchmark and StressDART inherit broader risks associated with VLM-based embodied systems. Models may still hallucinate answers, mislocalize objects, or overestimate their confidence under severe visual ambiguity. When integrated into robotic pipelines, such errors may lead to unsafe manipulation, navigation, or planning decisions, especially in high-stakes environments. StressDART can improve robustness through test-time rectification, but visual editing may also alter task-relevant evidence if applied incorrectly, so it should not be treated as a substitute for calibrated uncertainty estimation or safety checks.

We believe that releasing RoboStressBench to the research community can have positive impact by enabling more systematic evaluation of perception robustness under realistic physical stress. Open

access to the benchmark, annotations, and evaluation protocol can facilitate reproducible comparison, encourage stress-aware model development, and support safer embodied AI systems across robotic platforms such as mobile robots, robotic arms, and humanoids.

###### E Impact Mitigation Measures

RoboStressBench is an evaluation benchmark, not a deployed embodied agent, but we still consider the possible risks related to data release and benchmark usage. We will document the dataset sources, stress taxonomy, annotation process, task formats, evaluation metrics, and inference settings to make the benchmark transparent and reproducible. When releasing the data, we will follow the licenses of the original sources, remove or exclude images with personally identifiable or sensitive information, and clearly indicate which samples are real and which are synthesized. We will also release the benchmark data, annotation schema, evaluation scripts, and usage instructions as soon as possible to support responsible use by the community.

We will also provide clear guidance on what the benchmark should and should not be used for. RoboStressBench is intended to help researchers diagnose how VLMs fail under realistic physical visual stress, rather than to support surveillance, biometric identification, or high-stakes automated decisions. Similarly, StressDART should be viewed as an exploratory test-time strategy, not as a complete safety mechanism, since visual editing may sometimes change important visual cues. Therefore, we encourage users to report both successful and failed cases, keep the original image available during reasoning, and treat benchmark results as evidence for improving model robustness rather than as proof that a model is ready for real-world deployment.

###### F Licenses

RoboStressBench is constructed from existing public benchmarks, Pexels-sourced real-world images, and controlled stress synthesis. We retain the license and usage terms of each original data source. Our annotations, metadata, and benchmark construction code may be released under our chosen research license, while images and derived visual assets remain subject to the licenses or terms of their corresponding source data.

- 1. Existing public benchmarks. RoboStressBench uses samples from EmbSpatial-Bench [44], released under CC BY 4.0; RefSpatial-Bench [35], released under Apache 2.0; RoboAffordEval [45], released under CC BY 4.0; RoboSpatial-Home [34], released under Apache 2.0; ManipulationVQA [39], released under Apache 2.0; VABench-P [38], released under Apache 2.0; and Where2Place [36], released under Apache 2.0. RoboRefit [33] is distributed via the official VL-Grasp repository without an explicit dataset license; we use it for non-commercial academic research only, consistent with common practice for unlicensed academic datasets.
- 2. Pexels-sourced real-world images. The dataset contains images sourced from Pexels [48]. Under the Pexels License, content is free to use and modify for commercial or noncommercial purposes without required attribution. The terms explicitly prohibit redistributing or selling the photos on other stock photo or wallpaper platforms. We release these images exclusively as part of an academic benchmark dataset, which strictly complies with these terms. Users of our benchmark are also subject to the original Pexels License.
- 3. Controlled stress synthesis. Some controlled stress samples are synthesized from existing benchmark images, such as lighting-stress variants generated from public benchmark sources. These derived samples inherit the license and usage constraints of their underlying source datasets and are not relicensed independently. Synthesis based on proprietary in-house data uses raw images provided by an industrial partner. We plan to release these specific derived samples under a research-only, non-commercial license to protect the proprietary nature of the original assets.

