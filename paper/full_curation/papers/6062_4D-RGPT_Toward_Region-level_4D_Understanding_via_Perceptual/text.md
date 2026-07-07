[Figure 1]

2025-12-18

## 4D-RGPT: Toward Region-level 4D Understanding via Perceptual Distillation

Chiao-An Yang1, Ryo Hachiuma, Sifei Liu, Subhashree Radhakrishnan, Raymond A. Yeh1, Yu-Chiang Frank Wang, Min-Hung Chen

NVIDIA

Despite advances in Multimodal LLMs (MLLMs), their ability to reason over 3D structures and temporal dynamics remains limited, constrained by weak 4D perception and temporal understanding. Existing 3D and 4D Video Question Answering (VQA) benchmarks also emphasize static scenes and lack region-level prompting. We tackle these issues by introducing: (a) 4D-RGPT, a specialized MLLM designed to capture 4D representations from video inputs with enhanced temporal perception; (b) Perceptual 4D Distillation (P4D), a training framework that transfers 4D representations from a frozen expert model into 4D-RGPT for comprehensive 4D perception; and (c) R4D-Bench, a benchmark for depth-aware dynamic scenes with region-level prompting, built via a hybrid automated and human-verified pipeline. Our 4D-RGPT achieves notable improvements on both existing 4D VQA benchmarks and the proposed R4D-Bench benchmark.

# arXiv:2512.17012v4[cs.CV]13Apr2026

#### Links: Project Page | GitHub | Dataset

Region (2D): Where is ⟨R1⟩? Depth (3D): How far away is ⟨R1⟩? Time (4D): When does ⟨R1⟩ move?

[Figure 2]

[Figure 3]

[Figure 4]

- Frame 1

- Frame 2 Frame N (*) (*)

Ans: I am not sure.

[Figure 5]

[Figure 6]

⟨R1⟩

- (a) Baseline

[Figure 7]

R4D-Bench

[Figure 8]

- (b) Ours

|[Figure 9]|[Figure 10]|
|---|---|

Track ⟨R1⟩ across frames

[Figure 11]

[Figure 12]

[Figure 13]

|[Figure 14]|
|---|

Depth Perception

Ans: 7.0 m/s

[Figure 15]

[Figure 16]

[Figure 17]

Temporal Perception

|[Figure 18]|
|---|

|[Figure 19]|
|---|

|[Figure 20]|
|---|

[Figure 21]

[Figure 22]

[Figure 23]

| | | | |
|---|---|---|---|
| | | | |

0 2 12

Q: What is the average speed of ⟨R1⟩?

Figure 1 | Overview of Region-level 4D Understanding. 4D region-level VQA, e.g., our R4D-Bench, requires MLLMs to be able to track regions (2D), perceive depth (3D), and temporal progression (4D). Baseline MLLMs cannot recognize one or more of these aspects and thus fail to answer questions correctly. With our distillation framework, our 4D-RGPT better perceives these aspects and answers accurately. We note that the regions labeled with (*) are not provided in R4D-Bench; they are visualized for readability.

### 1. Introduction

temporal visual understanding.

In this paper, we advance MLLMs for one such challenging task: Region-level 4D Understanding. This unique problem combines two critical aspects: (1) 4D understanding, which demands answering questions regarding depth information, temporal dynamics, or object interactions in 3D space over time; and (2) region-level understanding, which requires grounding language queries to specific visual regions for controllable input. Region-level 4D VQA is essential for demanding real-world applications, such as au-

By integrating visual inputs with Large Language Models (LLMs) [1, 2, 3, 4], Multimodal LLMs (MLLMs) demonstrate remarkable capabilities in complex understanding across vision and language modalities. However, current MLLMs, even proprietary models such as GPT-4o [5], often struggle with highly specialized tasks that require fine-grained spatial1 and

[Figure 24]

1We use “spatial” in this paper to refer to 3D (i.e., 2D + depth), rather than 2D as in several general video understanding works.

1affiliated with Purdue University. Work done during Chiao-An’s internship at NVIDIA. © 2026 NVIDIA. All rights reserved.

tonomous driving and industrial inspection, where 4D information is critical and user queries must precisely target specific regions rather than rely on ambiguous descriptions. As an example, in Fig. 1, the 4D question “What is the average speed of ⟨𝑅1⟩?” specifically targets the speed of the car marked by the purple bounding box ⟨𝑅1⟩.

To achieve 4D understanding, previous works mainly rely on conventional Supervised Fine-Tuning (SFT) [6, 7, 8, 9] or Reinforcement Learning (RL) [10, 11, 12, 13, 14] paradigms, optimizing primarily over the final text output using self-curated data. However, due to the difficulty of curating large-scale, well-annotated dynamic video data, these works often struggle with dynamic scenarios. In regionlevel 4D VQA, having strong 4D understanding is even more critical, as it requires tracking region movement over time. More recently, several works [15, 16, 17, 18, 19, 20, 21] exploit external models to inject 3D knowledge into MLLMs to improve spatial understanding capabilities. However, external

- 3D knowledge mainly helps understand static videos, without fully achieving 4D understanding. Moreover, these approaches often integrate additional modules into the architecture, introducing additional inference burdens.

To address these challenges, we propose 4DRGPT, a specialized MLLM with effective 4D perception and thus better 4D understanding capabilities.

- 4D perception refers to the ability to extract low-level 4D perceptual knowledge, e.g., depth and optical flow. Specifically, 4D-RGPT perceives 4D knowledge via our proposed Perceptual 4D Distillation (P4D) training-only framework. P4D adopts both latent and explicit distillation processes to effectively distill 4D perceptual knowledge from an expert 4D teacher model into the student 4D-RGPT. Notably, unlike previous works, P4D contains only training-only modules, incurring no additional inference cost. Finally, we introduce Timestamp Positional Encoding (TPE) to provide explicit temporal cues, enhancing MLLMs’ temporal perception capability.

While various 3D/4D VQA benchmarks have been proposed recently [22, 23, 24, 25, 26, 18], they often lack either region-prompted questions or sufficient 4D understanding challenges. As demonstrated in Fig. 1, this limitation prevents comprehensive evaluation of region-based 4D VQA capabilities, namely, answering questions about specific regions (e.g., ⟨𝑅1⟩) in a 4D context. To bridge this gap, we construct R4DBench, a new benchmark containing both static and dynamic scene understanding tasks with region-based 4D questions.

Our experiments show that 4D-RGPT improves

over the baseline on both non-region-based 3D/4D benchmarks (+5.3% on average across 6 benchmarks) and our region-based R4D-Bench benchmark (+4.3%), while effectively capturing explicit 4D signals.

#### Our main contributions are as follows:

- • We propose 4D-RGPT (Sec. 4.1), a specialized MLLM that perceives 4D information for enhanced understanding.
- • We propose the P4D (Sec. 4.2) training framework to distill 4D perceptual knowledge into 4DRGPT without introducing additional inference cost.
- • We introduce R4D-Bench (Sec. 5), a region-based 4D VQA benchmark that requires region-level 4D understanding.

### 2. Related Work

#### 2.1. Multimodal LLMs (MLLMs)

The success of LLMs [1, 2, 27, 28, 4, 29, 3] has inspired various MLLMs [30, 5, 31, 32, 33, 34, 35, 36] for multi-modal understanding or generation. While several MLLMs [37, 38, 39, 40, 41] excel at video understanding, they lack specialization in region-level or 3D/4D tasks.

Region-Level MLLMs understand specified regions within visual inputs. Earlier works [42, 43, 44, 45, 46, 47, 48, 49, 50, 51] use bounding box coordinates as text prompts, while others [52, 53, 54, 55, 20, 56] extract Region of Interest (RoI) visual features. Visual markers [57, 58, 59, 60] provide intuitive region indication. However, region-level video understanding remains challenging, especially for dynamic scenes where user queries provide sparse region annotations without temporal tracking (Fig. 1). While recent works [61, 20] address this, they do not fully explore 4D dynamic scenarios. We propose 4D-RGPT (Sec. 4.1) to interpret 4D spatio-temporal knowledge without 4D annotations during training.

3D/4D MLLMs focus on spatial and temporal understanding. Previous works [17, 18, 16, 9, 62, 13, 63, 20, 24, 64] enhance MLLMs with depth or 3D reconstruction models but require additional modules, introducing inference costs. Others use SFT [6, 7, 8, 9] or RL [10, 11, 12, 13, 14] with text-based supervision, which is insufficient for 4D perception. We propose P4D (Sec. 4.2) to enhance 4D perception without modifying the architecture. Prior works [65, 66] distill into vision-only encoders, e.g., ViT, VideoMAE. 3DRS [64] employs distillation for static 3D scenes, while P4D addresses dynamic scenes with dual distil-

- Table 1 | Comparison among 3D / 4D VQA Benchmarks. Existing benchmarks either lack dynamic video data or region prompts, while our R4DBench is the first to provide both at scale. All benchmarks are downloaded from official sources as of August 2025, and the numbers of VQA might differ from the original papers. Static videos contain only camera movement, while dynamic videos contain both camera and object movement. †We only adopt real-world videos from the VLM4D benchmark.

Dataset Regions Input Type FPS # Visual # QA

SAT-real [24] ✗ Images - 196 150 MMSI-Bench [26] ✗ Images - 2.5k 1.0k OmniSpatial [25] ✗ Images - 561 1.5k VSTI-Bench [18] ✗ Static Video 24 312 6k STI-Bench [22] ✗ Dynamic Video 10 ∼ 30 369 2k VLM4D-real† [23] ✗ Dynamic Video 12 ∼ 24 600 1k

R4D-Bench (Ours) ✓ Dynamic Video 10 ∼ 30 780 1.5k

lation on latent and explicit representations to achieve 4D understanding.

#### 2.2. 3D/4D VQA Benchmarks

Several benchmarks evaluate MLLMs’ 3D and 4D understanding. OmniSpatial [25], VSTI-Bench [18], SAT [24], and MMSI-Bench [26] focus on 3D spatial understanding in images. STI-Bench [22] is a pioneering work that introduces 4D VQA on both static and dynamic videos, while VLM4D [23] focuses on semantic understanding in dynamic videos. However, these benchmarks lack region-level prompting or sufficient dynamic video data (Tab. 1). We introduce R4D-Bench (Sec. 5) with region-level prompts and diverse 4D understanding tasks.

### 3. Preliminaries and Notations

We briefly review the background and introduce notation for an MLLM and a 4D perception model.

Multimodal LLMs extend the understanding capabilities of LLMs to visual inputs such as images and videos. The architecture typically consists of: (a) EV: a vision encoder for input visuals, e.g., images or videos; (b) EP: a multi-modal projector that aligns the visual and textual features within a shared space; (c) LLM: an auto-regressive model that takes in both features and generates output hidden states or tokens in a step-by-step manner; (d) Dhead: a linear head layer that maps the hidden states to the final vocabulary space for text generation.

4D Perception Models, e.g., L4P [67], encode a latent feature from input visuals for multiple 4D lowlevel representations. They consist of a unified en-

coder E4D and specialized decoders D𝑚 for each 4D modality 𝑚 ∈ ℳ. Each 4D modality 𝑚 ∈ ℳ describes some per-pixel 4D properties of the input video. For example, 𝑚 can be either “depth,” which describes the per-pixel depth values, or “flow,” which describes the per-pixel optical flow between adjacent frames.

We denote the input video as 𝑉 = [𝐼(𝑛)]𝑛=1:𝑁 with each image frame 𝐼(𝑛) ∈ R𝐻×𝑊×3. Here, 𝑁 is the number of input frames and (𝐻,𝑊) is the spatial size. Given 𝑉 , we can acquire its 4D latent representation as follows,

𝐹4D = E4D(𝑉 ) ∈ R𝑁

′×ℎ′×𝑤′×𝑐′, (1)

where 𝑁′,ℎ′,𝑤′ are the down-sampled number of frames, height, and width of E4D’s outputs and 𝑐′ is the number of output channels.

For each 𝑚, the decoder D𝑚 decodes 𝐹4D to its corresponding low-level representation, i.e.,

𝑃𝑚 = D𝑚(𝐹4D). (2) We use the following 4D modalities ℳ in this work: (a) 𝑚 = depth where 𝑃depth(𝑛) ∈ R𝐻×𝑊×1 describes the per-pixel depth values; (b) 𝑚 = flow where 𝑃flow(𝑛) ∈ R𝐻×𝑊×2 describes the per-pixel optical flow between adjacent frames; (c) 𝑚 = motion where 𝑃motion(𝑛) ∈ R𝐻×𝑊×1 describes whether a pixel is moving or static in 3D space; (d) 𝑚 = camray where 𝑃camray(𝑛) ∈ R𝐻×𝑊×6 describes the per-pixel Plucker ray maps.

### 4. Approach

Overview. Given a video 𝑉 and a question 𝑄, an MLLM responds with an answer 𝐴 autoregressively. To tackle the complex, dynamic scenes presented in 4D VQA benchmarks, we develop an MLLM that can better answer questions by incorporating 4D knowledge from a teacher model and leveraging low-level representations, e.g., depth and flow, over time. To this end, we design 4D-RGPT to capture both latent 4D features and explicit 4D signals from 𝑉 with training-only modules. These 4D representations enable the model to better perceive 4D knowledge during training, without introducing additional inference cost. Additionally, to accurately capture temporal progression for answering 4D questions, we introduce Timestamp Positional Encoding (TPE) to provide explicit temporal cues to the MLLM.

To circumvent the extreme training cost and instability of training MLLMs from scratch, we introduce our Perceptual 4D Distillation (P4D) framework

###### Teacher: 4D Perception Model

Explicit Signals

[Figure 25]

Input Video Latent Features

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Unified 4D Encoder

Specialized 4D Decoder

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Embed

[Figure 52]

[Figure 53]

Latent Distillation (LD) Explicit Distillation (ED)

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

| |
|---|
| |
| |

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

|[Figure 65]|
|---|

|[Figure 66]|
|---|

| |[Figure 67]|
|---|---|
| | |

[Figure 68]

[Figure 69]

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

TPE

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

0.0

2.0

[Figure 89]

[Figure 90]

[Figure 91]

Question Answer

[Figure 92]

[Figure 93]

[Figure 94]

| |
|---|
| |

|[Figure 95]<br><br>Frozen Hidden states Trainable Training-only Distillation<br><br>[Figure 96]<br><br>| | | |
|---|---|---|
<br><br>[Figure 97]|
|---|

Which direction is the batter running towards?

Away from the camera.

[Figure 98]

[Figure 99]

Student: 4D-RGPT

- Figure 2 | Perceptual 4D Distillation (P4D) framework for 4D-RGPT. For each frame 𝐼(𝑖) in 𝑉 , 4D-RGPT extracts 4D representations through training-only modules, i.e., D4DP and D𝑚 for 𝑚 ∈ ℳ. This includes both latent features, i.e., 𝐹^4D, and explicit signals, e.g., depth 𝑃^depth or optical flow maps 𝑃^flow. We also incorporate timestamp positional encodings (TPE) to provide temporal cues for 4D-RGPT to be temporally aware. In the P4D framework, the frozen teacher, i.e., 4D perception model, captures 4D expert knowledge from 𝑉 . It is then distilled to the student 4D-RGPT via two strategies. (a) Latent Distillation (LD): We align the latent 𝐹^4D with the teacher’s intermediate 4D embeddings 𝐹4D. (b) Explicit Distillation (ED): We align the explicit 𝑃^𝑚 with the teacher’s final 4D signals 𝑃𝑚. 4D-RGPT is optimized end-to-end using both SFT loss and the distillation losses, i.e., ℒLD and ℒED.

[Figure 100]

projector EP, and LLM, each frame 𝐼(𝑛) is encoded as hidden state features 𝐹hidden(𝑛) ∈ Rℎ×𝑤×𝑐, where 𝑙 = ℎ𝑤 is the number of per-image tokens, (ℎ,𝑤) is the spatial size of visual features, and 𝑐 is the hidden dimension. We introduce a training-only MLP as a 4D perception decoder D4DP on top of the MLLM to decode latent 4D representations 𝐹^4D(𝑛). Specifically, we first sample and resize (Rearrange) the hidden 𝐹hidden(𝑛) to match the target shape of (𝑁′,ℎ′,𝑤′) in Eq. 1. Thus, for each down-sampled frame 𝑛′ ∈ [1,𝑁′], we have

to distill 4D knowledge into 4D-RGPT during training. As shown in Fig. 2, we leverages a frozen expert 4D perception model [67] (teacher), to supervise both latent and explicit 4D representations of 4DRGPT (student). The latent distillation provides intermediate guidance on abstract 4D features, while the explicit distillation ensures accurate extraction of interpretable low-level 4D signals. We describe the 4D-RGPT architecture in Sec. 4.1 and the P4D framework in Sec. 4.2.

4D = D4DP (︁Rearrange(𝐹hidden(𝑛) ))︁. (3)

𝐹^(𝑛

′)

#### 4.1. 4D-RGPT

Given an input video 𝑉 with 𝑁 sampled frames [𝐼(𝑛)]𝑁𝑛=1, and the timestamps {𝑡(𝑛)}𝑁𝑛=1 of each frame, our 4D-RGPT consists of training-only 4D perception modules that can extract 4D representations for distillation in P4D (Sec. 4.2). Moreover, 4D-RGPT can perceive temporal progression by incorporating timestamp positional encodings into input visual features. In short, we use a 4D perception decoder D4DP to extract latent 4D features and prediction heads D𝑚 for 𝑚 ∈ ℳ to extract explicit 4D signals.

Explicit 4D Representations. Although 𝐹^4D can capture rich 4D features, explicit 4D signals, e.g., depth maps, are more interpretable and provide unambiguous supervision. To capture explicit 4D repre-

sentations for P4D, we extract explicit 4D signals 𝑃^𝑚 given 𝐹^4D via the training-only prediction heads D𝑚 from the frozen 4D perception model. Specifically, for each 𝑚 ∈ ℳ, we have

𝑃^𝑚 = D𝑚(𝐹^4D). (4)

Latent 4D Representations. To capture latent 4D representations for P4D, we extract 𝐹^4D from the input video. Through the video encoder EV, multi-modal

Timestamp Positional Encoding (TPE). Accurate temporal perception, such as “when” an event

occurred and “how long” an action took, is fundamental to 4D VQA. For example, to answer “What is the average speed of the car?,” even if the MLLM can perceive depth and knows its displacement, it still needs to understand the time duration of the video to compute speed. Incorrect temporal perception can lead to significant errors in acquiring the displacement over the correct time duration, i.e., speed.

We observe that MLLMs struggle with temporal perception when there are no explicit time cues (see the experiments in Sec. 6.3 and Tab. 6). To provide temporal cues, we encode timestamps directly into the MLLM’s visual input as positional encodings. That is, for each input frame 𝐼(𝑛) from video 𝑉 that is sampled at time 𝑡(𝑛), we add a sinusoidal timestamp positional encoding 𝑝(𝑛) ∈ R𝐷 to the visual features EV(𝐼(𝑛)) before feeding them into the EP, where

)︂ and 𝑝(𝑛)[2𝑖+1] = cos(︂

)︂. (5)

𝑝(𝑛)[2𝑖] = sin (︂

𝑡(𝑛) 𝑇 2𝐷𝑖

𝑡(𝑛) 𝑇 2𝐷𝑖

Here 𝑇 is the maximum timescale and 𝑖 is the index.

#### 4.2. Perceptual 4D Distillation (P4D)

To answer 4D questions, MLLMs must understand not only semantic content but also various aspects of 4D knowledge, such as sub-pixel movements and numeric depth values. For example, to answer “Is the person moving closer to the camera?”, the MLLM must compare the depth values of the person across frames. Recent 3D/4D specialized MLLMs either rely on self-curated training datasets or exploit external models to enhance 3D knowledge. However, both are insufficient for MLLMs to fully achieve 4D understanding. Moreover, introducing external modules results in additional inference costs. Therefore, a mechanism that provides direct supervision on the MLLM’s internal 4D perception capabilities without introducing additional modules is desirable.

To this end, we propose our P4D framework. We leverage L4P [67] as the frozen expert 4D perception model (teacher) to transfer its expert representations to our student, 4D-RGPT. We use the same architecture and pre-trained weights as provided in their paper. To ensure comprehensive knowledge transfer, we propose dual-branch distillation: latent distillation and explicit distillation.

Latent Distillation. We start by introducing latent distillation to supervise the MLLM’s latent 4D representations, i.e., 𝐹^4D, on the latent space. Latent distillation serves as intermediate 4D guidance to the MLLM on the latent space. Specifically, our latent distillation loss ℒLD is defined to pull the margin ΔLD between the latent 4D features from the teacher model

𝐹4D and those from the student model 𝐹^4D:

ℒLD =

∑︁𝑁′

4D ,𝐹^(𝑛

ΔLD(𝐹(𝑛

′)

′)

4D ). (6)

𝑛′=1

Explicit Distillation. On the other hand, we introduce explicit distillation to supervise the MLLM’s explicit 4D representations, i.e., 𝑃^𝑚, on the signal space. Explicit distillation provides direct, interpretable supervision to ensure the MLLM captures accurate 4D signals in ℳ. Specifically, our explicit distillation loss ℒED is defined to pull the margin Δ𝑚 between the explicit 4D signals from the teacher model 𝑃𝑚 and those from the student model 𝑃^𝑚:

∑︁𝑁

∑︁

𝜆𝑚Δ𝑚(𝑃𝑚(𝑛),𝑃^𝑚(𝑛)), (7)

ℒED =

𝑛=1

𝑚∈ℳ

where 𝜆𝑚 describes the loss weights of each 𝑚.

Training. We optimize our 4D-RGPT using both SFT and P4D. The overall loss function is a combination of the standard cross-entropy SFT loss ℒSFT, latent distillation loss ℒLD, and explicit distillation loss ℒED. We train on various 3D / 4D conversation datasets, including RoboFAC [68], SAT [24], VSTIBench [18] (the training split), and Wolf [69]. Please refer to the supplementary material for more training details.

### 5. R4D-Bench

Recently, there has been significant progress in 3D/4D VQA [22, 23, 24, 26, 25, 18]. Several new benchmarks require MLLMs to have depth perception or understand 3D interactions among objects. However, existing benchmarks do not evaluate MLLMs on 4D region-based understanding in complex, real-world scenarios. As shown in Tab. 1, they lack the following critical properties:

- • Lack of Dynamic Scenes: Most focus on indoor scenes with minimal object interaction or constrained movement, which do not fully capture the complexity of real-world object manipulation and dynamic changes.
- • Lack of Region Prompting: Region prompts allow controlled and intuitive user queries in VQA. Without this ability, an MLLM’s interpretability and usability in practical applications are hindered.

To address these gaps, we introduce R4DBench (see the rightmost example in Fig. 3), a novel benchmark that challenges MLLMs with region-level

person drone

(a) Extract Keywords

How did the person move the drone?

###### (e) Matching

How did ⟨R1⟩ move ⟨R2⟩?

How did the person move the drone?

|person|
|---|

|person|
|---|

|[Figure 101]|[Figure 102]<br><br>|[Figure 103]<br><br>|
|---|---|---|

[Figure 104]

|[Figure 105]<br><br>|[Figure 106]<br><br>|[Figure 107]<br><br>|
|---|---|---|

[Figure 108]

[Figure 109]

[Figure 110]

(b-1) Else (c) Set-ofMarks

###### (b-1) Detect & Segment

… the person move …

|person|
|---|

###### (f) Verification

|drone|
|---|

|person|
|---|

Checked by human experts

⟨R1⟩: [0.1, 0.0, 0.3, 1.0], ⟨R2⟩: … (a) Move to the left (b) Not moving … (a) Move to the left (b) Not moving …

(b-2) If GT masks are provided

[Figure 111]

[Figure 112]

Non-region-based VQA

R4D-Bench

- Figure 3 | Curation pipeline of our R4D-Bench. Given existing non-region 4D VQA benchmarks, we (a) first extract the noun keywords from the question as candidates for objects of interest. (b) Next, if ground truth segmentation masks are provided, we use them for step (d). Otherwise, we use off-the-shelf GroundingDINO [70] and SAM2 [71] to extract segmentation masks for each object of interest. (c) We generate a SoM [59] image for the first frame. (d) We prompt Qwen-2.5VL [35] with the SoM image and the processed question to match the objects referred to in the question with the regions. (e) Finally, the generated matching results are verified by human experts.
- 4D VQA, where depth and temporal perception are critical.

[Figure 113]

referee, fighter MLLM

[Figure 114]

[Figure 115]

[Figure 116]

identify key objs

MLLM

matching

automatically match the generated region marks to the entities in the question (Matching). Finally, human annotators verify and correct any mismatches (Verification). We also trim videos to ensure all RoIs are visible in the first frame.

Is the <obj1> moving toward the red <obj2>?

Task Formulation. Given an input video 𝑉 = [𝐼(𝑛)]𝑛=1:𝑁 of 𝑁 frames, a region-prompted 4D question 𝑄, and a set of region masks 𝑀 describing the objects of interest in 𝑄 in 𝐼(1), the task is to respond with the correct or most suitable answer from a set of options.

Is the referee moving toward the red fighter?

This concludes our region prompting process. The original VQA is transformed into R4D-Bench format, where entities are replaced by region tokens, e.g., “How did ⟨𝑅1⟩ move ⟨𝑅2⟩?” with their corresponding region masks.

[Figure 117]

Benchmark. We curate R4D-Bench based on existing non-region-based 4D VQA benchmarks, i.e., STIBench [22] and VLM4D [23]. Our pipeline (Fig. 3) employs a hybrid automated and human-verified process to transform conventional VQ pairs into highly specific region-prompted questions.

Statistics. Our R4D-Bench benchmark consists of 1,517 region-prompted VQAs. Each question is a multiple-choice problem with four to five answer options. The benchmark provides region-prompted challenges to semantic and numerical 4D understanding in both static and dynamic scenes. The static split (418 VQAs) includes 3 categories: (1) Dimension Measurement; (2) 3D Video Grounding; and (3) Spatial Relation. The dynamic split (1,098 VQAs) includes 6 categories: (1) Counting objects; (2) Translational movement; (3) Rotational movement; (4) False Positive detection; (5) Speed & Acceleration estimation; and (6) Displacement & Path Length measurement. We provide more details for each question type in the supplementary material.

The process begins with a non-region-prompted 4D VQA. In the example of Fig. 3, we are given a video of two persons and a drone with the query question “How did the person move the drone?” First, we use Qwen2.5-VL [35] to perform keyword extraction (Extract) and identify objects of interest from the query question, e.g., the person and the drone. While videos from some sources, e.g., DAVIS [72], provide annotations of object masks, other real-world videos lack such detailed annotations. Hence, we leverage state-of-the-art object detection and segmentation models, i.e., GroundingDINO [70] and SAM2 [71], to generate accurate object masks (Detect & Segment) for the identified objects of interest. We then apply the segmentation masks with their corresponding keywords onto the video frame to generate an image with Set-of-Marks [59]. This serves as an intermediate and potential portrayal of the region-prompted QA before the final step of checking correctness.

### 6. Experiments

#### 6.1. Experiment Setup

Benchmarks. We evaluate our 4D-RGPT on various 4D VQA benchmarks, including our R4D-Bench and existing ones, i.e., STI-Bench [22], VLM4D-real [23], OmniSpatial [25], MMSI-Bench [26], SAT [24], and VSTI-Bench [18]. Please note that the first four benchmarks are testing-only benchmarks and are disjoint from our training data. Apart from the numerical questions in VSTI-Bench, where we report relative accuracy, we report the multiple-choice accuracy for all other benchmarks.

Since the objects of interest can be non-unique (e.g., multiple persons) and segmentation masks can be noisy, ensuring correct association between extracted keywords and found regions is critical. We check correctness with both automated and humanin-the-loop processes. We use Qwen2.5-VL [35] to

###### Answer

###### (from MH)

###### Input Video Question

[Figure 119]

[Figure 120]

- Table 2 | Evaluation on non-region-level 3D / 4D benchmarks. We report the average multiple-choice accuracy (↑) on each benchmark. For simplicity, we use the following abbreviations: STI (STI-Bench [22]), V4D (VLM4D-real [23]), MMSI (MMSI-Bench [26]), OS (OmniSpatial [25]), and VSTI (VSTI-Bench [18]).

[Figure 121]

[Figure 122]

[Figure 123]

"id": 555, "video": "videos/stibench/11356601648124485814_409_000_429_000_camera_1.mp4", "options": [

⟨R1⟩

(*)

(*)

"-0.10m/s²", "-0.40m/s²",

- "-0.70m/s²",
- "-1.30m/s²", "-0.60m/s²" ], "question": "At 2.00 sec, What is the most appropriate instantaneous acceleration of the <obj_1> over the specified time interval? (at 2s)", "response": "B", "output": "B", "task": "Speed & Acceleration", "answer": "C"

Q: At 2.0 sec, what is the instantaneous acceleration of ⟨R1⟩ (m/s^2)?

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

GPT: −0.40 Qwen: −0.40 NVILA: −0.60 Ours: −0.70

Methods STI V4D MMSI OS SAT VSTI

[Figure 128]

[Figure 129]

[Figure 130]

- ⟨R1⟩ (*)

- ⟨R2⟩

(*)

- GPT-4o [5] 34.8 60.0 30.3 47.8 57.5 38.2
- GPT-5 [2] 39.3 - 40.7 59.9 - Gemini-2.5-Pro [32] 41.4 63.5 36.9 55.4 - Gemini-1.5-Pro [31] - - - - 64.8 -

(*)

InternVL2.5-8B [73] - 42.4 28.7 - - Qwen2.5-VL-7B [35] 32.1 43.3 25.9 39.2 - VideoLLaMA3-7B [74] 35.2 46.5 - - - LLaVA-Video-7B [75] - - - - 53.5 LLaVA-OneVision-7B [76] 29.0 36.0 24.5 35.7 41.7 LLaVA-NeXT-Video-7B [77] 29.9 - 26.8 - - 40.0 VLM-3R-7B [18] - - - - - 58.8 LLaVA-Video-7B + SAT [24] - - - - 63.4 ViLaSR-7B [10] 33.4 46.9 30.2 - - SpatialReasoner-7B [12] 31.0 43.4 22.7 - - SpaceR-7B [13] 37.0 51.3 28.8 - 47.8 NVILA-Lite-8B [36] 33.8 46.5 31.3 37.2 62.0 45.2

Q: How many times did ⟨R1⟩ hit ⟨R2⟩ upward?

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

GPT: 1 Qwen: 0 NVILA: 1 Ours: 4

Figure 4 | VQA comparison among baseline MLLMs and 4D-RGPT on R4D-Bench. For the baseline MLLMs, we GPT-4o-20241120 [5], Qwen2.5VL-7B-Instruct and NVILA-Lite-8B [36]. We note that the regions labeled with (*) or (*) are not provided in R4D-Bench; they are visualized for readability.

[Figure 135]

⟨R1⟩

(*)

|use [35],|
|---|

(*)

Q: What direction is ⟨R1⟩ moving forward ?

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

GPT: right Qwen: right NVILA: left Ours: not moving

37.6 52.7 33.3 40.4 64.7 59.1 4D-RGPT-8B (Ours)

pared to other MLLMs with similar model sizes, 4D-RGPT achieves SOTA performance over opensource MLLMs and competitive performance with GPT-4o [5]. Please note that SpatialReasoner [12], ViLaSR [10], and SpaceR [13] are all trained with RL to further boost accuracy.

+3.8 +6.2 +2.0 +3.2 +2.7 +13.9

Comparison Models. We compare our 4DRGPT with various proprietary MLLMs, e.g., GPT4o [5], GPT-5 [2], Gemini-2.5-Pro [32]; open-source generalized MLLMs, e.g., Qwen2.5-VL [35]; and recent 3D/4D specialized MLLMs, e.g., SpatialReasoner [12], ViLaSR [10], and SpaceR [13].

R4D-Bench. In Tab. 3, we present quantitative comparisons of our 4D-RGPT on R4D-Bench against other MLLMs. For fair comparison, we use SoM [59] to indicate the regions of interest for all MLLMs. Additionally, for all open-source MLLMs and 4D-RGPT, we use the same number of sampled frames, i.e., 16 frames. We observe that although SpaceR [13] outperforms Qwen2.5-VL [35] in Tab. 2, it falls behind on R4D-Bench, suggesting that SpaceR is highly tuned for non-region VQA and its region understanding is weakened. Overall, 4D-RGPT achieves the best performance among all open-source MLLMs by at least 1.6% on average and 2.6% on the dynamic split.

Architecture. We select a SOTA open-source generalized MLLM, NVILA-Lite-8B [36], as our MLLM backbone, which uses SigLIP [78] as the EV and Qwen2 [79] as the LLM. For the 4D perception model E4D and D𝑚, we follow the exact architecture and weights of L4P [67]. We document training setups in the supplementary material.

#### 6.2. Main Results

In Fig. 4, we showcase two cases of 4DRGPT against other MLLMs on R4D-Bench. In both cases, the regions of interest are constantly moving. Only 4D-RGPT effectively perceives the 4D dynamics and provides the correct answers.

We present the effectiveness of 4D-RGPT in Tab. 2 and Tab. 3, showing improvements over baseline MLLMs.

Non-region-based 4D VQA. In Tab. 2, we evaluate 4D-RGPT on several non-region-level 3D/4D VQA benchmarks, including input modalities of both images and videos. We compare with various stateof-the-art proprietary MLLMs, open-source general MLLMs, and recent 3D/4D MLLMs. 4D-RGPT consistently improves over the baseline NVILA-Lite8B by a large margin across all benchmarks, especially on VLM4D [23] and VSTI-Bench [18]. Com-

#### 6.3. Ablation Studies

To justify our various designs, we conduct extensive ablation studies and analysis. For most experiments in this subsection, we report results on STI-Bench [22] and the static and dynamic question subsets of R4DBench. Without specific notes, we use the same training data, and all other components are kept identical

- Table 3 | Evaluation on R4D-Bench. We report performance on the static split (Sta), the dynamic split (Dyn), and all 9 tasks of R4D-Bench. For simplicity, we abbreviate them as follows: 3D Video Grounding (VG); Dimension Measurement (DM); Spatial Relationship (SR); Rotational (R); Counting (C); Translational (T); False Positive (FP); Speed & Acceleration (SA); and Displacement & Path Length (DP).

Methods Avg Sta Dyn VG DM SR R C T FP SA DP Random 23.4 20.0 24.7 20.0 20.0 20.0 25.0 25.0 25.0 25.0 20.0 20.0 GPT-4o [5] 42.8 30.3 47.5 30.7 26.8 43.9 49.1 35.2 51.8 54.1 27.0 10.7 Qwen2.5-VL-7B [35] 40.6 34.1 43.1 39.1 25.7 48.8 50.0 38.4 46.6 28.9 45.9 28.6 LLaVA-Video-7B [75] 39.7 26.9 44.6 23.4 28.4 36.6 46.2 30.2 50.4 33.6 48.6 35.7 ViLaSR-7B [10] 39.6 31.5 42.6 34.4 24.6 48.8 46.2 42.8 51.3 3.7 43.2 17.9 SpatialReasoner-7B [12] 38.3 31.2 41.0 35.4 25.7 36.6 43.4 37.1 49.3 11.9 32.4 17.9 SpaceR-7B [13] 37.0 26.2 41.1 30.7 18.0 41.5 47.2 40.3 43.8 25.9 51.4 21.4 NVILA-Lite-8B [36] 37.9 29.1 41.3 33.9 20.2 46.3 41.5 39.6 41.9 40.7 45.9 32.1

42.2 32.9 45.7 35.1 26.3 52.2 43.1 40.1 48.7 40.2 50.9 38.9 4D-RGPT-8B (Ours)

+4.3 +3.8 +4.4 +1.2 +6.1 +5.9 +1.6 +0.5 +6.8 -0.5 +5.0 +6.8

Table 4 | Alternative strategies for 4D VQA. We compare P4D with direct SFT (4DSFT) and straightforward designs of incorporating 𝐹4D from the 4D perception model, i.e., 4DConcat and 4D-PE. For simplicity, we use the same abbreviations as in Tab. 3 and STI for STI-Bench [22].

R4D-Bench Avg Sta Dyn Zero-shot ✗ 33.8 37.9 29.1 41.3 4D-SFT ✗ 34.7 40.1 32.2 43.8 4D-Concat ✓ 34.8 39.5 30.6 42.9 4D-PE ✓ 31.3 36.0 26.6 39.5 Ours (P4D) ✓ 37.6 42.2 32.9 45.7

Methods 𝐹4D STI

unless specified.

Alternative Strategies. Besides P4D, there are other strategies to utilize 4D conversation data or the latent feature 𝐹4D from the 4D perception models to enhance MLLMs’ 4D understanding. First, denoted as 4D-SFT, we apply solely SFT to the entire MLLM without access to 𝐹4D. Additionally, there are two straightforward ways to leverage 𝐹4D. Denoted as 4D-Concat, we directly concatenate 𝐹4D with the 2D visual features 𝐸V(𝑉 ). We note that this requires additional training on EP as the dimension differs from the original visual features. On the other hand, denoted as 4D-PE, we project 𝐹4D to positional encodings (PE) for the visual features, similar to the spatial PE proposed in SR-3D [20].

As shown in Tab. 4, apart from 4D-PE, both 4DSFT and 4D-Concat improve over the Zero-shot baseline. However, they all fall short compared to P4D. Moreover, 4D-Concat and 4D-PE require additional inference costs as they need to compute 𝐹4D for each input during inference. In comparison, P4D requires solely training-only 4D perception modules, making 4D-RGPT as efficient as Zero-shot during inference.

Perceptual 4D Distillation. To validate the effectiveness of P4D, we experiment with various distillation strategies used in latent distillation (ℒLD in Eq. (6)) and explicit distillation (ℒED in Eq. (7)). In Tab. 5, we ablate different combinations of distillation on 𝐹^4D and 𝑃^𝑚.

We first observe that applying ℒLD alone (LD-only) improves the performance over the Zero-shot baseline by 2.3% on R4D-Bench. For ℒED, adding more 𝑚 ∈ ℳ incrementally improves the performance steadily, with 𝑚 = depth and 𝑚 = flow being the most effective ones (see LD+D and LD+D+F). While ℒED alone (ED-

Table 5 | Analysis of 4D modalities in P4D. We ablate the effectiveness of different combinations of distillation in latent distillation (LD) on 𝐹^4D and explicit distillation (ED) on 𝑃^𝑚. For simplicity, we use the same abbreviations as Tab. 4 and Depth (D), Flow (F), Motion (M), and Camray (C) for each 𝑚 ∈ ℳ.

𝑃^𝑚 STI

R4D-Bench

Methods 𝐹^4D

D F M C Avg Sta Dyn Zero-shot ✗ ✗ ✗ ✗ ✗ 33.8 37.9 29.1 41.3 LD-Only ✓ ✗ ✗ ✗ ✗ 34.2 40.2 32.0 43.3 LD+D ✓ ✓ ✗ ✗ ✗ 33.4 40.8 32.5 44.0 LD+D+F ✓ ✓ ✓ ✗ ✗ 36.2 41.9 33.1 45.3 LD+D+F+M ✓ ✓ ✓ ✓ ✗ 36.5 42.0 33.1 45.4 ED-Only ✗ ✓ ✓ ✓ ✓ 35.4 39.8 31.5 42.9 Ours (LD+ED) ✓ ✓ ✓ ✓ ✓ 37.6 42.2 32.9 45.7

only) also improves the performance on R4D-Bench by 1.9%, combining both (LD+ED) achieves the best average performance, showing the complementary benefits of both LD and ED.

4D Perception Visualization. In Fig. 5, we visualize the progress of how 4D-RGPT learns to extract 4D signals through P4D. We show a video from our training set [68] with extracted 𝑃^depth at various steps. 𝑃^depth is barely meaningful at first but gradually captures the 3D structure of the scene as training proceeds. This indicates that P4D successfully distills 4D perception capabilities into 4D-RGPT.

Timestamp Positional Encoding (TPE). MLLMs often struggle with temporal perception when no explicit time cues are provided. We conduct a controlled toy experiment to validate this observation by curating a simple benchmark with VQAs that require temporal perception, such as “How many seconds have passed in the input video?” We observe that NVILALite-8B [36] is naively guessing the answers, resulting

[Figure 141]

###### Answer

###### Question

[Figure 142]

Table 7 | Ablation studies on different training designs in 4D-RGPT. We ablate different training designs on whether each module is trainable and whether to use LoRA [80]. For simplicity, we use the same abbreviations as Tab. 4.

[Figure 143]

[Figure 144]

[Figure 145]

Input video

At step 100 At step 1000

|[Figure 146]|
|---|
|[Figure 147]|

| |
|---|
| |

[Figure 148]

[Figure 149]

[Figure 150]

Trainable

R4D-Bench

[Figure 151]

[Figure 152]

Methods

STI

EV EP LLM Avg Sta Dyn Zero-shot ✗ ✗ ✗ 33.8 37.9 29.1 41.3 Tune-All ✓ ✓ ✓ 34.7 38.8 30.1 42.1 Tune-V ✓ ✗ ✗ 32.3 35.8 27.3 39.0 Tune-P ✗ ✓ ✗ 34.3 38.6 29.8 42.0 Tune-LLM ✗ ✗ ✓ 35.4 40.5 32.2 43.7 Tune-LLM-LoRA ✗ ✗ LoRA 37.0 41.1 33.0 44.2 Tune-P+LLM-LoRA ✗ ✓ LoRA 36.5 41.4 32.8 44.7 Ours (Tune-P+LLM) ✗ ✓ ✓ 37.6 42.2 32.9 45.7

[Figure 153]

[Figure 154]

[Figure 155]

Figure 5 | Predicted depth maps at different training steps. We p depth throughout training.

|Demo<br><br>visualize the<br><br>studies on|
|---|
|Demo<br><br>without cues. For as Tab. 4.|

|Demo<br><br>progress of 𝑃^<br><br>explicit temporal|
|---|
|Demo<br><br>with simplicity, we|

[Figure 156]

[Figure 157]

Table 6 | Ablation s exp ral cues. We experiment and different choices of explicit time si use the same abbreviations

Demo

[Figure 158]

[Figure 159]

### 7. Conclusion

We show that existing MLLMs struggle with regionlevel 4D VQA due to not fully perceiving 4D information. Without incurring additional inference cost, our 4D-RGPT effectively improves MLLMs’ 4D perception by learning from a 4D perception model via a novel distillation framework, P4D. Additionally, we introduce a proper benchmark, R4D-Bench, for this domain, contributing to region-level 4D VQA. Extensive experiments confirm the effectiveness of our approach on both non-region-level and region-level 4D VQA.

R4D-Bench Avg Sta Dyn Zero-shot ✗ 33.8 37.9 29.1 41.3 P4D ✗ 34.8 41.0 31.8 44.5 P4D+mark marks 35.1 41.1 31.5 44.7 P4D+prompt prompts 36.1 41.5 32.1 45.0 Ours (P4D+TPE) TPE 37.6 42.2 32.9 45.7

Methods Time cues STI

(D) 14.97 m/s (E) 7.48 m/s

3.74 m/s

- (A) 3.74 m/s
- (B) 0.75 m/s

[Figure 160]

in accuracy close to random guessing. This problem is further exacerbated by the inconsistency among multiple sources of data with different frame rates. We detail the toy experiment in the supplementary material.

### 8. Acknowledgment

We would like to express our gratitude to Abhishek Badki, Hang Su, Boyi Li, Ran Tian, Boris Ivanovic, and Marco Pavone for the model and data sharing and fruitful discussions during the 4D-RGPT development. We also appreciate the helpful discussions on problem formulation and potential applications with the VILA team (Hanrong Ye, Hongxu Yin, Yao Lu) and the Metropolis group (Vidya Murali, Varun Praveen, Tomasz Kornuta, Xiaolong Li, Zaid Pervaiz Bhat, Ryan Ji, Adityan Jothi, Thomas Tang, Paris Zhang, Yilin Zhao, Ratnesh Kumar, and Bhanu Pisupati) at NVIDIA.

Without introducing additional modules, we test two simple solutions to provide explicit temporal cues to MLLMs. First, denoted as P4D+mark, we add explicit time marks similar to SoM [59] on each 𝐼(𝑛), such as burned-in text showing the timestamp, e.g., “𝑡(𝑛) s” Second, denoted as P4D+prompt, we add explicit time information in 𝑄, such as “The following video frames are sampled from a video 19 seconds long and recorded at 30 frames per second.”

Both P4D+mark and P4D+prompt, as shown in Tab. 6, can improve 4D VQA performance. However, they require additional data preprocessing, distract MLLMs from the main visual and textual content, and do not generalize well to region-level settings, i.e., R4D-Bench. Our P4D+TPE consistently improves performance across both benchmarks, as shown in the last row of Tab. 6.

Architecture Design. In Tab. 7, we ablate different designs on whether EV, EP, or LLM is trainable or frozen. Our Tune-P+LLM achieves the best performance by tuning both EP and LLM, while keeping EV frozen. This is likely because EP requires finetuning for TPE and P4D works best on LLM.

Appendix

The appendix is organized as follows:

- • In Sec. A1, we provide implementation and training details for P4D and 4D-RGPT, including model architecture, training data, computational resources, and loss functions.
- • In Sec. A2, we provide the detailed design of R4DBench, including the nine question categories and dataset curation process.
- • In Sec. A3, we provide additional experimental results, including results with other NVILA variants, analysis of temporal perception capabilities, training data mixture, more qualitative results, and visualizations.

### A1. Additional Details

#### A1.1. Model Architecture

MLLM. As mentioned in Sec. 6.1, we use NVILALite-8B [36] as our base MLLM in the main experiments. NVILA is a unified open-sourced MLLM family that tackles both image and video understanding.

Considering the tradeoff between performance and inference efficiency, there are two groups of NVILA variants, e.g., NVILA (Base) and NVILA-Lite, where the latter is more efficient. For example, NVILA-Lite uses a 3 × 3 downsampling kernel in EP while NVILA (Base) uses 2 × 2. We select NVILA-Lite as our base MLLM due to its competitive performance and higher efficiency.

For all NVILA variants, we use their open-sourced weights from HuggingFace [81]. Specifically, we use the following checkpoints:

- • Efficient-Large-Model/NVILA-Lite-8B ;

- • Efficient-Large-Model/NVILA-Lite-15B ;

For the vision encoder (tower) EV, they use SigLIP [78], specifically

siglip-so400m-patch14-384 . For the multi-

modal projector EP, they use a 2-layer MLP with a hidden dimension of 4,608.

- 4D Perception Model. As mentioned in Sec. 6.1, we use L4P [36] as our 4D perception model. A 40layer ViT-based video encoder from VideoMAEv2 [82] is adopted for E4D, and DPT [83] is adopted for each D𝑚 where 𝑚 ∈ ℳ. Each D𝑚 has the same architecture but different output channels depending on the target modality. As mentioned

[Figure 161]

Figure A1 | An example from VSTI-Bench [18] training data. The corresponding conversation is as follows: (1) User: “These are frames of a video. Approximately how far (in meters) did the camera move between frame 14 and frame 20 of 32? Please answer the question using a single word or phrase.”; (2) GPT: “1.6”.

in Sec. 3, the output channels are 1,2,1,6 for the depth,flow,motion,camray, respectively. L4P has 1,337M parameters and takes approximately 300ms to process a 16-frame video on an A100 GPU. Since L4P is only required during training, its 4D signals can be pre-computed and stored offline, adding no inference overhead to 4D-RGPT.

4D-RGPT. In 4D-RGPT, we design a lightweight 4D perception decoder D4DP to efficiently extract 4D perceptual latent from LLM’s hidden states. It is a 3-layer MLP with a hidden dimension of 2,560. We use GELU [84] as the activation function between each layer. For initialization, we use Xavier initialization [85] for all weights and zeros for all biases. Additionally, 4D-RGPT employs Temporal Positional Encoding (TPE) to enhance the temporal understanding of the model. For TPE (Eq. (5)), we use 𝑇 = 10,000.

#### A1.2. Data Mixture

We provide more details about the training data mixture used in our training.

VSTI-Bench [18] is a new dataset built upon VSIBench [86]. While VSI-Bench focuses on the spatial understanding of static 3D scenes, VSTI-Bench further investigates the spatial-temporal understanding of how spatial relations evolve over time. We use only the training set of VSTI-Bench and do not use the VSIBench. The videos are sourced from ScanNet [87] and ScanNet++ [88]. The training set contains roughly 1.2k unique videos and 130k QA pairs. A training sample is shown in Fig. A1.

Wolf [69] is a large-scale video captioning dataset with high-quality captions generated by VLMs. Wolf provides detailed captions across three domains: autonomous driving, general scenes, and robotics. We use the NuScenes [89] portion of Wolf, i.e., the autonomous driving domain. We use Llama-3.1-70BInstruct [4] with the template-based text prompts to generate question-answer pairs based on these cap-

tions, creating conversational data suitable for 4D VQA training. The training set contains roughly 5k unique videos and 15k QA pairs. A training sample is shown in Fig. A2.

[Figure 162]

Figure A2 | An example from Wolf [69] training data. The corresponding conversation is as follows: (1) User: “What traffic participants are around the ego-centric vehicle in the video? Please count and list all of them.”; (2) GPT: “1 car is near the ego vehicle’s expected path. 1 traffic cone is in the ego vehicle’s lane and potentially blocking the ego vehicle. 4 barriers are in the ego vehicle’s lane and potentially blocking the ego vehicle.”.

RoboFAC [68] is a large-scale dataset for semantic understanding of robotic arm videos, including a training split with simulated robotic arm videos involving various actions. We adopt it into our training data mixture due to its stable camera views with limited background variations but rich robotic arm movements. It contains roughly 10k unique videos and 65k conversations. A training sample is shown in Fig. A3.

SAT [24] is an image-based VQA dataset. Though it is image-based, we consider it helpful for 4D VQA training due to its relevance on dynamic scene understanding across images. The training set contains roughly 190k unique simulated images and 170k QA pairs. A training sample is shown in Fig. A4.

#### A1.3. Training Details

Our training starts from the pre-trained NVILA weights with an initial learning rate of 1e−5. We use a cosine learning rate scheduler with a warmup ratio of 0.03. We train on a multi-node cluster comprising 8 nodes. Each node has NVIDIA A100-SXM4-80GB GPUs and an AMD EPYC 7J13 64-Core Processor CPU. The total batch size is 1,024. We train for 5 epochs over approximately 12 hours.

Losses. As mentioned in Sec. 4.2, we train our model with both SFT loss ℒSFT and P4D loss, i.e., latent distillation loss ℒLD and explicit distillation loss ℒED. Specifically, our total loss is

ℒ = ℒSFT + 𝛼ℒLD + 𝛽ℒED, (A8)

where 𝛼 and 𝛽 are hyperparameters to balance the three loss terms. We set 𝛼 = 0.5 and 𝛽 = 0.1.

[Figure 163]

- Figure A3 | An example from RoboFAC [68] training data. The corresponding conversation is as follows: (1) User: “From the video, what task is the robot engaged in?”; (2) GPT: “Put the fork in the cup and put them in the microwave.”.

[Figure 164]

(a) First frame. (b) Second frame.

[Figure 165]

- Figure A4 | An example from SAT [24] training data. The corresponding conversation is as follows:

(1) User: “Were any of the objects in the initial frame that you can still see in the second frame moved from their original positions? Options: [’green tapered square potted houseplant was moved right and towards the camera in the first frame’, ’green tapered square potted houseplant was moved left and away from the camera in the first frame’]”; (2) GPT: “green tapered square potted houseplant was moved right and towards the camera in the first frame.”.

In Eq. 6, we set ΔLD to be the Smooth-L1 distance function. In Eq. 7, we set each Δ𝑚 to be the SmoothL1 distance function and 𝜆𝑚 to be 1.0,0.1,0.05,0.05 for 𝑚 ∈ {depth,flow,motion,camray}, respectively.

### A2. R4D-Bench

We provide more details about R4D-Bench, including the 9 question categories (Sec. A2.2) and dataset curation process (Sec. A2.1).

#### A2.1. Dataset Curation

To construct R4D-Bench, we develop a hybrid automated and human-in-the-loop process that converts existing non-region-based 4D VQA benchmarks into region-based format. Recall Sec. 5 and Fig. 3, our curation process consists of the following stages.

(a) Keyord Extraction. Given a question 𝑄 and the first frame 𝐼(1) of a video, we first identify the objects mentioned in 𝑄. We employ Qwen2.5-VL32B-Instruct [35] to parse the question and extract object references. The model is given the following system prompt.

[Figure 166]

Figure A5 | An example of SoM visual input in R4D-Bench. We apply SoM [59] on 𝐼(1) to generate intermediate region-based visual inputs. The corresponding input 𝑄 is “At 9.00 sec, what is the positional relationship of the green truck model relative to the teddy bear?”

Task: You will receive (1) an RGB image (the first frame of a video) and (2) a natural-language question about objects in the image.

Instructions: Identify the object(s) mentioned in the question and wrap them with angle brackets <>. Do not change any other part of the text. If no object matches, return the original question.

###### Example:

Input: “What is the teacher right hand holding?” Output: “What is the <teacher> right hand holding?”

- (b) Detect & Segment. If the segmentation masks of the identified objects are annotated in the original source, e.g., DAVIS [90, 72], we skip this stage. Otherwise, we extract the 2D bounding boxes and segmentation masks for each identified object using a combination of GroundingDINO [70] and SAM2 [71]. Specifically, we use GrondingDINO ( IDEA-Research/grounding-dino-base from HuggingFace) to detect objects based on the extracted object classes from (a). We set both detection and text thresholds to 0.25. The detected bounding boxes are then refined using SAM2 ( sam2.1_hiera_large ) to obtain refined segmentation masks.

- (c) Set of Marks. We leverage Set-of-Mark (SoM) [59] to generate an intermediate region-based visual, serving as a bridge to convert non-region-based inputs into our final region-based format. We overlay numbered markers on the detected objects in 𝐼(1), creating an annotated image where each object is labeled with a unique ID and its class name, e.g., “0:cat”, “1:table”. An example image is shown in Fig. A5.

- (d) Matching. We feed the annotated image from (c) and 𝑄 into Qwen2.5-VL-32B-Instruct with the following prompt to match the objects in 𝑄 to the marked regions.

Task: You will receive (1) an RGB image with labeled objects (a frame from a video) and (2) a naturallanguage question.

Instructions:

- • Identify which labeled objects the question refers to
- • Replace object mentions with tokens: <obj_1>, <obj_2>, etc.
- • If no objects match, return the original question with empty obj_classes

Output Format: End your answer with “### Final Answer:” followed by JSON: {

"question": "...", "obj_classes": ["id:class_name", ...]

} Examples: Q: “What is the color of the car?” (car labeled as 1:car) A: {

"question": "What is the color of <obj_1>?", "obj_classes": ["1:car"]

} Q: “What is the color of the cars?” (two cars: 1:car, 2:car) A: {

"question": "What is the color of

<obj_1> and <obj_2>?", "obj_classes": ["1:car", "2:car"]

} Q: “What is the color of the car?” (no car labeled) A: {

"question": "What is the color of

the car?", "obj_classes": []

}

- (e) Verification. We manually verify all converted questions to ensure quality. We use Label Studio [91] to build a simple interface where human annotators can review each QA pair along with the video and the detected regions. Questions where the grounding fails, i.e., no objects detected or object misalignment,

are fixed by annotators. If a question cannot be fixed, it is filtered out. We trim down the input video if the object appears later in the video instead of the first frame. We exclude VQA sample where the object of interest in 𝑄 is too ambiguous to ground clearly for our human annotators. Overall, samples requiring correction or removal account for more than 50% of the candidate samples from the automated pipeline, underscoring the necessity of this human verification step. The final R4D-Bench contains 1,419 region-based QA pairs.

[Figure 167]

[Figure 168]

[Figure 169]

(*)

⟨R1⟩ (*)

|validation_37<br><br>validation_171 validation_972|
|---|

Q: From the camera perspective, what direction is the ⟨R1⟩ moving towards?

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

left not moving toward the camera

right

| | | |
|---|---|---|
|[Figure 174]<br><br>⟨R1⟩|[Figure 175]<br><br>(*)|[Figure 176]<br><br>(*)|

Q: Is the ⟨R1⟩ moving towards or away from the camera?

#### A2.2. Question Categories

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

right away not moving backwards

"left", "not moving", "right", "towards the camera"

R4D-Bench contains 9 question categories covering both static and dynamic aspects of 4D understanding. Of the 9 categories, 4 of them are sourced from VLM4D [23] and the other 5 are sourced from STIBench [22]. For each category, we provide its defintiion below. We also attach several video examples in the supplementary folder under r4d_examples/ .

| | | |
|---|---|---|
|[Figure 181]<br><br>⟨R1⟩|[Figure 182]<br><br>(*)|[Figure 183]<br><br>(*)<br><br>|

Q: In what direction does the person move the ⟨R1⟩

towards the end of the video?

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

away towards sideways upward

For the Translational (T), Rotational (R), Counting (C), and False Positive (FP) questions, we follow the definitions in VLM4D [23]. We downloaded the dataset from their official source on HuggingFace, i.e., shijiezhou/VLM4D . However, as of the time of writing, they do not provide the list of QA pairs for each category. Therefore, we leverage Qwen2.5-VL-32B-Instruct [35] and human annotators to classify each QA pair into the 4 categories. Of the region-based QA pairs in R4D-Bench obtained from VLM4D, the distribution across different categories is as follows:

#### Figure A6 | Translational questions in R4D-

Bench. We note that the regions labeled with (*) are not provided in R4D-Bench; they are visualized for readability.

dataset from their official source on HuggingFace, i.e., MINT-SJTU/STI-Bench . We note that the original STI-Bench contains two additional categories, i.e., Ego-centric Orientation and Trajectory Description, where these questions focuses on the ego-centric 4D understanding from the viewpoint itself. Since R4D-Bench focuses on region-based 4D VQA, where another region of interest needs to be provided, these questions are not applicable and removed from R4DBench.

- • Translational: 61.3%
- • Rotational: 10.2%
- • Counting: 15.4%
- • False Positive: 13.1%

The followings are the detailed explanations for each category:

In comparison, the official VLM4D benchmark has the following distribution:

Translational (T) questions target the MLLM’s capabilities to understand the linear movement of objects. They usually involve the following movementrelated diretion, such as left, right, north, south, away, towards, etc. We provide several examples of R4DBench translational questions in Fig. A6.

- • Translational: 55%
- • Rotational: 19%
- • Counting: 17%
- • False Positive: 9%

Our categorization results are largely consistent with the official distribution with slight difference.

For the 3D Video Grounding (VG), Spatial Relationship (SR), Dimension Measurement (DM), Displacement & Path Length (DP), and Speed & Acceleration (SA) questions, we follow the definition of STI-Bench [22]. We downloaded the

Rotational (R) questions, on the other hand, care about the rotational movement of objects. They usually involve the following movement-related words, such as rotate, spin, twist, turn, etc. We provide several examples of R4D-Bench rotational questions in Fig. A7.

Counting (C) questions focusing on the MLLM’s

|[Figure 189]<br><br>⟨R1⟩|[Figure 190]<br><br>(*)|[Figure 191]<br><br>| |
|---|
<br><br>(*)|
|---|---|---|
| | | |

| | | |
|---|---|---|
|[Figure 192]<br><br>⟨R1⟩|[Figure 193]<br><br>(*)|[Figure 194]<br><br>(*)|
| | | |

validation_1313

Q: Is ⟨R1⟩ spinning clockwise or counter-clockwise?

Q: How many times did ⟨R1⟩ dribble the ball with his left hand?

validation_10 validation_20

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

no dancers counterclockwise

clockwise

no spinning

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

4 8 1 0

validation_68 validation_77 validation_127

[Figure 203]

[Figure 204]

[Figure 205]

|[Figure 206]<br><br>⟨R1⟩|[Figure 207]<br><br>(*)|[Figure 208]<br><br>(*)|
|---|---|---|

(*)

(*)

(*)

(*)

Q: How many times did ⟨R1⟩, ⟨R2⟩, ⟨R3⟩ are facing away from the camera?

Q: Is the ⟨R1⟩ spinning clockwise or counter-clockwise?

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

1 0 3 2

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

clockwise no cars not moving counterclockwise

|[Figure 217]<br><br>|[Figure 218]<br><br>(*)<br><br>|[Figure 219]<br><br>(*)|
|---|---|---|

|[Figure 220]<br><br>⟨R1⟩|[Figure 221]<br><br>(*)<br><br>|[Figure 222]<br><br>(*)|
|---|---|---|

Q: How many spoonfuls of ⟨R1⟩ did the person pour to the right?

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

3 2 4 1

Q: Is ⟨R1⟩ turning to the left or right from its own perspective?

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

left right not moving not sure

- Figure A8 | Counting questions in R4D-Bench. We note that the regions labeled with (*), (*), or (*) are not provided in R4D-Bench; they are visualized for readability.

[Figure 232]

- Figure A9 | False positive questions in R4DBench. We note that the regions labeled with (*) are not provided in R4D-Bench; they are visualized for readability.

Figure A7 | Rotational questions in R4D-Bench. We note that the regions labeled with (*) are not provided in R4D-Bench; they are visualized for readability.

[Figure 233]

ability to accurately count the number of objects or occurrences of actions. We provide several examples of R4D-Bench counting questions in Fig. A8.

False Positive (FP) questions are designed to trick the MLLM. The questions will intentionally describe events that do not actually occur within the video,

- e.g., asking about movements when no object is moving. We note that the original VLM4D false positive questions also ask about objects that do not exist in the video. Due to the nature of region-based 4D VQA in R4D-Bench, we do not include these types of questions since the regions cannot refer to nonexistent objects. We provide several examples of R4D-Bench false positive questions in Fig. A9.

- 3D Video Grounding (VG) questions ask MLLMs to retrive the 3D bounding box of objects. The options are formatted as JSON with “dimension (size)” ∈ R3, “central point (coordinate)” ∈ R3 and “orientation” ∈ R3, (i.e., yawn, pitch, and roll) or “camera heading” ∈ R1. We provide an example in Fig. A10. As shown in the example, the MLLM needs to be fairly precise to answer these questions correctly, as the differences between options can be quite small.

Spatial Relationship (SR) questions assess the 3D spatial relationship between selected objects or the

|| | | |
|---|---|---|
|[Figure 241]<br><br>⟨R1⟩|[Figure 242]<br><br>(*)|[Figure 243]<br><br>| |
|---|
<br><br>(*)|
| | | |
|[Figure 237]<br><br>⟨R1⟩<br><br>⟨R2⟩|[Figure 238]<br><br>(*)<br><br>(*)|[Figure 239]<br><br>(*) (*)<br><br>|
|---|---|---|---|
| | | | |

[Figure 240]

| | | |
|---|---|---|
|[Figure 241]<br><br>⟨R1⟩|[Figure 242]<br><br>(*)|[Figure 243]<br><br>| |
|---|
<br><br>(*)|
| | | |

Q: At 7.00 sec, what is the positional relationship of the ⟨R1⟩ relative to

the ⟨R2⟩?

|stibench/16 stibench/80 424|
|---|

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

left right front back up

Q: At 7.00 sec, identify the correct 3D bounding box localization for ⟨R1⟩

from a single frame. (unit: cm, °).

[Figure 249]

[Figure 250]

[Figure 251]

stibench/16 validation_137 validation_175

|{<br><br>dimensions: [23.62, 3.51, 2.79], central_point: [5.88, 9.73, 51.40], orientation: {<br><br>yaw: 167.42, pitch: 15.93, roll: 59.99<br><br>} }|
|---|

|{<br><br>dimensions: [22.87, 3.12, 2.79], central_point: [5.88, 9.73, 51.15], orientation: {<br><br>yaw: 167.42, pitch: 12.18, roll: 63.74<br><br>} }|
|---|

[Figure 252]

[Figure 253]

⟨R2⟩

(*) (*)

| |
|---|

⟨R1⟩

(*)

|&|
|---|

#### Figure A13 | Displacement

path length questions in R4D-Bench. We note that the regions labeled with (*) are not provided in R4D-Bench; they are visualized for readability.

(*)

Q: At 0.00 sec, what is the most likely minimum relative distance between ⟨R1⟩ and ⟨R2⟩ (unit: cm)?

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

51.52

59.00 54.63 47.78 64.12

| | | |
|---|---|---|
|[Figure 259]<br><br>⟨R1⟩|[Figure 260]<br><br>(*)|[Figure 261]|
| | | |

Figure A10 | 3D video grounding questions in R4D-Bench. We note that the regions labeled with (*) are not provided in R4D-Bench; they are visualized for readability. For simplicity, we only show 1 correct option and 1 wrong option here, but there are 5 options for each 3D video grounding question in R4D-Bench.

Q: At 3.00 sec, What is the most appropriate instantaneous speed of ⟨R1⟩ over the specified time interval?

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

3.74 m/s 0.75 m/s 0.00 m/s 14.97 m/s 7.48 m/s

##### Figure A14 | Speed & acceleration questions in R4D-Bench. We note that the regions labeled with (*) are not provided in R4D-Bench; they are visualized for readability.

[Figure 268]

[Figure 269]

relation questions in Fig. A11.

Dimension Measurement (DM) questions care about the physical measurements of objects, such as size and distance. They usually require MLLMs to understand and perceive depth information in order to predict the numerical values. We provide an example of R4D-Bench dimension measurement questions in Fig. A12.

|Figure A11 | Spat Bench. The quest ship at 7 seconds, frame out of the<br><br>[Figure 270]<br><br>⟨R1⟩<br><br>⟨R2⟩|ial relation qu ion asks about<br><br>which correspo three frames sho<br><br>[Figure 271]<br><br>(*)<br><br>(*)|estions in R4D-<br><br>the spatial relationnds to the middle wn. We note that<br><br>[Figure 272]<br><br>(*) (*)<br><br>|
|---|---|---|
| | | |

F

- B s

- f the regions labeled with (*) or (*) are not provided in R4D-Bench; they are visualized for readability.

Q: At 7.00 sec, what is the positional relationship of the ⟨R1⟩ relative to

the ⟨R2⟩?

|stibench/16 stibench/80<br><br>Path424 Length (DP) questions<br><br>|
|---|

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

left right front back up

#### Displacement & P

measures the travel distance of objects. They often involve MLLMs to track motion across selected frames. We provide an example of R4D-Bench displacement and path length questions in Fig. A13.

| | | |
|---|---|---|
|[Figure 278]<br><br>⟨R2⟩ ⟨R1⟩<br><br>|[Figure 279]<br><br>(*)<br><br>(*)|[Figure 280]<br><br>| |
|---|
<br><br>(*)<br><br>| |
|---|
<br><br>(*)|
| | | |

Speed & Acceleration (SA) questions estimate the motion dynamics of objects. The MLLM needs to consider both the displacement and time intervals to answer them correctly. We provide an example of R4D-Bench speed and acceleration questions in Fig. A14.

Q: At 0.00 sec, what is the most likely minimum relative distance between ⟨R1⟩ and ⟨R2⟩ (unit: cm)?

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

51.52 59.00 54.63 47.78 64.12

| | | |
|---|---|---|
|[Figure 286]<br><br>⟨R1⟩<br><br>Figure A12 | Dime in R4D-Bench. with (*) or (*) are are visualized for<br><br>|[Figure 287]<br><br>(*)<br><br>nsion measur We note that<br><br>|not|
|---|
<br><br>provided in readability.|[Figure 288]<br><br>ement questions the regions labeled<br><br>R4D-Bench; they|
| | | |

F i

w a

### A3. Additional Results

Q: At 3.00 sec, What is the most appropriate instantaneous speed of ⟨R1⟩ over the specified time interval?

More NVILA variants. In Tab. A1 and Tab. A2, we provide additional results using NVILA-Lite-15B as the base MLLM on non-region-based 4D VQA and

camera. The options usually involve relative positioning terms, such as left, right, front, back, up, down, etc. We provide an example of R4D-Bench spatial

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

3.74 m/s 0.75 m/s 0.00 m/s 14.97 m/s 7.48 m/s

Q: How much time has passed in the video?

Input Video

5.8s

39.40 seconds. (2.00×) 9.85 seconds. (0.50×) 19.70 seconds. 59.10 seconds. (4.00×) 4.92 seconds. (0.25×)

12.0s

(From MH)

4D-RGPT: Toward Region-level 4D Understanding via Perceptual Distillation

Input Video

[Figure 306]

R4D-Bench, respectively. We observe consistent performance improvements across various benchmarks.

[Figure 307]

[Figure 308]

[Figure 309]

#### Table A1 | Evaluation on non-region-level 3D /

4D benchmarks. We report the average multiplechoice accuracy (↑) on each benchmark. For simplicity, we use the following abbreviations: STI (STIBench [22]), V4D (VLM4D-real [23]), MMSI (MMSIBench [26]), OS (OmniSpatial [25]), and VSTI (VSTIBench [18]).

Q: How much time has passed in the video?

(a) 39.40 s (2.00×) (b) 9.85 s (0.50×) (c) 19.70 s (d) 59.10 s (4.00×) (e) 4.92 s (0.25×)

[Figure 310]

Figure A15 | TimeBench VQA. We curate a toy benchmark to evaluate MLLMs’ temporal perception. We note that the “(𝑀×)” indicates the multiplier between the wrong option and the correct one. They are not provided in the actual question but are shown here for clarity.

Methods STI V4D MMSI OS SAT VSTI NVILA-Lite-8B 33.8 46.5 31.3 37.2 62.0 45.2

37.6 52.7 33.3 40.4 64.7 59.1 4D-RGPT-8B (Ours)

+3.8 +6.2 +2.0 +3.2 +2.7 +13.9

NVILA-Lite-15B 34.2 45.1 29.5 41.0 62.7 42.4

38.1 53.7 31.7 42.7 65.3 58.6 4D-RGPT-15B (Ours)

+3.9 +8.6 +2.2 +1.7 +2.6 +16.2

- Table A3 | Ablation studies on explicit temporal cues. We experiment without and with different choices of explicit time cues. For simplicity, we use the same abbreviations as Tab. 4.

Methods Time cues TimeBench STI R4D Zero-shot ✗ 22.7 33.8 37.9 P4D ✗ 30.1 34.8 41.0 P4D+mark marks 95.3 35.1 41.1 P4D+prompt prompts 98.0 36.1 41.5

We observe that both P4D+mark and P4D+prompt can greatly improve the performance on TimeBench, which is expected since they provide explicit temporal cues. However, they require additional data preprocessing and distract MLLMs from the main visual and textual content. This toy experiment inspires us to develop methods that can provide temporal cues without modifying the input data, i.e., our TPE.

Training Data Mixture. We incrementally add different datasets to analyze their contributions. In Tab. A4, we observe that compared to the Zeroshot baseline, adding the training data gradually improves the performance on both non-region-based (STI-Bench) and region-based 4D VQA (R4D-Bench). Though SAT [24] is an image-based VQA dataset, adding it also brings moderate performance gains.

- Table A4 | Incremental training data mixture. We use the same abbreviations as Tab. 4 and the following for each dataset: VSTI-Bench [18] (V); Wolf [69] (W); RoboFAC [68] (R); SAT [24] (S).

##### Table A2 | Evaluation on R4D-Bench. We report performance on the static split (Sta), the dynamic split (Dyn), and all 9 tasks of R4D-Bench. For simplicity, we abbreviate them as follows: 3D Video Grounding (VG); Dimension Measurement (DM); Spatial Relationship (SR); Rotational (R); Counting (C); Translational (T); False Positive (FP); Speed & Acceleration (SA); and Displacement & Path Length (DP).

Methods Avg Sta Dyn VG DM SR R C T FP SA DP NVILA-Lite-8B 37.9 29.1 41.3 33.9 20.2 46.3 41.5 39.6 41.9 40.7 45.9 32.1

42.2 32.9 45.7 35.1 26.3 52.2 43.1 40.1 48.7 40.2 50.9 38.9 4D-RGPT-8B (Ours)

+4.3 +3.8 +4.4 +1.2 +6.1 +5.9 +1.6 +0.5 +6.8 -0.5 +5.0 +6.8 NVILA-Lite-15B 39.7 31.7 42.7 36.5 26.8 31.7 50.9 34.0 46.4 34.8 37.8 21.4

43.0 35.8 45.7 38.5 32.2 39.0 50.0 38.4 49.6 36.3 45.9 28.6 4D-RGPT-15B (Ours)

###### +3.3 +4.1 +3.10 +2.0 +5.4 +7.3 -0.9 +4.4 +3.2 +1.5 +7.9 +7.2

Temporal Perception. As discussed in Sec. 4.1 and Sec. 6.3, we observe that MLLMs tend to struggle with temporal perception. To demonstrate such a deficiency, we conduct a toy experiment. As shown in Fig. A15, we curate TimeBench, a simple set of VQA questions that require temporal perception of input frames, such as “How many seconds have passed in the input video?”. All videos are acquired from the STI-Bench [22] and VLM4D [23]. We note that these two benchmarks have 4 different frame rates, ranging from 10 to 30, as shown in Tab. 1. This makes it even more challenging for MLLMs to infer time duration. To avoid ambiguity in answers, we provide 4 extra options for each question, ranging from 0.25× to 4× of the actual time duration.

Zero-shot and P4D in Tab. A3 show that without cues, MLLMs struggle to know how much time has passed in the input frames. The baselines are naively guessing the answers, resulting in an accuracy close to random guessing (20%). This problem is further exaggerated by the inconsistency that different sources of training data and evaluation benchmarks have different frame rates.

R4D-Bench Avg Sta Dyn Zero-shot ✗ ✗ ✗ ✗ 33.8 37.9 29.1 41.3 V ✓ ✗ ✗ ✗ 35.4 39.4 30.0 42.9 V+W ✓ ✓ ✗ ✗ 36.0 40.6 31.0 44.2 V+W+R ✓ ✓ ✓ ✗ 37.0 41.8 32.2 45.4 V+W+R+S (Ours) ✓ ✓ ✓ ✓ 37.6 42.2 32.9 45.7

Methods V W R S STI

|<br><br>⟨R1⟩|<br><br>|[Figure 314]|
|---|---|---|

|<br><br>⟨R1⟩<br><br>⟨R2⟩|<br><br>(*)<br><br>(*)|<br><br>(*)<br><br>(*)<br><br>|
|---|---|---|

|<br><br>⟨R1⟩|<br><br>| |
|---|
<br><br>(*)|<br><br>| |
|---|
<br><br>(*)|
|---|---|---|
| | | |

|[Figure 321]<br><br>⟨R1⟩|[Figure 322]<br><br>|[Figure 323]<br><br>|
|---|---|---|

|<br><br>⟨R1⟩|<br><br>||
|---|---|---|

|[Figure 327]<br><br>⟨R1⟩<br><br>⟨R2⟩|[Figure 328]<br><br>(*)<br><br>(*)<br><br>|[Figure 329]<br><br>(*)<br><br>(*)<br><br>|
|---|---|---|

|[Figure 330]<br><br>⟨R1⟩|<br><br>| |
|---|
<br><br>(*)|<br><br>| |
|---|
<br><br>(*)|
|---|---|---|
| | | |

|<br><br>⟨R1⟩|<br><br>|<br><br>|
|---|---|---|

Q: From 0.00 sec to 1.06 sec, what is the most appropriate height of ⟨R1⟩ ? (unit: m, m/s, m/s², °)

Q: Are ⟨R1⟩ picking up or putting down the ⟨R2⟩?

Q: What direction is ⟨R1⟩ moving towards?

Q: At 0.28 sec, What is the most appropriate trajectory length (total distance traveled) of ⟨R1⟩ between two frames? (unit: m, m/s, m/s², °)

Ours: 1.86 m GPT: 1.86

Ours: picking up

GPT: putting down

Ours: not moving GPT: not moving

Q: From 0.00 sec to 1.06 sec, what is the most appropriate height of ⟨R1⟩ ? (unit: m, m/s, m/s², °)

Q: Are ⟨R1⟩ picking up or putting down the ⟨R2⟩?

Q: What direction is ⟨R1⟩ moving towards?

Q: At 0.28 sec, What is the most appropriate trajectory length (total distance traveled) of ⟨R1⟩ between two frames? (unit: m, m/s, m/s², °)

|<br><br>⟨R1⟩<br><br>⟨R2⟩|<br><br>(*)<br><br>(*)|[Figure 344]<br><br>(*)<br><br>(*)<br><br>| |
|---|
|
|---|---|---|
| | | |

|<br><br>⟨R1⟩|<br><br>(*)|<br><br>(*)|
|---|---|---|

|<br><br>⟨R1⟩|<br><br>(*)|<br><br>(*)|
|---|---|---|

[Figure 351]

[Figure 352]

Ours: 0.0 m GPT: 0.2 m

[Figure 355]

[Figure 356]

[Figure 357]

Ours: 1.86 m GPT: 1.86

Ours: picking up

GPT: putting down

Ours: not moving GPT: not moving

|⟨R1⟩<br><br>[Figure 359]<br><br>⟨R1⟩|[Figure 360]<br><br>(*)|[Figure 361]<br><br>(*)|
|---|---|---|

|<br><br>⟨R1⟩<br><br>⟨R2⟩|<br><br>(*)<br><br>(*)|<br><br>(*)<br><br>(*)<br><br>| |
|---|
|
|---|---|---|
| | | |

|[Figure 365]<br><br>⟨R1⟩|[Figure 366]<br><br>(*)|[Figure 367]<br><br>(*)|
|---|---|---|

|[Figure 368]<br><br>⟨R1⟩|<br><br>(*)|<br><br>(*)|
|---|---|---|

Ours: 0.0 m GPT: 0.2 m

|⟨R1⟩<br><br><br><br>⟨R1⟩|<br><br>(*)|<br><br>(*)|
|---|---|---|

Q: At 0.00 sec, What is the most likely minimum relative distance between ⟨R1⟩ and ⟨R2⟩ in a given frame? (unit: cm, °)

Q: Is the ⟨R1⟩ moving upwards or downwards?

Q: How many scoops of ⟨R1⟩ does he move left?

Ours: 1 GPT: 2 Ans: 0

Ours: downwards

GPT: downwards

Ours: 18.54 cm GPT: 16.71 cm

Q: From 0.00 sec to 9.50 sec, what is the most likely displacement (straight-line distance) of the camera or object between two frames for ⟨R1⟩?

Q: At 0.00 sec, What is the most likely minimum relative distance between ⟨R1⟩ and ⟨R2⟩ in a given frame? (unit: cm, °)

Q: Is the ⟨R1⟩ moving upwards or downwards?

Q: How many scoops of ⟨R1⟩ does he move left?

|<br><br>⟨R1⟩|<br><br>(*)|<br><br>(*)|
|---|---|---|

|<br><br>⟨R1⟩|<br><br>(*)|<br><br>(*)|
|---|---|---|

|<br><br>⟨R1⟩<br><br>⟨R2⟩<br>|<br><br>(*)<br><br>(*)|[Figure 391]<br><br>(*)<br><br>(*)|
|---|---|---|
| | | |

[Figure 394]

[Figure 395]

[Figure 396]

Ours: 1 GPT: 2 Ans: 0

Ours: downwards

GPT: downwards

Ours: 18.54 cm GPT: 16.71 cm

Q: From 0.00 sec to 9.50 sec, what is the most likely displacement (straight-line distance) of the camera or object between two frames for ⟨R1⟩?

[Figure 399]

[Figure 400]

[Figure 401]

Ours: 7.53 m GPT: 10.06 m Ans: 8.50 m

|[Figure 402]<br><br>⟨R1⟩|<br><br>(*)|<br><br>(*)|
|---|---|---|

|[Figure 405]<br><br>⟨R1⟩|[Figure 406]<br><br>(*)|[Figure 407]<br><br>(*)|
|---|---|---|

|<br><br>⟨R1⟩<br><br>⟨R2⟩<br>|<br><br>(*)<br><br>(*)<br><br>|<br><br>(*)<br><br>(*)|
|---|---|---|
| | | |

Figure A16 | More VQA comparison between GPT-4o [5] and 4D-RGPT (Ours) on R4DBench. We provide 2 examples for each of the following categories: Displacement & Path Length.

Ours: 7.53 m GPT: 10.06 m Ans: 8.50 m

Q: At 6.00 sec, What is the positional relationship of ⟨R1⟩ relative to ⟨R2⟩ from the observer's perspective?

Q: At 27.00 sec, given a single frame, determine the 3D

Q: Are ⟨R1⟩ turning clockwise or counter-clockwise?

bounding box of ⟨R1⟩. Identify the correct dimensions, central point, and orientation including yaw, pitch, and roll. (unit: cm, °)

Ours: clockwise

GPT: counter-clockwise

Ours: left GPT: left

Q: At 6.00 sec, What is the positional relationship of ⟨R1⟩ relative to ⟨R2⟩ from the observer's perspective?

Q: At 27.00 sec, given a single frame, determine the 3D

Q: Are ⟨R1⟩ turning clockwise or counter-clockwise?

bounding box of ⟨R1⟩. Identify the correct dimensions, central point, and orientation including yaw, pitch, and roll. (unit: cm, °)

| | | |
|---|---|---|
|<br><br>⟨R1⟩<br><br>⟨R2⟩|<br><br>(*) (*)<br><br>|[Figure 420]<br><br>(*)|

[Figure 421]

[Figure 422]

Ours: clockwise

GPT: counter-clockwise

|{<br><br>dimensions: [25.62, 2.38, 15.33], central_point: [12.91, 2.77, 90.59], orientation: {<br><br>yaw: 117.10, pitch: 42.61, roll: 114.41<br><br>} }|
|---|

| |
|---|

Ours & GPT:

Ours: left GPT: left

⟨R1⟩

(*)

| | | |
|---|---|---|
|<br><br>⟨R1⟩<br><br>⟨R2⟩|<br><br>(*) (*)<br><br>|<br><br>(*)|

More Qualitative Results. Following the format in Fig. 4, we provide additional qualitative results on R4D-Bench in Fig. A16, Fig. A17, Fig. A18, and Fig. A19.

[Figure 432]

[Figure 433]

[Figure 434]

|{<br><br>dimensions: [25.62, 2.38, 15.33], central_point: [12.91, 2.77, 90.59], orientation: {<br><br>yaw: 117.10, pitch: 42.61, roll: 114.41<br><br>} }|
|---|

[Figure 435]

| |
|---|

Ours & GPT:

(*)

⟨R1⟩

(*)

Q: Is ⟨R1⟩ turning clockwise or counter-clockwise?

Q: At 3.00 sec, What is the positional relationship of the ⟨R2⟩ relative to ⟨R1⟩?

(*)

Ours: counter-clockwise

GPT: clockwise

More 𝑃^𝑚 Visualizations. In Fig. A20, we provide additional visualizations of the 4D-RGPT explicit signals 𝑃^𝑚 at different training steps. In earlier steps, we observe inaccurate predictions with grid-like structures. We hypothesize that this is due to the tokenization process in hidden states of the LLM transformer, i.e., 𝐹hidden. However, as training proceeds, the grid-like structures gradually diminish, leading to smoother and more reasonable predictions. We demonstrate that our 4D-RGPT can effectively learn to extract explicit 4D perceptual signals through the training of P4D.

Ours: left GPT: left

Q: Is ⟨R1⟩ turning clockwise or counter-clockwise?

Q: At 3.00 sec, What is the positional relationship of the ⟨R2⟩ relative to ⟨R1⟩?

|<br><br>⟨R1⟩|||
|---|---|---|
| | | |

|<br><br>⟨R1⟩|||
|---|---|---|
| | | |

|<br><br>⟨R1⟩|<br><br>(*)|[Figure 448]<br><br>(*)|
|---|---|---|

[Figure 449]

[Figure 450]

Ours: counter-clockwise

GPT: clockwise

Ours: left GPT: left

|[Figure 453]<br><br>⟨R1⟩|[Figure 454]|[Figure 455]|
|---|---|---|
| | | |

|[Figure 456]<br><br>⟨R1⟩|||
|---|---|---|
| | | |

|<br><br>⟨R1⟩|<br><br>(*)|<br><br>(*)|
|---|---|---|

Q: How many ⟨R1⟩ are standing still?

Q: At 18.52 sec, what is the 3D bounding box in camera

Q: At 6.00 sec, what is the most appropriate average or instantaneous speed of ⟨R1⟩?

coordinates of the ⟨R1⟩ from a single randomly selected frame? (unit: m, m/s, m/s^2, °)

Ours: 5

###### GPT: 3

Q: How many ⟨R1⟩ are standing still?

Ours: 4.60 m/s GPT: 4.60 m/s

Q: At 18.52 sec, what is the 3D bounding box in camera

Q: At 6.00 sec, what is the most appropriate average or instantaneous speed of ⟨R1⟩?

|<br><br>⟨R1⟩|<br><br>(*)|<br><br>(*)|
|---|---|---|

coordinates of the ⟨R1⟩ from a single randomly selected frame? (unit: m, m/s, m/s^2, °)

[Figure 469]

[Figure 470]

Ours: 5

###### GPT: 3

Ours:

GPT:

| | | |
|---|---|---|
|<br><br>⟨R1⟩|<br><br>(*)|[Figure 475]<br><br>(*)|

Ours: 4.60 m/s GPT: 4.60 m/s

|{<br><br>C_lwh: [0.86, 1.26, 0.74], C_central_point: [3.55, 1.33, 2.75], C_heading: 27.51<br><br>}|
|---|

|{<br><br>C_lwh: [0.86, 1.26, 0.74], C_central_point: [3.74, 1.39, 2.81], C_heading: 27.20<br><br>}|
|---|

|[Figure 478]<br><br>⟨R1⟩| |[Figure 479]<br><br>(*)|[Figure 480]<br><br>(*)|
|---|---|---|---|
|Q|: How many times does the ⟨R1⟩ jump towards the camera?| | |

[Figure 481]

Ours:

GPT:

| | | |
|---|---|---|
|<br><br>⟨R1⟩|<br><br>(*)|<br><br>(*)|

|{<br><br>C_lwh: [0.86, 1.26, 0.74], C_central_point: [3.55, 1.33, 2.75], C_heading: 27.51<br><br>}|
|---|

|{<br><br>C_lwh: [0.86, 1.26, 0.74], C_central_point: [3.74, 1.39, 2.81], C_heading: 27.20<br><br>}|
|---|

Limitations. 4D-RGPT can still produce suboptimal results, particularly in questions requiring precise numerical estimation, e.g., exact speed or displacement values, as illustrated in several failure cases in Fig. A17–A16. We attribute this to the lack of step-by-step reasoning during training.

Q: How many times does the ⟨R1⟩ jump towards the camera?

Q: At 14.00 sec, what is the most appropriate average or instantaneous speed of ⟨R1⟩?

[Figure 486]

Ours: 1

GPT: 1

Ours: 0.00 m/s GPT: 0.20 m/s

Q: At 14.00 sec, what is the most appropriate average or instantaneous speed of ⟨R1⟩?

[Figure 492]

[Figure 493]

Ours: 1 GPT: 1

Ours: 0.00 m/s GPT: 0.20 m/s

Figure A17 | More VQA comparison between GPT-4o [5] and 4D-RGPT (Ours) on R4DBench. We provide 2 examples for each of the following categories: Translational, Rotational, and Counting.

|[Figure 496]<br><br>[Figure 497]| |
|---|---|

|<br><br><br><br>[Figure 500]<br><br>⟨R2⟩<br><br>(*)<br><br><br><br><br><br>⟨R1⟩ (*)<br><br>⟨R2⟩<br><br>|(*)<br><br>[Figure 503]<br><br>(*)|
|---|---|

|[Figure 504]<br><br>⟨R1⟩<br><br>[Figure 505]<br><br>⟨R1⟩|[Figure 506]<br><br>[Figure 507]<br><br>(*)| |[Figure 508]<br><br>[Figure 509]<br><br>|
|---|---|---|---|
| | | | |

[Figure 510]

[Figure 511]

| |[Figure 512]<br><br>⟨R1⟩<br><br>[Figure 513]<br><br>⟨R1⟩|<br><br><br><br><br><br>|<br><br>|
|---|---|---|---|

[Figure 518]

[Figure 519]

⟨R1⟩ (*) (*)

(*) (*)

(*)

- ⟨R1⟩

(*)

- ⟨R2⟩ (*)

- ⟨R1⟩

(*)

- ⟨R2⟩ (*)

(*) (*)

Q: From 0.00 sec to 1.06 sec, what is the most appropriate height of ⟨R1⟩ ? (unit: m, m/s, m/s², °)

Q: From 0.00 sec to 1.06 sec, what is the most appropriate height of ⟨R1⟩ ? (unit: m, m/s, m/s², °)

Q: Are ⟨R1⟩ picking up or putting down the ⟨R2⟩?

Q: What direction is ⟨R1⟩ moving towards?

Q: At 0.28 sec, What is the most appropriate trajectory length (total distance traveled) of ⟨R1⟩ between two frames? (unit: m, m/s, m/s², °)

Q: Are ⟨R1⟩ picking up or putting down the ⟨R2⟩?

Q: What direction is ⟨R1⟩ moving towards?

Q: At 0.28 sec, What is the most appropriate trajectory length (total distance traveled) of ⟨R1⟩ between two frames? (unit: m, m/s, m/s², °)

[Figure 520]

[Figure 521]

[Figure 524]

[Figure 525]

Ours: 1.86 m GPT: 1.86

Ours: picking up

GPT: putting down

Ours: not moving GPT: not moving

[Figure 526]

[Figure 527]

[Figure 530]

[Figure 531]

Ours: 1.86 m GPT: 1.86

Ours: picking up

GPT: putting down

Ours: not moving GPT: not moving

|[Figure 532]<br><br>(*)<br><br>| |
|---|
<br><br>[Figure 533]<br><br>(*)<br><br>| |
|---|
| |
|---|---|

|<br><br><br><br>[Figure 536]<br><br>⟨R1⟩ ⟨R1⟩ (*)<br><br><br><br><br><br>|(*)| |
|---|---|
| | |
<br><br>| | |
|---|---|
| | |
|(*)<br><br>[Figure 539]<br><br>(*)|
|---|---|

|[Figure 540]<br><br>⟨R1⟨R1⟩ ⟩<br><br>[Figure 541]<br><br>|[Figure 542]<br><br>(*)<br><br>[Figure 543]<br><br>(*)|[Figure 544]<br><br>(*)<br><br>[Figure 545]<br><br>(*)|
|---|---|---|

Ours: 0.0 m GPT: 0.2 m

[Figure 548]

[Figure 549]

[Figure 551]

Ours: 0.0 m GPT: 0.2 m

[Figure 552]

[Figure 553]

(*)

(*)

| |⟨R1⟩<br><br>[Figure 554]<br><br>⟨R1⟩<br><br>⟨R1⟩<br><br>[Figure 555]<br><br>⟨R1⟩|<br><br>(*) (*)<br><br><br><br>|<br><br>(*) (*)<br><br><br><br>|
|---|---|---|---|

⟨R1⟩

⟨R1⟩

Q: At 0.00 sec, What is the most likely minimum relative distance between ⟨R1⟩ and ⟨R2⟩ in a given frame? (unit: cm, °)

Q: At 0.00 sec, What is the most likely minimum relative distance between ⟨R1⟩ and ⟨R2⟩ in a given frame? (unit: cm, °)

Q: Is the ⟨R1⟩ moving upwards or downwards?

Q: How many scoops of ⟨R1⟩ does he move left?

Q: Is the ⟨R1⟩ moving upwards or downwards?

Q: How many scoops of ⟨R1⟩ does he move left?

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 565]

[Figure 566]

Ours: 1 GPT: 2 Ans: 0

Ours: downwards

GPT: downwards

[Figure 569]

Ours: 1 GPT: 2 Ans: 0

Ours: downwards

GPT: downwards

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

Ours: 18.54 cm GPT: 16.71 cm

Ours: 18.54 cm GPT: 16.71 cm

Q: From 0.00 sec to 9.50 sec, what is the most likely displacement (straight-line distance) of the camera or object between two frames for ⟨R1⟩?

Q: From 0.00 sec to 9.50 sec, what is the most likely displacement (straight-line distance) of the camera or object between two frames for ⟨R1⟩?

|[Figure 574]<br><br>⟨R1⟨R1⟩⟩<br><br>[Figure 575]<br><br>|[Figure 576]<br><br>(*)(*)<br><br>[Figure 577]<br><br>|[Figure 578]<br><br>(*)(*)<br><br>[Figure 579]<br><br>|
|---|---|---|

|<br><br><br><br>[Figure 582]<br><br>⟨R1⟩ ⟨R1⟩ (*) (*)<br><br><br><br><br><br>|(*) (*)<br><br>[Figure 585]<br><br>|
|---|---|

|(*)<br><br>[Figure 586]<br><br>|(*)|
|---|
<br><br>(|[Figure 587]|
|---|---|

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

- ⟨R1⟩
- ⟨R2⟩

- ⟨R1⟩
- ⟨R2⟩

[Figure 595]

Ours: 7.53 m GPT: 10.06 m Ans: 8.50 m

Ours: 7.53 m GPT: 10.06 m Ans: 8.50 m

(*) (*)

(*) (*)

(*)

(*)

Q: At 6.00 sec, What is the positional relationship of ⟨R1⟩ relative to ⟨R2⟩ from the observer's perspective?

Q: At 27.00 sec, given a single frame, determine the 3D

Q: At 6.00 sec, What is the positional relationship of ⟨R1⟩ relative to ⟨R2⟩ from the observer's perspective?

Q: Are ⟨R1⟩ turning clockwise or counter-clockwise?

Q: At 27.00 sec, given a single frame, determine the 3D

Q: Are ⟨R1⟩ turning clockwise or counter-clockwise?

bounding box of ⟨R1⟩. Identify the correct dimensions, central point, and orientation including yaw, pitch, and roll. (unit: cm, °)

bounding box of ⟨R1⟩. Identify the correct dimensions, central point, and orientation including yaw, pitch, and roll. (unit: cm, °)

Ours: clockwise

GPT: counter-clockwise

[Figure 602]

[Figure 603]

Ours: clockwise

GPT: counter-clockwise

Ours: left GPT: left

[Figure 604]

[Figure 605]

Ours: left GPT: left

|<br><br>⟨R1⟩<br><br>(*)<br><br><br><br>⟨R1⟩<br><br>(*)|<br><br>[Figure 609]<br><br>(*)<br><br><br><br>[Figure 611]<br><br>(*)|
|---|---|

|[Figure 612]|[Figure 613]|
|---|---|

| |
|---|
|{<br><br>dimensions: [25.62, 2.38, 15.33], central_point: [12.91, 2.77, 90.59], orientation: {<br><br>yaw: 117.10, pitch: 42.61, roll: 114.41<br><br>}<br><br>{<br><br>dimensions: [25.62, 2.38, 15.33], central_point: [12.91, 2.77, 90.59], orientation: {<br><br>yaw: 117.10, pitch: 42.61, roll: 114.41<br><br>} }|
|}|

[Figure 614]

[Figure 615]

[Figure 616]

Ours & GPT:

[Figure 617]

[Figure 618]

[Figure 619]

Ours & GPT:

⟨R2⟩

⟨R2⟩

(*) (*)

⟨R1⟩

(*)

(*) (*)

⟨R1⟩

(*)

Q: Is ⟨R1⟩ turning clockwise or counter-clockwise?

Q: At 3.00 sec, What is the positional relationship of the ⟨R2⟩ relative to ⟨R1⟩?

Q: Is ⟨R1⟩ turning clockwise or counter-clockwise?

Q: At 3.00 sec, What is the positional relationship of the ⟨R2⟩ relative to ⟨R1⟩?

Ours: counter-clockwise

GPT: clockwise

Ours: counter-clockwise

GPT: clockwise

[Figure 624]

[Figure 625]

Ours: left GPT: left

[Figure 626]

[Figure 627]

Ours: left GPT: left

|<br><br>R1⟩<br><br><br><br>⟨R1⟩|<br><br><br><br>|[Figure 632]<br><br>|[Figure 633]|
|---|---|---|---|
|Q: How many ⟨R1⟩ are standing still?| | | |

|[Figure 634]<br><br>⟨R1⟩<br><br>[Figure 635]<br><br>⟨R1⟩|[Figure 636]<br><br>[Figure 637]|[Figure 638]<br><br>[Figure 639]|
|---|---|---|
| | | |

|[Figure 640]<br><br>[Figure 641]<br><br>⟨R1⟩|[Figure 642]<br><br>(*)<br><br>[Figure 643]<br><br>(*)|[Figure 644]<br><br>[Figure 645]<br><br>(*)| |
|---|---|---|---|

⟨

⟨R1⟩ (*)

Q: At 18.52 sec, what is the 3D bounding box in camera

Q: At 6.00 sec, what is the most appropriate average or instantaneous speed of ⟨R1⟩?

Q: How many ⟨R1⟩ are standing still?

coordinates of the ⟨R1⟩ from a single randomly selected frame? (unit: m, m/s, m/s^2, °)

Q: At 18.52 sec, what is the 3D bounding box in camera

Ours: 5 GPT: 3

Q: At 6.00 sec, what is the most appropriate average or instantaneous speed of ⟨R1⟩?

coordinates of the ⟨R1⟩ from a single randomly selected frame? (unit: m, m/s, m/s^2, °)

Ours: 5

###### GPT: 3

[Figure 650]

[Figure 651]

Ours: 4.60 m/s GPT: 4.60 m/s

[Figure 652]

[Figure 653]

Ours: 4.60 m/s GPT: 4.60 m/s

[Figure 654]

[Figure 655]

Ours: GPT:

[Figure 658]

| | | | |
|---|---|---|---|
|[Figure 659]<br><br>⟨R1⟩<br><br>[Figure 660]<br><br>⟨R1⟩|[Figure 661]<br><br>(*)<br><br>[Figure 662]<br><br>(*)|(*)<br><br>[Figure 663]<br><br>(*)|[Figure 664]|
|At 14.00 sec, wha|t is the most appropriate|average or| |

[Figure 665]

[Figure 666]

|GPT{ :|
|---|
|{<br><br>C_lwh: [0.86, 1.26, 0.74], C_central_point: [3.74, 1.39, 2.81], C_heading:<br><br>C_lwh: [0.86, 1.26, 0.74], C_central_point: [3.74, 1.39, 2.81], C_heading: 27.20<br><br>}|
|27.20 }|

Ours:

[Figure 669]

{

C_lwh: [0.86, 1.26, 0.74], C_central_point: [3.55, 1.33, 2.75], C_heading: 27.51

{

⟨R1⟩ (*) (*)

C_lwh: [0.86, 1.26, 0.74], C_central_point: [3.55, 1.33, 2.75], C_heading: 27.51

⟨R1⟩ (*) (*)

Q: How many times does the ⟨R1⟩ jump towards the camera?

Q: instantaneous speed of ⟨R1⟩?

Q: How many times does the ⟨R1⟩ jump towards the camera?

Ours: 1

GPT: 1

}

Q: At 14.00 sec, what is the most appropriate average or instantaneous speed of ⟨R1⟩?

[Figure 674]

[Figure 675]

Ours: 0.00 m/s GPT: 0.20 m/s

Ours: 1

GPT: 1

}

[Figure 678]

[Figure 679]

Ours: 0.00 m/s GPT: 0.20 m/s

Figure A19 | More VQA comparison between GPT-4o [5] and 4D-RGPT (Ours) on R4DBench. We provide 2 examples for each of the following categories: Spatial Relation, Dimension Measurement, and Speed & Acceleration.

Figure A18 | More VQA comparison between GPT-4o [5] and 4D-RGPT (Ours) on R4DBench. We provide 2 examples for each of the following categories: False Positive and 3D Video Grounding.

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

Input video At step 100 At step 1000 At step 100 At step 1000 At step 100 At step 1000

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

Figure A20 | More visualizations of 4D-RGPT explicit signals 𝑃^𝑚. Similar to the format of Fig. 5, we visualize the training progress of 𝑃^depth, 𝑃^flow, and 𝑃^motion.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] OpenAI. Gpt-5. https://openai.com/chatgpt,

2025. Large language model.

- [3] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.
- [4] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- [5] OpenAI. GPT-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [6] Wufei Ma, Luoxin Ye, Celso M de Melo, Alan Yuille, and Jieneng Chen. Spatialllm: A compound 3dinformed design towards spatially-intelligent large multimodal models. In CVPR, 2025.
- [7] Jiahui Zhang, Yurui Chen, Yanpeng Zhou, Yueming Xu, Ze Huang, Jilin Mei, Junhui Chen, Yu-Jie Yuan, Xinyue Cai, Guowei Huang, et al. From flatland to space: Teaching vision-language models to perceive and reason in 3d. In NeurIPS, 2025.
- [8] Dohwan Ko, Sihyeon Kim, Yumin Suh, Minseo Yoon, Manmohan Chandraker, Hyunwoo J Kim, et al. ST-VLM: Kinematic instruction tuning for spatiotemporal reasoning in vision-language models. arXiv preprint arXiv:2503.19355, 2025.
- [9] Runsen Xu, Weiyao Wang, Hao Tang, Xingyu Chen, Xiaodong Wang, Fu-Jen Chu, Dahua Lin, Matt Feiszli, and Kevin J Liang. Multi-SpatialMLLM: Multiframe spatial understanding with multi-modal large language models. arXiv preprint arXiv:2505.17015, 2025.
- [10] Junfei Wu, Jian Guan, Kaituo Feng, Qiang Liu, Shu Wu, Liang Wang, Wei Wu, and Tieniu Tan. Reinforcing spatial reasoning in vision-language models with interwoven thinking and visual drawing. In NeurIPS, 2025.
- [11] Yifan Shen, Yuanzhe Liu, Jingyuan Zhu, Xu Cao, Xiaofeng Zhang, Yixiao He, Wenming Ye, James Matthew Rehg, and Ismini Lourentzou. Fine-grained preference optimization improves spatial reasoning in vlms. In NeurIPS, 2025.
- [12] Wufei Ma, Yu-Cheng Chou, Qihao Liu, Xingrui Wang, Celso M de Melo, Jianwen Xie, and Alan Yuille. SpatialReasoner: Towards explicit and generalizable 3d spatial reasoning. In NeurIPS, 2025.
- [13] Kun Ouyang, Yuanxin Liu, Haoning Wu, Yi Liu, Hao Zhou, Jie Zhou, Fandong Meng, and Xu Sun. Spacer: Reinforcing mllms in video spatial reasoning. arXiv preprint arXiv:2504.01805, 2025.

- [14] Hongxing Li, Dingming Li, Zixuan Wang, Yuchen Yan, Hang Wu, Wenqi Zhang, Yongliang Shen, Weiming Lu, Jun Xiao, and Yueting Zhuang. SpatialLadder: Progressive training for spatial reasoning in vision-language models. arXiv preprint arXiv:2510.08531, 2025.
- [15] Diankun Wu, Fangfu Liu, Yi-Hsin Hung, and Yueqi Duan. Spatial-mllm: Boosting mllm capabilities in visual-based spatial intelligence. In NeurIPS, 2025.
- [16] Pingyi Chen, Yujing Lou, Shen Cao, Jinhui Guo, Lubin Fan, Yue Wu, Lin Yang, Lizhuang Ma, and Jieping Ye. SD-VLM: Spatial measuring and understanding with depth-encoded vision-language models. In NeurIPS, 2025.
- [17] Duo Zheng, Shijia Huang, Yanyang Li, and Liwei Wang. Learning from videos for 3d world: Enhancing mllms with 3d vision geometry priors. In NeurIPS, 2025.
- [18] Zhiwen Fan, Jian Zhang, Renjie Li, Junge Zhang, Runjin Chen, Hezhen Hu, Kevin Wang, Huaizhi Qu, Dilin Wang, Zhicheng Yan, et al. VLM-3R: Visionlanguage models augmented with instruction-aligned 3d reconstruction. arXiv preprint arXiv:2505.20279, 2025.
- [19] Hanyu Zhou and Gim Hee Lee. LLaVA-4D: Embedding spatiotemporal prompt into lmms for 4d scene understanding. arXiv preprint arXiv:2505.12253, 2025.
- [20] An-Chieh Cheng, Yang Fu, Yukang Chen, Zhijian Liu, Xiaolong Li, Subhashree Radhakrishnan, Song Han, Yao Lu, Jan Kautz, Pavlo Molchanov, et al. 3d aware region prompted vision language model. arXiv preprint arXiv:2509.13317, 2025.
- [21] Yiming Chen, Zekun Qi, Wenyao Zhang, Xin Jin, Li Zhang, and Peidong Liu. Reasoning in space via grounding in the world. arXiv preprint arXiv:2510.13800, 2025.
- [22] Yun Li, Yiming Zhang, Tao Lin, XiangRui Liu, Wenxiao Cai, Zheng Liu, and Bo Zhao. STI-Bench: Are MLLMs ready for precise spatial-temporal world understanding? In ICCV, 2025.
- [23] Shijie Zhou, Alexander Vilesov, Xuehai He, Ziyu Wan, Shuwang Zhang, Aditya Nagachandra, Di Chang, Dongdong Chen, Eric Xin Wang, and Achuta Kadambi. VLM4D: Towards spatiotemporal awareness in vision language models. In ICCV, 2025.
- [24] Arijit Ray, Jiafei Duan, Reuben Tan, Dina Bashkirova, Rose Hendrix, Kiana Ehsani, Aniruddha Kembhavi, Bryan A Plummer, Ranjay Krishna, Kuo-Hao Zeng, et al. SAT: Spatial aptitude training for multimodal language models. In COLM, 2025.
- [25] Mengdi Jia, Zekun Qi, Shaochen Zhang, Wenyao Zhang, Xinqiang Yu, Jiawei He, He Wang, and Li Yi. OmniSpatial: Towards comprehensive spatial reasoning benchmark for vision language models. arXiv preprint arXiv:2506.03135, 2025.

- [26] Sihan Yang, Runsen Xu, Yiman Xie, Sizhe Yang, Mo Li, Jingli Lin, Chenming Zhu, Xiaochen Chen, Haodong Duan, Xiangyu Yue, et al. MMSI-Bench: A benchmark for multi-image spatial intelligence. arXiv preprint arXiv:2505.23764, 2025.
- [27] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [28] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and finetuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [29] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.
- [30] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. VILA: On pretraining for visual language models. In CVPR, 2024.
- [31] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.
- [32] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- [33] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023.
- [34] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR, 2024.
- [35] Alibaba Group Qwen Team. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [36] Zhijian Liu, Ligeng Zhu, Baifeng Shi, Zhuoyang Zhang, Yuming Lou, Shang Yang, Haocheng Xi, Shiyi Cao, Yuxian Gu, Dacheng Li, et al. NVILA: Efficient frontier visual language models. In CVPR, 2025.
- [37] Honglu Zhou, Xiangyu Peng, Shrikant Kendre, Michael S Ryoo, Silvio Savarese, Caiming Xiong, and Juan Carlos Niebles. Strefer: Empowering video llms with space-time referring and reasoning via synthetic instruction data. In ICCV, 2025.
- [38] Zikang Liu, Longteng Guo, Yepeng Tang, Tongtian Yue, Junxian Cai, Kai Ma, Qingbin Liu, Xi Chen, and Jing Liu. Vrope: Rotary position embedding for video large language models. In EMNLP, 2025.

- [39] Yumeng Shi, Quanyu Long, Yin Wu, and Wenya Wang. Causality matters: How temporal information emerges in video language models. arXiv preprint arXiv:2508.11576, 2025.
- [40] Xiangyu Zeng, Kunchang Li, Chenting Wang, Xinhao Li, Tianxiang Jiang, Ziang Yan, Songze Li, Yansong Shi, Zhengrong Yue, Yi Wang, et al. Timesuite: Improving mllms for long video understanding via grounded tuning. In ICLR, 2025.
- [41] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. Timechat: A time-sensitive multimodal large language model for long video understanding. In CVPR, 2024.
- [42] Jinghui Lu, Haiyang Yu, Yanjie Wang, Yongjie Ye, Jingqun Tang, Ziwei Yang, Binghong Wu, Qi Liu, Hao Feng, Han Wang, et al. A bounding box is worth one token-interleaving layout and text in a large language model for document understanding. In ACL Findings, 2025.
- [43] Liang Zhao, En Yu, Zheng Ge, Jinrong Yang, Haoran Wei, Hongyu Zhou, Jianjian Sun, Yuang Peng, Runpei Dong, Chunrui Han, and Xiangyu Zhang. ChatSpot: Bootstrapping multimodal llms via precise referring instruction tuning. In IJCAI, 2024.
- [44] Shraman Pramanick, Guangxing Han, Rui Hou, Sayan Nag, Ser-Nam Lim, Nicolas Ballas, Qifan Wang, Rama Chellappa, and Amjad Almahairi. Jack of all tasks master of many: Designing generalpurpose coarse-to-fine vision-language model. In CVPR, 2024.
- [45] Yunjie Tian, Tianren Ma, Lingxi Xie, Jihao Qiu, Xi Tang, Yuan Zhang, Jianbin Jiao, Qi Tian, and Qixiang Ye. ChatterBox: Multi-round multimodal referring and grounding. In AAAI, 2025.
- [46] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023.
- [47] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, Qixiang Ye, and Furu Wei. Grounding multimodal large language models to the world. In ICLR, 2024.
- [48] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. MiniGPT-4: Enhancing vision-language understanding with advanced large language models. In ICLR, 2024.
- [49] Weiyun Wang, Yiming Ren, Haowen Luo, Tiantong Li, Chenxiang Yan, Zhe Chen, Wenhai Wang, Qingyun Li, Lewei Lu, Xizhou Zhu, et al. The AllSeeing project v2: Towards general relation comprehension of the open world. In ECCV, 2024.
- [50] Gongwei Chen, Leyang Shen, Rui Shao, Xiang Deng, and Liqiang Nie. LION: Empowering multimodal large language model with dual-level visual knowledge. In CVPR, 2024.
- [51] Byung-Kwan Lee, Beomchan Park, Chae Won Kim, and Yong Man Ro. CoLLaVO: Crayon large language and vision mOdel. In ACL, 2024.

- [52] Yunze Man, De-An Huang, Guilin Liu, Shiwei Sheng, Shilong Liu, Liang-Yan Gui, Jan Kautz, Yu-Xiong Wang, and Zhiding Yu. ARGUS: Vision-centric reasoning with grounded chain-of-thought. In CVPR, 2025.
- [53] Weifeng Lin, Xinyu Wei, Ruichuan An, Peng Gao, Bocheng Zou, Yulin Luo, Siyuan Huang, Shanghang Zhang, and Hongsheng Li. Draw-and-understand: Leveraging visual prompts to enable mllms to comprehend what you want. In ICLR, 2025.
- [54] Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Yu Liu, Kai Chen, and Ping Luo. GPT4RoI: Instruction tuning large language model on region-of-interest. In ECCV Workshop, 2024.
- [55] Chuofan Ma, Yi Jiang, Jiannan Wu, Zehuan Yuan, and Xiaojuan Qi. Groma: Localized visual tokenization for grounding multimodal large language models. In ECCV, 2024.
- [56] Weiyun Wang, Min Shi, Qingyun Li, Wenhai Wang, Zhenhang Huang, Linjie Xing, Zhe Chen, Hao Li, Xizhou Zhu, Zhiguo Cao, et al. The all-seeing project: Towards panoptic visual recognition and understanding of the open world. In ICLR, 2024.
- [57] Sangmin Woo, Kang Zhou, Yun Zhou, Shuai Wang, Sheng Guan, Haibo Ding, and Lin Lee Cheong. Blackbox visual prompt engineering for mitigating object hallucination in large vision language models. In NAACL, 2025.
- [58] Mu Cai, Haotian Liu, Siva Karthik Mustikovela, Gregory P Meyer, Yuning Chai, Dennis Park, and Yong Jae Lee. ViP-LLaVA: Making large multimodal models understand arbitrary visual prompts. In CVPR, 2024.
- [59] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt4v. arXiv preprint arXiv:2310.11441, 2023.
- [60] Xuanyu Lei, Zonghan Yang, Xinrui Chen, Peng Li, and Yang Liu. Scaffolding coordinates to promote vision-language coordination in large multi-modal models. In ACL, 2025.
- [61] Miran Heo, Min-Hung Chen, De-An Huang, Sifei Liu, Subhashree Radhakrishnan, Seon Joo Kim, YuChiang Frank Wang, and Ryo Hachiuma. Omni-rgpt: Unifying image and video region-level understanding via token marks. In CVPR, 2025.
- [62] Peiwen Sun, Shiqiang Lang, Dongming Wu, Yi Ding, Kaituo Feng, Huadai Liu, Zhen Ye, Rui Liu, Yun-Hui Liu, Jianan Wang, et al. Spacevista: All-scale visual spatial reasoning from mm to km. arXiv preprint arXiv:2510.09606, 2025.
- [63] Pengteng Li, Pinhao Song, Wuyang Li, Weiyu Guo, Huizai Yao, Yijie Xu, Dugang Liu, and Hui Xiong. See&trek: Training-free spatial prompting for multimodal large language model. In NeurIPS, 2025.

- [64] Xiaohu Huang, Jingjing Wu, Qunyi Xie, and Kai Han. Mllms need 3d-aware representation supervision for scene understanding. In NeurIPS, 2025.
- [65] Gorjan Radevski, Dusan Grujicic, Matthew Blaschko, Marie-Francine Moens, and Tinne Tuytelaars. Multimodal distillation for egocentric action recognition. In ICCV, 2023.
- [66] Rui Wang, Dongdong Chen, Zuxuan Wu, Yinpeng Chen, Xiyang Dai, Mengchen Liu, Lu Yuan, and YuGang Jiang. Masked video distillation: Rethinking masked feature modeling for self-supervised video representation learning. In CVPR, 2023.
- [67] Abhishek Badki, Hang Su, Bowen Wen, and Orazio Gallo. L4P: Low-level 4D vision perception unified. arXiv preprint arXiv:2502.13078, 2025.
- [68] Weifeng Lu, Minghao Ye, Zewei Ye, Ruihan Tao, Shuo Yang, and Bo Zhao. RoboFAC: A comprehensive framework for robotic failure analysis and correction. arXiv preprint arXiv:2505.12224, 2025.
- [69] Boyi Li, Ligeng Zhu, Ran Tian, Shuhan Tan, Yuxiao Chen, Yao Lu, Yin Cui, Sushant Veer, Max Ehrlich, Jonah Philion, et al. Wolf: Dense video captioning with a world summarization framework. TMLR, 2025.
- [70] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In ECCV, 2024.
- [71] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos. In ICLR, 2025.
- [72] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbeláez, Alexander Sorkine-Hornung, and Luc Van Gool. The 2017 DAVIS challenge on video object segmentation. arXiv:1704.00675, 2017.
- [73] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.
- [74] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, et al. Videollama 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106, 2025.
- [75] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024.

- [76] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. TMLR, 2025.
- [77] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llavanext: Improved reasoning, ocr, and world knowledge, 2024.
- [78] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In ICCV, 2023.
- [79] Qwen Team et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.
- [80] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. LoRA: Low-rank adaptation of large language models. In ICLR, 2022.
- [81] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al. Huggingface’s transformers: State-ofthe-art natural language processing. arXiv preprint arXiv:1910.03771, 2019.
- [82] Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, and Yu Qiao. VideoMAE V2: Scaling video masked autoencoders with dual masking. In CVPR, 2023.
- [83] René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In ICCV, 2021.
- [84] D Hendrycks. Gaussian error linear units (GELUs). arXiv preprint arXiv:1606.08415, 2016.
- [85] Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In AISTATS, 2010.
- [86] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In CVPR, 2025.
- [87] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. ScanNet: Richly-annotated 3d reconstructions of indoor scenes. In CVPR, 2017.
- [88] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. ScanNet++: A highfidelity dataset of 3d indoor scenes. In ICCV, 2023.
- [89] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuScenes: A multimodal dataset for autonomous driving. In CVPR, 2020.
- [90] Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. A benchmark dataset and evaluation methodology for video object segmentation. In CVPR, 2016.

[91] Maxim Tkachenko, Mikhail Malyuk, Andrey Holmanyuk, and Nikolai Liubimov. Label Studio: Data labeling software, 20202025. Open source software available from https://github.com/HumanSignal/label-studio.

