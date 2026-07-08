# arXiv:2512.22238v1[cs.LG]23Dec2025

[Figure 1]

2025-12-30

## Masking Teacher and Reinforcing Student for Distilling Vision-Language Models

Byung-Kwan Lee, Yu-Chiang Frank Wang, Ryo Hachiuma

##### Abstract

Large-scale vision–language models (VLMs) have recently achieved remarkable multimodal understanding, but their massive size makes them impractical for deployment on mobile or edge devices. This raises the need for compact yet capable VLMs that can efficiently learn from powerful large teacher. However, distilling knowledge from large teacher to small student remains challenging due to their large size gap: the student often fails to reproduce the teacher’s complex, high-dimensional representations, leading to unstable learning and degraded performance. To address this, we propose Masters (Masking teacher and reinforcing student), a mask-progressive reinforcement learning (RL) distillation framework. Masters first masks and non-dominant weights of the teacher to reduce unnecessary complexity, then progressively restores the teacher from mask to gradually increase the teacher capacity during training. This strategy allows the student to learn richer representations of teacher in a smooth and stable manner. To further refine knowledge transfer, Masters integrates an offline RL stage with two complementary rewards: an accuracy reward that measures the correctness of the generated responses, and a distillation reward that quantifies the ease of their responses’ transferability from teacher to student. Unlike online think–answer RL paradigms that are computationally expensive and generate lengthy responses, our offline RL leverages pre-generated responses from masked teachers. These provide rich yet efficient guidance, enabling the students to achieve strong performance without requiring the think–answer process. Extensive experiments across diverse VLM benchmarks demonstrate that Masters outperforms existing compact VLMs and partially surpasses large ones, while being far more efficient. Moreover, gradually increasing the teacher sizes during distillation (e.g., from 14B to 38B) yields smoother convergence and stronger generalization than one-shot distillation (e.g., 38B), revealing a scalable path toward efficient and deployable VLMs.

[Figure 2]

8B 7B

###### Masters

8B

32B

8B

4B

###### Qwen3-VL 78B

AveragePerformance(%)

38B

8B

InternVL3.5

2B 4B

8B

Unknown size

###### 3B

InternVL3

4B

4B

GPT Claude

2B

Qwen2.5-VL

8B

2B

2B

Gemini

2B

- 2B

72B

7B

- 3B

Model size (B)

Figure 1 | Comparing Masters-applied VLMs with diverse open- and closed-source VLMs across broad model sizes for averaged performance (%) of numerous evaluation benchmarks: AI2D [1], ChartQA [2], MathVista [3], MMB [4], MM-Vet [5], MMMU [6], MMMU-Pro [7], MMStar [8], BLINK [9], SEED-Bench [10], SEED-Bench-2-Plus [11], and RealWorldQA.

### 1. Introduction

In recent years, vision–language models (VLMs) [13, 14] have demonstrated remarkable capabilities across a wide range of multimodal

tasks [15], including visual captioning, reasoning, and open-ended question answering. By jointly understanding visual and textual information, large-scale VLMs have achieved impressive generalization and reasoning abilities

Correspondence to <byungkwanl@nvidia.com> © 2025 NVIDIA. All rights reserved.

Masking Ratio Average Performance (%) Average Performance (%)

[Figure 3]

[Figure 4]

[Figure 5]

InternVL3.5-14B

InternVL3.5-38B

InternVL3.5-38B

Training Iteration Training Iteration Distillation Method

(a) Masking Ratio Schedule

(b) Performance Trend (c) Distillation w/ and w/o RL

- Figure 2 | Illustrating training dynamics of Masters, where we represent how (a) mask ratio is controlled during distillation, and (b) its averaged performance log of student (InternVL3.5-8B [12]) for evaluation benchmarks in Tab. 1. In addition, we show (c) the effect of RL under naive and mask-progressive distillation. Note that asterisk (*) represents the combined distillation of mid-size and large teacher.

that in some domains approach human-level understanding [16]. However, these achievements come at the cost of massive model sizes [17, 18, 19, 12] and heavy computational requirements, making them impractical for deployment on mobile or edge devices [20]. As the demand for on-device intelligence continues to grow, there is an urgent need for compact yet powerful VLMs [21, 22, 23, 24] that can deliver competitive performance while maintaining high efficiency and deployability [25, 26].

A widely adopted approach for building such lightweight yet capable models is knowledge distillation [27, 28], where a large teacher transfers its knowledge to a smaller student. Despite its promise, distillation remains challenging due to the substantial size gap between teacher and student [29, 30, 31]. The student often struggles to reproduce the teacher’s rich and high-dimensional representations, leading to unstable learning and significant performance degradation [32, 33, 34]. Recent distillation has explored modified training objectives [35, 36, 37, 38], dynamic intermediatelayer distillation [39, 40], vision-attention distillation [41, 42], multi-step distillation [43, 44], crosstoken general distillation [45, 46], and reinforcement learning (RL)–based approaches [47, 48, 49]. However, few works have directly addressed the fundamental issue of the large size gap itself.

To tackle this challenge, we first mask nondominant weights of the teacher based on their magnitudes [50], thereby reducing unnecessary complexity. This is because the large number of weights in the teacher is a core factor that makes distillation difficult. After masking the weights in the teacher, we perform the knowledge distillation progressively where the teacher is gradually restored from the mask throughout the entire training to increase the teacher’s representation complexity. This mask-progressive strategy enables the student to first capture coarse-grained patterns and subsequently refine them into more detailed and higher-level representations, thereby learning richer and more complex representations of the teacher in a smooth and stable manner.

Furthermore, we identify critical limitations in existing supervised fine-tuning (SFT) datasets [51, 52, 53], where answer labels are typically generated by large closed-source models with an extremely large number of parameters, such as GPT-4o [14], Gemini [54], or Claude [55], and often filtered by humans [56]. Small student, however, has limited capacity to learn these rich answer labels due to its smaller vocabulary and lower hidden dimension of representation [57, 58]. To address this, we utilize the generated responses from masked teachers instead of directly using standard SFT samples. This enables the

[Figure 6]

###### Masking Ratio: 0.20 Masking Ratio: 0.15 Masking Ratio: 0.10 Masking Ratio: 0.05 Masking Ratio: 0

MaskedTeacher

|…<br><br>|
|---|

|…<br><br>|
|---|

|…<br><br>|
|---|

|…<br><br>|
|---|

|…<br><br>|
|---|

Compute

|𝑅distill|
|---|

|𝑅distill|
|---|

|𝑅distill|
|---|

|𝑅distill|
|---|

|𝑅distill|
|---|

|𝑅acc|
|---|

|𝑅acc|
|---|

|𝑅acc|
|---|

|𝑅acc|
|---|

|𝑅acc|
|---|

Compute

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Update

|[Figure 12]<br><br>[Figure 13]<br><br>Student|
|---|

|[Figure 14]<br><br>[Figure 15]<br><br>Student|
|---|

|[Figure 16]<br><br>[Figure 17]<br><br>Student|
|---|

|[Figure 18]<br><br>[Figure 19]<br><br>Student|
|---|

|[Figure 20]<br><br>[Figure 21]<br><br>Student|
|---|

- Figure 3 | Overview of mask-progressive distillation where teacher is masked with a decreasing masking ratio (0.20, 0.15, 0.10, 0.05, 0), gradually restoring its full capacity. At each masking stage for teacher, student is updated using two rewards: accuracy reward 𝑅acc and distillation reward 𝑅distill. This progressive distillation enables smooth and stable knowledge transfer to the student.

student to learn the responses that better match the student’s current capacity. Moreover, we incorporate the student’s own generated responses into training to maintain alignment between the teacher’s guidance and the student’s evolving representational capacity. This design ensures stable yet continual improvement toward the teacher’s behavior, going beyond the single-answer and over-rich label constraints of conventional SFT datasets.

However, some generated responses may contain factual errors or exhibit linguistic complexity that hinders effective knowledge transfer, ultimately degrading the distillation performance. Hence, we aim to evaluate both the accuracy of the responses and their ease of knowledge transfer, and further refine the student to avoid such undesirable responses. To achieve this, we integrate RL into the distillation process. Specifically, we compute two types of rewards: an accuracy reward evaluated by LLM-as-a-Judge [59] to account for flexible and diverse answers, and a distillation reward that measures how easily knowledge can be transferred from the teacher to the student. In practice, conventional online RL is too slow and inefficient training, since it requires the model to repeatedly generate multiple responses at every training step under the recent “think–answer” paradigm [60, 61]. As a result, only a very limited amount of data samples [62, 63, 64, 65, 66, 67, 68, 49] are utilized, constraining the scale and diversity of training samples. To address these limitations, we adopt

an offline RL approach in which both the teacher and the student pre-generate their multiple responses for all questions, without explicit thinkanswer. These pre-generated responses are then used for RL training, significantly reducing training time and computational costs.

Bringing these components together, we propose a mask-progressive RL distillation framework, referred to as Masking teacher and reinforcing students (Masters). It integrates teacher weight masking, progressive distillation, multi-response learning, and offline RL into a unified training paradigm. Through this design, it enables the student to effectively absorb knowledge from large teachers while maintaining high efficiency and stability. Through extensive experiments across diverse evaluation benchmarks, we demonstrate that Masters consistently outperforms existing compact VLMs and exceeds large ones in Fig. 1. Beyond empirical gains, we find that gradually increasing the teacher sizes during distillation (e.g., from 14B to 38B) leads to smoother convergence and superior generalization compared to one-shot distillation from a single large teacher (e.g., 38B), revealing an effective pathway for scalable model compression. We believe this framework marks an important step toward efficient, deployable, and continually improvable VLMs for on-device intelligence.

Our main contributions can be summarized in threefold as follows:

#### • Progressive capacity-aligned dis-

[Figure 22]

|𝑅acc|
|---|

|𝑅distill|
|---|

Question Input Utilizing Both Models Generated Multiple Responses

|[Figure 23]<br><br>𝑦ො1|[Figure 24]<br><br>|
|---|---|
| | |

[Figure 25]

[Figure 26]

- 1

[Figure 27]

[Figure 28]

- 2

[Figure 29]

[Figure 30]

- 3

[Figure 31]

[Figure 32]

- 5

[Figure 33]

[Figure 34]

4

[Figure 35]

[Figure 36]

- 6

[Figure 37]

“This image shows a dog wearing a leather vest and goggles while riding a motorcycle.”

rd

|[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>Masked<br><br>Teacher|
|---|

|[Figure 41]<br><br>[Figure 42]<br><br>𝑦ො2| |
|---|---|
| | |

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

“The dog is sitting on the Harley-style motorcycle.”

st

[Figure 49]

[Figure 50]

[Figure 51]

|[Figure 52]<br><br>[Figure 53]<br><br>𝑦ො3| |
|---|---|
| | |

[Figure 54]

[Figure 55]

[Figure 56]

th

“The composition humorously depicts a canine pilot controlling an aircraft.”

|[Figure 57]<br><br>[Figure 58]<br><br>𝑦ො4| |
|---|---|
| | |

[Figure 59]

[Figure 60]

[Figure 61]

nd

“The dog is riding a motorcycle.”

|[Figure 62]<br><br>[Figure 63]<br><br>Student|
|---|

[Figure 64]

|[Figure 65]<br><br>[Figure 66]<br><br>𝑦ො5| |
|---|---|
| | |

[Figure 67]

[Figure 68]

[Figure 69]

th

“This photo shows a dog wearing a leather vest and goggles.”

[Figure 70]

Question: What type of vehicle is the dog sitting on in the image?

|[Figure 71]<br><br>𝑦ො6|[Figure 72]<br><br>|
|---|---|
| | |

[Figure 73]

[Figure 74]

“A bulldog dressed in black leather appears on the motorcycle.”

th

- Figure 4 | Depicting multiple responses generated by both the masked teacher and the student, where an accuracy reward (𝑅acc) evaluates the binary correctness of each response, and a distillation reward (𝑅distill) measures the ease of knowledge transfer based on divergence objective between teacher and student logits. Note that the rank labels (1st, 2nd, 3rd, etc.) in 𝑅distill indicate the relative magnitude of the divergence values, where the smallest divergence (1st) receives the highest reward (1.0) and the largest divergence the lowest reward (0.0).

tillation: We propose a progressive teacher–student alignment strategy that adaptively matches the student’s learning capacity through restoring the teacher from the mask.

- • Offline RL with dual rewards: By pregenerating diverse responses from both the teacher and the student, our offline RL substantially reduces training time while efficiently achieving strong performance with two complementary objectives — an accuracy reward for correctness and a distillation reward for transferability.
- • Unified and scalable framework: We integrate progressive capacity-aligned distillation and offline RL into a unified training framework with scaled-up data, enabling compact VLMs to attain large model-level performance for practical efficiency and deployability.

### 2. Related Work

Previous approaches to building efficient VLMs (see Appendix A) have focused on inserting additional modules, modifying internal architectures, or altering propagation strategies within the models themselves. Recent emerging line of research on efficient VLMs has begun to leverage knowledge distillation [27, 28], where the knowledge

of a large, high-capacity teacher is transferred into a smaller, more efficient student. This is because training a small model alone inherently limits its representational capacity, prompting a shift toward utilizing rich logit or intermediate representations from the teacher.

Beyond the scope of VLM research, early studies on knowledge distillation primarily focused on aligning the logits between the teacher and student [69, 70]. Subsequent works extended this paradigm to intermediate feature representations [71, 72, 73, 74, 75], and later introduced probabilistic [76], relational [77], and contrastive formulations [78] to better capture internal dependencies [79, 80, 81, 82]. Several studies have explored multi-teacher frameworks [83, 84, 85, 86, 87, 88] to integrate diverse knowledge sources, while others have leveraged step-by-step reasoning traces from large models to enhance the student’s reasoning and compositional capabilities [89, 90]. More recent research has investigated alternative training objectives [35, 36, 37, 38], dynamic intermediatelayer distillation [39, 40], vision-attention distillation [41, 42], multi-step distillation [43, 44], cross-token general distillation [45, 46], and reinforcement learning (RL)–based approaches [47, 48, 49].

Despite these advances, model distillation re-

- Table 1 | Comparing the performance between naive, mask-progressive, and RL-applied distillation on AI2D [1], ChartQA [2], MathVista [3], MMB/MMBCN [4], MM-Vet [5], MMMU [6], MMMU-Pro [7], MMStar [8], BLINK [9], SEED [10], SEED2+ [11], RealWorldQA (RWQA). Note that ‘+Large Teacher’ refers to the largest model within the same model family in Sec. 4.1.

VLMs AI2D ChartQA MathVista MMB MMBCN MM-Vet MMMU MMMU-Pro MMStar BLINK SEED SEED2+ RWQA Avg Qwen2.5-VL-7B 83.9 87.3 67.8 83.5 83.4 71.8 55.0 38.3 63.9 56.4 77.0 70.4 68.5 69.8

- +Large Teacher 84.3 87.9 68.4 83.9 83.8 72.3 55.6 38.9 64.5 56.9 77.4 70.9 68.9 70.3

- +Mask-Progressive 85.1 89.2 70.3 84.8 84.6 74.1 57.0 40.2 66.2 58.5 78.5 71.9 70.3 71.6

+Reward Feedback 86.3 92.8 73.4 86.7 86.5 77.0 59.8 42.9 69.0 61.3 80.2 73.8 72.4 74.0 Qwen3-VL-8B 85.7 88.4 77.2 86.8 86.5 74.5 69.6 55.9 70.9 69.1 78.5 70.8 69.9 75.7 +Large Teacher 86.4 90.3 78.4 87.5 87.4 75.8 70.4 57.3 73.1 69.9 79.3 71.9 71.8 76.9 +Mask-Progressive 87.3 92.7 79.8 88.4 88.5 77.3 71.5 59.1 76.0 70.9 80.3 73.3 74.3 78.4 +Reward Feedback 88.5 95.9 81.8 89.5 89.9 79.4 72.9 61.4 79.7 72.3 81.7 75.1 77.5 80.4 InternVL3-8B 85.2 86.6 71.6 83.4 82.2 78.5 62.7 41.3 68.2 55.5 77.1 69.7 70.8 71.8 +Large Teacher 85.8 87.4 72.2 84.0 82.8 79.1 63.6 42.1 68.9 56.2 77.7 70.4 71.5 72.4

- +Mask-Progressive 86.5 88.5 73.0 84.8 83.6 79.8 64.7 43.0 70.1 57.4 78.5 71.2 72.8 73.4

+Reward Feedback 88.1 91.9 76.3 87.3 86.1 82.2 67.1 46.5 74.1 61.0 80.5 73.4 75.3 76.1 InternVL3.5-8B 84.0 86.7 78.4 86.5 85.9 83.1 73.4 57.2 69.3 59.5 77.4 70.8 67.5 75.4 +Large Teacher 84.8 87.3 78.9 86.9 86.3 83.5 73.8 57.6 69.8 60.0 77.8 71.2 67.9 75.8 +Mask-Progressive 85.3 87.8 79.4 87.4 86.8 84.0 74.3 58.1 70.3 60.5 78.3 71.7 68.4 76.3 +Reward Feedback 86.1 88.6 80.2 88.2 87.6 84.8 75.1 58.9 71.1 61.3 79.1 72.5 69.2 77.1

mains challenging due to the substantial parameter gap between teacher and student [29, 30, 31]. The student often struggles to reproduce the teacher’s rich, high-dimensional representations, leading to unstable training and noticeable performance degradation [32, 33, 34]. However, only limited efforts have been made to directly narrow the parameter gap between teacher and student.

Unlike existing approaches that primarily focus on modifying distillation objectives, aligning intermediate features, or combining multiple strategies, we aim to directly address the large parameter gap between teacher and student. To this end, we first mask non-dominant weights in the teacher based on their magnitudes [50], effectively reducing the number of active parameters and simplifying the teacher’s representational space. This masking step not only narrows the capacity gap between teacher and student but also filters out noisy or over-parameterized components that hinder stable knowledge transfer. As training proceeds, we progressively restore the teacher from mask, allowing the student to gradually learn complex representations of teacher in a capacity-aligned manner. This leads to smoother optimization and mitigates the instability in direct distillation from large teacher.

### 3. Masters

This section introduces the two pivotal components that form the core of the Masters: how to mask the teacher (Secs. 3.1 and 3.2) and reinforce the student (Sec. 3.3).

#### 3.1. Magnitude-based Teacher Masking

A key challenge in distilling knowledge from a large teacher to a small student lies in the significant parameter gap between them. To reduce this gap, we adopt a magnitude-based masking strategy inspired by classical network pruning [50], where weights with smaller magnitudes are masked to zero. Given a teacher 𝒯 with weight W𝒯 = {𝑤𝑛}𝑁𝑛=1 (𝑤𝑛 ∈ R), we construct a binary mask M𝑟 = {𝑚𝑛}𝑁𝑛=1 (𝑚𝑛 ∈ 0,1) as follows:

𝑚𝑛 = {︃

1, if |𝑤𝑛| ≥ 𝜆𝑟, 0, otherwise,

(1)

where 𝑁 denotes the total number of parameters (e.g., 𝑁 = 38B when using InternVL3.5-38B [12]), and 𝜆𝑟 is a magnitude threshold determined by the desired masking ratio 𝑟 ∈ [0,1]. For example, when 𝑟 = 0, it is the original teacher. Conversely, when 𝑟 = 1, all weights are masked to zero. With 𝑟 = 0.2, the magnitude threshold 𝜆0.2 is determined by sorting the magnitude of W𝒯 in ascending order, such that approximately

- Table 2 | Comparison of the performance among naive (using large and mid teacher), mask-progressive, and RL-applied distillation. Note that ‘+Mid Teacher’ denotes all intermediate models (e.g., 4B, 8B, and 14B) between student (e.g., 2B) and ‘Large Teacher’ (e.g., 38B).

VLMs AI2D ChartQA MathVista MMB MMBCN MM-Vet MMMU MMMU-Pro MMStar BLINK SEED SEED2+ RWQA Avg InternVL3.5-8B 84.0 86.7 78.4 86.5 85.9 83.1 73.4 57.2 69.3 59.5 77.4 70.8 67.5 75.4 +Large Teacher 84.8 87.3 78.9 86.9 86.3 83.5 73.8 57.6 69.8 60.0 77.8 71.2 67.9 75.8 +Mid Teacher 85.4 87.9 79.4 87.3 86.8 84.0 74.3 58.0 70.3 60.5 78.3 71.7 68.4 76.3 +Mask-Progressive 86.0 88.8 80.1 87.9 87.4 84.6 74.9 58.6 71.0 61.2 79.0 72.4 69.1 77.0 +Reward Feedback 87.2 95.1 85.0 88.2 88.3 85.6 72.7 58.1 80.8 67.8 81.4 75.5 74.9 80.0 −Mid Teacher 86.1 88.6 80.2 88.2 87.6 84.8 75.1 58.9 71.1 61.3 79.1 72.5 69.2 77.1 InternVL3.5-4B 82.6 86.0 77.1 86.9 86.1 76.6 66.6 51.4 65.0 58.1 75.2 69.4 66.3 72.9 +Large Teacher 82.8 86.4 77.3 86.9 86.2 76.8 66.9 51.7 65.4 58.4 75.5 69.7 66.6 73.1 +Mid Teacher 82.9 86.6 77.5 87.0 86.3 77.0 67.1 51.9 65.6 58.6 75.7 69.9 66.8 73.3 +Mask-Progressive 83.1 86.9 77.9 87.2 86.5 77.4 67.5 52.2 66.0 59.0 76.1 70.3 67.2 73.6 +Reward Feedback 83.3 94.9 78.8 86.1 86.0 79.5 63.2 52.7 70.5 62.5 79.1 71.4 68.1 75.1 −Mid Teacher 82.9 87.0 77.9 86.8 86.3 77.9 66.9 52.2 65.8 59.0 75.9 70.1 66.9 73.5 InternVL3.5-2B 78.8 80.7 71.8 82.3 81.8 71.7 59.0 46.3 62.7 51.3 75.2 68.0 62.0 68.6 +Large Teacher 79.1 81.8 72.3 82.6 82.2 72.5 59.4 46.8 63.3 52.1 75.5 68.3 62.5 69.1 +Mid Teacher 79.7 83.8 73.0 83.1 82.8 73.8 60.2 47.6 64.4 53.4 76.0 68.8 63.3 70.0 +Mask-Progressive 80.9 87.7 74.6 84.1 84.0 76.5 61.8 49.3 66.6 56.1 77.1 69.8 65.0 71.8 +Reward Feedback 83.0 94.8 77.5 85.9 86.3 81.4 64.6 52.4 70.7 61.1 79.0 71.6 68.1 75.1 −Mid Teacher 80.1 84.3 73.4 83.5 83.2 74.2 60.6 47.9 64.8 53.7 76.3 69.0 63.6 70.4

∑︀𝑁

𝑛=1 𝑚𝑛 ≈ 𝑁 · (1 − 0.2). The resulting masked teacher parameters are defined as:

Evaluation Benchmark Performances (%)

[Figure 75]

W𝒯𝑟 = M𝑟 ⊙ W𝒯 , (2)

where ⊙ denotes element-wise multiplication, and 𝒯𝑟 represents the masked teacher under masking ratio 𝑟. This masking process effectively removes low-magnitude weights that contribute marginally to prediction logits [50], yielding a simplified yet representative teacher. By reducing parameter of the teacher, the student 𝒮 learns a more capacity-aligned representation, mitigating optimization instability. Notably, unlike conventional pruning approaches [91, 92, 93, 94, 95, 96, 97, 98] designed for model compression, our masking is temporary and restored later. In practice, we found that using a global threshold 𝜆𝑟 across all layers often excessively prunes certain layers, which can make the model non-functional at inference time. To prevent this imbalance, we compute 𝜆𝑟 per layer and apply masking separately to each layer so that the overall masking remains balanced and consistent across the teacher network.

Number of Multiple Responses

Figure 5 | Comparing the performances by the number of generated responses from teacher and student.

limits the student’s exposure to the teacher’s rich representation. To solve it, we adopt a maskprogressive strategy that gradually restores the teacher’s full capacity over the entire training. At each training iteration 𝑖 ∈ {1,...,𝐼}, a masking ratio 𝑟[𝑖] is formulated as written:

𝑟[𝑖] = 𝑟max − 𝑠 · ⌊︂𝑖 ×

⌋︂, (3)

𝑀 𝐼

#### 3.2. Mask-Progressive Distillation

where 𝑠 denotes the decrement applied to the masking ratio at each masking stage, and 𝑀 represents the total number of masked teachers with different masking ratios, computed as 𝑀 =

Scheduling Masking Ratio. While the masked teacher provides a simplified representation, learning solely with a fixed masking ratio

- Table 3 | Comparing Masters-applied VLMs with standard or smaller model size open-source VLMs.

VLMs AI2D ChartQA MathVista MMB MMBCN MM-Vet MMMU MMMU-Pro MMStar BLINK SEED SEED2+ RWQA

LLaVA-OneVision-7B [99] 81.4 80.0 63.2 80.8 - 57.5 48.8 24.1 61.9 53.0 76.7 65.4 69.9 LLaVA-OneVision-1.5-8B [53] 84.2 86.5 69.6 84.1 81.0 - 55.4 37.4 67.7 - 77.3 69.2 68.1 MiniCPM-V2.6-8B [100] 82.1 - 60.6 - - 60.0 49.8 27.2 57.5 55.2 74.0 65.7 65.0 MiniCPM-o2.6-8B [100] 86.1 86.9 73.3 - - 67.2 50.9 - 63.3 53.9 75.5 67.9 68.0 Ovis2-8B [101] 86.6 - 71.8 - - 65.1 57.4 - 64.6 54.3 77.2 70.4 72.5 GLM-4.1V-9B [102] 87.9 70.0 80.7 - - 66.4 68.0 - 72.9 65.9 - 71.8 70.6 MiMo-VL-8B [103] 83.5 91.7 81.5 84.4 82.0 77.5 66.7 46.2 70.8 62.4 - 72.4 Keye-VL-8B [104] 86.7 72.5 80.7 92.0 - 68.6 71.4 - 75.5 52.0 - 67.8 67.7 Keye-VL-1.5-8B [105] 89.5 86.3 81.2 92.0 - 71.2 71.4 - 80.5 54.9 - - 73.5 Qwen2.5-VL-7B [17] 83.9 87.3 67.8 83.5 83.4 71.8 55.0 38.3 63.9 56.4 77.0 70.4 68.5 Qwen3-VL-8B [18] 85.7 88.4 77.2 86.8 86.5 74.5 69.6 55.9 70.9 69.1 78.5 70.8 69.9 InternVL3-8B [19] 85.2 86.6 71.6 83.4 82.2 78.5 62.7 41.3 68.2 55.5 77.1 69.7 70.8 InternVL3.5-8B [12] 84.0 86.7 78.4 86.5 85.9 83.1 73.4 57.2 69.3 59.5 77.4 70.8 67.5

Qwen2.5-VL-7B-Masters 88.6 95.6 78.8 89.1 89.5 81.7 71.3 60.6 74.9 67.2 81.8 75.9 77.3 Qwen3-VL-8B-Masters 88.5 95.9 81.8 89.5 89.9 79.4 72.9 61.4 79.7 72.3 81.7 75.1 77.5 InternVL3-8B-Masters 88.9 94.8 82.3 90.1 91.0 83.8 74.0 58.8 82.0 68.0 82.6 75.0 74.8 InternVL3.5-8B-Masters 87.2 95.1 85.0 88.2 88.3 85.6 72.7 58.1 80.8 67.8 81.4 75.5 74.9

Phi-3.5-Vision-4B [106] 77.8 81.8 43.9 76.0 66.1 43.2 43.0 19.7 47.5 58.3 69.7 62.2 53.6 Phi-4-Multimodal-5.6B [107] 83.0 81.4 65.8 86.7 - 51.9 56.0 38.5 58.9 42.1 73.2 68.5 64.1 LLaVA-OneVision-1.5-4B [53] 83.6 87.1 67.9 84.2 76.9 - 52.7 35.3 64.9 - 76.6 68.9 67.8 Ovis2-4B [101] 85.7 - 69.6 - - 65.5 49.0 - 61.9 53.0 76.2 69.3 71.1 Qwen2.5-VL-3B [17] 81.6 84.0 61.2 79.1 78.1 61.8 51.2 31.6 55.9 47.6 74.0 67.6 65.4 Qwen3-VL-4B [18] 84.1 87.4 73.7 84.6 84.3 71.0 67.4 53.2 69.8 65.8 78.4 70.2 70.6 InternVL3.5-4B [12] 82.6 86.0 77.1 86.9 86.1 76.6 66.6 51.4 65.0 58.1 75.2 69.4 66.3

Qwen2.5-VL-3B-Masters 85.4 95.6 75.6 85.9 86.3 78.3 61.2 51.0 68.9 53.0 78.8 72.1 70.7 Qwen3-VL-4B-Masters 88.0 94.7 79.6 88.7 89.3 79.4 70.3 57.0 75.3 69.1 81.5 72.9 77.8 InternVL3.5-4B-Masters 83.3 94.9 78.8 86.1 86.0 79.5 63.2 52.7 70.5 62.5 79.1 71.4 68.1

Aquila-VL-2B [108] 75.0 76.5 59.0 - - 43.8 47.4 - 54.9 34.1 73.9 63.0 65.0 Ovis2-2B [101] 82.7 - 64.1 - - 58.3 45.6 - 56.7 47.9 74.4 67.4 66.0 Qwen3-VL-2B [18] 76.9 81.2 61.3 81.9 81.4 61.4 53.4 36.5 58.3 53.8 76.0 66.4 63.9 InternVL3-2B [19] 78.7 80.2 57.0 81.1 78.4 62.2 48.6 24.9 60.7 47.0 75.0 64.6 64.3 InternVL3.5-2B [12] 78.8 80.7 71.8 82.3 81.8 71.7 59.0 46.3 62.7 51.3 75.2 68.0 62.0

Qwen3-VL-2B-Masters 82.1 94.1 70.4 84.2 84.1 71.3 55.8 39.5 66.1 62.3 78.3 69.4 69.0 InternVL3-2B-Masters 82.1 93.5 66.5 85.9 85.2 77.0 59.6 40.1 66.7 53.1 78.5 67.8 68.4 InternVL3.5-2B-Masters 83.0 94.8 77.5 85.9 86.3 81.4 64.6 52.4 70.7 61.1 79.0 71.6 68.1

𝑟max/𝑠 + 1. For example, setting 𝑟max = 0.2 and 𝑠 = 0.05 results in 𝑀 = 5, meaning that the masking ratio progressively decreases as 0.20, 0.15, 0.10, 0.05, and finally 0. Fig. 2(a) illustrates how the masking ratio 𝑟[𝑖] is scheduled during training.

As the masking ratio changes, the distillation objective also adapt to reflect the teacher’s gradually increasing capacity. Formally, the distillation objective at iteration 𝑖 is formulated as:

E(𝑥,𝑦)∼Data[𝑖] [︁𝒟 (︁𝑃𝒯𝑟[𝑖](𝑦|𝑥)‖𝑃𝒮(𝑦|𝑥))︁]︁,

min

W𝒮

(4) where W𝒮 indicates the weight set of the student 𝒮, and 𝑃(𝑦|𝑥) denotes the logit-softmax output for answer label 𝑦 given a question 𝑥 under the data subset Data[𝑖] at the iteration 𝑖. Note that 𝒟 refers to Jensen-Shannon Divergence (JSD) [36, 38], which has been empirically shown to outperform the standard KL divergence for distillation tasks. This mask-progressive distillation allows the student to first learn coarse-grained

knowledge and progressively refine its representations as the teacher’s representational capacity is restored throughout training.

Multiple Responses. While mask-progressive distillation enables the student to learn from a capacity-aligned teacher, the training data samples used for distillation still exhibit two inherent limitations. First, conventional SFT datasets are typically generated by large closed-source models [14, 54] or filtered by humans [56]. Injecting such responses into small student often leads to performance degradation due to representation mismatch—small vocabulary and low hidden dimension of representation. Second, standard SFT datasets generally provide only a single answer per question, which severely limits response diversity and generalization. This single-answer guidance forces the student to overfit to a narrow linguistic or reasoning style, reducing its versatility across broader vision–language contexts.

To address these limitations, Masters gener-

- Table 4 | Comparing Masters-applied Small VLMs with Closed-source and Large Open-source VLMs.

VLMs AI2D ChartQA MathVista MMB MM-Vet MMMU MMMU-Pro MMStar BLINK SEED SEED2+ RWQA

Claude-3.5-Sonnet [55] 81.2 90.8 67.7 82.6 70.1 68.3 51.5 65.1 60.1 61.7 71.7 65.8 Claude-3.7-Sonnet [55] 82.5 92.2 66.8 84.8 70.0 71.0 56.5 65.1 56.6 74.3 67.6 55.4 Claude-4-Sonnet [55] 83.0 - 74.6 - - 74.4 61.6 69.4 60.4 - - 69.8

- Gemini-1.5-Pro [109] 79.1 87.2 63.9 73.9 64.0 62.2 46.9 59.1 59.1 76.0 70.8 67.5
- Gemini-2.0-Flash [109] 83.1 - 70.4 90.0 73.6 69.9 54.4 69.4 64.0 - - 72.3 Gemini-2.5-Pro [54] 89.5 - 80.9 - 83.3 74.7 - 73.6 - - - GLM-4.5V [102] 88.1 86.6 84.6 - 75.2 75.4 65.2 75.3 65.3 - 74.0

- GPT-4o [110] 84.6 85.7 63.8 83.4 69.1 69.1 51.9 64.7 68.0 77.1 72.0 75.4 GPT-4.1 [110] 85.9 - 70.4 - 78.8 74.0 - 69.8 68.5 78.0 73.1 78.7 GPT-5-Mini [110] 88.2 - 79.1 - - 79.0 67.3 74.1 - - - 79.0
- GPT-5 [110] 89.5 - 81.9 - 77.6 84.2 - 75.7 - - - -

NVLM-72B [111] 85.2 86.0 66.6 - 58.9 59.7 - 63.7 48.0 75.5 68.4 69.9 LLaVA-OneVision-72B [99] 85.6 83.7 67.5 85.8 60.6 56.8 31.0 65.8 55.4 77.5 - 71.9 Molmo-72B [112] 83.4 87.3 58.6 - 61.1 54.1 - 63.3 49.7 74.6 67.6 73.7 Qwen2.5-VL-72B [17] 88.7 89.5 74.2 88.6 76.9 68.2 61.2 70.8 64.4 79.5 73.0 75.7 Qwen3-VL-32B [18] 89.5 89.8 83.8 90.6 79.4 76.0 65.3 77.7 67.3 79.9 72.8 79.0 InternVL3-78B [19] 89.7 89.7 79.0 89.0 81.3 72.2 62.3 72.5 66.3 78.7 71.9 78.0 InternVL3.5-38B [12] 87.8 88.8 81.9 90.3 82.2 76.9 66.0 75.3 60.9 79.1 71.0 75.9

Qwen2.5-VL-7B-Masters 88.6 95.6 78.8 89.1 81.7 71.3 60.6 74.9 67.2 81.8 75.9 77.3 Qwen3-VL-8B-Masters 88.5 95.9 81.8 89.5 79.4 72.9 61.4 79.7 72.3 81.7 75.1 77.5 InternVL3-8B-Masters 88.9 94.8 82.3 90.1 83.8 74.0 58.8 82.0 68.0 82.6 75.0 74.8 InternVL3.5-8B-Masters 87.2 95.1 85.0 88.2 85.6 72.7 58.1 80.8 67.8 81.4 75.5 74.9

ates multiple responses from the masked teacher, where the teacher’s representational capacity increases as the masking ratio decreases. These responses provide the student with adaptive guidance signals that evolve in step with their learning capacity. Furthermore, Masters incorporates the student’s own responses into training to maintain alignment between teacher guidance and the student’s evolving representational capacity. The formulation regarding multiple responses is written as follows:

min

W𝒮

E(𝑥,𝑦^)∼Gen-Data[𝑖] [︁𝒟 (︁𝑃𝒯𝑟[𝑖](^𝑦|𝑥)‖𝑃𝒮(^𝑦|𝑥))︁]︁,

(5) where Gen-Data[𝑖] denotes the pre-generated multi-response dataset obtained from both the masked teacher and the student at training iteration 𝑖. This enables stable yet continual improvement toward teacher-level behavior—effectively overcoming the single-answer and over-rich label constraints of conventional SFT datasets.

- 3.3. Reinforcing Student with Dual Rewards

curacy and the transferability of the responses and refine the student based on their feedback to avoid generating such undesirable responses.

Offline RL Setup. Conventional online RL paradigms, such as the “think-answer” process [60, 61], require generating multiple and long responses at each training iteration, incurring substantial computational overhead. To overcome this limitation, Masters adopts an offline RL setup where both the teacher and the student pre-generate multiple responses for all questions in the training dataset. This design dramatically reduces computational cost while enabling scaled-up training with diverse responses.

Accuracy Reward. To assess the correctness of generated responses, we adopt an LLM-asa-Judge [59], which measures semantic fidelity between the generated responses 𝑦^ and the answer labels 𝑦. The accuracy reward is defined as:

ℛacc = LLM-as-a-Judge(𝑥,𝑦,𝑦^ ) ∈ [0,1]. (6)

This reward guides the student to generate semantically accurate and faithful outputs, regardless of linguistic or stylistic variations. Unlike prior approaches such as DeepSeek-R1 [60] and its follow-up variants [64, 67, 66, 68], which depend on traditional parsers for limited domain evaluation metrics (e.g., math reasoning or spatial grounding), LLM-as-a-Judge method elimi-

Although generating multiple responses helps mitigate the constraints of conventional SFT datasets, some generated responses may include factual inaccuracies or overly complex language that hampers effective knowledge transfer, thereby reducing distillation performance. To address this, we evaluate both the factual ac-

nates such constraints. For example, traditional parsers fail when handling open-ended visual questions like “Describe the person’s emotion in this image”. Even, when comparing “about five minutes” (predicted) with “5” (answer label), traditional parsing says that the predicted response is wrong answer. On the other hands, LLM-asa-Judge [59] delivers consistent and robust judgments across diverse visual question answering (VQA) scenarios — including recognition, OCR, reasoning, chart and document understanding providing a more generalizable and semantically grounded measure of correctness.

Distillation Reward. While the accuracy reward ensures response correctness, it does not capture how effectively the student learns responses. To address this, we introduce a distillation reward that measures logit-level similarity between the teacher and student. Specifically, we compute this reward using 𝒟 defined in Eq. (4). Practically, we have observed that the values of 𝒟 exhibit small variance across the generated responses due to its low scaling. To stabilize optimization and enhance sensitivity [113], we apply a reverse min-max normalization [114] as follows:

ℛdistill = 𝒟max − 𝒟 𝒟max − 𝒟min

, (7)

where 𝒟min and 𝒟max denote the minimum and maximum divergence among the generated responses 𝑦^ for a given question 𝑥. This normalization ensures that a smaller divergence (indicating better alignment with the teacher) yields a higher reward value. The total reward is then computed as the sum of the accuracy and distillation rewards.

Final Objective. We adopt RL objective of GRPO [61] (see Appendix B) and extend it with Eq. (5), as follows:

[︀ℒGRPO + 𝒟 (︀

)︀]︀

𝑃𝒯𝑟[𝑖](^𝑦|𝑥) ‖ 𝑃𝒮(^𝑦|𝑥)

min

E(𝑥,𝑦^)∼Gen-Data[𝑖]

W𝒮

(8)

This unified objective integrates reinforcement and distillation, allowing the student to refine its responses in a stable manner. Fig. 3 and Fig. 4 represents overview of Masters.

### 4. Experiments

#### 4.1. Implementation Details

Model Selection. We select recently released, high-performing student and teacher VLMs based on Leaderboard [115]. For student, we employ several strong series, including Qwen2.5-VL-3B and -7B [17], Qwen3-VL2B, -4B, and -8B, InternVL3-2B, and -8B [19], as well as InternVL3.5-2B, -4B, and -8B [12]. For teacher, we select corresponding family of Qwen2.5-VL-32B and -72B [17], Qwen3-VL32B [18], InternVL3-14B, -38B, and -78B [19], and InternVL3.5-14B and -38B [12].

Training Setup. We train and evaluate Masters primarily on NVIDIA A100 80GB GPUs. We first set the decrement 𝑠 to 0.05 and save multiple masked teacher checkpoints with different masking ratios. For instance, when 𝑟max is set to 0.2, we store five masked teacher checkpoints corresponding to masking ratios of 0.20, 0.15,

- 0.10, 0.05, and 0. We then make all five masked teachers generate responses for the 1.5M dataset (see Appendix C), by using vLLM [116] for fast inference. Specifically, we generate 8 responses per question by setting the temperature to 1.0, top-p to 0.9, top-k to 50, and the repetition penalty to
- 1.05. We simultaneously evaluate the accuracy reward of the responses via LLM-as-a-Judge [59]. The model used for judging is the same as the one selected for generation, and we further refine the accuracy reward through additional parsing prompts (see Appendix D). For RL, we utilize the DeepSpeed engine with ZeRO-3 [117] to efficiently handle large teacher and student. The student is optimized using AdamW [118] with a fixed learning rate of 1 × 10−6.

#### 4.2. Step-wise Dissection of Masters

.

We conduct a series of step-wise experiments to construct Masters and identify the source of its effectiveness. The results of each distillation strategy are summarized in Tab. 1. We begin with a naive baseline that performs distillation from a large teacher using the objective of JSD [36, 38], without incorporating mask-

- Table 5 | Ablation studies on various configurations influencing the performance of Masters. Note that (a) reports the average performance across the evaluation benchmarks in Tab. 1, while (c), (d), and (e) are conducted using the InternVL3.5 series [12], with 8B student.

(a) Maximum of Masking Ratio

(b) Combinations of Teacher

(c) Source of Training Samples

Source AI2D MathVsita MMB MM-Vet MMMU 𝒯 (#8) 87.0 84.8 86.3 85.3 72.6

Teacher Size AI2D MathVista MMB MM-Vet MMMU

𝑟max Q2.5-VL-72B Q3-VL-32B IVL3-78B IVL3.5-38B 0 72.7 77.7 73.9 76.8

38B 86.8 73.4 85.3 82.1 65.1 78B 86.9 75.7 86.8 80.2 67.8 14B+38B 87.2 76.8 87.1 82.5 69.0 14B+78B 87.5 78.3 88.9 82.9 70.6 38B+78B 88.1 80.8 90.1 83.0 72.9 14B+38B+78B 88.9 82.3 90.1 83.8 74.0

- 𝒮(#2) + 𝒯 (#6) 87.0 84.6 86.8 85.0 72.4
- 𝒮(#3) + 𝒯 (#5) 87.1 84.8 87.2 85.2 72.2

- 𝒮(#4) + 𝒯 (#4) 87.2 85.0 88.2 85.6 72.7

- 𝒮(#5) + 𝒯 (#3) 87.0 84.6 87.0 85.1 72.0
- 𝒮(#6) + 𝒯 (#2) 86.8 84.3 86.5 84.6 71.5 𝒮(#8) 86.4 83.8 86.3 85.0 71.0

- 0.1 75.9 78.3 75.1 78.1
- 0.2 79.4 79.0 80.5 80.0

- 0.3 70.4 79.5 71.1 70.4
- 0.4 64.5 80.4 68.9 66.2

- 0.5 48.2 75.4 50.0 49.2

(e) Choice of Reward Design

(d) Decomposing Components

(f) Other Distillation Methods

Reward AI2D MathVista MMB MM-Vet MMStar

Mid Teacher Multi-Response Reward Feedback Avg

VLMs Masters MathVista MMB MM-Vet MMMU DistiLLM-7B [37]

Naive ✗ ✗ ✗ 75.8 Naive ✗ ✓ ✗ 76.2 Naive ✗ ✓ ✓ 76.8 Naive ✓ ✓ ✓ 77.3

✗ 61.5 84.8 65.3 57.9

ℛacc(△) 82.3 75.6 80.7 77.2 65.3 ℛacc 86.5 82.3 88.1 85.0 75.8 ℛdistill 86.3 80.3 88.0 84.8 72.0 ℛacc + ℛdistill(△) 86.6 82.5 88.1 85.2 76.3 ℛacc + ℛdistill 87.2 85.0 88.2 85.6 80.8 InternVL3.5-38B 87.8 81.9 90.3 82.2 75.3

###### ✓ 69.3 86.0 73.3 60.8

✗ 61.8 85.0 65.9 58.2

LLaVA-KD-7B [44]

###### ✓ 70.2 86.2 73.8 61.7

Mask-Progressive ✗ ✗ ✗ 76.0 Mask-Progressive ✗ ✓ ✗ 76.3 Mask-Progressive ✗ ✓ ✓ 77.1 Mask-Progressive ✓ ✓ ✓ 80.0

✗ 74.7 85.8 75.2 69.3 ✓ 75.1 86.9 75.8 69.8

VLsI-7B [39]

✗ 77.8 88.1 80.1 68.6 ✓ 82.3 90.1 83.8 74.0

RIL-8B [49]

progressive distillation or RL feedback. Next, we apply teacher masking and perform maskprogressive distillation based on Eq. (5), which yields clear improvements—confirming that capacity alignment between the teacher and the student stabilizes optimization. Finally, adding RL allows the student to further refine the responses through feedback (correctness and transferability). Overall, combining mask-progressive distillation with RL feedback results in a cumulative performance boost, demonstrating that each component contributes synergistically to the overall effectiveness of Masters.

Hours of Inference Time (Log)

[Figure 76]

Evaluation Benchmarks

Figure 6 | Comparing hours of inference time at one GPU across various models described in Tab. 3

We further analyze the impact of gradually increasing the teacher sizes during distillation (e.g., from 14B to 38B), as shown in Tab. 2. We first distill knowledge from an intermediate-sized (mid) teacher as a warm-up stage, and then perform distillation from a larger teacher. This gradual teacher-size scaling (Blue Color) consistently yields better performance than one-shot distillation from a large teacher (Green Color), indicating that progressive capacity alignment leads to smoother convergence and richer representations. Fig. 2(a) illustrates the masking ratio schedule and performance trends across different teacher settings. Notably, after the warm-up stage with the mid teacher shown in Fig. 2(b), performance accelerates significantly when distilling from the large teacher, highlighting the advantage of this teacher scaling strategy. In summary, when combined with mask-progressive distillation and RL feedback, Masters elevates the

student to a highly competitive level—matching or even surpassing many recent open- and closedsource VLMs, as reported in Tab. 3 and Tab. 4 across various model scales (Fig. 1).

4.3. Ablation Studies with Configurations We conduct comprehensive ablation studies to examine various configuration settings that influence the detailed performance of Masters. First, we investigate the number of pre-generated responses in Fig. 5. As this number increases, performance steadily improves and converges around eight responses. Therefore, we set the number of generated responses to eight to achieve a great balance between training efficiency and performance.

Next, Tab. 5(a) shows a grid search to determine the optimal maximum masking ratio (𝑟max) for each teacher, ranging from 0 to 0.5. For most teachers, the optimal value is 𝑟max = 0.2, whereas Qwen3-VL-32B [18] achieves its best results with 𝑟max = 0.4. Beyond this table, we similarly observe that InternVL3.5-8B and -14B [12] also perform best at 𝑟max = 0.4. We then analyze the effect of teacher composition when distilling InternVL3 series [19] from large teachers to an 8B student. Consistent with the results in Tab. 2, the best performance is achieved with gradual teacher-size scaling of 14B, 38B, and 78B in Tab. 5(b). This confirms that progressive teacher scaling enables more stable and capacityaligned knowledge transfer.

Furthermore, we examine how the source of generated responses affects performance. As shown in Tab. 5(c), using only teacher-generated responses limits the student’s adaptability to its own representational capacity, while using only student-generated responses restricts exposure to the teacher’s richer semantics. Consequently, we adopt a balanced 1:1 ratio of teacher- and studentgenerated responses to ensure both adaptability and semantic richness. Tab. 5(d) compares various configurations, including naive versus maskprogressive distillation, with and without midteacher scaling, multi-response generation, and reward feedback (Fig. 2(c)). The results demonstrate that each component contributes synergistically to performance, and removing any of them leads to degradation—highlighting their essential roles in performance acceleration.

Next, we analyze the impact of reward design by selectively removing the accuracy or distillation reward. The triangle symbol (△) for the accuracy reward indicates the rule-based evaluation, which does not rely on LLM-as-a-Judge [59], while the triangle for the distillation reward denotes exclusion of reverse min–max normalization. Removing the accuracy reward consistently results in performance below that of the large teacher, underscoring its importance in enabling the student to surpass the teacher. Without normalization, the distillation reward becomes less discriminative, yielding performance comparable to using only the accuracy reward. As

shown in Tab. 5(e), both accuracy and distillation rewards are essential for effectively identifying high-quality responses.

Finally, we apply Masters to other recent distillation frameworks: DistiLLM [37], LLaVAKD [44], VLsI [39], and RIL [49]. Masters consistently mitigates degradation caused by large parameter gaps. Notably, while most existing methods rely on multi-step pipelines with additional SFT phases to maintain stability, Masters achieves superior results in a single-step training process—demonstrating its simplicity, scalability, and training efficiency.

### 5. Discussion and Conclusion

Although Masters shows strong performance and scalability, it is trained in an offline manner, which limits adaptability to real-time feedback. In principle, training could continuously improve the model via online data sampling, but it remains computationally infeasible—for instance, even a single model requires over 30 days on 256 A100 GPUs for 1.5M samples, whereas Masters completes training in just two days. Future advances in efficient multi-node training and inference from vLLM-like libraries is strongly needed for online distillation within days.

Beyond this limitation, recent methods often adopt a “think-answer” paradigm, which improves reasoning but greatly increases latency (Fig. 6). Such approaches remain impractical for real-world or on-device use due to their heavy computational cost. In contrast, Masters attains strong performance without sacrificing inference speed, suggesting a promising direction for balancing intelligence and scalability toward efficient VLMs.

We present Masters, progressively restoring the teacher from mask and reinforcing the student with dual rewards in an offline setting. It achieves stable and scalable knowledge transfer from large to compact models. Future work may explore data-driven masking and RL-based replay buffers to better handle hard examples. We hope Masters will inspire further research toward deployable and efficient VLMs.

### References

- [1] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 235–251. Springer, 2016.
- [2] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022.
- [3] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.
- [4] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023.
- [5] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023.
- [6] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. arXiv preprint arXiv:2311.16502, 2023.
- [7] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024.
- [8] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330, 2024.

- [9] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. arXiv preprint arXiv:2404.12390, 2024.
- [10] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seedbench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023.
- [11] Bohao Li, Yuying Ge, Yi Chen, Yixiao Ge, Ruimao Zhang, and Ying Shan. Seed-bench2-plus: Benchmarking multimodal large language models with text-rich visual comprehension. arXiv preprint arXiv:2404.16790, 2024.
- [12] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.
- [13] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [14] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [15] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.
- [16] Yifan Li, Zhixin Lai, Wentao Bao, Zhen Tan, Anh Dao, Kewei Sui, Jiayi Shen, Dong Liu, Huan Liu, and Yu Kong. Visual large language models for generalized and specialized applications. arXiv preprint arXiv:2501.02765, 2025.
- [17] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [18] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,

- Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [19] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan, Hao Tian, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and testtime recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.
- [20] Guanqiao Qu, Qiyuan Chen, Wei Wei, Zheng Lin, Xianhao Chen, and Kaibin Huang. Mobile edge intelligence for large language models: A contemporary survey. IEEE Communications Surveys & Tutorials, 2025.
- [21] Xiangxiang Chu, Limeng Qiao, Xinyu Zhang, Shuang Xu, Fei Wei, Yang Yang, Xiaofei Sun, Yiming Hu, Xinyang Lin, Bo Zhang, et al. Mobilevlm v2: Faster and stronger baseline for vision language model. arXiv preprint arXiv:2402.03766, 2024.
- [22] Wei Chen, Zhiyuan Li, and Shuo Xin. Omnivlm: A token-compressed, sub-billion-parameter vision-language model for efficient on-device inference. arXiv preprint arXiv:2412.11475, 2024.
- [23] Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li, Cem Koc, Nate True, Albert Antony, Gokula Santhanam, James Gabriel, Peter Grasch, Oncel Tuzel, et al. Fastvlm: Efficient vision encoding for vision language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19769– 19780, 2025.
- [24] Andrés Marafioti, Orr Zohar, Miquel Farré, Merve Noyan, Elie Bakouch, Pedro Cuenca, Cyril Zakka, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, et al. Smolvlm: Redefining small and efficient multimodal models. arXiv preprint arXiv:2504.05299, 2025.
- [25] Ahmed Sharshar, Latif U Khan, Waseem Ullah, and Mohsen Guizani. Vision-language models for edge networks: A comprehensive survey. IEEE Internet of Things Journal, 2025.
- [26] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Chi Chen, Haoyu Li, Weilin Zhao, et al. Efficient gpt-4v level multimodal large language model for deployment on edge devices. Nature Communications, 16(1):5509, 2025.

- [27] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.
- [28] Jianping Gou, Baosheng Yu, Stephen J Maybank, and Dacheng Tao. Knowledge distillation: A survey. International journal of computer vision, 129(6):1789–1819, 2021.
- [29] Chen Zhang, Yang Yang, Jiahao Liu, Jingang Wang, Yunsen Xian, Benyou Wang, and Dawei Song. Lifting the curse of capacity gap in distilling language models. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4535–4553, Toronto, Canada, July 2023. Association for Computational Linguistics.
- [30] Jia Guo, Minghao Chen, Yao Hu, Chen Zhu, Xiaofei He, and Deng Cai. Reducing the teacherstudent gap via spherical knowledge disitllation. arXiv preprint arXiv:2010.07485, 2020.
- [31] Seyed Iman Mirzadeh, Mehrdad Farajtabar, Ang Li, Nir Levine, Akihiro Matsukawa, and Hassan Ghasemzadeh. Improved knowledge distillation via teacher assistant. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 5191–5198, 2020.
- [32] Wangchunshu Zhou, Canwen Xu, and Julian McAuley. Bert learns to teach: Knowledge distillation with meta learning. arXiv preprint arXiv:2106.04570, 2021.
- [33] Yi Yang, Chen Zhang, and Dawei Song. Sparse teachers can be dense with knowledge. arXiv preprint arXiv:2210.03923, 2022.
- [34] Chen Zhang, Qiuchi Li, Dawei Song, Zheyu Ye, Yan Gao, and Yan Hu. Towards the law of capacity gap in distilling language models. arXiv preprint arXiv:2311.07052, 2023.
- [35] Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. Minillm: Knowledge distillation of large language models. In The Twelfth International Conference on Learning Representations, 2024.
- [36] Shilin Xu, Xiangtai Li, Haobo Yuan, Lu Qi, Yunhai Tong, and Ming-Hsuan Yang. Llavadi: What matters for multimodal large language models distillation. arXiv preprint arXiv:2407.19409, 2024.
- [37] Jongwoo Ko, Sungnyun Kim, Tianyi Chen, and Se-Young Yun. Distillm: Towards streamlined

- distillation for large language models. arXiv preprint arXiv:2402.03898, 2024.
- [38] Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In The Twelfth International Conference on Learning Representations, 2024.
- [39] Byung-Kwan Lee, Ryo Hachiuma, YuChiang Frank Wang, Yong Man Ro, and Yueh-Hua Wu. Vlsi: Verbalized layers-tointeractions from large to small vision language models. arXiv preprint arXiv:2412.01822,

- 2024.

[40] Jiwan Kim, Kibum Kim, Sangwoo Seo, and Chanyoung Park. Compodistill: Attention distillation for compositional reasoning in multimodal llms. arXiv preprint arXiv:2510.12184,

- 2025.

- [41] Qianhan Feng, Wenshuo Li, Tong Lin, and Xinghao Chen. Align-kd: Distilling cross-modal alignment knowledge for mobile vision-language model. arXiv preprint arXiv:2412.01282, 2024.
- [42] Jiajun Cao, Yuan Zhang, Tao Huang, Ming Lu, Qizhe Zhang, Ruichuan An, Ningning Ma, and Shanghang Zhang. Move-kd: Knowledge distillation for vlms with mixture of visual encoders. arXiv preprint arXiv:2501.01709, 2025.
- [43] Cheng Han, Qifan Wang, Sohail A Dianat, Majid Rabbani, Raghuveer M Rao, Yi Fang, Qiang Guan, Lifu Huang, and Dongfang Liu. Amd: Automatic multi-step distillation of large-scale vision models. In European Conference on Computer Vision, pages 431–450. Springer, 2024.
- [44] Yuxuan Cai, Jiangning Zhang, Haoyang He, Xinwei He, Ao Tong, Zhenye Gan, Chengjie Wang, and Xiang Bai. Llava-kd: A framework of distilling multimodal large language models. arXiv preprint arXiv:2410.16236, 2024.
- [45] Nicolas Boizard, Kevin El Haddad, Céline Hudelot, and Pierre Colombo. Towards cross-tokenizer distillation: the universal logit distillation loss for llms. arXiv preprint arXiv:2402.12030, 2024.
- [46] Byung-Kwan Lee, Ryo Hachiuma, Yong Man Ro, Yu-Chiang Frank Wang, and Yueh-Hua Wu. Genrecal: Generation after recalibration from large to small vision-language models. arXiv preprint arXiv:2506.15681, 2025.

- [47] Hongling Xu, Qi Zhu, Heyuan Deng, Jinpeng Li, Lu Hou, Yasheng Wang, Lifeng Shang, Ruifeng Xu, and Fei Mi. Kdrl: Post-training reasoning llms via unified knowledge distillation and reinforcement learning. arXiv preprint arXiv:2506.02208, 2025.
- [48] Chuanguang Yang, Xinqiang Yu, Han Yang, Zhulin An, Chengqing Yu, Libo Huang, and Yongjun Xu. Multi-teacher knowledge distillation with reinforcement learning for visual recognition. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 9148–9156, 2025.
- [49] Byung-Kwan Lee, Ryo Hachiuma, Yong Man Ro, Yu-Chiang Frank Wang, and Yueh-Hua Wu. Unified reinforcement and imitation learning for vision-language models, 2025.
- [50] Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for efficient neural network. Advances in neural information processing systems, 28, 2015.
- [51] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860, 2024.
- [52] Xiang Yue, Tuney Zheng, Ge Zhang, and Wenhu Chen. Mammoth2: Scaling instructions from the web. 2024.
- [53] Xiang An, Yin Xie, Kaicheng Yang, Wenkang Zhang, Xiuwei Zhao, Zheng Cheng, Yirui Wang, Songcen Xu, Changrui Chen, Chunsheng Wu, et al. Llava-onevision-1.5: Fully open framework for democratized multimodal training. arXiv preprint arXiv:2509.23661, 2025.
- [54] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- [55] Anthropic. The claude 3 model family: Opus, sonnet, haiku. https://www.anthropic.com, 2024.
- [56] Zhiyang Xu, Chao Feng, Rulin Shao, Trevor Ashby, Ying Shen, Di Jin, Yu Cheng, Qifan

- Wang, and Lifu Huang. Vision-flan: Scaling human-labeled tasks in visual instruction tuning. arXiv preprint arXiv:2402.11690, 2024.
- [57] Sanqiang Zhao, Raghav Gupta, Yang Song, and Denny Zhou. Extremely small bert models from mixed-vocabulary training. arXiv preprint arXiv:1909.11687, 2019.
- [58] Songming Zhang, Xue Zhang, Zengkui Sun, Yufeng Chen, and Jinan Xu. Dual-space knowledge distillation for large language models. arXiv preprint arXiv:2406.17328, 2024.
- [59] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mtbench and chatbot arena. Advances in Neural Information Processing Systems, 36:46595– 46623, 2023.
- [60] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [61] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [62] Zhengxi Lu, Yuxiang Chai, Yaxuan Guo, Xi Yin, Liang Liu, Hao Wang, Guanjing Xiong, and Hongsheng Li. Ui-r1: Enhancing action prediction of gui agents by reinforcement learning. arXiv preprint arXiv:2503.21620, 2025.
- [63] Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai Yang, Xingzhong Xu, Xin Geng, and Xu Yang. Lmmr1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl. arXiv preprint arXiv:2503.07536, 2025.
- [64] En Yu, Kangheng Lin, Liang Zhao, Jisheng Yin, Yana Wei, Yuang Peng, Haoran Wei, Jianjian Sun, Chunrui Han, Zheng Ge, et al. Perception-r1: Pioneering perception policy with reinforcement learning. arXiv preprint arXiv:2504.07954, 2025.
- [65] Xiangyan Liu, Jinjie Ni, Zijian Wu, Chao Du, Longxu Dou, Haonan Wang, Tianyu Pang, and

- Michael Qizhe Shieh. Noisyrollout: Reinforcing visual reasoning with data augmentation. arXiv preprint arXiv:2504.13055, 2025.
- [66] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025.
- [67] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual-rft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025.
- [68] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025.
- [69] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108, 2019.
- [70] Iulia Turc, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Well-read students learn better: On the importance of pre-training compact models. arXiv preprint arXiv:1908.08962, 2019.
- [71] Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, and Yoshua Bengio. Fitnets: Hints for thin deep nets. arXiv preprint arXiv:1412.6550, 2014.
- [72] Siqi Sun, Yu Cheng, Zhe Gan, and Jingjing Liu. Patient knowledge distillation for bert model compression. arXiv preprint arXiv:1908.09355, 2019.
- [73] Pengguang Chen, Shu Liu, Hengshuang Zhao, and Jiaya Jia. Distilling knowledge via knowledge review. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5008–5017, 2021.
- [74] Emanuel Ben-Baruch, Matan Karklinsky, Yossi Biton, Avi Ben-Cohen, Hussam Lawen, and Nadav Zamir. It’s all in the head: Representation knowledge distillation through classifier sharing. arXiv preprint arXiv:2201.06945, 2022.

- [75] Jiabao Wang, Yuming Chen, Zhaohui Zheng, Xiang Li, Ming-Ming Cheng, and Qibin Hou. Crosskd: Cross-head knowledge distillation for object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16520–16530, 2024.
- [76] Nikolaos Passalis, Maria Tzelepi, and Anastasios Tefas. Probabilistic knowledge transfer for lightweight deep representation learning. IEEE Transactions on Neural Networks and learning systems, 32(5):2030–2039, 2020.
- [77] Wonpyo Park, Dongju Kim, Yan Lu, and Minsu Cho. Relational knowledge distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3967–3976, 2019.
- [78] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive representation distillation. arXiv preprint arXiv:1910.10699, 2019.
- [79] Sergey Zagoruyko and Nikos Komodakis. Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. arXiv preprint arXiv:1612.03928, 2016.
- [80] Junho Yim, Donggyu Joo, Jihoon Bae, and Junmo Kim. A gift from knowledge distillation: Fast optimization, network minimization and transfer learning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4133–4141, 2017.
- [81] Frederick Tung and Greg Mori. Similaritypreserving knowledge distillation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1365–1374, 2019.
- [82] Byeongho Heo, Jeesoo Kim, Sangdoo Yun, Hyojin Park, Nojun Kwak, and Jin Young Choi. A comprehensive overhaul of feature distillation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1921–1930, 2019.
- [83] Inar Timiryasov and Jean-Loup Tastet. Baby llama: knowledge distillation from an ensemble of teachers trained on a small dataset with no performance penalty. arXiv preprint arXiv:2308.02019, 2023.
- [84] Young-Suk Lee, Md Sultan, Yousef El-Kurdi, Tahira Naseem, Asim Munawar, Radu Florian, Salim Roukos, and Ramón Fernandez Astudillo. Ensemble-instruct: Instruction tuning data generation with a heterogeneous mixture of lms. In

- Findings of the Association for Computational Linguistics: EMNLP 2023, pages 12561–12571, 2023.
- [85] Fanqi Wan, Longguang Zhong, Ziyi Yang, Ruijun Chen, and Xiaojun Quan. Fusechat: Knowledge fusion of chat models. arXiv preprint arXiv:2408.07990, 2024.
- [86] Kunran Xu, Lai Rui, Yishi Li, and Lin Gu. Feature normalized knowledge distillation for image classification. In European conference on computer vision, pages 664–680. Springer, 2020.
- [87] Wonchul Son, Jaemin Na, Junyong Choi, and Wonjun Hwang. Densely guided knowledge distillation using multiple teacher assistants. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9395– 9404, 2021.
- [88] Guo-Hua Wang, Yifan Ge, and Jianxin Wu. Distilling knowledge by mimicking features. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(11):8183–8195, 2021.
- [89] Cheng-Yu Hsieh, Chun-Liang Li, Chih-Kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alexander Ratner, Ranjay Krishna, Chen-Yu Lee, and Tomas Pfister. Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes. arXiv preprint arXiv:2305.02301, 2023.
- [90] Yijun Tian, Yikun Han, Xiusi Chen, Wei Wang, and Nitesh V Chawla. Beyond answers: Transferring reasoning capabilities to smaller llms using multi-teacher knowledge distillation. arXiv preprint arXiv:2402.04616, 2024.
- [91] Yann LeCun, John Denker, and Sara Solla. Optimal brain damage. Advances in neural information processing systems, 2, 1989.
- [92] Babak Hassibi, David G Stork, and Gregory J Wolff. Optimal brain surgeon and general network pruning. In IEEE international conference on neural networks, pages 293–299. IEEE, 1993.
- [93] Yihui He, Xiangyu Zhang, and Jian Sun. Channel pruning for accelerating very deep neural networks. In Proceedings of the IEEE international conference on computer vision, pages 1389–1397, 2017.
- [94] Chaoqi Wang, Roger Grosse, Sanja Fidler, and Guodong Zhang. Eigendamage: Structured pruning in the kronecker-factored eigenbasis. In International conference on machine learning, pages 6566–6575. PMLR, 2019.

- [95] Sidak Pal Singh and Dan Alistarh. Woodfisher: Efficient second-order approximation for neural network compression. Advances in Neural Information Processing Systems, 33:18098–18109, 2020.
- [96] Byung-Kwan Lee, Junho Kim, and Yong Man Ro. Masking adversarial damage: Finding adversarial saliency for robust and sparse network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15126–15136, 2022.
- [97] Yang He and Lingao Xiao. Structured pruning for deep convolutional neural networks: A survey. IEEE transactions on pattern analysis and machine intelligence, 46(5):2900–2919, 2023.
- [98] Hongrong Cheng, Miao Zhang, and Javen Qinfeng Shi. A survey on deep neural network pruning: Taxonomy, comparison, analysis, and recommendations. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.
- [99] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llavaonevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [100] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024.
- [101] Shiyin Lu, Yang Li, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, and HanJia Ye. Ovis: Structural embedding alignment for multimodal large language model. arXiv:2405.20797, 2024.
- [102] Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. Glm4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv e-prints, pages arXiv–2507, 2025.
- [103] Core Team, Zihao Yue, Zhenru Lin, Yifan Song, Weikun Wang, Shuhuai Ren, Shuhao Gu, Shicheng Li, Peidian Li, Liang Zhao, Lei Li, Kainan Bao, Hao Tian, Hailin Zhang, Gang Wang, Dawei Zhu, Cici, Chenhong He, Bowen Ye, Bowen Shen, Zihan Zhang, Zihan Jiang, Zhixian Zheng, Zhichao Song, Zhenbo Luo, Yue Yu, Yudong Wang, Yuanyuan Tian, Yu Tu, Yihan Yan, Yi Huang, Xu Wang, Xinzhe Xu,

- Xingchen Song, Xing Zhang, Xing Yong, Xin Zhang, Xiangwei Deng, Wenyu Yang, Wenhan Ma, Weiwei Lv, Weiji Zhuang, Wei Liu, Sirui Deng, Shuo Liu, Shimao Chen, Shihua Yu, Shaohui Liu, Shande Wang, Rui Ma, Qiantong Wang, Peng Wang, Nuo Chen, Menghang Zhu, Kangyang Zhou, Kang Zhou, Kai Fang, Jun Shi, Jinhao Dong, Jiebao Xiao, Jiaming Xu, Huaqiu Liu, Hongshen Xu, Heng Qu, Haochen Zhao, Hanglong Lv, Guoan Wang, Duo Zhang, Dong Zhang, Di Zhang, Chong Ma, Chang Liu, Can Cai, and Bingquan Xia. Mimo-vl technical report, 2025.
- [104] Kwai Keye Team, Biao Yang, Bin Wen, Changyi Liu, Chenglong Chu, Chengru Song, Chongling Rao, Chuan Yi, Da Li, Dunju Zang, et al. Kwai keye-vl technical report. arXiv preprint arXiv:2507.01949, 2025.
- [105] Biao Yang, Bin Wen, Boyang Ding, Changyi Liu, Chenglong Chu, Chengru Song, Chongling Rao, Chuan Yi, Da Li, Dunju Zang, et al. Kwai keye-vl 1.5 technical report. arXiv preprint arXiv:2509.01563, 2025.
- [106] Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219, 2024.
- [107] Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, et al. Phi4-mini technical report: Compact yet powerful multimodal language models via mixture-ofloras. arXiv preprint arXiv:2503.01743, 2025.
- [108] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.
- [109] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

- [110] OpenAI. Gpt-4v(ision) system card,

2023. https://openai.com/research/ gpt-4v-system-card, Last accessed on 2024-02-13.

- [111] Wenliang Dai, Nayeon Lee, Boxin Wang, Zhuolin Yang, Zihan Liu, Jon Barker, Tuomas Rintamaki, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Nvlm: Open frontier-class multimodal llms. arXiv preprint, 2024.
- [112] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for stateof-the-art multimodal models. arXiv preprint arXiv:2409.17146, 2024.
- [113] Tom Schaul, Georg Ostrovski, Iurii Kemaev, and Diana Borsa. Return-based scaling: Yet another normalisation trick for deep rl. arXiv preprint arXiv:2105.05347, 2021.
- [114] Mingqi Yuan, Roger Creus Castanyer, Bo Li, Xin Jin, Wenjun Zeng, and Glen Berseth. Rlexplore: Accelerating research in intrinsicallymotivated reinforcement learning. arXiv preprint arXiv:2405.19548, 2024.
- [115] OpenCompass Contributors. Opencompass: A universal evaluation platform for foundation models. https://github.com/open-compass/ opencompass, 2023.
- [116] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, pages 611–626, 2023.
- [117] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–16. IEEE, 2020.
- [118] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.
- [119] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. InstructBLIP: Towards general-purpose vision-language models with instruction tuning. In Thirty-seventh

- Conference on Neural Information Processing Systems, 2023.
- [120] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.
- [121] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instructionfinetuned language models. arXiv preprint arXiv:2210.11416, 2022.
- [122] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1–113, 2023.
- [123] Young-Jun Lee, Byung-Kwan Lee, Jianshu Zhang, Yechan Hwang, Byungsoo Ko, Han-Gyu Kim, Dongyu Yao, Xuankun Rong, Eojin Joo, Seung-Ho Han, et al. Multiverse: A multi-turn conversation benchmark for evaluating large vision and language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 708–719, 2025.
- [124] Young-Jun Lee, Seungone Kim, ByungKwan Lee, Minkyeong Moon, Yechan Hwang, Jong Myoung Kim, Graham Neubig, Sean Welleck, and Ho-Jin Choi. Refinebench: Evaluating refinement capability of language models via checklists. arXiv preprint arXiv:2511.22173, 2025.
- [125] Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva: Exploring the limits of masked visual representation learning at scale. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19358–19369, 2023.
- [126] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

- [127] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015– 4026, 2023.
- [128] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023.
- [129] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1290–1299, 2022.
- [130] Matthias Minderer, Alexey A. Gritsenko, and Neil Houlsby. Scaling open-vocabulary object detection. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.
- [131] Jingkang Yang, Yi Zhe Ang, Zujin Guo, Kaiyang Zhou, Wayne Zhang, and Ziwei Liu. Panoptic scene graph generation. In European Conference on Computer Vision, pages 178–196. Springer, 2022.
- [132] Yuning Du, Chenxia Li, Ruoyu Guo, Cheng Cui, Weiwei Liu, Jun Zhou, Bin Lu, Yehua Yang, Qiwen Liu, Xiaoguang Hu, et al. Pp-ocrv2: Bag of tricks for ultra lightweight ocr system. arXiv preprint arXiv:2109.03144, 2021.
- [133] Byung-Kwan Lee, Chae Won Kim, Beomchan Park, and Yong Man Ro. Meteor: Mamba-based traversal of rationale for large language and vision models. arXiv preprint arXiv:2405.15574, 2024.
- [134] Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769, 2024.
- [135] Derong Xu, Xinhang Li, Ziheng Zhang, Zhenxi Lin, Zhihong Zhu, Zhi Zheng, Xian Wu, Xiangyu Zhao, Tong Xu, and Enhong Chen. Harnessing large language models for knowledge graph question answering via adaptive multiaspect retrieval-augmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 25570–25578, 2025.

- [136] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Yaofeng Sun, et al. Deepseekvl: towards real-world vision-language understanding. arXiv preprint arXiv:2403.05525, 2024.
- [137] Elizaveta Goncharova, Anton Razzhigaev, Matvey Mikhalchuk, Maxim Kurkin, Irina Abdullaeva, Matvey Skripkin, Ivan Oseledets, Denis Dimitrov, and Andrey Kuznetsov. Omnifusion technical report. arXiv preprint arXiv:2404.06212, 2024.
- [138] Zhuofan Zong, Bingqi Ma, Dazhong Shen, Guanglu Song, Hao Shao, Dongzhi Jiang, Hongsheng Li, and Yu Liu. Mova: Adapting mixture of vision experts to multimodal context. arXiv preprint arXiv:2404.13046, 2024.
- [139] Byung-Kwan Lee, Beomchan Park, Chae Won Kim, and Yong Man Ro. Collavo: Crayon large language and vision model. arXiv preprint arXiv:2402.11248, 2024.
- [140] Byung-Kwan Lee, Beomchan Park, Chae Won Kim, and Yong Man Ro. Moai: Mixture of all intelligence for large language and vision models. arXiv preprint arXiv:2403.07508, 2024.
- [141] Min Shi, Fuxiao Liu, Shihao Wang, Shijia Liao, Subhashree Radhakrishnan, De-An Huang, Hongxu Yin, Karan Sapra, Yaser Yacoob, Humphrey Shi, et al. Eagle: Exploring the design space for multimodal llms with mixture of encoders. arXiv preprint arXiv:2408.15998, 2024.
- [142] Omkar Thawakar, Ashmal Vayani, Salman Khan, Hisham Cholakal, Rao M. Anwer, Michael Felsberg, Tim Baldwin, Eric P. Xing, and Fahad Shahbaz Khan. Mobillama: Towards accurate and lightweight fully transparent gpt, 2024.
- [143] Sachin Mehta, Mohammad Hossein Sekhavat, Qingqing Cao, Maxwell Horton, Yanzi Jin, Chenfan Sun, Iman Mirzadeh, Mahyar Najibi, Dmitry Belenko, Peter Zatloukal, et al. Openelm: An efficient language model family with open-source training and inference framework. arXiv preprint arXiv:2404.14619, 2024.
- [144] Zechun Liu, Changsheng Zhao, Forrest Iandola, Chen Lai, Yuandong Tian, Igor Fedorov, Yunyang Xiong, Ernie Chang, Yangyang

- Shi, Raghuraman Krishnamoorthi, et al. Mobilellm: Optimizing sub-billion parameter language models for on-device use cases. arXiv preprint arXiv:2402.14905, 2024.
- [145] Byung-Kwan Lee, Sangyun Chung, Chae Won Kim, Beomchan Park, and Yong Man Ro. Trol: Traversal of layers for large language and vision models. arXiv preprint arXiv:2406.12246, 2024.
- [146] Byung-Kwan Lee, Sangyun Chung, Chae Won Kim, Beomchan Park, and Yong Man Ro. Phantom of latent for large language and vision models. arXiv preprint arXiv:2409.14713, 2024.
- [147] Baichuan Zhou, Ying Hu, Xi Weng, Junlong Jia, Jie Luo, Xien Liu, Ji Wu, and Lei Huang. Tinyllava: A framework of smallscale large multimodal models. arXiv preprint arXiv:2402.14289, 2024.
- [148] Xiangxiang Chu, Limeng Qiao, Xinyang Lin, Shuang Xu, Yang Yang, Yiming Hu, Fei Wei, Xinyu Zhang, Bo Zhang, Xiaolin Wei, et al. Mobilevlm: A fast, reproducible and strong vision language assistant for mobile devices. arXiv preprint arXiv:2312.16886, 2023.
- [149] Bin Lin, Zhenyu Tang, Yang Ye, Jiaxi Cui, Bin Zhu, Peng Jin, Junwu Zhang, Munan Ning, and Li Yuan. Moe-llava: Mixture of experts for large vision-language models. arXiv preprint arXiv:2401.15947, 2024.
- [150] Yanyuan Qiao, Zheng Yu, Longteng Guo, Sihan Chen, Zijia Zhao, Mingzhen Sun, Qi Wu, and Jing Liu. Vl-mamba: Exploring state space models for multimodal learning. arXiv preprint arXiv:2403.13600, 2024.
- [151] Han Zhao, Min Zhang, Wei Zhao, Pengxiang Ding, Siteng Huang, and Donglin Wang. Cobra: Extending mamba to multi-modal large language model for efficient inference. arXiv preprint arXiv:2403.14520, 2024.
- [152] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.
- [153] Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. Adaptive mixtures of local experts. Neural computation, 3(1):79–87, 1991.
- [154] David Eigen, Marc’Aurelio Ranzato, and Ilya Sutskever. Learning factored representations in a deep mixture of experts. arXiv preprint arXiv:1312.4314, 2013.

- [155] Yoshua Bengio, Nicholas Léonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013.
- [156] Noam Shazeer, *Azalia Mirhoseini, *Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-ofexperts layer. In International Conference on Learning Representations, 2017.
- [157] Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, André Susano Pinto, Daniel Keysers, and Neil Houlsby. Scaling vision with sparse mixture of experts. Advances in Neural Information Processing Systems, 34:8583–8595, 2021.
- [158] Shweta Singh, Aayan Yadav, Jitesh Jain, Humphrey Shi, Justin Johnson, and Karan Desai. Benchmarking object detectors with coco: A new path forward. In European Conference on Computer Vision, pages 279–295. Springer, 2024.
- [159] Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui, Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and Serge Belongie. The inaturalist species classification and detection dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8769–8778, 2018.
- [160] Yash Goyal, Tejas Khot, Douglas SummersStay, Dhruv Batra, and Devi Parikh. Making the V in VQA matter: Elevating the role of image understanding in Visual Question Answering. In Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
- [161] Zhuowan Li, Xingrui Wang, Elias StengelEskin, Adam Kortylewski, Wufei Ma, Benjamin Van Durme, and Alan L Yuille. Super-clevr: A virtual benchmark to diagnose domain robustness in visual reasoning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14963–14973, 2023.
- [162] Renrui Zhang, Xinyu Wei, Dongzhi Jiang, Yichi Zhang, Ziyu Guo, Chengzhuo Tong, Jiaming Liu, Aojun Zhou, Bin Wei, Shanghang Zhang, et al. Mavis: Mathematical visual instruction tuning. arXiv e-prints, pages arXiv–2407, 2024.
- [163] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun

- Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. In The 59th Annual Meeting of the Association for Computational Linguistics (ACL), 2021.
- [164] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521, 2022.
- [165] Yanzhe Zhang, Ruiyi Zhang, Jiuxiang Gu, Yufan Zhou, Nedim Lipka, Diyi Yang, and Tong Sun. Llavar: Enhanced visual instruction tuning for text-rich image understanding. arXiv preprint arXiv:2306.17107, 2023.
- [166] Fangyu Liu, Guy Emerson, and Nigel Collier. Visual spatial reasoning. Transactions of the Association for Computational Linguistics, 11:635– 651, 2023.
- [167] Manoj Acharya, Kushal Kafle, and Christopher Kanan. Tallyqa: Answering complex counting questions. In Proceedings of the AAAI conference on artificial intelligence, volume 33, pages 8076–8084, 2019.
- [168] Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. Dynamic prompt learning via policy gradient for semistructured mathematical reasoning. arXiv preprint arXiv:2209.14610, 2022.
- [169] Vlad Hosu, Hanhe Lin, Tamas Sziranyi, and Dietmar Saupe. Koniq-10k: An ecologically valid database for deep learning of blind image quality assessment. IEEE Transactions on Image Processing, 29:4041–4056, 2020.
- [170] Weiyun Wang, Zhe Chen, Wenhai Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Jinguo Zhu, Xizhou Zhu, Lewei Lu, Yu Qiao, et al. Enhancing the reasoning ability of multimodal large language models via mixed preference optimization. arXiv preprint arXiv:2411.10442, 2024.
- [171] Tianyu Yu, Haoye Zhang, Yuan Yao, Yunkai Dang, Da Chen, Xiaoman Lu, Ganqu Cui, Taiwen He, Zhiyuan Liu, Tat-Seng Chua, et al. Rlaif-v: Aligning mllms through open-source ai feedback for super gpt-4v trustworthiness. arXiv preprint arXiv:2405.17220, 2024.

- [172] Adam Dahlgren Lindström and Savitha Sam Abraham. Clevr-math: A dataset for compositional language, visual and mathematical reasoning. arXiv preprint arXiv:2208.05358, 2022.
- [173] Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shijian Lu, and CV Jawahar. Icdar2019 competition on scanned receipt ocr and information extraction. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1516–1520. IEEE, 2019.
- [174] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021.
- [175] Samira Ebrahimi Kahou, Vincent Michalski, Adam Atkinson, Ákos Kádár, Adam Trischler, and Yoshua Bengio. Figureqa: An annotated figure dataset for visual reasoning. arXiv preprint arXiv:1710.07300, 2017.
- [176] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019.
- [177] Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1697–1706, 2022.
- [178] Qiguang Chen, Libo Qin, Jin Zhang, Zhi Chen, Xiao Xu, and Wanxiang Che. M3cot: A novel benchmark for multi-domain multi-step multimodal chain-of-thought. In Proc. of ACL, 2024.
- [179] Shuaichen Chang, David Palzer, Jialin Li, Eric Fosler-Lussier, and Ningchuan Xiao. Mapqa: A dataset for question answering on choropleth maps. arXiv preprint arXiv:2211.08545, 2022.
- [180] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE/cvf conference on computer vision and pattern recognition, pages 3195–3204, 2019.
- [181] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi

- Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019.
- [182] Yujie Lu, Dongfu Jiang, Wenhu Chen, William Yang Wang, Yejin Choi, and Bill Yuchen Lin. Wildvision: Evaluating visionlanguage models in the wild with human preferences. arXiv preprint arXiv:2406.11069, 2024.
- [183] Kushal Kafle, Brian Price, Scott Cohen, and Christopher Kanan. Dvqa: Understanding data visualizations via question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5648–5656, 2018.
- [184] Jie Cao and Jing Xiao. An augmented benchmark dataset for geometric question answering through dual parallel text encoding. In Proceedings of the 29th international conference on computational linguistics, pages 1511–1520, 2022.
- [185] Minjoon Seo, Hannaneh Hajishirzi, Ali Farhadi, Oren Etzioni, and Clint Malcolm. Solving geometry problems: Combining text and diagram interpretation. In Proceedings of the 2015 conference on empirical methods in natural language processing, pages 1466–1476, 2015.
- [186] Pan Lu, Liang Qiu, Jiaqi Chen, Tony Xia, Yizhou Zhao, Wei Zhang, Zhou Yu, Xiaodan Liang, and Song-Chun Zhu. Iconqa: A new benchmark for abstract diagram understanding and visual language reasoning. arXiv preprint arXiv:2110.13214, 2021.
- [187] Jiaqi Chen, Tong Li, Jinghui Qin, Pan Lu, Liang Lin, Chongyu Chen, and Xiaodan Liang. Unigeo: Unifying geometry logical reasoning via reformulating mathematical expression. arXiv preprint arXiv:2212.02746, 2022.
- [188] Mehran Kazemi, Hamidreza Alvari, Ankit Anand, Jialin Wu, Xi Chen, and Radu Soricut. Geomverse: A systematic evaluation of large models for geometric reasoning. arXiv preprint arXiv:2312.12241, 2023.
- [189] Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, et al. Gllava: Solving geometric problem with multimodal large language model. arXiv preprint arXiv:2312.11370, 2023.

- [190] Wenhao Shi, Zhiqiang Hu, Yi Bin, Junhua Liu, Yang Yang, See-Kiong Ng, Lidong Bing, and Roy Ka-Wei Lee. Math-llava: Bootstrapping mathematical reasoning for multimodal large language models. arXiv preprint arXiv:2406.17294, 2024.
- [191] Xinyu Huang, Yi-Jie Huang, Youcai Zhang, Weiwei Tian, Rui Feng, Yuejie Zhang, Yanchun Xie, Yaqian Li, and Lei Zhang. Open-set image tagging with multi-grained text supervision. arXiv preprint arXiv:2310.15200, 2023.
- [192] Shuhao Gu, Jialing Zhang, Siyuan Zhou, Kevin Yu, Zhaohu Xing, Liangdong Wang, Zhou Cao, Jintao Jia, Zhuoyi Zhang, Yixuan Wang, et al. Infinity-mm: Scaling multimodal performance with large-scale and high-quality instruction data. arXiv preprint arXiv:2410.18558, 2024.

### A. Related Work of Efficient VLMs

With the advent of visual instruction tuning [15, 119] and the scaling of large language models (LLMs) [120, 121, 122], both large-scale open-source [17, 18, 19, 12] and closed-source [14, 54, 55] vision–language models (VLMs) have emerged. However, these large-scale VLMs impose substantial computational demands in real-world scenarios [123, 124], such as on-device or edge processing. Consequently, there is a growing demand for lightweight VLMs that can be efficiently deployed on resource-constrained devices while maintaining fast inference, driving active research in efficient VLM design. Early efforts have mainly focused on integrating additional visual encoders [125, 126, 127, 128], multiple computer vision backbones [129, 130, 131, 132], or rational embeddings [133, 134, 135] into LLMs [136, 137, 138, 139, 140, 141]. In addition, a growing body of research [142, 143, 144, 145, 146] has explored architectural strategies—such as shared or repetitive feed-forward network (FFN) structures and expanded hidden dimensions—to enhance efficiency without significant performance degradation. Furthermore, several studies [147, 148, 21, 149, 150, 151] propose vision–text aligned training strategies, adopt Mamba architecture [152], or incorporate the mixture-of-experts paradigm [153, 154, 155, 156, 157] to achieve scalable model capacity.

### B. The Objective of GRPO

For a question 𝑥 and its multiple generated responses {𝑦^𝑗}𝐺𝑗=1, the RL objective of GRPO [61] (Generalized Reinforcement Policy Optimization) is defined as:

ℒGRPO = −E𝑗 [︀E𝑡 [︀

(︀𝑟𝑗,𝑡𝐴𝑗, clip(𝑟𝑗,𝑡,1 − 𝜀,1 + 𝜀)𝐴𝑗)︀ − 𝛽𝐷KL(𝜋𝜃‖𝜋ref)

]︀]︀, (9)

min

and 𝐴𝑗 = ℛ𝑗 − mean (︁{ℛ𝑗}𝐺𝑗=1)︁ std(︁{ℛ𝑗}𝐺𝑗=1)︁ . (10)

𝜋𝜃(^𝑦𝑗,𝑡 | 𝑥,𝑦^𝑗,<𝑡) 𝜋𝜃old(^𝑦𝑗,𝑡 | 𝑥,𝑦^𝑗,<𝑡)

where 𝑟𝑗,𝑡 =

Here, 𝑟𝑗,𝑡 denotes the policy ratio for new policy 𝜋𝜃 and old policy 𝜋𝜃old for each token 𝑡, and 𝐴𝑗 indicates the advantage computed by normalized rewards ℛ. This objective encourages the new

policy 𝜋𝜃 to improve upon the old policy 𝜋𝜃old according the advantage 𝐴. The clipped surrogate objective limits the policy update ratio 𝑟𝑗,𝑡 to the range [1 − 𝜀, 1 + 𝜀], preventing excessively large updates. In addition, KL divergence term 𝐷KL(𝜋𝜃‖𝜋ref) penalizes deviation from a reference policy 𝜋ref, ensuring regularization for stable training.

In our Masters training setup, the total reward is computed as the sum ℛ = ℛacc + ℛdistill, from which the advantage is directly derived. Since the updating model is the student, the policy 𝜋𝜃 corresponds to the student’s logit-softmax output 𝑃𝒮, and the parameter 𝜃 represents the student’s weight set W𝒮. In our setup, the policy ratio 𝑟𝑗,𝑡 is always one because the student is updated only once per training iteration 𝑖; hence, the old policy 𝜋𝜃old and the new policy 𝜋𝜃 are identical. Therefore, the clipped surrogate term becomes redundant, and the objective of GRPO simplifies to

ℒGRPO = −E𝑗 [𝑟𝑗,𝑡𝐴𝑗 − 𝛽𝐷KL(𝜋𝜃‖𝜋ref)], (11)

where 𝑟𝑗,𝑡 = 1. Technically, we still keep the ratio term in the expression to ensure the gradient properly flows to the student parameters during training. Additionally, we set 𝛽 = 0.1 to prevent the student from being updated excessively, providing stable regularization.

### C. Visual Instruction Tuning Data

We assemble a 1.5M-sample visual instruction tuning dataset that encompasses both real-world and synthetic sources: COCO-ReM [158], iNaturalist2018 [159], VQA-v2 [160], Super-CLEVR [161], MAVIS [162], Geometry3K [163], SQA [164], AI2D [1], SA-1B [127], LLaVAR [165], VSR [166], TallyQA [167], TabMWP [168], KonIQ [169], InternVL [170]-filtered synthetic knowledge dataset covering politics, math, physics, chemistry, RLAI-F [171], CLEVR-Math [172], SROIE [173], ChartQA [2], DocVQA [174], FigureQA [175], GQA [176], InfoVQA [177], M3CoT [178], MapQA [179], OKVQA [180], TextVQA [181], WildVision [182], DVQA [183], GeoQA+ [184], GeOS [185], IconQA [186], UniGEO [187], GeomVerse [188], Geo170K [189], MathV360K [190], and RAM++ [191]-filtered synthetic data of Infinity-MM [192] covering coarse and fine-grained perception, relation, attribute, and logic reasoning.

### D. Additional Parsing Prompts for Accuracy Reward

#### Prediction Evaluation Prompt

System: You are an evaluation assistant that gives accuracy scores compared with Ground Truth and Generated Text from AI. Question is in <question> </question> tag. Ground Truth is in <ground truth> </ground truth> tag. Generated Text in <generated text> </generated text> tag. After reading the Question, compare the Generated Text against the Ground Truth summary:

- - If the Generated Text fully and correctly captures the core point → 1
- - If it is incorrect or irrelevant → 0
- - If it has repetitive response → 0
- - If it has empty response → 0

Output the numerical evaluation score (0 or 1) after giving a brief explanation.

- The evaluation score should be wrapped in <answer> </answer> tag.

User: <question> {} </question>

<ground truth> {} </ground truth>

<generated text> {} </generated text>

Provide the numerical evaluation score after giving a brief explanation. The evaluation score should be wrapped in <answer> </answer> tag.

#### Accuracy Reward Parsing Prompt

System: You are an evaluation assistant that gives binary accuracy scores (0 or 1) based on the provided overall summary. The summary will be wrapped inside <overall_summary> and </overall_summary> tag. After reading the summary, briefly output the integer score (0 or 1) without any text. Your final output must include only the integer value.

User: <overall_summary> {} </overall_summary>

Please output your integer accuracy score (0 or 1) based on the summary above without any text.

