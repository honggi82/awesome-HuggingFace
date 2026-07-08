### VideoSSR: Video Self-Supervised Reinforcement Learning

Zefeng He1,2 Xiaoye Qu1* Yafu Li3 Siyuan Huang4 Daizong Liu5 Yu Cheng3* 1Shanghai Artificial Intelligence Laboratory, 2Nanjing Univerisity 3The Chinese University of Hong Kong 4Shanghai Jiao Tong University, 5Wuhan University

## arXiv:2511.06281v1[cs.CV]9Nov2025

#### Abstract

Reinforcement Learning with Verifiable Rewards (RLVR) has substantially advanced the video understanding capabilities of Multimodal Large Language Models (MLLMs). However, the rapid progress of MLLMs is outpacing the complexity of existing video datasets, while the manual annotation of new, high-quality data remains prohibitively expensive. This work investigates a pivotal question: Can the rich, intrinsic information within videos be harnessed to self-generate high-quality, verifiable training data? To investigate this, we introduce three self-supervised pretext tasks: Anomaly Grounding, Object Counting, and Temporal Jigsaw. We construct the Video Intrinsic Understanding Benchmark (VIUBench) to validate their difficulty, revealing that current state-of-the-art MLLMs struggle significantly on these tasks. Building upon these pretext tasks, we develop the VideoSSR-30K dataset and propose VideoSSR, a novel video self-supervised reinforcement learning framework for RLVR. Extensive experiments across 17 benchmarks, spanning four major video domains (General Video QA, Long Video QA, Temporal Grounding, and Complex Reasoning), demonstrate that VideoSSR consistently enhances model performance, yielding an average improvement of over 5%. These results establish VideoSSR as a potent foundational framework for developing more advanced video understanding in MLLMs. The code is available at https://github.com/lcqysl/VideoSSR.

#### 1. Introduction

In past years, Multimodal Large Language Models (MLLMs) have achieved remarkable progress in the field of video understanding [2, 3, 10, 34, 35, 39, 45, 51]. Benefiting from recent Reinforcement Learning with Verifiable Reward (RLVR) [12, 17, 27, 43, 44, 65, 68], the performance of MLLMs has been further improved. A cornerstone of the RLVR approach is the availability of video

*Corresponding authors

| |
|---|

| |
|---|

Figure 1. Distribution of answer correctness on ReWatch and LongVideoReason. Across both models and datasets, a vast majority of questions yield a bimodal outcome, resulting in either zero or eight correct answers. This zero variance issue is notably more pronounced for the more powerful Qwen3-VL model.

datasets with verifiable answers. To obtain the verifiable answers, existing datasets, such as LongVideoReason [9] and ReWatch [65], utilize multi-agent collaboration to construct high-quality datasets with verifiable answers.

Although current datasets have effectively enhanced the performance of models like Qwen2.5-VL [3], significant limitations arise when applying them to more powerful models, such as the recent Qwen3-VL [39]. First, for highly capable models, many questions in existing datasets lack sufficient complexity to serve as effective training challenges. To illustrate this, we generate eight independent responses per question using Qwen2.5-VL [3] and Qwen3VL [39]. As shown in Figure 1, a vast majority of questions yield a perfect score where all eight responses are correct, indicating they are insufficiently challenging. Second, the multi-agent annotation process introduces systemic biases and artifacts, which create flawed or spurious reward signals for RLVR, particularly when the annotator models are less capable than the target models. This is evidenced by another large portion of questions where all generated responses are incorrect, suggesting either intractable difficulty or biased ground truths. The resulting bimodal distribution of scores, with most questions exhibiting zero variance, offers an ineffective learning signal for GRPO [17, 43] training in RLVR. Consequently, training advanced models on

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Figure 2. Performance comparison on four video tasks. Input frames for VideoSSR and Qwen3-VL-8B do not exceed 64.

such data yields marginal gains or even performance degradation (Section 4.2).

Compounding these issues is the prohibitive cost of manual annotation for video. This predicament, however, points to a compelling alternative: can the rich, intrinsic information within videos be harnessed to construct highquality, verifiable questions for RLVR? Inspired by traditional video self-supervised learning [13, 32, 33, 58], we first design three self-supervised pretext tasks with parametrically scalable difficulty, including Anomaly Grounding, Object Counting, and Temporal Jigsaw, to generate verifiable questions. To validate the difficulty of these tasks, we construct Video Intrinsic Understanding Bench (VIUBench) and found that questions targeting the intrinsic properties of the video itself remain profoundly challenging, even for leading closed-source models like GPT-5 [35].

Building on this insight, we introduce VideoSSR, a new Video Self-Supervised Reinforcement learning framework to enhance the video understanding of MLLM. We construct the VideoSSR-30K dataset using the aforementioned pretext tasks, which is entirely independent of human or MLLM annotations. This dataset is subsequently utilized to train our model with GRPO. To overcome the challenge of sparse reward signals arising from the inherent difficulty of these tasks, we design corresponding smooth reward functions for each pretext task to ensure efficient and stable RLVR training.

To validate the generalization capability of VideoSSR, we conduct extensive experiments on 17 benchmarks spanning four main video tasks: General Video QA, Long Video QA, Temporal Grounding, and Complex Reasoning. The results show that our proposed VideoSSR achieves consistent performance improvements across all benchmarks and under three different input frame settings, demonstrating an average gain of over 5%.

In summary, our main contributions are fourfold:

- • We generate verifiable training data for RLVR that harnesses intrinsic video signals. This self-supervised paradigm circumvents the prohibitive costs and inherent biases of prevailing multi-agent and manual annotation, thereby addressing a critical bottleneck in scaling MLLMs for video understanding.
- • We introduce three self-supervised pretext tasks with parametrically scalable difficulty. Moreover, we construct VIUBench benchmark from these tasks, which reveals profound limitations in state-of-the-art MLLMs for intrinsic video understanding.
- • We introduce VideoSSR, a self-supervised reinforcement learning framework for video RLVR training and construct VideoSSR-30K dataset. To facilitate efficient and stable RLVR training, in VideoSSR, we further design three tailored smooth reward functions.
- • Extensive experiments across 17 benchmarks demonstrate the superior generalization capability of VideoSSR, consistently achieving average performance improvements exceeding 5% and establishing it as a foundational approach for advancing video understanding.

#### 2. Related Works

##### 2.1. Reinforcement Learning for MLLMs

Reinforcement Learning with Verifiable Reward (RLVR) [17, 43] has been shown to significantly enhance the reasoning capabilities of language models, a success that has been rapidly extended to MLLMs [8, 9, 11, 19, 27, 38, 60]. For instance, Video-R1 [12] leverages existing Video QA datasets [37, 54, 57, 62, 69] to bolster performance on Video QA tasks. Time-R1 [52] utilizes datasets with precise timestamp annotations to improve Temporal Grounding. SpaceR [36] automatically generates verifiable questions from the geometric and semantic ground truths of 3D

[Figure 1]

Figure 3. An overview of our three self-supervised pretext tasks. (a) Anomaly Grounding: A temporal segment is perturbed (e.g., via rotation), and the task is to identify the start and end timestamps of this anomaly. (b) Object Counting: Procedurally generated shapes are overlaid onto selected frames, and the task is to count the total number of each shape type. (c) Temporal Jigsaw: The video is divided into clips which are then shuffled. The task is to predict the original temporal order of the segments.

scenes [4, 61], enhancing the model’s spatial reasoning abilities. ReWatch-R1 [65] leverages multi-agent collaboration to construct high-quality reasoning datasets, thereby advancing its capabilities in complex reasoning. Despite these diverse data sourcing strategies, several fundamental limitations persist. The reliance on external annotations often introduces significant bias. Meanwhile, many approaches often specialize in enhancing a single capability, which can limit their broader generalization.

##### 2.2. Self-supervised learning for Video

Self-supervised learning [13, 24, 28, 29, 32, 33, 42, 49, 58] for video aims to learn effective spatio-temporal representations from unlabeled video data. The core principle involves designing pretext tasks that capitalize on the inherent properties of video. For instance, early works leverage tasks such as video jigsaw puzzles [1, 23, 32, 47, 58] to learn representations. Similarly, recent research [56, 64] has employed the jigsaw puzzle task to facilitate the reinforcement learning of MLLMs. While training MLLMs with the video jigsaw task has been shown to enhance performance on tasks requiring temporal-centric understanding [56], the self-supervised paradigm has not been fully explored for video understanding. In this work, we move beyond a single task and investigate a richer suite of pretext tasks to cultivate more comprehensive generalization in MLLMs.

#### 3. Method

Considering there is rich information in the video, in this paper, we explore leveraging the intrinsic information within the video itself to construct high-quality questions with scalable difficulty. To investigate it, we begin by designing novel pretext tasks.

##### 3.1. Pretext Tasks

In this section, we introduce three pretext tasks, including Anomaly Grounding, Object Counting, and Temporal Jigsaw. These tasks share a common design philosophy, namely, they can generate verifiable question-answer pairs directly from raw videos, independent of any human or model-generated annotations. Furthermore, the difficulty of these pairs can be parametrically controlled. The overall process for these three tasks is illustrated in Figure 3.

###### 3.1.1. Anomaly Grounding

This task assesses the model’s ability to localize temporal segments that violate natural video dynamics. Let a video be represented as a sequence of frames V = {f1,f2,...,fT}, with a total duration of D seconds. We first randomly select a temporal interval [ts,te] ⊆ [0,D], where ts and te are the start and end timestamps, respectively. This interval corresponds to a contiguous segment of

frames S = {fi | timestamp(fi) ∈ [ts,te]}.

Next, we apply a perturbation function P to this segment to create a perturbed version, S′ = P(S). The function P is sampled from a set of predefined transformations targeting different core capabilities:

- • Fine-grained Perception: e.g., swapping the red and blue color channels for every frame in S.
- • Spatial Perception: e.g., rotating every frame in S by 180 degrees.
- • Temporal Perception: e.g., randomly shuffling the frame order within S.

The final video V ′ is constructed by replacing the original segment S with its perturbed counterpart S′. The model is then provided with the modified video V ′ and is tasked to identify the anomalous interval by predicting its start and end timestamps, (ts,te).

###### 3.1.2. Object Counting

This task targets the model’s fine-grained perception and counting abilities. We define a set of primitive geometric shapes C = {c1,c2,...,cK}, such as circles, rectangles, and triangles, which can be procedurally generated. For a given video V , we randomly select a subset of frames Fsub ⊂ V . For each frame fi ∈ Fsub, we synthesize a set of objects Oi, where each object o ∈ Oi is an instance of a shape class from C with randomized attributes (size, color, rotation, position). These modified frames, denoted as fi′, are then used to create the final video V ′ by replacing their original counterparts. The ground truth is a vector of counts n = [N1,N2,...,NK], where each element Nk is the total number of occurrences of shape ck:

|{o ∈ Oi | type(o) = ck}| (1)

Nk =

fi∈Fsub

Given V ′, the model is required to output the counts for each shape category.

###### 3.1.3. Temporal Jigsaw

This task is designed to evaluate the model’s temporal perception, specifically its understanding of temporal coherence and event ordering. We partition the video V into n contiguous, non-overlapping segments of equal duration, V = [S1,S2,...,Sn]. We then generate a random permutation π of the indices {1,2,...,n}. A new video V ′ is created by reordering the segments according to this permutation:

###### V ′ = [Sπ(1),Sπ(2),...,Sπ(n)] (2)

The model is presented with the shuffled video V ′ and is tasked with restoring the original temporal order. To achieve this, it must predict a sequence of indices that correctly reorders the shuffled segments. This target sequence

is the inverse of the permutation π that was used for shuffling. The answer is therefore the sequence defined by π−1:

Answer = (π−1(1),π−1(2),...,π−1(n)) (3)

##### 3.2. Video Intrinsic Understanding Benchmark

After defining the above three tasks, a critical question arises: are these pretext tasks sufficiently challenging for state-of-the-art MLLMs? To investigate this, we construct the Video Intrinsic Understanding Bench (VIUBench), which systematically evaluates a model’s ability to comprehend intrinsic video properties across three core axes: Fine-grained Perception, Spatial Perception, and Temporal Perception. The benchmark is composed of 2700 questionanswer pairs generated from our three pretext tasks. The proportional distribution of data across these tasks is illustrated in the left panel of Figure 4.

Figure 4. Task distribution in VIUBench and VideoSSR-30K. The left panel illustrates the proportional data distribution across our three pretext tasks and their subtypes for VIUBench. The right panel shows the corresponding composition of VideoSSR-30K.

Anomaly Grounding. For the Anomaly Grounding task, we select five representative perturbation types from a larger pool of 14 (detailed in Appendix B.1). The selected types include: (1) swapping the red and blue color channels, (2) rotation by 180 degrees, (3) zooming out, (4) horizontal mirroring, and (5) shuffling the intra-segment frame order.

We compute the Mean Intersection over Union (mIoU) between the predicted and ground-truth temporal intervals as the performance score.

Object Counting. For the Object Counting task, we use three primitive shapes (circles, rectangles, and triangles) and configure two difficulty levels:

- • Easy: Objects are overlaid onto a maximum of three frames, with no more than three instances of any single shape type appearing in any given frame.
- • Hard: The constraints are increased to a maximum of four frames and up to four instances per shape per frame.

A score of 1 is awarded for a specific shape type if the predicted count is exactly equal to the ground-truth count, and

Table 1. Performance comparison on VIUBench. The benchmark assesses three core abilities (Fine-Grained Perception, Spatial Perception, and Temporal Perception) via our three pretext tasks (Object Counting, Anomaly Grounding, and Temporal Jigsaw). For both open source and closed source models, the top result is shown in bold, and the second-best is underlined.

Ability Fine-Grained Perception Spatial Perception Temporal Perception Average Task Object Counting Anomaly Grounding Temporal Jigsaw Counting Grounding Jigsaw Overall Type Easy Hard Channel Rotate ZoomOut Mirror Shuffle Easy Hard – – – –

Random Guess Random Guess 11.1 6.3 25.9 25.4 25.4 25.2 25.2 0.1 0.0 8.7 25.4 0.1 16.1 Closed Source Models

GPT-5 [35] 88.4 70.3 82.6 81.8 56.5 48.9 34.1 39.0 27.0 79.4 60.8 33.0 58.7 Gemini-2.5-Pro [10] 80.8 61.3 84.6 82.3 51.0 55.7 52.1 25.3 17.7 71.1 65.1 21.5 56.7 Gemini-2.5-Flash [10] 35.7 22.4 75.8 73.7 28.5 30.2 28.6 8.3 4.0 29.1 47.4 6.2 34.1 Seed1.5-VL [16] 72.3 52.0 79.0 70.7 19.4 31.4 24.1 20.7 9.3 62.6 44.9 15.0 42.2

Open Source Models

Qwen2.5-VL-7B-Instruct [3] 11.3 5.3 6.4 13.7 8.1 5.2 7.0 0.7 0.0 8.3 8.1 0.3 6.4 VideoJigsaw-7B [56] 12.1 5.4 1.5 4.5 1.2 1.1 2.0 20.3 5.0 8.8 2.1 12.7 5.9 Qwen3-VL-8B-Instruct [39] 13.8 7.7 50.1 53.4 21.4 13.6 14.1 1.3 0.0 10.7 30.5 0.7 19.5 Qwen3-VL-32B-Instruct [39] 20.1 13.0 66.1 63.0 17.6 29.4 18.9 1.3 0.0 16.6 39.0 0.7 25.5 Qwen3-VL-235B-A22B-Instruct [39] 23.8 14.8 68.9 67.8 32.9 28.3 26.8 7.7 3.3 19.3 45.0 5.5 30.5 GLM-4.5V [20] 59.1 45.4 66.4 61.0 21.2 29.8 15.3 11.0 3.3 52.3 38.7 7.2 34.7 InternVL-3.5-8B [51] 15.2 9.6 25.9 42.4 6.1 9.8 3.1 0.0 0.0 12.4 17.5 0.0 12.5 InternVL-3.5-38B [51] 28.0 15.9 47.0 54.5 9.6 18.2 12.4 0.0 0.0 21.9 28.3 0.0 20.6

VideoSSR-8B (Ours) 29.0 24.6 88.7 89.0 94.4 67.8 41.0 24.3 8.0 26.8 76.2 16.2 51.9

0 otherwise. The final task score is the average of these binary scores across all shape types.

Temporal Jigsaw. The Temporal Jigsaw task is configured with two difficulty settings based on the number of segments the video is partitioned into:

- • Easy: The video is partitioned into 6 segments.
- • Hard: The video is partitioned into 8 segments. A score is awarded only if the predicted sequence of segments is identical to the ground-truth permutation.

As shown in Table 1, we evaluate a suite of powerful MLLMs on VIUBench. Our findings reveal that this benchmark poses a significant challenge even for the most advanced models. Notably, even a strong closed-source model like GPT-5 [35] only achieves a modest average score of 58.7. The performance of open-source models is even more limited. For instance, Qwen3-VL-8B attains an average score of just 19.5. These results underscore a critical insight that understanding and reasoning about intrinsic video properties, such as fine-grained details and temporal coherence, remains a substantial bottleneck for current MLLMs. This highlights the effectiveness of VIUBench in exposing the limitations of existing models and validates its role as a challenging benchmark for future research.

More importantly, our experiments with VIUBench reveal a key advantage of these pretext tasks: the difficulty of the generated questions can be easily scaled by adjusting simple parameters. For instance, in the Object Counting task, switching from the “Easy” to the “Hard” configuration caused the score of GPT-5 to drop sharply from 88.4 to 70.3. A similar trend is observed in the Temporal Jigsaw task. By increasing the number of video segments from six to eight,

the model’s score plummeted from 39.0 to 27.0. Even for Video Jigsaw [56], a model specifically trained on jigsaw tasks, its performance decreases significantly from 20.3 to 5.0 under the same conditions. To sum up, all these findings demonstrate that our method can dynamically generate tasks that challenge powerful MLLMs. The ability to parametrically control task difficulty ensures that VIUBench can remain a relevant and challenging benchmark for evaluating the continuous advancements of future models.

##### 3.3. Video Self-Supervised Reinforcement Learning

Motivated by the insights from our proposed VIUBench, we introduce a novel framework that leverages Video Self-Supervised Reinforcement learning (VideoSSR) to enhance the generalization of MLLMs. To perform reinforcement learning, we first construct the VideoSSR-30K dataset. Specifically, this dataset consists of the aforementioned three pretext tasks. The proportional distribution of VideoSSR-30K dataset is detailed in the right panel of Figure 4.

For training, we employ RLVR using GRPO [17, 43]. We do not use recent variants of GRPO [22, 63, 71], as our primary focus is on the data itself.

Our reward function is based solely on answer correctness. While these tasks are designed to be difficult, this very characteristic poses a problem for RLVR training: using a strict reward function often results in sparse rewards, leading to inefficient and unstable training.

To address this challenge, we design a specific smooth reward function for each task to provide a denser and more informative learning signal.

###### Table 2. Performance comparison on General Video QA and Long Video QA tasks.

General Video QA Long Video QA

Model Frames MVBench TempCompass AoTBench VinoGround Video-MME LVBench LongVideoBench CGBench Closed Source Models

GPT-4o [34] - 64.6 73.8 63.4 54 71.9 30.8 62.0 45.2 Gemini-1.5-Pro [45] - 60.5 67.1 58.3 35.8 75.0 33.1 58.6 37.2

Open Source Models 32 66.9 74.9 59.8 45.2 64.1 39.6 58.6 40.1

Qwen3-VL-8B-Instruct [39] 48 67.6 74.8 60.2 45.0 66.0 41.5 60.0 41.7 64 67.8 74.8 60.7 45.0 66.4 43.0 61.3 42.6 32 68.6(+1.7) 75.7(+0.8) 60.5(+0.7) 55.6(+10.4) 65.2(+1.1) 41.5(+1.9) 59.6(+1.0) 40.2(+0.1)

VideoSSR-8B (Ours) 48 68.8(+1.2) 75.8(+1.0) 61.7(+1.5) 55.6(+10.6) 66.7(+0.7) 42.9(+1.4) 61.1(+1.1) 42.5(+0.8) 64 68.9(+1.1) 75.7(+0.9) 61.8(+1.1) 55.6(+10.6) 67.6(+1.2) 44.0(+1.0) 61.5(+0.2) 43.4(+0.8)

Anomaly Grounding. For the temporal grounding of anomalies, the Mean Intersection over Union (mIoU) naturally serves as a smooth reward signal. It provides a score between 0 and 1 that reflects the degree of overlap between the predicted and ground-truth temporal segments. Let Tpred and Tgt be the predicted and ground-truth intervals, respectively. The reward Rground is simply as below:

Rground = IoU(Tpred,Tgt) = |Tpred ∩ Tgt| |Tpred ∪ Tgt|

(4)

Object Counting. For the counting task, our reward function provides a dense signal based on the average relative error across all shape categories. For each category k, we first compute a score Rcount,k that is inversely proportional to the relative error. Let yk be the ground-truth count and yˆk be the predicted count for category k. The score for a single category is:

|yˆk − yk| yk + ε

(5)

Rcount,k = max 0,1 −

Here, the absolute error is normalized by the magnitude of the ground-truth value, and a small constant ε (e.g., 10−9) ensures numerical stability. The final reward for the entire task, Rcount, is the average of these scores over all K shape categories:

K

1 K

Rcount,k (6)

Rcount =

k=1

Temporal Jigsaw. For the jigsaw puzzle, our reward function measures the structural correctness of the predicted sequence. We compute a penalty based on the cumulative displacement of elements from their correct positions. Let Pgt be the ground-truth permutation and Pˆ be the predicted permutation. Let pos(v,P) denote the position of an element v in a sequence P. The total displacement error Ejigsaw is defined as:

n

|pos(k,Pˆ) − pos(k,Pgt)| (7)

Ejigsaw =

k=1

This error is then normalized by the maximum possible error, Emax, which occurs for a reversed sequence. The final reward is given by:

Ejigsaw Emax

Rjigsaw = 1 −

(8)

#### 4. Experiments

Implementation Details. In this paper, our VideoSSR model is built upon the Qwen3-VL-8B-Instruct [39]. We perform RLVR on our newly constructed VideoSSR-30K dataset for one epoch. Key hyperparameters for training include a learning rate of 1 × 10−6, a global batch size of 64, and a rollout number (N) of 8 for generation, a KL divergence penalty with a coefficient of 1 × 10−3. MAX FRAMES is configured to 48, and MAX PIXELS is set to 256 × 256 for efficient training. The entire training process is conducted on 8 H200 GPUs and takes approximately 16 hours. To ensure a fair and reproducible comparison, both Qwen3-VL and VideoSSR are evaluated under identical conditions: FPS is set to 2, with MAX FRAMES configured to {32, 48, 64}. MAX PIXELS is set to 512 × 512. Greedy decoding is used to ensure reproducibility. Chain of thought [53] is not utilized to mitigate hallucination [31] and ensure correct output formatting, therefore enhancing performance.

Benchmarks and Baselines. To comprehensively evaluate the generalization capability of VideoSSR, we conduct experiments on 16 distinct benchmarks spanning four major video task categories:

- • General Video QA: MVBench [26], TempCompass [30], AoTBench [59], and VinoGround [66].
- • Long Video QA: Video-MME [14], LVBench [50], LongVideoBench [55], and CGBench [6].
- • Temporal Grounding: QVHighlights [25], ActivityNet [5], CharadesSTA [15], and TACoS [40].

###### Table 3. Performance comparison on Temporal Grounding and Complex Reasoning tasks.

Temporal Grounding Complex Reasoning

Model Frames QVHighlights ActivityNet CharadesSTA TACoS VideoMMMU Video-TT VCRBench CVBench Closed Source Models

GPT-4o [34] - - - 35.7 - 61.2 45.2 29.0 69.2 Gemini-1.5-Pro [45] - - - - - 53.9 38.2 48.2 -

Open Source Models 32 43.7 36.5 50.3 22.4 58.2 41.8 7.4 61.8

Qwen3-VL-8B-Instruct [39] 48 46.4 38.4 50.0 25.9 58.5 43.0 7.4 61.5 64 48.6 39.8 49.2 28.1 58.8 44.0 8.8 61.6 32 59.6(+15.9) 42.1(+5.6) 52.1(+1.8) 23.1(+0.7) 59.9(+1.7) 44.2(+2.4) 10.7(+3.3) 63.5(+1.7)

VideoSSR-8B (Ours) 48 61.1(+14.7) 43.0(+4.6) 51.1(+1.1) 27.7(+1.8) 60.0(+1.5) 44.9(+1.9) 15.3(+7.9) 63.8(+2.3) 64 62.6(+14.0) 43.7(+3.9) 49.9(+0.7) 30.6(+2.5) 60.9(+2.1) 45.8(+1.8) 17.8(+9.0) 63.3(+1.7)

• Complex Reasoning: VideoMMMU [21], VideoTT [70], VCRBench [41], and CVBench [72].

For the Temporal Grounding tasks, we report the Mean Intersection over Union (mIoU) as the primary evaluation metric. Further details regarding each benchmark and a full breakdown of the results can be found in Appendix A.2.3.

For our primary baseline, we select Qwen3-VL-8BInstruct, as it represents the state-of-the-art among opensource models. To further contextualize the performance of our method, we also provide a comparative analysis against two formidable proprietary models: GPT-4o [34] and Gemini-1.5-Pro [45].

##### 4.1. Main Results

General Video QA As shown in the left half of Table 2, VideoSSR achieves substantial improvements on temporally related benchmarks such as VinoGround [66], even surpassing closed source models. It also obtains improvements on more general benchmarks, for instance on MVBench [26], achieving a score of 68.9 and similarly outperforming the closed source models.

Long Video QA As shown in the right half of Table 2, VideoSSR also achieves consistent improvements on four mainstream benchmarks. Because we primarily conduct training and evaluation with a low number of frames, a gap remains compared to closed-source models on such long video understanding tasks, which is a direction for future research.

Temporal Grounding As shown in the left side of Table 3, benefiting from the Anomaly Grounding task, VideoSSR achieves remarkable zero-shot improvements on multiple mainstream temporal grounding benchmarks, especially on QVHighlights [25] and ActivityNet [5], with gains of +15.9 and +5.6, respectively.

Complex Reasoning As shown in the right side of Table 3, VideoSSR achieves a large improvement of +9.0 on VCRBench [41], a benchmark that is highly correlated with our

Temporal Jigsaw task. It also obtains consistent improvements on other video reasoning benchmarks.

In summary, we validate the generalization capability of VideoSSR on the 16 aforementioned benchmarks. Notably, VideoSSR achieves consistent performance improvements across four major video tasks under three different frame settings. Under the 48 frame setting, VideoSSR obtains an average improvement of 5.1% across all 17 benchmarks (including VIUBench), comprehensively demonstrating the effectiveness of VideoSSR.

##### 4.2. Ablation Study

Analysis on three pretext tasks. First, we individually validate the effectiveness of the three pretext tasks, as shown in Table 4. Benefiting from its design, the Anomaly Grounding task leads to a significant performance increase on CharadesSTA. Similarly, the Temporal Jigsaw task brings a substantial boost to VCRBench. Notably, all three tasks individually improve performance on VideoMME, confirming their contribution to enhancing general video understanding capabilities.

Moreover, Table 4 also shows the impact of the smooth reward function on the results. We observe that the model trained with a strict matching reward function performs closer to the baseline. This is because a strict reward function often leads to sparse reward signals, which are more likely to result in a zero advantage in GRPO. Consequently, the training becomes inefficient, leading to smaller update magnitudes. Furthermore, this approach introduces training instability. For instance, training the anomaly grounding task with a strict reward function even degrades performance on CharadesSTA.

To further investigate the benefits of task diversity, we conducted a comparative analysis between single task training and our mixed task VideoSSR-30K framework, controlling for the data scale at 30k samples for both settings. As illustrated in Figure 5, we observe that simply scaling up the data for a single task yields diminishing returns and even

Table 4. Ablation study of the three pretext tasks and their corresponding smooth reward functions. G, C, and J represent Anomaly Grounding, Object Counting, and Temporal Jigsaw, respectively. ✓ indicates the component is used for training. R@0.5 denotes recall at an IoU threshold of 0.5. Step denotes step accuracy for VCRBench. The best and second best results are shown in bold and underlined.

|Training Config| |Understanding|Grounding<br><br>|Reasoning|
|---|---|---|---|---|
|Pretext Tasks|Reward|Video-MME<br><br>|CharadesSTA<br><br>|VCRBench|
|G C J<br><br>|Smooth<br><br>|All Long|mIoU R@0.5<br><br>|Acc Step|

Baseline Model ✗ ✗ ✗ – 64.1 54.3 50.3 58.4 7.4 25.9 Models on Subtasks

|✓ ✗ ✗ ✓ ✗ ✗<br><br>|✗ ✓|64.7 54.8<br><br>64.8 55.9<br><br><br>|47.5 52.2 53.8 63.8|5.8 24.9 4.1 22.8<br><br>|
|---|---|---|---|---|
|✗ ✓ ✗<br><br>✗ ✓ ✗<br><br><br>|✗ ✓|64.7 54.7 64.9 56.2<br><br>|51.5 59.9 51.4 60.1|6.3 25.4 5.5 24.8<br><br>|
|✗ ✗ ✓<br><br>✗ ✗ ✓<br><br><br>|✗ ✓<br><br>|64.3 55.0 64.8 55.8<br><br>|51.3 59.1 51.0 59.0|13.4 32.7 15.9 35.5<br><br>|

Models on All Tasks ✓ ✓ ✓ ✗ 64.8 55.4 51.3 59.3 10.7 30.4 ✓ ✓ ✓ ✓ 65.2 57.1 52.1 60.6 10.7 32.3

degrades performance. This finding suggests that designing a diverse and rich set of pretext tasks, rather than focusing on a single one, is a more promising direction for enhancing model capabilities.

| | |
|---|---|
| | |
| | |
| | |

Figure 5. Comparison of single task and mixed task training at the 30k data scale. The results demonstrates that task diversity is more effective for improving performance than simply scaling up the data for a single pretext task.

We also compare our method against training with LongVideoReason [9] or ReWatch [65]. We utilize only the multiple-choice subsets from each dataset. The specific training procedures are:

- • For LongVideoReason, the model is trained for 500 steps with a batch size of 64.
- • For ReWatch, we use a composite subset of questions from its Video-R1 [12] and VideoEspresso [18] portions and train the model for one full epoch.

The results are presented in Table 5. Notably, the model trained with VideoSSR-30K surpasses the performance of models trained on annotated datasets of a comparable scale. Furthermore, we observe a critical limitation: fine-tuning

the powerful Qwen3-VL on LongVideoReason, a dataset annotated by a less capable MLLM, can even lead to performance degradation, which further demonstrate the importance of our self-supervised paradigm.

Table 5. Ablation study on different training datasets. “None” indicates the baseline Qwen-VL3.

|Training Config<br><br>| |Video-MME|CharadesSTA<br><br>|VCRBench|
|---|---|---|---|---|
|Fine-tuning Data<br><br>|Size|All Long<br><br>|mIoU R@0.5|Acc Step|

Baseline Model None – 64.1 54.3 50.3 58.4 7.4 25.9 Fine-tuned Models

|LongVideoReason [9] ReWatch [65]|32k 27k<br><br>|63.6 53.3<br><br>64.7 56.7<br><br><br>|51.7 59.4 51.6 59.2|7.1 26.1 2.7 22.2<br><br>|
|---|---|---|---|---|
|VideoSSR-30K<br><br>|30K<br><br>|65.2 57.1|52.1 60.6<br><br>|10.7 32.3|

Finally, we explored the optimal selection of subtasks for the Anomaly Grounding task. We investigate 14 distinct perturbation types and report their corresponding accuracies on Video-MME, as illustrated in Figure 6. Further details are provided in Appendix B.1. Based on these results, we select four perturbations that offer substantial improvements and create a uniform mixture to construct our final training set. Furthermore, we found that perturbations targeting temporal properties, such as simulating a fast forward effect by sampling denser frame sequences, does not appear to yield benefits and even introduced negative side effects. This may be because the base model, Qwen3-VL, relies on textual timestamps for its temporal awareness. Deliberately creating visual anomalies in this domain might confuse the model rather than enhance its learning.

| |
|---|

| |
|---|

Figure 6. Ablation study of the 14 perturbation subtypes for Anomaly Grounding. Accuracy is reported on Video-MME.

#### 5. Conclusion

In this paper, we introduce VideoSSR, a novel selfsupervised reinforcement learning framework designed to address the critical limitations of existing video datasets for training MLLMs. By designing the three pretext tasks of Anomaly Grounding, Object Counting, and Temporal Jigsaw, we construct the challenging VIUBench and the VideoSSR-30K dataset without reliance on manual or MLLM annotations. Our extensive experiments demonstrate that VideoSSR leads to consistent and significant performance gains across 17 diverse benchmarks, including

four main video tasks, achieving an average improvement of over 5%. Our work highlights self-supervision as a powerful method for generating scalable, low-cost, and highquality training data. Crucially, the parametric control over task difficulty ensures the long-term relevance of our framework for benchmarking increasingly capable MLLMs. Our approach moves beyond the limitations of static, annotated datasets, enabling the development of models that learn directly from the intrinsic structure of video.

#### References

- [1] Unaiza Ahsan, Rishi Madhok, and Irfan Essa. Video jigsaw: Unsupervised learning of spatiotemporal context for video action recognition. In 2019 IEEE Winter Conference on Applications of Computer Vision (WACV), pages 179–189. IEEE, 2019. 3
- [2] Lei Bai, Zhongrui Cai, Yuhang Cao, Maosong Cao, Weihan Cao, Chiyu Chen, Haojiong Chen, Kai Chen, Pengcheng Chen, Ying Chen, et al. Intern-s1: A scientific multimodal foundation model. arXiv preprint arXiv:2508.15763, 2025. 1
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, Junyang Lin, et al. Qwen2.5-VL Technical Report, 2025. 1, 5
- [4] Ellis Brown, Arijit Ray, Ranjay Krishna, Ross Girshick, Rob Fergus, and Saining Xie. SIMS-V: Simulated instructiontuning for spatial video understanding. arXiv preprint arXiv:2511.04668, 2025. 3
- [5] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings of the ieee conference on computer vision and pattern recognition, pages 961–970, 2015. 6, 7
- [6] Guo Chen, Yicheng Liu, Yifei Huang, Yuping He, Baoqi Pei, Jilan Xu, Yali Wang, Tong Lu, and Limin Wang. Cgbench: Clue-grounded question answering benchmark for long video understanding, 2024. 6
- [7] Xinlong Chen, Yuanxing Zhang, Yushuo Guan, Bohan Zeng, Yang Shi, Sihan Yang, Pengfei Wan, Qiang Liu, Liang Wang, and Tieniu Tan. Versavid-r1: A versatile video understanding and reasoning model from question answering to captioning tasks. arXiv preprint arXiv:2506.09079, 2025. 13
- [8] Yi Chen, Yuying Ge, Rui Wang, Yixiao Ge, Junhao Cheng, Ying Shan, and Xihui Liu. Grpo-care: Consistency-aware reinforcement learning for multimodal reasoning. arXiv

- preprint arXiv:2506.16141, 2025. 2

[9] Yukang Chen, Wei Huang, Baifeng Shi, Qinghao Hu, Hanrong Ye, Ligeng Zhu, Zhijian Liu, Pavlo Molchanov, Jan Kautz, Xiaojuan Qi, et al. Scaling rl to long videos. arXiv

- preprint arXiv:2507.07966, 2025. 1, 2, 8

- [10] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5:

Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 1, 5

- [11] Lu Dong, Haiyu Zhang, Han Lin, Ziang Yan, Xiangyu Zeng, Hongjie Zhang, Yifei Huang, Yi Wang, Zhen-Hua Ling, Limin Wang, et al. Videotg-r1: Boosting video temporal grounding via curriculum reinforcement learning on reflected boundary annotations. arXiv preprint arXiv:2510.23397, 2025. 2
- [12] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776,

2025. 1, 2, 8

- [13] Basura Fernando, Hakan Bilen, Efstratios Gavves, and Stephen Gould. Self-supervised video representation learning with odd-one-out networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3636–3645, 2017. 2, 3
- [14] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24108–24118, 2025. 6
- [15] Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia. Tall: Temporal activity localization via language query. In Proceedings of the IEEE international conference on computer vision, pages 5267–5275, 2017. 6
- [16] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025. 5
- [17] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 1, 2, 5
- [18] Songhao Han, Wei Huang, Hairong Shi, Le Zhuo, Xiu Su, Shifeng Zhang, Xu Zhou, Xiaojuan Qi, Yue Liao, and Si Liu. Videoespresso: A large-scale chain-of-thought dataset for fine-grained video reasoning via core frame selection. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26181–26191, 2025. 8
- [19] Zefeng He, Xiaoye Qu, Yafu Li, Siyuan Huang, Daizong Liu, and Yu Cheng. Framethinker: Learning to think with long videos via multi-turn frame spotlighting. arXiv preprint arXiv:2509.24304, 2025. 2
- [20] Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. Glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv e-prints, pages arXiv–2507, 2025. 5
- [21] Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Yuanhan Zhang, Xiang Yue, Bo Li, and Ziwei Liu. Video-mmmu: Evaluating knowledge acquisition from multi-discipline professional videos. arXiv preprint arXiv:2501.13826, 2025. 7

- [22] Siyuan Huang, Xiaoye Qu, Yafu Li, Yun Luo, Zefeng He, Daizong Liu, and Yu Cheng. Spotlight on token perception for multimodal reinforcement learning. arXiv preprint arXiv:2510.09285, 2025. 5
- [23] Yuqi Huo, Mingyu Ding, Haoyu Lu, Zhiwu Lu, Tao Xiang, Ji-Rong Wen, Ziyuan Huang, Jianwen Jiang, Shiwei Zhang, Mingqian Tang, et al. Self-supervised video representation learning with constrained spatiotemporal jigsaw. 2021. 3
- [24] Longlong Jing and Yingli Tian. Self-supervised visual feature learning with deep neural networks: A survey. IEEE transactions on pattern analysis and machine intelligence, 43(11):4037–4058, 2020. 3
- [25] Jie Lei, Tamara L Berg, and Mohit Bansal. Detecting moments and highlights in videos via natural language queries. Advances in Neural Information Processing Systems, 34: 11846–11858, 2021. 6, 7
- [26] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195– 22206, 2024. 6, 7
- [27] Xinhao Li, Ziang Yan, Desen Meng, Lu Dong, Xiangyu Zeng, Yinan He, Yali Wang, Yu Qiao, Yi Wang, and Limin Wang. Videochat-r1: Enhancing spatio-temporal perception via reinforcement fine-tuning. arXiv preprint arXiv:2504.06958, 2025. 1, 2
- [28] Daizong Liu, Xiaoye Qu, Yinzhen Wang, Xing Di, Kai Zou, Yu Cheng, Zichuan Xu, and Pan Zhou. Unsupervised temporal video grounding with deep semantic clustering. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1683–1691, 2022. 3
- [29] Daizong Liu, Xiang Fang, Xiaoye Qu, Jianfeng Dong, He Yan, Yang Yang, Pan Zhou, and Yu Cheng. Unsupervised domain adaptative temporal sentence localization with mutual information maximization. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 3567–3575, 2024. 3
- [30] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? arXiv preprint arXiv:2403.00476, 2024. 6
- [31] Mi Luo, Zihui Xue, Alex Dimakis, and Kristen Grauman. When thinking drifts: Evidential grounding for robust video reasoning. arXiv preprint arXiv:2510.06077, 2025. 6, 12
- [32] Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In European conference on computer vision, pages 69–84. Springer,

2016. 2, 3

- [33] Mehdi Noroozi, Hamed Pirsiavash, and Paolo Favaro. Representation learning by learning to count. In Proceedings of the IEEE international conference on computer vision, pages 5898–5906, 2017. 2, 3
- [34] OpenAI. Hello GPT-4o. OpenAI Blog, 2024. Accessed: 2024-05-14. 1, 6, 7
- [35] OpenAI. Chatgpt, 2025. 1, 2, 5
- [36] Kun Ouyang, Yuanxin Liu, Haoning Wu, Yi Liu, Hao Zhou, Jie Zhou, Fandong Meng, and Xu Sun. Spacer: Rein-

- forcing mllms in video spatial reasoning. arXiv preprint arXiv:2504.01805, 2025. 2
- [37] Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Mateusz Malinowski, Yi Yang, Carl Doersch, et al. Perception test: A diagnostic benchmark for multimodal video models. Advances in Neural Information Processing Systems, 36:42748–42761, 2023. 2
- [38] Xiaoye Qu, Pengwei Tang, Zhikang Zou, Yu Cheng, Jianfeng Dong, Pan Zhou, and Zichuan Xu. Fine-grained iterative attention network for temporal language localization in videos. In Proceedings of the 28th ACM International Conference on Multimedia, pages 4280–4288, 2020. 2
- [39] Qwen. Qwen3-vl, 2025. 1, 5, 6, 7, 13
- [40] Michaela Regneri, Marcus Rohrbach, Dominikus Wetzel, Stefan Thater, Bernt Schiele, and Manfred Pinkal. Grounding action descriptions in videos. Transactions of the Association for Computational Linguistics, 1:25–36, 2013. 6
- [41] Pritam Sarkar and Ali Etemad. Vcrbench: Exploring longform causal reasoning capabilities of large video language models. arXiv preprint arXiv:2505.08455, 2025. 7
- [42] Madeline C Schiappa, Yogesh S Rawat, and Mubarak Shah. Self-supervised learning for videos: A survey. ACM Computing Surveys, 55(13s):1–37, 2023. 3
- [43] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 1, 2, 5
- [44] Yunlong Tang, Jing Bi, Pinxin Liu, Zhenyu Pan, Zhangyun Tan, Qianxiang Shen, Jiani Liu, Hang Hua, Junjia Guo, Yunzhong Xiao, et al. Video-lmm post-training: A deep dive into video reasoning with large multimodal models. arXiv preprint arXiv:2510.05034, 2025. 1
- [45] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 1, 6, 7
- [46] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. Advances in neural information processing systems, 35:10078–10093, 2022. 13
- [47] Guodong Wang, Yunhong Wang, Jie Qin, Dongming Zhang, Xiuguo Bao, and Di Huang. Video anomaly detection by solving decoupled spatio-temporal jigsaw puzzles. In European Conference on Computer Vision, pages 494–511. Springer, 2022. 3
- [48] Haonan Wang, Hongfu Liu, Xiangyan Liu, Chao Du, Kenji Kawaguchi, Ye Wang, and Tianyu Pang. Fostering video reasoning via next-event prediction. arXiv preprint arXiv:2505.22457, 2025. 13
- [49] Jiangliu Wang, Jianbo Jiao, and Yun-Hui Liu. Selfsupervised video representation learning by pace prediction. In European conference on computer vision, pages 504–521. Springer, 2020. 3, 13
- [50] Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Xiaotao Gu, Shiyu Huang, Bin Xu, Yuxiao

- Dong, et al. Lvbench: An extreme long video understanding benchmark. arXiv preprint arXiv:2406.08035, 2024. 6
- [51] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. 1, 5
- [52] Ye Wang, Ziheng Wang, Boshen Xu, Yang Du, Kejun Lin, Zihan Xiao, Zihao Yue, Jianzhong Ju, Liang Zhang, Dingyi Yang, et al. Time-r1: Post-training large vision language model for temporal video grounding. arXiv preprint arXiv:2503.13377, 2025. 2
- [53] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. 6, 12
- [54] Bo Wu, Shoubin Yu, Zhenfang Chen, Joshua B Tenenbaum, and Chuang Gan. Star: A benchmark for situated reasoning in real-world videos. arXiv preprint arXiv:2405.09711,

2024. 2

- [55] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. Advances in Neural Information Processing Systems, 37:28828–28857, 2024. 6
- [56] Penghao Wu, Yushan Zhang, Haiwen Diao, Bo Li, Lewei Lu, and Ziwei Liu. Visual jigsaw post-training improves mllms. arXiv preprint arXiv:2509.25190, 2025. 3, 5
- [57] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9777–9786, 2021. 2
- [58] Dejing Xu, Jun Xiao, Zhou Zhao, Jian Shao, Di Xie, and Yueting Zhuang. Self-supervised spatiotemporal learning via video clip order prediction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10334–10343, 2019. 2, 3
- [59] Zihui Xue, Mi Luo, and Kristen Grauman. Seeing the arrow of time in large multimodal models. arXiv preprint arXiv:2506.03340, 2025. 6
- [60] Ziang Yan, Xinhao Li, Yinan He, Zhengrong Yue, Xiangyu Zeng, Yali Wang, Yu Qiao, Limin Wang, and Yi Wang. Videochat-r1. 5: Visual test-time scaling to reinforce multimodal reasoning by iterative perception. arXiv preprint arXiv:2509.21100, 2025. 2
- [61] Shusheng Yang, Jihan Yang, Pinzhi Huang, Ellis Brown, Zihao Yang, Yue Yu, Shengbang Tong, Zihan Zheng, Yifan Xu, Muhan Wang, Danhao Lu, Rob Fergus, Yann LeCun, Li FeiFei, and Saining Xie. Cambrian-s: Towards spatial supersensing in video. arXiv preprint arXiv:2511.04670, 2025. 3, 13
- [62] Kexin Yi, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Jiajun Wu, Antonio Torralba, and Joshua B Tenenbaum. Clevrer: Collision events for video representation and reasoning. arXiv preprint arXiv:1910.01442, 2019. 2

- [63] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025. 5
- [64] Yu Zeng, Wenxuan Huang, Shiting Huang, Xikun Bao, Yukun Qi, Yiming Zhao, Qiuchen Wang, Lin Chen, Zehui Chen, Huaian Chen, et al. Agentic jigsaw interaction learning for enhancing visual perception and reasoning in visionlanguage models. arXiv preprint arXiv:2510.01304, 2025. 3
- [65] Congzhi Zhang, Zhibin Wang, Yinchao Ma, Jiawei Peng, Yihan Wang, Qiang Zhou, Jun Song, and Bo Zheng. Rewatchr1: Boosting complex video reasoning in large visionlanguage models through agentic data synthesis. arXiv preprint arXiv:2509.23652, 2025. 1, 3, 8
- [66] Jianrui Zhang, Mu Cai, and Yong Jae Lee. Vinoground: Scrutinizing lmms over dense temporal reasoning with short videos. arXiv preprint arXiv:2410.02763, 2024. 6, 7
- [67] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Lmmseval: Reality check on the evaluation of large multimodal models, 2024. 12
- [68] Kaiyan Zhang, Yuxin Zuo, Bingxiang He, Youbang Sun, Runze Liu, Che Jiang, Yuchen Fan, Kai Tian, Guoli Jia, Pengfei Li, et al. A survey of reinforcement learning for large reasoning models. arXiv preprint arXiv:2509.08827,

2025. 1

- [69] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 2, 12
- [70] Yuanhan Zhang, Yunice Chew, Yuhao Dong, Aria Leo, Bo Hu, and Ziwei Liu. Towards video thinking test: A holistic benchmark for advanced video reasoning and understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20626–20636, 2025. 7
- [71] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025. 5
- [72] Nannan Zhu, Yonghao Dong, Teng Wang, Xueqian Li, Shengjun Deng, Yijia Wang, Zheng Hong, Tiantian Geng, Guo Niu, Hanyan Huang, et al. Cvbench: Evaluating crossvideo synergies for complex multimodal understanding and reasoning. arXiv preprint arXiv:2508.19542, 2025. 7

# Supplementary Materials

#### A. Implementation Details

##### A.1. Training Details

We utilize Llava-Video [69] as the primary video source for constructing both VideoSSR-30K dataset and VIUBench. During training, we did not employ chain of thought [53] for VideoSSR or any models in the ablation studies. This decision aligns with our focus on enhancing fundamental perceptual abilities, namely, Fine-Grained, Spatial, and Temporal Perception, rather than complex reasoning. This approach also yields greater training efficiency and reduces the potential for model hallucination [31].

##### A.2. Evaluation Details for VideoSSR

###### A.2.1. Prompts

For Video QA tasks, we prompt the model to generate a direct answer. The specific prompt template utilized for these tasks is illustrated in Figure 7.

For Temporal Grounding tasks, our prompt format is based on the one utilized in the lmms eval library [67], as depicted in Figure 8. While we observed that CharadesSTA seems to be particularly sensitive to prompt phrasing, we nonetheless applied this unified prompt across all benchmarks to ensure a fair and consistent evaluation.

For other specialized benchmarks, such as VCRBench, we adhere to the official prompts.

[Figure 2]

Figure 7. Prompt template for Video QA tasks

[Figure 3]

Figure 8. Prompt template for Temporal Grounding tasks

###### A.2.2. Benchmarks

We adhered to specific evaluation protocols for several benchmarks to ensure fair and accurate assessment.

- • VinoGround: We report the text score, which offers greater discriminative power between models.
- • Video-MME & LongVideoBench: For both benchmarks, evaluations are conducted without the use of subtitles. For LongVideoBench, we specifically test on its validation set.
- • CGBench: Our evaluation is performed on its 3k subset.
- • Temporal Grounding: For benehmarks in this category, the model is required to predict a single most likely temporal interval. Results for QVHighlights and ActivityNet are reported on their validation sets.
- • VideoMMMU & Video-TT: We report results on the multiple choice subset to facilitate answer extraction and comparison.
- • CVBench: Our evaluation uses configurations of 32, 48, and 64 frames for each video, resulting in a significantly larger total number of frames processed per query.

###### A.2.3. Detailed Results

For Temporal Grounding tasks, we provide a more detailed breakdown of the results, as detailed in Table 6 and Table 7.

##### A.3. Evaluation Details for VIUBench

All evaluations on VIUBench utilized a fixed input of 48 frames with a maximum resolution of 512 × 512 pixels.

#### B. Details of Pretext Tasks

##### B.1. Anomaly Grounding

Figure 9 illustrates the prompt template used for the Anomaly Grounding task. Table 8 provides the comprehensive list and definitions for all 14 perturbation subtypes designed for this task. The text in the “Description” column of the table is what replaces the {description} placeholder in the prompt for each respective subtype.

Notably, for perturbations targeting Temporal Perception (specifically Slow and Fast), we provided an expanded and highly detailed description within the prompt. This special note, as detailed at the bottom of Table 8, explicitly instructed the model to disregard the evenly spaced frame timestamps and instead rely solely on visual motion cues.

Despite this explicit guidance, the model’s performance on these tasks remained notably poor, as shown in Figure 6. We hypothesize that this is because the base model, Qwen3VL, has a strong inherent bias towards relying on textual timestamp information when it is available. Forcing the model to overcome this bias and learn true visual motion

###### Table 6. More results on QVHighlights and ActivityNet.

QVHighlights ActivityNet Model Frames mIoU R@0.3 R@0.5 R@0.7 mIoU R@0.3 R@0.5 R@0.7

32 43.7 62.3 42.5 24.2 36.5 52.3 34.5 18.3

Qwen3-VL-8B-Instruct [39] 48 46.4 64.4 46.5 30.3 38.4 54.3 36.4 21.0 64 48.6 64.5 48.6 33.9 39.8 55.6 38.6 23.0 32 59.6(+15.9) 83.3(+21.0) 66.0(+23.5) 43.4(+19.2) 42.1(+5.6) 63.0(+10.7) 41.4(+6.9) 21.5(+3.2)

VideoSSR-8B (Ours) 48 61.1(+14.7) 83.5(+19.1) 66.9(+20.4) 48.3(+18.0) 43.0(+4.6) 63.2(+8.9) 42.3(+5.9) 22.7(+1.7) 64 62.6(+14.0) 83.7(+19.2) 68.0(+19.4) 49.7(+15.8) 43.7(+3.9) 63.3(+7.7) 42.7(+4.1) 24.2(+1.2)

###### Table 7. More results on CharadesSTA and Tacos.

CharadesSTA Tacos Model Frames mIoU R@0.3 R@0.5 R@0.7 mIoU R@0.3 R@0.5 R@0.7

32 50.3 76.5 58.1 27.9 22.4 34.7 19.2 7.1

Qwen3-VL-8B-Instruct [39] 48 50.0 76.6 56.1 26.9 25.9 39.0 24.0 10.7 64 49.2 77.1 54.2 25.5 28.1 42.0 26.6 12.3 32 52.1(+1.8) 78.2(+1.7) 60.6(+2.5) 30.8(+2.9) 23.1(+0.7) 34.1(-0.6) 19.8(+0.6) 7.4(+0.3)

VideoSSR-8B (Ours) 48 51.1(+1.1) 79.0(+2.4) 59.9(+3.8) 27.5(+0.6) 27.7(+1.8) 40.0(+1.0) 24.7(+0.7) 12.3(+1.6) 64 49.9(+0.7) 78.7(+1.6) 57.6(+3.4) 24.2(-1.3) 30.6(+2.5) 43.8(+1.8) 28.1(+1.5) 14.4(+2.1)

perception appears to be a significant challenge, even with detailed and explicit prompting.

[Figure 4]

Figure 9. Prompt template for Anomaly Grounding.

Figures 12 through 15 illustrate several concrete examples of the Anomaly Grounding task, corresponding to four different perturbation types. For clarity, only a subset of key frames from each video is displayed. The model’s objective is to predict the temporal range of the introduced anomaly based on the visual evidence.

##### B.2. Object Counting

- Figure 10 illustrates the prompt template used for the Object Counting task. Concrete visual examples of this task are provided in Figure 16 and Figure 17. B.3. Temporal Jigsaw
- Figure 11 shows the prompt template for the Temporal Jigsaw task. Figure 18 provides a concrete visual example of the shuffled video sequence that is presented to the model. For a clearer understanding of the task and to provide a direct comparison, the corresponding original video with the clips in their correct temporal order is also shown in Figure 19.

[Figure 5]

Figure 10. Prompt template for Object Counting.

##### B.4. Exploration of Alternative Pretext Tasks

In addition to the three pretext tasks detailed in the main paper, we also investigated other self-supervised learning paradigms. Our exploration included generative modeling approaches, such as masked [7, 46] frame reconstruction and autoregressive [48, 61] next frame prediction. Furthermore, we experimented with a task focused on direct temporal speed prediction [49].

However, our preliminary experiments indicated that these alternative tasks did not yield significant or consistent performance improvements on our downstream evaluation benchmarks. This suggests that while these methods are powerful, their objectives may not be as directly aligned with cultivating the high level perceptual and reasoning skills targeted by our final task selection. The discovery of an even broader range of effective self-supervised

Table 8. Definitions of the 14 Perturbation Subtypes for Anomaly Grounding. For temporal perception tasks, an additional detailed note (marked with *) was provided to guide the model.

Category Perturbation Type Description Fine-Grained Perception

Saturation the colors in the video become oversaturated and unnaturally vibrant. Noise Gaussian noise is added to the video. Blur the video becomes blurry or out of focus. Grayscale the video becomes black and white. Invert the colors in the video are inverted. Channel Swap the red and blue color channels in the video are swapped.

###### Spatial Perception

Zoom In the video is zoomed in. Rotate the video is rotated 180 degrees. Zoom Out the video is zoomed out. Mirror The video is mirrored horizontally.

###### Temporal Perception

Slow the video slows down, this means the action unfolds at an unusually slow pace, making movements appear prolonged.* Fast the video speeds up, this means the segment plays at a high speed, compressing the action and making movements appear jerky or rushed.* StutterHold the video appears to freeze and stutter on a few frames, this means instead of playing smoothly, the video repeatedly freezes on a single frame before jumping to the next. Shuffle the frames are shuffled, this means the order of events is scrambled, making the action appear illogical and chaotic.

*Special Note for Slow/Fast perturbations: To ensure a fair challenge, even if the video’s actual speed changes (e.g., slow motion or fast forward), the timestamps for each frame have been intentionally kept evenly spaced. This creates the illusion of a constant playback speed. Therefore, you should not rely on the timestamps when judging the speed. Instead, your judgment must be based solely on the visual content. You should analyze the motion within the video itself by observing how much or how little the scene changes between consecutive frames to determine the true playback speed.

[Figure 6]

Figure 11. Prompt template for Temporal Jigsaw.

tasks for enhancing MLLMs remains a promising direction for future work.

#### C. Limitations and Future Work

While our work demonstrates the significant potential of VideoSSR, we also recognize several limitations that present clear opportunities for future research.

First, our experiments were primarily conducted using a low number of input frames for both training and evaluation. This decision was driven by considerations of computational efficiency, allowing for both rapid iteration on more pretext tasks and comprehensive coverage of evaluation benchmarks. However, this approach may limit the model’s scalability to long videos. A key direction for future work is to scale the VideoSSR framework to handle higher frame rates and longer video inputs. This will be crucial for enhancing the model’s capabilities on complex, long-form content where dense temporal information is paramount.

Second, our framework relies on only three pretext tasks. While effective, this approach overlooks both the potential of a broader range of self-supervised objectives and the pos-

sible synergies that could be unlocked with more sophisticated mixing strategies. Future work could therefore explore a richer suite of pretext tasks and investigate advanced mixing techniques like curriculum learning or adaptive task weighting to further enhance model generalization.

[Figure 7]

Figure 12. An example of Channel Swap. The ground truth is 6.9s–9.2s.

[Figure 8]

- Figure 13. An example of Rotate. The ground truth is 5.1s–11.7s.

[Figure 9]

- Figure 14. An example of ZoomOut. The ground truth is 5.4s–6.9s.

[Figure 10]

###### Figure 15. An example of Mirror. The ground truth is 14.1s–22.6s.

[Figure 11]

###### Figure 16. An example of Object Counting. The ground truth (circles, squares, and triangles) is 3,2,3.

[Figure 12]

###### Figure 17. An example of Object Counting. The ground truth (circles, squares, and triangles) is 1,3,1.

[Figure 13]

###### Figure 18. An example of Temporal Jigsaw. The ground truth is 452316. The corresponding unshuffled video is shown in Figure 19.

[Figure 14]

###### Figure 19. The original video corresponding to the example in Figure 18.

