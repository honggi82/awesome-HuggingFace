# arXiv:2503.21776v4[cs.CV]22Oct2025

## Video-R1: Reinforcing Video Reasoning in MLLMs

Kaituo Feng1, Kaixiong Gong1, Bohao Li2, Zonghao Guo3∗, Yibing Wang4, Tianshuo Peng1, Junfei Wu4, Xiaoying Zhang5, Benyou Wang2, Xiangyu Yue1∗

1CUHK MMLab, 2CUHK (SZ), 3Tsinghua University, 4UCAS, 5CUHK HCCL https://github.com/tulerfeng/Video-R1

### Abstract

Inspired by DeepSeek-R1’s success in eliciting reasoning abilities through rulebased reinforcement learning (RL), we introduce Video-R1 as the first attempt to systematically explore the R1 paradigm for incentivizing video reasoning within multimodal large language models (MLLMs). However, directly applying RL training with the GRPO algorithm to video reasoning presents two primary challenges: (i) a lack of temporal modeling for video reasoning, and (ii) the scarcity of highquality video-reasoning data. To address these issues, we first propose the T-GRPO algorithm, which encourages models to utilize temporal information in videos for reasoning. Additionally, instead of relying solely on video data, we incorporate high-quality image-reasoning data into the training process. We have constructed two datasets: Video-R1-CoT-165k for SFT cold start and Video-R1-260k for RL training, both comprising image and video data. Experimental results demonstrate that Video-R1 achieves significant improvements on video reasoning benchmarks such as VideoMMMU and VSI-Bench, as well as on general video benchmarks including MVBench and TempCompass, etc. Notably, Video-R1-7B attains a 37.1% accuracy on video spatial reasoning benchmark VSI-bench, surpassing the commercial proprietary model GPT-4o. All code, models, and data are released in https://github.com/tulerfeng/Video-R1.

### 1 Introduction

Recent advancements in rule-based Reinforcement Learning (RL) [17] have significantly enhanced the reasoning capabilities of Large Language Models (LLMs) [16, 11]. In particular, DeepSeek-R1 [11] has demonstrated that carefully designed RL pipelines can lead to emergent and robust reasoning abilities with long chain-of-thoughts (CoT) in text-based domains. Motivated by this success, several recent efforts have explored extending RL training to Multimodal Large Language Models (MLLMs) [33, 14, 39]. Notable examples include Kimi k1.5 [33] and Skywork R1V [39], which apply RL to improve reasoning over image-text pairs. However, despite these early explorations, the domain of video reasoning in MLLMs remains underexplored.

To bridge this gap, we present Video-R1 model, as the first attempt to systematically investigate eliciting strong video reasoning based on the R1 paradigm . However, directly applying RL training with the Group Relative Policy Optimization (GRPO) algorithm [30] to video reasoning introduces two fundamental challenges: First, original GRPO lacks explicit reward signals for encouraging temporal reasoning in a video. Without explicit temporal awareness, the model may take shortcuts for reasoning, focusing on a single frame or snapshot rather than reasoning over time (see Figure 1 for example; Video-UTR [40] also identifies a similar issue). The underexploitation for temporal cues can cause the learned reasoning strategies to "shortcut" the process—relying on superficial visual patterns, rather than engaging in deeper and temporally grounded reasoning. This could ultimately hindering generalization to more complex or diverse video reasoning tasks.

∗Corresponding Authors

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

Video:

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Question: Are there any cyan cubes that enter the scene? A. not sure B.no C.yes GRPO: <think>Let me think about this carefully……The question specifically asks about cyan cubes. I see a cyan cube in the scene (shortcut)……it was already present. Hmm, this is a bit tricky. The question is asking if any cyan cubes enter the scene, but the cyan cube was already there……. break it down one more time to be sure: No new cyan cubes enter the scene……</think>\n<answer>B</answer> T-GRPO: <think>Let me think about this carefully……Now, let's focus on the cyan cube ……it is not there at the beginning of the scene. It appears to be moving into the scene from the bottom right corner. This indicates that the cyan cube is indeed entering the scene. Hmm, it seems that the cyan cube is a new object that appears in the scene. It is not present in the initial state……but it is clearly visible in the final state…… This confirms that the cyan cube enters the scene……After this thorough analysis, I am confident that the answer is yes......</think>\n<answer>C</answer>

Figure 1: Reasoning paths of Video-R1 trained by GRPO and our proposed T-GRPO on test samples. Without explicit temporal modeling, models may learn sub-optimal video reasoning patterns by taking shortcuts, therefore failing to generalize well.

The second issue lies in the scarcity of high-quality video reasoning training data, especially samples that demand strong reasoning ability or involve long reasoning path. Most existing video training datasets mainly focus on simple recognition tasks, rather than reasoning. This scarcity makes it difficult to expose the model to diverse, challenging reasoning patterns during training, limiting the effectiveness of RL and hindering the emergence of robust reasoning behaviors.

To address these challenges, we propose two key solutions. First, we propose T-GRPO, an extension of the original GRPO algorithm [30] that explicitly encourages temporal reasoning. During training, the model is presented with both temporally ordered and randomly shuffled frame sequences, producing two groups of responses. A positive reward is assigned only when the proportion of correct answers from the ordered group exceeds that from the shuffled one. This strategy encourages the model to exploit temporal reasoning policy rather than relying on shortcuts derived from isolated frames.

Besides, to tackle the scarcity of high-quality video reasoning data, we strategically introduce imagebased reasoning data as part of training data. We construct two datasets: Video-R1-CoT-165k for SFT cold start and Video-R1-260k for RL training. The image data serves as a valuable foundation for training general reasoning skills, while the curated video samples provide the temporal complexity needed for video understanding. This hybrid training setup not only alleviates the data bottleneck but also enables the model to transfer reasoning skills learned from static images to dynamic video contexts. Combined with T-GRPO, this approach equips Video-R1 with stronger, more generalizable video reasoning capabilities.

Our experiments show that Video-R1 achieves consistent and significant improvements across a suite of challenging video reasoning benchmarks, including VSI-Bench [38], VideoMMMU [13], MMVU [48], MVBench [20], TempCompass [27], and VideoMME [9]. Notably, Video-R1-7B attains 37.1% accuracy on VSI-Bench, a challenging video spatial reasoning benchmark, outperforming even proprietary models like GPT-4o [15]. These results suggest that with carefully designed algorithms and data pipelines, RL can indeed unlock complex temporal reasoning capabilities in MLLMs, similar to the breakthroughs seen in the text domain. Our contributions can be summaried as follows:

- • We propose Video-R1, as the first attempt to systematically to explore developing video reasoning MLLMs based on the R1 paradigm. To support training, we construct two reasoning datasets: Video-R1-CoT-165k for SFT and Video-R1-260k for RL training, incorporating both image and video reasoning samples. We hope that Video-R1 will serve as a foundation for future research on video reasoning.
- • To address the lack of temporal modeling in existing RL methods, we introduce T-GRPO, a novel training algorithm that encourages the model to utilize temporal information by contrasting reasoning performance over ordered and shuffled video frames.
- • Extensive experiments on multiple video benchmarks, such as VideoMMMU, VSI-Bench, MVBench, etc, demonstrate the effectiveness of our approach. Notably, Video-R1-7B achieves 37.1% accuracy on VSI-Bench, outperforming the proprietary GPT-4o model.

### 2 Related Works

- 2.1 Multimodal Large Language Models for Video

Video understanding is an essential capability for Multimodal Large Language Models (MLLMs), enabling them to interpret and reason over dynamic visual content [31, 4, 37, 44, 46, 32, 2]. In recent years, a number of MLLMs have been developed specifically to advance progress in video understanding tasks. For example, LLaMA-VID [22] proposes a dual-token strategy (context and content tokens) to compress video input representations, enabling vision-language models to efficiently handle long videos while retaining essential visual information. VideoLLaMA2 [4] enhances video-language modeling by introducing spatial-temporal convolution for better dynamic understanding and an audio branch to integrate multimodal cues for richer video comprehension. LongVA [43] extends the context window of language backbones to process significantly longer video sequences without specialized video training, offering a language-centric solution to long-range temporal reasoning. VISA [37] introduces a knowledge-driven video object segmentation task that combines world knowledge with object tracking, addressing implicit, complex video queries through a segmentation-enabled multimodal LLM. These advancements highlight the potential of MLLMs in advancing video understanding. However, most prior works have primarily focused on video perception tasks. The development of MLLMs with strong video reasoning capabilities remains largely unexplored.

- 2.2 Large Language Model Reasoning

The reasoning abilities of Large Language Models (LLMs) have been a focal point of recent research, aiming to enhance their capacity to perform complex, multi-step problem-solving tasks [34, 47, 42, 49, 23, 8, 35, 41, 24]. Unlike earlier approaches that rely on dense, step-level supervision or learned reward models to supervise reasoning paths [10, 21], DeepSeek-R1 initiates a new wave of interest in rule-based reinforcement learning, demonstrating that even coarse, outcome-only rewards can effectively elicit strong reasoning behavior [11]. Its success shows that with a carefully designed reward structure and policy optimization strategy, models could learn to generate long CoT without requiring intermediate supervision. Following this paradigm, several recent efforts have attempted to reproduce R1’s success [14, 33, 36, 3, 28, 18, 6, 45]. For example, Open Reasoner Zero [12] and Kimi k1.5 [33] explore similar rule-based RL pipelines to enhance reasoning in the text and image domains, respectively. However, despite encouraging progress, few prior work has explored how to extend this approach to the video domain. Bridging this gap remains an open challenge and a promising direction for expanding the boundaries of reasoning models.

- 3 Methods

- 3.1 Dataset Construction

High-quality training data plays a crucial role in reinforcing video reasoning capabilities in MLLMs. In this section, we will introduce how we curate Video-R1-260k for RL training and Video-R1-CoT165k for SFT cold start.

Data Collection and Curation. To overcome the scarcity of high-quality video reasoning training data, we strategically introduce image-based reasoning data as part of training data. The image-based data serves primarily to teach the model a broad range of reasoning skills, covering various difficulty levels and domains such as math, spatial logic, expert-level knowledge, etc. These samples help the model develop generalized reasoning abilities in static contexts. In contrast, the video-based data is primarily used to train the model’s ability to perform temporal reasoning—including understanding event progression, capturing frame-to-frame dependencies, and drawing inferences based on motion and causal dynamics over time.

We collect data from a variety of public datasets and carefully sample and balance the proportion of each subset. The final composition of the Video-R1-260k dataset is illustrated in Figure 2. The distribution of Video-R1-260k dataset can be roughly categorized as follows:

• General (Video, 116k): A diverse set of open-domain video data, covering everyday scenarios and designed to build temporal comprehension and reasoning abilities.

[Figure 6]

[Figure 7]

General (Video) 44%

LLaVA-Video-178k (mc, 73k) LLaVA-Video-178k (free, 10k) NeXT-QA (mc, 7.5k) PerceptionTest (mc, 6k) CLEVRER (mc, 8k) STAR (mc, 11.5k)

[Figure 8]

8%

General (Image) 6%

A-OKVQA (mc, 4k) IconQA (mc, 2k) ShareGPT4V (free, 2k) Visual7W (mc, 1k) ShareGPT4o (free, 6k)

14%

Chart 8%

FigureQA (mc, 4k) DVQA(num, 2k) PlotQA(num, 2k) ChartQA(num, 1k) MapQA(mc, 1k) TabMWP(mc, 2k) TabMWP(num, 2k) Chart2Text(free, 2k) RoBUT SQA(free, 2k) VisualWebInstruct(free, 6k)

[Figure 9]

44%

OCR 6%

TextVQA (ocr, 2k) HME100k(ocr, 4k) ChromeWriting(ocr, 2k) IAM (ocr, 2k) Rendered Text(ocr, 4k) TextCaps(ocr, 2k) TextOCR(ocr, 2k)

Video-R1-260k

14%

Math 14%

CLEVR-Math (num, 1k) GEOS(mc, 0.2k) Geometry3K(mc, 2k) GeoQA+ (mc, 2k) UniGeo(mc, 2k) Multimath-300K(mc/num, 27k) Super-CLEVR(num, 1.3k)

[Figure 10]

Knowledge 14%

TQA (mc, 3k) AI2D(mc, 4k) ScienceQA(mc, 4k) PMC-VQA(mc, 6k) VQA-RAD(mc, 0.5k) GVLQA(mc, 4k) ArxivQA(mc, 10k) EXAMS-V (mc, 2k) AI2D-gpt4v (free, 4k)

[Figure 11]

[Figure 12]

6%

8% 6%

Spatial 8%

OpenSpaces(reg/mc/free, 10k) Spacellava(reg/mc/free, 10k)

Figure 2: The data distribution of our Video-R1-260k dataset.

- • General (Image, 15k): A general-purpose image question-answering data, used to provide basic visual understanding.
- • Chart (Image, 21k): Visual reasoning over charts, line graphs and scientific figures, focusing on data interpretation and quantitative logic.
- • OCR (Image, 16k): Facilitate reasoning tasks that require recognizing and interpreting embedded textual content such as signs, forms, or documents.
- • Math (Image, 37k): Image-based math reasoning questions, including formulas, geometry diagrams, and multi-step symbolic reasoning.
- • Knowledge (Image, 37k): Visual commonsense and multi-discipline reasoning tasks, testing the model’s ability to integrate world knowledge with visual cues.
- • Spatial (Image, 20k): Tasks that require understand spatial information for reasoning.

CoT Annotation. To facilitate an effective SFT cold start, we leverage Qwen2.5-VL-72B-Instruct [1] to generate CoT rationales for the samples in Video-R1-260k. The prompt template for CoT generation is provided in Appendix C.1. After applying basic rule-based filtering to remove lowquality or inconsistent outputs, we obtain a high-quality CoT dataset, Video-R1-CoT-165k, which is used for the cold-start SFT stage.

Data Type and Rule-based Reward Design. Since our reinforcement learning framework follows the rule-based reward paradigm of DeepSeek-R1 [11], it is crucial to ensure that the reward signals are both reliable and precise. To this end, the majority of our training data is designed around tasks with clearly verifiable outputs, such as multiple-choice and numerical answer formats. This allows for accurate reward computation using simple rules, ensuring stable and effective RL training.

However, to increase the model’s flexibility and its ability to generalize across diverse tasks and formats, we also incorporate a smaller portion of other data types. These include free-form generation, OCR tasks, and regression problems, which are essential for adapting to real-world applications and broader datasets.

The data types and corresponding correctness reward functions are summarized as follows:

- • Multiple Choice: The reward is assigned based on whether the predicted answer matches the ground-truth answer.
- • Numerical QA: A binary reward is given depending on whether the predicted number exactly matches the reference value.
- • OCR: We compute the reward using the Word Error Rate (WER), measuring the edit distance between the predicted and reference text.
- • Free-form QA: The reward is calculated as the average of ROUGE-1, ROUGE-2, and ROUGE-L scores between the model’s output and the ground-truth answer.
- • Regression: The closer the predicted value is to the ground truth, the higher the reward, calculated as one minus their relative error.

ordered group

- 𝑜𝑜1

- 𝑜𝑜2

Question q

rewards advantages

Ref Model

… 𝑅𝑅1 𝑅𝑅2

- 𝐴𝐴1

- 𝐴𝐴2

Temporally ordered frames

𝑜𝑜𝐺𝐺

Reward Function

Policy Model

Group

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Compute …

…

𝑅𝑅𝐺𝐺

𝐴𝐴𝐺𝐺

- 𝑜𝑜1

[Figure 18]

[Figure 19]

- 𝑜𝑜2

Randomly shuffled frames

Temporal Reward

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

…

[Figure 26]

𝑜𝑜 𝐺𝐺

[Figure 27]

shuffled group

[Figure 28]

[Figure 29]

Figure 3: An illustration of our proposed T-GRPO algorithm.

#### 3.2 Temporal Group Relative Policy Optimization (T-GRPO)

While GRPO [30] has proven effective in text-based reasoning, it lacks explicit reward signals for temporal reasoning—making it insufficient for training MLLMs to reason over videos. To address this, we propose Temporal Group Relative Policy Optimization (T-GRPO), which introduces a contrastive reward mechanism that explicitly encourages temporal reasoning, as illustrated in Figure 3.

The core idea behind T-GRPO is to compare the model’s performance on the same video question when frames are provided in two different orders: (1) the temporally ordered sequence, and (2) a randomly shuffled version. For each input question, we generate two groups of responses {oi}Gi=1 and {o˜i}Gi˜=1 using the ordered and shuffled frame inputs, respectively. Let p and p˜denote the proportion of correct answers in each group. We then define a temporal reward rt as:

α, if p ≥ p˜ 0, otherwise

(1)

rt =

where α is a hyperparameter controlling the magnitude of the temporal reward. Here we set α = 0.3. This contrastive design encourages the model to perform better when the video is presented in correct temporal order than when it is shuffled. The model is only granted this positive reward if its current reasoning strategy for a given question demonstrates a reliance on temporal information. For tasks with continuous rewards (e.g., free-form answers), a threshold (e.g., 0.5) can be used to determine whether a response is considered correct.

Importantly, rt is only applied to correct responses to ensure meaningful positive advantages. Applying it to all responses would dilute the reward signal and hinder effective learning. In other words, when the model’s reasoning policy successfully relies on temporal patterns, correct responses are reinforced with a higher reward, while incorrect ones remain unaffected.

Formally, the temporal-augmented reward is defined as:

Ri =

ri + rt, if oi is correct ri, otherwise

(2)

where ri is the reward for response i, containing both the correctness reward and the format reward, following [11]. Ri is the final reward used for calculating advantages. This reward shaping ensures that when the model answers correctly under a temporal setting but fails to outperform the shuffled baseline, it receives no additional reward—pushing the optimization toward adopting a more temporally aware reasoning policy. The temporal reward rt could also be added to the advantages directly.

Then, the advantage Ai is computed over the rewards within each group:

Ri − mean({Rj}) std({Rj})

Ai =

(3)

Following DeepSeek R1 [11], the final policy update is as follows:

JT-GRPO(θ) = Eq,{o

i}

G

1 G

i=1

min

πθ(oi|q) πθ

Ai, clip

(oi|q)

old

πθ(oi|q) πθ

,1 − ϵ,1 + ϵ Ai

(oi|q)

old

− β DKL(πθ∥πref)

(4)

By explicitly comparing the model’s performance under ordered and shuffled inputs, T-GRPO introduces a contrastive training signal that drives the model to prefer reasoning strategies that leverage temporal patterns. It is worth noting that T-GRPO is only employed for video-based inputs in the training process of Video-R1.

#### 3.3 Training Strategies

We adopt Qwen2.5-VL-7B-Instruct [1] as the base MLLMs for training. Similar to DeepSeek-R1 [11], the training process is conducted in two stages: SFT cold start followed by RL training. For these two stages, we both adopt image-video mixed training strategy.

In the first stage, we perform SFT on the Video-R1-CoT-165k dataset. This step serves as a cold-start initialization, equipping the model with basic reasoning capabilities across a variety of modalities. The resulting model is denoted as Qwen2.5-VL-7B-SFT.

In the second stage, we further train the Qwen2.5-VL-7B-SFT model on the broader Video-R1-260k dataset using our proposed T-GRPO algorithm. This reinforcement learning phase is designed to guide the model beyond the rigid, pattern-matching behavior induced by SFT, encouraging it to freely explore better reasoning strategies. The resulting model is denoted as Video-R1-7B.

To further enhance the quality of reasoning, we introduce a length-based reward to regulate the length of the model’s output. Specifically, this mechanism aims to strike a balance between encouraging deeper reasoning and preventing overthinking. For each reasoning path oi, if the predicted answer is correct and the response length falls within a predefined interval [lmin,lmax], the model receives an additional reward rl = ω. Formally:

Ri =

Ri + ω, if oi is correct and lmin ≤ len(oi) ≤ lmax Ri, otherwise

(5)

This reward encourages the model to think deeply without overthinking. In this paper, we set ω = 0.2, lmin = 320 and lmax = 512. The analysis of this reward’s effect is provided in Appendix A.2.

#### 3.4 Aha Moment in Video Reasoning

We find that Video-R1 is capable of performing complex reasoning with self-reflective behaviors, often referred to as “aha moments” [11]. Specifically, the model occasionally revisits its interpretation of the video or reflects on previous outputs, particularly when confronted with ambiguous temporal cues or multi-step inference tasks. These behaviors indicate that the model is not merely executing memorized patterns but is actively engaging in internal feedback loops by re-examining evidence. An example is demonstrated in Figure 4. More examples can be found in Appendix B.

### 4 Experiments

#### 4.1 Setup

Benchmarks. We evaluate our model on six video benchmarks: VSI-Bench [38], VideoMMMU [13], MMVU [48], MVBench [20], TempCompass [27], and VideoMME [9]. Among them, the first three are video reasoning benchmarks, which focus primarily on assessing the model’s reasoning capabilities in video understanding. The latter three are general-purpose video understanding benchmarks, which include a mixture of perception and reasoning tasks. For MMVU, we evaluate on its multiple-choice question set for stability and consistency. For all evaluations, we follow the decoding configuration used in the official Qwen2.5-VL demo, with top_p = 0.001 and temperature = 0.01.

Video:

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Question: You are a robot beginning at the blue built-in bookshelf facing the blue built-in bookshelf. You want to navigate to the bathroom sink . You will perform the following actions (Note: for each [please fill in], choose either 'turn back,' 'turn left,' or 'turn right.'): 1. [please fill in] 2. Go forward until the door of the bathroom. 3. [please fill in] 4. Go forward until the sink. You have reached the final destination. A. Turn Back, Turn Right B. Turn Left, Turn Left C. Turn Right, Turn Left D. Turn Back, Turn Left

Qwen2.5-VL-7B-Instruct: Let me think through this step by step. I start at the blue built-in bookshelf. To get to the bathroom sink, I need to move in a specific direction. The first action is to go forward, which means I need to move away from the bookshelf. Since I'm facing the bookshelf, I should turn left to move forward. After that, I need to go forward until I reach the bathroom door. Once I'm at the door, I should turn left to go into the bathroom and then turn left again to reach the sink. This sequence of actions aligns with option B. <answer>B</answer>

Video-R1-7B: <think>Let me think aboutthis carefully. The goal is to navigate from the blue built-in bookshelf to the bathroom sink. The first step is to determine the direction from the bookshelf to the bathroom.

- 1.The blue built-in bookshelf is on the left side of the room. The bathroom is on the right side of the room. So, I need to turn right from the bookshelf to head towards the bathroom.
- 2. After turning right, I need to go forward until I reach the door of the bathroom. This means I need to continue moving forward in the direction I just turned.
- 3. Once I reach the bathroom door, I need to turn left to enter the bathroom. This is because the sink is typically located on the left side of the bathroom.

So, the sequence of actions should be:1. Turn right from the bookshelf. 2. Go forward until the bathroom door. 3. Turn left to enter the bathroom. 4. Go forward until the sink. (Perception and Reasoning)

Hmm, this seems to align with the sequence of actions in option C. Let's break it down one more time to verify.

- Option A: Turn Back, Turn Right - This would mean going back to the bookshelf and then turning right, which doesn't make sense for reaching the bathroom.
- Option B: Turn Left, Turn Left - This would mean turning left from the bookshelf and then turning left again, which also doesn't make sense. Option D: Turn Back, Turn Left - This would mean going back to the bookshelf and then turning left, which is not the correct path. (Reflection)

Oh, I see. The correct sequence is indeed in option C. Let's confirm one last time. The correct sequence is to turn right, go forward to the bathroom door, turn left, and go forward to the sink. This matches option C. Final answer: C Let's verify one last time. Yes, C is the correct answer.</think> <answer>C</answer>

Figure 4: An example of Video-R1-7B’s reasoning output on VSI-Bench.

Training Details. We train our model using up to 8 NVIDIA A100 (80GB) GPUs. For efficiency considerations, we limit the maximum number of video frames to 16 during training. Each frame is processed at a max resolution of 128 × 28 × 28 pixels. During inference, we increase the frame resolution to 256 × 28 × 28 pixels and frames to 16∼64 to enhance performance. The ordered group size G is set to 8 and the shuffled group size G˜ is set to half of that for efficiency. More details are provided in Appendix D.

We first perform SFT on Video-R1-CoT-165k for one epoch to obtain the Qwen2.5-VL-7B-SFT model. This is followed by RL training on Video-R1-260k to produce the final Video-R1 model. Due to current computational resource limitations, we train the model for only 1k RL steps. Surprisingly, even within this limited training budget, the model exhibits significant improvements in video reasoning performance, indicating the strong effectiveness of both our data design and algorithm. Further experiments exploring the impact of scaling up RL training are presented in Appendix A.1.

#### 4.2 Main Results

As shown in Table 1. Our experimental results across six benchmarks validate the effectiveness of Video-R1 in video reasoning and general video understanding tasks. The key findings are as follows.

Superior Performance of Video-R1. Video-R1 significantly outperforms previous models across most benchmarks, with particularly strong gains on video reasoning tasks such as VSI-Bench, VideoMMMU, and MMVU. This highlights the necessity of explicit reasoning capability in solving video tasks, and confirms the effectiveness of reinforcement learning for video tasks.

RL Works Better Than SFT. We observe that the SFT model Qwen2.5-VL-7B-SFT does not consistently improve performance. In some cases (e.g., VideoMME), performance even slightly drops after SFT, likely due to overfitting or limited generalization in unseen scenarios [5]. In contrast, after only 1k steps of reinforcement learning, Video-R1 achieves significant performance boosts. This clearly demonstrates the importance of RL in unlocking generalizable video reasoning capability.

Table 1: Performance of different models on benchmarks.

Video Reasoning Benchmark Video General Benchmark

Models Frames

VSI-Bench VideoMMMU MMVU (mc) MVBench TempCompass VideoMME (wo sub)

GPT-4o [15] - 34.0 61.2 75.4 - - 71.9

LLaMA-VID [22] - - - - 41.9 45.6 VideoLLaMA2 [4] - - - 44.8 54.6 - 47.9

LongVA-7B [43] - 29.2 23.9 - - 56.9 52.6 VILA-1.5-8B [25] - 28.9 20.8 - - 58.8 -

VILA-1.5-40B [25] - 31.2 34.0 - - - 60.1 Video-UTR-7B [40] - - - - 58.8 59.7 52.6

LLaVA-OneVision-7B [19] - 32.4 33.8 49.2 56.7 - 58.2

Kangaroo-8B [26] - - - - 61.1 62.5 56.0 Qwen2.5-VL-7B (CoT) 16 27.7 47.8 59.2 57.4 72.2 53.1

Qwen2.5-VL-7B-SFT 16 31.8 47.4 61.3 59.4 69.2 52.8 Qwen2.5-VL-7B (CoT) 32 30.1 48.1 60.0 59.0 72.6 56.6

Qwen2.5-VL-7B-SFT 32 33.3 49.4 63.5 60.5 69.9 55.4 Qwen2.5-VL-7B (CoT) 64 31.4 50.4 60.0 59.2 72.9 59.6

Qwen2.5-VL-7B-SFT 64 34.8 49.4 61.6 60.6 70.0 58.8

Video-R1-7B 16 34.6 49.8 64.2 62.7 72.6 57.4 Video-R1-7B 32 35.8 52.3 63.8 63.9 73.2 59.3 Video-R1-7B 64 37.1 52.4 63.8 64.8 73.2 61.4

Table 2: Ablation Study.

Video Reasoning Benchmark Video General Benchmark

Models Frames

VSI-Bench VideoMMMU MMVU (mc) MVBench TempCompass VideoMME (wo sub)

Video-R1-7B-wo-image 16 32.3 45.8 60.6 60.9 69.8 53.8 Video-R1-7B-wo-temporal 16 32.7 48.3 62.1 61.1 71.3 54.5

Video-R1-7B-zero 16 31.8 49.5 63.8 60.4 70.9 53.8 Video-R1-7B 16 34.6 49.8 64.2 62.7 72.6 57.4

More Frames Lead to Better Reasoning. When increasing the input frame number, we observe performance improvements on almost all benchmarks. This indicates that richer context and temporal information contribute positively to the reasoning performance. Therefore, developing models capable of understanding and reasoning over longer video inputs is a promising direction for future research.

#### 4.3 Ablation Study

In this section, we conduct an ablation study by designing three variants of our model: Video-R17B-wo-image, which removes all image-based data during training and relies solely on video data; Video-R1-7B-wo-temporal, which replaces our proposed T-GRPO algorithm with the original GRPO method; Video-R1-zero, which skips the SFT cold start and directly conducts RL training.

As shown in Table 2, both ablated models perform worse than the full Video-R1-7B across all benchmarks. In particular, removing image data leads to a noticeable drop in performance on both video reasoning and general benchmarks, indicating that image-based samples play a crucial role in bootstrapping general reasoning ability. Similarly, without temporal-aware training via T-GRPO, the model struggles to fully leverage temporal cues, resulting in weaker performance on benchmarks. Additionally, skipping the SFT cold start leads to worse performance, underscoring the value of our Video-R1-CoT-165k dataset and the importance of initializing with SFT before RL training. These ablations validate the effectiveness of our proposed methods.

0.75

0.70

AccuracyReward

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.65

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.60

0.55

0.50

0 200 400 600 800 1000

Steps

(a) Accuracy reward

0.75

0.70

TemporalReward

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.65

0.60

0.55

0.50

0 200 400 600 800 1000

Steps

(b) Temporal reward rt (scaled)

Figure 5: RL training curves.

450

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
| | || |
|---|
<br><br>| | | | | |
| | || |
|---|
<br><br>| |
|---|
| | | | | |
| | | | | | | | |
| | | | | | | | |

400

ResponseLength

350

300

250

200

0 200 400 600 800 1000

Steps

(c) Response length

#### 4.4 Training Curves

Figure 5 illustrates the RL training dynamics of Video-R1. As shown in Figure 5(a), the accuracy reward exhibits a generally upward trend, indicating that the model continuously improves its ability to produce correct answers under reinforcement learning. In Figure 5(b), the temporal reward rt (scaled to 0∼1 for visibility) also demonstrates a steady increase. This suggests that the model is progressively adopting more temporally grounded reasoning strategies with T-GRPO. Interestingly, the response length in Figure 5(c) drops initially during RL training, then rises and stabilizes. We guess that this may reflect a learning transition: the model first discards its sub-optimal SFT reasoning style, and eventually settles on a new reasoning policy.

#### 4.5 Effect of Temporal Reward Analysis

Video-R1-wo-temporal 60.2%

To further assess the impact of T-GRPO, we measure the percentage of responses that incorporate temporal reasoning for questions requiring it. Specifically, we use Qwen2.5-VL-72B to identify all temporally grounded questions across six benchmarks, and then use Qwen2.5VL-72B to evaluate whether the model responses demonstrate temporal reasoning. The evaluation prompt can be found in Appendix C.2. As shown in Figure 6, Video-R1 trained by T-GRPO incorporates temporal reasoning in 75.0% of responses, compared to 60.2% for Video-R1-wo-temporal, which lacks temporal modeling. This clear gap demonstrates the effectiveness of T-GRPO in encouraging the model to leverage temporal information during reasoning.

Video-R1 75.0%

Percentage (%)

Figure 6: Percentage of temporal reasoning responses.

60

AveragePerformance

We also conduct a sensitivity analysis on the magnitude of the temporal reward, controlled by the hyperparameter α, as shown in Figure 7. The average performance is calculated as the mean performance across six benchmarks. We observe a slight drop in performance at α = 0.1 and α = 0.4, while α = 0.2 and α = 0.3 yield similar and favorable results. This indicates that the model is relatively insensitive to the choice of α within a reasonable range.

50

40

30

20

10

0.1 0.2 0.3 0.4

Figure 7: Sensitivity analysis.

### 5 Conclusions

In this work, we present Video-R1, as an attempt to investigate the R1 paradigm to enhances video reasoning in multimodal large language models. Motivated by DeepSeek-R1, we extend rule-based reinforcement learning to video by introducing T-GRPO, a temporal-aware extension of GRPO that explicitly encourages temporal reasoning in video. To support training, we curate two datasets, Video-R1-CoT-165k for SFT cold start and Video-R1-260k for RL. Experimental results across six benchmarks validate the effectiveness of our approach. We hope this work provides a foundation for further research in video reasoning with MLLMs.

### 6 Acknowledgment

This work is partially supported by the National Natural Science Foundation of China (Grant No. 62306261), and The Shun Hing Institute of Advanced Engineering (SHIAE) Grant (No. 8115074). This study was supported in part by the Centre for Perceptual and Interactive Intelligence, a CUHKled InnoCentre under the InnoHK initiative of the Innovation and Technology Commission of the Hong Kong Special Administrative Region Government.

### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [2] Joya Chen, Zhaoyang Lv, Shiwei Wu, Kevin Qinghong Lin, Chenan Song, Difei Gao, Jia-Wei Liu, Ziteng Gao, Dongxing Mao, and Mike Zheng Shou. Videollm-online: Online video large language model for streaming video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18407–18418, 2024.
- [3] Qiguang Chen, Libo Qin, Jinhao Liu, Dengyun Peng, Jiannan Guan, Peng Wang, Mengkang Hu, Yuhang Zhou, Te Gao, and Wanxiang Che. Towards reasoning era: A survey of long chain-of-thought for reasoning large language models. arXiv preprint arXiv:2503.09567, 2025.
- [4] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024.
- [5] Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V Le, Sergey Levine, and Yi Ma. Sft memorizes, rl generalizes: A comparative study of foundation model post-training. arXiv preprint arXiv:2501.17161, 2025.
- [6] Kaixuan Fan, Kaituo Feng, Haoming Lyu, Dongzhan Zhou, and Xiangyu Yue. Sophiavl-r1: Reinforcing mllms reasoning with thinking reward. arXiv preprint arXiv:2505.17018, 2025.
- [7] Jiazhan Feng, Shijue Huang, Xingwei Qu, Ge Zhang, Yujia Qin, Baoquan Zhong, Chengquan Jiang, Jinxin Chi, and Wanjun Zhong. Retool: Reinforcement learning for strategic tool use in llms. arXiv preprint arXiv:2504.11536, 2025.
- [8] Kaituo Feng, Changsheng Li, Xiaolu Zhang, Jun Zhou, Ye Yuan, and Guoren Wang. Keypointbased progressive chain-of-thought distillation for llms. arXiv preprint arXiv:2405.16064, 2024.
- [9] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024.
- [10] Jiaxuan Gao, Shusheng Xu, Wenjie Ye, Weilin Liu, Chuyi He, Wei Fu, Zhiyu Mei, Guangju Wang, and Yi Wu. On designing effective rl reward at training time for llm reasoning. arXiv preprint arXiv:2410.15115, 2024.
- [11] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [12] Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, and Heung-Yeung Shum Xiangyu Zhang. Open-reasoner-zero: An open source approach to scaling reinforcement learning on the base model, 2025.
- [13] Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Yuanhan Zhang, Xiang Yue, Bo Li, and Ziwei Liu. Video-mmmu: Evaluating knowledge acquisition from multi-discipline professional videos. arXiv preprint arXiv:2501.13826, 2025.

- [14] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025.
- [15] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [16] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.
- [17] Leslie Pack Kaelbling, Michael L Littman, and Andrew W Moore. Reinforcement learning: A survey. Journal of artificial intelligence research, 4:237–285, 1996.
- [18] Yuxiang Lai, Jike Zhong, Ming Li, Shitian Zhao, and Xiaofeng Yang. Med-r1: Reinforcement learning for generalizable medical reasoning in vision-language models. arXiv preprint arXiv:2503.13939, 2025.
- [19] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [20] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195–22206, 2024.
- [21] Wendi Li and Yixuan Li. Process reward model with q-value rankings. arXiv preprint arXiv:2410.11287, 2024.
- [22] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. In European Conference on Computer Vision, pages 323–340. Springer, 2024.
- [23] Zhong-Zhi Li, Duzhen Zhang, Ming-Liang Zhang, Jiaxin Zhang, Zengyan Liu, Yuxuan Yao, Haotian Xu, Junhao Zheng, Pei-Jie Wang, Xiuyi Chen, et al. From system 1 to system 2: A survey of reasoning large language models. arXiv preprint arXiv:2502.17419, 2025.
- [24] Zongzhao Li, Zongyang Ma, Mingze Li, Songyou Li, Yu Rong, Tingyang Xu, Ziqi Zhang, Deli Zhao, and Wenbing Huang. Star-r1: Spatial transformation reasoning by reinforcing multimodal llms. arXiv preprint arXiv:2505.15804, 2025.
- [25] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26689–26699, 2024.
- [26] Jiajun Liu, Yibing Wang, Hanghang Ma, Xiaoping Wu, Xiaoqi Ma, Xiaoming Wei, Jianbin Jiao, Enhua Wu, and Jie Hu. Kangaroo: A powerful video-language model supporting long-context video input. arXiv preprint arXiv:2408.15542, 2024.
- [27] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? arXiv preprint arXiv:2403.00476, 2024.
- [28] Zhaowei Liu, Xin Guo, Fangqi Lou, Lingfeng Zeng, Jinyi Niu, Zixuan Wang, Jiajie Xu, Weige Cai, Ziwei Yang, Xueqian Zhao, et al. Fin-r1: A large language model for financial reasoning through reinforcement learning. arXiv preprint arXiv:2503.16252, 2025.
- [29] Zhengxi Lu, Yuxiang Chai, Yaxuan Guo, Xi Yin, Liang Liu, Hao Wang, Guanjing Xiong, and Hongsheng Li. Ui-r1: Enhancing action prediction of gui agents by reinforcement learning. arXiv preprint arXiv:2503.21620, 2025.
- [30] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

- [31] Fangxun Shu, Lei Zhang, Hao Jiang, and Cihang Xie. Audio-visual llm for video understanding. arXiv preprint arXiv:2312.06720, 2023.
- [32] Yunlong Tang, Jing Bi, Siting Xu, Luchuan Song, Susan Liang, Teng Wang, Daoan Zhang, Jie An, Jingyang Lin, Rongyi Zhu, et al. Video understanding with large language models: A survey. IEEE Transactions on Circuits and Systems for Video Technology, 2025.
- [33] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.
- [34] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.
- [35] Junfei Wu, Jian Guan, Kaituo Feng, Qiang Liu, Shu Wu, Liang Wang, Wei Wu, and Tieniu Tan. Reinforcing spatial reasoning in vision-language models with interwoven thinking and visual drawing. arXiv preprint arXiv:2506.09965, 2025.
- [36] Tian Xie, Zitian Gao, Qingnan Ren, Haoming Luo, Yuqian Hong, Bryan Dai, Joey Zhou, Kai Qiu, Zhirong Wu, and Chong Luo. Logic-rl: Unleashing llm reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2502.14768, 2025.
- [37] Cilin Yan, Haochen Wang, Shilin Yan, Xiaolong Jiang, Yao Hu, Guoliang Kang, Weidi Xie, and Efstratios Gavves. Visa: Reasoning video object segmentation via large language models. In European Conference on Computer Vision, pages 98–115. Springer, 2024.
- [38] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. arXiv preprint arXiv:2412.14171, 2024.
- [39] Chris Yi Peng et al. Skywork r1v: Pioneering multimodal reasoning with chain-of-thought, 2025.
- [40] En Yu, Kangheng Lin, Liang Zhao, Yana Wei, Zining Zhu, Haoran Wei, Jianjian Sun, Zheng Ge, Xiangyu Zhang, Jingyu Wang, et al. Unhackable temporal rewarding for scalable video mllms. arXiv preprint arXiv:2502.12081, 2025.
- [41] Jiakang Yuan, Tianshuo Peng, Yilei Jiang, Yiting Lu, Renrui Zhang, Kaituo Feng, Chaoyou Fu, Tao Chen, Lei Bai, Bo Zhang, et al. Mme-reasoning: A comprehensive benchmark for logical reasoning in mllms. arXiv preprint arXiv:2505.21327, 2025.
- [42] Lifan Yuan, Ganqu Cui, Hanbin Wang, Ning Ding, Xingyao Wang, Jia Deng, Boji Shan, Huimin Chen, Ruobing Xie, Yankai Lin, et al. Advancing llm reasoning generalists with preference trees. arXiv preprint arXiv:2404.02078, 2024.
- [43] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024.
- [44] Xiaoying Zhang, Da Peng, Yipeng Zhang, Zonghao Guo, Chengyue Wu, Chi Chen, Wei Ke, Helen Meng, and Maosong Sun. Will pre-training ever end? a first step toward next-generation foundation mllms via self-improving systematic cognition. arXiv preprint arXiv:2503.12303, 2025.
- [45] Xiaoying Zhang, Hao Sun, Yipeng Zhang, Kaituo Feng, Chaochao Lu, Chao Yang, and Helen Meng. Critique-grpo: Advancing llm reasoning with natural language and numerical feedback. arXiv preprint arXiv:2506.03106, 2025.
- [46] Yipeng Zhang, Yifan Liu, Zonghao Guo, Yidan Zhang, Xuesong Yang, Chi Chen, Jun Song, Bo Zheng, Yuan Yao, Zhiyuan Liu, et al. Llava-uhd v2: an mllm integrating high-resolution feature pyramid via hierarchical window transformer. arXiv preprint arXiv:2412.13871, 2024.

- [47] Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-of-thought reasoning in language models. arXiv preprint arXiv:2302.00923, 2023.
- [48] Yilun Zhao, Lujing Xie, Haowei Zhang, Guo Gan, Yitao Long, Zhiyuan Hu, Tongyan Hu, Weiyuan Chen, Chuhan Li, Junyang Song, et al. Mmvu: Measuring expert-level multi-discipline video understanding. arXiv preprint arXiv:2501.12380, 2025.
- [49] Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, et al. Least-to-most prompting enables complex reasoning in large language models. arXiv preprint arXiv:2205.10625, 2022.

### A Additional Experiments

#### A.1 Scaling Up RL Training

In previous experiments, our Video-R1 is trained with 1k RL steps, similar to prior works that apply the R1 paradigm in other domains [7, 29]. To further explore the impact of scaling up reinforcement learning, we extend the training to 10k RL steps. As shown in Table 3, the performance of Video-R1 generally improves on various benchmarks with more RL training steps. These results demonstrate the effectiveness of large-scale RL training and suggest that additional training can further enhance the model’s reasoning capabilities.

Table 3: Results on more RL training steps.

Video Reasoning Benchmark Video General Benchmark

Models Frames

VSI-Bench VideoMMMU MMVU (mc) MVBench TempCompass VideoMME (wo sub)

Video-R1-7B (1k steps) 16 34.6 49.8 64.2 62.7 72.6 57.4 Video-R1-7B (1k steps) 32 35.8 52.3 63.8 63.9 73.2 59.3 Video-R1-7B (1k steps) 64 37.1 52.4 63.8 64.8 73.2 61.4

Video-R1-7B (10k steps) 16 34.5 50.7 64.2 63.7 73.7 57.2 Video-R1-7B (10k steps) 32 35.7 52.7 66.2 64.7 74.2 59.3 Video-R1-7B (10k steps) 64 37.8 51.4 65.0 65.5 74.2 61.8

#### A.2 Effect of Length Reward Analysis

To evaluate the impact of the proposed length reward, we conduct an ablation study by comparing Video-R1 with a variant trained without the length reward, denoted as Video-R1-wo-len. As shown in Figure 8(b), removing the length reward leads to a clear decline in response length during RL training, whereas Video-R1 maintains a steadily increasing length that eventually stabilizes at a higher level. Finally, Video-R1-wo-len achieves lower average performance across benchmarks compared to Video-R1, as shown in Figure 8(a). We guess that this is because the model fails to preserve a deep reasoning style without the length reward during training, instead favoring shorter and potentially less informative responses. This likely results in weaker generalization during evaluation. These findings highlight the importance of encouraging the model to maintain a moderate level of reasoning effort during training.

450

Video-R1

Video-R1-wo-len

60

400

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

ResponseLength

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

AveragePerformance

| |
|---|

| |
|---|

50

350

| |
|---|

300

| |
|---|

40

| |
|---|

| |
|---|

250

30

200

0 200 400 600 800 1000

20

Steps

Video-R1 Video-R1-wo-len

(a) Performance comparison

(b) Length curve comparison

Figure 8: Effect of length reward.

### B Reasoning Examples

Video:

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Question: Which move motion in the video lose the system energy? Video-R1-7B: <think>Let me think about this for a moment. The question is asking about which move in the video loses kinetic energy. To answer this, I need to carefully analyze the video and identify the different actions and their outcomes. First, let's break down the video content. The video shows a series of actions involving a pool table, a bowling alley, and a car crash. In the pool table scene, a player takes a shot, and the balls move around. In the bowling alley scene, a ball rolls down the lane and hits the pins. In the car crash scene, two cars collide. Now, let's consider the energy loss in each scene:

- 1. In the pool table scene, the player takes a shot, and the balls move. The kinetic energy is transferred from the cue stick to the balls, and the balls continue to move. This does not seem to lose kinetic energy.
- 2. In the bowling alley scene, the ball rolls down the lane and hits the pins. The ball loses some kinetic energy upon hitting the pins, but it still has some energy left as it rolls away.
- 3. In the car crash scene, the two cars collide. The kinetic energy is transferred between the cars, and some energy is lost as heat and deformation of the cars. This is a significant loss of kinetic energy. Hmm, it seems that the car crash scene is the one where the most kinetic energy is lost. The cars collide, and the energy is dissipated in the form of heat and deformation, which is a significant loss compared to the other scenes. (Perception and Reasoning)

Oh, I see. The car crash scene is the one where the most kinetic energy is lost. The other scenes involve transfers of energy, but the car crash involves a significant loss.

Let's verify this. The car crash scene is indeed the one where the most kinetic energy is lost, as the cars collide and the energy is dissipated in a dramatic way. (Reflection)

So, the move that loses the most kinetic energy in the video is the car crash.</think> <answer>The car crash in the video loses the system energy.</answer>

Figure 9: An example of Video-R1-7B’s reasoning output on MMVU Benchmark.

Video:

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Question: Why did Ross leave before responding to Rachael? A. He really had to pee. B. He forgot he left his stove on at home. C. He left his dog tied up outside. D. Because he was upset his feelings for her were not reciprocated. E. He was being called by someone outside.

Video-R1-7B: <think>Let me think about this for a moment. The question is asking why Ross left before responding to Rachael. The options provided give us a range of possible reasons, and we need to consider each one carefully.

First, let's break down the context. Ross and Rachael are likely in a romantic relationship, and Ross has been feeling something for Rachael. The question is why he left before responding to her.

- Option A: "He really had to pee." This seems unlikely because it's not a common reason for leaving a conversation abruptly. It's not a significant enough reason to leave before responding.
- Option B: "He forgot he left his stove on at home." This is also unlikely. While it's possible he could have forgotten, it's not a typical reason for leaving a conversation.
- Option C: "He left his dog tied up outside." This is possible, but it's not a common reason for leaving a conversation. It's more of a practical issue rather than a significant emotional reason.
- Option D: "Because he was upset his feelings for her were not reciprocated." This seems plausible. If Ross was upset that his feelings were not reciprocated, he might have left before responding to Rachael to avoid further emotional pain.
- Option E: "He was being called by someone outside." This is possible, but it's not a common reason for leaving a conversation. It's more of a practical issue rather than a significant emotional reason.

Hmm, after considering all the options, Option D seems the most plausible. It's a significant emotional reason that would likely cause Ross to leave before responding to Rachael. (Perception and Reasoning)

Oh, I see. The most logical reason for Ross to leave before responding to Rachael is that he was upset his feelings for her were not reciprocated. Let's verify this one last time. Yes, it fits the context and the emotional state of Ross. (Reflection) </think> <answer>D</answer>

##### Figure 10: An example of Video-R1-7B’s reasoning output on MVBench.

### C Prompt Template

- C.1 Prompt Template for Training and Inference

- Figure 11 illustrates the prompt template for training and inference of all models. We also use this prompt for the COT annotation.

{Question} Please think about this question as if you were a human pondering deeply. Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions. It's encouraged to include selfreflection or verification in the reasoning process. Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags. {OutputTemplate}

OutputTemplate:

"multiple choice": Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags. "numerical": Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags. "OCR": Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags. "free-form": Please provide your text answer within the <answer> </answer> tags. "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.

Prompt Template for Training and Inference

Figure 11: Prompt template for training and inference.

C.2 Prompt Template for Temporal Reasoning Evaluation

- Figure 12 illustrates the prompt template for temporal reasoning evaluation.

Prompt Template for Temporal Reasoning Evaluation

{Question} You are given a question about a video and a response to that question. Your task is to assess whether the response engages with any aspect of the video's temporal structure — that is, any elements related to time, sequence, or progression. This may include references to actions occurring in a particular order, events unfolding over time, descriptions of what happens before or after, mentions of duration, speed, or direction, or any general sense of temporal flow conveyed in the response. Consider a broad range of interpretations, including responses that reflect an understanding of how events evolve or follow one another. Provide your reasoning between the <think> and </think> tags, and then write either 'yes' or 'no' between the <answer> and </answer> tags.

Figure 12: Prompt template for temporal reasoning evaluation.

### D Additional Implementation Details

We use the Adam optimizer with a learning rate of 1e-6 to train our model. The SFT stage takes approximately 40 hours per epoch, while the RL stage takes around 15 hours for 1k steps. The hyperparameter β in the KL divergence term of the GRPO algorithm is set to 0.04. To ensure training stability, we apply a weight decay rate of 0.01 and clip the maximum gradient norm to 5. The maximum response length is set to 768 tokens.

### E Limitations and Future Works

We envision this work as a foundation for advancing research in video reasoning with MLLMs. Below, we outline its limitations and potential avenues for future work:

- • Increasing Frames Number. Currently, our model is trained with 16 video frames, which may limit its ability to handle long-range temporal dependencies. In future work, we can develop more efficient training and inference strategies that allow scaling to longer videos, enabling more comprehensive temporal reasoning.
- • Better Temporal Modeling Method. Although T-GRPO introduces effective temporalaware reasoning, it brings additional computational overhead due to contrastive evaluation and reward calculation. This could be mitigated through inference acceleration framework such as vLLM, or by exploring more efficient mechanisms for temporal modeling.
- • Adaptive Response Length Control. Our current length control mechanism applies a fixed reward within a predefined range, regardless of the complexity of each sample. Future work could explore dynamic length control strategies, where the model adaptively determines the appropriate response length based on the difficulty or type of the question.
- • Refined Image-to-Video Knowledge Transfer. At present, we incorporate image-based reasoning data in a straightforward manner by mixing it into the training set. Future research could design more principled approaches for leveraging image data to more effectively transfer reasoning ability from images to videos.
- • Generalist Video Reward Modeling. Currently, we use rule-based reward functions tailored to different tasks. A promising future direction is to develop a generalist video reward model, which is capable of providing consistent and scalable reward signals across various video reasoning tasks. Such a model could reduce reliance on handcrafted rules and enable broader applicability of reinforcement learning in video understanding.

