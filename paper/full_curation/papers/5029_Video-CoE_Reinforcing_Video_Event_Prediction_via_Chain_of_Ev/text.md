## Video-CoE: Reinforcing Video Event Prediction via Chain of Events

#### Qile Su§*, Jing Tang‡, Rui Chen, Lei Sun, Xiangxiang Chu

AMAP, Alibaba Group

‡ Project lead. § Corresponding author.

# arXiv:2603.14935v1[cs.CV]16Mar2026

#### Abstract

Despite advances in the application of MLLMs for various video tasks, video event prediction (VEP) remains relatively underexplored. VEP requires the model to perform finegrained temporal modeling of videos and establish logical relationships between videos and future events, which current MLLMs still struggle with. In this work, we first present a comprehensive evaluation of current leading MLLMs on the VEP task, revealing the reasons behind their inaccurate predictions, including lack of logical reasoning ability for future events prediction and insufficient utilization of visual information. To address these challenges, we propose Chain of Events (CoE) paradigm, which constructs temporal event chains to implicitly enforce MLLM focusing on the visual content and the logical connections between videos and future events, incentivizing model’s reasoning capability with multiple training protocols. Experimental results on public benchmarks demonstrate that our method outperforms both leading open-source and commercial MLLMs, establishing a new state-of-the-art on the VEP task. Codes and models will be released soon.

#### 1. Introduction

Multimodal Large Language Models (MLLMs) [1, 35, 46] have achieved remarkable results across a range of vision tasks [13, 19, 25, 32, 57, 64], demonstrating strong capabilities in video understanding, reasoning, and question answering. These tasks collectively underpin the predominant pre-training and post-training paradigms for MLLMs, enabling them to generalize effectively to diverse downstream applications [30, 51, 56]. Nevertheless, real-world scenarios, such as crisis early warning, require the ability to predict future events from observed videos, a capability that remains largely underexplored in current MLLM research.

To fill this gap, we first conduct a systematic evaluation of state-of-the-art open-source MLLMs [1, 40, 42, 46, 54, 61] and commercial GPT-series models [35] on the video

*Work done during the internship at AMAP, Alibaba Group.

event prediction (VEP) [29, 43] task, as shown in Tabs. 1 and 2. Our experiments indicate that current MLLMs perform markedly worse on VEP than on standard vision tasks. We attribute this gap to insufficient pretraining on the VEP task, which leaves models without the inductive biases and reasoning skills required for accurate future events prediction. Directly training these models for VEP task would require large-scale datasets and substantial computational resources, making it costly to incorporate this objective into pretraining. This motivates a more efficient approach to strengthen MLLMs’ video event prediction capabilities without large-scale annotation or extensive retraining.

To this end, we perform a systematic analysis of the limitations faced by state-of-the-art open-source MLLMs on zero-shot video event prediction. As illustrated in Fig. 1, our study uncovers two primary failure causes:

Lack of Logical Reasoning Ability for Future Events. Unlike standard video understanding and reasoning tasks, VEP aims to anticipate plausible future events that are not directly observable in the input video. This requires models to possess the ability to reason over the video content to predict the future events. However, as shown in Fig. 1a, current MLLMs often rely on cues in textual answer options rather than grounding predictions in video evidence, indicating a weak linkage between observed content and the future. This shortcut behavior contributes to their subpar performance on VEP. Moreover, in real-world applications, video event prediction is inherently an open-set problem, where future events are not confined to a fixed label space, further limiting the practical applicability of current MLLMs.

Insufficient Utilization of Visual Information. As shown in Fig. 1a, our observations indicate that current MLLMs make limited use of visual evidence during reasoning, instead over-relying on textual cues or answer choices. An analysis of attention distributions over visual and textual tokens, shown in Fig. 1b, further reveals that models allocate substantially less attention to visual tokens during prediction. Yet prior studies [16, 28, 29] demonstrate that fine-grained temporal modeling is essential for forecasting future events. This text-centric modality bias likely undermines robust predictive reasoning, leading to sub-

[Figure 1]

[Figure 2]

[Figure 3]

| | | | |
|---|---|---|---|
|[Figure 4]| | | |
| | | | |

SYSTEM PROMPT TOKENS

QUESTION TOKENS OPTIONS TOKENS

- Event 1
- Event 2
- Event 3
- Event 4

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

TOKEN SEQUENCE

×10

×10

×10

𝛿DifferenceinAttentionScores

𝛿DifferenceinAttentionScores

𝛿DifferenceinAttentionScores

1.75

2.0

2.0

video

1.50

1.5

1.25

1.5

[Figure 11]

1.00

###### ……. Which option is most likely to be the next event that occurs? [Options]

1.0

1.0

0.75

Visual Tokens

Visual Tokens Visual Tokens

0.50

0.5

0.5

[Figure 12]

0.25

"<think>\nThe video starts with a title card for a surfing event at Sandy Beach,

[Figure 13]

0.0

0.0

0.00

followed by various clips of surfers riding waves. The video ends with a surfer in a blue shirt and black shorts riding a white surfboard.

Visual Token Sequence S

Visual Token Sequence S

Visual Token Sequence S

[Figure 14]

[Figure 15]

[Figure 16]

GLM-4.1V

Qwen2.5-VL

InternVL3

[Figure 17]

[Figure 18]

[Figure 19]

logicalconnection

↑

×10

×10

×10

2.00

𝛿DifferenceinAttentionScores

𝛿DifferenceinAttentionScores

Visual Information Utilization Textual Information Utilization

𝛿DifferenceinAttentionScores

1.75

2.0

8.0

1.50

↑

\nOption A introduces…, which seems unrelateed ….\n\nOption B focuses…, maintaining…. \n \nOption C include-es …, which adds …. \n adds \nOption D prese-nts …, which could be …. \n \nConsidering the need to smoothly transition from the surfing action to the promotional content, Option D provides a natural pause and sets up the uncertainty that can lead into the promotional segment.\n</think> \n \n<answer>D.</answer>"

1.5

1.25

6.0

1.00

1.0

4.0

0.75

0.50

0.5

2.0

0.25

Visual Tokens Visual Tokens Visual Tokens

0.0

0.00

0.0

Visual Token Sequence S Visual Token Sequence S Visual Token Sequence S

[Figure 20]

[Figure 21]

Kimi-VL Mimo-VL

LLaVA-NeXT-Video

(a) An example of MLLMs performing VEP.

(b) Attention distribution over input token sequences.

- Figure 1. Analysis of open-source MLLMs on video event prediction tasks. Fig. 1a illustrates the reasoning process, indicating the lack of logical reasoning capabilities in VEP task. Fig. 1b illustrates the attention distribution of the option tokens over input tokens demonstrating the insufficient utilization of visual information.

a cold-start. In the second stage, CoE-GRPO strengthens model’s temporal localization and video understanding capabilities, enabling the model to construct fine-grained temporal event chains, providing sufficient visual information and logical support for prediction.

optimal performance on VEP. Although previous works [23, 31, 62, 66] have proposed (i) directly amplifying attention to visual tokens at inference and (ii) using prompts to encourage visual grounding, we find these approaches ineffective for VEP and even lead to performance degradation.

To address these challenges, we propose Chain of Events (CoE), a paradigm for video event prediction. CoE first constructs a fine-grained temporal representation by segmenting the input video into a sequence of historical events, forming an explicit event chain. This step promotes stronger visual grounding and mitigates the common visual–textual utilization bias in MLLMs, providing a more reliable basis for subsequent logical reasoning. The model then reasons jointly over the observed video and the constructed event chain to anticipate plausible future events, rather than relying on superficial cues from textual options analysis. By explicitly linking observed events to potential future events via causal–temporal reasoning, CoE enhances predictive performance on VEP task and directly addresses the limitations of current MLLMs.

We evaluate our approach on established video event prediction benchmarks using Qwen2.5-VL [1] as our base model and compare it with strong open-source and commercial MLLMs. Experimental results demonstrate that our method significantly enhances the utilization of visual information and enables logical reasoning over video content to predict future events, achieving state-of-the-art performance across various VEP benchmarks. Furthermore, we validate the superiority of our approach in open-set prediction scenarios through an evaluation with a judge model. Our main contributions are as follows:

- • We propose an effective video event prediction paradigm, Chain of Events, which addresses the challenges faced by existing MLLMs in video event prediction and significantly improves their accuracy in predicting future events.
- • We propose an efficient method to implement the CoE paradigm, which unlocks the MLLMs’ ability to construct temporal event chains and enables them to reason over the observed video to predict future events logically.
- • We establish one of the most comprehensive baselines to date for the VEP task through a systematic evaluation of our method and a wide range of MLLMs on this task, providing a solid foundation for future research in this area. Our experiments demonstrate that the proposed method

To enforce the model adhering to our proposed CoE paradigm, we introduce a two-stage training approach, CoE-SFT and CoE-GRPO, which facilitates model’s adaptation to the CoE framework and enhances video event prediction accuracy with modest training costs. In stage one training, CoE-SFT fine-tunes the model through supervised learning, enforcing the model to establish logical connections between historical video evidence and future events during the reasoning process, rather than serving merely as

effectively addresses the challenges faced by MLLMs in VEP, achieving SOTA performance across benchmarks.

#### 2. Related Works

##### 2.1. Video Event Prediction

The video event prediction (VEP) task was first introduced in [29], which requires the model to predict the next possible event based on the input video. Unlike other video reasoning tasks [3, 5, 6, 22, 32] focusing on the video content itself, VEP demands the model to infer unseen future content from current visible evidence, thereby posing higher requirements on model’s video understanding and logical reasoning capabilities. Previous works [16, 28, 29, 65] have shown that fine-grained temporal modeling of historical events is critical for accurately forecasting future events. Thus, the concept of Event Chains [16] has been widely adopted as an effective temporal representation paradigm in event modeling for both textual [11, 24, 27] and video event prediction tasks [29]. Recent works (VidEvent [29], AVEP [39], and NEP [43]) have analyzed the performance of MLLMs on VEP tasks, indicating that existing methods failed to achieve satisfactory results. However, no prior works have systematically investigated why MLLMs performed poorly in VEP tasks, nor have there been comprehensive evaluations or targeted methods to enhance their reasoning for future event prediction, particularly methods that enable large models to effectively model the evolution of historical events in videos.

##### 2.2. Visual Large Language Models for Reasoning

With the rapid advancement of MLLMs’ video understanding capabilities [1, 21, 35, 38, 46, 48, 60, 67] and LLMs’ reasoning abilities [9, 15, 17, 41, 55], recent studies have increasingly focused on exploring the reasoning capabilities of MLLMs [10, 12, 33, 36, 48, 50]. Several models, such as Qwen2.5-VL [1], GLM-4.1V [42], Kimi-VL [40], have been trained on various visual reasoning tasks, achieving competitive results and demonstrating great potential. Beyond supervised training, many works [4, 10, 20, 40, 55] have followed the GRPO approach proposed by DeepSeekR1 [9], leveraging RL to further enhance reasoning capabilities. For instance, Open-Reasoner [18], Kimi-VL [40], and Mimo [54] adopt similar RL pipelines to strengthen reasoning performance. Building upon GRPO, several works [2, 7, 8, 10, 12, 14, 26, 45, 49, 58, 59] have proposed adaptive modifications to further enhance the performance of MLLMs on visual reasoning tasks. However, these methods primarily focus on frame-level or local-region perception and are not tailored for event prediction. In the context of VEP, NEP [43] demonstrates that standard GRPO [9] training yields better performance than standard SFT. However, despite these promising advances, MLLMs’ performance

on VEP remains largely underexplored. And there is still a lack of targeted methods specifically designed to enhance their event prediction capabilities.

#### 3. Evaluation and Analysis of MLLMs on VEP

We conduct a systematic evaluation of various open-source and commercial MLLMs on the VEP task, as shown in Tabs. 1 and 2. The results indicate that current MLLMs do not exhibit the same strong performance on VEP as they do in other vision tasks. Among them, Qwen3-VL demonstrates the best performance across most metrics, yet the average accuracy is only 66.9%. The results suggest that MLLMs still have significant room for improvement in VEP, highlighting the research potential.

As shown in Fig. 1a, we visualized the model’s reasoning process and found that existing MLLMs generally follow a pattern: they first generate a high-level description of the video, then analyze each option, and finally select the most relevant option as the correct answer. This reasoning process lacks the logical connection between the video and future events, meaning the model does not truly reason about future events from the video but rather chooses the most relevant option. This often leads to incorrect predictions.

Additionally, as shown in Fig. 1a, we observed that the model tends to generate coarse-grained summaries of the video, which may cause it to overlook critical details relevant to future events and neglect the temporal dynamics underlying event evolution. Throughout the reasoning process, the utilization of visual information is significantly lower than that of textual information. We further investigate the attention distribution of MLLMs when performing the VEP task. Due to the causal attention mechanism, later tokens inherently contain information from earlier tokens. To avoid interference from this effect, we visualized the attention distribution specifically over the input option tokens, which also provides a fair comparison of attention patterns across different models by mitigating the influence caused by differences in generated tokens. As shown in Fig. 1b, we found that the attention distribution on visual tokens is much lower than that on textual tokens, indicating that the model fails to adequately focus on and utilize visual information when performing the VEP task.

Based on these experiments, we conclude that there is considerable room for improvement in the performance of MLLMs on VEP. The key challenges lie in addressing the lack of logical reasoning ability for future events and the insufficient utilization of visual information.

#### 4. Method

In the following section, we will provide a detailed overview of the CoE paradigm and how the CoE-SFT and CoE-GRPO are employed to implement it.

the specific nature of the event prediction task. Finally, the model’s prediction process can be expressed as:

[Figure 22]

[Figure 23]

|Video Description|
|---|

[Figure 24]

[Figure 25]

Logical Reasoning

[Figure 26]

|Future Event|
|---|

|Question|
|---|

P = P(Eˆ | V,Q,R′,EC). (2)

CoE-SFT Data

VL-72B

|Future Event|
|---|

The CoE paradigm addresses the limitations faced by MLLMs in VEP through two mechanisms: (i) a reasoning process that incorporates the logical connections between video content and the future event, and (ii) fine-grained temporal modeling via the construction of event chains.

"<think>\nThe video starts with a title card for a surfing event at Sandy Beach, followed by various clips of surfers riding waves. The sequence ends with a surfer in a blue shirt and black shorts riding a white surfboard, with 'SUBSCRIBE!' and 'KENUHAWAII.COM' branding, which suggests a promotional or concluding segment. \nOption A introduces…, which seems unrelated to the surfing theme and would disrupt the flow from the ocean scenes to the promotionnal content.\n\nOption B focuses…, maintaining the surfing theme but lacks the promotional elements seen in the final clip.\n\nOption C includes …, which adds a narrative element but again shifts away from the surfing action towards a more general beach setting.\n\nOption D presents …, which could be part of the surfing action but does not lead directly into the promotional content seen in the final clip.\n\nConsidering the need to smoothly transition from the surfing action to the promotional content, Option D provides a natural pause and sets up the uncertainty that can lead into the promotional segment.\n</think> \n\n<answer>D.</answer>

[Figure 27]

##### 4.2. CoE with Supervised Fine-Tuning

CoE-SFT Training

Unlike other video tasks, VEP requires models to predict unobservable future events based on the video content. This necessitates the model constructing a logical reasoning process between the observed video content and the unobserved future events. However, existing vanilla SFT data [43] is typically constructed by sequentially analyzing answer options, which fails to address the absence of a logical reasoning process. Consequently, despite fine-tuning on datasets of over 30K samples, the performance improvements remain limited [43]. To address this, following the CoE paradigm, we propose the CoE-SFT method, which focuses on constructing the logical connections between the video and future events during the reasoning process.

“<think>\nThe video starts with a title card for a surfing event at Sandy Beach, followed by various clips of surfers riding waves. The sequence ends with a surfer in a blue shirt and black shorts riding a white surfboard, with ‘SUBSCRIBE!’ and ‘KENUHAWAII.COM’ branding, which suggests a promotional or concluding segment. \nFrom the opening title card and the brand logos at the end of the video, it can be inferred that this video segment is likely a television commercial, and the brand logos at the end indicate that the advertisement has concluded. Meanwhile, the content of the advertisement features a surfer in a blue shirt and black shorts surfing on the ocean, which is a dynamic scene, so the broadcast of this advertisement is likely sports-related. This means that a scene transition is about to occur next, and considering that the previous scene was sports-related, the upcoming scene is more likely to be a relatively dynamic one as well. Option A correctly predicts the future event by introducing a clear thematic and visual shift from the energetic beach scenes to an urban backdrop featuring a sleek black sports car and corresponding song credit, which provides a logical bridge to the final clip.\n</think> \n\n<answer>A.</answer>

- Figure 2. An illustration of our proposed CoE-SFT method within Qwen2.5-VL-72B. We provide the larger model with the video and the future event, and prompt it to generate the intermediate logical reasoning process that connects them. Training on such data encourages the model to develop logical reasoning abilities rather than relying on option-based analysis.

Specifically, as shown in Fig. 2, we utilize a powerful large model, Qwen2.5-VL-72B, to construct a small-scale CoE-SFT training dataset. We provide the video, question, and correct future event to the model, and instruct it to output the logical reasoning process that derives the future event from the video content, while avoiding any analysis of other options. Following this, we perform a manual quality check to ensure data quality, achieving a pass rate of over 90%. It is worth noting that we did not construct the EC in the CoE-SFT data, as the quality of EC construction from large-scale MLLMs did not meet expectations and could potentially hinder the model’s training. However, we observed that the model still effectively retained its reasoning ability based on video content and achieved the expected results after CoE-GRPO training. We have provided examples of the model’s reasoning process in the Supplementary Material.

##### 4.1. Chain of Events (CoE) Paradigm

Previous works [34, 52] often use structured representations such as chains, trees, and graphs for video modeling. However, they are mostly action-centric and designed for localization or understanding tasks, while overly complex representations introduce unnecessary learning overhead for MLLMs. Therefore, we propose a CoE paradigm to model historical events in a fine-grained manner.

We define the model’s reasoning process as R = MLLMreason(V,Q), in which V denotes the input video, Q denotes the question. And the VEP process of the vanilla model is then expressed as:

P = P(Eˆ | V,Q,R), (1)

where Eˆ denotes the predicted event. In the CoE paradigm, we define an event E as a pair E = (T ,D), where T denotes the start and end timestamps of a video event and D denotes the textual description of the event. A temporal event chain EC, therefore, can be defined as a sequence of events occurring in the video, ordered temporally, EC = [E1,E2,...,En]. Consequently, this paradigm can be formalized as follows: The model first performs fine-grained temporal modeling of the video to construct the event chain EC = MLLMCoE(V ). Then, it reasons based on the video content and the event chain R′ = MLLMreason(V,Q,EC), where the R′ incorporates logical reasoning from video content to future events, given

To better assess the model’s logical reasoning ability and its performance in real-world applications, we propose an open-set judge model evaluation metric. In this evaluation, the judge model assesses both the validity of reasoning and the correctness of the answers, selecting the best response from multiple competing models and providing the reason behind its choice. The final win rate is then used as the evaluation metric.

##### 4.3. CoE with Group Relative Policy Optimization

The foundation of event prediction lies in temporal modeling of historical events [16, 28, 29, 65]. However, the insufficient utilization of visual information in MLLMs during

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

00:00 00:11 00:24 00:30

video

[Figure 39]

[Figure 40]

<event>Time:00:00-00:11, Des:xxx </event> <event>Time:00:11-00:24, Des:xxx </event> <event>Time:00:24-00:30, Des:xxx </event>

[Figure 41]

s

Event 1 Description

…What is most possible event happening after the video?

### Policy Model

Similarity

Model

[Figure 42]

[Figure 43]

[Figure 44]

s

…

Event 2 Description

Here are the options:

<think>……</think> <answer> A </answer>

s

[Figure 45]

[Figure 46]

CoE GRPO

[Figure 47]

Event N Description

A.xxx B.xxx C.xxx D.xxx

ra= 1 if oi = gt else 0 re= λI (oi) + (1−λ)[L− len(oi)−L+b]

- Figure 3. An illustration of our proposed CoE-GRPO method. The overall supervision signal consists of three components: re encourages the model to follow the CoE reasoning paradigm and constrains the CoE length; rs supervises the alignment between event timestamps and textual descriptions while preventing reward hacking; and ra provides verifiable reward signals. The scissor icon indicates the temporal segmentation of video clips based on timestamps.

where λ denotes the weight coefficient. In this, the indicator function I(oi) takes the value 1 if oi correctly contains all the required tag tokens , and 0 otherwise. Since both excessively long and short event chains hinder the model’s ability, according to our experiments, we introduce a length constraint term to control the length of the output event chain. The function len(oi) calculates the number of events in the event chain EC of completion oi. Here, L is a hyperparameter representing the model’s ideal output length, and b is a bias term used to ensure that the maximum value of re is 1.

event prediction hinders their ability to perform fine-grained temporal modeling, resulting in suboptimal performance.

To address this limitation, we propose the CoE-GRPO, an improved GRPO framework specifically designed for VEP. Our method effectively unlocks the model’s temporal localization and video understanding capabilities for constructing event chains, enabling fine-grained temporal modeling and improving the model’s utilization of visual information during event prediction.

Specifically, as illustrated in Fig. 3, we first introduce the special event tags <event> and </event> to explicitly mark the boundaries of event E within the model output. Each event tag pair contains the start and end timestamps of a corresponding event, tstart and tend, as well as a finegrained description D capturing its semantic details:

To ensure the consistency between the event descriptions and the video content in E, and to prevent the model from cheating by optimizing for rewards, we introduce a continuous similarity reward rs for supervision. Specifically, as shown in the Fig. 3, we crop the original video according to the start and end timestamps of each event E in the event chain EC output by the model, obtaining a set of video clips [clip1,clip2,...,clipn], which corresponds to the event chain. We then compute the cross-modal similarity sj between the event description and the video clips. The average of these similarity values is used as the similarity reward signal, ensuring that the model constructs event chains that align closely with the videos:

E = <event>Time:tstart − tend,Des:D</event>. (3)

During the CoT reasoning process, the model incrementally constructs a historical event chain EC consisting of multiple event tags organized in chronological order, which provides the visual grounding for subsequent logical reasoning steps. Since this relatively simple event representation method does not require additional data for cold start, we can directly employ reinforcement learning to train the model to construct event chains and leverage them for event prediction, as shown in Eq. (2).

n

1 n

sj, (5)

rs =

j=1

in which sj = cos(vj,tj), vj and tj are the visual and textual features of the event embedded by the similarity model. We use different similarity models and present their performance in the Experiment section, with the details of similarity calculation provided in the Supplementary Material.

To achieve this, we introduce a targeted, dense CoE reward re, which provides fine-grained supervision throughout the model’s event chain construction process, allowing control over both the correct construction and the length of the event chain:

Based on the aforementioned reward signals, the final reward is computed as the weighted sum of the individual

re(i) = λI(oi) + (1 − λ)[L − |len(oi) − L| + b]. (4)

reward components:

ri = αra(i) + βre(i) + (1 − α − β)rs(i), (6) where ra serves as the accuracy reward, α and β denote the reward weights. During training, we sample a group of N completions from the current policy πθ. For each completion, we compute a reward ri. The advantage Ai is then calculated by normalizing the rewards within the group:

ri − mean({ri}) std({ri}) + δ

, (7)

Ai =

where δ is a small constant for numerical stability. Following DeepSeek-R1 [9], the final policy update is as follows:

G

1 G

(min(rratioAi,clip(rratio,

J (θ) = Eq,{o

i}[

(8)

i=1

1 − ϵ,1 + ϵ)Ai) − βDKL(πθ||πref)],

θ(oi|q) πθold(oi|q).

where the importance sampling ratio rratio = π

CoE-GRPO can efficiently unlock the model’s temporal localization and video understanding capabilities, enabling fine-grained temporal modeling of historical videos through event chain construction. This enhances visual information utilization and improves event prediction accuracy. Additionally, the method leverages the model’s inherent capabilities without the need for additional data annotations, making it en efficient approach.

- 4.4. Training

We use Qwen2.5-VL-3B/7B as the base MLLMs. We adopt the two-phase training approach: CoE-SFT followed by CoE-GRPO. In the first phase, CoE-SFT, we train the model using the small-scale CoE-SFT reasoning dataset, enabling the model to logically reason from visual content to infer the potential future events. The resulting model is denoted as CoE-SFT. In the second phase, CoE-GRPO, we continue to train the model using the proposed method on the RL datasets across various benchmarks. This phase focuses on training the model to achieve fine-grained temporal modeling by constructing event chains. The resulting model is denoted as CoE-GRPO.

- 5. Experiments

- 5.1. Setup

Benchmarks and Metrics. We evaluate our method on public VEP benchmarks: FutureBench [43] and AVEP [39]. FutureBench measures the overall event prediction accuracy, where 1-HOP, 2-HOP, 3-HOP, and Interp. denote different prediction types, and AVG represents the overall average. AVEP provides an evaluation of the prediction accuracy for event components, including verbs and event participants. Verb denotes the accuracy of event verb prediction, while the Precision, Recall, and F1-Score of Noun and

×10−4 CoE-GRPO Vanilla GRPO

×10−4

[Figure 48]

[Figure 49]

𝛿𝛿DifferenceinAttentionScoresDifferenceinAttentionScores

8

2

6

1

4

0

2

−1

0 −2

Visual Token Sequence S Visual Token Sequence S

CoE-SFT Vanilla SFT

×10−4

×10−4

[Figure 50]

[Figure 51]

4 3 2 1 0

6

4

2

- −1
- −2

- −3

0

−2

Visual Token Sequence S Visual Token Sequence S

Figure 4. Attention difference of visual tokens between different methods and the base model. Portions greater than 0 indicate an improvement in attention.

Action respectively measure the precision, recall, and F1Score of event participants and overall event predictions. Since AVEP does not provide data for SFT, we do not include comparisons with vanilla SFT in our experiments on this benchmark. Details of these benchmarks and their evaluation metrics can be found in the Supplementary Material. Implementation Details. We train our models using up to 16 NVIDIA H20 GPUs. For efficiency considerations, we limit the maximum number of video frames to 32 and the max resolution to 128×28×28 during training. The GRPO group size G is set to 4, KL coefficient β is set to 0.04. The clipping parameter ϵ is set to 0.2, and the learning rate is set to 1e − 6. We train our model for 150 steps.

##### 5.2. Main Results

Results in Tabs. 1 and 2 indicate that CoE-GRPO consistently surpasses all baseline MLLMs on both benchmarks, delivering the best overall performance and highlighting the method’s effectiveness. CoE-SFT achieves superior performance relative to vanilla SFT [43], indicating that establishing logical connections between the video and future events helps improve prediction accuracy. CoE-GRPO also markedly outperforms vanilla GRPO [9, 43], validating that modeling historical event chains improves the model’s capacity to predict future events.

Moreover, we empirically demonstrate that the proposed method addresses key VEP limitations of existing MLLMs. As shown in Tab. 3 and Fig. 4, it substantially increases attention to visual tokens, whereas vanilla SFT [43] even reduces such attention. CoT [53] refers to using prompts to guide model reasoning. Qualitative results in the Supple-

Table 1. Evaluation results of open-source/commercial MLLMs and our proposed CoE on FutureBench [43].

Model Method Frames Futurebench ↑

1-Hop 2-Hop 3-Hop Interp. AVG GLM-4.1V-9B [42]

32 29.9 41.9 52.2 47.3 44.38 LLavA-NeXT-Video [61] 32 48.8 49.3 40.0 44.4 45.24 Kimi-VL-A3B [40] 32 44.3 42.8 51.3 51.9 48.87 MiMo-VL-7B [54] 32 59.0 59.6 50.5 43.8 50.45 InternVL3-8B [46] 32 54.3 58.0 63.2 54.4 56.72 Qwen2.5-VL-32B [1] 32 66.5 62.7 63.2 55.2 59.94 Qwen2.5-VL-72B [1] 32 55.5 68.4 63.7 53.2 58.33 Qwen3-VL-30B-A3B [41] 32 65.3 70.5 76.1 62.2 66.86

Vanilla

- GPT-4o [35] 32 61.9 61.7 72.1 51.6 59.04

- GPT-5 32 59.6 57.3 62.6 55.6 57.92

Instruct [1] 32 45.1 50.8 44.3 45.8 46.30 (NEP) SFT [43] 32 53.2 57.0 59.2 59.3 57.86 CoE-SFT (Ours) 32 68.8 75.1 55.7 60.5 63.60

Qwen2.5-VL-3B

(NEP) GRPO [43] 32 63.0 65.3 63.7 67.1 65.45 CoE-GRPO (Ours) 32 71.1 73.6 69.7 64.6 68.28

Instruct [1] 32 57.2 57.0 50.2 50.7 52.94 (NEP) SFT [43] 32 60.7 74.2 66.2 65.0 64.39 CoE-SFT (Ours) 32 67.6 74.1 62.2 63.2 65.72

Qwen2.5-VL-7B

(NEP) GRPO [43] 32 66.2 69.9 63.7 68.1 67.28 CoE-GRPO (Ours) 32 80.9 83.9 71.6 71.4 75.00

Table 2. Evaluation results of open-source MLLMs and our proposed CoE on AVEP [39].

Noun (Test / Val) ↑ Action (Test / Val) ↑ Precision Recall F1-Score Precision Recall F1-Score

Method Verb (Test / Val) ↑

LLaVA-Video-7B [63] 5.67 / 6.27 44.50 / 44.96 41.95 / 41.64 43.19 / 43.24 3.13 / 3.57 2.95 / 3.30 3.04 / 3.43 Kimi-VL-A3B 5.12 / 5.30 31.79 / 31.46 36.43 / 37.04 33.95 / 34.02 2.40 / 2.83 3.27 / 3.16 2.77 / 2.99 InternVL3-8B 5.80 / 7.00 48.77 / 43.64 45.77 / 37.94 47.22 / 40.59 3.37 / 3.60 3.23 / 3.07 3.30 / 3.31 GLM-4.1V-9B 9.10 / 10.62 33.80 / 34.94 33.66 / 35.42 33.73 / 35.18 3.01 / 3.27 3.02 / 3.34 3.01 / 3.30 MiMo-VL-7B 10.23 / 9.57 43.46 / 39.54 32.07 / 35.10 36.91 / 37.19 6.96 / 5.54 5.08 / 3.86 5.87 / 4.55 Qwen2.5-VL-7B 5.73 / 6.77 44.83 / 44.26 56.54 / 59.33 50.01 / 50.70 2.21 / 3.76 3.21 / 4.64 2.62 / 4.15 Qwen2.5-VL-72B 8.00 / 7.95 48.94 / 49.24 41.38 /41.57 44.84 / 45.08 5.32 / 5.38 4.25 / 4.41 4.72 / 4.85 Qwen2.5-VL-7B-GRPO [9] 9.64 / 10.42 47.79 / 47.35 90.07 / 91.35 62.45 / 63.49 5.00 / 5.80 9.21 / 9.72 6.48 / 7.26 CoE-SFT-7B (Ours) 9.84 / 11.44 45.57 / 43.18 54.74 / 59.02 49.74 / 49.87 4.94 / 5.72 5.14 / 6.27 5.04 / 5.98 CoE-GRPO-7B (Ours) 12.24 / 18.75 49.88 / 48.68 93.95 / 93.54 65.16 / 64.03 6.62 / 7.70 11.09 / 13.8 8.29 / 9.88

mentary Materials further illustrate the improved utilization of visual information and the accurate logical connection between video and future events during reasoning.

In Tab. 4, judge-based evaluation results based on Qwen2.5-VL-72B show that CoE-SFT obtains the highest win rate. The small deficit for CoE-GRPO relative to CoE-SFT reflects the judge model’s familiarity with SFTstyle reasoning rather than the CoE paradigm. Even so, their closely matched performance demonstrates that CoEGRPO effectively preserves the logical reasoning needed for event prediction. To ensure fairness, we manually inspected the judge model’s evaluation process. The complete judge outputs are provided in the Supplementary Material.

##### 5.3. Ablation Study

Visual Attention Enhancement Methods. As summarized in Tab. 5, we evaluate two common strategies for boosting visual attention. The first, Prompt-guided, instructs the

model via a prompt to produce a detailed video description. The second, Constant-Bias, adds a fixed value to the attention weights of visual tokens at inference. Despite testing multiple prompt formulations and bias magnitudes, both strategies yielded performance degradation.

Ablation of Group Size. The group size G is a hyperparameter that controls the number of rollouts during CoEGRPO. We evaluate the training results with different values of G, as shown in the Tab. 5. The results indicate that the model’s performance improves as G increases. However, an excessive number of rollouts leads to higher training costs. Therefore, we recommend setting it to 4 to achieve a favorable balance between performance and training cost.

Ablation of Event Chain Length. We explore different values of L to quantify the effect of event-chain length and granularity on prediction accuracy. The findings in Tab. 5 indicate a non-monotonic relationship: both extremes—too short and too long—are detrimental. Short chains fail to

- Table 3. Attention variation of visual tokens. WR (Winning Rate) denotes the overall percentage of samples showing attention improvement compared to the base model, while IR (Improvement Rate) indicates the average numerical increase in attention weight compared to the base model.

Method SFT CoT GRPO CoE-GRPO CoE-SFT Baseline Qwen2.5-VL-7B-Instruct

WR ↑ 0.32 0.44 0.59 0.77 0.93 IR(%) ↑ -3.33 +1.08 +1.47 +9.20 +15.11

- Table 4. The results of judge evaluation of different method. WR represents the ratio of samples where the model’s answers are more reasonable and accurate compared to other models.

Method CoE-SFT CoE-GRPO Instruct SFT GRPO WR(%) ↑ 38.13 32.42 16.21 7.88 5.37

Table 5. The results of the ablation studies.

Item 1-Hop 2-Hop 3-Hop Interp. AVG Visual Attention Enhancement Methods

Prompt-guided 44.5 46.6 43.8 46.6 45.74 Constant-Bias 54.6 52.3 57.6 50.6 52.57

CoE (Ours) 80.9 83.9 71.6 71.4 75.00 Group Size G

G = 2 57.8 64.8 60.7 59.9 60.61 G = 4 77.5 78.2 71.6 73.4 74.61 G = 8 78.6 80.3 74.1 76.7 77.20

Event Length L

L = 1 76.9 78.2 65.2 74.6 73.90 L = 3 77.5 78.2 71.6 73.4 74.61 L = 5 72.8 72.5 64.7 73.2 71.40

Similarity Model VideoCLIP-XL [44] 77.5 78.2 71.6 73.4 74.61

ViCLIP [47] 76.3 75.6 66.2 73.6 73.01 CLIP-large [37] 77.5 76.2 68.2 74.4 74.24

Similarity Reward

w/ rs 77.5 78.2 71.6 73.4 74.61 w/o rs 73.4 73.6 66.7 73.0 72.00

capture enough visual detail, while long chains introduce redundancy that complicates contextual reasoning.

Ablation of Similarity Model. We compare different schemes for computing the similarity reward rs. Specifically, we evaluate the use of video-text similarity models, VideoCLIP-XL and ViCLIP, as well as the image-text similarity model CLIP. The comparable performance across these models demonstrates the robustness of our proposed approach, with VideoCLIP-XL yielding the best results.

Ablation of Similarity Reward. To test the impact of the similarity reward rs on model performance, we conducted training without the similarity reward signal. The exper-

CoE-GRPO Training Curves

100

80

CoE reward

similarity reward accuracy reward

60

Rewards

40

20

0 20 40 60 80 100 120 140 Training Steps

Figure 5. The training curves of CoE-GRPO.

imental results show a noticeable decline in all metrics, demonstrating the effectiveness of the similarity reward.

##### 5.4. Training Curves

As shown in Fig. 5, the accuracy reward ra exhibits a generally upward trend, indicating that the model continuously improves its event prediction ability under the training strategy. As for the curve of CoE reward re, the rapid increase observed during the initial stages suggests that the model learns the conceptual framework for constructing event chains. The subsequent steady ascent indicates that the model is gradually approximating the target event chain length defined by parameter L. The consistent increase in similarity reward rs indicates a progressive enhancement in the alignment between the event descriptions and the corresponding video segments. Notably, we found that even highly accurate descriptions yield similarity scores only in the range of 0.2 − 0.3, indicating that the rs fall within a normal and reasonable range.

#### 6. Conclusion

Video event prediction holds significant practical value, yet research on MLLMs in this domain remains limited. In this work, we present the first evaluation of various MLLMs on the video event prediction task, establishing comprehensive baselines. Through experiments, we reveal the reasons behind the inaccurate predictions in VEP, including the lack of logical reasoning ability for future events and insufficient utilization of visual information. To address these challenges, we propose a Chain of Events (CoE) prediction paradigm, which unlocks the MLLMs’ ability to construct temporal event chains and enables them to logically reason over the observed video to predict future events. Extensive experiments show that our proposed method can effectively overcome the challenges encountered by MLLMs in VEP, achieving state-of-the-art performance across existing

benchmarks. We hope this work lays the foundation for future research on MLLMs to video event prediction.

#### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 1, 2, 3, 7
- [2] Sule Bai, Mingxing Li, Yong Liu, Jing Tang, Haoji Zhang, Lei Sun, Xiangxiang Chu, and Yansong Tang. Univg-r1: Reasoning guided universal visual grounding with reinforcement learning. arXiv preprint arXiv:2505.14231, 2025. 3
- [3] Mu Cai, Reuben Tan, Jianrui Zhang, Bocheng Zou, Kai Zhang, Feng Yao, Fangrui Zhu, Jing Gu, Yiwu Zhong, Yuzhang Shang, Yao Dou, Jaden Park, Jianfeng Gao, Yong Jae Lee, and Jianwei Yang. Temporalbench: Benchmarking fine-grained temporal understanding for multimodal video models, 2024. 3
- [4] Qiguang Chen, Libo Qin, Jinhao Liu, Dengyun Peng, Jiannan Guan, Peng Wang, Mengkang Hu, Yuhang Zhou, Te Gao, and Wanxiang Che. Towards reasoning era: A survey of long chain-of-thought for reasoning large language models,

2025. 3

- [5] Rui Chen, Lei Sun, Jing Tang, Geng Li, and Xiangxiang Chu. Finger: Content aware fine-grained evaluation with reasoning for ai-generated videos. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 3517–3526,

2025. 3

- [6] Yi Chen, Yuying Ge, Rui Wang, Yixiao Ge, Lu Qiu, Ying Shan, and Xihui Liu. Exploring the effect of reinforcement learning on video understanding: Insights from seed-benchr1, 2025. 3
- [7] Xiangxiang Chu, Hailang Huang, Xiao Zhang, Fei Wei, and Yong Wang. GPG: A simple and strong reinforcement learning baseline for model reasoning. In The Fourteenth International Conference on Learning Representations, 2026. 3
- [8] Jisheng Dang, Jingze Wu, Teng Wang, Xuanhui Lin, Nannan Zhu, Hongbo Chen, Wei-Shi Zheng, Meng Wang, and Tat-Seng Chua. Reinforcing video reasoning with focused thinking. arXiv preprint arXiv:2505.24718, 2025. 3
- [9] DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. 3, 6, 7
- [10] Yue Fan, Xuehai He, Diji Yang, Kaizhi Zheng, Ching-Chen Kuo, Yuting Zheng, Sravana Jyothi Narayanaraju, Xinze Guan, and Xin Eric Wang. Grit: Teaching mllms to think with images, 2025. 3
- [11] Zhiyu Fang, Shuai-Long Lei, Xiaobin Zhu, Chun Yang, ShiXue Zhang, Xu-Cheng Yin, and Jingyan Qin. Transformerbased reasoning for learning evolutionary chain of events on temporal knowledge graph. In Proceedings of the 47th International ACM SIGIR Conference on Research and Develop-

- ment in Information Retrieval, page 70–79, New York, NY, USA, 2024. Association for Computing Machinery. 3
- [12] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025. 3
- [13] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu, Xiawu Zheng, Enhong Chen, Caifeng Shan, Ran He, and Xing Sun. Videomme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24108–24118, 2025. 1
- [14] Shenghao Fu, Qize Yang, Yuan-Ming Li, Xihan Wei, Xiaohua Xie, and Wei-Shi Zheng. Love-r1: Advancing long video understanding with an adaptive zoom-in mechanism via multi-step reasoning, 2025. 3
- [15] Team GLM. Chatglm: A family of large language models from glm-130b to glm-4 all tools, 2024. 3
- [16] Mark Granroth-Wilding and Stephen Clark. What happens next? event prediction using a compositional neural network model. In Proceedings of the AAAI Conference on Artificial Intelligence, 2016. 1, 3, 4
- [17] Tingxu Han, Zhenting Wang, Chunrong Fang, Shiyu Zhao, Shiqing Ma, and Zhenyu Chen. Token-budget-aware llm reasoning. In Findings of the Association for Computational Linguistics: ACL 2025, pages 24842–24855, 2025. 3
- [18] Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model, 2025. 3
- [19] Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Yuanhan Zhang, Xiang Yue, Bo Li, and Ziwei Liu. Video-mmmu: Evaluating knowledge acquisition from multi-discipline professional videos. 2025. 1
- [20] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models, 2025. 3
- [21] Zhe Huang, Hao Wen, Aiming Hao, Bingze Song, Meiqi Wu, Jiahong Wu, Xiangxiang Chu, Sheng Lu, and Haoqian Wang. Taming hallucinations: Boosting mllms’ video understanding via counterfactual video generation. arXiv preprint arXiv:2512.24271, 2025. 3
- [22] Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 1988–

1997. IEEE Computer Society, 2017. 3

- [23] Seil Kang, Jinyeong Kim, Junhyeok Kim, and Seong Jae Hwang. See what you are told: Visual attention sink in large multimodal models. arXiv preprint arXiv:2503.03321, 2025. 2

- [24] Anton Klenitskiy, Artem Fatkulin, Daria Denisova, Anton Pembek, and Alexey Vasilev. Encode me if you can: Learning universal user representations via event sequence autoencoding. In Proceedings of the Recommender Systems Challenge 2025, page 26–30, New York, NY, USA, 2025. Association for Computing Machinery. 3
- [25] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Lou, Limin Wang, and Yu Qiao. Mvbench: A comprehensive multi-modal video understanding benchmark. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22195–22206, 2024. 1
- [26] Renda Li, Hailang Huang, Fei Wei, Feng Xiong, Yong Wang, and Xiangxiang Chu. Adacurl: Adaptive curriculum reinforcement learning with invalid sample mitigation and historical revisiting. AAAI, 2026. 3
- [27] Zhongyang Li, Xiao Ding, and Ting Liu. Constructing narrative event evolutionary graph for script event prediction. In Proceedings of the 27th International Joint Conference on Artificial Intelligence, page 4201–4207. AAAI Press, 2018. 3
- [28] Zhongyang Li, Xiao Ding, and Ting Liu. Constructing narrative event evolutionary graph for script event prediction. arXiv preprint arXiv:1805.05081, 2018. 1, 3, 4
- [29] Baoyu Liang, Qile Su, Shoutai Zhu, Yuchen Liang, and Chao Tong. Videvent: A large dataset for understanding dynamic evolution of events in videos, 2025. 1, 3, 4
- [30] Fenglin Liu, Tingting Zhu, Xian Wu, Bang Yang, Chenyu You, Chenyang Wang, Lei Lu, Zhangdaihong Liu, Yefeng Zheng, Xu Sun, et al. A medical multimodal large language model for future pandemics. NPJ Digital Medicine, 6(1): 226, 2023. 1
- [31] Shi Liu, Kecheng Zheng, and Wei Chen. Paying more attention to image: A training-free method for alleviating hallucination in lvlms. In European Conference on Computer Vision, pages 125–140, 2024. 2
- [32] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? arXiv preprint arXiv: 2403.00476, 2024. 1, 3
- [33] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visualrft: Visual reinforcement fine-tuning, 2025. 3
- [34] Trong-Thuan Nguyen, Pha Nguyen, Jackson Cothren, Alper Yilmaz, and Khoa Luu. Hyperglm: Hypergraph for video scene graph generation and anticipation, 2025. 4
- [35] OpenAI. Gpt-4o system card, 2024. 1, 3, 7
- [36] Yi Peng, Peiyu Wang, Xiaokun Wang, Yichen Wei, Jiangbo Pei, Weijie Qiu, Ai Jian, Yunzhuo Hao, Jiachun Pan, Tianyidan Xie, Li Ge, Rongxian Zhuang, Xuchen Song, Yang Liu, and Yahui Zhou. Skywork r1v: Pioneering multimodal reasoning with chain-of-thought, 2025. 3
- [37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 8

- [38] Yan Shu, Zheng Liu, Peitian Zhang, Minghao Qin, Junjie Zhou, Zhengyang Liang, Tiejun Huang, and Bo Zhao. Video-xl: Extra-long vision language model for hour-scale video understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26160–26169,

2025. 3

- [39] Qile Su, Shoutai Zhu, Shuai Zhang, Baoyu Liang, and Chao Tong. EventFormer: A Node-graph Hierarchical Attention Transformer for Action-centric Video Event Prediction, page 4698–4707. Association for Computing Machinery, New York, NY, USA, 2025. 3, 6, 7
- [40] Kimi Team. Kimi-VL technical report, 2025. 1, 3, 7
- [41] Qwen Team. Qwen3 technical report, 2025. 3, 7
- [42] V Team. Glm-4.5v and glm-4.1v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning,

2025. 1, 3, 7

- [43] Haonan Wang, Hongfu Liu, Xiangyan Liu, Chao Du, Kenji Kawaguchi, Ye Wang, and Tianyu Pang. Fostering video reasoning via next-event prediction, 2025. 1, 3, 4, 6, 7
- [44] Jiapeng Wang, Chengyu Wang, Kunzhe Huang, Jun Huang, and Lianwen Jin. Videoclip-xl: Advancing long description understanding for video clip models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 16061–16075, 2024. 8
- [45] Qi Wang, Yanrui Yu, Ye Yuan, Rui Mao, and Tianfei Zhou. Videorft: Incentivizing video reasoning capability in mllms via reinforced fine-tuning, 2025. 3
- [46] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. 1, 3, 7
- [47] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942, 2023. 8
- [48] Yu Wang, Yi Wang, Rui Dai, Yujie Wang, Kaikui Liu, Xiangxiang Chu, and Yansheng Li. Urban socio-semantic segmentation with vision-language reasoning. In The Fourteenth International Conference on Learning Representations, 2026. 3
- [49] Zikang Wang, Boyu Chen, Zhengrong Yue, Yi Wang, Yu Qiao, Limin Wang, and Yali Wang. Videochat-a1: Thinking with long videos by chain-of-shot reasoning. arXiv preprint arXiv:2506.06097, 2025. 3
- [50] Zikang Wang, Boyu Chen, Zhengrong Yue, Yi Wang, Yu Qiao, Limin Wang, and Yali Wang. Videochat-a1: Thinking with long videos by chain-of-shot reasoning, 2025. 3
- [51] Zhenting Wang, Shuming Hu, Shiyu Zhao, Xiaowen Lin, Felix Juefei-Xu, Zhuowei Li, Ligong Han, Harihar Subramanyam, Li Chen, Jianfa Chen, et al. Mllm-as-a-judge for image safety without human labeling. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 14657–14666, 2025. 1
- [52] Ziyang Wang, Shoubin Yu, Elias Stengel-Eskin, Jaehong Yoon, Feng Cheng, Gedas Bertasius, and Mohit Bansal.

- Videotree: Adaptive tree-based video representation for llm reasoning on long videos, 2025. 4
- [53] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. 6
- [54] LLM-Core-Team Xiaomi. Mimo-vl technical report, 2025. 1, 3, 7
- [55] Tian Xie, Zitian Gao, Qingnan Ren, Haoming Luo, Yuqian Hong, Bryan Dai, Joey Zhou, Kai Qiu, Zhirong Wu, and Chong Luo. Logic-rl: Unleashing llm reasoning with rulebased reinforcement learning, 2025. 3
- [56] Weicheng Xing, Tianqing Zhu, Jenny Wang, and Bo Liu. A survey on mllms in education: application and future directions. Future Internet, 2024. 1
- [57] Jihan Yang, Shusheng Yang, Anjali W. Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10632–10643, 2025. 1
- [58] Zhenlong Yuan, Chengxuan Qian, Jing Tang, Rui Chen, Zijian Song, Lei Sun, Xiangxiang Chu, Yujun Cai, Dapeng Zhang, and Shuo Li. Autodrive-r²: Incentivizing reasoning and self-reflection capacity for VLA model in autonomous driving. In The Fourteenth International Conference on Learning Representations, 2026. 3
- [59] Zhenlong Yuan, Xiangyan Qu, Chengxuan Qian, Rui Chen, Jing Tang, Lei Sun, Xiangxiang Chu, Dapeng Zhang, Yiwei Wang, Yujun Cai, and Shuo Li. Video-STAR: Reinforcing open-vocabulary action recognition with tools. In The Fourteenth International Conference on Learning Representations, 2026. 3
- [60] Zhenlong Yuan, Xiangyan Qu, Jing Tang, Rui Chen, Lei Sun, Ruidong Chen, Hongwei Yu, Chengxuan Qian, Xiangxiang Chu, Shuo Li, et al. What if agents could imagine? reinforcing open-vocabulary hoi comprehension through generation. arXiv preprint arXiv:2602.11499, 2026. 3
- [61] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llavanext: A strong zero-shot video understanding model, 2024. 1, 7
- [62] Yuechen Zhang, Shengju Qian, Bohao Peng, Shu Liu, and Jiaya Jia. Prompt highlighter: Interactive control for multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13215– 13224, 2024. 2
- [63] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Llava-video: Video instruction tuning with synthetic data, 2025. 7
- [64] Yilun Zhao, Haowei Zhang, Lujing Xie, Tongyan Hu, Guo Gan, Yitao Long, Zhiyuan Hu, Weiyuan Chen, Chuhan Li, Zhijian Xu, Chengye Wang, Ziyao Shangguan, Zhenwen Liang, Yixin Liu, Chen Zhao, and Arman Cohan. Mmvu: Measuring expert-level multi-discipline video understanding. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8475–8489, 2025. 1

- [65] Fangqi Zhu, Jun Gao, Changlong Yu, Wei Wang, Chen Xu, Xin Mu, Min Yang, and Ruifeng Xu. A generative approach for script event prediction via contrastive fine-tuning. Proceedings of the AAAI Conference on Artificial Intelligence, 37(11):14056–14064, 2023. 3, 4
- [66] Lanyun Zhu, Deyi Ji, Tianrun Chen, Peng Xu, Jieping Ye, and Jun Liu. Ibd: Alleviating hallucinations in large visionlanguage models via image-biased decoding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1624–1633, 2025. 2
- [67] Orr Zohar, Xiaohan Wang, Yann Dubois, Nikhil Mehta, Tong Xiao, Philippe Hansen-Estruch, Licheng Yu, Xiaofang Wang, Felix Juefei-Xu, Ning Zhang, et al. Apollo: An exploration of video understanding in large multimodal models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18891–18901, 2025. 3

## Video-CoE: Reinforcing Video Event Prediction via Chain of Events Supplementary Material

#### A. Additional Results

We additionally include comparisons with RL-based methods and traditional state-of-the-art approaches.

Table 6. Additional results, *denotes traditional method.

FutureBench ↑ AVEP ↑

Model

AVG Verb Noun-F1 Action-F1 VideoChat-R1 46.59 8.31 42.22 3.95

Video-R1 67.47 9.47 47.04 4.32 Ours 75.00 18.75 64.03 9.88

EventFormer* - 22.71 46.24 7.69

#### B. Attention Score Calculation Process

To quantitatively evaluate the MLLMs’ attention to visual information during the video event prediction task, we compute the attention weights assigned to both visual and text tokens. We record the attention score matrices of every head at every layer during inference on the test set. Since existing open-source MLLMs contain a large number of layers and heads, and we did not observe notable differences across them in our experiments, we visualize the average attention scores aggregated over all LLM layers and heads. To control for the variation in the number of tokens generated by different models, we visualize the attention score distribution from option tokens to all other tokens. This provides a direct insight into how each model allocates attention between visual and textual information during event prediction, while also enabling an unbiased comparison of their visual attention.

#### C. Similarity Reward Computation

For an event chain of length n, where the video segments are represented as [clip1,clip2,...,clipn] and the corresponding descriptions as [D1,D2,...,Dn], we use the similarity model fθ(·) to embed the video feature v and text feature t of the event chain:

(Dj). (9)

vj = fθ

(clipj),tj = fθ

text

visual

Thus, the similarity reward rs is obtained by averaging the similarities between the video and text features of the events:

n

1 n

(sim(vj,tj)), (10)

rs =

j=1

where sim(·) denotes the similarity computation function, which typically refers to cosine similarity.

In the experiments described in the main paper, we explore two different approaches for calculating the similarity reward. The first method directly uses a video-text alignment model to compute the video features of cropped video event segments and the text features of their corresponding descriptions. For this approach, we follow the official recommendation and use a frame rate of 8 frames for sampling. The second method employs an image-text alignment model, where we extract image frames from the video segments, calculate the similarity between each image feature and the text feature, and then average these similarity scores to obtain the overall similarity between the video event segment and its textual description. We use a frame rate of 8 frames for sampling as well.

#### D. Video Event Prediction Benchmarks

Futurebench. Futurebench is a benchmark specifically designed to evaluate the video event prediction capability of MLLMs, featuring both SFT and GRPO training datasets. This benchmark collects video data from various perspectives, different lengths, and types, offering a comprehensive assessment of event prediction performance across diverse scenarios. To evaluate the event prediction ability of MLLMs from multiple angles, Futurebench categorizes the event prediction tasks into four types:

- • 1-Hop: The model predicts a single future event that directly links the observed scenes to the final one, corresponding to a standard Next Event Prediction (NEP).
- • 2-Hop: The model infers a sequence of two consecutive future events, requiring a short chain reasoning process that sequentially connects the observed scenes to the final event.
- • 3-Hop: The model predicts three consecutive future events, significantly increasing task complexity by necessitating deeper causal reasoning across a longer temporal span.
- • Internp.: The model must infer multiple non-consecutive future events, given a set of partially observed scenes that include intermediate anchor events.

In this benchmark, we construct 2,000 CoE-SFT samples for CoE-SFT training and additionally utilize the provided 2,000 reinforcement learning samples as training data for CoE-GRPO. As the training set contains only 1-Hop and

- 2-Hop event prediction samples, performance gains on the
- 3-Hop and Interp. metrics provide a reliable measure of the model’s generalization capability on VEP task.

AVEP. Action-centric Video Event Prediction (AVEP) is a benchmark specifically designed for evaluating the video

CoE Training Data Scaling

75

70

AverageAccuracy

65

60

55

50

0 250 500 750 1000 1250 1500 1750 2000 Data Samples

Figure 6. The data scaling curve of CoE

event prediction capabilities of models. This benchmark primarily focuses on the events themselves, providing a comprehensive and fine-grained assessment of the model’s performance on video event prediction. Video events are decomposed into event arguments, and the model’s ability to predict future events is evaluated at the argument level. The evaluation of a model’s event prediction performance on AVEP is based on the following key aspects:

- • Verb Accuracy: This metric measures the accuracy of the model in predicting the trigger verb of future events. As verbs are the core triggers of events, they are crucial components in event construction, and their prediction reflects the model’s logical reasoning ability.
- • Noun Metrics: This set of metrics assesses the model’s ability to predict the participants in future events, including the subject, object, and tool. These metrics reflect the model’s consistency in role prediction, evaluating the logical coherence of the event arguments.
- • Action Metrics: These metrics evaluate the noun prediction performance when the verb is correctly predicted, providing a direct indication of the model’s video event prediction ability.

The AVEP dataset provides the ground truth but does not include the SFT data required for training MLLMs. Therefore, we use the 5,000 constructed CoE-SFT samples for supervised fine-tuning and experiments. In this benchmark, we select 5,000 samples from the provided dataset as the training data for GRPO and CoE-GRPO, and evaluate the model on the entire validation and test sets of the benchmark.

#### E. Training Data Scaling of CoE

Using the benchmark-provided training data, we train the model to follow the CoE paradigm, and observe strong im-

provements on VEP task, highlighting the data efficiency of our method.

To further investigate the effect of data scale, we conduct a data-scaling study. As shown in the Fig. 6, directly applying the CoE paradigm without any CoE-specific training leads to a performance drop, indicating that the model is unable to effectively leverage visual information to construct the logical connections to future events. When a small amount of training data is provided, performance begins to improve steadily, suggesting that the model is gradually acquiring the CoE reasoning pattern. As the data size increases, performance improves rapidly, demonstrating the strong data efficiency of our approach. Notably, although we train the model on only 2,000 samples from FutureBench, the upward trend in the scaling curve remains far from saturated. This indicates that additional data would likely yield further gains, highlighting the strong scalability and continued potential of our proposed method .

#### F. Details of Judge Model Evaluation

To more accurately assess MLLMs’ event prediction capabilities in real-world applications, the judge model evaluation is designed to reflect an open-set prediction setting. Specifically, we remove all answer options and require the model to directly reason about the observed video and predict the most plausible future event. In the evaluation, we provide the judge model with the video, question, and reference answer. The judge model then evaluates each output from two perspectives: (i) the logical consistency and soundness of the reasoning, and (ii) the correctness of the predicted event. Directly scoring model outputs may lead to inconsistencies or hallucination from the judge model, undermining evaluation fairness. To address this, we employ a group-wise comparison protocol and report the win rate as the evaluation metric, which yields a more reliable assessment.

We provide a comprehensive example of the judge model evaluation, as shown in Fig. 7. In this evaluation, it is observed that the models trained with the CoE-SFT method produce reasoning processes that are more closely aligned with the video content, and their reasoning is visually grounded, clear, and concise. In contrast, other methods either fail to focus on the visual content during reasoning or provide predictions that lack logical consistency. In other test samples, we find that both CoE-SFT and CoE-GRPO methods are able to deliver reasonable reasoning processes and accurate predictions in the absence of options.

#### G. Examples of CoE

As illustrated in Figs. 8 to 13, we present several randomly selected prediction examples generated by our proposed method. From the constructed event chains in these results

(marked as blue in the figures), it can be observed that our approach enables the model to perform fine-grained temporal modeling of the input videos and efficiently improves the utilization of viusal information. Notably, as the overall video duration varies, the granularity of event segmentation adaptively adjusts, while the length of the generated event chains remains relatively stable. In addition, the textual descriptions produced for each event are generally consistent with the corresponding video segments.

The followed reasoning process (marked as green in the figures) exhibited in these examples demonstrates that the model can logically infer future events based on the details present in the video. This not only shows that our proposed method enables the model to effectively establish logical connections between the video and future events, but also indicates that after CoE-GRPO training, the model retains the logical reasoning capabilities learned during CoE-SFT.

However, as illustrated in the Figs. 14 and 15, we also observed some bad cases, though their occurrence is extremely rare—approximately three instances out of one thousand samples. In these cases, the model fails to generate timestamps correctly according to the given instructions. Nevertheless, this issue has minimal impact on the model’s event descriptions and prediction results.

#### H. Prompt Templates

Fig. 17 illustrates the prompt template we use for training and inference of all MLLMs. We also present the prompt used for CoE-SFT data generation and judge model evaluation in Fig. 18

#### I. Visualizations of Attention Increase in Visual Tokens

Here are some visualizations of the attention differences between the post-trained models and the vanilla model (Qwen2.5-VL-7B-Instruct) for visual tokens, as shown in Fig. 16. The portions of the curves above 0 indicate an increase in attention. It is clear that both the CoE-GRPO and CoE-SFT methods effectively enhance the model’s focus on visual information. However, while both vanilla GRPO and vanilla SFT improve the model’s event prediction performance, they fail to adequately address the issue of insufficient visual information utilization, thus limiting the overall efficiency of performance improvement. Additionally, when we directly prompt the model to predict future events in a CoT manner without training, it does not significantly improve the model’s attention to visual content.

#### J. Limitations and Future Works

We hope that this work can serve as a foundation and provide inspiration for future explorations of MLLMs’ capabilities in video event prediction. Below, we also summarize

the limitations and outline several potential directions for future research:

- • Temporal Localization Ability. Our proposed method imposes certain requirements on the model’s temporal localization capability. Although most current MLLMs demonstrate strong temporal localization performance, there is still considerable room for improvement in accuracy. We believe that further enhancement of this capability could lead to additional performance gains for our method.
- • Structures of Historical Events. We explored the use of event-chain construction to strengthen the model’s fine-grained temporal modeling of videos. However, we only investigated relatively basic forms of event chains. More complex formulations for modeling historical event structures—such as relation-aware event chains or event graphs—remain promising directions for future research.
- • Perfomance on Other MLLMs and Tasks. Since our approach does not impose specific requirements on the base model, we believe that the CoE paradigm should be applicable to all MLLMs. However, due to resource constraints, we were unable to conduct experiments across all available models. Nevertheless, the exceptional performance of our method on VEP demonstrates its potential for application to other tasks as well. Exploring the applicability of CoE to other video-related tasks presents an interesting avenue for future research.

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Question: “Based on the given video, predict the sequence of future events and fill in the missing event slot with the next major scene that would lead into the final thank you and engagement prompts: 1. [?] 2. A red-curtain sequence displaying messages such as ‘Please Subscribe!’, ‘Thumbs Up’, and ‘Leave a Comment’. Which of the following potential events best fits the gap represented by [?]?” Options.

CoE-GRPO:

<event>Time:……, Des:…….</event>\n<event>Time:……, Des:…….</event>\n<event>Time:……, Des:…….</event>\n <think>Given the sequence of events, it is clear that the video is structured to build engagement with the audience. The initial scenes set up the context and introduce the main subject, followed by a series of frames encouraging viewer interaction. The next logical step in this sequence would be to transition from the interactive content to a closing segment where the creator thanks the viewers and encourages them to take action, such as subscribing or leaving comments.</think> The potential event that best fits the gap represented by [?] is a scene where the person concludes their presentation or demonstration, thanking the viewers for watching and encouraging them to subscribe, like, and comment. This would naturally lead into the red-curtain sequence displaying messages such as 'Please Subscribe!', 'Thumbs Up', and 'Leave a Comment'.

GRPO:

<think>\nThe video appears to be structured around a typical YouTube video format where the creator engages with their audience through various scenes before concluding with a call-to-action. Given the context provided, the missing event slot [?] should logically follow the initial setup or introduction of the content but precede the red-curtain sequence that encourages viewer interaction.\n\nA common structure for YouTube videos includes an opening scene that sets the tone or introduces the topic, followed by a main content segment, and then a closing segment that includes calls-to-action. Since the red-curtain sequence is designed to engage viewers after the main content, the missing event slot [?] should likely be a scene that transitions from the main content to the closing segment.\n\nConsidering this, the most fitting event for the gap would be a scene that summarizes the key points discussed in the main content or provides a brief recap. This would help prepare the audience for the closing segment and ensure they understand the value of the content before being prompted to subscribe, like, or comment.\n\n</think> The missing event slot [?] should be filled with a summary or recap of the main content.

CoE-SFT:

<think> The video transitions from a hookah demonstration to a driving scene at night, suggesting a shift in location or context. This could indicate a new segment or a transition to a different setting before concluding with the final thank you and engagement prompts. Therefore, the next major scene likely involves a return to the hookah theme or a related activity, fitting logically before the closing messages. </think>. A person preparing hookah tobacco and charcoal, followed by a scene showing the preparation of the hookah itself.

SFT:

<think> Event 4 (the red curtain sequence) is already provided in the final scene, so the missing event must occur between the observed scenes and this final segment. Given the narrative progression from the initial setup to the final call-to-action, the most logical intermediate event is the introduction of the product or service being promoted. This aligns with the observed scenes where the man is seen in various settings, including a kitchen and a living room, which suggests he is demonstrating or discussing something related to the product. The red curtain sequence then serves as a transition to the final call-to-action. </think>. The missing event is the demonstration or discussion of the product/service, which logically bridges the observed scenes and the final red curtain sequence.

Instrcut:

<think>\nThe video starts with a person in a kitchen setting, possibly preparing food or cooking. This is followed by a transition to a black screen with text, which seems to be an introduction or title card. After this, there's a scene with a person in a different setting, possibly a studio or a room with a desk, where they appear to be speaking or presenting something. The sequence ends with a red-curtain sequence displaying messages encouraging viewers to subscribe, like, and comment.\n\nGiven the structure of the video, it's logical to assume that the missing event slot [?] should follow the initial kitchen scene but precede the studio or presentation scene. This could be another segment where the person continues their activity, perhaps explaining a recipe or demonstrating a cooking technique. This would provide a smooth transition from the kitchen to the more formal presentation style seen later.\n</think> The missing event slot [?] should be a continuation of the kitchen scene, possibly showing the person preparing or demonstrating a specific step in the cooking process. This would logically lead into the subsequent scenes where the person moves to a more formal setting for a presentation or explanation.\n

Judge Model：

CoE-GRPOassumes the video follows a typical YouTube format with an introduction, main content, and a conclusion. It suggests the missing event is a summary or recap of the main content. The assumption about the video's structure is reasonable, but it doesn't consider the specific content shown in the video, such as the hookah demonstration and driving scene. GRPOalso assumes a typical YouTube video structure and suggests the missing event is a summary or recap of the main content. Similar to EGRPO, the reasoning is plausible but lacks specificity regarding the actual content of the video. CoE-SFTconsiders the transition from the hookah demonstration to the driving scene and suggests the missing event involves further preparation of the hookah. This reasoning is more aligned with the actual content of the video, as it directly addresses the hookah-related activities shown. SFTassumes the video is promoting a product or service and suggests the missing event is a demonstration or discussion of the product. While this reasoning is somewhat relevant, it does not specifically address the hookah-related activities depicted in the video. Instructassumes the video is about cooking and suggests the missing event is a continuation of the kitchen scene. This reasoning is not accurate, as the video clearly focuses on hookah-related activities rather than cooking. Best Model: CoE-SFT. Reason: CoE-SFT provides the most accurate and relevant response by considering the specific content of the video, particularly the hookah demonstration and the transition to the driving scene. It correctly identifies that the missing event should involve further preparation of the hookah, which logically fits the sequence of events shown in the video."

Figure 7. An example of judge model evaluation.

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

###### Input Video

###### Question:

"Based on the given video, predict future events and fill in the blanks in the following sequence: 1. A group of people gathers around snow-covered cars in a parking lot under streetlights. 2. [?] 3. An exterior view of a factory with a Russian sign indicating deep meat processing and production is shown with snow surrounding it. 4. [?] 5. A close-up shot of food being served with the caption 'Food at the factory is free' is displayed. Which option best completes the missing events with the correct intermediate scenes that occur after the observed scenes?”

###### Options:

- A. 2. An interior view showing a car’s dashboard and steering wheel as it drives on a snowy road, 4. A scene inside a locker room where a person adjusts their clothing in preparation for work.
- B. 2. A scene inside a locker room where a person adjusts their clothing in preparation for work, 4. An interior view showing a car’s dashboard and steering wheel as it drives on a snowy road.
- C. 2. A display of a white truck with 'АГРОХОЛДИНГ' in red letters parked in a snowy area, 4. A scene inside a locker room where a person adjusts their clothing in preparation for work.
- D. 2. An interior view showing a car’s dashboard and steering wheel as it drives on a snowy road, 4. A display of a white truck with 'АГРОХОЛДИНГ' in red letters parked in a snowy area.

CoE:

<event>Time:00:00-00:05, Des: A man in a red shirt speaks directly to the camera.</event> <event>Time:00:05-00:10, Des: Two individuals in protective gear are shown working in a factory environment.</event> <event>Time:00:10-00:15, Des: Text appears on screen asking if working for 30,000 rubles per month at a meat processing plant is worth it.</event> <event>Time:00:15-00:20, Des: The same two individuals continue working in the factory, with one person moving boxes and another handling machinery.</event> <event>Time:00:20-00:25, Des: A man in a black jacket is shown speaking while holding a phone, discussing the cost of a different apartment.</event> <event>Time:00:25-00:30, Des: An exterior view shows a group of people gathered around snow-covered cars in a parking lot under streetlights.</event>

<think>The video transitions from a discussion about working conditions in a factory to a scene of people gathering around snow-covered cars, suggesting a shift in location or context. The next logical step would be to show an interior view of the factory or related activities, followed by a scene that might involve the workers preparing for their day, which aligns with the typical workflow in such environments. The truck mentioned in the options could be related to transportation or logistics within the factory setting.</think>

<answer>A</answer>

Figure 8. An example of CoE paradigm.

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

###### Input Video

###### Question:

“Based on the given video, predict future events that would occur after the observed scenes (the filling, heating, and stirring of the substance behind the counter) and before the final scene. Complete the following sequence with the correct intermediate events: 1. The video shifts to the person displaying an orange, flexible flat sheet to the camera to emphasize its texture. 2. [Missing Event A] 3. The scene then focuses on a close-up of several skewers holding neatly arranged orange, rectangular food items. 4. [Missing Event B] 5. Finally, a detailed close-up shows a dark, glossy circular object being manipulated over a white bowl with a wooden skewer. Which pair of events best fits as Missing Event A and Missing Event B, respectively, based on the full video narrative?”

###### Options:

- A. The person interacts with a customer by handing over one of the orange sheets, then later explains or demonstrates product details to a colleague.
- B. The person quickly glances at the packaged snacks before moving on, then rearranges the skewers without further explanation.
- C. The person examines the flat sheet without engaging in any customer interaction, then briefly picks up a bag of small packaged snacks.
- D. The person ignores the flat sheet and starts organizing the counter, then later fixes the position of the skewers without any interaction.

CoE:

<event>Time:00:00-00:05, Des: A white powdery substance is poured into a metal container.</event> <event>Time:00:05-00:10, Des: A person in a striped shirt is seen stirring a pot on a stove.</event> <event>Time:00:10-00:15, Des: The person pours a dark liquid from a pot into a white bowl using a wooden skewer.</event> <event>Time:00:15-00:20, Des: The person places the bowl with the dark liquid on the counter next to various packaged snacks.</event> <event>Time:00:20-00:25, Des: The person picks up an orange, flexible flat sheet and displays it to the camera.</event> <event>Time:00:25-00:30, Des: The person arranges several skewers with neatly arranged orange, rectangular food items on the counter.</event> <event>Time:00:30-00:35, Des: The person manipulates a dark, glossy circular object over a white bowl using a wooden skewer.</event> <think>The video sequence shows a series of actions related to preparing and presenting food items. The first part involves pouring and stirring substances, which suggests a cooking or food preparation process. The middle part shows the person handling an orange flat sheet and arranging skewers with food items, indicating a display or presentation phase. The final part involves manipulating a dark, glossy object, likely as part of the food preparation or serving process. Given this context, the missing events should logically follow the sequence of actions and interactions. Option A provides a logical continuation where the person interacts with the flat sheet and then explains or demonstrates product details to a colleague, fitting the narrative flow.</think> <answer>A</answer>

Figure 9. An example of CoE paradigm.

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

###### Input Video

###### Question:

“Based on the given video, predict future events and fill in the missing steps in the following sequence: 1. A Pyrex cup is shown being filled with red wine vinegar, water, and Italian dressing mix. 2. [?] 3. The creamy dressing is poured over the bowl of pasta salad and mixed thoroughly. 4. [?] 5. A block of feta cheese is crumbled and added to the pasta salad as the final step. Which of the following options correctly predicts the missing events in the proper order?”

###### Options:

- A. A hand stirs in vegetable oil to blend the dressing ingredients; A container of Mediterranean spice mix is sprinkled over the salad.
- B. A hand stirs in mayonnaise to create a smooth dressing; A hand adds extra olives to the salad.
- C. A hand pours vegetable oil over the dressing, forming a distinct layer and then mixes it thoroughly; A hand sprinkles Mediterranean spice mix before finishing with crumbled feta cheese.
- D. A hand mixes in extra Italian dressing mix into the Pyrex cup; A hand tops the salad with an extra garnish of red bell pepper.

CoE:

<event>Time:00:00-00:10, Des: A close-up shot of a pasta salad is shown, followed by a hand chopping red onions on a wooden cutting board.</event> <event>Time:00:10-00:20, Des: The scene transitions to a hand slicing a red bell pepper, then cutting broccoli florets off the stalks.</event> <event>Time:00:20-00:30, Des: Next, a can of sliced black olives and a can of artichoke hearts are displayed. A hand slices an artichoke heart in half.</event> <event>Time:00:30-00:40, Des: The video then shows a pot of boiling water on a stove, with salt being added. Broccoli is added to the pot three minutes before the pasta finishes cooking.</event> <event>Time:00:40-00:50, Des: The pasta is drained and cooled under running water. The cooled pasta is then mixed with the chopped vegetables and other ingredients.</event> <event>Time:00:50-01:00, Des: The creamy dressing is poured over the bowl of pasta salad and mixed thoroughly. Finally, a block of feta cheese is crumbled and added to the pasta salad.</event> <think>In the video, the preparation of the dressing involves mixing red wine vinegar, water, and Italian dressing mix in a Pyrex cup. The next logical step would be to blend

these ingredients together, possibly adding vegetable oil for emulsification. Following this, the dressing should be poured over the salad and mixed thoroughly. The final step mentioned in the question is the addition of feta cheese. The correct sequence aligns with the provided options where the dressing is blended and then mixed with the salad, followed by the addition of feta cheese.</think>

<answer>C</answer>

Figure 10. An example of CoE paradigm.

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

###### Input Video

###### Question:

“Based on the given video, predict future events and fill in the potential event in the following sequence: 1. [?] 2. The final scene shows the person in a warm, softly lit room, still mimicking the animated character with red hair, glasses, and a striped shirt. Which of the following best describes the missing event that likely occurred between the currently observed scenes and the final scene?”

###### Options:

- A. The person mimics a blonde animated character with curly hair, maintaining the same expression in the warm room.
- B. The person mimics an animated white cat wearing a pink bow while staying in the warm, softly lit room.
- C. The person mimics an animated character with a blue suit and a red bow tie, similar to earlier scenes.
- D. The person mimics an animated character with purple hair and a red gem on the forehead while the background changes to a cool, dim room.

CoE:

<event>Time:00:00-00:05, Des:\"Challenge Accepted\" text appears on screen.</event> <event>Time:00:05-00:10, Des:A woman in a white shirt starts mimicking an animated character with a blue suit and a red bow tie.</event> <event>Time:00:10-00:15, Des:The woman continues to mimic different animated characters, including one with red hair and glasses, and another with purple hair and a red gem on the forehead.</event> <event>Time:00:15-00:20, Des:The woman is now in a warm, softly lit room, continuing to mimic the animated character with red hair, glasses, and a striped shirt.</event> <think>Given the sequence of events, it is clear that the woman is transitioning through various animated characters. The final scene shows her in a warm, softly lit room, which suggests a continuation of the previous scenes but with a slight change in the character being mimicked. This suggests that the missing future event should involve continuing to mimic a character, and that this character should not have appeared previously. The last event among the future events indicates that there is no change in the scene within the video, so the future event should involve this person mimicking different characters in the same scene. </think>

- <answer>B</answer>

Figure 11. An example of CoE paradigm.

CoE:

<event>Time:00:00-00:30, Des: A baby in a striped onesie is lying on a quilted blanket, moving its arms and hands playfully.</event>

- <event>Time:00:30-00:45, Des: The baby is now being held by an adult wearing a red hat and a striped shirt, both smiling and engaging with the baby.</event>

- <event>Time:00:45-01:00, Des: The baby is seated in a high chair, smiling and looking around, appearing content and curious.</event>
- <event>Time:01:00-00:30, Des: The baby is being held by an adult outdoors near a river, with the adult talking and gesturing while holding the baby close.</event>

- <event>Time:01:30-01:45, Des: The baby is back in the high chair, exploring its tray, indicating a return to a more structured environment.</event> <think>The video progresses from indoor scenes with the baby on a quilted blanket and in a high chair, to an outdoor scene with the baby being held by an adult near a river. The sequence then returns to an indoor setting before concluding with a promotional scene. The options provided should reflect this progression, starting with an outdoor scene, followed by an indoor scene, and ending with a promotional screen.\n\nOption C aligns with the observed sequence as it starts with an outdoor scene, transitions to an indoor scene, and ends with a promotional screen. The other options either do not follow the observed sequence or contain elements that do not match the video content.</think>

<answer>C</answer>

Question:

“Based on the given video, predict future events and fill in the potential events in the given future events: 1. [?] 2. [?] 3. [?] 4. [A promotional scene appears featuring a black background, an image of the baby (in a blue onesie) in front of a toilet, and call-to-action text urging viewers to check out the previous video]. Which option best represents the correct sequence of future events that bridge the observed scenes and the final scene?”

Options:

- A. 1. The video cuts to an indoor cozy scene with a woman on a couch playing with the baby; 2. A quick flash of a promotional black screen with social media web addresses; 3. A return to outdoor walking along a grassy path by the river; 4. Final promotional screen with the baby sitting in front of a toilet and call-to-action text.
- B. 1. An outdoor sequence where the baby and adult are seen exploring a dirt path with a visible staircase; 2. A sudden switch to a warm indoor scene with the baby interacting with toys on a colorful blanket; 3. A black screen appears with text promoting social media channels; 4. Final promotional screen with the described baby image and text.
- C. 1. A black screen appears with bold red text reading 'SUBSCRIBE YO!' followed by a screen displaying web addresses for social media; 2. The scene returns outdoors near the river showing the adult and baby walking along a grassy path surrounded by trees; 3. The video transitions indoors where a woman is seen on a couch with the baby (now in a white onesie with green text) amid cozy details like a colorful blanket and an open laptop; 4. Final promotional screen featuring a black background with white text urging viewers to 'CHECK OUT MY LAST RYDEN VIDEO!!' alongside an image of the baby in a blue onesie in front of a toilet.
- D. 1. A scene shows the baby back on the high chair exploring its tray in a neutral room; 2. The video shifts to an alternate outdoor setting with the adult in a red shirt pointing towards a river; 3. A black screen with red text appears but without any social media details; 4. Final promotional screen appears with minimal text and no clear baby image.

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

Input Video

Figure 12. An example of CoE paradigm.

CoE:

<event>Time:00:00-00:28, Des: A woman in a black and white striped top speaks directly to the camera in an indoor setting.</event>

- <event>Time:00:28-00:57, Des: The same woman is now outside in an alleyway, demonstrating various dance moves labeled on-screen.</event>

- <event>Time:00:57-01:16, Des: The woman continues her dance routine in the alleyway, showcasing different moves like \"SpongeBob + Patrick\" and \"Cold Shoulder.\”</event>
- <event>Time:01:16-01:39, Des: The scene returns indoors, where the woman speaks again, this time with a more animated expression.</event>

<event>Time:01:39-02:00, Des: The video cuts back to the alleyway, where the woman dances alongside a man, performing synchronized moves.</event> <event>Time:02:00-02:15, Des: The scene shifts to a party setting with streamers, balloons, and people holding red cups in a dimly lit room.</event>

- <event>Time:02:15-02:30, Des: The woman in the black and white striped top reappears, now in a white skirt and a large white bow, dancing energetically while a man watches.</event>

- <event>Time:2:30-2:45, Des: The video ends with a woman in a black sweater with a patterned collar observing the dancing couple.</event> <think> Given the sequence of events, we need to identify the correct option that fits logically between the party scene and the final scene with the observer. Option A correctly transitions from the party scene to the indoor setting where the woman speaks, then to the alleyway dance, and finally back to the original indoor setting where she speaks again. This sequence matches the observed video flow most accurately.</think> <answer>A</answer>

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

###### Input Video

###### Question:

“Based on the given video, predict future events and fill in the potential events in the following sequence: 1. The video shifts to a party setting with streamers, balloons, and people holding red cups in a dim space lit with pink and blue hues. 2. [?] 3. Soon after, the woman in the black and white striped top (now paired with a white skirt and a large white bow) takes the center stage and begins dancing energetically as a man watches. 4. [?] 5. Later, a woman wearing a black sweater with a patterned collar is shown observing the dancing couple. Which option correctly fills in the two missing events with scenes that occur between the observed scenes and the final end screen?”

###### Options:

- A. After the party is introduced, the dancing couple holds hands and moves in perfect sync before the energetic solo dance, and later the scene transitions to the original indoor setting where the woman is seen speaking expressively.
- B. Right after the party is introduced, the indoor set reappears with the woman speaking, then later the dancing couple holds hands and moves in sync before the observer is shown.
- C. Immediately following the party scene, another group of dancers appears performing similar moves, and later, a brief close-up of red cups and streamers is shown before the observer appears.
- D. After the party setting is established, a montage of previously seen alley dance moves is replayed, and then a quick scene shows party guests clapping before the observer appears.

Figure 13. An example of CoE paradigm.

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

###### Input Video

###### Question:

“Based on the given video, predict future events and fill in the potential events in the following future event sequence: 1. [?] 2. [?] 3. [?] 4. The man stands near the wall with hanging clothes under a distinct light source, intently reading or typing on his laptop with a contemplative look. Which sequence of intermediate events best fits the evolution from the observed scenes to this final future scene?”

###### Options:

- A. 1. The man abruptly leaves the room with the laptop, 2. He returns outdoors in broad daylight, 3. He interacts with multiple strangers, then 4. The man stands near the wall with hanging clothes under a distinct light source, intently reading or typing on his laptop with a contemplative look.
- B. 1. The man remains stationary in the room, 2. He sits down on the colorful mat and starts writing notes by hand, 3. He gradually closes his laptop as the room darkens further, then 4. The man stands near the wall with hanging clothes under a distinct light source, intently reading or typing on his laptop with a contemplative look.
- C. 1. The man moves toward a doorway bathed in natural light and interacts again with the switchboard, 2. He walks around the room pausing by the hanging clothes, 3. He returns to the center of the room, still focused on his laptop with subtle changes in lighting, then 4. The man stands near the wall with hanging clothes under a distinct light source, intently reading or typing on his laptop with a contemplative look.
- D. 1. The man disconnects his laptop and inspects the red bag on the floor, 2. He rearranges the clothes hanging on the wall in a frenzied manner, 3. He sits in silence and looks out of a window for a long pause, then 4. The man stands near the wall with hanging clothes under a distinct light source, intently reading or typing on his laptop with a contemplative looks.

CoE:

- <event>Time:00:00-0:5, Des: A man enters a room carrying a laptop.</event>

- <event>Time:0:5-1:5, Des: The man places the laptop on a colorful mat and begins using it.</event>
- <event>Time:1:5-3:0, Des: The man stands up and continues working on the laptop while occasionally looking around the room.</event>

<event>Time:3:0-4:0, Des: The man shifts his position slightly but remains engaged with the laptop, maintaining focus on the screen.</event> <event>Time:01:39-02:00, Des: The video cuts back to the alleyway, where the woman dances alongside a man, performing synchronized moves.</event> <event>Time:02:00-02:15, Des: The scene shifts to a party setting with streamers, balloons, and people holding red cups in a dimly lit room.</event>

- <event>Time:02:15-02:30, Des: The woman in the black and white striped top reappears, now in a white skirt and a large white bow, dancing energetically while a man watches.</event>

- <event>Time:2:30-2:45, Des: The video ends with a woman in a black sweater with a patterned collar observing the dancing couple.</event> <think> The video shows a man entering a room with a laptop, placing it on a mat, and starting to use it. He then stands up and continues working, occasionally looking around the room. The final scene describes him standing near a wall with hanging clothes, illuminated by a distinct light source, and focused on his laptop. This sequence suggests a progression where the man initially works on the laptop in various positions before settling into a more stationary posture near the wall, indicating a shift from movement to a more contemplative state.\n\nOption C aligns with this progression as it shows the man moving towards a well-lit area, interacting with the environment (hanging clothes), and returning to a focused state near the wall, which matches the final scene described in the question.</think> <answer>C</answer>

Figure 14. A bad case of CoE paradigm.

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

###### Input Video

###### Question:

“Based on the given video, predict future events and fill in the potential events in the given future sequence: 1. [?] 2. A close-up of a sign showing 'HACKETT MACAU TOWER' along with 'The World's Highest Bungy and Skyjump 233 meters' and the website 'www.ajhackett.com'. Which sequence of events best fits in slot 1?”

###### Options:

- A. The person waves at the camera, then leans forward and jumps off the platform. An aerial view follows as the person is seen diving headfirst with the bungee cord visible, landing on an inflatable platform where they are gradually pulled toward the edge by the cord.
- B. The person stays on the platform and engages in a lengthy conversation with the crew, while no jump occurs; the crew then steps away and the scene abruptly cuts to the close-up of the sign.
- C. Instead of a direct jump, the person performs a series of small hops on the platform, interspersed with short sidesteps, and only later does the camera cut to the detailed close-up of the sign.
- D. The person is seen descending slowly using an elevator mechanism attached to the structure, bypassing any free-fall or dynamic bungee jump, and later the scene shifts to the detailed view of the sign.

CoE:

<event>Time:00:00-0:5, Des: A person is seen preparing for a bungee jump at the Hackett Macau Tower. They are wearing safety gear and standing on a platform high above the city.</event>

- <event>Time:0:5-1:4, Des: The person waves at the camera, indicating readiness for the jump. The camera captures their expression and the surrounding cityscape.</event>
- <event>Time:1:4-2:26, Des:The person is secured with a harness and is being assisted by two individuals. They appear to be getting ready for the jump, with the camera focusing on the preparation process.</event>
- <event>Time:2:26-5:0, Des: The person is now on the edge of the platform, surrounded by safety equipment. They are being helped by the crew members who are ensuring everything is secure before the jump.</event> <event>Time:5:0-7:0, Des: The person is now standing alone on the platform, looking down at the city below. The camera captures the anticipation and the height of the jump.</event> <think> The video shows a sequence of events leading up to a bungee jump. The person is initially seen preparing, then being assisted, and finally standing alone on the platform. The close-up of the sign provides additional context about the location and activity but does not show the jump itself. Therefore, the most logical sequence that fits the given events would include the jump occurring after the preparation phase, which aligns with option A.</think> <answer>A</answer>

Figure 15. A bad case of CoE paradigm.

###### CoE-GRPO Vanilla GRPO CoE-SFT CoT

Vanilla SFT

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

𝛿𝛿𝛿𝛿𝛿DifferenceinAttentionScoresDifferenceinAttentionScoresDifferenceinAttentionScoresDifferenceinAttentionScoresDifferenceinAttentionScores

Sample1Sample2Sample3Sample4Sample5

Visual Token Sequence S Visual Token Sequence S Visual Token Sequence S Visual Token Sequence S Visual Token Sequence S

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

Visual Token Sequence S Visual Token Sequence S Visual Token Sequence S Visual Token Sequence S Visual Token Sequence S

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

Visual Token Sequence S Visual Token Sequence S Visual Token Sequence S Visual Token Sequence S Visual Token Sequence S

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

Visual Token Sequence S Visual Token Sequence S Visual Token Sequence S Visual Token Sequence S Visual Token Sequence S

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

Visual Token Sequence S Visual Token Sequence S Visual Token Sequence S Visual Token Sequence S Visual Token Sequence S

[Figure 136]

Figure 16. Some examples of attention differences comparing to the vanilla model.

System Prompt:

"You are a video event analyst and future event prediction expert."

CoE-GRPO:

"\nFirst output the seen events in the video within <event> </event> tags in chronological order. \nThen Based on the observed video and events, analyze and reason through the questions and options. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in <answer> </answer> tags. \nFormat: several <event>Time:xxx-xxx, Des:xxx </event>. <think> reasoning </think>. <answer> answer with the option's letter from the given choices </answer>."

###### Vanilla GRPO:

"\nFirst think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in <answer> </answer> tags. \nFormat: <think> reasoning </think>. <answer> answer with the option's letter from the given choices </answer>."

###### CoE-SFT:

“\nFirst analyze the video content and then perform logical event prediction based on the observed video. Afterward, answer the relevant questions. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in <answer> </answer> tags. \nFormat: <think> reasoning </think>. <answer> answer with the option's letter from the given choices </answer>."

###### Vanilla SFT:

"\nFirst analyze and reason through the questions and options. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in <answer> </answer> tags. \nFormat: <think> reasoning </think>. <answer> answer with the option's letter from the given choices </answer>."

Figure 17. Prompt template for training and inference.

###### CoE-SFT Data Generation Prompt:

"role": "You are an professional analytical video interpreter.", "task": "Based on the given video, question, options, and correct answer, you need to generate the reasoning process that derives the answer from the

video and the question. Avoid analyzing the answer choices, focusing instead on the video and question for analysis. Keep your reasoning concise and clear.", "question": "@question@", "answer": "@answer@", "note": "Directly generate the reasoning process for deriving the correct answer from the video and the question without any other words. Your reasoning

should be concise and impactful."

###### Judge Model Evaluation Prompt:

"role": "You are a video analysis and prediction expert and an impartial model evaluation judge.", "task": " I will provide you with a video, related questions, reference answers for the questions, and responses from several different models. Please

analyze the reasoning and accuracy of each model's responses to the questions, and finally select the model that provides the best answer.", "question": "@question@", "reference snswer": "@answer@", "model responses": "@model_responses@", "note": "Please ensure fairness and impartiality by comprehensively considering the reasoning and accuracy of each model's answers. Select the model

with the best response to the question, provide the number of the best model, and explain the reason."

Figure 18. Prompt template for CoE-SFT data generation and judge model evaluation.

