## Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment

arXiv:2412.19326v2[cs.CV]30Jun2025

Ziang Yan∗2,1, Zhilin Li∗3,1, Yinan He∗1 Chenting Wang4,1, Kunchang Li5,1, Xinhao Li6,1, Xiangyu Zeng6,1 Zilei Wang3, Yali Wang5,1,Yu Qiao1, Limin Wang6,1, Yi Wang†1,7

1Shanghai AI Laboratory 2Zhejiang University 3University of Science and Technology of China 4Shanghai Jiao Tong University 5Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences 6Nanjing University 7Shanghai Innovation Institute https://github.com/OpenGVLab/TPO

[Figure 1]

Which cup is the candy under, left one, middle one or right one?

#### Abstract

Show the tracking trajectory of cup with candy in middle. Which cup is the candy under, left one, middle one or right one?

[Figure 2]

[Figure 3]

Current multimodal large language models (MLLMs) struggle with fine-grained or precise understanding of visuals although they give comprehensive perception and reasoning in a spectrum of vision applications. Recent studies either develop tool-using or unify specific visual tasks into the autoregressive framework, often at the expense of overall multimodal performance. To address this issue and enhance MLLMs with visual tasks in a scalable fashion, we propose Task Preference Optimization (TPO), a novel method that utilizes differentiable task preferences derived from typical fine-grained visual tasks. TPO introduces learnable task tokens that establish connections between multiple task-specific heads and the MLLM. By leveraging rich visual labels during training, TPO significantly enhances the MLLM’s multimodal capabilities and task-specific performance. Through multi-task co-training within TPO, we observe synergistic benefits that elevate individual task performance beyond what is achievable through single-task training methodologies. Our instantiation of this approach with VideoChat and LLaVA demonstrates an overall 14.6% improvement in multimodal performance compared to baseline models. Additionally, MLLMTPO demonstrates robust zero-shot capabilities across various tasks, performing comparably to state-of-the-art supervised models.

[Figure 4]

[Figure 5]

The cups are shuffled in the video, and the candy is under the middle cup.

TPO

|[Figure 6]<br><br>|
|---|

|[Figure 7]<br><br>|
|---|

|[Figure 8]<br><br>|
|---|

|[Figure 9]|
|---|

[Figure 10]

[Figure 11]

[Figure 12]

| |
|---|

[Figure 13]

The candy is under the left cup.

The candy is under the first cup on the left.

Figure 1. TPO uses differentiable task preferences from dense visual supervisions via task-specific heads to enhance MLLMs in fine-grained understanding.

growing expectations of users in more accurate and detailed perception, taking the shell game as an example, further improving the generality of MLLM requires fine-grained knowledge representation beyond words, e.g. MLLM implicitly embeds how to track keys indicated by users.

Existing studies address the enhancement of perceptual granularity in MLLMs by focusing on specific visual tasks (e.g. temporal grounding, segmentation, tracking) via MLLMs. They usually fine-tune MLLMs on more task data in text format or enable MLLMs to activate the corresponding task heads. Shikra [8] applies MLLM to localization tasks, transforms object coordinates into dialogue formats, and learns them autoregressively. Meanwhile, TimeChat [70] and VideoChat-T [103] treat event timestamps as text for autoregressive prediction, thereby endowing MLLMs with temporal grounding. They do improve specific task performance significantly, while at the cost of multimodal performance more or less. This is counterintuitive as seminal research proves different visual tasks are correlated and training them together often yields mutual gains [31, 50, 73]. We conjecture that the presentation of different tasks influences this training and that the conflict

#### 1. Introduction

Multimodal large language models (MLLMs) show impressive visual perception and reasoning capabilities, with applications in personal assistants [75], embodied systems [19], and scientific discovery [7]. Considering the

*Equal contribution. †Corresponding author.

P2S

Textual Data Visual Labels

P2S

Textual Data Visual Label

###### MLLM

###### MLLM

[Figure 15]

PPO

P2S

Textual Data Visual Labels

P2S

Textual Data Visual Label

MLLM

###### MLLM

P2E Visual Labels

P2E Visual Label

MLLM

Task Head

###### MLLM

Task Head

arises from the learning discrepancies between discrete textual tokens and visually dense predictions. Our experiments in Section 4 validate that a decoupled representation design can effectively address this issue.

PPO

[Figure 17]

[Figure 18]

| | |
|---|---|
|P2E|MLLMMaximum Task Head Visual<br><br>|

[Figure 20]

Labels

P2E Visual Label

###### MLLM

Task Head

Maximum Likelihood

[Figure 22]

PPO MLLM RewardModel Reward

Likelihood

Reference DPO Model

Reference DPO Model

MLLM

MLLM

[Figure 24]

Maximum Likelihood

Maximum Likelihood

[Figure 26]

Reference DPO Model

[Figure 27]

Reference DPO Model

MLLM

MLLM

Maximum Likelihood

[Figure 28]

Visual Labels

TPO

To enhance the multimodal capabilities of MLLMs, we explore optimization methods to meet multiple visual task requirements in an end-to-end manner. We propose task preference optimization (TPO), incorporating visual task knowledge into the MLLMs by jointly maximizing the likelihoods of visual task estimations and multimodal dialogue. Inspired by direct preference optimization [66] (DPO) and related methods, we treat visual task annotations as human preferences in particular demands, as shown in Figure 2. DPO aids LLM (or MLLM) in aligning with human preferences through a binary classification that directs the model to generate responses that people prefer. Similarly, TPO enhances MLLMs’ visual sensing capabilities through differentiable task optimizations that guide MLLMs to yield dense predictions closely resembling human perception. To achieve this, TPO concretizes MLLM-specific visual perceptions into corresponding task tokens, disentangled from MLLM representation. Then TPO fine-tunes these task tokens and updates the MLLM accordingly.

Reference DPO Model

Visual Label

MLLM

TPO

Task Head

MLLM Task Head

MLLM

| | |
|---|---|
| | |

Visual Labels

TPO

Visual Label

MLLM Task Head

TPO

MLLM Task Head

Conversation Data

Conversation Data

Conversation Data

Conversation Data

Reward Model

###### TPO

MLLM Task Head

Visual Labels

PPO MLLM

Reward

Conversation Data

PPO

[Figure 29]

Figure 2. Comparison of Learning Method. A solid line indicates data flow, and a dotted line represents feedback.

Maximum

[Figure 30]

[Figure 31]

[Figure 32]

Likelihood

and

Reference DPO Model

MLLM

denote modules that are frozen and unfrozen.

PPO

Visual Label

###### TPO

mization (TPO). This method leverages supervised information from visual task-specific data to optimize the MLLM through the corresponding heads, resulting in significantly enhanced multimodal perception and reasoning performance. Specifically, TPO achieves an average improvement of 14.6% across multiple image and video multimodal benchmarks [21, 40, 48, 59, 112].

MLLM Task Head

PPO MLLM RewardModel Reward

Conversation Data

[Figure 33]

Maximum Likelihood

[Figure 34]

Reference DPO Model

MLLM

[Figure 35]

###### TPO

MLLM Task Head

Visual Labels

Conversation Data

- • TPO effectively equips MLLM with the capability to address several key visual perception tasks through the introduced task heads. MLLM-TPO achieves comparable performance in spatial grounding, moment retrieval, highlight detection, referring segmentation, and tracking comparable to expert models across various benchmarks.
- • TPO demonstrates scalability across various employed MLLMs, task heads, and scales of task data. We validate the effectiveness of TPO in multiple mainstream MLLMs, such as VideoChat2 [48] and LLaVA [41, 51]. Notably, multi-task joint training based on TPO enhances both multimodal performance and individual visual task, with improvements becoming increasingly significant as additional appropriate heads are introduced. Furthermore, the performance of the MLLM and task heads improves when scaling task data.

Specifically, TPO appends visual task heads to the partial output of the MLLM, using several learnable task tokens as inputs for these corresponding heads. During training, TPO enables the MLLM to first distinguish and activate the appropriate task tokens based on user instructions. Subsequently, TPO jointly trains the task tokens and their corresponding heads to enhance the MLLM’s understanding of visual tasks. Finally, TPO trains the entire model—including task tokens and heads—on both multimodal and visual-specific task data, promoting the perception and reasoning capabilities of the MLLM. Additionally, we note that multi-task co-training yields greater improvements than single-task training.

TPO demonstrates scalability across various MLLM approaches, encompassing a wide range of visual task categories and data quantities.

#### 2. Related Work

Vision Foundation Model. Vision foundation models [47, 73, 76, 78, 79, 81, 85, 91, 94, 111] are designed to be adaptable for various downstream tasks through extensive pre-training on large-scale and diverse datasets. VideoPrism [111] achieves the leading results in various video tasks by combining video-text contrastive learning and spatiotemporal token reconstruction using a dataset of public and proprietary videos. InternVideo2 [85] utilizes masked reconstruction, cross-modal contrastive learning, and nexttoken prediction to enhance the model’s perceptiveness, semantic understanding, and reasoning capabilities.

We validate the effectiveness of TPO within widely used MLLMs, such as LLaVA [51] and VideoChat [46], as detailed in Section 4.3. By fine-tuning these open-source MLLMs with TPO, we enhance their visual understanding capabilities and improve dialogue performance. Additionally, we explore several key spatiotemporal and spatial perception tasks, including spatial grounding, tracking, and temporal grounding. Our findings indicate that these tasks can mutually enhance each other’s performance, particularly contributing to the improvement of multimodal dialogue capabilities.

Based on vision foundation models, some studies [44, 55, 84, 114, 115] intend to incorporate downstream task heads into this framework and expect end-to-end task training. Unified-IO [55] introduces a versatile model capable

Our contributions can be summarized as:

• We propose a new training method for multimodal large language models, referred to as Task Preference Opti-

[Figure 36]

|The video shows the rabbit [98, 201, 135, 128] closer to the wall surrounded by vegetables. At<br><br>TPO 9.4s-10.3s, another rabbit jumps over it. Now the other rabbit is closer to the wall than it.<br><br>[Figure 37]<br><br>|
|---|

MLLM Output

# TPO

||[Figure 38]<br><br>|
|---|
<br><br>|[Figure 39]<br><br>|
|---|
<br><br>|[Figure 40]<br><br>|
|---|
<br><br>|[Figure 41]<br><br>|
|---|
|
|---|

Special Token

| |
|---|

|𝑣𝑖|
|---|

| |
|---|

| |
|---|

| |
|---|

Mask

Head

|Connector|
|---|

|𝑒𝑖|
|---|

||[Figure 42]|
|---|
<br><br>|[Figure 43]|
|---|
<br><br>|[Figure 44]|
|---|
<br><br>|[Figure 45]|
|---|
<br><br>9.4s 10.3s<br><br>|
|---|

LLM

Temporal Head

| |
|---|

Vision

LoRA

Encoder

### …

||[Figure 46]<br><br>| |
|---|
|
|---|
<br><br>|[Figure 47]<br><br>| |
|---|
|
|---|
<br><br>|[Figure 48]<br><br>| |
|---|
|
|---|
<br><br>|[Figure 49]<br><br>| |
|---|
|
|---|
|
|---|

Region Head

|Watch the rabbit near the wall, what<br><br>happens in the video?<br><br>|
|---|

|[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]|
|---|

Stage 1: Stage 2: Stage 3:

Visual Input

User Input

Figure 3. Overall Pipeline of TPO. The architecture of Task Preference Optimization (TPO) consists of four main components: (1) a vision encoder, (2) a connector, (3) a large language model, and (4) a series of visual task heads. Differently colored flame symbols indicate which components are unfrozen at various stages of the training process.

of handling a wide range of tasks across vision and language domains, requiring minimal or no task-specific modifications or additional parameters. Uni-Perceivers [44, 114, 115] formulate different tasks to find the maximum likelihood target for each input through the representation similarity regardless of their modality. Nevertheless, these generalist models are limited to predefined tasks and cannot support flexible open-ended task customization based on language instructions.

and inputs visuals into the downstream decoder for estimations. LISA [37], NExT-Chat [104], and VideoLISA [4] introduce SAM [35] as a segmentation tool, using a special token as a prompt to connect MLLM and SAM. VisionLLM v2 [89] designs routing tokens and super-link queries to bridge MLLM with multiple decoders.

Alignment in MLLM. Aligning MLLM with human preferences or values is crucial for MLLM’s development. Recent works [74, 101, 113] introduce alignment approaches to MLLM, including proximal policy optimization (PPO) [71] and direct preference optimization (DPO) [66], as shown in Figure 2. They usually exploit proprietary models like GPT4-V to build visual preference datasets and then tune MLLMs using PPO or DPO. Llava-RLHF [74] incorporates PPO into a MLLM framework Llava, argumented by image captions or question-answers (QA), while Zhou et al. [113] and Zhang et al. [109] give DPO implementations for MLLMs where visual preference data are created by GPT-4V and other open-sourced MLLMs. RLHFV [101] collects a dense human preference in segmentation and enhances MLLMs with DPO.

Multimodal Large Language Model. The effective understanding and reasoning of LMMs has attracted the attention of numerous researchers. Limited to its input modality, researchers expand the visual capabilities of LLM, leading to MLLMs. Seminal works, such as BLIP-2 [45], LLaVA [51], and mPLUG-Owl [96], introduce image captioning and visual question answering based on LLM using visual instruction-tuning data. Some video-based MLLMs have been proposed, such as VideoChat [46], VideoChatGPT [57], and Video-LLaMA [107], enabling LLM to gain video understanding capabilities by encoding multiple video frames and using video instruction data.

Vanilla MLLMs have achieved impressive results in visual-related dialogues, but barely address some finegrained tasks, such as segmentation and temporal grounding. To address this challenge, MLLMs usually take one of the following pipelines: pixel-to-sequence (P2S) and pixel-to-embedding (P2E). For P2S methods [8, 70, 77, 86, 97], MLLM directly outputs the textual predictions. TimeChat [70] introduces a time-aware frame encoder and a sliding video Q-former to enhance temporal perception. For P2E methods [4, 37, 82, 89, 104], MLLM compresses

#### 3. Method

Task Preference Optimization (TPO) aims to enable MLLMs to master classical visual perceptions (such as tracking, temporal grounding, etc.) for better task generalization, regarding that many multimodal reasoning tasks require precise visual cues for accurate and reliable responses. As shown in Figure 3, MLLM-TPO has a typical multimodal model M (consisting of a vision encoder E, a vision-language connector C, and a large language model

G) and a task preference model P with a series of visual task heads {Hi}i=1...n (n denotes the task head number). These heads connect with MLLM using the embeddings {ei}i=1...n (ei = G(vi) ∈ R1×C) transformed from the learnable task-specific tokens {vi}i=1...n (vi ∈ R1×C) via MLLM.

TPO employs a local-to-global training scheme, first adapting task heads to the MLLM and then training them jointly. Specifically, the MLLM M starts to recognize {vi}i=1...n by updating G according to user instructions, then we tune {vi}i=1...n and {Hi}i=1...n for adapting visual heads to M. Finally, we train both M and {Hi}i=1...n together. During inference, MLLM-TPO can respond to users’ queries in text, and produce structured visual outputs (like masks, timestamps, trajectories, etc.) when users ask (e.g. yielding time results for “find when the birthday party starts”). We detail how to form MLLM-TPO structurally and train it in the following.

##### 3.1. Task Preference Models

The task preference model (TPM) P contains a series of task tokens {vi}i=1...n and heads {Hi}i=1...n. Before TPM works, the attached MLLM generates special tokens indicating task types from the input queries. Then TPM dynamically calls the task token vj based on the special token, transforms it to the task embedding ej via the LLM G (the last hidden embedding), and feeds ej to the corresponding visual task head {Hj} for specific task predictions. Considering existing MLLMs demonstrate remarkable capabilities in common object and scene recognition, yet struggle to accurately locate things or actions, the employed task heads mainly focus on spatiotemporal localization and tracking. Specifically, we give three fundamental task types for compensating mainstream MLLMs’ gaps from expert models in visual perceptions: 1) region head, 2) temporal head, and 3) mask head. Their architectures are given below.

- • Region Head. A two-layer multilayer perceptron (MLP) with ReLU activations is employed for the region head. It takes embeddings from LLM and regresses them to the bounding box coordinates for spatial grounding.
- • Temporal Head. This head is composed of a video encoder, text encoder, and temporal grounding model for moment retrieval and highlight detection. The video and text queries are input into the video and text encoders to obtain their features respectively. Then we append the temporal task embedding after the text features. With the new text and video features, the MLLM estimates the start/end time and highlight score of the corresponding moment from the query via the temporal grounding model.
- • Mask Head. Pixel-level tasks pose significant challenges for MLLMs due to the lack of the corresponding learning data and optimization in MLLMs. To this end, we

introduce a specialized mask head, utilizing the image encoder, mask decoder, and memory bank components from SAM2 [68], replacing the prompt encoder with a single MLP layer called the mask adapter. Specifically, for the given visuals and user queries, the features extracted from the aforementioned image encoder and the text representation corresponding to the mask task embeddings from LLM are employed, fed into the mask decoder to produce the desired mask.

Most known discriminative vision tasks can be addressed by one or a combination of these three task heads. TPM builds the architectural foundations for leveraging existing vision data with annotations to enhance MLLMs.

##### 3.2. Task Preference Optimization

TPO improves MLLMs with extra supervisions from visual task heads by back-propagating gradients from heads to update MLLMs using visual task data. It enables the language model G in MLLM to discriminate specific task types when users demand (task assignment). Then, TPO trains TPM P via compact task representations ei (task optimization) from vi. Lastly, we train M and P together, tuning M for the refined spatiotemporal perception according to task preferences from P. Its optimization objective is given as:

n

Ltask(Ai,Hi(G(vi))) Task Optimization

###### L = Lmllm+Lassign(G(Tq),s)

+

###### ,

i=1

Task Assignment

(1) where Tq and s denote a user query (some contain specific task needs like tracking) in the form of text token sequence and the ground truth task indicating token, respectively. Ai is the task annotation of the input visual X, so it could be a number tuple for describing regional localization and area or mask for delineating object shape and position. Here (X,Tq,Ta,Ai) stands for an input data tuple to MLLM-TPO. Usually, we have (X,Tq,Ta) for typical MLLM training while (X,Tq,Ai) for task token and head training. Lassign is the cross-entropy loss. Meanwhile Ltaski varies according to the task, and is usually regression- or classification-related loss.

To train MLLM-TPO, we propose a 3-stage local-toglobal training scheme, given in Figure 3. Stage 1 learns to identify the task type based on user queries. In stage 2, we train task heads along with their corresponding task tokens, respectively. Lastly, we co-train task heads with MLLM by both task data and multimodal conversation data. Our 3stage training strategy mitigates the risk of degrading the MLLM’s general abilities. We describe them as:

Stage 1: Task Assignment. We design a variety of dialogue templates for different visual tasks to perform task recognition instruction tuning for LLM. The LLM is trained using LoRA [27] in this stage.

- Stage 2: Vision Task Training. We integrate task tokens and task-specific heads into the model. By training on task-specific data, the model equips the capacity with finegrained perception and aligns between the task head and the MLLM. The region head, temporal head, mask adapter, and corresponding task tokens are trained respectively. Similarly, the LLM is updated with LoRA.
- Stage 3: Multi-task Training. We tune the entire model on the mixed corpus combining multimodal conversations and specific task data. The synergistic training allows gradients from the heads to MLLM, supervised by human annotations in vision tasks. The vision encoder, connector, task tokens, region head, temporal head, mask adapter, and LLM (with LoRA) are joint-trained in this stage.

#### 4. Experiment

We give the implementation and training & testing specifics of our TPO, and then show its results and ablations.

Implementation Details. We employ VideoChat2 [48], a video-based MLLM, as the primary framework in experiments. For its configuration, we employ UMT-L [47] as the vision encoder, Q-former in BERTbase [34] as the connector, and Mistral-7B [30] as the language model (LLM).

Regarding the three task heads, the Region Head is initialized randomly. The Temporal Head utilizes CGDETR [60], with parameters also initialized randomly. The video features input to the temporal head are extracted from the pre-trained InternVideo2 [85], while query features are extracted using the LLM [14]. The Mask Head employs SAM2 [68] and is initialized with its pre-trained weights. Since SAM2 is originally designed solely for tracking, we incorporate a two-layer MLP to encode positioning prompts for spatiotemporal grounding. Additionally, to enable the MLLM to handle the spatial locations provided by the user, we utilize another two-layer MLP to encode the spatial input into the MLLM.

Our model is trained on a variety of visual task datasets and conversation datasets, as shown in Table 1.

More training details are provided in the Appendix.

Benchmarks. We evaluate TPO on both general image/video understanding benchmarks (mainly in multiple choice form) and specific visual tasks (e.g. grounding, tracking, and so on). Specifically, we run our model along with other approaches on MVBench, VideoMME, NExTGQA, MLVU, MMIU, and SEED-Bench2. MVBench [48] is designed to evaluate multi-modal fine-grained video understanding tasks for clips (lasting around 8 to 16 seconds), consisting of 20 video tasks relying heavily on temporal perception that can hardly be addressed by single-frame analysis. Video-MME [21] is for evaluating MLLMs in both perception and reasoning across varying lengths of

Task Samples Datasets Segmentation 114.6K MeViS [17], SAMv2 [68] Temporal Grounding

ActivityNet [6], TACoS [69], QVHighlight [38], DiDeMo [26], QuerYD [62], HiREST [102], NLQ [24]

116.5K

Spatial Grounding

Allseeing-V2 [83], Visual Genome [36], RefCOCO [98], RefCOCO+ [98], RefCOCOg [58]

540.0K

YouCook2 [16], ActivityNet [6], VideoChat2-IT [48], ShareGPT-4o [13], LLaVA-Hound-DPO [108], ShareGPT4V [10]

Conversation 3M

Table 1. Overview of Datasets Used in TPO for Various Tasks.

videos, annotated by humans. NExT-GQA [90] builds on NExT-QA by introducing timestamps that are crucial for understanding questions and determining answers. This requires MLLM to perform multi-step reasoning and emphasize a deeper understanding of both visual and textual content. MLVU [112] is constructed from a wide variety of long videos, with lengths ranging from 3 minutes to 2 hours, and includes nine distinct evaluation tasks. MMIU [59] is a comprehensive evaluation suite designed to assess MLLMs across a wide range of multi-image tasks. It encompasses 7 types of multi-image relationships, 52 tasks, 77K images, and 11K meticulously curated multiple-choice questions. SEED2-Bench2 [40] is a MLLM benchmark, featuring 24K multiple-choice questions with human annotations. It spans 27 evaluation dimensions, assessing both text and image generation.

Concerning visual tasks, we test spatial grounding, moment retrieval, highlight detection, tracking, and referring segmentation, including 7 related benchmarks [17, 20, 23, 29, 38, 72, 98] , and the results from several corresponding state-of-the-art expert models.

##### 4.1. General Understanding Evaluation

Multimodal Video Understanding. TPO improves its baseline (VideoChat2) on several video understanding benchmarks with a notable margin. As shown in Table 2, VideoChat-TPO, using a 7B LLM and only 16 input frames, achieves a 66.8 average score on MVBench [48], increasing by 6.4% over VideoChat2 and exceeding the performance of ST-LLM [52] which uses 64 frames. Considering MVBench focuses on characterizing subtle temporal variations involving fine-grained action types, action order, moving direction, and so on, TPO’s seamless integration of detailed video understanding (like spatiotemporal grounding and tracking) into its optimized MLLM makes it effectively handle these fine-grained tasks.

On VideoMME [21], VideoChat-TPO outperforms the compared models, achieving a 9.3% improvement over VideoChat2 and demonstrating significant gains across short and medium-length videos, regardless of subtitle availability. On MLVU [112], VideoChat-TPO achieves a M-AVG score of 54.7, surpassing VideoChat2 by 10.2%. This confirms the effectiveness of TPO’s enhanced percep-

VideoMME [21]

LLM Params

MVBench [48]

MLVU [112] Overall Short Medium Long

Model

Frames

AVG w/o s. w/ s. w/o s. w/ s. w/o s. w/ s. w/o s. w/ s. M-AVG

TimeChat [70] 7B 96 38.5 34.3 36.9 39.1 43.1 31.8 33.9 32.1 33.6 30.9 Video-LLAVA [49] 7B 8 43.0 41.1 41.9 46.9 47.3 38.7 40.4 37.8 37.9 47.3 ShareGPT4Video [11] 7B 16 51.2 39.9 43.6 48.3 53.6 36.3 39.3 35.0 37.9 46.4 LLaVA-Next-Video [110] 7B 16 44.0 38.0 40.8 44.6 47.4 37.7 39.4 31.9 35.6 39.3 ST-LLM [52] 7B 64 54.9 37.9 42.3 45.7 48.4 36.8 41.4 31.3 36.9 PLLaVA-34B [92] 34B 16 58.1 40.0 35.0 47.2 36.2 38.2 35.9 34.7 32.9 53.6 Chat-UniVi [32] 7B 64 40.8 40.6 45.9 45.7 51.2 41.3 47.3 39.1 43.4 VideoChat2(baseline) [48] 7B 16 60.4 39.5 43.8 48.3 52.8 37.0 39.4 33.2 39.2 44.5

VideoChat-TPO 7B 16 66.8 (+6.4) 48.8 (+9.3) 53.8 (+10.0) 58.8 64.9 46.7 50.0 41.0 46.4 54.7 (+10.2)

Table 2. Performance on Multimodal Video Understanding. We compare our model to others using LLMs of the same generation or 16-frame input. w/o s. indicates without subtitle, while w s. indicates with subtitle. M-AVG refers to the mean average of MLVU.

tions of long-form video evaluations. These results across three benchmarks demonstrate the notable advancements in multimodal video understanding achieved through TPO.

Grounded Video QA. Table 3 shows that VideoChatTPO outperforms other models, achieving superior accuracy (Acc) and intersection over union (IoU) scores in NExT-GQA [90]. Its intersection over prediction (IoP) scores are comparable to those of LLoVi [105], which employs large, closed-source commercial models like GPT-

- 4 [1]. The high Acc@IoP score not only reflects the enhanced capability of TPO-optimized VideoChat in effectively understanding and interpreting video content, but also demonstrates its robustness in handling complex reasoning scenarios. Furthermore, the higher Acc@GQA score indicates that the model successfully integrates fine-grained temporal grounding with complex reasoning tasks, enabling it to accurately provide temporal clues necessary for inferring answers. From this perspective, TPO training framework significantly allows MLLMs to overcome their limitations in identifying and localizing temporal elements within video data. This enhancement positions MLLM-TPO with a competitive edge in temporal-related reasoning.

Multimodal Multi-image Understanding. To explore TPO’s effect on multi-image understanding, we test it on the MMIU and SEED-Bench2, as shown in Table 4. On MMIU, VideoChat-TPO achieved an overall score of

- 40.2, a 5.2% improvement over VideoChat2. Besides temporal changes, MMIU evaluates models’ spatial sensing and semantic relations in scenes. On SEED-Bench2, across 27 performance indicators, VideoChat-TPO achieves
- 41.7% improvement on average performance compared to VideoChat2. VideoChat-TPO’s superior performance on multi-image understanding compared to LLaVA-Interleave and InternLM-XComposer demonstrates that TPO’s vision enhancements improve the MLLM’s spatial perception and image understanding.

##### 4.2. Vision Task Evaluation

Moment Retrieval. Moment Retrieval is to locate the target segments in a video based on queries. Table 5 and 6 compare the zero-shot and fine-tuned moment retrieval performance of VideoChat-TPO against other expert models and MLLMs. In zero-shot settings, VideoChat-TPO achieves a R@1 (IoU=0.5) score of 40.2 on the CharadesSTA [23]. It surpasses the previous state-of-the-art (SOTA) MLLM specialized for temporal tasks, ChatVTG [65], and the expert model, UniVTG [50]. This demonstrates VideoChat-TPO’s ability to accurately locate video moments corresponding to given text queries, thereby enhancing the practical applicability of MLLMs.

Highlight Detection. Highlight detection generates salient scores for emphasizing frames based on the input language query. We compare the fine-tuning highlight detection performance of VideoChat-TPO with that of other expert models and MLLMs, as shown in Table 6. VideoChat-TPO notably outperforms the previous state-of-the-art MLLM (such as TimeChat [70]) on both Charades-STA and QVHighlight. For expert models like QD-DETR and UniVTG, VideoChat-TPO beats them on Charades-STA non-trivially and achieves a comparable performance on QVHighlight. This also demonstrates the progress in extending MLLMs to broad temporal tasks.

Spatial Grounding. To verify the fine-grained localization ability of the model, we run the spatial grounding task which inputs textual descriptions into the model and gets the corresponding bounding box on RefCOCO [98]. As shown in Table 7, we compare VideoChat-TPO with pixel-to-sequence models, i.e., VisionLLM-H [82], pixelto-embedding methods, i.e., NExT-Chat [104], and expert models, i.e., G-DINO [53]. VideoChat-TPO performs better than the pixel-to-embedding methods by using only a simple task head and achieves comparable performance to pixel-to-sequence models fine-tuned on a large amount of spatial grounding data, as well as to specialized models.

Model Acc@IoP Acc@GQA mIoP IoP@0.3 IoP@0.5 mIoU IoU@0.3 IoU@0.5 VIOLETv2 [22] 54.9 12.8 23.6 25.1 23.3 3.1 4.3 1.3 SeViLA [100] 72.5 16.6 29.5 34.7 22.9 21.7 29.2 13.8 LangRepo [33] 59.6 17.1 31.3 - 28.7 18.5 - 12.2 FrozenBiLM NG+ [95] 73.8 17.5 24.2 28.5 23.7 9.6 13.5 6.1 VideoStreaming [64] 57.4 17.8 32.2 - 31.0 19.3 - 13.3 LLoVi [105] 65.9 24.3 37.3 - 36.9 20.0 - 15.3 HawkEye [86] - - - - - 25.7 37.0 19.5

Model MMIU [59] SEED2I [40] SEED2M [40] LLaVA-v1.5 [51] 19.2 58.3 39.2 ShareGPT4V [10] 18.5 - OpenFlamingo [2] 22.3 36.6 43.5 LLaVA-Interleave [43] 32.4 - VideoChat2 [48] 35.0 26.5 27.6 VideoChatGPT [57] - 38.3 49.8 InternLM-XComposer [18] 21.9 65.4 49.8 VideoChat-TPO 40.2 (+5.2) 67.3 (+40.8) 70.0 (+42.4)

VideoChat-TPO 77.7 25.5 35.6 47.5 32.8 27.7 41.2 23.4

###### Table 4. Performance on Image Understanding.

Table 3. Performance on Grounded QA.

Charades-STA [23] R@0.3 R@0.5 R@0.7 mIoU UniVTG [50] 44.1 25.2 10.0 27.1 VideoChat2 [48] 38.0 14.3 3.8 24.6 VTimeLLM [28] 51.0 27.5 11.4 31.2 TimeChat [70] - 32.2 13.4 HawkEye [86] 50.6 31.4 14.5 33.7 ChatVTG [65] 52.7 33.0 15.9 34.9

Charades-STA [23] QVHighlight [38] R@0.3 R@0.5 R@0.7 mIoU mAP HIT@1

RefCOCO [98] val testA testB

Model

Methods

Model

MAttNet ⋆ [99] 76.4 80.4 69.3 OFA-L [80] 80.0 83.7 76.4 G-DINO-L ⋆ [53] 90.6 93.2 88.2

M-DETR [38] 65.8 52.1 30.6 45.5 35.7 55.6 QD-DETR [61] - 57.3 32.6 - 38.9 62.4 UniVTG [50] 72.6 60.2 38.6 52.2 40.5 66.3

VisionLLM-H [82] - 86.7 Shikra-7B [8] 87.0 90.6 80.2 NExT-Chat-7B [104] 85.5 90.0 77.9

TimeChat [70] - 46.7 23.7 - 21.7 37.9 HawkEye [86] 72.5 58.3 28.8 - - -

VideoChat-TPO 77.0 65.0 40.7 55.0 38.6 65.4

VideoChat-TPO 85.9 90.8 81.3

VideoChat-TPO 58.3 40.2 18.4 38.1

Table 7. Spatial Grounding Task. ⋆ with a refined decoder.

Table 6. Fine-tuning Performance on Moment Retrieval and Highlight Detection. Gray means no LLM.

Table 5. Zero-Shot Performance on Moment Retrieval.Gray means no LLM.

Tracking. In the tracking task, the model receives the object coordinates from the first frame and outputs the coordinates for the remaining frames in the video. We evaluate the VideoChat-TPO on the mainstream tracking benchmarks LAOST [20] and GOT-10k [29], as shown in Table 8. In zero-shot testing, VideoChat-TPO ranks best among all MLLMs, even surpassing some fine-tuned expert models notably with around 10% increase on success rate, such as SiamFC [5] and ATOM [15]. With the TPO training method, the model is optimized from mask sequences that hard to represent by words, allowing the MLLM to achieve strong motion characterization for multimodal generalization in highly dynamic and occluded scenes.

Referring Segmentation. The referring segmentation task requires the model to output the mask sequence corresponding to the specified prompt. This capability is newly activated in VideoChat-TPO, as it is not supported or designed in the original VideoChat2 or SAM2. We compare the referring segmentation performance of VideoChat-TPO with other expert models and MLLMs in Table 9. The zero shot capability of VideoChat-TPO is comparable to the finetuning capabilities of other expert models, i.e. ReferFormer [88], showing its notable open-world video segmentation performance. Through TPO training, MLLM effectively optimizes its understanding in both tracking and segmenting objects indicated by users. It further enables the model to excel in pixel-level tasks, offering perceptual advantages to it in practical applications like robotic control.

##### 4.3. Ablation Studies

In this section, we analyze the effectiveness of the key components of TPO and evaluate its scalability.

Extending to Other MLLMs. To evaluate the effectiveness of the TPO method on different MLLMs, we apply TPO to LLaVA (LLaVA-OneVision [41]) in addition to VideoChat. In LLaVA, fine-grained video understanding is a weak spot. Despite the conversation data used in LLaVAOneVision being very similar to, or even more extensive than, what we employ, there remains significant room for improvement in its performance on short-term fine-grained understanding tasks. As shown in Table 10, with TPO, LLaVA not only gains capabilities that it originally lacked but also achieves an 8.1% improvement on MVBench, demonstrating the universality of the TPO method.

Task Preference Model vs. Textualized Task Data. We compare the TPO method with the approach of textualizing task data and training it in an autoregressive manner, which is a straightforward and efficient method for MLLMs to learn specific tasks. We use the same data as the TPO method and convert the data related to the task head into conversation format. Due to the limitations of textual representation, much data must be input in a more discrete manner. For instance, in tracking supervision data, we follow Merlin [97] by converting masks into a sequence of spatial coordinates to serve as both the model’s input and output.

As shown in Table 11, MLLM-TPO achieves a 2.1% performance improvement on MVBench [48] compared to MLLMs trained with textualized task data.

This demonstrates that the task head in the TPO method enables more effective utilization of the original supervision signals compared to the next-token prediction approach, which inevitably incurs information loss when converting fine-grained tasks into textual annotations. The next-token prediction method cannot truly capture the nuanced information present in videos, while TPO allows the MLLM to learn perceptual information beyond conversational data.

Ref-YouTube-VOS [72] MeViS [17]

LaSOT [20] GOT-10k [29] Success Pnorm P Overlap SR0.5 SR0.75 SiamFC [5] 33.6 42.0 33.9 34.8 35.3 9.8 ATOM [15] 51.5 - - 55.6 63.4 40.2 SiamRPN++ [39] 49.6 56.9 49.1 51.8 61.8 32.5 SiamFC++ [93] 54.4 62.3 54.7 59.5 69.5 47.9

Model MVBench LLaVA-OV [41] 56.7 LLaVA-OV-TPO 64.8 (+8.1)

Model

Method

J &F J F J &F J F ReferFormer [88] 62.9 61.3 64.6 31.0 29.8 32.2 OnlineRefer [87] 62.9 61.0 64.7 - - LISA [37] 52.6 52.1 53.0 - - VideoLISA [4] 63.7 61.7 63.7 44.4 41.3 47.6 VideoChat-TPO 63.9 52.3 75.4 47.0 42.6 51.3

Table 10. Applying TPO on LLaVA-OV. TPO significantly enhances the ability to understand fine-grained video details.

LLaVA-1.5 [51] 19.4 16.5 12.8 23.5 20.2 9.7 Merlin [97] 39.8 40.2 38.1 51.4 55.9 42.8 VideoChat-TPO 69.4 80.1 76.9 70.6 79.8 66.0

Table 9. Performance on Referring Segmentation. TPO is evaluated in a zero-shot on Ref-YouTube.

Table 8. Performance on Tracking Benchmarks.

Charades-STA [23] MVBench R@0.3 R@0.5 R@0.7 mIoU AVG

Model

VideoChat-TPO 58.3 40.2 18.4 38.1 66.8 w/o reasoning data 56.4 38.3 17.1 35.6 66.1 replace by simple head 31.5 17.8 6.1 15.4 65.8 textualized task data 33.7 18.6 6.2 16.0 64.7

- Table 11. Ablation of Reasoning Data and Head Performance.

Exp. T R M C R@0.5 Acc@0.5 J &F

- 1 ✓ 30.2 - -

- 2 ✓ - 77.3 -

- 3 ✓ - - 55.1

- 4 ✓ ✓ - 80.2 58.3

- 5 ✓ ✓ ✓ 36.7 81.6 61.4

- 6 ✓ ✓ ✓ ✓ 40.2 82.0 63.9

- Table 12. Impact of TPO Components and Data. T, R, M, and C denote temporal head, region head, mask head, and conversation data respectively. R1@0.5 means R1@0.5 in Charades-STA, Acc@0.5 represents the mean of Acc@0.5 in all COCO datasets, J &F means J &F in Ref-YouTube-VOS.

Impact of Task Head Performance on TPO. A stronger task head is expected to enhance the multimodal capabilities of TPO more effectively than a weaker one. To further explore the impact of the task head on model performance, we replaced the temporal head from CG-DETR [60] with a simple two-layer MLP.

As shown in Table 11, the simple temporal head exhibits a significant performance decline compared to CG-DETR on the corresponding tasks. However, it still achieves a 1.1% improvement compared to the model without TPO. In contrast to the simpler head, the well-designed head can provide more accurate expert knowledge through its architecture and pre-trained task preference weights, enabling better utilization of the data.

Impact of Data Scaling. Concerning foundation or large models, we expect to enhance their model performance by incorporating additional data at scale. In our approach, we expand the existing conversation data by integrating two reasoning datasets [6, 16] with temporal information. By incorporating these datasets, we observe a notable enhancement in the capabilities of the temporal head in Table 11. Furthermore, as shown in Table 12, adding conversation data further improves performance across various

tasks. Overall, increasing the amount of conversation and fine-grained task data leads to noticeable improvements in model performance, demonstrating the effectiveness of data scaling in training multimodal models with TPO.

Synergistic Gains from Co-training. Table 12 presents the impact of incorporating different visual tasks into TPO on performance.

This indicates that the collaborative learning of visual tasks facilitates the transfer of knowledge, ultimately leading to better performance across the board. Overall, these results confirm that co-training not only benefits individual tasks but also creates a synergistic effect that enhances overall capabilities in visual understanding.

#### 5. Conclusions

This paper introduces Task Preference Optimization (TPO). It enhances the overall multimodal performance of MLLMs by enhancing their precise visual understanding. TPO achieves this by integrating differentiable task preferences derived from fine-grained visual tasks. It introduces learnable task tokens that serve as bridges between multiple task-specific heads and the MLLM. Through the joint optimization of these task tokens, heads, and the MLLM, TPO leads to a substantial improvement in multimodal dialogue capabilities and performance across various visual tasks. Our results demonstrate the effectiveness of TPO in scaling MLLMs with task-specific data and seamlessly integrating them with existing expert vision models. We believe this study clarifies the prerequisites for fusing MLLMs with models and data from the pre-large model era, effectively bridging the gap between expert and generalist models, as well as between generation and understanding.

Limitations. Currently, TPO focuses exclusively on discriminative visual tasks, overlooking generative ones. Meanwhile, the framework is supervised by human annotations, which neglects potentially valuable unsupervised or self-supervised learning approaches such as contrastive learning [12, 25]. This limitation inherently restricts the scalability of TPO in terms of both task diversity and requirements. While we demonstrate TPO’s potential for enhancing MLLMs through increased task-specific data, a comprehensive investigation into the broader impact of this scaling remains a crucial area for future research.

#### Acknowledgements

This work is partially supported by the National Key R&D Program of China (No. 2022ZD0160102), Jiangsu Frontier Technology R&D Program (No. BF2024076), and the Science and Technology Commission of Shanghai Municipality under Grant No. 23YF1461900.

#### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 6
- [2] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, Jenia Jitsev, Simon Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Wortsman, and Ludwig Schmidt. Openflamingo: An opensource framework for training large autoregressive visionlanguage models. arXiv preprint arXiv:2308.01390, 2023. 7, 14
- [3] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 14
- [4] Zechen Bai, Tong He, Haiyang Mei, Pichao Wang, Ziteng Gao, Joya Chen, Lei Liu, Zheng Zhang, and Mike Zheng Shou. One token to seg them all: Language instructed reasoning segmentation in videos. arXiv preprint arXiv:2409.19603, 2024. 3, 8
- [5] Luca Bertinetto, Jack Valmadre, Joao F Henriques, Andrea Vedaldi, and Philip HS Torr. Fully-convolutional siamese networks for object tracking. In ECCV, pages 850–865,

2016. 7, 8

- [6] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In CVPR, pages 961–970, 2015. 5, 8, 15
- [7] Kang Chen, Tao Han, Junchao Gong, Lei Bai, Fenghua Ling, Jing-Jia Luo, Xi Chen, Leiming Ma, Tianning Zhang, Rui Su, et al. Fengwu: Pushing the skillful global mediumrange weather forecast beyond 10 days lead. arXiv preprint arXiv:2304.02948, 2023. 1
- [8] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023. 1, 3, 7
- [9] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023. 15
- [10] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. In ECCV, 2024. 5, 7, 14, 16

- [11] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, Li Yuan, Yu Qiao, Dahua Lin, Feng Zhao, and Jiaqi Wang. Sharegpt4video: Improving video understanding and generation with better captions. arXiv preprint arXiv:2406.04325, 2024. 6, 14
- [12] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML, pages 1597–1607. PMLR, 2020. 8
- [13] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In CVPR, pages 24185–24198, 2024. 5, 14, 15, 16
- [14] Yiming Cui, Ziqing Yang, and Xin Yao. Efficient and effective text encoding for chinese llama and alpaca. arXiv preprint arXiv:2304.08177, 2023. 5, 15
- [15] Martin Danelljan, Goutam Bhat, Fahad Shahbaz Khan, and Michael Felsberg. Atom: Accurate tracking by overlap maximization. In CVPR, pages 4660–4669, 2019. 7, 8
- [16] Pradipto Das, Chenliang Xu, Richard F Doell, and Jason J Corso. A thousand frames in just a few words: Lingual description of videos through latent topics and sparse object stitching. In CVPR, pages 2634–2641, 2013. 5, 8, 15
- [17] Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, and Chen Change Loy. MeViS: A large-scale benchmark for video segmentation with motion expressions. In ICCV,

2023. 5, 8, 15

- [18] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, et al. Internlm-xcomposer2: Mastering free-form text-image composition and comprehension in vision-language large model. arXiv preprint arXiv:2401.16420, 2024. 7, 14
- [19] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, et al. Palm-e: An embodied multimodal language model. 1
- [20] Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao, and Haibin Ling. Lasot: A high-quality benchmark for large-scale single object tracking. In CVPR, pages 5374–5383, 2019. 5, 7, 8
- [21] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024. 2, 5, 6, 14
- [22] Tsu-Jui Fu, Linjie Li, Zhe Gan, Kevin Lin, William Yang Wang, Lijuan Wang, and Zicheng Liu. An empirical study of end-to-end video-language transformers with masked visual modeling. In CVPR, 2023. 7
- [23] Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia. Tall: Temporal activity localization via language query. In ICCV, pages 5267–5275, 2017. 5, 6, 7, 8, 14

- [24] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In CVPR, pages 18995–19012, 2022. 5, 15
- [25] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, pages 9729–9738, 2020. 8
- [26] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In ICCV, 2017. 5, 15
- [27] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Lowrank adaptation of large language models. In ICLR, 2022. 4, 14
- [28] Bin Huang, Xin Wang, Hong Chen, Zihan Song, and Wenwu Zhu. Vtimellm: Empower llm to grasp video moments. In CVPR, 2024. 7
- [29] Lianghua Huang, Xin Zhao, and Kaiqi Huang. Got-10k: A large high-diversity benchmark for generic object tracking in the wild. TPAMI, 43(5):1562–1577, 2019. 5, 7, 8
- [30] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023. 5
- [31] Fan Jiang and Zilei Wang. Sparse sharing relation network for panoptic driving perception. In ACMMM, pages 800– 808, 2023. 1
- [32] Peng Jin, Ryuichi Takanobu, Wancai Zhang, Xiaochun Cao, and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding. In CVPR, pages 13700–13710, 2024. 6
- [33] Kumara Kahatapitiya, Kanchana Ranasinghe, Jongwoo Park, and Michael S Ryoo. Language repository for long video understanding. arXiv preprint arXiv:2403.14622,

2024. 7

- [34] Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In naacL-HLT, page 2. Minneapolis, Minnesota, 2019. 5
- [35] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, pages 4015–4026, 2023. 3
- [36] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. IJCV, 123:32–73, 2017. 5, 15
- [37] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. In CVPR, pages 9579–9589,

2024. 3, 8

- [38] Jie Lei, Tamara L Berg, and Mohit Bansal. Detecting moments and highlights in videos via natural language queries. NeurIPS, 34:11846–11858, 2021. 5, 7, 13, 15

- [39] Bo Li, Wei Wu, Qiang Wang, Fangyi Zhang, Junliang Xing, and Junjie Yan. SiamRPN++: Evolution of siamese visual tracking with very deep networks. In CVPR, pages 4282– 4291, 2019. 8
- [40] Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. Seed-bench: Benchmarking multimodal large language models. In CVPR, pages 13299–13308, 2024. 2, 5, 7
- [41] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 2, 7, 8
- [42] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. CoRR, abs/2408.03326, 2024. 14
- [43] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024. 7, 14
- [44] Hao Li, Jinguo Zhu, Xiaohu Jiang, Xizhou Zhu, Hongsheng Li, Chun Yuan, Xiaohua Wang, Yu Qiao, Xiaogang Wang, Wenhai Wang, et al. Uni-perceiver v2: A generalist model for large-scale vision and vision-language tasks. In CVPR, pages 2691–2700, 2023. 2, 3
- [45] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, pages 19730–19742. PMLR, 2023. 3
- [46] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 2, 3, 14
- [47] Kunchang Li, Yali Wang, Yizhuo Li, Yi Wang, Yinan He, Limin Wang, and Yu Qiao. Unmasked teacher: Towards training-efficient video foundation models. In ICCV, pages 19948–19960, 2023. 2, 5
- [48] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In CVPR, pages 22195–22206, 2024. 2, 5, 6, 7, 14, 15, 16
- [49] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023. 6, 14
- [50] Kevin Qinghong Lin, Pengchuan Zhang, Joya Chen, Shraman Pramanick, Difei Gao, Alex Jinpeng Wang, Rui Yan, and Mike Zheng Shou. Univtg: Towards unified videolanguage temporal grounding. In ICCV, pages 2794–2804,

2023. 1, 6, 7

- [51] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2024. 2, 3, 7, 8, 14
- [52] Ruyang Liu, Chen Li, Haoran Tang, Yixiao Ge, Ying Shan, and Ge Li. St-llm: Large language models are effective temporal learners. In ECCV, 2024. 5, 6, 14

- [53] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In ECCV, 2024. 6, 7
- [54] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 14, 15
- [55] Jiasen Lu, Christopher Clark, Rowan Zellers, Roozbeh Mottaghi, and Aniruddha Kembhavi. Unified-io: A unified model for vision, language, and multi-modal tasks. In ICLR, 2022. 2
- [56] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Khan. Videogpt+: Integrating image and video encoders for enhanced video understanding. arXiv preprint arXiv:2406.09418, 2024. 14
- [57] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In ACL, 2024. 3, 7, 14
- [58] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In CVPR, pages 11–20, 2016. 5, 15
- [59] Fanqing Meng, Jin Wang, Chuanhao Li, Quanfeng Lu, Hao Tian, Jiaqi Liao, Xizhou Zhu, Jifeng Dai, Yu Qiao, Ping Luo, et al. Mmiu: Multimodal multi-image understanding for evaluating large vision-language models. arXiv preprint arXiv:2408.02718, 2024. 2, 5, 7, 14
- [60] WonJun Moon, Sangeek Hyun, SuBeen Lee, and JaePil Heo. Correlation-guided query-dependency calibration in video representation learning for temporal grounding. arXiv preprint arXiv:2311.08835, 2023. 5, 8, 15
- [61] WonJun Moon, Sangeek Hyun, SangUk Park, Dongchan Park, and Jae-Pil Heo. Query-dependent video representation for moment retrieval and highlight detection. In CVPR, pages 23023–23033, 2023. 7
- [62] Andreea-Maria Oncescu, Joao F Henriques, Yang Liu, Andrew Zisserman, and Samuel Albanie. Queryd: A video dataset with high-quality text and audio narrations. In ICASSP, pages 2265–2269. IEEE, 2021. 5, 15
- [63] Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Mateusz Malinowski, Yi Yang, Carl Doersch, et al. Perception test: A diagnostic benchmark for multimodal video models. 2024. 14
- [64] Rui Qian, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Shuangrui Ding, Dahua Lin, and Jiaqi Wang. Streaming long video understanding with large language models. arXiv preprint arXiv:2405.16009, 2024. 7
- [65] Mengxue Qu, Xiaodong Chen, Wu Liu, Alicia Li, and Yao Zhao. Chatvtg: Video temporal grounding via chat with video dialogue large language models. In CVPR, pages 1847–1856, 2024. 6, 7
- [66] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. NeurIPS, 36, 2024. 2, 3

- [67] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In SIGKDD, pages 3505–3506, 2020. 15
- [68] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, ChaoYuan Wu, Ross Girshick, Piotr Doll´ar, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 4, 5, 15
- [69] Michaela Regneri, Marcus Rohrbach, Dominikus Wetzel, Stefan Thater, Bernt Schiele, and Manfred Pinkal. Grounding action descriptions in videos. Transactions of the Association for Computational Linguistics, 1:25–36, 2013. 5, 15
- [70] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. Timechat: A time-sensitive multimodal large language model for long video understanding. CVPR, abs/2312.02051, 2024. 1, 3, 6, 7, 14
- [71] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 3
- [72] Seonguk Seo, Joon-Young Lee, and Bohyung Han. Urvos: Unified referring video object segmentation network with a large-scale benchmark. In ECCV, pages 208–223. Springer,

2020. 5, 8

- [73] Jing Shao, Siyu Chen, Yangguang Li, Kun Wang, Zhenfei Yin, Yinan He, Jianing Teng, Qinghong Sun, Mengya Gao, Jihao Liu, et al. Intern: A new learning paradigm towards general vision. arXiv preprint arXiv:2111.08687, 2021. 1, 2
- [74] Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, Liang-Yan Gui, Yu-Xiong Wang, Yiming Yang, et al. Aligning large multimodal models with factually augmented rlhf. arXiv preprint arXiv:2309.14525, 2023. 3
- [75] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 1
- [76] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. In NeurIPS, pages 10078–10093, 2022. 2
- [77] Haibo Wang, Zhiyang Xu, Yu Cheng, Shizhe Diao, Yufan Zhou, Yixin Cao, Qifan Wang, Weifeng Ge, and Lifu Huang. Grounded-videollm: Sharpening fine-grained temporal grounding in video large language models. arXiv preprint arXiv:2410.03290, 2024. 3
- [78] Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, and Luc Van Gool. Temporal segment networks: Towards good practices for deep action recognition. In ECCV, pages 20–36. Springer, 2016. 2
- [79] Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, and Yu Qiao. Videomae v2:

- Scaling video masked autoencoders with dual masking. In CVPR, 2023. 2
- [80] Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In ICML, pages 23318–23340. PMLR, 2022. 7
- [81] Rui Wang, Dongdong Chen, Zuxuan Wu, Yinpeng Chen, Xiyang Dai, Mengchen Liu, Lu Yuan, and Yu-Gang Jiang. Masked video distillation: Rethinking masked feature modeling for self-supervised video representation learning. In CVPR, pages 6312–6322, 2023. 2
- [82] Wenhai Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Yu Qiao, et al. Visionllm: Large language model is also an open-ended decoder for vision-centric tasks. In NeurIPS,

2024. 3, 6, 7

- [83] Weiyun Wang, Yiming Ren, Haowen Luo, Tiantong Li, Chenxiang Yan, Zhe Chen, Wenhai Wang, Qingyun Li, Lewei Lu, Xizhou Zhu, et al. The all-seeing project v2: Towards general relation comprehension of the open world. arXiv preprint arXiv:2402.19474, 2024. 5, 15, 16
- [84] Xinlong Wang, Xiaosong Zhang, Yue Cao, Wen Wang, Chunhua Shen, and Tiejun Huang. Seggpt: Towards segmenting everything in context. In ICCV, pages 1130–1140,

2023. 2

- [85] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Jilan Xu, Zun Wang, et al. Internvideo2: Scaling video foundation models for multimodal video understanding. In ECCV, 2024. 2, 5, 15
- [86] Yueqian Wang, Xiaojun Meng, Jianxin Liang, Yuxuan Wang, Qun Liu, and Dongyan Zhao. Hawkeye: Training video-text llms for grounding text in videos. arXiv preprint arXiv:2403.10228, 2024. 3, 7
- [87] Dongming Wu, Tiancai Wang, Yuang Zhang, Xiangyu Zhang, and Jianbing Shen. Onlinerefer: A simple online baseline for referring video object segmentation. In ICCV, pages 2761–2770, 2023. 8
- [88] Jiannan Wu, Yi Jiang, Peize Sun, Zehuan Yuan, and Ping Luo. Language as queries for referring video object segmentation. In CVPR, pages 4974–4984, 2022. 7, 8
- [89] Jiannan Wu, Muyan Zhong, Sen Xing, Zeqiang Lai, Zhaoyang Liu, Wenhai Wang, Zhe Chen, Xizhou Zhu, Lewei Lu, Tong Lu, et al. Visionllm v2: An end-to-end generalist multimodal large language model for hundreds of vision-language tasks. arXiv preprint arXiv:2406.08394,

2024. 3

- [90] Junbin Xiao, Angela Yao, Yicong Li, and Tat-Seng Chua. Can i trust your answer? visually grounded video question answering. In CVPR, 2024. 5, 6
- [91] Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and Christoph Feichtenhofer. Videoclip: Contrastive pretraining for zero-shot video-text understanding. In EMNLP, pages 6787–6800, 2021. 2
- [92] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. Pllava: Parameter-free llava extension

- from images to videos for video dense captioning. arXiv preprint arXiv:2404.16994, 2024. 6, 14
- [93] Yinda Xu et al. Siamfc++: Towards robust and accurate visual tracking with target estimation guidelines. In AAAI, pages 140–148, 2020. 8
- [94] Shen Yan, Tao Zhu, Zirui Wang, Yuan Cao, Mi Zhang, Soham Ghosh, Yonghui Wu, and Jiahui Yu. Video-text modeling with zero-shot transfer from contrastive captioners. ArXiv, abs/2212.04979, 2022. 2
- [95] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Zero-shot video question answering via frozen bidirectional language models. In NeurIPS, 2022. 7
- [96] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, Chaoya Jiang, Chenliang Li, Yuanhong Xu, Hehong Chen, Junfeng Tian, Qian Qi, Ji Zhang, and Fei Huang. mplug-owl: Modularization empowers large language models with multimodality, 2023. 3
- [97] En Yu, Liang Zhao, Yana Wei, Jinrong Yang, Dongming Wu, Lingyu Kong, Haoran Wei, Tiancai Wang, Zheng Ge, Xiangyu Zhang, et al. Merlin: Empowering multimodal llms with foresight minds. In ECCV, pages 425–443. Springer, 2025. 3, 7, 8
- [98] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In ECCV, pages 69–85. Springer, 2016. 5, 6, 7, 15
- [99] Licheng Yu, Zhe Lin, Xiaohui Shen, Jimei Yang, Xin Lu, Mohit Bansal, and Tamara L Berg. Mattnet: Modular attention network for referring expression comprehension. In CVPR, pages 1307–1315, 2018. 7
- [100] Shoubin Yu, Jaemin Cho, Prateek Yadav, and Mohit Bansal. Self-chained image-language model for video localization and question answering. In NeurIPS, 2023. 7
- [101] Tianyu Yu, Yuan Yao, Haoye Zhang, Taiwen He, Yifeng Han, Ganqu Cui, Jinyi Hu, Zhiyuan Liu, Hai-Tao Zheng, Maosong Sun, et al. Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback. In CVPR, pages 13807–13816, 2024. 3
- [102] Abhay Zala, Jaemin Cho, Satwik Kottur, Xilun Chen, Barlas Oguz, Yashar Mehdad, and Mohit Bansal. Hierarchical video-moment retrieval and step-captioning. In CVPR, pages 23056–23065, 2023. 5, 15
- [103] Xiangyu Zeng, Kunchang Li, Chenting Wang, Xinhao Li, Tianxiang Jiang, Ziang Yan, Songze Li, Yansong Shi, Zhengrong Yue, Yi Wang, et al. Timesuite: Improving mllms for long video understanding via grounded tuning. In ICLR. 1
- [104] Ao Zhang, Liming Zhao, Chen-Wei Xie, Yun Zheng, Wei Ji, and Tat-Seng Chua. Next-chat: An lmm for chat, detection and segmentation. arXiv preprint arXiv:2311.04498,

2023. 3, 6, 7

- [105] Ce Zhang, Taixi Lu, Md Mohaiminul Islam, Ziyang Wang, Shoubin Yu, Mohit Bansal, and Gedas Bertasius. A simple llm framework for long-range video question-answering. arXiv preprint arXiv:2312.17235, 2023. 6, 7
- [106] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. 14

- [107] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. In EMNLP, pages 543–553, 2023. 3
- [108] Ruohong Zhang, Liangke Gui, Zhiqing Sun, Yihao Feng, Keyang Xu, Yuanhan Zhang, Di Fu, Chunyuan Li, Alexander Hauptmann, Yonatan Bisk, et al. Direct preference optimization of video large multimodal models from language model reward. arXiv preprint arXiv:2404.01258, 2024. 5, 15, 16
- [109] Yanzhe Zhang, Ruiyi Zhang, Jiuxiang Gu, Yufan Zhou, Nedim Lipka, Diyi Yang, and Tong Sun. Llavar: Enhanced visual instruction tuning for text-rich image understanding. arXiv preprint arXiv:2306.17107, 2023. 3
- [110] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llava-next: A strong zero-shot video understanding model,

2024. 6

- [111] Long Zhao, Nitesh Bharadwaj Gundavarapu, Liangzhe Yuan, Hao Zhou, Shen Yan, Jennifer J Sun, Luke Friedman, Rui Qian, Tobias Weyand, Yue Zhao, et al. Videoprism: A foundational visual encoder for video understanding. In ICML, 2024. 2
- [112] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: A comprehensive benchmark for multi-task long video understanding. arXiv preprint arXiv:2406.04264, 2024. 2, 5, 6
- [113] Yiyang Zhou, Chenhang Cui, Rafael Rafailov, Chelsea Finn, and Huaxiu Yao. Aligning modalities in vision large language models via preference fine-tuning. In ICLR Workshop on Reliable and Responsible Foundation Models,

2024. 3

- [114] Jinguo Zhu, Xizhou Zhu, Wenhai Wang, Xiaohua Wang, Hongsheng Li, Xiaogang Wang, and Jifeng Dai. Uniperceiver-moe: Learning sparse generalist models with conditional moes. In NeurIPS, pages 2664–2678, 2022. 2, 3
- [115] Xizhou Zhu, Jinguo Zhu, Hao Li, Xiaoshi Wu, Hongsheng Li, Xiaohua Wang, and Jifeng Dai. Uni-perceiver: Pretraining unified architecture for generic perception for zeroshot and few-shot tasks. In CVPR, pages 16804–16815,

2022. 2, 3

Appendix

1. Experiment Details

MVbench. We present the detailed performance of MVBench in Table 13, VideoChat-TPO achieves an average score of 66.8, increasing by 6.4 points based on VideoChat2. It gets superior performance among MLLMs with the same number of input frames and LLMs of comparable model scale. In Action Localization, temporal labels in the VideoChat2-Textualized-Task are defined as text. While the model demonstrates strong capabilities in zeroshot temporal grounding, converting the task into a QA

problem does not improve performance. However, by optimizing with TPO, the model can benefit from original label supervision, resulting in corresponding performance enhancements. Also, Its superior performance is particularly evident in tasks that require moment-based perception and reasoning, including Action Sequence (AS), Action Localization (AL) and Action Prediction (AP), with scores of 84.0 (+7.5%), 55.0 (+10%), and 69.5 (+13.5%) respectively. This demonstrates the excellent potential of TPO in sophisticated video understanding tasks.

MMIU. The results are shown in Table 14. VideoChatTPO shows a significant improvement over VideoChat2, achieving an overall score of 40.2 (+5.2%). Compared with VideoChat2, Our model has achieved clear improvements in Causality Reasoning (CR), Visually Grounded Reasoning (VGR), Multiple Image Captioning (MIC), Spot the Difference (STD), General Action Recognition (GAR), Temporal Localization (TL), Video Captioning (VidCap), Multiview Action Recognition (MAR), Image Captioning with Spatial Context (ICSC), and Egocentric Video Question Answering (EVQA), with scores of 73.0 (+26.5%), 69.5 (+15.2%), 83.0 (+19%), 92.5 (+61%), 88.0 (+15%), 94.5 (+13.0%), 73.4 (+35.8%), 48.5 (+12%) and 59.0 (14.5%), respectively. Among them, we suppose the improvement of TL capability comes from the optimization of our temporal head, and the improvement of VGR, STD, MAR and ICSC capabilities comes from the optimization of our region head and mask head. The enhancements observed in captioning, specifically in metrics such as MIC, IC, and VidCap, indicate an improvement of TPO to capture detailed visuals. Meanwhile, we find that the improvement in multi-image capabilities stems from enhanced instruction comprehension. Compared with video assessments, which primarily consist of multiple-choice questions, multi-image evaluations emphasize the accuracy of responses to specific questions. After optimization with TPO, the model has significantly improved its instruction following, leading to a higher success rate.

How Scaling Task Data Affect MLLMs. We perform an ablation experiment on the dataset of stage 2 to evaluate the impact of the task training data on the model performance. Specifically, we reduce the number of temporal grounding datasets from six to one (QVHighlight [38]). As shown in Table 15, using only one dataset leads to slightly worse conversational performance (-0.3%) on MVBench and significantly poorer expert task performance (-5.6%) on CharadesSTA R@0.5, when compared to employing multiple temporal grounding datasets for training the temporal task head. Notably, this approach remains more effective than training after textualizing the task data in QA tasks like MVBench. This finding indicates that scaling task data gives notable

Model Avg. AS AP AA FA UA OE OI OS MD AL ST AC MC MA SC FP CO EN ER CI

VideoChatGPT [57] 32.7 23.5 26.0 62.0 22.5 26.5 54.0 28.0 40.0 23.0 20.0 31.0 30.5 25.5 39.5 48.5 29.0 33.0 29.5 26.0 35.5 VideoLLaMA [106] 34.1 27.5 25.5 51.0 29.0 39.0 48.0 40.5 38.0 22.5 22.5 43.0 34.0 22.5 32.5 45.5 32.5 40.0 30.0 21.0 37.0 VideoChat [46] 35.5 33.5 26.5 56.0 33.5 40.5 53.0 40.5 30.0 25.5 27.0 48.5 35.0 20.5 42.5 46.0 26.5 41.0 23.5 23.5 36.0 TimeChat [70] 38.5 40.5 36.0 61.0 32.5 53.0 53.5 41.5 29.0 19.5 26.5 66.5 34.0 20.0 43.5 42.0 36.5 36.0 29.0 35.0 35.0 Video-LLaVA [49] 43.0 46.0 42.5 56.5 39.0 53.5 53.0 48.0 41.0 29.0 31.5 82.5 45.0 26.0 53.0 41.5 33.5 41.5 27.5 38.5 31.5 P-LLaVA-7B [92] 46.6 58.0 49.0 55.5 41.0 61.0 56.0 61.0 36.0 23.5 26.0 82.0 39.5 42.0 52.0 45.0 42.0 53.5 30.5 48.0 31.0 ShareGPT4Video [11] 51.2 49.5 39.5 79.5 40.0 54.5 82.5 54.5 32.5 50.5 41.5 84.5 35.5 62.5 75.0 51.0 25.5 46.5 28.5 39.0 51.5 ST-LLM [52] 54.9 66.0 53.5 84.0 44.0 58.5 80.5 73.5 38.5 42.5 31.0 86.5 36.5 56.5 78.5 43.0 44.5 46.5 34.5 41.5 58.5 VideoGPT+ [56] 58.7 69.0 60.0 83.0 48.5 66.5 85.5 75.5 36.0 44.0 34.0 89.5 39.5 71.0 90.5 45.0 53.0 50.0 29.5 44.0 60.0 VideoChat2 [48] 60.4 75.5 58.0 83.5 50.5 60.5 87.5 74.5 45.0 47.5 44.0 82.5 37.0 64.5 87.5 51.0 66.5 47.0 35.0 37.0 72.5

VideoChat2-textualized-task 64.8 76.5 56.0 88.5 52.5 77.0 92.5 74.0 41.0 50.5 45.0 87.0 47.0 74.0 89.0 48.0 85.0 45.0 34.0 58.5 73.0 VideoChat-TPO 66.8 84.0 69.5 87.5 52.0 77.0 92.0 81.0 40.5 42.5 55.0 89.0 47.5 68.0 89.0 58.0 87.0 57.5 27.0 60.0 72.0

###### Table 13. Results on MVBench Multi-choice Question Answering.

Model Overall CR ER FD FC SC VCor VQA VGR FR HR I2IR MIC PR S2IR STD STS T2IR VR AQA GAR MVU MEV NIP TL TO VidCap

GuAR GNAP TC VClz VCo VO EVQA HE IQASC ICSC ISTE ITRSC MAR MR JPS 3DE 3DOD 3DOT 3DPE 3DSR 3DQA PT RPM SOT 3DCR 3DIR OpenFlamingo [2] 22.3 25.5 25.8 24.6 21.6 25.0 28.2 34.5 49.0 14.5 19.0 13.5 22.5 17.5 26.0 39.0 49.0 20.0 27.5 10.0 13.5 16.5 30.0 20.0 18.7 24.5 22.5

25.0 21.5 25.5 25.0 14.5 13.5 15.5 27.5 4.0 25.5 23.0 7.0 22.1 3.0 1.5 26.5 22.0 35.0 17.0 28.5 20.5 23.5 11.5 31.0 25.0 23.5 XComposer2 [18] 21.9 24.0 21.0 10.8 5.8 0.0 0.0 34.2 24.0 14.5 2.5 23.0 63.5 19.0 26.0 14.5 31.0 9.5 28.5 31.5 59.5 44.0 30.0 4.5 15.5 12.0 66.0

55.0 35.0 42.5 22.5 2.5 19.0 20.0 8.0 15.5 45.0 0.0 0.0 20.6 0.0 16.5 0.0 7.0 0.0 4.5 0.0 33.5 63.0 1.5 38.5 42.0 33.0 Qwen-chat [3] 15.9 20.5 2.5 13.3 2.5 9.9 5.9 31.2 23.8 10.5 19.5 12.5 41.0 5.5 13.5 29.5 45.0 3.0 12.0 10.0 52.5 18.5 16.5 2.5 3.6 5.5 47.0

29.0 23.0 18.0 6.0 6.0 6.0 32.0 9.0 13.5 17.0 15.5 3.5 40.2 15.8 16.5 16.5 22.5 17.5 13.0 14.5 14.0 8.0 3.0 8.5 1.5 0.5

LLaVA-v1.5 [51] 19.2 14.1 4.2 13.7 5.8 1.9 6.9 27.3 35.0 6.5 12.5 12.5 53.0 10.0 25.5 66.5 43.0 19.0 3.5 2.5 23.5 36.5 12.0 16.5 6.7 7.0 28.0 24.5 17.5 40.0 15.0 21.5 4.0 26.0 7.5 26.5 17.5 5.0 4.5 25.6 27.1 8.5 8.0 4.0 6.0 6.0 14.5 29.5 66.0 2.0 35.0 34.5 28.5

ShareGPT4V [10] 18.5 16.4 5.0 10.8 6.2 9.0 2.7 34.2 28.5 4.5 10.5 3.5 57.0 4.0 12.5 55.5 44.5 13.5 5.0 5.0 26.0 38.0 14.0 15.5 10.9 6.0 25.0 26.5 19.0 42.0 7.5 14.0 7.5 31.5 7.0 29.0 18.0 5.0 1.5 28.1 23.3 9.5 3.0 7.0 6.0 2.0 8.0 27.5 65.5 0.0 44.0 36.5 31.0

LLaVA-interleave [43] 32.4 29.5 24.8 26.3 23.2 26.4 25.1 48.8 49.8 23.5 25.0 28.0 57.0 21.5 33.0 63.5 54.5 25.0 26.0 24.0 27.0 49.5 29.0 23.0 25.4 27.5 32.5

43.0 34.0 49.0 29.5 32.0 26.0 30.0 21.5 42.0 47.5 22.5 14.0 23.6 32.3 17.5 28.5 23.0 17.5 3.0 31.0 36.0 79.0 15.0 60.5 34.5 42.5 InternVL1.5-chat [13] 37.4 63.7 31.0 22.6 20.3 16.3 28.3 63.2 38.5 21.0 28.0 26.5 82.5 20.5 31.5 6.0 45.5 26.5 29.5 29.5 85.0 65.0 32.0 23.5 29.0 18.5 89.0

90.5 35.5 56.5 23.5 31.0 24.5 53.0 26.0 40.0 49.0 25.5 15.5 59.3 43.6 19.5 22.5 23.5 15.0 33.5 28.0 39.0 71.0 9.5 46.5 50.5 39.5 VideoChat2 [48] 35.0 46.8 27.5 31.6 23.6 25.6 28.8 45.3 54.3 20.5 25.5 25.5 64.0 21.0 31.0 31.5 50.0 21.0 31.0 30.5 73.0 51.0 31.5 23.5 21.8 24.0 81.5

54.0 42.0 59.0 23.0 30.5 23.0 44.5 26.5 44.0 36.5 25.0 18.0 38.6 44.4 21.0 26.5 24.0 13.0 0.0 28.5 43.0 65.5 11.5 58.0 36.0 35.0 VideoChat-TPO 40.2 73.3 24.3 37.0 24.6 26.5 26.9 45.0 69.5 20.5 23.5 29.5 83.0 21.0 31.0 92.5 49.5 29.5 30.0 24.5 88.0 67.5 34.5 29.5 36.8 24.5 94.5

59.0 39.5 56.5 27.5 29.5 21.0 59.0 25.0 44.0 48.5 27.5 14.5 73.4 44.4 23.5 27.5 24.5 7.5 0.0 24.0 38.5 67.0 11.5 58.5 47.0 40.5

Table 14. Quantitative results of MMIU [59]. Accuracy is the metric, and the Overall score is computed across all tasks.

Charades-STA [23] MVBench R@0.3 R@0.5 R@0.7 mIoU AVG

Model

VideoChat-TPO 58.3 40.2 18.4 38.1 66.8 Only QVHighlight 54.8 34.6 15.1 35.8 66.5

Table 15. Ablation task datasets.

performance improvements in both multimodal and specific vision tasks. Various datasets are necessary for effectively enhancing TPO’s dialogue capabilities and achieving zeroshot generalization to fine-grained visual tasks.

Model MVBench VideoMME PerceptionTest LLaVa-OV 56.7 58.2 57.1 LLaVa-OV-TPO 64.8(+8.1) 61.3(+3.1) 64.0(+6.9)

Table 16. Perfermance of LLaVA-OV on Video Benchmarks.

LLaVA-OV-TPO Performance on Video Benchmarks. According to Table 16, TPO method demonstrates performance improvements on LLaVA-OV [42] across multiple video benchmarks as it does in VideoChat [48] model. Since TPO uses extra visual cues to guide MLLM, LLaVAOV-TPO achieves an average score of 64.8(+8.1%) on MVBench [48] and 64.0(+6.9%) on PerceptionTest [63]. The notable improvement clearly demonstrates the model’s

greatly enhanced ability to perceive visual details. Moreover, LLaVA-OV-TPO achieves a 3.1% performance improvement on VideoMME [21] and shows that the model has also made progress in knowledge modeling and understanding long videos. These results suggest TPO method is effective across various models and is particularly beneficial for fine-grained perception tasks.

#### 2. Training and Data Details

Table 17 and 18 lists the detailed training configurations and data of VideoChat-TPO in different stages. In each stage, the model is parametrized from the weights from the previous stage and continues training.

Settings of Stage 1. The LLM is equipped with LoRA [27] for saving computational memory, using a LoRA rank of 16 and an alpha of 32. Only the LoRA is trained for efficiency. We adopt the AdamW optimizer [54] with the peak learning rate of 2e-5 and use cosine weight decay. The training involves a total batch size of 128 across 32 A100 GPUs. Since the purpose of stage 1 is to make MLLM identify tasks, we only use a small amount of data in this stage and adopt LLM loss so that LLM can generate task-specific tokens. For each task, we train the LLM with 50k examples to recognize the task. For training data, we

|Config<br><br>|Stage 1 Stage 2 Stage 3 w/o Con. Stage 3|
|---|---|
|Vision Enc. LR Connector LR Temporal Head LR Region Head LR Mask Head LR Mask Adapter LR Temporal Token LR Region Token LR Mask Token LR LLM LoRA LR LR Schedule Optimizer Weight Decay Input Resolution Input Frames LLM LoRA Rank LLM LoRA Alpha Warmup Ratio Total Batch Size Epoch Numerical Precision<br><br>|Frozen Frozen 2e-5 2e-5 Frozen Frozen 2e-5 2e-5<br><br>- 1e-4 2e-5 2e-5<br>- 1e-4 2e-5 2e-5<br>- Frozen Frozen Frozen<br>- 1e-4 2e-5 2e-5<br>- 2e-4 2e-5 2e-5<br>- 1e-4 2e-5 2e-5<br>- 1e-4 2e-5 2e-5<br><br><br>2e-5 2e-5 2e-5 2e-5 Cosine Decay Cosine Decay Cosine Decay Cosine Decay AdamW [54] AdamW [54] AdamW [54] AdamW [54] 0.02 0.02 0.02 0.02 2242 2242 2242 2242 16 16 16 16 16 16 16 16 32 32 32 32 0.2 0.2 0.2 0.2 128 64/128/128 256 256 1 25/3/1 1 3 DeepSpeed bf16 [67] DeepSpeed bf16 [67] DeepSpeed bf16 [67] DeepSpeed bf16 [67]|

Table 17. Training Settings of VideoChat-TPO. Con. means conversation data and LR means learning rate.

Stage Task Samples Datasets

- Stage 1

Temporal Grounding 50K DiDeMo [26], QuerYD [62]

Spatial Grounding 50K RefCOCO [98], RefCOCOg [98], RefCOCO+ [58]

Segmentation 50K SAMv2 [68], MeViS [17]

- Stage 2

Temporal Grounding 116.5K DiDeMo [26], QuerYD [62], HiRest [102], ActivityNet [6], TACoS [69], NLQ [24]

Spatial Grounding 540.0K AS-V2 [83], Visual Genome [36], RefCOCO [98], RefCOCO+ [98], RefCOCOg [58]

Segmentation 114.6K SAMv2 [68], MeViS [17]

- Stage 3

Temporal Grounding 7.5K QVHighlight [38]

Spatial Grounding 400K AS-V2 [83], Visual Genome [36], RefCOCO [98], RefCOCO+ [98], RefCOCOg [58]

Segmentation 116.5K MeViS [17], SAMv2 [68] Temporal Reasoning 40K YouCook2 [16], ActivityNet [6]

Conversation 3M VideoChat2-IT [48], ShareGPT-4o [13], LLaVA-Hound-DPO [108], ShareGPT4V [9]

- Table 18. Datasets Used at Three Training Stages. The temporal grounding task includes two subtasks: moment retrieval and highlight detection.

use DiDeMo [26] and QuerYD [62] for temporal grounding task, RefCOCO [98], RefCOCOg [98] and RefCOCO+ [98] for spatial grounding task, and SAMv2 [68], MeViS [17] for segmentation task.

- Settings of Stage 2. In stage 2, we add the task heads (i.e. temporal head, region head, and mask head) and learnable task tokens (temporal token, region token, and mask token). The objective of the second training stage is to learn the task head with preliminary functional capabilities. Therefore, we train LLM, task head and task token at this stage, and freeze vision encoder and connector.

In stage 2, the region head and token are trained with a learning rate of 2e-5 using a cosine learning rate scheduler. We use a two-layer MLP as region head to train from scratch and we use MSE loss for region head training. For training data, we use AS-V2 [83], Visual Genome [36], RefCOCO [98], RefCOCOg [98], RefCOCO+ [98] for one

epoch with a total batch size of 128 to train region head and token.

We use a learning rate of 1e-4 for the temporal head and 2e-4 for the temporal token in stage 2. The temporal head is the same as CG-DETR [60] in structure, but we use the pre-trained InternVideo2 [85] to extract video features, while query features are extracted using the ChineseLlama-Alpaca [14]. We use the same loss function in CGDETR. We train the model on DiDeMo [26], QuerYD [62], HiRest [102], ActivityNet [6], TACoS [69], NLQ [24] for 25 epochs with a total batch size of 64.

For the mask head, we use the pre-trained SAM2 [68] model, replacing the prompt encoder of SAM2 with a single MLP layer called the mask adapter. During training, the mask token and adapter are trained with a learning rate of 2e-5, and the rest of SAM2 is frozen. We use MeViS [17] , SAMv2 [68] for three epochs in this stage with a total batch size of 128. We supplement the training data by expanding

the ASv2 [83] image dataset into videos and adding it to this stage.

- Settings of Stage 3. The third training stage aims to strengthen the model’s conversational ability using TPO. This stage is divided into two parts. The first part involves training on a combined dataset of all tasks. The second part uses a dataset combining both task and conversation data. For conversatation data, we use VideoChat2IT [48], ShareGPT-4o [13], LLaVA-Hound-DPO [108], ShareGPT4V [10] for instruction finetuning. We adopt a peak learning rate of 2e-5 for all the model in this stage and use a total batch size of 128.

Model GPU Stage1 Stage2 Stage3

VideoChat-TPO 64 0.5h 11h 52h textualized task data 64 0.5h 10h 50h only conversation data 64 - - 42h

- Table 19. Training Cost of Three Stages on VideoChat. Textualized task data means converting task data into conversation form.

TPO Additional Training and Inference Cost. From the data perspective, as can be seen from Table 18 in the Appendix, we have very little training data in the first (around 0.15M) and second stages (around 0.7M), most of the data (around 3.5M) is used in third phase of the experiment. Among the data in the third stage, most of it is conversation data for fine-tuning MLLM. Therefore, TPO introduces little new data. Concerning training cost, according Table 19, when using the same amount of data, the training time of our TPO method and the autoregressive method is almost the same, and compared with the version without visual task, the TPO method increases the training cost by about 25%.

The Temporal Head and the Mask Head contains additional encoders. In training phase, the additional encoders are frozen, and we use the features extracted by the encoder for training. In inference phase, the additional encoders are only used when the task head is activated. When only performing conversation tasks, no additional inference cost is incurred.

Template Details. To support the proper invocation of task-specific decoders, we construct a series of instruction templates for different tasks and use them as instruction tuning data for MLLM. We comprehensively list all the instruction templates below, in Table 20, 21, 22, and 23.

- 1. Localize the visual content described by the given textual query ⟨query⟩ in the video, and output the start and end timestamps in seconds.
- 2. Detect and report the start and end timestamps of the video segment that semantically matches the given textual query ⟨query⟩.
- 3. Locate and describe the visual content mentioned in the text query ⟨query⟩ within the video, including timestamps.
- 4. The given natural language query ⟨query⟩ is semantically aligned with a video moment, please give the start time and end time of the video moment.
- 5. Find the video segment that corresponds to the given textual query ⟨query⟩ and determine its start and end seconds.

###### Table 20. Instructions for Temporal Grounding.

- 1. Track the object in the video using a box with initial coordinates ⟨track box⟩.

- 2. Use a bounding box with coordinates ⟨track box⟩ to follow the movement of the moving object in the visual input.

- 3. Given an initial bounding box with coordinates ⟨track box⟩, track the motion of the target object in the sequence of frames.

- 4. Starting from the box defined by the coordinates ⟨track box⟩, monitor the movement of the object in the video.

- 5. Utilizing the initial box specified by the coordinates ⟨track box⟩, continuously track and update the location of the object in the video stream.

- 6. Given a video with an object of interest enclosed in a bounding box with coordinates ⟨track box⟩, generate a sequence of bounding boxes that track the object’s movement.

- 7. With an initial box defined by ⟨track box⟩, trace the object’s trajectory by generating a sequence of bounding boxes that follow the object’s movement in the visual input.

- 8. Apply an object tracking algorithm to a video, starting with a bounding box defined by ⟨track box⟩.

- 9. Given a video and an initial bounding box defined by ⟨track box⟩, track the movement of the object within the video.

- 10. Starting from an initial box defined by ⟨track box⟩, track the movement of the object in the visual input.

###### Table 21. Instructions for Tracking.

- 1. Where is ⟨expr⟩?
- 2. Can you find ⟨expr⟩?
- 3. Can you detect ⟨expr⟩?
- 4. Can you locate ⟨expr⟩?
- 5. Please find ⟨expr⟩.
- 6. Please detect ⟨expr⟩?
- 7. Please locate ⟨expr⟩?
- 8. Find ⟨expr⟩.
- 9. Detect ⟨expr⟩?
- 10. Locate ⟨expr⟩?

- 1. Please give the motion path of ⟨obj⟩ in the video over time.
- 2. Show the tracking trajectory of ⟨obj⟩’s movement through the scene in the video.
- 3. Please generate a motion path of ⟨obj⟩’s movement in the video, highlighting its tracking trajectory.
- 4. Show the tracking trajectory of ⟨obj⟩.
- 5. Generate ⟨obj⟩’s tracking trajectory.
- 6. Visualize the tracking trajectory of ⟨obj⟩ in the video.
- 7. Please generate a visual representation of ⟨obj⟩’s movement in the video, highlighting its tracking trajectory.

Table 22. Instructions for Spatial Grounding.

Table 23. Instructions for Referring Segmentation.

#### 3. Qualitative Results

We evaluate VideoChat-TPO on various visual perception tasks and display the visualizations from Figure 4 to Figure Figure 7. In addition, we also show the results of multimodal video understanding in Figure 8.

Spatial Grounding. In Figure 4, we show the spatial grounding visualizations. VideoChat-TPO can infer the target object from the description of natural language and locate it. Our VideoChat-TPO can accurately locate the target among multiple similar objects. Even if the target object is occluded or in the background area, it can still be accurately located.

Referring Segmentation. We show the visualizations of the referring segmentation in Figure 5. VideoChat-TPO can delinear the target object in the video according to user input in complex scenes. Furthermore, VideoChat-TPO can separate the target object from multiple objects of the same kind according to the description of appearance or action characteristics indicated by the user.

Tracking. The tracking visualizations are shown in Figure 6. The user needs to include the bounding box coordinate information of the first frame of the tracked target in the video in the input. The visualizations show that when the target object is partially occluded in the video, it can still be tracked. Even if the target object is out of the camera’s view, our VideoChat-TPO can still track it when it appears in subsequent frames.

Moment Retrieval and Highlight Detection. The visualizations of the moment retrieval and highlight detection are given in Figure 7. Our VideoChat-TPO can infer the results and target events based on the user’s questions, and perform moment retrieval and highlight detection on the target events.

Multimodal Video Understanding. The multimodal video understanding visualizations are shown in Figure 8. Our VideoChat-TPO achieve decent results in fine-grained action description, spatial description, and video captioning.

Can you find this is a woman holding a thing of bananas?

Please find the orange closest to the banana.

Locate white colored bus in the image.

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

TPO

TPO

TPO

Find brown bread vegetable sandwich with tooth pick next

Can you detect the fork between the bread plate and dinner plate?

Please find a plant to the right of the blue sign in image.

to white bowl.

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

TPO TPO

TPO

Detect a picture of a traffic light with a red arrow pointing left lit up in the image?

Can you find a mother giraffe in the image.

Can you locate powdered donut.

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

TPO

[Figure 71]

TPO

TPO

###### Figure 4. Qualitative Results of Spatial Grounding.

Please give the motion path of “Dog playing with monkey” in the video over time.

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

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

TPO

Please give the motion path of “Sheep with the black head facing down to eat then walking forward” in the video over time.

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

###### TPO

Please give the motion path of “people playing with lizard” in the video over time.

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

###### TPO

Please give the motion path of “The darker-colored one among the two dogs playing together” in the video over time.

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

###### TPO

Please give the motion path of “Person holding a rope” in the video over time.

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

TPO

###### Figure 5. Qualitative Results of Referring Segmentation.

Track the object in the video using a box with initial coordinates [13.0, 403.0, 173.0, 586.0].

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

###### TPO

Track the object in the video using a box with initial coordinates [456.0, 346.0, 608.0, 631.0].

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

###### TPO

Track the object in the video using a box with initial coordinates [646.0, 338.0, 783.0, 496.0].

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

###### TPO

Track the object in the video using a box with initial coordinates [207.0, 120.0, 228.0, 166.0].

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

###### TPO

Track the object in the video using a box with initial coordinates [662.0, 166.0, 953.0, 719.0].

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

TPO

###### Figure 6. Qualitative Results of Tracking.

[Figure 208]

[Figure 209]

Locate and describe the visual content mentioned in the text query “what does the person in white do after walking past?” within the video, including timestamps.

The given query happens in 23.0-27.0 seconds. After walking past, the person in white walks back to the child.

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

TPO

|[Figure 227]|
|---|

[Figure 228]

[Figure 229]

Locate and describe the visual content mentioned in the text query “why did the lady smile at the start? ” within the video, including timestamps.

The given query happens in 0.0-4.0 seconds. The lady smiled at the start because she was happy to see her dog's enthusiasm for the treats she was giving it.

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

TPO

|[Figure 247]|
|---|

[Figure 248]

[Figure 249]

Locate and describe the visual content mentioned in the text query “why did the shorter girl bend down after opening the

orange cabinet? ” within the video, including timestamps.

The given query happens in 22.0-29.0 seconds. The shorter girl bends down after opening the orange cabinet because she may be searching for something inside the cabinet or reaching for an item on the floor.

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

TPO

|[Figure 267]|
|---|

[Figure 268]

[Figure 269]

Locate and describe the visual content mentioned in the text query “what is the girl doing as she put her hands on the fridge at the beginning of the video ” within the video, including timestamps.

The given query happens in 0.0-6.0 seconds. At the beginning of the video, the little girl is putting her hands on the fridge. She seems to be curious and is touching the fridge.

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

TPO

|[Figure 287]|
|---|

Figure 7. Qualitative Results of Moment Retrieval and Highlight Detection. The orange curve represents saliency score, the blue interval represents the time interval predicted by the model, and the green interval represents the ground truth.

Describe the cat's actions in the video.

(fine-grain action)

The video shows a cat walking down a street in the rain. The cat is seen walking on the wet ground and passing by various objects such as a blue trash can, a brown cardboard box, and a pile of garbage. The cat is also seen walking past a pile of garbage bags and a pile of garbage. The video captures the cat's movements and the surrounding

[Figure 288]

environment.

TPO

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

Describe the video in detail.

(caption)

The video shows a bird's eye view of a city with a large body of water in the background. The camera pans over the city, showing various buildings and structures, including a large building with a dome, a tall building with a clock tower, and a large building with a dome. The camera also captures a view of the ocean and a rocky cliff. The video provides a detailed view of the city's architecture and the surrounding natural beauty.

[Figure 303]

TPO

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

(spatial perception)

Which cup does the lid go under, the left one, the middle one, or the right one?

The lid goes under the middle cup.

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

TPO

(caption)

Describe the video in detail.

The video shows a person riding a bike down a dirt path in the woods. The camera is mounted on the bike, and we can see the rider's legs and feet as they pedal. The path is surrounded by trees, and we can see the shadows of the trees on the ground. The rider is wearing a helmet, and we can see the reflection of the trees in the helmet. The video is shot during the day, and we can see the sunlight shining through the trees. The rider is wearing a colorful shirt, and we can see the colors of the shirt as they ride. Overall, the video shows a peaceful and scenic bike ride through the woods.

[Figure 333]

TPO

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

Figure 8. Qualitative Results of Multimodal Video Understanding.

