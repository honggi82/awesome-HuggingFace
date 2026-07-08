## StreamChat: Chatting with Streaming Video

Jihao Liu1,2,∗ Zhiding Yu2 Shiyi Lan2 Shihao Wang2,5,∗ Rongyao Fang1 Jan Kautz2 Hongsheng Li1,3,4 Jose M. Alvarez2

1CUHK MMLab 2NVIDIA 3Shanghai AI Laboratory 4CPII under InnoHK 5The Hong Kong Polytechnic University

# arXiv:2412.08646v2[cs.CV]30Mar2025

(11s) Question: What you are seeing now is my POV. Please describe my surrounding environment.

Video Stream

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

(11s) Text stream: You are standing in a dim room with a carpet on the floor and a door to your left that is slightly ajar… some furniture and items scattered…

Video Stream

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

You can see a television turned on and a guitar leaning against the wall. The lighting in the room is dim...atmosphere feels quiet and somewhat cluttered. (22s)

Figure 1. Example of StreamChat on streaming video. In the example, the question is asked at the 11th second. As the model outputs its text steam, it continuously follows the dynamic content of the streaming video and uses up-to-date video content to answer the question.

### Abstract

streaming interaction scenarios compared to state-of-the-art video LMMs. Our project page is at StreamChat.

This paper presents StreamChat, a novel approach that enhances the interaction capabilities of Large Multimodal Models (LMMs) with streaming video content. In streaming interaction scenarios, existing methods rely solely on visual information available at the moment a question is posed, resulting in significant delays as the model remains unaware of subsequent changes in the streaming video. StreamChat addresses this limitation by innovatively updating the visual context at each decoding step, ensuring that the model utilizes up-to-date video content throughout the decoding process. Additionally, we introduce a flexible and efficient crossattention-based architecture to process dynamic streaming inputs while maintaining inference efficiency for streaming interactions. Furthermore, we construct a new dense instruction dataset to facilitate the training of streaming interaction models, complemented by a parallel 3D-RoPE mechanism that encodes the relative temporal information of visual and text tokens. Experimental results demonstrate that StreamChat achieves competitive performance on established image and video benchmarks and exhibits superior capabilities in

### 1. Introduction

The recent surge of large language models (LLMs) [4, 12, 21, 39, 44, 55] and large multimodal models (LMMs) [10, 24, 53, 57] has unlocked numerous application scenarios, including visual instruction following [31–34] and long video understanding [69, 80]. Notably, frontier models such as GPT-4o [41] and Gemini [51] have shown remarkable proficiency when interacting with streaming videos, attracting considerable interest in the field. While recent open approaches [7, 63, 67, 77] have emerged to enhance streaming video processing, they still fall short in interaction fluency and perceptual capabilities.

To enable effective interaction with streaming videos, LMMs must not only accurately identify the visual content of each frame but also track dynamic changes in the streaming video, leveraging the latest visual information to answer questions, as illustrated in Figure 1. Despite notable progress in video understanding of LMMs [11, 52, 68, 76, 81], existing models often overlook the crucial need to capture

∗ Work done during an internship at NVIDIA

Corresponding authors

question, which is different from the streaming interaction scenarios where the video content is dynamically changing during the answering process. To bridge this gap, we create a new dense instruction dataset based on existing dense caption datasets. One dense instruction data consists of several (time interval, instruction, answer) triplets, with each word of the instruction-answer pairs annotated with a timestamp in a heuristic manner. During training, we employ attention masks to ensure that each text token can only attend to video information before its corresponding timestamp. This method effectively simulates the conditions of streaming interaction throughout the training process.

|0|
|---|

|1|
|---|

|2|
|---|

|…|
|---|

###### Text stream

(existing models)

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

| |[Figure 30]|
|---|---|
|t| |
| |[Figure 31]|

|0|
|---|

|…|
|---|

|t-1|
|---|

|t+1| |
|---|---|
| |[Figure 32]|

|t+2| |
|---|---|
|[Figure 33]| |

|…|
|---|

Video stream

[Figure 34]

[Figure 35]

[Figure 36]

Question

|0|
|---|

|1|
|---|

|2|
|---|

|…|
|---|

Text stream

(StreamChat)

- Figure 2. Comparison of context in the decoding process with existing models. For each text token, the black and blue arrows indicate the beginning and end of the utilized visual context, respectively. While existing models (top) use a fixed visual context when decoding, StreamChat (bottom) aligns the video and text streams temporally and dynamically updates its visual context based on the streaming video.

Importantly, we do not directly input the absolute timestamp of each token into the model, as these timestamps are unavailable during inference. Instead, we propose a parallel 3D-RoPE mechanism that allows each token to be aware of its relative temporal position within the video. We use three components in RoPE [49] to represent temporal, height, and width, respectively. Unlike existing approaches that arrange video and text in an interleaved manner [49, 57], our method organizes them in a parallel way to ensure that visual and text tokens at the same timestamp share the same temporal context in RoPE, which enhances the continuity during streaming interactions.

dynamic changes, negatively impacting the interaction experience. Specifically, current methods typically rely on video information only up to the moment a question is asked; however, the streaming content may change significantly during the decoding process, leaving the model unaware of these updates. For instance, assume a question is posed at time t and the model takes t′ seconds to answer the question, existing methods only utilize the video content from the interval 0 to t to answer the question, leaving the model unaware of any changes that occur between t and t + t′. This delay can be particularly detrimental in highly dynamic video environments or when the answer to a question is lengthy, resulting in a suboptimal user experience. We illustrate the problem in Figure 2 (top).

Through extensive experiments, we demonstrate that StreamChat not only achieves competitive performance on established image and video benchmarks but also significantly improves capabilities in streaming interaction scenarios. Specifically, we create a benchmark designed to evaluate LMMs in streaming interaction scenarios. We demonstrate that our StreamChat-7B outperforms the state-of-theart LLaVA-Video-72B model.

To address these limitations, we propose StreamChat, a novel approach that enables LLMs to interact dynamically with streaming video content. The core idea is to provide the LLM with the latest video information at each decoding step, allowing it to better capture video dynamics and adjust its responses accordingly, which is illustrated in Figure 2 (bottom). Mechanistically, StreamChat enhances the model’s ability to interact with streaming video data, ensuring more temporally aligned responses, as demonstrated in Figure 1. To effectively handle the dynamic visual inputs of streaming videos, we design a flexible and efficient architecture based on cross-attention mechanism [3, 12, 56], bridging the LLM and visual inputs in StreamChat. The cross-attention design facilitates processing variable-length inputs in the streaming scenario and is more efficient when dealing with a large number of visual tokens.

### 2. Methods

Recent advancements in large multimodal models (LMMs) [11, 52, 68, 76, 81] have significantly improved the models’ video understanding capabilities. However, in streaming interaction scenarios, LMMs must also accurately capture the dynamic changes of the streaming video content, which is overlooked by existing models. To bridge this gap, we propose StreamChat, a novel LMM that can interact smoothly with streaming videos and track the latest changes in the videos to refine its answers. In this section, we outline the methodology underlying StreamChat, detailing the architectural innovations and techniques employed to enable dynamic interaction with streaming video. We begin by describing the StreamChat architecture design in Section 2.1 and then introduce how we generate and construct our training data in Section 2.2. Finally, we discuss the development of our training and inference pipeline in Section 2.3.

To facilitate the training of streaming interaction models, we introduce a dense instruction dataset to train StreamChat. Existing video instruction-tuning datasets [17, 22, 26, 50, 81] primarily focus on offline video understanding, i.e., the model can perceive the complete video before answering a

The video describe

Visual tokens

Text tokens

(0, 0, 0) (1, 1, 1) (t, t, t)

###### Time

[Figure 37]

|(0,0,0)|(0,0,1)|(0,0,2)|
|---|---|---|
|(0,1,0)|(0,1,1)|(0,1,2)|
|(0,2,0)|(0,2,1)|(0,2,2)|

|(1,1,1)|(1,1,2)|(1,1,3)|
|---|---|---|
|(1,2,1)|(1,2,2)|(1,2,3)|
|(1,3,1)|(1,3,2)|(1,3,3)|

|(t, t, t)|…|…|
|---|---|---|
|…|…|…|
|[Figure 38]<br><br>…|…|(t,t+h, t+w)|

…

Q

K & V

###### Cross-attention

Figure 4. The parallel 3D-RoPE. For visual and text tokens at the same timestamp, they share the same temporal position.

Linear gate

Self-attention

to improve the convergence speed during training.

We further utilize V-FFN experts to enhance the visual representations throughout the LLM’s forward process. Specifically, after each cross-attention block, we update the visual tokens with a V-FFN expert and feed the updated tokens into the subsequent cross-attention block. In contrast to previous cross-attention-based models [3, 8, 12, 72] that utilize the same visual representations for all cross-attention blocks, our V-FFN experts allow the visual representations to better align with the LLM’s hidden status and improve the final performance. Practically, these V-FFN experts are initialized from the LLM’s FFN instead of training from scratch to inherit the pretrained knowledge of the LLM.

V-FFN FFN

Linear gate

xN

- Figure 3. The StreamChat architecture. We utilize crossattention blocks to bridge the visual and text tokens and V-FFN blocks to update the visual tokens throughout the LLM’s forward process. Those two blocks’ outputs are scaled with a linear gate mechanism.

Previous cross-attention-based models [3, 12] typically employ a tanh-gating mechanism to ensure that the language model produces the same results as the original LLM at the early stage and stabilize the training. However, the tanh function suffers from the gradient vanishing problem, which results in suboptimal performance. Instead, we introduce a linear gate to scale the output of cross-attention and V-FFN blocks to a relatively small range during the initial training phase following CaiT [54]. The linear gate mechanism mitigates the gradient problem while also stabilizing the training process.

#### 2.1. StreamChat Architecture

To support streaming video content, we design a flexible and efficient architecture capable of handling dynamic video inputs through the cross-attention mechanism. Additionally, we introduce visual feedforward network (V-FFN) experts to enhance the visual representations throughout the large language model’s (LLM) forward process. We also propose a parallel 3D-RoPE mechanism to better encode the temporal information in streaming interaction scenarios. The architecture is illustrated in Figure 3.

Parallel 3D-RoPE. To better model the positional information of streaming video and text, we propose parallel 3D-RoPE that extends traditional 1D-RoPE [49] to 3D space with a parallel arrangement of visual and text tokens. Specifically, we split the embedding of RoPE into three components. For text tokens, these components are identical to represent the temporal location of each token. For visual tokens, these three components represent the temporal, height, and width locations of each token. Unlike previous approaches that arrange the visual and text tokens in an interleaved manner [57], we use a parallel way to arrange them, as illustrated in Figure 4. Given a text token and a visual token at the same timestamp, we apply the same temporal index for them. Our intuition is that in the streaming setting, one specific timestamp’s text and visual tokens are happening simultaneously, and therefore should share the same temporal location instead of an interleaved location. The parallel arrangement

Cross attention. We build a cross-attention-based architecture to bridge the visual and text tokens. Given an input streaming video, we utilize a pretrained vision model to extract visual tokens for each sampled frame separately. To integrate these visual tokens with the LLM, we insert several cross-attention blocks into the LLM architecture, where text tokens serve as queries and visual tokens act as keys and values. The visual tokens are dynamically updated during the interaction process, and the cross-attention design facilitates processing these dynamic inputs. Moreover, compared to self-attention-based architectures (e.g., LLaVA [33]), crossattention is significantly more efficient when the visual tokens are much more than the text tokens, especially in streaming interactions, where we have high frames per second (FPS) for inference. In practice, our cross-attention blocks share parameters with the self-attention blocks of the LLM

is crucial for the high FPS inference in the streaming setting, where the traditional arrangement may have a significant temporal positional gap between two adjacent text tokens while our approach ensures their continuity.

#### 2.2. Dense Instruction Data

Existing video instruction-tuning datasets [17, 22, 26, 50, 81] have made significant progress in offline video understanding, i.e., the model can see the whole video before answering a question. However, these datasets are not suitable for training streaming interaction models, where the input is a streaming video and each text token can only see part of the video. For instance, a text token at timestamp t can only perceive video frames at and before timestamp t. To tackle this problem, we create a new video instruction-tuning dataset from the existing dense caption datasets, whose captions are paired with timestamp intervals.

Given a video with its dense caption, we prompt an LLM (e.g., Gemini-1.5-Pro [51]) to pick a start time of one video segment and then generate an instruction-answer pair based on the caption of this segment. We instruct the LLM to focus on streaming interaction scenarios and generate relevant instructions. To enhance the diversity of the instruction data, we initially generate 5k pairs and conduct a clustering to eliminate highly similar instructions. We manually review the remaining examples and use them as in-context examples for subsequent data generation. Ultimately, we collect a total of 51k examples from two dense caption datasets, Ego4D [17] and Vript [70], with one representing the egocentric environment and the other one representing the natural environment.

#### 2.3. Training and Inference

Data arrangement. Given the initial instruction data with coarse temporal annotations, i.e., in the form of (time interval, instruction, answer), we employ a heuristic approach to assign timestamps to each word in the instruction data. For instance, consider a triplet where the time interval is 5-10 seconds, the instruction is “What is the person in the video doing now?” and the answer is “The person is cooking right now.” To generate fine-grained temporal annotations, we transform this coarse-grained triplet into a sequence of words including time indicators. The transformation results in the following format:

Instruction:<5>What is the person in the video doing now? Answer:<5>The <6>person <7>is <8>cooking <9>right <10>now.

Here, <t> represents the t-th second. The intuition behind this design is that the instruction is input by the user instantaneously, while the answer is decoded token by token by the model. For this example, we assume the model decodes one token per second. Note that the <t> indicators are not directly input to the model but serve solely for reference.

Attention mask. To ensure that the token at time <t> does not attend to video frames occurring after <t>, we utilize attention mask to block such attention. This mechanism is crucial for maintaining the temporal integrity of the streaming interaction, allowing the model to focus only on relevant visual information available at each decoding step.

Inference. During inference, StreamChat employs a parallel approach to ensure efficient processing of the streaming video content. Specifically, we utilize a separate thread to continuously read the video stream and store the extracted visual tokens in a First-In-First-Out (FIFO) queue. When the LLM requires decoding to generate a response, it acquires the latest video tokens from the FIFO queue. The model then incorporates this current information to decode the next token, ensuring that its responses are informed by the most up-to-date video stream context. This design not only enhances the relevance of the model’s outputs but also supports seamless streaming interactions, allowing users to engage with dynamic video content effectively.

### 3. Experiment Setups

In this section, we outline the experimental setups employed in our study. We build our model using the SigLIP vision encoder [75] with PaliGemma weights [6] and 7B/14B Qwen

- 2.5 LLM [52]. We utilize a Multi-Layer Perceptron (MLP) adapter [33] to align the hidden dimensions of the vision and language components.
- 3.1. Pretraining

We implement a two-stage pretraining process and gradually unfreeze the pretrained parameters for more effective pretraining. In both stages, we utilize a combination of ReCap data from LLaVA-Next [32], part of InternVL pretraining data [10], MMC4 [84], and dense caption datasets [17, 22, 50, 70, 83]. In stage 1, we only train the MLP adapter for alignment. We train the MLP for 5000 steps with a maximum learning rate of 5 × 10−4 and batch size of 512. In stage 2, we further unfreeze the vision encoder and the visual feedforward network (V-FFN) experts to achieve deeper alignment. We train for 5000 steps with a maximum learning rate of 2 × 10−5 and batch size of 512. For the dense caption datasets, we employ 1 frame per second (FPS) and a maximum of 40 frames for training. For other video data, we uniformly sample 40 frames for training. In total, we utilize 5.1 million samples for pretraining.

#### 3.2. Instruction Tuning

We construct a comprehensive instruction-tuning dataset mainly based on Eagle-1.8M [45]. In addition, we incorporate our dense instruction dataset and LLaVA-Video [81] for instruction tuning. We unfreeze all the parameters and train

StreamChat Wins Tie StreamChat Loses

StreamChat-7B

StreamChat-14B

| |37% 32% 31%<br><br>44% 22% 34%<br><br>53% 24% 23%<br><br>58% 27% 15%<br><br>55% 30% 15%| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| |41% 24% 35%<br><br>45% 21% 34%<br><br>56% 26% 18%<br><br>62% 19% 19%<br><br>52% 26% 22%| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

VILA-1.5-8B

VILA-1.5-8B

VILA-1.5-13B

VILA-1.5-13B

VILA-1.5-40B

VILA-1.5-40B

LLaVA-Video-7B

LLaVA-Video-7B

LLaVA-Video-72B

LLaVA-Video-72B

0% 25% 50% 75% 100%

0% 25% 50% 75% 100%

- Figure 5. Comparison of StreamChat with leading video LMMs on streaming evaluation. We use StreamChat-7B/-14B as one of the candidate models and report the win/tie/loss rate against VILA or LLaVA-Video models. Our StreamChat models demonstrate stronger streaming interaction capabilities, and can even outperform LLaVA-Video-72B which uses a much larger base LLM.

|[Figure 39]|
|---|

|[Figure 40]|
|---|

|[Figure 41]|
|---|

|[Figure 42]|
|---|

|[Figure 43]|
|---|

|[Figure 44]|
|---|

Video Stream

Ours: (1s) … transitions from a dark … to a bright … with a yellow background and a rotating gear … ends with a bright yellow background and the text "ГОРОД ИНСТРУМЕНТА”… VILA: (1s) … starts with a black screen, then transitions to a blurry image of a person's face. The person appears to be wearing a hat and has a beard. The image is not clear … LLaVA-V: (1s) …transitions from a completely black screen to a small, bright light in the center. The light gradually grows larger and brighter … becomes a large, glowing orb.

(1s) Question: Describes the scene transition of the video.

|[Figure 45]|
|---|

|[Figure 46]|
|---|

|[Figure 47]|
|---|

|[Figure 48]|
|---|

|[Figure 49]|
|---|

|[Figure 50]|
|---|

(21s) Question: Describe the environments and creatures featured in the current and subsequent video segment. Video Stream

Ours: (21s) The current…serene ocean with a whale breaking the surface … a flock of birds flying … mountains in the background. The subsequent…vibrant underwater scene… VILA: (21s) …a large whale swimming in the ocean. The whale is seen from different angles, … massive size and the intricate patterns on its body … ocean water is a deep blue… LLaVA-V: (21s) … features a serene ocean environment with a large whale swimming gracefully. The whale's tail is prominently visible, creating a dynamic and captivating scene.

- Figure 6. Qualitative evaluation of StreamChat on streaming video. In the example shown, the questions are asked at the first second (top) and the 21st second (bottom), respectively. Our model can capture the dynamic video content and adapt its answer accordingly. In comparison, VILA and LLaVA-Video fail to follow the streaming video and exhibit factual errors (highlighted in red).

on the dataset combination for 1 epoch. We use a maximum learning rate of 2 × 10−5 and a batch size of 768. For the dense instruction data, we use 1 FPS and a maximum of 32 frames for training. For other video instruction data, we uniformly sample 32 frames for training. In total, we use 2.9 million samples for instruction tuning.

### 4. Streaming Evaluation

To evaluate large multimodal models’ (LMMs) streaming interaction capabilities, we construct a streaming evaluation benchmark from existing dense caption datasets. Based on the dense caption of a video, we prompt Gemini-1.5-Pro to generate an instruction-answer pair for a specific timestamp. We remove samples that are not related to streaming

|Method<br><br>#VisTok.|PMME<br><br>MMB<br><br>VMMMU<br><br>VMMStar<br><br>ISEED<br><br>GQA<br><br>ISQA<br><br>AI2D<br><br>TextVQA<br><br>RealworldQA|
|---|---|
|Private models<br><br>GPT-4V UNK. Gemini-1.0 Pro UNK. Gemini-1.5 Pro UNK.<br><br>Grok-1.5 UNK. 7B-Level Base LLM<br><br>Mini-Gemini-HD-8B 2880 LLaVA-NeXT-8B 2880<br><br>Cambrian-1-8B 576 StreamChat-7B 256<br><br>14B-Level Base LLM Mini-Gemini-HD-13B 2880<br><br>LLaVA-NeXT-13B 2880<br><br>Cambrian-1-13B 576 StreamChat-14B 256<br><br>|1409 75.8 56.8 57.1 69.1 36.8 75.7 78.2 78.0 61.4<br><br>1496 73.6 47.9 42.6 70.7 - 79.5 - - -<br><br>- - 58.5 - - - - 80.3 73.5 67.5<br><br>- - 53.6 - - - - 88.3 78.1 68.7<br><br><br>1606 72.7 37.3 - 73.2 64.5 75.1 73.5 70.2 62.1 1603 72.1 41.7 - 72.7 65.2 72.8 71.6 64.6 60.1 1547 75.9 42.7 - 74.7 64.6 80.4 73.0 71.7 64.2 1520 74.4 48.1 46.0 74.3 62.4 85.5 76.6 72.4 61.7<br><br>1597 68.6 37.3 - 70.6 63.7 71.9 70.1 70.2 57.5 1575 70.0 36.2 - 65.6 65.4 73.5 70.0 67.1 59.1 1610 75.7 40.0 - 74.4 64.3 79.3 73.6 72.8 63.0 1617 79.0 50.1 53.6 75.5 63.3 85.8 79.5 74.4 63.3<br><br>|

RealworldQA

#VisTok.

TextVQA

VMMMU

VMMStar

PMME

ISEED

MMB

AI2D

ISQA

GQA

- Table 1. Comparison of StreamChat with leading LMMs on image benchmarks. StreamChat achieves competitive performance on these benchmarks while using only 256 visual tokens.

scenarios and manually review and refine each remaining sample to ensure that the instruction and answer align with the video content and timestamp. Ultimately, we collect 100 evaluation samples, with 80 sourced from Vript [70] and 20 from Ego4D [17].

Following [33, 34], we use Gemini-1.5-Pro as the judge for performance evaluation. Given a video and its corresponding instruction, we infer two candidate models to predict their respective answers. We then feed the ground truth answer along with the outputs from the two models to the judge. The judge is required to evaluate the two answers on adherence, helpfulness, relevance, and accuracy. We prompt the judge to determine which model’s answer is better or if both are tied in terms of quality and then require it to provide a detailed justification explaining its reasoning based on the judgment. We use our StreamChat model as one of the candidate models and calculate the overall win rate in comparison to the other models.

#### 4.1. Quantitative Results

We show the comparison between StreamChat and other video LLMs in Figure 5. The frames per second (FPS) is set to 5. We use 32 frames for StreamChat and LLaVAVideo [81] models, and 16 frames for VILA [29] since using 32 would exceed its context range. Our results demonstrate that our StreamChat models exhibit superior streaming interaction capabilities compared to LLaVA-Video and VILA models. Notably, compared to VILA-1.5-40B, our StreamChat-7B model produces equally or more preferable

answers in 77% of the evaluation cases despite using a much smaller LLM. While LLaVA-Video models excel in offline video understanding, StreamChat-7B outperforms them in streaming interaction scenarios, highlighting the importance of capturing video dynamics during streaming inference. Furthermore, we observe that our StreamChat-14B demonstrates overall better performance than StreamChat-7B, indicating that scaling the base LLM can also improve the streaming interaction performance.

#### 4.2. Qualitative Results

We provide a qualitative evaluation of StreamChat’s capabilities on streaming video, as illustrated in Figure 6. In the example, we pose a question at a specific timestamp. While previous methods only answer the question using the visual context up to the moment the question is asked, StreamChat can dynamically update its visual context alongside the streaming video and adapt its answer accordingly. We show that StreamChat can better capture dynamic video content and provide more accurate answers. In contrast, VILA [29] and LLaVA-Video [81] struggle to maintain temporal alignment with the streaming video and exhibit factual errors (highlighted in red).

### 5. Benchmark Results

We evaluate the performance of StreamChat models on popular image [9, 13, 19, 20, 23, 35, 37, 46, 64, 74] and video benchmarks [14, 26, 38, 42, 62, 65, 73, 82] using the LMMs-

LongVideoBench

|Method<br><br>#Frames<br><br>|ActNet-QA<br><br>EgoSchema<br><br>MLVU<br><br>MVBench<br><br>NExT-QA<br><br>PerceptionTest<br><br>LongVideoBench<br><br>VideoMME|
|---|---|
|Private models<br><br>GPT-4V UNK. GPT-4o UNK. Gemini-1.5-Flash UNK. Gemini-1.5-Pro UNK.<br><br>7B-Level Base LLM LongVA-7B 128 IXC-2.5-7B 64<br><br>PLLaVA-7B 16 VideoLLaMA2-7B 16<br><br>StreamChat-7B 40 14B+ Level Base LLM<br><br>VILA-40B UNK.<br><br>PLLaVA-13B 16 PLLaVA-34B 16<br><br>VideoLLaMA2-72B 16 StreamChat-14B 40<br><br>|57.0 - 49.2 43.5 - - 61.3 59.9/63.3 - - 64.6 - - - 66.7 71.9/77.2<br><br>55.3 65.7 - - - - 61.6 70.3/75.0<br><br>57.5 72.2 - - - - 64.0 75.0/81.3<br><br>50.0 - 56.3 - 68.3 - - 52.6/54.3 52.8 - 37.3 69.1 71.0 34.4 - 55.8/58.8 56.3 - - 46.6 - - 40.2 -<br><br>50.2 51.7 48.5 54.6 - - - 47.9/50.3<br><br>54.9 48.4 63.9 53.3 78.5 63.0 54.2 58.6/62.8<br><br>58.0 58.0 - - 67.9 54.0 - 60.1/61.1<br><br>56.3 - - 50.1 - - 45.6 60.9 - - 58.1 - - 53.2 -<br><br>55.2 63.9 61.2 62.0 - - - 61.4/63.1 55.9 57.2 66.6 55.2 79.4 63.7 57.1 63.1/66.3<br><br><br><br><br><br><br>|

PerceptionTest

EgoSchema

VideoMME

ActNet-QA

MVBench

NExT-QA

#Frames

MLVU

- Table 2. Comparison of StreamChat with leading LMMs on video benchmarks. StreamChat achieves competitive performance on these benchmarks and even outperforms models with a much larger base LLM. StreamChat’s cross-attention-based architecture is efficient in processing a large number of video frames.

Eval library [78]. Note that to maintain StreamChat’s efficiency for streaming interactions, we do not employ multiple vision encoders [45, 53] or image tiling techniques [10, 32], which could compromise performance on benchmarks requiring high-resolution inputs.

The performance of StreamChat on image benchmarks is presented in Table 1. StreamChat demonstrates strong results compared to Cambrian-1, which utilizes multiple vision encoders, and LLaVA-NeXT, which employs image tiling. Notably, our StreamChat-7B achieves a score of 48.1 on the MMMU benchmark, surpassing LLaVA-NeXT-8B and Cambrian-1-8B by 6.4 and 5.4 points, respectively. Additionally, StreamChat outperforms both LLaVA-NeXT and Cambrian-1 on TextVQA, despite using significantly fewer visual tokens. Overall, StreamChat achieves competitive performance on image benchmarks while ensuring computation efficiency.

We present StreamChat’s performance on video benchmarks in Table 2. Our model significantly outperforms PLLaVA [68] and VideoLLaMA2 [11], using a 7B-level base LLM. Specifically, we achieve scores of 58.6/62.8 on the VideoMME benchmark [14], outperforming VideoLLaMA27B by 10.7/11.5 points. Moreover, StreamChat-14B demonstrates superior performance compared to VILA-40B and VideoLLaMA2-72B, which utilize much larger base LLMs.

Importantly, our model remains efficient even when processing more frames for inference, as our cross-attention-based architecture mitigates the heavy computation associated with self-attention across frames.

### 6. Ablation Studies

We use a relatively efficient setting for ablation studies. We shrink the total training steps to 2000 while keeping other hyperparameters the same as our full training. In the instruction tuning stage, unless otherwise specified, we employ a combination of our dense instruction dataset and Eagle1.8M [45] and train on the combination for 1 epoch. The training hyperparameters are the same as our full training.

We present the results of our ablation study in Table 3, where we ablate our architectural designs and the proposed dense instruction dataset. We compare performance across four image benchmarks, four video benchmarks, and our streaming evaluation. In the streaming evaluation, we employ our StreamChat solution (last row) as one of the candidate models and report the performance of other models relative to StreamChat.

Our experiment results indicate that the architectural enhancements we introduced lead to improved overall performance. Specifically, the StreamChat model outperforms the

PerceptionTest

|V-FFN<br><br>Linear Gate<br><br>Param. Reuse<br><br>Dense Instruction|VMMMU<br><br>AI2D<br><br>TextVQA<br><br>RealworldQA<br><br>|MLVU<br><br>MVBench<br><br>PerceptionTest<br><br>VideoMME<br><br>|StreamEval Win/Tie/Loss|
|---|---|---|---|
|✗ ✓ ✓ ✓ ✓ ✗ ✓ ✓ ✓ ✓ ✗ ✓ ✓ ✓ ✓ ✗ ✓ ✓ ✓ ✓<br><br>|46.7 75.7 62.7 57.8<br><br>45.1 74.8 60.7 58.3<br><br>44.4 72.4 46.9 46.3<br><br>46.0 76.5 62.5 59.4<br><br>45.2 76.1 63.3 58.0<br><br><br><br><br>|58.2 47.3 49.3 51.1 60.0 49.4 51.8 52.4 53.5 43.4 46.6 47.1 57.0 49.0 52.8 51.3<br><br>59.7 49.5 51.1 52.6<br><br><br>|18/46/36 25/45/30 20/33/47 25/34/41 -/-/-<br><br>|

RealworldQA

VideoMME

MVBench

TextVQA

VMMMU

MLVU

AI2D

- Table 3. Ablation study results. StreamEval indicates our proposed streaming evaluation, in which we use our final solution (last row) as one of the candidate models and report other models’ performance against our final solution.

version without the visual feedforward network (V-FFN) experts on 8 out of 9 benchmarks. Additionally, we observe that using a tanh gate facilitates faster convergence during the early stages of training; however, it ultimately results in poorer final performance compared to the linear gate. The linear gate improves performance on 6 out of 9 benchmarks when compared to the tanh gate. Furthermore, we observe a significant training instability when not reusing the LLM’s parameter, which also leads to poor final performance. Our final solution substantially outperforms the model without parameter reuse across all evaluated benchmarks.

When compared to the model trained without dense instruction data, our final solution performs comparably on existing image and video benchmarks. However, in the streaming evaluation, we demonstrate that training on our dense instruction dataset significantly enhances interaction capabilities. Our final solution produces equally or more preferable answers in 75% of the evaluation cases, indicating that reliance on existing image or video instruction tuning datasets alone is insufficient for effective streaming interactions.

### 7. Related Works

Large Multimodal Models. Large multimodal models (LMMs) have garnered significant attention for their robust zero-shot capabilities across various tasks, including image captioning [30] and visual question answering (VQA) [16, 18, 40]. Notably, Flamingo [2] showcases visual in-context learning by training on extensive interleaved image-text datasets. GPT-4V [1] exhibits emerging imageunderstanding capabilities, providing coherent responses to user queries. In the open-source domain, LLaVA reproduces aspects of GPT-4V’s functionality by fine-tuning on generated instruction-following data. Subsequent works, including LLaVA-1.5 [31], Qwen-VL [5], and CogVLM [58], aim to enhance model capabilities through architectural refinements, improved training methodologies, and higherquality training datasets. More recently, open models like In-

ternVL2 [10], LLaVA-OneVision [24], and Qwen2-VL [57] have demonstrated even better performance than state-of-theart closed models like GPT-4o [41] or Gemini-1.5-Pro [51], paving the path for research of LMMs.

Multimodal Video Models. More recent LMMs also incorporate video understanding capability to support a wider range of application scenarios [25, 29, 57, 68, 71, 76]. To support extreme long video understanding, various techniques are proposed to process more video frames during training and inference [47, 48, 69, 80]. Additionally, to reduce the redundancy of video information, more recent approaches also propose to compress the video via spatial or temporal compression [27, 36, 60, 61]. However, these methods focus on offline video understanding, where the entire video is available beforehand. In contrast, StreamChat focuses on processing streaming video, which is more suitable for real-world applications and interaction.

Streaming Video Models. The advent of streaming video models began with OpenAI’s GPT-4o [41], which has demonstrated remarkable proficiency in real-time interaction with streaming videos, attracting considerable interest in the field. Following this landmark development, several subsequent works have aimed to enhance large multimodal models for processing streaming video content. Notable approaches include methods such as VideoLLMonline [7], Flash-VStream [77], and subsequent methods [15, 28, 43, 59, 63, 66, 67, 79] , which focus on improving the fluency or responsiveness of models during streaming video processing. However, unlike these existing models, which often rely on fixed video content up to the moment a question is posed to answer questions, our work emphasizes the dynamic updating of the visual context during the decoding process, thereby significantly enhancing the interactive experience and mitigating the detrimental delays inherent in highly dynamic environments.

### 8. Conclusion

This paper presents StreamChat, a novel approach that enhances the real-time interaction capabilities of large multimodal models (LMMs) with streaming video content. StreamChat is built on a flexible and efficient cross-attentionbased architecture with visual feedforward network (V-FFN) experts. By continuously updating the visual context at each decoding step, StreamChat effectively captures the dynamic changes of streaming video content, leading to temporally aligned responses. We also introduce a dense instruction dataset to facilitate the training of streaming interaction models, alongside a parallel 3D-RoPE mechanism to better arrange the streaming video and text. Our extensive evaluations on both established image and video benchmarks and a novel streaming benchmark demonstrate that StreamChat not only achieves competitive performance on existing benchmarks but also excels in streaming interaction scenarios.

### 9. Limitations

While StreamChat demonstrates significant advancements in streaming interaction capabilities for large multimodal models (LMMs), several limitations remain. One limitation is that the timestamps for each text token are generated heuristically from coarse-grained temporal annotation rather than being manually annotated. This reliance on heuristics may introduce inaccuracies in temporal alignment, particularly in complex video scenarios where precise timing is crucial.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 8
- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736,

2022. 8

- [3] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736,

- 2022. 2, 3

[4] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609,

- 2023. 1

- [5] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023. 8

- [6] Lucas Beyer, Andreas Steiner, Andr´e Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, et al. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726, 2024. 4
- [7] Joya Chen, Zhaoyang Lv, Shiwei Wu, Kevin Qinghong Lin, Chenan Song, Difei Gao, Jia-Wei Liu, Ziteng Gao, Dongxing Mao, and Mike Zheng Shou. Videollm-online: Online video large language model for streaming video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18407–18418, 2024. 1, 8
- [8] Kaibing Chen, Dong Shen, Hanwen Zhong, Huasong Zhong, Kui Xia, Di Xu, Wei Yuan, Yifei Hu, Bin Wen, Tianke Zhang, et al. Evlm: An efficient vision-language model for visual understanding. arXiv preprint arXiv:2407.14177, 2024. 3
- [9] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large visionlanguage models? arXiv preprint arXiv:2403.20330, 2024. 6
- [10] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024. 1, 4, 7, 8
- [11] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatialtemporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024. 1, 2, 7
- [12] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 1, 2, 3

- [13] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023. 6
- [14] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024. 6, 7
- [15] Chaoyou Fu, Haojia Lin, Zuwei Long, Yunhang Shen, Meng Zhao, Yifan Zhang, Xiong Wang, Di Yin, Long Ma, Xiawu Zheng, et al. Vita: Towards open-source interactive omni multimodal llm. arXiv preprint arXiv:2408.05211, 2024. 8
- [16] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913, 2017. 8
- [17] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger,

- Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18995–19012, 2022. 2, 4, 6
- [18] Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3608–3617,

2018. 8

- [19] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, 2019. 6
- [20] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 235–

251. Springer, 2016. 6

- [21] Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of NAACL-HLT, page 2. Minneapolis, Minnesota, 2019. 1
- [22] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In Proceedings of the IEEE international conference on computer vision, pages 706–715, 2017. 2, 4
- [23] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension, 2023. 6
- [24] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 1, 8
- [25] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 8
- [26] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195– 22206, 2024. 2, 4, 6
- [27] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. In European Conference on Computer Vision, pages 323–340. Springer,

2024. 8

- [28] Junming Lin, Zheng Fang, Chi Chen, Zihao Wan, Fuwen Luo, Peng Li, Yang Liu, and Maosong Sun. Streamingbench: Assessing the gap for mllms to achieve streaming video understanding. arXiv preprint arXiv:2411.03628, 2024. 8
- [29] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26689–26699, 2024. 6, 8

- [30] Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C. Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 8
- [31] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023. 1, 8
- [32] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 4, 7
- [33] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024. 3, 4, 6
- [34] Jihao Liu, Xin Huang, Jinliang Zheng, Boxiao Liu, Jia Wang, Osamu Yoshie, Yu Liu, and Hongsheng Li. Mm-instruct: Generated visual instructions for large multimodal model alignment. arXiv preprint arXiv:2406.19736, 2024. 1, 6
- [35] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023. 6
- [36] Zhijian Liu, Ligeng Zhu, Baifeng Shi, Zhuoyang Zhang, Yuming Lou, Shang Yang, Haocheng Xi, Shiyi Cao, Yuxian Gu, Dacheng Li, et al. Nvila: Efficient frontier visual language models. arXiv preprint arXiv:2412.04468, 2024. 8
- [37] Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In The 36th Conference on Neural Information Processing Systems (NeurIPS), 2022. 6
- [38] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very longform video language understanding. Advances in Neural Information Processing Systems, 36, 2024. 6
- [39] Ben Mann, N Ryder, M Subbiah, J Kaplan, P Dhariwal, A Neelakantan, P Shyam, G Sastry, A Askell, S Agarwal, et al. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 1, 2020. 1
- [40] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE/cvf conference on computer vision and pattern recognition, pages 3195–3204, 2019. 8
- [41] OpenAI. Hello gpt-4o. https://openai.com/index/ hello-gpt-4o/, 2024. 1, 8
- [42] Viorica P˘atr˘aucean, Lucas Smaira, Ankush Gupta, Adri`a Recasens Continente, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Joseph Heyward, Mateusz Malinowski, Yi Yang, Carl Doersch, Tatiana Matejovicova, Yury Sulsky, Antoine Miech, Alex Frechette, Hanna Klimczak, Raphael Koster, Junlin Zhang, Stephanie Winkler, Yusuf Aytar, Simon Osindero, Dima Damen, Andrew Zisserman, and Jo˜ao Carreira. Perception test: A diagnostic benchmark for multimodal video models. In Advances in Neural Information Processing Systems, 2023. 6

- [43] Rui Qian, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Shuangrui Ding, Dahua Lin, and Jiaqi Wang. Streaming long video understanding with large language models. Advances in Neural Information Processing Systems, 37:119336–119360, 2024. 8
- [44] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020. 1
- [45] Min Shi, Fuxiao Liu, Shihao Wang, Shijia Liao, Subhashree Radhakrishnan, De-An Huang, Hongxu Yin, Karan Sapra, Yaser Yacoob, Humphrey Shi, et al. Eagle: Exploring the design space for multimodal llms with mixture of encoders. arXiv preprint arXiv:2408.15998, 2024. 4, 7
- [46] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019. 6
- [47] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, et al. Moviechat: From dense token to sparse memory for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18221–18232, 2024. 8
- [48] Enxin Song, Wenhao Chai, Tian Ye, Jenq-Neng Hwang, Xi Li, and Gaoang Wang. Moviechat+: Question-aware sparse memory for long video question answering. arXiv preprint arXiv:2404.17176, 2024. 8
- [49] Jianlin Su. Totary position embedding, 2024. 2, 3
- [50] Yansong Tang, Dajun Ding, Yongming Rao, Yu Zheng, Danyang Zhang, Lili Zhao, Jiwen Lu, and Jie Zhou. Coin: A large-scale dataset for comprehensive instructional video analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1207–1216,

2019. 2, 4

- [51] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 1, 4, 8
- [52] Qwen Team. Qwen2.5: A party of foundation models, 2024. 1, 2, 4
- [53] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860, 2024. 1, 7
- [54] Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles, Gabriel Synnaeve, and Herv´e J´egou. Going deeper with image transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 32–42, 2021. 3
- [55] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 1

- [56] A Vaswani. Attention is all you need. Advances in Neural Information Processing Systems, 2017. 2
- [57] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 1, 2, 3, 8
- [58] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079, 2023. 8
- [59] Yueqian Wang, Xiaojun Meng, Yuxuan Wang, Jianxin Liang, Jiansheng Wei, Huishuai Zhang, and Dongyan Zhao. Videollm knows when to speak: Enhancing time-sensitive video comprehension with video-text duet interaction format. arXiv preprint arXiv:2411.17991, 2024. 8
- [60] Yuxuan Wang, Cihang Xie, Yang Liu, and Zilong Zheng. Videollamb: Long-context video understanding with recurrent memory bridges. arXiv preprint arXiv:2409.01071, 2024. 8
- [61] Yuetian Weng, Mingfei Han, Haoyu He, Xiaojun Chang, and Bohan Zhuang. Longvlm: Efficient long video understanding via large language models. In European Conference on Computer Vision, pages 453–470. Springer, 2024. 8
- [62] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding, 2024. 6
- [63] Shiwei Wu, Joya Chen, Kevin Qinghong Lin, Qimeng Wang, Yan Gao, Qianli Xu, Tong Xu, Yao Hu, Enhong Chen, and Mike Zheng Shou. Videollm-mod: Efficient video-language streaming with mixture-of-depths vision computation. arXiv preprint arXiv:2408.16730, 2024. 1, 8
- [64] x.ai. Grok-1.5 vision preview. 6
- [65] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9777–9786, 2021. 6
- [66] Zhifei Xie and Changqiao Wu. Mini-omni: Language models can hear, talk while thinking in streaming. arXiv preprint arXiv:2408.16725, 2024. 8
- [67] Zhifei Xie and Changqiao Wu. Mini-omni2: Towards opensource gpt-4o model with vision, speech and duplex. arXiv preprint arXiv:2410.11190, 2024. 1, 8
- [68] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. Pllava: Parameter-free llava extension from images to videos for video dense captioning. arXiv preprint arXiv:2404.16994, 2024. 1, 2, 7, 8
- [69] Fuzhao Xue, Yukang Chen, Dacheng Li, Qinghao Hu, Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang Yang, Zhijian Liu, et al. Longvila: Scaling long-context visual language models for long videos. arXiv preprint arXiv:2408.10188,

2024. 1, 8

- [70] Dongjie Yang, Suyuan Huang, Chengqiang Lu, Xiaodong Han, Haoxin Zhang, Yan Gao, Yao Hu, and Hai Zhao. Vript: A video is worth thousands of words. arXiv preprint arXiv:2406.06040, 2024. 4, 6

- [71] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024. 8
- [72] Jiabo Ye, Haiyang Xu, Haowei Liu, Anwen Hu, Ming Yan, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl3: Towards long image-sequence understanding in multi-modal large language models. arXiv preprint arXiv:2408.04840,

2024. 3

- [73] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 9127–9134, 2019. 6
- [74] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, and Yuxuan Sun. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In CVPR, 2024. 6
- [75] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023. 4
- [76] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. 1, 2, 8
- [77] Haoji Zhang, Yiqin Wang, Yansong Tang, Yong Liu, Jiashi Feng, Jifeng Dai, and Xiaojie Jin. Flash-vstream: Memorybased real-time understanding for long video streams. arXiv preprint arXiv:2406.08085, 2024. 1, 8
- [78] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Lmmseval: Reality check on the evaluation of large multimodal models, 2024. 7
- [79] Pan Zhang, Xiaoyi Dong, Yuhang Cao, Yuhang Zang, Rui Qian, Xilin Wei, Lin Chen, Yifei Li, Junbo Niu, Shuangrui Ding, et al. Internlm-xcomposer2. 5-omnilive: A comprehensive multimodal system for long-term streaming video and audio interactions. arXiv preprint arXiv:2412.09596, 2024. 8
- [80] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024. 1, 8
- [81] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 1, 2, 4, 6
- [82] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: A comprehensive benchmark for multi-task long video understanding. arXiv preprint arXiv:2406.04264, 2024. 6
- [83] Luowei Zhou, Nathan Louis, and Jason J Corso. Weaklysupervised video object grounding from text by loss weighting and object interaction. arXiv preprint arXiv:1805.02834,

2018. 4

[84] Wanrong Zhu, Jack Hessel, Anas Awadalla, Samir Yitzhak Gadre, Jesse Dodge, Alex Fang, Youngjae Yu, Ludwig Schmidt, William Yang Wang, and Yejin Choi. Multimodal c4: An open, billion-scale corpus of images interleaved with text. Advances in Neural Information Processing Systems, 36, 2024. 4

