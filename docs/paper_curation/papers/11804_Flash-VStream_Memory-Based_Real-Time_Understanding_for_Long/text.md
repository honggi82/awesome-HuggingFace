# arXiv:2406.08085v2[cs.CV]30Jun2024

## Flash-VStream: Memory-Based Real-Time Understanding for Long Video Streams

Haoji Zhang1∗ Yiqin Wang1∗ Yansong Tang1† Yong Liu1 Jifeng Dai2 Jiashi Feng3 Xiaojie Jin3†‡ 1Shenzhen International Graduate School, Tsinghua University 2Department of Electronic Engineering, Tsinghua University 3ByteDance Inc. {haoji-zh20@mails.,yq-wang23@mails.,tang.yansong@sz.}tsinghua.edu.cn jinxiaojie@bytedance.com

### Abstract

Benefiting from the advancements in large language models and cross-modal alignment, existing multi-modal video understanding methods have achieved prominent performance in offline scenario. However, online video streams, as one of the most common media forms in the real world, have seldom received attention. Compared to offline videos, the “dynamic” nature of online video streams poses challenges for the direct application of existing models and introduces new problems, such as the storage of extremely long-term information, interaction between continuous visual content and “asynchronous” user questions. Therefore, in this paper we present Flash-VStream, a video-language model that simulates the memory mechanism of human. Our model is able to process extremely long video streams in real-time and respond to user queries simultaneously. Compared to existing models, Flash-VStream achieves significant reductions in inference latency and VRAM consumption, which is intimately related to performing understanding of online streaming video. In addition, given that existing video understanding benchmarks predominantly concentrate on offline scenario, we propose VStream-QA, a novel question answering benchmark specifically designed for online video streaming understanding. Comparisons with popular existing methods on the proposed benchmark demonstrate the superiority of our method for such challenging setting. To verify the generalizability of our approach, we further evaluate it on existing video understanding benchmarks and achieves state-of-the-art performance in offline scenarios as well. All code, models, and datasets are available at the project page.

### 1 Introduction

Online video streaming is a prevalent media format with a broad spectrum of applications. In the field of robotics, for instance, robots operating in the wild can leverage stream understanding models to interpret and react to their environment in real-time [36, 38]. Similarly, in surveillance systems, stream understanding models can process and analyze video streams from specific locations continuously, thereby improving overall security [5, 32]. However, best existing large video-language models fails to perform real-time long video question-answering upon user queries [16, 20, 29, 37]. The main reason is that: visual tokens between consecutive frames are heavy and redundant without effective compression, making it impossible to save all visual features in limited GPU Memory (VRAM), as well as significantly increasing the decoding latency of language model.

∗Equal contribution. †Correspondence to Xiaojie Jin <jinxiaojie@bytedance.com> and Yansong Tang <tang.yansong@sz.tsinghua.edu.cn>. ‡Project lead. Project page https://InvinciblWyq.github.io/vstream-page .

Preprint. Under review.

###### Conventional(offline) Video LLM

###### Single Process: All steps done sequentially

Timeline

[Figure 1]

Video Encoder

[ 𝐭 = 𝟎𝐬 ]

|𝐀𝐧𝐬𝐰𝐞𝐫𝟏|
|---|

[ 𝐭 = 𝒕𝒆𝒏𝒄𝒐𝒅𝒆 + 𝒕𝒅𝒆𝒄𝒐𝒅𝒆]

LLM

All frames

[ 𝐭 = 𝟎𝐬 ] 𝐐𝐮𝐞𝐬𝐭𝐢𝐨𝐧𝟏

(a) Inference process of conventional (offline) Video LLM

###### Human (1fps)& Online Video Stream LLM (1fps)

Human memory STAR memory (ours)

Process 2: Answer upon user question Decode upon user question

Retrieve at 𝐭 = 𝐓𝑸𝟏

Update

[Figure 2]

Human eyes

Retrieve at 𝐭 = 𝐓𝑸𝟏

- at t=0s

Update

- at t=1s

[ 𝐭 = 𝟎𝐬 ]

Update

Frame Encoder

|𝐐𝐮𝐞𝐬𝐭𝐢𝐨𝐧𝟏|
|---|

[ 𝐭 = 𝐓𝑸𝟏 ]

- at t=0s

Update

- at t=1s

###### Frame 1

Timeline

LLM

[Figure 3]

Human eyes

[ 𝐭 = 𝐓𝑸𝟏 + 𝒕𝒅𝒆𝒄𝒐𝒅𝒆 ]

|𝐀𝐧𝐬𝐰𝐞𝐫𝟏|
|---|

[ 𝐭 = 𝟏𝐬 ]

Brain

Frame Encoder

[ 𝐭 = 𝐓𝑸𝟏 +

|𝐀𝐧𝐬𝐰𝐞𝐫𝟏|
|---|

###### Frame 2

Process 1: 𝒕𝒅𝒆𝒄𝒐𝒅𝒆 ] Non-stop eye perceiving

.

Non-stop online encoding

(b) Human process of online video stream QA (c) Inference process of online video stream LLM (ours)

###### Figure 1: Comparing (a) conventional offline pipeline and (b) human processing pipeline with (c) our proposed Flash-VStream for online video streaming understanding. Zoom in for better view.

Considering how humans process live video streams in real-time can provide inspiration for the design of video stream understanding models. This procedure can be divided into four steps [10]: 1) Perceiving: human eyes continuously encode an endless visual information into brain. 2) Memorizing: human brain compresses the visual information and update brain memory with it. With limited memory capacity, humans tend to have clearer detailed memories of recent events while they only remember the most important parts of events from the distant past. 3) Recalling: whenever a person is asked about what happens before, his/her brain retrieve the memory. 4) Answering: human brain integrates the memory information with the context provided by the question, and generate an answer.

It is worth noting that the four human processing steps above are not strictly sequential. As shown in

- Figure 1 (b) (focus on the brown part and ignore the blue part), the first two steps can be performed by a process (on the left), while the last two steps being performed by another process simultaneously (on the right). In other words, humans can perceive and memorize new information while recalling and answering questions about the past simultaneously. While the “process” for perceiving and memorizing is always running, the “process” for recalling and answering is only activated upon user questions. This is the key to online video stream understanding. In contrast, most existing video-QA methods [16, 20, 29] are based on offline video understanding, where user query and finite-length video are given to the model at the same time. As shown in Figure 1 (a), these methods only consist of the two strictly sequential steps: perceiving and answering. The lack of a compressed memory mechanism in these offline methods result in a dilemma: 1) If the model keeps the redundant visual tokens of all frames, the high VRAM consumption leads to limited input frame capacity. 2) If the model performs question-aware encoding and only keep those visual tokens that are relevant to the question, it has to re-encode all the visual information from scratch every time a new query is given, leading to an unacceptable inference latency for online video streams.

To address this challenge, we introduce Flash-VStream, a video-language model that is able to process extremely long video streams in real-time and respond to user queries simultaneously. As shown in Figure 1 (c), Flash-VStream (blue) highly resembles human processing pipeline (brown) in terms of “4-step, 2-process” design philosophy. The frame encoder resembles human eyes and the LLM resembles human brain. The learnable memory mechanism in Flash-VStream, named Spatial-Temporal-Abstract-Retrieved (STAR) memory, is carefully designed to compress necessary visual information and update memory in a online and real-time manner, as shown in Figure 3.

In addition, recognizing the limitations of existing offline and short-length video QA benchmarks, for evaluating video stream understanding in online settings, we propose VStream-QA, a novel question answering benchmark specifically designed for online video stream understanding. The main features of VStream-QA lies in: i) Each question-answer pair is marked with a specific timestamp in the video and only related to the visual information before that timestamp, which is consistent with

8

MovieChat Chat-UniVi LLaMA-VID Flash-VStream

6

Latency(seconds)

4

2

latency=1 second

0

100 101 102 103 104 Frames

- Figure 2: Inference latency (y-axis) v.s. frame number (x-axis). Latency tested on an A100 gpu. Our model is able to process extremely long video streams, and perform real-time answering within 1 second upon user’s query.

RVS-Ego RVS-Movie A S A S VRAM↓

Method

Video-ChatGPT [30] 51.0 3.7 51.7 3.3 16.62GB * MovieChat [37] 50.7 3.4 36.0 2.3 16.90GB † Chat-UniVi [16] 51.2 3.8 51.8 3.3 77.56GB † LLaMA-VID [20] 53.4 3.9 48.6 3.3 33.64GB †

###### Flash-VStream 57.3 4.0 53.1 3.3 16.03GB †

Table 1: Comparison with SoTA methods on zero-shot real-time VideoQA. A and S denote accuracy and score, respectively. VRAM tested on an A100 gpu. *: Tested with a 100-frame input video (maximum support of VideoChatGPT). †: Tested with a 1000-frame input video.

the online video stream understanding setting. ii) The video length ranges from 30 minutes to 60 minutes, which is significantly longer than existing benchmarks, making it capable of evaluating model’s performance on extremely long videos. iii) The videos cover a variety of content, including first-person perspective (ego-centric) videos, and third-person perspective movies.

On these challenging online benchmarks, Flash-VStream achieves state-of-the-art performance, while achieving significant reductions in inference latency and VRAM consumption as shown in

- Figure 2 and Table 1. Zero-shot video question answering experiments on 4 conventional offline video QA benchmarks further prove the generalization ability of Flash-VStream, as shown in Table 3. Comprehensive ablation studies prove the effectiveness of the memory mechanism we adopted. We summarize our contribution as follows:

- • We introduce Flash-VStream, a novel large video-language model that is able to process extremely long video streams in real-time and respond to user queries simultaneously. A cleverly designed memory mechanism named STAR is introduced to compress necessary visual information while leaving out the redundancy between consecutive frames.
- • While maintaining state-of-the-art performance on both online and offline benchmarks, FlashVStream achieves significant reductions in inference latency and GPU Memory (VRAM) consumption, enabling online video stream QA in real-time.
- • We also propose VStream-QA, a new QA benchmark specifically designed for video understanding in online settings. Its question-answer-timestamp triplet design is consistent with online scenario and its video length is significantly longer than existing benchmarks, making it capable of evaluating model’s performance on nearly-infinite long video streams.

### 2 Related work

Multi-modal large language models. With recent advances in Large Language Models (LLMs) [3, 34, 40, 41], many works try to build Multimodal Large Language Models (MLLMs) that integrate text with visual data or other modalities. For instance, the BLIP series [9, 17, 18] proposed a efficient strategy for bootstrapping multimodal understanding with pretrained LLMs and image encoders, and the LLaVA series [22, 23] leverage GPT-generated visual instruction data to tune open language models. With the development of image-text models, researchers have begun extending image data to videos. The biggest challenge for Video LLM is how to compress redundant frame features. LLaMA-VID [20] represents single-frame features with a few tokens, Chat-UniVi [16] employs dynamic tokens to model image and video features of different scale, and Vista-LLaMA [29] uses a sequential visual projector to represent an entire video with fewer tokens. These methods either requires a multi-step visual encoding process with high latency [16], or have a linearly increasing VRAM cost with the number of frames [20, 29], making them unsuitable for real-time long video stream understanding. MovieChat [37] proposed to combine all frame features through a simple average strategy. Though it is able to process long video with limited VRAM cost, its performance is suboptimal due to its training-free framework and non-learnable memory mechanism. In our proposed Flash-VStream, we introduce a learnable memory mechanism that encode frames in a

online and real-time manner, disentangling the visual encoding process and answer decoding process, thus enabling real-time video stream understanding.

Real-time video stream understanding. Real-time video stream understanding is a challenging task that requires the model to process video streams in real-time and finish specific tasks based on the video. Most existing real-time methods are designed to perform a single, specific vision task, such as real-time object tracking [14, 25, 42] and real-time action recognition [28, 48]. Considering natural language is becoming a general interface for various tasks and modalities [1, 11, 17, 26], our work focuses on real-time video stream question answering upon user queries, which is a more challenging and comprehensive task.

Memory mechanism for long sequence processing. Memory mechanism is widely used to store and retrieve information in all forms of sequence processing tasks, such as time series forecasting [4], recommendation system [39], machine translation [8], and video object segmentation [6]. Inspired by the idea of Neural Turing Machine (NTM) [13], a learnable mechanism that resembles the working memory system of human cognition, we proposed a learnable visual memory that is able to compress visual information and update memory in a online and real-time manner.

### 3 Flash-VStream

FrameHandlerProcess QuestionHandlerProcess

𝑒 𝑒 𝑒 𝑒 𝑒 𝑒 FeatureBuffer ……

Question

VisualEncoder

[Figure 4]

𝑒

S R R R

Projector

LLM

[Figure 5]

T T T T T T

- S

- T A R

Spatial Memory Temporal Memory Abstract Memory Retrieved Memory

[Figure 6]

[Figure 7]

🔥 🔥

A A A A A A A A A A A A

……

STARMemory

Answer

- Figure 3: The overview of Flash-VStream framework for real-time online video stream understanding. Flash-VStream is executed by two processes, namely “frame handle” and “question handler”. The frame handler is responsible for encoding frames and writing to memory, which contains a visual encoder, a STAR memory and a feature buffer. The question handler is responsible for reading from memory and answering questions anytime, which contains a projector and a Large Language Model.

As shown in Figure 3, our Flash-VStream framework consists of three main components: (1) a streaming visual encoder that continuously processes video frames, (2) a Spatial-Temporal-AbstractRetrieved memory mechanism (STAR memory), including memory writing and reading with the help of a feature buffer. (3) a LLM decoder capable of providing real-time responses to questions raised by users. To perform real-time inference, Flash-VStream is deployed in two asynchronous processes. The frame handler process manages the streaming visual encoder and STAR memory consolidation. The question handler process manages the real-time LLM decoder, STAR memory reading and interactions with users. The only connection between these two processes is the shared memory, which can be written by the first process and read by both.

- 3.1 Streaming visual encoder

Like human eyes, the streaming visual encoder can continuously encode visual information into embedded features. We use the pre-trained CLIP ViT-L [35] as visual encoder. Only patch tokens are used during training and inference. Specifically, given a frame stream {V t}∞t=1, the encoder maps the t-th frame V t ∈ RH×W×3 to feature map et ∈ RP×P×D, where P × P is the number of ViT patch tokens and D is the hidden dimension of ViT.

- 3.2 Spatial-Temporal-Abstract-Retrieved memory

In order to handle information of different levels of granularity, we design a STAR memory with

- 4 components: spatial memory Mspa ∈ RN

spa×Pspa2 ×D, temporal memory Mtem ∈ RN

tem×Ptem2 ×D,

Temporal Memory

Spatial Memory

𝑒 T T T T T T T T

𝑒 𝑒 𝑒 𝑒

Weighted K-means Clustering

𝑒 A A A A A A A A A A A A T T T T T T T T

|query projector|
|---|

|key projector|
|---|

Select top-K largest clusters

T T T

###### 𝑒 FeatureBuffer

| |softmax|
|---|---|
| | |

𝑒 𝑒 𝑒 𝑒 𝑒 ……

W

Semantic Attention

×(1 − 𝛼)

x

[Figure 8]

🔥

Retrieve K key frame features closest to K centroids

Abstract Memory

Retrieved Memory

R R R

A A A A A A A A A A A A

Figure 4: STAR memory writing mechanism. (a) Update spatial memory by a FIFO queue. (b) Update temporal memory by Weighted K-means Clustering. (c) Update abstract memory by Semantic Attention. (d) Update retrieved memory by key frame feature retrival. Here feature map eT has multiple sizes. “S”, “T”, “A” and “R” represent tokens of spatial, temporal, abstract and retrieved memory, respectively.

ret×Pspa2 ×D. A feature buffer Mbuff ∈ RN

abs×Pabs2 ×D and retrieved memory Mret ∈ RN

abstract memory Mabs ∈ RN

buff×Pspa2 ×D is used to store the feature of latest Nbuff frames. Therefore, the overall memory size is limited to MAXSIZE = (Nspa + Nret) × Pspa2 + Ntem × Ptem2 + Nabs × Pabs2 tokens.

Spatial memory. Spatial memory houses the most recent and detailed spatial information for short-term use, implemented as a FIFO (First-In-First-Out) queue, as illustrated in Figure 4 and Equation (2). This architecture enables continuous updating with the newest frames, facilitating immediate access to fine-grained spatial data.

Temporal memory. Temporal memory integrates dynamic information over time, crucial for longterm retention. When its size surpasses Ntem, the gwkmeans (Weighted K-means Clustering) algorithm is applied, as shown in Equation (3) and Algorithm 1. This strategy condenses the memory content into Ntem clusters which can be seen as the representation of key events in videos. Then the centroids of these clusters are used as the new memory for efficiently storing temporal contexts.

Abstract memory. Abstract memory supports high-level semantic concept interpretation through fSA, the Semantic Attention model. It follows Equation (4) to synthesize the insights gained from both spatial and temporal memories into abstracted, actionable knowledge. fSA keeps adjusting Mabs, the synopsis of whole video by newest features. Refer to Figure 4 and Algorithm 2 for details.

Retrieved memory. Retrieved memory focuses on recalling precise spatial details by identifying and retrieving the most substantial frame features. As shown in Figure 4, it first selects the top-K (where K equals Nret) largest clusters from the Ntem clusters obtained in temporal memory Mtem. Then the nearest frame features in feature buffer to centroids of these K clusters are retrieved to supplement the temporal memory with more detailed spatial information. This process is illustrated in Equation (5) and Algorithm 3.

In brief, a new feature et is written to STAR memory as follows:

Mbufft = concat gpooling(et,Pspa),Mbufft−1 [0 : Nbuff,:,:] (1) Mspat = Mbufft [0 : Nspa,:,:] (2) Mtemt = gwkmeans concat gpooling(et,Ptem),Mtemt−1 ,Ntem (3) Mabst = fSA Mabst−1,gpooling(et,Pabs),Nabs (4) Mrett = gretrieve(Mbufft ,Mtemt ,Nret) (5)

Here gpooling(e,P′) applies Average Pooling to compress feature map e from P2 to P′2 size along width and height dimensions. concat(a,b) means concatenating tensors a and b along time axis.

#### 3.3 Real-time LLM decoder

The LLM decoder works as part of a real-time question answering server. When triggered by a question Qt at time t, the LLM decoder first calculates the text embedding Itextt = fembed(Qt) and maps the STAR memory Mt = Mspat + Mtemt + Mabst + Mrett to embedding space with the projector Ivisiont = fproj(Mt). Then it starts to generate answer At = fLLM(Itextt ,Ivisiont ).decode() in real time.

#### 3.4 Implementation details

In this study, we utilize pre-trained CLIP ViT-L/14-224px [35] as streaming visual encoder. Following LLaVA [24], we choose a 2-layer-MLP as visual projector and pre-trained Vicuna-7B [7] as LLM decoder. Considering the balance between performance and resource consumption, we set Pspa = 8, Ptem = 4, Pabs = 1, Nbuff = 300, Nspa = 1, Ntem = Nabs = 25 and Nret = 3. The MAXSIZE of STAR memory is set to 681 tokens in order to keep computational efficiency.

We train Flash-VStream for 2 stages: modality alignment and instruction tuning. The training data keep the same with LLaMA-VID [20], including LLaVA-filtered-558K [23] image-caption pairs and LLaMA-VID-filtered-232K [20] video-caption pairs for stage 1, LLaVA-filtered-665K [23] image QA pairs and Video-ChatGPT-filtered-98K [30] video QA pairs for stage 2. For each stage, the model is trained for 1 epoch on 8 A100 80G GPUs. During training, the parameters of visual encoder are frozen and the parameters of LLM are frozen only for the first stage. All training and inference experiments was conducted under BF16 precision to save time and resources. Other hyper-parameters can be found at Table 7.

### 4 VStream-QA: A new benchmark for online video stream QA

Previous video QA benchmarks [43, 44, 47] mostly focus on offline video understanding, where user query and finite-length video are given to the model at the same time. To our best knowledge, there is no existing benchmark specifically designed for online video stream understanding. Also, most existing benchmarks are limited to short-length videos within 1 minute [43, 44] or medium-length videos within 10 minutes [29, 31, 37, 47], which are unsuitable for simulating online video stream.

To address this problem, we propose VStream-QA, a novel question answering benchmark specifically designed for online video stream understanding. VStream-QA consists of two parts: VStreamQA-Ego and VStream-QA-Movie, which are designed for evaluating first-perspective ego-centric understanding and third-perspective plot understanding, respectively. The prominent features of VStream-QA are i) each question-answer pair is marked with a specific timestamp in the video and only related to the visual information before that timestamp, ii) containing extremely videos (30 minutes to 60 minutes) that is significantly longer than existing benchmarks, and iii) covering a variety of video sources and question types.

- Table 2: Video QA Benchmark Comparison. V for video duration, Q for number of questions, and Desc for descriptive.

[Figure 9]

#### Benchmark Avg V. Total V. Q. Goal

MSVD-QA [44] 10s 1.4h 13K Desc. QA MSRVTT-QA [44] 15s 12.5h 73K Desc. QA ActivityNet-QA [47] 112s 25h 8K Desc. QA Next-QA [43] 40s 11h 9K Temporal QA CineCLIP-QA [29] 213s 9h 2.5K Movie QA

Online Video Stream QA Figure 5: Question Types.

VStream-QA 40min 21h 3.5K

Specifically, VStream-QA-Ego consists of 10 1-hour-long ego-centric video clips from Ego4D dataset [12] together with 1.5K question-answer-timestamp triplets , while VStream-QA-Movie consists of 22 half-an-hour-long movie clips from MovieNet dataset [15] together with 2K questionanswer-timestamp triplets. As shown in Table 2, these two parts consist of a total of 21 hours of video and 3.5K question-answer pairs. Our proposed VStream-QA fills the gap in existing benchmarks for online video stream understanding, and provides a extremely long video test set that can be used to evaluate in both online settings and conventional offline settings.

- Table 3: Comparison with SoTA methods on zero-shot VideoQA. Acc. and Sco. denote accuracy and score, respectively. *: Evaluated by us.

ActNet NExT MSVD MSRVTT VS-Ego VS-Movie

Method

Acc. Sco. Acc. Sco. Acc. Sco. Acc. Sco. Acc. Sco. Acc. Sco. Video-ChatGPT [30] 35.2 2.7 54.6 3.2 64.9 3.3 49.3 2.8 51.7 3.7 54.4 3.4 MovieChat [37] 45.7 3.4 49.9 2.7 75.2 3.8 52.7 2.6 52.2 3.4 39.1 2.3 Chat-UniVi [16] 45.8 3.2 60.8* 3.3 65.0 3.6 54.6 3.1 50.9 3.8 54.0 3.4 Vista-LLaMA [29] 48.3 3.3 60.7 3.4 65.3 3.6 60.5 3.3 - - - LLaMA-VID [20] 47.4 3.3 60.3* 3.4 69.7 3.7 57.7 3.2 54.8 3.9 51.4 3.4 Flash-VStream 51.9 3.4 61.6 3.4 80.3 3.9 72.4 3.4 59.0 3.9 56.1 3.4

We carefully design 5 types of questions to evaluate the model’s ability to understand both scene content and temporal information. As shown in Figure 5, the question types are well balanced. Specifically, [Scene Summary] and [Action Description] are open-ended questions designed to evaluate the model’s ability to understand static and dynamic scene content. [Event Occurrence] are yes/no questions designed to evaluate the model’s ability to detect whether a specific event or scene occurs in the video. [Ordered Event Narrative] and [Sequence Validation] are both designed to evaluate the model’s ability to understand the temporal order of events in the video, with the former being open-ended and the latter being yes/no questions. For yes/no questions, its answer ratio is well balanced with 46.3% yes and 53.7% no.

In order to balance the annotation quality, the data scale, and the total annotation expenses, we designed a 5-steps data generation pipeline as follows: 1) Video Selection; 2) Dense Captioning; 3) Summary Generation; 4) Question-Answer Generation; and 5) Human Filtering. For details of each steps, please refer to Appendix C.1.

### 5 Experiment

#### 5.1 Experimental setup

Datasets. For the purpose of real-time video stream understanding, it is crucial for models to keep accurate and efficient. To evaluate real-time understanding ability and computational efficiency of models, we them models on Realtime-VStream-QA-Ego/Movie datasets (or RVS-Ego/Movie for short). The real-time version of VStream-QA differentiates normal version by ensuring each question grounded before a predefined timestamp. To evaluate the basic question answering capability of FlashVStream, we conduct zero-shot open-ended video question answering experiments on ActivityNet-QA [47], NExT-QA [43], MSVD-QA [44], MSRVTT-QA [44] and the proposed VStream-QA-Ego/Movie datasets (or VS-Ego/Movie for short).

Evaluation Metrics. For open-ended video question answering tasks, we adopt GPT-3.5 metric following common practices in [16, 19, 20, 21, 27, 29, 30, 37, 46, 49, 50]. With question, ground truth answer and the prediction generated by model, GPT-3.5 is able to judge whether this prediction is correct and provide a score between 0 and 5. We report the GPT-3.5 accuracy and score of each model on VQA datasets. For computational efficiency test, we report the average respond latency (from questioning to answering) and maximum video random-access memory (VRAM) of models.

#### 5.2 Zero-shot video question answering

As our model is only trained on [2, 15, 23, 30], we compare Flash-VStream with other competitive methods Video-ChatGPT[30], MovieChat[37], Chat-UniVi[16], Vista-LLaMA[29] and LLaMAVID[20] on zero-shot real-time VideoQA datasets in Table 1, and on normal zero-shot VideoQA datasets in Table 3. Video-ChatGPT uses temporal pooling and spatial pooling for video understanding. This simple method performs well in real-time movie understanding. MovieChat implements a merge-based memory consolidation and uses a Q-Former [18] as feature aggregator. Although it is competitive in understanding some short-video scenes, it falls behind in the domain of extremely long-video understanding, such as with RVS-Ego and RVS-Movie, as shown in Table 1. The newly proposed Chat-UniVi and LLaMA-VID have relative high performances on real-time video understanding benchmark. However, the high computation burden and high latency make it difficult to

Table 6: Comparison of different spatial and temporal size of STAR memory. A and S denote accuracy and score, respectively.

(a) Spatial Size

(b) Temporal Length

Method VS-ego VS-movie Pspa Ptem Pabs Pspa A S A S

16 4 1 16 55.7 3.9 52.1 3.3 4 4 1 4 58.4 4.0 53.1 3.4 8 8 1 8 58.2 3.9 53.4 3.4 8 1 1 8 56.4 4.0 51.9 3.4 8 4 4 8 57.2 4.0 54.8 3.5 8 4 1 8 59.0 3.9 56.1 3.4

Method VS-ego VS-movie Nspa Ntem Nabs Nret A S A S

1 32 32 3 55.3 3.9 54.9 3.4 1 16 16 3 57.2 3.9 54.2 3.4 1 8 8 3 56.7 3.9 53.2 3.4 1 25 25 3 59.0 3.9 56.1 3.4

deploy them for real-time understanding scenes. Flash-VStream achieves SoTA on these benchmarks, demonstrating the proposed STAR memory’s exceptional capabilities in information compression and long video comprehension.

#### 5.3 Computational efficiency

We measure the inference latency of each model by counting the respond wall time of the question handler process, as presented in Figure 2. For many models, the inference latency scales up with number of frames because their architectures demand processing all frames at once. Distinct from them, Flash-VStream leverages an efficient multiprocessing STAR memory mechanism (see Section 3.2) for streaming processing frames, which allows relative low inference latency and VRAM cost (detailed in Table 1). These attributes enable real-time inference.

#### 5.4 Ablation study

Effect of components of memory mechanism. We conduct an ablation study to evaluate the effects of key components of the STAR memory mechanism, i.e., spatial, temporal, abstract and retrieved memory. Removing temporal memory can cause a severe performance drop (as shown in the second row of Table 4), indicating that temporal memory is vital in long video stream understanding, as it enables the integration of contextual information across frames for coherent comprehension. Other types of memory also contribute a lot as they capture different aspect of visual information, such as spatial layout, high-level concepts and pivotal experiences.

Table 4: Ablation studies of STAR memory

Memory Type VS-ego VS-movie S T A R A S A S

✗ ✓ ✓ ✓ 57.3 3.9 54.2 3.4 ✓ ✗ ✓ ✗ 55.1 3.9 51.4 3.4 ✓ ✓ ✗ ✓ 57.0 4.0 54.1 3.4 ✓ ✓ ✓ ✗ 58.0 3.9 54.4 3.4 ✓ ✓ ✓ ✓ 59.0 3.9 56.1 3.4

Semantic Attention. We compare the proposed Semantic Attention with other memory updating strategies as shown in Table 5. Q-Former [17] is widely used by many models [20, 37, 49] and Sequential Q-Former is used by [29]. These updating methods are all transformer-based. Despite its lightweight nature, the Semantic Attention model outperforms other methods by a large margin. We suppose the reason is that the training dataset is too small for Q-Former based model to adequately learn. The architecture of Semantic Attention facilitates the extraction of key information and the selectively forgetting of irrelevant details, enhancing the model’s ability to comprehend abstract concepts in long videos.

Table 5: Semantic Attention v.s. other updating strategies

Abstract memory VS-ego VS-movie A S A S

Q-Former 57.1 3.9 50.4 3.3 Sequential Q-Former 56.0 3.9 51.4 3.3 Semantic Attention 59.0 3.9 56.1 3.4

Design of spatial size and temporal length of memory. In Table 6, we evaluate how spatial size and temporal length of memory influence long video understanding tasks. For spatial size of memory, although a smaller feature map is harmful to the performance, an excessively larger feature map is not an optimal choice either (see the first row of Table 6a). A similar pattern can be observed by varying temporal length of memory in Table 6b, in line with findings from [45]. Considering the expensive computational cost of larger and longer memory, we adopt a balanced design.

#### 5.5 Memory token visualization

||[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]|
|---|
|
|---|

||[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]|
|---|
|
|---|

- Figure 6: PCA Visualization of memory tokens. Red points represent memory tokens and blue points represent raw vision tokens from visual encoder. Left: an example from ActivityNet. Right: an example from Ego4D.

|··· ···<br><br>3min43s 5min20s 6min55s<br><br>|[Figure 20]|
|---|
<br><br>|[Figure 21]|
|---|
<br><br>|[Figure 22]|
|---|
<br><br>|[Figure 23]|
|---|
<br><br>|[Figure 24]|
|---|
<br><br>|[Figure 25]|
|---|
<br><br>|[Figure 26]|
|---|
<br><br>|[Figure 27]|
|---|
<br><br>|[Figure 28]|
|---|
|
|---|

Question: Is the confrontation with the policeman friendly? Question Time Stamp: 6min55s

GT Answer: No.

Answer: No, the confrontation with the policeman is not friendly. The man is seen pulling the woman over and pulling her out of the car. Guns are also seen from the video.

Flash-VStream (Ours) LLaMA-VID Answer: Yes, the confrontation with the policeman is friendly. The man in the video is seen talking to the policeman in a

friendly manner.(❌)

VideoChatGPT Answer: Yes, the confrontation with the policeman is friendly.(❌)

MovieChat Answer: In the video, we see a man in a yellow shirt and a woman in a yellow dress. The man is holding a cell phone and the woman

is holding a purse ... (❌)(Not answering the question).

- Figure 7: Comparison of different video LLMs on VStream-QA-Movie. Zoom in for a better view. In this video, a policeman pulls over a vehicle driven by a couple, but they point a gun at him and kill him. Our Flash-VStream is the only model that successfully understands the theme of this long movie clip.

We investigate the memory consolidation procedure in deep feature space. Specifically, in the left part of Figure 6, when inputting a video stream containing 3 significantly different scenes (talking, playing the drums and end credits), the memory will focus on the scene with the longest duration, just like what human will do in their minds. Relatively static scenes and relatively dynamic scenes are both given lots of attention, as shown in the right part of Figure 6. The visualization proves that memory tokens effectively reveal the distribution of the vision tokens.

#### 5.6 Case study

To better demonstrate the feature of VStream-QA as well as the effectiveness of Flash-VStream model, we hereby provide a case study on VStream-QA-Movie dataset. As shown in Figure 7, a question timestamp is equipped with each question-answer pair, indicating the time when the question is asked. Models are only provided with the visual content before the question timestamp. Thanks to the carefully designed STAR memory mechanism, our Flash-VStream grasp the key visual information and turns out to be the only model that successfully understands the theme of this long movie clip, while LLaMA-VID, VideoChatGPT and VStream-QA fail to do so for various reasons. This proves the effectiveness of our proposed Flash-VStream model in long video understanding tasks. Refer to model generated answers and the figure caption for details.

### 6 Conclusion

In conclusion, we have introduced Flash-VStream, a video-language model for real-time processing of online video streams and answering user questions. It incorporates a smartly designed memory called STAR, and significantly reduces inference latency and VRAM consumption. In addition, we have proposed a new benchmark for online video understanding called VStream-QA. Our model outperforms existing methods on this new online benchmark and maintains SoTA performance on offline video understanding benchmarks. We hope our work could inspire further research and advancements in the field of online video stream understanding.

### References

- [1] Alayrac, J.B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al.: Flamingo: a visual language model for few-shot learning. NeurIPS pp. 35,23716–23736 (2022)
- [2] Bain, M., Nagrani, A., Varol, G., Zisserman, A.: Frozen in time: A joint video and image encoder for end-to-end retrieval. In: ICCV. pp. 1728–1738 (2021)
- [3] Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al.: Language models are few-shot learners. NeurIPS pp. 33, 1877–1901 (2020)
- [4] Chang, Y.Y., Sun, F.Y., Wu, Y.H., Lin, S.D.: A memory-network based solution for multivariate time-series forecasting. arXiv preprint arXiv:1809.02105 (2018)
- [5] Chen, J., Li, K., Deng, Q., Li, K., Philip, S.Y.: Distributed deep learning model for intelligent video surveillance systems with edge computing. IEEE Transactions on Industrial Informatics

(2019)

- [6] Cheng, H.K., Schwing, A.G.: Xmem: Long-term video object segmentation with an atkinsonshiffrin memory model. In: ECCV. pp. 640–658. Springer (2022)
- [7] Chiang, W.L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J.E., Stoica, I., Xing, E.P.: Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. https://lmsys.org/blog/2023-03-30-vicuna/ (March 2023)
- [8] Daelemans, W., van den Bosch, A.: Memory-Based Language Processing. Studies in Natural Language Processing, Cambridge University Press (2005)
- [9] Dai, W., Li, J., Li, D., Tiong, A.M.H., Zhao, J., Wang, W., Li, B.A., Fung, P., Hoi, S.C.H.: Instructblip: Towards general-purpose vision-language models with instruction tuning. arXiv preprint arXiv:2305.06500 (2023)
- [10] Feigenbaum, E.A.: Information processing and memory. Models of human memory pp. 451–468

(1970)

- [11] Gao, P., Geng, S., Zhang, R., Ma, T., Fang, R., Zhang, Y., Li, H., Qiao, Y.: Clip-adapter: Better vision-language models with feature adapters. arXiv preprint arXiv:2110.04544 (2021)
- [12] Grauman, K., Westbury, A., Byrne, E., Chavis, Z., Furnari, A., Girdhar, R., Hamburger, J., Jiang, H., Liu, M., Liu, X., et al.: Ego4d: Around the world in 3,000 hours of egocentric video. In: CVPR. pp. 18995–19012 (2022)
- [13] Graves, A., Wayne, G., Danihelka, I.: Neural turing machines. arXiv preprint arXiv:1410.5401

(2014)

- [14] He, A., Luo, C., Tian, X., Zeng, W.: A twofold siamese network for real-time object tracking. In: CVPR. pp. 4834–4843 (2018)
- [15] Huang, Q., Xiong, Y., Rao, A., Wang, J., Lin, D.: Movienet: A holistic dataset for movie understanding. In: ECCV. pp. 709–727 (2020)
- [16] Jin, P., Takanobu, R., Zhang, C., Cao, X., Yuan, L.: Chat-univi: Unified visual representation empowers large language models with image and video understanding. arXiv preprint arXiv:2311.08046 (2023)
- [17] Li, J., Li, D., Savarese, S., Hoi, S.: Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597 (2023)
- [18] Li, J., Li, D., Xiong, C., Hoi, S.: Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In: ICML. pp. 12888–12900 (2022)
- [19] Li, K., He, Y., Wang, Y., Li, Y., Wang, W., Luo, P., Wang, Y., Wang, L., Qiao, Y.: Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355 (2023)
- [20] Li, Y., Wang, C., Jia, J.: Llama-vid: An image is worth 2 tokens in large language models. arXiv preprint arXiv:2311.17043 (2023)
- [21] Lin, B., Zhu, B., Ye, Y., Ning, M., Jin, P., Yuan, L.: Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122 (2023)
- [22] Liu, H., Li, C., Li, Y., Lee, Y.J.: Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744 (2023)

- [23] Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. arXiv preprint arXiv:2304.08485

(2023)

- [24] Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. arXiv preprint arXiv:2304.08485

(2023)

- [25] Liu, Y., Yu, R., Yin, F., Zhao, X., Zhao, W., Xia, W., Yang, Y.: Learning quality-aware dynamic memory for video object segmentation. In: ECCV. pp. 468–486 (2022)
- [26] Liu, Y., Zhang, C., Wang, Y., Wang, J., Yang, Y., Tang, Y.: Universal segmentation at arbitrary granularity with language instruction. arXiv preprint arXiv:2312.01623 (2023)
- [27] Luo, R., Zhao, Z., Yang, M., Dong, J., Qiu, M., Lu, P., Wang, T., Wei, Z.: Valley: Video assistant with large language model enhanced ability. arXiv preprint arXiv:2306.07207 (2023)
- [28] Luvizon, D.C., Picard, D., Tabia, H.: Multi-task deep learning for real-time 3d human pose estimation and action recognition. IEEE TPAMI 43(8), 2752–2764 (2020)
- [29] Ma, F., Jin, X., Wang, H., Xian, Y., Feng, J., Yang, Y.: Vista-llama: Reliable video narrator via equal distance to visual tokens. arXiv preprint arXiv:2312.08870 (2023)
- [30] Maaz, M., Rasheed, H., Khan, S., Khan, F.S.: Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424 (2023)
- [31] Mangalam, K., Akshulakov, R., Malik, J.: Egoschema: A diagnostic benchmark for very long-form video language understanding. NeurIPS (2024)
- [32] Muhammad, K., Hussain, T., Del Ser, J., Palade, V., De Albuquerque, V.H.C.: Deepres: A deep learning-based video summarization strategy for resource-constrained industrial surveillance scenarios. IEEE Transactions on Industrial Informatics 16(9), 5938–5947 (2019)
- [33] OpenAI: Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023)
- [34] Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al.: Training language models to follow instructions with human feedback. NeurIPS pp. 27730–27744 (2022)
- [35] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: ICML. pp. 8748–8763 (2021)
- [36] Sermanet, P., Ding, T., Zhao, J., Xia, F., Dwibedi, D., Gopalakrishnan, K., Chan, C., DulacArnold, G., Maddineni, S., Joshi, N.J., et al.: Robovqa: Multimodal long-horizon reasoning for robotics. arXiv preprint arXiv:2311.00899 (2023)
- [37] Song, E., Chai, W., Wang, G., Zhang, Y., Zhou, H., Wu, F., Guo, X., Ye, T., Lu, Y., Hwang, J.N., et al.: Moviechat: From dense token to sparse memory for long video understanding. arXiv preprint arXiv:2307.16449 (2023)
- [38] Supancic III, J., Ramanan, D.: Tracking as online decision-making: Learning a policy from streaming videos with reinforcement learning. In: ICCV. pp. 322–331 (2017)
- [39] Tan, Q., Zhang, J., Liu, N., Huang, X., Yang, H., Zhou, J., Hu, X.: Dynamic memory based attention network for sequential recommendation. In: AAAI. pp. 4384–4392 (2021)
- [40] Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E., Lample, G.: Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 (2023)
- [41] Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al.: Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 (2023)
- [42] Wang, Z., Zheng, L., Liu, Y., Li, Y., Wang, S.: Towards real-time multi-object tracking. In: ECCV. pp. 107–122 (2020)
- [43] Xiao, J., Shang, X., Yao, A., Chua, T.S.: Next-qa: Next phase of question-answering to explaining temporal actions. In: CVPR. pp. 9777–9786 (2021)
- [44] Xu, D., Zhao, Z., Xiao, J., Wu, F., Zhang, H., He, X., Zhuang, Y.: Video question answering via gradually refined attention over appearance and motion. In: ACM MM. pp. 1645–1653 (2017)
- [45] Xu, L., Zhao, Y., Zhou, D., Lin, Z., Ng, S.K., Feng, J.: Pllava: Parameter-free llava extension from images to videos for video dense captioning. arXiv preprint arXiv:2404.16994 (2024)

- [46] Yang, A., Miech, A., Sivic, J., Laptev, I., Schmid, C.: Zero-shot video question answering via frozen bidirectional language models. NeurIPS 35, 124–141 (2022)
- [47] Yu, Z., Xu, D., Yu, J., Yu, T., Zhao, Z., Zhuang, Y., Tao, D.: Activitynet-qa: A dataset for understanding complex web videos via question answering. In: AAAI. pp. 9127–9134 (2019)
- [48] Zhang, B., Wang, L., Wang, Z., Qiao, Y., Wang, H.: Real-time action recognition with enhanced motion vector cnns. In: CVPR. pp. 2718–2726 (2016)
- [49] Zhang, H., Li, X., Bing, L.: Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858 (2023)
- [50] Zhang, R., Han, J., Zhou, A., Hu, X., Yan, S., Lu, P., Li, H., Gao, P., Qiao, Y.: Llama-adapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint arXiv:2303.16199

(2023)

Appendix

- A Memory implementation details

This section describes the details of the proposed Spatial-Temporal-Abstract-Retrieved memory mechanism in Section 3.2. The STAR memory has both parametric and non-parametric updating strategies. Spatial memory uses simple replacing method.

As shown in Algorithm 1, temporal memory performs a Weighted K-means Clustering Algorithm temporal-wise to condense (Ntem + 1) × Ptem2 tokens to Ntem × Ptem2 tokens. Each frame feature in temporal memory Mtem(i) = ci ∈ RP

2

tem represents the centroid of the i-th feature cluster.

- Algorithm 1 Weighted K-means Clustering Algorithm

Require: Current temporal memory Mtem = {Mtem1 , Mtem2 , . . . , MtemNtem} Require: Newest frame feature e

Require: Set of all data points X = {Mtem1 , Mtem2 , . . . , MtemNtem, e} Require: Maximum number of iterations T

Require: Weights vector of points w = {w1, w2, . . . , wNtem, 1}

- 1: procedure WEIGHTED K-MEANS(X, k, T, w)
- 2: Number of clusters k ← Ntem
- 3: Initialize t ← 0
- 4: Randomly initialize cluster centroids C = {c1, c2, . . . , ck} from the data points X
- 5: Initialize previous cluster assignment Pj ← {}
- 6: Initialize current cluster assignment Sj ← {}
- 7: while t < T do
- 8: for xi ∈ X do
- 9: j ← argminj∥xi − cj∥2
- 10: append (Sj, xi)
- 11: end for
- 12: if S == P then
- 13: break
- 14: end if
- 15: for j = 1, 2, . . . , k do
- 16: cj ← xi∈Sj

wi·xi

xi∈Sj

wi

- 17: end for
- 18: w ← UpdateWeights(S) ▷ Update the weights vector based on the current cluster assignment
- 19: P ← S
- 20: Clear S
- 21: t ← t + 1
- 22: end whileMtem ← C
- 23: return Mtem, w
- 24: end procedure

- Algorithm 2 Semantic Attention

Require: Current abstract memory Mabs = {Mabs1 ,Mabs2 ,...,MN

abs

abs }

Require: Newest frame features e Require: Memory decay factor α ∈ (0,1)

- 1: procedure SEMANTIC ATTENTION(Mabs,e,α)
- 2: K ← fk_proj(e)
- 3: Q ← fq_proj(Mabs)
- 4: W ← QKT
- 5: W ← Softmax(W,dim = 1)
- 6: Mabs ← (1 − α)Mabs + We
- 7: return Mabs
- 8: end procedure

For abstract memory, we design a learning-based Semantic Attention model for information integration and selective forgetting. Algorithm 2 describes the detailed forward procedure of Semantic

abs×Pabs2 with newest features e ∈ Rn×P

Attention model. In order to update abstract memory Mabs ∈ RN

2

abs(n is 1 by default), we first calculated the attention weight between newest features and current abstract memory. Then a softmax layer is applied to normalize the contribution of new features. Finally, the abstract memory is updated by a momentum updating mechanism with decay factor α.

- Algorithm 3 Key Feature Retrieval

Require: Current feature buffer Mbuff = {Mbuff1 ,Mbuff2 ,...} Require: Current temporal memory Mtem = {Mtem1 ,Mtem2 ,...,MN

tem

tem } Require: Weights vector of points w = {w1,w2,...,wN

tem}

- 1: procedure KEY FEATURE RETRIEVAL(Mbuff,Mtem,w,Nret)
- 2: k ← Nret
- 3: j1,j2,...,jk ← top-kj wj
- 4: Mret ← {}
- 5: for z = 1,2,...,k do
- 6: ekey ← min_item∥gc(ekey,Pspa) − Mj

z

tem∥2 for ekey ∈ Mbuff

- 7: append (Mret,ekey)
- 8: end for
- 9: return Mret
- 10: end procedure

For retrieved memory, we use a key feature retrieval Algorithm 3 to calculate the current retrieved memory Mret ∈ RN

ret×Pspa2 . Because retrieved memory and spatial memory are both renewed from the feature buffer Mbuff, we set their spatial sizes to the same. Here wj is equal to the size of j-th cluster, i.e., the number of tokens in this cluster. Therefore, we choose the centroids of the top-k large clusters as pivots. The features nearest to these centroids are considered as key features, which are added to the retrieved memory.

### B Training details

Table 7: Training settings of Flash-VStream

Settings Stage-1 Stage-2

Batch size 256 128 Learning rate 1e-3 2e-5 Learning schedule Cosine decay Warmup ratio 0.03 Weight decay 0 Epoch 1 Optimizer AdamW DeepSpeed stage 0 1 Visual encoder Freeze Semantic attention Open Projector Open LLM Freeze Open

The training procedure of Flash-VStream is similar to that of [23] [20]. In the modality alignment stage (stage 1), we train the Semantic attention model and the projector for one epoch. In the instruction tuning stage (Stage 2), we fine-tune the Semantic attention model, the projector and the LLM for another epoch. The overall training can be finished in 15 hours on 8 A100 80G GPUs (BFloat16) with extracted visual features. Detailed training settings are shown in Table 7.

- C VStream-QA benchmark design details Here we provide more details of VStream-QA online video understanding benchmark.

- C.1 Data generation pipeline in detail

- • Video Selection. We first select 10 videos from Ego4D dataset [12] with each video being 1 hour long, and 22 videos from MovieNet dataset [15] with each video being 30 minutes long. Both Ego-centric videos and movie clips are chosen to cover a wide range of content types. Refer to next subsection for details.
- • Dense Captioning. We use GPT-4V [33] to generate dense captions for each video clip. Long videos are divided into pieces of 30 seconds, and 8 frames are sparsely sampled from each piece as input to GPT-4V. Each output caption describes the content of the 30-second video piece, and marked with a specific timestamp.
- • Summary Generation. We use GPT-4 to deduplicate and summarize the dense captions generated by GPT-4V. The summary is designed to be a concise description scene-level clip, typically originated from multiple dense captions that correspond to several minutes of video content. Timestamps are carefully kept throughout the summarization process.
- • Question-Answer Generation. We use GPT-4 to generate 5 types of QA pair based on the scene summary. Each QA is generated from a single or several consecutive scene summaries, to ensure that the QA is only related to the visual information before the timestamp.
- • Human Filtering. Volunteers are invited to judge the relevance of the generated QA pairs to the video content. The following types of QA pairs are carefully filtered out: i) questions are irrelevant with the video or ambiguous, ii) questions require additional knowledge beyond the video, iii) questions are able to answered without the video, iv) answers are wrong or ambiguous. repetitive.

- C.2 Variety of video content

Besides the variety of question types, VStream-QA benchmark also involves various type of video content.

- • VStream-QA-Ego video topics: [’cooking’, ’playing-card’, ’writing’, ’home-maintenance’, ’sightseeing’, ’reading’].
- • VStream-QA-Movie movie genres: ["Action", "Adventure", "Sci-Fi", "Crime", "Drama", "Thriller", "War", "Mystery", "Comedy", "Fantasy", "History", "Biography", "Horror"].

- D Limitations

#### D.1 Representativeness of VStream-QA benchmark

Although the proposed VStream-QA is the first benchmark that aims to simulate real-world video streaming scenarios, it still falls short in fully representing the scenario of comprehending infinitely long video streams in the real world. Besides, the proposed approach only involves the coarsegrained understanding task, i.e., QA. In the real world, video streams encompass more complex comprehension tasks. It is our aspiration that the Flash-VStream could inspire related research in this field.

#### D.2 GPT-3.5-based evaluation metric

In the proposed VStream-QA benchmark and many other video question answering benchmarks, GPT-3.5 based evaluation is adopted as the preferred metric. However, we notice that there is always a discrepancy between the distribution of GPT accuracy and GPT score. Specifically, for answers classified as “no”, many of them are assigned with a high score like “4” or “5”, also discussed by [37]. This abnormal phenomenon reduces the credibility of this “0 ∼ 5 score” metric in GPT-3.5-based MLLM evaluation.

### E Broader Impacts

Real-time understanding models for long video streams may lead to potential negative societal impacts, including but not limited to unauthorized surveillance or privacy-infringing tracking. However, we firmly believe that the task itself is neutral with positive applications, such as health monitoring and emergency response.

