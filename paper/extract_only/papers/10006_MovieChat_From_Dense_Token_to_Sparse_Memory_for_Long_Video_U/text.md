## MovieChat: From Dense Token to Sparse Memory for Long Video Understanding

# arXiv:2307.16449v4[cs.CV]9Mar2024

Enxin Song1,* Wenhao Chai2,∗,† Guanhong Wang1,∗ Yucheng Zhang1,‡ Haoyang Zhou1,‡ Feiyang Wu1,‡ Haozhe Chi1 Xun Guo3 Tian Ye4 Yanting Zhang5 Yan Lu3 Jenq-Neng Hwang2 Gaoang Wang1, 1 Zhejiang University 2 University of Washington 3 Microsoft Research Asia 4 Hong Kong University of Science and Technology (GZ) 5 Donghua University

https://rese1f.github.io/MovieChat

###### Abstract

Recently, integrating video foundation models and large language models to build a video understanding system can overcome the limitations of specific pre-defined vision tasks. Yet, existing systems can only handle videos with very few frames. For long videos, the computation complexity, memory cost, and long-term temporal connection impose additional challenges. Taking advantage of the AtkinsonShiffrin memory model, with tokens in Transformers being employed as the carriers of memory in combination with our specially designed memory mechanism, we propose the MovieChat to overcome these challenges. MovieChat achieves state-of-the-art performance in long video understanding, along with the released MovieChat-1K benchmark with 1K long video and 14K manual annotations for validation of the effectiveness of our method.

###### 1. Introduction

Recent advances in Large Language Models (LLMs) [13,19,47,62,64] acheive great success in Natural Language Processing (NLP) . It is a natural progression to introduce multi-modality [16] into LLMs and turn it into Multi-modal Large Language Models (MLLMs), which is able to conduct multimodal rationalization and understanding. MLLMs have shown incredible emergent capabilities in various multimodal tasks such as perception (e.g., existence, count, position, OCR) [1,33,34,43,69,85], commonsense reasoning [26,28,33,34,36,43,61,85], and code reasoning [21,24,26,39,41,78], resulting in a potential path to Artificial General Intelligence (AGI). Compared to LLMs and other task-specific models, MLLMs provide a

*Equal contribution, † Project lead, ‡ Data collection, Corresponding Author.

|17.2GB<br><br>24.0GB<br><br>10<br><br>15<br><br>20<br><br>25<br><br>30<br><br>35<br><br>40<br><br>1 10 100 1000 10000<br><br>MovieChat (21.3KB/f) Video-LLaMA (187MB/f)<br><br>VideoChat (227MB/f) VideoChatGPT (25.6MB/f)<br><br>|
|---|

Figure 1. VRAM cost under gigabyte (GB) (y-axis) v.s. frame number (x-axis) comparison. We test the visual-only inference of all methods at a resolution of 224 × 224 without frame sampling. While the previous method can only support around 100 frames of inference, MovieChat can handle videos with >10K frames on a 24GB graphics card. MovieChat has a 10000× advantage over other methods in terms of the average increase in VRAM cost per frame (21.3KB to ∼ 200MB per frame).

more human-like interpretation of the scenarios, a userfriendly interface for interaction, and a broader range of capabilities.

Existing vision-centric MLLMs follow the paradigm that utilizing pre-trained LLMs and visual encoder with additional learnable modules (Q-former [21,34,36,82] or simple projection layer [22, 39, 43, 61]). In video field, some previous works [43,82] follow this paradigm to build video MLLMs, while works in the other paradigm [37,67] combine existing visual perception tools (e.g., tracking and classification) and LLMs through Application Programming In-

terface (API) to build a system without training. Yet, previously, there is no exploration of a model or system based on long videos (over one minute), and there is also a lack of a standardized benchmark to evaluate the capabilities of these systems.

In this paper, we present MovieChat, a novel framework that integrates vision models and LLMs to conduct long video understanding tasks. We claim that the computation complexity, memory cost, and long-term temporal connection are the main challenges for long video understanding. Atkinson-Shiffrin memory model [5] proposes that shortterm memory functions as a buffer of long-term memory, serving as a processor for the encoding of information into long-term memory. Inspired by this, we propose a memory mechanism to deal with long video understanding tasks, which includes a rapidly updated short-term memory and a compact thus sustained long-term memory. We use a sliding window approach to extract video features and represent them in token form, which are then sequentially fed into the short-term memory frame by frame. The shortterm memory has a fixed length, and when it reaches its set limit, the earliest tokens are popped and consolidated into the long-term memory. After passing through a projection layer, the video representation is inputted into a large language model for interaction with the user. As shown in Fig. 1, our proposed MovieChat mechanism outperforms other existing methods in terms of Video Random Access Memory (VRAM) cost. We also release a new benchmark, MovieChat-1K, with 1K long videos and 13K manual question-answering pairs for validation of the effectiveness of our proposed MovieChat.

The contributions of this work are summarized as:

- • We present MovieChat, a novel framework that integrates vision models and LLMs, which is the first to support long video (>10K frames) understanding tasks.
- • We propose an effective memory management mechanism to reduce the computation complexity and memory cost, while enhancing the long-term connection.
- • We release the first long video understanding benchmark, MovieChat-1K, with manual annotations and conduct extensive quantitative evaluation and case studies to evaluate the comparable performance of both understanding capability and inference cost.

###### 2. Related Works

###### 2.1. Multi-modal Large Language Models

LLMs [13,19,47,62,64,65] have achieved great success in natural language processing (NLP) tasks recently. Many works try to build MLLMs [1,21,26,28,33,34,43,69,78,85]

by combining models of other modalities. Flamingo [1] bridges powerful pre-trained vision-only and language-only models and achieves state-of-the-art performance with fewshot learning. BLIP-2 [34] proposes a generic and efficient pre-training strategy that bootstraps vision-language pre-training from an off-the-shelf frozen pre-trained image encoders and a frozen large language model. MiniGPT4 [85] also aligns a frozen visual encoder with a frozen LLM, Vicuna [19], using just one projection layer to realize the system. Otter [33] showcases improved instructionfollowing ability and in-context learning. In video field, ChatVideo [67] treats tracklets as the basic video unit and allows users’ interacting with the LLMs. VideoChat [37] integrates video foundation models and LLMs via a learnable neural interface, excelling in spatiotemporal reasoning, event localization, and causal relationship inference. VideoLLaMA [82] further leverages pre-trained models ImageBind [27] and LLaMA [64], bootstraping cross-modal training in videos following BLIP-2. Yet, these methods fail to handle long video understanding because of high computation complexity, large memory cost, and weak long-term temporal connection. Therefore, our main effort is to introduce an effective memory mechanism to overcome these challenges.

###### 2.2. Long Video Understanding

Understanding long videos is a challenging task in computer vision. Prior arts use 3D CNN for long-term feature bank [70], object/human-centric motion [50, 71], or other forms [54, 72] as video representations. MIST [25] decomposes dense self-attention into a cascade segment and region selection module to increase the computation efficiency for understanding minutes of long videos. Building long-form video understanding datasets is challenging and rarely explored. [57] captures large scale data from Kinetics-400 [15], but only for generic event boundary detection tasks. [58] creates a language grounding benchmark from audio descriptions of movies, but it lacks long-term understanding evaluation. [63] successfully builds a benchmark contains multiple sources of information (e.g., video clips, plots, and DVS) for question-answering tasks in the movie field. There are also several datasets of video-caption/description pairs among various domains, such as cooking (e.g., MPII Cooking [51–53] and TACoS [48, 49]), instruction (e.g., HowTo100M [45] and HiREST [80]), Ego [44], and movie (e.g., MovieQA [63] and MovieNet [31]) from different sources such as YouTube [17,45,81], Twitter [6–9], and Internet [10]. Yet, those datasets lack diverse and fine-grained dense captioning for long videos.

Memory Consolidation Long-term Memory 𝓛

② clear 𝓢 and initialization

###### …

###### …

- Step 1: build adjacent frame pairs
- Step 2: calculate cosine similarity …

- Step 3: select top-1 pair and merge

| | |
|---|---|
|𝐵𝑛| |
| | |

|𝑥2|
|---|

|𝑥𝐾−2|
|---|

|𝑥𝐾−1|
|---|

𝑥1

…

Current 𝓢

##### … 𝑥

|𝑥3|
|---|

|𝑥𝐾−1|
|---|

|𝑥2|
|---|

|𝑥𝐾|
|---|

𝑥1 𝐾

Short-term Memory 𝓢

###### …

𝐵1 𝐵2 𝐵𝐺−1 𝐵𝐺

Current Frame

𝑥𝑡

① apply

window-level

consolidation

0.52 0.94 0.81 0.88

if full

Extend Positional Encoding

Q-Former

frame-level Feature

###### …

…

|𝑥𝐶−1|
|---|

|𝑥𝐶|
|---|

|𝑥2|
|---|

𝑥1 𝑥2

𝑥2′

|𝑥3|
|---|

Visual Feature Extractor

𝑄uestion 𝓟 ProjectionLayer

weighted sum

repeat

𝓥

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Large Language Model

𝐴nswer

Global Mode Breakpoint Mode

Non-overlap Sliding Window

- Figure 2. Illustration of MovieChat. MovieChat extracts video features with a sliding window and represents them in token form, which are then sequentially fed into the short-term memory frame by frame. When the fixed-length short-term memory reaches its preset limit, the earliest tokens are popped and consolidated into the long-term memory. MovieChat incorporates two distinct inference modes: the global mode, which exclusively utilizes the long-term memory, and the breakpoint mode, which additionally incorporates the current short-term memory as part of the video representation. The breakpoint mode allows for understanding the video at a specific moment in time. After passing through a projection layer, the video representation is inputted into a large language model for interaction with the user.

- 2.3. Memory Models in Vision Tasks

There are some prior works exploring memory models [59] in various vision tasks in videos, such as video object segmentation (VOS) [18,30,55,56], multi-object tracking (MOT) [2,14,29,73], visual object tracking (VOT) [38,

- 42, 77, 84], and action understanding [68]. MeMOT [14] builds a large spatiotemporal memory that stores the past observations of the tracked objects. XMem [18] develops an architecture that incorporates multiple independent yet deeply-connected feature memory storage to handle long videos with thousands of frames. We learn from the experience of those prior arts and further adopt an effective memory mechanism in combination with LLMs.

Our method focuses on reducing the redundancy of visual tokens in the video and building a memory mechanism to pass the information among a large temporal range.

- 3. MovieChat

- 3.1. Overview

Our proposed method, MovieChat, comprises several key components, including the frame-wise visual feature

extractor, the short-term and long-term memory modules, the video projection layer, and the Large Language Model (LLM), as illustrated in Fig. 2. MovieChat is designed for ultra-long videos (>10K frames) understanding through interactive dialogue with the user. To address the impractical storage demands of concurrently storing a vast number of frames in both GPU memory and RAM, we employ a sliding window approach to efficiently process the video. The short-term memory module embeds dense tokens with sliding window and the long-term memory module periodically updates. MovieChat supports two inference modes: Breakpoint mode is used to understand a specific moment in the video, providing insights and answers based on that particular frame or scene; Global mode, on the other hand, is employed to comprehend the entire video as a whole, enabling a comprehensive understanding of the overall content and context.

###### 3.2. Visual Feature Extraction

For visual feature extraction, instead of utilizing videobased foundational models such as ViViT [4] or VideoSwin [40], we simply use an image-based model to get

frame-wise feature in the form of tokens. To be specific, we utilize pre-trained models as our visual feature extractor, including the ViT-G/14 from EVA-CLIP [23] and the Q-former from BLIP-2 [35]. This is mainly because 1) there is few video foundation model that makes good alignment with text, and 2) our proposed memory mechanism can effectively capture temporal features. Given a raw video, the visual input v ∈ ZT×3×H×W is a sequence of T RGB frames of size H × W sampled from the video. The visual features are extracted in a sliding window manner, which could be formulated as

T C ⌉, (1)

Bn = {xi = V(vi) | ∀i = 1,...,C},n = 1,...,⌈

where Bn is the n-th video clip feature within the sliding window spanning C frames. V(·) is the visual feature extractor, taking as input a single frame vi ∈ Z3×H×W. xi ∈ RN×D denotes N extracted visual tokens with respect to each frame, and D is the feature dimension of each token.

###### 3.3. Short-term Memory

Short-term memory stores the frame tokens in a temporary fixed-length buffer. The previously extracted visual features by sliding window G times without further processing are used to construct short-term memory, which can be formulated by:

S =

Bn = {xi | ∀i = 1,...,K},n = 1,..,G, (2)

n

where S is short-term memory, and K is equal to C × G. Note that we set short-term memory to contain a fixed length of K frames since the role of short-term memory is to assist in video understanding based on previous shortterm contextual information.

The update strategy for short-term memory is based on the First-in-First-out (FIFO) queue. As a new batch of visual tokens enters, when the short-term memory reaches its capacity, we pop the currently stored frames to the memory consolidation module and clear the short-term memory. The output video feature obtained from the consolidation module augments the long-term memory; on the other hand, it reinitializes the short-term memory with this feature. The initialization aims at communicating the information between different sliding windows, thereby achieving more efficient compression.

###### 3.4. Long-term Memory

Long-term memory can effectively avoid the problem of catastrophic knowledge forgetting, which is crucial for handling long video understanding tasks. The features stored in short-term memory are dense tokens, but due to the limitations of GPU memory and computation cost, storing all the tokens dropped from short-term memory into long-term

Algorithm 1 Memory consolidation Require: S ▷ short-term memory

- 1: while len(S)>RL do ▷ iterative merge
- 2: for xi in S do
- 3: s ← sim(xi,xi+1) ▷ tokens similarity
- 4: end for
- 5: m ← max(s) ▷ the maximum value index
- 6: xm ← merge(xm,xm+1) ▷ merge
- 7: del xm+1
- 8: end while

memory buffer in sequence is infeasible. Besides, we observe significant temporal redundancy in videos, where activities span multiple frames with minimal visual changes. To this end, we propose a method to merge adjacent similar frames to simplify video feature representation and accelerate video encoding. This method transforms the dense tokens to the sparse memories, which are stored in long-term memory.

To be specific, as shown in Algorithm 1, we conduct memory consolidation by merging the most similar tokens in the adjacent frames following ToMe [12] periodically. We find that the token embedding in Transformer already summarize the information of each frame for using in calculating the average cosine similarity s of N embedded tokens:

N

1 N

cos(xji,xji+1) . (3)

s =

j=1

Our goal is to keep RL frames after every merge operation, which also embeds rich information stored in the long-term memory. RL is the hyper-parameter to control the trade-offs between performance and efficiency. Therefore, we greedily merge each set of adjacent frames with the highest similarity via weighted averaging. The merge operation is iteratively conducted until the token count reaches the predefined value set RL for each consolidation operation, resulting in the output video feature v′ ∈ ZR

L×3×H×W (operational details in appendix). The above algorithm is parameter-free, and can be easily plugged into a frame-based video encoder. Although the frame similarity calculation brings additional computing overhead, it is negligible compared to the efficiency gained by reducing stored frames.

Extend positional encoding. For long-term memory, the number of tokens exceeds the maximum length of the positional encoding from the pre-trained model. Thus, our model utilizes the positional encoding mechanism following BERT [32], which results in a portion exceeding the length threshold n without available positional encoding. In order to handle long enough long memory, we adopt the hi-

[Figure 7]

Does/Do…? Is/Are…?

How does/do…? How is/are …?

[Figure 8]

[Figure 9]

25.73%

2.29%

11k-12k frm

< 10k frm

Action Film

Animation Film

How many…?

8.6%

Who…?

59.3%

6.70%

17.00%

13.83%

0.26%

Detective Film

Which…?

15.10%

0.11%

Family Film

> 12k frm

Where…?

[Figure 10]

10k-11k frm

- 4.90%

15.16%

14.6%

31.2%

Documentary Film

Epic Film

What…?

When…?

21.80%

11.40%

37.04%

5.58%

(a) Category.

(b) Video length.

(c) Question type.

- Figure 3. Video-text statistics in MovieChat-1K. It encompasses a diverse set of categories, gathered from multiple question types and containing a diverse distribution of clip durations. We annotate the video categories that account for more than 4.5% of the total (the complete list of video categories and their percentages in Appendix B). “frm” represents the number of video frames.

[Figure 11]

Figure 4. Word Cloud of the answer set in MovieChat-1K.

erarchically decomposed positional encoding method proposed by Su et al. [60], which allows to extend the absolute positional encoding of length from n to n2.

###### 3.5. Inference

Previous methods always use the representation of the whole video to conduct understanding and questionanswering, which may fail in localizing specific moment especially in long videos. To this end, we propose two inference modes, global and breakpoint, for long video understanding task as follows.

Global mode. Global mode is defined as the understanding and question-answering for the whole video. In this case, we only use long-term memory L as the video representation V.

Breakpoint mode. Breakpoint mode is distinctly defined as understanding specific moments in a video. Since events inherently possess continuity, we need to consider not only the information directly related to the moments stored in short-term memory S but also the information indirectly related stored in long-term memory L. Based on this, we hypothesize that when querying the movie at a specific moment t, the video representation V should be the aggrega-

tion of L, S, and the current video frame feature xt. We find that simply concatenating these items yields excellent performance and leave further exploration of additional aggregation choices for future work.

Subsequently, the video representation V goes through a Q-former and a linear projection layer before being fed into the LLM O, which can be formulated as:

###### A = O(Q,P(V)), (4)

where P is the projection from visual space to text space. A represents the answer or instruction, and Q is employed to denote the question, respectively.

###### 4. A New Benchmark: MovieChat-1K

Previous works on building long video understanding benchmarks either focus on non-question-answering tasks (e.g., language grounding [58], generic event boundary detection [57], user engagement and movie metadata prediction [71], etc.) or lack long-form understanding evaluation [31]. To better evaluate the performance of MovieChat, we collect a new benchmark for long video understanding tasks, MovieChat-1K, which contains 1K high quality video clips sourced from various movies and TV series with 14K manual annotations.

As shown in Fig. 3a, we collect videos from 15 popular categories with varying distribution, including documentary film, detective film, animation film, and so on. Among these, each video comprises multiple alternating scenes, contributing to a diverse and dynamic visual narrative within the context of the collection. The visual representation in Fig. 3b demonstrates the clip duration distribution of MovieChat-1K. Over 90% of the videos exhibit a duration ranging from 10K to 12K frames, while 14.6% of videos extending beyond 12K frames. Only 8.6% of videos have duration less than 10k frames.

For each video, we manually set and provide 1 dense caption for the whole video, 3 question-answering pairs for global mode and 10 question-answering pairs with times-

MSVD-QA MSRVTT-QA ActivityNet-QA

Method

Accuracy Score Accuracy Score Accuracy Score FrozenBiLM [76] 32.2 – 16.8 – 24.7 – Video Chat [37] 56.3 2.8 45.0 2.5 26.5 2.2 LLaMA Adapter [83] 54.9 3.1 43.8 2.7 34.2 2.7 Video LLaMA [82] 51.6 2.5 29.6 1.8 12.4 1.1 Video-ChatGPT [43] 64.9 3.3 49.3 2.8 35.2 2.7 MovieChat (Ours) 75.2 3.8 52.7 2.6 45.7 3.4

- Table 1. Quantitative evaluation for short video question answering with GPT-3.5 [46]. MovieChat achieves comparable performance even it is not specifically designed for for short video question-answering tasks. The best result is highlighted in bold, and the second best is underlined.

Method CI DO CU TU CO Video Chat [37] 2.23 2.50 2.53 1.94 2.24 LLaMA Adapter [83] 2.03 2.32 2.30 1.98 2.15 Video LLaMA [82] 1.96 2.18 2.16 1.82 1.79 Video-ChatGPT [43] 2.40 2.52 2.62 1.98 2.37 MovieChat (Ours) 2.76 2.93 3.01 2.24 2.42

- Table 2. Quantitative evaluation for short video generation performance with GPT-3.5 [46]. CI stands for correctness of information, DO stands for detail orientation, CU stands for contextual understanding, TU stands for temporal understanding, and CO stands for consistency. The best result is highlighted in bold, and the second best is underlined.

tamps for breakpoint mode. Fig. 3c illustrates the distribution of question types in MovieChat-1K. Note that MovieChat-1K is specifically designed for long video comprehension tasks, the majority of questions are open-ended, with only a quarter classified as multiple-choice questions, marked by initiators such as ‘Do,’ ‘Does,’ ‘Is,’ or ‘Are.’ We also compute the word distributions of our provided question-answer pairs. As illustrated in Fig. 4, which includes common objects (people, clothes, etc.), time (day, night, etc.), scenes (indoor, outdoor, etc.), and so on. More statistics information can be found in appendix.

###### 5. Experiments

We conduct quantitative and qualitative evaluations between MovieChat and previous methods. Additionally, we perform ablation studies to investigate MovieChat. Experimental settings and analyses can be found in appendix.

###### 5.1. Quantitative Evaluation

Short video question-answering. We use several widely used open-ended datasets: MSVD-QA [74], MSRVTTQA [75], and ActivityNet-QA [79] for short video questionanswering tasks. The evaluation process is under the assistance of LLM with the default hyper-parameter settings. The accuracy and relative scores on a scale of 0 to 5 are

Global Mode Breakpoint Mode Accuracy Score Accuracy Score

Method # Frames

Video Chat [37] 32 57.8 3.00 46.1 2.29 Video LLaMA [82] 32 51.7 2.67 39.1 2.04 Video-ChatGPT [43] 100 47.6 2.55 48.0 2.45

MovieChat (ours) 2048 62.3 3.23 48.3 2.57

Table 3. Quantitative evaluation for long video question answering on MovieChat-1K test set in global mode with the average of GPT-3.5 [46], Claude [3] and human bling rating. HBR stands for human blind rating. The best result is highlighted in bold, and the second best is underlined.

reported. Compared to previous methods [37, 43, 82, 83], MovieChat achieves comparable performance even it is not specifically designed for short video question-answering tasks, as shown in Tab. 1.

Short video generative performance. Following [43], we employ GPT-assisted evaluation to conduct a more comprehensive comparison of the text generation performance between MovieChat and previous methods [37, 43, 76] on processed ActivityNet-QA [79]. The evaluation pipeline covers crucial metrics (including Correctness of Information, Detailed Orientation, Contextual Understanding, Temporal Understanding and Consistency) and assigns relative scores to the generated predictions on a scale of 1-5. We present the results of the generation performance evaluation in Tab. 2. The results reveal its competitive performance across all key aspects compared to previous methods.

Long video question-answering. We evaluate the long video question-answering performance of MovieChat with our proposed MovieChat-1K. We split 1,000 videos into training set (800), test set (100), validation set (100) and only use test set for final performance evaluation. We select three recent LLM-based video understanding models (e.g. Video Chat [37], Video LLaMA [82], and VideoChatGPT [43]) as the baselines. Yet, none of those methods can support such long video (>10K frames). Therefore, to accommodate their length limitations in global questions, we uniformly sample from the original video up to the maximum frame count which can be officially supported by each individual model. For breakpoint questions, we extend half of the maximum frame count before and after the breakpoint (i.e., placing the breakpoint at the center frame).

To enhance the robustness of the results, we simultaneously employ GPT-3.5 [46] and Claude [3] as LLM assistants, with the additional support of human blind rating. We observe a discrepancy between the accuracy and relative score generated by the previously LLM-assisted evaluation method [43] for video question-answering tasks. However,

- 6 0

- 6 5
- 7 0

- 7 5

- 3 .7 5
- 4 .0 0

- 4
- 5 6 1 2

6 8

- 3 .3
- 3 .4
- 3 .5
- 3 .6
- 3 .7
- 3 .8

6 8

- 6 3
- 6 4
- 6 5
- 6 6
- 6 7
- 6 8

- 3 .4
- 3 .5
- 3 .6
- 3 .7
- 3 .8

c c u r a c y c o r e

3 .8 1

|6 2 5<br><br>|
|---|

|a s<br><br>| |
|---|
|
|---|

| |
|---|
| |

6 6

6 6

3 .5 0

| |
|---|
| |

6 4

6 4

6 7 .8

- 2 .5 0

- 2 .7 5
- 3 .0 0

- 3 .2 5

3 .1 9

6 2

6 2

6 5 .7

2 .9 9

- 5 8
- 6 0

- 5 8
- 6 0

6 2 .3

a c c u r a c y s c o r e

a c c u r a c y s c o r e

| |
|---|
| |
| |

| |
|---|
| |

6 4 1 2 8 2 5 6 5 1 2 1 0 2 4

6 8 1 0 1 2 1 4 1 6 1 8 2 0 2 2 2 4 2 6

1 2 3 4 5

la s t to k e n u n if o r m s a m p lin g m e r g e d to k e n

s h o r t - t e r m in it ia liz a t io n

lo n g - t e r m m e m o r y le n g t h

c o n s o lid a t io n le n g t h

s h o r t - t e r m m e m o r y le n g t h

- Figure 5. Hyperparameter ablation studies on how length of long-term memory buffer llength, short-term memory buffer lshort, consolidation length lmerge and short-term initialization affect the performance of MovieChat on long video understanding. We set lshort = 16, lmerge = 2 in ablation study of long-term memory, llong = 256, lmerge = 2 in ablation study of short-term memory and lshort = 16 in ablation study of consolidation length and short-term initialization.

Method CI DO CU TU CO Video Chat [37] 3.04 2.75 3.09 3.00 3.21 Video LLaMA [82] 2.75 2.24 2.83 2.62 2.97 Video-ChatGPT [43] 2.37 2.30 2.58 2.49 2.69 MovieChat (Ours) 3.11 2.93 3.24 3.17 3.25

Table 4. Quantitative evaluation for long video generation performance in global mode with the average of GPT-3.5 [46], Claude [3] and human blind rating. CI stands for correctness of information, DO stands for detail orientation, CU stands for contextual understanding, TU stands for temporal understanding, and CO stands for consistency. The best result is in bold, and the second best is underlined.

merely adjusting the prompt for the LLM cannot effectively address this issue. Therefore, after obtaining the accuracy and score from the LLM-assisted evaluation method, we implement manual filtering to remove results with inconsistent values, thus improving the reliability of our outcomes.

As shown in Tab. 3, compared to previous methods [37,

- 43,82], MovieChat reads more video frames. In both global mode and breakpoint mode, our method maintains a performance gain in terms of the average accuracy and score provided by LLM assistants and human blind rating. We comprehensively evaluate MovieChat’s question-answering performance across different question types compared to baselines. The results indicate that our approach outperforms the baselines in both open-ended and true-false questions.

Long video generative performance. We compare the quality of answers generated by MovieChat and previous methods [37, 43, 82] in long video question-answering on MovieChat-1K. As shown in Tab. 4, with the average score provided by GPT-3.5 [46], Claude [3] and human bling rating, our approach continues to generate higher-quality answers even as the video contents become more extensive.

###### 5.2. Ablation Study

Short-term and long-term memory buffers. As MovieChat incorporates a memory mechanism including

Global Mode Breakpoint Mode Accuracy Score Accuracy Score w/o MM 51.4 3.10 38.2 2.31

Method

base 67.8 3.81 50.4 2.96

- Table 5. Ablation study on how memory mechanism (MM) affects the long video question answering. The best result is in bold.

Method

Global Mode Breakpoint Mode CI DO CU TU CO CI DO CU TU CO

w/o MM 3.30 2.53 3.28 2.77 3.42 2.42 2.85 2.87 2.00 2.87 base 3.32 3.28 3.40 2.97 3.48 2.97 3.24 3.31 2.70 3.45

- Table 6. Ablation study on how memory mechanism (MM) affects the long video generative performance. CI stands for correctness of information, DO stands for detail orientation, CU stands for contextual understanding, TU stands for temporal understanding, and CO stands for consistency. The best result is in bold.

short-term memory and long-term memory, it is imperative to evaluate how the proposed memory mechanism influences the performance. Tab. 5 and Tab. 6 provide the memory-dependent performance of MovieChat for long video question-answering and generative tasks with the average results of GPT-3.5 [46], Claude [3], and human blind rating. MovieChat with the memory mechanism significantly outperforms the memory-independent variant, which signifies the importance of memory mechanisms.

Hyper-parameter ablations. We perform a series of hyperparameter ablations based on the MovieChat-1K dataset to better understand MovieChat. Fig. 5 shows the performance when ablating the length of memory buffers, consolidation length and short-term initialization with the average results of GPT-3.5 [46], Claude [3], and human blind rating. The performance of MovieChat degrades when all four are significantly changed, showing the validity of our empirically chosen hyperparameyers. Fig. 5 demonstrates that information obtained from the video expands with the growing length of memory buffers, while the loss of finer details intensifies with the fixed length of consolidation. Further-

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

- Figure 6. Question and answer about a clip from YouTube, which is a tutorial on how to cook steak. The entire instructional process begins with marinating the steak, followed by pan-searing it, preparing side dishes, and ultimately plating the meal. Green ( Red ) highlights the correct (wrong) answer and yellow indicates that the model is hallucinating.

more, using merged tokens for short-term initialization outperforms last few tokens and uniform sampling. Additionally, the length of merged tokens and the memory buffer size have a combined effect on MovieChat’s performance.

(Q#2). The evaluation is conducted between MovieChat and previous methods [37,43,83] as shown in Fig. 6 . For Q#1 in breakpoint mode, we mark the timestamp when the question is asked. For long videos over 10K frames, MovieChat is still capable of providing excellent responses to questions regarding both the current moment and the entire video content with less hallucination. More examples to show long video scene understanding and temporal understanding ability of MovieChat are available in appendix.

###### 5.3. Case Study

We perform an extensive case study of MovieChat on a variety of open-ended long video (such as cartoon movie and TV series) for long video question-answering, including the breakpoint mode (Q#1) and the global mode

###### 6. Limitation

Although MovieChat has demonstrated impressive abilities in long video understanding, it is still an early-stage prototype and has some limitations, including: 1) Limited perception capacities. MovieChat’s performance is hindered by the pretrained short video understanding model. 2) Inadequate Time Processing. MovieChat provides only rough estimates of the duration proportions of events within long videos, lacking precision in temporal details.

###### 7. Conclusion

Conclusively, we presents an innovative video understanding system integrating video foundation models and large language models. By incorporating a memory mechanism represented by tokens in Transformers, our proposed system, MovieChat overcomes challenges associated with analyzing long videos. MovieChat achieves state-of-the-art performance in long video understanding, surpassing existing systems limited to handling videos with few frames.

###### References

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736,

2022. 1, 2

- [2] Roy Allen, Peter Mcgeorge, David G Pearson, and Alan Milne. Multiple-target tracking: A role for working memory? Quarterly journal of experimental psychology, 59(6):1101–1116, 2006. 3
- [3] Anthropic. Meet claude, 2023. 6, 7, 13, 17, 19, 20, 21
- [4] Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Luˇci´c, and Cordelia Schmid. Vivit: A video vision transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6836–6846,

2021. 3

- [5] Richard C Atkinson and Richard M Shiffrin. Chapter: Human memory: A proposed system and its control processes. The psychology of learning and motivation, 2:89–195, 1968. 2
- [6] George Awad, Asad A Butt, Keith Curtis, Jonathan Fiscus, Afzal Godil, Yooyoung Lee, Andrew Delgado, Jesse Zhang, Eliot Godard, Baptiste Chocot, et al. Trecvid 2020: A comprehensive campaign for evaluating video retrieval tasks across multiple application domains. arXiv preprint arXiv:2104.13473, 2021. 2
- [7] George Awad, Asad A Butt, Keith Curtis, Yooyoung Lee, Jonathan Fiscus, Afzal Godil, Andrew Delgado, Jesse Zhang, Eliot Godard, Lukas Diduch, et al. Trecvid 2019: An evaluation campaign to benchmark video activity detection, video captioning and matching, and video search & retrieval. arXiv preprint arXiv:2009.09984, 2020. 2
- [8] George Awad, Asad A Butt, Keith Curtis, Yooyoung Lee, Jonathan Fiscus, Afzad Godil, David Joy, Andrew Delgado, Alan F Smeaton, Yvette Graham, et al. Trecvid 2018: Benchmarking video activity detection, video captioning and matching, video storytelling linking and video search. In Proceedings of TRECVID 2018, 2018. 2
- [9] George Awad, Asad A Butt, Jonathan Fiscus, David Joy, Andrew Delgado, Willie Mcclinton, Martial Michel, Alan F Smeaton, Yvette Graham, Wessel Kraaij, et al. Trecvid 2017: evaluating ad-hoc and instance video search, events detection, video captioning, and hyperlinking. In TREC Video Retrieval Evaluation (TRECVID), 2017. 2
- [10] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1728–1738,

2021. 2

- [11] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. CoRR, abs/2104.00650, 2021. 14
- [12] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. In The Eleventh International Conference on Learning Representations, 2022. 4

- [13] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 1, 2
- [14] Jiarui Cai, Mingze Xu, Wei Li, Yuanjun Xiong, Wei Xia, Zhuowen Tu, and Stefano Soatto. Memot: multi-object tracking with memory. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8090–8100, 2022. 3
- [15] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6299–6308, 2017. 2
- [16] Wenhao Chai and Gaoang Wang. Deep vision multimodal learning: Methodology, benchmark, and trend. Applied Sciences, 12(13):6588, 2022. 1
- [17] David Chen and William B Dolan. Collecting highly parallel data for paraphrase evaluation. In Proceedings of the 49th annual meeting of the association for computational linguistics: human language technologies, pages 190–200, 2011. 2
- [18] Ho Kei Cheng and Alexander G Schwing. Xmem: Longterm video object segmentation with an atkinson-shiffrin memory model. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXVIII, pages 640–658. Springer, 2022. 3
- [19] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2023. 1, 2
- [20] StableLM contributors. Stablevicuna. 2023. 17
- [21] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning. arXiv preprint arXiv:2305.06500, 2023. 1, 2
- [22] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palme: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023. 1
- [23] Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva: Exploring the limits of masked visual representation learning at scale. 2022. 4
- [24] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Zhenyu Qiu, Wei Lin, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023. 1
- [25] Difei Gao, Luowei Zhou, Lei Ji, Linchao Zhu, Yi Yang, and Mike Zheng Shou. Mist: Multi-modal iterative spatialtemporal transformer for long-form video question answering. In Proceedings of the IEEE/CVF Conference on Com-

puter Vision and Pattern Recognition, pages 14773–14783,

2023. 2

- [26] Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, et al. Llama-adapter v2: Parameter-efficient visual instruction model. arXiv preprint arXiv:2304.15010,

2023. 1, 2

- [27] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind: One embedding space to bind them all. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15180–15190, 2023. 2
- [28] Tao Gong, Chengqi Lyu, Shilong Zhang, Yudong Wang, Miao Zheng, Qian Zhao, Kuikun Liu, Wenwei Zhang, Ping Luo, and Kai Chen. Multimodal-gpt: A vision and language model for dialogue with humans. arXiv preprint arXiv:2305.04790, 2023. 1, 2
- [29] Zhicheng Hao, Jun Qiu, Haimiao Zhang, Guangbo Ren, and Chang Liu. Umotma: Underwater multiple object tracking with memory aggregation. Frontiers in Marine Science, 9:1071618, 2022. 3
- [30] Li Hu, Peng Zhang, Bang Zhang, Pan Pan, Yinghui Xu, and Rong Jin. Learning position and target consistency for memory-based video object segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4144–4154, 2021. 3
- [31] Qingqiu Huang, Yu Xiong, Anyi Rao, Jiaze Wang, and Dahua Lin. Movienet: A holistic dataset for movie understanding. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IV 16, pages 709–727. Springer, 2020. 2, 5, 14, 15
- [32] Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of NAACL-HLT, pages 4171–4186, 2019. 4
- [33] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning. arXiv preprint arXiv:2305.03726, 2023. 1, 2
- [34] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023. 1, 2
- [35] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. 2023. 4
- [36] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International Conference on Machine Learning, pages 12888–

12900. PMLR, 2022. 1

- [37] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 1, 2, 6, 7, 8, 17, 19, 20, 21

- [38] Boyu Liu, Yanzhao Wang, Yu-Wing Tai, and Chi-Keung Tang. Mavot: Memory-augmented video object tracking. arXiv preprint arXiv:1711.09414, 2017. 3
- [39] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485,

2023. 1

- [40] Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. Video swin transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3202–3211, 2022. 3
- [41] Chenyang Lyu, Minghao Wu, Longyue Wang, Xinting Huang, Bingshuai Liu, Zefeng Du, Shuming Shi, and Zhaopeng Tu. Macaw-llm: Multi-modal language modeling with image, audio, video, and text integration. arXiv preprint arXiv:2306.09093, 2023. 1
- [42] Chao Ma, Jia-Bin Huang, Xiaokang Yang, and Ming-Hsuan Yang. Adaptive correlation filters with long-term and shortterm memory for object tracking. International Journal of Computer Vision, 126:771–796, 2018. 3
- [43] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023. 1, 2, 6, 7, 8, 15, 17, 18, 19, 20, 21
- [44] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. arXiv preprint arXiv:2308.09126, 2023. 2
- [45] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2630–2640, 2019. 2
- [46] openai. Gpt3.5, 2021. 2021. 6, 7, 13, 17, 18, 19, 20, 21
- [47] OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 1, 2
- [48] Michaela Regneri, Marcus Rohrbach, Dominikus Wetzel, Stefan Thater, Bernt Schiele, and Manfred Pinkal. Grounding action descriptions in videos. Transactions of the Association for Computational Linguistics, 1:25–36, 2013. 2
- [49] Anna Rohrbach, Marcus Rohrbach, Wei Qiu, Annemarie Friedrich, Manfred Pinkal, and Bernt Schiele. Coherent multi-sentence video description with variable level of detail. In Pattern Recognition: 36th German Conference, GCPR 2014, M¨unster, Germany, September 2-5, 2014, Proceedings 36, pages 184–195. Springer, 2014. 2
- [50] Anna Rohrbach, Marcus Rohrbach, Siyu Tang, Seong Joon Oh, and Bernt Schiele. Generating descriptions with grounded and co-referenced people. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4979–4989, 2017. 2
- [51] Marcus Rohrbach, Sikandar Amin, Mykhaylo Andriluka, and Bernt Schiele. A database for fine grained activity detection of cooking activities. In 2012 IEEE conference on computer vision and pattern recognition, pages 1194–1201. IEEE, 2012. 2

- [52] Marcus Rohrbach, Michaela Regneri, Mykhaylo Andriluka, Sikandar Amin, Manfred Pinkal, and Bernt Schiele. Script data for attribute-based recognition of composite activities. In Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part I 12, pages 144–157. Springer,

2012. 2

- [53] Marcus Rohrbach, Anna Rohrbach, Michaela Regneri, Sikandar Amin, Mykhaylo Andriluka, Manfred Pinkal, and Bernt Schiele. Recognizing fine-grained and composite activities using hand-centric features and script data. International Journal of Computer Vision, 119:346–373, 2016. 2
- [54] Fadime Sener, Dipika Singhania, and Angela Yao. Temporal aggregate representations for long-range video understanding. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XVI 16, pages 154–171. Springer, 2020. 2
- [55] Hongje Seong, Junhyuk Hyun, and Euntai Kim. Kernelized memory network for video object segmentation. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXII 16, pages 629–645. Springer, 2020. 3
- [56] Hongje Seong, Seoung Wug Oh, Joon-Young Lee, Seongwon Lee, Suhyeon Lee, and Euntai Kim. Hierarchical memory matching network for video object segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12889–12898, 2021. 3
- [57] Mike Zheng Shou, Stan Weixian Lei, Weiyao Wang, Deepti Ghadiyaram, and Matt Feiszli. Generic event boundary detection: A benchmark for event segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8075–8084, 2021. 2, 5
- [58] Mattia Soldan, Alejandro Pardo, Juan Le´on Alc´azar, Fabian Caba, Chen Zhao, Silvio Giancola, and Bernard Ghanem. Mad: A scalable dataset for language grounding in videos from movie audio descriptions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5026–5035, 2022. 2, 5
- [59] Larry R Squire, Lisa Genzel, John T Wixted, and Richard G Morris. Memory consolidation. Cold Spring Harbor perspectives in biology, 7(8):a021766, 2015. 3
- [60] Jianlin Su. Bert position encoding. https://kexue. fm/archives/7947, 2023. 5
- [61] Yixuan Su, Tian Lan, Huayang Li, Jialu Xu, Yan Wang, and Deng Cai. Pandagpt: One model to instruction-follow them all. arXiv preprint arXiv:2305.16355, 2023. 1
- [62] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. Stanford alpaca: An instruction-following llama model, 2023. 1, 2
- [63] Makarand Tapaswi, Yukun Zhu, Rainer Stiefelhagen, Antonio Torralba, Raquel Urtasun, and Sanja Fidler. Movieqa: Understanding stories in movies through questionanswering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4631–4640,

2016. 2, 14, 15

- [64] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste

- Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 1, 2, 17, 18
- [65] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 2, 17, 18
- [66] Paul Vicol, Makarand Tapaswi, Lluis Castrejon, and Sanja Fidler. Moviegraphs: Towards understanding human-centric situations from videos, 2018. 14, 15
- [67] Junke Wang, Dongdong Chen, Chong Luo, Xiyang Dai, Lu Yuan, Zuxuan Wu, and Yu-Gang Jiang. Chatvideo: A tracklet-centric multimodal and versatile video understanding system. arXiv preprint arXiv:2304.14407, 2023. 1, 2
- [68] Jiahao Wang, Guo Chen, Yifei Huang, Limin Wang, and Tong Lu. Memory-and-anticipation transformer for online action understanding. arXiv preprint arXiv:2308.07863,

2023. 3

- [69] Wenhai Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Yu Qiao, et al. Visionllm: Large language model is also an open-ended decoder for vision-centric tasks. arXiv preprint arXiv:2305.11175, 2023. 1, 2
- [70] Chao-Yuan Wu, Christoph Feichtenhofer, Haoqi Fan, Kaiming He, Philipp Krahenbuhl, and Ross Girshick. Long-term feature banks for detailed video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 284–293, 2019. 2
- [71] Chao-Yuan Wu and Philipp Krahenbuhl. Towards long-form video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1884–1894, 2021. 2, 5
- [72] Chao-Yuan Wu, Yanghao Li, Karttikeya Mangalam, Haoqi Fan, Bo Xiong, Jitendra Malik, and Christoph Feichtenhofer. Memvit: Memory-augmented multiscale vision transformer for efficient long-term video recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13587–13597, 2022. 2
- [73] Ming Xin, Wenjie Sun, Kaifang Li, and Guancheng Hui. Multi-object tracking with spatial-temporal correlation memory networks. In 2022 3rd International Conference on Computer Vision, Image and Deep Learning & International Conference on Computer Engineering and Applications (CVIDL & ICCEA), pages 616–619. IEEE, 2022. 3
- [74] Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and motion. In Proceedings of the 25th ACM international conference on Multimedia, pages 1645–1653, 2017. 6
- [75] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. June 2016. 6
- [76] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Zero-shot video question answering via frozen bidirectional language models. Advances in Neural Information Processing Systems, 35:124–141, 2022. 6

- [77] Tianyu Yang and Antoni B Chan. Learning dynamic memory networks for object tracking. In Proceedings of the European conference on computer vision (ECCV), pages 152– 167, 2018. 3
- [78] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178, 2023. 1, 2
- [79] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 9127–9134, 2019. 6
- [80] Abhay Zala, Jaemin Cho, Satwik Kottur, Xilun Chen, Barlas Oguz, Yashar Mehdad, and Mohit Bansal. Hierarchical video-moment retrieval and step-captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23056–23065, 2023. 2
- [81] Kuo-Hao Zeng, Tseng-Hung Chen, Juan Carlos Niebles, and Min Sun. Title generation for user generated videos. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14, pages 609–625. Springer, 2016. 2
- [82] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. 1, 2, 6, 7, 17, 19, 20, 21
- [83] Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint arXiv:2303.16199,

2023. 6, 8, 17, 19, 20

- [84] Zechu Zhou, Xinyu Zhou, Zhaoyu Chen, Pinxue Guo, QianYu Liu, and Wenqiang Zhang. Memory network with pixellevel spatio-temporal learning for visual object tracking. IEEE Transactions on Circuits and Systems for Video Technology, 2023. 3
- [85] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 1, 2

## MovieChat: From Dense Token to Sparse Memory for Long Video Understanding

### Supplementary Material

The supplementary material is structured as follows:

- 1. We first present schematic diagram of the memory consolidation algorithm of MovieChat in Section A.
- 2. We provide detailed supplementary statistical information for MovieChat-1K in Section B.
- 3. The prompt template we use for LLM-Assisted Evaluation is shown in Section C.
- 4. We also list the hyperparameter settings of MovieChat in Section D.
- 5. We mention the specifical LLM-Assisted Evaluation method employed for the assessment of short video generative performance in Section E.
- 6. In comparison to the LLM currently used by MovieChat, we switch to a different LLM and compare the results in Section F.
- 7. To avoid the impact of misjudgments by LLM assistants on the results, we introduce the manual filtering strategy in Section G.
- 8. The performance of MovieChat varies across different categories of questions, and we present the results in

- Section H.

9. We add the result of quantitative evaluation for long video generative performance in breakpoint mode in

- Section I.

- 10. To demonstrate the outstanding performance of MovieChat across a wide range of categories, we calculate the Pearson correlation coefficient of different score methods in Section J.
- 11. We then list the evaluaotion results with GPT-3.5 [46], Claude [3] and human blind rating in Section K.
- 12. We conduct result analyses for the ablation study on hyperparameter settings in Section L.
- 13. Lastly, we give more examples for scene understanding and temporal understanding of MovieChat in Section M.

###### A. Memory consolidation algorithm of MovieChat.

As shown in Fig. A1, for each sampled frame xi, we calculate its similarity with adjacent frames. After that, we select the pair with the greatest similarity, merge and replace these two frames, resulting in a new sequence. We conduct the merge operation repeatedly until the count of existing frames in short-term memory reaches the predefined value.

###### B. MovieChat-1K Statistics Information

Distribution of video categories. MovieChat-1K contains videos from 15 popular categories with varying distribution. As shown in Tab. B1, every video comprises multiple alternating scenes.

Category Percentage

Documentary Film 21.80% Animation Film 17.00% Detective Film 15.10% Epic Film 11.40%

Action Film 6.70% Family Film 4.90% Crime Film 3.80%

Science Fiction Film 3.70%

War Film 3.70% Adventure Film 3.50%

Romance Film 3.30%

History Film 2.10% Suspense Film 1.30%

Fantasy 0.90% School Film 0.80%

Table B1. Distribution of video categories in MovieChat-1K.

Video information and visual question-answer data format. To the best of our knowledge, a long video understanding dataset has not yet been established. Our work represents the initial step in creating and making it publicly available.We create MovieChat1K, containing 1k long videos and corresponding 1k dense captions, and 13k visual question-answer pairs.One visual example of these arrangements is provided in Figure B2.

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Short-term Memory:

…

16 frames

[Figure 31]

[Figure 32]

[Figure 33]

…

[Figure 34]

[Figure 35]

Merged Tokens: 2 frames

[Figure 36]

[Figure 37]

𝑥 𝑥

Memory Consolidation

[Figure 38]

[Figure 39]

[Figure 40]

𝑥 𝑥 … 𝑥 𝑥

#### … 𝑠 𝑠 … 𝑠

[Figure 41]

#### …

𝑥

[Figure 42]

𝑥 𝑥 𝑥

𝑥

𝑥

𝑥 𝑥

𝑥

0.81

0.88

0.94

0.52

3.Merge the two frames then form a new sequence.

𝑥 𝑥

[Figure 43]

[Figure 44]

2.Select the pair with the greatest similarity.

1.For each set of adjacent frames, calculate its cosine similarity.

4. Until reach the predefined count.

[Figure 45]

[Figure 46]

𝑥 𝑥

Figure A1. Question and answer about clips from YouTube, which is a tutorial on how to cook steak. The entire instructional process begins with marinating the steak, followed by pan-searing it, preparing side dishes, and ultimately plating the meal.

Sentence length distribution of question-answer pairs. MovieChat1K exhibits diverse lengths of question-answer pairs in the segmented clip level. Fig. B3 and Fig. B4 demonstrate the length distribution of question-answer pairs in different modes. Despite the distribution of questionanswer pairs varies between the global mode and breakpoint mode, the majority of questions tends to concentrate between 5-15 words in length, while the length of answers generally have fewer than 10 words.

analyze the number of verbs in captions, focusing on extracting and tagging all unique verbs. We find a total of 109,485 verbs in the WebVid10M caption dataset, while the MovieChat-1K captions contain 102,988 unique instances of verbs. While these counts may not be entirely accurate due to our simple counting method, we believe they provide a rough indication of the actionness of the two datasets.

Stastics information of dense captions. To facilitate a more detailed understanding of long videos, we provide a dense caption for each video. As shown in Fig. B5, MovieChat-1K exhibits diverse caption lengths in the segmented clip level. Approximately two-thirds of the clips have captions with 100-149 words, while one-fifth of the clip captions have fewer than 100 words. About 11% of clips have long captions with more than 150 words.

Comparison between MovieChat-1K and other benchmarks. MovieChat-1K provides a large-scale benchmark for long video understanding, which contains 1K movies, 1K dense captions and 13k question-answer pairs. The comparison between different datasets are shown in Tab. B2. It is evident that MovieChat-1K provides the longest average duration for movie clips. MovieQA [63] exclusively offers question-answer pairs related to movies, while MovieGraphs [66] supplies captions associated with movies. Unlike other datasets, MovieNet [31] encompasses three main types of texts: subtitle, synopsis, and script, excluding question-answer pairs. Additionally, the synopsis category is designed for the entire movie rather than video clips. Consequently, MovieChat-1K is more suitable for studying long video comprehension compared to other datasets.

To analyze the word distribution of our generated captions, we compute their distributions. The resulting word distribution of the captions is presented in Fig. B6, which includes common objects (man, woman, people, girl, etc.), attributes (detective, various, small, white, etc.), locations (inside, behind, south, next, etc.), scenes (room, house, building, office, etc.), actions/events (talk, enter, leave, take, etc.), and more.

In terms of actionness, MovieChat-1K captions contains nearly the same number of verbs as with the WebVid10M dataset [11]. To evaluate this, we use the NLTK toolkit to

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

"info": { "video_path": "MI6-19.mp4", "url": "", "class": "action film", "w": 720, "h": 480, "num_frame": 10500, "fps": 25

}, "caption": "It is a part of a movie. The theme of the movie is spy and agent. The helicopter is crashed on the snow land on the cliff.

At the same time, in a room, a man is tied on the rope, and another man is trying to kill a woman. The fight against each other. The man tied on the rope helps the woman, but then he is nearly killed by the rope. Luckily, the woman finally kills the man and saves the man who is nearly killed by the rope. At the cliff, a man is kicked down the cliff by another man. ",

"global": [ {

"question": "When does the things in the video happens, ancient age, modern age or future?", "answer": "Modern age."

}, ... {

"question": "Does it happen during day or night?", "answer": "Day."

}

], "breakpoint": [

{

"time": 750, "question": "What are the people doing?", "answer": "Fighting."

}, ... {

"time": 9750, "question": "Are there any plants?", "answer": "Yes."

} ]

Figure B2. Video information and visual question-answer data format in MovieChat1K.

Dataset Avg. Duration (min) Number of Captions Avg. Caption Length Number of Question-Answer Pairs Avg. Question Length Avg. Answer Length MovieQA [63] 3.5 - - 14.9K 9.3 5.1

MovieGraphs [66] 0.73 15K 35 - - MovieNet [31] 2.1 2.5K - - - MovieChat-1K 9.4 1K 121 13K 7.8 2.3

Table B2. Comparison between MovieChat-1K and other benchmarks.

###### C. LLM-Assisted Evaluation for the short video question-answering task.

Following [43], we use LLM-Assisted Evaluation for the short video question-answering task in Section 5.1. Given

|0<br><br>1000<br><br>2000<br><br>3000<br><br>4000<br><br>5000<br><br>5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21|
|---|

- (a) Length of total questions.

|0<br><br>100<br><br>200<br><br>300<br><br>400<br><br>500<br><br>600<br><br>700<br><br>6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21|
|---|

- (b) Length of global questions.

|0<br><br>1000<br><br>2000<br><br>3000<br><br>4000<br><br>5000<br><br>5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21|
|---|

- (c) Length of breakpoint questions.

Figure B3. Length distribution of questions.

the question, correct answer, and predicted answer by the model, the LLM assistants should return the True or False judgement and relative score (0 to 5). The whole prompt is shown in Fig. C1. It takes about 250 tokens per question. We report the baseline results of short video questionanswering from https://github.com/mbzuaioryx/Video-ChatGPT.

###### D. Hyperparameter Setting

Description Default Value

size of sliding window 16 frames size of short-term memory 18 frames × 32 tokens per frames size of long-term memory 256 frames

consolidation length 2

Table D3. Hyper-parameter settings of MovieChat.

We report the detailed hyperparameter settings of MovieChat in Tab. D3. The sliding window size of MovieChat is set to 16, which means that every slide involves the extraction of 16 frames. We configure the shortterm memory to consist of 18 frames, with each frame con-

|0<br><br>1000<br><br>2000<br><br>3000<br><br>4000<br><br>5000<br><br>6000<br><br>7000<br><br>1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 24|
|---|

- (a) Length of total answers.

|0<br><br>200<br><br>400<br><br>600<br><br>800<br><br>1000<br><br>1200<br><br>1400<br><br>1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24|
|---|

- (b) Length of global answers.

|0<br><br>1000<br><br>2000<br><br>3000<br><br>4000<br><br>5000<br><br>6000<br><br>1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 24|
|---|

- (c) Length of breakpoint answers.

- Figure B4. Length distribution of answers.

[Figure 51]

[Figure 52]

100-149 words

67.78%

0-49 words

1.30%

250-299 words

- 0.40%

200-249 words

- 1.60%

150-199 words

9.91%

50-99 words

18.92%

- Figure B5. Distribution of caption length.

[Figure 53]

Figure B6. Word Cloud of the caption set in MovieChat-1K.

|openai.ChatCompletion.create( model="gpt-3.5-turbo", messages=[<br><br>{<br><br>"role": "system", "content":<br><br>"You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. " "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"<br><br>"------" "##INSTRUCTIONS: " "- Focus on the meaningful match between the predicted answer and the correct answer.\n" "- Consider synonyms or paraphrases as valid matches.\n" "- Evaluate the correctness of the prediction compared to the answer."<br><br>}, {<br><br>"role": "user", "content":<br><br>"Please evaluate the following video-based question-answer pair:\n\n" f"Question: {question}\n" f"Correct Answer: {answer}\n" f"Predicted Answer: {pred}\n\n"<br><br>"Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. " "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING." "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "<br><br>"For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."<br><br>} ]<br><br>)|
|---|

Figure C1. Prompt for ChatGPT in LLM-Assisted Evaluation for the short video question-answering task.

taining 32 tokens. When the short-term memory reaches its capacity, it is directed to the memory consolidation module to be merged into 2 representative frames. The 2 frames are simultaneously input into the long-term memory with a total length of 256 and used to reinitialize the short-term memory.

###### E. LLM-Assisted Evaluation for short video generative performance.

We use LLM-Assisted Evaluation proposed by [43] for short video generative performance in Section 5.1. The evaluation pipeline assesses various capabilities of the model and assigns a relative score (1 to 5) to the generated predictions, in the following five aspects: Correctness of Information, Detail Orientation, Contextual Understanding,

Temporal Understanding and Consistency. We follow the corresponding prompts provided in https://github. com/mbzuai-oryx/Video-ChatGPT and report the baseline results of short video generative performance from it.

###### F. Ablation study on large language models.

Most previous video understanding methods [37,43,82, 83] primarily employed LLama [64] and its variants [20] as text decoders. With the average results of GPT-3.5 [46], Claude [3] and human blind rating, Tab. F4 and Tab. F5 illustrate how the performance of MovieChat changes when using LLama [64] and LLama2 [65] as the large language model respectively.

Contrary to our hypothesis, under every evaluation

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

Q#1 at 4250

Question

###### Which step took the longest to complete this dish?

MovieChatLLama

The longest step to complete this dish would be the preparation of the food on the cutting board. It requires chopping the meat and adding other ingredients like broccoli and carrots, and mixing them together before putting them in the pan. The chef had to carefully chop and arrange the ingredients before cooking the dish.

The step that took the longest to complete this dish was the cooking the meat on the stovetop. It took around 30 minutes to complete this step, followed by 20 minutes to prepare the ingredients and cook the pasta, and 10 minutes to cook the meat and vegetables.

MovieChatLLama2

Figure F2. Question and answer about clips from YouTube, which is a tutorial on how to cook steak. When asked “Which step took the longest to complete this dish?” in global mode, MovieChat powered by different large language models provided divergent answers.

fer an approximation of the time required for each step (indicated by underlines Fig. F2). While its time estimates do not precisely match the actual durations, the proportion of time provided was realistic. Even though LLama2 [65] cannot obtain specific time information when processing feature-rich video frames, MovieChat’s memory buffer design allows for dense sampling of video frames, enabling LLama2 [65] to estimate the proportion of time for each scene based on adjacent similar frames. Therefore, we propose that the lower evaluation metric results of MovieChat with LLama2 [65] compared to MovieChat with LLama [64] may be attributed to the question-answer pairs provided by the dataset.

Global Mode Breakpoint Mode Accuracy Score Accuracy Score

Method

LLama [64] 67.8 3.81 50.4 2.96 LLama2 [65] 64.2 3.79 48.1 2.98

- Table F4. Ablation Study on how LLM affects the long video question answering. The best result is highlighted in bold.

Method

Global Mode Breakpoint Mode CI DO CU TU CO CI DO CU TU CO

LLama [64] 3.32 3.28 3.40 2.97 3.48 2.97 3.24 3.31 2.70 3.45 LLama2 [65] 3.27 3.28 3.41 2.95 3.45 2.96 3.12 3.38 2.68 3.34

- Table F5. Ablation Study on how the large language model affects the long video generative performance. MM stands for memory mechanism, CI stands for correctness of information, DO stands for detail orientation, CU stands for contextual understanding, TU stands for temporal understanding, and CO stands for consistency.The best result is highlighted in bold.

###### G. Manual filtering strategy for LLM-Assisted Evaluation.

For each test data, [43] utilized GPT-3.5 [46] to provide an evaluation result in terms of a ’yes/no’ response and a corresponding score, as demonstrated in Fig. C1. The score is an integer value ranging from 0 to 5, where a score of 5 indicates the highest degree of meaningful correspondence. However, we observe instances where GPT-3.5 [46] offered judgments and scores that do not align, such as providing a ’yes’ response with a score of 0 or a ’no’ response with a score of 5. This discrepancy has the potential to impact the accuracy of results and introduce fluctuations. We adapt the prompts used for GPT-3.5 [46] with the aim of addressing this concern and did not yield the desired mitigation. Hence,

conditions, the performance metrics of MovieChat with LLama2 [65] hardly surpassed those of MovieChat with LLama [64]. We further investigate a specific example to analyze this phenomenon. As shown in Fig. F2, the bold segments represent direct responses to the questions from two versions of MovieChat. MovieChat with LLama [64] provided answers that are more aligned with the video content. Surprisingly, MovieChat with LLama2 [65] of-

Total Multi-choice Open-ended Accuracy Score Accuracy Score Accuracy Score

Method

Video Chat [37] 61.0 3.34 74.8 3.83 56.4 3.02 Video LLaMA [82] 51.4 3.10 78.3 3.58 38.8 2.67 Video-ChatGPT [43] 44.2 2.71 52.5 3.16 37.7 2.54

MovieChat (ours) 67.8 3.81 80.9 4.02 57.5 3.74

- Table H6. Quantitative evaluation for long video different types question answering in global mode. The best result is highlighted in bold, and the second best is underlined.

Method

Total Multi-choice Open-ended Accuracy Score Accuracy Score Accuracy Score

Video Chat [37] 48.3 2.43 62.4 3.46 44.5 2.19 Video LLaMA [82] 38.2 2.33 57.3 2.39 33.1 2.31 Video-ChatGPT [43] 49.8 2.71 58.3 3.05 47.5 2.37

MovieChat (ours) 50.2 2.96 62.4 3.65 46.7 2.70

- Table H7. Quantitative evaluation for long video different types question answering in breakpoint mode. The best result is highlighted in bold, and the second best is underlined.

we introduce an artificial filtering strategy. For each evaluation result generated by GPT-3.5 [46], we conduct manual screening. We retain only those outcomes that exhibited consistency between the ’yes/no’ judgments and the associated scores, thus enhancing the reliability of the evaluations. Similarly, we applied the same filtering strategy to the evaluation results generated by Claude [3].

###### H. Quantitative evaluation for long video different types question answering.

As shown in Fig. 3, MovieChat-1K contains questionanswer pairs of varies types. To better assess the performance of MovieChat, we conduct evaluations on the long video question answering task using various types of questions. We roughly categorize the question types into multiple-choice questions and open-ended questions. With the average results of GPT-3.5 [46], Claude [3] and human blind rating, Tab. H6 and Tab. H7 respectively present the accuracy and scores of MovieChat and the baseline across different question categories in both global mode and breakpoint mode. In various research conditions, our approach consistently outperforms the baseline, thus substantiating the robustness of MovieChat.

###### I. Quantitative evaluation for long video generative performance in breakpoint mode

Similar to Tab. 4, with the average results of GPT3.5 [46], Claude [3] and human blind rating, Tab. I8 demonstrates that our method outperforms the baseline in long video generative performance in breakpoint mode.

Method CI DO CU TU CO Video Chat [37] 2.42 2.51 2.81 2.10 2.78 Video LLaMA [82] 2.04 2.29 2.63 2.00 2.87 Video-ChatGPT [43] 2.62 2.65 2.86 2.32 2.96 MovieChat (Ours) 2.64 2.60 2.87 2.49 3.08

Table I8. Quantitative evaluation for long video generation performance in breakpoint mode with the average of GPT-3.5 [46], Claude [3] and human blind rating. CI stands for correctness of information, DO stands for detail orientation, CU stands for contextual understanding, TU stands for temporal understanding, and CO stands for consistency. The best result is highlighted in bold, and the second best is underlined.

###### J. Pearson correlation coefficient of different score methods.

Evaluation Method Pearson Correlation Coefficient

GPT3.5 VS. Claude 0.927 GPT3.5 VS. Human Blind Rating 0.955 Claude VS. Human Blind Rating 0.978

Table J9. Pearson correlation coefficient of GPT-3.5 [46], Claude [3], and human blind rating on score. We calculate the mean score across each score dimensions for MovieChat and previous methods [37,43,82,83], and then computes the Pearson correlation between these means for each pair of evaluation methods. The Pearson correlation coefficient, which ranges from -1 to +1, indicates a stronger positive linear relationship between the two sets of data when the coefficient is higher (closer to +1).

The Pearson correlation coefficient is represented by the formula:

n i=1(xi − x¯)(yi − y¯)

rxy =

n i=1(xi − x¯)2 ni=1(yi − y¯)2

where rxy is the Pearson correlation coefficient between two variables x and y, xi and yi are the individual sample points for variables x and y, x and y are the averages of the x and y samples respectively, and n is the number of sample points. The formula essentially assesses the extent of linear correlation between two variables by evaluating the product of their deviations from their respective means. The numerator represents the covariance between the two variables, and the denominator normalizes this value, ensuring that the coefficient remains between -1 and +1. The Pearson correlation coefficient quantifies the extent to which two variables co-vary in comparison to their individual variations.

As shown in Tab. J9and Fig. J3, we conduct pearson correlation analysis between GPT-3.5 [46], Claude [3], and human blind rating. The result indicates a substantial agree-

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

4.0

3.5

Claude

3.0

2.5

2.0

1.5

1.5 2.0 2.5 3.0 3.5 4.0

GPT3.5

(a) GPT-3.5 VS. Claude PCC = 0.927

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

4.0

3.5

GPT-3.5

3.0

2.5

2.0

1.5

1.5 2.0 2.5 3.0 3.5 4.0

Human Blind Rating

(b) GPT-3.5 VS. Human Blind Rating PCC = 0.955

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

4.0

3.5

Claude

3.0

2.5

2.0

1.5

1.5 2.0 2.5 3.0 3.5 4.0

Human Blind Rating

(c) Claude VS. Human Blind Rating PCC = 0.978

Figure J3. Results of the Pearson correlation analysis between three evaluation methods, including GPT-3.5 [46], Claude [3], and human blind rating. PCC stands for Pearson Correlation Coefficient.

Global Mode Breakpoint Mode Accuracy Score Accuracy Score

Method # Frames

Video Chat [37] 32 61.0 3.34 48.3 2.43 Video LLaMA [82] 32 51.4 3.10 38.2 2.31 Video-ChatGPT [43] 100 44.2 2.71 49.8 2.71

MovieChat (ours) 2048 67.8 3.81 50.4 2.96

Table K10. Quantitative evaluation for long video question answering on MovieChat-1K test set with GPT-3.5 [46]. The best result is highlighted in bold, and the second best is underlined.

ment among these evaluation methods. The alignment of scores across different score methods strengthens the reliability of our assessment. Crucially, our proposed method, MovieChat outperforms previous methods [37,43,82,83] in long video understanding tasks. The superior performance of MovieChat is evident across a broad spectrum of categories, suggesting that our model not only has a deeper understanding of long videos and respective questions but also exhibits a more accurate and consistent ability to generate relevant responses.

###### K. Evaluation results with GPT, Claude and human blind rating.

As shown in K10–K18, we provide detailed scoring results for GPT-3.5 [46], Claude [3], and human blind rating across various experiments.

###### L. Analysis on hyperparameter ablations.

As the lengths of the short-term and long-term memory buffers increase, the information acquired by MovieChat from the video expands, as illustrated in Fig. 5. However, more video compression leads to the loss of more detailed information, while the length of the merged tokens remains constant. Therefore, as the lengths of two memory buffers increase, the performance of MovieChat exhibits a trend of initially rising and then declining.

Global Mode Breakpoint Mode Accuracy Score Accuracy Score

Method # Frames

Video Chat [37] 32 52.1 2.59 43.8 2.12 Video LLaMA [82] 32 47.3 2.19 33.2 1.69 Video-ChatGPT [43] 100 39.8 2.04 46.4 2.21

MovieChat (ours) 2048 55.3 2.73 46.4 2.28

- Table K11. Quantitative evaluation for long video question answering on MovieChat-1K test set with Claude [3]. The best result is highlighted in bold, and the second best is underlined.

Method # Frames

Global Mode Breakpoint Mode Accuracy Score Accuracy Score

Video Chat [37] 32 60.2 3.08 46.3 2.32 Video LLaMA [82] 32 56.3 2.72 45.8 2.11 Video-ChatGPT [43] 100 58.7 2.89 47.8 2.43

MovieChat (ours) 2048 63.7 3.15 48.1 2.46

- Table K12. Quantitative evaluation for long video question answering on MovieChat-1K test set with human blind rating. The best result is highlighted in bold, and the second best is underlined.

Method CI DO CU TU CO Video Chat [37] 3.26 3.20 3.38 2.97 3.47 Video LLaMA [82] 3.30 2.53 3.28 2.77 3.42 Video-ChatGPT [43] 2.48 2.78 3.03 2.48 2.99 MovieChat (Ours) 3.32 3.28 3.44 3.06 3.48

- Table K13. Quantitative evaluation for long video generation performance in global mode with GPT-3.5 [46]. CI stands for correctness of information, DO stands for detail orientation, CU stands for contextual understanding, TU stands for temporal understanding, and CO stands for consistency. The best result is highlighted in bold, and the second best is underlined.

Fig. 5 demonstrates how memory consolidation influences the performance. Since the LLM-based evaluation shows a positive correlation between accuracy and score,

Method CI DO CU TU CO Video Chat [37] 2.83 2.43 3.02 2.87 2.93 Video LLaMA [82] 2.04 1.66 2.46 2.07 2.36 Video-ChatGPT [43] 1.81 1.65 2.05 2.07 2.07

- MovieChat (Ours) 2.88 2.82 3.11 3.04 2.96

- Table K14. Quantitative evaluation for long video generation performance in global mode with Claude [3]. CI stands for correctness of information, DO stands for detail orientation, CU stands for contextual understanding, TU stands for temporal understanding, and CO stands for consistency. The best result is highlighted in bold, and the second best is underlined.

Method CI DO CU TU CO Video Chat [37] 3.03 2.61 2.87 3.15 3.23 Video LLaMA [82] 2.91 2.54 2.74 3.01 3.12 Video-ChatGPT [43] 2.83 2.47 2.66 2.92 3.01 MovieChat (Ours) 3.12 2.68 3.17 3.41 3.31

- Table K15. Quantitative evaluation for long video generation performance in global mode with human blind rating. CI stands for correctness of information, DO stands for detail orientation, CU stands for contextual understanding, TU stands for temporal understanding, and CO stands for consistency. The best result is highlighted in bold, and the second best is underlined.

Method CI DO CU TU CO Video Chat [37] 2.96 3.09 3.24 2.46 3.22 Video LLaMA [82] 2.42 2.85 2.87 2.00 2.87 Video-ChatGPT [43] 3.11 3.32 3.29 2.62 3.29 MovieChat (Ours) 3.07 3.24 3.31 2.70 3.45

- Table K16. Quantitative evaluation for long video generation performance in breakpoint mode with GPT-3.5 [46]. CI stands for correctness of information, DO stands for detail orientation, CU stands for contextual understanding, TU stands for temporal understanding, and CO stands for consistency. The best result is highlighted in bold, and the second best is underlined.

Method CI DO CU TU CO Video Chat [37] 2.12 2.20 2.30 1.97 2.37 Video LLaMA [82] 1.62 1.85 2.20 1.34 2.02 Video-ChatGPT [43] 2.36 2.26 2.34 2.23 2.70 MovieChat (Ours) 2.38 2.16 2.35 2.43 2.68

- Table K17. Quantitative evaluation for long video generation performance in breakpoint mode with Claude [3]. CI stands for correctness of information, DO stands for detail orientation, CU stands for contextual understanding, TU stands for temporal understanding, and CO stands for consistency. The best result is highlighted in bold, and the second best is underlined.

Method CI DO CU TU CO Video Chat [37] 2.17 2.24 2.89 1.87 2.75 Video LLaMA [82] 2.09 2.18 2.82 1.74 2.68 Video-ChatGPT [43] 2.39 2.36 2.96 2.10 2.89 MovieChat (Ours) 2.48 2.41 2.94 2.33 3.12

Table K18. Quantitative evaluation for long video generation performance in breakpoint mode with human blind rating. CI stands for correctness of information, DO stands for detail orientation, CU stands for contextual understanding, TU stands for temporal understanding, and CO stands for consistency. The best result is highlighted in bold, and the second best is underlined.

we use accuracy to gauge performance. When memory buffer parameters remain constant, shorter merged tokens indicate increased frame information compression, potentially resulting in information loss when excessive. Conversely, longer merged tokens, despite retaining a greater extent of short-term memory in the face of compression, correspondingly result in less overall information acquisition. Moreover, when the length of the memory buffer changes, as exemplified by long-term memory, the corresponding peak performance of MovieChat shifts in response to the altered length of merged tokens. This demonstrates the need to strike a balance between dense information extraction and information compression in long video understanding tasks.

We also conduct experiments to compare various methods for initializing the short-term memory, including selecting the last few tokens, uniform sampling, and using merged tokens. The results indicate that the use of merged tokens produces the best performance. When initializing the next short-term memory with the last few tokens from the previous short-term memory, it is unable to adequately represent the information from the previous time step. Consequently, this leads to the final merged tokens being either repetitive or lacking coherence with the previous time step. Uniform sampling faces similar issues, but it manages to capture information with representative frames from the previous time step. Consequently, its performance surpasses that of initializing with the last few tokens, yet it remains inferior to using merged tokens for initialization.

###### M. Examples for scene understanding and temporal understanding of MovieChat.

We perform an extensive case study of MovieChat on a variety of open-ended long video (such as cartoon movie in and TV series) for long video question-answering and captioning task, including the global mode and the

breakpoint mode. The evaluation tasks include scene understanding and temporal understanding as shown in Fig. M4, Fig. M5, Fig. M6 and Fig. M7. For Q#1 and

Q#2, we remarks timestamps in frames. For long videos over 10K frames, MovieChat is still capable of providing excellent responses to questions regarding both the current moment and the entire video content.

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

Q#1 at 3930 Q#2 at 11130

Question #1 What is Chief Bogo doing? Question #2 Who is this rabbit named Judy talking to?

Answer #1 In the video, the character Chief Bogo is seen Answer #2 talking to someone on a laptop while sitting at a table in a dark room. Chief Bogo is a cartoon character wearing a brown outfit.

In the video, the rabbit police named Judy is talking to the person in a movie theater, who is a fox.

Question #3

Can you describe this video?

Answer #3

The video shows various scenes of an animation with different animals in a restaurant or office environment. At the beginning, we see an animated cartoon rabbit and a deer talking to each other in a restaurant. Next, there is a cartoon bunny talking to a cat at a desk, followed by a scene of a cartoon deer and a mouse talking to each other. The video then continues with the animated cartoon dog and cat sitting at a table in a restaurant, and another scene where a man and a dog are in a room with a computer on a table. There are more scenes of the same characters later on, including one with a bunny and a fox sitting at a desk, and another with a bunny and a dog talking to each other in the background. Finally, there are several more scenes with different animals, including a fox, a mouse, and a cat, all of which are in a different environment.

Figure M4. Question and answer about clips from Zootopia, a cartoon, which tells the story of a determined police officer rabbit named Judy who pairs up with a cunning fox to uncover a conspiracy about missing animals and develop an unexpected friendship.

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Q#1 at 10290 Q#2 at 14250

Question #1 What is she doing? Question #2 Where is this couple?

Answer #1 In the given video, we see a young woman Answer #2 walking down a street in the city, and a young man also walks down the street behind her. The woman is wearing a red and black outfit.

This couple is standing in the middle of a city street with a black hat on the man's. The scene is shot in a park, and there are a few trees and bushes visible.

Question #3

Can you tell me what happened in this movie?

Answer #3

Certainly! In the video, you can see a group of people standing in a park with a green hill in the background. There are also people walking on the street and in a courtyard with trees. Some of the people are holding red umbrellas, and there is a woman wearing a red coat. She is then seen walking down a street at night with a man following her. Next, we see a couple standing on a balcony overlooking a street in the evening with a man walking by and then walking down the street. Towards the end of the video, a group of people are seen walking through a snowy town.

Figure M5. Question and answer about clips from Goblin, which tells the story of Kim Shin, an immortal ”goblin” who needs to find a human bride to end his endless life but instead meets Ji Eun-tak, a girl fated to die who claims to be the ”goblin’s bride,” leading to a romantic tale unfolding between them.

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Q#1 at 8250 Q#2 at 14400

Do they let the soldiers go first or negotiate with the general first?

In which scene does the movie mainly focus on what happened? Are there other scenarios?

Question #1 Question #2

In this video, the soldiers are being let go of by the general first, then the general negotiates with the soldiers.

Answer #1 In the video, the main focus is on the scene Answer #2

where people are standing in front of a group of buildings, followed by another scene where people are standing at a table with a man standing behind them.

What plot does this movie clip include?

Question #3

The movie clip starts with a black background and text written in white on it. Next, we see a man in a black robe with a shining light and a large white flag on the background. There are also people standing and sitting in front of a large building on a black background with text. Then, we see a man with a beard and a man in a dress, standing on the ground. We also see a man with a shirt on and a man in a blue shirt standing on the ground. Finally, we see a black background with a large black bear and a large bird on it.

Answer #3

Figure M6. Question and answer about clips from Game of Thrones, which tells the epic fantasy tale of power struggles and political intrigue among the Seven Kingdoms, entwined with intricate family relationships, all set against the backdrop of an ancient, mystical threat.

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Q#1 at 4500 Q#2 at 10000

Has there been a black speaker in the video so far?

Question #1 Question #2

What is this clip talking about now?

It shows a young man standing in a dugout on a black background. He is then seen kneeling on the field during a football game, followed by two young men in a dugout. Next, we see a young man standing on the field. And finally, we see a group of young men standing on the field and watching the game. The video also features football players and fans, as well as a football field, field, and dugout.

Yes, there is a man in a red shirt standing on a street with a dark blue sky in the background, and a man with a tie standing on a street with a red car in the background. There is also a man in a black suit and another man in a white jacket standing in front of a building with a street light.

Answer #1 Answer #2

Which movie sequence lasts the longest?

Question #3

Answer #3

The video shows the scene of a dark night with a football field and a group of people playing football.

Figure M7. Question and answer about clips from YouTube, which contains a compilation of some inspirational movies scenes. This video clip comprises several segments from The Death Crawl, Coach Carter, Rocky Balboa, and We Are Marshall, which vary in duration.

