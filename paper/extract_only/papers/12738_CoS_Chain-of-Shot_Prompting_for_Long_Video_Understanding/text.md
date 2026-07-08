# arXiv:2502.06428v2[cs.CV]11Feb2025

## CoS: Chain-of-Shot Prompting for Long Video Understanding

Jian Hu1 Zixu Cheng1 Chenyang Si2 Wei Li2 Shaogang Gong1 https://lwpyh.github.io/CoS/

### Abstract

Multi-modal Large Language Models (MLLMs) struggle with long videos due to the need for excessive visual tokens. These tokens exceed massively the context length of MLLMs, resulting in filled by redundant task-irrelevant shots. How to select shots is an unsolved critical problem: sparse sampling risks missing key details, while exhaustive sampling overwhelms the model with irrelevant content, leading to video misunderstanding. To solve this problem, we propose Chain-ofShot prompting (CoS). The key idea is to frame shot selection as test-time visual prompt optimisation, choosing shots adaptive to video understanding semantic task by optimising shots-task alignment. CoS has two key parts: (1) a binary video summary mechanism that performs pseudo temporal grounding, discovering a binary coding to identify task-relevant shots, and (2) a video coreasoning module that deploys the binary coding to pair (learning to align) task-relevant positive shots with irrelevant negative shots. It embeds the optimised shot selections into the original video, facilitating a focus on relevant context to optimize long video understanding. Experiments across three baselines and five datasets demonstrate the effectiveness and adaptability of CoS. Code given in https://lwpyh.github.io/CoS.

### 1. Introduction

Driven by advancements in Large Language Models (LLMs) (OpenAI, 2023; Jiang et al., 2024; Guo et al., 2025), researchers have extended LLMs to visual understanding tasks (Liu et al., 2024a; OpenAI, March 2024; Shu et al., 2024). By modality alignment and visual instruction tuning, MLLMs have demonstrated effectiveness in tasks such as captioning and visual question answering. Despite MLLMs perform well on single images and short videos (usually

1Queen Mary University of London 2Nanyang Technological University. Correspondence to: Jian Hu <jian.hu@qmul.ac.uk>. Preprint.

[Figure 1]

Figure 1. The effects of changing shot-sampling rates on video understanding task performance on videos of different lengths in the VideoMME (Fu et al., 2024a) dataset. Two models are evaluated including LongVA (Zhang et al., 2024a) and VideoXL (Shu et al., 2024). As the number of sampled shots increased, performance did not consistently improve across various video lengths. That is because while sparse sampling may miss crucial details, exhaustive sampling often overwhelms the model with excessive irrelevant content. This illustrates the key challenge of optimal shot selection especially in long video understanding. That is, how to sample variable details in order to maximise semantic task information extraction whilst minimising distractions from irrelevant details (noise) in video understanding.

under three minutes) (Zhu et al., 2023; Kim et al., 2024), understanding long videos, such as hour-long videos, remains a significant problem unsolved (Zhang et al., 2024a).

This challenge arises from the massive visual tokens generated in long videos by contemporary MLLMs, often exceeding the context length and computational capacity of these models, making it computationally intractable. Existing solutions to extend input capacity include token compression (Li et al., 2025; Wang et al., 2024b; Xue et al., 2024) and specialised memory mechanisms (Song et al., 2024; He et al., 2024; Shu et al., 2024), all aimed at retaining critical information. However, as shown in Fig.1, task-relevant shots in long videos are sparsely distributed. Developing an effective sampling strategy is nontrivial and remains an open problem due to two main reasons. Sampling fewer shots reduces noise and helps the model focuses on relevant information but risks missing critical, sparsely distributed shots. Conversely, sampling more shots captures additional details

[Figure 2]

- Figure 2. The critical problem of how to select shots in video understanding. In a video that depicts how a boy gradually gains a dragon’s trust, different sampling methods create two distinct narratives: split video A shows the boy being attacked by the dragon, while split video B shows him happily sharing food with the dragon. This shows that minor differences in video sampling leads to significant variations in semantic understanding (interpretation).

but introduces significant noise, diluting critical insights. In essence, a solution needs not only optimises (minimises) the number of shots by reducing redundancy and distractions, but also simultaneously captures (maximises) selectively task-relevant information by reducing omissions.

Moreover, there is a representation bias problem with existing methods: the role of visual shot selection in affecting a model’s semantic reasoning process. Current MLLMs mainly process multi-modal inputs by encoding textual and visual information separately, before cross-modal alignment (Liu et al., 2024a; Wang et al., 2024a). While input quality can significantly affect performance, most research has focused only on optimising textual prompts for reasoning tasks, neglecting the importance of visual inputs. For example, VideoCoT (Wang et al., 2024c) relies on handcrafted textual prompts, while VoT (Fei et al., 2024a) uses video sense graphs or query decomposition to enhance reasoning. Such methods mainly refine text inputs but overlook the optimisation of visual inputs, which is essential for long videos when task-relevant information is sparsely distributed. As a result, visual selection from the outset (input) becomes critical. That is illustrated in Fig.2, where different shot selections from the same video can lead to entirely different interpretations, demonstrating how video shots can serve as effective visual prompts to guide a model’s reasoning process. However, this is missing in existing methods. This oversight highlights an unresolved issue: determining how to optimally sample shots that can effectively maximise task-relevant information selection whilst simultaneously minimise noise (distractions) in long-video understanding.

In this work, we propose a novel test-time optimisation strategy named Chain-of-Shot prompting (CoS). It consists of two parts: Binary Video Summary and Video CoReasoning. Binary Video Summary identifies sparsely distributed task-relevant shots by a mosaicing based binary coding on long videos. It leverages MLLMs’ reasoning

and summarisation capacity for pseudo temporal grounding. Video Co-Reasoning then explores this binary coding to construct simultaneously task-relevant positive videos and task-irrelevant negative videos. This guides the model to focus on critical information while filtering out noise. CoS enables test-time model optimisation in long-video understanding by dynamically optimising video inputs during inference. CoS is training-free and designed for automatic adapting and optimising in task-specific (per video instance) temporal-spatial modelling. Comparative experiments on 17 contemporary models using five datasets validate the effectiveness of CoS. Our contributions are:

(1) Long-video understanding by visual prompt learning. We are the first to approach this challenge by optimising input video information to fully utilise the model’s ability to comprehend long videos. (2) Chain-of-Shot prompting (CoS), a training-free mosaicing binary coding together with pseudo temporal grounding is introduced for long video understanding. CoS explores MLLMs’ summary capacity for binary coding and pseudo temporal grounding on long videos. Moreover, it explores test-time model optimisation to dynamically construct per video-instance task-specific positive and negative videos as visual prompts, enabling optimal selection to capture sparsely distributed task-relevant knowledge in long videos while minimising interference from irrelevant information. (3) Comprehensive validation. Extensive experiments across 5 different datasets on 3 diverse baseline methods against 17 models demonstrate the effectiveness of CoS.

### 2. Related Works

MLLMs for visual understanding. In recent years, significant progress has been made in the field of MLLMs for visual understanding (Radford et al., 2021; Zhang et al., 2024c; Maaz et al., 2023). Models like LLaVA (Liu et al.,

[Figure 3]

- Figure 3. The overall framework of CoS. It first utilises LLaVA to perform a mosaicing binary coding to bootstrap video summarisation for temporal grounding on a long video. Specifically, every four shots are aggregated into a mosaicing composition image. LLaVA determines whether task-related elements exist within each composition image by encoding a binary value of 1 or 0 (‘yes’ or ‘no’), thereby identifying sparsely distributed task-related shots to achieve pseudo temporal grounding. Given this binary video summary, task-related positive shots Sp and irrelevant negative shots Sn are generated and represented by binary codes. Sp, Sn and the original frame sequence X sampled from original video V are then fed into the MLLM for co-reasoning, minimising interference of irrelevant video content.

- 2024a) achieved cross-modal feature alignment through projectors, enhancing understanding of single images. As the focus of research is shifting from image-only models to those for multi-image and video inputs, various enhancements to the visual language connector have been proposed. He et al. (2024) and Wang et al. (2023) implemented average pooling, while Jin et al. (2024) and Shu et al. (2024) introduced techniques to dynamically drop visual tokens. Moreover, Cheng et al. (2024) adopted spatial-temporal convolution to better capture the dynamics of a video and reduce feature size. However, memory constraints and the lack of large-scale annotated hour-long datasets limit current models. They struggle to process and understand temporal information in long videos beyond a few minutes, leading to poor performance on long video understanding.

MLLMs for Long Video Understanding. To improve performance on long videos, several studies have introduced more fine-grained annotations in datasets at various scales to aid training (Fu et al., 2024a; Wu et al., 2024). Zhang et al. (2024a) and He et al. (2024) extended the context window of LLMs to encompass more extensive temporal information. LongVILA (Xue et al., 2024) further utilized a parallel processing system to achieve context compression at the input level. LLaVA-Vid (Li et al., 2025) and VideoXL (Shu et al., 2024) sought to obtain a highly compact representation that preserves key information for effective token compression. However, these compression techniques invariably lead to loss of information and poorer video understanding. Critically, most of these studies focus on learning from the entire video as a single input without selection, neglecting the fact that relevant information in long videos is often sparsely located. When the presence of irrelevant information is not minimised, it detracts the reasoning power of MLLMs.

Prompt Engineering. To enable more effective reason-

ing in visual understanding tasks, VideoCoT (Wang et al., 2024c) decomposed input questions to facilitate image-level visual reasoning by MLLMs. Similarly, VoT (Fei et al., 2024a) used a sense graph and problem decomposition to enhance short video comprehension and reasoning. AoTD (Shi et al., 2024) realized the reasoning of thought chain through agent-of-thought. VideoGen (Zheng et al., 2024) utilised chain-of-thought to assist the video generation process. Himakunthala et al. (2023) and Han et al. (2024) built Chainof-thought from a dataset perspective to help better evaluate the model’s video understanding capabilities. However, these methods mainly focus on optimising text inputs to improve reasoning, neglecting the significant temporal changes between adjacent shots in long videos. Blindly inputting an entire long video for model processing affect the model’s understanding of both the video and the questions. Our approach is the first to explore temporal and spatial modelling on visual inputs for long video understanding, ensuring the visual data better aligns with text questions and enhances model reasoning on long videos.

### 3. Methodology

In this work, we introduce a training-free plug-in mechanism called Chain-of-Shot prompting (CoS), which dynamically optimises the visual input at test-time per video instance subject to the given video understanding task. Specifically, given a video V , a video MLLM samples a sequence of shots X = {x1,x2,x3,...,xn} containing n shots. CoS leverages the spatial reasoning and summarisation power of a MLLM to perform binary coding for pseudo temporal grounding. Based on this binary coding, task-relevant positive shots and irrelevant negative shots are constructed. These sub-shots, together with the original raw long video, are input to the MLLM for co-reasoning, allowing the model

to effectively extract task-relevant information and minimise the negative impact of irrelevant shots, thereby enhancing its reasoning capabilities.

#### 3.1. A Closer Look at MLLM Reasoning

To elaborate on how CoS works, we first revisit how MLLMs typically perform visual understanding tasks.

Given a video V and a query P, a shot sampler first uniformly samples n shots to form the set X. A MLLM with parameters θ generates a response y by auto-regressively sampling from a probability distribution conditioned on P, X, and previously generated tokens:

yt ∼ pθ(yt | X,P,y<t) ∝ exp(logitθ(yt | X,P,y<t)),

(1) where yt denotes the token at time t, and y<t represents a sequence of tokens generated up to time t − 1.

Despite the advanced capabilities of MLLMs, handling long videos remains a challenge. Task-relevant shots are often sparsely located and unknown in advance. Low sampling rates may miss these critical shots. Conversely, increasing the sampling rate introduces irrelevant information, making it harder for the model to focus on key visual features. Subtle variations in visual inputs can significantly affect the model’s outputs, making it crucial to balance sampling efficiency and information relevance.

#### 3.2. Binary Video Summary

To provide the model with effective and clear visual inputs, we need to perform video temporal grounding based on a given query (task), identifying which shots are related to the task. However, MLLMs exhibit poor temporal grounding capabilities (Wang et al., 2024a), especially for long videos where critical information is sparse, and the volume of irrelevant information is overwhelming.

While MLLMs often struggle with direct temporal grounding, they possess strong visual reasoning and summary abilities (Liu et al., 2024a). To leverage these abilities, we perform indirect key shot localization through a binary video summary. Specifically, the model performs spatial localization for each shot to identify whether task-relevant elements exist. By framing this process as a binary classification task (e.g., answering “yes” or “no”), we achieve a simplified yet effective way to distinguish between relevant and irrelevant shots. Given a query-specific prompt Ps (“Is anything in the keyword list present in the image? Just answer yes or no.”+ video-specific question Qi), the model processes each shot xi of the video and outputs a binary result oi:

oi = MLLM(Ps,xi), (2)

shots classified as “yes” are labelled as task-relevant (positive), while shots classified as “no” are labelled as task-

irrelevant (negative). This step enables a binary coding of the video shots, where each shot is tagged as either relevant or irrelevant. Consequently, long videos can be summarised into task-relevant and task-irrelevant segments, forming a binary representation of the visual input.

However, this process has two computational problems: (1) Due to time complexity, evaluating every shot individually is computationally expensive, particularly when the number of sampled shots n is large. (2) Certain temporal-spatial events span multiple consecutive shots (e.g., dynamic actions like cooking), and analysing single shots may fail to capture these temporal dependencies.

To solve these problems, inspired by the idea of using image gird (aka mosaicing) for visual understanding (Kim et al., 2024), we extend the binary video summary concept by combining every k consecutive shots into m aggregated mosaicing images for reasoning:

A = a1,a2,a3,...,am, where m =

n k

. (3)

Each aggregated image as, consisting of k shots, is processed as a single unit by MLLM with the same prompt Ps as follows:

oi = MLLM(Ps,as), (4)

If MLLM outputs “yes”, the corresponding group is classified as task-relevant; otherwise, it is deemed irrelevant. This grouping allows us to reduce computational complexity while preserving temporal information across multiple shots. Here, we set k as 4, and LLaVA (Liu et al., 2024a) as the MLLM, More analysis on the hyper-parameter selection is in Tab.5. We use this binary video summary strategy to encode the long video into task-relevant segments for pseudo temporal grounding.

#### 3.3. Video Co-Reasoning

In long videos, task-relevant shots are usually sparsely distributed, making it hard for models to identify critical content among irrelevant information. Therefore, we use LLaVA (Liu et al., 2024a) to generate pseudo grounding labels and further process the video to construct balanced sub-shots, providing structured visual inputs for reasoning.

3.3.1. CONSTRUCTING BALANCED SUB-SHOTS

The original video V is first sampled to obtain a sequence of shots X = {x1,x2,x3,...,xn}, where n is the total number of sampled shots. Based on the MLLM’s output, we classify each shot in X as either task-relevant (“yes”) or irrelevant (“no”). Shots labelled as ”yes” are included in the positive sub-shot Sp, while shots labelled as “no” are included in the negative sub-shot Sn. Specifically, the index

- Table 1. Experimental results on VideoMME benchmarks, we report results with and without subtitle assistance. † indicates that the results were reproduced using their official weights. The best is in bold.

|Models|Size|shots<br><br>|VideoMME w/o sub.<br><br>|VideoMME w/ sub.|
|---|---|---|---|---|
| | | |Short Medium Long Avg<br><br>|Short Medium Long Avg|

Proprietary Models

GPT-4V (OpenAI, 2023) - 384 70.5 55.8 53.5 59.9 73.2 59.7 56.9 63.3 GPT-4o (OpenAI, March 2024) - 384 80.0 70.3 65.3 71.9 82.8 76.6 72.1 77.2

Gemini-1.5-Pro (Team et al., 2024) - 0.5 fps 81.7 74.3 67.4 75.0 84.5 81.0 77.4 81.3 Claude3-Opus (Anthropic, March 2024) - - 71.0 57.4 51.2 60.0 73.5 60.1 54.7 62.9

Open-source MLLMs

|VideoChat2 (Li et al., 2024) VideoLLaVA (Lin et al., 2023) Sharegpt4Video (Chen et al., 2024a) InternVL-Chat-V1.5 (Chen et al., 2024b) Video-CCAM (Fei et al., 2024b) Long-LLaVA (Wang et al., 2024b) VITA (Fu et al., 2024b) Kangaroo (Liu et al., 2024b) LongVILA (Xue et al., 2024)<br><br>|7B 7B 7B 20B 14B<br><br>7B<br><br>8x7B<br><br>8B 7B<br>|196 49 16 10 96 128 20 64 256<br><br>|48.3 37.0 33.2 39.5 45.3 38.0 36.2 39.9 48.3 36.3 35.0 39.9<br><br>60.2 46.4 45.6 47.8 62.2 50.6 46.7 53.2<br><br>61.9 51.4 45.4 52.9 64.2 53.3 47.6 55.0 66.1 55.3 46.7 56.0 69.0 58.3 53.0 60.1<br><br><br>|52.8 39.4 39.2 43.8 46.1 40.7 38.1 41.6<br>53.6 39.3 37.9 43.6 61.7 49.1 46.6 52.4 66.0 56.3 49.9 57.4<br><br><br>66.2 54.7 50.3 57.1<br>67.9 55.3 49.6 57.6<br>68.0 55.4 49.3 57.6 72.9 64.9 57.4 65.1<br>|
|---|---|---|---|---|
|LongVA (Zhang et al., 2024a) LongVA+Ours<br><br>|7B 7B<br><br>|128 128<br><br>|61.1 50.4 46.2 52.6 61.6 52.0 46.8 53.5<br><br>|61.6 53.6 47.6 54.3 64.2 54.4 48.5 55.7<br><br>|
|Video-XL† (Shu et al., 2024) Video-XL+Ours<br><br>|7B 7B<br><br>|128 128<br><br>|63.1 52.4 48.7 54.7<br><br>64.1 53.6 49.1 55.6<br><br><br>|68.3 55.7 52.1 58.7 68.9 57.1 52.3 59.5<br><br>|
|LLaVA-Video (Zhang et al., 2024c) LLaVA-Video+Ours<br><br>|7B 7B<br><br>|64 64<br><br>|76.1 61.8 52.1 63.3<br><br>77.2 62.4 53.8 64.4<br><br><br>|78.0 69.3 61.8 69.7 80.1 69.4 65.1 71.5<br><br>|

set of task-relevant shots Ibasic is defined as:

Ibasic = {i | MLLM output for shot xi = ”yes”}, (5)

Positive Shot Sp. Task-relevant shots are often sparsely distributed, and directly sampling based on task relevance may result in too few shots, causing significant imbalance between Sp and Sn. To ensure Sp includes only task-relevant shots while maintaining a balanced length relative to the video, we adopt the following strategy:

 

xk, if k ∈ [i + 1, n] and k ∈ Ibasic xj, if k not found, j ∈ [1, i − 1] and j ∈ Ibasic Xi, if no valid j or k is found.

Sip =

, (6)



this ensures Sp contains only frames from Ibasic by prioritising neighbouring key shots from Ibasic, which maintains Sp ’s length consistent with the original video. If no suitable key shots are found, we set Sp as the original video X.

Negative Shot Sn. For each shot xi in X, if i ∈/ Ibasic, the shot is directly included in Sn. We employ the following replacement strategy to ensure Sn primarily contains taskirrelevant content:

 

xk, if k ∈ [i + 1, n] and k ∈/ Ibasic xj, if k not found, j ∈ [1, i − 1] and j ∈/ Ibasic black shot, if no valid j or k is found.

Sin =

###### ,



(7)

this ensures Sn captures irrelevant shots while maintaining the same length as Sp and X.

3.3.2. CO-REASONING WITH SUB-SHOTS

After constructing the balanced sub-shots Sp, Sn, and the sampled sequence X from the original video V , we jointly input these components into the model for reasoning. The model combines outputs from X, Sp, and Sn to produce a final response as follows:

α

pθ(yt | Sp, Q, y<t) pθ(yt | Sn, Q, y<t)

yt ∝ pθ(yt | X, Q, y<t) ·

∼ softmax logitθ(yt | X, Q, y<t) + α · logitθ(yt | Sp, Q, y<t) − α · logitθ(yt | Sn, Q, y<t) ,

(8)

where α is a weighting parameter to adjust the influence of Sp and Sn during reasoning. Q is a question for the video.

Dynamic Weighting Mechanism. Since Sp and Sn are constructed from mutually exclusive pseudo grounding labels, their confidence levels are linked: accurate identification of shots in Sp implies high accuracy for task-irrelevant shots in Sn, and vice versa.

Intuitively, when Sp contains many shots, it closely resembles the sampled sequence X, meaning the gain from the pseudo grounding process is limited. Conversely, when Sp contains fewer shots, it indicates that the task-relevant information in the video is sparsely distributed. In this case, the content in Sp and Sn has a significant impact on the MLLM’s reasoning. Here, α should increase to amplify the

contributions of Sp and Sn. α is defined as:

Table 3. Results on NEXT-QA and MVBench. Models Size MVBench NEXT-QA

|Sp| |X|

, (9)

Proprietary Models GPT-4V (OpenAI, 2023) - 43.5 -

α = 1 −

GPT-4o (OpenAI, March 2024) - - 76.0 Open-source MLLMs

where |Sp| is the number of shots in the positive sub-shot, and |X| is the total number of sampled shots. A smaller ratio |S

|mPLUG-Owl (Ye et al., 2023) Video-LLaVA (Lin et al., 2023) VideoChat2 (Li et al., 2024) TimeChat (Ren et al., 2024) ST-LLM (Liu et al., 2025) PLLaVA (Xu et al., 2024) Long-LLaVA (Wang et al., 2024b) VideoLLava (Lin et al., 2023)<br><br>|7B 7B 7B 7B 7B 7B 7B 7B<br><br>|29.7 51.9 38.5 54.9 58.1 54.6 52.5|33.8 40.2 78.6 45.6 71.1<br><br>|
|---|---|---|---|
|LongVA (Zhang et al., 2024a) LongVA+Ours LLaVA-Video (Zhang et al., 2024c) LLaVA-Video+Ours<br><br>|7B 7B 7B 7B<br><br>|49.7 50.9 58.6 60.5<br><br>|69.3 69.9 74.2 75.1<br><br>|

p|

|X| reflects stronger shot selection. This mechanism allows the model to adaptively balance its reliance on Sp and Sn. When α is large, Sp and Sn have a greater impact, reflecting high confidence in the pseudo grounding. When α is small, the model relies more on X, as Sp offers limited additional information. When α = 0, the model ignores Sp and Sn, reasoning solely with X.

- Table 2. Experimental results on MLVU and LongVideoBench benchmarks, ”LongVideo.” refers to LongVideoBench.

Models Size shots MLVU LongVideo. Proprietary Models

2024): A benchmark designed for tasks requiring precise retrieval and reasoning over detailed multimodal information within extended inputs. MVBench (Li et al., 2024): A benchmark cross over 20 challenging video understanding tasks, focusing on temporal understanding in dynamic video tasks. It is particularly suited for evaluating CoS’s image concatenation strategy. NEXT-QA (Xiao et al., 2021): A short-video benchmark emphasizing causal and temporal reasoning, challenging models to understand complex sequences and interactions to answer related questions accurately. Additionally, we compared CoS against state-ofthe-art general video understanding methods and long-video understanding approaches (both open- and closed-source) to comprehensively demonstrate its effectiveness.

GPT-4V (OpenAI, 2023) - 384 49.2 60.7 GPT-4o (OpenAI, March 2024) - 384 64.6 66.7

Gemini-1.5-Pro (Team et al., 2024) - 0.5 fps - 64.4 Open-source MLLMs

|VideoChat2 (Li et al., 2024) VideoLLaVA (Lin et al., 2023) Shargpt4Video (Chen et al., 2024a) Video-CCAM (Fei et al., 2024b)|7B 7B 7B 14B<br><br>|196 49 16 96<br><br>|47.9 47.3 46.4 63.1|39.3 37.6 41.8 -<br><br>|
|---|---|---|---|---|
|LongVA (Zhang et al., 2024a) LongVA+Ours<br><br>|7B 7B<br><br>|128 128<br><br>|56.3 58.9<br><br>|47.8 52.8<br><br>|
|Video-XL† (Shu et al., 2024) Video-XL+Ours<br><br>|7B 7B<br><br>|128 128<br><br>|64.3 65.2<br><br>|49.8 50.6<br><br>|
|LLaVA-Video (Zhang et al., 2024c) LLaVA-Video+Ours<br><br>|7B 7B<br><br>|64 64<br><br>|70.8 71.4<br><br>|58.2 58.9<br><br>|

### 4. Experiments

Metrics. All five datasets are evaluated using the accuracy metric, where a higher value indicates better performance.

To evaluate the effectiveness of the proposed method, we conducted experiments with three various baselines on five datasets across videos of varying lengths, including the VideoMME (Fu et al., 2024a) dataset, the long-video datasets MLVU (Zhou et al., 2024) and LongVideoBench (Wu et al., 2024), as well as two short-tomedium video datasets, NEXT-QA (Xiao et al., 2021) and MVBench (Li et al., 2024) for diversity.

Implementation Details. CoS is a training-free, test-time adaptive plug-in. We followed the shot sampling setup predefined in the baselines for evaluation. Specifically, we set the sampling rate to 128 shots for LongVA and Video-XL, and 64 shots for LLaVA-Video. During the binary coding phase, every four sampled shots are concatenated to form a composite shot for input into the model, enabling temporal-spatial modelling. The binary coding process uses LLaVA1.5-13B as the backbone MLLM. To ensure computational efficiency, we employed 4-bit quantization and parallel computation using batch decode. Our method runs efficiently on a single 80G A100 GPU. Although our algorithm introduces an additional sample selection module, its inference time complexity and space complexity remain the same as the baseline, both being O(n). More analysis on time costing is in Tab.5(b).

#### 4.1. Experimental Setup

Baselines. To validate the effectiveness of CoS, we integrated CoS into three contemporary long-video understanding baselines: LongVA (Zhang et al., 2024a), VideoXL (Shu et al., 2024), and LLaVA-Video (Zhang et al.,

- 2024b). To ensure robustness, we evaluated CoS across five datasets: VideoMME (Fu et al., 2024a): A large-scale dataset containing videos of varying lengths (short, medium, long) and diverse scenarios, ideal for evaluating model performance across different temporal scales. MLVU (Zhou et al., 2024): A large-scale long-video dataset featuring diverse scenes and tasks. LongVideoBench (Wu et al.,

#### 4.2. Results and Analysis

VideoMME. VideoMME dataset allows for evaluating performance across videos of different lengths. As shown in

- Table 4. Ablation Study on VideoMME with VideoXL and LLaVA-Video

|Method’s Variants<br><br>|VideoXL|LLaVA-Video|
|---|---|---|
|BVS OFL PFL NFL DWM|short medium long avg|short medium long avg|
|✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓<br><br>✓ ✓ ✓ ✓ ✓<br><br>|63.1 52.4 48.7 54.7 52.3 45.6 47.2 48.4 63.8 53.3 48.8 55.3 63.5 53.2 48.6 55.2<br><br>63.4 53.3 48.5 55.1<br><br>64.1 53.6 49.1 55.6<br><br><br>|76.1 61.8 52.1 63.3 58.8 52.4 51.6 54.3<br><br>76.8 61.7 52.6 63.7<br>77.1 61.0 52.0 63.4<br><br><br>76.5 61.8 53.1 63.9<br><br>77.2 62.4 53.8 64.4<br><br><br>|

- Table 5. Parameter ablation study on VideoMME with LongVA as the baseline.

(a) Various MLLM for Binary video summary. (b) Shot-sampling rate. (c) Aggregation shot count.

|MLLM|short medium long avg|
|---|---|
|LongVA (Zhang et al., 2024a) MinichatGPT (Zhu et al., 2023) Qwen2 (Wang et al., 2024a) LLaVA1.5 (Liu et al., 2024a)<br><br>|58.8 49.6 43.2 50.4<br><br>60.7 51.3 44.6 52.2<br><br>61.2 52.6 45.9 53.2 61.6 52.0 46.8 53.5<br><br><br>|

|shots<br><br>|short medium long avg|
|---|---|
|64 96 128 192<br><br>|61.1 50.2 44.9 52.1<br><br>60.9 52.0 46.1 53.0<br><br>61.6 52.0 46.8 53.5 60.8 51.8 46.0 52.9<br><br><br>|

|k|short medium long avg time (s)|
|---|---|
|2 4 8 16<br><br>|61.2 51.9 46.9 53.3 20.7 61.6 52.0 46.8 53.5 15.7 60.8 52.0 46.3 53.0 13.6 60.3 51.1 46.1 52.4 11.9<br><br>|

Tab.1, we integrated CoS into three baselines and compared the results against closed-source methods and open-source general video methods as well as long-video understanding ones. Evaluations were conducted under both with subtitle and without subtitle settings. Results show that CoS achieves significant improvements across all baselines and temporal scales (short, medium, and long videos). Notably, CoS exhibits larger performance gains on LLaVA-Video and LongVA, with relatively smaller gains on Video-XL due to its built-in context attention mechanism, which overlaps with CoS’s design. Nevertheless, CoS still delivers improvements, validating its effectiveness.

MLVU and LongVideoBench. These datasets are long video benchmarks. As shown in Tab.2, on MLVU’s dev set and LongVideoBench’s dev set, CoS achieves superior performance compared to all closed-source methods and other open-source 7B-scale models. This demonstrates CoS’s strong performance in long-video understanding tasks.

NEXT-QA and MVBench. These datasets focus on shortvideo understanding, including temporal reasoning and inference tasks. As shown in Tab.3, CoS delivers significant improvements on two baselines across both datasets, achieving leading performance on their respective benchmarks. This highlights that CoS’s visual prompting modification not only yields gains in long-video tasks but also generalizes well to short-video tasks, underscoring its effectiveness.

Module Analysis. As illustrated in Tab.4, we conducted an ablation study on the VideoMME dataset using VideoXL and LLaVA-Video as baselines to assess the impact of various modules. “BVS” stands for the binary video summary module, and “OFL” refers to the original shots inputted into Eq. 8, “PFL” denotes the selected positive shots inputted into Eq. 8, and ”NFL” represents the selected negative shots inputted into Eq. 8, while “DWM” is the dynamic weighting

mechanism. In the first row, without the binary video summary module, only original videos are fed into the MLLM for visual understanding, causing the model to regress to a baseline model and significantly underperforming compared to the CoS-enhanced model, thereby demonstrating the effectiveness of our approach. The second row removes the original videos in Eq. 8 and relies solely on the selected positive and negative shots for inference, it falls significantly compared to CoS, indicating that the content of the original videos provides a margin of error for the shot selection strategy. It ensures that incorrectly classified information can still be processed by the model through the original video feed. The third and fourth rows evaluate the influence of positive and negative videos, respectively, indicating that both contribute to visual understanding. The penultimate row, which omits the dynamic weighting mechanism, performs worse than the full CoS model, highlighting the effectiveness of dynamic weighting strategy.

Shot Selection Model Analysis. In Tab.5(a), we used LongVA (Zhang et al., 2024a) as the baseline to assess the impact of different MLLMs during the binary video summary phase on shot selection. Initially, we employed LongVA itself for shot selection, which yields the poorest results. This is because LongVA is better suited for temporal tasks and the entire pipeline relies solely on LongVA for inference, making it difficult to correct the inherent biases. Performance improves with other MLLMs such as miniChatGPT (Zhu et al., 2023), indicating that employing diverse MLLMs for their respective strengths can better mitigate the biases a single model might exhibit under unlabelled conditions. This also suggests that general-purpose MLLMs might possess superior capabilities in spatial positioning and reasoning compared to large models specifically designed for videos. The performance of Qwen2 (Wang et al., 2024a) and LLaVA1.5 are comparable, as we leverage

[Figure 4]

Figure 4. An qualitative evaluation example from MLVU (Zhou et al., 2024) dataset.

visual reasoning and summaries to achieve pseudo temporal grounding, where Qwen2’s superior temporal reasoning capabilities have limited scope for impact.

Impact of Shot-sampling Rate. As depicted in Tab. 5(b), we utilized LongVA as the baseline to investigate the impact of different shot-sampling rates on performance with the VideoMME. When the frame sampling count is limited to only 64, the performance observed is relatively mediocre. It is attributed to the inadequate sampling, which fails to capture essential information effectively. However, as the sampling count increases, ranging from 96 to 192 frames, the model’s performance exhibits stability, underscoring the robustness of our approach. It suggests that our CoS is capable of dynamically selecting the optimal number of shots, thereby efficiently aggregating information even when the distribution of relevant shots is sparse.

Image Aggregation Shot Count Analysis. In Tab.5(c), we used LongVA as the baseline to evaluate the impact of different image aggregation shot counts on performance with VideoMME, where ”time” indicates the average inference speed per video. A smaller number of aggregated images results in finer granularity for pseudo temporal grounding, leading to more accurate grounding but also increased processing time and difficulty in capturing temporal relations between shots. Conversely, more aggregated shots increase the model’s inference speed but reduce the granularity of pseudo grounding. We find that while an aggregation count of 2 offers good key shot location ability in longer videos due to finer grounding granularity, it is more

time-consuming. When the aggregation count exceeds 4, although the inference speed is faster, the accuracy of pseudo temporal grounding decreases and the increased number of shots aggregated per image poses challenges in spatial positioning for the model, leading to a significant decrease in performance. However, with an aggregation count of 4, the inference speed is reasonable, and the grounding granularity is moderately balanced, achieving effective temporal-spatial grounding, hence we chose an aggregation shot count of 4.

Qualitative Evaluation. We present qualitative examples of CoS on LLaVA-Video baseline in Fig.4. CoS+LLaVAVideo excels at pinpointing precise details within extensive videos. This underscores its adeptness at retrieving and analysing visual data across prolonged sequences. Moreover, CoS can effectively answer the question by detailing key characters, settings, and plot events, showcasing its capacity to handle and interpret exceedingly long videos.

### 5. Conclusion

In this work, we introduced a training-free test-time optimisation plug-in mechanism called Chain-of-Shot prompting (CoS) for long video understanding. CoS dynamically selects shots from videos based on per video instance specific query task, constructing task-relevant positive and task-irrelevant negative videos from the sparsely distributed useful shots. This approach enhances models’ video understanding ability to comprehend tasks and achieve better reasoning performance. Extensive experiments demonstrate the effectiveness of our method.

### References

Anthropic. https://www.anthropic.com/news/claude-3family. Technical report, March 2024.

Chen, L., Wei, X., Li, J., Dong, X., Zhang, P., Zang, Y., Chen, Z., Duan, H., Lin, B., Tang, Z., et al. Sharegpt4video: Improving video understanding and generation with better captions. arXiv preprint arXiv:2406.04325, 2024a.

Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Zhong, M., Zhang, Q., Zhu, X., Lu, L., et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24185–24198, 2024b.

Cheng, Z., Leng, S., Zhang, H., Xin, Y., Li, X., Chen, G., Zhu, Y., Zhang, W., Luo, Z., Zhao, D., et al. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024.

Fei, H., Wu, S., Ji, W., Zhang, H., Zhang, M., Lee, M.L., and Hsu, W. Video-of-thought: Step-by-step video reasoning from perception to cognition. In Forty-first International Conference on Machine Learning, 2024a.

Fei, J., Li, D., Deng, Z., Wang, Z., Liu, G., and Wang, H. Video-ccam: Enhancing video-language understanding with causal cross-attention masks for short and long videos. arXiv preprint arXiv:2408.14023, 2024b.

Fu, C., Dai, Y., Luo, Y., Li, L., Ren, S., Zhang, R., Wang, Z., Zhou, C., Shen, Y., Zhang, M., et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024a.

Fu, C., Lin, H., Long, Z., Shen, Y., Zhao, M., Zhang, Y., Dong, S., Wang, X., Yin, D., Ma, L., et al. Vita: Towards open-source interactive omni multimodal llm. arXiv preprint arXiv:2408.05211, 2024b.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Han, S., Huang, W., Shi, H., Zhuo, L., Su, X., Zhang, S., Zhou, X., Qi, X., Liao, Y., and Liu, S. Videoespresso: A large-scale chain-of-thought dataset for fine-grained video reasoning via core frame selection. arXiv preprint arXiv:2411.14794, 2024.

He, B., Li, H., Jang, Y. K., Jia, M., Cao, X., Shah, A., Shrivastava, A., and Lim, S.-N. Ma-lmm: Memoryaugmented large multimodal model for long-term video

understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13504–13514, 2024.

Himakunthala, V., Ouyang, A., Rose, D., He, R., Mei, A., Lu, Y., Sonar, C., Saxon, M., and Wang, W. Y. Let’s think frame by frame with vip: A video infilling and prediction dataset for evaluating video chain-of-thought. arXiv preprint arXiv:2305.13903, 2023.

Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D. S., Casas, D. d. l., Hanna, E. B., Bressand, F., et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.

Jin, P., Takanobu, R., Zhang, W., Cao, X., and Yuan, L. Chat-univi: Unified visual representation empowers large language models with image and video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13700–13710, 2024.

Kim, W., Choi, C., Lee, W., and Rhee, W. An image grid can be worth a video: Zero-shot video question answering using a vlm. arXiv preprint arXiv:2403.18406, 2024.

Li, K., Wang, Y., He, Y., Li, Y., Wang, Y., Liu, Y., Wang, Z., Xu, J., Chen, G., Luo, P., et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22195–22206, 2024.

Li, Y., Wang, C., and Jia, J. Llama-vid: An image is worth 2 tokens in large language models. In European Conference on Computer Vision, pp. 323–340. Springer, 2025.

Lin, B., Ye, Y., Zhu, B., Cui, J., Ning, M., Jin, P., and Yuan, L. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. Advances in neural information processing systems, 36, 2024a.

Liu, J., Wang, Y., Ma, H., Wu, X., Ma, X., Wei, X., Jiao, J., Wu, E., and Hu, J. Kangaroo: A powerful videolanguage model supporting long-context video input. arXiv preprint arXiv:2408.15542, 2024b.

Liu, R., Li, C., Tang, H., Ge, Y., Shan, Y., and Li, G. St-llm: Large language models are effective temporal learners. In European Conference on Computer Vision, pp. 1–18. Springer, 2025.

Maaz, M., Rasheed, H., Khan, S., and Khan, F. S. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023.

OpenAI. Gpt-4 technical report. Technical report, OpenAI, 2023.

OpenAI. Chatgpt: Optimizing language models for dialogue. https://openai.com/chatgpt, 2023.

OpenAI. Openai, gpt-40. Technical report, March 2024.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Ren, S., Yao, L., Li, S., Sun, X., and Hou, L. Timechat: A time-sensitive multimodal large language model for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14313–14323, 2024.

Shi, Y., Di, S., Chen, Q., and Xie, W. Unlocking videollm via agent-of-thoughts distillation. arXiv preprint arXiv:2412.01694, 2024.

Shu, Y., Zhang, P., Liu, Z., Qin, M., Zhou, J., Huang, T., and Zhao, B. Video-xl: Extra-long vision language model for hour-scale video understanding. arXiv preprint arXiv:2409.14485, 2024.

Song, E., Chai, W., Wang, G., Zhang, Y., Zhou, H., Wu, F., Chi, H., Guo, X., Ye, T., Zhang, Y., et al. Moviechat: From dense token to sparse memory for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18221– 18232, 2024.

Team, G., Georgiev, P., Lei, V. I., Burnell, R., Bai, L., Gulati, A., Tanzer, G., Vincent, D., Pan, Z., Wang, S., et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.

Wang, J., Chen, D., Luo, C., Dai, X., Yuan, L., Wu, Z., and Jiang, Y.-G. Chatvideo: A tracklet-centric multimodal and versatile video understanding system. arXiv preprint arXiv:2304.14407, 2023.

Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024a.

Wang, X., Song, D., Chen, S., Zhang, C., and Wang, B. Longllava: Scaling multi-modal llms to 1000 images efficiently via a hybrid architecture. arXiv preprint arXiv:2409.02889, 2024b.

Wang, Y., Zeng, Y., Zheng, J., Xing, X., Xu, J., and Xu, X. Videocot: A video chain-of-thought dataset with active annotation tool. arXiv preprint arXiv:2407.05355, 2024c.

Wu, H., Li, D., Chen, B., and Li, J. Longvideobench: A benchmark for long-context interleaved video-language understanding. arXiv preprint arXiv:2407.15754, 2024.

Xiao, J., Shang, X., Yao, A., and Chua, T.-S. Next-qa: Next phase of question-answering to explaining temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9777–9786, 2021.

Xu, L., Zhao, Y., Zhou, D., Lin, Z., Ng, S. K., and Feng, J. Pllava: Parameter-free llava extension from images to videos for video dense captioning. arXiv preprint arXiv:2404.16994, 2024.

Xue, F., Chen, Y., Li, D., Hu, Q., Zhu, L., Li, X., Fang, Y., Tang, H., Yang, S., Liu, Z., et al. Longvila: Scaling longcontext visual language models for long videos. arXiv preprint arXiv:2408.10188, 2024.

Ye, Q., Xu, H., Xu, G., Ye, J., Yan, M., Zhou, Y., Wang, J., Hu, A., Shi, P., Shi, Y., et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178, 2023.

Zhang, P., Zhang, K., Li, B., Zeng, G., Yang, J., Zhang, Y., Wang, Z., Tan, H., Li, C., and Liu, Z. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024a.

Zhang, Y., Li, B., Liu, h., Lee, Y. j., Gui, L., Fu, D., Feng, J., Liu, Z., and Li, C. Llava-next: A strong zero-shot video understanding model, April 2024b. URL https://llava-vl.github.io/ blog/2024-04-30-llava-next-video/.

Zhang, Y., Wu, J., Li, W., Li, B., Ma, Z., Liu, Z., and Li, C. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024c.

Zheng, M., Xu, Y., Huang, H., Ma, X., Liu, Y., Shu, W., Pang, Y., Tang, F., Chen, Q., Yang, H., et al. Videogen-ofthought: A collaborative framework for multi-shot video generation. arXiv preprint arXiv:2412.02259, 2024.

Zhou, J., Shu, Y., Zhao, B., Wu, B., Xiao, S., Yang, X., Xiong, Y., Zhang, B., Huang, T., and Liu, Z. Mlvu: A comprehensive benchmark for multi-task long video understanding. arXiv preprint arXiv:2406.04264, 2024.

Zhu, D., Chen, J., Shen, X., Li, X., and Elhoseiny, M. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.

