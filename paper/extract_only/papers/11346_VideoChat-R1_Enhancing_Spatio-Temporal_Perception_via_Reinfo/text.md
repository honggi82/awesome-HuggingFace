# arXiv:2504.06958v5[cs.CV]11Nov2025

### VideoChat-R1: Enhancing Spatio-Temporal Perception via Reinforcement Fine-Tuning

##### Xinhao Li2,1∗, Ziang Yan3,1∗, Desen Meng2∗, Lu Dong4,1, Xiangyu Zeng2,1, Yinan He1 Yali Wang6,1, Yu Qiao1,6, Yi Wang†1,5, Limin Wang†2,1

1Shanghai AI Laboratory 2Nanjing University 3Zhejiang University 4University of Science and Technology of China 5Shanghai Innovation Institute 6Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences https://github.com/OpenGVLab/VideoChat-R1

#### Abstract

Reinforcement Learning (RL) benefits Large Language Models (LLMs) for complex reasoning. Inspired by this, we explore integrating spatio-temporal specific rewards into Multimodal Large Language Models (MLLMs) to address the unique challenges of video understanding, such as long-range temporal associations. This paper investigates how rule-based rewards, particularly temporal ones, can improve video reasoning and their generalizability. Our study proposes Reinforcement Fine-Tuning (RFT) as a data-efficient method to enhance video reasoning on specific tasks without sacrificing original capabilities. Through joint RFT on multiple spatio-temporal perception tasks, we developed VideoChat-R1, a powerful Video MLLM. VideoChat-R1 achieves state-of-the-art spatio-temporal perception, demonstrating significant improvements in tasks like temporal grounding (+31.8) and object tracking (+31.2), while also improving general QA benchmarks. The enhanced perception and preserved chat abilities contribute to a more reliable video dialogue system, leading to our “Temporal Clue-driven Reasoning” inference schema. This work provides a foundation for developing robust, real-world video comprehension agents.

#### 1 Introduction

The integration of reinforcement learning (RL) has notably propelled the capabilities of large language models (LLMs) forward, particularly in complex reasoning. Studies, such as o-series from OpenAI [12] and R1 of DeepSeek [9], demonstrated that test-time scaling improves model reasoning and such reasoning is elicit-able via only rule-based rewarding. Inspired by these advances, community is interested in tuning multimodal large language models (MLLMs) with vision-related verifications in Group Relative Policy Optimization (GRPO) [9] manner [43, 18, 38, 3, 22, 19, 35, 40, 39, 4]. They primarily consider visual mathematical reasoning and spatial localization.

Regarding video understanding, reasoning is also crucial as some of its core abilties (e.g. its longrange temporal association, fine-grained spatiotemporal understanding based on user queries) are barely addressed via perception or simple analysis. Initial works [42, 27, 5] have validated the superiority of the GRPO algorithm over supervised fine-tuning in some specific video tasks, such as temporal grounding and video question answer. Considering advancing the agent development with real-world video comprehension, we need to investigate and integrate spatio-temporal specific rewards or verification mechanisms directly into the learning of current MLLMs in a scalable manner. This could foster a more tightly coupled, near-closed-loop learning environment, enabling MLLMs to master intricate spatio-temporal skills. While our current work may not fully achieve this ambitious

* Equal contribution. † Corresponding authors.

Preprint. Under review.

###### (1)

Video Reinforcement Fine-Tuning with GRPO Spatio-Temporal Perception

[Figure 2]

Track the rabbit in the video using a box with initial coordinates [0.2, 0.5, 0.3, 0.9].

[Figure 3]

[Figure 4]

|[Figure 5]<br><br>[Figure 6]<br><br>| |
|---|
|
|---|

|[Figure 7]<br><br>[Figure 8]<br><br>| |
|---|
|
|---|

[Figure 9]

[Figure 10]

Few training samples (1000-10000)

Policy Model

| |
|---|

| |
|---|

[Figure 11]

KL divergence constraint

[Figure 12]

9.2s 10.0s

At what point in the video is a rabbit jumping over another rabbit?

Group Relative Policy Optimization (GRPO)

Generate a set of competitions

Chat with enhanced Spatio-Temporal Perception

Reference Model

[Figure 13]

[Figure 14]

|[Figure 15]<br><br>[Figure 16]|
|---|

|[Figure 17]<br><br>[Figure 18]|
|---|

|[Figure 19]<br><br>[Figure 20]|
|---|

|[Figure 21]<br><br>[Figure 22]|
|---|

|[Figure 23]<br><br>[Figure 24]|
|---|

[Figure 25]

[Figure 26]

3.8s 17.9s

Format reward Accuracy reward

What does the driver do after letting go of the steering wheel at the beginning?

VideoChat-R1: <think>\nThe driver is seen holding the steering wheel and driving the car. After letting go of the steering wheel, he appears to be dancing. This can be observed from his body language and facial expressions as he moves his arms and head rhythmically.\n<[think> <answer>Dance. <[answer>\n<clue>[3.8s, 17.9s]<[clue>

Reward Function for various video tasks

IoU reward Caption reward

- Figure 1: Overview of VideoChat-R1. Through reinforcement learning fine-tuning using GRPO, VideoChat-R1 has powerful spatio-temporal perception capabilities and can apply these capabilities in chatting scenarios.

goal, it endeavors to lay foundational groundwork for the community by exploring how to embed such mechanisms effectively. Specifically, we in this paper chart the landscape of how rule-based rewards (especially temporal related ones) work in video understanding, along with systematic evaluations of the its generalizability across diverse video-based reasoning scenarios.

Our study begins with spatio-temporal perception, extending to other popular video tasks (e.g. video caption), forming a practical recipe to enhance video reasoning via proper Reinforcement Fine-Tuning (RFT). We evaluate the robustness of various video verifications through different training and testing setting, also benchmarking their respective data requirements. Moreover, how these rewards interact with each other are also systematically explored. Based on these experiences, we give an intuitive and effective inference schema named “Temporal Clue-driven Reasoning”. Generally, our main findings are as follows.

- • Reinforcement fine-tuning is data-efficient for enhancing models on specific tasks without sacrificing original capabilities. With a small amount of data, training via RFT can yield a remarkable improvement in spatio-temporal perception ability, and there is negligible impact on the performance of out-domain tasks and the original general capabilities of the model, which outperforms traditional supervised fine-tuning significantly.
- • Through joint reinforcement fine-tuning on multiple spatio-temporal perception tasks, we construct VideoChat-R1, a powerful Video MLLM that boasts state-of-the-art spatiotemporal perception capabilities while also taking into account chat abilities. We have also discovered that training on spatio-temporal perception tasks has slightly strengthened the model’s spatio-temporal reasoning abilities. Compared with Qwen2.5-VL-7B, VideoChatR1 achieves several times the performance improvement in spatiotemporal perception tasks such as temporal grounding (+31.8) and object track (+31.2). At the same time, it also achieves non-trivial improvements on general QA benchmarks, such as VideoMME (+0.9), MVBench (+1.0), and Perception Test (+0.9)
- • The improvement of spatio-temporal perception ability and the preservation of the original chat capability can contribute to a more reliable and efficient video dialogue system. Our VideoChat-R1 can provide reference video segments when answering users’ questions. Meanwhile, we propose to utilize these video segments for “Temporal Clue-driven Reasoning” to further obtain more accurate answers. Our experimental results reveal the potential of the approach that enhances the model’s spatiotemporal perception ability through reinforcement learning for future research in the directions of reliable video dialogue systems and long video understanding.

#### 2 Related work

Reasoning in MLLMs. The field LLMs has witnessed significant breakthroughs in reasoning, exemplified by recent works [12, 9]. These developments [23, 9, 24] have markedly improved LLMs’ proficiency in tackling complex, multi-step tasks, such as challenging mathematical problems and coding exercises. Extending these RL techniques to multimodal large language models (MLLMs) has seen numerous efforts [43, 18, 38, 3, 22, 19, 35, 40, 39, 4] focused on leveraging verifiable reward mechanisms to boost visual reasoning performance. However, the application of RL-based strategies to spatio-temporal reasoning within the video domain remains comparatively less explored, with a few studies [27, 42, 5] investigating this area. [27] and [42] show GRPO’s potential in temporal grounding and sentiment analysis, while, [5] demonstrates how GRPO can facilitate temporal reasoning.

Spatio-Temporal Understanding with MLLMs. Video understanding heavily relies on spatiotemporal perception. Progress in Video Multimodal Large Language Models (video MLLMs) [14, 20, 15, 28, 41, 29, 17, 11, 1] addresses video question answering and captioning in the unfied dialogue format, leveraging LLMs to organize predictions over visual perceptions. [36] and [37] leverage data augmentation for temporal modeling, yet compromising their general performance. [33] improves fine-grained spatio-temporal perception by task preference optimization with solid supervised fine-tuning using a lot of data.

#### 3 Methodology

We present how to exploit the spatio-temporal rewards to improve MLLMs in video domains using GRPO. Before that, we revisit the Group Relative Policy Optimization (GRPO) [23] first.

##### 3.1 Preliminary

Group Relative Policy Optimization (GRPO) [23] compares groups of candidates responses directly, independent on critic models. Regarding this, it significantly lowers training resources. Formally, with an input query q, GRPO initiates by generating a set of G distinct candidate responses, denoted as o = {o1,...,oG}, via policy sampling. Each of these candidates is then assigned a score from a predefined reward function, yielding R1,...,RG. To normalize these scores and ascertain the relative quality of each response, GRPO computes their mean and standard deviation:

Ri − mean({Ri}Gi=1) std({Ri}Gi=1)

. (1)

Ai =

Here Ai quantifies the relative quality of the i-th candidate response within its group. The objective of GRPO is to encourage the model to prioritize and generate responses that achieve higher relative scores within such groups. Furthermore, the overall training objective incorporates a KL-divergence term DKL(·|·), which serves to constrain the optimized policy πθ from diverging excessively from the original MLLM parameters πref.

##### 3.2 Spatio-Temporal Rewards of Video MLLM in GRPO

We explore how to use GRPO to enhance the performance of Video MLLM in video-language understanding. We consider the five most common types of video related tasks: temporal grounding, object tracking, video question answering, captioning, and quality assessment in our experiments.

Format reward. To enable the model to output responses in the format we desire. For example, we expect the model to enclose its thought process with <think>...</think> and the answer with <answer>...</answer>, we designed a format reward Rformat for each task. We use regular expression matching to determine whether the model adheres to our specified format:

Rformat =

1, if output matches format, 0, if output doesn’t match format.

(2)

IoU reward in spatio-temporal perception. For the spatio-temporal perception such as temporal grounding and object tracking, it requires the Video MLLM to output the time interval in the video that is associated with the content of a given textual query. Evidently, we can use the Intersection over Union (IoU) between the predicted interval by the model and the ground-truth interval as the reward function. This reward function effectively characterizes the accuracy of the interval predicted by the model.

RIoU = |Ipred ∩ Igt| |Ipred ∪ Igt|

, (3)

where Ipred and Igt are the predicted and the ground truth of time intervals or detection boxes, respectively.

Accuracy reward in classification. Discriminative tasks, such as multiple-choice video question answering and classification, aim to determine whether the model’s prediction is consistent with the answer to the question. Therefore, we define:

Raccuacy =

- 0, if Apred ̸= Agt
- 1, if Apred = Agt,

where Apred and Agt denote the predicted and the ground truth answers, respectively.

(4)

Recall reward in video captioning. For tasks like video captioning with open-ended outputs, it is impossible to simply compare and determine the gap between the generated caption and the ground truth caption. Therefore, we use a LLM as a “judge” to provide a reward score. In order to reduce the uncertainty of the evaluation criteria for the LLM, we first make the LLM decompose the ground truth and predicted captions into events list. Specifically, we utilize Qwen2.5-72B [34] to extract the events in the description and judge whether the events in a ground truth description can be entailed by the description predicted by the model. We calculate the event recall score as the ratio of events in a ground truth description that are entailed by the predicted description, and set different rewards according to the event recall score:

###### Rrecall = Recallevent(Cpred,Cgt), (5)

where Cpred and Cgt represent the predicted and the ground truth captions, respectively. By combining the above reward functions, we explored how to utilize GRPO to enhance the performance of Video MLLM in various tasks. The specific details can be found in the Section 4.

##### 3.3 Enhance Spatio-Temporal Perception of Video MLLM through GRPO

The combination of reward functions. We adopt different combinations of reward functions for training in different tasks. Specifically, for the temporal grounding and object tracking task, Rst = Rformat + RIoU. For the multi-choice QA and video quality assessment, Rqa = Rformat + Raccuacy. For the multi-choice QA with glue (e.g. Grounding QA), Rgqa = Rformat + RIoU + RAcc. For the video caption, Rcap = Rformat + RCaption.

VideoChat-R1-18k. We collect diverse video corpus from existing public video datasets for reinforcement learning training to enhance the model’s spatiotemporal perception ability. For the temporal grounding task, we use the training set of Charade - STA [7] (5,338 samples) for training. For the object tracking task, training is conducted on the GoT - 10k [10] dataset, which has 9,335 samples. For the QA and grounding QA tasks, the validation set of NExTGQA [31] (3,358 samples) is used for training. For video captioning, FIBER-1k [32] (1,000 samples) is adopted for training. For video quality assessment, we use the quality assessment task from VidTAB [16] under the 100-shot setting, with 200 samples for training. Finally, for the training of VideoChat-R1, we perform joint training on three spatio-temporal perception-related tasks: temporal grounding, object tracking, and grounding QA. In total, 18,031 samples are used for training.

Algorithm 1: VideoChat M with “Clue-Perception” Input: Vlow: Low-resolution/low-fps video Q: User question

∆res: Resolution boost factor ∆fps: Frame rate boost factor Output: A(2): Final refined answer

Initial Answer Generation: (A(1),C) ← M(Vlow,Q) Clue Processing & Upsampling:

Tc ← ExtractTemporalClues(C) Vseg ← SelectSegments(Vlow,Tc)

Vhigh ← Upsample(Vseg,∆res,∆fps) Final Answer Generation: A(2) ← M(Vhigh,Q) return A(2)

Chat with Enhanced Spatio-Temporal Perception. After enhancing the spatiotemporal perception capability of MLLMs, we can construct a more reliable video chat system. Specifically, after the model answers a user’s question, it can provide relatively accurate clues that support the answer to that question. We can further leverage these clues to improve the accuracy of the responses. Herein, we propose a simple “Temporal Clue-driven Reasoning” schema: after the model gives the first answer, we re-input the video segments corresponding to the obtained clues into the model at a higher resolution and frame rate, prompting it to answer again. By perceiving more details, the model can generate more accurate responses. Additionally, this operation is also well-suited for long video understanding under conditions of limited computing resources.

#### 4 Experiments

Implementation details. The main experiments are all conducted based on Qwen2.5-VL-7B [1] (except for the video captioning, for which Qwen2-VL-7B [26] is used).

Benchmarks. We employ MVBench [15], Perception Test [21], VideoMME [6] for evaluation of general video understanding. Given that the majority of videos in our training set are short-length, we only use the short subset of VideoMME in testing. For the temporal grounding task, we use the test set of Charade-STA [7] for in-domain testing and the test set of ActivityNet-Grounding [13] as out-domain test data. For the object tracking task, testing is done using the GoT-10k [10] dataset. For the QA and grounding QA tasks, the test set of NExTGQA [31] is used for testing. And we use Dream-1k [25] and VidTAB-QA [16] for the video captioning and video quality access.

##### 4.1 Evaluation of VideoChat-R1

Charades-STA ActivityNet NExTGQA GoT VideoMME MVBench Peception Test

Method

mIoU R@0.5 R@0.7 mIoU R@0.5 R@0.7 mIoU acc Overlap R@0.5 Short-Avg Avg Val

Baseline Qwen2.5-VL-7B 29.0 24.2 11.1 21.1 15.8 7.5 15.4 59.5 12.6 1.1 71.3 66.9 69.1

SFT on specific tasks

+SFT w/ Charades-STA 46.3 45.0 25.3 20.6 16.7 7.9 - - - - N/A* N/A* N/A*

+SFT w/ GoT - - - - - - - - 41.8 29.5 59.2 58.6 58.5 +SFT w/ NExTGQA - - - - - - 28.2 64.8 - - 60.1 59.2 60.7

GRPO on various tasks

VideoChat-R1 60.8 71.7 50.2 36.6 33.4 17.7 32.4 70.6 43.8 38.2 72.2 67.9 70.0 VideoChat-R1-thinking 59.9 70.6 47.2 35.5 33.3 16.7 36.1 69.2 43.3 33.9 74.2 66.2 69.6

Table 1: Results of VideoChat-R1 on various Video Benchmarks. * indicates that the model has suffered from overfitting and is unable to answer the question properly. Since the number of input pixels is fixed during our evaluation, the baseline results are slightly lower than those reported in their origin paper [1].

As shown in Table 1, after training with GRPO on spatio-temporal perception datasets, both VideoChat-R1 and VideoChat-R1-thinking significantly outperform the performance of Qwen2.5-VL and that of models fine-tuned through SFT for a single specific task across various spatiotemporal perception benchmarks and the general understanding benchmark VideoMME. This validates the effectiveness of our approach, which leverages multiple spatiotemporal perception datasets and RFT for enhancing spatiotemporal perception.

Meanwhile, we observe that for spatio-temporal perception tasks, engaging in thinking processes does not necessarily lead to performance gains. However, for tasks such as QA and VideoMME, which may require complex reasoning, conducting inferences during testing can result in notable performance improvements.

##### 4.2 Ablation Studies and Discussions

Muti-task Co-training. As shown in Table 2, we found that mixed training of different spatiotemporal perception tasks using GRPO can yield a synergistic improvement effect. Training with the multiple tasks achieves nearly the best results across all benchmarks. This reveals the potential of GRPO for larger-scale and multi-task collaborative training in the future.

Charades-STA ANet NExTGQA GoT VideoMME

Method

mIoU R@0.5 R@0.7 mIoU R@0.5 R@0.7 mIoU acc Overlap R@0.5 Short-Avg

Qwen2.5-VL-7B 29.0 24.2 11.1 21.1 15.8 7.5 15.4 59.5 12.6 1.1 71.3 +GRPO w/ STA 59.3 70.4 46.0 30.7 27.5 12.9 31.4 61.2 27.8 12.9 72.6 +GRPO w/GQA 36.0 33.5 15.5 24.9 20.6 10.7 35.1 68.7 36.1 26.7 72.0 +GRPO w/ GoT 28.7 25.1 9.6 20.1 16.2 6.8 15.6 60.5 42.5 30.6 71.4 +GRPO w/ STA-GQA 59.8 69.7 47.0 33.7 31.0 16.0 35.7 67.7 36.5 28.9 72.2 +GRPO w/ STA-GQA-GoT 60.8 71.7 50.2 36.6 33.4 17.7 32.4 70.6 43.8 38.2 72.2

##### Table 2: Ablation results of Cotraining on Spatio-Temporal Tasks.

Training Prompt Test Prompt Charades-STA (in domain) ActivityNet (out domain) VideoMME

Method Epochs

Think Answer Think Answer mIoU R@0.3 R@0.5 R@0.7 mIoU R@0.3 R@0.5 R@0.7 Short-Avg Vision Experts

FlashVTG [2] - - - - - - - 70.3 49.9 - - - - InternVideo2-6B [28] - - - - - - - 70.0 49.0 - - - - SG-DETR [8] - - 71.1 52.8 - - - - -

MLLMs

Qwen2.5-VL-7B - - - ✓ 29.0 44.7 24.2 11.1 21.1 28.3 15.8 7.4 71.3 (baseline) - - - ✓ ✓ 28.1 41.8 23.4 11.1 17.7 22.7 13.4 7.7 71.3

+ SFT 1 ✓ ✓ 46.3 63.9 45.0 25.3 20.6 30.2 16.7 7.9 N/A*(-71.3) 3 ✓ ✓ 34.6(+6.5) 51.7 36.3 20.6 17.3(-3.8) 26.1 10.0 3.9 N/A*(-71.3)

+ GRPO 1 ✓ ✓ 58.7 80.9 67.7 45.4 31.9 46.3 28.8 14.1 72.6

1 ✓ ✓ ✓ ✓ 59.3(+31.2) 81.7 70.4 46.0 30.7(+13.0) 45.0 27.5 12.9 73.6(+2.3)

3 ✓ ✓ ✓ ✓ 61.3(+33.2) 83.1 72.8 51.5 34.3(+16.6) 50.4 32.2 16.2 70.9(-0.4)

- Table 3: Ablation results of Temporal Grounding Task. * indicates that the model has suffered from overfitting and is unable to answer the question properly.

Method

GoT VideoMME

Average overlap R@0.5 R@0.7 Short-Avg

Qwen2.5-VL-7B 12.6 1.1 0 71.3 +SFT 41.8 29.5 3.9 59.2 +GRPO 42.5(+29.9) 30.6(+29.5) 3.9(+3.9) 71.4(+0.1)

- Table 4: Ablation results of Object Tracking. We use 8 frames as input for training and evaluation.

Temporal Grounding and Object tracking. As shown in Table 3 and Table 4, fine-tuning Qwen2.5VL using GRPO significantly improves the performance of temporal grounding and object tracking tasks. Additionally, it slightly enhances the performance on the general understanding benchmark VideoMME. Even when training for more epochs, GRPO is less prone to overfitting compared to SFT. Instead, it can continuously improve the performance of temporal grounding, eventually surpassing the performance of previous expert models. Moreover, stimulating the model’s thinking ability provides some benefits for both temporal grounding and VideoMME tasks.

Training Prompt Test Prompt NExTGQA VideoMME

Method

Think Answer Glue Think Answer Glue mIoU acc Short-Avg

Direct Output Qwen2.5-VL-7B ✓ - 41.7 71.3 (baseline) ✓ ✓ 15.4 59.5 -

+ SFT ✓ ✓ - 65.1 60.2

✓ ✓ ✓ ✓ 28.2(+12.8) 64.8(+5.3) 60.1(-11.2)

+ GRPO ✓ ✓ - 70.1 71.7 ✓ ✓ ✓ 16.2 70.2 71.7

✓ ✓ ✓ ✓ 35.1(+19.7) 68.7(+9.2) 72.0(+0.7)

Chain-of-thought Output Qwen2.5-VL-7B ✓ ✓ - 47.7 73.0

✓ ✓ ✓ 20.2 53.3 72.2 + GRPO ✓ ✓ ✓ ✓ - 68.8 74.7

✓ ✓ ✓ ✓ ✓ ✓ 32.9(+12.7) 66.9(+13.6) 75.3(+3.1)

##### Table 5: Ablation results of Multi-Choice Video QA.

Video Question Answer. As shown in Table 5, for the video question answering task, we selected the multi-choice QA task, which is easy to evaluate, for our experiments. Additionally, we explored the grounding QA task. In this task, when answering questions, the model is required to simultaneously provide the temporal cues on which its answers are based. Using merely a little over three thousand training data samples, we found that GRPO demonstrated remarkable fine-tuning capabilities. Not only did it lead to a substantial improvement in the performance of the NExTGQA task, but it also brought about a noticeable enhancement in the VideoMME task. We noticed that, unlike the previous strongly spatiotemporal perception tasks such as temporal grounding, thinking played a significant role in the QA task. Meanwhile, the glue signals also provided some assistance for relatively complex video understanding tasks like VideoMME.

Dream-1k VidTAB-QA

Method

F1 Precision Recall Accuracy

Baseline 30.6 33.8 27.9 70.7 + SFT 31.4 32.6 30.2 71.7 + GRPO 38.2(+7.6) 45.4(+11.6) 33.1(+5.2) 72.6(+1.9)

##### Table 6: Results of Video Caption and Video Quality Access.

###### Dream-1k

LLM Judge

Method

F1 Precision Recall Baseline - 30.6 33.8 27.9

GPT-3.5-turbo-0125 37.9(+7.3) 44.4(+10.6) 33(+5.1)

+ GRPO

Qwen2.5-72B 38.2(+7.6) 45.4(+11.6) 33.1(+5.2)

##### Table 7: Ablation of Video Caption Task.

Video Caption and Video Quality Assessment. For the Video Caption and Video Quality Assessment tasks, we found that GRPO still demonstrated its advantages over SFT, As shown in Table 6. The significant metric improvements on these two benchmarks demonstrate the effectiveness of our approach.

Ablation of Reward Evaluators To assess the impact of different large language models (LLMs) as reward evaluators, we conducted parallel experiments using GPT-3.5-turbo-0125 and Qwen2.5-72B as distinct judges (Table 7). Models trained under both evaluators achieved nearly identical performance, demonstrating consistent caption reward generation across LLMs. We attribute this consistency to GRPO’s fundamental mechanism: it relies on relative differential scoring within response groups rather than absolute reward values. This confirms that Rrecall produces discriminative reward signals for predicted captions independent of the choice of LLM judge, validating both the efficacy of our reward design and the stability of its signaling mechanism. Crucially, when guided by these reliable

reward signals, our approach delivers substantial performance gains in description tasks using only limited high-quality data, demonstrating remarkable data-efficiency and significant optimization potential.

Model Clue VideoMME LongVideoBench Avg. Duration Perception 1010s 473s

|Qwen2.5-VL-7B|✓|64.4 56.0<br><br>63.3(-1.1) 55.2(-0.8)|
|---|---|---|
|VideoChat-R1-thinking<br><br>|✓|62.1 51.9<br><br>63.6(+1.5) 58.2(+6.3)<br>|

Table 8: Ablation results of “Clue-Perception”. It should be noted that due to our adoption of a lower number of input pixels, the absolute performance is not entirely consistent with that reported for Qwen2.5-VL.

Ablation of “Clue-Perception” As shown in Table. 8, we compared the performance changes of the model with and without perception enhancement when applying the ”Clue-Perception” strategy on two representative long video benchmarks [6, 30]. It is noteworthy that without the use of ”CluePerception”, VideoChat-R1 showed no significant performance improvement over Qwen2.5-VL-7B in long video tasks, which can be attributed to the fact that our training dataset consists entirely of short videos under 1 minute. However, after the application of the ”Clue-Perception” operation, VideoChatR1 demonstrated a significant performance enhancement, indicating that the clues it provides are more accurate and thus revealing the potential of clue-perception in long video understanding. In contrast, due to its insufficient spatiotemporal perception capability, Qwen2.5-VL-7B even exhibited a performance decline after the implementation of the ”Clue-Perception” operation.

GRPO vs. SFT. It can be observed that across various types of tasks, GRPO outperforms SFT. Whether it is in terms of the performance on in-domain tasks, out-domain tasks, or the preservation of the original general performance, our experimental results demonstrate that GRPO is a promising fine-tuning approach. We will leave the large-scale comparison for future research.

Chain-of-thought vs. Direct Output. Based on the video tasks and experiments we have explored, which focus on spatiotemporal perception, the output of the chain of thought has not demonstrated obvious advantages. In some cases, it is even inferior to the direct output. We believe that how to define appropriate video reasoning tasks and evaluation methods remains to be explored. The existing training data is insufficient to activate the model to output truly effective video reasoning chains.

## Question: VideoChat-R1:

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Video:

Query: A person puts on a pair of shoes. Ground Truth: 1.8s 9.4s SFT: 1.3s 4.1s

VideoChat-R1: 2.0s 10.7s

<think>\nThe man is seen standing in front of a pantry, looking down at something on the floor. He then bends down and picks up a pair of shoes from the floor. He then proceeds to put on the shoes. The event \"a person puts on a pair of shoes\" occurs during this time. \n</think>\n<timestep>\nThe event happens in 2.00 - 10.7 seconds.\n</timestep>\n<answer>\n 2.00 to 10.67\n</answer>

- Figure 2: Examples on temporal grounding task. VideoChat-R1 gives a more accurate time interval after thinking.

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

[Figure 48]

[Figure 49]

[Figure 50]

Video:

8

Query: Person puts it back on the table. Ground Truth: 25.7s 32.0s SFT: 13.5s 17.1s VideoChat-R1: <think>\nThe person is lying on the couch and appears to be reading a book. They then place the book on the coffee table in front of them. The event \"person puts it back on the table\" occurs when they place the book on the table.\n</think>\n\n<answer>\n26.90 to 32.00\n</answer>

26.9s 32.0s

(2)

|Question: Why did the girl suddenly jump up from where she was standing? Ground Truth: Excited. SFT: VideoChat-R1: <think> The girl jumps up from her position after playing with the rocket toy and then walks around the room holding it. This suggests that she is excited about the toy and wants to show it off. The jump could be interpreted as an expression of excitement. </think> <answer>E</answer> <clue>[(20.1, 28.0)]</clue><br><br>Video:<br><br>[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>Dance. [19.4s, 37.2s]<br><br>[20.9s, 25.0s]<br><br>Excited. [20.1s, 28.0s]<br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]|
|---|

- Figure 3: Examples on Video QA task. It can be seen that VideoChat-R1 can not only answer questions correctly but also provide relatively accurate reference time periods (clue).

##### 4.3 Qualitative Results

As shown in Figure 2 and 3, we provide visualizations of VideoChat-R1’s outputs for the temporal grounding and video QA tasks. We observe that due to the enhanced spatiotemporal perception capability, VideoChat-R1 can more accurately locate video segments relevant to the question and conduct reasonable reasoning to answer the question. However, compared with the complex chains of thought involved in solving mathematical problems or coding tasks in text and image domains, the chains of thought generated in our current video tasks are relatively simple. We will explore more complex video reasoning tasks in future research.

#### 5 Conclusions

In this work, we systematically investigate the role of reinforcement fine-tuning (RFT) with Group Relative Policy Optimization (GRPO) in enhancing video-centric multimodal large language models (MLLMs). Our experiments demonstrate that RFT is a highly data-efficient paradigm for taskspecific improvements, enabling VideoChat-R1—a model trained with limited samples via multi-task RFT—to achieve state-of-the-art performance on spatio-temporal perception tasks while preserving general chat capabilities and exhibiting emergent spatiotemporal reasoning. We believe our work can present relevant insights for future research efforts in reinforcement learning of video MLLMs.

#### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [2] Zhuo Cao, Bingqing Zhang, Heming Du, Xin Yu, Xue Li, and Sen Wang. Flashvtg: Feature layering and adaptive score handling network for video temporal grounding. arXiv preprint arXiv:2412.13441, 2024.
- [3] Huilin Deng, Ding Zou, Rui Ma, Hongchen Luo, Yang Cao, and Yu Kang. Boosting the generalization and reasoning of vision language models with curriculum reinforcement learning. arXiv preprint arXiv:2503.07065, 2025.
- [4] Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, and Kai-Wei Chang. Openvlthinker: An early exploration to complex vision-language reasoning via iterative self-improvement. arXiv preprint arXiv:2503.17352, 2025.
- [5] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025.
- [6] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024.
- [7] Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia. Tall: Temporal activity localization via language query. In Proceedings of the IEEE international conference on computer vision, pages 5267–5275, 2017.
- [8] Aleksandr Gordeev, Vladimir Dokholyan, Irina Tolstykh, and Maksim Kuprashevich. Saliency-guided detr for moment retrieval and highlight detection. arXiv preprint arXiv:2410.01615, 2024.
- [9] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [10] Lianghua Huang, Xin Zhao, and Kaiqi Huang. Got-10k: A large high-diversity benchmark for generic object tracking in the wild. IEEE transactions on pattern analysis and machine intelligence, 43(5): 1562–1577, 2019.
- [11] Zhenpeng Huang, Xinhao Li, Jiaqi Li, Jing Wang, Xiangyu Zeng, Cheng Liang, Tao Wu, Xi Chen, Liang Li, and Limin Wang. Online video understanding: A comprehensive benchmark and memory-augmented method. arXiv preprint arXiv:2501.00584, 2024.
- [12] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.
- [13] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In Proceedings of the IEEE international conference on computer vision, pages 706–715, 2017.
- [14] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023.
- [15] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195–22206, 2024.
- [16] Xinhao Li, Zhenpeng Huang, Jing Wang, Kunchang Li, and Limin Wang. Videoeval: Comprehensive benchmark suite for low-cost evaluation of video foundation model. arXiv preprint arXiv:2407.06491, 2024.
- [17] Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan Zhu, Haian Huang, Jianfei Gao, Kunchang Li, Yinan He, Chenting Wang, et al. Videochat-flash: Hierarchical compression for long-context video modeling. arXiv preprint arXiv:2501.00574, 2024.
- [18] Yuqi Liu, Bohao Peng, Zhisheng Zhong, Zihao Yue, Fanbin Lu, Bei Yu, and Jiaya Jia. Seg-zero: Reasoningchain guided segmentation via cognitive reinforcement. arXiv preprint arXiv:2503.06520, 2025.
- [19] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual-rft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025.

- [20] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023.
- [21] Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Mateusz Malinowski, Yi Yang, Carl Doersch, et al. Perception test: A diagnostic benchmark for multimodal video models. Advances in Neural Information Processing Systems, 36: 42748–42761, 2023.
- [22] Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai Yang, Xingzhong Xu, Xin Geng, and Xu Yang. Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl. arXiv preprint arXiv:2503.07536, 2025.
- [23] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [24] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.
- [25] Jiawei Wang, Liping Yuan, Yuchen Zhang, and Haomiao Sun. Tarsier: Recipes for training and evaluating large video description models. arXiv preprint arXiv:2407.00634, 2024.
- [26] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- [27] Ye Wang, Boshen Xu, Zihao Yue, Zihan Xiao, Ziheng Wang, Liang Zhang, Dingyi Yang, Wenxuan Wang, and Qin Jin. Timezero: Temporal video grounding with reasoning-guided lvlm. arXiv preprint arXiv:2503.13377, 2025.
- [28] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Zun Wang, Yansong Shi, et al. Internvideo2: Scaling foundation models for multimodal video understanding. In European Conference on Computer Vision, pages 396–416. Springer, 2024.
- [29] Yi Wang, Xinhao Li, Ziang Yan, Yinan He, Jiashuo Yu, Xiangyu Zeng, Chenting Wang, Changlian Ma, Haian Huang, Jianfei Gao, et al. Internvideo2. 5: Empowering video mllms with long and rich context modeling. arXiv preprint arXiv:2501.12386, 2025.
- [30] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. Advances in Neural Information Processing Systems, 37: 28828–28857, 2024.
- [31] Junbin Xiao, Angela Yao, Yicong Li, and Tat-Seng Chua. Can i trust your answer? visually grounded video question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13204–13214, 2024.
- [32] Yifan Xu, Xinhao Li, Yichun Yang, Rui Huang, and Limin Wang. Fine-grained video-text retrieval: A new benchmark and method. arXiv preprint arXiv:2501.00513, 2024.
- [33] Ziang Yan, Zhilin Li, Yinan He, Chenting Wang, Kunchang Li, Xinhao Li, Xiangyu Zeng, Zilei Wang, Yali Wang, Yu Qiao, et al. Task preference optimization: Improving multimodal large language models with vision task alignment. arXiv preprint arXiv:2412.19326, 2024.
- [34] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.
- [35] Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615, 2025.
- [36] En Yu, Liang Zhao, Yana Wei, Jinrong Yang, Dongming Wu, Lingyu Kong, Haoran Wei, Tiancai Wang, Zheng Ge, Xiangyu Zhang, et al. Merlin: Empowering multimodal llms with foresight minds. In European Conference on Computer Vision, pages 425–443. Springer, 2024.
- [37] Xiangyu Zeng, Kunchang Li, Chenting Wang, Xinhao Li, Tianxiang Jiang, Ziang Yan, Songze Li, Yansong Shi, Zhengrong Yue, Yi Wang, et al. Timesuite: Improving mllms for long video understanding via grounded tuning. arXiv preprint arXiv:2410.19702, 2024.

- [38] Yufei Zhan, Yousong Zhu, Shurong Zheng, Hongyin Zhao, Fan Yang, Ming Tang, and Jinqiao Wang. Vision-r1: Evolving human-free alignment in large vision-language models via vision-guided reinforcement learning. arXiv preprint arXiv:2503.18013, 2025.
- [39] Yufei Zhan, Yousong Zhu, Shurong Zheng, Hongyin Zhao, Fan Yang, Ming Tang, and Jinqiao Wang. Vision-r1: Evolving human-free alignment in large vision-language models via vision-guided reinforcement learning. arXiv preprint arXiv:2503.18013, 2025.
- [40] Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao. R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization. arXiv preprint arXiv:2503.12937, 2025.
- [41] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024.
- [42] Jiaxing Zhao, Xihan Wei, and Liefeng Bo. R1-omni: Explainable omni-multimodal emotion recognition with reinforcement learning. arXiv e-prints, pages arXiv–2503, 2025.
- [43] Hengguang Zhou, Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, and Cho-Jui Hsieh. R1-zero’s” aha moment” in visual reasoning on a 2b non-sft model. arXiv preprint arXiv:2503.05132, 2025.

