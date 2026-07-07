# arXiv:2504.00883v2[cs.CV]14Apr2025

## Improved Visual-Spatial Reasoning via R1-Zero-Like Training

Zhenyi Liao1,∗Qingsong Xie2†, Yanhao Zhang2, Zijian Kong1,Haonan Lu2, Zhenyu Yang2, Zhijie Deng1†

1Shanghai Jiao Tong University 2OPPO AI Center {l-justice, kong-zijian, zhijied}@sjtu.edu.cn {xieqingsong,zhangyanhao,luhaonan,yangzhenyu}@oppo.com

### Abstract

Increasing attention has been placed on improving the reasoning capacities of multi-modal large language models (MLLMs). As the cornerstone for AI agents that function in the physical realm, video-based visual-spatial intelligence (VSI) emerges as one of the most pivotal reasoning capabilities of MLLMs. This work conducts a first, in-depth study on improving the visual-spatial reasoning of MLLMs via R1-Zero-like training. Technically, we first identify that the visual-spatial reasoning capacities of small- to mediumsized Qwen2-VL models cannot be activated via Chain of Thought (CoT) prompts. We then incorporate GRPO training for improved visual-spatial reasoning, using the carefully curated VSI-100k dataset, following DeepSeek-R1-Zero. During the investigation, we identify the necessity to keep the KL penalty (even with a small value) in GRPO. With just 120 GPU hours, our vsGRPO-2B model, fine-tuned from Qwen2-VL-2B, can outperform the base model by 12.1% and surpass GPT-4o. Moreover, our vsGRPO-7B model, fine-tuned from Qwen2-VL-7B, achieves performance comparable to that of the best open-source model LLaVA-NeXT-Video-72B. Additionally, we compare vsGRPO to supervised fine-tuning and direct preference optimization baselines and observe strong performance superiority. The code and dataset will be available at https://github.com/zhijie-group/R1-Zero-VSI.

### 1 Introduction

Multimodal Large Language Models (MLLMs) have emerged as a significant advancement in AI. They typically accept text, images, and videos as inputs and output textual responses, serving as the foundations for various applications, including multi-modal understanding (Liu et al., 2023; Wang et al., 2024a), visual language agents (Hong et al., 2024; Wang et al., 2024b), autonomous driving (Pan et al., 2024; Wang et al., 2024c), etc.

The exhaustive understanding of multi-modal observations hinges on advanced reasoning capabilities, which has spurred growing interest in investigating reasoning mechanisms within MLLMs (Sur´ıs et al., 2023; Xu et al., 2024; Zhang et al., 2025). This trend mirrors concurrent advancements in vanilla LLMs (Yao et al., 2023; Lightman et al., 2023; Wei et al., 2022; Wang et al.). As the foundation for AI agents operating in the physical world, the video-based visual-spatial reasoning stands out as one of the most crucial capacities of MLLMs. Yet, existing models often fall short in this (see VSI-bench (Yang et al., 2024)).

This work conducts a systematic study on improving the visual-spatial reasoning capacities of MLLMs based on R1-Zero-like training. Focusing on the Qwen2-VL models (Wang et al., 2024a), we first perform an initial study on whether simple reasoning-oriented prompts can activate the visual-spatial reasoning capacities. We investigate various CoT (Wei et al., 2022) strategies, but find that vanilla non-CoT prompts perform the best for small- to medium-sized Qwen2-VL on VSI-bench. This exposes the issue that such models does not have the ability to trade inference FLOPs for improved visual-spatial reasoning.

∗Works done during the internship in OPPO AI Center. †Corresponding authors.

Appr. Order

Obj. Count

Obj. Size

Route Plan

Room Size

Abs. Dist.

Rel. Dist.

Rel. Dir.

Backbone Methods Avg

Think-mode 22.9 18.4 4.3 31.5 17.3 28.3 22.9 26.2 16.8 Observe-mode 21.8 16.8 1.7 32.7 22.7 28.8 27.6 26.2 18.1 Vanilla-mode 23.3 21.4 3.4 32.3 31.1 26.7 27.7 24.7 18.9

Qwen2VL-2B

Think-mode 31.3 44.8 26.1 25.3 23.4 34.7 30.9 32.9 31.5 Observe-mode 32.0 29.9 19.0 39.6 32.0 34.6 40.0 36.0 24.4 Vanilla-mode 32.2 39.4 25.0 25.8 43.2 32.6 30.9 27.8 32.6

Qwen2VL-7B

- Table 1: Quantitative comparisons of different prompting strategies on Qwen2-VL-2B and Qwen2-VL-7B on VSI-bench.

Following the journey of DeepSeek-R1-Zero (Guo et al., 2025), we decide to improve the visual-spatial reasoning capacities of Qwen2-VL with GRPO (Shao et al., 2024), sharing a similar task with the very recent work VisualThinker-R1-Zero (Zhou et al., 2025). Considering the scarcity of training data for visual-spatial question answering, we construct a video-based question answering dataset of more than 100k samples, VSI-100k, following the protocol of VSI-bench. Specifically, we leverage ScanNet (Dai et al., 2017) to get high-fidelity video scans accompanied by meticulous object-level 3D annotations, based on which (question, answer) pairs regarding spatial information can be easily crafted.

Following common practice (Guo et al., 2025; Chen et al., 2025; Zhou et al., 2025), we define the rule-based reward function based on the alignment between the model prediction and the ground-truth answer to perform GRPO. We also include a format reward when trying to activate the CoT reasoning behaviour. The GRPO on VSI-100k turns the pretrained Qwen2-VL-2B model into the performant vsGRPO-2B within just 120 GPU hours. We observe that vsGRPO-2B outperforms the base model by 12.1%. The same pipeline also transforms the Qwen2-VL-7B model into vsGRPO-7B, achieving performance similar to that of the best open-source model with 72B parameters. During GRPO training, we have identified the necessity to keep the KL penalty (even with a small value) in the training of GRPO and observed phenomena such as reward hacking. We also compare GRPO with supervised fine-tuning (SFT) and direct preference optimization (DPO) (Rafailov et al., 2023), and confirm the superiority of GRPO in improving the spatial-visual reasoning capacities of Qwen2-VL.

### 2 Can Visual-spatial Reasoning Capacities Be Activated by Prompting?

We initiate by evaluating Qwen2-VL on the VSI-bench with various prompting strategies. Concretely, the VSI-bench includes two types of question-answer problems:

- • Numerical Answer (NA), including tasks such as object counting, absolute distance measurement, object size evaluation, and room size assessment;
- • Multiple-Choice Answer (MCA), including tasks related to relative distance, relative direction, route planning, and appearance order.

To evaluate the reasoning capacities of Qwen2-VL on this dataset, we consider two CoT prompting strategies: the widely adopted think-mode, where the model first thinks and then replies to the question, and the observe-mode, where the model first observes the input video and then replies. The latter follows a human-like pattern and has been explored in related works (Wu et al., 2024). We also include a non-CoT vanilla-mode for comparison. Here is a summarization of them:

• Think-mode: Let’s think step by step and then answer the question using a single word or phrase.

[Figure 1]

Question: Measuring from the closest point of each object, which of these objects (stove, tv, table, sofa) is the closest to the stool? Options: A. stove B. tv C. table D. sofa

Ground Truth: D Vanilla-mode: C Think-mode:

[Figure 2]

To determine which object is the closest to the stool, let's analyze the positions of the objects in the video:

- 1. The stove is located on the left side of the kitchen.
- 2. The TV is located on the right side of the kitchen.
- 3. The table is located in the center of the kitchen.
- 4. The sofa is not visible in the video. Given this information, the closest object to the stool is the table. So, the correct answer is: C. table.

[Figure 3]

[Figure 4]

Figure 1: Comparison between the vanilla-mode and think-mode predictions.

- • Observe-mode: Please observe the video first and then answer the question using a single word or phrase.
- • Vanilla-mode: Please answer the question using a single word or phrase.

CoT prompting is ineffective for small- to medium-sized Qwen2-VL on VSI-bench. As shown in Table 1, despite longer responses, think-mode and observe-mode underperform the simple vanilla-mode. Namely, small- to medium-sized Qwen2-VL cannot trade inference FLOPs for improved visual-spatial reasoning.

We visualize some output examples given by Qwen2-VL-2B in Figure 1. We see that the model can actually understand the instructions for activating thinking, but the final answer is still wrong, the same as that of the vanilla prompting. From the exposed chain of thoughts, we realize that the error may arise from the failure to perceive the sofa in the video.

### 3 R1-Zero-like Training for Visual-spatial Reasoning

Given the above observations, we realize it is necessary to fine-tune the Qwen2-VL models for improved visual-spatial reasoning. Typically, we opt to focus on the GRPO approach (Shao et al., 2024) given its success in building DeepSeek-R1-Zero.

#### 3.1 Training Data Construction

We first create a video-based question-answering dataset named VSI-100k for visual-spatial reasoning. It consists of more than 100k samples and follows the VSI-bench protocol. Specifically, we utilize ScanNet (Dai et al., 2017) to obtain high-fidelity video scans that come with detailed object-level 3D annotations. With such information, it becomes straightforward to generate (question, answer) pairs related to spatial information.

Specifically, we construct questions regarding six topics, including object count, relative direction, relative distance, object size, room size, and absolute distance. We leave the other two topics in VSI-bench, route planning and appearance order, held out. This corresponds to two reasons: 1) (question, answer) pairs of the latter two topics cannot be simply constructed given rarely the static 3D information, which implies that expensive manual annotation can be required; 2) with this, we can test the task generalization ability of the trained models. For the NA type problems, we implement a question template similar to that used in (Yang et al., 2024). For the MCA one, we simplify the question format by removing the options. This adjustment enhances the model’s capacity to recognize entity correspondence instead of simply matching symbols. Some examples are provided in Figure 2.

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

###### Relative Direction

###### Object Count

Question: If I am standing by the shelf and facing the shower is the bicycle to the left or the right of the shower?

Question: How many trash can(s)in the room? Answer: 3

Answer: left

###### Object Size

###### Absolute Distance

Question: What is the length of the longest dimension (length,width, or height) of the coffee table, measured in centimeters?

Question: What is the distance between the shower and the kitchen

counter (in meters)?

Answer: 113

Answer: 6.1

###### Relative Distance

###### Room Size

Question: Which of these objects (sink, pillow, bed, guitar)is the closest to the bicycle?

Question: What is the size of this room (in square meters)? Answer: 47.9

Answer: sink

Figure 2: Illustrations of the constructed dataset.

For object count, we construct answers with directly the object labels included in the annotations document, yielding a total of 6.4k samples. For absolute distance, we first remove objects that appear multiple times to ensure specification preciseness, and then calculate the distance between geometric centers of various 3D point-cloud objects, obtaining 75k samples. For relative distance, we fix one targeted object and compute the absolute distance between it and four other objects to estimate relative distance, yielding 13k samples. For relative direction, we select one object as the front and determine the relative direction of two objects based on their geometric centers of point clouds, getting a total of 8k samples. For object size, we leverage the 3D bounding box to compute the longest dimension of the object, yielding a total of 13k samples. For room size, we apply the alpha shape algorithm to the total 1.5k scenes, resulting in 1.5k samples.

#### 3.2 GRPO

Group Relative Policy Optimization, also abbreviated as GRPO (Shao et al., 2024), is a type of reinforcement learning (RL) that eliminates the critic model to reduce training costs. Specifically, a group of generated output set {o1, o2, · · · , oG} is sampled for each question q from policy model πθold. Then GRPO optimizes the model πθ using the following objective:

JGRPO(θ) = Eq∼P(Q),{o

i}iG=1∼πθold(O|q)

G

1 G

πθ(oi | q) πθold(oi | q)

∑

Ai, clip

min

i=1

πθ(oi | q) πθold(oi | q)

,1 − ε, 1 + ε Ai − β DKL πθ πref ,

(1)

where ε and β are the clipping hyper-parameter and the coefficient controlling the Kullback–Leibler (KL) penalty, respectively, and Ai = ri−stdmean({r({r1,r2,...,rG})

1,r2,...,rG}) is the computed advantage using the group rewards {r1,r2, · · · ,rG}. DKL πθ ∥ πref = ππref(oi|q)

θ(oi|q) − log ππref(oi|q)

θ(oi|q) −

- 1 is the KL divergence.

The reward r guides the direction of the training process and is crucial. We adhere to (Chen et al., 2025; Meng et al., 2025) of using format rewards and accuracy rewards, but with necessary modifications.

Appr. Order

Obj. Count

Obj. Size

Eval. Mode Avg

Route Plan

Room Size

Abs. Dist.

Rel. Dist.

Rel. Dir.

Methods

#### Open-source

Qwen2-VL-2B V 23.3 21.4 3.4 32.3 31.1 26.7 27.7 24.7 18.9 + SFT V 29.6 29.6 23.5 47.4 33.5 26.9 28.3 28.8 18.6 + DPO V 23.9 21.7 3.7 34.8 32.4 27.1 28.5 24.2 18.6 + vsGRPO-T V 26.1 24.7 10.7 37.4 36.2 27.3 29.5 25.7 17.9 + vsGRPO-O V 28.0 26.2 16.4 44.8 38.2 27.0 29.3 24.2 18.2 + vsGRPO-T T 29.6 35.0 28.2 34.7 25.2 28.0 38.5 28.5 18.7 + vsGRPO-O O 31.2 34.6 22.5 44.8 33.7 29.4 41.8 26.8 15.8 + vsGRPO-V V 35.4 53.6 29.0 52.7 43.4 28.1 30.9 26.8 18.9

Qwen2-VL-7B V 32.2 39.4 25.0 25.8 43.2 32.6 30.9 27.8 32.6 + SFT V 38.1 44.7 27.6 46.1 50.4 34.0 35.7 33.0 33.4 + DPO V 32.6 39.1 25.2 26.5 44.2 32.6 30.9 29.3 33.3 + vsGRPO-V V 40.7 59.9 29.6 50.8 48.3 35.4 35.6 34.0 31.5

IVL2-2B V 27.4 21.8 24.9 22.0 35.0 33.8 44.2 30.5 7.1 LNV-7B V 35.6 48.5 14.0 47.8 24.2 43.5 42.4 34.0 30.6 IVL2-40B V 36.0 34.9 26.9 46.5 31.8 42.1 32.2 34.0 39.6 LNV-72B V 40.9 48.9 22.8 57.4 35.3 42.4 36.7 35.0 48.6

Close-source

GPT-4o V 34.0 46.2 5.3 43.8 38.2 37.0 41.3 31.5 28.5 Gemini-1.5 Pro V 48.8 49.6 28.8 58.6 49.4 46.0 48.1 42.0 68.0

- Table 2: Quantitative results on VSI-bench. vsGRPO-T, vsGRPO-O, and vsGRPO-V refer to GRPO training on VSI-100k with prompts of think-mode, observe-mode, and vanilla-mode respectively. V, T, and O in the Eval. Mode column refer to using vanilla-mode, think-mode, and observe-mode prompts for evaluation, respectively. We also present the best performance of open-source models under the specific model size, like LLaVA-NeXT-Video (Li et al., 2024) (LNV for short) and InternVL2 (Chen et al., 2024) (IVL2 for short), and closesource ones like GPT-4o (Hurst et al., 2024) and Gemini-1.5 Pro (Team et al., 2024).

Format Reward. Although the CoT prompts are useless for the small-sized Qwen2-VL-2B in inference time, we still wonder if training with them is beneficial for GRPO. As a result, following recent progress in the community, we consider three training prompts for GRPO:

- • Think-mode: Please think step by step and enclose your thinking process in <think> </think> tags and then provide the short answer with one or two words or a number in <answer> </answer>.
- • Observe-mode: Please observe carefully and analyze what you see that helps you to solve the question in the video and enclose it in <observe> </observe> tag, and then provide the short answer with one or two words or a number in <answer> </answer>.
- • Vanilla-mode: Please provide the short answer with one or two words or a number.

The format reward quantifies how the responses follow the specified format. It returns a score of 1 or 0. Note that such a reward is omitted for the vanilla-mode.

Accuracy Reward. In the case of non-NA tasks, we employ a character matching method to assess accuracy, awarding a score of 1 for a match and 0 for a mismatch. For NA tasks, we develop a function that computes the absolute difference between the true value and the predicted one and divides the result by the minimum of the two values.

[Figure 17]

[Figure 18]

Figure 3: Left: the format reward curve of β = 0 and β = 0.0001 during training. Right: the curve of the format reward, the accuracy reward curve of one subtask of VSI-100k, and the total reward curve during GRPO training.

Experimental Settings. Unless specified otherwise, we use Qwen2-VL-2B/7B as the base models due to resource constraints. We employ LoRA (Hu et al., 2022) training with a learning rate of 10−5 for Qwen2-VL-2B and 5 × 10−6 for Qwen2-VL-7B. We conduct 14 rollouts per question and set the default sampling temperature to 1. The KL divergence coefficient β is set to 0.0001.

- 3.3 Results and Analyses

- 3.3.1 Main Results

Let vsGRPO-T, vsGRPO-O, and vsGRPO-V denote the GRPO training on VSI-100k with prompts of think-mode, observe-mode, and vanilla-mode respectively. We evaluate them with the corresponding test prompts by default. Given the studies in the previous section, we also test the trained models with vanilla-mode prompts.

As shown in Table 2, for models based on Qwen-VL-2B, all GRPO fine-tuned models improve over the baseline. Besides, for the models trained with CoT prompting strategies, their CoT test performance outperforms vanilla one. This indicates that GRPO training can effectively enhance the model’s long reasoning capabilities. Notably, directly applying the vanilla-mode prompting strategy yields the best performance improvements, particularly for NA questions, and even outperforms GPT-4o. We refer to this model as vsGRPO-2B by default. This underscores the conclusion that CoT prompting is ineffective for the small-sized Qwen2-VL-2B on the VSI-bench.

In terms of Qwen2-VL-7B, we only tried vsGRPO-V considering the above results. We observe that vsGRPO-V performs the best on two subtasks—object counting and absolute distance. Moreover, the test performance on the Route Plan is also improved, similar to the 2B case. This is possibly because the Route Plan can be divided into sub-tasks that include relative direction, indicating inter-task generalization. With only 7B model size, we note that our model shows performance comparable to that of the leading open-source model, LLaVA-NeXT-Video-72B (Li et al., 2024).

- 3.3.2 Importance of KL Penalty

The KL penalty term plays a crucial role in regulating the divergence between the online policy and the frozen reference one. It avoids the model straying too far from the initial point. While some works (Yu et al., 2025; Meng et al., 2025) advocate for removing the KL penalty to enhance performance, we have observed that doing so can easily lead to training collapse, as illustrated in Figure 3 (left). In contrast, introducing a positive β (even very small, such as 0.0001) can effectively address this issue. This may be attributed to the specific nature of VSI reasoning problems.

#### 3.3.3 Reward Hacking

During the training, we observed that the model sometimes finds ways to obtain high rewards that do not align with our intentions. One example is that when training with observe-mode, there are some extreme samples in the rollouts, e.g., <think> </think> <answer>xx</answer>. While this format is technically correct, it degrades to a missing observation, a similar phenomenon also noted in VisualThinker-R1-Zero (Zhou et al., 2025). Observing this, we opt to incorporate a length reward function for mitigation. However, we note that some new generations just add extra <think></think> and <answer></answer> tags to exploit the length reward, which does not contribute to a meaningful thinking process. So, more reasonable reward functions should be explored.

#### 3.3.4 Dynamics of Various Rewards

As shown in Figure 3 (right), during the GRPO training, the format reward converges to 1 quickly in the early stage, while the accuracy reward increases slowly. It seems that there is an upper bound on the accuracy reward. How to address this remains an open problem.

#### 3.3.5 Comparison to Other Training Approaches

We also compare our approach with commonly used fine-tuning algorithms, supervised fine-tuning (SFT) and direct preference optimization (DPO) (Rafailov et al., 2023), in Table 2. For SFT, we directly use the constructed VSI-100k for tuning. For DPO, the correct answer is modified to a wrong one to serve as the less-preferred answer.

As shown, the two approaches both improve over the base model on the VSI-bench, but still lag behind GRPO-V. Besides, the improvement of DPO is minor, which is perhaps because of the sub-optimal preference pair construction.

### 4 Conclusion

In this work, we center on the video-based visual-spatial intelligence of MLLMs. Using Qwen2-VL as the base models, we identify that visual-spatial reasoning capacities of Qwen2VL-2B/7B cannot be activated via CoT prompts. We construct VSI-100k to combat data scarcity and adapt GRPO training. Extensive experiments demonstrate that vsGRPO-2B and vsGRPO-7B outperform models of the same size, highlighting the superiority of the GRPO approach in comparison to SFT and DPO. We also share some findings for future research.

### References

Liang Chen, Lei Li, Haozhe Zhao, Yifan Song, and Vinci. R1-v: Reinforcing super generalization ability in vision-language models with less than $3. https://github.com/ Deep-Agent/R1-V, 2025. Accessed: 2025-02-02.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 24185–24198, 2024.

Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5828–5839, 2017.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui

agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14281–14290, 2024.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, et al. Mm-eureka: Exploring visual aha moment with rule-based large-scale reinforcement learning. arXiv preprint arXiv:2503.07365, 2025.

Chenbin Pan, Burhaneddin Yaman, Tommaso Nesti, Abhirup Mallik, Alessandro G Allievi, Senem Velipasalar, and Liu Ren. Vlp: Vision language planning for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14760–14769, 2024.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2023.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

D´ıdac Sur´ıs, Sachit Menon, and Carl Vondrick. Vipergpt: Visual inference via python execution for reasoning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 11888–11898, 2023.

Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024a.

Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy. Videoagent: Long-form video understanding with large language model as agent. In European Conference on Computer Vision, pp. 58–76. Springer, 2024b.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations.

Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, and Zhaoxiang Zhang. Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14749–14759, 2024c.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Yixuan Wu, Yizhou Wang, Shixiang Tang, Wenhao Wu, Tong He, Wanli Ouyang, Philip Torr, and Jian Wu. Dettoolchain: A new prompting paradigm to unleash detection ability of mllm. In European Conference on Computer Vision, pp. 164–182. Springer, 2024.

Guowei Xu, Peng Jin, Li Hao, Yibing Song, Lichao Sun, and Li Yuan. Llava-o1: Let vision language models reason step-by-step. arXiv preprint arXiv:2411.10440, 2024.

Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. arXiv preprint arXiv:2412.14171, 2024.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36:11809–11822, 2023.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao. R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization. arXiv preprint arXiv:2503.12937, 2025.

Hengguang Zhou, Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, and Cho-Jui Hsieh. R1-zero’s” aha moment” in visual reasoning on a 2b non-sft model. arXiv preprint arXiv:2503.05132, 2025.

