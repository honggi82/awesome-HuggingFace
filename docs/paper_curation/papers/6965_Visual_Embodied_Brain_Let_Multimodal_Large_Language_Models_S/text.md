# arXiv:2506.00123v1[cs.CV]30May2025

[Figure 1]

## Visual Embodied Brain: Let Multimodal Large Language Models See, Think, and Control in Spaces

###### Gen Luo1∗, Ganlin Yang3,1∗, Ziyang Gong4,1∗, Guanzhou Chen4,1∗, Haonan Duan6, Erfei Cui4,1

Ronglei Tong6, Zhi Hou1, Tianyi Zhang7,1, Zhe Chen8,1, Shenglong Ye1, Lewei Lu6 Jingbo Wang1, Wenhai Wang1, Jifeng Dai2,1, Yu Qiao1, Rongrong Ji5, Xizhou Zhu2† 1Shanghai AI Laboratory 2Tsinghua University 3University of Science and Technology of China 4Shanghai Jiao Tong University 5Xiamen University 6SenseTime Research 7Zhejiang University 8Nanjing University

Project Page: VeBrain

Abstract

The remarkable progress of Multimodal Large Language Models (MLLMs) has attracted increasing attention to extend them to physical entities like legged robot. This typically requires MLLMs to not only grasp multimodal understanding abilities, but also integrate visual-spatial reasoning and physical interaction capabilities. Nevertheless, existing methods struggle to unify these capabilities due to their fundamental differences. In this paper, we present the Visual Embodied Brain (VeBrain), a unified framework for perception, reasoning, and control in real world. VeBrain reformulates robotic control into common text-based MLLM tasks in the 2D visual space, thus unifying the objectives and mapping spaces of different tasks. Then, a novel robotic adapter is proposed to convert textual control signals from MLLMs to motion policies of real robots. From the data perspective, we further introduce VeBrain-600k, a high-quality instruction dataset encompassing various capabilities of VeBrain. In VeBrain-600k, we take hundreds of hours to collect, curate and annotate the data, and adopt multimodal chain-of-thought (CoT) to mix the different capabilities into a single conversation. Extensive experiments on 13 multimodal benchmarks and 5 spatial intelligence benchmarks demonstrate the superior performance of VeBrain to existing MLLMs like Qwen2.5-VL. When deployed to legged robots and robotic arms, VeBrain shows strong adaptability, flexibility, and compositional capabilities compared to existing methods. For example, compared to Qwen2.5-VL, VeBrain not only achieves substantial gains on MMVet by +5.6%, but also excels in legged robot tasks with +50% average gains.

#### 1 Introduction

In recent years, Multimodal Large Language Models (MLLMs) have achieved significant progress in computer vision [99, 91, 8, 21, 69], continually pushing the boundaries of various vision-language tasks [87, 74, 48]. The next milestone of MLLMs is generally considered to be the migration of multimodal intelligence to physical entities, i.e., robotic arm [2, 92, 46], which could naturally possess the ability to perceive the surrounding world, reason in visual space, and actively engage with the environment. This requires MLLMs to go beyond traditional multimodal understanding and incorporate both visual-spatial intelligence [98, 6, 70] and physical interaction capabilities [49, 72].

Nevertheless, existing MLLMs struggle to seamlessly incorporate these basic abilities into a single unified model. A notable approach is the vision-language-action (VLA) model [86, 45, 88, 10, 49, 28, 110, 9], where MLLMs are trained on large-scale robotic datasets to map multimodal observations into control policies. While effective for control tasks, the large-scale robotic learning of VLAs inevitably compromises their multimodal

∗Equal contribution. †Corresponding author.

###### MMVet

Understanding

###### Reasoning Control

###### InfoVQA

Loco-4

[Figure 2]

[Figure 3]

[Figure 4]

72.7

[Figure 5]

[Figure 6]

[Figure 7]

Q:Describ e the image in short?

[Figure 8]

80.5

Loco-3

[Figure 9]

[Figure 10]

[Figure 11]

90.0

OCRBench

866

90.0

MME

Loco-2

[Figure 12]

[Figure 13]

100.0

2322

A: Three people riding on an elephant.

Q: What order do the door and cabinet appear in?

A: Cabinet, door.

85.3 TextVQA

Loco-1

[Figure 14]

[Figure 15]

100.0

[Figure 16]

[Figure 17]

Q:How many people in the image?

“Put the banana into the box.”

39.9

94.4 DocVQA

VSIBench

Robotic Arm

62.7

83.1 AI2D

[Figure 18]

Multi3DRef

[Figure 19]

[Figure 20]

A: Four.

##### VeBrain

87.7

60.2

......

ChartQA

ScanRefer

[Figure 21]

[Figure 22]

64.9

63.1

[Figure 23]

[Figure 24]

101.5

83.7

MMStar

SQA3D

[Figure 25]

[Figure 26]

ScanQA

MMBench

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

: GPT4Scene : RoboBrain

: GPT-4o : ChatVLA

“Detect all people in the image.”

“Deliver the ball into the box.”

Image Video Text

: Qwen2.5VL-7B : VeBrain (Ours)

Legged Robot

l 200k data for multimodal

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

understanding

Human Expert

Multimodal CoT Pipeline

### Dirverse Data Sources

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

l 312k data for visual-spatial

reasoning

GPT-4o， Gemini

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Data Engine

l 88k data for robot control

VeBrain-600k dataset

- Figure 1: Overview of VeBrain and VeBrain-600k. Compared to existing MLLMs, VeBrain achieves the best trade-off performance on benchmarks of multimodal understanding, visual-spatial reasoning, and robot control into one MLLM. To support the unified training of VeBrain, VeBrain-600k is built with a semi-automated data engine covering a variety of data sources and tasks.

understanding capability [110]. To compensate for this shortcoming, some attempts [13, 76] directly construct MLLM-based agents to control robots via text descriptions while preserving their multimodal reasoning abilities. However, due to the large task gap, their control accuracy and generalizability are still far from the requirements of real robots.

In this paper, we argue that the challenges of unifying these capabilities into MLLMs mainly arise from their inherent differences. Specifically, learning robot control demands a precise mapping from multimodal inputs to physical motion policies in the real world, e.g., vision-language-action (VLA) models [49, 9], which fundamentally differ from the cross-modal alignment objective of existing MLLMs in 2D visual space. This distinct objective makes it difficult for MLLMs to effectively balance these capabilities, leading to knowledge forgetting and task conflicts [110]. Moreover, the community still lacks a suitable data recipe to seamlessly integrate and balance these capabilities in MLLMs, further exacerbating the problem.

To overcome these limitations, we propose the Visual Embodied Brain (VeBrain), a unified framework for perception, reasoning, and control in real world. The core idea of VeBrain is to formulate robot control as common text-based MLLM tasks in 2D visual space, thereby unifying the learning objectives across different capabilities for MLLMs. Specifically, robot control is decomposed into the tasks of keypoint detection [31, 51] and embodied skill recognition [77]. The former serves as a visual-spatial anchor encoding the movement signals of the robot, and the latter represents action commands for execution signals. Based on these control signals, we design a novel robotic adapter that converts them into motion policies in a dynamic and robust manner. With these designs, VeBrain can achieve efficient robot control while retaining the strong capabilities of MLLMs.

Based on VeBrain, we construct a high-quality instruction dataset to support the unified training, namely

VeBrain-600k. In addition to integrating open-source datasets, we construct an innovative semi-automated data engine that can generate task data requiring compositional capabilities. In particular, we first collect raw images and videos from public datasets and our collections, and then annotate the key information by human experts, e.g., keypoints. Finally, capabilities like perception and reasoning are embedded into these data via multimodal chain-of-thought. Through these pipelines, VeBrain-600k not only encourages MLLMs to jointly learn the basic capabilities of an embodied brain, but also maximizes its abilities in handling complex tasks.

To validate VeBrain, we conduct extensive experiments on over 20 benchmarks covering multimodal understanding, visual-spatial reasoning, and robot control. As shown in Fig. 1, experiments not only show the better performance of VeBrain than existing methods on multimodal tasks and spatial reasoning tasks, but also confirm its strong capabilities across diverse robotic tasks. For example, VeBrain outperforms the state-of-the-art MLLM (Qwen2.5-VL [8]) by +5.6% on MMVet [102] and achieves +0.2% averaged gains of 13 multimodal benchmarks. On legged robots and robotic arms, VeBrain also demonstrates superior adaptability than all baselines, e.g., +42.9% average success rate against π0. In conclusion, our contributions are four folds:

- • We present VeBrain, a novel framework to unify multimodal understanding, visual-spatial reasoning, and robotic control. VeBrain formulates robot control as two MLLM tasks in 2D visual space, i.e., keypoint detection and embodied skill recognition, thereby avoiding potential conflicts with multimodal understanding.
- • In VeBrain, we propose a novel robotic adapter that converts text-based control signals from MLLMs into the motion policies for real robots. The MLLM and the robotic adapter form a closed loop and jointly accomplish robotic tasks in a dynamic and robust manner.
- • We present a high-quality instruction dataset covering the basic capabilities of VeBrain, namely VeBrain600k, which is constructed by human experts and semi-automated data engines. Besides, VeBrain-600k includes a large amount of carefully constructed CoT data that can significantly benefit the compositional capabilities of VeBrain.
- • VeBrain is the first to outperform existing MLLMs in terms of the average performance on 18 multimodal and spatial benchmarks. Moreover, its generalization ability in robotic control is also confirmed on 14 tasks of legged robots and robotic arms.

#### 2 Related Work

###### 2.1 Multimodal Large Language Models

The rapid advancement of large language models (LLMs) [1, 7, 90, 93, 94, 82] has significantly propelled the development of multimodal large language models (MLLMs) [99, 91, 8, 21, 69]. Models based on contrastive learning [84, 19, 47, 33] demonstrate strong open-world semantic alignment by matching imagetext pairs, yet are inherently limited by their lack of generative capabilities. By combining LLMs [12, 83] with vision foundation models [38, 84, 29], MLLMs exhibit exceptional performance in various vision-language tasks [62, 36, 73, 37, 87]. Concurrently, some works [3, 5, 57, 61] focus on cross-modal alignment through image-text interleaved training to boost contextual reasoning, while parallel efforts [16, 50, 68, 80, 95] pioneer spatial-aware architectures for granular region understanding. Despite many significant advances, these models mainly focus on perception and face challenges in real-world control scenarios, where embodied agents are expected to perceive and interact with physical environments [30, 77, 23, 63, 111]. While building upon Qwen2.5-VL as our pretrained backbone, we uniquely employ a keypoint-based control mechanism, thus closing this critical operational gap prevalent in prior implementations.

###### 2.2 Vision-Language-Action Models

Previous methods for robotic policy learning predominantly relied on reinforcement learning frameworks limited to narrow skill domains [52, 35, 112, 113]. The impressive success of MLLMs, e.g., Flamingo [3], BLIP-2 [56], and LLaVA [64], has inspired the emergence of vision-language-action models (VLAs). By

###### Robotic Ego View

MLLM Architecture Robotic Adapter

[Figure 49]

Initial Current

Go to Avocado, generate the next position keypoint as <x1, y1>, and select the action from ['Turn Left', 'Turn Right', 'Stop', 'Shake', 'Sit', 'Touch', ...

[Figure 50]

[Figure 51]

Excute

Point Tracker

[Figure 52]

[Figure 53]

[Figure 54]

###### State Convert

2D to 3D Convert

ViT

Current Point

,  ,  “Turn Right”

Tokenizer Projector

[Figure 55]

[Figure 56]

Skill Executor

[Figure 57]

###### Multimodal Large Language Model

Movement Skill

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Torch Dump

Squat Stop

Walk Jump Sit Shake

Goal Positioning: The avocado appears to the right. Goal Accessibility: There are obstacles, ... Global Planning: Move forward to the white container near the box. Turn right to face the avocado, ... Current Optimal Move: Move forward to the front of the white container near the box. Next-Step Optimal Action: <378, 457>, Turn Right.

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

Policy Pool

Takeover

End State

Success Failure

- Figure 2: Illustration of VeBrain architecture and robotic adapter. In VeBrain, the MLLM is capable of perception, thinking, and decision-making in common MLLM tasks. For robot control, an additional adapter is combined with the MLLM to achieve closed-loop control of the real robot.

fine-tuning on robotic interaction data, VLAs [86, 45, 88, 10, 32, 40] process visual observations and textual instructions through dedicated encoders, transforming these inputs into latent representations, which are subsequently decoded into executable action sequences. Some approaches [78, 85, 71, 89, 59] focus on the enhancement of individual components, such as pretrained visual representations, dynamics learning, and reasoning. Further studies like OpenVLA [49], RT-2 [11], QUART [28], and NaVILA [22] demonstrate how VLAs enable precise action generation across diverse platforms, including robotic manipulators and quadruped robots. Recently, long-horizon tasks have gained increasing attention, giving rise to zero-shot agents [44, 60, 30] that serve as high-level planners responsible for task decomposition. Nevertheless, existing VLAs struggle to match the understanding capabilities of MLLMs, which are significant for real-world applications.

- 3 Method

###### 3.1 Task Formulations

Previous endeavor [110] directly combines the formulation of vision-language-action (VLA) models and MLLMs, yet learning multimodal understanding and robotic control often negatively influence each other. To bridge this gap, we formulate these tasks based on three principles. Firstly, for all tasks, we adopt the task modeling with the same input-output space, i.e., p(t|x,y;θ). Here, x ∈ RT×H×W×3, y ∈ RL, t ∈ RN and θ denote the visual input (images or video), the textual prompt, the answer and the model parameters, respectively. Secondly, robot control is defined as a common text-based MLLM task in 2D visual space, e.g., point detection [8], thus reducing the learning difficulty. Thirdly, task-specific chain-of-thoughts are designed to help the model solve challenging problems step by step. Based on these principles, we describe the task definitions below.

Multimodal Understanding & Visual-Spatial Reasoning. These tasks are already well-defined in existing MLLMs, so we adopt a similar task formulation accordingly. Besides, we introduce the task-specific chain-of-thought (CoT) for these samples, where the MLLM is required to carefully analyze the problem and then give a response. Therefore, the template of these tasks can be written as: “ prompt: <task description>, answer: <thinking process> <answer> ”, where <·> denotes a slot of specific textual content. Note that the thinking process is removed for easy samples.

Robot Control. The key challenge of robotic control lies in planning a given task step by step and then executing. As shown in Fig. 2, we design the environment perception, task planning, and current decision as a CoT process. The current decision is decomposed into two sub-tasks for the MLLM, namely keypoint detection and embodied skill recognition. Specifically, keypoints represent a set of end locations in an image that the robot should move to or interact with, while embodied skill denotes the action to be executed after the movement, e.g., clamp. Note that the keypoints and embodied skills will be converted into executable programs via our robotic adapter.

###### 3.2 VeBrain Framework

As shown in Fig. 2, VeBrain consists of an MLLM for understanding, thinking and controlling, and a robotic adapter to convert the MLLM decisions into executable policies. These two parts are connected via a closed loop to enable dynamic and robust control.

###### 3.2.1 MLLM Architecture

Our MLLM follows Qwen2.5-VL [8], which consists of the vision encoder, the projector, and the LLM. In particular, we use the optimized vision transformer (ViT) with a stride of 14 to extract visual features, where RMSNorm [105], SwiGLU [27], 2D-RoPE [8], and window attention [58] are applied for improved efficiency and capability. In particular, given the input images x ∈ Rt×h×w×3 and textual prompt y ∈ Rl, the MLLM architecture of VeBrain can be defined by

p = Fllm(tN|Fv(x;θv),Ft(y),t0:N−1;θ), (1)

where p ∈ Rm is the next-token probability and m denotes the vocabulary size. Here, Fv(·) denotes the ViT and the MLP, and θv is their parameters. Ft(·) is the textual tokenizer. Fllm(·) and θ are the LLM and its parameters, respectively. ti denotes the i-th predicted word. For tasks defined in Sec. 3.1, MLLM can directly predict the textual answer via Eq. 1.

###### 3.2.2 Robotic Adapter

As defined in Sec. 3.1, there is still a large gap between predictions of the MLLM and real-world deployment. Firstly, these 2D keypoints struggle to be directly applied to the real-world 3D scene. Secondly, the ego-view of legged robots changes in real time as they move, leading to the misalignment between keypoints and visual perspective. Thirdly, since the MLLM cannot perceive the robotic state, it is difficult to take control in time when unexpected situations occur. To overcome these limitations, we propose a robotic adapter that consists of four modules, namely the point tracker, the movement controller, the skill executor, and the dynamic takeover.

Point tracker. In legged robots, the visual perspective from the ego-view camera differs when the robot moves. Thereby, 2D keypoints should also be updated accordingly to match the new perspective. To approach this target, we introduce a point tracking model, namely LocoTrack [24], where the keypoints of the current perspective are predicted in a real-time manner.

Movement controller. The movement controller aims to generate the movement policy for a robot or robotic arm based on the 2D keypoints. Given the keypoints predicted by the MLLM, we can obtain the corresponding depth information from the RGBD camera. Then, these 2D keypoints are converted to

###### 3D ones via a simple transformation of the calibration matrix. Based on these 3D points, we estimate the movement velocity of the robot and drive the underlying movement policy model.

Skill executor. Existing robots have accessed a variety of skills through pre-trained policies, e.g., walking [72] and jumping [55], which are sufficient to allow the robot to interact with humans or spaces. Therefore, we collect these action policies and categorize them by name. Given the skill predicted by the MLLM, the skill executor will call the corresponding policy to accomplish the task.

Dynamic takeover. In the real world, the environmental uncertainty and the mistakes of the policy model often lead to the loss of targets. In this case, dynamic takeover aims to exchange control to the MLLM when the robotic adapter fails. In particular, the takeover happens when the key points disappear from view for several frames or when a subtask is done.

###### Table 1: Ablation study of VeBrain on frameworks and data. Our baseline follows VLM-PC [13] and replaces the MLLM with Qwen2.5-VL.

Multimodal Understanding Visual-spatial Reasoning Robot Control

Avg MMVet MMBench ScanQA (CIDEr) VSI-Bench Complex Find Transporting

Qwen2.5-VL [8] 67.1 83.5 62.7 35.9 0.0 10.0 43.2 + Robotic Adapter 67.1 83.5 62.7 35.9 0.0 40.0 48.2 + Control Data 67.2 82.7 56.8 32.8 30.0 50.0 53.3 + Spatial Reasoning Data 64.7 82.1 102.2 40.3 65.0 70.0 70.7 + Multimodal Understanding Data 72.7 83.7 101.5 39.9 80.0 90.0 78.0

###### Table 2: Comparison of VeBrain with two common frameworks. For VLA, the training data remains the same as VeBrain, except that the robot annotations are replaced with motion policies [28].

Control Signal

Robotic Adapter

Multimodal Understanding Visual-spatial Reasoning Robot Control

Forms

Avg MMVet MMBench ScanQA (CIDEr) VSI-Bench Complex Find Transporting

MLLM [13] Text 67.1 83.5 62.7 35.9 10.0 20.0 46.5 VLA [28] Action policy 50.8 73.4 55.1 29.8 50.0 30.0 48.2 VeBrain (Ours) Points & Action 72.7 83.7 101.5 39.9 80.0 90.0 78.0

###### 3.3 VeBrain-600k Data Engine

VeBrain-600k contains extensive datasets covering basic capabilities of VeBrain. For each capability, our data collection strives to be as diverse as possible.

Data collection and annotation. For multimodal understanding, we construct a dataset with 200k conversations, including data formats of 2D images, videos, and textual descriptions. Most of the data is collected from open-source datasets, such as ShareGPT4V [17] and MMInstruct [67], etc. Another part of the data is generated by GPT4o and its CoT process is annotated by our pipeline.

For visual-spatial reasoning, we collect 312k items from open-source datasets, i.e., GPT4Scene [81], and our annotated dataset. In our annotated dataset, given images from ScanNet [26], we annotate images using two pipelines: 1) combining image frames and point-cloud snapshots to generate descriptive conversations via GPT-4o; 2) labeling information of counting, object size, and object distance via annotations from ScanNet and human experts.

For robot control, we collect a multimodal robot dataset with 88k items from scratch, including locomotion and manipulation of legged robots and robotic arms. For data collection, 4 human experts take over 80 hours to collect video episodes and motion states of legged robots and robotic arms. Then, 5 human experts manually annotate the keypoints and actions of these episodes.

Chain-of-thoughts generation. Our CoT generation aims to embed different capabilities into one conversation via CoT. For multimodal understanding and visual-spatial reasoning, CoT content aims to integrate reasoning capabilities into these tasks. Thus, we design different CoT templates according to the task properties and generate CoT content using Gemini-2.0 and GPT-4o. For robot control, the CoT process further integrates contents of perception, reasoning, and control, which firstly describes the visual observation, then decomposes the task, and finally makes the control decision. More details about CoT generation are given in our appendix.

Quality verification. During the robot data collection process, three experts carefully reviewed each video to ensure that the objects were within the robot’s field of view. For CoT generation, we adopt a cross-model validation pipeline. In particular, we employ Gemini-2.0 as the reference model to assess the logical and physical plausibility of CoT data generated by GPT-4o. Finally, we randomly select 10% of the data for manual inspection by 5 human experts, and only 5.3% of them are further excluded, demonstrating the reliability of our data generation pipeline.

- Table 3: Comparison with existing MLLMs and VLAs on general MLLM benchmarks. For MME, we sum the perception and cognition scores. Avg denotes the normalized average performance of MLLM benchmarks and VQA benchmarks. The highest score among open-source MLLMs and VLAs are colored in bold.

Model MMBench MMVet MMMU MME MMStar RWQA TextVQA DocVQA AI2D ChartQA InfoVQA SEED2+ OCRBench Avg

▼ Closed-source MLLMs:

GPT-4o [79] 83.4 69.1 69.1 2328 64.7 75.4 77.4 91.1 84.6 86.7 80.7 72.0 736 76.5 Claude-3.5 Sonnet [4] 82.6 70.1 68.3 1920 65.1 60.1 76.5 95.2 81.2 90.8 74.3 71.7 788 74.6 Gemini-1.5-Pro [91] 73.9 64.0 62.2 − 59.1 67.5 78.8 93.1 88.4 87.2 81.0 70.8 754 −

▼ Open-source MLLMs: MiniCPM-V2.6-8B [100] 81.5 60.0 49.8 2348 57.5 65.0 80.1 90.8 82.1 82.4 − 65.7 852 − LLaVA-One-Vision-7B [54] 80.9 57.5 48.8 1998 61.7 66.3 − 87.5 81.4 80.0 68.8 − 622 − Eagle2.5-8B [15] − 62.9 55.8 − 66.2 76.7 83.7 94.1 84.5 87.5 80.4 − 869 − InternVL2.5-8B [20] 84.6 62.8 56.2 2344 62.8 70.1 79.1 93.0 84.5 84.8 77.6 69.7 822 75.0 Qwen2.5-VL-7B [8] 83.5 67.1 58.6 2347 63.9 68.5 84.9 95.7 83.9 87.3 82.6 70.4 864 76.9

▼ Vision-language-action Models: OpenVLA [49] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ECoT [104] 0 0 5.4 0 0 0 0 0 0 0 0 0 12 1.3 DiVLA [96] − − 17.2 187 21.1 25.2 7.5 15.2 43.1 17.2 14.7 − 294 − ChatVLA [110] 69.0 − 37.4 1435 47.2 57.0 71.2 83.3 67.6 59.9 53.3 − 729 − RoboBrain [46] 80.4 − 49.0 2084 61.2 68.8 75.9 88.0 82.0 80.5 − − 677 −

VeBrain (Ours) 83.7 72.7 56.6 2322 63.1 71.0 85.3 94.4 83.1 87.7 80.5 70.7 866 77.1

Table 4: Comparison of VeBrain and existing MLLMs on four 3D spatial benchmarks.

Models

ScanQA (val) SQA3D (val) ScanRefer Multi3DRef BLEU-1 BLEU-4 METEOR ROUGE CIDEr EM-1 EM-R1 Acc@0.25 Acc@0.50 all F1@0.25 all F1@0.50

▼ 3D MLLMs: Chat-3D v2 [42] 38.4 7.3 16.1 40.1 77.1 − − − − − − 3D-LLM [39] 39.3 12.0 14.5 35.7 69.4 − − − − − − LEO [43] − 11.5 16.2 39.3 80.0 50.0 52.4 − − − − 3DVG-Transformer [109] − − − − − − − 47.6 34.7 − 25.5 M3DRef-CLIP [107] − − − − − − − 51.9 44.7 42.8 38.4 Chat-Scene [41] 43.2 14.3 18.0 41.6 87.7 54.6 57.5 55.5 50.2 57.1 52.4

▼ 2D MLLMs:

GPT-4o [79] 27.3 7.3 12.5 37.7 59.1 42.7 46.4 5.4 5.1 21.1 19.9 Qwen2.5-VL-7B [8] 25.7 9.8 12.3 33.1 62.7 45.8 49.5 5.4 5.1 21.1 19.9 GPT4Scene-HDM [81] 44.4 15.5 18.9 46.5 96.3 60.6 63.3 62.6 57.0 64.5 59.8

VeBrain (Ours) 47.7 17.3 20.1 48.2 101.5 61.6 64.9 66.4 60.2 67.8 62.7

- 4 Experiments

###### 4.1 Evaluation Datasets

For multimodal capability, we evaluate VeBrain and existing MLLMs on 13 comprehensive multimodal benchmarks. Specifically, MLLM benchmarks encompass MMBench-EN test [65], MMVet [101], MMMU val [103], MME [34], MMStar [18] and RealWorldQA [97]. Visual question answering benchmarks include TextVQA val [87], DocVQA test [25], AI2D test [48], ChartQA test [74], InfoVQA test [75], SEED-Bench-2Plus [53] and OCRBench [66]. For visual-spatial reasoning, we evaluate VeBrain on five benchmarks, including ScanQA val [6], SQA3D val [70], ScanRefer [14], Multi3DRef [107], and VSI-Bench [98]. For robot control, evaluation benchmarks are built from our self-collected scenes, as described in the appendix.

###### 4.2 Implementation Details

VeBrain is implemented based on Qwen2.5-VL-7B [8], which is fine-tuned on our VeBrain-600k dataset with a learning rate of 5e-6 for one epoch, taking approximately 2 days on 32 NVIDIA A100 GPUs. After training, VeBrain is deployed to a Tesla A100 GPU in the cloud, running at 0.5Hz. The tracking model is deployed on the NVIDIA Jetson Orin with a 15Hz running frequency. Our legged robot (Unitree Go2) is equipped with an ego-view RGBD camera (RealSense D435i) and a Jetson AGX Orin platform. The robotic arm consists of

###### Table 5: Comparison of VeBrain and existing MLLMs on VSI benchmark.

Models AVG Obj. Count Abs. Dist Obj. Size Room. Size Rel. Dist Rel. Dir Router Plan Appr. Order

GPT-4o [79] 34.0 46.2 5.3 43.8 38.2 37.0 41.3 31.5 28.5 Gemini-1.5 Pro [91] 45.4 56.2 30.9 64.1 43.6 51.3 46.3 36.0 34.6

VILA-1.5-8B [61] 28.9 17.4 21.8 50.3 18.8 32.1 34.8 31.0 24.8 LongVA-7B [106] 29.2 38.0 16.6 38.9 22.2 33.1 43.3 25.4 15.7 LLaVA-NeXT-Video-7B [108] 35.6 48.5 14.0 47.8 24.2 43.5 42.4 34.0 30.6 LLaVA-OneVision-7B [54] 32.4 47.7 20.2 47.4 12.3 42.5 35.2 29.4 24.4 Qwen2.5-VL-7B [8] 35.9 40.6 20.3 50.5 35.7 37.0 40.5 29.9 32.4

VeBrain (Ours) 39.9 59.5 31.2 53.7 40.3 41.0 36.9 29.9 26.2

- Table 6: Performance comparison on 7 tasks of legged robot. We report the success rate over ten trials for each task. Task details are given in the appendix.

Easy Middle Hard

Robotic Adapter

Models

Overall Find Track Interaction

Complex Find

Complex Interaction

Complex Transport

Transport

VLA [28] 70.0 30.0 20.0 50.0 15.0 30.0 10.0 32.1 VLM-PC [13] 60.0 40.0 10.0 10.0 5.0 20.0 0.0 20.7 GPT-4o [79] 40.0 10.0 35.0 10.0 5.0 40.0 10.0 21.4 Gemini-1.5 Pro [91] 20.0 0.0 20.0 0.0 0.0 30.0 0.0 10.0 Qwen2.5-VL-7B [8] 100.0 100.0 15.0 20.0 10.0 40.0 10.0 42.1

VeBrain (Ours) 100.0 100.0 90.0 80.0 85.0 90.0 60.0 86.4

a 7-DoF tabletop Franka Emika Panda robot arm and a Robotiq 2F-85 gripper, which is equipped with an RGBD camera in a third-person view. VeBrain and these robots communicate via a wireless network. More details are given in the appendix.

###### 4.3 Quantitative Results

Ablation studies. In Tab. 1, we validate the effectiveness of the data and architecture of VeBrain. From this table, the first observation is that despite the promising performance on multimodal understanding, existing MLLMs often fall short in visual-spatial reasoning and robot control, i.e., 0% success rate on the “Complex Find” task. After equipping the model with our robotic adapter, the success rate of Qwen2.5-VL on two robot control tasks is obviously improved. Another observation is that after fine-tuning on control data, the multimodal capabilities of VeBrain are well preserved, greatly confirming the design of VeBrain. In addition, each type of data makes a significant contribution to the corresponding capabilities, e.g., +7.5% of spatial reasoning data on VSI-Bench [98].

In Tab. 2, we compare VeBrain with two common frameworks, i.e., MLLM [13] and VLA [28]. From these results, we find that MLLM struggles to directly control the robot on two tasks due to its weak control capabilities. In contrast, VLA can perform well on robot control tasks, but it greatly sacrifices multimodal abilities, e.g., -16.3% on MMVet compared to MLLM. Compared to these frameworks, VeBrain achieves the best trade-off performance on all tasks, yielding up to +31.5% average gains against other frameworks. These results not only validate the shortcomings of existing frameworks in unifying multimodal understanding, visual-spatial reasoning, and robot control, but also confirm the effectiveness of each design in VeBrain.

Results of multimodal understanding. In Tab. 3, we observe that existing VLA models perform inferiorly on these multimodal benchmarks, while OpenVLA completely loses the multimodal capabilities. Among them, RoboBrain [46] integrates a large amount of multimodal understanding data in its training and achieves better results than other VLA models. However, compared to advanced MLLMs like Qwen2.5VL, RoboBrain obviously falls short in OCR and chart benchmarks, e.g., -6.8% on ChartQA and -187 on OCRBench. In contrast, VeBrain demonstrates the comprehensive advantages across 13 MLLM benchmarks, e.g., +6.4% against RoboBrain on DocVQA. Besides, VeBrain performs better than advanced open-source and closed-source MLLMs on most benchmarks, as proved by +5.6% on MMVet and the best normalized average performance of 77.1, suggesting its superior multimodal capabilities. Considering the much smaller

- Table 7: Performance comparison on 7 tasks of robotic arm. All models are allowed to see 10 demonstrations per task during training. Task details are given in the appendix.

Move into Box Move out of Box Open Drawer

Long-Horizon

Model

Overall Banana Pepper Carrot Kiwifruit Carrot Pepper

OpenVLA [49] 0.0 10.0 10.0 20.0 40.0 0.0 0.0 11.4 π0 [9] 0.0 30.0 90.0 50.0 50.0 0.0 0.0 31.4 VeBrain (Ours) 70.0 70.0 90.0 60.0 90.0 60.0 80.0 74.3

[Figure 69]

[Figure 70]

[Figure 71]

: Now you need to open the drawer and pick up the pepper in the drawer...

: Your task is to find a chili pepper on the table and grab it...

: Now you need to open the drawer and pick up the pepper in the drawer...

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

: The drawer is fully opened, revealing its contents clearly. Within the drawer, ... NextStep Optimal Action: [<349, 287>, <360, 280>], Grasp.

:The target object is located inside the drawer on the right side of the workspace relative to the robot's position... Next-Step Optimal Action: [<597, 393>, <604, 393>], Pull.

:... None of these items appear to be the chili pepper... I notice that the drawer is partially open and the chili pepper could potentially be hidden...I will move closer to the drawer to start this process.

[Figure 79]

[Figure 80]

[Figure 81]

: Now you need to transport goods to the brown box...

: Now you need to transport goods to the brown box...

: Determine whether the task has been completed successfully...

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

:I see a cardboard box...There are now three items inside... This confirms that my task was successful as the box contains the required number of items.

:The brown box is positioned slightly to my right, with no obstacles blocking the pat... Next-Step Optimal Action: <764, 582>, Dump.

:Goal Positioning: The brown box appears slightly right of center in my view... Next-Step Optimal Action: <302, 481>, Turn Right.

Figure 3: Visualization of VeBrain on robotic arm and legged robot. VeBrain demonstrates compositional capabilities in handling complex robotic tasks. Due to space limitations, most of the text regarding the thinking process is omitted in the figure.

size of VeBrain than closed-source MLLMs, these results further confirm the strong multimodal abilities of VeBrain.

Results of visual-spatial reasoning. Tab. 4 first demonstrates the results of four 3D spatial benchmarks. In this table, the models require strong 3D spatial perception and reasoning capabilities to answer different types of questions. Therefore, most previous approaches [41, 107] adopt the 3D-based MLLM structure and achieve promising results on the four benchmarks. In comparison, directly transferring 2D MLLMs to these tasks leads to poor performance, e.g., -50.1 Acc@0.25 of Qwen2.5-VL-7B [8], suggesting their shortcoming in

- 3D spatial understanding and reasoning. Compared to these methods, GPT4Scene-HDM [81] achieves better results via the video-based 2D MLLM and object markers. However, as a specialist model, GPT4Scene-HDM struggles to be applied to common 2D multimodal tasks. In contrast, VeBrain, as a generalist MLLM, can even outperform GPT4Scene-HDM on all tasks, e.g., +5.2 CIDEr on ScanQA val, greatly validating its generalizability. Tab. 5 further diagnoses the visual-spatial reasoning abilities of existing MLLMs and VeBrain. From it, we can see that VeBrain outperforms all existing MLLMs on the VSI benchmark in terms of the average score, e.g., +4.0% against the Qwen2.5-VL-7B. Compared to much larger MLLMs like GPT-4o, VeBrain can also perform even better. Considering the great challenge of the VSI benchmark, these results further confirm the spatial reasoning capabilities of VeBrain.

Results of robot control. In Tab 6, we compare VLA, MLLMs, and VeBrain on seven tasks of the legged robot. From it, we find that existing VLA [28] and MLLMs [79, 91, 8] have difficulty in directly accomplishing most tasks like “Interaction” and “Transport”. Among them, Qwen2.5-VL with our robotic adapter achieves the best results. However, when adapting it to harder tasks like “Complex Find”, their success rate drops significantly, e.g., 20% success rate. These tasks typically require compositional capabilities like spatial reasoning and embodied control, which common MLLMs are not good at. Compared to these methods, VeBrain unifies these capabilities and achieves significantly better results on various complex tasks of legged robots. For example, VeBrain outperforms all models by 50% on the long-horizon task “Complex Transport”. Similar merits of VeBrain can also be witnessed in the robotic arm. As shown in Tab. 7, common VLAs [49, 9] demonstrate limited success rate in most manipulation tasks, e.g., 30% success rate of π0 [9] on “Move Pepper into Box”. In long-horizon tasks, the success rate of π0 further drops to 0%. Compared with these methods, VeBrain achieves the highest success rate in all tasks. In the most challenging task, VeBrain outperforms π0 by up to 80%, which further validates its effectiveness in robot control.

###### 4.4 Visualizations

In Fig. 3, we visualize the results of VeBrain on real robots, i.e., robotic arm and legged robot. From this figure, we see that VeBrain can handle complex robotic tasks through compositional capabilities. For example, when asked to find a hidden chili pepper, VeBrain can correctly guess its potential location and then step through the steps to catch it. This requires not only control abilities, but also excellent perception and reasoning capabilities. Similar merits are also reflected in the second example, where VeBrain further determines whether the goods have been delivered to the destination.

#### 5 Conclusion

In this paper, we propose VeBrain, a visual embodied brain that unifies multimodal understanding, visualspatial reasoning, and robot control. In VeBrain, robot control is formulated as multimodal tasks of key point detection and embodied skill recognition, and a novel robotic adapter is proposed to convert these signals of VeBrain to motion policies. Based on this framework, we propose the VeBrain-600k dataset to enable the learning of different capabilities. Extensive experiments on over 20 benchmarks demonstrate the superior performance of VeBrain than existing methods in tasks of three capabilities. In particular, VeBrain is the first to outperform the state-of-the-art MLLM in most multimodal tasks while retaining strong capabilities of spatial reasoning and robot control.

Acknowledgments. This work was supported by the National Key R&D Program of China (NO. 2022ZD0161302), the National Natural Science Foundation of China (No. 623B2088) and the China Postdoctoral Science Foundation (No. 2024M761548).

#### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 3
- [2] Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, et al. Do as i can, not as i say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691, 2022. 1
- [3] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L. Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karén Simonyan. Flamingo: a visual language model for few-shot learning. In NeurIPS, 2022. 3
- [4] Anthropic. Claude 3.5 sonnet. https://www.anthropic.com/news/claude-3-5-sonnet, 2024. 7

- [5] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, et al. Openflamingo: An open-source framework for training large autoregressive vision-language models. arXiv preprint arXiv:2308.01390, 2023. 3
- [6] Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, and Motoaki Kawanabe. Scanqa: 3d question answering for spatial scene understanding. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19129–19139, 2022. 1, 7
- [7] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 3
- [8] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 1, 3, 4, 5, 6, 7, 8, 9, 10, 18, 20
- [9] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024. 1, 2, 9, 10
- [10] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022. 1, 4
- [11] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023. 4
- [12] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. NeurIPS, 33:1877–1901, 2020. 3
- [13] Annie S Chen, Alec M Lessing, Andy Tang, Govind Chada, Laura Smith, Sergey Levine, and Chelsea Finn. Commonsense reasoning for legged robot adaptation with vision-language models. arXiv preprint arXiv:2407.02666,

2024. 2, 6, 8

- [14] Dave Zhenyu Chen, Angel X Chang, and Matthias Nießner. Scanrefer: 3d object localization in rgb-d scans using natural language. In European conference on computer vision, pages 202–221. Springer, 2020. 7
- [15] Guo Chen, Zhiqi Li, Shihao Wang, Jindong Jiang, Yicheng Liu, Lidong Lu, De-An Huang, Wonmin Byeon, Matthieu Le, Tuomas Rintamaki, et al. Eagle 2.5: Boosting long-context post-training for frontier vision-language models. arXiv preprint arXiv:2504.15271, 2025. 7
- [16] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023. 3
- [17] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. In European Conference on Computer Vision, pages 370–387. Springer, 2024. 6, 19
- [18] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330, 2024. 7
- [19] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv: 2312.14238, 2023. 3

- [20] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024. 7
- [21] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv:2404.16821, 2024. 1, 3
- [22] An-Chieh Cheng, Yandong Ji, Zhaojing Yang, Zaitian Gongye, Xueyan Zou, Jan Kautz, Erdem Bıyık, Hongxu Yin, Sifei Liu, and Xiaolong Wang. Navila: Legged robot vision-language-action model for navigation. arXiv preprint arXiv:2412.04453, 2024. 4
- [23] Junmo Cho, Jaesik Yoon, and Sungjin Ahn. Spatially-aware transformer for embodied agents. arXiv preprint arXiv:2402.15160, 2024. 3
- [24] Seokju Cho, Jiahui Huang, Jisu Nam, Honggyu An, Seungryong Kim, and Joon-Young Lee. Local all-pair correspondence for point tracking. In European Conference on Computer Vision, pages 306–325. Springer, 2024. 5, 19
- [25] Christopher Clark and Matt Gardner. Simple and effective multi-paragraph reading comprehension. In ACL, pages 845–855, 2018. 7
- [26] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017. 6, 19
- [27] Yann N Dauphin, Angela Fan, Michael Auli, and David Grangier. Language modeling with gated convolutional networks. In International conference on machine learning, pages 933–941. PMLR, 2017. 5
- [28] Pengxiang Ding, Han Zhao, Wenjie Zhang, Wenxuan Song, Min Zhang, Siteng Huang, Ningxi Yang, and Donglin Wang. Quar-vla: Vision-language-action model for quadruped robots. In European Conference on Computer Vision, pages 352–367. Springer, 2024. 1, 4, 6, 8, 10
- [29] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021. 3
- [30] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023. 3, 4
- [31] Kaiwen Duan, Song Bai, Lingxi Xie, Honggang Qi, Qingming Huang, and Qi Tian. Centernet: Keypoint triplets for object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6569–6578, 2019. 2
- [32] Cunxin Fan, Xiaosong Jia, Yihang Sun, Yixiao Wang, Jianglan Wei, Ziyang Gong, Xiangyu Zhao, Masayoshi Tomizuka, Xue Yang, Junchi Yan, et al. Interleave-vla: Enhancing robot manipulation with interleaved image-text instructions. arXiv preprint arXiv:2505.02152, 2025. 4
- [33] Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva: Exploring the limits of masked visual representation learning at scale. In CVPR, pages 19358–19369, 2023. 3
- [34] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and Rongrong Ji. MME: A comprehensive evaluation benchmark for multimodal large language models. arXiv: 2306.13394, 2023. 7
- [35] Yiran Geng, Boshi An, Haoran Geng, Yuanpei Chen, Yaodong Yang, and Hao Dong. Rlafford: End-to-end affordance learning for robotic manipulation. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 5880–5886. IEEE, 2023. 3
- [36] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In CVPR, pages 6904–6913, 2017. 3

- [37] Danna Gurari, Qing Li, Abigale J. Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P. Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In CVPR, pages 3608–3617,

2018. 3

- [38] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross B. Girshick. Masked autoencoders are scalable vision learners. In CVPR, pages 15979–15988, 2022. 3
- [39] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3d-llm: Injecting the 3d world into large language models. Advances in Neural Information Processing Systems, 36: 20482–20494, 2023. 7
- [40] Zhi Hou, Tianyi Zhang, Yuwen Xiong, Haonan Duan, Hengjun Pu, Ronglei Tong, Chengyang Zhao, Xizhou Zhu, Yu Qiao, Jifeng Dai, et al. Dita: Scaling diffusion transformer for generalist vision-language-action policy. arXiv preprint arXiv:2503.19757, 2025. 4
- [41] Haifeng Huang, Yilun Chen, Zehan Wang, Rongjie Huang, Runsen Xu, Tai Wang, Luping Liu, Xize Cheng, Yang Zhao, Jiangmiao Pang, et al. Chat-scene: Bridging 3d scene and large language models with object identifiers. arXiv preprint arXiv:2312.08168, 2023. 7, 9
- [42] Haifeng Huang, Zehan Wang, Rongjie Huang, Luping Liu, Xize Cheng, Yang Zhao, Tao Jin, and Zhou Zhao. Chat-3d v2: Bridging 3d scene and large language models with object identifiers. CoRR, 2023. 7
- [43] Jiangyong Huang, Silong Yong, Xiaojian Ma, Xiongkun Linghu, Puhao Li, Yan Wang, Qing Li, Song-Chun Zhu, Baoxiong Jia, and Siyuan Huang. An embodied generalist agent in 3d world. arXiv preprint arXiv:2311.12871,

2023. 7

- [44] Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, et al. Inner monologue: Embodied reasoning through planning with language models. arXiv preprint arXiv:2207.05608, 2022. 4
- [45] Eric Jang, Alex Irpan, Mohi Khansari, Daniel Kappler, Frederik Ebert, Corey Lynch, Sergey Levine, and Chelsea Finn. Bc-z: Zero-shot task generalization with robotic imitation learning. In Conference on Robot Learning, pages 991–1002. PMLR, 2022. 1, 4
- [46] Yuheng Ji, Huajie Tan, Jiayu Shi, Xiaoshuai Hao, Yuan Zhang, Hengyuan Zhang, Pengwei Wang, Mengdi Zhao, Yao Mu, Pengju An, et al. Robobrain: A unified brain model for robotic manipulation from abstract to concrete. arXiv preprint arXiv:2502.21257, 2025. 1, 7, 8
- [47] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In ICML, pages 4904–4916, 2021. 3
- [48] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In ECCV, pages 235–251, 2016. 1, 7
- [49] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024. 1, 2, 4, 7, 9, 10
- [50] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. arXiv preprint arXiv:2308.00692, 2023. 3
- [51] Hei Law and Jia Deng. Cornernet: Detecting objects as paired keypoints. In Proceedings of the European conference on computer vision (ECCV), pages 734–750, 2018. 2
- [52] Sergey Levine, Peter Pastor, Alex Krizhevsky, Julian Ibarz, and Deirdre Quillen. Learning hand-eye coordination for robotic grasping with deep learning and large-scale data collection. The International journal of robotics research, 37(4-5):421–436, 2018. 3
- [53] Bohao Li, Yuying Ge, Yi Chen, Yixiao Ge, Ruimao Zhang, and Ying Shan. Seed-bench-2-plus: Benchmarking multimodal large language models with text-rich visual comprehension. arXiv preprint arXiv:2404.16790, 2024. 7

- [54] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 7, 8
- [55] Chenhao Li, Marin Vlastelica, Sebastian Blaes, Jonas Frey, Felix Grimminger, and Georg Martius. Learning agile skills via adversarial imitation of rough partial demonstrations. In Conference on Robot Learning, pages 342–352. PMLR, 2023. 5
- [56] Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, pages 19730–19742, 2023. 3
- [57] Qingyun Li, Zhe Chen, Weiyun Wang, Wenhai Wang, Shenglong Ye, Zhenjiang Jin, et al. Omnicorpus: A unified multimodal corpus of 10 billion-level images interleaved with text. In The Thirteenth International Conference on Learning Representations, 2025. 3
- [58] Yanghao Li, Hanzi Mao, Ross Girshick, and Kaiming He. Exploring plain vision transformer backbones for object detection. arXiv preprint arXiv:2203.16527, 2022. 5
- [59] Yanbang Li, Ziyang Gong, Haoyang Li, Xiaoqi Huang, Haolan Kang, Guangping Bai, and Xianzheng Ma. Robotic visual instruction. arXiv preprint arXiv:2505.00693, 2025. 4
- [60] Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. Code as policies: Language model programs for embodied control. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 9493–9500. IEEE, 2023. 4
- [61] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26689–26699, 2024. 3, 8
- [62] Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft COCO: common objects in context. In ECCV, pages 740–755, 2014. 3
- [63] Benlin Liu, Yuhao Dong, Yiqin Wang, Yongming Rao, Yansong Tang, Wei-Chiu Ma, and Ranjay Krishna. Coarse correspondence elicit 3d spacetime understanding in multimodal language model. arXiv preprint arXiv:2408.00754, 2024. 3
- [64] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 3
- [65] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player? arXiv: 2307.06281, 2023. 7
- [66] Yuliang Liu, Zhang Li, Hongliang Li, Wenwen Yu, Mingxin Huang, Dezhi Peng, Mingyu Liu, Mingrui Chen, Chunyuan Li, Lianwen Jin, et al. On the hidden mystery of ocr in large multimodal models. arXiv preprint arXiv:2305.07895, 2023. 7
- [67] Yangzhou Liu, Yue Cao, Zhangwei Gao, Weiyun Wang, Zhe Chen, Wenhai Wang, Hao Tian, Lewei Lu, Xizhou Zhu, Tong Lu, et al. Mminstruct: A high-quality multi-modal instruction tuning dataset with extensive diversity. Science China Information Sciences, 67(12):1–16, 2024. 6, 19
- [68] Zhaoyang Liu, Yinan He, Wenhai Wang, Weiyun Wang, Yi Wang, Shoufa Chen, Qinglong Zhang, Zeqiang Lai, Yang Yang, Qingyun Li, Jiashuo Yu, et al. Interngpt: Solving vision-centric tasks by interacting with chatgpt beyond language. arXiv preprint arXiv:2305.05662, 2023. 3
- [69] Gen Luo, Xue Yang, Wenhan Dou, Zhaokai Wang, Jiawen Liu, Jifeng Dai, Yu Qiao, and Xizhou Zhu. Monointernvl: Pushing the boundaries of monolithic multimodal large language models with endogenous visual pre-training. arXiv preprint arXiv:2410.08202, 2024. 1, 3
- [70] Xiaojian Ma, Silong Yong, Zilong Zheng, Qing Li, Yitao Liang, Song-Chun Zhu, and Siyuan Huang. Sqa3d: Situated question answering in 3d scenes. arXiv preprint arXiv:2210.07474, 2022. 1, 7
- [71] Arjun Majumdar, Karmesh Yadav, Sergio Arnaud, Jason Ma, Claire Chen, Sneha Silwal, Aryan Jain, VincentPierre Berges, Tingfan Wu, Jay Vakil, et al. Where are we in the search for an artificial visual cortex for embodied intelligence? Advances in Neural Information Processing Systems, 36:655–677, 2023. 4

- [72] Gabriel B Margolis and Pulkit Agrawal. Walk these ways: Tuning robot control for generalization with multiplicity of behavior. In Conference on Robot Learning, pages 22–31. PMLR, 2023. 1, 5
- [73] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In CVPR, pages 3195–3204, 2019. 3
- [74] Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In ACL, pages 2263–2279, 2022. 1, 7
- [75] Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In WACV, pages 1697–1706, 2022. 7
- [76] Yuting Mei, Ye Wang, Sipeng Zheng, and Qin Jin. Quadrupedgpt: Towards a versatile quadruped agent in open-ended worlds. arXiv preprint arXiv:2406.16578, 2024. 2
- [77] Yao Mu, Qinglong Zhang, Mengkang Hu, Wenhai Wang, Mingyu Ding, Jun Jin, Bin Wang, Jifeng Dai, Yu Qiao, and Ping Luo. Embodiedgpt: Vision-language pre-training via embodied chain of thought. Advances in Neural Information Processing Systems, 36:25081–25094, 2023. 2, 3
- [78] Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, and Abhinav Gupta. R3m: A universal visual representation for robot manipulation. arXiv preprint arXiv:2203.12601, 2022. 4
- [79] OpenAI. Gpt-4o system card. https://cdn.openai.com/gpt-4o-system-card.pdf, 2024. 7, 8, 10
- [80] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023. 3
- [81] Zhangyang Qi, Zhixiong Zhang, Ye Fang, Jiaqi Wang, and Hengshuang Zhao. Gpt4scene: Understand 3d scenes from videos with vision-language models. arXiv preprint arXiv:2501.01428, 2025. 6, 7, 9, 19
- [82] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. 2018. 3
- [83] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019. 3
- [84] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 3
- [85] Ilija Radosavovic, Tete Xiao, Stephen James, Pieter Abbeel, Jitendra Malik, and Trevor Darrell. Real-world robot learning with masked visual pre-training. In Conference on Robot Learning, pages 416–426. PMLR, 2023. 4
- [86] Mohit Shridhar, Lucas Manuelli, and Dieter Fox. Cliport: What and where pathways for robotic manipulation. In Conference on robot learning, pages 894–906. PMLR, 2022. 1, 4
- [87] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards VQA models that can read. In CVPR, 2019. 1, 3, 7
- [88] Austin Stone, Ted Xiao, Yao Lu, Keerthana Gopalakrishnan, Kuang-Huei Lee, Quan Vuong, Paul Wohlhart, Sean Kirmani, Brianna Zitkovich, Fei Xia, et al. Open-world object manipulation using pre-trained vision-language models. arXiv preprint arXiv:2303.00905, 2023. 1, 4
- [89] Yanchao Sun, Shuang Ma, Ratnesh Madaan, Rogerio Bonatti, Furong Huang, and Ashish Kapoor. Smart: Self-supervised multi-task pretraining with control transformers. arXiv preprint arXiv:2301.09816, 2023. 4
- [90] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 3
- [91] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 1, 3, 7, 8, 10

- [92] Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, et al. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025. 1
- [93] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 3
- [94] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv: 2307.09288, 2023. 3
- [95] Weiyun Wang, Min Shi, Qingyun Li, Wenhai Wang, Zhenhang Huang, Linjie Xing, Zhe Chen, Hao Li, Xizhou Zhu, Zhiguo Cao, et al. The all-seeing project: Towards panoptic visual recognition and understanding of the open world. In ICLR, 2024. 3
- [96] Junjie Wen, Minjie Zhu, Yichen Zhu, Zhibin Tang, Jinming Li, Zhongyi Zhou, Chengmeng Li, Xiaoyu Liu, Yaxin Peng, Chaomin Shen, et al. Diffusion-vla: Scaling robot foundation models via unified diffusion and autoregression. arXiv preprint arXiv:2412.03293, 2024. 7
- [97] X.ai. Grok-1.5 vision preview. https://x.ai/blog/grok-1.5v, 2024. 7
- [98] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. arXiv preprint arXiv:2412.14171, 2024. 1, 7, 8
- [99] Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. The dawn of lmms: Preliminary explorations with gpt-4v (ision). arXiv: 2309.17421, 9, 2023. 1, 3
- [100] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024. 7
- [101] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv: 2308.02490, 2023. 7
- [102] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490,

2023. 3

- [103] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. arXiv: 2311.16502, 2023. 7
- [104] Michał Zawalski, William Chen, Karl Pertsch, Oier Mees, Chelsea Finn, and Sergey Levine. Robotic control via embodied chain-of-thought reasoning. arXiv preprint arXiv:2407.08693, 2024. 7
- [105] Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32, 2019. 5
- [106] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852,

2024. 8

- [107] Yiming Zhang, ZeMing Gong, and Angel X Chang. Multi3drefer: Grounding text description to multiple 3d objects. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15225–15236,

2023. 7, 9

- [108] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llava-next: A strong zero-shot video understanding model, 2024. 8
- [109] Lichen Zhao, Daigang Cai, Lu Sheng, and Dong Xu. 3dvg-transformer: Relation modeling for visual grounding on point clouds. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2928–2937,

2021. 7

- [110] Zhongyi Zhou, Yichen Zhu, Minjie Zhu, Junjie Wen, Ning Liu, Zhiyuan Xu, Weibin Meng, Ran Cheng, Yaxin Peng, Chaomin Shen, et al. Chatvla: Unified multimodal understanding and robot control with vision-languageaction model. arXiv preprint arXiv:2502.14420, 2025. 1, 2, 4, 7
- [111] Chenming Zhu, Tai Wang, Wenwei Zhang, Jiangmiao Pang, and Xihui Liu. Llava-3d: A simple yet effective pathway to empowering lmms with 3d-awareness. arXiv preprint arXiv:2409.18125, 2024. 3
- [112] Yuke Zhu, Roozbeh Mottaghi, Eric Kolve, Joseph J Lim, Abhinav Gupta, Li Fei-Fei, and Ali Farhadi. Targetdriven visual navigation in indoor scenes using deep reinforcement learning. In 2017 IEEE international conference on robotics and automation (ICRA), pages 3357–3364. IEEE, 2017. 3
- [113] Ziwen Zhuang, Zipeng Fu, Jianren Wang, Christopher G Atkeson, Sören Schwertfeger, Chelsea Finn, and Hang Zhao. Robot parkour learning. In Conference on Robot Learning CoRL, 2023. 3

#### A More Implementation Details

###### A.1 Training Details

Our VeBrain is optimized in a fully supervised finetuning (SFT) manner based on Qwen2.5-VL-7B-Instruct [8]. We let the whole parameters in large language model (Qwen2.5 LLM Decoder) trainable while keeping others frozen. The detailed training configuration is located in Tab.8.

Table 8: Hyper-parameters used in the training of VeBrain.

Configurations Values LLM sequence length 16384 Max pixel length 12845056 Freeze vision tower True Freeze multimodal projector True Freeze language model False Optimizer AdamW Optimizer hyperparameters β1 = 0.9, β2 = 0.999, eps = 1e − 8 Peak learning rate 5e-6 Learning rate schedule cosine decay Training epochs 1 Training steps 4865 Warm-up steps 500 Global batch size 128 Gradient accumulation 4 Numerical precision bfloat16

###### A.2 Data Generation Details

CoT Generation. In this section, we will introduce more details of CoT data generation, including the main designs and detailed prompts.

Specifically, our CoT data generation is primarily based on our annotated conversation data. Based on conversation data, we leverage diverse prompts to leverage closed-source MLLMs to generate the thinking process. A key constraint during this process is that the MLLMs are prohibited from pre-revealing the answer, ensuring that the generated CoT reflects genuine reasoning.

Besides, the selection of MLLMs used to annotate is also different. For visual-spatial reasoning, the input images derive from complex indoor scenes, which contain rich information. Therefore, to effectively use this information for CoT generation, we need to choose an MLLM with more creativity. Thus, for generating CoT in this task, we choose Gemini-2.0, which is more creative than GPT-4o despite being weaker in instruction following. During this process, we have some interesting observations. For instance, when assessing the distance between a chair and a door, Gemini 2.0 adopts a practical approach. It first estimates the chair’s width and then infers the distance based on how many chairs can fit between the chair and the door. In this way, these high-quality CoT data can enable VeBrain with strong spatial reasoning ability. On the other hand, GPT-4o tends to produce shorter content and relies on non-existent elements in the scene, such as floor tiles, as a basis for distance judgment.

In the embodied CoT generation, the complexity is significantly lower compared to 3D indoor scene video data, mainly due to the limited field of view of the robot. Therefore, we adopt a more conservative strategy and used GPT-4o for CoT generation. Key designs here include requiring the MLLMs to think from the robot’s perspective for obstacle avoidance, interaction, tracking, grasping, and destination-reaching, among other tasks. Since our control mechanism is based on a key-point policy, we visualize the key points on the input image using red circles. These marked images, combined with the QA information, are then fed into GPT-4o. This enables the robot to perform tasks such as obstacle avoidance and interaction with a high degree of reliability and interpretability. We also demonstrate the detailed prompt for generating CoT data for Complex Find tasks in locomotion.

In conclusion, our CoT generation strategy is tailored to the specific characteristics of visual spatial understanding and embodied scenes. By leveraging different models and data pre-processing techniques, we aim to enhance the interpretability and performance of VeBrain in real-world applications.

###### Table 9: Policies in the Policy Pool.

###### Platform Policy Description Policy Description

Dump Pour the loaded items out of the basket. Turn Right Turn right. Touch Lower head for humans to touch. Turn Left Turn left. Shake Shake hands with humans. Sit Sit down. Jump Jump forward. Wallow Roll left and right.

Legged Robots

Scrape Stand on two feet and scrape. Lie Down Lie down. Squat Lower the body height. Stand Up Recover to standing. Heart Stand on two feet and make a heart sign. Stretch Stretch the body.

Robot Arms

Grasp Close the robotic claw and grasp the item. Pull Pull in a specific direction. Release Open the robotic claw and release the item.

Data composition Our VeBrain-600k consists of 200k items from multimodal understanding, 312k items from visual-spatial reasoning, and 88k items from self-collected robotic control. Specifically, the 200k multimodal data are collected from open-source datasets, such as ShareGPT4V [17] and MMInstruct [67], etc. The 312k spatial reasoning data consists of 155k items released by GPT4Scene [81], and 157k self-collected video data generated from ScanNet [26] (We actually generate 31.4k items and repeat five times). The 88k robotic data consists of 76k from the legged robot and 12k from the robotic arm. Similarly we self-collect 15.2k and 1.2k non-redundant data items and repeat for five and ten times for the legged robot and robotic arm respectively.

###### A.3 Low-level Policy Details

Tab.9 presents all the collected actions in the policy pool, including 17 distinct implementations. With comprehensive coverage of legged robots and robot arms, this systematic organization enables robust control across different platforms.

###### A.3.1 Legged Robot

Setup. The setup for legged robots consists of Unitree Go2, a RealSense D435i RGB-D camera, and a Jetson AGX Orin platform. The camera is fixed at the head of Unitree Go2, providing egocentric visual perception, and calibrated to its optical center. All perception-action pipelines operate under an edge computing paradigm, utilizing the onboard computational resources, except for MLLM inference.

Point Tracker. In locomotion tasks, the visual observation varies as the legged robot moves, highlighting the necessity of temporal alignment between original and current keypoints. To enhance real-time performance, we adopt a series of optimizations and achieve a 15Hz execution frequency for our tracker model deployed on NVIDIA Jetson Orin. Specifically, we employ Locotrack-small[24], a lightweight yet efficient architecture enabling near-dense point tracking. The tracking model processes two consecutive observation frames while leveraging keypoint predictions from the preceding frame for temporal consistency.

Low-level Controller. As described in Sec.3.2.2, the 3D coordinates for MLLM-derived keypoints in the camera frame are obtained through a series of geometric transformations. These 3D coordinates are then projected into velocity control commands (vx,vy,vyaw), maintaining a mathematically constrained relationship tan(vyaw) =

vy vx

. The velocity control commands are then executed by a low-level walking policy that generates dynamically stable motion trajectories.

###### A.3.2 Robot Arm

Setup. The robot setup consists of a 7-DoF tabletop Franka Emika Panda robot arm equipped with a Robotiq 2F-85 gripper. A RealSense D435i RGB-D camera, positioned approximately 1.5m away from the robot in a third-person view, captures RGB-D scenes at each inference timestamp. The camera is calibrated

###### Table 10: Definition and skill involvement of embodied tasks.

###### Task Name Task Definition Skills Involved

▼ Locomotion Easy Tasks

Find Approach the target object. Walk Track Approach the target object and move along with its position. Walk Interaction Recognize human gestures, approach humans, and respond. Walk, Shake, Sit, Touch

▼ Locomotion Middle Tasks Complex Find Approach the target object with obstacle avoidance. Walk, Turn Left, Turn Right Complex Interaction Interaction with obstacle avoidance. Walk, Turn Left, Turn Right, Shake, Sit, Touch Transport Approach to the box, and dump the object into the box. Walk, Dump

▼ Locomotion Hard Tasks Complex Transport Transport with obstacle avoidance Walk, Turn Left, Turn Right, Dump

▼ Manipulation Tasks

Banana Move the banana on the table to the box Grasp, Release Pepper Move the pepper on the table to the box Grasp, Release Carrot Move the pepper outside of the box Grasp, Release Kiwifruit Move the kiwifruit outside of the box Grasp, Release Open Drawer Pull open the half-open drawer Pull

▼ Manipulation Long-Horizon Tasks

Carrot Open the drawer first, and move the carrot outside from the drawer. Grasp, Release, Pull Pepper Open the drawer first, and move the pepper outside from the drawer. Grasp, Release, Pull

###### Table 11: Ablation studies of VeBrain regarding dataset proportion, learning rate and training parameters. The default configuration on the last row best balance the three types of capabilities.

Multimodal Understanding Visual-spatial Reasoning Robot Control

Avg MMVet MMBench ScanQA (CIDEr) VSI-Bench Complex Find Transporting

Qwen2.5-VL [8] 67.1 83.5 62.7 35.9 0.0 10.0 43.2

▼ Dataset proportion: (Default: [Multimodal, Visual-spatial Intelligence, Robot control]=[200k,312k,88k])

400k,312k,88k 72.8 84.3 97.8 38.5 70.0 85.0 74.7 200k,400k,88k 71.5 83.5 101.6 40.2 75.0 85.0 76.1 200k,312k,200k 69.9 83.1 98.4 39.1 85.0 90.0 77.6 ▼ Learning rate: (Default: 5e-6)

2e-6 70.4 83.3 99.8 39.2 75.0 80.0 74.6

- 1e-5 43.5 70.2 90.2 36.9 80.0 90.0 68.5

▼ Training Manner: (Default: Fully finetuning)

LoRA Adapter 71.3 83.5 97.6 38.7 80.0 90.0 76.9 VeBrain (Default) 72.7 83.7 101.5 39.9 80.0 90.0 78.0

with the origin of the world coordinate frame is aligned with the base of the robot arm. Robot control is managed from a desktop computer running ROS.

Robot Control. Upon acquiring the RGB image Ic and the depth image Id captured by the camera, Ic together with prompt are fed into VeBrain to predict 2 target points, p1 and p2, which correspond to the intended antipodal grasping positions for the gripper at each inference. To determine the target pose of the robot arm, the translation component is obtained by querying the depth value at the midpoint p∗ of p1 and p2 in Id, and subsequently transforming this pixel location into the world coordinate frame. As for rotation, we adopt a top-down grasping paradigm, which requires only adjusting the gripper’s orientation around its z-axis according to the direction defined by p1 and p2. The computed 6D end-effector pose is then passed to the MoveIt library1 for motion planning. The gripper’s closing width is determined by a predefined force threshold. Notably, when the robot task involves opening a drawer, the gripper skips the closing action.

###### A.4 Robotic Task Details

In this paper, we design hierarchical embodied tasks for both locomotion and manipulation. The definitions of each task are shown in Tab. 10. For locomotion, the Easy Tasks include three key tasks: find, track, and interaction. These tasks evaluate the basic abilities of VeBrain-equipped legged robots in static object searching, dynamic following, and human intention recognition, respectively. The Interaction task involves

1https://github.com/moveit/moveit

four gestures: come, sit, shake, and touch. In the Middle Tasks, we introduce obstacle avoidance into the find and interaction tasks, and add a new task, transport, to assess VeBrain’s practical application capabilities. Finally, the Hard Locomotion Tasks involve complex transport with obstacle avoidance, further challenging the robot’s mobility and coordination in dynamic environments.

For manipulation, we also design two levels of tasks: normal tasks and long-horizon tasks. The normal manipulation tasks include moving objects like bananas, peppers, and kiwis to specific locations, e.g., from the table to the box or outside the box, as well as pulling open a half-open drawer. These tasks test the basic manipulation skills, such as grasping and releasing. The long-horizon manipulation tasks require performing multiple-step sequences, such as opening the drawer first and then moving the carrot or pepper outside of it. These tasks assess the robot’s ability to execute complex, multi-step actions and demonstrate precise control over its manipulation process.

In our evaluation, all tasks, except for the Interaction and Complex Interaction tasks, will be tested 10 times to ensure reliability. For the Interaction task, each of the four gestures (come, sit, shake, touch) will be tested 5 times to assess the robot’s responsiveness and accuracy in recognizing human gestures. Moreover, the object layout in the scene will be slightly modified in each test.

#### B More Results

###### B.1 Ablation Study

In Tab. 11, we further conduct ablation studies from three perspectives. 1) For the dataset proportion, our default VeBrain-600k comprises 200k from multimodal conversations, 312k from visual-spatial reasoning, and 88k from self-collected robotic dataset. We respectively increase the amount of data items for the three types of tasks, and found that the corresponding capability increases slightly but at the cost of the other two capabilities. Therefore, we maintain the original data proportion for the best trade-off among the three types of capabilities. 2) For the learning rate, we adopt the peak learning rate of 5e-6. A lower learning rate, like

- 2e-6, results in marginal degradation across all three capabilities, while a higher learning rate, such as 1e-5, leads to a remarkable decline in multimodal understanding performance. 3) If we optimize VeBrain via LoRA adapter rather than fully optimize the LLM, the overall performance also shrinks slightly due to the less learnable parameters. The ablations for the above three perspectives further substantiate the superiority of training configurations in our default setting.

