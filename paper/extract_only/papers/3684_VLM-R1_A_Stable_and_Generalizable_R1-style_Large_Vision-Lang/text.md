# arXiv:2504.07615v2[cs.CV]14Apr2025

## VLM-R1: A Stable and Generalizable R1-style Large Vision-Language Model

Haozhan Shen1, Peng Liu2, Jingcheng Li2, Chunxin Fang2, Yibo Ma2, Jiajia Liao2, Qiaoli Shen2, Zilun Zhang1, Kangjia Zhao1, Qianqian Zhang2, Ruochen Xu2, Tiancheng Zhao2,3

1 Zhejiang University 2 Om AI Research 3 Binjiang Institute of Zhejiang University

{tianchez}@zju-bj.com

[Figure 1]

[Figure 2]

[Figure 3]

Base Model VLM-R1 Framework

RL Model

[Figure 4]

[Figure 5]

Data preparation

Build VLM Module

This is our constructed component to support various VLMs

We only use non-reasoning open-source grounding data

Only train the model with 500 steps

###### GRPO

[Figure 6]

[Figure 7]

<think>...</think> <answer>...</answer>

lady with the blue shirt

| |
|---|

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

<think>...</think> <answer>...</answer>

<think> The image shows a desk with various items including a laptop, a red mug, a notebook, a pen holder, an alarm clock, and a small potted plant. The red mug is placed on the desk near the laptop. Given that mugs can contain liquids like coffee or tea, which could potentially cause damage if spilled onto electronic devices, it is reasonable to assume that the red mug might have the potential to short-circuit the laptop if spilled. </think> <answer>{"bbox_2d": [379, 336, 558, 520],...}</answer>

Question: Which item may have the potential to short-circuit the laptop?Please provide the bounding box coordinate of this region.

| |
|---|

Policy Model

[Figure 12]

[Figure 13]

<think>...</think> <answer>...</answer>

###### QwenVL, InternVL, LLaVA, LLaMA-V etc: ```json [

| |
|---|

summarize reward

{"bbox_2d": [647, 103, 765, 306], ...}

🔥Update

] ```

Figure 1. VLM-R1 provides a standard pipeline to enhance base vision-language models (VLMs) with reinforcement learning.

### Abstract

Recently, DeepSeek R1 has shown that reinforcement learning (RL) can substantially improve the reasoning capabilities of Large Language Models (LLMs) through a simple yet effective design. The core of R1 lies in its rule-based reward formulation, which leverages tasks with deterministic ground-truth answers to enable precise and stable reward computation. In the visual domain, we similarly observe that a wide range of visual understanding tasks are inherently equipped with well-defined ground-truth annotations. This property makes them naturally compatible with rule-based reward mechanisms. Motivated by this observation, we investigate the extension of R1-style reinforcement learning to Vision-Language Models (VLMs), aiming to enhance their visual reasoning capabilities. To this end, we develop VLM-R1, a dedicated framework designed to harness RL for improving VLMs’ performance on general

Corresponding author.

vision-language tasks. Using this framework, we further explore the feasibility of applying RL to visual domain. Experimental results indicate that the RL-based model not only delivers competitive performance on visual understanding tasks but also surpasses Supervised Fine-Tuning (SFT) in generalization ability. Furthermore, we conduct comprehensive ablation studies that uncover a series of noteworthy insights, including the presence of reward hacking in object detection, the emergence of the “OD aha moment”, the impact of training data quality, and the scaling behavior of RL across different model sizes. Through these analyses, we aim to deepen the understanding of how reinforcement learning enhances the capabilities of vision-language models, and we hope our findings and open-source contributions will support continued progress in the visionlanguage RL community. Our code and model are available at https://github.com/om-ai-lab/VLM-R1.

### 1. Introduction

The introduction of OpenAI o1 [20] demonstrated that reinforcement learning (RL), which enables Large Language Models (LLMs) to directly learn from feedback on their outputs, can significantly enhance their reasoning capabilities. More recently, DeepSeek R1 [17] further advanced this insight by showing that simple rule-based rewards, without the need for a separate learned reward model [28, 39, 56], are sufficient to autonomously equip LLMs with complex reasoning performance.

A key factor behind this success is that the rule-based reward design is easily applicable to tasks with deterministic ground-truth answers, allowing for stable and interpretable reward signals. In the visual domain, analogously, there exist numerous visual understanding tasks inherently including precise and objectively defined groundtruth annotations. For example, tasks such as Referring Expression Comprehension (REC) [37, 55] can directly adopt Intersection-over-Union (IoU) between the predicted bounding box and the ground-truth annotation as an explicit reward metric. Motivated by these observations, it becomes intuitive to investigate whether similar RL methodologies can comparably enhance the reasoning capabilities of Vision-Language Models (VLMs).

To this end, we develop VLM-R1, a dedicated and extensible framework designed to apply RL to improve the performance of VLMs on general vision-language tasks. VLM-R1 is built with flexibility, scalability, and ease of experimentation in mind. It supports a wide range of configurations and is tailored for research on RL-based optimization in the context of VLMs. Key features of VLM-R1 include:

- • GRPO Compatibility: Fully supports the native GRPO [46] algorithm with fine-grained control over all hyperparameters.
- • LoRA-based Training: Enables parameter-efficient training via LoRA [18], suitable for limited-resource settings.
- • Multi-node Training: Supports distributed training across multiple GPUs or server nodes for scalability.
- • Multi-image Input: Supports multiple images per sample, facilitating complex multi-image reasoning tasks.
- • Model Flexibility: Compatible with various VLMs, currently supporting QwenVL [6, 50] and InternVL [10, 11].
- • Custom Dataset Support: Easily integrates user-defined datasets, allowing for task-specific or domain-specific experiments
- • Mixed Modality Training: Supports training on both image-text and pure-text datasets, including hybrid combinations.

By providing a unified, modular, and highly adaptable training pipeline, VLM-R1 serves as a powerful tool for advancing research at the intersection of reinforcement learning and vision-language modeling.

Model Model Size Refcocoval Refcoco+val Refcocogval ODinW

Qwen2.5-VL-3B 3.75B 89.1 82.4 85.2 37.5 Grounding DINO 341M 90.6 88.2 86.1 55.0

Table 1. Performance comparison between Qwen2.5-VL-3B and Grounding DINO on REC and OVD tasks. Even though having over 10× the number of Grounding DINO, Qwen2.5-VL-3B still falls short on these evaluation datasets. It shows the drawback of VLMs on these visual understanding tasks.

In this report, utilizing VLM-R1, we select two visual understanding tasks — Referring Expression Compression (REC) and Open-Vocabulary Object Detection (OVD) — to explore the feasibility and effectiveness of applying RL to VLMs. REC and OVD share a common output format—bounding boxes—but differ significantly in task complexity. In REC, the model is expected to predict a single bounding box based on a given query, whereas in OVD, the model must accurately output each corresponding bounding box for every queried target. This contrast allows us to analyze how tasks with similar output structures but varying difficulty levels influence the effectiveness of reinforcement learning in VLMs. Moreover, we observe that VLMs often underperform compared to specialized vision models (e.g., Grounding DINO [32, 44], OmDet [59, 60]) on these tasks. As shown in Table 1, despite having over 10× the number of parameters as Grounding DINO, Qwen2.5-VL-3B still lags behind in performance on both REC and OVD benchmarks. This performance gap raises an important question: can reinforcement learning be leveraged to enhance VLMs’ effectiveness on these challenging visual understanding tasks?

The experimental results demonstrate that RL substantially improves visual understanding performance in VLMs compared to supervised fine-tuning (SFT), and, more importantly, yields significantly greater gains in generalization ability on complicated, real-world benchmarks. In the context of REC, our 3B RL model achieves a score of 63.16 on the out-of-domain evaluation benchmark LISAGrounding [24](vs. 54.82 for SFT). For OVD task, the 3B RL model reaches 21.1 AP on COCO [27](vs. 17.8 for SFT; 14.2 for 7B baseline model), and new SOTA 31.01 nms-AP on OVDEval [54](vs. 26.50 for SFT; 29.08 for 7B model), especially excelling in complex sub-tasks.

In addition, comprehensive ablation studies further uncover a range of important insights. For instance, we observe the reward hacking in object detection, and conduct reward engineering to alleviate it, where the model emerges “OD aha moment”, first reasoning about object presence before predicting. Furthermore, we also demonstrate that the careful selection of training data could improve the final performance, and analyze the impact of model size. Taken together, our findings suggest that more complex taskssuch as open-vocabulary object detection—demand addi-

tional optimization to achieve strong performance, whereas relatively simpler tasks like REC can be effectively addressed with fewer modifications. Our contributions could be summarized as:

- • We develop VLM-R1 based on open-r1, a dedicated and extensible framework designed to apply reinforcement learning to improve the performance of vision-language models, aiming for flexibility, scalability, ease of experimentation, and the supporting a wide range of RL configurations.
- • We demonstrate the effectiveness of applying reinforcement learning to vision-language models through training two essential visual understanding tasks: referring expression compression and open-vocabulary object detection. Trained with VLM-R1, our RL model achieves a performance improvement compared to the SFT counterpart, especially on the complicated, real-world out-ofdomain benchmarks.
- • Our extended ablation studies reveal a series of interesting insights, including the presence of reward hacking in object detection, the emergence of “OD aha moment”, the influence of training data quality, and the RL effects across model scales. We report these insights and analyze how to well-tune reinforcement learning to enhances the performance of VLMs.
- • We release the framework codebase and all model weights, with the hope of contributing to the open-source community in vision-language reinforcement learning.

### 2. Related Work

#### 2.1. Vision-Language Models

Since the advent of large language models (LLMs), they have achieved success in various linguistic applications, facilitating the emergence of Vision-Language Models (VLMs), with pioneering works including [4, 22, 26]. Following these, LLaVA [31] employed GPT-4 [2] to develop training data and achieve promising performance in visual dialogue and visual reasoning, inspiring a series of works focused on visual instruction data [8, 13, 29]. However, a key limitation of the VLMs at that time lies in their constrained image input resolution, which is restricted by the capabilities of their underlying vision encoders [43, 47, 57]. To overcome this, the AnyRes mechanism was introduced [7, 11, 30], allowing flexible handling of images with varying resolutions and aspect ratios. This advancement improves the perceptual capacity of VLMs for diverse visual inputs and further enhances their reasoning abilities. Today, some of the most widely adopted open-source VLM series include LLaVA[25, 30, 31], QwenVL[6, 50], and InternVL [10, 11].

#### 2.2. Attempts of applying R1 to VLMs

Several concurrent studies have explored the application of R1 to Vision-Language Models (VLMs). Concurrent work R1-OneVision [53] and R1-V [9] are among the notable works in this direction. R1-OneVision proposed a crossmodal reasoning pipeline that converts images into visual formal representations, which are then used to construct a visual reasoning dataset via a language model. The VLM is first trained on this dataset, followed by an RL stage to further enhance its reasoning capability. In parallel, R1-V introduced the GRPO method [46] from DeepSeek R1 [17] into VLM training, targeting object-counting tasks, and remarkably enabled a 3B model to outperform a 72B model. Soon afterward, VisualThinker-R1-Zero [61] was presented, which demonstrated that applying R1 to base VLM instead of an instruction-tuned model could achieve more considerable performance improvements and successfully trigger the emergence of the so-called “visual aha moment”. Another work observing the appearance of the aha moment and the increasing length of the model response that is akin to the phenomena in DeepSeek R1 is MMEureka [38], which applied RLOO [3, 23] to both the 8B instruction-tuned VLM and the 38B base VLM. Analogous to R1-OneVision, Vision-R1 [19] constructed a multimodal CoT dataset in terms of converting vision information to a language format and feeding it into a language reasoning model. This dataset serves as the cold start training data followed by the GRPO to further strengthen the multimodal reasoning ability of the model. In addition, Curr-ReFT[14] proposed a three-stage reinforcement learning with progressive difficulty-level reward to optimize RL training, and LMM-R1[42] presented a two-stage rule-based RL, where they first adopted text-only data to strengthen the basic reasoning abilities of the model and then continued RL on limited complex multimodal reasoning tasks.

Most of the above studies focus mainly on improving performance in multimodal mathematics tasks [36, 48, 58]. In contrast, Visual-RFT [35] applies RL to visual perception tasks, making it more closely related to our work. However, our study provides a more comprehensive investigation, going beyond a simple comparison between supervised finetuning (SFT) and RL. Specifically, we further analyze the role of reward engineering and systematically examine the impact of careful training data selection, particularly for complex tasks.

### 3. VLM-R1 Framework

In this section, we provide a brief introduction to the proposed VLM-R1 framework. VLM-R1 is built upon OpenR1 [16], an open-source framework for reproducing the language reasoning capabilities of DeepSeek R1. We extend its implementation to the vision-language domain.

grpo_jsonl.py

legend

Support User-custom Reward Function

Support User-custom Training Data

Support Various VLMs

function point key features of framework function transformation

Define 📚 Reward Function

VLM Module Definition

Data 🤖 Preparation

🎯

abc.py python file name

[Figure 14]

good reward bad reward

[Figure 15]

grpo_trainer.py

🛠Model Init Confirm

🔄 Calculate

Generate x sequences

Compute GRPO Loss

🤔

📉

📝

Training Param

Reward

x sequences

[Figure 16]

[Figure 17]

[Figure 18]

Support all GRPO Param

[Figure 19]

if peft_config is not None: LoRA Fine-tuning

elif args.freeze_vision: Only Fine-tuning LLM

else: Full Fine-tuning

x = args.num_generation

Figure 2. Flow chart of VLM-R1 framework. This chart exhibits the function transformation of the framework. The key features of VLM-R1 are displyed by the green rectangle.

In addition to ours, there are several other open source frameworks that target vision language reinforcement learning [1, 9]. It should be noted that our primary objective is to adapt the R1-style methodology for Vision-Language Models (VLMs). Therefore, our current implementation focuses exclusively on the GRPO [46] algorithm, as originally adopted by DeepSeek R1. Consequently, VLM-R1 currently supports only GRPO, with plans to integrate additional RL algorithms in future work. In the following, we first present an overview of the framework, followed by a detailed description of the VLM Module, which enables seamless support for various VLM architectures.

#### 3.1. Overview

As shown in Figure 2, the VLM-R1 framework is composed of two main components: grpo jsonl.py and grpo trainer.py, which together form a complete pipeline for GRPO [46] algorithm to VLMs.

In the first stage (grpo jsonl.py) serving as a preparation stage, users can flexibly define custom reward functions and prepare training data tailored to their tasks. The framework also supports various VLMs through a modular VLM Module Definition, which will be described in § 3.2. The second stage (grpo trainer.py) manages the GRPO training process. It begins with model initialization, followed by confirmation of training parameters decided by the usercustom parameters. We support LoRA fine-tuning, vision tower freezing training, and full parameters training, which could be adapted to distinct compute resources and task requirements. The model subsequently generates multiple sequences, which are scored using the defined reward function. These reward signals are then used to compute the GRPO loss for parameter optimization.

GRPO Tainer

VLM Module

Model init

get_model_class

model = model_cls. from_pretrained(...)

Qwen2_5_VLForConditionalGeneration

InternVLChatModel LlavaNextForConditionalGeneration

.......

Freeze vision tower

###### get_vision_modules_keywords

for n, p in vision_tower: p.require_grad = False

QwenVL: ['visual'] InternVL: ['vision_model'] .......

Inject the question into the chat template

prepare_prompt

Question

<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user <image> Question<|im_end|> <|im_start|>assistant

prepare_model_inputs

Convert image and prompt into input_ids

More other functions

Image, Prompt

(see our github repo for details)

{

input_ids: ... attention_mask:... pixel_valuse:...

To be implemented future

(welcome PR or issue)

}

Figure 3. The interaction between Trainer and VLM Module. With the VLM Module, the GRPOTrainer can interact with different VLMs by simply invoking the standardized interfaces without the need to handle model-specific implementations.

VLM-R1 provides full support for GRPO training while offering flexibility in reward design, model selection, and optimization strategies, making it a versatile tool for RLbased vision-language research.

#### 3.2. VLM Module

To facilitate the seamless integration of various VLMs into the training process, we design a unified component, which we refer to as VLM Module. This module encapsulates general VLM functionalities, such as retrieving the model’s class name and formatting input questions into the modelspecific chat template. By abstracting these operations, the GRPOTrainer can interact with different VLMs by simply invoking the standardized interfaces provided by the VLM Module, without the need to handle model-specific implementations. This design not only streamlines the integration of new models but also enhances the modularity and readability of the overall framework. The interaction between Trainer and the VLM Module is shown in Figure 3.

### 4. Reward Design

As discussed in § 1, we select referring expression comprehension (REC) and open-vocabulary object detection (OVD) as representative tasks due to two considerations. First, both tasks share a common bounding box output format but differ in complexity, providing a suitable setting to examine the impact of RL across tasks with varying difficulty. Second, specialized vision models have consistently outperformed VLMs on these benchmarks, offering a valuable opportunity to assess whether RL can help close this performance gap.

In this section, we first brief the general GRPO algorithm and then introduce the reward design for REC and OVD tasks that be integrated into the GRPO.

#### 4.1. Abstraction of GRPO

Unlike reinforcement learning algorithms such as PPO [45], which require an additional critic model to estimate policy performance, Group Relative Policy Optimization (GRPO) directly compares groups of candidate responses, eliminating the need for a separate critic. Given a question q, GRPO samples N candidate responses {o1,o2,...,oN} from the policy πθ and evaluates each response oi using a reward function R(q,oi), which measures the quality of the candidate in the context of the given question. To determine the relative quality of these responses, GRPO normalizes the rewards by computing their mean and standard deviation and subsequently derives the advantage as:

ri − mean{r1,r2,...,rN} std{r1,r2,...,rN}

(1)

Ai =

where Ai represents the advantage of the candidate response oi relative to other sampled responses. GRPO encourages the model to generate responses with higher advantages within the group by updating the policy πθ using

the following objective:

JGRPO(θ) = E[{oi}Ni=1 ∼ πθ

(q)] (2) 1

old

N

{min[s1 · Ai, s2 · Ai] − βDKL[πθ||πref]} (3)

N

i=1

- s1 =

πθ(oi|q) πθ

old

(oi|q)

(4)

- s2 = clip

πθ(oi|q) πθ

,1 + ϵ,1 − ϵ (5)

(oi|q)

old

As mentioned in 3.1, all hyperparameters in the above equations are included in our proposed VLM-R1 framework.

Subsequently, we will introduce the reward function R adopted for REC and OVD tasks. Following DeepSeek-R1, we use two types of rewards: accuracy reward and format reward.

#### 4.2. Reward function for referring expression comprehension

Accuracy reward. Referring expression comprehension (REC) is a task that requires the model to identify the region bounding box of the object described by a referring expression. Denote q as the input question, b∗ as the ground truth bounding box, o as the VLM output sentence, and frec as the function that is used to extract the bounding box from the output sentence. The accuracy reward for REC is defined as:

Raccrec(q,o) = IoU(b∗,frec(o)) (6)

where IoU is the intersection over union metric. This reward function is designed to encourage the model to generate bounding boxes that closely match the ground truth.

Format reward. Format reward of REC checks whether the response follows the specified format that require the model has to output the json-style response in the <answer> tag and include a boundingbox (<think>...</think><answer>{...[x1, y1, x2, y2]...}</answer>), returning 1 or 0 based on compliance.

#### 4.3. Reward function for open-vocabulary object detection

Accuracy reward. Open-vocabulary object detection (OVD) requires the model to detect the given object labels in the image and output the corresponding bounding boxes and class labels. This task has a similar output format as REC, but is more complex due to the need for both bounding box and class label generation. In this task, we prompt the VLM to output a list of bounding boxes along with their corresponding class labels, which can be extracted as a list of combination bpred = {(b1,c1),(b2,c2),...,(bn,cn)}

by a function fovd, where bi is the bounding box and ci is the class label. Let q denote the input question, mAP(·) the function calculating mean average precision metric, bgt the list of the combination of ground-truth bounding-boxes and class labels, Lgt the number of the ground truth combinations, and Lpred the number of the predicted combinations. The accuracy reward for OVD is defined as:

Lgt Lpred

) (7)

sovd = min(1,

###### Raccovd(q,o) = sovd · mAP(bpred,bgt) (8)

where sovd is the penalty factor to a redundant prediction from VLMs, and our experiment shows that this penalty factor is helpful to improve the performance on OVD task. This reward is designated as odLength reward.

Format reward. Format reward of OVD checks whether the response follows the specified format, which requires the model to output a markdown-style JSON response in the <answer> tag (<think>...</think><answer> ‘‘‘json...‘‘‘</answer>), returning 1 or 0 based on compliance.

### 5. Experiments

#### 5.1. Implementation details

Selected VLM. We employ Qwen2.5VL-3B-Instruct as our base model due to its strong potential performance on vision-language understanding that is expected to be exploited through reinforcement learning, and we also introduce Qwen2.5VL-7B-Instruct and 32B in some experiments to investigate the model size impact.

Hyper-parameters setup. When training REC with RL, we adopt the default GRPO parameter settings, setting N to 8, temperature to 0.9, number of iterations to 1, and the KL divergence ratio (i.e., β) to 0.04. We train the model for 2 epochs, using a learning rate of 1e-6 for both RL and SFT. For OVD, we set only β to 0, keeping all other parameters identical.

Prompt template.

Problem Template of REC

Please provide the bounding box coordinates of the region this sentence describes: {query}.

[Figure 20]

Figure 4. Difference between in-domain and out-of-domain dataset for REC task. In-domain data only describes the spatial or appearance attribute information for the object, while out-ofdomain data require the model to use the open-world knowledge to recognize the role of soccer goalkeeper and then locate it.

##### Problem Template of OVD

Please carefully check the image and detect the following objects: {target list}. Output each detected target’s bbox coordinates in JSON format. The format of the bbox coordinates is: “‘json [”bbox 2d”: [x1, y1, x2, y2], ”label”: ”target name”, ”bbox 2d”: [x1, y1, x2, y2], ”label”: ”target name”] “‘. If there are no such targets in the image, simply respond with None.

##### Thinking Prompt

{problem} Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.

#### 5.2. Main results

##### 5.2.1 Referring expression comprehension

Training dataset. We use the training splits of Refcoco/+/g [37, 55] as our training data. These are the most widely used datasets for the REC task and primarily contain descriptions of objects based on spatial or appearance attributes without involving explicit reasoning information. Our objective is to investigate whether a model trained on this non-reasoning dataset can generalize reasoning capabilities acquired through the RL process to more challenging evaluation scenarios.

Evaluation dataset. We select the val split of Refcoco/+/g [37, 55] for in-domain evaluation and test split of LISA-Grounding [24] for out-of-domain evaluation. LISAGrounding is a reasoning-intensive dataset that requires models to perform fine-grained visual perception, precise

Evaluation Training Steps Training method Dataset 0 100 200 300 400 500 600

SFT

88.7 88.7 88.85 88.7 88.25 88.85 88.7

Refcocoval

RL 88.7 88.7 88.7 89.4 89.25 90 90.55 SFT

81.95 82.55 82.15 81.85 81.9 82.3 82.25 RL 81.95 82.6 81.9 82.8 83.35 83.6 84.3

Refcoco+val

SFT

86.05 85.65 85.95 85.85 85.6 85.95 85.95 RL 86.05 85.95 85.05 85.45 85.65 87.15 87.1

Refcocogval

SFT

56.51 55.91 56.51 55.66 55.18 55.66 54.82 RL 56.51 61.82 61.27 61.64 62.6 61.88 63.14

LISA-Grounding

∆RL−SFT 0 +5.91 +4.76 +5.98 +7.42 +6.22 +8.32

Table 2. Performance comparison of SFT and RL on in-domain and out-of-domain evaluation datasets. All results are from Qwen2.5VL3B-Instruct trained on the training split of Refcoco/+/g. Step 0 represents the results from Qwen2.5VL-3B-Instruct itself. ∆RL−SFT denotes the improved value of the RL model compared to the SFT model.

Performance on in-domain test data (Avg Acc on Val split of RefCOCO/+/g)

COCOfiltered Model mAP GP (IoU=0.5) GR (IoU=0.5)

Base 3B 14.2 56.06 33.79 Base 7B 14.4 54.73 33.36 SFT Model 3B 18.5 53.15 39.4 RL Model 3B 21.1 67.34 43.84

87.31

RL SFT

86.91

- 84
- 85
- 86
- 87

86.08 85.88

85.75

Acc

85.65

Table 3. Results of OVD task on COCOfiltered. Base 3B denotes the Qwen2.5VL-3B-Instruct and Base 7B denotes the 7B model. GP and GR represents Greedy Precision and Greedy Recall, respectively.

85.57 85.7

85.63

85.63

85.47

85.22 85.25

Grounding serves as a crucial test of the model’s ability to generalize its reasoning skills, acquired from less reasoning-demanding in-domain datasets, to a significantly more challenging out-of-domain scenario.

100 200 300 400 500

600

Steps

Results. Table 2 shows the performance of the SFT and RL models in four datasets, with a corresponding visualization provided in Figure 5 for a clearer comparison. In the in-domain test data, the SFT model shows limited improvement over the base model (i.e., steps 0), regardless of the number of training steps, while the RL model consistently achieves steady performance gains (top of Figure 5). More critically, on the out-of-domain test data, the SFT model suffers from a slight performance degradation as training progresses. In contrast, the RL model effectively generalizes its reasoning ability to the out-of-domain setting, maintaining stable and superior performance (bottom of Figure 5). These results clearly demonstrate the advantage of reinforcement learning in improving the generalization of VLMs in challenging scenarios that require intense reasoning.

63.14

62.6 61.64

61.82

61.27

RL SFT

Acc

- 55

60

Performance on out-of-domain test data (Acc on LISA-Grounding)

61.88

- 56.51

55.91 56.51

55.66

55.66

55.18

54.82

50

100 200 300 400 500

600

Steps

Figure 5. Performance comparison between the SFT and RL models. The RL model shows significantly better generalization on the out-of-domain evaluation dataset compared to the SFT model.

##### 5.2.2 Open-vocabulary object detection

understanding of referring expressions, and relational reasoning among objects to correctly localize the target bounding box. An example of the difference between the two datasets is shown in Figure 4. The evaluation on LISA-

Training dataset. We use Description Detection Dataset (D3)[52] as our training data, which provides several unique advantages for training object detection models: (1) complete annotation coverage, (2) unrestricted

Model Celebrity Logo Landmark Color Material Position Relationship Negation Overall NMS-AP Specialized OVD Model

Grounding-DINO [32] 0.7 10.3 15.1 9.4 9.0 67.5 10.7 52.5 25.30 OmDet [60] 1.8 6.1 26.3 22.9 16.3 21.2 41.98 35.1 25.86

###### VLM

Base 3B 13.2 26.5 21.3 2.9 11.6 47.9 13.1 38.7 25.46 Base 7B 48.4 35.8 44.6 3.0 10.5 40.5 16.2 39 29.08 SFT Model 3B 50.4 34.9 50.7 4.3 7.6 33.7 13.1 34.4 26.50 RL Model 3B 55.0 34.6 47.9 4.5 9.7 42.9 21.5 37.7 31.01

∆RL−SFT +4.6 -0.3 -2.8 +0.2 +2.1 +9.2 +8.4 +3.3 +4.51

- Table 4. Results of OVD task on OVDEval. Base denotes the Qwen2.5VL-3B-Instruct, and Base 7B denotes the 7B model. ∆RL−SFT denotes the improved value of the RL model compared to the SFT model. We also list the performance of OmDet, the current state-of-theart in specialized open-vocabulary detection, for the comprehensive comparison.

language descriptions, (3) instance-level annotations, and (4) absence expression support. During training, we randomly introduce 1∼3 descriptions from other training samples as negative expressions.

Evaluation dataset. We select COCOfiltered and OVDEval [54] for evaluation. COCOfiltered is created from the COCO [27] dataset’s instances val2017.json file. Since VLMs generally struggle at recall in OD tasks (see [21] for details), we filter out categories with more than 10 annotation boxes, ensuring that only categories with fewer boxes are included. OVDEval is utilized to evaluate the model’s capabilities. This is a comprehensive benchmark specifically designed for open-vocabulary detection, which systematically evaluates models across six key linguistic aspects 1. It further introduces hard negative samples to assess robustness and a novel NMS-AP metric to address the “Inflated AP Problem” common issues, providing a more accurate OVD assessment. All output boxes generated from VLM are assigned a confidence score of 1 when calculating AP. During COCO evaluation, {target list} is consistently set as all COCO 80 categories. For OVDEval evaluation, we keep the official evaluation setting.

Results. Table 3 shows the performance on COCOfiltered. The RL-trained model demonstrated substantial improvements over the SFT model, with a 2.6 percentage point increase in mAP (21.1% vs 18.5%), 4.42 points higher in Greedy Precision (57.57% vs 53.15%), and 4.33 points better in Greedy Recall (43.73% vs 39.4%). These consistent improvements across all metrics demonstrate RL’s superior generalization capability.

On the more challenging and comprehensive benchmark OVDEval, from Table 4, it is observed that the RL model demonstrated superior generalization by outperforming SFT in 7 out of 9 detection categories. Most notably, it shows significant improvements in complex tasks requir-

1Including Object detection, Proper noun recognition (celebrities, logos, landmarks), Attribute detection, Position understanding, Relationship comprehension , and Negation handling

ing deeper understanding: Position detection (+9.2 points), Relationship detection (+8.4 points), and Negation handling (+3.3 points). Moreover, although SFT shows strong performance in specific categories like Celebrity, Logo, and Landmark detection, RL demonstrates more balanced improvements across different visual tasks, suggesting better overall generalization of visual understanding.

The results demonstrate that while SFT can be effective for certain specific tasks, RL provides more comprehensive improvements. The 4.51 point improvement in average nms-ap (31.01 vs 26.50) indicates RL’s superior ability to learn generalizable features.

Comparison with SoTA OD: OmDet. OmDet [60] represents the current state-of-the-art in specialized openvocabulary detection. However, our VLM-R1 model demonstrates that VLMs can outperform specialized architectures in several key aspects.

The performance gap between RL model and OmDet reveals interesting insights about the strengths and limitations of different approaches:

- • World Knowledge and Entity Recognition: In celebrity detection, VLM-R1 achieves 55.0 nms-ap compared to OmDet’s 1.8. This dramatic difference (>50 points) demonstrates the value of VLMs’ pre-trained world knowledge, and similar patterns appear in logo and landmark detection, where semantic understanding is crucial.
- • Fine-grained Detection: We note that the attribute category in OVDEval contains a lot of small objects. In these small-object detection scenarios, OmDet shows a stronger performance gap (color: 22.9 vs 4.5). This suggests specialized architectures excel at fine-grained, local feature detection.

These comparisons suggest a promising future direction: combining the complementary strengths of both approaches. Specialized OD architectures excel at finegrained detection and high-recall scenarios, while VLMs bring rich world knowledge. Future research could focus on creating hybrid architectures that leverage both the pre-

COCOfiltered OVDEval

###### Reward mAP GP GR Cel Logo Land Color Mat Pos Rel Neg Overall

AP50 11.4 45.33 31.85 7.7 13.8 17.3 2.2 3.5 39.3 17.0 35.2 21.46 mAP 11.8 46.02 33.34 10.5 15.2 18.7 2.3 4.1 39.6 15.9 34.9 21.68

sovd · mAP 21.1 67.34 43.84 55.0 34.6 47.9 4.5 9.7 42.9 21.5 37.7 31.01

- Table 5. Performance comparison across AP50 reward, mAP reward and odLength reward. All results are obtained by the RL model trained from Qwen2.5VL-3B-Instruct. GP: Greedy Precision; GR: Greedy Recall; Cel: Celebrity; Land: Landmark; Mat: Material; Pos: Position; Rel: Relationship; Neg: Negation .

cise localization abilities of dedicated OD models and the semantic understanding of VLMs.

#### 5.3. Ablations & Extended experiments

##### 5.3.1 Investigation about “reward hacking”

What is reward hacking? Reward hacking [5] in reinforcement learning refers to a phenomenon where an agent exploits loopholes in the reward function to achieve high reward without truly fulfilling the intended task. This occurs when the reward function is misaligned with the designer’s actual goals, leading the agent to adopt unintended or shortcut behaviors. For instance, in a maze navigation task where the agent receives +1 for each step and +100 for exiting the maze, the agent may learn to walk in circles indefinitely to accumulate step rewards rather than solving the maze. Such behavior technically maximizes reward but fails to meet the task’s true objective. Several literature [15, 33, 40, 41, 49, 51] also investigate this phenomenon in large language model research.

Reward hacking in OVD task. Table 5 exhibits the superior performance of our proposed odLength compared to the native AP50 and mAP reward. Upon closer examination, we identify key limitations of the native AP50 and mAP reward functions. Specifically, we observe that when computing AP values using the official COCO evaluation API, categories not present in the ground truth for a given image are excluded from the evaluation. Given our prompt design, which consistently includes all positive and several negative categories, the model is incentivized to predict all categories in order to maximize its reward—an instance of reward hacking. This behavior negatively impacts precision when evaluated on the full dataset, where all COCO 80 categories are present, and no category will be excluded. In contrast, our odLength reward addresses this issue by introducing an additional penalty term for redundant predictions. This encourages the model to align the number of predicted objects with the ground truth, thereby promoting more precise and faithful outputs from VLMs.

Visualization of the completion length. Figure 6 illustrates the variation in output sequence lengths across different reward settings. Notably, models trained with the native AP50 reward—particularly those without KL

[Figure 21]

Figure 6. Visualization of the completion length across different reward settings on OVD task. It is observed that the model always generates the overlong completion with the native AP reward, indicating the redundant predicted objects.

regularization—exhibit a dramatic increase in output length over the course of training. This trend indicates the presence of severe reward hacking, where the model is incentivized to enumerate an excessive number of object categories in order to maximize the reward, leading to highly redundant outputs. In contrast, models trained with our proposed odLength reward maintain stable and significantly shorter outputs, effectively suppressing unnecessary predictions.

OD aha moment. Figure 7 illustrates the cases with and without the proposed odLength reward. Without the odLength reward, the VLM produces extremely redundant outputs, including both correct-but-duplicated and incorrect-but-duplicated detections. Despite the poor quality of the detection results, the native mAP still assigns a relatively high reward, revealing its susceptibility to reward hacking. However, with our proposed odLength reward, the VLM is incentivized to precisely locate every object, demonstrating an emergent reasoning behavior, which we refer to as the “OD aha moment”. Faced with complex detection tasks involving multiple potential targets (including hard negatives), the model spontaneously adopts a two-step strategy: it first identifies which objects are truly present through an explicit “thinking” step, and then proceeds with accurate bounding box prediction.

###### Without odLength Reward With odLength Reward

|[Figure 22]|
|---|

[Figure 23]

a gymnast who touches the bar

| |
|---|

Given task: Please carefully check the image and detect the following objects:

8 9 12 16

Given task: Please carefully check the image and detect the following objects:

22 23

a surfboard on the beach

| |
|---|

- - white swan in the water
- - person playing tennis with a racket
- - person wearing a hat on a tennis court
- - tennis racket
- - incomplete apple
- - non-human mask
- - volleyball on the beach
- - person doing yoga

- - a person in santa claus clothes without bags
- - a person wearing santa claus clothes
- - a gymnast who touches the bar
- - a person wearing santa claus clothes
- - aircraft flying in the air without smoke trail
- - wild lioness
- - a surfboard on the beach

5 7 13 19

3 4 6 15 17

Model response: <think> The task requires object detection. From analyzing the image, I can see this is a tennis-related scene.

a person wearing santa claus clothes

Model response: <think>...</think> <answer> {"bbox_2d": , "label": "..."}, {"bbox_2d": , "label": "..."}, {"bbox_2d": , "label": "..."}, </answer>

Only tennis-related objects are present - specifically a player with a racket wearing a hat. Other objects like swans, apples, masks, volleyballs and yoga poses are not in this image.

{"bbox_2d": , "label": "..."}, {"bbox_2d": , "label": "..."}, {"bbox_2d": , "label": "..."},

{"bbox_2d": , "label": "..."}, {"bbox_2d": , "label": "..."}, {"bbox_2d": , "label": "..."},

- 1

- 2

- 3

- 4

- 5

- 6

- 22

- 23

- 24

</think> <answer> ```json [

OD aha moment

extremely redundant outputs

{"bbox_2d": [160, 17, 534, 507], "label": "person playing tennis with a racket"}, {"bbox_2d": [160, 17, 534, 507], "label": "person wearing a hat on a tennis court"}, {"bbox_2d": [146, 74, 228, 133], "label": "tennis racket"}

Correct detection but reduplicative

| |
|---|

[Figure 24]

Calculate with native mAP reward: 0.99 Calculate with odLength reward: 0.12

[Figure 25]

] ``` </answer>

Wrong detection and reduplicative

Figure 7. Comparison of cases with and without the proposed odLength reward. Left: Without odLength, the model generates redundant and duplicated boxes, yet still receives a high reward from native mAP. Each circle denotes a predicted bounding box, and circles of the same color indicate bounding boxes with identical coordinates. Right: With odLength, the model exhibits an “OD aha moment”, first reasoning about object presence before producing accurate bounding boxes.

COCOfiltered OVDEval

###### Model mAP GP GR Cel Logo Land Color Mat Pos Rel Neg Overall

Qwen2.5VL-3B-Instruct 14.2 56.06 33.79 13.2 26.5 21.3 2.9 11.6 47.9 13.1 38.7 25.46 w/ COCO training 12.6 48.43 31.86 9.9 18.3 26.6 2.5 6.2 45.3 18.1 36.4 24.48 w/ D3 training 21.1 67.34 43.84 55.0 34.6 47.9 4.5 9.7 42.9 21.5 37.7 31.01

- Table 6. Performance comparison of models trained on different training data. GP: Greedy Precision; GR: Greedy Recall; Cel: Celebrity; Land: Landmark; Mat: Material; Pos: Position; Rel: Relationship; Neg: Negation .

COCOfiltered OVDEval Model mAP GP GR Cel Logo Land Color Mat Pos Rel Neg Overall

|Qwen2.5VL-3B-Instruct 14.2 56.06 33.79 w/ RL 21.1 67.34 43.84|13.2 26.5 21.3 2.9 11.6 47.9 13.1 38.7 25.46 55.0 34.6 47.9 4.5 9.7 42.9 21.5 37.7 31.01<br><br>|
|---|---|
|Qwen2.5VL-7B-Instruct 14.4 54.73 33.36 w/ RL 21.9 74.46 41.2|48.4 35.8 44.6 3.0 10.5 40.5 16.2 39.0 29.08 57.1 38.3 49.4 7.8 14.7 39.4 20.1 43.1 32.42<br><br>|
|Qwen2.5VL-32B-Instruct 18.6 57.26 47.58 w/ RL 23.0 74.04 48.67<br><br>|57.7 32.5 46.7 4.4 13.6 41.7 20.6 47.0 32.79<br>57.8 35.8 48.3 7.8 19.1 44.5 27.0 51.7 36.79<br>|

- Table 7. Performance comparison across models with different sizes and their corresponding RL models. GP: Greedy Precision; GR: Greedy Recall; Cel: Celebrity; Land: Landmark; Mat: Material; Pos: Position; Rel: Relationship; Neg: Negation.

##### 5.3.2 Selection of training data

- Table 6 presents a performance comparison between models trained on COCO and D3 datasets. Notably, the model trained on D3 significantly outperforms the one trained on

COCO—even on the in-domain COCOfiltered evaluation set, which aligns with the distribution of the COCO training data. One key difference lies in the semantic complexity of the training queries: COCO categories are typically

simple, often consisting of single-word labels (e.g., person, car), whereas D3 queries are semantically richer, typically formulated as full, meaning-intensive sentences (see Figure 7 for examples). We hypothesize that this difference in semantic richness plays a pivotal role in the observed performance gap. In the context of reinforcement learning, challenging and semantically complex data is essential for encouraging the model to develop more robust reasoning chains, ultimately leading to superior task performance.

##### 5.3.3 RL effects across model scales

- Table 7 presents the performance comparison between models of different sizes and their corresponding RL-enhanced versions. Several noteworthy observations emerge:

- • The Relation sub-task, which requires reasoning ability, shows a substantial performance boost after applying RL across all model sizes (13.1→21.5, 16.2→20.1, 20.6→ 27.0), indicating that RL could exploit the superior reasoning capabilities of VLMs.
- • Another reasoning-intensive sub-task, Negation, both 7B and 32B RL model achieve improvement (39.0→43.1, 47.0→51.7), whereas 3B model suffers from slight performance degradation (38.7→ 37.7). We posit that this discrepancy arises from the inherent capacity of the base models. As demonstrated by [34], reinforcement learning predominantly serves to reinforce correct reasoning patterns rather than infuse novel knowledge. Given the greater capacity of the 7B and 32B base models, it is plausible that reinforcement learning more effectively harnesses their latent reasoning abilities.
- • In the context of the Color sub-task, 7B and 32B RL model exhibit more performance gains than 3B model (2.9 →4.5 vs. 3.0→7.8, 4.4→7.8). Given that the Color sub-task in OVDEval primarily involves small objects, this comparison highlights the superior visual perception capabilities about fine-grained visual details of the large VLMs.
- • On the COCOfiltered subset, models across all sizes demonstrate greater gains in Greedy Precision relative to Greedy Recall. This discrepancy aligns with the design of the odLength reward, which explicitly penalizes redundant bounding box predictions. While this amendment improves precision by discouraging over-prediction, it can lead to a slight reduction in recall due to the model’s increased conservativeness when outputting predictions.
- • Larger models generally perform slightly better.

### 6. Discussion

#### 6.1. Reinforcement Learning vs. Supervised FineTuning

In the context of referring expression comprehension, in addition to achieving steady performance gains on in-domain tasks, the RL model generalizes the reasoning patterns acquired from the non-reasoning training data to the out-ofdomain settings that require a more nuanced understanding and complex reasoning. This suggests that RL not only optimizes for performance on seen scenarios, but also encourages models to develop transferable capabilities applicable to more challenging, unseen tasks.

Furthermore, in the open-vocabulary object detection experiments, RL models outperform their supervised SFT counterparts in most subtasks on the complex OVDEval

benchmark, particularly achieving substantial gains in some challenging subtasks. Moreover, as discussed in5.3.3, models of nearly all sizes benefit from RL in these reasoningfocused tasks, further validating the generalization advantage of this training paradigm.

These findings strongly support the conclusion proposed by [12]: “SFT Memorizes, RL Generalizes”. Our results reinforce the effectiveness of RL in enhancing the generalization capabilities of VLMs, especially in scenarios that require reasoning patterns.

#### 6.2. Preventing Reward Hacking via Reward Engineering

In this report, we reveal the phenomenon of reward hacking on OVD tasks when using the native mAP reward and demonstrate the effectiveness of our proposed odLength reward in mitigating this issue. As illustrated in Figure 7, the poorly designed reward function incentivizes the model to generate excessive and indiscriminate predictions in pursuit of higher reward values. This behavior results in degraded performance on evaluation benchmarks. By contrast, incorporating the odLength reward significantly suppresses such redundant outputs, leading to improved alignment between reward signals and evaluation metrics, and more importantly, emerging “OD aha moment”, first reasoning about object presence before producing accurate bounding boxes.

These results highlight the importance of careful reward design in reinforcement learning pipelines, particularly for complex tasks where naively defined objectives may fail to capture desired model behavior.

#### 6.3. Role of Data in Reasoning and Generalization

Our findings highlight the pivotal role of training data in shaping model performance. We observe that complex and challenging training samples can effectively elicit reasoning behaviors in VLMs, consistent with the observations in [38]. Conversely, low-quality or over-simple data may hinder learning and even negatively impact generalization (Table 6). These insights emphasize the necessity of careful training data selection.

Equally important is the choice of evaluation data. Comprehensive and appropriately challenging benchmarks are essential for accurately assessing a model’s reasoning and perception capabilities. In this study, we select LISAGrounding and OVDEval as our evaluation datasets, as they are both designed to probe complex semantic understanding and generalization in complicated, real-world scenarios. Together, our results reinforce the importance of highquality training and evaluation data to advance the capabilities of VLMs.

#### 6.4. From Simple to Complex: Adapting RL for OVD

In this report, we explore the feasibility of applying the R1style reinforcement learning framework to two structurally similar tasks: referring expression comprehension (REC) and open-vocabulary object detection (OVD), both of which require the model to output bounding boxes based on textual descriptions. Despite their surface similarity, our comparative analysis reveals that additional optimization is essential to successfully apply RL to the more complex OVD task.

First, while a naive reward function suffices for REC, it fails to yield effective training in OVD due to reward hacking, necessitating the design of a more robust, customized reward—such as our proposed odLength. Second, although models trained on relatively simple in-domain datasets (i.e., RefCOCO) generalize well in the REC setting, the same approach does not transfer effectively to OVD. To address this, we carefully choose a more appropriate training dataset for OVD (i.e., D3), thus achieving a superior result.

These findings underscore the need for task-specific optimizations when applying RL to more complex scenarios.

### 7. Conclusion

In this work, we introduce VLM-R1, the unified framework that brings R1-style reinforcement learning into the domain of visual understanding. Our framework is tailored to vision-language models and supports flexible data definition, model modularity, and training scalability. Using VLM-R1, we successfully apply RL to two representative visual understanding tasks—referring expression comprehension and open-vocabulary object detectiondemonstrating substantial gains in both task performance and out-of-domain generalization. Beyond empirical results, we provide practical insights into reward engineering, data selection, and model scaling that are critical for applying RL effectively to complex vision-language tasks. Our work lays the foundation for broader adaption of reinforcement learning in vision-language research. In future work, we aim to explore cross-task generalization and extend VLM-R1 to more challenging multimodal scenarios.

### References

- [1] Easyr1: An efficient, scalable, multi-modality rl training framework. https://github.com/hiyouga/ EasyR1, 2025. 4
- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 3

- [3] Arash Ahmadian, Chris Cremer, Matthias Gall´e, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Ust¨¨ un, and Sara Hooker. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. arXiv preprint arXiv:2402.14740, 2024. 3
- [4] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736,

2022. 3

- [5] Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Man´e. Concrete problems in ai safety. arXiv preprint arXiv:1606.06565, 2016. 9
- [6] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 2, 3
- [7] Kezhen Chen, Rahul Thapa, Rahul Chalamala, Ben Athiwaratkun, Shuaiwen Leon Song, and James Zou. Dragonfly: Multi-resolution zoom supercharges large visual-language model. arXiv preprint arXiv:2406.00977, 2024. 3
- [8] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023. 3
- [9] Liang Chen, Lei Li, Haozhe Zhao, Yifan Song, and Vinci. R1-v: Reinforcing super generalization ability in visionlanguage models with less than $3. https://github. com/Deep-Agent/R1-V, 2025. Accessed: 2025-02-02. 3, 4
- [10] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and testtime scaling. arXiv preprint arXiv:2412.05271, 2024. 2, 3
- [11] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024. 2, 3
- [12] Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V Le, Sergey Levine, and Yi Ma. Sft memorizes, rl generalizes: A comparative study of foundation model post-training. arXiv preprint arXiv:2501.17161, 2025. 11
- [13] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale

- Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning, 2023. 3
- [14] Huilin Deng, Ding Zou, Rui Ma, Hongchen Luo, Yang Cao, and Yu Kang. Boosting the generalization and reasoning of vision language models with curriculum reinforcement learning. arXiv preprint arXiv:2503.07065, 2025. 3
- [15] Carson Denison, Monte MacDiarmid, Fazl Barez, David Duvenaud, Shauna Kravec, Samuel Marks, Nicholas Schiefer, Ryan Soklaski, Alex Tamkin, Jared Kaplan, et al. Sycophancy to subterfuge: Investigating reward-tampering in large language models. arXiv preprint arXiv:2406.10162,

2024. 9

- [16] Hugging Face. Open r1: A fully open reproduction of deepseek-r1, 2025. 3
- [17] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 2, 3
- [18] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022. 2
- [19] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025. 3
- [20] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024. 2
- [21] Qing Jiang, Gen luo, Yuqin Yang, Yuda Xiong, Yihao Chen, Zhaoyang Zeng, Tianhe Ren, and Lei Zhang. Chatrex: Taming multimodal llm for joint perception and understanding,

2024. 8

- [22] Jing Yu Koh, Ruslan Salakhutdinov, and Daniel Fried. Grounding language models to images for multimodal inputs and outputs. In International Conference on Machine Learning, pages 17283–17300. PMLR, 2023. 3
- [23] Wouter Kool, Herke van Hoof, and Max Welling. Buy 4 reinforce samples, get a baseline for free! 2019. 3
- [24] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9579–9589, 2024. 2, 6
- [25] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 3
- [26] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–

19742. PMLR, 2023. 3

- [27] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence

- Zitnick. Microsoft coco: Common objects in context. In Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13, pages 740–755. Springer, 2014. 2, 8
- [28] Chris Yuhao Liu, Liang Zeng, Jiacai Liu, Rui Yan, Jujie He, Chaojie Wang, Shuicheng Yan, Yang Liu, and Yahui Zhou. Skywork-reward: Bag of tricks for reward modeling in llms. arXiv preprint arXiv:2410.18451, 2024. 2
- [29] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024. 3
- [30] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 3
- [31] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024. 3
- [32] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European Conference on Computer Vision, pages 38–55. Springer, 2024. 2, 8
- [33] Yiqi Liu, Nafise Sadat Moosavi, and Chenghua Lin. Llms as narcissistic evaluators: When ego inflates evaluation scores. arXiv preprint arXiv:2311.09766, 2023. 9
- [34] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025. 11
- [35] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visualrft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025. 3
- [36] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023. 3
- [37] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 11–20, 2016. 2, 6
- [38] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, et al. Mm-eureka: Exploring visual aha moment with rule-based large-scale reinforcement learning. arXiv preprint arXiv:2503.07365, 2025. 3, 11
- [39] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730– 27744, 2022. 2

- [40] Alexander Pan, Erik Jones, Meena Jagadeesan, and Jacob Steinhardt. Feedback loops with language models drive incontext reward hacking. arXiv preprint arXiv:2402.06627,

2024. 9

- [41] Jane Pan, He He, Samuel R Bowman, and Shi Feng. Spontaneous reward hacking in iterative self-refinement. arXiv preprint arXiv:2407.04549, 2024. 9
- [42] Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai Yang, Xingzhong Xu, Xin Geng, and Xu Yang. Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl. arXiv preprint arXiv:2503.07536, 2025. 3
- [43] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 3
- [44] Tianhe Ren, Qing Jiang, Shilong Liu, Zhaoyang Zeng, Wenlong Liu, Han Gao, Hongjie Huang, Zhengyu Ma, Xiaoke Jiang, Yihao Chen, et al. Grounding dino 1.5: Advance the” edge” of open-set object detection. arXiv preprint arXiv:2405.10300, 2024. 2
- [45] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 5
- [46] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 2, 3, 4
- [47] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. arXiv preprint arXiv:2303.15389, 2023. 3
- [48] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 37:95095–95169, 2024. 3
- [49] Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Qi Liu, Tianyu Liu, and Zhifang Sui. Large language models are not fair evaluators. arXiv preprint arXiv:2305.17926, 2023. 9
- [50] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 2, 3
- [51] Jiaxin Wen, Ruiqi Zhong, Akbir Khan, Ethan Perez, Jacob Steinhardt, Minlie Huang, Samuel R Bowman, He He, and Shi Feng. Language models learn to mislead humans via rlhf. arXiv preprint arXiv:2409.12822, 2024. 9
- [52] Chi Xie, Zhao Zhang, Yixuan Wu, Feng Zhu, Rui Zhao, and Shuang Liang. Described object detection: Liberating object detection with flexible expressions. Advances in Neural Information Processing Systems, 36:79095–79107, 2023. 7
- [53] Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun

- Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615, 2025. 3
- [54] Yiyang Yao, Peng Liu, Tiancheng Zhao, Qianqian Zhang, Jiajia Liao, Chunxin Fang, Kyusong Lee, and Qing Wang. How to evaluate the generalization of detection? a benchmark for comprehensive open-vocabulary detection. arXiv preprint arXiv:2308.13177, 2023. 2, 8
- [55] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14, pages 69–85. Springer, 2016. 2, 6
- [56] Yuhang Zang, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Ziyu Liu, Shengyuan Ding, Shenxi Wu, Yubo Ma, Haodong Duan, Wenwei Zhang, et al. Internlm-xcomposer2. 5-reward: A simple yet effective multi-modal reward model. arXiv preprint arXiv:2501.12368, 2025. 2
- [57] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training,

2023. 3

- [58] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pages 169–186. Springer, 2024. 3
- [59] Tiancheng Zhao, Peng Liu, Xiaopeng Lu, and Kyusong Lee. Omdet: Language-aware object detection with large-scale vision-language multi-dataset pre-training. CoRR, 2022. 2
- [60] Tiancheng Zhao, Peng Liu, and Kyusong Lee. Omdet: Large-scale vision-language multi-dataset pre-training with multimodal detection network. IET Computer Vision, 18(5): 626–639, 2024. 2, 8
- [61] Hengguang Zhou, Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, and Cho-Jui Hsieh. R1-zero’s” aha moment” in visual reasoning on a 2b non-sft model. arXiv preprint arXiv:2503.05132, 2025. 3

