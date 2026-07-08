## arXiv:2503.16081v2[cs.LG]28Mar2025

# OThink-MR1: Stimulating multimodal generalized reasoning capabilities via dynamic reinforcement learning

Zhiyuan Liu

liuzhiyuan1@oppo.com OPPO Research Institute Shenzhen, China

Yuting Zhang

yzhang755@connect.hkustgz.edu.cn The Hong Kong University of Science and Technology (Guangzhou) Guangzhou, China

Feng Liu

liufeng4hit@gmail.com OPPO Research Institute Shenzhen, China

Changwang Zhang∗

zhangchangwang@oppo.com OPPO Research Institute Shenzhen, China

Ying Sun∗

yings@hkust-gz.edu.cn The Hong Kong University of Science and Technology (Guangzhou) Guangzhou, China

Jun Wang∗

junwang.lu@gmail.com OPPO Research Institute Shenzhen, China

### Abstract

Multimodal Large Language Models (MLLMs) have gained significant traction for their ability to process diverse input data types and generate coherent, contextually relevant outputs across various applications. While supervised fine-tuning (SFT) has been the predominant approach to enhance MLLM capabilities in task-specific optimization, it often falls short in fostering crucial generalized reasoning abilities. Although reinforcement learning (RL) holds great promise in overcoming these limitations, it encounters two significant challenges: (1) its generalized capacities in multimodal tasks remain largely unexplored, and (2) its training constraints, including the constant Kullback-Leibler divergence or the clamp strategy, often result in suboptimal bottlenecks. To address these challenges, we propose OThink-MR1, an advanced MLLM equipped with profound comprehension and reasoning capabilities across multimodal tasks. Specifically, we introduce Group Relative Policy Optimization with a dynamic Kullback-Leibler strategy (GRPOD), which markedly enhances reinforcement learning (RL) performance. For Qwen2-VL-2B-Instruct, GRPO-D achieves

∗corresponding authors.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. Conference acronym ’XX, Woodstock, NY

© 2018 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-1-4503-XXXX-X/2018/06 https://doi.org/XXXXXXX.XXXXXXX

a relative improvement of more than 5.72% over SFT and more than 13.59% over GRPO in same-task evaluation on two adapted datasets. Furthermore, GRPO-D demonstrates remarkable cross-task generalization capabilities, with an average relative improvement of more than 61.63% over SFT in cross-task evaluation. These results highlight that the MLLM trained with GRPO-D on one multimodal task can be effectively transferred to another task, underscoring the superior generalized reasoning capabilities of our proposed OThink-MR1 model.

Keywords: Cross-task Generalization; Dynamic Reinforce Learning; Multimodal Large Language Model

ACM Reference Format:

Zhiyuan Liu, Yuting Zhang, Feng Liu, Changwang Zhang, Ying Sun, and Jun Wang. 2018. OThink-MR1: Stimulating multimodal generalized reasoning capabilities via dynamic reinforcement learning. In Proceedings of Make sure to enter the correct conference title from your rights confirmation email (Conference acronym ’XX). ACM, New York, NY, USA, 8 pages. https://doi.org/XXXXXXX.XXXXXXX

### 1 Introduction

Multimodal Large Language Models [10] integrate and process multiple types of input data, such as text and images, to generate coherent and contextually relevant outputs. These models aim to enhance multimodal understandingby leveraging the strengths of different modalities and have been widely used in applications such as Visual Question Answering [7, 8] and Image Captioning [1, 9].

Compared to unimodal large language models (LLMs), MLLMs must process more complex data structures, making their training process significantly more resource-intensive. Furthermore, the multimodal nature of MLLMs positions them as ideal candidates for deployment in complex and dynamic real-world scenarios. These factors, in turn, place

[Figure 1]

Visual Counting Geometry Reasoning

image

image

[Figure 2]

[Figure 3]

problem

In the given triangle ABC, if line BD is perpendicular to AC and also bisects it, and angle A measures 20 degrees, what is the measure of angle CBD?

problem

How many items are there in the image?

solution

solution

<answer> 70° </answer>

<answer> 3 </answer>

(a) Multi-modal Content Understanding (b) Multi-modal Reasoning

Figure 1. Test accuracy metric curves of SFT and GRPO on geometry reasoning task.

Figure 2. Examples of multimodal content understanding and multimodal reasoning tasks.

higher demands on the models’ cross-task generalization capabilities–the ability to handle different tasks with different data distributions.

Currently, most MLLMs rely on SFT [11, 17] as the primary post-training approach. However, research [4] has shown SFT tends to memorize patterns in the training data, performing poorly in out-of-distribution (OOD) scenarios [4], not to mention cross-task generalization in multimodal settings. This limitation highlights the need for more robust training paradigms that can enhance the generalization capabilities of MLLMs.

Recent advancements in LLMs, such as DeepSeek R1 [6], have demonstrated the effectiveness of reinforcement learning with verifiable rewards (RLVR) in improving reasoning abilities. These methods [6, 15] leverage reinforcement finetuning to significantly enhance the reasoning performance of models. However, existing research has primarily focused on unimodal tasks, leaving the potential of RLVR for multimodal tasks and its ability to enable cross-task generalization largely unexplored.

We directly extend reinforcement learning with verifiable rewards to enhance MLLMs for multimodal tasks. Specifically, we train and validate on the multimodal geometry reasoning task. The test accuracy metric curves of SFT and GRPO (Group Relative Policy Optimization [6], a method based on RLVR) are shown in Figure 1. As the number of training steps increases, both SFT and GRPO exhibit performance improvements. However, SFT demonstrates a more significant enhancement and eventually surpasses GRPO when the training steps reach thousands. This can be attributed to GRPO’s insufficient exploration, leading to suboptimal solutions.

To address these issues, we propose OThink-MR1, an advanced MLLM equipped with generalized understanding and reasoning across multimodal tasks. First, to address GRPO’s insufficient exploration capacity, we propose a Dynamic KL strategy(GRPO-D) inspired by the 𝜖-greedy strategy in classical reinforcement learning. This strategy follows the principle of "early exploration, late exploitation" by dynamically

adjusting the KL divergence to balance exploration and exploitation. The validation results demonstrate the dynamic KL strategy effectively enables GRPO-D to surpass SFT in same-task evaluation.

Additionally, to assess the cross-task generalization ability, we propose cross-task validation, where models are posttrained on one task type and evaluated on another. We adopt the visual counting task (a multimodal content understanding task that extracts basic multimodal information) and the geometry reasoning task (a multimodal reasoning task that requires logical analysis of visual relationships). The examples of these two tasks are provided in Figure 2. The cross-task validation results demonstrate the strong generalization capabilities of GRPO-D, highlighting its ability to effectively transfer knowledge across different tasks.

To sum up, our contributions are listed below:

- • We propose OThink-MR1, a dynamic reinforcement learning framework for fine-tuning MLLMs, which outperforms SFT in the same-task validation. This approach dynamically balances exploration and exploitation, resulting in more effective learning.
- • We are among the first to demonstrate significant crosstask generalization of dynamic reinforcement learning for MLLMs. This demonstrates that models posttrained with GRPO-D on one multimodal task can be effectively transferred to other multimodal tasks, greatly reducing the need for extensive task-specific data collection and retraining across diverse applications.
- • We conduct extensive experiments on various multimodal tasks, including visual counting, geometry reasoning, cross-task validation and more, to verify our model’s generalized reasoning ability.

### 2 Related Work

Multimodal Large Language Models such as LLaVA [12] integrate and process multimodal inputs (e.g., text and images) to understand and generate coherent content. Training of

(b) Same task validation

[Figure 4]

[Figure 5]

As shown in the figure, BD bisects angle ABC, CD parallel AB, if angle BCD = 70.0, then the degree of angle CDB is ()

[Figure 6]

[Figure 7]

<answer> 110 </answer>

[Figure 8]

[Figure 9]

As shown in the figure, BD bisects angle ABC, CD parallel AB, if angle BCD = 70.0, then the degree of angle CDB is ()

<think> To find the degree of angle CDB, we can use the property of angles subtended by the same chord in a circle. Since CD is parallel to AB, angle CDB is equal to angle BCD. Given that angle BCD = 70.0 degrees, angle CDB = 70.0 degrees. </think> <answer>70.0</answer>

SFT

[Figure 10]

[Figure 11]

[Figure 12]

<answer> 110 </answer>

Dynamic

KL SFT

[Figure 13]

GRPO-D

<think> Since CD is parallel to AB, angle BCD corresponds to angle ABC (as they are corresponding angles), so angle ABC = 70°. BD bisects angle ABC into two angles of 35° each. Now, in triangle BCD, angles at B is 35°, at C is 70°, so angle at D is 180 - 35 - 70 = 75. </think> <answer>75</answer>

[Figure 14]

[Figure 15]

[Figure 16]

How many different items are there in the image?

GRPO

[Figure 17]

[Figure 18]

<think>There are 7 different objects in the image. The objects include a green bus, a blue motorcycle, a gray chopper, a purple car, a purple school bus, a red school bus, and a small metallic motorcycle. </think> <answer>7</answer>

[Figure 19]

<answer> 6 </answer>

(a) Constant KL+ Same-task validation

[Figure 20]

SFT GRPO-D

[Figure 21]

(c) Cross-task validation

Figure 3. Qualitative illustration of GRPO-D vs SFT in OThink-MR1.

MLLMs typically involves pre-training and post-training stages, where post-training is crucial for enhancing model performance. Most works [3, 14, 19, 20] rely on SFT, while a few [2, 13] explore RL. However, SFT often struggles with memorization rather than generalization. Reinforcement learning(RL),particularlywithverifiable rewards (e.g., GRPObased RLVR in Visual-RFT [13]), excels in few-shot learning scenarios. However, as data scales up, RLVR may be outperformed by SFT due to imbalanced exploration and exploitation, leading to suboptimal solutions. Additionally, prior works, such as R1-V [2], have evaluated RLVR’s outof-distribution (OOD) performance but are limited to sametask validation, where models are tested only on different data distributions within the same task domain. To address these issues, we propose a dynamic reinforcement learning algorithm, named GRPO-D. This algorithm adapts an "exploration early, exploitation late" strategy during training, enhancing RLVR’s performance and scalability in large-data scenarios where traditional RLVR falls short. Furthermore, we introduce a novel cross-task validation method. This method underscores GRPO-D’s ability to effectively transfer to new multimodal tasks when post-trained on data from another task type, showcasing its ability to foster MLLMs’ task-generalized reasoning capability.

### 3 Methodology

In OThink-MR1, we adopt Group Relative Policy Optimization with a Dynamic KL divergence strategy (GRPO-D) to enhance MLLMs for multimodal tasks. The input data for multimodal tasks include images and texts. For each multimodal input, GRPO-D samples a group of outputs {𝑜1,𝑜2, · · · ,𝑜𝐺}

from the old policy 𝜋old and receives corresponding rewards {𝑟1,𝑟2, · · · ,𝑟𝐺},which arescoredbyaverifiable reward model. In the following, we present the verifiable reward models for visual counting and geometry reasoning tasks.

- 3.1 Verifiable Reward for Multimodal Tasks In Visual counting and Geometry Reasoning tasks, we

employ a verifiable accuracy reward 𝑅𝑎𝑐𝑐 to assess the numerical or symbolic correctness of the model’s output. Specifically, in visual counting, the reward measures the model’s output against the ground truth count, whereas in geometry reasoning, it assesses the output against true geometric properties such as angles or lengths.

Also, following DeepSeek-R1-Zero [6], we employ a format reward 𝑅𝑓 𝑜𝑟𝑚𝑎𝑡 for model optimization. This format reward assesses whether the model’s output follows the required format, i.e., placing the reasoning process and final answer in the correct tags.

𝑅 = 𝑅𝑎𝑐𝑐 + 𝛼𝑅𝑓 𝑜𝑟𝑚𝑎𝑡. (1)

In prior studies [2, 13], the hyper- parameter 𝛼 is typically set to 1. For each response𝑜𝑖, its reward value 𝑟𝑖 is calculated using Eq. 1. The relative quality of responses is then assessed by normalizing these rewards using their mean and standard deviation, as follows:

𝐴ˆ𝑖,𝑡 =

𝑟𝑖 − mean(r) std(r)

. (2)

- 3.2 Dynamic KL divergence-based Optimization After obtaining the relative quality of the output, we need to optimize the policy model 𝜋𝜃. Generally GRPO [6] employs a

KL divergence term to constrain the difference between the policy model and the reference model, such as with a default constant weight of 0.04. However, in same-task validation with thousands of samples, GRPO is unable to outperform SFT. We analyze that this may be due to an imbalance between exploration (trying a new policy) and exploitation (maintaining the current effective policy). A large KL weight restricts exploration early in training, leading to suboptimal performance, while a small weight causes instability due to drastic policy changes later in training.

Inspired by the 𝜖-greedy strategy commonly used in Qlearning [18], a classical reinforcement learning method, we propose a Dynamic KL divergence strategy. Specifically, the 𝜖-greedy strategy adopts the principle of early exploration and late exploitation to ensure robust learning and efficient long-term rewards. Therefore, we introduce a dynamic weight 𝛽(𝑠) for the KL divergence term, which starts from a small value and gradually increases over the training step 𝑠. It can be formulated as follows:

𝛽mid − (𝛽mid − 𝛽𝑚𝑖𝑛) × 𝑤𝑠 , if 𝑠 ≤ 𝑤, 𝛽min + (𝛽max − 𝛽𝑚𝑖𝑛) × 𝑡𝑠−-w𝑤, if 𝑤 < 𝑠 ≤ 𝑡,

(3)

𝛽(𝑠) =

where 𝑤 is the predefined number of exploration steps and 𝑡 is the total number of training steps. 𝛽min and 𝛽max are the predefined minimum and maximum values of the weight, with 𝛽mid = (𝛽min + 𝛽max)/2. Then, we optimize the policy model by maximizing the following objective:

∑︁𝐺

∑︁|𝑜𝑖|

1 |𝑜𝑖|

1 𝐺

LGRPO−𝐷(𝜃) = −

𝑖=1

𝑡=1

𝜋𝜃 (𝑜𝑖,𝑡 | 𝑞,𝑜𝑖,<𝑡) 𝜋𝜃old (𝑜𝑖,𝑡 | 𝑞,𝑜𝑖,<𝑡)

𝜋𝜃 (𝑜𝑖,𝑡 | 𝑞,𝑜𝑖,<𝑡) 𝜋𝜃old (𝑜𝑖,𝑡 | 𝑞,𝑜𝑖,<𝑡)

, 1 − 𝜖, 1 + 𝜖))𝐴ˆ𝑖,𝑡 −𝛽(𝑠)DKL[𝜋𝜃 | 𝜋ref]],

, clip(

[min(

(4)

where 𝜖 is a hyper-parameter, 𝜋ref is the reference model before optimization. We follow DeepSeek-R1-Zero [6] to estimate the KL divergence using an unbiased estimator as follows:

𝜋ref (𝑜𝑖,𝑡 | 𝑞,𝑜𝑖,<𝑡) 𝜋𝜃 (𝑜𝑖,𝑡 | 𝑞,𝑜𝑖,<𝑡)

𝜋ref (𝑜𝑖,𝑡 | 𝑞,𝑜𝑖,<𝑡) 𝜋𝜃 (𝑜𝑖,𝑡 | 𝑞,𝑜𝑖,<𝑡)

− log

− 1.

DKL[𝜋𝜃 ∥𝜋ref] =

(5)

Overall, in GRPO-D, reducing 𝛽(𝑠) in the early stage of training allows for greater exploration of the solution space. Later, increasing 𝛽(𝑠) helps align the model with the reference model. This balance ensures robust learning and efficient long-term rewards.

### 4 Experiments

In this section, we conduct a series of experiments to analyze the impact of various post-training settings on the performance of MLLMs. Specifically, our experiments address the following key research questions:

• RQ1 How do the reward term and KL divergence term impact the original GRPO in same-task validation?

[Figure 22]

Figure 4. Impact of the weight of format reward.

- • RQ2 How does the GRPO-D affect the model’s performance in the same-task validation.
- • RQ3 how does SFT or GRPO-D affect the model’s generalization to different tasks?

For all experiments,weadoptQwen2-VL-2/7B-Instruct [16] as the backbone model. For dataset usage, we follow the settings from R1-V [2]. Specifically, for the geometry reasoning task, we train the model on the 8k GeoQA-Train dataset and validate it on the 753 GeoQA-Test dataset1 [5]. For the visual counting task, we train the model on the R1 Distilled 37K ClevR dataset2 and validate it on the SuperClevR dataset 3. Regarding the crucial hyperparameter settings of dynamic KL divergence in Equation 3, we set 𝑤 and 𝑡 proportionally, that is 𝑤/𝑡 = 0.3. For the visual counting task, we set 𝛽𝑚𝑖𝑛 = 0.04 and 𝛽𝑚𝑎𝑥 = 0.1. The larger range of 𝛽 values reflects the need for more precise and conservative exploration in simple visual reasoning tasks, where stronger constraints are required to guide the model towards reliable solutions. For the geometric reasoning task, we set 𝛽𝑚𝑖𝑛 = 0.0 and 𝛽𝑚𝑎𝑥 = 0.02. The smaller values were selected to allow for a broader range of exploration, which is beneficial for handling the greater complexity and variability in geometric reasoning tasks, where weaker constraints are needed to encourage more diverse solutions. The other parameters were set following R1-V [2].

#### 4.1 Analysis of GPRO in Multimodal tasks (RQ1)

Eq. 1 and Eq. 4 introduce two critical hyper-parameters, 𝛼 and 𝛽(𝑠) , which control the influence of the format reward and the KL divergence term, respectively, on the model’s optimization process. The following analysis examines their individual impacts for the geometry reasoning task.

4.1.1 Impact of the weight of format reward. We vary the weight of format reward 𝛼 in [0.00, 0.25, 0.50, 0.75, 1.00]. As shown in Figure 4, Qwen2-VL-2B-Instruct+GRPO with

- 1https://huggingface.co/datasets/leonardPKU/GEOQA_R1V_Train_8K
- 2https://huggingface.co/datasets/leonardPKU/clevr_cogen_a_train
- 3https://huggingface.co/leonardPKU/superclevr/tree/main

format reward (i.e., 𝛼 ∈ [0.25, 0.50, 0.75, 1.00]) consistently outperforms the version without format reward (i.e., 𝛼 = 0.0) on almost all metrics, particularly in terms of accuracy. For instance, when 𝛼 = 1, the model achieves a relative improvement of 20.02%. This improvement is likely due to the combined effect of format reward and accuracy reward, which together guide the model’s training more effectively. Even though the format accuracy of the model without format reward was already high (99.70%), explicitly rewarding the correct format can create a more robust and aligned training signal. This synergy ensures that the model not only generates outputs in the correct format but also does so in a way that enhances its ability to produce accurate answers.

Additionally, we present the training curves for accuracy reward and format reward with 𝛼 = 1 in Figure 5. The format reward quickly increases from 0.0 to 0.2 within the first 50 steps, while the accuracy reward shows a slow and steady upward trend in the visual reasoning task. This pattern is expected, as mastering the reasoning process and generating accurate answers is significantly more challenging than simply aligning the format.

[Figure 23]

- (a) format reward w.r.t training steps.

[Figure 24]

- (b) accuracy reward w.r.t training steps.

Figure 5. Training curves for format reward and accuracy reward.

4.1.2 Impact of the weight of KL divergence. We vary the weight of KL divergence 𝛽(𝑠) across a range of values [0.00, 0.01, 0.02, 0.03, 0.04, 0.05] for geometry reasoning task. In each experiment, 𝛽(𝑠) is held constant throughout training. From the results presented in Figure 6, we can observe

[Figure 25]

Figure 6. Impact of the weight of KL divergence

that: as the weight of KL divergence increases, model performance first improves and then drops. This occurs because moderate regularization helps in enhancing exploration, but excessive regularization overly constrains the model, leading to underfitting and performance degradation.

#### 4.2 Same-Task Evaluation (RQ2)

However, although adjusting the constant weight of different KL divergence terms can improve the effectiveness of MLLMs in same-task validation, as discussed in subsection 4.1.2, we find that GRPO with a constant KL weight (best-performing value 30.24%) consistently underperforms compared to SFT in same geometry reasoning task validation (32.49% shown in Table 1). Thereby, we introduce GRPO-D, which incorporates a dynamic KL divergence strategy to better balance exploration and exploitation in OThink-MR1. The overall performance on the two adopted datasets for same-task validation is presented in Table 1. In the experiments for the visual counting task, we observe that the performance of MLLMs with post-training methods all initially increases and then drops as the training steps increase. This phenomenon may be due to the simplicity of the visual counting task, which makes it prone to overfitting. To address this issue, we adopt a few-shot learning (i.e., 100 samples for Qwen2VL-2B-Instruct and 120 samples for Qwen2-VL-7B-Instruct post-training). This strategy helps prevent overfitting by limiting the amount of training data, ensuring that the model generalizes well to unseen data.Moreover, to ensure fairness, we use the optimal hyperparameters for GRPO: 𝛼 = 1, a constant 𝛽(𝑠) = 0.01 for geometry reasoning task and 𝛼 = 1, a constant 𝛽(𝑠) = 0.04 for visual counting task. From the results, we can observe that:

- • Qwen2-VL-2/7B-Instruct+GRPO underperforms Qwen2VL-2/7B-Instruct+SFT on most settings. This is primarily because, with sufficient training data, SFT can effectively memorize the comprehensive data distribution, leading to better performance in same-task validation.
- • Qwen2-VL-2/7B-Instruct+GRPO-D outperforms Qwen2VL-2/7B-Instruct+GRPO with constant KL divergence. For instance, for Qwen2-VL-2B-Instruct, the dynamic

Table 1. Overall performance results for same-task validation, including the visual counting (VC) task and the geometry reasoning (GR) task. The best-performing method is bolded, and the strongest baseline is underlined.

|Accuracy| |
|---|---|
|VC<br><br>|GR|

Models

Qwen2-VL-2B-Instruct 42.50% 15.52% + GRPO 64.50% 30.24% + SFT 68.50% 32.49% + GRPO-D 76.50% 34.35%

Qwen2-VL-7B-Instruct 76.00% 33.16% + GRPO 76.50% 42.04% + SFT 74.50% 43.50% + GRPO-D 78.00% 45.49%

KL strategy in GRPO-D achieves relative improvements of +18.60% and +13.59% for visual counting and geometry reasoning tasks, respectively. This improvement is attributed to the dynamic KL strategy’s ability to balance exploration and exploitation more effectively.

- • Qwen2-VL-2/7B-Instruct+GRPO-D outperforms Qwen2VL-2/7B-Instruct+SFT. For example, for Qwen2-VL2B-Instruct, compared to SFT post-training, GRPO-D achieves relative improvements of +11.68% and +5.72% on the same tasks.
- • Qwen2-VL-2B-Instruct+GRPO-D outperformed Qwen2VL-7B-Instruct on both tasks. This suggests that posttraining algorithm improvements can be more impactful than simply increasing the parameter scale. Specifically, GRPO-D employs dynamic KL divergence to optimize the exploration-exploitation balance, which appears to mitigate some limitations typically associated with smaller model sizes.

#### 4.3 Cross-Task Evaluation(RQ3)

Previous studies [2, 13] typically evaluate out-of-distribution performance within the same task. In contrast, our work assesses the generalized reasoning ability across different types of tasks. This requires model have greatly most generalization ability, as it must deal with the totally different tasks and data distributions that it has never encountered before. By doing so, we can robustly evaluate whether the model has truly acquired a deep understanding of multimodal knowledge, rather than simply memorizing the post-training data.

- 4.3.1 Cross Task Overall Performance. We validate the multimodal generalization ability on multimodal content understanding and reasoning tasks, i.e., visual counting(VC)

[Figure 26]

Figure 7. Cross-task validation.

and geometry reasoning(GR) tasks respectively. The multimodal reasoning tasks are more challenging than multimodal content understanding as they require complex logical deduction beyond direct multimodal interpretation. To assess generalization, we conduct cross-task experiments in both directions as follows:

- • Reasoning to Understanding Generalization: posttrained the Qwen2-VL-2B-Instruct on the geometry reasoning training data and validate the trained model on visual counting testing data.
- • Understanding to Reasoning Generalization: posttrained the Qwen2-VL-2B-Instruct on the visual counting training data and validate the trained model on geometry reasoning testing data.

The overall cross-task validation results are presented in Figure 7 and we can observe that:

- • In the Reasoning to Understanding Generalization experiment, the model with SFT post-training shows similar performanceto themodel withoutpost-training In the Understanding to Reasoning Generalization experiment, the model with SFT experiences a relative

-77.77% performance drop. These results indicate that SFT tends to memorize training data rather than learn generalized knowledge, leading to poor cross-task generalization, especially in the more challenging Understanding to Reasoning scenario.

- • In contrast, the model with GRPO-D post-training achieves significant improvements in both generalization experiments. For the Reasoning to Understanding Generalization experiment, it gains a +63.53% (vs. without post-training) and +61.63% (vs. SFT) relative improvement. For Understanding to Reasoning Generalization experiment, it gains a +4.25% (vs. without post-training) and +368.99% (vs. SFT) relative improvement. These results demonstrate that GRPO-D effectively enables the model to acquire task-generalized

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

As shown in the figure, in circle O, chord AC parallel radius OB, angle BOC = 50.0, then the degree of angle OAB is ()

How many different items are there in the image?

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

<answer> 6 </answer>

<answer> 50 </answer>

SFT

SFT

<think> There are 7 different objects in the image. The objects include a green bus, a blue motorcycle, a gray chopper, a purple car, a purple school bus, a red school bus, and a small metallic motorcycle. </think> <answer>7</answer>

<think> To find the degree of angle OAB, we can use the properties of parallel lines and the inscribed angle theorem. </think> <answer> The degree of angle OAB is 25.0. </answer>

[Figure 35]

[Figure 36]

GRPO-D

GRPO-D

[Figure 37]

[Figure 38]

(a) Reasoning to Understanding Generalization (b) Understanding to Reasoning Generalization

Figure 8. Case studies for cross-task evaluation.

knowledge in complex multimodal tasks, particularly compared with SFT post-training method.

increase, SFT matches or surpasses GRPO. This phenomenon occurs because SFT, when exposed to sufficient training data, can memorize the comprehensive distribution of the data. GRPO, on the other hand, struggles with balancing exploration and exploitation, leading to suboptimal performance. To address this, we propose a dynamic reinforcement learning algorithm, named GRPO-D with dynamic KL divergence. Extensive experiments demonstrate that GRPO-D consistently outperforms SFT.

These results are promising, as they demonstrate that a model post-trained with GRPO-D on one task can effectively generalize to other multimodal tasks. This is particularly evident when the model is post-trained on complex reasoning task. Overall, these results highlight GRPOD’s ability to foster genuine understanding and reasoning of multimodal knowledge, while SFT is often limited to memorization.

• When trained and validated on different task types (e.g., multimodal understanding vs. reasoning), GRPOD significantly outperforms SFT and the base model without post-training. This demonstrates GRPO-D’s strong generalized ability to enable MLLMs to achieve deeper understanding and reasoning, highlighting its potential for developing generalized MLLMs applicable to diverse multimodal tasks.

- 4.3.2 Case Study. To further evaluate the model’s generalized reasoning ability, we perform case studies on cross-task generalization. We select representative samples from both Reasoning to Understanding and Understanding to Reasoning experiments, as shown in Figure 8. We can observe that after GRPO-D post-training on the geometry reasoning task, the model generalizes effectively to detect and recognize objects in the image, such as identifying the green bus and blue motorcycle. Similarly, in Figure 8(b), GRPO-D post-training on the visual counting task enables the model to analyze relationships within visual data, such as incorporating the inscribed angle theorem—a key factor contributing to the correct result. These results demonstrate GRPO-D’s strong generalized reasoning ability, as evidenced by its adaptable and logical thinking process across diverse tasks.

Overall, OThink-MR1 enables the development of generalizable multimodal models, representing an advancement in the field of multimodal generalized reasoning. Additionally, several interesting findings have emerged that warrant further exploration: the optimal range of KL weight values appears to be correlated with the complexity of the tasks. Moreover, the generalization ability of GRPO-D appears to be influenced by the reasoning demands placed on the training samples. In the future, we plan to explore these relationships more deeply.

### 5 Conclusion and Discussion

In this work, we propose OThink-MR1, an advanced MLLM equipped with profound comprehension and reasoning capabilities across multimodal tasks. The key contributions are detailed below:

### References

- [1] Davide Bucciarelli, Nicholas Moratelli, Marcella Cornia, Lorenzo Baraldi, and Rita Cucchiara. 2024. Personalizing multimodal large language models for image captioning: an Experimental analysis. arXiv preprint arXiv:2412.03665 (2024).
- [2] Liang Chen, Lei Li, Haozhe Zhao, Yifan Song, Vinci, Lingpeng Kong, Qi Liu, and Baobao Chang. 2025. RLVR in Vision Language Models:

• For same-task validation, GRPO initially outperforms SFT in early training stages. However, as training steps

- Findings, Questions and Directions. Notion Post (Feb 2025). https: //deepagent.notion.site/rlvr-in-vlms
- [3] Daixuan Cheng, Shaohan Huang, Ziyu Zhu, Xintong Zhang, Wayne Xin Zhao, Zhongzhi Luan, Bo Dai, and Zhenliang Zhang. 2024. On Domain-Specific Post-Training for Multimodal Large Language Models. arXiv preprint arXiv:2411.19930 (2024).
- [4] Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V Le, Sergey Levine, and Yi Ma. 2025. Sft memorizes, rl generalizes: A comparative study of foundation model post-training. arXiv preprint arXiv:2501.17161 (2025).
- [5] Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, et al. 2023. G-llava: Solving geometric problem with multi-modal large language model. arXiv preprint arXiv:2312.11370 (2023).
- [6] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948 (2025).
- [7] Jiayi Kuang, Ying Shen, Jingyou Xie, Haohao Luo, Zhe Xu, Ronghao Li, Yinghui Li, Xianfeng Cheng, Xika Lin, and Yu Han. 2024. Natural Language Understanding and Inference with MLLM in Visual Question Answering: A Survey. Comput. Surveys (2024).
- [8] Jusung Lee, Sungguk Cha, Younghyun Lee, and Cheoljong Yang. 2024. Visual question answering instruction: Unlocking multimodal large language model to domain-specific visual multitasks. arXiv preprint arXiv:2402.08360 (2024).
- [9] Wei Li, Hehe Fan, Yongkang Wong, Yi Yang, and Mohan Kankanhalli.

2024. Improving context understanding in multimodal large language models via multimodal composition learning. In Forty-first International Conference on Machine Learning.

- [10] Zijing Liang, Yanjie Xu, Yifan Hong, Penghui Shang, Qi Wang, Qiang Fu, and Ke Liu. 2024. A Survey of Multimodel Large Language Models. In Proceedings of the 3rd International Conference on Computer, Artificial Intelligence and Control Engineering. 405–409.
- [11] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. 2024. Vila: On pre-training for visual language models.

- In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 26689–26699.
- [12] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. Advances in neural information processing systems 36 (2023), 34892–34916.
- [13] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. 2025. Visual-RFT: Visual Reinforcement Fine-Tuning. arXiv preprint arXiv:2503.01785 (2025).
- [14] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan.

2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems 35 (2022), 2507–2521.

- [15] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al.

2025. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599 (2025).

- [16] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. 2024. Qwen2vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191 (2024).
- [17] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Song XiXuan, et al. 2024. Cogvlm: Visual expert for pretrained language models. Advances in Neural Information Processing Systems 37 (2024), 121475–121499.
- [18] Christopher JCH Watkins and Peter Dayan. 1992. Q-learning. Machine learning 8 (1992), 279–292.
- [19] Xiangyu Zeng, Kunchang Li, Chenting Wang, Xinhao Li, Tianxiang Jiang, Ziang Yan, Songze Li, Yansong Shi, Zhengrong Yue, Yi Wang, et al. [n. d.]. TimeSuite: Improving MLLMs for Long Video Understanding via Grounded Tuning. In The Thirteenth International Conference on Learning Representations.
- [20] Xiaoman Zhang, Chaoyi Wu, Ziheng Zhao, Weixiong Lin, Ya Zhang, Yanfeng Wang, and Weidi Xie. 2023. Pmc-vqa: Visual instruction tuning for medical visual question answering. arXiv preprint arXiv:2305.10415

(2023).

