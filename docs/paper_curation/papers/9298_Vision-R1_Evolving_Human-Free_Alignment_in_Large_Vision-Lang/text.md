# arXiv:2503.18013v1[cs.CV]23Mar2025

## Vision-R1: Evolving Human-Free Alignment in Large Vision-Language Models via Vision-Guided Reinforcement Learning

Yufei Zhan1,2, Yousong Zhu1,∗, Shurong Zheng1,3, Hongyin Zhao1, Fan Yang1,3, Ming Tang1,2, Jinqiao Wang1,2,3,4 1 Foundation Model Research Center, Institute of Automation, Chinese Academy of Sciences, Beijing, China 2 School of Artificial Intelligence, University of Chinese Academy of Sciences, Beijing, China 3 Peng Cheng Laboratory, Shenzhen, China 4 Wuhan AI Research, Wuhan, China

{zhanyufei2021, zhengshurong2023, zhaohongyin2020, yangfan 2022}@ia.ac.cn {yousong.zhu, tangm, jqwang}@nlpr.ia.ac.cn

Github: https://github.com/jefferyZhan/Griffon/tree/master/Vision-R1

#### Abstract

Large Vision-Language Models (LVLMs) typically follow a two-stage training paradigm—pretraining and supervised fine-tuning. Recently, preference optimization, derived from the language domain, has emerged as an effective post-training reinforcement strategy to enhance capabilities of LVLMs. However, constructing high-quality human-annotated preference data and developing robust reward models to mimic these preferences are both costly and challenging. Motivated by this observation, we propose Vision-R1, a novel vision-guided R1-like reinforcement learning algorithm for LVLMs that rewards models with definitive vision feedback. It only leverages curated instruction data, eliminating the need for specialized reward models and handcrafted preference datasets. We incorporate a criterion-driven reward function that further integrates multi-dimensional feedback to evaluate model completions comprehensively based on the vision task logic. Furthermore, we introduce a progressive rule refinement strategy that dynamically adjusts the reward criteria during training, enabling continuous model improvement and mitigating reward hacking. Extensive experiments on both in-distribution and out-of-distribution benchmarks demonstrate that fine-tuning the 7B LVLMs with Vision-R1 achieves consistent performance gains, with even up to 50% improvement and surpassing the state-of-the-art 10× size model.

#### 1. Introduction

Recently, notable progress has been made in Large Vision Language Models (LVLMs) [2, 12, 25, 30, 31, 42], which encode images into textual tokens and respond to instruc-

Curate Instruction Data

Vision-R1: Gen. & Cal.

Advanced

Vision Criteria-Driven

Level

[Figure 1]

Reward Function

Step-Progression

RuleRefinement

Dual Format Reward

| |
|---|

Precision Reward

| |
|---|

[Figure 2]

Beginner’s

…

| |
|---|

Level

LVLMs

Vision-R1: Cal. &Update

Figure 1. Key designs of Vision-R1. Vision-R1 introduces vision criteria-driven reward function to holistically assess model completions and presents progressive rule refinement to ensure sustained improvement.

tions based on visual cues. These models typically follow a two-stage training paradigm, where pertaining establishes a foundational understanding of visual information, while supervised fine-tuning [30] enhances their ability to follow instructions and solve problems. Through this process, advanced LVLMs have shown remarkable potential in integrating vision and language to address complex tasks.

Despite these advancements, LVLMs still fall short of meeting human expectations as effectively as Large Language Models (LLMs) [1, 5, 29, 43], primarily due to limitations in vision-language data. To bridge this gap, preference optimization[13, 40, 46, 53], derived from LLMs[10, 35, 36] for its data efficiency and performance benefits, has been introduced as a post-training reinforcement strategy to refine LVLM responses based on human feedback. Although these methods reduce data consumption to the

thousand level, constructing high-quality vision-language preference datasets remains resource-intensive. Meanwhile, training a reliable reward model to capture nuanced preferences with varying subjectivity remains a major challenge.

With the success of LLM Deekseek-R1 [17], the rulebased Group Relative Policy Optimization (GRPO) [38] algorithm offers a new approach to track this challenge. While previously validated in reasoning tasks such as math [38] and code [16], R1 model further prove that rule-based rewards enhance comprehension and reasoning across multiple domains, enabling both reasoning and non-reasoning tasks performance improvement. Moreover, with the incorporation of visual information, vision-language questionanswer data becomes more objective and definitive, providing clearer solutions and cues. Existing human-annotated instruction data [26, 51] naturally provide precise responses that align with human preferences. This raises a critical question: Can an R1-like reinforcement learning method further enhance LVLM capabilities with curated visionlanguage instruction data?

In this paper, we propose Vision-R1, a novel visionguided R1-like reinforcement learning algorithm for LVLMs that eliminates the need for specialized reward models and handcrafted preference datasets. To achieve this, we conduct a comprehensive investigation into reward modeling and training strategy, as indicated in Figure 1. We first introduce a criterion-driven reward function to quantitatively evaluate each completion based on visual feedback, providing an objective standard for absolute rewards without relatively ranking based on preference data. This function delivers multi-dimensional reward signals guided by vision task criteria, such as precision to measure box accuracy via transforming textually numerical tokens to coordinates. Our design enables the model to develop a deeper understanding of task characteristics and generate more accurate responses, surpassing the token-level supervision used in SFT that ignores spatial identity. Building on the reward modeling, we further introduce a progressive rule refinement strategy that dynamically adjusts reward criteria throughout training to facilitate continuous improvement. Inspired by curriculum learning [4] and human learning processes, this strategy follows two key principles: differentiation and staged progression. This differentiation mechanism encourages the model to continuously refine its predictions for optimal performance. Meanwhile, training is structured into beginner and advanced phases with progressively stricter reward criteria in the advanced phase to prevent reward hacking and ensure sustained progression.

To validate the effectiveness of our approach, we train two advanced LVLMs, Griffon-G-7B [50] and Qwen2.5VL-7B [3], on the curated data and evaluate them across multiple in-domain and out-of-domain object localization tasks, as well as general QA benchmarks. Extensive experi-

ments demonstrate that: (1) Vision-R1 achieves significant performance enhancement across diverse tasks, including wild visual grounding and dense object detection, even surpassing the state-of-the-art Qwen2.5-VL-72B [3] model. (2) Compared to SFT, Vision-R1 demonstrates better generalization capabilities with an average of 6% improvement on unseen scenarios while maintaining advanced QA capabilities. The contribution of this paper is summarized as follows:

- • We propose a novel vision-guided reinforcement learning method Vision-R1 for LVLMs, which grants reward guided by vision feedback to facilitate a deeper task understanding beyond SFT.
- • We present an effective progressive rule refinement strategy, which ensures continuous improvement by dynamically adjusting reward criteria during training.
- • Comprehensive experiments demonstrate that Vision-R1 achieves superior performance gains on different models across both in-domain and out-of-domain scenarios with even up to 50% improvement for Qwen2.5-VL and maintains well generalization capabilities.

#### 2. Related Works

##### 2.1. Large Vision Language Models

In recent years, LVLMs[2, 12, 25, 30, 31, 39, 42] have made significant progress. By aligning with advanced LLMs [5, 29, 43] and leveraging high-quality instruction data [26, 42] for end-to-end training, LVLMs have greatly expanded their capabilities in tasks such as question answering and reasoning, achieving notable breakthroughs across various domains. Among these advancements, numerous open-source LVLMs have contributed through extensive research in data construction, alignment methods, model architecture, etc. Currently, InternVL-2.5 [11] and Qwen2.5VL [3] stand as the leading LVLM series, gradually closing the gap with close-source [1] models and even surpassing them on challenging benchmarks like MMMU [49]. Beyond these achievements, there is a growing focus on more challenging object localization tasks [3], such as visual grounding and object detection. While LVLMs have surpassed expert models in simpler fine-grained localization tasks like Referring Expression Comprehension (REC) [21], they still lag significantly behind specialized models in complex and dense object detection tasks. Although some studies, such as Griffon [50] and Lumen, [20] have explored this area, they remain limited to supervised finetuning, which offers limited performance gains. As object localization serves as a fundamental capability for enabling more advanced reasoning in LVLMs, it presents both a key research direction and a major challenge. In this paper, we further explore reinforcement learning-based post-training to enhance the performance of state-of-the-art LVLMs on

more demanding object localization tasks.

##### 2.2. Vision-Language Reinforcement Learning

With the rapid advancement of LVLMs, researchers have begun exploring reinforcement learning methods to better align these models with human preferences, inspired by the success of reinforcement learning in LLMs [10, 35, 36]. The first application in LVLMs named RLHF [40] aims to reduce hallucinations by iteratively optimizing model responses based on human feedback. To further enhance alignment and simplify training, Direct Preference Optimization (DPO) [36] is introduced, allowing models to be trained directly on human-annotated preference data. Since then, various preference optimization algorithms [47, 48] have been developed to improve dialogue capabilities, mitigate hallucinations, etc. As LVLMs continue to advance, some methods [13, 45] have also attempted to leverage reinforcement learning to enhance long-sequence reasoning. Despite reducing computational costs compared to pretraining while improving model performance, these approaches still rely on manually annotated preference data [53] and reward model training, making them resource-intensive and challenging. Inspired by the success of the rule-based GRPO [38] method in DeepSeek-R1 [17], we explore its application in the vision-language domain, where instruction datasets with precise annotations inherently align with human preferences. Our work shows that rule-based reinforcement learning, guided by visual feedback, can significantly enhance object localization tasks without requiring re-annotated preference data or reward model training. This further highlights its potential for broader applications in LVLMs.

#### 3. Vision-R1

In this section, we systematically introduce vision-anchored R1-like reinforcement learning algorithm Vision-R1, a success extension of GRPO [38] reinforcement learning algorithm to the vision field. We start with brief preliminaries about the rule-based GRPO algorithm, which is the success source of R1 models and our foundations. Then, we detail the pivotal component of Vision-R1 algorithm criteria-driven reward function in Section 3.2, specifically the criteria-driven reward function. Moreover, we introduce the progressive rule refinement strategy in Section 3.3. We illustrate the framework of Vision-R1 in Figure 2.

##### 3.1. Preliminaries

Building on the success of GRPO in enabling self-evolving, multi-domain reasoning within DeepSeek-R1 [17], this reinforcement learning algorithm provides valuable insights to both the language and vision communities. Since its supervision is based solely on the final outcome, GRPO is especially suited for tasks with explicit, objective answers.

Unlike other preference optimization methods relying on reward models or value models, it significantly reduces memory overhead for LVLMs. Furthermore, GRPO computes the relative advantages within a group of completions for a given sample, eliminating the need for manually annotated preference data. We further detail its training procedure and optimization loss as follows.

Given an initial model to be optimized, GRPO begins by initializing a trainable policy model πθ and a frozen reference model πref. For a given sample q, the old policy model πθ

first generates a group of completions {o1,o2,...,oN}. Then, the reward function freward computes the whole group rewards {r1,r2,...,rN}, which are further used to calculate the advantage Ai of each completion with the group by:

old

ri − mean({rj}Nj=1) std({rj}Nj=1)

(1)

Ai =

After the reference model computes the logits to output each completion given the question, the policy model πθ is optimized by maximizing the following objective:

N

πθ(oi|q) πθ

1 N

JGRPO(θ) =

Ai−βKL(πθ(oi|q)|πref(oi|q))

(

(oi|q)

old

i=1

(2) where N is the number of completions in one group and β is the hyper-parameter. This objective motivates the model to tend to produce the completion with a higher advantage within a group, but not to bias far away from the initial model.

##### 3.2. Criteria-Driven Reward Function

Previous approaches [16, 38] have primarily focused on domains such as mathematics and coding, where answers are often summarized using structured templates and evaluated through character-level matching. In contrast, visionlanguage tasks inherently have definitive answers, and object localization tasks typically do not involve intermediate steps but directly output the final result. While object localization tasks have clear objectives that identify all objects of interest, such visual feedback does not require strict character-level matching. Simply applying previous matching-based reward overlooks the unique characteristics of vision tasks and their feedback, as well as the advantages of reinforcement post-training that operates at the completion level.

To address this, we investigate to design a reward function that accounts for both the nature of object localization tasks and the limitations of current LVLMs in handling them. As shown in the task analysis in Figure 2, LVLMs [3, 11, 50] face three major challenges in object localization tasks. First, in multi-instance, long-sequence predictions, they often fail to follow instructions correctly, leading to formatting errors. Second, the model produces an

- 1 Completions Gen.

- 2 Criteria-Driven Reward Cal.

[Figure 3]

|<Plain Text> 160 171 172 182 teddy bear 594 276 639 368 potted plant<br><br>608 233 625 242 vase|
|---|

|<Plain Text> 210 174 289 232 tv<br><br>550 227 604 318 chair<br><br>579 222 608 280 chair 408 250 598 367 couch 186 242 413 414 couch 583 240 604 245 table|
|---|

[Figure 4]

[Figure 5]

Q: Category-Customized

Detection Prompt

Completion 1

Completion N

Policy Model

Dual Format Reward

Recall Reward

Precision Reward

Template Format Check

- 1. Match Prediction BBox
- 2. Cal. Per Completion Recall (Eq.5)

- 1. Match Prediction BBox
- 2. Cal. Completion-Mean IoU (Eq.6 }

Update

3 Cal. & Update

Content Format Check

Cal.Advantage

Optimization

PolicyGrad.

Adjusted Output: 0.6 Adjusted Output: 0.86

Output: 1.0

Cal.KL

Progressive Rule Refinement

Design

0 Task Analysis

[Figure 6]

[Figure 7]

218 177 288 232 tv

545 227 600 318 chair 579 222 608 28x chair 408,246,598,367 couch 594 276 639 368 plant

Long

Reference Model

Template & Content

Limited Num. of Valid

Inaccurate Predictions

Format Error

Predictions

Figure 2. Overall framework of Vision-R1. We start by analyzing the object localization tasks, and design our criteria-driven reward function based on the analysis. We illustrate the key step in one forward starting from inputting, generating all completions, calculating the reward with adjustment, to the final objective calculation and model update. Boxes in red indicate missed or wrong predictions, while the green ones are correct. Also, the solid line represents the predicted result, while the dashed line indicates the annotated ground truth.

insufficient number of valid predictions, failing to detect all mentioned objects. Third, it struggles with small or challenging objects, resulting in inaccurate predictions. Besides the formatting errors, the latter two issues are typically evaluated in object detection. Therefore, we propose a criteriondriven reward function, incorporating dual-format reward, recall reward, and precision reward to comprehensively assess model performance and incentivize improvement.

Box-prioritized Prediction Matching. LVLMs outputs object coordinates as textual sequences for object localization tasks due to the unified sequence modeling. To compute rewards based on visual feedback, we first convert these textual sequences into coordinate-based visual feedback as mentioned earlier. Existing LVLMs that support object localization tasks typically follow a fixed sequence representation for object coordinates, such as the plain-text format shown in Figure 2. Based on this representation, we extract individual objects from the sequence. However, object localization tasks often involve multiple objects, requiring exact matches between predictions and ground truth. To address this in training, we unify all object localization tasks under the general framework of object detection and conduct matching before computing rewards. Unlike detection

expert models, LVLMs do not generate class probabilities and are generally less precise in bounding box accuracy despite correctly predicting object categories. Based on our experiments, we introduce a simplification to the Hungarian matcher [6], prioritizing box-based loss for alignment. As indicated in Equation 3, after matching, each predicted instance contains coordinates, a category label, and an Intersection over Union score (IoU).

{Pmi }Mm=1 = extract match(oi) Pmi = {[x1,y1,x2,y2]im,labelmi ,IoUmi }

(3)

Dual Format Reward. Previous methods introduce format rewards to encourage adherence to predefined templates for easy answer extraction. Different from these methods, as illustrated in the first challenge, LVLMs directly output results for object localization tasks, but fall short in long-sequence prediction with both content and template format error. To address this, we design the dual format reward. For each completion oi, the template-format checking ftem will verify whether the completion follows the designated template format, such as JSON-format coordinates structure in Qwen2.5-VL [3]. Once met, we further validate the numerical content to ensure it adheres to

coordinate constraints, indicated as fcont, such as staying within valid bounds and correctly placing decimal points. We adopt a binary reward scheme, assigning a reward of 1 only when the prediction fully satisfies both format and content criteria as follows:

rewardDF(oi) =

1, if ftem = 1 ∧ fcont = 1 0, otherwise

(4)

Recall Reward. Recall is a crucial metric in object localization tasks, reflecting whether a model can predict all instances of interest as comprehensively as possible without omission. As shown in Figure 2, unlike specialized localization models, LVLMs typically predict confirmed but fewer valid instances than the actual number. Therefore, it is essential to incorporate recall quality into the evaluation of completion to encourage the model to identify all targets as it can. As shown in Equation 5, we follow the definition of recall in object detection and design a recall-based reward for each predicted completion. When the IoU of a matched predicted instance exceeds the predefined threshold ξ0, it is considered a valid prediction. The recall reward is the ratio of valid predictions in all GTs.

num(V alid Predictions) num(GT)

rewardrecall(oi) =

(5)

Precision Reward. Unlike the global perspective of recall, the precision reward focuses on the quality of the predicted instances of each completion for the third challenge. The precision reward works in conjunction with the recall reward: while the latter encourages the model to predict as many relevant instances as possible, the former ensures that the predictions are as accurate as possible. To directly motivate models to predict high-quality bounding boxes, we define the precision reward as the average IoU of all valid predictions:

M m=1[(IoUmi ≥ ξ0) · IoUmi ]

rewardprec(oi) =

M

(6)

The overall reward for each completion oi is the sum of all three rewards to comprehensively assess the completion anchored on the visual task criteria.

reward = rewardDF + rewardrecall + rewardprec (7)

##### 3.3. Progressive Rule Refinement Strategy

In localization tasks, accurately predicting a bounding box with high IoU to the ground truth is challenging, especially in dense scenes. This difficulty may lead to similar completion rewards for different predictions within the same group, limiting the model’s optimization. To address this, we propose a progressive rule refinement strategy, inspired by curriculum learning [4] and human learning processes, which

dynamically adjusts reward calculation criteria during training for continuous performance improvement. As shown in Figure 2, this strategy is applied to both recall and precision rewards, refining their final values for computing the advantage Ai. It consists of two key components: differentiation policy and Staged Progression policy.

Differentiation. The differentiation strategy focuses on increasing the contrast in the mapping between predictions and actual rewards. Unlike the previous linear mapping, we penalize predictions with low recall and average IoU while granting full rewards to those with relatively high recall and IoU. This adjustment encourages the model to generate high-quality responses within its current capability for optimal rewards. We denote the penalty threshold as ξ1 and the full reward threshold as ξ2, with the differentiation strategy expressed as Eq. 8. We apply this strategy to each instance for the precision reward for better stability, and directly adjust the recall reward for one completion.

 

1, if x ≥ ξ2 0, elif x < ξ1 x, otherwise

(8)

f(x) =



Staged Progression. Providing beginners with an easierto-achieve standard and gradually increasing the difficulty as their capability improves is a common learning strategy. We incorporate this principle into our design to encourage continuous model improvement and prevent reward hacking. The training process is divided into two phases: initial learning and advanced learning, based on training steps (STEP). In the initial phase, we set relatively low TP thresholds ξ0 and reward criteria ξ1,ξ2, referring to the threshold settings in object detection evaluations with 0.5, 0.5, and intermediate 0.75. With advancing, we tighten the criteria by adjusting the thresholds to their previous upper bounds: 0.75, 0.75, and 0.9. Since achieving perfectly accurate bounding box predictions is nearly impossible in object localization tasks, we set ξ2 slightly below 1. Through these strategy adjustments, the model can achieve continuous learning and improvement over time.

#### 4. Experiments

We conduct experiments on multiple object localization tasks and datasets to validate the effectiveness of our approach. In this section, we first introduce the implementation details of Vision-R1, including model configuration and training data in Sec. 4.1. Then, we compare our method with state-of-the-art LVLM models and benchmarks in Sec. 4.2, demonstrating its advanced performance in object detection, referring expression comprehension, and out-ofdomain scene localization. Furthermore, we provide indepth analytical experiments and ablation studies to examine various aspects of our method’s design in Sec. 4.3.

MSCOCO Val2017 ODINW-13

Type Model Res.

###### AP50 AP75 mAP Pistols Pothole Thermal Avg. mAP

Faster RCNN-FPN [37] 1022 58.6 40.9 37.9 - - - DAB-DETR [32] 1333 60.3 39.8 38.0 - - - DETR [6] 1333 62.4 44.2 42.0 - - - Pix2Seq [9] 1333 61.0 45.6 43.0 - - - GroundingDINO [33] 1333 - - 46.7 - - - 55.0

Specialists

Griffon-13B [51] 448 40.6 25.1 24.8 - - - Griffon v2 [52] 1022 54.3 41.2 38.5 - - - Lumen [20] 448 53.2 35.8 35.3 - - - InternVL2.5-8B [11] Dynamic 11.9 19.4 12.1 29.2 1.1 3.8 20.2 InternVL2.5-78B [11] Dynamic - - - - - - 31.7 Qwen2.5-VL-72B [3] Dynamic - - - - - - 43.1 Gemini 1.5 Pro [41] - - - - - - - 36.7

Generalist

Griffon-G-7B [50] 1022 57.4 42.8 40.2 59.3 11.7 49.2 43.8 + SFT 57.4 43.3 40.5 52.8 10.9 49.5 45.3 + Vision-R1 59.3 45.0 42.0 (+1.8) 56.3 17.2 55.9 46.3 (+2.5) Qwen2.5-VL-7B [3] Dynamic 27.3 18.0 17.7 48.5 7.4 38.0 37.0†

+ SFT 36.1 24.3 23.6 49.6 9.1 46.9 35.0

+ Vision-R1 40.0 27.8 26.6 (+8.9) 55.2 13.3 50.6 46.0 (+9.0)

- Table 1. Object localization results on common detection benchmark MSCOCO val2017 [28] and ODINW-13 [27] benchmarks. We follow ODINW evaluation of [3] using the visual grounding setting and report the Avg. mAP, which indicates the average mAP on all 13 evaluation datasets. The mAP of all datasets in ODINW-13 are detailed in the Appendix and three representative datasets are listed as below. †indicates we reproduce the result following the official setting.

##### 4.1. Implementation Details

Model Setting. We integrate Vision-R1 with several advanced LVLMs to verify the broad effectiveness of VisionR1. Specifically, we implement Vision-R1 based on the latest Qwen2.5-VL-7B [3] and Griffon-G-7B [50] models. Qwen2.5-VL-7B is the latest and most comprehensive multimodal large model, demonstrating competitive object localization capabilities in addition to its advanced VQA performance. In contrast, Griffon-G is the first LVLM to approach the performance of specialized localization models. Given their differing localization abilities, we select these two models to evaluate the effectiveness of our method across different model proficiency levels. As a post-training reinforcement learning approach, we directly fine-tune the open-source models using our constructed dataset of 49K samples which we introduce below. Training is conducted with the open-source Open-R1 [15] and its multimodal variant framework [8], utilizing the default configuration. Specifically, we set β to 0.2 and train for 1 epoch with the learning rate of 1e-6. For the comparison method SFT, we use the same data and fine-tune each model for 1 epoch with the learning rate of 2e-6 and batch size of 128. For rapid evaluation, we employ VLMEvalKit [14] and Griffon [51].

Training Data. As previously mentioned, Vision-R1 does

not require human-annotated preference data and can be directly trained using question-answer pairs with precise answer annotations. To construct the reinforcement learning data, we carefully curate samples from a previously fineannotated object localization instruction dataset. During the curation process, we adhere to two key principles: diversity and challenge. Ultimately, we construct a 49K reinforcement learning dataset, consisting of 30K object detection samples, 9K visual grounding samples, and 10K Referring Expression Comprehension samples, as object detection is generally more challenging than the other two tasks. Within each data category, we ensure that approximately 50% of the samples are challenging, featuring a greater number of object categories and instances, as well as a proportion of negative samples. A detailed illustration of the dataset is provided in the Appendix.

##### 4.2. Main Results on Object Localization

Setup. We provide extensive experimental results on a wide range of object localization benchmarks, which challenge the model to accurately detect and localize objects across diverse and complex environments, showcasing its advanced object localization abilities. We incorporate several widely recognized and representative in-domain datasets, spanning dense object detection and real-world scene localization.

Method boggleBoards MountainDewCommercial ThermalCheetah Vector Avg. GroundingDINO [33] 0.8 18.2 12.9 - InternVL2.5-8B [11] 0.1 0.0 0.7 6.7 1.9

Griffon-G-7B [50] 3.4 28.1 8.9 8.2 12.2 + SFT 2.3 13.7 9.4 24.0 12.4 + Vision-R1 3.9 41.5 7.8 24.1 19.3 (+7.1)

Qwen2.5-VL-7B [3] 4.5 3.8 7.8 51.3 16.9 + SFT 8.4 6.5 8.3 48.3 17.9 + Vision-R1 8.2 13.7 9.9 54.8 21.7 (+4.8)

- Table 2. Result on out-of-domain datasets collected from non-overlapping ODINW [27]. We follow the grounding setting in [3] for evaluation.

COCO [28] serves as a rigorous and well-acknowledged benchmark for assessing multi-object localization in dense scenes. ODINW-13 [27] covers 13 distinct real-world settings with rare object categories, testing the model’s capacity to apply its knowledge for object inference in practical scenarios. We also assess methods’ generalization ability on out-of-domain untrained localization datasets in challenging scenarios. We employ four Non-overlapping subsets from ODINW [27] individually.

In-domain Object Localization. The results in Tab. 1 demonstrate the broad effectiveness of the Vision-R1 in object localization tasks. When applied to the Griffon-G model, which excels in object detection, Vision-R1 further improves its performance by 1.8 on COCO and achieves an average mAP increase of 2.5 on ODINW-13. This significantly outperforms the state-of-the-art Qwen2.5-VL72B on ODINW-13 and brings Griffon-G-7B closer to the performance of specialized vision models. When integrated with the Qwen2.5-VL-7B model, which has relatively weaker localization capabilities, Vision-R1 yields even more substantial improvements, boosting COCO object detection performance by 8.9 points and achieving an 8.7-point gain on ODINW, surpassing the performance of its larger 72B counterpart. Compared to the Supervised Fine-Tuning method, Vision-R1 consistently outperforms it by an average of 1.25 and 7 points on the two models, respectively. Notably, SFT reduces Qwen2.5-VL-7B’s performance on ODINW-13, possibly due to over-fitting when training with limited data. These results highlight VisionR1’s strength in enhancing the LVLMs’ object localization capabilities with limited training data across different models and scenarios, particularly benefiting weaker models.

Out-of-domain Object Localization. As introduced in the setup, we incorporate four non-overlapping datasets from ODINW for out-of-domain localization evaluation. Unlike traditional out-of-domain detection setups, we relax the constraint that both images and object categories must be entirely unseen during training. Given the large-scale train-

Matcher Choice mAP AR100

Box-only 42.1 54.2 Box & Label 41.9 53.4

Table 3. Ablation study on different box matcher for LVLMs.

ing data of LVLMs, strictly ensuring complete novelty is challenging, we here define an experiment setting where either the object category or the scene is absent from the posttraining stage to assess generalization ability. As shown in Table 2, Vision-R1 improves performance when integrated with the Griffon-G-7B and Qwen2.5-VL-7B models, achieving average gains of 7.1 and 4.8, respectively. Notably, it surpasses expert models on BoggleBoards and MountainDewCommercial, further demonstrating its strong generalization capability beyond specific datasets. While the SFT performs competitively in challenging scenarios involving heatmaps, etc., where LVLMs initially struggle, it exhibits a significant performance drop in more common scenes compared to the base model. This suggests that SFT lacks robust generalization, whereas Vision-R1 effectively enhances both in-domain and out-of-domain performance.

##### 4.3. Ablation Studies

In this section, we provide comprehensive experiments to validate the design of Vision-R1, underscoring our key contributions. Unless otherwise specified, we conduct ablation experiments using the detection data from our constructed dataset, which can be regarded as a general form of localization tasks, making the experiments more representative and broadly applicable.

Discussion on Different Matcher Approaches. As mentioned in Section 3.2, prior box matching is typically based on Hungarian matching, which minimizes loss by considering both box accuracy and category prediction scores. However, unlike detection expert models, LVLMs do not rely on a predefined category set with probabilistic outputs, and di-

###### precision recall mAP AP50 AP75 AR100

Baseline 40.2 57.4 42.8 52.2 ✓ 41.5 55.6 45.1 49.6 ✓ ✓ 42.1 58.7 45.3 54.2

Table 4. Ablation study on Reward Function Design.

###### STEP mAP AP50 AP75 AR100

Baseline 40.2 57.4 42.8 52.2 1/3 41.5 58.0 44.8 54.6 1/2 42.1 58.7 45.3 54.2

1 39.9 57.0 42.6 56.7

Table 5. Ablation study on Progressive Rule Refinement. STEP determines the training stage at which adjustments begin.

rectly produce deterministic category labels instead. Building on this, we simplify the assignment process by either considering only box accuracy or incorporating both box accuracy and category correctness. As shown in Table 3, the two approaches exhibit limited significant performance difference, with the box-only matching method performing slightly better. We attribute this to the strong classification ability of LVLMs, which rarely misclassify objects, when predicting a small number of targets. Matching solely based on bounding boxes helps the model recall more objects, leading to a slight performance gain after training by enabling more accurate predictions.

Effectiveness of Reward Function Design. To comprehensively evaluate the design of our reward function, we first conduct an ablation study to compare the effects of the three reward components. Among them, the dual format reward primarily serves as feedback for some completions where the model fails to follow the expected format or content template. Therefore, we focus our ablation comparison on precision reward and recall reward. When excluding the recall reward, we introduce a binary prediction count reward, which grants a reward only when the predicted number of instances matches the ground truth. This prevents the model from continuously generating redundant outputs. As shown in Table 4, when only precision is considered, the model produces higher-quality bounding boxes, leading to an increase in all levels of AP. However, the number of recalled instances decreases. With the introduction of the recall reward, the model’s recall rate increases by 2% compared to the baseline and the overall mAP further improves by 0.6, demonstrating that our design to integrate recall and precision leads to more effective performance.

Effectiveness of Progressive Rule Refinement. The progressive rule refinement strategy serves as a mechanism to encourage continuous model improvement. In our experi-

Method GQA AI2D ChartQA SEED

Griffon-G-7B [50] 64.6 70.1 68.7 71.7 + SFT 63.5 70.5 67.5 72.2 + Vision-R1 64.8 70.3 68.8 71.8

Table 6. Ablation on generalization QA capabilities. Results are all produced by VLMEvalKit under the same setting.

ments, we set and fixed ξ following object detection evaluation criteria while adjusting STEP to determine the optimal transition point for the advanced phase. To examine the impact of different configurations, we conducted a comparative study on the Griffon-G-7B model, evaluating three settings where STEP was set to 1/3, 1/2, and 1, and tested the performance on COCO. As shown in Table 5, adjusting the model at STEP = 1/2 yielded the best performance, whereas keeping STEP = 1 (i.e., no adjustment) resulted in performance lower than the baseline. Our analysis suggests that for the Griffon-G model, which initially possesses strong localization abilities, recall has a greater impact during training. As a result, it achieved an AR100 of 56.7. However, without progressive reward adjustments, the model generated a large number of lower-quality bounding boxes, contributing to more false positives in AP metrics, ultimately reducing mAP slightly below the baseline. When adding our strategy, it will suppress these low-quality boxes chasing for more high-quality boxes. While for the comparably weak Qwen2.5-VL-7B model, the situation is different with STEP = 1 yielding the best results as reported, which we detail in the Appendix. These overall results validate the importance and effectiveness of our progressive rule refinement strategy, demonstrating properly tuning the training process leads to meaningful performance improvements.

Effects on General QAs. Vision-R1 aligns LVLMs with subjective annotations that human naturally prefers to advance their object localization capabilities. However, it is also well preferred to remain LVLMs’ strong general QA capabilities. We evaluate both LVLMs integrated with Vision-R1 in Table 1 and 2 across various general VQAs, including knowledge (AI2D [22]), commonsense (GQA [19]), chart (ChartQA [34]), and interdisciplinary (SEED [24]) domains. As shown in Table 6, training with VisionR1 results in minimal fluctuations in general QA performance, maintaining a performance similar to the baseline model, while SFT methods show a significant drop. This indicates that our method significantly enhances object localization without heavily compromising general QA abilities. Moreover, the improvement in object localization leads to a performance boost on object-perception-based commonsense tasks like GQA, further showcasing the advantages of our approach. We also provide experimental results for Qwen2.5-VL-7B in the appendix, further demonstrating the

effectiveness of our method.

#### 5. Conclusion

In this paper, we introduce Vision-R1, a novel reinforcement learning algorithm for LVLMs that combines a vision criterion-driven reward function and progressive rule refinement strategy to enhance their object localization capabilities. By designing this algorithm, we present a human-annotating-free approach to leverage abundant instruction data with subjective and definite responses embodied to boost LVLMs’ localization performance. Comprehensive evaluation across various benchmarks under diverse scenarios demonstrates the generalized effectiveness of our method, encouraging more research to equip LVLMs with advanced precise object localization capabilities to support complex tasks and real-life applications.

#### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 1, 2

- [2] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023. 1, 2
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 2, 3, 4, 6, 7, 1
- [4] Yoshua Bengio, J´erˆome Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In Proceedings of the 26th annual international conference on machine learning, pages 41–48, 2009. 2, 5
- [5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 1, 2
- [6] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020. 4, 6
- [7] Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jiarui Xu, Zheng Zhang, Dazhi Cheng, Chenchen Zhu, Tianheng Cheng, Qijie Zhao, Buyu Li, Xin Lu, Rui Zhu, Yue Wu, Jifeng Dai, Jingdong Wang, Jianping Shi, Wanli Ouyang,

- Chen Change Loy, and Dahua Lin. MMDetection: Open mmlab detection toolbox and benchmark. arXiv preprint arXiv:1906.07155, 2019. 1
- [8] Liang Chen, Lei Li, Haozhe Zhao, Yifan Song, and Vinci. R1-v: Reinforcing super generalization ability in visionlanguage models with less than $3. https://github. com/Deep-Agent/R1-V, 2025. Accessed: 2025-02-02. 6
- [9] Ting Chen, Saurabh Saxena, Lala Li, David J Fleet, and Geoffrey Hinton. Pix2seq: A language modeling framework for object detection. arXiv preprint arXiv:2109.10852, 2021. 6
- [10] Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning converts weak language models to strong language models. arXiv preprint arXiv:2401.01335, 2024. 1, 3
- [11] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and testtime scaling. arXiv preprint arXiv:2412.05271, 2024. 2, 3, 6, 7
- [12] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198, 2024. 1, 2
- [13] Yuhao Dong, Zuyan Liu, Hai-Long Sun, Jingkang Yang, Winston Hu, Yongming Rao, and Ziwei Liu. Insight-v: Exploring long-chain visual reasoning with multimodal large language models. arXiv preprint arXiv:2411.14432, 2024. 1, 3
- [14] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 11198–11201, 2024. 6
- [15] Hugging Face. Open r1: A fully open reproduction of deepseek-r1, 2025. 6
- [16] Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Yu Wu, YK Li, et al. Deepseek-coder: When the large language model meets programming–the rise of code intelligence. arXiv preprint arXiv:2401.14196, 2024. 2, 3
- [17] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 2, 3
- [18] Shuting He, Henghui Ding, Chang Liu, and Xudong Jiang. Grec: Generalized referring expression comprehension. arXiv preprint arXiv:2308.16182, 2023. 1
- [19] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019. 8

- [20] Yang Jiao, Shaoxiang Chen, Zequn Jie, Jingjing Chen, Lin Ma, and Yu-Gang Jiang. Lumen: Unleashing versatile vision-centric capabilities of large multimodal models. Advances in Neural Information Processing Systems, 37: 81461–81488, 2025. 2, 6
- [21] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 787–798, 2014. 2, 1
- [22] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 235–

251. Springer, 2016. 8, 1

- [23] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123:32–73, 2017. 1
- [24] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023. 8
- [25] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Fanyi Pu, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Mimicit: Multi-modal in-context instruction tuning. arXiv preprint arXiv:2306.05425, 2023. 1, 2
- [26] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 2
- [27] Liunian Harold Li*, Pengchuan Zhang*, Haotian Zhang*, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, Kai-Wei Chang, and Jianfeng Gao. Grounded language-image pre-training. In CVPR, 2022. 6, 7, 1
- [28] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13, pages 740–755. Springer, 2014. 6, 7, 1
- [29] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024. 1, 2
- [30] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023. 1, 2
- [31] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024. 1, 2
- [32] Shilong Liu, Feng Li, Hao Zhang, Xiao Yang, Xianbiao Qi, Hang Su, Jun Zhu, and Lei Zhang. Dab-detr: Dynamic

- anchor boxes are better queries for detr. arXiv preprint arXiv:2201.12329, 2022. 6
- [33] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European Conference on Computer Vision, pages 38–55. Springer, 2024. 6, 7
- [34] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022. 8
- [35] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730– 27744, 2022. 1, 3
- [36] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2023. 1, 3
- [37] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. IEEE transactions on pattern analysis and machine intelligence, 39(6):1137–1149, 2016. 6
- [38] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 2, 3
- [39] Andreas Steiner, Andr´e Susano Pinto, Michael Tschannen, Daniel Keysers, Xiao Wang, Yonatan Bitton, Alexey Gritsenko, Matthias Minderer, Anthony Sherbondy, Shangbang Long, et al. Paligemma 2: A family of versatile vlms for transfer. arXiv preprint arXiv:2412.03555, 2024. 2
- [40] Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, Liang-Yan Gui, Yu-Xiong Wang, Yiming Yang, et al. Aligning large multimodal models with factually augmented rlhf. arXiv preprint arXiv:2309.14525, 2023. 1, 3
- [41] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 6
- [42] Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Jairam Vedagiri IYER, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. Advances in Neural Information Processing Systems, 37:87310–87356, 2024. 1, 2
- [43] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 1, 2

- [44] Jiaqi Wang, Pan Zhang, Tao Chu, Yuhang Cao, Yujie Zhou, Tong Wu, Bin Wang, Conghui He, and Dahua Lin. V3det: Vast vocabulary visual detection dataset. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19844–19854, 2023. 1
- [45] Weiyun Wang, Zhe Chen, Wenhai Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Jinguo Zhu, Xizhou Zhu, Lewei Lu, Yu Qiao, and Jifeng Dai. Enhancing the reasoning ability of multimodal large language models via mixed preference optimization. arXiv preprint arXiv:2411.10442,

2024. 3

- [46] Tianyi Xiong, Xiyao Wang, Dong Guo, Qinghao Ye, Haoqi Fan, Quanquan Gu, Heng Huang, and Chunyuan Li. Llavacritic: Learning to evaluate multimodal models. arXiv preprint arXiv:2410.02712, 2024. 1
- [47] Tianyu Yu, Yuan Yao, Haoye Zhang, Taiwen He, Yifeng Han, Ganqu Cui, Jinyi Hu, Zhiyuan Liu, Hai-Tao Zheng, Maosong Sun, et al. Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13807– 13816, 2024. 3
- [48] Tianyu Yu, Haoye Zhang, Yuan Yao, Yunkai Dang, Da Chen, Xiaoman Lu, Ganqu Cui, Taiwen He, Zhiyuan Liu, Tat-Seng Chua, et al. Rlaif-v: Aligning mllms through open-source ai feedback for super gpt-4v trustworthiness. arXiv preprint arXiv:2405.17220, 2024. 3
- [49] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556– 9567, 2024. 2
- [50] Yufei Zhan, Hongyin Zhao, Yousong Zhu, Fan Yang, Ming Tang, and Jinqiao Wang. Griffon-g: Bridging visionlanguage and vision-centric tasks via large multimodal models. arXiv preprint arXiv:2410.16163, 2024. 2, 3, 6, 7, 8, 1
- [51] Yufei Zhan, Yousong Zhu, Zhiyang Chen, Fan Yang, Ming Tang, and Jinqiao Wang. Griffon: Spelling out all object locations at any granularity with large language models. In European Conference on Computer Vision, pages 405–422. Springer, 2024. 2, 6
- [52] Yufei Zhan, Yousong Zhu, Hongyin Zhao, Fan Yang, Ming Tang, and Jinqiao Wang. Griffon v2: Advancing multimodal perception with high-resolution scaling and visual-language co-referring. arXiv preprint arXiv:2403.09333, 2024. 6
- [53] Yi-Fan Zhang, Tao Yu, Haochen Tian, Chaoyou Fu, Peiyan Li, Jianshu Zeng, Wulin Xie, Yang Shi, Huanyu Zhang, Junkang Wu, et al. Mm-rlhf: The next step forward in multimodal llm alignment. arXiv preprint arXiv:2502.10391,

2025. 1, 3

## Vision-R1: Evolving Human-Free Alignment in Large Vision-Language Models via Vision-Guided Reinforcement Learning

### Supplementary Material

In this supplementary material, we provide details on dataset construction, task templates, evaluation data, and detailed results. We also present a further analysis of the progressive adjustment strategy.

#### 1. Training Data Construction

Our method does not rely on human preference data but instead selects localization-related instruction data for training. As described in Section 4.1, we curate data from open-source localization instruction datasets, primarily covering object detection, visual grounding, and REC. The data sources and their quantities are summarized in Table 7, with the selection process detailed below:

Object Detection For object detection, we select data from MS COCO [28], which includes a diverse range of scenes and object categories. We define that images with more than 10 instances as hard cases, considering the category count and the current capabilities of LVLMs. We follow the distribution of raw data, and sample one-third of the data from COCO’s difficult and easy samples. Since object detection serves as a general representation of localization tasks, it is inherently more complex and can help improve localization performance across other tasks. As a result, we prioritize object detection data, selecting a total of 30K samples.

Visual Grounding To enhance the model’s ability to localize objects across diverse scenes and categories, we incorporate ODINW and V3Det datasets, which cover over 13K categories. We convert these datasets into a visual grounding format, ensuring consistency with our task requirements. Additionally, following Griffon-G, we include a small portion of multi-category annotations from V3Det. From these two datasets, we collect 5K and 4K samples each.

Referring Expression Comprehension For REC tasks, we start with the widely used RefCOCO dataset [21] and further consider more general cases where a referring expression corresponds to multiple objects, as highlighted in GRefCOCO [18]. We collect 5K samples from RefCOCO and additionally extract multi-object referring expressions from the Visual Genome dataset [22].

This dataset selection process ensures a balanced and comprehensive training foundation for improving LVLM performance on challenging object localization tasks.

Type Num. Source Object Detection 30K COCO [28] Visual Grounding 9K ODINW [27], V3Det [44]

REC 10K RefCOCO [21], Visual Genome [23]

Table 7. Details of constructed training dataset.

#### 2. Task Templates

After constructing the dataset, we further build the instruction templates for object localization tasks. Our approach primarily follows the task templates used during the instruction fine-tuning phase of the selected models [3, 50]. For tasks not explicitly covered in the instruction fine-tuning phase, we adapt and adjust templates based on related task formats. Finally, we create five instruction templates for model training and evaluation, as summarized in Table 8.

#### 3. Detailed Results

As demonstrated in Section 4.2, we follow the Qwen2.5VL to evaluate our method on ODINW-13 [27], in which the evaluation sets follows the setting of the mmdetection toolkit [7]. Due to the page length limitation, we do not provide the detailed results of all 13 datasets in the main body. To clearly demonstrate Vision-R1’s advancement, we list the results of all 13 datasets from different models in Table 9. As seen in the table, Vision-R1 improves these two models on most of the sets by a large margin, which highlights the effectiveness of our method.

#### 4. Additional Ablation Studies

Further Analysis of Progressive Rule Refinement Strategy

The progressive rule refinement strategy is an effective approach to preventing reward hacking, ensuring continuous performance improvement. As demonstrated in Section 4.3, for the Griffon-G-7B model, which already exhibits relatively strong performance, the model consistently earns high precision rewards on average, while recall is relatively lower, making the recall reward more influential. As recall increases without an improvement in box quality, overall precision remains unchanged or slightly decreases. After employing our strategy, we assign full rewards (value = 1) to high-quality bounding boxes, encouraging the model to refine box accuracy. This, combined with the increased recall, leads to a mAP improvement. By adjusting STEP, we

Model Task Template

Examine the image for any objects from the category set. Report the coordinates of each detected object. The category set includes {category list}.

Object Detection

Griffon-G

Visual Grounding Locate the exact position of {category} in the picture, if you can. REC

Can you point out {ref expression} in the image and provide the coordinates of its location?

Locate every item from the category list in the image and output the coordinates in JSON format. The category set includes {category list}.

Object Detection

Qwen2.5-VL

Visual Grounding REC

Locate every {category} in the image and output the coordinates in JSON format.

Table 8. Training and evaluation templates for each model and task.

ODINW-13

Model

AerialDrone Aquarium Rabbits

EgoHands Mushrooms

Packages PascalVOC

Pistols Pothole Raccoon Shellfish Thermal Vehicles

AVG

InternVL2.5-8B[11] 0 6.9 38.5 0.2 26.7 16.4 37.0 29.2 1.1 46.6 28.5 3.8 27.1 20.2 Qwen2.5-VL-3B[3] 6.2 16.4 75.0 24.6 8.3 66.6 52.0 42.3 10.2 47.7 36.7 40.7 57.1 37.2

Griffon-G-7B [50] 9.5 18.7 74.3 25.6 39.7 56.5 59.3 55.1 11.7 51.3 51.8 49.2 67.2 43.8 Griffon-G-7B-SFT 12.9 20.7 75.5 50.6 39.9 54.6 61.2 52.8 10.9 49.5 48.1 49.5 62.9 45.3 Griffon-G-7B-Vision-R1 12.2 21.5 75.2 48.0 50.4 54.1 62.2 56.3 17.2 48.2 40.7 55.9 59.9 46.3 Qwen2.5-VL-7B[3] 7.8 20.3 73.5 32.2 7.0 57.6 49.8 48.5 7.4 40.1 42.7 38.0 56.3 37.0 Qwen2.5-VL-7B-SFT 0.9 8.8 78.8 17.4 8.0 48.4 51.5 49.6 9.1 44.9 36.0 46.9 55.1 35.0 Qwen2.5-VL-7B-Vision-R1 7.5 22.6 77.1 49.9 67.7 50.2 54.4 55.2 13.3 43.9 49.9 50.6 56.5 46.0

Table 9. Detailed results on ODINW-13 dataset.

STEP mAP AP50 AP75 AR100 Baseline 17.7 27.3 18.8 29.8

1/2 23.3 32.0 25.2 28.9 1 26.6 40.0 27.8 36.7

Table 10. Experiments on Progressive Rule Refinement with Qwen2.5-VL-7B.

Method GQA AI2D ChartQA SEED

Qwen2.5-VL-7B [3] 58.8 80.8 87.5 76.7 + SFT 53.5 80.2 83.4 76.2 + Vision-R1 61.0 80.3 86.0 75.5

Table 11. Ablation on generalization QA capabilities. Results are all produced by VLMEvalKit under the same setting.

control when the model transitions into the advanced training phase to get the optimal performance.

For Qwen2.5-VL-7B, which has weaker localization capabilities, the optimal STEP setting differs, as shown in Table 10. Given our ξ hyperparameter settings, Qwen2.5-VL struggles to consistently achieve an average precision and recall above 0.75. In this case, setting STEP = 1 (no adjustment) yields the best performance. When adjusting the reward criteria midway, the model fails to meet the stricter standards, leading to reduced training efficiency and weaker performance. However, all settings still outperform the baseline, demonstrating the effectiveness of our method.

In summary, the STEP hyperparameter makes our progressive rule refinement strategy highly adaptable to models with different capability levels. For stronger models, progressive rule adjustment during training continuously reinforces optimization. For weaker models, a later STEP setting allows the model to first meet the initial reward criteria before transitioning. If the model fails to meet the predefined adjustment threshold within the given data volume, setting STEP = 1 is a viable option. By following these guidelines, users can tune hyperparameters based on specific models and datasets to achieve optimal performance.

###### General VQAs results for Qwen2.5-VL model. We

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Qwen2.5-VL

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

+Vision-R1

|[Figure 16]|
|---|

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Grounding Truth

Figure 3. Qualitative analysis results using Qwen2.5-VL-7B

provide ablation results on generalization QA capabilities for Qwen2.5-VL-7B here in Table 11. The result with slight difference, further demonstrating the effectiveness of our method, that our method significantly enhances object localization without heavily compromising general QA abilities.

#### 5. Qualitative Analysis

We provide a quantitative analysis to better demonstrate the effectiveness of our approach. For this analysis, we use the Qwen2.5-VL-7B model, which shows a more significant performance improvement, making the qualitative impact of our method more evident.

As shown in Figure 3, the original Qwen2.5-VL-7B model often produces redundant outputs, suffers from a high number of missed detections, and generates imprecise bounding boxes in detection tasks. After training with our method, the model eliminates redundant and invalid outputs, significantly improves recall, and maximizes the retrieval of relevant targets, ultimately achieving more precise localization.

