# Efficient Medical VIE via Reinforcement Learning

## Lijun Liu1,*, Ruiyang Li1,*, Zhaocheng Liu1,† Chenglin Zhu1,2, Chong Li1,2, Jiehan Cheng1,3, Qiang Ju1, Jian Xie1 1Baichuan Inc. 2Peking University 3Renmin University of China

Correspondence: lio.h.zen@gmail.com

arXiv:2506.13363v1[cs.CL]16Jun2025

## Abstract

Visual Information Extraction (VIE) converts unstructured document images into structured formats like JSON, critical for medical applications such as report analysis and online consultations. Traditional methods rely on OCR and language models, while end-to-end multimodal models offer direct JSON generation. However, domain-specific schemas and high annotation costs limit their effectiveness in medical VIE. We base our approach on the Reinforcement Learning with Verifiable Rewards (RLVR) framework to address these challenges using only 100 annotated samples. Our approach ensures dataset diversity, a balanced precision-recall reward mechanism to reduce hallucinations and improve field coverage, and innovative sampling strategies to enhance reasoning capabilities. Fine-tuning Qwen2.5-VL-7B with our RLVR method, we achieve state-of-the-art performance on medical VIE tasks, significantly improving F1, precision, and recall. While our models excel on tasks similar to medical datasets, performance drops on dissimilar tasks, highlighting the need for domain-specific optimization. Case studies further demonstrate the value of reasoning during training and inference for VIE.

## 1 Introduction

Visual Information Extraction (VIE) (Wan et al., 2024; Kuang et al., 2023; Hong et al., 2022; Kim et al., 2022) aims to generate structured information, such as JSON, from unstructured document images. This capability is crucial for various medical applications such as report interpretation (Li et al., 2024) and online consultations (Liu et al., 2025b). The most common approach involves first applying Optical Character Recognition (OCR) (Feng et al., 2025; Poznanski et al., 2025; Wei et al., 2024) to extract text, followed by leveraging large language models (LLMs) to extract and organize the text into a JSON structure. Additionally, end-to-end methods (Wan et al., 2024; Bai et al., 2025; Kuang et al., 2023; Kim et al., 2022) have emerged, including multimodal large models that directly output JSON from image inputs.

However, VIE tasks are highly domain-specific, with each domain requiring customized schemas. (Park et al., 2019; Huang et al., 2019b) The keys and values within these schemas are often defined by intricate domain-specific details, posing significant challenges for applying general-purpose VIE models to specialized fields. This aspect fundamentally differentiates structured VIE from OCR. Moreover, the annotation cost for VIE tasks is relatively high. These challenges have resulted in suboptimal performance of existing methods in medical VIE scenarios.

Given the nontrivial relationship between diverse image inputs and outputs conforming to predefined schemas, we argue that VIE models need reasoning capabilities (OpenAI, 2024b) to address these complexities. To mitigate the high annotation cost, we explore efficient training paradigms using only 100 annotated samples. Combining these two considerations, we adopt Reinforcement Learning with Verifiable Rewards (RLVR) (Guo et al., 2025; Team et al., 2025) to achieve efficient medical visual extraction.

Specifically, our design within the RLVR framework focuses on three key aspects. First, we ensure diversity in the 100 image samples to make the dataset representative and varied. Second, we carefully design the reward mechanism by incorporating a weighted combination of precision and recall, where precision reduces model hallucinations and recall ensures the model captures all the predefined fields of interest. Lastly, we adopt two sampling strategies: one requires each response to include all fields,

with rewards calculated against the ground truth for all fields, while the other evaluates responses using a random subset of fields from the total schema. By integrating these carefully designed components, we aim to establish an efficient and robust solution for medical VIE tasks.

Based on our proposed method, we finetuned Qwen2.5-VL-7B to have VIE RLVR models. We evaluate the models on medical and general VIE tasks. Our VIE RLVR models achieve SOTA performance on F1, precision and recall metrics on medical VIE tasks, indicating the advantage of our proposed method. We chose four widely-considered general VIE tasks for further evaluation. On two tasks that are similar to the medical report dataset, our models highly outperforms Qwen2.5-VL-7B. Meanwhile on other two tasks that are much different with our medical report dataset, our models failed to outperforms the base model, revealing the significant gap between different VIE tasks. The comparison of VIE models with and without thinking process while training and inferring are also delivered. In our case studies one can see how model benefits from thinking in dealing with VIE tasks.

## 2 Related Work

- 2.1 Visual Information Extraction

Visual Information Extraction (VIE) converts unstructured document images into structured outputs (e.g., key–value pairs or JSON), supporting applications like receipt understanding, form parsing, and medical document analysis(Huang et al., 2022; Powalski et al., 2021; Appalaraju et al., 2021). Existing methods fall into two main types: two-stage approaches that apply OCR followed by language models for structural parsing(Xu et al., 2020b,a), and end-to-end models that directly generate outputs from images without OCR (Kim et al., 2022; Zhang et al., 2020). Though effective on low-complexity benchmarks such as FUNSD, SROIE, and CORD (Jaume et al., 2019; Huang et al., 2019a; Park et al., 2019; Cao et al., 2022; Wang et al., 2021a), these models often omit required fields, hallucinate content, and generalize poorly to unseen layouts—especially under few-shot or domain-shift conditions. These issues are exacerbated in the medical domain(Ma et al., 2023; Zheng et al., 2022), where layouts vary widely and annotated data is scarce. While recent advances like layout-aware pretraining (Chen et al., 2022; Adnan et al., 2024; Luo et al., 2023), graph-based models (Yu et al., 2021), and schema-guided prompting (Wang et al.; Li

- et al., 2024; Yao et al., 2024) provide partial solutions, they often fall short of ensuring both structural completeness and semantic accuracy under low-resource constraints.

2.2 Reinforcement Learning for MLLM Reasoning

Reinforcement Learning (RL) has emerged as a pivotal research direction for enhancing the complex reasoning capabilities of LLMs (Guo et al., 2025; Jaech et al., 2024; Shao et al., 2024; Hui et al., 2024; Ying et al., 2024). OpenAI-o1 (Jaech et al., 2024) adopted Reinforcement Learning from Human Feedback (RLHF) during the fine-tuning process, significantly enhancing the model’s reasoning abilities and its alignment with human preferences. More recently, DeepSeek-R1 (Guo et al., 2025) employed GRPO (Shao et al., 2024), which, unlike traditional RL algorithms dependent on critic models, directly utilizes rule-based verifiable rewards to guide the model’s reasoning process. This approach has greatly simplified the training procedure and proven highly effective in improving reasoning capabilities. This trend is gradually extending to MLLMs to further enhance their visual reasoning abilities (Xu et al., 2024; Liu

- et al., 2025a; Yu et al., 2025a; Yang et al., 2025; Zhou et al., 2025). Studies such as Visual-RFT (Liu et al., 2025c) and VLM-R1 (Shen et al., 2025) have shown that for single-image visual grounding tasks, direct application of few-shot GRPO can achieve improvements surpassing supervised fine-tuning. The GoT-R1 (Duan et al., 2025) framework applies RL to enhance semantic spatial reasoning in visual generation. Vision-R1 (Huang et al., 2025) enhances multimodal mathematical reasoning capabilities by using DeepSeek-R1 to augment multimodal Chain of Thought (CoT) datasets and adopting step-by-step thought inhibition during GRPO training. In this research, we aim to extend this paradigm to the Medical VIE tasks mentioned earlier.

Response <think> Based on the content of the image and in accordance with the requirements, I will extract and structure the report content with the following specific steps:

Key descriptions: response { "Examination Name": "", # The names of the examination ...

..., "Indicators": [ # The various indicators for the examination

items conducted on the patient are generally presented in... {"Item Name": "", #... "Result": ""#... ,"Unit": ""#...,... },

[Figure 1]

- 1. Sample Collection Time: Extracted from the report.
- 2. Department: Pediatrics.
- 3. Examination Description: Typically, there is descriptive information located below the charts.

LM Decoder

... ]

}

... The extracted key information is as follows:

- 1. Add comments for keys;
- 2. Remove values;
- 3. Keep only one element for list.

texts tokens pictures tokens

- - **Examination Name**: The title appears as Pulmonary Function Diffusing Capacity Test Report.
- - **Treatment Recommendations**: Provided after analyzing the relevant medical indicators extracted using a table.
- - **Indicator Extraction**:
- - Information such as "FEV1", "FVC" with results

You are a annotator, mainly ... You need ... The answer format you need to follow is as follows:

Sampled key-value pairs as Json GT { "Examination Name": "Lung Function Diffusing Capacity Test",

[Figure 2]

Vision-Language Projection

..., "Indicators": [

needs to be extracted from the table. </think>

{"Item Name": "FEV1","Result": "1.61","Unit": "[L]",... }, {"Item Name": "FEV1 % FVC","Result": "","Unit": "[%]",...},...]

{key_descriptions} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format.

}

<answer> ```json { "Examination Name": "Lung Function Diffusing Capacity Test",

Vision Encoder ❄

️

Sampling strategy

..., "Indicators": [

{"Item Name": "FEV1","Result": "1.61","Unit": "[L]",... },

Key-value pairs {

{"Item Name": "FEV1 % FVC","Result": "","Unit": "[%]",...},...] } </answer>

[Figure 3]

"Examination Name": "Lung Function Diffusing Capacity Test", "Age": "6 Years",

Json Prediction and Json GT are the inputs of our rule-based reward function

...,

"Indicators": [ {"Item Name": "FEV1","Result": "1.61","Unit": "[L]",...}, {"Item Name": "FEV1 % FVC","Result": "","Unit": "[%]", ...},

Json Prediction

... ], ...

❄

[Figure 4]

Frozen Trainable

️

}

- Figure 1: Overview of our proposed method. A sampling strategy is employed to randomly select key-value pairs from the annotated JSON data associated with each image, serving as the ground truth. The descriptions of these keys are incorporated into a predefined prompt to generate queries. These queries, along with the corresponding images, are input into the model to produce responses. The JSON predictions are then extracted from the responses, and rewards are computed by comparing them with the ground truth.

## 3 Method

- 3.1 Preliminary

Due to the intricate relationship between heterogeneous image inputs and outputs that adhere to predefined schemas, we argue that VIE models must possess reasoning abilities to effectively manage these complexities. In contrast to approaches that explicitly replicate intermediate reasoning steps, RLVR (Guo et al., 2025; Team et al., 2025) relies solely on outcome-driven feedback, facilitating scalable reinforcement learning across extensive task datasets.

Group Relative Policy Optimization (GRPO) (Guo et al., 2025) is an efficient RL algorithm that eliminates the need for a separate critic model. Given a query q, GRPO samples a group of G outputs {o1,o2,...,oG} from the old policy πθold. These outputs are evaluated using reward functions to obtain individual rewards {r1,r2,...,rG}. The advantage is computed by normalizing the rewards within the group:

ri − mean({r1,r2,...,rG}) std({r1,r2,...,rG})

. (1) Then the policy is updated by optimizing the following objective:

Ai =

JGRPO(θ) = Eq∼D,{o

i}Gi=1∼πθold

|oi|

G

1 G

1 |oi|

min φi,t(θ)Ai,t, clip(φi,t(θ), 1 − ϵ, 1 + ϵ)Ai,t − β DKL [πθ∥πref] , (2)

t=1

i=1

where

πθ(oi,t | q,oi,<t) πθold(oi,t | q,oi,<t)

. (3) Additionally, we adopt several key techniques from DAPO (Yu et al., 2025b), including Clip-Higher

φi,t(θ) =

and Token-Level Policy Gradient Loss. With the introduction of the two, the objective function undergoes

some slight modifications as follows:

JGRPO(θ) = Eq∼D,{o

i}Gi=1∼πθold

|oi|

G

1

min φi,t(θ)Ai,t, clip(φi,t(θ), 1 − ϵlow, 1 + ϵhigh)Ai,t − β DKL [πθ∥πref] .

G i=1 |oi|

t=1

i=1

(4)

#### 3.2 Image Diversity

We collected over 17,000 medical domain images along with their corresponding OCR ground truth. These images encompass a diverse range of report types, including laboratory reports (e.g., blood, urine, and stool tests), diagnostic reports (e.g., endoscopy, electrocardiograms, ultrasounds, and CT scans), and pathological reports (e.g., biopsy analyses and tumor staging). Furthermore, the diversity of the images extends to factors such as shooting angles, creases in the reports, the presence of obstructions, handwritten elements (e.g., doctor signatures), and varying backgrounds in the photographs.

From this dataset, we manually selected 100 images that exhibit high diversity across these dimensions. Using GPT-4o (OpenAI, 2024a), the OCR ground truth was converted into JSON format based on a predefined medical schema (see Appendix A). The JSON outputs were then manually reviewed and corrected to produce the final JSON ground truth.

#### 3.3 Rule-based Reward Mechanism

We design a rule-based reward function to optimize the model’s ability to generate JSON outputs by measuring similarity with the ground truth. The reward computation consists of the following steps:

Format Score. We generally adopt the format of R1-Zero (Guo et al., 2025), which includes two components: think and answer . The format score rformat is 1 if both components meet the required specifications; otherwise, rformat is 0.

JSON Preprocessing. Parse the JSON object from the answer and flatten it into a non-nested key-value dictionary. Specifically, this involves traversing all leaf nodes in the JSON structure. Each leaf node’s key in the dictionary is formed by concatenating the keys along the path from the root to the leaf, and its corresponding value is the value of the leaf node. Given the model output yˆ and the ground truth y, the preprocessing step converts them into Sp and Sg, respectively.

Matching Score. The similarity between Sp and Sg is measured through a weighted combination of precision and recall. We define nmatched as the number of correctly matched key-value pairs between Sp and Sg. Accordingly, precision and recall are defined as nmatched|S

p| and nmatched|S

g| , respectively. Therefore, the matching score is defined as:

rmatching =

αnmatched|S

p| + (1 − α)nmatched|S

g| if |Sp| > 0, 0 else.

(5)

As shown in Figure 2, α serves as a critical hyperparameter to balance precision and recall during optimization:

- • When α equals 1, the reward function focuses solely on precision, allowing the model to achieve 100% precision by outputting just a single perfectly matched key-value pair.
- • when α equals 0, the reward function emphasizes recall, potentially causing the model to generate numerous hallucinated key-value pairs in an attempt to retrieve all key-value pairs from Sg.

Final Reward Score. The reward for the i-th sample is calculated as the sum of the format score and the matching score, expressed as:

ri = rformat + rmatching (6)

Comparison with SFT. Compared to SFT, our proposed reward function better accommodates the unordered nature of JSON data. Specifically, the unordered property of JSON allows a single image to correspond to multiple ground truths. SFT uses cross-entropy loss on fixed JSON ground truths during training, which may lead to data ambiguity and affect model performance.

| |= 0<br><br>= 1<br><br><br>| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

600

TrainResponseLength

500

400

300

200

100

0 5 11 17 22 28 34 39 45 51

- Figure 2: Impact of the hyperparameter α on response length when the Sampling Strategy is enabled. The semi-transparent and the solid lines indicate raw samples and the smoothed trend.

#### 3.4 Sampling Strategy

To assess the impact of query diversity on experimental outcomes, we employ two data construction strategies. The first strategy involves random sampling of keys corresponding to the JSON data of an image, thereby generating varied queries. The second strategy forgoes sampling, utilizing all keys, which results in all samples sharing a same query. Observations from Figure 3 indicate that key sampling leads to shorter response lengths, as the number of keys decreases post-sampling, consequently shortening response lengths. Additionally, from the reward curve, it can be observed that key sampling results in faster reward growth, which we attribute to the reduced number of keys making the training task simpler and thus accelerating reward acquisition.

## 4 Experiments

#### 4.1 VIE Metrics

Various metrics are used for evaluation, including the field-level precision, recall, F1 scores and treeedit-distance(TED) based accuracy as in (Kim et al., 2021). It is noted that TED based accuracy mainly reflects the correctness of trees’ topology. In practical VIE scenarios, we pay more attention to indicators such as F1 score, precision, and recall, which also reflect the correctness of extracted text information.

- • TED based accuracy measures the degree of match between the model’s output and the ground truth by calculating the edit distance between two tree structures, and the tree edit distance is used to quantify discrepancies between the predicted and actual structures.
- • Field-level precision is the proportion of correctly extracted fields among all predicted fields, defined as nmatched/|Sp| in Eq. 5.
- • Field-level recall is the proportion of correctly extracted fields among all actual fields, defined as nmatched/|Sg| in Eq. 5.
- • Field-level F1 score is the harmonic mean balancing precision and recall, measuring overall accuracy of field-specific extraction, respectively.

900

2.0

800

1.8

700

TrainResponseLength

600

1.6

TrainReward

500

1.4

400

1.2

300

1.0

200

w/ Sample

w/ Sample

0.8

w/o Sample

w/o Sample

100

0 7 15 23 30 38 46 53 61 69 Train Epoch

- Figure 3: Comparison of Reward and Response Length Trends Between Sampling and Non-Sampling Strategies During Training. The semi-transparent and the solid lines indicate raw samples and the smoothed trend.

#### 4.2 VIE Baselines

To compare our results with existing works, we introduce models with various types and different outputs.

Pipeline models. The pipeline models for OCR task are always composed of layout recognizer and OCR models for plain text, math functions and tables, and the OCR results are collected and rearranged into markdown, HTML or LaTeX format.

- • MinerU (Wang et al., 2024b) is a widely used pipeline model for OCR tasks, it uses LayoutLMv3 (Huang et al., 2022) or DocLayout-YOLO (Zhao et al., 2024) for document layout detection, an YOLO-v8 model 1 for math function detection, UniMERNet (Wang et al., 2024a) for math function recognition, RapidTable 2, TableMaster (Authors, 2020) or StructEqTable (Xia et al., 2024) for table recognition, PaddleOCR (Authors, 2020) for plain text OCR and LayOutReader (Pang, 2024) for reading order analysis. When evaluating MinerU, we align the version and parameter settings with those in OmniDocBench (Ouyang et al., 2025).
- • Marker (Wang et al., 2021b) integrates several open source models to parse document, and we align the version and parameter settings of Marker as in OmniDocBench.

Expert models. GOT-OCR (Wei et al., 2024) is a large multimodal model trained for document parsing, which firstly used a multi-stage training strategy to train an end-to-end OCR model.

General MLLMs. We include general purpose MLLMs such as GPT4o(OpenAI, 2024a), Qwen2.5VL-7B(Bai et al., 2025), Qwen2.5-VL-72B and InternVL-2.5-78B (Chen et al., 2025) as baselines. The usage and parameter settings of these models are aligned with those in OmniDocBench.

#### 4.3 Our VIE Models

In this subsection we first introduce our VIE models finetuned by SFT and RLVR methods, and we report the implementation details of training.

VIE SFT models. We show the result of models trained by VIE SFT and compare them with the VIE RLVR finetuned models, in order to analyze the benefits given by RLVR instead of SFT in VIE tasks.

- • JSON-SFT-100 model is finetuned with 100 high quality medical VIE data, model leans to extract key information by directly supervised finetuning.

- 1https://yolov8.com/
- 2https://github.com/RapidAI/RapidTable

- • OCR-SFT-17K model is finetuned with 17K medical report OCR data. The model parse images into markdown format, and we apply GPT4o to rearrange the markdown into JSON while evaluating its VIE performance. VIE RLVR models. Three VIE RLVR models are trained to evaluate our proposed method:
- • RL-100 is trained on 100 high quality data, during training the model extract information of randomly sampled fields. Note that the images of the 100 training dataset is same to JSON-SFT-100.
- • RL-100(w/o sample) is trained with similar schedule and 100 data, but during training the model is required to extract all the key information in the images.
- • OCR-SFT-17K-RL-100 model has the same RL stage with RL-100, and it is additionally supervised finetuned with 17K high quality OCR data composed of in-service medical reports and their manually corrected ground truths.

Our proposed method is implemented in pytorch. We use 32×H20 96GB GPUs to train our model with batch size 1 and the AdamW optimizer. In the reinforce learning stage, the learning rate starts from 1e-6 and decays to 0 following the liner schedule. During the rollout process, we sample 8 responses for each input prompt, with the KL divergence coefficient β set to 0.04.

#### 4.4 Comparisons on Medical VIE task

- Table 1: Performances on medical VIE task. The field-level precision, recall, F1 scores and TED based accuracies are reported. For each metric, we bold the best results and underline the second-best results. Note that all VIE SFT and VIE RLVR models are finetuned from Qwen2.5-VL-7B, and VIE RLVR shows the best precision, recall, F1 scores. # In the column named ’output’, ’OCR’ means we use models to parse medical reports into markdown format and then apply GPT4o to extract JSON format answers, ’JSON’ means we prompt models to directly output with JSON format.

Type Model Name #Output F1 Precision Recall TED Acc Pipeline MinerU OCR 63.16 67.85 59.07 64.01

Marker OCR 56.89 57.68 56.12 58.69 Expert GOT-OCR OCR 55.12 62.12 49.54 62.14 Commercial TextIn OCR 70.49 71.18 69.82 77.28 General GPT4o JSON 64.55 65.56 63.57 65.71

InternVL-2.5-78B JSON 71.15 71.49 70.82 74.42 Qwen2.5-VL-72B JSON 52.29 66.92 42.91 65.14 Qwen2.5-VL-7B JSON 67.61 65.18 70.23 54.54

VIE SFT JSON-SFT-100 JSON 66.69 63.91 69.71 55.21 OCR-SFT-17K OCR 66.94 63.25 71.08 57.08

VIE RLVR OCR-SFT-17K-RL-100 JSON 76.23 76.18 76.28 72.72 RL-100(w/o sample) JSON 77.10 77.67 76.55 69.64 RL-100 JSON 77.81 79.85 75.88 68.49

Our medical evaluation dataset consists of 203 medical report images uploaded by users, covering CT, MRI, X-ray, physical examination reports, endoscopy, prescriptions, urine tests, electrocardiograms, medical records, pathology, diagnostic tests, medicine boxes, blood tests, and ultrasound. These images include screenshots and scans, exhibiting diverse clarity levels and aspect ratios. To generate ground truth, under the guidance of doctors, we selected important fields from these medical images, used GPT-4o to extract values of these fields from the images, and manually corrected the answers. Finally, we obtained 203 image-JSON pairs as our test dataset.

The results of our models and baseline models are shown in Table 1. We find that VIE RLVR models outperform all other models on the medical VIE task. They achieve SOTA performance in F1, precision, and recall metrics. TextIn and InternVL-2.5-78B perform well on the TED accuracy metric. However,

their other scores are relatively low. This indicates that these models can correctly extract the structure of medical reports but fail to parse text information accurately. It is important to note that F1 score, precision, and recall are more critical than TED accuracy. TED accuracy only evaluates the tree topology extracted by the model, whereas F1 score, precision, and recall also assess the text information on each tree node. These metrics hold greater value in practical applications.

The substantial improvement of RLVR over SFT validates the effectiveness of our proposed reinforcement learning approach. Compared with SFT, RLVR is better suited to the unordered nature of JSON data, where a single image can correspond to multiple ground truths. In contrast, SFT uses cross-entropy loss on fixed JSON ground truths during training, which may cause data ambiguity and degrade model performance. In addition, the model gains the ability to think and plan during the reinforcement learning process. This ability allows it to understand image structures more accurately and extract key textual information with higher precision.

For different VIE RLVR models, we first note that RL-100 maintains a 77.81 F1 score. It outperforms pipeline, expert, closed-source, general multimodal models, and VIE SFT models by nearly 10 points. This reveals that in domain-specific VIE tasks, leveraging high-quality small-scale datasets through RLVR enables significant performance gains for models. Meanwhile, RL-100(w/o sample) achieves SOTA performance in the recall metric. The model attempts to extract as many fields as possible, but extracting redundant fields leads to a lower precision score. Furthermore, OCR-SFT-17K-RL-100 reaches a higher TED accuracy. This means the model learns the tree topology of medical reports during the supervised finetuning stage. To observe the OCR ability gained in the SFT stage, refer to Appendix B. Our model, trained on the OCR task with the medical report dataset, outperforms several pipeline and OCR expert models. Its overall score is close to the SOTA model with 72B parameters, demonstrating the high quality of our medical report dataset.

#### 4.5 The Impact of Think

- Table 2: Impact of model thinking on medical VIE task. The field-level precision, recall, F1 scores and TED based accuracies are reported. Note that both two models are finetuned from Qwen2.5-VL-7B, the only difference is the thinking process in training and inferring stages.

Model Name F1 Precision Recall TED Accuracy

RL-100 (w/o think) 75.90 75.68 76.13 70.06 RL-100 (w/ think) 77.81 79.85 75.88 68.49

To analyze the impact of model thinking, we compare the RL-100 model which is required to think while training and inferring, to the RL-100 model without thinking in training and inferring stages. Their performance on the medical VIE dataset are reported in Table 2. For the more considerable metrics, i.e. F1 scores that better reflects the model’s VIE ability, model with thinking outperforms the model without thinking. In the thinking process, model understands the image better and plans to extract key information, thus thinking progress is important for MLLMs in VIE tasks. For case studies, refer to Appendix C.

#### 4.6 Analysis on General VIE tasks

We also evaluate the medical VIE models trained with RLVR on general VIE tasks in order to demonstrate that VIE tasks exhibit strong domain-specific characteristics, VIE tasks in different domains vary significantly, making it challenging for models to acquire strong general VIE capabilities through training on specific-domain VIE tasks. There are four widerly used VIE benchmarks:

• CORD(Park et al., 2019): The Consolidated Receipt Dataset (CORD) serves as a public benchmark comprising 800 training, 100 validation, and 100 test receipt images. The textual content of these receipts is encoded in the Latin alphabet. The dataset features 30 unique fields, including menu name, quantity, total price, and others. Notably, the information exhibits complex structures, such as nested groups and hierarchical organizations, which bears a resemblance to our medical VIE dataset.

- Table 3: Performances on general VIE tasks. Various widely used benchmarks are chosen to evaluate the models general VIE performance. The field-level F1 scores and TED based accuracies are reported. For each metric, we bold the best results and underline the second-best results.

CORD FUNSD SROIE Ticket Model TED Acc F1 TED Acc F1 TED Acc F1 TED Acc F1

Qwen2.5-VL-7B 34.6 39.8 12.2 12.2 80.7 67.0 80.4 76.4 OCR-SFT-17K-RL-100 59.6 46.8 32.6 32.7 78.4 66.4 83.7 64.6 JSON-SFT-100 36.1 40.8 14.5 14.5 80.4 66.4 72.8 72.7 RL-100(w/o sample) 68.7 49.2 34.0 30.6 79.3 65.0 87.5 72.3 RL-100 45.7 48.5 32.5 29.4 78.2 61.7 80.4 60.8

- • FUNSD(Jaume et al., 2019): FUNSD is a dataset for form understanding in noisy scanned documents, with 199 real, fully annotated forms for tasks like text detection and layout analysis. There are four fields to extracted, named ’question’ ’answer’ ’header’ and ’others’, and each field correspond to a list of values, which are similar to our medical VIE dataset.
- • SROIE(Huang et al., 2019a): SROIE is the most widely adopted dataset, significantly advancing the field’s development. The dataset comprises scanned English printed receipts, with each image accompanied by comprehensive OCR annotations and values for four key text fields.
- • Ticket(Guo et al., 2019): This public benchmark dataset comprises 1,500 training and 400 test images of Chinese train tickets. It includes eight fields, such as ticket number, departure station, and train number. The information structure is straightforward, with each key guaranteed to appear only once and the location of each field fixed.

From Table 3 we find that on CORD dataset, the VIE RLVR models outperform the base model with nearly 10 points in F1 score and 34 points in TED accuracy, and on FUNSD dataset VIE RLVR models outperform the base model with nearly 20 points in F1 score and 22 points in TED accuracy. However the performance of VIE RLVR models are poor on Ticket and SROIE datasets. We note that images in CORD benchmark are receipts with 30 complex fields, having some sub-fields to extract, which is very similar to our medical VIE evaluation dataset, while the SROIE and Ticket datasets are composed of images with few information to extract. Additionally, the comparision of RL-100(w/o sample) with RL-100 in Table 3 is different from results in Table 1 , indicating a significant gap exists among different VIE tasks.

## 5 Conclusion

In this paper, we propose a novel method for training MLLMs for medical VIE tasks, leveraging the RLVR framework. Our method applies merely 100 high quality samples for training, with our proposed reward mechanism being able to reduce model hallucination as well as ensure models to capture all the key fields. Extensive experiments demonstrate the effectiveness of our VIE RLVR models. On medical report dataset, our models trained with merely 100 samples approach SOTA performance on the mainly considered metrics, such as F1, precision and recall, exceeding commercial models like GPT-4o for more than 10 points, showing the advantage of our proposed method. Results on different general VIE tasks also reveal the gap between different tasks. Studies considering different data sampling strategy and thinking process are also given, validating the high efficiency of our method.

## References

Wiam Adnan, Joel Tang, Yassine Bel Khayat Zouggari, Seif Edinne Laatiri, Laurent Lam, and Fabien Caspani.

2024. A layoutlmv3-based model for enhanced relation extraction in visually-rich documents. In International Conference on Document Analysis and Recognition, pages 160–174. Springer.

Srikar Appalaraju, Bhavan Jasani, Bhargava Urala Kota, Yusheng Xie, and R Manmatha. 2021. Docformer: Endto-end transformer for document understanding. In Proceedings of the IEEE/CVF international conference on computer vision, pages 993–1003.

PaddlePaddle Authors. 2020. Paddleocr, awesome multilingual ocr toolkits based on paddlepaddle. https: //github.com/PaddlePaddle/PaddleOCR.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, and 1 others. 2025. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923.

Haoyu Cao, Xin Li, Jiefeng Ma, Deqiang Jiang, Antai Guo, Yiqing Hu, Hao Liu, Yinsong Liu, and Bo Ren. 2022. Query-driven generative network for document information extraction in the wild. In Proceedings of the 30th ACM International Conference on Multimedia, pages 4261–4271.

Jingye Chen, Tengchao Lv, Lei Cui, Cha Zhang, and Furu Wei. 2022. Xdoc: Unified pre-training for cross-format document understanding. arXiv preprint arXiv:2210.02849.

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, Lixin Gu, Xuehui Wang, Qingyun Li, Yimin Ren, Zixuan Chen, Jiapeng Luo, Jiahao Wang, Tan Jiang, Bo Wang, and 23 others. 2025. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. Preprint, arXiv:2412.05271.

Chengqi Duan, Rongyao Fang, Yuqing Wang, Kun Wang, Linjiang Huang, Xingyu Zeng, Hongsheng Li, and Xihui Liu. 2025. Got-r1: Unleashing reasoning capability of mllm for visual generation with reinforcement learning. arXiv preprint arXiv:2505.17022.

Hao Feng, Shu Wei, Xiang Fei, Wei Shi, Yingdong Han, Lei Liao, Jinghui Lu, Binghong Wu, Qi Liu, Chunhui Lin, and 1 others. 2025. Dolphin: Document image parsing via heterogeneous anchor prompting. arXiv preprint arXiv:2505.14059.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

He Guo, Xiameng Qin, Jiaming Liu, Junyu Han, Jingtuo Liu, and Errui Ding. 2019. Eaten: Entity-aware attention for single shot visual text extraction. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 254–259. IEEE.

Teakgyu Hong, Donghyun Kim, Mingi Ji, Wonseok Hwang, Daehyun Nam, and Sungrae Park. 2022. Bros: A pre-trained language model focusing on text and layout for better key information extraction from documents. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 10767–10775.

Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. 2025. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749.

Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. 2022. Layoutlmv3: Pre-training for document ai with unified text and image masking. In Proceedings of the 30th ACM international conference on multimedia, pages 4083–4091.

Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shijian Lu, and C.V. Jawahar. 2019a. ICDAR 2019 competition on scanned receipt OCR and information extraction (SROIE). In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1516–1520.

Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shijian Lu, and CV Jawahar. 2019b. Icdar2019 competition on scanned receipt ocr and information extraction. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1516–1520. IEEE.

Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, and 1 others. 2024. Qwen2. 5-coder technical report. arXiv preprint arXiv:2409.12186.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander

Madry, Alex Beutel, Alex Carney, and 1 others. 2024. Openai o1 system card. arXiv preprint arXiv:2412.16720. Guillaume Jaume, Hazım Kemal Ekenel, and Jean-Philippe Thiran. 2019. FUNSD: A dataset for form understanding

in noisy scanned documents. arXiv preprint arXiv:1905.13538.

Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. 2022. Ocr-free document understanding transformer. In European Conference on Computer Vision, pages 498–517. Springer.

Geewook Kim, Teakgyu Hong, Moonbin Yim, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. 2021. Donut: Document understanding transformer without ocr. arXiv preprint arXiv:2111.15664, 7(15):2.

Jianfeng Kuang, Wei Hua, Dingkang Liang, Mingkun Yang, Deqiang Jiang, Bo Ren, and Xiang Bai. 2023. Visual information extraction in the wild: practical dataset and end-to-end solution. In International Conference on Document Analysis and Recognition, pages 36–53. Springer.

Yiming Li, Qiang Wei, Xinghan Chen, Jianfu Li, Cui Tao, and Hua Xu. 2024. Improving tabular data extraction in scanned laboratory reports using deep learning models. Journal of Biomedical Informatics, 159:104735.

Yuqi Liu, Bohao Peng, Zhisheng Zhong, Zihao Yue, Fanbin Lu, Bei Yu, and Jiaya Jia. 2025a. Seg-zero: Reasoningchain guided segmentation via cognitive reinforcement. arXiv preprint arXiv:2503.06520.

Zhaocheng Liu, Quan Tu, Wen Ye, Yu Xiao, Zhishou Zhang, Hengfu Cui, Yalun Zhu, Qiang Ju, Shizheng Li, and Jian Xie. 2025b. Exploring the inquiry-diagnosis relationship with advanced patient simulators. arXiv preprint arXiv:2501.09484.

Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. 2025c. Visual-rft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785.

Chuwei Luo, Changxu Cheng, Qi Zheng, and Cong Yao. 2023. Geolayoutlm: Geometric pre-training for visual information extraction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7092–7101.

Ming-Wei Ma, Xian-Shu Gao, Ze-Yu Zhang, Shi-Yu Shang, Ling Jin, Pei-Lin Liu, Feng Lv, Wei Ni, Yu-Chen Han, and Hui Zong. 2023. Extracting laboratory test information from paper-based reports. BMC Medical Informatics and Decision Making, 23(1):251.

- OpenAI. 2024a. GPT-4o system card. https://openai.com/index/gpt-4o-system-card/.
- OpenAI. 2024b. Openai o1 system card. https://openai.com/index/openai-o1-system-card/.

Linke Ouyang, Yuan Qu, Hongbin Zhou, Jiawei Zhu, Rui Zhang, Qunshu Lin, Bin Wang, Zhiyuan Zhao, Man Jiang, Xiaomeng Zhao, and 1 others. 2025. Omnidocbench: Benchmarking diverse pdf document parsing with comprehensive annotations. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24838–24848.

Hantian Pang. 2024. Faster LayoutReader based on LayoutLMv3. Seunghyun Park, Seung Shin, Bado Lee, Junyeop Lee, Jaeheung Surh, Minjoon Seo, and Hwalsuk Lee. 2019. Cord:

A consolidated receipt dataset for post-ocr parsing. In Document Intelligence Workshop at NeurIPS 2019.

Rafał Powalski, Łukasz Borchmann, Dawid Jurkiewicz, Tomasz Dwojak, Michał Pietruszka, and Gabriela Pałka. 2021. Going full-tilt boogie on document understanding with text-image-layout transformer. In Document Analysis and Recognition–ICDAR 2021: 16th International Conference, Lausanne, Switzerland, September 5–10, 2021, Proceedings, Part II 16, pages 732–747. Springer.

Jake Poznanski, Jon Borchardt, Jason Dunkelberger, Regan Huff, Daniel Lin, Aman Rangapur, Christopher Wilhelm, Kyle Lo, and Luca Soldaini. 2025. olmocr: Unlocking trillions of tokens in pdfs with vision language models. arXiv preprint arXiv:2502.18443.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, and 1 others. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, and 1 others. 2025. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, and 1 others. 2025. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599.

Jianqiang Wan, Sibo Song, Wenwen Yu, Yuliang Liu, Wenqing Cheng, Fei Huang, Xiang Bai, Cong Yao, and Zhibo Yang. 2024. Omniparser: A unified framework for text spotting key information extraction and table recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15641–15653.

Bin Wang, Zhuangcheng Gu, Guang Liang, Chao Xu, Bo Zhang, Botian Shi, and Conghui He. 2024a. Unimernet: A universal network for real-world mathematical expression recognition. Preprint, arXiv:2404.15254.

Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, and 1 others. 2024b. Mineru: An open-source solution for precise document content extraction. arXiv preprint arXiv:2409.18839.

Fei Wang, Yuewen Zheng, Jingyi Wu, Qing Li, Pengfei Li, and Luxia Zhang. Chatschema: development and validation of a pipeline for extracting structured nursing information with large multimodal models. Interdisciplinary Nursing Research, pages 10–1097.

Jiapeng Wang, Chongyu Liu, Lianwen Jin, Guozhi Tang, Jiaxin Zhang, Shuaitao Zhang, Qianying Wang, Yaqiang Wu, and Mingxiang Cai. 2021a. Towards robust visual information extraction in real world: new dataset and novel solution. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 2738–2745.

Pengfei Wang, Chengquan Zhang, Fei Qi, Shanshan Liu, Xiaoqiang Zhang, Pengyuan Lyu, Junyu Han, Jingtuo Liu, Errui Ding, and Guangming Shi. 2021b. Pgnet: Real-time arbitrarily-shaped text spotting with point gathering network. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 2782–2790.

Haoran Wei, Chenglong Liu, Jinyue Chen, Jia Wang, Lingyu Kong, Yanming Xu, Zheng Ge, Liang Zhao, Jianjian Sun, Yuang Peng, and 1 others. 2024. General ocr theory: Towards ocr-2.0 via a unified end-to-end model.

Renqiu Xia, Song Mao, Xiangchao Yan, Hongbin Zhou, Bo Zhang, Haoyang Peng, Jiahao Pi, Daocheng Fu, Wenjie Wu, Hancheng Ye, Shiyang Feng, Bin Wang, Chao Xu, Conghui He, Pinlong Cai, Min Dou, Botian Shi, Sheng Zhou, Yongwei Wang, and 4 others. 2024. Docgenome: An open large-scale scientific document benchmark for training and testing multi-modal large language models. Preprint, arXiv:2406.11633.

Guowei Xu, Peng Jin, Li Hao, Yibing Song, Lichao Sun, and Li Yuan. 2024. Llava-o1: Let vision language models reason step-by-step. arXiv preprint arXiv:2411.10440.

Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, and 1 others. 2020a. Layoutlmv2: Multi-modal pre-training for visually-rich document understanding. arXiv preprint arXiv:2012.14740.

Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. 2020b. Layoutlm: Pre-training of text and layout for document image understanding. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pages 1192–1200.

Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, and 1 others. 2025. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615.

Minghong Yao, Liansheng Zhuang, Houqiang Li, and Jiuchang Wei. 2024. Learning label dependencies for visual information extraction. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, pages 6615–6623.

Huaiyuan Ying, Shuo Zhang, Linyang Li, Zhejian Zhou, Yunfan Shao, Zhaoye Fei, Yichuan Ma, Jiawei Hong, Kuikun Liu, Ziyi Wang, and 1 others. 2024. Internlm-math: Open math large language models toward verifiable reasoning. arXiv preprint arXiv:2402.06332.

En Yu, Kangheng Lin, Liang Zhao, Jisheng Yin, Yana Wei, Yuang Peng, Haoran Wei, Jianjian Sun, Chunrui Han, Zheng Ge, and 1 others. 2025a. Perception-r1: Pioneering perception policy with reinforcement learning. arXiv preprint arXiv:2504.07954.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, and 1 others. 2025b. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476.

Wenwen Yu, Ning Lu, Xianbiao Qi, Ping Gong, and Rong Xiao. 2021. Pick: processing key information extraction from documents using improved graph learning-convolutional networks. In 2020 25th International conference on pattern recognition (ICPR), pages 4363–4370. IEEE.

Peng Zhang, Yunlu Xu, Zhanzhan Cheng, Shiliang Pu, Jing Lu, Liang Qiao, Yi Niu, and Fei Wu. 2020. Trie: end-to-end text reading and information extraction for document understanding. In Proceedings of the 28th ACM International Conference on Multimedia, pages 1413–1422.

Zhiyuan Zhao, Hengrui Kang, Bin Wang, and Conghui He. 2024. Doclayout-yolo: Enhancing document layout analysis through diverse synthetic data and global-to-local adaptive perception. Preprint, arXiv:2410.12628.

Lianchi Zheng, Xiaoming Liu, Zhihui Sun, and Yuxiang He. 2022. Improving medical ocr information extraction with integrated bert and layoutxlm models. In China Health Information Processing Conference, pages 164–173. Springer.

Hengguang Zhou, Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, and Cho-Jui Hsieh. 2025. R1-zero’s" aha moment" in visual reasoning on a 2b non-sft model, 2025. URL https://arxiv. org/abs/2503.05132.

## A Predefined Medical Schema

{

"Name": "", // Patient's name, output as empty if not available "Gender": "", // Patient's gender, output as empty if not available "Age": "", // Patient's age, output as empty if not available "Examination Time": "", // Time when the patient was examined, output as empty

if not available "Department": "", // Department where the patient was examined, output as empty if not available "Examination Name": "", // Name of the examination performed on the patient, fill in if listed separately, output as empty if not available, do not extract

from indicators or examination descriptions "Examination Site": "", // Site where the patient was examined, fill in if listed separately, output as empty if not available, do not extract from indicators or examination descriptions "Indicators": [ // Various indicators of the examination items for the patient,

generally displayed in a table {

"Item Name": "", "Result": "", "Unit": "", "Reference Range": "", "Abnormal Mark": "", "Detection Method": "", "Result Status": "", "Clinical Indication": "", "Critical Value": ""

}

], "Examination Description": "", // Description of the examination results, such

as imaging findings, ultrasound findings, specimen descriptions, gross findings, microscopic findings, pathological descriptions, etc., output as empty if not available "Diagnosis": "", // Preliminary/clinical/pathological diagnosis of the examination results; and identify which type of diagnosis it is. Output as empty if not available "Treatment Recommendations": "", // Treatment recommendations given in the report, fill in if listed separately, output as empty if not available, do not

extract from examination descriptions or diagnoses "Sample Collection Time": "", // Time when the examination sample was

collected, output as empty if not available "Others": "" // Information in the report that affects the interpretation of the medical report but is not included in the above fields, output as empty if

not available }

## B Analysis on General OCR tasks

Table 4: Performances on general OCR task. Results of different models on OmniDocBench(Ouyang et al., 2025) are reported, the scores are normalized edit distances. For each metric, we bold the best results and underline the second-best results.

|Models|Book Slides Financial Textbook Exam Magazine Academic Notes News<br><br>|Overall|
|---|---|---|
|MinerU|0.055 0.124 0.033 0.102 0.159 0.072 0.025 0.984 0.171<br><br>|0.206|
|Marker|0.074 0.34 0.089 0.319 0.452 0.153 0.059 0.651 0.192<br><br>|0.274<br><br>|
|Mathpix<br><br>|0.131 0.22 0.202 0.216 0.278 0.147 0.091 0.634 0.69|0.3<br><br>|
|GOT-OCR|0.111 0.222 0.067 0.132 0.204 0.198 0.179 0.388 0.771<br><br>|0.267|
|Nougat|0.734 0.958 1.000 0.820 0.930 0.83 0.214 0.991 0.871|0.806<br><br>|
|GPT4o<br><br>|0.157 0.163 0.348 0.187 0.281 0.173 0.146 0.607 0.751|0.316|
|Qwen2-VL-72B|0.096 0.061 0.047 0.149 0.195 0.071 0.085 0.168 0.676<br><br>|0.179|
|InternVL2-76B<br><br>|0.216 0.098 0.162 0.184 0.247 0.150 0.419 0.226 0.903<br><br>|0.3|
|Qwen2.5-VL-7B<br><br>|0.2225 0.5704 0.1094 0.2545 0.1897 0.3580 0.2419 0.4176 0.6975<br><br>|0.3591|
|OCR-SFT-17K<br><br>|0.1813 0.0878 0.0915 0.0943 0.1612 0.1504 0.1599 0.1778 0.6334<br><br>|0.1984|

In Tab 4 we compare our model OCR-SFT-17K trained for OCR task using our 17K medical report dataset, with other widely considered pipeline models, expert models, general MLLMs as in 3. On the overall metric our model outperform GPT4o and many other models, reached the second place and very close to the SOTA performance, which means the medical report dataset is high quality and high diversity.

### C Case study of model with and without thinking We give study of two cases to analyze the effect of model’s thinking progress, refering to Tab 5 and Tab 6.

[Figure 5]

| |RL-100 (w/ think)|RL-100 (w/o think)|
|---|---|---|
|Model Output<br><br>|<think> 从图片中可以看到， 这是一份肠镜检查报告。报 告 中 没 有 明 确 提 到 患 者 的 姓名、性别、年龄、检查时 间、科室、检查名称、检查 部位等信息。因此，这些信 息无法从图片中提取。报告 中主要描述了肠镜检查的过 程和结果，包括检查部位、 检查描述、诊断等信息。根 据这些信息，可以构建出一 个符合要求的JSON格式的输 出。 </think> <answer> “‘json { "姓名": "", "性别": "", "年龄": "", "检查时间": "", "科室": "", "检查名称": "", "检查部位": "", "指标": [], "检查描述": "静 脉麻醉，持续吸氧、心电监 护下肠镜检查：患者左侧卧 位，行肛门指诊，进指5cm， 未触及明确肿物，退指指套 无 染 血 。 肠 道 准 备 一 般 ， 中等黄色粪水残留肠腔，充 分吸引后观察。钩拉法循腔 插镜至回肠末段10cm，进镜 顺利，所见部分末段回肠黏 膜光整。回盲瓣呈唇型，阑 尾开口清晰。所见结肠粘膜 光滑，血管纹理清晰，无充 血、糜烂、溃疡及新生物。 直肠粘膜未见异常。可见轻 度内痔。", "诊断": "", "处理 建议": "", "样本采集时间": "", "其他": "" } “‘ </answer>|{ "姓名": "", "性别": "", "年 龄": "", "检查时间": "", "科 室": "", "检查名称": "", "检 查部位": "肠道", "指标": [], "检查描述": "钩拉法循腔插 镜至回肠末段10cm，进镜顺 利 ， 所 见 部 分 末 段 回 肠 黏 膜光整。回盲瓣呈唇型，阑 尾开口清晰。所见结肠粘膜 光滑，血管纹理清晰，无充 血、糜烂、溃疡及新生物。 直肠粘膜未见异常。可见轻 度内痔。", "诊断": "", "处理 建议": "", "样本采集时间": "", "其他": "肠道准备一般，中 等黄色粪水残留肠腔，充分 吸引后观察。" }|
|F1 Precision Recall TED Acc<br><br>|1.0 1.0 1.0 1.0|0.0 0.0 0.0 0.52<br><br>|

##### Table 5: Model performances of one case. Blue letters stands for good parts in model’s thinking and output contexts, and red letters stands for wrong contents given by model, revealing that thinking progress makes model do VIE tasks better.

[Figure 6]

| |RL-100 (w/ think)<br><br>|RL-100 (w/o think)|
|---|---|---|
|Model Output|<think> 从图片内容来看， 这是一份超声检查报告。报 告 中 没 有 明 确 提 到 患 者 的 姓 名 、 性 别 、 年 龄 、 检 查 时 间 、 科 室 、 检 查 名 称 、 检查部位、样本采集时间等 信息。因此，这些信息无法 从图片中提取。报告中主要 描述了超声检查的结果和诊 断。报告中提到的检查结果 包括右肾轻度积水、右输尿 管上段扩张（中下段梗阻可 能）和双肾结石。报告中没 有单独列出的检查名称、检 查部位、指标、检测方法、 结 果 状 态 、 临 床 提 示 、 是 否为危急值等信息。因此， 这些信息也无法从图片中提 取。 </think> <answer> { "姓 名": "", "性别": "", "年龄": "", "检查时间": "", "科室": "", "检查名称": "", "检查部 位": "", "指标": [], "检查描 述": "夜间急诊检查：右肾 大小、形态正常，皮质回声 均匀，集合系统分离6mm， 盏 内 可 见 数 枚 强 回 声 团 ， 最大为4×3mm，后方无明显 声影。输尿管上段扩张，中 下段因气体干扰显示不清。 左 肾 大 小 、 形 态 正 常 ， 皮 质回声均匀，集合系统未见 分离，盏内可见数枚强回声 团，最大为4×3mm，后方无 明显声影。输尿管无扩张。 膀 胱 少 量 充 盈 。 前 列 腺 形 态、大小正常，回声均匀， 内未见异常回声。CDFI：上 述器官未见异常血流信号。", "诊断": "右肾轻度积水；右 输尿管上段扩张（中下段梗 阻可能），建议复查双肾结 石", "处理建议": "", "样本采 集时间": "", "其他": "" } </answer>|{ "姓名": "", "性别": "", "年 龄": "", "检查时间": "夜间急 诊检查", "科室": "", "检查名 称": "", "检查部位": "", "指 标": [], "检查描述": "右肾 大小、形态正常，皮质回声 均匀，集合系统分离6mm， 盏 内 可 见 数 枚 强 回 声 团 ， 最大为4×3mm，后方无明显 声影。输尿管上段扩张，中 下段因气体干扰显示不清。 左 肾 大 小 、 形 态 正 常 ， 皮 质回声均匀，集合系统未见 分离，盏内可见数枚强回声 团，最大为4×3mm，后方无 明显声影。输尿管无扩张。 膀 胱 少 量 充 盈 。 前 列 腺 形 态、大小正常，回声均匀， 内未见异常回声。CDFI：上 述器官未见异常血流信号。", "诊断": "右肾轻度积水；右 输尿管上段扩张（中下段梗 阻可能），建议复查双肾结 石", "处理建议": "双肾结石", "样本采集时间": "", "其他": "检查结果未见异常血流信 号" }|
|F1 Precision Recall TED Acc<br><br>|0.4 0.5 0.33 0.96|0.0 0.0 0.0 0.87<br><br>|

##### Table 6: Model performances of one case. Blue letters stands for good parts in model’s thinking and output contexts, and red letters stands for wrong contents given by model, revealing that thinking progress makes model do VIE tasks better.

