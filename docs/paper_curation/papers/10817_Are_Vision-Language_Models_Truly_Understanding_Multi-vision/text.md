### Enhanced Vision-Language Models for Diverse Sensor Understanding: Cost-Efficient Optimization and Benchmarking

Sangyun Chung†, Youngjoon Yu†, Seyeon Kim, Youngchae Chee, and Yong Man Ro

## arXiv:2412.20750v2[cs.CV]1Aug2025

Abstract—Large-scale Vision-Language Models (VLMs) have achieved notable progress in aligning visual inputs with text. However, their ability to deeply understand the unique physical properties of non-RGB vision sensor images remains limited. In this paper, we revisit and analyze these limitations and introduce a novel, cost-efficient paradigm that significantly advances sensor image understanding—without requiring extensive training data or any modifications to the existing VLM architectures. Specifically, we propose Sensor-Aware Attributes Fine-Tuning (SAFT) with the Diverse Negative Attributes (DNA) optimization, which leverages minimal sensor-specific data to enable robust learning of non-RGB characteristics and overcome RGB-centric biases inherent in current VLMs. In addition, we present VSTDX—the first comprehensive, public benchmark designed to rigorously evaluate VLMs’ sensor-specific understanding across diverse and realistic scenarios. Through extensive experiments on VLMs and various sensor modalities, we validate that our method consistently delivers superior performance and generalization under resource-constrained and architecture-invariant settings. Our approach provides a practical advance towards scalable deployment of VLMs in increasingly sensor-diverse real-world environments.

Index Terms—Vision Language Models, Non-RGB Vision Sensor, Vision Sensor Understanding

I. INTRODUCTION

# L

ARGE-scale Vision-Language Models (VLMs) have made rapid advances, proving instrumental in tasks such

as visual dialogue [1]–[3], video analysis [4]–[7], and document understanding [8]. By efficiently processing sensory data and generating complex inferences—often rivaling or even surpassing human-level performance in reasoning, as in the case of models such as GPT-4o [9]—VLMs are quickly becoming core components of general-purpose intelligence systems. Their applications are expanding to real-world domains including autonomous vehicles [10]–[13], Internet of Things (IoT) devices [14]–[16], and robotics [17]–[21], where robust visual understanding is essential. Devices that connect to the real world often use vision sensors, making it essential for VLMs to understand these kinds of information. Vision sensors, such as thermal imaging, depth sensing, and X-ray detection, provide information that goes beyond human eyesight, enriching the understanding of real-world environments.

However, a fundamental bottleneck remains largely unresolved: existing VLMs are heavily biased toward conventional

S. Chung, Y. Yu, S. Kim, Y. Chee, and Y. M. Ro are with the Integrated Vision Language Lab., School of Electrical Engineering, Korea Advanced Institute of Science and Technology (KAIST), 291 Daehak-ro, Yuseong-gu, Daejeon, 34141, Republic of Korea (email: jelarum@kaist.ac.kr; greatday@kaist.ac.kr; seyeon.kim@kaist.ac.kr; litcoderr@kaist.ac.kr, ymro@kaist.ac.kr). Corresponding author: Y. M. Ro (fax: 82-42-350-5494).†Both authors are equally contributed to this manuscript.

[Figure 1]

Fig. 1: Examples of vision sensor-related questions and responses by recent VLM [22]. This figure illustrates the limitations of current vision-language models in reasoning about physical properties specific to non-RGB modalities. While the model correctly identifies the use of a thermal camera, it fails to understand the core principle of thermal imaging, incorrectly attributing brightness to reflected sunlight rather than emitted heat.

RGB imagery and remain deficient in understanding vision sensor data such as thermal, depth, and X-ray images. Vision sensors capture physical properties of the environment beyond the spectrum of human vision, rendering their data crucial for safety and perception in advanced real-world systems. While humans can intuitively interpret such sensory information by leveraging scientific and contextual knowledge, state-of-the-art VLMs frequently misinterpret fundamental cues (see Figure 1), leading to critical errors—e.g., mistaking heat patterns in thermal images for sunlight reflections—due to their RGBcentric training and reasoning.

Figure 1 illustrates two contrasting examples of human–VLM interaction with vision sensor data [22]. In the first case, the VLM correctly identifies the sensor type and interprets the image. In the second, it fails to answer a question requiring deeper understanding of thermal image characteristics, confusing brightness with sunlight reflection rather than heat emission. This highlights a key gap: humans can infer sensor-specific meaning by integrating physical knowledge, whereas VLMs often rely on superficial RGB-based priors.

This persistent limitation stems from two core challenges: (1) the inherent physical differences between RGB and sensor images, which complicate perceptual alignment, and (2) the acute scarcity of high-quality vision sensor datasets, which

limits training diversity. As a result, current VLMs tend to fall back on shallow, RGB-bounded reasoning, lacking true sensor-specific interpretability. These limitations constrain their deployment in safety-critical, sensor-rich environments.

To address this gap, we introduce two key innovations. First, we propose VS-TDX, the first comprehensive benchmark designed to evaluate and quantify VLMs’ sensor-specific reasoning and perception capabilities across multiple vision sensor modalities. Second, we present Diverse Negative Attributes (DNA) optimization, a novel, cost-efficient training strategy that instills deep sensor understanding without requiring model architecture modification or large-scale data collection. DNA optimization leverages diverse and challenging negative samples, such as incorrect or confusing answers, to guide learning in a resource-efficient manner.

We evaluate 10 state-of-the-art VLMs using the VS-TDX benchmark and find that most models exhibit significant deficiencies in sensor-specific reasoning. However, with DNA optimization, we observe substantial and consistent improvements in vision sensor understanding, even under data- and compute-constrained settings. Unlike previous approaches that rely on costly data augmentation or extensive model retraining [23]–[27], our method establishes a scalable and practical paradigm for extending VLM generalization to diverse modalities which is crucial for real-world, safety-critical deployments.

This paper makes the following key contributions:

- • Identification of a critical limitation in current VLMs. We systematically analyze the inability of state-of-the-art VLMs to interpret non-RGB vision sensor data—such as thermal, depth, and X-ray images—due to their inherent RGB-centric training and reasoning biases. We demonstrate that this limitation leads to shallow, sensor-agnostic inferences that hinder real-world deployment in sensorrich environments.
- • Introduction of the VS-TDX benchmark. We propose VS-TDX, the first comprehensive benchmark specifically designed to evaluate VLMs’ perception and reasoning capabilities across diverse vision sensor modalities. The benchmark includes six tasks—four for sensor perception and two for sensor understanding—enabling fine-grained, quantitative assessment of sensor-specific performance.
- • Proposal of DNA optimization for Sensor-Aware Attributes Fine-Tuning (SAFT). We introduce Diverse Negative Attributes (DNA) optimization, a novel training strategy that leverages challenging negative samples to instill deep sensor understanding. DNA optimization is architecture-invariant and data-efficient, requiring no modification to the original VLM and minimal sensorspecific data.
- • Extensive empirical validation across 10 state-of-the-art VLMs. We evaluate ten leading VLMs on the VS-TDX benchmark and reveal widespread deficiencies in sensor reasoning. Our experiments show that DNA optimization consistently improves sensor image understanding across all models, even under constrained data and compute settings.

• A scalable and practical paradigm for multimodal generalization. Unlike prior approaches that rely on costly retraining or large-scale data augmentation, our method offers a scalable, low-cost solution for extending VLMs to new sensor modalities—paving the way for robust, realworld deployment in safety-critical systems.

The remainder of this paper is organized as follows. Section 2 reviews related work on large-scale vision language models and evaluation benchmark for VLMs. Section 3 introduces the VS-TDX benchmark and its evaluation protocol. Section 4 presents enhancing vision sensor understanding with the DNA optimization method. Section 5 details the experimental setup and results. Finally, Section 6 concludes the paper and discusses future directions.

II. RELATED WORK

- A. Large-scale Vision Language Models

Recently, there has been growing interest in multimodal vision-language learning. Vision-Language Models (VLMs) such as LLAVA [28], [29], BLIP-2 [30], InternVL2 [22], VideoLLaMA2 [31], MiniCPMv2.5 [32], and Qwen2-VL [33] have demonstrated impressive performance across a wide range of downstream tasks. For example, InternVL2 [22] is an open-source multimodal large language model that narrows the gap between open-source and commercial systems by enhancing visual understanding, supporting dynamic highresolution processing, and improving bilingual dataset quality.

To capture richer contextual information, recent VLMs have also begun incorporating vision sensor inputs. ImageBind [34], for example, creates a joint embedding space across multiple sensory modalities, including depth and thermal data. PandaGPT [35] integrates multimodal encoders with large language models to enable instruction-following across modalities, allowing it to perform complex tasks.

However, despite these advancements, relatively little attention has been paid to whether VLMs truly understand the physical properties and semantics of vision sensor data.

- B. Evaluation Benchmark for VLMs

Numerous studies have proposed benchmarks based on existing vision-language datasets to assess the reliability and performance of VLMs. MME [36] includes 14 sub-tasks built from publicly available images with manually crafted annotations, evaluating recognition and perception capabilities through yes/no question answering. SEED-Benchmark [37] is designed to assess generative comprehension via humanannotated multiple-choice questions across 12 evaluation dimensions. Other notable benchmarks include MMMU [38], Q-Bench [39], Q-Bench+ [40], and MMBench [41] each targeting different aspects of multimodal understanding.

In contrast to these prior efforts, the proposed VS-TDX benchmark is specifically designed to rigorously evaluate a VLM’s ability to understand the physical characteristics and semantics of vision sensor data. It focuses not only on perception but also on deeper reasoning grounded in sensorspecific information—an area largely overlooked by existing benchmarks.

##### VS-TDX Benchmark

##### VS-TDX Benchmark Testing Flow

Multi-vision Perception

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Existence

Counting

Q: What is the arrangement of objects in the image?

Q: How many people are visible in the image?

|[Figure 8]|
|---|

|[Figure 9]|
|---|

- [A] There are four people visible.

- [B] There are two people visible.
- [C] There are seven people visible.
- [D] There is one people visible.

- [A] Books are stacked horizontally and vertically.

- [B] Shoes are placed neatly.
- [C] Dishes are stacked irregularly.
- [D] Plants are arranged symmetrically.

Q: What could be the likely reason for capturing this image?

Q: What could be the likely reason for capturing this image?

Q: What could be the likely reason for capturing this image?

[Figure 10]

[Figure 11]

[Figure 12]

- [A] To assess the environmental conditions of the area.
- [B] To study the feeding habits of livestock.
- [C] To evaluate the breeding patterns of the animal.
- [D] To monitor the health and temperature of animal.

- [A] To assess the environmental conditions of the area.
- [B] To study the feeding habits of livestock.
- [C] To evaluate the breeding patterns of the animal.
- [D] To monitor the health and temperature of animal.

General Description

Position

Q: What is the position of the animal relative to the human?

|[Figure 13]|
|---|

Q: What items are commonly seen inside these bags?

|[Figure 14]|
|---|

- [A] The animal is to the right of the human.

- [B] The animal is behind the human.
- [C] The animal is to the left of the human.
- [D] The animal is on top of the human.

- [A] Toys and books.
- [B] Food and drinks.
- [C] Clothing and shoes.
- [D] Electronic devices and cables.

[Figure 15]

Vision-Language Model

Multi-vision Understanding

Contextual Understanding

Sensory Understanding

Ground Truth: D

Model Answer : D, To monitor…

Q: What might be the reason for the vehicles lined up in the image?

Q: What could be the likely reason for capturing this image?

|[Figure 16]|
|---|

|[Figure 17]|
|---|

- [A] To assess the environmental conditions of the area.
- [B] To study the feeding habits of livestock.
- [C] To evaluate the breeding patterns of the animal.
- [D] To monitor the health and temperature of animal.

- [A] They are waiting for a traffic signal.

- [B] They are parked for the night.
- [C] They are part of a parade.
- [D] They are in a car wash.

Accuracy per sensor

- Fig. 2: Overview of the proposed VS-TDX benchmark and its evaluation flow. The VS-TDX benchmark is designed to assess the vision-language models’ ability to understand non-RGB visual modalities. It comprises four multi-vision perception tasks—Existence, Counting, Position, and General Description—and two multi-vision understanding tasks—Contextual Understanding and Sensory understanding. The right panel illustrates the testing flow, where a vision-language model is evaluated based on its ability to produce the correct answer for sensor-specific visual questions. Final performance is computed as accuracy per sensor modality.

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

- Fig. 3: VS-TDX benchmark data source distribution. The outer ring shows six core vision sensor tasks, with the inner ring indicating the sample count for each.

the model can consistently recognize and categorize visual elements across different contexts and sensor modalities.

In contrast, vision sensor understanding requires the model not only to perceive but also to reason about the visual input. This involves higher-order cognitive tasks such as inferring relationships between objects, predicting the intended use of sensors, and applying domain-specific sensor knowledge. This task assesses a VLM’s ability to integrate visual information with contextual understanding, enabling logical deductions that go beyond surface-level perception.

1) Vision Sensor Perception: Vision sensor perception is the foundational process by which large-scale VLMs analyze images captured by vision sensors, including thermal, depth, and X-ray images. This process involves recognizing and interpreting fundamental visual elements, informed by early cognitive models of how humans attend to and organize visual information [42], [43].

- • Existence: Identifying and listing common objects present in the image, such as people, vehicles, animals, etc. (e.g., “Are there any cars in this thermal image?”)
- • Counting: Determining the number of identified objects or entities. (e.g., “How many people are visible in this image?”)
- • Position: Understanding the spatial arrangement of objects, including their relative positions within the image. (e.g., “Where is the heat source located relative to the object in this image?”)
- • General Description: Generating nuanced, coherent descriptions of the overall scene, articulating what is happening and providing contextual information that enhances image comprehension. (e.g., “Describe the overall activity in this image.”)

III. THE VS-TDX BENCHMARK

- A. Vision Sensor Tasks: Perception and Understanding

The VS-TDX benchmark is constructed around two core tasks: vision sensor perception and vision sensor understanding. As illustrated in Figure 2, vision sensor perception evaluates a VLM’s ability to accurately interpret and identify objects, scenes, and relationships from sensor-derived visual inputs. This includes tasks such as object detection, image classification, scene recognition, and relationship detection—where the model must process and comprehend content from various types of vision sensor images. The goal is to ensure that

At this stage, VLMs focus on extracting essential information directly from raw sensor data. This foundational perception is

[Figure 24]

|[Figure 25]<br><br>|
|---|

[Figure 26]

[Figure 27]

###### Thermal Public Dataset Depth Public Dataset X-ray Public Dataset

###### Input Prompt

Large-scale Vision Language Model

Question: Why is the dog's head appearing brightest compared to its body?

+

Sensor Knowledge Task

- 1.Thermal Sensor
- 2.Depth Sensor
- 3.X-ray Sensor

- 1. Existence
- 2. Position
- 3. Counting
- 4. General Description
- 5. Contextual Understanding
- 6. Sensor Understanding

Positive Answer Set (Sensor-Matched Answer): The dog's head emits more heat due to higher blood flow.

Human Review

Negative Answer Set (Sensor-Mismatched Answer): The dog is facing a bright light source. The head is painted with a shiny color. The dog is standing in the sun. The dog's head is covered in reflective material.

Generated Question and Answer Set for VS-TDX benchmark

###### …

- Fig. 4: VS-TDX benchmark dataset generation pipeline. Public thermal, depth, and X-ray datasets are fed into a large-scale vision-language model, which, prompted by vision sensor knowledge and tasks, generates initial question-answer pairs. Human annotators then review and refine these pairs by creating positive and negative sets, classifying each into an evaluation dimension.

we filtered images based on six task categories (as shown in Figure 3), excluding low-resolution and sequentially captured images. Our final VS-TDX benchmark comprises approximately 6,791 sensor images, consisting of 1,867 thermal, 2,781 depth, and 2,143 X-ray images. It includes 10,160 diverse question-answer pairs, with an average of 1.50 pairs per image. These three modalities (thermal, depth, and X-ray) represent distinct physical properties and practical application domains, making them a representative initial set for evaluating and enhancing non-RGB sensor understanding.

critical for enabling more complex reasoning tasks, serving as the base upon which deeper understanding is built.

2) Vision Sensor Understanding: Vision sensor understanding is where VLMs demonstrate their advanced reasoning capabilities. Beyond simple recognition, this task requires models to engage in logical inference and contextual interpretation—distinguishing modern VLMs from traditional computer vision systems. This task includes two key components:

- • Contextual Understanding: The ability to apply fundamental knowledge and contextual cues to make informed judgments about a given scenario. This ensures that the model’s reasoning remains coherent with the visual and situational context provided by the image. (e.g., “Given this image of a house at night, why is the window significantly brighter?”)
- • Sensor Understanding: A more sophisticated capability that involves mapping 2D image data to the physical meanings associated with different vision sensors. This goes beyond processing raw pixels—it requires integrating sensor-specific knowledge (e.g., thermal radiation, depth cues, or X-ray attenuation) to interpret the image in a way that reflects real-world physical properties. Sensor understanding demands that VLMs move past naive recognition and engage with the underlying semantics of the sensor data. (e.g., “In this image, what does a sudden change from dark to light signify physically?”)

As illustrated in Figure 4, the benchmark construction begins with curating a detailed set of questions involving vision sensor inputs. These questions are designed to guide VLMs in interpreting image content beyond surface-level perception. To generate challenging and contextually rich question–answer pairs, we leverage ChatGPT/GPT-4o(version 2024-08-06) by using input prompts that include sensor-specific knowledge and task instructions. Each sensor type—thermal, depth, and X-ray—is associated with distinct physical properties, which are reflected in the generated questions. This approach enables the creation of multi-hop reasoning tasks that require deep understanding of sensor characteristics.

Every question–answer pair passes through a rigorous and thorough quality-control review process. A team of six human annotators, each with expertise in vision and language models, vision-sensor data interpretation, and dataset curation, independently reviews every draft item to verify factual accuracy, ensure descriptive clarity, and maintain overall quality standards. Additionally, twenty more well-educated native English speakers were recruited through the Prolific crowdsourcing survey platform [48] to evaluate and validate the quality of the benchmark. The finalized benchmark of question-answer pairs achieved an inter-rater agreement score of 0.95, high-

- B. Design and Construction of the VS-TDX Benchmark

The goal of our benchmark is to evaluate the vision sensor understanding capabilities of large-scale Vision-Language Models (VLMs). To ensure high-quality question generation,

- TABLE I: Evaluation results of various Vision-Language Models (VLMs) on the VS-TDX benchmark, with accuracy as the primary metric. ”Vision Sensor Perception” represents the average performance across four dimensions (Existence, Counting, Position, and General Description) for visual perception tasks. ”Vision Sensor Understanding” shows the average performance on two dimensions (Contextual Understanding and Sensor Understanding) for tasks evaluating sensor-specific comprehension. VLMs are sorted by release date in ascending order.

Contextual Understanding

Sensor Understanding

General Description

Vision Sensor Perception

Vision Sensor Understanding

Model Vision Sensors Existence Counting Position

Open Source Large-scale Vision-Language Models

Thermal 59.2 32.2 57.8 65.3 53.6 74.8 42.7 58.7 Depth 60.4 40.0 52.4 71.6 56.1 71.3 26.3 48.8 X-ray 65.2 55.0 58.6 81.7 65.1 75.8 59.3 67.5

BLIP-2 [30]

Thermal 60.7 27.6 65.6 60.7 53.7 74.4 41.1 57.8 Depth 73.6 22.1 61.0 77.6 58.6 73.0 22.1 47.5 X-ray 63.2 35.3 54.6 75.0 57.0 73.9 49.6 61.7

LLaVA-1.5-7B [44]

Thermal 66.7 47.7 70.3 73.0 64.4 74.8 50.4 60.6 Depth 71.2 40.5 67.2 77.6 64.1 68.8 28.7 48.7

InternVL2-8B [22]

- X-ray 69.5 39.8 64.9 82.8 64.3 75.6 65.0 70.3

VideoLLaMA2-7B [31]

Thermal 82.4 49.8 69.5 81.7 70.8 83.8 76.2 80.0 Depth 82.2 40.5 66.9 83.5 68.3 77.9 29.9 53.9

- X-ray 70.2 49.0 60.2 85.7 66.2 80.6 72.9 76.7

Thermal 76.1 52.8 72.7 77.8 69.8 80.9 59.8 70.4 Depth 76.8 43.7 71.6 84.7 69.2 77.4 51.3 64.3 X-ray 75.2 51.0 72.1 85.3 70.9 85.7 81.6 83.7

MiniCPM-V-2.5-8B [32]

Thermal 76.1 47.7 72.7 77.6 68.5 70.6 62.8 66.7 Depth 75.1 38.4 64.1 81.6 64.8 65.0 19.3 42.1 X-ray 71.0 39.8 63.8 84.4 64.7 76.0 64.4 70.2

Qwen2-VL-7B [33]

Thermal 71.1 46.3 75.0 72.7 66.3 77.4 50.6 64.0 Depth 67.8 36.3 68.1 76.6 62.2 66.9 29.6 48.2 X-ray 69.9 44.6 64.1 82.4 65.3 76.8 67.6 72.2

Phantom-7B [45]

Closed Source Large-scale Vision-Language Models

Thermal 81.8 57.3 79.7 80.7 74.9 84.5 68.7 76.6 Depth 82.1 38.4 73.7 86.6 70.2 78.2 32.5 55.3 X-ray 76.7 49.4 66.5 89.8 70.6 86.9 76.2 81.5

Gemini-Pro [46]

Thermal 79.3 55.3 78.9 84.4 74.5 90.6 69.7 80.2 Depth 84.9 45.8 73.2 90.2 73.5 85.0 33.6 59.3 X-ray 78.2 41.0 72.5 90.6 70.6 85.5 79.3 82.4

GPT-4o [9]

Thermal 75.3 46.2 64.1 67.8 63.3 65.4 64.4 64.9 Depth 63.3 30.5 52.3 73.0 54.8 53.8 44.5 49.1 X-ray 66.8 33.1 68.1 82.4 62.6 76.9 72.9 74.9

Claude-3.5-Sonnet [47]

lighting its outstanding reliability and alignment with human judgments (See also Appendix A). The dataset comprises two distinct sets: (1) A positive answer set, containing accurate, sensor-matched responses, and (2) A negative answer set, consisting of plausible but sensor-mismatched answers designed to test the model’s reasoning. The negative answers are specifically crafted to represent common VLM misconceptions or sensor-mismatched interpretations, ensuring they serve as effective distractors.

To ensure high-quality question-answer pairs, we designed the input prompts to emulate human chain-of-thought (CoT) reasoning. Specifically, before generating each question, the model is instructed to first strategize how to make the question challenging, then formulate the question and answer based on the given image and context. This step-by-step formulation encourages deliberate reasoning and leads to more contextaware and sensor-sensitive questions.

To support this process, we designed a set of input prompts that form the foundation of the VS-TDX benchmark. These prompts are crafted to be comprehensive and targeted, ensuring that the generated data effectively captures the vision sensor reasoning capabilities of VLMs. The input prompt components include:

• Sensor Knowledge: A detailed description of each sensor

type (thermal, depth, and X-ray) is provided to the model. This ensures the VLM has access to the physical principles and contextual applications unique to each sensor, guiding the generation of contextually relevant questions and answers.

- • Sensor Type: The prompt explicitly specifies the sensor type relevant to the task. This helps the model generate examples that are aligned with the characteristics of the corresponding sensor modality.
- • Question Types and Examples: The prompt includes the desired question type and representative examples for each of the six predefined tasks. This guides the model in generating questions with the correct structure and intent.
- • Negative Samples: The prompt also specifies the number of negative samples to be generated. These are designed to be plausible yet incorrect, encouraging the model to distinguish between correct reasoning and distractors. The prompt specifies the desired attributes of these negatives, enabling the generation of challenging counterfactuals.

A detailed description of the input prompt design used to generate the VS-TDX benchmark is provided in the Appendix.

#### Sensor-Aware Attributes Fine-Tuning

###### Reward Model (Reward Function 𝒓𝒓𝝓𝝓)

[Figure 28]

Question: Why does the deer appear brighter than the background?

Input 𝒙𝒙

Positive Reward

[Figure 29]

Negative Reward

[Figure 30]

Vision-Language Model

Learning

Cross Entropy

Loss

Sensor-Matched Answer: The deer emits more heat than its cooler surroundings.

Sensor-Matched Answer: 𝒚𝒚+

| | |
|---|---|
| | |

[Figure 31]

###### Margin 𝜷𝜷

Sensor-Mismatched Answers: 𝒀𝒀− Sensor-Mismatched Answer 𝒊𝒊: 𝒚𝒚𝒊𝒊−

Sensor-Mismatched Answer k: The deer have white fur...

Diverse Negative Attributes Optimization Loss

- Fig. 5: The proposed framework for Sensor-Aware Attributes Fine-Tuning. A Vision-Language Model takes an input image and question, generating responses. Fine-tuning occurs through two main mechanisms: a Cross-Entropy Loss for producing sensor-matched positive answers (y+), and a Reward Model that leverages positive and diverse sensor-mismatched negative answers (Y −) to provide feedback. This feedback, optimized through a Diverse Negative Attributes Optimization Loss, guides the model to learn sensor-aware preferences.

- C. Challenges in Vision Sensor Data with Current VLMs

included in the VS-TDX evaluation benchmark, ensuring an unbiased evaluation.

As shown in Table I, the VS-TDX benchmark reveals that vision sensor understanding remains a widespread challenge for current VLMs. A primary cause of this limitation is the scarcity of publicly available instruction-tuning datasets that incorporate vision sensor data. Without sufficient exposure to sensor-specific knowledge during training, VLMs often misinterpret image content derived from non-RGB modalities.

The overall framework for the proposed Sensor-Aware Attributes Fine-Tuning (SAFT) is illustrated in Fig 5. As depicted, our approach focuses on enhancing the VLM’s ability to discern sensor-specific attributes and provide accurate, contextually relevant answers. Our designed loss, Diverse Negative Attributes (DNA) optimization loss, is central to this fine-tuning process. As detailed in Fig 5, the fine-tuning of the Vision-Language Model is achieved through two primary mechanisms. Firstly, a standard Cross-Entropy Loss is applied to encourage the generation of sensor-matched correct answers (y+). Secondly, and critically, a reward model is employed to provide nuanced feedback. This model leverages both positive and a diverse set of k sensor-mismatched negative answers (Y − = {y1−,y2−,...,yk−}) to guide the learning. The feedback from the reward model is then used to optimize the VLM through the DNA optimization Loss, which explicitly trains the model to prioritize sensor-aware preferences and reject incorrect, sensor-agnostic responses.

This data limitation leads VLMs to rely heavily on RGBbiased reasoning, causing them to confuse or overlook the unique physical characteristics of vision sensor inputs such as thermal, depth, and X-ray imagery. These shortcomings highlight the need for more effective and efficient training strategies tailored to sensor-aware understanding, which our proposed DNA Optimization aims to address.

IV. PROPOSED METHOD FOR ENHANCING VISION SENSOR UNDERSTANDING

To address the challenges in vision sensor data with VLMs, we propose a cost-efficient approach that enables VLMs to learn sensor-specific reasoning even with limited data. To demonstrate its effectiveness, we design a method that achieves competitive performance using only a small subset of training data. Specifically, we construct a supplementary dataset of 600 vision sensor images—200 for each sensor type (thermal, depth, and X-ray). For each of these images, we generate a corresponding set of positive and diverse negative question-answer pairs using the same methodology as the VSTDX benchmark (Section III-B), resulting in 3,600 training samples. None of these images or question-answer pairs are

A. Sensor-Aware Attributes Fine-Tuning (SAFT)

1) Motivation: Through empirical analysis, we observed that Supervised Fine-Tuning (SFT) was largely ineffective in improving model performance for vision sensor understanding. SFT typically employs cross-entropy loss to increase the likelihood of words in positive (i.e., correct) answers. However, this approach relies solely on positive examples, limiting its ability to correct model misconceptions, especially when the model already assigns high probabilities to negative (incorrect or misleading) answers.

[Figure 32]

- (a) In Supervised Fine-Tuning (SFT), negative answer probabilities remain higher than positive ones.

[Figure 33]

- (b) In Sensor-aware Attributes Fine-Tuning (SAFT), which includes DNA Optimization, positive answer probabilities surpass and remain higher than negative ones after initial training, demonstrating effective discrimination.

- Fig. 6: Log probabilities of positive and negative answers during fine-tuning.

As shown in Figure 6a, SFT often increases the probabilities of both positive and negative answers simultaneously, preventing the model from learning to distinguish between them. This phenomenon highlights a key limitation: SFT lacks a mechanism to penalize incorrect reasoning, particularly when models rely on superficial RGB-based cues.

2) Diverse Negative Attributes (DNA) Optimization: To address this issue, we propose Diverse Negative Attributes (DNA) Optimization, a training strategy designed to explicitly reduce the model’s dependence on RGB-bounded reasoning. By learning from diverse negative examples, the model gains richer information about potential errors—learn not only what to do but also what not to do. This leads to more robust and accurate reasoning grounded in sensor-specific understanding.

DNA optimization trains the model to directly compare a positive answer against a set of negative answers, sharpening the contrast between their probability distributions. This approach is inspired by the Bradley-Terry model [49] (Eq. 2) and is conceptually aligned with Direct Preference Optimization (DPO) [50] method.

Let x denote the input context, y+ the positive answer, and

y− the negative answer. The comparison dataset is defined as N i=1, where N is the total number of samples. The reward function rϕ is defined as the average logprobability of generating answer y given context x, scaled by a weight factor α. Here, πθ(y|x) denotes the conditional probability of generating answer y given context x, parameterized by the model θ:

- D = {x(i),y+(i),y−(i)}

log πθ(y|x) |y|

rϕ(x,y) = α ·

(1)

where α is hyperparmeter that controls the influence of the logprobability term in the reward function, typically set to 1.0 by default to maintain the standard log-probability scaling. The probability that the model prefers y+ over y− is given by:

exp(r(x,y+)) exp(r(x,y+)) + exp(r(x,y−))

Pr(y+ ≻ y−|x) =

(2)

The objective function, based on Maximum Likelihood Estimation (MLE), is:

LR(rϕ, D) = −E(x,y+,y−)∼D log σ(rϕ(x, y+) − rϕ(x, y−)) (3)

where σ is the sigmoid function.

- 3) Leveraging Diverse Negatives: While a single question

typically has one correct answer, it can have multiple plausible but incorrect (negative) answers. These negatives inherently contain rich counterfactual information. To exploit this, we extend the objective to consider a set of k negative answers Y − = {y1−,y2−,...,yk−}, and average the reward differences:

LDNA(rϕ, D) = −E(x,y+,Y −)∼D

k

i=1

1 k

log σ(rϕ(x, y+) − rϕ(x, yi−))

= −EDEY − log σ(rϕ(x, y+) − rϕ(x, yi−))

(4)

- 4) Margin-Based Separation: As shown in Figure 6a, the

basic reward function may still fail to create a clear separation between positive and hard negative answers. To enforce a stronger distinction, we introduce a margin β, inspired by the Triplet Loss [51]. This margin ensures that the reward for the positive answer exceeds the average reward of the negative answers by at least β. The β value serves as a minimum desired difference between the positive and negative rewards, effectively pushing the decision boundary further apart for more robust classification. The modified objective becomes:

LDNA(rϕ, D) = −EDEY − log σ(rϕ(x, y+) − rϕ(x, yi−) − β)

(5)

log πθ(yi−|x) |yi−|

log πθ(y+|x) |y+|

= −EDEY − log σ α

− α

− β

To obtain the final SAFT loss, we combine the standard SFT loss with our proposed DNA loss:

LSAFT = LSFT + LDNA (6)

This formulation encourages the model to favor correct answers while actively discouraging reliance on misleading visual cues. As demonstrated in Figure 6b, DNA optimization leads to clearer decision boundaries and improved reasoning performance.

Importantly, our DNA optimization strategy is fundamentally modality-agnostic. Its core principles—learning from diverse negative attributes and enforcing distinct decision boundaries—can be universally applied to any sensor modality, provided that appropriately curated, modality-specific negative samples are available. Algorithm 1 shows pseudo code for Sensor-Aware Attributes Fine-Tuning (SAFT) algorithms.

Algorithm 1 Sensor-Aware Attributes Fine-Tuning (SAFT) Input:

# Training dataset with context, positive, and k negative answers

- D = {(xi,yi+,Yi−)}Ni=1

πθ # Pretrained VLM with parameters θ α # Scaling factor for reward β # Margin for separation k # Number of negative samples per instance η # Learning rate T # Number of training steps

###### Initialize:

θ ← pretrained parameters Define

rϕ(x,y) = α · log πθ(y | x)/|y| # Reward function σ(z) = 1+exp(1 −z) # Sigmoid function

for t = 1 to T do Sample minibatch B ⊆ D for (x,y+,Y − = {y1−,...,yk−}) ∈ B do

Compute positive reward: r+ ← rϕ(x,y+) Compute negative reward: ri− ← rϕ(x,yi−) Compute average preference loss with margin:

k

1 k

log σ(r+ − ri− − β) Compute supervised fine-tuning loss:

LDNA ← −

i=1

y∗ ∼ πθ(· | x) LSFT ← CrossEntropyLoss(y∗,y+) Combine losses:

LSAFT ← LSFT + LDNA Update θ via gradient descent:

θ ← θ − η · ∇θLSAFT end for

end for return Optimized model parameters θ

V. EXPERIMENT A. Experimental Setup

1) Dataset Collection: To construct the VS-TDX benchmark, we assembled a diverse and sensor-rich dataset designed to capture a wide range of real-world scenarios. The data were curated from 13 publicly available datasets spanning thermal, depth, and X-ray modalities (see Appendix for details). In total, the collection comprises approximately 7k images that reflect a broad spectrum of environments and object types. From these images, we generated roughly 10k unique evaluation questions to support comprehensive benchmarking (as seen in Section III).

The dataset includes:

• Thermal images covering a broad spectrum of scenarios, such as in-vehicle sensing, landscapes, people, animals, and thermal screening/scanning applications [52]–[59].

- • Depth images from both indoor and outdoor environments, featuring various objects in diverse settings [60]– [62].
- • X-ray images depicting human body parts and security inspection scenes, such as luggage scans from airport datasets [63], [64].

This comprehensive collection ensures robust representation of different vision sensors across a wide array of real-world contexts. The overall data distribution of the VS-TDX benchmark is illustrated in Figure 3.

For training purposes (for Sensor-Aware Attributes FineTuning), we selected 600 images (200 per sensor type) from the 13 datasets—ensuring these were not included in the VSTDX benchmark itself. From this training set, we generated 3,600 question-answer pairs.

We focused on six high-level reasoning task types to compile the source dataset: Existence, Counting, Position, General Description, Contextual Understanding, Sensor Understanding. More detailed information about the visual context of each dataset is provided in the Appendix A4.

2) Implementation Details: For our evaluation, we selected 10 state-of-the-art (SOTA) Vision-Language Models (VLMs) that represent the forefront of current research. These models were chosen to provide a comprehensive assessment of both open-source and closed-source VLMs across a variety of vision sensor tasks within the VS-TDX benchmark. Opensource models include BLIP-2 [30], LLAVA-v1.5-7B [28], InternVL2-8B [22], VideoLLaMA2-7B [31], MiniCPM-V-2.58B [32], Qwen2-VL-7B [33], and Phantom-7B [45]. The closed source models include GPT-4o(version 2024-08-06) [9], Claude 3.5 Sonnet(version 2024-03-07) [47], and GeminiPro(version 2024-09-24) [46]. All experiments were conducted with the temperature parameter set to 0 to ensure deterministic outputs. For DNA Optimization, we set the hyperparameters to α = 2, β = 0.2, and k = 3. Each VLM was trained using QLoRA [65] with the AdamW optimizer [66]. Specific training configuations were as follows: For Phantom-7B [45], the learning rate was 2e − 5 with one training epoch. LoRA configurations are rank = 256 and alpha = 256. For Qwen2VL-7B [33] and InternVL-8B [22], the learning rate was 2e−5 with one training epoch. LoRA configurations are rank = 64 and alpha = 64

B. Experiment Result

1) Evaluation on VS-TDX Benchmark: In this section, we present a comprehensive evaluation using the proposed VSTDX benchmark, a rigorous framework designed to assess the capabilities of VLMs across two core tasks: Vision Sensor Perception and Vision Sensor Understanding.

- • Vision Sensor Perception evaluates visual perception performance across four dimensions, providing an average score that reflects general perceptual capabilities.
- • Vision Sensor Understanding assesses the VLMs’ ability to reason about and interpret sensor-derived data, based on two dimensions-emphasize contextual and sensorspecific comprehension.

- TABLE II: Performance improvements achieved with the proposed Sensor-Aware Attributes Fine-Tuning (SAFT) with the Diverse Negative Attributes (DNA) optimization. Highlighted columns indicate the average performance for perception and understanding capabilities. The best results in each category are denoted in bold.

General Description

Vision Sensor Perception

Contextual Understanding

Sensor Understanding

Vision Sensor Understanding

Model Sensor Type Existence Counting Position

Thermal 71.1 46.3 75.0 72.7 66.3 77.4 50.6 64.0 Depth 67.8 36.3 68.1 76.6 62.2 66.9 29.6 48.2 X-ray 69.9 44.6 64.1 82.4 65.3 76.8 67.6 72.2

Phantom-7B

Thermal 82.8 46.2 73.4 81.7 71.0 79.3 78.5 78.9 Depth 71.0 48.4 71.7 84.7 69.0 77.1 65.3 71.2 X-ray 73.5 47.4 67.7 82.0 67.7 78.3 73.5 75.9

Phantom-7B

+ SFT

Thermal 86.8 49.8 75.8 86.4 74.3 82.9 86.4 84.6 Depth 79.1 49.0 74.5 87.9 72.6 81.2 86.1 83.7 X-ray 78.2 49.4 73.3 84.8 71.4 85.8 82.1 84.0

Phantom-7B

+ Ours

Thermal 76.1 47.7 72.7 77.6 68.5 70.6 62.8 66.7 Depth 75.1 38.4 64.1 81.6 64.8 65.0 19.3 42.1 X-ray 71.0 39.7 63.7 84.4 64.7 76.0 64.4 70.2

Qwen2-VL-7B

Thermal 85.7 50.8 80.5 82.6 74.9 85.8 80.6 83.2 Depth 83.0 44.2 73.3 89.0 72.4 75.6 30.6 53.1

Qwen2-VL-7B

+ SFT

- X-ray 78.2 43.8 70.5 89.8 70.6 84.4 84.2 84.3

Qwen2-VL-7B

+ Ours

Thermal 89.1 52.3 80.5 88.4 77.6 89.0 85.7 87.4 Depth 84.4 44.2 74.8 90.0 73.3 80.5 59.8 70.2

- X-ray 79.8 45.0 73.7 91.4 72.5 86.4 86.0 86.2

Thermal 66.7 47.7 70.3 73.0 64.4 74.8 50.4 60.6 Depth 71.2 40.5 67.2 77.6 64.1 68.8 28.7 48.7 X-ray 69.5 39.8 64.9 82.8 64.3 75.6 65.0 70.3

InternVL2-8B

Thermal 80.8 48.8 70.3 78.7 69.6 77.0 69.4 73.2 Depth 72.0 41.1 69.8 81.9 66.2 72.1 49.7 60.9 X-ray 72.8 46.2 67.3 84.8 67.8 78.6 73.7 76.1

InternVL2-8B

+ SFT

Thermal 84.0 50.3 75.0 84.5 73.4 82.9 82.0 82.4 Depth 74.1 42.6 71.9 84.7 68.3 77.3 77.8 77.6 X-ray 75.2 47.0 70.9 85.7 69.7 83.0 78.1 80.6

InternVL2-8B

+ Ours

As shown in Table I, performance varies significantly depending on the type of vision sensor used to capture the input images. While VLMs generally achieve moderate results on perception tasks, their performance diverges more noticeably in tasks requiring contextual and sensor-specific understanding.

Sensor Understanding is particularly challenging, as it requires models not only to recognize and describe visual content but also to grasp the underlying physical principles of the sensing modality. For instance: Interpreting thermal imagery involves understanding heat signatures and temperature gradients. Analyzing depth data demands spatial reasoning that extends beyond simple 2D interpretation.

Our experimental findings underscore the current limitations of VLMs in translating sensor data into meaningful, physically grounded interpretations.

To further validate the benchmark’s alignment with humanlevel reasoning, we conducted a human evaluation study using participants recruited via the Prolific crowdsourcing platform [48]. A total of 20 native English-speaking participants with at least a bachelor’s degree and high approval ratings completed 45 sampled questions across thermal, depth, and X-ray modalities. Participants achieved an average accuracy of 95.1%, indicating strong alignment between the benchmark and human judgment. In contrast, the best-performing VLM (GPT-4o) scored 73.3%, revealing a 21.8% performance gap and emphasizing the challenges VLMs face in achieving human-level understanding in sensor-based reasoning tasks. Further details on the human evaluation are provided in the

Appendix.

2) Evaluation on the Effects of DNA Optimization: As shown in Table II, we validate the effectiveness of the proposed Diverse Negative Attributes (DNA) optimization in enhancing vision sensor understanding performance across VLMs. We applied the Sensor-Aware Attributes Fine-Tuning, which includes DNA Optimization, to three representative models—Phantom-7B [45], Qwen2-VL-7B [33], InternVL28B [22]—to assess its generalizability and impact.

As introduced earlier, DNA optimization is both flexible and architecture-agnostic, enabling seamless integration into various VLMs without requiring modifications to their underlying network structures. Compared to standard supervised finetuning (SFT), DNA optimization consistently yields superior performance across nearly all evaluation metrics.

Notably, in the Vision Sensor Understanding task, DNA optimization delivers an average performance improvement of 8.89p% over SFT-trained models across all three sensor modalities. These results demonstrate that DNA optimization significantly boosts the ability of VLMs to comprehend sensorspecific characteristics, making it especially effective for tasks that require deeper reasoning about the physical properties embedded in sensor data.

C. Qualitative Analysis

Figure 7 presents qualitative visualization examples comparing the responses of Phantom-7B [45] under three different training strategies: (1) base model, (2) supervised fine-tuning (SFT), and (3) sensor-aware attributes fine-tuning

[Figure 34]

[Figure 35]

[Figure 36]

Why is the woman’s hair color colorful? What causes the table to appear darker than the sofa?

Why do the objects inside the box appear in different colors? Explain what orange and blue colors mean.

User

###### Orange and blue colors represent warmth, playfulness, and creativity, making the picture enjoyable to look at.

The table appears darker because it is positioned between the bright source of light coming from the window and an object

The woman’s hair can appear colorful because of the way light interacts with it

Base Model

The table appears darker than the sofa due to the presence of the potted plant in close proximity to the sofa, causing the shadows on the table.

The objects inside the box appear in different col ors to make a picture in the window like a painting …

Base Model +SFT

The image depicts the woman using a hair dye product

The objects inside the box are visually separated with different colors to suggest textur e and variety of materials. The orange color might indicate the presence of paper or fabric, while the blue could represent metal or plastic

The table is darker than the sofa because it is positioned closer to the camera perspective than the sofa

Base Model +Ours

The heat of the iron is causing her hair to change colors due to the way thermography works.

components …

- Fig. 7: Qualitative visualization examples comparing different Phantom-7B model variants (Base Model, SFT, and SAFT(Ours)) across thermal, depth, and X-ray modalities. The proposed DNA optimization significantly enhances the model’s sensor-aware reasoning capabilities, enabling appropriate interpretations aligned with sensor-specific characteristics. This is in contrast to the inaccurate or less precise responses generated by the base and Supervised Fine-Tuning (SFT) models.

(SAFT). These comparisons span three vision sensor modalities—thermal, depth, and X-ray.

As illustrated, the choice of training strategy significantly influences the model’s ability to interpret sensor-specific visual information and generate accurate, context-aware responses. For instance, in the thermal image example, the model is asked: “Why is a woman’s hair color colorful?”

- • The base model produces a generic response: “A woman’s hair can appear colorful because of the way light interacts with it,” which fails to reflect any understanding of thermal imaging.
- • The SFT-trained model, despite being exposed to sensor data, also lacks thermal-specific reasoning, responding with: “The image depicts a woman using a hair dye product.”
- • In contrast, the SAFT-trained model demonstrates a deeper understanding of both the image content and the underlying sensor characteristics, responding: “The heat of the iron is causing her hair to change colors due to the way thermography works.”

This response reflects a more advanced level of multimodal reasoning. It not only captures the temperature-based color variation inherent in thermal imagery but also connects the observed human action (using a heated iron) to the resulting thermal pattern—demonstrating an understanding grounded in

the physical principles of the sensor.

These findings highlight the effectiveness of our SAFT in enhancing a model’s ability to interpret and reason about sensor-specific visual data. Importantly, this improvement is not limited to thermal imagery but is consistently observed across depth and X-ray modalities as well.

D. Ablation on the Number of Negative Sample

Table III presents an ablation study evaluating the impact of varying the number of negative samples (denoted as k) on vision sensor understanding performance. The baseline model used for this analysis is Phantom-7B [45] fine-tuned with SAFT.

The results demonstrate that increasing the number of negative samples generally leads to improved performance in both vision sensor perception and understanding tasks—particularly in the areas of Contextual Understanding and Sensor Understanding. This trend suggests that a larger pool of negative samples helps VLMs more effectively distinguish relevant features within sensor-specific contexts—by drawing closer to meaningful cues while pushing away irrelevant or misleading ones—thereby enhancing their reasoning capabilities.

Overall, the findings confirm that incorporating a diverse set of negative samples not only boosts task performance but also

TABLE III: Ablation study: VLM performance versus the number of negative samples (k).

General Description

Vision Sensor Perception

Contextual Understanding

Sensor Understanding

Vision Sensor Understanding

Negative Sample k Sensor Type Existence Counting Position

- k = 1

Thermal 86.6 48.8 74.2 83.3 73.2 81.2 85.0 83.1 Depth 79.9 48.4 73.5 87.6 72.3 80.8 80.2 80.5 X-ray 76.8 48.2 69.7 82.8 69.4 84.2 80.5 82.3

- k = 2

Thermal 86.9 49.3 73.4 84.2 73.4 81.6 86.4 84.0 Depth 76.6 46.8 74.9 88.5 71.7 81.0 84.4 82.7 X-ray 76.0 49.4 71.3 81.6 69.6 84.8 81.9 83.3

- k = 3

Thermal 86.8 49.8 75.8 86.4 74.3 82.9 86.4 84.6 Depth 79.1 49.0 74.5 87.9 72.6 81.2 86.1 83.7 X-ray 78.2 49.4 73.3 84.8 71.4 85.8 82.1 84.0

TABLE IV: Ablation study: VLM performance versus the number of training images per sensor (n)

Contextual Understanding

Sensor Understanding

Number of Training Images per Sensor n

General Description

Vision Sensor Perception

Vision Sensor Understanding

Model

Sensor Type Existence Counting Position

Thermal 71.1 46.3 75.0 72.7 66.3 77.4 50.6 64.0 Depth 67.8 36.3 68.1 76.6 62.2 66.9 29.6 48.2 X-ray 69.9 44.6 64.1 82.4 65.3 76.8 67.6 72.2

Phantom-7B -

Thermal 75.7 46.8 73.4 74.9 67.7 77.8 56.6 67.1 Depth 69.5 43.7 69.5 81.2 66.0 72.9 43.7 58.3 X-ray 71.5 45.0 66.7 82.4 66.4 77.7 69.3 73.5

Phantom-7B

n = 50

+ SFT

Thermal 85.7 46.8 75.0 81.4 72.2 81.6 77.5 79.5 Depth 73.7 47.4 72.0 85.2 69.6 78.0 66.6 72.3 X-ray 73.4 49.0 70.1 83.2 68.9 84.2 81.6 82.9

Phantom-7B

n = 50

+ Ours

Thermal 80.9 46.3 73.4 78.5 69.8 79.3 70.1 74.7 Depth 71.2 43.7 68.0 82.3 66.3 72.9 50.0 61.4 X-ray 71.7 46.6 67.3 82.0 66.9 78.8 70.7 74.8

Phantom-7B

n = 100

+ SFT

Thermal 86.5 48.2 75.8 83.0 73.4 82.2 83.8 83.0 Depth 76.8 48.4 73.6 86.2 71.3 80.0 69.9 75.0 X-ray 76.2 49.0 71.3 83.6 70.0 84.7 81.9 83.3

Phantom-7B

n = 100

+ Ours

promotes a deeper and more nuanced understanding of sensor data across modalities.

- E. Ablation on the Number of Training Images

Table IV presents an ablation study examining how vision sensor understanding performance varies with the number of training images (n). The analysis explores how different quantities of training data per sensor affect the performance of both Supervised Fine-Tuning (SFT) and Sensor-Aware Attributes Fine-Tuning (SAFT).

The results reveal that under extremely limited training data conditions, SFT yields only marginal improvements. In contrast, SAFT achieves substantial performance gains, even with minimal data. This suggests that DNA optimization is particularly effective in low-resource scenarios, where its ability to leverage diverse negative samples compensates for the lack of abundant training images.

Overall, the findings highlight the robustness and data efficiency of DNA optimization, especially when training data is scarce.

- F. Generalization Capability of DNA Optimization on RGB Images

- TABLE V: Performance comparison of Phantom-7B base model versus Sensor-Aware Attributes Fine-Tuning (SAFT) across various benchmarks.

Method MME [36] MMBench [41] MMMU [38] SEEDI [37] - 2126 79.8 47.8 75.3 Ours 2113 80.2 49.3 75.7

To assess whether our method compromises performance on standard RGB images, we evaluated the same model

(Phantom-7B) used in the main experiments—fine-tuned with SAFT using 200 images per sensor—on widely used benchmarks for general VLM tasks (as shown in TABLE V).

The results reveal negligible performance differences compared to the original model, clearly indicating that our optimization strategy does not lead to overfitting. Instead, DNA optimization effectively suppresses incorrect, sensor-biased reasoning while preserving the model’s general understanding capabilities.

G. Comparison of DNA with Other Optimization Methods

Table VI presents a comparative analysis of various optimization methods—DPO [50], IPO [67], SimPO [68]—against the proposed DNA optimization method. Prior approaches primarily focus on preference alignment: DPO through direct response comparisons, IPO by leveraging implicit signals without explicit reward supervision, and SimPO by incorporating simplex-based constraints to enable more efficient optimization.

In contrast, DNA optimization distinguishes itself by effectively leveraging a larger number of diverse negative samples, even from limited datasets. This strategy provides richer learning signals and enables models to better differentiate between relevant and irrelevant features.

As a result, DNA consistently outperforms prior methods across the VS-TDX benchmark. Its advantage becomes especially pronounced in low-resource settings—for example, when the number of training images per sensor is extremely limited (e.g., n = 50). In such scenarios, DNA’s ability to extract more informative gradients from sparse data proves particularly beneficial, leading to superior performance in vision sensor understanding tasks.

- TABLE VI: Comparison of performance between the proposed DNA optimization and other optimization methods. The best results are denoted in bold.

Number of Training Images per Sensor n

General Description

Vision Sensor Perception

Contextual Understanding

Sensor Understanding

Vision Sensor Understanding

Model

Sensor Type Existence Counting Position

Thermal 82.8 46.2 73.4 81.7 71.0 79.3 78.5 78.9 Depth 71.0 48.4 71.7 84.7 69.0 77.1 65.3 71.2 X-ray 73.5 47.4 67.7 82.0 67.7 78.3 73.5 75.9

Phantom-7B

n = 200

+ SFT

Thermal 83.8 48.2 74.2 82.0 72.1 79.6 81.4 80.5 Depth 72.8 48.4 71.9 84.0 69.3 76.2 74.0 75.1 X-ray 75.0 48.6 69.3 81.2 68.5 82.1 78.0 80.0

Phantom-7B

n = 200

+ SFT + DPO [50]

Thermal 84.4 48.7 72.7 81.3 71.8 80.6 85.1 82.8 Depth 72.8 46.3 72.3 84.5 69.0 78.5 74.9 76.7 X-ray 74.8 48.2 70.5 82.4 69.0 82.0 77.4 79.7

Phantom-7B

n = 200

+ SFT + IPO [67]

Thermal 84.5 46.2 75.0 84.0 72.4 79.9 83.2 81.6 Depth 71.4 47.9 69.8 83.8 68.2 76.3 72.4 74.3 X-ray 75.7 48.2 71.3 81.6 69.2 81.0 79.5 80.3

Phantom-7B

n = 200

+ SFT + SimPO [68]

Thermal 86.8 49.8 75.8 86.4 74.3 82.9 86.4 84.6 Depth 79.1 49.0 74.5 87.9 72.6 81.2 86.1 83.7 X-ray 78.2 49.4 73.3 84.8 71.4 85.8 82.1 84.0

Phantom-7B

n = 200

+ Ours

Thermal 75.7 46.8 73.4 74.9 67.7 77.8 56.6 67.1 Depth 69.5 43.7 69.5 81.2 66.0 72.9 43.7 58.3 X-ray 71.5 45.0 66.7 82.4 66.4 77.7 69.3 73.5

Phantom-7B

n = 50

+ SFT

Thermal 79.7 46.2 71.9 76.9 68.7 78.0 63.4 70.7 Depth 68.9 43.2 67.6 78.5 64.5 71.8 51.2 61.5 X-ray 70.5 43.4 65.7 82.8 65.6 78.4 67.8 73.1

Phantom-7B

n = 50

+ SFT + DPO

Thermal 79.5 48.2 71.1 77.8 69.1 78.6 61.6 70.1 Depth 70.9 46.3 70.4 81.9 67.4 73.5 50.8 62.2 X-ray 71.6 44.2 66.1 82.0 66.0 78.7 68.2 73.5

Phantom-7B

n = 50

+ SFT + IPO

Thermal 78.4 47.2 68.0 74.8 67.1 78.3 64.4 71.4 Depth 71.2 45.3 68.2 81.6 66.6 74.0 48.8 61.4 X-ray 70.7 44.6 65.3 81.2 65.4 78.1 69.1 73.6

Phantom-7B

n = 50

+ SFT + SimPO

Thermal 85.7 46.8 75.0 81.4 72.2 81.6 77.5 79.5 Depth 73.7 47.4 72.0 85.2 69.6 78.0 66.6 72.3 X-ray 73.4 49.0 70.1 83.2 68.9 84.2 81.6 82.9

Phantom-7B

n = 50

+ Ours

H. Real-World Scenario Test

To further assess the robustness and practical applicability of our SAFT-optimized VLM beyond curated datasets and synthetic benchmarks, we conducted an evaluation using real thermal images captured by a different commercial-grade thermal sensor (TE-SQ1 [69])—distinct from the sensor type used during training. This experimental setup simulates a realistic deployment scenario where models must interpret sensor data originating from previously unseen devices.

As illustrated in Figure 8, baseline VLMs frequently revert to RGB-trained priors when processing unfamiliar thermal imagery, often resulting in misinterpretations—for example, mistaking thermal gradients for lighting effects. In contrast, our model, fine-tuned via Sensor-Aware Attributes FineTuning (SAFT), accurately identifies temperature-related cues and provides physically grounded interpretations (e.g., “the bright region indicates high surface temperature due to heat emission”).

These findings suggest that SAFT enhances the model’s ability to handle variations in thermal sensor types, even when such types were not encountered during training. While not explicitly designed for out-of-distribution generalization, the model demonstrates encouraging robustness, highlighting SAFT’s potential for reliable deployment in real-world environments with diverse sensor hardware.

VI. CONCLUSION

In this study, we focus on assessing and improving the ability of large-scale Vision-Language Models (VLMs) to understand and process vision sensor inputs. As VLMs are increasingly deployed in real-world applications, their ability

to accurately interpret and reason about data from diverse vision sensors has become crucial. To address this:

- 1) We propose a new evaluation benchmark called VS-TDX, which generates samples aimed at specific physical sensor understanding.
- 2) We also propose Sensor-Aware Attributes Fine-Tuning (SAFT) with the novel Diverse Negative Attribute (DNA) optimization to improve vision sensor understanding ability. Through extensive experiments, we evaluate the capability of state-of-the-art VLMs to understand sensor-derived knowledge from various vision inputs. Our results demonstrate that the proposed DNA optimization strategy significantly enhances the performance of VLMs in interpreting vision sensor data. We argue that combining a sensor-knowledge-annotated evaluation benchmark with a tailored, cost-efficient optimization framework lays a strong foundation for future applications of vision sensors in VLMs.

Although our current benchmark and optimization efforts focus on thermal, depth, and X-ray modalities, the proposed framework is inherently extensible. Its modality-agnostic design enables adaptation to a broader spectrum of sensor types, paving the way toward truly sensor-agnostic VLMs. A major challenge in this domain remains the scarcity of highquality, comprehensively annotated datasets across diverse sensor modalities. Our approach addresses this limitation by targeting a representative yet tractable subset of modalities, thereby demonstrating the effectiveness of our methods while establishing a scalable foundation for future expansion.

While this work represents a significant step forward in multi-modal VLM understanding, our current scope is limited to three non-RGB modalities. Achieving full sensoragnosticism will require incorporating a wider array of sensory

[Figure 37]

[Figure 38]

Why do blue or purple areas appear in the central region? And what happened in that area?

When the cup was moved, a black mark remained. Why does this happen?

User

When you move the cup, a black mark is left behind because of the process called adhesion. In many inks, tiny particles or molecules stick to surfaces

In the central region where blue or purple areas appear, it indicates areas of the road that are shaded by trees or other obstacles.

Base Model

Base Model +Ours

These colors indicate cooler spots where vehicles were recently removed, possibly due to moving or parking.

Due to the heat conductivity of the table, the temperature remains at the previous location where the cup was placed.

- Fig. 8: Real-world thermal images captured by the thermal camera, evaluated using different Phantom-7B variants (Base Model and SAFT (Ours)). Our SAFT-trained model accurately interprets thermal-specific cues—such as residual heat from removed vehicles and object-induced temperature patterns—unlike baseline VLMs constrained by RGB assumptions, demonstrating robust sensor-aware reasoning in real-world scenarios.

inputs. As part of future work, we plan to expand the VSTDX benchmark to include additional modalities such as LiDAR point clouds, hyperspectral imagery, and synthetic aperture radar (SAR). We will also explore the scalability and generalizability of DNA optimization across these new domains.

43,470 video frames captured in diverse scenarios. Animal detection(AnimalDet) dataset [57] consists of thermal images of eight animal species—deer, bear, cow, dog, elephant, fox, goat, and wild boar. The average image size is 369×363 pixels. The Chips Thermal Face Dataset [58] comprises over 1,200 thermal face images of male and female subjects aged 18–23 from three continents. It supports research in advanced thermal facial classification and recognition systems using deep learning techniques. IFSOD dataset [59] contains thermal sensor images of various objects, including bicycles, birds, dogs, and humans, with an average resolution of 640×480 pixels. A Dense Indoor and Outdoor DEpth Dataset(DIODE) [60] provides high-resolution color images paired with precise, dense, long-range depth measurements. It is the first publicly available RGBD dataset featuring both indoor and outdoor scenes captured using a single sensor suite. The NYU-Depth V2 dataset [61] includes video sequences of indoor environments captured with the RGB and depth cameras of Microsoft Kinect. It contains 1,449 densely labeled pairs of aligned RGB and depth images. Digital Image Media Laboratory(DIML)/Computer Vision Laboratory(CVL) RGBD dataset [62] contains 2 million color images paired with depth maps, covering diverse indoor and outdoor scenes. The RGB images have a resolution of 1920×1080, while depth maps are captured at 512×424 pixels. UNIFESP X-ray Body Part dataset [63] comprises X-ray images of various body parts, such as the knee, leg, hip, ankle, thigh, and pelvis. It stands out for its diversity of human anatomical coverage. Baggage Detection X-Ray(BDXR) dataset contains X-ray images of baggage inspected at airports to ensure diversity and generalization. The average image resolution is 1225×954 pixels. We described the overall distribution of data sources

APPENDIX DETAILED DESCRIPTION OF THE VS-TDX BENCHMARK DATASET

We curated a comprehensive collection comprising 13 distinct datasets, each tailored to specific sensor vision tasks. Detailed descriptions of each dataset are provided below.

M3FD [52] dataset contains images from three primary scenes: road views, university campuses, and resort settings. The dataset comprises 24-bit grayscale infrared and visible images, each with a resolution of 1024×768 pixels. Thermal Dogs and People [53] dataset includes 203 thermal infrared images captured at varying distances from people and dogs in park and home environments. Images are available in both portrait and landscape orientations with a spectral color palette applied. Pet dataset [54] features 640×640 images depicting diverse activities and motions of cats, dogs, and humans. This dataset has 640×640 image size. Thermal Computer Vision Project(TCVP) dataset [55] focuses on heat detection in groups of humans, with an average image size of 640×640 pixels. A high-altitude infrared thermal dataset for object detection applications on Unmanned Aerial Vehicles(HITUAV) [56] is a high-altitude infrared thermal dataset for object detection applications involving unmanned aerial vehicles (UAVs). It includes 2,898 infrared images derived from

of the VS-TDX benchmark in Figure 3.

HUMAN EVALUATION FOR THE VS-TDX BENCHMARK DATASET

We conducted a human evaluation study to assess how closely our newly proposed VS-TDX benchmark aligns with the answers a human would select when viewing sensor images. A total of 20 participants were recruited through the crowd-sourcing Prolific platform [48]. We only accepted reviewers with English as their first language and who had at least bachelor’s degree. In the human study, we recruited Prolific participants with approval rates higher than 95% and with at least 200 prior submissions.

Participants were rewarded e9.4/hr for completing all multiple choice questions. We sampled 45 vision sensor understanding questions from VS-TDX benchmark, with 15 questions allocated to each sensor type: thermal, depth, and X-ray. Experiment results on human evaluation are demonstrated in Figure A1. Participants achieved a 95.1% accuracy rate, demonstrating that the proposed VS-TDX benchmark significantly aligns with human assessment. We also evaluated how other VLMs responded to the 45 sampled questions and verified that their performance on vision sensor understanding, as shown in Table 1 of the main text, aligns within the margin of error. To be specific, GPT-4o [9] achieved the highest score of 73.3, followed by InternVL2-8B [22] and Phantom7B [45], both scoring 62.2, Qwen2-VL-7B [33] with 60.0, and LLaVA-1.5-7B [28], which recorded the lowest score of 53.3. The performance difference between the top VLMs (GPT-

- 4o) and human participants is notable at 21.8%, reflecting the challenges that current VLMs face in achieving human-level understanding in vision sensor understanding tasks.

THE INPUT PROMPT FOR GENERATING VS-TDX BENCHMARK

We designed the input prompts to create the proposed VS-TDX benchmark (in Figure 4), ensuring the prompts are comprehensive and tailored to extract meaningful vision sensor capabilities from challenging question and answer sets. These prompts require five additional information to effectively guide the VLMs in generating benchmark data:

- • To provide sufficient sensor information to the ChatGPT/GPT-4o(version 2024-08-06), we developed Sensor Knowledge (Figure A2) and incorporated it into <sensor knowledge>. This information contains detailed descriptions and context about thermal, depth, and X-ray sensors. This ensures the VLMs understand the unique physical properties and contextual applications of each sensor type.

- • The appropriate vision sensor type is included in <sensor type>. This explicitly informs the VLMs which sensor (thermal, depth, or X-ray) the input prompt relates to, ensuring that the generated examples are relevant to the specific sensor.

- • The desired question type and corresponding examples are provided in <question type> and <question examples>, respectively (Figure A2). This ensures that the model understands the format and context of the questions it needs to generate.

- • The number of negative samples to be generated is specified in <negative samples num>. These negative samples are designed to include plausible yet incorrect answers, encouraging the model to distinguish correct answers from distractors.

Our input prompts for generating VS-TDX benchmark is described in Figure A3.

SAMPLE QUESTION-ANSWER PAIRS FROM THE VS-TDX BENCHMARK

[Figure 39]

Performance on Vision Sensor Understanding among Different Models

Figure A4a-A4f provide examples of benchmark evaluations conducted using various Vision-Language Models (LLaVA-1.5-7B [44], InternVL2-8B [22], Phantom-7B [45], and Phantom-7B with Sensor-Aware Attribtues Fine-Tuning) across three vision sensors: thermal, depth, and X-ray. The answers selected by the models are displayed next to the corresponding options using the models’ representative icons and pictograms, and they are color-coded based on correctness: green indicates a correct answer, while red indicates an incorrect answer. By displaying these visual examples with clear indicators and detailed observations, we provide valuable insights into how different VLMs perform on vision sensor understanding tasks. These examples underscore the importance of tailored optimization techniques, like Diverse Negative Attributes(DNA) optimization, in enhancing the vision sensor understanding capabilities of VLMs across diverse sensor modalities.

Fig. A1: Comparative Performance Analysis of Different Models in Vision Sensor Understanding, including Human Agreement Results. The bar chart presents the accuracy (%) of various Vision-Language Models (VLMs) and human performance in understanding vision sensor data.

###### Sensor Knowledge:

{

Thermal: “Thermal images visualize infrared radiation emitted by objects using heat-sensing sensors. They can be used to analyze temperature distribution, detect objects, and inspect equipment conditions.”,

Depth: “Depth images visualize the distance between a sensor and objects in a scene by capturing depth information. They can be used to measure object dimensions, map environments in 3D, and assist in object recognition and navigation tasks.”,

X-ray: “X-ray images visualize the internal structures of objects by capturing the varying absorption of X-rays. They can be used to inspect internal components, identify structural defects, and analyze materials or biological tissues for diagnostic purposes.” }

###### Question Types and Examples:

{

Object Recognition: [“What is the object in the image?”, …], Counting Objects: [“How many objects are there in the image?”, …], Position Relationships: [“What object is next to person?”, …], Scene Description: [“What is this image?”, …], Contextual Understanding: [“What is this place for?", "What is that person doing?”, …], Sensor Understanding: [“What information can be gathered from this image?”, “Why the object are bright?”, …]

}

Fig. A2: Examples for sensor knowledge and question types in the input prompt for generating VS-TDX benchmark.

Input Prompt:

[ROLE] You are an expert at understanding images and generating relevant questions and answers based on them. [SENSOR KNOWLEDGE] <sensor_knowledge> [TASK] You will be given a <sensor_type> image and question type and a list of examples of questions: [QUESTION TYPE] <question_type> [QUESTION EXAMPLES] <question_examples> Your task is to:

- 1. Analyze the image, question type, and question examples thoroughly.
- 2. Make a strategy to create challenging questions when it is not known at the time that the image is from a <sensor_type> image.
- 3. Create questions based on the strategy and image.
- 4. Provide the correct answer to your question.
- 5. Generate <negative_samples_num> plausible incorrect answers that someone might give if they are confused by the image or lack sensor information. [REQUIREMENT]

- 1. Please ensure that the image information is fully utilized, and the answer cannot be inferred from the question alone.
- 2. Make the length of the correct answers similar to the length of the incorrect answers.
- 3. Create questions according to the question type, but please do not completely copy the content of the example questions. [OUTPUT FORMAT] Your output MUST be in JSON format as follows:

{

"strategy" : "[HOW TO MAKE IT CHALLENGING]", "question": "[YOUR QUESTION]", "answer": "[YOUR CORRECT ANSWER]", "question_category": "[YOUR QUESTION TYPE]", "incorrect_answers": [

- "[INCORRECT ANSWER 1]",
- "[INCORRECT ANSWER 2]",
- "[INCORRECT ANSWER 3]", ...

] }

Fig. A3: Examples of the input prompt for generating VS-TDX Benchmark

Fig. A3. Description of Prompts for Generating Challenging Multiple-Choice Questions and Answers in Vision Sensor Tasks

###### Existence

|LLaVA InternVL2 Phantom Phantom + SAFT<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]|
|---|

|Thermal<br><br>Q: What object is depicted in the cooler blue area on the right side of the image?<br><br>A. A lamp post<br>B. A trash can<br>C. A bicycle<br>D. A bench<br><br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]|
|---|
|Depth<br><br>Q: What large object can be seen near the center of the depth image?<br><br>A. A bed<br><br>B. A chair<br>C. A table<br>D. A sofa<br><br><br>[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]|
|XR<br><br>Q: What item is inside, indicated by its sharp vertical shape?<br><br>A. A toothbrush<br>B. A paintbrush<br>C. A pen<br><br>D. A knife<br><br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]|

(a) The comparison of performance across different sensors with respect to the representative VLMs in the Vision Sensor Perception task (Existence). Green font denotes the correct answer, while red font denotes the incorrect answer.

###### Counting

|LLaVA InternVL2 Phantom Phantom + SAFT<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[Figure 67]|
|---|

|Thermal<br><br>Q: How many vehicles are visible in the image?<br><br>A. One vehicle.<br>B. Three vehicles.<br>C. No vehicles.<br>D. Two vehicles.<br><br><br>[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>[Figure 72]<br><br>[Figure 73]|
|---|
|Depth<br><br>Q: How many rows of benches are visible in the image?<br><br>A. Eight rows<br>B. Six rows<br>C. Five rows<br><br>D. Three rows<br><br><br>[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]<br><br>[Figure 78]<br><br>[Figure 79]|
|XR<br><br>A. Four<br>B. Two<br>C. Five<br>D. Three<br><br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>Q: What is the approximate number of electronic devices visible in the image?<br><br>[Figure 84]<br><br>[Figure 85]|

(b) The comparison of performance across different vision sensors with respect to the representative VLMs in the Vision Sensor Perception task (Counting). Green font denotes the correct answer, while red font denotes the incorrect answer.

###### Position

|LLaVA InternVL2 Phantom Phantom + SAFT<br><br>[Figure 86]<br><br>[Figure 87]<br><br>[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]|
|---|

|Thermal<br><br>Q: What is the position of the warm object relative to the person in the image?<br><br>A. The warm object is beside the person.<br><br>B. The warm object, a dog, is in front of the person.<br><br>C. The warm object is behind the person.<br>D. The warm object is above the person.<br><br><br>[Figure 91]<br><br>[Figure 92]<br><br>[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]|
|---|
|Depth<br><br>Q: What is positioned centrally in front of the desk in the image?<br><br>A. A lamp is positioned centrally.<br>B. A table is positioned centrally.<br>C. A chair is positioned centrally.<br>D. A sofa is positioned centrally.<br><br><br>[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]<br><br>[Figure 101]<br><br>[Figure 102]|
|XR<br><br>A. A laptop charger.<br>B. Coins and small items.<br><br>C. A set of keys.<br>D. A large suitcase.<br><br><br>[Figure 103]<br><br>[Figure 104]<br><br>[Figure 105]<br><br>[Figure 106]<br><br>Q: What is located in the top right corner of the left compartment?<br><br>[Figure 107]<br><br>[Figure 108]|

(c) The comparison of performance across different vision sensors with respect to the representative VLMs in the Vision Sensor Perception task (Position). Green font denotes the correct answer, while red font denotes the incorrect answer.

###### General Description

|LLaVA InternVL2 Phantom Phantom + SAFT<br><br>[Figure 109]<br><br>[Figure 110]<br><br>[Figure 111]<br><br>[Figure 112]<br><br>[Figure 113]|
|---|

|Thermal<br><br>Q: What is the shape of the area marked by lines in the center of the image?<br><br>A. Soccer fields with goals.<br>B. Tennis courts with nets.<br>C. Parking lots with cars.<br>D. Basketball courts with hoops.<br><br><br>[Figure 114]<br><br>[Figure 115]<br><br>[Figure 116]<br><br>[Figure 117]<br><br>[Figure 118]<br><br>[Figure 119]|
|---|
|Depth<br><br>Q: What layout is primarily depicted in the image?<br><br>A. Amusement park layout.<br>B. Residential neighborhood.<br>C. Library or study hall layout.<br><br>D. Sports stadium seating.<br><br><br>[Figure 120]<br><br>[Figure 121]<br><br>[Figure 122]<br><br>[Figure 123]<br><br>[Figure 124]<br><br>[Figure 125]|
|XR<br><br>Q: What items are commonly seen inside these bags?<br><br>A. Toys and books.<br>B. Food and drinks.<br>C. Clothing and shoes.<br>D. Electronic devices and cables<br><br><br>[Figure 126]<br><br>[Figure 127]<br><br>[Figure 128]<br><br>[Figure 129]<br><br>[Figure 130]<br><br>[Figure 131]|

(d) The comparison of performance across different vision sensors with respect to the representative VLMs in the Vision Sensor Perception task (General Description). Green font denotes the correct answer, while red font denotes the incorrect answer.

###### Contextual Understanding

|LLaVA InternVL2 Phantom Phantom + SAFT<br><br>[Figure 132]<br><br>[Figure 133]<br><br>[Figure 134]<br><br>[Figure 135]<br><br>[Figure 136]|
|---|

|Thermal<br><br>Q: What is the interaction between the two objects in this image?<br><br>A. A cat is chasing a ball across the floor.<br>B. A person is sitting beside another person.<br>C. Two cats are playing together on a chair.<br><br>D. A person is reaching towards a cat on a chair.<br><br><br>[Figure 137]<br><br>[Figure 138]<br><br>[Figure 139]<br><br>[Figure 140]<br><br>[Figure 141]<br><br>[Figure 142]|
|---|
|Depth<br><br>Q: What is the object on the table likely used for?<br><br>A. Providing light in the room.<br>B. Charging electronic devices.<br>C. Displaying time and date.<br>D. Holding water or other beverages.<br><br><br>[Figure 143]<br><br>[Figure 144]<br><br>[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]<br><br>[Figure 148]|
|XR<br><br>Q: What is the likely setting where these items might be found?<br><br>A. In a toolbox in a garage.<br>B. Inside a suitcase at an airport.<br><br>C. Inside a school backpack.<br>D. In a kitchen drawer.<br><br><br>[Figure 149]<br><br>[Figure 150]<br><br>[Figure 151]<br><br>[Figure 152]<br><br>[Figure 153]<br><br>[Figure 154]|

(e) The comparison of performance across different vision sensors with respect to the representative VLMs in the Vision Sensor Understanding task (Contextual Understanding). Green font denotes the correct answer, while red font denotes the incorrect answer.

###### Sensor Understanding

|LLaVA InternVL2 Phantom Phantom + SAFT<br><br>[Figure 155]<br><br>[Figure 156]<br><br>[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]|
|---|

|Thermal<br><br>[Figure 160]<br><br>Q: Why is the person in the image brighter compared to the surroundings?<br><br>A. The camera focuses on them.<br>B. They are wearing reflective clothing.<br><br>C. They emit more heat than the surroundings.<br><br>D. The light is brighter on them.<br><br><br>[Figure 161]<br><br>[Figure 162]<br><br>[Figure 163]<br><br>[Figure 164]<br><br>[Figure 165]|
|---|
|Depth<br><br>Q: What causes the table to appear darker than the surrounding chairs?<br><br>A. The lighting on the table is dimmer.<br>B. The table is reflecting less light.<br>C. The table is closer to the sensor.<br><br>D. The table is made of darker material.<br><br><br>[Figure 166]<br><br>[Figure 167]<br><br>[Figure 168]<br><br>[Figure 169]<br><br>[Figure 170]<br><br>[Figure 171]|
|XR<br><br>Q: What causes the green and blue areas to appear in the image?<br><br>A. They indicate materials with different densities.<br><br>B. They are natural artifacts of light.<br>C. They show signs of digital editing.<br>D. They represent temperature variations.<br><br><br>[Figure 172]<br><br>[Figure 173]<br><br>[Figure 174]<br><br>[Figure 175]<br><br>[Figure 176]<br><br>[Figure 177]|

(f) The comparison of performance across different vision sensors with respect to the representative VLMs in the Vision Sensor Understanding task (Sensor Understanding). Green font denotes the correct answer, while red font denotes the incorrect answer.

Fig. A4: Comprehensive performance comparison of various representative Vision-Language Models (VLMs) across different vision sensor modalities (e.g., thermal, depth, X-ray) on diverse perception and understanding tasks.

REFERENCES

- [1] J. Y. Koh, R. Salakhutdinov, and D. Fried, “Grounding language models to images for multimodal inputs and outputs,” in International Conference on Machine Learning. PMLR, 2023, pp. 17283–17300.
- [2] Z. Wang, Y. Long, Q. Jiang, C. Huang, and X. Cao, “Harnessing multi-modal large language models for measuring and interpreting color differences,” IEEE Transactions on Image Processing, 2025.
- [3] N. Yellinek, L. Karlinsky, and R. Giryes, “3vl: Using trees to improve vision-language models’ interpretability,” IEEE Transactions on Image Processing, vol. 34, pp. 495–509, 2025.
- [4] S. Ren, L. Yao, S. Li, X. Sun, and L. Hou, “Timechat: A time-sensitive multimodal large language model for long video understanding,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 14313–14323.
- [5] J. Guo, A. Lu, Z. Wu, Z. Wang, and C. Liang, “Who, what and where: Composite-semantics instance search for story videos,” IEEE Transactions on Image Processing, 2025.
- [6] X. Ding, N. Wang, S. Zhang, Z. Huang, X. Li, M. Tang, T. Liu, and X. Gao, “Exploring language hierarchy for video grounding,” IEEE Transactions on Image Processing, vol. 31, pp. 4693–4706, 2022.
- [7] W. Jin, Z. Zhao, X. Cao, J. Zhu, X. He, and Y. Zhuang, “Adaptive spatiotemporal graph enhanced vision-language representation for video qa,” IEEE Transactions on Image Processing, vol. 30, pp. 5477–5489, 2021.
- [8] J. Ye, A. Hu, H. Xu, Q. Ye, M. Yan, Y. Dan, C. Zhao, G. Xu, C. Li, J. Tian et al., “mplug-docowl: Modularized multimodal large language model for document understanding,” arXiv preprint arXiv:2307.02499, 2023.
- [9] OpenAI. (2024) Hello gpt-4o. https://openai.com/index/hello-gpt-4o/. [Online]. Available: https://openai.com/index/hello-gpt-4o/
- [10] J. Mao, Y. Qian, H. Zhao, and Y. Wang, “Gpt-driver: Learning to drive with gpt,” arXiv preprint arXiv:2310.01415, 2023.
- [11] Z. Xu, Y. Zhang, E. Xie, Z. Zhao, Y. Guo, K.-Y. K. Wong, Z. Li, and H. Zhao, “Drivegpt4: Interpretable end-to-end autonomous driving via large language model,” IEEE Robotics and Automation Letters, 2024.
- [12] Z. Guo, Z. Yagudin, A. Lykov, M. Konenkov, and D. Tsetserukou, “Vlm-auto: Vlm-based autonomous driving assistant with human-like behavior and understanding for complex road scenes,” 2024. [Online]. Available: https://arxiv.org/abs/2405.05885
- [13] W. Shi, C. Chen, K. Li, Y. Xiong, X. Cao, and Z. Zhou, “Langloc: Language-driven localization via formatted spatial description generation,” IEEE Transactions on Image Processing, 2025.
- [14] X. Chu, L. Qiao, X. Lin, S. Xu, Y. Yang, Y. Hu, F. Wei, X. Zhang, B. Zhang, X. Wei et al., “Mobilevlm: A fast, reproducible and strong vision language assistant for mobile devices,” arXiv preprint arXiv:2312.16886, 2023.
- [15] Q. M. Dinh, M. K. Ho, A. Q. Dang, and H. P. Tran, “Trafficvlm: A controllable visual language model for traffic video captioning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2024, pp. 7134–7143.
- [16] Y. Cho, T. Kim, H. Shin, S. Cho, and D. Shin, “Pretraining vision-language model for difference visual question answering in longitudinal chest x-rays,” 2024. [Online]. Available: https: //arxiv.org/abs/2402.08966
- [17] J. Gao, B. Sarkar, F. Xia, T. Xiao, J. Wu, B. Ichter, A. Majumdar, and D. Sadigh, “Physically grounded vision-language models for robotic manipulation,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 12462–12469.
- [18] W. Huang, C. Wang, Y. Li, R. Zhang, and L. Fei-Fei, “Rekep: Spatio-temporal reasoning of relational keypoint constraints for robotic manipulation,” 2024. [Online]. Available: https://arxiv.org/abs/2409. 01652
- [19] J. Duan, W. Pumacay, N. Kumar, Y. R. Wang, S. Tian, W. Yuan, R. Krishna, D. Fox, A. Mandlekar, and Y. Guo, “Aha: A vision-languagemodel for detecting and reasoning over failures in robotic manipulation,”

2024. [Online]. Available: https://arxiv.org/abs/2410.00371

- [20] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, P. Florence, C. Fu, M. G. Arenas, K. Gopalakrishnan, K. Han, K. Hausman, A. Herzog, J. Hsu, B. Ichter, A. Irpan, N. Joshi,

- R. Julian, D. Kalashnikov, Y. Kuang, I. Leal, L. Lee, T.-W. E. Lee, S. Levine, Y. Lu, H. Michalewski, I. Mordatch, K. Pertsch, K. Rao, K. Reymann, M. Ryoo, G. Salazar, P. Sanketi, P. Sermanet, J. Singh, A. Singh, R. Soricut, H. Tran, V. Vanhoucke, Q. Vuong, A. Wahid, S. Welker, P. Wohlhart, J. Wu, F. Xia, T. Xiao, P. Xu,
- S. Xu, T. Yu, and B. Zitkovich, “Rt-2: Vision-language-action models

transfer web knowledge to robotic control,” 2023. [Online]. Available: https://arxiv.org/abs/2307.15818

- [21] S. Huang, H. Chang, Y. Liu, Y. Zhu, H. Dong, P. Gao, A. Boularias, and H. Li, “A3vlm: Actionable articulation-aware vision language model,” 2024. [Online]. Available: https://arxiv.org/abs/2406.07549
- [22] OpenGVLab. (2024) Internvl2: Better than the best—expanding performance boundaries of open-source multimodal models with the progressive scaling strategy. https://internvl.github.io/blog/ 2024-07-02-InternVL-2.0/. [Online]. Available: https://internvl.github. io/blog/2024-07-02-InternVL-2.0/
- [23] W. Cai, I. Ponomarenko, J. Yuan, X. Li, W. Yang, H. Dong, and B. Zhao, “Spatialbot: Precise spatial understanding with vision language models,” arXiv preprint arXiv:2406.13642, 2024.
- [24] W. Wang, J. Xie, C. Hu, H. Zou, J. Fan, W. Tong, Y. Wen, S. Wu, H. Deng, Z. Li et al., “Drivemlm: Aligning multi-modal large language models with behavioral planning states for autonomous driving,” arXiv preprint arXiv:2312.09245, 2023.
- [25] J. Han, K. Gong, Y. Zhang, J. Wang, K. Zhang, D. Lin, Y. Qiao, P. Gao, and X. Yue, “Onellm: One framework to align all modalities with language,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 26584–26595.
- [26] O. Thawkar, A. Shaker, S. S. Mullappilly, H. Cholakkal, R. M. Anwer, S. Khan, J. Laaksonen, and F. S. Khan, “Xraygpt: Chest radiographs summarization using medical vision-language models,” arXiv preprint arXiv:2306.07971, 2023.
- [27] J. Yu, H. Xiong, L. Zhang, H. Diao, Y. Zhuge, L. Hong, D. Wang, H. Lu, Y. He, and L. Chen, “Llms can evolve continually on modality for x-modal reasoning,” arXiv preprint arXiv:2410.20178, 2024.
- [28] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” 2023.
- [29] H. Liu, C. Li, Y. Li, B. Li, Y. Zhang, S. Shen, and Y. J. Lee, “Llava-next: Improved reasoning, ocr, and world knowledge,” January 2024. [Online]. Available: https://llava-vl.github.io/blog/2024-01-30-llava-next/
- [30] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models,” in International conference on machine learning. PMLR, 2023, pp. 19730–19742.
- [31] Z. Cheng, S. Leng, H. Zhang, Y. Xin, X. Li, G. Chen, Y. Zhu, W. Zhang, Z. Luo, D. Zhao et al., “Videollama 2: Advancing spatialtemporal modeling and audio understanding in video-llms,” arXiv preprint arXiv:2406.07476, 2024.
- [32] Y. Yao, T. Yu, A. Zhang, C. Wang, J. Cui, H. Zhu, T. Cai, H. Li, W. Zhao, Z. He et al., “Minicpm-v: A gpt-4v level mllm on your phone,” arXiv preprint arXiv:2408.01800, 2024.
- [33] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge et al., “Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution,” arXiv preprint arXiv:2409.12191, 2024.
- [34] R. Girdhar, A. El-Nouby, Z. Liu, M. Singh, K. V. Alwala, A. Joulin, and I. Misra, “Imagebind: One embedding space to bind them all,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 15180–15190.
- [35] Y. Su, T. Lan, H. Li, J. Xu, Y. Wang, and D. Cai, “Pandagpt: One model to instruction-follow them all,” arXiv preprint arXiv:2305.16355, 2023.
- [36] C. Fu, P. Chen, Y. Shen, Y. Qin, M. Zhang, X. Lin, J. Yang, X. Zheng, K. Li, X. Sun, Y. Wu, and R. Ji, “Mme: A comprehensive evaluation benchmark for multimodal large language models,” 2024. [Online]. Available: https://arxiv.org/abs/2306.13394
- [37] B. Li, R. Wang, G. Wang, Y. Ge, Y. Ge, and Y. Shan, “Seed-bench: Benchmarking multimodal llms with generative comprehension,” arXiv preprint arXiv:2307.16125, 2023.
- [38] X. Yue, Y. Ni, K. Zhang, T. Zheng, R. Liu, G. Zhang, S. Stevens, D. Jiang, W. Ren, Y. Sun et al., “Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 9556–9567.
- [39] H. Wu, Z. Zhang, E. Zhang, C. Chen, L. Liao, A. Wang, C. Li, W. Sun, Q. Yan, G. Zhai et al., “Q-bench: A benchmark for general-purpose foundation models on low-level vision,” arXiv preprint arXiv:2309.14181, 2023.
- [40] Z. Zhang, H. Wu, E. Zhang, G. Zhai, and W. Lin, “Q-bench++: A benchmark for multi-modal foundation models on low-level vision from single images to pairs,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 12, pp. 10404–10418, 2024.
- [41] Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu et al., “Mmbench: Is your multi-modal model an all-around player?” in European Conference on Computer Vision. Springer, 2025, pp. 216–233.

- [42] D. Kahneman, A. Treisman, and B. J. Gibbs, “The reviewing of object files: Object-specific integration of information,” Cognitive psychology, vol. 24, no. 2, pp. 175–219, 1992.
- [43] D. E. Broadbent, Perception and communication. Elsevier, 2013.
- [44] H. Liu, C. Li, Y. Li, and Y. J. Lee, “Improved baselines with visual instruction tuning,” 2023.
- [45] B.-K. Lee, S. Chung, C. W. Kim, B. Park, and Y. M. Ro, “Phantom of latent for large language and vision models,” arXiv preprint arXiv:2409.14713, 2024.
- [46] G. Team, P. Georgiev, V. I. Lei, R. Burnell, L. Bai, A. Gulati, G. Tanzer, D. Vincent, Z. Pan, S. Wang et al., “Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context,” arXiv preprint arXiv:2403.05530, 2024.
- [47] Anthropic. (2024) Claude 3.5 sonnet. https://www.anthropic.com/ news/claude-3-5-sonnet. [Online]. Available: https://www.anthropic. com/news/claude-3-5-sonnet
- [48] “Prolific,” https://www.prolific.com/, Prolific, 2025.
- [49] R. A. Bradley and M. E. Terry, “Rank analysis of incomplete block designs,” Biometrika, 1952. [Online]. Available: https://api. semanticscholar.org/CorpusID:115965399
- [50] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, “Direct preference optimization: Your language model is secretly a reward model,” Advances in Neural Information Processing Systems, vol. 36, 2024.
- [51] F. Schroff, D. Kalenichenko, and J. Philbin, “Facenet: A unified embedding for face recognition and clustering,” in 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, Jun. 2015, p. 815–823. [Online]. Available: http://dx.doi.org/10. 1109/CVPR.2015.7298682
- [52] J. Liu, X. Fan, Z. Huang, G. Wu, R. Liu, W. Zhong, and Z. Luo, “Target-aware dual adversarial learning and a multi-scenario multimodality benchmark to fuse infrared and visible for object detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5802–5811.
- [53] Roboflow, “thermal dogs and people x6ejw dataset,” https://universe. roboflow.com/object-detection/thermal-dogs-and-people-x6ejw, nov 2022, visited on 2023-03-29. [Online]. Available: https://universe. roboflow.com/object-detection/thermal-dogs-and-people-x6ejw
- [54] harang, “pet dataset,” https://universe.roboflow.com/harang/pet-kjl3x, jul 2024, visited on 2024-10-28. [Online]. Available: https://universe. roboflow.com/harang/pet-kjl3x
- [55] Visual, “Thermal dataset,” https://universe.roboflow.com/visual-iqhyh/ thermal-duv93, dec 2023, visited on 2024-10-22. [Online]. Available: https://universe.roboflow.com/visual-iqhyh/thermal-duv93
- [56] J. Suo, T. Wang, X. Zhang, H. Chen, W. Zhou, and W. Shi, “Hit-uav: A high-altitude infrared thermal dataset for unmanned aerial vehicle-based object detection,” Scientific Data, vol. 10, p. 227, 2023.
- [57] one, “animal-detection-flir-extra dataset,” https://universe.roboflow.com/ one-rphct/animal detection flir extra, apr 2023, visited on 2024-10-

28. [Online]. Available: https://universe.roboflow.com/one-rphct/animal detection flir extra

- [58] james cook, “chips-thermal-face-dataset,” https://www.kaggle.com/ datasets/kagglechip/chips-thermal-face-dataset, apr 2020, visited on 2024-10-28. [Online]. Available: https://www.kaggle.com/datasets/ kagglechip/chips-thermal-face-dataset
- [59] NJUST, “Ifsod dataset,” https://universe.roboflow.com/njust-oxpbo/ ifsod, aug 2023, visited on 2024-10-28. [Online]. Available: https://universe.roboflow.com/njust-oxpbo/ifsod
- [60] I. Vasiljevic, N. Kolkin, S. Zhang, R. Luo, H. Wang, F. Z. Dai, A. F. Daniele, M. Mostajabi, S. Basart, M. R. Walter, and G. Shakhnarovich, “Diode: A dense indoor and outdoor depth dataset,” 2019. [Online]. Available: https://arxiv.org/abs/1908.00463
- [61] P. K. Nathan Silberman, Derek Hoiem and R. Fergus, “Indoor segmentation and support inference from rgbd images,” in ECCV, 2012.
- [62] J. Cho, D. Min, Y. Kim, and K. Sohn, “Diml/cvl rgb-d dataset: 2m rgb-d images of natural indoor and outdoor scenes,” 2021. [Online]. Available: https://arxiv.org/abs/2110.11590
- [63] F. Eduardo Farina, “Unifesp x-ray body part classifier competition,” 2022. [Online]. Available: https://kaggle.com/competitions/ unifesp-x-ray-body-part-classifier
- [64] Malek, “X-ray baggage detection dataset,” https://universe.roboflow. com/malek-mhnrl/x-ray-baggage-detection, apr 2022, visited on 202411-11. [Online]. Available: https://universe.roboflow.com/malek-mhnrl/ x-ray-baggage-detection
- [65] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “Qlora: Efficient finetuning of quantized llms,” Advances in Neural Information Processing Systems, vol. 36, 2024.

- [66] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,”

2019. [Online]. Available: https://arxiv.org/abs/1711.05101

- [67] M. G. Azar, M. Rowland, B. Piot, D. Guo, D. Calandriello, M. Valko, and R. Munos, “A general theoretical paradigm to understand learning from human preferences,” 2023. [Online]. Available: https://arxiv.org/abs/2310.12036
- [68] Y. Meng, M. Xia, and D. Chen, “Simpo: Simple preference optimization with a reference-free reward,” arXiv preprint arXiv:2405.14734, 2024.
- [69] “TE-SQ1 Thermal Camera,” https://www.i3-thermalexpert.com/ products/te-sq1/, i3 Thermal Expert, 2025.

[Figure 178]

SANGYUN CHUNG received the B.S. degree from Hanyang University, Seoul, South Korea, in 2023. He is currently working toward the Ph.D. degree in electronic engineering, Korea Advanced Institute of Science and Technology (KAIST), Deajeon, South Korea. His research interests include deep learning, object detection, and multimodal large language models.

[Figure 179]

YOUNGJOON YU received the B.S. degree in electrical engineering from Korea Advanced Institute of Science and Technology (KAIST), Daejeon, South Korea in 2013, and the M.S. degree in the management engineering from KAIST in 2017. He is currently pursuing the Ph.D. in electrical engineering at KAIST, Daejeon, South Korea. His research interests include deep learning, multi-sensor learning, and multimodal large language models.

[Figure 180]

SEYEON KIM received the B.S. degree in Electronic Engineering from Hanyang University, Seoul, South Korea, in 2024. She is currently pursuing her M.S. degree in Electrical and Electronic Engineering at the Korea Advanced Institute of Science and Technology (KAIST), Daejeon, South Korea. Her research interests include deep learning, multimodal large language models, and vision-language understanding.

[Figure 181]

YOUNGCHAE CHEE received the B.S. degree from Hanyang University, Seoul, South Korea, in 2025. He is currently working toward the Masters degree in electronic engineering, Korea Advanced Institute of Science and Technology (KAIST), Daejeon, South Korea. His research interests include deep learning and multimodal large language models.

YONG MAN RO (Senior Member, IEEE) received a B.S. degree from Yonsei University, Seoul, South Korea, and a M.S. and Ph.D. degrees from the Korea Advanced Institute of Science and Technology (KAIST), Daejeon, South Korea. He was a Researcher at Columbia University, a Visiting Researcher at the University of California at Irvine, Irvine, CA, USA, and a Research Fellow of the University of California at Berkeley, Berkeley, CA, USA. He was a Visiting Professor with the Department of Electrical and Computer Engineering,

[Figure 182]

University of Toronto, Canada. He is currently a Professor at the Department of Electrical Engineering and the Director of the Center for Applied Research in Artificial Intelligence (CARAI), KAIST. Among the years, he has been conducting research in a wide spectrum of image and video systems research topics. Among those topics, his interests include image processing, computer vision, visual recognition, multimodal learning, video representation/compression, and object detection. He received the Young Investigator Finalist Award of ISMRM, in 1992, and the Year’s Scientist Award (Korea), in 2003. He served as an Associate Editor for IEEE SIGNAL PROCESSING LETTERS. He currently serves as an Associate Editor for IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY and IEEE TRANSACTIONS ON IMAGE PROCESSING. He served as a TPC in many international conferences, including the program chair, and organized special sessions.

