# arXiv:2408.12114v3[cs.CV]11Oct2024

## SPARK: Multi-Vision Sensor Perception and Reasoning Benchmark for Large-scale Vision-Language Models

#### Youngjoon Yu†, Sangyun Chung†, Byung-Kwan Lee, and Yong Man Ro*

Integrated Vision Language Lab, KAIST, South Korea {greatday, jelarum, leebk, ymro}@kaist.ac.kr

###### Abstract

|Sensory Reasoning Performance Across Different LVLMs and Vision Sensors<br><br>[Figure 1]<br><br>Multi-Vision Sensor<br><br>SensoryReasoning(Acc)<br><br>LLAVA-v1.5 CogVLM InternVL2 TroL<br><br>Meteor IXC 2.5 QwenVL GPT4o<br><br>|
|---|
|[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>Yes! the ribs are visible in this chest X-ray image<br><br>Are there ribs visible in this image?<br><br>[Figure 8]<br><br>What could be the likely reason for capturing this image?<br><br>[A] To measure the temperature of items.<br>[B] To describe the lighting condition.<br>[C] To detect hazy, foggy atmosphere.<br>[D] To analyze the spatial arrangement of objects in a room.<br><br><br>[Figure 9]<br><br>[Figure 10]<br><br>[C] To detect hazy, foggy atmosphere.<br><br>[Figure 11]<br><br>XR image<br><br>Depth image|

Large-scale Vision-Language Models (LVLMs) have significantly advanced with text-aligned vision inputs. They have made remarkable progress in computer vision tasks by aligning text modality with vision inputs. There are also endeavors to incorporate multi-vision sensors beyond RGB, including thermal, depth, and medical X-ray images. However, we observe that current LVLMs view images taken from multivision sensors as if they were in the same RGB domain without considering the physical characteristics of multi-vision sensors. They fail to convey the fundamental multi-vision sensor information from the dataset and the corresponding contextual knowledge properly. Consequently, alignment between the information from the actual physical environment and the text is not achieved correctly, making it difficult to answer complex sensor-related questions that consider the physical environment. In this paper, we aim to establish a multi-vision Sensor Perception And Reasoning benchmarK called SPARK that can reduce the fundamental multi-vision sensor information gap between images and multi-vision sensors. We generated 6,248 vision-language test samples to investigate multi-vision sensory perception and multi-vision sensory reasoning on physical sensor knowledge proficiency across different formats, covering different types of sensor-related questions. We utilized these samples to assess ten leading LVLMs. The results showed that most models displayed deficiencies in multi-vision sensory reasoning to varying extents. Codes and data are available at https://github.com/top-yun/SPARK

Figure 1: The comparison of sensory reasoning performance across different multi-vision sensors with respect to the recent LVLMs. Note that, sensory reasoning performance significantly drops across different multi-vision sensors.

#### Introduction

In recent days, Large-scale Vision-Language Models (LVLMs) have achieved significant breakthroughs in areas such as visual dialogue (Koh, Salakhutdinov, and Fried 2023), video analysis (Ren et al. 2024), and document understanding (Ye et al. 2023), establishing themselves as critical tools in the pursuit of artificial general intelligence (AGI). These models function similarly to the human brain by processing multimodal information and generating sophisticated inferences. For instance, the latest LVLMs, like OpenAI’s GPT-4o (OpenAI 2024), have exhibited exceptional reasoning abilities that not only rival but in some cases exceed human performance.

*Corresponding author. † Both authors are equally contributed.

One emerging concept in modern AI research gaining significant attention is the development of large vision language models (LVLMs) capable of handling a variety of multimodal inputs, surpassing the capabilities of previous large language models (LLMs). LVLMs can process diverse forms of data simultaneously, including images, videos, and text (OpenAI 2024; OpenGVLab 2024; Zhang et al. 2024). This ability also allows them to use multi-vision sensor data as input, including thermal sensors, depth sensors, and medical imaging (Girdhar et al. 2023; Su et al. 2023). To fully harness the potential of LVLMs, recent research has focused on effectively integrating various multi-vision sensor data to develop more sophisticated and practical AI systems for the real world.

However, despite the remarkable advancements in LVLM models, significant challenges still remain in fully utilizing multi-vision sensors. LVLMs often overlook the nuances of the physical properties of individual vision sensors. Instead, they tend to make judgments based on prior visual or linguistic information from images they have learned using low-level features in two-dimensional data. This results in the models recognizing only superficial patterns in image inputs, missing the underlying logical structures or contextual understanding. When identifying specific objects in an image input, a model might rely on patterns learned from similar-looking images rather than considering the actual physical properties of the multi-vision sensors used to capture the image. This can hinder accurate identification and a deep understanding of the input images in fields where the LVLM’s decision-making is crucial such as autonomous driving (Mao et al. 2023; Xu et al. 2024), security systems (Shi et al. 2024), and medical image diagnosis (Bazi et al. 2023).

We evaluate the behavior of the recent LVLMs using multi-vision sensor images as input in Figure 1. The performance of sensory reasoning, which we devised to assess the understanding of fundamental knowledge of multi-vision sensors in the real world, significantly drops across different multi-vision sensors such as thermal infrared, depth, and X-ray (XR) images. This highlights the challenges that LVLMs face in accurately interpreting multi-vision sensor data and making correct inferences based on the physical properties of sensors. Additionally, from the interaction example shown below in Figure 1, while the LVLM can accurately identify the vision sensor used to capture the image for a relatively simple question, it struggles with understanding the actual purpose or context of the image in the sensor-related, more complicated questions. This indicates that current LVLMs have difficulty in understanding the fundamental knowledge of physical vision sensors beyond what the image looks like.

For example, as illustrated in Figure 1, when humans look at a photograph of an X-ray medical image, they interpret it deeply, drawing upon their knowledge base and their physical understanding of the human body beyond the X-ray image itself. Despite never having seen their internal organs and the structure of bones with the naked eye, humans can comprehend the image through scientific contextual knowledge and their inherent understanding of the physical world. In contrast, current LVLMs try to understand the inside of the human body based solely on the two-dimensional data they have been trained on, revealing their limitations in fully grasping the physical environment of the real world. Therefore, establishing a comprehensive evaluation benchmark is necessary before LVLMs are implemented in critical and sensitive real-world applications. However, the assessment of Large Vision-Language Models (LVLMs) has significantly lagged behind their rapid development. Several initiatives are striving to close this gap by introducing a variety of multimodal evaluation benchmarks. Notable examples include MME (Fu et al. 2024), MMBench (Liu et al. 2024b), LVLM-eHub (Xu et al. 2023), and SEED-Bench (Li et al. 2023a). These benchmarks aim to define key dimensions of

multimodal capabilities and provide corresponding test samples. But, they cover a relatively narrow range of multimodal tasks, primarily focusing on fundamental abilities such as visual recognition and OCR.

In this paper, to handle the aforementioned challenge, we design the SPARK benchmark to evaluate multi-vision input LVLMs on two fronts: multi-vision perception and multivision reasoning. Multi-vision perception pertains to the information needed, which measures the LVLM’s effectiveness in satisfying visual perception needs. Multi-vision reasoning measures the LVLM’s ability to base its responses on fundamental information from the provided sensor knowledge. To be specific, we generated 6,248 vision-language test samples to investigate multi-vision sensory perception and reasoning related to physical sensor knowledge proficiency, covering 6 types of multi-vision sensory instruction tasks across 2 different question-and-answer formats. We used these samples to assess 10 leading large-scale vision language models. The experiment results validate that most LVLMs displayed deficiencies in sensory reasoning to varying extents.

In summary, the contributions of this work are as follows:

- • To the best of our knowledge, we first reveal the incapability of current LVLMs, which suffer from limited multi-vision sensory reasoning across different multivision sensors due to an absence of fundamental understanding of sensors in the physical world.
- • We propose a novel benchmark, SPARK, to rigorously test and evaluate the capabilities of LVLMs in understanding sensory knowledge, providing a comprehensive framework for assessing their performance.
- • We evaluated a total of 10 state-of-the-art LVLMs using our SPARK benchmark, which is designed to rigorously assess the capability of the LVLMs in handling fundamental knowledge related to multi-vision sensors.

#### Related work

Large-scale Vison-Language Models. Recently, there has been significant interest in visual language multimodal learning. Visual language models such as LLAVA (Liu et al.

- 2023b, 2024a), CollaVO (Lee et al. 2024c), MoAI (Lee et al. 2024d), TroL (Lee et al. 2024a), Meteor (Lee et al.
- 2024b), IXC2.5 (Zhang et al. 2024), and QwenVL (Bai et al. 2023) have shown impressive performance in a variety of downstream tasks. In addition, to obtain richer contextual information, LVLMs have developed the capability to handle multimodal inputs. Wang et al. introduces CogVLM, an advanced visual language foundation multimodal model that integrates a trainable visual expert module with a pretrained language model. InternVL2 (Chen et al. 2024) is an open-source multimodal large language model that bridges the gap between open-source and commercial models by enhancing visual understanding, dynamic high-resolution processing, and bilingual dataset quality. GPT4o (OpenAI

2024) possesses advanced multimodal capabilities, allowing it to process and generate diverse multimodalities. This enables the model to understand and create content that integrates visual and textual information, making it suitable

### SPARK

###### Counting (Depth)

###### Existence (RGB)

|[Figure 12]|
|---|

|[Figure 13]|
|---|

[Figure 14]

[Figure 15]

[Y] Are there two bottles in this image? [N] Are there two monitors in this image?

[Y] Is there a flower on the man's suit in this image? [N] Is there a hat being worn by anyone in this image?

[Figure 16]

[Figure 17]

[Figure 18]

Multi-vision Perception

[Figure 19]

###### General Description (XR)

###### Position (Thermal)

|[Figure 20]|
|---|

|[Figure 21]|
|---|

[Figure 22]

[Figure 23]

[Y] Are the ribs visible in the image? [N] Is there a distinct liver shape visible in the image?

[Y] Is the person standing to the left of the dog ? [N] Is the dog positioned behind the person?

###### Sensory Reasoning (Thermal)

###### Contextual Reasoning (Thermal)

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Multi-vision Reasoning

Q: What might be the reason for the vehicles lined up in the image?

Q: What could be the likely reason for capturing this image?

[Figure 29]

|[Figure 30]|
|---|

|[Figure 31]|
|---|

- [A] They are waiting for a traffic signal.

- [B] They are parked for the night.
- [C] They are part of a parade.
- [D] They are in a car wash.

- [A] To assess the environmental conditions of the area.
- [B] To study the feeding habits of livestock.
- [C] To evaluate the breeding patterns of the animal.
- [D] To monitor the health and temperature of animal.

- Figure 2: In the proposed SPARK, we build the first benchmark for evaluating the abilities of LVLMs in multi-vision sensor understanding, which covers four types of multi-vision perception tasks (Existence, Counting, Position, and General Description) and two types of multi-vision reasoning tasks (Contextual Reasoning and Sensory Reasoning).

for a wide range of applications that require various modalities. Consequently, many LVLMs have emerged that take multi-vision sensor images as input. Girdhar et al. presents ImageBind, which creates a joint embedding space across multi-vision sensors including depth and thermal sensor data. PandaGPT (Su et al. 2023) is a LVLM that integrates multimodal encoders and large language models to enable multi-vision and auditory instruction-following capabilities, performing complex tasks. However, relatively less attention has been devoted to whether LVLMs truly understand the physical meanings of multi-vision sensors used to capture the input image.

of LVLMs. POPE (Li et al. 2023b) focuses on evaluating object hallucination by asking yes/no questions about the presence of objects in the input image. M-HalDetect (Gunjal, Yin, and Bas 2024) introduces hallucination tasks using human-annotated labels for sentence-level classification. Unlike those previous evaluation benchmarks, the proposed SPARK is designed to rigorously test and evaluate the capabilities of understanding the physical meaning of multivision sensors.

#### Evaluation and Instruction Design

There are multiple formats available for evaluating the multi-sensor perception and reasoning capabilities of LVLM, each with distinct advantages and limitations. Freeform questions (Yarom et al. 2024) offer flexibility and ease of creation but demand labor-intensive human assessment and present challenges in maintaining consistent scoring. Similarity-based assessment are less resource-intensive but can be significantly affected by biases present in the similarity metrics. Yes-or-No questions (Fu et al. 2024) are straightforward and easier to assess, but they may oversimplify the evaluation, failing to capture the full extent of LVLM’s comprehension of multi-vision reasoning ability.

Evaluation Benchmark for LVLMs. Numerous studies have leveraged existing vision-language datasets to develop benchmarks for assessing the reliability of LVLMs (Li and Lu 2024). MME (Fu et al. 2024) includes 14 sub-tasks based on publicly available images with manually created annotations, evaluating both the recognition and perception capabilities of LVLMs through yes/no question answering. SEED-benchmark (Li et al. 2023a) designed to evaluate the generative comprehension capabilities of multimodal LVLM through human-annotated multi-choice questions across 12 evaluation dimensions. Other comparable benchmarks include LVLM-eHub (Xu et al. 2023), MM-Vet (Yu et al. 2023), and MMBench (Liu et al. 2024b). Additionally, there are benchmarks aimed at assessing specific target properties

First of all, to enable quantitative performance metrics for multi-vision perception, the instruction design aims to elicit “yes” or “no” responses from the model. This binary re-

as a specific case of multi-choice question, where the options are limited to “(A) Yes” and “(B) No.” This simplification retains the benefits of the multi-choice question format while providing a straightforward way to measure binary decisions.

Using accuracy as the evaluation metric for both multichoice questions and Yes-or-No questions ensures consistency in how we assess the model’s performance. Accuracy, defined as the proportion of correctly answered questions, provides a clear and intuitive measure of how well the model understands the given inputs. The adoption of the multichoice question based evaluation design supports the development of a more comprehensive evaluation framework. The incorporation of both simple Yes-or-No questions and more complex multi-choice questions ensures that the evaluation covers both basic and advanced aspects of LVLM’s understanding.

Counting

Position

Sensory Reasoning

720

1423

Existence

SPARK

Contextual Reasoning

General Description

#### Evaluation on Multi-vision Sensor Tasks

Our instruction dataset was collected according to two multi-vision tasks: multi-vision perception and multi-vision reasoning. As illustrated in Figure 2, first of all, multi-vision perception focuses on the LVLM’s ability to accurately interpret and identify objects, scenes, and relationships from various multi-vision inputs. This involves tasks such as object detection, image classification, scene recognition, and relationship detection, where the model must process and understand the content of images from multiple vision sensors. The goal is to ensure that the model can consistently recognize and categorize visual elements across different contexts from different vision sensors. On the other hand, multi-vision reasoning requires the model to not only perceive but also make inferences based on the multi-vision sensory data. This involves higher-order cognitive tasks such as understanding relationships between objects, prediction of intent of sensor use, and understanding sensor knowledge. For instance, the model might need to infer the cause of an event depicted in an image sequence or predict the purpose of a captured image. Multi-vision reasoning tests the LVLM’s capability to integrate multi-vision information with contextual sensory knowledge, making logical deductions that go beyond mere perception.

- Figure 3: Distribution of data sources of the SPARK benchmark. In SPARK, we demonstrate six core multi-vision sensory tasks in the inner ring, and the outer ring displays the number of samples for each specific task.

sponse format simplifies the evaluation process, allowing for clear, objective performance measurement. As a result, each instruction comprises two parts: a brief, targeted question and an explanation corresponding to either “yes” or “no.” This structure ensures that the LVLM’s comprehension can be precisely assessed. For every test image, two instructions are manually crafted, each posing a different question to the model. These questions are designed to test different aspects of the image’s content and context. The rationale behind this approach is to ensure that the model’s answers are not based on chance. When the LVLMs correctly answer both questions, it demonstrates an understanding of the image and its related information, rather than merely guessing.

In addition, we also introduce a multi-vision sensor understanding evaluation design based on multi-choice questions. This format presents questions with a set of predetermined choices, allowing respondents to select the correct options. The multi-choice question format is advantageous for several reasons. First, it enables efficient grading and analysis of responses, as answers can be objectively evaluated against a fixed set of possible responses. Also, the multi-choice question format allows for precise control over the difficulty level of the questions. By varying the validity of each option, we can create questions that test different levels of understanding and comprehension. For example, including more plausible but incorrect options can increase the difficulty, ensuring that only models with a deeper understanding can consistently choose the correct answer. This flexibility in question design makes multi-choice questions a powerful tool for assessing the nuanced capabilities of multi-vision sensor systems. Furthermore, the Yes-or-No format can be seen

##### Multi-vision Perception

Multi-vision perception is the foundational process by which Large Vision-Language Models (LVLMs) analyze images captured by various multi-vision sensors, including RGB, thermal, depth, and X-ray images. This process involves recognizing and interpreting the fundamental elements within each visual input based on cognitive science (Kahneman, Treisman, and Gibbs 1992; Broadbent 2013).

- • Existence: LVLMs can identify and list common objects present in the image, such as people, vehicles, animals, furniture, and so on.
- • Count: LVLMs can count the number of identified objects or entities, providing a quantitative understanding of the scene.

Multi-vision Reasoning Open Source Large-scale Vision-Language Models

General Description

Multi-vision Perception

Contextual Reasoning

Sensory Reasoning

Models Vision Sensors Existence Count Position

RGB 93.9 68.5 62.6 97.9 80.7 95.1 97.2 96.1 Qwen-VL-Chat

Thermal 86.1 66.9 59.3 95.3 76.9 90.3 83.5 86.9 (Bai et al. 2023)

Depth 76.6 59.6 53.3 84.9 68.6 78.1 68.4 73.3 XR 68.0 71.3 55.1 74.1 67.1 81.8 74.7 78.3 RGB 94.2 75.5 59.8 96.9 81.6 88.7 94.8 91.8

LLAVA-v1.5-7B

Thermal 93.3 76.1 62.4 95.1 81.7 85.5 51.0 68.2 (Liu et al. 2023a)

Depth 87.1 70.7 53.3 93.7 76.2 87.4 73.8 80.6 XR 74.2 57.4 67.4 72.3 67.8 62.1 50.7 56.4 RGB 96.5 73.4 61.4 97.2 82.1 98.0 97.2 97.6

CogVLM-Chat

Thermal 94.9 76.1 64.6 96.2 82.9 96.2 59.0 77.6 (Wang et al. 2023)

Depth 94.9 76.1 64.5 96.5 83.0 90.1 71.7 80.9 XR 86.1 72.8 61.6 79.4 74.9 90.9 84.0 87.5 RGB 97.2 78.5 72.2 97.9 86.4 98.0 99.5 98.8

Meteor-7B

Thermal 93.5 68.9 71.7 95.3 82.3 90.9 62.0 76.4 (Lee et al. 2024b)

Depth 83.5 65.9 62.2 91.6 75.8 89.5 77.3 83.4 XR 79.5 70.6 63.8 76.6 72.6 86.4 84.0 85.2 RGB 96.9 81.2 69.3 96.5 85.9 98.0 99.5 98.8

TroL-7B

Thermal 93.9 72.8 68.1 92.8 81.9 94.1 65.5 79.8 (Lee et al. 2024a)

Depth 83.3 67.7 67.3 90.7 77.2 84.8 73.8 79.3 XR 82.8 69.1 71.0 78.7 75.4 83.3 84.0 83.7 RGB 96.5 76.9 69.3 98.6 85.3 98.6 99.5 99.1

IXC2.5-VL-7B

Thermal 93.0 70.6 66.8 95.5 81.5 92.5 60.0 76.2 (Zhang et al. 2024)

Depth 86.1 59.9 59.4 93.3 74.7 90.6 74.3 82.4 XR 86.1 73.5 63.8 76.6 75.0 89.4 88.0 88.7 RGB 97.2 78.3 72.4 97.9 86.5 97.6 99.1 98.3

InternVL2-8B

Thermal 90.5 75.8 61.1 93.7 80.3 94.6 61.5 78.1 (OpenGVLab 2024)

Depth 83.0 60.2 60.3 91.4 73.7 86.9 79.9 83.5 XR 92.7 77.9 71.7 84.9 81.8 89.4 82.7 86.0

Closed Source Large-scale Vision-Language Models

RGB 94.6 79.6 65.2 95.3 83.7 97.6 98.6 98.1 Gemini 1.5 Pro

Thermal 91.4 73.6 68.8 93.9 81.9 90.3 93.0 91.7 (Team et al. 2024)

Depth 87.8 73.7 62.6 94.2 79.6 78.0 88.4 83.2 XR 89.9 81.6 63.0 82.0 79.2 92.4 88.0 90.2 RGB 95.1 79.0 69.7 95.8 84.9 99.5 97.2 98.3

Claude 3.5 Sonnet

Thermal 92.1 79.2 62.9 95.0 82.3 94.1 85.0 89.6 (Anthropic 2024)

Depth 72.9 67.7 55.6 84.4 70.2 86.4 75.5 80.9 XR 83.2 76.5 74.6 83.5 79.5 93.9 82.7 88.3 RGB 96.9 80.9 71.4 97.4 86.7 98.5 98.6 98.6

GPT-4o

Thermal 96.1 75.6 71.4 98.2 85.3 95.2 92.0 93.6 (OpenAI 2024)

Depth 87.6 77.3 71.0 94.4 82.6 95.8 85.8 90.8 XR 91.9 83.8 65.2 85.6 81.7 95.5 82.7 89.1

- Table 1: Evaluation results of different models on SPARK benchmark. Accuracy is the metric. “Multi-vision Perception” shows the average performance on four dimensions (Existence, Count, Position, and General Description) for evaluating visual perception, and “Multi-vision Reasoning” shows the average performance on two dimensions (Contextual Reasoning and Sensory Reasoning) for evaluating vision sensory understanding. LVLMs are sorted in ascending order of release date.

- • Position: LVLMs can determine the spatial arrangement of objects within the image, noting their positions relative to one another.
- • General Description: LVLMs are also equipped to generate nuanced descriptions of the overall scene depicted in an image. They can articulate what is happening, identify objects, and provide factual information that enhances the understanding of the image itself.

At the perception stage, LVLMs focus on extracting essential information directly from raw image data captured by multi-vision sensors. This foundational perception is critical for all subsequent reasoning tasks, serving as the foundation upon which more complex interpretations are built.

##### Multi-vision Reasoning

Multi-vision reasoning is where LVLMs truly showcase their advanced capabilities. Beyond simply perceiving im-

|Vision Sensors<br><br>|RGB|Thermal<br><br>|Depth|XR|ALL|
|---|---|---|---|---|---|
|Models<br><br>|Multi-Vison Perception<br><br>Multi-Vision Reasoning<br><br>|Multi-Vison Perception<br><br>Multi-Vision Reasoning|Multi-Vison Perception<br><br>Multi-Vision Reasoning<br><br>|Multi-Vison Perception<br><br>Multi-Vision Reasoning| |

Open Source Large-scale Vision-Language Models

LLaVA-v1.5-7B (Liu et al. 2023a)

81.6 91.8 81.7 68.2 76.2 80.6 67.8 56.4 75.6

Qwen-VL-Chat (Bai et al. 2023)

80.7 96.1 76.9 86.9 68.6 73.3 67.1 78.3 78.5

Meteor-7B (Lee et al. 2024b)

86.4 98.8 82.3 76.4 75.8 83.4 72.6 85.2 82.6

TroL-7B (Lee et al. 2024a)

85.9 98.8 81.9 79.8 77.2 79.3 75.4 83.7 82.8

IXC2.5-VL-7B (Zhang et al. 2024)

85.3 99.1 81.5 76.2 74.7 82.4 75.0 88.7 82.9

CogVLM-Chat (Wang et al. 2023)

82.1 97.6 82.9 77.6 83.0 80.9 74.9 87.5 83.3

InternVL2-8B (OpenGVLab 2024)

86.5 98.3 80.3 78.1 73.7 83.5 81.8 86.0 83.5

Closed Source Large-scale Vision-Language Models

Claude 3.5 Sonnet (Anthropic 2024)

84.9 98.3 82.3 89.6 70.2 80.9 79.5 88.3 84.3

Gemini 1.5 Pro (Team et al. 2024)

83.7 98.1 81.9 91.7 79.6 83.2 79.2 90.2 85.9

GPT-4o (OpenAI 2024)

###### 86.7 98.6 85.3 93.6 82.6 90.8 81.7 89.1 88.5

- Table 2: Leaderboards of 10 advanced leading LVLMs on proposed SPARK benchmark according to different multi-vision sensors. Accuracy is the metric and the best accuracy is denoted in bold and underlined. LVLMs are sorted in ascending order of overall accuracy (ALL).

ages, LVLMs can engage in logical reasoning to derive deeper insights and make informed decisions. This distinguishes the recent LVLMs from traditional computer vision models, which primarily focus on understanding and interacting with the real world.

- • Contextual reasoning: LVLMs can utilize fundamental knowledge and contextual clues to make judgments about a given scenario. This type of reasoning allows LVLMs to refer to the underlying basis of physical sensor knowledge and ensure that the reasoning process remains consistent with the context provided by the image and the associated information.
- • Sensory reasoning: A more complex reasoning ability requires LVLMs to map 2D image data to the physical meanings associated with different multi-vision sensors. This process not only involves processing the raw data from images but also integrates it with contextual information about the underlying physical sensor knowledge in the real world. By combining fundamental sensor information, LVLMs can derive conclusions that are both accurate and contextually relevant. Sensory reasoning requires a deep understanding of the knowledge underlying the physical meaning of multi-vision sensor data. This goes beyond surface-level image recognition, demanding that LVLMs make sense of the sensor data in a way that reflects real-world physics and usage scenarios.

Next, we integrate both visual and textual inputs into GPT-4, guided by meticulously crafted prompts. These prompts are specifically designed to align with various evaluation dimensions, ensuring that the generated questions are

both relevant and focused. To further enhance the quality of the benchmark, we introduce an additional filtering step. In the final stages of development, human annotators play a crucial role, selecting the correct answers and categorizing the questions according to their respective evaluation dimensions.

#### Experiment

##### Implementation Details

Dataset Collection We collect six subsets for each multisensor vision task type, together with 4k images and 6k unique questions and answers. These instructions are built from five public datasets: MS-COCO (Lin et al. 2015), M3FD (Liu et al. 2022), Dog&People (Roboflow 2022), RGB-D scene dataset (Cho et al. 2021), and UNIFESP Xray Body Part Classifier Competition dataset (Eduardo Farina 2022). The MS-COCO dataset is a commonly used object detection dataset that contains RGB images with finegrained object bounding boxes, categories, and attribute annotations. We sampled 1.2k images from validation dataset. Furthermore, for thermal sensor datasets, we sampled 1.2k images from two different thermal datasets (M3FD and Dog&People) in order to collect a thermal dataset covering the widest possible range of diverse situations and objects. Additionally, we sampled 1.2k images from RGB-D scene dataset (Cho et al. 2021) for depth sensor because it covers a variety of indoor and outdoor scenes. Finally, we sampled 0.4k images from the public X-ray body part dataset for the XR sensor dataset because of the diversity of multiple human body parts. We described the overall distribution of data

sources of the SPARK benchmark in Figure 3.

Large Vision Language Models In our evaluation, we selected 10 state-of-the-art (SOTA) Large Vision-Language Models (LVLMs) that represent the leading edge of current research. These models were chosen to provide a comprehensive assessment of the capabilities and performance of both open-source and closed-source LVLMs across a variety of multi-vision sensor tasks on the SPARK benchmark.

- • Open source: CogVLM-Chat (Wang et al. 2023), LLAVA-v1.5-7B (Liu et al. 2023b), InternVL28B (OpenGVLab 2024), TroL-7B (Lee et al. 2024a), Meteor-7B (Lee et al. 2024b), IXC2.5-VL-7B (Zhang et al. 2024), Qwen-VL-Chat (Bai et al. 2023)
- • Closed source: GPT-4o (OpenAI 2024), Claude 3.5 Sonnet (Anthropic 2024), Gemini-Pro1.5 (Team et al. 2024)

##### Experiment Result

In this section, we conduct a comprehensive evaluation using the proposed SPARK benchmark, a rigorous framework designed to assess the capabilities of Large Vision-Language Models (LVLMs) in two target tasks: Multi-vision Perception and Multi-vision Reasoning. Multi-vision Perception presents the averaged performance on four dimensions for evaluating visual perception. Meanwhile, Multi-vision Reasoning demonstrates the averaged performance on two dimensions for evaluating the LVLMs’ ability to understand and reason about multi-vision sensory data.

As shown in Table 1, the evaluation revealed that performance varies significantly depending on the type of multivision sensor used to capture the input images. LVLMs generally perform well in simple Multi-vision perception tasks such as generating general descriptions, but more complex reasoning tasks like Multi-vision Reasoning reveal significant differences in model capabilities. Since they mainly trained with general RGB images, the performance of multivision perception and reasoning in RGB sensor is consistently maintained at high levels. However, the performance of LVLMs drops noticeably when dealing with images captured using thermal, depth, and X-ray(XR) sensors. This decline is particularly evident in the Multi-vision Reasoning task, especially in Sensory Reasoning.

Sensory Reasoning requires LVLMs to not only recognize and describe images but also to understand the physical principles underlying the sensor data. For example, interpreting thermal data involves understanding heat signatures, while depth data requires an understanding of the need for spatial geometry beyond simple 2D interpretation. The experiment demonstrates LVLMs’ limited proficiency in interpreting and mapping sensor data to its physical meaning.

Table 2 provides a clear comparison of the performance of various LVLMs across different multi-vision sensors and tasks. It highlights the strengths and weaknesses of each model, particularly the advantage that closed-source models have in maintaining high performance across more complex reasoning tasks with diverse vision sensor types. Considering the overall accuracy score (ALL), GPT-4o excels in the proposed SPARK benchmark.

|Model<br><br>|Vision Sensor<br><br>|Sensor Reasoning w/o Sensor Info.|Sensor Reasoning w/ Sensor Info.<br><br>|∆|
|---|---|---|---|---|
|LLAVA-v1.5-7B (Liu et al. 2023b) Open source LVLM|Thermal<br><br>Depth<br><br>XR|51.0 73.8 50.7<br><br>|81.0 87.6 54.0|+30.0 +13.8 +3.3<br><br>|
|TroL-7B (Lee et al. 2024a) Open source LVLM|Thermal<br><br>Depth<br><br>XR<br><br>|65.5 73.8 84.0|97.0 99.1 84.0<br><br>|+31.5 +25.3 -|
|InternVL2-8B (OpenGVLab 2024) Open source LVLM|Thermal<br><br>Depth<br><br>XR<br><br>|61.5 79.9 82.7|85.5 99.6 85.3<br><br>|+24.0 +19.7 +2.6<br><br>|
|Claude 3.5 Sonnet (Anthropic 2024) Closed source LVLM<br><br>|Thermal Depth XR<br><br>|85.0 75.5 82.7|96.5 99.6 82.7<br><br>|+11.5 +24.1 -|
|GPT-4o (OpenAI 2024) Closed source LVLM|Thermal<br><br>Depth<br><br>XR|92.0 85.8 82.7<br><br>|94.0 99.6 84.0|+2.0 +13.8 +1.3<br><br>|

Table 3: Ablation study on sensor reasoning performance change whether the sensor information is given. We choose three LVLMs from open source and two from closed source.

##### Ablation study

In the previous section, we observed that LVLMs frequently struggle to accurately infer the purpose or context of an image when the data is sourced from multi-vision sensors other than RGB. However, as demonstrated in Figure 1, even when the input image lacks explicit information about the sensor type, LVLMs can still identify the sensor correctly. This suggests that while LVLMs have already acquired sensorrelated knowledge through textual data, they face challenges in mapping fundamental knowledge to real-world scenarios.

Thus, in Table 3, we conducted an ablation experiment on data-centric enhancement by adding sensor information as a text prompt at the beginning of the question (“This is a {Thermal, Depth, X-Ray} image.”) and measured the sensory reasoning performance change. The experiment demonstrated that sensor information can enhance the reasoning capabilities of LVLMs, particularly for thermal and depth images, while XR data showed the least impact. This implies that LVLM models, including GPT-4o, are not fully utilizing the knowledge they already possess to understand multi-vision sensory data.

#### Conclusion

In this study, we focus on evaluating the ability of Large Vision-Language Models (LVLMs) to understand and process multi-vision sensory inputs. As LVLMs are increasingly deployed in real-world applications, their ability to accurately interpret and reason about data from diverse vision sensors has become crucial. To address this, we propose an evaluation benchmark called SPARK, which generates instruction tuning samples aimed at specific physical sensor understanding in various question-and-answer formats. Through extensive experiments, we assess the performance of understanding sensory knowledge in the latest state-ofthe-art LVLMs handling multi-vision input. We believe this approach, integrating a sensory knowledge annotated evaluation benchmark paves the way for promising future applications of LVLMs.

#### References

Anthropic. 2024. Claude 3.5 sonnet. https://www.anthropic. com/news/claude-3-5-sonnet.

Bai, J.; Bai, S.; Yang, S.; Wang, S.; Tan, S.; Wang, P.; Lin, J.; Zhou, C.; and Zhou, J. 2023. Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond. arXiv:2308.12966.

Bazi, Y.; Rahhal, M. M. A.; Bashmal, L.; and Zuair, M.

- 2023. Vision–language model for visual question answering in medical imagery. Bioengineering, 10(3): 380.

Broadbent, D. E. 2013. Perception and communication. Elsevier.

Chen, Z.; Wang, W.; Tian, H.; Ye, S.; Gao, Z.; Cui, E.; Tong, W.; Hu, K.; Luo, J.; Ma, Z.; et al. 2024. How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites. arXiv preprint arXiv:2404.16821.

Cho, J.; Min, D.; Kim, Y.; and Sohn, K. 2021. DIML/CVL RGB-D Dataset: 2M RGB-D Images of Natural Indoor and Outdoor Scenes. arXiv:2110.11590.

Eduardo Farina, M. P., FelipeKitamura. 2022. UNIFESP Xray Body Part Classifier Competition.

Fu, C.; Chen, P.; Shen, Y.; Qin, Y.; Zhang, M.; Lin, X.; Yang, J.; Zheng, X.; Li, K.; Sun, X.; Wu, Y.; and Ji, R. 2024. MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models. arXiv:2306.13394.

Girdhar, R.; El-Nouby, A.; Liu, Z.; Singh, M.; Alwala, K. V.; Joulin, A.; and Misra, I. 2023. Imagebind: One embedding space to bind them all. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 15180–15190.

Gunjal, A.; Yin, J.; and Bas, E. 2024. Detecting and Preventing Hallucinations in Large Vision Language Models. arXiv:2308.06394.

Kahneman, D.; Treisman, A.; and Gibbs, B. J. 1992. The reviewing of object files: Object-specific integration of information. Cognitive psychology, 24(2): 175–219.

Koh, J. Y.; Salakhutdinov, R.; and Fried, D. 2023. Grounding language models to images for multimodal inputs and outputs. In International Conference on Machine Learning, 17283–17300. PMLR.

Lee, B.-K.; Chung, S.; Kim, C. W.; Park, B.; and Ro, Y. M.

- 2024a. TroL: Traversal of Layers for Large Language and Vision Models. arXiv:2406.12246.

Lee, B.-K.; Kim, C. W.; Park, B.; and Ro, Y. M. 2024b. Meteor: Mamba-based Traversal of Rationale for Large Language and Vision Models. arXiv:2405.15574.

- Lee, B.-K.; Park, B.; Kim, C. W.; and Ro, Y. M. 2024c. CoLLaVO: Crayon Large Language and Vision mOdel.

- arXiv:2402.11248.

- Lee, B.-K.; Park, B.; Kim, C. W.; and Ro, Y. M. 2024d. MoAI: Mixture of All Intelligence for Large Language and Vision Models. arXiv:2403.07508.

Li, B.; Wang, R.; Wang, G.; Ge, Y.; Ge, Y.; and Shan, Y. 2023a. Seed-bench: Benchmarking multimodal

llms with generative comprehension. arXiv preprint arXiv:2307.16125.

Li, J.; and Lu, W. 2024. A Survey on Benchmarks of Multimodal Large Language Models. arXiv preprint arXiv:2408.08632.

Li, Y.; Du, Y.; Zhou, K.; Wang, J.; Zhao, W. X.; and Wen, J.R. 2023b. Evaluating Object Hallucination in Large VisionLanguage Models. arXiv:2305.10355.

Lin, T.-Y.; Maire, M.; Belongie, S.; Bourdev, L.; Girshick, R.; Hays, J.; Perona, P.; Ramanan, D.; Zitnick, C. L.; and Doll´ar, P. 2015. Microsoft COCO: Common Objects in Context. arXiv:1405.0312.

Liu, H.; Li, C.; Li, Y.; and Lee, Y. J. 2023a. Improved Baselines with Visual Instruction Tuning.

Liu, H.; Li, C.; Li, Y.; Li, B.; Zhang, Y.; Shen, S.; and Lee, Y. J. 2024a. LLaVA-NeXT: Improved reasoning, OCR, and world knowledge.

Liu, H.; Li, C.; Wu, Q.; and Lee, Y. J. 2023b. Visual Instruction Tuning.

Liu, J.; Fan, X.; Huang, Z.; Wu, G.; Liu, R.; Zhong, W.; and Luo, Z. 2022. Target-aware Dual Adversarial Learning and a Multi-scenario Multi-Modality Benchmark to Fuse Infrared and Visible for Object Detection. arXiv:2203.16220.

Liu, Y.; Duan, H.; Zhang, Y.; Li, B.; Zhang, S.; Zhao, W.; Yuan, Y.; Wang, J.; He, C.; Liu, Z.; Chen, K.; and Lin, D. 2024b. MMBench: Is Your Multi-modal Model an Allaround Player? arXiv:2307.06281.

Mao, J.; Qian, Y.; Zhao, H.; and Wang, Y. 2023. Gptdriver: Learning to drive with gpt. arXiv preprint arXiv:2310.01415.

OpenAI. 2024. Hello GPT-4o. https://openai.com/index/ hello-gpt-4o/.

OpenGVLab. 2024. InternVL2: Better than the Best—Expanding Performance Boundaries of Open-Source Multimodal Models with the Progressive Scaling Strategy. https://internvl.github.io/blog/2024-07-02-InternVL-2.0/.

Ren, S.; Yao, L.; Li, S.; Sun, X.; and Hou, L. 2024. Timechat: A time-sensitive multimodal large language model for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14313–14323.

Roboflow. 2022. thermal dogs and people x6ejw Dataset. https://universe.roboflow.com/object-detection/ thermal-dogs-and-people-x6ejw. Visited on 2023-03-29.

Shi, Y.; Gao, Y.; Lai, Y.; Wang, H.; Feng, J.; He, L.; Wan, J.; Chen, C.; Yu, Z.; and Cao, X. 2024. Shield: An evaluation benchmark for face spoofing and forgery detection with multimodal large language models. arXiv preprint arXiv:2402.04178.

Su, Y.; Lan, T.; Li, H.; Xu, J.; Wang, Y.; and Cai, D. 2023. Pandagpt: One model to instruction-follow them all. arXiv preprint arXiv:2305.16355.

Team, G.; Georgiev, P.; Lei, V. I.; Burnell, R.; Bai, L.; Gulati, A.; Tanzer, G.; Vincent, D.; Pan, Z.; Wang, S.;

et al. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint

- arXiv:2403.05530.

Wang, W.; Lv, Q.; Yu, W.; Hong, W.; Qi, J.; Wang, Y.; Ji, J.; Yang, Z.; Zhao, L.; Song, X.; et al. 2023. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079.

Xu, P.; Shao, W.; Zhang, K.; Gao, P.; Liu, S.; Lei, M.; Meng, F.; Huang, S.; Qiao, Y.; and Luo, P. 2023. LVLM-eHub: A Comprehensive Evaluation Benchmark for Large VisionLanguage Models. arXiv:2306.09265.

Xu, Z.; Zhang, Y.; Xie, E.; Zhao, Z.; Guo, Y.; Wong, K.Y. K.; Li, Z.; and Zhao, H. 2024. Drivegpt4: Interpretable end-to-end autonomous driving via large language model. IEEE Robotics and Automation Letters.

Yarom, M.; Bitton, Y.; Changpinyo, S.; Aharoni, R.; Herzig, J.; Lang, O.; Ofek, E.; and Szpektor, I. 2024. What you see is what you read? improving text-image alignment evaluation. Advances in Neural Information Processing Systems, 36.

Ye, J.; Hu, A.; Xu, H.; Ye, Q.; Yan, M.; Dan, Y.; Zhao, C.; Xu, G.; Li, C.; Tian, J.; et al. 2023. mplug-docowl: Modularized multimodal large language model for document understanding. arXiv preprint arXiv:2307.02499.

Yu, W.; Yang, Z.; Li, L.; Wang, J.; Lin, K.; Liu, Z.; Wang, X.; and Wang, L. 2023. MM-Vet: Evaluating Large Multimodal Models for Integrated Capabilities. arXiv:2308.02490.

Zhang, P.; Dong, X.; Zang, Y.; Cao, Y.; Qian, R.; Chen, L.; Guo, Q.; Duan, H.; Wang, B.; Ouyang, L.; Zhang, S.; Zhang, W.; Li, Y.; Gao, Y.; Sun, P.; Zhang, X.; Li, W.; Li, J.; Wang, W.; Yan, H.; He, C.; Zhang, X.; Chen, K.; Dai, J.; Qiao, Y.; Lin, D.; and Wang, J. 2024. InternLM-XComposer-2.5: A Versatile Large Vision Language Model Supporting LongContextual Input and Output. arXiv:2407.03320.

