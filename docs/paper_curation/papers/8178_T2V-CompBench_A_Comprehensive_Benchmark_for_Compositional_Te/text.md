## T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation

Kaiyue Sun1 Kaiyi Huang1 Xian Liu2 Yue Wu3 Zihan Xu1 Zhenguo Li3 Xihui Liu1 1The University of Hong Kong 2The Chinese University of Hong Kong 3Huawei Noah’s Ark Lab

Project Page Evaluation Code Leaderboard

# arXiv:2407.14505v2[cs.CV]15Jan2025

[Figure 1]

[Figure 2]

(a) Prompt Suite (b) Evaluation Metrics (c) Benchmarking T2V Models

| |MLLM-based evaluation metrics| |
|---|---|---|
| | | |

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

objectsMulti

Describe the video.

Color

Single object

[Figure 7]

Shape Texture Human

The video shows...

[Figure 8]

Evaluate the alignment with text.

[Figure 9]

[Figure 10]

Consistent attribute binding

[Figure 11]

[Figure 12]

[Figure 13]

Generative numeracy

score:3, explanation: this video...

Social

Attr change

Detection-based evaluation metrics

[Figure 14]

[Figure 15]

Dynamic attribute binding

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Object interactions

[Figure 23]

d1 d2

Physical

Combined

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Spatial relationships

Action binding

Uncommon

Left Right

Object locations Object depths (3D relationships)

score

[Figure 28]

[Figure 29]

| |Tracking-based evaluation metrics| |
|---|---|---|
| | | |

Motion binding

Above Below

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Common

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

in front of

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

behind

Horizontal

Vertical

Foreground motion Background motion

score

Figure 1. Overview of T2V-CompBench. We propose T2V-CompBench, a comprehensive compositional text-to-video generation benchmark that consists of seven categories: consistent attribute binding, dynamic attribute binding, spatial relationships, motion binding, action binding, object interactions, and generative numeracy. We propose three types of evaluation metrics: MLLM-based, Detectionbased, and Tracking-based metrics. We benchmark various text-to-video generation models.

#### Abstract

seven proposed categories with 1400 text prompts. The effectiveness of the proposed metrics is verified by correlation with human evaluations. We also benchmark various text-to-video generative models and conduct in-depth analysis across different models and various compositional categories. We find that compositional text-to-video generation is highly challenging for current models, and we hope our attempt could shed light on future research in this direction.

Text-to-video (T2V) generative models have advanced significantly, yet their ability to compose different objects, attributes, actions, and motions into a video remains unexplored. Previous text-to-video benchmarks also neglect this important ability for evaluation. In this work, we conduct the first systematic study on compositional text-to-video generation. We propose T2V-CompBench, the first benchmark tailored for compositional text-to-video generation. T2V-CompBench encompasses diverse aspects of compositionality, including consistent attribute binding, dynamic attribute binding, spatial relationships, motion binding, action binding, object interactions, and generative numeracy. We further carefully design evaluation metrics of multimodal large language model (MLLM)-based, detectionbased, and tracking-based metrics, which can better reflect the compositional text-to-video generation quality of

#### 1. Introduction

Text-to-video (T2V) generation has made significant progress in recent years [2, 3, 14, 17, 18, 22, 42, 59, 64, 65, 71, 72, 82]. However, generating videos that accurately depict multiple objects, attributes, and motions in complex and dynamic scenes based on fine-grained text prompts remains a challenging task. In this work, we aim to conduct a

systematic study on compositional T2V generation.

Compositional text-to-image (T2I) generation, which aims to compose multiple objects, attributes, and their relationships into complex scenes, has been widely studied in previous methods [5, 11, 36]. Benchmarks for compositional T2I generation [20] have been accepted as an important evaluation dimension for T2I foundation models [1, 7, 10]. However, most works on T2V generation focus on generating videos with simple text prompts, neglecting the significance of compositional T2V generation. Moreover, existing video generation benchmarks [21, 39, 40] primarily evaluate video quality, motion quality, and text-video alignment with single-object text prompts, and benchmarks for compositional T2V generation have not been systematically and extensively investigated in previous literature.

To this end, we propose T2V-CompBench, a comprehensive benchmark designed for compositional text-to-video generation. This benchmark emphasizes compositionality through multiple objects with attributes, quantities, actions, interactions, and spatio-temporal dynamics. We design a prompt suite composed of seven categories, where each category consists of 200 text prompts for video generation. When constructing the prompts, we emphasize temporal dynamics and guarantee that each prompt contains at least one active verb. The seven categories are as follows and examples are illustrated in Figure 2: 1) Consistent attribute binding. This category includes prompts featuring two objects, each with a distinct attribute. The attributes associated with each object are consistent throughout the video. 2) Dynamic attribute binding. Prompts in this category focus on dynamic attribute binding for objects, where the attributes change over time. 3) Spatial relationships. In this category, each prompt mentions two objects and specifies the spatial relationship between them. 4) Motion binding. Each prompt in this category includes one or two objects and a moving direction is specified for each object. 5) Action binding. Prompts in this category describe two objects, each with a distinct action. 6) Object interactions. This category tests the models’ abilities to understand and generate dynamic interactions between multiple objects, including physical interactions and social interactions. 7) Generative numeracy. The text prompts in this category include one or two objects with quantities ranging from one to eight.

Another challenge lies in the evaluation of compositional T2V generation. Commonly used metrics, such as Inception Score [58], Fr´echet Inception Distance (FID) [16], Fr´echet Video Distance (FVD) [63], and CLIPScore [15], cannot fully reflect the compositionality of T2V generation models. Evaluating compositionality of T2V models requires a fine-grained understanding of not only objects and attributes in each frame but also the dynamics and motions across frames. It is orders of magnitude more complex than evaluating T2I models.

To address this challenge, we take temporal dynamics across frames into consideration and design different metrics to evaluate different categories in the benchmark. Specifically, we design multimodal large language model (MLLM)-based metrics, including image-LLM and videoLLM, to evaluate consistent attribute binding, dynamic attribute binding, action binding, and object interactions. We devise detection-based metrics to evaluate spatial relationships and generative numeracy. We propose tracking-based metrics to evaluate motion binding. The effectiveness of our proposed metrics is validated by computing the correlation with human evaluations. We evaluate various T2V generation models on T2V-CompBench and analyze their performances on different compositional categories.

The contributions of our paper are three-fold. 1) To our best knowledge, we are the first to propose a benchmark for compositional text-to-video generation, featuring seven categories with 1400 text prompts. 2) We propose comprehensive evaluation metrics for the seven categories and verify their effectiveness through correlation with human evaluations. 3) We benchmark various T2V models and provide a systematic study with insightful analysis, which will inspire future research in this direction.

#### 2. Related Work

##### 2.1. Text-to-video Generation.

Recently, diffusion models have gained significant attention in the field of text-to-video generation, building on the success of text-to-image models. Existing text-tovideo diffusion models can be roughly categorized into two types, namely the diffusion unet-based [3, 14, 17, 22, 42, 59, 65, 82] and the Diffusion Transformer (DiT)based [19, 33, 43, 61, 78]. In this paper, we evaluate a list of text-to-video diffusion models, including officially open-sourced models: ModelScope [65], ZeroScope, LVD [32], AnimateDiff [13], MagicTime [79], Show1 [81], VideoCrafter2 [6], VideoTetris [62], Vico [77], T2VTurbo-V2 [29], Latte [43], Open-Sora [19], Open-SoraPlan [33], CogVideoX-5B [78], and Mochi [61], and commercial models: Pika-1.0 [51], Gen-2 [56], Gen-3 [57], Dreamina 1.2 [4], PixVerse-V3 [52], and Kling-1.0 [26]. This comprehensive evaluation ensures the diversity of T2V approaches and provides insights into their capabilities and limitations across different architectures.

##### 2.2. Compositional Text-to-image Generation.

Recent studies have delved into compositionality in text-toimage generation [5, 8, 11, 12, 20, 24, 30–32, 36, 38, 44, 49, 50, 54, 66, 69, 73, 76]. T2I-CompBench [20] proposed the first comprehensive benchmark to evaluate compositionality in text-to-image models, focusing on attributes binding, relationships, and complex compositions. While these

[Figure 46]

[Figure 47]

###### Consistent Attribute Binding

Category Information

[Figure 48]

###### Meta data

{

[Figure 49]

[Figure 50]

A blue car drives past a white picket fence on a

Example

"object 1": "a penguin",

###### Prompt:

###### Human verification

sunny day.

Prompt

- "action 1": "a penguin

waddles on the ice",

"object 2": "a camel",

- "action 2": "a camel treks

###### Nouns: ["dog", "cat"…]. Verbs: ["walk", "run"…]

“A penguin waddles

[Figure 51]

Obj. number = 2

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Generate prompts featuring two objects, each

on the ice, a camel treks by.”

[Figure 58]

Action verbs = 2

Generated Video

[Figure 59]

[Figure 60]

with a natural active verb. Select the two sets

[Figure 61]

Common/uncommon object-activity pairs

by T2V Models

of object nouns and verbs from the lists

by"

provide above.

}

MLLM-based metric: Grid-LLaVA

[Figure 62]

[Figure 63]

[Figure 64]

###### Dynamic Attribute Binding

###### Spatial Relationships

###### Motion Binding

[Figure 65]

[Figure 66]

[Figure 67]

A timelapse of a leaf transitioning from green to bright red as autumn progresses.

A robot walking from right to left across the moon with a car driving left to right in the background.

Example Prompt

Example Prompt

Example Prompt

A toddler walking on the left of a dog in a park.

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

Generated Video

Generated Video

Generated Video

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

by T2V Models

by T2V Models

by T2V Models

MLLM-based metric: D-LLaVA

Detection-based metric: G-Dino

Tracking-based metric: DOT

[Figure 89]

[Figure 90]

[Figure 91]

###### Action Binding

###### Object Interactions

###### Generative Numeracy

[Figure 92]

[Figure 93]

[Figure 94]

Example Prompt

Example Prompt

Example Prompt

A penguin waddles on the ice, a camel treks by.

Two cars collide at an intersection.

Seven oranges roll on the counter near three cups.

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

Generated Video

Generated Video

Generated Video

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

by T2V Models

by T2V Models

by T2V Models

MLLM-based metric: Grid-LLaVA

MLLM-based metric: Grid-LLaVA

Detection-based metric: G-Dino

Figure 2. Prompt generation process and illustrations of the seven compositional categories.

evaluations are tailored exclusively to the image domain, video generation requires a deeper consideration of spatiotemporal dynamics. Our work pioneers the development of benchmarking compositional text-to-video generation.

the first benchmark for evaluating compositional text-tovideo generation, with tailored evaluation metrics that we validate through extensive human correlation studies.

#### 3. Benchmark Construction

##### 2.3. Benchmarks for Text-to-video Generation.

##### 3.1. Problem Definition and Categorization

Previous methods evaluate T2V models from the perspectives of video quality and text-video alignment. For video quality, the commonly used metrics, such as Inception Score (IS) [58] and Fr´echet Video Distance (FVD) [63] are adopted to evaluate video diversity and fidelity. For textvideo alignment, CLIPScore [15] is proposed to measure the similarity of the text prompt and the frames, using the pre-trained CLIP model [53]. However, these metrics are not suitable for complex compositional prompts.

Compositional T2V generation has not been comprehensively explored in prior research. Therefore, we first clarify the problem definition and categorization. Previous literature on compositional T2I generation [5, 11, 20] typically focus on attribute binding, object relationships and numeracy. However, in the context of compositional T2V, we need to consider the composition in both spatial and temporal dimensions. In spatial dimension, we follow the framework established in compositional T2I and define categories of consistent attribute binding, spatial relationships, and numeracy. These categories require the generated video frames consistently align with the text prompts. In temporal dimension, we introduce categories of dynamic attribute binding, motion binding, action binding, and object interactions. These categories specifically address whether the temporal dynamics of the video follow the description of the text prompts.

Recent T2V benchmarks design text prompts and metrics to evaluate the video quality and text-video alignment in open domains. VBench [21] and EvalCrafter [39] propose comprehensive benchmarks to evaluate T2V models from various perspectives. FETV [40] categorizes prompts based on major content, controllable attributes, and prompt complexity. ChronoMagic-Bench [80] evaluates T2V models’ ability in generating time-lapse videos. However, most prompts provided in these benchmarks focus on singleobject rather than composition of multiple objects. Although some of them involves evaluation dimensions that include multiple objects, their prompts, such as “a bird and a cat” in “multiple objects” dimension of VBench [21], do not reflect the dynamics in videos. ChronoMagic-Bench [80] emphasizes the generation of natural metamorphic timelapse videos, while excluding unnatural attribute-change videos, which are also essential for evaluating the design and creative capabilities of T2V models. A comprehensive definition of compositionality in text-to-video generation is currently lacking in the literature. Therefore, we introduce

##### 3.2. Prompt Categories

Consistent Attribute Binding. We define four attribute types including color, shape, texture and human-related attributes. Each prompt has two objects, two attributes, and at least one active verb, with each attribute associates with a specific object. Among all the prompts, 20% are challenging and uncommon cases that aim to test the model’s ability to generalize to unseen combinations, such as “Blue apple bouncing near a pink tree”.

###### Dynamic Attribute Binding. This category focuses on

|0<br><br>20<br><br>40<br><br>60<br><br>80<br><br>100<br><br>120<br><br>140<br><br>160<br><br>180<br><br>200<br><br>Consist-attr Dynamic-attr Spatial Motion Action Interaction Numeracy<br><br>图表标题<br><br>系列 1 系列 2 系列 3 系列 4 系列 5 系列 6 系列 7<br><br>| |Human: 20| |Combined: 50| |Behind: 33| |2-obj<br><br>left & right: 20| |Uncommon: 40| |Social: 100|2-obj: 50| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |Texture: 40| | | | | |2-obj up & down: 20| | | | | | |
| | | | | | | | | | | | | | |
| | | | | |In front: 33| |1-obj down: 40| |Common: 160| | | | |
| | | |Texture: 30| | | | | | | | |>6: 20| |
| |Shape: 40| | | | | | | | | | | | |
| | | | | |Below: 33| | | | | | | | |
| | | | | | | | | | | | |=5: 30| |
| | | |Shape: 50| | | |1-obj up: 40| | | | | | |
| |Color: 100| | | |Above: 33| | | | | |Physical: 100|=4: 30| |
| | | | | | | |1-obj right: 40| | | | | | |
| | | |Color: 70| | | | | | | | |=3: 30| |
| | | | | | | | | | | | | | |
| | | | | |Right: 34| | | | | | | | |
| | | | | | | |1-obj left: 40| | | | |=2: 30| |
| | | | | |Left: 34| | | | | | | | |
| | | | | | | | | | | | | | |
<br><br>=1: 10|
|---|

##### 3.3. Prompt Suite Generation

Vocabulary Construction. To ensure our prompt suite matches the real users’ desires, we analyze 1.67 million unique T2V prompts collected by VidProM [67] from Pika Discord channels. We use WordNet [45] to identify the metaclasses of nouns and verbs, and their distributions are visualized in Figure 4. Firstly, as our goal is to benchmark T2V models ability to compose multiple concepts, we focus on entry-level objects that occur with high frequency. Secondly, to facilitate evaluation, we select “thing” categories, i.e. individual objects that can be easily labeled with bounding boxes like “car” and “dog”, rather than “stuff” categories, i.e. objects without clear boundaries such as “sky”. Based on these principles, we analyze real user prompts to identify high-frequency nouns that belong to specific metaclasses, such as person, artifacts (e.g., conveyance and device), animals, plants, and food. We identify high-frequency verbs and attributes in a similar way. To account for the inherent dynamics of videos, we pair objects with vibrant active verbs, like those in the metaclasses of “move”, “travel”, and “act”, while avoiding static actions like “think”, “see” or “rest”. Attributes are also drawn from relevant metaclasses. In total, we collected 260 object nouns, 200 active verbs, and 80 attributes in color, shape, and texture. The prompt suite is generated using these words that reflect the interests of real users. For more details about word selection, please refer to Appendix A.1.

Figure 3. Illustration of prompt categories. We show the number of prompts for the seven categories and their respective subgroups.

how the attributes of objects change over time. For example, “Green avocado darkens to black as the tomato beside it ripens to a deep red”. We define four aspects inspired by TempCompass [41]: color & light change, shape & size change, texture change, and combined change. 80% of the prompts describe common attribute changes in real world, while the remaining 20% are less common or artificial.

Spatial Relationships. This category requires the model to generate two objects with specified spatial relationships across the video. We define six types of spatial relationships: “on the left of”, “on the right of”, “above”, “below”, “in front of”, and “behind”. We construct contrastive “left” and “right” prompts by inverting the spatial relationships.

Motion Binding. Prompts in this category contain one or two objects with specified moving directions. We define four types of moving directions: “leftwards”, “rightwards”, “upwards” and “downwards”. Each object in the prompt moves in one of the directions.

Action Binding. This category tests the models’ abilities to bind actions to corresponding objects when there are multiple objects and multiple actions described. Two objects are involved in the prompt, each engaged in an activity. This category contains 80% common prompts and 20% uncommon prompts, which can be further divided into uncommon object coexistence and uncommon object-activity pairs.

(a) Meta types of nouns (b) Meta types of verbs

Object Interactions. This category tests the models’ abilities to understand and generate dynamic interactions, including physical interactions causing motion change or state change and social interactions between live entities.

Figure 4. Word distributions of real-user prompts. We show the types of nouns and verbs of real-user prompts from VidProM [67].

Prompt Generation. Since we evaluate compositional T2V generation, the prompts must follow specified requirements. As a result, sourcing from existing dataset captions is challenging. Additionally, using a fixed template for automatic prompt generation is not ideal, since real user prompts are free-form and diverse. Therefore, we have opted to use GPT-4 [47] to generate the prompts. The 200 prompts in each category are generated by prompting GPT4 [47] with the collected high-frequency words and the specific requirements for the category. Although not all prompt categories are designed to evaluate actions and motions, we

Generative Numeracy. To analyze the models’ abilities to generate correct number of objects, we construct prompts in quantity groups. In each group, the same noun is paired with different quantities, such as “Three/Four/Five dogs run through a field”. We also construct prompts with two objects like “Three/Four cows graze in the pasture, and one/two sheep wanders/wander nearby”.

We prepare 200 prompts for each of the seven categories. Figure 3 displays the prompt subgroups along with the number of prompts associated with each subgroup.

ensure all prompts in our benchmark contain at least one active verb, to prevent the T2V model from generating static videos. GPT-4 [47] returns both prompts and the parsed meta information for the prompts that facilitates evaluation. All the generated prompts are verified by humans, and improper prompts are filtered out. For more details about prompt generation, please refer to Appendix A.2.

##### 3.4. Prompt Suite Statistics

T2V-CompBench stands out for its focus on multiple objects and temporal dynamics. Figure 5 shows the statistics on benchmark prompts: (1) In contrast to previous benchmarks, which predominantly focus on single-object, our prompts involve more than two nouns on average, with each prompt containing approximately 3.6 nouns. (2) T2VCompBench considers temporal dynamics, with all prompts containing verbs, averaging at 1.4 per prompt. (3) To prevent T2V models from being distracted by irrelevant contents, we avoid using excessively long prompts, the average length is 10.4 words, ranging from 3 to 23. (4) Since we utilize an LLM to generate free-form prompts, many auxiliary words, such as those representing time and spot, are produced in addition to the provided words we instruct the LLM to select from. As a result, there is a rich variety of nouns, verbs, and adjectives present in the generated dataset, ensuring it encompasses a diverse range of topics. For more information about T2V-CompBench prompts, please refer to Appendix A.3

Prompt

Avg. Nouns Avg. Verbs Avg. length

3.6 1.4 10.4words

Number

Dataset

Unique number of Adj.

Unique number of

Unique number of Nouns

Verbs

1512 682 344

Prompt length

- Figure 5. Statistics on benchmark prompts. Left: Prompt length distribution. Right: Prompt Suite statistics.

#### 4. Evaluation Metrics

We observe that the evaluation metrics for compositional T2I generation [20] cannot be directly adopted for evaluating compositional T2V generation, due to the large number of frames and complex spatio-temporal dynamics in videos. Most T2V models generate short videos in 2-5 seconds. For a fair comparison, we evenly extract 6 frames for MLLMbased evaluation, and 16 frames for detection-based evaluation, and sample the videos to a frame rate of 8 frames per second (FPS) for tracking-based evaluation. Figure 1(b) illustrates the three types of evaluation metrics.

##### 4.1. MLLM-based Evaluation Metrics

Multimodal Large Language Models (MLLMs) have shown great capabilities in understanding complex contents in images and videos [35, 47, 74, 83]. Inspired by their effectiveness in video understanding, we exploit MLLMs as evaluators for compositional text-to-video generation.

Video LLM-based metrics for consistent attribute binding, action binding, and object interactions. To handle the complex spatio-temporal information in videos, we investigate video LLMs such as Image Grid [23] and PLLaVA [74], which extends LLaVA [35] from single image input to multi-frame input. We empirically find that Image Grid performs better than PLLaVA in our compositional categories. Specifically, Image Grid uniformly samples 6 frames from the video to form an image grid as the input to LLaVA [35]. Additionally, we boost the ability of video LLMs and avoid hallucinations by the chain-ofthought mechanism [70] along with disentangled questions, where we first ask the MLLM to describe the video content, and then request it to grade each aspect of the text-video alignment. We denote this metric as Grid-LLaVA.

- • To evaluate consistent attribute binding, we use GPT4 [47] to parse the prompts into disentangled phrases (e.g., “A blue car drives past a white picket fence on a sunny day” is parsed into “a blue car” and “a white picket fence”), and then ask the video LLM to assign a matching grade for each disentangled phrase in relation to the Image Grid. The grades for both phrases are combined and averaged to produce a final numerical score.
- • For action binding, we use GPT-4 [47] to extract objects and their actions. For example, given the prompt “A dog runs through a field while a cat climbs a tree”, we extract the phrases “a dog”, “a dog runs through a field”, “a cat”, and “a cat climbs a tree”. We then ask the video LLM to check the presence of objects and evaluate the alignment between each object-action pair and the Image Grid.
- • For object interactions, we prompt the video LLM to check the presence of objects and then assess the quality of the interaction based on the text. This evaluation includes examining dynamics of objects, the overall development and outcome of the interaction process.

Image LLM-based metrics for dynamic attribute binding. Evaluating dynamic attribute binding, such as the prompt “Bright green leaf wilts to brown” is challenging as it requires a deep understanding of dynamic changes across frames. We find that current video LLMs perform poorly in this area, so we develop a frame-by-frame evaluation metric based on an Image LLM such as LLaVA [35]. We utilize GPT-4 [47] to parse the initial state (“bright green leaf”) and the final state (“brown leaf”). We then prompt LLaVA [35] to score the alignment between each frame and each of the two states. Our scoring function is designed to encourage the first frame to align with the initial state, the last frame

to align with the final state, and the middle frames to be in between. We denote this metric as D-LLaVA.

For more details about using MLLMs as evaluation metrics, please refer to Appendix C.

##### 4.2. Detection-based Evaluation Metrics

Most vision-language models face difficulties with spatial relationships and numeracy-related understandings. So we introduce the object detection model GroundingDINO (GDino) [37] to detect objects for each frame, filter out duplicate bounding boxes with high intersection-over-union (IoU), and then define rule-based metrics based on the object detection results.

2D Spatial Relationships. For 2D spatial relationships including “left”, “right”, “above”, and “below”, we define rule-based metrics for each frame similar to T2ICompBench [20]. Specifically, for each pair of objects, we denote their centers as (x1,y1) and (x2,y2), respectively. The first object is on the left of the second object if x1 < x2, and |x1 − x2| > |y1 − y2|. The rule is similar for other

- 2D spatial relationships. If there is more than one pair of objects in a frame, we select the most probable one based on their IoU and confidence scores. The per-frame score is (1−IoU) if there is an object pair that satisfies the relationship, or 0 if no object pair satisfies the relationship. The video-level score is the average of per-frame scores.
- 3D Spatial Relationships. 3D spatial relationships (“in front of”, “behind”) cannot be identified by purely 2D bounding boxes. With the 2D boxes detected by GroundingDINO [37], we further leverage Segment Anything [25] to predict masks of the specified objects and then leverage Depth Anything [75] to predict depth maps. The depth of an object is defined as the average depth values of the pixels inside the object mask. We define per-frame score based on the IoU and relative depth between two objects, and the video-level score is the average of per-frame scores. Generative Numeracy. To evaluate generative numeracy, we count the number of objects detected for each object class. If the detected quantity matches the number in text prompt, we assign a score of 1 for that object class. Otherwise, we assign a score of 0. The frame-level score is calculated as the average score of all object classes, and the video-level score is the average of all frames.

##### 4.3. Tracking-based Evaluation Metrics

The evaluation metric for motion binding aims to identify the moving direction of objects in videos. However, in many cases, object motions are entangled with camera motions, making it difficult to determine the true direction of an object’s movement. In videos, the actual moving direction of an object is the relative moving direction between the foreground object and the background. Therefore, we introduce a tracking-based method to determine the mov-

ing directions of the foreground and background separately. Specifically, we use GroundingSAM [55] to obtain masks for the foreground objects and the background. Then, we apply DOT [46] to track points in both the foreground and background throughout the video. We compute the average motion vectors for the points in both the foreground and background, and the difference between these two vectors gives us the actual moving direction of the object. The final score reflects whether this actual moving direction aligns with the motion described in the text prompts. We denote this metric as DOT.

#### 5. Experiments

##### 5.1. Evaluated Text-to-video Models

We evaluate the performance of 23 T2V models on T2VCompBench, including 17 open-source models and 6 commercial models. Among these models,

- • ModelScope [65] is a diffusion unet-based T2V generation model that evolves from a T2I model by incorporating spatio-temporal blocks, ensuring consistent frame generation and smooth movement transitions.
- • AnimateDiff [13] features a plug-and-play motion module that can be applied to diffusion unet-based T2I models to produce temporally smooth animations.
- • Show-1 [81] is a high-quality T2V diffusion model with unet backbone that initially employs pixel-based diffusion to create a low-resolution video, followed by a latentbased approach to upsample the video to high resolution.
- • VideoCrafter2 [6] is a diffusion unet-based video generation model that utilizing low-quality videos and synthesized high-quality images to create a video model with high visual quality and precise text-video alignment.
- • Latte [43] is a novel Diffusion Transformer (DiT)-based video generation model, which extracts spatio-temporal tokens from input videos and utilizes Transformer blocks to model video distribution in latent space.
- • Open-Sora [19] and Open-Sora-Plan [33] are both DiTbased open-source video generation projects, aiming at reproducing Sora.
- • CogVideoX-5B [78] is a DiT-based large scale T2V model designed to generate significant movement.
- • Mochi [61] is also a large scale T2V model built on Asymmetric DiT architecture, with emphasis on quality of motion and prompt following.
- • LVD [32] is specifically designed to leverage LLMguided layout planning for videos with multiple objects. Here, we use the version adapted from ModelScope [65]. ZeroScope [60] is optimized from ModelScope [65] to create high-quality 16:9 compositions and smooth videos.
- • VideoTetris [62] is proposed to handle compositional T2V generation that involves multiple objects and dynamic changes in object numbers. Vico [77] is a frame-

work that explicitly ensures that complex compositional interactions between multiple concepts and actions can be represented properly. T2V-Turbo-V2 [29] introduces a consistency distillation process of a T2V model from a pretrained T2V model by integrating various supervision signals. These three model are all adapted or distilled from VideoCrafter2 [6].

• MagicTime [79] is trained from AnimateDiff [13]. It is designed to generate metamorphic time-lapse videos, so we specifically test it on dynamic attribute binding.

To organize the T2V models, we categorize them into five groups: 1) DiT-based models: Latte [43], Open-Sora 1.1 and 1.2 [19], Open-Sora-Plan v1.0.0 and v1.3.0 [33], CogVideoX-5B [78] and Mochi [61]. Models with the same foundation model: 2) ModelScope [65], ZeroScope [60], and LVD [32]; 3) AnimateDiff [13] and MagicTime [79]; 4) Videocrafter2 [6], VideoTetris [62], Vico [77], and T2V-Turbo-V2 [29].* 5) Commercial models: Pika1.0 [51], Gen-2 [56], Gen-3 [57], Dreamina 1.2 [4], PixVerse-V3 [52], and Kling-1.0 [26].

We follow the official default implementations of these T2V models in evaluation, please refer to Appendix B.

##### 5.2. Evaluation Metrics

Conventional Metrics. We compare our proposed metrics with five metrics widely used in previous studies: 1) CLIPScore [15] (denoted as CLIP) calculates the cosine similarity between CLIP text and image embeddings. 2) BLIPCLIP [5] (denoted as B-CLIP) applies BLIP [27] to generate captions for images, and then calculates the text-text cosine similarity between the CLIP embeddings of captions and input prompts. 3) BLIP-BLEU [39] (denoted as BBLEU) employs BLIP2 [28] for caption generation, then calculates the BLEU [48] similarity between the captions and input prompts. Here, we use the same implementation as EvalCrafter [39], which averages five captions generated by BLIP2 [28]. 4) BLIP-VQA [20] (denoted as B-VQA) leverages the visual question answering (VQA) ability of BLIP2 [27] to evaluate the text-image alignment, focusing specifically on attribute binding. The video-level scores of these metrics are calculated by averaging across all frames. We also include 5) ViCLIP score [21], which measures textvideo alignment by calculating the similarity between text and video features extracted by ViCLIP model [68].

In addition, VPEval [9] introduces detection-based metrics using GroundingDINO [37] to evaluate spatial relationships in T2I generation. VideoDirectorGPT [34] adapts VPEval [9] to evaluate object movement direction in videos by obtaining the object locations in the first and last frames. Therefore, we include the detection-based metrics from VPEval [9]: 6) VPEval-S, and from VideoDirec-

*Show-1 [81] and videocrafter2 [6] are concurrent diffusion unet-based models, so we place them together.

torGPT [34]: 7) M-GDino, as compared metrics to evaluate the spatial relationships and motion binding in our benchmark. Similarly, VPEval-S is adapted for T2V evaluation by averaging the scores over all frames.

Our Proposed Metrics. As introduced in Section 4, the image LLM-based metric D-LLaVA is designed to evaluate dynamic attribute binding. The detection-based metric G-Dino is designed for spatial relationships and numeracy. The tracking-based metric DOT is designed for motion binding. Additionally, we test the video LLM-based metrics Grid-LLaVA and PLLaVA for all categories. We also test the image LLM-based metric LLaVA, which evaluates the text-video alignment on a frame basis for all categories except dynamic attribute binding and motion binding.

In next section, we identify the best metric for each category by analyzing the correlation between results given by these metrics and human annotators.

##### 5.3. Human Evaluation Correlation Analysis

In this section, we conduct human evaluations and compute the correlation between scores from automatic metrics and humans to identify the best metric for each category.

Human Evaluation. For the human evaluation of each category, we randomly select 15 prompts out of 200 prompts and use 6 T2V models to generate a total of 90 videos. Additionally, we include 10 ground truth videos for dynamic attribute binding and 11 for object interactions. The total number of videos for human evaluation is 651. We employ the platform of Amazon Mechanical Turk, where we ask three annotators to score the text-video alignment for each video. We then average across the three scores for each text-video pair and calculate the correlation between these human scores and the automatic evaluation scores with Kendall’s τ and Spearman’s ρ. For more details about human evaluation, please refer Appendix E.

Comparisons across Evaluation Metrics. The correlation results are presented in Table 1. These results validate the effectiveness of our proposed metrics, which are highlighted in bold. Specifically, Grid-LLaVA excels in consistent attribute binding, action binding and object interactions, D-LLaVA is the best for dynamic attribute binding, G-Dino is the top metric for spatial relationships and generative numeracy, and DOT is the most reliable for motion binding. In contrast, CLIP, ViCLIP, B-CLIP and B-BLEU show relatively low human correlations, indicating their limitations in capturing the fine-grained attributes, spatial relationships and dynamics in complex videos. Although Grid-LLaVA is the top metric for three categories, its performance is marginally inferior to those best metrics in other categories in capturing temporal changes in attributes, understanding spatial relationships, motion direction, and numeracy. LLaVA performs reasonably well across all the categories it evaluates, which reveals its abilities in captur-

Consist-attr Dynamic-attr Spatial Motion Action Interaction Numeracy τ(↑) ρ(↑) τ(↑) ρ(↑) τ(↑) ρ(↑) τ(↑) ρ(↑) τ(↑) ρ(↑) τ(↑) ρ(↑) τ(↑) ρ(↑)

Metric

conventional metrics CLIP 0.3667 0.4859 −0.0096 −0.01401 0.2395 0.3343 0.1381 0.1818 0.2796 0.3799 0.1187 0.1587 0.0560 0.0821 B-CLIP 0.2609 0.3562 0.2100 0.2917 0.1247 0.1647 −0.0582 −0.0889 0.0915 0.1246 0.1455 0.2103 0.0694 0.0829 B-BLEU 0.2030 0.2777 0.0854 0.1041 0.1006 0.1396 −0.1450 −0.1978 0.2505 0.3577 0.1040 0.1479 0.1275 0.1770 B-VQA 0.5194 0.6964 − − − − − − − − − − − − ViCLIP 0.4520 0.6116 0.0079 0.0074 0.2257 0.3222 0.0834 0.1144 0.2481 0.3361 0.2308 0.3229 0.1036 0.1421

MLLM-based metrics

PLLaVA 0.2715 0.3105 0.1845 0.2201 −0.1252 −0.1509 0.0401 0.0498 0.4326 0.5066 −0.2233 −0.2607 0.2253 0.3142 LLaVA 0.6373 0.7461 − − 0.4258 0.5332 − − 0.5272 0.6714 0.4213 0.5358 0.3212 0.4540 Grid-LLaVA 0.6636 0.7893 0.1435 0.1678 0.4815 0.5763 0.1349 0.1619 0.5969 0.7353 0.4557 0.5925 0.2266 0.2809 D-LLaVA − − 0.4362 0.5061 − − − − − − − − − −

detection-based metrics

VPEval-S − − − − 0.3137 0.3853 − − − − − − − − M-GDino − − − − − − 0.2059 0.2306 − − − − − − G-Dino − − − − 0.5769 0.7057 − − − − − − 0.4063 0.5378

tracking-based metric

###### DOT − − − − − − 0.4523 0.5366 − − − − − −

- Table 1. The correlation between automatic evaluations and human evaluations. Our proposed evaluation metrics show enhanced performance in Kendall’s τ and Spearman’s ρ.

ing attributes, spatial layout, numeracy and object relationships. It falls short compared to Grid-LLaVA in action binding and object interactions. This is because Grid-LLaVA has the advantage of analyzing multiple frames simultaneously, which allows it to account for temporal changes rather than relying solely on a static frame.

##### 5.4. Quantitative Evaluation

The performance of the models in T2V-CompBench evaluated by our proposed metrics is shown in Table 2. A comparison of different models reveals the following findings:

- #1: Top Performers. T2V-Turbo-V2 [29] exhibits the best overall performance in diffusion unet-based models. CogVideoX-5B [78] shows the best overall performance in DiT-based models. PixVerse-V3 [52] excels in all models.
- #2: Adapted Models. Models adapted from VideoCrafter2 [6], including VideoTetris [62], Vico [77], and T2V-Turbo-V2 [29], improves in most of the categories. MagicTime [79] shows a significant improvement in dynamic attribute binding compared to AnimateDiff [13]. LVD [32] demonstrates enhancements in almost all categories compared to ModelScope [65]. Its design, which leverages LLM-guided layout planning, allows it to excel in spatial relationships in diffusion unet-based models.
- #3: Models with Strong Dynamic Capabilities. Among open-source models, T2V-Turbo-V2[29] and CogVideoX5B[78] excel in motion binding and action binding. They show strong performance in dynamic attribute binding and interactions as well. The commercial models Gen-3[57] and PixVerse-V3[52] lead all models in dynamic attribute binding and motion binding. This can be attributed to their capabilities to generate videos with a high dynamic degree.
- #4: Other Noteworthy Results. Other notable models include Open-Sora 1.2 [19], which excels in numeracy

among open-source models, and VideoTetris [62], which demonstrates strong capabilities in numeracy and interaction among diffusion unet-based models.

##### 5.5. Qualitative Evaluation

The challenging cases for the seven compositional categories are illustrated in Figure 6 and 7, with the difficulty level decreasing from top to bottom rows. Figure 6 shows example videos from open-source models, while Figure 7 shows those from commercial models. In our evaluation of various categories, we identify the following insights:

- #1: Dynamic Attribute Binding is the most challenging category. As shown in row #1 of Figure 6 and 7, the evaluated T2V models tend to focus on certain keywords in the prompts, but overlook the required transitions in attributes. Consequently, they usually generate fixed objects or attributes that do not have any change through the video.
- #2: T2V models struggle with generating correct spatial relationships, motion directions, and quantities. The second most challenging categories include spatial relationships, motion binding, and numeracy (rows #4, #3, and #2 in Figure 6 and 7). In spatial relationships, most T2V models struggle to differentiate between locality terms such as “left” and “right”, resulting in random spatial layouts in the generated videos. The issue is even more pronounced in motion binding, where models can hardly ever understand moving directions, such as “sail to the left” or “flying right towards”. Most T2V models not only fail to move in the correct directions but also have difficulties generating significant movement for objects. In addition, generating videos with correct number of objects requires accurate counting. While the T2V models perform well when the quantities are fewer than three, they often fail to accurately generate larger quantities of objects.

Model Consist-attr Dynamic-attr Spatial Motion Action Interaction Numeracy Metric Grid-LLaVA D-LLaVA G-Dino DOT Grid-LLaVA Grid-LLaVA G-Dino

diffusion unet-based

ModelScope [65] 0.5148 0.0161 0.4118 0.2408 0.3639 0.4613 0.1986 ZeroScope [60] 0.4011 0.0091 0.4287 0.2454 0.3661 0.4196 0.2408 LVD [32] 0.5439 0.0171 0.5405 0.2457 0.3802 0.4502 0.2008

AnimateDiff [13] 0.4325 0.0097 0.3920 0.2227 0.2844 0.3970 0.1767 MagicTime [79] - 0.0151 - - - - -

Show-1 [81] 0.5670 0.0115 0.4544 0.2291 0.3881 0.6244 0.3086 VideoCrafter2 [6] 0.6182 0.0103 0.4838 0.2259 0.5030 0.6365 0.3330 VideoTetris [62] 0.6211 0.0104 0.4832 0.2249 0.4939 0.6578 0.3467 Vico [77] 0.5887 0.0107 0.4974 0.2219 0.5111 0.5957 0.3230 T2V-Turbo-V2 [29] 0.6723 0.0127 0.5025 0.2556 0.6087 0.6439 0.3261

DiT-based Latte [43] 0.4713 0.0080 0.4340 0.2155 0.4146 0.4146 0.2320

- Open-Sora 1.1 [19] 0.5414 0.0109 0.5406 0.2261 0.5037 0.5565 0.2259
- Open-Sora 1.2 [19] 0.5639 0.0189 0.5063 0.2468 0.4833 0.5039 0.3719 Open-Sora-Plan v1.0.0 [33] 0.4246 0.0086 0.4520 0.2148 0.4009 0.4150 0.2331 Open-Sora-Plan v1.3.0 [33] 0.6076 0.0119 0.5162 0.2377 0.4524 0.4483 0.2952 CogVideoX-5B [78] 0.6164 0.0219 0.5172 0.2658 0.5333 0.6069 0.3706 Mochi [61] 0.5973 0.0246 0.5480 0.2334 0.4759 0.5381 0.2718

commercial Pika-1.0 [51] 0.5536 0.0128 0.4650 0.2234 0.4250 0.5198 0.3870

- Gen-2 [56] 0.5795 0.0109 0.5126 0.2173 0.4413 0.6144 0.3039
- Gen-3 [57] 0.5980 0.0687 0.5194 0.2754 0.5233 0.5906 0.2306 Dreamina 1.2 [4] 0.6913 0.0051 0.5773 0.2361 0.5924 0.6824 0.4380 PixVerse-V3 [52] 0.7060 0.0624 0.5979 0.2867 0.8722 0.8309 0.6066 Kling-1.0 [26] 0.6931 0.0098 0.5690 0.2562 0.5787 0.7128 0.4413

- Table 2. T2V-CompBench evaluation results using proposed metrics. Scores are normalized between 0 and 1. A higher score indicates better performance. Bold signifies the highest score within each category. Blue highlights the top score among diffusion unet-based models. Yellow highlights the top score among DiT-based models. Red highlights the top score among commercial models.

#3: While Object Interactions, Action Binding, and Consistent Attribute Binding are generally easier to handle, T2V models still encounter challenging scenarios. Text-to-video examples from object interactions, action binding, and consistent attribute binding are shown in rows #5-7 in Figure 6 and Figure 7. In object interactions, some T2V models tend to produce static videos that do not depict the full interaction process. Regarding action binding, models may find it difficult to generate correct actions. For instance, given the prompt “A dog runs through a field while a cat climbs a tree”, the models might incorrectly show both animals running instead of representing their respective actions, or they may only depict one animal while ignoring the other. In terms of consistent attribute binding, models sometimes fail to accurately associate attributes with the correct objects or overlook certain object entirely.

For additional analysis of the evaluation results, please refer to Appendix D.

#### 6. Conclusion

We conduct the first systematic study on compositionality in text-to-video generation. We propose T2V-CompBench, a comprehensive benchmark for compositional text-to-video generation, with 1400 prompts in seven categories. We further design a suite of evaluation metrics for the seven categories, all of them are validated by correlations with human evaluation. Finally, we benchmark various text-tovideo generation models with different architectures and provide insightful analysis and findings based on the results. Compositional text-to-video generation is highly challenging for current models, and we hope our work will inspire future works to improve the compositionality of text-tovideo models. For limitations and social impacts of our work, please refer to Appendix G and F.

LVD T2V-Turbo-V2 CogVideoX-5B

[Figure 116]

[Figure 117]

[Figure 118]

A timelapse of a leaf transitioning from green to bright red as autumn progresses.

[Figure 119]

[Figure 120]

[Figure 121]

Six horses gallop across the beach.

[Figure 122]

[Figure 123]

[Figure 124]

A paper airplane gliding to the left across a classroom.

[Figure 125]

[Figure 126]

[Figure 127]

A llama walking on the left side of a motorcycle in a city street.

[Figure 128]

[Figure 129]

[Figure 130]

Two cars collide at an intersection.

[Figure 131]

[Figure 132]

[Figure 133]

A dog runs through a field while a cat climbs a tree.

[Figure 134]

[Figure 135]

[Figure 136]

White tractor plowing near a green farmhouse.

###### Figure 6. Qualitative comparison with different open-source T2V models for the seven compositional categories in T2VCompBench. We show text-to-video examples generated by LVD [32], T2V-Turbo-V2 [29], and CogVideoX-5B [78].

Gen-3 PixVerse-V3 Kling-1.0

[Figure 137]

[Figure 138]

[Figure 139]

A timelapse of a leaf transitioning from green to bright red as autumn progresses.

[Figure 140]

[Figure 141]

[Figure 142]

Six horses gallop across the beach.

[Figure 143]

[Figure 144]

[Figure 145]

A paper airplane gliding to the left across a classroom.

[Figure 146]

[Figure 147]

[Figure 148]

A llama walking on the left side of a motorcycle in a city street.

[Figure 149]

[Figure 150]

[Figure 151]

Two cars collide at an intersection.

[Figure 152]

[Figure 153]

[Figure 154]

A dog runs through a field while a cat climbs a tree.

[Figure 155]

[Figure 156]

[Figure 157]

White tractor plowing near a green farmhouse.

###### Figure 7. Qualitative comparison with different commercial T2V models for the seven compositional categories in T2V-CompBench. We show text-to-video examples generated by Gen-3 [57], PixVerse-V3 [52], and Kling-1.0 [26].

#### References

- [1] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, Wesam Manassra, Prafulla Dhariwal, Casey Chu, Yunxin Jiao, and Aditya Ramesh. Improving image generation with better captions. https://cdn.openai.com/ papers/dall-e-3.pdf, 2023. 2
- [2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 1
- [3] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR, 2023. 1, 2
- [4] Capcut. Dreamina. https://dreamina.capcut. com/ai-tool/home, 2024. 2, 7, 9, 16, 19
- [5] Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. In ACM Trans. Graph., 2023. 2, 3, 7
- [6] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. arXiv preprint arXiv:2401.09047, 2024. 2, 6, 7, 8, 9, 16, 19, 20
- [7] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In ICLR, 2024. 2
- [8] Minghao Chen, Iro Laina, and Andrea Vedaldi. Trainingfree layout control with cross-attention guidance. In WACV,

2024. 2

- [9] Jaemin Cho, Abhay Zala, and Mohit Bansal. Visual programming for text-to-image generation and evaluation. In NeurIPS, 2023. 7
- [10] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024. 2
- [11] Weixi Feng, Xuehai He, Tsu-Jui Fu, Varun Jampani, Arjun Akula, Pradyumna Narayana, Sugato Basu, Xin Eric Wang, and William Yang Wang. Training-free structured diffusion guidance for compositional text-to-image synthesis. In ICLR, 2023. 2, 3
- [12] Hanan Gani, Shariq Farooq Bhat, Muzammal Naseer, Salman Khan, and Peter Wonka. Llm blueprint: Enabling text-to-image generation with complex and detailed prompts. In ICLR, 2024. 2
- [13] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your

- personalized text-to-image diffusion models without specific tuning. In ICLR, 2024. 2, 6, 7, 8, 9, 16, 19
- [14] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity long video generation. arXiv preprint arXiv:2211.13221,

2022. 1, 2

- [15] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. CLIPScore: a reference-free evaluation metric for image captioning. In EMNLP, 2021. 2, 3, 7
- [16] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, 2017. 2
- [17] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 1, 2
- [18] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. In ICLR, 2023. 1
- [19] hpcaitech. Open-sora: Democratizing efficient video production for all, 2024. 2, 6, 7, 8, 9, 16, 19, 21
- [20] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. In NeurIPS, 2023. 2, 3, 5, 6, 7
- [21] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In CVPR, 2024. 2, 3, 7
- [22] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Textto-image diffusion models are zero-shot video generators. In ICCV, 2023. 1, 2
- [23] Wonkyun Kim, Changin Choi, Wonseok Lee, and Wonjong Rhee. An image grid can be worth a video: Zeroshot video question answering using a vlm. arXiv preprint arXiv:2403.18406, 2024. 5
- [24] Yunji Kim, Jiyoung Lee, Jin-Hwa Kim, Jung-Woo Ha, and Jun-Yan Zhu. Dense text-to-image generation with attention modulation. In ICCV, 2023. 2
- [25] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar, and Ross Girshick. Segment anything. arXiv:2304.02643, 2023. 6
- [26] Kuaishou. Kling. https://kling.kuaishou.com/,

2024. 2, 7, 9, 10, 16, 19

- [27] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML,

2022. 7

- [28] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML,

2023. 7

- [29] Jiachen Li, Qian Long, Jian Zheng, Xiaofeng Gao, Robinson Piramuthu, Wenhu Chen, and William Yang Wang. T2vturbo-v2: Enhancing video generation model post-training through data, reward, and conditional guidance design. arXiv preprint arXiv:2410.05677, 2024. 2, 7, 8, 9, 10, 16, 18, 19
- [30] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In ICCV, 2023. 2
- [31] Zhiheng Li, Martin Renqiang Min, Kai Li, and Chenliang Xu. Stylet2i: Toward compositional and high-fidelity textto-image synthesis. In CVPR, 2022.
- [32] Long Lian, Baifeng Shi, Adam Yala, Trevor Darrell, and Boyi Li. Llm-grounded video diffusion models. In ICLR,

2023. 2, 6, 7, 8, 9, 10, 16, 18, 19, 20

- [33] Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024. 2, 6, 7, 9, 16, 19, 21
- [34] Han Lin, Abhay Zala, Jaemin Cho, and Mohit Bansal. Videodirectorgpt: Consistent multi-scene video generation via llm-guided planning. In COLM, 2024. 7
- [35] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR,

2024. 5, 17

- [36] Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B Tenenbaum. Compositional visual generation with composable diffusion models. In ECCV, 2022. 2
- [37] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In ECCV, 2024. 6, 7
- [38] Xuantong Liu, Tianyang Hu, Wenjia Wang, Kenji Kawaguchi, and Yuan Yao. Referee can play: An alternative approach to conditional generation via model inversion. In ICML, 2024. 2
- [39] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. In CVPR, 2024. 2, 3, 7
- [40] Yuanxin Liu, Lei Li, Shuhuai Ren, Rundong Gao, Shicheng Li, Sishuo Chen, Xu Sun, and Lu Hou. Fetv: A benchmark for fine-grained evaluation of open-domain text-tovideo generation. In NeurIPS, 2024. 2, 3
- [41] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? In ACL Findings, 2024. 4, 23
- [42] Zhengxiong Luo, Dayou Chen, Yingya Zhang, Yan Huang, Liang Wang, Yujun Shen, Deli Zhao, Jingren Zhou, and Tie-

- niu Tan. Videofusion: Decomposed diffusion models for high-quality video generation. In CVPR, 2023. 1, 2
- [43] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024. 2, 6, 7, 9, 16, 19
- [44] Tuna Han Salih Meral, Enis Simsar, Federico Tombari, and Pinar Yanardag. Conform: Contrast is all you need for highfidelity text-to-image diffusion models. In CVPR, 2024. 2
- [45] George A Miller. Wordnet: a lexical database for english. Communications of the ACM, 38(11):39–41, 1995. 4, 14, 15
- [46] Guillaume Le Moing, Jean Ponce, and Cordelia Schmid. Dense optical tracking: Connecting the dots. In CVPR, 2024. 6
- [47] OpenAI. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2024. 4, 5, 22
- [48] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In ACL, 2002. 7
- [49] Dong Huk Park, Samaneh Azadi, Xihui Liu, Trevor Darrell, and Anna Rohrbach. Benchmark for compositional text-toimage synthesis. In NeurIPS, 2021. 2
- [50] Maitreya Patel, Changhoon Kim, Sheng Cheng, Chitta Baral, and Yezhou Yang. Eclipse: A resource-efficient text-toimage prior for image generations. In CVPR, 2024. 2
- [51] Pika. Pika. https://www.pika.art, 2024. 2, 7, 9, 16, 19
- [52] PixVerse. Pixverse. https://app.pixverse.ai,

2024. 2, 7, 8, 9, 10, 16, 18, 19, 20, 21

- [53] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 3
- [54] Royi Rassin, Eran Hirsch, Daniel Glickman, Shauli Ravfogel, Yoav Goldberg, and Gal Chechik. Linguistic binding in diffusion models: Enhancing attribute correspondence through attention map alignment. In NeurIPS, 2024. 2
- [55] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, Zhaoyang Zeng, Hao Zhang, Feng Li, Jie Yang, Hongyang Li, Qing Jiang, and Lei Zhang. Grounded sam: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159, 2024. 6
- [56] Runway. Gen-2: Generate novel videos with text, images or video clips. https://research.runwayml.com/ gen2, 2024. 2, 7, 9, 16, 19
- [57] Runway. Introducing gen-3 alpha: A new frontier for video generation. https://runwayml.com/blog/ introducing-gen-3-alpha/, 2024. 2, 7, 8, 9, 10, 16, 19, 20
- [58] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. In NeurIPS, 2016. 2, 3
- [59] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation

without text-video data. arXiv preprint arXiv:2209.14792,

2022. 1, 2

- [60] Spencer Sterling. Zeroscope. https://huggingface. co/cerspense/zeroscope_v2_576w, 2023. 6, 7, 9, 16, 19
- [61] Genmo Team. Mochi 1. https://github.com/ genmoai/models, 2024. 2, 6, 7, 9, 16, 19, 20
- [62] Ye Tian, Ling Yang, Haotian Yang, Yuan Gao, Yufan Deng, Jingmin Chen, Xintao Wang, Zhaochen Yu, Xin Tao, Pengfei Wan, Di Zhang, and Bin Cui. Videotetris: Towards compositional text-to-video generation. In NeurIPS, 2024. 2, 6, 7, 8, 9, 16, 19, 21
- [63] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Rapha¨el Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 2, 3
- [64] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable length video generation from open domain textual descriptions. In International Conference on Learning Representations, 2022. 1
- [65] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 1, 2, 6, 7, 8, 9, 16, 18, 19, 21
- [66] Ruichen Wang, Zekang Chen, Chen Chen, Jian Ma, Haonan Lu, and Xiaodong Lin. Compositional text-to-image synthesis with attention map control of diffusion models. arXiv preprint arXiv:2305.13921, 2023. 2
- [67] Wenhao Wang and Yi Yang. Vidprom: A million-scale real prompt-gallery dataset for text-to-video diffusion models. In NeurIPS, 2024. 4, 14
- [68] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, Conghui He, Ping Luo, Ziwei Liu, Yali Wang, Limin Wang, and Yu Qiao. Internvid: A large-scale video-text dataset for multimodal understanding and generation. In ICLR, 2023. 7
- [69] Zhenyu Wang, Enze Xie, Aoxue Li, Zhongdao Wang, Xihui Liu, and Zhenguo Li. Divide and conquer: Language models can plan and self-correct for compositional text-to-image generation. arXiv preprint arXiv:2401.15688, 2024. 2
- [70] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. In NeurIPS, 2022. 5, 17
- [71] Chenfei Wu, Lun Huang, Qianxi Zhang, Binyang Li, Lei Ji, Fan Yang, Guillermo Sapiro, and Nan Duan. Godiva: Generating open-domain videos from natural descriptions. arXiv preprint arXiv:2104.14806, 2021. 1
- [72] Chenfei Wu, Jian Liang, Lei Ji, Fan Yang, Yuejian Fang, Daxin Jiang, and Nan Duan. N¨uwa: Visual synthesis pretraining for neural visual world creation. In European conference on computer vision, pages 720–736. Springer, 2022. 1

- [73] Qiucheng Wu, Yujian Liu, Handong Zhao, Trung Bui, Zhe Lin, Yang Zhang, and Shiyu Chang. Harnessing the spatialtemporal attention of diffusion models for high-fidelity textto-image synthesis. In ICCV, 2023. 2
- [74] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. Pllava: Parameter-free llava extension from images to videos for video dense captioning. arXiv preprint arXiv:2404.16994, 2024. 5
- [75] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In CVPR, 2024. 6
- [76] Ling Yang, Zhaochen Yu, Chenlin Meng, Minkai Xu, Stefano Ermon, and Bin Cui. Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal llms. In ICML, 2024. 2
- [77] Xingyi Yang and Xinchao Wang. Compositional video generation as flow equalization. arXiv preprint

- arXiv:2407.06182, 2024. 2, 6, 7, 8, 9, 16, 19, 20

[78] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint

- arXiv:2408.06072, 2024. 2, 6, 7, 8, 9, 10, 16, 19, 20, 21

- [79] Shenghai Yuan, Jinfa Huang, Yujun Shi, Yongqi Xu, Ruijie Zhu, Bin Lin, Xinhua Cheng, Li Yuan, and Jiebo Luo. Magictime: Time-lapse video generation models as metamorphic simulators. arXiv preprint arXiv:2404.05014, 2024. 2, 7, 8, 9, 16
- [80] Shenghai Yuan, Jinfa Huang, Yongqi Xu, Yaoyang Liu, Shaofeng Zhang, Yujun Shi, Ruijie Zhu, Xinhua Cheng, Jiebo Luo, and Li Yuan. Chronomagic-bench: A benchmark for metamorphic evaluation of text-to-time-lapse video generation. 2024. 3
- [81] David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and Mike Zheng Shou. Show-1: Marrying pixel and latent diffusion models for text-to-video generation. IJCV, pages 1–15, 2024. 2, 6, 7, 9, 16, 19
- [82] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018, 2022. 1, 2
- [83] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 5

## T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation

### Appendices

#### Appendix Contents

- A. Prompt Suite 14

- A.1. Vocabulary Construction . . . . . . . . . . . 14
- A.2. Prompt Generation with LLM . . . . . . . . 15
- A.3. Prompt Suite Statistics . . . . . . . . . . . . 15
- A.4. Stability of 1400 Prompts and Videos . . . . 15

- B. Implementation Details 16
- C. Using MLLMs as Evaluation Metrics 16

- C.1. How to query MLLMs . . . . . . . . . . . . 16
- C.2. How to obtain reliable and reproducible results 17

- D. Evaluation Results Analysis 18

- D.1. Consistent Attribute Binding . . . . . . . . . 18
- D.2. Dynamic Attribute Binding . . . . . . . . . 18
- D.3. Spatial Relationships . . . . . . . . . . . . . 19
- D.4. Motion Binding . . . . . . . . . . . . . . . . 20
- D.5. Action Binding . . . . . . . . . . . . . . . . 20
- D.6. Object Interactions . . . . . . . . . . . . . . 20
- D.7. Generative Numeracy . . . . . . . . . . . . 21

- E. Human Evaluation 21
- F. Societal Impacts 22
- G. Limitations and Future Work 22
- H. Templates for Generating Prompts and Metadata Using GPT-4 22

#### A. Prompt Suite

##### A.1. Vocabulary Construction

We utilize WordNet [45] to group words from real-user prompts provided by VidProM [67] into multi-level meta classes. From these groups, we select high-frequency object nouns, active verbs, and adjectives.

Nouns. Figure 8 illustrates the structure of these multi-level classes. The nouns organized into multi-level meta classes come along with their frequencies of occurrence in realuser prompts. To select high-frequency words, we first analyze the frequency distribution of the entire real-user prompt dataset. Figure 9(a) displays the frequencies of the 10000 most frequent nouns, arranged in descending order. It is apparent that users do not use words with uniform frequency;

rather, a significant number of words center around the first 2000 most frequent words, while the remaining words receive much less attention. In this dataset, all of the top 2000 words have frequencies greater than 900, so we primarily focus on words that exceed this threshold. Following our word selection criteria outlined in Section 3, we start with words that meet our criteria, such as “dog” and “car”. From there, we select additional words within the same class as “dog” or “car” to expand our noun list, for example, “cat”, “lion” and “truck”, “boat”. As a result, we identify a total of 260 object nouns, within which 66% have frequencies greater than 900. Although the remaining words fall below this frequency, they are still commonly used in natural language and contribute to the diversity of our noun selection.

Verbs. We apply the same method to group the verbs. Figure 9(b) illustrates the frequency distribution of verbs. The top 1200 words have frequencies over 370. Then, we select 200 active and vibrant verbs, with 60% having frequencies greater than 370.

Attributes. For words that describe attributes, we select from both adjectives and nouns. When analyzing adjectives, we identify relevant words by looking for specific keywords in their definitions. If a definition includes terms like “color” or “colored”, we classify the word as a color attribute. Similarly, if the definition contains “made of” or “texture”, we categorize it under texture attributes. Word definitions including “shape” or “shaped” are classified as shape attributes. However, some words that should belong to these categories may be excluded if their definitions do not contain the relevant keywords. To address this, we also consider attributes derived from nouns. Words that represent color or shape can function as either nouns or adjectives in a sentence, so we refer to the meta classes of “color” and “shape” within the noun classification. Additionally, there is a meta class for “material”, which encompasses words that describe the texture of objects. We draw from these classes to supplement the attributes identified from adjectives. In total, we select 14 color attributes, 26 shape attributes, and 26 texture attributes. For humanrelated attributes, we focus on nouns in meta classes such as “body covering”, which includes words like “hair” and “beard”, as well as “clothing”, which features words like “dress” and “hat”.

The list of the 260 nouns, 200 verbs and 100 attributes are provided in T2V-CompBench word list.

|Level 1 - Meta Class: "object"<br><br>Level 2 - Meta Class: "living_thing"<br><br>Level 3 – Meta Class: "animal":<br><br><br>{ "cat": 27826, "dog" : 18615,<br><br>"lion" : 9258, "bird" : 8843, …}<br><br>Level 3 – Meta Class: "plant":<br><br>{ "tree": 18708, "grass": 8390,<br><br>"flower": 5832, …}<br><br>Level 2 - Meta Class: "natural object"<br><br>Level 3 – Meta Class: "body_covering":<br><br><br>{"hair": 45339,<br><br>"skin": 9509, "beard": 4514, …}<br><br>Level 3 – Meta Class:<br><br>"plant_part": { "seed": 8507,<br><br>"leaves": 7349, "fruit": 2033,<br><br>"petals": 1975, …}<br><br>|
|---|

Figure 8. Illustration of multi-level meta class structure.

(a) Meta types of nouns (b) Meta types of verbs

[Figure 158]

[Figure 159]

- Figure 10. Word distributions of T2V-CompBench prompts We show the types of nouns and verbs of T2V-CompBench prompts.

spatial relationships and numeracy, stabilize earlier, at approximately 125 videos, while others, such as Grid-LLaVA for consistent attribute binding, action binding and object interactions, stabilize around 150 videos. The stability of these metrics can also be influenced by different T2V models, as evidenced by the varying levels of fluctuations at the ends of the lines in the figures. However, given that current T2V models typically take minutes to inference a video, it is more practical to limit the number of videos per category to 200, resulting in a total of 1400 prompts and videos. This ensures that the evaluation does not consume excessive computational time and resources.

[Figure 160]

- Figure 11. The stability of Grid-LLaVA with increasing number of videos in Consistent Attribute Binding.

[Figure 161]

- Figure 12. The stability of D-LLaVA with increasing number of videos in Dynamic Attribute Binding.

(a) noun frequency distribution (b) verb frequency distribution

Figure 9. Analysis of word frequencies. We show the frequency of occurrence for nouns and verbs in real-user prompts.

##### A.2. Prompt Generation with LLM

When generating prompts with LLM, we provide the highfrequency nouns, verbs, attributes or other word lists to LLM. The templates used to generate prompts are displayed in section H Table 5, 6, 7, 8, 9, 10 and 11.

##### A.3. Prompt Suite Statistics

After obtaining the 1400 prompts of T2V-CompBench, we use WordNet [45] to identify the metaclasses of nouns and verbs, and their distributions are visualized in Figure 10. In our prompt dataset, the high-frequency nouns are primarily concentrated in the categories of artifact, object, and person, most of which belong to the “thing” categories. Meanwhile, high-frequency verbs are mainly concentrated in actions like “travel”, “move”, and “change”. The occurrence of less dynamic verbs, such as those associated with “express” and “think”, is nearly negligible in our dataset.

##### A.4. Stability of 1400 Prompts and Videos

We conduct an analysis on the stability of our proposed metrics in evaluating the seven categories. Figures 11, 12, 13, 14, 15, 16, and 17 illustrate how the average score changes as the number of videos increases within these categories. Here, we generate one video for each prompt. The analysis shows that the average score tends to stabilize as the number of videos increases. Some metrics, such as G-Dino for

[Figure 162]

- Figure 13. The stability of G-Dino with increasing number of videos in Spatial Relationships.

[Figure 163]

- Figure 14. The stability of DOT with increasing number of videos in Motion Binding.

[Figure 164]

- Figure 15. The stability of Grid-LLaVA with increasing number of videos in Action Binding.

[Figure 165]

- Figure 16. The stability of Grid-LLaVA with increasing number of videos in Object Interactions.

[Figure 166]

Figure 17. The stability of G-Dino with increasing number of videos in Generative Numeracy.

#### B. Implementation Details

We follow the official and default implementations of the T2V models in evaluation. Details of the videos generated by the T2V models, including resolution, total frames, FPS, and duration are presented in Table 3.

Model Resolution Frames FPS Duration (s)

ModelScope [65] 256×256 16 8 2.0 ZeroScope [60] 576×320 36 10 3.6 LVD [32] 256×256 16 8 2.0

AnimateDiff [13] 384×256 16 8 2.0 MagicTime [79] 512×512 16 8 2.0

Show-1 [81] 576×320 29 8 3.6 VideoCrafter2 [6] 512×320 16 8 2.0 VideoTetris [62] 512×320 16 8 2.0 Vico [77] 512×320 16 8 2.0 T2V-Turbo-V2 [29] 512×320 16 8 2.0

Latte [43] 512×512 16 8 2.0

- Open-Sora 1.1 [19] 512×512 16 8 2.0
- Open-Sora 1.2 [19] 640×360 51 24 2.1 Open-Sora-Plan v1.0.0 [33] 512×512 65 24 2.7 Open-Sora-Plan v1.3.0 [33] 640×352 93 18 5.2 CogVideoX-5B [78] 512×320 40 8 5.0 Mochi [61] 512×320 151 30 5.0 Pika-1.0 [51] 1280×720 72 24 3.0

- Gen-2 [56] 1408×768 96 24 4.0
- Gen-3 [57] 1280×768 128 24 5.3 Dreamina 1.2 [4] 1280×720 24 8 3.0 PixVerse-V3 [52] 1408×768 161 30 5.4 Kling-1.0 [26] 1280×720 153 30 5.1

Table 3. Details of generated videos by T2V models. The table shows resolution, total frames, FPS, and video duration in second for videos generated by T2V models.

#### C. Using MLLMs as Evaluation Metrics C.1. How to query MLLMs

One challenge of using MLLMs as evaluation metrics is their tendency to generate hallucinations. This can manifest as mistakes in identifying visual content or selection of unmatched grades or scores.

To mitigate these hallucinations, we employ the chain-

(a) Metric: Grid-LLaVA; Category: Consistent Attribute Binding (b) Metric: Grid-LLaVA; Category: Action Binding

Prompt: A blue car drives past a white picket fence on a sunny day. Prompt: A dog runs through a field while a cat climbs a tree

[Figure 167]

Q: Describe the video,

[Figure 168]

examining objects rendering quality and visual attributes.

Q: Describe the video, highlight the characters and objects, indicate how they act.

A: The image depicts a blue car. The background features a white picket fence…

A: The characters are a dog and a cat. The actions suggest a playful chase…

Q: To evaluate if '{A dog runs through a field while a cat climbs a tree}' is portrayed, please select one option:

Q: Please select one option from A to E:

Q: Please select one option from A to E:

- A: '{a blue car}' is clearly portrayed in all the frames.
- B: '{a blue car}' is present in some frames.
- C: '{a blue car}' is not strictly portrayed
- D: '{a blue car}' is incorrectly portrayed
- E: '{a blue car}' is not present at all.

- A: '{a white fence}' is clearly portrayed in all the frames.
- B: '{a white fence}' is present in some frames.
- C: '{a white fence}' is not strictly portrayed
- D: '{a white fence}' is incorrectly portrayed
- E: '{a white fence}' is not present at all.

- A: Both '{a dog}' and '{a cat}' are clearly present.
- B: Only '{a dog}' is present, '{a cat}' is not depicted.
- C: Only '{a cat}' is present, '{a dog}' is not depicted.
- D: Neither '{a dog}' nor '{a cat}' appears in the video. Choose the best option based on your previous description.

Choose the best option based on your previous description. Put the answer in JSON format with keys: option (e.g., A), explanation, adjust (adjusted option, e.g., C). A:

Choose the best option based on your previous description. Put the answer in JSON format with keys: option (e.g., A), explanation, adjust (adjusted option, e.g., C). A:

- Put the answer in JSON format with keys: option (e.g., A), explanation, adjust (adjusted option, e.g., C). A:

{"option": "A",

"explanation": "Both a dog and a cat are clearly present in the video, as they are shown interacting in a grassy field.",

"adjust": "A"}

Q: Please select options for the two questions:

- Question 1:

- A1: '{a dog runs through a field}' is clearly depicted.
- B1: It is not obvious if '{a dog runs through a field}'.
- C1: '{a dog runs through a field}' is not depicted.

- Question 2:

- A2: '{a cat climbs a tree}' is clearly depicted.
- B2: It is not obvious if '{a cat climbs a tree}'.
- C2: '{a cat climbs a tree}' is not depicted. Put the answer in JSON format with keys: options (e.g., A1,B2), explanation, adjust (adjusted options, e.g., A1,C2). A:

{"option": "A1,C2",

"explanation": "'a dog runs through a field' is clearly depicted, as the dog is chasing the cat across the grassy field. 'a cat climbs a tree' is not depicted. The cat is shown running on the ground, not climbing a tree.",

"adjust": "A1,C2"}

Q: Please select one option:

- A: '{a dog runs through a field}' is clearly depicted.
- B: It is not obvious if '{a dog runs through a field}'.
- C: '{a dog runs through a field}' is not depicted. Put the answer in JSON format with keys: option (e.g., A), explanation, adjust (adjusted option, e.g., B).

Q: Please select one option:

- A: '{a cat climbs a tree}' is clearly depicted.
- B: It is not obvious if '{a cat climbs a tree}'.
- C: '{a cat climbs a tree}' is not depicted. Put the answer in JSON format with keys: option (e.g., A), explanation, adjust (adjusted option, e.g., B).

- B

[Figure 169]

A

- C

[Figure 170]

[Figure 171]

Q: Describe the video, focusing on the interactions between characters or objects.

A: The skateboarder's arms and body position change to maintain balance and control…

Q: To evaluate if this prompt '{A skateboarder performing a trick, launching their board into the air before landing back on the pavement.}' is portrayed, please select one option:

- A: All the objects in the interaction are clearly portrayed.
- B: Some objects in the interaction are not depicted clearly.
- C: Some objects in the interaction are missing.
- D: None of the objects in the interaction are present. Choose the best option based on your previous description.

- Put the answer in JSON format with keys: option (e.g., B), explanation, adjust (adjusted option, e.g., A). A:

{"option": "A",

{"option": "A",

"explanation": "The blue car is clearly portrayed in all the frames, with no significant changes in texture or detail",

"explanation": "A white picket fence is clearly portrayed in all the frames.", "adjust": "A"}

"adjust": "A"}

(c) Metric: D-LLaVA; Category: Dynamic Attribute Binding (d) Metric: Grid-LLaVA; Category: Object Interactions

Prompt: The 3D pixel is transforming from a white square into the shape of a flower. Prompt: A skateboarder performing a trick, launching their board into the air before landing back on the pavement.

[Figure 172]

Q: Please select one option:

- A: Interaction is clear, with objects engaging in a dynamic

manner. The outcome logically follow from the preceding actions and aligns with the prompt.

- B: Interaction is mostly clear, with objects engaging

actively. The outcome generally aligns with the prompt.

- C: Interaction is somewhat clear, but objects show

limited engagement. Development is unclear. Outcome aligns with the prompt but with no previous actions.

- D: Interaction is unclear, with minimal engagement from

the objects. Little to no development. Outcome is vague.

- E: Interaction is nonexistent, with no visible engagement

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

from the objects. The outcome is irrelevant. Put the answer in a JSON format with keys: option (e.g., A), explanation, adjust (adjusted option, e.g., B).

Q: Evaluate if the image aligns with '{white square 3D pixel}' or '{flower-shaped 3D pixel}': 3: the image matches '{white square 3D pixel}'. 2: the image matches '{flower-shaped 3D pixel}'. 1: the image contains objects in both prompts. 0: the image does not match either. Put answer in JSON with keys: score, explanation.

Q1: Describe the image, highlight the visual attributes and states of all the objects.

Q: Evaluate if the image aligns with '{white square 3D pixel}' or '{flower-shaped 3D pixel}': 3: the image matches '{white square 3D pixel}'. 2: the image matches '{flower-shaped 3D pixel}'. 1: the image contains objects in both prompts. 0: the image does not match either. Put answer in JSON with keys: score, explanation.

Q: Evaluate if the image aligns with '{white square 3D pixel}' or '{flower-shaped 3D pixel}': 3: the image matches '{white square 3D pixel}'. 2: the image matches '{flower-shaped 3D pixel}'. 1: the image contains objects in both prompts. 0: the image does not match either. Put answer in JSON with keys: score, explanation. A: …

Q: Please select one option:

A: The image shows a white, three-dimensional rectangular object with glowing effect…

Q1-A: The image shows a single, white, paper flower with a stem…

- A: Interaction of the clearly presented object(s) is dynamic,

with a logical outcome that implies the interaction.

- B: Interaction of the clearly presented object(s) is weak,

with an irrelevant outcome to the prompt. Put the answer in a JSON format with keys: option (e.g., A), explanation, adjust (adjusted option, e.g., B). A:

- B

A

- C

[Figure 177]

Q2: To evaluate if this prompt '{white square 3D pixel}' is described in the image, select one option:

Q3: To evaluate if this prompt '{flower-shaped 3D pixel}' is described in the image, select one option:

{"option": "A",

Q2-A: {"score": "D",

Q3-A: {"score": "B",

"explanation": "The skateboarder is highly dynamic. His movements suggests the progression of a trick, even though the exact moment of launching the board into the air and landing back on the pavement is not depicted.",

"explanation": "The image does not match the prompt at all. The image shows a white paper flower with a stem."}

"explanation": "The image shows a paper flower, which is a 3D object, but it is not pixelated."}

{"option": "B",

- A: Object(s) are clearly visible, visual attributes or states are accurately depicted in the image.
- B: Object(s) are present, but their visual attributes or states are a little different from the prompt.
- C: Object(s)' attributes or states are significantly different from the prompt.
- D: Object(s)' attributes or states are totally different, or the image does not match the prompt at all. Put the answer in JSON format with the keys: score (e.g., C), explanation.

- A: Object(s) are clearly visible, visual attributes or states are accurately depicted in the image.
- B: Object(s) are present, but their visual attributes or states are a little different from the prompt.
- C: Object(s)' attributes or states are significantly different from the prompt.
- D: Object(s)' attributes or states are totally different, or the image does not match the prompt at all. Put the answer in JSON format with the keys: score (e.g., C), explanation.

"explanation": "The skateboarder are clearly portrayed, the action of launching the board into the air and landing back on the pavement is not depicted very clear."

"adjust": "A"}

"adjust": "B"}

Q: Please select one option:

- A: Interaction of the clearly presented object(s) is dynamic and compensates effectively for the missing ones, with a logical outcome that implies the interaction.
- B: Interaction of the clearly presented object(s) is weak, with confusing relationship to the missing ones and an irrelevant outcome to the prompt. Put the answer in a JSON format with keys: option (e.g., A), explanation, adjust (adjusted option, e.g., B).

A: {"score": "A",

A: {"score": "D",

"explanation": "The white, three-dimensional rectangular object matches 'white square 3D pixel'. The object is accurately depicted."}

"explanation": "The image does not match the prompt at all. The object is not flower-shaped, and it is not a pixel."}

- Figure 18. Question and answer examples for MLLM-based evaluation metrics. We replace the parts marked in cyan with metadata for different prompts. Answers are generated by LLaVA [35]. Note that the examples are simplified slightly for clarity and brevity.

of-thought mechanism [70]. We firstly ask the MLLM to describe the image without revealing the specific question we intend to ask. This allows the MLLM to independently describe the visual content, without being affected by the subsequent questions.

of objects, then evaluate the quality of their interaction. Figure 18 shows the question and answer examples for our proposed MLLM-based evaluation metrics.

##### C.2. How to obtain reliableand reproducible results

When querying the MLLM for evaluation, it is important to prepare sufficient grading options to differentiate performances of the models. However, too many options make it difficult for the MLLM to identify the subtle differences between them. To this end, we disentangle the evaluation aspects into parallel or sequential queries, asking about each aspect separately. This ensures a sufficient number of grades while preventing the MLLM from being overwhelmed with too many options at once. For consistent attribute binding, we ask whether each of the two objects possesses the correct attribute. For dynamic attribute binding, we ask whether the image depicts the initial or final state. For action binding, we first inquire about the presence of the two objects. Based on that response, we then ask if the actions of the present object(s) align with the prompt. For object interactions, we start by confirming the presence

It is essential for evaluation metrics to produce reliable and reproducible results. To ensure that the results generated by an MLLM are reproducible, we fix its parameters and seed each time it evaluates a video. However, we empirically find that varying the seed during evaluations can change the ranking of the models, especially when their scores are close. To obtain a more reliable score for each video and ranking for the models, an effective approach is to query the MLLM multiple times and average the resulting scores. This method is similar to asking multiple annotators to rate each video, as they may have different opinions. To decide how many times to query the MLLM, we run Grid-LLaVA 8 times with the same settings, varying only the seed. This is done on videos evaluated by humans for the categories of consistent attribute binding, action binding, and object interactions. We randomly sample 2 or 3 experiments from

the 8 conducted and calculate the average score for each video. We repeat this process 8 times and obtain 8 correlations with human scores for a sample size of 2 and 8 correlations for a sample size of 3. The results are summarized in

- Figure 19. As shown in the box plot, increasing the sample size leads to higher mean and median and generally more stable correlations, which in turn indicates that the scores are more stable. Therefore, considering both reliability and practicality, we decide to sample 3 scores for each video, and use their average as the final result.

Consist-attr Action Interaction

0.9

Mean

CorrelationwithHumanEvaluations(T=0.1)

0.8

| | |
|---|---|
| | |
| | |
| | |

0.7

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

0.6

0.5

0.4

0.3

1 2 3 1 2 3 1 2 3

Sample Size

Figure 19. Box Plot Analysis of Correlation Between Human Evaluations and Grid-LLaVA. We show the box plot of correlations (Spearman’s ρ) between human evaluations and Grid-LLaVA in consistent attribute binding, action binding and object interactions at temperature 0.1 with sample size of 1, 2, and 3.

#### D. Evaluation Results Analysis

As mentioned in Section 3, we divide each category in T2VCompBench into several subgroups. In addition, for certain categories, the final score is determined through multiple stages. Here, we extract the scores for some important subgroups, and also extrapolate some meaningful data from the interim stages. The results for these sub-dimensions are documented in Table 4. In this section, we provide comprehensive explanations for these sub-dimensions and analysis for their results, along with visualizations for each category.

##### D.1. Consistent Attribute Binding

The first three columns of Table 4 present the average score for the subgroups of color, shape, and texture within the category of consistent attribute binding. Among the three subgroups, color is the easiest to manage, followed by tex-

ture, while shape proves to be the most challenging. In Figure 20, we present a concatenation of frames from videos generated by ModelScope [65] and T2V-Turbo-V2 [29]. Not only does T2V-Turbo-V2 [29] accurately represent the color-object binding in the prompt, but it also demonstrates noticeable object movement.

Prompt: A blue car drives past a white picket fence on a sunny day.

[Figure 178]

- (a) T2V Model: ModelScope [65]. Score: 0.5000

[Figure 179]

- (b) T2V Model: T2V-Turbo-V2 [29]. Score: 0.9524

- Figure 20. Visualization of Consistent Attribute Binding examples.

D.2. Dynamic Attribute Binding

For the category of dynamic attribute binding, T2V models rarely produce qualified videos. For example, PixVerseV3 [52] achieves the highest score of 0.0687 in this category. Out of the 200 generated videos, only 31 show relevant elements or transitions, and just 9 of those exhibit meaningful transitions. Models always struggle to generate changing attributes. LVD [32] and T2V-Turbo-V2 [29] exemplify this difficulty in Figure 21. They tend to produce static content that only reflects either the initial or final state of the objects described in the prompt, or both states simultaneously. In this case, PixVerse-V3 [52] demonstrates better prompt understanding and temporal dynamics.

Prompt: A yellow rubber duck in bath water turns a deep purple.

[Figure 180]

(a) T2V Model: LVD [32]. Score: 0.0000

[Figure 181]

- (b) T2V Model: T2V-Turbo-V2 [29]. Score: 0.0000

[Figure 182]

- (c) T2V Model: PixVerse-V3 [52]. Score: 0.8000

- Figure 21. Visualization of Dynamic Attribute Binding examples.

###### Model Consist-attr Spatial-2D Motion Action Interaction

Sub-dimension Color Shape Texture Coexist Acc. Acc.Score Motion Level Acc. Common Uncommon Physical Social diffusion unet-based

ModelScope [65] 0.5826 0.3440 0.4560 62% 44% 0.8011 16.98 32% 0.4025 0.2093 0.3815 0.5411 ZeroScope [60] 0.4829 0.1905 0.3012 50% 65% 0.7836 21.00 29% 0.4042 0.2139 0.2970 0.5422 LVD [32] 0.6155 0.4139 0.4458 63% 90% 0.9054 13.73 47% 0.4120 0.2528 0.3752 0.5252

AnimateDiff [13] 0.4619 0.3113 0.4196 50% 52% 0.8180 12.79 21% 0.3053 0.2009 0.3604 0.4337 Show-1 [81] 0.5964 0.4774 0.5601 62% 56% 0.8387 11.74 26% 0.4331 0.2083 0.5659 0.6830 VideoCrafter2 [6] 0.6717 0.4792 0.5780 73% 59% 0.7914 8.96 24% 0.5472 0.3259 0.5607 0.7122 VideoTetris [62] 0.6957 0.4667 0.5589 74% 53% 0.7831 9.94 21% 0.5410 0.3056 0.6200 0.6956 Vico [77] 0.6705 0.4476 0.4619 80% 50% 0.8083 9.12 20% 0.5609 0.3120 0.5263 0.6652 T2V-Turbo-V2 [29] 0.7781 0.5351 0.5452 77% 54% 0.7967 19.14 37% 0.6694 0.3657 0.5907 0.6970

DiT-based Latte [43] 0.4657 0.3613 0.5280 61% 51% 0.7856 12.76 14% 0.4623 0.2241 0.3904 0.4389

- Open-Sora 1.1 [19] 0.5669 0.4220 0.5304 77% 69% 0.8323 10.87 25% 0.5613 0.2731 0.5070 0.6059
- Open-Sora 1.2 [19] 0.6038 0.4321 0.5607 76% 50% 0.8905 15.98 30% 0.5069 0.3889 0.4396 0.5681 Open-Sora-Plan v1.0.0 [33] 0.4357 0.3887 0.4107 66% 58% 0.7320 7.54 17% 0.4583 0.1713 0.4067 0.4233 Open-Sora-Plan v1.3.0 [33] 0.7107 0.4244 0.5429 76% 59% 0.8726 15.89 22% 0.4938 0.2870 0.4393 0.4574 CogVideoX-5B [78] 0.7202 0.4333 0.5167 71% 61% 0.8706 23.33 34% 0.5824 0.3370 0.5607 0.6530 Mochi [61] 0.7043 0.4393 0.4774 78% 75% 0.8045 18.55 34% 0.5336 0.2454 0.4930 0.5833

commercial Pika-1.0 [51] 0.6095 0.4315 0.5119 72% 55% 0.7841 8.13 17% 0.4650 0.2648 0.4348 0.6048

- Gen-2 [56] 0.6721 0.3994 0.5131 83% 53% 0.8162 7.73 20% 0.4894 0.2491 0.5681 0.6607
- Gen-3 [57] 0.6652 0.5077 0.5280 68% 67% 0.8728 22.59 42% 0.5639 0.3611 0.5663 0.6148 Dreamina 1.2 [4] 0.7798 0.5601 0.5738 90% 62% 0.8777 12.71 23% 0.6225 0.4722 0.6411 0.7237 PixVerse-V3 [52] 0.7898 0.5768 0.6381 89% 71% 0.8231 28.09 38% 0.9016 0.7546 0.8219 0.8400 Kling-1.0 [26] 0.7971 0.4673 0.6149 90% 57% 0.8634 17.27 35% 0.5863 0.5481 0.6530 0.7726

- Table 4. T2V-CompBench sub-dimension evaluation results using proposed metrics. Scores are normalized between 0 and 1. A higher score indicates better performance. Bold signifies the highest score within each category. Blue highlights the top score among diffusion unet-based models. Yellow highlights the top score among DiT-based models. Red highlights the top score among commercial models.

Average Score per Quantity Group

Average Score per Quantity Group

Average Score per Quantity Group

0.9

0.9

0.9

VideoCrafter2

Open-Sora 1.2

Pika-1.0

VideoTetris

Open-Sora-Plan v1.3.0

Dreamina

0.8

0.8

0.8

Vico

CogVideoX-5B

PixVerse-V3

T2V-Turbo-V2

Mochi

Kling-1.0

0.7

0.7

0.7

0.6

0.6

0.6

AverageScore

AverageScore

AverageScore

0.5

0.5

0.5

0.4

0.4

0.4

0.3

0.3

0.3

0.2

0.2

0.2

0.1

0.1

0.1

0.0

0.0

0.0

=2 =3 =4 =5 >=6

=2 =3 =4 =5 >=6

=2 =3 =4 =5 >=6

Quantity

Quantity

Quantity

(a) diffusion unet-based models

(b) DiT-based models

(c) commercial models

- Figure 22. Analysis of text-to-video generation accuracy by object quantity. We plot the average score versus the quantity groups for videos generated by prompts with single object class in Generative Numeracy. This plot draws four T2V models from each of the following groups: (a) diffusion unet-based models, (b) DiT-based models, and (c) commercial models.

##### D.3. Spatial Relationships

To investigate whether T2V models can understand 2D spatial relationships, we extract three key indicators from the evaluated spatial scores. As shown in columns #4-6 of Table 4, the first indicator, “Coexist”, represents the percent-

age of videos that successfully generate both objects involved in a specific 2D spatial relationship. A higher “Coexist” percentage indicates the model is better at generating multiple objects simultaneously. The second indicator, “Acc.” (accuracy), reflects the percentage of videos that accurately depict the spatial relationships among all videos

that successfully generate both objects. A higher “Acc.” suggests a greater likelihood that the model comprehends 2D spatial relationships. The third indicator, “Acc.Score”, is the average score of those videos with correct spatial relationships. A higher “Acc.Score” indicates that the two objects are more distinctly separated, leading to a clearer spatial layout. For the model to generate accurate spatial relationships, both “Acc.” and “Acc.Score” must be high.

According to the results in Table 4, Vico [77] and Mochi [61] demonstrate the best performances in generating multiple objects among open-source models. Although LVD [32] does not achieve a high “Coexist” percentage, it ranks highest in “Acc.” and “Acc.Score” among all models, which verifies its strong capability in layout planning.

Figure 23 showcases a pair of videos generated by Mochi [61]. In this example, the model accurately depicts the spatial relationship between the cat and the fireplace. Notably, the model also achieves a high score in “Acc.”. This suggests that the model may possess some understanding of 2D spatial relationships.

[Figure 183]

- (a) Prompt: A cat sitting on the left of a fireplace. Score: 0.8394

[Figure 184]

- (b) Prompt: A cat sitting on the right of a fireplace. Score: 0.9706

- Figure 23. Visualization of Spatial Relationships examples generated by Mochi [61].

Another notable value is 42% “Acc.” achieved by Gen3 [57], which suggests it may have some understanding of motion direction. Additionally, we can see from “Motion Level” that PixVerse-V3 [52] and CogVideoX-5B [78] can generate significant object motion in their videos.

Figure 24 compares two examples from VideoCrafter2 [6] and Gen-3 [57]. The first features a static background with limited foreground motion, while the second displays noticeable background movement.

Prompt: A golden retriever scampering leftwards across a garden.

[Figure 185]

(a) T2V Model: VideoCrafter2 [6]. Score: 0.2688

[Figure 186]

(b) T2V Model: Gen-3 [57]. Score: 0.4381

- Figure 24. Visualization of Motion Binding examples.

- D.5. Action Binding

Columns #9 and #10 in Table 4 show the average scores for the subgroups of common and uncommon prompts in the category of action binding. Some of the uncommon prompts instruct animals to perform anthropomorphic actions, which are shown in Figure 25. This subgroup is clearly more challenging than the common prompts because executing these uncommon actions is more difficult.

Prompt: A dog plays guitar while a cat takes a selfie.

[Figure 187]

- (a) T2V model: VideoCrafter2 [6]. Score: 0.7037

[Figure 188]

- (b) T2V model: PixVerse-V3 [52]. Score: 0.9630

Figure 25. Visualization of Action Binding examples.

- D.6. Object Interactions

##### D.4. Motion Binding

As described in Section 4, motion vectors are obtained when evaluating motion binding. Therefore, we use these vectors to derive two meaningful indicators, “Motion Level” and “Acc.” for videos generated by prompts with a single object in motion binding. They are recorded in columns #7-8 in Table 4. We first normalize the image to a size of 100x100. “Motion Level” represents the displacement the object travels. Although this indicator does not account for the direction of movement, it reflects the overall motion level and the spatio-temporal dynamics of the video. “Acc.” in this category represents the percentage of videos in which the object moves at least 5 units in the correct direction, among all videos that successfully generate the object. LVD [32] is particularly effective in planning the dynamic scene layouts, allowing it to determine the correct motion directions for objects. This is validated by its “Acc.” of 47%, the highest percentage among all models.

The last two columns in Table 4 record the average scores for the subgroups of physical and social interactions within the category of object interactions. In these two subgroups, depicting physical interactions proves to be more challenging, as it requires an understanding of physical laws. Figure 26 illustrates two examples of physical interactions gen-

erated by Open-Sora-Plan v1.3.0 [33] and VideoTetris [62]. The first example fails to accurately represent the interaction process described in the prompt, while the second effectively captures both the progression and outcome of the interaction, following the prompt more accurately.

Prompt: A skateboarder performing a trick, launching their board into the air before landing back on the pavement.

[Figure 189]

(a) T2V model: Open-Sora-Plan v1.3.0 [33]. Score: 0.5185

[Figure 190]

(b) T2V model: CogVideoX-5B [78]. Score: 1.000

Figure 26. Visualization of Object Interactions examples.

##### D.7. Generative Numeracy

Figure 22 illustrates how the score in generative numeracy changes in relation to the quantity of object specified in the prompt. It is evident that, as the quantity increases, the average score tends to decrease.

Among all models, commercial models generally outperform open-source models. In particular, PixVerseV3 [52] achieves the highest scores across almost all quantity groups. The diffusion unet-based models in Figure 22(a) demonstrate comparable results in terms of numeracy. Among DiT-based models in Figure 22(b), OpenSora 1.2 [19] stands out with the best overall performance.

Figure 27 presents videos generated by ModelScope [65] and Open-Sora 1.2 [19]. Although the video generated by Open-Sora 1.2 [19] is not of realistic style, it successfully represents the correct quantity of object.

Prompt: Four pandas munch bamboo in the garden.

[Figure 191]

- (a) T2V Model: ModelScope [65]. Score: 0.0625

[Figure 192]

- (b) T2V Model: Open-Sora 1.2 [19]. Score: 0.8750

- Figure 27. Visualization of Generative Numeracy examples.

#### E. Human Evaluation

To assess the correlation of the scores given by T2VCompBench metrics with human preferences, we prepare a human evaluation interface for each of the seven categories. The human evaluations are conducted on Amazon Mechanical Turk (AMT). Figure 28 shows the interface for consistent attribute binding.

Annotation Instruction. Each interface includes several key components. First, we clarify the evaluation dimension on which this category is focusing. At the top of each interface, we thoroughly explain the key information in the prompt and the specific dimension of interest. For example, in consistent attribute binding, annotators are instructed to focus solely on the objects and attributes. Next, we provide clear rating criteria along with examples in this category. For each prompt-video example, we provide a detailed rationale for the assigned score based on the specific objects and attributes depicted in the video. Annotators may refer to these examples at any time when they are uncertain about the appropriate rating. Finally, we present the video-text pair that needs evaluation, along with the scoring options (ranging from 5 to 1). To reinforce the rating criteria for the annotators, we include a concise summary of the criteria following each score.

Strategies for Ensuring Quality. To ensure the quality of human evaluations, we employ several strategies:

- • Interface Notification: We include a note in the interface informing annotators that we will review the evaluation results and reject tasks from workers who obviously do not follow the instructions.
- • Random Sampling: We randomly sample 20% of the total tasks completed by each worker. Any task that evidently fails to follow the instructions will be rejected. If an annotator is rejected more than five times, we will block that individual from further participation, and the tasks completed by that annotator will be reassigned for evaluation.
- • Multiple Raters: To balance different opinions, we require three human annotators to rate each task. The average score from these ratings is then assigned to the corresponding video-text pair.
- • Selection of Annotators: We only select experienced and responsible AMT workers by establishing a high historical task acceptance threshold of 90%.
- • Compensation: The estimated hourly wage for each participant is 9.60 USD. We have communicated with several annotators, and they agree that this wage is sufficiently competitive to encourage reliable task ratings. In total, we spent 195 USD on participant compensation.

These measures collectively enhance the reliability and accuracy of our human evaluations.

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

#### G. Limitations and Future Work

While we have made the first step in evaluating compositional T2V generation, there are still many challenges:

- • A limitation of our work is the lack of a unified evaluation metric for all categories, and we believe that this limitation points out new challenges for better and larger multimodal LLMs or video understanding models. If such models become available, we can utilize them as our evaluation metrics.
- • Our benchmark aims to evaluate videos within 2 to 5 seconds. For categories other than motion binding, we sample a fixed number of frames for evaluation, which may not be sufficient for videos longer than 5 seconds. In motion binding, longer videos may lead to greater object displacement and better performance. We leave the evaluation of long videos for future work.

#### H. Templates for Generating Prompts and Metadata Using GPT-4

This section provides the instructions used to prompt GPT4 [47] to generate the text prompts for T2V-CompBench and the corresponding metadata for evaluation. The text prompts in T2V-CompBench and metadata generated for evaluation are available in the code repository.

Table 5, 6, 7, 8, 9, 10, 11 are the input templates used to generate the prompts for the seven categories, each with specific requirements.

Table 12, 13, 14, 15, 16, 17 are the templates used to generate the metadata for evaluation.

- Figure 28. AMT Interface for human evaluation of video-text alignment on consistent attribute binding.

#### F. Societal Impacts

The community should be aware of the potential negative social impact that can arise from the misuse of video generation models, particularly in generating misleading or harmful content, which could exacerbate issues related to misinformation and deepfakes. Additionally, the biases inherent in the training data may lead to the perpetuation of stereotypes or exclusion of underrepresented groups, thereby influencing public perception and societal norms. Therefore, it is imperative that the evaluation of T2V models not only assesses their technical performances but also considers broader social implications, ensuring that these technologies contribute positively to society while mitigating potential risks. To this end, we plan to incorporate an evaluation dimension focused on unbiased composition in the future.

<USER>: This is a list of object nouns: [object nouns]. Please reply ‘Yes’ if received. <ASSISTANT>: Yes <USER>: This is a list of color attributes: [color attributes]. Please reply ‘Yes’ if received. <ASSISTANT>: Yes <USER>: This is a list of active verbs: [verbs]. Please reply ‘Yes’ if received. <ASSISTANT>: Yes <USER>: Please generate 50 prompts featuring two objects with color attributes. Requirements:

- 1. Specify a unique color for each object in the prompt to enrich the visual narrative. Use the color attributes in the color attributes list provided above.
- 2. Select object nouns that have multiple possible colors from the list of object nouns provided above.
- 3. The objects can engage in separate or shared activities. Describe them using vibrant, active verbs, avoiding static verbs such as ‘see’, ‘rest’, or ‘park’. Select verbs from the list of active verbs and feel free to adjust their tense. Here are some examples: A blue car drives past a white picket fence on a sunny day. Yellow rubber duck floating next to a blue bath towel.

- Table 5. Template for generating consistent attribute binding prompts using GPT-4. The attribute type in the template can be replaced by shape or texture; here we take color as an example. The object nouns, color attributes, and verbs marked in cyan should be replaced with words derived from real user prompts.

<USER>: Here are some prompts that describe color & light change, please generation 30 prompts by referring these examples: The leaf is turning from green to yellow. The sky is shifting from bright to dark.

- Table 6. Template for generating dynamic attribute binding prompts using GPT-4. The attribute type in the template can be replaced by shape & size change, texture change or combined change; here we take color & light change as an example. The examples are sourced from TempCompass [41].

<USER>: This is a list of object nouns: [object nouns]. Please reply ‘Yes’ if received. <ASSISTANT>: Yes <USER>: This is a list of active verbs: [verbs]. Please reply ‘Yes’ if received. <ASSISTANT>: Yes <USER>: Please generate 50 pairs of prompts featuring two objects where, in each pair, one object is described as being on the left of the other in the first prompt, and the second prompt should state that the same object is on the right of the other. Requirements:

- 1. Select object nouns from the list provided above.
- 2. The prompt must contain active verbs to ensure the scene is dynamic. You can select verbs from the list of active verbs and feel free to adjust their tense. Here are some examples: ‘A dog running on the left of a bicycle’ and ‘A dog running on the right of a bicycle’. ‘A llama walking on the left side of a motorcycle in a city street’ and ‘A llama walking on the right side of a motorcycle in a city street’.

- Table 7. Templates for generating spatial relationships prompts using GPT-4. The spatial relationship in the template can be replaced by above&below or in front of&behind; here we take left&right as an example. The object nouns and verbs marked in cyan should be replaced with words derived from real user prompts.

<USER>: This is a list of object nouns: [object nouns]. Please reply ‘Yes’ if received. <ASSISTANT>: Yes <USER>: Please generate 50 pairs of prompts where, in each pair, an object is being describe as moving leftwards in the first prompt and the second prompt should state the same object is moving rightwards. Requirements:

- 1. Select object nouns from the list provided above.
- 2. Describe the objects using vibrant, active verbs. Here are some examples: ‘A golden retriever scampering leftwards across a garden’ and ‘A golden retriever scampering rightwards across a garden’. ‘A football rolling from the right to the left on the grass’ and ‘A football rolling from the left to the right on the grass’.

- Table 8. Templates for generating motion binding prompts using GPT-4. The moving direction in the template can be replaced by upwards&downwards; here we take leftward&rightwards as an example. The object nouns marked in cyan should be replaced with words derived from real user prompts.

<USER>: This is a list of object nouns: [object nouns]. Please reply ‘Yes’ if received. <ASSISTANT>: Yes <USER>: This is a list of active verbs: [verbs]. Please reply ‘Yes’ if received. <ASSISTANT>: Yes <USER>: Please generate 50 prompts that describe an object engaging in an activity, while another object is engaging in a different activity. Requirements:

- 1. Select the two object nouns from the list of object nouns provided above.
- 2. Specify a unique verb for each object in the prompt.
- 3. Use vibrant, active verbs, avoiding static verbs such as ‘see’, ‘rest’, or ‘park’. You can select verbs from the list of active verbs and feel free to adjust their tense. Here are some examples: A dog runs through a field while a cat climbs a tree. A man takes photos and a boy dances on the street.

- Table 9. Templates for generating action binding prompts using GPT-4. The object nouns, marked in cyan should be replaced with words derived from real user prompts.

<USER>: This is a list of object nouns: [object nouns]. Please reply ‘Yes’ if received. <ASSISTANT>: Yes <USER>: Please generate 50 prompts describing physical interaction between two objects that can change their state of motion. Requirements:

- 1. For the two objects involved in the interaction, select at least one object noun from the list of object nouns provided above.
- 2. Use vibrant, active verbs, avoiding static verbs such as ‘see’, ‘rest’, or ‘park’. Here are some examples: Two cars collide at an intersection. A dog dragging a blanket off a bed, leaving it tangled on the floor.

- Table 10. Templates for generating object interactions prompts using GPT-4. The physical interaction in the template can be replaced by social interaction; here we take physical interaction as an example. The object nouns marked in cyan should be replaced with words derived from real user prompts.

<USER>: This is a list of object nouns: [object nouns]. Please reply ‘Yes’ if received. <ASSISTANT>: Yes <USER>: This is a list of active verbs: [verbs]. Please reply ‘Yes’ if received. <ASSISTANT>: Yes <USER>: Please generate one prompt for each object in the list [object nouns] with a quantity of two. Use vibrant and active verbs in the prompt. You can select verbs from the list provided above and feel free to adjust their tense. Here are some examples: Two cats walking on the floor. Two dogs running through a field.

- Table 11. Templates for generating numeracy prompts using GPT-4. The quantity of ‘two’ in the template can be replaced by ‘three’, ‘four’, ‘five’, ‘six’, ‘seven’ and ‘eight’; here we take ‘two’ as an example. The object nouns marked in cyan should be replaced with words derived from real user prompts, here we restrict the length of the list to 30.

<USER>: I will provide you with several prompts, each on a separate line. Each prompt will describe two objects, each associated with an attribute. For each prompt, you are to create a JSON object. This JSON object should include two keys: “prompt” and “phrases”. The “prompt” key should simply replicate the original prompt text. The “phrases” key should contain the two described objects separated by a semicolon. For instance, for the prompt “A blue car drives past a white picket fence on a sunny day”, the value under “phrases” should be “a blue car; a white picket fence”. Below are the prompts for which you need to generate the JSON objects: A blue car drives past a white picket fence on a sunny day. Yellow rubber duck floating next to a blue bath towel.

...... <ASSISTANT>: Here are the JSON representations for the prompts: {“prompt”: “A blue car drives past a white picket fence on a sunny day”, “phrases”: “a blue car; a white picket fence”}, {“prompt”: “Yellow rubber duck floating next to a blue bath towel”, “phrases”: “yellow rubber duck; a blue bath towel”},

......

- Table 12. Template for generating the metadata of consistent attribute binding prompts using GPT-4.

<USER>: I will provide you with several prompts, each on a separate line. Each prompt describes a color, shape, texture or combined change of one or multiple objects. For each prompt, you are to create a JSON object. This JSON object should include the following keys: “prompt”: simply replicate the original prompt text. “state 0”: the object(s) with initial states. “state 1”: the object(s) with final states. For instance, for the prompt “The sky is shifting from bright to dark”, “state 0” should be “bright sky”, “state 1” should be “dark sky”. For the prompt “a leaf is transitioning from green to yellow and a fresh apple is decaying”, “state 0”: “a green leaf and a fresh apple”; “state 1”: “a yellow leaf a decayed apple”. Below are the prompts for which you need to generate the JSON objects: The sky is shifting from bright to dark. The leaf is turning from green to yellow.

...... <ASSISTANT>: Here are the JSON representations for the prompts: {“prompt”: “The sky is shifting from bright to dark”, “state 0”: “bright sky”, “state 1”: “dark sky”}, {“prompt”: “The leaf is turning from green to yellow”, “state 0”: “green leaf”, “state 1”: “yellow leaf”},

......

- Table 13. Template for generating the metadata of dynamic attribute binding prompts using GPT-4.

<USER>: I will provide you with several prompts, each on a separate line. For each prompt, you are to create a JSON object. This JSON object should include the following keys: “prompt”: simply replicate the original prompt text. “spatial”: contain the word from the prompt that represents the spatial relationship. Choose one from the following six options: left, right, above, below, in front of, and behind. For example, if the prompt uses the word “under”, set the value of “spatial” to “below” because these words have the same meaning.

- “object 1” and “object 2”: These keys represent the two objects mentioned in the prompt. The value of “object 1” should be positioned to the left of, right of, above, below, in front of, or behind “object 2” based on the spatial word used in the prompt. Here’s an example to illustrate the format: Prompt: “A toddler walking on the left of a dog in a park” JSON object: {“prompt”: “A toddler walking on the left of a dog in a park”, “spatial”: “left”, “object 1”: “toddler”,

- “object 2”: “dog”} Below are the prompts for which you need to generate the JSON objects: A bird flying on the left side of a hot air balloon. A llama walking on the left side of a motorcycle in a city street.

...... <ASSISTANT>: Here are the JSON objects for each of your prompts, accurately capturing the spatial relationship, object 1, and object 2: {“prompt”: “A bird flying on the left side of a hot air balloon.”, “spatial”: “left”, “object 1”: “bird”, “object 2”: “hot air balloon”}, {“prompt”: “A llama walking on the left side of a motorcycle in a city street.”, “spatial”: “left”, “object 1”: “llama”, “object 2”: “motorcycle”},

......

- Table 14. Template for generating the metadata of spatial relationships prompts using GPT-4.

<USER>: I will provide you with several prompts, each on a separate line. Each prompt describes the movement of one or two objects in different directions. For each prompt, you are to create a JSON object. This JSON object should include the following keys: “prompt”: simply replicate the original prompt text.

- “object 1”: the first object mentioned in the prompt that moves in a direction.

- “d 1”: motion direction for “object 1” and can only be one of the following: left, right, up, or down.

“object 2”: If there is a second object mentioned in the prompt that moves in a direction, this key will contain its name. Otherwise, the value will be an empty string.

- “d 2”: If there is “object 2”, this key will represent its motion direction. Otherwise, the value will be an empty string. Below are the prompts for which you need to generate the JSON objects: A golden retriever scampering leftwards across a garden A ball rolls left while a puppy dashes right

...... <ASSISTANT>: Here are the JSON representations for each prompt: {“prompt”: “A golden retriever scampering leftwards across a garden”, “object 1”: “golden retriever”, “d 1”: “left”, “object 2”: “”, “d 2”: “”} {“prompt”: “A ball rolls left while a puppy dashes right”, “object 1”: “ball”, “d 1”: “left”, “object 2”: “puppy”, “d 2”: “right”},

......

- Table 15. Template for generating the metadata of motion binding prompts using GPT-4.

<USER>: I will provide you with several prompts, each on a separate line. For each prompt, your task is to extract the prompt into a noun phrase and a noun phrase with an action. For example, given the prompt “A dog runs through a field while a cat climbs a tree”, the output should be: {“prompt”: “A dog runs through a field while a cat climbs a tree”, “phrase 0”: [“a dog?”, “a dog runs through a field?”], “phrase 1”: [“a cat?”, “a cat climbs a tree?”]}. Below are the prompts for which you need to generate the JSON objects: A dog runs through a field while a cat climbs a tree. A man takes photos and a boy dances on the street.

...... <ASSISTANT>: Here are the extracted phrases for each prompt: {“prompt”: “A dog runs through a field while a cat climbs a tree”, “phrase 0”: [“a dog?”, “a dog runs through a field?”], “phrase 1”: [“a cat?”, “a cat climbs a tree?”]}, {“prompt”: “A man takes photos and a boy dances on the street”, “phrase 0”: [“a man?”, “a man takes photos?”], “phrase 1”: [“a boy?”, “a boy dances on the street?”]},

......

- Table 16. Template for generating the metadata of action binding prompts using GPT-4.

<USER>: I will provide you with several prompts, each on a separate line. Each prompt describes a certain number of objects engaging in an activity, where each object is associated with a specific quantity. For example, the prompt may include phrases like “two cat” or “three dogs”. For each prompt, you are to create a JSON object. This JSON object should include the following keys: “prompt”: simply replicate the original prompt text. “objects”: This key will contain the objects mentioned in the prompt. If there are multiple objects, they will be separated by commas. “numbers”: This key will represent the corresponding numbers associated with each object. The numbers will be concatenated with commas. When describing the action the objects are performing and the environment, there may be mentions of other objects, articles like “a road”, “the woods” or other unnecessary details. Please ignore these and focus only on extracting the main objects and their respective quantities. Below are the prompts for which you need to generate the JSON objects: Three cars speed along a road. Two pandas munch on bamboo, and three elephants spray water.

...... <ASSISTANT>: Here are the JSON representations for each prompt: {“prompt”: “Three cars speed along a road”, “objects”: “car”, “numbers”: “3”}, {“prompt”: “Two pandas munch on bamboo, and three elephants spray water”, “objects”: “panda,elephants”, “numbers”: “2,3”},

......

Table 17. Template for generating the metadata of generative numeracy prompts using GPT-4.

