## VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness

Dian Zheng*, Ziqi Huang*, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng , Yu Qiao , Ziwei Liu

### arXiv:2503.21755v2[cs.CV]20Aug2025

Abstract—Video generation has advanced significantly, evolving from producing unrealistic outputs to generating videos that appear visually convincing and temporally coherent. To evaluate these video generative models, benchmarks such as VBench have been developed to assess their faithfulness, measuring factors like per-frame aesthetics, temporal consistency, and basic prompt adherence. However, these aspects mainly represent superficial faithfulness, which focus on whether the video appears visually convincing rather than whether it adheres to real-world principles. While recent models perform increasingly well on these metrics, they still struggle to generate videos that are not just visually plausible but fundamentally realistic. To achieve real “world models” through video generation, the next frontier lies in intrinsic faithfulness to ensure that generated videos adhere to physical laws, commonsense reasoning, anatomical correctness, and compositional integrity. Achieving this level of realism is essential for applications such as AI-assisted filmmaking and simulated world modeling.

To bridge this gap, we introduce VBench-2.0, a next-generation benchmark designed to automatically evaluate video generative models for their intrinsic faithfulness. VBench-2.0 assesses five key dimensions: Human Fidelity, Controllability, Creativity, Physics, and Commonsense, each further broken down into fine-grained capabilities. Tailored to individual dimensions, our evaluation framework integrates generalists such as state-ofthe-art VLMs and LLMs, and specialists, including anomaly detection methods proposed for video generation. We conduct extensive human preference annotations to ensure evaluation alignment with human judgment. By pushing beyond superficial faithfulness toward intrinsic faithfulness, VBench-2.0 aims to set a new standard for the next generation of video generative models in pursuit of intrinsic faithfulness.

Index Terms—Video Generative Models, Evaluation Benchmark.

I. INTRODUCTION

# V

IDEO generation aims to create realistic and temporally coherent video sequences, with a wide range of applica-

tions in video editing [1]–[14], customization [15]–[17], image animation [18], [19], and world models [20].

Earlier video generative models [21]–[24] primarily focused on generating short video clips of around two seconds, emphasizing fundamental capabilities like per-frame aesthetics and temporal consistency. To systematically evaluate these capabilities, benchmarks [25]–[27] such as VBench [25], [26] have

* equal contribution. corresponding authors. Email: Dian Zheng zd1423606603@gmail.com and Ziqi Huang

ziqi002@ntu.edu.sg

D. Zheng, H. Liu, K. Zou, F. Zhang, L. Gu, J. He, Y. Qiao are with Shanghai Artificial Intelligence Laboratory. Z. Huang, Y. Zhang, Z. Liu are with the S-Lab, Nanyang Technological University. W. Zheng is with Sun Yat-sen University. J. He is also with The Chinese University of Hong Kong.

been developed to assess aspects like per-frame aesthetics, frame-to-frame temporal smoothness, and adherence to simple text prompts, which we refer to as superficial faithfulness, the degree to which generated videos appear visually convincing. As video generative models continue to evolve, recent stateof-the-art models, including Sora [28], Kling [29], Gen-3 [30], HunyuanVideo [31], and Veo 2 [32], have demonstrated strong performance on these metrics, and many aspects of superficial faithfulness are now approaching saturation.

However, as video generation moves towards more advanced applications, particularly in areas that require AI models to simulate and reason about the real world [20], such as AI-driven storytelling and video-generation-based simulation, the new frontier shifts from merely appearing real to being intrinsically real. We term this as intrinsic faithfulness - a concept that extends beyond per-frame quality and smooth motion, requiring that generated videos adhere to deeper principles such as physical laws, commonsense reasoning, anatomical correctness, and compositional integrity. Achieving this level of faithfulness is essential for applications ranging from AI-assisted filmmaking to virtual environments for embodied intelligence, ultimately paving the way for the development of true world models that can accurately represent and predict real-world dynamics.

To drive video generative models towards this next generation of capabilities, we introduce VBench-2.0, a benchmark suite designed to evaluate video generative models along five emerging dimensions beyond superficial faithfulness: Human Fidelity, Controllability, Creativity, Physics, and Commonsense. Each dimension is further broken down into isolated sub-abilities (shown in Figure 1(b)), providing a fine-grained assessment of the intrinsic faithfulness of video generative models. Given the complexity of these evaluations, we leverage generalists such as state-of-the-art Video Language Models (VLMs) and Large Language Models (LLMs) to perform structured reasoning and judgment. Specifically, we design two complementary evaluation methods: 1) text description alignment to assess abstract concept and semantic understanding, leveraging modern VLM’s strong captioning and LLM’s reasoning ability, and 2) video-based multi-question answering for basic visual understanding. To enhance evaluation robustness in specific domains, we incorporate specialists, such as human anomaly detection pipelines trained for generated videos. Furthermore, we use evaluation safeguards like prefiltering, and redundant questioning to mitigate hallucinations and inconsistencies in our tailored evaluation pipelines for each dimension. Additionally, we follow VBench [25], [26]

Evolution of Video Generative Models’ Capabilities

#### VBench-2.0 Evaluation Dimension Suite

Human Anatomy

VBench VBench-2.0

Human Fidelity

Identity

Temporal Consistency

Creativity

Clothes

Frame-to-Frame Temporal Consistency

Diversity Composition

Human Fidelity

Creativity

Aesthetic Quality

Adherence to Complex

Dynamic Spatial Relationship Dynamic Attribute

Prompts

Entity

Imaging Quality

Commonsense

Adherence to Simple Prompts

Motion Order Understanding Human Interaction Complex Landscape Complex Plot

###### VBench-2.0

Event

Physics

Controllability

Content

Superficial Faithfulness Intrinsic Faithfulness

time

Camera Motion

Motion Rationality Instance Preservation

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

| |
|---|

Commonsense

but

Mechanics

Thermotics Geometry

State Change

Physics

Material Multi-View Consistency

visually unreal

visually real intrinsically unreal

intrinsically real

(a) Scope of VBench-2.0 (b) VBench-2.0 Evaluation Dimension Suite

- Fig. 1: Overview of VBench-2.0. (a) Scope of VBench-2.0. Video generative models have progressed from achieving superficial faithfulness in fundamental technical aspects such as pixel fidelity and basic prompt adherence, to addressing more complex challenges associated with intrinsic faithfulness, including commonsense reasoning, physics-based realism, human motion, and creative composition. While VBench primarily assessed early-stage technical quality, VBench-2.0 expands the benchmarking framework to evaluate these advanced capabilities, ensuring a more comprehensive assessment of next-generation models. (b) Evaluation Dimension of VBench-2.0. VBench-2.0 introduces a structured evaluation suite comprising five broad categories and 18 fine-grained capability dimensions.

Human Anatomy Human Identity

Human Clothes

Diversity

Composition

Dynamic Spatial Relationship

Dynamic Attribute

Motion Order Understanding

Human Interaction

Complex Landscape

Complex Plot

Camera Motion

Motion Rationality

Instance Preservation

Mechanics

Themotics

Material

Multi-View Consistency

HunyuanVideo CogVideoX-1.5 Sora-480p Kling 1.6

- Fig. 2: VBench-2.0 Evaluation Results of SOTA Models. The figure presents the evaluation results of four recent stateof-the-art video generation models across 18 VBench-2.0 dimensions. The results are normalized per dimension for a clearer comparison. For detailed numerical results, refer to Table II.

unstable in commonsense reasoning, highlighting key open challenges in video generation towards synthesizing the world with intrinsic faithfulness. We provide an in-depth discussion on possible causes, potential solutions, inherent trade-offs, and future work in Section V.

VBench-2.0 will be fully open-sourced, being complementary to VBench [25], [26], and providing a standardized framework for evaluating future breakthroughs in video generation. While VBench remains essential for assessing superficial faithfulness, VBench-2.0 extends the evaluation scope to intrinsic faithfulness, addressing deeper aspects of video realism. We will continually integrate newly released video generative models into VBench-2.0. By setting a higher standard for evaluation, VBench-2.0 aims to play a pivotal role in guiding the development of next-generation video generative models. Together, VBench and VBench-2.0 form a comprehensive benchmarking system, driving the field beyond superficial faithfulness towards truly intrinsically faithful video generation.

II. RELATED WORKS A. Video Generative Models

With the advancements in diffusion models [33]–[44], variational autoencoder-based compression techniques [45]–[49], and transformer architectures [50], [51], video generation has emerged as one of the most dynamic frontiers in artificial intelligence research. Prior to Sora’s breakthrough, predominant text-to-video models primarily focused on synthesizing short video clips (2-3 seconds duration) [2], [18], [21]–[24], [28], [32], [52]–[60] through incremental improvements in visual fidelity and temporal consistency. Sora [28] pioneered the scaling paradigm in video generation by demonstrating unprecedented model capacity through large-scale training, paving the

and conduct extensive human preference annotations to validate and align our automated evaluation results with human judgment.

Our evaluation provides comprehensive insights into the strengths and weaknesses of state-of-the-art video generative models. While recent models demonstrate emerging abilities in human anatomy, consistency, and some degree of novel creativity, they still struggle with generating complex plots, handling simple dynamic changes in objects, and remain

TABLE I: Comparison of Video Generation Benchmarks. We compare existing video generation benchmarks based on their evaluation aspects. VBench-2.0 is the first comprehensive benchmark to assess intrinsic faithfulness in video generation, complementing VBench [25], [26]. Detailed aspects include per-frame quality (Frame Wise), temporal consistency (Temp Cons), adherence to simple prompts (Simp Pmpt), compositional creativity (Comp Crea), commonsense reasoning (Com Sense), physics-based realism (Phy), human anatomy (Human Anat), and adherence to complex prompts (Cplx Pmpt).

Superficial Faithfulness Intrinsic Faithfulness

Frame Wise Temp Cons Simp Pmpt Comp Crea Com Sense Phy Human Anat Cplx Pmpt VBench [25], [26] ✓ ✓ ✓

T2V-CompBench [73] ✓ ✓ ✓ PhyGenBench [72] ✓

StoryEval [74] ✓ ✓ VBench-2.0 (Ours) ✓ ✓ ✓ ✓ ✓ ✓

way for the development of next-generation video foundation models [20], [29]–[31], [59], [61]–[66] that achieve remarkable visual quality and robust spatiotemporal coherence. They have shifted focus toward enhancing video generation adhere to deeper principles such as physical laws and commonsense reasoning that focuses more on action continuity and realistic physical-world perception [20], [28], [59], or high-quality human-centric generation with creativity potential [29], [31]. Existing benchmarks cannot systematically evaluate these new explorations, and VBench-2.0 takes the initiative to provide a comprehensive benchmark for evaluating emerging capabilities towards the goal of achieving intrinsic faithfulness through video generation.

B. Evaluation of Video Generative Models

Initially, video generative models primarily relied on conventional evaluation metrics such as Fr´echet inception distance (FID) [67], Inception Score (IS) [68], and Fr´echet video distance (FVD) [69]. However, these metrics provided limited insight into the diverse and complex capabilities of modern video generation. Recent evaluation frameworks [25]– [27], [70], [71] such as VBench [25], [26] introduced a more structured approach by disentangling evaluation into multiple capability dimensions, enabling more detailed and interpretable assessments. These benchmarks focus on fundamental technical attributes such as per-frame quality, temporal consistency, and basic prompt adherence. However, as models continue to improve, certain dimensions within VBench begin to saturate, necessitating broader evaluation scope that assess deeper aspects of intrinsic faithfulness in video generation. To address this, specialized benchmarks have emerged. PhyGenBench [72] evaluates a model’s understanding of physical laws through Vision-Language Models (VLMs). T2VCompBench [73] assesses compositionality, including motion, actions, spatial relationships, and attributes. StoryEval [74] focuses on storytelling capabilities by aggregating the responses from two VLMs. Unlike prior benchmarks, which either focus on fundamental capabilities [25]–[27] or specific emerging domains [72], [73], VBench-2.0 introduces a comprehensive framework to systematically evaluate next-generation video generation capabilities, bridging the gap in the evolving landscape of video generation.

III. VBENCH-2.0 SUITE FOR INTRINSIC FAITHFULNESS

In this section, we introduce the evaluation framework of VBench-2.0. Section III-A presents the five key evaluation dimensions and their respective assessment methods. Unlike VBench, which primarily evaluates superficial faithfulness, VBench-2.0 introduces a suite of tests to assess intrinsic faithfulness, focusing on deeper properties such as physics, commonsense, and creativity, human, and controllability. To ensure robust evaluation, we integrate multiple assessment methodologies, including LLM-assisted text alignment, videobased multi-question answering, and specialist models trained for anomaly detection. Prompt suite is introduced in Section III-B and Section III-C describes the human annotation pipeline of VBench-2.0, including the collection and processing of video data, labeling of human annotations, and the definition of scoring criteria.

A. Evaluation Dimension Suite

VBench-2.0 evaluates video generation along five key dimensions: Human Fidelity, Creativity, Controllability, Physics, and Commonsense. Each dimension is further decomposed into sub-dimensions, ensuring a fine-grained assessment of a model’s capabilities. We employ a structured approach that combines generalist reasoning models (VLMs/LLMs) with specialist detectors. When using generalist, we adopt two evaluation schemes where each is tailored to different types of semantic understanding required across evaluation dimensions. Text Description Alignment. This scheme is suitable for complex or subtle scenarios, such as those involving nuanced human interactions or multi-step plots. In these cases, the composite scene is broken down and interpreted step by step by the Vision-Language Model (VLM), which generates a descriptive caption guided by system prompts that focused on specific aspects of the video (e.g., prompting only about human interactions or a specific part of the plots). The correctness of the generated content is then judged by a Large Language Model (LLM), which compares the VLM-generated caption with a ground-truth reference. The reference may be the original text prompt, a predefined answer, or relevant metadata. This process is formalized as:

###### Answer = LLM(V LM(V |Sv),T | Sl), (1)

assess the consistency of human identity and clothing across frames.

|[cls] token|
|---|

Anomaly Classifier

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Question: Is there any anatomy abnormal in the video?

[Figure 12]

SoftMax

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

0.2

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

SimMIM

[Figure 22]

BCE Loss

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

- (a) many abnormalities (score: 0.24)
- (b) minor abnormalities (score: 0.91)

[Figure 27]

Normal

0

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Trainable Modules

##### …

…

Image Features

[Figure 37]

[cls] token Features

Fig. 4: Visualization for Human Anatomy.

- Fig. 3: Human Anatomy Detector Framework. The [cls] token is used to aggregate information related to anomalies in the image. The human body, hand, and face detectors share the same pipeline with different data. The input is frame level and pre-detected.

(1-a) Human Anatomy. This dimension focuses on detecting potential anomalies in the appearance and structure of the human body in videos. We use a pre-trained ViT-base model [77] to train three anomaly detection models, which will detect the human in each frame and judge the anomaly score of each human, targeting the human body, hands, and faces, respectively. The classification token (cls_token) is passed through an MLP to produce binary outputs. We show the pipeline in Figure 3 and the dataset consists of two components:

where V is the generated video, T denotes the text prompt or crafted reference, and Sv and Sl are the system prompts for the VLM and LLM, respectively. The LLM outputs a binary judgment (“yes” or “no”) that is discretized to a score of 1 or 0 to reflect caption-reference matching. This scheme excels in semantic understanding dimensions like Complex Plot and Human Interaction, where VLMs usually struggle with high-level interpretation, but LLMs demonstrate stronger reasoning capabilities. Thus, we decouple caption generation (by VLMs) from semantic alignment (by LLMs) to improve evaluation reliability. Unless otherwise specified, we adopt LLaVA-Video-7B [75] as our VLM model and Qwen2.5-7BInstruct [76] as LLM model.

- • Real samples: We collected approximately 1,000 real motion videos from the web and used the YOLO-World [78] detection model to extract pre-frame patches of human bodies, hands, and faces as positive samples.
- • Generated samples: We generated around 1,000 videos related to human motion using CogVideo [24], [59] and HunyuanVideo [31]. These were processed with YOLOWorld to extract image patches, followed by meticulous manual annotations for training. Additionally, negative samples were sampled from the HumanRefiner [79] dataset, focusing on human body, hands, and face.

Video-Based Multi-Question Answering. It is designed for evaluation dimensions where one salient concept is prominent and can be directly queried through video question answering (VQA). In this approach, we construct a series of complementary and sometimes redundant questions to reduce the risk of accidental errors and ask the VLM to perform direct VQA. The formalization is as follows:

As a result, we collect 150k labeled real and generated human frame-level data. During inference, we first applied the YOLOWorld model to detect all human instances in the input video. For each detected human region, we further detected hands and faces, extracting corresponding patches as inputs. A human instance is flagged as abnormal if any of the three models consider this human as an anomaly and the final score is the percentage of frames that are not flagged as abnormal.

N

V QA(Qi,V | S), (2)

Answer =

i

- (1-b) Human Temporal Consistency - Clothes. Clothing con-

sistency is assessed using video-based multi-question answering to assess whether outfits remain stable throughout video. Note that we do not use traditional methods such as feature similarity calculation to handle this problem based on two considerations: 1) Current models often generate characters with their clothing partially obscured by objects or with changes in visible body parts (e.g., the upper body transforming into the lower body). In such cases, traditional algorithms are unable to address these cross-temporal judgment issues. 2) LLaVAvideo-7B has strong clothing color perception and a certain degree of cross-temporal memory capability. Therefore, we use it to evaluate this dimension.

- (1-c) Human Temporal Consistency - Identity. Identity con-

where Q is a set of multiple questions. For example, in the Dynamic Attribute dimension focusing on color changes, we may ask: 1. Initially, is the color of the river mostly blue? 2. Finally, is the color of the river mostly brown? 3. Does the color of the river change? The answer to each question is binary (“yes” or “no”), and scores are either averaged or awarded only if all responses are correct, depending on the dimension’s scoring scheme. This scheme is particularly effective for surface-level visual understanding, where modern VLMs can confidently answer targeted queries without requiring high-level semantic reasoning. Unless otherwise specified, we adopt LLaVA-Video-7B [75] as our VLM model.

- 1) Human Fidelity: We evaluate both the structural correctness and temporal consistency of human figures in generated videos. Structural issues commonly seen in current video generation models, such as sudden turns or the “thousand-hand yoga” effect, are considered. For temporal consistency, we

sistency is evaluated by measuring facial feature similarity using ArcFace [80], with face detection performed by RetinaFace [81]. Specifically, we select the first frame of the

Question: Which video has higher diversity?

Prompt

- 1." The camera starts with a high-altitude view of an endless ice plain, where the snow-covered land stretches out as far as the eye can see. ",
- 2." The cold air gives everything a still, pristine quality, with animal tracks in the snow. ",
- 3." The night sky becomes clearer, and the aurora borealis begins to dance across the heavens in vibrant colors. ",
- 4." The camera follows the shifting aurora, contrasting with the white snow and creating a magical atmosphere. ",
- 5." The camera pulls back, revealing the vast ice plains interspersed with the aurora, as snow and lights mirror each other. "

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

- (a) matching 0 prompt (score: 0.0)
- (b) matching 1 prompt (score: 0.2)

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Question: Is there any anatomy abnormal in the video?

Fig. 6: Visualization for Complex Landscape.

(a) B has higher diversity (score A: 0.34 < score B: 0.70)

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

Fig. 5: Visualization for Diversity.

the text prompt. Specifically, we first split the text prompt into two descriptions with action only in turn, then we obtain the video caption by VLM with an additional system prompt “Return the action order in video. Here is the template: ‘1. ; 2. .”’. Then we assess whether both actions match the ground truth descriptions in turn; only when both actions match are they considered correct. This approach cleverly the hallucination problem of LLM in order judgment.

- (a) minor abnormalities (score: 0.24)
- (b) many abnormalities (score: 0.91)

video as the anchor by default, and all subsequent frames are compared to the first frame to calculate similarity. However, time periods where there are multiple people or no one present are not taken into consideration. Note that after a scene change, whether it is still the same person is taken into consideration, which represents a more challenging scenario.

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

Question: Does the clothes of human remain consistency during the video?

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

| |
|---|

| |
|---|

|avoids|
|---|

- 2) Creativity: We evaluate creativity by analyzing a model’s ability to generate diverse outputs and complex compositions beyond real-world constraints.

- (2-a) Diversity. Given a text prompt, we sample 20 videos

from a model, and measure inter-sample variation using style and content diversity metrics, computed from pre-trained VGG-19 [82] feature representations ( i.e., the metric is modified from [83]).

- (2-b) Composition. This dimension measures whether the

model can generate novel and uncommon compositions and we assess it through three sub-dimensions: species combination, single-entity actions, and multi-entity interactions, using video-based multi-question answering pipeline. We additionally observe that pure VQA can not ensure robust results as current video generation models will generate separate species for the given prompt and VQA is a discrete process that could not consider the separate situations. So in this paper, we omit the cases that show more than one creature in the video by a pre-VQA question “Is there only one creature in the video?”.

- 3) Controllability: We evaluate a model’s ability to follow complex prompts and simulate dynamic changes during video generation. This dimension measures how accurately the model can render specific entities, events, content, and camera movements in response to detailed textual instructions.

- (a) no (score: 0)
- (b) yes (score: 1)

- (3-d) Event - Human Interaction. We assess whether two

humans can interact (e.g., “One person hands an object to another”) based on the text prompt using text description alignment. The system prompt for VLM is “Describe the human interaction in the video, following the template as [a person xx to another person.]”. After converting the description into a standardized format, the LLM will determine whether the standardized description matches the ground truth. However, the wrong number of people may lead to interaction hallucinations in VLMs, such as treating situations with only one person as interactions between two individuals, so we pre-filter the video with wrong human number with another text description alignment pipeline. Specifically, we obtain the dense description of the video with system prompt “Describe the video in detail” and ask the LLM to determine whether the number of people mentioned in the video description is exactly one.

- (3-e) Content - Complex Plot. We test a model’s ability

to construct multi-scene narratives from prompts describing multi-stage events (e.g., a five-act story with 150+ words) to meet the demands of future movie-level video generation. We utilize text description alignment to evaluate it. Specifically, we first manually split and summarize the long text prompt into 5 plot descriptions as ground truth. Then we feed the video into VLM to obtain 5 video captions with the system prompt “Return the plot in video. Here is the template: [1. ; 2. ; 3. ; 4. ; 5. .]” ( i.e., note that when dealing with long videos at the minute level, VLMs may make errors in their pointby-point summarization. Therefore, we additionally employ an LLM for post-processing verification to correct captions with incorrect numbering). Finally, we sequentially match the corresponding text and captions in the order of the plot. If the LLM determines that a plot element appears in the caption, we proceed to evaluate the next plot element; otherwise, the evaluation stops. We discretize the plot elements to calculate the accurate score, simulating the effect of LLM scoring.

- (3-f) Content - Complex Landscape. We evaluate whether

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

Question: Whether the soda can be squeezed as air being gradually and forcefully removed.

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

- (a) The soda can remains unchanged. (score: 0)
- (b) The soda can is compressed. (score: 1)

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

- (3-a) Entity - Dynamic Attribute. We test whether models

can modify attributes (e.g., color, size, texture) mid-video using the video-based multi-question answering pipeline. We take an example of questions here: “Initially, is the color of leaves mostly red?”; “Finally, is the color of leaves mostly green?”; “Does the color of leaves change?”

- (3-b) Entity - Dynamic Spatial Relationship. We assess

whether models accurately reposition objects in response to spatial instructions (e.g., “A dog is on the left of a sofa, then the dog runs to the front of the sofa.”) using video-based multi-question answering. The question template is similar to Dynamic Attribute.

- (3-c) Event - Motion Order Understanding. We evaluate

whether models generate several actions or motions in the specified order. We use the text description alignment pipeline to measure whether the generated motion sequence matches

models faithfully follow long-form landscape descriptions (150+ words) that include multiple scene transitions driven by

Question: What will the soda can be as air being gradually and forcefully removed.

Question: Does the video contain one or more of the following anomalies: sudden appearance, disappearance, fusion, fission??

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

- (a) The soda can remains unchanged. (score: 0)
- (b) The soda can is compressed. (score: 1)

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

(a) yes, it has sudden appearance (score: 0)

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

(b) no (score: 1)

Fig. 7: Visualization for Mechanics.

Fig. 8: Visualization for Instance Preservation.

camera movements. We assess adherence using text description alignment with landscape-specific system prompt, similar to Complex Plot.

ated videos properly follow material properties.

- (4-d) Geometry - Multi-View Consistency. Geometry is a

critical aspect for 3D/4D video generation, ensuring that generated entities and scenes maintain structural consistency when viewed from different angles or as the camera moves. However, we do not have access to explicit 3D ground truth of the generated videos, making direct 3D validation infeasible. Instead, we assess multi-view consistency using two complementary metrics: 1) Feature Matching Stability – Measures how well objects retain their geometric consistency across frames, inspired by [85], and 2) Camera Motion Speed – Accounts for the effect of motion strengths on feature stability, ensuring fair cross-model comparisons. Specifically, we follow [85] to extract frame-level keypoint features using SIFT [86], efficiently match points across frames with FLANN [87], and eliminate incorrect matches using RANSAC [88]. We then estimate camera motion speed using RAFT [89], and adjust feature-matching frame intervals based on the motion strength and the video frame rate ( i.e., FPS and camera motion speed will influence the score of feature matching).

5) Commonsense: We assess commonsense reasoning in video generation models across two key aspects: 1) Motion Rationality, evaluating whether generated motions are physically plausible and correctly executed, and 2) Instance Preservation, ensuring that abnormal entity states (entity sudden merging, splitting, appearing, and disappearing) do not exist.

- (5-a) Motion Rationality. We evaluate whether the generated

- (3-g) Camera Motion. We evaluate whether models

can generate specified camera movements. We extend VBench++ [26]’s camera motion taxonomy to nine types, adding “Orbit” and “Oblique shot, airborne dolly movement.” The generated camera motion is assessed via point tracking with CoTracker-v2 [84] and carefully tailored heuristics (e.g., different camera motions exhibit distinct behaviors at different points. For example, pan left results in all points shifting to the right.).

4) Physics: The Physics dimension evaluates whether video generation models adhere to real-world physical principles. We assess two key areas: 1) State Change, which examines how well models simulate mechanical, thermal, and material transformations, and 2) Geometry, which evaluates the 3D consistency of objects and scenes across different frames. We extend prior benchmarks (e.g., PhyGenBench [72]) by increasing physics scenario difficulty and significantly improving evaluation accuracy with tailored pipelines.

- (4-a) State Change - Mechanics. We evaluate whether

models follow basic mechanical principles such as gravity, buoyancy, and stress. This is done using video-based multiquestion answering. Unlike PhyGenBench, which relies on abstract physics concepts, we prompt GPT-4o to generate explicit visual descriptions of expected physical behavior (e.g., unlike PhyGenBench that uses terminologies to describe the physical phenomena “The liquid’s behavior aligns with the microgravity environment, floating freely, and forming natural blobs without noticeable distortion.”, we describe the physical phenomena’s visual results “The liquid floating, and forming blobs”)). Notably, the GPT-4o-generated descriptions are based solely on the text prompt, remain fixed during evaluation, and do not compromise evaluation reproducibility. To ensure a focused assessment of state changes, we also apply a pre-filtering step to exclude cases where the initial state of the generated video does not align with the prompt.

motion leads to the correct real-world consequences. A prevalent issue in video generation is the presence of fake motions that visually appear correct but lack expected effects on the environment. For instance: 1) Fake eating: A person bites into food, but the food remains unchanged. 2) Fake walking: A character moves their legs, but does not actually progress forward. and 3) Fake cutting: A knife moves through an object, but the object does not split. To address this, we adopt the video-based multi-question answering pipeline, specifically to ensure the motion’s impact is checked explicitly (e.g., “After slurping, is the amount of noodles visibly reduced?”). Redundant questioning (e.g., 1. “Does the person appear to be slurping noodles?”, 2. “Is the person’s mouth in contact with the noodles?”, 3. “Is the bowl of noodles remaining after slurping?”) helps filter out false positives and accidental misinterpretations.

- (4-b) State Change - Thermotics. We evaluate how well

the models simulate state transitions such as vaporization, liquefaction, and sublimation. To increase complexity, we introduce temperature-specific prompts (e.g., “A timelapse captures dry ice transforming at -90°C”). Evaluation follows the same video-based multi-question answering approach used in the Mechanics dimension.

- (4-c) State Change - Material. We evaluate whether the

(5-b) Instance Preservation. Video generative models frequently generate videos with unnatural entity merging, duplication, or disappearance, particularly during large-motion sequences or object interactions. As current VLMs could not capture this clip-level or even frame-level abnormal (

models correctly depict color mixing, hardness, combustion, and solubility. Similar to Mechanics and Thermotics, we use video-based multi-question answering, testing whether gener-

[Figure 99]

[Figure 100]

[Figure 101]

Fig. 10: Interface for Human Preference Annotation. Top: Question descriptions. Right: Choices available to annotators. Bottom left: Controls for stopping and playback.

Fig. 9: Overview of Prompt Suite Statistics. Left: distribution of words in the prompt suites. Right: number of prompts per evaluation dimension.

i.e., Unnatural merging can appear between adjacent frames, despite both frames being normal on their own). We tune a clip-level entity abnormal detector based on Qwen2.5-VL-3BInstruct [90]. Specifically, we use a multiple-choice question format to ask the VLM to determine whether a given clip contains any of the aforementioned anomalies (A for “yes” and B for “no”). Only the all of the clips in a video is normal will be considered as normal (score 1), otherwise 0.

prompts in Thermotics, require models to demonstrate a deeper understanding in how temperature affect thermotics beyond simple pattern matching.

Ensuring Disentangled Evaluation. Prompts are designed to eliminate confounding factors and ensure focused assessment in the dimension of interest. For example, in Dynamic Spatial Relationship and Dynamic Attribute, only one entity is allowed to move or changed, ensuring the test solely assesses positioning and attribute rather than taking the irrelevant ability of multi-object interactions into account.

In terms of training data, we first collected a set of videos that align with our prompt theme as normal samples. We also sampled several generative models (CogVideo, HunyuanVideo, Wan2.1, StepVideo), and manually annotated anomalous samples. To reduce the gap between virtual and real data, both virtual and real normal data is used for training and we further apply LoRA fine-tuning to maintain the model’s generalization ability as much as possible.

Evaluation Robustness. To improve the robustness of evaluation, we design the actions and events in each dimension’s prompts to be explicitly recognizable by VLMs and LLMs. In the Human Interaction dimension, we ensure that prompts include physical contact interactions (e.g., “A person shakes hands with another”) rather than ambiguous social scenarios that are harder to verify visually (e.g., “Two people are having a picnic”). For Motion Rationality, prompts are designed to ensure that the counterexamples involve visually observable outcomes, such as fake eating (food remains unchanged), fake walking (not moving forward), or fake cutting (objects remain unaltered), rather than examples like lip-syncing, which are difficult to conclusively assess through visual observation alone.

B. Prompt Suite

The VBench-2.0 Prompt Suite is designed to be compact yet representative. Given the increasing computational cost of video sampling, especially for longer and higher-resolution videos (e.g., HunyuanVideo and CogVideoX-1.5 taking over five minutes per sample on 8×A100 GPUs), we strategically limit the number of test cases to reduct sampling costs during evaluation, while ensuring coverage across diverse evaluation dimensions and content scenarios. Figure 9 visualizes the prompt distributions.

By following these structured design principles, VBench2.0 provides a compact, diverse, and reliable benchmark for evaluating video generation models across a diverse range of real-world and abstract scenarios.

Tailored Prompts for Each Dimension. For each evaluation dimension in VBench-2.0, we carefully construct a suite of approximately 70 prompts, specifically tailored to probe the model’s capabilities in that dimension. Prompts are designed to systematically analyze the core ability being tested. For example, in the Multi-View Consistency dimension, we evaluate both object-level and scene-level prompts, ensuring that models are tested across varying spatial structures. In the Composition dimension, we systematically divide prompts into species combination, single-entity action, and multi-entity tasks, covering different levels of creativity and compositional reasoning. The Physics dimension follows a structured approach, incorporating Mechanics (e.g., gravity, buoyancy, stress), Thermotics (e.g., vaporization, freezing), and Material properties (e.g., color mixing, solubility) to comprehensively assess adherence to physical laws. To further challenge model reasoning, additional constraints, such as temperature-specific

C. Human Preference Annotation

Following the approach of VBench [25], [26], we conduct large-scale human preference labeling on generated videos to validate the alignment between VBench-2.0’s evaluation and human perception across all evaluation dimensions. The collected human annotations also serve as a valuable resource for future research on fine-tuning generation and evaluation models to better reflect human judgments.

Data Preparation. There are two types of annotation formats in VBench-2.0.

The first type follows VBench and the human annotators are tasked to select a preferred video from two generated videos based on specific criteria. Given a text prompt pi and four video generation models {A,B,C,D}, we generate a set of videos, forming a “group” Gi,j =

TABLE II: VBench-2.0 Evaluation Results per Dimension. This table presents evaluation results for four recent state-ofthe-art video generation models across all 18 VBench-2.0 dimensions. A higher score indicates better performance in the corresponding dimension.

|Models|Anatomy<br><br>Human<br><br>|Clothes<br><br>Human<br><br>|Identity<br><br>Human|Composition|Diversity|Mechanics<br><br>|Material|Thermotics<br><br>|Consistency<br><br>Multi-view|
|---|---|---|---|---|---|---|---|---|---|
|HunyuanVideo [31] CogVideoX-1.5 [59] Sora [28] Kling 1.6 [29]|88.58% 59.72% 86.45% 86.99%<br><br>|82.97% 87.18% 98.15% 91.75%|75.67% 69.51% 78.57% 71.95%<br><br>|43.96%<br>44.70% 53.65% 43.89%<br>|39.73% 42.61% 67.48% 53.26%<br><br>|76.09% 80.80% 62.22% 65.55%|64.37% 83.19% 64.94% 68.00%<br><br>|56.52% 67.13% 43.36% 59.46%|43.80% 21.79% 58.22% 64.38%<br><br>|
|Models<br><br>|Relationship<br><br>Dynamic Spatial<br><br>|Attribute<br><br>Dynamic|Understanding<br><br>Motion Order<br><br>|Interaction<br><br>Human|Landscape<br><br>Complex|Plot<br><br>Complex|Motion<br><br>Camera<br><br>|Rationality<br><br>Motion<br><br>|Preservation<br><br>Instance|
|HunyuanVideo [31] CogVideoX-1.5 [59] Sora [28] Kling 1.6 [29]|21.26% 19.32%<br><br>19.81%<br><br>20.77%<br><br><br>|22.71% 24.18% 8.06% 19.41%|26.94% 26.60% 15.15% 29.29%<br><br>|66.67% 72.33% 58.00% 72.00%<br><br>|18.89% 20.00% 15.33% 17.33%|9.78%<br><br>11.21% 11.11%<br><br>10.83%<br><br><br>|33.95% 33.33% 27.16% 61.73%|34.48%<br><br>33.91%<br><br>34.48%<br><br><br>38.51%|92.40% 82.46% 94.15% 92.40%<br><br>|

{Vi,A,j,Vi,B,j,Vi,C,j,Vi,D,j}. For each prompt pi, we sample five such groups {Gi,0,Gi,1,Gi,2,Gi,3,Gi,4} and construct pairwise comparisons: (VA,VB), (VA,VC), (VA,VD), (VB,VC), (VB,VD), (VC,VD). Human annotators are asked to select their preferred video for each pair. To ensure unbiased annotations, the video order within each pair is randomized.

The second type might involve two groups of videos, and the dimension in evaluation is related to the content distribution in these two groups of videos. In this type, the pairwise construction is the same as the first type while each Vi contains 20 videos generated by single prompt pi.

Labeling Instructions. The annotation process follows VBench but incorporates refinements in interface design and evaluation methodology. Since VBench-2.0 introduces multiple dimensions with detailed multi-question evaluations and long text prompts (some exceeding 150 words), we enhance the interface for improved readability and efficiency. Instead of displaying extensive text in video titles, we sequentially list all key annotation instructions directly within the interface. This structured layout allows annotators to efficiently reference the necessary details while conducting evaluations.

Win Ratio. Given human annotations, we calculate the win ratio for each model. During pairwise comparisons, if a model’s video is preferred, it scores 1, while the other model scores 0. In case of a tie, both models score 0.5. The win ratio for each model is computed as the total score divided by the number of pairwise comparisons. We show the result in Table IV.

Quality Assurance. The annotation process maintains the rigorous quality control measures established in VBench while further refining the review criteria. We randomly sample 20% of the annotated pairs for verification, with a required success rate of 95%.

To quantify the annotation effort across all 18 evaluation dimensions, we account for the cumulative time spent on documentation drafting, initial trials, formal annotation, and re-annotation. In total, the process required 15.75 hours of individual effort and 284 hours across 18 annotators, reflecting the scale of the task and our commitment to ensuring highquality human preference annotations. Most evaluation dimensions go through 2 rounds of trial labeling, official labeling, and post-labeling verification.

IV. EXPERIMENTS A. Video Generation Models in Evaluation

To assess our benchmark against recent advancements, we utilize four models for comparison, with more to be included as they become open-sourced. We show the basical information of the four models in Table III and introduce the detailed implementation and a unified prompt refiner to pre-process the text prompt as follows.

Prompt Refiner. As video generation models continue to improve their understanding of text and support longer inputs, many methods have begun adopting prompt rewriting techniques to refine text input. To ensure fair comparisons and higher quality video in this paper, we employ a modified version of the original Prompt Refiner from CogVideoX to uniformly process all input prompts. We tested the following four models and found that except for Sora, our Prompt Refiner consistently result in higher-quality videos with better alignment to the input text. We hypothesize that Sora may have an embedded Prompt Refiner, which could explain its performance. Therefore, we applied our Prompt Refiner to all dimensions of the other three models. Kling 1.6 [29]. We adopt the standard API version of Kling 1.6. For each prompt, we sample 241 continuous frames of size 720×1280 at 24 frames per second (FPS). For the classifierfree guidance (cfg) scale, we used the official default value of 0.5.

Sora-480p [28]. Sora represents one of the earliest largescale video generation models to emerge. For each prompt, we directly perform sampling on the official website, containing 150 continuous frames of size 480×854 at 30 FPS ( i.e., , the low-resolution version of Sora).

HunyuanVideo [31]. HunyuanVideo is a powerful opensourced video generation model. It is equipped with a pretrained Multimodal Large Language Model (MLLM) [91] to better understand text prompts, supporting a large maximum token length and incorporating a dual-branch diffusion transformer generative architecture, which produces visually convincing results. We use the best version of HunyuanVideo with no classifier-free guidance. For each prompt, we sample 129 continuous frames of size 720×1280 at 24 FPS. All of the hyper-parameters are followed the default value in the official inference code. The initial random seed is set to 42 during sampling.

CogVideoX-1.5 [59]. CogVideoX-1.5 is a transformer-based video generation model, which is equipped with a 3D causal

Human Anatomy

Human Identity

Human Clothes

Dynamic Spatial Relationship

Dynamic Attribute

Motion Order Understanding

Human Interaction

Diversity

Composition

1.0

1.0

1.0

1.0

1.0

1.0

1.0

1.0

1.0

= 0.9546

= 0.9946

= 0.9089

= 0.9453

= 0.8170

= 0.9736

= 0.9095

= 0.9931

= 0.8928

0.5

0.5

0.5

0.5

0.5

0.5

0.5

0.5

0.5

VBench-2.0

0.0

0.0

0.0

0.0

0.0

0.0

0.0

0.0

0.0

0.5 1.0

0.5 1.0

0.5 1.0

0.5 1.0

0.5 1.0

0.5 1.0

0.5 1.0

0.5 1.0

0.5 1.0

Complex Landscape

Complex Plot

Camera Motion

Motion Rationality

Instance Preservation

Multi-View Consistency

Mechanics

Themotics

Material

1.0

1.0

1.0

1.0

1.0

1.0

1.0

1.0

1.0

= 0.9182

= 0.8959

= 0.9795

= 0.8797

= 0.9911

= 0.9356

= 0.9386

= 0.9371

= 0.9844

0.5

0.5

0.5

0.5

0.5

0.5

0.5

0.5

0.5

0.0

0.0

0.0

0.0

0.0

0.0

0.0

0.0

0.0

0.5 1.0

0.5 1.0

0.5 1.0

0.5 1.0

0.5 1.0

0.5 1.0

0.5 1.0

0.5 1.0

0.5 1.0

Human

HunyuanVideo win ratio

CogVideoX-1.5 win ratio Sora-480p win ratio Kling 1.6 win ratio Fitting

| | | |
|---|---|---|
| | | |

Fig. 11: Human Alignment of VBench-2.0 Evaluation. Each plot represents the alignment verification for a specific VBench2.0 dimension. In each plot, a dot corresponds to the human preference win ratio (horizontal axis) and the VBench-2.0 evaluation win ratio (vertical axis) for a given video generation model. A linear fit is applied to visualize the correlation, and Spearman’s correlation coefficient (ρ) is computed for each dimension. Experiments show that VBench-2.0 evaluations closely align with human judgement in all dimensions.

###### TABLE III: Information on Evaluated Models.

|Model Name<br><br>|Video Length Per-Frame Resolution Frame Rate (FPS)|
|---|---|
|HunyuanVideo [31] CogVideoX-1.5 [59] Sora-480p [28] Kling 1.6 [29]|5.3s 720×1280 24 10.1s 768×1360 16 5.0s 480×854 30 10.0s 720×1280 24<br><br>|

VAE to compress the spatial and temporal dimensions and an expert transformer generative architecture to achieve better text-video alignment. We use the best version of it (5B) to evaluate. For each prompt, we sample 161 continuous frames of size 768×1360 at 16 FPS. The initial random seed is also set to 42 for a fair comparison.

- B. VBench-2.0 Evaluation Results

For each sub-ability dimension, videos were generated using the models based on the corresponding prompt suite described in Section III-B. The evaluation method introduced in Section III-A is then applied to obtain numerical scores between 0 and 1, where a higher value indicate relatively stronger performance in that dimension. The evaluation results of the four video generative models are summarized in Table II and visualized in Figure 2. Additionally, for each of the five major dimensions, we visualize the evaluation results of one model pair to demonstrate the accuracy of our method in Figure 6 (Complex Landscape in Controllability), Figure 5 (Diversity in Creativity), Figure 4 (Human Anatomy in Human Fidelity), Figure 7 (Mechanics in Physics) and Figure 8 (Instance Preservation in Commonsense).

- C. Human Alignment of VBench-2.0

To ensure that VBench-2.0’s evaluation aligns closely with human judgment across all evaluation dimensions, we conducted human preference labeling on a large set of generated videos, following the approach of VBench [25], [26]. Specifically, we computed the correlation between our evaluation results and human annotations. Figure 11 and Table IV presents the correlation plot and numerical win ratios respectively, illustrating the alignment between human judgment and VBench2.0 evaluations in terms of model-level win ratios across video generation models in each dimension.

V. INSIGHTS AND DISCUSSIONS

In this section, we present key insights from VBench-2.0’s evaluation, highlighting trade-offs, model characteristics, and in-depth discussion on superficial versus intrinsic faithfulness in video generation.

A. Characteristics of Recent SOTA Models

From Figure 2, we can observe the relative strengths and weaknesses of each model under evaluation.

Sora-480p [28]. Sora clearly excels in Human Fidelity and Creativity dimensions compared to other SOTA models. It demonstrates a strong ability to generate human figures with reasonable anatomical consistency throughout a video while also showing improvisational skill in producing novel and imaginative content. This makes Sora a potential tool for human-centric filmmaking and artistic exploration. However, it falls short in Controllability, Physics and Commonsense dimensions, indicating that the generated videos may not align well with user-provided text prompts and sometimes violate real-world principles.

Kling 1.6 [29]. Kling demonstrates relative strengths in the Commonsense, Controllability and camera-related (Multi-View Consistency, Camera Motion) dimensions. These capabilities make Kling well-suited not only for tasks that require precise camera control, but also for broader applications involving coherent, accurate, and user-guided visual storytelling or simulation. Additionally, Kling does not show significantly weaker performance in any particular area, suggesting that its training data is broad and well-rounded, making it a valuable reference for future model development.

CogVideoX-1.5 [59]. This model is relatively strong in most dimensions related to complex prompt adherence (e.g., Complex Landscape and Complex Plot) and Physics, but shows notably poorer results in human-centric dimensions such as Human Fidelity and Motion Rationality. These outcomes suggest that CogVideoX’s training data contains limited highquality human-related content.

HunyuanVideo [31]. Although HunyuanVideo is relatively weaker in many VBench-2.0 dimensions, it demonstrates

TABLE IV: Human Alignment of VBench-2.0 Evaluation Methods. For each evaluation dimension and each video generative model, we report “VBench-2.0 Win Ratios (left) / Human Win Ratios (right)”. The results demonstrate that our evaluation metrics closely align with human perception across all dimensions.

|Models|Anatomy<br><br>Human<br><br>|Clothes<br><br>Human|Identity<br><br>Human<br><br>|Composition|Diversity|Mechanics|Material<br><br>|Thermotics|Consistency<br><br>Multi-View|
|---|---|---|---|---|---|---|---|---|---|
|HunyuanVideo [31] CogVideoX-1.5 [59] Sora [28] Kling 1.6 [29]|67.73% / 58.73% 13.10% / 5.28% 60.05% / 69.71% 59.12% / 66.36%<br><br>|44.49% / 45.67% 48.12% / 44.28% 55.61% / 57.15% 51.18% / 52.86%|52.60% / 58.19%<br><br>36.81% / 20.62%<br><br>53.88% / 56.25%<br><br><br>56.48% / 63.05%<br><br>|46.02% / 40.25% 50.00% / 46.23% 56.08% / 59.96%<br>47.90% / 53.56%<br>|16.67% / 36.67% 26.67% / 28.33% 93.33% / 83.33% 63.33% / 51.67%<br><br>|53.30% / 50.64% 57.57% / 51.20% 43.47% / 49.39% 45.56% / 48.78%|45.70% / 45.49%<br><br>59.43% / 57.49% 47.38% / 50.79%<br><br>46.65% / 45.90%<br>|50.13% / 46.08% 56.69% / 58.51% 40.37% / 41.19% 52.74% / 54.02%<br><br>|52.83% / 50.39% 8.77% / 18.07% 64.10% / 63.94% 80.00% / 65.98%|
|Correlation<br><br>|95.46%|90.89%|99.46%<br><br>|81.70%<br><br>|94.53%|93.56%<br><br>|93.71%<br><br>|93.86%|98.44%|

|Models|Relationship<br><br>Dynamic Spatial|Attribute<br><br>Dynamic<br><br>|Understanding<br><br>Motion Order<br><br>|Interaction<br><br>Human|Landscape<br><br>Complex<br><br>|Plot<br><br>Complex<br><br>|Motion<br><br>Camera|Rationality<br><br>Motion<br><br>|Preservation<br><br>Instance|
|---|---|---|---|---|---|---|---|---|---|
|HunyuanVideo [31] CogVideoX-1.5 [59] Sora [28] Kling 1.6 [29]<br><br>|50.64% / 51.37% 49.36% / 48.07%<br><br>49.68% / 49.60%<br>50.32% / 50.97%<br>|52.75% / 55.07%<br><br>53.72% / 54.70%<br><br><br>42.98% / 43.16% 50.55% / 47.07%<br><br>|51.63% / 56.35% 51.40% / 52.13% 43.77% / 30.81% 53.20% / 60.73%|49.61% / 52.45% 53.39% / 51.34% 43.83% / 43.49% 53.17% / 52.73%<br><br>|52.41% / 60.04%<br><br>53.89% / 54.46% 42.04% / 35.87% 51.67% / 49.63%<br><br><br>|46.85% / 30.46% 52.31% / 65.37% 51.30% / 49.35% 49.54% / 54.81%|46.60% / 49.28% 46.19% / 44.08% 42.08% / 42.64% 65.12% / 63.99%<br><br>|49.43% / 52.68% 49.04% / 42.24% 49.43% / 44.83% 52.11% / 60.25%<br><br>|51.36% / 50.88% 44.74% / 43.27%<br>52.53% / 53.80% 51.36% / 52.05%<br>|
|Correlation<br><br>|97.36%|90.95%<br><br>|99.31%<br><br>|89.28%|91.82%|89.59%|97.95%<br><br>|87.97%|99.11%|

impressive strengths in human-related aspects (Human Fidelity and Motion Rationality). This suggests that HunyuanVideo likely benefits from training data rich in high-quality, humanrelated content.

- B. Key Limitations of Recent SOTA Models

Key Challenge: Generating Complex Plots. In the Complex Plot dimension, state-of-the-art video generation models struggle to follow detailed text descriptions involving multiple scenes, character interactions, and logical story progression. A major limitation is that current foundation video generative models typically produce single-shot videos under 10 seconds in length, insufficient for conveying coherent narratives. This highlights an important direction for future research: building models capable of functioning more like true filmmakers.

Surprising Weakness: Controllability in Simple Dynamics. Most models perform poorly in capturing Dynamic Spatial Relationships and Dynamic Attributes. Even in relatively simple cases, where an entity’s position, relationship, or attribute (e.g., color) is instructed to change, and models fail in about 80% of the time. These shortcomings, despite the simplicity of the underlying semantics, are likely due to inadequate captioning granularity in video generation datasets. Existing video captioning pipelines may not be intentionally describing how object attributes or positions evolve over time, weakening models’ understanding of these dynamics. Enhancing these pipelines with more focused, temporally grounded instructions during video captioning could help address this gap.

- C. The Role of Prompt Engineering

In modern video generation, the Prompt Refiner rewrites or augments input text prompts to enhance video generation quality. For implementation details, please refer to the Sec IV-A. Below, we highlight several notable and potentially surprising observations.

Controllability vs. Creativity. Our evaluation results reveal a trade-off between Creativity and Controllability. Sora performs well in creative tasks but struggles with controllability, while the other models show the reverse pattern. This suggests that models emphasizing creativity and diversity are better at flexibly imagining novel content, but may be less capable of strictly and accurately following control signals. Alternatively, the Prompt Refiner used with the other three models may improve controllability by fine-graining the text at the expense

of diversity. Sora’s internal prompt optimization appears to take a different approach. Going forward, prompt refinement strategies should consider not only precision but also the preservation of diversity, which is essential for creating openended content and simulating the distributions of the real world.

Prompting Partially Compensates for Physical Reasoning Gaps. Physics is often challenging in video generation. However, with the exception of Sora-480p, models guided by the external Prompt Refiner demonstrated reasonably strong performance in the Physics dimension. This suggests that even without an inherent understanding of physical laws, models can be steered toward physically plausible outcomes through carefully designed prompts. Therefore, the core difficulty may lie less in physical reasoning itself, and more in achieving precise video-text alignment.

Limited Impact on Knowledge-Driven Dimensions. For dimensions that rely on model’s intrinsic visual understanding and prior knowledge, like Human Fidelity, Camera Motion, Geometry, and Commonsense, we observe no consistent performance trend from models using different Prompt Refiners. These aspects likely extend beyond the scope of logical inference or direct text-to-video mapping, limiting the Prompt Refiner’s influence. This suggests that success in these areas may depend less on prompt engineering and more on underlying data quality and model architecture.

D. Superficial Faithfulness vs. Intrinsic Faithfulness: Do Not Miss Out on Any Pillar

Superficial Faithfulness (e.g., cinematographic quality) often shapes the first impression viewers get from a video. As a result, models that produce aesthetically pleasing and smooth outputs are frequently perceived as “better”. However, this perception could be misleading. In practice, Intrinsic Faithfulness, which includes elements like storytelling, logical progression, and world simulation, is equally important for determining a model’s potential for real-world applications. For example, as shown in Figure 2 and Table II, CogVideoX performs relatively well across many VBench-2.0 dimensions, though its visual Quality Score in VBench suggests room for improvement compared to models like Sora and HunyuanVideo. Conversely, HunyuanVideo produces visually impressive results, though its performance in many structure-driven dimensions in VBench2.0 suggests opportunities for further growth in those areas.

These observations highlight a common bias toward relying primarily on visual quality when judging video generation models. To address this, we encourage the community to use both VBench and VBench-2.0 together, enabling a more comprehensive and in-depth evaluation across both Superficial Faithfulness and Intrinsic Faithfulness.

E. Challenge, Current Solution and Future Work of Video Generation Evaluation

Challenge. Since Intrinsic Faithfulness is more difficult to accurately evaluate compared to Superficial Faithfulness ( i.e., evaluation models need to have capabilities that include complex scene understanding, commonsense reasoning, and physical world perception), the demand for using large models such as VLMs and LLMs has gradually increased. However, in practical applications, dimensions such as Human Anatomy and Motion Rationality are areas where current large models cannot serve as reliable evaluators. The main reasons can be summarized into two points: 1) Training data gap: Real-world video data used for training VLMs does not contain anomaly or unnatural entity states that are commonly present in generated videos; 2) Capability limitations: VLMs perform poorly in aspects such as object counting and 3D understanding.

Current Solution. In this paper, to address the above two challenges, we construct the corresponding solutions: 1) For different tasks and scenarios, we build anomaly specialists to collect abnormal data regarding human body, motion, entity splitting and merging. This data is then mixed with real datasets for training, achieving high accuracy while maintaining generalizability. 2) Currently, there are two mainstream approaches: fine-tuning-based and non-fine-tuning methods. Fine-tuning-based methods can easily overfit to the generated training data (due to limited data size), achieving high test accuracy. However, since most generated videos are less than 10 seconds in length, models tuned in this way struggle to generalize to longer videos in the future, and the overall generalization ability of the model is also a major concern. Due to the above drawbacks, VBench-2.0 adopts a non-finetuning approach. We break down a large problem into multiple visual phenomena that can be understood by VLMs, and design redundant questions to improve the accuracy of VLMs. For dimensions that require high-level understanding ability, we further integrate LLM into the evaluation.

Future Work. We believe that some dimensions will inevitably require the understanding capabilities of large models to be resolved in the future. However, other dimensions (particularly those involving content that should not exist in real data) indeed need to be handled by specialized expert models, as outlined in the two points mentioned above. Here, following the above classification, we outline the key directions that future next-generation video generation evaluation methods should explore: 1) Although the types of anomalies judged by different dimensions vary (Human Anatomy and Instance Preservation evaluate anomalies in human motion, appearance anomalies and entity state anomalies, respectively), there is an overlap between these anomaly types. For example, during the entity splitting process, limb anomalies will also occur.

Therefore, integrating abnormal data to train a unified cliplevel anomaly detection model can help the model capture both the differences and commonalities across various anomalies, thereby enabling the development of a more generalized and unified anomaly detection large model. 2) Accuracy and sustainability are two important pillar off a good benchmark. Fine-tuning-based methods cannot meet the sustainability requirement under current circumstances (unless transform into a clip-level formulation that is independent of video length and construct a virtual database with the same scale as the VLM training data). Therefore, what needs to be done in the future is to more thoroughly evaluate existing models’ understanding of real-world physical laws and 3D scenes (e.g., VLM benchmarks for 3D space reasoning [92], cinematography understanding [93] and physics understanding [94], [95]). When a truly capable and qualified intelligent large model emerges in the future, it will mark the beginning of the next generation of video generation benchmarks.

VI. CONCLUSION

While recent video generative models have achieved superficial faithfulness, true progress requires advancing towards intrinsic faithfulness, ensuring adherence to physical laws, commonsense reasoning, and structured coherence. To address this, we introduced VBench-2.0, a benchmark assessing models on five key dimensions beyond superficial faithfulness. To accurately evaluate these more challenging dimensions, we fully explore the capabilities of state-of-the-art vision-language models (VLMs) and large language models (LLMs). We build generalists in dimensions where they excel, and employ various specialists for assessment in areas where they are less proficient. VBench-2.0 complements VBench by expanding evaluation to deeper aspects of video generation, providing a multi-dimensional and human-aligned evaluation framework. We believe that VBench-2.0 is an important contribution to the video generation community, shaping the field into its next era.

Future Work. We will continually add more video generative models to VBench-2.0 when they become available.

Potential Negative Societal Impacts. Although VBench2.0 does not directly generate videos, the evaluation process inevitably involves working with generated content. As video generative models grow more powerful and capable of producing increasingly realistic scenes, we emphasize the importance of safety and ethical considerations when using these models. We encourage responsible use to mitigate potential risks associated with AI-generated media.

ACKNOWLEDGMENTS

This study is supported by the Ministry of Education, Singapore, under its MOE AcRF Tier 2 (MOE-T2EP20221-0012, MOE-T2EP20223-0002), and under the RIE2020 Industry Alignment Fund – Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s).

REFERENCES

- [1] J. H. Liew, H. Yan, J. Zhang, Z. Xu, and J. Feng, “Magicedit: High-fidelity and temporally coherent video editing,” arXiv preprint arXiv:2308.14749, 2023.
- [2] W. Chai, X. Guo, G. Wang, and Y. Lu, “Stablevideo: Textdriven consistency-aware diffusion video editing,” arXiv preprint arXiv:2308.09592, 2023.
- [3] M. Geyer, O. Bar-Tal, S. Bagon, and T. Dekel, “Tokenflow: Consistent diffusion features for consistent video editing,” arXiv preprint arxiv:2307.10373, 2023.
- [4] J. Huang, L. Sigal, K. M. Yi, O. Wang, and J.-Y. Lee, “Inve: Interactive neural video editing,” arXiv preprint arXiv:2307.07663, 2023.
- [5] P. Couairon, C. Rambour, J.-E. Haugeard, and N. Thome, “Videdit: Zero-shot and spatially aware text-driven video editing,” arXiv preprint arXiv:2306.08707, 2023.
- [6] S. Liu, Y. Zhang, W. Li, Z. Lin, and J. Jia, “Video-p2p: Video editing with cross-attention control,” arXiv preprint arXiv:2303.04761, 2023.
- [7] Z. Zhang, B. Li, X. Nie, C. Han, T. Guo, and L. Liu, “Towards consistent video editing with text-to-image diffusion models,” arXiv preprint arXiv:2305.17431, 2023.
- [8] M. Zhao, R. Wang, F. Bao, C. Li, and J. Zhu, “Controlvideo: Adding conditional control for one shot text-to-video editing,” arXiv preprint arXiv:2305.17098, 2023.
- [9] W. Wang, k. Xie, Z. Liu, H. Chen, Y. Cao, X. Wang, and C. Shen, “Zero-shot video editing using off-the-shelf image diffusion models,” arXiv preprint arXiv:2303.17599, 2023.
- [10] D. Ceylan, C.-H. P. Huang, and N. J. Mitra, “Pix2video: Video editing using image diffusion,” in ICCV, 2023.
- [11] C. Qi, X. Cun, Y. Zhang, C. Lei, X. Wang, Y. Shan, and Q. Chen, “Fatezero: Fusing attentions for zero-shot text-based video editing,” arXiv preprint arXiv:2303.09535, 2023.
- [12] Y.-C. Lee, J.-Z. G. J. Jang, Y.-T. Chen, E. Qiu, and J.-B. Huang, “Shape-aware text-driven layered video editing demo,” arXiv preprint arXiv:2301.13173, 2023.
- [13] Y. Zhao, E. Xie, L. Hong, Z. Li, and G. H. Lee, “Make-a-protagonist: Generic video editing with an ensemble of experts,” arXiv preprint arXiv:2305.08850, 2023.
- [14] X. Yang, L. Zhu, H. Fan, and Y. Yang, “Videograin: Modulating spacetime attention for multi-grained video editing,” in ICLR, 2025.
- [15] N. Kumari, B. Zhang, R. Zhang, E. Shechtman, and J.-Y. Zhu, “Multi-concept customization of text-to-image diffusion,” arXiv preprint arXiv:2212.04488, 2022.
- [16] J. Karras, A. Holynski, T.-C. Wang, and I. Kemelmacher-Shlizerman, “Dreampose: Fashion image-to-video synthesis via stable diffusion,” arXiv preprint arXiv:2304.06025, 2023.
- [17] Y. He, M. Xia, H. Chen, X. Cun, Y. Gong, J. Xing, Y. Zhang, X. Wang, C. Weng, Y. Shan et al., “Animate-a-story: Storytelling with retrievalaugmented video generation,” arXiv preprint arXiv:2307.06940, 2023.
- [18] Y. Guo, C. Yang, A. Rao, Y. Wang, Y. Qiao, D. Lin, and B. Dai, “Animatediff: Animate your personalized text-to-image diffusion models without specific tuning,” in ICLR, 2024.
- [19] J. Xing, M. Xia, Y. Zhang, H. Chen, X. Wang, T.-T. Wong, and Y. Shan, “Dynamicrafter: Animating open-domain images with video diffusion priors,” arXiv preprint arXiv:2310.12190, 2023.
- [20] N. Agarwal, A. Ali, M. Bala, Y. Balaji, E. Barker, T. Cai, P. Chattopadhyay, Y. Chen, Y. Cui, Y. Ding et al., “Cosmos world foundation model platform for physical ai,” arXiv preprint arXiv:2501.03575, 2025.
- [21] Y. Wang, X. Chen, X. Ma, S. Zhou, Z. Huang, Y. Wang, C. Yang, Y. He, J. Yu, P. Yang et al., “Lavie: High-quality video generation with cascaded latent diffusion models,” arXiv preprint arXiv:2309.15103, 2023.
- [22] J. Wang, H. Yuan, D. Chen, Y. Zhang, X. Wang, and S. Zhang, “Modelscope text-to-video technical report,” arXiv preprint arXiv:2308.06571, 2023.
- [23] H. Chen, M. Xia, Y. He, Y. Zhang, X. Cun, S. Yang, J. Xing, Y. Liu, Q. Chen, X. Wang, C. Weng, and Y. Shan, “Videocrafter1: Open diffusion models for high-quality video generation,” arXiv preprint arXiv:2310.19512, 2023.
- [24] W. Hong, M. Ding, W. Zheng, X. Liu, and J. Tang, “CogVideo: Largescale pretraining for text-to-video generation via transformers,” arXiv preprint arXiv:2205.15868, 2022.
- [25] Z. Huang, Y. He, J. Yu, F. Zhang, C. Si, Y. Jiang, Y. Zhang, T. Wu, Q. Jin, N. Chanpaisit et al., “Vbench: Comprehensive benchmark suite for video generative models,” in CVPR, 2024.

- [26] Z. Huang, F. Zhang, X. Xu, Y. He, J. Yu, Z. Dong, Q. Ma, N. Chanpaisit, C. Si, Y. Jiang, Y. Wang, X. Chen, Y.-C. Chen, L. Wang, D. Lin, Y. Qiao, and Z. Liu, “Vbench++: Comprehensive and versatile benchmark suite for video generative models,” arXiv preprint arXiv:2411.13503, 2024.
- [27] Y. Liu, X. Cun, X. Liu, X. Wang, Y. Zhang, H. Chen, Y. Liu, T. Zeng, R. Chan, and Y. Shan, “Evalcrafter: Benchmarking and evaluating large video generation models,” in CVPR, 2024.
- [28] OpenAI, “Sora,” Accessed February 15, 2024 [Online] https: //sora.com/library, 2024. [Online]. Available: https://sora.com/library
- [29] K. Team, “Kling,” Accessed December 9, 2024 [Online] https://klingai. kuaishou.com/, 2024. [Online]. Available: https://klingai.kuaishou.com/
- [30] runway, “Gen-3,” Accessed June 17, 2024 [Online] https://runwayml. com/research/introducing-gen-3-alpha, 2024. [Online]. Available: https: //runwayml.com/research/introducing-gen-3-alpha
- [31] T. Team, “Hunyuanvideo: A systematic framework for large video generative models,” 2024.
- [32] G. Team, “Veo2,” Accessed December 18, 2024 [Online] https: //deepmind.google/technologies/veo/veo-2/, 2025. [Online]. Available: https://deepmind.google/technologies/veo/veo-2/
- [33] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, “Deep unsupervised learning using nonequilibrium thermodynamics,” in ICML, 2015.
- [34] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-based generative modeling through stochastic differential equations,” in ICLR, 2021.
- [35] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in NeurIPS, 2020.
- [36] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” in ICLR, 2021.
- [37] L. Zhang and M. Agrawala, “Adding conditional control to text-to-image diffusion models,” arXiv preprint arXiv:2302.05543, 2023.
- [38] A. Blattmann, T. Dockhorn, S. Kulal, D. Mendelevitch, M. Kilian, D. Lorenz, Y. Levi, Z. English, V. Voleti, A. Letts et al., “Stable video diffusion: Scaling latent video diffusion models to large datasets,” arXiv preprint arXiv:2311.15127, 2023.
- [39] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. M¨uller, H. Saini, Y. Levi, D. Lorenz, A. Sauer, F. Boesel et al., “Scaling rectified flow transformers for high-resolution image synthesis,” in ICML, 2024.
- [40] C. Mou, X. Wang, L. Xie, J. Zhang, Z. Qi, Y. Shan, and X. Qie, “T2iadapter: Learning adapters to dig out more controllable ability for textto-image diffusion models,” arXiv preprint arXiv:2302.08453, 2023.
- [41] Z. Huang, K. C. Chan, Y. Jiang, and Z. Liu, “Collaborative diffusion for multi-modal face generation and editing,” in CVPR, 2023.
- [42] M. Ding, Z. Yang, W. Hong, W. Zheng, C. Zhou, D. Yin, J. Lin, X. Zou, Z. Shao, H. Yang et al., “CogView: Mastering text-to-image generation via transformers,” in NeurIPS, 2021.
- [43] M. Ding, W. Zheng, W. Hong, and J. Tang, “Cogview2: Faster and better text-to-image generation via hierarchical transformers,” in NeurIPS, 2022.
- [44] J. Ho, W. Chan, C. Saharia, J. Whang, R. Gao, A. Gritsenko, D. P. Kingma, B. Poole, M. Norouzi, D. J. Fleet et al., “Imagen video: High definition video generation with diffusion models,” arXiv preprint arXiv:2210.02303, 2022.
- [45] D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” arXiv preprint arXiv:1312.6114, 2013.
- [46] A. Van Den Oord, O. Vinyals et al., “Neural discrete representation learning,” in NeurIPS, 2017.
- [47] P. Esser, R. Rombach, and B. Ommer, “Taming transformers for highresolution image synthesis,” in CVPR, 2021.
- [48] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. M¨uller, J. Penna, and R. Rombach, “Sdxl: Improving latent diffusion models for high-resolution image synthesis,” arXiv preprint arXiv:2307.01952, 2023.
- [49] L. Yu, Y. Cheng, K. Sohn, J. Lezama, H. Zhang, H. Chang, A. G. Hauptmann, M.-H. Yang, Y. Hao, I. Essa et al., “Magvit: Masked generative video transformer,” in CVPR, 2023.
- [50] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.
- [51] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” arXiv preprint arXiv:2212.09748, 2022.
- [52] Z. Luo, D. Chen, Y. Zhang, Y. Huang, L. Wang, Y. Shen, D. Zhao, J. Zhou, and T. Tan, “Videofusion: Decomposed diffusion models for high-quality video generation,” in CVPR, 2023.

- [53] Y. He, T. Yang, Y. Zhang, Y. Shan, and Q. Chen, “Latent video diffusion models for high-fidelity video generation with arbitrary lengths,” arXiv preprint arXiv:2211.13221, 2022.
- [54] D. Zhou, W. Wang, H. Yan, W. Lv, Y. Zhu, and J. Feng, “Magicvideo: Efficient video generation with latent diffusion models,” arXiv preprint arXiv:2211.11018, 2023.
- [55] D. J. Zhang, J. Z. Wu, J.-W. Liu, R. Zhao, L. Ran, Y. Gu, D. Gao, and M. Z. Shou, “Show-1: Marrying pixel and latent diffusion models for text-to-video generation,” arXiv preprint arXiv:2309.15818, 2023.
- [56] S. Ge, S. Nah, G. Liu, T. Poon, A. Tao, B. Catanzaro, D. Jacobs, J.B. Huang, M.-Y. Liu, and Y. Balaji, “Preserve your own correlation: A noise prior for video diffusion models,” in ICCV, 2023.
- [57] A. Blattmann, R. Rombach, H. Ling, T. Dockhorn, S. W. Kim, S. Fidler, and K. Kreis, “Align your latents: High-resolution video synthesis with latent diffusion models,” in CVPR, 2023.
- [58] L. Khachatryan, A. Movsisyan, V. Tadevosyan, R. Henschel, Z. Wang, S. Navasardyan, and H. Shi, “Text2video-zero: Text-to-image diffusion models are zero-shot video generators,” arXiv preprint arXiv:2303.13439, 2023.
- [59] Z. Yang, J. Teng, W. Zheng, M. Ding, S. Huang, J. Xu, Y. Yang, W. Hong, X. Zhang, G. Feng et al., “Cogvideox: Text-to-video diffusion models with an expert transformer,” arXiv preprint arXiv:2408.06072, 2024.
- [60] A. Polyak, A. Zohar, A. Brown, A. Tjandra, A. Sinha, A. Lee, A. Vyas, B. Shi, C.-Y. Ma, C.-Y. Chuang, D. Yan, D. Choudhary, D. Wang, G. Sethi, G. Pang, H. Ma, I. Misra, J. Hou, J. Wang, K. Jagadeesh, K. Li, L. Zhang, M. Singh, M. Williamson, M. Le, M. Yu, M. K. Singh, P. Zhang, P. Vajda, Q. Duval, R. Girdhar,

- R. Sumbaly, S. S. Rambhatla, S. Tsai, S. Azadi, S. Datta, S. Chen,
- S. Bell, S. Ramaswamy, S. Sheynin, S. Bhattacharya, S. Motwani,
- T. Xu, T. Li, T. Hou, W.-N. Hsu, X. Yin, X. Dai, Y. Taigman, Y. Luo, Y.-C. Liu, Y.-C. Wu, Y. Zhao, Y. Kirstain, Z. He, Z. He, A. Pumarola, A. Thabet, A. Sanakoyeu, A. Mallya, B. Guo, B. Araya, B. Kerr, C. Wood, C. Liu, C. Peng, D. Vengertsev, E. Schonfeld, E. Blanchard, F. Juefei-Xu, F. Nord, J. Liang, J. Hoffman, J. Kohler, K. Fire, K. Sivakumar, L. Chen, L. Yu, L. Gao, M. Georgopoulos, R. Moritz, S. K. Sampson, S. Li, S. Parmeggiani, S. Fine, T. Fowler, V. Petrovic, and Y. Du, “Movie gen: A cast of media foundation models,” 2025. [Online]. Available: https://arxiv.org/abs/2410.13720

- [61] W. Team, “Wan: Open and advanced large-scale video generative models,” 2025.
- [62] S. Team, 2025. [Online]. Available: https://arxiv.org/abs/2502.10248
- [63] M. Team, “Minmax,” Accessed August 31, 2024 [Online] https: //hailuoai.com/, 2023. [Online]. Available: https://hailuoai.com/
- [64] W. Fan, C. Si, J. Song, Z. Yang, Y. He, L. Zhuo, Z. Huang, Z. Dong, J. He, D. Pan et al., “Vchitect-2.0: Parallel transformer for scaling up video diffusion models,” arXiv preprint arXiv:2501.08453, 2025.
- [65] C. Si, W. Fan, Z. Lv, Z. Huang, Y. Qiao, and Z. Liu, “Repvideo: Rethinking cross-layer representation for video generation,” arXiv 2501.08994, 2025.
- [66] X. Peng, Z. Zheng, C. Shen, T. Young, X. Guo, B. Wang, H. Xu, H. Liu, M. Jiang, W. Li, Y. Wang, A. Ye, G. Ren, Q. Ma, W. Liang, X. Lian,

- X. Wu, Y. Zhong, Z. Li, C. Gong, G. Lei, L. Cheng, L. Zhang, M. Li, R. Zhang, S. Hu, S. Huang, X. Wang, Y. Zhao, Y. Wang, Z. Wei, and
- Y. You, “Open-sora 2.0: Training a commercial-level video generation model in $200k,” arXiv preprint arXiv:2503.09642, 2025.

- [67] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “GANs trained by a two time-scale update rule converge to a local nash equilibrium,” in NeurIPS, 2017.
- [68] T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, X. Chen, and X. Chen, “Improved techniques for training gans,” in NeurIPS, 2016.
- [69] T. Unterthiner, S. van Steenkiste, K. Kurach, R. Marinier, M. Michalski, and S. Gelly, “FVD: A new metric for video generation,” in ICLRW, 2019.
- [70] Y. Liu, L. Li, S. Ren, R. Gao, S. Li, S. Chen, X. Sun, and L. Hou, “Fetv: A benchmark for fine-grained evaluation of open-domain text-to-video generation,” in NeurIPS, 2023.
- [71] F. Zhang, S. Tian, Z. Huang, Y. Qiao, and Z. Liu, “Evaluation agent: Efficient and promptable evaluation framework for visual generative models,” arXiv preprint arXiv:2412.09645, 2024.
- [72] F. Meng, J. Liao, X. Tan, W. Shao, Q. Lu, K. Zhang, Y. Cheng, D. Li, Y. Qiao, and P. Luo, “Towards world simulator: Crafting physical commonsense-based benchmark for video generation,” arXiv preprint arXiv:2410.05363, 2024.

- [73] K. Sun, K. Huang, X. Liu, Y. Wu, Z. Xu, Z. Li, and X. Liu, “T2vcompbench: A comprehensive benchmark for compositional text-tovideo generation,” arXiv preprint arXiv:2407.14505, 2024.
- [74] Y. Wang, X. He, K. Wang, L. Ma, J. Yang, S. Wang, S. S. Du, and Y. Shen, “Is your world simulator a good story presenter? a consecutive events-based benchmark for future long video generation,” arXiv preprint arXiv:2412.16211, 2024.
- [75] Y. Zhang, J. Wu, W. Li, B. Li, Z. Ma, Z. Liu, and C. Li, “Video instruction tuning with synthetic data,” arXiv preprint arXiv:2410.02713, 2024.
- [76] A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei et al., “Qwen2. 5 technical report,” arXiv preprint arXiv:2412.15115, 2024.
- [77] Z. Xie, Z. Zhang, Y. Cao, Y. Lin, J. Bao, Z. Yao, Q. Dai, and H. Hu, “Simmim: A simple framework for masked image modeling,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 9653–9663.
- [78] T. Cheng, L. Song, Y. Ge, W. Liu, X. Wang, and Y. Shan, “Yoloworld: Real-time open-vocabulary object detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 16901–16911.
- [79] G. Fang, W. Yan, Y. Guo, J. Han, Z. Jiang, H. Xu, S. Liao, and X. Liang, “Humanrefiner: Benchmarking abnormal human generation and refining with coarse-to-fine pose-reversible guidance,” in European Conference on Computer Vision. Springer, 2024, pp. 201–217.
- [80] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, “Arcface: Additive angular margin loss for deep face recognition,” in CVPR, 2019.
- [81] J. Deng, J. Guo, E. Ververas, I. Kotsia, and S. Zafeiriou, “Retinaface: Single-shot multi-level face localisation in the wild,” in CVPR, 2020.
- [82] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in ICLR, 2015.
- [83] L. A. Gatys, A. S. Ecker, and M. Bethge, “A neural algorithm of artistic style,” arXiv preprint arXiv:1508.06576, 2015.
- [84] N. Karaev, I. Rocco, B. Graham, N. Neverova, A. Vedaldi, and C. Rupprecht, “Cotracker: It is better to track together,” in ECCV, 2024.
- [85] X. Li, D. Zhou, C. Zhang, S. Wei, Q. Hou, and M.-M. Cheng, “Sora generates videos with stunning geometrical consistency,” arXiv preprint arXiv:2402.17403, 2024.
- [86] G. Lowe, “Sift-the scale invariant feature transform,” Int. J, 2004.
- [87] M. Muja and D. G. Lowe, “Fast approximate nearest neighbors with automatic algorithm configuration.” VISAPP (1), 2009.
- [88] M. A. Fischler and R. C. Bolles, “Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography,” Communications of the ACM, 1981.
- [89] Z. Teed and J. Deng, “Raft: Recurrent all-pairs field transforms for optical flow,” in ECCV, 2020.
- [90] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang et al., “Qwen2. 5-vl technical report,” arXiv preprint arXiv:2502.13923, 2025.
- [91] X. Sun, Y. Chen, Y. Huang, R. Xie, J. Zhu, K. Zhang, S. Li, Z. Yang, J. Han, X. Shu et al., “Hunyuan-large: An open-source moe model with 52 billion activated parameters by tencent,” arXiv preprint arXiv:2411.02265, 2024.
- [92] J. Yang, S. Yang, A. W. Gupta, R. Han, L. Fei-Fei, and S. Xie, “Thinking in space: How multimodal large language models see, remember, and recall spaces,” in CVPR, 2025.
- [93] H. Liu, J. He, Y. Jin, D. Zheng, Y. Dong, F. Zhang, Z. Huang, Y. He, Y. Li, W. Chen et al., “Shotbench: Expert-level cinematic understanding in vision-language models,” arXiv preprint arXiv:2506.21356, 2025.
- [94] V. Balazadeh, M. Ataei, H. Cheong, A. Hosein Khasahmadi, and R. G. Krishnan, “Synthetic vision: Training vision-language models to understand physics,” arXiv e-prints, 2024.
- [95] H. Al-Tahan, Q. Garrido, R. Balestriero, D. Bouchacourt, C. Hazirbas, and M. Ibrahim, “Unibench: Visual reasoning requires rethinking visionlanguage beyond scaling,” in NeruIPS, 2024.

[Figure 102]

Dian Zheng is an incoming Ph.D. student at MMLab@CUHK, Chinese University of Hong Kong (CUHK), supervised by Prof. Hongsheng Li. He received his Master’s degree in 2025 and his Bachelor’s degree in 2022 from Sun Yat-sen University and Dalian University of Technology respectively. His current research interests focus on visual generation and evaluation.

[Figure 103]

Fan Zhang is currently a research assistant at the Shanghai Artificial Intelligence Laboratory, supervised by Prof. Ziwei Liu. He received his bachelor’s degree from the University of Electronic Science and Technology of China (UESTC). His research interests include generative models and computer vision.

[Figure 104]

Ziqi Huang is currently a Ph.D. student at MMLab@NTU, Nanyang Technological University (NTU), supervised by Prof. Ziwei Liu. She received her Bachelor’s degree from NTU in 2022. Her current research interests include visual generation and evaluation. She is awarded Google PhD Fellowship 2023, and is a recipient of the 2025 Apple Scholars in AI/ML PhD Fellowship.

[Figure 105]

Lulu Gu is a research intern at the Shanghai Artificial Intelligence Laboratory. She is currently pursuing her master’s degree at the School of Electronic Information and Electrical Engineering, Shanghai Jiao Tong University. Her research interests lie in deep learning and generative modeling.

[Figure 106]

Hongbo Liu is currently pursuing his Ph.D. degree at Tongji University, under the supervision of Prof. Shengjie Zhao. He received his Bachelor’s degree from Tongji University in 2023. His research interests lie in multimodal understanding and generation.

Yuanhan Zhang is currently a Ph.D. student at MMLab@NTU, Nanyang Technological University, supervised by Prof. Ziwei Liu. His interests lie in computer vision and deep learning. In particular, He is focused on adapting foundation models—from vision to multi-modal—for real-world exploration. He has published several papers in ICCV, ECCV, CVPR, NeurIPS and IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI). He also served as a reviewer for CVPR, ICCV, ECCV, NeurIPS, ICML, ICLR, IEEE Transactions on Pat-

[Figure 107]

tern Analysis and Machine Intelligence (TPAMI), and International Journal of Computer Vision (IJCV).

[Figure 108]

[Figure 109]

Kai Zou is currently a Master’s student at the University of Science and Technology of China, under the supervision of Professor Bin Liu. He obtained his Bachelor’s degree from Shanghai University in 2023. His current research interests lie in multimodal generation and understanding.

[Figure 110]

Jingwen He is currently a Ph.D. student at Chinese University of Hong Kong, supervised by Prof. Wanli Ouyang. She received her M.Phil. degree in electronic and information engineering from the University of Sydney, Australia, in 2019, supervised by Prof. Dong Xu. Her research interests include image generation, video generation, and vision language models.

Yinan He is currently a Research Engineer at Shanghai AI Laboratory, where he is a member of the OpenGVLab. He received his Master’s degree from Beijing University of Posts and Telecommunications. His current research interests include video understanding and multi-modality large language models.

Wei-Shi Zheng is now a full Professor with Sun Yat-sen University. His research interests include person association and activity understanding, AI robotics learning, and the related weakly supervised/unsupervised and continuous learning machine learning algorithms. He has now published more than 200 papers, including more than 150 publications in main journals (TPAMI, IJCV, SIGGRAPH, TIP) and top conferences (ICCV, CVPR, ECCV, NeurIPS). He has ever served as area chairs of ICCV, CVPR, ECCV, BMVC, NeurIPS and etc.

[Figure 111]

He is associate editors/on the editorial board of IEEE-TPAMI, Artificial Intelligence Journal, Pattern Recognition. He has ever joined Microsoft Research Asia Young Faculty Visiting Programme. He is a Cheung Kong Scholar Distinguished Professor, a recipient of the NSFC Excellent Young Scientists Fund, a Fellow of IAPR and a recipient of the Royal SocietyNewton Advanced Fellowship of the United Kingdom.

Yu Qiao (Senior Member, IEEE) is a professor and Leading Scientist at the Shanghai AI Laboratory, previously the Director of the Institute of Advanced Computing and Digital Engineering at the Shenzhen Institutes of Advanced Technology, Chinese Academy of Science. He has published more than 300 research papers with more than 100,000 citations. His team won the AAAI 2021 Best Paper, the CVPR 2023 Best Paper, and the ACL 2024 Distinguished Paper. He received the Young Scholar Award of Wang Xuan Award, the First Prize of

[Figure 112]

the Guangdong Technological Invention Award, and the Jiaxi Lv Young Researcher Award from the Chinese Academy of Sciences. His research interests include deep learning on computer vision, video understanding and generation, and multimodal large models.

Ziwei Liu is currently an Associate Professor at Nanyang Technological University, Singapore. His research revolves around computer vision, machine learning and computer graphics. He has published extensively on top-tier conferences and journals in relevant fields, including CVPR, ICCV, ECCV, NeurIPS, ICLR, SIGGRAPH, TPAMI, TOG and Nature - Machine Intelligence. He is the recipient of PAMI Mark Everingham Prize, CVPR Best Paper Award Candidate, Asian Young Scientist Fellowship, International Congress of Basic Science Frontiers of

[Figure 113]

Science Award and MIT Technology Review Innovators under 35 Asia Pacific. He serves as an Area Chair of CVPR, ICCV, ECCV, NeurIPS and ICLR, as well as an Associate Editor of IJCV.

