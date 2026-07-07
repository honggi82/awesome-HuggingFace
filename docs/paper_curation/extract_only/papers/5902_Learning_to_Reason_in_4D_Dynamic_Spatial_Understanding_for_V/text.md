# arXiv:2512.20557v1[cs.CV]23Dec2025

## Learning to Reason in 4D: Dynamic Spatial Understanding for Vision Language Models

Shengchao Zhou1, Yuxin Chen2, Yuying Ge2, Wei Huang1, Jiehong Lin1, Ying Shan2 and Xiaojuan Qi1† 1The University of Hong Kong 2ARC Lab, Tencent PCG

{zhoushengchao2024, weih}@connect.hku.hk, {uasonchen, yingsshan}@tencent.com {yyge13, mortimer.jh.lin}@gmail.com, xjqi@eee.hku.hk

https://github.com/TencentARC/DSR Suite

(a)

Ours

Abs Spd 73.8% Abs Ori 84.1%

(c)

Abs Spd Comp

VG-LLM

|Which object is closer to the chair?<br><br>Cabinet✓<br><br>[Figure 1]<br><br>… …<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>| | |
|---|---|
| | |
| | |
<br><br>Static Static<br><br>Static Static<br><br>Static Spatial Reasoning<br><br>VLM|
|---|

72.0%

[Figure 5]

Gemini-2.5-Pro

Abs Dir Pred

Qwen2.5-VL-7B

35.5%

Abs Dir 73.8%

Rel Dis

75.8%

Abs Dis

87.0%

|How does distance between truck and bus change?<br><br>[Figure 6]<br><br>… …<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>| | | |
|---|---|---|
| | | |
<br><br>| | |
|---|---|
<br><br>| | |
|---|---|
| | |
<br><br>Forward Forward<br><br>Forward Forward<br><br>Larger <br><br>Dynamic Spatial Reasoning<br><br>VLM<br><br>|
|---|

Rel Dir 76.1%

Non-Temp Based

46.4%

Rel Ori 77.7%

Rel Dir Pred Rel Spd Comp 35.1%

Rel Spd 60.0%

37.1%

(b)

[Figure 10]

Automatically Generated

|[Figure 11]<br><br>… …<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>… …<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>… …<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>(1) Sports & Recreation<br>(2) Transportation & Vehicle Operation<br>(3) Art Performance<br><br><br>[Figure 23]<br><br>… …<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>… …<br><br>[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]<br><br>… …<br><br>(4) Manual Labor & Craftsmanship<br>(5) Daily Activities & Hobbies<br>(6) Nature & Wildlife<br><br><br>In-the-Wild Video Scene|
|---|

|Question: Between 3s and 16s, following perspective of boy within initial bounding box [221,80,246,262], how does direction of dog within initial bounding box [371,195, 386, 243] to dog within initial bounding box [323,187,340,236] change?<br><br>[Figure 35]<br><br>[Figure 36]<br><br>| |
|---|
<br><br>| | | |
|---|---|---|
<br><br>[Figure 37]<br><br>| |
|---|
<br><br>[Figure 38]<br><br>| | |
|---|---|
| | |
| | |
<br><br>… …<br><br>Forward Forward & Left<br><br>Forward & Left Forward Forward & Left<br><br>Question-Answer Pairs<br><br>Viewpoint Transform; Multi-Object Interact; Fine-Grained Answer<br><br>[Figure 39]<br><br>Answer: (1) Left; (2) Behind|Left|
|---|

Figure 1. Overview of DSR Suite. (a) Comparison between static and dynamic spatial reasoning: Unlike static scenarios, dynamic spatial reasoning (DSR) requires understanding environments with moving objects, posing greater reasoning challenges. (b) Data examples from DSR Suite: Built upon automated pipeline and in-the-wild videos, DSR Suite generates scalable question-answer pairs that feature viewpoint transform, multi-object interact and fine-grained answers for comprehensive training and evaluation of DSR. (c) Benchmark comparison: Evaluations on our proposed DSR-Bench highlight the capability of our model trained on constructed DSR-Train.

###### Abstract

Vision-language models (VLM) excel at general understanding yet remain weak at dynamic spatial reasoning (DSR), i.e., reasoning about the evolvement of object geometry and relationship in 3D space over time, largely due

†Corresponding author.

to the scarcity of scalable 4D-aware training resources. To bridge this gap across aspects of dataset, benchmark and model, we introduce DSR Suite. First, we propose an automated pipeline that generates multiple-choice questionanswer pairs from in-the-wild videos for DSR. By leveraging modern vision foundation models, the pipeline extracts rich geometric and motion information, including camera

poses, local point clouds, object masks, orientations, and 3D trajectories. These geometric cues enable the construction of DSR-Train for learning and further humanrefined DSR-Bench for evaluation. Compared with previous works, our data emphasize (i) in-the-wild video sources, (ii) object- and scene-level 3D requirements, (iii) viewpoint transformations, (iv) multi-object interactions, and (v) finegrained, procedural answers. Beyond data, we propose a lightweight Geometry Selection Module (GSM) to seamlessly integrate geometric priors into VLMs, which condenses question semantics and extracts question-relevant knowledge from pretrained 4D reconstruction priors into a compact set of geometry tokens. This targeted extraction avoids overwhelming the model with irrelevant knowledge. Experiments show that integrating DSR-Train and GSM into Qwen2.5-VL-7B significantly enhances its dynamic spatial reasoning capability, while maintaining accuracy on general video understanding benchmarks.

###### 1. Introduction

Recent advances in vision-language models (VLMs) have led to remarkable progress in video understanding, driven by large-scale multimodal pretraining and alignment between visual and textual representations [6, 17, 19, 34, 40]. Despite these achievements, VLMs remain limited to perform 3D spatial reasoning in dynamic environments [31], which requires understanding how the object geometry and relationship evolve in 3D space over time, i.e., dynamic spatial reasoning (DSR). This capability is fundamental for interactive AI systems in robotics, autonomy, AR/VR and embodied intelligence, where the environments are dynamic and spatial relationships constantly evolve.

Recent works have started to explore 3D spatial reasoning within VLMs [4, 7, 46]. However, as in Figure 1(a), most efforts remain constrained to static scenes with immobile objects [3, 36, 41] or short-horizon motion [14, 24, 35, 37], providing limited insight into 4D intelligence. Only a handful of studies consider dynamic scenes [18, 45], yet they suffer from narrow scene diversity, limited reasoning types and lack of training data. Beyond data, current methods typically inject geometric knowledge to VLMs through direct cross-attention or na¨ıve fusion with a large set of pretrained reconstructive priors [10, 43, 44], compromising general-purpose performance by overfitting to task-specific cues. Consequently, a scalable pipeline, training corpus, comprehensive benchmark and effective model for systematically investigating DSR is in demand.

To this end, we present DSR Suite, a framework designed to endow VLMs with DSR ability. At its core, DSR Suite introduces an automated data-generation pipeline that constructs training and evaluation datasets. The pipeline curates in-the-wild videos and leverages modern vision foun-

dation models to extract camera poses, local point clouds, object masks, orientations and 3D trajectories. As in Figure 1(b), with these annotations, we derive: (1) DSR-Train, a large-scale multiple-choice corpus for DSR training; and (2) DSR-Bench, a further human-refined benchmark that evaluates object- and scene-level 3D understanding, multiobject interactions, viewpoint transformations, and finegrained temporal reasoning in dynamic scenes. Together, these resources bridge a critical gap for studying geometric and temporal reasoning in realistic, dynamic settings.

Beyond data, we further propose a simple yet effective mechanism to integrate geometric priors from pretrained

- 3D foundation models into VLMs. While training on DSRTrain through na¨ıvely adding a large set of features from 3D reconstruction models to vision tokens can improve DSR performance, it will lead to noticeable degradation on general benchmarks (see Table 2 in Section 5.3). To mitigate this trade-off, we propose a lightweight Geometry Selection Module (GSM) that performs text-guided geometric knowledge selection. Concretely, GSM adopts a dual QFormer design: the first condenses question semantics, and the second extracts only question-relevant knowledge from
- 4D reconstruction priors into a compact set of geometry tokens. This targeted filtering suppresses irrelevant geometric noise, enabling precise 4D grounding without compromising general multimodal understanding performance.

Our contributions can be summarized as follows:

- • We build an automated pipeline that transforms in-thewild videos into multiple-choice question-answer pairs for dynamic spatial reasoning by extracting camera poses, local point clouds, object masks, orientations and 3D trajectories, yielding DSR-Train for model training and DSR-Bench for comprehensive evaluation.
- • We propose GSM, a lightweight module that integrates geometric priors into VLMs via two stacked Q-Formers: the first condenses question semantics, and the second extracts question-relevant knowledge from 4D reconstruction priors into a compact set of geometry tokens, mitigating effect of irrelevant noise for general understanding.
- • Experiments show that after training on DSR-Train, Qwen2.5-VL-7B with GSM achieves state-of-the-art performance on DSR-Bench, while maintaining competitive results across general multimodal benchmarks.

###### 2. Related Work

Spatial Reasoning Data Spatial reasoning aims to endow VLMs with the ability to reason in 3D space, thereby extending their applicability to more downstream tasks such as robotic navigation. To foster this capability, a growing number of training datasets and benchmarks have recently been proposed. Among those using multi-image or video input, training data in [41] and benchmark in [36] present early attempts that demand aggregation of spatial informa-

tion across frames. Despite these advances, most existing works are restricted to static scenes, where objects are motionless. In contrast, real-world environments are inherently dynamic, with objects moving and changing positions.

Several studies attempt to incorporate object motion into their benchmark designs. In [24, 35, 37], VLMs are provided with two images to infer object changes. However, this short temporal horizon limits the evaluation of longterm reasoning ability. Subsequent efforts [16, 18, 45] extend input modality to videos, yet these datasets are confined primarily to autonomous driving or human–object interaction scenarios. Moreover, they often suffer from limited question diversity, weak 3D understanding requirements, and coarse-grained answer formulations, hindering comprehensive evaluation of DSR. Beyond these benchmarks, there is still a lack of datasets for task training, leaving current models without sufficient supervision.

In contrast, leveraging our automated data generation pipeline, we construct both a training dataset, DSR-Train, and an evaluation benchmark, DSR-Bench, that overcome these limitations. Based on in-the-wild videos, diverse question types, randomized target object and viewpoint, and fine-grained answers, these resources enables comprehensive and fine-grained training and evaluation of dynamic spatial reasoning in realistic settings. A detailed comparison with existing benchmarks is presented in Section 3.4.

Spatial Reasoning Models To enhance spatial reasoning capabilities, several methods are proposed to integrate 3D information into VLMs. In [46], the authors encode 3D coordinates of each pixel as position embeddings to provide geometric context for downstream tasks. In [10, 43], knowledge extracted from 3D foundation models, such as CUT3R [29] and VGGT [27], are directly fused with vision tokens. While these approaches introduce explicit geometric priors, the injected features are often task-specific and may lead to performance degradation on benchmarks unrelated to spatial reasoning. In contrast, our GSM selectively extracts question-relevant 3D knowledge into a compact set of tokens, thereby reducing interference with general multimodal understanding while strengthening DSR ability.

###### 3. Dynamic Spatial Reasoning Data

To study DSR at scale, our DSR Suite first builds a fully automated pipeline that constructs both a large-scale training set (DSR-Train) and a further human-refined evaluation benchmark (DSR-Bench) from in-the-wild videos. The pipeline consists of three stages as illustrated in Figure 2:

(1) Video Curation–selecting videos with meaningful object motion; (2) Geometric Clue Extraction–extracting both object- and scene-level 3D cues with vision foundation models such as Grounded SAM2 [25], Orient Anything [33], π3 [32]; (3) Data Generation–constructing diverse

multiple-choice questions and fine-grained answers. Note that for DSR-Bench, all question-answer pairs are further refined by human annotators to ensure accuracy. Together, this pipeline yields scalable, high-quality data for training and evaluation of dynamic spatial reasoning.

###### 3.1. Video Curation

We begin with Koala-36M [28], a diverse in-the-wild video corpus originally curated for video generation. It is a largescale, general-purpose video dataset encompassing a wide range of scenes. Moreover, it has been preprocessed to remove low-quality videos and each video is accompanied by a caption describing objects and overall event.

However, many videos in Koala-36M are unsuitable for DSR, as they exhibit no or minimal object motion–such as articulated movements–without significant object position change. To ensure meaningful spatial content, we filter out clips with negligible global or object motion as in Stage 1 of Figure 2: (i) For DSR-Train, we use DeepSeek-R1 [12] to filter candidates based on captions; (ii) For DSR-Bench, we adopt Gemini-2.5-Pro [8], prompting directly on video content for more reliable selection. We retain videos between 20s–120s to balance temporal context and computational cost. After filtering, we obtain 10,000 videos for training and a curated subset of 575 videos for evaluation. This step ensures scene diversity, real-world dynamics, and sufficient temporal span for multi-object spatial understanding.

###### 3.2. Geometric Clue Extraction

If without awareness to absolute scale, vision foundation models cannot produce reliable metric-scale 3D structure. Worse still, annotating ground-truth scale is labor-intensive and often inconsistent, resulting in inaccurate questionanswer pairs (QA). Therefore, with the curated videos in hand, we primarily process the monocular footage with vision foundation models that yield relative (non-metric) 3D structure to enable generation of qualitative, trend-based QAs (e.g., larger/smaller, left/right, faster/slower) for faithful supervision and evaluation. For DSR-Train, we uniformly sample 32 frames per video to control computation; for DSR-Bench, we sample at 1 FPS to maximize answer fidelity. Because objects may enter or exit, we randomly select a time sub-interval per QA instance and keep only the frames within it, then apply vision foundation models to extract the geometric cues necessary for QA construction.

As shown in Stage 2 of Figure 2, at the scene level we estimate camera poses and local point clouds using π3 [32], which provides robust relative-scale reconstructions in dynamic, in-the-wild settings. At the object level, we use DeepSeek-R1 (caption-guided) to extract agent & nonagent categories and Grounded SAM2 [25] for tracking and segmentation, yielding temporally consistent masks. Each mask is lifted to 3D by projecting it onto the correspond-

###### Stage3: Data Generation

###### Stage1: Video Curation

Q: Between 2s and 4s, from camera’s viewpoint,

Random Target

Template-Based Question-Answer

[Figure 40]

###### In-the-Wild Videos

how does direction of camera to the car within

| | |
|---|---|
| | |

& Template

bounding box (432, 180, 461, 192) change? A: (1) Behind B: (1) Behind|Left C: (1) Left. (2) Behind D: (1) Above Q: Between 2s and 4s, from camera’s viewpoint, which car decreases distance to camera faster?

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

… …

- Caption A
- Caption B
- Caption C

…

[Figure 45]

LLM Non-Template-Based

[Figure 46]

[Figure 47]

[Figure 48]

… …

[Figure 49]

- A: Car within bounding box (501, 174, 568, 189)
- B: Car within bounding box (432, 180, 461, 192)
- C: Neither of them D: Both of them

Question-Answer

Random Viewpoint & Coord Transform

…

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

… …

Stage2: Geometric Clue Extraction

|b<br><br>[Figure 54]|
|---|

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Caption-based

… …

[Figure 65]

Video-based

… …

Motion Filter

Motion Filter

+

…

…

+

+

[Figure 66]

[Figure 67]

LLM VLM

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

… …

… …

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

Camera Poses & Local Point Clouds

Object Masks Agent Orientations

| | |
|---|---|
|𝝅𝟑| |

Valid Videos

Grounded SAM2 Orient Anything

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

Caption A … …

| | |
|---|---|
| | |

…

[Figure 83]

Sample & Sub-interval

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

… …

… …

Caption C

- Figure 2. Multiple-choice question–answer generation pipeline in our DSR Suite. It comprises three stages: Video Curation, Geometric Clue Extraction and Data Generation. In Video Curation stage, in-the-wild videos are filtered by LLMs or VLMs to remove motionless ones based on captions or visual cues. During Geometric Clue Extraction, vision foundation models extract key geometric cues, including camera poses, point clouds, object masks and orientations. Finally, in Data Generation, object coordinates are transformed into a randomly selected viewpoint and question–answer pairs are produced using either predefined templates or LLM-based free-form generation.

Table 1. Type, target object and description of candidate questions.

Type Target Object Description Distance

Determining how the distance between target objects changes over time Direction

Two random objects (agents or non-agents)

Identifying how the direction of one object with respect to another changes Orientation

Two random objects (agents or non-agents)

Describing how the orientation of an agent evolves Speed

One random object (agent only)

Assessing how the speed of an object varies Speed Comparison

One random object (agent or non-agent)

Two random objects (agents or non-agents)

Comparing the speed of two target objects throughout the video

Direction Prediction

One random object (agent or non-agent)

Predicting the moving direction of a target object at the end of video

Table 2. Basic answer choices for each type of question.

Type Basic Choices

- (1) Keep nearly constant then become larger;
- (2)Keep nearly constant then become smaller;
- (3) Keep nearly constant; (4) Become larger;

Distance & Speed

(5) Become smaller Direction & Orientation & Direction Prediction

Combinations of (1) Front/Behind; (2) Left/Right; (3) Above/Below

Speed Comparison

(1) Nearly the same; (2) The former is faster; (3) The latter is fater

ing point cloud from the π3 model [32]. The mean of the associated points serves as the object’s 3D center at each timestamp, producing smooth 3D trajectories across the sub-interval. For objects of agent classes, we further estimate orientations (azimuth, elevation, roll w.r.t. the camera) with Orient Anything [33] while orientations for nonagents are omitted to avoid spurious estimates. We prune objects that are not continuously visible to ensure reliability. The resulting geometric cues, including camera poses,

per-timestamp geometry, multi-object trajectories, 3D centers and agent orientations, form a compact, expressive basis for constructing questions and answers for DSR task.

###### 3.3. Data Generation

Types of data. With all geometric cues in place, we synthesize multiple-choice QA pairs in two families: templatebased items that probe core dynamic-spatial skills and nontemplate (free-form) items that assess holistic reasoning.

Sports & Recreation

Rel Dis

Abs Dis

[Figure 91]

7%

18%

11%

6%

- 21%
- 22%

Rel Dir

Abs Dir

Transportation & Vehicle Operation

6%

14%

Abs Ori

Rel Ori Rel Spd

Video

Art Performance

8%

Question

- 8%
- 9%

Abs Spd

Scenes

6%

Manual Labor & Craftsmanship

Types

11%

Abs Spd Comp

Rel Spd Comp

6%

Daily Activities & Hobbies

21%

10% 4% 6%

Abs Dir Pred

Rel Dir Pred

6%

Nature & Wildlife

Non-Temp Based

(a) Video scenes distribution (b) Question types distribution (c) Word cloud of questions

- Figure 3. Statistical overview of our DSR-Bench. In (a), we present the proportion of videos corresponding to different scene classes. In (b), we show the proportion of questions of various types. In (c), we depict a word cloud of the questions.

Each template QA is defined over a time sub-interval, a question type, a set of target objects, and an observing viewpoint. We instantiate six types of template QA whose target objects and descriptions are listed in Table 1.

ing: it grounds every question in 4D evidence, supports both egocentric and allocentric querying, and provides finegrained procedural answers that evaluate a model’s understanding of continuous object motion, spatial change, and multi-object interactions. All prompts, templates and generation details are described in the Appendix.

Viewpoints. Spatial intelligence is inherently tied to the observer’s viewpoint, which defines the operative coordinate system. To encourage reasoning across different situations, we vary the viewpoint and its mobility: (i) viewpoint can be posed from the camera or from an agent; (ii) viewpoint can follow the motion of its corresponding agent and evolve across time sub-interval (relative viewpoint) or fixed to its status at one specific timestamp (absolute viewpoint). We transform all object 3D centers into the selected reference coordinate using camera poses and agent orientations, enabling questions that require egocentric-allocentric transformation reasoning rather than passive observation.

###### 3.4. Benchmark Statistics and Comparisons

We first report the comprehensive statistics of DSR-Bench. Figure 3(a), (b) summarizes the distributions of videos and questions belonging to different categories. Following Gemini-2.5-Pro’s advice, videos span six broad categories: Sports & Recreation; Transportation & Vehicle Operation; Art Performance; Manual Labor & Craftsmanship; Daily Activities & Hobbies; Nature & Wildlife. There are totally 1484 questions organized into 12 template-based types (the Cartesian product of viewpoint mobility and question type) plus one non-template class. The results show that both the video and question distributions are balanced, supporting broad generalization. We also provide the word cloud of questions in Figure 3(c) to show our evaluation emphasis on the complete state evolvement procedure, reasoning from different viewpoints and multi-object interactions.

Question-answer pair generation. When generating one template-based QA, we randomly select one question type, required target objects according to Table 1 and one viewpoint as well as its mobility. Together with time sub-interval selected before, we construct the question according to corresponding template. For multiple choices, since monocular 3D reconstruction produces relative-scale geometry and humans also struggle to consistently judge metric values in unconstrained videos, our choices are qualitative rather than numeric and are combinations of basic choices in Table 2. We generate the correct answer by comparing the queried attribute every 2 adjacent frames to derive one basic choice and then merging consecutive identical states to form a concise, procedural answer that reflects the state evolvement over time, rather than a single frame snapshot.

Second, we contrast DSR-Bench with prior spatial reasoning benchmarks containing object position changes in Table 3. Part of benchmarks only consider reasoning the difference between two images, prevented from long-term evaluation. For benchmarks with video inputs, while most of them draw from domain-constrained sources (e.g., autonomous driving [2, 26] or human-object interaction [21]), DSR-Bench is built from in-the-wild videos, enabling evaluation in diverse and unconstrained environments. Except for DynSuperCLEVR [31], earlier efforts largely emphasize single-object motion from the camera’s viewpoint, limiting assessment of multi-object and multi-viewpoint reasoning.

While template-based QAs mainly evaluate fundamental spatial-temporal abilities, they may not fully reflect how humans naturally query motion and geometry. To broaden linguistic variety and reasoning patterns, we additionally employ DeepSeek-R1 to auto-generate non-template-based questions and answers conditioned on extracted 3D trajectories, object identities and viewpoints, aiming to probe more general, free-form understanding of scene dynamics.

Furthermore, we quantify 3D knowledge requirements in order to resolve a task. We conduct two complementary checks: (i) object-level, which uses DeepSeek-R1 to judge whether a question does not require 3D attributes such as orientation, shape, or size (a benchmark is marked as requiring object-level knowledge if less than half of

Together, this hybrid QA generation strategy yields a data suite that rigorously targets dynamic spatial reason-

Table 3. Comparison between our DSR-Bench and prior benchmarks from different aspects.

Answer Temporal Granularity MMSI-Bench [37]

Target Object Number

Viewpoint Transformation

3D Awareness Requirement

Benchmark Input Scene

In-the-Wild Multi-Object ✓ SpatialScore [35] In-the-Wild Single-Object × - -

Two Images

SAT [24] In-the-Wild Single-Object × DynSuperCLEVR [31]

Driving Multi-Object × Medium Coarse

VLM4D [45] In-the-Wild Single-Object × Weak Coarse STI-Bench [18] Driving Single-Object × Medium Coarse

Video

OmniSpatial [16] Human-Object Interaction Single-Object × Weak Coarse DSR-Bench In-the-Wild Multi-Object ✓ Strong Fine

responses are positive); and (ii) scene-level, which uses DeepSeek-R1 to determine whether a question can be answered from 2D changes alone (if fewer than half respond “yes” scene-level knowledge is deemed necessary). We then label overall 3D demand as weak, medium, or strong depending on whether none, one, or both knowledge types are required. Our results show that proportion of positive answers about two checks for different benchmarks are: (1) DynSuperCLEVR–63% & 24%; (2) VLM4D–73% & 85%; (3) STI-Bench–69% & 21%; (4) OmniSpatial–56% & 79%; (5) DSR-Bench–34% & 18%. Therefore, their 3D demand levels are medium, weak, medium, weak and strong.

Finally, while most benchmarks provide coarse, aggregate answers, DSR-Bench supplies fine-grained, procedural answers that describe the change over time, compelling VLMs to reason about continuous dynamics rather than isolated outcomes. We also quantify this by prompting DeepSeek-R1 to judge whether an answer describes the changing procedure of the asked attribute across video. The proportion of positive answers for different benchmarks are: (1) DynSuperCLEVER–2%; (2) VLM4D–19%; (3) STIBench–22%; (4) OmniSpatial–18%; (5) DSR-Bench–78%. Therefore, aside from our DSR-Bench, all previous benchmarks are dominated by course-grained answers. Collectively, these properties make DSR-Bench a comprehensive benchmark for DSR evaluation.

###### 4. Geometric Prior Enhanced VLM for Dynamic Spatial Reasoning

A common strategy to enhance spatial reasoning is to inject features from pretrained 3D/4D foundation models into VLMs. Prior works typically either (i) apply cross-attention between vision tokens and 3D tokens (e.g., from CUT3R [29]), or (ii) directly add geometry features (e.g., from VGGT [27]) to vision tokens [10, 43]. While these approaches improve spatial reasoning performance, they tend to degrade general video understanding (e.g., Video-MME, see Table 5 in Section 5.3). This happens because geometric

###### LLM

+ +

Vision Tokens

Geometry Tokens

Text Tokens

Relevant-Geometry Selector

###### Semantic Condenser

3D Tokens

3D Foundation

Model

Text Tokens

Learnable

Queries

Video Tokenizer

Text Tokenizer

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

How does the distance between boy and black dog change?

… …

Figure 4. Illustraction of our proposed GSM that consists of two stacked Q-Formers. The first Q-Former condenses question semantics, and the second one extracts question-relevant geometric knowledge into a compact set of geometry tokens. These tokens are appended to original vision tokens to be processed by LLM.

models cam produce noisy geometric cues when processing in-the-wild videos, overwhelming the VLM and causing task-specific overfitting and reduced generality.

To address this trade-off, we propose a lightweight Geometry Selection Module (GSM) to selectively incorporate geometric priors into VLMs. Instead of exposing VLM to all 3D tokens, GSM retrieve a compact, task-relevant subset of geometric knowledge based on questions. Concretely, as in Figure 4, given a video, we compute (i) the VLM vision tokens Tvis, (ii) the question tokens Ttext, (iii) 3D tokens T3D obtained by applying π3 encoder [32] to video frames to form the geometric prior. GSM employs two stacked Q-

Table 4. Performance comparison among different VLMs on different subtasks of DSR-Bench.

|Models|Subtask Types|
|---|---|
| |Abs Dis<br><br>Abs Dir<br><br>Abs Ori<br><br>Abs Spd<br><br>Abs Spd Comp<br><br>Abs Dir Pred<br><br>Rel Dis<br><br>Rel Dir<br><br>Rel Ori<br><br>Rel Spd<br><br>Rel Spd Comp<br><br>Rel Dir Pred<br><br>Non-Temp Based<br><br>Avg|

Propriety Models

- GPT-4o [15] 18.8 29.2 26.8 29.7 21.5 26.2 24.1 23.8 17.2 32.3 22.8 24.4 34.7 26.4

- GPT-5 [22] 21.1 41.5 48.7 34.5 33.3 34.7 17.2 44.3 41.9 21.2 25.0 30.9 26.7 30.8

Gemini-2.5-Flash [8] 18.8 27.6 19.5 25.0 23.6 22.0 11.2 28.4 30.8 23.2 17.1 22.6 38.8 24.9 Gemini-2.5-Pro [8] 20.0 44.6 53.6 27.3 38.7 30.5 23.2 32.9 43.2 17.1 28.5 27.9 34.3 31.7

Video Understanding Models LLaVA-Video-7B [42] 22.3 16.9 25.6 33.3 45.1 24.5 24.1 15.9 17.2 19.1 24.2 21.4 33.9 25.9

VideoRefer [39] 23.5 18.4 25.6 33.5 45.4 27.1 25.0 16.1 18.5 20.2 26.4 22.6 34.7 26.9 LongVILA-R1 [5] 20.0 21.5 23.1 21.4 37.6 22.8 24.1 26.1 28.3 22.2 17.8 20.8 33.9 25.3

General-Purpose Models

Qwen2.5-VL-7B [1] 18.8 15.3 14.6 42.8 29.0 19.4 31.8 19.3 11.1 22.2 19.2 20.2 30.1 23.5 Qwen2.5-VL-32B [1] 31.7 21.5 23.1 44.0 36.5 25.4 27.5 23.8 37.0 27.2 29.2 21.4 36.2 29.9

Qwen3-VL-8B-Instruct [23] 23.5 24.6 42.6 29.7 27.9 33.8 18.1 28.4 34.5 24.2 22.1 27.9 33.5 28.7 Qwen3-VL-30B-A3B-Instruct [23] 25.8 27.6 46.3 30.9 31.1 34.7 20.6 29.5 37.0 28.2 24.2 31.5 35.4 31.1

InternVL3.5-8B [30] 23.5 27.6 28.0 34.5 24.7 27.9 22.4 17.0 19.7 28.2 30.0 14.2 30.1 25.4 InternVL3.5-38B [30] 25.8 27.8 29.2 34.2 24.7 28.5 26.7 16.3 23.4 29.2 32.1 15.4 31.3 26.7

Spatial Reasoning Models

VLM-3R [10] 28.2 27.6 31.7 42.8 38.7 33.0 34.4 23.8 30.8 22.2 26.4 29.1 35.0 31.4 VG-LLM [43] 55.2 32.3 58.5 57.1 51.6 32.2 56.0 36.3 32.0 30.3 32.1 29.1 27.9 38.4

Ours 87.0 73.8 84.1 73.8 72.0 35.5 75.8 76.1 77.7 60.6 37.1 35.1 46.4 58.9

Formers to produce a fixed-size set of N geometry tokens:

- 1. Language condensation. The first Q-Former (Semantic Condenser) takes N learnable queries and attends to Ttext, distilling the question semantics into a set of language-conditioned query embeddings Qlang ∈RN×d.
- 2. Selective geometry retrieval. The second Q-Former

(Relevant-Geometry Selector) attends Qlang to T3D, extracting only the geometry relevant to the question and yielding compact geometry tokens Qgeo∈RN×d.

Because N is fixed, GSM presents the language model with a bounded and task-aligned geometric summary, avoiding the brittleness of directly exposing long, variablelength T3D to the VLM. The extracted geometry tokens are concatenated with vision tokens and question tokens:

###### T˜total = [Tvis ; Qgeo ; Ttext ],

and the combined stream is passed to the LLM head. This late, compact fusion injects essential geometric priors while preserving the VLM’s general reasoning capacity.

Discussions. GSM is architecture-agnostic (works with different video VLM backbones and geometry encoders), parameter-efficient (fixed N queries), and robust to question length (language condensation normalizes variable Ttext). Empirically, it yields strong gains for DSR while maintaining performance on general video/VLM benchmarks.

###### 5. Experiments

###### 5.1. Experimental Settings

We adopt Qwen2.5-VL-7B as our base model and integrate it with the proposed GSM. The model is trained on 50K question-answer pairs from DSR-Train. All components are trainable except freezing the vision encoder of Qwen2.5VL-7B. The number of learnable queries N in GSM is set to 32. The model is trained for 1 epoch the learning rate of 2×10−7 and the batch size of 32.

###### 5.2. Comparison with State-of-The-Arts

We first compare our model’s performance against the following state-of-the-art VLMs on DSR-Bench and report their performance on different subtasks:

- • Proprietary models, including GPT-4o [15], GPT-5 [22], Gemini-2.5-Flash [8], Gemini-2.5-Pro [8];
- • Video understanding models, including LLaVA-Video7B [42], VideoRefer [39], LongVILA-R1 [5];
- • General-purpose models, including Qwen2.5-VL-7B [1], Qwen2.5-VL-32B [1], Qwen3-VL-8B-Instruct [23], Qwen3-VL-30B-A3B-Instruct [23], InternVL3.5-8B [30], InternVL3.5-38B [30];
- • Spatial reasoning models–VLM-3R [10], VG-LLM [43]. The results in Table 4 indicate that our model achieves

the best performance across all subtasks, as well as the

Table 5. Comparison between GSW and other training methods.

|Methods<br><br>|Benchmarks|
|---|---|
| |DSR-Bench VLM4D STI-Bench Video-MME Avg.|
|Baseline SFT Addition GSM|23.5 43.1 33.2 60.2 40.0 54.4 46.7 34.6 60.1 48.9 57.7 48.5 35.3 48.6 47.5 57.4 48.3 35.2 59.9 50.2<br><br>|

highest average performance. For models not explicitly designed for spatial reasoning, the performance is only marginally above random guess, revealing their limited capacity for this task. Even spatial reasoning models, though trained on static scenes, still fall short, underscoring the need of dedicated DSR training data. Together, these observations highlight the inherent challenges of DSR.

###### 5.3. Ablation Studies

Effect of GSM As discussed in Section 4, GSM enables integration of geometric prior from 3D foundation model into VLMs while mitigating performance degradation on general video understanding tasks. In this section, we compare GSM with the following training paradigms to demonstrate its effectiveness: (1) SFT, which directly applies supervised fine-tuning to Qwen2.5-VL-7B on the same training dataset; and (2) Addition, which directly adds 3D tokens from π3 to the vision tokens before training on the same dataset. For efficiency, we randomly sample 20K QAs from DSR-Train for training all models.

Furthermore, to evaluate on other 3D benchmarks and general video understanding tasks, we adopt VLM4D [45], STI-Bench [18] as additional spatial reasoning benchmarks and Video-MME [11] as general video benchmark. The results in Table 5 show that training on DSR-Train helps improve performance not only on DSR-Bench, but also other benchmarks requiring dynamic spatial reasoning, regardless of training paradigm. Therefore, DSR-Train is effective for improving dynamic spatial reasoning capability. When compared with SFT, GSM helps improve performance on benchmarks requiring spatial reasoning, taking advantage of integration of geometric priors. When compared with Addition, GSM is competitive for dynamic spatial reasoning while retains general video understanding ability.

Effect of Query Number In this section, we conduct experiments to evaluate the impact of learnable query number in GSM. Specifically, we vary the number of queries across 8, 16, 32, 64, training all models on the same dataset containing 20K QAs randomly sampled from DSR-Train.

The results in Table 6 show that the model with more queries performs better for dynamic spatial reasoning. However, the increased queries also harm general video understanding performance, resulting in lower average perfor-

Table 6. Ablation of learnable query numbers.

|Query Number|Benchmarks|
|---|---|
| |DSR-Bench VLM4D STI-Bench Video-MME Avg.<br><br>|
|8 16 32 64<br><br>|55.7 47.2 34.6 59.9 49.3<br>56.9 47.8 34.8 60.0 49.8<br>57.4 48.3 35.2 59.9 50.2 57.6 48.5 35.1 59.2 50.0<br>|

58.9

[Figure 96]

60

57.4

55

53.3

50

47.3

###### Accuracy(%)

45

###### Data Scaling

40

35

30

25

23.5

###### Base Accuracy

20

15

5k 10k 20k 50k

Data Number

Figure 5. Performance curve of accuracy on DSR-Bench with varying numbers of question-answer pairs for training.

mance. Therefore, it is necessary to set a proper learnable query number to obtain the best overall performance.

Scalability of DSR-Train In this section, we analyze the scalability of DSR-Train by evaluating model performance when trained on varying numbers of QAs that are 5K, 10K, 20K and 50K. The results illustrated in Figure 5 show that the model performs better on DSR-Bench when training data increases, showing the effectiveness of DSR-Train.

###### 6. Conclusion

In this work, we present a unified framework DSR Suite for dynamic spatial reasoning in VLMs. It comprises an automated pipeline that builds DSR-Train for supervision and further human-refined DSR-Bench for comprehensive evaluation on in-the-wild videos. The data emphasize viewpoint transform, multi-object interact and fine-grained procedural answers. Beyond data, we introduced a lightweight Geometry Selection Module (GSM) that selectively integrates geometric priors via two stacked Q-Formers, avoiding noisy geometric overload to preserve general video understanding capability. Trained on DSR-Train, Qwen2.5-VL-7B + GSM achieves superior results on DSR-Bench without regressions on general benchmarks. We hope this suite and approach catalyze future work on 4D multimodal intelligence, including embodied perception, predictive reasoning and world modeling in dynamic environments.

###### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 7, 11
- [2] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. Nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11621–11631, 2020. 5
- [3] Wenxiao Cai, Iaroslav Ponomarenko, Jianhao Yuan, Xiaoqi Li, Wankou Yang, Hao Dong, and Bo Zhao. Spatialbot: Precise spatial understanding with vision language models. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 9490–9498. IEEE, 2025. 2
- [4] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14455–14465,

2024. 2

- [5] Yukang Chen, Wei Huang, Baifeng Shi, Qinghao Hu, Hanrong Ye, Ligeng Zhu, Zhijian Liu, Pavlo Molchanov, Jan Kautz, Xiaojuan Qi, et al. Scaling rl to long videos. Advances in Neural Information Processing Systems, 2025. 7
- [6] Yukang Chen, Fuzhao Xue, Dacheng Li, Qinghao Hu, Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang Yang, Zhijian Liu, et al. Longvila: Scaling long-context visual language models for long videos. In The Thirteenth International Conference on Learning Representations, 2025. 2
- [7] An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision-language models. Advances in Neural Information Processing Systems, 37:135062–135093, 2024. 2
- [8] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 3, 7, 11
- [9] Linxi Fan, Guanzhi Wang, Yunfan Jiang, Ajay Mandlekar, Yuncong Yang, Haoyi Zhu, Andrew Tang, De-An Huang, Yuke Zhu, and Anima Anandkumar. Minedojo: Building open-ended embodied agents with internet-scale knowledge. Advances in Neural Information Processing Systems, 35: 18343–18362, 2022. 14, 15
- [10] Zhiwen Fan, Jian Zhang, Renjie Li, Junge Zhang, Runjin Chen, Hezhen Hu, Kevin Wang, Huaizhi Qu, Dilin Wang, Zhicheng Yan, et al. Vlm-3r: Vision-language models augmented with instruction-aligned 3d reconstruction. arXiv preprint arXiv:2505.20279, 2025. 2, 3, 6, 7, 14
- [11] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever

- comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the IEEE/CVF Computer Vision and Pattern Recognition Conference, pages 24108– 24118, 2025. 8, 11, 14
- [12] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 3, 11
- [13] William H Guss, Brandon Houghton, Nicholay Topin, Phillip Wang, Cayden Codel, Manuela Veloso, and Ruslan Salakhutdinov. Minerl: a large-scale dataset of minecraft demonstrations. In Proceedings of the 28th International Joint Conference on Artificial Intelligence, pages 2442– 2448, 2019. 15
- [14] Songhao Han, Wei Huang, Hairong Shi, Le Zhuo, Xiu Su, Shifeng Zhang, Xu Zhou, Xiaojuan Qi, Yue Liao, and Si Liu. Videoespresso: A large-scale chain-of-thought dataset for fine-grained video reasoning via core frame selection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26181–26191, 2025. 2
- [15] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 7
- [16] Mengdi Jia, Zekun Qi, Shaochen Zhang, Wenyao Zhang, Xinqiang Yu, Jiawei He, He Wang, and Li Yi. Omnispatial: Towards comprehensive spatial reasoning benchmark for vision language models. arXiv preprint arXiv:2506.03135,

2025. 3, 6

- [17] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. Transactions on Machine Learning Research, 2024. 2
- [18] Yun Li, Yiming Zhang, Tao Lin, XiangRui Liu, Wenxiao Cai, Zheng Liu, and Bo Zhao. Sti-bench: Are mllms ready for precise spatial-temporal world understanding? In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025. 2, 3, 6, 8, 14
- [19] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 5971–5984, 2024. 2
- [20] Mingxian Lin, Wei Huang, Yitang Li, Chengjie Jiang, Kui Wu, Fangwei Zhong, Shengju Qian, Xin Wang, and Xiaojuan Qi. Embrace-3k: Embodied reasoning and action in complex environments. arXiv preprint arXiv:2507.10548,

2025. 15

- [21] Yunze Liu, Yun Liu, Che Jiang, Kangbo Lyu, Weikang Wan, Hao Shen, Boqiang Liang, Zhoujie Fu, He Wang, and Li Yi. Hoi4d: A 4d egocentric dataset for category-level humanobject interaction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21013–21022, 2022. 5
- [22] OpenAI. Introducing GPT-5, 2025. Accessed: 2025-09-21. 7

- [23] QwenTeam. Qwen3-vl: Sharper vision, deeper thought, broader action, 2025. Accessed: 2025-09-23. 7, 11
- [24] Arijit Ray, Jiafei Duan, Ellis Brown, Reuben Tan, Dina Bashkirova, Rose Hendrix, Kiana Ehsani, Aniruddha Kembhavi, Bryan A Plummer, Ranjay Krishna, et al. Sat: Dynamic spatial aptitude training for multimodal language models. In Second Conference on Language Modeling, 2025. 2, 3, 6
- [25] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, et al. Grounded sam: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159,

2024. 3

- [26] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al. Scalability in perception for autonomous driving: Waymo open dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2446–2454, 2020. 5
- [27] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5294–5306, 2025. 3, 6
- [28] Qiuheng Wang, Yukai Shi, Jiarong Ou, Rui Chen, Ke Lin, Jiahao Wang, Boyuan Jiang, Haotian Yang, Mingwu Zheng, Xin Tao, et al. Koala-36m: A large-scale video dataset improving consistency between fine-grained conditions and video content. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8428– 8437, 2025. 3, 11
- [29] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10510–10522, 2025. 3, 6
- [30] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. 7
- [31] Xingrui Wang, Wufei Ma, Angtian Wang, Shuo Chen, Adam Kortylewski, and Alan Yuille. Compositional 4d dynamic scenes understanding with physics priors for video question answering. In The Thirteenth International Conference on Learning Representations, 2025. 2, 5, 6
- [32] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π3: Scalable permutation-equivariant visual geometry learning. arXiv preprint arXiv:2507.13347,

2025. 3, 4, 6

- [33] Zehan Wang, Ziang Zhang, Tianyu Pang, Chao Du, Hengshuang Zhao, and Zhou Zhao. Orient anything: Learning robust object orientation estimation from rendering 3d models. In Forty-second International Conference on Machine Learning, 2025. 3, 4
- [34] Yuetian Weng, Mingfei Han, Haoyu He, Xiaojun Chang, and Bohan Zhuang. Longvlm: Efficient long video understand-

- ing via large language models. In European Conference on Computer Vision, pages 453–470. Springer, 2024. 2
- [35] Haoning Wu, Xiao Huang, Yaohui Chen, Ya Zhang, Yanfeng Wang, and Weidi Xie. Spatialscore: Towards unified evaluation for multimodal spatial understanding. arXiv preprint arXiv:2505.17012, 2025. 2, 3, 6
- [36] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10632– 10643, 2025. 2, 14
- [37] Sihan Yang, Runsen Xu, Yiman Xie, Sizhe Yang, Mo Li, Jingli Lin, Chenming Zhu, Xiaochen Chen, Haodong Duan, Xiangyu Yue, Dahua Lin, Tai Wang, and Jiangmiao Pang. Mmsi-bench: A benchmark for multi-image spatial intelligence. arXiv preprint arXiv:2505.23764, 2025. 2, 3, 6
- [38] Shusheng Yang, Jihan Yang, Pinzhi Huang, Ellis Brown, Zihao Yang, Yue Yu, Shengbang Tong, Zihan Zheng, Yifan Xu, Muhan Wang, et al. Cambrian-s: Towards spatial supersensing in video. arXiv preprint arXiv:2511.04670, 2025. 14
- [39] Yuqian Yuan, Hang Zhang, Wentong Li, Zesen Cheng, Boqiang Zhang, Long Li, Xin Li, Deli Zhao, Wenqiao Zhang, Yueting Zhuang, et al. Videorefer suite: Advancing spatialtemporal object understanding with video llm. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18970–18980, 2025. 7, 14
- [40] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 543–553, 2023. 2
- [41] Jiahui Zhang, Yurui Chen, Yanpeng Zhou, Yueming Xu, Ze Huang, Jilin Mei, Junhui Chen, Yu-Jie Yuan, Xinyue Cai, Guowei Huang, et al. From flatland to space: Teaching vision-language models to perceive and reason in 3d. CoRR,

2025. 2

- [42] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 7
- [43] Duo Zheng, Shijia Huang, Yanyang Li, and Liwei Wang. Learning from videos for 3d world: Enhancing mllms with 3d vision geometry priors. In Advances in Neural Information Processing Systems, 2025. 2, 3, 6, 7, 14
- [44] Duo Zheng, Shijia Huang, and Liwei Wang. Video-3d llm: Learning position-aware video representation for 3d scene understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8995– 9006, 2025. 2
- [45] Shijie Zhou, Alexander Vilesov, Xuehai He, Ziyu Wan, Shuwang Zhang, Aditya Nagachandra, Di Chang, Dongdong Chen, Xin Eric Wang, and Achuta Kadambi. Vlm4d: Towards spatiotemporal awareness in vision language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8600–8612, 2025. 2, 3, 6, 8, 14
- [46] Chenming Zhu, Tai Wang, Wenwei Zhang, Jiangmiao Pang, and Xihui Liu. Llava-3d: A simple yet effective pathway to empowering lmms with 3d-awareness. CoRR, 2024. 2, 3

## Learning to Reason in 4D: Dynamic Spatial Understanding for Vision Language Models Supplementary Material

This supplementary material contains additional details of the main manuscript. Section 7 presents additional implementation details for training data, evaluation and QA generation. Section 8 complements more experiments and analysis. Section 9 illustrates some examples of QA in our DSR-Train and DSR-Bench.

###### 7. Implementation Details

###### 7.1. Training Data Details

For 50K QAs in DSR-Train, we present the proportion of different question types in Figure 6.

Abs Dis Abs Dir Abs Ori

Rel Dis Rel Dir Rel Ori

8%

19%

8%

- 7%
- 8%

- 5%
- 6%
- 7%

Question

Abs Spd

Rel Spd

Type

Abs Spd Comp

Rel Spd Comp

5%

Abs Dir Pred

Rel Dir Pred

7% 6% 6%

7%

Non-Temp Based

Figure 6. Proportion of questions types in our DSR-Train.

The illustration shows that DSR-Train has similar question type distribution with DSR-Bench as in Figure 3(b) of the main manuscript. For the randomly sampled 5K, 10K, 20K QAs for training in Section 5.3, we restrict their distributions to be the same as that in Figure 6.

###### 7.2. Evaluation Details

During evaluation, we uniformly sample 32 frames from each input video as visual inputs for all VLMs. For all models except Qwen series [1, 23] and ours, which natively encode absolute timestamps, we additionally append the frame timestamps to the text input. To ensure a fair comparison, all VLMs are instructed to directly output final choice without intermediate reasoning. For evaluation on Video-MME [11], video subtitles are excluded from input.

###### 7.3. QA Generation Details

Object Denotation. In both DSR-Train and DSR-Bench, each object is referenced using a combination of its category and bounding box coordinates either at a specific timestamp or at the beginning of the queried sub-interval. This differs from some prior benchmarks that refer to objects solely through appearance descriptions, which can be ambiguous

in our setting where many objects share similar visual characteristics. When an object is used to determine the absolute viewpoint, it will be denoted as “{class} with bounding box coordinates (x1,y1,x2,y2) at {timev}”, where “{class}” is the object category and “(x1,y1,x2,y2)” correspond to the top-left and bottom-right coordinates of the bounding box in the frame sampled at the timestamp “{timev}”. Otherwise, the object is referred to as “{class} with initial bounding box coordinates (x1,y1,x2,y2), indicating that the coordinates correspond to the first frame of the queried subinterval and can change thereafter, or as “{class} with final bounding box coordinates (x1,y1,x2,y2)” to indicate that the coordinates are taken from the last sampled video frame and the task is to predict object’s moving direction.

Prompt Design. As described in Section 3.1 and 3.2 of the main manuscript, our automated QA generation pipeline uses DeepSeek-R1 [12] to filter out Koala-36M [28] videos exhibiting negligible object motion and to identify agent and non-agent object categories based on the video captions. To construct DSR-Bench, we further employ Gemini-

- 2.5-Pro [8] to filter videos with visual content and to assign scene classes with higher accuracy. Figure 7 presents

- our prompts for DeepSeek-R1 and Gemini-2.5-Pro, where “{caption}” denotes the video caption, and “{agent}” and “{object}” are agent and non-agent classes in the video.

In Section 3.3 of the main manuscript, to broaden linguistic variety and reasoning patterns, we also propose to prompt DeepSeek-R1 for non-template-based QA generation. In Figure 8, we show the detailed prompt for DeepSeek-R1 for QA generation, where “{viewpoint}” is the observer to decide the viewpoint for observing the objects, “{coord}” is the 3D coordinate trajectories of all objects, “{time1}” and “{time2}” are the start and end timestamp of the sub-interval, “{timestamps}” is the list of timestamps at which the object 3D coordinates are collected.

Question Generation Template. Part of our QAs are generated based on pre-defined templates, as stated in Section 3.3 of the main manuscript. We list these templates for each type of question in Table 7. “{times}”, “{timee}” are the timestamps of the start, end of the queries sub-interval and “{timev}” is the timestamp to determine the absolute viewpoint. “{obj1}”, “{obj2}” are the target objects and “{objv}” is the observer to decide the viewpoint, where their denotations are in the format described in the previ-

- ous Object Denotation section. Answer Generation Rule. In Section 3.3 of the main manuscript, we describe our approach to generate answers by comparing the queried attribute every 2 adjacent frames

Caption-Based Motion Filter Prompt: """{caption} is the caption of a video. Think about whether the video satisfies the following condition: Regardless of camera movement and articulated motion, the world coordinates of at least 3 main subjects in the video

change substantially. Only consider the objects that can be regarded as a whole and not part of other objects.

If the condition is satisfied, output “yes” in a new line. If the condition is not satisfied, output “no” in a new line."""

Caption-Based Agent/Non-Agent Extract Prompt: """{caption} is the caption of a video. In one line, list all classes of agents in the video and use dot to separate them. If there is no agent, output “None” in this line. Then in a new line, list the classes of remaining foreground objects in the video and use dot to separate them. An object should be a discrete, countable item (e.g. a bottle, a rock). The object must not be a part of other objects. If there is no object, output “None” in this line."""

(a) Caption-based motion filter and agent/non-agent extract prompt for DeepSeek-R1

Video-Based Motion Filter and Scene Classification Prompt: """The video belongs to one of following categories: Sports & Recreation, Transportation & Vehicle Operation, Art Performance, Manual Labor & Craftsmanship, Daily Activities & Hobbies, Nature & Wildlife. The agent in the video is {agent}. The object in the video is {object}. Think about whether the video satisfies the following conditions:

- (1) The camera is not mainly filming the inside of one moving object;
- (2) Regardless of camera movement, articulated motion and shape variance, the world coordinates of at least 3 agents (exclude hands) or object change substantially;
- (3) Every object cannot occupy less than 1% of the screen;
- (4) Every agent or object cannot occupy more than 50% of the screen.

Only consider the objects that can be regarded as a whole and not part of or containing other objects. Only consider the objects that are discrete, countable items (e.g. a bottle, a rock). Only consider the objects that are not in contact with each other. Only consider agents and objects belonging to classes given at the beginning. If all conditions are satisfied, output “yes” in a new line. If any condition is not satisfied, output “no” in a new line. Then in a new line, output the category of the video."""

(b) Video-based motion filter and scene classification prompt for Gemini-2.5-Pro

- Figure 7. Detailed prompts leveraged for DeepSeek-R1 and Gemini-2.5-Pro for data process.

Non-Template-Based QA Generation Prompt: """ From the perspective of {viewpoint} in a video, {coord} are the 3D coordinates

(x, y, z) of some objects between {time1} and {time2}. The positive direction of x, y, z axis are forward, left, up respectively. The

coordinates are collected at {timestamps}. According to the change of 3D relationship between objects, please generate one

question, which cannot be answered only with 2D knowledge and without 3D knowledge, and 4 answers where one of them is the

correct answer. The question should not include description of perspective. The question should not be quantitative and related to

accurate 3D coordinates. All objects must be referred with their bounding box coordinates given at the beginning. In a new line,

only output your question. Then in 4 new lines, only output your 4 answers. After that, only output the symbol of the correct

answer in a new line."""

- Figure 8. Detailed prompts leveraged for DeepSeek-R1 to generate non-template-based QAs.

to derive a sequence of basic choice defined in Table 2 of the main manuscript and then merging consecutive identical states. Table 8 provides the detailed derivation rules for each basic choice. After determining the correct answer, we compute the number of basic choices N it contains and generate alternative choices with random lengths within the range [max(1,N−3),N+3]. To maintain a balanced distribution, the correct answer label (A, B, C, or D) is randomly assigned. We note that frames in the final second are hidden from VLMs, as they are used to determine the object’s movement direction for direction prediction questions.

Table 7. Templates for different question types.

Type Template Rel Dis Between {times} and {timee}, following the perspective of {objv}, how does the distance between {obj1} and {obj2} change? Rel Dir Between {times} and {timee}, following the perspective of {objv}, how does the direction of {obj1} to {obj2} change? Rel Ori Between {times} and {timee}, following the perspective of {objv}, how does the orientation of {obj1} change? Rel Spd Between {times} and {timee}, following the perspective of {objv}, how does the speed of {obj1} change?

Rel Spd Comp Between {times} and {timee}, following the perspective of {objv}, compare the speed between {obj1} and {obj2}.

Rel Dir Pred Following the perspective of {objv}, predict the moving direction of {obj1}. Abs Dis Between {times} and {timee}, from the perspective of {objv} at {timev}, how does the distance between {obj1} and {obj2} change? Abs Dir Between {times} and {timee}, from the perspective of {objv} at {timev}, how does the direction of {obj1} to {obj2} change? Abs Ori Between {times} and {timee}, from the perspective of {objv} at {timev}, how does the orientation of {obj1} change? Abs Spd Between {times} and {timee}, from the perspective of {objv} at {timev}, how does the speed of {obj1} change?

Abs Spd Comp Between {times} and {timee}, from the perspective of {objv} at {timev}, compare the speed between {obj1} and {obj2}. Abs Dir Pred From the perspective of {objv} at {timev}, predict the moving direction of {obj1}.

Table 8. Derivation rules of different basic choices to make up answers.

Type Basic Choice Derivation Rule

- (1) Keep nearly constant then become larger At each timestamp within a period, except final one, the distance/speed remains within 0.8× to 1.2× that of the first timestamp. At the final timestamp, the distance/speed exceeds 1.2× that of the first timestamp.
- (2) Keep nearly constant then become smaller At each timestamp within a period, except final one, the distance/speed remains within 0.8× to 1.2× that of the first timestamp. At the final timestamp, the distance/speed is below 0.8× that of the first timestamp.

Distance & Speed

(3) Keep nearly constant At each timestamp within a period, except final one, the distance/speed remains within 0.8× to 1.2× that of the first timestamp.

- (4) Become larger At each timestamp within a period, the distance/speed exceeds 1.2× that of the former one.
- (5) Become smaller At each timestamp within a period, the distance/speed falls below 0.8× that of the former one.

- (1) Front/Behind The angle between the unit vector of direction/orientation and the forward unit vector from viewpoint is smaller/larger than 70/110 degrees.
- (2) Left/Right The angle between the unit vector of direction/orientation and the left unit vector from viewpoint is smaller/larger than 70/110 degrees.
- (3) Above/Below The angle between the unit vector of direction/orientation and the upward unit vector from viewpoint is smaller/larger than 70/110 degrees.

Direction & Orientation & Direction Prediction

- (1) Nearly the same The speed of the former object is within 0.83× to 1.20× that of the latter object at one timestamp.
- (2) The former is faster The speed of the former object exceeds 1.20× that of the latter object at one timestamp.
- (3) The latter is faster The speed of the former object falls below 0.83× that of the latter object at one timestamp.

Speed Comparison

###### 8. Complementary Experiments

In this section, we present additional experiments that were not included in the main manuscript. In Section 8.1, we provide further analysis of the results in Table 4 of main manuscript. Section 8.2 compare the performance of models trained on data with different question type distributions. In Section 8.3, we replace the original base model, i.e., Qwen2.5-VL-7B, with Qwen3-VL-8B and report performance across different benchmarks to demonstrate the effectiveness of GSM and DSR-Train on different models. Section 8.4 incorporates DSR-Train with question–answer pairs for spatial reasoning in static scenes to train Qwen2.5VL-7B with unified static and dynamic spatial reasoning capabilities. Finally, Section 8.5 extends the DSR-Train finetuned model fine-tuned to MineDojo [9], demonstrating its broader applicability to downstream agent tasks requiring dynamic spatial reasoning.

###### 8.1. Additional Result Analysis

In addition to the conclusions in Section 5.2, Table 4 of the main manuscript further reveals that spatial reasoning models, i.e., VLM-3R [10] and VG-LLM [43], achieve even better performance than proprietary models, despite being trained only for static spatial reasoning. Therefore, 3D relevant data is necessary to improve spatial reasoning capability. Moreover, compared with its base model LLaVAVideo-7B, the marginal performance gain of VideoRefer [39], which is explicitly trained to understand location information for object reference, suggests that the unsatisfactory performance on DSR-Bench is not attributable to our use of bounding-box–based object references. Instead, it highlights the insufficient capability of current models to perform dynamic spatial reasoning.

###### 8.2. Effect of Training Question Distribution

For all experiments in the main manuscript, our model is trained on QAs with the type distribution shown in Figure 6. In this section, we fix the total number of training QAs to be 20K and vary the proportion of template-based and non-template-based QAs under the following configurations: (1) 20K template-based QAs; (2) 16K template-based QAs and 4K non-template-based QAs (the setting adopted in the main manuscript experiments); (3) 10K templatebased QAs and 10K non-template-based QAs; (4) 20K nontemplate-based QAs. Table 9 reports the performance on the template-based subset, non-template-based subset, the full DSR-Bench and Video-MME under each configuration. The results indicate that both QA types are essential for achieving strong overall performance, and that templatebased QAs should constitute the majority.

- Table 9. Comparison between different training data settings.

|Settings<br><br>|Benchmarks|
|---|---|
| |DSR-Bench (Temp)<br><br>DSR-Bench (Non-Temp)<br><br>DSR-Bench Video-MME|
|Baseline<br><br>1<br><br>2<br><br>3<br><br>4<br><br><br>|22.1 30.1 23.5 60.2 61.5 32.3 56.2 59.6 60.4 43.7 57.4 59.9 58.1 46.7 56.1 60.1 35.2 49.1 37.6 60.5|

- Table 10. Comparison between GSM and other methods with Qwen3-VL-8B-Instruct as the base model.

|Methods<br><br>|Benchmarks|
|---|---|
| |DSR-Bench VLM4D STI-Bench Video-MME Avg.|
|Baseline SFT Addition GSM<br><br>|28.7 46.3 36.3 64.9 44.0 56.8 47.8 37.3 64.7 51.6 59.0 49.5 37.8 53.6 49.9 58.6 49.2 37.9 64.4 52.5|

###### 8.3. Ablation on Different Base Model

In the main manuscript experiments, we adopt Qwen2.5VL-7B as the base model, integrate our proposed GSM into it and train it on DSR-Train. To demonstrate the effectiveness of GSM and DSR-Train independent of the underlying base model, we further replace Qwen2.5-VL-7B with Qwen3-VL-8B-Instruct, incorporate GSM, and train it on the same 20K samples from DSR-Train as used in Section 5.3 of the main manuscript. We also compare its performance with different training methods when adopting the same compared training paradigm in Section 5.3 of main manuscript, i.e., SFT and Addition. The results on DSRBench, STI-Bench [18], VLM4D [45] and Video-MME [11] listed in Table 10 show that our DSR-Train improves the dynamic spatial reasoning performance of Qwen3-VL8B-Instruct as well. Compared with SFT, GSM further improves performance by integrating explicit geometric priors. In contrast to Addition, GSM maintains the general video-understanding ability of Qwen3-VL-8B-Instruct, preventing degradation of broader visual reasoning skills.

###### 8.4. Performance with Data Mixture

In all previous experimental settings, the base model Qwen2.5-VL-7B is fine-tuned exclusively on our DSRTrain to perform spatial reasoning in dynamic scenes. In this section, we further mix QAs for static spatial reasoning with DSR-Train to fine-tune Qwen2.5-VL-7B, improving both static and dynamic spatial reasoning ability. Specifically, we construct a total of 800K static spatial reasoning QAs using data from VLM-3R [10] and Cambrian-S [38], and adopt VSI-Bench [36] as the benchmark for evaluating

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

(a) hunt a cow in plains with a iron sword, shield, and a full suite of iron armors

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

(b) combat a spider in night plains with a diamond sword, shield, and a full suite of leather armors

Figure 9. Examples of agent results on MineDojo.

Table 11. Performance analysis of models trained with different spatial reasoning scenes.

|Training Scene<br><br>|Benchmarks|
|---|---|
| |DSR-Bench VSI-Bench Video-MME<br><br>|
|Baseline Static Static+Dynamic|23.5 33.4 60.2<br><br>24.1 55.3 58.9<br><br><br>60.2 56.1 59.2|

Table 12. Success rate comparison on MineDojo between agents adopted from VLM trained with different spatial reasoning scenes.

|Training Scene<br><br>|Tasks|
|---|---|
| |Animals Hostiles|
|Baseline Static Dynamic<br><br>|15.6±3.4 10.2±2.4<br>16.3±2.3 12.4±3.1 26.5±1.7 22.3±2.3<br>|

static spatial reasoning. The performance of model trained only on static spatial reasoning or mixed QAs on DSRBench, VSI-Bench and Video-MME are listed in Table 11. From the results, when the model is trained only with static spatial reasoning QAs, it is still unable to perform dynamic spatial reasoning. With both static and dynamic spatial reasoning data, the trained model obtains the best performance simultaneously on DSR-Bench and VSI-Bench while general understanding capability is still preserved.

###### 8.5. Extension to Agent Tasks

In this section, we also explore the downstream performance on agent tasks [9, 20], showing the spatial capability on real-time working agent. We further extend the Qwen2.5-VL-7B fine-tuned with DSR-Train to MineDojo [9] Benchmark, which simulates a Minecraft game agent to solve different tasks, to show the application of dynamic spatial reasoning models in downstream agent tasks. Since the training dataset of MineDojo only provides videos and task instructions without intermediate action labels, we instead leverage the data from MineRL [13] for training by mapping its action space to that of MineDojo. For evaluation, we select part of tasks from MineDojo that are related to the following two sets of objects: (1) Animals including cow, sheep, pig and chicken, whose related tasks are targeted at combating or harvesting them; (2) Hostiles including spider, zombie, pigman and enderman, whose related tasks aim to combat them. Since these tasks require

interacting with dynamic objects, they command dynamic spatial reasoning ability of agents. For compared models, we train original Qwen2.5-VL-7B and that fine-tuned with static spatial reasoning data as stated in Section 8.4 on the same MineRL data. Each task is tested on 3 random seeds and with 200 episodes. The average and variance of success rate are listed in Table 12, showing the advantage of our model in downstream agent tasks. Some task results of the agent adopted from model trained with DSR-Train are illustrated in Figure 9.

###### 9. Visualization

In this section, we illustrate the video and QA examples of different categories in our DSR-Train and DSR-Bench in Figure 10, 11 and 12, which shows our in-the-wild video source and comprehensive evaluation aspects including multi-object interaction, viewpoint transformation and fine-grained temporal reasoning. Note that the arrows in the figures indicate the movement from the viewpoint of camera, which can be different when the viewpoint changes.

|Question (Relative Distance): Between 13s and 35s, following the perspective of camera, how does the distance between the car with initial bounding box coordinates [124, 116, 263, 224] and camera change?<br><br>[Figure 107]<br><br>…<br><br>Forward & Right<br><br>Choices: A. (1) Become smaller. (2) Become larger. B. (1) Keep nearly constant then become larger. (2) Become larger. (3) Become smaller.<br><br>C. (1) Become larger. (2) Keep nearly constant then become smaller. (3) Become smaller. D. (1) Keep nearly constant.<br><br>[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]<br><br>[Figure 111]<br><br>[Figure 112]<br><br>… … …<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Forward Forward Backward & Left<br><br>Question (Relative Direction): Between 16s and 18s, following the perspective of the car with initial bounding box coordinates [78, 156, 304, 284] , how does the direction of the car with initial bounding box coordinates [0, 174, 67, 234] to the car with initial bounding box coordinates [78, 156, 304, 284] change?<br><br>Choices: A. (1) Front|Right|Above. (2) Behind|Left. B. (1) Left. (2) Behind|Left C. (1) Right|Below. D. (1) Left. (2) Left|Above.<br><br>[Figure 113]<br><br>[Figure 114]<br><br>[Figure 115]<br><br>[Figure 116]<br><br>[Figure 117]<br><br>[Figure 118]<br><br>… … … …<br><br>| | |
|---|---|
| | |
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Forward & Right Right Forward & Right Forward & Right<br><br>Left Left Left Left<br><br>Transportation & Vehicle Operation|
|---|

|Question (Relative Orientation): Between 0s and 10s, following the perspective of camera, how does the orientation of the horse with initial bounding box coordinates [204, 182, 236, 223] change?<br><br>[Figure 119]<br><br>Choices: A. (1) Behind|Left|Below. (2) Behind|Left. (3) Left. (4) Left|Above. (5) Front|Left|Above<br><br>B. (1) Left. (2) Left|Below. (3) Front|Right|Above. (4) Left|Above. (5) Front|Left|Above. (6) Left|Above<br>C. (1) Right. (2) Front|Left. (3) Left|Above. (4) Front|Left|Below<br>D. (1) Left|Below. (2) Behind|Left. (3) Left|Below<br><br><br>Question (Relative Speed): Between 3s and 10s, following the perspective of the man with initial bounding box coordinates [44, 96, 79, 275] , how does the speed of the man with initial bounding box coordinates [413, 87, 436, 162] change?<br><br>Choices: A. (1) Keep nearly constant then become larger. (2) Keep nearly constant.<br><br>B. (1) Become larger. (2) Become smaller. (3) Become larger. (4) Keep nearly constant<br>C. (1) Become smaller. (2) Become larger. (3) Keep nearly constant.<br>D. (1) Become larger. (2) Become smaller. (3) Keep nearly constant<br><br><br>Sports & Recreation<br><br>[Figure 120]<br><br>[Figure 121]<br><br>[Figure 122]<br><br>[Figure 123]<br><br>[Figure 124]<br><br>… … … …<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Backward & Left Left Forward & Right & Upward Forward &Upward<br><br>[Figure 125]<br><br>[Figure 126]<br><br>[Figure 127]<br><br>[Figure 128]<br><br>[Figure 129]<br><br>[Figure 130]<br><br>… … … …<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Forward & Right Backward & Left Upward Static<br><br>Static Static Static Static|
|---|

|Question (Relative Speed Comparison): Between 0s and 6s, following the perspective of camera, compare the speed between the animal with<br><br>initial bounding box coordinates [253, 44, 326, 208] and the animal with initial bounding box coordinates [384, 43, 447, 196].<br><br>[Figure 131]<br><br>Right<br><br>Choices: A. (1) The former is faster. (2) The latter is faster. (3)The former is faster. (4) The latter is faster.<br><br>B. (1) Nearly the same. (2) The former is faster. (3) The latter is faster. (4) Nearly the same. (5) The latter is faster.<br>C. (1) The latter is faster. (2) Nearly the same . (3) The former is faster. (4) Nearly the same.<br>D. (1) The former is faster. (2) Nearly the same. (3) The former is faster.<br><br><br>Left Left Left<br><br>Question (Relative Direction Prediction): Following the perspective of camera, predict the moving direction of the man with final bounding box coordinates [143, 57, 164, 306]?<br><br>Choices: A. (1) Behind|Left B. (1) Right|Below C. (1) Right D. (1) Behind|Right|Above<br><br>|[Figure 132]<br><br>[Figure 133]<br><br>[Figure 134]<br><br>[Figure 135]<br><br>[Figure 136]<br><br>… … …<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Right Right Static|
|---|
<br><br>Art Performance<br><br>[Figure 137]<br><br>[Figure 138]<br><br>[Figure 139]<br><br>[Figure 140]<br><br>[Figure 141]<br><br>… … … …<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| | |
|---|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Right Left Right Backward & Right<br><br>[Figure 142]<br><br>…<br><br>| |
|---|
<br><br>Right<br><br>Invisible to VLM|
|---|

|Manual Labor & Craftsmanship<br><br>Question (Absolute Distance): Between 21s and 25s, from the perspective of camera at 18s, how does the distance between the man with initial bounding box coordinates [125, 86, 324, 359] and the camera change?<br><br>[Figure 143]<br><br>Choices: A. (1) Become larger. (2) Become smaller. B. (1) Become smaller<br><br>C. (1) Keep nearly constant then become larger. (2) Become larger. D. (1) Become larger.<br><br>|Viewpoint<br><br>[Figure 144]|
|---|
<br><br>Question (Absolute Direction): Between 16s and 21s, from the perspective of the woman with bounding box coordinates [134, 156, 163, 359] at 14s, how does the direction of the man with initial bounding box coordinates [547, 136, 639, 359] to the woman with initial bounding box coordinates [47, 153, 82, 359] change?<br><br>[Figure 145]<br><br>Choices: A. (1) Behind|Left. (2) Front|Right. (3) Left. (4) Left|Above. B. (1) Front|Right. (2) Front. (3) Front|Left<br><br>C. (1) Right. (2) Front|Right. (3) Left|Above. (4) Front|Left D. (1) Left|Below. (2) Behind|Left.<br><br>|[Figure 146]<br><br>Viewpoint<br><br>| |
|---|
|
|---|
<br><br>[Figure 147]<br><br>[Figure 148]<br><br>[Figure 149]<br><br>[Figure 150]<br><br>… … …<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| | |
|---|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Right Forward & Right<br><br>Forward Forward Static<br><br>Forward & Right<br><br>[Figure 151]<br><br>[Figure 152]<br><br>[Figure 153]<br><br>[Figure 154]<br><br>… … …<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Backward & Left Backward Backward|
|---|

|Question (Absolute Orientation): Between 0s and 5s, from the perspective of the woman with bounding box coordinates [194, 86, 241, 213] at 14s, how does the orientation of the woman with initial bounding box coordinates [304, 65, 363, 226] change?<br><br>Choices: A. (1) Front. (2) Behind. B. (1) Front|Left. (2) Right. C. (1) Behind. (2) Behind|Left. (3) Left|Above. D. (1) Front.<br><br>Daily Activities & Hobbies<br><br>[Figure 155]<br><br>|Viewpoint<br><br>[Figure 156]<br><br>| |
|---|
|
|---|
<br><br>[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]<br><br>[Figure 160]<br><br>… … …<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Forward Backward & Right Backward<br><br>Question (Absolute Speed): Between 3s and 21s, from the perspective of camera at 1s, how does the speed of the woman with initial bounding box coordinates [206, 154, 247, 325] change?<br><br>Choices: A. (1) Become larger. B. (1) Become smaller. C. (1) Keep nearly constant then become larger. D. (1) Keep nearly constant.<br><br>[Figure 161]<br><br>|Viewpoint<br><br>[Figure 162]|
|---|
<br><br>[Figure 163]<br><br>[Figure 164]<br><br>[Figure 165]<br><br>[Figure 166]<br><br>… … …<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Forward Forward Forward|
|---|

|Nature & Wildlife<br><br>Question (Absolute Speed Comparison): Between 1s and 8s, from the perspective of camera at 3s, compare the speed between the dog with initial bounding box coordinates [43, 165, 286, 197] and the dog with initial bounding box coordinates [322, 183, 361, 204]?<br><br>[Figure 167]<br><br>Choices: A. (1) The latter is faster. (2) The former is faster. (3) Nearly the same. B. (1) The former is faster. (2) Nearly the same. C. (1) The former is faster. (2) The latter is faster. (3) The former is faster. D. (1) Nearly the same. (2) The former is faster.<br><br>|Viewpoint<br><br>[Figure 168]|
|---|
<br><br>Question (Absolute Direction Prediction): From the perspective of camera at 16s, predict the moving direction of the animal with final bounding box coordinates [514, 164, 558, 192].<br><br>Choices: A. (1) Right|Above. B. (1) Front|Left. C. (1) Left. D. (1) Left|Below.<br><br>|Viewpoint<br><br>[Figure 169]|
|---|
<br><br>[Figure 170]<br><br>[Figure 171]<br><br>[Figure 172]<br><br>[Figure 173]<br><br>| | |
|---|---|
| | |
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Forward &Right Backward Static<br><br>Static Static Static<br><br>|[Figure 174]<br><br>[Figure 175]<br><br>[Figure 176]<br><br>[Figure 177]<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Left Left & Upward Left|
|---|
<br><br>Invisible to VLM<br><br>[Figure 178]<br><br>… … …<br><br>| |
|---|
<br><br>[Figure 179]<br><br>|Viewpoint<br><br>[Figure 180]|
|---|
<br><br>… … …<br><br>[Figure 181]<br><br>[Figure 182]<br><br>[Figure 183]<br><br>[Figure 184]<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| | |
|---|---|
| | |
<br><br>| | |
|---|---|
| | |
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Backward &Right Left Static<br><br>Static Static Static<br><br>Question (Non-Template-Based): From the perspective of camera at 5s, between 3s and 12s, which dog is averagely closer to the doghouse? Choices: A. The dog with initial bounding box coordinates [56, 196, 253, 284]. B. Neither of them.<br><br>C. The dog with initial bounding box coordinates [314, 204. 396, 321]. D. Both of them.|
|---|

###### Figure 12. Examples of QA and video in DSR-Train and DSR-Bench.

