# arXiv:2512.13660v3[cs.RO]27Jun2026

## Towards Spatial Trace with Reasoning in Vision-Language Models for Robotics

Enshen Zhou1∗‡, Yibo Li1∗, Jingkun An1∗, Jiayuan Zhang1∗, Shanyu Rong2,3, Mengzhen Liu2, Yi Han1,3, Yuheng Ji3,5, Huajie Tan2,3, Jiawei He3, Pengwei Wang3, Zhongyuan Wang3, Cheng Chi3†, Lu Sheng1,4†, Shanghang Zhang2,3†

1School of Software, Beihang University, 2 State Key Laboratory of Multimedia Information Processing, School of Computer Science, Peking University, 3 Beijing Academy of Artificial Intelligence, 4 Beijing Key Laboratory of Intelligent Creative Content Generation and Immersive Experience, 5 CASIA zhouenshen@buaa.edu.cn, leeibo@buaa.edu.cn, anjingkun02@gmail.com chicheng@baai.ac.cn, lsheng@buaa.edu.cn, shanghang@pku.edu.cn

Abstract. Spatial tracing, as a fundamental embodied interaction ability for robots, is inherently challenging as it requires multi-step metricgrounded reasoning compounded with complex spatial referring and realworld metric measurement. However, existing methods struggle with this compositional task. To this end, we propose RoboTracer, a 3D-aware VLM that first achieves both 3D spatial referring and measuring via a universal spatial encoder and a regression-supervised decoder to enhance scale awareness during supervised fine-tuning (SFT). Moreover, RoboTracer advances multi-step metric-grounded reasoning via reinforcement fine-tuning (RFT) with metric-sensitive process rewards, supervising key intermediate perceptual cues to accurately generate spatial traces. To support SFT and RFT training, we introduce TraceSpatial, a largescale dataset of 30M QA pairs, spanning outdoor/indoor/tabletop scenes and supporting complex reasoning processes (up to 9 steps). We further present TraceSpatial-Bench, a challenging benchmark filling the gap to evaluate spatial tracing. Experimental results show that RoboTracer surpasses baselines in spatial understanding, measuring, and referring, with an average success rate of 79.1%, and also achieves SOTA performance on TraceSpatial-Bench by a large margin, exceeding Gemini2.5-Pro by 36% accuracy. Notably, RoboTracer can be integrated with various control policies to execute long-horizon, dynamic tasks across diverse robots (UR5, G1 humanoid) in cluttered real-world scenes. Please see the project page at https://zhoues.github.io/RoboTracer.

Keywords: Spatial tracing · Spatial Reasoning · Vision-Language Model

### 1 Introduction

Embodied robots usually have to execute actions based on increasingly complex, spatially constrained instructions [1,80,82,112], such as “Water flowers from left

∗ Equal contribution † Corresponding author ‡ Project leader

[Figure 1]

Spatially Constrained Instruction: Water flowers from left to right with watering can hovering 1-5 cm above each flower.

2

- (a) 3D Spatial Referring The Second Flower from left

(493,659,0.963)

(240,684,0.986)

The First Flower from left

Height:0.119m (c) Spatial Trace in 2D (d) Spatial Trace

The Third Flower from left

Height:0.278m

The fourth Flower from left

Height:0.356m

[Figure 2]

- 1

- 2

3

4

5

[Figure 3]

[Figure 4]

- Fig. 1: Spatial tracing is pivotal for embodied robots to translate the spatially constrained instructions into 3D positional sequences (i.e., spatial traces) in complex 3D scenes. This task demands (a) 3D spatial referring to resolve spatial relations and locate relevant objects involved in the trace, and (b) 3D spatial measuring to understand absolute, real-world metric quantities related to the trace. For example, (a) 3D positions of the watering can, and each flower pot are localized from left to right, and

- (b) their corresponding heights in meters are measured. By performing multi-step, metric-grounded reasoning over the key information above, the generated spatial trace can support not only (c) multi-step manipulation, but also (d) collision-free motion, thereby (e) enabling efficient control of diverse robots across tasks in cluttered scenes.

1

[Figure 5]

3

4

(b) 3D Spatial Measuring

[Figure 6]

5

Watering can

(e) Real-world Test

to right with watering can hovering 1–5 cm above each one” in Fig. 1, where recent data-scarce Vision-Language-Action (VLA) models fail to master. In this case, it would be beneficial to generate a 3D positional sequence, named as spatial trace, as an intuitive bridge to interpret the instruction following procedure in 3D space and guide the generation of actual action trajectories for robots. However, this surrogate task (i.e., spatial tracing) is inherently challenging as it requires multi-step, metric-grounded reasoning in complex 3D scenes. To be specific, each reasoning step requires two key components: (1) 3D spatial referring to resolve spatial relationships and accurately localize objects involved in the trace generation (e.g., identifying flowers with their from left to right order and locating them). (2) 3D spatial measuring to understand absolute, real-world metric quantities related to the trace in captured scene (e.g., quantifying each flower’s physical height and 1–5 cm height above each).

While recent Vision-Language Models (VLMs) [5, 48, 56] can perform 2D spatial reasoning [11, 18, 97, 112] and even 2D visual trace (i.e., 2D positional sequence) generation [33, 43, 99], they overlook the multi-step nature of this task, particularly the crucial participation of intermediate objects involved in the trace, resulting in suboptimal generation. Moreover, their outputs are mainly in 2D space, lacking 3D space grounding and absolute metric understanding, creating a fundamental chasm between 2D visual trace and 3D spatial trace.

To this end, we propose RoboTracer, a 3D-aware reasoning VLM that not only acquires precise 3D spatial referring and measuring via Supervised Finetuning (SFT), but also exhibits multi-step metric-grounded reasoning capabilities for spatial tracing via reinforcement fine-tuning (RFT). The core of our approach is to introduce a set of metric-sensitive reward functions (e.g., refer-

ring, measuring, scale) during RFT. These rewards supervise the key perceptual objects involved in the trace and offer crucial intermediate evidence for accurate spatial trace generation. In addition, as 3D spatial referring and measuring require better metric-grounded understanding, we introduce a scale decoder for VLM, supervised by a regression loss on the predicted metric scale factor to enhance metric perception, even from RGB inputs. Moreover, we incorporate a universal spatial encoder with the architectural flexibility to integrate diverse geometric configurations (e.g., camera intrinsics, absolute depth) and further improve the precision of spatial trace generation.

To support RoboTracer’s training, we present TraceSpatial, a large-scale dataset with 4.5M high-quality examples and 30M QA pairs to first learn 3D spatial referring and measuring in SFT, and further achieve spatial tracing with compositional reasoning on both in RFT. It spans diverse indoor, outdoor and tabletop scenes with fine-grained annotations (e.g., precise geometry, object-level spatial referents) and contains a greatly higher proportion (48.2%) of absolute-scale data (14x prior [75]) for metric-grounded understanding. To advance metric-grounded multi-step reasoning capabilities, it provides step-wise annotations of the reasoning process (up to 9 steps). Moreover, it has objectcentric and end-effector-centric spatial traces spanning 3 single-arm and dualarm robot configurations, enhancing generalization and real-world applicability.

We evaluate RoboTracer on spatial understanding, measuring, and referring benchmarks, achieving SOTA average success rate of 79.1%, exceeding Gemini-

- 2.5-Pro by 11%. It also outperforms all baselines on 2D visual trace benchmarks. To address the lack of benchmarks for spatial tracing, we introduce TraceSpatialBench, which contains 100 real-world images with manually annotated tasks involving object localization, movement, and placement. Each sample requires metric-grounded, multi-step reasoning (up to 8 steps), with precise start-point masks, end-point 3D bounding boxes, precise geometry annotation, and diverse metrics for fine-grained evaluation. RoboTracer still achieves best performance, surpassing Gemini-2.5-Pro by 36%. Moreover, in Fig. 1 and Sec. 4.5, RoboTracer can execute long-horizon, dynamic tasks in cluttered real-world scenes, showing strong generalization across robots (e.g., UR5, G1 humanoid).

Our contributions are summarized as follows: (1) We propose RoboTracer, a 3D-aware VLM that accepts arbitrary geometric inputs, uses scale supervision, guided by metric-sensitive rewards to achieve spatial tracing. (2) We construct TraceSpatial, a well-annotated dataset tailored for spatial tracing, enabling both SFT and RFT training, and TraceSpatial-Bench, a benchmark that fills the gap in evaluating it. (3) Extensive experiments show that RoboTracer generalizes well, surpasses baselines in spatial understanding, measuring, referring, tracing, and efficiently controls diverse robots across tasks in real world.

### 2 Related work

Spatial Reasoning with VLMs. Spatial reasoning refers to the ability to perceive and reason about 3D space, comprising metric-agnostic and -grounded

types. Metric-agnostic reasoning [11,18,74,93,112] captures object-centric properties (e.g., position, orientation) and inter-object relations (e.g., left/right, near/far), whereas metric-grounded reasoning involves precise, absolute-scale measurements [9,12,75] (e.g., distance, depth, size) in the physical world. Compared to metric-agnostic one, metric-grounded reasoning requires better 3D understanding. Despite using 3D modalities, either explicitly [9,12,17,61,112] (e.g., point clouds, depth maps) or implicitly [16,26,91,109] (e.g., VGGT [87]), existing methods still struggle to reason complex absolute-scale scenes for 3D spatial referring and measuring. We thus propose a 3D-aware VLM that uses scale supervision and accepts arbitrary geometric inputs to address this gap.

Trace Generation with VLMs for Robotics. Trace enhances manipulation by capturing the spatio-temporal dynamics of objects. Recent advances in VLMs [2,4,5,15,32,42,65,66,76,77,81,89,113,114] focus on predicting 2D visual traces (i.e., 2D point sequences) and guiding robotic actions in two ways. (1) Lift-to-3D [33,80,92,99]: Projects 2D traces into 3D spatial trace using depth maps and camera intrinsics. (2) Overlap-on-2D [41,43,110]: Renders the 2D trace onto the image by steering the control policies for action generation. However,

- 2D traces struggle to fully capture object dynamics in 3D space. Moreover, existing works for 2D visual trace struggle to handle complex spatially constrained instructions and supervise the key perception steps, also required by multi-step
- 3D spatial tracing, largely due to the lack of datasets. We thus propose a new dataset and benchmark tailored for spatial tracing.

#### Reinforcement Fine-tuning for VLMs. Reinforcement Fine-tuning (RFT) [7,

67, 71] is a post-training strategy that aligns models with human preferences or specific goals via feedback, complementing SFT [90,111], which adapts pretrained models using task-oriented data. Recent advances in VLMs use RFT to improve visual/spatial reasoning [3,34,78,79,95,105], grounding [57,72,103,112], segmentation [55], tool-use [29,107], coding [104], safety [58,59], and 2D visual trace prediction [30,55,99]. However, most approaches remain confined to 2Drelative perception and outcome-based reward, limiting their performance on spatial tracing tasks requiring 3D multi-step metric-grounded reasoning. Therefore, we design metric-sensitive process rewards to supervise key perception steps in the reasoning process to meet the above expectations.

### 3 Method

#### 3.1 Problem Formulation and Representation Advantages

We formulate spatial tracing as predicting an ordered sequence of 6−12 3D sparse and discrete keypoints τ = {pt}Tt=1 from visual inputs O (e.g., RGB, RGB-D), optional camera geometry G (e.g., intrinsics) and a textual instruction T via a VLM in a open-loop manner. Each point pt = (ut,vt,dt) comprises a imageplane coordinate (ut,vt) and corresponding absolute depth dt, encoding essential object-related and collision-critical waypoints. The resulting trace τ serves as a

spatial plan for entities (e.g., a robot end-effector or object), and can be integrated with motion planning to predict physically feasible actions via adherence to the trace to follow the instruction. Notably, the instruction encodes both 3D spatial referring and measuring, often requiring multi-step compositional reasoning. For example, in Fig. 2, to accomplish the task of “Water flowers from left to right with watering can hovering 1-5 cm above each flower”, requires inferring the 3D positions and heights of all flowers in 3D scenes. These spatial cues provide crucial intermediate evidence for multi-step reasoning, enabling accurate trace generation under spatial constraints at the start, end, and along the path.

Instead of predicting 3D (x,y,z) coordinates directly, we adopt a decoupled (u,v,d) formulation that is trivially convertible to 3D via camera intrinsics. This circumvents the need for VLMs to implicitly learn camera geometry, thereby simplifying training and improving accuracy. Moreover, (u,v,d) points project into lower-dimensional spaces easily: omitting d yields 2D visual traces, and retaining only start/end points yields 3D or 2D spatial referring data. This formulation enhances data reusability, aligns seamlessly with existing 2D datasets [24,41,63] for co-training, thus bolstering multi-task learning performance.

Our spatial trace, combined with downstream motion planning for robotic control, offers the following advantages: (1) Compared to end-to-end VLAs [6, 8,41,44,45,51] that directly predict 6D dense actions, our approach can achieve superior zero-shot generalization without task-specific training, enabled by our generalizable formulation and less-constrained 3D space of this trace. (2) As an intermediate representation that delegates spatial reasoning and metric perception from instructions to spatial trace, this trace can handle more spatially and metrically constrained tasks compared to direct VLAs. (3) As a sequence of keypoints, spatial trace captures the intermediate execution process more effectively than start or end keypoint-based planning methods.

#### 3.2 RoboTracer

VLM Architecture. Spatial tracing requires a metric-grounded 3D understanding for referring and measuring, yet simply fine-tuning existing VLMs encounters two hurdles: (1) insufficient absolute-scale supervision, especially with RGB-only data; (2) underuse of absolute-scale geometric cues (e.g., camera intrinsics, depth) to enhance precision. In Fig. 2, we address these issues by introducing a scale decoder and a universal spatial encoder, each aligned with the LLM via projectors, akin to the existing RGB encoder. The scale decoder maps the <SCALE> token embedding into a numeric factor, linking scale-invariant representations to absolute metric scales. Rather than classification loss, we use regression loss to supervise it to heighten real-world 3D scale awareness. Moreover, we find that leveraging strong priors from a powerful feed-forward metric 3D geometry model [35] greatly enhances spatial and scale understanding. Building on this model, our universal spatial encoder flexibly integrates optional geometric cues (e.g., camera intrinsics, poses, depth) to refine spatial representations as more geometry becomes available. This modular design enables: (1) Flexible

###### Stage1—SFT

Stage2—RFT

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

/

<SCALE>

Metric Scale Factor

RGB Input

Opt. Geometric Input

Task Instruction

Task Instruction

Scale Decoder

[Figure 12]

2.307

[Figure 13]

[Figure 14]

Water flowers from left to right with watering

5

[Figure 15]

can hovering 1-5 cm above each flower.

[Figure 16]

Water flowers from left to right with watering can hovering 1-5 cm

4

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

/

/

/

3

[Figure 25]

[Figure 26]

[Figure 27]

RGB Projector

Vision

above each flower.

2

[Figure 28]

Encoder LLM

[Figure 29]

1

[Figure 30]

[Figure 31]

[Figure 32]

RGB Input

[Figure 33]

RoboTracer

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

/

/

[Figure 41]

Spatial Trace Visualization

[Figure 42]

[Figure 43]

[Figure 44]

Universal Spatial Encoder

Spatial Projector

[Figure 45]

[Figure 46]

Answer

Answer

1

(176,788,0.945) (278,518,1.005) (420,445,1.045) (568,165,1.017) (728,122,0.978)

[Figure 47]

<think>Step1…Step2...Step3…Step4…Step5…Step6…Step7…</think> <answer> (176,788,0.945), (278,518,1.005), (420,445,1.045),

2 3

1 2 3 4 5

[Figure 48]

(568,165,1.017), (728,122,0.978) </answer>

4 5

RGB w/ Opt. Geo. Input

Frozen Trainable Step1 in SFT/Step2 in SFT

###### Reasoning Process

5

Step1: [Scale] [Scene] 2.406

Step3: [Measuring] [Watering can] 0.104m

Step5: [Measuring] [The first flower from left] 0.195m

Step7: [Measuring] [The second flower from left] 0.403m

4

[Figure 49]

Step2: [Referring] [Watering can] (179, 789, 0.941)

3

Step4: [Referring] [The first flower from left] (454, 723, 0.965)

Step6: [Referring] [The second flower from left] (729, 624, 0.972)

2

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

0.403m

1

0.195m

0.104m

Step2 Step3 Step4 Step5 Step6 Step7 Output

- Fig. 2: Overview of RoboTracer. RoboTracer can process RGB images and task instructions, while flexibly integrating various geometric configurations when available to improve spatial precision (enabled by the integrated universal spatial encoder) with real-world scale awareness (powered by the scale decoder with regression loss). After SFT, metric-sensitive rewards in RFT further supervise the key perceptual steps in the trace and offer crucial intermediate evidence for accurate spatial trace generation.

training, leveraging diverse scale-aware annotations in datasets via flexible input augmentation to enrich spatial learning; (2) Geometry-adaptive inference, integrating available geometric cues without retraining or architectural changes, and more cues for better performance. See Supp. D.1 for details.

Supervised Fine-tuning. Since general VLMs’ 2D-only pretraining limits 3D metric-grounded understanding, we propose a two-step SFT. (1) Metric Alignment. In Fig. 2, we align the spatial encoder and scale decoder with LLM by using geometric annotations (e.g., depth, scale) from the TraceSpatial dataset (see Sec. 3.3). During this stage, only the projector and scale decoder are updated. (2) Metric Enhancement. We freeze the spatial encoder and finetune all other components on TraceSpatial and additional instruction-following datasets [49,50,96]. Crucially, we train on both RGB-only and RGB+X inputs, where X indicates arbitrary combinations of geometric annotations. This preserves image encoder’s general VQA ability while flexibly accommodating diverse geometric configurations without retraining during inference. The SFT loss is defined as:

L = Lntp + 0.1∥log(ˆs) − stopgrad(log(s∗)∥22 (1)

where Lntp is next-token prediction loss, log(ˆs) is predicted scale in logarithmic space, s∗ is ground-truth scale. Moreover, TraceSpatial contains multi-step data with reasoning processes, providing a “cold start” for the subsequent RFT stage. Please check Supp. D.2 for details about SFT training configuration.

Reinforcement Fine-tuning. We use RFT after SFT to improve compositional metric-grounded reasoning using GRPO [71] with multi-step reasoning data from TraceSpatial. We first define outcome-based rewards: (1) Out-

###### 2D Web image Pipeline

###### Dataset Statistics

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

i

t

o

u

n

b

i

r

- s

- t

###### Metric-agnostic QA

###### Detection & Seg

i

[Figure 62]

[Figure 63]

D

e

l

I

a

- n

[Figure 64]

d

[Figure 65]

- o

[Figure 66]

[Figure 67]

c

S

[Figure 68]

o

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

r

c

(a). Metric Scale

[Figure 73]

3D Spatial Referring 3D Spatial Measuring

i

Depth Estimation

r

t

Distribution

e

M

Q: Please point the 3D position of the shipping containers closest to the camera.

Q: Please measure the height of the cargo ship at left near the port terminal scene.

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

Camera Calibration

O

[Figure 78]

u

[Figure 79]

t

[Figure 80]

d

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

o

[Figure 85]

o

LLM & VLM

[Figure 86]

###### A: (898,603,8.586)

A: About 43 meters.

r

[Figure 87]

T

- o

[Figure 88]

- p

- a

[Figure 89]

- b

[Figure 90]

l

t

e

[Figure 91]

[Figure 92]

[Figure 93]

Filtered RGB Image

Foundation Model as Tools Data Type & Example Data Entry

3D Scene Graph

3D Scanning Data Pipeline

o

i

t

n

u

b

i

r

- s

- t

[Figure 94]

[Figure 95]

i

D

Metric-agnostic QA

[Figure 96]

S

c

D

a

a

[Figure 97]

[Figure 98]

3

[Figure 99]

[Figure 100]

|[Figure 101]<br><br>2 D<br><br>[Figure 102]<br><br>[Figure 103]<br><br>W<br><br>[Figure 104]<br><br>e<br><br>[Figure 105]<br><br>b<br><br>[Figure 106]<br><br>I m<br><br>[Figure 107]<br><br>[Figure 108]<br><br>a<br><br>[Figure 109]<br><br>g<br><br>[Figure 110]<br><br>e<br><br>[Figure 111]<br><br>S<br><br>o<br><br>u<br><br>r<br><br>c<br><br>e<br><br>D<br><br>a<br><br>t<br><br>[Figure 112]<br><br>CA-1M<br><br>[Figure 113]<br><br>[Figure 114]<br><br>[Figure 115]<br><br>[Figure 116]<br><br>[Figure 117]<br><br>[Figure 118]<br><br>n<br><br>[Figure 119]<br><br>n<br><br>[Figure 120]<br><br>i<br><br>[Figure 121]<br><br>n<br><br>[Figure 122]<br><br>g<br><br>[Figure 123]<br><br>D<br><br>[Figure 124]<br><br>a<br><br>[Figure 125]<br><br>t a<br><br>[Figure 126]<br><br>[Figure 127]<br><br>ScanNet<br><br>DROID<br><br>[Figure 128]<br><br>O p<br><br>[Figure 129]<br><br>[Figure 130]<br><br>e<br><br>[Figure 131]<br><br>n<br><br>[Figure 132]<br><br>I m<br><br>[Figure 133]<br><br>[Figure 134]<br><br>a<br><br>[Figure 135]<br><br>g<br><br>[Figure 136]<br><br>e<br><br>Distribution| |
|---|---|
|[Figure 137]<br><br>V<br><br>i|[Figure 138]<br><br>d<br><br>e<br><br><br>o<br><br>|

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

- (b). Source Data

Distribution

- (c). Question Types

3D Spatial Measuring

3D Spatial Referring

[Figure 143]

Object-centric Spatial Tracing Q: Pick up the leftmost object on the second floor and place it 0.25 meters to right of yellow object.

A: (327,531,1.853) -> (487,193,1.764)

RoboTwin AgiBotWorld

Filtered Image 3D Occupancy Map

Object-centric Spatial Trace Data Type & Example Data Entry

Manipulation Video Pipeline

u

t

i

o

b

i

n

r

- s

- t

[Figure 144]

i

[Figure 145]

[Figure 146]

D

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

s

Error Camera Info.

e

###### 3D Spatial Referring 3D Spatial Measuring

p

|[Figure 151]<br><br>Q<br><br>u<br><br>e<br><br>s<br><br>t<br><br><br>i<br><br>o<br><br>n<br><br>T<br><br>y<br><br>[Figure 152]<br><br>Metric-agnostic VQA<br><br>[Figure 153]<br><br>[Figure 154]<br><br>[Figure 155]<br><br>Metric-grounded VQA<br><br>[Figure 156]<br><br>[Figure 157]<br><br>|
|---|

[Figure 158]

Raw Video Episodes

Distribution

[Figure 159]

[Figure 160]

Object-centric Spatial Tracing

[Figure 161]

End-effector-centric Spatial Tracing

[Figure 162]

Q: Pick up the can closest to the camera and move it in the empty space on the upper shelf.

3D Spatial 3D Spatial Referring Measuring

Spatial Tracing

A: (666,371,0.607) -> (209,200,0.725) Meaningless Task

Unclear Trace

Filtered Video End-effector-centric Spatial Trace

Data Type & Example Data Entry

- Fig. 3: TraceSpatial has 4.5M data samples from 2D, 3D, and video sources, covering outdoor, indoor, and tabletop scenes. It contains metric-agnostic/-grounded QA pairs for 3D spatial referring, measuring, tracing with a detailed reasoning process.

come Format Reward (ROF) for structured outputs; (2) Point Reward (RP) for start/end point consistency between the predicted trajectory (p1,pT) in τˆ = {pˆt}Tt=1 vs. τ. Formally, RP = 12 f(p1,pˆ1)+f(pT,pˆT) ,f(p,p′) = max 0, 1− ∥p−p′∥22 . (3) Trace Reward (RT) for trajectory-level alignment using a distance metric d(τ,τˆ): RT = max 0, 1 − d(τ,τˆ) . All (u,v,d) values are normalized to [0,1], with depth scaled by the scene’s maximum depth. However, the outcomebased rewards described above are metric-agnostic and provide no supervision over the key perceptual objects involved in trace generation (e.g., 3D spatial measuring and referring). To address this, we introduce metric-sensitive process rewards that leverage key-step perception annotations from TraceSpatial: (1) Process Format Reward (RPF), enforcing the format “[Perception Type] [Target Object]:”; (2) Accuracy Reward (RAcc), which applies to steps included in the key-step perception annotations. For each relevant step, we measure the prediction error using a specific metric, according to the perception type (e.g., L1 distance for referring). Notably, this design is order-invariant, allowing flexible step ordering. The final reward sums both outcome-/process-based terms, with process-based terms scaled by 0.25. Fig. 2 shows that RFT-trained model generalizes well to 7-step spatial tracing, progressively resolving complex spatial relations and producing accurate trace. Please check Supp. D.3 for more details.

#### 3.3 TraceSpatial Dataset

Overview. Key features are: (1) Rich Diversity. TraceSpatial spans outdoor, indoor, and tabletop scenes (Fig. 3 (a)) and includes both object-centric and end-effector-centric spatial traces; the latter captures gripper motions across 3 single-arm and dual-arm robot configurations, fostering model generalization in open-world scenarios. (2) Multi-Dimensionality. Beyond metric-agnostic

spatial concepts and relations, the dataset includes 48.2% metric-grouded QA (Fig. 3 (b)). These samples cover 3D spatial measuring and referring, and support multi-step spatial tracing by providing detailed annotations of reasoning process, addressing limitations in existing datasets [33, 99]. (3) Large Scale. With 4.5M samples and 30M QA pairs, TraceSpatial is the largest dataset for 3D spatial reasoning, supporting bottom-up spatial tracing learning. (4) FineGrained. Hierarchical object captions, from coarse categories (e.g., “flower”) to fine-grained spatial referents (e.g., “the first flower from the left”), enable 3D spatial measuring, referring, and tracing in cluttered scenes. Absolute-scale geometry (e.g., intrinsics, depth) further enriches spatial learning and flexible input augmentation. (5) High Quality. Rigorous filtering preserves spatial relevance. From 1.7M OpenImages [39], 466k images remain; from CA-1M [40](2M) and ScanNet [22](190k), 100k and 12k frames are retained based on text-identifiable 3D boxes, respectively. From DROID [37](116k) and AgiBot [21](167k) videos, 20k and 59k episodes are preserved after verifying valid camera poses, coherent tasks, and clean trajectories, respectively. We also manually inspect a subset to verify data quality and iteratively refined the dataset. (6) Easy Scalability. Our pipeline scalably integrates 2D images, 3D scans with bounding boxes, and manipulation videos. Even faced with RGB-only datasets in new domains, we can still scale up easily and acquire multi-step reasoning annotations automatically.

Data Recipe. In Fig. 3, we propose a data pipeline that progressively integrates

- 2D/3D/video sources for general VLMs to perform 3D spatial referring/measuring/tracing. (1) 2D Web Images aim to provide basic spatial concepts and broad-scale perception across indoor and outdoor scenes. We filter 1.7M images from OpenImage [39] down to 466K and employ VLM [5] with hierarchical region-captioning to produce fine-grained spatial references, surpassing previous approaches [23,74]. Object captions serve as nodes for 3D scene graphs, where edges depict spatial relations inferred via object detection, depth, and camera estimation. Using template-/LLM-based methods, we generate metric-agnostic and 3D spatial QA grounded in these captions. (2) 3D Scanning Datasets want to arm the model with a focused metric-grounded spatial reasoning of indoor scenes. We thus leverage the richly annotated CA-1M [40] and ScanNet [22]. After fine-grained filtering, we construct 3D scene graphs with more diverse spatial relations, enabled by precise 3D bounding boxes compared to 2D approaches. Moreover, we generate 3D occupancy maps that encode positions, orientations, and metric distances (e.g., “35cm right of the toy”) for accurate object-centric spatial trace generation. (3) Manipulation Videos provide spatial traces aligned with the embodied manipulation in tabletop settings. While
- 3D scans enable object-centric tracing, they lack physically plausible manipulations. Hence, we curate real and simulated [14] tabletop videos (from 167k to 59k for AgiBot [21], and from 116k to 24k for DROID [37]) with calibrated cameras, accurate task execution, and clear trajectories. We further leverage VLM [5] to decompose these tasks into subgoals, enabling precise multi-step spatial tracing for single-/dual-arm across 3 robots. Notably, as 3D datasets and simulation videos are all-seeing, we construct multi-step, metric-grounded spatial tracing

###### Table 1: Performance on spatial understanding benchmarks. Rel., Dep., Dist. denote relation, depth, distance. Top-1/-2 success rate are indicated by bold/underlined text.

CV-Bench [85] BLINKval [28]

Method Input

RoboSpatial [74] EmbSpatial [25]

2D-Rel. 3D-Dep. 3D-Dist. 2D-Rel. 3D-Dep.

GPT-4o [2] RGB 84.62 86.50 83.33 82.52 78.23 77.20 63.38 Gemini-2.5-Pro [20] RGB 93.54 91.00 90.67 91.61 87.90 77.24 76.67

NVILA-2B [56] RGB 70.15 79.67 60.00 67.83 62.10 51.79 47.34 NVILA-8B [56] RGB 91.54 91.83 90.67 76.92 76.61 59.35 67.72 Qwen-3-VL-4B [83] RGB 92.31 94.67 87.50 87.71 85.48 79.67 77.01 Qwen-3-VL-8B [83] RGB 93.85 94.50 90.33 87.41 85.48 77.24 77.86

Molmo 7B-D [24] RGB 87.69 66.00 61.83 59.44 77.42 58.60 58.74 SpaceVLM-13B [11] RGB 63.69 66.83 70.17 72.73 62.90 61.00 49.40 RoboBrain 2.0-7B [80] RGB 96.00 94.83 90.00 79.72 85.48 74.80 74.78 SpatialBot-3B [9] RGB-D 69.38 77.33 60.83 67.83 67.74 72.36 50.66

RoboTracer-2B-SFT RGB 96.62 96.00 89.83 83.22 91.94 82.93 70.66 RoboTracer-8B-SFT RGB 97.08 97.17 93.50 91.61 92.74 83.74 81.75

###### Table 2: Performance on spatial measuring benchmarks. S.N., S.E., R.E. denote Plus, Scannet, Scale Estimation, Refer Estimation. Top-1/-2 success rate (%) are indicated by bold/underlined text.

Table 3: Performance on 2D spatial referring benchmarks. W.2.P., RoboS., RefS.L. and RefS.-P. denote Where2Place [97], RoboSpatial [74], and Location and Placement parts of RefSpatial-Bench [112], respectively. Top-1/-2 success rate (%) are indicated by bold/underlined text.

Q-spatial [46] MSMU [13]

Method Input

Plus S.N. S.E. R.E. GPT-4o [2] RGB 31.68 37.06 3.86 2.09 Gemini-2.5-Pro [20] RGB 56.44 70.00 64.86 48.42 NVILA-2B [56] RGB 36.90 40.59 40.15 37.37 NVILA-8B [56] RGB 46.87 44.71 33.98 38.95 Qwen-3-VL-4B [83] RGB 53.47 70.00 63.71 42.11 Qwen-3-VL-8B [83] RGB 29.70 56.47 63.32 52.63 Molmo 7B-D [24] RGB 51.49 63.53 59.85 43.68 SpaceVLM-13B [11] RGB 25.74 45.29 32.42 31.58 RoboBrain 2.0-7B RGB 44.55 55.88 69.11 48.42 SpatialBot-3B [9] RGB-D 20.79 29.41 16.60 22.62 RoboTracer-2B-SFT RGB 68.32 70.59 78.38 60.00 RoboTracer-8B-SFT RGB 73.27 78.82 83.01 70.00

Model W.2.P. RoboS. RefS.-L. RefS.-P.

Gemini-2.5-Pro [20] 61.90 40.20 46.96 24.21 Qwen3-VL-4B 66.0 63.11 43.00 54.55 Qwen3-VL-8B [83] 64.00 61.48 51.00 42.00

SpaceLLaVA [11] 11.8 16.0 5.82 4.31 RoboPoint [97] 46.80 41.30 22.87 9.27 Molmo-7B [24] 45.00 38.00 21.91 12.85 Molmo-72B [24] 63.80 40.90 45.77 14.74 RoboBrain 2.0-7B [80] 63.59 54.87 36.00 29.00

RoboTracer-2B-SFT 63.00 62.30 49.00 45.00 RoboTracer-8B-SFT 69.00 66.40 55.00 53.00

data, under the assumption that the generated code reflects optimal reasoning, with each line translated into textual form and intermediate results filled into structured formats (e.g., coordinates, distances). See Supp. B for details.

### 4 Experiments

Model Configuration. We adopt NVILA [56] (2B/8B) as base model and apply SFT. Due to computational limits, we only perform RFT on 2B model. Since the model can accept arbitrary geometric inputs, it defaults to using only images when input types are unspecified. Please check Supp. D for details.

Robotic Evaluation Configuration. We ensure physical feasibility and smooth-

ness by motion planning via adherence to the spatial trace, collision avoidance, and joint limits with a smoothness cost. We also correct VLM-generated spatial traces before use to satisfy physical constraints. Please see Supp. E.8 for details.

#### 4.1 Spatial Understanding and Measuring

We evaluate our SFT model on public spatial understanding benchmarks, including CV-Bench [85], BLINK [28] (validation set), RoboSpatial [74] (configuration

- Table 4: Performance on the 2D visual trace benchmarks. RMSE denotes Root Mean Square Error. Top-1/-2 scores are shown by bold/underlined text.

ShareRobot-Bench [33] VABench-V [98]

Model

Discrete Fréchet Hausdorff

Discrete Fréchet Hausdorff Distance ↓ Distance ↓ Distance ↓ Distance ↓ RMSE ↓

RMSE ↓

Qwen3-VL-4B [83] 0.3808 0.3294 0.2204 0.2792 0.2528 0.2037 Qwen3-VL-8B [83] 0.3922 0.3411 0.2328 0.2741 0.2549 0.2021

MolmoAct [41] 0.7764 0.7764 0.6771 0.8136 0.8136 0.6877 HAMSTER [43] 0.4365 0.3919 0.3554 0.2124 0.2045 0.1825 RoboBrain 2.0-3B [80] 0.1974 0.1853 0.1380 0.3642 0.2820 0.2445 RoboBrain 2.0-7B [80] 0.1669 0.1575 0.1250 0.3289 0.2604 0.2237 Embodied-R1-3B [100] 0.3426 0.3002 0.2388 0.3028 0.2588 0.2129

RoboTracer-2B-SFT 0.1605 0.1544 0.1114 0.1664 0.1551 0.1237 RoboTracer-8B-SFT 0.1449 0.1384 0.0966 0.1494 0.1367 0.1065

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

Reasoning Step = 5

Reasoning Step = 4

Gemini-

Gemini-

2.5-Pro Qwen3VL-4B

2.5-Pro Qwen3VL-4B

Qwen3VL-8B

Qwen3-

VL-8B

Ours

Ours

Ours w/

Ours w/ Geo. Input

View 1

View 2

View 1

View 2

Geo. Input

Pick up the top orange fruit on the dark blue plate, and move it to the orange bowl on the top right. Place paper towel on white table to the top of the highest brown wooden shelf on the right.

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

Reasoning Step = 5

Reasoning Step = 5

Gemini2.5-Pro Qwen3VL-4B

Gemini-

2.5-Pro

Qwen3-

VL-4B

Qwen3-

Qwen3VL-8B

VL-8B

Ours

Ours

Ours w/ Geo. Input

Ours w/ Geo. Input

View 1

View 2 View 2

View 1

Pick up pale blue pillow on the sofa which is the second pale blue pillow from the right, and move it

Pick up the blue object which is on the window sill, and move it to a spot which is on the sink's edge

to the top of the wooden stool on the left.

and closest to the right wall.

- Fig. 4: TraceSpatial-Bench Results. White masks denote ground-truth 3D starting region; pink 3D boxes mark correct end regions. Despite similar 2D projections (left views of each case), our model yields more accurate spatial traces than strong general VLMs, which often produce floating or colliding traces due to inaccurate depth estimation. Leveraging richer geometric cues further improves performance.

subset), and EmbSpatial [25]. We also assess spatial measuring performance on Q-Spatial [46] and MSMU [13]. Please refer to Supp. E.2 and E.3 for details.

SFT learns strong spatial understanding and metric measuring. In Tab. 1 and Tab. 2, RoboTracer-8B-SFT trained solely on TraceSpatial surpasses all baselines with an average success rate of 85.7%, even outperforming Gemini2.5-Pro by 8.58% and base model NVILA-8B by 20.3%. Notably, we find that its improvements are more pronounced in 3D-related and measurement tasks compared to 2D tasks (23.6% vs.14.7%), showing that our architectural design and curated dataset during SFT enhance the model’s 3D spatial and scale awareness.

#### 4.2 2D Spatial Referring and Visual Trace

We evaluate 2D spatial referring (Where2Place [97], RoboSpatial [74], RefSpatialBench [112]) and visual trace benchmarks (ShareRobot-Bench [33] (end-effectorcentric), VABench-V [98]) (object-centric). See Supp. E.4 and ?? for details.

- Table 5: Performance on TraceSpatial-Bench. R., I., D. denote RGB image, intrinsics, absolute depth. N. means adding noise to depth input. We report success rate (SR).

Process 2D Start 2D End 3D Start 3D End Overall

Model Input

Reward SR (%) ↑SR (%) ↑ SR (%) ↑SR (%) ↑ SR (%) ↑

Gemini-2.5-Pro [20] RGB - 31 33 9 16 3 Qwen3-VL-4B [83] RGB - 64 16 43 24 4 Qwen3-VL-8B [83] RGB - 60 21 47 22 6

MolmoAct [41] RGB - 4 15 - - Magma [94] RGB - 28 17 - - RoboBrain 2.0-7B [80] RGB - 46 23 - - Embodied-R1-3B [100] RGB - 63 13 - - -

RoboRefer-2B (finetuned) R.D. - 48 43 60 51 28 LEO-7B (finetuned) PointCloud - 46 39 58 48 27

RoboTracer-2B-SFT RGB - 56 44 63 52 31 RoboTracer-2B-SFT R.I.D. - 59 45 75 56 38

RoboTracer-RFT RGB ✗ 61 46 64 54 33 RoboTracer-RFT RGB ✓ 63 47 73 55 39 RoboTracer-RFT R.I. ✓ 64 48 73 56 40 RoboTracer-RFT (2D pipeline) R.I.D. ✓ 66 51 75 58 42 RoboTracer-RFT R.I.D. w/ N. ✓ 68 53 76 59 44 RoboTracer-RFT R.I.D. ✓ 69 53 78 61 45

Decoupled formulation makes multi-task learning easy. Our RoboTracer achieves best performance on both 2D spatial referring (see Tab. 3) and visual trace (see Tab. 4) benchmarks. While TraceSpatial focuses on 3D metricgrounded data without explicitly targeting 2D tasks, our decoupled point formulation enables seamless projection of TraceSpatial into the data formats required for such 2D tasks (see Sec. 3.1). Moreover, in Tab. 9 (ID G & H), modeling (u,v,d) surpasses direct 3D (x,y,z) modeling. We attribute this to (1) our dataset reuse via dimensionality reduction at various levels (e.g., 3D to 2D, point sequences to individual points), (2) stronger alignment with existing 2D datasets for co-training, thereby improving multi-task learning performance.

#### 4.3 Multi-Step Spatial Tracing

To evaluate complex multi-step, metric-grounded spatial tracing, we introduce TraceSpatial-Bench, a challenging real-world benchmark focusing on cluttered indoor/tabletop scenes. The dataset comprises 100 images with precise geometric annotations (e.g., camera geometry, absolute depth, and 3D occupancy). Each sample requires 3˘8 reasoning steps, specifying both the manipulated object’s start mask and its end 3D box. We assess performance in 2D and 3D; a trajectory succeeds if its start and end positions are correct and all intermediate paths remain collision-free. See Supp. E.6 for details. We present our analyses below.

RFT enables better multi-step metric-guided reasoning. In Tab. 5, RoboTracer-RFT outperforms all baselines on 3D metrics, exceeding Gemini-2.5-Pro by 36%. We find that while these VLMs perform well on 2D referring and tracing, they fall short in 3D spatial tracing due to their limited metric depth understanding, often producing traces that float or collide with objects. Moreover, RoboTracer-RFT also outperforms the baselines (e.g., LEO, RoboRefer) with 3D

- Table 6: Performance on general benchmarks. Top-1 scores are shown by bold text. Model MMEtest MMBenchdev OK-VQA POPE NVILA-2B [56] 1547 78.63 64.9 81.96 RoboTracer-2B-SFT 1751 77.62 65.22 82.52

- Table 7: Performance on RoboTwin 2.0 [14] hard tasks. We report the success rate (%) compared to end-to-end VLA and VLM-based models. Gray rows indicate unseen tasks not present in TraceSpatial. RoboTracer demonstrates strong generalization.

End-to-End Policy Vision-Language Model Ours ACT DP DP3 RDT π0 Qwen3-VL-8B Gemini-2.5-Pro RoboTracer-2B

Task

Place A2B Left 0 0 2 1 1 0 2 84 Move Playingcard Away 0 0 3 11 22 0 5 94 Click Alarmclock 4 5 14 12 11 0 0 79 Place Burger Fries 0 0 18 27 4 0 0 99 · · ·

Place Container Plate 1 0 1 17 45 0 0 52 Stack Blocks Two 0 0 0 2 1 0 0 33 Place Empty Cup 0 0 1 7 11 0 0 85 Place Object Stand 0 0 0 5 11 0 0 38

Seen Avg. Success Rate 0.9 0.4 3.7 6.3 6.6 0 0.7 75.4 Unseen Avg. Success Rate 0.1 0 0.6 4.7 11.6 0 0.3 44.4 Total Avg. Success Rate 0.6 0.2 2.5 5.7 8.6 0 0.5 64.0

input after being finetuned by TraceSpatial. This shows that RoboTracer-RFT leverages learned 3D spatial knowledge and compositional reasoning to generate more accurate spatial traces. Fig. 4 further shows complex spatial tracing cases from TraceSpatial-Bench with model comparisons.

Multi-step processes enable better reasoning with robustness. In Tab. 5 and Tab. 9 (ID D), we find that metric-grounded reasoning relies more on multistep process RFT training than high-quality external 3D inputs (e.g., depth map) from spatial encoder equipped in SFT. Specifically, RFT without spatial encoder, initialized from SFT, slightly underperforms RFT with spatial encoder but largely exceeds SFT. Moreover, RFT model remains robust under noisy geometric input, showing its robustness without the high-quality external features. Accurate geometry refines metric reasoning. In Tab. 5, using more precise geometric cues greatly improves performance, yield up to 6% absolute gain. This suggests that explicit, readily available geometry in embodied settings can further improve metric reasoning, rather than relying solely on implicit learning of VLM to understand them via RGB-only input. Moreover, our model supports more geometry input to enhance itself without retraining, broadening its applicability. Fig. 4 shows the generated traces with different geometric inputs.

#### 4.4 Public vision-language Benchmarks

Joint training preserves common-sense knowledge. In Tab. 6, our model performs on par with or slightly surpasses the base model. This benefit stems from our use of TraceSpatial for RGB and RGB+X (See Sec. 3.2) joint training, enriched by general visual instruction datasets.

- Table 8: Real-world robot evaluation of spatially and metrically constrained tasks. MolmoAct is the end-to-end VLA baseline. RoboRefer is the keypoint-based baseline.

Success Rate(%) ↑

Spatially and Metrically Constrained Tasks

MolmoAct [41] RoboRefer [112] MolmoAct (finetuned) Ours

|Pick the rightmost hamburger and place it on the keyboard in front of the laptop without collisions<br><br>|0.00 0.00<br><br>|40.00|60.00|
|---|---|---|---|
|Water flowers from right to left with watering can hovering 1-5 cm above each flower.|0.00 0.00<br><br>|10.00<br><br>|30.00|

###### Task with Spatial Tracing: Pick the rightmost hamburger and place it on the keyboard in front of the laptop, avoiding collisions with nearby objects.

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

hamburger is moved, target changed

- Fig. 5: Real-World Evaluation. The blue trace denotes predicted spatial trace in 2D, and the blue dot marks current target. RoboTracer can generate spatial traces whose start and end points all satisfy spatial constraints in cluttered and dynamic scenes.

#### 4.5 Simulator and Real-world Evaluation

Spatial trace offers superior zero-shot generalization. In Tab. 7, we evaluate our model on 19 Robotwin 2.0 [14] hard tasks involving clustered scenes. Among them, 12 tasks are seen and 7 are unseen in the TraceSpatial dataset. Unlike task-specific end-to-end VLA baselines trained on each task, our model, without any task-specific training, outperforms the best baseline by 32.8% in unseen tasks, showing strong generalization. Moreover, current powerful VLMs struggle to generate spatial traces, highlighting the importance of our model design and dataset contribution. Please check Supp. E.7 for details.

Spatial tracing is critical for spatially and metrically constrained tasks. In Tab. 8, only our method can handle long-horizon tasks requiring multi-step, metric-grounded spatial tracing in cluttered, dynamic scenes compared to VLAbased (e.g., MolmoAct [41]) and keypoint-based (e.g., RoboRefer [112]) method. These tasks demand precise identification and placement of objects under evolving spatial constraints, while continuously avoiding collisions. In Fig. 5, integrating RoboTracer with motion planning and Code-as-Monitor [115] enables rapid updates at 1.5 Hz. Thus, when the rightmost hamburger is moved, the robot adapts by grasping the newly rightmost one, reaching over the large doll and the computer screen to place it on the keyboard. Notably, this embodiment-agnostic spatial trace can also be executed by G1 humanoid, enabling even more complex, long-horizon tasks such as flower-watering (Fig. 1). See Supp. E.8 for details.

RoboTracer serves as a practical tool to generate action data. RoboTracer equipped with motion planning and Code-as-Monitor [115] can generate action data for spatially and metrically tasks. In Tab. 8, after finetuning with successful episode and employing generated spatial traces as auxiliary supervision, we improve MolmoAct’s 3D reasoning and spatial performance (success rate raised from 0% to 25%), showing RoboTracer’s application for data generation.

- Table 9: Ablation Study on data recipe, spatial encoder, scale supervision with regression (Reg.) and next-token prediction (N.T.P) loss, point formulation in QSpatial, ShareRobotBench(S.R.B.) and TraceSpatial-Bench(T.S.B). We report success rate (SR)/Discrete Fréchet Distance (DFD) of 2B model.

- ID

Data Recipe Spatial Scale Point Model Q-Spatial S.R.B T.S.B.

| |2D 3D Video|Encoder<br><br>|Supervision|Formulation<br><br>|Type|SR(%) ↑ DFD ↓ SR (%) ↑|
|---|---|---|---|---|---|---|
|A<br><br>B<br><br>C<br><br><br>|✗ ✓ ✓ ✓ ✗ ✓ ✓ ✓ ✗<br><br>|✓ ✓ ✓|Reg. Reg. Reg.<br><br>|(u, v, d) (u, v, d) (u, v, d)|SFT SFT SFT<br><br>|51.49 0.1723 27 33.52 0.1809 19 63.40 0.4376 24<br><br>|

- D ✓ ✓ ✓ ✗ Reg. (u, v, d) RFT - - 36

- E ✓ ✓ ✓ ✓ ✗ (u, v, d) SFT 53.47 0.1693 24

- F ✓ ✓ ✓ ✓ N.T.P. (u, v, d) SFT 57.43 0.1671 26

- G ✓ ✓ ✓ ✓ Reg. (x, y, z) SFT 68.32 0.2426 30

- H ✓ ✓ ✓ ✓ Reg. (u, v, d) SFT 69.61 0.1605 31

- I ✓ ✓ ✓ ✓ Reg. (u, v, d) RFT - - 39

- 4.6 Ablation Study

Data recipe is crucial for SFT training. Tab. 9 shows that combining

- 2D, 3D, and video data yields optimal performance. 2D/3D data offer critical scale supervision for both indoor and outdoor scenes; their absence degrades metric-grounded Q-spatial [46] accuracy. Video data enables gripper motion learning across diverse robot configurations, improving end-effector-centric performance on ShareRobotBench [33]. Moreover, 3D data and simulated videos increase spatial instruction diversity and metric-grounded reasoning, crucial for TraceSpatial-Bench. This three-way data synergy is thus key to SFT training.

Dataset construction pipeline has scalability. Our dataset pipeline scales easily to RGB-only datasets in new domains, enabling automatic multi-step reasoning annotations. We re-annotate TraceSpatial via 2D web image pipeline and a gripper detector [63] without ground-truth metadata. Tab. 5 shows comparable performance of newly RFT-trained model, demonstrating its scalability.

Universal spatial encoder improves 3D reasoning. We fine-tune NVILA2B [56] on TraceSpatial without universal spatial encoder, followed by RFT. In Tab. 9 (ID D & I), we find that the spatial encoder enhances metric-grounded reasoning and greatly improves multi-step spatial tracing. This is mainly due to: (1) encoder’s prior 3D knowledge that facilitates implicit 3D learning, (2) cumulative reasoning across steps, amplifying the utility of spatial cues.

Scale regression supervision boosts metric awareness. In Tab. 9 (ID E & F & H), we compare supervising the metric scale factor using regression loss, next-token prediction loss, and no supervision. The regression-based model performs best. While textual (next-token) supervision of offers slight gains, it remains inferior to regression. We observe that: (1) pure next-token prediction supervision—whether or not the scale factor is included as output—demands extensive data to improve scale awareness as shown in recent work [10]; (2) explicitly predicting the scale factor, particularly for RGB-only inputs (as in our

RGB/RGB+X mixed training), forces models to learn scale information without relying on auxiliary geometry, thereby bolstering capability.

Metric-sensitive reward advances the accuracy. In Tab. 5, integrating process rewards boosts overall success rate by 4% compared to purely outcomebased methods. Notably, using only these metric-agnostic outcome-based rewards yields smaller improvements on 3D metrics and overall success rates compared to the gains observed on 2D metrics, relative to SFT. This highlights that leveraging TraceSpatial’s key step annotations to supervise metric-grounded step-wise perception enables more accurate traces in complex spatial relations.

### 5 Conclusion

In this paper, we propose RoboTracer, a novel 3D-aware VLM that addresses spatial tracing via multi-step, metric-grounded reasoning on both 3D spatial referring and measuring. In detail, we empower the model to flexibly process arbitrary geometric inputs for precision, employ scale supervision to enhance scale awareness, and guide it with metric-sensitive rewards to improve its reasoning. We also present TraceSpatial, a large-scale well-designed dataset for SFT and RFT training, with TraceSpatial-Bench, a benchmark tailored to evaluate spatial tracing. Extensive experiments show the effectiveness of RoboTracer and highlight its potential for a broad range of robotic applications.

Limitation and Future Work. Despite the strong performance of our 3D spatial-trace prediction, we rely on motion planning to recover the 6D endeffector pose for robotic control, which is less effective for rotation-rich manipulation where orientation constraints are critical. Dense spatial traces may handle such tasks, which is a potential area for further research to explore.

### Acknowledgement

This work was supported by the National Natural Science Foundation of China (62132001, 62476011), the Beijing Natural Science Foundation (L252218, L252060), and the Fundamental Research Funds for the Central Universities.

## Supplementary Material of RoboTracer

The supplementary document is organized as follows:

- – Sec. A: More Discussions like Technical Details, Evaluation.
- – Sec. B: Implementation Details of TraceSpatial, including data filtering, collection, and QA generation.
- – Sec. C: Implementation details of TraceSpatial-Bench, including benchmark annotation, metrics, and statistics.
- – Sec. D: Implementation details of RoboTracer, including architecture and training details of each stage.
- – Sec. E: More Details on experimental settings, including VQA benchmarks, simulation, and real-world evaluation.
- – Sec. F: More Demonstrations of RoboTracer.
- – Sec. G: More Discussion on Limitations and Future Work.

### A More Discussions

TraceSpatial Benefit. In Tab. 11 (ID ABCD), finetuning the same base model on our dataset surpasses specialized models on SpatialRGPT-Bench/SAT, even without benchmark-specific tuning, showing the effectiveness of data.

Larger TraceSpatial-Bench Evaluation and Risk of Benchmark-Model Alignment. In Tab. 10, two checks rule out benchmark-model alignment. (1) Independent annotators relabeled all TraceSpatial-Bench samples, yet our model still surpasses the baselines. (2) Scaling the benchmark to 800 samples under the same relabeled protocol further increases our margin over Gemini-2.5-Pro.

More Real-world Evaluation. In Tab. 12, we add 3 real-robot tasks. Each task uses 20 trials with human-annotated success. MolmoAct is fine-tuned using 50 successful rollouts.

8B-RFT Results. In Tab. 10, RFT yields a larger gain on the 8B model than on the 2B model (12% vs. 8%), further showing both its effectiveness and scalability. More Details about Metric-grounded Outdoor Data. Outdoor data constitutes 30.23% of the metric-grounded data. Tab 12 shows that the outdoor part has a negligible effect on tabletop manipulation.

Lift 3D Comparison. Tab. 10 (ID BCE) compares ours (lift label to 3D) with two baselines (lift output to 3D). (1) This baseline predicts (u,v) via the first model and lift trace with depth d via the second model, both of which use the same data, model backbone, and training schedule as ours. (2) We replace the

###### Table 10: TraceSpatial-Bench’ results w/ only RGB as model input.

ID Model Input Lift 3D Training 100-Original 100-Relabeled 800-Relabeled

- A Gemini=2.5-Pro RGB - - 3 2 3.375

- B Two-stage baseline 1 RGB Model-lift SFT+RFT 31 30 34.125

- C Two-stage baseline 2 RGB Oracle-lift SFT+RFT 34 33 35.625

- D RoboTracer-2B RGB Native SFT 31 30 33.250

- E RoboTracer-2B RGB Native SFT+RFT 39 39 41.750

- F RoboTracer-8B RGB Native SFT 42 40 44.375

- G RoboTracer-8B RGB Native SFT+RFT 54 53 57.125

###### Table 11: Performance on VQA of SpatialRGPT-Bench and SAT.

ID Model Base Model Dataset SpatialRGPT-Bench SAT

- A SpatialRGPT-8B VILA-1.5-8B OSD 92.69 -

- B SAT-7B LLaVA-Video-7B SAT - 63.4

- C Our finetuned VILA-8B VILA-1.5-8B TraceSpatial 94.82 -

- D Our finetuned LLaVA-Video-7B LLaVA-Video-7B TraceSpatial - 82.00

- E RoboTracer-8B NVILA-8B TraceSpatial 97.41 86.67

second model with oracle-lift. Our model surpasses them, showing the superiority of 3D-native formulation. Overall, under same training data, lifting to (u,v,d)

- 3D label (39%) > direct output lifting with oracle-lift (34%) > lifting to (x,y,z) 3D label (30%).

Formulation Generality. Our (u,v,d) naturally supports navigation: (u,v) offers explicit 2D object referring or waypoints, coupled with SLAM for downstream navigation, as in recent work (e.g., RoboPoint [97]).

Compared to RoboRefer [112]. (1) Task setting: We address a more challenging 3D spatial tracing task: predicting a 3D trace rather than a 2D keypoint. We formulate tracing in (u,v,d) space, extending beyond RoboRefer’s 2D keypoint setting. (2) Metric features. 3D tracing requires explicit metric reasoning. Our framework incorporates metric awareness at every stage—input, output, supervision, and training—beyond RoboRefer’s qualitative spatial referring. (3) Temporal modeling. Our trace data is task-level and temporally structured. Using manipulation and simulation videos, we collect full-task-execution 3D keypoint sequences that capture intermediate dynamics and temporal consistency. RoboRefer’s single-image, single-point feature cannot support such 3D tracing. Stronger Evaluation of spatial reasoning. We add SpatialRGPT-Bench/SAT

results in Tab. 11 (ID ABE), where we still surpass specialized models.

### B TraceSpatial Details

In this section, we provide a detailed overview of the implementation procedures and representative data samples, highlighting the construction of the TraceSpatial dataset. The dataset is specifically designed to empower general VLMs with a step-by-step capacity to: (1) adapt to 3D spatial referring and measuring tasks, and (2) subsequently progress toward spatial tracing tasks in a bottom-up manner. To achieve this, we meticulously establish a multi-data-source genera-

###### Table 12: More evaluation. F. O. means Finetuned & Outdoor data.

Spatially and Metrically Constrained Tasks MolmoAct MolmoAct (F.) Ours (w/o O.) Ours Place the hot dog 0.1 m to the right of the leftmost peach. 0 40 55 55 Place the nearest blue object to the right of the tallest object,

0 30 50 50

5 cm above the bowl center and release it. Move the green block behind the tallest central obstacle and

0 15 30 35 5 cm to the left of the yellow block via a collision-free motion.

tion pipeline. In the following, we describe the three fundamental components of this pipeline in detail:

- – 2D Web Image (Supp. B.1): We present a 2D data pipeline comprising image filtering, pseudo-3D scene graph construction, hierarchical referential description generation—from coarse categories to fine-grained spatial referents—and diverse QA pair creation. Since part of our pipeline builds on prior great work [112], we focus here on the key modifications we introduce and the motivations behind them.
- – 3D Scanning Data (Supp. B.2): This section outlines the 3D data selection process from CA-1M [40] and ScanNet [22] and presents methods for enriched scene graph construction compared to the 2D data source. We further describe a QA generation framework that leverages detailed 3D annotations (e.g., depth maps, oriented 3D bounding boxes) to capture richer spatial relations for 3D spatial referring and measuring.
- – Object-centric Spatial Tracing from 3D scanning data (Supp. B.3): To bridge the gap between static 3D scanning scenes and manipulation, we introduce a simulation-based generation pipeline. This section details our hierarchical framework (e.g., RRT* with escape mechanisms), the taxonomy of five manipulation primitives including active obstacle bypass, and the rigorous spatial trace refinement process. We further explain how we synthesize metric-grounded tracing data with diverse types (e.g., 2D, 3D, and Lifting 2D to 3D).
- – Manipulation Video (Supp. B.4): We describe how to construct endeffector-centric spatial tracing data, including cleaning real-world manipulation video datasets and generating large-scale data in simulation. This enables data collection across three different robot embodiments and both single-arm and dual-arm configurations.

#### B.1 2D Web Image

- 2D Web Images aim to provide basic spatial concepts and broad-scale perception across indoor and outdoor scenes. Here we use OpenImage [39] as 2D data source.

Dataset Construction Inspired by recent notable work [11,18,112], we adopt a multi-step pipeline for data cleaning and construction: (1) multi-stage image filtering, (2) pseudo-3D scene graph construction, (3) hierarchical object

https://storage.googleapis.com/openimages/web/index.html

description generation, and (4) diverse QA pair generation based on the scene graphs. While our overall approach aligns with prior work [112], we further refine Step 2 by developing a more accurate method for constructing pseudo-3D scene graphs.

Multi-Stage Image Filtering. The OpenImages [39] dataset covers 1.7M training images with extensive visual diversity. However, a large portion (e.g., text-only graphics, QR codes, medical scans, abstract art) is not well-suited for spatial reasoning, especially for 3D spatial referring and measuring. To curate a subset amenable to referential and reasoning tasks, we adopt a two-stage filtering approach. We detail below.

(1) Stage 1: Coarse Filtering. We quickly eliminate off-theme or lowquality imagery lacking multiple everyday objects by using the “siglip2-giantopt-patch16-384” model [86]. We define positive labels (desired content) and negative labels (undesired content), then compute the cosine similarity between each image embedding and all label embeddings. Images closest to positive labels are retained; otherwise, they are discarded. Iterative refinement of label sets balances recall and precision. From the original 1.7M images, 934k remain after this step (see Listings 1.1 and 1.2 for label details).

Listing 1.1: Positive Labels used during SigLIP2 filtering.

Positive Labels = [ "Mid-distance observation of some objects on a table", "Some objects on the desktop", "Distant view of some animals", "Mid-distance observation of some animals", "Distant view of one object", "Mid-distance observation of one object", "Distant view of some objects", "Mid-distance observation of some objects", "Distant view of a person", "Mid-distance observation of a person", "Distant view of some people", "Mid-distance observation of some people", "Distant view of indoor scene", "Distant view of outdoor scene", "Distant view of traffic", "Distant view of Urban architecture"

]

Listing 1.2: Negative Labels used during SigLIP2 filtering.

Negative Labels = [ "Macro shot of an animal", "Macro shot of one object", "Macro shot of a person", "Macro shot of flowers", "A piece of text",

"A person displayed in front of a white background", "A product displayed in front of a white background", "A screenshot of the graphics user interface", "A dimly lit environment"

]

SigLIP2 preserves images enriched with diverse objects, robust depth cues, and contextual variety (indoor/outdoor) through the aforementioned labeling. However, it struggles with certain image categories: (1) paintings/artworks with visible brushstrokes or textures, (2) low-light scenes with minimal illumination or heavy shadows, (3) grayscale photographs lacking color cues, (4) distorted images exhibiting geometric anomalies, and (5) multi-scene collages containing three or more distinct segments. These types hinder reliable detection and interpretation, underscoring the need for a secondary, fine-grained filtering stage.

Stage 2: Fine-grained Filtering Due to SigLIP2’s limited capacity for certain visual content, we introduce a fine-grained filtering stage using the Qwen2.5-VL-

- 7B [5] model to improve dataset quality. This ensures that remaining images are clear, authentic, and suitable for the spatial understanding and reasoning essential to 3D spatial referring and measuring tasks. In total, Qwen2.5-VL processed 934k images pre-filtered by SigLIP2, retaining 846k. While Qwen2.5-VL offers superior precision, its slower speed necessitated SigLIP2 for rapid initial filtering, thereby increasing overall efficiency. For accurate and consistent fine-grained filtering, we employ a structured prompt engineering approach. A system prompt (see Listing 1.3) designates Qwen2.5-VL as an image analysis expert, specifying key visual attributes to verify and negative categories to detect, with a strict workflow. For each image, a user prompt (see Listing 1.4) instructs the model to determine whether it falls into any predefined negative category. The model’s response follows a fixed format: if the text after the pipe symbol (|) is ”Yes”, the image is discarded; otherwise, it is retained. This scheme enforces a consistent output format and enhances the reliability of filtering.

Listing 1.3: System Prompt for Qwen2.5-VL-7B filtering. system_prompt = """ You are an image analysis expert. Follow this workflow

rigidly:

- 1. **Content Analysis **:

- - Inspect: Main subjects , artistic style , visual characteristics

- - Check: Lighting intensity , color channels , geometric integrity , composition structure

- 2. **Category Verification** (YES if matches ANY):

- a) Painting/Artwork - Visible brushstrokes/canvas texture

- b) Dim Lighting - Very low brightness , heavy shadows

- c) B&W Photo - Grayscale only (0 color channels)

- d) Distorted Image - Warping/mirroring anomalies

[Figure 179]

- Fig. 6: The visualization of pseudo-3D scene graphs generated from 2D images with detected objects, and corresponding point clouds.

- e) Multi -image Collage - >=3 distinct scenes with hard borders

- 3. **Structured Response **: Output EXACTLY in this format: "[Analysis sentence]. | Yes/No"

- - Analysis must contain observable evidence

- - Final answer MUST use pipe separator Examples of VALID responses:

"This image is a composite created by stitching together multiple smaller images , with distinct white borders visible between the individual components. | Yes"

"This image features vibrant colors , is neither an artistic painting nor a composite of multiple images ,

and does not conform to any of the specified categories. | No"

"""

Listing 1.4: User Prompt for Qwen2.5-VL-7B filtering. user_prompt = """ Analyze if this image belongs to ANY of these categories:

- 1. Painting/artwork

- 2. Dim lighting

- 3. Black -and -white

- 4. Geometric distortion

- 5. Multi -image collage

Respond EXACTLY FORMATTED as: "[Your evidence -based analysis]. | Yes/No" """

Pseudo-3D Scene Graphs Construction Although 2D images provide limited spatial cues, deriving rich 3D spatial information (e.g., near vs. far, distances), especially for 3D spatial referring and measuring, directly from these images remains challenging. Inspired by prior work [11, 18, 112], we construct pseudo-3D scene graphs from 2D images to enhance QA pairs with robust 3D

spatial semantics. In these graphs, nodes represent object attributes, while edges denote inter-object spatial relationships. The following sections detail this conversion process.

Object Detection and Annotation. While the OpenImages dataset provides annotations, its limited vocabulary and coarse labeling hamper open-world applications. To remedy this, we integrate state-of-the-art foundation models for refined object detection and labeling. Specifically, the Recognize Anything Model (RAM++) [106] and GroundingDINO [53] jointly assign semantic labels and bounding boxes to key objects in filtered 2D images. The pipeline proceeds as follows:

- 1. Semantic Labeling via RAM++: RAM++ analyzes each image to generate comprehensive category labels, ensuring broad semantic coverage.
- 2. Bounding Box Localization via GroundingDINO: Label outputs from RAM++ serve as text prompts for GroundingDINO, which localizes objects and produces precise bounding boxes.

- 3D-aware information Extraction. To further extract 3D-aware information from 2D images, we need to convert the 2D images into 3D representations. Unlike prior works (e.g., RoboRefer [112]) that rely on multiple foundation models for metric depth estimation (e.g., UniDepth V2 [64]) and camera intrinsics prediction (e.g., WildeCamera [116]), we adopt MoGe-2 [88], a single accurate monocular geometry reconstruction model with metric scale and sharp details. Compared to existing methods such as MoGe, DepthPro, UniDepth V2, Metric3D V2, and Depth Anything V2, MoGe-2 provides more accurate relative geometry, precise metric depth, and sharper detail recovery. It reconstructs scaled point clouds and estimates camera intrinsics from a single image, enabling robust 3D scene reconstruction. we adopt for metric depth estimation due to its recent state-of-the-art performance. Based on previously annotated object bounding boxes, we apply SAM 2.1 [68] to generate instance masks. Each resulting Pseudo3D scene graph comprises object labels (via RAM++), 2D bounding boxes (via GroundingDINO), instance masks (via SAM 2.1), and object-level point clouds (via MoGe-2). Some visualizations are provided in Fig. 6.

Hierarchical Object Description Generation While 3D scene graphs commonly encode broad object categories, real-world scenes frequently contain multiple instances of each category. To distinguish among these instances, we augment object descriptions with attributes and spatial relations, yielding fine-grained disambiguation. Below, we outline our two-stage generation pipeline.

- Stage 1: Generating Object Descriptions in image space. We employ the Qwen2.5-VL-7B model to generate detailed object- and image-level captions. These global captions provide contextual grounding for the QwQ-32B model during LLM QA generation (see Supp. B.1), thereby enhancing relevance and accuracy. Prompt templates can be found in Listings 1.5 and 1.6. Notably, the object_caption_user_text_prompt includes a dynamic placeholder

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

black frying pan at left yellow cup at upper left white book at lower left

black chair at left

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

black motorcycle at left, the second motorcycle from the left

woman at left, which is the first person from the left

red barrel,which is the first barrel in back-tofront order

black goose at right, placed as the first goose when sorted from right to left

- Fig. 7: Generated object descriptions. Top: Unique-category captions. Bottom: Spatially-aware captions for the same categories. Red boxes indicate referenced objects.

[class_name], populated with categories predicted by the RAM++ model (see Supp. B.1).

Listing 1.5: Prompts for Image Caption Generation with Qwen-VL. image_caption_system_text_prompt = """

You are an expert image analysis assistant. Your task is to generate a detailed and comprehensive description of the image.

Please focus on accurately capturing all visual elements present in the image , including objects , scenery , colors , shapes , textures , and lighting. Your description should be clear , precise , and professional. Additionally , ensure that your description begins with either ‘this image ’ or ‘the image ’.

""" image_caption_user_text_prompt = """

Please carefully examine the provided image and generate a detailed description. Include all visible elements such as objects , scenery , colors , shapes , textures , and lighting. Ensure that your description is thorough , accurate , and complete , and that it starts with either ‘this image ’

or ‘the image ’. """

Listing 1.6: Prompts for Object Caption Generation with Qwen-VL. object_caption_system_text_prompt = """

You are a visual localization analyzer working with TWO distinct images:

- 1. [POSITION -REFERENCE] (First Image):

- - Contains ONLY location clues with background

- - Strictly use ONLY for determining spatial position ( left/right/upper/lower/center)

- - Ignore all visual features except object placement

- 2. [DETAIL -SOURCE] (Second Image):

- - Shows the object ’s TRUE APPEARANCE without background

- - Extract EXCLUSIVELY from this: color , texture , material , shape

- - Never infer details from the first image

Generate phrase in pattern: [Color][Material][Object] at [Position] Example: "Matte black laptop on the left" NOT "Red -boxed laptop"

""" object_caption_user_text_prompt = """

For the [class_name] marked by red box in FIRST image and fully shown in SECOND image:

- -> COLOR/MATERIAL: Must come from SECOND image

- -> POSITION: Only from FIRST image ’s placement Forbidden actions: x Mention ’red box ’ or background elements x Use location terms in second image x Combine features across images

Describe the [class_name] marked by red box in FIRST image and fully shown in SECOND image with this format:

[Color][Material/Texture][Object] at [Position] Samples:

- - "Brushed metal water at bottle left"

- - "Glossy ceramic mug at upper center"

- - "Faded denim jacket at lower right"

"""

- Stage 2: Generating Object Description with Spatial Cues. To enhance referential specificity when multiple same-category objects coexist, we adopt a heuristic that appends spatially oriented references (e.g., “the third chair from the front”). This strategy leverages 3D object coordinates from the scene graph (see Supp. B.1). By comparing same-category objects along the three principal axes (front–back, left–right, top–bottom), we identify the axis with maximal spatial variation as the primary direction for generating relative spatial descriptions. After identifying the main sorting axis, we retrieve suitable templates from a predefined library (see Listing 1.7) to refine the initial object descriptions. These

[Figure 188]

This image captures a bustling urban street scene, likely taken during the daytime as indicated by the bright lighting and clear skies. The street is lined with a mix of historic and modern buildings, showcasing a variety of architectural styles. On the left side of the street, there are older, multi-story buildings with brick facades and classic designs, featuring large windows and decorative cornices. A prominent sign reading "TENNESSEE" is visible, suggesting the location could be in Tennessee. The right side of the street features a taller, contemporary glass building, reflecting the surrounding environment, which indicates a blend of old and new architecture typical of many cities. The street itself is well-maintained, with a few vehicles parked along the sides and others moving, as suggested by some visible headlights and taillights. The vehicles are a mix of sedans and SUVs, reflecting a typical urban car ownership. The sidewalks are spacious, with lampposts lining the street, providing a warm glow that enhances the urban ambiance. In the background, a lush green hillside rises, offering a natural contrast to the urban setting. The hill is densely covered with green foliage, suggesting a park or a natural reserve, which may be a significant feature in this area. The overall scene conveys a lively yet harmonious blend of urban activity and nature. The lighting in the image is soft and natural, indicative of a sunny day, and the shadows are not harsh, suggesting it might be taken either in the morning or late afternoon.

- Fig. 8: The visualization of generated image detailed descriptions from VLM (e.g., Qwen2.5-VL).

templates capture diverse natural language patterns. For instance, if a row of chairs is arranged left to right, potential templates include “{dense_caption}, which is the {ordinal} {class_name} from left to right,” or “{dense_caption}, the {ordinal} {class_name} in the left-to-right sequence”. Here, dense_caption is the Qwen2.5-VL–generated description, ordinal indicates the object’s position in the sequence, and class_name is the category label output by RAM++. This spatially aware enhancement applies only when multiple instances of the same category appear, preventing redundancy; otherwise, the original dense caption is used. To ensure spatial diversity, we enforce a variance threshold across the three principal axes, discarding images with multiple same-category instances but low variance. This process yields a final set of 466k images. By integrating spatial ordering with visual descriptions, our heuristic produces precise, discriminative referential expressions, which are essential for generating high-quality, unambiguous question-answer pairs (e.g., 3D spatial referring and measuring).

Listing 1.7: Templates for Spatial Order Description Enhancement. TEMPLATES = {

"left_to_right": [ "{dense_caption}, which is the {ordinal} {class_name} from left to right", "{dense_caption}, marked as the {ordinal} {class_name } in a left -to-right arrangement",

], "right_to_left": [

"{dense_caption}, the {ordinal} {class_name} viewed from the right", "{dense_caption}, the {ordinal} {class_name} from the right",

], "front_to_back": [

"{dense_caption}, which appears as the {ordinal} {

class_name} when viewed from the front", "{dense_caption}, positioned as the {ordinal} {

class_name} in front -to-back order", ],

"back_to_front": [ "{dense_caption}, which is counted as the {ordinal} { class_name}, starting from the back", "{dense_caption}, the {ordinal} {class_name} in the back -to-front sequence",

], "top_to_bottom": [

"{dense_caption}, the {ordinal} {class_name} viewed from the top", "{dense_caption}, placed as the {ordinal} {class_name } when sorted from top to bottom",

], "bottom_to_top": [

"{dense_caption}, which ranks as the {ordinal} { class_name} in bottom -to-top order", "{dense_caption}, arranged as the {ordinal} {

class_name} when ordered from the bottom", ]

}

Examples of Object and Image Descriptions This part qualitatively show representative examples with generated descriptions. As shown in Fig. 7, we present two types of object captions: the top row displays simple Qwen2.5VL captions for single-instance categories where spatial ordering is unnecessary, while the bottom row includes captions enriched with spatial information to distinguish multiple instances of the same category. Additionally, Fig. 8 shows Qwen2.5-VL’s ability to generate detailed global descriptions of entire images used in the following.

Generating Diverse QA Pairs via 3D Scene Graphs After constructing scene graphs and generating hierarchical object descriptions, we can leverage this information to generate diverse QA pairs from pseudo-3D scene graphs to support SFT training for improved spatial understanding, especially for 3D spatial referring and measuring.

Template, Choice and Fact QA Generation. We first adopt a templatebased approach to generate structured QA pairs, multiple-choice questions, and factual statements. These templates are derived from scene graph information (e.g., object attributes for 3D spatial measuring, 3D positions for 3D spatial referring) and refined hierarchical object descriptions. The spatial concepts addressed by our QA templates encompass the following categories:

- 1. Relative position relations: capture spatial layouts (left/right, above/below, front/behind).
- 2. Relative size comparisons: describe object attributes (e.g., bigger/smaller, taller/shorter, wider/thinner) often inferred from image-plane projections.
- 3. Quantitative information from 2D or pseudo-3D label: include spatial reasoning based on estimated depth maps, 2D object coordinates, 3D

object coordinates (i.e., 3D spatial referring), coarse depth approximations, and coarse metric estimation for 3D spatial measuring (e.g., object’s length, width, height, distance).

We further design fact templates to generate declarative statements, forming a structured basis for prompting Reasoning LLM to produce richer and more natural QA pairs.

Reasoning QA Generation. To produce more diverse, complex, and natural QA pairs beyond templated formats, we employ QwQ-32B [84], a powerful reasoning LLM. Given factual statements, initial QA pairs, multiple-choice questions (if available), as well as global image captions and precise object descriptions, QwQ-32B generates more challenging and conversational spatial reasoning QA.

- B.2 3D Scanning Data
- 3D scanning data typically provides richer and more accurate 3D information (e.g., camera geometry, oriented 3D bounding boxes), whereas 2D web images rely entirely on foundation models to infer 3D structure. However, a major challenge in utilizing 3D datasets lies in the lack of comprehensive annotations that support full 2D-3D understanding. This requires, at a minimum, accurate camera parameters (i.e., intrinsic and extrinsic), as well as complete per-object annotations in both 2D (e.g., bounding boxes, masks, categories, spatial captions) and 3D (e.g., oriented 3D bounding boxes). While prior work [112] addresses several problems of specific dataset, such as resolving the missing object category labels in CA1M [40], we note that a key resource, ScanNet [22], is omitted, due to its lack of

- 2D bounding box annotations. To this end, we leverage the well-processed and comprehensive versions of the CA-1M and ScanNet datasets—the former processed following the approach of prior work [112], and the latter completed by filling in missing information as detailed below—to construct spatially related QA pairs. These QA pairs focus on 3D spaital reasoning, especialy for
- 3D spatial referring and measuring, and more importantly, on building objectcentric, multi-step, metric-grounded spatial tracing data enabled by complete 3D information. In the following, we first detail the comprehensive annotation pipeline for ScanNet, followed by how CA-1M and ScanNet are used to construct the desired data (e.g., 3D spatial referring, measuring and tracing).

ScanNet Data Processing We leverage the ScanNet dataset, utilizing the original 3D bounding boxes, object labels, depth maps, and camera intrinsic and extrinsic parameters provided by EmbodiedScan. However, EmbodiedScan does not provide 2D bounding box annotations, which are necessary for our processing pipeline. Therefore, we implement a procedure to automatically generate these

- 2D boxes from the 3D data. Our extraction process is as follows:

[Figure 189]

- Fig. 9: The visualization of the 2D bounding box generation pipeline for the ScanNet dataset.

- Step 1: 3D Point Sampling. For a given 3D bounding box, we first sample 5,000 points uniformly from its surface.
- Step 2: Projection and Filtering. We first project these 3D points into the 2D image plane using the provided camera parameters. We then retain valid points based on two criteria: (1) Depth Consistency Check: We compare the depth of the projected point (Zproj) with the depth in the corresponding pixel of the depth map (Dmap). A point is retained only if: Dmap−5cm < Zproj < Dmap+5cm. (2) Image Boundary Check: We discard any point that projects to a 2D coordinate (x,y) outside the image boundaries (i.e., x < 0, x ≥ width, y < 0, or y ≥ height).
- Step 3: 2D Bounding Box Generation. After filtering, we compute the 2D

bounding box by finding the minimum and maximum coordinates (xmin,ymin,xmax,ymax) from the set of all remaining valid 2D points.

The visualization of this process is shown in Fig. 9.

- 3D Object Description Generation and Scene Graph Construction for 3D Scanning data Following the 2D scene graph pipeline (Supp. B.1) and the object description generation approach (Supp. B.1), we build 3D scene graphs that structurally resemble those in OpenImages (Fig. 6). Unlike purely

- 2D counterparts, these 3D graphs emphasize indoor settings and incorporate precise geometric cues (e.g., ground-truth depth, camera intrinsics/extrinsics,
- 3D oriented bounding boxes). The enhanced 3D scene graph outperforms

- 2D-based counterparts in object localization and spatial relation accuracy, enabling structurally rich and quantitatively grounded 3D QA data across 28 spatial relation types. Similar to the 2D pipeline, we construct template-based, choice-based, and fact-based QA pairs from the 3D scene graph. Fact QA pairs are further paraphrased and enhanced using a reasoning

[Figure 190]

- Fig. 10: The visualization of our semantic dimension definitions. Length (red) is defined as the intersection of the front and bottom faces. Width (green) is the intersection of the side and bottom faces. Height (blue) is the intersection of the front and side faces.

LLM to generate reasoning questions. Moreover, due to the richer and more precise information in 3D scanning data, we additionally introduce methods for constructing fine-grained 3D spatial measuring, referring, and tracing data, as detailed below.

Human-Like Measuring Descriptions Generation During the generation of ground-truth text related to distances and dimensional measurements (i.e., length, width, height), we make two critical observations regarding model performance: (1) we find that strictly constraining the model to output a single, standardized unit (e.g., forcing all values to be in “meters") negatively impacts its predictive accuracy. (2), we observe a similar degradation in performance when the target text omits the name of the object being measured (e.g., providing only the numerical value “1.5 meters" rather than a contextual phrase like “the table is 1.5 meters wide"). Based on these findings, we design a ground-truth generation strategy to create more natural, “human-like" descriptions. This strategy has two main components:

- 1. Contextual Phrasing: All measurement-related descriptions are formed in a sentence with addition context, i.e., explicitly includes the name of the object and related measurement type (e.g., length, width, height), rather than being a standalone numerical value of the answer. For example, for the question “What is the height of the cargo ship at left near the port terminal scene?”, the contextualized answer “The height of the cargo ship at left near the port terminal scene is about 43 meters” performs better than the raw response “43 meters”.
- 2. Dynamic Unit Selection: Instead of a fixed unit, we implement a dynamic unit selection mechanism that adapts to the magnitude of the value. This method converts the base metric value into a set of plausible units (e.g., meters, centimeters, feet, inches) and stochastically selects one based on human-like preferences. For example, values less than 1 meter are randomly converted to “centimeters" (with a higher probability) or “inches". Values

greater than 1 meter are typically expressed in “meters" but are also stochastically converted to “feet". This approach ensures the model is exposed to a diverse and realistic range of textual formats, which our experiments show leads to better performance.

- 3D Spatial Measuring Data Generation To generate the data for 3D spatial measuring, we first define the object dimensions (length, width, height), which is very crucial for 3D scanning dataset, and then extract them. Herein we mainly focus on CA-1M and ScanNet.

Data Normalization Across Datasets. A critical issue we notice during data processing is the inconsistent definition of bounding box dimensions across datasets. We find that the 3D bounding box scales provided by CA-1M [40] follow a [width, height, length] order. In contrast, the ScanNet annotations provided by EmbodiedScan adopt a [width, length, height] order. To resolve this, we implement a normalization step to unify all bounding box data into a single, consistent format before extracting measurements.

Semantic Dimension Definitions. A simple extraction of [width, height, length] dimensions from dataset’s raw annotations is insufficient, as these are arbitrary and do not align with human concepts of “length" or “width". We thus establish a consistent, semantically-grounded definition for object dimensions based on their canonical orientation (e.g., their “front", "“side", and “bottom" faces). Our definitions are as follows:

- – Length: Defined as the length of the edge where the object’s front and bottom face intersect.
- – Width: Defined as the length of the edge where the object’s side and bottom face intersect.
- – Height: Defined as the length of the edge where the object’s front and side face intersect (i.e., the vertical edge).

This semantic-based definition, shown in Figure 10, ensures that a query for “how wide is the cabinet" consistently refers to the same geometric property, regardless of the object’s orientation in the scene.

3D Spatial Referring Data Generation For 3D spatial referring tasks, we generate data that pairs multi-step descriptions with precise 3D surface points. Our approach is inspired by previous work in 2D spatial referring [112], which typically associates a 2D pixel coordinate (x,y) with a phrase like “the corner of the leftmost table”. A key limitation of 2D referring data is that a 2D point represents an infinite ray in 3D space, lacking specific 3D localization. To overcome this, we build upon the 2D referring data by incorporating depth information. For a given 2D referring point (x,y) that indicates a specific object, we also extract its corresponding depth value d from the depth map. This critical step ensures that the ground-truth target is not an ambiguous ray but a precise 3D coordinate (x,y,d). This 3D point is guaranteed to lie on the visible surface

of the target object, providing an unambiguous 3D grounding for the language description.

#### B.3 Object-centric Spatial Tracing Generation

Since static 3D scanning datasets (e.g., CA-1M, ScanNet) lack ground-truth manipulation trajectories to exect spatial traces, we develop a simulation-based generation pipeline to synthesize high-quality object-centric spatial traces. This pipeline transforms static 3D scene graphs with occupancy information into simulated manipulation trajectories through a structured process: initialization, planning, and reasoning.

Scene Initialization & Task Formulation Before trace generation, we must standardize the environment and rigorously define the roles of objects within the scene to ensure plausible task formulation.

Gravity Alignment and Data Validation. Raw 3D scans often exhibit arbitrary coordinate orientations. To support physics-based reasoning (e.g., “stacking”), we first standardize the scene geometry using a gravity alignment matrix Rgravity. For every object Oi with corner coordinates Pi ∈ R8×3, we apply the transformation Pi′ = PiRgravityT . Following alignment, we perform strict geometric validation. Objects with degenerate bounding boxes (fewer than 4 valid corners) or containing numerical anomalies (NaN/Inf values) are filtered out to prevent simulation instability.

Object Role Assignment. A critical challenge in data generation is balancing semantic richness for VQA with physical completeness for simulation. As detailed in Sec. B.2, our filtering pipeline produces a subset of “High-Quality Objects” with rich VLM-generated captions, while the original raw dataset contains all geometric instances but lacks detailed descriptions, especially in CA-1M (more details can be found in prior work [112]). To address this, we assign roles as follows:

- – Moving Objects (Osrc) & Via Objects (Ovia): Selected exclusively from the High-Quality Object Subset. This is crucial for generating unambiguous referring expressions in VQA tasks. For instance, in a scene with multiple apples, raw category labels are insufficient to distinguish the target. We rely on the rich ‘dense’ and ‘spatial’ captions (e.g., “the apple on the right”) available in the high-quality subset to ensure the generated instruction uniquely identifies the specific object (e.g., “Move the apple on the right around...”), rather than a generic and ambiguous “Move the apple”.
- – Reference Objects (Oref): A subset of the High-Quality Objects that satisfy stability constraints. We identify Oref by verifying they are resting on a supporting platform (e.g., a table or shelf) using a vertical overlap heuristic. This ensures that the destination region near Oref is physically reachable and capable of supporting Osrc.

- – Obstacles (Oobs): Sourced from the Original Raw Dataset (containing all annotated instances). While some objects may lack captions and are excluded

from being Osrc or Oref, they still physically exist in the scene. To ensure collision-free motion planning, our motion planner considers the union of all raw instances as the obstacle set, preventing the generated trace from hallucinating paths through uncaptioned objects.

Taxonomy of Manipulation Primitives. To improve spatial reasoning capabilities, we define five distinct manipulation primitives (Methods 1–5). Additionally, we introduce a mechanism to identify “Potential Via Objects” for standard tasks to enrich spatial trace descriptions, shown in Fig. 11.

- – Method 1 (Place Relative): Moves Osrc to a spatial relation (e.g., left, right) relative to Oref. Instruction Example: “Place the {source_obj} to the {endpoint_direction} of the {reference_obj}.”
- – Method 2 (Directional Move): Moves Osrc by a specific distance towards a cardinal direction. Instruction Example: “Move the {source_obj} toward the {endpoint_direction}.”
- – Method 3 (Stacking): Places Osrc on top of Oref, enforcing surface area constraints (Area_ref ≥ Area_src) to ensure stability. Instruction Example: “Place the {source_obj} on top of the {reference_obj}.”
- – Method 4 (Active Bypass & Place): A high-difficulty primitive where

the planner explicitly identifies an obstacle (Ovia ∈ High-Quality Subset) blocking the direct path and generates a trace to move around it before reaching the target.

Instruction Example: “Move the {source_obj} around the {via_obj} on its {via_direction} side, then place it to the {endpoint_direction} of the {reference_obj}.”

- – Method 5 (Active Bypass & Stack): Combines obstacle avoidance with a final stacking operation. Instruction Example: “Move the {source_obj} around the {via_obj} on its {via_direction} side, then place it on top of the {reference_obj}.”

Note on Potential Via Objects (Method 1–3): Even for standard primitives (Methods 1–3), the generated spatial trace may coincidentally pass close to other objects. We implement a Retroactive Via Discovery logic to identify these objects from the High-Quality Subset and add the spatial constraints along the trace path into original spatially constrained instructions. If a spatial trace passes within a proximity threshold of an object Ok, Ok is labeled as a “Potential Via Object,” adding implicit spatial constraints to the VQA annotations (e.g., “Move {source_obj} around {via_obj} on its {via_direction} side, then place it...”).

Simulation-based Spatial Trace Generation Based on the initialized scene and defined tasks, we employ a rigorous motion planning pipeline in continuous

[Figure 191]

[Figure 192]

[Figure 193]

|Put the matte black remote control<br><br>at center-right which is on the table above the white matte paper at lower left which is inside the drawer, ranking as the 1st paper in left-toright order .<br><br>Method3|
|---|

|Push the matte black remote control at center-right which is on the table left.<br><br>Method2|
|---|

|Pick up the matte black remote control at center-right which is on the table on the silver metallic stapler at upper left which is on the desk's back side.<br><br>Method1|
|---|

[Figure 194]

[Figure 195]

[Figure 196]

|Move the white matte paper at center which is on the table, appearing as the 3rd paper when viewed from the right past the light beige wooden table at center which is next to the curtain on the up side, then place it to the right of the dark gray matte speaker at upper left which is next to the desk.<br><br>Potential Via|
|---|

|Move the red matte apple at lower center past the black matte laptop at center on its back side, then place it on the top of the white matte cabinet at center, the fifth cabinetry from the left.<br><br>Method5|
|---|

|Move the metallic silver food processor at upper right, acting as the third appliance starting from the front around the wooden cabinet at lower right on its front side, then place it to the right of the white matte door at center, arranged as the second door when viewed from the left.<br><br>Method4|
|---|

- Fig. 11: Visual Taxonomy of Manipulation Primitives. We define five distinct primitives ranging from simple relative placement (Method 1) to complex active obstacle bypass and stacking (Methods 4–5). Additionally, we retroactively identify “Potential Via Objects” (bottom) for standard traces to enrich spatial descriptions in cluttered scenes.

3D space (C ⊂ R3). Unlike rasterized grid searches (e.g., A*), our approach utilizes sampling-based planning to handle the high-resolution geometry of scanned environments, ensuring feasibility and plausibility.

Region-based Endpoint Selection. While Sec. B.3 defines valid destination regions, the motion planner requires a specific goal configuration pgoal. To enhance robustness against clutter, we implement a Centroidoutward Polar Sampling Strategy within the candidate region Rdest. Instead of uniform random sampling, we discretize Rdest into a structured polar grid centered at its geometric centroid. We define a set of concentric radii R = {0,0.03,0.06,0.10,0.15,0.20} (meters) and angular steps. The pipeline iteratively evaluates candidate points from the inner rings outwards. For each

candidate pcand, we apply a height adjustment based on the target object’s dimensions relative to the platform surface (ytarget = yplatform +hobj/2+δsafety) and perform a static collision check. The first valid sample is selected as pgoal. This strategy implicitly maximizes the safety margin by preferring locations furthest from region boundaries and obstacles.

RRT* Planning with Dual-modal Escape. We utilize the RRT* (Rapidlyexploring Random Tree Star) algorithm to generate collision-free traces. The planner grows a tree from the start configuration pstart towards pgoal with a goal-biasing probability of 0.25 and a step size of 0.05m. Crucially, to optimize trace quality, we employ a rewiring radius of 0.25m, allowing the algorithm to asymptotically converge towards the optimal path by restructuring the tree topology within the local neighborhood.

Escape Strategy for Constrained Starts. A common failure mode in simulation is when the source object’s initial pose slightly intersects with the environment (soft collision) due to meshing noise or scanning artifacts, which locks the standard planner. To address this, we implement a heuristic Dual-modal Escape Mechanism:

- 1. Visual Opening Analysis: If pstart is in collision, we first analyze the depth map along 6 cardinal directions (up, down, left, right, front, back). We compute an “Opening Score” for each direction by ray-casting into the depth buffer; directions with longer free-space rays are prioritized.
- 2. Geometric Push Fallback: If visual analysis is inconclusive (e.g., occluded camera), we fall back to a geometric approach. We calculate the penetration depth of the object’s AABB against all overlapping obstacles and determine a “Push Vector” that minimizes the translation required to resolve the collision.
- 3. Recovery: The planner moves the object along the optimal escape vector until it reaches free space pfree (max distance 0.6m). The escape segment pstart → pfree is then prepended to the main RRT* plan.

Hierarchical Collision Detection. Precise collision checking is paramount for generating non-penetrating traces. We reject simple sphere-based approximations in favor of a two-phase hierarchical collision detection system:

- – Broad Phase (AABB): We first filter potential colliders using Axis-Aligned Bounding Boxes (AABB). Objects with non-overlapping AABBs are immediately discarded, significantly reducing the computational overhead given the large number of objects in scanned scenes.
- – Narrow Phase (OBB via SAT): For objects passing the broad phase, we perform precise intersection tests. Since scanned objects are often oriented arbitrarily (not axis-aligned), AABBs are overly conservative and prevent valid close-proximity manipulations. We construct Oriented Bounding Boxes (OBB) for the moving object and obstacles by performing Principal Component Analysis (PCA) on their vertices to extract the three principal orthogonal axes. We then employ the Separating Axis Theorem (SAT) to test for intersection across 15 potential separating axes (3 axes from object

A, 3 from B, and 9 cross-products). This rigorous check allows the planner to generate tight, realistic traces that graze valid obstacles without false positives.

Advanced Spatial Reasoning Logic Beyond basic collision avoidance, our pipeline incorporates advanced spatial reasoning modules to generate and interpret complex, multi-step spatial tracing data. This includes proactive planning for obstacle bypass (Methods 4–5) and retroactive analysis of spatial relations for standard traces (Methods 1–3).

Active Bypass Planning (Methods 4 & 5). To synthesize training data that necessitates reasoning about intermediate constraints, we implement a heuristicguided bypass planner. This module forces the robot to deviate from the greedy trace to navigate around a specific “Via Object” (Ovia).

- 1. Obstacle Identification: We first compute a hypothetical direct trace using the standard RRT* planner. The system identifies potential blocking objects by calculating the segment-to-point distance for all scene obstacles.

If an object Oblock lies within a collision margin (< 0.05m) of the direct path, it is designated as the target Via Object.

- 2. Multi-directional Candidate Generation: Unlike simple heuristics that always bypass “to the right”, our system evaluates 6 cardinal bypass directions relative to Ovia’s geometry: left, right, front, back, up, down. For each direction d, we define a valid via-region Vd. The center of this region is calculated as:

cvia = cobs + (robs + rsrc + δmargin) · nd (1)

where cobs is the obstacle center, nd is the normal vector for direction d, and robs,rsrc are the radii of the obstacle and moving object, respectively. This explicitly ensures kinematic clearance.

- 3. Heuristic Scoring for Naturalness: To select the most human-like bypass spatial trace, we score each candidate region using a composite cost function J. Based on our empirical tuning, the cost is defined as:

J = 1.0 · Ltotal + 0.3 · Pangle + 2.0 · Pbacktrack + 0.2 · Plateral (2)

Here, Ltotal is the total trace length. Pangle penalizes sharp turns (cosine similarity). Pbacktrack is a binary penalty (weight 2.0) that strictly discourages candidates requiring movement opposite to the goal vector. Plateral penalizes excessive deviation from the main axis. The candidate minimizing J is selected as the optimal via-point pvia.

- 4. Two-stage Planning & Fusion: The planner executes two independent

RRT* searches: τ1 : pstart → pvia and τ2 : pvia → pgoal. The resulting paths are concatenated. Crucially, during the smoothing phase (Catmull-Rom), we enforce a via-point constraint: any optimization that moves the spatial trace further than 0.12m from pvia is rejected, preserving the intentional bypass behavior.

Retroactive Via Discovery (Methods 1–3). Standard manipulation tasks (e.g.., “Place A next to B”) often implicitly involve spatial constraints relative to other objects in cluttered scenes. To capture this rich semantic information, we implement a retroactive discovery module.

- 1. Proximity Analysis: For a generated spatial trace T , we compute the minimum Euclidean distance to all High-Quality objects in the scene. We calculate the distance to the object’s surface rather than its center: dsurf = ||ptraj−cobj||−robj. Objects satisfying dsurf < 0.15m are flagged as Potential Via Objects.
- 2. Relative Direction Classification: For each flagged object, we determine the spatial relationship of the bypass. We identify the closest point pclose on the trace and calculate the offset vector voff = cobj −pclose. By determining the dominant component of voff orthogonal to the trace’s instantaneous velocity vector, we classify the bypass direction into natural language labels (e.g., “passing on the left”, “passing above”).
- 3. Instruction Enrichment: These discovered relations are injected into the VQA generation pipeline, transforming simple instructions into spatially dense descriptions (e.g., from “Move mug to plate” to “Move mug to plate, passing to the right of the bottle”). This supervision signal is critical for training RoboTracer to attend to collision boundaries and intermediate spatial context.

Trace Refinement & Quality Assurance The raw traces generated from the motion planner are geometrically valid in 3D but may lack the visual alignment and naturalness required for VLM training. We implement a comprehensive refinement pipeline followed by strict quality assurance protocols.

Geometric Smoothing & Compression. To mitigate the robotic artifacts of RRT*, we apply Catmull-Rom Spline smoothing (α = 0.5) to generate fluid, human-like curves. Subsequently, we utilize the Ramer-Douglas-Peucker (RDP) algorithm to downsample the dense path into a sparse sequence of keypoints (N ≤ 8), reducing token complexity while preserving critical motion geometry.

Physics-aware Grounding (End-Point Refinement). A common simulation artifact is “floating placement,” where the trace ends at the object’s geometric center, leaving it visually suspended. We implement a depth-guided correction: by projecting the endpoint onto the image plane and querying the ground-truth depth map, we iteratively descend the trace’s final point along the gravity vector until it makes contact with the physical surface (i.e., projected depth matches sensor depth). This ensures a physically grounded placement.

Visual Alignment Correction (Start-Point Refinement). While 3D planning operates on the object’s geometric center, simply projecting this 3D center into the 2D image may result in a starting point that falls outside the object’s visual mask (e.g., for C-shaped objects or due to perspective distortion). To ensure precise visual grounding:

- 1. We retrieve the ground-truth 2D segmentation mask (RLE-encoded) of the source object.
- 2. We calculate the visual centroid (umask,vmask) of the mask’s largest connected component.
- 3. We override the 2D coordinates of the trace’s start point with (umask,vmask), while retaining the original metric depth dstart.

This alignment guarantees that the visual trace originates perceptually from the object’s body, eliminating “off-target” supervision signals.

Rigorous Quality Control. To filter out hallucinations and low-quality samples, we enforce a set of validation criteria. A trace is discarded if it fails any of the following checks:

- – Occlusion Ratio: We compute the visibility of every interpolated waypoint against the scene’s depth buffer. Traces with > 30% occlusion are rejected to ensure the trace is visually trackable.
- – Field-of-View Constraint: All keypoints, especially the start and end, must project strictly within the camera frame boundaries.
- – Volume-Adaptive Dynamics: Fixed minimum-length thresholds fail to

account for scale variance. We apply a dynamic threshold Lmin = Lbase ·

3

Vobj. This requires large objects (e.g., laptops) to move significant distances to constitute a valid action, while allowing subtle manipulations for small objects (e.g., keys).

- – Semantic Movability: We filter out immovable fixtures (e.g., “floor”, “wall”, “countertop”) using a semantic blocklist, ensuring the model focuses solely on interactable entities.

VQA Data Generation The final stage transforms the refined metric-grounded geometric-aware traces into multimodal instruction-following pairs. To foster robust spatial reasoning, we synthesize instructions with varying levels of granularity and generate three distinct types (e.g., 2D, 3D, and Lifting) for each trace.

Instruction Generation. For each generated trace, we synthesize a language instruction using a template-based approach. A critical feature of our pipeline is the injection of Metric Scale Awareness. As shown in Listing 1.8, our template pool is designed such that approximately 20% of the templates include explicit metric placeholders ({distance:.3f}m). During generation, we calculate the ground-truth trace displacement Ltraj and populate these slots. This design forces the model to not only understand relative directions (e.g., “to the right”) but also correlate visual magnitudes with precise numerical values (e.g., “0.25m”).

Listing 1.8: Templates for Metric-aware Instruction Generation. INSTRUCTION_TEMPLATES = {

"method1_place_relative ": [ # Metric -aware templates (~20%)

"Move the {source_obj} to a position {distance :.3f}m to the {endpoint_direction} of the {reference_obj }.",

"Pick up the {source_obj} and move it to a position { distance :.3f}m to the {endpoint_direction} of the

{reference_obj}.", # Standard templates "Place the {source_obj} to the {endpoint_direction}

of the {reference_obj}.", "Pick up the {source_obj} on the {reference_obj}’s { endpoint_direction} side.",

], "method2_directional_move ": [

# Metric -aware templates "Move the {source_obj} {distance :.3f}m in the {

endpoint_direction} direction.", "Pick up the {source_obj} and move it {distance :.3f}m

toward the {endpoint_direction}.", # Standard templates "Push the {source_obj} toward the {endpoint_direction

}.", "Slide the {source_obj} toward {endpoint_direction }.",

], "method3_stacking": [

# Stacking specific templates "Place the {source_obj} on top of the {reference_obj

}.", "Stack the {source_obj} on the {reference_obj}.", "Put the {source_obj} above the {reference_obj}.", # Vertical relative positioning "Move the {source_obj} onto the {reference_obj}.", "Set the {source_obj} on the {reference_obj}.",

], "method4_bypass_place": [

# Implicit constraints via ’Via Object ’ "Move the {source_obj} around the {via_obj} on its {

via_direction} side , then place it to the { endpoint_direction} of the {reference_obj}.",

"Pick up the {source_obj} around the {via_obj} from the {via_direction} side , then position it to the {endpoint_direction} of the {reference_obj}.",

], "method5_bypass_stack": [

# Complex composite task: Bypass + Stacking "Move the {source_obj} around the {via_obj} on its {

via_direction} side , then place it on top of the {reference_obj}.",

"Pick up the {source_obj} around the {via_obj} from the {via_direction} side , then place it on the { reference_obj}.",

], "potential_via_enrichment ": [

# Retroactive description for standard tasks "Move the {source_obj} around the {via_obj} on its {

via_direction} side , then {final_action}.",

"Pick up the {source_obj}, passing to the { via_direction} of the {via_obj}, then { final_action}.",

] }

Multi-task VQA Formatting. To fully exploit the generated data, we construct three distinct VQA tasks for every single trace sample. We utilize specific prompt templates (sourced from the “Droid” template set in our codebase) to format the queries. This multi-task formulation encourages the model to learn consistent representations across 2D and 3D spaces.

- Type 1: 2D Visual Trace. This task supervises the model to ground instructions into the 2D image plane. The output is a sequence of 2D coordinates τ2D = {(ui,vi)}Ni=1.

Listing 1.9: Prompts for 2D Visual Tracing.

- PROMPTS_2D = [ "Please predict 2D object -centric waypoints to complete

the task successfully. The task is \"<instruction >\".

Your answer should be formatted as a tuple , i.e. [(x , y)], where the tuple contains the x and y coordinates of a point satisfying the conditions above.",

"Point the 2D object -centric waypoints for the task \"< instruction >\". Your answer should be formatted as a tuple , i.e. [(x, y)].",

"You are currently a robot performing robotic manipulation tasks. The task instruction is: \"< instruction >\". Use 2D points to mark the manipulated

object -centric waypoints...",

"Please predict 2D object -centric visual trace to complete the task successfully. The task is \"< instruction >\". Your answer should be formatted as a tuple , i.e. [(x, y)]."

]

- Type 2: 3D Spatial Trace. This is the core task, requiring the model to infer depth and 3D structure from monocular input. The output is a sequence of 3D coordinates τ3D = {(ui,vi,di)}Ni=1, where di is the absolute metric depth.

Listing 1.10: Prompts for 3D Spatial Tracing.

- PROMPTS_3D = [ "Please predict 3D object -centric waypoints to complete

the task successfully. The task is \"<instruction >\". Your answer should be formatted as a list of tuples , i.e., [(x1, y1, d1), (x2, y2, d2), ...], where each

tuple contains the x and y coordinates and the depth of the point.",

"Point the 3D object -centric visual trace for the task \"<instruction >\". Your answer should be formatted as a list of tuples , i.e., [(x1, y1, d1), ...].",

"You are currently a robot performing robotic manipulation tasks. The task instruction is: \"< instruction >\". Use 3D points to mark the manipulated

object -centric waypoints to guide the robot...", ]

- Type 3: 2D-to-3D Trace Lifting. This task isolates the geometric reasoning capability. The model is provided with the ground-truth 2D trace in the text prompt and must “lift” it into 3D space. This effectively trains the model to perform trace-conditioned depth estimation.

Listing 1.11: Prompts for 2D-to-3D Trace Lifting. PROMPTS_LIFT = [

"Please lift the 2D object -centric waypoints to 3D object -centric waypoints to complete the task successfully.

The task is \"<instruction >\". The 2D waypoints is < trace >. Your answer should be formatted as a list of tuples , i.e., [(x1, y1, d1), ...].",

"Lift the 2D object -centric visual trace to 3D object centric visual trace for the task \"<instruction >\". The 2D visual trace is <trace >. Your answer should be formatted as a list of tuples , i.e., [(x1, y1, d1),

...].",

"Please lift the 2D object -centric visual trace to 3D object -centric visual trace to complete the task successfully. The task is \"<instruction >\". The 2D visual trace is <trace >. Your answer should be formatted as a list of tuples ..."

]

Trace Representation. For all tasks, the spatial trace is represented as a sequence of discrete tokens. We normalize the 2D pixel coordinates (u,v) to the range [0,1000] and keep the depth d in absolute meters. This hybrid representation allows the model to leverage its pre-trained 2D visual grounding capabilities while learning precise 3D metric structures.

#### B.4 Manipulation Video

While 3D scans enable object-centric tracing, they lack physically plausible manipulations. We further leverage manipulation video datasets, either real-world

end-effector-centric end-effector-centric end-effector-centric end-effector-centric

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

Place the yellow block on the right on the wooden board on top of the blue box.

Put the tiger plush toy in the bowl.

Straighten the rope on the metallic tube.

Put the black cloth in the box.

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

object-centric object-centric object-centric

object-centric

###### Fig. 12: Extract end-effector-centric and object-centric traces from Droid’s original data by projecting gripper positions into a base frame; downsample traces using RDP and remove tasks with too many points after downsampling.

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

Open Closed Closed Open

###### Fig. 13: Visualization of the end-effector coordinate frame. We obtain the gripper position (blue point) by extending the end-effector’s local z-axis and record both the spatial trace of the gripper and its open/closed states.

videos or simulation videos, to provide spatial traces aligned with the embodied manipulation in tabletop settings.

Droid Data Processing The Droid dataset [37] provides two versions: RLDSformat version and raw version containing raw stereo videos and metadata. Since the RLDS version does not allow extraction of camera intrinsics or depth information, we use the raw version. Ignoring the wrist cameras, the Droid raw dataset contains 117k stereo videos stored in the original SVO format. We extract the camera intrinsics and depth frames from these SVO files using the ZED SDK. The Droid dataset contains a substantial number of samples with incorrect extrinsics (see Fig.14), which manifest as the projected endeffector positions not aligning with the actual end-effector locations in the images. After processing the Droid Raw dataset, we obtain 20.5k unique manipulation trajectories and generate a total of 46.8k end-effector-centric QAs and 58.4k object-centric QAs based on templates, covering both 2D and 3D trace generation tasks (see Fig. 12, 15). The process is as follows:

Raw Data Process. The original dataset is filtered through four criteria, resulting in 80k valid SVO files. The criteria include: (1) Presence of metadata; (2) Availability of language annotations; (3) Complete H5 file provided; (4) Presence of the SVO file. For each valid SVO file, camera intrinsics and extrinsics, as well as depth frames, are extracted using the ZED SDK.

Extrinsic Validation. To verify the correctness of the camera extrinsics, we apply a strict depth-alignment check, which also filters out some occluded frames. The procedure is as follows:

- 1. Project the end-effector coordinates into the corresponding image frame using the camera intrinsics K and extrinsics (R,t):

u = K (Rpeef + t),

where peef is the end-effector position in world coordinates, and u is the projected pixel coordinate.

- 2. Compare the depth value at the projected pixel D(u) with the end-effector’s

z-coordinate in the camera frame, zeefcam. The projection is considered consistent if:

|D(u) − zeefcam| < 5cm.

Only frames where the projected pixel lies within the image bounds are considered.

- 3. If the fraction of frames satisfying the above consistency exceeds 13 of the total tested frames, the extrinsics are deemed valid. This procedure also filters out frames in which the end-effector is occluded by the robot arm. Fig. 16 presents several examples illustrating the correctness of our method.

End-Effector Position Definition and Task Segmentation. We define the gripper position along the end-effector coordinate system’s z-axis at a distance

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

###### Fig. 14: Preview images from the Droid raw dataset with the end-effector projected as a red dot. Each row shows a pair of images from the same scene: the left image corresponds to incorrect camera extrinsics, and the right image corresponds to correct extrinsics.

2D Visual Trace 2D Visual Trace 2D Visual Trace 2D Visual Trace

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

Put the orange plush toy in the colander.

Move the can on the counter.

Put the yellow block in the cup.

Put the marker in the bowl.

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

3D Spatial Trace 3D Spatial Trace 3D Spatial Trace

3D Spatial Trace

- Fig. 15: Droid’s 2D and 3D trace visualizations under the object-centric setting. The 3D traces are visualized in the point-cloud space.

of 0.15m from the base of the end-effector (See Fig. 13). Since Droid does not provide pre-segmented tasks, we perform segmentation based on gripper closure events: (1) Detect intervals where the gripper is closed. (2) Retain only videos containing a single gripper-closure interval. (3) Assign the video instruction directly as the description of the corresponding motion segment.

Trace Extraction and Downsampling. For each selected video, we extract end-effector-centric and object-centric traces as follows: (1) Let the gripper closure interval be [Fs,Fe). Define a base frame: Fb = max(0,Fs − 60). (2) Project the gripper positions within [Fs,Fe) into the base frame to obtain the object trace. (3) Project the gripper positions within [Fb,Fe) into the base frame to obtain the end-effector trace. (4) Downsample each trace using the RDP algorithm and remove tasks with excessive points remaining after RDP downsampling. The visualization of the extracted results is shown in Fig. 12, 15.

AgiBot Data Processing We use the AgiBot World Beta dataset [21] to extract spatial traces from dual-arm robotic manipulation. However, the raw data contains several issues as shown in Fig. 17, including invalid extrinsics, inaccurate motion segmentation, and severe occlusions, which should be resolved prior to trace extraction. We detail our full preprocessing pipeline below.

Tasks Filtering. We begin with a coarse filtering stage to retain only demonstrations that can meaningfully support spatial trace supervision. In Fig. 17, demonstrations exhibiting global robot-based motion while the camera extrinsics remain unchanged implicitly introduce incorrect camera poses; such sequences are manually removed. AgiBot additionally contains many samples with extrinsic errors, which are later detected using depth-alignment validation.

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

Alignment Rate = 71.3% Alignment Rate = 86.5% Alignment Rate = 36.4% Alignment Rate = 56.0%

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

Alignment Rate = 74.4% Alignment Rate = 87.1% Alignment Rate = 81.7%

Alignment Rate = 59.1%

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

Alignment Rate = 0.0% Alignment Rate = 30.2%

Alignment Rate = 4.5% Alignment Rate = 14.1%

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

Alignment Rate = 8.6% Alignment Rate = 0.0% Alignment Rate = 0.0% Alignment Rate = 13.5%

- Fig. 16: DROID depth-alignment results. We verify camera extrinsics by projecting the end-effector into the image and comparing pixel depth with the camera-frame endeffector z value; extrinsics are accepted if over 13 frames have a depth difference under

- 5 cm.

[Figure 289]

[Figure 290]

[Figure 291]

(d).Occlusion.(c).InvalidExtrinsics.

(b).InaccurateMotionSegmentation.

(a).MeaninglessTrajectories.

[Figure 292]

[Figure 293]

[Figure 294]

- Fig. 17: Common issues in the AgiBot dataset. (a) Meaningless Traces. For certain dual-arm collaborative tasks, spatial traces alone cannot accurately describe the motion, making them difficult to interpret even for humans. (b) Inaccurate Motion Segmentation. The original segmentation is often temporally misaligned; for example, trace recording continues even after the object has already been placed on the platform or inserted into the container. (c) Invalid Extrinsics. Incorrect camera extrinsics lead to inaccurate trace projections. (d) Occlusion. The manipulated object (highlighted in the red box) is occluded in the image, rendering the resulting trace meaningless.

Extrinsics Validation via Depth Alignment. Unlike Droid, AgiBot uses a head-mounted RGB-D camera whose depth maps encode near-range geometry, including the robot arm, as zero depth. To continue using depth consistency for extrinsics validation, we extend the definition of aligned depth:

D(u) = 0 or D(u) − zeefcam < 0.05 m.

A video is accepted if the proportion of aligned frames exceeds a threshold. The coverage of retained samples under different choices of the threshold in Fig. 18. We use 0.83 as the threshold in practice. Fig. 19 visualizes all samples as a 2D grid (black = misaligned) and reveals that misaligned frames appear in spatially contiguous clusters, matching manual observations that extrinsic failures occur in temporally coherent segments under fixed scene configurations.

Improved Motion Segmentation. The original AgiBot annotations decompose tasks into atomic labels (e.g., Pick, Place), but these labels are often temporally inaccurate. For instance, the Place interval frequently extends well after the object has reached the target surface. Such coarse segmentation prevents the reliable construction of object-centric traces. We redesign the segmentation to obtain precise end-effector- and object-centric traces.

– Merging Pick–Place Pairs: We scan the annotation and merge each Pick with its subsequent Place segment into a unified Pick-and-Place (P&P) subtask. We define the gripper position as the point located at z = 0.15 m

[Figure 295]

Coverage(%)

Threshold

###### Fig. 18: Proportion of retained samples under different threshold values. A sample is considered to have correct extrinsics if the proportion of aligned frames exceeds a given threshold.

[Figure 296]

###### Fig. 19: Visualizing all samples as a 2D grid (black = misaligned). Misaligned frames form spatially contiguous clusters, consistent with the observation that extrinsic failures occur in temporally coherent segments under fixed scene configurations.

along the end-effector’s local z-axis. All traces are expressed in the coordinate frame of the first frame of the subtask, which we refer to as the base frame.

– Trace Extraction Rules: For each subtask with exactly one gripper-closure

interval, we extract traces as follows:

- • Pick subtasks: ∗ If only one arm closes, we record the end-effector-centric trace from

the subtask start to the closure-interval start. ∗ If both arms close and both move, the sample is discarded (ambiguous). ∗ If both arms close but only one moves, we use the moving arm only.

- • Pick-and-Place subtasks: ∗ If only one arm closes, we record the end-effector-centric trace from

the subtask start to the closure-interval end, and the object-centric trace in the closure-interval.

∗ If both arms close but only one moves, we use the moving arm following the same rule.

- • Other subtasks: ∗ If exactly one arm contains a closure interval, we record the end-

effector-centric trace from the subtask start → closure-interval end. ∗ If both arms close but only one arm moves, we use the moving arm.

This redesigned segmentation enables the construction of both end-effectorcentric and object-centric traces across a wide variety of manipulation behaviors.

Occlusion Detection. Head-camera viewpoints introduce significant occlusion, especially when the robot arm blocks the manipulated object. To filter such cases, we evaluate gripper visibility at a designated check frame:

- – For Pick and P&P subtasks: use the start of the closure interval.
- – For other subtasks: use the end of the closure interval or the subtask end.

We project the gripper position at the check frame into the base frame and inspect depth values within a radius-30 pixel neighborhood. If more than 60% of these depth values are zero, the sample is marked as occluded and discarded.

QA Generation. We apply RDP simplification to all extracted traces to reduce redundancy while preserving geometric structure. Traces that contain out-ofbounds projected points or retain an excessively large number of points after RDP compression are discarded. Finally, for all valid traces, we instantiate tracedescription templates according to their types (end-effector–centric or objectcentric) to construct three categories of QA pairs, yielding a total of 977.5k samples.

Table 13: Statistics of the collected simulation dataset. We utilize the modified RoboTwin environment to generate spatial traces across 16 manipulation tasks. The table details the number of episodes, total traces (including both end-effector–centric and object–centric traces), and the data collection time in hours (using 8 × NVIDIA H100 GPUs).

Task Name Episodes Traces Collection Time (h) Task Name Episodes Traces Collection Time (h)

Click Bell 8827 17654 10.8 Move Playingcard Away 8104 16208 14.3 Click Alarmclock 8222 16444 4.2 Move Stapler Pad 12340 24680 27.2 Blocks Ranking Size 8863 17726 41.6 Open Laptop 3932 7864 20.3 Blocks Ranking RGB 9135 18270 64.2 Place A2B Left 7329 14658 16.0 Handover Block 4234 8468 16.4 Place A2B Right 6437 12874 16.2 Handover Mic 7562 15124 27.9 Place Bread Basket 4337 8674 17.2 Hanging Mug 2577 5154 17.8 Place Bread Skillet 1370 2740 6.9 Move Can Pot 9275 18550 28.9 Place Burger Fries 6005 12010 28.8

#### B.5 Simulation Data

We utilize RoboTwin 2.0 [14] to collect high-precision spatial traces in a simulation environment. While the original RoboTwin data collection pipeline captures only end-effector positions, we modified the simulator to support our specific spatial trace requirements. Our modifications include: (1) Implementation of multi-GPU parallel simulation to accelerate data collection; (2) Extended tracking capabilities to record traces for both grippers and manipulated objects; (3) Refined instruction annotation, allowing a single execution frame to be mapped to multiple fine-grained sub-task instructions; (4) Incorporation of spatial referring expressions to localize objects, thereby enhancing the model’s spatial understanding. We re-engineered 16 original RoboTwin tasks to generate the necessary spatial traces. The statistics of the collected simulation data are listed in Tab. 13. The processing pipeline is as follows:

Coordinate Mapping. We define the functional grasp point by applying an offset of 0.14m along the x-axis of the robot’s end-effector coordinate system to align with the actual gripper center. Using the camera intrinsic and extrinsic parameters, we project the 3D world coordinates of the grasp point into 2D pixel coordinates and depth values within the camera frame.

Task Segmentation and Trace Collection. During task definition, we annotate every simulation frame with sub-task labels. A single recorded frame may correspond to multiple active sub-tasks. To ensure geometric consistency, traces within a specific sub-task are projected into the coordinate system of that subtask’s start frame. The extraction process involves two steps:

- 1. Subject Identification: We determine the active arm based on the instantaneous velocities of the left and right end-effectors. Similarly, the currently manipulated object is identified by calculating the velocities of all objects in the scene.
- 2. Trace Recording: The motion of the active arm is recorded as the endeffector-centric trace, while the position history of the manipulated object constitutes the object-centric trace.

Spatial Referring Generation. To enhance the model’s capability in spatial reasoning and object grounding, we augment the simulation environment by algorithmically generating cluttered scenes. For each task, we instantiate random distractor objects around the target object within a defined radius. These distractors are sampled from a diverse asset library and include both distinct categories and identical categories to the target, necessitating fine-grained discrimination. We develop a rule-based engine to synthesize unique referring expressions for the target object based on its spatial relationship with the environment. The engine parses the scene graph and selects the most concise unambiguous description from the following four categories:

- – Object-to-Object Spatial Relations: Describes the target’s position relative to a unique reference object (e.g., “in front of”, “behind”, “to the left/right of”, “next to”).
- – Proximity Comparisons: Describes the target based on its distance to reference objects or camera viewpoint (e.g., “closest to the [reference]”, “farthest from the camera”).
- – Ordinal Positioning: Identifies the target based on its sorting order along the Cartesian axes (e.g., “leftmost”, “rightmost”, “frontmost”, “the second from the left”).
- – Attribute Comparisons: Distinguishes the target from same-category distractors based on physical properties such as scale (e.g., “the largest”, “the smallest”).

QA Generation. Finally, after applying the RDP algorithm to downsample the traces, we obtain 443.1k unique spatial traces (see Fig. 23- 26). Using templatebased generation, we further produce 1.329M QA pairs, including 914.1k endeffector–centric and 415.1k object-centric samples.

### C TraceSpatial-Bench Details

The TraceSpatial-Bench is a manually annotated, object-centric evaluation suite. To the best of our knowledge, it is the first benchmark designed to evaluate 3D spatial trace prediction for object-centric tasks. The data is sourced from complex indoor scenes in ScanNet [22] and CA-1M [40], which include precise geometry annotations (e.g., absolute depth maps, camera geometry, and 3D occupancy).

#### C.1 Benchmark Composition and Data

Each scene in the benchmark is comprehensive and provides the following components:

- – Visual Observation: The source image and its corresponding absolute depth map.

- – Camera Parameters: Full intrinsic camera parameters (e.g., intrinsic and extrinsic).
- – Object Definition: A 2D mask identifying the specific object to be moved.
- – Target Destination: A 3D bounding box specifying the object’s final intended location.
- – Reference Trace: A feasible 3D object-centric movement path, represented as a series of 3D coordinates. We emphasize that this path is a reference, not a unique ground-truth; many other valid, collision-free paths to the destination may exist.

This data structure allows for rich 3D analysis. Using the depth map and camera intrinsics, we can reconstruct a 3D occupancy map of the environment. Similarly, we can isolate the point cloud of the target object by combining its 2D mask with the depth map. Although TraceSpatialBench is an inherent 3D benchmark, its components can be projected into the

- 2D image plane. We can project the 3D spatial trace and 3D destination box to create 2D paths and 2D boxes, enabling a rigorous evaluation of model performance on analogous 2D tasks.

C.2 Benchmark Statistics

The benchmark consists of 100 manually annotated scenes. The distribution of data sources and task categories is summarized in Tab. 14.

Table 14: Statistics of the TraceSpatial-Bench Benchmark. Criterion Category Count Data Source

CA1M 51 ScanNet 49

Pick & Place 82 Push & Pull 18

Task Category

Total 100

The language prompts have an overall average length of 21.73 words. As defined in our dataset, a “step" refers to the number of spatial relations and associated objects within a single prompt. As shown in Tab. 15, there is a clear correlation: as the “step” count (i.e., task complexity) increases, the average word count of the prompt also increases, reflecting the need for more descriptive language.

Table 15: Benchmark Prompt Statistics by “Step" Complexity.

##### Step Count Number of Prompts Average Word Count

- Step 2 7 10.14
- Step 3 16 14.81
- Step 4 16 17.19
- Step 5 28 22.29
- Step 6 21 27.48
- Step 7 7 30.86
- Step 8 5 34.60

### D RoboTracer Details

#### D.1 Architecture

We adopt NVILA [56] as the base model, including a visual encoder, an LLM, and a multimodal projector. We further incorporate a universal spatial encoder and a scale decoder into the base model to complete our architecture.

Visual Encoder. The image encoder in the base model is siglip-so400m-patch14448 [102], supporting 448 × 448 resolution for richer visual details. Rather than simply resizing the image to a fixed resolution and producing the same number of tokens, this image encoder processes inputs at dynamic resolutions, yielding more visual tokens from higher-resolution images via finer patch division. This enables fine-grained vision-language understanding, crucial for tasks like point prediction that require detailed perception beyond VQA.

Large Language Model. We adopt the Qwen2 LLM backbone from NVILA [56],

which has been fully fine-tuned with extensive data during supervised training. This endows the model with rich visual knowledge, facilitating downstream 3D spatial understanding and reasoning tasks.

Universal Spatial Encoder. We leverage the encoder of the universal feedforward metric 3D reconstruction model, MapAnything [35], as our universal spatial encoder by removing the task-specific DPT and pose heads. During training, the encoder is kept frozen to preserve its pretrained representations. It provides strong 3D metric-grounded priors and flexibly accepts various geometric inputs (e.g., absolute depth, camera geometry). Incorporating more geometric cues leads to more accurate spatial representations, enhancing the VLM’s 3D spatial awareness.

Scale Decoder. We introduce an MLP-based scale decoder that maps the hidden embedding of a special token, “<SCALE>”, to a precise metric scale factor representing the ratio between predicted and real-world scale. The decoder is initialized from the scale head of MapAnything [35] and is trainable throughout our training process. By supervising the predicted scale with a regression loss, we enhance the model’s metric awareness, going beyond standard classificationbased next-token prediction.

- Table 16: Details about the training datasets used in the SFT and RFT stages. M.A. and M.E denote the Metric Alignment and Metric Enhancement step in the SFT stage, respectively.

Stage Categories Datasets SFT (M.A) Spatial TraceSpatial (RGB+X)

Spatial TraceSpatial (RGB), TraceSpatial (RGB+X), ShareRobotBench [33], EmbSpatial [25]

SFT (M.E)

General COCO [47], GQA [31], OCR-VQA [62], TextVQA [73], VG [38], LRV [49]

REC RefCOCO/+/g [96] RFT Spatial TraceSpatial (RGB+X) w/ Reasoning Processing

Multi-Modal Projector. To align multi-modal representations (e.g., image to language, spatial representations to language, language to scale), we use linear connectors, the same as NVILA [56], which is better than Q-Former, to allow the LLM to focus on fine-grained visual understanding and improve generalization.

#### D.2 Training Data

Here we highlight the training data used at each stage, including the number of samples per dataset and the overall total. See Tab. 16 for details.

SFT stage. Specifically, in the first step of the SFT stage, i.e., Metric Alignment, we not only train a spatial projector to align spatial representations and language space, but also train a scale projector with the scale decoder to predict the metric scale factor, using the TraceSpatial (RGB+X) dataset with 4.5M samples, where X indicates arbitrary combinations of geometric annotations (e.g., absolute depth map, camera geometry). In the second step, i.e., Metric Enhancement, we use both TraceSpatial (RGB) and TraceSpatial (RGB+X) datasets, yielding 9M samples to enhance 3D spatial understanding and reasoning (e.g., 3D spatial referring, measuring, tracing). To further improve instruction-following and basic capabilities, we incorporate auxiliary datasets: 965k samples from instruction-tuned data (LLaVA-1.5 [50], LRV [49]), 321k from referring datasets (RefCOCO/+/g [96]), 3k from ShareRobotBench [69] benchmark training sets, and 127k from EmbSpatial [25] benchmark training sets. These additions help bridge distribution gaps between TraceSpatial and benchmark-style queries. The total number of samples used in this stage reaches 14M.

RFT Stage. In the RFT stage, we train the model using TraceSpatial data annotated with detailed reasoning processes, including key intermediate steps (e.g., 3D spatial referring, measuring, scale) and final answers. To ensure both training efficiency and effective learning, we use moderately difficult samples (typically involving 4 − 6 reasoning steps), resulting in a 120k-sample dataset.

#### D.3 SFT Training Details

We formulate the SFT training stage as follows: given a dataset D consisting of samples in the form of triplets (O, Q, A), where O is a sensor image with geometry information (either RGB or RGB+X), Q is a textual question, and

A is the corresponding answer. The answer A may be a direct response (e.g., a sequence of 3D point coordinates) or include intermediate reasoning steps (e.g., key perceptual evidence followed by the final answer, such as 3D spatial referring and measuring). The training objective is to maximize the likelihood of generating the answer given the input pair (Q, A):

LSFT = −E(O,Q,A)∼D

T

log πθ(yt | O,Q,y<t), (3)

t=1

where πθ is the model’s token distribution. The output model πSFT serves as the initialization for the next RFT stage, ensuring a robust foundation for reinforcement learning. To be specific, our SFT consists of two steps. In the first step, Metric Alignment, only the spatial projector, scale projector, and scale decoder are updated by using the TraceSpatial (RGB+X). We employ a maximum learning rate of 1e-4, a weight decay of 0, and a warm-up ratio of 0.03. The 2B variant is trained with a batch size of 7 per GPU, and the 8B variant with 3, both for one epoch. In the second step of Metric Enhancement, we fine-tune all model parameters without the universal spatial encoder using the datasets described in Sec. D.2. Training is conducted for one epoch with a maximum learning rate of 5e-5. We use a batch size of 6 per GPU for the 2B model and 2 for the 8B model. Other hyperparameters follow those in the first step. For more details, please refer to NVILA [56] settings during alignment and SFT.

#### D.4 RFT Training Details

During the RFT stage, we refine πSFT via GRPO [71], a reinforcement learning method designed for efficiency and scalability. Unlike PPO [70], which relies on a costly value network, GRPO estimates relative advantages by comparing intragroup rewards, reducing computation, and simplifying optimization. This makes it well-suited for reasoning-intensive spatial tracing tasks. In detail, we modify R1-V [105] to support our 3D-aware architecture. Training is conducted for two epochs with a batch size of 1 per GPU and 8 outputs in GRPO. For details about hyperparameters, see R1-V [105].

Sampling Action Groups Given an input state s = (O,Q), where O denotes the visual encoding of the RGB or RGB+X observation and Q the textual encoding of the question, GRPO samples a set of actions {a1,a2,...,aN} from the current policy πθ, initialized from πSFT. The sampling process is:

ai ∼ πθ(a | O,Q), for i = 1,2,...,N (2)

This strategy ensures diverse responses, promoting exploration and preventing premature convergence.

Reward Design and Policy Update Each sampled action ai is assigned a reward R(ai) based on verifiable criteria, yielding a reward set r1,r2,...,rN. For spatial tracing tasks, R(ai) integrates three outcome-based and our proposed two process-based components. The outcome-based reward functions are defined as follows:

Outcome Format Reward ROF. This component ensures structured and interpretable outputs by requiring the model to a predefined format: reasoning within “<think>...</think>” and the final answer in “<answer>...</answer>”. A reward is assigned 1 for strict compliance, 0 otherwise.

Point Reward (RP). This component primarily evaluates the consistency between the predicted trace τˆ = {pˆt}Tt=1’s start and end points (p1,pT) and those of the ground-truth trace τ from the annotations of TraceSpatial. This process can be formulated as:

RP = 12 f(p1,pˆ1) + f(pT,pˆT) , (4) f(p,p′) = max 0, 1 − ∥p − p′∥22 (5)

Trace Reward (RP). This component measures the consistency between the predicted trace τˆ and the ground-truth trace τ, using a distance metric d(τ,τˆ) (e.g., Dynamic Time Warping). This process can be formulated as:

d(τ,τˆ) : RT = max 0, 1 − d(τ,τˆ)

All (u,v,d) points are normalized to the interval [0,1], with the depth dimension scaled by the maximum scene depth.

Nevertheless, the outcome-based rewards described above are metric-agnostic and thus fail to provide explicit supervision for the crucial perceptual evidence (e.g., 3D spatial measuring and referring) involved in trace generation. Accordingly, we need process rewards to more effectively guide the model. Howeever, most process-based reward mechanisms rely on a Process Reward Model (PRM), typically a fine-tuned large language model (LLM) or vision–language model (VLM) responsible for providing feedback. However, integrating such an approach into our framework presents two main challenges. (1) Since LLMs cannot process images, they are unable to verify whether the predicted coordinates correspond to the intended object. (2) Although VLMs combine visual and textual inputs, prior work [60] indicates that they may fail to exhibit precise visual understanding when interpreting textual coordinates or metric estimation. Because accurate verification of predicted coordinates or metric approximation is essential for reward assignment, additional or specialized methods must be employed to ensure reliable feedback.

To address this issue, we introduce a metric-sensitive rule-based process reward for spatial tracing, which eliminates the need for a Process Reward Model. Specifically, our method directly evaluates crucial intermediate perceptual steps by leveraging the ground-truth stepwise annotations avail-

able in TraceSpatial. This approach differs from most existing methods that focus on process-based rewards [36,52], which typically assume strictly sequential reasoning and rely on dedicated Process Reward Models (PRMs) for evaluating intermediate outputs. In contrast, we employ metric-sensitive rulebased process reward functions to evaluate intermediate perceptual outcomes in an order-invariant fashion. Our key insight is twofold:

- – (1) Metric-sensitivity: Different spatial attributes (e.g., 3D spatial referring requires 3D points with depth information, while 3D spatial measuring demands precise numerical predictions, including scale) all involve metric scale and require distinct reward formulations due to their inherently different representations.
- – (2) Order-invariance: The reasoning process in spatial tracing is not strictly sequential; for instance, identifying the placement position of the keyboard or the mouse first does not affect the final interpretation of “the free area between the keyboard and the mouse”.

By using this rule-based yet flexible strategy, we overcome the limitations of relying on rigid sequential processes, instead allowing for more robust and adaptable spatial reasoning. We have two process-based reward functions:

Process Format Reward RPF. Similar to the Outcome Format Reward strategy, this component enforces a structured and interpretable reasoning process, thereby facilitating accurate reward computation. In particular, the model is required to produce outputs in the following format:

[Perception Type] [Target Object]: [Value] (6)

where “Perception Type” must be one of three categories: “Referring”, “Measuring”, or “Scale”. The “Target Object” corresponds to a uniquely identifiable entity (e.g., “the second leftmost cup” or “the second large cup from left to right”). The “Value” depends on the selected “Perception Type”:

- – For “Referring”, the value should be a normalized 2D coordinate with absolute depth of the form [(u, v, d)], where both u and v lie in the interval [0,1000], and d rounded to three decimal places.
- – For “Measuring”, the value represents a scalar indicating an object’s length, width, or height, rounded to three decimal places, expressed in potentially different but appropriate units, as discussed in Sec. B.2.
- – For “Scale”, the value is a scalar representing the scale ratio between the predicted scene and the real-world scale, rounded to three decimal places.

Below are examples to illustrate the expected format:

- – [Position] [the second largest cup]: [(245, 147, 1.837)]
- – [Measuring] [the height of the second largest cup]: 20 centimeters
- – [Scale] [Scene]: 2.342

Accuracy Reward RAcc. The reward is computed only for steps annotated as key steps in TraceSpatial. In detail, we use regex matching to determine whether the “Target Object” in the current process format appears in the keystep annotations. If not, the step receives no reward. Since the model has already undergone a cold-start phase in SFT, it can interpret instructions and identify relevant target objects. Thus, a failed match implies that the model cannot accurately refer to the object linguistically, and no reward is assigned. For each perception type, we apply a specific metric to compute the reward:

- – “Referring”: For 3D point modeling (u,v,d), we evaluate each component separately. For the 2D coordinates (u,v), we compute the L1 distance between the predicted and ground-truth points. If the error is within 10% of the image’s longer side, a reward of 0.5 is assigned; otherwise, 0. For the depth d, if the predicted value falls within ±30% of the ground truth, a reward of 0.5 is given; otherwise, 0.
- – “Measuring”: If the predicted value falls within ±30% of the ground truth, the reward is 1; otherwise, 0.
- – “Scale”: If the predicted value falls within ±30% of the ground truth, the reward is 1; otherwise, 0.

We prioritize the correctness of the final outcome over intermediate steps. To prevent reward accumulation from multi-step processes, we scale the process reward by 0.25. The final reward function is defined as:

ri = ROF(ai) + RP(ai) + αRPF(ai) + αRAcc(ai) (7)

where α is set to 0.25. By normalizing the rewards within the sampled group, we obtain the set of relative advantages {A1,A2,...,AN} defined as

ri − mean({rj}) std({rj})

, (8)

Ai =

which measures how each reward compares to the mean in units of standard deviation. We then update the policy based on these advantages, reinforcing actions with higher relative advantages while reducing the likelihood of those deemed less effective. To ensure stable reinforcement learning, the update is further constrained by minimizing the KL divergence between the updated policy and its reference counterpart, thereby promoting incremental and controlled policy adjustments.

### E Experimental Setting and Details

#### E.1 Experiments Compute Resources

We conduct experiments on an H100 GPU cluster, with each node equipped with 8 GPUs.

Object-centric Spatial Trace Generation from 3D Scanning Data. Given that the trace generation pipeline relies heavily on computationally intensive geometric operations (e.g., RRT* planning, OBB-SAT collision detection, and visibility checks) rather than neural network inference, we utilize high-performance CPU parallelization without GPU acceleration. The generation and refinement pipeline is executed on a single high-end server equipped with dual Intel(R) Xeon(R) Platinum 8468 CPUs (totaling 96 physical cores and 192 threads). We configure the pipeline to utilize 180 concurrent processes to maximize throughput. The most time-consuming components are the RRT* obstacle-aware planning, where we set a high iteration count to ensure the generation of complex, high-difficulty traces, and the rendering of visualization assets for quality verification. Consequently, the complete generation, cleaning, and visualization process takes approximately 32 hours for CA-1M and 6 hours for ScanNet.

Real-world Manipulation Video. For Droid, we use the ZED SDK to extract depth and camera intrinsics on a single NVIDIA H100 GPU, which takes 46 hours in total. For the data processing pipelines of Droid and AgiBot, we use 64-way parallel processing, which takes 5 hours and 27 hours, respectively. These processing times include the generation of visualization images and videos.

Simulation Manipulation Video and Evaluation. The data collection time for the RoboTwin simulation is reported in Tab. 13. For the RoboTwin simulation evaluation of our model, we use 8 NVIDIA H100 GPUs, with a total runtime of 36 hours. We further observe that RoboTwin executes significantly slower on H100 GPUs compared to RTX 4090 GPUs.

Metric Alignment in SFT. The process is conducted on 8 nodes over 20 hours for 2B variants and 8 nodes over 50 hours for 8B variants. Both variants training use ZeRO-3.

Metric Enhancement in SFT. The process is conducted on 8 nodes over

- 2 days for 2B variants and 8 nodes over nearly 1 week for 8B variants. Both variants training use ZeRO-3.

Spatial Tracing in RFT. The process is conducted on 1 node over 3 days for 2B variants. However, our model is over twice as slow as other Qwen 2.5-VL-based methods [72,105], mainly because they process only a single RGB image during training and can leverage vLLM for group inference acceleration. In contrast, our method requires RGB+X inputs and modifies the original NVILA architecture, making it incompatible with vLLM or SGLang acceleration.

#### E.2 Spatial Understanding Benchmarks

We evaluate several publicly available spatial understanding benchmarks, strictly adhering to their official evaluation protocols. These benchmarks include CVBench [85] (covering 2D Spatial Relation, 3D Depth Order, and 3D Distance), the BLINK [28] validation set (Spatial Relation, Relative Depth), RoboSpatial [74] (configuration), and EmbSpatial [25]. We omit tasks that are not directly related to spatial understanding—such as the 2D Counting task in CV-Bench and

the Art Style or IQ Test tasks in BLINK—from our analysis. As all the selected benchmarks employ multiple-choice formats, we report accuracy as the evaluation metric. We compare three categories of models in our experiments:

- 1. Proprietary VLMs (e.g., Gemini-2.5-Pro [81]), which have demonstrated strong spatial perception, as discussed in Gemini-Robotics [82].
- 2. Open-source VLMs trained on general VQA datasets (e.g., NVILA [56], Qwen3-VL [83]).
- 3. Spatial specialist models trained specifically on spatially relevant datasets, such as SpatialBot [9] and RoboBrain 2.0 [80], which offer fundamental spatial understanding capabilities.

#### E.3 Spatial Measuring Benchmarks

We evaluate our model’s spatial measurement capabilities on three distinct benchmarks: QSpatial [46] (Plus, ScanNet) and MSMU [13] bench. To determine the success of a single question-answering instance, we adopt the evaluation protocol established in the original QSpatial paper. First, both the model’s predicted value and the ground-truth answer are converted to centimeters (cm). An answer is considered successful if the ratio of the predicted value to the ground-truth value falls within the range of [0.5, 2.0]. Formally, let vpred be the predicted value in cm and vgt be the ground-truth value in cm. The prediction is marked as successful if: 0.5 ≤ vvpred

≤ 2.0.

gt

#### E.4 2D Spatial Referring Benchmarks

We strictly follow the official evaluation protocols of several publicly available 2D spatial referring benchmarks, including Where2Place [97], RoboSpatial [74], and RefSpatial-Bench [112]. These benchmarks gauge a model’s ability to predict one or more points given an image and a textual instruction. The primary evaluation metric is the average success rate, which quantifies the fraction of predicted points that lie within the ground-truth mask for each sample. We compare two categories of models in our experiments:

- 1. General VLMs trained on general VQA datasets (e.g., Gemini-2.5-Pro [81], Qwen3-VL [83]).
- 2. Referring specialist models trained specifically on spatial referring/grounding datasets, such as Molmo [24] and RoboPoint [97], which offer fundamental 2D point referring capabilities.

#### E.5 2D Visual Trace Benchmarks

We evaluate our method on public 2D visual trace benchmarks, including ShareRobotBenchV [33] and VABench-V [98]. ShareRobotBench-V focuses on end-effector-centric

- 2D visual traces, while VABench-V emphasizes object-centric 2D visual traces. Unlike 3D spatial traces, 2D visual traces lack depth information and cannot

assess collision-free properties. Therefore, these benchmarks evaluate the similarity between predicted traces and ground-truth references using metrics such

- as Discrete Fréchet Distance, Hausdorff Distance, and Root Mean Square Error, where lower values indicate better performance. We compare two categories of models in our experiments:

- 1. General VLMs trained on general VQA datasets (e.g., Qwen3-VL [83]).
- 2. 2D visual trace specialist models trained specifically on tracing datasets, such as MolmoAct [41] and Embodied-R1 [99], which offer fundamental 2D visual trace generation capabilities.

- E.6 Spatial Tracing Benchmarks We evaluate model performance in both 2D and 3D for spatial tracing tasks.

- 2D Evaluation For 2D evaluation, all 3D ground-truth attributes (e.g., generated 3d spatial trace, ground-truth 3D bounding box for object’s final intended location) are projected into the 2D image plane. We use the following metrics:

– 2D Start Success: The predicted starting point must be located in the

ground-truth 2D mask of the target object.

– 2D End Success: At least one of the final three (3) predicted points in the projected 2D trace must be located inside the projected 2D bounding box of the destination.

- 3D Evaluation Given the high difficulty of the 3D task, we adopt slightly more lenient spatial thresholds:

– 3D Start Success: The predicted 3D starting point must be in a 20cm

distance of the target object’s point cloud.

– 3D End Success: At least one of the final three (3) predicted 3D points

must be within a 20cm distance of the 3D destination bounding box.

– Overall 3D Success: This composite metric requires a task to first achieve both 3D Start Success and 3D End Success. If it passes, we then simulate the movement of the object’s point cloud along the predicted trace. The path is considered successful if, during this movement, no more than 20% of the object’s points intersect with the environmental 3D occupancy map (i.e., it is collision-free).

Baseline Performance This benchmark proves to be highly challenging. Even with these metrics, advanced baselines like Gemini-2.5-Pro and Qwen-3VL achieve an Overall 3D Success rate of less than 10%. In contrast, our model demonstrates significantly stronger performance, exceeding a 30% success rate.

#### E.7 Simulation Evaluation

We evaluate the capability of RoboTracer to support embodied tasks through spatial trace generation in the RoboTwin simulation environment. Specifically, we select 19 tasks whose behaviors can be intuitively specified using spatial traces–12 tasks have variants that appear in TraceSpatial, while 7 tasks are entirely novel. It is worth noting that the end-to-end model [8,19,54,101,108] results on these tasks are taken from the RoboTwin 2.0 online leaderboard (November 2025). All end-to-end models on the leaderboard are trained or fine-tuned for a single specific task setting and are evaluated in the same task environment. In contrast, the simulation data in TraceSpatial were collected in newly designed scenes with our own instructions.

Our evaluation environment configuration follows that of the RoboTwin 2.0 VLA benchmark (demo_randomized). For Click Bell and Click Alarmclock, we generate end-effector–centric traces, and the gripper executes the predicted trace after closing. For all remaining tasks, we generate object-centric traces: the gripper grasps the object corresponding to the starting point of the trace, follows the model-generated trace, and places the object at the final point. Both grasping and placing operations are implemented using the RoboTwin API. For complex multi-stage tasks, we decompose the instruction into several segments and execute them sequentially. The detailed evaluation results are provided in Tab. 17, and the visualization of the evaluation process are provided in Fig. 27-32. Under the traces predicted by our model, all tasks can be executed with high success rates, demonstrating both the strong spatial understanding capability of our model and its effectiveness in supporting embodied tasks.

We construct the prompts used for RoboTwin evaluation following the template below:

Listing 1.12: Evaluation Prompts.

Please predict 3D \{object -centric | end -effector -centric\}

visual trace to complete the task successfully. The task is "<instruction >". Your answer should be formatted as a list of tuples , i.e., [(x1, y1, d1), (x2, y2, d2), ...], where each tuple contains the x and y coordinates and the

depth of the point. ]

#### E.8 Real-world Evaluation

UR5 Manipulation We show the demo for UR5 Manipulation with human disturbance. In this case, RoboTracer runs at 1.5Hz. Significant shifts in the predicted end-effector-centric spatial trace endpoint, particularly at the 2D pixel level (u,v), can trigger motion interruption and re-planning. Specifically, for

https://robotwin-platform.github.io/leaderboard https://www.universal-robots.com/products/ur5e/

###### Table 17: Performance on RoboTwin hard tasks. We report the success rate (%) compared to end-to-end and VLM-based models. Gray rows indicate the task variants that are not present in TraceSpatial.

End-to-End Policy Vision-Language Model Ours ACT DP DP3 RDT π0 Qwen3-VL-8B Gemini-2.5-Pro RoboTracer-2B

Task

Click Bell 3 0 0 9 3 0 0 96 Click Alarmclock 4 5 14 12 11 0 0 79 Blocks Ranking Size 0 0 0 0 1 0 0 89 Blocks Ranking RGB 0 0 0 0 5 0 0 96 Move Can Pot 4 0 6 12 21 0 0 27 Move Playingcard Away 0 0 3 11 22 0 5 94 Move Stapler Pad 0 0 0 0 2 0 0 18 Place A2B Left 0 0 2 1 1 0 2 84 Place A2B Right 0 0 0 1 6 0 1 93 Place Bread Basket 0 0 1 2 4 0 0 82 Place Bread Skillet 0 0 0 1 1 0 0 48 Place Burger Fries 0 0 18 27 4 0 0 99 Place Container Plate 1 0 1 17 45 0 0 52 Place Empty Cup 0 0 1 7 11 0 0 85 Place Fan 0 0 1 2 10 0 2 66 Place Mouse Pad 0 0 1 0 1 0 0 30 Place Object Stand 0 0 0 5 11 0 0 38 Stack Blocks Two 0 0 0 2 1 0 0 33 Stamp Seal 0 0 0 0 4 0 0 7

grasping, the 2D pixel level (u,v) of the predicted end-effector-centric spatial trace endpoint is fed into SAM2 [68] to generate a segmentation mask, which filters the target object’s point cloud from the scene captured by a third-person Intel RealSense L515 depth camera. The extracted point cloud is input to AnyGrasp [27] to predict a grasp pose in the camera coordinate frame. Using an eye-to-hand calibration method, the grasp pose is transformed into the UR5 robot’s base frame for execution. For final placement, the 2D pixel level (u,v) of the predicted end-effector-centric spatial trace endpoint is also converted to

- 3D coordinates using the camera’s intrinsic parameters and depth data. The

- 3D point is then transformed into the robot’s coordinate system to identify the final placement loccation. Notably, during grasping and final placement, we do not directly use the depth predicted by the spatial trace. Since these points (e.g., grasp point, placement point) lie on contact surfaces (e.g., table, target object), more accurate values can be obtained using known camera intrinsics and observed depth data, enabling robust pick-andplace operations. In contrast, during movement, the waypoints are in mid-air without contact surfaces, so we rely on the model’s predicted

- 3D points. The robotic arm’s end-effector is guided to sequentially reach these 3D waypoint positions for manipulation.

G1 Humanoid Manipulation The main pipeline is the same as UR5 Manipulation. For grasping, we employ a head-mounted Intel RealSense D435 on the Unitree G1 humanoid to capture RGB-D images, which are processed by RoboTracer (i.e., RGB images, depth maps, camera intrinsic as model input) to extract 2D target coordinate (u,v) from the predicted end-effector-centric spatial trace endpoint. These coordinates guide SAM2 [68] to generate a segmentation

https://www.intelrealsense.com/lidar-camera-l515/ https://www.intelrealsense.com/depth-camera-d435/

mask, which filters the third-person D435 point cloud to isolate the target object. The filtered point cloud is then passed to AnyGrasp [27] to predict a grasp pose in the third-person frame, which is transformed to the robot’s base frame using known camera-to-robot calibration. Since the waypoints during robotic watering are in mid-air and not on a surface, 2D-to-3D projection using RGB-D cameras is not applicable. Therefore, the robot directly moves the end-effector to the predicted 3D spatial waypoints for manipulation.

### F More Demonstrations

Visualization of TraceSpatial-Bench. We present examples of TraceSpatialBench and our model’s rollouts in Fig. 20, 21, 22. Visualization of Simulation Data for Spatial Tracing. We present data examples in Fig. 23, 24, 25, 26. Visualization of Simulator. We present example rollouts with RoboTracer predictions in Fig. 27, 28, 29, 30, 31, 32.

### G Discussion on Limitations and Future Work

Despite achieving promising results, our model still has limitations. First, the spatial tracing task requires the model to fully understand 3D space and interpret various spatial constraints from instructions. This demands strong capabilities in perception, reasoning, and generalization across tasks, scenes, and objects. As a result, a VLM-based architecture is essential. However, VLMs often suffer from slow inference, making them less suitable for highly dynamic environments. Acceleration techniques such as quantization or parallel inference may help mitigate this limitation. Second, spatial tracing requires predicting 3D positional sequences with real-world scale information (e.g., depth), which is particularly challenging for VLM-based architectures that are primarily pretrained on 2D data. While RoboTracer makes significant efforts, such as collecting metric-grounded data during SFT, aligning the new inputs via a universal spatial encoder, supervising scale factor prediction through regression loss, using metric-sensitive function to generate rewards, the resulting improvements in 3D metric-grounded spatial understanding, especially driven by scaling up data, are limited compared to the gains observed in 2D spatial reasoning tasks (e.g., RoboRefer [112]). Recent work like DepthLM [10] further highlights this limitation. Despite processing over 30M+ images with unified camera intrinsics, DepthLM sacrifices language understanding to focus solely on depth estimation at the pixel level, achieving performance only comparable to task-specific expert models. These results indicate that enhancing the 3D metricgrounded capabilities of VLMs, while maintaining their language understanding for reasoning, remains an open and challenging problem. We argue that scaling up data alone yields limited benefits due to two key reasons: (1) The lack of a scene-level 3D representation that is naturally aligned with language

and rich enough to encode fine-grained geometry. Although this work attempts alignment in the initial SFT stage, the spatial encoder’s features remain poorly aligned with the language space, and supervision via next-token prediction alone may be insufficient. (2) The lack of comprehensive, scaleaware supervision signals. This work only supervises a simple metric scale factor, but dense geometric outputs (e.g., depth maps), which are crucial for full 3D understanding, are not utilized due to computational overhead. Overall, enabling VLMs to fully understand 3D space requires more than large-scale data. Therefore, better input representations and richer supervision signals, especially

- at both the input and output levels, are a promising direction for future research to further advance spatial intelligence.

[Figure 297]

[Figure 298]

[Figure 299]

ReasoningStep = 4 Pick up the sponge wipe in the white cup, and move it into the sink.

ReasoningStep = 4 Pick up the towel hanging on the handle of the oven, and move it into the sink.

ReasoningStep = 4 Pick up the brown small bottle on the table, and move it to the left of the white mouse.

[Figure 300]

[Figure 301]

[Figure 302]

ReasoningStep = 6 Pick up the second black remote controller from the front on the table, and move it to the spot which is on the bed and on the left of the closest pillow.

ReasoningStep = 6 Pick up the orange object at right which is on the window sill, and move it to a spot which is on the sink's edge and closest to the right wall

ReasoningStep = 6 Pick up the gray toy on the shelf, and move it to the spot which is on the left white table and near the left side of the white shelf.

[Figure 303]

[Figure 304]

[Figure 305]

- ReasoningStep = 3

ReasoningStep = 5 Pick up the third picture frame from the left on the piano, and move it to the right of the biggest wooden chair.

ReasoningStep = 6 Pick up the white box on the table, and move it to the spot which is on the window sill and between the red card and vase.

Pick up the bule bag, and move it to the left of the yellow towel.

[Figure 306]

[Figure 307]

[Figure 308]

- ReasoningStep = 4

ReasoningStep = 7 Pick up the colorful card on the right table, and move it to the right of the dark blue clothes which is on the left wooden table.

ReasoningStep = 6 Pick up the rightmost vase on the desk, and move it to the spot between the black monitor and the water bottle.

Pick up the dark blue clothes which is hanging on the clothes rack, and move it to the white basket on the right.

[Figure 309]

[Figure 310]

[Figure 311]

ReasoningStep = 6 Pick up the white closest clothes, and move it to the spot in the sink and in front of the yellow bottle.

ReasoningStep = 6 Pick up the dark blue cup which is the closest cup to the keyboard, and move it to the spot which is in front of the stapler on the left.

ReasoningStep = 3 Pick up the toothpaste, and move it to the top of the stack of books.

[Figure 312]

[Figure 313]

[Figure 314]

ReasoningStep = 5 Pick up the blue clock on the wooden stand, and move it to the front of the pillow which is the first pillow from the left.

ReasoningStep = 6 Pick up the red object which is on the rightmost table, and move it to the spot which is on the center cabinet and in front of the black object.

ReasoningStep = 5 Pick up the napkin in the box on the left, and move it to the right of round porcelain.

[Figure 315]

[Figure 316]

[Figure 317]

ReasoningStep = 2 Pick up the brown towel, and move it to the metal sink.

ReasoningStep = 3 Pick up the toothpaste on the table, and move it to the blue waste bin next to the table.

ReasoningStep = 3 Pick up the paper cup on the table, and move it to the waste bin.

[Figure 318]

[Figure 319]

[Figure 320]

ReasoningStep = 7 Pick up the white cup on the left wooden table, and move it to the spot on the top of the white shelf and between the stack of paper and calendar.

ReasoningStep = 5 Pick up the glass cup, and move it to the spot which is between the closest monitor and keyboard.

ReasoningStep = 6 Pick up the green paper box on the stand next to the bed, and move it to the right of the pillow.

[Figure 321]

[Figure 322]

[Figure 323]

ReasoningStep = 7 Pick up the brown toy dog next to the door, and move it to the spot which is on the wooden table, between the glass jars filled with colorful candies and white remote control.

ReasoningStep = 6 Pick up the picture frame which is closest to the lamp, and move it to the spot which is on the window sill and on the right side of the small picture frame.

ReasoningStep = 7 Pick up the black controller which is the second black object on the stand from the left, and move it to the white plate on the right side of the lamp.

[Figure 324]

[Figure 325]

[Figure 326]

Touch the bell in front of the rounded square loaf.

Tap the bell to the right of the alarm-clock with rounded edges.

Click the bell closest to milk box.

Click Bell

[Figure 327]

[Figure 328]

[Figure 329]

Tap the top button of the closest alarm-clock to the left of the glass bottle storing olive oil.

Tap the top button of the alarm clock to the left of globe.

Touch the top button of the closest alarm-clock to mouse.

Click Alarmclock

[Figure 330]

[Figure 331]

[Figure 332]

Place the second largest block and place it to the right side of the largest block.

Pick the third block from right to left and place it to the left area of the table center.

Grasp the block farthest from the camera and place it to the right side of the medium largest block.

Blocks Ranking Size

[Figure 333]

[Figure 334]

[Figure 335]

Grasp the second block from right to left and place it to the left area of the table center.

Move the green block and place it to the middle area of the table center.

Grasp the block farthest from the camera.

Blocks Ranking RGB

[Figure 336]

[Figure 337]

[Figure 338]

Pick the red block with the right arm and place it on the green area.

Grasp the red block with the right arm.

Grasp the red block behind the cylindrical blue pen cup.

Handover Block

[Figure 339]

[Figure 340]

[Figure 341]

Pick the second microphone from left to right.

Pick the microphone with the left arm.

Grasp the closest microphone to the right of apple.

Handover Mic

[Figure 342]

[Figure 343]

[Figure 344]

Pick the closest mug to the compact matte gray computer mouse then hang it onto the rack.

Grasp the mug closest to drill then move the it to the table center.

Pick the mug to the right of tabletrashbin.

Hanging Mug

[Figure 345]

[Figure 346]

[Figure 347]

Grasp the largest sauce can then move the it to the right side of the the cooking pot with two handles.

Grasp the second sauce can from front to back then move the it to the left side of the kitchenpot.

Pick the second sauce can from right to left with the left arm.

Move Can Pot

[Figure 348]

[Figure 349]

[Figure 350]

Pick the second playingcards from back to front then move the it to the right side of the table.

Grasp the largest playingcards then move the it to the right side of the table.

Grasp the smallest playingcards with the right arm.

Move Playingcard Away

[Figure 351]

[Figure 352]

[Figure 353]

Pick the stapler to the right of the mouse with two buttons and a scroll wheel then move the it to the left magenta area.

Pick the stapler closest to candlestick then move the it to the left cyan area.

Grasp the stapler closest to the easy-to-carry hamburg box with the left arm.

Move Stapler Pad

[Figure 354]

[Figure 355]

[Figure 356]

Open the laptop closest to the cylindrical bottle for soysauce with the left arm.

Open the laptop with the right arm.

Open the laptop with the right arm.

Open Laptop

[Figure 357]

[Figure 358]

[Figure 359]

Move the closest coffee-box to the bread with golden crust to the left side of the golden brown bread.

Grasp the coffee-box in front of the bread with golden crust with the left arm.

Move the playingcards to the left side of woodenblock.

Place A2B Left

[Figure 360]

[Figure 361]

[Figure 362]

Move the closest bread to the candlestick with wide round base to the right side of the smooth rubikscube with black frame edges.

Move the closest playingcards to the left of remotecontrol to the right side of bread.

Grasp the mouse to the right of coffee box with the left arm.

Place A2B Right

[Figure 363]

[Figure 364]

[Figure 365]

Pick the rubikscube with the left arm.

Move the rubikscube to the right of breadbasket into the basket.

Move the stapler to the left of bread into the basket.

Place Bread Basket

[Figure 366]

[Figure 367]

[Figure 368]

Move the soap closest to the smooth cardboard coffeebox to the skillet.

Move the skillet to the right of toycar to the front of the camera.

Pick the skillet in front of apple.

Place Bread Skillet

[Figure 369]

[Figure 370]

[Figure 371]

Move the closest french fries to the rectangular wooden pencup to the tray.

Grasp the french fries to the right of tray.

Move the closest hamburg to tray to the tray.

Place Burger Fries

Click Bell

[Figure 372]

[Figure 373]

[Figure 374]

Click the bell. Click Alarmclock

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

Click the alarm clock's center of the top button. Blocks Ranking Size

[Figure 379]

[Figure 380]

[Figure 381]

Grasp the largest block and place it to the left area of the table center.

[Figure 382]

[Figure 383]

[Figure 384]

Grasp the medium block and place it to the middle area of the table center.

[Figure 385]

[Figure 386]

[Figure 387]

Grasp the smallest block and place it to the right area of the table center.

###### Move Can Pot

[Figure 388]

[Figure 389]

[Figure 390]

Move the can to the left side of the pot. Move Playingcard Away

[Figure 391]

[Figure 392]

[Figure 393]

Put the playing cards to the right side area of the right grasp. Blocks Ranking RGB

[Figure 394]

[Figure 395]

[Figure 396]

Grasp the red block and place it to the left area of the table center.

[Figure 397]

[Figure 398]

[Figure 399]

Grasp the green block and place it to the middle area of the table center.

[Figure 400]

[Figure 401]

[Figure 402]

Grasp the blue block and place it to the right area of the table center.

###### Place Mouse Pad

[Figure 403]

[Figure 404]

[Figure 405]

Put the mouse on the yellow pad. Place Object Stand

[Figure 406]

[Figure 407]

[Figure 408]

Put the stapler onto the display stand. Stamp Seal

[Figure 409]

[Figure 410]

[Figure 411]

Grasp the seal and move the seal on the cyan pad. Stack Blocks Two

[Figure 412]

[Figure 413]

[Figure 414]

Move the red block on the front side of the table.

[Figure 415]

[Figure 416]

[Figure 417]

Move the green block on the red block.

###### Place Container Plate

[Figure 418]

[Figure 419]

[Figure 420]

Put the bowl into the plate. Place Empty Cup

[Figure 421]

[Figure 422]

[Figure 423]

Put the cup onto the coaster. Place Fan

[Figure 424]

[Figure 425]

[Figure 426]

Grasp the fan and put the fan onto the lime pad. Place Bread Skillet

[Figure 427]

[Figure 428]

[Figure 429]

Move the skillet to the front of the camera.

[Figure 430]

[Figure 431]

[Figure 432]

Move the bread into the skillet.

###### Place Mouse Pad

[Figure 433]

[Figure 434]

[Figure 435]

Put the mouse on the yellow pad. Place Object Stand

[Figure 436]

[Figure 437]

[Figure 438]

Put the stapler onto the display stand. Stamp Seal

[Figure 439]

[Figure 440]

[Figure 441]

Grasp the seal and move the seal on the cyan pad. Stack Blocks Two

[Figure 442]

[Figure 443]

[Figure 444]

Move the red block on the front side of the table.

[Figure 445]

[Figure 446]

[Figure 447]

Move the green block on the red block.

###### Fig. 31: Visualized RoboTwin simulation evaluation process.

Place Burger Fries

[Figure 448]

[Figure 449]

[Figure 450]

Put the french fries on the right side of the tray.

[Figure 451]

[Figure 452]

[Figure 453]

Put the hamburg on the left side of the tray.

###### Fig. 32: Visualized RoboTwin simulation evaluation process.

### References

- 1. Abdolmaleki, A., Abeyruwan, S., Ainslie, J., Alayrac, J.B., Arenas, M.G., Balakrishna, A., Batchelor, N., Bewley, A., Bingham, J., Bloesch, M., et al.: Gemini robotics 1.5: Pushing the frontier of generalist robots with advanced embodied reasoning, thinking, and motion transfer. arXiv preprint arXiv:2510.03342 (2025)
- 2. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report. arXiv

(2023)

- 3. An, J., Zhu, Y., Li, Z., Zhou, E., Feng, H., Huang, X., Chen, B., Shi, Y., Pan, C.: Agfsync: Leveraging ai-generated feedback for preference optimization in text-toimage generation. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 39, pp. 1746–1754 (2025)
- 4. Anthropic, A.: The claude 3 model family: Opus, sonnet, haiku. Claude-3 Model Card 1, 1 (2024)
- 5. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al.: Qwen2. 5-vl technical report. arXiv (2025)
- 6. Bai, S., Lyu, J., Zhou, W., Li, Z., Wang, D., Xing, L., Zhao, X., Wang, P., Wang, Z., Chi, C., et al.: Latent reasoning vla: Latent thinking and prediction for visionlanguage-action models. arXiv preprint arXiv:2602.01166 (2026)
- 7. Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., et al.: Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv (2022)
- 8. Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Groom, L., Hausman, K., Ichter, B., et al.: π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164 (2024)
- 9. Cai, W., Ponomarenko, I., Yuan, J., Li, X., Yang, W., Dong, H., Zhao, B.: Spatialbot: Precise spatial understanding with vision language models. ICRA (2025)
- 10. Cai, Z., Yeh, C.F., Xu, H., Liu, Z., Meyer, G., Lei, X., Zhao, C., Li, S.W., Chandra, V., Shi, Y.: Depthlm: Metric depth from vision language models. arXiv preprint arXiv:2509.25413 (2025)
- 11. Chen, B., Xu, Z., Kirmani, S., Ichter, B., Sadigh, D., Guibas, L., Xia, F.: Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14455–14465 (2024)
- 12. Chen, P., Lou, Y., Cao, S., Guo, J., Fan, L., Wu, Y., Yang, L., Ma, L., Ye, J.: Sdvlm: Spatial measuring and understanding with depth-encoded vision-language models. arXiv preprint arXiv:2509.17664 (2025)
- 13. Chen, P., Lou, Y., Cao, S., Guo, J., Fan, L., Wu, Y., Yang, L., Ma, L., Ye, J.: Sdvlm: Spatial measuring and understanding with depth-encoded vision-language models (2025), https://arxiv.org/abs/2509.17664
- 14. Chen, T., Chen, Z., Chen, B., Cai, Z., Liu, Y., Li, Z., Liang, Q., Lin, X., Ge, Y., Gu, Z., et al.: Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088 (2025)
- 15. Chen, Z., Shi, Z., Lu, X., He, L., Qian, S., Yin, Z., Ouyang, W., Shao, J., Qiao, Y., Lu, C., et al.: Rh20t-p: A primitive-level robotic dataset towards composable generalization agents. arXiv preprint arXiv:2403.19622 (2024)
- 16. Chen, Z., Zhang, M., Yu, X., Luo, X., Sun, M., Pan, Z., Feng, Y., Pei, P., Cai, X., Huang, R.: Think with 3d: Geometric imagination grounded spatial reasoning from limited views. arXiv preprint arXiv:2510.18632 (2025)

- 17. Cheng, A.C., Fu, Y., Chen, Y., Liu, Z., Li, X., Radhakrishnan, S., Han, S., Lu, Y., Kautz, J., Molchanov, P., et al.: 3d aware region prompted vision language model. arXiv preprint arXiv:2509.13317 (2025)
- 18. Cheng, A.C., Yin, H., Fu, Y., Guo, Q., Yang, R., Kautz, J., Wang, X., Liu, S.: Spatialrgpt: Grounded spatial reasoning in vision language models. NeurIPS

(2024)

- 19. Chi, C., Xu, Z., Feng, S., Cousineau, E., Du, Y., Burchfiel, B., Tedrake, R., Song, S.: Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research 44(10-11), 1684–1704 (2025)
- 20. Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon,

I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al.: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261 (2025)

- 21. contributors, A.W.C.: Agibot world colosseum. https : / / github . com / OpenDriveLab/AgiBot-World (2024)
- 22. Dai, A., Chang, A.X., Savva, M., Halber, M., Funkhouser, T., Nießner, M.: Scannet: Richly-annotated 3d reconstructions of indoor scenes. In: CVPR (2017)
- 23. Daxberger, E., Wenzel, N., Griffiths, D., Gang, H., Lazarow, J., Kohavi, G., Kang, K., Eichner, M., Yang, Y., Dehghan, A., et al.: Mm-spatial: Exploring 3d spatial understanding in multimodal llms. arXiv (2025)
- 24. Deitke, M., Clark, C., Lee, S., Tripathi, R., Yang, Y., Park, J.S., Salehi, M., Muennighoff, N., Lo, K., Soldaini, L., et al.: Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 91–104 (2025)
- 25. Du, M., Wu, B., Li, Z., Huang, X.J., Wei, Z.: Embspatial-bench: Benchmarking spatial understanding for embodied tasks with large vision-language models. In: ACL (Volume 2: Short Papers) (2024)
- 26. Fan, Z., Zhang, J., Li, R., Zhang, J., Chen, R., Hu, H., Wang, K., Qu, H., Wang, D., Yan, Z., et al.: Vlm-3r: Vision-language models augmented with instructionaligned 3d reconstruction. arXiv preprint arXiv:2505.20279 (2025)
- 27. Fang, H.S., Wang, C., Fang, H., Gou, M., Liu, J., Yan, H., Liu, W., Xie, Y., Lu, C.: Anygrasp: Robust and efficient grasp perception in spatial and temporal domains. IEEE Transactions on Robotics (2023)
- 28. Fu, X., Hu, Y., Li, B., Feng, Y., Wang, H., Lin, X., Roth, D., Smith, N.A., Ma, W.C., Krishna, R.: Blink: Multimodal large language models can see but not perceive. In: ECCV (2024)
- 29. Han, Y., Chi, C., Zhou, E., Rong, S., An, J., Wang, P., Wang, Z., Sheng, L., Zhang, S.: Tiger: Tool-integrated geometric reasoning in vision-language models for robotics. arXiv preprint arXiv:2510.07181 (2025)
- 30. Huang, C.P., Wu, Y.H., Chen, M.H., Wang, Y.C.F., Yang, F.E.: Thinkact: Visionlanguage-action reasoning via reinforced visual latent planning. arXiv preprint arXiv:2507.16815 (2025)
- 31. Hudson, D.A., Manning, C.D.: Gqa: A new dataset for real-world visual reasoning and compositional question answering. In: CVPR (2019)
- 32. Ji, Y., Liu, Y., Tan, H., Huang, X., Huang, F., Xu, Y., Chi, C., Zhao, Y., Lyu, H., Co, P., et al.: Prm-as-a-judge: A dense evaluation paradigm for fine-grained robotic auditing. arXiv preprint arXiv:2603.21669 (2026)
- 33. Ji, Y., Tan, H., Shi, J., Hao, X., Zhang, Y., Zhang, H., Wang, P., Zhao, M., Mu, Y., An, P., et al.: Robobrain: A unified brain model for robotic manipulation from abstract to concrete. arXiv preprint arXiv:2502.21257 (2025)

- 34. Kang, L., Song, X., Zhou, H., Qin, Y., Yang, J., Liu, X., Torr, P., Bai, L., Yin, Z.: Viki-r: Coordinating embodied multi-agent cooperation via reinforcement learning. arXiv preprint arXiv:2506.09049 (2025)
- 35. Keetha, N., Müller, N., Schönberger, J., Porzi, L., Zhang, Y., Fischer, T., Knapitsch, A., Zauss, D., Weber, E., Antunes, N., et al.: Mapanything: Universal feed-forward metric 3d reconstruction. arXiv preprint arXiv:2509.13414 (2025)
- 36. Khalifa, M., Agarwal, R., Logeswaran, L., Kim, J., Peng, H., Lee, M., Lee, H., Wang, L.: Process reward models that think. arXiv (2025)
- 37. Khazatsky, A., Pertsch, K., Nair, S., Balakrishna, A., Dasari, S., Karamcheti, S., Nasiriany, S., Srirama, M.K., Chen, L.Y., Ellis, K., et al.: Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945 (2024)
- 38. Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K., Kravitz, J., Chen, S., Kalantidis, Y., Li, L.J., Shamma, D.A., et al.: Visual genome: Connecting language and vision using crowdsourced dense image annotations. IJCV (2017)
- 39. Kuznetsova, A., Rom, H., Alldrin, N., Uijlings, J., Krasin, I., Pont-Tuset, J., Kamali, S., Popov, S., Malloci, M., Kolesnikov, A., et al.: The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. IJCV (2020)
- 40. Lazarow, J., Griffiths, D., Kohavi, G., Crespo, F., Dehghan, A.: Cubify anything: Scaling indoor 3d object detection. arXiv preprint arXiv:2412.04458 (2024)
- 41. Lee, J., Duan, J., Fang, H., Deng, Y., Liu, S., Li, B., Fang, B., Zhang, J., Wang, Y.R., Lee, S., et al.: Molmoact: Action reasoning models that can reason in space. arXiv preprint arXiv:2508.07917 (2025)
- 42. Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Zhang, P., Li, Y., Liu, Z., Li, C.: LLaVA-onevision: Easy visual task transfer. Transactions on Machine Learning Research (2025)
- 43. Li, Y., Deng, Y., Zhang, J., Jang, J., Memmel, M., Yu, R., Garrett, C.R., Ramos, F., Fox, D., Li, A., et al.: Hamster: Hierarchical action models for open-world robot manipulation. arXiv preprint arXiv:2502.05485 (2025)
- 44. Li, Z., Chi, C., Wei, Y., Zhu, B., Huang, T., Sun, Z., Peng, Y., Wang, P., Wang, Z., Liu, F., et al.: Do you have freestyle? expressive humanoid locomotion via audio control. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 956–965 (2026)
- 45. Li, Z., Chi, C., Zhu, B., Wei, Y., Bai, S., Ji, Y., Peng, Y., Huang, T., Wang, P., Wang, Z., et al.: Robomirror: Understand before you imitate for video to humanoid locomotion. arXiv preprint arXiv:2512.23649 (2025)
- 46. Liao, Y.H., Mahmood, R., Fidler, S., Acuna, D.: Reasoning paths with reference objects elicit quantitative spatial reasoning in large vision-language models

(2024), https://arxiv.org/abs/2409.09788

- 47. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: ECCV (2014)
- 48. Liu, F., Emerson, G., Collier, N.: Visual spatial reasoning. Transactions of the Association for Computational Linguistics (2023)
- 49. Liu, F., Lin, K., Li, L., Wang, J., Yacoob, Y., Wang, L.: Mitigating hallucination in large multi-modal models via robust instruction tuning. In: ICLR (2024)
- 50. Liu, H., Li, C., Li, Y., Lee, Y.J.: Improved baselines with visual instruction tuning. In: CVPR (2024)
- 51. Liu, M., Zhou, E., Chi, C., Han, Y., Rong, S., Chen, L., Wang, P., Wang, Z., Zhang, S.: Sapave: Towards active perception and manipulation in vision-language action models for robotics. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 37164–37174 (2026)

- 52. Liu, R., Gao, J., Zhao, J., Zhang, K., Li, X., Qi, B., Ouyang, W., Zhou, B.: Can 1b llm surpass 405b llm? rethinking compute-optimal test-time scaling. arXiv (2025)
- 53. Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Jiang, Q., Li, C., Yang, J., Su, H., et al.: Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In: ECCV (2024)
- 54. Liu, S., Wu, L., Li, B., Tan, H., Chen, H., Wang, Z., Xu, K., Su, H., Zhu, J.: Rdt-1b: a diffusion foundation model for bimanual manipulation. arXiv preprint arXiv:2410.07864 (2024)
- 55. Liu, Y., Peng, B., Zhong, Z., Yue, Z., Lu, F., Yu, B., Jia, J.: Seg-zero: Reasoningchain guided segmentation via cognitive reinforcement. arXiv (2025)
- 56. Liu, Z., Zhu, L., Shi, B., Zhang, Z., Lou, Y., Yang, S., Xi, H., Cao, S., Gu, Y., Li, D., et al.: Nvila: Efficient frontier visual language models. arXiv (2024)
- 57. Liu, Z., Sun, Z., Zang, Y., Dong, X., Cao, Y., Duan, H., Lin, D., Wang, J.: Visualrft: Visual reinforcement fine-tuning. arXiv (2025)
- 58. Lu, X., Chen, Z., Hu, X., Zhou, Y., Zhang, W., Liu, D., Sheng, L., Shao, J.: Is-bench: Evaluating interactive safety of vlm-driven embodied agents in daily household tasks. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 40, pp. 35680–35688 (2026)
- 59. Lu, X., Zhou, Y., Chen, Z., Wang, R., Sima, B., Zhou, E., Sheng, L., Liu, D., Shao, J.: Homeguard: Vlm-based embodied safeguard for identifying contextual risk in household task. arXiv preprint arXiv:2603.14367 (2026)
- 60. Majumdar, A., Ajay, A., Zhang, X., Putta, P., Yenamandra, S., Henaff, M., Silwal, S., Mcvay, P., Maksymets, O., Arnaud, S., et al.: Openeqa: Embodied question answering in the era of foundation models. In: CVPR (2024)
- 61. Mao, Y., Zhong, J., Fang, C., Zheng, J., Tang, R., Zhu, H., Tan, P., Zhou, Z.: Spatiallm: Training large language models for structured indoor modeling. arXiv preprint arXiv:2506.07491 (2025)
- 62. Mishra, A., Shekhar, S., Singh, A.K., Chakraborty, A.: Ocr-vqa: Visual question answering by reading text in images. In: ICDAR (2019)
- 63. Niu, D., Sharma, Y., Biamby, G., Quenum, J., Bai, Y., Shi, B., Darrell, T., Herzig, R.: Llarva: Vision-action instruction tuning enhances robot learning. arXiv preprint arXiv:2406.11815 (2024)
- 64. Piccinelli, L., Sakaridis, C., Yang, Y.H., Segu, M., Li, S., Abbeloos, W., Van Gool, L.: Unidepthv2: Universal monocular metric depth estimation made simpler. arXiv (2025)
- 65. Qin, Y., Shi, Z., Yu, J., Wang, X., Zhou, E., Li, L., Yin, Z., Liu, X., Sheng, L., Shao, J., et al.: Worldsimbench: Towards video generation models as world simulators. arXiv preprint arXiv:2410.18072 (2024)
- 66. Qin, Y., Zhou, E., Liu, Q., Yin, Z., Sheng, L., Zhang, R., Qiao, Y., Shao, J.: Mp5: A multi-modal open-ended embodied system in minecraft via active perception. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16307–16316 (2024)
- 67. Rafailov, R., Sharma, A., Mitchell, E., Manning, C.D., Ermon, S., Finn, C.: Direct preference optimization: Your language model is secretly a reward model. NeurIPS

(2023)

- 68. Ravi, N., Gabeur, V., Hu, Y.T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., et al.: Sam 2: Segment anything in images and videos. ICLR (2025)
- 69. Ray, A., Duan, J., Tan, R., Bashkirova, D., Hendrix, R., Ehsani, K., Kembhavi, A., Plummer, B.A., Krishna, R., Zeng, K.H., et al.: Sat: Spatial aptitude training for multimodal language models. arXiv (2024)

- 70. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O.: Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 (2017)
- 71. Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al.: Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv (2024)
- 72. Shen, H., Liu, P., Li, J., Fang, C., Ma, Y., Liao, J., Shen, Q., Zhang, Z., Zhao, K., Zhang, Q., et al.: Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615 (2025)
- 73. Singh, A., Natarjan, V., Shah, M., Jiang, Y., Chen, X., Parikh, D., Rohrbach, M.: Towards vqa models that can read. In: CVPR (2019)
- 74. Song, C.H., Blukis, V., Tremblay, J., Tyree, S., Su, Y., Birchfield, S.: Robospatial: Teaching spatial understanding to 2d and 3d vision-language models for robotics. CVPR (2025)
- 75. Sun, P., Lang, S., Wu, D., Ding, Y., Feng, K., Liu, H., Ye, Z., Liu, R., Liu, Y.H., Wang, J., et al.: Spacevista: All-scale visual spatial reasoning from mm to km. arXiv preprint arXiv:2510.09606 (2025)
- 76. Tan, H., Chi, C., Chen, X., Ji, Y., Zhao, Z., Hao, X., Lyu, Y., Cao, M., Zhao, J., Lyu, H., et al.: Roboos-next: A unified memory-based framework for lifelong, scalable, and robust multi-robot collaboration. arXiv preprint arXiv:2510.26536

(2025)

- 77. Tan, H., Hao, X., Chi, C., Lin, M., Lyu, Y., Cao, M., Liang, D., Chen, Z., Lyu, M., Peng, C., et al.: Roboos: A hierarchical embodied framework for crossembodiment and multi-agent collaboration. arXiv preprint arXiv:2505.03673

(2025)

- 78. Tan, H., Ji, Y., Hao, X., Lin, M., Wang, P., Wang, Z., Zhang, S.: Reason-rft: Reinforcement fine-tuning for visual reasoning. arXiv (2025)
- 79. Tan, H., Zhou, E., Li, Z., Xu, Y., Ji, Y., Chen, X., Chi, C., Wang, P., Jia, H., Ao, Y., et al.: Robobrain 2.5: Depth in sight, time in mind. arXiv preprint arXiv:2601.14352 (2026)
- 80. Team, B.R., Cao, M., Tan, H., Ji, Y., Chen, X., Lin, M., Li, Z., Cao, Z., Wang, P., Zhou, E., et al.: Robobrain 2.0 technical report. arXiv preprint arXiv:2507.02029

(2025)

- 81. Team, G., Anil, R., Borgeaud, S., Alayrac, J.B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A.M., Hauth, A., Millican, K., et al.: Gemini: a family of highly capable multimodal models. arXiv (2023)
- 82. Team, G.R., Abeyruwan, S., Ainslie, J., Alayrac, J.B., Arenas, M.G., Armstrong, T., Balakrishna, A., Baruch, R., Bauza, M., Blokzijl, M., et al.: Gemini robotics: Bringing ai into the physical world. arXiv (2025)
- 83. Team, Q.: Qwen3 technical report (2025), https://arxiv.org/abs/2505.09388
- 84. Team, Q.: Qwq-32b: Embracing the power of reinforcement learning (March 2025), https://qwenlm.github.io/blog/qwq-32b/
- 85. Tong, P., Brown, E., Wu, P., Woo, S., IYER, A.J.V., Akula, S.C., Yang, S., Yang, J., Middepogu, M., Wang, Z., et al.: Cambrian-1: A fully open, vision-centric exploration of multimodal llms. NeurIPS (2024)
- 86. Tschannen, M., Gritsenko, A., Wang, X., Naeem, M.F., Alabdulmohsin, I., Parthasarathy, N., Evans, T., Beyer, L., Xia, Y., Mustafa, B., et al.: Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv (2025)
- 87. Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., Novotny, D.: Vggt: Visual geometry grounded transformer. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 5294–5306 (2025)

- 88. Wang, R., Xu, S., Dong, Y., Deng, Y., Xiang, J., Lv, Z., Sun, G., Tong, X., Yang, J.: Moge-2: Accurate monocular geometry with metric scale and sharp details. arXiv preprint arXiv:2507.02546 (2025)
- 89. Wang, Y., Ji, Y., Liu, Y., Zhou, E., Yang, Z., Tian, Y., Qin, Z., Liu, Y., Tan, H., Chi, C., et al.: Towards cross-view point correspondence in vision-language models. arXiv preprint arXiv:2512.04686 (2025)
- 90. Wei, J., Bosma, M., Zhao, V.Y., Guu, K., Yu, A.W., Lester, B., Du, N., Dai, A.M., Le, Q.V.: Finetuned language models are zero-shot learners. arXiv (2021)
- 91. Wu, D., Liu, F., Hung, Y.H., Duan, Y.: Spatial-mllm: Boosting mllm capabilities in visual-based spatial intelligence. arXiv preprint arXiv:2505.23747 (2025)
- 92. Xu, R., Zhang, J., Guo, M., Wen, Y., Yang, H., Lin, M., Huang, J., Li, Z., Zhang, K., Wang, L., et al.: A0: An affordance-aware hierarchical model for general robotic manipulation. arXiv preprint arXiv:2504.12636 (2025)
- 93. Yang, G., Zhang, T., Hao, H., Wang, W., Liu, Y., Wang, D., Chen, G., Cai, Z., Chen, J., Su, W., et al.: Vlaser: Vision-language-action model with synergistic embodied reasoning. arXiv preprint arXiv:2510.11027 (2025)
- 94. Yang, J., Tan, R., Wu, Q., Zheng, R., Peng, B., Liang, Y., Gu, Y., Cai, M., Ye, S., Jang, J., et al.: Magma: A foundation model for multimodal ai agents. arXiv

(2025)

- 95. Yang, Y., He, X., Pan, H., Jiang, X., Deng, Y., Yang, X., Lu, H., Yin, D., Rao, F., Zhu, M., et al.: R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv (2025)
- 96. Yu, L., Poirson, P., Yang, S., Berg, A.C., Berg, T.L.: Modeling context in referring expressions. In: ECCV (2016)
- 97. Yuan, W., Duan, J., Blukis, V., Pumacay, W., Krishna, R., Murali, A., Mousavian, A., Fox, D.: Robopoint: A vision-language model for spatial affordance prediction for robotics. arXiv preprint arXiv:2406.10721 (2024)
- 98. Yuan, Y., Cui, H., Chen, Y., Dong, Z., Ni, F., Kou, L., Liu, J., Li, P., Zheng, Y., Hao, J.: From seeing to doing: Bridging reasoning and decision for robotic manipulation. arXiv preprint arXiv:2505.08548 (2025)
- 99. Yuan, Y., Cui, H., Huang, Y., Chen, Y., Ni, F., Dong, Z., Li, P., Zheng, Y., Hao, J.: Embodied-r1: Reinforced embodied reasoning for general robotic manipulation. arXiv preprint arXiv:2508.13998 (2025)
- 100. Yuan, Y., Cui, H., Huang, Y., Chen, Y., Ni, F., Dong, Z., Li, P., Zheng, Y., Hao, J.: Embodied-r1: Reinforced embodied reasoning for general robotic manipulation

(2025), https://arxiv.org/abs/2508.13998

- 101. Ze, Y., Zhang, G., Zhang, K., Hu, C., Wang, M., Xu, H.: 3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations. arXiv preprint arXiv:2403.03954 (2024)
- 102. Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L.: Sigmoid loss for language image pre-training. In: CVPR (2023)
- 103. Zhan, Y., Zhu, Y., Zheng, S., Zhao, H., Yang, F., Tang, M., Wang, J.: Vision-r1: Evolving human-free alignment in large vision-language models via vision-guided reinforcement learning. arXiv (2025)
- 104. Zhang, J., Chen, K., Lu, Z., Zhou, E., Yu, Q., Zhang, J.: Prune4web: Dom tree pruning programming for web agent. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 40, pp. 34710–34718 (2026)
- 105. Zhang, J., Huang, J., Yao, H., Liu, S., Zhang, X., Lu, S., Tao, D.: R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization. arXiv (2025)

- 106. Zhang, Y., Huang, X., Ma, J., Li, Z., Luo, Z., Xie, Y., Qin, Y., Luo, T., Li, Y., Liu, S., et al.: Recognize anything: A strong image tagging model. In: CVPR (2024)
- 107. Zhang, Z., Wu, Y., Jia, L., Wang, Y., Zhang, Z., Li, Y., Ran, B., Zhang, F., Sun, Z., Yin, Z., et al.: Think3d: Thinking with space for spatial reasoning. arXiv preprint arXiv:2601.13029 (2026)
- 108. Zhao, T.Z., Kumar, V., Levine, S., Finn, C.: Learning fine-grained bimanual manipulation with low-cost hardware (2023), https://arxiv.org/abs/2304.13705
- 109. Zheng, D., Huang, S., Li, Y., Wang, L.: Learning from videos for 3d world: Enhancing mllms with 3d vision geometry priors. arXiv preprint arXiv:2505.24625

(2025)

- 110. Zheng, R., Liang, Y., Huang, S., Gao, J., Daumé III, H., Kolobov, A., Huang, F., Yang, J.: Tracevla: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies. arXiv preprint arXiv:2412.10345 (2024)
- 111. Zhou, C., Liu, P., Xu, P., Iyer, S., Sun, J., Mao, Y., Ma, X., Efrat, A., Yu, P., Yu, L., et al.: Lima: Less is more for alignment. NeurIPS (2023)
- 112. Zhou, E., An, J., Chi, C., Han, Y., Rong, S., Zhang, C., Wang, P., Wang, Z., Huang, T., Sheng, L., et al.: Roborefer: Towards spatial referring with reasoning in vision-language models for robotics. arXiv preprint arXiv:2506.04308 (2025)
- 113. Zhou, E., Qin, Y., Yin, Z., Huang, Y., Zhang, R., Sheng, L., Qiao, Y., Shao, J.: Minedreamer: Learning to follow instructions via chain-of-imagination for simulated-world control. arXiv preprint arXiv:2403.12037 (2024)
- 114. Zhou, E., Qin, Y., Yin, Z., Shi, Z., Huang, Y., Zhang, R., Sheng, L., Shao, J.: Chain-of-imagination for reliable instruction following in decision making. In: 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). pp. 10010–10017. IEEE (2025)
- 115. Zhou, E., Su, Q., Chi, C., Zhang, Z., Wang, Z., Huang, T., Sheng, L., Wang, H.: Code-as-monitor: Constraint-aware visual programming for reactive and proactive robotic failure detection. arXiv preprint arXiv:2412.04455 (2024)
- 116. Zhu, S., Kumar, A., Hu, M., Liu, X.: Tame a wild camera: in-the-wild monocular camera calibration. NeurIPS (2023)

