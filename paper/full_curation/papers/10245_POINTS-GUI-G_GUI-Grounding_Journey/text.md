## arXiv:2602.06391v1[cs.CV]6Feb2026

[Figure 1]

# POINTS-GUI-G: GUI-Grounding Journey

##### Zhongyin Zhao∗1, Yuan Liu∗†1, Yikun Liu1, Haicheng Wang1, Le Tian1, Xiao Zhou1, Yangxiu You1, Zilin Yu1, Yang Yu1, Jie Zhou1 1WeChat AI ∗Equal Contribution, †Corresponding Author

Github: https://github.com/Tencent/POINTS-GUI HuggingFace: https://huggingface.co/tencent/POINTS-GUI-G

[Figure 2]

[Figure 3]

The rapid advancement of vision-language models has catalyzed the emergence of GUI agents, which hold immense potential for automating complex tasks—from online shopping to flight booking—thereby alleviating the burden of repetitive digital workflows. As a foundational capability, GUI grounding is typically established as a prerequisite for end-to-end task execution. It enables models to precisely locate interface elements, such as text and icons, to perform accurate operations like clicking and typing. Unlike prior works that fine-tune models already possessing strong spatial awareness (e.g., Qwen3-VL), we aim to master the full technical pipeline by starting from a base model with minimal grounding ability, such as POINTS-1.5. We introduce POINTS-GUI-G-8B, which achieves state-of-the-art performance with scores of 59.9 on ScreenSpot-Pro, 66.0 on OSWorld-G, 95.7 on ScreenSpot-v2, and 49.9 on UI-Vision. Our model’s success is driven by three key factors: (1) Refined Data Engineering, involving the unification of diverse open-source datasets format alongside sophisticated strategies for augmentation, filtering, and difficulty grading; (2) Improved Training Strategies, including continuous fine-tuning of the vision encoder to enhance perceptual accuracy and maintaining resolution consistency between training and inference; and (3) Reinforcement Learning (RL) with Verifiable Rewards. While RL is traditionally used to bolster reasoning, we demonstrate that it significantly improves precision in the perception-intensive GUI grounding task. Furthermore, GUI grounding provides a natural advantage for RL, as rewards are easily verifiable and highly accurate.

Date: Feb 6, 2026 Correspondence: bensenliu@tencent.com

### 1 Introduction

Driven by rapid advancements in Vision-Language Models (VLMs) (Liu et al., 2024g,e,f; Yang et al., 2025a; Zhu et al., 2025; Wang et al., 2025b; Zeng et al., 2025; Bai et al., 2025b; Team et al., 2025; Li et al., 2024; Liu et al., 2024b), Graphical User Interface (GUI) agents have emerged as a transformative research frontier. Deploying these agents in real-world production environments enables the automation of complex tasks—ranging from ticket booking to online shopping—thereby streamlining daily workflows. Compared to MCP-based agents, GUI agents (Qin et al., 2025; Wang et al., 2025a; Ye et al., 2025; Zhou et al., 2025; Gu et al., 2025b; Yan et al., 2025) offer broader applicability as they interact directly with interfaces without requiring modifications to existing internet infrastructure. To execute tasks effectively, an agent must decompose high-level goals into discrete steps, each requiring the precise localization of interface elements like buttons and input fields. Consequently, GUI grounding is a fundamental prerequisite. While current research often leverages models with pre-existing, well-optimized grounding capabilities (e.g., Qwen3-VL) to focus on end-to-end execution, this approach may overlook critical technical insights essential for iterative development. To achieve a comprehensive, “full-stack” mastery of GUI agent development, we deliberately start with POINTS-1.5, a

model lacking native grounding capabilities. We build the GUI grounding capability for the WePOINTS series from the ground up through three pillars: 1) refined data engineering, 2) precise training strategies, and 3) reinforcement learning (RL) with verifiable rewards.

- Figure 1 Comparison with existing models of comparable size on ScreenSpot-v2 (Wu et al., 2024), SreenSpot-Pro (Li et al., 2025), MMBench-GUI-L2 (Wang et al., 2025d), and OSWorld-G (Xie et al., 2025).

Refined Data Engineering We implement a multi-stage data engineering pipeline to address the limitations of existing GUI grounding datasets. 1) Standardization: While numerous open-source datasets (Feizi et al., 2025; Mu et al., 2025; Chen et al., 2025c; Xie et al., 2025; Liu et al., 2024c; Wu et al., 2024; Cheng et al., 2024; Bai et al., 2021; Wu et al., 2023) enhance agent grounding, they exhibit significant heterogeneity in coordinate scales (e.g., normalized vs. raw pixels) and task formats. We unify these by normalizing all coordinates to a [0,1] range and reformatting diverse tasks into a consistent UI element localization task, thereby mitigating training interference. 2) Noise Reduction: Our analysis reveals that existing datasets—whether constructed via URL crawling (e.g., from FineWeb (Penedo et al., 2024)) or model-based annotation of screenshots (e.g., RICO)—often contain significant noise. To ensure data quality, we developed an automated filtering solution to prune inaccurate annotations. 3) Complexity Enhancement: To push the model’s performance ceiling, we increase training difficulty by filtering out “simple” samples—those with sparse layouts or oversized clickable regions—using metrics such as element layout entropy and occupancy ratios. Furthermore, we synthesize challenging data by: (i) rendering professional software with complex, small-scale elements, and (ii) layering multiple pages onto a desktop-style interface to introduce realistic distractors.

Improved Training Strategies In previous iterations of the WePOINTS series (Liu et al., 2024e, 2025c), the vision encoder was kept frozen during training. However, GUI grounding is a perception-heavy task where the quality of extracted image features directly determines model performance. Our experiments demonstrate that general-purpose vision encoders, such as Qwen2VL-ViT (Wang et al., 2024), are not fully optimized for this specialized domain. Consequently, we fully unfreeze the vision encoder during training, leading to substantial performance gains. Additionally, we address the often-overlooked factor of image resolution. Previously, in order to accelerate training and save memory, we restrict the maximum image resolution below 2000 × 2000 pixels. As a result, models trained on these data often suffer from significant performance degradation when encountering higher-resolution images, such as those in ScreenSpot-Pro (Li et al., 2025), during inference. To mitigate this, we propose two strategies: (1) increasing the maximum training resolution cap to 3072 × 3072 and (2) constraining the inference resolution to under 2000 × 2000 pixels. Together, these strategies yielded a

performance boost of over 10 points on the ScreenSpot-Pro benchmark.

Reinforcement Learning with Verifiable Rewards Reinforcement Learning with Verifiable Rewards (RLVR) has been primarily utilized to enhance the reasoning and agentic capabilities of large-scale models (Shao et al.,

- 2024; Meng et al., 2025; Zeng et al., 2025; Chen et al., 2025a). Compared to domains such as mathematical theorem proving or open-ended generation, GUI grounding offers distinct advantages for reinforcement learning. The output space is highly constrained—consisting of points or bounding boxes—enabling the definition of precise, objective reward functions. Diverging from certain prior approaches (Chen et al., 2025b) and following GTA1 (Yang et al., 2025b), we do not require the model to generate a reasoning trace prior to outputting the target coordinates. Although GUI grounding is fundamentally a perception-based task, we observe that RLVR yields consistent and substantial improvements over the supervised baseline. Furthermore, we implement a curriculum-based strategy, progressively increasing the difficulty of training samples to stabilize and enhance the learning process.

In summary, we present POINTS-GUI-G, a new state-of-the-art model for GUI grounding. Our primary contributions are as follows:

- • Unified Data Engineering: We reformat existing large-scale open-source GUI grounding datasets into a unified task formulation and annotation standard. Furthermore, we implement an automated pipeline for comprehensive data cleaning and introduce novel data construction strategies designed to enhance task complexity.
- • Resolution Optimization: We revisit critical but previously overlooked factors in GUI grounding, specifically the image resolution mismatch between training and inference. We propose effective strategies to mitigate this discrepancy, resulting in substantial performance gains on high-resolution benchmarks such as ScreenSpot-Pro.
- • Reinforcement Learning: We demonstrate the effectiveness of applying Reinforcement Learning (RL) with verifiable rewards to GUI grounding tasks, achieving consistent and significant improvements across benchmarks.

We hope POINTS-GUI-G serves as a robust foundation for future GUI agent development and that our findings provide new perspectives on enhancing GUI grounding capabilities.

- 2 Related Works

Vision-Language Models. Recent vision-language models (VLMs) have advanced rapidly across architectural efficiency and scaling (Liu et al., 2024a,g,e; Bai et al., 2025a; Wang et al., 2024; Bai et al., 2025b; Zhu et al.,

- 2025; Wang et al., 2025b). Early milestones like BLIP-2 (Li et al., 2023) minimized training overhead by optimizing only the Q-Former, while LLaVA (Liu et al., 2023a) streamlined modality alignment through large-scale instruction tuning. Subsequent research has focused on visual fidelity; for instance, InternLMXComposer2 and CogVLM (Chen et al., 2024; Zhang et al., 2024) utilize image tiling to handle higher resolutions. More recently, Qwen2-VL (Wang et al., 2024) and Qwen2.5-VL (Bai et al., 2025b) implemented NaViT-style (Dehghani et al., 2023) encoders to natively support arbitrary aspect ratios and resolutions. This progress has been further catalyzed by the development of comprehensive evaluation benchmarks like MMBench (Liu et al., 2024d).

GUI Grounding GUI grounding evaluates a model’s ability to map natural language instructions to precise coordinates within a user interface. As a fundamental ability of GUI agents, it serves as a critical prerequisite for high-level task execution. Conventional approaches (Wu et al., 2024; Qin et al., 2025; Wang et al., 2025a; Xu et al., 2024; Xie et al., 2025; Gou et al., 2024; Yang et al., 2025c) primarily optimize this capability through extensive Supervised Fine-Tuning (SFT). However, following the success of Reinforcement Learning (RL) (Shao et al., 2024; Schulman et al., 2017) in enhancing reasoning for domains such as mathematics and coding, several pioneering studies (Luo et al., 2025; Wu et al., 2025; Yang et al., 2025b) have begun integrating RL into the GUI grounding pipeline. These efforts have demonstrated substantial performance gains over SFT-only baselines—a finding consistent with the empirical observations made during the development of POINTS-GUI-G.

### 3 Methods

This section is organized into three parts. First, subsection 3.1 details our data engineering pipeline across three dimensions: (i) preprocessing operations applied to existing large-scale open-source GUI grounding datasets, (ii) filtering strategies used to curate a high-quality subset, and (iii) a complexity enhancement strategy designed to facilitate more effective learning in the later stages of training. Next, subsection 3.2 revisits critical but often overlooked aspects of the training process that significantly impact final model performance. Finally, subsection 3.3 demonstrates how Reinforcement Learning with Verifiable Rewards (RLVR) further elevates the performance ceiling and describes the construction of the dataset used for this stage.

#### 3.1 Data Engineering

- Figure 2 illustrates the three-step data engineering pipeline.

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Anno format

[Figure 13]

[Figure 14]

[Figure 15]

……

[Figure 16]

[Figure 17]

OmniParser-v2

[Figure 18]

Filter and Synthesis

Spatial scale

Heterogeneous

Homogeneous

Clean

Difficult

Datasets

Datasets

Datasets

Datasets

- Figure 2 The three-stage data engineering pipeline: Data preprocessing, data filtering, and complexity enhancement.

Data Preprocessing Existing open-source GUI grounding datasets (Wu et al., 2024, 2023; Bai et al., 2021; Cheng et al., 2024; Liu et al., 2024c) are highly heterogeneous, characterized by disparate organizational structures, instruction sets, and annotation formats (e.g., varying between list-based and tag-based representations like <box></box>). Furthermore, spatial scales vary significantly, ranging from normalized [0,1] intervals to raw pixel dimensions. Such inconsistencies complicate the effective utilization of these resources. We think that GUI grounding is fundamentally a perception-centric task; therefore, optimization should prioritize spatial localization rather than being confounded by auxiliary capabilities like instruction following. To address this, we standardized all instructions into two primary categories (if the original task is GUI grounding task): bounding box prediction and center point localization (Prompt is given in subsection A.1). Additionally, we unified all spatial annotations to a [0,1] scale with three-decimal precision, formatted as consistent lists or tuples. Finally, the obtained homogeneous datasets are used to enhance the GUI grounding ability. Figure 3 presents examples of data samples reformatted from existing open-source datasets.

Data Filtering The construction of existing open-source datasets typically follows two paradigms: leveraging high-performance models for screenshot annotation (Xie et al., 2025), or utilizing attribute trees from web pages and applications to derive element coordinates (Wu et al., 2024). Since both approaches inherently introduce noise, we adopt the strategy of recent studies (Ye et al., 2025; Chen et al., 2025b; Zhou et al., 2025) by employing OmniParser-v2 (Lu et al., 2024) to extract UI element coordinates, such as clickable text and icons. Our filtering pipeline handles two annotation formats: points and bounding boxes. For point-format annotations (x,y), we first expand them into a square bounding box Bgt with side length l. To validate these annotations, we calculate a coverage score S relative to the set of n elements Bdet = {Bdet,1,Bdet,2,...,Bdet,n} detected by OmniParser-v2. This metric is defined as:

n i=1 Area(Bgt ∩ Bdet,i)

S =

Area(Bgt)

(1)

where Bgt ∩ Bdet,i denotes the intersection between the ground-truth annotation and the i-th detected UI element. We retain only samples where S ≥ τ, where τ represents a predefined reliability threshold. This

user: <image>Please complete the

[Figure 19]

[Figure 20]

[Figure 21]

system: You are a GUI agent. Based on the UI screenshot provided, please locate the exact position of the element that matches the

user: Is there a way for me to report an invasive species on this website? Please

following task: Determine the position of 'SOLAR' within the image.

grounding all object in this web page. <image>

assistant: (0.239, 0.124) user: Please complete the following task: Pinpoint the location of 'MY PLANET' in the image. assistant: (0.833, 0.087) user: Please complete the following task: Spot 'My Planet's name is' in the image. assistant: (0.833, 0.101)

instruction given by the user. Requirements for the output:

- - Return only the point (x, y) representing the center of the target element
- - Coordinates must be normalized to the range [0, 1]
- - Round each coordinate to three decimal places
- - Format the output as strictly (x, y) without any additional text user: <image>Please complete the following task: Spot 'Text' in the image. assistant: (0.342, 0.38) …

assistant: Yes, there is a way for you to report an invasive species on the website. At the top of the webpage, there is a link that reads "Report an Invasive Species" [0.76, 0.01, 0.98, 0.04]. This suggests that the website has a dedicated feature

user: Please complete the following task: Search for the position of 'Made out of:' in the image.

allowing users to report sightings or concerns about invasive species. To report an invasive species, you would likely

assistant: (0.772, 0.145) user: Please complete the following task: Trace the location of 'Number of rings:' in the image. assistant: (0.839, 0.145)

click on this link, which would presumably take you to a page or form where you can submit your report.

[Figure 22]

[Figure 23]

[Figure 24]

system: Operating as a GUI element localization agent, your primary function is to process screenshots and element descriptions to identify the exact position of interface components. Your output will be a set of four normalized coordinates (x1, y1, x2, y2) representing the top-left and bottom-

right corners of the bounding box, with each value rounded to three decimal places.

user: <image> Positional details of the Avatar: The avatar is positioned in the top-right corner of the application window. It is directly to the right of the search bar and above the main content area, which displays messages and notifications. The avatar is part of the header section, which spans the width of the application. assistant: (0.210, 0.386, 0.229, 0.419)

user: Picture 1: <image> Please generate a description for the element

user: <image>open parameter configuration assistant: [0.046, 0.078]

at (0.940,0.502). assistant: save option

- Figure 3 Reformatted samples from existing open-source datasets. All spatial annotations are unified to a [0, 1] scale.

refinement ensures our training data aligns strictly with perceivable visual structures, effectively pruning hallucinated or misaligned coordinates.

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

- Figure 4 Some samples filtered by our proposed filtering strategy. We place the original annotation onto the image.

Complexity Enhancement As models evolve, the original datasets often fail to sustain continuous performance gains. Figure 5 shows some easy samples from existing datasets. Consequently, it is essential to increase data complexity to drive more effective model learning. In this work, we propose a complexity filtering strategy based on layout entropy. Specifically, we define Layout Entropy, Elayout, to quantify the geometric complexity of a UI screen, containing a set of N elements with location bounding boxes B = {b1,...,bN} within a normalized scale [0,1]2. The total entropy is formulated as:

· w1DH¯1D + w2DH2D (2) where wN,w1D,w2D are weighting coefficients. H¯1D, H2D are defined as follows:

Elayout = Nw

N

- • 1D Projection Entropy (H¯1D): This metric captures the alignment and distribution density of elements from multiple perspectives. Given N bounding boxes with center points {(xn,yn)}Nn=1 ∈ [0,1]2, we define D uniform projection directions. Let the vertical axis be the reference direction (θ1 = 0), the j-th projection

angle is defined as θj = (j−D1)π. For each direction θj, we define a projection unit vector uj = (sinθj,cosθj). The projection of the n-th center point onto this direction is calculated as:

zn,j = xn sinθj + yn cosθj (3)

The projection direction is then partitioned into B intervals (bins), denoted as {b1,j,...,bB,j}. The probability pi,j that a box falls into the i-th bin is:

N

1 N

I(zn,j ∈ bi,j) (4)

pi,j =

n=1

where I(·) is the indicator function. The entropy for direction θj is H1D(θj) = − Bi=1 pi,j log pi,j, and the final 1D entropy is the average across all directions:

1 D

H¯1D =

D

H1D(θj) (5)

j=1

- • 2D Grid Entropy (H2D): To evaluate the global spatial dispersion of elements, the screen is splitted into M × M grids. Similar to the probability estimation in the 1D projection, the probability pg for each cell g is defined as the proportion of bounding box centers falling within its spatial boundaries:

1 N

pg =

The 2D spatial entropy is then formulated as:

N

I((xn,yn) ∈ cellg) (6)

n=1

G

H2D = −

g=1

pg log pg (7)

A lower H2D indicates that elements are concentrated in a few grid cells, while a higher value signifies a more uniform and complex distribution across the interface.

After calculating the Layout Entropy Elayout for all data, we utilize this metric to categorize the dataset into three difficulty levels (easy, medium, and hard).

For image resolution, we differentiate samples based on their total pixels S = W × H. High resolution samples are prioritized as they typically contain higher information density and fine-grained visual details.

Regarding data synthesis, we introduce two strategies designed to increase task complexity. 1) GUI-CodeGen: We leverage the sophisticated layouts of professional software—such as VS Code—which are characterized by a high density of functional components. Utilizing Large Language Models (e.g., Claude), we synthesize the underlying frontend HTML for these interfaces, extract operable elements like icons and input fields, and render them into high-resolution images (e.g. 1920 × 2560). 2) GUI-Overlay: To simulate real-world scenarios, we overlay multiple application windows onto diverse desktop backgrounds. This method introduces significant visual distractors and occlusions, challenging the model’s ability to accurately identify and interact with target elements amidst background interference.

#### 3.2 Training Strategies

Adapting Vision Encoder to GUI Scenario Historically, the WePOINTS series—including POINTSReader (Liu et al., 2025c) and POINTS-1.5 (Liu et al., 2024e)—maintained a frozen vision encoder, which yielded strong performance across various tasks. However, we observed that for GUI grounding, existing encoders such as Qwen2-VL-NaViT (Wang et al., 2024) are no longer sufficient. This limitation likely stems from the relative scarcity of GUI-specific data during the encoder’s initial pre-training. To address this, we unfreeze the vision encoder throughout the training process, which resulted in substantial improvements in model performance.

Maintaining Resolution Consistency To optimize memory usage and training efficiency, we initially capped the training image resolution at 2000 × 2000 pixels. However, no such constraints were applied during inference. We observed that this resolution discrepancy led to suboptimal performance on benchmarks

[Figure 29]

[Figure 30]

system: You are an AI assistant trained to locate UI elements in graphical interfaces. Your goal is to analyze screenshots

and element descriptions to determine precise bounding box coordinates (x, y, width, height).

user: <image> Regarding the Card's position: The card is located on the right side of the screen, within a vertical list of booking details. It is positioned at the top of this

list, directly above the booking date and time information.

This Card allows users to: The card provides details about the venue for a booking, including the name and location. It serves as an informational component to help users identify the place associated with

user: <image>expand the product information.

their reservation. assistant: (0.664, 0.191, 0.208, 0.118)

assistant: (0.261, 0.213)

[Figure 31]

[Figure 32]

system: You are an intelligent interface analysis system designed to locate UI elements in screenshots. Your task is to process visual and descriptive

information to output accurate bounding box coordinates (x, y, width, height).

user: <image> The visual attributes of this Dropdown menu are: The element is a dropdown menu labeled

with the text "English" followed by a downward-facing arrow. It is styled with

a simple, clean font in black, indicating the current language selection. The arrow is a small, black triangle pointing downwards, suggesting the ability to

expand the menu to select other languages.

assistant: (0.079, 0.866, 0.126, 0.045)

user: <image>Google Chrome

assistant: (0.139, 0.018)

- Figure 5 Easy data samples from existing datasets filtered out by our complexity enhancement strategy.

featuring high-resolution images, such as ScreenSpot-Pro (Li et al., 2025). Drawing on prior research regarding the impact of resolution consistency (Liu et al., 2023b,c; He et al., 2022; Dosovitskiy, 2020), we investigated two adjustments to our pipeline: (1) capping inference resolution at 2000 × 2000 to align with training, and (2) increasing the training resolution limit to 2500 × 2500. Our experiments demonstrate that both strategies significantly improve model performance by mitigating the train-test inconsistency.

#### 3.3 Reinforcement Learning with Verifiable Rewards

Training Data Construction Following the pipeline described in subsection 3.1, we construct training samples by overlaying multiple application windows onto desktop backgrounds. To facilitate more effective learning and mitigate zero-gradient issues, we use the initialized model to perform eight rollouts per task. We retain only those samples with a pass rate between 0% and 75%, similar to recent work (Yu et al., 2025).

Training Algorithm We employ Group Relative Policy Optimization (GRPO) (Shao et al., 2024), optimizing the objective function:

  (8)

  1

|oi|

G

min ri,t(θ)Aˆi,t,clip(ri,t(θ),1 − ε,1 + ε)Aˆi,t

JGRPO(θ) = Eq∼D,{o

i}Gi=1∼πθold

G c=1 |oc|

t=1

i=1

G j=1)

θold(oi,t|q,oi,<t) is the importance sampling ratio, and Aˆi,t = Ri−mean({Rj}

where ri,t(θ) = ππθ(oi,t|q,oi,<t)

std({Rj}Gj=1) is the groupnormalized advantage. We found that a group size of G = 8 strikes an optimal balance between training effectiveness and efficiency. A key advantage of the GUI grounding task is that it permits a precise, verifiable reward function. Specifically, the reward Ri for a predicted coordinate (xn,yn) is binary, determined by whether it falls within the annotated bounding box bann = (xmin,ymin,xmax,ymax):

Ri =

1, if xmin ≤ xn ≤ xmax and ymin ≤ yn ≤ ymax, 0, otherwise.

(9)

### 4 Experiments Settings

Training Building upon the architecture of POINTS-1.5 (Liu et al., 2024e), we replace the Qwen2.5-7BInstruct backbone (Yang et al., 2024b) with Qwen3-8B (Team, 2025). Prior to specialized optimization for GUI tasks, the model underwent extensive pre-training and mid-training (Bai et al., 2025a; Guo et al., 2025; Hong et al., 2025) on large-scale datasets. The optimization of the GUI Grounding ability is diveded into two stages: supervised learning stage and reinforcement learning stage. During the supervised learning stage, we jointly fine-tune all components, including the vision encoder, projector, and large language model (LLM). We set the learning rate to 1 × 10−4 for the vision encoder and 5 × 10−5 for the projector and LLM. For the reinforcement learning stage, we employ 8 rollouts per sample with a global batch size of 64 and a learning rate of 1 × 10−5.

|[Figure 33]|
|---|

- Figure 6 Data distribution of all these GUI grounding datasets. Left: image shape distribution. Right: composition of these GUI Grounding datasets.

Datasets In addition to the dataset synthesized via the strategies detailed in subsection 3.1, we curate a text-centric subset from DataComp (Gadre et al., 2023) by filtering for images containing textual elements. We employ PaddleOCR (Cui et al., 2025) to extract text bounding boxes, utilizing the text as the query and the coordinates as the ground-truth answer (GUI-DataComp). To further enhance diversity, we incorporate 13 additional open-source datasets (Liu et al., 2024c; Xie et al., 2025; HelloKKMe, 2024; Cheng et al., 2024; Chen et al., 2025c; Mu et al., 2025; Hsiao et al., 2024; Feizi et al., 2025; agentsea, 2024; ivelin, 2024; Liu et al.,

- 2025a). The geometric and numerical distributions of these GUI datasets are visualized in Figure 6. Beyond specialized GUI data, our training pipeline integrates general-purpose corpora, such as Bee (Zhang et al.,
- 2025b). Performance is evaluated across five commonly-used benchmarks: ScreenSpot-v2 (Wu et al., 2024), ScreenSpot-Pro (Li et al., 2025), OSWorld-G (Xie et al., 2025), MMBench-GUI-L2 (Wang et al., 2025d), and UI-Vision (Nayak et al., 2025). Together, these benchmarks provide a comprehensive assessment across mobile, web, and desktop environments.

### 5 Experiment Results

The Most Influential Factors The development of large-scale models necessitate the integration of diverse technologies, including data curation and training strategies. This evolutionary process is often characterized by distinct plateau periods, where specific pivotal factors determine the transition to higher performance levels. Figure 4(left) illustrates the key milestones that proved decisive in the iteration of POINTS-GUI-G. Notably, the data presented reflects the aggregate average performance of all models within each plateau period, rather than the performance of a single individual model.

- Table 1 Performance comparison on ScreenSpot-V2. The best results are highlighted in bold, and the second-best results are underlined.

Mobile Desktop Web

Model

Avg. Text Icon. Text Icon. Text Icon.

Open-Source Models

Qwen3-VL-2B (Bai et al., 2025a) 95.5 82.0 95.4 73.6 89.7 76.4 86.7 Phi-ground (Zhang et al., 2025a) 90.2 76.4 93.6 75.9 96.5 62.0 83.8 OS-Atlas-7B (Wu et al., 2024) 95.2 75.8 90.7 63.6 90.6 77.3 85.1 UGround-v1-7B (Gou et al., 2025) 83.6 90.5 85.8 86.3 95.5 83.2 87.7 UI-Tars-1.5-7B (Seed, 2025b) 92.2 81.5 91.0 84.2 95.5 84.5 89.0 SE-GUI-7B (Yuan et al., 2025) 99.3 89.1 96.4 78.6 92.7 81.3 90.8 UI-TARS-7B (Qin et al., 2025) 96.9 89.1 95.4 85.0 93.6 85.2 91.6 Qwen3-VL-8B (Bai et al., 2025a) 97.9 84.8 95.9 87.9 95.7 83.7 91.7 GUI-Actor-7B (Wu et al., 2025) 97.6 88.2 96.9 85.7 93.2 86.7 92.1 OpenCUA-7B (Wang et al., 2025c) - - - - - - 92.3 GTA1-7B (Yang et al., 2025b) 99.0 88.6 94.9 89.3 92.3 86.7 92.4 GUI-Owl-7B (Ye et al., 2025) 99.0 92.4 96.9 85.0 93.6 85.2 92.8 GUI-G2-7B (Tang et al., 2025) 98.3 91.9 95.4 89.3 94.0 87.7 93.3 InfiGUI-G1-7B (Liu et al., 2025b) 99.0 91.9 94.3 82.1 97.9 89.2 93.5 UI-Venus-7B (Gu et al., 2025a) 99.0 90.0 96.9 90.7 96.2 88.7 94.1 MAI-UI-2B (Zhou et al., 2025) 99.3 87.2 97.4 88.6 94.0 84.7 92.5 MAI-UI-8B (Zhou et al., 2025) 99.3 89.1 99.0 92.1 97.9 91.1 95.2

Qwen3-VL-32B (Bai et al., 2025a) 96.2 90.0 97.4 85.0 95.7 89.7 93.0 GUI-Owl-32B (Ye et al., 2025) 98.6 90.0 97.9 87.8 94.4 86.7 93.2 OpenCUA-32B (Wang et al., 2025c) - - - - - - 93.4 GTA1-32B (Yang et al., 2025b) 99.7 90.5 99.0 94.3 95.7 90.1 95.2 UI-Venus-72B (Gu et al., 2025a) 99.7 93.8 95.9 90.0 96.2 92.6 95.3

Ours POINTS-GUI-G-8B 99.0 91.0 100.0 94.3 95.3 92.6 95.7

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| |[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]| | | | | |
| | | | | | | |

- Figure 7 Analysis of influential factors and the RL dynamics. Left: Comparison of key factors affecting performance (average score across the five evaluation benchmarks we use). Middle & Right: The Reinforcement Learning (RL) training procedure. Naive: baseline without GUI grounding optimization; DE: Data Engineering (see subsection 3.1); UVE: Unfrozen Vision Encoder; IR: Image Resolution consistency; RL: Reinforcement Learning.

The Training Procedure of Reinforcement Learning Due to the high variance and instability characteristic of RL training, tracking intermediate performance via Reward and Entropy Loss is crucial. Reward serves as a proxy for policy improvement, whereas Entropy Loss characterizes the model’s exploration-exploitation trade-off and the sharpening of its output distribution. Our experimental results (Figure 4(right)) show a consistent rise in reward followed by a stable plateau. Simultaneously, the fluctuating decrease in entropy loss demonstrates that the model continues to explore while progressively increasing the likelihood of generating optimal tokens (Figure 4(middle)).

- Table 2 Performance comparison of state-of-the-art models on the OSWorld-G. The best results are highlighted in bold, and the second-best results are underlined.

Text Matching

Element Recognition

Layout Understanding

Fine-grained Manipulation

Agent Model

Refusal Avg Proprietary Models

Operator (OpenAI, 2025) 51.3 42.4 46.6 31.5 0.0 40.6 Seed1.5-VL (Guo et al., 2025) 73.9 66.7 69.6 47.0 18.5 62.9

Open-Source Models

Jedi-3B (Xie et al., 2025) 67.4 53.0 53.8 44.3 7.4 50.9 Qwen3-VL-2B (Bai et al., 2025a) 61.7 45.8 54.2 39.6 - 45.9 OS-Atlas-7B (Wu et al., 2024) 44.1 29.4 35.2 16.8 7.4 27.7 UGround-7B (Gou et al., 2025) 51.3 40.3 43.5 24.8 0.0 36.4 Aguvis-7B (Xu et al., 2024) 55.9 41.2 43.9 28.2 0.0 38.7 UI-TARS-7B (Qin et al., 2025) 60.2 51.8 54.9 35.6 0.0 47.5 UI-TARS-1.5-7B (Seed, 2025b) 36.8 62.7 62.2 50.8 0.0 52.8 Jedi-7B (Xie et al., 2025) 65.9 55.5 57.7 46.9 7.4 54.1 Qwen3-VL-8B (Bai et al., 2025a) 69.0 55.5 59.7 47.7 - 54.8 GTA1-7B (Yang et al., 2025b) 42.1 65.7 62.7 56.1 0.0 55.1 GUI-Owl-7B (Ye et al., 2025) 64.8 63.6 61.3 41.0 - 55.9 UI-Venus-7B (Gu et al., 2025a) 74.6 60.5 61.5 45.5 - 58.8 MAI-UI-2B (Zhou et al., 2025) 62.8 56.7 59.3 40.3 - 52.0 MAI-UI-8B (Zhou et al., 2025) 72.0 63.3 66.0 51.0 - 60.1

OpenCUA-32B (Wang et al., 2025c) - - - - - 59.6 GUI-Owl-32B (Ye et al., 2025) 67.0 64.5 67.2 45.6 - 58.0 Qwen3-VL-32B (Bai et al., 2025a) 72.8 63.3 66.4 51.7 - 60.6 GTA1-32B (Yang et al., 2025b) 63.2 78.4 73.3 65.2 0.0 65.2 Ours

POINTS-GUI-G-8B 73.9 73.6 70.8 55.7 - 66.0

### 6 Comparison with Other Models

In this section, we evaluate POINTS-GUI-G against models of comparable scale across the five benchmarks previously described. All comparative data is sourced from MAI-UI (Zhou et al., 2025). Overall, POINTSGUI-G demonstrates robust performance across desktop, mobile, and web environments, either establishing a significant lead over or achieving comparable performance with current state-of-the-art (SOTA) models. Specifically, POINTS-GUI-G ranks first on three of the five benchmarks.

On ScreenSpot-Pro (Li et al., 2025), which tests the ability to execute complex tasks within high-resolution interface scenarios, POINTS-GUI-G yields impressive results. It outperforms GTA1-7B (Yang et al., 2025b) by 9.8 points and GUI-Owl-7B (Ye et al., 2025) by 5 points. Notably, it even surpasses significantly larger models, such as OpenCUA-32B (OpenAI, 2025), trailing only MAI-UI-8B on this specific metric.

Regarding complex desktop environments like OSWorld-G(Xie et al., 2025), POINTS-GUI-G secures the top rank among SOTA models, outperforming MAI-UI-8B by approximately 6 points. Its superiority is also evident in UI-Vision, where it handles diverse and complex instructions with a margin often exceeding 10 points over its competitors. Furthermore, while achieving comparable performance to MAI-UI-8B on MMBench-GUI, POINTS-GUI-G remains the top performer in general grounding tasks that span from mobile to desktop environments (ScreenSpotv2). This consistently high ranking underscores its versatility and effectiveness in GUI grounding across all evaluated scenarios.

### 7 Conclusion

In this work, we introduce POINTS-GUI-G, a high-performance GUI grounding model. Despite its compact 8B-parameter architecture, the model achieves state-of-the-art results, ranking first among similarly sized models across multiple benchmarks and outperforming several significantly larger counterparts. Beyond its

- Table 3 Performance comparison on UI-Vision grounding dataset. The best results are highlighted in bold, and the second-best results are underlined.

Models Basic Functional Spatial Avg Proprietary Models

GPT-4o (OpenAI, 2024) 1.6 1.5 1.0 1.4 Claude-3.7-Sonnet (Anthropic, 2024) 9.5 7.7 7.6 8.3

Open-source Models Qwen3-VL-2B (Bai et al., 2025a) 0.0 19.2 0.1 6.2 InfiGUI-G1-3B (Liu et al., 2025b) 31.2 28.0 8.2 22.0 Qwen2.5-VL-7B (Bai et al., 2025b) 1.2 0.8 0.5 0.9 SeeClick (Cheng et al., 2024) 9.4 4.7 2.1 5.4 UGround-V1-7B (Gou et al., 2025) 15.4 17.1 6.3 12.9 OS-Atlas-7B (Wu et al., 2024) 12.2 11.2 3.7 9.0 Qwen3-VL-8B (Bai et al., 2025a) 25.0 27.9 1.2 17.5 UI-TARS-7B (Qin et al., 2025) 20.1 24.3 8.4 17.6 UI-TARS-1.5-7B (Seed, 2025b) 28.8 27.5 10.7 22.3 InfiGUI-G1-7B (Liu et al., 2025b) 36.2 31.9 11.5 26.1 UI-Venus-7B (Gu et al., 2025a) 36.1 32.8 11.9 26.5 Phi-Ground (Zhang et al., 2025a) 36.8 37.1 7.6 27.2 MAI-UI-2B (Zhou et al., 2025) 41.0 41.2 10.4 30.3 MAI-UI-8B (Zhou et al., 2025) 51.7 49.6 22.5 40.7

Qwen3-VL-32B (Bai et al., 2025a) 32.8 34.2 14.7 26.9 UI-TARS-72B (Qin et al., 2025) 31.4 30.5 14.7 25.5 UI-Venus-72B (Gu et al., 2025a) 45.6 42.3 23.7 36.8

Ours POINTS-GUI-G-8B 63.2 55.6 30.9 49.9

performance, we detail our comprehensive optimization methodology, covering data engineering, training strategies, and the specific techniques most critical to enhancing grounding accuracy. To support the research community and address inconsistent evaluation standards, we are open-sourcing our model and evaluation suite. We believe this full-scale contribution will catalyze progress in the field, providing a robust foundation for our future work in optimizing end-to-end task execution for GUI agents.

- Table 4 Performance comparison on the MMBench-GUI L2 benchmark. The best results are highlighted in bold, and the second-best results are underlined.

Windows MacOS Linux iOS Android Web

Model

Avg. Bas. Adv. Bas. Adv. Bas. Adv. Bas. Adv. Bas. Adv. Bas. Adv.

Proprietary Models

GPT-4o (OpenAI, 2024) 1.5 1.1 8.7 4.3 1.1 1.0 5.1 3.3 2.5 1.4 3.2 2.9 2.9 Claude-3.7 (Anthropic, 2024) 1.5 0.7 12.5 7.5 1.1 0.0 13.7 10.6 1.4 1.4 3.2 2.3 4.7 Qwen-Max-VL (Yang et al., 2024a) 43.9 36.8 58.8 56.1 53.9 30.1 77.4 59.1 79.5 70.1 74.8 58.8 58.0

Open-Source Models

Qwen3-VL-2B (Bai et al., 2025a) 81.9 0.0 80.3 46.2 67.5 0.0 90.8 0.0 91.0 0.0 88.1 0.0 46.5 OS-Atlas-7B (Wu et al., 2024) 36.9 18.8 44.4 21.7 31.4 13.3 74.8 48.8 69.6 46.8 61.3 35.4 41.4 Aguvis-7B (Xu et al., 2024) 37.3 21.7 48.1 33.3 33.5 25.0 67.5 65.2 61.0 51.0 61.6 45.5 45.7 UI-TARS-1.5-7B (Seed, 2025b) 68.3 39.0 69.0 44.5 64.4 37.8 88.5 69.4 90.5 69.3 81.0 56.5 64.3 UGround-V1-7B (Gou et al., 2024) 66.8 39.0 71.3 48.6 56.5 31.1 92.7 70.9 93.5 71.0 88.7 64.6 65.7 GUI-Actor-7B (Wu et al., 2025) 80.8 55.1 81.4 60.4 64.9 41.8 94.3 82.7 93.5 79.7 89.7 72.1 76.5 SE-GUI-7B (Yuan et al., 2025) 77.5 57.7 77.1 60.7 68.6 44.9 95.5 80.0 95.5 83.7 89.7 68.8 76.6 Qwen3-VL-8B (Bai et al., 2025a) 88.6 61.8 85.5 69.1 74.9 53.1 95.2 82.4 95.5 84.5 96.8 72.1 81.3 GTA1-7B (Yang et al., 2025b) 76.8 57.4 80.3 63.9 68.6 53.6 93.9 83.3 96.3 84.5 90.3 74.7 78.5 GUI-G2-7B (Tang et al., 2025) 79.7 55.1 79.7 64.7 69.6 50.0 95.2 82.7 96.6 85.4 91.9 75.6 78.8 GUI-Owl-7B (Ye et al., 2025) 86.4 61.8 81.7 64.5 74.4 61.7 94.9 83.0 95.8 83.7 93.2 72.7 80.5 InfiGUI-G1-7B (Liu et al., 2025b) 82.7 61.8 83.8 63.9 72.3 52.0 94.9 89.4 95.2 85.6 93.5 76.3 80.8 MAI-UI-2B (Zhou et al., 2025) 84.9 64.0 89.3 72.5 75.4 60.2 95.2 85.2 96.3 84.2 92.9 76.0 82.6 MAI-UI-8B (Zhou et al., 2025) 92.3 74.3 90.7 86.4 81.2 67.3 97.1 90.0 97.5 92.7 95.8 86.0 88.8

GUI-Owl-32B (Ye et al., 2025) 85.6 65.1 84.9 67.1 77.0 63.3 95.2 85.5 96.1 87.0 95.5 80.8 83.0 GTA1-32B (Yang et al., 2025b) 82.3 66.9 89.0 74.0 73.3 52.0 96.2 88.2 95.8 88.5 95.2 79.9 83.4 Qwen3-VL-32B (Bai et al., 2025a) 93.4 71.3 92.8 74.3 78.0 56.1 95.5 88.8 97.2 88.5 92.6 78.6 85.3 UI-TARS-DPO-72B (Qin et al., 2025) 78.6 51.8 80.3 62.7 68.6 51.5 90.8 81.2 93.0 80.0 88.1 68.5 74.3 InternVL3-78B (Zhu et al., 2025) 70.1 42.6 75.7 52.3 59.2 41.3 93.6 80.6 92.7 78.6 90.7 65.9 72.2

Ours POINTS-GUI-G-8B 93.7 68.4 87.5 75.7 85.9 69.9 97.1 88.8 97.2 90.1 95.8 84.4 87.0

- Table 5 Performance comparison on the ScreenSpot-Pro benchmark. The best results are highlighted in bold, and the second-best results are underlined.

CAD Dev. Creative Scientific Office OS

Model

Avg. Text Icon Text Icon Text Icon Text Icon Text Icon Text Icon

Proprietary Models GPT-4o (OpenAI, 2024) 2.0 0.0 1.3 0.0 1.0 0.0 2.1 0.0 1.1 0.0 0.0 0.0 0.8 Claude C. (Anthropic, 2024) 14.5 3.7 22.0 3.9 25.9 3.4 33.9 15.8 30.1 16.3 11.0 4.5 17.1 Gemini-3-Pro (DeepMind, 2025) - - - - - - - - - - - - 72.7 Seed1.8 (Seed, 2025a) - - - - - - - - - - - - 73.1

Open-Source Models

Qwen3-VL-2B (Bai et al., 2025a) 31.0 15.6 55.2 11.7 59.1 16.1 64.6 22.7 72.3 34.0 59.8 23.6 41.9 InfiGUI-3B (Liu et al., 2025b) 50.8 25.0 64.9 20.0 51.5 16.8 68.8 32.7 70.6 32.1 49.5 19.7 45.2 Ferret-UI Lite (Yang et al., 2025d) - - - - - - - - - - - - 53.3 UI-TARS-7B (Qin et al., 2025) 20.8 9.4 58.4 12.4 50.0 9.1 63.9 31.8 63.3 20.8 30.8 16.9 35.7 Phi-Ground (Zhang et al., 2025a) 26.9 17.2 70.8 16.7 56.6 13.3 58.0 29.1 76.4 44.0 55.1 25.8 43.2 GUI-Actor-7B (Wu et al., 2025) 47.7 9.4 59.1 15.9 59.6 16.1 70.1 25.5 69.5 41.5 55.1 19.1 44.6 SE-GUI-7B (Yuan et al., 2025) 51.3 14.1 68.2 19.3 57.6 9.1 75.0 28.2 78.5 43.4 49.5 25.8 47.2 GUI-G2-7B (Tang et al., 2025) 55.8 12.5 68.8 17.2 57.1 15.4 77.1 24.5 74.0 32.7 57.9 21.3 47.5 Qwen3-VL-8B (Bai et al., 2025a) 46.7 10.9 79.2 23.4 68.2 14.0 73.6 30.0 76.3 30.2 65.4 21.3 49.9 OpenCUA-7B (Wang et al., 2025c) - - - - - - - - - - - - 50.0 GTA1-7B (Yang et al., 2025b) 53.3 17.2 66.9 20.7 62.6 18.9 76.4 31.8 82.5 50.9 48.6 25.9 50.1 UI-Venus-7B (Gu et al., 2025a) 60.4 21.9 74.7 24.1 63.1 14.7 76.4 31.8 75.7 41.5 49.5 22.5 50.8 InfiGUI-G1-7B (Liu et al., 2025b) 57.4 23.4 74.7 24.1 64.6 18.2 80.6 31.8 75.7 39.6 57.0 29.2 51.9 GUI-Owl-7B (Ye et al., 2025) 64.5 21.9 76.6 31.0 59.6 27.3 79.1 37.3 77.4 39.6 59.8 33.7 54.9 MAI-UI-2B (Zhou et al., 2025) 61.4 23.4 76.6 32.4 69.2 21.7 81.2 34.5 85.9 39.6 68.2 41.6 57.4 MAI-UI-8B (Zhou et al., 2025) 72.6 35.9 83.8 52.4 76.3 33.6 79.9 37.3 88.7 60.4 76.6 49.4 65.8

Qwen3-VL-32B (Bai et al., 2025a) 60.4 28.1 69.5 22.1 75.8 25.2 84.7 25.5 85.9 43.4 62.6 15.7 54.9 OpenCUA-32B (Wang et al., 2025c) - - - - - - - - - - - - 55.3 GUI-Owl-32B (Ye et al., 2025) 62.4 28.1 84.4 39.3 65.2 18.2 82.6 39.1 81.4 39.6 70.1 36.0 58.0 UGround-v1-72B (Gou et al., 2025) 16.8 4.7 55.8 4.8 54.0 10.5 70.8 22.7 61.0 18.9 40.2 7.9 34.5 UI-Tars-72B (Qin et al., 2025) 18.8 12.5 63.0 17.2 57.0 15.4 64.6 20.9 63.3 26.4 42.1 15.7 38.1

Ours POINTS-GUI-G-8B 46.7 28.1 76.0 44.8 66.2 33.6 81.3 48.2 88.7 62.3 72.0 43.8 59.9

### References

agentsea. wave-ui. https://huggingface.co/datasets/agentsea/wave-ui, 2024. Anthropic. Claude 3.7 sonnet. https://www.anthropic.com/news/claude-3-7-sonnet, 2024. Accessed: 2025-08-02. Anthropic. Our 3.5 models and computer use. https://www.anthropic.com/news/3-5-models-and-computer-use,

2024. Accessed: 2025-09-22. Chongyang Bai, Xiaoxue Zang, Ying Xu, Srinivas Sunkara, Abhinav Rastogi, Jindong Chen, et al. Uibert: Learning generic multimodal representations for ui understanding. arXiv preprint arXiv:2107.13731, 2021.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xiong-Hui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Rongyao Fang, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Li Ying Meng, Xuancheng Ren, Xin yi Ren, Sibo Song, Yu-Chen Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yihe Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Botao Zheng, Humen Zhong, Jingren Zhou, Fanxi Zhou, Jingren Zhou, Yuanzhi Zhu, and Keming Zhu. Qwen3-vl technical report. ArXiv, abs/2511.21631, 2025a.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b.

Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo Yang, Boji Shan, Changqing Yu, Chao Wang, Cheng Zhu, et al. Minimax-m1: Scaling test-time compute efficiently with lightning attention. arXiv preprint arXiv:2506.13585, 2025a.

Liangyu Chen, Hanzhang Zhou, Chenglin Cai, Jianan Zhang, Panrong Tong, Quyu Kong, Xu Zhang, Chen Liu, Yuqi Liu, Wenxuan Wang, et al. Ui-ins: Enhancing gui grounding with multi-perspective instruction-as-reasoning. arXiv preprint arXiv:2510.20286, 2025b.

Wentong Chen, Junbo Cui, Jinyi Hu, Yujia Qin, Junjie Fang, Yue Zhao, Chongyi Wang, Jun Liu, Guirong Chen, Yupeng Huo, et al. Guicourse: From general vision language model to versatile gui agent. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 21936–21959, 2025c.

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.

Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Li YanTao, Jianbing Zhang, and Zhiyong Wu. SeeClick: Harnessing GUI grounding for advanced visual GUI agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9313–9332, Bangkok, Thailand, 2024. Association for Computational Linguistics.

Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao, Yubo Zhang, Jiaxuan Liu, Xueqing Wang, Zelun Zhang, Changda Zhou, Hongen Liu, Yue Zhang, Wenyu Lv, Kui Huang, Yichao Zhang, Jing Zhang, Jun Zhang, Yi Liu, Dianhai Yu, and Yanjun Ma. Paddleocr 3.0 technical report, 2025.

Google DeepMind. Gemini 3 pro, 2025. Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas

Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim M Alabdulmohsin, et al. Patch n’pack: Navit, a vision transformer for any aspect ratio and resolution. Advances in Neural Information Processing Systems, 36:2252–2274, 2023.

Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Aarash Feizi, Shravan Nayak, Xiangru Jian, Kevin Qinghong Lin, Kaixin Li, Rabiul Awal, Xing Han Lù, Johan Obando-Ceron, Juan A. Rodriguez, Nicolas Chapados, David Vazquez, Adriana Romero-Soriano, Reihaneh Rabbany, Perouz Taslakian, Christopher Pal, Spandana Gella, and Sai Rajeswar. Grounding computer use agents on human demonstrations, 2025.

Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. Advances in Neural Information Processing Systems, 36:27092–27112, 2023.

Boyu Gou, Ruohan Wang, Boyuan Zheng, Yanan Xie, Cheng Chang, Yiheng Shu, Huan Sun, and Yu Su. Navigating the digital world as humans do: Universal visual grounding for gui agents. ArXiv, abs/2410.05243, 2024.

Boyu Gou, Ruohan Wang, Boyuan Zheng, Yanan Xie, Cheng Chang, Yiheng Shu, Huan Sun, and Yu Su. Navigating the digital world as humans do: Universal visual grounding for GUI agents. In The Thirteenth International Conference on Learning Representations, 2025.

Zhangxuan Gu, Zhengwen Zeng, Zhenyu Xu, Xingran Zhou, Shuheng Shen, Yunfei Liu, Beitong Zhou, Changhua Meng, Tianyu Xia, Weizhi Chen, Yue Wen, Jingya Dou, Fei Tang, Jinzhen Lin, Yulin Liu, Zhenlin Guo, Yichen Gong, Heng Jia, Changlong Gao, Yuan Guo, Yong Deng, Zhenyu Guo, Liang Chen, and Weiqiang Wang. Ui-venus technical report: Building high-performance ui agents with rft, 2025a.

Zhangxuan Gu, Zhengwen Zeng, Zhenyu Xu, Xingran Zhou, Shuheng Shen, Yunfei Liu, Beitong Zhou, Changhua Meng, Tianyu Xia, Weizhi Chen, et al. Ui-venus technical report: Building high-performance ui agents with rft. arXiv preprint arXiv:2508.10833, 2025b.

Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, Jingji Chen, Jingjia Huang, Kang Lei, Liping Yuan, Lishu Luo, Pengfei Liu, Qinghao Ye, Rui Qian, Shen Yan, Shixiong Zhao, Shuai Peng, Shuangye Li, Sihang Yuan, Si-Ming Wu, Tianheng Cheng, Weiwei Liu, Wenqian Wang, Xianhan Zeng, Xiao Liu, Xiaobo Qin, Xiaohan Ding, Xiaojun Xiao, Xiaoying Zhang, Xuanwei Zhang, Xuehan Xiong, Yanghua Peng, Yangrui Chen, Yanwei Li, Ya-Fang Hu, Yi Lin, Yi Chun Hu, Yiyuan Zhang, Youbin Wu, Yu Li, Yudong Liu, Yueming Ling, Yujia Qin, Zanbo Wang, Zhiwu He, Aoxue Zhang, Bairen Yi, Ben Ben Liao, Can Huang, Can Zhang, Chaorui Deng, Chaoyi Deng, Cheng Lin, Cheng Yuan, Chenggang Li, Chenhui Gou, Chenwei Lou, Chengzhi Wei, Chundian Liu, Chunyuan Li, Deyao Zhu, Donghong Zhong, Feng Li, Feng Zhang, Gang Wu, Guodong Li, Guohong Xiao, Haibin Lin, Haihua Yang, Haoming Wang, Heng Ji, Hongxiang Hao, Hui Shen, Huixia Li, Jiahao Li, Jialong Wu, Jianhua Zhu, Jianpeng Jiao, Jiashi Feng, Jiaze Chen, Jianhui Duan, Jihao Liu, Jin Zeng, Jingqun Tang, Jingyu Sun, Joya Chen, Jun Long, Junda Feng, Junfeng Zhan, Junjie Fang, Ju Lu, Kai Hua, Kai Liu, Kai Shen, Kai-Hua Zhang, Ke Shen, Ke Wang, Keyu Pan, Kun Zhang, Kunchang Li, Lanxin Li, Lei Li, Lei Shi, Li Han, Liang Xiang, Liangqiang Chen, Lin Chen, Lin Li, Lin Yan, Liying Chi, Longxiang Liu, Meng-Han Du, Mingxuan Wang, Ningxin Pan, Peibin Chen, Pengfei Chen, Pengfei Wu, Qing-Yun Yuan, Qi Shuai, Qiuyan Tao, Ren Kui Zheng, Renrui Zhang, Ru Zhang, Rui Wang, Rui Yang, Rui Zhao, Shaoqiang Xu, Shihao Liang, Shi feng Yan, Shu Zhong, Shuai Shuai Cao, Shuangzhi Wu, Shufan Liu, Shuhan Chang, Songhua Cai, Tenglong Ao, Tian-Yi Yang, Tingting Zhang, Wanjun Zhong, Wei Jia, Wei Weng, Weihao Yu, Wenhao Huang, Wenjia Zhu, Wenli Yang, Wenzhi Wang, Xiang Long, Xian gang Yin, Xiao Li, Xiaolei Zhu, Xiaoying Jia, Xijin Zhang, Xin Liu, Xincheng Zhang, Xinyu Yang, Xiongcai Luo, Xiuli Chen, Xuantong Zhong, Xuefeng Xiao, Xujing Li, Yan Wu, Ya-Feng Wen, Yi-Mei Du, Yihao Zhang, Yining Ye, Yong-Xu Wu, Yu Liu, Yuanhang Yue, Yufeng Zhou, Yufeng Yuan, Yuhang Xu, Yuhong Yang, Yun Zhang, Yu-Qing Fang, Yuntao Li, Yurui Ren, Yuwen Xiong, Zehua Hong, Zehua Wang, Ze-Bang Sun, Zeyu Wang, Zhao Cai, Zhaoyue Zha, Zhecheng An, Zhehui Zhao, Zheng Xu, Zhipeng Chen, Zhiyong Wu, Zhuofan Zheng, Zihao Wang, Zilong Huang, Ziyu Zhu, and Zuquan Song. Seed1.5-vl technical report. ArXiv, abs/2505.07062, 2025.

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000–16009, 2022.

HelloKKMe. Grounding dataset. https://huggingface.co/datasets/HelloKKMe/grounding_dataset, 2024. Accessed: 2024-05-22.

GLM-V Team Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, Shuaiqi Duan, Weihan Wang, Yan Wang, Yean Cheng, Zehai He, Zhe Su, Zhen Yang, Ziyang Pan, Aohan Zeng, Baoxu Wang, Boyan Shi, Changyu Pang, Chenhui Zhang, Da Yin, Fan Yang, Guoqing Chen, Jiazheng Xu, Jiali Chen, Jing Chen, Jinhao Chen, Jinghao Lin, Jinjiang Wang, Junjie Chen, Leqi Lei, Letian Gong, Leyi Pan, Mingzhi Zhang, Qinkai Zheng, Shengchao Yang, Shilong Zhong, Shiyu Huang, Shuyuan Zhao, Siyan Xue, Shangqing Tu, Shengbiao Meng, Tianshu Zhang, Tian-Yuan Luo, Tianxiang Hao, Wenkai Li, Wei Jia, Xinpeng Lyu, Xuancheng Huang, Yanling Wang, Ya-Qi Xue, Yanfeng Wang, Yifan An, Yifan Du, Yi Shi, Yiheng Huang, Yilin Niu, Yuan Wang, Yuanchang Yue, Yuchen Li, Yutao Zhang, Yuxuan Zhang, Zhanxiao Du, Zhenyu Hou, Zhao Xue, Zhengxiao Du, Zihan Wang, Peng Zhang, De-Huan Liu, Bin Xu, Juanzi Li, Minlie Huang, Yuxiao Dong, and Jie Tang. Glm-4.5v and glm-4.1v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. 2025.

Yu-Chung Hsiao, Fedir Zubach, Maria Wang, and Jindong Chen. Screenqa: Large-scale question-answer pairs over

mobile app screenshots, 2024. ivelin. uirefexp. https://huggingface.co/datasets/ivelin/ui_refexp_saved, 2024. Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li,

Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR, 2023.

Kaixin Li, Ziyang Meng, Hongzhan Lin, Ziyang Luo, Yuchen Tian, Jing Ma, Zhiyong Huang, and Tat-Seng Chua. Screenspot-pro: Gui grounding for professional high-resolution computer use. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 8778–8786, 2025.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023a.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In

Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024a. Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved

reasoning, ocr, and world knowledge, 2024b. Junpeng Liu, Tianyue Ou, Yifan Song, Yuxiao Qu, Wai Lam, Chenyan Xiong, Wenhu Chen, Graham Neubig, and Xiang Yue. Harnessing webpage uis for text-rich visual understanding, 2024c. Yuan Liu, Songyang Zhang, Jiacheng Chen, Kai Chen, and Dahua Lin. Pixmim: Rethinking pixel reconstruction in masked image modeling. arXiv preprint arXiv:2303.02416, 2023b.

Yuan Liu, Songyang Zhang, Jiacheng Chen, Zhaohui Yu, Kai Chen, and Dahua Lin. Improving pixel-based mim by reducing wasted modeling capability. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5361–5372, 2023c.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024d.

Yuan Liu, Le Tian, Xiao Zhou, Xinyu Gao, Kavio Yu, Yang Yu, and Jie Zhou. Points1. 5: Building a vision-language model towards real world applications. arXiv preprint arXiv:2412.08443, 2024e.

Yuan Liu, Le Tian, Xiao Zhou, and Jie Zhou. Rethinking overlooked aspects in vision-language models. arXiv preprint arXiv:2405.11850, 2024f.

Yuan Liu, Zhongyin Zhao, Ziyuan Zhuang, Le Tian, Xiao Zhou, and Jie Zhou. Points: Improving your vision-language model with affordable strategies. arXiv preprint arXiv:2409.04828, 2024g.

Yuhang Liu, Pengxiang Li, Zishu Wei, Congkai Xie, Xueyu Hu, Xinchen Xu, Shengyu Zhang, Xiaotian Han, Hongxia Yang, and Fei Wu. Infiguiagent: A multimodal generalist gui agent with native reasoning and reflection. arXiv preprint arXiv:2501.04575, 2025a.

Yuhang Liu, Zeyu Liu, Shuanghe Zhu, Pengxiang Li, Congkai Xie, Jiasheng Wang, Xueyu Hu, Xiaotian Han, Jianbo Yuan, Xinyao Wang, et al. Infigui-g1: Advancing gui grounding with adaptive exploration policy optimization. arXiv preprint arXiv:2508.05731, 2025b.

Yuan Liu, Zhongyin Zhao, Le Tian, Haicheng Wang, Xubing Ye, Yangxiu You, Zilin Yu, Chuhan Wu, Zhou Xiao, Yang Yu, and Jie Zhou. POINTS-reader: Distillation-free adaptation of vision-language models for document conversion. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 1576–1601, Suzhou, China, 2025c. Association for Computational Linguistics.

Yadong Lu, Jianwei Yang, Yelong Shen, and Ahmed Awadallah. Omniparser for pure vision based gui agent, 2024. Run Luo, Lu Wang, Wanwei He, and Xiaobo Xia. Gui-r1 : A generalist r1-style vision-language action model for gui

agents. ArXiv, abs/2504.10458, 2025.

Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, et al. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2503.07365, 2025.

Jian Mu, Chaoyun Zhang, Chiming Ni, Lu Wang, Bo Qiao, Kartik Mathur, Qianhui Wu, Yuhang Xie, Xiaojun Ma, Mengyu Zhou, et al. Gui-360: A comprehensive dataset and benchmark for computer-using agents. arXiv preprint arXiv:2511.04307, 2025.

Shravan Nayak, Xiangru Jian, Kevin Qinghong Lin, Juan A. Rodriguez, Montek Kalsi, Rabiul Awal, Nicolas Chapados, M. Tamer ¨Ozsu, Aishwarya Agrawal, David Vázquez, Christopher Pal, Perouz Taslakian, Spandana Gella, Sai Rajeswar, and Human Annotator. Ui-vision: A desktop-centric gui benchmark for visual perception and interaction. ArXiv, abs/2503.15661, 2025.

OpenAI. Gpt-4o system card, 2024. OpenAI. Developing a generalist computer-using agent. OpenAI, 2025. Accessed: October 22, 2025. Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von

Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest text data at scale. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024.

Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. Ui-tars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326, 2025.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization

algorithms. ArXiv, abs/1707.06347, 2017. Bytedance Seed. Seed1.8 model card: Towards generalized real-world agency. arXiv preprint, 2025a. Technical Report. ByteDance Seed. Ui-tars-1.5. https://seed-tars.com/1.5, 2025b. Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK

Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Fei Tang, Zhangxuan Gu, Zhengxi Lu, Xuyang Liu, Shuheng Shen, Changhua Meng, Wen Wang, Wenqi Zhang, Yongliang Shen, Weiming Lu, Jun Xiao, and Yueting Zhuang. Gui-g2: Gaussian reward modeling for gui grounding, 2025.

Kimi Team, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang

Du, Chu Wei, et al. Kimi-vl technical report. arXiv preprint arXiv:2504.07491, 2025. Qwen Team. Qwen3 technical report, 2025. Haoming Wang, Haoyang Zou, Huatong Song, Jiazhan Feng, Junjie Fang, Junting Lu, Longxiang Liu, Qinyu Luo,

Shihao Liang, Shijue Huang, et al. Ui-tars-2 technical report: Advancing gui agent with multi-turn reinforcement learning. arXiv preprint arXiv:2509.02544, 2025a.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025b.

Xinyuan Wang, Bowen Wang, Dunjie Lu, Junlin Yang, Tianbao Xie, Junli Wang, Jiaqi Deng, Xiaole Guo, Yiheng Xu, Chen Henry Wu, Zhennan Shen, Zhuokai Li, Ryan Li, Xiaochuan Li, Junda Chen, Boyuan Zheng, Peihang Li, Fangyu Lei, Ruisheng Cao, Yeqiao Fu, Dongchan Shin, Martin Shin, Jiarui Hu, Yuyan Wang, Jixuan Chen, Yuxiao Ye, Danyang Zhang, Dikang Du, Hao Hu, Huarong Chen, Zaida Zhou, Haotian Yao, Ziwei Chen, Qizheng Gu, Yipu Wang, Heng Wang, Diyi Yang, Victor Zhong, Flood Sung, Y. Charles, Zhilin Yang, and Tao Yu. Opencua: Open foundations for computer-use agents, 2025c.

Xuehui Wang, Zhenyu Wu, Jingjing Xie, Zichen Ding, Bowen Yang, Zehao Li, Zhaoyang Liu, Qingyun Li, Xuan Dong, Zhe Chen, Weiyun Wang, Xiangyu Zhao, Jixuan Chen, Haodong Duan, Tianbao Xie, Chenyu Yang, Shiqian Su, Yue Yu, Yuan Huang, Yiqian Liu, Xiao Zhang, Yanting Zhang, Xiangyu Yue, Weijie Su, Xizhou Zhu, Wei Shen, Jifeng Dai, and Wenhai Wang. Mmbench-gui: Hierarchical multi-platform evaluation framework for gui agents. ArXiv, abs/2507.19478, 2025d.

Jason Wu, Siyan Wang, Siman Shen, Yi-Hao Peng, Jeffrey Nichols, and Jeffrey Bigham. Webui: A dataset for enhancing visual ui understanding with web semantics. ACM Conference on Human Factors in Computing Systems (CHI), 2023.

Qianhui Wu, Kanzhi Cheng, Rui Yang, Chaoyun Zhang, Jianwei Yang, Huiqiang Jiang, Jian Mu, Baolin Peng, Bo Qiao, Reuben Tan, et al. Gui-actor: Coordinate-free visual grounding for gui agents. arXiv preprint arXiv:2506.03143, 2025.

Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, et al. Os-atlas: A foundation action model for generalist gui agents. arXiv preprint arXiv:2410.23218, 2024.

Tianbao Xie, Jiaqi Deng, Xiaochuan Li, Junlin Yang, Haoyuan Wu, Jixuan Chen, Wenjing Hu, Xinyuan Wang, Yuhui Xu, Zekun Wang, Yiheng Xu, Junli Wang, Doyen Sahoo, Tao Yu, and Caiming Xiong. Scaling computer-use grounding via user interface decomposition and synthesis, 2025.

Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tianbao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, and Caiming Xiong. Aguvis: Unified pure vision agents for autonomous gui interaction. ArXiv, abs/2412.04454, 2024.

Haolong Yan, Jia Wang, Xin Huang, Yeqing Shen, Ziyang Meng, Zhimin Fan, Kaijun Tan, Jin Gao, Lieyu Shi, Mi Yang, et al. Step-gui technical report. arXiv preprint arXiv:2512.15431, 2025.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 technical report, 2024a.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Qwen An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxin Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yi-Chao Zhang, Yunyang Wan, Yuqi Liu, Zeyu Cui, Zhenru Zhang, Zihan Qiu, Shanghaoran Quan, and Zekun Wang. Qwen2.5 technical report. ArXiv, abs/2412.15115, 2024b.

Yan Yang, Dongxu Li, Yutong Dai, Yuhao Yang, Ziyang Luo, Zirui Zhao, Zhiyuan Hu, Junzhe Huang, Amrita Saha, Zeyuan Chen, et al. Gta1: Gui test-time scaling agent. arXiv preprint arXiv:2507.05791, 2025b.

Yuhao Yang, Yue Wang, Dongxu Li, Ziyang Luo, Bei Chen, Chao Huang, and Junnan Li. Aria-UI: Visual grounding for GUI instructions. In Findings of the Association for Computational Linguistics: ACL 2025, pages 22418–22433, Vienna, Austria, 2025c. Association for Computational Linguistics.

Zhen Yang, Zi-Yi Dou, Di Feng, Forrest Huang, Anh Nguyen, Keen You, Omar Attia, Yuhao Yang, Michael Feng, Haotian Zhang, et al. Ferret-ui lite: Lessons from building small on-device gui agents. arXiv preprint arXiv:2509.26539, 2025d.

Jiabo Ye, Xi Zhang, Haiyang Xu, Haowei Liu, Junyang Wang, Zhaoqing Zhu, Ziwei Zheng, Feiyu Gao, Junjie Cao, Zhengxi Lu, et al. Mobile-agent-v3: Fundamental agents for gui automation. arXiv preprint arXiv:2508.15144, 2025.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Xinbin Yuan, Jian Zhang, Kaixin Li, Zhuoxuan Cai, Lujian Yao, Jie Chen, Enguang Wang, Qibin Hou, Jinwei Chen, Peng-Tao Jiang, et al. Enhancing visual grounding for gui agents via self-evolutionary reinforcement learning. arXiv preprint arXiv:2505.12370, 2025.

Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, et al. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models. arXiv preprint arXiv:2508.06471, 2025.

Miaosen Zhang, Ziqiang Xu, Jialiang Zhu, Qi Dai, Kai Qiu, Yifan Yang, Chong Luo, Tianyi Chen, Justin Wagle, Tim Franklin, et al. Phi-ground tech report: Advancing perception in gui grounding. arXiv preprint arXiv:2507.23779, 2025a.

Pan Zhang, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Rui Qian, Lin Chen, Qipeng Guo, Haodong Duan, Bin Wang, Linke Ouyang, et al. Internlm-xcomposer-2.5: A versatile large vision language model supporting long-contextual input and output. arXiv preprint arXiv:2407.03320, 2024.

Yi Zhang, Bolin Ni, Xin-Sheng Chen, Heng-Rui Zhang, Yongming Rao, Houwen Peng, Qinglin Lu, Han Hu, Meng-Hao

Guo, and Shi-Min Hu. Bee: A high-quality corpus and full-stack suite to unlock advanced fully open mllms, 2025b. Hanzhang Zhou, Xu Zhang, Panrong Tong, Jianan Zhang, Liangyu Chen, Quyu Kong, Chenglin Cai, Chen Liu, Yue

Wang, Jingren Zhou, and Steven Hoi. Mai-ui technical report: Real-world centric foundation gui agents, 2025.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

### A Appendix

- A.1 Unified System Prompt for Grounding Task Bounding box prediction

Output the bounding box in the image of the UI element corresponding to the instruction "Doesn’t start checkbox" and the description "A square checkbox with the label ’Doesn’t start’" with grounding.

Requirements for the output:

- - Return only the bounding box coordinates (x0, y0, x1, y1)
- - Coordinates must be normalized to the range [0, 1]
- - Round each coordinate to three decimal places
- - Format the output as strictly (x0, y0, x1, y1) without any additional text.

Center point localization

You are a GUI agent. Based on the UI screenshot provided, please locate the exact position of the element that matches the instruction given by the user. Requirements for the output:

- - Return only the point (x, y) representing the center of the target element
- - Coordinates must be normalized to the range [0, 1]
- - Round each coordinate to three decimal places
- - Format the output as strictly (x, y) without any additional text

- A.2 GUI Grounding Examples

[Figure 39]

[Figure 40]

Instruction: align the text in the selected text box to the left, and use the bounding box [x0, y0, x1, y1] to outline the button response: [0.358, 0.103, 0.37, 0.123]

[Figure 41]

[Figure 42]

Instruction: adjust the volume to 50% response: (0.825, 0.024)

[Figure 43]

[Figure 44]

Instruction: Please click on '(RL)' in point 3 response: (0.5, 0.595)

###### Figure 8 Prediction on desktop screenshots.

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Instruction: switch branches response: (0.15, 0.575) Instruction: see more recommendations response: (0.916, 0.543) Instruction: download Microsoft 365 response: (0.499, 0.795)

###### Figure 9 Prediction on mobile screenshots.

[Figure 51]

[Figure 52]

Instruction: select the second thumbnail to see more information response: (0.294, 0.884)

[Figure 53]

[Figure 54]

Instruction: Share on Facebook response: (0.977, 0.33)

[Figure 55]

[Figure 56]

Instruction: view the summary for a specific day in January response: (0.5215, 0.3095)

###### Figure 10 Prediction on web screenshots.

