# arXiv:2603.15030v2[cs.AI]19Mar2026

## VTC-Bench: Evaluating Agentic Multimodal Models via Compositional Visual Tool Chaining

Xuanyu Zhu1,7, Yuhao Dong2†, Rundong Wang3, Yang Shi1, Zhipeng Wu4, Yinlun Peng5, YiFan Zhang6, Yihang Lou1, Yuanxing Zhang1, Ziwei Liu2,

Yan Bai7, , and Yuan Zhou

1PKU 2NTU 3USTC 4CQU 5NUDT 6CASIA 7Meituan

#### https://github.com/zhuzil/VTC-Bench

Abstract. Recent advancements extend Multimodal Large Language Models (MLLMs) beyond standard visual question answering to utilizing external tools for advanced visual tasks. Despite this progress, precisely executing and effectively composing diverse tools for complex tasks remain persistent bottleneck. Constrained by sparse tool-sets and simple tool-use trajectories, existing benchmarks fail to capture complex and diverse tool interactions, falling short in evaluating model performance under practical, real-world conditions. To bridge this gap, we introduce VisualToolChain-Bench (VTC-Bench), a comprehensive benchmark designed to evaluate tool-use proficiency in MLLMs. To align with realistic computer vision pipelines, our framework features 32 diverse OpenCVbased visual operations. This rich tool-set enables extensive combinations, allowing VTC-Bench to rigorously assess multi-tool composition and long-horizon, multi-step plan execution. For precise evaluation, we provide 680 curated problems structured across a nine-category cognitive hierarchy, each with ground-truth execution trajectories. Extensive experiments on 19 leading MLLMs reveal critical limitations in current models’ visual agentic capabilities. Specifically, models struggle to adapt to diverse tool-sets and generalize to unseen operations, with the leading model Gemini-3.0-Pro only achieving 51% on our benchmark. Furthermore, multi-tool composition remains a persistent challenge. When facing complex tasks, models struggle to formulate efficient execution plans, relying heavily on a narrow, suboptimal subset of familiar functions rather than selecting the optimal tools. By identifying these fundamental challenges, VTC-Bench establishes a rigorous baseline to guide the development of more generalized visual agentic models.

Keywords: Multimodal Agentic Models · Visual Tool Use Benchmark

### 1 Introduction

The rapid evolution [2–4, 7–9, 15, 26, 31, 36, 49] of Multimodal Large Language Models (MLLMs) has led to remarkable improvements in foundational capabil-

† Project leader. Corresponding authors.

###### OpenCV Toolkit

###### Task Taxonomy

###### MLLM Controller

[Figure 1]

ENHANCEMENT

GEOMETRY

Compositional Reasoning (Tier 3)

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

Cognitive Difficulty

Math

Rotate Crop Resize Flip

Color Brightness HistEq Binarize

FloodFill

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Spatial Chart

Translate Pyramid

ZoomIn

Denoise Sharpen Inpaint InrangeColor Morph

[Figure 23]

[Figure 24]

Quantitative Estimation (Tier 2)

[Figure 25]

FEATURES

OpenCV DRAW/MEASURE

[Figure 26]

Measure Color Counting

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

CannyEdge HoughLine

GrabCut

Sobel

Contour Area Perimeter Poly

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Robust OCR RestorationPerceptual AttentionFocusing

Watershed

ConnComp DFT

HoughCircle

DrawLine DrawCircle

Structured Access to 30+ Functions

Thought-Action-Observation

Visual Perception Enhancement (Tier 1)

###### A. Input

###### B. Agentic Visual Reasoning Process C. Final Output:

How many days did the shortest task take? (A) 28 (B) 30 (C) 32 (D) 34

Stage 1: Visual Perception Enhancement

Stage 2: Quantitative Visual Estimation

Stage 3: Compositional Visual Reasoning

| | |
|---|---|
| | |
| | |

[Figure 43]

üShortest task

[Figure 44]

[Figure 45]

[Figure 46]

###### 30 days

ØCorrect answer:

(B) 30

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Rotate Brightness InrangeColor ConnComp DrawLine

- Fig. 1: Overview of VTC-Bench and the agentic reasoning workflow. The top panel illustrates the framework architecture, featuring a hierarchical task taxonomy and an MLLM-driven toolkit. The bottom panel exemplifies the multi-stage reasoning trajectory, evolving from visual perception enhancement to compositional reasoning.

ities such as visual question answering. Building on this progress, recent advancements have expanded their scope by integrating external tools, transforming these models into active, agentic problem solvers. This tool-use ability empowers MLLMs to move beyond basic image understanding to execute complex visual workflows for enhanced image comprehension. By strategically leveraging specialized external visual tools, MLLMs process information more effectively and complete advanced operations. This integration significantly expands their practical skills, making them substantially more versatile for real-world applications.

To assess these emerging agentic capabilities, several benchmarks [10,18,33] have been introduced to evaluate how effectively current MLLMs utilize visual tools. However, existing frameworks [10, 33] typically rely on limited tool-sets and simple invocations, rarely testing the complex combinations required for advanced visual reasoning. Furthermore, in practical applications, visual agents must dynamically adapt to a highly diverse array of available tools, and resolving real-world tasks often demands chaining multiple distinct operations together to form a successful execution plan. Failing to capture this necessary diversity and multi-tool composition, current benchmarks obscure the true operational limits of existing models, and this fundamental gap renders them inadequate for guiding the development of more reliable visual agents.

To bridge this critical gap, we introduce VisualToolChain-Bench (VTC-Bench), a comprehensive benchmark designed to rigorously evaluate the advanced tool-

use proficiency of MLLMs on foundational image-based tools. As shown in Fig. 1, to emulate authentic computer vision pipelines, our framework integrates 32 distinct visual operations derived from the OpenCV library. These operations serve as the essential building blocks for solving complex visual tasks. By leveraging this versatile tool-set, our benchmark naturally supports advanced tool combinations and multi-step reasoning strategies that reflect real-world challenges. To guarantee a thorough assessment of these capabilities, we constructed 680 meticulously designed problems organized into a nine-level cognitive hierarchy. Furthermore, every problem is paired with a ground-truth execution trajectory to enable the precise evaluation of both intermediate planning and final outcomes. This ensures models are assessed on their underlying logical reasoning rather than merely their final predictions.

We comprehensively evaluate 19 leading MLLMs on VTC-Bench to assess their visual agentic capabilities. Our extensive experiments underscore the highly challenging nature of this benchmark and reveal critical limitations in current models. The overall performance is consistently low, with the top-tier model, like Gemini-3.0-Pro, achieving only 51.2% on the benchmark. Furthermore, we observe a distinct divergence in tool utilization. While closed-source models demonstrate substantial improvements when equipped with tools, open-source models exhibit minimal gains and sometimes suffer performance degradation. These results highlight a severe pronounced disparity between the current theoretical capability and actual practical proficiency of state-of-the-art models.

To understand the fundamental limitations of current models, we further conduct detailed analysis experiments. Our evaluations reveal that existing models struggle to adapt to diverse tool-sets and generalize to unseen operations. Furthermore, multi-tool composition remains a highly persistent obstacle. We find that models heavily favor a narrow subset of familiar functions instead of actively selecting the optimal tools for a specific task. This strict reliance on known patterns causes significant operational inefficiencies and ultimately leads to execution failures during multi-step reasoning processes. By systematically exposing these specific challenges, VTC-Bench establishes a rigorous baseline to guide the future development of truly generalized visual agents.

### 2 Related Work

Visual Agentic Model Recent Multimodal Large Language Models (MLLMs) are evolving from static textual reasoning toward a dynamic visual agentic paradigm. Early tool-driven approaches [44, 46] coordinate external vision experts or fixed APIs for basic visual analysis. To enhance perception, interactive attention mechanisms, such as active zooming [29,47] and visual masking, are employed to refine inputs. Recent reinforcement learning methods [13,17,34,39, 41, 51, 52] optimize these strategies for specific toolset orchestration. However, the reliance on fixed collections of visual parsers fundamentally restricts generalizability. This rigid design confines models to predefined visual scenarios, preventing adaptation to unseen structures. Programmatic visual manipulation

- Table 1: Comparison of VTC-Bench with representative multimodal benchmarks. Compared to existing benchmarks, our work encompasses 32 diverse tools and supports dual interaction paradigms (Code & Interface). RT: Reference tool-call trajectory provided for analysis. MTC: Multi-tool composition for a single task. LHC: Long-horizon calling with extended execution steps. Tasks necessitate extensive sequential tool invocations. SFD: Strict Functional Dependency where prior outputs are mandatory inputs for subsequent tools. As exemplified in Fig. 8a, pre-processing serves as a mandatory prerequisite for accurate downstream tool execution.

Benchmark Paradigm #QA #Tasks #Tools RT MTC LHC SFD

V* [43] Static 191 2 0 ✗ ✗ ✗ ✗ HRBench [42] Static 200 6 0 ✗ ✗ ✗ ✗ GTA [40] Code 229 3 14 ✓ ✓ ✗ ✗ TIR-Bench [18] Code 1.2K 13 1/Code ✗ ✗ ✗ ✗ Agent-X [1] Code 828 5 14 ✓ ✓ ✗ ✗ VisualToolBench [10] Code 1.2K 5 6 ✓ ✓ ✗ ✗ AgentVista [33] Code 209 7 4 ✗ ✓ ✓ ✗

Ours Code & Interface 680 9 32 ✓ ✓ ✓ ✓

addresses these limitations by utilizing python code as a primitive tool [12,35]. This approach enables on-demand tool construction with complex logic, including loops and conditionals. Advanced frameworks [6,14,28,38,50] dynamically generate code for targeted visual editing, while Thyme [48] provides code testing for open-source tool-use models. Leading models, including GPT-o3 [27], GPT-o4-mini [27], and GPT-5.2 [26], leverage code execution to construct taskspecific tools dynamically. Consequently, as state-of-the-art models increasingly embrace this agentic tool-calling paradigm, there is a critical need for a benchmark equipped with a sufficiently diverse tool library to rigorously evaluate their complex, multi-tool compositional capabilities.

Agentic Benchmark in MLLM Benchmark Standard evaluations of multimodal large language models primarily focus on static perception and reasoning. Previous works [5, 19, 21, 23, 24, 30, 32, 45] test models using static questions and treat vision as a passive input. For visual agent evaluation, some studies [42, 43] introduce active visual exploration tasks. However, these early evaluations only examine basic operations like cropping and zooming. Recent research [1,10,11,18,33,40] further advances this field. These methods evaluate multimodal agentic reasoning [11], assess image processing capabilities [18], and combine multiple tools for open-ended visual tasks [10]. Despite these advancements, existing benchmarks are inherently constrained by limited tool inventories and lack systematic requirements for compositional multi-tool reasoning, and often fail to capture the nuanced demands of practical, real-world applications. In contrast, deeply rooted in authentic real-world tasks, our proposed benchmark explicitly targets tool diversity and the complexity of multi-step tool composition. We design 680 problems requiring complex multi-step tool combinations. Models can flexibly call and combine 32 distinct OpenCV [16] tools. Agents address these tasks by synthesizing Python code or utilizing our predefined interface, thereby comprehensively evaluating their ability to generate programmatic solutions for deep visual reasoning. A detailed comparison be-

- Tier 1: Visual Perception Enhancement

- Tier 2: Quantitative Visual Estimation

- Tier 3: Compositional Visual Reasoning

Robust OCR Attention Focusing Perceptual Restoration

Q：This is a left-right mirrored image. Please tell me what the first character in the first row is after restoring the image.

Q: In this picture of smog, what is the white object in the upper left corner of the screen? (A) Foam board (B) Chair (C) Fence (D) Low wall

Q: What's the name of the store near '雞大爺' whose sign is mirrored?

|[Figure 52]|
|---|

|[Figure 53]|
|---|

[Figure 54]

ToolChain: Crop ➜ Flip ➜ Convert Color ➜ Histogram Eq

ToolChain: Flip Convert ➜ Color ➜ Histogram Eq ➜ Binarize

ToolChain: Crop➜ Convert Color ➜ Adjust Brightness ➜ Histogram Eq

➜ Sharpen

Answer: 天

Answer: 光阳机车

Answer: (B) 4

Chart Measure Counting

Q:According to the chart, what is the number of people in China who think "less"? (A) 49 (B) 50 (C) 51 (D) 53

Q:Please measure the length of the item in the image, how many cm is it? (A) 8.0 (B) 8.3 (C) 8.1 (D) 8.2

Q:Count the total number of rice grains in the picture. (A) 103 (B) 117 (C) 128 (D) 136

|[Figure 55]|
|---|

|[Figure 56]|
|---|

|[Figure 57]|
|---|

ToolChain: Convert Color ➜ Blur ➜ Binarize ➜ Morphology ➜ Connected Components

ToolChain: Convert Color ➜ Color Filter ➜ Line Detect ➜ Draw Line

ToolChain: Convert Color ➜ Color Filter ➜ Draw Contours ➜ Measure Area

Answer: (B) 50

Answer: (A) 8.0

Answer: (B) 117

Math Spatial Reasoning Color

Q:Between the red door and the yellow door on two adjacent floors, which color door is on the lower floor?

|[Figure 58]|
|---|

|[Figure 59]|
|---|

|[Figure 60]|
|---|

Q:Solve the math problem in the picture.

Q:How many yellow chips are there in the stack of chips on the right? (A) 7 (B) 8 (C) 9 (D) 10

ToolChain: Convert Color ➜ Binarize ➜ Draw Contours

(A) Red door (B) Yellow door (C) They are on the same floor

➜Approximate

ToolChain: Convert Color ➜ Color Filter ➜Connected Components

ToolChain: Crop ➜ Convert Color ➜ Color Filter ➜ Morphology ➜ Connected Components

➜ Polygon ➜ Draw Line

Answer: 540

Answer: (B) 8

Answer: (A) Red door

- Fig. 2: Overview of VTC-Bench. Our benchmark’s progressive design structures 9 VQA tasks into a three-tier cognitive hierarchy, evaluating MLLMs’ multi-tool orchestration from basic visual recovery to high-level logical deduction.Each task provides a reference toolchain to enable fine-grained diagnostic analysis.

tween VTC-Bench and existing benchmarks is presented in Tab. 1 to highlight our unique contributions.

- 3 VisualToolChain-Bench

This section introduces VisualToolChain-Bench(VTC-Bench), with an overview provided in Fig. 2. We first present the benchmark design in Sec. 3.1 by establishing a systematic task taxonomy and corresponding toolset. Building upon this foundation, Sec. 3.2 details the benchmark construction process, encompassing data collection, statistical analysis of the dataset, and evaluation metrics.

##### 3.1 Benchmark Design

Tool Set Due to OpenCV’s extensiveness and versatility, we identify OpenCV [16]

as our primary tool source to address the sparse tool diversity in the existing benchmarks. We curated 32 tools, aligning our selection with the standard human cognitive pipeline: initial restoration, feature distillation, and verification. These tools are organized into four functional modules: (1) Geometry for spatial transformations (e.g., rotation and image pyramids); (2) Enhancement for signal optimization (e.g., color space conversion and binarization); (3) Feature Extraction for deriving structural and semantic primitives (e.g., edge detection and watershed segmentation); and (4) Drawing for reasoning verification and attribute quantification (e.g., contour visualization and area measurement). This integrated suite enables controlled visual operations for various MLLMs, with a more comprehensive taxonomy and technical definitions detailed in the App. B.1.

Statistic Number Total questions 680 Multiple-choice 538 (79.12%) Open 142 (20.88%) Questions with image 680 (100.00%) Average chain length 5.04 Average unique tools 4.97 Total tool calls 3,428 Min chain length 1 Median chain length 5 Max chain length 10 Average prompt length 18.52

[Figure 61]

- Table 2: Statistical summary of VTC-Bench.

Fig. 3: Domain distribution (inner) & toolchain lengths (outer). Lengths range from 1-3 to 7+, where darker shades denote longer steps.

Task Design Rather than a fragmented collection of benchmarks, our evaluation suite is structured around a cognitive hierarchy, comprising 9 tasks designed to map the evolution of multimodal agents from passive visual sensing to active constructive reasoning. This hierarchy is organized into three progressive tiers:

- Tier 1: Visual Perception Enhancement. This foundational stage comprises

Robust OCR, Perceptual Restoration, and Attention Focusing. These tasks require models to employ specialized tools to mitigate environmental interference (e.g., haze, low light) and rectify geometric distortions (e.g., rotation). Specifically, Robust OCR targets text recognition under synthetic degradation that remains human-readable; Perceptual Restoration focuses on scene recovery in adverse conditions such as haze or low light; and Attention Focusing emphasizes fine-grained analysis under geometric transformations like rotation or flipping.

- Tier 2: Quantitative Visual Estimation. Building upon the foundational stage,

the tasks of Measurement, Color, and Counting evaluate the model’s capacity to perceive and precisely quantify physical attributes. Specifically, Measurement requires extracting size, position, and shape; Color examines the precise extraction of chromatic information; and Counting focuses on scene analysis and the strategic invocation of specialized counting tools, rather than relying on the model’s intrinsic counting capabilities.

- Tier 3: Compositional Visual Reasoning. Finally, the Chart, Math, and Spa-

tial Reasoning tasks demand complex logical deduction through multi-step tool orchestration. Chart is a comprehensive task requiring simultaneous restoration, perception, and inference. Math evaluates the construction of auxiliary geometric elements, while Spatial Reasoning tests the robust analysis of spatial relations under extreme conditions, such as overexposure or heavy blur.

This hierarchical taxonomy not only ensures comprehensive evaluation dimensions but also reveals the complete cognitive spectrum of multimodal agents,

- Table 3: Overview of 9 distinct tasks in our benchmark. All tasks feature human-annotated questions and single-choice answers to ensure rigorous evaluation.

Task Category Data Source QA Construction Samples Answer Type

Attention Focusing HRBench [42] & Web Human-annotated 45 Open-ended Chart ChartQA [25] & Web Human-annotated 100 Single-choice Color ColorBench [20] & Web Human-annotated 90 Single-choice Counting Kaggle & Web Human-annotated 85 Single-choice Math Web & Manual Construction Human-annotated 110 Single-choice Measurement Web Human-annotated 105 Single-choice Perceptual Restoration Kaggle & Web Human-annotated 50 Single-choice Robust OCR OCRBench [22] & Kaggle Human-annotated 50 Open-ended Spatial Reasoning Kaggle & Web Human-annotated 45 Single-choice

marking a transition from passive visual perception to the sophisticated active constructive capability known as visual agentic model. Consequently, a detailed conceptual overview of the proposed benchmark is illustrated in Fig. 2.

##### 3.2 Benchmark Construction

Data collection Our data curation pipeline is guided by several core principles to ensure a rigorous evaluation. We combine web-crawled images with the strategic repurposing of open-source datasets to balance contextual breadth and environmental noise. Rather than relying on original annotations, we synthesize novel instructions from a tool-centric perspective to compel the exploration of latent logical reasoning. This approach transforms static samples into dynamic challenges that require multi-hop execution. Furthermore, we introduce controlled visual perturbations, such as geometric distortions and radiometric noise, to evaluate model robustness. These non-ideal conditions necessitate a transition from passive recognition to active toolchain planning for image restoration.

Verification Protocol VTC-Bench follows a rigorous verification protocol to ensure data integrity. Expert annotators first sanitize images by removing metadata and extracting initial labels. We utilize MLLMs including Gemini-3.0Pro [9] and GPT-5.2 [26] strictly to validate these manual annotations. Subsequently, Gemini-3.0-Pro drafts the reference toolchains for all samples. Expert researchers then conduct a secondary manual verification on these generated trajectories to determine the finalized ground truth. Finally, a reciprocal auditing phase achieves consensus on accuracy through mutual cross-verification. This rigorous pipeline ultimately yields 680 high-quality robust samples. A technical summary of the entire data collection process is shown in Tab. 3.

Benchmark Statistics As summarized in Tab. 2, VTC-Bench comprises a diverse set of 680 VQA instances, consisting of 538 multiple-choice and 142 openended questions. Each question includes a detailed reference toolchain with an average length of 5.04 steps and 4.97 unique tools, indicating the high complexity of the required operations. Overall, the dataset contains a total of 3,428 tool calls, with chain lengths ranging from 1 to 10 and a median of 5. Furthermore, the average prompt length across all questions is 18.52 words. To better visual-

- Track A: Code Interpreter
- Track B: Atomic OpenCV Toolbox

Max Round

Sandbox

- (a) Code-driven
- (b) Interface-driven

Thought: I’ll use grayscale and Otsu to segment and count filtered grains.

[Figure 62]

Reasoning&ExecutionLayer

|image = cv2.imread(image_path) gray = cv2.cvtColor(…) blurred = cv2.GaussianBlur(…) opening = cv2.morphologyEx(…) contours, _ = cv2.findContours(…) ……..|
|---|

Q: Count the total number of rice grains in the picture.

Call: code_interpreter(…)

[Figure 63]

[Figure 64]

|[Figure 65]|
|---|

stdout: Refined count of rice grains: 117

Options: (A) 117 (B) 92 (C) 135 (D) 104

[Figure 66]

Agent

|[Figure 67]|
|---|

ToolBox

[Figure 68]

Thought: First, I need to find the sign of "雞大爺"

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

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

Call: Clip(img, [x, y, w, h])

Q: Locate the shop sign near '雞大 爺' that features mirrored Chinese

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

characters and identify what they are.

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

Iteration

|[Figure 102]|
|---|

Agent

[Figure 103]

[Figure 104]

Global Context-Aware Summary: Synthesizing thoughts, code outputs, and tool observations from the history stack.

###### Final Answer

###### History Stack

(Thoughts, Code, Tools, Observations)

- Fig. 4: Overview of workflows for two tool-use protocols of LLM agents during evaluation. The agent iteratively interacts with tools within the Max Round constraint, ultimately synthesizing tool execution results to generate the final answer.

ize this, Fig. 3 illustrates the distribution of each task category, while App. E provides representative image and question examples for further clarity.

##### 3.3 Evaluation Metrics

We adopt Average Pass Rate (APR) as the primary evaluation metric to represent the proportion of correctly answered questions. To analyze tool-use behavior more granularly, we define the Effective Toolchain as the minimal sequence of tool calls to produce the final answer. This sequence is determined through a backtracking process from the final output to the original input image.

We evaluate the models using three additional metrics: Tool Call Rate (TCR), which measures the proportion of tasks where a model invokes at least one tool; Mean Absolute Error (MAE), which quantifies the discrepancy in length between the predicted and ground-truth toolchains; and Tool Usage Efficiency (Efftool), which assesses the precision and conciseness of the tool-calling sequences by comparing the number of effective steps to the total predicted steps. Mathematically, MAE and Efftool are formulated as Eq. 1:

1 N |LG,i − LT,i|, Efftool =

Le,i LT,i

MAE =

(1)

where N represents the total number of evaluated samples. The variables LG,i, LT,i, and Le,i denote the length of the ground-truth toolchain, the total toolchain, and the length of the effective toolchain for the i-th sample, respectively.

- Table 4: Comprehensive evaluation results. Bold and Underlined indicates the highest value in each column. For Overall and Delta columns: Green indicates improvement and Red indicates degradation compared to base. For subcategories,

Yellow indicates an improvement of more than 10% compared to base.

Model Size Setting Overall Delta OCR Attn. Rest. Chart Meas. Count. Math Spat. Color Category 1: Proprietary Tool-use Models

- GPT-o3 — Base 31.91 20.00 26.67 32.00 30.00 38.10 35.29 28.18 35.56 35.56 Code 36.62 +4.71 28.00 31.11 34.00 40.00 38.10 34.12 30.91 51.11 42.22 Inter. 36.76 +4.85 28.00 24.44 34.00 39.00 35.24 40.00 34.55 44.44 44.44

- GPT-o4-mini — Base 31.18 18.00 13.33 30.00 29.00 33.33 44.71 30.91 42.22 30.00 Code 33.68 +2.50 12.00 11.11 30.00 41.00 31.43 35.29 40.00 46.67 37.78 Inter. 32.50 +1.32 24.00 13.33 28.00 30.00 33.33 40.00 33.64 46.67 35.56

###### Category 2: Proprietary General-purpose Models

GPT-4o — Base 24.26 18.00 15.56 22.00 25.00 25.71 25.88 22.73 26.67 30.00 Code 31.62 +7.36 18.00 22.22 24.00 38.00 37.14 40.00 20.91 51.11 30.00 Inter. 33.82 +9.56 20.00 22.22 36.00 31.00 41.90 40.00 24.55 48.89 37.78

- Gemini-2.5-Pro — Base 38.09 48.00 26.67 28.00 44.00 40.95 49.41 30.00 40.00 32.22 Code 36.03 -2.06 46.00 40.00 34.00 37.00 35.24 37.65 30.00 44.44 31.11

- Inter. 39.85 +1.76 60.00 44.44 40.00 38.00 44.76 41.18 26.36 35.56 40.00

GPT-5.2 — Base 36.03 24.00 24.44 30.00 43.00 35.24 43.53 33.64 46.67 35.56 Code 44.56 +8.53 36.00 31.11 26.00 56.00 50.48 44.71 45.45 51.11 42.22

- Inter. 40.74 +4.71 28.00 37.78 34.00 47.00 39.05 37.65 41.82 51.11 44.44

- Gemini-3.0-Flash — Base 46.47 44.00 51.11 46.00 57.00 50.48 52.94 27.27 60.00 40.00 Code 50.59 +4.12 70.00 57.78 48.00 59.00 50.48 52.94 35.45 51.11 44.44

- Inter. 50.74 +4.27 68.00 53.33 50.00 61.00 44.76 58.82 39.09 55.56 40.00

Gemini-3.0-Pro — Base 44.41 54.00 51.11 42.00 51.00 45.71 42.35 34.55 53.33 37.78 Code 51.18 +6.77 74.00 62.22 38.00 60.00 54.29 36.47 41.82 55.56 50.00

- Inter. 51.03 +6.62 70.00 73.33 38.00 59.00 50.48 35.29 37.27 71.11 50.00

###### Category 3: Open-source Tool-use Models

V-Thinker 7B Base 22.06 8.00 11.11 22.00 27.00 23.81 27.06 18.18 28.89 24.44 Code 23.82 +1.76 10.00 8.89 24.00 21.00 23.81 43.53 17.27 42.22 22.22 Inter. 24.41 +2.35 16.00 8.89 34.00 23.00 21.90 34.12 17.27 44.44 25.56

DeepEyes-v2 7B Base 24.85 22.00 11.11 18.00 27.00 38.10 21.18 20.00 28.89 26.67 Code 30.00 +5.15 20.00 8.89 20.00 30.00 44.76 32.94 28.18 44.44 26.67 Inter. 23.38 -1.47 18.00 6.67 28.00 16.00 28.57 30.59 18.18 35.56 27.78

DeepEyes 7B Base 26.47 12.00 8.89 28.00 34.00 33.33 27.06 17.27 44.44 27.78 Code 27.06 +0.59 22.00 8.89 28.00 30.00 32.38 36.47 17.27 40.00 25.56 Inter. 29.26 +2.79 22.00 6.67 28.00 29.00 43.81 37.65 11.82 46.67 33.33

Thyme 7B Base 27.35 8.00 8.89 34.00 29.00 41.90 35.29 18.18 35.56 24.44 Code 28.82 +1.47 18.00 2.22 24.00 35.00 40.00 38.82 20.91 33.33 28.89 Inter. 27.06 -0.29 14.00 4.44 24.00 33.00 33.33 34.12 20.00 40.00 28.89

Category 4: Open-source General-purpose Models Qwen3-VL-Instruct 8B Base 30.74 32.00 11.11 28.00 27.00 31.43 38.82 31.82 35.56 33.33

Code 28.24 -2.50 26.00 17.78 20.00 27.00 39.05 29.41 23.64 17.78 37.78

- Inter. 28.68 -2.06 30.00 17.78 34.00 20.00 30.48 42.35 24.55 26.67 31.11

Qwen3-VL-Thinking 8B Base 30.29 20.00 11.11 24.00 33.00 40.00 35.29 29.09 35.56 28.89 Code 26.32 -3.97 12.00 6.67 22.00 29.00 32.38 36.47 22.73 20.00 34.44

- Inter. 29.41 -0.88 20.00 15.56 22.00 34.00 32.38 37.65 24.55 28.89 35.56

Qwen3-VL-Instruct 32B Base 32.79 28.00 8.89 26.00 37.00 31.43 48.24 27.27 44.44 34.44 Code 30.59 -2.20 28.00 15.56 20.00 32.00 31.43 34.12 33.64 31.11 35.56 Inter. 33.09 +0.30 34.00 13.33 16.00 21.00 37.14 51.76 29.09 51.11 38.89

Qwen3-VL-Thinking 32B Base 32.94 20.00 4.44 28.00 32.00 40.95 43.53 34.55 42.22 32.22 Code 33.09 +0.15 22.00 8.89 26.00 41.00 35.24 43.53 31.82 33.33 35.56 Inter. 33.97 +1.03 30.00 17.78 28.00 32.00 35.24 44.71 31.82 40.00 37.78

Qwen3-VL-Instruct 30B-A3B Base 33.09 42.00 15.56 22.00 23.00 41.90 41.18 32.73 42.22 32.22 Code 31.03 -2.06 42.00 22.22 32.00 31.00 31.43 36.47 23.64 37.78 28.89 Inter. 31.91 -1.18 40.00 24.44 28.00 25.00 37.14 40.00 28.18 44.44 25.56

Qwen3-VL-Thinking 30B-A3B Base 29.56 16.00 8.89 22.00 31.00 30.48 44.71 24.55 46.67 32.22 Code 28.24 -1.32 18.00 13.33 22.00 35.00 36.19 23.53 27.27 40.00 27.78

- Inter. 33.24 +3.68 28.00 15.56 24.00 29.00 39.05 43.53 30.00 48.89 34.44

Qwen3-VL-Instruct 235B Base 36.32 42.00 11.11 28.00 33.00 40.95 48.24 35.45 35.56 38.89 Code 33.68 -2.64 36.00 15.56 28.00 26.00 40.00 42.35 30.00 46.67 35.56

- Inter. 34.85 -1.47 40.00 22.22 20.00 32.00 38.10 48.24 27.27 37.78 41.11

Qwen3-VL-Thinking 235B Base 35.15 34.00 8.89 24.00 38.00 39.05 45.88 32.73 37.78 38.89 Code 34.41 -0.74 36.00 22.22 26.00 37.00 36.19 35.29 34.55 37.78 36.67 Inter. 38.09 +2.94 40.00 24.44 38.00 35.00 32.38 52.94 34.55 48.89 38.89

### 4 Experiment

##### 4.1 Experiment Setup

Model Selection We evaluate 19 mainstream MLLMs, encompassing both proprietary and open-source models with diverse tool-use training backgrounds. We stratified these evaluated models into four distinct categories. The first category comprises proprietary tool-use models, featuring closed-source systems with robust native capabilities like GPT-o3 [27] and GPT-o4-mini [27]. The second category incorporates leading proprietary general-purpose models, such as GPT-

###### 5.2 [26], Gemini-3.0-Pro [9], Gemini-3.0-Flash [8], Gemini-2.5-Pro [4], and GPT-

- 4o [27]. Transitioning to the open-source domain, the third group consists of dedicated open-source tool-use models, including specialized architectures such as DeepEyes [51], Thyme [48], DeepEyesV2 [13], and V-Thinker [28]. Finally, for open-source general-purpose models, we evaluate both the Instruct and Thinking variants of Qwen3-VL [2] across parameter scales ranging from 8B to 235B. Evaluation Framework To measure the benefits of tool invocation, we compare direct answering baselines against tool-augmented reasoning across both interface-driven and code-driven paradigms. This comparison reveals how a model’s programming proficiency influences overall orchestration success. The evaluation workflow is illustrated in Fig. 4. We utilize the Qwen-Agent [37] framework to manage both code and interface implementations for models with robust native tool-calling support. Conversely, we employ the Thyme [48] framework for models with limited native tool-calling proficiency, applying this approach primarily to open-source tool-use models to facilitate code generation. To facilitate this process, we provide detailed tool descriptions that explicitly outline usage instructions and parameter specifications, ensuring that even models lacking specialized OpenCV training can comprehend and invoke the available interfaces. Finally, we implement a comprehensive dual-stream evaluation protocol, integrating deterministic rule-based matching with a GPT-4o [15] LLM-as-a-Judge paradigm. Detailed prompt templates are provided in the App. D.

##### 4.2 Main Result

The benchmark presents a significant challenge to current MLLMs. Performance in the base setting remains relatively low, with scores ranging from 22.06% to 46.47%. Gemini-3.0-Flash [8] achieves the highest base score of 46.47%. Most models cluster around the 30% mark. Notably, simply scaling up model parameters does not inherently solve this critical bottleneck; for instance, Qwen3VL-235B-A22B [2] achieves a performance improvement from 30.74 to 36.32% compared to Qwen3-VL-8B; however, it still struggles to surpass 40% even with tool augmentation. This universal performance ceiling underscores that transitioning from passive visual perception to active, multi-step tool orchestration remains an unsolved frontier, even for advanced industry-leading architectures. Proprietary models significantly outperform open-source models. As illustrated in Tab. 4, our evaluation reveals a pronounced performance disparity

between proprietary and open-source models, particularly concerning their capacity to leverage external tools. Proprietary models not only establish robust baselines but also exhibit substantial performance surges when augmented with tool-use capabilities; notably, GPT-4o achieves a remarkable gain of +9.56% under the interface setting, while GPT-5.2 realizes an +8.53% improvement in the code setting. Furthermore, Gemini-3.0-Pro attains the highest overall score of 51.18% with tools, underscoring the advanced reasoning and execution stability inherent in closed-source paradigms. Conversely, open-source models frequently fail to effectively harness tool integration, often experiencing limited gains when augmented with tool-use capabilities, indicating a considerable gap between open-source and proprietary models in native tool-use capabilities.

Models perform better at coarse-grained tool invocation, and intrinsic perception is the prerequisite for tool use. In visual perception enhancement tasks, models exhibit strong performance, with GPT-4o boosting Perceptual Restoration by +14.00% via interface. These tasks require only basic, coarse-grained tool operations, such as rotation or mirroring, to achieve significant improvements. Conversely, quantitative visual estimation yields bifurcated outcomes, with noticeable degradation in high-precision perception tasks. Because Tier 2 rigorously tests precise tool selection and parameter configuration, it exposes models’ inability to execute fine-grained tool manipulations. Finally, in compositional visual reasoning tasks, models achieve remarkable leaps, highlighted by a +17.78% improvement in Gemini-3.0-Pro’s Spatial Reasoning score. This resurgence stems from the comprehensive nature of Tier 3, which simultaneously challenges intrinsic perception and tool proficiency. Notably, models with stronger foundational perception excel here, underscoring that robust intrinsic perception is an indispensable prerequisite for advanced tool calling.

The proposed toolset provides robust support for complex visual reasoning. Notably, contrary to the common assumption that code execution is inherently superior, our findings demonstrate that interface-based invocation is comparable to code-based approaches overall. Specifically, among the five proprietary general-purpose models evaluated, three exhibited better performance with interface-based methods. These quantitative results further validate that our structured 32-tool library effectively meets the requirements of visual agents.

##### 4.3 Analysis Experiments

Analysis of Tool Utilization and Efficiency. The left panel of Fig. 5 reveals a positive correlation between Tool Call Rate (TCR) and Average Pass Rate (APR), validating the utility of active tool invocation. Conversely, the efficiency analysis in the right panel indicates that general-purpose models achieve superior efficiency even with lower TCR. Specialized models cluster in the highredundancy region at the bottom right. Their specialized training incentivizes frequent tool usage but often causes them to neglect their inherent perceptual capabilities. Consequently, a disconnect exists between perception and tool calling in these architectures. Our results indicate that current specialized models fail to integrate these two abilities effectively.

- Table 5: Statistical analysis of tool invocation performance. Beyond the Toolcalling Rate(TCR) and Average Pass Rate (APR), we provide a fine-grained evaluation categorized by total attempted tool calls (All Calls) and the effective toolchain (Effective Calls). For both categories, we detail the average number of calls (Avg. T.), the number of unique tools used (Uniq. T.), and the Mean Absolute Error (MAE) from

the ground-truth length. Finally, the Tool Usage Efficiency (Efftool, denoted as Eff.) represents the ratio of effective steps to total attempted steps.

All Calls Effective Calls

Model TCR (%) APR (%)

Eff. (%) Avg. T. Uniq. T. MAE Avg. T. Uniq. T. MAE

DeepEyes 66.32 29.26 1.90 1.04 4.09 1.21 0.90 3.97 63.68 Qwen3-VL-235B 86.62 38.09 2.19 1.77 3.02 1.51 1.47 3.36 68.95 GPT-5.2 94.26 40.74 13.23 4.08 9.96 2.22 1.97 3.01 16.78 GPT-o3 93.68 36.76 9.77 3.47 6.98 1.68 1.51 3.32 17.20 GPT-o4-mini 76.62 32.50 3.63 1.77 3.63 1.04 1.02 3.76 28.65 Gemini-3.0-Pro 93.97 51.03 5.04 2.72 3.87 1.84 1.76 3.19 36.51 GPT-4o 65.44 33.82 2.26 1.81 3.91 1.38 1.26 3.82 61.06 Gemini-3.0-Flash 93.24 50.74 1.89 1.38 3.26 1.20 1.19 3.64 63.49

70

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

50

50

ToolCallEfficiency(Eff.%)

60

Models

Accuracy(APR%)

Accuracy(APR%)

45

45

DeepEyes Qwen3-VL

50

- GPT-o3

- GPT-o4-mini Gemini 3 Flash Gemini 3 Pro

40

40

40

| |
|---|

35

35

30

| |
|---|

- GPT-4o

- GPT-5.2

20

30

30

70 80 90

20 40 60

70 80 90

Tool Call Rate (%)

Tool Call Efficiency (Eff. %)

Tool Call Rate (%)

###### Fig. 5: Relationships among Tool Call Rate, Tool Call Efficiency, and APR. The panels illustrate APR vs. Tool Call Rate (left), APR vs. Tool Call Efficiency (middle), and Tool Call Rate vs. Tool Call Efficiency (right).

Uneven Distribution of Tool Utilization. Furthermore, we analyze the tooluse distribution within representative models from the Gemini and GPT series.

###### Fig. 6 reveals that a small number of tools account for a disproportionately large percentage of total calls. For instance, models frequently utilize basic tools such as crop, zoom in, and rotate. This concentration suggests that existing models possess limited diversity in their tool selection. They primarily rely on simple and common operations that were likely heavily emphasized during their training. These empirical results indicate that most current models still lack the capability for diverse and complex tool invocation. Tool composition enhances performance but remains a fundamental execution bottleneck. Visualized in Fig. 7a, multi-round combinations positively correlate with overall performance, with trajectories exceeding two steps consistently surpassing average accuracy. However, distributional discrepancies (Fig. 7b and Fig. 10) alongside the quantitative metrics in Tab. 5 expose a striking inefficiency. Paradoxically, even models with high APR fail to align with optimal, ground-truth tool-calling patterns. For example, despite its advanced reasoning capabilities, GPT-5.2 records a MAE of 9.96 in attempted calls and an

###### GPT-o3

###### GPT-5.2

###### Gemini-3.0-Pro

###### Gemini-3.0-Flash

6.1%

9.6%

5.4%

5.8%

5.5%

6.1%

6.0%

11.8%

6.5%

29.5%

7.1%

8.9%

39.8%

6.6%

43.1%

10.0%

53.0%

11.7%

6.9%

11.8%

Zoom In Crop

Inrange Color Connected Components

Histogram Rotate

Threshold Hough Lines

Resize Hough Circles

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Fig. 6: Distribution of tool utilization across different models. The usage patterns remain largely consistent across models, with the most frequently invoked tools including Zoom In, Inrange Color, Rotate, Histogram, and Connected Components.

0 1 2 3 4+

Total Tool Call Count

0

100

200

300

400

500

600

QuestionCount

Question Count

0 1 2 3 4+

Number of Unique Tools

Question Count

0

20

40

60

80

100

Accuracy(%)

(a) Sample distribution vs. Accuracy.

0 1 2 3 4 5 6 7 8 9 10

Total Number of Tools in Chain

0

100

200

300

400

500

600

Count(NumberofSamples)

GT ToolChain

Model ToolChain

0 1 2 3 4 5 6 7 8 9

Number of Unique Tool Types

GT ToolChain

Model ToolChain

(b) Distributional Discrepancy.

- Fig. 7: Comprehensive analysis for Gemini-3.0-Flash. (a) Correlation between sample density and APR. (b) Distributional gap between Ground Truth and Model. Both metrics are stratified by chain length (left) and tool diversity (right).

Eff. of 16.78%. Similarly, GPT-o3 and Gemini-3.0-Pro only achieve efficiencies of 17.20% and 36.51%, respectively. These pronounced deviations and excessive redundancies demonstrate that instead of executing precise programmatic orchestration, current models resort to suboptimal trial-and-error heuristics, exposing a critical deficit in coordinating complex, multi-step tool sequences.

Case Study A qualitative analysis of model outputs reveals two primary failure modes in tool utilization. The first mode involves a compounded failure of strategic misselection and operational errors during tool invocation. As Fig. 8a illustrates, the model not only selects the Draw Circle and Draw Line tools—which are fundamentally inappropriate for the given context—but also executes them incorrectly. Specifically, it attempts to apply these tools based solely on its flawed intrinsic perception, entirely bypassing the prerequisite step of extracting accurate spatial coordinates. This highlights a critical deficiency in understanding specific tool functionalities and their proper application contexts. The second mode is an over-reliance on intermediate tool results. Fig. 8b demonstrates this behavior, where the model conducts a superficial analysis of the returned output and directly adopts it to form the final answer. By failing to cross-verify these intermediate results with the original visual input, the model completely misses the correct target. This lack of critical verification prevents models from identifying and correcting potential errors generated by the tools.

Incorrect Case

|Call: In Range Color Args: colorspace='hsv’, range='red’ Return: IMG|
|---|

Incorrect Case

[Figure 105]

|Call: Zoom In Args: ((100, 120, 150,<br><br>80) ,Scale:2.0) Return: IMG|
|---|

Thought: Isolate red pieces using color segmentation.

[Figure 106]

Thought: Zoom in on eyes for accurate measurement.

|[Figure 107]|
|---|

| | |
|---|---|
| | |
| | |

Thought: Filtering 37 noisy components to 6 based on area. Output: C. 6

[Figure 108]

Question: What is the approxi mate distance in pix els between the left and right pupils of th e person the image?

Thought: Calc: |260 - 140| = 120px. Still mismatches. Guesses closest option (100).

###### Question:

How many red chess pieces are in the pict ure?

|Call: Draw Circle Args: (80,70) & (200,70) Return: IMG|
|---|

|[Figure 109]|
|---|

[Figure 110]

###### Thought:

|Call: Connected Components Args: connectivity=8 Return: Found 37 components|
|---|

Thought: Count isolated regions on the color mask.

[Figure 111]

[Figure 112]

Mark pupil centers on zoomed image.

[Figure 113]

Output: B. 100

|[Figure 114]|
|---|

Options:

Correct Case

###### Thought:

|Call: Draw Line Args: (140,190) to (260,190) Return: IMG|
|---|

[Figure 115]

|Call: In Range Color Args: colorspace='hsv’, range='red’ Return: IMG|
|---|

A. 7 B. 4 C. 6 D. 5

Thought: Isolate red pieces using color segmentation.

[Figure 116]

[Figure 117]

Calc: |200 - 80| / 2.0 = 60px. Mismatches options. Switch to Draw line.

Options:

[Figure 118]

[Figure 119]

- A.90
- B. 100
- C.70
- D.80

|[Figure 120]|
|---|

Thought: Visual check confirms exactly 7 distinct red pieces.

[Figure 121]

Correct Case

|Call: Morphology Args: op='open', kernel=(5,5) Return: IMG|
|---|

Thought: Apply morphological opening to remove noise.

|Call: Zoom In Args: ((100, 120, 150, 80) ,Scale:2.0) Return: IMG|
|---|

|[Figure 122]|
|---|

[Figure 123]

Thought: Zoom in on eyes for accurate measurement.

[Figure 124]

[Figure 125]

Output: A. 7

[Figure 126]

Call: Connected Components Args: connectivity=8 Return: Found 9 components

Thought: Calc: |250 - 70| / 2.0 = 90px. The answer is A. Output: A. 90

|Call: Threshold Args: (Otsu, "threshold": 111.0) Return: IMG|
|---|

Thought: Count isolated regions on the cleaned mask.

[Figure 127]

Thought: Binarize image to isolate pupils.

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

|Call: Hough Circles Args: (minDist=100, minR=3, maxR=15) Return: (70,70), (250,70)|
|---|

|[Figure 133]|
|---|

|Call: Draw Lines Args: img=orig_img, points=centroids Return: IMG|
|---|

Thought: Draw lines on centers for visual verification.

Thought: Detect circles to extract coordinates.

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

(a) Tool selection and execution errors.

(b) Over-reliance on unverified tool outputs.

- Fig. 8: Qualitative analysis of failure modes in tool utilization. (a) The agent mistakenly invokes Draw Circle and Draw Line based on unfounded intuition, leading to a speculative answer. (b) The agent blindly adopts preliminary tool outputs without visual validation, bypassing essential verification steps.

DeepEyes Qwen3-VL-30B-A3B Gemini-3.0-Flash

0

10

20

30

40

50

APR(%)

26.47

29.56

46.47

27.35

31.03

48.68

29.26

33.24

50.74

29.27

35.88

52.06

Direct

Weak Prompt

Strong Prompt

Strong Prompt + GT Tools

- Fig. 9: Impact of system prompts and prompt information density on model APR. We evaluate the performance of three distinct models, under four prompting configurations: Direct, Weak Prompt, Strong Prompt, and Strong Prompt + GT Tools.

Prompt Ablation We evaluate model performance across four distinct settings—direct answer, weak prompt, strong prompt (the default configuration for our main evaluations), and strong prompt augmented with ground-truth (GT) tools—with results detailed in Fig. 9. Our evaluations on Gemini-3.0-Flash [8], Qwen3-VL-30B-A3B [2], and DeepEyes [51] reveal a consistent upward trajectory in performance correlating with prompt comprehensiveness. Notably, the performance improvements observed upon introducing GT tools effectively validate the utility and correctness of our annotated reference toolchains. Nevertheless, the overall gains remain surprisingly bounded even for leading models like Gemini and Qwen, while improvements for DeepEyes are entirely marginal. This performance ceiling suggests that providing the correct toolset is merely half the battle: restricted by their inherent reasoning capacities, current models fundamentally struggle to synthesize the correct multi-step execution logic. This inability to reliably formulate tool trajectories, even with oracle tool knowledge, underscores a profound bottleneck in complex compositional reasoning.

### 5 Conclusion

In this paper, we introduced VTC-Bench, a meticulously crafted benchmark designed to evaluate the multi-step tool composition capabilities of MLLMs. By integrating 32 diverse tools across 9 real-world categories, our suite substantially surpasses prior evaluations. Extensive experiments across 19 MLLMs , including both open and proprietary, reveal pronounced deficiencies in current tool orchestration. Models frequently gravitate toward a narrow subset of tools, resulting in suboptimal execution efficacy. Moreover, models often struggle with precise tool invocation or demonstrate an over-reliance on tool outputs, neglecting their intrinsic perceptual capabilities. These findings underscore a critical disconnect between foundational visual perception and active agentic reasoning, illuminating a clear frontier for future MLLM architectures.

### References

- 1. Ashraf, T., Saqib, A., Ghani, H., AlMahri, M., Li, Y., Ahsan, N., Nawaz, U., Lahoud, J., Cholakkal, H., Shah, M., et al.: Agent-x: Evaluating deep multimodal reasoning in vision-centric agentic tasks. arXiv preprint arXiv:2505.24876 (2025)
- 2. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631

(2025)

- 3. ByteDance Seed Team: Seed2.0 model card: Towards intelligence frontier for realworld complexity (February 2026), https://github.com/ByteDance-Seed/Seed2. 0, model Card
- 4. Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon,

I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al.: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261 (2025)

- 5. Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., et al.: Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394 (2023)
- 6. Fu, X., Liu, M., Yang, Z., Corring, J., Lu, Y., Yang, J., Roth, D., Florencio, D., Zhang, C.: Refocus: Visual editing as a chain of thought for structured image understanding. arXiv preprint arXiv:2501.05452 (2025)
- 7. GLM-5-Team, :, Zeng, A., Lv, X., Hou, Z., Du, Z., Zheng, Q., Chen, B., Yin, D., Ge, C., Huang, C., Xie, C., Zhu, C., Yin, C., Wang, C., Pan, G., Zeng, H., Zhang, H., Wang, H., Chen, H., Zhang, J., Jiao, J., Guo, J., Wang, J., Du, J., Wu, J., Wang, K., Li, L., Fan, L., Zhong, L., Liu, M., Zhao, M., Du, P., Dong, Q., Lu, R., Shuang-Li, Cao, S., Liu, S., Jiang, T., Chen, X., Zhang, X., Huang, X., Dong, X., Xu, Y., Wei, Y., An, Y., Niu, Y., Zhu, Y., Wen, Y., Cen, Y., Bai, Y., Qiao, Z., Wang, Z., Wang, Z., Zhu, Z., Liu, Z., Li, Z., Wang, B., Wen, B., Huang, C., Cai, C., Yu, C., Li, C., Hu, C., Zhang, C., Zhang, D., Lin, D., Yang, D., Wang, D., Ai, D., Zhu, E., Yi, F., Chen, F., Wen, G., Sun, H., Zhao, H., Hu, H., Zhang, H., Liu, H., Zhang, H., Peng, H., Tai, H., Zhang, H., Liu, H., Wang, H., Yan, H., Ge, H., Liu, H., Chu, H., Zhao, J., Wang, J., Zhao, J., Ren, J., Wang, J., Zhang, J., Gui,

- J., Zhao, J., Li, J., An, J., Li, J., Yuan, J., Du, J., Liu, J., Zhi, J., Duan, J., Zhou,
- K., Wei, K., Wang, K., Luo, K., Zhang, L., Sha, L., Xu, L., Wu, L., Ding, L., Chen,

- L., Li, M., Lin, N., Ta, P., Zou, Q., Song, R., Yang, R., Tu, S., Yang, S., Wu, S., Zhang, S., Li, S., Li, S., Fan, S., Qin, W., Tian, W., Zhang, W., Yu, W., Liang, W., Kuang, X., Cheng, X., Li, X., Yan, X., Hu, X., Ling, X., Fan, X., Xia, X., Zhang,

- X., Zhang, X., Pan, X., Zou, X., Zhang, X., Liu, Y., Wu, Y., Li, Y., Wang, Y., Zhu, Y., Tan, Y., Zhou, Y., Pan, Y., Zhang, Y., Su, Y., Geng, Y., Yan, Y., Tan,
- Y., Bi, Y., Shen, Y., Yang, Y., Li, Y., Liu, Y., Wang, Y., Li, Y., Wu, Y., Zhang, Y., Duan, Y., Zhang, Y., Liu, Z., Jiang, Z., Yan, Z., Zhang, Z., Wei, Z., Chen, Z., Feng, Z., Yao, Z., Chai, Z., Wang, Z., Zhang, Z., Xu, B., Huang, M., Wang, H., Li, J., Dong, Y., Tang, J.: Glm-5: from vibe coding to agentic engineering (2026), https://arxiv.org/abs/2602.15763

- 8. Google: Gemini 3 Flash: Frontier Intelligence Built for Speed. https://blog. google/technology/ai/google-gemini-3-flash/ (Dec 2025), published December 10, 2025. Accessed: 2026-01-29
- 9. Google: Gemini: The Most Capable and General AI Model. https://gemini. google.com/ (2026), accessed: 2026-01-29
- 10. Guo, X., Tyagi, U., Gosai, A., Vergara, P., Park, J., Montoya, E.G.H., Zhang, C.B.C., Hu, B., He, Y., Liu, B., et al.: Beyond seeing: Evaluating multimodal llms on tool-enabled image perception, transformation, and reasoning. arXiv preprint arXiv:2510.12712 (2025)
- 11. Guo, Y., Xu, Z., Yao, Z., Lu, Y., Lin, J., Hu, S., Tang, Z., Wang, H., Chen, R.: Octopus: Agentic multimodal reasoning with six-capability orchestration. arXiv preprint arXiv:2511.15351 (2025)
- 12. Gupta, T., Kembhavi, A.: Visual programming: Compositional visual reasoning without training. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 14953–14962 (2023)
- 13. Hong, J., Zhao, C., Zhu, C., Lu, W., Xu, G., Yu, X.: Deepeyesv2: Toward agentic multimodal model. arXiv preprint arXiv:2511.05271 (2025)
- 14. Hu, Y., Shi, W., Fu, X., Roth, D., Ostendorf, M., Zettlemoyer, L., Smith, N.A., Krishna, R.: Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. Advances in Neural Information Processing Systems 37, 139348–139379 (2024)
- 15. Hurst, A., Lerer, A., Goucher, A.P., Perelman, A., Ramesh, A., Clark, A., Ostrow, A., Welihinda, A., Hayes, A., Radford, A., et al.: Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024)
- 16. Itseez: The OpenCV Reference Manual, 2.4.9.0 edn. (April 2014)
- 17. Lai, X., Li, J., Li, W., Liu, T., Li, T., Zhao, H.: Mini-o3: Scaling up reasoning patterns and interaction turns for visual search. arXiv preprint arXiv:2509.07969

(2025)

- 18. Li, M., Zhong, J., Zhao, S., Zhang, H., Lin, S., Lai, Y., Wei, C., Psounis, K., Zhang, K.: Tir-bench: A comprehensive benchmark for agentic thinking-with-images reasoning. arXiv preprint arXiv:2511.01833 (2025)
- 19. Li, Y., Qian, S., Liang, H., Zheng, L., An, R., Guo, Y., Zhang, W.: Capgeo: A caption-assisted approach to geometric reasoning. arXiv preprint arXiv:2510.09302

(2025)

- 20. Liang, Y., Li, M., Fan, C., Li, Z., Nguyen, D., Cobbina, K., Bhardwaj, S., Chen, J., Liu, F., Zhou, T.: Colorbench: Can vlms see and understand the colorful world? a comprehensive benchmark for color perception, reasoning, and robustness. arXiv preprint arXiv:2504.10514 (2025)
- 21. Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al.: Mmbench: Is your multi-modal model an all-around player? In: European conference on computer vision. pp. 216–233. Springer (2024)

- 22. Liu, Y., Li, Z., Huang, M., Yang, B., Yu, W., Li, C., Yin, X.C., Liu, C.L., Jin, L., Bai, X.: Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences 67(12), 220102 (2024)
- 23. Lu, P., Bansal, H., Xia, T., Liu, J., Li, C., Hajishirzi, H., Cheng, H., Chang, K.W., Galley, M., Gao, J.: Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255 (2023)
- 24. Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.W., Zhu, S.C., Tafjord, O., Clark, P., Kalyan, A.: Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in neural information processing systems 35, 2507– 2521 (2022)
- 25. Masry, A., Do, X.L., Tan, J.Q., Joty, S., Hoque, E.: Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In: Findings of the association for computational linguistics: ACL 2022. pp. 2263–2279 (2022)
- 26. OpenAI: Introducing GPT-5.2. https://openai.com/index/introducing-gpt5-2/ (Dec 2025), published December 11, 2025. Accessed: 2026-01-29
- 27. OpenAI: Introducing OpenAI o3 and o4-mini. https://openai.com/index/ introducing-o3-and-o4-mini/ (Apr 2025), published April 16, 2025. Accessed: 2026-01-29
- 28. Qiao, R., Tan, Q., Yang, M., Dong, G., Yang, P., Lang, S., Wan, E., Wang, X., Xu, Y., Yang, L., et al.: V-thinker: Interactive thinking with images. arXiv preprint arXiv:2511.04460 (2025)
- 29. Shen, H., Zhao, K., Zhao, T., Xu, R., Zhang, Z., Zhu, M., Yin, J.: Zoomeye: Enhancing multimodal llms with human-like zooming capabilities through treebased image exploration. In: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. pp. 6613–6629 (2025)
- 30. Shi, Y., Dong, Y., Ding, Y., Wang, Y., Zhu, X., Zhou, S., Liu, W., Tian, H., Wang, R., Wang, H., Liu, Z., Zeng, B., Chen, R., Wang, Q., Zhang, Z., Chen, X., Tong, C., Li, B., Fu, C., Liu, Q., Wang, H., Yang, W., Zhang, Y., Wan, P., Zhang, Y.F., Liu, Z.: Realunify: Do unified models truly benefit from unification? a comprehensive benchmark (2025), https://arxiv.org/abs/2509.24897
- 31. Shi, Y., Liu, J., Guan, Y., Wu, Z., Zhang, Y., Wang, Z., Lin, W., Hua, J., Wang, Z., Chen, X., et al.: Mavors: Multi-granularity video representation for multimodal large language model. In: Proceedings of the 33rd ACM International Conference on Multimedia. pp. 10994–11003 (2025)
- 32. Shi, Y., Wang, H., Xie, W., Zhang, H., Zhao, L., Zhang, Y.F., Li, X., Fu, C., Wen, Z., Liu, W., Zhang, Z., Chen, X., Zeng, B., Yang, S., Guan, Y., Zhang, Z., Wang, L., Li, H., Lin, Z., Zhang, Y., Wan, P., Wang, H., Yang, W.: Mme-videoocr: Evaluating ocr-based capabilities of multimodal llms in video scenarios (2025), https://arxiv.org/abs/2505.21333
- 33. Su, Z., Gao, J., Guo, H., Liu, Z., Zhang, L., Geng, X., Huang, S., Xia, P., Jiang, G., Wang, C., et al.: Agentvista: Evaluating multimodal agents in ultra-challenging realistic visual scenarios. arXiv preprint arXiv:2602.23166 (2026)
- 34. Su, Z., Li, L., Song, M., Hao, Y., Yang, Z., Zhang, J., Chen, G., Gu, J., Li, J., Qu, X., et al.: Openthinkimg: Learning to think with images via visual tool reinforcement learning. arXiv preprint arXiv:2505.08617 (2025)
- 35. Surís, D., Menon, S., Vondrick, C.: Vipergpt: Visual inference via python execution for reasoning. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 11888–11898 (2023)
- 36. Team, K., Bai, T., Bai, Y., Bao, Y., Cai, S.H., Cao, Y., Charles, Y., Che, H.S., Chen, C., Chen, G., Chen, H., Chen, J., Chen, J., Chen, J., Chen, J., Chen, K.,

Chen, L., Chen, R., Chen, X., Chen, Y., Chen, Y., Chen, Y., Chen, Y., Chen, Y., Chen, Y., Chen, Y., Chen, Y., Chen, Z., Chen, Z., Cheng, D., Chu, M., Cui, J., Deng, J., Diao, M., Ding, H., Dong, M., Dong, M., Dong, Y., Dong, Y., Du, A., Du, C., Du, D., Du, L., Du, Y., Fan, Y., Fang, S., Feng, Q., Feng, Y., Fu, G., Fu, K., Gao, H., Gao, T., Ge, Y., Geng, S., Gong, C., Gong, X., Gongque, Z., Gu, Q., Gu, X., Gu, Y., Guan, L., Guo, Y., Hao, X., He, W., He, W., He, Y., Hong, C., Hu, H., Hu, J., Hu, Y., Hu, Z., Huang, K., Huang, R., Huang, W., Huang, Z., Jiang, T., Jiang, Z., Jin, X., Jing, Y., Lai, G., Li, A., Li, C., Li, C., Li, F., Li, G., Li, G., Li, H., Li, H., Li, J., Li, J., Li, J., Li, L., Li, M., Li, W., Li, W., Li, X., Li, X., Li, Y., Li, Y., Li, Y., Li, Y., Li, Z., Li, Z., Liao, W., Lin, J., Lin, X., Lin, Z., Lin, Z., Liu, C., Liu, C., Liu, H., Liu, L., Liu, S., Liu, S., Liu, S., Liu, T., Liu, T., Liu, W., Liu, X., Liu, Y., Liu, Y., Liu, Y., Liu, Y., Liu, Y., Liu, Z., Liu, Z., Lu, E., Lu, H., Lu, Z., Luo, J., Luo, T., Luo, Y., Ma, L., Ma, Y., Mao, S., Mei, Y., Men, X., Meng, F., Meng, Z., Miao, Y., Ni, M., Ouyang, K., Pan, S., Pang, B., Qian, Y., Qin, R., Qin, Z., Qiu, J., Qu, B., Shang, Z., Shao, Y., Shen, T., Shen, Z., Shi, J., Shi, L., Shi, S., Song, F., Song, P., Song, T., Song, X., Su, H., Su, J., Su, Z., Sui, L., Sun, J., Sun, J., Sun, T., Sung, F., Tai, Y., Tang, C., Tang, H., Tang,

- X., Tang, Z., Tao, J., Teng, S., Tian, C., Tian, P., Wang, A., Wang, B., Wang, C., Wang, C., Wang, C., Wang, D., Wang, D., Wang, D., Wang, F., Wang, H., Wang, H., Wang, H., Wang, H., Wang, H., Wang, J., Wang, J., Wang, J., Wang,

- K., Wang, L., Wang, Q., Wang, S., Wang, S., Wang, S., Wang, W., Wang, X., Wang, X., Wang, Y., Wang, Y., Wang, Y., Wang, Y., Wang, Y., Wang, Y., Wang, Z., Wang, Z., Wang, Z., Wang, Z., Wang, Z., Wang, Z., Wei, C., Wei, M., Wen, C., Wen, Z., Wu, C., Wu, H., Wu, J., Wu, R., Wu, W., Wu, Y., Wu, Y., Wu, Y., Wu, Z., Xiao, C., Xie, J., Xie, X., Xie, Y., Xin, Y., Xing, B., Xu, B., Xu, J., Xu, J., Xu, J., Xu, L.H., Xu, L., Xu, S., Xu, W., Xu, X., Xu, X., Xu, Y., Xu, Y., Xu,

Y., Xu, Z., Xu, Z., Yan, J., Yan, Y., Yang, G., Yang, H., Yang, J., Yang, K., Yang, N., Yang, R., Yang, X., Yang, X., Yang, Y., Yang, Y., Yang, Y., Yang, Z., Yang, Z., Yang, Z., Yao, H., Ye, D., Ye, W., Ye, Z., Yin, B., Yu, C., Yu, L., Yu, T., Yu, T., Yuan, E., Yuan, M., Yuan, X., Yue, Y., Zeng, W., Zha, D., Zhan, H., Zhang, D., Zhang, H., Zhang, J., Zhang, P., Zhang, Q., Zhang, R., Zhang, X., Zhang, Y.,

- Zhang, Y., Zhang, Y., Zhang, Y., Zhang, Y., Zhang, Y., Zhang, Y., Zhang, Y.,
- Zhang, Y., Zhang, Y., Zhang, Z., Zhao, C., Zhao, F., Zhao, J., Zhao, S., Zhao, X., Zhao, Y., Zhao, Z., Zheng, H., Zheng, R., Zheng, S., Zheng, T., Zhong, J., Zhong,

- L., Zhong, W., Zhou, M., Zhou, R., Zhou, X., Zhou, Z., Zhu, J., Zhu, L., Zhu, X., Zhu, Y., Zhu, Z., Zhuang, J., Zhuang, W., Zou, Y., Zu, X.: Kimi k2.5: Visual agentic intelligence (2026), https://arxiv.org/abs/2602.02276

- 37. Team, Q.: Qwen-agent. https://github.com/QwenLM/Qwen-Agent (2025)
- 38. Vinker, Y., Shaham, T.R., Zheng, K., Zhao, A., E Fan, J., Torralba, A.: Sketchagent: Language-driven sequential sketch generation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 23355–23368 (2025)
- 39. Wang, H., Su, A., Ren, W., Lin, F., Chen, W.: Pixel reasoner: Incentivizing pixel-space reasoning with curiosity-driven reinforcement learning. arXiv preprint arXiv:2505.15966 (2025)
- 40. Wang, J., Zerun, M., Li, Y., Zhang, S., Chen, C., Chen, K., Le, X.: Gta: a benchmark for general tool agents. Advances in Neural Information Processing Systems 37, 75749–75790 (2024)
- 41. Wang, Q., Shi, Y., Wang, Y., Zhang, Y., Wan, P., Gai, K., Ying, X., Wang, Y.: Monet: Reasoning in latent visual space beyond images and language (2025), https://arxiv.org/abs/2511.21395

- 42. Wang, W., Ding, L., Zeng, M., Zhou, X., Shen, L., Luo, Y., Yu, W., Tao, D.: Divide, conquer and combine: A training-free framework for high-resolution image perception in multimodal large language models. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 39, pp. 7907–7915 (2025)
- 43. Wu, P., Xie, S.: V?: Guided visual search as a core mechanism in multimodal llms. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13084–13094 (2024)
- 44. Yang, Z., Li, L., Wang, J., Lin, K., Azarnasab, E., Ahmed, F., Liu, Z., Liu, C., Zeng, M., Wang, L.: Mm-react: Prompting chatgpt for multimodal reasoning and action. arXiv preprint arXiv:2303.11381 (2023)
- 45. Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G., Stevens, S., Jiang, D., Ren, W., Sun, Y., et al.: Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 9556–9567

(2024)

- 46. Zeng, A., Attarian, M., Ichter, B., Choromanski, K., Wong, A., Welker, S., Tombari, F., Purohit, A., Ryoo, M., Sindhwani, V., et al.: Socratic models: Composing zeroshot multimodal reasoning with language. arXiv preprint arXiv:2204.00598 (2022)
- 47. Zhang, X., Gao, Z., Zhang, B., Li, P., Zhang, X., Liu, Y., Yuan, T., Wu, Y., Jia, Y., Zhu, S.C., et al.: Adaptive chain-of-focus reasoning via dynamic visual search and zooming for efficient vlms. arXiv preprint arXiv:2505.15436 (2025)
- 48. Zhang, Y.F., Lu, X., Yin, S., Fu, C., Chen, W., Hu, X., Wen, B., Jiang, K., Liu, C., Zhang, T., et al.: Thyme: Think beyond images. arXiv preprint arXiv:2508.11630

(2025)

- 49. Zhang, Y., Shi, Y., Yu, W., Wen, Q., Wang, X., Yang, W., Zhang, Z., Wang, L., Jin, R.: Debiasing multimodal large language models via penalization of language priors. In: Proceedings of the 33rd ACM International Conference on Multimedia. pp. 4232–4241 (2025)
- 50. Zhao, S., Zhang, H., Lin, S., Li, M., Wu, Q., Zhang, K., Wei, C.: Pyvision: Agentic vision with dynamic tooling. arXiv preprint arXiv:2507.07998 (2025)
- 51. Zheng, Z., Yang, M., Hong, J., Zhao, C., Xu, G., Yang, L., Shen, C., Yu, X.: Deepeyes: Incentivizing" thinking with images" via reinforcement learning. arXiv preprint arXiv:2505.14362 (2025)
- 52. Zhou, Z., Chen, D., Ma, Z., Hu, Z., Fu, M., Wang, S., Wan, Y., Zhao, Z., Krishna, R.: Reinforced visual perception with tools. arXiv preprint arXiv:2509.01656 (2025)

### A Implementation Details

We conduct all evaluations in a zero-shot setting to ensure fair comparison and better generalization. Open-source models are deployed on NVIDIA H100 GPUs using the vLLM framework, while proprietary models are accessed via their official APIs. The detailed generation hyperparameters are set as follows:

- – Open-source Instruct Models: temperature = 0.7, top-p = 0.8, top-k

= 20, repetition penalty = 1.0, presence penalty = 1.5, max tokens = 16,384, and seed = 3407.

- – Open-source Thinking Models: temperature = 0.6, top-p = 0.95, top-k

= 20, repetition penalty = 1.0, presence penalty = 0, max tokens = 40,960, and seed = 1234.

- – Proprietary Models: We strictly enable the High Reasoning mode for all assessments and set the max new tokens to 65,536. In Qwen-Agent, we set the maximum number of rounds to 20.

### B Detailed Tool Set And Data Collection

- B.1 Tool Set All tools and their corresponding descriptions are detailed in Tab. 6.
- B.2 Data Collection

We describe the specific data collection process for the nine tasks in this section.

Attention Focusing: Sourced from HRBench [42] and web-crawled images. We evaluate the agent’s robustness against Geometric Perturbations by applying manual transformations (e.g., rotation, reflection). This requires models to reorient their "focus" via tool-assisted spatial normalization.

Chart: Derived from ChartQA [25] and targeted web searches. We implement information sanitization by erasing original ground-truth labels, compelling the agent to utilize Auxiliary Construction for logic and Precision Sampling for element extraction.

Color: Based on ColorBench [20] and internet data. After subjecting images to radiometric noise (blur, illumination variance), we force the model to bypass raw RGB perception in favor of chromatic space manipulations (HSV/LAB) to quantify color proportions.

Counting: Adapted from Kaggle and web sources. We specifically target visual occlusion and density bottlenecks (e.g., overlapping objects) where direct counting is intractable, necessitating a "segment-and-count" pipeline using morphological utilities.

Math: A specialized set of web-collected and manually synthesized samples focusing on STEM-oriented Geometric Reasoning. The core challenge lies in

constructive logic, where models must identify and draw auxiliary lines to solve complex polygonal problems.

Measurement: Sourced from high-fidelity web images with manual calibration. It bridges the gap between vision and Metrology, featuring scenarios like Industrial Component Calibration that require sub-pixel precision through toolassisted physical dimension estimation.

Perceptual Restoration: Leveraging Kaggle open-source data, this task evaluates the agent’s capacity to act as a "pre-processor," effectively neutralizing atmospheric haze and photon noise to recover latent semantic information from degraded scenes.

Robust OCR: An integration of OCRBench [22] and Kaggle competition data. Unlike standard OCR, these samples are embedded with compound degradations. Models must exhibit strategic planning by chaining binarization and sharpening tools before attempting text recognition.

Spatial Reasoning: Derived from Kaggle and web sources. This task focuses on the quantification of relative topologies, requiring models to transform qualitative visual cues into precise spatial coordinates to resolve complex positional queries.

### C More On Analysis Results

##### C.1 Tool-Use Analysis Results

Figure 10 visualizes the distribution difference of toolchain lengths between Gemini-3.0-Flash and the ground truth at a granular task level. The results demonstrate a critical mismatch between predicted toolchain depths and groundtruth requirements across all task categories. The model consistently generates significantly shorter toolchains than those required by expert trajectories. Most predicted sequences peak at only 1 or 2 steps while the ground truth frequently requires 4 to 6 steps. In categories such as color and measure, the model produces a single-step chain for the vast majority of samples despite the ground truth peaking at 5 steps. This "shortcutting" behavior suggests that the model often terminates its reasoning process prematurely without completing the necessary intermediate steps. Such a depth deficiency identifies a major bottleneck in the planning and orchestration capabilities of current large multi-modal models.

###### C.2 Evaluation Metrics We define the Tool Call Rate (TCR) as Eq 2:

TCR = |{i | LT,i > 0}| N

, (2)

where N denotes the total number of samples and LT,i represents the toolchain length for the i-th sample.

attention

chart

color

100

GT ToolChain

GT ToolChain

GT ToolChain

35

70

Model ToolChain

Model ToolChain

Model ToolChain

30

80

60

Count(Samples)

25

50

60

20

40

15

40

30

10

20

20

5

10

0

0

0

0 1 2 3 4 5 6 7 8

0 1 2 3 4 5 6 7

0 1 2 3 4 5 6 7 8 9

Tool Chain Length

Tool Chain Length

Tool Chain Length

counting

math

measure

100

GT ToolChain

GT ToolChain

GT ToolChain

Model ToolChain

Model ToolChain

Model ToolChain

80

80

80

Count(Samples)

60

60

60

40

40

40

20

20

20

0

0

0

0 1 2 3 4 5 6 7 8

0 1 2 3 4 5 6 7 8 9 10

0 1 2 3 4 5 6 7 8

Tool Chain Length

Tool Chain Length

Tool Chain Length

ocr

perceptual

spatial

GT ToolChain

GT ToolChain

GT ToolChain

30

Model ToolChain

Model ToolChain

Model ToolChain

25

40

25

Count(Samples)

20

30

20

15

15

20

10

10

10

5

5

0

0

0

0 1 2 3 4 5 6 7 8

0 1 2 3 4 5 6 7

0 1 2 3 4 5 6 7 8 9

Tool Chain Length

Tool Chain Length

Tool Chain Length

###### Fig. 10: Category-level distribution difference compared with ground truth for Gemini3.0-Flash.

Table 6: Detailed taxonomy and definitions of the 35 OpenCV-based tools utilized in our benchmark. The tools are categorized into four logical groups: Geometry, Enhancement, Feature Extraction, and Drawing.

Tool Name Description

Geometry Resize Adjusts the image dimensions to a specified size. Rotate Rotates the image by a specified angle. Translate Shifts the image location along the x or y axis. Flip Mirrors the image horizontally, vertically, or both. Crop Extracts a specific rectangular region of interest (ROI). Zoom In Scales up a specific region of the image for detail inspection. Pyramid Performs pyramid upsampling or downsampling.

Enhancement Convert Color Converts color spaces (e.g., RGB to Gray, HSV, or LAB). In-Range Color Filters pixels within a specific color range (segmentation mask). Blur Smooths image noise using Gaussian, Median, or Bilateral filters. Denoise Applies Non-local Means Denoising to smooth while preserving edges. Threshold Applies binary, Otsu, or adaptive thresholding techniques. Morphology Performs morphological operations (Erosion, Dilation, Open, Close). Histogram Enhances contrast using Histogram Equalization or CLAHE. Adjust Brightness Adjusts brightness using absolute scaling (ConvertScaleAbs). Inpaint Restores damaged or missing areas within an image.

Feature Extraction Canny Edge Identifies edges using the Canny edge detector. Compute Gradients Calculates gradients using Sobel or Laplacian operators. Watershed Segments touching objects based on topographical distance. GrabCut Extracts foreground objects interactively using graph cuts. Flood Fill Fills a connected component starting from a seed point. Conn. Components Computes connected components and their statistics (bounding box,

area). Keypoint Features Detects feature points using algorithms like Harris, SIFT, or ORB. Hough Lines Detects straight lines using the Hough Line Transform. Hough Circles Detects circular objects using the Hough Circle Transform. Template Match Locates the area in the image that matches a template. DFT Converts the image to the frequency domain (Discrete Fourier Trans-

form).

Drawing Draw Contours Detects and retrieves contours from the binary image. Approx Poly Approximates a contour shape to a polygon with fewer vertices. Draw Line Annotates the image with straight lines. Draw Circle Annotates the image with circles. Contour Area Calculates the area of a specific contour. Arc Length Computes the perimeter or curve length of a contour.

### D Detailed Prompts

##### System Prompt

###### System Prompt (Strong)

Your role is that of a helpful assistant specialized in solving real-world visual problems using image processing tools. Answer questions about images by combining your visual understanding with the precision of available tools. Please follow this structured thinking process and show your work. Start an iterative loop for each question:

- - **First, look closely:** Begin with a detailed description of the image in the context of the user’s real-life query. List what is immediately visible, and explicitly identify what specific visual details need to be clarified, adjusted, or measured using tools.
- - **Next, apply tools:** Select and invoke just one appropriate function to process the image.
- - **Then, review the findings:** Carefully analyze the tool’s output (e.g., the processed image, detected features, or status) and decide on your next action (e.g., did the rotation fix the view? do you need further adjustments?). Continue this loop until you have sufficient information. To finish, bring everything together in a clear, synthesized answer that fully responds to the user’s question.

###### System Prompt (Weak)

Your role is that of a helpful assistant specialized in solving real-world visual problems using image processing tools.

##### User Prompt

###### User Prompt without GT Toolchains

<image> {question} ### User Image Path: {image_path} ### User Image Size: {image_size} ### Output Format (strict adherence required): <think>Your detailed reasoning process, should go here.</think> <answer>Your final answer to the user’s question goes here.</answer>

###### User Prompt with GT Toolchains

<image> {question} ### User Image Path: {image_path} ### User Image Size: {image_size} ### Here is the **Reference Trajectory**. You can refer to this trajectory to use tools to solve problems : {reference_trajectory} ### Output Format (strict adherence required): <think>Your detailed reasoning process, should go here.</think> <answer>Your final answer to the user’s question goes here.</answer>

##### Code Interpreter Description

Executes Python code to handle images based strictly on the allowed capabilities listed below.

CRITICAL INSTRUCTIONS FOR MODEL:

- 1. RESTRICTED SCOPE: You are strictly permitted to implement only the specific image handling operations defined in the Allowed Capabilities Reference section below. Do not generate code for operations outside this list.
- 2. NO PRE-DEFINED FUNCTIONS: The list below defines behaviors, NOT callable functions.

- – INCORRECT: Calling colorspace_gray(image) or resize(image, param). These functions DO NOT exist.
- – CORRECT: You must write the raw Python code using the cv2 library to implement the logic described. For example, to achieve colorspace_gray, you must write cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).

- 3. PARAMETER ADHERENCE: When writing the code, you must strictly follow the parameter logic described in the reference list (e.g., input ranges, default values, and calculation logic).

Allowed Capabilities Reference (Implement these using raw cv2 code)

- 1. COLOR SPACE CONVERSION TOOLS

colorspace_gray Description: Converts the image to grayscale color space. Useful for enhancing contrast or isolating intensity information. Parameters:

- – image (string, required): The input image identifier
- – param (object, optional): Empty parameter object

colorspace_hsv Description: Converts the image to HSV (Hue, Saturation, Value) color space.

Useful for color-based segmentation or isolating specific color channels. Parameters:

- – image (string, required): The input image identifier
- – param (object, optional): Empty parameter object

colorspace_lab Description: Converts the image to LAB color space. Useful for perceptual color operations or isolating specific color channels. Parameters:

- – image (string, required): The input image identifier
- – param (object, optional): Empty parameter object

- 2. GEOMETRIC TRANSFORMATION TOOLS

resize Description: Resizes the image to specified dimensions or by a preset scale (half or double). Parameters:

- – image (string, required): The input image identifier
- – param (object, required): width (integer), height (integer), preset (string, optional)

rotate Description: Rotates the image by specified angle in degrees (clockwise). Parameters:

- – image (string, required): The input image identifier
- – param (object, required): angle (number, required)

translate Description: Shifts the image by a specified number of pixels in the specified direction. Parameters:

- – image (string, required): The input image identifier
- – param (object, required): direction (string, optional), distance (integer, optional)

flip Description: Flips the image horizontally or vertically. Parameters:

- – image (string, required): The input image identifier
- – param (object, required): direction (string, optional)

crop Description: Crops a rectangular region from the image. Parameters:

- – image (string, required): The input image identifier
- – param (object, optional): x (integer), y (integer), width (integer), height (integer)

zoom_in Description: Zooms into a specific region of the image by cropping and optionally resizing. Parameters:

- – image (string, required): The input image identifier
- – param (object, optional): x, y, width, height, scale, target_width, target_height

###### 3. FILTERING AND SMOOTHING TOOLS

blur Description: Applies blurring to reduce noise or smooth the image. Supports average, gaussian, median, and bilateral. Parameters:

- – image (string, required): The input image identifier
- – param (object, required): method (string), ksize (integer), sigma_x,

sigma_y, d, sigma_color, sigma_space

denoise Description: Applies fast non-local means denoising. Parameters:

- – image (string, required): The input image identifier
- – param (object, optional): mode, h, h_color, template_window,

search_window, channels

###### 4. THRESHOLDING AND BINARIZATION TOOLS

threshold Description: Applies thresholding to create a binary image. Parameters:

- – image (string, required): The input image identifier
- – param (object, required): mode, invert, color_mode, threshold_value, adaptive_block_size, adaptive_constant, channels

inrange_color Description: Creates a mask for pixels within a specified color range. Parameters:

- – image (string, required): The input image identifier
- – param (object, optional): colorspace, lower, upper, output_format

###### 5. MORPHOLOGICAL OPERATIONS

morphology Description: Applies morphological operations (erode, dilate, open, close). Parameters:

- – image (string, required): The input image identifier
- – param (object, optional): op, kernel_size, iterations, kernel_shape

###### 6. EDGE DETECTION TOOLS

gradients Description: Computes gradient images using Sobel or Laplacian operator. Parameters:

- – image (string, required): The input image identifier
- – param (object, required): mode (sobel_x, sobel_y, laplacian)

canny Description: Detects edges using Canny edge detector. Parameters:

- – image (string, required): The input image identifier
- – param (object, optional): preset, threshold_low, threshold_high,

color_mode, channels

convertscaleabs Description: Converts image to absolute value and scales it to 0-255 range. Parameters:

- – image (string, required): The input image identifier
- – param (object, optional): alpha, beta

###### 7. CONTOUR DETECTION AND ANALYSIS TOOLS

contours Description: Finds contours using Canny edges and returns bounding boxes/areas. Parameters:

- – image (string, required): The input image identifier
- – param (object, optional): mode, canny_low, canny_high, rank,

max_contours

contour_area Description: Calculates contour areas using cv2.contourArea. Parameters:

- – image (string, required): The input image identifier
- – param (object, optional): mode, canny_low, canny_high, rank,

max_contours, color, thickness

arc_length Description: Computes contour perimeters using cv2.arcLength. Parameters:

- – image (string, required): The input image identifier
- – param (object, optional): mode, canny_low, canny_high, rank,

max_contours, color, thickness

approx_poly Description: Approximates contours with fewer points using cv2.approxPolyDP. Parameters:

- – image (string, required): The input image identifier
- – param (object, optional): epsilon_ratio, mode, canny_low, canny_high, rank, max_contours, color, thickness

connected_components_with_stats Description: Finds and analyzes connected components in a binary image. Parameters:

- – image (string, required): The input image identifier
- – param (object, optional): threshold_mode, threshold_value, connectivity,

max_components

###### 8. SHAPE DETECTION TOOLS

hough_lines Description: Detects line segments in the image using Hough transform. Parameters:

- – image (string, required): The input image identifier
- – param (object, optional): threshold, minLineLength, maxLineGap, canny_low, canny_high, max_lines

hough_circles Description: Detects circles in the image using Hough transform. Parameters:

- – image (string, required): The input image identifier
- – param (object, optional): dp, minDist, param1, param2, minRadius,

maxRadius, max_circles

###### 9. HISTOGRAM AND CONTRAST ENHANCEMENT

histogram Description: Applies histogram equalization or CLAHE to enhance image contrast. Parameters:

- – image (string, required): The input image identifier

- – param (object, optional): mode, color_mode, channels, clip_limit, tile_grid_size

###### 10. IMAGE SEGMENTATION TOOLS

watershed Description: Applies watershed segmentation to separate overlapping objects. Parameters:

- – image (string, required): The input image identifier
- – param (object, optional): max_regions

grabcut Description: Performs foreground/background segmentation using GrabCut algorithm. Parameters:

- – image (string, required): The input image identifier
- – param (object, required): preset (center, tight, loose)

floodfill Description: Performs flood fill operation starting from a seed point. Parameters:

- – image (string, required): The input image identifier
- – param (object, optional): x, y, loDiff, upDiff, newVal, flags

###### 11. ADVANCED OPERATIONS

pyramid Description: Applies image pyramid operations (upsample or downsample). Parameters:

- – image (string, required): The input image identifier
- – param (object, required): mode (pyr_up, pyr_down)

dft Description: Computes and visualizes the Discrete Fourier Transform magnitude spectrum. Parameters:

- – image (string, required): The input image identifier
- – param (object, optional): Empty parameter object

template_match Description: Matches a template image within the source image. Parameters:

- – image (string, required): The input image identifier
- – param (object, optional): template_path, method

features Description: Detects and draws keypoints using various feature detection methods. Parameters:

- – image (string, required): The input image identifier
- – param (object, optional): method (harris, sift, orb, etc.), max_points

inpaint Description: Inpaints (fills) regions in the image using automatically generated mask. Parameters:

- – image (string, required): The input image identifier
- – param (object, optional): preset, canny_low, canny_high, threshold_value,

radius, method

- 12. DRAWING TOOLS

draw_line Description: Draws a line segment between two points. Parameters:

- – image (string, required): The input image identifier
- – param (object, required): x1, y1, x2, y2, color, thickness

draw_circle Description: Draws a circle on the image at specified center coordinates. Parameters:

- – image (string, required): The input image identifier
- – param (object, optional): x, y, radius, color, thickness

### E Detailed Task Example

|Question:|Question:|Question:|Question:|
|---|---|---|---|
|Please find the slogan board on the low wall near the girl|Please find the orange book on the third shelf from the top|Please find the glider in the sky in this image, read the text|Please find the text in blue font on the sail in this image,|
|wearing light blue clothes in|on the right side of the|on it, and answer what the text|read the text on it, and answer|
|this image, read the text on it,|bookshelf opposite the cabinet|content is.|what the text says.|
|and answer what the last 5 letters are.|labeled 'Luisterpunt' in the picture. Read the text on it| | |
| |and answer what the book is| | |
| |titled.| | |
|Reference Tool Trajectory:|Reference Tool Trajectory:|Reference Tool Trajectory:|Reference Tool Trajectory:|
|Rotate, Crop, Convert Color, Binarize|Crop, Resize, Convert Color, Adjust Brightness, Sharpen|Crop, Resize, Flip|Crop, Flip, Rotate, Convert Color, Color Filter, Convert|
| | | |Color, Binarize|
|Answer: ataud|Answer: IN BRUSSEL|Answer: Windy|Answer: ebasto|

Fig. 11: Supplementary examples of Attention Focusing tasks.

|Question: Please calculate the proportion of Greece among|Question: Please find the longest task cycle.|Question: Please calculate the percentage of Mumbai relative|Question: What is the total sales volume of pineapple juice in January|
|---|---|---|---|
|European countries with coal| |to New Delhi?|and November?|
|production exceeding 5 million| | | |
|tons in 2020.| | | |
|Options:|Options:|Options:|Options:|
|(A)2.0 (B)1.5|(A)5 (B)6|(A)83 (B)82|(A)360 (B)340|
|(C)2.5 (D)3.0|(C)5.5 (D)6.5|(C)84 (D)85|(C)330 (D)350|
|Reference Tool Trajectory: Convert Color, Color Filter, Morphology, Draw Contours, Measure Area|Reference Tool Trajectory: Convert Color, Color Filter, Morphology, Draw Contours, Measure Area|Reference Tool Trajectory: Rotate, Adjust Brightness, Convert Color, Binarize, Draw Contours, Measure Area|Reference Tool Trajectory: Convert Color, Color Filter, Draw Contours, Measure Area, Draw Line|
|Answer: (A)2.0|Answer: (B)6|Answer: (C)84|Answer: (C)330|

###### Fig. 12: Supplementary examples of Chart tasks.

|Question: How many yellow chips are there in the stack of chips on|Question: How many white chips are in the picture?|Question: Excluding the black background, what is the|Question: Please use the HSV color space to count beans of|
|---|---|---|---|
|the right?| |proportion of blue in the|different colors. How many|
| | |image? A B C D|blue beans are there in the picture?|
|Options:|Options:|Options:|Options:|
|(A)8 (B)10|(A)5 (B)7|(A) 58%-60% (B) 60%-62%|(A)9 (B)6|
|(C)7 (D)9|(C)9 (D)11|(C) 54%-56% (D) 56%-58%|(C)8 (D)7|
|Reference Tool Trajectory: Rotate, Convert Color, Color Filter, Morphology, Draw Contours|Reference Tool Trajectory: Rotate, Crop, Convert Color, Color Filter, Morphology, Draw Contours|Reference Tool Trajectory: Blur, Convert Color, Color Filter, Morphology, Draw Contours, Measure Area|Reference Tool Trajectory: Convert Color, Color Filter, Morphology, Draw Contours|
|Answer: (A)8|Answer: (C)9|Answer: (C) 54%-56%|Answer: (D)7|

###### Fig. 13: Supplementary examples of Color tasks.

|Question: Count the total number of rice grains in the picture.|Question: Please count the total number of players on the field in the image. It is recommended to use color filtering or edge detection to assist in counting.|Question: Please count the number of white cotton swab heads inside the circular container in the image. The cotton swabs are densely arranged in the container, presenting a dense white dotted texture. Please carefully identify each individual cotton swab head and count them.|Question: Please count the number of birds flying in the sky in the image. The image contains a large number of scattered black bird silhouettes. Please carefully identify all targets and count them.|
|---|---|---|---|
|Options:|Options:|Options:|Options:|
|(A)110 (B)100|(A)17 (B)14|(A) 385 (B) 415|(A)153 (B)189|
|(C)95 (D)105|(C)15 (D)16|(C) 400 (D) 360|(C)177 (D)165|
|Reference Tool Trajectory: Convert Color, Blur, Binarize, Morphology, Watershed, Draw Contours|Reference Tool Trajectory: Convert Color, Blur, Edge Detect, Binarize, Morphology, Draw Contours|Reference Tool Trajectory: Convert Color, Blur, Binarize, Morphology, Watershed, Draw Contours|Reference Tool Trajectory: Convert Color, Binarize, Morphology, Draw Contours|
|Answer:(B)100|Answer: (D)16|Answer: (A) 385|Answer: (C)177|

###### Fig. 14: Supplementary examples of Counting tasks.

|Question: Solve the math problem in the diagram.|Question: Please calculate the number of edges of the shape in the image.|Question: What is the distance from point P to line l?|Question: As shown in the figure, in<br><br>△ABC, D and E are on AB and AC respectively, and DE∥BC. Find the length of EC (rounded to two decimal places).|
|---|---|---|---|
| | |Options:|Options:|
| | |(A) 135 (B) 125|(A) 3.43 (B) 4.03|
| | |(C) 140 (D) 130|(C) 3.63 (D) 3.83|
|Reference Tool Trajectory: Convert Color, Draw Contours, Edge Detect, Draw Line|Reference Tool Trajectory: Convert Color, Binarize, Edge Detect, Line Detect|Reference Tool Trajectory: Convert Color, Edge Detect, Line Detect, Draw Line|Reference Tool Trajectory: Rotate, Crop, Convert Color, Binarize, Line Detect, Draw|
| | | |Line|
|Answer: 540|Answer: 54|Answer: (D) 130|Answer: (A) 3.43|

###### Fig. 15: Supplementary examples of Math tasks.

|Question:<br><br>What is the inner diameter in pixels of the largest nut in the picture?|Question: What is the length in pixels of the black screw in the upper right corner of the table?|Question: Measure the length of the wooden block in the figure. How many cm is it?|Question: Measure the length shown by the vernier caliper in the figure. How many mm is it?|
|---|---|---|---|
|Options:|Options:|Options:|Options:|
|(A) 160 (B) 170|(A) 115 (B) 145|(A) 1.25 (B) 1.05|(A) 11.14 (B) 11.25|
|(C) 180 (D) 190|(C) 125 (D) 135|(C) 1.15 (D) 1.35|(C) 11.35 (D) 11.05|
|Reference Tool Trajectory: Convert Color, Binarize, Morphology, Draw Contours, Measure Area, Draw Circle|Reference Tool Trajectory: Crop, Convert Color, Binarize, Morphology, Draw Contours, Approximate Polygon|Reference Tool Trajectory: Rotate, Draw Contours, Draw Line, Draw Line|Reference Tool Trajectory:<br><br>Rotate, Flip, Convert Color, Binarize, Line Detect, Draw Line|
|Answer: (B) 170|Answer: (C) 125|Answer: (C) 1.15|Answer: (A) 11.14|

###### Fig. 16: Supplementary examples of Measure tasks.

|Question: When does the validity period of the certificate start and end?|Question: What is the license plate number shown in the picture?|Question: Which company's building is this?|Question: Who is the preparer of this form?|
|---|---|---|---|
|Reference Tool Trajectory: Adjust Brightness, Convert Color, Histogram Eq, Sharpen, Crop, Binarize|Reference Tool Trajectory: Crop, Resize, Convert Color, Adjust Brightness, Binarize, Morphology|Reference Tool Trajectory: Adjust Brightness, Convert Color, Histogram Eq, Sharpen, Binarize|Reference Tool Trajectory: Rotate, Adjust Brightness, Convert Color, Binarize|
|Answer:|Answer: MH02DN6980|Answer: NOKIA|Answer: 任璐|
|1994-12-18 to 2000-12-18| | | |

###### Fig. 17: Supplementary examples of OCR tasks.

|Question: This is a photo that is partially obscured by shadows. Please tell me what the third-to-last word in the 13th row is.|Question:<br><br>This is a photo that is covered with haze. Please tell me how many cars are in the image.|Question: On the globe, which continent is facing us?|Question: How many people are sitting on the chair in the picture?|
|---|---|---|---|
| |Options:|Options:|Options:|
| |(A) 3 (B) 4|(A) Eurasia (B) North America|(A) 3 (B) 2|
| |(C) 5 (D) 6|(C) Africa (D) Antarctica|(C) 4 (D) 1|
|Reference Tool Trajectory: Convert Color, Histogram Eq, Sharpen, Binarize, Crop|Reference Tool Trajectory: Histogram Eq, Convert Color, Crop|Reference Tool Trajectory: Rotate, Histogram Eq, Adjust Brightness, Crop|Reference Tool Trajectory:<br><br>Adjust Brightness, Convert Color, Crop , Edge Detect,|
| | | |Draw Contours|
|Answer: the|Answer: (D) 6|Answer: (A) Eurasia|Answer: (C) 4|

###### Fig. 18: Supplementary examples of Perceptual Restoration tasks.

|Question: Is the floor in the picture northsouth or east-west oriented?|Question: Between the red door and the yellow door on two adjacent floors, which color door is on the lower floor?|Question: Using the vertical line at the center of the image as a reference, how many blue lights are on the left side?|Question: Counting from right to left, what is the color of the clothes to the right of the third green clothing item?|
|---|---|---|---|
|Options:|Options:|Options:|Options:|
|(A) It is uncertain|(A) They are on the same floor|(A) 2 (B) 4|( A ) O r a n g e - ( B ) B l a c k y e l l o w|
|(B) East-west (C) North-south|(B) Red door (C) Yellow door|(C) 3 (D) 1|( C ) G r e e n ( D ) R e d|
|Reference Tool Trajectory: Rotate, Convert Color, Line Detect|Reference Tool Trajectory: Flip, Convert Color, Color Filter, Morphology, Draw Contours|Reference Tool Trajectory: Crop, Blur, Convert Color, Color Filter, Morphology, Draw Contours|Reference Tool Trajectory: Blur, Convert Color, Edge Detect, Morphology, Draw Contours|
|Answer: (C) North-south|Answer: (C) Yellow door|Answer: (C) 3|Answer: (B) Black|

###### Fig. 19: Supplementary examples of Spatial Reasoning tasks.

### F Detailed Model Reasoning Process

|Q: What is the direction of the airplane's flight in the picture? Options:<br><br>A: Up<br>B: Right<br>C: Left<br>D: No airplane<br><br><br>[Figure 139]| |
|---|---|
| | |

|Though: The photo shows a bright sky with heavy sun glare near the topcenter, cherry blossom trees on the left, and an open field with a small pond on the right. If there s an airplane, it would likely be a tiny, low-contrast shape in the bright sky; the glare makes it hard to see. I need to enhance contrast in the sky region to confirm whether an airplane is present and, if so, its orientation (to infer left/right/up direction).Next, apply tools: I ll enhance overall contrast (especially in bright regions) using CLAHE on a grayscale conversion to make any faint airplane silhouette stand out.|
|---|

|Action: opencv_histogram<br><br>Args:{"mode":"clahe","color_mode":"gr ayscale","clip_limit":2.0,"tile_grid_size": [8,8]}<br><br>Response: Applied CLAHE (grayscale mode, clipLimit=2.0, tileGridSize=(8, 8)).<br><br>[Figure 140]| |
|---|---|
| | |

|Action: opencv_zoom_in<br><br>Args:"{"image_1.png","param":{"x":220,"y":90,"width":260,"height":180,"scale ":3}}”<br><br>Response: Zoomed into region (220, 90, 260, 180) from 640x425 image, result size: 780x540. INFO: {"original_size": [640, 425], "zoom_region": {"x": 220, "y": 90, "width": 260, "height": 180}, "zoomed_size": [780, 540], "scale": 3.0}|
|---|

|Action: opencv_canny<br><br>Args:{"image_2.png","param":{"threshold_low":10,"threshold_high":30,"col or_mode":"grayscale"}}<br><br>Response: Applied Canny edges (10, 30) in grayscale mode. INFO: {"preset": "custom","color_mode": "grayscale", "thresholds": [10, 30], "edge_pixels": 18753,"edge_ratio": 0.044522792022792024}| |
|---|---|
| | |

[Figure 141]

[Figure 142]

|Answer: <think>I enhanced and zoomed into the bright sky area to make the faint airplane silhouette visible. In the zoomed/contrast-boosted view, the airplane s vertical tail fin is on the left side of the silhouette and the longer fuselage/nose points to the right, indicating the plane is flying toward the right.</think><br><br><answer>B. Right</answer>|
|---|

###### Fig. 20: A case study of the reasoning process for Perceptual Restoration tasks.

|Though: Let's zoom into the green area around the black screw. Approximate ROI coordinates by visual: screw near center left. We'll view zoom.<br><br>Code: import cv2, numpy as np from matplotlib import pyplot as plt img_path = measure_7.jpg' img = cv2.imread(img_path) roi = rgb[220:520, 130:330] # y,x| |
|---|---|
| | |

[Figure 143]

|Q: What is the length in pixels of the black screw in the green area? Options:<br><br>A: 130<br>B: 170<br>C: 150<br>D: 190<br><br><br>[Figure 144]| |
|---|---|
| | |

|Though: Define a tighter ROI around the black screw in the green area, and draw it on the original image for verification.<br><br>Code: import cv2, numpy as np from matplotlib import pyplot as plt img = cv2.imread(img_path) # define a tighter ROI around black screw in green area # We'll find by trial: use ROI in original image coordinates x1,y1,x2,y2 = 170,250,290,480 roi_bgr = img[y1:y2, x1:x2] roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB) # grayscale gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) # blur gb = cv2.GaussianBlur(gray,(5,5),0) # Otsu threshold inverted to get dark objects _,th = cv2.threshold(gb,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) print('otsu thresh unique:', np.unique(th, return_counts=True))<br><br>Stdout: otsu thresh unique: (array([ 0, 255], dtype=uint8), array([25466, 2134]))|
|---|

|Code: _,th = cv2.threshold(gb,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)) th2 = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1) num, labels, stats, centroids = cv2.connectedComponentsWithStats(th2, connectivity=8) # visualize bounding box of largest component idx = 2 x,y,w,h,area = stats[idx] vis = roi_rgb.copy() cv2.rectangle(vis, (x,y), (x+w, y+h), (255,0,0), 2) print('bbox w,h,area', w,h,area)<br><br>Stdout: bbox w,h,area 22 153 2115| |
|---|---|
| | |

|Though: Use skeletonization? Not allowed. But can compute length by projecting points onto major axis and taking range.<br><br>Code: largest_idx = 1 + np.argmax(stats[1:,4]) mask = (labels==largest_idx).astype(np.uint8) ys, xs = np.where(mask>0) pts = np.column_stack((xs, ys)).astype(np.float32) # PCA: subtract mean mean = pts.mean(axis=0) pts_centered = pts - mean cov = np.cov(pts_centered.T) evals, evecs = np.linalg.eigh(cov) # major axis eigenvector = evec with largest eval major = evecs[:, np.argmax(evals)] # project points onto major axis proj = pts_centered @ major length_proj = proj.max() - proj.min() length_proj<br><br>Stdout: 152.15572639834184|
|---|

[Figure 145]

[Figure 146]

|Code: # open pts_c=pts-mean cov=np.cov(pts_c.T) evals,evecs=np.linalg.eigh(cov) major=evecs[:,np.argmax(evals)] proj=pts_c@major length = proj.max()-proj.min() # oriented bounding box length contours,_=cv2.findContours((mask*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) rect = cv2.minAreaRect(contours[0]) (w_rect,h_rect) = rect[1] length_rect=max(w_rect,h_rect) print('proj length', length, 'minAreaRect length', length_rect, 'bbox h', stats[idx][3]) # visualize vis=roi_rgb.copy() x,y,w,h,area=stats[idx] cv2.rectangle(vis,(x,y),(x+w,y+h),(255,0,0),2)<br><br>Stdout: proj length 154.18852160863173 minAreaRect length 154.103515625 bbox h 155|
|---|

|Answer: <think><br><br>Loaded the image from the given path and zoomed into the green sector where the black screw is located. To measure its pixel length, I:<br><br>1) Cropped a ROI around the screw.<br>2) Converted the ROI to grayscale, applied a slight Gaussian blur, then used an inverted Otsu threshold to segment dark objects (the screw) from the brighter background.<br>3) Cleaned small noise via a morphological opening.4) Used connected-components to isolate the long, thin component (high aspect ratio) corresponding to the screw.<br><br><br>5) Measured the screw s length along its major axis (using PCA projection / minAreaRect major side). This consistently gave a length of about **154 pixels** ( 152 154 px across reasonable thresholds).<br><br>The closest option to ~154 px is **150**.<br><br></think><br><br><answer>C. 150</answer>|
|---|

###### Fig. 21: A case study of the reasoning process for Measure tasks.

